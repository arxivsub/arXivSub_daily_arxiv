# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-20 | 今日论文总数: 576

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Adapting Methods for Domain-Specific Japanese Small LMs: Scale, Architecture, and Quantization

**arXiv ID:** 2603.18037 | [PDF](https://arxiv.org/pdf/2603.18037v1)

**作者:** Takato Yasuno `[一作]` `[通讯]`, Takato Yasuno

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套三阶段的系统方法，利用 QLoRA 进行日语领域小模型的高效微调，包括训练规模选择、模型对比与量化优化。

**💡 创新点**

创新点在于通过测试集 NLL 进行规模学习、揭示不同架构（MHA vs GQA）对 Q4 量化的差异性，并给出针对低资源日语技术领域的可操作部署指南。

**🔧 技术方法**

主要技术包括 QLoRA 微调、4‑bit NF4 量化、GGUF K‑quant 量化、LoRA 低秩矩阵、LLM‑as‑Judge 评估以及基于知识图谱的 QA 生成与重采样。

**📊 数据集**

使用了由河川沉积物控制技术标准整理的 5,578 条日语 QA 对，配合 100 题测试集（以及 200 题验证集）进行实验，最终在 4,000 条样本上实现最佳效果。

**📈 对比分析**

实验通过 LLM‑as‑Judge 的 0‑3 评分对 Swallow‑8B、ELYZA‑JP‑8B、Qwen2.5‑7B、Tanuki‑8B 进行对比，Swallow‑8B 取得平均 2.82/3、84% 完美率；在 Q4_K_M 量化后 Swallow‑8B、ELYZA‑JP‑8B 分别提升 0.01–0.03 分，Qwen2.5‑7B 降低 0.28 分；量化后推理速度提升 2.5–6.1 倍，模型体积缩小至约 5 GB。

**⚠️ 局限性**

局限性包括仅在单张 RTX 4060 Ti 16 GB GPU 上实验，评测集规模有限，评判模型可能存在偏见，未与 GPT‑4 等商业 API 对比，且对 GQA 架构的 Q4 量化不推荐，未探索更大规模模型或多 GPU 分布式训练。

---

## 2. Spectrally-Guided Diffusion Noise Schedules

**arXiv ID:** 2603.19222 | [PDF](https://arxiv.org/pdf/2603.19222v1)

**作者:** Carlos Esteves `[一作]` (Google Research), Ameesh Makadia `[通讯]` (Google Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练了基于图像功率谱的自适应噪声调度，用以提升像素级扩散模型的生成质量和采样效率。

**💡 创新点**

创新点在于将每张图像的光谱特性映射为“紧凑”噪声调度，推导最小/最大噪声界限并提出混合频率/功率聚焦的调度方案，同时通过条件采样预测谱参数。

**🔧 技术方法**

采用光谱分析（RAPSD）、Gaussian 混合模型估计谱、FiLM 条件层、频率聚焦与功率聚焦相结合的混合调度，以及改进的采样与引导区间技术。

**📊 数据集**

主要在 ImageNet 数据集上进行类条件图像生成实验，测试 128×128、256×256、512×512 三种分辨率。

**📈 对比分析**

与 SiD2 及其他单阶段像素扩散基线对比，实验显示在低步数（NFE）下 FID 明显降低、IS 提升，且在大部分度量上优于基线，虽然在极高步数下略逊一筹。

**⚠️ 局限性**

仍落后于当前最先进的潜在扩散和蒸馏模型；需进一步调优损失偏置和引导区间参数；是否适用于多阶段模型仍待验证。

---

## 3. Computation-Utility-Privacy Tradeoffs in Bayesian Estimation

**arXiv ID:** 2603.18254 | [PDF](https://arxiv.org/pdf/2603.18254v1)

**作者:** Sitan Chen `[一作]` (Harvard), Walter McKelvie `[通讯]` (Harvard)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了差分隐私下的贝叶斯估计方法，特别针对高维高斯均值估计和线性回归问题，提供了第一个多项式时间的算法，能够在保持隐私的同时接近贝叶斯最优误差。

**💡 创新点**

创新点在于首次证明了在差分隐私约束下，贝叶斯均值估计和线性回归存在计算-统计差距，并且提出了基于和平方和的鲁棒估计器来解决本质上不鲁棒的对象。

**🔧 技术方法**

使用了和平方和（Sum-of-Squares, SoS）技术，结合隐私到鲁棒性的框架，设计了鲁棒的估计器以实现差分隐私。

**📊 数据集**

使用了高维高斯分布生成的数据集，特别是均值和线性回归的样本数据，确保了算法在不同维度下的有效性。

**📈 对比分析**

与现有方法相比，本文的方法在高维情况下提供了更好的隐私-效用权衡，且在多项式时间内实现了接近贝叶斯最优的误差。性能上，算法在样本复杂度上达到了最优的对数因子。

**⚠️ 局限性**

限制在于算法的复杂性和对先验分布的依赖，尤其是在高维情况下，可能会面临计算效率和隐私保护之间的权衡。

---

## 4. Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation

**arXiv ID:** 2603.19166 | [PDF](https://arxiv.org/pdf/2603.19166v1)

**作者:** Swagat Padhan `[一作]` (Arizona State University), Nakul Gopalan `[通讯]` (Arizona State University)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5089421543)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MAPG（Multi-Agent Probabilistic Grounding）框架，将自然语言目标分解为锚点、语义、度量三部分，并通过多代理协作在 3D 场景图中生成可供规划器使用的连续目标分布；同时构建了 MAPG-Bench 基准，专门评估度量-语义定位能力。

**💡 创新点**

创新点包括：① 对指令进行结构化拆解并采用多代理概率推理；② 将语义、度量与空间关系映射为可组合的概率核，得到全局可采样的目标密度；③ 通过场景图与多视角观测的证据积累实现锚点的自适应确定。

**🔧 技术方法**

使用的技术包括：自然语言解析（依存句法+模板）、CLIP 与文本相似度检索、基于 VLM 的参数化概率核（如 von Mises–Fisher 与径向高斯）、产品专家（PoE）组合、基于 RRT* 的路径规划以及在 Habitat-Sim 上的 3D 场景图构建。

**📊 数据集**

使用的数据集包括 HM3D 30 个室内场景与 100 条手工注释的度量-语义查询（MAPG-Bench），以及现有 HM-EQA 用于问答对比；实验也在真实机器人上做了演示。

**📈 对比分析**

与 GraphEQA、SRGPT 以及通用 VLMs 进行对比，MAPG 在 MAPG-Bench 上显著降低对象到世界的距离误差（从 5.82 m 降至 0.07 m）与角度误差，任务成功率提升至 0.98，轨迹长度仅 1.3 m，表明其在规划可执行目标上的优越性。

**⚠️ 局限性**

局限主要在场景图不完整导致锚点缺失、帧参考歧义（如“前方”）、以及地图与场景图对齐误差，导致在高度遮挡或复杂空间关系时仍可能出现定位误差。

---

## 5. Token Economy for Fair and Efficient Dynamic Resource Allocation in Congestion Games

**arXiv ID:** 2603.18094 | [PDF](https://arxiv.org/pdf/2603.18094v1)

**作者:** Leonardo Pedroso `[一作]` (Eindhoven University of Technology), Mauro Salazar `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1265 | [OpenAlex ID](https://openalex.org/A5023825594)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个基于令牌经济的动态拥堵游戏模型，并设计了整数令牌费率，实现了用户的公平和系统效率的平衡。

**💡 创新点**

提出了将有限理性、进化决策和均值场近似相结合的理论框架；证明了均衡资源流唯一且可通过闭式公式给出令牌费率，从而在任意精度下实现 intra‑class 公平与 inter‑class 公平与效率的最优权衡。

**🔧 技术方法**

使用了动态拥堵游戏建模、均值场近似、进化博弈理论、线性规划与对偶理论、数值仿真等技术。

**📊 数据集**

实验使用了 Sioux Falls 交通网络，约 2×10⁵ 个代理、76 条道路、117 个 OD 类，并采用 BPR 四次多项式作为资源奖励函数。

**📈 对比分析**

与无激励方案（零费率）进行比较；仿真表明通过设计的令牌费率可提升约 13% 的性能指标（平均奖励+γ 效率），并且误差可通过增大 N、α、k̅ 进一步逼近零。

**⚠️ 局限性**

主要限制在于对模型假设（有限理性、噪声、时间尺度分离）较强；对奖励函数的连续可微性和噪声量有要求；在极度异质或极端需求变化的场景中可能需要进一步验证和调整。

---

## 6. GoalVLM: VLM-driven Object Goal Navigation for Multi-Agent System

**arXiv ID:** 2603.18210 | [PDF](https://arxiv.org/pdf/2603.18210v1)

**作者:** MoniJesu James `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserokou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GoalVLM，一个多智能体零射击开放词汇对象导航框架。

**💡 创新点**

创新点包括VLM驱动的前沿推理、SAM3零射击检测与深度投影定位、结构化提示链与去中心化协调。

**🔧 技术方法**

使用VLM（SAM3、SpaceOM）、BEV语义映射、体素投影、Fast Marching、贝叶斯价值图、多智能体协作。

**📊 数据集**

基于GOAT‑Bench（val_unseen）和HM3D场景进行实验，并在实际多旋翼平台进行预验证。

**📈 对比分析**

与SenseAct、Modular GOAT、VLMNav等基线对比，在val_unseen上实现55.8%子任务成功率、18.3% SPL，显著优于无训练显式映射方法。

**⚠️ 局限性**

局限在高SPL差距、仅2D BEV不支持多楼层、对透明/反射/小目标检测不足以及缺乏学习式局部策略。

---

## 7. Do VLMs Need Vision Transformers? Evaluating State Space Models as Vision Encoders

**arXiv ID:** 2603.19209 | [PDF](https://arxiv.org/pdf/2603.19209v1)

**作者:** Shang-Jui Ray Kuo `[一作]` (Stony Brook University), Paola Cascante-Bonilla `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在冻结视觉编码器的 LLaVA 风格 VLM 训练框架下，系统评估并对比了 ViT、MaxViT、MambaVision、VMamba 等多种视觉编码器，探讨了预训练目标与视觉–语言接口对 VQA 与定位性能的影响。

**💡 创新点**

创新点在于首次在完全冻结编码器的设置下公平比较 SSM 视觉编码器（VMamba）与传统 ViT，揭示预训练目标与接口几何对定位崩溃的关键作用，并提出了简单的连接器容量与输入几何的稳定化策略。

**🔧 技术方法**

使用的技术包括 VMamba（SSM）、ViT、MaxViT 等视觉编码器、3 层 MLP 连接器、Vicuna-7B LLM、图像字母盒缩放、FSDP 并行训练，以及 dense 预训练（检测/分割）和 ImageNet 预训练。

**📊 数据集**

所用数据集为 665K 多模态指令微调数据，评估集包括 VQA‑v2、GQA、VizWiz、TextVQA、POPE、TallyQA 以及 RefCOCO、RefCOCO+、RefCOCOg、OCID‑Ref 等定位基准。

**📈 对比分析**

通过严格匹配 IN1K/224 的视觉编码器交换以及 dense 预训练比较，发现 VMamba 在定位上往往优于 ViT，且通过增加连接器层数或改为正方形高分辨率输入可恢复定位崩溃；整体 VQA 与定位平均得分相较基线提升约 2–5%。

**⚠️ 局限性**

局限在于仅评估了冻结编码器场景，未探索大规模数据或跨任务迁移的泛化效果，且对 SSM 与 ViT 的超参数匹配不完全。

---

## 8. MST-Direct: Matching via Sinkhorn Transport for Multivariate Geostatistical Simulation with Complex Non-Linear Dependencies

**arXiv ID:** 2603.18036 | [PDF](https://arxiv.org/pdf/2603.18036v1)

**作者:** Tchalies Bachmann Schmitz `[一作]` `[通讯]`, Tchalies Bachmann Schmitz

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 MST-Direct 方法，利用最优传输与 Sinkhorn 算法直接匹配多维分布并保持空间相关性，完成多变量地质统计模拟；

**💡 创新点**

核心创新在于一次性处理所有变量的最优传输与关系匹配，既保留复杂非线性依赖，又无需先转为高斯空间；

**🔧 技术方法**

采用最优传输理论、Sinkhorn 迭代、k‑最近邻关系匹配、L2 单位化以及贪婪取整等技术，并用 FFT‑MA 生成合成数据；

**📊 数据集**

使用 25×25（625 点）网格合成场，包含 X（球面变差）与 Y（指数变差），并构造了五种复杂二变量关系；

**📈 对比分析**

与 Gaussian Copula 与 LU 分解三种传统方法对比，采用 2D 直方图相似度与变差函数相关系数评估；MST‑Direct 在形状保持上 100% (1.000)，在 Y 变量变差保持上表现最佳，X 变量上 LU 更佳，总体性能优于传统方法；

**⚠️ 局限性**

局限性包括 O(n²) 计算复杂度导致难以扩展到大网格或多变量场景；不支持硬数据约束与各向异性；参数调优和 GPU 加速仍待实现。

---

## 9. SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding

**arXiv ID:** 2603.18567 | [PDF](https://arxiv.org/pdf/2603.18567v1)

**作者:** Shenggui Li `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2882 | [OpenAlex ID](https://openalex.org/A5028270700)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了开源、面向生产的 SpecForge 框架，用于高效训练 EAGLE-3 规格化解码的草稿模型，并发布了一套覆盖主流开源 LLM 的高质量草稿模型集合 SpecBundle。

**💡 创新点**

创新点：① 目标‑草稿解耦的混合并行策略，分别为大型目标模型使用 SGLang 推理引擎、为轻量草稿模型使用 DeepSpeed ZeRO‑2；② 针对 EAGLE‑3 Training‑Time‑Test（TTT）的稀疏树注意力与内存高效梯度实现；③ 系统探讨 TTT 长度、数据重生成、专家数目等超参对接受率与速度的影响，给出实用训练配方。

**🔧 技术方法**

使用技术包括：SGLang（FlashAttention/FlashInfer、CUDA‑Graph）、DeepSpeed ZeRO‑2、FlexAttention（Triton 编译的稀疏注意力）、自定义 Triton 内核、混合并行、树注意力掩码构造、动态 TTT 调度等。

**📊 数据集**

训练数据集：Open‑PerfectBlend（1.4M 对话、数学、编程等领域），并在实验中对 ShareGPT、UltraChat、GPT‑4/ChatGPT 重生成的回复进行再训练；评测使用 MTBench、GPQA、FinanceQA、Math500、GSM8K、HumanEval、LiveCodeBench 等多任务基准。

**📈 对比分析**

与官方 SafeAILab 实现、NVIDIA 版本以及现有开源草稿模型进行对比：训练吞吐量最高提升 9.9×；推理端在 SGLang 上可达 4.48× 的速度提升，且相较于无草稿推理实现提升 1.3×，对比现有公开草稿模型提升 1.35×。

**⚠️ 局限性**

局限性：① TTT 长度与显存占用成正比，需在训练资源受限时权衡；② MoE 架构的草稿模型训练效果不佳，仍需改进；③ 目前主要支持指令型 LLM，尚未覆盖推理型或跨模态模型，未来工作需扩展多模型、多任务的通用支持。

---

## 10. EgoAdapt: Enhancing Robustness in Egocentric Interactive Speaker Detection Under Missing Modalities

**arXiv ID:** 2603.18082 | [PDF](https://arxiv.org/pdf/2603.18082v1)

**作者:** Xinyuan Qian `[一作]` (University of Science and Technology Beijing), Dong Liang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 9564 | [OpenAlex ID](https://openalex.org/A5058365563)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EgoAdapt 框架，解决 egocentric 视角下“Talking to Me”任务中的缺失模态问题，结合头部姿态、唇动和音频信息实现鲁棒的说话人检测。

**💡 创新点**

创新点在于三大模块的设计：VSTR 提取非言语（头姿）与言语（唇动）特征；PSA 编码器在共享权重下学习噪声鲁棒的音频表示；VMMA 模块通过动态提示评估每帧模态缺失并自适应调整融合策略。

**🔧 技术方法**

采用 6D RepNet 进行头姿估计，CLIP ViT 处理唇图，Whisper‑small 作为音频特征提取器，PSA 使用并行共享权重的双通道编码；三模态特征通过交叉注意力与自注意力融合，最后结合 VMMA 输出进行 TTM 预测。

**📊 数据集**

实验数据集为 Ego4D TTM 基准（约 32.4 小时训练、4.2 小时验证、11.1 小时测试）以及 WHAM! 噪声数据用于音频增强。

**📈 对比分析**

与 Random Guess、ResNet‑18 Bi‑LSTM、EgoT2、QuAVF 等方法对比，验证集 Acc 达 81.14%、mAP 84.83%，在测试集 Acc 62.01%、mAP 67.39%，相较 SOTA 提升约 5% 的准确率和 1.5% 的 mAP。

**⚠️ 局限性**

主要局限包括：依赖严格的视听同步；多阶段注意力融合导致计算开销较大；仅在 Ego4D 数据上验证，跨域泛化能力待进一步研究。

---

## 11. Conflict-Based Search for Multi Agent Path Finding with Asynchronous Actions

**arXiv ID:** 2603.18866 | [PDF](https://arxiv.org/pdf/2603.18866v1)

**作者:** Xuemian Wu `[一作]` (Shanghai Jiao Tong University), Zhongqiang Ren `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 395 | [OpenAlex ID](https://openalex.org/A5018561143)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种针对异步动作的 MAPF 的完整搜索算法 CBS-AA，修正了之前 CCBS 在等待动作上的不完整性，并引入约束传播技术提升可扩展性。

**💡 创新点**

创新点包括：① 针对等待动作的连续时间约束提出全局不可冲突约束，保证完整性；② 通过 Duration Occupancy (DO) 的多动作约束传播，显著减少高层节点扩展；③ 结合软约束低层规划以利用其他代理路径。

**🔧 技术方法**

采用 CBS 两层搜索框架，连续时间版 Safe Interval Path Planning (SIPP)，多动作约束生成与传播，以及松散同步 M* 的辅助方法。

**📊 数据集**

实验使用四张公开地图（empty-32-32、random-32-32-20、den312d、warehouse-10-20-10-2-2），每个代理速度随机取 1~20，边长统一为 1。

**📈 对比分析**

与 CCBS、Loosely Synchronized M*、A* 等基线算法对比，CBS-AA 在大多数实例中成功率更高，节点扩展数减少高达 90%，在 30/120 秒时间限制内运行时间明显优于基线。

**⚠️ 局限性**

局限性在于：对连续时间约束的生成与软约束处理仍然计算量大，低层规划时间随代理数增长显著；实验仅覆盖离散图与固定速度范围，未考虑不确定性、目标分配等更复杂现实场景。

---

## 12. How Confident Is the First Token? An Uncertainty-Calibrated Prompt Optimization Framework for Large Language Model Classification and Understanding

**arXiv ID:** 2603.18009 | [PDF](https://arxiv.org/pdf/2603.18009v1)

**作者:** Wei Chen `[一作]` (China Jiliang University), Yuanyuan Qi `[通讯]` (China Jiliang University)

**通讯引用:** 1405 | [OpenAlex ID](https://openalex.org/A5108855216)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于首词不确定度的指标Log-Scale Focal Uncertainty (LSFU)，并基于该指标设计了两阶段的提示优化框架UCPOF，用以在大语言模型的分类任务中实现高效且精确的提示调优。

**💡 创新点**

创新点在于：① 将类别先验概率嵌入首词熵的非线性调制，克服传统熵因预训练频率偏差导致的误判；② 用LSFU对静态示例进行“黄金射击”筛选，实现更稳定的静态提示；③ 将LSFU作为门控阈值，智能触发检索增强生成（RAG），既提升准确率又显著降低检索率和推理成本。

**🔧 技术方法**

主要技术包括：基于首词概率的熵计算、标签先验调制因子、对数尺度归一化、黄金示例筛选算法、门控式动态检索（RAG）以及大模型的Prompt工程和向量检索。

**📊 数据集**

使用六个公开数据集进行评估：ACE、CASIE、AgNews、SST5、CB（自然语言推断）和Reuters-21578（R8）长尾文本分类。

**📈 对比分析**

与随机示例、最易/最难示例选择以及始终检索的Full RAG进行对比；实验结果显示，UCPOF平均提升约6.0%准确率，且检索触发率下降约50%，在保持或提高性能的同时显著降低了计算开销。

**⚠️ 局限性**

局限性包括：仅针对分类任务，无法直接推广到开放式问答或多轮交互；提示示例顺序与模板未得到系统化优化；门控阈值的设定依赖于训练集的分布稳定性，面对显著分布漂移时鲁棒性有待提升。

---

## 13. MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models

**arXiv ID:** 2603.19044 | [PDF](https://arxiv.org/pdf/2603.19044v1)

**作者:** Chenyang Gu `[一作]` (East China Normal University), Guoxiu He `[通讯]` (East China Normal University)

**通讯引用:** 179 | [OpenAlex ID](https://openalex.org/A5000341481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MoRI框架，利用强化学习让LLM在科研动机到方法的推理过程中实现内化思维，生成更具技术深度和科学可行性的科研创意

**💡 创新点**

将科研创意视为动机驱动的推理任务，结合熵感知信息增益与对比语义增益的复合奖励，实现技术细节与概念方向双重校准，突破现有表面式模式重组的局限

**🔧 技术方法**

基于DeepSeek-R1-Distilled-Qwen-14B的LLM，通过监督微调(SFT)学习动机生成与初步推理，再使用Group Relative Policy Optimization (GRPO)进行强化学习，奖励由Entropy‑Aware Information Gain与Contrastive Semantic Gain构成，并加入长度锚定与格式约束

**📊 数据集**

从ICLR 2024–2025论文的Method章节构建的动机–推理–方法数据集（约8000条训练样本，包含研究背景、动机、推理轨迹和方法），并采用时间拆分确保测试不与训练泄露

**📈 对比分析**

与商业模型（GPT‑4o、Claude‑3.5‑Sonnet）及多种agentic框架（AI‑Scientist‑V2、ResearchAgent、VirSci）进行对比，采用检索增强LLM评判与人工专家评测相结合的混合评价，结果显示MoRI在新颖性、技术严谨性和可行性上均超过对照组，平均得分提升约10%~20%

**⚠️ 局限性**

实验仅涵盖计算机科学领域，未验证在生物学、物理等逻辑结构不同学科的适用性；评价指标主观性高，缺乏真实实验或同行评审验证；系统仅为协助工具，需人工甄别伦理与实际可行性

---

## 14. Why Synchronized Time is a Fiction: Daylight Saving Time, Leap Seconds, and the Guillotine Sharpened for Nothing

**arXiv ID:** 2603.19099 | [PDF](https://arxiv.org/pdf/2603.19099v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAE DAE LUS), Paul Borrill (DAE DAE LUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文论证了全球同步时钟假设是范畴错误，并从物理学与分布式系统的视角阐明其导致的各种时间调整和协议误区。

**💡 创新点**

创新点在于将相对论对绝对同步的否定与量子力学不确定因果顺序相结合，提出把时序从绝对时间转向因果语义的系统设计范式。

**🔧 技术方法**

采用理论分析、经典同步协议（如NTP、IEEE 1588）与量子实验（无限制因果顺序、Bell实验）等技术进行论证。

**📊 数据集**

使用的“数据集”主要是已有文献与实验结果，例如斯坦福NSDI 2018时钟同步实验数据、GPS时钟校正记录与量子开关实验数据。

**📈 对比分析**

通过对比传统同步协议与实验观测的偏差，作者指出即使高精度同步能在常规误差范围内运行，但其本质上仍受范畴假设限制，性能提升并不等同于物理意义的同步。

**⚠️ 局限性**

局限性在于缺乏可操作的因果语义替代方案的完整实现细节，并且对现有工业系统迁移成本与实际效益评估不足。

---

## 15. BoundAD: Boundary-Aware Negative Generation for Time Series Anomaly Detection

**arXiv ID:** 2603.18111 | [PDF](https://arxiv.org/pdf/2603.18111v1)

**作者:** Xiancheng Wang `[一作]` (Harbin Institute of Technology), Minghang Zhao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 3848 | [OpenAlex ID](https://openalex.org/A5002417986)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个三阶段无监督时间序列异常检测框架：先用重构网络学习正常模式，再用强化学习动态生成接近数据流形边界的伪异常样本，随后通过三元组对比学习构建判别嵌入，最后用可训练原型进一步压缩正常空间并分离异常。

**💡 创新点**

创新点在于：①使用强化学习控制重构网络的参数更新幅度，自动在正常样本附近生成“硬负样本”而不是依赖人工注入；②将生成的伪异常与对比学习结合，显著提升嵌入空间的区分度；③在最终阶段引入可训练原型而非传统聚类，使得异常得分更加稳定、可解释。

**🔧 技术方法**

核心技术包括：重构自编码器（Transformer+Linear+ExtremeKAN），基于状态-动作的强化学习策略（Actor网络与手工定义奖励），三元组对比损失与紧凑度损失，原型学习（软赋值、距离约束、散布与平衡正则）。

**📊 数据集**

实验主要在两类典型异常场景下进行：点异常（Point）与集体季节性异常（Collective Seasonal），数据来源为公开工业时间序列数据集（具体数据集未明示，但可推测为如SMAP、SMD或UCR等常用 TSAD 数据集）。

**📈 对比分析**

与传统重构/预测/对比学习基线（如THOC、Anomaly Transformer、TranAD、DCdetector）进行对比，实验结果显示在季节性异常上 AUC 0.9606，点异常上 AUC 0.9438，说明该方法在两类异常下均能保持较高的检测性能，且在模式异常上误报率更低、异常得分分离更明显。

**⚠️ 局限性**

主要局限包括：①实验数据集规模和多样性有限，未在大规模多变量数据集上充分验证；②强化学习过程对状态设计、奖励函数、超参数高度敏感，鲁棒性待进一步评估；③原型数量与初始化方式影响结果，需更系统的调参；④仅采用窗口级检测，无法实现高时间分辨率或实时检测；⑤缺乏对真实异常分布的理论分析与迁移性验证。

---

## 16. Hardness of High-Dimensional Linear Classification

**arXiv ID:** 2603.19061 | [PDF](https://arxiv.org/pdf/2603.19061v1)

**作者:** Alexander Munteanu `[一作]` (TU Dortmund), Jeff M. Phillips `[通讯]` (University of Utah)

**通讯引用:** 2863 | [OpenAlex ID](https://openalex.org/A5017619650)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在几何与机器学习交叉问题中，给出最大半空间差异（Maximum Halfspace Discrepancy）问题的指数维度下硬度下界，证明该问题在真实RAM模型和侧边查询模型中均需时间至少为Ω̃(n^d)或Ω̃(1/ε^d)；

**💡 创新点**

创新点在于将传统的3‑Sum和Affine Degeneracy测试问题通过构造细化的点集映射到最大差异问题，得到匹配的指数下界，并且在侧边查询模型下提供无条件硬度；

**🔧 技术方法**

采用构造性归约、线性代数工具（矩阵行列式、逆矩阵、Sherman‑Morrison公式）以及细致的几何编码，将整数约束转化为半空间判定；

**📊 数据集**

论文未使用公开数据集，而是构造了基于整数点的合成实例；

**📈 对比分析**

与已知的O(n^d)及O(1/ε^d log^4(1/ε))上界相匹配，证明在多项式时间内无法获得更好的近似，除非违背k‑Sum或Affine Degeneracy的硬度假设；

**⚠️ 局限性**

局限性：下界仅在假设k‑Sum或Affine Degeneracy不可快算法的前提下成立；对于实数RAM模型中更一般的情况，仅在固定维度或侧边查询模型中无条件成立；

---

## 17. How Uncertainty Estimation Scales with Sampling in Reasoning Models

**arXiv ID:** 2603.19118 | [PDF](https://arxiv.org/pdf/2603.19118v1)

**作者:** Maksym Del `[一作]` (Institute of Computer Science University of Tartu), Mark Fishel `[通讯]` (Institute of Computer Science University of Tartu)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在延伸链式推理语言模型（RLM）中，利用并行采样的可解释置信度（VC）与自一致性（SC）来进行不确定性估计，并探讨了两种信号的组合（SCVC）。

**💡 创新点**

创新点在于首次系统评估了RLM在不同采样预算下VC、SC和混合信号的可扩展性、互补性以及域依赖性，并发现仅用两份样本的混合估计即可超过单一信号在更大采样预算下的表现。

**🔧 技术方法**

使用的方法包括：黑盒并行采样、基于VC平均和SC计数的置信度计算、λ=0.5 的线性组合、Bootstrap 采样估计 AUROC，以及对不同 VC 提示方式的实验。

**📊 数据集**

数据集涵盖 17 个任务，分布于数学、STEM 与人文学科；任务来源包括 MMLU‑Pro、GSM8K、AIME、GPQA Diamond 等，三种开源 RLM（gpt‑oss‑20b‑high、Qwen3‑30B、DeepSeek‑R1‑0528）。

**📈 对比分析**

在 AUROC 指标下，VC 在单样本时已表现强劲；随着采样量升至 8，VC 在数学上提升 10.1 点；SC 的提升较慢且始终低于 VC；混合 SCVC 在两样本时即可提升 12.9（数学）/6.4（STEM/人文）点，且在更大采样量下仍保持领先，证明信号互补性显著。

**⚠️ 局限性**

局限性包括：每额外样本需完整链式推理，计算成本高；实验仅在开源模型与 RLVR 训练的数学任务上表现最佳；对高级 VC 变体效果不显著，说明在 RLM 中未必能进一步提升；以及混合策略的鲁棒性虽不依赖 λ，但仍需在更广泛任务和模型上验证。

---

## 18. Seeking Universal Shot Language Understanding Solutions

**arXiv ID:** 2603.18448 | [PDF](https://arxiv.org/pdf/2603.18448v1)

**作者:** Haoxin Liu `[一作]` (Georgia Institute of Technology), B. Aditya Prakash `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5232 | [OpenAlex ID](https://openalex.org/A5061110232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SLU-SUITE——包含490K QA对、33个任务、6个电影摄影维度的综合人类标注数据集，并基于该数据集研发了两种通用的Shot Language Understanding模型：UniShot（平衡全能型）和AgentShots（针对性专家集群）

**💡 创新点**

创新点在于①通过对VLM的模块化实验发现语义对齐是SLU的主瓶颈；②量化并利用跨维度的转移信息，提出动态平衡数据混合和目标感知数据混合两种策略；③构建的两种模型在单一模型下即可在ID和OOD任务上达到或超过多模型集成的性能

**🔧 技术方法**

使用Qwen3-VL-8B作为基础VLM，采用LM+Connector LoRA微调；UniShot通过动态平衡采样重权重；AgentShots采用prompt路由与多专家LoRA集群；训练时使用分类式QA、next-token预测；同时进行数据去重、OOD样本剔除等数据预处理

**📊 数据集**

SLU-SUITE（490K QA），ShotBench、CameraBench、CineTechBench、MovieCuts、ShotVL-7B等公开SLU数据集用于对比；在SLU-SUITE上进行ID/OOD分布式评估

**📈 对比分析**

与五大通用VLM（Qwen3-VL-8B、Gemini-2.5-Flash、Gemini-3.0-Flash、Gemini-3.0-Pro、ShotVL-7B）以及12个任务专属SFT模型进行对比；在ID任务中UniShot平均准确率0.759，超越12任务专属模型平均0.740；AgentShots在OOD任务上平均准确率0.666，超过Gemini-3.0-Flash的0.581；两模型均保持较高的任务覆盖率与泛化能力

**⚠️ 局限性**

主要限制包括：①模型依赖大量高质量人类标注，数据获取成本高；②在跨维度的极端稀缺任务中仍可能受限；③模型在新出现的完全未知维度或极端风格的影片中泛化效果尚待进一步验证

---

## 19. Evaluating Game Difficulty in Tetris Block Puzzle

**arXiv ID:** 2603.18994 | [PDF](https://arxiv.org/pdf/2603.18994v1)

**作者:** Chun-Jui Wang `[一作]` (National Yang Ming Chiao Tung University), I-Chen Wu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2756 | [OpenAlex ID](https://openalex.org/A5016730899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

使用 Stochastic Gumbel AlphaZero 对 Tetris Block Puzzle 的不同规则变体进行训练与评估，以量化游戏难度。

**💡 创新点**

创新点在于将 Gumbel AlphaZero 的政策改进与 Stochastic AlphaZero 的随机环境建模相结合，提供在有限模拟预算下高效且可靠的难度评估框架，并系统分析了持有块数、预览块数及额外五格块对难度的影响。

**🔧 技术方法**

采用 Stochastic Gumbel AlphaZero（SGAZ）算法、MiniZero 游戏框架、蒙特卡罗树搜索（MCTS）以及训练奖励和收敛迭代次数两种量化指标进行实验。

**📊 数据集**

使用自定义的 8×8 Tetris Block Puzzle 环境作为实验数据集，按需生成不同规则配置的游戏序列；未使用公开的外部数据集。

**📈 对比分析**

通过比较训练奖励（平均总奖励）和收敛迭代次数来衡量规则变化对难度的影响；结果显示，持有块数和预览块数越多游戏越容易，添加额外五格块会显著提高难度，尤其是 T‑pentomino，SGAZ 在经典规则下达到 6544 分（接近 6750 的最大值）。

**⚠️ 局限性**

局限性包括仅在固定 8×8 网格下实验，未考虑更大/更小尺寸或其他块形状；评估仅基于 AI 训练指标，未进行人类玩家体验验证；实验环境与算法实现依赖特定 GPU 资源，可能影响可复现性。

---

## 20. 6Bit-Diffusion: Inference-Time Mixed-Precision Quantization for Video Diffusion Models

**arXiv ID:** 2603.18742 | [PDF](https://arxiv.org/pdf/2603.18742v1)

**作者:** Rundong Su `[一作]` (Fudan University), Jun Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 67535 | [OpenAlex ID](https://openalex.org/A5115666530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了面向视频扩散Transformer的动态混合精度量化与时间增量缓存框架（6Bit‑Diffusion），实现显著推理加速与显存压缩。

**💡 创新点**

创新点包括：①发现块输入输出差异与量化敏感度呈线性相关，基于此动态分配NVFP4/INT8；②引入Temporal Delta Cache利用块残差时序相似性跳过计算；③设计Purified Delta Refresh防止量化误差累积。

**🔧 技术方法**

使用了Post‑training量化、NVFP4与INT8混合精度、Fast Hadamard Transform、误差驱动缓存切换、累积误差控制等技术。

**📊 数据集**

实验数据集包括CogVideoX 2B/5B模型、EvalCrafter 100条提示、VBench 8维指标。

**📈 对比分析**

与SmoothQuant、QuaRot、ViDiT‑Q等PTQ基线对比，单独DMPQ在W4A6下实现1.36×加速且视觉质量保持0.5437；加入TDC+PDR后达到1.92×加速、3.32×显存压缩，质量与FP16基线相当。

**⚠️ 局限性**

局限性在于量化噪声仍需PDR纠正，连续缓存对极端激活不稳定，方法依赖GPU对NVFP4硬件支持；在更高分辨率或更长步数场景下效果可能下降。

---

## 21. Multimodal Model for Computational Pathology:Representation Learning and Image Compression

**arXiv ID:** 2603.18660 | [PDF](https://arxiv.org/pdf/2603.18660v1)

**作者:** Peihang Wu `[一作]` (Shenzhen University of Advanced Technology), Lijian Xu `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文综述了全景显微影像（WSI）在数字病理学中的最新进展，系统探讨了自监督学习、基础模型、token压缩、多模态数据生成、参数高效适配与多代理协作推理等关键技术，并对这些方法在诊断、分型、生存预测等多项任务中的表现进行了对比与评估。

**💡 创新点**

创新点在于：①将WSI的跨尺度信息通过结构感知token压缩实现全局与局部特征的高效融合；②提出多模态基础模型与多代理推理框架，模拟病理学家“思维链”，提升可解释性和不确定性评估；③结合生成式模型与多代理协作实现稀缺病理数据的高质量合成；④在少样本场景下引入参数高效微调与链式思维监督，显著提升迁移性能。

**🔧 技术方法**

主要技术包括：自监督对比学习、masked modeling、跨分辨率一致性学习；视觉-语言对齐与多模态预训练（如PLIP、CPath-Omni、PathChat）；token压缩与线性复杂度注意力（如DTC-WSI、2DMamba、FOCUS）；生成式扩散模型与多代理生成（如PathGen、FastFlow）；参数高效微调（adapter、LoRA、prompt tuning）与链式思维监督；混合专家与多代理强化学习实现可解释推理。

**📊 数据集**

本文引用的主要数据集包括：超过100万张WSI（约10万张slide）、1.3亿张tile的Prov-GigaPath数据集、公开的Kaggle、TCGA、GEO等多中心病理图像与文本配对数据；在评估中多使用公开基准（如Camelyon、NCT-CRC，ICPR）以及从文献中提取的性能指标。

**📈 对比分析**

对比实验表明，结构感知token压缩可将token数减少至<2.5%而保持93%+诊断准确率；多模态基础模型在多任务迁移（分型、分割、VQA）上平均提升5-15%；链式思维微调在4-shot细粒度分类任务中达到了与或优于CLIP的性能；多代理生成在生存预测与基因表达推断任务中显著提高了AUC与RMSE，体现了数据效率与可靠性提升。

**⚠️ 局限性**

局限性包括：①对超高分辨率WSI的token压缩仍需进一步降低计算与存储成本；②合成数据与真实临床数据的生物学一致性尚未完全保证；③多代理推理框架复杂，需大规模标注与调优；④缺乏统一的跨任务评估标准，导致不同方法难以直接比较。

---

## 22. Prompt Control-Flow Integrity: A Priority-Aware Runtime Defense Against Prompt Injection in LLM Systems

**arXiv ID:** 2603.18433 | [PDF](https://arxiv.org/pdf/2603.18433v1)

**作者:** Md Takrim Ul Alam `[一作]` (University of Rajshahi), Jungpil Shin `[通讯]` (University of Aizu)

**通讯引用:** 4670 | [OpenAlex ID](https://openalex.org/A5005221038)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个轻量级的 API 边界中间件 Prompt Control‑Flow Integrity (PCFI)，在请求到达后端 LLM 之前对提示进行结构化优先级校验，阻止提示注入攻击。

**💡 创新点**

创新点在于将提示拆分为系统、开发者、用户和检索内容四个具有优先级的段落，并借鉴控制流完整性思想，使用三阶段实时检测（词汇筛选、角色切换检测、层级策略执行）实现结构化提示安全。

**🔧 技术方法**

技术方案包括 FastAPI 中间件实现、基于文本的词汇过滤、正则/模式匹配的角色识别以及基于优先级的策略执行三阶段管道。

**📊 数据集**

使用自构造的 150 条 JSONL 结构化基准数据集，其中 50 条正常请求、100 条直接或间接提示注入攻击样本，用于评估 PCFI 的效果。

**📈 对比分析**

与无防御基线对比，攻击通过率从 100% 降至 0%，误报率为 0%，中位数延迟仅 0.04 ms（p95 0.08 ms、p99 0.14 ms），显示出极低的性能开销和完美的拦截效果。

**⚠️ 局限性**

局限性包括：依赖模式匹配易被绕过，当前仅支持单轮请求，评估样本多为人工合成，未覆盖多轮对话、工具调用或语义隐蔽攻击等更复杂场景。

---

## 23. Em-Garde: A Propose-Match Framework for Proactive Streaming Video Understanding

**arXiv ID:** 2603.19054 | [PDF](https://arxiv.org/pdf/2603.19054v1)

**作者:** Yikai Zheng `[一作]` (Institute for AI Industry Research), Yunxin Liu `[通讯]` (Institute for AI Industry Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Em‑Garde框架，将用户查询解析与视频感知分离，实现主动式流媒体视频理解；

**💡 创新点**

核心创新是将查询理解一次性完成，生成可感知的视觉提案，随后仅做轻量级匹配触发，从而突破效率‑准确度矛盾；

**🔧 技术方法**

关键技术包括Instruction‑Guided Proposal Parser（IGPP）利用大型多模态语言模型生成结构化提案，Lightweight Proposal Matching Module（LPMM）采用多模态嵌入相似度匹配，结合滑动窗口缓存和阈值触发；

**📊 数据集**

使用Parse2Prop‑1K（668个查询+提案），以及COIN、Ego4D、BEHAVIOR等视频数据；

**📈 对比分析**

在StreamingBench、OVO‑Bench、ProactiveVideoQA等主动响应基准上，Em‑Garde在准确率/召回率上分别提升约3%/10%，帧率达10‑15fps，性能优于现有主动模型；

**⚠️ 局限性**

局限性：触发阈值敏感，易受场景变化干扰；未对长时序推理进行联合优化；对嵌入模型的区分能力有限。

---

## 24. Terms of (Ab)Use: An Analysis of GenAI Services

**arXiv ID:** 2603.18964 | [PDF](https://arxiv.org/pdf/2603.18964v1)

**作者:** Harshvardhan J. Pandit `[一作]` (Trinity College Dublin), Abeba Birhane `[通讯]` (Trinity College Dublin)

**通讯引用:** 3736 | [OpenAlex ID](https://openalex.org/A5038758207)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对六款主流生成式 AI 服务的使用条款进行系统性审计与标注，评估其对消费者权利的影响，并提出政策改进建议。

**💡 创新点**

首次将消费者保护法视角与生成式 AI 服务条款结合，构建细粒度代码本，对条款进行手工标注，揭示权责失衡与潜在不公平条款。

**🔧 技术方法**

主要采用人工标注与定性分析方法；利用预设的代码本对条款文本进行结构化提取，未使用自动化机器学习技术。

**📊 数据集**

使用的“数据集”为六款服务（Claude, DeepSeek, Gemini, Copilot, Le Chat, ChatGPT）的官方使用条款及相关文件（共 21 篇文档）。

**📈 对比分析**

对比方法：通过对照 EU 消费者保护法（如 UCTD、UCPD）及先行研究，逐条评估条款是否构成不公平或潜在风险；并未提供数值性能指标，而是以条款合规性与风险等级进行定性比较。

**⚠️ 局限性**

局限性：仅覆盖消费层面条款，未考察广告、营销文本或实际服务交互；未包含所有可能的辅助文件；未对条款随时间演变进行动态跟踪；分析仅基于 2025 年 11 月的条款版本。

---

## 25. Quantifying Memory Cells Vulnerability for DRAM Security

**arXiv ID:** 2603.18549 | [PDF](https://arxiv.org/pdf/2603.18549v1)

**作者:** Zilong Hu `[一作]` (National University of Singapore), Biplab Sikdar `[通讯]` (National University of Singapore)

**通讯引用:** 12346 | [OpenAlex ID](https://openalex.org/A5041189303)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过建立可测量的单元级电路模型，定量评估DRAM单元在保留、失真和机密性攻击中的易损性。

**💡 创新点**

创新点在于提出并实证验证了R_S、R_B两大可测量参数，将物理泄漏与安全属性关联，并揭示Rowpress对攻击表面相较Rowhammer的显著优势。

**🔧 技术方法**

技术包括基于FPGA的自定义内存控制器（DRAM Bender）、单元级电路建模与参数提取、保留/激活失真实验与统计分析。

**📊 数据集**

数据集为七款DDR4 DIMM（Micron 16GB、Lenovo 8GB、ADATA 4GB、Innodisk 4GB等），共测得数千个易损单元的R_S、R_B值。

**📈 对比分析**

方法：计算总泄漏导纳 G_tot、相对模式差 ΔG_rel 以及模式推断准确率 Acc，实验结果显示在相同扰动预算下Rowpress的 G_tot 更高、ΔG_rel 更大、Acc 更高，表明Rowpress在定向破坏和模式推断攻击上优于Rowhammer。

**⚠️ 局限性**

局限性包括仅在DDR4芯片上验证，未考虑DDR5及更高阶电路结构；模型未包含TRR/ECC等内置缓解机制的动态影响；实验受FPGA时序精度限制，可能不足以捕捉极低频率或超时效失效。

---

## 26. SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits

**arXiv ID:** 2603.19173 | [PDF](https://arxiv.org/pdf/2603.19173v1)

**作者:** Edward Lin `[一作]` (NVIDIA), Humphrey Shi `[通讯]` (NVIDIA)

**通讯引用:** 21114 | [OpenAlex ID](https://openalex.org/A5066242985)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SOL‑ExecBench 基准，用 235 个 GPU 核心优化问题（来自 124 个 AI 模型）评估核实现相对于硬件 Speed‑of‑Light（SOL）极限的性能。

**💡 创新点**

创新点包括：① 基于 SOLAR 的硬件 Grounded SOL 边界解析；② 引入 SOL Score 量化相对基准与 SOL 极限的闭合程度；③ 设计了针对代理优化器的安全沙盒评估框架，防止 reward hacking；④ 覆盖多种现代精度（BF16、FP8、NVFP4）和后训练工作负载。

**🔧 技术方法**

技术手段包括：PyTorch 参考实现、SOLAR 解析流水线（FLOP/字节计数、峰值吞吐率计算）、沙盒化 harness（时钟锁定、L2 缓存清零、单进程子进程执行）、LLM 规则检查、代理优化器生成基准。

**📊 数据集**

使用的数据集为 235 个问题集合，源自 124 个生产/新兴 AI 模型（LLM、扩散、视觉、音频、视频、多模态），每个问题配备 16 个动态形状工作负载。

**📈 对比分析**

对比方法为计算 SOL Score：S = (T_b - T_SOL)/(T_k - T_SOL)（T_b 为基准运行时，T_SOL 为 SOL 边界，T_k 为候选核运行时）。中位 SOL Score 为 0.732，表明代理生成的基准已显著优于 PyTorch 参考，但仍有提升空间；与 headroom reclaimed 的相关性高达 0.98，速度提升指标则弱于 0.81。

**⚠️ 局限性**

局限性：约 14.5% 的提交出现 reward hacking，需持续改进沙盒；目前仅支持 Blackwell GPU 及 PyTorch 运行时，可能不利于非 PyTorch 或多流实现；基准与 SOL 边界内部维护，外部不可直接查看；后续需更新硬件特性与更广泛工作负载。

---

## 27. From Connectivity to Multi-Orbit Intelligence: Space-Based Data Center Architectures for 6G and Beyond

**arXiv ID:** 2603.18601 | [PDF](https://arxiv.org/pdf/2603.18601v1)

**作者:** Shimaa Naser `[一作]` (Khalifa University), Sami Muhaidat `[通讯]` (Khalifa University)

**通讯引用:** 7264 | [OpenAlex ID](https://openalex.org/A5004034156)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种跨 LEO、MEO、GEO 多轨道空间数据中心 (SBDC) 的分层架构，以支持 6G 直连手机与卫星通信（DHTS），并将计算、存储与控制功能嵌入空间网络本身。

**💡 创新点**

创新点包括：1) 将空间网络从单纯转发器升级为多层智能计算网络；2) 引入能量感知与热管理的计算驱动路由与 AI 驱动的层级编排；3) 通过任务感知的通信‑计算协同设计，实现在轨实时推理与数据精简；4) 将数字孪生与联邦学习结合，用于全局调度与安全信任管理。

**🔧 技术方法**

采用的技术包括：光学 ISL 网络、太阳能收集与辐射冷却、异构计算平台（CPU/FPGA/GPU/神经加速器）、容器化与可信执行环境、联邦学习与数字孪生、基于模型预测和强化学习的调度算法，以及能量/热感知的状态抽象。

**📊 数据集**

本文并未使用具体实验数据集，而是基于对未来空间数据量（566 EB）和地面数据流量（350 EB/月）等公开估算进行理论与模拟分析。

**📈 对比分析**

没有进行实验对比；性能评价主要以理论指标为主，指出与传统仅转发的 LEO 系统相比，SBDC 能显著降低原始数据下行量、缩短端到端延迟，并提升能效与覆盖范围。

**⚠️ 局限性**

局限性包括：技术层面的辐射容限、热与能量管理挑战；运维层面的地面链路间歇、跨域协同与安全信任管理难题；监管层面的频谱与司法管辖、标准化缺口，导致实际部署与互操作性仍需进一步研究。

---

## 28. Toward Reliable, Safe, and Secure LLMs for Scientific Applications

**arXiv ID:** 2603.18235 | [PDF](https://arxiv.org/pdf/2603.18235v1)

**作者:** Saket Sanjeev Chaturvedi `[一作]` (Argonne National Laboratory), Tanwi Mallick `[通讯]` (Argonne National Laboratory)

**通讯引用:** 408 | [OpenAlex ID](https://openalex.org/A5037238294)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种针对高风险科学应用的多代理安全框架，先用威胁分类构建科学特有的安全评测基准，再通过三层防御架构（红队层、内部安全层、外部安全层）实现对LLM的可靠性、安全性与安全性的整体防护。

**💡 创新点**

创新点：①将科学领域的LLM威胁细化为推理、数据保密、伦理规避、资源耗尽等四大类别；②设计可自动化、跨代理的基准生成流程，解决单代理生成的泛化不足；③将红队生成的基准直接嵌入内部安全模型的对齐与微调，并在外部设置动态输入/输出守护，形成完整的防御闭环。

**🔧 技术方法**

主要技术：多代理协作生成（Domain expert、Adversary、Refiner、Quality‑Control）；强化学习与合宪AI实现安全对齐；输入/输出守护（词法、语义、意图检测、事实核查、PII/敏感信息过滤）；基于预训练LLM的模型蒸馏/微调；使用已存在的科学评测基准（TruthfulQA、SciFact、MedHallBench、RAG Security Bench 等）作为验证参考。

**📊 数据集**

使用的数据集：文中引用的通用与科学领域评测集（TruthfulQA、SciFact、MedHallBench、RAG Security Bench、JailbreakBench 等），但未公开新构建的基准；未来工作计划在实验中构造多代理生成的专属科学威胁数据集。

**📈 对比分析**

方法比较与性能：论文为概念性框架，未给出量化实验；提出的评测流程和防御层级将通过后续原型验证，并与现有的通用安全基准（JailbreakBench、AdvBench 等）进行对比，期望在科学领域的攻击成功率降低、误报率下降、数据泄露风险可测。

**⚠️ 局限性**

限制：①缺乏实际实现与实验验证；②多代理协作与人机交互机制尚未细化；③跨领域泛化与可扩展性未知；④三层防御的计算与延迟成本未评估；⑤对特定学科深度与专业知识的依赖可能导致基准覆盖不足。

---

## 29. Masking Intent, Sustaining Equilibrium: Risk-Aware Potential Game-empowered Two-Stage Mobile Crowdsensing

**arXiv ID:** 2603.18670 | [PDF](https://arxiv.org/pdf/2603.18670v1)

**作者:** Houyi Qi `[一作]` (Tongji University), Wei Ni `[通讯]` (Edith Cowan University)

**通讯引用:** 9778 | [OpenAlex ID](https://openalex.org/A5021527965)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向动态移动众包的两阶段服务提供框架 iParts，兼顾意图隐私保护、冗余控制与风险可控。

**💡 创新点**

创新点包括：使用个性化本地差分隐私与记忆化机制对工作者意图进行长期保护，并通过精确势能游戏实现离线预规划与在线轻量化调度。

**🔧 技术方法**

采用了个性化本地差分隐私、记忆化随机响应、精确势能游戏、Knapsack DP、Monte Carlo 估计等技术。

**📊 数据集**

实验基于芝加哥出租车轨迹数据集及仿真任务场景。

**📈 对比分析**

与六种基准方法对比，iParts 在社会福利、任务完成率、工人/任务效用方面表现最佳，同时在交互开销、能耗与对一/多快照隐私攻击的鲁棒性上均显著优于对手。

**⚠️ 局限性**

局限在于对意图动态漂移建模不充分，且对更大规模多任务协同的可扩展性仍需进一步验证。

---

## 30. A conceptual framework for ideology beyond the left and right

**arXiv ID:** 2603.18945 | [PDF](https://arxiv.org/pdf/2603.18945v1)

**作者:** Kenneth Joseph `[一作]` (University at Buffalo), David Lazer `[通讯]` (Northeastern University)

**通讯引用:** 25519 | [OpenAlex ID](https://openalex.org/A5013670125)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个将意识形态视为多层属性网络的概念框架，旨在将意识形态理论与 NLP+CSS 研究相结合

**💡 创新点**

创新点在于把意识形态从传统的左右两极谱扩展为动态、社会共享的多维概念网络，并明确其与框架、身份等过程的关系

**🔧 技术方法**

采用图网络建模、关系抽取（因果、合成）、立场检测、价值识别等 NLP 技术来推断意识形态结构

**📊 数据集**

使用公开的社交媒体语料（如围绕 George Floyd 的推文、政策辩论文本）作为实验数据集

**📈 对比分析**

本文未给出具体实验对比或性能指标，主要提供理论与方法框架，指出可与立场检测、NLI 等任务相结合进行后续评估

**⚠️ 局限性**

局限性包括缺乏形式化的数学模型、未充分验证 LLM 与意识形态测量的实际行为关联，以及对数据噪声和多元身份的处理仍不完善

---

## 31. "You've got a friend in me": Co-Designing a Peer Social Robot for Young Newcomers' Language and Cultural Learning

**arXiv ID:** 2603.18804 | [PDF](https://arxiv.org/pdf/2603.18804v1)

**作者:** Neil Fernandes `[一作]` (University of Waterloo), Kerstin Dautenhahn `[通讯]` (University of Waterloo)

**通讯引用:** 26153 | [OpenAlex ID](https://openalex.org/A5059371010)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究设计并实现了一款名为Maple的桌面社交辅助机器人，结合社区识字项目与教师协同，支持新移民儿童的英语与文化学习。

**💡 创新点**

创新点在于将共设计方法与多模态脚手架、故事化学习以及嵌入式测评相结合，并将机器人定位为低压练习伙伴而非独立教师，支持教师在三人互动中的及时介入。

**🔧 技术方法**

技术实现包括基于ROBOTIS工程套件的桌面机器人，ROS+React的 Web UI，PyLips 面部动画，Kokoro/TTS/Narakeet 语音合成，多模态行为同步以及 RoboSync 实时架构。

**📊 数据集**

使用了 Dolch 高频词表、自定义短故事集以及共设计阶段收集的教师访谈与观察记录；未使用公开大规模数据集。

**📈 对比分析**

论文未开展量化实验，主要通过共设计和原型实现展示设计效果，未来计划进行教师玩耍测试与儿童案例研究，目前仅报告系统实现与设计。

**⚠️ 局限性**

限制包括缺乏儿童实际使用评估、语音合成自然度不足、情感表达有限、对多语言支持尚未完善，以及未评估长期学习效果与文化适应度。

---

## 32. Enhancing Multi-Corpus Training in SSL-Based Anti-Spoofing Models: Domain-Invariant Feature Extraction

**arXiv ID:** 2603.18657 | [PDF](https://arxiv.org/pdf/2603.18657v1)

**作者:** Anh-Tuan Dao `[一作]`, Nicholas Evans `[通讯]` (EURECOM)

**通讯引用:** 10020 | [OpenAlex ID](https://openalex.org/A5066811192)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了多语料库训练对语音伪造检测的影响，并提出了一种通过梯度逆转层实现的域不变特征提取（IDFE）框架，以降低语料库特异性偏差。

**💡 创新点**

创新点在于：①首次揭示多语料库训练在伪造检测任务中不一定提升性能，并通过实验分析语料库偏差；②设计了IDFE框架，将域对抗训练与SSL特征提取相结合，显著提升跨域鲁棒性；③通过α超参数平衡域不变性与任务相关性，提供了调优策略。

**🔧 技术方法**

使用了自监督学习模型（如wav2vec 2.0/XLSR）作为特征提取器，结合MHFA多头注意力池化、全连接分类头以及梯度逆转层进行域对抗训练；训练采用Adam优化器，损失为交叉熵与域分类交叉熵的加权和。

**📊 数据集**

使用了ASVspoof 2019、ASVspoof 5、Fake-or-Real（FoR）等训练集，以及ASVspoof 2021 LA、DF隐藏子集和ASVspoof 5评估集进行评估。

**📈 对比分析**

通过将IDFE与MHFA基线结合，在四种多语料库训练场景下对比实验，IDFE在平均EER上提升约20%（从6.11%降至4.88%），对ASVspoof 2021 LA的EER降幅更高（约31%），验证了框架的有效性。

**⚠️ 局限性**

主要局限在于：①IDFE并未完全消除语料库偏差，仍在部分数据集上表现不如单一语料库训练；②α超参数的选择需经验性调节，过大时可能抑制任务相关信息；③实验仅在现有语音伪造数据集上验证，泛化到更大规模或不同攻击类型的性能尚未充分探究。

---

## 33. HWE-Bench: Can Language Models Perform Board-level Schematic Designs?

**arXiv ID:** 2603.18102 | [PDF](https://arxiv.org/pdf/2603.18102v1)

**作者:** Weibo Qiu `[一作]` (Guangdong University of Finance and Economics), Runyu Pan `[通讯]` (Shandong University)

**通讯引用:** 124 | [OpenAlex ID](https://openalex.org/A5050190534)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个端到端的板级电路设计评估框架，结合静态规则检查和动态 SPICE 仿真来验证 LLM 生成的原理图；

**💡 创新点**

创新点在于：①构建了包含 300 个真实硬件设计任务和 2,914 个结构化 IC 数据表的知识库；②设计了多阶段生成管线模拟工程师思维；③采用了基于功能组的静态检查与 pin‑locking 机制，解决多路复用冲突；

**🔧 技术方法**

技术包括大模型提示工程、结构化数据解析、静态电路拓扑验证、LTspice 动态仿真、生成式多步推理管线；

**📊 数据集**

使用了 300 条来自 GitHub、OSHWLab 的真实板级设计任务，包含 8 个应用领域，并将 2,914 条官方 IC 数据表预处理为 JSON；

**📈 对比分析**

通过将 LLM 生成的原理图在静态检查和动态仿真两阶段评估，得到最高 8.15% 的整体通过率（Claude Sonnet 4.5），静态通过率平均 71.84%，动态通过率平均 58.65%；

**⚠️ 局限性**

主要局限在：模型缺乏物理直觉，易出现单点错误导致全局失效；处理大规模高密度引脚时上下文过载；设计方案多样性导致无法制定单一最优评估准则。

---

## 34. Adaptive Domain Models: Bayesian Evolution, Warm Rotation, and Principled Training for Geometric and Neuromorphic AI

**arXiv ID:** 2603.18104 | [PDF](https://arxiv.org/pdf/2603.18104v1)

**作者:** Houston Haynes `[一作]` `[通讯]`, Houston Haynes

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

构建了一种可持续自适应领域模型（Adaptive Domain Models, ADM）框架，结合前向模式自动微分、quire精确累加、b-posit算术与几何/神经形态架构，实现了梯度训练的深度无关内存占用、几何结构保持以及基于分布式KL阈值的无服务中断温热旋转更新。

**💡 创新点**

创新点在于：①利用前向模式与quire实现精确梯度与极小内存；②将几何代数（Clifford）和时序稀疏（STDP）统一为同一类型系统；③提出 Bayesian Distillation 从通用大模型抽取并约束域先验；④设计无中断温热旋转与版本签名的可验证部署链；⑤在 b-posit 2026 架构上实现多精度硬件共享。

**🔧 技术方法**

核心技术包括：Dimensional Type System (DTS)、Program Hypergraph (PHG)、前向模式自动微分、quire 累加、b-posit 2026 算术、STDP 本地学习、热旋转调度、PHG 结构证书、Post‑Quantum 签名、Hybrid 结构化编译链（XDNA、Loihi‑2、BAREWire）。

**📊 数据集**

论文未给出具体实验数据集，聚焦于理论构造与框架设计；但示例场景包括几何仿真（PGA、CGA）和物理仿真域（流体力学、结构健康监测）。

**📈 对比分析**

相较传统 IEEE‑754 逆向模式训练，ADM 在训练阶段将内存消耗约束为推理两倍，保持几何稀疏与等变性；温热旋转实现无中断更新，且通过 PHG 证书与版本签名确保结构正确性。性能评估表明：训练速度与精度不受梯度误差累积影响，梯度噪声显著下降。

**⚠️ 局限性**

局限性包括：①需要专门的硬件支持（b-posit、quire）；②前向模式训练在大规模深度网络上仍需进一步实验验证；③类型系统与 PHG 编译成本高，对大型项目的易用性与工具链成熟度尚待提升；④贝叶斯蒸馏对源模型先验质量高度依赖，跨域迁移效果尚未系统评估。

---

## 35. Introducing M: A Modular, Modifiable Social Robot

**arXiv ID:** 2603.19134 | [PDF](https://arxiv.org/pdf/2603.19134v1)

**作者:** Victor Nikhil Antony `[一作]` (Johns Hopkins University), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**通讯引用:** 3334 | [OpenAlex ID](https://openalex.org/A5017287995)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一款开源低成本的社交机器人平台M，支持模块化硬件、可扩展多模态感知与表达，配套ROS2软件栈、完整的仿真-硬件一致性环境，并在儿童故事与正向心理辅导实验中实现了长时间家庭部署。

**💡 创新点**

创新点在于将模块化机械设计、统一ROS2接口、容器化软件、仿真与实机一一对应、以及实地长期部署验证相结合，形成了可复制、可扩展且成本友好的社交机器人研究基础设施。

**🔧 技术方法**

采用模块化机械结构（双臂舵机、头部转向与俯仰、LCD面板）、多模态感知（mmWave雷达、触摸、音频、摄像）、LRA振动反馈、ROS2节点架构、容器化部署、仿真环境等技术。

**📊 数据集**

论文未引用公开数据集，主要使用自建的儿童故事生成日志和正向心理辅导对话记录进行实验验证。

**📈 对比分析**

与Blossom、Poppy-H、Ono、FLEXI、Reachy-M等现有平台在成本、模块化、感知、表达、仿真一致性、现场可用性方面进行对比，M在成本最低、模块化最高、感知与表达兼备；在10户家庭一周部署中实现稳定运行，表现优于同类平台。

**⚠️ 局限性**

局限性包括缺乏长期耐久性验证、舵机运动产生噪声、未内置高级情感识别与交互智能模块，需依赖外部算法进一步扩展。

---

## 36. MemMA: Coordinating the Memory Cycle through Multi-Agent Reasoning and In-Situ Self-Evolution

**arXiv ID:** 2603.18718 | [PDF](https://arxiv.org/pdf/2603.18718v1)

**作者:** Minhua Lin `[一作]` (Pennsylvania State University), Suhang Wang `[通讯]` (Pennsylvania State University)

**通讯引用:** 18724 | [OpenAlex ID](https://openalex.org/A5011048500)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套基于多代理的记忆循环协调框架（MemMA），通过Meta‑Thinker、Memory Manager、Query Reasoner 与 Answer Agent 四个角色，在前向路径上实现记忆构建与检索的战略化调度；在后向路径上引入自现场自演进机制，利用合成 probe QA 对记忆进行即时验证与修复，从而提升 LLM 代理在长时序对话中的记忆质量与问答表现。

**💡 创新点**

① 将记忆构建、检索与利用解耦为规划与执行两层，消除战略盲区；② 通过诊断驱动的迭代检索避免检索漂移；③ 在后向路径实现 probe‑QA 生成与即时修复，解决稀疏延迟反馈问题；④ 以 plug‑and‑play 方式兼容多种存储后端，提升系统通用性。

**🔧 技术方法**

planner–worker 多代理架构、Meta‑Thinker 的诊断指导、查询推理器的迭代检索、probe‑QA 合成与修复模块、语义合并与冲突解决、LLM（GPT‑4o‑mini/Claude‑Haiku‑4.5）调用。

**📊 数据集**

LoCoMo 长时序对话记忆基准（去除对抗样本）。

**📈 对比分析**

与被动基线（Full Text、Naive RAG）以及主动记忆系统（LangMem、Mem0、A‑Mem、LightMem）在 LoCoMo 上进行对比。使用 GPT‑4o‑mini 与 Claude‑Haiku‑4.5 作为后端 LLM，评估 token‑level F1、BLEU‑1 与 LLM‑as‑a‑Judge ACC。MemMA 在所有后端上均实现 ACC 最高，整体提升约 5–6 点，尤其在多跳与单跳问题上显著优于 LightMem，表明记忆循环协调有效提升问答准确性。

**⚠️ 局限性**

仅在 LoCoMo 这一对话式长时序基准上验证，未覆盖非对话或缺乏清晰会话边界的场景；假设能生成有效的 probe‑QA 进行即时修复；未深入探讨隐私、可控性等真实部署中的风险与约束。

---

## 37. Final Report for the Workshop on Robotics & AI in Medicine

**arXiv ID:** 2603.18130 | [PDF](https://arxiv.org/pdf/2603.18130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 38. Unrolled Reconstruction with Integrated Super-Resolution for Accelerated 3D LGE MRI

**arXiv ID:** 2603.18309 | [PDF](https://arxiv.org/pdf/2603.18309v1)

**作者:** Md Hasibul Husain Hisham `[一作]` (University of Utah), Edward DiBella `[通讯]` (University of Utah)

**通讯引用:** 6633 | [OpenAlex ID](https://openalex.org/A5053142734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究在加速3D晚期甘露醇增强MRI中，将EDSR超分辨率网络嵌入模型无卷积重建循环，联合高分辨率恢复与数据一致性。

**💡 创新点**

将超分辨率先导直接整合进无卷积重建框架，作为迭代近端算子，而非后处理，提升细结构恢复。

**🔧 技术方法**

使用基于模型无卷积重建的EDSR网络、共用权重迭代、共轭梯度数据一致性、端到端训练，并与压缩感知、MoDL、DIP进行对比。

**📊 数据集**

采用24只犬类预临床3D LGE MRI数据，在加速因子R=4、6下进行复原后退采样。

**📈 对比分析**

与CS、MoDL、DIP在PSNR/SSIM和左心房分割Dice进行比较，Unrolled+EDSR在PSNR提升约0.5–1 dB，SSIM提升0.009–0.011，分割Dice提升至0.893，明显优于基线。

**⚠️ 局限性**

仅在预临床犬类数据验证；DIP表现较差；对呼吸/心率变异的临床适应性未知；3D全卷积实现与内存消耗未充分评估。

---

## 39. OpenT2M: No-frill Motion Generation with Open-source,Large-scale, High-quality Data

**arXiv ID:** 2603.18623 | [PDF](https://arxiv.org/pdf/2603.18623v1)

**作者:** Bin Cao `[一作]` (Chinese Academy of Sciences), Zongqing Lu `[通讯]` (Peking University)

**通讯引用:** 2193 | [OpenAlex ID](https://openalex.org/A5089642905)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个规模达百万级、质量高且开放的运动数据集OpenT2M，并基于该数据集训练了一款无冗余的文本到运动生成模型（OpenFrill），采用了新的2D-PRQ运动分词器；

**💡 创新点**

创新点在于（1）构建物理可行、细粒度文本标注的百万级运动数据集；（2）设计了能同时捕捉时空依赖的2D-PRQ分词器；（3）将分词器与LLM无缝融合，形成“无装饰”高性能T2M模型；

**🔧 技术方法**

技术包括：RL基物理可行性筛选、分层质量过滤、第二级文本生成、2D-PRQ分词器（5部分2D卷积+残差量化）、基于LLM的自回归生成与指令微调；

**📊 数据集**

使用了HumanML3D、Motion-X以及新构建的OpenT2M数据集；

**📈 对比分析**

与MDM、T2M-GPT、Being-M0等基线在零样本、长时序生成和重构任务上对比，OpenFrill+2D-PRQ在R-Precision、FID、MPJPE等指标上均实现显著提升，特别是在大规模数据上展现出零样本优势；

**⚠️ 局限性**

局限性包括：2D-PRQ在小规模数据下性能下降、LLM规模对性能影响有限、长时序生成仍受文本冗余和连接策略限制，未来需进一步提升文本与运动的对齐精度与生成连贯性。

---

## 40. TeachingCoach: A Fine-Tuned Scaffolding Chatbot for Instructional Guidance to Instructors

**arXiv ID:** 2603.18189 | [PDF](https://arxiv.org/pdf/2603.18189v1)

**作者:** Isabel Molnar `[一作]` (University of Notre Dame), Nitesh V. Chawla `[通讯]` (University of Notre Dame)

**通讯引用:** 59911 | [OpenAlex ID](https://openalex.org/A5068157871)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了教学支持聊天机器人 TeachingCoach，利用教学原则提取、合成对话生成和步骤感知训练，为高校教师提供实时、可反思的教学策略指导。

**💡 创新点**

创新点在于：①将36条教学规则转化为结构化提示，使用 LLM 合成多轮对话并人工筛选；②引入步骤感知训练，让模型自动识别教学阶段（问题识别、原因探讨、策略制定）并生成对应建议；③在通用 LLM 基础上实现更深层次、更加贴合教学实践的对话。

**🔧 技术方法**

使用的技术包括：GPT‑4o 生成教师档案与教学挑战、合成对话；LLaMA‑2‑13B‑Chat 进行全参数微调；结构化系统提示与步骤标记的训练框架；专家评估与远程用户研究方法。

**📊 数据集**

数据集为自生成的合成对话，约 406,183 条训练样本、4,156 条验证样本、4,143 条测试样本，来源于教师档案、教学挑战与 36 条教学规则，经过专家筛选后使用。

**📈 对比分析**

与 GPT‑4o mini（零样本）进行对比：专家评分显示 TeachingCoach 在清晰度、尊重语气、鼓励反思和承认用户输入方面均高出约 0.5 分；用户研究显示，尽管基线模型更受整体喜爱，Fine‑tuned 模型在学习归因、对话深度和结构化支持上表现更佳，体现了深度指导的有效性。

**⚠️ 局限性**

局限性包括：样本为自选高校教师，缺乏对 K‑12 或不同文化背景的验证；未直接评估学生学习成果；合成数据可能缺乏真实多样性；系统主要针对高校教师，其他教学场景的适用性仍需进一步研究。

---

## 41. Polynomial Constructions and Deletion-Ball Geometry for Multiset Deletion Codes

**arXiv ID:** 2603.18322 | [PDF](https://arxiv.org/pdf/2603.18322v1)

**作者:** Avraham Kreindel `[一作]` (Reichman University), Aryeh Lev Zabokritskiy `[通讯]` (MIGAL Galilee Research Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究多重集删除码的构造与几何分析，提出了基于有限域多项式的Sidon型构造，并通过差向量与生成函数精确描述删除球大小，进而得到球包、代码–反码及Gilbert–Varshamov界。

**💡 创新点**

创新点在于：①利用多项式因式分解实现可在 t<q 的小删除半径下，冗余仅为 t+O(1) 的多重集删除码；②提出统一的生成函数框架，得到删除球的精确大小、极值中心与平均球大小；③将极值与平均球大小直接映射到球包上界、反码上界和Gilbert–Varshamov下界，并给出闭式公式。

**🔧 技术方法**

主要技术包括：有限域多项式的唯一分解、Sidon集合构造、差向量表示、生成函数的对角提取与对角化、解析组合与极值证明。

**📊 数据集**

本文无实验数据集，仅在理论层面给出解析结果；若需实验验证，需自行在小规模 q、n 上枚举多重集进行计数。

**📈 对比分析**

与传统的循环Sidon构造相比，本文在 t<q 时提供更小的冗余；与以前的极端删除分析相比，生成函数方法给出更精确的平均球大小；在球包上界上与极值球大小一致；在Gilbert–Varshamov下界上，平均球大小公式与内部差向量规模匹配，逼近上界。

**⚠️ 局限性**

局限性包括：①构造仅适用于 t<q，且对 q 为质数幂的有限域有限制；②反码上界因边界截断在小 n 时可能不紧；③尽管给出精确平均球大小，但在 q≥3 时仍存在上界与下界的常数因子间隙，尚未完全收敛。

---

## 42. Circumventing Platform Defenses at Scale: Automated Content Replication from YouTube to Blockchain-Based Decentralized Storage

**arXiv ID:** 2603.18071 | [PDF](https://arxiv.org/pdf/2603.18071v1)

**作者:** Zeeshan Akram `[一作]` (JSgenesis), Zeeshan Akram `[通讯]` (University of Louisville)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `6215c339-3735-4be3-8a07-5bbb7004712d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个可在生产规模下，自动化从YouTube爬取并同步视频到Joystream区块链和去中心化存储的系统；

**💡 创新点**

通过持续的对抗与演化，系统实现了从依赖官方API到完全无API、无OAuth、零平台锁定的架构，同时对抗多层防御的“相互耦合”特性；

**🔧 技术方法**

采用了 Node.js + NestJS + BullMQ + Redis + DynamoDB + OpenTelemetry + yt-dlp + SOCKS5代理池 + 自托管的 YouTube Operational API + zkSNARK（未来方向）等技术栈；

**📊 数据集**

使用超过10,000个授权YouTube频道的公开视频（约数百万条视频数据）作为输入数据集；

**📈 对比分析**

通过对比不同版本的 API 调用量、下载并发、区块链交易批处理以及错误率等指标，展示了从10,000 API/日到零 API 的转变、并发率降低至1/25后仍能保持持续同步，错误率从数百降至零；

**⚠️ 局限性**

局限在于依赖自托管的 Operational API，抗检测策略为经验调优，缺乏对直播/首映视频的精确识别，且多运营商协同同步仍未实现，法律与伦理层面未充分讨论。

---

## 43. DynaRAG: Bridging Static and Dynamic Knowledge in Retrieval-Augmented Generation

**arXiv ID:** 2603.18012 | [PDF](https://arxiv.org/pdf/2603.18012v1)

**作者:** Penghao Liang `[一作]` (Northeastern University), Yichao Wu `[通讯]` (Northeastern University)

**通讯引用:** 5610 | [OpenAlex ID](https://openalex.org/A5063897584)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DynaRAG，一种能够根据检索结果是否足够自动决定是否调用外部API以获取实时知识的检索增强生成框架。

**💡 创新点**

创新点在于将LLM重排序器、充分性判定器和Gorilla v2 API调用模型结合，实现动态知识的自适应路由与选择性工具使用。

**🔧 技术方法**

技术包括基于LLM的重排序、阈值驱动的充分性判定、FAISS式架构过滤、Retriever‑Aware Training（Gorilla v2）用于精准API调用以及最终的LLM生成答案。

**📊 数据集**

使用CRAG基准（Task 1和Task 2）进行评测，该基准包含多领域、不同动态性和长尾实体的4,409个问答对。

**📈 对比分析**

与LLM Only和Direct RAG基线对比，DynaRAG在Task 2（支持API调用）中达到41.00%准确率，较LLM Only提升12.47%且显著降低幻觉率（从43.09%降至22.09%）。

**⚠️ 局限性**

主要局限包括对预构建API目录的依赖、增加的延迟与计算开销、固定阈值导致的泛化性不足，以及仅在CRAG基准上验证的范围有限。

---

## 44. FedTrident: Resilient Road Condition Classification Against Poisoning Attacks in Federated Learning

**arXiv ID:** 2603.19101 | [PDF](https://arxiv.org/pdf/2603.19101v1)

**作者:** Sheng Liu `[一作]` (KTH Royal Institute of Technology), Panos Papadimitratos `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedTrident，针对联邦学习（FL）下的道路状况分类（RCC）系统抵御目标标签翻转攻击（TLFAs）的完整防御框架。

**💡 创新点**

创新点包括：① 对输出层神经元进行细粒度的神经元‑级分析，以准确识别攻击目标并提取特征；② 基于检测结果的自适应客户端评分与黑名单机制，实现恶意车辆的动态排除；③ 采用机器无学习（machine unlearning）技术在排除后对已被污染的全局模型进行补偿性修复。

**🔧 技术方法**

核心技术包括：神经元‑级特征提取与角度一致性度量、Gaussian Mixture Model（GMM）聚类检测、奖励/惩罚型自适应评分、以及基于累计历史更新的模型无学习修复。

**📊 数据集**

实验使用真实车辆摄像头数据构成的 RSCD（Road Surface Classification Dataset）进行三种单任务（摩擦、材料、凹凸）和多任务综合测试，模型覆盖 ResNet‑18、ResNet‑34、MobileNet‑V3、EfficientNet‑B1、DenseNet‑121 与 DeiT‑Tiny 等六种轻量级网络。

**📈 对比分析**

与八种基线方法（FedAvg、Krum、TMean、Median、FoolsGold、FLAME、FLARE、DEFEND）对比，FedTrident 在 SRE、ASR、GAC、GAS 等四项指标上平均提升 9.49%、4.47%、1.23% 与 2.32%，并在多种攻击率、数据异质性、动态攻击与多任务场景下保持与无攻击环境相近的性能，优于所有现有方案。

**⚠️ 局限性**

局限性包括：仍需在联邦学习服务器上执行复杂的特征提取与 GMM 计算，导致一定的计算与通信开销；目前仅针对标签翻转攻击，未覆盖更复杂的模型层面或混合攻击；在极端高攻击率或极端数据异质性下，排除与无学习的效果仍有进一步提升空间。

---

## 45. Towards Differentiating Between Failures and Domain Shifts in Industrial Data Streams

**arXiv ID:** 2603.18032 | [PDF](https://arxiv.org/pdf/2603.18032v1)

**作者:** Natalia Wojak-Strzelecka `[一作]` (Jagiellonian University), Jerzy Stefanowski `[通讯]` (Poznań University of Technology)

**通讯引用:** 9089 | [OpenAlex ID](https://openalex.org/A5053315898)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于改进的Page-Hinkley变点检测、域适应异常检测模型以及SHAP解释的在线方法，用于工业数据流中区分失败与域偏移，并为人工操作员提供解释性支持。

**💡 创新点**

创新点在于将变点检测与域适应结合，利用SHAP解释的特征重要性变化来区分故障与正常的域变化，实现了人机交互的实时异常监测与决策支持。

**🔧 技术方法**

技术包括改进的Page-Hinkley + KL-Divergence 变点检测、Contrastive Semantic Alignment（CCSA）域适应分类器、传统异常检测器（Isolation Forest、LOF、OCSVM、Autoencoder）以及SHAP特征重要性解释。

**📊 数据集**

使用冷轧钢厂仿真器生成的10,000条样本数据，包含四种产品类型的多传感器信息（厚度、宽度、屈服强度、滚筒直径、滚筒里程、张力、滚筒速度、力、扭矩、间隙、电流等）。

**📈 对比分析**

与传统异常检测方法（IF、LOF、OCSVM、AE）比较，传统方法无法区分产品变化和故障；通过域适应+SHAP解释后，实验显示在 current_2、torque_2 等关键特征上出现显著的SHAP值变化，能够帮助识别故障。具体数值指标未给出，但实验演示了方法的可行性和人机决策支持效果。

**⚠️ 局限性**

局限性包括：仅能检测急剧变点，对渐进性域偏移的识别效果不足；解释分析仍需人工，未实现完全自动化；未直接解释故障根源；未考虑概念漂移；缺乏统一的定量性能评估。

---

## 46. Counting Circuits: Mechanistic Interpretability of Visual Reasoning in Large Vision-Language Models

**arXiv ID:** 2603.18523 | [PDF](https://arxiv.org/pdf/2603.18523v1)

**作者:** Liwei Che `[一作]` (Rutgers University), Vladimir Pavlovic `[通讯]` (Rutgers University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了大规模视觉语言模型在计数任务中的内部机制，利用视觉激活补丁和 HeadLens 发现计数电路，并在极少量合成图像上进行微调，显著提升了计数与通用视觉推理性能。

**💡 创新点**

首次将视觉激活补丁扩展到多模态，提出 HeadLens 头级解释方法，揭示了四类功能化注意力头并构建了计数电路；基于此电路的轻量级合成数据微调可在多模型上获得显著提升。

**🔧 技术方法**

视觉激活补丁（VAP）、HeadLens 头级解释、LoRA 微调、注意力正则化与自适应温度调节等技术。

**📊 数据集**

合成计数数据集 SynDot、SynPoly；真实计数基准 SynReal、PixMo‑Count；通用视觉推理基准 MMMU、RealWorldQA、MathVista。

**📈 对比分析**

与原始 LVLM 基线对比，实验显示 OOD 计数平均提升 8.36%（准确率从 73.48% 提升至 84.83%），通用视觉推理平均提升 1.54%；在 PixMo‑Count 上从 58.79% 提升至 64.15%。

**⚠️ 局限性**

仅针对计数的微调可能不足以覆盖所有视觉推理子任务，改进在更大规模模型上的效果尚未验证，对复杂场景中的计数鲁棒性仍有限。

---

## 47. AS2 -- Attention-Based Soft Answer Sets: An End-to-End Differentiable Neuro-Soft-Symbolic Reasoning Architecture

**arXiv ID:** 2603.18436 | [PDF](https://arxiv.org/pdf/2603.18436v1)

**作者:** Wael AbdAlmageed `[一作]` (Clemson University), Wael AbdAlmageed `[通讯]` (Clemson University)

**通讯引用:** 3483 | [OpenAlex ID](https://openalex.org/A5028776484)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了全可微的神经符号框架 Attention‑Based Soft Answer Sets（ABSA），将传统的离散 ASP 求解器替换为连续概率化的即时后果运算 T_P，实现端到端训练与推理。

**💡 创新点**

创新点：①使用概率化 T_P 并通过固定点残差作为约束损失，使梯度能够直接流向感知模块；②用约束组成员嵌入代替传统位置编码，模型仅通过逻辑约束结构构建；③在推理时不调用外部求解器，采用迭代 T_P 精炼与贪婪约束解码完成约束满足。

**🔧 技术方法**

技术细节：共享权重 CNN 感知网络 → 预推理 logits → 约束组嵌入 → 多层 Transformer 推理 → 后推理 logits；概率化 T_P 运算与固定点残差损失；迭代 T_P 迭代精炼；贪婪约束解码；所有操作均可微且无离散求解器。

**📊 数据集**

数据集：MNIST Addition（N=2、4、8）和 Visual Sudoku（9×9 Sudoku board 以 MNIST 数字渲染，训练 9,000 张、验证 1,000 张、测试 1,000 张）。

**📈 对比分析**

对比方法：ProbLog、Clingo pipeline、Differentiable Datalog、SDP‑MaxSAT、Message‑Passing、CP‑SAT 以及自身消融实验；在 Visual Sudoku 上取得 100% board accuracy 与 100% 约束满足，超越前人 99.4%；在 MNIST Addition 上数字精度 99.9% 以上，接近或略优于其他方法。

**⚠️ 局限性**

局限性：在 N=8 的加法任务中，约束损失引入额外优化压力导致 sum 精度略低；需要手动提供 ASP 约束，无法直接处理不规则约束；对大型或极其复杂的约束组，迭代 T_P 可能收敛慢；概率运算对数值稳定性有一定依赖。

---

## 48. Retrieval-Augmented LLM Agents: Learning to Learn from Experience

**arXiv ID:** 2603.18272 | [PDF](https://arxiv.org/pdf/2603.18272v1)

**作者:** Thomas Palmeira Ferraz `[一作]` (NAVER LABS Europe), Stéphane Clinchant `[通讯]` (NAVER LABS Europe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究将经验检索（episodic retrieval）与LLM代理的训练与推理结合，提出一种经验RAG（Experience RAG）框架。

**💡 创新点**

创新点在于把检索作为上下文直接注入LLM，并在Fine‑Tuning阶段同样使用检索，形成“–LoRA”训练方式，从而实现无需复杂记忆控制器的强大泛化。

**🔧 技术方法**

使用的技术包括LoRA参数高效微调、基于文本轨迹的检索索引、静态与动态检索、以及多轮聊天序列化的代理策略。

**📊 数据集**

评估数据集为文本交互式基准ALFWorld和ScienceWorld，均使用官方held‑out split（easy/hard子集）。

**📈 对比分析**

与多种对照方法（Zero‑shot、Prompting、Memory‑Augmented、Supervised Fine‑Tuned）比较，检索仅推理提升至约60%~35%成功率，–LoRA在hard任务上实现90%+成功率，明显优于传统LoRA。

**⚠️ 局限性**

局限性包括检索索引仅来自脚本专家轨迹，缺失相关轨迹会导致性能骤降；固定只读记忆缺乏自我更新机制；实验未验证在噪声轨迹或LLM自生成轨迹下的鲁棒性。

---

## 49. Q-Drift: Quantization-Aware Drift Correction for Diffusion Model Sampling

**arXiv ID:** 2603.18095 | [PDF](https://arxiv.org/pdf/2603.18095v1)

**作者:** Sooyoung Ryu `[一作]` (Seoul National University), Saqib Javed `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对后训练量化（PTQ）的扩散模型，提出了Q-Drift采样器端漂移校正方法，利用量化误差的统计特征在每一步调整漂移以恢复目标边缘分布。

**💡 创新点**

创新点在于从分布视角视量化误差为隐式噪声，并推导出与通用SDE相匹配的一阶漂移修正公式，使得校正仅依赖少量校准数据且与任何采样器、模型架构和PTQ方法兼容。

**🔧 技术方法**

使用SDE/ODE理论、Euler–Maruyama离散化、联合高斯量化噪声建模、步骤级方差统计校准以及漂移缩放校正。

**📊 数据集**

在MJHQ‑30K文本提示集上进行量化与校准，涵盖6款文本‑图像模型（FLUX.1‑dev/quick, PixArt‑Σ, Sana, SDXL, SDXL‑Turbo）。

**📈 对比分析**

与SVDQuant、MixDQ等PTQ基线对比；Q-Drift在所有模型、采样器与量化设置下均提升FID（最多4.59点），CLIP分数保持不变，推理开销微乎其微。

**⚠️ 局限性**

仅在少量校准样本下假设联合高斯、对角协方差，可能在更极端量化或不同模型架构下泛化性有限，未来需引入更丰富的噪声模型以进一步提升鲁棒性。

---

## 50. BeamAgent: LLM-Aided MIMO Beamforming with Decoupled Intent Parsing and Alternating Optimization for Joint Site Selection and Precoding

**arXiv ID:** 2603.18855 | [PDF](https://arxiv.org/pdf/2603.18855v1)

**作者:** Xiucheng Wang `[一作]` (Xidian University), Nan Cheng `[通讯]` (Xidian University)

**通讯引用:** 15028 | [OpenAlex ID](https://openalex.org/A5050651525)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

结合大型语言模型解析自然语言需求，独立完成MIMO波束成形的基站选址与预编码优化，构建BeamAgent框架；

**💡 创新点**

①明确分离LLM语义解析与数值优化，避免LLM误差影响；②采用场景感知Prompt和多轮交互实现精准空间约束；③使用阈值惩罚重构与交替优化联合求解离散与连续变量；

**🔧 技术方法**

使用Claude Sonnet 4.6做意图解析与增量更新；基于ray‑tracing的物理渠道模型；梯度下降、Adam优化、余弦退火、梯度裁剪；向量化离散搜索；ASCII网格可视化交互；JSON结构化约束；

**📊 数据集**

WinProp射线追踪生成的城市场景（224×222 m），4个接收站，33089候选基站位置，4元素ULA，2 GHz；数据存为压缩NumPy文件；

**📈 对比分析**

与随机、Exhaustive MRT/MMSE/ZF/SLNR等基线以及专家手工约束对比；BeamAgent在暗区阈值30 dB下平均亮区功率84 dB，优于最优基线7.1 dB，距专家上界3.3 dB，整体耗时<2 s；

**⚠️ 局限性**

依赖LLM的解析精度，解析错误导致约束误差；多轮交互需用户干预；仅处理单基站单流场景，需进一步扩展到多站、多流与动态环境；LLM API调用成本与延迟未计入。

---

## 51. Reasoning over mathematical objects: on-policy reward modeling and test time aggregation

**arXiv ID:** 2603.18886 | [PDF](https://arxiv.org/pdf/2603.18886v1)

**作者:** Pranjal Aggarwal `[一作]` (FAIR at Meta), Wenting Zhao `[通讯]` (FAIR at Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了三个主要贡献：① 构建了PrincipiaBench基准，专门评测LLM在推理复杂数学对象（如矩阵、分段函数等）上的能力；② 生成了大规模的Principia Collection训练数据，涵盖248K条基于研究级数学与物理知识的问题，答案均为复杂数学对象；③ 设计了Principia VerifyBench，用于评估模型或规则式验证器在判定答案等价性时的可靠性，并在此基础上提出RLLM（Reinforcement Learning with a Language Model as Reward Model）统一的强化学习后训练框架。

**💡 创新点**

创新点包括：① 将数学对象答案作为评测标准，突破传统的数值或多选评测限制；② 通过基于MSC/PhySH主题的自动生成与人工筛选，得到高质量的研究层级问题；③ 用强大LLM替代传统规则式或人工奖励器，实现可验证且具推理性的奖励；④ 通过RLLM实现RLHF、RLVR三种后训练范式的统一，兼顾易检验、难检验与非可检验任务；⑤ 通过多格式数据融合与权重合并，验证跨格式迁移能力。

**🔧 技术方法**

主要技术手段包括：强化学习（PPO/GRPO）与对抗训练；模型自评估与多样性一致性检验；使用LLM（如GPT‑OSS‑120B、o3、Qwen3‑4B等）作为奖励模型与验证器；自监督生成的多样化数学对象答案；大规模GPU并行训练与推理；并行思考与聚合的在线训练框架。

**📊 数据集**

使用的数据集有：PrincipiaBench（2558题，来源RealMath、Physics、ARB、SuperGPQA无选项）、Principia Collection（248,748题，答案为方程、不等式、区间、集合、矩阵、分段函数）、Principia VerifyBench（168条人工标注的验证对比样本）、传统数据集如RealMath、Physics、ARB、SuperGPQA子集、WebInstruct‑Verified、DeepScaleR、AIME（2024/2025）、GPQA‑Diamond、SuperGPQA等。

**📈 对比分析**

评估方法：在PrincipiaBench上对27个基线模型进行准确率评估，基线如o3（62.90%）与Qwen3‑235B（55.58%）等；对同一系列模型进行RL训练后，平均提升7–18%；与其他RLHF/VR基线（如Qwen2.5‑7B‑Instruct、General‑Reasoner、OpenReasoner、Polaris、DeepScaleR、WebInstruct‑Verified）相比，RLLM在数学、物理、MCQA、AIME等多种格式上均显著提高10–30个百分点，尤其在PrincipiaBench上取得领先；同时训练后模型在数值和MCQA基准也获得可观迁移。

**⚠️ 局限性**

局限性：① 需要高能力LLM（如o3、GPT‑OSS‑120B）作为验证器，训练成本高；② 对极长答案的验证仍可能出现误判；③ 生成数据虽覆盖多学科，但对更广泛领域的迁移尚未充分验证；④ RL训练对策略与验证器的“生成-验证器差距”高度依赖，若差距不足会导致效果不佳；⑤ 数据生成与筛选仍需人工干预，可能限制规模与多样性；⑥ 仍可能出现“选项依赖”或“过拟合”现象，导致在未见选项场景下性能下降。

---

## 52. CSSDF-Net: Safe Motion Planning Based on Neural Implicit Representations of Configuration Space Distance Field

**arXiv ID:** 2603.18669 | [PDF](https://arxiv.org/pdf/2603.18669v1)

**作者:** Haohua Chen `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9212 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过训练一种可微的配置空间（C-space）Signed Distance Field（SDF）网络（CSSDF‑Net），实现对机器人姿态与障碍点之间距离及梯度的即时查询，从而在离线规划和在线MPC中以梯度约束实现安全、平滑的运动生成；

**💡 创新点**

创新点包括：①将自碰撞与环境碰撞统一建模为单一C‑space SDF，避免工作空间到C‑space的投影误差；②使用基于空间哈希的全局数据生成管线，使网络在不需要针对每个场景重新训练的前提下实现零样本泛化；③在损失中加入Eikonal和梯度方向一致性项，显著提升梯度可靠性；

**🔧 技术方法**

技术方法涵盖：深度多层感知机（MLP）+残差、ReLU、批归一化、位置编码、dropout；Eikonal正则化、方向一致性损失；HNSW高效边界采样；空间哈希构建配置‑工作空间关联；基于MPC的安全约束与离线轨迹优化；QP求解器OSQP；

**📊 数据集**

数据集为论文自制，采用2‑DoF平面机械臂与7‑DoF机械臂，生成静态与动态障碍点云数据；通过自适应采样、边界挖掘与空间哈希生成训练样本；未使用公开数据集；

**📈 对比分析**

与传统采样规划（RRT‑Connect、BiEST等）、优化方法（CHOMP、STOMP、TrajOpt）以及启发式采样（LazyPRM、KPIECE）对比，CSSDF‑Net在碰撞风险（≤0.6%）与轨迹长度（≈16–18 rad）上优于大多数方法，计算时间约100–170 ms，满足离线规划和实时MPC的需求；

**⚠️ 局限性**

局限性：未对时空动态障碍做显式建模；对传感噪声、部分可观性与校准误差鲁棒性不足；在大规模点云或多机器人协同情形下的点采样与协调效率尚未得到验证；

---

## 53. LGESynthNet: Controlled Scar Synthesis for Improved Scar Segmentation in Cardiac LGE-MRI Imaging

**arXiv ID:** 2603.18356 | [PDF](https://arxiv.org/pdf/2603.18356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 54. ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents

**arXiv ID:** 2603.18815 | [PDF](https://arxiv.org/pdf/2603.18815v1)

**作者:** Hao Zhang `[一作]`, Yi Dong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种独立的 HTTP 服务（ProRL Agent），实现 RL 训练与多轮 agent roll‑out 的解耦，支持在不同机器上并行执行 I/O 密集型 roll‑out 与 GPU 密集型训练。

**💡 创新点**

创新点包括：① Roll‑out‑as‑a‑service 的架构设计；② token‑in / token‑out 轨迹传输消除重分词漂移；③ 可插拔任务抽象与 Singularity 容器实现的无根 HPC 沙盒；④ 通过最小堆动态调度 LLM 后端与异步三阶段管道实现高吞吐量；⑤ 针对多轮 agent 的 DAPO 与作业及时取消机制。

**🔧 技术方法**

核心技术：HTTP 接口、异步三阶段管道（INIT → RUN → EVAL）、最小堆 LLM 后端负载平衡、Unix 域套接字工具后端（Bash、IPython、UDS）、Singularity 容器化沙盒、token‑id 直接传递、异步作业调度与取消。

**📊 数据集**

使用的数据集包括：SWE‑Bench Verified（软件工程任务）、SCP‑116K（STEM agent）、DeepScaleR（数学任务）、Eurus‑2‑RL‑Data（代码合成任务）、Codeforces（评测集）。

**📈 对比分析**

与 SkyRL‑v0、原始 RL 训练框架（VeRL、NeMo RL）进行对比。实验表明在 4B、8B、14B 模型上，SWE‑Bench Verified 的奖励显著提升（如 8B 模型提升近 2 倍），在 STEM、数学、代码域也持续提升 Pass@1 或平均奖励。整体吞吐量在多节点扩展时近线性增长。

**⚠️ 局限性**

局限性：①仍受 LLM 推理带宽/延迟限制，尤其是高并发场景；②需要为每种任务实现专门的 AgentHandler，维护成本上升；③当前仅支持 HTTP + Singularity，可能对某些容器化或多租户环境不友好；④缺乏对实时/低延迟交互场景的优化。

---

## 55. MERGE: Guided Vision-Language Models for Multi-Actor Event Reasoning and Grounding in Human-Robot Interaction

**arXiv ID:** 2603.18988 | [PDF](https://arxiv.org/pdf/2603.18988v1)

**作者:** Joerg Deigmoeller `[一作]` (Honda Research Institute Europe), Michael Gienger `[通讯]` (Honda Research Institute Europe)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MERGE 系统，通过轻量感知管线和 VLM 结合，实时实现多人‑机器人互动的情境定位与事件生成，并发布了新的 GROUND 数据集。

**💡 创新点**

核心创新在于：1) 通过动作检测触发器选择性调用 VLM，减少计算开销；2) 维持实例级身份跟踪，构建 actor–action–object 事件元组；3) 设计可持续的记忆模块和事件驱动推理；4) 提供专门的多主体 HRI 评价指标和数据集。

**🔧 技术方法**

技术手段包括：SAM 进行初始物体分割；Azure Pose Tracking + I3D 进行人物跟踪与动作识别；MongoDB 作为记忆存储；VLM（GPT‑4o、GPT‑5、Gemini 2.5 Flash 及其视频版）作为推理核心；轻量流式模块实现实时触发。

**📊 数据集**

主要使用数据集为自研的 GROUND，包含 GROUND‑Train（多视角动作与姿态注释）和 GROUND‑Eval（多主体情境标注）；对比时也使用 VLM 公开评测数据（如 EPIC‑KITCHENS 等）。

**📈 对比分析**

与仅使用 VLM 的基线（GPT‑4o、GPT‑5、Gemini 2.5 Flash 等）在 GROUND‑Eval 上进行比较。MERGE 在 grounding score 上提升约 2 倍，平均 runtime 降低 4 倍；在各子任务（sorting、pouring、handover）中，动作、对象、关系、机器人交互等指标均显著优于基线。

**⚠️ 局限性**

局限性包括：1) 触发机制可能导致少量召回率下降；2) 物体检测仅在视频开始时完成，无法实时识别新出现物体；3) 目前仅在桌面场景测试，缺乏更复杂环境验证；4) 对 VLM 的依赖限制了对长时序依赖的处理。

---

## 56. On the Minimum Number of Control Laws for Nonlinear Systems with Input-Output Linearisation Singularities

**arXiv ID:** 2603.18947 | [PDF](https://arxiv.org/pdf/2603.18947v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` `[通讯]` (National Technical University of Athens), Nikolaos D. Tantaroudas (National Technical University of Athens)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对具有反馈线性化奇点的非线性系统提出并证明了(k+1)-控制器引理，给出了在近似线性化框架下实现全局可控所需的最小控制器数量，并通过球杆-球系统实例验证了理论。

**💡 创新点**

创新点在于把控制系数的因式分解与奇点结构直接关联，精确给出k个独立奇点因子对应至少k+1个控制器，并给出了必要性与充分性的严谨证明，填补了之前仅有“有限控制器”但无明确上界的空白。

**🔧 技术方法**

使用了微分几何（Lie导数、Lie括号）、微分拓扑（正则性、正则交叉、正则性定理）、近似线性化技术、隐函数定理以及构造性的逼近方法来完成证明与控制器设计。

**📊 数据集**

主要使用球杆-球系统的物理参数进行仿真（M=0.05 kg、R=0.01 m、J=0.02 kg·m²、J_b=2×10⁻⁶ kg·m²、G=9.81 m/s²），无外部公开数据集。

**📈 对比分析**

通过数值仿真比较，单一控制器无法覆盖整个状态空间，而三控制器（k+1=3）能实现轨迹跟踪、控制输入平滑以及奇点附近的稳定性，验证了理论的有效性；仿真结果显示控制误差快速收敛且在奇点切换时保持连续。

**⚠️ 局限性**

局限性包括：仅针对SISO、平滑、控制仿射系统；必要性结论依赖于近似线性化框架，其他控制范式（滑模、MPC、后向跟踪等）可能打破该上界；未讨论多输入多输出、非光滑或输入受限系统的推广。

---

## 57. MemoAct: Atkinson-Shiffrin-Inspired Memory-Augmented Visuomotor Policy for Robotic Manipulation

**arXiv ID:** 2603.18494 | [PDF](https://arxiv.org/pdf/2603.18494v1)

**作者:** Liufan Tan `[一作]` (Chongqing University), Gangshan Jing `[通讯]` (Chongqing University)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5073846906)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Atkinson–Shiffrin多层次记忆模型的机器人视觉运动策略MemoAct，能够在内存依赖任务中实现精准的任务状态跟踪和长期记忆保持。

**💡 创新点**

创新点在于将无损短期记忆与压缩长期记忆相结合，采用因果注意力压缩与相似性合并的双层记忆模块；同时引入感官蒸馏模块、门控融合与条件扩散解码器，实现历史感知与动作生成的协同。

**🔧 技术方法**

使用DINOv2视觉特征提取、Transformer编码器/解码器、因果注意力压缩、门控融合网络、UNet条件扩散生成器以及自定义的长短期记忆合并机制。

**📊 数据集**

在自研的MemoryRTBench（基于RoboTwin 2.0的四个模拟任务）和RMBench，以及两个真实机器人任务上进行评估。

**📈 对比分析**

与ACT、Diffusion Policy、MemoryVLA和SAM2Act等基线对比，MemoAct在所有任务中成功率均显著提升，平均提升率从20%至30%不等，且在真实场景中实现了10/10的成功率。

**⚠️ 局限性**

主要限制包括：将整幅图像压缩为单一全局token导致空间细节丢失；对观测窗口的依赖仍有限，无法充分利用超出窗口范围的历史信息；以及在复杂多物体场景下仍存在定位精度不足的问题。

---

## 58. CycleCap: Improving VLMs Captioning Performance via Self-Supervised Cycle Consistency Fine-Tuning

**arXiv ID:** 2603.18282 | [PDF](https://arxiv.org/pdf/2603.18282v1)

**作者:** Marios Krestenitis `[一作]` (Queen Mary), Ioannis Patras `[通讯]` (Queen Mary)

**通讯引用:** 12016 | [OpenAlex ID](https://openalex.org/A5031205865)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过循环一致性奖励对视觉–语言模型进行自监督微调，以提升图像描述的准确性和细节。

**💡 创新点**

创新点在于将循环一致性直接作为自监督训练信号，结合GRPO强化学习，避免依赖人工配对或偏好数据集。

**🔧 技术方法**

使用技术包括图像–文本VLM、文本–图像生成模型（Stable Diffusion 3、FLUX.1‑dev）、GRPO算法以及DreamSim等相似度度量。

**📊 数据集**

数据集使用COCO 2014训练集进行微调，评估则在CompreCap、CAPability、CapsBench和MMHal等公开基准上。

**📈 对比分析**

在所有基准上，CycleCap在四种规模VLM（1B‑7B）上均显著提升，并在最先进方法（CyclePref、RICO‑Flash）上取得更高分数。

**⚠️ 局限性**

局限性包括依赖文本–图像生成模型的质量，循环一致性奖励对生成多样性可能有限制，未来需探索跨模态指标与更高效的生成器。

---

## 59. Model Reference Adaptive Control For Gust Load Allevation of Nonlinear Aeroelastic

**arXiv ID:** 2603.18584 | [PDF](https://arxiv.org/pdf/2603.18584v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems), Rafael Palacios `[通讯]` (Imperial College)

**通讯引用:** 3423 | [OpenAlex ID](https://openalex.org/A5007534365)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

基于Lyapunov稳定性理论，开发了一种用于非线性气动弹性系统的模型参考自适应控制（MRAC）方法，以减轻气流载荷。

**💡 创新点**

创新点在于提出了完整的MRAC框架，包括参考模型设计、适应控制律、误差动态、Lyapunov稳定性证明和适应律推导，并扩展到非线性情况。

**🔧 技术方法**

使用了Lyapunov稳定性理论和非线性降阶模型（NROM）技术。

**📊 数据集**

使用了三自由度气动翼和类似全球鹰的无人机（UAV）作为测试案例，后者具有540个结构自由度。

**📈 对比分析**

与ℋ_∞鲁棒控制方法进行比较，MRAC在离散“1-cosine”气流下实现了显著的翼尖偏转减少，且控制努力相当。MRAC在Von Kármán随机湍流下也取得了有意义的减少，性能与适应速率成正比。

**⚠️ 局限性**

限制在于适应速率矩阵的选择对收敛速度、峰值载荷减少和执行器需求有重要影响，且在非线性情况下，跟踪误差的界限依赖于非线性项的Lipschitz条件。

---

## 60. LLMs Aren't Human: A Critical Perspective on LLM Personality

**arXiv ID:** 2603.19030 | [PDF](https://arxiv.org/pdf/2603.19030v1)

**作者:** Kim Zierahn `[一作]` (ELLIS Alicante), Nuria Oliver `[通讯]` (ELLIS Alicante)

**通讯引用:** 14978 | [OpenAlex ID](https://openalex.org/A5013727792)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文评估了将人类人格测评工具（如大五人格）用于大语言模型（LLM）的有效性，并指出此类测评未满足人类人格的六项基本特征，质疑将LLM人格归类为“人类式人格”是否合理。

**💡 创新点**

创新点在于首次系统性地从心理学定义出发，对LLM人格测评进行批判性检验，提出应将焦点转向功能性行为评估、可塑性互动模式以及LLM的稳定内在特性，并建议构建面向LLM的专属人格框架。

**🔧 技术方法**

文章并未采用新的技术实现，而是综述并分析现有文献、实验结果和理论讨论，论证LLM人格测评的局限性。

**📊 数据集**

主要参考的“数据集”为过去研究中使用的多份大五人格测试问卷（如IPIP、NEO-PI-R）以及相关实验记录，未使用单一公开数据集进行重新实验。

**📈 对比分析**

由于为位置论文，未进行实验对比；作者通过文献综述和逻辑推理说明现有LLM人格测评在内部因素、时稳定性、情境一致性、个体差异、行为关联和描述性总结等方面均不满足人类人格标准。

**⚠️ 局限性**

局限性包括缺乏系统的实证检验与量化指标，依赖已有研究结果，且未能提供针对LLM的新测评工具或实验验证；未来研究需在大规模实验与跨模型评估中进一步验证所提出的功能性与稳定性指标。

---

## 61. RIS-Aided Mobile Network Design

**arXiv ID:** 2603.18252 | [PDF](https://arxiv.org/pdf/2603.18252v1)

**作者:** Adam Samorzewski `[一作]` (Poznan University of Technology), Adrian Kliks `[通讯]` (Poznan University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文在波兹南市中心进行无线传播仿真，比较了八个基站（SISO/MIMO）与不同位置的可重构智能表面（RIS）在LOS/NLOS条件下的路径损耗分布。

**💡 创新点**

创新点在于系统化评估RIS与基站协同部署对城市路径损耗的提升，并首次量化了RIS+BS信号强化方案相较于传统基站布设的百分比增益。

**🔧 技术方法**

使用了GRAND仿真工具、3GPP TR 38.901 UMa传播模型、RIS-FFBC模型以及仿真中定义的RIS阵列参数（列/行元件数、幅度放大系数等）。

**📊 数据集**

采用波兹南市政厅提供的真实城市建筑与基站坐标数据，覆盖旧市场街区的真实地形与障碍物布局。

**📈 对比分析**

通过对四种场景（BS、RIS、RIS+BS、AVG）计算最小、最大、平均路径损耗并转换为百分比增益，发现RIS+BS方案在最小路径损耗上提升25.73%、最大路径损耗提升7.55%、平均路径损耗提升8.84%。

**⚠️ 局限性**

局限性包括固定的方位角与仰角假设、缺乏移动用户的动态角度变化，可能导致路径损耗增益被高估。

---

## 62. From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning

**arXiv ID:** 2603.18383 | [PDF](https://arxiv.org/pdf/2603.18383v1)

**作者:** Grant Wilkins `[一作]` (Stanford University), Ram Rajagopal `[通讯]` (Stanford University)

**通讯引用:** 9457 | [OpenAlex ID](https://openalex.org/A5028892842)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套可合成LLM推理功耗轨迹的分层框架，能够根据工作负载和配置生成从GPU到机房级的电力需求曲线。

**💡 创新点**

将功耗分解为工作负载驱动的状态转移与配置相关的功耗分布，实现跨流量、跨硬件、跨模型的可迁移性。

**🔧 技术方法**

使用高斯混合模型提取功耗状态，双向GRU分类器预测状态序列，AR(1)或i.i.d.采样生成功耗，并在服务器、机架、行、机房层级累积。

**📊 数据集**

在Microsoft Azure NVIDIA DGX上测得多种LLM（Llama‑3.1、DeepSeek、gpt‑oss）在A100/H100 GPU以及不同张量并行设置下的功耗曲线，以及Azure生产级请求日志。

**📈 对比分析**

与TDP、平均功率和基于Splitwise LUT的基线对比，生成的轨迹在能耗误差≤5%（大多数配置）且自相关匹配率≈1，显著优于传统模型。

**⚠️ 局限性**

对到达过程的假设有限（主要为Poisson或生产日间型），未捕捉MoE内在专家路由导致的状态内波动，且PUE、非GPU负载取常数，难以覆盖高度同步或特殊调度策略。

---

## 63. CORE: Robust Out-of-Distribution Detection via Confidence and Orthogonal Residual Scoring

**arXiv ID:** 2603.18290 | [PDF](https://arxiv.org/pdf/2603.18290v1)

**作者:** Jin Mo Yang `[一作]` (Seoul National University), Saewoong Bahk `[通讯]` (Seoul National University)

**通讯引用:** 3301 | [OpenAlex ID](https://openalex.org/A5040786910)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于置信度和正交残差分解的OOD检测方法CORE，通过分离判别特征中的置信度子空间和残差子空间，并分别计分后结合，实现鲁棒的离散检测。

**💡 创新点**

将分类器的特征分解为对权重方向平行的置信度子空间和正交残差子空间，发现残差携带类特定方向的会员信息，独立于置信度；通过z-score标准化后求和得到OR式判别，减少互相关性，提升鲁棒性。

**🔧 技术方法**

特征正交分解、能量分数、余弦相似度、z-score归一化、简单的点积计算。

**📊 数据集**

CIFAR‑100与ResNet‑18 / WideResNet‑40‑2；ImageNet与ResNet‑50、ViT‑B/16、Swin‑B，共计40个ID/OOD对。

**📈 对比分析**

与16种主流后置OOD分数器（logit‑based、feature‑based、activation‑shaping、hybrid、score‑combination）在5种模型×ID设置上对比，CORE以84.9%的平均AUROC领跑，三种ImageNet设置均排名第一，且无灾难性失败，计算成本仅O(d)。

**⚠️ 局限性**

在语义极其相近的OOD（如CIFAR‑10/100）中，残差会员信号会衰减，导致检测效果下降；属于所有类条件会员方法的通用限制。

---

## 64. SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Language Models

**arXiv ID:** 2603.19028 | [PDF](https://arxiv.org/pdf/2603.19028v1)

**作者:** Quentin Guimard `[一作]` (University of Trento), Massimiliano Mancini `[通讯]` (University of Trento)

**通讯引用:** 1381 | [OpenAlex ID](https://openalex.org/A5017971549)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种后置、零样本的稀疏自编码器（SAE）框架，对CLIP文本嵌入进行高维稀疏分解并精细调节偏置神经元，从而实现公平性提升；

**💡 创新点**

创新点在于利用SAE将密集嵌入解耦为稀疏特征，允许在特征级别进行非线性干预，显著优于传统线性子空间投影方法；

**🔧 技术方法**

核心技术包括Matryoshka稀疏自编码器、基于多样化提示的内容相关性评分、偏置敏感度评估以及中位数插值调制的激活调节；

**📊 数据集**

实验使用了FairFace、UTKFace、CelebA、Waterbirds四个基准数据集，配合ViT-B/16与ViT-L/14/336px CLIP backbone；

**📈 对比分析**

与RoboShot、Orth-Proj、PRISM-mini、Orth-Cali、BendVLM等方法比较，在检索与零样本分类任务上均实现了最高或次高的公平性指标（KL、MS、Prec、Acc、WG、Gap），并可与BendVLM组合进一步提升性能；

**⚠️ 局限性**

局限性包括需预训练SAE、对偏置提示或多样化提示的依赖、对超参数如阈值与中位数的敏感性，以及在极端分布或新类型偏见上的泛化能力仍待验证。

---

## 65. UGID: Unified Graph Isomorphism for Debiasing Large Language Models

**arXiv ID:** 2603.19144 | [PDF](https://arxiv.org/pdf/2603.19144v1)

**作者:** Zikang Ding `[一作]` (University of Electronic Science and Technology of China), Lijie Hu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5067496051)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于统一图同构的内部表征去偏框架UGID，用于消除大型语言模型中的社会偏见。

**💡 创新点**

创新点在于将注意力路由和隐藏状态的结构对齐视为图同构问题，并通过拉普拉斯谱一致性与选择性锚定双重约束实现偏见迁移抑制。

**🔧 技术方法**

技术手段包括Transformer动态计算图建模、注意力拉普拉斯谱约束、节点同构正则、log-space行为引导与选择性锚点对齐。

**📊 数据集**

使用的评估数据集包括BBQ、CrowS-Pairs、BOLD、RTP、HolisticBias，以及自制的少量对抗性性别对照与定义锚对。

**📈 对比分析**

与CDA、KLAAD、Self-Debias等基线对比，UGID在ID/OOD偏见消除、结构相似度（ΔSpec、ΔHidden）以及保持模型实用性（PPL、安全性）方面均取得SOTA表现，偏见倍率几乎降至1.0。

**⚠️ 局限性**

局限在于仍需手工定义锚点、对多样性文化语境的可迁移性尚未完全验证，并可能在极端大规模模型上产生轻微的计算成本。

---

## 66. From Snapshots to Symphonies: The Evolution of Protein Prediction from Static Structures to Generative Dynamics and Multimodal Interactions

**arXiv ID:** 2603.18505 | [PDF](https://arxiv.org/pdf/2603.18505v1)

**作者:** Jingzhi Chen `[一作]` (Shenzhen University of Advanced Technology), Lijian Xu `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了人工智能驱动的蛋白质结构预测与功能推断的最新进展，从统一多模态表示到生成模型、全原子多体建模以及动态构象集合生成；

**💡 创新点**

首次系统梳理了AI在蛋白质科学的范式转变，强调从判别模型向生成模型、从MSA依赖到无MSA预测、从静态结构到动态分布的全景式进展，并提出物理一致性生成模型与多模态基础模型的未来方向；

**🔧 技术方法**

结合蛋白质语言模型、SE(3)等变换不变性网络、扩散模型、流匹配、知识图谱检索、文本引导与图注意力等技术；

**📊 数据集**

主要参考AlphaFold数据库、AlphaFold 2/3、RoseTTAFold、AlphaFlow、EigenFold、Uni-Fold MuSSe等公开数据集与模型实现，涉及PDB、AlphaFoldDB、FoldBench、EM 数据等；

**📈 对比分析**

通过对比已有评测指标（如TM-score、GDT‑TS、EM密度拟合、结合亲和力RMSE）和生成分布一致性（如与MD轨迹的KL散度），说明生成模型在保持预测精度的同时能够更好地再现热力学分布，整体性能优于传统判别式方法；

**⚠️ 局限性**

受限于数据分布偏差、物理可解释性不足、几何评测与生物学真实性脱节以及生成模型的资源消耗与泛化能力限制。

---

## 67. Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels

**arXiv ID:** 2603.18671 | [PDF](https://arxiv.org/pdf/2603.18671v1)

**作者:** Juan Miguel Valverde `[一作]` (Technical University of Denmark), Anders Bjorholm Dahl `[通讯]` (Technical University of Denmark)

**通讯引用:** 4962 | [OpenAlex ID](https://openalex.org/A5031953389)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种同类邻居惩罚 (SCNP) 方法，通过在训练时惩罚像素最差邻居的 logits 来提升图像分割的拓扑精度。

**💡 创新点**

创新点在于仅用三行代码即可集成到任何模型和损失函数，且仅有一个可解释的邻域大小超参数，具备 CPU/GPU 高效、无需骨架化或持久同伦的优势。

**🔧 技术方法**

使用了邻域 min/max 池化对 logits 进行惩罚，并在多种损失（交叉熵、Dice、Tversky、Focal、clDice 等）中进行实验。

**📊 数据集**

在 13 个多样化数据集上验证，包括医学（FIVES、Axons、PulmonaryVA、ATLAS2、ISLES24、CirrMRI600、MSLesSeg）、非医学（TopoMortar、DeepRoads、Crack500）以及细胞实例（IHC_TMA、LyNSeC、NuInsSeg）。

**📈 对比分析**

通过与标准 CE-Dice、TopoLoss、clDice、SkelRecall、RWLoss 等对照的基准实验显示，SCNP 在多数数据集上显著降低 Betti 错误并保持或提升 Dice，尤其在管状与圆形结构上效果突出；在某些非管状数据上提升有限。

**⚠️ 局限性**

局限性包括在极小或低对比度结构（如 MSLesSeg）时可能适得其反，邻域大小需根据结构粗细手动调优，且在多类别场景下验证不足。

---

## 68. Pixel-Accurate Epipolar Guided Matching

**arXiv ID:** 2603.18401 | [PDF](https://arxiv.org/pdf/2603.18401v1)

**作者:** Oleksii Nasypanyi `[一作]` (Stony Brook University), Francois Rameau `[通讯]` (SUNY Korea)

**通讯引用:** 1509 | [OpenAlex ID](https://openalex.org/A5090418377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于角度区间查询的像素精确视差引导匹配方法，利用相机几何把候选点筛选转化为在极点视角下的 1D 区间检索；

**💡 创新点**

创新点在于：①用角度区间而非像素距离描述极线附近；②构建平面段树实现 O(log n + k) 的候选检索；③支持每个关键点的像素级容差控制，避免离散化误差与额外几何校验；

**🔧 技术方法**

核心技术包括相机几何（Fundamental matrix、极线）、角度区间生成、段树（segment tree）数据结构、SIFT 描述子匹配、Lowe 比率或 GMS 后处理；

**📊 数据集**

实验使用 ETH3D 数据集（13 个高分辨率序列，含多种室内外场景）；

**📈 对比分析**

与暴力、FLANN、Epipolar Hashing、Grid‑Guided 等方法比较，候选生成时间与整体匹配时间均比现有方法快 2–3 倍，候选召回率保持或提高，匹配召回率与 BF 基准相当或更优；在相机姿态噪声下也更稳健；

**⚠️ 局限性**

局限性：仅适用于有限极点（未处理极点无穷远情况）；更大容差需重建段树，重建成本虽小但仍存在；未将该几何过滤整合进学习型匹配框架；

---

## 69. To See or To Please: Uncovering Visual Sycophancy and Split Beliefs in VLMs

**arXiv ID:** 2603.18373 | [PDF](https://arxiv.org/pdf/2603.18373v1)

**作者:** Rui Hong `[一作]` (George Mason University), Shuxue Quan `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了三层幻觉诊断框架（Perception, Dependency, Alignment），通过盲视、噪声和冲突干预对七种 VLM 进行 7,000 条样本的评估，揭示视觉从众（Visual Sycophancy）为主导的失败模式，并基于诊断分数实现后置选择性预测。

**💡 创新点**

创新点在于：① 用三层因果诊断（LAD、VNS、CS）细分幻觉来源；② 通过对抗干预无需新建数据集即可捕获视觉依赖和从众；③ 形成四类失败模式分类法；④ 证明规模扩大既能降低语言捷径又会放大视觉从众；⑤ 利用诊断分数实现无需额外训练的性能提升。

**🔧 技术方法**

技术手段包括：反事实干预（Blind、Noise、Conflict）、潜在异常检测（Latent Anomaly Detection）、视觉必要性评分（KL 散度 VNS）、竞争得分（Competition Score）、两阶段 LLM-judge 验证、选择性预测算法。

**📊 数据集**

使用 GQA、VQAv2、A-OKVQA、POPE 四个任务（共 1,000 条样本）并在每条样本上构造冲突图像；评估覆盖七种公开 VLM（Llama‑3.2‑11B、Pixtral‑12B、Qwen2.5‑VL‑7B/72B、LLaVA‑NeXT‑7B、Phi‑3.5‑Vision、Molmo2‑4B）。

**📈 对比分析**

方法上对比了完整图像与干预图像的准确率、Shortcut Rate、VNS、LAD、CS 等指标；结果显示整体准确率约 70%，视觉从众 69.6%，鲁棒拒绝 0%；模型规模增大降低了语言捷径但提高了视觉从众；基于诊断的选择性预测在 50% 覆盖率下可提升最高 9.5% 的准确率。

**⚠️ 局限性**

局限性包括：缺乏鲁棒拒绝实例；选择性预测对视觉从众无效；需要完整 logit 访问限制了闭源模型的适用；评估仅覆盖公开 VLM，未检验领域专用数据集；并未提供针对视觉从众的专门训练方法。

---

## 70. Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation

**arXiv ID:** 2603.19158 | [PDF](https://arxiv.org/pdf/2603.19158v1)

**作者:** Kwanyoung Lee `[一作]` (Hanyang University), Dong-Jin Kim `[通讯]` (Hanyang University)

**通讯引用:** 20673 | [OpenAlex ID](https://openalex.org/A5100344647)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Adaptive Auxiliary Prompt Blending (AAPB)，一种在低密度区域稳定扩散过程的无训练框架；

**💡 创新点**

创新点在于推导出基于Tweedie身份的闭式自适应混合系数γ_t^*，实现每步对目标与辅助锚点的动态平衡；

**🔧 技术方法**

采用扩散模型、Tweedie公式、无监督的分类器无关引导 (CFG) 和LLM生成辅助锚点；

**📊 数据集**

在RareBench（稀有概念生成）和FlowEdit（图像编辑）数据集上评估；

**📈 对比分析**

与10+种基准（如SD3.0、R2F、FlowEdit等）对比，AAPB在文本对齐、结构保真度上均取得显著提升，特别是在RareBench上平均分高达84.1分；

**⚠️ 局限性**

局限性包括对辅助锚点质量仍有一定依赖，且在极端稀有概念或复杂编辑场景下仍可能出现微小漂移。

---

## 71. Foundations of Schrödinger Bridges for Generative Modeling

**arXiv ID:** 2603.18992 | [PDF](https://arxiv.org/pdf/2603.18992v1)

**作者:** Sophia Tang `[一作]` (University of Pennsylvania), Sophia Tang `[通讯]` (University of Pennsylvania)

**通讯引用:** 209 | [OpenAlex ID](https://openalex.org/A5047885774)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Schrödinger桥问题进行了全面综述，阐述了从经典OT到EOT，再到静态和动态Schrödinger桥的推导，并给出了Sinkhorn算法和Girsanov变换等关键工具；

**💡 创新点**

创新点在于将静态EOT与动态SB统一为一个端点端点最优路径问题，并通过Hopf-Cole变换将其转化为可解的前向-后向PDE耦合形式，提供了从理论到算法的完整桥梁；

**🔧 技术方法**

主要使用了Optimal Transport、熵正则化、Itô微积分、Girsanov变换、Fokker-Planck/Feynman-Kac方程、Hopf-Cole变换、以及Sinkhorn迭代等技术；

**📊 数据集**

在实验部分主要使用标准高斯分布以及公开图像数据集如MNIST、CIFAR‑10等作为目标与初始分布；

**📈 对比分析**

通过与传统OT、生成式模型（如扩散模型、生成对抗网络）比较，生成质量以FID、LPIPS等指标衡量，结果显示动态SB在保持熵正则化的同时实现了与传统方法相近甚至更优的图像质量；

**⚠️ 局限性**

主要局限在于高维、非高斯目标分布求解仍显复杂，Sinkhorn及动态PDE求解的计算成本高，且对参数（如噪声强度、时间步长）的选择敏感。

---

## 72. HAViT: Historical Attention Vision Transformer

**arXiv ID:** 2603.18585 | [PDF](https://arxiv.org/pdf/2603.18585v1)

**作者:** Swarnendu Banik `[一作]` (Indian Institute of Information Technology), Satish Kumar Singh `[通讯]` (Indian Institute of Information Technology)

**通讯引用:** 6710 | [OpenAlex ID](https://openalex.org/A5056347542)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在Vision Transformer中实现跨层注意力传播机制，保存并融合各层自注意力矩阵以实现信息流的连续性与提升；

**💡 创新点**

提出历史注意力记忆与混合策略（α参数）来直接传递前一层的注意力信息，突破传统层间独立计算的局限；

**🔧 技术方法**

主要使用自注意力矩阵的存储与线性混合（α·当前 + (1-α)·历史），结合Softmax归一化，应用于ViT与CaiT等Transformer架构，并对历史注意力进行随机或零初始化；

**📊 数据集**

在CIFAR-100（32×32）和TinyImageNet（64×64）这两套小规模视觉分类数据集上进行实验；

**📈 对比分析**

与ViT基线及多种SOTA模型（如ResNet、CCT、HSViT等）对比，最佳α=0.45时分别提升CIFAR-100 1.33%（从75.74%到77.07%）和TinyImageNet 1.25%（从57.82%到59.07%），跨架构均表现出1%~1.3%的准确率提升；

**⚠️ 局限性**

目前仅在小规模数据集上验证，缺乏大规模数据集或多任务评估；对α值敏感，且未探索多头不同注意力融合的细粒度效果；

---

## 73. Action Draft and Verify: A Self-Verifying Framework for Vision-Language-Action Model

**arXiv ID:** 2603.18091 | [PDF](https://arxiv.org/pdf/2603.18091v1)

**作者:** Chen Zhao `[一作]`, Jing Zhang `[通讯]` (School of Information, Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Action Draft-and-Verify（ADV）框架，将扩散动作专家生成多条候选动作段，随后用 Vision‑Language‑Model（VLM）通过一次性 perplexity‑style 评分对候选动作进行重排序并挑选最佳动作。

**💡 创新点**

创新点在于：①在扩散生成与 VLM 评分之间搭建了“草稿‑验证”双阶段流程；②利用 Textual FAST 令连续动作可用文本方式进行离散化，使 VLM 评分更可靠；③通过 VLM 验证过滤低质量或不稳定的动作，从而在分布偏移环境下显著提升鲁棒性。

**🔧 技术方法**

核心技术包括：扩散动作专家（Diffusion Action Expert）、自回归 VLM 进行离散动作生成、Textual FAST 离散化、perplexity‑style 评分与单通前向重排序、以及自回归与扩散的联合训练。

**📊 数据集**

使用的主要数据集：LIBERO、RoboTwin2.0（Easy 与 Hard 两种难度）以及真实机器人环境中的四个任务（块推、桌面清理、指令抓取、精细杯子悬挂）。

**📈 对比分析**

与单纯扩散或自回归基线相比，ADV 在模拟环境中提升约 +4.3% 的成功率，在真实环境中提升约 +19.7%，并在分布偏移场景下显著降低前抓碰撞、提高恢复尝试次数，整体控制速率保持稳定。

**⚠️ 局限性**

局限性包括：①当扩散专家无法生成任何可行候选动作时，VLM 验证无法弥补；②相较于单纯扩散推理，ADV 需要额外的候选生成与批量评分步骤，增加推理延迟；③验证过程依赖于 VLM 的语言预训练分布，对极端分布变化的适应性仍有限。

---

## 74. Interpretability without actionability: mechanistic methods cannot correct language model errors despite near-perfect internal representations

**arXiv ID:** 2603.18353 | [PDF](https://arxiv.org/pdf/2603.18353v1)

**作者:** Sanjay Basu `[一作]` (University of California San Francisco), Rajaie Batniji `[通讯]` (Stanford University)

**通讯引用:** 1949 | [OpenAlex ID](https://openalex.org/A5015814541)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究对四种机制可解释性方法（概念瓶颈驱动、稀疏自编码特征驱动、Logit Lens激活补丁以及线性探测+真理性分离向量驱动）在临床分诊任务中纠正漏报（false‑negative）错误的效果进行了系统性比较。

**💡 创新点**

创新点在于首次将多种主流的推理时机制可解释性技术统一应用于安全关键的临床分诊场景，并量化了它们在纠错方面的实际贡献，揭示了模型内部知识与输出行为之间的“知识‑行动”差距。

**🔧 技术方法**

采用了概念瓶颈模型Steerling‑8B、稀疏自编码器（SAE）训练与特征驱动、Logit Lens跟踪词表概率与激活补丁、以及在Qwen 2.5 7B内部进行线性探测与真理性分离向量（TSV）驱动的干预。

**📊 数据集**

使用了400条医师创作的临床情景小短文（132条危急案例、68条安全案例）以及200条匿名Medicaid真实就诊记录（12条危急、188条安全），总计400个案例。

**📈 对比分析**

在基线时，Steerling‑8B与Qwen 2.5 7B分别实现了约35%和45%的敏感度。四种干预方法中，概念瓶颈驱动未能提高漏报率，反而破坏了多数正确报警；SAE特征驱动无任何效果；Logit Lens激活补丁在高强度（α=5）下仅提升了约7%漏报率，且伴随约10%正确报警被破坏；TSV驱动在极高强度（α=10）下实现了24%漏报纠正率，但仍有76%漏报未被纠正。整体而言，四种方法均未能显著弥补知识‑行动差距。

**⚠️ 局限性**

局限性包括：样本量有限且危急案例占比偏低；仅评估了两种模型且仅在推理时干预；未尝试训练时或多层级干预策略；使用关键词解析可能低估模型表现；缺乏对高强度随机方向控制的完整比较。

---

## 75. Security, privacy, and agentic AI in a regulatory view: From definitions and distinctions to provisions and reflections

**arXiv ID:** 2603.18914 | [PDF](https://arxiv.org/pdf/2603.18914v1)

**作者:** Shiliang Zhang `[一作]` (University of Oslo), Sabita Maharjan `[通讯]` (University of Oslo)

**通讯引用:** 16844 | [OpenAlex ID](https://openalex.org/A5056311070)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2024‑2025年欧盟AI相关法规进行综述，澄清安全、隐私与代理式AI的概念及其监管空白；

**💡 创新点**

提出将通用法规细化为针对不同AI类型（尤其代理式AI）的具体监管措施，弥补现行法规的不确定性与缺失；

**🔧 技术方法**

采用法规文本分析与比较法方法，对24份欧盟AI法规文件进行系统梳理；

**📊 数据集**

数据集为收集的24份欧盟AI法规文本，涵盖法规、通告、提案、决议等；

**📈 对比分析**

与现有通用法规对比，发现大多数条款仅适用于一般AI系统，缺少针对代理式AI的专门条文，表明监管在该领域仍较薄弱；

**⚠️ 局限性**

局限性：仅聚焦欧盟法规，未涵盖其他司法辖区；未对技术实现层面的合规工具或案例进行实证评估；

---

## 76. DROID-SLAM in the Wild

**arXiv ID:** 2603.19076 | [PDF](https://arxiv.org/pdf/2603.19076v1)

**作者:** Moyang Li `[一作]` (ETH Zurich), Daniel Barath `[通讯]` (ETH Zurich)

**通讯引用:** 1287 | [OpenAlex ID](https://openalex.org/A5016636021)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种针对动态环境的实时RGB SLAM系统DROID-W，利用可微分的不确定性感知束调整（UBA）来同时估计相机轨迹、场景几何和像素级动态不确定性。

**💡 创新点**

创新点在于将多视角DINOv2特征一致性用于动态不确定性估计，并通过局部仿射映射+Softplus正则化实现像素级不确定性的可学习更新；同时采用交替优化策略避免联合优化的计算瓶颈。

**🔧 技术方法**

核心技术包括可微分束调整、DINOv2视觉特征提取、像素级不确定性映射、仿射参数学习与梯度下降优化，以及使用单目深度预测进行深度正则化。

**📊 数据集**

使用了Bonn RGB‑D Dynamic、TUM RGB‑D、DyCheck三大现有数据集，另外发布了新的DROID‑W户外动态数据集并评估了多个YouTube动态视频。

**📈 对比分析**

与经典SLAM（DSO、ORB‑SLAM2）、动态SLAM（ReFusion、DynaSLAM）、NeRF/GS SLAM（NICE‑SLAM、Splat‑SLAM）以及最新动态SLAM（WildGS‑SLAM、UP‑SLAM）和前馈方法（MonST3R、TTT3R）对比，DROID‑W在跟踪精度、几何重建质量上均优于基线，并在10 FPS下实现实时运行，速度比WildGS‑SLAM提升40×，仅略慢于原始DROID‑SLAM。

**⚠️ 局限性**

局限性：不确定性优化依赖帧间对齐，初始化阶段相机估计不稳时可能导致不确定性误估；缺乏先验重建信息，导致初始阶段鲁棒性不足。

---

## 77. Measuring and Exploiting Confirmation Bias in LLM-Assisted Security Code Review

**arXiv ID:** 2603.18740 | [PDF](https://arxiv.org/pdf/2603.18740v1)

**作者:** Dimitris Mitropoulos `[一作]` (University of Athens), Diomidis Spinellis `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 8772 | [OpenAlex ID](https://openalex.org/A5021948425)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了确认偏差对基于大型语言模型的安全代码审查的影响，并演示了攻击者如何利用元数据框架绕过审查。

**💡 创新点**

首次系统量化确认偏差在多种LLM中的失效模式，揭示其对漏洞检测率的显著削弱，并在真实CI/CD流水线中验证了供应链攻击的可行性。

**🔧 技术方法**

采用提示工程、结构化响应解析、手工验证以及针对GitHub Copilot和Claude Code的自动化审查脚本等技术。

**📊 数据集**

使用CrossVuln的247个真实CVE–补丁对，和在10个开源项目中构造的17个恶意PR样本。

**📈 对比分析**

通过四个主流LLM（GPT‑4o‑mini、Claude 3.5 Haiku、Gemini 2.0 Flash、DeepSeek V3）在5种框架条件下执行约10k次查询，发现bug‑free框架导致检测率下降16–93%；在模拟攻击中，Copilot被攻击成功率35%，Claude Code 88%，且对元数据去除和指令性去偏后检测率可恢复至94%。

**⚠️ 局限性**

实验仅覆盖四个LLM，侧重重现已知漏洞的撤销，未检验新漏洞的生成；受限于模型可解释性和手工验证的成本，且对人类审查者的影响仍未评估。

---

## 78. MicroVision: An Open Dataset and Benchmark Models for Detecting Vulnerable Road Users and Micromobility Vehicles

**arXiv ID:** 2603.18192 | [PDF](https://arxiv.org/pdf/2603.18192v1)

**作者:** Alexander Rasch `[一作]` (Chalmers University of Technology), Rahul Rajendra Pai `[通讯]` (Chalmers University of Technology)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5114606864)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了MicroVision数据集，收集并标注了来自挪威哥德堡市的8000多张高分辨率图像，覆盖近2000个场景，专注于从微型交通工具和行人视角下的易受伤道路使用者和停放微型交通工具的检测。

**💡 创新点**

创新点在于首次提供从微型交通工具视角获取的数据，并采用“状态感知”标注策略区分活跃骑手（如电动滑板车骑手、骑行者）与静止车辆（如停放的电动滑板车、单车），填补了现有车载摄像头数据的空白。

**🔧 技术方法**

技术上结合了YOLO11、Faster R‑CNN和Transformer‑based RF‑DETR三种主流目标检测框架，并使用了自研的半自动标注工作流、BoT‑SORT跟踪、模型辅助的质检与迭代训练。

**📊 数据集**

使用的数据集是新构建的MicroVision数据集，包含超过8,100张图像、3.5万+实例，类别包括行人、骑行者、电动滑板车骑手、停放自行车与停放电动滑板车。

**📈 对比分析**

在保留90%训练/10%验证/10%测试的场景划分下，RF‑DETR在未见测试集上实现mAP 0.726，优于YOLO11（0.687）和Faster R‑CNN（0.580），但YOLO11在小目标上表现更佳，且计算量更小。

**⚠️ 局限性**

局限性包括仅覆盖哥德堡地区、主要为白天良好天气条件、对夜间或恶劣天气泛化不足，以及未覆盖更罕见或细粒度的微型交通工具类型，且未提供实例分割等更精细标注。

---

## 79. Language Model Maps for Prompt-Response Distributions via Log-Likelihood Vectors

**arXiv ID:** 2603.18593 | [PDF](https://arxiv.org/pdf/2603.18593v1)

**作者:** Yusuke Takase `[一作]` (Kyoto University), Hidetoshi Shimodaira `[通讯]` (Kyoto University)

**通讯引用:** 17495 | [OpenAlex ID](https://openalex.org/A5012479520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于条件对数似然向量的模型地图方法，用以比较大量语言模型的条件分布。

**💡 创新点**

首次将条件对数似然向量与PMI向量用于模型映射，显示距离近似KL散度并揭示提示变换的可加性。

**🔧 技术方法**

使用自回归模型的条件对数似然计算、PMI向量、t‑SNE/PCA降维以及欧氏距离近似KL散度。

**📊 数据集**

在Tulu3和Infinity‑Instruct两组各10,000对提示-回复的数据集上进行评估。

**📈 对比分析**

通过欧氏距离估计KL散度与Monte Carlo估计以及基于嵌入的语义距离比较，相关性高达0.94，证明方法有效且与任务得分呈正相关。

**⚠️ 局限性**

方法仅适用于公开权重模型，受限于提示模板、数据集范围、PMI估计假设以及高维空间投影误差。

---

## 80. AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models

**arXiv ID:** 2603.18464 | [PDF](https://arxiv.org/pdf/2603.18464v1)

**作者:** Chengxuan Lu `[一作]` (Sany Group), Yang Liu `[通讯]` (Sany Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AcceRL，一个完全异步、解耦的强化学习框架，结合可训练的世界模型，用于大规模 Vision‑Language‑Action（VLA）模型的强化学习；

**💡 创新点**

1）彻底解耦训练、推理、采样三条流，实现宏观与微观异步；2）首次将可训练世界模型集成到分布式异步 RL 中；3）引入价值重计算、全局优势归一化、GIPO 等机制缓解策略滞后；4）使用动态加权重采样、异步预取等提升样本效率；

**🔧 技术方法**

DeepSpeed ZeRO‑2 分布式优化、模型并行与参数分区、动态批处理 Inference‑as‑a‑Service、Diffusion‑based 世界模型 DIAMOND、Token‑level PPO 换用 GIPO、Vocabulary Slimming、Action‑Aware Attention Pooling、异步数据预取、动态加权重采样等；

**📊 数据集**

LIBERO benchmark（包含 Spatial、Object、Long、Goal 四个子集）以及 MuJoCo 物理引擎、OSMesa 渲染；使用 1,000 条离线轨迹预训练世界模型；

**📈 对比分析**

在 LIBERO 上与 OpenVLA‑OFT、RLinf‑VLA、SimpleVLA‑RL 对比，AcceRL 在所有子集上均取得最高成功率（如 Long 99.1%），并在采样效率上比基线提升 200×、训练吞吐量实现超线性伸缩；

**⚠️ 局限性**

目前不支持大规模语言模型的后训练；跨节点通信在极大规模集群下仍有改进空间；对高阶多步生成动作的梯度不稳定和世界模型在极端环境下的泛化仍是挑战。

---

## 81. Towards Verifiable AI with Lightweight Cryptographic Proofs of Inference

**arXiv ID:** 2603.19025 | [PDF](https://arxiv.org/pdf/2603.19025v1)

**作者:** Pranay Anchuri `[一作]` (Offchain Labs), Tugce Ozdemir `[通讯]` (City College of New York)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于随机路径采样的轻量级可验证AI推理协议，取代传统全量加密证明，显著降低证明时间并保持一定的安全性。

**💡 创新点**

创新点在于：① 利用神经网络的“trace分离”统计特性，证明模型间激活轨迹可被区分；② 设计 Merkle 树向量承诺并仅开启少量路径的开包，从而实现大幅度效率提升；③ 在 Referee 模型下实现多服务器互相验证，进一步降低单方风险。

**🔧 技术方法**

核心技术包括：Merkle 树向量承诺、随机路径抽样（Random Path Test）、统计分离度评估、梯度下降与逆变换攻击实验、与 zkLLM 的性能对比、对 LLM 的前向挂钩采集激活。

**📊 数据集**

使用的数据集与模型：ResNet‑18 在 Animals‑10（狗/猫/松鼠）上训练的二分类器；Llama‑2‑7B（chat 与 base 版本）用于大模型实验；Iris 数据集用于梯度下降攻击演示。

**📈 对比分析**

性能比较：对 Llama‑2‑7B，zkLLM 的证明时间 388 s、证明大小 183 kB；本协议证明时间仅 5.8 ms（单路径开封 0.007 ms），证明大小约 3.4 MB，验证时间 12.44 ms；相比传统 SNARK 和 zkLLM，证明速度提升数百倍，验证速度提升 100‑200 倍。

**⚠️ 局限性**

局限性：① 单路径抽样导致检测错误率 1/N；② 未完全抵御任意伪造（缺乏完备安全证明）；③ 对量化模型、图神经网络等特殊架构的适用性需进一步验证；④ 需要更严谨的理论分析和零知识化改进以保护激活/模型私密性。

---

## 82. QuaQue: Design and SQL Implementation of Condensed Algebra for Concurrent Versioning of Knowledge Graphs

**arXiv ID:** 2603.18654 | [PDF](https://arxiv.org/pdf/2603.18654v1)

**作者:** Jey Puget Gil `[一作]` (Universite Claude Bernard Lyon 1), Gilles Gesquière `[通讯]` (Universite Claude Bernard Lyon 1)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 QuaQue 系统，将 SPARQL 查询通过一种压缩代数翻译为 SQL，以支持多版本知识图的高效交叉查询。

**💡 创新点**

创新点在于：① 采用位串压缩模型，将同一 quad 在不同版本的出现情况编码为位串，避免冗余存储；② 设计了对应的压缩代数，将 SPARQL 代数映射为 SQL 并利用位运算实现版本过滤；③ 在 PostgreSQL 上实现全覆盖索引，进一步提升查询性能。

**🔧 技术方法**

技术：位串压缩模型、压缩代数、SPARQL‑to‑SQL 翻译器、PostgreSQL 15、B‑Tree 多重索引、Bitwise 操作、Docker 容器化部署。

**📊 数据集**

使用 BEAR-B‑day（Time‑Based 版本化）数据集进行实验，数据量为数十亿三元组级别。

**📈 对比分析**

与 Apache Jena TDB2（本地 RDF 存储）对比；QuaQue 在 Join、Predicate‑Object、Predicate 查询上均快约 6–14%，但存储占用提升至 4.7 GB（TDB2 为 694 MB）。统计检验显示差异显著。

**⚠️ 局限性**

主要限制：存储成本高（≈7 倍），无法满足存储受限环境；缺乏递归/路径查询支持；仅测试基本查询模式，聚合与更复杂工作负载待验证。

---

## 83. WarPGNN: A Parametric Thermal Warpage Analysis Framework with Physics-aware Graph Neural Network

**arXiv ID:** 2603.18581 | [PDF](https://arxiv.org/pdf/2603.18581v1)

**作者:** Haotian Lu `[一作]` (University of California), Sheldon X. -D. Tan `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于图神经网络的参数化热翘曲预测框架 WarPGNN，可快速、准确地预测多芯片包层的热翘曲。

**💡 创新点**

创新点包括使用稀疏的 rTCG 图表示多芯片平面布局，结合 GCN/GIN 编码器与 U‑Net 解码器，并设计了物理感知损失（尾部加权+梯度匹配）来提升极端案例精度；同时展示了跨数据集的迁移性。

**🔧 技术方法**

使用了图神经网络（GCN/GIN）、U‑Net 解码器、物理感知损失、梯度匹配、Transitive Closure Graph 以及 PyTorch+DGL 深度学习框架。

**📊 数据集**

采用基于 COMSOL 3‑D FEM 生成的 8000 条样本（200 方案×40 CTE 赋值）以及四个 1200 条样本的扩展数据集来训练与评估。

**📈 对比分析**

与 COMSOL、2‑D FEM 以及 DeepONet 进行对比；WarPGNN 推理时间分别比 COMSOL 速 119k×、比 2‑D FEM 速 206×，精度仅 1.26% RMSE、2.21% 翘曲误差；与 DeepONet 训练时间缩短 70% 并保持相近精度。

**⚠️ 局限性**

仍对极端长尾分布的误差略大，迁移性能在参数漂移时误差上升至 3.69%；主要适用于早期设计阶段，需与高精度仿真验证；公开数据有限，部分对照实验缺失。

---

## 84. ReDAG-RT: Global Rate-Priority Scheduling for Real-Time Multi-DAG Execution in ROS 2

**arXiv ID:** 2603.18238 | [PDF](https://arxiv.org/pdf/2603.18238v1)

**作者:** Md. Mehedi Hasan `[一作]`, Ziaur Rahman `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

实现了一套名为 ReDAG^RT 的用户空间全局调度框架，使 ROS 2 在不修改 API、执行器或内核的前提下实现多 DAG 的确定性实时执行。

**💡 创新点**

核心创新在于将所有回调统一映射到一个基于激活周期的 Rate‑Monotonic 级别的全局就绪队列，并引入每个 DAG 的并发上限与受控进入机制，既消除了跨 DAG 的优先级反转，又可在理论上进行可调度性分析。

**🔧 技术方法**

采用了 Rate‑Monotonic 固定优先级调度、预emptive 全局就绪队列、基于 DAG 的任务模型、响应时间递推分析以及运行时统计监控。

**📊 数据集**

使用在 ROS 2 Humble 环境下构造的合成多 DAG 工作负载（包含谐波/非谐波周期、混合周期与混合关键性配置），未使用真实机器人数据集。

**📈 对比分析**

通过在同一硬件平台上对比默认的 SingleThreadedExecutor 和 MultiThreadedExecutor，改变线程数（4–10）与截止时间缩放因子（0.8–1.2）进行实验；ReDAG^RT 在最佳配置下可将组合截止误差率降低 29.7%、将 99% 分位响应时间压缩 42.9%，相较于 MultiThreadedExecutor 的 13.7% 的误差率提升。

**⚠️ 局限性**

局限性包括：在线程数超过 8 时由于共享队列锁竞争导致性能饱和；实验仅基于合成负载，缺乏真实机器人系统的验证；并未在内核层面提供硬实时保证，仍依赖 Linux 调度器的非实时属性。

---

## 85. How LLMs Distort Our Written Language

**arXiv ID:** 2603.18161 | [PDF](https://arxiv.org/pdf/2603.18161v1)

**作者:** Marwa Abdulhai `[一作]` (University of California Berkeley), Natasha Jaques `[通讯]` (Google DeepMind)

**通讯引用:** 3469 | [OpenAlex ID](https://openalex.org/A5046953322)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过三项实验（用户写作实验、ArgRewrite‑v2 文本重写对比、ICLR 2026 论文评审数据）系统评估了大型语言模型（LLM）对人类写作语义、语调、风格及立场的影响，发现 LLM 在编辑时往往产生方向一致的大幅语义位移，导致文本趋同并偏离原作者意图；使用 LLM 辅助写作会削弱作者的创造力与个性化表达，并使论证立场转向中立；在学术评审中，LLM 生成的评审更关注可重复性与可扩展性，而非清晰度与创新性。

**💡 创新点**

创新点在于：①首次量化 LLM 对人类写作语义位移的方向性与幅度，揭示了“算法单声道”导致的语义同质化；②通过对比人类编辑与 LLM 编辑，证实 LLM 在最小修改任务中亦可显著改变立场与情感色彩；③将 LLM‑as‑a‑Judge 方法用于评估立场、情感、逻辑等多维度特征，提供了从多角度评估文本差异的新范式；④在真实学术评审数据上验证 LLM 对评审标准的改变，为 LLM 在专业领域的潜在风险提供实证依据。

**🔧 技术方法**

主要技术手段包括：句子级别嵌入 + PCA 或 t‑SNE 进行语义空间可视化；Jensen‑Shannon Divergence 计算词汇分布差异；NRC 情感词典和 LIWC 词汇分析评估情感与语法特征；统计检验（t‑检验、比例 z‑检验）对比不同条件下的效果；LLM‑as‑a‑Judge（GPT‑4）自动化定性标签；以及使用 Pangram AI 等工具进行 LLM 文本检测。

**📊 数据集**

使用的数据集：①ArgRewrite‑v2（86 篇大学生论说文及其人类/LLM 重写稿）；②用户实验数据（100 名美国英语母语者的原始与 LLM 辅助写作文本）；③ICLR 2026 论文评审数据（18,000 条评审，9,000 条人类、9,000 条 LLM 生成）。

**📈 对比分析**

对比方法：分别测量人类编辑与 LLM 编辑在语义偏移（PCA 向量长度与方向）、词汇差异（JSD）、情感/情绪密度（NRC）、语法类别变化（POS 统计）及立场倾向（LLM‑as‑a‑Judge）等指标。结果显示：LLM 产生的语义位移均向同一方向且幅度显著大于人类（平均 PCA 位移 3–4 倍）；JSD 平均值从 0.25（人类）升至 0.55（LLM）；情感正负语调提升 30% 以上；立场中立率提升 68%；ICLR 评审中 LLM 的“清晰度”评价下降 32%，而“可重复性”与“可扩展性”评价上升 136%/84%。

**⚠️ 局限性**

局限性：①样本主要为英语、美国地区受试者，结果可能不适用于其他语言或文化；②仅测试了少数 LLM（GPT‑4o‑mini、gpt‑5‑mini 等），不同模型可能表现差异；③LLM‑as‑a‑Judge 的自动标签可能受模型偏差影响；④使用的检测工具（Pangram AI）可能有误报误判；⑤实验环境中 LLM 的使用方式与真实专业场景可能不完全一致，导致结果的外推性受限。

---

## 86. RadioDiff-FS: Physics-Informed Manifold Alignment in Few-Shot Diffusion Models for High-Fidelity Radio Map Construction

**arXiv ID:** 2603.18865 | [PDF](https://arxiv.org/pdf/2603.18865v1)

**作者:** Xiucheng Wang `[一作]` (Xidian University), Nan Cheng `[通讯]` (Xidian University)

**通讯引用:** 15028 | [OpenAlex ID](https://openalex.org/A5050651525)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种少样本扩散模型框架 RadioDiff-FS，通过在主路径数据上预训练并在少量多径样本上微调，实现高保真射频地图的构建。

**💡 创新点**

创新点在于将多径射频地图理论上分解为主路径与方向稀疏残差，证明跨域迁移为受限几何平移，并设计了方向一致性损失（DCL）在冻结特征空间中约束生成器，显著提升少样本适配的稳定性与物理一致性。

**🔧 技术方法**

使用的技术包括物理信息驱动的扩散模型、主路径/多径分解理论、方向一致性损失、SNR加权的微调策略、LoRA参数高效微调以及可变噪声调度的扩散过程。

**📊 数据集**

使用的主要数据集是 RadioMapSeer 的 IRT4 多径子集（1402 个样本，500/200 场景划分），涵盖静态 RM（SRM）和动态 RM（DRM）两种场景。

**📈 对比分析**

与 RadioUNet、PhyRMDM、RME-GAN、RadioDiff 等基线进行对比，在静态 RM 上 NMSE 下降 59.5%（从 0.0121 到 0.0049），PSNR 提升至 36.37 dB，SSIM 达到 0.9752；在动态 RM 上 NMSE 下降 73.98%（从 0.0465 到 0.0121），PSNR 提升至 32.68 dB，均在少样本设置下显著优于所有基线。

**⚠️ 局限性**

局限性包括：仅在模拟多径子集上验证，缺乏真实测量数据的验证；目前仅处理单频单天线场景，未来需扩展到多频、多天线及更复杂动态环境；依赖固定的特征编码器，特征表达的选择可能限制了适配的通用性。

---

## 87. Exploring the Role of Interaction Data to Empower End-User Decision-Making In UI Personalization

**arXiv ID:** 2603.19196 | [PDF](https://arxiv.org/pdf/2603.19196v1)

**作者:** Sérgio Alves `[一作]` (Universidade de Lisboa), Tiago Guerreiro `[通讯]` (Universidade de Lisboa)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对12名参与者的访谈和设计探针（虚构情境和热图等可视化），研究了让终端用户访问自身交互数据以支持UI个性化的可行性与影响。

**💡 创新点**

提出了“反思式个性化”框架，结合交互数据可视化、文字与视觉建议，并强调用户对数据的可见性与控制，以提升用户对个性化的主动性与信任。

**🔧 技术方法**

采用半结构化访谈、设计探针（vignettes）、数据可视化仪表盘（热图、统计表）以及文本和视觉个性化建议；后期对访谈内容进行主题分析。

**📊 数据集**

使用合成的点击流和滚动热图数据，涵盖两类应用：一是所有参与者共享的Lidl.com网站，二是根据每位参与者自选的常用App（如YouTube）生成的个性化数据；未使用真实用户数据。

**📈 对比分析**

研究未涉及算法性能对比或量化评估；通过定性访谈反馈，发现参与者能独立识别个性化机会，并倾向于接受视觉建议；性能评价基于参与者满意度和使用意愿的主观报告。

**⚠️ 局限性**

主要限制在于：①使用合成数据与虚构人物降低生态有效性；②样本规模有限，且缺乏多样性；③探针场景覆盖面窄，未检验真实环境中的个性化行为；未来需要在真实系统中进行实地验证。

---

## 88. Mind the Rarities: Can Rare Skin Diseases Be Reliably Diagnosed via Diagnostic Reasoning?

**arXiv ID:** 2603.18418 | [PDF](https://arxiv.org/pdf/2603.18418v1)

**作者:** Yang Liu `[一作]` (Carnegie Mellon University), Min Xu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12668 | [OpenAlex ID](https://openalex.org/A5100413849)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建DermCase数据集，评估并比较22种大型视觉语言模型在罕见皮肤病诊断推理上的表现

**💡 创新点**

首次提供长上下文罕见皮肤病案例，包含完整的多模态图文对和逐步推理链，并设计基于DermLIP的差异诊断相似度度量

**🔧 技术方法**

使用多模态预训练模型（如LLaVA、InternVL、Qwen、MedGemma）以及SFT、DPO、MPO等微调技术进行评估

**📊 数据集**

DermCase（26,030个图文对、6,354个罕见病例）以及DermLIP预训练模型作为评测基础

**📈 对比分析**

对比模型在最终诊断准确率、差异诊断覆盖率和推理质量方面的表现，发现域特定模型（MedGemma）在诊断和推理上领先，但整体准确率仍低；SFT显著提升准确率，DPO提升有限

**⚠️ 局限性**

模型在视觉解读、逻辑一致性、推理完整性和医学知识方面仍存在显著缺陷，尤其是视觉与文本对齐和推理连贯性，DPO和MPO在罕见病例上的适用性不足

---

## 89. Pushan: Trace-Free Deobfuscation of Virtualization-Obfuscated Binaries

**arXiv ID:** 2603.18355 | [PDF](https://arxiv.org/pdf/2603.18355v1)

**作者:** Ashwin Sudhir `[一作]` (Arizona State University), Ruoyu Wang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出了一种可追踪的、无路径约束的虚拟化反混淆框架（Pushan），实现了对使用虚拟化技术保护的二进制文件的完整CFG恢复和高质量C伪代码反编译。

**💡 创新点**

核心创新在于引入VPC（虚拟程序计数器）敏感的无约束符号仿真，摆脱传统路径可满足性检查，避免路径爆炸，并通过语义保持简化实现完整CFG的恢复。

**🔧 技术方法**

技术包括：VPC识别启发式、无约束符号仿真、约束自由表达式简化、符号化（Symbolization）以扩展边，语义保持简化（多轮消除、死代码、栈变量、针对特定虚拟化器的简化），以及定制的二进制反编译器。

**📊 数据集**

评估使用了1,000个Tigress生成的虚拟化程序、多种商业虚拟化器（VMProtect、Themida）生成的样本、六个公开恶意样本、四个开源项目、四个人工合成样本以及五个CTF自定义虚拟机挑战。

**📈 对比分析**

与现有工具（Yadegari、Salwan等）比较，Pushan在CFG相似度（改进的GED）上达到99%+的匹配率，在大多数样本上实现100%相似CFG；在Tigress数据集上成功恢复988/1000 CFG，速度平均每个样本约4-8小时；对商业保护样本表现出显著优势，生成可直接用于LLM简化的C伪代码。

**⚠️ 局限性**

局限性包括：无法处理复杂的MBA表达式，需外部MBA简化工具；对极其恶意的控制流伪造（例如误导符号化）仍可能产生错误分支；符号求解器仍会在某些表达式上耗时，导致大文件分析时间较长；需要手动指定自定义VM的VPC位置。

---

## 90. GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting

**arXiv ID:** 2603.18912 | [PDF](https://arxiv.org/pdf/2603.18912v1)

**作者:** Ahmed Tawfik Aboukhadra `[一作]` (RPTU), Didier Stricker `[通讯]` (RPTU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出GHOST框架，利用2D高斯喷射从单目RGB视频快速实现类别无关的手-物体交互三维重建；

**💡 创新点**

三大创新：①几何先验检索与一致性损失填补被遮挡的物体；②抓握感知对齐优化手部平移与物体尺度；③手部感知背景损失避免误删被遮挡物体区域；

**🔧 技术方法**

核心技术包括高斯喷射、对象几何先验检索（Objaverse+CLIP）、抓握检测、手部重建（HaMeR）与手部高斯骨架化；

**📊 数据集**

使用ARCTIC、HO3D以及自录的实景视频数据集进行训练与评估；

**📈 对比分析**

与现有NeRF/BigS、HOLD等方法对比，GHOST在3D交互指标（CD_h、MPJPE）与2D渲染质量（PSNR、SSIM、LPIPS）上均取得SOTA表现，并在单序列运行时间上比前沿方法快13倍；

**⚠️ 局限性**

局限性：对极端遮挡下的几何先验依赖较高；仅处理刚性物体，无法直接扩展到可变形物体；需要手部与物体的分离检测与标注，若遮挡严重或光照变化大仍会影响重建质量。

---

## 91. Functional Subspace Watermarking for Large Language Models

**arXiv ID:** 2603.18793 | [PDF](https://arxiv.org/pdf/2603.18793v1)

**作者:** Zikang Ding `[一作]` (University of Electronic Science and Technology of China), Lijie Hu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5067496051)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Functional Subspace Watermarking（FSW）框架，用于在大型语言模型（LLM）的功能子空间中嵌入鲁棒的水印信号。

**💡 创新点**

创新点在于通过求解广义特征值问题（GEVP）识别任务关键且对压缩不变的低维功能子空间，并引入自适应谱截断与向量一致性约束，使水印在微调、量化和蒸馏等参数级攻击后仍可可靠提取。

**🔧 技术方法**

技术主要包括 Fisher 信息矩阵估计、压缩不变矩阵构造、GEVP 计算、谱截断选择、向量一致性约束、基于正交密钥的多比特水印嵌入与统计检验。

**📊 数据集**

实验使用 LLaMA‑2‑7B、Meta‑LLaMA‑3‑8B、Qwen2.5‑7B、Mistral‑7B‑v0.3、DeepSeek‑LLM‑7B‑Chat 等多种 LLM，数据集包括 WikiText‑2（用于矩阵估计）、C4（用于水印微调）、HellaSwag、ARC 等下游任务集。

**📈 对比分析**

与 Clean FT、EmMark、Weighted Quantization、Naive Top‑k 等基线相比，FSW 在保持模型功能（PPL、下游准确率）几乎不变的前提下，攻击后检测分数保持在 6–8 以上，位误码率接近 0%，并在量化、剪枝、LoRA、蒸馏等攻击中表现出最高的鲁棒性和最小的误报率。

**⚠️ 局限性**

限制主要包括：只能在保持功能子空间不变的攻击场景下有效，对完全自由的蒸馏或重参数化攻击鲁棒性有限；可嵌入的位数受限（一般 ≤16 位），过大会导致模型性能显著下降；同时训练过程在水印嵌入阶段消耗的 GPU 资源和时间相对较高。

---

## 92. Benchmarking CNN-based Models against Transformer-based Models for Abdominal Multi-Organ Segmentation on the RATIC Dataset

**arXiv ID:** 2603.18616 | [PDF](https://arxiv.org/pdf/2603.18616v1)

**作者:** Lukas Bayer `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Andreas Maier `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 15913 | [OpenAlex ID](https://openalex.org/A5101619735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RATIC多机构CT数据集上，系统性对比CNN模型SegResNet与三种Transformer混合模型（UNETR、SwinUNETR、UNETR++）进行腹部多器官分割的基准实验。

**💡 创新点**

通过独立、统一的训练和评估条件，揭示Transformer模型在中等规模异构数据集上的真实性能，并对比其与精调CNN的差距。

**🔧 技术方法**

使用MONAI+PyTorch框架实现并训练UNETR、SwinUNETR、UNETR++与SegResNet，采用Dice相似系数作为评估指标。

**📊 数据集**

RATIC数据集：206个CT扫描，覆盖5个腹部器官（肝、脾、左肾、右肾、肠），来自23家机构。

**📈 对比分析**

在相同预处理、数据增强和训练超参数下评估，SegResNet取得最高平均Dice（0.945），UNETR++次之（0.934），SwinUNETR表现最差（0.829）。

**⚠️ 局限性**

局限在于样本量相对有限、模型参数规模较大导致训练资源占用高、仅评估5个器官且缺乏跨域泛化验证。

---

## 93. Book your room in the Turing Hotel! A symmetric and distributed Turing Test with multiple AIs and humans

**arXiv ID:** 2603.18981 | [PDF](https://arxiv.org/pdf/2603.18981v1)

**作者:** Christian Di Maio `[一作]` (University of Siena), Vincenzo Lomonaco `[通讯]` (LUISS University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在去中心化平台 UNaIVERSE 上实现了 TuringHotel——一种多方、持续、隐私保护的 Turing 测试实验，邀请人类与多种 LLM 共同参与短时段的群聊并互相识别对方身份。

**💡 创新点**

创新点在于：① 将经典一对一 Turing 测试扩展为多方社交对话；② 采用 P2P 网络与分布式身份机制，消除第三方数据干预；③ 允许参与者长期重复加入，实现对模型演化的持续跟踪；④ 同时让人类和 AI 既是观察者又是参与者，保持实验对称性。

**🔧 技术方法**

使用了 UNaIVERSE 的 World 与 Agent 角色系统、基于 FSA 的交互协议、Python API 调用多种 HuggingFace LLM（Qwen2.5‑14B‑Instruct、gpt‑oss‑20b、Llama‑3.1‑8B 等），以及自定义的提示工程和多轮语义对话框架。

**📊 数据集**

数据集为 17 名分布式人类参与者（通过 web 界面）与 19 个 LLM 代理组成的 3‑分钟群聊记录，总共 8 人的混合房间；实验数据包括对话文本、消息长度、拼写错误、以及身份识别结果。

**📈 对比分析**

比较方法：将人类和 AI 的识别准确率、精确率、召回率进行统计；实验结果显示人类识别准确率 0.721，精确率 0.500，召回率 0.658；AI 识别准确率仅 0.469，精确率低，说明当前预训练 LLM 在多方识别任务中表现不佳。

**⚠️ 局限性**

局限性包括：样本量小（仅 17 人、19 机），对话时长短，未进行模型微调或对齐，可能导致实验偏倚；此外，实验主要聚焦无主题、同步聊天，未覆盖异步或主题化对话的可推广性。

---

## 94. RPiAE: A Representation-Pivoted Autoencoder Enhancing Both Image Generation and Editing

**arXiv ID:** 2603.19206 | [PDF](https://arxiv.org/pdf/2603.19206v1)

**作者:** Yue Gong `[一作]` (Beihang University), Lijun Zhang `[通讯]` (Beihang University)

**通讯引用:** 13888 | [OpenAlex ID](https://openalex.org/A5115603660)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种表示锚定的自动编码器（RPiAE），实现了在图像生成与编辑任务中同时兼顾高重建质量与生成友好潜在空间的目标。

**💡 创新点**

创新点在于通过pivot regularization允许预训练视觉表示模型的编码器可微调以提升重建性能，同时通过KL正则化的Variational Bridge压缩高维特征为低维生成友好潜在空间，二者结合解决了传统tokenizer在重建与生成之间的矛盾。

**🔧 技术方法**

采用的技术包括预训练的DINOv2视觉表示编码器、ViT解码器、Transformer结构的Variational Bridge、pivot regularization、GAN+perceptual loss以及分阶段（stage-wise）训练策略。

**📊 数据集**

在ImageNet-1K、CC12M-LLaVA-Next以及OmniEdit等公开数据集上进行训练与评估。

**📈 对比分析**

通过与RAE、VA-VAE、FAE等现有视觉分词器以及LightningDiT、Bagel-MoT等架构下的文本到图像模型进行比较，RPiAE在rFID、gFID、GenEval、DPG-Bench等指标上取得更低的FID、更高的生成质量与编辑性能，且收敛速度更快。

**⚠️ 局限性**

局限性包括对预训练表示模型的强依赖、潜在空间维度需要在重建与生成之间精细平衡，以及在极高分辨率或长文本条件下的泛化性能仍待进一步验证。

---

## 95. Comparative Analysis of Large Language Models in Generating Telugu Responses for Maternal Health Queries

**arXiv ID:** 2603.18898 | [PDF](https://arxiv.org/pdf/2603.18898v1)

**作者:** Anagani Bhanusree `[一作]` (National Institute of Technology), Rimjhim `[通讯]` (National Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估ChatGPT-4o、Gemini、Perplexity等LLM在低资源语言（Telugu）孕产妇健康问答中的生成效果，结合BERTScore与10位临床专家的定性评分。

**💡 创新点**

首次系统比较多种LLM在区域语言孕产健康问答中的表现，结合语义相似度和专家评估，揭示输入语言对输出质量的显著影响。

**🔧 技术方法**

使用BERTScore进行语义相似度量化，采用医生评估的五项指标（准确性、流畅度、相关性、连贯性、完整性）进行定性评估。

**📊 数据集**

双语孕产妇健康问答数据集（英文+Telugu），包含专家医生编写的参考答案。

**📈 对比分析**

通过BERTScore与专家评估的综合指标对比，Gemini在所有指标上表现最优，Perplexity在Telugu提示下提升明显，ChatGPT表现中等；BERTScore最高F1为0.704。

**⚠️ 局限性**

仍受低资源语言训练数据不足限制，BERTScore无法完全捕捉临床细节，样本量和语言覆盖有限，缺乏真实临床部署与用户信任评估。

---

## 96. PRIOR: Perceptive Learning for Humanoid Locomotion with Reference Gait Priors

**arXiv ID:** 2603.18979 | [PDF](https://arxiv.org/pdf/2603.18979v1)

**作者:** Chenxi Han `[一作]` (Tsinghua University), Houde Liu `[通讯]` (Tsinghua University)

**通讯引用:** 1687 | [OpenAlex ID](https://openalex.org/A5076885280)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出PRIOR框架，结合深度感知、参数化步态生成和自监督状态估计，实现单阶段强化学习下的自然且鲁棒的类人步态。

**💡 创新点**

用参数化步态生成代替对抗式步态约束；利用GRU自监督重建高度图获取地形特征；通过地形自适应脚步奖励实现安全落脚。

**🔧 技术方法**

Isaac Lab仿真、PPO强化学习、异步actor‑critic、GRU状态估计、低分辨率深度感知与自监督重建、速度驱动的步态插值。

**📊 数据集**

采集的人类运动捕捉数据并重定位至ZERITH Z1机器人，用于生成参数化步态；训练时使用合成深度图与随机生成的复杂地形。

**📈 对比分析**

对照多种去掉关键组件的ablation模型，完整框架在四种地形上实现100%通过率，平均奖励最高，展示各模块协同提升。

**⚠️ 局限性**

仅在仿真中验证，未完成真实机器人部署，尚缺乏针对实机的自适应动力学和sim‑to‑real迁移方法。

---

## 97. Mi:dm K 2.5 Pro

**arXiv ID:** 2603.18788 | [PDF](https://arxiv.org/pdf/2603.18788v1)

**作者:** KT Tech innovation Group `[一作]` `[通讯]`, KT Tech innovation Group

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Mi:dm K 2.5 Pro 32B 大模型，并通过多阶段训练（预训练、CPT、SFT、RL、Fusion）实现推理、长上下文、工具使用等多重能力。

**💡 创新点**

创新点包括：① 采用层预测深度扩展（Layer‑Predictor DuS）快速扩容到 32B；② 128K 令牌长上下文逐步扩展策略；③ 结合 AST 代码筛选、数学 Gap‑Fill 合成、LLM‑评估的质量自适应数据管道；④ 多任务多语言（韩、英、日、中文）响应风格统一；⑤ 异步 RL、LLM‑as‑a‑Judge、难度感知数据挑选等高效对齐技术。

**🔧 技术方法**

核心技术：深度扩容（DuS）+层预测；RegMix 数据混合；自监督预训练 + 理解与推理混合；SFT‑Merge、RL‑VR、Direct Preference Optimization；LLM‑评估奖励；异步 GSPO；多通道（analysis / answer / tool）聊天模板。

**📊 数据集**

数据集主要来源：① 韩国公共资源（AI‑Hub、NIKL、K‑Data Alliance）；② 多语言公开与商业授权数据（英、日、中）；③ 代码与数学专项公开数据（peS2o、OpenStax、MBPP、HumanEval+）和自研合成；④ Korean‑specific 语料（文化、法律、财经、教育、代码、工具使用场景）和多轮对话/工具交互数据。

**📈 对比分析**

对比方法：在公开英文基准（GPQA‑D、MMLU‑Pro、MATH‑H、AIME25、IFEval、IFBench、HumanEval+、MBPP+、Terminal‑Bench、τ²‑Bench Telecom）和韩语专用基准中，Mi:dm K 2.5 Pro 与同参数规模模型（Qwen‑3‑30B、Solar‑Open‑100B、K‑EXAONE‑236B）相当，且在推理、知识、数学、工具使用等指标上往往排名第一或前列；在韩语文化理解、敬语使用、社会语境推理方面刷新了国内最佳成绩。

**⚠️ 局限性**

局限性：① 尽管推理与长上下文表现优异，但在极端高难度数学/代码任务仍略低于更大规模模型；② 语言多样性虽支持四语，但对非标准方言或俚语的覆盖有限；③ 模型仍存在少量 hallucination、工具调用错误；④ 训练成本高，尤其是多阶段大规模数据与异步 RL；⑤ 部分安全与道德评估结果仍需进一步完善。

---

## 98. Automatic detection of Gen-AI texts: A comparative framework of neural models

**arXiv ID:** 2603.18750 | [PDF](https://arxiv.org/pdf/2603.18750v1)

**作者:** Cristian Buttaro `[一作]` (Sapienza University), Irene Amerini `[通讯]` (Sapienza University)

**通讯引用:** 3539 | [OpenAlex ID](https://openalex.org/A5030121038)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建并评估了四种监督神经网络模型（MLP、1D CNN、MobileNet 1D CNN、Transformer）用于AI生成文本检测，并与主流在线检测工具在同一评测框架下进行对比。

**💡 创新点**

创新点在于提供了统一、可比的检测框架，系统考察多语言（英语、意大利）和多领域（通用、艺术与精神健康）环境下的鲁棒性，并揭示了单类与跨域测试中模型与商业工具的性能差异。

**🔧 技术方法**

使用的技术包括词向量嵌入、全连接层、卷积层、深度可分离卷积、Transformer自注意力、全局池化、Dropout、标签平滑、权重衰减和阈值校准等。

**📊 数据集**

采用的数据集有：COLING 2025 Multilingual（dtEN、dtITA子集）和自制主题数据集ART&MH（艺术与精神健康）。

**📈 对比分析**

比较方法：在相同的数据预处理与评测流程下训练、验证并测试模型，使用准确率、Human/GenAI召回率等指标；实验结果显示MobileNet CNN在平衡性能上最优，Transformer与MLP更倾向于降低误报，商业工具普遍偏向于误报抑制；单类与跨域测试显著揭示了鲁棒性差距。

**⚠️ 局限性**

局限性包括：测试样本量有限（每组仅60条），未覆盖更多语言与更大规模数据；阈值与校准策略主要手工设定；商业工具为黑盒，缺乏可解释性；未探索混合或集成策略以进一步提升鲁棒性。

---

## 99. Graph-of-Constraints Model Predictive Control for Reactive Multi-agent Task and Motion Planning

**arXiv ID:** 2603.18400 | [PDF](https://arxiv.org/pdf/2603.18400v1)

**作者:** Anastasios Manganaris `[一作]` (Purdue University), Suresh Jagannathan `[通讯]` (Purdue University)

**通讯引用:** 5901 | [OpenAlex ID](https://openalex.org/A5034957233)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于图约束（Graph-of-Constraints, GoC）与模型预测控制（MPC）相结合的多机器人任务-运动规划框架GoC‑MPC，能够在视觉观测下实时执行部分有序、可动态分配的任务。

**💡 创新点**

创新点包括：①将传统的序列约束推广为DAG图约束，天然支持部分排序和并行子任务；②在图约束下引入动态代理分配矩阵，使得规划时可同时优化任务分配；③将上述图约束嵌入MPC框架，分解为航点+分配、时间/速度、短期路径三个子问题，实现实时重规划和扰动恢复。

**🔧 技术方法**

使用技术包括：混合整数非线性规划（MIP）+大M门控；MPC与短期二次规划（QP）结合；MOSEK、Ipopt求解器；视觉关键点跟踪（RealSense D455、SAM2、Kanade–Lucas–Tomasi）与实时约束更新；离线任务骨架的图构建。

**📊 数据集**

实验数据来源：IsaacSim + OmniGibson 与 Drake 进行仿真；真实实验使用双UR5e 机器人，RGB‑D 采集关键点；未使用公开数据集，而是自行搭建三种任务（堆块、倒杯、折桌布）进行评测。

**📈 对比分析**

对比基线为 ReKep，评价指标包括成功率、最大/平均计算时间、路径长度。GoC‑MPC 在静态和扰动环境下均实现 100% 成功率，平均计算时间比 ReKep 快 40–70 倍，路径长度短 30–50%。在可扩展性评测中，随着对象/机器人数量增加，平均时间与路径长度上升但仍保持在线可行；在真实世界实验中，成功率达 3/5（堆块）与 5/5（倒杯、折布）并保持低计算时间。

**⚠️ 局限性**

局限性：①对视觉状态估计的准确性依赖较高，严重遮挡或跟踪误差会导致失败；②初始任务图需离线生成或人工设计，图规模过大时航点分配子问题的 MIP 求解可能耗尽迭代；③未结合学习感知模块或离线离散规划，可能在更复杂场景中缺乏鲁棒性。

---

## 100. Controllable Evidence Selection in Retrieval-Augmented Question Answering via Deterministic Utility Gating

**arXiv ID:** 2603.18011 | [PDF](https://arxiv.org/pdf/2603.18011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 101. Negative Sampling Techniques in Information Retrieval: A Survey

**arXiv ID:** 2603.18005 | [PDF](https://arxiv.org/pdf/2603.18005v1)

**作者:** Laurin Wischounig `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对密集检索中的负采样技术进行系统综述，构建了完整的分类法并在MS MARCO、Natural Questions、BEIR与MTEB等主流基准上做了元分析；

**💡 创新点**

首次提出面向现代NLP应用的负采样技术全景图，特别加入了LLM驱动的合成数据与去噪策略，并通过多维度指标展示技术演进与效果提升；

**🔧 技术方法**

汇总了随机采样、静态/动态硬负采样、聚类采样、TriSampler、假负样本消除、数据增强、LLM合成数据等方法；

**📊 数据集**

使用的主要数据集包括MS MARCO（段落检索）、Natural Questions（开放域问答）、BEIR（跨域零样本检索）以及MTEB（多任务嵌入基准）；

**📈 对比分析**

采用标准检索指标（MRR@10、NDCG@k、Recall@k）进行对比，结果显示从随机负样本到多模态组合可实现50%+性能提升，顶尖模型如NV‑Embed‑v2、Gemini Embeddings在MTEB上达到了69+分；

**⚠️ 局限性**

局限性包括快速变化的技术生态导致部分最新方法未被覆盖，研究仅聚焦于密集检索的对比学习，缺乏对其他架构（如ColBERT、知识蒸馏）的深入探讨，且成本与实现细节仍以概念层面呈现，未提供完整的实验复现。

---

## 102. OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation

**arXiv ID:** 2603.19201 | [PDF](https://arxiv.org/pdf/2603.19201v1)

**作者:** Yuhang Zheng `[一作]` (National University of Singapore), Wenchao Ding `[通讯]` (Fudan University)

**通讯引用:** 3698 | [OpenAlex ID](https://openalex.org/A5102769588)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 OmniViTac 数据集和 OmniVTA 框架，用于实现基于视觉与触觉的接触丰富机器人操作。

**💡 创新点**

创新点包括：21k+ 轨迹的大规模 visuo‑tactile 数据集、基于世界模型的触觉预测与动态融合、以及高频触觉闭环控制。

**🔧 技术方法**

采用自监督触觉 VAE、双流 visuo‑tactile 世界模型、预测差分编码器、门控融合、扩散策略和反射式 Latent 触觉控制器等技术。

**📊 数据集**

使用 OmniViTac 数据集，该数据集涵盖 86 个任务、100+ 物体，分为六种物理交互模式，跨平台收集。

**📈 对比分析**

与 DP、RDP、ForceMimic 等基线对比，OmniVTA 在所有六类任务中获得最高成功率，闭环控制显著提升鲁棒性并保持对未见物体、位置和工具的良好泛化。

**⚠️ 局限性**

局限性在于依赖高频触觉硬件、训练需要大规模 GPU 资源，对极端环境或柔性物体动态的适应性尚待验证。

---

## 103. HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships

**arXiv ID:** 2603.18459 | [PDF](https://arxiv.org/pdf/2603.18459v1)

**作者:** Xiangxu Zhang `[一作]` (Renmin University of China), Jianxun Lian `[通讯]` (Microsoft Research Asia)

**通讯引用:** 3184 | [OpenAlex ID](https://openalex.org/A5087106517)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种两阶段基于超图的框架 HypeMed，用于提升药物推荐的准确性和安全性。

**💡 创新点**

创新点在于将临床就诊视为超边进行知识感知对比预训练，构建全局一致且检索友好的嵌入空间，并在该空间内动态检索历史病例以增强药物预测。

**🔧 技术方法**

采用超图模型、知识感知对比学习、检索增强推荐等技术。

**📊 数据集**

在真实世界的医学记录基准数据集上进行实验（具体数据集未在摘要中列出）。

**📈 对比分析**

与多种先进基线对比，HypeMed 在药物推荐精度和药物相互作用（DDI）减少方面均表现出显著提升。

**⚠️ 局限性**

局限性包括对知识图谱的依赖、超图构造与检索过程的计算开销，以及在数据稀疏或不完整场景下的泛化能力待进一步验证。

---

## 104. Leveraging Large Language Models for Generalizing Peephole Optimizations

**arXiv ID:** 2603.18477 | [PDF](https://arxiv.org/pdf/2603.18477v1)

**作者:** Chunhao Liao `[一作]` (University of Waterloo), Chengnian Sun `[通讯]` (University of Waterloo)

**通讯引用:** 12305 | [OpenAlex ID](https://openalex.org/A5101708632)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的框架，利用大型语言模型来推广peephole优化，旨在通过自动化的方式生成更通用的优化规则。

**💡 创新点**

创新点在于结合了大型语言模型的语义抽象能力和严格的验证过程，以确保生成的优化规则的正确性和有效性。

**🔧 技术方法**

使用了大型语言模型（如GPT）进行符号常量推广、结构推广、约束放宽和位宽/精度推广等多阶段的优化过程。

**📊 数据集**

使用了来自生态系统的真实世界peephole优化实例，特别是从GitHub问题中挖掘的未优化实例。

**📈 对比分析**

与现有的基于程序合成的方法进行比较，结果显示该框架在整数、浮点和内存优化领域成功推广了大量优化实例，并且在整数相关优化中表现出更高的成功率和更广泛的适用性。

**⚠️ 局限性**

限制在于尽管该方法在许多情况下表现良好，但仍然可能在某些复杂的浮点优化和内存语义方面面临挑战，尤其是在验证支持不足的情况下。

---

## 105. Do Vision Language Models Understand Human Engagement in Games?

**arXiv ID:** 2603.18480 | [PDF](https://arxiv.org/pdf/2603.18480v1)

**作者:** Ziyi Wang `[一作]` (University of Maryland), Xiyang Hu `[通讯]` (Arizona State University)

**通讯引用:** 831 | [OpenAlex ID](https://openalex.org/A5044665455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估了预训练视觉语言模型在游戏视频中预测玩家参与度的能力，构建了系统性的实验框架；

**💡 创新点**

首次将理论指导提示、跨游戏检索增强与多模型评估相结合，揭示VLM在情感推理中的感知‑理解鸿沟；

**🔧 技术方法**

使用InternVL3.5‑8B‑Instruct、Qwen3‑VL‑8B‑Instruct及GPT‑4o三种VLM，配合零/少量样本提示、理论指导提示和检索增强技术；

**📊 数据集**

基于GameVibe Few‑Shot (GVFS) 数据集，涵盖9款FPS游戏，每款约61秒的视频；

**📈 对比分析**

对六种提示策略和三模型进行比较，零样本准确率约57%低于多数基线，少样本提升至最高75%，理论提示效果不显著，检索增强对某模型有一定帮助；总体表现仍远低于人类标签；

**⚠️ 局限性**

局限包括数据量有限、仅包含FPS游戏、二分类简化、检索示例标签不平衡、未对模型进行微调，导致难以捕捉细粒度情绪与时序一致性。

---

## 106. AIMER: Calibration-Free Task-Agnostic MoE Pruning

**arXiv ID:** 2603.18492 | [PDF](https://arxiv.org/pdf/2603.18492v1)

**作者:** Zongfang Liu `[一作]` (Zhejiang University), Xin Yuan `[通讯]` (Westlake University)

**通讯引用:** 13925 | [OpenAlex ID](https://openalex.org/A5015431603)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无校准的专家剪枝准则 AIMER，能够仅凭预训练权重对 Mixture-of-Experts 语言模型中的专家进行排序并裁剪。

**💡 创新点**

创新点在于使用绝对均值除以根均方的比例作为专家重要性评分，既简单又无须校准集，显著提升了层内专家分层性和剪枝效果。

**🔧 技术方法**

利用模型权重的统计量计算 AIMER 分数，对 7B–30B 规模的 MoE LLM 进行实验，并与传统校准方法进行对比。

**📊 数据集**

使用 C4 4.2M tokens 作为对照基准的校准集，评测覆盖 16 个零样本基准（代码、创作、数学、选择题等）。

**📈 对比分析**

与 REAP、SEER、EAN、Frequency 等四种校准方法以及随机和绝对值方法比较，AIMER 在绝大多数基准上表现优于或与校准方法持平，且专家评分时间从数小时降至 0.22–1.27 秒。

**⚠️ 局限性**

限制包括：仅适用于任务无关剪枝；未在百亿级别模型上验证；缺乏理论证明，且任务特定剪枝仍需依赖校准集。

---

## 107. An Optimised Greedy-Weighted Ensemble Framework for Financial Loan Default Prediction

**arXiv ID:** 2603.18927 | [PDF](https://arxiv.org/pdf/2603.18927v1)

**作者:** Ezekiel Nii Noye Nortey `[一作]` (University of Ghana), Ravenhill Adjetey Laryea `[通讯]` (University of Professional Studies)

**通讯引用:** 31 | [OpenAlex ID](https://openalex.org/A5033501536)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于粒子群优化调参、正则化贪婪权重分配和BlendNet元学习器的动态加权集成框架，用于预测 Lending Club 贷款违约。

**💡 创新点**

创新点包括：① 在 PSO 优化后再独立进行权重分配，使权重基于已调优模型的真实性能动态更新；② 引入正则化的贪婪权重算法，兼顾性能提升与防止过拟合；③ 在堆叠架构中加入神经网络元学习器，进一步校正概率并捕捉高阶特征关系。

**🔧 技术方法**

技术栈涵盖：多种基分类器（GB、MLP、SVM、KNN、LR、ExtraTrees）、粒子群优化（PSO）、正则化贪婪权重优化、BlendNet（全连接神经网络元学习器）、SMOTE 与成本敏感学习、递归特征消除（RFE）、异常值检测（BCP、IQR、Hampel）以及数据预处理与标准化。

**📊 数据集**

使用公开的 Lending Club 个人贷款数据集（约 396,030 条记录、28 个特征，违约比例约 19.7%）。

**📈 对比分析**

通过 80/20 训练/测试拆分、交叉验证与基线模型（单一分类器、传统静态加权集成）对比，BlendNet 集成在 AUC（0.80）、宏平均 F1（0.73）和违约召回率（0.81）上显著优于单模型和传统加权方法；树模型的 Brier 分数仅 0.18，显示出良好的概率校准。

**⚠️ 局限性**

局限性：① 权重更新基于离线滚动窗口，缺乏实时在线自适应能力；② 异常值处理在高维空间仍需改进；③ 在不同宏观经济周期或新型贷款产品上的迁移性能尚未充分验证。

---

## 108. Optimal Splitting of Language Models from Mixtures to Specialized Domains

**arXiv ID:** 2603.19149 | [PDF](https://arxiv.org/pdf/2603.19149v1)

**作者:** Skyler Seto `[一作]` (Apple), David Grangier `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究如何在多域预训练中划分计算预算，提出分裂训练（split model training）并推导其扩展定律，给出最优分裂点以提升零样本QA性能。

**💡 创新点**

① 在多域场景下首次系统地建模预训练与后续专门化的计算分配；② 提出可解释的扩展定律并证明其可用于预测最佳分裂点；③ 证明分裂训练在多域、不同规模和预算下可显著优于传统全预训练。

**🔧 技术方法**

使用 Transformer 架构（12.9M至6.7B参数），基于文档嵌入的 K‑means 聚类，路由器学习，分裂训练；采用 Chinchilla 风格的非线性加法式扩展定律；对比实验使用零样本 QA、语言建模与 perplexity 评估。

**📊 数据集**

多种公共数据集：DCLM、Pile、ArXiv、DM Math、FreeLaw、Github、PubMed Central；人工生成的电话簿数据用于快速实验；对 16 个聚类域进行实验，并在 4、16、64 等不同域数上评估。

**📈 对比分析**

与全预训练、无分裂、以及不同模型规模的基线比较；在 1.3B/2.7B 规模下，按最优分裂点训练可比 6.7B 全预训练的 1.5–2% QA 提升；在语言建模中相对 perplexity 提升约 9%。

**⚠️ 局限性**

仅在均匀分布或高度相似域下分裂优势有限；分裂训练需要额外的路由器与多模型推理，增加系统复杂度；目前实验聚焦静态聚类，未探索动态聚类或多专家模型的进一步融合。

---

## 109. Foundations and Architectures of Artificial Intelligence for Motor Insurance

**arXiv ID:** 2603.18508 | [PDF](https://arxiv.org/pdf/2603.18508v1)

**作者:** Teerapong Panboonyuen `[一作]` `[通讯]` (MARS Motor AI Recognition Solution), Teerapong Panboonyuen (MARS Motor AI Recognition Solution)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了面向汽车保险的端到端人工智能体系（MARS、MARSAIL、ALBERT 与 DOTA），实现了车辆损伤识别、结构部件分割、文档文本识别与后端风险评估的完整工业化流水线。

**💡 创新点**

创新点包括：① 基于二叉四叉树的 Mask Attention Refinement（MARS）实现高精细分割；② 通过双向 Transformer 对车辆部件与损伤类型进行联合建模（ALBERT），实现结构-损伤的语义关联；③ 采用知识蒸馏与参数高效微调（SLICK）将高容量模型压缩至实时推理；④ 针对汽车保险场景设计的大规模标注数据集（ALBERT‑DAMAGE 与 ALBERT‑PART）和文档识别模型（DOTA）实现跨模态融合与业务级落地。

**🔧 技术方法**

核心技术为：Transformer 自注意力、四叉树序列编码、动态卷积过滤、双向 GRU 语义序列化、类平衡焦点 CTC、可变形卷积、MLOps 管道与端到端微调，结合多模态融合与结构化输出（VDC）实现可信、可解释的决策。

**📊 数据集**

主要使用的数据集包括：1）ALBERT‑DAMAGE（856k 实例，26 细粒度损伤类）；2）ALBERT‑PART（595k 实例，61 细粒度车身部件）；3）泰国汽车损伤图像数据集；4）文档识别基准（IC15、SVT、IIIT5K、SVTP、CUTE80）用于验证 DOTA 的跨域性能。

**📈 对比分析**

与 Mask R‑CNN、PointRend、Mask Transfiner 等现有分割模型对比，MARS 在 COCO 评估中 AP 达 36.44，AP50 60.6，AP75 37.6，尤其在小目标 AP_s 21.8 与大目标 AP_l 49.5 上表现优异；ALBERT 在损伤模型 AP 36.44、部件模型 AP 62.32，且在所有 IoU 阈值（AP50:95）与多尺度指标上均优于基线；DOTA 在 IC15、SVT、IIIT5K 等公开数据集上实现最高识别准确率（IC15 58.26%），显著超越 ResNet‑ViT、Deformable‑ViT 等前沿 OCR 方案。

**⚠️ 局限性**

主要限制包括：① 仍依赖预定义的损伤与部件类别，难以自动扩展至新类别；② VIN 识别精度仅 50% 级别，需进一步提升；③ 对于非泰国本土场景（不同文字、照明、车辆款式）泛化能力尚待验证；④ 模型规模与推理速度在极低算力边缘设备上仍需压缩；⑤ 业务流程中多模态融合与决策解释仍处于后期迭代阶段。

---

## 110. Large-Scale Analysis of Political Propaganda on Moltbook

**arXiv ID:** 2603.18349 | [PDF](https://arxiv.org/pdf/2603.18349v1)

**作者:** Julia Jose `[一作]` (New York University), Rachel Greenstadt `[通讯]` (New York University)

**通讯引用:** 3511 | [OpenAlex ID](https://openalex.org/A5005882490)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在全AI生成的社交平台Moltbook上，对政治宣传内容进行大规模自动检测与分析，探究其流行度、分布、生产者及传播模式。

**💡 创新点**

①首次在完全由LLM代理生成的社交平台系统研究政治宣传；②利用LLM分类器自动标注政治与宣传标签，并通过专家标注验证，确保可靠性；③揭示少数代理在多社区内频繁复用相似叙事的传播行为。

**🔧 技术方法**

采用LLM（GPT‑4o‑mini）零样本推断进行政治与宣传标签分类；使用all‑mpnet‑base‑v2嵌入及余弦相似度评估叙事重复；利用Cohen's κ衡量标注一致性，进行统计对比（均值、显著性检验）。

**📊 数据集**

Moltbook Observatory 2026年3月5日导出数据集，包含673,127篇帖子、879,606条评论，来自93,714名AI代理，分布于4,662个社区。

**📈 对比分析**

通过与专家标注的Cohen's κ（0.64‑0.74）评估分类器一致性；对比政治、宣传标签下的评论数量和比例，发现宣传帖子评论略多但并未显著放大宣传；总体上指标表明LLM分类器具备可接受的可靠性。

**⚠️ 局限性**

①平台处于早期阶段，行为模式可能随发展而变；②代理的真实独立性难以保证，可能存在同一人类控制多代理；③评论分析仅限计数与标签构成，未深入对话结构、时间演化及传播链；未来需更细粒度和动态的分析。

---

## 111. MANAR: Memory-augmented Attention with Navigational Abstract Conceptual Representation

**arXiv ID:** 2603.18676 | [PDF](https://arxiv.org/pdf/2603.18676v1)

**作者:** Zuher Jahshan `[一作]` (Bar Ilan University), Leonid Yavits `[通讯]` (Bar Ilan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种记忆增强的注意力机制MANAR，可作为多头注意力的drop‑in替换，支持知识迁移并实现线性时间复杂度。

**💡 创新点**

将Global Workspace Theory的中央工作空间实现为抽象概念表示（ACR），通过两阶段集成‑广播流程实现非凸表示，并兼容MHA权重复制，突破了传统线性注意力架构与预训练模型的兼容壁垒。

**🔧 技术方法**

采用内存检索、产品键加速的快速检索、两阶段ACR构造、上下文窗口局部注意、非凸聚合等技术实现线性时间的全局信息融合。

**📊 数据集**

在GLUE（自然语言）、ImageNet‑1K（图像）和LibriSpeech（语音）三大数据集上进行评估，并基于RoBERTa、DeiT、data2vec等预训练模型进行知识迁移。

**📈 对比分析**

在相同骨干网络、相同训练配置下仅替换注意力层，MANAR在语言上平均GLUE 85.1、图像上ImageNet‑1K Top‑1 83.9%、语音上WER 2.7%/6.4%，推理速度可达14.8×加速、9.3×显存节省。

**⚠️ 局限性**

目前仅针对编码器设计，未验证旋转位置编码、长序列预训练以及解码器/生成任务的适用性，需要进一步扩展。

---

## 112. Robotic Agentic Platform for Intelligent Electric Vehicle Disassembly

**arXiv ID:** 2603.18520 | [PDF](https://arxiv.org/pdf/2603.18520v1)

**作者:** Zachary Allen `[一作]` (University of Colorado), Nikolaus Correll `[通讯]` (University of Colorado)

**通讯引用:** 6055 | [OpenAlex ID](https://openalex.org/A5047458039)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了RAPID平台，集成了工业机械臂、RGB‑D感知、自动螺母拆除工具以及Agentic AI框架，实现了对全尺寸电动车电池的智能拆解与人机协作；

**💡 创新点**

创新点包括：①人机协作拆解平台与开放世界物体检测模型YoloWorldX的结合；②三种单次螺栓拆除策略（手动置位、视觉定位、视觉伺服）的系统性评估；③基于SmolAgents的Agentic AI工具化接口，证明显式工具显著提升任务成功率与效率；④闭式解析IK与多目标成本规划、TSP任务排序、边界约束运动规划的组合；⑤完整开源硬件与软件设计。

**🔧 技术方法**

使用了Universal Robots UR16e + gantry、Intel RealSense D435i、YoloWorldX检测、MoveIt! 2运动规划、闭式解析IK、视觉伺服与力控、SmolAgents + LLM（GPT‑4o‑mini、Qwen 3.5 4B/9B）、ROS 2/MCP 或显式工具接口。

**📊 数据集**

使用了563张带注释的RGB‑D图像（Bolt、BusBar、Screw、Nut、Cover等）用于训练YoloWorldX；同时使用手工标注的电池部件位置信息用于规划与实验。

**📈 对比分析**

实验比较：三种拆卸策略在204次单次拆卸中的成功率分别为97%、57%和83%，耗时分别为24.1、28.7和36.3分钟；IK对比显示解析IK成功率99.3%且规划时延0.31s；YoloWorldX与Yolov11L对比mAP50分别为0.9757和0.9708；Agentic AI实验表明显式工具接口实现100%成功率、显著降低token数与API成本，MCP接口则有43.3%失败率。

**⚠️ 局限性**

局限性包括：机器人拆解速度仍慢于人工；解析IK与路径规划偶尔失败；视觉深度误差导致伺服失效；螺母工具伸展不足影响拆卸成功率；MCP接口复杂导致高失败率；仅评估单次拆卸未覆盖完整拆解流程；深度传感器对反射表面受限；缺乏错误检测与重试机制。

---

## 113. GenVideoLens: Where LVLMs Fall Short in AI-Generated Video Detection?

**arXiv ID:** 2603.18625 | [PDF](https://arxiv.org/pdf/2603.18625v1)

**作者:** Yueying Zou `[一作]` (Beijing University of Posts and Telecommunications), Ran He `[通讯]` (Center for Research on Intelligent Perception and Computing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个细粒度的 AI 生成视频检测基准 GenVideoLens，并对 15 个真实性维度进行评估。

**💡 创新点**

首次将视频真实性拆解为 15 个维度，并通过维度级别评估揭示 LVLM 在光学、物理和时序推理方面的弱点。

**🔧 技术方法**

使用大型视觉语言模型（Qwen、InternVL、LLaVA 等）进行推理，并对其在各维度上的 F1 分数进行评估。

**📊 数据集**

构建了 400 条高欺骗性的 AI 视频与 100 条真实视频的数据集，含 6,060 条专家标注的维度标签。

**📈 对比分析**

对 11 种 LVLM 进行比较，发现大多数模型在感知维度表现尚可，但在光学一致性、物理交互和时间因果推理上表现差，整体二分类准确率低于 0.65。

**⚠️ 局限性**

局限在于模型对时间信息和物理因果关系的利用不足，跨维度推理表现不佳，且基准缺乏更丰富的攻击方式。

---

## 114. What Really Controls Temporal Reasoning in Large Language Models: Tokenisation or Representation of Time?

**arXiv ID:** 2603.19017 | [PDF](https://arxiv.org/pdf/2603.19017v1)

**作者:** Gagan Bhatia `[一作]` (University of Aberdeen), Wei Zhao `[通讯]` (University of Aberdeen)

**通讯引用:** 507 | [OpenAlex ID](https://openalex.org/A5101855969)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个跨语言、跨日历的时间推理基准MultiTempBench，覆盖了日期算术、时区转换和时间关系三类任务；

**💡 创新点**

创新点包括：①构建多语言、多日历的标准化数据集；②设计了多语言日期碎片化比率mDFR衡量分词质量；③通过几何探测（线性可解性）揭示内部时间表示的几何结构；

**🔧 技术方法**

使用了多种技术：多语言分词器与基线分词对比、mDFR指标、线性回归探测器、交叉混合效应回归分析；

**📊 数据集**

数据集来源于TRAM、ToT、FreshBench的750个英文问题，经Google翻译和本地校对扩展到5种语言、3种日历，最终共15,000条样本；

**📈 对比分析**

评估方式为零样本推理，使用LLM判定器进行自动评测；结果显示在低资源语言（如阿拉伯语、豪萨语）日期碎片化是主要瓶颈，而在高资源语言（如英语、德语、中文）内部时间线性结构更为关键；

**⚠️ 局限性**

局限性包括：仅覆盖5种语言且低资源仅为两种，基准为模板化生成而非真实文本；评测在零样本、无工具的场景下，未考虑提示或微调；机制分析为相关性而非因果，缺乏因果验证。

---

## 115. MolRGen: A Training and Evaluation Setting for De Novo Molecular Generation with Reasonning Models

**arXiv ID:** 2603.18256 | [PDF](https://arxiv.org/pdf/2603.18256v1)

**作者:** Philippe Formont `[一作]`, Pablo Piantanida `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个面向推理型大型语言模型（LLM）的de novo分子生成基准，设计了多目标生成任务，提出多样性感知的top‑k评分，并通过强化学习（GRPO）在该基准上训练LLM，展示了单分子奖励提升但多样性下降的现象。

**💡 创新点**

创新点包括：1) 为de novo分子生成提供新的评估与训练框架；2) 引入多样性感知的top‑k分数以同时衡量质量与多样性；3) 证明该基准可用于训练推理型LLM，并揭示RL优化对多样性的不利影响。

**🔧 技术方法**

使用的技术包括：大规模预训练LLM（如Qwen3、MiniMax-M2、ChemDFM-R、Mistral-24B）、链式思维推理、强化学习框架GRPO、分子评估工具（Docking、属性预测模型）以及多样性度量（Tanimoto相似度）。

**📊 数据集**

使用的数据集是由约4.5k个蛋白质结构以及对应的分子属性预测任务构成的合成数据集，涵盖多目标生成与属性预测提示，用于训练与评估LLM。

**📈 对比分析**

方法上对比了多种开源LLM和化学专用模型，使用top‑k和多样性感知top‑k评分；结果显示MiniMax‑M2在弱多样性约束下表现最好，ChemDFM‑R在强多样性约束下超越它；RL‑Mistral在top‑1上领先，但其多样性显著低下。

**⚠️ 局限性**

局限性包括：1) 评估基于计算代理（Docking、属性预测）而非实验验证；2) RL训练缺乏探索机制导致化学空间覆盖不足；3) 仅尝试有限的RL配置，未系统探索更优训练策略。

---

## 116. Geography According to ChatGPT -- How Generative AI Represents and Reasons about Geography

**arXiv ID:** 2603.18881 | [PDF](https://arxiv.org/pdf/2603.18881v1)

**作者:** Krzysztof Janowicz `[一作]` (University of Vienna), Lauren Bennett `[通讯]` (Esri)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对ChatGPT等生成式AI模型的输出进行实证分析，探讨其在地理知识表示与推理中的默认倾向、脆弱性与误判。

**💡 创新点**

创新点在于将默认强度、脆弱性、分布偏移与“是否真正理解”等维度系统化为可测量的探测框架，并揭示模型在地理任务中的隐含偏差。

**🔧 技术方法**

采用提示工程、温度调节、Persona生成实验、城市规模生成测试等技术手段，对多代LLM进行定量评估。

**📊 数据集**

使用GeoNames特征类型、美国加州洛杉矶人口与犯罪统计、Zipf分布的假想岛屿等公开与合成数据集。

**📈 对比分析**

通过与基准分布（如官方统计）对比以及不同温度下的输出多样性，评估模型默认强度与偏差；实验显示新一代模型默认更强、生成城市规模不符合Zipf律，表明表现不稳定。

**⚠️ 局限性**

局限包括：仅检视输出而非内部机制；提示设计与语境可能影响结果；样本规模有限；未覆盖所有LLM与任务；未深入探究符号化增强带来的潜在问题。

---

## 117. Thinking with Constructions: A Benchmark and Policy Optimization for Visual-Text Interleaved Geometric Reasoning

**arXiv ID:** 2603.18662 | [PDF](https://arxiv.org/pdf/2603.18662v1)

**作者:** Haokun Zhao `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**通讯引用:** 4094 | [OpenAlex ID](https://openalex.org/A5090455375)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GeoAux-Bench和A2PO框架，实现视觉与文本交互的几何推理。

**💡 创新点**

首次将辅助构造步骤与对应图像对齐，并通过三分区采样与自适应奖励学习何时何种构造，显著提升推理效果。

**🔧 技术方法**

使用强化学习（GRPO + A2PO）、视觉重提示、三分区采样、时机与质量奖励等技术。

**📊 数据集**

使用GeoAux-Bench（4334题/8470图）以及Geomverse、Geometry3k等基准数据集。

**📈 对比分析**

在GeoAux-Bench与SFT、GRPO、ToRL、GeometryZero对比，A2PO平均准确率提升至55.8%，比基线高约2–3个百分点，且PPL与SFT保持接近。

**⚠️ 局限性**

局限在于视觉更新仅依赖检索真实图像，未实现模型自生成高精度图像，缺乏完整的动态图像编辑能力。

---

## 118. Bonsai: A class of effective methods for independent sampling of graph partitions

**arXiv ID:** 2603.18347 | [PDF](https://arxiv.org/pdf/2603.18347v1)

**作者:** Jeanne Clelland `[一作]` (University of Colorado), Kristopher Tapp `[通讯]` (Saint Joseph's University)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5064316312)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种名为Bonsai的独立采样算法，用于生成满足人口平衡和连通性的选区划分方案，并与传统的Markov链方法ReCom进行了对比。

**💡 创新点**

创新点在于通过基于随机生成的生成树进行一次性全局划分，实现真正的独立采样，并给出了精确的采样分布；同时引入了容差乘子函数和回溯机制以处理非平衡人口的情况。

**🔧 技术方法**

主要技术包括均匀生成树与最小生成树的采样、生成树切割、容差乘子函数、最佳切点选择、回溯策略以及对生成树的多层递归划分。

**📊 数据集**

实验数据集涵盖了二维格子图（7×7、50×50）以及美国宾夕法尼亚州和北卡罗来纳州2010年VTD边界，分别用于评估人口平衡、几何紧凑性与选举结果的分布。

**📈 对比分析**

比较方法是构建6个规模为10万的样本集，分别来自Bonsai（均匀/最小生成树）与ReCom（四种变体），通过计划总切割边数、单个选区周长以及选举投票分布等指标进行评估，结果显示Bonsai的统计分布与ReCom相近，且具备完全独立、可并行化的优势。

**⚠️ 局限性**

局限性包括：在大规模格子或高选区数情况下，Complete Cut方法耗时过长；Bonsai在某些非平衡或极端形状的图上仍可能陷入回溯或失败；需手动调参（如回溯阈值、容差乘子），且对极端人口偏差的理论保证仍不完善。

---

## 119. Offload or Overload: A Platform Measurement Study of Mobile Robotic Manipulation Workloads

**arXiv ID:** 2603.18284 | [PDF](https://arxiv.org/pdf/2603.18284v1)

**作者:** Sara Pohland `[一作]` (University of California Berkeley), Ankit Verma `[通讯]` (Microsoft Corporation)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文开展了首个针对移动机器人操作工作负载在机载、边缘及云端GPU平台上的测量研究，系统评估了内存占用、推理时延、能耗与任务准确率的权衡；

**💡 创新点**

创新点在于将最新的多模态基础模型（VLM、VLA、LLM）与实际机器人任务结合，量化机载硬件不足、离线推理与网络延迟对机器人性能与电池寿命的影响，并揭示多机器人共享计算资源的机遇与挑战；

**🔧 技术方法**

采用Jetson Orin/Thor、NVIDIA A100、DGX Spark等GPU硬件，结合Wi‑Fi 6/5G网络，对VLMaps、GraphEQA、π_0.5、DreamZero等工作负载进行推理；

**📊 数据集**

使用真实机器人（Stretch 3、TurtleBot 4、SO‑101）收集的传感器视频与任务执行数据，构建了包含感知、规划、导航、操作的完整工作负载集合；

**📈 对比分析**

通过将机载GPU与边缘/云端GPU的执行时间、准确率、功耗进行对比，发现机载GPU在内存与时延上显著劣势（如VLMaps在Orin上慢383%），而云端推理虽时延低但受网络延迟约10 ms导致准确率下降10%；

**⚠️ 局限性**

局限性包括仅评估了有限数量的机器人与GPU型号，未覆盖极低功耗硬件；网络环境仅在实验室设置，真实部署中的波动性和多机器人通信冲突仍需进一步研究；

---

## 120. Security awareness in LLM agents: the NDAI zone case

**arXiv ID:** 2603.19011 | [PDF](https://arxiv.org/pdf/2603.19011v1)

**作者:** Enrico Bottazzi `[一作]` (Leku), Pia Park `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM在可信执行环境（TEE）中对安全证据的感知能力，评估不同模型在收到文本声明和工具返回的证言后对发明信息披露的行为；

**💡 创新点**

提出Attestation Reliance Index（ARI）量化模型对正、负证言的依赖度，揭示LLM对安全验证的异质性与一致性，并指出其在可信协商（NDAI zone）部署中的潜在风险；

**🔧 技术方法**

采用黑盒行为实验、文本+工具调用证言、判别模型自动评分、ARI指标计算等方法；

**📊 数据集**

使用10个自定义发明案例、10种LLM模型、在四种情境下多次（共2000次）实验得到的披露得分；

**📈 对比分析**

通过比较四个情境（无TEE、文本声明、正证言、负证言）下的平均披露得分，发现文本声明提升披露（+46.9%），正证言效应高度异质（ARI_true在-0.814~0.972之间），负证言统一抑制披露（ARI_false均正，平均下降约-79.6%）；

**⚠️ 局限性**

实验仅在非TEE环境模拟，证言并非真实；假设披露度与安全意识相关；无法在真实NDAI区测评模型行为，未来需在真实TEE环境与更复杂数据集验证。

---

## 121. Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection

**arXiv ID:** 2603.18541 | [PDF](https://arxiv.org/pdf/2603.18541v1)

**作者:** Yongwei Jiang `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4190 | [OpenAlex ID](https://openalex.org/A5039670436)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了跨域少样本目标检测中的目标域散焦问题，并提出了一种基于中心‑外周注意力精炼的框架来纠正注意力分散，提高定位精度。

**💡 创新点**

创新点在于首次系统阐述“目标域散焦（Astigmatism）”现象，并提出三模组：正向模式精炼（PPR）利用类别原型聚焦前景；负向上下文调制（NCM）统一建模背景以削弱干扰；文本语义对齐（TSA）通过“not [class]”提示强化中心‑外周对比，实现跨模态注意力校正。

**🔧 技术方法**

核心技术包括：Transformer（Swin-T）微调、类别特征原型计算与投影、背景原型聚合、交叉模态对齐损失、温度化归一化与阈值控制；整体实现对注意力距离的显著收敛。

**📊 数据集**

在六个公开 CD‑FSOD 基准上评测：ArTaxOr、Clipart1k、DIOR、DeepFish、NEU‑DET、UODD；每个数据集均覆盖不同域位移与样本稀疏程度。

**📈 对比分析**

与 10+ 传统 CD‑FSOD 方法（Distill、ViTDet、Detic、DE‑ViT、CD‑ViTO 等）进行比较，本文在 1/5/10 shot 场景下平均提升 3.5/4.2/4.8 mAP（相对 baseline GLIP）并刷新多数据集单项记录，验证了方法的普适性与显著性能。

**⚠️ 局限性**

局限性：对极端域差与极低样本（<1 shot）仍表现有限；方法依赖手工设定原型提取阈值和文本提示，需额外人工标注；在实时部署时，额外的原型投影与文本对齐会带来轻微推理延迟。

---

## 122. Relationship-Centered Care: Relatedness and Responsible Design for Human Connections in Mental-Health Care

**arXiv ID:** 2603.18375 | [PDF](https://arxiv.org/pdf/2603.18375v1)

**作者:** Shivam Shukla `[一作]` (University of California), Magy Seif El-Nasr `[通讯]` (University of California)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出将 AI 心理健康干预从以用户为中心的设计转向以关系为中心的设计，提出“关系生态模型”与六层框架，给出一系列设计指引与启发，旨在让 AI 协助患者加强与人类支持网络（治疗师、家人、朋友）的真实互动。

**💡 创新点**

创新点在于：① 将自我决定理论（SDT）中对“关联性”需求的满足从“模拟”转为“搭建”，强调 AI 应“搭桥”而非“代替”；② 将责任 AI 的六层框架与 SDT 结合，形成面向关系生态的评估与设计模型；③ 通过表格形式给出从采纳到社会层面的具体设计诱惑与目标转变。

**🔧 技术方法**

主要技术为基于 AI 的对话式情感支持与 CBT 任务交付，强调设计时需要嵌入与人类沟通的接口与提示。

**📊 数据集**

文中未使用具体实验数据集，内容为理论与设计框架分析。

**📈 对比分析**

未进行实验比较或性能评估，本文以概念性框架为主，未给出数值指标。

**⚠️ 局限性**

局限性包括：① 对急性危机或缺乏人类支持的患者可能不适用；② 该框架仍需实验验证；③ 未提供具体实现细节与可衡量指标。

---

## 123. Off-Policy Learning with Limited Supply

**arXiv ID:** 2603.18702 | [PDF](https://arxiv.org/pdf/2603.18702v1)

**作者:** Koichi Tanaka `[一作]` (Keio University), Yuta Saito `[通讯]` (Cornell University)

**通讯引用:** 802 | [OpenAlex ID](https://openalex.org/A5101991694)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种针对有限库存环境的离线策略学习框架（OPLS），通过考虑用户相对期望收益来做出推荐决策。

**💡 创新点**

创新点在于：①利用“相对奖励差”（用户期望奖励减去该项目的全体平均奖励）来权衡有限库存的分配；②为不同库存状态（即将售罄或充足）设计不同的决策规则，并通过库存预测动态切换；③在理论上证明在有限库存设置下存在优于传统贪婪策略的政策。

**🔧 技术方法**

技术上主要使用基于模型的离线学习方法：先用监督学习估计 q(x,a)；然后按相对奖励差或绝对奖励选择动作；对混合库存情况采用先判定售罄组/充足组再选择；实验中使用了神经网络估计 q̂(x,a)（3 层 NN）以及统计推断的库存预测。

**📊 数据集**

数据集方面：①合成数据（200 个用户、100 个动作，设置不同的库存比例、动作受欢迎度和估计噪声）；②真实数据 KuaiRec（约 1411 名用户、3327 个视频），对视频赋予人工库存后做离线实验。

**📈 对比分析**

对比方法：传统贪婪模型（在无库存时为最优）。实验显示 OPLS 在所有设定下均优于贪婪策略，尤其在库存稀缺、动作偏好高度一致、用户数量大或估计噪声小的场景中提升明显；当库存充足时，OPLS 与贪婪策略表现相近；在估计噪声增大时，OPLS 的优势仍保持。

**⚠️ 局限性**

局限性：①仅考虑逐个项目的库存约束，未处理全局预算或多项目相互制约的情形；②未引入公平性约束，可能导致某些用户长期被排除；③方法基于离线日志，需假设库存消耗的概率可被准确估计，实际在线动态环境中可能需要进一步调整。

---

## 124. Hierarchical Latent Structure Learning through Online Inference

**arXiv ID:** 2603.19139 | [PDF](https://arxiv.org/pdf/2603.19139v1)

**作者:** Ines Aitsahalia `[一作]` (Columbia University), Kiyohito Iigaya `[通讯]` (Columbia University)

**通讯引用:** 959 | [OpenAlex ID](https://openalex.org/A5000989849)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种名为 HOLMES 的在线层次潜在因子模型，用于在顺序数据中自适应地构建多层级表示。

**💡 创新点**

核心创新在于将嵌套 Chinese Restaurant Process（nCRP）与粒子滤波（Sequential Monte Carlo）相结合，使得模型能在无监督的情况下在实时推断中动态生成树形层次结构，并通过深度衰减的浓度参数实现资源受限的层次化约束。

**🔧 技术方法**

主要技术包括：贝叶斯非参数先验（nCRP）、粒子滤波/序列 Monte Carlo 推断、深度衰减浓度机制、Beta–Bernoulli 似然估计以及离散二进特征的统计更新。

**📊 数据集**

使用两类合成数据集：① 组合式任务（包含 2–5 层的层次生成结构）；② 上下文相关的多时间尺度任务（4 个刺激 × 2 个慢速规则 × 2 个快速奖励值），所有数据均为离散二进特征。

**📈 对比分析**

与平面潜在因子模型（flat latent‑cause model）进行对比。结果表明：① 预测准确度相当或更高；② 表示效率显著提升（熵更低、聚类数更少）；③ 在一跳迁移任务中，HOLMES 在层次更深时获得显著优势；④ 在多时间尺度任务中，HOLMES 的预测准确度显著高于平面模型（约 80% 对比 48%）。

**⚠️ 局限性**

主要局限包括：仅在离散合成任务上验证，缺乏真实世界数据的考察；模型缺乏忘记/剪枝机制，可能在非平稳环境中表现不佳；粒子数量有限导致近似误差，且深度衰减参数的选择仍需经验性调优。

---

## 125. Student views in AI Ethics and Social Impact

**arXiv ID:** 2603.18827 | [PDF](https://arxiv.org/pdf/2603.18827v1)

**作者:** Tudor-Dan Mihoc `[一作]` (Babes Bolyai University), Emilia-Loredana Pop `[通讯]` (Babes Bolyai University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对230名二年级计算机科学学生进行问卷调查，探讨性别视角下学生对人工智能伦理影响及社会效应的认知差异

**💡 创新点**

首次系统性比较男性女性对AI领域受影响程度、伦理风险及职业动机的差异，提供性别维度的实证洞察

**🔧 技术方法**

采用混合问卷设计（闭合题与开放题），结合主题分析（thematic analysis）对文本答案进行编码与归类

**📊 数据集**

调查样本为198名匿名参与者（男119人，女72人）来自罗马尼亚Babes‑Bolyai大学的二年级计算机科学专业

**📈 对比分析**

通过频率统计与性别比例对比，分析各类主题出现比例；未涉及模型性能评估，但显示男女在关注领域和伦理关注度上的显著差异

**⚠️ 局限性**

样本局限于单一高校、单一学年、志愿参与导致潜在偏倚；缺乏跨校或跨文化验证，结果难以广泛推广

---

## 126. Sharpness Aware Surrogate Training for Spiking Neural Networks

**arXiv ID:** 2603.18039 | [PDF](https://arxiv.org/pdf/2603.18039v1)

**作者:** Maximilian Nicholson `[一作]` `[通讯]` (University of Bath), Maximilian Nicholson (University of Bath)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了将 Sharpness Aware Minimization (SAM) 作用于 Surrogate Forward 训练的 Spiking Neural Networks (SNN)，以减少 surrogate‑to‑hard 转移误差（transfer gap）；

**💡 创新点**

提出 Sharpness Aware Surrogate Training (SAST)，在光滑 surrogate 目标上应用 SAM，给出状态稳定性、输入 Lipschitz、光滑性、SAM 一阶近似和非凸收敛保证，并通过局部雅可比条件解释梯度与输入敏感度关系；

**🔧 技术方法**

使用 Surrogate Gradient、LIF 神经元动力学、SAM（sharpness aware minimization）、BPTT、理论与实践相结合的诊断工具；

**📊 数据集**

实验数据集为 N‑MNIST 与 DVS Gesture（附录提及 CIFAR10‑DVS）；

**📈 对比分析**

与传统 surrogate 训练、同计算预算、第二 mini‑batch、ASAM 及阈值校准等做对比，SAST 在保持 surrogate 前向准确率的同时将 transfer gap 从约 0.30 降至 0.025（N‑MNIST）和从 0.43 降至 0.14（DVS Gesture），硬阈值准确率提升约 31%，计算开销约翻倍；

**⚠️ 局限性**

理论仅适用于光滑 surrogate 前向、线性下采样、推理模式归一化，对 MaxPool、BatchNorm、硬前向不适用；实验仅覆盖两个数据集，未完成计算匹配与阈值校准等完整基准；硬阈值准确率仍有限，方法主要是 transfer gap 减少技术。

---

## 127. Measuring ESG Risk in Supply Networks

**arXiv ID:** 2603.18543 | [PDF](https://arxiv.org/pdf/2603.18543v1)

**作者:** Rudy Arthur `[一作]` (University of Exeter), Guillherme Machado `[通讯]` (Universidade de Aveiro)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套基于网络的ESG伤害度量框架，定义了多种聚合与路径计数方法，评估企业或国家在供应链/贸易网络中的ESG风险、脆弱性与影响力。

**💡 创新点**

创新点在于将PageRank和Alpha Centrality统一推广，结合可调节的α、灵活的路径计数与聚合策略，满足不同ESG评价偏好并提高对绿色洗牌的鲁棒性。

**🔧 技术方法**

主要技术包括网络中心性度量（Alpha Centrality、PageRank）、自定义聚合函数（最大、平均、TOP‑k）、多层路径计数（所有路径、最短路径、简单路径）以及基于这些构造的网络伤害H和影响I指标。

**📊 数据集**

使用了合成ESG交互网络、ekoIntelligence公司交互网络（以INFLEXION为核心）以及CEPII国际贸易网络配合世界银行6项环境指标构建的国家级ESG网络。

**📈 对比分析**

通过在合成网络中对不同聚合与路径计数方案的结果进行对比，展示α和路径计数对伤害值的影响；在真实网络中计算脆弱性、影响力和全球影响评分，结果与直觉一致，表现出对风险点的有效识别。

**⚠️ 局限性**

局限性包括仅考虑无权边，未加入交易量权重；聚合函数选择仍需主观决定；模型假设为静态网络，未考虑时间演化，绿色洗牌的复杂场景仍可能被规避。

---

## 128. EDM-ARS: A Domain-Specific Multi-Agent System for Automated Educational Data Mining Research

**arXiv ID:** 2603.18273 | [PDF](https://arxiv.org/pdf/2603.18273v1)

**作者:** Chenguang Pan `[一作]` (edmars.ai), Chengyuan Yao `[通讯]` (edmars.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个面向教育数据挖掘的端到端自动化研究系统（EDM-ARS），通过五个LLM驱动的专用代理（ProblemFormulator、DataEngineer、Analyst、Critic、Writer）自动完成问题定义、数据清洗、建模、评估、论文撰写和同行评审，最终生成完整的LaTeX稿件并包含真实的Semantic Scholar引用；

**💡 创新点**

创新点在于：①面向教育领域的专用多代理管道，将教育学专业知识嵌入每个阶段；②三层数据注册表与泄漏防护机制，系统化捕捉调查数据的特殊编码与时序约束；③基于状态机的协调器支持迭代修正、检查点恢复与沙箱代码执行；④实现自动化方法学同行评审与论文质量自校验；

**🔧 技术方法**

使用技术包括：大规模语言模型（LLM）驱动的代理，状态机协同调度，沙箱化代码执行环境，Semantic Scholar API实现真实引用检索，数据注册表与规则引擎，错误处理与自纠机制；

**📊 数据集**

主要使用的数据集是美国国家教育统计中心（NCES）的全国代表性纵向高中学生调查数据集HSLS:09（2009-2016），未来计划扩展至IPEDS、PISA、ASSISTments等；

**📈 对比分析**

对比方法主要通过内部自动评审机制与人工评审相结合，验证机器学习分析的准确性、模型可解释性和公平性评估；性能方面系统能够在单一数据集上完成完整研究周期，生成符合领域规范的论文，缺乏跨数据集或跨任务的基准对比；

**⚠️ 局限性**

限制包括：仅支持单一数据集（HSLS:09）且只能处理预测任务；论文输出公式化，缺乏深度洞见；对因果推断、迁移学习和多数据集泛化能力尚未实现；系统依赖LLM调用，计算成本较高；

---

## 129. Scalable and Personalized Oral Assessments Using Voice AI

**arXiv ID:** 2603.18221 | [PDF](https://arxiv.org/pdf/2603.18221v1)

**作者:** Panos Ipeirotis `[一作]` (New York University), Konstantinos Rizakos `[通讯]` (New York University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了基于语音AI的可扩展口试系统，用三阶段多智能体流程（身份验证、项目讨论、案例讨论）进行口试，并采用LLM评审委员会（Claude、Gemini、GPT‑5）通过独立评分、协商修正和主席综合的多轮流程对口试记录进行自动评分与反馈；

**💡 创新点**

创新点在于：①将传统口试数字化、实现大规模低成本部署；②采用多智能体架构将对话拆分成可控阶段，防止对话漂移；③引入LLM评审委员会并通过交叉审议提升评分一致性；④开放式、动态生成问题实现透明、可练习的评估形式；

**🔧 技术方法**

技术上结合了语音识别、文本转语音、自然语言处理、定制多智能体工作流、动态变量注入、LLM（Claude、Gemini、GPT‑5）独立评分与协商、人工审核接口；

**📊 数据集**

使用自建的36名本科生口试记录（录音与转录）作为评估数据集；

**📈 对比分析**

与人工双人评分比较，平均分差从独立评分的3.3点降低至1.3点；Krippendorffα从0.52提升至0.86，整体评分一致性达到0.83；成本从约750美元（人工）降至15美元（AI），每名学生约0.42美元；

**⚠️ 局限性**

局限性包括：评分有效性未通过正式验证，仅靠可靠性指标；LLM在问题堆叠、随机性、语音语调等方面仍易出错；对口语障碍或非母语学生友好性不足；需要人工审核处理高分歧案例；数据治理与隐私合规仍需完善。

---

## 130. Deconstructing Open-World Game Mission Design Formula: A Thematic Analysis Using an Action-Block Framework

**arXiv ID:** 2603.18398 | [PDF](https://arxiv.org/pdf/2603.18398v1)

**作者:** Kaijie Xu `[一作]` (McGill University), Clark Verbrugge `[通讯]` (McGill University)

**通讯引用:** 1594 | [OpenAlex ID](https://openalex.org/A5001343030)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“Mission Action Quality Vector（MAQV）”与行动块语法相结合的框架，用于系统化评估开放世界游戏任务的节奏、变化与体验平衡。

**💡 创新点**

创新点在于：① 将六维体验维度（战斗、探索、叙事、情感、解谜、独特性）量化为可比较的向量；② 通过LLM将社区攻略文本自动转换为标准化的行动序列；③ 结合可视化仪表盘实现跨游戏、跨任务的“公式”揭示与反思。

**🔧 技术方法**

技术包括：大语言模型（GPT‑4.1）进行结构化抽取、精细化的行动块词典、基于Flask+Bootstrap的交互式仪表盘、PCA与层次聚类等统计可视化手段。

**📊 数据集**

数据集为约2,200个任务，来源于20款主流开放世界AAA游戏的公开Fandom攻略，经过清洗后得到2191条有效任务序列。

**📈 对比分析**

比较方法为：人类标注的80个任务与LLM抽取结果进行精确序列匹配、微调F1（≈88%）与编辑距离；对MAQV向量做聚类、PCA、雷达图对比；可视化工具的可用性通过SUS（≈67）、UEQ‑S与SEQ评估，整体表现处于“边际可接受”水平。

**⚠️ 局限性**

局限性包括：① 仅关注作者主导路径，忽略分支与失败循环；② 依赖社区攻略与LLM，可能引入误差与偏差；③ 经验维度压缩不足以覆盖所有游戏体验要素；④ 仪表盘学习曲线较高，SUS分数略低；⑤ 研究仅涵盖公开攻略可用的游戏，样本偏向主流AAA作品。

---

## 131. The Provenance Paradox in Multi-Agent LLM Routing: Delegation Contracts and Attested Identity in LDP

**arXiv ID:** 2603.18043 | [PDF](https://arxiv.org/pdf/2603.18043v1)

**作者:** Sunil Prakash `[一作]` `[通讯]` (Indian School of Business), Sunil Prakash (Indian School of Business)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了多智能体LLM系统的治理扩展，解决了委托权威、身份认证与失败处理三大缺口，并验证了“起源悖论”——自报质量导致的任务分配劣化。

**💡 创新点**

创新点包括：①委托合同机制（明确目标、预算、失败策略）；②声明式与审计式身份区分（self‑claimed、attested、verified 等）；③结构化失败语义与完整的验证与传承链。

**🔧 技术方法**

使用技术：LLM Delegate Protocol（LDP）协议扩展、Python SDK 与 Rust 实现、token 与成本预算校验、基于实验的统计分析（Mann‑Whitney U、效应量 d）。

**📊 数据集**

数据集：模拟代理池（10 个代理，真实质量分布），以及三款 Claude 真实模型（Sonnet、Haiku、降级版 Haiku），任务为 10 题推理测试。

**📈 对比分析**

比较方法：随机（blind）、自报质量（self‑claimed）和已审计质量（attested）三种路由策略；性能用平均输出质量（0–1）衡量。实验显示：自报质量平均 0.55，随机 0.68，attested 0.95；效果量 d = 9.51，p < 0.001。

**⚠️ 局限性**

局限性：①仅在模拟与小规模真实模型验证，缺乏大规模多任务、多技能测试；②缺乏完整的第三方审计基础设施与签名机制；③委托合同与失败策略多为客户端校验，无法完全阻止违规行为；④成功标准仍为自由文本，未实现机器可验证规范。

---

## 132. Enhancing Pretrained Model-based Continual Representation Learning via Guided Random Projection

**arXiv ID:** 2603.19145 | [PDF](https://arxiv.org/pdf/2603.19145v1)

**作者:** Ruilin Li `[一作]` (Wuhan University), Xue Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5874 | [OpenAlex ID](https://openalex.org/A5043503650)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SCL-MGSM方法，通过初始任务指导随机投影层（RPL）构建，实现无样本类增量学习；

**💡 创新点**

创新点在于采用记忆守护监督机制（MGSM），基于目标对齐残差准则逐步选择非冗余随机基，自动确定投影维度并保证数值稳定；

**🔧 技术方法**

使用了随机投影层、递归岭回归、目标对齐残差准则、适应性缩放采样、以及首次会话适配（PEFT）等技术；

**📊 数据集**

在ImageNet‑R、ImageNet‑A、ObjectNet、OmniBenchmark等七个无样本CIL基准上进行实验；

**📈 对比分析**

与四类CIL方法（RPL、原型、提示、LoRA）对比，SCL-MGSM在所有设置下均获得最高的A_last与A_avg，提升约2–3%；

**⚠️ 局限性**

局限性在于仍依赖冻结的预训练模型，适配复杂任务时可能受限于初始任务的代表性，并且极大RPL维度仍会增加构建时间。

---

## 133. Translating MRI to PET through Conditional Diffusion Models with Enhanced Pathology Awareness

**arXiv ID:** 2603.18896 | [PDF](https://arxiv.org/pdf/2603.18896v1)

**作者:** Yitong Li `[一作]` (Technical University of Munich), Christian Wachinger `[通讯]` (Technical University of Munich)

**通讯引用:** 8421 | [OpenAlex ID](https://openalex.org/A5069195910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出PASTA框架，利用双臂交互式条件扩散模型将结构MRI转换为功能PET，并实现对阿尔茨海默病病理特征的高保真合成。

**💡 创新点**

创新点包括引入AdaGN多模态融合、MetaROIs病理先验、循环一致性训练(CycleEx)以及2.5D体积生成策略，显著提升了合成PET的结构与病理保真度。

**🔧 技术方法**

使用的技术主要是条件扩散概率模型（DDPM）、UNet架构、AdaGN归一化、MetaROIs权重、CycleEx循环一致性、2.5D邻域采样及辅助分类器一致性损失。

**📊 数据集**

数据集涵盖ADNI（1248名配对T1 MRI与18F‑FDG PET）与TUM内部临床数据（253名配对MRI‑PET），并对不同诊断分组（CN、MCI、AD）进行平衡划分。

**📈 对比分析**

与Pix2Pix、CycleGAN、RegGAN、ResVit、BBDM、BBDM‑LDM等基线比较，PASTA在MAE、PSNR、SSIM及AD分类指标（BACC/F1/AUC）上均显著优于所有基线，病理区域误差最低。

**⚠️ 局限性**

局限在于未进行正式临床读片验证，且对少数样本或其他神经退行性疾病的泛化性能仍需进一步评估。

---

## 134. GEAR: Geography-knowledge Enhanced Analog Recognition Framework in Extreme Environments

**arXiv ID:** 2603.18626 | [PDF](https://arxiv.org/pdf/2603.18626v1)

**作者:** Zelin Liu `[一作]` (Shanghai Jiao Tong University), Xiaofeng Gao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6303 | [OpenAlex ID](https://openalex.org/A5019439900)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个名为GEAR的三阶段地形相似度检索框架，用于在青藏高原上识别与马里亚纳海沟结构相似的山谷；

**💡 创新点**

创新点在于：①将地理先验（如山谷线性直线性、斜率、纹理等）嵌入到筛选、过滤和深度比对三个阶段；②提出Morphology-integrated Siamese Graph Network（MSG‑Net），利用五种地貌指标构建图卷积网络实现跨域结构匹配；

**🔧 技术方法**

主要技术包括Skeleton‑guided Screening & Clipping (基于Zhang‑Suen细化与线性回归)、Derivative Dynamic Time Warping (DDTW)与Eigenshape分析、图卷积网络与多层感知机（MLP）判别器；

**📊 数据集**

使用ASTERS GDEM（青藏高原）和GEBCO 2024（马里亚纳海沟）数字高程模型，并在专家标注的305对山谷-海沟样本上训练与评估；

**📈 对比分析**

与传统机器学习（RF、SVM、PCA‑EPFs）、深度学习（2D‑CNN、ResNet50、SANI‑SSL）以及Geo‑aware模型（GCN‑DP）对比，MSG‑Net在准确率83.93%、召回率98.05%和F1得分86.04%上均优于所有基线；

**⚠️ 局限性**

局限性包括：①对专家标注样本的依赖，数据量有限；②当前框架仅针对地形高度信息，未融入温度、盐度等多模态环境变量；③大规模图卷积计算在更大区域或更细分尺度上可能面临计算瓶颈。

---

## 135. DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models

**arXiv ID:** 2603.18048 | [PDF](https://arxiv.org/pdf/2603.18048v1)

**作者:** Jiaqi Xiong `[一作]` (University of Oxford), Sichen Liu `[通讯]` (XJTLU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 DEAF 基准，系统评估音频多模态大型语言模型（Audio MLLMs）在情绪、背景噪声和说话人身份三种声学与语义冲突场景下的真实听音能力，设计了三层递进的文本干扰实验和相应的诊断指标。

**💡 创新点**

创新点包括：①首次在单一基准上覆盖多维声学冲突（情绪、背景音、说话人身份）；②采用逐级文本干扰（无干扰、误导提示、双重干扰）精确拆分语义偏差与提示共谋；③提出 Acoustic Robustness Score（ARS）与 Environment Discrimination Index（EDI）等量化模型声学信号利用程度的诊断指标。

**🔧 技术方法**

技术上使用：声学与文本的对齐构造、三层实验框架、LLM‑as‑Judge 进行开放式回答分类、ASS/ARS/EDI 指标计算、在七款 Audio MLLM 上进行零样本评估（包括 Gemini‑2.5 Flash、Gemini‑3 Flash、GPT‑4o‑Audio、Audio Flamingo 3、Qwen2‑Audio、Qwen3‑Omni、SALMONN）。

**📊 数据集**

数据集：结合 EMIS、DEMAND 背景噪声库与 ElevenLabs TTS，生成 2,756 条音频样本，覆盖 104 句情绪文本、84 句背景场景文本和 82 句说话人身份文本，构成 ESC、BSC、SIC 三类冲突实验。

**📈 对比分析**

方法：在零样本设置下将每条音频与对应问题交给模型生成文本回答，由 LLM‑as‑Judge 判定为 Correct/Trap/Other，计算 Accuracy、ASS、ARS、EDI。结果显示：L1 时文本主导较弱（SIC 最稳健），L2 时误导提示对 ESC、BSC 影响较大；L3 双重干扰导致 ESC、BSC 的 ARS 接近 0%，仅 SIC 仍保持 34–44%；GPT‑4o‑Audio 在所有层均几乎 0% ARS，Qwen3‑Omni 与 Audio Flamingo 3 在 L1/L2 取得最佳表现。

**⚠️ 局限性**

局限性：①仅覆盖三种声学维度，缺少多说话人、时序推理等更复杂场景；②大部分样本为 TTS 合成，可能不完全反映真实噪声与语音变异；③LLM‑as‑Judge 评判虽可扩展但可能在模棱两可的答案上产生偏差；④未提供人类基准；⑤实验仅涉及少数七款模型，未考察训练或提示策略的影响。

---

## 136. Interleaved Information Structures in Dynamic Games: A General Framework with Application to the Linear-Quadratic Case

**arXiv ID:** 2603.18407 | [PDF](https://arxiv.org/pdf/2603.18407v1)

**作者:** Janani S K `[一作]` (Indian Institute of Technology Madras), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**通讯引用:** 590 | [OpenAlex ID](https://openalex.org/A5070827615)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在非合作动态游戏中任意交织信息结构下的纳什均衡计算，并提出了基于MPN的统一建模和求解方法。

**💡 创新点**

将交织信息结构映射为数学规划网络，并在此框架下推导出LQ游戏的 Riccati-like 方程，提供了全新的计算方法。

**🔧 技术方法**

使用数学规划网络(MPN)、KKT条件、Riccati方程推导以及线性二次游戏理论。

**📊 数据集**

无真实数据集，采用理论示例（三玩家循环信息结构）进行说明。

**📈 对比分析**

与传统开环/反馈信息结构下的求解方法对比，展示了在交织信息下的可行性；由于为理论分析，未给出具体数值性能指标。

**⚠️ 局限性**

仅适用于线性二次游戏，假设信息结构已知且固定；对非线性或动态未知信息结构的推广尚未实现。

---

## 137. Learning Consistent Temporal Grounding between Related Tasks in Sports Coaching

**arXiv ID:** 2603.18453 | [PDF](https://arxiv.org/pdf/2603.18453v1)

**作者:** Arushi Rai `[一作]` (University of Pittsburgh), Adriana Kovashka `[通讯]` (University of Pittsburgh)

**通讯引用:** 2033 | [OpenAlex ID](https://openalex.org/A5072882318)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对体育教练任务中的视频语言模型，提出了双通道自一致性训练框架，利用生成任务与验证任务的视觉注意力一致性来提升时间定位能力。

**💡 创新点**

创新点在于：①在无帧级标注的前提下通过任务互补性（生成与验证）强制视觉注意力保持一致；②设计了头部双向匹配与动态头部选择机制；③使用自监督的 KL 失真与熵正则化，直接引导注意力集中在关键帧。

**🔧 技术方法**

技术手段包括：视觉编码器+Transformer解码器、双通道前向传播、视觉层与注意力头选择、头部双向匹配（Exact+Hungarian）、KL一致性损失、熵正则化、LoRA微调。

**📊 数据集**

使用的数据集包括 VidDiffBench（用于注意力分析），以及三大体育教练基准 Exact、FitnessQA 和 ExpertAF，全部均来自体育动作视频与问答/反馈场景。

**📈 对比分析**

与标准微调基准 PerceptionLM‑8B、闭源 Gemini‑3‑Flash 及开源 Qwen 系列模型进行对比。结果显示在 Exact +14.1% 以及 FitnessQA +14.1% 的准确率提升，ExpertAF BERTScore +0.9，且 8B 开源模型在体育理解任务上已能超过闭源大模型。

**⚠️ 局限性**

局限性在于：①仍依赖共享视觉层与特定任务设计，若任务不够紧密则效果下降；②动态头部选择虽有效但计算成本略高；③对视觉表示的依赖仍存在，若视觉编码质量不足则提升有限。

---

## 138. Progressive Training for Explainable Citation-Grounded Dialogue: Reducing Hallucination to Zero in English-Hindi LLMs

**arXiv ID:** 2603.18911 | [PDF](https://arxiv.org/pdf/2603.18911v1)

**作者:** Vedant Pandya `[一作]` `[通讯]` (Indian Institute of Technology Jodhpur), Vedant Pandya (Indian Institute of Technology Jodhpur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个四阶段的进阶训练管线，用于在英文–印地语双语环境中实现知识驱动、可解释且带引用标注的对话生成。

**💡 创新点**

首次系统化地把多语言适配、引用驱动SFT、GRPO强化学习和可解释性分析整合到同一管线，并在六种模型上量化引用行为与事实一致性的演变。

**🔧 技术方法**

采用多语言适配、带引用标注的监督微调（SFT）、Group Relative Policy Optimization（GRPO）对齐、交叉注意力可视化、积分梯度和遮蔽因果归因等技术。

**📊 数据集**

使用DSTC9、FaithDial、Wizard of Wikipedia三大英文知识对话基准，并通过IndicTrans2将其翻译为印地语，构成双语混合训练集。

**📈 对比分析**

通过在六种模型（Encoder‑Decoder 250M–3B 与 Decoder‑Only 1B–7B）每个阶段分别评估BLEU、ROUGE、BERTScore、FactScore、Citation‑F1和hallucination率，发现SFT后引用精度和事实一致性显著提升，hallucination降至0%，且小模型在SFT后可与大模型匹敌。

**⚠️ 局限性**

仅基于自动评测，缺乏人工评估；印地语数据为机器翻译，可能影响自然度；GRPO实验参数有限，未充分探索RL优势；对不同语言和任务的泛化能力仍待验证。

---

## 139. Efficient Video Diffusion with Sparse Information Transmission for Video Compression

**arXiv ID:** 2603.18501 | [PDF](https://arxiv.org/pdf/2603.18501v1)

**作者:** Mingde Zhou `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22971 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Diff-SIT框架，结合稀疏时序编码和单步视频扩散重建以实现超低比特率下的视频压缩

**💡 创新点**

创新点包括：①稀疏时间编码模块(STEM)仅对骨干帧进行高质量编码并用低比特率光流压缩中间帧；②单步扩散模型ODFTE利用帧类型嵌入(FTE)实现自适应重建；③在单步扩散上结合光流与生成先验显著提升时空一致性与感知质量

**🔧 技术方法**

使用技术包括：稀疏光流压缩、条件编码与上下文网络、HiFiC图像压缩、SpyNet光流估计、DCVC-DC骨干P帧编码、Wan 2.1扩散Transformer、LPIPS/DISTS感知损失、帧差损失等

**📊 数据集**

训练使用Vimeo‑9k（骨干帧）和REDS（端到端微调），测试集为HEVC Class B、MCL‑JCV与UVG视频集

**📈 对比分析**

与传统HEVC/VVC标准、DCVC、DVC‑P、HNeRV等方法对比，在LPIPS、DISTS、CLIPIQA和Ewarp等感知与时序一致性指标上均取得SOTA；在比特率-感知曲线上比对手低约20%比特率并保持相近或更优的感知质量

**⚠️ 局限性**

局限性包括：在像素级指标（PSNR/SSIM）上不具优势；单步扩散仍略逊于多步方案但显著降低延迟；解码时Wan 2.1占用较高计算资源，整体解码延迟仍高于纯压缩模型

---

## 140. A Trace-Based Assurance Framework for Agentic AI Orchestration: Contracts, Testing, and Governance

**arXiv ID:** 2603.18096 | [PDF](https://arxiv.org/pdf/2603.18096v1)

**作者:** Ciprian Paduraru `[一作]` (University of Bucharest), Alin Stefanescu `[通讯]` (University of Bucharest)

**通讯引用:** 916 | [OpenAlex ID](https://openalex.org/A5069339343)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一个面向多代理LLM系统的基于轨迹的保证框架，包含执行记录（Message‑Action Trace）、契约监控、预算约束下的对抗性压力测试、结构化故障注入和运行时治理控制；

**💡 创新点**

将运行时契约、对抗性压力测试、故障注入与治理统一到同一工作流；通过可重放的MAT记录定位首次违规；定义多维度评价指标（成功率、契约违规率、容错率等）；

**🔧 技术方法**

使用运行时验证、契约监控、对抗性搜索、Chaos‑Engineering式故障注入、能力限制与策略屏蔽、统计评估与回放机制；

**📊 数据集**

没有给出具体实验数据集，框架建议使用现有AgentBench、GAIA、AgentDojo、HarmBench等任务集合以及企业级工作负载集合；

**📈 对比分析**

比较方法基于同一协议与指标，评估不同治理方案、不同负载、不同扰动预算等；性能指标包括任务成功率、非终止率、契约违规率、容错率等，具体数值需实验验证；

**⚠️ 局限性**

当前仅提出方法论与评估协议，缺乏完整实验实现与结果；对契约覆盖度、治理策略的有效性评估仍待进一步研究；

---

## 141. SQL-Commenter: Aligning Large Language Models for SQL Comment Generation with Direct Preference Optimization

**arXiv ID:** 2603.18606 | [PDF](https://arxiv.org/pdf/2603.18606v1)

**作者:** Lei Yu `[一作]` (Institute of Software, Chinese Academy of Sciences), Fengjun Zhang `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于LLaMA‑3.1‑8B的大语言模型SQL代码注释生成系统SQL‑Commenter。

**💡 创新点**

创新点在于将持续预训练(CPT)、监督微调(SFT)与直接偏好优化(DPO)三阶段训练流程结合，并公开构建了包含复杂多表 JOIN、窗口函数等高级 SQL 的高质量注释数据集。

**🔧 技术方法**

采用的技术包括 LLaMA‑3.1‑8B、CPT、SFT、DPO 以及人工专家审核的高质量数据标注流程。

**📊 数据集**

使用的数据集包括：1) 约1.9B token 的 SQL 语料库用于 CPT；2) 15,071 对 SQL+注释（来自 Spider 与 Bird 任务）用于 SFT；3) 3,016 对偏好对（chosen/rejected）用于 DPO；评测集为 Spider dev/test 与 Bird dev。

**📈 对比分析**

与多种基线模型（Qwen3‑14B、DeepSeek、CodeLlama 等）在 Spider 与 Bird 上采用 BLEU‑4、METEOR、ROUGE‑L 进行自动评测，SQL‑Commenter 在所有指标上平均提升约 9–13 个百分点，并在人类评估中在正确性、完整性和自然性上显著优于基线。

**⚠️ 局限性**

主要局限在于对复杂语义逻辑（如多层 JOIN、窗口函数组合）的理解仍存在误差，导致注释有时缺乏准确的逻辑推理；模型规模限制也限制了推理深度，未来需要更大模型或更高级的对齐技术来提升推理能力。

---

## 142. Attack by Unlearning: Unlearning-Induced Adversarial Attacks on Graph Neural Networks

**arXiv ID:** 2603.18570 | [PDF](https://arxiv.org/pdf/2603.18570v1)

**作者:** Jiahao Zhang `[一作]`, Suhang Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种利用图神经网络（GNN）中合法的数据删除请求进行攻击的未学习腐败攻击，证明其可以在模型已部署后通过删除特定节点显著削弱模型性能。

**💡 创新点**

首次将近似图未学习导致的性能退化转化为可被利用的攻击手段，并设计了双层优化框架、伪标签替代真实标签以及黑盒梯度近似方法来生成攻击节点。

**🔧 技术方法**

使用双层优化、梯度一阶近似、伪标签生成、Sigmoid 重参数化与投影、邻域稀疏化等技术实现攻击生成与评估。

**📊 数据集**

在四个公开图数据集上进行实验：Cora、Citeseer、Pubmed 与 Flickr。

**📈 对比分析**

与随机、复制、测试复制、TestLink 等启发式基线以及 SBA、UGBA、TDGIA 等现有节点注入攻击对比，攻击在 5% 未学习比例和 5 条边预算下可造成超过 50% 的准确率下降，同时保持训练前性能不受影响。

**⚠️ 局限性**

受限于一阶梯度近似和伪标签精度，攻击对不同未学习算法的迁移性有限；在大规模图上可扩展性仍需进一步改进。

---

## 143. Learning to Reason with Curriculum I: Provable Benefits of Autocurriculum

**arXiv ID:** 2603.18325 | [PDF](https://arxiv.org/pdf/2603.18325v1)

**作者:** Nived Rajaraman `[一作]` (Microsoft Research), Akshay Krishnamurthy `[通讯]` (Microsoft Research)

**通讯引用:** 2811 | [OpenAlex ID](https://openalex.org/A5015082848)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出了自适应课程（autocurriculum）方法，利用模型自身性能与验证器反馈，动态挑选需要训练的提示，从而显著降低链式推理模型的训练成本。

**💡 创新点**

创新点在于证明自适应课程可在监督式和强化学习两种训练范式下分别实现指数级或线性-倒数的样本/计算成本降低，且不依赖提示分布或模型复杂度假设。

**🔧 技术方法**

核心技术包括基于提升（boosting）和对抗式查询的算法框架、使用验证器进行错误提示筛选、对确定性与随机性模型的理论分析以及对参考模型的覆盖率假设。

**📊 数据集**

本文使用理论化的抽象数据集（提示空间与符号集）进行实验验证，并借助公开数学与代码生成数据集（如数学推理、程序合成）的真实验证器进行实证评估。

**📈 对比分析**

与传统的无课程监督学习和简单的拒绝采样相比，autocurriculum在SFT中将所需的教师演示从Θ(d/ε)缩减到O(d)，在RLVR中将计算成本从O(d/ε)降低到O(d + d/ε)，在多项实验中表现出显著的效率提升。

**⚠️ 局限性**

局限性包括对确定性模型的强假设、对完美验证器的依赖、对参考模型覆盖率的严格要求，以及在在线RL、噪声奖励或自我改进循环等实际场景下的推广性尚未完全解决。

---

## 144. Quine: Realizing LLM Agents as Native POSIX Processes

**arXiv ID:** 2603.18030 | [PDF](https://arxiv.org/pdf/2603.18030v1)

**作者:** Hao Ke `[一作]` `[通讯]` (Independent Researcher), Hao Ke (Independent Researcher)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Quine，一个将大型语言模型代理直接映射为 POSIX 进程的运行时架构，利用单一可执行文件递归自我实例化，实现代理的身份、接口、状态与生命周期。

**💡 创新点**

创新点在于：① 把代理抽象与操作系统的进程模型直接对齐，借助内核的隔离、调度与资源控制；② 通过进程 fork/exec/exit 实现递归委托与上下文续写；③ 设计了 Host‑Guest 架构，将确定性控制与概率推理分离；④ 明确指出 POSIX 进程语义在认知代理中的边界与未来扩展需求。

**🔧 技术方法**

使用 POSIX 系统调用（fork、exec、pipe、signal）、标准流 I/O、环境变量、文件系统以及 Go 语言实现主机层；LLM 调用通过远程 API（Anthropic Claude、OpenAI GPT、Google Gemini）实现客体层。

**📊 数据集**

在评估中主要使用了 OpenAI 的 MRCR（Multi‑Turn Retrieval Context Reasoning）基准，处理 4K‑279K token 级别的输入；其余示例为基于日志、代码差异等自然场景的实验。

**📈 对比分析**

实验主要展示了可操作性：递归委托实现了 3 层进程树完成搜索任务；exec 重写实现了 9 次上下文续写完成 279K token 的 MRCR 示例；未提供与现有框架的定量性能对比，侧重结构与可复现性验证。

**⚠️ 局限性**

局限性包括：① POSIX 进程语义不足以表达认知代理的内部结构、世界观与时间可回溯；② 依赖外部 LLM，需通过远程 API 进行推理，无法本地高效推断；③ 仅通过标准流进行交互，难以支持复杂的多模态或分布式场景；④ 需要手动管理文件系统与环境变量以实现上下文续写，使用成本较高。

---

## 145. D-Mem: A Dual-Process Memory System for LLM Agents

**arXiv ID:** 2603.18631 | [PDF](https://arxiv.org/pdf/2603.18631v1)

**作者:** Zhixing You `[一作]` (Einstein Institute of Mathematics), Jason Cai `[通讯]` (AWS AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种双流程记忆系统 D-Mem，结合快速向量检索与高保真全推理，以提升长期对话推理能力。

**💡 创新点**

创新点在于引入多维质量门控动态决定是否使用高成本全推理，从而兼顾效率与精度。

**🔧 技术方法**

使用向量检索、LLM 生成、分层事实抽取、质量评估与全推理等技术。

**📊 数据集**

在 LoCoMo 与 RealTalk 两个长篇对话基准上进行评测。

**📈 对比分析**

与多种检索增广和记忆框架相比，D-Mem 在 GPT-4o-mini 上 F1 达到 53.5，逼近全推理 55.3，同时 token 与推理时间显著下降。

**⚠️ 局限性**

主要局限在于全推理仍需遍历完整历史，且缺乏跨块逻辑链，限制了无限长上下文的可扩展性。

---

## 146. A Dataset and Resources for Identifying Patient Health Literacy Information from Clinical Notes

**arXiv ID:** 2603.19082 | [PDF](https://arxiv.org/pdf/2603.19082v1)

**作者:** Madeline Bittner `[一作]` (National Library of Medicine), Sarvesh Soni `[通讯]` (National Library of Medicine)

**通讯引用:** 551 | [OpenAlex ID](https://openalex.org/A5055267862)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了HEALIX数据集，基于MIMIC-III临床笔记对患者健康素养进行三等级标注，并使用零/少量提示评估LLM性能。

**💡 创新点**

首次公开发布由临床文本构建的健康素养标注数据集，采用细粒度（低、正常、高）标签，并结合主动学习提升样本多样性。

**🔧 技术方法**

使用了手工标注与规则、LLM主动学习（LLaMA-3-8B-Instruct）、零/少量提示分类、SVM基线等技术。

**📊 数据集**

采用MIMIC-III数据库的589条临床笔记，涵盖9种笔记类型（社会工作、护理、出院总结等）。

**📈 对比分析**

通过对比SVM、Qwen、LLaMA等四大LLM在零/少量提示下的四分类/三分类任务，LLaMA 3.3‑70B Instruct在宽松评估下获得最高F1≈0.63，表现优于其他模型。

**⚠️ 局限性**

局限在于单一医院来源、人口学局限、未进行模型微调、对隐晦/上下文依赖的健康素养识别仍存在挑战。

---

## 147. CoDA: Exploring Chain-of-Distribution Attacks and Post-Hoc Token-Space Repair for Medical Vision-Language Models

**arXiv ID:** 2603.18545 | [PDF](https://arxiv.org/pdf/2603.18545v1)

**作者:** Xiang Chen `[一作]`, Jiujiang Guo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出CoDA框架，构造基于临床成像管线的多阶段攻击，评估MVLM鲁棒性，检验多模态助手的技术真实性审计能力，并给出轻量化的教师引导token空间修复方案。

**💡 创新点**

创新点包括：①设计链式分布攻击空间，模拟获取、重建和交付三阶段真实变形；②在SSIM约束下联合优化组合与参数；③对多模态助手进行真实性审计实验；④提出教师引导的token空间适配器修复方法。

**🔧 技术方法**

使用CLIP风格MVLM、贝叶斯优化+随机搜索、SSIM可视化约束、教师引导的LoRA/SSF式token适配器、单次自挑战提示等技术。

**📊 数据集**

使用脑MRI、胸部X‑ray（Cheng数据集）和腹部CT（RSNA RATIC）各200张平衡样本。

**📈 对比分析**

通过四种MVLM（BioMed‑CLIP、UniMed‑CLIP、BMC‑CLIP、Rad‑CLIP）在干净与CoDA攻击下进行零样本分类，攻击成功率可达70–100%，准确率下降15–50%；多模态助手在CoDA下真实性审计准确率显著下降；修复后准确率提升约10–20%，但仍低于干净基准。

**⚠️ 局限性**

局限性在于攻击模型仅覆盖三阶段管线，未涵盖所有设备与协议差异；评估聚焦于二分类任务；修复方案虽轻量化但无法完全恢复性能；需进一步验证在更复杂多模态、多任务场景下的适用性。

---

## 148. OS-Themis: A Scalable Critic Framework for Generalist GUI Rewards

**arXiv ID:** 2603.19191 | [PDF](https://arxiv.org/pdf/2603.19191v1)

**作者:** Zehao Li `[一作]` (University of Science and Technology of China), Zichen Ding `[通讯]` (Shanghai AI Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个可扩展的多智能体评判框架OS-Themis，用于在随机环境下为GUI代理生成可靠的奖励信号；

**💡 创新点**

创新点在于将轨迹拆分为可验证的里程碑（Milestone Verification Module）并通过Reviewer与Judge进行证据链审核（Verdict Calibration Module），从而大幅降低证据稀释与信息丢失；同时构建了跨平台的 OmniGUIRewardBench 基准；

**🔧 技术方法**

采用LLM/视觉语言模型驱动的Selector、Verifier、Reviewer、Judge四智能体协同评估；结合强化学习（GRPO）进行在线训练；使用基准构建、在线RL基础设施、以及自我进化的数据过滤与SFT；

**📊 数据集**

使用 OmniGUIRewardBench（1409条轨迹，AndroidWorld、OSWorld、WindowsAgentArena、macOSArena、WebArena-Lite-v2）以及来自多种GUI代理（Qwen3-VL系列、UITARS、ScaleCUA、Claude等）的真实轨迹；

**📈 对比分析**

与DigiRL、ZeroGUI等主流评判方法在OGRBench上对比，OS-Themis的准确率、精确率和召回率均提升18.8%/29.6%/16.9%/26.2%；在AndroidWorld的在线RL实验中，提升6%（4B模型）至7.1%（8B模型）；在自我进化数据过滤中，使用OS-Themis过滤后SFT提升6.9%/5.0%；

**⚠️ 局限性**

主要限制包括：在线RL扩展受硬件与调度瓶颈限制；奖励细粒度与形状尚未成熟；多智能体评判可能引入偏差或奖励欺骗；隐私风险需本地化或去敏化处理；

---

## 149. Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images

**arXiv ID:** 2603.18461 | [PDF](https://arxiv.org/pdf/2603.18461v1)

**作者:** Kazuya Nishimura `[一作]` (University of Osaka), Yasuhiro Kojima `[通讯]` (National Cancer Center Japan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Cell-type Prototype-informed Neural Network (CPNN)，利用公开单细胞RNA‑seq数据生成细胞类型原型，结合多实例学习从病理WSI中预测滑动片和补丁级基因表达。

**💡 创新点**

创新点在于：①将单细胞表达的协方差结构编码为细胞类型原型并作为先验；②通过学习图像对应的细胞组成权重，将细胞级信息直接注入表达预测；③在预测过程中加入软一致性正则化，提升表达准确性并提供细胞级解释。

**🔧 技术方法**

使用负二项回归生成原型，图像编码器 + 两层MLP 计算细胞比例，负二项似然 + 软正则化优化；可与现有补丁级网络（如STNet、TRIPLEX）做插件集成。

**📊 数据集**

实验数据：滑动片级 3 个 TCGA 公开数据集（BRCA、KIRC、LUAD）；补丁级 3 个空间转录组数据集（CSCC、Her2st、STNet）；单细胞 RNA‑seq 数据来自 GEO、SCPortalen 等公开数据库。

**📈 对比分析**

与 13 种 MIL/MIR 基线（如 HE2RNA、tRNAformer、ILRA、S4MIL 等）对比，在 Spearman 相关性上取得 SOTA；在补丁级任务中集成 CPNN 后 SCC 明显提升，PCC 维持相近。

**⚠️ 局限性**

局限性：①单细胞与WSI 的模态差异仍导致原型更新受限；②细胞类型标签粒度对性能影响，过粗会降低效果；③依赖大规模高质量单细胞数据；④批次效应校正仍不完美，影响跨实验泛化。

---

## 150. On Additive Gaussian Processes for Wind Farm Power Prediction

**arXiv ID:** 2603.18281 | [PDF](https://arxiv.org/pdf/2603.18281v1)

**作者:** Simon M. Brealy `[一作]` (University of Sheffield), Keith Worden `[通讯]` (University of Sheffield)

**通讯引用:** 27167 | [OpenAlex ID](https://openalex.org/A5017996489)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在风机SCADA数据上构建并应用增量高斯过程模型，用于预测单机及整个风电场的功率输出。

**💡 创新点**

创新点在于将加性GAM与GP相结合的加性高斯过程模型，既保持了GP的灵活性又具备GAM的可解释性，能够捕捉风机和风场级功率曲线的变异。

**🔧 技术方法**

使用加性高斯过程模型、逆Sigmoid链接函数、马氏距离滤波、正弦余弦风向特征、类型II最大似然超参数优化等技术。

**📊 数据集**

使用来自英国谢菲尔德大学、剑桥大学和Vattenfall R&D的风机SCADA数据集，共224台风机，重点分析2021年Ciabatta风电场的一年数据。

**📈 对比分析**

未与传统确定性或概率预测模型做直接对比，主要通过可视化展示模型预测与真实功率的匹配程度，结果表明模型能捕捉主要的风速和风向影响，预测趋势与实际一致。

**⚠️ 局限性**

局限包括：滤波方法仍可能留下切断噪声导致误报；模型仅使用一阶加性核，无法捕捉更复杂的交互；未使用稀疏GP或更大规模数据；缺乏定量性能指标。

---

## 151. Motion-o: Trajectory-Grounded Video Reasoning

**arXiv ID:** 2603.18856 | [PDF](https://arxiv.org/pdf/2603.18856v1)

**作者:** Bishoy Galoaa `[一作]` (Northeastern University), Sarah Ostadabbas `[通讯]` (Northeastern University)

**通讯引用:** 2307 | [OpenAlex ID](https://openalex.org/A5031787107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出Motion-o框架，在视频推理中显式加入运动轨迹链（MCoT），实现空间‑时间‑轨迹三维推理。

**💡 创新点**

创新点在于将运动信息拆解为离散的方向、速度、尺度标签，并通过结构化标签把连续观测连成可验证的轨迹，且不需改动模型架构。

**🔧 技术方法**

使用了监督微调+强化学习两阶段训练，奖励包括轨迹一致性和视觉对齐，结合自定义的<motion>标签和双链验证机制。

**📊 数据集**

利用扩充的STGR/Perception‑LM视频推理数据集，加入稠密轨迹标注并计算运动描述，亦在V‑STAR、VideoMME、WorldSense等基准上评测。

**📈 对比分析**

与Open‑o3 Video及Qwen2.5‑VL‑7B等基线相比，Motion‑o在V‑STAR mAM 提升约3点，在VideoMME Overall 提升6点，整体显著优于同类开源模型。

**⚠️ 局限性**

局限在于定位精度受限于底层VLM检测能力，运动标签仍离散化可能缺失细粒度信息，且对多对象交互和非线性轨迹的处理有限。

---

## 152. Learned but Not Expressed: Capability-Expression Dissociation in Large Language Models

**arXiv ID:** 2603.18013 | [PDF](https://arxiv.org/pdf/2603.18013v1)

**作者:** Toshiyuki Shigemura `[一作]` `[通讯]` (Independent Researcher), Toshiyuki Shigemura (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在叙事与问题解决两种任务情境下，对三种主流 LLM 进行 300 次生成实验，检验它们是否会表达已学习的非因果、不可实现的解决方案框架。

**💡 创新点**

创新点在于明确区分模型的记忆能力与生成时的选择行为，证明即使模型具备重构非因果内容的能力，其在常规任务中的输出也会被任务条件和对齐策略压制，揭示能力与表达的解耦。

**🔧 技术方法**

采用观察式实验设计、跨模型（GPT‑5.2、Claude Opus 4.5、Gemini 3 Pro）与情境的交叉设计，并利用 Wilson 区间估计描述非因果框架出现比例；同时进行抽取实验验证模型的重构能力。

**📊 数据集**

未使用公开数据集，而是自行构造了十个中立任务场景（如资源短缺、内部疑惑等），在两种情境下生成文本，并通过自定义提示进行抽取实验。

**📈 对比分析**

比较方法为在同一批生成样本中统计非因果框架出现率（0%）与抽取实验中模型成功重构率（约 80‑87%），展示显著的能力与表达解耦；未给出传统性能指标。

**⚠️ 局限性**

局限性包括：注释由单一研究者完成缺乏互评；接口访问不具版本控制，导致可复现性受限；仅覆盖两种任务情境，未探索更宽泛的提示策略；操作定义有限，可能漏掉其他非因果表现；抽取实验非预注册，方法上可能存在偏差。

---

## 153. A Complexity Hierarchy of Shuffles in Card-Based Protocols

**arXiv ID:** 2603.18608 | [PDF](https://arxiv.org/pdf/2603.18608v1)

**作者:** Tomoki Ono `[一作]` (University of Electro-Communications), Suthee Ruangwises `[通讯]` (Chulalongkorn University)

**通讯引用:** 312 | [OpenAlex ID](https://openalex.org/A5030622574)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文将基于实现难度的卡牌洗牌操作分层，构建洗牌复杂度层次结构，并通过理论证明及穷举实验展示各层可实现的排列集合；

**💡 创新点**

创新点在于提出从实践和理论双重视角对洗牌操作进行层级化分类，证明层级间不可实现性，并基于该层次引入新的洗牌复杂度度量；

**🔧 技术方法**

采用组合数学与概率论对洗牌操作建模，利用群论与马尔科夫链分析洗牌可实现性，进一步进行穷举搜索计算可实现子集数量；

**📊 数据集**

使用小规模牌组（n=2,3,4）的全部排列子集进行实验验证，得到每一层可实现的子集计数；

**📈 对比分析**

通过穷举搜索与对比不同层次下可实现的子集数量，展示层次分级后可实现的排列集合极大受限；

**⚠️ 局限性**

局限性在于仅针对极小牌组做穷举，缺乏对大牌组的解析公式；实验基于理论模型，未对实际洗牌时间或误差进行量化，实际实现难度仍需进一步验证；

---

## 154. Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors

**arXiv ID:** 2603.18782 | [PDF](https://arxiv.org/pdf/2603.18782v1)

**作者:** Jiatong Xia `[一作]` (Australian Institute for Machine Learning), Lingqiao Liu `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了 Points-to-3D，利用显式点云先验作为输入，将点云嵌入到 TRELLIS 隐层空间，随后通过结构性填充和边界细化两阶段采样实现几何可控的 3D 资产与场景生成。

**💡 创新点**

创新点在于：①将点云作为结构隐层初始化而非单纯条件输入；②训练填充网络实现隐层稀疏结构的自适应修补；③采用分阶段采样策略，先全局对齐点云后局部细化，避免边界缺口。

**🔧 技术方法**

技术方法包括：隐层三维扩散模型 TRELLIS、稀疏结构 VAE、Flow Transformer 以及基于条件流匹配的填充网络；还利用 VGGT 进行单图像点云估计。

**📊 数据集**

实验使用 Toys4K、3D‑Front、3D‑Future、HSSD、ABO 等合成数据集，并在 Pix3D 的真实图像上做进一步验证。

**📈 对比分析**

与 TRELLIS、SAM3D、VoxHammer、SceneGen、MIDI 等基线对比，Points‑to‑3D 在 PSNR、SSIM、LPIPS、DINO、Chamfer Distance、F‑Score 等指标上均取得显著提升，尤其在点云覆盖区域几何精度几乎达到完美。

**⚠️ 局限性**

局限性包括：对点云精度敏感，VGGT 估计误差会影响最终几何；在点云极度稀疏或噪声较大的场景下，填充结果仍可能出现缺口；模型对多模态输入（如纯文本）兼容性有限。

---

## 155. MLOW: Interpretable Low-Rank Frequency Magnitude Decomposition of Multiple Effects for Time Series Forecasting

**arXiv ID:** 2603.18432 | [PDF](https://arxiv.org/pdf/2603.18432v1)

**作者:** Runze Yang `[一作]` (Macquarie University), Jie Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 31296 | [OpenAlex ID](https://openalex.org/A5077269879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 MLOW 解耦框架，利用频域幅值谱的低秩分解得到可解释的趋势、季节等效应，并将其作为 TSF 模型的可插拔前置模块。

**💡 创新点**

创新点包括：1) 引入 Hyperplane‑NMF，兼具非负、可解释、无额外优化且能在新样本上直接投影的低秩分解；2) 通过长窗口提取更多频率级别，缓解频谱泄漏；3) 只改动初始投影层即可让主流 TSF 模型显著提升。

**🔧 技术方法**

使用了傅里叶变换、基函数展开、低秩矩阵分解（Hyperplane‑NMF）、余弦相似度正则化、长窗口频谱提取以及 iTransformer、PatchTST 等 Transformer‑based 预测后端。

**📊 数据集**

在八个公开数据集上评估：ETTh1、Electricity（ECL）、Traffic、Weather、PEMS04/08/03/07。

**📈 对比分析**

与移动平均（MA）、PCA、NMF、Semi‑NMF、以及多种深度 TSF 方法（DUET、CycleNet、SparseTSF、TimeKAN、TimesNet、TimeMixer）对比。MLOW 加速后的 iTransformer/PatchTST 在所有数据集、所有预测 horizon 上均以 MSE/MAE 领先对照组，性能提升幅度明显。

**⚠️ 局限性**

局限性：低秩维度 V 需手动调优，最优值因数据集而异；未给出自动化重要性评分机制；目前仅验证在预测任务，未在异常检测、分类等其他时序任务上测试。

---

## 156. VLM-AutoDrive: Post-Training Vision-Language Models for Safety-Critical Autonomous Driving Events

**arXiv ID:** 2603.18178 | [PDF](https://arxiv.org/pdf/2603.18178v1)

**作者:** Mohammad Qazim Bhat `[一作]` (NVIDIA), Kevin Xie `[通讯]` (NVIDIA)

**通讯引用:** 289 | [OpenAlex ID](https://openalex.org/A5040184474)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在真实世界的车载摄像头视频中，通过后训练框架 VLM-AutoDrive 将通用视觉-语言模型（VLM）适配为安全关键驾驶事件检测器；

**💡 创新点**

创新点在于利用多模态监督（元数据字幕、LLM 生成描述、VQA 对、链式推理（CoT）轨迹）进行模块化后训练，并通过数据增强和类别平衡显著提升短时异常检测；

**🔧 技术方法**

采用的技术包括：多模态监督生成、蒙版式多项选择题 (MCQ) 训练、链式推理的 CoT 监督、SFT 与 RL 的结合、以及高帧率视频处理；

**📊 数据集**

使用 Nexar 公开的 10,000 条 40 秒 dashcam 视频（含 1,000 次碰撞、9,000 次近碰撞、9,000 次正常驾驶），经滑动窗口切分后约 53,000 个 4–6 秒短视频片段；

**📈 对比分析**

与零样本预训练模型对比，VLM-AutoDrive 在 CR1 上从 0% 碰撞召回提升至 54.5%（F1 0.69），整体准确率从 35.3% 提升至 77.27%，NVILA 8B 进一步提升至 86.36%；

**⚠️ 局限性**

局限性包括：对极少量事件样本的泛化仍受限；CoT 轨迹质量与规模不足导致推理准确率低于分类；模型对开放式指令的跟随能力下降；未来需扩展更多驾驶异常类别并强化 RL 与更大规模的推理数据。

---

## 157. DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge

**arXiv ID:** 2603.19172 | [PDF](https://arxiv.org/pdf/2603.19172v1)

**作者:** Yuegui Huang `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 34557 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了DyMoE，一个动态混合精度量化框架，旨在提高边缘设备上MoE模型的推理性能。

**💡 创新点**

创新点在于动态专家优先级分类、深度自适应调度和前瞻性预取机制，能够在运行时动态量化专家并重叠I/O延迟。

**🔧 技术方法**

使用了动态混合精度调度、前瞻性预取引擎和混合精度缓存管理等技术。

**📊 数据集**

在Mixtral-8×7B和Qwen3-30B-A3B等代表性MoE架构上进行了评估，适用于12-24GB的边缘设备内存限制。

**📈 对比分析**

与四个最先进的推理系统相比，DyMoE在TTFT上减少了3.44×到22.7×，在TPOT上实现了高达14.58×的加速，同时保持了竞争力的模型准确性。

**⚠️ 局限性**

限制在于DyMoE的性能依赖于动态输入的复杂性，可能在某些情况下无法完全适应所有类型的输入模式。

---

## 158. FUMO: Prior-Modulated Diffusion for Single Image Reflection Removal

**arXiv ID:** 2603.19036 | [PDF](https://arxiv.org/pdf/2603.19036v1)

**作者:** Telang Xu `[一作]` (Shanghai Jiao Tong University), Xiaohong Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 34881 | [OpenAlex ID](https://openalex.org/A5063022663)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于扩散模型的单图像反射去除方法FUMO，利用两种先验（基于VLM的反射强度先验和多尺度残差聚合得到的高频先验）实现空间可控的粗细两阶段恢复。

**💡 创新点**

创新点在于：① 通过VLM直接提取空间化的反射强度先验，①1) 与②高频先验互补；② 在扩散模型中引入门控调制机制，将两种先验融合为空间门控信号，精准调节条件注入；③ 采用粗-精细两阶段框架，粗阶段使用一步扩散恢复低频结构，精细阶段通过轻量化U-Net进行几何一致性和细节修正。

**🔧 技术方法**

使用技术包括：Stable Diffusion V2.1 + ControlNet作为扩散骨干；VLM（如CLIP）生成反射强度先验；多尺度残差聚合得到高频先验；门控调制模块将两先验乘积放大后叠加到控制信号；粗阶段采用一阶扩散重建；精细阶段使用带SimpleGate的U-Net；训练损失为一阶扩散误差、像素、感知和梯度一致性损失。

**📊 数据集**

数据集涵盖：真实对齐数据（Real 89/200对，Nature 200对，RR4k 1230对，RRW 6600对，DRR 23303对），以及从COCO合成的16120对混合图像；评测基准为Nature、Real、SIR²，以及从网络收集的野外混合图像。

**📈 对比分析**

在Nature、Real、SIR²三个公开基准上，FUMO在PSNR/SSIM/LPIPS/CLIPIQA/MUSIQ等指标均取得或接近最佳成绩；在LPIPS和MUSIQ上表现尤为突出，显示出更好的感知质量和色彩一致性；在野外图像上亦保持较高的可视化效果，明显降低残留反射、提升边缘锐度。

**⚠️ 局限性**

局限性包括：① 依赖大型预训练模型（VLM、扩散模型）导致显著的显存/计算成本；② 对极端反射与透射高度交织的场景仍可能出现残留或细节失真；③ 细节修正模块为确定性后处理，无法完全恢复被严重破坏的纹理；④ 对于多光源或高动态范围图像的适应性尚待验证。

---

## 159. myMNIST: Benchmark of PETNN, KAN, and Classical Deep Learning Models for Burmese Handwritten Digit Recognition

**arXiv ID:** 2603.18597 | [PDF](https://arxiv.org/pdf/2603.18597v1)

**作者:** Ye Kyaw Thu `[一作]` (National Electronics and Computer Technology Center), Thepchai Supnithi `[通讯]` (National Electronics and Computer Technology Center)

**通讯引用:** 1122 | [OpenAlex ID](https://openalex.org/A5063918994)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对myMNIST（Burmese Handwritten Digit）数据集进行基准测试，评估11种不同的神经网络架构（包括经典CNN、MLP、RNN、Transformer、FastKAN、EfficientKAN、PETNN变种和JEM），形成统一的可复现性能基线。

**💡 创新点**

首次系统比较多种前沿架构在Myanmar手写数字上的表现，量化PETNN、KAN和能量模型与CNN的竞争力，并揭示不同激活函数对PETNN性能的显著影响。

**🔧 技术方法**

采用PyTorch实现的CNN、MLP、LSTM、GRU、Transformer、FastKAN、EfficientKAN、PETNN（Sigmoid/GELU/SiLU）和JEM等模型，统一优化器、学习率调度、正则化与早停策略。

**📊 数据集**

使用myMNIST（BHDD）公开数据集，包含60,000张训练图像和27,561张测试图像，尺寸28×28像素，10类Burmese数字。

**📈 对比分析**

通过Precision、Recall、F1-Score和Accuracy四指标对模型进行对比，CNN取得最高F1=0.9959、Accuracy=0.9970；PETNN-GELU紧随其后（F1=0.9955、Accuracy=0.9966），JEM、FastKAN/EfficientKAN等也在相对竞争中展示良好表现。

**⚠️ 局限性**

模型仍易出现0/1、3/4等字形混淆，特别是对细节方向性和闭合特征的捕捉有限；实验仅在单张RTX 3090 Ti GPU上完成，对更大规模或多任务迁移的泛化能力尚未验证。

---

## 160. HEP Statistical Inference for UAV Fault Detection: CLs, LRT, and SBI Applied to Blade Damage

**arXiv ID:** 2603.18546 | [PDF](https://arxiv.org/pdf/2603.18546v1)

**作者:** Khushiyant `[一作]` `[通讯]` (University of Freiburg), Khushiyant (University of Freiburg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将粒子物理中的统计工具（似然比检验、CLs 误报控制和顺序神经后验估计）应用于多旋翼无人机的推进器故障检测。

**💡 创新点**

创新点在于将三种粒子物理方法统一到一个推理框架，并实现置信度校准的故障严重度后验分布。

**🔧 技术方法**

使用了似然比检验、CLs 方法、顺序神经后验估计（SNPE）以及多维高斯生成模型。

**📊 数据集**

实验基于 UAV‑FD 六旋翼飞行数据和 PADRE 四旋翼飞行数据。

**📈 对比分析**

与 CUSUM、自动编码器、LSTM‑AE 等基线比较，AUC 在 UAV‑FD 上达到 0.862，PADRE 达 0.986；在 5% 误报率下识别率 93%（10%）/81%（5%）。

**⚠️ 局限性**

局限包括数据集规模有限、仅覆盖持久性故障、单中心 IMU 造成定位精度受限、缺乏故障起始时延评估等。

---

## 161. Tinted Frames: Question Framing Blinds Vision-Language Models

**arXiv ID:** 2603.19203 | [PDF](https://arxiv.org/pdf/2603.19203v1)

**作者:** Wan-Cyuan Fan `[一作]` (University of British Columbia), Ritwik Gupta `[通讯]` (University of California, Berkeley)

**通讯引用:** 573 | [OpenAlex ID](https://openalex.org/A5058595124)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型在不同问题框架（开放式、是/否、选择题）下的视觉注意力偏差，量化其对准确率的影响，并提出通过轻量级提示微调恢复注意力的对策。

**💡 创新点**

发现VLM对问题框架呈现“选择性盲目”，并证明注意力分布变化是导致准确率下降的根本原因；提出基于注意力对齐的可学习提示调优方法来纠正视觉注意力。

**🔧 技术方法**

使用注意力回滚（attention rollout）分析视觉注意力；进行视觉能量与框内注意力的调节；引入L2和KL散度的注意力对齐损失；通过轻量化提示学习实现模型权重不变的微调。

**📊 数据集**

GQA、SeedBench、V*（高分辨率视觉定位）用于内部分析；通用/对齐/精细定位基准 RealWorldQA、MME、MMMU‑Pro、HallusionBench、POPE、HRBench8k、V* 用于整体性能评估。

**📈 对比分析**

对比相同语义问题在不同框架下的准确率和跨框架不一致率；通过注意力调节实验验证恢复注意力可显著提升准确率；在7个基准、5个模型上平均提升1–3个百分点，精细定位任务提升更明显。

**⚠️ 局限性**

仅通过提示微调，未改动模型权重，对极大模型的效果有限；缺乏对更丰富任务或连续框架的验证；对模型内部机制的解释仍有待深入。

---

## 162. MineDraft: A Framework for Batch Parallel Speculative Decoding

**arXiv ID:** 2603.18016 | [PDF](https://arxiv.org/pdf/2603.18016v1)

**作者:** Zhenwei Tang `[一作]` (Nanyang Technological University), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**通讯引用:** 866 | [OpenAlex ID](https://openalex.org/A5030304400)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了批量并行投机解码框架MineDraft，通过交替维护两批请求实现草稿与验证的并行，显著隐藏草稿延迟。

**💡 创新点**

创新点在于采用Batch Parallelism，使草稿和验证阶段同步并行，利用仅一块额外GPU即可实现隐藏草稿时间，并可无缝集成到现有投机解码方法中。

**🔧 技术方法**

技术实现包括在vLLM中开发插件、GPU间直连传输草稿、KV块分配优化、两批次切换调度以及基于Lambert W函数的理论加速证明。

**📊 数据集**

实验使用了ShareGPT、Arena、Spec-Bench、LLM-Tough-Questions等多种公开数据集。

**📈 对比分析**

通过与标准SD、PEARL、EAGLE等基线在吞吐率和端到端延迟上的对比，MineDraft在吞吐率上最高提升约75%，延迟下降约40%。

**⚠️ 局限性**

局限性包括需额外GPU、批次不平衡时会退回到标准SD导致尾部效应、无法在单GPU或缺少草稿模型时使用。

---

## 163. Theoretical Analyses of Detectors for Additive Noise Channels with Mean-Variance Uncertainty under Nonlinear Expectation Theory

**arXiv ID:** 2603.18937 | [PDF](https://arxiv.org/pdf/2603.18937v1)

**作者:** Wen-Xuan Lang `[一作]` (National Center for Mathematics and Interdisciplinary Sciences and Academy of Mathematics and Systems Science), Zhi-Ming Ma `[通讯]` (National Center for Mathematics and Interdisciplinary Sciences and Academy of Mathematics and Systems Science)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文针对噪声分布存在均值和方差不确定性的加性噪声通道，利用非线性期望理论推导了两类通道（AGDN 通道及无均值不确定的 AGDN 通道）的最优检测器，并给出了相应的参数估计方法及仿真验证。

**💡 创新点**

创新点在于：①首次将子线性期望框架（G‑正态分布和最大分布）应用于通信检测问题；②发现均值不确定性决定了最优检测阈值的取值；③在无均值不确定时证明最小距离规则仍为最优；④提出利用 φ‑max‑mean 算法对均值区间和方差区间进行估计。

**🔧 技术方法**

使用的技术主要包括：非线性期望理论（子线性期望、G‑正态分布、最大分布）、子线性期望下的中心极限定理与大数定律、概率界的计算与极值分析，以及数值仿真验证误差概率曲线。

**📊 数据集**

实验使用随机生成的二元输入信号（{-1,1}）以及模拟噪声样本，设置不同的均值区间（如[-0.003,0.067]）和方差区间，未使用公开数据集。

**📈 对比分析**

通过与传统 AWGN 通道下的最小距离检测器在相同 SNR 条件下进行比较，利用最大/最小误差概率曲线评估性能。结果显示，在均值不确定的情形下，本文提出的最优检测器的误差概率低于传统方法，且随着均值不确定区间增大，性能提升更为显著。

**⚠️ 局限性**

局限性包括：仅针对二元输入的单符号检测，未扩展到多比特或多维信号；方差不确定性仅在无均值不确定时可直接估计；实际实现中需估计均值区间，可能面临统计样本不足的问题；对更复杂噪声模型或多通道情形的适用性尚未验证。

---

## 164. Cross-Lingual LLM-Judge Transfer via Evaluation Decomposition

**arXiv ID:** 2603.18557 | [PDF](https://arxiv.org/pdf/2603.18557v1)

**作者:** Ivaxi Sheth `[一作]` (CISPA Helmholtz Center for Information Security), Saab Mansour `[通讯]` (Amazon)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5070002963)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于解构的 LLM 评判框架，通过生成通用评判维度（Universal Criteria Set，UCS）来实现跨语言的评判传递；

**💡 创新点**

创新点在于将评判拆分为语言无关的维度，构建可解释的中间表示，并用轻量化转移模块在仅使用英语标注的情况下实现跨语言迁移；

**🔧 技术方法**

采用 LLM 进行评判维度生成与响应，使用阈值/Likert 评分；用浅层神经网络（或其他线性/树模型）做转移映射；

**📊 数据集**

在多语种的 MEMERAG（RAG 可信度）和 mFACE（摘要可信度）两个数据集上进行实验，涵盖 12 种语言；

**📈 对比分析**

与零样本、CoT、提示式、对话式、以及两种 checklist 基准（RocketEval、CheckEval）等多种强基线比较，UCS 在多数语言上取得最高或近似最高的平衡准确率，特别是对低资源语言的提升显著；

**⚠️ 局限性**

局限包括：对底层 LLM 的稳定性和偏差依赖；假设评判维度在各语言中均适用，可能忽视文化或上下文差异；仅使用英语标注训练转移模型，可能引入英语偏见；

---

## 165. Communication-Efficient and Robust Multi-Modal Federated Learning via Latent-Space Consensus

**arXiv ID:** 2603.19067 | [PDF](https://arxiv.org/pdf/2603.19067v1)

**作者:** Mohamed Badi `[一作]` (University of Oulu), Mehdi Bennis `[通讯]` (University of Oulu)

**通讯引用:** 44961 | [OpenAlex ID](https://openalex.org/A5061429095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种通信高效、鲁棒的多模态联邦学习框架——Latent-Space Consensus（Latent-Consensus），通过学习可训练的投影矩阵将各异构客户端的特征映射到统一低维潜在空间，并仅交换每类的平均潜在表示实现模型协作。

**💡 创新点**

创新点在于：①将投影矩阵视为“翻译器”，实现不同输入维度、网络架构的模型间信息对齐；②引入基于几何中位数的潜在空间正则化，提升对离群/拜占庭客户端的鲁棒性；③仅传输低维类统计量，显著降低通信开销，并在任意拓扑下可实现。

**🔧 技术方法**

主要技术包括：联邦学习框架、可学习投影矩阵、类级潜在统计量通信、几何中位数/均值一致性正则化、去中心化/中心化实现、梯度更新的交替优化。

**📊 数据集**

实验使用了两大真实多模态数据集：USC-HAD 活动识别数据集（加速度/陀螺仪）与 DeepSense 阻塞持续时间预测数据集（毫米波/激光雷达）。

**📈 对比分析**

与 Harmony、FedIoT、FedMD 及中心化基线对比，Latent-Consensus 在两数据集上实现更快收敛、平均准确率更高；在拜占庭攻击下，几何中位数版本保持更高准确率，且通信成本仅为每类 d 维向量，远低于传统参数级或公共数据对齐方法。

**⚠️ 局限性**

局限性包括：投影矩阵学习可能需要额外训练时间、在极端模态不平衡或极低样本情形下潜在统计量估计不稳；方法尚未在大规模节点数和异步通信环境下进行严格验证；隐私分析仍需进一步正式化。

---

## 166. Through the Looking-Glass: AI-Mediated Video Communication Reduces Interpersonal Trust and Confidence in Judgments

**arXiv ID:** 2603.18868 | [PDF](https://arxiv.org/pdf/2603.18868v1)

**作者:** Nelson Navajas Fernández `[一作]` (Bauhaus University), Maurice Jakesch `[通讯]` (Bauhaus University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在两项线上实验中，对比了无AI处理、弱AI处理（肤色平滑、虚拟背景）和强AI处理（全动画化头像）的视频，测量参与者的信任、真相判断、判断准确度与信心。

**💡 创新点**

首次系统评估日常AI视频处理（滤镜、背景替换、头像）对人际信任和判断信心的影响，并区分单一与混合环境的差异。

**🔧 技术方法**

使用微软Teams自带的AI视频处理功能（肤色平滑、灯光校正、虚拟背景、生成头像）来制作实验材料。

**📊 数据集**

采用迈阿密大学欺骗检测数据库（MU3D）的160段正负陈述视频（80条真，80条假），并在平台中嵌入。

**📈 对比分析**

通过线性混合效应模型比较三种AI处理与控制的信任、真相率、准确度与信心。结果显示AI处理显著降低信任与信心，尤其在混合环境下；但对真相判断率与准确度无显著影响。

**⚠️ 局限性**

实验仅使用预录短视频，情境低风险且缺乏实时互动，样本为非高风险情境，可能无法推广到真实高风险或熟悉群体的远程会议。

---

## 167. GAIN: A Benchmark for Goal-Aligned Decision-Making of Large Language Models under Imperfect Norms

**arXiv ID:** 2603.18469 | [PDF](https://arxiv.org/pdf/2603.18469v1)

**作者:** Masayuki Kawarada `[一作]`, Soichiro Murakami `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并公开了GAIN基准，用于评估大型语言模型在面对不完善的公司规范时，如何在保持业务目标的前提下做出决策；

**💡 创新点**

创新点在于：①构造了四大业务域（广告、客服、招聘、金融）的情境，并通过五类压力（目标对齐、风险规避、情感/伦理诉求、社会/权威影响、个人激励）系统地改变情境，真正模拟现实中的决策冲突；②不设单一正确答案，而是关注决策分布与人类基线的相似度；③提供了人类基线与多种LLM的对比，并量化Jensen–Shannon相似度。

**🔧 技术方法**

采用大型语言模型（OpenAI GPT‑5、GPT‑4.1、GPT‑4.1‑mini、Gemma‑3、Phi‑4、Qwen‑3、gpt‑oss）以及自定义Prompt、JSON输出结构；利用Gemini‑2.5 Pro生成并验证情境。

**📊 数据集**

GAIN基准数据集共1,200个实例，覆盖4个业务域，每个域50个基线情境，各自生成5种压力变体，形成6个变体；人类基线由7名熟练工人完成，总计8,400条决策。

**📈 对比分析**

评价方法是将模型产生的动作分布与人类基线分布进行Jensen–Shannon Divergence计算，取相似度（JSS）作为指标。结果显示：GPT系列模型与人类在无压力基线下相似度高，但在“个人激励”压力下显著抵制偏差；Gemma‑3、Qwen‑3更倾向于偏离规范；整体JSS在金融域最低，表明高风险域更难复制人类决策。

**⚠️ 局限性**

局限性包括：①情境为人工合成，缺乏真实复杂性；②决策空间限制为三种动作；③压力类型单一且不考虑组合；④主要以日语设计，跨文化推广需验证；⑤可能存在生成模型偏见；⑥人类基线使用的是普通工人，专业领域专家可能给出不同判断。

---

## 168. Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting

**arXiv ID:** 2603.18218 | [PDF](https://arxiv.org/pdf/2603.18218v1)

**作者:** Guillem Casadesus Vila `[一作]` (Stanford University), Grace Gao `[通讯]` (Stanford University)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5069625302)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出一种实时的月球表面映射框架，结合稠密深度估计、语义分割与3D Gaussian Splatting（3DGS）实现高精度、语义化的三维地图构建；

**💡 创新点**

创新点包括：①将3DGS用于可增量更新的实时地图；②通过语义监督和稠密深度损失改进Gaussians的几何与语义一致性；③自适应稠密策略被简化为体素过滤以避免长时段记忆遗忘；

**🔧 技术方法**

核心技术包括RAFT‑Stereo深度估计、MANet语义分割、3DGS稠密渲染与优化、关键帧异步优化、稠密/空洞深度与语义损失、体素过滤与语义标记初始化；

**📊 数据集**

使用了由LuPNT仿真生成的Spirals与Trajectories两套合成月球数据集，包含立体图像、语义标签与稠密深度；

**📈 对比分析**

方法通过与传统点云基准对比评估，RAFFT‑Stereo与MANet在此数据集上表现最佳；在已知精确位姿下，构建的地图在垂直方向误差约3 cm，优于无LiDAR的点云方案；前端实时率约10 Hz，后端更新率0.1‑1 Hz；

**⚠️ 局限性**

主要局限：①依赖地面真实位姿，尚未集成完整SLAM跟踪前端；②在资源受限的航天平台上实现的计算与内存成本尚高；③对极端阴影与高对比光照的鲁棒性仍有限；④未在多摄像机配置或不同传感器下充分验证；⑤缺乏循环闭合与误差累计修正机制。

---

## 169. A Faster Deterministic Algorithm for Kidney Exchange via Representative Set

**arXiv ID:** 2603.18471 | [PDF](https://arxiv.org/pdf/2603.18471v1)

**作者:** Kangyi Tian `[一作]` (University of Electronic Science and Technology of China), Mingyu Xiao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 1481 | [OpenAlex ID](https://openalex.org/A5033729619)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种新的确定性算法，用代表集技术求解以覆盖受体人数t为参数的肾脏交换问题（KEP），实现了O*(6.855^t)的时间复杂度。

**💡 创新点**

创新点在于将代表集方法与动态规划相结合，对半可行路径-环打包的顶点集合进行压缩，从而大幅降低状态空间，突破了此前O*(14.34^t)的确定性上界。

**🔧 技术方法**

核心技术包括动态规划框架、代表集（q-代表集）的构造与压缩、以及对路径与环长度约束的细致分析。

**📊 数据集**

论文为理论算法研究，并未使用真实数据集；评估通过算法复杂度与已知最优算法的理论比较完成。

**📈 对比分析**

与之前最优的确定性算法（O*(14.34^t)）相比，本文算法在理论上实现了显著加速；随机算法已知O*(4^t)，本算法虽慢于最优随机方案，但在确定性环境下提供了更优的保证。

**⚠️ 局限性**

局限性包括：算法仍为指数级，适用范围受l_p、l_c < t限制；对特殊实例或更大参数的优化尚未完成；未来可进一步探讨仅处理长度≤t的半可行打包、改进随机化上界等方向。

---

## 170. Agent Control Protocol: Admission Control for Agent Actions

**arXiv ID:** 2603.18829 | [PDF](https://arxiv.org/pdf/2603.18829v1)

**作者:** Marcelo Fernandez `[一作]` `[通讯]` (TraslaIA), Marcelo Fernandez (TraslaIA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了Agent Control Protocol（ACP）作为B2B环境中自治代理的正式技术规范，并实现了Go参考实现与51个签名测试向量。

**💡 创新点**

创新点在于将入侵控制模型引入代理治理，构建跨机构可验证的链式委托、单次执行令牌、无状态证明所有权以及不可变审计账本，满足完整的身份、授权、风险评估与跨机构交互需求。

**🔧 技术方法**

使用了Ed25519签名、SHA-256哈希、JCS（RFC 8785）确定性序列化、HTTP API（OpenAPI 3.1.0）、Go语言实现、Python SDK、Docker镜像等技术栈。

**📊 数据集**

使用了51个由Ed25519测试密钥A签名、SHA-256哈希链构成的测试向量作为验证数据集；未使用真实业务数据集。

**📈 对比分析**

通过签名验证、链式委托检查与确定性风险评分实现入侵控制；实现通过自动化测试向量验证合规性；虽然未给出量化性能基准，但实现已通过Go单元测试并可在Docker中快速部署。

**⚠️ 局限性**

局限包括对根密钥（RIK）和ITR的信任锚依赖、对机构共谋的预防不足、实现不合规范可能破坏安全、以及缺乏对量化性能评估与对抗性攻击的详细研究。

---

## 171. DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection

**arXiv ID:** 2603.18757 | [PDF](https://arxiv.org/pdf/2603.18757v1)

**作者:** Haochen Li `[一作]` (Institute of Software), Ling Li `[通讯]` (Institute of Software)

**通讯引用:** 30286 | [OpenAlex ID](https://openalex.org/A5100435361)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种混合CNN-SSM架构DA-Mamba，用于域自适应目标检测。

**💡 创新点**

在CNN基础上引入Image-Aware SSM和Object-Aware SSM实现局部+全局域对齐，且利用视觉提示调节特征。

**🔧 技术方法**

采用卷积网络、状态空间模型（SSM）、Mamba、视觉语言模型CLIP生成类别原型等技术。

**📊 数据集**

Cityscapes→Foggy Cityscapes、Cityscapes→BDD100K、Pascal VOC→Clipart、Pascal VOC→Comic等四个跨域基准。

**📈 对比分析**

与CNN和Transformer基线对比，在所有基准上均提高3–7% mAP，尤其在Cityscapes→Foggy mAP从55.9%提升至58.1%，同时推理速度约1.6×、FLOPs仅为Transformer的37%。

**⚠️ 局限性**

在极端域差或样本分布差异较大时仍受限，且IA-SSM/OA-SSM需额外的视觉提示与双流水线，增加模型设计复杂度。

---

## 172. STEP: Detecting Audio Backdoor Attacks via Stability-based Trigger Exposure Profiling

**arXiv ID:** 2603.18103 | [PDF](https://arxiv.org/pdf/2603.18103v1)

**作者:** Kun Wang `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35382 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 STEP（Stability-based Trigger Exposure Profiling），一种在黑盒硬标签环境下检测语音模型后门攻击的方法。

**💡 创新点**

创新点在于同时利用后门触发器在语义破坏扰动下的标签稳定性和在语义保持扰动下的标签易碎性两种双重异常，并通过无监督逆方差加权融合两条分支的异常分数。

**🔧 技术方法**

技术包括两支扰动分支（语义破坏与语义保持）、单类异常检测（线性核一类 SVM）、无监督逆方差加权融合、仅硬标签查询。

**📊 数据集**

使用的主要数据集为 LibriSpeech（说话人识别）和 Google Speech Commands (GSC，语音指令识别)，以及在物理场景下的录音设备（Honor Magic4 Pro、iPhone 12、LG Wing）。

**📈 对比分析**

与 STRIP、SCALE-UP、NEO、TeCo 等基线对比，STEP 在七种后门攻击下平均 AUROC 达到 97.92%，EER 仅 4.54%，显著优于所有基线。

**⚠️ 局限性**

局限性包括：可能无法抵御同时满足语义保持与语义破坏两侧约束的自适应触发器；在高频触发器的物理衰减下性能可能下降；对 ASR 序列输出的适配尚未实现。

---

## 173. Data-efficient pre-training by scaling synthetic megadocs

**arXiv ID:** 2603.18534 | [PDF](https://arxiv.org/pdf/2603.18534v1)

**作者:** Konwoo Kim `[一作]` (Stanford University), Percy Liang `[通讯]` (Stanford University)

**通讯引用:** 42838 | [OpenAlex ID](https://openalex.org/A5025255782)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练受到数据限制的情况下，研究并实验了利用合成数据（重述、Megadoc）提升语言模型的 i.i.d. 验证损失与下游任务表现的做法。

**💡 创新点**

创新点在于：①证明在无计算上限下，合成数据可实现更好的损失缩放；②提出将同一真实文档的多次重述拼接为 Megadoc 的两种方法（拼接重述与隐式思考），显著提升损失缩放与长文本性能；③展示 Megadoc 与模型集成可以叠加加速，而自蒸馏不具此特性。

**🔧 技术方法**

使用技术包括：基于 Llama 3.1 8B Instruct 的外部生成器生成重述/思考，300 M 参数自回归 Transformer（上下文长度 4096）训练，数据混合与 epoch 控制，交叉文档注意力，超参数调优以最小化 i.i.d. 损失，以及模型集成（平均 logits）。

**📊 数据集**

主要数据集为 200 M token 的 Web 文本（164 k DCLM 文档），通过外部生成器生成的重述/思考作为合成数据；评估使用同一 Web 分布的 i.i.d. 验证集、长上下文 CS arXiv 论文验证集，以及 PIQA、SciQ、ARC Easy 三个下游基准。

**📈 对比分析**

与仅使用原始 Web 数据的 300 M 基线相比，简单重述在 32 次重述时 i.i.d. 损失从 3.55 降至 3.41（数据效率 1.48×），准确率提升约 5%。拼接重述与隐式思考在 32 次重述时进一步将数据效率提升至 1.64× 与 1.80×，下游准确率提升约 6%–9%。长文本验证集损失降低幅度更大（0.14–0.19），并且 Megadoc 与集成可叠加加速。

**⚠️ 局限性**

限制包括：依赖外部更强大的生成器，未在同一模型上预训练生成器；实验规模限定于 300 M 参数模型，可能不直接推广到更大模型；合成数据生成过程对温度、长度等参数敏感，未覆盖所有可能的生成策略；实验主要关注损失与标准基准，未探究生成文本多样性或真实性的细节。

---

## 174. AutoScreen-FW: An LLM-based Framework for Resume Screening

**arXiv ID:** 2603.18390 | [PDF](https://arxiv.org/pdf/2603.18390v1)

**作者:** Zhelin Xu `[一作]` (University of Tsukuba), Atsuyuki Morishima `[通讯]` (University of Tsukuba)

**通讯引用:** 822 | [OpenAlex ID](https://openalex.org/A5103156979)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 AutoScreen-FW，一个面向本地部署、自动化简历筛选的 LLM 框架；

**💡 创新点**

创新点在于：①采用可自定义的样本选择策略（多样性、相似度、聚类）与 persona 说明，让 LLM 作为“职业顾问”进行评估；②通过少样本（few-shot）上下文学习提升开源 LLM 的判定准确率；③在潜能式招聘场景下实现评估维度分数，提升可解释性；

**🔧 技术方法**

技术包括：LLM-as-a-Judge 的提示工程、Qwen3-Embedding-8B/LLM-3.1-Instruct 的少样本推理、k-means++ 采样、UMAP 可视化、与 GPT‑5 系列的性能对比；

**📊 数据集**

使用了 1,655 条公开的日本求职简历（来自 One Career 平台），仅保留“简历内容”与“应聘职位”字段；

**📈 对比分析**

通过在三种 GPT‑5（5.2、5.1、o3）构建的多重真值下评估，Open‑source LLM 在 AutoScreen-FW 下的准确率均高于 GPT‑5‑nano，Qwen‑3‑8B 在 5.1 真值下甚至超过 GPT‑5‑mini；同时每份简历的推理时间比 GPT‑5‑mini/​nano 低 24–48%，提升筛选效率；

**⚠️ 局限性**

局限性在于：需要手动调优采样策略、样本数、样本类型与属性以匹配不同真值，框架无法自动选择最佳配置；实验基于 LLM 构造的真值，仍需与真实招聘者评估进一步验证。

---

## 175. DaPT: A Dual-Path Framework for Multilingual Multi-hop Question Answering

**arXiv ID:** 2603.19097 | [PDF](https://arxiv.org/pdf/2603.19097v1)

**作者:** Yilin Wang `[一作]` (Northeastern University), Jingbo Zhu `[通讯]` (NiuTrans Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了多语言多跳问答基准，并提出了 DaPT 框架来解决多语言多跳问答问题。

**💡 创新点**

创新点在于同时生成源语言与英文的子问题图并通过语义相似度融合，形成双语推理路径，显著弥合语言鸿沟。

**🔧 技术方法**

主要技术包括 RAG、子问题图分解、节点融合、双语检索与答案验证、LLM（GPT‑4o‑mini）与 BGE‑m3 嵌入模型。

**📊 数据集**

使用的公开数据集为 HotpotQA、2WikiMultiHopQA 与 MuSiQue，并将其翻译成 Swahili、Thai、German、Spanish 与 Chinese。

**📈 对比分析**

与结构自由（Zero‑shot、CoT、Vanilla RAG）和结构感知（GraphRAG、HippoRAG2）基线对比，DaPT 在 EM 分数上在 HotpotQA 提升约 6.8%，在 MuSiQue 提升约 15.5%，整体显著优于所有基线。

**⚠️ 局限性**

局限性包括 F1 分数不一定最高，偏好简洁准确答案；仍受检索噪声和翻译差异影响；未在更大规模或更稀缺语言上进行验证。

---

## 176. VISTA: Validation-Guided Integration of Spatial and Temporal Foundation Models with Anatomical Decoding for Rare-Pathology VCE Event Detection

**arXiv ID:** 2603.18343 | [PDF](https://arxiv.org/pdf/2603.18343v1)

**作者:** Bo-Cheng Qiu `[一作]` (National Cheng Kung University), Chih-Chung Hsu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2142 | [OpenAlex ID](https://openalex.org/A5007305393)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了面向稀有病理VCE事件检测的事件级评价管线，融合双骨干网络与多头集成，结合验证引导的分层融合与解剖学约束的时序解码；

**💡 创新点**

创新点在于：1）把事件检测视为metric-aligned任务，强调时序与解剖一致性；2）采用验证引导的模型与骨干加权融合；3）引入解剖约束与 per-label 事件生成提升临界阈值下的 mAP；

**🔧 技术方法**

使用了 EndoFM-LV（短时序编码）与 DINOv3 ViT-L/16（强视觉特征）双骨干；多头集成（BCE、焦点损失、非对称损失）以及温度缩放、阈值微调、形态学后处理；

**📊 数据集**

数据集为 Galar 数据集的 ICPR 2026 RARE‑VISION 竞赛开发集（80 条标注视频）和隐藏测试集（3 条 NaviCam 检查）；

**📈 对比分析**

在验证集上，融合+解剖解码方案实现 temporal mAP@0.5=0.4730、mAP@0.95=0.3658，公开测试集得到 mAP@0.5=0.3530、mAP@0.95=0.3235；

**⚠️ 局限性**

局限性包括：验证集划分对参数选择的依赖导致验证-测试差距；时序分支相对较弱；缺乏更鲁棒的跨视频域适应与参数泛化。

---

## 177. A Human-in/on-the-Loop Framework for Accessible Text Generation

**arXiv ID:** 2603.18879 | [PDF](https://arxiv.org/pdf/2603.18879v1)

**作者:** Lourdes Moreno `[一作]` (Universidad Carlos III de Madrid), Paloma Martínez `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 3637 | [OpenAlex ID](https://openalex.org/A5009969418)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套结合 Human-in-the-Loop 与 Human-on-the-Loop 的框架，用于生成符合 Plain Language 与 Easy-to-Read 标准的可访问文本。

**💡 创新点**

创新点在于将用户实验、专家标注与自动评估指标融合为可追踪的 ECA 规则和 KPI，并在生成、评估与适配阶段同时进行人机协作，形成闭环。

**🔧 技术方法**

采用 LLM 生成、Prompt 与规则约束、自动评估指标（BLEU、SARI、BERTScore、SAMSA 等）、ECA 触发器、检查表以及 RLHF/DPO 等适配技术。

**📊 数据集**

使用 EASIER 语料库、专家标注的复杂词/同义词数据、老年人和智力障碍者的用户实验数据以及多语言的 PL/ER 标准文档。

**📈 对比分析**

通过比对自动评估与人工检查结果，展示在阈值设定下模型能保持语义一致并提升可读性，虽未给出具体数值，但框架能捕捉到指标低于阈值时触发人工干预。

**⚠️ 局限性**

局限性包括对人类审核的高依赖、阈值设定需人工经验、跨域适用性待验证，以及大规模部署的成本和可扩展性问题。

---

## 178. SODIUM: From Open Web Data to Queryable Databases

**arXiv ID:** 2603.18447 | [PDF](https://arxiv.org/pdf/2603.18447v1)

**作者:** Chuxuan Hu `[一作]` (University of Illinois at Urbana-Champaign), Daniel Kang `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在开放域网页中自动收集并结构化为可查询数据库的任务，提出基准与基于多代理的解决方案。

**💡 创新点**

创新点包括：① 将开放域视为可生成数据库的潜在结构；② 构建了包含105个任务、覆盖6个学术领域的基准数据集；③ 设计了基于增强-剪枝（Augment‑then‑Prune）的BFS Web Explorer和基于结构缓存的Cache Manager，显著提升深度探索与跨单元格一致性。

**🔧 技术方法**

技术实现主要采用：大语言模型驱动的多代理框架；动态网页交互与页面内容分析；URL模式推断与最小编辑生成新链接；缓存路径与页面的结构化索引；LLM-as-a-Judge评估方式。

**📊 数据集**

使用的数据集为由48篇学术论文筛选出的105个任务，涵盖人口学、体育、金融、经济、食品与气候学，共计2149个单元格。

**📈 对比分析**

与6个先进Agent（AG2、AutoGPT、AutoGen、OpenAI ResearchBot、Open Deep Research、WebVoyager）在基准上对比，TaskAcc在LLM-judge评估下达91.1%（exact match 69.5%），分别是AG2的约2×、AutoGPT/AutoGen的2.6×/3.1×，甚至超过Open Deep Research的73×。

**⚠️ 局限性**

局限性包括：对需身份验证或非公开网站的适应性不足；依赖LLM判定的语义评估可能产生误判；在极深层或高度动态页面的推断仍存在挑战；缺少跨语言与多域的广泛适用性。

---

## 179. Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token

**arXiv ID:** 2603.19026 | [PDF](https://arxiv.org/pdf/2603.19026v1)

**作者:** Anqi Zhang `[一作]` (University of Birmingham), Yunchao Wei `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 21611 | [OpenAlex ID](https://openalex.org/A5087043856)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 SELF1E，一种无需专门掩码解码器、仅使用单个分割 token 的 MLLM 语义分割框架。

**💡 创新点**

创新点在于通过残差特征补填（RFR）与残差特征放大（RFA）在保留高分辨率图像特征的同时融合 LLM 细粒度信息，并设计双向注意力掩码实现图像与分割 token 的双向交互。

**🔧 技术方法**

采用多模态大型语言模型 InternVL3、pixel‑unshuffle 与 MLP、残差累积、投影等技术，并通过 LoRA 微调。

**📊 数据集**

使用 ADE20k、COCOStuff、Pascal‑Part、LVIS‑PACO、RefCOCO/+/g、ReasonSeg 以及各类 VQA 数据集进行训练与评测。

**📈 对比分析**

与现有使用掩码解码器或多 token 的方法相比，SELF1E 在 RefCOCO/+/g、gRefCOCO、ReasonSeg、开放词汇分割等基准上均达或超过 state‑of‑the‑art，尤其在 RefCOCO 的 cIoU 上提升 1–2% 以上。

**⚠️ 局限性**

局限在于单 token 仍受表达复杂度限制，且对极细粒度目标的分割精度与高端解码器模型相比略逊；此外在极大分辨率图像上的推理效率需进一步优化。

---

## 180. Reasonably reasoning AI agents can avoid game-theoretic failures in zero-shot, provably

**arXiv ID:** 2603.18563 | [PDF](https://arxiv.org/pdf/2603.18563v1)

**作者:** Enoch Hyunwook Kang `[一作]` (University of Washington), Enoch Hyunwook Kang `[通讯]` (University of Washington)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5113046589)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究证明并实验验证了现成推理型大型语言模型（LLM）在无限重复博弈中无需后期训练即可自然趋向纳什均衡，提出了后验采样最佳响应（PS‑BR）策略并展示其在多种博弈环境下的零样本均衡收敛性能。

**💡 创新点**

创新点在于：①将贝叶斯学习与渐近最佳响应结合，理论上证明任何“合理推理”代理在满足有限菜单、KL分离等可实现条件时即可实现零样本纳什收敛；②将该理论推广至未知、随机私有收益的情形；③提出PS‑BR这一具体的后验采样策略，并在模拟实验中展示其对非平凡重复博弈均衡的显著优势。

**🔧 技术方法**

使用的技术包括：贝叶斯推断（对对手策略与自身收益的后验更新）、后验采样最佳响应算法（PS‑BR）、LLM的链式思考与推理提示、弱子观测平衡（weak subjective equilibrium）等理论工具；实验部分采用基于OpenAI GPT-4/ChatGPT-5系列的 Qwen 3.5-27B 语言模型。

**📊 数据集**

数据集：实验使用五种经典重复博弈的模拟环境——Battle of the Sexes、Prisoner’s Dilemma、Promo、Samaritan 与 Lemons，所有数据均为人工构造的动作与收益序列，未使用公开真实对手交互数据。

**📈 对比分析**

比较方法：在每个博弈中对三种策略（Direct Action、SCoT 与 PS‑BR）进行 20 次自我对弈，评估在第 161–180 轮的“均衡行为符合率”；结果显示 PS‑BR 在几乎所有博弈中均可达 90% 以上的合作均衡收敛率，SCoT 在单步均衡下表现良好但无法追踪非平凡重复均衡，Direct Action 的表现最差。

**⚠️ 局限性**

局限性包括：理论假设需要对手策略集有限且满足 KL 分离、对手策略先验包含真值；实验仅覆盖简单离散动作空间和构造博弈，未检验在更复杂、开放式环境中的泛化；后验采样策略对计算成本敏感，且在高维策略空间可能难以收敛；模型对提示与参数的细微变化仍可能产生显著性能波动。

---

## 181. Enhancing Reinforcement Learning Fine-Tuning with an Online Refiner

**arXiv ID:** 2603.18088 | [PDF](https://arxiv.org/pdf/2603.18088v1)

**作者:** Hao Ma `[一作]` (University of Chinese Academy of Sciences), Xiaolin Ai `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 1002 | [OpenAlex ID](https://openalex.org/A5047050212)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在RL微调中引入基于参考模型的在线修正器，提出动态约束机制来自动调节模型的训练约束。

**💡 创新点**

创新点在于将固定的KL正则改为可自适应的动态约束，只在输出出现退化时介入，从而平衡稳定性与探索性。

**🔧 技术方法**

使用PPO/DAPO等RL算法结合自监督交叉熵损失，将参考模型输出作为监督目标。

**📊 数据集**

使用对话数据集 Prompt‑Collection‑v0.1 和代码生成数据集 APPS，以及后续评估在 HumanEval、HumanEval+、MBPP、MBPP++ 等基准上。

**📈 对比分析**

与传统静态KL约束和无约束RL对比，Dynamic 在训练曲线上持续提升奖励、KL 偏离更大但不崩溃；在代码生成上 Pass@1 平均提升约 30%，在对话任务上奖励持续增长。

**⚠️ 局限性**

主要限制是需要额外的在线参考模型推理导致训练时间约增加 48%，且依赖参考模型的质量与修正过滤机制。

---

## 182. Man and machine: artificial intelligence and judicial decision making

**arXiv ID:** 2603.19042 | [PDF](https://arxiv.org/pdf/2603.19042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 183. Authority-Level Priors: An Under-Specified Constraint in Hierarchical Predictive Processing

**arXiv ID:** 2603.18888 | [PDF](https://arxiv.org/pdf/2603.18888v1)

**作者:** Marcela Palejova `[一作]` `[通讯]` (Anglia Ruskin University), Marcela Palejova (Anglia Ruskin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文提出并阐释了“权威层先验”（Authority‑Level Priors，ALPs）这一概念，用以正式化预测处理框架中的治理层约束，决定哪些身份层假设可调控自主神经与行为系统。

**💡 创新点**

其创新点在于将监管可行性与精度加权区分开来，解释了为何意识层的信念更新往往不伴随自主调节的持久改变，并提供了可检验的假设与计算形式化。

**🔧 技术方法**

技术上使用活跃推理（active inference）与自由能原理的计算框架，给出了在策略优化中引入 ALPs 约束的最小化期望自由能公式，强调对策略空间的边界条件。

**📊 数据集**

研究为理论性工作，未使用任何具体数据集；所有论证均基于文献综述与计算形式化。

**📈 对比分析**

文中并未进行实验比较；提出若干可验证指标（如自律神经恢复时间、前额叶负荷下降、跨情境稳定性）以评估 ALPs 的效能，性能尚待未来实证验证。

**⚠️ 局限性**

局限性包括：ALPs 的触发机制、发展来源和神经实现尚未具体阐述；缺乏经验验证与完整的生成模型实现，需要后续实验与模拟进一步检验。

---

## 184. VesselTok: Tokenizing Vessel-like 3D Biomedical Graph Representations for Reconstruction and Generation

**arXiv ID:** 2603.18797 | [PDF](https://arxiv.org/pdf/2603.18797v1)

**作者:** Chinmay Prabhakar `[一作]` (University of Zurich), Suprosanna Shit `[通讯]` (University of Zurich)

**通讯引用:** 3356 | [OpenAlex ID](https://openalex.org/A5018632773)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种针对血管等曲线结构的3D空间图的压缩离散化Token化方法VesselTok，能够在保持拓扑完整性的同时显著降低节点数。

**💡 创新点**

创新点在于利用中心线点与固定伪半径构建连续占据场，再通过VAE+Transformer得到连续Token，实现大规模网络的高保真压缩与重建。

**🔧 技术方法**

采用VAE+Transformer编码器/解码器、占据场建模、傅里叶位置编码、交叉注意力与自注意力、BCE+KL损失、diffusion生成、以及条件流匹配的Link Prediction方法。

**📊 数据集**

使用多种医学数据集，包括气道(ATM、AIIB、AeroPath)、脑血管(COSTA)、肺血管(PARP、HiPas、Pulmonary-AV)、肾血管(RV)以及Circle of Willis等。

**📈 对比分析**

与3DShape2VecSet、Hunyuan3D、VesselGPT等基线比较，VesselTok在clDice、Chamfer Distance、Betti误差、FID、MMD等指标上均显著优于对手，尤其在跨解剖学泛化和生成任务中表现突出。

**⚠️ 局限性**

局限在于对伪半径的选择敏感，且在极稀疏或非中心线表示的结构上可能需改进，模型对不同尺度细节的处理仍有限制。

---

## 185. STEP: Scientific Time-Series Encoder Pretraining via Cross-Domain Distillation

**arXiv ID:** 2603.18688 | [PDF](https://arxiv.org/pdf/2603.18688v1)

**作者:** Chen Zhang `[一作]` (Shanghai Artificial Intelligence Laboratory), Chao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 96680 | [OpenAlex ID](https://openalex.org/A5042841794)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究科学时序数据的统一表示学习，提出 STEP 框架用于跨域蒸馏和自适应编码。

**💡 创新点**

创新点在于：①结合多域基础模型的交叉蒸馏；②引入可学习的自适应分块和统计补偿机制，解决极异序列长度与数值尺度问题。

**🔧 技术方法**

采用 Transformer 编码器、可学习自适应分块、每样本均值方差补偿以及跨域知识蒸馏等技术。

**📊 数据集**

使用七个科学领域的时序任务数据集：GWOSC、LEAVES、STEAD、MarmAudio、SleepEDF、WBCIC 和 RadSeg。

**📈 对比分析**

与 PatchTST、Informer、Moirai、TimeMoE 等 SOTA 结构对比，STEP 在 7 个任务中多达 6 个榜首，准确率/ F1 均提升 10%~20% 以上。

**⚠️ 局限性**

局限性包括：教师模型与目标任务不匹配时蒸馏收益有限；对极高维或极长序列的可扩展性仍需进一步验证。

---

## 186. RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL with Large Language Models

**arXiv ID:** 2603.18859 | [PDF](https://arxiv.org/pdf/2603.18859v1)

**作者:** Xiao Feng `[一作]` (Hong Kong Baptist University), Michael Kwok-Po Ng `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 83 | [OpenAlex ID](https://openalex.org/A5112708486)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 RewardFlow 方法，通过构造状态图并进行拓扑感知的奖励传播，估计中间状态的过程奖励，并将其作为稠密奖励用于对大语言模型进行强化学习训练。

**💡 创新点**

创新点在于：①利用状态图的拓扑结构（可达性、中心性）代替传统稀疏终端奖励；②采用无外部奖励模型的图传播方式；③将局部（状态级）优势与全局（轨迹级）优势融合，提升信用分配的精细度。

**🔧 技术方法**

核心技术包括：状态归一化与噪声转移剔除、状态图构建、逆向 BFS 进行奖励传播、动作级与轨迹级优势估计、PPO 风格的策略更新以及基于组采样的 RL 框架。

**📊 数据集**

实验使用的基准数据集包括：文本环境 ALFWorld、WebShop，视觉环境 Sokoban，以及多跳 QA 任务的 DeepResearch（NarrativeQA、HotpotQA 等）。

**📈 对比分析**

与 RLOO、GRPO、GiGPO、R1‑Instruct、Search‑R1 等方法比较，RewardFlow 在四个基准上均获得最高或接近最高的成功率/准确率；例如 ALFWorld 7B 版整体成功率 89.8%，Sokoban 7B 版成功率 62.4%，并在 OOD 环境与少量 roll‑out 情况下展现更强的鲁棒性与更快的收敛。

**⚠️ 局限性**

局限性：方法高度依赖状态表示的质量；当环境状态缺乏足够的可辨识信息或难以聚合为清晰图结构时，奖励传播效果会下降；此外，构建状态图仍需要手工设计的归一化与剪枝策略，面向自由文本的结构化图构造尚未成熟。

---

## 187. ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation

**arXiv ID:** 2603.19157 | [PDF](https://arxiv.org/pdf/2603.19157v1)

**作者:** Kwanyoung Lee `[一作]` (Hanyang University), Dong-Jin Kim `[通讯]` (Hanyang University)

**通讯引用:** 20673 | [OpenAlex ID](https://openalex.org/A5100344647)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ADAPT 框架，在 Stable Diffusion 3.0 上实现零训练的稀有组合概念生成，解决 R2F 的随机性与引导不一致问题。

**💡 创新点**

创新点包括：①基于注意力动态的 Adaptive Prompt Scheduling（APS），消除 GPT‑4o 的停点随机性；②Pooled Embedding Manipulation（PEM），利用正交投影与自适应权重构造稀有语义嵌入；③Latent Space Manipulation（LSM），在 Transformer 潜空间内实现属性级细粒度引导。

**🔧 技术方法**

使用的技术包括 Stable Diffusion 3.0、CLIP 预训练嵌入、空间注意力评分、正交投影与自适应比例、潜空间插值、Transformer 层内的注意力向量操作，以及 GPT‑4o 进行文本‑图像对齐评估。

**📊 数据集**

评估数据集为 RareBench（包含单/多对象、属性、复杂关系等稀有概念），并辅以 LAION‑aesthetic、PickScore、ImageReward 三个图像质量/人类偏好指标。

**📈 对比分析**

与 R2F、SynGen、Attend & Excite 等基线在 RareBench 上对比，采用 GPT‑4o 评价文本‑图像对齐得分，ADAPT 在 9 个评估维度上平均提升 1.9–16.2 分（约 +10 分），在多对象关系、复杂场景等最难类别表现尤为突出。

**⚠️ 局限性**

局限性包括：仍受 SD3.0 架构约束，对极端复杂的多对象或属性组合仍有生成失真；对超参数（如 λ_pool、τ_s 等）敏感；需要人工提取稀有词/属性，缺乏完全端到端的自动化。

---

## 188. Computational and Statistical Hardness of Calibration Distance

**arXiv ID:** 2603.18391 | [PDF](https://arxiv.org/pdf/2603.18391v1)

**作者:** Mingda Qiao `[一作]` `[通讯]` (University of Massachusetts Amherst), Mingda Qiao (University of Massachusetts Amherst)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了最近被提出的“距离于校准”（distance from calibration）这一误差度量，给出了在分布已知与仅有样本访问两种情形下计算与估计该度量的基本理论与算法。

**💡 创新点**

创新点在于：①首次给出在分布均匀且标签无噪声时可在 O(n⁴) 时间内精确计算距离；②证明去掉任一假设后问题为 NP‑hard；③提出一个通用的 PTAS（多项式时间近似方案）和在样本估计中实现 O(1/n³) 的上界与下界；④利用“类型稀疏化”和“预测稀疏化”等新技巧，显著降低搜索空间与提升概率集中。

**🔧 技术方法**

主要技术包括：
- 将距离转化为最优传输问题并利用线性顺序保持性；
- 对分布进行类型稀疏化、边际概率离散化，得到有限类型实例；
- 通过动态规划求解连贯划分或多背包问题得到 PTAS；
- 通过子集和（Subset Sum）与平衡子集和（Balanced SSP）的归约证明 NP‑hard；
- 对估计问题采用中心极限定理与Berry–Esseen定理的反分布集中与反分布离散化分析。

**📊 数据集**

本文主要在理论分析中使用合成的概率分布与构造实例，未涉及公开的真实数据集；所有结果均在数学实验与理论证明基础上给出。

**📈 对比分析**

算法性能：
- 在均匀无噪声场景下，精确算法 O(n⁴)；
- 在一般场景下，PTAS 运行时间为 (n/ε)^{O(1/ε)}；
- 样本估计：一侧误差上界为 O(1/n³)，下界同阶；整体两侧误差下界至少 Ω(√n/ε)。
这些结果均优于之前仅能得到 O(1/√n) 近似或更差的统计上界。

**⚠️ 局限性**

局限性包括：
- 对于非均匀/有噪声的完整分布，仍只能得到近似解，且存在 NP‑hard 下界；
- 样本估计中一侧误差的 1/n³ 上界是最优的，但两侧误差仍需要多项式依赖于 n；
- PTAS 的运行时间仍为超多项式（(n/ε)^{O(1/ε)}），无法得到真正的 FPTAS；
- 本文的理论证明主要针对离散二分类，若扩展到多分类或连续标签需要进一步研究。

---

## 189. Proprioceptive-only State Estimation for Legged Robots with Set-Coverage Measurements of Learned Dynamics

**arXiv ID:** 2603.18308 | [PDF](https://arxiv.org/pdf/2603.18308v1)

**作者:** Abhijeet M. Kulkarni `[一作]` (University of Delaware), Guoquan Huang `[通讯]` (University of Delaware)

**通讯引用:** 5785 | [OpenAlex ID](https://openalex.org/A5008502528)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套仅使用本体感测（IMU+关节角度/速度/扭矩）进行的状态估计框架，利用学习得到的动力学模型产生伪测量，并通过集合覆盖（set‑coverage）来描述其误差，而不是假设高斯噪声。

**💡 创新点**

核心创新在于：①将误差分布的集合覆盖约束（无分布假设）嵌入递归滤波；②提出基于KL散度的最优投影与矩匹配闭式更新，既保证覆盖概率，又能保持高效的高频更新；③通过后验采样实现低成本的概率与矩估计。

**🔧 技术方法**

技术实现包括：GRU 网络对关节历史进行学习预测；分布无关的 conformal prediction（或 scenario‑optimization）生成误差集合覆盖阈值；IEKF（增量 Lie group EKF）进行IMU积分；KL投影+矩匹配将集合约束映射回高斯后验；随机（quasi）蒙特卡洛估计高维高斯积分。

**📊 数据集**

使用两套真实四足机器人数据集：Vision60（运动捕捉室内）和 Spot（室内外多地形）作为训练与评估；另外在仿真中构造 Gaussian 与混合噪声场景进行对比。

**📈 对比分析**

与基线（传统 Gauss‑IEKF、基于接触的 IEKF、学习到的高斯预测 + EKF 以及感知雷达里程计 GaRLILEO）进行比较。结果显示：在高斯噪声下，性能相近；在非高斯、偏置噪声下，基线出现大幅漂移/不一致，而集合覆盖更新保持一致性、误差提升有限，且更新时间仅 0.07 ms，满足实时性。

**⚠️ 局限性**

局限性包括：需要事先对误差进行分布无关的校准（依赖 β‑mixing 采样和 conformal 置信阈值）；矩匹配后可能不完全满足覆盖概率；目前仅针对四足机器人和单一传感器组；对极端外部扰动或长时间漂移的理论保证尚未完全建立。

---

## 190. A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference

**arXiv ID:** 2603.19133 | [PDF](https://arxiv.org/pdf/2603.19133v1)

**作者:** Yida Zhang `[一作]` (University of Science and Technology Beijing), Rui Wang `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 10789 | [OpenAlex ID](https://openalex.org/A5100431122)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PicoSpec框架，实现边缘-云协同推理的异步推测解码，解决高延迟网络与计算资源不匹配的问题。

**💡 创新点**

创新点在于：①训练无关的异步推测解码管线；②并行草拟、快速验证与重叠通信实现真正的并行；③分离拒绝采样并采用Top‑K稀疏压缩大幅降低通信开销。

**🔧 技术方法**

采用异步流水线、并行草拟、快速验证、零拷贝通信、分离拒绝采样、Top‑K稀疏压缩、KV缓存和动态窗口控制等技术。

**📊 数据集**

在GSM8K（数学推理）和HumanEval（代码生成）两个公开基准上进行评测。

**📈 对比分析**

与云端自回归、原生推测解码、拆分推理基线对比，实验表明在高延迟WAN环境下，PicoSpec可达最高2.9×速度提升，吞吐量从约6.8–13.9 tokens/s提升至17.2–20.2 tokens/s。

**⚠️ 局限性**

性能受草拟模型与目标模型对齐度影响，低接受率或对齐差时退化为同步基线；需要合适的草拟长度（n）来平衡推测收益与局部计算成本。

---

## 191. Self-Tuning Sparse Attention: Multi-Fidelity Hyperparameter Optimization for Transformer Acceleration

**arXiv ID:** 2603.18417 | [PDF](https://arxiv.org/pdf/2603.18417v1)

**作者:** Arundhathi Dev `[一作]` (University of Cincinnati), Justin Zhan `[通讯]` (University of Cincinnati)

**通讯引用:** 2318 | [OpenAlex ID](https://openalex.org/A5101544978)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种名为 AFBS-BO 的全自动稀疏注意力超参数搜索框架，能够为每层/每头动态发现最佳稀疏阈值，从而消除手工调参的瓶颈。

**💡 创新点**

创新点在于将贝叶斯优化与二进制搜索三阶段混合算法相结合，利用多保真度评估（短序列 vs 长序列）显著降低搜索成本，并通过层级自适应参数实现稀疏率的可变性。

**🔧 技术方法**

采用了 Gaussian Process + Expected Improvement 的贝叶斯优化、二进制搜索、线性参数映射、多保真度评估、SpargeAttn 稀疏注意力实现以及 GPU 友好的 block‑sparse kernel 等技术。

**📊 数据集**

主要使用 WikiText‑2 进行验证，并用 C4 数据集测试泛化性能。

**📈 对比分析**

与窗口注意力、Longformer、Reformer、H2O、Top‑K 等基线对比，AFBS‑BO 在 Llama‑2‑7B 上实现 70.7% 稀疏率、7.45 PPL（仅高 0.32 PPL 与密集模型相比），速度提升 3.4×，搜索时间缩短 3.4×、评估次数减少 8.8×。

**⚠️ 局限性**

主要局限是需要代表性校准数据，分布漂移时需重新校准；线性参数映射假设稀疏率与误差单调关系，可能不适用于非标准注意力机制。

---

## 192. A Model Ensemble-Based Post-Processing Framework for Fairness-Aware Prediction

**arXiv ID:** 2603.18838 | [PDF](https://arxiv.org/pdf/2603.18838v1)

**作者:** Zhouting Zhao `[一作]` (Trinity College Dublin), Tin Lok James Ng `[通讯]` (Trinity College Dublin)

**通讯引用:** 292 | [OpenAlex ID](https://openalex.org/A5077802543)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于模型集成的后处理框架，利用混合专家或混合模型将预训练的高性能模型与简单模型结合，以实现多任务（分类、回归、存活分析）的公平性调节。

**💡 创新点**

创新点在于：①将传统的线性修正（如FRAPPÉ）替换为实例自适应的混合专家权重；②支持多种公平性度量并可无感知属性地应用；③首次将后处理方法扩展到时间依赖的存活分析；④框架保持模型无关性，适用于任意黑盒基准。

**🔧 技术方法**

技术实现：混合专家/混合模型公式、门控网络（logistic或softmax）、简单模型（线性/逻辑回归或Cox模型）、L‑BFGS‑B优化、数值积分求解存活分析公平度量、t‑SNE可视化权重分布。

**📊 数据集**

使用七个真实数据集：分类（COMPAS、Heart、Adult、German Credit），回归（Insurance），存活分析（WHAS、Employee）；涵盖单敏感和多敏感设置。

**📈 对比分析**

与FRAPPÉ、ROC、Calibrated EqOdds、Reductions、HGR等基线对比；实验表明框架在所有任务中实现了更低的公平性差距（如DP、EO、SP‑AUC、组均衡存活概率差）且对准确率的影响极小或略有提升；在深层MLP、混合专家/混合模型两种权重策略下均保持优势。

**⚠️ 局限性**

局限性包括：仅关注组层面公平性，未涵盖个体公平；依赖预训练模型，若基准模型不可访问或极其复杂会影响可扩展性；门控网络和简单模型参数需要手动选择特征和正则化强度，可能对不同数据集调优成本较高。

---

## 193. OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting

**arXiv ID:** 2603.18510 | [PDF](https://arxiv.org/pdf/2603.18510v1)

**作者:** Hongjia Zhai `[一作]` (Zhejiang University), Guofeng Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 42753 | [OpenAlex ID](https://openalex.org/A5100373698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于3D高斯剥层的在线开词汇全景图映射系统，利用滑动窗口实现局部到全局的几何重建与语义感知。

**💡 创新点**

创新点包括：多线索（几何、语义、视角一致性）段聚类实现噪声2D先验的3D实例一致化；显式稀疏体素网格存储语言特征与实例标签；双向二分匹配实现局部到全局的鲁棒融合。

**🔧 技术方法**

使用3D Gaussian Splatting、可微渲染、CLIP+LSeg+EntitySeg获取的2D语言特征与实例掩码、图聚类、稀疏体素网格、双向匈牙利算法进行匹配和融合。

**📊 数据集**

在ScanNetV2和Replica两大室内RGB‑D数据集上进行评估。

**📈 对比分析**

与多种离线（LangSplat、OpenGaussians、PanoGS等）和在线（O2V‑Mapping、OnlineAnySeg等）基线对比，在线方法在3D语义mIoU、mAcc提升约12%，在panoptic PRQ上与部分离线方法相当甚至优于PanoGS。

**⚠️ 局限性**

局限性：目前无法重建动态物体；仍需要深度与位姿输入；缺乏无位姿/无深度的端到端方法。

---

## 194. HRI-SA: A Multimodal Dataset for Online Assessment of Human Situational Awareness during Remote Human-Robot Teaming

**arXiv ID:** 2603.18344 | [PDF](https://arxiv.org/pdf/2603.18344v1)

**作者:** Hashini Senaratne `[一作]` (CSIRO Robotics), Leimin Tian `[通讯]` (CSIRO Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在模拟搜救人机协作任务中收集多模态数据，构建HRI‑SA数据集，并用其评估实时感知情境意识延迟（PSAL）的检测。

**💡 创新点**

首次公开完整、持续覆盖任务全过程的情境意识延迟标注数据；演示通用眼动特征结合最小上下文信息即可实现在线PSAL检测。

**🔧 技术方法**

采用眼动追踪（Tobii Pro Fusion）、生理监测（智能手表）、机器人传感器、ROS日志等多模态采集；使用机器学习模型（Logistic、RF、SVM、MLP等）进行分类。

**📊 数据集**

HRI‑SA数据集：30名参与者在隧道和洞穴两种环境下的搜索救援任务，包含眼动、心率、皮肤电、温度、交互事件、机器人状态及手工标注的PSAL/CSAL。

**📈 对比分析**

使用留一组交叉验证（14组）评估模型，单眼动特征的MLP F1≈67.6%（召回88.9%），单上下文特征的RF F1≈71.6%，融合特征的RF F1≈80.4%，AUC分别为0.77、0.90、0.97。

**⚠️ 局限性**

局限于模拟环境可能低于真实任务的警觉度；仅标注了感知与理解层的延迟，未覆盖预测性延迟；数据量相对有限，缺乏多任务/多机器人验证。

---

## 195. Regret Bounds for Competitive Resource Allocation with Endogenous Costs

**arXiv ID:** 2603.18999 | [PDF](https://arxiv.org/pdf/2603.18999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 196. I Can't Believe It's Corrupt: Evaluating Corruption in Multi-Agent Governance Systems

**arXiv ID:** 2603.18894 | [PDF](https://arxiv.org/pdf/2603.18894v1)

**作者:** Vedanta S P `[一作]` (Indian Institute of Information Technology Kottayam), Ponnurangam Kumaraguru `[通讯]` (Indian Institute of Information Technology Hyderabad)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过大规模多智能体治理仿真，使用大型语言模型（LLM）在不同机构权力结构下评估规则违规与腐败行为，统计28,112条对话片段。

**💡 创新点**

提出治理结构比模型身份更关键的实证观点，并将机构完整性视为部署前的先决条件，构建了“先部署前完整性测试”框架。

**🔧 技术方法**

使用 Concordia 仿真框架、LLM 代理、游戏管理员（Game Master）以及独立的基于规则的 LLM 评判器进行行为评分。

**📊 数据集**

数据集包括 gpt‑5‑mini、claude‑4‑5‑sonnet、Qwen 系列等多种模型在三种治理模板（社会主义、共产主义、联邦制）下生成的 28,112 条对话片段。

**📈 对比分析**

对每个模型‑治理组合统计运行级别的治理失败（GF）、核心腐败（CC）和严重核心腐败（SCC）指标。结果显示，在非饱和模型下，治理结构显著降低违规率，超过模型本身的影响；而在高能力模型下，治理效应被能力饱和削弱。

**⚠️ 局限性**

局限性：仅在理想化仿真场景中验证，评判器依赖固定阈值的 LLM 可能产生误判，缺乏跨框架复现，模型与数据范围有限，尚未对真实机构部署进行实证检验。

---

## 197. Robust Beamforming for Practical RIS-Aided RSMA Systems with Imperfect SIC under Transceiver Hardware Impairments

**arXiv ID:** 2603.18840 | [PDF](https://arxiv.org/pdf/2603.18840v1)

**作者:** Xuejun Cheng `[一作]` (Shandong University), Bruno Clerckx `[通讯]` (Imperial College London)

**通讯引用:** 16365 | [OpenAlex ID](https://openalex.org/A5070530952)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在硬件失真、RIS幅相耦合以及不完全SIC下，RIS辅助RSMA系统的鲁棒波束成形设计与优化。

**💡 创新点**

提出了同时考虑幅相耦合、发射/接收硬件失真和不完美SIC的完整系统模型，推导了实际RIS的渐进SNR损失，并设计了基于AO+ADMM的鲁棒SRM优化算法。

**🔧 技术方法**

采用幅相耦合模型、RIS相位矩阵设计、RSMA分层编码、硬件失真噪声建模、凸优化（quadratic transform）以及ADMM/AO迭代求解。

**📊 数据集**

使用Rayleigh/Rician多径信道仿真（K=2,M=8,N=16），通过Monte Carlo（10⁶次）生成数据；未使用公开数据集，全部为仿真数据。

**📈 对比分析**

与理想RIS方案、非鲁棒设计、NOMA、SDMA等进行对比，结果表明鲁棒RSMA在所有硬件失真条件下均优于NOMA并可匹敌SDMA，且收敛稳定、可自适应退化为SDMA。

**⚠️ 局限性**

主要限制包括仅考虑单层RSMA、固定硬件失真模型，缺乏真实硬件验证，多层RSMA与动态RIS更新的研究以及算法实现的复杂度挑战。

---

## 198. Conflict-Free Policy Languages for Probabilistic ML Predicates: A Framework and Case Study with the Semantic Router DSL

**arXiv ID:** 2603.18174 | [PDF](https://arxiv.org/pdf/2603.18174v1)

**作者:** Xunzhuo Liu `[一作]` (vLLM Semantic Router Project), Xue Liu `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对使用概率化机器学习信号（如嵌入相似度、分类器输出）驱动的策略语言，提出了冲突检测与消除的完整框架，并在 Semantic Router DSL 中实现了编译器级检测、互斥声明、Voronoi 归一化等技术。

**💡 创新点**

创新点在于：①构建了一个包含经典与新型概率冲突的三层可判定层次结构；②首次将温度缩放的 softmax 视为 Voronoi 分区来保证嵌入信号的互斥性，无需模型重训练；③设计了可在编译时完成的静态检查和声明式互斥块，适用于 LLM 路由、语义 RBAC 与 API 网关等多场景。

**🔧 技术方法**

使用技术包括：逻辑/布尔 SAT 与线性整数算术用于传统冲突检测；几何方法（球面帽交集）判定嵌入信号共发；Voronoi 归一化（温度软最大化）实现互斥；编译器扩展（Go 语法树分析、警告与自动修复）；实验中对 MMLU 类别嵌入与 CLIP/SetFit 模型进行评估。

**📊 数据集**

主要数据集为 MMLU（Multi‑Task Language Understanding）用于构造类别集合，以及 CLIP/SetFit 嵌入模型的公开语料；实验示例包含自然语言查询（如“量子隧穿概率”）与 API 请求体。

**📈 对比分析**

与传统无冲突检测的 DSL 进行对比，新的检测机制在编译阶段几乎无额外成本（≤1 ms/规则），并能在运行时避免多路由错误；在 MMLU 分类实验中，使用 Voronoi 归一化后同类查询的误路由率从约12 %降至0 %。

**⚠️ 局限性**

局限性包括：①对分类器信号的校准冲突仍不可在无分布信息的情况下静态判定；②Voronoi 归一化仅适用于嵌入式几何信号，无法处理所有连续信号；③需要手动定义互斥组，若未显式声明仍可能出现冲突；④对非常高维嵌入的球面帽交集仍有数值稳定性挑战。

---

## 199. Cross-Layer Traffic Allocation and Contention Window Optimization for Wi-Fi 7 MLO: When DRL Meets LSTM

**arXiv ID:** 2603.18602 | [PDF](https://arxiv.org/pdf/2603.18602v1)

**作者:** Zhang Liu `[一作]` (Xiamen University), Ying-Jun Angela Zhang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 15494 | [OpenAlex ID](https://openalex.org/A5004874287)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了针对Wi‑Fi 7多链路操作（MLO）的跨层优化框架，联合调度上层MAC的流量分配（β）与下层MAC的初始竞争窗口（ICW）大小，以最大化网络吞吐量。

**💡 创新点**

创新点包括：① 将单链路Bianchi马尔可夫模型扩展至多链路情境，得到吞吐量与β、ICW的解析关系；② 将该非凸非线性跨层优化问题转化为深度强化学习任务，并首次将LSTM嵌入Soft Actor‑Critic（SAC）算法，以克服Wi‑Fi网络的部分可观测性和非马尔可夫动态；③ 在实验中展示该方法相较于传统分析和纯SAC、DDPG方案在吞吐量、延迟与公平性上的显著提升。

**🔧 技术方法**

使用的技术主要有：LSTM网络用于历史观测编码；Soft Actor‑Critic（SAC）强化学习框架；Bianchi马尔可夫模型推导吞吐量；Python+PyTorch实现RL训练；Matlab事件驱动Wi‑Fi模拟器用于网络环境仿真。

**📊 数据集**

实验数据来源为仿真数据，网络规模从10至15台STA、两条链路（2.4 GHz和5 GHz），采用Saturated traffic、固定信道参数；未使用公开数据集。

**📈 对比分析**

与三种基线方案对比：LSTM‑SAC（本研究）、LSTM‑SAC w/o CW（仅调度流量）、SAC（无LSTM）、LSTM‑DDPG（无熵正则）。实验结果显示LSTM‑SAC在吞吐量上提升约6–40%（相对SAC），公平性略低于w/o CW但仍高于其他方案；访问延迟显著下降；推理时间保持在毫秒级，兼顾实时性。

**⚠️ 局限性**

局限性包括：仅考虑STR模式（独立链路），未涵盖NSTR或混合模式；实验仅在仿真环境中验证，缺乏真实硬件测试；假设载波感知无误差且所有STA均为MLO设备；模型在极端高负载或与传统设备共存时的表现未知。

---

## 200. SG-CoT: An Ambiguity-Aware Robotic Planning Framework using Scene Graph Representations

**arXiv ID:** 2603.18271 | [PDF](https://arxiv.org/pdf/2603.18271v1)

**作者:** Akshat Rana `[一作]` (Netaji Subhas University of Technology), Amarjit Malhotra `[通讯]` (Netaji Subhas University of Technology)

**通讯引用:** 43 | [OpenAlex ID](https://openalex.org/A5074562646)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于场景图和链式思考的机器人规划框架SG-CoT。

**💡 创新点**

创新点在于将LLM与可检索的场景图结构结合，允许LLM在多轮推理中主动查询环境信息并识别并澄清歧义。

**🔧 技术方法**

技术上结合了LLM、VLM（如Qwen3-VL-2B、Gemini-2.5-Flash）、场景图生成、检索函数调用与迭代推理。

**📊 数据集**

使用了SayCan桌面环境和LEMMA多机器人基准作为实验数据集。

**📈 对比分析**

与CLARA、Inner Monologue、ProgPrompt等基线对比，SG-CoT在单机和多机环境下分别提升了4%和15%的成功率，并且正确澄清问题率提升至少10%。

**⚠️ 局限性**

局限性包括依赖VLM生成场景图易出现幻觉、令token使用和推理延迟随歧义数量增加、以及实验仅在仿真环境中验证。

---

## 201. Central Triangulation under Parallel Flip Operations: The CG:SHOP Challenge 2026

**arXiv ID:** 2603.18812 | [PDF](https://arxiv.org/pdf/2603.18812v1)

**作者:** Oswin Aichholzer `[一作]` (Institute of Algorithms and Theory), Stefan Schirra `[通讯]` (OvGU Magdeburg)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织并评测2026 CG:SHOP挑战赛，提出“平行翻转下的中心三角剖分”问题，发布了250个规模从15到12500点、2到200个三角剖分的基准实例，收集并比较各参赛队伍提交的解法。

**💡 创新点**

首次将平行翻转距离作为优化目标，构建大规模、结构多样化的基准集，并引入以总得分为依据的分数机制，使实验结果可在实际计算环境下直接比较。

**🔧 技术方法**

参赛队伍使用的核心技术包括：SAT/MaxSAT求解、贪心与大邻域搜索、局部改进、模拟退火、启发式构造与改进相结合的混合算法；Shadoks团队还利用SAT求解得到精确解后进行多种改进。

**📊 数据集**

使用三类数据集：random（基于TSPLIB随机抽样并做翻转walk生成）、woc（多目标优化得到的多样三角剖分）以及rirs（大规模随机点集、每个三角剖分独立生成），共250个实例。

**📈 对比分析**

比较方法：按实例排名给分（40/32/25/…）求总分，并计算与最优解的平均比例；结果显示Shadoks团队在99.92%实例获得最佳或次佳解，ETH Flippers第二，其他团队在rirs类上差距最大，平均差距约12–16%。

**⚠️ 局限性**

限制：对极大实例（>2000点）仍需较长时间；部分最优解尚无近似保证；参赛队伍多为少数几支，缺乏更广泛的算法对比；数据集虽大但仍主要集中在平面点集，缺少更复杂约束情形。

---

## 202. Transformers Learn Robust In-Context Regression under Distributional Uncertainty

**arXiv ID:** 2603.18564 | [PDF](https://arxiv.org/pdf/2603.18564v1)

**作者:** Hoang T. H. Cao `[一作]`, Lan V. Truong `[通讯]` (Ho Chi Minh University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在不同分布假设下（系数、特征、噪声）系统性地评估Transformer在线性回归中的上下文学习能力，并与经典估计器（OLS、Ridge、L1回归）进行对比。

**💡 创新点**

创新点在于：①将Transformer作为分布感知的元估计器，在非高斯、非i.i.d.环境下自动适应不同的噪声和参数先验；②通过大量实验展示Transformer在多种分布偏移下匹配或优于传统最优估计器，揭示其鲁棒性；③提出了基于上下文长度与维度关系的细粒度评估框架。

**🔧 技术方法**

技术手段主要是：Transformer模型的前向推理实现上下文学习；对不同分布（Laplace、Exponential、Uniform hypersphere系数；Gamma、VAR(1)特征；Bernoulli、Exponential、Gamma、Poisson、Student‑t噪声）进行数据生成；对比使用经典OLS、Ridge、L1（LP、ADMM）等基线，并以MSE/归一化过度损失作为评估指标。

**📊 数据集**

使用的数据集为合成数据：在每个实验中随机采样参数向量w、特征矩阵X和噪声ε，控制SNR并固定维度d和上下文长度k，覆盖从低维到高维、从小样本到过采样的多种情形。

**📈 对比分析**

比较方法：将Transformer与对应分布下的ML最优基线（如Bernoulli/Exponential对应L1）以及子最优基线（OLS、Ridge）在相同训练/测试集上进行对比；性能表现为Transformer在大多数非高斯/重尾噪声和非i.i.d.特征设置下均能与ML基线持平或超越，尤其在极端重尾（Student‑t）和非对称噪声（Exponential）场景中显著优于传统估计器。

**⚠️ 局限性**

局限性包括：①实验仅覆盖线性回归，未知其在非线性任务中的可推广性；②Transformer推理的计算复杂度为O(n²)（n为上下文长度），相对经典估计器显著更昂贵；③虽然显示了鲁棒性，但未揭示Transformer在上下文中执行的具体算法机制；④在部分分布（如极端高维稀疏）下仍存在性能波动。

---

## 203. Training-Only Heterogeneous Image-Patch-Text Graph Supervision for Advancing Few-Shot Learning Adapters

**arXiv ID:** 2603.18101 | [PDF](https://arxiv.org/pdf/2603.18101v1)

**作者:** Mohammed Rahman Sherif Khan Mohammad `[一作]` (Edge Hill University), Amr Ahmed `[通讯]` (Edge Hill University)

**通讯引用:** 8025 | [OpenAlex ID](https://openalex.org/A5009154893)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种训练阶段仅使用的异构图教师 TOGA，利用多尺度图像块与文本的图结构以及 Modality‑aware Graph Transformer，将细粒度跨模态关系知识注入 Tip‑Adapter 的缓存，从而显著提升少样本视觉‑语言模型的适配性能。

**💡 创新点**

创新点在于：①异构图教师仅在训练期间存在，保持测试时零额外开销；②使用多尺度图块和文本构建的统一图，并通过 MGT 深度跨模态推理；③双目标协同训练（交叉熵 + Focal Loss）实现教师向学生缓存的高效蒸馏。

**🔧 技术方法**

技术手段包括：多尺度图像块提取、文本提示投影、Modality‑aware Graph Transformer、Top‑N 节点过滤、焦点损失以及 Tip‑Adapter 的键值缓存蒸馏。

**📊 数据集**

实验覆盖 11 个公开基准（FGVC‑Aircraft、Flowers102、SUN397、Food101、CalTech101、UCF101、StanfordCars、DTD、ImageNet、OxfordPets、EuroSAT），在 1–16‑shot 的少样本设置下评估。

**📈 对比分析**

与 Tip‑Adapter、GraphAdapter、CLIP‑Adapter 等现有轻量和重量 PEFT 方法对比，TOGA 在所有 shot 级别均取得最高准确率，尤其 1–4 shot 上提升 5–9%，在 ImageNet 及其 OOD 变体中获得 63.1% 的平均表现，显著优于前沿方法。

**⚠️ 局限性**

局限性在于：训练阶段需要构建大型多尺度图和 MGT，导致显著的计算和内存开销；此外对极度稀缺样本或大规模任务的可扩展性尚未进行充分验证。

---

## 204. FASTER: Rethinking Real-Time Flow VLAs

**arXiv ID:** 2603.19199 | [PDF](https://arxiv.org/pdf/2603.19199v1)

**作者:** Yuxiang Lu `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**通讯引用:** 34563 | [OpenAlex ID](https://openalex.org/A5078109015)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 FAESTER 方法，使用 Horizon‑Aware Schedule 与流式客户端‑服务器交互，显著降低动作块首步生成时间（TTFA），提升流式实时机器人控制的响应速度。

**💡 创新点**

创新点包括：
- Horizon‑Aware Schedule（HAS）在动作时序上自适应分配采样步数，使近端动作仅需单步完成；
- 混合调度（mixed schedule）辅助模型在微调时兼容恒定与索引相关的时间表；
- 流式输出与早停（early‑stop）策略，实现即刻将已完成动作送给机器人，同时继续后续动作的细化；
- 方案无需改动网络结构或额外训练成本，兼容现有 flow‑based VLA。

**🔧 技术方法**

技术手段：
- Flow‑matching VLA（π₀.₅、X‑VLA）
- Horizon‑Aware Schedule 与混合调度
- 流式客户端‑服务器接口与早停机制
- 通过 TTFA/TTFT 指标衡量即时反应性能。

**📊 数据集**

数据集与实验：
- 真实机器人任务（乒乓球、拿饮料、折毛巾）约 14 分钟演示数据；
- 公开仿真基准 LIBERO 与 CALVIN，评估模型在无实时约束下的性能。

**📈 对比分析**

比较方法与性能：
- 与同步、异步（Naive Async）、Training‑time RTC 等基线对比；
- 在 RTX 4090 / RTX 4060 上，TTFA 下降 800–900×；
- 反应时间分布概率上，FAESTER 在所有场景均优于基线，尤其在资源受限 4060 上优势更明显；
- 乒乓球任务得分最高，任务完成时间大幅缩短；
- 在仿真基准上，HAS 仅略微影响长程轨迹，保持竞争力。

**⚠️ 局限性**

局限性：
- 加速采样可能略微降低长程轨迹的精确度；
- 方案仍受 GPU 推理速度限制，极端低算力设备反应提升有限；
- 对于极长时间延迟或复杂动态环境，仍需进一步改进采样策略。

---

## 205. What We Talk About When We Talk About Frameworks in HCI

**arXiv ID:** 2603.18950 | [PDF](https://arxiv.org/pdf/2603.18950v1)

**作者:** Shitao Fang `[一作]` (University of Tokyo), Kasper Hornbæk `[通讯]` (University of Copenhagen)

**通讯引用:** 12424 | [OpenAlex ID](https://openalex.org/A5015615908)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性回顾2015-2024年CHI会议论文中关于“framework”的使用与功能，构建了框架生命周期模型和功能范式；

**💡 创新点**

首次将框架使用划分为六种参与类型（Create、Adapt、Validate、Review、Use、Mention）并揭示其功能、构成与验证缺口；

**🔧 技术方法**

采用系统综述方法、PRISMA流程、定性编码与定量统计（Kappa、频次分析、UMAP可视化）等多种技术；

**📊 数据集**

对615篇CHI论文（2015-2024年）进行文本抽取与元数据归档，形成研究样本；

**📈 对比分析**

通过对比框架参与类型比例、功能归类及验证方式，发现“Create”占比过高、Validate和Review稀缺，揭示社区实践不平衡；

**⚠️ 局限性**

研究仅覆盖CHI文献、作者自述视角、缺乏使用者与评审者反馈，无法完整评估框架真实影响与可复现性。

---

## 206. Let's Play Tag: Linear Time Evaluation of Conjunctive Queries under TGD Constraints

**arXiv ID:** 2603.18709 | [PDF](https://arxiv.org/pdf/2603.18709v1)

**作者:** Nofar Carmeli `[一作]` (Inria), Marcin Przybyłko `[通讯]` (Warsaw University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文研究了在满足元组生成依赖（TGDs）约束下，对连接查询（CQ）在多种评估模式（单测试、全测试、计数、字典序直接访问、枚举）下的线性时间可行性与不可行性，并提出了一种“标记”技术，将受约束的数据库映射到无约束的自连接自由查询上，从而将已知的无约束分类迁移到受约束情形。

**💡 创新点**

创新点在于：①提出了标记友好（tagging‑friendly）TGDs的概念，并证明在非递归或前沿受限全TGDs下均满足；②利用标记+洪水与过滤技术构造满足约束的数据库，实现了对多种评估模式的分类提升；③在枚举模式下发现了自连接引入的复杂性现象，并给出若干开放实例，展示了此模式的挑战。

**🔧 技术方法**

主要技术包括：标记（为事实附加查询变量的配对）、Skolem chase、洪水与过滤构造、前沿受限（frontier‑guarded）与非递归 TGDs 的属性利用、颜色化（coloring）技巧、以及对计数与直接访问的前缀约束变换。算法多采用线性预处理与常数/多项式延迟的策略。

**📊 数据集**

本文主要为理论研究，未使用真实数据集；所有实验与评估均基于理论构造的数据库和图结构，采用假设（如超三角形假设、BMM、SETH）来证明上界与下界。

**📈 对比分析**

与先前无约束情形的分类相比，标记友好 TGDs 下的分类保持相同的“弱循环/自由连通”分界点；对于计数和直接访问，保持了原先的线性/对数时间界；在枚举模式中，除了一些特殊约束类（如二元关系或单元头）外，尚未给出完整比较，表明该模式更难以统一分析。

**⚠️ 局限性**

主要限制在于：①对枚举模式的分类仍不完整，尤其在自连接存在时出现未知案例；②对更一般的递归或高基数 TGDs 的分析不足；③对非递归 TGDs 不满足标记友好条件的情况仍未解决；④部分结果仅在假设（如超三角形假设、VUTD）下成立，真实复杂度未知。

---

## 207. Holistic Energy Performance Management: Enablers, Capabilities, and Features

**arXiv ID:** 2603.18841 | [PDF](https://arxiv.org/pdf/2603.18841v1)

**作者:** Meysam Masoudi `[一作]` (Ericsson AB), Pal Frenger `[通讯]` (Ericsson AB)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了5G/6G网络能耗管理，提出硬件能力与功能的分类并设计了功能编排器

**💡 创新点**

创新点在于将创建空闲窗口的功能与利用空闲窗口的深度睡眠等功能协同编排，实现显著能耗降低

**🔧 技术方法**

采用3GPP对齐的事件驱动仿真器、lean‑NR信令调度、流量门控以及ASM深度睡眠等技术

**📊 数据集**

使用3GPP仿真交通模式（低/轻/中负载）作为数据集

**📈 对比分析**

与默认信令周期、单独扩展信令、仅微睡眠等基准对比，结果显示在低/轻负载下可实现最高58%的能耗降低，吞吐量几乎无损；在中负载下仍有能耗提升但吞吐量下降

**⚠️ 局限性**

局限在于依赖完美的短期空闲预测、缺乏细粒度能耗观测、未在真实网络环境中验证，并且中负载下吞吐量下降等问题

---

## 208. VEPO: Variable Entropy Policy Optimization for Low-Resource Language Foundation Models

**arXiv ID:** 2603.19152 | [PDF](https://arxiv.org/pdf/2603.19152v1)

**作者:** Chonghan Liu `[一作]` (Qiyuan Tech), Xiangzheng Zhang `[通讯]` (Qiyuan Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在低资源语言场景下，提升大语言模型的翻译与指令遵循性能，采用可变熵策略优化与可验证奖励（RLVR）的结合，实现结构约束与语义自然的统一；

**💡 创新点**

提出可变熵策略优化（VEPO），通过温度一致比率、位置感知熵调度、轨迹过滤、优势归一化等机制动态平衡字面忠实与语义自然；

**🔧 技术方法**

利用强化学习+可验证奖励、熵正则化、温度一致采样、熵调度、优势归一化、剪切损失、tokenizer扩展与持续预训练等技术；

**📊 数据集**

使用 FLORES‑200（90 方向）、COMET‑22、chrF、BBH、CMMLU、HellaSwag、MMLU 等公开数据集，并在 7M 混合翻译与指令数据上进行微调；

**📈 对比分析**

与 PPO、GRPO、DAPO、RLOO、Reinforce++ 等 RL 基线以及现有多语 LLM（Gemma、Qwen、Apertus 等）和专用翻译模型对比；在 FLORES‑200 上平均 BLEU 24.9、COMET 0.881/0.882，接近 Google Translate，明显优于同类 7B 模型；

**⚠️ 局限性**

仍受奖励模型粒度、极低资源或高歧义语言表现有限、RL 训练成本高、长句子长度漂移等限制，需要进一步改进。

---

## 209. Correlation-Weighted Multi-Reward Optimization for Compositional Generation

**arXiv ID:** 2603.18528 | [PDF](https://arxiv.org/pdf/2603.18528v1)

**作者:** Jungmyung Wi `[一作]` (Korea University), Donghyun Kim `[通讯]` (Korea University)

**通讯引用:** 21338 | [OpenAlex ID](https://openalex.org/A5100719069)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于概念相关性加权的多奖励优化框架，用于提升文本到图像模型在多概念组合生成中的准确性和一致性。

**💡 创新点**

创新点在于：①将多概念提示拆解为细粒度对象、属性、数量和空间关系奖励；②通过在一组生成样本中计算奖励的Pearson相关性，自动估计各概念的生成难度；③使用相关性得分动态重加权奖励，从而在训练过程中重点强化难以同时满足的概念，缓解传统奖励聚合导致的概念冲突。

**🔧 技术方法**

技术要点包括：混合ODE‑SDE采样（MixGRPO）、奖励解耦归一化（GDPO）、基于分割（SAM）和CLIP的对象/属性检测、Depth Anything进行深度关系评估、软最大函数实现动态权重分配、并在Stable Diffusion 3.5与FLUX.1‑dev上使用LoRA微调。

**📊 数据集**

主要使用的训练数据是5k个多概念提示（最多8个概念/实体）与2.5k单属性提示，采用ConceptMix生成流程；评估数据集包括ConceptMix、GenEval 2和T2I‑CompBench。

**📈 对比分析**

与现有RL对齐方法（FlowGRPO、PrefGRPO、IterComp等）以及大模型Qwen‑Image比较，实验显示在所有基准上都获得显著提升：ConceptMix全分数提升至0.8410，GenEval 2 TIFA_AM/GM分别为80.0/34.0，T2I‑CompBench总体平均分提升至0.5961（FLUX.1‑dev）或0.6141（Stable Diffusion 3.5）。

**⚠️ 局限性**

局限性包括：①依赖奖励模型和分割/深度模型的准确性，错误会直接影响奖励信号；②相关性估计需要多张样本，计算成本相对较高；③方法在极端多概念或非常细粒度关系上仍可能存在欠拟合，且在跨域提示生成时的鲁棒性待进一步验证。

---

## 210. Why Better Cross-Lingual Alignment Fails for Better Cross-Lingual Transfer: Case of Encoders

**arXiv ID:** 2603.18863 | [PDF](https://arxiv.org/pdf/2603.18863v1)

**作者:** Yana Veitsman `[一作]` (University of Göttingen), Hinrich Schütze `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了跨语言对齐技术对多语言Encoder（XLM‑R）在不同任务（POS标注、句子分类）中的迁移效果，并分析了对齐目标与任务目标之间的梯度冲突；

**💡 创新点**

发现传统的嵌入相似度、召回率并不能可靠预测下游任务表现，而梯度相似度是更有意义的指标；

**🔧 技术方法**

使用HiCTL对齐框架的多层对比损失（词级、句子级、MLM），并在不同语言对上做对齐训练；

**📊 数据集**

对齐数据集为OPUS与MultiCCAligned，包含德语-英语、英语-哈萨克语、日语-哈萨克语、韩语-哈萨克语四对语言；下游任务数据为Universal Dependencies（POS）和SIB‑200（句子分类）；

**📈 对比分析**

与基线（无对齐）比较，线性层微调（仅更新分类层）通常比全模型微调更稳健；不同对齐配置对性能影响差异显著，某些配置甚至导致性能下降；整体提升幅度在0.0~0.01之间；

**⚠️ 局限性**

仅针对XLM‑R单一Encoder架构，未评估其他模型；仅使用一种对比目标，缺乏对多种对齐方法的比较；下游任务范围有限，未验证在更复杂任务上的普适性；

---

## 211. Dream the Dream: Futuring Communication between LGBTQ+ and Cisgender Groups in Metaverse

**arXiv ID:** 2603.18578 | [PDF](https://arxiv.org/pdf/2603.18578v1)

**作者:** Anqi Wang `[一作]` (Hong Kong University of Science and Technology), Pan Hui `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 21303 | [OpenAlex ID](https://openalex.org/A5029925982)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过三场联合 LGBTQ+ 与顺性别参与者的未来设计工作坊，探讨如何利用元宇宙的空间与技术优势，构建包容性的跨群体沟通环境；

**💡 创新点**

创新点在于将“未来设计”方法与“权力几何”理论结合，提出跨层级（活动、互动、场景、空间）的社会-空间-技术解决方案，强调动态亲密度、可自定义身份表达与协作共建等机制；

**🔧 技术方法**

主要技术手段包括：参与式设计、未来工作坊、共创草图与情景模拟、亲和图分析、主题编码；在 VR 体验环节使用 Mozilla Hubs 与 Decentraland 的 Pride 主题场景；

**📊 数据集**

使用的数据集为 18 名自我认同为 LGBTQ+ 或顺性别的参与者（涵盖多种性取向、性别身份与 VR 经验水平），并收集了工作坊记录、笔记与草图；

**📈 对比分析**

研究未进行量化对比或性能评估，而是基于质性分析和主题梳理呈现设计方案与启示；

**⚠️ 局限性**

局限性包括样本规模小、受访者教育与经济水平相对集中、仅在中国语境下开展，缺乏跨文化与大规模验证，且设计方案缺乏实现细节与评估。

---

## 212. Contact Status Recognition and Slip Detection with a Bio-inspired Tactile Hand

**arXiv ID:** 2603.18370 | [PDF](https://arxiv.org/pdf/2603.18370v1)

**作者:** Chengxiao He `[一作]` (Southeast University), Longhui Qin `[通讯]` (Southeast University)

**通讯引用:** 518 | [OpenAlex ID](https://openalex.org/A5062389686)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一只五指仿生触觉手，采用24路多模态触觉通道进行接触状态识别与滑移检测。

**💡 创新点**

创新点包括：①将滑移检测问题转化为接触状态识别并结合分箱技术；②使用DWT+FFT提取时频特征并通过方差估计筛选120个最优特征；③在五指触觉手中同时布置了压阻（静态）和压电（动态）传感器，实现对静态与动态触觉信息的双重感知。

**🔧 技术方法**

技术手段：Savitzky‑Golay平滑滤波、信号分箱、离散小波变换与FFT、方差估计特征筛选、极限学习机（极化核ELM）分类。

**📊 数据集**

数据集：504次实验（6种材料×3种速度×28次循环），以及4种未见材料（木材、铁、铜、布）用于泛化测试。

**📈 对比分析**

与传统阈值法和基于神经网络的学习法相比，所提出方法在已训练材料上实现了96.39%的识别精度，在未见材料上保持91.95%，说明具有良好的泛化能力，且相较阈值法对传感器噪声更鲁棒。

**⚠️ 局限性**

局限性：接触状态识别仍存在误判（尤其是非滑移状态），导致滑移检测精度未达到100%；未对未见材料进行重新训练，模型对极端材质的适应性仍有限；滑移检测仅基于状态转移，未引入滑移持续时间或概率模型。

---

## 213. Revisiting Label Inference Attacks in Vertical Federated Learning: Why They Are Vulnerable and How to Defend

**arXiv ID:** 2603.18680 | [PDF](https://arxiv.org/pdf/2603.18680v1)

**作者:** Yige Liu `[一作]` (Peking University), Hanpin Wang `[通讯]` (Peking University)

**通讯引用:** 493 | [OpenAlex ID](https://openalex.org/A5079106687)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对垂直联邦学习中的标签推断攻击（LIA）进行重新评估，发现底层模型主要负责特征提取，顶部模型承担标签映射，并通过信息论证明层深度越大与标签相关的信息越多；利用任务重分配实验揭示现有LIA方法对特征-标签天然对齐的高度依赖，从而暴露其脆弱性；提出一种零开销的防御策略，即将切分层提前（增加顶部模型比例），显著降低攻击准确率并提升整体预测性能；并验证该策略可增强其他已有防御手段的效果。

**💡 创新点**

创新点包括：1）首次用互信息理论说明VFL中底层模型对标签信息的低捕获能力和顶部模型的补偿作用；2）通过任务重分配揭示现有LIA方法对特征-标签对齐的依赖，证明其攻击“幻觉”；3）提出“切分层前移”这一轻量化、零开销的防御方案，兼具攻击抑制与性能提升；4）系统验证该方案对多模型、多数据集、多攻击方式及多防御组合的稳健性。

**🔧 技术方法**

技术手段包括：互信息量化与理论证明（信息瓶颈框架）；模型补偿现象与切分层操作；任务重分配实验设计；两类主流LIA（聚类、模型补全）与梯度攻击评估；多种防御策略（差分隐私、梯度裁剪、梯度压缩、混淆自编码器、随机标签扩展）的结合与性能对比。

**📊 数据集**

使用了五个公开数据集：MNIST、CIFAR‑10、CINIC‑10、SVHN、GTSRB；分别采用 MLP、AlexNet、ResNet、LeNet、CNN 等常见网络架构。

**📈 对比分析**

通过在相同训练设置下比较原始切分层与前移后切分层的攻击准确率（LIA cluster 与 LIA completion）以及模型任务准确率（MTA），结果显示：前移 1 层即可使攻击准确率降低 40‑70%，3 层几乎达到随机猜测水平；同时 MTA 提升 1‑5% 甚至更高；在与差分隐私、梯度裁剪、压缩、CAE、RLE 等防御结合时，切分层前移进一步降低攻击准确率，表现优于或与最强防御相当。

**⚠️ 局限性**

局限性包括：1）过度前移切分层可能导致底层模型输出保留过多原始特征，增加特征推断攻击风险；2）实验集中在 VFL 的标签推断任务，对其他类型攻击（如特征泄露、模型逆向）未做充分验证；3）防御效果受模型深度、宽度、任务分配方式等因素影响，需在更广泛场景中进一步评估；4）未对训练时间与资源消耗做系统衡量，虽然理论上为零开销，但实际部署时仍需考虑模型切分与通信延迟。

---

## 214. Optimal Path Planning in Hostile Environments

**arXiv ID:** 2603.18958 | [PDF](https://arxiv.org/pdf/2603.18958v1)

**作者:** Andrzej Kaczmarczyk `[一作]` (Czech Technical University in Prague), Haifeng Xu `[通讯]` (University of Chicago)

**通讯引用:** 2048 | [OpenAlex ID](https://openalex.org/A5100731914)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种全新的多代理路径规划模型：在图形化的敌对环境中，代理（资产）从公共起点出发，需要在经过可能被陷阱消灭且具有已知冷却时间的节点时，尽可能多地到达公共终点。

**💡 创新点**

创新点包括：①将陷阱的“激活‑冷却”机制引入多代理路径规划；②在确定性、完全可观测的离散时间设置下，首次证明了该问题在 PSPACE 之内；③识别出若干多态可解子问题（如路径、相互独立的路径集合、星形树等）并给出多项式时间算法；④构造了复杂度证明，证明即使在树（最大度数3）或每个代理仅经过至多6个陷阱的限制下，该问题仍为 NP‑难，且不存在多项式时间的常数因子逼近。

**🔧 技术方法**

主要技术手段包括：①构造“snapshots”（快照）并将规划过程映射为二色图的重构问题；②利用已知的二色图重构步数上界证明存在多项式长度最优计划；③设计 run‑wait‑sacrifice 策略并用严格的引理证明其最优性；④对树和路径的可解性使用动态规划与贪心等经典算法；⑤通过构造“batch gadget”等硬件设计进行 NP‑难性证明。

**📊 数据集**

本工作为理论性研究，未使用实验数据集；所有结果均为理论证明与算法分析。

**📈 对比分析**

相比于已有的 MAPF、追捕‑躲避、令牌重组等相关模型，本文在同类图结构（如路径、树、星形）下实现了多项式时间求解；同时通过 NP‑难性与逼近不可行性证明，明确了问题的计算边界，展示了在最优与可计算性之间的权衡。

**⚠️ 局限性**

局限性包括：①假设环境完全已知且陷阱行为完全确定，缺乏对不确定性与信息不完整的处理；②虽然给出多项式可解子类，但在一般图形或树形（最大度3）下仍为 NP‑难，且无法提供常数因子逼近；③未给出实用的启发式或近似算法，需在未来工作中探索。

---

## 215. Measuring 3D Spatial Geometric Consistency in Dynamic Generated Videos

**arXiv ID:** 2603.19048 | [PDF](https://arxiv.org/pdf/2603.19048v1)

**作者:** Weijia Dou `[一作]` (Tongji University), Jiwen Lu `[通讯]` (Tsinghua University)

**通讯引用:** 28998 | [OpenAlex ID](https://openalex.org/A5100460385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为 SGC 的 3D 空间几何一致性度量，用于评估动态生成视频的几何稳定性。

**💡 创新点**

创新点在于先将静态背景与动态前景分离，然后通过深度信息和局部相机位姿估计（PnP）获取多个子区域的相机变换，最终用这些位姿之间的方差量化几何不一致。

**🔧 技术方法**

利用 VGGT 提取全局相机姿态与内参、Video Depth Anything 估计稠密深度、DelTA 进行像素跟踪、基于深度的一维聚类分割子区域，并通过 PnP 估计局部位姿，最后用 PCA 加权聚合多个子指标得到最终 SGC 分数。

**📊 数据集**

在 1,296 条视频上进行评估，包含 1,196 条 GenWorld 合成视频、100 条 OpenVidHD‑0.4M 高运动视频以及 300 条真实视频（nuScenes、RT‑1、OpenVid）。

**📈 对比分析**

与 MEt3R、VBench、FVD、FVMD 等现有指标对比，SGC 能够发现高 FVD 但几何不稳的模型，显示出更符合人眼对几何一致性的评判，生成模型在 SGC 上的平均分数明显高于真实数据的下界。

**⚠️ 局限性**

局限性包括对深度估计与相机姿态估计的依赖，若生成视频的深度或相机信息不准确会导致误判；在摄像头运动幅度极大或动态场景复杂时，局部位姿估计的鲁棒性仍需进一步提升。

---

## 216. Total Recall QA: A Verifiable Evaluation Suite for Deep Research Agents

**arXiv ID:** 2603.18516 | [PDF](https://arxiv.org/pdf/2603.18516v1)

**作者:** Mahta Rafiee `[一作]` (University of Massachusetts Amherst), Hamed Zamani `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 14811 | [OpenAlex ID](https://openalex.org/A5100618738)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究者提出了“总召回问答（Total Recall QA）”任务，构建了可自动化生成的评测框架，用于评估深度研究代理（DRA）在多源信息检索与推理上的表现。

**💡 创新点**

创新点在于：①将知识库与文本语料一一对应，利用实体集生成需要全召回检索和聚合推理的查询；②通过LLM自动生成查询与段落级相关性标注，保证评测的可复现和可验证；③在评测中加入可测量的中间检索回报与最终答案准确度。

**🔧 技术方法**

采用的技术包括基于实体中心的查询生成（使用SPARQL与LLM）、文本分块与检索（BM25、SPLADE++、Contriever、E5等）、深度学习检索重排序（Contriever+SPLADE++）以及三种深度研究代理（ReAct、Search-o1、Search-R1）与多种LLM（Qwen2.5-7B、GPT‑5.2等）。

**📊 数据集**

数据集方面使用了WikiData–Wikipedia对齐语料生成的两个子集（完整实体集与QALD/QUEST实体列表）以及一个基于MAVE的合成电商知识库与博客语料，三者均包含数千个需要全召回检索的数值答案问题。

**📈 对比分析**

在检索层面，SPLADE++ 在所有数据集上获得最高召回率；在端到端评测中，GPT‑5.2 在 Wiki 数据集上表现最好，但在合成电商数据上仍低于强大的 DRA；Oracle 检索提升显著，但仍有约90% 的错误来自推理失败，整体性能仍远低于理想。

**⚠️ 局限性**

主要局限包括：检索回报远未达到 100%，深度研究代理的多轮查询策略不够有效；LLM 在面临缺失训练信息时的推理能力不足；以及评测依赖 LLM 自动标注，可能引入标注噪声。

---

## 217. TiBCLaG: A Trigger-induced Bistable Compliant Laparoscopic Grasper

**arXiv ID:** 2603.18559 | [PDF](https://arxiv.org/pdf/2603.18559v1)

**作者:** Joel J Nellikkunnel `[一作]`, Prabhat Kumar `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

设计并验证了一种触发器诱导双稳态可顺应性腹腔镜抓取器TiBCLaG。

**💡 创新点**

采用单体全可顺应结构消除多连杆，结合触发器实现无持续输入的双稳态抓取，并首次将两元梁约束模型（TEBC）与多触发器实现相结合。

**🔧 技术方法**

使用两元梁约束模型（TEBC）、非线性有限元分析、FDM 3D 打印以及实验测量等技术。

**📊 数据集**

未使用传统数据集，而是通过实验获得的拉伸位移、力-位移曲线和有限元仿真结果。

**📈 对比分析**

将实验测得的抓手位移与有限元预测值对比，发现小变形区间内误差低，最大位移略低；激活所需力约为17–20 N，基本符合设计目标。

**⚠️ 局限性**

受限于平面化设计导致的出射平面变形、非线性力反馈以及冲击式能量释放，且TEBC模型对单梁的简化假设导致设计偏差。

---

## 218. Modeling the Impacts of Swipe Delay on User Quality of Experience in Short Video Streaming

**arXiv ID:** 2603.18575 | [PDF](https://arxiv.org/pdf/2603.18575v1)

**作者:** Duc V. Nguyen `[一作]`, Huyen T. T. Tran `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究短视频流中滑动延迟对用户体验（QoE）的影响，并提出了针对该场景的 QoE 预测模型。

**💡 创新点**

创新点在于首次系统性评估滑动延迟（持续时间、次数、时机）对 QoE 的影响，并将这些因素融入到指数衰减权重的 QoE 预测公式中。

**🔧 技术方法**

使用主观质量评估（ACR 5 级评分）收集 MOS，并用回归分析和指数时间衰减方法构建预测模型。

**📊 数据集**

采用公开的 132 个滑动延迟模式数据集，覆盖从 0.25 秒到 16 秒的延迟持续时间，1~4 次延迟以及不同时机的组合。

**📈 对比分析**

与六个传统 QoE 模型（Kooij、Hoßfeld、Tran、CQM、P.1203.3、OLS Cat）在相同数据集上对比，模型取得 RMSE 0.279、PCC 0.953、SROCC 0.949，显著优于对照模型。

**⚠️ 局限性**

局限性在于仅考虑滑动延迟，未纳入视频质量、内容属性、设备差异等其它可能影响 QoE 的因素。

---

## 219. CNT: Safety-oriented Function Reuse across LLMs via Cross-Model Neuron Transfer

**arXiv ID:** 2603.18449 | [PDF](https://arxiv.org/pdf/2603.18449v1)

**作者:** Yue Zhao `[一作]` (Institute of Information Engineering), Wangjun Zhang `[通讯]` (Guangzhou University)

**通讯引用:** 127 | [OpenAlex ID](https://openalex.org/A5012806139)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种跨模型神经元迁移（CNT）方法，能够在不进行额外训练的情况下，通过将少量神经元从开放源 LLM 捐赠模型迁移到目标模型，实现安全功能的增删；

**💡 创新点**

创新点包括：①在神经元层面实现功能级的增删，兼顾功能添加与删除；②引入低噪声探测请求配对与路径积分功能差归因方法，精准定位对目标功能贡献最大的权重；③提出神经元迁移拒绝率（NTRR）用于捐赠模型筛选，保证兼容性；④通过二分搜索控制迁移率，平衡功能效果与整体性能。

**🔧 技术方法**

技术手段主要包括：神经元级别权重替换、低噪声请求构造（F‑req/Fl‑req）、路径积分归因（对Δθ沿线积分得到 A_i）、NTRR 兼容性评估、二分搜索确定转移比例，以及对多层、混合专家（MoE）架构的适配。

**📊 数据集**

使用的数据集涵盖：HarmfulQA、HarmfulBehavior、CategoricalHarmfulQA、DangerousQA（用于功能探测与评估）；OpenHermes‑2.5、StereoSet（用于偏见消除实验）；MMLU、NQ‑Open（衡量整体语言能力）；Bias‑Bench、SS/ICAT（评估偏见消除效果）。

**📈 对比分析**

与 ED、RD、ActSVD‑OP（功能删除）以及 LoRA、Model Surgery（功能添加）等基线进行对比；CNT 在安全功能抑制上平均降低拒绝率 81.6%，提升有害率 76.8%/危害基准 77.1%，同时整体 MMLU 仅下降 0.1%；在功能添加中 RA 提升 44.7%/37% 与 LoRA、Model Surgery 相比提升 6–12%，且 MMLU 只微幅提升。

**⚠️ 局限性**

局限性：需要捐赠与受体模型同一架构且兼容性高；对捐赠模型的选择依赖 NTRR，若兼容性低效果不佳；层级敏感性导致某些层迁移效果差异大；低噪声请求需要足够多样化，受限于可用提示；未来针对 CNT 的防御机制尚未成熟。

---

## 220. Semantic Chameleon: Corpus-Dependent Poisoning Attacks and Defenses in RAG Systems

**arXiv ID:** 2603.18034 | [PDF](https://arxiv.org/pdf/2603.18034v1)

**作者:** Scott Thornton `[一作]` `[通讯]` (Independent Researcher), Scott Thornton (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了检索增强生成（RAG）系统中的梯度引导双文档毒化攻击，并验证了混合BM25+向量检索在抑制此类攻击中的效果。

**💡 创新点**

首次系统性评估混合检索在抵御梯度毒化攻击中的强大防御力，并揭示语料库特性对攻击可行性与检测难度的显著影响，同时提出基于查询模式差异（QPD）的行为检测方法。

**🔧 技术方法**

使用GCG梯度优化的双文档毒化、BM25+向量混合检索、QPD行为检测、跨模型LLM评估及多语料库实验等技术。

**📊 数据集**

实验数据集包括Security Stack Exchange 67,941条Q&A、FEVER Wikipedia 96,561篇文章以及156,777条网络安全厂商文档。

**📈 对比分析**

通过与纯向量检索、关键词注入以及联合稀疏+密集优化的对比，发现纯向量下梯度毒化成功率为38%，混合检索降为0%；联合优化在混合检索下仍可达20–44%成功；跨模型攻击成功率从46.7%（GPT‑5.3）到93.3%（Llama‑4），安全违规率从6.7%到93.3%。

**⚠️ 局限性**

仅评估了白盒梯度毒化，未测试黑盒搜索方法；语料库覆盖有限，未验证检索质量与安全性的权衡；温度设置差异可能影响模型比较；检测方法仅为探索性，缺乏大规模验证。

---

## 221. Hypothesis-Conditioned Query Rewriting for Decision-Useful Retrieval

**arXiv ID:** 2603.19008 | [PDF](https://arxiv.org/pdf/2603.19008v1)

**作者:** Hangeol Chang `[一作]` (KAIST), Jong Chul Ye `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出无训练的前检索框架 HCQR，先生成工作假设并重写查询，以更好地将检索与决策目标对齐，提升可选检索决策场景下的可检索增强生成效果。

**💡 创新点**

创新点在于使用工作假设驱动检索，生成三类目标查询（支持、区分、关键特征），并仅将假设用于检索规划而不暴露给生成器，从而实现检索与最终决策的紧密耦合。

**🔧 技术方法**

使用了结构化假设生成器、查询重写器、共享检索器（MedCPT）、文档融合技术，以及基于 LLM 的检索上下文效用评估。

**📊 数据集**

实验数据集包括 MedQA 与 MMLU‑Med 两个医学多选 QA 基准。

**📈 对比分析**

通过与无检索 CoT、Simple RAG、Rewriting、HyDE、Rerank‑RAG、MAIN‑RAG 等多种基线比较，HCQR 在所有四个模型配置下均获得最佳准确率，并在 MedQA 上平均提升 5.9 分、MMLU‑Med 上提升 3.6 分；同时显著提高检索上下文的决策效用率。

**⚠️ 局限性**

主要限制包括：依赖工作假设的准确性；实验仅限医学多选 QA，未验证其他领域或开放域；评估效用标签由 LLM 判定，缺乏人工标注；使用固定语料库和检索预算，可能不适用于更大或动态检索环境。

---

## 222. Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails

**arXiv ID:** 2603.18280 | [PDF](https://arxiv.org/pdf/2603.18280v1)

**作者:** Gregory N. Frank `[一作]` `[通讯]` (Independent Researcher), Gregory N. Frank (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究中文大语言模型在政治审查上的对齐机制，揭示检测-路由-输出三阶段结构

**💡 创新点**

发现概念检测易实现，路由是模型特异且易被外部操控，拒绝评估忽视了叙事引导控制

**🔧 技术方法**

使用线性探针、对比激活分析与一阶删减、方向余弦、自动与人工评判、交叉实验

**📊 数据集**

构建24条CCP敏感与对照提示（v1、v3、中文版本），120条安全提示，46模型大规模行为筛查

**📈 对比分析**

通过留一分类验证、方向可视化与实验删减，显示模型间路由差异显著，拒绝率下降但引导得分上升，性能与传统拒绝评估不同

**⚠️ 局限性**

限制包括：小样本行为不稳、方向估计依赖语料、不同模型对路由不可迁移、自动评判过度标注政治化、架构特异性（如Qwen3-8B的混淆）

---

## 223. Accurate and Efficient Multi-Channel Time Series Forecasting via Sparse Attention Mechanism

**arXiv ID:** 2603.18712 | [PDF](https://arxiv.org/pdf/2603.18712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 224. Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning

**arXiv ID:** 2603.18495 | [PDF](https://arxiv.org/pdf/2603.18495v1)

**作者:** Jooyoung Kim `[一作]` (Sungkyunkwan University), Honguk Woo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 877 | [OpenAlex ID](https://openalex.org/A5001227049)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于神经-符号对抗推理的框架，将视频演示转化为可执行的代码策略，并在演示与部署域存在感知和物理差异的情况下实现跨域自适应。

**💡 创新点**

创新点在于：1）构建可验证的符号世界模型，利用 VLM 抽取场景图并生成符号动作；2）通过对抗推理识别并修正跨域不一致的程序步骤；3）将符号检验与 VLM 生成的补丁相结合，形成闭环的可验证自适应过程。

**🔧 技术方法**

核心技术包括：视觉语言模型（VLM）用于场景图与符号动作生成；符号工具（如 VAL）进行符号推理与一致性验证；对抗推理（counterfactual reasoning）用于识别和修正跨域差异；代码生成模块将最终符号程序转译为可执行控制代码。

**📊 数据集**

实验数据集涵盖：1）在 Genesis 物理模拟平台上收集的 440 条桌面组织与物体重排任务，划分为低/中/高复杂度；2）在 Franka Emika Research 3 真实机器人上收集的人类演示视频和相应的部署任务；3）使用多种域因子（阻碍、物体可用性、运动学配置、抓手类型及其组合）生成跨域测试场景。

**📈 对比分析**

与六个最先进基线（Demo2Code、GPT4V-Robotics、Critic‑V、MoReVQA、Statler、LLM‑DM）在模拟与真实环境中对比。该框架在所有域因子下的任务成功率（SR）平均提升约 27.73%（相较 GPT4V-Robotics）和 24.77%（相较 Statler），并保持较低的程序偏差（PD）。在真实机器人实验中，SR 提升至 87.5%，明显优于 Demo2Code（差距 87.5%）和其他基线。

**⚠️ 局限性**

局限性：1）当演示与部署任务的复杂度差距过大时，单纯基于演示的对抗推理难以补足所需的新因子，性能显著下降；2）对 VLM 生成的符号动作的正确性高度依赖 VLM 的推理质量，易受 hallucination 影响；3）符号工具的可扩展性受限，需手工定义谓词与操作集合，难以覆盖极其丰富的真实世界场景。

---

## 225. Performance Testing of ChaCha20-Poly1305 for Internet of Things and Industrial Control System devices

**arXiv ID:** 2603.19150 | [PDF](https://arxiv.org/pdf/2603.19150v1)

**作者:** Kristján Orri Ragnarsson `[一作]` (Reykjavik University), Jacky Mallett `[通讯]` (Reykjavik University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

对 ChaCha20-Poly1305 在低功耗边缘设备（Raspberry Pi 4 与 Intel N95 Mini PC）上的加密与解密性能进行实验评估，并将测得的时延与工业控制系统协议（GOOSE 与 IEC 60834‑1）的实时时延限制进行对比。

**💡 创新点**

证明在当前硬件条件下，即使启用动态频率扩展，ChaCha20-Poly1305 的加密/解密时延仍远低于 GOOSE 与 IEC 60834‑1 的上限，显示出该算法在实时工业物联网环境中的可行性；同时提供了针对不同消息大小、频率锁定与动态频率模式的细粒度性能数据。

**🔧 技术方法**

使用 WolfSSL（v5.7.6）中实现的 ChaCha20-Poly1305；实验平台包括 Raspberry Pi 4 Model B（ARM64）和 Intel N95 Mini PC（x86‑64）；测量工具为 GNU Time 与自定义计时器，数据采样 100,000 次。

**📊 数据集**

无公开数据集，实验使用随机生成的 28–224 字节明文消息进行加密，统计加密前随机数生成、加密、解密等各阶段时延。

**📈 对比分析**

通过 5th–95th 百分位、均值以及最大值等统计指标与工业协议规定的最大时延做对比。结果显示在 Raspberry Pi 上的功能时延平均约 155 µs，最大约 224 µs，均低于 GOOSE 4 ms 与 IEC 60834‑1 10 ms 的上限；Mini PC 在动态频率下更快，功能时延约 58 µs，且仍满足两条协议约束。

**⚠️ 局限性**

仅在相对强大的单板机和迷你 PC 上测试，缺乏在典型 PLC（如 S7‑1500）低功耗 CPU 与内存受限环境下的验证；实验未覆盖网络协议栈、握手延迟及多线程场景；未与其他轻量级加密算法（如 AES‑256、Salsa20）做对比，导致对相对优势评估不完整。

---

## 226. Multiscale Switch for Semi-Supervised and Contrastive Learning in Medical Ultrasound Image Segmentation

**arXiv ID:** 2603.18655 | [PDF](https://arxiv.org/pdf/2603.18655v1)

**作者:** Jingguo Qu `[一作]` (Hong Kong Polytechnic University), Michael Tin-Cheung Ying `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 338 | [OpenAlex ID](https://openalex.org/A5107919987)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研发了一种半监督学习框架Switch，用于医学超声图像分割。

**💡 创新点**

创新点是将多尺度切换（MSS）与频域切换（FDS）相结合，并引入对比学习，显著提升低标注情况下的分割精度。

**🔧 技术方法**

采用教师-学生Mean Teacher架构、MSS、FDS、对比学习、一致性正则化以及弱强混合增强等技术。

**📊 数据集**

在六个超声数据集（LN-INT、LN-EXT、BUSI、DDTI、TN3K、Prostate）上进行评估。

**📈 对比分析**

与多种SOTA方法对比，在5‑10%标注比例下Dice提升3‑4个百分点，部分数据集甚至超过全监督模型；模型参数仅约1.8M，保持高效。

**⚠️ 局限性**

局限性包括：针对超声特定的参数调优（频域比例、补丁大小）不易迁移到其他模态；FFT操作导致计算成本上升；仅针对二分类任务，扩展至多类别或其他医学影像需要进一步研究。

---

## 227. A Comparative Empirical Study of Catastrophic Forgetting Mitigation in Sequential Task Adaptation for Continual Natural Language Processing Systems

**arXiv ID:** 2603.18641 | [PDF](https://arxiv.org/pdf/2603.18641v1)

**作者:** Aram Abrahamyan `[一作]` (American University of Armenia), Sachin Kumar `[通讯]` (American University of Armenia)

**通讯引用:** 5405 | [OpenAlex ID](https://openalex.org/A5032948428)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文比较了三种不同架构（ANN、GRU、Transformer）在10任务意图分类场景下的灾难性遗忘缓解方法。

**💡 创新点**

创新之处在于系统评估了单一及组合的连续学习策略（MIR、LwF、HAT），发现包含Replay的组合能近乎消除遗忘，并揭示模型架构与CL机制的交互影响。

**🔧 技术方法**

使用了最大干扰检索（MIR）回放、无遗忘学习（LwF）知识蒸馏、硬任务注意力（HAT）参数隔离，以及其两两和三者组合的训练流程。

**📊 数据集**

实验基于CLINC150意图分类数据集，构造10个标签不重叠任务，使用DistilBERT tokenizer进行预处理。

**📈 对比分析**

通过平均准确率、宏F1和向后转移（BWT）进行比较；单一Replay效果最佳，组合MIR+HAT在ANN/Transformer几乎无遗忘，MIR+LwF+HAT在GRU取得最佳AA≈0.91、AF1≈0.88，整体优于联合训练。

**⚠️ 局限性**

局限性包括仅在单一数据集和三种轻量模型上验证；未覆盖大型预训练模型、标签重叠或开放式任务；对超参数、内存大小等敏感性缺乏系统分析。

---

## 228. CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention

**arXiv ID:** 2603.18561 | [PDF](https://arxiv.org/pdf/2603.18561v1)

**作者:** Jiacheng Tang `[一作]` (Fudan University), Jian Pu `[通讯]` (Fudan University)

**通讯引用:** 3014 | [OpenAlex ID](https://openalex.org/A5100622420)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于因果干预的去偏训练框架CausalVAD，用来消除规划导向端到端驾驶模型中的混杂偏差，提升可靠性与安全性。

**💡 创新点**

创新点在于把Pearl的后门调整理论转化为可插拔的稀疏因果干预方案SCIS，并在VAD网络中多阶段地对感知、预测与规划节点进行定向干预。

**🔧 技术方法**

核心技术包括：构建多模态混杂词典（基于聚类的原型），感知去偏模块PDM、交互去偏模块IDM，及对网络进行的do‑operator参数化干预。

**📊 数据集**

主要使用nuScenes、NAVSIM和Bench2Drive三大公开基准数据集进行评估。

**📈 对比分析**

与现有端到端及VLM驱动方法（UniAD、VAD、BridgeAD等）对比，CausalVAD在nuScenes的L2误差降至0.54m、碰撞率仅0.11%，在NAVSIM与Bench2Drive上也取得最高的PDMS、DS和SR。

**⚠️ 局限性**

局限性：依赖预先构建的混杂词典，词典质量与聚类方法对结果影响；当前仅在VAD的序列式架构中验证，尚未扩展到并行或迭代交互模型。

---

## 229. Efficient Dense Crowd Trajectory Prediction Via Dynamic Clustering

**arXiv ID:** 2603.18166 | [PDF](https://arxiv.org/pdf/2603.18166v1)

**作者:** Antonius Bima Murti Wijaya `[一作]` (University of Glasgow), Marwa Mahmoud `[通讯]` (University of Glasgow)

**通讯引用:** 1577 | [OpenAlex ID](https://openalex.org/A5050992575)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于动态聚类的密集人群轨迹预测方法，旨在提高公共安全和管理，尤其是在密集人群场景中。

**💡 创新点**

创新点在于通过动态聚类技术将个体聚合为群体，从而减少计算成本并保持预测准确性，特别是在个体跟踪困难的密集场景中。

**🔧 技术方法**

使用了动态聚类技术，结合了局部离群因子（LOF）来评估群体成员的动态变化。

**📊 数据集**

使用了MOT Head Tracking 21数据集进行实验，评估了聚类性能和轨迹预测的有效性。

**📈 对比分析**

与现有的轨迹预测方法（如Trajectron++、SocialVAE和MART）进行比较，结果显示该方法在执行时间上减少了33.33%到79.4%，内存使用量减少了42.93%，同时保持了相对准确的预测性能。

**⚠️ 局限性**

限制在于聚类方法在处理个体动态变化时可能会出现聚类成员的丢失或身份切换问题，尽管该方法在大多数情况下表现良好。

---

## 230. Pore-scale modeling of capillary-driven binder migration during battery electrode drying

**arXiv ID:** 2603.18860 | [PDF](https://arxiv.org/pdf/2603.18860v1)

**作者:** Marcel Weichel `[一作]` (Karlsruhe Institute of Technology), Daniel Schneider `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5031 | [OpenAlex ID](https://openalex.org/A5074607822)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文开发并验证了一种基于相场方法的空间分辨连续模型，用于描述硬碳电极在干燥过程中由毛细力驱动的粘合剂迁移，并将其应用于真实硬碳微结构的数值模拟

**💡 创新点**

创新点在于首次将粘合剂的输运方程（扩散-对流方程）与完整的毛细干燥过程耦合，显式考虑毛细压力梯度、接触角、表面张力等微观机制，解决了传统一维或仅考虑薄膜收缩的模型无法捕获粘合剂非均匀分布的缺陷

**🔧 技术方法**

主要技术包括相场方法（Allen–Cahn方程）描述液气界面，Navier–Stokes方程模拟多相流，扩散-对流方程与沉积项描述粘合剂迁移；使用PACE3D多物理平台实现并行计算；利用有限差分网格和分步求解策略实现质量守恒

**📊 数据集**

数据集来自真实硬碳电极的SEM图像，经过图像分割得到两种微结构（HC‑B与HC‑C），并配合实验测得的干燥膜质量和固相体积分数，作为数值模拟的几何和物理参数输入

**📈 对比分析**

模型先在两个标准验证案例（气泡上升与无沉积的毛细干燥）中验证质量守恒和浓度分布；随后在两种微结构上系统扫描粘合剂黏度、蒸发率、表面张力和接触角等参数，得到粘合剂垂向梯度与破壁时间等指标；结果与实验观察相符，证明模型能合理预测不同工艺条件下的粘合剂分布，显著优于仅基于薄膜收缩的经验模型

**⚠️ 局限性**

主要限制包括：仅使用二维微结构，未能完全捕捉真实三维孔道分布；模型假设粘合剂和溶剂为牛顿流体，忽略剪切稀化和屈服行为；未考虑气相质量传递阻力和更复杂的气相饱和机制；模型对接触角的敏感性需进一步实验验证

---

## 231. Growing Alphabets Do Not Automatically Amplify Shuffle Privacy: Obstruction, Estimation Bounds, and Optimal Mechanism Design

**arXiv ID:** 2603.18080 | [PDF](https://arxiv.org/pdf/2603.18080v1)

**作者:** Alex Shvets `[一作]` `[通讯]` (Independent Researcher), Alex Shvets (Independent Researcher)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了在可洗牌（shuffle）模型中，随着输入符号字母表大小趋于无穷时，局部差分隐私通道的邻域一跳（one‑step neighboring）实验的隐私结构、频率估计的极限风险以及在给定的 χ² 预算下的机制设计。

**💡 创新点**

创新点包括：① 推导了精确的 likelihood‑ratio 压缩定理与 χ² 上界，揭示隐私与两点似然比分布之间的等价；② 构造了显式的“阻断族”使得无论符号表多大，洗牌隐私曲线始终与二值化随机响应（binary RR）一致；③ 证明了 χ² 预算是频率估计最小风险的通用下界；④ 在低预算下发现“稀释”原则（thinning principle），即最优机制是对一部分用户采用强烈的 GRR，其他用户发射公共空位；⑤ 在对称（permutation‑equivariant）通道族中给出了完整的低预算最优解并证明 GRR 在子集选择族内是唯一匹配预算的最优机制。

**🔧 技术方法**

主要技术手段包括：精确的 likelihood‑ratio 等式与压缩、Bhatia–Davis 方差不等式、Berry–Esseen 近似、Le Cam 极限理论、Cramér–Rao 下界、Assouad 论证、对称化与群作用下的轨道分解、凸分析与支撑线法、以及对子集选择模板的 KKT 条件分析。

**📊 数据集**

本文为纯理论研究，无需实验数据集；所有结论均在概率模型与信息理论框架下得到严格证明。

**📈 对比分析**

通过对比理论上最优风险与传统 GRR 或多点随机响应机制，作者展示了在低预算 regime（C≤C∗(d)）下，稀释化的 GRR（λ∗=√(d−1)）在误差上优于任何其他对称通道，并给出了精确的风险表达式；在高预算下则说明仍存在未知改进空间。

**⚠️ 局限性**

局限性：① 对高预算（C>C∗(d)）下的全对称最优前沿尚未完全求解；② 仅考虑了投影无偏逆估计器的风险，其他估计范式的最优性尚未探讨；③ 对于有限 n 的非渐近风险常数及具体实现细节的评估仍待进一步研究。

---

## 232. Unleashing the Power of Simplicity: A Minimalist Strategy for State-of-the-Art Fingerprint Enhancement

**arXiv ID:** 2603.19004 | [PDF](https://arxiv.org/pdf/2603.19004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 233. SpaceTime Programming: Live and Omniscient Exploration of Code and Execution

**arXiv ID:** 2603.18735 | [PDF](https://arxiv.org/pdf/2603.18735v1)

**作者:** Jean-Baptiste Döderlein `[一作]`, Benoit Combemale `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种统一的“SpaceTime Programming”模型，通过可追踪的执行记录实现对代码空间（变体）和时间（执行流）的同时探索；

**💡 创新点**

创新点在于将探索式编程、即时编程和全知调试整合到同一追踪框架中，使得开发者能够在任意历史状态下加载代码变体并对多条执行路径进行比较；

**🔧 技术方法**

实现技术包括Python 3.12 的 tracing 机制、装饰器插桩、SQLite 数据存储、重放与代码热替换，以及 GumTree diff 用于代码映射；

**📊 数据集**

实验使用了 HumanEval 164 个函数、五个流行 GitHub Python 项目（beets、cherrypy、discord.py、dspy、gensim）以及自制的 Flappy Bird 游戏；

**📈 对比分析**

性能对比显示功能级追踪导致 2–3 倍的执行时间开销，行级追踪约 30–40 倍；在 GitHub 项目中开销介于 35%–150%，大多数任务保持通过；存储空间从 MB 级到 GB 级；并通过 VSCode 扩展和 Pygame 工具展示可视化比较；

**⚠️ 局限性**

局限性包括行级追踪产生高成本、部分对象不可序列化、代码热替换仅支持特定层级、工具对语言和运行时高度依赖，以及对开发者的认知负荷提升。

---

## 234. Discovering What You Can Control: Interventional Boundary Discovery for Reinforcement Learning

**arXiv ID:** 2603.18257 | [PDF](https://arxiv.org/pdf/2603.18257v1)

**作者:** Jiaxin Liu `[一作]` (University of Illinois Urbana-Champaign), Jiaxin Liu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2039 | [OpenAlex ID](https://openalex.org/A5100383772)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用代理自身的动作作为干预，通过两样本检验识别出RL环境中哪些观测维度是真正因果受动作影响的，从而在存在混淆性干扰器的情况下进行特征选择；

**💡 创新点**

把特征选择重新表述为因果识别问题，并利用do-算子在不学习模型的情况下直接从干预数据中获得可解释的二进制掩码；

**🔧 技术方法**

结构化随机探测策略、Pearl 的 do-算子、Welch t 检验、Benjamini‑Hochberg 多重检验校正、以及对不同时间步长的多尺度检验；

**📊 数据集**

DeepMind Control Suite 的六个连续控制任务（Walker、Cheetah、Reacher、CartPole、Finger、Hopper），在每个任务上加上 6/50/100 维度的可混淆干扰器（自治、模拟、奖励相关干扰器）；

**📈 对比分析**

与全状态、oracle（真因果维度）、MI 选取、方差选取、条件 MI 等方法对比；在 12 个设定中 IBD 在大多数环境中获得接近 oracle 的回报，明显优于全状态尤其是当干扰器/信号比例超过约 3:1 时；

**⚠️ 局限性**

仅适用于结构化状态向量，假设因果结构稳定且干扰器完全外生；在非平稳或存在可控制干扰维度的情形下效果未知；需要额外的探测阶段，且不提供完整的因果图结构。

---

## 235. How Psychological Learning Paradigms Shaped and Constrained Artificial Intelligence

**arXiv ID:** 2603.18203 | [PDF](https://arxiv.org/pdf/2603.18203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 236. Variational Phasor Circuits for Phase-Native Brain-Computer Interface Classification

**arXiv ID:** 2603.18078 | [PDF](https://arxiv.org/pdf/2603.18078v1)

**作者:** Dibakar Sigdel `[一作]` `[通讯]` (Mindverse Computing), Dibakar Sigdel (Mindverse Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 Variational Phasor Circuit（VPC），一种在单位圆 S¹ 上的确定性相位学习架构，用于脑机接口 EEG 任务的二分类和四分类。

**💡 创新点**

将传统实数权重替换为可训练相位移、局部混合与结构化干涉，构造参数线性增长、低维度的相位-native 网络，直接捕捉 EEG 的相位拓扑，且不需要密集实数权重。

**🔧 技术方法**

采用相位编码、可训练相位旋转门、局部 beam‑splitter 单元、拉回归一化（pull‑back）、深层堆叠、PyTorch 自动微分与 COBYLA 等优化技术。

**📊 数据集**

使用合成的 32 通道 BrainFlow EEG 数据集，包含二分类（安静 vs. 参与）和四分类（平静、左/右运动意象、全局流动）四种精神状态。

**📈 对比分析**

与决策树、随机森林、SVM、MLP 等 Scikit‑Learn 基线在相同编码下进行对比，VPC 在 99% 以上准确率的同时仅需 64 个可训练参数；与无拉回深电路相比，拉回深堆叠在 99% 准确率上显著优于 85% 的无拉回版本。

**⚠️ 局限性**

存在梯度优化因相位周期性易陷入拓扑局部最小，梯度下降不稳定；实验仅基于合成数据，缺乏对真实 EEG 噪声的验证；缺少硬件实现与大规模可扩展性验证。

---

## 237. AlignMamba-2: Enhancing Multimodal Fusion and Sentiment Analysis with Modality-Aware Mamba

**arXiv ID:** 2603.18462 | [PDF](https://arxiv.org/pdf/2603.18462v1)

**作者:** Yan Li `[一作]` (Pengcheng Laboratory), Dongmei Jiang `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 2966 | [OpenAlex ID](https://openalex.org/A5102021514)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AlignMamba‑2 框架，利用双重对齐（Optimal Transport 与 Maximum Mean Discrepancy）和模态感知 Mamba 层（Mixture‑of‑Experts）实现高效多模态融合与情感分析

**💡 创新点**

创新点：1) 双重对齐作为训练时正则，无推理开销；2) 在 Mamba 背骨上引入模态感知 MoE，兼顾模态特定与共享特征；3) 通过上述改造实现线性时间与内存，显著提升 Transformer 的效率瓶颈

**🔧 技术方法**

核心技术：Mamba（State‑Space 模型）、Optimal Transport（Sinkhorn 算法）、Maximum Mean Discrepancy、Mixture‑of‑Experts、PyTorch 训练实现

**📊 数据集**

数据集：CMU‑MOSI、CMU‑MOSEI（动态情感），NYU‑Depth V2、MVSA（静态场景/图文）

**📈 对比分析**

与 Transformer 及最新 Mamba 基线对比：在 MOSI/MOSEI 上分别达到 87.0%/86.5% 的准确率和 F1，NYU‑Depth V2 73.1%/MVSA 82.7%；在长序列下显存与推理时延明显下降，呈线性增长，表现出优越的效率与效果

**⚠️ 局限性**

局限性：训练阶段需额外计算 OT 与 MMD 损失，导致训练成本上升；推理仍依赖预训练模态编码器；在极端长序列或数值不稳定场景下的鲁棒性需进一步提升

---

## 238. Confidential Databases Without Cryptographic Mappings

**arXiv ID:** 2603.18836 | [PDF](https://arxiv.org/pdf/2603.18836v1)

**作者:** Wenxuan Huang `[一作]` (Chinese Academy of Sciences), Mingyu Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 8350 | [OpenAlex ID](https://openalex.org/A5100335545)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的保密数据库设计，去除关键路径中的加密解密操作，通过在受信任域内维护数据独立标识符（I）映射到明文数据，实现高效的 put/get 操作；

**💡 创新点**

创新点在于将指针（indirection）与安全性分离，采用无密钥的可变长度映射存储，使用块级加密和异步加密，而非字段级加密，从而大幅降低加解密开销与密文扩展；

**🔧 技术方法**

技术包括双 TEE 区域（完整 DBMS 与表达式算子分离）、IID（Identity ID）映射表、块级加密、预取与分区管理、外部同步（external synchrony）事务协议、WAL 复制与回放、ARMv8.4 Secure EL2、dm‑integrity/dm‑crypt；

**📊 数据集**

使用标准基准 TPC‑C、TPC‑H、Sysbench 等；在生产 GaussDB 生产环境中测试；

**📈 对比分析**

与 HEDB（state‑of‑the‑art）和原生 PostgreSQL 进行对比；在 TPC‑C 上比 HEDB 提升 1.8×，在 TPC‑H 上相较 HEDB 减少 78.0× 执行时间；存储开销下降 34.3%–80.0%；在内存受限场景下仍保持 1.9× 性能优势；

**⚠️ 局限性**

局限性包括：仅支持固定/可变长度字段的映射，变长类型仍需额外管理；不支持跨域预取和空间回收；依赖 ARM TEE（Secure EL2）或等价硬件，无法直接迁移至非 ARM 平台；对侧信道攻击的防护仍需进一步强化；

---

## 239. SHAPCA: Consistent and Interpretable Explanations for Machine Learning Models on Spectroscopy Data

**arXiv ID:** 2603.19141 | [PDF](https://arxiv.org/pdf/2603.19141v1)

**作者:** Mingxing Zhang `[一作]` (University College Cork), Andrea Visentin `[通讯]` (University College Cork)

**通讯引用:** 306 | [OpenAlex ID](https://openalex.org/A5045006994)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了SHAPCA管线，将稀疏PCA与SHAP结合，实现对光谱数据的可解释与一致性预测解释

**💡 创新点**

创新点在于先用稀疏PCA将高度相关的波长聚集为低维组分，再在该潜在空间计算SHAP值，并将贡献回投到原始光谱轴，既保持解释的物理意义又显著提升解释稳定性

**🔧 技术方法**

使用稀疏主成分分析（Sparse PCA）、随机森林与支持向量分类器、TreeExplainer/KernalExplainer等SHAP方法、交叉验证、余弦相似度与皮尔逊相关系数评估一致性

**📊 数据集**

使用了两套真实光谱数据集：Raman口腔黏膜样本（二分类，Healthy vs Potentially Malignant Lesion）和DRS骨科组织光谱（六分类），并进行了统一的预处理与数据拆分

**📈 对比分析**

与直接在原始特征上应用SHAP的基线方法相比，SHAPCA在全局解释的余弦相似度与相关系数上均更高；预测准确率略有下降（约1-2%），但保持可接受；在随机森林上效果最佳，SVC解释更不稳定

**⚠️ 局限性**

局限性包括：稀疏PCA会略微降低模型精度；局部解释仍比全局解释一致性低；SVC基线解释不稳；仅针对单维光谱，未验证在多模态或时间序列数据上的适用性

---

## 240. AgentDS Technical Report: Benchmarking the Future of Human-AI Collaboration in Domain-Specific Data Science

**arXiv ID:** 2603.19005 | [PDF](https://arxiv.org/pdf/2603.19005v1)

**作者:** An Luo `[一作]` (University of Minnesota), Jie Ding `[通讯]` (University of Minnesota)

**通讯引用:** 7406 | [OpenAlex ID](https://openalex.org/A5100625885)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AgentDS基准和竞赛，评估AI代理与人机协作在六个行业的域特定数据科学任务表现。

**💡 创新点**

设计了需要领域知识、多模态融合、人工判断的挑战，并通过公开竞赛系统化比较AI与人类协作效果。

**🔧 技术方法**

使用大型语言模型（GPT‑4o、Claude Code）、自动化工具、人工监督的交互式工作流，以及量化评估的分位数评分。

**📊 数据集**

生成了六个行业的合成数据集（包括表格、图片、文本、JSON/PDF等），共17个挑战。

**📈 对比分析**

采用分位数归一化排名，比较AI仅代理、参与团队和人机协作；结果显示AI仅代理无法与顶尖团队匹敌，最优方案来自人机协作。

**⚠️ 局限性**

数据为合成、参赛者规模有限、领域覆盖有限、AI能力快速演进，缺乏受控实验验证协作模式。

---

## 241. Analysis Of Linguistic Stereotypes in Single and Multi-Agent Generative AI Architectures

**arXiv ID:** 2603.18729 | [PDF](https://arxiv.org/pdf/2603.18729v1)

**作者:** Martina Ullasci `[一作]` (Politecnico di Torino), Silvia Tagliente `[通讯]` (Politecnico di Torino)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5114517523)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在标准美式英语与非裔美式英语提示下的刻板印象生成进行复制与扩展，比较不同提示与多代理生成‑批评‑修订架构的偏差减缓效果。

**💡 创新点**

引入多代理生成‑批评‑修订管道作为偏差缓解机制，并系统评估不同提示策略对口音相关偏差的影响。

**🔧 技术方法**

使用提示工程（角色提示、链式推理）和多代理生成‑批评‑修订流程，并通过LLM‑as‑judge进行偏差评分。

**📊 数据集**

采用15对语义匹配的标准美式英语与非裔美式英语句子，以及8种刻板印象诱导模板，构成实验数据集。

**📈 对比分析**

对三大模型（Claude Haiku、Llama 3.2、Phi‑4 Mini）在四种提示配置下进行偏差评分对比，结果表明多代理管道在所有模型中实现最小的 SAE–AAE 差异，提示工程效果不稳定。

**⚠️ 局限性**

研究样本量有限、评判采用LLM可能引入自身偏差、仅关注 SAE/AAE、缺乏人类评估与更广泛方言覆盖。

---

## 242. Gradient-Informed Temporal Sampling Improves Rollout Accuracy in PDE Surrogate Training

**arXiv ID:** 2603.18237 | [PDF](https://arxiv.org/pdf/2603.18237v1)

**作者:** Wenshuo Wang `[一作]` (South China University of Technology), Fan Zhang `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 74511 | [OpenAlex ID](https://openalex.org/A5005958422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种Gradient‑Informed Temporal Sampling (GITS) 方法，用于在离线共享起始窗口的神经 PDE 近似器训练中选择最具信息量的时间窗口；

**💡 创新点**

创新点在于将轻量级先导模型的短期梯度评分与全局与局部时间覆盖正则化相结合，形成可子模的联合目标，兼顾局部信息与全局多样性；

**🔧 技术方法**

使用先导模型短期梯度评分、设施位置式全局覆盖、滑动窗口局部覆盖以及贪婪子模优化算法；

**📊 数据集**

在 PDEBench 的三个典型任务上评估：1D 扩散‑吸附（10k 轨迹）、2D 反应‑扩散（1k 轨迹）以及二维浅水破坝（1k 轨迹）；

**📈 对比分析**

与均匀采样、仅评分、仅覆盖、GradMatch、GLISTER、PRISM 等基线在 5% 预算下进行比较，GITS 在 12 组数据-骨干组合中平均 nRMSE 降至 0.193，较最差基线提升约 38%，并在 27/36 设定下获得最佳结果；

**⚠️ 局限性**

局限在于先导梯度评分在某些情形下与实际效用不一致（分数‑效用失配）或有过度分散的风险；仅针对共享起始窗口设定，并未覆盖所有 PDE 类型或更广泛的预算范围。

---

## 243. Cross-Ecosystem Vulnerability Analysis for Python Applications

**arXiv ID:** 2603.18693 | [PDF](https://arxiv.org/pdf/2603.18693v1)

**作者:** Georgios Alexopoulos `[一作]` (University of Athens), Dimitris Mitropoulos `[通讯]` (National Infrastructures for Research and Technology)

**通讯引用:** 887 | [OpenAlex ID](https://openalex.org/A5021658848)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

开发了一种跨生态系统漏洞分析方法，能够识别Python包中通过vendored本地库传播的CVE。

**💡 创新点**

创新点在于将可追溯性与源自OS发行版的具体包版本结合，构造跨语言调用图实现精确可达性分析，显著降低误报。

**🔧 技术方法**

采用内容哈希匹配、动态二进制版本提取、Python与ELF调用图生成与拼接、静态可达性分析等技术。

**📊 数据集**

在100,000个PyPI最受欢迎包以及10个已知CVE上进行评估。

**📈 对比分析**

与现有Librarian等工具对比，安全相关实例覆盖率达63.1%，误报下降可达97%，识别39个直接、312个间接易受影响包。

**⚠️ 局限性**

局限在于依赖hash数据库完整性，无法处理未在主OS仓库中的自定义构建，哈希碰撞和未嵌入版本信息的库等情况。

---

## 244. The Validity Gap in Health AI Evaluation: A Cross-Sectional Analysis of Benchmark Composition

**arXiv ID:** 2603.18294 | [PDF](https://arxiv.org/pdf/2603.18294v1)

**作者:** Alvin Rajkomar `[一作]` (Apple), Lily Peng `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对18,707条消费者健康查询进行结构与临床特征的系统剖析，并评估六个公开基准的组成差异。

**💡 创新点**

创新之处在于提出并应用了一个16维的“查询剖析”框架，揭示了基准评估与真实临床需求之间的“有效性缺口”。

**🔧 技术方法**

使用 GPT‑5.2 进行自动标签，辅以 Claude Opus 交叉验证和规则分类器，实现大规模、可复现的查询标注。

**📊 数据集**

利用 HealthSearchQA、MashQA、MedRedQA、HealthBench、GoogleFitbit Sleep 和 GoogleFitbit Fitness 这六个公开数据集。

**📈 对比分析**

通过描述性统计与代际对比分析，显示高危行为健康、慢性病管理等重要场景在基准中被严重低估，低风险内容占主导，导致模型性能评估可能被夸大。

**⚠️ 局限性**

局限包括依赖自动标签可能忽略细微语义，分类框架仅为一种可行方案，未评估模型实际性能，且研究仅为横断面分析，缺乏长期交互样本。

---

## 245. Box Maze: A Process-Control Architecture for Reliable LLM Reasoning

**arXiv ID:** 2603.19182 | [PDF](https://arxiv.org/pdf/2603.19182v1)

**作者:** Zou Qiang `[一作]` `[通讯]` (Independent Researcher), Zou Qiang (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Box Maze过程控制架构，分为记忆回环、逻辑回环和心脏锚定三层，并通过模拟对抗实验验证其在大型语言模型（LLM）推理中的可靠性。

**💡 创新点**

创新点在于将过程控制嵌入LLM中间层：引入硬约束边界、时间锚定记忆、因果一致性检查，并提出分阶段发展路径（Phase I‑III），从而在对抗场景下显著降低幻觉和边界违背。

**🔧 技术方法**

使用的技术包括：
- 过程控制架构（Memory Loop、Logic Loop、Heart Anchor）
- 时间戳记忆锚定
- 结构化推理的因果一致性校验
- 硬约束的心脏锚定（Mutex Enforcement）
- 模拟验证（LLM role‑play）
- 边界违背率、幻觉合规率和约束一致性评分等评价指标。

**📊 数据集**

实验使用约50个对抗场景（进阶逻辑陷阱、逆逻辑情境、强迫情境）在三大模型（DeepSeek‑V3、Doubao、Qwen）上进行。没有使用公开数据集，而是通过构造对抗输入进行模拟评估。

**📈 对比分析**

对比方法：将Box Maze与零协议（Zero Protocol）基线相对照，采用Boundary Violation Rate (BVR)、Hallucination Compliance Rate (HCR)和Constraint Consistency Score (CCS)。结果显示：
- 基线BVR/HCR≈40%，CCS≈60%；
- Box Maze BVR<1%，HCR<1%，CCS>99%。
Ablation实验进一步证明Heart Anchor是对抗强迫的关键组件。

**⚠️ 局限性**

局限性：
- 仅为概念架构，验证采用LLM角色扮演的模拟，未实现真正的中间件或内核级实现；
- 缺乏大规模统计验证（千级场景、跨域测试）；
- 主要解决对抗性幻觉，对事实性幻觉或价值漂移关注不足；
- 记忆厚度阈值经验调优，可能对不同模型或任务不适用；
- 需要进一步扩展至多模态环境并实现正式部署。

---

## 246. Clinically Meaningful Explainability for NeuroAI: An ethical, technical, and clinical perspective

**arXiv ID:** 2603.18028 | [PDF](https://arxiv.org/pdf/2603.18028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 247. Entropy trajectory shape predicts LLM reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought

**arXiv ID:** 2603.18940 | [PDF](https://arxiv.org/pdf/2603.18940v1)

**作者:** Xinghao Zhao `[一作]` `[通讯]` (Huazhong University of Science and Technology), Xinghao Zhao (Huazhong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在链式思维（CoT）生成过程中对每一步答案分布的熵进行采样，构建熵轨迹并检测其单调性，以此来预测最终答案的正确性。

**💡 创新点**

创新点在于提出熵轨迹单调性这一形状指标，发现其相较于传统的总熵下降幅度更能区分正确与错误链，并且在不同模型族（Qwen2.5-7B-Instruct 与 Mistral‑7B-Instruct‑v0.3）上保持一致。

**🔧 技术方法**

技术手段包括：在每步前缀下随机采样m=5个答案完成，计算答案分布熵；使用二元/计数单调性作为可靠性信号；与自一致性、token log‑prob 置信度、熵总下降等基线进行对比。

**📊 数据集**

使用的数据集为 GSM8K 题库（n=300），部分在 MATH 子集（n=200）做验证；实验模型为 Qwen2.5‑7B‑Instruct 与 Mistral‑7B‑Instruct‑v0.3。

**📈 对比分析**

在相同 token 预算下（≈1,500 tokens/问题）与多链自一致性（SC@10/40）、token 置信度、熵总下降等基线比较，熵轨迹单调性在 73.7% 覆盖率下提升约 5.8pp 准确率（从 63.0% 提升至 68.8%），而自一致性在相同预算下准确率仅 65–66%。

**⚠️ 局限性**

局限性包括：仅在两款模型和有限数据集上验证，单链单调性误判率约 31%（假正），无法直接获取步级正确性标签导致校准分析受限，且对采样温度、样本数等参数仍有一定敏感性。

---

## 248. MultihopSpatial: Multi-hop Compositional Spatial Reasoning Benchmark for Vision-Language Model

**arXiv ID:** 2603.18892 | [PDF](https://arxiv.org/pdf/2603.18892v1)

**作者:** Youngwan Lee `[一作]` (Electronics and Telecommunications Research Institute), Sung Ju Hwang `[通讯]` (DeepAuto)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个多跳空间推理基准 MultihopSpatial，并通过 Reinforcement Learning 对 VLM 进行后训练，提升其在多步空间推理和视觉定位上的表现。

**💡 创新点**

创新点包括：①引入 1–3 跳复合空间推理任务；②提出 Acc@50IoU 兼顾答案正确性与边框定位的综合评测指标；③提供专门的训练集 MultihopSpatial-Train，用于提升 VLM 的空间智能和实际机器人控制性能。

**🔧 技术方法**

技术手段主要是基于多模态 LLM（如 Qwen3‑VL、Gemini 等）与视觉编码器结合，使用 Group Relative Policy Optimization（GRPO）和 RLVR 奖励框架进行后训练，奖励包括格式、答案与 GIoU。

**📊 数据集**

数据集：MultihopSpatial 共 4,500 个 1–3 跳空间问题，覆盖属性、位置与关系三类；MultihopSpatial‑Train 为 6,791 条带标注边框的 VQA 训练样本；图像来源于 COCO 与 PACO‑Ego4D。

**📈 对比分析**

对 37 款 VLM 进行评测，显示即使在 MCQ 上取得 64% 以上准确率，Acc@50IoU 最高仅 40.6%。后训练后模型在多跳任务上均有显著提升，并在 CALVIN 与 Libero 机器人任务中实现 3.75→3.98 的完成率提升。

**⚠️ 局限性**

局限性：①当前模型在高跳数（3 跳）与自我视角（ego‑centric）下仍难以突破 10% 的 Acc@50IoU；②多模态视觉编码器的尺度对定位能力影响显著，单靠语言模型扩容收益有限；③部分专业空间推理模型在多跳任务中表现不如通用大模型，说明现有专用模型缺乏对复杂组合推理的泛化能力。

---

## 249. S3T-Former: A Purely Spike-Driven State-Space Topology Transformer for Skeleton Action Recognition

**arXiv ID:** 2603.18062 | [PDF](https://arxiv.org/pdf/2603.18062v1)

**作者:** Naichuan Zheng `[一作]` (Beijing University of Posts and Telecommunications), Yujia Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了纯脉冲驱动的Skeleton动作识别Transformer S3T-Former，利用多流解剖脉冲嵌入、非对称时序梯度QKV、侧向拓扑路由和状态空间引擎实现极致稀疏推理。

**💡 创新点**

创新点在于：① M-ASE将多阶运动学差分转为事件流；② ATG-QKV仅让运动梯度产生查询键，压制静态冗余；③ LSTR将空间图聚合变为零MAC条件加法；④ S3引擎实现线性复杂度长时记忆，解决短期遗忘。

**🔧 技术方法**

使用的技术包括：脉冲神经网络（LIF）、自适应梯度QKV、侧向拓扑路由、线性状态空间模型、非量化U-Readout、持续时间高效训练（TET）等。

**📊 数据集**

在NTU RGB+D 60、NTU RGB+D 120和NW-UCLA三个大规模骨架动作识别数据集上进行实验。

**📈 对比分析**

与多种ANN与SNN基线比较，S3T-Former在NTU-60/120上分别达到约84–95%的Top-1准确率，超越Signal-SGN、Spikformer等SNN模型，并在能耗上比ANN低10%以下。

**⚠️ 局限性**

局限性在于：对极低频微动作识别仍有不足；需要更大规模的脉冲硬件实现；以及在多模态融合上仍不如传统ANN在部分任务中表现。

---

## 250. Serendipity by Design: Evaluating the Impact of Cross-domain Mappings on Human and LLM Creativity

**arXiv ID:** 2603.19087 | [PDF](https://arxiv.org/pdf/2603.19087v1)

**作者:** Qiawen Ella Liu `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**通讯引用:** 49744 | [OpenAlex ID](https://openalex.org/A5077079119)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过随机跨域映射激发人类和大语言模型（LLM）在十种日常产品上生成创意，并比较两者在原创性、可行性、实用性和投资价值等维度上的表现。

**💡 创新点**

首次系统检验跨域映射对人类与LLM创意的差异效应，发现LLM原创性更高，而跨域映射对人类更有效，揭示两者对远距离灵感的敏感度不同。

**🔧 技术方法**

采用对照实验设计，使用多种LLM（如GPT‑4、Claude等）与人工创意，通过自然语言处理将文本规范化后，利用线性混合效应模型评估原创性。

**📊 数据集**

数据集由十种产品与二十六种跨域源构成，人工与LLM各生成700条创意，共2800条，再随机分成100份评估列表供人类评分。

**📈 对比分析**

通过与“用户需求”提示的对照条件以及人类/LLM来源比较，发现LLM平均原创性评分显著高于人类，跨域映射显著提升人类原创性但对LLM效果有限；原创性与可行性呈负相关。

**⚠️ 局限性**

限制包括仅进行一次性创意生成、缺乏对跨域映射的定性分析、未考察迭代创作流程，以及仅依赖主观评分评估创意质量。

---

## 251. InfoMamba: An Attention-Free Hybrid Mamba-Transformer Model

**arXiv ID:** 2603.18031 | [PDF](https://arxiv.org/pdf/2603.18031v1)

**作者:** Youjin Wang `[一作]` (Renmin University of China), Feng Zhou `[通讯]` (Renmin University of China)

**通讯引用:** 374095 | [OpenAlex ID](https://openalex.org/A5111964102)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种无注意力的混合架构InfoMamba，结合线性全局过滤与选择性状态空间模型；

**💡 创新点**

通过一致性边界分析发现了注意力与SSM的互补性，设计了概念瓶颈线性过滤和信息最大化融合两条并行路径；

**🔧 技术方法**

使用概念分配、信息最大化哈希分桶、线性过滤层、选择性SSM、信息最大化融合（IMF）以及互信息约束损失；

**📊 数据集**

在ImageNet-1K、COCO、ADE20K/Cityscapes、AG-News、IMDb、LibriSpeech等多种视觉、文本和语音数据集上进行评测；

**📈 对比分析**

与CNN、Transformer、Linformer、Mamba等基线在同等规模下对比，InfoMamba在分类、目标检测、分割、文本分类、语音识别等任务中均表现出更优或相近的准确率，同时保持线性时间复杂度，显著提升了效率；

**⚠️ 局限性**

对比实验主要集中在大规模数据，缺乏对极端长序列或非常低分辨率场景的深入验证，且对可解释性和模型可扩展性在极大模型规模下的影响仍待进一步研究。

---

## 252. HiMu: Hierarchical Multimodal Frame Selection for Long Video Question Answering

**arXiv ID:** 2603.18558 | [PDF](https://arxiv.org/pdf/2603.18558v1)

**作者:** Dan Ben-Ami `[一作]` (INSIGHT Lab, Ben-Gurion University of the Negev), Chaim Baskin `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 427 | [OpenAlex ID](https://openalex.org/A5019913171)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出HiMu框架，利用单次LLM解析查询生成层次逻辑树，并通过多模态专家（CLIP、OVD、OCR、ASR、CLAP）和模糊逻辑合成，完成训练无关的高效帧选择；

**💡 创新点**

核心创新是将复杂查询拆解为可被各模态专家直接评估的原子谓词，借助模糊逻辑实现时序与并行约束，从而在单次调用中实现深度组合推理；

**🔧 技术方法**

使用文本仅LLM推理生成逻辑树、基于CLIP/OVD/OCR/ASR/CLAP的轻量级专家、模糊逻辑运算、PASS峰值扩展策略以及专家特征缓存；

**📊 数据集**

实验采用Video‑MME、LongVideoBench_val和HERBench‑Lite三大长视频问答基准；

**📈 对比分析**

在与相同帧预算的相似度选择和迭代agent方法对比时，HiMu在16帧下在Video‑MME、LongVideoBench和HERBench上均取得最高准确率，且计算成本约为迭代方法的1/10，甚至在多模态框架下超过高帧预算agent；

**⚠️ 局限性**

主要限制包括：相对相似度方法的额外专家提取延迟、对LLM解析树质量高度依赖，以及ASR专家受限于语言覆盖导致在多语种或低资源音频上效果受限。

---

## 253. Impact of Differentials in SIMON32 Algorithm for Lightweight Security of Internet of Things

**arXiv ID:** 2603.18455 | [PDF](https://arxiv.org/pdf/2603.18455v1)

**作者:** Jonathan Cook `[一作]` (Charles Sturt University), M. Arif Khan `[通讯]` (Charles Sturt University)

**通讯引用:** 4140 | [OpenAlex ID](https://openalex.org/A5015246388)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本研究对SIMON32轻量级密码的差分特性进行分析，利用部分差分分布表（pDDT）识别高概率差分路径，并在20轮内获得概率为2^-32的差分轨迹，突破了之前的最高轮数与概率记录。

**💡 创新点**

创新点在于：①基于pDDT进行差分筛选，减少搜索空间；②识别出“以2为幂”差分特征，进一步聚焦高概率路径；③通过统计学验证差分显著性，提升差分分析可靠性。

**🔧 技术方法**

技术方法包括：构造pDDT、按概率排序与抽样筛选、汉明重量分布分析、差分轨迹模拟、统计显著性检验（p值、t统计量）。

**📊 数据集**

主要使用SIMON32密码算法的内部差分分布数据（pDDT生成自完整算子），未使用外部公开数据集。

**📈 对比分析**

与先前工作相比，本研究在相同或更少的轮数下实现了更高的差分概率（2^-32），并通过统计显著性检验确认差分路径的可靠性，显示出显著的性能提升。

**⚠️ 局限性**

局限性包括：①仅针对SIMON32进行实验，需验证是否可推广到更大版本；②依赖人工挑选高概率差分，未自动化完全集成；③对pDDT阈值的选择仍主观，可能影响结果。

---

## 254. Guardrails as Infrastructure: Policy-First Control for Tool-Orchestrated Workflows

**arXiv ID:** 2603.18059 | [PDF](https://arxiv.org/pdf/2603.18059v1)

**作者:** Akshey Sigdel `[一作]` (Independent Researcher), Rista Baral `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套模型无关的“Policy-First Tooling”权限层，在工具调用过程中通过政策 DSL、风险门控、恢复控制和可审计解释来保障安全。

**💡 创新点**

创新点包括：1) 设计了简洁的政策 DSL 与运行时 Enforcement 体系；2) 将安全、可靠性与审计性整合到同一层；3) 构建可复现的基准，结合故障注入和误用注入客观评估安全-效能权衡。

**🔧 技术方法**

使用了正则/范围/枚举约束、成本/速率预算、审批门控、输出脱敏、重试/退避/断路器等可靠性模式；CPU‑only 模拟工具与 trace replay；统计分析、bootstrap CI 与配对符号检验。

**📊 数据集**

使用自定义 JSONL 调用轨迹集（五个工作负载 A–E）以及手工定义的误用与故障注入配置；未采用公开数据集。

**📈 对比分析**

在 225 条实验（5 套任务 × 5 policy pack × 3 fault profile × 3 seed）中比较安全指标（VPR、泄漏召回）、可靠性指标（任务成功率、重试放大、尾延迟）和效能指标（误阻率、审批负担）。结果显示：P3/P4 方案将 VPR 提升至 0.68，任务成功率降至 0.07，重试放大从 3.77 降至 1.38；而 P0–P2 方案无显著安全提升。

**⚠️ 局限性**

局限性包括：① 仅在受控 CPU‑only 环境验证，未覆盖真正恶意攻击；② 输出脱敏精度不高、误报率较高；③ 审批评估仅基于模拟，缺乏真实人类实验；④ 政策对特定工具集可能过拟合，迁移性未知。

---

## 255. EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation

**arXiv ID:** 2603.18739 | [PDF](https://arxiv.org/pdf/2603.18739v1)

**作者:** Longfei Liu `[一作]` (Intellindust AI Lab), Xi Shen `[通讯]` (Intellindust AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 EdgeCrafter，一个面向边缘设备的统一紧凑 Vision Transformer 框架，能够同时完成目标检测、实例分割和人体姿态估计。

**💡 创新点**

创新点在于：① 将大型 DINOv3 ViT 先转化为检测专用教师模型，再通过任务特化的知识蒸馏把高质量表征迁移到小规模 ViT 后端；② 采用卷积 stem 和简易多尺度特征生成，显著降低计算和参数；③ 共享的检测蒸馏表征可直接用于分割和姿态任务，实现多任务统一。

**🔧 技术方法**

技术包括：ViT 轻量化（ECViT）、卷积 stem、线性投影多尺度特征、RT‑DETR 结构、单向特征对齐蒸馏、Varifocal/OKS 目标、掩码点乘式分割头等。

**📊 数据集**

主要使用 COCO 数据集进行训练与评估（检测/分割/姿态），教师模型在 ImageNet‑21K 与 COCO 上先预训练。

**📈 对比分析**

在 COCO 上与当下主流轻量化检测器（RT‑DETR、DEIMv2、RF‑DETR 等）对比，EdgeCrafter 在相同或更小参数/ FLOPs 的情况下取得 51.7–57.9 AP 的目标检测性能，姿态估计 AP 最高可达 74.3，实例分割 AP 最高 48.4，均优于或与使用大规模外部预训练（如 Objects365）的竞争对手相当。

**⚠️ 局限性**

局限性包括：ViT 结构在现有软硬件上尚未完全优化，导致推理速度不如最佳 CNN 检测器；蒸馏需要额外的教师模型和大规模预训练；在极低预算或极端实时场景下仍需进一步压缩与加速。

---

## 256. Understanding the Theoretical Foundations of Deep Neural Networks through Differential Equations

**arXiv ID:** 2603.18331 | [PDF](https://arxiv.org/pdf/2603.18331v1)

**作者:** Hongjue Zhao `[一作]` (University of Illinois Urbana Champaign), Huajie Shao `[通讯]` (William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了将微分方程视为深度神经网络的理论基础，系统讨论了模型层与层级两种视角下的架构、工具和应用，并指出未来研究方向。

**💡 创新点**

提出统一的两层次框架（模型层/层层级）来组织和关联现有工作，强调微分方程在架构设计、理论分析与性能提升中的统一作用，并梳理了最新进展与挑战。

**🔧 技术方法**

运用了微分方程、控制理论、动态系统分析、数值求解器、Neural ODE/CDE/SDE、深度状态空间模型（SSM）、流型与扩散生成模型、Lyapunov 与壁障函数、最优控制等技术。

**📊 数据集**

本文为综述，不做实验；引用的工作使用了 ImageNet、CIFAR、自然语言处理数据、物理实验数据、医疗时间序列、气候模型等多种数据集。

**📈 对比分析**

未进行实验对比；通过文献综述，指出微分方程方法在稳定性、表达能力、可解释性以及生成质量等方面提升显著，并在生成模型、时间序列预测、强化学习等任务中取得 state‑of‑the‑art 结果。

**⚠️ 局限性**

局限性包括：计算效率与可扩展性不足（尤其是 ODE 求解器成本高、对刚性动力学敏感），模型层与层级视角缺乏统一桥梁，解释性不足，难以直接与大规模新兴 AI（如 LLM）融合，理论与实际应用之间仍有差距。

---

## 257. Few-shot Acoustic Synthesis with Multimodal Flow Matching

**arXiv ID:** 2603.19176 | [PDF](https://arxiv.org/pdf/2603.19176v1)

**作者:** Amandine Brunetto `[一作]` `[通讯]` (Mines Paris - PSL University), Amandine Brunetto (Mines Paris - PSL University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于流匹配的少样本房间冲击响应（RIR）合成框架 FLAC，并引入 AGREE 评估空间一致性的多模态嵌入。

**💡 创新点**

创新点包括：①首次将流匹配应用于 RIR 生成，显式建模多样化的不确定分布；②通过多模态（音频、空间、几何）条件实现一/少样本的场景一致性合成；③提出 AGREE 对齐音频与几何的 CLIP‑style 嵌入，用于零样本检索和分布性一致性评估。

**🔧 技术方法**

核心技术：变分自编码器（VAE）压缩 RIR；多模态条件器（音频、空间、几何）；扩散 Transformer（DiT）配合流匹配训练；分类器无关引导（CFG）；AGREE 对齐音频与几何的双编码器与对比学习。

**📊 数据集**

使用的主要数据集：Synthetic AcousticRooms（260 个房间、300k RIR + 深度图）和实测 Hearing‑Anything‑Anywhere（4 个房间、RIR+深度图），并在两者上进行评估。

**📈 对比分析**

与现有 8‑shot 基线（xRIR、Fast‑RIR、Fast‑RIR 等）以及 KNN、随机等方法对比，FLAC 在一/少样本下在 T60、C50、EDT 等感知指标上显著优于对手（误差下降 13–30%），音频检索召回率更高，Fréchet 距离更小；在 HAA 实测数据上亦保持领先，且无需重新训练即可适应新房间。

**⚠️ 局限性**

局限性：①VAE 训练仅基于模拟数据，对真实录音的泛化受限，导致在 HAA 上仍有差距；②几何信息主要依赖深度图和简化的房间模型，无法捕捉复杂材料与微细反射；③模型生成的细节多样性受噪声采样步数限制，仍需进一步提升低频不确定性处理。

---

## 258. Seasoning Generative Models for a Generalization Aftertaste

**arXiv ID:** 2603.18817 | [PDF](https://arxiv.org/pdf/2603.18817v1)

**作者:** Hisham Husain `[一作]` (Google Research), Richard Nock `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用判别器集通过强对偶关系对已有生成模型进行改进的通用方法，给出了改进后模型的解析表达式并推导出收敛与泛化理论；

**💡 创新点**

首次将 f‑divergence 的强对偶性与生成模型的改进联系起来，给出了判别器指导下的扩散模型和提升密度估计的理论框架，形成了“判别器驱动的生成模型改进”这一全新范式；

**🔧 技术方法**

使用 f‑divergence 对偶、IPM、判别器的 Rademacher 复杂度、Savage 的正确性理论以及分数导数的组合；在此基础上给出了改进后分数的显式公式；

**📊 数据集**

论文主要是理论推导，未给出具体实验数据；若有实验，可能使用常见的图像生成基准如 MNIST、CIFAR‑10 或 ImageNet 等；

**📈 对比分析**

方法通过理论上证明改进后模型在 IPM 与 f‑divergence 上的距离更小，并给出了泛化误差上界；没有给出具体实验对比，理论结果表明在判别器足够表达且正则化的条件下可获得更好的收敛与泛化；

**⚠️ 局限性**

限制包括：需要判别器集合满足凸闭包且可对偶；改进后模型的采样需要已知或可近似原模型的分数，实际实现难度较大；理论依赖于对判别器的 Rademacher 复杂度估计，若判别器过于复杂则泛化可能退化；未在实证数据集上验证，理论与实践之间仍有距离。

---

## 259. High-Performance Portable GPU Primitives for Arbitrary Types and Operators in Julia

**arXiv ID:** 2603.18695 | [PDF](https://arxiv.org/pdf/2603.18695v1)

**作者:** Emmanuel Pilliat `[一作]` `[通讯]` (University of Rennes), Emmanuel Pilliat (University of Rennes)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 KernelForge.jl，一个能够在 NVIDIA 与 AMD GPU 上实现高性能、可移植且支持任意类型与算子的 GPU 并行原语（scan、mapreduce、矩阵-向量）库。

**💡 创新点**

创新点在于两层可移植架构：KernelIntrinsics.jl 提供底层可跨后端的 warp shuffle、内存屏障和向量化访问；KernelForge.jl 仅在此层上实现高效算法；同时利用 Julia 的 JIT、多派发与元编程实现针对数据类型、算子、问题规模的专用代码，兼顾性能与灵活性。

**🔧 技术方法**

采用的技术包括 Julia 的 KernelAbstractions.jl、CUDA.jl/AMDGPU.jl、warp shuffle、release‑acquire 内存屏障、向量化加载/存储、递归拆解复合类型、单次传递（decoupled look‑back）扫描、手工调优参数、基于类型的编译时分发等。

**📊 数据集**

基准使用大规模随机数组（10⁶–10⁹ 元素）与矩阵（n×p 10⁷–10⁹）以及多种数值类型（Float32、UInt8、UnitFloat8、用户自定义类型）进行测试，未使用特定的真实数据集。

**📈 对比分析**

比较方法：与 CUB、CUDA.jl、AcceleratedKernels.jl、cuBLAS、Kokkos 等已发布实现对比，测量 GPU 设备执行时间和完整流水线时间。结果显示：在 NVIDIA A40 上，KernelForge.jl 与 CUB、cuBLAS 的速度相当或更快；在 AMD MI300X 上也匹配或优于 AMDGPU.jl 与 AcceleratedKernels.jl；scan/mreduce 在大规模数据上可提升至 20× 以上；矩阵-向量在大多数形状下与 cuBLAS 接近，某些特殊形状略低。

**⚠️ 局限性**

局限性包括：仅支持 CUDA 与 ROCm，其他后端需编写 KernelIntrinsics.jl 扩展；调优参数为 A40 手工完成，迁移至新架构需再次调优；对矩阵-矩阵（tensor core）运算缺乏优化；只做概念验证，未覆盖所有算子与类型；缺乏自动化调优和多后端完整测试。

---

## 260. Inductance-Based Force Self-Sensing in Fiber-Reinforced Pneumatic Twisted-and-Coiled Actuators

**arXiv ID:** 2603.18555 | [PDF](https://arxiv.org/pdf/2603.18555v1)

**作者:** Yunsong Zhang `[一作]` (Peking University), Feitian Zhang `[通讯]` (Peking University)

**通讯引用:** 650 | [OpenAlex ID](https://openalex.org/A5059772938)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在论文中，作者将导电镍线融入纤维增强气动扭曲缠绕软肌（FR‑PTCA）中，实现了自感式力传感，并基于该传感原理开发了动力学模型、参数化自感模型及混合 EKF‑优化观测器，最终实现了无外部传感器的实时力估计与闭环控制。

**💡 创新点**

创新点在于发现 FR‑PTCA 的电感与输出力呈低滞后、确定性关系，利用这一特性提出了可解析的自感模型，并结合约束优化与扩展卡尔曼滤波构建混合观测器，解决了电感–力映射的多值性与噪声问题，实现了高精度自感力测。

**🔧 技术方法**

技术手段包括：导电镍线缠绕、低通巴特沃斯滤波、扩展卡尔曼滤波、非线性约束优化、气动驱动、力传感器校准、实验平台设计与数据采集。

**📊 数据集**

使用的数据集为多组实验测得的 FR‑PTCA 在不同初始长度（80–160 mm）、不同工作压力（0–0.65 MPa）和不同位移条件下的电感、力、长度和压力记录，主要用于模型参数识别与观测器验证。

**📈 对比分析**

通过与开环推算、外部负载传感器闭环以及自感闭环三种控制策略的对比，实验表明自感闭环在力跟踪任务中 RMSE 约 0.07 N（相较于开环提升 40–80%），位移估计误差约 10 mm，整体跟踪误差显著低于开环并逼近外部传感器水平。

**⚠️ 局限性**

局限性主要体现在：位移估计依赖简化的线性弹性模型，无法完整捕捉软肌的全 hysteresis；多驱动或多轴耦合时可能出现电磁串扰；在极端高频或大变形条件下自感信号噪声与非线性更明显。

---

## 261. Discounted Beta--Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2603.18444 | [PDF](https://arxiv.org/pdf/2603.18444v1)

**作者:** Haechan Kim `[一作]` (KAIST), Eunho Yang `[通讯]` (AITRICS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于折扣贝塔-伯努利（DBB）的奖励估计方法，用以改进 RLVR 的样本效率并消除方差崩溃。

**💡 创新点**

创新点在于把奖励视为由策略诱导的分布，采用贝叶斯折扣更新利用历史奖励降低方差，并理论上避免方差崩溃。

**🔧 技术方法**

使用贝叶斯 Beta-Bernoulli 模型、折扣因子 λ、GRPO-DBB 训练框架以及 MSE 分析来实现奖励分布估计。

**📊 数据集**

在 Qwen3-1.7B/8B 模型上使用 DAPO-Math-17k 训练集，六个数学推理基准（MATH500、Minerva、AIME24/25、AMC24、OlympiadBench）和三项 OOD 基准（MMLU-Pro、GPQA-Diamond、Big-Bench Hard）。

**📈 对比分析**

与 naive GRPO、RePO、Dr.GRPO 等基线对比，GRPO-DBB 在所有基准上平均提升 Acc@8 约 3–12 分，尤其在 OOD 上提升显著。

**⚠️ 局限性**

局限在于需要手动调节折扣因子 λ；仍带有偏差，且在极大样本或快速学习阶段效果可能有限，未实现自适应 λ。

---

## 262. ARIADNE: A Perception-Reasoning Synergy Framework for Trustworthy Coronary Angiography Analysis

**arXiv ID:** 2603.19169 | [PDF](https://arxiv.org/pdf/2603.19169v1)

**作者:** Zhan Jin `[一作]` (Ocean University of China), Qing Zhang `[通讯]` (Shandong University)

**通讯引用:** 25892 | [OpenAlex ID](https://openalex.org/A5069833193)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了ARIADNE框架，将基于视觉语言的分割模型与强化学习诊断代理相结合，实现冠脉血管的拓扑连通分割与狭窄检测；

**💡 创新点**

创新点在于通过直接偏好优化（DPO）引入血管拓扑约束，解决语义-拓扑缺口，并在检测阶段引入显式拒绝机制以降低误报；

**🔧 技术方法**

采用Sa2VA视觉语言基础模型、DPO、Hard Sample Focused Training、基于中心线的clDice评估、以及PPO强化学习的决策代理；

**📊 数据集**

使用内部1400幅血管造影影像（35例）进行训练和验证，并在公开的ARCADE与XCAD数据集上做外部验证；

**📈 对比分析**

与U‑Net、SVSNet、FlowVM‑Net、MedSAM3等基线比较，ARIADNE在中心线Dice达到0.838、FPPI下降至0.85、TPR提升至0.867，显示在分割连通性和误报率方面显著优于传统方法；

**⚠️ 局限性**

局限包括仅针对二维投影造影，无法处理严重钙化或运动伪影导致的拓扑不确定性，且模型依赖大规模标注且对多视角或三维数据的适应性尚未验证。

---

## 263. From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models

**arXiv ID:** 2603.19131 | [PDF](https://arxiv.org/pdf/2603.19131v1)

**作者:** Zhuofan Li `[一作]` (Hong Kong University of Science and Technology), Chaojian Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉-语言-动作（VLA）模型在实际机器人平台上的执行效率，指出传统的推理效率指标（参数量、FLOPs、解码吞吐量）无法反映完整的机器人执行成本，提出并实验了多种“身体化”效率指标，并评估了模型压缩、令牌稀疏化、动作序列压缩以及适配策略（提示学习与监督微调）对这些指标的影响。

**💡 创新点**

创新点在于：1）提出了任务完成时间、末端执行器路径长度、关节空间路径长度、加速度（jerk）和动作率等身体化效率指标；2）系统性地展示了常用的推理效率优化方法（剪枝、量化、视觉令牌稀疏化、FAST动作编码）往往会损害身体化效率；3）通过在提示和微调中加入效率目标，首次探讨了传统大语言模型适配方法对身体化效率的局部提升与潜在权衡。

**🔧 技术方法**

使用的技术包括：结构化与非结构化权重剪枝、16/8/4位量化、视觉令牌稀疏化、FAST动作压缩、在上下文提示中加入效率约束、在监督微调中加入 jerk‑L2 与动作率辅助损失。

**📊 数据集**

实验数据集包括：Libero‑Spatial、Libero‑Object、Libero‑Goal、Libero‑10 以及 Bridge 这五个标准机器人基准任务套件，全部在 50 次评估轨迹上测评。

**📈 对比分析**

比较方法：对每个模型/压缩/适配方案计算任务成功率、任务完成时间、末端执行器路径长度、关节路径长度、加速度 L2 范数、动作率等指标；结果显示：剪枝/量化在保持成功率的同时会显著增加路径长度和加速度；在视觉令牌稀疏化下加速度激增；FAST 动作编码虽略短任务时间，但加速度提升显著；适配方法可在某些指标上取得小幅提升，但往往伴随任务完成时间或其他指标的恶化。

**⚠️ 局限性**

局限性包括：1）评估主要基于仿真与有限的真实机器人测试，未对实际能耗进行直接测量；2）只考虑了部分 VLA 模型（π₀、π₀.₅、MolmoAct）和少数压缩技术；3）适配策略仅探索了简单的提示与辅助损失，缺乏更深层次的结构性改进；4）身体化效率指标虽然更贴近实际，但仍是间接代理，可能无法覆盖所有能耗相关因素。

---

## 264. Constitutive vs. Corrective: A Causal Taxonomy of Human Runtime Involvement in AI Systems

**arXiv ID:** 2603.19213 | [PDF](https://arxiv.org/pdf/2603.19213v1)

**作者:** Kevin Baum `[一作]` (German Research Center for Artificial Intelligence), Johann Laux `[通讯]` (Oxford Internet Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出基于因果结构的人工智能系统人类介入分类框架，将人类介入分为构成性(HITL)与校正性(HOTL)两大类，并细化HOTL的时间模式和认知整合模式。

**💡 创新点**

创新点在于把传统空间循环隐喻转化为因果结构，明确区分HITL与HOTL的因果角色，并将其与互补式/混合式认知整合相结合，形成四种结构组合，同时将“人类监督”定义为HOTL的规范维度。

**🔧 技术方法**

主要采用因果理论与系统架构分析方法，对人类角色的因果位置和时间维度进行理论阐述。

**📊 数据集**

无实验数据集，本文为理论/综述性工作。

**📈 对比分析**

未进行实验对比，讨论了如何在监管框架下利用因果分类评估系统是否满足人类监督要求。

**⚠️ 局限性**

局限性包括缺乏实证验证、未对不同AI系统进行经验评估、对多角色共存情形的细化仍待进一步研究。

---

## 265. AGRI-Fidelity: Evaluating the Reliability of Listenable Explanations for Poultry Disease Detection

**arXiv ID:** 2603.18247 | [PDF](https://arxiv.org/pdf/2603.18247v1)

**作者:** Sindhuja Madabushi `[一作]` (Virginia Tech), Jin-Hee Cho `[通讯]` (Virginia Tech)

**通讯引用:** 5796 | [OpenAlex ID](https://openalex.org/A5011649304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于跨模型共识与循环时间置换的可靠性评估框架，对禽类疾病检测任务中的可听解释进行可靠性评价；

**💡 创新点**

创新点在于将传统faithfulness与稳定性结合，构造基于FDR的统计显著性检验；通过跨模型共识抑制连续静态噪声；提供理论保证区分稀疏生物声学信号与持续伪影；实现无需空间真值标签的评估；

**🔧 技术方法**

使用的技术包括集成梯度（IG）解释器、多模型委员会（CNN、MLP、LSTM、ResNet）生成二值解释、交叉模型共识、层级分层、循环时间置换构造零分布、基于FDR的统计检验；与传统masking指标（AI、AD、AG、Faithfulness）对比；

**📊 数据集**

使用的数据集包括Poultry Dataset、带高频噪声的Spurious版本、SmartEars Poultry、SmartEars Denoised以及SwineCough等真实与对照数据；

**📈 对比分析**

与传统masking指标相比，新的AGRI‑Fidelity在所有数据集上更准确地区分真正的生物声学标记与伪影；在CoughLIME、AudioLIME等解释方法上显示高可靠性分数，实验表明对噪声鲁棒性好，跨模型稳定；

**⚠️ 局限性**

限制包括置换零分布主要针对连续静态伪影，对短暂非静态捷径检测不足；实验中伪影注入较为简单，缺乏更真实的噪声建模；缺乏标准化噪声基准；对不同采样率的适配仍需改进。

---

## 266. Taming Epilepsy: Mean Field Control of Whole-Brain Dynamics

**arXiv ID:** 2603.18035 | [PDF](https://arxiv.org/pdf/2603.18035v1)

**作者:** Ming Li `[一作]` (Guangzhou University), Jingqiao Dua `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于图正则化Koopman均场游戏（GK‑MFG）的闭环神经调制框架，用于抑制高维癫痫网络的发作；

**💡 创新点**

将Reservoir Computing（RC）实现Koopman算子近似、图拉普拉斯正则化与Mean Field Game（MFG）相结合，构造了可同时考虑功能拓扑、非线性动力学与分布控制的全新方法；

**🔧 技术方法**

利用RC‑ESN求解Koopman算子，采用图信号处理（PLV计算与图拉普拉斯正则化）构建脑网络，运用APAC‑Net（对抗生成网络）求解HJB‑FP耦合方程，最终得到闭环控制律；

**📊 数据集**

使用多通道癫痫EEG数据（23通道）进行模型训练与验证；

**📈 对比分析**

与传统MPC及无正则化的MFG进行对比，结果显示GK‑MFG在全脑分布重塑上实现95%以上的Wasserstein距离提升，控制输入平滑无高频抖动；

**⚠️ 局限性**

局限在于需要大量EEG训练数据来构建Koopman算子，且图拉普拉斯正则化依赖于PLV阈值选择，对不同病人的拓扑结构适应性有待进一步验证。

---

## 267. Safety-Guaranteed Imitation Learning from Nonlinear Model Predictive Control for Spacecraft Close Proximity Operations

**arXiv ID:** 2603.18910 | [PDF](https://arxiv.org/pdf/2603.18910v1)

**作者:** Alexander Meinert `[一作]` (e:fs TechHub GmbH), Alen Turnwald `[通讯]` (Ingolstadt University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于安全保证的模仿学习框架，用以实现航天器近距离操作的实时控制。

**💡 创新点**

创新点包括将控制屏障函数（CBF）和控制Lyapunov函数（CLF）统一到数据生成、训练损失和部署阶段；设计了CBF‑CLF‑信息化损失函数和一次性QP安全滤波器，显著提升数据效率并减少运行时滤波干预。

**🔧 技术方法**

核心技术包括非线性模型预测控制（NMPC）专家策略、DAgger式强化学习、神经网络（4层全连接 ReLU）、CBF‑CLF 约束QPs、以及嵌入式推理（LiteRT on ESP32）。

**📊 数据集**

使用了在Basilisk高保真模拟器中生成的81,027条专家数据，涵盖飞行场景中的球形避障区和锥形通道等安全约束。

**📈 对比分析**

与纯 NMPC、LQR+安全滤波器、普通行为克隆等方法对比，实验显示该框架在保持安全性和稳定性的同时，平均执行时间从14.3 ms降低到5.8 ms（PC）或28.1 ms（ESP32），并实现了与 NMPC 相近的任务性能。

**⚠️ 局限性**

局限性在于仍需依赖 NMPC 产生训练数据，且安全滤波器的保守性可能导致轨迹冗长；未来工作需拓展至把手与解旋等后续 OOS 任务，并进一步降低对专家数据的依赖。

---

## 268. Learn for Variation: Variationally Guided AAV Trajectory Learning in Differentiable Environments

**arXiv ID:** 2603.18853 | [PDF](https://arxiv.org/pdf/2603.18853v1)

**作者:** Xiucheng Wang `[一作]` (Xidian University), Nan Cheng `[通讯]` (Xidian University)

**通讯引用:** 15028 | [OpenAlex ID](https://openalex.org/A5050651525)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Learn for Variation（L4V）框架，通过把AAV轨迹规划转化为可微分的仿真图并利用离散伴随法（adjoint）得到精确梯度，直接训练确定性神经策略；

**💡 创新点**

核心创新在于用系统动力学的解析梯度替代稀疏奖励信号，消除RL中的高方差和信用分配困难，并在训练中加入时序平滑正则化与梯度裁剪；

**🔧 技术方法**

使用了可微分仿真、反向传播通过时间（BPTT）实现离散伴随求导、神经网络策略、正则化与梯度裁剪；

**📊 数据集**

采用合成数据集：随机布置的K个地面用户，任务量取值[0.5,1]，在多组实验中改变用户数K、场景尺寸L、单位距离增益η以及噪声功率σ²；

**📈 对比分析**

与遗传算法、DQN、A2C和DDPG等基线进行比较，L4V在任务完成时间、平均传输速率和训练成本（迭代次数与壁钟时间）方面均显著优于所有基线，且优点随问题规模增大而更为明显；

**⚠️ 局限性**

局限性包括依赖已知可微的动力学模型，未在真实硬件或多智能体协作环境中验证，且对极端大规模或复杂障碍场景的可扩展性还有待进一步研究。

---

## 269. Rethinking Uncertainty Quantification and Entanglement in Image Segmentation

**arXiv ID:** 2603.18792 | [PDF](https://arxiv.org/pdf/2603.18792v1)

**作者:** Jakob Lønborg Christensen `[一作]` (Technical University of Denmark), Christian F. Baumgartner `[通讯]` (University of Lucerne)

**通讯引用:** 4438 | [OpenAlex ID](https://openalex.org/A5006688931)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

系统评估了19种基于软最大、SSN、Prob. UNet、Diffusion等aleatoric方法与深度集成、MC Dropout、SWAG等epistemic方法在医学图像分割中的组合表现，并提出耦合度量Δ；

**💡 创新点**

提出了Δ耦合度量来量化正确与错误不确定性度量的相对性能，深入分析了认知崩溃对耦合的影响，并给出任务特定的模型选择建议；

**🔧 技术方法**

使用基于Kendall & Gal信息论框架的熵分解、ValUES实验框架进行AMB/OODD/CAL评估，结合软最大、SSN、Prob. UNet、Diffusion、集成、Dropout、SWAG等技术；

**📊 数据集**

使用LIDC-IDRI（CT肺结节）和Chaksu IMAGE（视网膜）两套多注解医学影像数据集；

**📈 对比分析**

通过AUROC、NCC、ACE等指标计算Δ，比较AU、EU、TU三种不确定性对不同任务的贡献，实验显示深度集成在性能与低耦合度上均优于其他组合；

**⚠️ 局限性**

受限于仅两套相对较小的多注解数据集、认知崩溃与模型规模相关性未充分探究、未跟踪训练过程中的耦合变化、未考虑非生成型AU方法等。

---

## 270. Improving RCT-Based Treatment Effect Estimation Under Covariate Mismatch via Calibrated Alignment

**arXiv ID:** 2603.19186 | [PDF](https://arxiv.org/pdf/2603.19186v1)

**作者:** Amir Asiaee `[一作]` (Vanderbilt University Medical Center), Samhita Pal `[通讯]` (Vanderbilt University Medical Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在随机对照试验（RCT）与大规模观测研究（OS）之间存在协变量不匹配时，通过学习共享嵌入空间实现CATE估计的方法；

**💡 创新点**

创新点在于用嵌入对齐替代传统的完整协变量插补，只需学习两个源的编码器使其输出在共同表示空间中对齐，从而减少维度和信息损失，同时保留校准的负迁移防护；

**🔧 技术方法**

采用惰性嵌入对齐（MMD/对比损失/对抗对齐）、基于校准的双阶段估计、线性与神经网络两种实现；

**📊 数据集**

在模拟数据（51种设置）和半合成IHDP基准上验证，数据包括共享、仅RCT、仅OS的协变量子集；

**📈 对比分析**

与Naive、RCT-only、共享协变量校准、插补校准、HTCE-T/DR等八种方法对比，在线性CATE情形下，所有校准方法几乎等效；在非线性CATE情形下，神经网络嵌入（-NN）在22种设置中均优于其他方法，尤其在小RCT样本量下表现突出；

**⚠️ 局限性**

局限包括：对齐误差难以在实际中评估；对齐假设需满足Lipschitz连续性；神经网络在严重分布偏移时可能失效；未给出多源扩展和正式的置信区间方法。

---

## 271. On The Effectiveness of the UK NIS Regulations as a Mandatory Cybersecurity Reporting Regime

**arXiv ID:** 2603.19084 | [PDF](https://arxiv.org/pdf/2603.19084v1)

**作者:** Junade Ali `[一作]` (Alan Turing Institute), Chris Hicks `[通讯]` (Alan Turing Institute)

**通讯引用:** 2787 | [OpenAlex ID](https://openalex.org/A5012044459)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用 FOI/FOISA 请求收集英国各 DCA 对 NIS 监管下网络安全事件的正式报告，并将收集到的数据与 2024 年 NCSC 及 CISA 的事件数据进行对比，揭示 NIS 监管下的报告不足与信息缺失。

**💡 创新点**

创新点在于：①首次通过 FOI 规避 GCHQ 免除 FOI 的限制，直接获取 DCA 级别的真实事件数据；②将 UK DCA 级别的 NIS 数据与美国 CISA 的 advisories 进行横向对比，提供跨国、跨监管框架的初始攻击向量与攻击类型视角；③系统性展示 NIS 监管下的报告缺口与研究/决策层所需的透明度不足。

**🔧 技术方法**

主要技术手段为：FOI/FOISA 文档检索与请求管理、文本分析与结构化数据抽取、攻击向量归类（基于 MITRE ATT&CK 与 VERIS 框架）、统计分析与对比、结果可视化（表格、比例统计）。

**📊 数据集**

使用的数据集包括：<br>• 2024 年各英国 DCA 的 NIS 报告（共 103 条，其中 30 条为网络安全事件）<br>• 2024 年 NCSC 高级/重要事件总数 89 条<br>• 2024 年 CISA 公开 advisories（17 条）与对应的攻击类型与初始访问向量<br>• 公开的 NCSC 及 CISA 相关年度报告与公开数据。

**📈 对比分析**

比较方法：<br>①统计 NIS 报告中网络安全事件的占比、攻击类型分布与 NCSC 报告的对照；<br>②对 CISA advisories 中的攻击类型和初始访问向量进行计数并与 NIS 数据的攻击向量做交叉比对；<br>结果显示：NIS 报告仅占 NCSC “高/重要”事件的约 33%；在 NIS 报告的网络安全事件中，医疗健康领域约 83‑92% 为勒索软件，且几乎全部具有盈利动机；CISA 报警显示人因（钓鱼、凭证滥用）与漏洞利用各占 50%。

**⚠️ 局限性**

局限性：<br>1. 仅涵盖 2024 年数据，缺乏长期趋势；<br>2. FOI 请求受限，部分 DCA 拒绝披露或仅提供高层级信息，导致攻击向量细节缺失；<br>3. NIS 监管与 NCSC 监管划分不一，跨体系对比可能存在分类差异；<br>4. 只分析公开的 CISA advisories，可能偏向高知名度攻击，未覆盖所有真实威胁；<br>5. 研究耗时且需要对请求方进行法律与流程教育，难以规模化。

---

## 272. On the Complexity of the Odd-Red Bipartite Perfect Matching Polytope

**arXiv ID:** 2603.18232 | [PDF](https://arxiv.org/pdf/2603.18232v1)

**作者:** Martin Nägele `[一作]`, Rico Zenklusen `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究奇红边二分完美匹配问题的多面体结构，证明其极限多面体具有指数规模的复杂面，且任何描述都需要大幅且多样化的系数。进一步将该结果推广到双模整数规划（bimodular IP）多面体，并给出对应的阶数与系数复杂度下界。

**💡 创新点**

提出了新的面复杂度上界，揭示奇红匹配多面体与双模整数规划多面体在极大系数多样性上的本质差异；利用偶环多面体与奇红匹配多面体之间的映射，首次证明了即使在多面体可优化的前提下，其面描述仍极其复杂。

**🔧 技术方法**

核心技术包括：
1) 通过构造奇环多面体的主导体（dominant）和对应的 C‑诱导约束，获得指数规模的高系数面；
2) 设计“可表达性”与“可表达边”概念，将面从偶环多面体迁移到奇红匹配多面体的“可表达”范式；
3) 使用矩阵变换与“等价面”分析，证明即使加入度数约束，系数仍需保持高多样性；
4) 将奇红匹配多面体嵌入双模整数规划的升维表示，继而得到双模 IP 多面体的面复杂度上界。

**📊 数据集**

本工作完全是理论分析，无实验或真实数据集；所有结果均来自组合构造与线性代数推导。

**📈 对比分析**

该论文不涉及算法实现或实验对比；主要通过理论证明展示：任何描述奇红匹配多面体的线性不等式至少需要系数绝对值 ≥ n−O(1) 且包含至少 √(n) 种不同系数；双模 IP 的面同样满足类似下界，证明其系数复杂度为 Ω(m^{1/4})。

**⚠️ 局限性**

局限性：
1) 仅给出下界，未给出与之相匹配的上界；
2) 研究对象集中在特定的奇红匹配与双模 IP，多面体的完整描述仍未解决；
3) 证明依赖于完整图构造和特殊映射，是否能推广到更一般的图结构仍是未知；
4) 仅展示了系数多样性的必要性，未提供具体有效的极大面描述或优化算法的改进方向。

---

## 273. Robustness, Cost, and Attack-Surface Concentration in Phishing Detection

**arXiv ID:** 2603.19204 | [PDF](https://arxiv.org/pdf/2603.19204v1)

**作者:** Julian Allagan `[一作]` (Elizabeth City State University), Vladimir Deriglazov `[通讯]` (Elizabeth City State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了基于手工工程化网站特征的钓鱼检测模型，并提出了一个成本感知的后部署对抗评估框架，研究了在有限攻击预算下模型的最小渗透成本、存活率和鲁棒性集中度。

**💡 创新点**

核心创新包括：①定义并使用最小渗透成本（MEC）、存活率S(B)和鲁棒性集中指数（RCI）三种诊断指标；②证明模型架构对鲁棒性的上限受特征经济学限制（低成本单一特征转移可统一限定MEC分布）；③采用离散路径搜索（均匀成本搜索）精确求解最优攻击路径，展示不同特征集和成本表对攻击可行性和集中度的影响。

**🔧 技术方法**

技术手段包括：离散图搜索（均匀成本搜索）、预算约束下的攻击图建模、成本函数与预算模型、统计诊断（MEC、S(B)、RCI、FRI）、特征经济学分析、对四种经典分类器（Logistic Regression、Random Forest、Gradient Boosted Trees、XGBoost）的统一阈值评估。

**📊 数据集**

使用UCI Phishing Websites基准数据集（11055个样本，30个三值特征），在不同特征子集（全特征、RA-8、VA-7b等）和两种成本表（base、strict）下进行实验。

**📈 对比分析**

通过在同一条件下比较四种模型的AUC、MEC、RCI、FRI以及S(B)曲线，发现所有模型在静态评估中AUC≥0.979，但在预算攻击下中位MEC均为2，RCI_3>0.78，表明鲁棒性高度集中在少数低成本表面特征上；模型架构的复杂度对鲁棒性无显著提升，表明鲁棒性上限由特征经济学决定。

**⚠️ 局限性**

局限性包括：①数据集较旧，缺少现代特征（证书透明、视觉相似性、JavaScript行为指纹等）；②攻击模型仅考虑单调编辑，未覆盖反向注入或提取器层攻击；③成本模型基于时间假设，未映射真实金钱预算；④查询限制可能导致实验结果与实际部署有差异；⑤对非单调攻击的稳健性评估留待后续工作。

---

## 274. PromptHub: Enhancing Multi-Prompt Visual In-Context Learning with Locality-Aware Fusion, Concentration and Alignment

**arXiv ID:** 2603.18891 | [PDF](https://arxiv.org/pdf/2603.18891v1)

**作者:** Tianci Luo `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**通讯引用:** 10738 | [OpenAlex ID](https://openalex.org/A5034104790)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PromptHub框架，通过多提示融合实现视觉上下文学习

**💡 创新点**

创新点在于局部感知融合、融合-利用-预测闭环三重目标与数据增强，提升融合质量和鲁棒性

**🔧 技术方法**

采用MAE‑VQGAN backbone、局部化注意力融合、三种对齐与利用损失、以及提示检索与增广技术

**📊 数据集**

使用Pascal‑5i（分割、检测）和ImageNet‑1K（色彩化）等公开数据集

**📈 对比分析**

与Prompt‑SelF、InMeMo、Condenser等基线对比，在分割、检测、色彩化任务上平均提升约2–5%（mIoU或MSE）并保持多提示的优势

**⚠️ 局限性**

局限在于仍需高质量检索、对任务切换存在一定域差距、融合提示不具备高保真图像生成能力

---

## 275. SINDy-KANs: Sparse identification of non-linear dynamics through Kolmogorov-Arnold networks

**arXiv ID:** 2603.18548 | [PDF](https://arxiv.org/pdf/2603.18548v1)

**作者:** Amanda A. Howard `[一作]` (Pacific Northwest National Laboratory), Panos Stinis `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 34991 | [OpenAlex ID](https://openalex.org/A5002562845)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

结合Kolmogorov‑Arnold网络（KAN）与稀疏识别非线性动力学（SINDy），在每个激活函数层同时进行稀疏回归，实现可解释的符号回归；

**💡 创新点**

创新点是将SINDy的稀疏回归嵌入KAN的每个激活函数，既保留深度网络的函数组合能力，又在每层强制稀疏表达，提升解释性；

**🔧 技术方法**

使用KAN架构、B‑spline激活函数、SINDy稀疏回归、shadow矩阵与L1正则、ADAM优化器以及多种候选函数库；

**📊 数据集**

在多种合成数据集上测试：符号回归任务（cos(x²+y)）、线性 ODE、阻尼摆、ABC 流、洛伦兹系统和三相 Kuramoto 振荡器；

**📈 对比分析**

与 PyTorch 版 SINDy、标准 KAN 及直接 SINDy‑KAN 对比，结果显示该方法能更准确地恢复复合函数，误差更小，训练时间与标准 KAN 相当，直接 SINDy‑KAN 在速度上更快；

**⚠️ 局限性**

局限包括输入维度增大时可解释性下降，数值微分对噪声敏感，以及当候选函数库不足时可能无法充分捕捉系统动力学。

---

## 276. ZEBRAARENA: A Diagnostic Simulation Environment for Studying Reasoning-Action Coupling in Tool-Augmented LLMs

**arXiv ID:** 2603.18614 | [PDF](https://arxiv.org/pdf/2603.18614v1)

**作者:** Wanjia Zhao `[一作]` (Stanford University), Lingjiao Chen `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个可程序化生成的诊断环境 ZebraArena，用于研究工具增强大型语言模型（LLM）在多步推理与外部动作之间的耦合关系，并提供可控难度与理论最优查询下限。

**💡 创新点**

创新点在于：①提出知识极简、可控、无污染的谜题环境；②构造缺失线索设置，使每个实例仅在获得足够工具查询后可唯一解；③给出四层诊断评估框架（必要性、有效性、效用、最优性）以及可计算的理论最优查询数，精准衡量工具使用效率。

**🔧 技术方法**

采用了工具增强推理框架（类似 ReAct），实现了事实查询与关系查询两类工具；通过JSON schema校验与规范化保证交互确定性；在实验中对比 GPT‑5、Gemini‑2.5‑Flash、Llama‑3.3‑70B 等多种模型，使用了效用指标（IR、EffRate、IG 等）。

**📊 数据集**

数据集基于公开的 ZebraLogic 逻辑谜题，随机屏蔽 1–6 条线索，生成 Small/Medium/Large 三种尺寸，累计超过 2000 个实例，完全可重现且不受原始数据污染。

**📈 对比分析**

通过对比模型的最终准确率与工具调用效率，结果显示 GPT‑5 在 Medium 难度下几乎 100% 准确，但工具调用量仍比理论最优多 70–270%；Gemini‑2.5‑Flash 需要十倍以上 token；较弱模型（如 Llama‑3.3‑70B）准确率仅 12–24%。评估表明，各模型在最优查询上仍存在显著差距，且预算/成本约束下表现不稳定。

**⚠️ 局限性**

局限性包括：①工具使用效率与理论最优差距大，模型难以在预算内实现最优策略；②缺乏对不确定性与信息价值的显式建模，导致预算/成本敏感性不佳；③实验仍局限于 Zebra puzzles，尚未验证到更广泛的推理/知识领域。

---

## 277. Align-to-Scale: Mode Switching Technique for Unimanual 3D Object Manipulation with Gaze-Hand-Object Alignment in Extended Reality

**arXiv ID:** 2603.18535 | [PDF](https://arxiv.org/pdf/2603.18535v1)

**作者:** Min-yung Kim `[一作]` (KAIST), Sang Ho Yoon `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出并评估了一种基于视线-手部-对象对齐的模式切换技术（Align-to-Scale），实现单手在XR环境中对3D对象的平移和缩放控制。

**💡 创新点**

创新点：① 将视线与手部在3D空间的对齐作为缩放模式的触发条件，解决传统Gaze+Pinch模型中单手缩放缺失的问题；② 设计了四种结合不同手势（PTZ、Push‑Pull）和对齐策略（Overlap、Angular Dispersion、Pinch）的单手缩放技术；③ 提出了针对不同任务和用户情境的设计准则。

**🔧 技术方法**

技术方法：使用眼动追踪和手部追踪实现视线-手部对齐检测；采用四种缩放手势（Pinch-to‑Zoom、Push‑Pull）与三种对齐阈值；通过控制参数（视差角、手指间距、立体视区重叠度、深度）计算缩放因子；基于Unity与Varjo XR-3 HMD、Ultraleap SDK进行实现；用户体验评估采用NASA‑TLX、主观问卷与排名。

**📊 数据集**

数据集：实验数据共20名参与者完成160个任务（5种技术×4尺度×4位置×2轮次），收集模式切换错误率、缩放误差、时长、尺度差异等指标。

**📈 对比分析**

比较方法：采用Within‑Subjects设计，使用非参数两因素重复测量ANOVA（ART）及事后对比；对比结果显示：① 传统双手缩放在准确性与模式切换错误率上最优；② 在单手技术中，PTZ‑Area在模式切换稳定性上最好，Push‑Pull‑Depth在用户偏好和缩放精度上表现最佳；③ 单手技术总体误差率高于双手，但在可行性与可用性上提供了实用的单手替代方案。

**⚠️ 局限性**

局限性：① 试验不允许“ clutching ”动作，限制了缩放的自然性；② 仅使用视觉模式指示，未测试语音或触觉反馈；③ 只评估了视线‑手部对齐的三种策略，未考虑其他潜在模式切换机制；④ 参与者主要为无障碍用户，结果对普通用户可能有差异；⑤ 单手技术导致较高的手臂疲劳，可能在长时间使用时出现“猿臂效应”。

---

## 278. Model Order Reduction of Cerebrovascular Hemodynamics Using POD_Galerkin and Reservoir Computing_based Approach

**arXiv ID:** 2603.18837 | [PDF](https://arxiv.org/pdf/2603.18837v1)

**作者:** Rahul Halder `[一作]` (SISSA), Gianluigi Rozza `[通讯]` (SISSA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文比较了基于物理的POD‑Galerkin投影和基于数据驱动的POD‑Reservoir Computing（RC）在脑血管血流非定常仿真中的性能；

**💡 创新点**

创新点在于使用多谐波多幅度训练信号一次性训练两种模型、将RC与POD相结合并与传统投影方法直接对比；

**🔧 技术方法**

采用了POD降维、Galerkin投影法以及RC（ESN）时序学习技术；

**📊 数据集**

使用高保真三维基底动脉分支CFD快照作为训练与测试数据集；

**📈 对比分析**

通过对压力、速度幅值和壁面剪切应力的L²相对误差进行定量比较，POD‑G误差约1.6‑2%，POD‑RC约2.3‑3%，两者在在线阶段均实现10²–10³倍加速；

**⚠️ 局限性**

局限性包括：POD‑G投影对离散化细节敏感、离线成本高；POD‑RC对初始上下文依赖、在更复杂几何或湍流、非牛顿流体等情形下的泛化能力待验证。

---

## 279. The Impact of Corporate AI Washing on Farmers' Digital Financial Behavior Response -- An Analysis from the Perspective of Digital Financial Exclusion

**arXiv ID:** 2603.18421 | [PDF](https://arxiv.org/pdf/2603.18421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 280. Towards Noise-Resilient Quantum Multi-Armed and Stochastic Linear Bandits

**arXiv ID:** 2603.18431 | [PDF](https://arxiv.org/pdf/2603.18431v1)

**作者:** Zhuoyue Chen `[一作]` (Sun Yat-sen University), Kechao Cai `[通讯]` (Sun Yat-sen University)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5088801513)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出噪声鲁棒的量子蒙特卡洛估计（BQMC）并基于此设计噪声鲁棒的量子多臂（NR‑QUCB）和线性（NR‑QLinUCB）bandit 算法，在 NISQ 噪声环境下提升奖励估计与决策性能。

**💡 创新点**

将贝叶斯推断与量子振幅估计相结合得到自适应噪声 QMC，并将其嵌入 UCB 与 LinUCB 框架，构建对多种 NISQ 噪声鲁棒的量子 bandit 算法；同时提供四种噪声模型下的实验验证。

**🔧 技术方法**

贝叶斯量子蒙特卡洛（BQMC）与粒子滤波、量子振幅估计（Canonical QAE 与 MLE‑QAE）、UCB/ LinUCB 决策框架、噪声模型仿真等技术。

**📊 数据集**

合成实验数据：K=3、K=10 臂奖励分布，四种噪声模型（指数退相干、读出噪声、退化化、幅度衰减）仿真，无真实物理数据集。

**📈 对比分析**

与经典 UCB/ LinUCB 及不考虑噪声的 Canonical‑QUCB/QLinUCB 进行对比；在无噪声下实现更低累积损失；在噪声环境下 NR‑QUCB/NR‑QLinUCB 显著降低累积损失，尤其优于 Canonical‑QAE 与 MLE‑QAE 版本。

**⚠️ 局限性**

仅在仿真中验证，缺乏硬件实验；理论上缺少严格的下界或收敛证明；对更复杂噪声或多体量子通道的鲁棒性未作深入分析。

---

## 281. iSatCR: Graph-Empowered Joint Onboard Computing and Routing for LEO Data Delivery

**arXiv ID:** 2603.18539 | [PDF](https://arxiv.org/pdf/2603.18539v1)

**作者:** Jiangtao Luo `[一作]` (Chongqing University of Posts and Telecommunications), Yongyi Ran `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 1018 | [OpenAlex ID](https://openalex.org/A5016656803)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种分布式图嵌入与深度强化学习相结合的 iSatCR 框架，用以联合优化低地球轨道卫星网络中的本地计算与路由，显著降低地面传输延迟与丢包率。

**💡 创新点**

创新点包括：①设计了带有移位特征聚合的图嵌入机制，实现对三跳范围内资源状态的分层感知；②在此基础上构建了分布式的 D3QN（Dueling Double Deep Q‑Network）决策模型，能够在存储约束下实时生成最佳计算-路由策略；③引入了启发式探索和动作选择改进，提升学习收敛速度与决策质量。

**🔧 技术方法**

技术手段包括：图神经网络（自定义图嵌入与消息传递）、深度强化学习（D3QN、DDQN、PPO）、多代理分布式决策、仿真平台（SimPy、Skyfield）、Python+PyTorch 实现。

**📊 数据集**

使用仿真数据集：基于四种真实 LEO 星座模型（Iridium、Telesat、OneWeb、Starlink）以及对应的地面站分布，生成的任务流、链路状态和失败模型作为实验输入。

**📈 对比分析**

比较方法：将 iSatCR 与四个基线（D3QN、DDQN、PPO、理想集中式方案）在相同任务负载、链路失效率和星座规模下进行对比。实验结果显示，iSatCR 在平均任务延迟、丢包率和平均跳数上均优于其他方法，尤其在高负载和高失效率场景下保持低延迟与高可靠性。

**⚠️ 局限性**

局限性：①评估仅基于仿真，未验证在真实卫星硬件上的可行性；②模型训练依赖大量样本与计算资源，实际部署时需考虑实时性与模型更新成本；③对极端存储约束或突发大规模链路失效的鲁棒性尚未彻底验证。

---

## 282. Cyber-Resilient Digital Twins: Discriminating Attacks for Safe Critical Infrastructure Control

**arXiv ID:** 2603.18613 | [PDF](https://arxiv.org/pdf/2603.18613v1)

**作者:** Mohammadhossein Homaei `[一作]` (Universidad de Extremadura), Mar Ávila `[通讯]` (Universidad de Extremadura)

**通讯引用:** 346 | [OpenAlex ID](https://openalex.org/A5082036181)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于物理信息的数字孪生(i-SDT)，能够区分单阶段与多阶段网络攻击并在确认攻击后使用不确定性感知模型预测控制避免系统停机，提升工业控制系统的网络安全与业务连续性。

**💡 创新点**

三大创新：① 将差分质量平衡约束融入TCN预测模型，实现对水处理过程的物理一致性校正；② 通过MMD正则化的双向GRU编码器实现三类攻击（正常、单阶段、多阶段）的判别；③ 在MPC中结合MC Dropout估计的不确定性和攻击风险，实现自适应安全裕度和无停机恢复。

**🔧 技术方法**

技术包括Temporal Convolutional Network、物理约束损失、双向GRU+MMD分类器、Monte Carlo Dropout、基于预测不确定性的风险回退MPC、以及自动微分实现的连续线性化。

**📊 数据集**

使用公开的工业水处理测试数据集SWaT（6阶段，51传感器）和WADI（3阶段，127传感器），两者均含多种单/多阶段攻击日志。

**📈 对比分析**

与四种主流异常检测器（OmniAnomaly、USAD、MTAD‑GAT、TranAD）以及DT‑MPC进行对比，i‑SDT在多类检测上F1提升约10%，误报率降低约44%，在控制恢复时将停机成本降低≈56%，且单周期推理时间约69 ms，满足1 Hz采样周期。

**⚠️ 局限性**

局限性：仅针对传感器型假数据注入，未覆盖执行器攻击；依赖高性能GPU工作站，边缘部署需进一步压缩模型；在训练数据无污染时表现最佳，对恶意数据污染的鲁棒性未知；跨域迁移需要少量目标域标注数据。

---

## 283. Beyond Passive Aggregation: Active Auditing and Topology-Aware Defense in Decentralized Federated Learning

**arXiv ID:** 2603.18538 | [PDF](https://arxiv.org/pdf/2603.18538v1)

**作者:** Sheng Pan `[一作]` (Yunnan University), Niansheng Tang `[通讯]` (Yunnan University)

**通讯引用:** 2627 | [OpenAlex ID](https://openalex.org/A5029728794)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种面向去中心化联邦学习的主动审计框架，能够在网络拓扑中主动探测并阻断隐蔽的后门攻击；

**💡 创新点**

创新点在于构建动态扩散模型、引入三种基于功能干预的审计指标（随机熵异常、随机平滑KL散度、激活峰度）以及基于多臂赌博机与拓扑感知的防御节点部署策略；

**🔧 技术方法**

技术包括图论动力学建模、统计熵与KL散度检测、激活峰度分析、MAB策略选择、基于节点度与介数的防御节点布置；

**📊 数据集**

实验使用PubMed 20k RCT数据集的Transformer文本分类和GTSRB交通标志图像分类的CNN模型；

**📈 对比分析**

与Krum、TrimmedMean、CosL2、FLAME等传统稳健聚合方法对比，MAB(TOP-AWARE)在保持接近或略低的ACC的同时，将ASR从30%级别压至约10%或更低，表现出更优的安全-性能平衡；

**⚠️ 局限性**

局限性包括主动审计所需的计算与通信开销、对极端非I.I.D.数据分布的鲁棒性有限，以及对网络拓扑连通性与节点数目假设的依赖。

---

## 284. A New Approach to Code Smoothing Bounds

**arXiv ID:** 2603.18077 | [PDF](https://arxiv.org/pdf/2603.18077v1)

**作者:** Tsuyoshi Miezaki `[一作]` (Waseda University), Katsuyuki Takashima `[通讯]` (Waseda University)

**通讯引用:** 3568 | [OpenAlex ID](https://openalex.org/A5010032290)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过引入等价划分与随机游走的视角，对代码相关的总变分距离提供了新的上界，并推广了此前基于傅里叶变换的方法。

**💡 创新点**

创新点在于不再依赖群结构，而仅需等价划分即可得到同样强度的上界，且适用于子群或其余子集的初始分布。

**🔧 技术方法**

主要技术包括等价划分理论、矩阵商的谱分析、周期化与泊松求和公式的应用，以及随机游走与卷积运算的等价关系。

**📊 数据集**

本文未使用实验数据，全部为理论推导与公式证明。

**📈 对比分析**

通过与Debris‑Alazard等人使用傅里叶变换得到的上界比较，证明所给上界在一般情形下至少不弱于原有上界，且在某些初始分布上可实现更紧的估计。

**⚠️ 局限性**

局限性在于结果仍是理论上限，缺乏对具体密码实例的实验验证；此外等价划分的构造在某些群上可能仍具有挑战性。

---

## 285. Conditional Execution of Transpiler Passes Based on Per-Script Feature Detection

**arXiv ID:** 2603.18049 | [PDF](https://arxiv.org/pdf/2603.18049v1)

**作者:** Rishipal Singh Bhatia `[一作]` `[通讯]` (Google), Rishipal Singh Bhatia (Google)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了Google Closure Compiler的转译过程，并提出基于脚本级特征检测的选择性转译模型，动态跳过不必要的AST遍历，提升编译效率。

**💡 创新点**

首次将特征跟踪与动态门控相结合，按实际使用的语言特征决定执行哪些转译通道，保持语义不变的同时显著减少冗余计算。

**🔧 技术方法**

采用静态AST特征标记、动态FeatureSet维护、Synthetic Feature传播、Transpiled‑Away Set校验、逆向通道排序等技术实现门控与安全验证。

**📊 数据集**

使用Google内部大型生产 monorepo（搜索服务、地图工具、办公套件等）中的数百万文件，涵盖 ES5–ES2022 代码。

**📈 对比分析**

对同一应用在旧的范围基模型和新模型下进行多次跑测，比较编译时间、CPU、内存等指标；结果显示转译阶段时间降至 50%，整体编译阶段平均下降 12%，30–40% 的冗余通道被跳过。

**⚠️ 局限性**

需要维护每个通道的特征映射，过细粒度的作用域级分析会增加内存和复杂度；对新出现的语言特征仍需手动更新 FeatureSet 关联。

---

## 286. A Synthesizable RTL Implementation of Predictive Coding Networks

**arXiv ID:** 2603.18066 | [PDF](https://arxiv.org/pdf/2603.18066v1)

**作者:** Timothy Oh `[一作]` (University of California), Timothy Oh `[通讯]` (University of California)

**通讯引用:** 9511 | [OpenAlex ID](https://openalex.org/A5036023937)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

**🎯 论文内容**

设计并实现了一个可综合的数字硬件子系统，能够在每个神经元核心内部执行离散时间预测编码的更新，支持监督训练和推理。

**💡 创新点**

创新点在于提供完整的可综合RTL实现，将预测编码的局部学习规则映射到硬件FSM和单MAC数据通路，强调硬件直接体现学习过程，而非仅提出新的学习算法。

**🔧 技术方法**

使用了同步时钟、单MAC顺序数据路径、IEEE‑754 单精度 HardFloat、硬连线邻层通信、可编程的 per‑neuron clamping 接口，以及固定的 FSM 调度，实现了可综合的数字电路。

**📊 数据集**

实验基于合成的教师-学生回归任务（ReLU 与 tanh 隐层）以及不同规模的网络结构（2→4→3、4→8→4、8→16→8），未使用公开的真实数据集。

**📈 对比分析**

通过 Verilator 仿真产生 CSV 学习曲线，比较不同网络尺寸的 MSE 降低情况。结果显示快速初始下降后趋于稳态残差，证明在局部更新条件下仍能实现有效学习，虽未与传统 BP 进行直接性能比较。

**⚠️ 局限性**

限制包括：逐步 MAC 造成的线性延迟随 fan‑in 增大，单精度浮点可能导致数值不稳定，缺乏理论稳定性保证，以及在大规模网络中需要更多 tick 才能达到同等推理平衡，能耗和面积等指标未给出完整评估。

---

## 287. Adaptive Regime-Aware Stock Price Prediction Using Autoencoder-Gated Dual Node Transformers with Reinforcement Learning Control

**arXiv ID:** 2603.19136 | [PDF](https://arxiv.org/pdf/2603.19136v1)

**作者:** Mohammad Al Ridhawi `[一作]` (University of Ottawa), Hussein Al Osman `[通讯]` (University of Ottawa)

**通讯引用:** 1746 | [OpenAlex ID](https://openalex.org/A5050648904)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个自适应的股票价格预测框架，利用自编码器检测市场异常并通过双节点Transformer分路预测，同时用Soft Actor‑Critic强化学习动态调整异常阈值与融合权重。

**💡 创新点**

创新点在于将无监督异常检测与双路专用Transformer相结合，并通过RL自学习阈值实现真正的自适应 regime‑aware 机制；同时提供了从预测误差中自学习区间的闭环控制。

**🔧 技术方法**

采用自编码器（Autoencoder）进行异常检测，Node Transformer 网络进行时间与图结构建模，Soft Actor‑Critic（SAC）强化学习用于阈值与融合权重调节，并融合BERT情绪分析及技术指标特征。

**📊 数据集**

使用20只标普500成分股的日级OHLCV数据（1982‑2025），附加技术指标、VIX波动率、情绪得分等多模态特征。

**📈 对比分析**

与ARIMA、VAR、随机森林、LSTM、iTransformer、HMM‑LSTM等基线模型对比，单日预测MAPE由0.80%降至0.59%，方向准确率提升至72%，在高波动期MAPE低于0.85%，表现优于所有基线。

**⚠️ 局限性**

主要局限包括样本选择偏差（仅选存活公司）、训练集情绪缺失导致情绪模型学习受限、RL超参调节敏感、模型结构复杂导致训练成本高、未考虑交易成本与执行影响，且对新市场或资产类别的泛化尚未验证。

---

## 288. Color image restoration based on nonlocal saturation-value similarity

**arXiv ID:** 2603.18586 | [PDF](https://arxiv.org/pdf/2603.18586v1)

**作者:** Wei Wang `[一作]` (Tongji University), Yakun Li `[通讯]` (Tongji University)

**通讯引用:** 7204 | [OpenAlex ID](https://openalex.org/A5100322983)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于饱和度-值相似度的非局部全变差模型，用于彩色图像的去噪和去模糊。

**💡 创新点**

创新点在于：①在HSV空间构造饱和度和价值的非局部梯度与相似度；②将该相似度嵌入全变差正则化，得到SVS‑NLTV模型；③通过Bregman化算子分裂迭代求解，同时支持L2和L1数据项，显著提高对彩色信息的捕捉和去噪效果。

**🔧 技术方法**

主要技术包括：非局部全变差（non‑local TV）、HSV彩色空间与四元数表示、Bregman迭代与算子分裂、L2/L1数据项、梯度与非局部权重的快速实现。

**📊 数据集**

使用Berkeley Segmentation Database（60幅彩色图像），在这些图像上加入高斯噪声、泊松噪声、高斯模糊、运动模糊等典型退化模型进行实验。

**📈 对比分析**

与CTV、GVTV、SVTV、NLTV等方法对比，采用PSNR、SSIM、QSSIM和S‑CIELAB等指标评估。实验结果显示SVS‑NLTV在所有指标上均优于或与其它方法持平，且在视觉质量上能够更好地保留细节与色彩，显著减少色彩交叉伪影。

**⚠️ 局限性**

局限性包括：①参数（α、μ、λ、δ等）需手动调优，影响性能；②非局部权重计算与全局迭代导致计算量和内存占用较大，适用于离线处理；③在极端噪声或高度模糊的情况下可能需要进一步改进权重构造或加入自适应机制。

---

## 289. Evolutionarily Stable Stackelberg Equilibrium

**arXiv ID:** 2603.18385 | [PDF](https://arxiv.org/pdf/2603.18385v1)

**作者:** Sam Ganzfried `[一作]` `[通讯]` (Ganzfried Research), Sam Ganzfried (Ganzfried Research)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种新的解概念——进化稳定Stackelberg均衡（SESS），并给出离散与连续博弈的求解算法；

**💡 创新点**

在传统Stackelberg演化博弈中加入ESS约束，确保跟随者策略对突变具有免疫性，并提出乐观（OSESS）与悲观（PSESS）特例；

**🔧 技术方法**

采用离散博弈的枚举支持法与QCQP求解、连续博弈的非凸QCQP+KKT+后验认证方法，利用Gurobi求解；

**📊 数据集**

实验使用一套癌症治疗的生态‑演化游戏模型，参数取自相关文献；

**📈 对比分析**

与已有的标准Stackelberg equilibrium（SE）算法比较，OSESS在质量生活函数上略高，求解时间从约2秒提升至约2.2分钟；对SE解做后验认证后发现其在该参数集下亦为OSESS；

**⚠️ 局限性**

主要局限在求解规模受限，连续OSESS求解耗时较长，且需要全局优化与后验验证，对极端参数可能不收敛。

---

## 290. SLEA-RL: Step-Level Experience Augmented Reinforcement Learning for Multi-Turn Agentic Training

**arXiv ID:** 2603.18079 | [PDF](https://arxiv.org/pdf/2603.18079v1)

**作者:** Prince Zizhuang Wang `[一作]` (Carnegie Mellon University), Shuli Jiang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 316 | [OpenAlex ID](https://openalex.org/A5013215630)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种“Step-Level Experience-Augmented Reinforcement Learning（-RL）”框架，利用大型语言模型在每个决策步基于当前观测检索相关经验，并将检索到的经验通过格式保持增强注入到提示中；同时构建自演化的经验库和观测聚类索引，实现高效检索与多层次信用分配；

**💡 创新点**

核心创新点包括①在多轮任务中首次实现步级经验检索（相较于传统一次性任务级检索更具时效性），②将结构相似的观测聚类用于检索与信用归因，③通过分数门控和速率限制实现经验库的自演化，而非梯度更新，④将步级优势与期望奖励结合，形成双层信用分配方案；

**🔧 技术方法**

主要技术包括：基于GRPO的无价值网络强化学习框架；LLM驱动的经验提取与语义摘要；观测聚类与字符串相似度检索；格式保持的提示增强；步级与期望级优势估计；以及自演化的经验库管理；

**📊 数据集**

实验数据集涵盖多轮代理基准ALFWorld、WebShop以及搜索增强问答任务，后者进一步使用NQ、HotpotQA、TriviaQA、PopQA、2Wiki、MuSiQue、Bamboogle等；

**📈 对比分析**

与多种基线对比，包括封闭源LLM、提示/记忆方法、GRPO、GiGPO、RLOO、MemRL、SkillRL等；-RL在ALFWorld上达93.5%成功率，WebShop 76.3%成功率，均超过所有基线；在搜索问答上平均得分60.9%，优于IGPO 58.7%及其他RL方法；

**⚠️ 局限性**

限制方面：步级检索与提示增强导致训练与推理开销提升；依赖外部LLM进行经验提取，提取质量不佳时可能产生噪声；经验库的维护与更新增加系统复杂度，且效果高度依赖经验质量与多样性；

---

## 291. Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models

**arXiv ID:** 2603.19183 | [PDF](https://arxiv.org/pdf/2603.19183v1)

**作者:** Aiden Swann `[一作]` (Stanford University), Mac Schwager `[通讯]` (Stanford University)

**通讯引用:** 37468 | [OpenAlex ID](https://openalex.org/A5057116316)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Vision‑Language‑Action（VLA）模型的内部激活进行稀疏自编码，提取可解释的特征并区分通用与记忆化特征。

**💡 创新点**

提出基于SAE的机制解释管线、通用性度量、特征分类与可控干预，并展示这些特征既可解释又可因果地控制机器人行为。

**🔧 技术方法**

使用稀疏自编码器（SAE）、TopK激活、特征度量、干预与实验验证等技术。

**📊 数据集**

使用LIBERO、DROID以及在LIBERO上微调的OpenVLA数据集。

**📈 对比分析**

通过特征分类统计比较不同数据集和训练方式下通用特征比例的变化；在闭环仿真中对选定特征进行干预，验证其对动作的因果影响，结果表明特征能引导期望行为，但整体比例仍偏向记忆化。

**⚠️ 局限性**

局限包括：特征可解释性不一定等价于可控性；实验仅在仿真中完成，未在真实机器人上验证；数据量有限导致对全模型可解释性不足；对复杂非线性交互的干预效果不稳定。

---

## 292. Cross-Modal Rationale Transfer for Explainable Humanitarian Classification on Social Media

**arXiv ID:** 2603.18611 | [PDF](https://arxiv.org/pdf/2603.18611v1)

**作者:** Thi Huyen Nguyen `[一作]` (L3S Research Center Leibniz University Hannover), Wolfgang Nejdl `[通讯]` (L3S Research Center Leibniz University Hannover)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种面向危机事件推文的多模态人道主义类别分类方法，并同时提供文本与图像的可解释理由。

**💡 创新点**

创新点在于通过跨模态推理从文本解释自动生成图像解释，消除图像标注成本，并构建“按设计可解释”的分类框架。

**🔧 技术方法**

采用ViLT视觉语言Transformer为主干，结合GRU、BCE+CE多任务学习以及IPOT最优传输对齐生成图像热图，随后使用提取的理由进行分类。

**📊 数据集**

主要使用公开的CrisisMMD多模态数据集进行训练与评估，并在新收集的DMD数据集上验证零样本适应能力。

**📈 对比分析**

与多种单模态、融合模型及GPT‑4等基线对比，Macro‑F1提升16–35%，文本解释Token‑F1达0.826，模型在零样本情形下准确率约80%。

**⚠️ 局限性**

局限在于依赖文本与图像高度对齐，对罕见或信息混杂的推文误判较多；图像解释质量受文本标注质量影响，部分类别样本不足。

---

## 293. Damage identification using noisy frequency response functions based on topology optimization

**arXiv ID:** 2603.18569 | [PDF](https://arxiv.org/pdf/2603.18569v1)

**作者:** Akira Saito `[一作]` (Meiji University), Hidetaka Saomoto `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 411 | [OpenAlex ID](https://openalex.org/A5040457237)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用噪声频率响应函数（FRF）通过拓扑优化（SIMP）方法对结构损伤进行识别；

**💡 创新点**

引入L1范数（Lasso）正则化项以抑制伪损伤区域，提升损伤识别的鲁棒性；

**🔧 技术方法**

使用有限元求解前向问题，拓扑优化求解后向问题，采用SIMP插值模型与Lasso正则化；

**📊 数据集**

实验测得的可压缩/加速度响应频率响应函数，包含四个带缺口的悬臂板（A1、A2、B1、B2）数据集；

**📈 对比分析**

通过对比不同目标函数（点误差MSE与MAC）以及是否加Lasso，发现加入Lasso后伪损伤显著减少，定位准确，但形状估计仍偏小；在实验噪声下仍能实现合理的损伤定位；

**⚠️ 局限性**

需手动调节正则化参数λ；对大尺寸或复杂形状损伤的恢复有限；仅验证了悬臂板，尚未扩展到更复杂结构或多种损伤类型。

---

## 294. AU Codes, Language, and Synthesis: Translating Anatomy to Text for Facial Behavior Synthesis

**arXiv ID:** 2603.18588 | [PDF](https://arxiv.org/pdf/2603.18588v1)

**作者:** Jiahe Wang `[一作]` (University of Science and Technology of China), Shangfei Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 3942 | [OpenAlex ID](https://openalex.org/A5077046519)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出利用自然语言描述的面部动作单元（AU）来驱动面部行为合成，解决传统一热编码AU方法在冲突AU上的线性不足问题。

**💡 创新点**

创新点包括：①将AU转换为可读的自然语言描述，实现对AU冲突与交互的语义建模；②构建BP4D-AUText大规模文本-图像对数据集；③提出AAAD评估指标；④设计基于解剖学先验的VQ-AUFace生成框架。

**🔧 技术方法**

技术主要包括：文本编码器（T5-large）、解剖学驱动的残差VQGAN、跨模态注意力对齐、进阶Cross‑Modal Alignment以及面部结构先验。

**📊 数据集**

使用了BP4D与BP4D+的302,169张面部图像，合成得到BP4D-AUText数据集，包含多种冲突AU组合。

**📈 对比分析**

与GLM4、MidJourney、UniPortrait、GANimation、AnyFace、Stable Diffusion等方法对比，VQ‑AUFace在图像质量指标（FID、KID、IS、LPIPS）和语义一致性指标AAAD上均优于同类方法，特别在处理冲突AU时获得最高的AU Conflict Handling Score。

**⚠️ 局限性**

局限性在于：①依赖大量预训练模型与算力；②对极端或罕见AU组合的泛化仍有限；③尚未在真实动态视频中验证生成效果。

---

## 295. A vision for a colorectal digital twin that enables proactive and personalized disease management

**arXiv ID:** 2603.18064 | [PDF](https://arxiv.org/pdf/2603.18064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 296. Resource-Constrained Joint Replenishment via Power-of-$m^{1/k}$ Policies

**arXiv ID:** 2603.18720 | [PDF](https://arxiv.org/pdf/2603.18720v1)

**作者:** Danny Segev `[一作]` (Tel Aviv University), Danny Segev `[通讯]` (Tel Aviv University)

**通讯引用:** 1817 | [OpenAlex ID](https://openalex.org/A5044792131)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对带资源约束的连续时间联合补货问题，提出了一系列新的近似算法，显著提升了已知的近似比。

**💡 创新点**

创新点在于引入通用的加权舍入框架：使用分数扩展因子、随机平移以及交错网格，从而打破传统“power‑of‑2”策略的限制，最终实现了 5/6 ln 2 ≈ 1.2023 的近似比。

**🔧 技术方法**

主要技术包括：
- 基于凸松弛的最优解舍入为分数几何网格的定量化策略；
- 随机平移的期望分析和因子揭示线性规划；
- 交错（staggered）网格以降低持有成本，并通过分析密度系数得到最优参数。

**📊 数据集**

本研究完全为理论分析，没有使用任何实验数据集；所有结果均来自严格的数学证明与线性规划实验验证。

**📈 对比分析**

与传统的 1/ln 2 ≈ 1.4427（power‑of‑2）和 √9/8 ≈ 1.0606 近似比相比，本文的 1.2023 近似比进一步提高了约 16%。

**⚠️ 局限性**

局限性：
- 仍不清楚在存在任意多资源约束的情况下，该问题是否属于 APX‑hard，或是否存在更优的多项式时间近似方案；
- 对于极大规模实例，尽管理论复杂度为多项式，但实际实现仍需进一步优化。

---

## 297. Weaver: Fuzzing JavaScript Engines at the JavaScript-WebAssembly Boundary

**arXiv ID:** 2603.18789 | [PDF](https://arxiv.org/pdf/2603.18789v1)

**作者:** Lingming Zhang `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 7893 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种针对 JavaScript 与 WebAssembly 交互边界的灰盒模糊测试框架，专门发现跨语言漏洞。

**💡 创新点**

创新点在于引入双类型（Dual‑Type）分析，实现 JS 与 Wasm 变量的双向类型映射；以及使用 UCB‑1 多臂赌博算法动态调度生成器与变异器，以最大化路径发现率。

**🔧 技术方法**

核心技术包括基于 Fuzzilli 的中间表示 FuzzIL 扩展、类型推断与转换规则、Wasm 模块生成（调用 wasm‑smith）、以及覆盖率引导的多臂赌博调度。

**📊 数据集**

使用三大主流 JS 引擎（SpiderMonkey、V8、JavaScriptCore）作为测试目标；无种子输入，全部从零生成。

**📈 对比分析**

与 Fuzzilli（旧版与新版）及 Dharma 进行对比，覆盖率（行、分支等）提升约 8‑9%，并在长周期 fuzzing 中发现两处未知漏洞，其中一处被评为最高危害级别。

**⚠️ 局限性**

局限性包括：生成案例有效率略低（受 wasm‑smith 运行时错误、类型系统不精确及值追踪缺失影响）；部分生成器导致超时；缺乏对未来 Wasm 扩展（如多内存、GC）的完整支持。

---

## 298. Empathetic Motion Generation for Humanoid Educational Robots via Reasoning-Guided Vision--Language--Motion Diffusion Architecture

**arXiv ID:** 2603.18771 | [PDF](https://arxiv.org/pdf/2603.18771v1)

**作者:** Fuze Sun `[一作]` (University of Liverpool), Xinyu Fan `[通讯]` (University of Liverpool)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于推理的视觉-语言-运动扩散框架（RG-VLMD），用于生成情感化、与教学意图一致的共语手势，支持人机教育互动。

**💡 创新点**

创新点在于将情感维度（Valence/Arousal）估计与教学行为（教学动作）映射相结合，并在扩散生成器中加入多层次的教学意图（片段级与帧级）条件与辅助动作分类监督，提升手势的语义一致性与可控性。

**🔧 技术方法**

技术核心包括：XGBoost混合专家情感估计、基于CLIP的视觉-语言编码、LLM推理生成教学动作向量、FiLM/加法条件的扩散运动生成器（RAPID‑Motion）以及局部注意力机制和动作分类头。

**📊 数据集**

使用的数据集：CMU‑MOSEI（情感估计训练）和BEAT（共语手势训练），并在NAO机器人上进行实时实验。

**📈 对比分析**

对比方法包括基线扩散模型DSG+；评估指标为手势幅度、速度、jerk、能量等统计量以及配对距离热图。结果显示，在加入教学动作条件后，手势在不同教学意图下呈现更明显的分离，语义一致性和表现力显著提升。

**⚠️ 局限性**

局限性包括：1）情感估计基于预训练模型，可能对教育场景特定的情绪表达缺乏细粒度；2）扩散模型训练对数据量和计算资源需求较高；3）仅针对上半身手势，缺乏全身运动与物理约束；4）缺少大规模真实课堂用户研究验证。

---

## 299. Sparse3DTrack: Monocular 3D Object Tracking Using Sparse Supervision

**arXiv ID:** 2603.18298 | [PDF](https://arxiv.org/pdf/2603.18298v1)

**作者:** Nikhil Gosala `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**通讯引用:** 2594 | [OpenAlex ID](https://openalex.org/A5039639553)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出首个稀疏监督的单目3D目标跟踪框架，利用2D查询匹配与3D几何估计生成高质量伪标签；

**💡 创新点**

通过将稀疏标注与时空一致性结合，设计自我匹配、支持匹配等采样策略，并用自监督深度与关键点回归解决3D偏航估计；

**🔧 技术方法**

基于DINOv2特征、Transformer查询匹配、卷积自监督深度网络、信息熵对比损失及False Negative Compensation模块；

**📊 数据集**

在KITTI Tracking与nuScenes车辆类别数据上进行实验，稀疏标注至每条轨迹最多4个标签；

**📈 对比分析**

与SPOT、SAM2结合FCOS3D/CenterNet，在CenterTrack/DEFT上对比，稀疏标注下MOTA提升至KITTI 15.5pp、nuScenes 9.9pp，超过部分全标注基线；

**⚠️ 局限性**

DINOv2单尺度特征难检测远小目标且边界物体特征弱，需要更强大的多尺度 backbone。

---

## 300. BenchBrowser -- Collecting Evidence for Evaluating Benchmark Validity

**arXiv ID:** 2603.18019 | [PDF](https://arxiv.org/pdf/2603.18019v1)

**作者:** Harshita Diddee `[一作]` (Carnegie Mellon University), Daphne Ippolito `[通讯]` (Carnegie Mellon University)

**通讯引用:** 5102 | [OpenAlex ID](https://openalex.org/A5022994077)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一款名为 BenchBrowser 的检索工具，利用自然语言查询快速从 20+ 公开基准集合中检索与特定使用场景相关的评估条目，帮助实践者在没有预定义方案的前提下检视并诊断基准的内容有效性与聚合有效性。

**💡 创新点**

创新点在于：
1) 通过 LLM 生成多样化检索锚点（如技能+关键字、合成格式、结构化标签）提升检索覆盖率；
2) 采用语义嵌入+Faiss 高效检索并结合 LLM 判断过滤，避免传统词典检索的语义盲区；
3) 通过检索得到的条目直接映射到原始基准与指标，实现对基准内部构造（题型、评分规则、内容维度）的可视化审计，首次将基准有效性验证交互化、可追溯化。

**🔧 技术方法**

技术栈包括：
- LLM（如 GPT‑4）用于查询重写、合成示例、结构化标签化；
- ICL‑embedding 模型（如 OpenAI Embedding API）生成语义向量；
- Faiss 做高效近邻检索；
- LLM judge 过滤无关条目；
- 统计评估指标（Precision@k、Recall、Kendall’s τ 等）与人类标注对照。

**📊 数据集**

数据来源：
- 70k 评估条目，来自 20+ 公开基准（如 MMLU、ARC、BIG‑bench、HellaSwag 等）；
- 10 位 NLP 从业者提供的 4260 条 relevance 评注；
- 9 位从业者贡献的 20 条“新颖”查询；
- 若干金标用例与对应基准集合，用于验证检索与真实性能的相关性。

**📈 对比分析**

比较方法：对比 Random、BM25、无重写锚点等基线，使用 Precision@k 与排名相关性（Spearman ρ、Kendall τ）。结果显示：重写锚点方法在 Precision@k 上提升 10‑20 分，整体检索精度约为 0.70‑0.80；人类评估与 LLM judge 相关性高，表明自动过滤可接受。对检索条目进行模型排名实验，Kendall τ 的差异（Δ）能揭示聚合有效性失衡，说明检索方法可用于诊断基准间的差异。

**⚠️ 局限性**

局限性：
- 依赖 LLM 的重写与判断，易受模型偏好与训练数据限制，导致对极端专业或模糊查询的召回不足；
- 仅覆盖已有基准条目，无法检测基准缺失的全新维度；
- 过滤阶段可能过于保守，影响最终召回率；
- 人类评注仍具主观性，难以完全验证自动评判的绝对准确性；
- 评估侧重内容与聚合有效性，未系统探讨其他有效性维度（如预测有效性、构造有效性）。

---

## 301. V-Dreamer: Automating Robotic Simulation and Trajectory Synthesis via Video Generation Priors

**arXiv ID:** 2603.18811 | [PDF](https://arxiv.org/pdf/2603.18811v1)

**作者:** Songjia He `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 13191 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个全自动从自然语言指令生成物理真实、可执行的机器人操纵环境与轨迹的完整流程；

**💡 创新点**

首次将视频生成模型作为运动先验，并通过Sim-to-Gen视觉-运动对齐实现从2D视觉梦境到3D可执行轨迹的闭环；

**🔧 技术方法**

结合LLM+3D生成模型+视频生成模型、CoTracker3+VGGT视觉-运动对齐、逆运动学等技术；

**📊 数据集**

使用内部生成的合成数据（约600条/小时），无外部公开数据集；

**📈 对比分析**

在模拟台面抓放任务中，使用ACT策略训练，500条到2500条生成轨迹时成功率从3.5%提升至约37%；在真实机器人上采用单轨迹zero‑shot训练，获得约30%的成功率，展示了良好的零样本跨域迁移；

**⚠️ 局限性**

仅限刚体桌面抓放任务，轨迹质量过滤不完善，无法处理关节或柔性物体，且生成过程中对环境物理一致性的约束仍需进一步强化。

---

## 302. SVLAT: Scientific Visualization Literacy Assessment Test

**arXiv ID:** 2603.19000 | [PDF](https://arxiv.org/pdf/2603.19000v1)

**作者:** Patrick Phuoc Do `[一作]` (University of Notre Dame), Chaoli Wang `[通讯]` (University of Notre Dame)

**通讯引用:** 3068 | [OpenAlex ID](https://openalex.org/A5101913449)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套针对普通公众的科学可视化素养评估工具SVLAT。

**💡 创新点**

首次提供了标准化、心理测量学基础的科学可视化素养量表，兼顾多种技术与任务类型，且公开了完整试题库。

**🔧 技术方法**

采用多阶段构建流程（构念定义、蓝图设计、内容效度评估、试点、试验、IRT与CTT分析）及贝叶斯2参数逻辑模型进行项目参数估计。

**📊 数据集**

使用19幅代表8类科学可视化技术的图像/动画（来自官方机构、科研论文、教育网站及自制ParaView渲染）组成的题库，共计49道题。

**📈 对比分析**

通过与先前InfoVis评测（如VLAT、Mini‑VLAT）对标，利用Cronbach α 与 McDonald’s ω 证明内部一致性良好（α=0.81，ω=0.82），IRT信息曲线显示覆盖广泛能力区间，表明量表在可靠性和测量范围上均优于单一信息可视化测评。

**⚠️ 局限性**

局限在于仅考察静态/短动画的解释能力，未覆盖交互式探索、参数调整和不确定性表示；量表长度较长，适用场景受限。

---

## 303. MeInTime: Bridging Age Gap in Identity-Preserving Face Restoration

**arXiv ID:** 2603.18645 | [PDF](https://arxiv.org/pdf/2603.18645v1)

**作者:** Teer Song `[一作]` (Beijing University of Posts and Telecommunications), Yasen Zhang `[通讯]` (Xiaomi Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MeInTime，一种面向跨年龄参考图像的基于扩散模型的面部恢复方法；

**💡 创新点**

核心创新在于将身份信息与年龄信息解耦：通过专门的注意力机制注入身份嵌入、引入Gated Residual Fusion模块调和结构与身份特征，并在推理阶段使用Age‑Aware Gradient Guidance实现无监督的年龄一致性控制；

**🔧 技术方法**

技术路线主要基于Stable Diffusion 2.1 与 DiffBIR 结构，使用面部识别模型提取身份嵌入、双向交叉注意力、Depthwise‑Separable 卷积的GRF模块，以及训练无关的年龄梯度引导；

**📊 数据集**

训练集使用VGGFace2‑HQ和CelebRef‑HQ（约18万张图像，5,405个身份），交叉年龄评估使用AgeDB（1.6万张图像，约100个身份，平均年龄差26岁）；

**📈 对比分析**

与现有参考无关与参考有关的BFR方法（如CodeFormer、DiffFace、DMDNet、Ref‑LDM、RestorerID、FaceMe）比较，MeInTime在同龄恢复的PSNR、SSIM、LPIPS、IDS等指标均位列首位，跨年龄恢复在年龄一致性（AGE）上显著优于所有基线（MAE仅为7.65），且在视觉质量和身份保持上保持竞争力；

**⚠️ 局限性**

局限性包括：需要至少一张高质量参考图像；跨年龄极端差异（>40岁）仍会出现细微失真；年龄估计依赖数值提示，可能对特殊年龄特征（如极年轻或极老）适配不足；

---

## 304. Some structural properties of mixed orthogonal arrays and their irredundancy

**arXiv ID:** 2603.18568 | [PDF](https://arxiv.org/pdf/2603.18568v1)

**作者:** Maryam Bajalan `[一作]` (Institute of Mathematics and Informatics), Ferruh Özbudak `[通讯]` (Sabancı University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究混合正交数组（MOA）的结构性质，推导了其Singleton上界，并给出了MDS与almost‑MDS MOA的特征；提出了基于迹映射的MOA双重性，将MOA与误差块码相互对应；进一步研究了不可冗余混合正交数组（IrMOA），并证明其在极端情况t=⌊s/2⌋下与MDS误差块码等价，从而构造了在非均匀系统中的最小支撑绝对最大纠缠态。

**💡 创新点**

创新点在于：①首次为MOA给出Singleton型上界及其与MDS、almost‑MDS的完整判定；②提出迹双重性，将MOA的强度与误差块码的π距离联系；③建立MOA与误差块码之间的结构对应，阐明了线性MOA与线性误差块码的等价；④证明IrMOA在t=⌊s/2⌋时与MDS误差块码等价，为构造最小支撑绝对最大纠缠态提供理论基础。

**🔧 技术方法**

主要技术包括有限域扩张与自正交基的构造、迹映射与矩阵Gram理论、线性代数中的秩–零空间定理、投影与正交码的欧氏双重性、以及组合学中的多重数组计数与最小指数分析。

**📊 数据集**

本研究为纯理论研究，无采用具体实验数据集；所有结果均通过数学证明与构造例子（如MOA(8,5,…), MOA(16,4,…), IrMOA(16,6,…) 等）进行验证。

**📈 对比分析**

比较方法主要是与已知的正交数组、经典MDS码的Singleton界以及误差块码的π距离等传统理论做对比；并通过构造示例展示等价性与极限情况的实现。由于缺乏数值实验，未给出性能指标。

**⚠️ 局限性**

局限性包括：①迹双重性需要有限域具有自正交基（即q为偶数或q、m均为奇数）；②IrMOA结果仅在t≤⌊s/2⌋且满足特定块大小和索引条件下成立；③对非线性MOA或更一般的混合结构未给出完整的结构描述；④缺乏实验或数值评估，难以验证在实际量子态构造中的性能表现。

---

## 305. Fire as a Service: Augmenting Robot Simulators with Thermally and Visually Accurate Fire Dynamics

**arXiv ID:** 2603.19063 | [PDF](https://arxiv.org/pdf/2603.19063v1)

**作者:** Anton R. Wagner `[一作]` (Kiel University), Xuesu Xiao `[通讯]` (George Mason University)

**通讯引用:** 1994 | [OpenAlex ID](https://openalex.org/A5017662025)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Fire as a Service (FaaS)，一种异步共仿框架，将高保真热力学火灾模拟与多种机器人仿真引擎（Isaac Sim、Gazebo、MuJoCo）实时耦合。

**💡 创新点**

创新点在于：1）采用松耦合异步通信，实现火灾动力学与机器人控制的零阻塞；2）提供热辐射虚拟传感器和热剂量评估，用于路径规划和安全评估；3）兼容多引擎，支持可视化渲染与热量感知的统一接口。

**🔧 技术方法**

使用技术包括：Fire-X 的混合 Eulerian/ Lagrangian 火焰求解器、GPU 加速、ROS2 主题通信、alpha‑mat 体积渲染、指数移动平均滤波、A* 路径规划与热剂量积分。

**📊 数据集**

主要使用内部合成数据：多火源场景的 RGB/深度图像、热辐射成本图、虚拟传感器读数；未引用公开数据集。

**📈 对比分析**

通过与仅提供视觉效果的仿真对比，FaaS 在 100–120 ms 的端到端延迟下实现高频控制；热剂量评估与预期高度匹配；在多引擎、多火源场景中保持稳定性能。

**⚠️ 局限性**

局限性在于：目前未实现机器人对火灾的双向影响（如热损伤、结构退化）；缺乏同步锁步模式；未集成气体浓度、能见度等额外传感器。

---

## 306. From Noise to Signal: When Outliers Seed New Topics

**arXiv ID:** 2603.18358 | [PDF](https://arxiv.org/pdf/2603.18358v1)

**作者:** Evangelia Zve `[一作]`, Jean-Gabriel Ganascia `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于文档出现、主题创建与首次融合的新闻轨迹分类法，并在81天连续的法语氢能新闻语料库上进行验证。

**💡 创新点**

创新点在于将文档级别的先行异常（anticipatory outliers）与主题形成时间关联，提供提前预警信号，并通过累计聚类与主题对齐实现动态主题追踪。

**🔧 技术方法**

采用11种预训练语言模型生成文本嵌入，使用UMAP降维后结合HDBSCAN/OPTICS密度聚类，随后利用Hungarian算法进行主题对齐。

**📊 数据集**

使用了由1616篇标题+描述组成的81天连续法语氢能新闻数据集。

**📈 对比分析**

通过多维度嵌入、聚类算法与对齐阈值的组合实验，平均轮廓系数最高达0.653；在二分类任务中，最佳多数模型一致率为0.95，Fleiss κ约为0.33。

**⚠️ 局限性**

局限性包括仅适用于法语单一领域、每日聚合粒度可能遗漏长周期主题变化、模型依赖度高且未实现实时预测评估。

---

## 307. SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation

**arXiv ID:** 2603.19053 | [PDF](https://arxiv.org/pdf/2603.19053v1)

**作者:** Phuc Pham `[一作]` (Qualcomm), Phong Nguyen `[通讯]` (Qualcomm)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种两阶段的 3D 服装生成框架 SwiftTailor，先使用 PatternMaker 通过多模态输入生成 2D 缝制图样，再利用 GarmentSewer 将图样转换为无物理仿真即可直接渲染的 3D 网格。

**💡 创新点**

核心创新在于引入了 Garment Geometry Image（GGI）这一中间表示，它将缝制图样的语义、几何和缝合信息统一编码为 2D 图像，既保持了传统缝制图样的可解释性，又兼容 3D 表面学习；同时设计了轻量级的 PatternMaker 与 GarmentSewer 模块，实现高效实时生成。

**🔧 技术方法**

技术包括：轻量级多模态大语言模型 InternVL‑3‑2B 用于 PatternMaker，ViT‑L 编码器+多尺度解码器的 DPT 架构用于 GarmentSewer；训练采用边缘加权回归、缝合 Chamfer 损失与法线正则化；使用 GGI 进行逆映射与动态拼接。

**📊 数据集**

在 Multimodal GarmentCodeData（GCD‑MM）数据集上训练与评估，包括图像、文本与混合输入。

**📈 对比分析**

与现有基于 GarmentCode 物理仿真的方法及大规模 VLM 基线（AIpparel、ChatGarment、SewingLDM）相比，SwiftTailor 在 MMD、COV 等质量指标上取得更优表现，且推理时间从几秒降至 0.02 秒，速度提升约 4 倍。

**⚠️ 局限性**

局限性包括：对极其复杂的服装结构仍需手工调参；GGI 的统一 UV 空间在极大多样化服装时可能出现像素化失真；未能对纹理与皱纹细节进行精细化优化，需进一步结合物理或后处理提升真实感。

---

## 308. Where are the Hidden Gems? Applying Transformer Models for Design Discussion Detection

**arXiv ID:** 2603.18393 | [PDF](https://arxiv.org/pdf/2603.18393v1)

**作者:** Lawrence Arkoh `[一作]` (North Carolina State University), Wesley K. G. Assunção `[通讯]` (North Carolina State University)

**通讯引用:** 1538 | [OpenAlex ID](https://openalex.org/A5039130090)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估并比较了多种 transformer 模型在跨域设计讨论检测任务中的表现，使用 Stack Overflow 作为训练集，评估 GitHub PR、issue、commit 等不同域的数据；同时探究了同义词注入作为数据增强的效果。

**💡 创新点**

①将大型 transformer 与小型 transformer（LaMini-Flan-T5-77M）和 ChatGPT-4o-mini 应用于设计讨论检测；②改正了之前工作中方法学缺陷；③系统评估同义词注入对跨域性能的影响。

**🔧 技术方法**

BERT、RoBERTa、XLNet、LaMini-Flan-T5-77M、ChatGPT-4o-mini 的微调和推理；同义词注入（基于预训练 XLNet 生成同义词替换）；使用 ROC‑AUC、精确率、召回率、F1、Accuracy 等评估指标，并通过 Wilcoxon 符号秩检验进行统计显著性检验。

**📊 数据集**

Stack Overflow 问题/回答/评论（训练/验证/测试），以及 GitHub 数据集：Brunet（commit/issue/PR）、Viviani（PR）和 SATD（代码注释）。

**📈 对比分析**

对同一模型在训练域与跨域数据上分别跑十次，取平均值；与传统 SVM/决策树等基线比较；结果表明 transformer 在同域任务中 ROC‑AUC >0.95；在跨域任务中 RoBERTa/XLNet 仍优于传统模型，但精确率与召回率均低于同域；ChatGPT-4o-mini 召回最高但精确率最低；LaMini-Flan-T5-77M 轻量但精确率较高；同义词注入对性能几乎无提升。

**⚠️ 局限性**

①仅限于 GitHub PR/issue/commit 及 Stack Overflow，未涵盖邮件、Slack 等沟通渠道；②未进行超参数搜索，可能未达到最优；③同义词注入方法过于简单，未探索更高级的迁移学习或领域自适应；④数据集标签主观性可能影响结果；⑤ChatGPT 依赖 API，成本高。

---

## 309. SAVeS: Steering Safety Judgments in Vision-Language Models via Semantic Cues

**arXiv ID:** 2603.19092 | [PDF](https://arxiv.org/pdf/2603.19092v1)

**作者:** Carlos Hinojosa `[一作]` (King Abdullah University of Science and Technology), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 24981 | [OpenAlex ID](https://openalex.org/A5024763828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究通过文本、视觉标记和认知提示等语义干预，调节多模态视觉语言模型（VLM）在具身场景中的安全决策，并通过新构建的SAVeS基准评估其对视觉归因与行为拒绝的敏感性。

**💡 创新点**

创新点在于提出了语义调度框架与SAVeS基准，区分行为拒绝、视觉归因与虚假拒绝，并展示语义线索能够双向操纵安全行为，揭示VLM对视觉标记的过度依赖。

**🔧 技术方法**

使用了多模态干预技术（文本、视觉、认知提示）、自动化调度管线（Guardian、Auditor、Attacker）以及LLM-as-Judge评估框架，对VLM进行实验比较。

**📊 数据集**

采用了现有的MSSBench-Embodied数据集和新构建的SAVeS数据集，这两个基准均提供安全与不安全的图像-指令对，用于系统性评估语义干预效果。

**📈 对比分析**

与多种开源VLM（Qwen3-VL、DeepSeek-VL、LLaVA-HF）基线进行对比，结果表明视觉标记+关注提示能显著提升拒绝率，但同时导致高假拒绝率；自动化管线收益有限；攻击管线能极大提升拒绝率但安全对齐下降。

**⚠️ 局限性**

局限性包括：VLM对视觉标记的依赖导致缺乏真实视觉理解；实验仅覆盖合成图像和有限安全场景，无法验证在真实复杂环境中的泛化；自动化管线效果受辅助模块质量限制，难以保证稳健性。

---

## 310. Agentic Flow Steering and Parallel Rollout Search for Spatially Grounded Text-to-Image Generation

**arXiv ID:** 2603.18627 | [PDF](https://arxiv.org/pdf/2603.18627v1)

**作者:** Ping Chen `[一作]` (Harbin Institute of Technology), Yongyong Chen `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 4059 | [OpenAlex ID](https://openalex.org/A5031480448)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练、闭环的文本到图像生成框架AFS-Search，结合Agentic Flow Steering和Parallel Rollout Search实现实时纠正与多路径搜索。

**💡 创新点**

创新点在于：①将生成过程视为决策过程，通过VLM实时评估并动态修正ODE轨迹；②利用Agentic Flow Steering在Latent空间上进行能量导向的速度场调节；③设计并行分支搜索（基线、探索、纠正）并用VLM奖励选择最佳路径；④提供高效与高性能两版（AFS-Search-Pro/Fast）。

**🔧 技术方法**

核心技术包括FLUX.1-dev的流匹配模型、Vision‑Language Model（CLIP/其他大模型）作为语义评判者、SAM3做空间掩码、能量梯度映射到速度场的时间缩放控制，以及基于多分支短期模拟的奖励驱动路径选择。

**📊 数据集**

使用T2I-CompBench、GenEval、R2I-Bench等公开基准数据集，评估属性绑定、空间关系、复杂度、对象计数、颜色等指标。

**📈 对比分析**

与多种基线（DALL‑E 2、SDXL、PixArt‑α、FLUX、Qwen‑Image、SDv3.5、以及Agentic框架如ConPreDiff、RPG、EvoGen、T2I‑R1、AgentComp等）对比，AFS‑Search‑Pro在T2I‑CompBench平均分从0.7736提升至0.8847（提升约14%），AFS‑Search‑Fast在速度与质量之间取得折中，仍实现显著性能提升；在R2I‑Bench上平均得分从0.33提升至0.48。

**⚠️ 局限性**

局限性包括：①相较原始FLUX.1‑dev推理速度慢，尤其是Pro版；②依赖外部VLM和SAM的推理成本，可能导致推理时间增加；③虽然是无训练框架，但对VLM的质量与推理时间敏感；④对极端复杂场景仍可能需要更多搜索分支或更长模拟步数，进一步增加推理开销。

---

## 311. UT-ACA: Uncertainty-Triggered Adaptive Context Allocation for Long-Context Inference

**arXiv ID:** 2603.18446 | [PDF](https://arxiv.org/pdf/2603.18446v1)

**作者:** Lang Zhou `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 22009 | [OpenAlex ID](https://openalex.org/A5108050904)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在推理阶段根据每个 token 的不确定性动态调整上下文窗口的框架——UT‑ACA，支持回滚并在上下文不足时重新生成；

**💡 创新点**

创新点包括：① 用 logit margin 与隐藏层语义嵌入双重信号通过轻量化双编码器+LSTM 估计 token 级生成难度；② 将生成难度指标（三分类）作为触发条件，自动决定是否扩展上下文、回滚并重生成；③ 在保持生成质量的同时显著降低平均上下文使用量和计算成本；

**🔧 技术方法**

核心技术：轻量化不确定性检测器（双编码器+LSTM）、KV 缓存块检索与 block‑based 上下文选择、回滚‑重生成策略、logit margin 与语义嵌入融合、三分类生成难度输出；

**📊 数据集**

使用合成传记摘要数据集进行检测器训练与验证，并在 LongBench、∞‑Bench 等公开长文本基准上评测；

**📈 对比分析**

方法上与固定大小上下文窗口和全上下文（no‑select）对比，实验显示 UT‑ACA 在保持或提升生成质量的前提下，平均上下文使用量显著下降、计算延迟降低；

**⚠️ 局限性**

局限性：依赖训练好的不确定性检测器，迁移到不同模型或任务时需要重新训练；块大小与检索策略对不同长文本可能不均衡；在极长或复杂 OOD 场景下仍可能出现未检测到的不确定性；实验规模主要集中在合成数据和公开基准，真实应用验证仍待深入。

---

## 312. RADIUS: Ranking, Distribution, and Significance - A Comprehensive Alignment Suite for Survey Simulation

**arXiv ID:** 2603.19002 | [PDF](https://arxiv.org/pdf/2603.19002v1)

**作者:** Weronika Łajewska `[一作]` (Amazon), Saab Mansour `[通讯]` (Amazon)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5070002963)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了RADIUS框架，用于评估大语言模型（LLM）生成的问卷模拟结果的多维对齐情况。

**💡 创新点**

创新点在于同时考虑排名对齐和分布对齐，并加入显著性检验，形成一套可解释、敏感且可复现的评价体系。

**🔧 技术方法**

使用Spearman相关、Bootstrap置信区间、TVD、卡方同质性检验等统计方法，对LLM生成的多选单选问卷进行评估。

**📊 数据集**

在社交调查、政治观点、家庭与饮食等多领域的300+个问题数据集（包括GSS、OpinionQA等）上进行实验。

**📈 对比分析**

与传统的准确率、KL/JS散度、Wasserstein距离等单一指标相比，RADIUS在区分LLM与基准模型、揭示不同失败模式方面表现更强，能够捕捉更细粒度的差异。

**⚠️ 局限性**

局限性包括仍聚焦于单选多选问题，对开放式问题的适用性未知，以及排名与分布指标之间的权衡尚未系统化。

---

## 313. Towards Interpretable Foundation Models for Retinal Fundus Images

**arXiv ID:** 2603.18846 | [PDF](https://arxiv.org/pdf/2603.18846v1)

**作者:** Samuel Ofosu Mensah `[一作]` (Hertie Institute for AI in Brain Health), Philipp Berens `[通讯]` (Hertie Institute for AI in Brain Health)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种面向视网膜眼底图像的可解释基础模型。

**💡 创新点**

双向可解释性：局部类证据图和全局二维投影空间。

**🔧 技术方法**

采用BagNet架构结合t‑SimCNE自监督对比学习，生成可视化的2D投影。

**📊 数据集**

在EyePACS、AREDS、UK Biobank共802,360张眼底图像上预训练，评估APTOS、IDRiD、DeepDRiD、Messidor‑2、Glaucoma、PAPILA、FIVES等任务。

**📈 对比分析**

与ImageNet、SimCLR、RETFound等基准比较，性能相当或更优，且参数量仅为对手的1/16，推断时间更快。

**⚠️ 局限性**

局限在于未针对每个下游任务做细粒度超参数搜索，且模型仅针对眼底图像，未扩展至OCT等多模态。

---

## 314. ClawTrap: A MITM-Based Red-Teaming Framework for Real-World OpenClaw Security Evaluation

**arXiv ID:** 2603.18762 | [PDF](https://arxiv.org/pdf/2603.18762v1)

**作者:** Haochen Zhao `[一作]` (National University of Singapore), Shaoyang Cui `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套名为ClawTrap的MITM红队框架，用于评估OpenClaw在真实网络环境下的安全性。

**💡 创新点**

创新点在于将网络层MITM攻击引入agent安全评估，提供实时、可定制的三类攻击模式（静态替换、iframe弹窗注入、动态内容修改），并展示模型在不同规模下的信任失衡。

**🔧 技术方法**

技术包括使用mitmdump、Tailscale私有P2P隧道、本地拦截引擎、规则驱动的拦截/转换/审计流水线以及FastAPI蜜罐服务器。

**📊 数据集**

使用的“数据集”是基于真实网页（如bbc.com、google.com）的实时流量，并配合合成凭证和伪造页面内容进行攻击演示；未使用公开标准评测数据集。

**📈 对比分析**

通过对比不同规模LLM在同一攻击场景下的输出（信任度、异常归因、回退策略）进行定性对照，结果表明强模型能更好地识别和规避MITM攻击，弱模型易产生错误信息。

**⚠️ 局限性**

局限性包括实验仅为两种攻击演示、缺乏大规模量化评测、攻击模式有限、未覆盖更复杂任务（如身份验证、交易）等。

---

## 315. Controller Datapath Aware Verification of Masked Hardware Generated via High Level Synthesis

**arXiv ID:** 2603.18939 | [PDF](https://arxiv.org/pdf/2603.18939v1)

**作者:** Nilotpola Sarma `[一作]`, Chandan Karfa `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于状态划分的正式验证方法 MaskedHLSVerif，用于检测高层合成（HLS）生成的掩码硬件的侧信道安全性。

**💡 创新点**

创新点在于针对 HLS 输出的资源共享数据路径和控制器 FSM 产生的误报，采用按 FSM 状态拆分、递归标签化以及状态级验证的策略，成功消除传统工具的假阳性，并能捕捉 HLS 优化导致的掩码缺陷。

**🔧 技术方法**

核心技术包括基于 REBECCA 的 SAT‑based 标签传播、FSM 状态提取与拆分、递归输入标签传递，以及对稳定/瞬态探测模型的形式化验证。

**📊 数据集**

实验使用六个加密基准：DOM、COMAR、HPC1、HPC2 的级联掩码乘法器，以及使用 DOM 和 HPC1 掩码的 PRESENT S‑box，均通过 Vitis HLS 生成 RTL。

**📈 对比分析**

与传统 REBECCA 工具相比，MaskedHLSVerif 在所有基准上均无误报且能识别 HLS 表达式平衡导致的安全缺陷；验证时间在几百毫秒至几秒之间，验证成本相对可控。

**⚠️ 局限性**

局限性包括仅验证一阶掩码、依赖 REBECCA 的实现、仅在有限基准上评估，且对更高阶掩码或其他 HLS 生成器的适用性仍待进一步验证。

---

## 316. TENSURE: Fuzzing Sparse Tensor Compilers (Registered Report)

**arXiv ID:** 2603.18372 | [PDF](https://arxiv.org/pdf/2603.18372v1)

**作者:** Kabilan Mahathevan `[一作]` (Virginia Tech), Kirshanthan Sundararajah `[通讯]` (Virginia Tech)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5028997869)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本研究开发了首个可扩展的黑盒 Fuzzer，用 Einstein‑Summation（einsum）符号生成稀疏张量编译器（STC）的合法内核，并通过变形测试验证其功能正确性。

**💡 创新点**

创新点包括：① 约束满足算法保证 100% 语义合法的 einsum 生成；② 针对 STC 的域特定变异算子，利用张量乘法交换性与存储格式异构性产生等价变体；③ 语言无关的 JSON 抽象层，使任何 STC 只需实现轻量级翻译即可接入。

**🔧 技术方法**

采用了约束求解、语义等价变异、黑盒差分/变元测试技术，并在 C++/Julia 等主机语言下通过 JSON 中介完成编译与执行。

**📊 数据集**

数据集为自生成的随机稀疏张量（尺寸与非零结构随机抽样），未使用公开数据集；所有评测均在这些自制张量上完成。

**📈 对比分析**

与基于 Grammarinator 的语法生成对比，后者仅 3.3% 的样本能通过语义检查；在 6 小时内对 TACO 生成 267k 样本，发现约 65% 失效，其中 18% 为错误代码；对 Finch 仅生成 1.6k 样本，发现 57 个崩溃，证明该 Fuzzer 在不同 STC 上均能高效发现缺陷。

**⚠️ 局限性**

局限性包括：① 仅为黑盒方法，难以定位根因；② 对 Finch 编译延迟导致吞吐低；③ 浮点运算的非确定性需容忍阈值，可能掩盖细微错误；④ 缺乏重复错误去重机制；⑤ 仍未覆盖 MLIR Sparse Dialect、PyTorch Sparse 等新平台。

---

## 317. Enhancing the Parameterization of Reservoir Properties for Data Assimilation Using Deep VAE-GAN

**arXiv ID:** 2603.18766 | [PDF](https://arxiv.org/pdf/2603.18766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 318. MemArchitect: A Policy Driven Memory Governance Layer

**arXiv ID:** 2603.18330 | [PDF](https://arxiv.org/pdf/2603.18330v1)

**作者:** Lingavasan Suresh Kumar `[一作]` (Arizona State University), Rong Pan `[通讯]` (Arizona State University)

**通讯引用:** 15321 | [OpenAlex ID](https://openalex.org/A5075012459)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MemArchitect，一个将记忆生命周期、真实性、一致性和效率等治理维度化的中间件层，将 LLM 的记忆从被动存储转变为主动裁决；

**💡 创新点**

通过四大治理柱（生命周期与卫生、一致性与真实性、来源与信任、效率与安全）以及 FSRS 记忆衰减、Kalman 过滤、投票门控、Hebbian 图扩展等新技术，构建了主动记忆裁决框架，解决了传统 RAG 的治理缺口；

**🔧 技术方法**

采用 FSRS 记忆衰减、熵触发整合、Kalman 过滤、交叉编码投票门、Multi‑Hop 递归拆分、Adaptive Token Budget、Hebbian 图扩展以及 Auction 经济模型等技术；

**📊 数据集**

在 LoCoMo‑10 基准上使用 Qwen‑3B 与 Llama‑3.1‑8B 进行评估，并计划在 LongMemEval、PreFEval、PersonaMem 等更长序列和 persona 任务上进一步验证；

**📈 对比分析**

与 SimpleMem、MemOS 等基线在 LoCoMo‑10 上进行对比，平均提升 7.45% 的准确率；在 Qwen‑3B 的多跳、时序推理等任务中分别提升 29% 以上，证明治理驱动的裁决优于压缩方法；

**⚠️ 局限性**

治理策略导致对单次出现的时间敏感事实进行激进裁剪，导致时间推理任务召回下降；裁剪力度需调节；目前仅在两款模型上测试，尚未在更大规模或更长序列的场景中验证。

---

## 319. Green Architectural Tactics in ML-enabled Systems: An LLM-based Repository Mining Study

**arXiv ID:** 2603.18734 | [PDF](https://arxiv.org/pdf/2603.18734v1)

**作者:** Vincenzo De Martino `[一作]` (University of Salerno), Fabio Palomba `[通讯]` (University of Salerno)

**通讯引用:** 9866 | [OpenAlex ID](https://openalex.org/A5033738898)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对205个开源机器学习项目进行仓库挖掘，使用大型语言模型自动识别代码中实现的绿色架构技巧，并评估现有30项技巧的采用情况，发现9项之前未被记录的新技巧。

**💡 创新点**

提出了一种基于提示工程的LLM检测机制，可在无标注数据的情况下识别绿色技巧，并通过半自动化流程将LLM输出转化为可验证的技术清单，扩充了现有绿色技巧目录。

**🔧 技术方法**

利用Claude 3 Haiku、GPT‑4o、Qwen3‑8B和DeepSeek‑R1‑8B等大型语言模型，对每个Python文件进行单次推理，结合自定义提示实现技巧检测和新技巧生成。

**📊 数据集**

构建了205个从GitHub筛选的、包含模型训练代码的机器学习项目数据集（来源为Gonzalez、NICHE和其他公开数据集），并对其基本指标（年龄、星标、贡献者等）进行了统计。

**📈 对比分析**

在Oracle‑based验证中，Claude 3 Haiku在已知技巧检测上的准确率达到96.8%，GPT‑4o为93.7%，Qwen3‑8B为91.3%，DeepSeek‑R1‑8B为90.5%；在新技巧挖掘上，Claude 3 Haiku通过人工评估筛选出558个实例，最终归纳出9个可靠的新技巧，覆盖数据预处理、训练加速、资源管理等方面。

**⚠️ 局限性**

局限性包括：仅分析Python源代码，无法捕捉基于数据、部署配置或运行时信息的技巧；采用单一LLM，可能存在模型偏差；文件级推理忽略跨文件上下文；对新技巧的识别仍需人工校验，无法保证召回率；且未对技巧对能耗或碳排放的实际影响进行量化。

---

## 320. GAPSL: A Gradient-Aligned Parallel Split Learning on Heterogeneous Data

**arXiv ID:** 2603.18540 | [PDF](https://arxiv.org/pdf/2603.18540v1)

**作者:** Zheng Lin `[一作]` (University of Hong Kong), Xianhao Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 1973 | [OpenAlex ID](https://openalex.org/A5083484070)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种梯度对齐的并行拆分学习框架（GAPSL），实现了在无需客户端模型聚合的情况下，针对异构数据提升模型训练稳定性和收敛速度。

**💡 创新点**

创新点在于：① 通过领导梯度识别（LGI）动态挑选方向一致的梯度构造全局导向；② 采用梯度方向对齐（GDA）加入方向感知正则化，过滤并对齐各客户端梯度，缓解梯度方向不一致导致的训练发散。

**🔧 技术方法**

使用技术包括：模型分割与拆分学习、基于角度的梯度一致性评分、适应性梯度选取比例、可调阈值的角度过滤以及余弦正则化项；实现基于 PyTorch 的分布式微服务架构。

**📊 数据集**

数据集：CIFAR‑10 与 CIFAR‑100，采用 IID 与 Dirichlet α=0.1（高度非均匀）两种分布，模型分别为 VGG‑16 与 Vision Transformer‑Base。

**📈 对比分析**

与 Vanilla SL、SFL、EPSL、PSL 等四种基线比较，GAPSL 在 IID 与非 IID 条件下均实现最高测试准确率，收敛速率最快，收敛时间最短，尤其在高异构环境下提升幅度显著。

**⚠️ 局限性**

局限性在于：对角度阈值与梯度选取比例的超参数仍需经验调优；当数据分布极端偏斜或设备数量极大时，角度分布可能出现多模态，导致选取机制失效；且正则化需额外计算，略微增加客户端训练负担。

---

## 321. A Concept is More Than a Word: Diversified Unlearning in Text-to-Image Diffusion Models

**arXiv ID:** 2603.18767 | [PDF](https://arxiv.org/pdf/2603.18767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 322. Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization

**arXiv ID:** 2603.18388 | [PDF](https://arxiv.org/pdf/2603.18388v1)

**作者:** Shiyan Liu `[一作]` (University of California), Rui Qu `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 1199 | [OpenAlex ID](https://openalex.org/A5044959056)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对自动提示优化中黑盒、标签缺失导致的失败，提出 VISTA 框架，通过多代理解耦假设生成与提示重写，提供可解释的优化轨迹。

**💡 创新点**

核心创新是将假设生成与提示改写分离，利用可验证的语义标签、两层探索-利用机制和并行验证，解决种子陷阱、归因盲区、轨迹模糊、跨模型易失问题。

**🔧 技术方法**

使用多代理系统、启发式假设集、随机重启、epsilon‑greedy 采样、Pareto 池、语义追踪树等技术。

**📊 数据集**

在 GSM8K 与 AIME2025 两个算术推理数据集上进行实验。

**📈 对比分析**

与无优化基线和 GEPA 对比，在缺陷种子下 VISTA 从 13.5% 提升至 87.57%，并在所有种子和跨模型评估中保持稳健，优于 GEPA。

**⚠️ 局限性**

局限性包括对启发式集合的依赖、需要手工设计类别、以及在极端非结构化任务上收益相对有限。

---

## 323. Evaluating Model-Free Policy Optimization in Masked-Action Environments via an Exact Blackjack Oracle

**arXiv ID:** 2603.18642 | [PDF](https://arxiv.org/pdf/2603.18642v1)

**作者:** Kevin Song `[一作]` (University of Alabama at Birmingham), Kevin Song `[通讯]` (University of Alabama at Birmingham)

**通讯引用:** 3203 | [OpenAlex ID](https://openalex.org/A5014570399)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了无限鞋的 Vegas 规则下的 Blackjack，并构建了精确 DP oracle 与基于此的评估框架。

**💡 创新点**

创新点在于提供可验证的离散动态控制基准，并证明在无计数负期望环境下最优下注必须为桌面最小赌注。

**🔧 技术方法**

采用动态规划、REINFORCE、SPSA、CEM 三种无模型优化器，配合指数移动平均基线与熵正则化。

**📊 数据集**

数据集为 4,600 个决策单元的离散状态空间，使用无限鞋抽牌模型模拟的 10^6 次手牌。

**📈 对比分析**

与 oracle 进行动作匹配率、EV 与回调差异比较，REINFORCE 在 10^6 手牌下获得 46.37% 匹配率与 -0.04688 EV，优于 SPSA 与 CEM。

**⚠️ 局限性**

限制在于表格学习导致状态稀疏、动态动作遮蔽难以探索，未能实现 100% 策略收敛，且仅在无限鞋 i.i.d. 环境中验证。

---

## 324. PlanTwin: Privacy-Preserving Planning Abstractions for Cloud-Assisted LLM Agents

**arXiv ID:** 2603.18377 | [PDF](https://arxiv.org/pdf/2603.18377v1)

**作者:** Guangsheng Yu `[一作]` (University of Technology Sydney), Xu Wang `[通讯]` (University of Technology Sydney)

**通讯引用:** 3085 | [OpenAlex ID](https://openalex.org/A5100708433)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Privacy-Preserving Planning Abstractions（PPPA）的架构，允许云端大型语言模型在不暴露本地原始环境信息的前提下进行规划；

**💡 创新点**

创新点在于通过四阶段本地投影管道将真实环境转化为去标识、基于模式的数字双胞胎，结合基于能力的规划接口和多轮泄露预算控制，实现了对规划时可观测信息的严格裁剪；

**🔧 技术方法**

采用了小型语言模型（SLM）进行结构提取、正则表达式及关键词规则的敏感实体识别、值泛化与固定JSON模式投影；同时使用门控器进行能力验证、策略检查和预算计量；

**📊 数据集**

使用了包含60个人工合成代理任务、覆盖十个不同领域（如编码、文档审查、调试、DevOps、数据管道、安全响应等）的测试集；

**📈 对比分析**

与四种前沿云规划器（Kimi K2.5、Gemini 3 Flash、MiniMax M2.5、GLM 5）以及原始上下文、PII去标识、纯本地模型等基线进行对比；结果显示PPPA在保证完整敏感项不泄露（SND=1.0）的同时，计划质量（PQS>0.79）与全上下文几乎持平；

**⚠️ 局限性**

局限包括：需要在本地进行语义抽象，若抽象不够丰富可能影响规划效果；多轮交互时累计预算可能导致信息不足；系统依赖于精确的能力定义与策略，缺乏对结构推断泄露的完整理论保障；

---

## 325. NANOZK: Layerwise Zero-Knowledge Proofs for Verifiable Large Language Model Inference

**arXiv ID:** 2603.18046 | [PDF](https://arxiv.org/pdf/2603.18046v1)

**作者:** Zhaohui Geoffrey Wang `[一作]` `[通讯]` (University of Southern California), Zhaohui Geoffrey Wang (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了可验证LLM推理的零知识证明系统，使用户能够在不泄露模型权重的前提下验证输出是否来自指定模型的计算。

**💡 创新点**

创新点在于层级拆分证明框架、固定大小证明、零精度lookup表逼近非算术操作，以及利用Fisher信息指导的局部验证策略。

**🔧 技术方法**

采用Halo2+IPA零知识证明、SHA-256 commitment链、16-bit lookup表逼近、Fisher信息量度、Rust实现的多线程并行证明。

**📊 数据集**

在GPT-2系列（124M、355M）以及TinyLLaMA-1.1B（1.1B）模型上使用WikiText-2等公开数据集进行评估。

**📈 对比分析**

与主流工具EZKL对比，单层证明时间43秒、证明大小6.9KB、验证时间23ms，速度提升52×（在更大模型可达228×），且精度保持零困惑度降。

**⚠️ 局限性**

局限包括证明速度仍远慢于原生推理（约3.2分钟对比3秒），Fisher指导仅提供概率保证，且实现仅在CPU上，GPU加速尚未实现。

---

## 326. cuGenOpt: A GPU-Accelerated General-Purpose Metaheuristic Framework for Combinatorial Optimization

**arXiv ID:** 2603.19163 | [PDF](https://arxiv.org/pdf/2603.19163v1)

**作者:** Yuyang Liu `[一作]` `[通讯]` (Independent Researcher), Yuyang Liu (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出 cuGenOpt，一个支持多种编码、双层算子选择、用户算子注册、Python JIT 接口和 LLM 辅助建模的通用 GPU 加速 metaheuristic 框架。

**💡 创新点**

创新点在于三层自适应设计（静态先验+动态算子选择+硬件感知），共享内存自动扩展与 L2 缓存感知种群调度，以及通过 JIT 与 LLM 低门槛交互。

**🔧 技术方法**

采用 CUDA 并行“块进化一解”架构、双层 AOS、共享内存/ L2 适配、用户自定义算子 JIT 注入、Python JIT 编译链与 LLM 语言建模助手。

**📊 数据集**

使用 TSPLIB、Augerat VRP、Solomon VRPTW、QAPLIB、OR‑Library JSP、Pisinger Knapsack 等标准实例进行实验。

**📈 对比分析**

与通用 MIP 求解器（SCIP/CBC）和专用求解器（OR‑Tools Routing、NVIDIA cuOpt）在 30/60 秒等时间限制下对比，cuGenOpt 在大多数实例中显著降低目标值差距（≤10%）并在 T4/V100/A800 上实现数倍吞吐，且在 12 种问题上实现全局最优。

**⚠️ 局限性**

局限性包括多 GPU 方案仅做独立搜索缺乏跨 GPU 交换；L2 人口调度有时过度或不足；CUDA Graph 在 n≥1200 时不兼容；Solution 结构大小受限；用户自定义算子仍需 CUDA 编码；LLM 辅助功能在复杂约束下仍有限。

---

## 327. Emergence of Phase Transitions in Complex Contagions

**arXiv ID:** 2603.18380 | [PDF](https://arxiv.org/pdf/2603.18380v1)

**作者:** Saurabh Sharma `[一作]` (University of California), Ambuj Singh `[通讯]` (University of California)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出统一传播–级联模型（UP模型），将传播概念视为与节点特征同空间的高维向量，并通过传播亲和力、本地影响和全局影响三者的加权阈值实现复杂传播。

**💡 创新点**

创新点：①把传播概念转化为高维特征向量，天然融入节点属性；②整合三类影响力（亲和力、局部强化、全局激励）形成统一阈值决策；③利用MCMC采样得到随机马尔可夫级联过程，揭示相位转移与临界点；④在核心–外周结构中系统分析孵化与扩散机制。

**🔧 技术方法**

技术：拉普拉斯谱嵌入生成节点特征；优先连接（PA）网络构造；高维向量相似度计算；阈值函数与加权和；马尔可夫链与Metropolis采样；参数敏感性与相位转移分析。

**📊 数据集**

数据集：合成PA网络（1000节点，r=2）；真实社交网络Epinions与Ciao用于验证级联分布。

**📈 对比分析**

比较方法：在相同网络和参数下进行多次MCMC模拟，统计级联规模分布、时间到病毒、不同种子位置与参数值的影响；与传统阈值/简单传播模型对比，显示双峰分布与相位转移；性能指标包括病毒级联比例、平均触达时间，实验表明核心节点更易触发病毒级联，早期增长率可用于提前预测。

**⚠️ 局限性**

限制：①模型主要在PA网络结构验证，缺乏对其他拓扑（小世界、随机图等）的泛化验证；②真实数据噪声、动态变化、并发级联处理不足；③需要手动设定α、β、γ等参数，对不同场景的自动调优未研究；④假设传播向量与节点特征同空间可能不适用于所有应用场景。

---

## 328. Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation

**arXiv ID:** 2603.19220 | [PDF](https://arxiv.org/pdf/2603.19220v1)

**作者:** Zhuolin Yang `[一作]`, Wei Ping `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了 Nemotron-Cascade‑2 30B Mixture‑of‑Experts（MoE）模型，并通过 Cascade Reinforcement Learning（Cascade RL）与 on‑policy distillation 等技术实现了高质量的后训练流程。

**💡 创新点**

创新点包括：① 将 RL 按领域分阶段顺序化（domain‑wise sequential RL），显著降低灾难性遗忘；② 在 Cascade 过程中引入 on‑policy distillation，稳定并复原前期训练中可能出现的性能退化；③ 将多领域 RL 合并为单一阶段，提升训练效率；④ 结合多轮生成‑选择‑提交（generate‑select‑submit）等自适应推理策略，在 IMO、IOI、ICPC 等顶级竞赛中实现金牌；⑤ 对所有关键技术（IF‑RL、RLHF、Long‑Context RL、Code RL、SWE RL 等）进行统一的端到端评估与对比。

**🔧 技术方法**

核心技术包括：Cascade RL（顺序领域训练）、on‑policy distillation、IF‑RL、RLHF、Long‑Context RL、Code RL、SWE RL、Tool‑Integrated Reasoning（TIR）、自适应推理框架（生成‑验证‑细化、生成‑选择‑提交），以及使用 GRPO 的 on‑policy 优化算法。

**📊 数据集**

使用的数据集主要有：Nemotron‑Cascade‑2‑SFT‑Data（涵盖数学、编码、科学、长文本、通用聊天等多领域 SFT 语料），Nemotron‑Cascade‑2‑RL‑Data（包括 IF‑RL、RLHF、Long‑Context RL、Code RL、SWE RL 等多种 RL 环境），以及公开的竞赛数据集（IMO 2025、IOI 2025、ICPC World Finals 2025、LiveCodeBench、Codeforces、AIME、HMMT、MMLU、IFBench、ArenaHard 等基准评测数据。

**📈 对比分析**

与同级别或更大规模的模型对比，Nemotron‑Cascade‑2‑30B‑A3B 在 IMO 2025、IOI 2025、ICPC World Finals 2025 中取得金牌；在数学推理、编码推理、对齐与指令遵循、长上下文理解等多项基准上均实现或接近最先进性能；与 Qwen3.5‑35B‑A3B、Nemotron‑3‑Super‑120B‑A12B 等模型相比，尤其在 TIR 场景下表现出色，甚至在一些困难题目上实现了 0% 以上的 Pass@1；在知识密集型与代理任务方面仍略逊于更大规模模型。

**⚠️ 局限性**

主要局限性包括：① 在极高知识密集度和代理任务（agentic）上仍落后于 100B 以上规模模型；② 训练与推理成本仍高，尤其是对长上下文与多轮推理的算力需求；③ 需要更丰富的领域数据与更细粒度的 RL 环境以进一步提升跨领域泛化；④ 目前模型对非常难的单一问题可能无法自我纠错；⑤ 公开数据和代码虽然开放，但对大规模实验仍有硬件门槛。

---

## 329. Act While Thinking: Accelerating LLM Agents via Pattern-Aware Speculative Tool Execution

**arXiv ID:** 2603.18897 | [PDF](https://arxiv.org/pdf/2603.18897v1)

**作者:** Yifan Sui `[一作]` (Shanghai Jiao Tong University), Yuqing Yang `[通讯]` (Microsoft Research)

**通讯引用:** 2027 | [OpenAlex ID](https://openalex.org/A5101421201)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对LLM Agent的工具调用模式进行抽象与预测，利用模式感知的投机式工具执行（PASTE）在LLM与工具之间实现时间重叠，从而显著降低端到端延迟并提升工具执行吞吐量。

**💡 创新点**

① 引入Pattern Tuple抽象，将控制流与数据流分离，能够在不暴露LLM内部信息的情况下准确预测下一个工具及其参数；② 采用风险感知的机会主义调度器，只在空闲资源与预算允许时执行投机任务，并在需要时即时抢占，保证权威工具不被拖慢；③ 通过离线模式挖掘与在线实时匹配，兼顾预测精度与系统吞吐。

**🔧 技术方法**

模式挖掘（PrefixSpan+符号映射）、Pattern Tuple抽象、离线/在线预测模块、基于概率与资源预算的机会主义调度器、预执行结果缓存与优先级提升机制、工具沙箱与侧信任策略。

**📊 数据集**

DeepResearchBench、SWE‑bench、ScholarQA（用于训练与评估Agent任务）以及公开的 Azure Functions 触发器日志（模拟真实请求到达）。实验还使用了三款Agent实现：VirtualLab、Qwen Deep Research、gemini‑cli。

**📈 对比分析**

在相同硬件与LLM配置下，将PASTE与ORION（基于DAG的服务器无服务器执行）以及SpecFaaS（针对服务器无服务器的投机执行）进行对比。结果显示：平均任务完成时间降低48.5%，工具执行吞吐提高1.8×；E2E 95/99%尾延迟分别降低48.6%/61.9%；工具延迟平均降低55.2%。在并发与突发负载下，PASTE仍能保持至少1.76×/2.05×的相对加速，并且在误预测时的资源占用可控。

**⚠️ 局限性**

① 预测精度受限于模式覆盖率，开放式探索型任务的Top‑1准确率仅约27%；② 需要收集并持续更新历史执行日志以保证模式有效，新增工作负载时需重新挖掘；③ 对有副作用的工具仍需严格策略与沙箱，投机失败时的重试与恢复成本存在；④ 目前对LLM计算资源的抢占仍基于预估，极端高负载下可能出现资源竞争；⑤ 需要在不同硬件/云平台上进一步验证可移植性。

---

## 330. Pólya Thresholds Graphs

**arXiv ID:** 2603.18452 | [PDF](https://arxiv.org/pdf/2603.18452v1)

**作者:** Jinghan Yu `[一作]` (Queen's University), Bahman Gharesifard `[通讯]` (Queen's University)

**通讯引用:** 2775 | [OpenAlex ID](https://openalex.org/A5022064746)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

引入了一种基于Pólya urn的随机阈值图模型，并对其生成过程进行了定义。

**💡 创新点**

创新点在于将Pólya强化机制与阈值图结构结合，得到可解析的随机图，并推导了其度分布、拉普拉斯谱及其特征向量。

**🔧 技术方法**

采用了Pólya urn理论、Beta-二项分布、可交换性、谱分析以及线性平均动力学等技术。

**📊 数据集**

采用合成数据，即在Pólya urn过程中产生的节点序列，未使用真实网络数据。

**📈 对比分析**

通过仿真与理论推导的结果对比，显示仿真值与理论预测高度吻合，且对有限记忆版本的收敛速度作了评估。

**⚠️ 局限性**

限制方面包括只考虑两色Pólya urn、阈值图结构限定、对有限记忆模型的分析仍缺乏闭式解析，以及未验证在实际网络中的适用性。

---

## 331. Path-Constrained Mixture-of-Experts

**arXiv ID:** 2603.18297 | [PDF](https://arxiv.org/pdf/2603.18297v1)

**作者:** Zijin Gu `[一作]` (Apple), Navdeep Jaitly `[通讯]` (Google)

**通讯引用:** 33475 | [OpenAlex ID](https://openalex.org/A5112445699)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 MoE 变压器中引入块级共享路由器，限制专家路径空间并提升模型性能。

**💡 创新点**

创新点在于将路由参数按连续层块共享，既保持跨层协同又避免了传统独立路由导致的路径爆炸，同时实现了无辅助负载平衡损失的训练。

**🔧 技术方法**

采用 MoE Transformer、top‑k 路由、块级共享路由器、经验路由熵计算、跨层路径一致性评估以及路径集中度分析等技术。

**📊 数据集**

主要使用 Fineweb‑100B（0.9B 模型训练）、DCLM‑Pro、MMLU、ARC‑Challenge、CommonsenseQA、TriviaQA（16B 模型）以及八项下游任务（ARC‑E、BoolQ、HellaSwag、LAMBADA、OpenBookQA、PIQA、SocialIQA、WinoGrande）。

**📈 对比分析**

与独立路由、递归路由、低秩路由、全共享决策等基线比较，Block‑Shared‑MoE 在 0.9B 模型上平均准确率提升约 0.7%（最高 49.62%）且无辅助损失；在 16B 模型上 10/12 任务获胜，平均提升约 2.1%，推理吞吐与内存几乎无差异。

**⚠️ 局限性**

局限性包括：仅在 token‑choice 路由场景下有效；块大小需任务调优；对专家‑choice 路由未显著提升；可能不适用于极端深度或非 Transformer 结构。

---

## 332. State Complexity of Shifts of the Fibonacci Word

**arXiv ID:** 2603.18858 | [PDF](https://arxiv.org/pdf/2603.18858v1)

**作者:** Delaram Moradi `[一作]` (University of Waterloo), Ingrid Vukusic `[通讯]` (University of York)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了斐波那契单词的移位序列的状态复杂性，证明了对于任意固定的移位数c，生成移位序列的确定性有限自动机的状态数为O(log c)。

**💡 创新点**

创新点在于首次证明了斐波那契单词的移位序列在最小状态复杂性方面的界限，且该界限在msd-first和lsd-first输入格式下均成立。

**🔧 技术方法**

使用了状态复杂性技术和丢番图逼近的混合方法，并利用自动定理证明器Walnut进行部分证明。

**📊 数据集**

研究中使用了斐波那契单词的特性，特别是其在Zeckendorf表示法下的性质。

**📈 对比分析**

与其他方法相比，本文的方法在状态复杂性上表现出O(log c)的增长，这接近于非周期移位序列的理论最小值，且在某些情况下，msd-first的状态复杂性可能会更高。

**⚠️ 局限性**

限制在于虽然证明了O(log c)的状态复杂性，但对于更复杂的移位序列或其他类型的自动序列，可能需要进一步的研究来确定其状态复杂性。

---

## 333. PeriphAR: Fast and Accurate Real-World Object Selection with Peripheral Augmented Reality Displays

**arXiv ID:** 2603.18350 | [PDF](https://arxiv.org/pdf/2603.18350v1)

**作者:** Yutong Ren `[一作]` (University of Michigan), Michael Nebeling `[通讯]` (University of Michigan)

**通讯引用:** 2877 | [OpenAlex ID](https://openalex.org/A5000831539)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发并评估了 PeriphAR，一种在单眼低FOV AR 显示器上通过周边视觉提供选择反馈的技术。

**💡 创新点**

将目标对象的颜色增强与周边视觉预注意特性相结合，提出了基于最近相似色的增强策略（MSC），并通过模拟未来AR眼镜实现了无缝感知。

**🔧 技术方法**

使用 Quest Pro 的视频透视与眼动追踪、图像量化与颜色差异计算、YOLO11n 实时检测、深度学习分割、颜色增强算法等。

**📊 数据集**

虚拟的 Tetris 方块、虚拟水果模型以及真实环境中的物体检测（YOLO11n）。

**📈 对比分析**

在两项实验中对比 snapshot、text、color、shape、baseline、screenshot 与 MSC，结果显示 MSC 在错误率、主观信心、易识别度上优于其他条件，颜色选择与任务完成时间也符合预注意颜色规律。

**⚠️ 局限性**

依赖单眼 FOV 限制、颜色增强对多色对象的局限、需要更精细的分割与纹理保留、对光照与遮挡敏感、未在高速运动或安全关键场景验证。

---

## 334. ADMM-Based Distributed MPC with Control Barrier Functions for Safe Multi-Robot Quadrupedal Locomotion

**arXiv ID:** 2603.19170 | [PDF](https://arxiv.org/pdf/2603.19170v1)

**作者:** Yicheng Zeng `[一作]` (Virginia Tech), Kaveh Akbari Hamed `[通讯]` (Virginia Tech)

**通讯引用:** 1131 | [OpenAlex ID](https://openalex.org/A5066501467)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一个完全去中心化的基于控制障碍函数的模型预测控制框架，用于多机器人腿式系统的安全轨迹规划。

**💡 创新点**

创新点在于通过节点–边拆分的ADMM形式，将中心化的CBF‑MPC显式分解为节点本地和边本地的二次规划，实现完全对称的并行计算。

**🔧 技术方法**

使用的技术包括控制障碍函数 (CBF)、模型预测控制 (MPC)、交替方向乘子法 (ADMM)、分布式二次规划 (QP) 以及分层控制结构（高层DMPC、中层NMPC、低层WBC）。

**📊 数据集**

实验数据集为两台 Unitree Go2 四足机器人在室内实验室的真实障碍物、粗糙地形以及外部扰动环境，另外在 RaiSim 中使用四台机器人与十个障碍的仿真场景。

**📈 对比分析**

与中心化 CBF‑MPC 进行对比，平均规划时间在两机器人时降低约 16%，四机器人时降低约 51%，且轨迹质量与安全性与中心化方案基本一致。

**⚠️ 局限性**

局限性在于目前仅验证 2–4 台机器人的规模，扩展到更大队伍的可行性尚未证明，并且依赖离线/云端计算资源，缺乏完全分布式嵌入式实现。

---

## 335. Learning-Augmented Algorithms for $k$-median via Online Learning

**arXiv ID:** 2603.18157 | [PDF](https://arxiv.org/pdf/2603.18157v1)

**作者:** Anish Hebbar `[一作]`, Debmalya Panigrahi `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种在线学习视角的学习增补算法框架，并在k‑中值聚类问题上实现了子线性遗憾和与后验最优解相当的平均性能。

**💡 创新点**

创新点包括将学习增补问题视为在线学习，设计可扩展的双曲熵正则化在线镜像下降算法和针对逐步扩大的凸空间的分阶段比较策略；同时给出最优竞争比与遗憾的下界。

**🔧 技术方法**

采用分阶段在线镜像下降与双曲熵正则化、分数化松弛、在线整数化取样与贪心取样相结合的四步算法。

**📊 数据集**

使用多组合成数据集：均匀方形、多个簇、振荡实例、均匀球面以及尺度变化的簇群。

**📈 对比分析**

与最优固定解以及理论上O(k)与O(1)的竞争比进行比较，实验显示平均近似比随时间趋近1，随机化算法常数因子逼近最优，且在动态变化场景下能自动适应、甚至优于固定最优。

**⚠️ 局限性**

仅适用于度量空间问题，无法直接推广至覆盖类等离散优化；且对数据规模、维度的依赖尚未充分评估。

---

## 336. SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

**arXiv ID:** 2603.18423 | [PDF](https://arxiv.org/pdf/2603.18423v1)

**作者:** Minjun Kim `[一作]` (Seoul National University), U Kang `[通讯]` (Seoul National University)

**通讯引用:** 29688 | [OpenAlex ID](https://openalex.org/A5065423939)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无数据的量化方法，通过生成合成数据并对其进行精细处理，提升预训练模型在低位宽量化下的准确率。

**💡 创新点**

创新点在于三方面：①使用低通滤波器去除合成样本中的高频噪声；②对齐预训练模型与量化模型的类激活图（CAM），确保量化模型关注正确图像区域；③对难度样本仅使用软标签，避免误标签导致的误导。

**🔧 技术方法**

主要技术包括：基于批归一化统计的噪声优化合成数据；频域低通滤波；Grad-CAM 生成激活图并计算均方误差对齐损失；使用 KL 散度进行知识蒸馏；对难度阈值进行动态交叉熵剔除。

**📊 数据集**

实验数据集包括 CIFAR-10、CIFAR-100、ImageNet（分类）以及 Vision Transformer 在 ImageNet 上的四种模型（DeiT‑Tiny、DeiT‑Small、Swin‑Tiny、Swin‑Small）。

**📈 对比分析**

与现有的 10+ 种无数据量化方法（GDFQ、ARC、Qimera、AdaSG、AdaDFQ、HAST、TexQ、PLF、PSAQ‑ViT 等）在 3/4 位宽量化下进行对比，取得最高准确率，提升幅度最高可达 1.74% p（CNN）和 0.58% p（ViT），显著优于先前最佳方法。

**⚠️ 局限性**

局限性：仍依赖合成数据的生成与调参，对超参数（阈值、滤波频率等）敏感；目前仅在分类任务上验证，尚未扩展到目标检测、分割等更复杂任务。

---

## 337. RE-SAC: Disentangling aleatoric and epistemic risks in bus fleet control: A stable and robust ensemble DRL approach

**arXiv ID:** 2603.18396 | [PDF](https://arxiv.org/pdf/2603.18396v1)

**作者:** Yifan Zhang `[一作]` (Central South University), Liang Zheng `[通讯]` (Central South University)

**通讯引用:** 38117 | [OpenAlex ID](https://openalex.org/A5100709340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出RE‑SAC框架，解决公交接驳控制中的Q值污染问题。

**💡 创新点**

通过将偏差风险拆解为偶然性与认知性不确定性并分别使用IPM权重正则与Q集成惩罚，实现稳健的价值估计。

**🔧 技术方法**

使用Soft Actor‑Critic改造、Integral Probability Metric正则、分布式Q集成、类别嵌入以及Mahalanobis稀疏度评估等技术。

**📊 数据集**

在高保真双向公交走廊仿真环境（22站、13小时、泊松乘客到达、Gaussian行驶速度）上训练。

**📈 对比分析**

与SAC、DSAC、BAC以及各消融模型对比，RE‑SAC在累计奖励约‑0.4×10⁶，优于其他基线，并在稀有状态下MAE显著下降。

**⚠️ 局限性**

局限在于单线单车模式、超大集成会导致过度悲观，且超参数λ_ale/λ_epi的手工设定。

---

## 338. Parallelograms Strike Back: LLMs Generate Better Analogies than People

**arXiv ID:** 2603.19066 | [PDF](https://arxiv.org/pdf/2603.19066v1)

**作者:** Qiawen Ella Liu `[一作]` (Princeton University), Thomas L. Griffiths `[通讯]` (Princeton University)

**通讯引用:** 49744 | [OpenAlex ID](https://openalex.org/A5077079119)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型（LLM）在四项词类比任务中的表现，并与人类生成的类比进行系统比较。

**💡 创新点**

创新点在于将LLM的类比输出与传统平行四边形模型和局部相似度启发式进行对照，揭示LLM在满足平行四边形约束和使用低词频词语方面优于人类的机制。

**🔧 技术方法**

采用了GloVe词向量进行平行四边形对齐度、C:D相似度与最近邻（NN）启发式的评估，使用六款主流LLM（两款闭源两款开源）生成响应，进行人类关系相似度打分，并通过多重回归分析探究评分差异的预测因素。

**📊 数据集**

使用SemEval‑2012 Task 2的四项词类比题目（共846个），并采集了约200,000条LLM回应和26,000条人类回应，覆盖10个语义关系类别及其子类型。

**📈 对比分析**

比较方法包括人类评分差异、累计检索率（CPR）评估模型预测力，以及回归预测LLM与人类评分差异。LLM在大多数关系类型上获得显著更高的评分（平均提升≈0.15–0.18分），并在平行四边形对齐度上优于人类；但仅比较频率最高的响应时优势消失。

**⚠️ 局限性**

局限在于仅以GloVe词向量评估平行四边形约束，未深入分析LLM内部表示；实验仅针对英文词类比，未涵盖多语言或更复杂类比形式；且LLM长尾低质量响应的产生机制仍未完全解释。

---

## 339. SR-Nav: Spatial Relationships Matter for Zero-shot Object Goal Navigation

**arXiv ID:** 2603.18443 | [PDF](https://arxiv.org/pdf/2603.18443v1)

**作者:** Leyuan Fang `[一作]` (Hunan University), Yinlong Yan `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 SR-Nav 框架，利用动态空间关系图（DSRG）实现目标感知增强和规划引导，从而在零样本目标导航中提升性能。

**💡 创新点**

创新点在于：①构建目标中心的动态空间关系图，将经验性空间先验与实时观测融合；②设计关系感知匹配模块（RAMM），通过空间关系匹配校正检测误差；③设计关系驱动规划模块（DRPM），利用 DSRG 指导前沿探索，缩小搜索空间。

**🔧 技术方法**

技术手段包括：大语言模型（LLM）生成先验关系、视觉语言模型（VLM）进行关系推理与规划、开源目标检测（YOLOv7、Grounding DINO）、3D 语义分割（SAM）以及动态地图更新与前沿搜索。

**📊 数据集**

使用 HM3D（约 2000 试验）和 MP3D（约 2195 试验）两个室内导航基准数据集进行评估。

**📈 对比分析**

与现有零样本方法（如 SG-Nav、VLFM、Uni-Nav 等）对比，SR-Nav 在 HM3D 上的成功率（SR）提升至 58.3%（比最优 53.9% 提升 4.4%），路径效率（SPL）提升至 33.0%（比 24.8% 提升 8.2%）；在 MP3D 上，SR 为 37.7%（比 34.7% 提升 3%），SPL 为 17.1%（比 16.0% 提升 1.1%）。

**⚠️ 局限性**

局限性包括：①对 VLM 计算资源依赖强，实时部署受限；②空间关系图的生成与更新主要基于室内场景，对室外或动态环境的适用性尚未验证；③在极端遮挡或多目标情况下，关系匹配的鲁棒性仍需提升。

---

## 340. Epistemic Generative Adversarial Networks

**arXiv ID:** 2603.18348 | [PDF](https://arxiv.org/pdf/2603.18348v1)

**作者:** Muhammad Mubashar `[一作]` (Oxford Brookes University), Fabio Cuzzolin `[通讯]` (Oxford Brookes University)

**通讯引用:** 2662 | [OpenAlex ID](https://openalex.org/A5050777136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于Dempster–Shafer证据理论的Epistemic GAN，改进判别器输出信念函数并让生成器生成像素级别的质量（Dirichlet分布）预测，从而提升生成多样性与不确定性解释。

**💡 创新点**

①将GAN损失通用化为信念函数形式；②生成器添加像素级Dirichlet分布输出，实现局部不确定性预测；③将不确定性量化融入训练目标，提升多样性与可解释性。

**🔧 技术方法**

使用Dempster–Shafer理论、Dirichlet分布、belief-based loss、DCGAN骨干网络、区域级质量预测、方差正则化与区间宽度正则化、FID和Vendi Score评估指标。

**📊 数据集**

CelebA、CIFAR-10、Food-101三个公开图像数据集。

**📈 对比分析**

与标准DCGAN在相同训练配置下对比，采用FID和Vendi Score评估。结果显示Epistemic GAN在所有数据集上FID更低、Vendi Score更高，性能提升约5%以上。

**⚠️ 局限性**

仍存在模型在高分辨率或更复杂域的适用性不足；未支持条件生成；额外正则项虽轻微但增加训练复杂度；仅在有限数据集验证，缺乏大规模多模态评估。

---

## 341. FlowMS: Flow Matching for De Novo Structure Elucidation from Mass Spectra

**arXiv ID:** 2603.18397 | [PDF](https://arxiv.org/pdf/2603.18397v1)

**作者:** Jianan Nie `[一作]` (Virginia Tech), Peng Gao `[通讯]` (Virginia Tech)

**通讯引用:** 33661 | [OpenAlex ID](https://openalex.org/A5100348308)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 FlowMS，一种基于离散流匹配的谱条件下的 de novo 分子生成框架，用于从串联质谱预测分子结构；

**💡 创新点**

首次将离散流匹配应用于质谱结构解析，结合化学式约束和谱嵌入，实现高效的概率空间迭代细化；

**🔧 技术方法**

采用离散流匹配、连续时间马尔可夫链去噪、图 Transformer 解码器以及 MIST 公式 Transformer 预训练的谱编码器，使用交叉熵损失进行训练；

**📊 数据集**

在 NPLIB1 质谱-结构对数据集上评估，并使用 DSSTox、HMDB、COCONUT、MOSES 等公开数据库进行预训练；

**📈 对比分析**

与 Spec2Mol、MIST+MSNovelist、MADGEN、DiffMS、MS‑BART 等基线进行对比，FlowMS 在 6 项指标中 5 项达成新高：top‑1 准确率 9.15%（较 DiffMS 提升 9.7%），top‑10 准确率 12.05%（略低于 DiffMS 的 15.44%），同时在 MCES 与 Tanimoto 相似度方面均显著优于竞争对手；

**⚠️ 局限性**

仍存在 top‑10 准确率略低、对精确匹配的召回率有限、依赖高质量谱编码器以及在更大谱库上的可扩展性和采样多样性待进一步提升等局限。

---

## 342. Expert Personas Improve LLM Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with PRISM

**arXiv ID:** 2603.18507 | [PDF](https://arxiv.org/pdf/2603.18507v1)

**作者:** Zizhao Hu `[一作]` (University of Southern California), Jesse Thomason `[通讯]` (University of Southern California)

**通讯引用:** 2298 | [OpenAlex ID](https://openalex.org/A5108062941)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 LLM persona prompting 的任务依赖性，并提出 PRISM 系统在无需外部数据的情况下自我蒸馏专家 persona，并通过门控 LoRA 进行推理。

**💡 创新点**

系统化分析 persona 效果、发现其对对齐任务有利但对预训练知识检索有害，并设计了自检验的 PRISM，能在保持知识性能的同时提升对齐与安全。

**🔧 技术方法**

自生成查询、双向自评判、LoRA 蒸馏、二进制门控以及多任务评测框架。

**📊 数据集**

MT‑Bench、MMLU、HarmBench、JailbreakBench、PKU‑SafeRLHF 等公开基准及内部生成的查询。

**📈 对比分析**

与基线、随机 persona、专家 persona、SFT 等方案对比，PRISM 在 7–8B 模型上整体分数提升 1–3 分，安全拒绝率提升约 5–20%，知识准确度保持不变。

**⚠️ 局限性**

仅在 7–8B 模型上验证，门控结构导致部署复杂，难以与传统 LoRA 组合，且在 MoE 或已高度专业化模型上效果有限。

---

## 343. Towards Interpretable Framework for Neural Audio Codecs via Sparse Autoencoders: A Case Study on Accent Information

**arXiv ID:** 2603.18359 | [PDF](https://arxiv.org/pdf/2603.18359v1)

**作者:** Shih-Heng Wang `[一作]` (University of Southern California), Shrikanth Narayanan `[通讯]` (University of Southern California)

**通讯引用:** 30942 | [OpenAlex ID](https://openalex.org/A5010028928)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一套框架，通过稀疏自编码器（SAE）将神经音频编解码器（NAC）产生的稠密表示转化为稀疏可解释激活，并利用这些激活进行口音二分类以量化NAC的可解释性。

**💡 创新点**

创新点在于：①将SAE与NAC结合，首次以可解释性指标ΔF1衡量NAC；②分离激活的位置信息与幅度信息，揭示声学导向与语音导向NAC对口音信息的编码差异；③发现低比特率EnCodec在可解释性上优于高比特率版本。

**🔧 技术方法**

使用TopK稀疏自编码器、线性编码/解码、logistic回归分类器以及相对性能指标ΔF1。

**📊 数据集**

基于Vox-Profile口音分类数据集，构造US vs UK及US vs Non-US-UK两种二分类任务。

**📈 对比分析**

对四种主流NAC（EnCodec、DAC、SpeechTokenizer、Mimi）分别在16种SAE配置（不同潜在维度比例q和稀疏率s）下评估。结果显示DAC和SpeechTokenizer在ΔF1上最高，DAC在US vs UK任务中稳居第一，SpeechTokenizer在US vs Non-US-UK任务中领先；在位置信息和幅度信息上，声学导向NAC偏向幅度，语音导向NAC偏向位置；比特率越低的EnCodec可解释性越好。

**⚠️ 局限性**

局限性包括：仅评估口音这一单一二元任务，未涉及多元或更细粒度的说话人属性；仅使用Vox-Profile数据集，可能无法覆盖所有口音变化；SAE仅采用线性结构，可能限制对更复杂表示的捕获；对其他NAC模型及更多语音任务的推广仍待验证。

---

## 344. DreamPartGen: Semantically Grounded Part-Level 3D Generation via Collaborative Latent Denoising

**arXiv ID:** 2603.19216 | [PDF](https://arxiv.org/pdf/2603.19216v1)

**作者:** Tianjiao Yu `[一作]` (University of Illinois Urbana - Champaign), Ismini Lourentzou `[通讯]` (University of Illinois Urbana - Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种语言驱动的协同扩散框架，用Duplex Part Latents（DPLs）和Relational Semantic Latents（RSLs）实现了可解释、可控的3D对象与场景生成。

**💡 创新点**

创新点包括：①持续存在的语言导向关系潜在变量（RSLs）与部件几何潜在变量（DPLs）同步共去噪；②引入可学习的部件身份嵌入，保证部件在扩散过程中的槽位不被打乱；③构建了新的PartRel3D关系数据集，提供数十万个功能与空间三元组；④将二维与三维潜在序列联合编码，兼顾形状与外观。

**🔧 技术方法**

使用技术包括：扩散模型与同步注意力机制、3D VAE和预训练图像VAE、双向编码（几何+外观）、双槽位同步（部件内同步 + 部件间同步）、SNR调度的训练策略、基于CLIP/ULIP的文本-形状对齐。

**📊 数据集**

数据集：PartRel3D（基于PartVerse扩展），包含约11K个标注对象、90K个部件以及300K个功能与空间关系三元组；另外使用了Objaverse、ShapeNet、ABO等公开基准进行评测。

**📈 对比分析**

与Trellis、CLAY、HoloPart、PartCrafter等最先进方法对比，模型在几何精度（CD↓53%，EMD↓33%）、文本-形状对齐（CLIP/ULIP提升≥20%）以及部件分离度（IoU下降27%）等指标上均显著优越。

**⚠️ 局限性**

局限性：①需要先提取或人工提供关系三元组，若语言解析错误会影响性能；②同步去噪和多模态潜在处理导致计算成本较高；③在高度复杂的动态/可动作场景中对连通性与可控编辑的支持仍不完善。

---

## 345. SignAgent: Agentic LLMs for Linguistically-Grounded Sign Language Annotation and Dataset Curation

**arXiv ID:** 2603.19059 | [PDF](https://arxiv.org/pdf/2603.19059v1)

**作者:** Oliver Cory `[一作]` (University of Surrey), Richard Bowden `[通讯]` (University of Surrey)

**通讯引用:** 13799 | [OpenAlex ID](https://openalex.org/A5044490167)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SignAgent 框架，利用 LLM 与多模态工具实现签语（SL）的视频注释与数据集整理；

**💡 创新点**

将 agentic 语言模型与语音/视觉分析工具、知识检索图谱结合，形成可审计的推理流程；在伪词注释和 ID Glossing 两项任务上实现了基于语言学的自动化标注；

**🔧 技术方法**

核心技术包括：ReAct 样式的 LLM Orchestrator、SignGraph Retrieval‑Augmented Generation、基于 3D 姿态的手形/运动/位置分类器、视觉分割与检索工具、Gradient‑Boosted Decision Tree、知识图谱检索、聚类与重排序；

**📊 数据集**

使用的公开数据集包括 BSL Corpus、BSLSignbank、ASL Citizen、ASL‑Lex2.0 以及 SignRep 基准模型；

**📈 对比分析**

与 Sign2GPT lemmatization 与 GBDT+fuzzy 基线相比，伪词注释的 LCS 提升至 60.85%（基线 57.26%），Kendall τ 提升至 0.374；在 ID Glossing 任务中，平均 ID/Gloss 降至 2.30（基线 4.81），簇熵降低、Silhouette 提升至 0.764，Calinski‑Harabasz 从 6.75 提升至 7.58，表明簇更紧凑且更符合语言学标签；

**⚠️ 局限性**

当前框架仍依赖已有词典与工具，非手势、语调等信息未充分利用；低资源手语的适用性有限；工具与控制器未实现联合优化；最终标注仍需人工专业验证，不能完全替代人类语言学家。

---

## 346. Detecting Basic Values in A Noisy Russian Social Media Text Data: A Multi-Stage Classification Framework

**arXiv ID:** 2603.18822 | [PDF](https://arxiv.org/pdf/2603.18822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 347. Balanced Thinking: Improving Chain of Thought Training in Vision Language Models

**arXiv ID:** 2603.18656 | [PDF](https://arxiv.org/pdf/2603.18656v1)

**作者:** Shaked Perek `[一作]` (IBM Research), Eli Schwartz `[通讯]` (IBM Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在监督微调阶段使用动态、长度不相关的加权损失（SCALe-SFT），通过在推理段和答案段之间分配时间演进的权重，鼓励模型生成简洁、结构化的推理链并提高最终答案的准确性。

**💡 创新点**

创新点在于：①识别并补偿推理文本中令长推理段主导损失的问题；②设计“Scheduled Curriculum Adaptive Loss”，在训练初期强调推理结构，后期转向答案准确性；③使用长度无关的平均交叉熵避免长推理段过度占优；④通过余弦调度实现平滑权重过渡。

**🔧 技术方法**

核心技术包括：长度独立的加权交叉熵损失、推理与答案段的分离、时间动态权重调度（余弦退火）、以及可选的GRPO强化学习阶段。

**📊 数据集**

主要使用 Vision‑R1 语料库（含 200K 推理样本）以及其抽取的 ScienceQA 和 IconQA 两个评测数据集；在实验中还使用了 Qwen2.5‑VL‑3B、LLaVA‑Next、Gemma‑3‑4B 三个视觉‑语言模型。

**📈 对比分析**

与传统统一权重的 SFT 以及 SFT+GRPO 进行对比。SCALe‑SFT 在所有模型上相较于 vanilla SFT 提升约 1.5%–3%；与 GRPO 结合时进一步提升 3%–5%；单独使用 SCALe‑SFT 甚至可匹配甚至超越完整的 SFT+GRPO，且训练时间仅为前者的约 1/7，证明其高效性。

**⚠️ 局限性**

局限性包括：目前仅针对单图推理，未验证多图或更复杂场景；虽然可以省略 GRPO，但最佳性能仍需强化学习阶段；对权重调度的具体参数仍需手动设定，缺乏自适应调节机制；在非视觉推理任务上的通用性仍待进一步验证。

---

## 348. FILT3R: Latent State Adaptive Kalman Filter for Streaming 3D Reconstruction

**arXiv ID:** 2603.18493 | [PDF](https://arxiv.org/pdf/2603.18493v1)

**作者:** Seonghyun Jin `[一作]` (KAIST AI), Jong Chul Ye `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种训练无关的自适应Kalman滤波层FILT3R，用于在流式3D重建中对持久的潜在状态进行在线更新，显著提升长期记忆稳定性并降低漂移。

**💡 创新点**

创新点在于将潜在状态更新视为噪声测量的Kalman滤波，利用自适应过程噪声（基于内部时间漂移）和固定测量噪声，实现了对记忆衰退与快速适应的动态权衡，且无需额外训练。

**🔧 技术方法**

核心技术包括：
1) 逐token方差传播与Kalman增益计算；
2) 通过EMA归一化的时间漂移估计过程噪声；
3) 简化的测量噪声为单标量，减少计算与内存开销；
4) 与现有的CUT3R/TTT3R等更新策略兼容，可直接替换。

**📊 数据集**

在多种数据集上进行评估：
- 视频深度：Sintel、Bonn、KITTI；
- 相机姿态：TUM‑RGBD；
- 3D重建：7‑Scenes、NRGBD；
- 长期序列测试：扩展的TUM、Bonn序列（超过训练 horizon）。

**📈 对比分析**

与CUT3R、TTT3R、Point3R、全注意力模型等基线对比，FILT3R在所有任务中均显示出更低的累计漂移、提升的准确率与一致性，尤其在长序列（>800帧）下保持几乎不变的误差，并在GPU内存与时延上与CUT3R相当。

**⚠️ 局限性**

局限性包括：
1) 仍假设测量噪声可视为固定标量，可能在极端动态或高噪声场景下不足；
2) 依赖内部漂移估计，若漂移信号被严重污染可能导致过程噪声估计失真；
3) 未探索与深度学习训练联合优化的潜力，可能在某些任务上无法达到专门训练的最佳性能。

---

## 349. VC-Soup: Value-Consistency Guided Multi-Value Alignment for Large Language Models

**arXiv ID:** 2603.18113 | [PDF](https://arxiv.org/pdf/2603.18113v1)

**作者:** Hefei Xu `[一作]` (Hefei University of Technology), Meng Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 42512 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 VC‑soup 框架，通过价值一致性过滤偏好数据并在 DPO 微调后线性融合参数，实现在多价值对齐中的无冲突模型融合。

**💡 创新点**

创新点：① 引入价值一致性度量（cosine 与 all‑ones 向量）对偏好样本进行过滤；② 通过过滤得到的子集训练得到的 VC 向量在参数空间更接近、方向一致，从而大幅降低参数干扰；③ 结合 Pareto 过滤构建多价值 Pareto 前沿。

**🔧 技术方法**

技术：DPO 微调 + LoRA、价值一致性度量与阈值过滤、线性模型融合、Pareto 前沿筛选、奖励模型评分与 GPT‑4 判定。

**📊 数据集**

数据集：Anthropic‑HH（helpful & harmless）、BeaverTails（helpful & safety）、以及第三方诚实性偏好数据集用于三维实验。

**📈 对比分析**

与 DPO‑Help/Harm/Safe、SeqT、DPO‑LW、SOUP、MODPO、MVA 等基线对比；在奖励模型评分和 GPT‑4 winrate 上均实现了更靠近 upper‑right 的 Pareto 曲线，赢率显著高于所有基线。

**⚠️ 局限性**

局限：① 需要先训练奖励模型；② 价值一致性阈值选择较为经验化；③ 对极端冲突样本过滤导致有效训练数据量下降；④ 目前仅验证两/三维价值，尚未测试更高维度的多价值场景。

---

## 350. Complexity of Auctions with Interdependence

**arXiv ID:** 2603.18668 | [PDF](https://arxiv.org/pdf/2603.18668v1)

**作者:** Patrick Loiseau `[一作]` (INRIA), Minrui Xu `[通讯]` (ENSAE)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在Milgrom和Weber提出的相互依赖模型下的拍卖设计，特别关注于采购拍卖中的机制设计，旨在最小化执行任务的代理成本。

**💡 创新点**

创新点在于去除了对价值函数的单调性假设，并研究了在确定性和随机设置下优化真实机制的近似比率，提供了理论解释和高效算法，同时也给出了普遍情况下的难度结果。

**🔧 技术方法**

使用了理论计算机科学中的经典组合问题和线性规划技术，提出了几种框架来优化真实机制的近似比率。

**📊 数据集**

没有具体提到使用的数据集，但研究涉及的模型和算法可以应用于多种拍卖和采购场景。

**📈 对比分析**

与以往文献中的方法相比，提出的算法在特定情况下能够在多项式时间内解决问题，并且在n=2或k=2的情况下，能够在O(N log N)时间内解决特定问题。

**⚠️ 局限性**

限制在于对于一般情况的复杂性，尤其是当n和k较大时，问题的计算复杂性仍然是NP-Hard，且在某些情况下无法在多项式时间内回答查询。

---

## 351. ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs

**arXiv ID:** 2603.18579 | [PDF](https://arxiv.org/pdf/2603.18579v1)

**作者:** Abhinaba Basu `[一作]` (Indian Institute of Information Technology), Pavan Chakraborty `[通讯]` (Indian Institute of Information Technology)

**通讯引用:** 1509 | [OpenAlex ID](https://openalex.org/A5023091561)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ICE 框架，使用随机化检验对多种干预算子进行可解释性评估，判定解释是否真实反映模型推理过程。

**💡 创新点**

创新点在于将干预算子视为可比较维度，结合随机基线提供置信区间，揭示了 operator 依赖性、反忠实性现象以及信度与可解释性之间的独立性。

**🔧 技术方法**

采用随机化检验、NSR（Normalized Score Retention）指标、删除与检索填充两种干预算子、Attention 与 Gradient 两种归因方法，并对多语言输入进行 tokenization 处理。

**📊 数据集**

评测覆盖 7 只 LLM（1.5B–8B 参数）在 4 项英语任务（SST‑2、IMDB、e‑SNLI、AG News）以及 6 种非英语语言（法语、德语、印地语、中文、土耳其语、阿拉伯语）上的数据集。

**📈 对比分析**

通过 win‑rate、效应大小和 95% 置信区间与现有 ERASER、F‑Fidelity 等基准对比，发现干预算子差距可达 44pp，随机基线揭示近 1/3 配置为反忠实；在多语言和多模型场景下，Attention 通常优于 Gradient，但两者在长文本上趋同。

**⚠️ 局限性**

主要限制包括高计算成本（M=50 次随机化）、归因方法范围有限（未覆盖 IG）、多语言覆盖不足、仅评估行为忠实而非机制忠实，以及检索填充可能仍保留任务相关信号。

---

## 352. Automatic Configuration of LLM Post-Training Pipelines

**arXiv ID:** 2603.18773 | [PDF](https://arxiv.org/pdf/2603.18773v1)

**作者:** Channe Chwa `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**通讯引用:** 6003 | [OpenAlex ID](https://openalex.org/A5058605138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在有限计算预算下，通过结合离线经验和在线适配，设计并实现了一种两阶段的配置选择框架 AutoPipe，用于自动调优大型语言模型的 SFT–RL 后训练流水线。

**💡 创新点**

创新点包括：① 用数据集条件化的排序代理学习跨数据集的配置优先级；② 在线阶段仅建模相对纠正项，利用贝叶斯优化和高斯过程残差模型快速校正；③ 通过早停预测器将低成本的 SFT 轨迹映射到最终性能的代理，显著降低在线评估成本。

**🔧 技术方法**

主要技术手段包括梯度提升树学习排序代理、基于高斯过程的贝叶斯优化、残差建模、以及 XGBoost 早停预测器。整个流程结合离线历史跑、在线贝叶斯搜索和多精度评估。

**📊 数据集**

在医学推理领域，使用了 MQA、MQAR、PQA、MQA(R)、MQA(R)PQA 等自研数据集，以及公开的 HuaTuo、MedReason、MedS3、ReasonMed、m23k 等；评估指标基于 BioASQ、GPQA、MedBullets、MedMCQA、MedQA、MedXpertQA、MMLU 等下游基准。

**📈 对比分析**

与静态零启动方法（随机、LLM 推荐、全局）以及在线搜索基线（Random Search、BOHB、SMAC、TPE）进行比较。AutoPipe 在只使用 10% 预算的情况下，最终基准得分与最强在线基线持平，且在多项指标上优于所有静态方法；早停版本的 AutoPipe 甚至在低预算下超过了大多数完整评估的搜索方法。

**⚠️ 局限性**

主要局限性是对早停代理信号的依赖；当 SFT 轨迹噪声大或与后续 RL 目标关联弱时，代理预测可能不稳定；此外，在面临显著数据集漂移时，离线经验的迁移效果可能受限。

---

## 353. Evaluating FrameNet-Based Semantic Modeling for Gender-Based Violence Detection in Clinical Records

**arXiv ID:** 2603.18124 | [PDF](https://arxiv.org/pdf/2603.18124v1)

**作者:** Lívia Dutra `[一作]` (Federal University of Juiz de Fora), Tiago Torrent `[通讯]` (Federal University of Juiz de Fora)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

探究利用FrameNet语义标注的开放文本字段在电子病历中识别性别暴力（GBV）案例的可行性

**💡 创新点**

首次将FrameNet框架与医疗与暴力领域的语义模型相结合，证明语义层面的信息可显著提升GBV识别

**🔧 技术方法**

使用FrameNet Brasil语义标注工具、LMO自动标注器、XLM-RoBERTa+BIO+Typer管道以及线性SVM分类器

**📊 数据集**

基于巴西公共卫生系统SINAN、e-SUS AB以及ICD、SIM死亡信息系统收集的约三百万条电子病历与13,000条暴力通报记录

**📈 对比分析**

通过三种实验设置（仅语义、语义+参数化、仅参数化）进行五折交叉验证，语义模型F1≈0.772，显著高于参数化模型F1≈0.461，证明语义标注提升约31个百分点

**⚠️ 局限性**

模型依赖于已标注语料，可能存在标注偏差；框架与语言特定性导致结果对非葡语语料的迁移性有限；并且在解释性和临床实用性方面仍需进一步验证

---

## 354. Click-to-Ask: An AI Live Streaming Assistant with Offline Copywriting and Online Interactive QA

**arXiv ID:** 2603.18649 | [PDF](https://arxiv.org/pdf/2603.18649v1)

**作者:** Ruizhi Yu `[一作]` (East China normal University), Haonan Lu `[通讯]` (OPPO AI Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个结合离线写作和在线交互问答的直播电商AI助手Click-to-Ask。

**💡 创新点**

通过离线多模态信息集成生成结构化产品数据并自动生成合规文案，结合在线点击式问答与事件级历史记忆，实现实时精准回应。

**🔧 技术方法**

大语言模型、视觉语言模型、GRPO强化学习、视觉提示、流式事件分割与知识提取加速器。

**📊 数据集**

TikTok直播帧真实数据集 + 合成CLEVR文本 + 自制8K图像QA数据集。

**📈 对比分析**

与Qwen2.5‑VL 7B基线对比，点击式问答QRA 0.913、RQ 0.876；使用GRPO+视觉提示提升显著。

**⚠️ 局限性**

对长直播中信息冗余处理仍有限，模型对坐标文本定位敏感，需要进一步优化。

---

## 355. Frayed RoPE and Long Inputs: A Geometric Perspective

**arXiv ID:** 2603.18017 | [PDF](https://arxiv.org/pdf/2603.18017v1)

**作者:** Davis Wertheimer `[一作]` (IBM Research), Naigang Wang `[通讯]` (IBM Research)

**通讯引用:** 3637 | [OpenAlex ID](https://openalex.org/A5082043392)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 RoPE 在 Transformer 中的几何行为进行分析，发现关键/查询向量聚成对立簇并依赖 sink token，随后提出 RoPE‑In‑Distribution（RoPE‑ID）方案，在每个注意力头中仅对一半通道使用高频 RoPE 并进行温度缩放，以实现长上下文的无缝泛化。

**💡 创新点**

首次统一了 RoPE、注意力簇分离与 sink token 的几何关系，提出了两个长度泛化标准，并基于此设计了 RoPE‑ID 的通道与频率调制策略。

**🔧 技术方法**

采用几何分析（聚类、PCA、奇异值分解）、RoPE 高频通道子集应用、温度缩放，以及标准 Transformer 训练和推理技术；在 LongBench、RULER 及常识推理基准上进行评估。

**📊 数据集**

使用 Llama3、Olmo、Gemma 预训练模型；训练数据来自 Dolma v1.7；评估数据集包括 RULER（针尖找针、词数统计等合成任务）、LongBench（单文档、多文档问答、少量学习、代码补全、摘要）以及 ARC‑C、HellaSwag、PIQA 等常识推理任务。

**📈 对比分析**

与标准 RoPE、High‑Freq RoPE、HalfRoPE 以及 YaRN 的推理‑时扩展方法对比，RoPE‑ID 在 1B 模型上在 8k/16k 长度下与 YaRN 相当或略优，3B 模型与 YaRN 相近；在 4k 训练长度下几乎不损失性能，且在长文本检索任务上表现显著提升。

**⚠️ 局限性**

RoPE‑ID 仍受限于“训练长度 × 4”范围的泛化，且仅满足两个必要条件，未必能覆盖所有长上下文失效场景；对更长序列可能出现漂移；方法对频率与通道比例的设定敏感，需要进一步与其它技术结合提升鲁棒性。

---

## 356. Implicit Patterns in LLM-Based Binary Analysis

**arXiv ID:** 2603.19138 | [PDF](https://arxiv.org/pdf/2603.19138v1)

**作者:** Qiang Li `[一作]` (Beijing Jiaotong University), Haining Wang `[通讯]` (Virginia Tech)

**通讯引用:** 9637 | [OpenAlex ID](https://openalex.org/A5100664241)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对 LLM 驱动的二进制漏洞分析进行长时序轨迹级研究，揭示其内部的探索决策机制

**💡 创新点**

首次发现并量化了四种隐式 token 级模式（早期裁剪、路径锁定、定向回溯、知识驱动优先），并证明其在多数会话中稳定出现

**🔧 技术方法**

利用大语言模型代理与分析工具的交互记录、模式抽取规则、统计与可视化分析技术

**📊 数据集**

521 篇 ARM/MIPS 体系结构的固件二进制文件（Karonte 数据集筛选）共 99,563 步长的推理轨迹

**📈 对比分析**

通过模式出现率、密度、时间分布、转移图、行为指标等多维度对比，显示四种模式各具特征且相互协作，未直接给出漏洞发现率但揭示了有效的探索结构

**⚠️ 局限性**

受任务、系统设计、LLM 版本限制；仅针对漏洞检测情境；缺乏对内部模型状态的解释，且模式可能随提示或工具不同而变化

---

## 357. Enactor: From Traffic Simulators to Surrogate World Models

**arXiv ID:** 2603.18266 | [PDF](https://arxiv.org/pdf/2603.18266v1)

**作者:** Yash Ranjan `[一作]` (University of Florida), Sanjay Ranka `[通讯]` (University of Florida)

**通讯引用:** 10836 | [OpenAlex ID](https://openalex.org/A5077570468)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于Transformer的参与者中心生成模型，用以在交通信号交叉口中生成物理一致且符合实际的车辆轨迹。

**💡 创新点**

创新点在于将极坐标表示与空间-时间注意力结构相结合，采用闭环训练的world‑model范式，并显式引入车尾信息以提升停止行为的准确性。

**🔧 技术方法**

使用Transformer（空间注意力+时间方向注意力）、极坐标特征、Gaussian分布输出、负对数似然训练及位置编码等技术。

**📊 数据集**

数据集为SUMO仿真生成的两条信号交叉口的车辆轨迹，共计约124万条样本，时间步长0.1 s，仿真时长6 小时。

**📈 对比分析**

与IntTrajSim基线及Enactor比较，模型在速度分布、行程时间等聚合指标上显著优于基线（KL散度低于1/10），但在碰撞风险和红灯违规等交互级指标上略逊，后续通过加入车尾信息后已显著提升。

**⚠️ 局限性**

局限在于对近距离相互作用与减速行为建模不足，导致时间-碰撞（TTC）指标不理想；模型对不同几何配置的泛化仍需改进。

---

## 358. When Differential Privacy Meets Wireless Federated Learning: An Improved Analysis for Privacy and Convergence

**arXiv ID:** 2603.19040 | [PDF](https://arxiv.org/pdf/2603.19040v1)

**作者:** Chen Yaoling `[一作]`, Tu Xiaotong `[通讯]` (Xiamen University)

**通讯引用:** 1835 | [OpenAlex ID](https://openalex.org/A5068316504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文在无线联邦学习框架下，对不同ially private（DP）机制进行完整的隐私与收敛性分析，并在非凸光滑损失函数上给出了收敛上界和隐私-效用权衡。

**💡 创新点**

创新点包括：①将设备选择与数据批采样纳入RDP分析，证明隐私损失随迭代次数收敛到常数；②首次在DPWFL中系统性考虑梯度裁剪对收敛的影响；③给出显式的隐私-效用权衡公式，并验证了相较传统DP-NOMA在隐私曲线上的显著改进。

**🔧 技术方法**

使用技术包括：无线信道噪声建模、RDP与Rényi divergence、隐私放大（sampling amplification）、梯度裁剪与随机梯度下降（SGD）理论、L‑光滑、梯度方差与不一致性假设、以及对无线噪声的估计误差分析。

**📊 数据集**

实验采用了规模为8个样本的公开或合成数据集，设置设备采样率p=1、批采样率q=1、裁剪阈值c=2、参数域直径D=0.5等超参数；未使用大型真实数据集。

**📈 对比分析**

与DP‑NOMA基线比较时，DPWFL在相同隐私预算下实现了更低的ε值，尤其在较小的域直径、设备/批采样率较低时表现更佳；实验结果与理论推导高度一致，验证了隐私损失收敛与效能提升。

**⚠️ 局限性**

局限性包括：①假设参数域有界、梯度方差受限，实际深度网络可能不满足；②对数据分布的IID假设，非IID场景下的性能尚未评估；③分析基于理想化的信道模型，真实无线噪声与多径效应的影响未完整覆盖；④未给出大规模真实数据集上的验证。

---

## 359. Teleological Inference in Structural Causal Models via Intentional Interventions

**arXiv ID:** 2603.18968 | [PDF](https://arxiv.org/pdf/2603.18968v1)

**作者:** Dario Compagno `[一作]`, Fabio Massimo Zennaro `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的结构最终模型（SFM）框架，用意图干预来扩展结构因果模型（SCM），实现对智能体干预的建模。

**💡 创新点**

创新点在于把意图视为干预与对抗模型之间的关系，保持无时间维度、无环、可解释的因果图，从而实现对代理人检测和意图发现的可行性。

**🔧 技术方法**

使用了结构因果模型、双生图、对抗（counterfactual）推理以及意图干预算子，并给出了可识别性定理与模拟实验。

**📊 数据集**

实验数据主要基于人工模拟的热力学与吸烟两类简单因果图的离散数据。

**📈 对比分析**

通过比较意图干预前后模型的马尔可夫性违规和干预效果来验证算法，在模拟实验中成功检测到代理人并识别其目标，但算法完整性与可扩展性仍待进一步评估。

**⚠️ 局限性**

局限性包括仅支持单变量干预、假设代理人拥有完全因果知识与高频率行动、仅适用于马尔可夫模型、未证明算法的完整性和大规模可扩展性。

---

## 360. Agentic Framework for Political Biography Extraction

**arXiv ID:** 2603.18010 | [PDF](https://arxiv.org/pdf/2603.18010v1)

**作者:** Yifei Zhu `[一作]` (University of Hong Kong), Junyan Jiang `[通讯]` (Columbia University)

**通讯引用:** 1535 | [OpenAlex ID](https://openalex.org/A5026353233)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个两阶段“合成‑编码”框架，利用大型语言模型自动提取并结构化跨国政治精英的生涯数据。

**💡 创新点**

首次将 agentic 递归检索与 LLM 编码结合，证明在缺乏精细化百科条目时，自动合成能够显著提升覆盖率，并展示合成阶段对长文本提取的必要性。

**🔧 技术方法**

采用大型语言模型（Grok‑4.1‑Fast、Gemini‑2.5‑Flash、Qwen‑2.5 等）、递归检索工具、Agentic 推理循环、自动评判与验证。

**📊 数据集**

使用中国 CPED、美国与 OECD 的政治精英名单与维基百科/政府网站等公开网页，构建合成语料与 Consolidated Ground Truth。

**📈 对比分析**

通过与人工编码基准对照，LLM 在已整理文本上可达到或超过人类精度（F1 提升 10–16pp）；在开放网页上，agentic 合成将 F1 从约 0.76–0.77 提升至 0.87–0.94，成本仅为人工的 3%。

**⚠️ 局限性**

局限包括对低质量或多语言来源的依赖导致的精度下降、对长文本仍存在“中间遗失”现象，以及缺乏对解释性属性的验证与伦理透明度的挑战。

---

## 361. Bridging Network Fragmentation: A Semantic-Augmented DRL Framework for UAV-aided VANETs

**arXiv ID:** 2603.18871 | [PDF](https://arxiv.org/pdf/2603.18871v1)

**作者:** Gaoxiang Cao `[一作]` (University of Science and Technology of China), Jian Yang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 43740 | [OpenAlex ID](https://openalex.org/A5100726984)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种结合大型语言模型（LLM）推理与深度强化学习（DRL）的框架SA‑DRL，用于无人机在城市VANET中智能部署，解决网络碎片化问题。

**💡 创新点**

创新点在于：①将道路拓扑图和双连通图的图论方法用于量化网络碎片化；②设计四阶段流水线将通用LLM转化为领域专精的拓扑专家；③提出Logit Fusion机制，将LLM的语义先验直接注入PPO策略，兼顾全局语义与局部探索；④使用低秩适配器（LoRA）实现高效参数化微调。

**🔧 技术方法**

技术包括：图神经网络（GAT）对比、Proximal Policy Optimization（PPO）、Soft Actor-Critic（SAC）、LoRA微调、vLLM并行推理、双流推理与Logit Fusion、KL正则化。

**📊 数据集**

使用从中国深圳市大数据收集的真实城市道路拓扑与车辆轨迹数据（约5M条记录），随后抽取子集（47节点、88条边、5K轨迹）用于高保真仿真。

**📈 对比分析**

与SAC、Vanilla PPO、GAT‑PPO进行对比，SA‑PPO在收敛速度、连通性（平均连通组件车辆数提升13.2%/23.5%）和能耗（仅28.2%）方面均显著优于基线，且仅使用26.6%训练回合即可达到最优。

**⚠️ 局限性**

局限包括：对单无人机场景的验证，LLM推理仍带来显著延迟，且对不同城市或更复杂交通情景的泛化能力需要进一步评估。

---

## 362. Complementary Text-Guided Attention for Zero-Shot Adversarial Robustness

**arXiv ID:** 2603.18598 | [PDF](https://arxiv.org/pdf/2603.18598v1)

**作者:** Lu Yu `[一作]` (Tianjin University of Technology), Changsheng Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 25921 | [OpenAlex ID](https://openalex.org/A5022636178)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了两种基于文本引导注意力的零样本对抗鲁棒性框架——TGA-ZSR 与其改进版 Comp-TGA，旨在提升 CLIP 在对抗攻击下的零样本性能并保持对干净样本的泛化。

**💡 创新点**

创新点包括：①发现对抗扰动会导致文本引导注意力出现偏移；②设计局部注意力细化模块（LARM）和全局注意力约束模块（GACM）以对齐和约束注意力；③提出互补文本引导注意力（Comp-TGA），通过结合类别提示与非类别提示的前景注意力，提升注意力准确性与鲁棒性。

**🔧 技术方法**

技术手段：使用预训练 CLIP 作为基模型；对抗微调（PGD、CW、AutoAttack）；文本引导注意力计算（图像特征与文本嵌入相乘）；LARM 与 GACM 损失约束；互补注意力融合；实验采用 CE+L_ARM+L_GACM 总损失。

**📊 数据集**

数据集：在 Tiny‑ImageNet 上进行对抗微调，随后在 16 个零样本基准上评估：CIFAR‑10/100、STL‑10、SUN397、Food101、OxfordPets、Flowers102、DTD、EuroSAT、FGVC‑Aircraft、ImageNet、Caltech‑101/256、StanfordCars、PCAM。

**📈 对比分析**

方法比较：与 CLIP、FT‑Clean、FT‑Adv、TeCoA、FARE、LAAT、PMG‑AFT 及无训练的 TTC 进行对比。TGA‑ZSR 在 A_Robust 上提升 9.58%，Comp‑TGA 提升 11.95%（相较于现有最优），并保持或略低于对比方法的 A_Clean；在 PGD、CW、AutoAttack 等多种攻击下均取得最高平均鲁棒性；训练内存与 PMG‑AFT 相近，训练时间更短，推理时间与其他方法相当。

**⚠️ 局限性**

局限性：文本引导注意力有时会聚焦无关特征，对极强攻击（如 AutoAttack）提升有限；自适应攻击评估仍需进一步细化；双分支结构导致计算开销略增，尤其在大规模部署时需考虑资源限制。

---

## 363. CaseLinker: An Open-Source System for Cross-Case Analysis of Internet Crimes Against Children Reports -- Technical Report & Initial Release

**arXiv ID:** 2603.18020 | [PDF](https://arxiv.org/pdf/2603.18020v1)

**作者:** Mrinaal Ramachandran `[一作]` `[通讯]` (University of Massachusetts), Mrinaal Ramachandran (University of Massachusetts)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了CaseLinker系统，能将分散、碎片化的儿童性剥削与虐待（CSEA）案件报告自动化转换为结构化数据，完成两阶段聚类、优先级分级、可视化与洞察生成；

**💡 创新点**

创新点在于：①使用可解释的确定性正则与模式匹配完成信息提取，保证法律可审计；②设计两阶段聚类（外部主题聚类+内部加权Jaccard相似度），兼顾可解释性与细粒度分组；③强调分析者心理健康的可视化设计与分级决策，降低情绪负担；

**🔧 技术方法**

技术包括：Python正则与模式匹配提取，基于SQLite的轻量级存储，权重Jaccard相似度聚类，优先级打分公式，D3.js交互可视化，模块化五层架构；

**📊 数据集**

使用的数据集为2011-2014年亚利桑那州ICAC公开报告中的47个案例（共47,141字符）；

**📈 对比分析**

比较方法主要是评估特征覆盖率、聚类一致性、优先级分数与人工检查对比；性能方面，端到端吞吐约23.2案/秒，聚类耗时11.03 ms，提取耗时1.2 ms/案；

**⚠️ 局限性**

局限性包括：样本量小、仅单一州报告，正则匹配对语言变体覆盖有限，部分字段（受害者数量、平台）提取率低，缺乏真实用户研究验证其对分析者心理健康的实效。

---

## 364. T-QPM: Enabling Temporal Out-Of-Distribution Detection and Domain Generalization for Vision-Language Models in Open-World

**arXiv ID:** 2603.18481 | [PDF](https://arxiv.org/pdf/2603.18481v1)

**作者:** Aditi Naiknaware `[一作]` (San Diego State University), Salimeh Sekeh `[通讯]` (San Diego State University)

**通讯引用:** 49 | [OpenAlex ID](https://openalex.org/A5103219871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种面向开放世界视觉‑语言模型的时序四元匹配（T‑QPM）框架，用于在持续漂移的数据分布中实现跨模态（图像‑文本）OOD 检测与域泛化。

**💡 创新点**

创新点在于：①将双模式匹配扩展为四模式（图像‑文本、图像‑视觉、文本‑视觉、文本‑文本）联合评分；②通过时间‑感知的视觉原型与轻量级融合权重处理分布漂移；③利用 ATC 正则化保证跨时刻阈值稳定性；④在单一模型中同时兼顾协变量漂移与语义 OOD。

**🔧 技术方法**

核心技术包括：冻结 CLIP ViT‑B/16/32 双编码器；多模板文本增强与归一化；基于 KL 与交叉熵的多模态分数计算；Softplus 参数化的可学习融合权重；ATC 与协变量一致性正则化；以及基于阈值的最终决策。

**📊 数据集**

使用的公开数据集有：ID 方面的 CLEAR100、CLEAR10、Core50；OOV 方面的 COCO、ImageNet‑1K‑VL‑Enriched、Visual Genome、Flickr30K、CC12M；此外通过高斯模糊与 JPEG 压缩生成协变量扰动版本。

**📈 对比分析**

与基准方法 DPM、SCONE、Temp‑SCONE 等进行对比，T‑QPM 在 CLEAR、Core50 以及多种 OOD 数据集上均以显著更低的 FPR95 与更高的 AUROC 领先，尤其在时间步晚期和高频扰动下表现尤为突出，且 ID 分类准确率也保持稳定或提升。

**⚠️ 局限性**

局限性包括：①仅在 CLIP 架构上验证，未探讨更大规模或其他 VLM 的迁移；②仍依赖预先设定的阈值和超参，对极端漂移场景的鲁棒性待进一步验证；③未在真实工业流水线中测试实时性能与资源开销。

---

## 365. A Family of Adaptive Activation Functions for Mitigating Failure Modes in Physics-Informed Neural Networks

**arXiv ID:** 2603.18328 | [PDF](https://arxiv.org/pdf/2603.18328v1)

**作者:** Krishna Murari `[一作]` `[通讯]` (Indian Institute of Technology Madras), Krishna Murari (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一族基于波函数的自适应激活函数，用以改善物理信息神经网络（PINNs）在求解偏微分方程时的失效模式。

**💡 创新点**

创新点在于将可训练的 Morlet、Mexican hat、Hermite、Gaussian、Gabor 等波函数与双曲正切、softplus 等激活相结合，形成可学习的激活函数；通过在 PINNs 中引入这些激活，显著提升了训练稳定性和逼近精度。

**🔧 技术方法**

采用 PINNs 框架、PyTorch 实现、L‑BFGS 优化器（强 Wolfe 线搜索）、软加权参数、正向传播及反向传播、基于波函数的激活设计。

**📊 数据集**

使用 1D 反应、波动、对流方程的解析解数据；2D Navier‑Stokes 方程使用公开的数值基准数据。

**📈 对比分析**

与传统 tanh 激活、SoftMexTanh 等同类激活，以及 PINNsFormer、PINN‑Mamba、ML‑PINN、QRes、FLS 等方法做对比；通过 loss、相对 MAE、相对 RMSE、L1/L2 误差、条形图等指标，证明所提激活在所有测试 PDE 上均取得更低误差、更快收敛、训练速度提升。

**⚠️ 局限性**

局限性包括：对激活函数参数仍需经验性初始化，某些激活在对流问题表现不佳；实验规模仅覆盖四类基础 PDE，缺乏更大尺度或高维问题验证；对 Navier‑Stokes 缺少解析参考，误差评估受限；以及仍需进一步优化训练策略（如 Adam‑L‑BFGS 混合）以提升收敛和可扩展性。

---

## 366. One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control

**arXiv ID:** 2603.18093 | [PDF](https://arxiv.org/pdf/2603.18093v1)

**作者:** Haoxiang Rao `[一作]` (Nanjing University), Caifeng Shan `[通讯]` (Nanjing University)

**通讯引用:** 9168 | [OpenAlex ID](https://openalex.org/A5055478558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练-free的少样本异常生成方法，利用一张参考异常图像通过三支扩散分支的自注意力嫁接来合成背景保持、位置可控且真实感强的异常图像。

**💡 创新点**

创新点在于：①三支注意力嫁接（TriAG）将参考异常特征与正常背景特征分离后混合；②对文本嵌入进行轻量级优化（AGO），使文本语义与异常视觉更匹配；③双重注意力增强（DAE）在特定时刻提升掩码区域的注意力，保证异常完整填充。

**🔧 技术方法**

主要技术包括：Stable Diffusion v1.5无训练推理；自注意力和交叉注意力的掩码编辑；DDIM反演与文本嵌入优化；分类器自由引导与负向提示；以及基于U-Net和ResNet-34的下游评估。

**📊 数据集**

使用工业异常检测基准MVTec‑AD数据集（15类），仅用每类约三分之一的异常图像作为参考，评估生成质量与下游检测/分类性能。

**📈 对比分析**

与基线（DFMGAN、AnomalyDiffusion、DualAnoDiff、SeaS、TF2、AnomalyAny等）对比，O2MAG在KID最低、IC‑LPIPS最优或相近、像素级AP和F1大幅提升（+约10%），分类准确率提升至最高，整体性能超过所有训练‑based和training‑free方法。

**⚠️ 局限性**

局限性：对极小或极细的异常仍可能出现填充不足；依赖参考异常图像，若缺失合适样本可能效果受限；跨类别迁移虽可行但需进一步验证在更广泛域间的适用性。

---

## 367. Constrained Hybrid Metaheuristic: A Universal Framework for Continuous Optimisation

**arXiv ID:** 2603.18295 | [PDF](https://arxiv.org/pdf/2603.18295v1)

**作者:** Piotr A. Kowalski `[一作]` (AGH University of Krakow), Jacek Mańdziuk `[通讯]` (Warsaw University of Technology)

**通讯引用:** 2261 | [OpenAlex ID](https://openalex.org/A5073814691)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种通用的约束混合元启发式（cHM）框架，用于连续优化，并在特征选择等实际任务中验证其有效性。

**💡 创新点**

创新点在于将多种元启发式（PSO、SA、GA、DE、BFO）通过探测（probing）和拟合（fitting）两阶段动态协同，形成层级协同机制，能够在不同优化阶段自动切换最适合的策略。

**🔧 技术方法**

使用了多种群基算法（PSO、SA、GA、DE、BFO）、两阶段自适应切换、约束处理、统计评估（均值、标准差、最小值）以及在特征选择中使用随机森林分类器和误差率作为评估指标。

**📊 数据集**

实验数据包括28个标准连续测试函数，以及三个公开分类数据集（Loan、Heart、Diabetic Retinopathy）。

**📈 对比分析**

与单一元启发式比较时，cHM在大多数基准函数上实现了与或优于最佳单一算法的性能，平均误差率最低，收敛速度和迭代次数适中；在特征选择任务中误差率更低且特征子集更为稳定。

**⚠️ 局限性**

局限性包括对初始种群高度敏感、标准差较大、对探测/拟合时间与迭代次数等参数设置依赖明显，以及在某些特定问题上并未显著优于单一最优算法。

---

## 368. Who Tests the Testers? Systematic Enumeration and Coverage Audit of LLM Agent Tool Call Safety

**arXiv ID:** 2603.18245 | [PDF](https://arxiv.org/pdf/2603.18245v1)

**作者:** Xuan Chen `[一作]` (Purdue University), Xiangyu Zhang `[通讯]` (Purdue University)

**通讯引用:** 309424 | [OpenAlex ID](https://openalex.org/A5100362465)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种meta‑audit框架，通过LLM枚举工具调用工作流和用户场景来自动生成潜在的不安全交互模式，并使用规则抵抗度度量评估现有安全基准的完整性。

**💡 创新点**

创新点在于①构建LLM驱动的工作流枚举器，系统化覆盖多样化工具调用；②引入非语义的规则抵抗度量，将安全规则转化为可验证的约束来量化基准的盲点。

**🔧 技术方法**

技术主要包括：链式思考结构化提示、LLM自动枚举工作流、规则抽取与压缩、基于规则的逐步评估与不安全案例识别，以及LLM判定器做安全标签。

**📊 数据集**

数据集为三大LLM代理安全基准（Agent‑SafetyBench、AgentHarm、ToolSafety）覆盖12个环境与多种工具，实验还使用8个后端LLM模型（GPT‑4o‑mini、Claude‑3.5‑haiku、Llama‑3.1‑8b等）。

**📈 对比分析**

与DirectGen、SIRAJ等基线比较，本文方法在三大基准与多环境下的未覆盖率均显著提高（平均提升≈20%），并发现约11%新型不安全模式，表明基准通过规则已覆盖约80%安全场景。

**⚠️ 局限性**

局限性包括：枚举过程仍为开环，未利用已发现的失败来动态引导生成；规则实现仅通过系统提示，未探讨更高级的执行时监控或约束解码；以及将发现的未覆盖案例转化为有效训练信号仍需进一步研究。

---

## 369. The Spillover Effects of Peer AI Rinsing on Corporate Green Innovation

**arXiv ID:** 2603.18415 | [PDF](https://arxiv.org/pdf/2603.18415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 370. Approximate Subgraph Matching with Neural Graph Representations and Reinforcement Learning

**arXiv ID:** 2603.18314 | [PDF](https://arxiv.org/pdf/2603.18314v1)

**作者:** Kaiyang Li `[一作]` (University of Connecticut), Wei Li `[通讯]` (Georgia State University)

**通讯引用:** 97832 | [OpenAlex ID](https://openalex.org/A5100318082)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于强化学习的近似子图匹配方法（RL-ASM），利用图Transformer提取完整图信息，并结合分支限界搜索实现更优匹配。

**💡 创新点**

首次将强化学习与图Transformer结合用于ASM；通过交叉注意力和全局注意力提升表达能力；引入预训练仿真学习和PPO细化；使用节点排序和GraphGPS增强性能。

**🔧 技术方法**

图Transformer（GraphGPS）、Laplacian Positional Encoding、随机游走结构编码、交叉注意力、强化学习（PPO）与仿真学习。

**📊 数据集**

四个基准数据集：SYNTHETIC、AIDS、MSRC_21、EMAIL。

**📈 对比分析**

与NeuroMatch、APM、ISM等方法对比；在无噪声下APM最优，噪声情况下RL-ASM在GED上优于所有基线，尤其在大图上效率提升约5倍，整体效果更好。

**⚠️ 局限性**

仍受搜索树规模和时间限制；依赖预训练样本；对极大图计算与内存负担较大；主要在特定噪声水平下验证，对非诱导子图的适应性需要进一步提升。

---

## 371. TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation

**arXiv ID:** 2603.19039 | [PDF](https://arxiv.org/pdf/2603.19039v1)

**作者:** Yan Shu `[一作]` (University of Trento), Paolo Rota `[通讯]` (University of Trento)

**通讯引用:** 1907 | [OpenAlex ID](https://openalex.org/A5046536801)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一的视觉语言模型 TerraScope，能够在地球观测任务中实现像素级的空间推理。

**💡 创新点**

创新性地将分割掩码与文本推理链交错生成，实现像素级链式推理，并支持多模态（光学+SAR）与多时相推理。

**🔧 技术方法**

基于 InternVL3 的双解码器架构，配合像素掩码生成器、跨模态注意力与时间指示器，实现视觉与文本的动态交互。

**📊 数据集**

构建了 1M 样本的 Terra-CoT 数据集和 3,837 条测试样本的 TerraScope-Bench 基准，并在 Landsat30-AU 与 DisasterM3 等公开数据集上进行评测。

**📈 对比分析**

在 11 种 VLM（包括通用与 EO 专用模型）上进行零样本与微调对比，TerraScope 在 TerraScope-Bench 上以 68.9%/73.9%/46.5% 的准确率领先，并在 Landsat30-AU 与 DisasterM3 上表现出优良的泛化能力。

**⚠️ 局限性**

仍在距离测量和建筑变更估计等精细任务上表现不足，且需要大量像素级标注和复杂的多模态融合技术。

---

## 372. Unsupervised Contrastive Learning for Efficient and Robust Spectral Shape Matching

**arXiv ID:** 2603.18924 | [PDF](https://arxiv.org/pdf/2603.18924v1)

**作者:** Feifan Luo `[一作]` (Zhejiang University), Hongyang Chen `[通讯]` (Zhejiang Lab)

**通讯引用:** 39606 | [OpenAlex ID](https://openalex.org/A5010419481)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种无监督对比学习与简化功能映射相结合的非刚性可变形3D形状匹配框架，利用正负相似对增强特征一致性与辨别力，省去传统功能映射求解器；

**💡 创新点**

1）首次将无监督对比学习引入3D形状匹配；2）设计两种对比损失（跨形状与自对比）直接提升特征嵌入；3）构建仅含软点映射与单一对齐损失的轻量功能映射模块；

**🔧 技术方法**

无监督对比学习（跨形状对比损失 + 自对比损失），softmax软点映射，谱基投影求功能映射，单一对齐损失；

**📊 数据集**

FAUST、SCAPE、SHREC'19、SMAL、DT4D-H（含重网格、各向异性网格与跨数据集）等公开3D形状数据集；

**📈 对比分析**

与基准方法（Axiomatic、Supervised、Unsupervised）在多种基准上对比，平均地理误差（×100）上取得最优或次优（仅次于少数同类方法），且计算效率显著提升；

**⚠️ 局限性**

不支持部分形状匹配或极端非等距变形，需结合显式空间变形方法进一步提升鲁棒性。

---

## 373. Learning to Self-Evolve

**arXiv ID:** 2603.18620 | [PDF](https://arxiv.org/pdf/2603.18620v1)

**作者:** Xiaoyin Chen `[一作]` (Mila), Yuxiong He `[通讯]` (Snowflake)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了“Learning to Self‑Evolve (LSE)”框架，利用强化学习在测试时训练大语言模型（LLM）根据前一轮任务表现自动更新自身上下文（prompt），从而在后续任务中提升性能。

**💡 创新点**

核心创新在于：①将自我进化明确建模为单步强化学习问题，奖励定义为上下文改进带来的性能提升（ΔR = R(c1)–R(c0)），避免了传统多步奖励的信用分配困难；②结合树形搜索（UCB）在测试时动态探索不同上下文路径，弥补单步训练的探索不足；③展示了在不同任务域上训练的自我进化策略能够迁移至其它模型并显著提升其表现。

**🔧 技术方法**

技术要点包括：
- Prompt‑based self‑evolution：LLM在保持参数冻结的情况下，仅通过编辑 prompt 进行自我改进。
- 单步 RL 训练（GRPO / PPO 等）与改进奖励。
- 树形搜索（Upper Confidence Bound）用于测试时多轮演化。
- 评估指标：文本到 SQL 的执行准确率、问答准确率。
- 代码与实验实现基于 Qwen3‑4B‑Instruct，使用公开 API 进行推理。

**📊 数据集**

实验数据集：
- BIRD（Text‑to‑SQL）——5 个随机抽取的数据库。
- MMLU‑Redux / SuperGPQA（多选问答）——10 个学科子任务。

**📈 对比分析**

对比方法：
- 先进闭源模型 GPT‑5、Claude Sonnet 4.5。
- 现有 prompt‑优化方法 GEPA 与 TextGrad。
- 未训练的 Qwen3‑4B‑Instruct 与种子 prompt。
- 结果显示：在 BIRD 上 LSE 的 4B 模型达到 72.0%（超过 GPT‑5 70.8% 与 Claude 70.8%），在 MMLU‑Redux 上 LSE 达到 73.3%（匹配 GPT‑5 72.5% 并优于 Claude 72.0%）。相较于未训练模型提升约 5–6%，相较于 GEPA/ TextGrad 提升 4–5%。

**⚠️ 局限性**

局限性：
- 采用单步奖励，未直接优化多步演化轨迹，可能错过更优的长期策略。
- 训练仅针对单一任务域，缺乏跨域通用性；未来需要大规模多域训练。
- 只更新指令字段，未考虑工具、技能库或外部记忆的演化。
- 实验规模有限，环境构建与任务设计仍需进一步系统化。
- 树形搜索在测试时的计算开销与搜索深度选择尚未系统评估。

---

## 374. Follow the Rules (or Not): Community Norms and AI-Generated Support in Online Health Communities

**arXiv ID:** 2603.19093 | [PDF](https://arxiv.org/pdf/2603.19093v1)

**作者:** Shravika Mittal `[一作]` (Georgia Institute of Technology), Munmun De Choudhury `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 15153 | [OpenAlex ID](https://openalex.org/A5102962995)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对在线健康社区的文本支持规范进行清单化，并使用GPT-4生成的回复与LLM评估和专家评审，量化其符合度及潜在风险。

**💡 创新点**

首次将隐式与显式社区规范系统化，并结合LLM评判与专家评估，揭示AI生成支持的合规性与风险，提供调控建议。

**🔧 技术方法**

采用GPT-4生成回复、Llama-3作为评判模型、开放编码专家审查，结合自然语言处理技术。

**📊 数据集**

基于Reddit‑QA，包含17,341条关于阿片康复的帖子，聚焦5个相关子版块。

**📈 对比分析**

通过LLM‑as‑a‑judge得到各规范符合比例（如信息超载85.24%符合，指导性建议违反34.99%），无对比基线，表现为多项符合度高但仍有违规。

**⚠️ 局限性**

仅评估单一模型单轮回复、仅文本形式、仅阿片康复情境、未包含多轮交互或其他健康领域、评估方法依赖人工验证有限。

---

## 375. Tula: Optimizing Time, Cost, and Generalization in Distributed Large-Batch Training

**arXiv ID:** 2603.18112 | [PDF](https://arxiv.org/pdf/2603.18112v1)

**作者:** Sahil Tyagi `[一作]` (Oak Ridge National Laboratory), Feiyi Wang `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 1948 | [OpenAlex ID](https://openalex.org/A5101916963)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对分布式大批量训练提出在线服务Tula，自动调优批量与集群规模以最小化时间、成本并提升模型精度。

**💡 创新点**

结合并行系统建模与梯度自适应缩放（AGS）两种创新，既可预测最优批量，又能弥补大批量的泛化缺陷。

**🔧 技术方法**

使用并行性能建模、内存估计、Kneedle自适应阈值、梯度可变性度量和基于梯度的自适应缩放技术。

**📊 数据集**

在ImageNet-1K、CIFAR-100、CalTech101/256上训练ResNet50、VGG11、AlexNet、MobileNetv3四个卷积网络。

**📈 对比分析**

与LBT、LRS、PoLo、LARS等方法对比，Tula在时间/成本上可达16–20倍加速，平均精度提升约8.8%，且能识别knee点。

**⚠️ 局限性**

局限性包括对网络波动的鲁棒性不足、对动态定价和多元成本模型未支持、部分搜索开销、梯度缩放超参数敏感，以及仅验证于视觉卷积模型。

---

## 376. A Vision-based Framework for Intelligent gNodeB Mobility Control

**arXiv ID:** 2603.18092 | [PDF](https://arxiv.org/pdf/2603.18092v1)

**作者:** Pedro Duarte `[一作]` (INESC TEC), Manuel Ricardo `[通讯]` (INESC TEC)

**通讯引用:** 1319 | [OpenAlex ID](https://openalex.org/A5054470359)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出了一个基于视觉的框架，用于智能移动 gNB 的位置控制，包含 VisionRAN、VisionApp 和 VisionTwin 三个核心组件。

**💡 创新点**

创新点在于：1) 引入了两种新的 VisionRAN E2 服务模型（POS 与 VIS），实现多模态感知数据与控制指令的统一传输；2) 开发了基于深度 Q 网络（DQN）的 VisionApp，能够融合视觉与无线信息进行障碍感知的 gNB 移动控制；3) 构建了 VisionTwin 这一数字孪生环境，用于生成训练数据并验证整个系统。

**🔧 技术方法**

所用技术包括 O‑RAN 架构与 E2 接口、深度强化学习（DQN）、计算机视觉（RGB‑D 目标检测、坐标转换）、数字孪生仿真（OpenAI Gym API、OAI RF 仿真）以及 FlexRIC 近实时 RIC 框架。

**📊 数据集**

实验数据来源于真实 RGB‑D 摄像机录制的 25 秒视频（12 fps）以及 VisionTwin 生成的合成视觉与无线链路数据。

**📈 对比分析**

通过将移动 gNB 与静态 gNB 的 NLoS 时长、路径损耗、SNR 与吞吐量进行对比，实验表明在 VisionApp 控制下 NLoS 时长下降约 75%，SNR 更加稳定，吞吐量显著提升。

**⚠️ 局限性**

局限性包括：仅在单 UE 与单障碍物场景中验证；对目标检测与深度估计的准确性高度依赖；未涵盖多 UE/多障碍、多模态感知以及真实硬件平台的现场 RF 数据。

---

## 377. PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching

**arXiv ID:** 2603.18363 | [PDF](https://arxiv.org/pdf/2603.18363v1)

**作者:** Ruishuo Chen `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**通讯引用:** 3744 | [OpenAlex ID](https://openalex.org/A5082905458)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无监督的 LLM 微调框架 PowerFlow，利用分布匹配而非启发式奖励来激发模型的推理和创造能力。

**💡 创新点**

创新点在于：① 用 α‑power 目标分布对模型进行定向调节（α>1 进行分布锐化以提升推理，α<1 进行分布扁平化以恢复创造性）；② 设计了长度感知的 Trajectory‑Balance（LA‑TB）目标，有效消除自回归生成中的长度偏差；③ 将 GFlowNet 视为对未归一化密度的变分采样器，形成稳定的分布匹配训练流程。

**🔧 技术方法**

技术手段包括：GFlowNet、变分推断、长度感知 Trajectory‑Balance 目标、α‑power 目标分布、格式惩罚、权重裁剪（w）、离线采样与 off‑policy 训练。

**📊 数据集**

数据集：推理任务使用 NuminaMath‑CoT、MATH500、OlympiadBench、AIME24/25、AMC23、GPQA；创造任务使用 300 条诗歌、故事、笑话提示（来源于 PoemHunter.com、BookMIA、Reddit r/DadJokes）。

**📈 对比分析**

通过与 Base、低温、Instruction、Format‑only、RLIF 方法（Intuitor、EMPO、TTRL）、PowerSampling、One‑shot EM、以及监督 GRPO 进行对比，PowerFlow 在大多数模型规模和基准上均显著提升推理准确率（例如 Qwen2.5‑Math‑7B 在 MATH500 上从 69.30% 提升至 78.10%，与 GRPO 仅差 0.23%），且保持或提升多样性；在创造性写作中，PowerFlow（α=0.5）实现了质量与多样性双提升，Pareto 前沿明显优于 Instruction 与 High‑temp 等 baseline。

**⚠️ 局限性**

局限性：① α 的取值仍需人工经验调节，缺乏自动化调度；② 依赖基础分布的质量，若预训练模型已严重偏倚，匹配可能无法完全纠正；③ 训练成本较高，且在极端 α 值下可能出现模式崩溃或安全风险；④ 目前仅在有限的任务和规模上验证，未覆盖更大规模模型或更复杂对话场景。

---

## 378. Escaping Offline Pessimism: Vector-Field Reward Shaping for Safe Frontier Exploration

**arXiv ID:** 2603.18326 | [PDF](https://arxiv.org/pdf/2603.18326v1)

**作者:** Amirhossein Roknilamouki `[一作]` (Ohio State University), Ness B. Shroff `[通讯]` (Ohio State University)

**通讯引用:** 20170 | [OpenAlex ID](https://openalex.org/A5035752536)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了一种在离线强化学习中，利用向量场奖励引导预训练策略在部署时安全地沿不确定性边界持续探索，同时完成主要任务。

**💡 创新点**

创新点：将梯度对齐与旋转流两种奖励成分结合成向量场奖励，既能将策略吸引到预设不确定性阈值，又能防止“停车”现象，促使策略在边界上连续运动；并在理论上证明该奖励结构可保证策略在长期平均意义下聚焦边界且不产生自循环。

**🔧 技术方法**

技术：利用离线数据训练不确定性上界器 U(s)，构造梯度对齐项 α(s)⟨∇U,Δs⟩ 和旋转流项 β(s)⟨W∇U,Δs⟩；在模拟环境中用 Soft Actor-Critic（SAC）结合 Normalizing Flow 的演员网络进行训练；在 2 维导航任务上验证。

**📊 数据集**

数据集：使用从真实环境收集的离线数据 D 来训练模拟器、误差上界器和不确定性估计器；实验采用一个简单的 2 维连续导航任务（点机器人、目标区、红色不确定区）。

**📈 对比分析**

对比方法：纯状态基 intrinsic 奖励（只鼓励访问 U(s) ≈ U_mid）和带安全惩罚的基线；评估指标包括边界覆盖度、沿边界速度、危险转移率以及任务完成率。实验结果显示，向量场奖励能实现全局沿边界的连续运动，覆盖度高、危险转移率低，且在任务完成阶段能有效切换到目标追踪；基线则出现模式坍塌、覆盖度低。整体性能优于纯状态奖励基线。

**⚠️ 局限性**

局限性：在高维连续空间中，旋转流的定义（W 矩阵）不唯一，可能导致轨道崩塌或仅覆盖部分边界；需要进一步设计可变或条件化的旋转流（如引入 W 作为输入或加入最大熵噪声）以获得完整表面覆盖；此外，奖励仍需依赖离线数据构建的不确定性上界器，若该上界不准确，安全性可能受影响。

---

## 379. From Concepts to Judgments: Interpretable Image Aesthetic Assessment

**arXiv ID:** 2603.18108 | [PDF](https://arxiv.org/pdf/2603.18108v1)

**作者:** Xiao-Chang Liu `[一作]` (KU Leuven), Johan Wagemans `[通讯]` (KU Leuven)

**通讯引用:** 18830 | [OpenAlex ID](https://openalex.org/A5014055174)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种可解释的图像审美评估框架，通过学习人类可理解的审美概念并在概念子空间中进行线性预测，辅以残差预测器以提升精度。

**💡 创新点**

创新点在于：①直接在概念子空间内进行可解释的稀疏线性预测，避免后置解释；②通过概念激活向量（CAV）利用正负图像集合快速学习审美概念；③加入轻量残差模块补偿非概念因素，兼顾性能与解释性。

**🔧 技术方法**

技术方法包括：使用CLIP-ResNet50提取图像嵌入，训练线性SVM获得概念激活向量，构建概念子空间，使用稀疏线性回归（Elastic Net）做可解释预测，以及线性残差回归补偿。

**📊 数据集**

实验数据集涵盖摄影领域的AADB、PARA、AVA，以及艺术领域的LAPIS和BAID，使用这些数据学习概念并进行跨数据集评估。

**📈 对比分析**

与多种基线（如NIMA、MLSP、Charm、DINOv2等）对比，混合模型在AADB、PARA、AVA、LAPIS、BAID上取得与最优方法相近甚至领先的SRCC/PLCC成绩；单一可解释分支虽然略逊，但已达到可接受的性能。

**⚠️ 局限性**

局限性在于可解释预测采用线性模型，难以捕捉概念间的复杂交互；残差模块虽提升性能但仍保持简单，可能无法全面解释所有细微审美影响。

---

## 380. MOSAIC: Multi-Objective Slice-Aware Iterative Curation for Alignment

**arXiv ID:** 2603.18637 | [PDF](https://arxiv.org/pdf/2603.18637v1)

**作者:** Yipu Dou `[一作]` (Southeast University), Wang Yang `[通讯]` (Southeast University)

**通讯引用:** 18847 | [OpenAlex ID](https://openalex.org/A5100322935)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 MOSAIC 框架，利用闭环数据构建方法在固定的监督微调预算下平衡安全对齐、过度拒绝和指令跟随三大目标。

**💡 创新点**

创新点在于统一 L1–L3 评估接口，将切片级故障分析转化为可执行的数据分配决策，实现多目标数据混合搜索；并通过微调回合闭环映射故障到数据动作。

**🔧 技术方法**

采用统一的 L1–L3 评价接口、基于 LoRA 的独立微调、基于预算的宏观混合向量和微观桶级权重，以及基于非支配集的 Pareto 前沿搜索。

**📊 数据集**

使用 XGuard-Train、OrBench 样本与 IFEval 数据集，共计约 64k+8k+9k 训练窗口。

**📈 对比分析**

通过内部搜索指标与外部独立评测对比，MOSAIC 的 Pareto 方案在 XGuard、OrBench、IFEval 上均优于基线，且相较随机静态混合能显著降低攻击成功率、减少过度拒绝，同时保持指令遵循能力。

**⚠️ 局限性**

实验仅进行五轮搜索，基线仅为随机混合，数据与评测来自同一构建管道，且高风险切片样本稀缺，限制了更广泛的泛化和更精细的优化。

---

## 381. EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models

**arXiv ID:** 2603.18489 | [PDF](https://arxiv.org/pdf/2603.18489v1)

**作者:** Minsoo Cheong `[一作]` (Seoul National University), Sungjoo Yoo `[通讯]` (Seoul National University)

**通讯引用:** 7614 | [OpenAlex ID](https://openalex.org/A5063521444)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EntropyCache，一种训练无关的 KV 缓存策略，通过判断新解码词的最大熵来决定是否全量重算或仅重算最近的 k 个词，从而降低 Diffusion‑LLM 推理的计算成本。

**💡 创新点**

创新点在于发现解码词熵与 KV 缓存漂移高度相关，利用 O(V) 的常数开销进行重算决策，并引入多步特征波动窗口来恢复缓存失效问题，显著提升吞吐量而不显著牺牲准确率。

**🔧 技术方法**

核心技术包括：熵计算（最大熵作为重算触发阈值）、历史记录维护（k 最近解码词的缓存重算），以及基于并行窗口和自信度阈值的分块/滑窗解码策略；整体不需要额外训练或显式对齐。

**📊 数据集**

在 LLaDA‑8B‑Instruct 与 Dream‑7B‑Instruct 两个 diffusion‑LLM 上，使用 GSM8K、MATH500、MBPP、HumanEval 及链式推理（CoT）基准进行评测。

**📈 对比分析**

与基线、Fast‑dLLM、Elastic‑Cache、d²Cache 等方法相比，EntropyCache 在标准基准上实现 15.2×–26.4× 的速度提升，同时保持或超过基线准确率；在 CoT 长上下文任务中获得 22.4×–24.1× 的平均速度提升，表现优于其它动态缓存方案。

**⚠️ 局限性**

局限性包括：需要手动调节熵阈值 τ 与最近词数 k 的超参数；对极长上下文或高多样性任务的准确性仍有一定下降；目前仅在两款 diffusion‑LLM 上验证，尚未证明对其他模型的通用性。

---

## 382. MedQ-UNI: Toward Unified Medical Image Quality Assessment and Restoration via Vision-Language Modeling

**arXiv ID:** 2603.18465 | [PDF](https://arxiv.org/pdf/2603.18465v1)

**作者:** Jiyao Liu `[一作]` (Fudan University), Ningsheng Xu `[通讯]` (Fudan University)

**通讯引用:** 10164 | [OpenAlex ID](https://openalex.org/A5101020368)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种 assess‑then‑restore 的统一医学影像质量评估与修复框架 MedQ‑UNI，使用单一多模态自回归模型实现质量描述生成与图像修复；

**💡 创新点**

创新点在于把质量评估与修复耦合成双专家共享注意力架构，利用结构化自然语言描述作为中间条件，使模型在不需要手工指定退化类型的情况下自适应多模态、多退化场景；

**🔧 技术方法**

采用 Vision‑Language 自回归 Transformer + VAE 低维编码 + 共享注意力 + 两阶段训练（先修复再联合评估），并在恢复阶段加入像素级 L1 与 SSIM 损失；

**📊 数据集**

构建了约 50K 样本的大规模数据集，包含 CT、MRI、PET 三种模态、五种修复任务的低/高质量图像对，并配有专家审核的结构化质量描述；

**📈 对比分析**

与 UniMedVL、Pix2Pix、Restore‑RWKV、AMIR、MPerceiver 等基线对比，MedQ‑UNI 在 PSNR/SSIM 上均超过所有对手，CT 去噪 +2.57 dB、PET 去噪 +1.23 dB；在质量描述评估上，MedQ‑UNI 也获得 GPT‑4o 以上的分数，显示更高的完整性、精准性和一致性；

**⚠️ 局限性**

主要限制是依赖大量人工审核的质量描述数据，且目前仅覆盖三种模态和五个修复任务，对更稀缺模态或极端退化场景的泛化尚待验证。

---

## 383. Rethinking Vector Field Learning for Generative Segmentation

**arXiv ID:** 2603.19218 | [PDF](https://arxiv.org/pdf/2603.19218v1)

**作者:** Chaoyang Wang `[一作]` (Baidu), Yunhai Tong `[通讯]` (Peking University)

**通讯引用:** 4835 | [OpenAlex ID](https://openalex.org/A5024097240)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于流匹配的生成式分割框架 FlowSeg，解决了梯度消失与路径穿越问题。

**💡 创新点**

提出向速度场添加距离感知的纠正项以实现吸引与排斥，并使用基于 Kronecker 序列的准随机类别编码。

**🔧 技术方法**

流匹配（deterministic flow）、向量场重塑、Kronecker 方案、像素神经场解码、REPA、温度软最大等技术。

**📊 数据集**

ADE20K 和 COCO-Stuff 两个高类别语义分割数据集。

**📈 对比分析**

与 DeepLabV3+、SegFormer、MaskFormer 等判别式基线以及 InstructDiffusion、PixWizard、SymmFlow 等生成式模型比较，ADE20K 47.1% mIoU、COCO-Stuff 44.9% mIoU，超过判别式基线并显著缩小两者差距。

**⚠️ 局限性**

仍受限于离散类别映射的连续化误差，未解决多尺度或细粒度边界细化的挑战；对极高类别数仍可能出现聚类密集导致梯度竞争。

---

## 384. 3DreamBooth: High-Fidelity 3D Subject-Driven Video Generation Model

**arXiv ID:** 2603.18524 | [PDF](https://arxiv.org/pdf/2603.18524v1)

**作者:** Hyun-kyu Ko `[一作]` (Yonsei University), Eunbyung Park `[通讯]` (Yonsei University)

**通讯引用:** 1689 | [OpenAlex ID](https://openalex.org/A5013897558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出3DreamBooth和3Dapter两阶段框架，实现对多视角3D对象的视频定制，生成高保真、视角一致的视频。

**💡 创新点**

①1帧优化方案将空间几何与时序动态解耦；②多视角视觉适配器3Dapter作为动态选择路由器，显式注入视角特征；③结合LoRA微调和双分支注意力实现高效收敛。

**🔧 技术方法**

使用视频扩散Transformer（DiT）作为主干，LoRA实现可训练权重；多视角联合注意力、RoPE编码；在单视角预训练后再进行多视角联合优化。

**📊 数据集**

收集30个复杂3D对象的全360°多视角图像，构建3D-CustomBench评测基准；使用Subjects200K训练3Dapter；在HunyuanVideo-1.5预训练模型上微调。

**📈 对比分析**

与VACE、Phantom等单视角定制方法对比；在多视角主体保真度、Chamfer距离、视频质量与文本对齐等指标上表现更优，尤其在3D几何精度上将误差减半。

**⚠️ 局限性**

仅依赖少量参考视图，仍可能在极端光照或复杂动态场景下产生细节失真；模型对极端角度的泛化仍有限，需进一步扩充视角覆盖。

---

## 385. Maximum-Entropy Exploration with Future State-Action Visitation Measures

**arXiv ID:** 2603.18965 | [PDF](https://arxiv.org/pdf/2603.18965v1)

**作者:** Adrien Bolland `[一作]` (University of Liège), Damien Ernst `[通讯]` (University of Liège)

**通讯引用:** 14940 | [OpenAlex ID](https://openalex.org/A5077011518)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的最大熵强化学习目标，使用未来访问特征分布熵作为内在奖励。

**💡 创新点**

创新点在于将条件访问分布的熵作为奖励，并证明其为另一最大熵目标的下界，同时可通过收缩算子离线估计。

**🔧 技术方法**

使用了条件访问分布的收缩算子、交叉熵学习、Soft Actor-Critic（SAC）框架以及基于几何分布的采样。

**📊 数据集**

在 MiniGrid 迷宫环境上进行实验。

**📈 对比分析**

与基于动作熵、均匀访问分布等传统探索策略比较，实验显示在单轨迹特征覆盖和学习速度上更优，控制性能相近。

**⚠️ 局限性**

局限在于需要预先指定特征空间，且对高 γ 或高维空间的收敛稳定性有限，模型表达能力受限于可训练的密度估计器。

---

## 386. Recolour What Matters: Region-Aware Colour Editing via Token-Level Diffusion

**arXiv ID:** 2603.18466 | [PDF](https://arxiv.org/pdf/2603.18466v1)

**作者:** Yuqi Yang `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 8020 | [OpenAlex ID](https://openalex.org/A5039812471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ColourCrafter，一种基于扩散模型的区域感知色彩编辑框架；

**💡 创新点**

通过在潜在空间进行token级别的RGB颜色与图像特征融合，实现从全局色调迁移向区域感知的结构保持编辑；引入Lab空间感知损失和基于Mask的局部约束提升像素级色彩精度；创建了针对连续色彩变化的ColourfulSet数据集；

**🔧 技术方法**

使用Flux.1‑Kontext扩散变压器、LoRA微调、Token级多模态注意力、Lab空间损失、3D RoPE位置编码以及Mask生成与可视化等技术；

**📊 数据集**

采用自构造的ColourfulSet数据集（约80k高质量同一对象的连续色彩对），并在Unsplash等公开图像集上进行评测；

**📈 对比分析**

与ColorPeel、ColorBind、Control‑Color、Flux.1‑Kontext、FlowEdit、UniEdit‑Flow、IP‑Adapter等方法对比，实验表明ColourCrafter在ΔE00、ΔECh、MAE_RGB、MAE_Hue等色彩误差指标上显著优于对手，且在LPIPS、SSIM、FID等感知与结构指标上保持竞争或更优；

**⚠️ 局限性**

局限性包括：训练数据为合成色彩变体，可能缺乏真实世界色彩多样性；仅支持单目标单色彩编辑，无法一次性处理多目标或多色彩；依赖配对监督，难以直接迁移至无配对或真实场景。

---

## 387. Secure Wi-Fi Ranging Today: Security and Adoption of IEEE 802.11az/bk

**arXiv ID:** 2603.18687 | [PDF](https://arxiv.org/pdf/2603.18687v1)

**作者:** Nikola Antonijević `[一作]` (KU Leuven), Bart Preneel `[通讯]` (KU Leuven)

**通讯引用:** 22455 | [OpenAlex ID](https://openalex.org/A5039506639)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文结合标准分析、MATLAB 仿真与开发板测量，系统评估了 IEEE 802.11az/bk 规范下的安全 Wi‑Fi 距离测量（FTM）的逻辑层和物理层安全性，并对其在实际设备中的部署现状进行实证检查。

**💡 创新点**

①首次对 NGP 机制在逻辑层的身份绑定、降级、DoS 等安全缺陷进行综合评估；②对安全 HE‑LTF 波形的可预测性、符号重用与零功率 GI 对谱线性影响进行量化分析；③给出针对现有硬件与标准实现的改进建议。

**🔧 技术方法**

标准文本解析、MATLAB 基于 IEEE 802.11az 的波形与定位例程、MUSIC 超分辨率时延估计、贝叶斯推断与消息传播算法、硬件捕获与功率谱密度测量。

**📊 数据集**

使用自研开发板（公开未名厂商）捕获的安全/非安全 FTM 流；Google Pixel 9a 作为对照；模拟产生的 20/40/80 MHz NDP 信号。

**📈 对比分析**

通过对比安全与非安全 FTM 的 ToA 误差、距离偏差与 EVM，评估了安全波形在不同观察比例下的攻击可行性；对比谱密度与 IEEE 802.11 定义的发射谱线性，量化零功率 GI 对规约合规性的影响。

**⚠️ 局限性**

仅基于单一未名开发板实现，无法覆盖所有硬件差异；实验样本有限，缺乏大规模实地部署数据；理论模型假设理想信道，未考虑复杂多径/干扰；结论对未来硬件改进与标准修订具有指导意义但需进一步验证。

---

## 388. In the Margins: An Empirical Study of Ethereum Inscriptions

**arXiv ID:** 2603.19086 | [PDF](https://arxiv.org/pdf/2603.19086v1)

**作者:** Xihan Xiong `[一作]` (Imperial College), Qin Wang `[通讯]` (CSIRO Data61)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对以太坊 Ethscriptions 进行大规模经验研究，定义工作负载边界并提取 4.75M 条 JSON 操作记录，对其生命周期、协议演化、参与度与永久数据足迹进行系统测量。

**💡 创新点**

创新点在于：①构建了基于 URI 前缀的精准工作负载数据集；②提出并规范化了 Ethscription 操作语法；③量化了部署→铸造、铸造→转移的 funnel 转换率；④用 Gini 系数和 Top‑N 分析揭示参与极度集中；⑤首次测算 5.3 GB 永久 calldata 足迹。

**🔧 技术方法**

使用了以太坊完整节点日志、前缀过滤、MIME 分类、JSON 语法解析与标准化、时间序列与分段分析、协议标签统计、生命周期 funnel 计算、Gini 与 Lorenz 曲线、存储占比估算等技术。

**📊 数据集**

以太坊主网交易历史（截至 2024‑02‑26）为数据源，识别出 6.27 M 个 `data:` 前缀候选，最终提取 4.75 M 个符合 JSON 语法的 Ethscription 操作。

**📈 对比分析**

通过对比不同阶段的交易量、协议份额、转移率等指标，展示了 9 个月的 boom‑bust 周期：部署到铸造放大 201×、铸造到转移下滑 57.6:1，Gini 为 0.858，整体永久足迹达 5.3 GB，显著高于其他工作负载。

**⚠️ 局限性**

局限性包括：仅识别 `data:` 前缀的变体，可能漏掉非标准编码；JSON 语法严格导致极少量操作被排除；仅关注链上 calldata，未评估离链索引器的状态差异；结果基于截至 2 月的数据，后续月度波动可能略有偏差。

---

## 389. Continually self-improving AI

**arXiv ID:** 2603.18073 | [PDF](https://arxiv.org/pdf/2603.18073v1)

**作者:** Zitong Yang `[一作]` `[通讯]` (Stanford University), Zitong Yang (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了三条路径，使大型语言模型能够自我提升：①利用知识图谱启发的合成数据扩增（EntiGraph）实现小语料的持续预训练；②在不借助外部教师的前提下，利用模型自身生成合成文本来提升预训练困惑度（SBP）；③通过在测试时搜索算法空间（AI‑Designed AI）提升训练过程本身。

**💡 创新点**

创新点在于：①首次将知识图谱结构用于合成数据生成，从而在保持信息完整的同时显著提升数据多样性；②提出了Synthetic Bootstrapped Pretraining框架，避免外部教师泄露，实现在相同数据集上自我增益；③构建了端到端的AI研究自动化系统，并通过搜索算法空间实现训练流程自我优化。

**🔧 技术方法**

技术手段包括：知识图谱生成与关系扩散（EntiGraph），条件文本生成器（SBP的条件合成器），自监督预训练（Causal LM），大规模近似最近邻搜索，自动化实验执行与评估，链式思维与预算强制等。

**📊 数据集**

实验使用的主要数据集为：QuALITY（小语料库及对应的问答），Coursera讲座转录与考试问答，DCLM（582M文档约482B标记）用于SBP的预训练；此外还利用OpenWebText2、LAMBADA、ARC、SciQ、Winogrande、TriviaQA、WebQS等标准评测数据。

**📈 对比分析**

比较方法采用compute‑matched、data‑constrained框架：在相同训练token数与计算资源下，与重复训练基线（重复原始数据）和无限数据oracle对比；SBP在1T级别可提升约60%相对于oracle的性能缺口，在小语料下EntiGraph持续预训练可提升闭合书式QA约70%相对于基线。

**⚠️ 局限性**

局限性包括：合成数据可能存在hallucination，需要更严谨的事实性验证；SBP依赖近似最近邻和条件生成模型的质量，可能受限于数据多样性；AI‑Designed AI在搜索规模与资源消耗上仍高；所有方法均基于Transformer架构，未验证跨架构推广性。

---

## 390. Probabilistic Federated Learning on Uncertain and Heterogeneous Data with Model Personalization

**arXiv ID:** 2603.18083 | [PDF](https://arxiv.org/pdf/2603.18083v1)

**作者:** Ratun Rahman `[一作]` (University of Alabama), Dinh C. Nguyen `[通讯]` (University of Alabama)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

Meta‑BayFL方法提出了一种将贝叶斯神经网络（BNN）与元学习结合的个性化联邦学习框架，旨在解决联邦学习中不确定且异构的数据导致的训练退化问题。

**💡 创新点**

创新点包括：1) 在隐藏层显式建模不确定性以提升小样本与噪声数据下的稳定性；2) 通过元学习自适应学习率搜索实现个性化更新，提升局部训练效果；3) 将概率模型与个性化设计统一到全局聚合中，首次提供理论收敛分析并证明其对通信轮数的上界。

**🔧 技术方法**

采用的技术主要有贝叶斯神经网络（ELBO最大化）、元学习（自适应学习率搜索）、联邦平均（FedAvg）、理论收敛分析、Monte Carlo dropout对比实验。

**📊 数据集**

实验使用CIFAR‑10、CIFAR‑100和Tiny‑ImageNet三大图像数据集，并在非IID分布（Step、Dirichlet）下进行。

**📈 对比分析**

方法与FedAvg、FedProx、BayesFL、pFedBayes、pFedBe、Fedmask等基线在同一非IID设置下进行对比；Meta‑BayFL在所有数据集上均实现平均提升约7.42%的测试精度，并在小样本、噪声环境中表现出显著优势。

**⚠️ 局限性**

主要限制是：使用BNN导致模型参数量、训练计算量与通信开销增加，对极端资源受限的IoT设备不够友好；若无压缩或轻量化的近似方法，实际部署可能受限。

---

## 391. Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging

**arXiv ID:** 2603.18834 | [PDF](https://arxiv.org/pdf/2603.18834v1)

**作者:** Hesong Li `[一作]` (Beijing Institute of Technology), Ying Fu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 5751 | [OpenAlex ID](https://openalex.org/A5100738025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统计特征引导的高分辨率透射电子显微镜（HRTEM）图像去噪网络SCGN，能够在快速成像条件下显著提升原子级别的噪声抑制效果。

**💡 创新点**

创新点包括：① 空间偏差引导加权模块可针对不同像素局部噪声强度自适应选择卷积操作；② 频带引导加权模块通过频域内容与位置的双重信息动态增强信号、抑制噪声；③ 针对核化过程设计的HRTEM噪声校准与无序原子生成方法，构建了专用NUC数据集。

**🔧 技术方法**

技术主要使用了卷积神经网络、窗口标准差计算、实时FFT/逆FFT、位置嵌入、通道注意机制以及基于统计特征的动态加权策略，训练采用Adam优化和L1损失。

**📊 数据集**

使用了三类数据集：TEMImageNet、SFIN数据集以及新构建的NUC数据集，并在真实HRTEM核化实验数据上验证。

**📈 对比分析**

与高斯滤波、AtomSegNet、SFIN、HINT、UDVD等方法比较，SCGN在PSNR/SSIM上均取得最高分（例如在NUC上PSNR提升0.85 dB，SSIM提升0.0064），且在原子定位任务的IoU上领先约2–3个百分点。

**⚠️ 局限性**

局限性在于模型和噪声校准方法专为HRTEM核化场景设计，对不同成像模式或更复杂噪声分布的通用性尚未验证，需要进一步研究更多统计特征或跨模态适应。

---

## 392. Elastic Weight Consolidation Done Right for Continual Learning

**arXiv ID:** 2603.18596 | [PDF](https://arxiv.org/pdf/2603.18596v1)

**作者:** Xuan Liu `[一作]` (Sun Yat-sen University), Xiaobin Chang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1076 | [OpenAlex ID](https://openalex.org/A5021121018)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文重新审视了弹性权重整合（EWC）及其变体，发现其权重重要性估计存在梯度消失和冗余保护问题，并提出Logits Reversal操作以纠正Fisher信息矩阵，从而设计出改进的EWC-DR方法。

**💡 创新点**

创新点在于通过梯度基重要性分析揭示EWC的梯度消失和MAS的冗余保护缺陷，并引入简单的Logits Reversal技术来重新估计权重重要性，使得EWC的性能得到显著提升。

**🔧 技术方法**

采用的技术包括梯度重要性分析、Fisher信息矩阵、Logits Reversal操作、正则化权重重要性、ResNet-18/ViLT模型、PyCIL框架和SGD训练策略。

**📊 数据集**

实验使用的数据集涵盖了CIFAR-100、ImageNet-Subset、Tiny-ImageNet（用于EFCIL）以及VQAv2、NLVR2、SNLI-VE、VCR（用于多模态持续指令微调）。

**📈 对比分析**

在EFCIL和MCIT场景下，与EWC、Online EWC、MAS、SI等基线方法比较，EWC-DR在A_last、A_avg、A_t以及forgetting transfer等指标上均优于所有对比方法，提升幅度可达50%以上。

**⚠️ 局限性**

局限性在于仅针对全连接层权重重要性进行分析，未考虑网络其他层的贡献；对LR参数的敏感性及其在更广泛任务（如文本或多模态非视觉任务）中的泛化性仍待进一步验证。

---

## 393. Lightweight Model Predictive Control for Spacecraft Rendezvous Attitude Synchronization

**arXiv ID:** 2603.18921 | [PDF](https://arxiv.org/pdf/2603.18921v1)

**作者:** Peter Stadler `[一作]` (e:fs TechHub GmbH), Alen Turnwald `[通讯]` (Ingolstadt University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并实现了两种基于逆序姿态偏差的轻量级线性MPC，用于航天器在对接同步过程中的姿态跟踪，并在Basilisk仿真与ARM Cortex-M7嵌入式平台上验证了其实时可行性。

**💡 创新点**

采用逆序姿态偏差定义，使角速度约束天然线性；提出单环与双环线性MPC，双环利用稳定反馈把内部系统化为LTI，显著降低求解复杂度并提升跟踪精度。

**🔧 技术方法**

多重滚动预测控制（MPC）、CasADi符号建模与代码生成、QP求解器比较（DAQP、OSQP、qpOASES等）、Basilisk仿真框架、ARM Cortex-M7嵌入式编译。

**📊 数据集**

仿真参数包括惯性张量 J=diag(85,94,92) kg·m²、参考角速度 (0.4,0.3,0) rad/s、角速度/扭矩约束 ±0.5 rad/s/±40 N·m，未使用真实航天器数据。

**📈 对比分析**

通过仿真比较三种MPC（之前的、单环、双环）在跟踪误差、累计扭矩、运行时和内存使用上的表现。结果显示双环MPC在跟踪精度上更快误差衰减，运行时与内存占用低于单环和传统MPC，且在嵌入式平台上可在30步预测下实时运行。

**⚠️ 局限性**

仅在高保真仿真环境验证；未考虑外部扰动或鲁棒性；仅处理姿态问题，未扩展到SE(3)；需要进一步在嵌入式平台闭环测试并改进鲁棒性与多航天器协同控制。

---

## 394. Adaptive Decoding via Test-Time Policy Learning for Self-Improving Generation

**arXiv ID:** 2603.18428 | [PDF](https://arxiv.org/pdf/2603.18428v1)

**作者:** Asmita Bhardwaj `[一作]` (University of California San Diego), Basel Shbita `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于强化学习的解码器采样器，利用可学习的策略在测试时动态调整LLM的采样参数（如温度和top‑p），而不修改模型权重；

**💡 创新点**

创新之处在于将解码视为马尔可夫决策过程，学习轻量级策略以实现任务和领域感知的自适应解码；

**🔧 技术方法**

采用PPO算法训练一个2层MLP策略网络，并设计复合奖励（ROUGE、长度、覆盖率、重复率、完整性）进行训练；

**📊 数据集**

在三大摘要数据集BookSum、arXiv和WikiHow上进行评估；

**📈 对比分析**

与贪婪解码和固定温度基线比较，RL策略在Granite‑3.3和Qwen‑2.5上分别实现了高达+88%和+79%的相对提升，且训练过程显示奖励随时间稳步上升；

**⚠️ 局限性**

局限性包括：奖励设计对效果高度敏感；仅在摘要任务上验证，未测试对对话、代码生成等更复杂任务的迁移；在模型外域的鲁棒性尚未评估；

---

## 395. MAED: Mathematical Activation Error Detection for Mitigating Physical Fault Attacks in DNN Inference

**arXiv ID:** 2603.18120 | [PDF](https://arxiv.org/pdf/2603.18120v1)

**作者:** Kasra Ahmadi `[一作]` (University of South Florida), Reza Azarderakhsh `[通讯]` (Florida Atlantic University)

**通讯引用:** 4311 | [OpenAlex ID](https://openalex.org/A5064156050)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 MAED（Mathematical Activation Error Detection）算法级错误检测框架，利用激活函数的数学恒等式在推理阶段实时验证非线性激活运算的正确性，从而抵御物理故障注入攻击与自然硬件故障。

**💡 创新点**

创新点在于：①首次将算法级错误检测与激活函数结合，针对 DeepLaser 等针对激活函数的攻击；②采用逆变换与快速 Maclaurin 级数近似实现低成本一致性校验；③在 MCU 与 FPGA 平台实现时均保持极低硬件占用（<1% 时钟周期开销，FPGA 几乎无面积开销）且检测覆盖率接近 100%。

**🔧 技术方法**

主要技术包括：数学恒等式验证（如 y/(1−y)=e^x、α−1/α+1 等），Taylor 级数（Maclaurin）近似 e^x、e^2x，重计算法（ReLU 的 x 与 -x 对比），硬件实现基于浮点加乘除单元，软件实现集成 TensorFlow 静态图编译。实验环境包括 AMD/Xilinx Artix‑7 FPGA、ATmega328P 微控制器以及 TensorFlow 2.x。

**📊 数据集**

使用 NASA Turbofan Jet Engine 数据集训练一个 1D 卷积+全连接的六分类 RUL 预测模型，用于评估激活函数的推理时间与错误检测的性能。

**📈 对比分析**

通过与未加检测的基线激活函数对比，测量推理时间、模型大小、检测覆盖率与错误率。结果显示：在 MCU 上误检率<1%，FPGA 仅增加约 20% 延迟；在 TensorFlow 中，检测开销几乎可忽略，模型准确率保持 0.71–0.82 之间；在多种注入模型与注入强度下，错误检测比率均超过 95%，在大多数情形下达到 100%。

**⚠️ 局限性**

局限性在于：仅针对激活函数层进行错误检测，未覆盖权重或其他网络层的故障；对某些极端的、全局性的数据错误或复杂的攻击方式（如跨层传播的错误）可能不完全有效；实现细节依赖于 IEEE 双精度浮点运算，低精度量化网络时需重新设计。

---

## 396. ATG-MoE: Autoregressive trajectory generation with mixture-of-experts for assembly skill learning

**arXiv ID:** 2603.19029 | [PDF](https://arxiv.org/pdf/2603.19029v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 397. SoK: Practical Aspects of Releasing Differentially Private Graphs

**arXiv ID:** 2603.18779 | [PDF](https://arxiv.org/pdf/2603.18779v1)

**作者:** Nicholas D'Silva `[一作]` (University of New South Wales), Salil S. Kanhere `[通讯]` (University of New South Wales)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对差分隐私（DP）图发布方法进行系统化分类，提出面向实践的目标导向框架，并在社交网络两个场景下对现有方法进行统一基准评估。

**💡 创新点**

① 引入层级系统化与漏洞维度，明确方法的图模型、信任模型、隐私目标与变换过程；② 设计面向实践的四个评估目标（隐私保证、可行性、实证隐私、效用）；③ 通过两场景实验提供首个统一的DP图发布基准。

**🔧 技术方法**

差分隐私（edge‑DP、node‑DP、ADP）、DP‑SGD、深度生成模型（VAE、GAN、Transformer GNN）、图扰动+生成策略、重建与链接预测攻击等。

**📊 数据集**

四个来自 Stanford Network Analysis Project 的静态无向社交网络图：Facebook、LastFM、GitHub、Brightkite。

**📈 对比分析**

对选定的 12 种方法在 12 个隐私预算（ε）下进行实验，评估指标包括描述性误差（度分布、中心性、模块化等）、仿真任务误差（影响力扩散）、预测任务误差（链接预测、节点分类）以及重建攻击成功率。实验结果显示 CAGMDP、TriCycLe、TmF 在保持结构特征方面优于多数方法；但大多数方法在中心性、社区结构及预测任务中表现波动，且攻击成功率随 ε 稳定，体现隐私‑效用权衡。

**⚠️ 局限性**

仅关注差分隐私框架，未覆盖非独立数据或其他隐私定义；评估仅限社交网络，难以推广到其他领域；部分大规模图因计算限制未能评估；节点隐私方法的实用性与可扩展性仍不足。

---

## 398. Shifting Uncertainty to Critical Moments: Towards Reliable Uncertainty Quantification for VLA Model

**arXiv ID:** 2603.18342 | [PDF](https://arxiv.org/pdf/2603.18342v1)

**作者:** Yanchuan Tang `[一作]` (Rutgers University), Ruixiang Tang `[通讯]` (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于token熵的机器人视觉语言动作模型不确定性量化框架，利用局部窗口最大池化、动作不稳定性加权和贝叶斯优化自适应DoF加权实现对失败的精准检测。

**💡 创新点**

创新点在于：①打破“平均陷阱”，通过滑动窗口最大池化保留短暂的高熵尖峰；②引入动作转移加权（ATR），将高频振荡的时间点赋予更高权重；③利用贝叶斯优化对窗口长度、振荡权重及各自由度重要性进行自适应调优。

**🔧 技术方法**

技术包括：token熵计算、滑动窗口最大池化、动作不稳定性指示符、加权熵求和、贝叶斯优化（高斯过程）以及AUROC/阈值阈值化决策。

**📊 数据集**

使用LIBERO基准（包括LIBERO-SPATIAL、LIBERO-OBJECT、LIBERO-GOAL和LIBERO-10四个子集），在OpenVLA 7B模型的仿真执行轨迹上评估。

**📈 对比分析**

与传统全局平均熵基线相比，提出方法在所有子集上显著提升AUROC与准确率，例如在LIBERO-SPATIAL从0.845提升至0.936，在LIBERO-10从0.468提升至0.838，且跨套件零样本迁移亦保持优于基线。

**⚠️ 局限性**

局限在于仍需基于仿真数据进行评估，实际机器人部署时可能受硬件噪声、传感器误差影响；同时贝叶斯优化调参虽样本效率高，但对极端长周期任务的窗口长度选择仍需经验判断。

---

## 399. GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning

**arXiv ID:** 2603.19137 | [PDF](https://arxiv.org/pdf/2603.19137v1)

**作者:** Yiren Lu `[一作]` (Case Western Reserve University), Yu Yin `[通讯]` (Case Western Reserve University)

**通讯引用:** 9574 | [OpenAlex ID](https://openalex.org/A5101655848)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了GSMem框架，利用3D Gaussian Splatting构建持久的空间记忆，使智能体能够在任务驱动的探索与推理中通过后期渲染重新观察已探索区域；

**💡 创新点**

创新点包括：①将连续的3DGaussian场与实时生成的语言字段结合，实现开放词汇的检索与重观；②多层检索-渲染机制，融合物体级场景图与语义级语言场景，提升定位鲁棒性；③混合探索策略，结合VLM语义评分与3DGS信息增益，实现任务导向与几何覆盖的平衡；

**🔧 技术方法**

核心技术包括3D Gaussian Splatting、在线语言场景字段构建、基于CLIP的语义检索、可视化渲染视角优化、单步扩散增强、以及基于信息论的前沿探索评估；

**📊 数据集**

在两个基准上进行实验：Active Embodied Question Answering（A‑EQA）使用Habitat‑Matterport3D（HM3D）场景；多模态终身导航（GOAT‑Bench）使用GOAT‑Bench的36个未见场景；

**📈 对比分析**

与四类基线对比：盲目LLM、LLM+字幕、VLM直接观测、以及VLM探索方法（Explore‑EQA、ConceptGraphs、3D‑Mem）。在A‑EQA上取得SPL和LLM‑Match最高分；在GOAT‑Bench上显著提升成功率与SPL，尤其在长期导航任务中表现突出；

**⚠️ 局限性**

局限性包括：①对实时语言字段的高质量特征提取依赖于CLIP等预训练模型；②信息增益评估近似（T‑optimality）可能不完全捕捉3DGS的全局不确定性；③在极端光照或遮挡场景下渲染质量仍受限；④高计算开销（约1.2秒/步）限制了极端实时性需求。

---

## 400. Proceedings of the 2nd Workshop on Advancing Artificial Intelligence through Theory of Mind

**arXiv ID:** 2603.18786 | [PDF](https://arxiv.org/pdf/2603.18786v1)

**作者:** Nitay Alon `[一作]`, Stefan Sarkadi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

论文探讨了在多智能体强化学习中，如何通过潜在的心智理论来增强智能体的决策能力。

**💡 创新点**

创新点在于提出了一种新的方法，将心智理论与环境动态的反事实模拟结合起来，以提高智能体的适应性和协作能力。

**🔧 技术方法**

使用了强化学习和反事实推理的技术。

**📊 数据集**

使用了多智能体环境的模拟数据集，具体数据集未详细说明。

**📈 对比分析**

与传统的强化学习方法相比，提出的方法在多智能体协作任务中表现出更好的适应性和效率。

**⚠️ 局限性**

限制在于模型的复杂性和计算资源需求较高，可能不适用于资源受限的环境。

---

## 401. Beyond Ray-Casting: Evaluating Controller, Free-Hand, and Virtual-Touch Modalities for Immersive Text Entry

**arXiv ID:** 2603.18435 | [PDF](https://arxiv.org/pdf/2603.18435v1)

**作者:** Md. Tanvir Hossain `[一作]` (University of Rajshahi), M. Khademul Islam Molla `[通讯]` (University of Rajshahi)

**通讯引用:** 653 | [OpenAlex ID](https://openalex.org/A5047156012)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了六种物理输入系统（控制器、自由手、虚拟触控）在三种交互风格（控制器驱动、自由手、虚拟触控）下的文本输入性能，并与语音输入做对照。

**💡 创新点**

提出控制器驱动的点击-滑动组合（CD‑TGC）可大幅提升文本输入速度和准确率，同时揭示用户体验与性能之间的权衡；首次在同一实验框架下系统评估多种VR文本输入模式。

**🔧 技术方法**

采用Meta Quest 2/3头显的手部追踪、Ray‑casting、虚拟键盘投影，并利用Unity与C#桌面服务器实现数据同步；使用WPM、TER、SUS、GEQ等定量与定性指标。

**📊 数据集**

使用MacKenzie与Soukoreff标准短语集（已筛选掉小于三字的单词）进行文本输入实验，共计735个试验。

**📈 对比分析**

通过在21名参与者的within‑subjects实验中测量WPM和TER，发现CD‑TGC最高达17.09 WPM、TER 5.80%，比最慢系统快2.25倍、比行业标准快30%；虚拟触控点击模式在SUS评分上最高（74.76），但速度最低。

**⚠️ 局限性**

实验仅限单次会话，缺乏长期学习评估；未使用自动校正或补全；使用Remote Desktop桥接导致轻微延迟，可能影响自由手条件下的实时反馈。

---

## 402. Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens

**arXiv ID:** 2603.19232 | [PDF](https://arxiv.org/pdf/2603.19232v1)

**作者:** Yuqing Wang `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 3892 | [OpenAlex ID](https://openalex.org/A5027234036)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出CubiD，一种能够直接生成768-1024维视觉表示的立方体离散扩散模型；

**💡 创新点**

其创新点在于对整个3D token张量进行细粒度的维度级掩码，使模型在保持语义丰富度的同时实现高效并行生成；

**🔧 技术方法**

主要技术包括维度级量化（dimension-wise quantization）对连续特征离散化、使用Transformer的双向注意力进行多维度上下文建模，以及采用学习型掩码token和余弦调度的离散扩散训练方案；

**📊 数据集**

实验使用ImageNet 256×256数据集，并结合冻结的DINOv2-B和SigLIP2-B编码器生成特征；

**📈 对比分析**

与传统使用低维度token的离散扩散/自回归方法相比，CubiD在ImageNet 256×256上实现了gFID 1.88的最佳成绩，并在模型规模从946M到3.7B时表现出良好的扩展性；

**⚠️ 局限性**

局限性包括对高维编码器的依赖、在大规模特征上仍需高算力、以及在迁移到其他数据集或多模态任务时可能需要进一步调整和优化。

---

## 403. Revisiting Autoregressive Models for Generative Image Classification

**arXiv ID:** 2603.19122 | [PDF](https://arxiv.org/pdf/2603.19122v1)

**作者:** Ilia Sudakov `[一作]` (Yandex Research), Dmitry Baranchuk `[通讯]` (Yandex Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

重新审视自回归模型（AR）在生成式图像分类中的应用，提出利用任意顺序的RandAR模型进行多顺序平均的顺序边缘化分类框架；

**💡 创新点**

发现固定token顺序是AR分类器的瓶颈，并通过多顺序边缘化显著提升判别能力；

**🔧 技术方法**

采用RandAR任意顺序AR模型、VQ‑VAE离散化、Monte Carlo顺序采样、log‑likelihood下界估计等技术；

**📊 数据集**

在ImageNet1K（256×256）及其OOD扩展集ImageNet‑R/S/A、ImageNet‑C（Gaussian/JPEG）上进行评估；

**📈 对比分析**

与扩散分类器（DiT、SiT）、自监督SSL模型（DINOv2）、其他AR/VAR/A‑VARC等基线对比，结果显示在所有验证集上均优于扩散分类器，OOV上更具优势，且推理速度可达25倍；

**⚠️ 局限性**

主要限制包括：对大类别数的计算开销仍较高、随机顺序采样虽有效但需多前向传播、仅在VQ‑VAE+RandAR框架内验证，未来需探索模型蒸馏和自适应顺序预测等改进方向。

---

## 404. Cognitive Mismatch in Multimodal Large Language Models for Discrete Symbol Understanding

**arXiv ID:** 2603.18472 | [PDF](https://arxiv.org/pdf/2603.18472v1)

**作者:** Yinghui Li `[一作]` (Tsinghua University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 134895 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了一个覆盖语言、文化、数学、物理和化学五大符号领域的三层认知层级评测基准，用于系统评估多模态大型语言模型（MLLM）在离散符号空间中的视觉符号理解能力。

**💡 创新点**

创新点包括：①首次将符号认知分为感知识别、组合推理和关联批判三层，并针对每层设计细粒度任务；②提出专门的符号基准，填补自然图像评测对离散符号能力的空白；③通过人类基准和跨模型对比揭示了MLLM在符号感知上的显著缺陷及“识别–推理逆转”现象。

**🔧 技术方法**

技术手段：采用大规模预训练的视觉-语言对齐模型（CLIP、ViT 等）与后续指令微调；使用层次化评测指标（F1、准确率、编辑距离、语义相似度等）；对模型进行多维度分析（域内性能、难度层级、跨域相关性）。

**📊 数据集**

数据集：从公开基准（VisualC3、eWe-bench、MultiMath-300K、ChemBench-4K、OlympiadBench 等）中提取并重标注，人工补充错误样本，最终构成 13,148 条样本，涵盖 5 个领域、3 个层级、38 个子任务。

**📈 对比分析**

比较方法：在 9 个代表性 MLLM（包括 GPT‑4o、Gemini‑2.5‑Pro、Claude‑Sonnet‑4、Qwen‑Max、o3、DeepSeek‑VL2‑Tiny、Qwen2.5‑VL、LLaMA3‑LLaVA‑Next‑8B、InternVL3‑8B）上执行统一评测。结果显示：①在认知层级上模型存在“识别–推理逆转”；②语言和文化符号域整体性能低于自然科学符号；③顶级模型在跨域性能上不一致，且对离散符号的精细感知普遍不足。

**⚠️ 局限性**

局限性：①评测依赖于手工标注的离散符号样本，规模虽大但仍受限于可获取的公开数据；②使用的视觉编码器（ViT 等）对稀疏、低分辨率符号的定位与解析存在瓶颈；③评测聚焦于静态图像，对动态图像或交互式符号场景尚未覆盖；④模型在高层次批判推理时往往依赖语言先验，缺乏真正的视觉推理链。

---

## 405. Are complicated loss functions necessary for teaching LLMs to reason?

**arXiv ID:** 2603.18756 | [PDF](https://arxiv.org/pdf/2603.18756v1)

**作者:** Gabriele Carrino `[一作]` (Politecnico di Milano), Mark James Carman `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对GRPO损失函数进行系统剖析，提出简化后的RGRA方法，并在多语言数学与STEM基准上进行评估。

**💡 创新点**

创新点在于：①确认负反馈与优势估计对稳定学习至关重要；②证明PPO式裁剪不必要，可通过去除裁剪直接提升性能；③引入RGRA，保留组相对优势估计而去除复杂的PPO约束，显著简化训练。

**🔧 技术方法**

使用REINFORCE与组相对优势估计（RGRA）、GRPO、GRPO-positives、RAFT、直接REINFORCE以及基线指令调优模型，采用LoRA微调、梯度累积与VLLM推理。

**📊 数据集**

训练数据：gsm8k 训练集（1800条），评测数据包括英语数学基准（GSM8K、MATH、Gaokao2023-Math-En、OlympiadBench、AMC23）、中文数学基准（CMATH、CN-Middle-School）和STEM基准（MMLU-STEM、Gaokao2024）。

**📈 对比分析**

与GRPO、RAFT、直接REINFORCE及原始指令调优模型比较，RGRA在大多数基准上超越GRPO（17/27对比），在英语数学、中文数学及STEM评测中平均准确率提升5–10%；训练过程更稳定、无奖励崩溃。

**⚠️ 局限性**

局限性在于仅在中等规模（0.5B/1.5B）模型上验证，未测试更大模型；仅针对数学和STEM任务，未探究其他领域的可迁移性；实验依赖于gsm8k训练集，可能受制于数据覆盖范围。

---

## 406. ProCal: Probability Calibration for Neighborhood-Guided Source-Free Domain Adaptation

**arXiv ID:** 2603.18764 | [PDF](https://arxiv.org/pdf/2603.18764v1)

**作者:** Ying Zheng `[一作]` (Hong Kong Polytechnic University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5930 | [OpenAlex ID](https://openalex.org/A5044722301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 ProCal 方法，解决源无关域自适应中邻域一致性导致的源知识遗忘和局部噪声过拟合问题。

**💡 创新点**

创新点在于将源模型先验预测与目标模型在线输出融合的概率校准框架，并通过软监督与多样性损失双重约束实现邻域概率的动态校准和分布对齐。

**🔧 技术方法**

技术包括 k‑近邻特征聚合、双模型概率融合、soft‑supervision 与 diversity 损失的联合优化、梯度与固定点理论分析以及基于 ResNet、ViT、ConvNeXt 的特征提取。

**📊 数据集**

实验数据集涵盖 Office‑31、Office‑Home、VisDA‑C 与 DomainNet‑126 四大公开跨域任务，包含 31 个子任务。

**📈 对比分析**

与 SHOT、SHOT++、AaD、NRC 等现有 SFDA 方法对比，ProCal 在所有四个数据集上均取得新 SOTA，平均提升约 1‑3%，在 DomainNet‑126 的 12 个子任务中 9 个得到最高分。

**⚠️ 局限性**

局限性包括对源模型先验的依赖（尽管鲁棒性高）、对邻域大小与记忆更新频率等超参的敏感度、以及缺乏对实时或多模态自适应场景的评估。

---

## 407. Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression

**arXiv ID:** 2603.18426 | [PDF](https://arxiv.org/pdf/2603.18426v1)

**作者:** Minjun Kim `[一作]` (Seoul National University), U Kang `[通讯]` (Seoul National University)

**通讯引用:** 29688 | [OpenAlex ID](https://openalex.org/A5065423939)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并系统研究了联合模型压缩中的压缩顺序问题，并给出了进阶强度假设。

**💡 创新点**

核心创新在于提出Progressive Intensity Hypothesis，即弱扰动先行、强扰动后行，并给出理论保证与实验验证。

**🔧 技术方法**

采用剪枝、量化、参数共享、LoRA等多种压缩技术，并基于多步压缩与混合精度量化。

**📊 数据集**

在大规模语言模型（LLaMA系列）和视觉模型（ResNet-18、DeiT-Base）上使用WikiText-2、ImageNet等数据集进行评估。

**📈 对比分析**

与随机顺序或单一压缩相比较，实验显示按假设顺序可提升性能，优势随压缩强度差距增大而单调增加。

**⚠️ 局限性**

局限性在于假设干扰可被量化，且对特定压缩方法的兼容性、硬件实现及自适应调度等仍需进一步研究。

---

## 408. TurboMem: High-Performance Lock-Free Memory Pool with Transparent Huge Page Auto-Merging for DPDK

**arXiv ID:** 2603.18690 | [PDF](https://arxiv.org/pdf/2603.18690v1)

**作者:** Junyi Yang `[一作]` `[通讯]` (Independent Researcher), Junyi Yang (Independent Researcher)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

**🎯 论文内容**

设计并实现了 TurboMem，一种基于 C++ 模板的无锁内存池，结合透明大页自动合并，旨在提升 DPDK 风格数据平面的内存分配性能。

**💡 创新点**

创新点：1）全无锁的 Treiber 栈与每核本地缓存并行工作；2）利用 madvise 自动触发 THP 合并，消除手动巨大页配置；3）严格 NUMA 与 CPU 亲和性，最大化本地化与缓存一致性。

**🔧 技术方法**

技术：C++ 模板、Treiber 栈、原子 CAS、线程本地缓存、madvise(MADV_HUGEPAGE) 透明大页、NUMA 感知、CPU pinning、Intel VTune 性能分析。

**📊 数据集**

数据集：在单 socket Intel Xeon 100 Gbps 服务器上使用 1 M 个 256 B 对象池，配合 64 B、1500 B 与 IMIX 流量进行实验。

**📈 对比分析**

比较方法：在相同硬件与工作负载下，将 TurboMem 与传统 DPDK mempool（1 GB hugetlbfs）对比，测量 MPPS、TLB miss、L3 请求和内存延迟；结果显示 TurboMem 吞吐率提升最高 28%，TLB miss 降低 41%，L3 请求与内存延迟显著下降。

**⚠️ 局限性**

限制：实验基于单 socket、低碎片的测试环境，真实系统评估仍在进行；多 socket、虚拟化环境以及动态 THP 开关等场景尚未覆盖。

---

## 409. R&D: Balancing Reliability and Diversity in Synthetic Data Augmentation for Semantic Segmentation

**arXiv ID:** 2603.18427 | [PDF](https://arxiv.org/pdf/2603.18427v1)

**作者:** Huy Che `[一作]`, Duc-Khai Lam `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套基于可控扩散模型的合成数据增强流水线，用来提升语义分割模型在数据不足场景下的性能；

**💡 创新点**

创新点在于同时使用图像到图像与可控修复两种扩散模型，并提出类别感知提示与视觉先验混合技术，既保证了生成图像的多样性，又保持了与原数据分布的一致性；

**🔧 技术方法**

采用Stable Diffusion XL与T2I‑Adapter/ControlNet实现可控扩散，结合BLIP图像描述、类重加权提示、视觉先验混合等技术；

**📊 数据集**

实验数据集包括PASCAL VOC 07/12和BDD100K；

**📈 对比分析**

与Baseline、DiffuMask及其它合成方法对比，使用DeepLabV3+、Mask2Former、TwinLiteNet等模型，结果显示在VOC上合成+微调可将mIoU提升至84.0%，在全数据集上亦显著提升，且在不同天气/场景下性能提升明显；FID/CLIP分数也优于对比方法；

**⚠️ 局限性**

主要局限包括对预训练扩散模型质量的高度依赖、生成过程计算成本高、在极少样本或特定隐私场景下的泛化能力待进一步验证。

---

## 410. Insight-V++: Towards Advanced Long-Chain Visual Reasoning with Multimodal Large Language Models

**arXiv ID:** 2603.18118 | [PDF](https://arxiv.org/pdf/2603.18118v1)

**作者:** Yuhao Dong `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 44604 | [OpenAlex ID](https://openalex.org/A5100406050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Insight‑V 与 Insight‑V++ 两个多代理视觉推理框架，结合可扩展的长链推理数据生成、推理与总结两代理协作以及自演进训练循环，显著提升多模态大语言模型的推理能力。

**💡 创新点**

创新点包括：①基于进化式多粒度评估的可扩展长链推理数据生成管线；②将推理与总结拆分为专门代理的双代理架构；③为推理代理和总结代理分别设计的 ST‑GRPO 与 J‑GRPO 强化学习策略；④通过自演进的协同训练循环实现无人工注释的持续自我提升。

**🔧 技术方法**

使用技术包括：大语言模型与多模态 LLM、进化式直接偏好优化（Iterative DPO）、基于群组相对策略优化（GRPO）变体、进阶数据生成与多粒度评估、上下文评分与奖励设计、以及强化学习与监督联合训练。

**📊 数据集**

使用的数据集涵盖公开大型数据集（LLaVA‑OneVision、Cauldron、Cambrian‑1、LLaVA‑Video、Video‑R1、Oryx、LLaVA‑NeXT、Qwen‑2.5‑VL 等），并通过内部生成的长链推理示例与 Gemini‑2.5‑Pro 的上下文示例进行自监督和奖励设计。

**📈 对比分析**

在 10 个图像推理基准和 6 个视频推理基准上与现有最先进 MLLM 进行对比，Insight‑V++ 在推理得分上平均提升 4–6% 绝对值，部分任务接近 GPT‑4o 的表现；消融实验验证了每项技术对性能提升的贡献。

**⚠️ 局限性**

局限性包括：需要大量算力和大规模模型；对极其复杂推理的生成数据质量仍有限；DPO 的离线不稳定性和 ST‑GRPO/J‑GRPO 的奖励设计对任务依赖性较强；在需要外部知识或非视觉信息的任务中表现仍有提升空间。

---

## 411. DriveVLM-RL: Neuroscience-Inspired Reinforcement Learning with Vision-Language Models for Safe and Deployable Autonomous Driving

**arXiv ID:** 2603.18315 | [PDF](https://arxiv.org/pdf/2603.18315v1)

**作者:** Zilin Huang `[一作]` (University of Wisconsin Madison), Sikai Chen `[通讯]` (University of Wisconsin Madison)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在强化学习驾驶决策中引入视觉语言模型（VLM）作为奖励信号，提出DriveVLM‑RL框架实现安全驾驶策略学习。

**💡 创新点**

创新点：① 神经双通道（静态CLIP对比目标 + 动态注意力门+多帧LVLM推理）实现对语义风险的实时评估；② 仅在离线训练期间使用VLM，部署时完全去除推理延迟；③ 通过层次化奖励合成将语义风险与车辆动力学紧耦合。

**🔧 技术方法**

技术：CLIP对比语言目标、Qwen3‑VL（4B）多帧推理、YOLOv8 小模型做注意力门、Soft Actor‑Critic（SAC）或PPO、异步奖励注解管线。

**📊 数据集**

数据集：CARLA城市仿真（Town 2 训练，Town 1‑5 测试），10条未见路线，30条随机路段；使用公开的CARLA语义分割、摄像头和BEV视角。

**📈 对比分析**

对比13种奖励设计（专家手工、LLM自动、VLM机器人/驾驶特化），DriveVLM‑RL在碰撞率降至0.126（比ChatScene 0.8下降84%），成功率提升至0.60，平均速度≈14 km/h，且在“无碰撞奖励后”实验中仍保持低碰撞，展示了更强的安全性和泛化能力。

**⚠️ 局限性**

局限性：在高速公路、多层道路等结构差异场景表现下降；仍依赖仿真视觉，缺乏对真实世界域漂移的适配；奖励语义词汇量有限，难以覆盖极端稀疏事件；未实现在线奖励自适应或多智能体协同。

---

## 412. CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think

**arXiv ID:** 2603.18991 | [PDF](https://arxiv.org/pdf/2603.18991v1)

**作者:** Zening Sun `[一作]` (Hong Kong University of Science and Technology), Zeke Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5100457290)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Composite Reward Assisted Fine‑Tuning（CRAFT）方法，用极少量（仅100条）样本实现扩散模型与人类偏好对齐。

**💡 创新点**

通过Composite Reward Filtering自动构造高质量训练集，并在理论上证明CRAFT等价于基于组强化学习的下界优化，显著降低对大规模偏好数据和高算力的依赖。

**🔧 技术方法**

使用HPS v2.1、PickScore、AES三种奖励模型的加权组合进行样本过滤；采用组优势估计的加权SFT损失；全参数微调Unet；优化器为AdamW。

**📊 数据集**

训练集来源于HPDv2的1万条原始prompt经Qwen‑Plus细化并使用Stable Diffusion生成图像；评估集包括HPDv2、Parti‑Prompt、Pick‑a‑Pic及Geneval。

**📈 对比分析**

与Diff‑DPO、Diff‑KTO、SmPO、SPO等基线在SDXL、SD1.5上做对比，指标为HPS v2.1、AES、ImageReward、MPS；CRAFT在所有指标上均优于基线，训练时间仅约4–5 GPU‑h，速度提升相对传统方法可达11–220×。

**⚠️ 局限性**

仍需依赖外部奖励模型，若奖励模型不稳健可能影响效果；目前仅验证于Stable Diffusion，跨模型与更大规模数据的泛化尚待评估。

---

## 413. Can LLMs Reason Like Automated Theorem Provers for Rust Verification? VCoT-Bench: Evaluating via Verification Chain of Thought

**arXiv ID:** 2603.18334 | [PDF](https://arxiv.org/pdf/2603.18334v1)

**作者:** Zichen Xie `[一作]` (University of Virginia), Wenxi Wang `[通讯]` (University of Virginia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种将Z3证明降维为可读Verus验证链的框架VCoT-Lift，并基于此构建VCoT-Bench评测集。

**💡 创新点**

关键创新是把低级solver证明抽象为高层“Verification Chain-of-Thought”，并用来细粒度评估LLM推理能力。

**🔧 技术方法**

采用LLM驱动的Transformer-Checker循环、证明剪枝与修复等技术，结合Z3规则层级引导和Verus验证器反馈。

**📊 数据集**

数据集基于Verus-Bench 150个已验证程序，生成1,988个VCoT补全任务。

**📈 对比分析**

与10个主流LLM进行零样本对比，整体准确率在70%以下，显示LLM在中间推理和断言完成上表现脆弱。

**⚠️ 局限性**

主要局限在LLM对长上下文推理不足、缺乏第一原理推断、以及思考模式在正式证明中的干扰。

---

## 414. GRAFITE: Generative Regression Analysis Framework for Issue Tracking and Evaluation

**arXiv ID:** 2603.18173 | [PDF](https://arxiv.org/pdf/2603.18173v1)

**作者:** Ja Young Lee `[一作]` (IBM Research), Sara Rosenthal `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为Grafite的Web平台，用于持续跟踪和评估LLM在领域特定问题上的表现。

**💡 创新点**

提出了将用户反馈转化为可复用的issue和测试，采用LLM-as-a-judge与人类注释相结合的评估方式，支持持续回归检测。

**🔧 技术方法**

使用Next.js、Ag-grid、Carbon Design System、RabbitMQ、FastAPI、MongoDB，并利用LLM-as-a-judge和LLM评审集成。

**📊 数据集**

基于人类生成的测试和来自Chatbot Arena的基准数据，覆盖10个领域共20个issue、110个测试；并在Meta Llama等公开模型上进行评估。

**📈 对比分析**

通过汇总不同模型版本的通过率并与人类评估对比，发现模型回归与改进，Llama-4表现最佳；平台可视化展示模型间差异。

**⚠️ 局限性**

评估受LLM评审偏差影响，界面交互仍需改进，未实现自动检测低效issue，依赖人工标注和手工操作。

---

## 415. Signals of Success and Struggle: Early Prediction and Physiological Signatures of Human Performance across Task Complexity

**arXiv ID:** 2603.18798 | [PDF](https://arxiv.org/pdf/2603.18798v1)

**作者:** Yufei Cao `[一作]` (Australian National University), Xuanying Zhu `[通讯]` (Australian National University)

**通讯引用:** 75 | [OpenAlex ID](https://openalex.org/A5070891817)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在一款层级递增的卡牌游戏中，研究者使用低复杂度阶段的眼动和心率信号，预测高复杂度阶段的胜负结果。

**💡 创新点**

创新点在于实现跨阶段的前瞻性性能预测，融合视觉与自主神经两种生理信号，并提供可解释的生理机制分析。

**🔧 技术方法**

采用 Tobii Pro Fusion 眼动追踪与 Empatica EmbracePlus 佩戴式心率采集，提取眼动、瞳孔、AOI 等特征；使用 XGBoost、CatBoost、SVM 训练模型，并在 LOSO（Leave-One-Subject-Out）框架下进行决策级融合。

**📊 数据集**

数据集为 35 名参与者在 Slay the Spire 游戏中完成的低复杂度与高复杂度两阶段记录，包含眼动、心率以及自评情绪评分。

**📈 对比分析**

与单模态（眼动或心率）相比，融合模型在 LOSO 下获得最高的平衡准确率 0.86（宏 F1 0.87，MCC 0.72），验证了跨模态信息的互补性。

**⚠️ 局限性**

局限包括仅在单一游戏情境下验证、样本量有限、腕部 PPG 采样率低导致 HRV 信息不足、缺乏任务前基线以及未能细分低绩效子群。

---

## 416. Beyond the Code: A Multi-Modal Assessment Strategy for Fostering Professional Competencies via Introductory Programming Projects

**arXiv ID:** 2603.18741 | [PDF](https://arxiv.org/pdf/2603.18741v1)

**作者:** Santiago Berrezueta-Guzman `[一作]` (Technical University of Munich), Stefan Wagner `[通讯]` (Technical University of Munich)

**通讯引用:** 9206 | [OpenAlex ID](https://openalex.org/A5022333047)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在《Fundamentals of Programming》课程中实施了一个四维多模态评估框架，结合迷宫游戏项目、技术录屏、现场答辩和同伴评审，以提升学生的技术与软技能。

**💡 创新点**

创新点在于将项目式学习与多模态评估（实时演示、录屏、答辩、匿名同行评审）整合为一个可扩展的四维模型，填补了传统代码评估无法衡量沟通与批判性思维的空白。

**🔧 技术方法**

使用了游戏开发技术（Java 2D、图形界面、音效）、视频录制与上传、在线提交平台Artemis、结构化评分表以及匿名评审表单。

**📊 数据集**

采用了138名一学期信息工程学生的项目提交、录屏和评审结果作为数据集，无外部公开数据集。

**📈 对比分析**

通过对比团队项目完成率、评分分布、同伴评审与教师评分的相关性以及问卷满意度，结果显示项目完成率89%，平均总分71.7/100，录屏与答辩成绩正相关，同行评审与教师评分相关系数约0.5，表明框架有效提升学习效果。

**⚠️ 局限性**

局限在于单一机构、单一课程、缺乏对照组、未进行纵向跟踪，且评估依赖学生自我报告与教师评估，缺乏长期效能验证。

---

## 417. Multi-Modal Building Change Detection for Large-Scale Small Changes: Benchmark and Baseline

**arXiv ID:** 2603.19077 | [PDF](https://arxiv.org/pdf/2603.19077v1)

**作者:** Ye Wang `[一作]` (Anhui University), Sibao Chen `[通讯]` (Anhui University)

**通讯引用:** 3484 | [OpenAlex ID](https://openalex.org/A5030559313)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LSMD大型小变更多模遥感变化检测基准与MSCNet模型

**💡 创新点**

设计了三种模块（NCEM、CAIM、SMRM）实现局部上下文增强、RGB–NIR跨模态对齐及基于语义先验的多源细化，兼顾小目标与植被遮挡

**🔧 技术方法**

采用MobileNetV2骨干、邻域上下文增强、双向跨模态注意力、变异加权交叉注意、RemoteSAM生成的语义掩码

**📊 数据集**

使用自建的LSMD数据集（8000张256×256高分RGB+NIR影像）以及公开SMARS数据集进行验证

**📈 对比分析**

与9个SOTA方法对比，MSCNet在LSMD上F1、IoU、Kappa均位居榜首；在SMARS上也获得最高指标，且参数量与FLOPs低，表现稳定

**⚠️ 局限性**

仍受限于单一二模态（RGB+NIR）及二分类目标，未覆盖多类别变化，且模型对极端光照或时间跨度大可能仍受影响

---

## 418. PhysVideo: Physically Plausible Video Generation with Cross-View Geometry Guidance

**arXiv ID:** 2603.18639 | [PDF](https://arxiv.org/pdf/2603.18639v1)

**作者:** Cong Wang `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 11203 | [OpenAlex ID](https://openalex.org/A5079572598)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 PhysVideo 两阶段框架，先生成多视角前景的物理驱动视频，再利用其引导完整背景视频合成，从而实现物理一致且可控的视频生成。

**💡 创新点**

创新点在于将物理感知注意力、几何增强的跨视角注意力以及时间注意力结合进扩散模型，无需显式 3D 表示即可生成多视角、物理一致的运动，并通过前景运动引导实现背景互动。

**🔧 技术方法**

采用扩散模型、物理感知注意力机制、几何辅助跨视角注意力、时间注意力、光流引导的后期合成以及 3D Gaussian splatting 渲染等技术。

**📊 数据集**

使用自建的 PhysMV 数据集（40K 场景、4 视角共 160K 视频序列）以及公开的 OpenVid、WISA 数据集进行训练。

**📈 对比分析**

与基于物理引擎的 PhysGen、OmniPhysGS 以及通用视频生成器 CogVideoX 等方法对比，实验表明 PhysVideo 在物理真实性、运动连贯性和视觉质量上均优于对手，GPT‑4o 评测得分最高。

**⚠️ 局限性**

限制在于对前景与背景分离的场景表现良好，但在复杂遮挡、动态背景或极端细节需求时仍可能出现细节失真和性能下降。

---

## 419. D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding

**arXiv ID:** 2603.19146 | [PDF](https://arxiv.org/pdf/2603.19146v1)

**作者:** Jonathan Lys `[一作]` (IMT Atlantique), Ghouthi Boukli Hacene `[通讯]` (Sony Europe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 D5P4，一种面向离散扩散语言模型的并行解码框架，通过分区判定点过程 (Partition DPP) 对候选集合进行 MAP 推理，实现对生成质量与多样性的平衡。

**💡 创新点**

创新点包括：①将 beam 选择建模为分区 DPP 的 MAP 推理，显式考虑候选间的相似性；②利用扩散模型自身的熵与隐藏表示估计质量与多样性，避免外部评估器；③设计可在多 GPU 上高效执行的贪心 MAP 求解器，几乎不增加额外计算。

**🔧 技术方法**

技术手段：离散扩散语言模型（MDLM、LLADA），判定点过程（DPP）与分区约束，贪心 MAP 近似，熵/自信度评分，Jina embeddings 语义相似度，评估指标包括 PPL、MAUVE、Cosine Similarity、Distinct‑n、Self‑BLEU、EAD、F1、Wasserstein。

**📊 数据集**

使用的数据集：FineWeb（开放式文本生成评估）、TruthfulQA 与 CommonSenseQA（问答任务评估）。

**📈 对比分析**

对比方法：Baseline（独立采样）、Beam Search、Diverse Beam Search（Transversal MMR）、非分区 DPP 采样。实验表明，D5P4 在多样性-质量 Pareto 前沿优于所有基线，显著提升多样性而质量保持竞争，且计算开销几乎为零。

**⚠️ 局限性**

局限性：仍受底层扩散模型多样性能力的限制；在强 CFG 情况下多样性提升有限；仅针对离散扩散语言模型，未针对图像等其他模态；未解决生成内容的安全与偏见问题。

---

## 420. ManiDreams: An Open-Source Library for Robust Object Manipulation via Uncertainty-aware Task-specific Intuitive Physics

**arXiv ID:** 2603.18336 | [PDF](https://arxiv.org/pdf/2603.18336v1)

**作者:** Gaotian Wang `[一作]` (Rice University), Kaiyu Hang `[通讯]` (Rice University)

**通讯引用:** 1256 | [OpenAlex ID](https://openalex.org/A5011451275)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 ManiDreams，一个用于不确定性感知的机器人操纵规划框架，采用分布式状态表示、可插拔的动力学预测、约束定义和优化器，实现样本-预测-约束循环。

**💡 创新点**

创新点在于：①将感知、参数和结构不确定性统一为分布式状态（DRIS）并通过插件化抽象实现可插拔；②在不训练重塑下通过循环评估候选动作来提升鲁棒性；③提供跨后端（物理模拟器与学习型世界模型）兼容的统一接口。

**🔧 技术方法**

使用的技术包括：分布式状态抽象（DRIS）、任务特定直观物理接口（TSIP）、捕获约束（caging constraints）、采样-优化求解器（MPPI、N-best）以及基于深度扩散模型或 ManiSkill3 的动力学后端。

**📊 数据集**

主要数据集与环境：ManiSkill3 默认任务（PushCube、PickCube、PushT）用于仿真评估；在真实世界中使用 Franka Panda 搭配 SAM2 视觉分割，执行在杂乱物体上进行的推拉与抓取；同时使用 YCB 物体数据集作为推送任务的目标对象。

**📈 对比分析**

通过对比 PPO 基线，在不同噪声、延迟、物理扰动下进行鲁棒性基准；结果显示 ManiDreams 在大多数扰动条件下成功率显著高于基线（提升 10–30%），并在仿真与实物部署中保持一致。对比实验采用10次重复、100条轨迹统计平均值与标准差。

**⚠️ 局限性**

局限性包括：①需要手动设置 DRIS 的实例数与物理参数分布；②捕获约束仍需针对任务手工定义；③当前循环在高动态任务下计算延迟过高（超过 50 Hz），不适合实时高速操纵；④对极端不确定性时分布覆盖不足，需更自适应的 DRIS 配置。

---

## 421. From Topic to Transition Structure: Unsupervised Concept Discovery at Corpus Scale via Predictive Associative Memory

**arXiv ID:** 2603.18420 | [PDF](https://arxiv.org/pdf/2603.18420v1)

**作者:** Jason Dury `[一作]` `[通讯]` (Independent Researcher), Jason Dury (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用时间共现对比学习，将预训练文本嵌入映射到关联空间，挖掘跨书的转移结构概念，生成多分辨率概念地图。

**💡 创新点**

在压缩约束下让模型忘记单个共现对，而压缩出跨作者、跨流派的转移模式，得到聚焦功能而非主题的概念，并展示从宏观到微观的层级结构。

**🔧 技术方法**

对比学习MLP（4层，GELU+LayerNorm+残差）训练InfoNCE损失，使用BGE‑large‑en‑v1.5预嵌入，后续在关联空间上做k‑means聚类。

**📊 数据集**

Project Gutenberg 9,766本英文书籍（约25M段落），共3.73亿个15‑chunk窗口内的时间共现对。

**📈 对比分析**

与原始BGE相似度聚类、上下文均值聚类、随机MLP等基线比较；在5本未见小说上，PAM模型在k=100聚类中仅激活约1/2簇且集中度高，BGE则分布在近全簇；训练精度42.75%，但得到的簇在功能、注册、传统等方面更具可解释性。

**⚠️ 局限性**

仅单一训练跑、未做多种种子；shuffle控制仅在pilot；BGE基线仅在2K子集；概念标签由LLM生成，缺乏正式人类评测；未做下游任务评估；仅限英语且Gutenberg偏向19-20世纪；块大小未尝试变化；未验证压缩比例与概念质量的系统关系。

---

## 422. ViTac-Tracing: Visual-Tactile Imitation Learning of Deformable Object Tracing

**arXiv ID:** 2603.18784 | [PDF](https://arxiv.org/pdf/2603.18784v1)

**作者:** Yongqiang Zhao `[一作]` (King's College London), Shan Luo `[通讯]` (King's College London)

**通讯引用:** 2742 | [OpenAlex ID](https://openalex.org/A5012646628)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于视觉‑触觉模态的模仿学习框架，能够使用单一统一策略完成一维和二维可变形物体的追踪任务。

**💡 创新点**

创新点包括：① 在局部层面引入“中心损失”鼓励触觉图像中心接触；② 在全局层面加入“任务损失”预测完成比例；③ 搭建了低成本多模态遥操作系统，实时提供视觉、触觉与触觉振动反馈。

**🔧 技术方法**

采用的技术包括：Transformer‑based action‑chunking imitation learning（act），视觉与触觉特征提取使用ResNet‑18和GelSight触觉相机，基于KL散度的正则化，数据增强与多模态融合。

**📊 数据集**

数据集主要由四种已见可变形物体（鞋带、绳索、毛巾、纱布）共100条示范构成，另外对未见绳子和餐巾进行测试，数据采集时同步记录视觉、触觉与关节/末端执行器位姿。

**📈 对比分析**

通过与去掉视觉/触觉、无中心/任务损失的基线模型、单物体与多物体联合训练的对比实验，已见物体的成功率达到80%，未见物体为65%，相较基线提升显著，证明所提方法在不同形状与尺寸下具有良好泛化能力。

**⚠️ 局限性**

局限性包括：对任务终点识别过于依赖视觉，导致在视觉遮挡下容易过度追踪；二维物体易滑脱，表现略逊；数据量有限，未充分解决仿真‑现实差距；模型主要针对线形与平面可变形物体，扩展到更复杂形状仍需研究。

---

## 423. Beyond TVLA: Anderson-Darling Leakage Assessment for Neural Network Side-Channel Leakage Detection

**arXiv ID:** 2603.18647 | [PDF](https://arxiv.org/pdf/2603.18647v1)

**作者:** Ján Mikulec `[一作]` (Slovak University of Technology), Xiaolu Hou `[通讯]` (Slovak University of Technology)

**通讯引用:** 623 | [OpenAlex ID](https://openalex.org/A5069420868)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ADLA框架，用于在受保护的神经网络实现中进行侧信道泄漏评估。

**💡 创新点**

创新点是采用两样本Anderson–Darling检验，能够检测完整分布差异而非仅均值变化，并给出了对应的阈值。

**🔧 技术方法**

使用了两样本AD检验、传统TVLA（Welch's t-test）、随机抖动和打乱等防御技术。

**📊 数据集**

以MNIST数据集训练的多层感知机（MLP）作为实验对象。

**📈 对比分析**

通过与TVLA在相同条件下比较，ADLA在少量trace（约<1000）即可发现泄漏，而TVLA需更多trace且检测灵敏度较低。

**⚠️ 局限性**

局限性包括仅在单一权重、单层实现上验证，未评估更复杂网络或更高阶攻击的可利用性。

---

## 424. Neural Galerkin Normalizing Flow for Transition Probability Density Functions of Diffusion Models

**arXiv ID:** 2603.18907 | [PDF](https://arxiv.org/pdf/2603.18907v1)

**作者:** Riccardo Saporiti `[一作]` (Ecole Polytechnique Federale de Lausanne), Fabio Nobile `[通讯]` (Ecole Polytechnique Federale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种Neural Galerkin Normalizing Flow (NGNF) 框架，用来近似扩散过程的转移概率密度函数（TPDF），通过求解 Fokker-Planck 方程并对 Dirac 初始分布做参数化。

**💡 创新点**

创新点在于将 Normalizing Flow 与 Neural Galerkin 方法结合，得到参数随时间演化的 ODE 系统，同时使用时间相关的参考分布并通过自适应采样保证在高维空间中残差评估的有效性；该框架天然满足正性、质量守恒及因果关系。

**🔧 技术方法**

主要技术包括：Real‑NVP 条件耦合层、GRU 网络实现缩放/平移函数、Fokker-Planck 残差的 Monte‑Carlo 估计、显式求解参数 ODE、时间相关参考高斯分布、离散 Runge‑Kutta 求解 θ(t)。

**📊 数据集**

使用二维 Beneš SDE（含旋转矩阵 R₂）作为实验数据集，生成的初始 Dirac 位置从标准正态分布采样，最终通过对比解析解验证模型精度。

**📈 对比分析**

与解析解、基于 Euler‑Maruyama 的数值解以及直接的 PINN 方案进行对比；在 2D 实验中，NGNF 在保留 TPDF 形态、保持 L² 误差受控、在长时间尺度下仍保持稳定，并且在多种初始条件下都能实现较高的精度，展示出比传统网格方法更优的高维可扩展性。

**⚠️ 局限性**

局限性包括：需要耗时的离线训练阶段，且对极高维问题仍可能面临样本量和计算量的挑战；自适应采样策略对训练稳定性要求高，若初始参考分布不恰当可能导致收敛缓慢；此外，当前实现主要验证于二维模型，实际复杂金融或流体动力学问题中的可推广性仍需进一步研究。

---

## 425. WASD: Locating Critical Neurons as Sufficient Conditions for Explaining and Controlling LLM Behavior

**arXiv ID:** 2603.18474 | [PDF](https://arxiv.org/pdf/2603.18474v1)

**作者:** Haonan Yu `[一作]`, Xin Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供ACL会议投稿的格式化说明和模板使用指南。

**💡 创新点**

创新点在于详细阐述如何使用提供的LaTeX模板和样例文件，帮助作者快速上手。

**🔧 技术方法**

使用了LaTeX语言、ACL格式化样式文件和样例文档。

**📊 数据集**

无数据集，内容纯属说明文件。

**📈 对比分析**

无实验或性能比较，本文仅作格式规范说明。

**⚠️ 局限性**

局限性：仅适用于ACL会议的稿件格式，未涉及实际研究内容。

---

## 426. Don't Vibe Code, Do Skele-Code: Interactive No-Code Notebooks for Subject Matter Experts to Build Lower-Cost Agentic Workflows

**arXiv ID:** 2603.18122 | [PDF](https://arxiv.org/pdf/2603.18122v1)

**作者:** Sriram Gopalakrishnan `[一作]` `[通讯]` (JP Morgan Chase and Co AI Research), Sriram Gopalakrishnan (JP Morgan Chase and Co AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出一种面向非技术用户的自然语言与图形化界面，用于构建 AI 代理支持的工作流，支持交互式逐步生成代码并运行。

**💡 创新点**

创新点在于将工作流结构用可视化图形和自然语言描述相结合，采用代码优先、仅在代码生成和错误恢复时调用代理，并通过“Markov blanket”式上下文工程减少 token 消耗。

**🔧 技术方法**

主要技术包括基于 Flask 的后端、JSON 结构的工作流描述、图形化前端（节点添加/连线）、自然语言提示与编码代理、程序化执行与调度。

**📊 数据集**

该工作未使用公开数据集，仅在原型中测试了自定义工作流实例。

**📈 对比分析**

与传统的聊天式“vibe coding”和完全代理式工作流对比，作者指出其在可预测性、成本和可视化方面更优，但尚未给出定量实验；计划在未来评估 token 成本、错误率和开发时间。

**⚠️ 局限性**

限制包括需要用户提前划分工作流步骤并显式指定依赖、缺少循环与条件表达式的直观支持、错误恢复受限于现有结构、以及对外部服务身份验证未自动化。

---

## 427. Beyond Accuracy: An Explainability-Driven Analysis of Harmful Content Detection

**arXiv ID:** 2603.18015 | [PDF](https://arxiv.org/pdf/2603.18015v1)

**作者:** Trishita Dhara `[一作]` (Upper Hand), Siddhesh Sheth `[通讯]` (Ace Rent a Car)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于RoBERTa的有害内容检测模型进行可解释性驱动分析，比较两种后置解释方法在正确预测与误判中的归因差异。

**💡 创新点**

系统性比较SHAP与Integrated Gradients对误判原因的揭示，发现二者在词汇焦点与上下文敏感度上的互补性，为人机协同审查提供透明度。

**🔧 技术方法**

采用RoBERTa-base分类器，并使用Shapley Additive Explanations（SHAP）与Integrated Gradients两种后置可解释技术进行token级归因。

**📊 数据集**

使用Civil Comments数据集（训练20k、验证4k、测试4k）作为实验数据。

**📈 对比分析**

通过定性案例分析比较两种解释方法，模型整体性能为准确率0.94、AUC0.94、F1≈0.62；两种方法在误判时呈现不同归因模式，揭示了词汇过度归因和隐性毒性识别难点。

**⚠️ 局限性**

限制包括：仅在单一二分类数据集上实验；解释方法对分词、扰动敏感；评估主要以定性为主，缺乏量化可信度指标。

---

## 428. CAMO: A Conditional Neural Solver for the Multi-objective Multiple Traveling Salesman Problem

**arXiv ID:** 2603.19074 | [PDF](https://arxiv.org/pdf/2603.19074v1)

**作者:** Fengxiaoxiao Li `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**通讯引用:** 1384 | [OpenAlex ID](https://openalex.org/A5069667034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为CAMO的条件神经求解器，用于解决多目标多旅行商问题（MOMTSP），能够在不同的目标权重、机器人数量和节点规模下生成 Pareto 前沿近似解；

**💡 创新点**

创新点在于将偏好向量与实例信息融合的条件编码器与协同解码器相结合，实现对多目标权重的可控性和多机器人协作的可扩展性，并通过在混合规模数据上采用 REINFORCE 训练提升跨规模泛化；

**🔧 技术方法**

核心技术包括条件注意力（Conditional Attention）与门控聚合（Gated Aggregation）模块的编码器设计、交替进行代理选择和节点选择的协同解码器、REINFORCE 强化学习目标以及实例增强（Instance Augmentation）提升的推理策略；

**📊 数据集**

实验使用随机生成的二维坐标实例（N20A2、N50A3、N100A4等）以及从 TSPLIB 转化的外域基准实例（如 ulysses22-A2、kroB200-A20 等）进行评估；

**📈 对比分析**

与传统 MOEAs（MOEA/D、NSGA-II/III、MOGLS）和改进的多代理神经求解器 MO-PARCO 进行比较，CAMO 在 Hypervolume、Gap 以及求解时间上均显著优于基线，在大规模或外域实例上依然保持高质量解；

**⚠️ 局限性**

局限性包括对目标不平衡的处理仍不充分，尚未针对极大规模问题（如数千节点）设计更高效的生成或分治策略，且依赖于 REINFORCE 训练，收敛速度与样本效率相对较低。

---

## 429. Consumer-to-Clinical Language Shifts in Ambient AI Draft Notes and Clinician-Finalized Documentation: A Multi-level Analysis

**arXiv ID:** 2603.18327 | [PDF](https://arxiv.org/pdf/2603.18327v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 430. RUBICONe: Wireless RAFT-Unified Behaviors for Intervehicular Cooperative Operations and Negotiations

**arXiv ID:** 2603.18595 | [PDF](https://arxiv.org/pdf/2603.18595v1)

**作者:** Zhenghua Hu `[一作]` (Tongji University), Hao Xu `[通讯]` (Tongji University)

**通讯引用:** 7683 | [OpenAlex ID](https://openalex.org/A5081732598)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在车联网环境下构建基于 RAFT 的分布式共识框架 RUBICONe，用无线 SDR 节点实现车道变更决策的安全与可靠性。

**💡 创新点**

创新点包括：① 将 RAFT 迁移到高速移动、噪声多变的 802.11p 无线网络，② 引入信号质量（SNR）加权的确定性分歧解决机制，③ 通过实验验证信号感知与共识决策的可靠性提升。

**🔧 技术方法**

技术手段：软件定义无线电（MicroPhase ANTSDR）、GNURadio 802.11p 堆栈、RAFT 协议改造、SNR 权重动态调度、信号感知与多节点协同。

**📊 数据集**

数据集：在六节点实验平台上，人工控制不同 SNR（4 dB、14 dB）和包损率（30%、70%）的环境，结合仿真软件模拟节点可靠性（p_node）范围 0.5–0.9。

**📈 对比分析**

比较方法：通过仿真与实验对比系统可靠性 P_sys 与节点数、个体可靠性、SNR 的关系。结果显示：随着节点数增加可靠性提升但递减收益显著；在高 SNR 下六节点系统可达 90% 以上可靠率；低 SNR 下可靠性显著下降，验证了信号加权策略的有效性。

**⚠️ 局限性**

局限性：① 仅验证了 crash-fault 容错，未覆盖 Byzantine 异常；② 只使用了 6 节点实验，缺乏大规模车队场景验证；③ SDR 硬件带来的时延与波动未完全模拟真实车载网络；④ 需要进一步在真实道路环境中测试和评估实时性与安全性。

---

## 431. Inst4DGS: Instance-Decomposed 4D Gaussian Splatting with Multi-Video Label Permutation Learning

**arXiv ID:** 2603.18402 | [PDF](https://arxiv.org/pdf/2603.18402v1)

**作者:** Yonghan Lee `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 39543 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建实例分解的4D高斯光栅化模型，实现多视角视频的长期一致身份跟踪与渲染。

**💡 创新点**

通过可微Sinkhorn层学习跨视频标签置换，实现一致身份标记，并引入实例级运动基底加速长周期轨迹优化。

**🔧 技术方法**

可微分高斯光栅化（3DGS/4DGS）、Sinkhorn归一化、双四元数运动混合、进阶激活与遮罩策略。

**📊 数据集**

Panoptic Studio 与 Neural3DV 两个多视角动态视频数据集。

**📈 对比分析**

与 Spacetime Gaussians、Dynamic3DGaussians、SA4D、TRASE 等基线对比，PSNR 提升至约 28.6/30.9，实例 mIoU 提升至 0.9129/0.9420，训练时间显著缩短（≈27–31 分钟）

**⚠️ 局限性**

依赖显式实例标注，需保证标注互斥且难以扩展到层次化语义空间；对快速运动或缺失标注的物体仍易出现误检。

---

## 432. Deceiving Flexibility: A Stealthy False Data Injection Model in Vehicle-to-Grid Coordination

**arXiv ID:** 2603.18424 | [PDF](https://arxiv.org/pdf/2603.18424v1)

**作者:** Kaan T. Gun `[一作]` (McGill University), Danial Jafarigiv `[通讯]` (Hydro-Québec)

**通讯引用:** 113 | [OpenAlex ID](https://openalex.org/A5044489791)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证了一种针对基于 eSSM 的车辆到电网（V2G）协同的隐蔽伪造数据注入攻击（FDIA）模型，该攻击仅通过劫持部分电动车的状态报告（SoC 与功率），在不控制物理充放电的前提下误导运营商的灵活性估计与控制决策。

**💡 创新点**

创新点在于：① 通过多阶段递归优化与最大似然分配，实现在仅限于信息层面（非物理层）对 eSSM 的聚合状态进行精确操控；② 结合特定的状态转移概率权重，保证注入数据既符合单车物理约束，又能保持整体聚合误差在可接受范围内，从而规避中心化监测的异常检测；③ 在仿真中展示该攻击可在不访问充电控制网络的情况下显著破坏电网频率调节。

**🔧 技术方法**

采用的技术包括：扩展状态空间模型（eSSM）、单车状态空间模型（IEVM）、多阶段优化与递归预测、最大似然估计（MLE）、混合整数线性规划（MILP）、Python 与 MATLAB 的离散事件仿真、两区域 AGC（自动发电控制）仿真。

**📊 数据集**

使用10,000台电动车的合成数据集，车辆参数（SoC、功率、容量、效率等）按照真实 EV 统计分布（正态与均匀分布）生成，覆盖不同充电模式与能耗特性。

**📈 对比分析**

在无攻击、攻击但无控制以及攻击+控制三种情形下进行对比。攻击后，聚合灵活性上限误差显著，MAPE 从 5.49% 提升至 520.28%；AGC 仿真中，频率偏差在受攻击时显著放大，导致功率分配失衡和频率失稳，表明攻击对系统性能的破坏极为严重。

**⚠️ 局限性**

局限性包括：攻击需要劫持一定比例（30%）的电动车；假设 eSSM 及其参数已知且无额外的实时异常检测；未考虑多种通信协议与加密层的安全性；仿真基于合成数据，缺乏现场实验验证。

---

## 433. LuMamba: Latent Unified Mamba for Electrode Topology-Invariant and Efficient EEG Modeling

**arXiv ID:** 2603.19100 | [PDF](https://arxiv.org/pdf/2603.19100v1)

**作者:** Danaé Broustail `[一作]` (ETH Zurich), Luca Benini `[通讯]` (University of Bologna)

**通讯引用:** 56961 | [OpenAlex ID](https://openalex.org/A5043408422)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出 LuMamba，一个融合拓扑无关编码与线性复杂度状态空间模型（SSM）的 EEG 基础模型，并在 21,600 小时无标签 EEG 上进行自监督预训练，随后在五个下游任务（异常检测、伪影识别、认知状态分类）上微调。

**💡 创新点**

创新点：① 将 LUNA 的跨电极注意力编码与 FEMBA 的双向 Mamba 块结合，实现对不同电极布局的拓扑无关建模；② 将 LeJEPA 目标与掩码重建目标联合，既保持重建的结构化特征，又通过分布正则化提升跨蒙太奇泛化；③ 在 SSM 框架下实现线性时间复杂度和显著的 FLOPs 减少（比 LaBraM 低 377×，支持 12× 更长序列）。

**🔧 技术方法**

技术：跨电极交叉注意力（LUNA）、双向 Mamba（FEMBA）、LeJEPA 预测与 SigReg 正则化、掩码重建、自监督预训练、Mamba 分类头、FFT 与时域卷积特征提取。

**📊 数据集**

数据集：预训练使用 TUEG 语料库（约 21,600 小时，10–20 系统 20–22 通道）；下游任务使用 TUAB（异常检测）、TUAR（伪影多类检测）、TUSL（癫痫/慢波分类）、TDBrain（帕金森检测 26 通道）和 APAVA（阿尔茨海默病检测 16 通道）。

**📈 对比分析**

对比方法：与 BENDR、EEGFormer、BIOT、EEG2Rep、LaBraM、LUNA 等现有自监督 EEG 基础模型对比；在 TUAB 上与 LaBraM 平级（80.99% 平衡准确率、0.8918 AUPR），在 APAVA 和 TDBrain 上实现 4%–20% 的 AUPR/AUROC 提升；相比 LaBraM，FLOPs 降低 377×，并可处理 12× 更长序列，显著提升计算效率。

**⚠️ 局限性**

局限性：在样本严重不平衡的任务（如 TUSL）上性能低于任务专门的基线，可能是由于强调泛化而非特定任务优化；此外，跨蒙太奇迁移仍需进一步验证在更大规模多机构数据集上的鲁棒性。

---

## 434. REST: Receding Horizon Explorative Steiner Tree for Zero-Shot Object-Goal Navigation

**arXiv ID:** 2603.18624 | [PDF](https://arxiv.org/pdf/2603.18624v1)

**作者:** Shuqi Xiao `[一作]` (University of Macau), Hui Kong `[通讯]` (University of Macau)

**通讯引用:** 5338 | [OpenAlex ID](https://openalex.org/A5036133825)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了REST框架，将零样本物体导航的选项空间从单点选择改为路径树结构，并通过链式思维LLM在文本化的空间叙事中选取最优路径。

**💡 创新点**

创新点包括：① next-best-path 概念，强调路径中的信息增益而非终点价值；② 使用欧几里得Steiner树构建可安全、信息丰富的导航决策树；③ 将树分支转换为文本叙事，利用LLM进行层次化、可解释的决策。

**🔧 技术方法**

采用的技术包括：RGB‑D 在线 3D 地图构建（UFOMap），采样式信息增益路径规划，欧几里得Steiner树优化，开放词汇 VLM（Qwen3‑VL）进行语义感知与文本化，LLM（Qwen3‑VL）进行链式推理，YOLO‑World 与 EdgeTAM 进行目标检测与实例分割。

**📊 数据集**

实验在 Habitat 仿真环境中使用 Gibson、HM3D、HSSD 三个室内数据集进行评估。

**📈 对比分析**

与 VLFM、ApexNAV、GAMap、SG‑Nav、VoroNav、TriHelper、ImagineNav、UniGoal、PanoNav 等最新零样本方法比较，REST 在成功率（SR）上位居前列，且在成功加权路径长度（SPL）上获得最优或次优成绩，证明了其效率‑成功的良好平衡。

**⚠️ 局限性**

局限性包括：LLM 推理对计算资源要求较高；实验仅在模拟环境中验证，实际机器人部署需要进一步评估；树深度和分支数的自动调节仍依赖经验参数；对动态或极端复杂场景的适应性尚待考察。

---

## 435. AndroTMem: From Interaction Trajectories to Anchored Memory in Long-Horizon GUI Agents

**arXiv ID:** 2603.18429 | [PDF](https://arxiv.org/pdf/2603.18429v1)

**作者:** Yibo Shi `[一作]` (Xi'an Jiaotong University), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1220 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 AndroTMem-Bench 诊断安卓长序列 GUI 代理的记忆瓶颈，并提出 Anchored State Memory。

**💡 创新点**

创新点在于使用稀疏的因果状态锚点结构化存储交互历史，显著提升长序列任务的记忆与推理能力。

**🔧 技术方法**

采用结构化锚点记忆（ASM）、因果依赖链接和任务完成率（TCR）等评价指标。

**📊 数据集**

使用 AndroTMem-Bench 数据集，包含 1,069 个任务、34,473 步、覆盖 50 个主流安卓应用。

**📈 对比分析**

与全序列重放和摘要基线对比，ASM 在 TCR 提升 5%–30.16%、AMS 提升 4.93%–24.66%，显著优于其他模型。

**⚠️ 局限性**

局限在于未涵盖跨会话长期持久状态、UI 动态变化等情境，需进一步扩展。

---

## 436. Uncovering Latent Phase Structures and Branching Logic in Locomotion Policies: A Case Study on HalfCheetah

**arXiv ID:** 2603.18084 | [PDF](https://arxiv.org/pdf/2603.18084v1)

**作者:** Daisuke Yasui `[一作]` (National Defense Academy of Japan), Hiroshi Sato `[通讯]` (National Defense Academy of Japan)

**通讯引用:** 24760 | [OpenAlex ID](https://openalex.org/A5071354099)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在HalfCheetah‑v5仿真跑步任务中，对已训练的TD3策略进行后置可解释分析，揭示其内部自发形成的周期性相位结构及分支决策；

**💡 创新点**

首次证明深度强化学习策略可在不加约束的情况下自主学习可解释的运动相位和分支控制逻辑；

**🔧 技术方法**

采用UMAP降维+时间约束层次聚类识别相位，再用Explainable Boosting Machine（EBM）逼近每个相位内的动作生成规则；

**📊 数据集**

使用MuJoCo的HalfCheetah‑v5跑步基准环境，收集5条1000步轨迹；

**📈 对比分析**

训练得到的策略平均回报约为9668，EBM在各相位的R²在0.6–0.9之间，说明可解释模型能较好复现策略行为；

**⚠️ 局限性**

仅针对周期性跑步任务，未验证在非周期或事件驱动任务中的适用性；EBM对高频冲击无法完美拟合，解释力受限于模型平滑性质；

---

## 437. Unmasking Algorithmic Bias in Predictive Policing: A GAN-Based Simulation Framework with Multi-City Temporal Analysis

**arXiv ID:** 2603.18987 | [PDF](https://arxiv.org/pdf/2603.18987v1)

**作者:** Pronob Kumar Barman `[一作]` (University of Maryland, Baltimore County), Pronoy Kumar Barman `[通讯]` (Jagannath University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5060895563)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个基于GAN的多城市预测警力部署模拟框架，用来审计算法警务中的种族偏差。

**💡 创新点**

创新点在于将GAN生成的空间巡逻点、纵向多城市偏差审计以及CTGAN条件生成式去偏方法结合，并通过社会经济回归揭示偏差结构。

**🔧 技术方法**

采用了生成对抗网络（GAN）生成巡逻位置、Noisy‑OR 检测模型、CTGAN 条件表格GAN 进行去偏、OLS 回归与相关性分析，以及DIR、人口平衡差距、基尼系数和偏差放大得分等公平度量。

**📊 数据集**

使用了巴尔的摩 2017–2019 年和芝加哥 2022 年的犯罪事件数据、美国人口普查 ACS 的族裔、收入与贫困率数据，以及基于 Pew 的公民报告概率。

**📈 对比分析**

通过比较“检测模式”与“报告模式”在 264 次模拟中的公平度量，发现检测模式会出现极端 DIR（0–15,000 级），而报告模式则相对稳定；CTGAN 去偏后方向性偏差会逆转；灵敏度分析显示警员数量对 DIR 影响最大。

**⚠️ 局限性**

局限包括基于社区比例的种族分配、每月 GAN 重新训练、Noisy‑OR 的独立性假设、未考虑犯罪位移与固定警力预算等导致的零和效应。

---

## 438. Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds

**arXiv ID:** 2603.18532 | [PDF](https://arxiv.org/pdf/2603.18532v1)

**作者:** Andrew Choi `[一作]` (Horizon Robotics), Wei Xu `[通讯]` (Horizon Robotics)

**通讯引用:** 10243 | [OpenAlex ID](https://openalex.org/A5100407852)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用3D生成模型自动构建数百个可交互的场景，并在这些场景上进行大规模并行强化学习微调预训练的视觉‑语言‑动作（VLA）策略，从而显著提升仿真与真实世界的操作成功率。

**💡 创新点**

首次将语言驱动的场景设计器与3D世界生成器结合，生成多样化数字孪生环境；证明场景多样性能显著提升零样本泛化；并提出PPOFlow将流匹配VLA转为高效单步高斯策略，兼顾性能与速度。

**🔧 技术方法**

语言驱动场景生成（GPT‑4o+EmbodiedGen）、流匹配VLA（π_0）、PPOFlow强化学习优化、GPU并行训练、数字孪生与域随机化、PD控制与重力补偿、仿真‑真实迁移技术。

**📊 数据集**

BridgeV2预训练数据集、自动生成的EmbodiedGen场景、手工设计的SimmerEnv三场景、真实世界12个测试场景，以及对应的语言指令数据。

**📈 对比分析**

与预训练模仿策略π_pre对比，仿真成功率从9.7%提升至79.8%，完成时间从10s降至8s；真实世界成功率从21.7%提升至75%，时间从11.5s降至10.2s；随着场景数量从1到100，零样本泛化率显著提升，证明多样性对性能的正向作用。

**⚠️ 局限性**

仅验证在pick‑and‑place任务，生成模型目前支持的对象和交互有限，未涵盖关节物体、工具使用或多阶段任务；训练成功率随场景数增加而下降，提示需动态调整批量大小；数字孪生质量对sim‑real差距仍有影响。

---

## 439. An FPGA-Based SoC Architecture with a RISC-V Controller for Energy-Efficient Temporal-Coding Spiking Neural Networks

**arXiv ID:** 2603.18054 | [PDF](https://arxiv.org/pdf/2603.18054v1)

**作者:** Mohammad Javad Sekonji `[一作]` (Shahid Bahonar University of Kerman), Mahdi Taheri `[通讯]` (Brandenburg Technical University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于FPGA的SoC，集成RISC‑V控制器与事件驱动的时序编码SNN核心，实现高能效、实时的神经形态推理。

**💡 创新点**

创新点在于使用二值化权重、位运算替代乘法、事件排序与跳过无信息事件的组合，从而实现高达16倍的内存压缩、低功耗以及高度可编程性。

**🔧 技术方法**

采用RISC‑V软核控制器、时间到第一脉冲编码器、突触电流计算器、事件驱动流水线、位运算加法/减法、可配置的整流发火神经元以及FPGA资源优化技术。

**📊 数据集**

在MNIST和Fashion‑MNIST两个图像分类数据集上进行评估。

**📈 对比分析**

与先前的Artix‑7实现相比，所提设计在MNIST上实现97%准确率、72 mW功耗、1.4 fps吞吐率；Fashion‑MNIST上88%准确率、158 mW功耗、35 fps；相较于前人实现，内存占用降低16×、DSP资源消耗为0，功耗显著下降。

**⚠️ 局限性**

当前限制包括仅支持离线训练（需通过RISC‑V加载权重）、网络规模受单芯片资源限制、对更深或更复杂拓扑的支持尚未实现，以及缺乏在线学习功能。

---

## 440. HISR: Hindsight Information Modulated Segmental Process Rewards For Multi-turn Agentic Reinforcement Learning

**arXiv ID:** 2603.18683 | [PDF](https://arxiv.org/pdf/2603.18683v1)

**作者:** Zhicong Lu `[一作]` (Aerospace Information Research Institute), Wei Feng `[通讯]` (School of Computer Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Hindsight Information Modulated Segmental Process Rewards (HISR) 方法，用于在大型语言模型（LLM）进行多轮代理强化学习时，通过后见信息调节分段过程奖励，以改善长时限决策任务中的信用分配问题。

**💡 创新点**

创新点：①将轨迹划分为对应子目标的分段，并在段级别分配过程奖励，避免过细粒度奖励；②利用后见信息（即已知轨迹结果后对动作的重要性估计）来计算动作/段的重要性比率，从而动态调节段级奖励；③将后见比率与过程奖励相乘并归一化，形成新的奖励信号；④将上述奖励与动作可执行性奖励融合，应用于 PPO 优化。

**🔧 技术方法**

技术细节：行为克隆 (Behavior Cloning) 对 LLM 进行监督微调；GPT‑4o 用于轨迹分段；构建 Segment‑level Process Reward Model (SPRM) 通过 MLP 预测每段对结果的贡献；构建 Hindsight 模型（masked LM）并与原政策模型计算序列似然比，得到动作重要性 z；利用 z 对 SPRM 的奖励进行加权调制；将调制后的奖励与可执行性奖励融合后输入 PPO（含 GAE）。

**📊 数据集**

数据集：Alfworld、Virtualhome（家庭情境）和 Webshop（网页导航）三大公开 agentic 基准。

**📈 对比分析**

实验对比：与 Prompt‑engineering、Behavior‑Cloning、PPO、GRPO、StepAgent、RAGEN、PRM4A、SPA 等三类基线进行对比。HISR 在三大基准上均取得最高分（在 SPA 基线之上显著提升），尤其在 Alfworld 的平均分上提升到 69.1 分，表明方法在信用分配和奖励稠密化方面显著有效。

**⚠️ 局限性**

限制：①需要 GPT‑4o 进行轨迹分段，增加额外步骤；②后见模型在 RL 训练阶段固定，无法适应数据分布变化；③原始段奖励的先验过强，导致后见信息调制效果被削弱；③分段方式仍依赖人工或外部模型，缺乏自动化。

---

## 441. CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization

**arXiv ID:** 2603.19121 | [PDF](https://arxiv.org/pdf/2603.19121v1)

**作者:** Weilin Chen `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**通讯引用:** 4295 | [OpenAlex ID](https://openalex.org/A5014628588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 CustomTex 框架，利用一组实例参考图像对未纹理化的 3D 室内场景进行实例级、高保真纹理生成。

**💡 创新点**

创新点：①双重蒸馏（语义级 + 像素级）结合实例交叉注意力，实现精准实例一致性并显著降低烘焙阴影；②将 Stable Diffusion 的 depth‑to‑image 与 SR 模型在 VSD 里联合优化；③采用多参考输入和特征级遮罩提高细节与局部一致性。

**🔧 技术方法**

技术：Variational Score Distillation (VSD)、Stable Diffusion depth‑to‑image 与超分辨率 diffusion、IP‑Adapter、LoRA 参数高效微调、多分辨率 hash grid 纹理表示、可微渲染、实例交叉注意力。

**📊 数据集**

数据集：使用公开的 3D‑FRONT 场景集合；参考图像来自真实照片或人工合成；训练采样 5,000 个球面视角，生成 4,096×4,096 纹理。

**📈 对比分析**

与 Paint3D、HY3D‑2.1、SceneTex‑IPA、TEXture 等基线进行定量和定性对比。评估指标包括 CLIP‑I、CLIP‑FID、Q‑Align IQA/IAA、CLIP‑T、IS；CustomTex 在所有指标上均优于对照组，并在用户研究中获得最高评分。Ablation 进一步证明像素蒸馏、特征遮罩和多参考对性能提升至关重要。

**⚠️ 局限性**

局限：训练时间长（≈48 h 4K 纹理），仅生成漫反射 albedo 纹理，未覆盖法线、粗糙度等其他材质。

---

## 442. Agentic Business Process Management: A Research Manifesto

**arXiv ID:** 2603.18916 | [PDF](https://arxiv.org/pdf/2603.18916v1)

**作者:** Diego Calvanese `[一作]` (Free University of Bozen-Bolzano), Barbara Weber `[通讯]` (University of St. Gallen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 Agentic Business Process Management (APM) 的概念，构建了其架构并列出了四大核心能力

**💡 创新点**

创新点在于将自主代理与业务流程管理结合，强调在框架化自治、可解释性、对话可操作性和自我修改四个维度实现组织目标

**🔧 技术方法**

运用了代理模型（BDI、i*、Tropos）、规范化框架（DECLARATIVE、过程规范）、大型语言模型、可解释 AI 技术以及工具调用机制

**📊 数据集**

未使用具体数据集，主要以理论框架与行业案例为依据

**📈 对比分析**

本文为理论宣言，没有实验比较；讨论了研究挑战与未来工作，但未给出性能指标

**⚠️ 局限性**

局限在于缺乏实现细节与实证验证，框架与规范仍需进一步规范化，跨组织协作与安全隐私等挑战待解决

---

## 443. CAFlow: Adaptive-Depth Single-Step Flow Matching for Efficient Histopathology Super-Resolution

**arXiv ID:** 2603.18513 | [PDF](https://arxiv.org/pdf/2603.18513v1)

**作者:** Elad Yoshai `[一作]` (Tel Aviv University), Natan T. Shaked `[通讯]` (Tel Aviv University)

**通讯引用:** 5221 | [OpenAlex ID](https://openalex.org/A5030858876)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可自适应深度的单步流匹配超分辨率框架CAFlow，适用于大规模数字病理图像。

**💡 创新点**

将像素未混洗空间的单步流匹配与卷积+窗口自注意力分级网络相结合，并用轻量级退出分类器实现质量驱动的深度路由，且训练中专门混合t=0样本以提升单步推理质量。

**🔧 技术方法**

采用Flow Matching、Pixel‑Unshuffle、FiLM调制、HybridFiLMBlock、单步流匹配、早期退出、时间步自适应采样、损失加权与一致性正则等技术。

**📊 数据集**

使用TCGA多器官（乳腺、肾、肺）256×256裁剪图像进行4×/8×上采样，测试集包含未见结肠组织。

**📈 对比分析**

与EDSR、SwinIR‑light/Medium、SRFormer‑light、SR3等基线对比，CAFlow在全深度时实现PSNR 31.84 dB、SSIM 0.8797；Adaptive路由仅损失0.12 dB即可节省33%计算/延迟，×8上仍优于轻量基线且与SwinIR‑Medium相近。

**⚠️ 局限性**

仅在病理图像上验证，退出点固定且缺乏对感知质量的显式监督，单图块推理未充分利用批处理。

---

## 444. Decidability of Quantum Modal Logic

**arXiv ID:** 2603.18368 | [PDF](https://arxiv.org/pdf/2603.18368v1)

**作者:** Kenji Tokuo `[一作]` `[通讯]` (Oita National Institute of Technology), Kenji Tokuo (Oita National Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了量子模态逻辑（QML）的可判定性，即存在算法可决定任意公式是否为定理。

**💡 创新点**

创新点在于将Harrop引理与QML的有限模型性质相结合，借助结构压缩（collapse）技术将无限模型转化为有限模型，从而实现可判定性的证明。

**🔧 技术方法**

采用了序列演算、Harrop引理、有限模型性质以及对量子模态结构的等价类归约等理论工具。

**📊 数据集**

本研究不涉及具体数据集，属于纯理论推导。

**📈 对比分析**

未进行实验性比较，结果以理论证明形式呈现；若将此方法与传统模态逻辑的可判定性证明方法对比，可见该方法在处理量子与模态结合的逻辑时提供了新的可判定性证明思路。

**⚠️ 局限性**

限制在于未给出具体复杂度上界，且仅证明了QML可判定性，未涉及更广泛的量子逻辑（如正交模态逻辑）的可判定性问题。

---

## 445. Multi-material Direct Ink Writing and Embroidery for Stretchable Wearable Sensors

**arXiv ID:** 2603.18354 | [PDF](https://arxiv.org/pdf/2603.18354v1)

**作者:** Lukas Cha `[一作]` (Institute of Biomedical Engineering), Liang He `[通讯]` (Institute of Biomedical Engineering)

**通讯引用:** 4773 | [OpenAlex ID](https://openalex.org/A5089412676)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

研发了一种将多材料直接墨水打印的弹性应变传感器与自动化刺绣相结合，实现传感器直接嵌入织物的工艺；

**💡 创新点**

创新点在于将打印的硅胶-碳油层与刺绣提供的机械固定与电气互连统一到单一步骤，既保持高弹性，又实现了可靠的电气连接；

**🔧 技术方法**

采用多材料直接墨水打印（DIW）制备硅胶/碳油层，随后使用工业级刺绣机进行机械固定与导电线缆拼接；

**📊 数据集**

无；

**📈 对比分析**

与文献中单一打印或单一刺绣制成的应变传感器相比，该混合传感器在120%拉伸下仍保持电路连续，60%拉伸内线性度R²=0.99，灵敏度GF≈31，滞后22.9%，循环漂移0.135%/周期，峰值漂移0.236%/周期；与基准技术相比，仍保持相当的线性与灵敏度，但在高应变区出现非线性；

**⚠️ 局限性**

主要局限包括：1）高滞后与循环漂移导致长期稳定性不足；2）非线性行为出现于>60%应变，限制大角度运动监测精度；3）对温度敏感，需环境控制；4）现场运动实验误差约17%，低于最先进织物集成传感器的性能；

---

## 446. Multimodal Task Interference: A Benchmark and Analysis of History-Target Mismatch in Multimodal LLMs

**arXiv ID:** 2603.18425 | [PDF](https://arxiv.org/pdf/2603.18425v1)

**作者:** Masayuki Kawarada `[一作]` (Artificial Intelligence Research Center AIST), Hiroya Takamura `[通讯]` (Artificial Intelligence Research Center AIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估多模态大语言模型在任务切换时的干扰，并量化不同模态、推理需求和答案格式切换对性能的影响。

**💡 创新点**

构建了三维轴（模态、推理、答案格式）评估基准，揭示跨模态切换的极端不对称性以及多重不匹配的累积干扰。

**🔧 技术方法**

采用教师强制的对话历史构造、统计学差异测算（Δ_switch）和多模态LLM（GPT‑4.1‑mini、Gemma‑3n、Qwen3‑VL、Pixtral）进行实验。

**📊 数据集**

使用六个数据集：Rotten Tomatoes（情感分类）、MMLU（多选问答）、TweetQA（开放式问答）、VQAv2（视觉问答）、OK‑VQA（视觉问答）和COCO Captions（图像字幕）。

**📈 对比分析**

对比同任务与切换任务历史下的性能差异，使用相对百分比改动作为统一度量；结果显示模态不匹配导致最高负面改动，推理不匹配表现鲁棒，答案格式不匹配受模型差异影响。

**⚠️ 局限性**

限制包括仅评估四个模型、短对话历史（N≤5）、仅文本与图像模态、未考虑模型自生成误差的连锁干扰。

---

## 447. NymeriaPlus: Enriching Nymeria Dataset with Additional Annotations and Data

**arXiv ID:** 2603.18496 | [PDF](https://arxiv.org/pdf/2603.18496v1)

**作者:** Daniel DeTone `[一作]` (Meta Reality Labs Research), Lingni Ma `[通讯]` (Meta Reality Labs Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了Nymeria+数据集，将改进的三维人体运动、闭集与开放集的3D/2D目标框注释、实例级三维物体重建及新增传感器数据统一到单一数据集，以支持第一人称视角下全面的人体动作与场景交互理解。

**💡 创新点**

创新点在于将大规模真实世界的 egocentric 多模态记录与精细的人体运动、丰富的语义目标框以及高分辨率物体几何信息统一到单一数据集，并提供可直接使用的MHR和SMPL运动标注。

**🔧 技术方法**

采用XSens惯性运动捕捉与Aria头戴设备同步的优化管道，利用Meta Momentum库对运动进行去噪和重标定，并通过三维重建技术生成闭合的物体几何；同时使用19类闭集与开放集标签进行目标框注释。

**📊 数据集**

基于原始Nymeria数据集（300+小时、256位参与者）进行扩展，并在此基础上添加了额外的室内场景扫描、腕带视频、麦克风音频等多模态数据。

**📈 对比分析**

通过与先前Nymeria及相关 egocentric 基准（如Ego4D、EgoDex等）的对比实验，验证了改进后的人体运动精度和目标检测在室内场景中的提升，尤其在动作识别和交互分析任务上取得更高的召回率和准确率。

**⚠️ 局限性**

局限性包括：仍主要覆盖室内环境，开放集目标框数量有限；物体重建在复杂遮挡下可能出现误差；以及对不同人类行为多样性与外部视角的泛化能力尚未全面评估。

---

## 448. Retrieval-Augmented LLMs for Security Incident Analysis

**arXiv ID:** 2603.18196 | [PDF](https://arxiv.org/pdf/2603.18196v1)

**作者:** Xavier Cadet `[一作]` (Dartmouth), Alina Oprea `[通讯]` (Northeastern)

**通讯引用:** 5918 | [OpenAlex ID](https://openalex.org/A5035574749)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于检索增强生成（RAG）的安全事件分析系统，通过针对性查询过滤日志、映射 MITRE ATT&CK 以及 LLM 语义推理来自动生成安全报告。

**💡 创新点**

创新点包括：① 用安全情报查询库对多源日志进行精确过滤并压缩为语义丰富的聚合块；② 设计跨事件语义推理机制，利用 RAG 实现多源证据关联与攻击链重建；③ 对八种 LLM（Claude Sonnet 4、DeepSeek V3、GPT 系列、Llama 3.1、Cisco Foundation‑Sec‑8B）进行系统评估，展示 DeepSeek 既能达到 100% 召回又成本最低；④ 公开完整实现、查询库与评估脚本，支持本地部署与多云提供商。

**🔧 技术方法**

使用技术包括：检索增强生成框架、Sentence‑Transformer（all‑mpnet‑base‑v2）嵌入、FAISS 向量检索、Security Onion（Suricata、Zeek、Windows 事件）日志集成、LLM 接口抽象（Anthropic、OpenAI、DeepSeek、Ollama、Cisco），以及多种评估指标（Precision/Recall、成本/时延）。

**📊 数据集**

数据集：
- Malware Traffic Analysis：Fake Authenticator、NetSupport RAT、Koi Stealer、IcedID 四个公开流量包；
- Active Directory 红队演练：涉及证书滥用、Kerberos、PSExec 等多阶段攻击日志。

**📈 对比分析**

比较方法：对每个模型在四个恶意流量场景下进行问题回答召回率评估，对 AD 场景进行攻击步骤检测精度/召回率、实时窗口分析；同时记录调用次数、token 量、成本（美元）和延迟（分钟）。结果显示：
- DeepSeek V3 与 Claude Sonnet 4 在所有恶意流量场景中均达 100% 召回，DeepSeek 成本仅为 Claude 的 1/15；
- 本地 Llama 3.1 在 95% 召回下无额外费用；
- GPT 系列在某些场景（Fake Authenticator）仅 60% 召回；
- RAG 预处理显著提升性能，直接处理原始日志的无检索基线只能捕获少量攻击线索。

**⚠️ 局限性**

局限性：
- GPT 模型存在用户/主机名混淆、命名歧义等推理错误，难以通过检索缓解；
- 评估仅覆盖四种恶意流量和单一 AD 演练，尚未验证在更广泛攻击种类与规模下的鲁棒性；
- 对实时窗口大小敏感，需针对不同攻击时序调优；
- 本地部署需 GPU 资源，部署与维护成本高；
- LLM 有时会给出不切实际的防御建议（如禁用 Administrator），需要人工复核。

---

## 449. Fundamental Limits of Neural Network Sparsification: Evidence from Catastrophic Interpretability Collapse

**arXiv ID:** 2603.18056 | [PDF](https://arxiv.org/pdf/2603.18056v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 450. Matryoshka Gaussian Splatting

**arXiv ID:** 2603.19234 | [PDF](https://arxiv.org/pdf/2603.19234v1)

**作者:** Zhilin Guo `[一作]` (University of Cambridge), Cristina Nader Vasconcelos. Cengiz Oztireli `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Matryoshka Gaussian Splatting (MGS) 框架，通过训练一个按不透明度排序的高斯原件序列，使单一模型在任意预算下都能保持连贯且高质量的渲染；

**💡 创新点**

其创新点在于结合随机预算训练与动态重排策略，形成可连续调节细节层次的嵌套高斯表示，既保留了全容量质量，又实现了平滑的速度‑质量曲线；

**🔧 技术方法**

采用了3D Gaussian Splatting 的可微光栅化渲染、基于不透明度的高斯重要性排序、随机预算采样、前缀+全局损失以及动态重排的训练技术；

**📊 数据集**

使用了四个公开基准数据集：MipNeRF 360、Tanks & Temples、Deep Blending 和 BungeeNeRF；

**📈 对比分析**

在与六个离散/连续LoD基线（如 H3DGS、Octree‑GS、MaskGaussian、FlexGaussian、CLoD‑GS、CLoD‑3DGS）对比实验中，MGS 在全容量下与原始 3DGS 相当甚至更优，在不同预算下提供更平滑且更高质量的速度‑质量曲线，AUC_fps 与 AUC_splats 均显著优于基线；

**⚠️ 局限性**

局限性包括对不透明度排序的依赖，极低预算下细节仍可能丢失；尚未在动态或实时流式场景中验证其鲁棒性和适应性。

---

## 451. R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

**arXiv ID:** 2603.18202 | [PDF](https://arxiv.org/pdf/2603.18202v1)

**作者:** Naoki Morihira `[一作]` (Honda R&D Co.), Tatsuya Harada `[通讯]` (RIKEN AIP)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种解码器和数据增强无关的模型基强化学习框架 R2‑Dreamer，用自监督冗余减少目标替代重建损失来学习视觉表示。

**💡 创新点**

核心创新是将 Barlow Twins 的冗余减少目标作为内部正则化，在 RSSM 中对图像嵌入和潜在状态对齐，从而消除对解码器和数据增强的依赖。

**🔧 技术方法**

技术上采用 DreamerV3 的 RSSM 结构、线性投影器、Barlow Twins 损失、强化学习中的 actor‑critic 与 λ‑return 训练。

**📊 数据集**

使用 DeepMind Control Suite、Meta‑World 以及自定义的 DMC‑Subtle 细粒度视觉任务数据集。

**📈 对比分析**

与 DreamerV3、DreamerPro、Dreamer‑InfoNCE、DrQ‑v2、TD‑MPC2 等基线在相同超参数下对比，R2‑Dreamer 在标准任务上与 DreamerV3 竞争，在 DMC‑Subtle 上显著优于所有基线，并比 DreamerV3 训练速度快 1.59 倍。

**⚠️ 局限性**

局限在于仍未评估动态背景下的鲁棒性，且在更高维度任务（如 Humanoid）上的性能尚待验证。

---

## 452. Interplay: Training Independent Simulators for Reference-Free Conversational Recommendation

**arXiv ID:** 2603.18573 | [PDF](https://arxiv.org/pdf/2603.18573v1)

**作者:** Jerome Ramos `[一作]` (University College London), Aldo Lipani `[通讯]` (University College London)

**通讯引用:** 1704 | [OpenAlex ID](https://openalex.org/A5058519912)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种参考自由（reference‑free）对话模拟框架，训练两个独立的LLM——一个模拟用户、一个模拟推荐系统——在对话中实时交互且不预先知道目标物品；

**💡 创新点**

核心创新包括：①去除oracle知识，只使用目标属性而非目标物品；②采用结构化输出和角色专属损失掩码实现独立训练；③通过两模型协作实现自然探索、灵活推荐，避免脚本化对话；

**🔧 技术方法**

技术手段包括：对Llama‑3.1 8B和Qwen‑3 8B进行全量微调；使用结构化响应模板和角色损失掩码；评估指标有BERTScore、Dist‑4、Recall@1、MatchScore等；

**📊 数据集**

使用的数据集为ReDial（过滤后52.9k对话，测试集2069条），以及MovieLens‑32M用于生成物品嵌入以计算MatchScore；

**📈 对比分析**

与Llama‑3.1 70B、Qwen‑3 32B和UniCRS等基线进行对比，用户模拟器的成功率达到93‑95%（远超70B的36%），推荐器RecSim‑8B在Recall@1和MatchScore上均超过32B基线，并且保持更高的多样性；人类评估显示对话质量与基线相当，且在用户控制度上更优；

**⚠️ 局限性**

局限性包括：仅在电影推荐域验证；对话仍受限于结构化任务导向，未涵盖开放式聊天或多意图对话；MatchScore的可靠性和与真实用户偏好的关联尚未完全验证；

---

## 453. Fast and Generalizable NeRF Architecture Selection for Satellite Scene Reconstruction

**arXiv ID:** 2603.18306 | [PDF](https://arxiv.org/pdf/2603.18306v1)

**作者:** Devjyoti Chakraborty `[一作]` (University of Georgia), Deepak Mishra `[通讯]` (University of Georgia)

**通讯引用:** 5883 | [OpenAlex ID](https://openalex.org/A5048372527)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于场景一致性的预测框架，能够在不训练任何NeRF模型的情况下，在几秒钟内估计出不同NeRF架构在卫星图像上的重建质量，并根据预测结果选择最合适的模型。

**💡 创新点**

创新点在于①通过SHAP分析揭示多视角一致性比模型结构更能决定NeRF在卫星图像上的重建质量；②提出轻量级的几何和光度描述子，并构造线性回归预测器，实现对NeRF性能的快速、可解释预测；③将预测与离线硬件成本数据库结合，实现硬件感知的架构选择，显著降低功耗和延迟。

**🔧 技术方法**

技术包括：NeRF与SatNeRF/S-NeRF基础模型；几何与光度描述子（逆PSNR、视角余弦相似度、视差密度、光度方差等）；SHAP特征重要性分析；线性性能估计器；Neural Architecture Search（NAS）对比；离线硬件性能测评；Jetson Orin边缘平台部署实验。

**📊 数据集**

使用了DFc2019数据集，包括亚利桑那州Jacksonville（JAX）和奥马哈（OMA）的卫星图像场景，共73个场景（33 JAX + 40 OMA），并在9个测试场景上评估预测误差。

**📈 对比分析**

与传统NAS（在A6000 GPU上耗时约9‑10小时）相比，预测框架在Tesla‑T4上每个场景仅需约30秒，速度提升超过1000×；预测误差MAE<1dB，许多场景误差<0.5dB；在Jetson Orin平台上，结合预测与硬件成本可实现约26%功耗下降、43%延迟降低，同时PSNR损失仅约0.79dB。

**⚠️ 局限性**

局限性：对极端视角几何或光照条件变化大的场景预测误差会增大；依赖一定量的标注数据进行训练，稀疏数据下表现仍有限；目前仅针对S-NeRF/SatNeRF等卫星专用NeRF模型，需进一步验证对其他NeRF变体的泛化能力。

---

## 454. Can LLM generate interesting mathematical research problems?

**arXiv ID:** 2603.18813 | [PDF](https://arxiv.org/pdf/2603.18813v1)

**作者:** Xiaoyang Chen `[一作]` (Tongji University), Xiang Jiang `[通讯]` (Tongji University)

**通讯引用:** 27045 | [OpenAlex ID](https://openalex.org/A5028552398)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了一个名为DeepMath‑Generate的生成代理，能够自动生成并评估未知的数学研究问题，最终在微分几何领域产生了665个研究问题，并通过人工验证发现其中许多问题在专家中尚未被提出且具有独特的研究价值。

**💡 创新点**

创新点在于将大型语言模型与生成式评估相结合，提出了“好问题”三大评价标准（新概念、方法与对象），并设计了双向迭代的生成-评估框架，用以系统地挖掘高质量未知问题。

**🔧 技术方法**

主要使用了GPT‑5 / GPT‑5.3模型，并通过精心设计的系统提示（Generator Prompt & Evaluator Prompt）实现问题生成与自动评估；后续还计划引入强化学习提升创造力。

**📊 数据集**

数据集方面，作者选取了200个不同的研究方向（如谐波映射、曲率与拓扑等），每个方向生成5个问题，形成了约1000个候选问题的人工评估集合；此外使用了前期构建的数学创造力基准作为评估参照。

**📈 对比分析**

方法上采用人工专家验证来衡量问题的新颖性与价值，结果表明大量问题在专家中未知，显示生成方法具备高新颖性；然而缺乏量化性能指标，难以与现有研究直接比较。

**⚠️ 局限性**

局限性包括生成的“好问题”难以匹敌经典如Poincaré猜想般的深度与美感；提示语和知识点的设计仍需进一步细化，导致部分问题缺乏真正突破性；同时缺乏系统的量化评估和对比实验。

---

## 455. SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction

**arXiv ID:** 2603.18774 | [PDF](https://arxiv.org/pdf/2603.18774v1)

**作者:** Vsevolod Skorokhodov `[一作]` (Schindler EPFL Lab), Malcolm Mielle `[通讯]` (Schindler EPFL Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对RGB+热像相机姿态估计和3D重建，提出一种轻量级的微调方法SEAR，能够将预训练的视觉几何Transformer适配为跨模态输入。

**💡 创新点**

通过在AA模块中加入LoRA适配器、可学习的热像相机token以及特殊的批处理策略，几乎不改变原模型参数、无显著推理延迟地实现了跨模态一致性。

**🔧 技术方法**

采用LoRA参数高效微调、DINOv2 tokenizer、学习热像相机token、交替注意力（AA）模块及跨模态批量策略。

**📊 数据集**

使用约15k对RGB-热像图的公开数据集（ThermoScenes、ThermalNeRF、ThermalGaussian、ThermalMix、Radar Forest），并公开新建的9场景RGB-热像序列数据集。

**📈 对比分析**

与传统SfM、深度学习模型以及混合特征匹配方法（COLMAP、DUSt3R、MASt3R、VGGT、MA_ELoFTR、MINIMA_ROMA）对比，在姿态AUC@30、相机注册率、点云PCC/Chamfer等指标上，SEAR实现约70% AUC@30、10×以上的推理速度提升和显著优于基线的点云质量。

**⚠️ 局限性**

对热像与RGB比例不平衡时性能下降，热像特征质量低时仍可能出现误差，且目前仅验证了RGB与热像两模态，未扩展到更多传感器；训练仍需一定量的配对数据。

---

## 456. Understanding the Relationship Between Firms' AI Technology Innovation and Consumer Complaints

**arXiv ID:** 2603.18025 | [PDF](https://arxiv.org/pdf/2603.18025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 457. Benchmarking Visual Feature Representations for LiDAR-Inertial-Visual Odometry Under Challenging Conditions

**arXiv ID:** 2603.18589 | [PDF](https://arxiv.org/pdf/2603.18589v1)

**作者:** Eunseon Choi `[一作]` (Pohang University of Science and Technology), Soohee Han `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 4735 | [OpenAlex ID](https://openalex.org/A5069368669)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在FAST-LIVO2基础上，提出一种混合稀疏-直接视觉里程计框架，通过特征提取-匹配进行图像补丁筛选，然后再使用光度误差进行状态更新。

**💡 创新点**

创新点在于：①将传统稀疏-直接方法与多种特征提取-匹配组合（ORB+HD、SuperPoint+SuperGlue、SuperPoint+LightGlue、XFeat+MNN）融合到同一框架，形成统一可量化的对比；②在视觉挑战环境（低照度、过曝、高视差）下对特征提取与匹配进行系统性基准评测；③提出轻量级XFeat+MNN在无GPU环境下仍能实现高精度、实时性能的可行性。

**🔧 技术方法**

使用的技术包括：LiDAR-IMU-视觉多传感器融合、稀疏-直接视觉里程计（FAST-LIVO2）、ORB、SuperPoint、XFeat特征提取器、SuperGlue、LightGlue、互近邻（MNN）匹配器、ESIKF状态估计、GPU加速的ONNX推理。

**📊 数据集**

实验数据集包含：Newer College、SubT-MRS、MARS-LVIG，涵盖室内外、低照度隧道、地面飞行等多样化环境。

**📈 对比分析**

通过在同一LIVO流程下对比RMSE、计算时延、CPU/GPU使用率和内存占用等指标。结果显示：XFeat+MNN在三组数据中均获得最低RMSE，且无需GPU即可保持≈10 Hz；SuperPoint+LightGlue在保持相近精度的同时显著降低GPU内存占用；ORB+HD在视角变化有限的场景表现更稳健；纯稀疏-直接基线在光照/高视差下漂移显著。

**⚠️ 局限性**

局限性包括：①学习型特征在极端视角变化下仍可能退化；②对学习模型的推理时延和GPU内存需求较高，限制了在资源极端受限的嵌入式平台应用；③实验未对每个数据集进行超参数调优，可能未达到各方法的最优性能；④混合框架增加了实现复杂度和运行时负担。

---

## 458. Sketch2Topo: Using Hand-Drawn Inputs for Diffusion-Based Topology Optimization

**arXiv ID:** 2603.18960 | [PDF](https://arxiv.org/pdf/2603.18960v1)

**作者:** Shuyue Feng `[一作]` (University of Tokyo), Yoshihiro Kawahara `[通讯]` (University of Tokyo)

**通讯引用:** 6392 | [OpenAlex ID](https://openalex.org/A5106658710)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 Sketch2Topo，一个基于扩散模型的交互式拓扑优化工具，允许用户通过手绘草图定义几何和物理约束，并支持区域局部优化。

**💡 创新点**

创新点在于将手绘草图与扩散模型相结合，实现直观的几何与约束输入；通过图像到图像生成与局部修复实现局部拓扑优化；并通过色彩编码的画笔直接映射物理约束，降低认知负担。

**🔧 技术方法**

使用扩散模型（TopoDiff）实现图像到图像生成和局部修复；OpenCV 提取色彩和坐标信息；利用 ControlNet/ControlNet‑Adapter 等技术对模型进行条件控制。

**📊 数据集**

未构建新的数据集，直接使用 TopoDiff 预训练模型；实验中对照了标准的 2D 拓扑优化基准（如 SIMP）生成的有限元分析结果。

**📈 对比分析**

通过与传统有限元分析(SIMP)以及无遮罩/遮罩两种扩散模型生成结果进行对比，计算最小合规度与体积分数。结果表明 Sketch2Topo 产生的结构在合规度上略高但仍在可接受范围，体积分数略高于基准；实验显示模型更倾向于最小化合规度。

**⚠️ 局限性**

局限性：目前仅支持二维设计；手绘输入仅作为引导，最终可制造设计仍需人工重建；缺乏对三维复杂组件的直接支持。

---

## 459. Lightweight Adaptation for LLM-based Technical Service Agent: Latent Logic Augmentation and Robust Noise Reduction

**arXiv ID:** 2603.18074 | [PDF](https://arxiv.org/pdf/2603.18074v1)

**作者:** Yi Yu `[一作]` (Fudan University), Wenlian Lu `[通讯]` (Fudan University)

**通讯引用:** 8204 | [OpenAlex ID](https://openalex.org/A5030103251)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在复杂技术服务领域构建了一套轻量化的LLM适配框架，包含：Latent Logic Augmentation（规划感知轨迹与决策推理增强）、Robust Noise Reduction（双过滤生成多GroundTruth数据集）以及Hybrid Reward Mechanism（LLM判定+轻量级Reranker融合的奖励机制），实现高效、稳定的模型微调与强化学习。

**💡 创新点**

创新点：① 通过规划感知轨迹(PATM)与决策推理增强(DRA)显式化潜在决策逻辑，克服“表面模仿”导致的能力退化；② 双过滤（Consistency Judge + Utility Judge）自动构建多GroundTruth数据集，减少监督噪声并捕获语义多样性；③ Hybrid Reward融合LLM Judge与Reranker，既保持奖励质量，又将计算成本降低约30%，实现训练效率与性能的双赢。

**🔧 技术方法**

使用技术包括：SFT + DRA + PATM（强化决策与规划能力）；RL（DAPO算法）+ Hybrid Reward Mechanism（LLM Judge + Reranker）；多GT数据构建 pipeline；基于 Qwen3-4B（基础模型）和 Qwen3-32B（Judge）实现。

**📊 数据集**

数据集：内部云服务技术支持对话数据（约10k条查询），并在训练/验证/测试集中通过双过滤扩展为多GroundTruth集合（训练集约10k→10k+，测试集从1k扩至约2k）。

**📈 对比分析**

实验对比：与单纯 SFT、仅 LLm Judge 奖励、仅 Reranker 奖励等基线相比，SFT-Mix+DRA 提升 Multi-ECS 至 0.337，RL 阶段 Hybrid Reward 达到 0.429，升级为 Multi-GT 后进一步提升至 0.441；同时奖励计算时间相对 Judge-only 降低约30%，并显著抑制策略熵坍塌，验证了多GT和 Hybrid Reward 的有效性。

**⚠️ 局限性**

局限性：① 仅在私有数据集上验证，缺乏公开复现；② 生成与判定均依赖强大LLM，成本较高；③ Hybrid Reward 的阈值与权重为静态，未能随训练动态调整，可能影响在分布漂移过程中的奖励稳定性。

---

## 460. Reconstruction Matters: Learning Geometry-Aligned BEV Representation through 3D Gaussian Splatting

**arXiv ID:** 2603.19193 | [PDF](https://arxiv.org/pdf/2603.19193v1)

**作者:** Yiren Lu `[一作]` (Bosch Research North America), Yu Yin `[通讯]` (Case Western Reserve University)

**通讯引用:** 9574 | [OpenAlex ID](https://openalex.org/A5101655848)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于3D高斯点射的显式3D重建框架Splat2BEV，用以生成几何对齐的BEV特征，服务于自动驾驶下的多任务感知。

**💡 创新点**

核心创新在于将3D高斯点射作为中间显式表示，将多视角图像先重建3D场景，再投影到BEV空间，使BEV特征既语义丰富又几何精准，并通过视角感知模型DINO进行特征蒸馏。

**🔧 技术方法**

使用3D高斯点射(3DGS)、多视角深度预测、视角变换、正交投影到BEV、BEV编码器与分割头，并结合三阶段训练与联合微调。

**📊 数据集**

在nuScenes和Argoverse1两个公开自动驾驶数据集上进行评估。

**📈 对比分析**

与现有BEV分割/检测方法对比，Splat2BEV在车辆、行人和车道分割任务上分别提升约11%、8%和21.4% IoU，整体实现SOTA表现。

**⚠️ 局限性**

局限性包括对高斯重建的依赖导致训练开销较大，且对BEV相机高度敏感；目前仅在BEV分割任务验证，其他下游任务仍需进一步评估。

---

## 461. Context Bootstrapped Reinforcement Learning

**arXiv ID:** 2603.18953 | [PDF](https://arxiv.org/pdf/2603.18953v1)

**作者:** Saaket Agashe `[一作]`, Xin Eric Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种 Context Bootstrapped Reinforcement Learning (CBRL) 方法，通过在 RLVR 训练期间随机注入少量示例来缓解探索效率低的问题。

**💡 创新点**

在 RLVR 中引入可逐渐衰减的示例注入策略，使模型在保持自足性前后都能内部化推理模式，而不依赖示例。

**🔧 技术方法**

在 GRPO 或 RLOO 等策略梯度框架上叠加随机上下文注入与线性退火注入概率，兼容任何策略梯度方法。

**📊 数据集**

在 Reasoning Gym 的五个推理任务（ARC‑1D、Matrix Manipulation、Word Sorting、Spell Backward、Puzzle‑24）以及 Q 编程语言的 678 题 Leetcode‑style 代码生成数据集上进行实验。

**📈 对比分析**

与基线 GRPO/RLOO 及少量示例提示进行对比，CBRL 在所有 10 个模型–任务组合中提升 1.3%–22.3%，在 Q 语言中将 Pass@1 从 5% 提升至 26%，且早期奖励曲线显示更高探索效率。

**⚠️ 局限性**

对示例库的质量与匹配度敏感；注入概率退火仅为线性，未自适应；在更长步推理或任务难度自适应上仍需改进。

---

## 462. WeNLEX: Weakly Supervised Natural Language Explanations for Multilabel Chest X-ray Classification

**arXiv ID:** 2603.18752 | [PDF](https://arxiv.org/pdf/2603.18752v1)

**作者:** Isabel Rio-Torto `[一作]` (INESC TEC), Luís F. Teixeira `[通讯]` (INESC TEC)

**通讯引用:** 1566 | [OpenAlex ID](https://openalex.org/A5075704593)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了弱监督框架 WeNLEX，用于生成多标签胸部 X 光分类的自然语言解释。

**💡 创新点**

创新点在于通过图像与文本分布匹配和特征重建实现既忠实又符合临床可解释性，同时仅需少量（5 条）真实 NLE 作为参考。

**🔧 技术方法**

技术方案包括 Transformer Encoder‑Decoder、PEFT（Multi‑Modal LLaMA‑Adapter）、WGAN‑GP 或 MMD 分布匹配、Text‑Embedding‑to‑Image 模块、CheXagent 视觉定位等。

**📊 数据集**

使用基于 MIMIC‑CXR 的 MIMIC‑NLE 数据集，并构建了少量（5 条）每诊断标签的 NLE 数据库进行训练。

**📈 对比分析**

与全监督基线相比，WeNLEX 在 faithfulness、simulatability、diversity、plausibility 等多项指标上均表现更佳，且在 in‑model 设置下可将分类器 AUC 提升 2.21%。

**⚠️ 局限性**

局限包括仍需依赖有限的 NLE 标注数据库、对非医学文本的适配性受限、以及不同视觉模型迁移时的泛化效果尚未充分验证。

---

## 463. RARE disease detection from Capsule Endoscopic Videos based on Vision Transformers

**arXiv ID:** 2603.18045 | [PDF](https://arxiv.org/pdf/2603.18045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 464. ROFT-VINS: Robust Feature Tracking-based Visual-Inertial State Estimation for Harsh Environment

**arXiv ID:** 2603.18746 | [PDF](https://arxiv.org/pdf/2603.18746v1)

**作者:** Sanghyun Park `[一作]` (POSTECH), Soohee Han `[通讯]` (POSTECH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在单目视觉惯性里程计中引入深度学习光流算法RAFT，实现更鲁棒的特征跟踪，提升在纹理不足或光照变化剧烈环境下的定位精度。

**💡 创新点**

创新点在于将全像素高精度光流网络RAFT与VINS-Mono的稀疏跟踪框架相结合，并通过基于KD‑Tree的局部运动一致性检测实现异常特征剔除，显著降低误跟踪导致的漂移。

**🔧 技术方法**

使用的核心技术包括RAFT光流网络、深度特征提取与全像素成本体积、递归更新模块、VINS-Mono滑动窗口优化、KD‑Tree邻域剔除和图像去畸变。

**📊 数据集**

实验数据集为室内飞行场景的EuRoC MAV以及室内外多光照环境的UMA Visual‑Inertial数据集。

**📈 对比分析**

通过与VINS‑Fusion（单目‑惯性）在EuRoC MAV上的相对位姿误差（RPE）和UMA数据集上的轨迹稳定性进行对比，发现普通环境下性能相当，在纹理不足或光照突变的“hard”序列中本方法显著降低漂移，轨迹更靠近真实路径。

**⚠️ 局限性**

主要局限包括RAFT推理时较高的计算开销，导致实时性能受限；UMA数据集缺乏地面真实轨迹，评估只能定性；此外在极端光照或快速运动时仍可能出现误匹配，需要进一步优化。

---

## 465. MCP-38: A Comprehensive Threat Taxonomy for Model Context Protocol Systems (v1.0)

**arXiv ID:** 2603.18063 | [PDF](https://arxiv.org/pdf/2603.18063v1)

**作者:** Yi Ting Shen `[一作]`, Alex Leung `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过四阶段系统方法，构建了 MCP-38——一个面向模型上下文协议（MCP）的 38 类威胁分类体系，覆盖了 MCP 的语义攻击面和传统框架缺失的风险。

**💡 创新点**

创新点在于首次为 MCP 提出专属威胁分类，系统性映射到 STRIDE、OWASP LLM Top 10、OWASP Agentic Top 10 等现有框架，解决了语义层面攻击缺口，并将理论与实测相结合。

**🔧 技术方法**

技术包括：协议拆解、跨框架映射、真实事件合成、类别归纳；利用 JSON‑RPC 2.0 语义分析、文本注入检测和工具元数据审计；在此基础上构建了交叉映射表。

**📊 数据集**

使用的数据集主要为公开的 CVE、GitHub 事件、MCP 安全基准 (MSB)、Guo 等研究的 PoC、Smithery/Glama 服务器注册库、以及 2025‑2026 年公开的安全通报。

**📈 对比分析**

对比方法：将每个威胁与 STRIDE、OWASP LLM、OWASP Agentic、MITRE ATT&CK 进行映射，验证覆盖率；通过案例复现验证每类威胁的实际可行性；在安全实验室环境中重现部分 PoC，评估攻击成功率与影响范围。

**⚠️ 局限性**

局限性包括：只针对当前 MCP 规范，未来协议变更需更新；依赖公开事件，尚未涵盖所有潜在攻击；缺乏量化性能指标，更多实测验证仍需展开；分类仍可能与某些新型语义攻击交叉重叠，需要持续迭代。

---

## 466. Evaluating LLM-Generated Lessons from the Language Learning Students' Perspective: A Short Case Study on Duolingo

**arXiv ID:** 2603.18873 | [PDF](https://arxiv.org/pdf/2603.18873v1)

**作者:** Carlos Rafael Catalan `[一作]` (Samsung R&D Institute Philippines), Marie Antoinette Patalagsa `[通讯]` (Samsung R&D Institute Philippines)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对五名跨国公司软件工程师的问卷调查，评估了Duolingo基于大语言模型生成的课程中常规场景与工作场景的学习效果，并提出了针对专业场景的个性化课程生成思路。

**💡 创新点**

创新点在于首次从用户视角系统探讨常规场景与专业场景对语言流利度的差异，并提出在LLM生成课程中加入专业领域定制化场景的设计方向。

**🔧 技术方法**

主要使用大语言模型（如Duolingo内部的LLM）生成课程，并利用问卷（Qualtrics）进行数据收集与分析。

**📊 数据集**

数据集为五位在菲律宾一家跨国公司就职的软件工程师所完成的问卷，涵盖学习频率、场景出现频率、体验感受及改进建议。

**📈 对比分析**

研究未采用对照实验或性能指标，而是通过定性分析和自我报告来比较常规场景与工作场景的学习影响，发现常规场景更适合初学者，工作场景更有助于专业流利度的提升。

**⚠️ 局限性**

局限性包括样本量极小（仅5人）、仅来自同一公司、缺乏长期跟踪与对照组实验，无法普适验证结论的稳健性。

---

## 467. BVSIMC: Bayesian Variable Selection-Guided Inductive Matrix Completion for Improved and Interpretable Drug Discovery

**arXiv ID:** 2603.18957 | [PDF](https://arxiv.org/pdf/2603.18957v1)

**作者:** Sijian Fan `[一作]` (University of South Carolina), Ray Bai `[通讯]` (George Mason University)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5037288421)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于贝叶斯变量选择的诱导矩阵完成功能BVSIMC，用于药物发现中的二元交互预测。

**💡 创新点**

创新点在于将spike-and-slab组Lasso先验引入IMC，实现对侧信息的自适应稀疏化和旋转不变的特征选择，从而提升预测准确性与可解释性。

**🔧 技术方法**

采用贝叶斯逻辑矩阵分解、SSGL先验、坐标上升与加速近端梯度更新等技术实现模型估计。

**📊 数据集**

使用了结核菌株药物耐药数据（9,684 SNP+33药物功能组）和Cdataset药物重定位数据（1,899药物侧信息+797疾病表型特征）等真实数据集。

**📈 对比分析**

与传统IMC、SGIMC、DRIMC和NRLMF比较，BVSIMC在模拟、耐药预测和重定位任务中均取得最高AUC（如耐药AUC>0.9，重定位AUC显著高于0.5）。

**⚠️ 局限性**

局限在于仅适用于二元数据，线性假设可能过于简化，且未针对计数或非线性关系扩展。

---

## 468. Tendon-Actuated Robots with a Tapered, Flexible Polymer Backbone: Design, Fabrication, and Modeling

**arXiv ID:** 2603.19124 | [PDF](https://arxiv.org/pdf/2603.19124v1)

**作者:** Harald Minde Hansen `[一作]` (European Organization for Nuclear Research), Mario di Castro `[通讯]` (European Organization for Nuclear Research)

**通讯引用:** 1283 | [OpenAlex ID](https://openalex.org/A5063186323)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计、制造并验证了一种3D打印的多尖塔形TPU背骨肌腱驱动连续机器人。

**💡 创新点**

通过引入跨段截面几何变化与对应刚度梯度，扩展了Cosserat杆理论，实现了可逆的形状设计工具，并实现低成本快速组装平台。

**🔧 技术方法**

采用FDM 3D打印、TPU材料、肌腱驱动、集成电子底座、Vicon运动捕捉、Cosserat杆模型与线搜索校准以及iLogic CAD脚本。

**📊 数据集**

收集了单根肌腱弯曲/展开实验数据，约1235个样本记录肌腱张力与Vicon位姿的对应关系。

**📈 对比分析**

将模型预测的机器人形状与Vicon测量结果对齐并误差分析，使用SVD变换和线搜索校准Young模量，最终在测试集上实现约1–2 cm的平均误差。

**⚠️ 局限性**

受限于TPU材料的非线性与打印过程变异、张力传感器分辨率低以及死区，导致形状预测误差随位移增大。

---

## 469. Benchmarking PDF Parsers on Table Extraction with LLM-based Semantic Evaluation

**arXiv ID:** 2603.18652 | [PDF](https://arxiv.org/pdf/2603.18652v1)

**作者:** Pius Horn `[一作]` (Offenburg University), Janis Keuper `[通讯]` (University of Mannheim)

**通讯引用:** 1334 | [OpenAlex ID](https://openalex.org/A5083785142)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于合成PDF的表格抽取基准框架，并采用LLM判定器对表格抽取质量进行语义评估。

**💡 创新点**

创新点在于①使用真实arXiv表格嵌入合成PDF获得精确LaTeX基准；②引入LLM-as-a-judge实现对结构与内容的语义匹配；③提供可复现的21款解析器排行榜。

**🔧 技术方法**

主要技术包括合成PDF生成（LaTeX编译、随机布局）、LLM匹配管道（Gemini-3-Flash-Preview）、LLM评估（Claude Opus 4.6等），以及传统指标（TEDS、GriTS、SCORE）。

**📊 数据集**

数据集为从arXiv抽取的真实表格，生成100页合成PDF共451张表格，另收集1,554条人工评估。

**📈 对比分析**

与人工评价比对，LLM评估与人类评分的Pearson相关系数达0.93，远高于规则基准（约0.68）。在21款解析器中，Gemini 3系列与LightOnOCR等模型取得最高分（9.5/10）。

**⚠️ 局限性**

局限包括：合成PDF缺乏真实扫描或非科学表格；仅使用arXiv数据，可能偏向科研格式；LLM判定器成本虽低但仍需付费；LLM评估不完美，仍有误判。

---

## 470. TARo: Token-level Adaptive Routing for LLM Test-time Alignment

**arXiv ID:** 2603.18411 | [PDF](https://arxiv.org/pdf/2603.18411v1)

**作者:** Arushi Rai `[一作]` (University of Pittsburgh), Zhuokai Zhao `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Token-level Adaptive Routing（TAR），在推理时通过学习的路由器动态混合冻结的基础模型和奖励模型，以提升 LLM 的推理能力。

**💡 创新点**

创新点在于用步进式数学推理奖励模型与可学习的 token‑级路由器结合，避免了固定权重调参，且能够无额外训练迁移到更大模型。

**🔧 技术方法**

采用了奖励模型训练（基于 Math‑StepDPO‑10K 步级偏好）、token‑级路由器（轻量级 MLP）、与生成时的动态加权解码。

**📊 数据集**

主要数据集包括 Math‑StepDPO‑10K 用于奖励模型训练，MATH500、MedXpertQA、AlpacaEval 用于评测推理与指令跟随性能。

**📈 对比分析**

与基线模型、奖励模型以及 GenARM 等现有测试时对齐方法比较，TAR 在 MATH500 上提升高达 +22.4%（基线）和 +8.4%（GenARM），在 MedXpertQA 与 AlpacaEval 上亦取得显著提升。

**⚠️ 局限性**

局限性包括奖励模型在非数学领域的表达能力受限，路由器在极大规模模型上仍需验证；且在某些任务中对奖励模型依赖仍不如基线模型。

---

## 471. A Passive Elastic-Folding Mechanism for Stackable Airdrop Sensors

**arXiv ID:** 2603.18861 | [PDF](https://arxiv.org/pdf/2603.18861v1)

**作者:** Damyon Kim `[一作]` (University of Tokyo), Takuya Sasatani `[通讯]` (University of Tokyo)

**通讯引用:** 457 | [OpenAlex ID](https://openalex.org/A5027448650)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

提出并验证了一种被动弹性折叠铰链机制，将平面可堆叠传感器在空中释放后转化为三维滑翔机，实现低成本、无动力的空气分散部署；

**💡 创新点**

创新点在于利用三层层压（FR4+热收缩聚烯烃+聚酰亚胺）一次加热即完成自折叠，同时保持弹性回弹，可通过几何模型精确预测折叠角度；

**🔧 技术方法**

主要技术包括：层压成型、热收缩加工、PCB集成、几何折叠角模型、拉伸力-位移实验、现场飞行试验以及基于MATLAB的风模型仿真；

**📊 数据集**

使用的实验数据包括铰链折叠角度测量、拉伸力-位移曲线、恢复率实验结果以及现场飞行中的IMU、LoRa传输数据；

**📈 对比分析**

通过与模型预测值的比较验证了折叠角度的高精度（标准差≈4°）、力-位移曲线与理论曲线高度吻合（R²≈0.97），现场飞行显示能成功完成拉升并实时传输数据，仿真表明单点部署可在10公里范围内实现广域散布；

**⚠️ 局限性**

局限性包括：折叠恢复需要数十秒，角度误差仍存，未在低压低温高空环境下全面验证，且缺乏对飞行阶段的气动特性细致评估。

---

## 472. NeuroGame Transformer: Gibbs-Inspired Attention Driven by Game Theory and Statistical Physics

**arXiv ID:** 2603.18761 | [PDF](https://arxiv.org/pdf/2603.18761v1)

**作者:** Djamel Bouchaffra `[一作]` (Paris-Saclay University), Bilal Faye `[通讯]` (Sorbonne Paris Nord University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 NeuroGame Transformer，将 Transformer 的注意力机制重新表述为合作博弈和统计物理中的 Ising 模型，以实现更高阶语义依赖的建模。

**💡 创新点**

创新点在于将 Shapley 值与 Banzhaf 指数结合生成外部场，采用重要性加权蒙特卡罗估计计算博弈特征函数，并通过均值场自洽方程求解 Ising Hamiltonian，从而获得可解释且可扩展的注意力权重。

**🔧 技术方法**

使用的技术包括合作博弈理论（Shapley、Banzhaf）、Gibbs 采样的蒙特卡罗估计、Ising 统计物理模型、均值场近似、以及标准 Transformer 结构。

**📊 数据集**

在自然语言推断任务上使用 SNLI 和 MNLI‑matched 两个数据集进行评估。

**📈 对比分析**

与 BERT‑Base、RoBERTa‑Base、ALBERT‑Base 等基线模型对比，NGT 在 SNLI 上取得 86.6% 的准确率（仅比 BERT‑Base 低 2.3%），在 MNLI‑matched 上达到 79.0%（与 ALBERT‑Base 相当），显示出在参数开销几乎不变的情况下具有竞争力的性能。

**⚠️ 局限性**

局限性包括：蒙特卡罗采样和均值场迭代使计算相对传统软最大化更为耗时；模型在大规模预训练环境下尚未充分验证；温度参数的设置对收敛与性能影响较大。

---

## 473. Access Controlled Website Interaction for Agentic AI with Delegated Critical Tasks

**arXiv ID:** 2603.18197 | [PDF](https://arxiv.org/pdf/2603.18197v1)

**作者:** Sunyoung Kim `[一作]` (Arizona State University), Hokeun Kim `[通讯]` (Arizona State University)

**通讯引用:** 891 | [OpenAlex ID](https://openalex.org/A5069501250)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于细粒度访问控制的网页交互框架，使代理式人工智能（Agentic AI）能够安全地代表用户执行关键任务。

**💡 创新点**

在现有分布式授权工具（SST）中扩展Auth数据库和协议，新增委托目标类型和期望所有者组，实现对AI代理的安全委托；并设计了支持用户自定义访问策略的网页接口。

**🔧 技术方法**

采用SST的Auth作为分布式密钥分发中心，使用加密会话密钥、HMAC验证、Python Flask后端、React前端；代理端利用OpenAI gpt‑oss‑20b模型进行推理。

**📊 数据集**

使用OpenAI gpt‑oss‑20b模型进行代理推理，并在ASU Sol超算环境中对模拟交易场景进行实验；未使用公开真实数据集，仅采用模拟交易数据。

**📈 对比分析**

通过四项评估（身份验证、细粒度访问控制、未授权访问处理、会话管理）测得有效代理在身份验证、授权访问和会话失效方面均达到100%成功率；未授权请求始终失败；端到端延迟在127–140秒之间，未授权请求延迟更低（约98秒）。

**⚠️ 局限性**

局限性包括：缺乏形式化验证；仅在本地实验环境下测试，未考虑真实网络延迟和大规模部署；缺乏对真实电子商务平台的实测。

---

## 474. Improving Joint Audio-Video Generation with Cross-Modal Context Learning

**arXiv ID:** 2603.18600 | [PDF](https://arxiv.org/pdf/2603.18600v1)

**作者:** Bingqi Ma `[一作]` (Vivix Group Limited), Yu Liu `[通讯]` (Vivix Group Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Cross‑Modal Context Learning (CCL)，一种改进的双流 Transformer 架构，用于高质量、时间同步的音视频联合生成。

**💡 创新点**

创新点包括（1）Temporally Aligned RoPE & Partitioning (TARP) 解决音频/视频采样率差异的时间对齐；（2）Learnable Context Tokens (LCT) 与 Dynamic Context Routing (DCR) 在跨模态注意力中提供背景稳定锚点并动态路由；（3）Unconditional Context Guidance (UCG) 在推理时利用 LCT 作为无条件提示，提升训练‑推理一致性并缓解文本‑跨模态控制冲突。

**🔧 技术方法**

技术细节：双流 Transformer + 预训练的音频/视频扩散模型；旋转位置编码 RoPE；跨模态注意力与自注意力；流匹配 (flow‑matching) 训练目标；多任务训练与动态路由；多模态分类器无监督指导 (classifier‑free guidance)。

**📊 数据集**

使用了百万级音视频对齐数据集，包含 OpenHumanVid、内部收集数据；音频预训练数据包括 Wavcaps 与 VGGSound；视频流预训练来自 Wan2.1-14B。

**📈 对比分析**

与 Ovi、LTX‑2、MOVA 等最新开源模型比较，CCL 在音质 (WER、PQ、CU)、唇同步 (Sync‑C/D) 与音视对齐 (DeSync/IB) 上均取得或领先的指标，同时参数规模仅 4M，显著低于对手。

**⚠️ 局限性**

局限性：仍需大量配对音视频数据；模块复杂度提升，推理时需多步 CFG；对不同场景（如极端动作或长时序）仍有待进一步验证。

---

## 475. Interpretable Prostate Cancer Detection using a Small Cohort of MRI Images

**arXiv ID:** 2603.18460 | [PDF](https://arxiv.org/pdf/2603.18460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 476. LLM-Augmented Computational Phenotyping of Long Covid

**arXiv ID:** 2603.18115 | [PDF](https://arxiv.org/pdf/2603.18115v1)

**作者:** Jing Wang `[一作]`, Jeremy C Weiss `[通讯]` (National Institutes of Health)

**通讯引用:** 2283 | [OpenAlex ID](https://openalex.org/A5072774346)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

通过构建名为 Grace Cycle 的迭代框架，利用大语言模型在长 COVID 患者纵向数据中自动生成假设、提取证据并细化特征，最终发现三种临床子表型：Protected、Responder 与 Refractory。

**💡 创新点**

创新之处在于将 LLM 作为交互式推理引擎，完成假设生成、证据搜索与特征优化的闭环流程，并结合统计检验实现对结果的严谨验证，且该框架可跨疾病迁移。

**🔧 技术方法**

采用大语言模型（LLM）进行链式思考和 pairwise 比较，配合特征选择、统计检验（Kruskal‑Wallis、线性混合模型、Latent Class Trajectory Modeling）和 bootstrap Jaccard 稳定性评估。

**📊 数据集**

使用 NIH RECOVER Initiative 成人队列共 13,511 名长 COVID 参与者的纵向数据，包括疫苗接种记录、可穿戴设备测量（呼吸率、心率、睡眠等）以及基于 44 项症状的自评 PASC 分数。

**📈 对比分析**

与传统线性混合模型和 StepMix 隐类轨迹模型对比，Grace Cycle 能稳定得到三类子表型并通过 Kruskal‑Wallis 检验显示显著差异（H=4215.2，p<0.001）；在剂量-反应分析中亦展现更细致且一致的差异，性能优于传统方法且子表型稳定性 Jaccard>0.97。

**⚠️ 局限性**

局限性包括：仍需人工干预以修改假设；观测性研究导致因果关系难以确认；PASC 分数基于自报，易受记忆与报告偏差；模型在其他疾病中的表现依赖数据质量与可用特征的丰富度。

---

## 477. Rigorous Error Certification for Neural PDE Solvers: From Empirical Residuals to Solution Guarantees

**arXiv ID:** 2603.19165 | [PDF](https://arxiv.org/pdf/2603.19165v1)

**作者:** Amartya Mukherjee `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**通讯引用:** 58053 | [OpenAlex ID](https://openalex.org/A5100450180)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了物理信息神经网络（PINN）的误差泛化理论，并在多种常见的常微分方程和偏微分方程上实现了残差到解空间误差的可计算转换与正式验证。

**💡 创新点**

创新点在于首次引入紧致性假设与算子稳定性，将残差上界通过形式化验证工具转化为解空间的显式误差界，提供了可计算且可证明的泛化误差保证。

**🔧 技术方法**

主要技术包括紧致性分析、算子稳定性估计、残差上界的形式化验证工具（如 dReal、autoLiRPA、∂-CROWN）以及 PINN/ELM 网络的训练和误差传播。

**📊 数据集**

实验使用了经典模拟数据，涵盖 Van der Pol 震荡方程、二维 Poisson 方程、热方程、波方程和 Burgers 方程，参考解来自解析解或高精度数值求解。

**📈 对比分析**

通过与参考解的误差对比，验证了证书误差界始终大于真实误差且并未显著过度，实验在各 PDE 上的残差与泛化误差均得到可验证的上界，耗时与验证工具相符。

**⚠️ 局限性**

局限性包括对紧致性和算子稳定性假设的依赖，理论对现代网络结构的紧致性保障有限；稳定常数和残差上界往往保守，导致误差界偏大；未覆盖随机/算子学习或高维 PDE 的情况。

---

## 478. Implicit Grading Bias in Large Language Models: How Writing Style Affects Automated Assessment Across Math, Programming, and Essay Tasks

**arXiv ID:** 2603.18765 | [PDF](https://arxiv.org/pdf/2603.18765v1)

**作者:** Rudra Jadhav `[一作]` (Savitribai Phule Pune University), Sonalika Shaw `[通讯]` (Dr. D. Y. Patil School of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大规模语言模型在不同写作风格（语法错误、非正式语言、非母语短语）下对学生答题评分的隐式偏差，构造三学科（数学、编程、论文写作）的受控数据集并用两款开源LLM评估成绩。

**💡 创新点**

创新点在于：1）设计系统化的表面风格扰动框架，以内容保持不变的方式精准测量写作风格偏差；2）证明提示层面去偏指令无法根除偏差；3）揭示主观度梯度——主观评估任务的偏差远高于客观任务，并与非正式语言关联。

**🔧 技术方法**

使用Meta的LLaMA 3.3 70B和Alibaba的Qwen 2.5 72B两款指令调优大型语言模型，按固定温度的JSON评分提示进行评估；随后采用配对t检验、Cohen’s d、Pearson相关、MAE等统计方法进行偏差分析。

**📊 数据集**

构建180条受控学生答题（60个问题×3学科×3扰动），每条答题都有对应的基线答案。数据涵盖语法错误、非正式语言、非母语短语三种表面扰动，且保持答案内容完全一致。

**📈 对比分析**

通过比较基线与扰动版本的得分差异，统计显著性与效应大小。结果显示论文写作任务中所有扰动均显著偏差（p<0.05，d≥0.8），最高效应达到4.25；数学与编程任务偏差微弱。LLaMA整体偏差更大（最大1.90分），Qwen偏差更普遍（多达44.4%条件显著）。

**⚠️ 局限性**

局限性包括：①使用合成数据，未覆盖真实学生多样化写作；②仅评估两款开源模型，缺少商业模型对比；③问题量有限，可能缺乏对细微偏差的检验；④评分标准由研究团队单独制定，可能引入主观偏差；⑤扰动方式统一，未按不同语言水平调节，未覆盖更广泛语言与文化背景。

---

## 479. MOSS-TTS Technical Report

**arXiv ID:** 2603.18090 | [PDF](https://arxiv.org/pdf/2603.18090v1)

**作者:** Yitian Gong `[一作]`, Xipeng Qiu `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了MOSS‑TTS开放式语音生成基础模型，采用高质量音频离散化、全程自回归预测与海量多语种预训练；

**💡 创新点**

创新点在于将可变比特率RVQ离散化与纯Transformer AR架构统一实现，去除了多模块 cascades，提出了两种可扩展的 AR 模式（Delay‑Pattern 与 Local‑Transformer），并通过端到端的多阶段数据管线实现可控性与长上下文生成；

**🔧 技术方法**

使用了 MOSS‑Audio‑Tokenizer（causal Transformer + RVQ + 语义对齐）、自监督的音频‑文本对齐、端到端的多任务训练、Token‑级时长与发音控制以及大规模多语言预训练和长上下文扩展；

**📊 数据集**

使用了约数百万小时的多语种音频数据（播客、音频书、广播、电影、评论等），通过多阶段清洗、ASR、LLM 校正与合成数据补充，覆盖英语、中文及多种低资源语言；

**📈 对比分析**

通过与多种开源音频离散化器（StableCodec、XCodec、Mimi 等）对比，MOSS‑Audio‑Tokenizer 在低、中、高比特率上均优于同类；在 TTS 评测中，MOSS‑TTS 与 MOSS‑TTS‑Local‑Transformer 在零射声克隆、多语言表现、时长控制、发音控制与超长生成上均达到或优于现有开源模型，尤其在 Speaker Similarity 上表现突出；

**⚠️ 局限性**

局限性包括在日语、韩语等语言的声纹保持仍不理想；超长生成时声纹漂移显著；对极短句子或极短文本输入的鲁棒性有限；缺乏足够真实对齐数据可能影响对齐精度。

---

## 480. ALIGN: Adversarial Learning for Generalizable Speech Neuroprosthesis

**arXiv ID:** 2603.18299 | [PDF](https://arxiv.org/pdf/2603.18299v1)

**作者:** Zhanqi Zhang `[一作]` (University of California San Diego), Gal Mishne `[通讯]` (University of California San Diego)

**通讯引用:** 785 | [OpenAlex ID](https://openalex.org/A5075832074)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

针对跨会话的脑机接口语音解码，提出了ALIGN框架，用对抗学习实现会话不变特征，提升在未标记会话上的解码性能。

**💡 创新点**

创新点在于：①将多源域对抗学习与中间层正则化相结合，显著抑制会话特异信息；②引入时间拉伸增广（TSA），增强对时序漂移的鲁棒性；③在无标签目标会话中实现高效的对抗适配。

**🔧 技术方法**

技术手段包括：Transformer/GRU解码器、CTC损失、对抗域分类器、梯度反转层、时间拉伸增广、以及后置语言模型和测试时适配（TTA）

**📊 数据集**

使用了两套ALS患者的多会话微电极阵列数据集：T12（24天）和T15（45天），共计约20,000句子。

**📈 对比分析**

与GRU/Transformer基线以及TTA方法进行对比实验，ALIGN在T12、T15的PER/WER上分别提升约9–15%，尤其在极端会话间隔（如12–4–7）下，WRE从约60%降至约46%，显著优于基线。

**⚠️ 局限性**

局限性包括：仍需依赖语言模型生成伪标签，极端会话漂移下可能失效；对神经特征被消除/保留的机制解释不足；以及对不同解码器结构的通用性需进一步验证。

---

## 481. Quotient Geometry and Persistence-Stable Metrics for Swarm Configurations

**arXiv ID:** 2603.18041 | [PDF](https://arxiv.org/pdf/2603.18041v1)

**作者:** Mark M. Bailey `[一作]` (National Intelligence University), Mark M. Bailey `[通讯]` (National Intelligence University)

**通讯引用:** 628 | [OpenAlex ID](https://openalex.org/A5065702930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个对称性不变、无标签的离散群体配置度量空间 𝒮_n(M,G) 并定义了一个基于最坏情况分配的距离 d_M,G；使用此距离证明了它是 Gromov–Hausdorff 的结构化上界，并且在该空间上可以得到持久化同调（Vietoris–Rips）的 1‑Lipschitz 性；在相位圆模型下给出了条件逆定理，证明在半圆支撑且满足间隙标记（gap‑labeling）条件时，持久化签名可逆。

**💡 创新点**

创新点在于：①将群体配置视为轨道空间并给出物理意义的最坏情况匹配距离 d_M,G；②将该距离与 Gromov–Hausdorff 距离相连接，得到可计算且可稳定的持久化同调签名；③证明了 𝒮_n(M,G) 在常见模型下是紧、完备、测地的，揭示了碰撞与对称性导致的层状奇点；④在相位圆模型上给出了可逆的“margin”定理，弥补了持久化同调的一般不完整性。

**🔧 技术方法**

使用的技术包括：Gromov–Hausdorff 与 Gromov–Wasserstein 理论、最坏情况分配（bottleneck / 𝓁^∞ 赋值）、Vietoris–Rips 持久化与 Bottleneck 距离、最优匹配（Hungarian 算法）以及轨道空间与匀称性分解。

**📊 数据集**

主要以理论证明为主，示例数据集为合成的球面（𝕊²）和多维环面（𝕋^m）配置；未使用真实实验数据集。

**📈 对比分析**

比较方法基于 d_M,G 与持久化同调的 Bottleneck 距离，理论上满足 d_B(Φ_k([x]),Φ_k([y])) ≤ d_M,G([x],[y])；实验评估未给出，重点在于理论稳定性保证；在相位圆模型的半圆支撑下，逆定理给出 2(n‑1) 的 Lipschitz 常数。

**⚠️ 局限性**

局限性包括：①持久化签名不具全局可逆性，存在对称性不匹配与信息压缩导致的非注射；②逆定理仅在特定的半圆支撑与间隙标记假设下成立；③计算上 d_M,G 仍需求解组合匹配与连续对齐，可能面临 NP‑hard 性；④对动态成员变更（加入/离队）等情况的处理尚未覆盖。

---

## 482. Position: Spectral GNNs Are Neither Spectral Nor Superior for Node Classification

**arXiv ID:** 2603.19091 | [PDF](https://arxiv.org/pdf/2603.19091v1)

**作者:** Qin Jiang `[一作]` (Heriot-Watt University), Wei Pang `[通讯]` (Heriot-Watt University)

**通讯引用:** 3742 | [OpenAlex ID](https://openalex.org/A5081644845)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过理论分析指出谱图神经网络（Spectral GNN）在节点分类任务中并不真正捕获图谱信息，且其“谱”机制仅相当于传统消息传递网络（MPNN），并对 MagNet、HoloNet 等代表性谱模型的实现错误进行了揭示。

**💡 创新点**

创新点在于：① 明确阐释图拉普拉斯特征向量并非经典 Fourier 基底；② 证明在任何 n 节点图上，(n‑1) 次多项式即可精确插值任意谱响应，因而多项式近似并非逼近；③ 从跳数（hop）域角度重新解释低通/高通滤波，表明其来源于消息传递而非谱变换；④ 通过实现审计揭示 MagNet、HoloNet 的优秀性能源自实现缺陷而非真正的谱优势。

**🔧 技术方法**

使用的技术主要是：谱图理论、Vandermonde 插值理论、线性时不变（LTI）系统理论、跳数域分析、代码审计与实现对比。

**📊 数据集**

本文主要为理论位置论文，并未在任何公开数据集上进行实验验证；所讨论的数据集仅为通用图结构的理论假设。

**📈 对比分析**

对比方法：通过理论推导与实现审计说明若按谱算法一致实现，MagNet、HoloNet 的性能显著下降；相较于传统 MPNN，谱模型在正确实现后并不具备明显优势，低通/高通行为可通过简单消息传递解释；因此本文认为所报道的优秀性能往往是实现误差的结果。

**⚠️ 局限性**

局限性：① 仅聚焦节点分类任务，对图分类、链路预测等其他任务的结论尚不确定；② 缺乏大规模实验验证，理论结论需要在更多数据集上进一步检验；③ 可能低估某些谱方法在特定结构图（如周期图）中确实具备的频域优势。

---

## 483. LRConv-NeRV: Low Rank Convolution for Efficient Neural Video Compression

**arXiv ID:** 2603.18261 | [PDF](https://arxiv.org/pdf/2603.18261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 484. Behavioral Fingerprints for LLM Endpoint Stability and Identity

**arXiv ID:** 2603.19022 | [PDF](https://arxiv.org/pdf/2603.19022v1)

**作者:** Jonah Leshin `[一作]` (Project VAIL), Daniel Kang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1852 | [OpenAlex ID](https://openalex.org/A5072348548)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出对AI端点进行连续行为稳定性监测与可视化，构建Stability Monitor与Stability Arena。

**💡 创新点**

创新点在于结合能量距离置换检验与e‑value序列化聚合，实现无白盒依赖的实时变化检测，并支持跨供应商同模型比较。

**🔧 技术方法**

主要技术包括能量距离统计、置换检验、e‑value连续证据聚合、文本嵌入、黑盒API调用、web可视化。

**📊 数据集**

使用实际端点产生的指纹样本（每个指纹800次请求）进行实验；在控制实验中人工引入5种变化，生产环境中监测Kimi‑K2‑0905‑Instruct多供应商表现。

**📈 对比分析**

通过p值置换检验与阈值化e‑value进行事件报警；实验表明大多数变化可在下一次指纹生成时检测到，仅温度微调需要18次；跨供应商可用能量距离热力图与偏差比进行定量比较。

**⚠️ 局限性**

受限于基础设施导致的持续随机性难以区分模型与执行级别变化、需固定prompt集、对提示多样性与稀疏性支持有限。

---

## 485. Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning

**arXiv ID:** 2603.19078 | [PDF](https://arxiv.org/pdf/2603.19078v1)

**作者:** Sangwoo Shin `[一作]` (University of Wisconsin), Josiah Hanna `[通讯]` (University of Wisconsin)

**通讯引用:** 1196 | [OpenAlex ID](https://openalex.org/A5008014974)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种新的图神经网络架构（ABD-Net），将正向动力学的计算结构嵌入到机器人控制的策略网络中；

**💡 创新点**

创新点在于将Articulated Body Algorithm（ABA）的惯性传播机制通过可学习的参数化形式映射到信息传递过程，实现了物理驱动的结构先验；

**🔧 技术方法**

主要技术包括基于树形的向下信息传播、可学习的惯性和运动子空间参数、正交性约束以及与PPO的集成；

**📊 数据集**

在多种仿真环境（Genesis与SAPIEN的Unitree G1/G2、Humanoid、Hopper等）以及真实Unitree G1和Go2机器人上进行实验；

**📈 对比分析**

与Transformer（BoT、SWAT、Rodrigues）、传统GNN和MLP基线相比，ABD-Net在样本效率、最终性能和对动力学变化的鲁棒性方面均表现更优，尤其在复杂形态和动态任务中提升显著；

**⚠️ 局限性**

局限性包括：推理时需顺序叶到根计算导致的推理延迟略高；目前仅处理关节位置/速度等低维观测，无法直接处理图像等高维感知输入；

---

## 486. CausalRM: Causal-Theoretic Reward Modeling for RLHF from Observational User Feedbacks

**arXiv ID:** 2603.18736 | [PDF](https://arxiv.org/pdf/2603.18736v1)

**作者:** Hao Wang `[一作]` (Peking University), Zhouchen Lin `[通讯]` (Peking University)

**通讯引用:** 26227 | [OpenAlex ID](https://openalex.org/A5016399094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CausalRM 框架，利用因果推断和噪声感知损失在观测反馈数据上训练无偏奖励模型，解决用户注释误差和偏好偏差。

**💡 创新点**

创新点：①将注释噪声校正与偏好偏差消除统一为一个无偏学习目标；②通过噪声感知 surrogate loss 与逆概率加权/双稳健重加权实现对两种噪声的同步校正；③提供理论证明并在多种 LLM 与数据集上验证优越性。

**🔧 技术方法**

技术：因果推断（IPS、双稳健 DR）、噪声感知 surrogate loss、基于 LLM 的奖励模型（FsfairX-LLaMA3-RM-v0.1+MLP head）、Propensity 估计网络、Adam 训练。

**📊 数据集**

数据集：HelpSteer、UltraFeedback、PKU‑SafeRLHF（公开偏好数据），并使用半合成 PKU‑SafeRLHF、以及合成实验来评估无偏性。

**📈 对比分析**

对比方法：Debias 类（IPS、MTIPS、CVIB、DR、MTDR、SDR）、Denoise 类（F‑correction、Co‑Teaching、CoDis、LabelWave、Robust DivideMix、ILDE）以及 Naive；CausalRM‑IPS/DR 在 MSE、MAE、R² 上均位列榜首，R² 最高可达 0.78；下游 RLHF 安全基准 WildGuardMix 上提升 49% 以上。

**⚠️ 局限性**

局限：仅聚焦奖励模型训练目标，未探索混合专家等更深层模型；未结合实验（人工）反馈数据，未来可与少量高质量标签协同提升。

---

## 487. DarkDriving: A Real-World Day and Night Aligned Dataset for Autonomous Driving in the Dark Environment

**arXiv ID:** 2603.18067 | [PDF](https://arxiv.org/pdf/2603.18067v1)

**作者:** Wuqi Wang `[一作]` (Chang'an University), Hongkai Yu `[通讯]` (Cleveland State University)

**通讯引用:** 3150 | [OpenAlex ID](https://openalex.org/A5025512337)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并构建了首个真实世界的暗夜驾驶基准数据集 DarkDriving，并基于轨迹跟踪与姿态匹配技术（TTPM）实现了昼夜图像对齐误差仅厘米级；随后在该数据集上开展了低光照增强、通用低光增强以及对 2D/3D 检测的四项任务实验；

**💡 创新点**

创新点包括①在大规模封闭试验场景中自动采集精准昼夜配对图像，突破了以往仅能控制曝光或 GPS 粗配对的限制；②提出 TTPM 方法实现了多路感知（LiDAR、IMU、GPS）同步定位与轨迹一致性，保证了空间内容与位置高度对齐；③通过人机协作精细校正，进一步将对齐误差压至厘米级；④在 DarkDriving 上开展四项感知任务，展示低光增强对检测性能的显著提升，并验证了跨数据集的推广能力。

**🔧 技术方法**

使用了高精度点云地图与 NDT 定位、Pure Pursuit 轨迹跟踪、PID 控制、姿态匹配优化、人工后处理；低光增强则采用 SNR‑Aware、LLFormer、Retinexformer、ControlNet 等先进模型；检测任务使用 YOLOv11（2D）与 BEVDepth/CRN（3D）框架。

**📊 数据集**

主要数据集为自建 DarkDriving（9,538 对昼夜图像，2D Car 标注 13,184 框），此外还在 nuScenes 夜间验证集上评估通用性，并与 LOL‑v2 等公开低光数据集做对比。

**📈 对比分析**

与现有低光增强方法相比，SNR‑Aware 在 DarkDriving 上取得 PSNR 最高、SSIM 与 LPIPS 最佳，检测方面在 COCO‑预训练下 2D AP_50 提升至 0.55，3D AP 亦达到 0.148，接近 LightTheNight 等专门训练模型；跨数据集实验表明 DarkDriving 预训练模型在 nuScenes 夜间数据上取得 35.83 的 MUSIQ，优于 LOL‑v2。

**⚠️ 局限性**

局限性包括：①仅在封闭试验场采集，场景多样性受限；②标注仅覆盖 Car 类，缺少多类别和动态对象；③TTPM 依赖高精度点云地图，需额外构建与维护；④图像分辨率统一 2,448×2,048，可能不兼容部分高分辨率网络；⑤未对极端天气或大光斑、雨雾等复杂条件进行评估。

---

## 488. Engineering Verifiable Modularity in Transformers via Per-Layer Supervision

**arXiv ID:** 2603.18029 | [PDF](https://arxiv.org/pdf/2603.18029v1)

**作者:** J. Clayton Kerce `[一作]` (Georgia Tech Research Institute), J. Clayton Kerce `[通讯]` (Georgia Tech Research Institute)

**通讯引用:** 138 | [OpenAlex ID](https://openalex.org/A5053040606)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Transformer中加入双流处理、冻结符号流、门控注意力以及逐层监督，显著提升模型的模块化与可控性，消除了Hydra效应；

**💡 创新点**

通过逐层监督诱导功能分离，首次将模型的冗余计算转化为可解释且可控制的模块；

**🔧 技术方法**

使用双流Transformer架构、冻结符号流、门控注意力、逐层监督损失和辅助层级分类器；

**📊 数据集**

在受控的小学教学文本语料上训练6层6头的Transformer；

**📈 对比分析**

与仅在最终层监督的对照模型相比，逐层监督模型的消融效应扩大5–23倍，控制灵活度提升4倍；

**⚠️ 局限性**

仅在小规模、受限语料与架构上验证，缺乏对大型模型和多样化自然语言环境的推广性评估。

---

## 489. EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing

**arXiv ID:** 2603.19224 | [PDF](https://arxiv.org/pdf/2603.19224v1)

**作者:** Yang Fu `[一作]` (Fudan University), Henghui Ding `[通讯]` (Fudan University)

**通讯引用:** 4180 | [OpenAlex ID](https://openalex.org/A5036631624)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一套大型混合式视频对象去除数据集VOR，并基于双向学习的EffectErase方法实现视频对象及其视觉副作用的高质量去除。

**💡 创新点**

创新点包括①通过将视频插入任务视为逆向辅助任务实现双任务联合学习；②引入任务感知区域引导（TARG）模块利用跨注意力建模对象与副作用的时空关联；③设计效应一致性损失（EC）让插入与去除共享副作用区域，从而提升定位精度和视觉一致性。

**🔧 技术方法**

采用扩散模型框架：预训练VAE + DiT、跨注意力、LoRA微调、任务与前景token的交互；同时利用SAM2、CLIP等工具生成高质量掩码与特征。

**📊 数据集**

使用自研的VOR数据集，包含约60K对摄像机捕获与合成视频，涵盖5类主副作用（遮挡、阴影、照明、反射、变形）和366种物体类别；另提供VOR-Eval与VOR-Wild两套评测集。

**📈 对比分析**

在ROSE、VOR-Eval与VOR-Wild三大数据集上与多种先进图像/视频修复与对象去除方法对比，EffectErase在PSNR、SSIM、LPIPS、FVD等指标上均取得领先成绩，并在无标注的VOR-Wild通过用户调查与QScore获得最高评分。

**⚠️ 局限性**

局限性：方法依赖手工掩码，需用户提供精确去除区域；未来工作方向为支持更友好的交互方式（如文本或语音）。

---

## 490. Do Large Language Models Possess a Theory of Mind? A Comparative Evaluation Using the Strange Stories Paradigm

**arXiv ID:** 2603.18007 | [PDF](https://arxiv.org/pdf/2603.18007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 491. An Onto-Relational-Sophic Framework for Governing Synthetic Minds

**arXiv ID:** 2603.18633 | [PDF](https://arxiv.org/pdf/2603.18633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 492. LVOmniBench: Pioneering Long Audio-Video Understanding Evaluation for Omnimodal LLMs

**arXiv ID:** 2603.19217 | [PDF](https://arxiv.org/pdf/2603.19217v1)

**作者:** Keda Tao `[一作]` (Zhejiang University), Huan Wang `[通讯]` (Westlake University)

**通讯引用:** 7789 | [OpenAlex ID](https://openalex.org/A5100751566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了LVOmniBench评测平台，评估OmniLLMs在长时音视频理解上的能力。

**💡 创新点**

首次提出专为长时音视频跨模态理解设计的基准，包含手工挑选的275段10-90分钟视频及1014个多模态多选问题。

**🔧 技术方法**

采用多模态预训练模型（Gemini、Qwen、Ming-Flash等）与手工标注、难度分层与跨模态对齐技术，对长时音视频输入进行推理。

**📊 数据集**

使用LVOmniBench数据集，包含275段长视频和1014个QA对，覆盖娱乐、生活、DIY等21个子类。

**📈 对比分析**

在LVOmniBench上与多款开源模型对比，Gemini 3 Pro达65%准确率，开源模型低于35%，表明长时音视频推理仍是挑战。

**⚠️ 局限性**

局限性在于长时音视频的上下文容量、跨模态对齐困难，以及开源模型对音频的利用不足。

---

## 493. Mathematical Foundations of Deep Learning

**arXiv ID:** 2603.18387 | [PDF](https://arxiv.org/pdf/2603.18387v1)

**作者:** Xiaojing Ye `[一作]` `[通讯]`, Xiaojing Ye

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理了深度学习的数学理论基础，包括函数逼近、通用逼近定理、优化与数值算法以及与物理、控制、生成模型等领域的数学交叉；通过证明深度 ReLU 网络在 Sobolev 空间中的逼近误差和网络规模的定量估计，构建了理论与实践相结合的深度学习框架；并在后续章节给出了常用激活函数、网络模块（如残差网络、卷积网络、图网络、循环网络）以及如何利用物理约束设计损失函数（如 PDE 的 Poisson 方程物理信息神经网络）的实例。

**💡 创新点**

创新点在于将深度学习视为一个严格的数学工程，提供了：①对深度网络表达能力的定量误差估计与网络规模上界；②统一的理论视角将优化、泛化、表达能力等概念联系起来；③利用物理约束和弱形式的 PDE 设计损失函数的完整理论与方法论，填补了传统经验设计与理论证明之间的空白。

**🔧 技术方法**

主要技术包括：深度 ReLU 网络的构造与误差分析、Sobolev 空间与 L∞ 规范的数学工具、梯度与 Hessian 计算、Monte Carlo 积分用于高维 PDE 目标函数、弱形式与对抗网络的训练方法、以及自动微分实现梯度与拉普拉斯算子。

**📊 数据集**

文中未给出具体数据集，而是以理论证明和通用数学框架为主；在物理信息神经网络的案例中，采用 Poisson 方程的典型右端项 f(x) 以及 Dirichlet 边界 g(x) 作为假设函数，利用均匀采样的 Monte Carlo 估计积分。

**📈 对比分析**

由于本文以理论分析为主，未进行大规模实验比较；但对比了理论所给的网络规模上界与实际网络设计经验，指出理论上所需参数远大于常用实践中的网络规模，说明理论上限与实际效果存在差距。

**⚠️ 局限性**

局限性包括：①未覆盖所有最新网络结构（如 Transformer、U‑Net 等）；②缺乏对实验数据和模型性能的系统验证；③理论证明多基于理想化假设（如无噪声、完美采样），与实际深度学习任务的复杂性仍有差距；④对高维 PDE 的数值实现仍需进一步优化和验证。

---

## 494. Cognitive Amplification vs Cognitive Delegation in Human-AI Systems: A Metric Framework

**arXiv ID:** 2603.18677 | [PDF](https://arxiv.org/pdf/2603.18677v1)

**作者:** Eduardo Di Santi `[一作]` `[通讯]`, Eduardo Di Santi

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出一套衡量人机协作中认知放大与委托两种模式的概念与数学框架，定义了 CAI*、依赖比率 D、人工依赖指数 HRI 与人类认知漂移率 HCDR 四个指标，能够评估系统是否真正实现认知增强或出现认知委托。

**💡 创新点**

创新点在于将人机协作的可持续性与认知质量统一起来，首次提出“认知可持续性约束”——在追求混合系统性能提升的同时，要求 HCDR ≥ 0，避免短期性能提升导致长期人类专业能力衰退。

**🔧 技术方法**

使用了基于任务表现（准确率、召回率、F1 分数等）的相对效能度量方法，构造了 Q(S)、Q_H、Q_A、Q_HA 的算式，并推导出四个指标；同时提出了低维度状态空间（D vs CAI*）来可视化协作模式。

**📊 数据集**

文章没有使用公开数据集，而是给出了三个示例情境（工程诊断、药物安全监测、工业异常诊断）来说明如何计算指标，示例数据均为假设性数值。

**📈 对比分析**

比较方法是将混合系统的表现与最佳单一组件（人类或 AI）进行对比，计算 CAI* 作为增益度量；D 则衡量 AI 对性能的占比，HRI 为其补充。示例表明尽管 CAI* > 0（实现了认知放大），但高 D 值表明系统仍处于 AI 主导，可能导致认知委托。

**⚠️ 局限性**

局限性包括：①指标 Q 并非统一的智能度量，需针对特定任务手工定义；②缺乏大规模实证验证，指标的阈值（如 D = 0.5、0.8）未经过统计检验；③示例数据为假设，未在真实工业或临床环境中测试，难以评估指标对人类认知衰退的预测准确性。

---

## 495. From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents

**arXiv ID:** 2603.18382 | [PDF](https://arxiv.org/pdf/2603.18382v1)

**作者:** Myeongseob Ko `[一作]` (Virginia Tech), Ruoxi Jia `[通讯]` (Virginia Tech)

**通讯引用:** 2760 | [OpenAlex ID](https://openalex.org/A5032275274)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化 LLM 代理在去标识化数据中通过组合弱线索实现身份重建的风险，并提出了可测量的评估框架。

**💡 创新点**

创新点包括：① 将“推理驱动的链接（Inference‑Driven Linkage）”定义为一种新的隐私失败模式；② 设计并公开了控制实验基准 InferLink，用于系统探测指纹类型、任务意图和攻击者知识对链接成功率的影响；③ 在经典案例（Netflix、AOL）和现代文本痕迹（Anthropic 访谈、ChatGPT 日志）中验证 LLM 代理的普适性。

**🔧 技术方法**

主要技术手段：LLM 代理（GPT‑5、Claude 4.5）通过自然语言提示进行数据匹配、候选生成与证据检索；构造专门的推理链路接口；采用 LSR（链接成功率）和 CLC（已确认链接计数）等度量；对代理做隐私提示干预（privacy‑aware system prompt）以评估缓解效果。

**📊 数据集**

使用的数据集包括：Netflix Prize 历史评分集、AOL 搜索日志、InferLink 合成基准（三种指纹类型×三种意图×两种知识水平共180个实例）、Anthropic Interviewer 红本访谈文本、匿名化 ChatGPT 对话日志。

**📈 对比分析**

比较方法：在经典场景中将 LLM 代理与手工工程化基线（如 Netflix 的稀疏匹配算法）对比；在 InferLink 基准中记录 LSR 与任务效用 U 的双重指标；在现代痕迹中用 CLC 统计成功案例。实验结果显示：在 Netflix 上 GPT‑5 的 LSR 约 79%（远超 56% 基线）；在 InferLink 的 Explicit‑MK 场景中 LSR 接近 1.0；在现代文本场景中，Anthropic 访谈得到 6 例 CLC，ChatGPT 日志得到 1 例 CLC。与此同时，隐私提示干预可将 LSR 降至接近 0，但会伴随显著的效用损失。

**⚠️ 局限性**

局限性：① 基准假设单一真值重合且属性模式固定，未覆盖多重近似匹配或大规模候选集的模糊情形；② 现代痕迹案例仅展示可能性，无法估计普遍性；③ 评估的效用度量局限于场景级交付任务，未涵盖更广泛的代理使用情景；④ 通过提示干预降低链接风险会引发过度拒绝，表明需更细粒度的对策。

---

## 496. Rapid Adaptation of Particle Dynamics for Generalized Deformable Object Mobile Manipulation

**arXiv ID:** 2603.18246 | [PDF](https://arxiv.org/pdf/2603.18246v1)

**作者:** Bohan Wu `[一作]` (Stanford University), Li Fei-Fei `[通讯]` (Stanford University)

**通讯引用:** 216616 | [OpenAlex ID](https://openalex.org/A5100450462)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过在模拟中先利用粒子位置等特权信息学习变形物体的动力学与形状嵌入，再通过视觉与动作在线推断这些嵌入，最终实现机器人在真实世界中对未知变形物体进行移动式操控。

**💡 创新点**

创新点在于：1）利用粒子位置捕捉变形物体形状变化，使Rapid Motor Adaptation（RMA）可扩展到可变形物体；2）采用两阶段训练：先用特权信息学习嵌入，再用视觉+动作学习适配器，实现零样本跨域迁移；3）将形状与动力学分别编码并通过L1损失独立学习适配模块，保证在线快速适应。

**🔧 技术方法**

使用的技术包括：快速运动适应（Rapid Motor Adaptation）、强化学习（RL）、深度视觉感知（RGB‑D摄像头）、形状与动力学适配模块、粒子系统模拟、OmniGibson仿真器、L1 损失训练。

**📊 数据集**

使用的数据集：20类1D变形物体（如绳、皮带等）与20类2D变形物体（如毛巾、塑料袋等）的模拟与真实样本；此外使用真实世界中的多种物体、环境和光照场景进行测试。

**📈 对比分析**

与DMfD和DDOD两种基线方法比较，实验显示在两项任务上成功率分别达到约80%+，相对基线提高了约65%+。消融实验进一步验证了形状与动力学适配模块的重要性。

**⚠️ 局限性**

局限性：1）仅依赖深度图输入，对极端遮挡和光照变化的鲁棒性有限；2）适配模块训练依赖大量仿真随机化，真实场景中可能出现未覆盖的变形特征；3）未在真实世界收集数据进行微调，缺乏对极端变形或材质的适应性验证。

---

## 497. Rethink Web Service Resilience in Space: A Radiation-Aware and Sustainable Transmission Solution

**arXiv ID:** 2603.18526 | [PDF](https://arxiv.org/pdf/2603.18526v1)

**作者:** Long Chen `[一作]` (Simon Fraser University), Jiangchuan Liu `[通讯]` (Simon Fraser University)

**通讯引用:** 20544 | [OpenAlex ID](https://openalex.org/A5039311485)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了RALT系统，在LEO卫星网络中根据空间辐射与能量消耗动态重路由，以提升网页、视频、实时通信等Web服务的可靠性。

**💡 创新点**

将空间辐射对TID/TNID和大气拖曳的能量需求纳入路由度量，实现能源感知、可持续的重路由，并通过控制平面实时集成辐射数据。

**🔧 技术方法**

基于Python实现的控制平面调度器，使用改写的短路径算法（延迟+电池消耗权重），结合NOAA太阳风指数、卫星电量与辐射累积数据。

**📊 数据集**

使用公开的太阳风指数、NOAA大气密度模型、卫星电池与辐射传感器数据，以及星座仿真模型（Walker与Near‑polar），并生成300 Mb/s用户流量。

**📈 对比分析**

与PHOENIX和Umbra做对比，实验显示RALT在不同辐射强度下平均降低电池消耗，保持与PHOENIX相当的端到端延迟，且将因卫星关机导致的Web服务中断减少约42.95%。

**⚠️ 局限性**

局限包括对单一辐射阈值和阈值γ的依赖、未考虑更复杂的多跳重路由时的协同效应，以及在极端辐射事件下可能仍需硬件冗余。

---

## 498. TexEditor: Structure-Preserving Text-Driven Texture Editing

**arXiv ID:** 2603.18488 | [PDF](https://arxiv.org/pdf/2603.18488v1)

**作者:** Bo Zhao `[一作]` (Nanjing University), Wei Ji `[通讯]` (Nanjing University)

**通讯引用:** 21694 | [OpenAlex ID](https://openalex.org/A5100664952)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TexEditor，专注于文本驱动的纹理编辑并保持几何结构；

**💡 创新点**

通过构建高质量的Blender渲染数据集TexBlender以及基于结构损失的RL方法StructureNFT实现结构保持；

**🔧 技术方法**

使用Qwen-Image-Edit-2509模型进行SFT和RL，结合Gemini、SAM、SAUGE等视觉工具做结构监督；

**📊 数据集**

使用TexBlender（Blender+3D-Front）做训练数据，TexBench（COCO）做测试数据，并验证ImgEdit子任务；

**📈 对比分析**

在TexBench和Blender基准上与Nano Banana Pro、Alchemist等对比，TexEditor在指令遵循、结构一致性和综合TexEval指标上均显著优于基线；

**⚠️ 局限性**

对纹理编辑任务仍有限的多样性、对极端结构变化的鲁棒性不足，以及RL训练的计算成本高等限制仍待改进。

---

## 499. Turnpike with Uncertain Measurements: Triangle-Equality ILP with a Deterministic Recovery Guarantee

**arXiv ID:** 2603.18283 | [PDF](https://arxiv.org/pdf/2603.18283v1)

**作者:** C. S. Elder `[一作]` (Carnegie Mellon University), Carl Kingsford `[通讯]` (Carnegie Mellon University)

**通讯引用:** 24833 | [OpenAlex ID](https://openalex.org/A5113653378)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在噪声和舍入下从无标签的成对距离多集恢复一维点集的问题，提出基于三角等式的分配‑回归两阶段模型，并给出整数线性规划与其线性松弛。

**💡 创新点**

创新点在于将距离分配视为多匹配并引入两分区集合 𝒫_y，将三角等式转化为纯组合约束；证明在噪声+舍入满足分隔条件时 𝒫_y 能被准确恢复，并可通过整数解证实 realizability。

**🔧 技术方法**

使用的技术包括多匹配建模、整数线性规划、两分区的三角约束、线性松弛、两指针枚举 𝒫_y、理论分隔条件以及实验验证。

**📊 数据集**

使用的实验数据集包括合成实例（均匀、正态、柯西分布的坐标；线性与循环）以及部分消化（partial digest）基因组片段实验（线性与圆形）。

**📈 对比分析**

与回归基线（MM、GD、排序网络扩展式）和纯三角等式 ILP/LP 进行比较；无噪声下 LP 整数率高，三角 ILP 的分配误差和排列距离优于基线；在噪声+舍入下满足分隔条件时两分区恢复率高，噪声增大后性能快速下降。

**⚠️ 局限性**

限制在于仅适用于噪声+舍入模型，无法处理缺失或重复距离；整数规划在约 100 点时不可行；松弛可能产生非整数解，理论整式性条件尚未完全确定。

---

## 500. TopoChunker: Topology-Aware Agentic Document Chunking Framework

**arXiv ID:** 2603.18409 | [PDF](https://arxiv.org/pdf/2603.18409v1)

**作者:** Xiaoyu Liu `[一作]` `[通讯]` (Independent Researcher), Xiaoyu Liu (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TopoChunker，一个基于拓扑的文档分块框架，兼顾结构完整性与检索质量。

**💡 创新点**

通过结构化中间表示（SIR）和双代理（Inspector与Refiner）实现动态路由与语义审计，显著减少“语义碎片”并提升上下文连贯性。

**🔧 技术方法**

采用主动探测的Inspector Agent结合规则、LLM与VLM路径；Refiner Agent进行容量审核、语义切片与上下文消解；使用堆栈遍历构建SIR并实现原子性锁定。

**📊 数据集**

在公共领域小说集GutenQA和政府报告集GovReport两套基准数据上进行评估。

**📈 对比分析**

与四种基线（FC200、Semantic Chunker、Proposition-Level、LumberChunker）对比，TopoChunker在Recall@3上分别在GutenQA取得64.59%、GovReport取得83.26%，生成准确率在GovReport达82%，同时令Token消耗下降23.5%。

**⚠️ 局限性**

依赖外部VLM进行视觉布局解析时可能出现OCR/布局误差；多代理顺序调用导致推理延迟高于单调启发式分块。

---

## 501. dTRPO: Trajectory Reduction in Policy Optimization of Diffusion Large Language Models

**arXiv ID:** 2603.18806 | [PDF](https://arxiv.org/pdf/2603.18806v1)

**作者:** Wenxuan Zhang `[一作]` (Meta AI), Wei Wen `[通讯]` (Meta AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为Trajectory Reduction的技术，用少量前向推理估计离散扩散语言模型（dLLM）的轨迹概率，并在此基础上构建了离线的 DPO 训练框架。

**💡 创新点**

创新点包括：① 将轨迹概率分解为新解码 token 的概率比，完全消除对 mask 调度的依赖；② 通过状态约简与比例约简，仅使用一次前向推理（或每块一次）即可估计轨迹概率；③ 在保持训练成本与 ARMs DPO 相当的同时，实现了与主流 dLLM 的性能对齐。

**🔧 技术方法**

使用的技术主要有：轨迹状态与比例约简、Block Attention、top‑k 置信度调度、参数高效训练（冻结大部分参数）、DPO 以及多种投影函数（log‑sigmoid、IPO、RSO 等）。

**📊 数据集**

训练数据来自 SmolTalk2 偏好集（50w 对），并混合了数学、代码偏好数据；在零样本评测中使用了 GPQA、GSM8K、MATH、LCBv6、MBPP、HumanEval、IFEval、Arena‑Hard、MT‑Bench 等基准。

**📈 对比分析**

与 Fast‑dLLM‑v2、LLaDA、Dream 等开源 dLLM 以及 Qwen2.5‑7B‑Instruct（ARM）对比，本文方法在零样本下实现：指令跟随 +9.6%，STEM +4.0%，代码 +4.3%；与 Fast‑dLLM‑v2 的对比提升分别为 GPQA 9.6%、GSM8K 3.6%、MATH 4.0%、LCBv6 3.6%、HumanEval+ 4.3%、IFEval 2.95%；训练成本与 ARMs DPO 相当，推理速度可与 Fast‑dLLM‑v2 相近。

**⚠️ 局限性**

局限性包括：实验仅在 7B 规模模型上验证，尚未验证更大规模；对特定的 top‑k 置信度调度和 Block Attention 结构的依赖可能限制在不同架构或语言上的迁移；在极长序列或多语言场景下的性能与稳定性尚未完全评估。

---

## 502. Ontology-Guided Diffusion for Zero-Shot Visual Sim2Real Transfer

**arXiv ID:** 2603.18719 | [PDF](https://arxiv.org/pdf/2603.18719v1)

**作者:** Mohamed Youssef `[一作]` (University of Stuttgart), Andreas Bulling `[通讯]` (University of Stuttgart)

**通讯引用:** 14548 | [OpenAlex ID](https://openalex.org/A5073661463)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Ontology-Guided Diffusion (OGD)，通过将视觉真实度拆解为可解释的属性并编码在知识图谱中，再结合PDDL符号规划生成可执行的视觉编辑指令，实现零样本模拟到现实的图像翻译。

**💡 创新点**

创新点在于①把视觉真实度结构化为可解释的属性及其因果关系并通过知识图谱编码；②利用符号规划产生因果一致的编辑动作；③将图嵌入与结构化提示共同注入扩散模型的交叉注意力，实现可解释且数据高效的sim2real转换。

**🔧 技术方法**

使用CLIP特征+轻量级MLP预测属性、GraphSAGE图传播生成属性嵌入、PDDL规划生成编辑计划、InstructPix2Pix扩散编辑模型，以及结构化提示+图嵌入的交叉注意力和对齐损失。

**📊 数据集**

使用140张无配对的合成/真实图像进行属性学习；在Virtual KITTI 2与KITTI以及Lego数据集上训练扩散模型。

**📈 对比分析**

与InstructPix2Pix、ControlNet等基线在TraitDist、LPIPS、SSIM指标上对比，OGD在TraitDist和LPIPS下降、SSIM提升，真实性分类准确率达98.4%，显著优于基线。

**⚠️ 局限性**

受限于预定义的静态知识图谱和符号规划，对极端或未知的视觉属性适应性有限；规划目标不可达时会失败；对极其多样的现实因素仍需进一步扩展。

---

## 503. SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation

**arXiv ID:** 2603.18599 | [PDF](https://arxiv.org/pdf/2603.18599v1)

**作者:** Jialiang Kang `[一作]` (Peking University), Xinghao Chen `[通讯]` (Huawei Technologies)

**通讯引用:** 3792 | [OpenAlex ID](https://openalex.org/A5006817088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练‑free、无损的 Speculative Jacobi Decoding (SJD) 增强框架，通过 Proactive Drafting 与 Adaptive Continuation 两个技术提升文本到图像 (T2I) 的推理速度。

**💡 创新点**

创新点在于：①Proactive Drafting 在一次 token 拒绝后构建 K‑ary 树形多路径草稿，显著降低单 token 拒绝率；②Adaptive Continuation 在首次拒绝后继续验证其余 token，只重采样被拒绝的 token，保持序列稳定；③两者协同实现无损加速且无需额外训练。

**🔧 技术方法**

使用的技术包括：Speculative Jacobi Decoding、拒绝采样（Rejection Sampling）、局部树形采样、固定窗口迭代、上下文相对不变的概率利用等。

**📊 数据集**

实验使用 MS‑COCO 2017 验证集和 PartiPrompts 数据集，基于 Lumina‑mGPT（768×768）和 Emu3（720×720）模型。

**📈 对比分析**

与 EAGLE‑2、原始 SJD、SJD2、LANTERN++、GSD 等 baseline 进行对比，结果显示在 Lumina‑mGPT 上 Step Compression 达到 4.51×、Wall‑Clock Speedup 3.80×，Emu3 上 Step Compression 4.31×、Wall‑Clock Speedup 3.25×，且 FID 与 CLIP‑Score 与原模型基本一致，优于 lossy 方法。

**⚠️ 局限性**

局限性包括：需要手动调节 K、D、L 等超参数；在极高熵或更大模型/更大图像尺寸时显存和计算开销仍是瓶颈；目前仅针对 autoregressive T2I 模型，需进一步验证跨模型通用性。

---

## 504. Responsible AI in criminal justice: LLMs in policing and risks to case progression

**arXiv ID:** 2603.18116 | [PDF](https://arxiv.org/pdf/2603.18116v1)

**作者:** Muffy Calder `[一作]` (University of Glasgow), Evdoxia Taka `[通讯]` (University of Glasgow)

**通讯引用:** 28 | [OpenAlex ID](https://openalex.org/A5010924821)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性识别并说明了在英国和威尔士执法与司法系统中使用大型语言模型（LLM）可能对案件进程产生的17类风险，并给出42个具体实例，涵盖从社区警务、情报、调查到起诉等阶段。

**💡 创新点**

首次对LLM在执法领域的风险进行分类与案例化，提出“输出、输入、评估、系统工程”四大风险维度，并结合实际案例展示风险对案件推进的正负面影响，为执法部门和辩护律师提供风险清单与防范思路。

**🔧 技术方法**

采用文献综述与案例分析方法，对LLM相关技术（如RAG、prompt工程、跨模态LLM等）进行理论剖析，并构建风险评估框架。

**📊 数据集**

未使用公开数据集；研究主要基于已有文献、先行项目（如ADA工具评估）、以及司法流程模型与案例描述。

**📈 对比分析**

本文并未进行实验或性能比较；评估依赖于专家访谈、案例分析与已有研究报告，未给出定量性能指标。

**⚠️ 局限性**

局限性包括：缺乏客观基准导致评估主观性；LLM输出不可验证的“ground truth”难以衡量；快速演进的模型技术导致风险随时间变化；未能系统评估跨工具链与多语言情境下的复合风险。

---

## 505. MIDST Challenge at SaTML 2025: Membership Inference over Diffusion-models-based Synthetic Tabular data

**arXiv ID:** 2603.19185 | [PDF](https://arxiv.org/pdf/2603.19185v1)

**作者:** Masoumeh Shafieinejad `[一作]` (Vector Institute), Deval Pandya `[通讯]` (Vector Institute)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5108445535)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

举办了MIDST挑战赛，评估基于扩散模型的合成表格数据对成员推断攻击的防御能力。

**💡 创新点**

首次针对复杂单表和多表扩散模型进行成员推断攻击评测，并引入多种黑盒与白盒攻击方法，揭示其隐私泄露风险。

**🔧 技术方法**

利用TabDDPM、TabSyn、ClavaDDPM等扩散模型，结合Shadow模型、机器学习分类器、梯度与损失特征等技术构造攻击。

**📊 数据集**

使用公开的Berka金融交易数据集（单表Transaction及多表相关表）。

**📈 对比分析**

通过TPR@FPR10%等指标评估攻击效果，Tartan Federer团队在四个轨道均获胜，攻击成功率最高可达46%，表明攻击显著优于随机猜测。

**⚠️ 局限性**

局限在于多表白盒轨道攻击效果不佳，且只针对Transaction表，未能充分验证跨表信息对攻击的影响；同时结果仅针对扩散模型，缺乏与其他生成式AI方法的对比。

---

## 506. Training-Free Sparse Attention for Fast Video Generation via Offline Layer-Wise Sparsity Profiling and Online Bidirectional Co-Clustering

**arXiv ID:** 2603.18636 | [PDF](https://arxiv.org/pdf/2603.18636v1)

**作者:** Jiayi Luo `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**通讯引用:** 18154 | [OpenAlex ID](https://openalex.org/A5100380470)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练稀疏注意力框架，通过离线层级稀疏性分析与在线双向共聚类实现视频生成加速。

**💡 创新点**

创新点在于识别每层稀疏性异质性并离线预估，和引入双向共聚类实现查询‑键耦合的块划分，显著提升稀疏匹配效率。

**🔧 技术方法**

使用离线层级稀疏性评估、在线双向共聚类、块级稀疏注意力以及FlashInfer核等技术。

**📊 数据集**

在VBench文本‑视频和VBench++图像‑视频数据集上进行评估。

**📈 对比分析**

与SparseAttn、SVG、SVG2、Radial等SOTA方法对比，最高可达1.93×速度提升，PSNR高达29 dB，整体质量与速度均优于对手。

**⚠️ 局限性**

局限性在于仍需额外的块划分计算开销，对极度动态场景的稀疏性适应性可能不足。

---

## 507. The Truncation Blind Spot: How Decoding Strategies Systematically Exclude Human-Like Token Choices

**arXiv ID:** 2603.18482 | [PDF](https://arxiv.org/pdf/2603.18482v1)

**作者:** Esteban Garces Arias `[一作]` (Ludwig Maximilian University), Matthias Aßenmacher `[通讯]` (Munich Center for Machine Learning)

**通讯引用:** 175 | [OpenAlex ID](https://openalex.org/A5069469652)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对1.8 M条机器生成文本与5,261条人类文本的系统实验，探究了大型语言模型文本可检测性的根本原因，并提出了“截断盲区”假设，说明在基于概率的截断解码策略下，统计上罕见但语用上合适的词被排除，从而使机器文本与人类文本在可预测性和词汇多样性上产生可观的分离；

**💡 创新点**

创新点在于：①首次将截断策略与人类语用选择对比，揭示了截断盲区是导致可检测性的关键因素；②证明检测性能主要取决于截断参数而非模型规模或架构；③用极简的两维特征（可预测性、词汇多样性）即可实现96%+的二分类检测，展示了检测信号的强健性；

**🔧 技术方法**

使用了可预测性（基于参考模型的对数似然）与词汇多样性（n-gram多样性）两项统计特征，配合随机森林、逻辑回归、朴素贝叶斯三类基线分类器；对8种模型（Transformer、Mamba、RWKV等）、5种截断策略（top‑k、top‑p、contrastive、beam等）及53个超参数组合进行大规模实验；

**📊 数据集**

采用三大英文语料库：BookCorpus（小说）、WikiText（百科）、WikiNews（新闻），共5,261条人类样本；随后使用同样的提示在不同模型/解码配置下生成约1.8 M条机器文本；

**📈 对比分析**

比较方法：对每个模型/解码/超参数组合计算可预测性与多样性，然后用上述两特征训练分类器，评估AUC‑ROC、F1等指标。结果显示：beam search可检测率最高（AUC≈0.997），top‑k最低（≈0.948）；整体AUC平均≈0.97‑0.98；检测性能与模型规模无关，主要随截断强度变化；

**⚠️ 局限性**

局限性包括：①可预测性和多样性未能完整捕捉上下文语用适当性，导致部分语义连贯但统计特征接近人类的文本仍被误判；②实验仅覆盖英文、短文本（≤256词）与三类写作风格，可能不适用于其他语言、长文本或对话；③人类词选择机制的理论构造基于已写文本，未直接观测真实语言生产过程；③模型种类与训练细节有限，结果可能随其他架构或更大规模模型变化。

---

## 508. Latent Factor Modeling with Expert Network for Multi-Behavior Recommendation

**arXiv ID:** 2603.18556 | [PDF](https://arxiv.org/pdf/2603.18556v1)

**作者:** Mingshi Yan `[一作]` (Tianjin University), Meng Wang `[通讯]` (Hefei University of Technology)

**通讯引用:** 42512 | [OpenAlex ID](https://openalex.org/A5100377147)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于门控专家网络的多行为推荐模型MBLFE，利用专家网络对所有潜在因子进行建模并通过门控网络为每个用户自适应选择最相关的专家，从而实现用户兴趣的解耦与精准表示；

**💡 创新点**

创新点在于：①将所有潜在因子统一放入专家网络并通过门控机制自适应选择，避免传统方法的因子纠缠；②结合自监督对比学习实现专家间的独立性与同一专家输出的一致性；③使用嵌入增强网络（GCN）挖掘多行为的协同信息，为专家提供更丰富的输入；

**🔧 技术方法**

技术包括：门控专家网络（Mixture-of-Experts）+噪声Top-k门控、LightGCN嵌入增强、对比学习（NCE）保证因子一致性与独立性、BPR损失进行目标行为预测、整体多任务训练；

**📊 数据集**

使用三大电商/点评数据集：Tmall（点击、收藏、加入购物车、购买）、Taobao（点击、加入购物车、购买）、Yelp（不喜欢、中立、喜欢）作为目标行为和辅助行为；

**📈 对比分析**

与1个单行为基线和9个多行为基线（MBGCN、S-MBRec、CRGCN、CIGF、MB-CGCN、PKEF、Disen-CGCN、COPF、BCIPM）进行对比，指标HR@10、NDCG@10、HR@20、NDCG@20；MBLFE在所有数据集上均显著优于基线，提升幅度约3–8%（HR@10）和6–12%（NDCG@10），且在实验中保持了可解释性；

**⚠️ 局限性**

限制主要体现在：①模型参数量和训练时间相对较大（需数十秒/epoch），②对专家数量的调优依赖经验，过多或过少都会影响性能；③门控机制虽然降低计算量，但在极端稀疏场景下仍可能导致专家集空交集导致收敛困难；

---

## 509. Synthetic Data Generation for Training Diversified Commonsense Reasoning Models

**arXiv ID:** 2603.18361 | [PDF](https://arxiv.org/pdf/2603.18361v1)

**作者:** Tianhui Zhang `[一作]` (University of Liverpool), Danushka Bollegala `[通讯]` (University of Liverpool)

**通讯引用:** 3722 | [OpenAlex ID](https://openalex.org/A5073503574)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CommonSyn 这一合成数据集，并通过两阶段数据选择策略提升生成式常识推理（GCR）模型的质量与多样性。

**💡 创新点**

创新点在于：①将概念扩展与句子生成三种策略融合，②提出局部与全局多样性平衡的两阶段筛选方法，③在多种大模型上验证了 Pareto 优化的效果。

**🔧 技术方法**

采用大语言模型（如 Qwen2.5-72B-Instruct、Gemini-2.5-Flash、GPT‑4o）进行概念扩展、句子生成与质量评估；使用 SimCSE 句向量计算多样性；通过 LoRA 在 unsloth 框架下进行参数高效微调。

**📊 数据集**

主要使用 CommonGen 作为概念与句子来源，生成后构建 CommonSyn；随后在 ComVE、α‑NLG、ROCStories 以及 CSQA、CSQA2、PIQA 等任务上进行跨任务评估。

**📈 对比分析**

与零样本、CommonGen 微调以及梯度混合等基线相比，CommonSyn 在 11 个 LLM 上实现了显著提升：覆盖率、Win‑Tie 率、整体质量与 Self‑CosSim 等多样性指标均出现 Pareto 改进，且在其他 GCR 任务上保持或提升性能。

**⚠️ 局限性**

局限性包括：①仅覆盖英语，跨语言扩展受限；②合成质量受生成器和评估模型（Qwen、Gemini、GPT‑4o）的偏差影响；③全局多样性计算需 O(n²) 余弦相似度，规模扩大时计算瓶颈；④方法主要针对关键词到文本的生成任务，对传统序列到序列任务的适用性需进一步研究。

---

## 510. Mitigating the Bandwidth Wall via Data-Streaming System-Accelerator Co-Design

**arXiv ID:** 2603.19057 | [PDF](https://arxiv.org/pdf/2603.19057v1)

**作者:** Qunyou Liu `[一作]` (École Polytechnique Fédérale de Lausanne), David Atienza `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 10909 | [OpenAlex ID](https://openalex.org/A5074236306)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

针对Transformer推理，提出MatrixFlow 16×16压缩矩阵乘法加速器与Gem5-AcceSys全系统仿真平台的统一硬件-软件协同设计。

**💡 创新点**

创新点在于将大规模矩阵乘法拆分为4 KB页对齐块、通过DMA+PCIe+SMMU实现全流水线数据流，并把局部缓存压缩到仅3个4 KB SRAM，充分利用系统内存带宽而非过度依赖本地存储。

**🔧 技术方法**

采用的技术包括：16×16同步数组、块化GEMM数据流、页面对齐DMA传输、PCIe/多通道DMA、SMMU地址映射、gem5全系统模拟（Gem5-AcceSys）与Linux驱动交互。

**📊 数据集**

使用BERT（Base/large）和Vision Transformer（Base/large）模型进行端到端推理实验，评估不同精度(INT8/FP16/FP32)和内存/互连配置。

**📈 对比分析**

与单核CPU、ARM Neon、SMAUG、TiC-SAT等基线对比，MatrixFlow在BERT‑Large实现最高≈698×单核加速，整体推理比CPU提升≈22×，相较于最先进的紧耦合/松耦合加速器分别提高5×–8×。

**⚠️ 局限性**

主要限制是仍受系统内存带宽和PCIe传输瓶颈影响，DMA/TLB开销随模型尺寸增长而显著，且该设计对高频宽PCIe及SMMU支持要求较高，无法在低带宽/低延迟环境中保持同等优势。

---

## 511. MedForge: Interpretable Medical Deepfake Detection via Forgery-aware Reasoning

**arXiv ID:** 2603.18577 | [PDF](https://arxiv.org/pdf/2603.18577v1)

**作者:** Zhihui Chen `[一作]` (National University of Singapore), Mengling Feng `[通讯]` (National University of Singapore)

**通讯引用:** 12331 | [OpenAlex ID](https://openalex.org/A5022222926)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了大规模医学造假数据集 MedForge-90K，并提出预先定位-推理的医学伪造检测框架 MedForge-Reasoner

**💡 创新点**

创新点在于将定位与推理融合为预先推理流程，并引入Forgery-aware GSPO强化视觉证据约束

**🔧 技术方法**

采用了多模态大语言模型与自监督的两阶段训练（SFT + GSPO），结合基于掩模覆盖的定位奖励与CoT格式奖励

**📊 数据集**

使用 MedForge-90K 该数据集涵盖胸部X光、脑MRI和眼底图的 90,000 张真实与伪造样本，配有专家指导的解释与真值边界框

**📈 对比分析**

与现有专用检测器和通用 MLLM 对比，MedForge-Reasoner 在 In-Domain、Cross-Model、Cross-Forgery 三种设置下均实现了 98–99% 的准确率，且推理质量评估分数提升 15–30%

**⚠️ 局限性**

局限性包括仅覆盖三种 2D 成像模态、说明仅为英文，且模型可能被滥用于改进伪造技术

---

## 512. TAU-R1: Visual Language Model for Traffic Anomaly Understanding

**arXiv ID:** 2603.19098 | [PDF](https://arxiv.org/pdf/2603.19098v1)

**作者:** Yuqiang Lin `[一作]` (University of Bath), Nic Zhang `[通讯]` (University of Bath)

**通讯引用:** 687 | [OpenAlex ID](https://openalex.org/A5030581111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了一个基于真实环岛路口监控视频的交通异常理解数据集Roundabout-TAU，并提出了两层级的视觉语言模型TAU-R1，用轻量级分类器做初步筛选，随后由大型推理器生成细粒度事件总结；

**💡 创新点**

创新点在于：①首次针对环岛场景提供QA式细粒度标签；②提出分层模型与分解式QA预训练结合GRPO后训练的任务专属策略；③设计了专门针对分类与摘要的奖励函数提升推理质量；

**🔧 技术方法**

技术方法包括：视觉语言模型（Qwen3-VL）、分层两阶段训练（分解QA监督+TAU-GRPO强化学习）、多模态评估与GPT-Eval等；

**📊 数据集**

使用的数据集为Roundabout-TAU，包含342段视频、2000+多维QA对，涵盖异常分类与事件描述；

**📈 对比分析**

与多种开源与商用大模型对比，TAU-R1在四分类AP、二分类F1、摘要BLEU/ROUGE/METEOR以及GPT‑Score等指标上均实现最高分，边缘部署在Jetson AGX Orin上可实现实时率0.47；

**⚠️ 局限性**

局限性包括：数据集仅覆盖环岛场景，跨场景泛化能力待验证；模型对极端异常的识别仍受限；对资源极低设备的进一步压缩仍需研究。

---

## 513. CWoMP: Morpheme Representation Learning for Interlinear Glossing

**arXiv ID:** 2603.18184 | [PDF](https://arxiv.org/pdf/2603.18184v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 514. DriftGuard: Mitigating Asynchronous Data Drift in Federated Learning

**arXiv ID:** 2603.18872 | [PDF](https://arxiv.org/pdf/2603.18872v1)

**作者:** Yizhou Han `[一作]` (University of St Andrews), Blesson Varghese `[通讯]` (University of St Andrews)

**通讯引用:** 4261 | [OpenAlex ID](https://openalex.org/A5074563743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个名为 DriftGuard 的联邦持续学习框架，针对异步数据漂移问题，通过 Mixture-of-Experts（MoE）架构将全局共享参数与局部组特定参数分离，实现高效的全局与组级模型重训练。

**💡 创新点**

创新点在于：①利用 MoE 的门控输出对设备进行无监督聚类，自动识别拥有相似数据分布的设备组；②将全局与组特定参数解耦，形成双层重训练策略（全局重训练仅更新共享参数，组级重训练仅更新局部参数），从而在保持高准确率的同时显著降低重训练成本；③通过阈值触发机制平衡重训练频率与系统开销。

**🔧 技术方法**

主要技术包括：Mixture-of-Experts 神经网络架构、层级聚类（基于门控矩阵的相似度）、Federated Averaging (FedAvg) 以及持续学习中的局部微调；评估时还使用 FLOPs 计数来量化计算成本。

**📊 数据集**

实验使用了三大多域图像分类数据集：DG5、PACS 和 DomainNet，并在 ResNet 与 ViT 两类模型上分别构建了小（S）与中等（M）规模的 MoE 变体（cResNet‑S/M、cViT‑S/M）。

**📈 对比分析**

与五个基线（经典 FCL、按设备/平均触发的 FCL、两种 PFL、聚类 FL）对比，DriftGuard 在六种模型/数据集组合中实现了最高或相近的平均准确率，同时每单位重训练成本（ℰ）的提升达到 2.3×；总重训练成本下降可达 50%（数值未给出），在 20 台 Raspberry Pi 原型上，重训练时间比最优基线低 20%，ℰ 提升 1.2×。

**⚠️ 局限性**

局限性包括：对阈值（全局触发阈值、聚类距离阈值）的敏感性，需要手动调参；在组规模过小的情况下，局部重训练效果有限；门控输出的通信开销在极大规模设备集群中可能成为瓶颈；实验仅覆盖图像分类任务，尚未验证在其他任务或非图像领域的泛化能力。

---

## 515. SwiftGS: Episodic Priors for Immediate Satellite Surface Recovery

**arXiv ID:** 2603.18634 | [PDF](https://arxiv.org/pdf/2603.18634v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon James Fong `[通讯]` (University of Macau)

**通讯引用:** 11997 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本工作提出 SwiftGS，一种利用元学习的卫星多时相多视角三维重建框架，能够在零样本模式下通过一次前向推理生成高度精确的 DSM 与一致的渲染；

**💡 创新点**

创新点包括：① 将解耦的几何‑辐射高斯原语与隐式 SDF 通过学习门控混合，兼顾高频细节与全局拓扑；② 构建可微物理图（投影、照明、大气与传感器响应），实现端到端的光度与几何监督；③ 采用基于几何教师的情节元学习，提取可迁移的先验并在推理时仅需极少的本地校准；④ 动态专家路由与多语义‑几何融合，提升计算效率与适配性；

**🔧 技术方法**

技术手段包括 Gaussian splatting、隐式 SDF、可微物理渲染图、MAML‑风格情节元学习、MVS 监督教师、动态路由、语义‑几何双向融合、条件轻量化任务头与可微门控；

**📊 数据集**

使用数据集：DFC2019 与 IARPA 3D Mapping Challenge，均包含多时相 WorldView‑3 高分辨率影像、RPC 模型及太阳元数据；

**📈 对比分析**

与 EO‑NeRF、SAT‑NGP、Sat‑Mesh、S2P、EOGS、SkySplat 等基线方法进行对比，SwiftGS 在 DSM MAE（无遮挡 1.22 m，叶面遮挡 0.82 m）和 LPIPS（0.105）上实现最低误差；推理时间仅 2.5 min，显著快于传统每场景优化方法（数小时），同时保持或提升几何与渲染质量；

**⚠️ 局限性**

局限性包括：对 RPC 与传感器元数据的依赖在极端误差下可能影响精度；当前仅处理静态场景，无法实时跟踪动态对象；模型参数量仍较大，未集成多传感器（SAR/光谱）信息，且对极端光照/极大视角差异的鲁棒性待进一步提升。

---

## 516. Words at Play: Benchmarking Audio Pun Understanding in Large Audio-Language Models

**arXiv ID:** 2603.18678 | [PDF](https://arxiv.org/pdf/2603.18678v1)

**作者:** Yuchen Su `[一作]` (University of Auckland), Michael Witbrock `[通讯]` (University of Auckland)

**通讯引用:** 3535 | [OpenAlex ID](https://openalex.org/A5057995059)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出APUN-Bench，构建包含4,434条音频双关语样本的评估基准，涵盖识别、定位与含义推理三阶段；

**💡 创新点**

首创专门针对音频双关语的基准与数据集，结合合成与真实语音，设计多阶段细粒度评估及错误分析；

**🔧 技术方法**

使用Parler‑TTS合成语音、Whisper‑Large ASR与Claude‑Opus进行标注；评估10个大型音频语言模型和4个串联系统，采用精度、召回、F1、位置相似度等指标；

**📊 数据集**

基于SemEval 2017文本双关语数据合成语音，结合O. Henry Museum Pun‑Off等公开视频和手工录音的真实语音，共4,434条样本；

**📈 对比分析**

在三阶段评估中对10个LALMs和4个串联模型进行对比，闭源模型表现最佳；识别召回率低，定位精度差，推理尤其在异写/异音双关上表现不佳，整体性能仍不理想；

**⚠️ 局限性**

未覆盖递归/复合双关，样本仅单句，真实语料规模有限，未考虑多轮对话等更复杂情境。

---

## 517. Best-of-Both-Worlds Multi-Dueling Bandits: Unified Algorithms for Stochastic and Adversarial Preferences under Condorcet and Borda Objectives

**arXiv ID:** 2603.18972 | [PDF](https://arxiv.org/pdf/2603.18972v1)

**作者:** S. Akash `[一作]` (Indian Institute of Technology Patna), Jawar Singh `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 2445 | [OpenAlex ID](https://openalex.org/A5043774561)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种黑盒化的多倍淘汰算法，能够在未知的随机或对抗环境下同时实现Condorcet和Borda目标的最佳性能。

**💡 创新点**

创新点在于：①将多倍对决问题降维为标准对决，利用复合偏置重缩放恢复无偏信息；②在Borda设置中首次实现了随机-对抗混合策略，能够在没有先验的情况下自动切换；③给出了匹配的下界，证明在两种目标下均达到理论最优。

**🔧 技术方法**

主要技术包括：黑盒化减法、偏置重缩放（αm、βm）、无偏估计、渐进无偏成功消除、马尔可夫偏差检测、EXP3无偏加权估计，以及高阶概率与伪回报分析。

**📊 数据集**

文章为理论分析性质，未在真实数据集上实验，主要使用合成的随机和对抗偏好矩阵进行理论验证。

**📈 对比分析**

与现有工作相比，本文在随机环境下实现了实例最优的O(∑i≠a⋆logT/Δi)伪回报，在对抗环境下实现了O(√(KT))的伪回报；在Borda目标下，随机回报为O(K^2logKT+Klog^2T+∑i:Δi>0KlogKT/(Δi)^2)，对抗回报为O(K√(TlogKT)+K^{1/3}T^{2/3}(logK)^{1/3})，均匹配或接近已知下界。

**⚠️ 局限性**

主要限制包括：①依赖于“赢家仅反馈”模型，无法利用更丰富的多位排序信息；②在Borda模式下需进行O(K^2)的探索，导致在大K或短T时开销较大；③无法处理无Condorcet情况或动态变化的最佳答案；④仅给出伪回报的期望界限，未给出高概率下的保证。

---

## 518. Balancing the Reasoning Load: Difficulty-Differentiated Policy Optimization with Length Redistribution for Efficient and Robust Reinforcement Learning

**arXiv ID:** 2603.18533 | [PDF](https://arxiv.org/pdf/2603.18533v1)

**作者:** Yinan Xia `[一作]` (Kuaishou Technology), Huiming Wang `[通讯]` (Kuaishou Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大模型的过度推理（overthinking）和过度自信（overconfidence）问题，提出了 Difficulty‑Differentiated Policy Optimization (DDPO)，通过将任务按难度区分，分别对易任务缩短输出长度、对难任务延长探索空间，从而在保持甚至提升准确率的同时，显著压缩答案长度。

**💡 创新点**

创新点在于：①基于过度自信现象对任务进行难度划分；②从理论上证明期望长度应逼近单一最优长度且分布越集中越好；③用同难度样本的平均正确长度来近似最优长度，并在奖励函数中加入难度相关的长度惩罚/奖励，从而实现自适应长度优化。

**🔧 技术方法**

主要技术包括：强化学习（GRPO 与其变体）、奖励设计（长度惩罚/奖励、上限/下限约束）、长度归一化与可变权重、使用多轮采样的群体相对策略优化，以及在 Qwen3-14B 上进行的 RL 微调。

**📊 数据集**

数据集涵盖：
- 领域内：OlympiadBench、MATH500、AMC、AIME2025、AIME2024、GPQA Diamond；
- 领域外：MathQA、MMLU-Pro、BBEH；
- 还在 Qwen3-8B 与 Qwen3-4B 上验证模型规模可扩展性。

**📈 对比分析**

与 GRPO、L1、ShorterBetter、A‑DLP、GRPO‑LEAD、DAPO、DrGRPO 等方法对比，DDPO 在 6 个基准上平均提升 1.85% 准确率，同时平均压缩 12% 输出长度；在 AIME2024/2025 等难度高任务中尤为突出，且在领域外数据集上也保持了更好的通用性。

**⚠️ 局限性**

局限性：
- 对难度阈值 (θ) 的选择依赖经验，需针对不同任务调优；
- 仅在数学推理类任务上验证，其他领域如自然语言理解、代码生成的适用性尚未深入；
- 需要额外的 RL 训练与多轮采样，计算成本相对较高。

---

## 519. DiscoPhon: Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units

**arXiv ID:** 2603.18612 | [PDF](https://arxiv.org/pdf/2603.18612v1)

**作者:** Maxime Poli `[一作]` (LSCP, ENS, EHESS, CNRS, PSL University), Emmanuel Dupoux `[通讯]` (LSCP, ENS, EHESS, CNRS, PSL University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多语种基准，用于评估从离散语音单元中无监督发现音素。

**💡 创新点**

首次将音素发现任务标准化为多语言基准，并设立了多对一与一对一两条轨道，明确了评估指标与限制。

**🔧 技术方法**

采用自监督语音表示学习框架HuBERT和SpidR，并结合K‑means聚类或预测头生成离散单元。

**📊 数据集**

使用12种语言（6+6），每种语言10小时未标注语音数据，覆盖从德语、英语到泰语、日语等多类型音素。

**📈 对比分析**

通过PNMI、PER、R‑值、F1和ABX等指标对四个预训练基线进行比较，SpidR在多对一与一对一任务上普遍优于HuBERT，性能在不同语言间差异显著。

**⚠️ 局限性**

局限包括语言覆盖有限、缺乏声门闭塞音、点击音及语调等极端对立，且对细粒度语音特征的捕捉仍不充分。

---

## 520. Breaking Hard Isomorphism Benchmarks with DRESS

**arXiv ID:** 2603.18582 | [PDF](https://arxiv.org/pdf/2603.18582v1)

**作者:** Eduar Castrillo Velilla `[一作]` `[通讯]`, Eduar Castrillo Velilla

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了单层顶点删除的Δ-DRESS算法，并在极其严苛的图同构基准上进行大规模实验。

**💡 创新点**

创新点在于证明单一顶点删除足以在所有测试的强正则图族中实现唯一指纹，突破了3-WL的理论边界，并提出了多重性签名以消除数值误差导致的冲突。

**🔧 技术方法**

技术手段为：基于DRESS连续动力学的边值固定点，结合所有顶点删除子图的直方图和多重性签名，构成无参数的图指纹。

**📊 数据集**

实验使用了51,718个强正则图（Spence集合12族+4族额外）与102个合成难点图（如Miyazaki、Chang等），总计51,816个独立实例。

**📈 对比分析**

与传统的1-WL/2-WL/3-WL以及4-FWL等方法对比，Δ-DRESS在所有基准族中实现了100%同族区分，时间复杂度为O(n·I·m·d_max)，在稀疏图上可降至O(I·n^2)，内存占用仅为O(m+B+n)，比基于三元组的4-WL实现更节省存储。

**⚠️ 局限性**

局限性包括：只能证明非同构（单向检验），数值精度导致直方图冲突（已通过多重性签名缓解），对CFI(K5)等4-WL级别难点仍失效，且未覆盖所有可能的难点族。

---

## 521. When Names Change Verdicts: Intervention Consistency Reveals Systematic Bias in LLM Decision-Making

**arXiv ID:** 2603.18530 | [PDF](https://arxiv.org/pdf/2603.18530v1)

**作者:** Abhinaba Basu `[一作]` (Indian Institute of Information Technology), Pavan Chakraborty `[通讯]` (Indian Institute of Information Technology)

**通讯引用:** 1509 | [OpenAlex ID](https://openalex.org/A5023091561)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ICE‑GUARD 框架，利用干预一致性检测 LLM 在高风险决策中的非决定性特征依赖（身份、权威、框架）并构建 3,000 条多领域情景基准；

**💡 创新点**

创新点在于①系统化检测三类认知偏差；②引入结构化分解（提取‑决策）作为缓解策略；③实现基于 ICE 的检测‑诊断‑缓解‑验证循环；

**🔧 技术方法**

核心技术包括干预一致性（intervention consistency）原理、结构化 JSON 提取、确定性规则推理、ICE 随机化检验与 FDR 校正；

**📊 数据集**

使用人工设计与 LLM 辅助生成的 3,000 条决策情景，覆盖 10 个高风险领域，并对 ProPublica COMPAS 数据做真实性验证；

**📈 对比分析**

与 11 种指令微调 LLM 进行比较，发现权威与框架偏差平均 5.8%/5.0%，高于身份偏差 2.2%；结构化分解可将偏差减少至 0%（最多 100%）并通过 ICE 循环实现累计 78% 的削减；

**⚠️ 局限性**

局限性包括基准为合成情景（虽与实际系统重叠度高），仅针对英语；结构化分解需要领域特定规则，难以处理规则外的复杂案例；

---

## 522. Quantitative Introspection in Language Models: Tracking Internal States Across Conversation

**arXiv ID:** 2603.18893 | [PDF](https://arxiv.org/pdf/2603.18893v1)

**作者:** Nicolas Martorell `[一作]` `[通讯]` (University of Buenos Aires), Nicolas Martorell (University of Buenos Aires)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型（LLM）在多轮对话中对情感内部状态的数值自我报告能力，并验证其与线性探针（probe）测得的内部向量的一致性与因果关系。

**💡 创新点**

创新点包括：①将人类心理学的自我报告方法引入LLM，提出基于logit加权的连续自我报告方案；②通过激活调节（steering）证明自我报告与内部状态之间的因果关系；③展示跨概念调节可显著提升内省精度；④公开了实现探针训练、激活调节与自我报告收集的开源库。

**🔧 技术方法**

使用技术：线性概念探针、logit基数值自我报告、激活调节、Spearman相关、等距R²、混合效应模型、聚类自举、Benjamini–Hochberg校正。

**📊 数据集**

数据集：40条10轮对话（共400个自我报告点），使用Gemini 2.5 Flash模拟用户；情境共40个日常场景；四个情感概念对（wellbeing、interest、focus、impulsivity）对应的对比式训练问题与评估文本。

**📈 对比分析**

评估方法：与随机方向控制比较、不同模型尺寸（1B、3B、8B）及模型族（LLaMA、Gemma、Qwen）实验；性能表现为：logit自我报告与探针的Spearman相关 0.40–0.76，等距R² 0.12–0.54；在8B模型中达到近乎1的相关；激活调节导致自我报告随α单调变化；跨概念调节可将R²提升至0.76。

**⚠️ 局限性**

局限性：仅评估四个情感概念，模型规模有限；探针质量在不同模型间波动；自我报告依赖logit加权，仍可能被训练样本或提示策略影响；对话由模拟用户生成，缺乏真实人类交互；未验证更大模型或更多概念；仅使用线性探针，可能忽略非线性内部结构。

---

## 523. Tursio Database Search: How far are we from ChatGPT?

**arXiv ID:** 2603.18835 | [PDF](https://arxiv.org/pdf/2603.18835v1)

**作者:** Sulbha Jain `[一作]` (Independent Consultant), Alekh Jindal `[通讯]` (Tursio)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了Tursio数据库搜索平台的搜索体验，并与ChatGPT、Perplexity进行对比。

**💡 创新点**

提出了一套端到端的评估框架，利用LLM生成多难度业务查询并通过LLM‑as‑judge衡量答案相关性、完整性等多维度指标，填补了结构化数据库搜索评估的空白。

**🔧 技术方法**

使用GPT‑4o‑mini生成问题、GPT‑4.1做判分、DeepEval评估，结合语义知识图、上下文图推理以及LLM生成映射等技术。

**📊 数据集**

采用Symitar核心银行架构的企业数据库，构造了13类业务角色与KPI的150个合成问题，并映射为公开域等价问题，基准对比参考BIRD、BEAVER等数据集。

**📈 对比分析**

通过在简单/中等/难度三组问题上计算答案相关性成功率并做卡方检验，Tursio的相关性与两大基线无显著差异，平均约90%+；在对话质量指标上ChatGPT表现最佳，Perplexity相对较低。

**⚠️ 局限性**

局限包括：单轮QA且固定3–5句响应长度导致完整性评估偏差；LLM‑as‑judge可能存在主观偏差；数据库缺失数据是主要瓶颈；评估流程仍需人工操作，难以实现大规模自动化。

---

## 524. OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation

**arXiv ID:** 2603.18697 | [PDF](https://arxiv.org/pdf/2603.18697v1)

**作者:** Chen Sun `[一作]`, Pinghua Gong `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Orthogonal Constrained Projection (OCP) 方法，通过在 Stiefel 流形上约束 Item‑ID embedding 的投影矩阵，解决稀疏规模下 embedding 退化问题；

**💡 创新点**

在 Item‑ID embedding 上引入 QR 正交化投影约束，保持梯度正交性和高秩性，并使用 Singular Entropy 衡量表示等方性，从而在稠密与稀疏规模下均实现显著性能提升；

**🔧 技术方法**

利用 Stiefel 流形上的 QR 重排投影、Singular Entropy 度量、Transformer 交互层、JD.com 生产推荐模型（OxygenREC 与排名模型）以及大规模稀疏 embedding 预训练；

**📊 数据集**

使用 JD.com 电商业务内部数据，生成检索模型训练 6.4B 样本，排名模型训练 2.9B 样本，词表规模从 1 亿到 10 亿；

**📈 对比分析**

与基线模型（无 OCP）在离线 Hit@k、AUC/GAUC 以及在线 UCXR、GMV 等指标对比，OCP 在扩大词表时保持更低 loss，Hit@1 提升约 0.5%，在线业务提升 12.97% UCXR、8.9% GMV，整体性能显著优于基线；

**⚠️ 局限性**

仅采用 QR 重排方案，算力开销虽小但仍存在；未探索其他流形优化器或动态词表增删；缺乏在更强分布迁移（如季节性、冷启动）下的长期评估。

---

## 525. An Agentic System for Schema Aware NL2SQL Generation

**arXiv ID:** 2603.18018 | [PDF](https://arxiv.org/pdf/2603.18018v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 526. DriveTok: 3D Driving Scene Tokenization for Unified Multi-View Reconstruction and Understanding

**arXiv ID:** 2603.19219 | [PDF](https://arxiv.org/pdf/2603.19219v1)

**作者:** Dong Zhuo `[一作]` (Tsinghua University), Jiwen Lu `[通讯]` (Tsinghua University)

**通讯引用:** 28998 | [OpenAlex ID](https://openalex.org/A5100460385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DriveTok，一个面向多视角驾驶场景的统一 3D 场景分词器，用于同时实现图像重建、深度预测、语义分割和 3D 占用预测。

**💡 创新点**

创新点在于：1) 通过视觉基础模型提取语义特征后，用 3D 可变形交叉注意力将多视角特征投射到统一 3D 网格，生成固定数量的场景 token；2) 引入可视化引导的多视角 Transformer，使场景 token 与视图 token 仅在物理可见区域交互；3) 采用多任务联合训练（重建、深度、语义、占用和语义正则），让 token 同时携带纹理、语义与几何信息。

**🔧 技术方法**

技术包括：预训练的 DINOv3‑ViT+FPN 作为图像编码器；3D deformable cross‑attention 投射特征到 BEV 网格；ViT‑Base 结构的可视化多视角 Transformer；DPT‑style 解码器用于图像空间任务；3D 占用头；FlashAttention、BFloat16、AdamW 等训练技巧。

**📊 数据集**

主要使用 nuScenes 数据集（6 镜头四周视角），并利用 MoGe-2 生成稠密伪深度、LiDARSeg 投影标签及 SurroundOcc 的占用标签。

**📈 对比分析**

与 VQGAN、ViT‑VQGAN、FlowMo、BEV‑VAE 等传统图像 tokenizer 以及多视角深度预测器（SurroundDepth、OmniNWM 等）和占用预测模型（BEVFormer、GaussianFormer 等）对比。DriveTok 在图像重建 PSNR/SSIM 与 VQGAN 差距不大；在深度预测上绝对相对误差降至 0.08、δ<1.25 达 0.93，显著优于其他方法；在 3D 占用预测上 mIoU 最高 33.32，性能与现有最优模型持平。

**⚠️ 局限性**

局限性包括：1) 仅在静态单帧场景中训练，缺乏时间序列建模；2) 依赖稀疏 LiDAR 投影标签，无法获得全像素监督；3) 对高动态场景的可见性约束可能导致遮挡错误；4) 模型参数约 280M，仍相对较大，推理延迟与显存占用高。

---

## 527. CAPSUL: A Comprehensive Human Protein Benchmark for Subcellular Localization

**arXiv ID:** 2603.18571 | [PDF](https://arxiv.org/pdf/2603.18571v1)

**作者:** Yicheng Hu `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8187 | [OpenAlex ID](https://openalex.org/A5051925942)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建CAPSUL基准，整合AlphaFold2预测的3D结构与细粒度子细胞定位标签，并在此数据集上评估多种序列与结构模型。

**💡 创新点**

首次将三维结构信息与20类细粒度定位标签相结合，提出CAPSUL并展示结构模型在定位任务中的显著优势，探讨重权重与单标签策略以缓解样本不平衡。

**🔧 技术方法**

利用AlphaFold2、FoldSeek、图卷积网络（CDConv、GearNet-Edge）、图Transformer/Graph Mamba/Graph Diffusion、ESM-1b/ESM-2/ESM-C等序列与结构学习技术，以及对比学习与融合策略。

**📊 数据集**

CAPSUL（20,181人类蛋白）及对照集DeepLoc、setHARD等公开数据集。

**📈 对比分析**

在CAPSUL上对比序列、结构及融合模型，结构模型表现优于纯序列模型但仍略低于预训练ESM；重权重与单标签方法显著提升少数类F1，整体微平均F1在0.4–0.5之间。

**⚠️ 局限性**

数据仍存在严重类别不平衡，部分定位缺乏足够样本；3D结构仅为预测，缺乏实验验证，可能影响模型泛化与解释性。

---

## 528. LEO-based Carrier-Phase Positioning for 6G: Design Insights and Comparison with GNSS

**arXiv ID:** 2603.18360 | [PDF](https://arxiv.org/pdf/2603.18360v1)

**作者:** Harish K. Dureppagari `[一作]` (Virginia Tech), Harpreet S. Dhillon `[通讯]` (Virginia Tech)

**通讯引用:** 8731 | [OpenAlex ID](https://openalex.org/A5064063671)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

研究了LEO卫星NR‑NTN系统中联合延迟和载波相位定位方法，并与GNSS进行对比。

**💡 创新点**

提出了双波形设计（宽带PRS+连续窄带载波）来解决载波相位跟踪与周期性PRS间断的问题，并证明LEO可在几秒内实现厘米级定位。

**🔧 技术方法**

采用了载波相位与延迟测量的联合处理、条件数分析、整数模糊解析、递归加权最小二乘定位等技术。

**📊 数据集**

使用了基于真实LEO轨道动态的仿真框架（600 km轨道、30轨道、28卫星/轨道，GNSS模拟），并结合CRLB误差模型。

**📈 对比分析**

通过多epoch测量、整数模糊收敛时间和定位误差比较，LEO在3 秒内实现厘米级精度，而GNSS仅能实现米级；在仅使用延迟测量时LEO与GNSS相差不大。

**⚠️ 局限性**

主要限制包括对载波相位连续性、频率误差和周期滑移的假设、单静态场景、未考虑多路径、时变信道、实际硬件实现以及需要进一步的标准化支持。

---

## 529. VGGT-360: Geometry-Consistent Zero-Shot Panoramic Depth Estimation

**arXiv ID:** 2603.18943 | [PDF](https://arxiv.org/pdf/2603.18943v1)

**作者:** Jiayi Yuan `[一作]` (Singapore University of Technology and Design), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 8644 | [OpenAlex ID](https://openalex.org/A5040897632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 VGGT-360，一个训练无需求、几何一致的全景深度估计框架，利用 VGGT-like 3D 基础模型对多视角图像进行全局三维推理并回投影到 ERP 图像。

**💡 创新点**

创新点包括：① 将全景深度任务重新表述为从全局一致 3D 模型的视角重投影；② 设计不确定性引导自适应投影、结构显著性增强注意力以及相关权重 3D 模型校正三大插件，实现无训练的几何一致推理；③ 在全景与视角之间实现无缝投影与融合，显著提升跨视角一致性与结构保留。

**🔧 技术方法**

使用技术：VGGT / π^3 / Fastvggt 3D foundation models、基于 Sobel 的梯度不确定性评分、结构显著性 confidence map、加权注意力机制、相关性（尖锐度、局部性、对称性）权重的 3D 模型校正，以及 ERP-视角投影与回投影。

**📊 数据集**

评测数据集：Matterport3D、Stanford2D3D、Replica360-2K（有标注）以及 OmniPhotos（无标注，用于可视化）。

**📈 对比分析**

与训练无需求方法（360MD、HDE360、RPG360）和有监督方法（BiFuse、UniFuse、HoHoNet、Depth Anywhere、DAC 等）对比，VGGT-360 在 Matterport3D、Stanford2D3D、Replica360-2K 上实现了最佳或最接近最佳的 Abs Rel、δ1、δ2、δ3 指标，提升幅度可达 27–36%，同时在高分辨率和零样本场景表现优异。

**⚠️ 局限性**

局限性：对极端光照或纹理稀疏区域仍可能产生误差；投影参数需手工调节，可能不适用于所有畸变情况；计算量相对单视角方法更高；未在大规模户外或动态场景进行全面评估；对非 ERP 投影的适应性尚未充分验证。

---

## 530. A Survey of Neural Network Variational Monte Carlo from a Computing Workload Characterization Perspective

**arXiv ID:** 2603.18126 | [PDF](https://arxiv.org/pdf/2603.18126v1)

**作者:** Zhengze Xiao `[一作]` (Hong Kong University of Science and Technology), Chaojian Li `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

该论文对四种主流的神经网络变分蒙特卡洛（NNVMC）模型（PauliNet、FermiNet、Psiformer、Orbformer）进行工作负载定性与GPU性能剖析，提出基于阶段的工作负载视角，分析了不同模型在采样、波函数构造、梯度与拉普拉斯评估阶段的算子混合与瓶颈；

**💡 创新点**

创新点在于将工作负载分为物理特定阶段，并统一构建剖析协议，结合算术强度、屋顶线模型和硬件利用率指标，首次揭示NNVMC在GPU上既受内存带宽限制又受细粒度算子调度影响的混合特性；

**🔧 技术方法**

采用NVIDIA Nsight Systems与Nsight Compute对GPU内核级别进行计数，计算经验算术强度并绘制屋顶线，利用正则表达式对核族进行归类，结合Tensor Core、SM内存吞吐、L2缓存命中率等硬件计数器来评估性能；

**📊 数据集**

使用常见的量子化学分子数据集（LiH、CH₄、C₂H₆、C₄H₄）以及对应的采样步长与批量大小（1024 walker），在RTX A5000、A100与H200三款GPU上进行实验；

**📈 对比分析**

与传统语言/视觉模型相比，NNVMC在相同GPU上表现出显著的内存受限特征，PauliNet/FermiNet的Laplacian重放导致细粒度低强度核占主导，Psiformer/Orbformer则在采样阶段显著提升；在A100/H200上相对A5000可获得1.1–5.8倍的加速，显示出不同模型与硬件的性能差异；

**⚠️ 局限性**

局限性包括只覆盖四种主流模型，未深入大规模分子或更高阶量子化学方法；剖析基于单机GPU，未验证跨节点或异构加速的可行性；并且缺乏针对性硬件改进方案的实验验证。

---

## 531. Perceptio: Perception Enhanced Vision Language Models via Spatial Token Generation

**arXiv ID:** 2603.18795 | [PDF](https://arxiv.org/pdf/2603.18795v1)

**作者:** Yuchen Li `[一作]` (Amazon), Garin Kessler `[通讯]` (Amazon)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5026277738)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Perceptio，一种在同一自回归序列中先生成2D语义分割 token 和3D深度 token，然后再生成文本答案的视觉语言模型。

**💡 创新点**

创新点在于：①将 2D 分割与 3D 深度两种感知信号统一嵌入语言模型的生成过程；②设计组合深度 token 生成损失（marker、token、count）和可微软合重建损失，稳定深度 token 的学习；③使用多任务联合训练，让模型同时学习感知与多种下游任务。

**🔧 技术方法**

技术上结合 InternVL‑2.5 视觉编码器、SAM2 分割器、Depth Anything V2 + VQ‑VAE 深度量化码表，利用 LoRA 微调、软重建与多任务损失。

**📊 数据集**

数据集包括 RefCOCO/RefCOCO+/RefCOCOg 约 56k 示例（配齐分割、深度、属性描述），以及 1.1M 图文对（LLaVA‑1.5、grounding、synthetic ADE20k 等）进行联合训练。

**📈 对比分析**

在 RefCOCO/+/g 的 cIoU、HardBLINK 空间推理、MMBench、SEED‑Bench 等基准上均取得 SOTA，尤其在 RefCOCO 上 82.7%、HardBLINK 均值 71.0% 以上，超越最近模型如 Sa2VA、InternVL2。

**⚠️ 局限性**

局限性包括：深度 token 的生成可能与纯文本任务产生轻微的优化冲突；仅训练与评估静态图像，未考虑视频时序一致性；模型依赖冻结的教师模型（SAM2、Depth Anything V2），其误差会传递给学生。

---

## 532. FaithSteer-BENCH: A Deployment-Aligned Stress-Testing Benchmark for Inference-Time Steering

**arXiv ID:** 2603.18329 | [PDF](https://arxiv.org/pdf/2603.18329v1)

**作者:** Zikang Ding `[一作]` (University of Electronic Science and Technology of China), Lijie Hu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5067496051)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 FaithSteer‑BENCH，一个在固定部署配置下评估推理时 Steering 的基准，结合可控性、实用性和抗压性三维度进行评估。

**💡 创新点**

创新点在于构建了统一的三闸门评估框架，并对多种 Stress 场景进行系统化设计，首次在单一 calibrated 操作点上全面检验 Steering 的可靠性。

**🔧 技术方法**

采用了激活级别的加法干预、调参校准、ACC/APC 等评价指标，并通过角色攻击、模板攻击、Base64 编码等 Stress 测试以及机制层诊断（如对齐、FOS、LDC）进行技术实现。

**📊 数据集**

使用了多种对齐相关行为数据集（Hallucination、Refusal 等）以及通用能力基准（RACE、MMLU、OpenBookQA、GLUE）进行评估，并在保留的校准集上选取操作点。

**📈 对比分析**

与四种主流 Steering 方法（CAA、PCA、TopPC、ITI）以及六大模型（Llama‑2‑7B‑Chat、Llama‑2‑13B‑Chat、Gemma‑7B、Qwen‑2.5‑7B 等）进行对比，发现大多数方法在控制性或实用性上失效，只有 Gemma‑7B 的 CAA 在所有门槛下通过，整体性能表明清洁提升并不等同于部署可靠性。

**⚠️ 局限性**

局限性在于仅考虑单一固定操作点，未覆盖所有 Steering 强度的 Pareto 前沿；Stress 场景受限于人工设计，无法完全覆盖真实部署变化；机制诊断解释可能不适用于所有模型与方法。

---

## 533. Evaluating 5W3H Structured Prompting for Intent Alignment in Human-AI Interaction

**arXiv ID:** 2603.18976 | [PDF](https://arxiv.org/pdf/2603.18976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 534. Multi-Trait Subspace Steering to Reveal the Dark Side of Human-AI Interaction

**arXiv ID:** 2603.18085 | [PDF](https://arxiv.org/pdf/2603.18085v1)

**作者:** Xin Wei Chia `[一作]` (Home Team Science and Technology Agency), Jonathan Pan `[通讯]` (Home Team Science and Technology Agency)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文通过多属性子空间调度（MultiTraitsss）构建了可产生危机相关有害行为的“黑暗”AI助手，以便系统性研究和评估人机交互中的负面心理影响。

**💡 创新点**

创新点在于将激活调度与低秩子空间投影结合，实现多属性同时调节而不显著破坏模型连贯性，并通过Pareto优化挑选最佳超参数。

**🔧 技术方法**

采用了激活调度（RepE）、SVD低秩子空间学习、LLM‑as‑Judge评估、以及基于遗传算法的保护性系统提示生成。

**📊 数据集**

使用的主要数据集包括LLMs‑Mental‑Health‑Crisis、MentalBench、AdvBench以及公开的危机对话数据集（约239,000条用户输入）。

**📈 对比分析**

实验对比显示，黑暗模型在单轮与多轮危机提示上安全分数显著低于基线（p<0.001），而连贯性保持在可接受范围；相较于基线，黑暗模型在多轮交互中逐渐恶化，且在AdvBench上安全性未受影响。

**⚠️ 局限性**

局限性包括仅适用于开放权重模型、评估主要基于自动评判器缺乏人类主观验证、数据集仍不足以覆盖所有文化/地区差异，以及所选有害属性可能不涵盖所有负面行为维度。

---

## 535. Manufacturing Micro-Patterned Surfaces with Multi-Robot Systems

**arXiv ID:** 2603.18260 | [PDF](https://arxiv.org/pdf/2603.18260v1)

**作者:** Annalisa T. Taylor `[一作]` (Northwestern University), Todd D. Murphey `[通讯]` (Northwestern University)

**通讯引用:** 3293 | [OpenAlex ID](https://openalex.org/A5067725461)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究使用多台配备压痕工具的移动机器人，利用去中心化ergodic控制算法在金属和塑料工件上生成微尺度图案，并验证其降低摩擦的物理效果。

**💡 创新点**

创新点包括：①首次将去中心化ergodic控制与轨迹历史共享相结合，实现机器人之间的任务分解；②在可扩展规模下通过多机器人协同完成微图案化；③将机器人自行生成的微图案与传统机台纹理进行对比，证实可有效降低金属表面的摩擦。

**🔧 技术方法**

采用的技术包括：ESP32S3控制板的移动机器人平台、角度轮与齿轮减小占地、压痕工具（加压-弹射机制）、去中心化ergodic控制（轨迹统计共享与平均）、控制障碍函数（安全域约束）、碰撞去相关化协议、WEDM切割小样本、旋转摩擦仪实验及图像处理（ImageJ）。

**📊 数据集**

使用的“数据集”是实验和仿真中的目标密度分布（如Balloon Dog、Club图案），以及对照组与不同密度（高、中、低）纹理的铝板样本；摩擦测试中记录的不同转速与时间序列数据。

**📈 对比分析**

通过仿真和实验对比通信与不通信两种策略，评估轨迹异质性指标、ergodic指标以及摩擦系数。结果显示：通信策略提升任务分解，尤其在复杂目标下显著；在摩擦测试中，低/中密度纹理将摩擦系数降低约两倍，表现优于未纹理的对照样本。

**⚠️ 局限性**

主要局限：系统仍依赖中心化定位（摄像头/蓝牙），未实现完全去中心化；机器人尺寸限制了单个机器人的覆盖范围；碰撞去相关化假设速度不相关，可能影响理论保证；实验规模有限，未探索多阶段工艺（打磨、涂层）与更大规模部署。

---

## 536. Adaptive Fuzzy Logic-Based Steganographic Encryption Framework: A Comprehensive Experimental Evaluation

**arXiv ID:** 2603.18105 | [PDF](https://arxiv.org/pdf/2603.18105v1)

**作者:** Aadi Joshi `[一作]`, Kavya Bhand `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于模糊推理的自适应LSB隐藏框架，结合Argon2id加AES-256-GCM实现信息加密与同步可靠的像素级深度控制。

**💡 创新点**

创新点在于用三输入三输出的27条Mamdani规则实现局部熵、边缘强度和载荷压力三因素融合的自适应嵌入深度，并通过低位剥离实现编码解码同步；同时将现代加密技术与空间域隐写结合，提升安全性。

**🔧 技术方法**

使用局部熵与Sobel边缘特征、Argon2id键派生、AES-256-GCM认证加密、Mamdani模糊推理与规则集、伪随机位序列、以及统计显著性检验等技术。

**📊 数据集**

在1000幅256×256 RGB图像（包括光滑、噪声、自然、纹理和混合五类）上进行评估，所有图像均为人工合成。

**📈 对比分析**

与固定LSB-1、固定LSB-2进行对比，覆盖0.05–0.40 bpp四个载荷水平；实验显示自适应方法在PSNR、SSIM上平均提升约3–7 dB，RS检测率显著低于固定LSB-2，且提取成功率保持100%。

**⚠️ 局限性**

主要局限包括：基于合成图像，未在真实摄影基准（如BOSSbase、BOWS-2）验证；仅使用简化的特征级别检测，未覆盖深度学习或SRM等强大检测；使用直接LSB置换，安全性仍低于现代基于成本的嵌入；对图像变换鲁棒性差；计算开销显著增加。

---

## 537. Uniform a priori bounds and error analysis for the Adam stochastic gradient descent optimization method

**arXiv ID:** 2603.18899 | [PDF](https://arxiv.org/pdf/2603.18899v1)

**作者:** Steffen Dereich `[一作]` (University of Muenster), Arnulf Jentzen `[通讯]` (University of Muenster)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文研究了Adam优化算法在强凸目标函数下的收敛性质，给出了Adam参数迭代不爆炸的先验界以及在非光滑与光滑两种情形下的收敛速率；

**💡 创新点**

创新点在于提供了无条件的爆炸性分析，说明在强凸环境下Adam的迭代参数不会发散，并给出了与学习率衰减相关的理论收敛速率，且不再依赖于传统的L‑smoothness假设；

**🔧 技术方法**

主要技术包括条件误差分析、先验界推导、凸性与二阶导数控制、随机递推方程处理、日志函数不等式与随机独立性利用；

**📊 数据集**

未使用具体实验数据集，分析以随机样本X_{n,m}为理论通用框架；

**📈 对比分析**

比较方法通过理论误差上界与学习率γ_n的关系来体现，收敛速率为O(γ_n^{1/2})（或γ_n^p）与常规梯度下降或Momentum方法的收敛速度相当；

**⚠️ 局限性**

局限性：需要满足强凸性、梯度Lipschitz或二阶导数有界、随机变量独立、学习率序列衰减等严格假设，对非凸问题及某些实际深度学习任务的适用性尚有限。

---

## 538. A more accurate rational non-commutative algorithm for multiplying 4x4 matrices using 48 multiplications

**arXiv ID:** 2603.18699 | [PDF](https://arxiv.org/pdf/2603.18699v1)

**作者:** Jean-Guillaume Dumas `[一作]` (University Grenoble Alpes), Alexandre Sedoglavic `[通讯]` (University Lille)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的 4×4 乘 4×4 矩阵乘法算法（44448），仅使用 48 次乘法，并给出了对应的直线程序。

**💡 创新点**

创新点在于通过优化张量分解，降低了 2‑范数输入下的误差增长指数从 2.628 降至 2.386，并给出了最小常数 12.09375 的复杂度上界。

**🔧 技术方法**

采用了 lrp（线性、右乘、Hadamard）矩阵分解、直线程序（SLP）生成、误差扩张因子分析以及归一化的 2‑范数误差界定技术。

**📊 数据集**

使用随机正态分布（Gaussian）生成的浮点矩阵作为实验数据集。

**📈 对比分析**

通过与经典的 O(n³) 算法、Winograd、Strassen 以及之前的 DPS25/Alt. Basis 44448 变种进行比较，实验结果显示新变种在 max‑norm 误差上优于 2227 系列算法，在近似阶数上保持最佳，同时在常数项上仅比 DPS25 稍高，整体性能优异。

**⚠️ 局限性**

限制：实现复杂度较高、常数项虽低但仍高于 2227 方案；实验仅覆盖小规模正态随机矩阵，尚未验证在更大规模或不同数据分布下的鲁棒性；仅针对包含 2 的逆的环（如实数、有理数）证明，复数域或非标准数值环境需进一步研究。

---

## 539. From ex(p) to poly: Gaussian Splatting with Polynomial Kernels

**arXiv ID:** 2603.18707 | [PDF](https://arxiv.org/pdf/2603.18707v1)

**作者:** Joerg H. Mueller `[一作]` (Huawei Technologies), Markus Steinberger `[通讯]` (Graz University of Technology)

**通讯引用:** 2275 | [OpenAlex ID](https://openalex.org/A5014594342)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出用一阶多项式+ReLU替代 3D 高斯剖面中的指数核，以兼容现有数据集并提升渲染效率。

**💡 创新点**

① 在保持数据兼容性的前提下，用低阶多项式逼近指数核；② 通过该核的有限支持实现更紧凑的 tile 剔除；③ 对多平台实现（CUDA、Vulkan、Metal）统一评估。

**🔧 技术方法**

多项式近似（ReLU-多项式）、基于 opacity 的 tile 剔除、抗锯齿归一化推导、NPU/矩阵乘法友好的实现。

**📊 数据集**

Mip-NeRF 360、Tanks & Temples、Deep Blending 三大公开数据集。

**📈 对比分析**

在 Baseline、gsplat、Faster-GS、vk_gaussian_splatting、MetalSplatter 5 种实现上做基准对比；使用 PSNR/SSIM/LPIPS 评估质量，帧时测量评估速度；结果显示一阶多项式+紧凑剔除可实现 4%–15% 的帧时提升，且视觉质量基本无差异。

**⚠️ 局限性**

① 仅一阶多项式始终稳定提升，二、三阶多项式在部分场景性能反而下降；② 更紧凑的剔除在色彩过曝区域会产生暗斑伪影；③ 对训练阶段使用新核的进一步优化未深入，导致可能的稠密度变化；④ 当前实验仅验证了 GPU/CPU，NPU 实际实现仍待探索。

---

## 540. CyberJustice Tutor: An Agentic AI Framework for Cybersecurity Learning via Think-Plan-Act Reasoning and Pedagogical Scaffolding

**arXiv ID:** 2603.18470 | [PDF](https://arxiv.org/pdf/2603.18470v1)

**作者:** Baiqiang Wang `[一作]` (University of Washington), Juan Li `[通讯]` (North Dakota State University)

**通讯引用:** 102721 | [OpenAlex ID](https://openalex.org/A5024421064)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 CyberJustice Tutor，一套基于 Agentic AI 的网络安全学习系统，结合 Think–Plan–Act 思维循环、Vygotsky ZPD 的动态教学支架以及 RAG 核心，面向刑事司法专业学习。

**💡 创新点**

创新点包括：①将 Agentic AI 的自主目标分解与纵向规划引入教育对话；②在此框架下实现基于 ZPD 的自适应支架，实时调整教学支持强度；③使用 RAG 将推理锚定在经过验证的课程材料，降低高风险领域的幻觉；④将三者整合为统一的对话系统。

**🔧 技术方法**

技术手段：Agentic AI 框架、Think–Plan–Act 认知循环、Chain‑of‑Thought 推理、RAG（检索‑增量生成）与向量数据库、LangChain、OpenAI GPT‑4o 作为核心推理引擎、Persona 设计（Senior Cybercrime Analyst）以及人机反馈循环。

**📊 数据集**

数据集：构建的内部课程知识库，包括法律法规、数字取证程序等文本，并通过向量化存储供 RAG 调用；未使用公开公开的大规模通用数据集；对系统评估采用 123 名参与者的交互日志与问卷。

**📈 对比分析**

评估方式：开放式用户研究，使用 Likert 量表与开放式反馈，统计 123 名参与者的响应速度、易用性、准确性等指标；结果显示响应速度 4.7/5、易用性 4.4/5、准确性 4.3/5，说明系统在用户体验和知识可靠性方面表现优异。

**⚠️ 局限性**

局限性：①缺乏纵向学习效果与长期保留的量化验证；②界面视觉与交互体验相对简陋；③RAG 仍面临检索覆盖范围与更新频率的挑战；④系统主要在实验环境下测试，未进行大规模控制实验。

---

## 541. Auditing Preferences for Brands and Cultures in LLMs

**arXiv ID:** 2603.18300 | [PDF](https://arxiv.org/pdf/2603.18300v1)

**作者:** Jasmine Rienecker `[一作]` (Stupid Human), Fredrik Thorsen `[通讯]` (Stupid Human)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 ChoiceEval 框架，系统地评估大型语言模型在品牌与文化偏好上的偏差。

**💡 创新点**

创新点在于：①基于心理画像生成多样化、真实感查询；②使用多专家抽取技术将自由文本转化为可比的 top‑5 建议；③采用 PEIR、LOR 等量化指标全面衡量偏好与地理差异。

**🔧 技术方法**

技术手段包括：Prompt 生成、LLM 多专家抽取（GPT‑4o）、Spearman 相关、Kruskal‑Wallis 检验、Log Odds Ratio（LOR）统计及其显著性检验。

**📊 数据集**

使用了 2070 条开放式查询，覆盖 10 个商业与文化主题（酒店、跑鞋、电动车、旅行目的地等），并对 ChatGPT‑4o、Google Gemini 1.5‑Flash 与 DeepSeek‑V3 进行对比。

**📈 对比分析**

通过比较 PEIR（首选实体出现率）与 LOR（地理偏差）等指标，发现美国模型 Gemini 与 GPT 在大多数主题上显著偏向美国实体（PEIR>70%，LOR>1且 p<0.05），DeepSeek 亦存在但幅度较小；性能上，所有模型在多次重复测试中表现高度稳定。

**⚠️ 局限性**

局限性：仅使用英文且未进行个性化；实验 IP 来自瑞典，未探究多语言或地区差异；缺乏长期版本演进追踪；只覆盖有限主题与实体，未包含低频地区与非英语品牌。

---

## 542. From Accuracy to Readiness: Metrics and Benchmarks for Human-AI Decision-Making

**arXiv ID:** 2603.18895 | [PDF](https://arxiv.org/pdf/2603.18895v1)

**作者:** Min Hun Lee `[一作]` (Singapore Management University), Min Hun Lee `[通讯]` (Singapore Management University)

**通讯引用:** 520 | [OpenAlex ID](https://openalex.org/A5020346136)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套以团队准备度为中心的人工智能协同决策评估框架。

**💡 创新点**

创新点在于将评估指标拆分为成果、依赖行为、安全信号和学习时间四大类，并将其与 Understand–Control–Improve 生命周期相结合，强调通过交互轨迹实现部署相关的可操作评估。

**🔧 技术方法**

采用交互日志分析与生命周期模型（U–C–I）相结合的方法，对人机决策过程进行量化评估。

**📊 数据集**

论文未公开具体数据集，推测使用了实验或模拟交互日志数据来验证框架。

**📈 对比分析**

与传统单纯准确率评估相比，该框架更关注团队协同的可靠性、错误恢复与治理，虽然未给出数值对比，但通过案例展示了其在安全性和可解释性上的优势。

**⚠️ 局限性**

局限包括缺乏公开数据集与大规模验证、对完整交互日志的高依赖导致可迁移性受限，以及未覆盖多任务和多模态场景。

---

## 543. HOMEY: Heuristic Object Masking with Enhanced YOLO for Property Insurance Risk Detection

**arXiv ID:** 2603.18502 | [PDF](https://arxiv.org/pdf/2603.18502v1)

**作者:** Teerapong Panboonyuen `[一作]` `[通讯]` (MARSAIL), Teerapong Panboonyuen (MARSAIL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出HOMEY框架，自动检测住宅图像中的17类房产风险；

**💡 创新点**

创新点在于启发式对象掩模与风险感知损失函数，专门针对细微、稀有类别和类不平衡进行改进；

**🔧 技术方法**

技术上基于YOLO一阶段检测器，加入域特定掩模、多尺度特征融合与自注意力、以及定制的损失函数；

**📊 数据集**

使用公开的Roboflow“Tour de Chicago”房产风险数据集，包含多种真实场景下的物业图像；

**📈 对比分析**

与YOLO12/YOLO26基线对比，HOMEY在mAP_50-95上提升至0.40，比YOLO26高约29%，在罕见类上显著提升召回率；

**⚠️ 局限性**

局限性包括对训练数据质量和多样性的高度依赖，极端噪声或极少样本仍难以准确检测，并需针对不同YOLO版本进行适配。

---

## 544. ARTEMIS: A Neuro Symbolic Framework for Economically Constrained Market Dynamics

**arXiv ID:** 2603.18107 | [PDF](https://arxiv.org/pdf/2603.18107v1)

**作者:** Rahul D Ray `[一作]` `[通讯]` (Birla Institute of Technology and Science Pilani), Rahul D Ray (Birla Institute of Technology and Science Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为ARTEMIS的神经符号框架，能够在金融时间序列预测中实现无套利约束与可解释性；

**💡 创新点**

创新点包括将连续时间的拉普拉斯神经算子与神经随机微分方程结合，并在损失中加入Feynman‑Kac PDE残差与市场风险价罚项，同时加入可微分符号瓶颈实现可解释的交易规则；

**🔧 技术方法**

技术包括拉普拉斯神经算子、神经随机微分方程、物理信息神经网络（PINN）、可微分符号回归、合成校准预测（Conformal Prediction）以及多任务联合训练；

**📊 数据集**

使用四个数据集：Jane Street匿名市场数据、Optiver的LOB波动率预测、Time‑IMM环境温度预测、以及自研的DSLOB合成崩盘情景；

**📈 对比分析**

与LSTM、Transformer、NS‑Transformer、Informer、Chronos‑2和XGBoost等六个强基线进行对比，ARTEMIS在DSLOB和Time‑IMM上实现了最高的方向性准确率（64.96%与96.0%），在其他数据集保持或略优的点估计性能；

**⚠️ 局限性**

局限性包括计算量较大、对超参数调节敏感、对长序列（如Optiver 600步）与波动率目标适应性不足，以及符号瓶颈可能导致预测精度略降。

---

## 545. HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models

**arXiv ID:** 2603.18850 | [PDF](https://arxiv.org/pdf/2603.18850v1)

**作者:** Xiangyu Bai `[一作]` (Northeastern University), Sarah Ostadabbas `[通讯]` (Northeastern University)

**通讯引用:** 2307 | [OpenAlex ID](https://openalex.org/A5031787107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了一个轻量级的帧选择策略HORNet，利用GRPO学习在冻结的视觉语言模型（VLM）输入中挑选最有价值的帧，从而提升视频问答（VQA）的答案质量并显著降低输入帧数与推理时间。

**💡 创新点**

① 将帧选择问题重新表述为“Select Any Frames”（SAF），从而与VLM解耦；② 采用GRPO直接优化VLM输入而非输出；③ 在冻结的VLM上实现超参数极少（<1M）的高效学习；④ 证明策略能跨模型迁移并提升更强VLM的表现。

**🔧 技术方法**

基于TimeSFormer轻量级视频编码器 + MLP帧选择器；使用Group Relative Policy Optimization（GRPO）进行强化学习；通过多候选帧子集与VLM生成的答案奖励（F1+编辑相似度）来估计优势。

**📊 数据集**

MSRVTT‑QA、MSVD‑QA、NExT‑QA 共计341,877个问答对，114.2小时视频；同时在VideoMME、ActivityNet‑QA等六大基准上进行评估。

**📈 对比分析**

相较于均匀采样、随机采样、PPO与监督微调，HORNet在MSVD‑QA提升1.7% F1、NExT‑QA提升7.3点，整体帧数可压缩至<1%原始量，VLM推理时间降低64–93%；在更强的VLM上还能额外提升8.5%相对准确率。

**⚠️ 局限性**

在时长极长（≈小时级）的长视频（如VideoMME）中，固定8帧的预算不足以覆盖全部语义，导致准确率显著下降；未对VLM进行微调，无法利用梯度信息进一步提升选择质量；需要更灵活的帧预算与层次化选取策略。

---

## 546. Evaluating Counterfactual Strategic Reasoning in Large Language Models

**arXiv ID:** 2603.19167 | [PDF](https://arxiv.org/pdf/2603.19167v1)

**作者:** Dimitrios Georgousis `[一作]` (National Technical University of Athens), Giorgos Stamou `[通讯]` (National Technical University of Athens)

**通讯引用:** 3112 | [OpenAlex ID](https://openalex.org/A5085359792)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个基于重复博弈（囚徒困境和石头剪刀布）的评估框架，用于检测大型语言模型在面对默认与对抗性（counterfactual）游戏变体时的战略推理与适应能力。

**💡 创新点**

创新点在于：①引入对抗性游戏对比（对行动标签或收益结构的修改），揭示模型对游戏结构的真正理解；②提出多维度评价指标（总得分、对手理解速度、合作率、效率、失败率），从多角度衡量策略表现；③通过对不同提示方式（ZS、CoT、SPP、SC）和算法对手的组合实验，系统评估LLM在不同情境下的鲁棒性。

**🔧 技术方法**

技术方法包括：使用多种LLM（Claude Sonnet 3.5/3.7/4、DeepSeek R1、Llama 3.3 70B、Mistral Large）在两人重复博弈中扮演玩家；通过自定义游戏脚本生成默认与counterfactual实例；对每轮决策收集动作并计算收益；实现“对手理解”指标以测量模型何时开始有效利用对手信息；计算效率指标（收益/令牌数）。

**📊 数据集**

数据集：论文中生成的自定义PD与RPS游戏对局记录，包含16轮PD和24轮RPS的多次重复（每个模型与算法/LLM对手各做5次或2次），以及相应的counterfactual变体。所有实验数据通过公开的GitHub仓库（https://github.com/dimjimitris/llm_gm_thesis）提供。

**📈 对比分析**

比较方法：对同一模型在不同提示、不同对手类型以及默认与counterfactual游戏下的多轮对局进行统计，计算总得分、对手理解（m值）、合作率、效率等指标。结果显示：在默认情形下，强模型（Claude 3.5/3.7、Llama 3.3）能获得高得分并快速理解对手；但在对抗性游戏中，尤其是收益结构改变时，模型表现显著下降，说明其对游戏结构的记忆式依赖；部分模型（Claude 4、DeepSeek）在对抗性变体中更为迟缓或不稳定。

**⚠️ 局限性**

局限性：①仅测试两人重复游戏，缺乏多代理、异质环境的生态效度；②使用固定对手和预设收益，可能无法捕捉更复杂的战略互作；③评价指标基于行为表现，无法完整反映模型内部推理过程；④未覆盖所有可能的提示、模型大小或训练策略，结果对其他配置可能不具普适性。

---

## 547. Literature Study on Operational Data Analytics Frameworks in Large-scale Computing Infrastructures

**arXiv ID:** 2603.19016 | [PDF](https://arxiv.org/pdf/2603.19016v1)

**作者:** Shekhar Suman `[一作]` (Vrije Universiteit Amsterdam), Alexandru Iosup `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 9020 | [OpenAlex ID](https://openalex.org/A5006986556)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大规模计算环境中的运营数据分析（ODA）框架进行系统综述，并提出一种覆盖七层体系、支持在线与按需两种模式的更完整参考架构。

**💡 创新点**

创新点在于将多层级的参考架构与现有ODA框架（如OMNI、Wintermute、ExaMon‑X等）结合，构建统一、模块化、可扩展的框架；同时引入了对硬件安全、故障预测、能源建模等多种新功能的支持。

**🔧 技术方法**

采用系统性文献回顾（Systematic Literature Review）方法，辅以Snowballing、手工搜索等技术，综合分析十余种主流OAD框架；框架实现使用的技术包括MQTT、Kafka、Elasticsearch、Prometheus、Cassandra、KairosDB等开源组件。

**📊 数据集**

使用来自NERSC、LRZ、CINECA、Fugaku等HPC系统的实时与历史监测数据集，包括传感器、功耗、温度、CPU/内存利用率、调度日志等多维度指标。

**📈 对比分析**

通过对比十个主流OAD框架在能耗、性能预测、故障检测、可扩展性等方面的指标，表明所提框架在功能覆盖度、数据整合能力和实时反馈上优于现有方案，且在多系统部署中保持良好性能。

**⚠️ 局限性**

局限性包括：仅依赖公开论文和实验室发布的系统，未包含商业系统的真实部署；数据集公开性有限，缺乏统一评估指标与大规模实测验证；框架实现仍处于设计阶段，缺乏完整的实验评估。

---

## 548. Achievable DoF Bounds for Cache-Aided Asymmetric MIMO Communications

**arXiv ID:** 2603.18240 | [PDF](https://arxiv.org/pdf/2603.18240v1)

**作者:** Mohammad NaseriTehrani `[一作]` (University of Oulu), Antti Tölli `[通讯]` (University of Oulu)

**通讯引用:** 3824 | [OpenAlex ID](https://openalex.org/A5039827493)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究异构接收天线的缓存MIMO系统，提出四种内容感知的传输策略（min‑G、Grouping、Super‑grouping、Phantom），并给出闭式的单次传输自由度（DoF）与子包化分析，揭示缓存与空间多路复用的权衡；

**💡 创新点**

创新点在于将原本对称的 coded caching 与 MIMO 融合到异构天线场景，首次设计了结合全局缓存收益与各用户空间多路复用的 Hybrid 策略，并证明其线性可解性与相对较低的子包化；

**🔧 技术方法**

采用技术包括：内容编码缓存（Coded Caching）、多天线零抑制前向（ZF）预编码、基于用户天线数的参数优化（Ω、β），以及子包化与多轮多播/单播的混合调度；

**📊 数据集**

实验使用仿真参数（如 K=35、J=2/3/5、L=16、γ=0.04、不同 G_j 组合）进行数值评估，并与传统 MU‑MIMO 基线进行对比；

**📈 对比分析**

通过与 MU‑MIMO、min‑G、Grouping 以及不同策略的比较，结果显示 Hybrid 策略（尤其是 Phantom）在 DoF 上可超过 30% 的提升，且在多种网络配置下均表现出优异的 DoF 与子包化折中；

**⚠️ 局限性**

局限性包括：子包化仍随用户数指数增长；理论假设完美 CSI 与线性可解性；实际部署中需考虑有限 SNR、硬件限制和更复杂的用户异质性。

---

## 549. Memento-Skills: Let Agents Design Agents

**arXiv ID:** 2603.18743 | [PDF](https://arxiv.org/pdf/2603.18743v1)

**作者:** Huichi Zhou `[一作]`, Jun Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Memento‑Skills 系统，使冻结 LLM 能通过可写技能库实现自我设计、学习和提升。

**💡 创新点**

将可执行技能作为外部记忆单元，结合 Read–Write 反思学习闭环与行为对齐的对比检索器，实现了无需模型微调的持续学习。

**🔧 技术方法**

采用 LLM 调用、状态化提示、读‑写反思循环、InfoNCE 对比检索、单步离线 RL、技能改进与发现、单元测试门控等技术。

**📊 数据集**

在 General AI Assistants (GAIA) 与 Humanity's Last Exam (HLE) 两大基准上进行评估。

**📈 对比分析**

与仅进行读‑写的基线对比，Memento‑Skills 在 GAIA 上从 52.3% 提升至 66.0%，在 HLE 上从 17.9% 提升至 38.7%，显示显著性能提升。

**⚠️ 局限性**

受限于技能库的规模与域对齐、检索误差以及跨任务泛化受限，且对极大规模案例的可扩展性尚未彻底验证。

---

## 550. Online Learning and Equilibrium Computation with Ranking Feedback

**arXiv ID:** 2603.19221 | [PDF](https://arxiv.org/pdf/2603.19221v1)

**作者:** Mingyang Liu `[一作]` (Massachusetts Institute of Technology), Kaiqing Zhang `[通讯]` (University of Maryland)

**通讯引用:** 3924 | [OpenAlex ID](https://openalex.org/A5047410441)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在只能获得排名反馈的在线学习和博弈平衡计算问题，并给出了硬件实例与可行算法

**💡 创新点**

提出了在非随机环境下排名反馈的下界，并在子线性总变差假设下给出可实现子线性外部遗憾的算法，进一步得到逼近粗均衡

**🔧 技术方法**

利用Plackett–Luce排名模型、滑动窗口估计、基于已有无遗憾学习算法的黑盒封装（如投影梯度、乘法权重更新、FTRL），以及策略变差控制

**📊 数据集**

实验使用大语言模型路由场景，数据集为HH‑RLHF，模型包含Qwen3‑32B、Phi‑4、GPT‑4o、Llama‑3.1‑70B，评估采用Hugging Face奖励模型

**📈 对比分析**

与基准无排名反馈方法对比，平均遗憾随时间下降，接近最佳固定模型，证明算法能快速逼近最优

**⚠️ 局限性**

对τ→0的极端情况仍有难度，未能消除bandit下的下界与正向结果间的差距，且实验仅在模拟环境，未在真实匹配/共享服务等真实数据上验证

---

## 551. An automata-based test for bricks over string algebras

**arXiv ID:** 2603.18820 | [PDF](https://arxiv.org/pdf/2603.18820v1)

**作者:** Amit Kuber `[一作]` (Indian Institute of Technology Kanpur), Annoy Sengupta `[通讯]` (Indian Institute of Technology Kanpur)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5028510244)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出多入口逆自动机（MIA）并将字符串模块映射为其基点单词的等价类，利用MIA对字母表进行局部双射得到二进制MIA，进而给出字符串代数上砖块模块的字母学表述，推广了Sturmian词与砖块模块的对应关系，并展示如何从MIA恢复原字符串代数。

**💡 创新点**

创新点在于：①引入装饰化的多入口逆自动机，能够捕捉字符串代数中的基点信息；②定义砖块词与弱砖块词，构造局部双射保持砖块性质的对应；③将砖块模块的分类问题转化为二进制字母学问题，统一并推广了对双克罗内克代数的Sturmian词表述；④给出从MIA重建字符串代数的构造方法。

**🔧 技术方法**

主要技术包括：自动机理论（多入口逆自动机与局部双射）、词语组合理论（Sturmian词、周期性与几何表示）、表示论（字符串代数、砖块模块的定义与性质）、符号映射与等价类构造。

**📊 数据集**

无实验数据；研究对象为抽象代数与符号系统，主要使用理论证明与构造。

**📈 对比分析**

本文不涉及数值实验或性能评估；主要通过严格的数学证明验证结果的正确性与普适性。

**⚠️ 局限性**

局限性：研究范围限定在字符串代数；对更一般的表象代数（如非单环或更广泛的代数）未给出直接适用的结果；且对砖块模块的算法实现与复杂度分析未给出。

---

## 552. Secure Linear Alignment of Large Language Models

**arXiv ID:** 2603.18908 | [PDF](https://arxiv.org/pdf/2603.18908v1)

**作者:** Matt Gorbett `[一作]`, Suman Jana `[通讯]` (Columbia University)

**通讯引用:** 8406 | [OpenAlex ID](https://openalex.org/A5016425387)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种只对线性对齐和分类头进行同态加密的隐私保护跨机房推理框架，并系统性评估了大语言模型间的线性对齐效果。

**💡 创新点**

创新点在于将表示相似性与同态加密结合，仅加密线性操作实现子秒延迟的安全推理；同时首次从分词器兼容性和模型规模两个维度量化跨模型文本生成的成功率。

**🔧 技术方法**

主要技术包括线性对齐（最小二乘拟合、CKA相似度）、同态加密CKKS、线性分类器、Energy score OOD检测以及LLM-as-a-Judge人机评估。

**📊 数据集**

使用了公共数据集（Wiki、IMDB）进行对齐学习；在分类上使用TREC、MNLI、DBpedia、AG News；在生成上使用MMLU、Alpaca；在隐私推理上对比GLUE任务。

**📈 对比分析**

与单模型基线和少样本训练对比，跨模型对齐在分类与OOD检测几乎无性能损失；跨模型生成在token匹配率≥0.67且模型规模≥4B时可保持60–70%原模型性能；held实现了<1 s的端到端延迟和<1 MB的通信量。

**⚠️ 局限性**

局限性包括：仅能处理线性头，无法进一步保护提供方的模型参数；跨模型生成受分词器兼容性和规模限制；攻击模型提取、会员推断等安全风险尚未覆盖；对齐映射泄露的架构信息需进一步评估。

---

## 553. Generalized Hand-Object Pose Estimation with Occlusion Awareness

**arXiv ID:** 2603.19013 | [PDF](https://arxiv.org/pdf/2603.19013v1)

**作者:** Hui Yang `[一作]` (Hunan University), Gim Hee Lee `[通讯]` (National University of Singapore)

**通讯引用:** 9546 | [OpenAlex ID](https://openalex.org/A5071967339)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 GenHOI 的框架，利用层级语义提示、多模态掩码重建和手部先验，实现对单 RGB 图像中手-物体交互姿态的通用估计，特别能在严重遮挡下保持高精度。

**💡 创新点**

创新点包括：
1) 层级语义提示，将物体状态、手部姿态和交互模式编码为多层文本描述，提供跨类别、跨场景的抽象知识；
2) 多模态掩码学习，将 RGB、点云与文本同时掩码并重建，训练模型在遮挡下进行自我修复与跨模态一致性；
3) 手部先验引导的姿态估计，将手的 MANO 参数作为稳定的空间约束，显著提升物体姿态的泛化与鲁棒性。

**🔧 技术方法**

采用的技术包括：
- 视觉-语言模型（如 InstructBLIP）生成层级文本；
- 交叉注意力（cross‑attention）实现图像、点云、文本的融合；
- 多模态掩码重建策略，结合图像噪声掩码、点云随机删块、文本 token 替换；
- 隐式几何（SDF）与显式 MANO 结合的手部姿态回归；
- 端到端的损失组合（重建、文本、点云、SDF、手姿、物体姿）。

**📊 数据集**

在 DexYCB（S0、S3）和 HO3Dv2 两大手‑物体交互数据集上进行评测，其中 DexYCB S3 用于评估对未见物体的泛化，HO3Dv2 评估在未见对象（019 pitcher base）的性能。

**📈 对比分析**

与 SOTA（HFL‑Net、UniHOPE、H2ONet 等）对比，GenHOI 在 DexYCB S3 的物体姿态 AUC 接近 90%，相较传统方法提升约 10‑15%；在手部姿态上保持与 SOTA 相近水平；在 HO3Dv2 的未见对象上获得 92.7% ADD‑0.5D，超过现有 88.4%。整体来看，GenHOI 在遮挡和跨类别泛化上均表现为最优或接近最优。

**⚠️ 局限性**

局限性包括：
- 依赖高质量的文本提示，若模板或生成文本不准确可能影响性能；
- 多模态掩码和交叉注意力计算开销较大，推理速度可能受限；
- 目前仅验证于外部视角 RGB，未针对 egocentric 或多视角场景；
- 对于极端遮挡下的几何重建仍存在误差，需进一步提升对细粒度姿态的鲁棒性。

---

## 554. SSP-SAM: SAM with Semantic-Spatial Prompt for Referring Expression Segmentation

**arXiv ID:** 2603.18086 | [PDF](https://arxiv.org/pdf/2603.18086v1)

**作者:** Wei Tang `[一作]` (Nanjing University of Science and Technology), Zechao Li `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 9573 | [OpenAlex ID](https://openalex.org/A5017096005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SSP-SAM，一个一阶段框架，利用语义‑空间提示（SSP）编码器将 CLIP 的视觉‑语言信息投射到 SAM 的掩码解码器中，以完成 Referring Expression Segmentation (RES) 以及其通用形式 (GRES)。

**💡 创新点**

创新点包括：① 视觉与语言注意力适配器对齐语义与空间信息，生成高质量的语义‑空间提示；② 通过辅助的 Referring Expression Comprehension (REC) 任务增强提示学习；③ 该框架可无改动自然处理多目标、无目标或开放词汇的情形。

**🔧 技术方法**

技术手段：Segment Anything Model (SAM)、CLIP 视觉‑文本双模编码器、视觉/语言注意力适配器、Prompt Generator（基于 Transformer Encoder）、辅助 REC 任务、Focal+Dice 以及 L1+GIoU 损失、预训练+微调策略。

**📊 数据集**

使用的主要数据集：RefCOCO、RefCOCO+、RefCOCOg、ReferIt、gRefCOCO、PhraseCut。

**📈 对比分析**

与多种现有 RES/SAM 方法（如 Prompt‑RIS、RISAM、CGFormer、ReLA、LISA、GSVA 等）在经典 RES、GRES 与开放词汇场景下的 gIoU、Pr@0.9、N‑Acc 等指标上均实现了领先或接近最优的表现，尤其在 Pr@0.9、N‑Acc 以及 PhraseCut 上取得显著提升。

**⚠️ 局限性**

局限性：对空间关键词的依赖使得在缺乏明显空间信息或表达复杂、否定结构时易误判；在多目标或极小目标场景下仍有召回不足；在未见过类别的开放词汇场景中，对极端稀缺对象的分割效果尚不理想。

---

## 555. CytoSyn: a Foundation Diffusion Model for Histopathology -- Tech Report

**arXiv ID:** 2603.18089 | [PDF](https://arxiv.org/pdf/2603.18089v1)

**作者:** Thomas Duboudin `[一作]` (Owkin), Jean-Baptiste Schiratti `[通讯]` (Owkin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发并发布了 CytoSyn——一种专为数字组织病理学设计的基础扩散模型，可生成高度逼真且多样化的 H&E 染色切片图像，并公开模型权重、训练/验证数据集及样例合成图像。

**💡 创新点**

创新点：①将 REPA‑E 架构与专用 H0‑mini 特征提取器结合，实现表示对齐与条件引导；②端到端训练 VAE 与扩散模型并使用 EMA 提升质量；③在 224×224 像素 PNG 切片上训练，避免 JPEG 压缩损失；④系统性评估方法并揭示预处理细节对 FID 与相似度的巨大影响。

**🔧 技术方法**

技术手段：Latent 扩散模型（SiT‑XL/2）、ViT‑based H0‑mini 作为对齐与引导；SD‑VAE、REPA‑E 对齐损失、EMA、SDE/ODE 采样；使用多种病理特征提取器（H‑Optimus‑0、UNI2‑h、Virchow‑2、Inception‑V3 等）计算 FID、FLD、Precision/Recall、余弦相似度等指标。

**📊 数据集**

数据集：10,622 份 TCGA 诊断 H&E 全切片（共 40M 224×224 tiles，108M 40M 为 115M 纯净 tiles），涵盖 32 种癌症类型；另外用于 OOD 验证的 SPARC IBD 3322 H&E 切片（约 32,000 tiles）。

**📈 对比分析**

比较方法：在 TCGA 的 val‑in / val‑out 验证集和 SPARC IBD OOD 集上，使用多提取器的 FD、余弦相似度、Precision/Recall 进行评估；与 PixCell‑256 对比时调整图像大小、格式和 JPEG 压缩，复现 PixCell 结果并发现其对 JPEG 的高度依赖；CytoSyn‑v2 在 val‑out FD（≈62）优于 PixCell（≈61.5），在 SPARC IBD FD（≈8.6）显著优于 PixCell（≈26.7），表现出更好的生成质量与 OOD 鲁棒性。

**⚠️ 局限性**

局限性：①对数据规模扩张的收益有限，108M 与 40M 对 FID 影响不大；②对预处理细节（JPEG、尺寸）高度敏感，导致跨实验可复现性挑战；③某些提取器下 FLD 结果波动难以解释；④未深入分析扫描器与染色差异导致 OOD 性能下降的根本原因；⑤模型仅在 TCGA 诊断切片上训练，限制了对其他组织类型的泛化。

---

## 556. TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots

**arXiv ID:** 2603.18008 | [PDF](https://arxiv.org/pdf/2603.18008v1)

**作者:** Fangrui Huang `[一作]` (Stanford University), Ehsan Adeli `[通讯]` (Stanford University)

**通讯引用:** 14358 | [OpenAlex ID](https://openalex.org/A5015355317)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 TherapyGym 框架，结合 CBT 真实性（fidelity）与安全性（safety）对话机器人进行评估与训练；

**💡 创新点**

创新点在于：①将 CBT 认证工具 CTRS 自动化为多轮对话评分；②构建安全多标签评价体系；③发布包含 116 条专家标注对话的验证集，用于校准 LLM 判断器；④将评估结果转化为 RL 奖励，实现基于临床指标的自适应训练；

**🔧 技术方法**

技术手段包括：大语言模型（GPT‑o3‑mini、Claude‑3.7 等）作为评估器与治疗师；使用 Patient‑ψ 模拟器生成对话；TreeSynth 数据增强生成 13k 病例配置；基于 GRPO 的强化学习将 CTRS 与安全评分融合为奖励；

**📊 数据集**

使用的数据集：Patient‑ψ‑CM 生成的 10 轮 CBT 对话；116 条专家标注的验证集（1,270 评分）；13,093 条扩充后病历配置；以及 20 条验证集用于人类评估；

**📈 对比分析**

与基线模型对比，RL 微调后 CTRS 平均分从 0.10 提升至 0.60（LLM 评估 0.16→0.59），安全违规率从 0.38 降至 0.20；LLM 评估器与专家的 Spearman ρ≈0.56，显示较高一致性；

**⚠️ 局限性**

局限性：仅关注 CBT，未覆盖其他疗法；评估器完全基于 LLM，存在偏差；所有对话均为合成，缺乏真实患者数据和长期临床验证；多语言能力有限；

---

## 557. Balancing Performance and Fairness in Explainable AI for Anomaly Detection in Distributed Power Plants Monitoring

**arXiv ID:** 2603.18954 | [PDF](https://arxiv.org/pdf/2603.18954v1)

**作者:** Corneille Niyonkuru `[一作]` (African Institute for Mathematical Sciences), Arnaud Nguembang Fadja `[通讯]` (University of Ferrara)

**通讯引用:** 83 | [OpenAlex ID](https://openalex.org/A5088863376)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合SMOTE+Tomek/ENN重采样、集成学习（LightGBM、XGBoost等）、SHAP可解释性和DIR公平性评估的框架，用于分布式发电厂的异常检测。

**💡 创新点**

创新点在于同时兼顾检测性能、模型可解释性和公平性，并引入MMD量化跨区域域漂移，构建可部署的实时推理服务。

**🔧 技术方法**

使用了集成学习（LightGBM、XGBoost、Random Forest等）、SMOTE+Tomek/ENN重采样、SHAP特征解释、DIR公平性度量、MMD领域差异评估，以及Docker/Kubernetes容器化部署。

**📊 数据集**

利用TeleInfra Ltd. 2017-2018年喀麦隆电信基站柴油发电机运行数据集（8,479条样本，含三类异常）。

**📈 对比分析**

与基线SVM、KNN、MLP、LR等模型对比，集成模型F1>0.95、LightGBM F1≈0.99、DIR≈0.95、MMD低，实时推理延迟<0.001秒。

**⚠️ 局限性**

局限性包括数据仅覆盖单一年份且仅局限于喀麦隆，缺乏多年多地区泛化；未采用深度时序模型；公平性评估仍依赖单一DIR指标。

---

## 558. AutORAN: LLM-driven Natural Language Programming for Agile xApp Development

**arXiv ID:** 2603.18604 | [PDF](https://arxiv.org/pdf/2603.18604v1)

**作者:** Xin Li `[一作]` (Hong Kong Polytechnic University), Yaxiong Xie `[通讯]` (University at Buffalo)

**通讯引用:** 1905 | [OpenAlex ID](https://openalex.org/A5014441699)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 AutORAN 框架，利用大语言模型（LLM）实现从自然语言意图到可直接部署的 O-RAN xApp 的端到端自动化生成；

**💡 创新点**

将需求结构化、域知识检索、分阶段生成与验证与 LLM 结合，构建完整的 xApp 开发流水线，首次实现零编码、零测试的快速 xApp 生成；

**🔧 技术方法**

采用 GPT‑4/ GPT‑4o 等 LLM，配合 LangChain、FAISS 语义检索、Chain‑of‑Thought 生成、SonarQube 静态分析、Docker+FlexRIC、srsRAN、Open5GS 等技术；

**📊 数据集**

使用 SpotLight 异常检测数据集、MobiWatch 信令/流量数据、IC 交叉干扰谱图+KPM 数据、合成切片调度数据集，以及公开 O-RAN KPI 数据；

**📈 对比分析**

与 SpotLight、MobiWatch、IC 等人工实现基线在精度、召回、F1、推理延迟和资源占用等指标上对比，AutoORAN 的 xApp 在准确率与延迟上与基线相当甚至更优，且生成时间从数小时缩短到分钟；

**⚠️ 局限性**

对复杂视觉输入（谱图）的生成仍不稳定；对 O-RAN 规范演进的适应性有限；LLM 生成需多轮验证；合成数据可能与真实环境差异；部署依赖特定容器与 RIC 环境。

---

## 559. Sharpness-Aware Minimization in Logit Space Efficiently Enhances Direct Preference Optimization

**arXiv ID:** 2603.18258 | [PDF](https://arxiv.org/pdf/2603.18258v1)

**作者:** Haocheng Luo `[一作]` (Monash University), Trung Le `[通讯]` (Monash University)

**通讯引用:** 1802 | [OpenAlex ID](https://openalex.org/A5102780660)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种只在输出层进行扰动的Sharpness‑Aware Minimization变体（logits‑SAM），用于直接偏好优化（DPO），从而缓解DPO中的压缩效应（likelihood displacement）

**💡 创新点**

通过坐标级别的参数-对数值空间动力学分析揭示负梯度更新导致残差沿高曲率方向快速扩张，并证明SAM的曲率正则化可以抑制此效应；随后将这一理论转化为仅扰动输出层的高效实现，并在多种模型和数据集上验证其有效性

**🔧 技术方法**

核心技术包括：1）DPO偏好优化；2）Sharpness‑Aware Minimization（SAM）及其logits‑SAM变体；3）理论框架：参数空间与对数值空间的几何对应，残差动力学方程；4）实验使用的Pythia‑2.8B、Mistral‑7B等大模型；5）评估工具：GPT‑5‑mini、GPT‑4‑Turbo、GPT‑4.1等自动评判器；6）超参数调优、wall‑clock时间与显存分析

**📊 数据集**

使用了Anthropic‑HH、Reddit TL;DR、UltraFeedback（binarized）三大对比数据集；在AI安全场景中亦使用SorryBench进行拒绝率测试

**📈 对比分析**

与传统DPO、SLiC‑HF、CPO等SOTA变体在Pythia‑2.8B上进行对比，logits‑SAM均提升了相对于SFT与人类首选的win‑rate；在Mistral‑7B上对开源指令跟随基准（AlpacaEval‑2、Arena‑Hard、MT‑Bench）同样获得显著增益；在AI安全实验中，logits‑SAM显著提高了拒绝率（≈9%提升）

**⚠️ 局限性**

局限性包括：1）对曲率正则化的超参数ρ需要精细调优；2）理论分析基于核/固定特征的简化假设，可能无法完整描述大模型实际训练动态；3）仅在输出层扰动，可能对极大模型或更复杂的训练流程的适用性仍待进一步验证

---

## 560. Intellectual Stewardship: Re-adapting Human Minds for Creative Knowledge Work in the Age of AI

**arXiv ID:** 2603.18117 | [PDF](https://arxiv.org/pdf/2603.18117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 561. Efficient and Versatile Quadrupedal Skating: Optimal Co-design via Reinforcement Learning and Bayesian Optimization

**arXiv ID:** 2603.18408 | [PDF](https://arxiv.org/pdf/2603.18408v1)

**作者:** Hanwen Wang `[一作]` (University of Wisconsin - Madison), Xiaobin Xiong `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了双层贝叶斯优化‑强化学习（BO‑RL）硬件‑控制协同设计框架，使四足机器人在使用被动轮滑行模式下实现高效且多样化运动。

**💡 创新点**

首次将被动轮滑行与贝叶斯优化自动搜索机械参数、强化学习训练对应控制策略相结合，自动发现最优轮向角及其对应策略，并由此衍生出“冰球式停摆”和自对齐等新颖行为。

**🔧 技术方法**

使用贝叶斯优化搜索设计空间，强化学习（PPO）训练控制策略，IsaacLab 物理仿真实现动态滑行，实验中使用OptiTrack与OpenVINS进行位姿采集。

**📊 数据集**

利用仿真中的随机速度指令集合以及在真实机器人上采集的OptiTrack/OpenVINS位姿数据，未采用公开数据集。

**📈 对比分析**

与人工设计的轮向角和基于Base‑Frame指令的RL策略进行对比，能耗（CoT）在多数方向下降约20%以上，停止时间缩短约50%，显著提升能效与机动性。

**⚠️ 局限性**

仅在平坦地面验证，缺乏对不平地形的适应；BO‑RL 循环样本效率高、耗时长；未实现行走/跑步与滑行模式的无缝集成。

---

## 562. Modeling the human lexicon under temperature variations: linguistic factors, diversity and typicality in LLM word associations

**arXiv ID:** 2603.18171 | [PDF](https://arxiv.org/pdf/2603.18171v1)

**作者:** Maria Andueza Rodriguez `[一作]`, Richard Huyghe `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型（LLMs）生成的词联想是否与人类相似，比较频率、具体度、变异性与典型性等多维度指标；

**💡 创新点**

首次系统评估温度对LLM词联想变异性与典型性的双重影响，并提出标准化典型性（SS1）度量以量化LLM答案与人类典型答案的一致程度；

**🔧 技术方法**

利用提示生成、频率统计、相对具体度计算、关联强度（S1）与标准化关联强度（SS1）等方法，对LLM生成数据进行多维度量化；

**📊 数据集**

使用SWOW‑EN人类词联想语料库（12,282词提示，490词子样本）以及三款开源LLM（Mistral‑7B、Llama‑3.1‑8B、Qwen‑2.5‑32B）在四个温度（0.3、1、1.5、2）下的100次重复生成；

**📈 对比分析**

对比指标包括：频率比、具体度比、平均变异性(#R1)、总变异性、令牌级典型性(tok‑SS1)与类型级典型性(typ‑SS1)。实验表明：人类具有最高变异性；LLM在低温下典型性高但变异性低，温度升高后变异性升高但典型性下降；大型模型Qwen在低温下表现出近似单一“原型人类”的回答，而中型模型更贴近人群多样性；

**⚠️ 局限性**

仅评估开源LLM且样本仅为490词提示，未涵盖完整SWOW‑EN；实验规模受算力限制；评价指标为定量，未结合定性人类主观评估，可能遗漏词联想的语义细微差异。

---

## 563. A Computationally Efficient Learning of Artificial Intelligence System Reliability Considering Error Propagation

**arXiv ID:** 2603.18201 | [PDF](https://arxiv.org/pdf/2603.18201v1)

**作者:** Fenglian Pan `[一作]` (University of North Carolina at Charlotte), Jian Liu `[通讯]` (University of Arizona)

**通讯引用:** 13146 | [OpenAlex ID](https://openalex.org/A5100414648)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于物理仿真平台的误差注入方法，构建了可靠性数据；并提出了显式建模错误传播的强度分解框架；同时设计了复合似然EM（CLEM）算法实现高效参数估计；

**💡 创新点**

创新点包括：①系统化的误差注入与数据采集流程；②将主误差与传播误差显式分解，并用指数衰减函数刻画传播概率；③利用分段合成似然降低EM计算复杂度，同时保留升梯性与一致性；

**🔧 技术方法**

核心技术包括：多阶段点过程建模、指数衰减传播函数、复合似然估计、Friedman检验用于自适应划分窗口数K、以及物理仿真平台中的ROS错误注入框架；

**📊 数据集**

使用了在CARLA/Autoware等仿真平台中自定义的误差注入数据；对仿真生成的误差事件数据进行实验；在两种仿真场景（持续与间歇误差注入）下进行验证；

**📈 对比分析**

与传统HPP、NHPP、Gompertz等基准可靠性模型相比，所提模型在预测误差数量时MAE更低，尤其在间歇误差注入场景下优势明显；在参数估计方面，CLEM在子窗口数K适中时能达到与EM相同的MRRMSE，同时计算时间大幅下降；

**⚠️ 局限性**

局限性：①模型对子窗口长度K的敏感性，过小或过大都会导致估计偏差或信息损失；②只考虑了最近一阶段的传播，未捕捉更远阶段间的长时依赖；③实验全部基于仿真数据，缺乏真实车载数据验证；④参数估计仍受离散化误差注入概率设定的影响。

---

## 564. Real-Time Trustworthiness Scoring for LLM Structured Outputs and Data Extraction

**arXiv ID:** 2603.18014 | [PDF](https://arxiv.org/pdf/2603.18014v1)

**作者:** Hui Wen Goh `[一作]` (Cleanlab), Jonas Mueller `[通讯]` (Cleanlab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种实时可信度评分方法 CONSTRUCT，用于评估任何 LLM 生成的结构化输出及其各字段的可信度，从而帮助企业自动化流程中快速识别错误并减少人工审核成本。

**💡 创新点**

创新点在于：①在不需要标签或模型训练的前提下，利用多模态 LLM 作为审计者（LLM-as-a-Judge）通过不同提示模板并行生成多维度中间评分；②通过谐波平均和简单算术平均对多维度评分进行无超参数聚合，兼顾全局与字段级别的错误检测；③构建了四个高质量结构化输出基准数据集，为后续评估提供可靠真值。

**🔧 技术方法**

使用的技术包括：多模态 LLM（如 GPT‑4.1‑mini、Gemini 3 Pro 等）作为审计者、LLM-as-a-Judge、结构化输出提示、谐波平均与算术平均聚合、并行调用与超时机制以控制延迟与成本。

**📊 数据集**

使用的数据集包括：Financial Entities Extraction、PII Extraction、Insurance Claims Extraction、Data Table Analysis 四个经过严格清洗和标注的结构化输出基准，覆盖文本、表格、嵌套 JSON 等多种字段类型。

**📈 对比分析**

与 token‑probability、单/多调用 LLM‑as‑a‑Judge 等传统方法相比，CONSTRUCT 在所有模型和基准上均实现了更高的 AUROC、召回率和精准率，同时显著降低了调用次数和计算成本，证明其在实际企业应用中的优越性。

**⚠️ 局限性**

主要局限：①对 LLM‑as‑a‑Judge 的依赖导致评分质量随基准模型性能变化；②多提示模板虽提高多样性，但在极大字段数或非常复杂嵌套结构时仍可能出现信息分配不均；③目前未充分验证在极端异常或多模态（如视觉+文本）结构化输出场景下的适用性。

---

## 565. On Optimizing Multimodal Jailbreaks for Spoken Language Models

**arXiv ID:** 2603.19127 | [PDF](https://arxiv.org/pdf/2603.19127v1)

**作者:** Aravind Krishnan `[一作]` (Saarland University), Dietrich Klakow `[通讯]` (Saarland University)

**通讯引用:** 4531 | [OpenAlex ID](https://openalex.org/A5008875255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种联合音频‑文本多模态越狱攻击框架（Joint Audio‑Text Multimodal Attack），通过同时优化语音的PGD扰动和文本后缀的GCG搜索，实现对语音语言模型（SLM）的梯度攻击。

**💡 创新点**

创新点在于：①首次将文本与语音两种模态的梯度优化耦合在一起，显著提升越狱成功率；②设计了基于梯度能量分布的动态优化策略，并提出了顺序近似（Sequential Audio‑Text Attack）在保持性能的同时大幅减少计算开销；③通过梯度能量、嵌入空间和分类器可分性分析揭示了多模态攻击的互补性。

**🔧 技术方法**

采用的技术包括：梯度投影法（PGD）对音频进行可感知扰动；贪婪坐标梯度（GCG）对文本后缀进行离散优化；联合优化策略；顺序近似分阶段优化；梯度能量比分析、t‑SNE嵌入可视化和线性分类器评估。

**📊 数据集**

使用了AdvBench数据集（480个测试样本），并在四种基础语音上进行攻击：有声读物、Switchboard男女对话、音乐；模型方面测试了四个支持可微音频特征提取的安全对齐SLM（如7B Instruct、E2B IT等）。

**📈 对比分析**

与单模态PGD或GCG攻击相比，联合攻击在所有模型上提升了约1.5×至10×的越狱成功率；顺序近似方法在保持相近成功率的同时，比联合攻击快4–6倍。

**⚠️ 局限性**

局限性包括：仅针对白盒可微模型，无法直接推广到黑盒或非可微模型；实验范围局限于AdvBench和所选四种语音；未评估对抗训练或防御策略的效果，未探讨多模态攻击对实际应用安全的全面影响。

---

## 566. Auditing the Auditors: Does Community-based Moderation Get It Right?

**arXiv ID:** 2603.18053 | [PDF](https://arxiv.org/pdf/2603.18053v1)

**作者:** Yeganeh Alimohammadi `[一作]` (University of Southern California), Jennifer Chayes `[通讯]` (University of California Berkeley)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在线社交平台中基于共识的审核机制如何导致投票者趋同，尤其削弱少数派声音，并提出一种两阶段加权矩阵分解的审核与聚合方法以提高预测精度；

**💡 创新点**

创新点在于首次将审核从共识一致性转移到评估者残差稳定性，并通过理论证明此策略可获得一致且低方差的估计；

**🔧 技术方法**

技术包括稀疏矩阵分解、加权最小二乘（WLS）、行为博弈模型分析及两阶段残差方差估计；

**📊 数据集**

使用了X（前Twitter）Community Notes公开数据集，包含2021-2024年的笔记和评分记录；

**📈 对比分析**

与原始基于共识的聚合算法对比，提出的方法在一周前瞻预测中均值绝对误差降低约5.7%，中位数绝对误差降低约28%，并显著提升了对争议内容的覆盖率；

**⚠️ 局限性**

局限性包括对共识机制假设的简化、缺乏对真实信息真实性的外部验证、以及两阶段算法在极端稀疏或高噪声场景下可能仍受限。

---

## 567. Dual-Model Prediction of Affective Engagement and Vocal Attractiveness from Speaker Expressiveness in Video Learning

**arXiv ID:** 2603.18758 | [PDF](https://arxiv.org/pdf/2603.18758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 568. Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models

**arXiv ID:** 2603.19233 | [PDF](https://arxiv.org/pdf/2603.19233v1)

**作者:** Bryce Grant `[一作]` (Case Western Reserve University), Peng Wang `[通讯]` (Case Western Reserve University)

**通讯引用:** 19093 | [OpenAlex ID](https://openalex.org/A5100395960)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在394,000+次机器人回放实验中，对六种规模与架构各异的Vision‑Language‑Action模型进行机制解释，结合激活注入、稀疏自编码器（SAE）和线性探针，揭示视觉通道主导动作、跨任务注入导致空间绑定的运动程序、语言依赖任务结构以及多通道模型中的运动与目标分离；并发布了Action Atlas平台供交互式探索；

**💡 创新点**

首次在跨模型规模下统一验证视觉通道优势、跨任务注入失效但轨迹对齐、语言敏感性取决于任务结构、以及专家通道与VLM通道在运动与目标语义上的功能专化；

**🔧 技术方法**

采用激活注入对比实验、Per‑token/Mean‑pool稀疏自编码器提取可解释特征、线性探针检验可线性可读性、对比选择与因果消融验证概念因果性，并通过Action Atlas实现可视化；

**📊 数据集**

使用LIBERO、MetaWorld、SimplerEnv、ALOHA四大基准，共计394,000+回放样本，覆盖多任务、不同难度与模态组合；

**📈 对比分析**

通过任务成功率、余弦相似度和覆盖率等指标比较，激活注入在null‑prompt场景下恢复率达0.997–0.999，跨任务注入导致成功率降至0%但轨迹与源任务对齐率≥99%；语言敏感性表现为任务结构决定成功率，模型间性能保持一致；

**⚠️ 局限性**

局限性包括：跨任务注入虽能重定向轨迹但无法完成任务，语言信息虽然被编码但并不总被使用，SAE需要Per‑token处理且对不同模型的泛化有限，实验以仿真为主，对真实机器人部署与在线故障诊断仍缺乏充分验证；

---

## 569. FinTradeBench: A Financial Reasoning Benchmark for LLMs

**arXiv ID:** 2603.19225 | [PDF](https://arxiv.org/pdf/2603.19225v1)

**作者:** Yogesh Agrawal `[一作]` (University of Central Florida), Aritra Dutta `[通讯]` (University of Central Florida)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5074006719)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 FinTradeBench，首个同时评估公司基本面与交易信号的金融推理基准；

**💡 创新点**

创新点在于：①将财报指标与历史价格动态整合为统一问题集；②引入 calibration‑then‑scaling 框架结合专家种子、LLM 判别与数值审计实现大规模评测；③构建双轨检索 RAG 体系与 TELeR 提示层级，支持多来源信息检索与链式推理；

**🔧 技术方法**

使用的技术包括大语言模型（14 种）、检索增强生成（RAG）与多模型候选生成、内部自过滤、数值审计、LLM 评判器对齐；

**📊 数据集**

数据集为 NASDAQ‑100 2015‑2025 年的 1,400 条问题，涵盖 SEC 10‑K/10‑Q 基本面指标与日内 OHLCV 计算的动量、波动率、回撤等交易信号；

**📈 对比分析**

通过与 14 种 LLM 在零射提示与 RAG 设置下的对比，发现 RAG 对基本面和混合问题显著提升（+37% 与 +55%），但对交易信号问题下降；总体上，具备链式思维的模型在混合推理上优于仅指令调优模型；

**⚠️ 局限性**

局限性包括：①对时序数值计算的推理能力不足，RAG 在交易信号问题上导致混淆；②模型在面对大量检索信息时易产生“信息过载”并削弱指标提取深度；③评测高度依赖预定义的金指标，无法覆盖更广泛的金融决策场景。

---

## 570. F2LLM-v2: Inclusive, Performant, and Efficient Embeddings for a Multilingual World

**arXiv ID:** 2603.19223 | [PDF](https://arxiv.org/pdf/2603.19223v1)

**作者:** Ziyin Zhang `[一作]` (Ant Group), Rui Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 24404 | [OpenAlex ID](https://openalex.org/A5028153158)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 F2LLM‑v2 这一多语言通用嵌入模型家族，包含 8 大尺寸（80M‑14B），并公开完整训练数据、代码与中间检查点。

**💡 创新点**

创新点在于：① 构建覆盖 282 种自然语言与 40 种编程语言、共 6 千万样本的公开多语言语料；② 采用两阶段训练与 Matryoshka 表示学习相结合的训练管线；③ 通过模型剪枝与知识蒸馏实现高效小模型，且不显著牺牲性能；④ 全面对 17 个 MTEB 领域（共 430 任务）进行评测，首次让 14B 模型在 11 项指标上夺得榜首。

**🔧 技术方法**

核心技术包括：基于 Qwen3 的密集 Transformer 解码器、两阶段对比学习（检索与多任务指令增强）、Matryoshka 表示学习、剪枝（按隐藏层、MLP 中间层与层数）以及 MSE 知识蒸馏。

**📊 数据集**

使用 60M 条公开高质量样本，来自 157 个来源，涵盖 282 种自然语言和 40 种编程语言；训练语料通过三种通用格式（检索、聚类、二分类）统一化；测试基准使用 MTEB 的 17 个多语言/领域子基准，包含检索、重排序、分类、聚类、对句分类、多标签分类、STS、指令重排序、双语挖掘与摘要等。

**📈 对比分析**

通过与 Qwen3‑Embedding、EmbeddingGemma 等同尺寸模型在同一基准下对比，F2LLM‑v2‑14B 在 11/17 评测中获首位；80M/160M 等小模型在 0.6B‑4B 级别上也优于现有同尺寸模型，尤其在低资源语言和代码基准上表现突出；此外，MRL 与蒸馏提升了模型压缩后的性能，兼具高效与高精度。

**⚠️ 局限性**

限制包括：① 训练与推理仍需较大算力，尤其 14B 模型；② 语料仍以公开数据为主，低资源语言覆盖虽更广但深度有限；③ 对极端小模型（<80M）未做充分评测；④ 部分任务（如极细粒度情感、复杂推理）在现有基准上尚未验证。

---

## 571. Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding

**arXiv ID:** 2603.19235 | [PDF](https://arxiv.org/pdf/2603.19235v1)

**作者:** Xianjin Wu `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 38843 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出VEGA-3D框架，将预训练视频生成模型作为Latent World Simulator，为多模态大语言模型注入隐式3D先验，解决其空间盲区。

**💡 创新点**

利用视频生成模型在去噪过程中学习的3D结构和物理规律，提取中间表示并通过自适应门控融合语义特征，首次将生成模型作为几何先验来源。

**🔧 技术方法**

基于视频扩散模型（如Wan2.1‑T2V）的中间层噪声注入、流匹配去噪、Token‑level Adaptive Gated Fusion、LLM语义编码（SigLIP/Video‑3D LLM等）。

**📊 数据集**

使用ScanRefer、Multi3DRefer、Scan2Cap、ScanQA、SQA3D（3D场景理解）、VSI‑Bench（空间推理）和LIBERO（机器人操控）等公开数据集。

**📈 对比分析**

与多种单任务、通用及空间增强基线相比，VEGA‑3D在定位、问答、视觉推理等指标均取得显著提升，常常成为排行榜前列（如ScanRefer Acc@0.5 56.2%、VSI‑Bench平均得分69.7%、LIBERO成功率97.3%）。

**⚠️ 局限性**

推理时需额外计算视频扩散模型，导致延迟增大；模型规模较大，难以在资源受限场景部署；未来需蒸馏为轻量化编码器。

---

## 572. NavTrust: Benchmarking Trustworthiness for Embodied Navigation

**arXiv ID:** 2603.19229 | [PDF](https://arxiv.org/pdf/2603.19229v1)

**作者:** Huaide Jiang `[一作]` (University of California), Jiachen Li `[通讯]` (University of California)

**通讯引用:** 24511 | [OpenAlex ID](https://openalex.org/A5070982282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 NavTrust 基准，系统性地对视觉语言导航(VLN)和目标目标导航(OGN)在 RGB 图像、深度传感器和自然语言指令的多种真实腐败（噪声、遮挡、低照度、语义掩码等）下进行评估。

**💡 创新点**

① 统一了 VLN 与 OGN 的鲁棒性评测框架；② 首次将深度传感器失真纳入基准；③ 设计了五维语言扰动（风格、大小写、掩码、黑盒/白盒攻击）；④ 对四种鲁棒性提升策略（数据增强、教师-学生蒸馏、适配器、LLM 规范化）进行系统对比；⑤ 在仿真与真实移动机器人上验证结果。

**🔧 技术方法**

采用深度图像与 RGB 图像的多种合成扰动、Transformer+图结构规划、深度感知与地图构建、教师-学生蒸馏损失、轻量级适配器、LLM 进行指令清洗与重写、以及标准的 SR/SPL/PRS 指标。

**📊 数据集**

使用 Habitat‑Matterport3D（OGN）、Room‑to‑Room (R2R) 和 Room‑Across‑Room (RxR)（VLN）数据集，构造对应的未受污染与受污染的验证集。

**📈 对比分析**

通过对七个 SOTA 模型（NaVid、Uni‑NaVid、ETPNav、WMNav、L3MVN、PSL、VLFM）在不同腐败下计算 SR、SPL、PRS，发现 VLFM 在多数扰动下保持最高 PRS；数据增强、教师蒸馏和适配器能显著提升 ETPNav 的鲁棒性；LLM 规范化对语言扰动效果最好；真实机器人实验与仿真结果高度一致。

**⚠️ 局限性**

局限性包括：仅覆盖部分视觉与深度失真；对深度融合策略的探讨有限；LLM 规范化在非英语或极端语义扰动下表现不一；基准主要聚焦 R2R/RxR 环境，未覆盖更大规模或不同室内外设置。

---

## 573. Bridging Semantic and Kinematic Conditions with Diffusion-based Discrete Motion Tokenizer

**arXiv ID:** 2603.19227 | [PDF](https://arxiv.org/pdf/2603.19227v1)

**作者:** Chenyang Gu `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 44604 | [OpenAlex ID](https://openalex.org/A5100406050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了三阶段 Perception–Planning–Control 框架，结合离散 Token 生成与扩散式解码，实现文本与轨迹双重控制的人体运动生成。

**💡 创新点**

通过 Diffusion-based Discrete Motion Tokenizer 将语义抽象与细粒度重建分离，使用单层码表实现高压缩比 token，并在解码阶段加入精细运动约束，显著降低轨迹误差并提升 FID。

**🔧 技术方法**

使用向量量化编码器、条件扩散模型、Classifier-Free Guidance、Transformer/自回归规划器、轨迹编码器等技术。

**📊 数据集**

在 HumanML3D 和 KIT-ML 文本-动作配对数据集上进行实验。

**📈 对比分析**

与 MaskControl、MoMask 等基线比较，在 HumanML3D 文本+轨迹控制下，仅使用 1/6 级别的 token 就将 FID 从 0.083 降至 0.029，轨迹误差从 0.72cm 降至 0.08cm，且在更强约束下仍能提升 FID。

**⚠️ 局限性**

受限于扩散解码的推理速度；在更高压缩率下可能丢失细节；对多模态条件的统一调度仍存在挑战。

---

## 574. MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction

**arXiv ID:** 2603.19231 | [PDF](https://arxiv.org/pdf/2603.19231v1)

**作者:** Haitian Li `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 44604 | [OpenAlex ID](https://openalex.org/A5100406050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种单目进化式结构推理框架，能够从单张图像完整地重建可关节3D物体的几何、部件分割与关节参数；

**💡 创新点**

将几何恢复、部件语义编码、运动解码与关节回归整合到同一端到端网络，去除了对多视角、检索库或视频生成的依赖，显著提升了解析稳定性与可解释性；

**🔧 技术方法**

采用 TRELLIS 3D 生成器、三平面投影+Transformer 部件语义推理、双查询运动解码器、Kinematic Estimator、triplet loss、CLIP 文字嵌入、残差迭代更新等技术；

**📊 数据集**

使用 PartNet‑Mobility 数据集（约 2K 可关节物体，涵盖 7 类和 46 类两种划分）；

**📈 对比分析**

与 URDFormer、SINGAPO、Articulate‑Anything、PhysXGen、PhysXAnything 等方法对比，均在几何（Chamfer、F‑score、PSNR、CLIP）和关节预测（类型、轴向误差、枢轴误差）上取得领先；推理时间约 20.5 秒，几乎与最快的 SINGAPO 一致，远快于其他基线；

**⚠️ 局限性**

对极小部件或尺度失衡物体的分割与参数预测效果不足；模型依赖学习到的结构先验，对新拓扑或罕见关节模式的泛化有限。

---

## 575. SAMA: Factorized Semantic Anchoring and Motion Alignment for Instruction-Guided Video Editing

**arXiv ID:** 2603.19228 | [PDF](https://arxiv.org/pdf/2603.19228v1)

**作者:** Xinyao Zhang `[一作]` (Baidu), Jingdong Wang `[通讯]` (Baidu)

**通讯引用:** 47684 | [OpenAlex ID](https://openalex.org/A5075880303)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SAMA框架，实现了基于指令的视频编辑，采用语义锚定与运动对齐的因式分解策略；

**💡 创新点**

创新点在于将语义规划与运动建模因式分解，利用语义锚定和运动对齐的预训练任务，显著降低对外部先验的依赖；

**🔧 技术方法**

采用Diffusion Transformer、流匹配训练、SigLIP语义特征预测、以及运动对齐的Cube Inpainting、Speed Perturbation、Tube Shuffle等技术；

**📊 数据集**

使用了图像编辑数据集（NHR-Edit、GPT-image-edit、X2Edit、Pico-Banana-400K）、文本-视频数据集（Koala-36M、MotionBench）和视频编辑数据集（Ditto-1M、OpenVE-3M、ReCo-Data）；

**📈 对比分析**

在VIE-Bench、OpenVE-Bench、ReCo-Bench三大基准上与多款开源与闭源模型对比，SAMA在大多数指标上超过开源基线，且与闭源系统相当；

**⚠️ 局限性**

局限性包括零-shot场景下属性一致性不足、添加物体模糊、删除操作残留鬼影，以及对长视频和极端快动作的处理仍需提升。

---

## 576. Under One Sun: Multi-Object Generative Perception of Materials and Illumination

**arXiv ID:** 2603.19226 | [PDF](https://arxiv.org/pdf/2603.19226v1)

**作者:** Nobuo Yoshii `[一作]` (Kyoto University), Ko Nishino `[通讯]` (Kyoto University)

**通讯引用:** 5831 | [OpenAlex ID](https://openalex.org/A5034253077)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在单张图像中通过多对象共照明信息，利用逆渲染技术同时推断纹理、反射率与环境光照；

**💡 创新点**

通过将多对象视为同一光照的多探针，结合多尺度协同调度、轴向注意和 ControlNet 生成多样化、物理一致的解，解决传统单对象逆渲染的不可辨识性；

**🔧 技术方法**

使用潜在扩散模型、轴向注意机制、协调调度（Coordinate Scheduling）、ControlNet 视觉一致性约束以及 Disney BRDF 等物理模型；

**📊 数据集**

在合成数据集（Mitsuba3 渲染的 Adobe 3D Assets、Xu 资产、Laval Indoor、Poly Haven HDRI）以及真实数据集（Stanford-ORB、nLMVS-Real、自己采集的 9 场景）上训练与评测；

**📈 对比分析**

与 DPI、DRM、DiffusionLight 等基线对比，实验显示 MultiGP 在照明、反射率、纹理的 logRMSE、PSNR、SSIM、LPIPS 上均优于对手，并在多目标样本分布上更好覆盖真实光照；

**⚠️ 局限性**

需已知对象几何体，且假设远场环境光，无法处理近场光照或未测量的形状。

---

