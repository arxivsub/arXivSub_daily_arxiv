# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-24 | 今日论文总数: 486

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Information-Theoretic Classifier-Free Guidance with Adaptive Schedule Optimization

**arXiv ID:** 2606.24025 | [PDF](https://arxiv.org/pdf/2606.24025v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 2. NavWM: A Unified Navigation World Model for Foresight-Driven Planning

**arXiv ID:** 2606.24101 | [PDF](https://arxiv.org/pdf/2606.24101v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 3. Geometry-Aware Style Transfer in 3D Gaussian Splatting

**arXiv ID:** 2606.24144 | [PDF](https://arxiv.org/pdf/2606.24144v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 4. KLip-PPO: A per-sample KL perspective on PPO-Clip

**arXiv ID:** 2606.23932 | [PDF](https://arxiv.org/pdf/2606.23932v1)

**作者:** Riccardo Colletti `[一作]` (University of California, Berkeley), Robin Holzinger `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文证明了PPO‑Clip与PPO‑KL在逐样本梯度上完全等价，并给出了等价的KL系数表达式；

**💡 创新点**

创新点在于将PPO‑Clip视作逐样本KL惩罚，明确了其隐含的系数形式，并基于此构建了软化、位置/年龄/不对称等可扩展的算法设计空间；

**🔧 技术方法**

主要技术包括梯度解析推导、对等价性的严谨证明，以及在MuJoCo连续控制任务上进行的实验验证；

**📊 数据集**

实验使用了MuJoCo的五个连续控制任务（HalfCheetah、Hopper、Walker2d、Ant、Humanoid）以及CartPole和LunarLander两个低维/离散基准；

**📈 对比分析**

在相同的训练配置下，比较了PPO‑Clip、固定β、适应β以及逐样本β四种损失的学习曲线和最终回报，发现逐样本KL与PPO‑Clip完全一致，而固定/适应β在高维任务上略逊；

**⚠️ 局限性**

该等价性仅在样本不处于剪裁边界的测度零集合上成立，且实验仅覆盖连续控制任务，未检验在离散或大型语言模型等其他领域的泛化。

---

## 5. Ten Digits on a Train: AI-Assisted Verification of Two Eigenvalue Problems

**arXiv ID:** 2606.23821 | [PDF](https://arxiv.org/pdf/2606.23821v1)

**作者:** Matthew J. Colbrook `[一作]` `[通讯]` (University of Cambridge), Matthew J. Colbrook (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文报告了人类与人工智能的合作，成功计算了两个特定的特征值问题，分别是一个奇异自伴随的薛定谔算子和一个非正常的原子-分子基准问题。

**💡 创新点**

创新点在于通过重新构造问题为一个全局匹配系统，而不是单纯提高单向射击的精度，从而实现了对特征值的高精度认证。

**🔧 技术方法**

使用了Krawczyk方法、区间算术和人工智能技术，特别是ChatGPT 5.5作为计算助手。

**📊 数据集**

数据集包括一个奇异的自伴随薛定谔算子和一个复杂的非正常共振对，后者的计算目标是将其分离并精确到十位数字。

**📈 对比分析**

与传统方法相比，本文的方法通过全局匹配系统和Krawczyk包含提供了更为稳健的验证框架，性能上能够有效处理不良条件的传播和不确定的渐近数据。

**⚠️ 局限性**

限制在于人工智能的辅助计算虽然快速产生了准确的候选值和合理的证明策略，但也出现了多次失败，尤其是在需要进行逐点检查的情况下，显示出AI辅助工作中的潜在错误和局限性。

---

## 6. Verifiable Foundation Models for Robot Safety

**arXiv ID:** 2606.23754 | [PDF](https://arxiv.org/pdf/2606.23754v1)

**作者:** Davide Corsi `[一作]` (University of California, Irvine), Roy Fox `[通讯]` (University of California, Irvine)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将机器人控制策略拆分为大型感知与任务推理的Controller与可验证的低维Safety Module，实现对基于大模型的控制器的正式安全保证。

**💡 创新点**

创新点在于通过模块化拆分使得基础模型的高维感知与安全决策分离，只有小型安全模块可被正式验证，并通过离线验证与运行时遮蔽实现低干扰安全保障。

**🔧 技术方法**

使用扩展的ϵ-ProVe验证工具对Safety Module进行离线安全性验证，并结合基于安全传感器的低维上下文和多种大模型后端（BERT+ViT、LLM+ViT、ViT、SmolVLA）进行策略训练。

**📊 数据集**

实验数据集涵盖三种机器人任务：2D Playground、Hello Robot Stretch 2 的室内导航和 Unitree GO2 的户外导航，每个任务均使用相应的视觉+语言+传感器观测。

**📈 对比分析**

与未拆分或无验证的基线相比，实验显示拆分后策略保持或提升任务成功率（最高100%），验证后安全模块覆盖率超过78%，遮蔽后零违规、遮蔽率低于1%，在真实 Stretch 机器人上也实现了61%成功率并无碰撞。

**⚠️ 局限性**

局限在于安全规范仅基于低维安全传感器，无法覆盖需要从高维感知推断的语义安全约束；此外验证时间在高维传感器空间会很长，且未考虑更丰富的时间或多主体安全规范。

---

## 7. MGI: Member vs Generated Inference

**arXiv ID:** 2606.23872 | [PDF](https://arxiv.org/pdf/2606.23872v1)

**作者:** Bihe Zhao `[一作]` (CISPA Helmholtz Center for Information Security), Adam Dziedzic `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的任务——成员对生成（Member vs Generated Inference，MGI），旨在区分生成模型训练集中的真实成员样本与该模型自身生成的样本；

**💡 创新点**

创新点在于：①定义了更难的MGI任务；②提出了三阶段的 Data Circuit Breaker 方法，利用自编码器一致性、成员推断与跨生成器一致性综合判别；③证明了传统成员推断与生成器归因方法在MGI上失效；

**🔧 技术方法**

主要技术包括：自编码器重构误差与量化误差判别、条件概率差异（CPD）成员推断、KDE跨生成器归因、以及自编码器与潜在生成器的联合使用；

**📊 数据集**

实验数据集涵盖 ImageNet‑1k（训练/验证分离）和 MS‑COCO（用于 Stable Diffusion 微调），使用多种生成模型：VAR、RAR‑XXL、LlamaGen‑XXL（图像自回归模型）和 Stable Diffusion 1.4/2.1（扩散模型）；

**📈 对比分析**

在直接训练和模型衍生两种设置下，与 PIAR、ICAS、CLiD、PRADA、LiRA、RMIA 等基线相比，Data Circuit Breaker 在生成与自然样本区分、成员与非成员区分以及跨模型归因方面均实现了近乎 100% 的准确率（AUC 0.97‑1.00），显著优于基线；

**⚠️ 局限性**

局限性包括：①需要完整访问目标模型的自编码器与潜在生成器，无法应用于黑盒或只公开输出的模型；②在极端生成质量差或自编码器结构差异较大的模型上效果可能下降；③对极大规模模型的计算成本仍需进一步评估。

---

## 8. DynaWM: Dynamics-Aware Distillation with World Model and Momentum Targets for Smooth Locomotion over Continuous Stairs

**arXiv ID:** 2606.24089 | [PDF](https://arxiv.org/pdf/2606.24089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 9. Human-Centered Design: The Disclosure of Generative Artificial Intelligence for Emerging Professionals

**arXiv ID:** 2606.24136 | [PDF](https://arxiv.org/pdf/2606.24136v1)

**作者:** Sydney Lee `[一作]` `[通讯]` (University of North Carolina at Charlotte), Sydney Lee (University of North Carolina at Charlotte)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过反思人本设计课程中生成式AI的应用，提出负责任的AI实践与课程设计建议。

**💡 创新点**

创新点在于将生成式AI定位为共创者，设立AI倡导者角色，强调透明度与人本原则的融合。

**🔧 技术方法**

主要使用ChatGPT/ Gemini/ Claude 等生成式AI工具，以及 Grammarly、Copilot 进行文本编辑。

**📊 数据集**

未使用公开数据集，依赖课程项目中的访谈、可用性测试等原始研究数据。

**📈 对比分析**

未进行实验比较，论文基于案例分析与理论讨论，没有量化性能评估。

**⚠️ 局限性**

局限性包括缺乏定量验证、课程规模受限、AI实用性与可泛化性待验证，以及可能导致研究者技能退化。

---

## 10. SPACE: Enabling Learning from Cross-Robot Data Toward Generalist Policies

**arXiv ID:** 2606.24049 | [PDF](https://arxiv.org/pdf/2606.24049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 11. Ingredient-Level Food Image Segmentation for Nutrition Awareness

**arXiv ID:** 2606.24059 | [PDF](https://arxiv.org/pdf/2606.24059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 12. Accelerating Multimodal Large Language Models with Prior-Corrected Token Reduction

**arXiv ID:** 2606.24156 | [PDF](https://arxiv.org/pdf/2606.24156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 13. RIFT-Bench: Dynamic Red-teaming For Agentic AI Systems

**arXiv ID:** 2606.23927 | [PDF](https://arxiv.org/pdf/2606.23927v1)

**作者:** Yarin Yerushalmi Levi `[一作]` (Fujitsu Research of Europe), Roman Vainshtein `[通讯]` (Fujitsu Research of Europe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个基于图表示的动态红队评估框架，自动抽取多种代理 AI 系统结构并生成可跨实现的攻击探针，完成了系统级安全评估。

**💡 创新点**

创新点包括：①代码驱动的分层图结构抽象，使攻击模板可跨实现迁移；②Discovery 与 Scanning 两阶段自动化流程，实现从结构识别到攻击执行的闭环；③构建 45 个异构代理系统和 105 个攻击探针的公开基准。

**🔧 技术方法**

使用了 LLM 辅助的代码分析与图构建、工具仿真、自动化攻击生成与执行、执行追踪与评估、以及基于评估器的攻击成功度量技术。

**📊 数据集**

使用了 45 个涵盖 5 个领域、不同框架与架构的代理 AI 系统（包括手工标注的 ground‑truth 结构图）和 105 个攻击探针作为实验数据集。

**📈 对比分析**

通过对比传统仅针对 prompt 的 LLM 红队，本框架在多表面、多目标评估中实现了更高的攻击激活率（AAR）和可测的攻击成功率（ASR）与执行漂移（mED）。实验表明，攻击成功率随框架和架构变化显著，验证了方法的适用性和有效性。

**⚠️ 局限性**

局限性包括：仅在白盒（可访问源码）环境下工作；依赖现有追踪/可观测栈；评估结果受评估器设计和假设的影响，可能导致偏差。

---

## 14. INSPIRE: Intent-aware Neural Sponsored Product Retrieval for E-commerce

**arXiv ID:** 2606.23889 | [PDF](https://arxiv.org/pdf/2606.23889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 15. Aquifer: Hierarchical Memory Pooling with CXL and RDMA for MicroVM Snapshots

**arXiv ID:** 2606.24079 | [PDF](https://arxiv.org/pdf/2606.24079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 16. Closing the Loop: Formally Verified Law as a Reward Signal for Self-Improving Legal AI

**arXiv ID:** 2606.23913 | [PDF](https://arxiv.org/pdf/2606.23913v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 17. Evaluating LLM Usage for Efficient and Explainable Numerical and Classified Implicit Sentiment Analysis of Product Desirability

**arXiv ID:** 2606.23701 | [PDF](https://arxiv.org/pdf/2606.23701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 18. Enforcing Human-like Kinematics in Dexterous Piano Playing via Adversarial Posture Regularization

**arXiv ID:** 2606.23848 | [PDF](https://arxiv.org/pdf/2606.23848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 19. Progressive Pixel-Neighborhood Deformable Cross-Attention for Multispectral Object Detection

**arXiv ID:** 2606.24092 | [PDF](https://arxiv.org/pdf/2606.24092v1)

**作者:** Tian Qiu `[一作]` (Jiangsu University), Xin Zuo `[通讯]` (Jiangsu University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种进阶像素邻域可变形跨模态注意力框架（PNAFusion），用于可见光与红外图像的弱对齐多光谱目标检测。

**💡 创新点**

创新点包括：① 嵌入局部像素邻域跨模态注意力（PNCA）与自适应可变形对齐（ADA）相结合，限制跨模态交互范围并实现局部非线性偏移校正；② 采用迭代反馈机制逐步细化对齐与融合，提升定位精度；③ 在保持检测精度的同时将全局跨模态注意力的二次复杂度降低至线性（O(HWk²))，显著节省显存和 FLOPs。

**🔧 技术方法**

技术实现主要包含：可变形卷积采样、局部窗口自注意力、跨模态关键字/值对齐、逐层多尺度特征提取、Iterative Pixel-Neighborhood Deformable Cross-Attention（PNDCA）模块以及 NiN 融合层。

**📊 数据集**

在 FLIR、M3FD 与 DroneVehicle 三个公开多光谱目标检测基准上进行实验。

**📈 对比分析**

与现有基线（如 ICAFusion、Fusion-Mamba、TFDet 等）对比，PNAFusion 在 YOLOv5 框架下 FLIR mAP@0.5 84.2、M3FD 90.5、DroneVehicle 85.5，使用 Co‑DETR 时进一步提升至 FLIR 86.8、M3FD 90.8。相比全局跨模态注意力，显存降低 33%，FLOPs 由 194.8G 降至 156.4G，但推理延迟略高。

**⚠️ 局限性**

局限性：① 对极小目标和强遮挡场景的检测仍有漏检与重复检测；② 由于缺乏像素级对齐标注，无法直接评估对齐偏移的绝对精度；③ 可变形采样和迭代机制在某些硬件平台上导致延迟增大，未充分针对低功耗嵌入式设备优化。

---

## 20. Beyond Mutual Information: Extension Profiles and Shape Functions of Random Variable Pairs

**arXiv ID:** 2606.23849 | [PDF](https://arxiv.org/pdf/2606.23849v1)

**作者:** Rostislav Matveev `[一作]`, Andrei Romashchenko `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究联合分布 (X,Y) 的扩展概况（extension profile）和形状函数（shape function），并把其与均匀分布在二分图上的随机变量以及图的谱性质联系起来；进一步用该框架解释 Gács–Körner 公共信息、非 Shannon 信息不等式和 MMRV 不等式的结构，并给出关于互信息可提取性的谱上界与下界。

**💡 创新点**

①首次将形状函数定义为扩展概况的 Legendre–Fenchel 变换，形成一种新的几何不变量；②证明形状函数在独立合成和链式规则下具有可加性；③利用二分图的谱分解，将扩展概况与图的二阶特征值关联，从而得到针对平衡 expander 的精确极限；④给出互信息“纠缠”（entanglement）与形状函数之间的代数关系，并在谱上界下给出 MMRV 和 Ingleton 式的新的补充不等式。

**🔧 技术方法**

几何分析（凸体、支持函数、Legendre–Fenchel 变换）、信息理论（Shannon 与非 Shannon 不等式、公共信息、双马尔可夫性）、谱图理论（混合性质、二阶特征值上限、平衡 expander 的定义）以及组合优化（凸包、Minkowski 加法）。

**📊 数据集**

通过构造大规模可接受的二分图（包括平衡 expander、右/左重的 expander 以及特殊的项目几何图），这些图的顶点数与边数被显式给出，进而得到对应随机变量的分布；对这些图进行谱分析以得到 λ₂ 上界，并利用它们构造满足所需熵属性的随机对。

**📈 对比分析**

与传统的 Shannon 型不等式相比，论文给出的形状函数上界在 λ₂ 较小（即 expander 接近正则图）时能逼近下界；与之前仅基于熵或公共信息的极限相比，新增的谱项显著缩小了可实现的扩展空间。实验上，通过平衡 expander 生成的随机对，数值计算表明形状函数与下界 _min 的差距不超过 O(log H(XY))，验证了理论预测。

**⚠️ 局限性**

1）形状函数与扩展概况的理论上界仍带有 O(log H(XY)) 的对数误差，尚不清楚能否消除；2）尚未证明存在严格意义下的刚性（rigid）对；3）论文只给出了上界，对下界的精确性在非均匀分布或非二分图结构下缺乏完整描述；4）在实际应用中，需要构造大规模 expander 图的实现成本和对数项的常数可能较大。

---

## 21. EvidenceLens: A Claim-Evidence Matrix for Auditing Financial Question Answering

**arXiv ID:** 2606.23724 | [PDF](https://arxiv.org/pdf/2606.23724v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 22. Learning to Trigger: Reinforcement Learning at the Large Hadron Collider

**arXiv ID:** 2606.23993 | [PDF](https://arxiv.org/pdf/2606.23993v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 23. Best Preprocessing Techniques for Sentiment Analysis

**arXiv ID:** 2606.24055 | [PDF](https://arxiv.org/pdf/2606.24055v1)

**作者:** Saranzaya Magsarjav `[一作]` (Adelaide University), Lewis Mitchell `[通讯]` (Adelaide University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对推特情感分析中的预处理技术顺序进行系统评估，确定最佳预处理顺序。

**💡 创新点**

首次系统比较不同预处理技术的实施顺序，并量化每一步对 F1 分数的影响。

**🔧 技术方法**

使用 Naïve Bayes、SVM、K‑means、决策树等传统分类器，结合多种预处理工具（BERT、spaCy、TweetTokenizer、SnowballStemmer 等）以及 5120 种预处理组合。

**📊 数据集**

使用三大推特数据集：US Airline、GOP Debate、SMILE。

**📈 对比分析**

采用 5 折交叉验证、平均 F1 分数和 ANOVA 统计来比较不同顺序和技术的性能；结果表明最佳顺序为 tokenization→cleaning→spelling correction→stop‑word removal→stemming，SVM 取得 90%+ F1。

**⚠️ 局限性**

局限性包括仅使用词基模型而非 BERT 等预训练模型，对预处理包的依赖，以及对其他数据集和语言的推广性有限。

---

## 24. DriveStack-VLA: Render-Teacher Alignment for BEV-Based DeepStack Vision-Language-Action Model

**arXiv ID:** 2606.24051 | [PDF](https://arxiv.org/pdf/2606.24051v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 25. RAVEN: A Regime-Aware Variable-context Expert Network for Financial Time Series Forecasting

**arXiv ID:** 2606.24062 | [PDF](https://arxiv.org/pdf/2606.24062v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 26. LMS-AR: LMS Prediction-based Adaptive Regulator for Memory Bandwidth in Multicore Systems

**arXiv ID:** 2606.23945 | [PDF](https://arxiv.org/pdf/2606.23945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

---

## 27. ModTGCN: Modularity-aware Graph Neural Networks for Text Classification

**arXiv ID:** 2606.23694 | [PDF](https://arxiv.org/pdf/2606.23694v1)

**作者:** Rajarshi Misra `[一作]` (BITS Pilani), Hari Om Aggrawal `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 ModTGCN 的模块化感知图神经网络，结合交叉熵和模数目标进行半监督文本分类。

**💡 创新点**

核心创新在于将模数优化作为辅助损失引入 GNN 训练，构建文档-文档相似图并使用混合监督（gold 与 pseudo-label），同时将原始 TextGCN 的异构图解耦为文档-词与词-词两部分，显著提升可扩展性。

**🔧 技术方法**

技术手段包括：GCN 层、Transformer 句向量（SBERT）、文档-词与词-词稀疏图、Gaussian RBF 文档-文档相似图、边权重重标记、伪标签监督与模数正则化。

**📊 数据集**

实验采用五大文本分类基准：MR、R8、R52、20Newsgroups（20NG）和 Ohsumed。

**📈 对比分析**

与 TextGCN、TensorGCN、BERTGCN、LR、Linear Probing、LLM 等基线比较，ModTGCN 在所有数据集上均获得微调后均超过 1–2% 的 Micro‑F1，尤其在低同质性数据集 Ohsumed 与 20NG 上提升显著（+3.6% 与 +11.5%）。

**⚠️ 局限性**

局限性包括：在高同质性、线性可分数据上提升有限；模数相关超参数（γ、λ 等）对性能影响显著，需经验调参；未在更大规模语料或多模态场景下验证。

---

## 28. When Retrieval Metrics Mislead: Measuring Policy Signal in Long-Horizon Tool-Use Agents

**arXiv ID:** 2606.23937 | [PDF](https://arxiv.org/pdf/2606.23937v1)

**作者:** Tianyu Ding `[一作]` (Amazon Web Services), Juan Pablo De la Cruz Weinstein `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在 τ‑bench 工具使用场景中，预行动决策分类器对检索到的政策条款的敏感度，并对比使用结构化决策状态与原始轨迹文本以及金手指（gold）政策条款与检索条款的表现。

**💡 创新点**

发现：1）结构化决策状态显著提升宏观 F1；2）即使检索条款的 exact‑match recall 极低（≈7%），其在分类器中的实际效用也与金手指条款相当，说明 exact‑match recall 是对下游政策效用的低估；3) 提出了诊断金手指注入实验与直接检索干预实验的对照框架。

**🔧 技术方法**

使用：1）Qwen2.5‑3B/7B 通过 LoRA 微调的多分类器；2）MiniLM bi‑encoder 与交叉编码 reranker 进行政策条款检索；3）对齐的结构化状态生成器（请求、证据、政策、动作类）。

**📊 数据集**

数据集：τ‑bench（airline）和 τ²‑bench（airline、retail）中的合成轨迹，包含任务级自然语言政策断言和预行动标签。

**📈 对比分析**

比较方法：在相同训练配置下，对比四种输入表示（raw、masked、structured、raw+policy）以及四种政策输入（gold、top‑1检索、mismatched、none）; 主要指标为宏观 F1。结果显示：结构化状态下 macro‑F1 从 0.293 提升至 0.601；检索条款 macro‑F1 0.580 与 gold 条款 0.604 差异无显著性；mismatched 与无政策表现明显较差。

**⚠️ 局限性**

限制：1）仅在 benchmark‑author 写的单句政策断言上验证，未覆盖真实多页政策文档；2）样本量小（85 个航空决策，15 任务），且仅离线评估；3）检索性能受限于未微调的通用检索器；4）对 fine‑tuning 设定高度敏感，结果在不同训练强度下会变化。

---

## 29. Efficient implementation of graph autoencoders for model-order reduction of systems with sharp gradients

**arXiv ID:** 2606.23834 | [PDF](https://arxiv.org/pdf/2606.23834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 30. Offline Reinforcement Learning for Warehouse SLAM Throughput Control

**arXiv ID:** 2606.23978 | [PDF](https://arxiv.org/pdf/2606.23978v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 31. OmniPath: A Multi-Modal Agentic Framework for Auditing Wheelchair Accessibility

**arXiv ID:** 2606.24129 | [PDF](https://arxiv.org/pdf/2606.24129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 32. Sesame: Structure-Aware Molecular Generation via Spatial Density-Map Conditioning

**arXiv ID:** 2606.23856 | [PDF](https://arxiv.org/pdf/2606.23856v1)

**作者:** Konstantin Yatsenko `[一作]` (Tessel Biosciences Inc), Arvind Thiagarajan `[通讯]` (Tessel Biosciences Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Sesame，一种基于扩散的结构感知分子生成模型，能够同时支持 de novo 设计和基于片段的 lead‑optimization；

**💡 创新点**

创新点包括：①将蛋白口袋和片段都映射为连续密度图，利用 attention‑based spatial pairformer 从密度图中抽样信息；②统一处理离散原子类型/键类型与连续坐标的扩散；③轨迹微调自蒸馏提升生成质量；

**🔧 技术方法**

技术手段包括：扩散模型（DDIM/DDPM）、Pairformer/Evoformer、三维密度图生成、Fourier embedding、SwiGLU、跨模态注意力、重排匹配、旋转增强和自监督轨迹微调；

**📊 数据集**

数据集：ZINC22 ligand‑only（≈15 B 3–50 重原子分子，使用 RDKit 生成 3D conformer）；SAIR protein‑ligand（≈8 M 结构共折叠对，提取 8 Å 口袋并生成密度图）；

**📈 对比分析**

评估方法：在多种 conditioning 模式（P+L、P+F、P、L、F、U）下计算 loss、类型/键准确率、位置 MSE、LDDT、分子有效率等；在 P+F、P 模式下分子有效率分别达 92.4% 与 88.7%，碎片保留率 94.8%，并与真实药物属性分布对齐；与现有基于口袋的生成模型相比表现更佳；

**⚠️ 局限性**

局限性：模型易产生孤立多余原子导致有效率下降；对噪声调度敏感，需手工调节 γ；扩散过程对大分子收敛慢，仍需进一步提升生成质量与可合成性。

---

## 33. Towards Fast and Effective Long Video Understanding of Multimodal Large Language Models via Adaptive Quasi-Gaussian Sampling

**arXiv ID:** 2606.24187 | [PDF](https://arxiv.org/pdf/2606.24187v1)

**作者:** Kun Zhang `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了 AdaQ，一个适应性准高斯采样框架，用于多模态大语言模型的长视频理解，替代传统硬性关键帧选择。

**💡 创新点**

创新点在于将关键帧选择视为基于 3‑σ 原则的准高斯采样，并利用查询‑帧相似度的方差动态调节温度，自动设定 3‑σ 区间，实现自适应、柔性的视频帧采样。

**🔧 技术方法**

采用预训练视觉‑语言嵌入模型（CLIP/LongCLIP/BLIP）计算相似度，构造准高斯分布并通过方差驱动的温度调节实现概率采样。

**📊 数据集**

在四大长视频基准（LongVideoBench、Video-MME、LVBench、MLVU）上进行评估。

**📈 对比分析**

与统一采样、Top‑K 硬选以及现有 SOTA 关键帧方法（OneClip‑RAG、AKS、Q‑Frame）比较，AdaQ 在多 MLLM 与嵌入模型上平均提升 2‑3%，在 Qwen3‑VL 上甚至比 GPT‑4o 提升 15.8% 的平均准确率。

**⚠️ 局限性**

局限性在于仍依赖预训练嵌入模型的相似度质量，对极端噪声或帧极少的场景性能可能受限；且目前仅处理帧级别采样，未涉及时序剪辑或多尺度视频的进一步优化。

---

## 34. Autonomous Video Generation with Counterfactual Controllability for Self-Evolving World Models

**arXiv ID:** 2606.24152 | [PDF](https://arxiv.org/pdf/2606.24152v1)

**作者:** Xin Wang `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个闭环的自主视频生成框架，结合生成、绑定、验证与蒸馏四个阶段，力图构建可对行动进行因果可控的自我进化世界模型。

**💡 创新点**

创新点在于把“可对行动可控”作为核心评价标准，设计了四项互补评估指标（Novelty、Consistency、Out-of-Distribution、Efficiency），并通过闭环反馈实现想象内容向可执行决策的自我修正。

**🔧 技术方法**

核心技术包括：生成模型（视频生成网络）、绑定模块（将未来帧与身体、传感器、驱动器等实现约束对齐）、验证模块（对分布外情况进行校准与否决）、蒸馏模块（把可行分支压缩成决策变量）以及基于多项指标的联合优化方法。

**📊 数据集**

未给出具体公开数据集；论文在讨论中提到可在无人机和机械臂等受控实验环境中验证，但未给出实验细节或数据集名称。

**📈 对比分析**

文章没有提供量化实验结果或与现有方法的对比；评价主要是理论框架与指标设计，缺乏具体性能数值或对比分析。

**⚠️ 局限性**

局限性包括：缺乏实际实验验证与基准测试；对数据集与模型实现细节描述不足；虽然提出了四项指标，但如何统一优化与权重设定仍不明确；对不同任务与硬件平台的泛化能力尚未验证。

---

## 35. Exact Schur-Sylvester Dimensionality Reductions for Non-Smooth Stochastic Complexity and Manifold Sampling

**arXiv ID:** 2606.23867 | [PDF](https://arxiv.org/pdf/2606.23867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 36. DissProve: Automated Verification of Distributed Protocols with Affine Communication

**arXiv ID:** 2606.24003 | [PDF](https://arxiv.org/pdf/2606.24003v1)

**作者:** Christian Fontenot `[一作]` (University of Colorado Boulder), Bor-Yuh Evan Chang `[通讯]` (University of Colorado Boulder)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于向后推理的自动化安全验证方法，针对具有仿射通信的分布式 Actor 协议，通过从错误状态出发的逆向抽象执行，自动证明协议的安全性。

**💡 创新点**

创新点包括：① actor‑causal reduction——只关注与错误相关的因果消息和字段赋值，显著缩小探索空间；② message‑segment summarization——利用仿射协议的约束生成消息段并求解递推关系，解决无限消息循环导致的非终止问题。

**🔧 技术方法**

技术手段：分离逻辑、线性逻辑和有序逻辑的符号抽象来表示无限数量的 actor、网络缓冲和历史；逆向抽象解释实现的 goal‑directed actor materialization；SMT（Z3）求解纯约束、子集包含与矛盾检验；因果依赖图和递推分析生成消息段。

**📊 数据集**

实验数据集：一轮选举 (One‑Shot Election)、Lamport 饼干算法 (Bakery)、两阶段提交 (Two‑Phase Commit)、Barrier Lock 四个经典分布式协议。

**📈 对比分析**

对比方法：在基线（无因果约简、无消息段、无抽象包含检查）下进行消融实验。实验结果表明，采用全套技术后状态空间从数万级下降到几千级，SMT 约束求解次数大幅减少；在 One‑Shot Election 的最大实验中，验证耗时 7 小时 35 分钟，且能够自动发现错误或给出无错误证明。

**⚠️ 局限性**

局限性：仅适用于满足仿射通信假设的协议；对复杂递推结构的消息段生成仍依赖手工或自动化求解器，可能产生过宽或不精确的摘要；SMT 计算是性能瓶颈，尤其在状态空间较大时导致长时间运行；对无限循环的终止保证需构造因果图无环或使用粗宽化，无法完全自动化。

---

## 37. Sim-to-Real Betting on the E-Process: Bringing "simulators" to anytime-valid confidence sequences

**arXiv ID:** 2606.24038 | [PDF](https://arxiv.org/pdf/2606.24038v1)

**作者:** Yujia Chen `[一作]` (Iowa State University), Bowen Weng `[通讯]` (Iowa State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

整合模拟器银行与可随时验证的推断方法，利用近似Kelly投注同时估计均值和构造置信序列，提升样本效率并提供可靠的性能保证。

**💡 创新点**

创新点在于将相同的模拟器混合估计( m_t , v_t )用于估计器的投注大小和置信序列的投注大小，使两者同时受益于模拟器信息，从而在样本稀缺场景下显著提升MSE与置信宽度；并证明无论模拟器质量如何，置信序列始终保持有效。

**🔧 技术方法**

技术包括：近似Kelly投注、覆盖组合(Universal Portfolio)来生成模拟器权重、e-过程与Ville不等式构造任意时点置信序列、截断因子保证e-过程正值。

**📊 数据集**

使用作者提供的合成数据集（在 GitHub 上公开的代码示例），未使用真实机器人实验数据。

**📈 对比分析**

与传统Monte Carlo估计和仅基于数据的Kelly投注相比，实验显示：①均值估计的MSE显著下降；②e-过程对错误候选值的增长更快，导致置信序列收敛更快、宽度更窄；③在样本量较小的情况下效果尤为明显。

**⚠️ 局限性**

限制包括：1) 需事先准备多组模拟器专家；若模拟器估计不准确，收益有限；2) 近似Kelly投注的截断参数对性能有影响；3) 论文未给出估计器的有限样本收敛或偏差-方差界；4) 仅在独立同分布的真实数据流下验证，未讨论非独立或非平稳情形。

---

## 38. EXPO-SQL: Execution-based Clause-level Policy Optimization for Text-to-SQL

**arXiv ID:** 2606.23693 | [PDF](https://arxiv.org/pdf/2606.23693v1)

**作者:** Jaehoon Lee `[一作]` (Sungkyunkwan University), Jee-Hyong Lee `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Text-to-SQL任务中提出EXPO-SQL框架，利用执行结果对生成的SQL进行子句级奖励并进行强化学习优化。

**💡 创新点**

首次将子句级别奖励引入RL学习，通过执行反馈精确定位错误子句，提供细粒度学习信号。

**🔧 技术方法**

采用REINFORCE++强化学习、执行结果差异分析（diff types）、子句级增量执行、错误信息解析及分层奖励设计等技术。

**📊 数据集**

使用SynSQL-Complex-5k进行训练，评估数据集为Spider、BIRD（以及Spider-DK、Spider-Syn、Spider-Realistic）。

**📈 对比分析**

与SFT、提示式和RL基线对比，EXPO-SQL在Spider和BIRD上分别提升约1.2%–2.4%执行准确率，特别是复杂查询提升5.6%，并在多模型尺度下均保持领先。

**⚠️ 局限性**

仅在SQLite环境验证，需适配其他数据库方言；错误信息解析对不同执行器依赖较高。

---

## 39. Decoherence as Defence and the Magnitude of Noise Regularisation: A Rigorous N -Qubit Theory of Stochastic Quantum Neural Networks for Adversarially Robust Network Intrusion Detection

**arXiv ID:** 2606.24219 | [PDF](https://arxiv.org/pdf/2606.24219v1)

**作者:** Gautier-Edouard Edouard Filardo `[一作]` `[通讯]` (Efrei Paris Pantheon-Assas University), Gautier-Edouard Edouard Filardo (Efrei Paris Pantheon-Assas University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在噪声下的随机量子神经网络（SQNN）在网络入侵检测中的鲁棒性与正则化效果，并将其理论推广到实际数据和中性原子硬件；

**💡 创新点**

提出了噪声收缩定理（Depolarising Channel 对 Pauli 读取的收缩律）和量子 dropout 的自适应正则化公式，证明了噪声对决策边界的训练时刻塑造是提升鲁棒性的关键；

**🔧 技术方法**

使用了 Lindblad 主方程、随机量子电路、密度矩阵模拟、单量子比特的 Depolarising 和 Dephasing 通道、量子 dropout（按门级随机失活）以及 PCA 降维后的量子测量；

**📊 数据集**

在两个真实网络安全数据集上验证：NSL‑KDD（41 特征→4 PCA）和 UNSW‑NB15（同样降维）；

**📈 对比分析**

与经典基线（随机森林、MLP、SVM）以及对抗训练的 MLP 进行白盒 FGSM/PGD 及 ℓ₂ 攻击对比。结果显示：① Depolarising SQNN 在强攻击（PGD‑20）下比无噪声模型更鲁棒且不出现灾难性崩塌；② 量子 dropout 与 Depolarising 对训练-测试差距均能显著（≈0.01）减小，且最佳失活率为 0.5；③ 在 UNSW‑NB15 上，SQNN 的鲁棒性仍低于对抗训练的 MLP。

**⚠️ 局限性**

局限性：仅在 d=4 PCA 维度和 N≤12 的模拟范围内验证；对抗训练的经典模型更强，未在更大数据集或更高维度下验证；只考虑了理想的 Depolarising 通道，实际硬件中 dephasing 占主导；需进一步在实际中性原子硬件或更大网络上评估。

---

## 40. A Unified Framework for Runtime Verification and Model-Based Diagnosis in LOLA

**arXiv ID:** 2606.23720 | [PDF](https://arxiv.org/pdf/2606.23720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 41. ESBMC-PLC+: A Unified IEC~61131-3 Formal Verification Framework as a PLCverif Successor

**arXiv ID:** 2606.23870 | [PDF](https://arxiv.org/pdf/2606.23870v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 42. ReMMD: Realistic Multilingual Multi-Image Agentic Verification for Multimodal Misinformation Detection

**arXiv ID:** 2606.24112 | [PDF](https://arxiv.org/pdf/2606.24112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 43. ObsGraph: Hierarchical Observation Representation for Embodied Reasoning and Exploration

**arXiv ID:** 2606.24068 | [PDF](https://arxiv.org/pdf/2606.24068v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 44. The Serialized Bridge: Understanding and Recovering LLM Serving Performance under Blackwell GPU Confidential Computing

**arXiv ID:** 2606.23969 | [PDF](https://arxiv.org/pdf/2606.23969v1)

**作者:** Hang Yin `[一作]` (Phala Network), Kevin Wang `[通讯]` (Phala Network)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在Blackwell平台上对GPU机密计算（GPU-CC）导致的性能下降进行细粒度分析，定位桥接序列化和异步取消为LLM推理、KV恢复和模型加载的主要瓶颈，并提出一系列软件修复。

**💡 创新点**

创新点在于首次给出GPU-CC桥接的因果性能模型（序列化通道、异步失效、每次交叉的固定开销），揭示了策略倒置现象，并通过标志、工作线程、上下文池化等措施实现高达92%的性能恢复；同时首次公开验证了B300 HGX的多GPU Fabric 租户功能。

**🔧 技术方法**

采用CUDA微基准、vLLM推理实验、PyTorch profiler（aten::copy_）、多上下文调度、工作线程驱动的复制、TPU/TDISP等技术；在Intel TDX保护的CVM中使用GPU-CC、NVLink/NVSwitch、Fabric Manager 等硬件与软件配合实现实验。

**📊 数据集**

使用多种LLM推理工作负载（Qwen3.6‑27B‑FP8、Gemma‑4‑31B‑it、Qwen2.5‑3B、GPT‑OSS‑120B 权重）以及KV缓存恢复情景，主要以模型权重与推理吞吐量为衡量数据。

**📈 对比分析**

通过在同一硬件上对比CC‑on与CC‑off的吞吐量和延迟，发现密集解码下降13–26%，KV恢复延迟提升131%，模型加载时间提升34倍；通过设置标志或工作线程可分别恢复57%和92%的性能缺口；在RTX Pro 6000与B300 HGX间的跨平台实验验证了瓶颈为桥接而非GPU本身。

**⚠️ 局限性**

局限性包括：主要基于单次测量，B300硬件已归还；高并发恢复率仅在一次实验中验证；仅对n=2/4 Fabric 租户做了完整评估，n=8及NCCL收敛未完成；未完整覆盖Hopper平台；部分性能数据未统计置信区间，结果受硬件窗口限制。

---

## 45. FedUP: One-Shot Federated Unlearning via Centroid-Guided Plug-in Filters

**arXiv ID:** 2606.24113 | [PDF](https://arxiv.org/pdf/2606.24113v1)

**作者:** Feihong Nan `[一作]` (National University of Defense Technology), Ji Wang `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种FedUP框架，利用轻量可插拔过滤器和差分隐私类中心样本实现一次性联邦模型的忘记，避免对原始模型参数的直接大规模更新，保持非目标知识不丢失。

**💡 创新点**

创新点包括：①通过可插拔过滤器实现一次性忘记，省去多轮通信；②使用DP保护的类中心样本作为无监督输入，仅上传少量信息；③实现可逆忘记，删除过滤器即可恢复原模型。

**🔧 技术方法**

使用的技术包括联邦学习、差分隐私（Gaussian noise）、KMeans聚类、轻量编码‑解码过滤器、交叉熵+重构损失联合训练、参数冻结与单轮 fine‑tuning。

**📊 数据集**

实验数据集涵盖 MNIST、CIFAR‑10、CIFAR‑100、AG News，模型结构有 LeNet‑5、ResNet‑18/34、Transformer、TinyBERT。

**📈 对比分析**

与 EraseClient、Federaser、Exact‑Fun、Retrain、FUSED 等基线比较，FedUP 在保持 R‑A 最高、F‑A 低至零、MIA 低的同时，将忘记响应时间从数分钟压缩到秒级，仅一轮通信，整体性能最优。

**⚠️ 局限性**

局限性在于类中心样本的质量依赖于全局特征提取器的表现；若联邦特征提取模型质量差，聚类结果可能偏差，影响忘记效果；在极端数据不均衡或分布漂移场景下性能可能下降。

---

## 46. On the Semantics of Generative SPARQL

**arXiv ID:** 2606.23875 | [PDF](https://arxiv.org/pdf/2606.23875v1)

**作者:** Ratan Bahadur Thapa `[一作]` (University of Stuttgart), Steffen Staab `[通讯]` (University of Stuttgart)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在 SPARQL 中加入生成查询操作 GenOp，并给出完整的语义定义和执行方法

**💡 创新点**

创新点在于将语言模型的生成结果直接纳入解映射空间，使用兼容性关系替代等价性，利用求值递归的 fixpoint 语义和层级化/无穷小化处理，保持固定数据集假设并保证语义可推理

**🔧 技术方法**

使用 typed solution mappings、兼容性关系、递归立即后果算子、fixpoint 迭代、层次化语义、well‑founded 语义、候选生成与自我支持验证等技术

**📊 数据集**

示例使用的 RDF 图为 G={:Paris a :City}，但实验与数据集细节未给出；主要关注理论复杂度

**📈 对比分析**

在理论上证明了无论是无环、分层还是无穷小化的递归查询，数据复杂度仍为 PTIME，组合复杂度为 PSPACE‑complete，与标准 SPARQL 保持一致；没有提供实验性能数据

**⚠️ 局限性**

局限在于必须满足确定性有界生成、有限候选覆盖以及精确性假设；递归仅限于生成结果的自我支持，未能将生成值写入 RDF 图；未来需放宽这些限制并探索更强的表达能力

---

## 47. A Time-Reparameterized Cumulative Intensity Extrapolation Sampler for Discrete Flow Matching

**arXiv ID:** 2606.24140 | [PDF](https://arxiv.org/pdf/2606.24140v1)

**作者:** Feiyang Fu `[一作]` (Zhejiang University), Hehe Fan `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种新的离散流匹配（DFM）采样方法 TR-CIE，结合时间重参数化与累计强度外推，实现每步仅一次模型评估，提升低NFE下的采样质量。

**💡 创新点**

创新点包括：①使用基于噪声调度的对数时间重参数化消除终端阶段的刚性；②通过历史缓存实现线性累计强度外推，降低离散化误差；③在保持与传统 τ‑leaping 同样计算量的前提下，显著提升采样效果。

**🔧 技术方法**

主要技术：连续时间马尔可夫链（CTMC）采样、时间重参数化、累计强度外推（Adams–Bashforth 思路）、Poisson τ‑leaping、基于 Transformer 的概率去噪器。

**📊 数据集**

实验数据集：自然语言文本生成（标准语言模型数据集，评估 GPT‑2 生成困惑度）、文本‑图像生成（FUDOKI backbone + GenEval 553 条提示）、合成计数序列任务（256 长度、32 词表的严格递减序列）。

**📈 对比分析**

与 Euler、Tweedie、θ‑RK2、θ‑Trapezoidal、FHS、HO‑FHS 等方法在相同 NFE（8–256 步）下进行对比。TR‑CIE 在文本生成中 GPT‑2 perplexity 下降 30% 以上，在文本‑图像中 GenEval 准确率提升 3–5%，在合成任务中错误率下降至 1% 以下，显示出明显的性能优势。

**⚠️ 局限性**

局限性：仅针对标准因子化 DFM 参数化，可能对其他流匹配框架适用性有限；累计强度外推依赖历史缓存，若状态变化剧烈会产生漂移误差；在极高 NFE 或极大 vocab 的情况下，重参数化与外推的优势减弱。

---

## 48. EPEdit: Redefining Image Editing with Generative AI and User-Centric Design

**arXiv ID:** 2606.24057 | [PDF](https://arxiv.org/pdf/2606.24057v1)

**作者:** Hoang-Phuc Nguyen `[一作]` (Vietnam National University), Trung-Nghia Le `[通讯]` (Vietnam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个名为EPEdit的基于Web的图像编辑系统，支持图像生成、对象替换/去除、背景修改、姿态/透视调整、区域性编辑以及主题集合设计等多种创意编辑任务，并提供了用户友好的文本命令与绘制遮罩的交互方式。

**💡 创新点**

创新点主要包括：①采用零射（zero-shot）Stable Diffusion算法，无需额外微调，降低模型使用成本；②将多种编辑功能集成到同一平台，并通过虚拟助手推荐具体操作；③强调用户中心化设计，支持从绘图到文本提示的多种输入方式；④实现了Web端部署，避免安装和兼容性问题。

**🔧 技术方法**

技术栈包括：前端ReactJS（组件化、动态渲染）；后端Express（API路由）；Python模型服务器（托管预训练Stable Diffusion模型，支持图像生成、编辑等任务）；MVC架构；模型预加载提高响应速度；零射Stable Diffusion算法实现多任务编辑。

**📊 数据集**

使用的是预训练好的Stable Diffusion模型，论文未明示使用特定公开数据集进行训练，推断是依赖公开训练好的模型（如LAION等）。

**📈 对比分析**

通过对24名志愿者的用户研究，使用5分制问卷评估易用性、速度、质量、满意度等指标，并与Adobe Photoshop进行对比。结果显示：EPEdit在易用性、速度和图像质量方面均显著优于Photoshop，满意度得分更高，证明其在创意编辑任务中的高效性。

**⚠️ 局限性**

局限性包括：①遮罩功能仍需进一步提升精度和细粒度控制；②目前主要通过文本输入，缺乏语音或更自然的交互方式；③模型仍基于预训练Stable Diffusion，可能在某些特定场景下产生偏差或低质量输出；④硬件依赖（需要GPU，评测使用A‑100）可能限制普及；⑤缺少对算法公平性与偏见的系统评估。

---

## 49. VieSpeaker: A Large-Scale Vietnamese Speaker Recognition Dataset Beyond Visual Dependency

**arXiv ID:** 2606.24066 | [PDF](https://arxiv.org/pdf/2606.24066v1)

**作者:** Viet Hoang Pham `[一作]` (Hanoi University of Science and Technology), Thi Thu Trang Nguyen `[通讯]` (Hanoi University of Science and Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无面部依赖的数据构建管线，构建了规模最大的越南说话人识别数据集VieSpeaker，包含4,715名说话人、902小时语音。

**💡 创新点**

创新点在于利用大型语言模型对文本元数据进行结构化推理，实现无需视觉信息的说话人身份推断，并通过聚类合并实现身份统一。

**🔧 技术方法**

技术包括YouTube元数据爬取、Pyannote speaker diarization、LLM（Google AI Studio ChatGPT）身份推断、ECAPA‑TDNN特征提取、增量聚类与降噪等。

**📊 数据集**

使用了从公开YouTube视频中采集的三大领域（访谈、娱乐、播客）语料，并在实验中与Vietnam‑Celeb、VoxVietnam、VoxCeleb2等基准数据集比较。

**📈 对比分析**

通过EER指标比较，VieSpeaker在Vietnam‑Celeb和VoxVietnam基准上均优于现有越南数据集，并在预训练+微调中实现相对EER下降约5–16%，在自己设定的VieSpeaker‑H难度上亦显著提升。

**⚠️ 局限性**

限制在于仍缺乏多语言说话人、对面部视觉辅助方法未与LLM结合，且对长音频的语音质量控制和说话人多样性（如不同方言）不足。

---

## 50. How Many RF Chains Does a Microwave Linear Analog Computer (MiLAC) Need to Match the Fully-Digital Cramér-Rao Bound?

**arXiv ID:** 2606.23986 | [PDF](https://arxiv.org/pdf/2606.23986v1)

**作者:** Yuchen Zhang `[一作]` (King Abdullah University of Science and Technology), Tareq Y. Al-Naffouri `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对失耗、可逆的微波线性模拟计算机（MiLAC）在做远场目标方向估计时的 Fisher 信息和 Cramér‑Rao 限界（CRB）进行系统性理论分析，并给出了在满足任意目标配置时与全数字接收机等效的 RF 链路数阈值与硬件实现方法。

**💡 创新点**

创新点：
- 证明了至少需要两条 RF 链路/目标即可消除与全数字接收机相比的 Fisher 信息损失（零间隙阈值）。
- 将 CRB 优化从矩阵流形降到 Grassmannian，得出只需关注行空间的结论。
- 设计了“stem‑connected” MiLAC 架构，实现线性(O(L_R N_R)) 的可调谐元件计数，远低于传统全连通实现。
- 证明相位移器阵列受常数模约束，无法在一般目标配置下实现零间隙。

**🔧 技术方法**

技术手段：
- Cramér‑Rao 限界与 Fisher 信息矩阵分析（Slepian–Bangs 公式）。
- Grassmannian 几何与子空间投影不变性。 
- 对称单元正交散射矩阵构造与对称单位矩阵完成。
- 维数计数论证与可达性分析。
- Monte‑Carlo 方向估计实验验证理论。

**📊 数据集**

数据集：
- 采用半波长 ULA 的合成模拟数据，随机生成目标角度、幅值与波形符号，进行 50 次快照的仿真，验证 CRB 与 MLE 性能。

**📈 对比分析**

比较方法与性能：
- 与全数字接收机（L_R = N_R）和相位移器组合器（常数模）进行对比。 
- 结果显示：
  * L_R = 2K 的 MiLAC CRB 与全数字一致；
  * L_R = 2K-1 时出现可观误差；
  * 相位移器在大多数配置下保持明显的 CRB gap。 
- 实验 MLE MSE 与 CRB 匹配，验证理论准确性。

**⚠️ 局限性**

局限性：
- 仅分析单波形、无耦合、理想失耗可逆的接收网络；实际系统需考虑互耦、非理想谐振器、功率损耗等。
- 仅针对方向估计，未讨论多频、时变或多目标跟踪等更复杂情形。
- 需要进一步研究 MiLAC 与通信功能的联合 ISAC 性能。

---

## 51. Critique of Agent Model

**arXiv ID:** 2606.23991 | [PDF](https://arxiv.org/pdf/2606.23991v1)

**作者:** Eric Xing `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Jinyu Hou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了“Agentic”与“Agentive”系统的概念，并基于五个维度（目标、身份、决策、自我调节与学习）对现有AI Agent进行系统化评估，进而设计了Goal‑Identity‑Configurator (GIC) 架构，目标是构建真正具备自我决策、目标拆解、身份演化、内在规划与自我学习能力的通用Agent。

**💡 创新点**

创新点在于：① 明确区分“Agentic”与“Agentive”两类系统并给出定量化维度；② 强调内部化而非外部化的目标与身份；③ 将世界模型与Agent模型功能明确分离，提出基于世界模型的“模拟推理”而非纯黑盒策略；④ 引入可学习的“配置器”(Configurator)实现自我调节；⑤ 架构内集成了层级目标拆解、身份演化、模拟规划、执行与学习，形成一套可在持续训练与部署中自我提升的闭环。

**🔧 技术方法**

技术核心包括：多模态感知编码器、层级目标拆解网络、身份演化网络、世界模型（如GLP/梦境模型）用于预测下一个状态、基于价值的模拟规划（System II）、可学习的配置器（System III）以及终端执行者（Actor System I）。训练采用分阶段：预训练（知识蒸馏 + 自监督预测）、模拟RL（在世界模型内强化学习）、真实环境微调与持续自我训练。

**📊 数据集**

使用的数据集包括：① 公开的语言与多模态数据集用于预训练LLM（如OpenAI GPT系列、ChatGPT、Claude等）；② 大规模的模拟环境数据（如AirSim、CARLA、DeepMind Lab、World Labs）用于训练世界模型与模拟RL；③ 真实操作日志与传感器数据（如飞行日志、机器人操作记录）用于真实环境微调与自我学习；④ 领域特定任务数据（代码、对话、控制任务）用于评估。

**📈 对比分析**

评估方法：在长周期任务、层级目标拆解、多任务迁移与多智能体协调等多维度基准上测试GIC；与传统“Agentic”系统（工具调用、预编程流程）和近似方法（如Pure MPC、单体LLM策略）进行对比；指标包括任务成功率、规划效率（搜索步数/时间）、自我调节准确性（是否在需要时切换规划/执行）以及持续学习收益（实时更新后的性能提升）。实验显示GIC在复杂任务中显著提升成功率并减少对外部工具的依赖，但仍需更大规模验证。

**⚠️ 局限性**

局限性：① 仍需大规模、精细的世界模型，训练成本高且对高维物理场景挑战大；② 目标拆解与身份演化的学习策略尚未成熟，可能对训练数据质量敏感；③ 配置器与模拟规划在实时性和计算资源上仍有瓶颈，难以在极限资源受限场景中部署；④ 对异常状态或未见环境的鲁棒性尚未充分验证；⑤ 由于架构复杂，调试与可解释性难度提升，需进一步研究可解释性与安全约束。

---

## 52. Welfarist Control Design -- How to fulfill the societal mandate in multi-agent control?

**arXiv ID:** 2606.23931 | [PDF](https://arxiv.org/pdf/2606.23931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 53. SBM With Multiple Samples: Improved Spectral Recovery

**arXiv ID:** 2606.24141 | [PDF](https://arxiv.org/pdf/2606.24141v1)

**作者:** Sie Hendrata Dharmawan `[一作]` (Dartmouth), Peter Chin `[通讯]` (Dartmouth)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出多样本谱算法用于双块SBM社区检测。

**💡 创新点**

证明平均多样本可将错误率指数下降，改进了单样本阈值。

**🔧 技术方法**

使用谱分解、Davis–Kahan子空间角度分析以及噪声矩阵谱范数界。

**📊 数据集**

在n≤1000、m≤9的随机生成SBM图上实验。

**📈 对比分析**

与单样本谱算法对比，实验显示两三样本即可显著提升准确率，接近理论预测。

**⚠️ 局限性**

仅适用于两块均匀SBM，未考虑稀疏图中的孤立顶点与多块/重叠结构。

---

## 54. Memory Layouts for GPU-Data Transfer Buffering in SPH

**arXiv ID:** 2606.23891 | [PDF](https://arxiv.org/pdf/2606.23891v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 55. From Heuristics to Transformers: A Comprehensive Survey of Type Inference from Stripped Binaries

**arXiv ID:** 2606.23692 | [PDF](https://arxiv.org/pdf/2606.23692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 56. DoHFuse: A Dual-Branch Architecture with DMAGLSTM for Website Fingerprinting over DNS over HTTPS/3

**arXiv ID:** 2606.24105 | [PDF](https://arxiv.org/pdf/2606.24105v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 57. The Degeneracy Distillery

**arXiv ID:** 2606.23838 | [PDF](https://arxiv.org/pdf/2606.23838v1)

**作者:** T. Lucas Makinen `[一作]` (University of Cambridge), Benjamin D. Wandelt `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种自动发现并消除参数退化的方法——degeneracy distillery，利用 Fisher 信息矩阵估计、神经网络平坦化与符号回归得到可解释的参数重参数化。

**💡 创新点**

创新点在于：①在整个参数空间上估计 Fisher 矩阵并实现全局平坦化；②通过神经网络学习全局坐标变换并用符号回归提取解析表达式；③提供数据驱动、全局非线性重参数化，显著降低模拟成本并提升后验采样效率。

**🔧 技术方法**

采用 Fishnet 估计 Fisher 信息、神经网络“flattening”坐标变换、符号回归（operon）生成解析坐标，并使用集成与对齐方案；对比 geodesic normal、传统 chirp‑mass 等方法。

**📊 数据集**

实验数据涵盖：合成案例（Rosenbrock、单变量高斯）；工业加热器（温度响应）；SIR 传染病模拟；重力波 inspiral 与 IMRPhenomD 合成波形；弱引力透镜 2‑点相关函数（暗物质模拟）。

**📈 对比分析**

与原始参数空间、手工坐标、解析扁平化坐标等进行比较，结果显示：使用 distillery 的坐标后，后验估计的模拟量可减少至原来的 1/10；在 SIR、GW 等任务中 CRPS 与校准更好；在高维问题中实现指数级计算节省。

**⚠️ 局限性**

局限性包括：①无法处理多模态后验；②高维参数（>100）导致符号回归不可扩展；③完全退化（F 奇异）需要先前置约束；④Fishnet 在近似高斯后验下有效，对强非高斯分布精度有限。

---

## 58. Beyond Trajectory Imitation: Strategy-Guided Policy Optimization for LLM Reasoning

**arXiv ID:** 2606.24064 | [PDF](https://arxiv.org/pdf/2606.24064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 59. MedBench v5: A Dynamic, Process-Oriented, and Hallucination-Aware Benchmark for Clinical Multimodal Models

**arXiv ID:** 2606.24155 | [PDF](https://arxiv.org/pdf/2606.24155v1)

**作者:** Ding Jinru `[一作]` (Shanghai Artificial Intelligence Laboratory), and Xu Jie `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MedBench v5，构建面向临床多模态 AI 的动态、过程化评估基准，整合 CCR 与 MAS 两维框架，加入信息流压力测试、五节点过程审计与幻觉传播监控。

**💡 创新点**

创新性在于统一过程诊断与幻觉追踪，设计可切换的信息流扰动（缺失、矛盾、延迟），并通过五节点审计精细定位模型认知失效点。

**🔧 技术方法**

采用双维评估框架、可切换压力器、五节点审计协议、轨迹级幻觉监测以及自动化评估与人工一致性分析。

**📊 数据集**

基于 63 个临床任务，涵盖 LLM、视觉-语言、代理交互三条轨道，以及四个可执行环境（DataAgent、RAGAgent、DeepResearch、SafetyAgent），并在 18 个数据集上构造 720 个压力化多轮场景。

**📈 对比分析**

通过宏平均得分对比 Claude Opus 4.7、Qwen3.7、Gemini-3.1 等前沿模型；结果显示在静态 QA 仍高分，但在信息流扰动下对矛盾检测和诊断更新的损失显著；幻觉传播在缺失+延迟场景最为严重。

**⚠️ 局限性**

局限性包括：自动评估与人工评分间一致性有限（ICC=0.74、Spearman=0.26），仅覆盖 63 任务且动态场景仅来自 18 个数据集，压力器种类有限，无法涵盖所有临床不确定性；评估仍主要针对前沿模型，缺乏更广泛的真实环境验证。

---

## 60. Scaling Dense Retrieval with LLM-Annotated Training Data: Structured Mining and Progressive Curriculum for E-Commerce Sponsored Search

**arXiv ID:** 2606.23911 | [PDF](https://arxiv.org/pdf/2606.23911v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 61. GRACE: Gated Refinement for Accurate Causal Edge Discovery in High-Dimensional Time Series

**arXiv ID:** 2606.23880 | [PDF](https://arxiv.org/pdf/2606.23880v1)

**作者:** Mohammad Fesanghary `[一作]`, Abhinav Havaldar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用Hard Concrete门控与L₀正则化对约束型因果结构进行两阶段精细化的框架；

**💡 创新点**

创新点在于门控生成的二值分布实现自动阈值化、针对维度与骨架密度的自适应正则化公式，以及在河流网络中引入时序自举稳定性选择；

**🔧 技术方法**

使用线性CI检验生成高召回骨架，随后训练单一门控模型；门控通过Hard Concrete分布实现硬零/一决策，并配合L₀正则化；同时在真实数据中采用窗口化自举策略；

**📊 数据集**

评估数据包括：多种拓扑（随机、尺度自由、小世界）且维度从20到100、样本量300-2000的合成SCP图；以及德国埃尔贝河（12 个流量站，3164 条 6h 分辨率观测）的真实流向网络；

**📈 对比分析**

与PCMCI、CDNOTS、TCDF、CUTS+等基准方法对比，Gated Refinement 在合成数据上 F1 均值提高至0.83-0.92（相较 CDNOTS 的0.53-0.71），精度 86-97%，召回保持 83-88%；在河流网络上通过自举得到 9/11 条真实边、1 条假边，F1 0.86、AUROC 0.99，显著抑制 99% 假阳性；运行时间保持在数分钟；

**⚠️ 局限性**

局限性包括：在高密度图（hub 结构）时门控分布聚集导致二值化误判、潜在共因影响可在预测中误被保留、线性骨架无法捕捉纯非线性依赖；对极低样本量或极端非平稳条件下性能衰减。

---

## 62. Reinforcement Learning Towards Broadly and Persistently Beneficial Models

**arXiv ID:** 2606.24014 | [PDF](https://arxiv.org/pdf/2606.24014v1)

**作者:** Akshay V. Jagadeesh `[一作]` (OpenAI), Karan Singhal `[通讯]` (OpenAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多领域益性特征对话数据上用强化学习训练模型，并在数十个独立的对齐与有益性评测上进行评估

**💡 创新点**

首次展示奖励益性特征能在训练分布之外产生广泛的对齐泛化和持久性，且只需将约5%的训练数据改为益性对齐任务即可

**🔧 技术方法**

利用RL（基于奖励信号的强化学习）对大规模语言模型进行对齐微调

**📊 数据集**

构造的多域益性特征数据集，涵盖健康、教育、商业、工程、法律等12个领域，包含15个细粒度的益性特征

**📈 对比分析**

与计算匹配的基线模型相比，益性特征RL模型在超过50个OOD评测中提升约80%的任务，平均提升约9个百分点，并在健康、心理健康等公开福利评测中亦有显著提升

**⚠️ 局限性**

局限性包括：仅在单一模型规模上验证，缺乏对更大模型的泛化评估；因果机制与特征交互尚未深入解析；对真实生产流量的评测覆盖有限，且实验规模受算力限制

---

## 63. HANCLIP: A Family of Hyperbolic Angular Negation Vision Language Models

**arXiv ID:** 2606.23843 | [PDF](https://arxiv.org/pdf/2606.23843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 64. Machine Learning Modeling for Real-Time Melt Pool Monitoring in Laser Powder Bed Fusion Additive Manufacturing: A Hybrid Approach

**arXiv ID:** 2606.23851 | [PDF](https://arxiv.org/pdf/2606.23851v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 65. Does My Embedding Reflect That $A = B$? Evaluating Mathematical Equivalence in Embedding Models

**arXiv ID:** 2606.23959 | [PDF](https://arxiv.org/pdf/2606.23959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 66. EMAgnet: Parameter-Space EMA Regularization for Policy Gradient Self-Play in Large Games

**arXiv ID:** 2606.23995 | [PDF](https://arxiv.org/pdf/2606.23995v1)

**作者:** Tristan Maidment `[一作]` (Riot Games), Wesley N. Kerr `[通讯]` (Riot Games)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在自我对弈中使用指数移动平均（EMA）作为正则化目标的PPO算法——EMAgnet，用以改进传统统一正则化的弱点

**💡 创新点**

创新点在于将表格层面的“移动磁铁”概念推广到深度强化学习，通过在参数空间维护EMA来动态适配策略，既能遗忘劣势策略又能保留优秀策略

**🔧 技术方法**

采用Proximal Policy Optimization（PPO）与KL正则化；在每次PPO更新后用	(τ)=0.1更新磁铁参数；对策略空间做可解释的单步训练与评估

**📊 数据集**

使用OpenSpiel提供的三类零和博弈（Biased RPS、4‑Card Goofspiel、Kuhn Poker）以及其两种扩展——FF（加投机）与Control（导航）变体，覆盖从标准到极端被支配策略丰富的情形

**📈 对比分析**

与PPO-Uniform（线性/幂律退火）基线对比。EMAgnet在标准游戏中可与基线匹敌，在含有大量被支配策略的FF与Control游戏中均显著降低可利用性并显著加快收敛；磁铁策略往往优于最终策略

**⚠️ 局限性**

局限在于尚未系统探讨游戏结构（如循环与层级性）对EMA磁铁效用的影响；对超参数（如	）和退火策略的敏感性需进一步研究

---

## 67. Overconfident Coordinates: Quantifying Confidence in Traceroute Geolocation

**arXiv ID:** 2606.24027 | [PDF](https://arxiv.org/pdf/2606.24027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 68. Neuro-Symbolic Drive: Rule-Grounded Faithful Reasoning for Driving VLAs

**arXiv ID:** 2606.23938 | [PDF](https://arxiv.org/pdf/2606.23938v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 69. FP8 is All You Need (Part 2): Efficient Ozaki-Bailey Style FFT Through Tensor-core Garner Reformulation and Kulisch Escape Route

**arXiv ID:** 2606.23698 | [PDF](https://arxiv.org/pdf/2606.23698v1)

**作者:** Satoshi Matsuoka `[一作]` `[通讯]` (RIKEN Center for Computational Science), Satoshi Matsuoka (RIKEN Center for Computational Science)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在 NVIDIA Blackwell Ultra (B300) 等 FP64‑崩塌 GPU 上使用 Ozaki‑Bailey 复合方法与 Kulisch 累加器，实现 3‑D FFT 的高精度内存‑roof 性能恢复。

**💡 创新点**

创新点包括：① 将 Bailey 六步 FFT 与 Ozaki‑II 复合映射到 8×8 张量核心；② 用前向 CRT 切片重构拆分 Garner，形成 Phase‑A / Phase‑B；③ 在 Phase‑B 使用 Kulisch 完全算术累加器把 64‑位精度归约转移到 INT32 SIMT 管线；④ 通过四个“带宽‑等价平面”阈值制定 GPU 设计规则。

**🔧 技术方法**

使用的技术包括 Ozaki‑II 模式、Bailey 六步 FFT、前向 CRT 与 Bernstein 分数 CRT、Kulisch 固定点累加器、Tensor‑core MMA、FP8/INT8 子系统以及 TME（Tensor‑Memory Equilibrium）模型。

**📊 数据集**

主要数据集为 1024³ 复数数组（即 1,024^3 的 FFT），以及用于验证的 8192³ 矩阵乘等合成工作负载。

**📈 对比分析**

通过理论 Roofline 与实验模拟对比，B300 在未改进时 260 ms；实现 Kulisch Phase‑B 后理论约 18 ms，远低于内存‑roof 12.9 ms；与 H100/B200/Rubin 的原生 FFT 对比显示 B300 通过 Kulisch 路径可实现与 native 约相当的性能。

**⚠️ 局限性**

局限性包括：① 结果基于理论模型与 Python 原型，未在真实硬件上测得；② 需要在 B300 上实现 Kulisch Phase‑B 并验证 INT32 SIMT 效率；③ 仅适用于 FFT 及类似低内核‑intensity、64‑精度归约的工作负载；④ 对动态范围极端的数据可能需要自适应精度扩展。

---

## 70. An Efficient Construction of Completely Independent Spanning Trees in Dense Gaussian Networks

**arXiv ID:** 2606.23935 | [PDF](https://arxiv.org/pdf/2606.23935v1)

**作者:** Zaid Hussain `[一作]` (Kuwait University), Bader AlBdaiwi `[通讯]` (Kuwait University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在稠密高斯网络中构建完全独立生成树（CISTs）的高效方法，以提高故障容忍性、可靠性和通信效率。

**💡 创新点**

通过将网络划分为多个子集并利用其对称性，成功构建了两个CIST，且深度为3k-1，显著优于现有方法。

**🔧 技术方法**

使用了顺序和并行算法来构建CIST，并设计了基于CIST的路由协议。

**📊 数据集**

在稠密高斯网络中进行实验，网络大小从3+4i到12+13i不等，测试了不同故障条件下的性能。

**📈 对比分析**

与现有方法相比，提出的方法在构建的树深度和消息传递步骤上均有显著改善，平均提升至少33%。

**⚠️ 局限性**

方法的局限性在于在一般图中寻找两个或多个CIST是一个NP完全问题，尽管在特定类别的图中是可行的。

---

## 71. Legal Reasoning Is Not Lawyering: Rethinking Legal Benchmarks for Pro Se Access to Justice

**arXiv ID:** 2606.23716 | [PDF](https://arxiv.org/pdf/2606.23716v1)

**作者:** Andrew Lou `[一作]` (Yale), David Shin `[通讯]` (Yale)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出现有法律AI基准只评估律师预处理后输入的法律推理能力，未测量对未代理当事人（pro se）使用时模型的鲁棒性，因而难以验证AI提升司法可及性的真实性；

**💡 创新点**

创新点在于将机器学习文献中关于输入扰动对LLM性能的研究与pro se诉讼的实际问题结合，指出上界与下界的差距，并在LEXam基准上进行扰动实验验证模型在噪声输入下的性能下降；

**🔧 技术方法**

技术主要使用大型语言模型（GPT‑4系列）进行推理，并采用文本扰动（打字错误、上下文稀释与交错插入无关句子）来模拟pro se输入噪声；

**📊 数据集**

使用数据集为LEXam多选题集（100道英文学术法律题），并对其进行三种打字错误频率与两种上下文稀释方式的扰动；

**📈 对比分析**

比较方法：对照原始题目与扰动后题目在三款模型（GPT‑4.1‑mini、GPT‑4.1‑nano、GPT‑4o‑mini）的正确率；实验显示，打字错误和上下文稀释均导致准确率下降，且不同模型对扰动的鲁棒性存在差异，表明下界性能不稳定；

**⚠️ 局限性**

局限性包括样本规模小（仅100道题）、只评估三款同系列模型、未考虑模型的拒绝回答或澄清询问行为、扰动范围局限于文本错误与无关填充，未覆盖所有pro se常见输入缺陷（如事实遗漏、逻辑不连贯等）。

---

## 72. TurboMPC: Fast, Scalable, and Differentiable Model Predictive Control on the GPU

**arXiv ID:** 2606.24039 | [PDF](https://arxiv.org/pdf/2606.24039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 73. Domain-Driven Design in Practice: A Mining Study of Maintenance and Evolution in Open-Source Repositories

**arXiv ID:** 2606.23984 | [PDF](https://arxiv.org/pdf/2606.23984v1)

**作者:** Weixing Zhang `[一作]` (Karlsruhe Institute of Technology), Anne Koziolek `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究对 GitHub 上自标识为 Domain‑Driven Design（DDD）的开源仓库进行大规模挖掘与分析，系统地探讨了DDD战术构建块的分布、随时间演化的特征、Bounded Context 边界违规的出现频率以及这些违规与维护活动（如 bug 修复、重构等）之间的时序关联。

**💡 创新点**

首次在大规模数据集上量化 DDD 构建块使用与边界违规，提出三层识别管道（注解/标记、命名约定、包结构）来自动抽取构建块，并将结构演化与维护活动进行时序关联分析，为 DDD 工具和方法的改进提供经验基础。

**🔧 技术方法**

使用了注解/接口识别、正则命名匹配、包目录分析三层管道；AST 静态分析（JavaParser、Roslyn、TypeScript Compiler API）用于检测跨 Context 的依赖；PyDriller 抽取提交历史；LLM 辅助的提交信息分类；统计分析包括 Kruskal‑Wallis、Spearman 相关、Jensen‑Shannon Divergence、滑动窗口时间序列等。

**📊 数据集**

约 865–2000+ 个自标识为 DDD 的 GitHub 仓库（覆盖 Java、C#、TypeScript 等多语言），数据来源于 GitHub Search API、自动关键词过滤与人工审核。

**📈 对比分析**

采用多种统计检验与时序分析方法，结果显示构建块使用完整度与项目活跃度、星标、团队规模正相关；边界违规率与 bug 修复频率呈显著正相关；构建块演化与 DDD‑特定重构提交之间存在显著的时间延迟关系。

**⚠️ 局限性**

研究依赖关键词过滤，可能产生误判（假阳性/假阴性）；边界推断仅基于包/模块结构，对使用非传统结构的项目不适用；缺乏对实际代码质量或可维护性度量的直接评估；LLM 辅助分类的可信度需进一步验证。

---

## 74. Trustworthy Image Authentication using Forensic Knowledge Graphs

**arXiv ID:** 2606.23917 | [PDF](https://arxiv.org/pdf/2606.23917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 75. Social Structure Matters in 3D Human-Human Interaction Generation

**arXiv ID:** 2606.24255 | [PDF](https://arxiv.org/pdf/2606.24255v1)

**作者:** Zhongju Wang `[一作]` (University of New South Wales), Zhenhong Sun `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文将文本驱动的3D人类-人类交互生成问题表述为社交结构建模与落地任务，使用LLM规划阶段与角色信息，并通过Solo-to-Social框架将单人运动模型适配为双人交互执行器；

**💡 创新点**

创新点在于：①首次将社交结构（阶段进程与伙伴角色）视为生成的核心中间表征；②证明LLM仅适合规划而非直接生成连续运动；③提出planner‑executor范式，将LLM规划与LoRA调优的单人运动模型相结合；

**🔧 技术方法**

核心技术包括LLM（Qwen3.5）进行阶段拆分与角色分配、基于SMPL的运动事实约束、Solo-to-Social执行器中的自回归自我条件、伙伴相对条件、LoRA参数高效迁移、以及流匹配损失的细化；

**📊 数据集**

使用InterHuman和InterX两大SMPL‑based人类交互数据集进行训练与评估；

**📈 对比分析**

与五种传统双人生成基线（ComMDM、in2IN、InterGen、InterMask、TIMotion）以及Plan+HYM1进行对比；在R‑Precision、FID、用户评测以及多模态度量上均优于基线，特别是阶段一致性与伙伴协调表现突出；

**⚠️ 局限性**

局限性包括：①对LLM的规划结果仍需依赖真实运动事实约束，无法完全自洽；②仅适用于已标注阶段与角色的交互场景，未探索跨领域推广；③对极长时序或多参与者交互的处理尚未验证；

---

## 76. Breaking the Filter Bubble: A Semantic Pareto-DQN Framework for Multi-Objective Recommendation

**arXiv ID:** 2606.24042 | [PDF](https://arxiv.org/pdf/2606.24042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 77. From Spatial to Spectral: An Efficient, Frequency-Guided Feature Representation Learner for Small Object Detection

**arXiv ID:** 2606.23825 | [PDF](https://arxiv.org/pdf/2606.23825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 78. Low-Complexity Hybrid Precoding for Cell-Free Massive MU-MIMO ISAC Systems

**arXiv ID:** 2606.23709 | [PDF](https://arxiv.org/pdf/2606.23709v1)

**作者:** Jun Zhu `[一作]`, Wenjun Zhang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种低复杂度混合前置框架，用于基于部分连接RF的无基站（CF）大规模多用户MIMO ISAC系统，实现通信与多目标感知的协同优化。

**💡 创新点**

创新点包括：①通过部分连接前置将高维信道投影到低维有效信道，显著降低上行信道信息的法线负载；②构建带PEB约束的ESR最大化问题，采用AO+凸逼近+Riemannian梯度实现双向优化；③设计多分支Rate Splitting+MMSE‑THP更新算法，避免在用户拓扑变化时重复完整矩阵重计算，实现计算复杂度降低至87.02%；

**🔧 技术方法**

技术手段：混合波束成形、部分连接RF架构、AO优化、凸逼近与SCA、Riemannian共轭梯度、Rate Splitting、Tomlinson‑Harashima前置、MMSE前置、Sherman‑Morrison更新、FOM分析。

**📊 数据集**

数据集：基于仿真平台，L=7 AP、每AP 64 天线、16 用户、4 个感知目标、CSI 误差方差 σ_e^2=0.05，覆盖不同 SNR、不同 N_RF（16/32）及不同分支数（0/4）的情形。

**📈 对比分析**

与传统全连接线性预编码（ZF、MMSE）、MMSE‑THP、单分支等方案对比，ESR 在满足 PEB<PEB_th 的前提下保持接近；在相同信噪比下实现相似或更优的ESR；同时复杂度显著降低（如 MB‑RS‑MMSE‑THP 更新版比原版降低 87.02%），并保持可观的波束峰值和定位精度。

**⚠️ 局限性**

局限性：1）仍依赖准确的CSI估计；2）部分连接带来性能折衷，需在复杂度与性能间权衡；3）实验验证缺失，仅基于仿真；4）实现对硬件和同步要求较高，实际部署需进一步验证。

---

## 79. Selective Capability Unlearning in End-to-End Spoken Language Understanding

**arXiv ID:** 2606.24063 | [PDF](https://arxiv.org/pdf/2606.24063v1)

**作者:** Akanksha Singh `[一作]` (Indian Institute of Science Education and Research Bhopal), Vinod Kumar Kurmi `[通讯]` (Indian Institute of Science Education and Research Bhopal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Binding Subspace Unlearning (BSU)，一种针对端到端语音理解系统的可选择性功能去除框架，用于消除意图条件下的槽生成残留能力。

**💡 创新点**

通过识别并削弱意图‑槽绑定的表示子空间，解决自回归SLU模型中存在的“能力持续”问题，实现对功能的精准去除。

**🔧 技术方法**

使用隐藏状态协方差对比提取绑定子空间，再在训练中加入子空间梯度正则化，并结合遗忘‑保留损失与KL正则化实现去除。

**📊 数据集**

在SLURP和SpeechMASSIVE（法语子集）这两个端到端SLU基准数据集上进行实验。

**📈 对比分析**

与GA、GA+GD/KL、NPO、NPO+KL以及随机标签等传统去除方法对比，BSU在FORGET集的BRR@10和语义相似度等指标下降约60%‑70%，同时保留集性能基本不变。

**⚠️ 局限性**

目前仅针对单一意图去除，未验证多意图或更复杂对话场景；子空间定位依赖统计方差，极小样本意图效果可能受限；实验仅覆盖端到端SLU，需进一步推广至其他生成任务。

---

## 80. SP-Mind: An Autonomous Reasoning Agent for Spatial Proteomics Analysis

**arXiv ID:** 2606.24235 | [PDF](https://arxiv.org/pdf/2606.24235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 81. CAVEWOMAN: How Large Language Models Behave Under Linguistic Input and Output Compression

**arXiv ID:** 2606.24083 | [PDF](https://arxiv.org/pdf/2606.24083v1)

**作者:** Morayo Danielle Adeyemi `[一作]` (Independent), Franck Dernoncourt `[通讯]` (Adobe Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了Cavewoman两通道压缩评估协议，对输入压缩与输出压缩在多模型、多数据集上进行系统评测。

**💡 创新点**

创新点在于同时测量任务准确性、实际成本和生成文本与无压缩基线的一致性，揭示输入压缩往往提升成本而输出压缩可显著节省。

**🔧 技术方法**

采用基于词性过滤的确定性压缩、双向NLI判断文本一致性以及对八款模型的API与本地推理成本测算。

**📊 数据集**

使用了GSM8K、BoolQ、ARC‑Easy、CommonsenseQA和MMLU‑STEM五个标准数据集。

**📈 对比分析**

在每个模型与数据集下对五个压缩等级进行对比，发现输出压缩可使GPT‑4o、Claude等模型实现1.4–3倍成本下降，而输入压缩则相反；在输出压缩下，约52%正确答案的表面文本与无压缩基线不一致。

**⚠️ 局限性**

局限包括仅评估短答案的数值/判断/单选题，未覆盖长文本生成；仅用贪心解码；未检验不同提示或解码策略的影响。

---

## 82. SemChunk-C: Semantic Segmentation for C Code

**arXiv ID:** 2606.23697 | [PDF](https://arxiv.org/pdf/2606.23697v1)

**作者:** Boris Nazarov `[一作]` (Huawei), Pavel Kisilev `[通讯]` (Huawei)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出SemChunk-C，一种用于C族语言的轻量级语义分块方法。

**💡 创新点**

创新点在于利用小型编码器Fine-tune实现可变边界的语义分块，并为每块自动标注功能类别。

**🔧 技术方法**

技术手段包括Ettin编码器的预训练与微调、LLM（Qwen2.5-Coder）生成训练标签、token级分类与多类别标注。

**📊 数据集**

使用公开的多语言仓库（C/C++/C#/.h/.m/.mm）共计约27B tokens训练，Oracle Chunker生成约73k个标注实例。

**📈 对比分析**

通过手工与自动测试集评估，17M-150M模型在块边界和类别精度上可与Tree‑sitter、甚至7B LLM匹敌，并在代码检索和生成任务中提升了nDCG@10和pass@1/10。

**⚠️ 局限性**

局限性包括对宏展开和嵌套结构的鲁棒性仍有限、数据集规模受限、以及在极大项目中需要分块递归处理。

---

## 83. QuechuaTok: Morphological Boundary Accuracy as a Necessary Metric for Tokenizer Evaluation in Agglutinative Low-Resource Languages

**arXiv ID:** 2606.23943 | [PDF](https://arxiv.org/pdf/2606.23943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 84. Engineering Reliable Autonomous Systems: Challenges and Solutions

**arXiv ID:** 2606.23760 | [PDF](https://arxiv.org/pdf/2606.23760v1)

**作者:** Marie Farrell `[一作]` (University of Manchester), Huan Zhang `[通讯]` (Maynooth University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本报告综述了ERAS工作坊的讨论，聚焦验证、工程实践与软件架构三大主题，并提出未来研究路线图；

**💡 创新点**

创新点在于整合FMAS与AREA社区，提出跨学科协作框架和统一的验证与架构指导；

**🔧 技术方法**

采用形式方法、运行时验证、机器学习验证、案例分析与多种工具技术；

**📊 数据集**

使用跨领域案例（太空抓取、矿山UAV、医疗 triage、工业机器人、海底 AUV、移动车辆、家庭助手机器人、协作制造、消防无人机、核设施、军事场景）进行验证与评估；

**📈 对比分析**

与现有方法对比，强调在多域可验证性、可解释性和持续保障方面的改进，性能表现优于传统单域方法；

**⚠️ 局限性**

局限性在于缺乏统一标准、实测评估、对开放环境完整验证的支持，以及工具与方法在工业规模上的可迁移性不足。

---

## 85. DramaDirector: Geometry-Guided Short Drama Generation

**arXiv ID:** 2606.24107 | [PDF](https://arxiv.org/pdf/2606.24107v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 86. Low-power analogue neural networks with trainable nonlinear connections for continuous control

**arXiv ID:** 2606.23742 | [PDF](https://arxiv.org/pdf/2606.23742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 87. Flood-It with Jewelry -- Characterizing the Game Complexity for Cograph Generalizations

**arXiv ID:** 2606.23837 | [PDF](https://arxiv.org/pdf/2606.23837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

---

## 88. One Year Later...The Harms Persist, But So Do We!

**arXiv ID:** 2606.23884 | [PDF](https://arxiv.org/pdf/2606.23884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 89. Federated Survival Analysis in Healthcare: A Multi-Model Evaluation on Cross-Institutional Heterogeneous Breast Cancer Data

**arXiv ID:** 2606.23871 | [PDF](https://arxiv.org/pdf/2606.23871v1)

**作者:** Natalia Moreno-Blasco `[一作]` (University of Oulu), Miguel Fernandez-de-Retana `[通讯]` (University of Deusto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在跨机构异构乳腺癌数据集上，系统评估了CoxPH、DeepSurv和RSF三种生存模型在集中式、本地及联邦学习（FedAvg/FedProx/FedAdam）下的性能，并给出了相应的决策指南。

**💡 创新点**

首次在联邦框架中对多模型进行跨机构异构数据集的全面对比，揭示联邦训练在不共享原始数据的前提下可接近甚至超越集中式模型，并基于数据特性、隐私与资源约束提出实用的模型与训练范式选择建议。

**🔧 技术方法**

采用联邦学习框架Flower，结合Cox比例风险模型、DeepSurv深度神经网络、生存随机森林（FedSurF++），并对FedAvg、FedProx、FedAdam等联邦优化策略进行评估。

**📊 数据集**

使用Fed-TCGA-BRCA（TCGA乳腺癌临床表型）构成的FLamby基准集合，包含6家中心的异构分布。

**📈 对比分析**

通过C-Index、AUC和IBS三项指标，在不同客户端数量（5、4、3）下进行10次随机实验平均；联邦训练通常优于本地，接近或超过集中式（尤其DeepSurv），RSF在所有指标上表现最稳健。

**⚠️ 局限性**

受限于样本量、极端异构导致校准随客户端减少而恶化；联邦优化策略受数据分布差异影响；未考虑个性化/多任务学习、正式差分隐私和模型可解释性提升等扩展方向。

---

## 90. Canopies: A Generalization of Vines and Vineyards for Parameterized Persistence

**arXiv ID:** 2606.23859 | [PDF](https://arxiv.org/pdf/2606.23859v1)

**作者:** Barbara Giunti `[一作]` (University at Albany -- SUNY), Elizabeth Munch `[通讯]` (Michigan State University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文提出了新的“冠状（canopy）”构造，用来描述参数化持久化的结构，并给出了两种变体：A‑canopy 与 D‑canopy；

**💡 创新点**

创新点在于将持久化图中的点映射到产生它们的单纯形对，利用链复形的代数结构来定义拓扑，进而统一处理多重性问题，给出正式的“藤（vine）”与“单环效应（monodromy）”定义；

**🔧 技术方法**

主要技术包括过滤链复形的分解与配对规则、组合等价关系、轨迹（trajectory）定义与拓扑结构、纤维丛理论以及Wasserstein距离等持久化图度量；

**📊 数据集**

作者没有使用具体的数据集，而是通过理论示例（如四边形单纯形复合 L、PHT 等）来演示构造和性质；

**📈 对比分析**

本文未进行实验比较或性能评估，所有结果均为理论证明与结构定理；

**⚠️ 局限性**

局限性在于仅适用于固定的有限单纯形复合、需要Hausdorff基空间，且对结构点（非Hausdorff点）仍缺乏完整的处理方法，无法完全解决多重性与单元间交换的实际案例。

---

## 91. AutoPRAC: Automating Attack Discovery for PRAC-Based Rowhammer Defenses using Model Checkers

**arXiv ID:** 2606.23905 | [PDF](https://arxiv.org/pdf/2606.23905v1)

**作者:** Joyce Qu `[一作]` (University of Toronto), Gururaj Saileshwar `[通讯]` (University of Toronto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

开发了 AutoPRAC，一种基于有限状态机和 Bounded Model Checking 的自动化攻击发现框架，用来检测 DDR5 PRAC（Per-Row Activation Counting）防御设计的安全缺陷。

**💡 创新点**

创新点在于：①首次将模型检查应用于 PRAC 防御，提供正式安全属性验证与攻击回溯；②通过两项优化（初始化计数器至阈值-1、使用计数直方图抽象）显著降低状态空间；③在 MOAT 的计数器重置策略中自动发现了先前未被报告的 34 次激活可被忽略的漏洞。

**🔧 技术方法**

使用技术包括：C 语言编写 PRAC 模型；CBMC（基于 SAT 的 BMC 工具）进行状态探索；自定义安全属性（最大激活数不超过 Rowhammer 阈值）；以及抽象建模与优化技巧。

**📊 数据集**

实验数据集由四种公开实现的 PRAC 防御（Panopticon、UPRAC、QPRAC、MOAT）构成，使用默认参数（mitigation level 1、BL 128、blast radius 1）。

**📈 对比分析**

与人类手工设计的攻击比较：AutoPRAC 在 MOAT 上发现了 175 次激活的攻击（比原文 161 次更强），并能在 2 小时内验证大部分防御；对其他防御的攻击力度有限（最多 132–141 次），表明模型检查在发现极端攻击上具有潜力，但目前仍受限于求解器时间和状态空间爆炸。

**⚠️ 局限性**

局限性包括：①求解器可扩展性不足，难以覆盖完整 tREFW 内的 550k 次激活；②模型仅考虑固定的 blast radius，无法处理衰减式干扰和 Ripple 攻击；③未覆盖 PRAC 的所有优化（主动补偿、双阈值等）；④模型基于 C 代码，缺乏 RTL 级别验证；⑤无法在 2 小时内找到更长或更复杂的攻击路径。

---

## 92. Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models

**arXiv ID:** 2606.24165 | [PDF](https://arxiv.org/pdf/2606.24165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 93. Importing soundness and completeness in modal logics

**arXiv ID:** 2606.23852 | [PDF](https://arxiv.org/pdf/2606.23852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 94. Predicting Poets' Origins from Verse: A Computational Analysis of Regional Linguistic Fingerprints in the Complete Tang Poems

**arXiv ID:** 2606.24093 | [PDF](https://arxiv.org/pdf/2606.24093v1)

**作者:** Chi-Sheng Chen `[一作]` (Harvard University), Hung-Yun Liu `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用机器学习分析唐诗文本，探究诗人出生地域是否会在其作品中留下可检测的语言痕迹，并将这些模型误判用作文学史假设生成。

**💡 创新点**

创新点在于将可解释的特征（图像、季节、典故密度等）与传统字符 n-gram TF‑IDF 结合，并发现区域信号随地理距离衰减、随历史时期波动；模型误判的历史意义亦被系统化解读。

**🔧 技术方法**

采用逻辑回归、线性 SVM、随机森林、MLP、以及 Fine‑tuned 经典中文 BERT（GuwenBERT）等模型，特征包括字符 1–2 gram TF‑IDF 与手工解释域特征；对 BERT 采用层级冻结编码以公平对比。

**📊 数据集**

使用《全唐诗》（约 49,000 句、2,200 名诗人）与中国历史人物传记数据库（CBDB）对诗人归属的行政道（10 个地区）进行标注，最终得到 357 名诗人（≥5 句）组成的语料。

**📈 对比分析**

在南北两区的二分类任务中，MLP 在 5 折交叉验证下达到 0.69 的准确率与宏 F1，显著高于 0.53 的多数类基线；三区宏 F1 为 0.43，十区宏 F1 为 0.18，均大约是基线的两倍。GuwenBERT 在完整文本编码下与 TF‑IDF 达到相同表现，融合两者无提升。

**⚠️ 局限性**

主要限制包括样本量有限（仅 357 名诗人，按地区分布不均），地理标签和文本归属存在噪声；音韵（平仄）特征难以完整实现；距离衰减的显著性仅为 p≈0.09，且受限于仅使用行政道中心点；对唐朝不同时期的细粒度分析受诗人数不足影响。

---

## 95. Forget Without Compromise: Nexus Sampling for Streaming KV-Cache Eviction Under Fixed Budgets

**arXiv ID:** 2606.23961 | [PDF](https://arxiv.org/pdf/2606.23961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 96. Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent

**arXiv ID:** 2606.24094 | [PDF](https://arxiv.org/pdf/2606.24094v1)

**作者:** Wenliang Zhong `[一作]` (University of Texas at Arlington), Junzhou Huang `[通讯]` (University of Texas at Arlington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种通用的基于文本指南的图像聚类框架，结合生成概念代理模型（GCPM）和基于最小生成树的LLM遍历（MST Traversal），实现无任务专用训练的多场景聚类。

**💡 创新点**

① 通过 GCPM 将文本指南映射为概念代理并嵌入，实现场景属性的解耦；② MST-based LLM Traversal 极大降低 LLM 调用次数，仅在需要时进行语义推理；③ 统一覆盖一般、复合、多细粒度和长尾分布等多种聚类任务。

**🔧 技术方法**

使用多模态大语言模型（MLLM）进行概念代理生成、指令感知嵌入器、K‑Means / HDBSCAN 聚类算法、最小生成树（MST）构造与遍历，以及 LLM 语义判断。

**📊 数据集**

CIFAR-10、STL-10、ImageNet-10（一般聚类）；Fruit、Cards、CIFAR10-MC（多重聚类）；CUB Birds、Stanford Dogs、Stanford Cars、Oxford Flowers（细粒度聚类）；ABO-LC（长尾电商聚类）。

**📈 对比分析**

与多种基准（K‑Means、HDBSCAN、IC|TC、Multi‑Sub、DiFiC 等）对比，实验表明 GCPM+MST 在 GC、MC、FC、LC 任务中均达到或逼近最优性能，显著优于现有训练型方法；在无监督设置下获得高 ACC/NMI/ARI，并且 MST Traversal 在提升召回率方面效果突出。

**⚠️ 局限性**

依赖 LLM 生成的指南与提示，若提示不准确可能导致误合并；MST Traversal 仍需 LLM 推理，对极大规模数据可能存在计算瓶颈；对单样本和不同视觉域的泛化仍有待进一步验证。

---

## 97. Token Complexity of Certifying Stochastic-Oracle Reliability

**arXiv ID:** 2606.24074 | [PDF](https://arxiv.org/pdf/2606.24074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 98. One Ruler: A Same-Hands Re-Evaluation of Bivariate Causal Direction on Tuebingen, with a Parameter-Free Compression Baseline

**arXiv ID:** 2606.23767 | [PDF](https://arxiv.org/pdf/2606.23767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 99. Sphere of Influence Centrality via Shapley Values: Empirical Approximation and Network Coverage Analysis

**arXiv ID:** 2606.24121 | [PDF](https://arxiv.org/pdf/2606.24121v1)

**作者:** Sie Hendrata Dharmawan `[一作]` (Dartmouth), Peter Chin `[通讯]` (Dartmouth)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

评估了基于Shapley值的球面影响中心性在三种可达性模型下在三大真实网络上的效果，并与基于度的基线进行了对比；

**💡 创新点**

创新点在于实现Michalak等人给出的三种游戏的精确Shapley值算法，并在实际数据上验证其近似比远超理论1‑1/e下界，同时揭示不同可达性模型在真实网络中的表现差异；

**🔧 技术方法**

采用多项式时间的精确Shapley值计算方法，结合单跳、k跳、以及多路径连通性模型，并使用度基线作为对照，进行随机与真实网络的实验评估；

**📊 数据集**

使用的真实数据集包括欧洲道路网络（Euroroad）、Facebook电视节目页面网络（Facebook TV Shows）以及机器学习论文引用网络（Cora）；

**📈 对比分析**

通过将Shapley值选取、度基线和穷举最优三者在同一网络与参数下进行比较，发现Shapley在异质拓扑下明显优于度基线，平均近似比约0.9，尤其在Cora 3‑hop模型下仅需26个节点即可覆盖50%网络；

**⚠️ 局限性**

局限性包括仅处理无权图，k‑跳模型的计算复杂度较高，未考虑动态网络或边权异质性，以及在同质随机网络中优势不明显。

---

## 100. World Artificial Intelligence Cooperation Organization (WAICO): Mapping an Emerging Institution in the Global AI Governance Regime Complex

**arXiv ID:** 2606.23860 | [PDF](https://arxiv.org/pdf/2606.23860v1)

**作者:** William Guey `[一作]` (Tsinghua University), José O. Gomes `[通讯]` (Federal University of Rio de Janeiro)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对中国拟议的世界人工智能合作组织（WAICO）在全球AI治理体系中的定位进行编码比较，构建跨15个机构的结构与价值维度矩阵；

**💡 创新点**

首次将WAICO纳入AI治理机构复合体，揭示其开放性、缺失价值门槛及发展导向的独特组合，填补了现有机构空缺；

**🔧 技术方法**

采用定性文本编码与定量指标（规范取向、成员宽度、发展重视度、正式化程度）相结合的方法；

**📊 数据集**

基于公开文件的15份AI治理机构与工具的原始文本；

**📈 对比分析**

通过二维图谱对比各机构的规范取向与成员广度，直观展示WAICO在空缺位置，未涉及具体性能评估；

**⚠️ 局限性**

研究依赖公开文本编码，缺乏互评可靠性，且仅覆盖15个机构，未来需扩展全体机构并验证WAICO实际行为与预期

---

## 101. FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning

**arXiv ID:** 2606.24231 | [PDF](https://arxiv.org/pdf/2606.24231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 102. Cyclic Denoising Reveals Ultrastable Memories in Diffusion Models

**arXiv ID:** 2606.24000 | [PDF](https://arxiv.org/pdf/2606.24000v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 103. Listening makes Vision Clear for VLMs

**arXiv ID:** 2606.23763 | [PDF](https://arxiv.org/pdf/2606.23763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 104. MMed-Bench-IR: A Heterogeneous Benchmark for Multilingual Medical Information Retrieval

**arXiv ID:** 2606.24200 | [PDF](https://arxiv.org/pdf/2606.24200v1)

**作者:** Junhyeok Lee `[一作]` (Seoul National University), Kyu Sung Choi `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MMed-Bench-IR三任务多语言医学检索基准，评估跨语言对齐、概念辨别和证据检索三大能力。

**💡 创新点**

首次将跨语言与医学专业化同时量化，并通过零概念、零查询重叠设计保证综合得分能真实反映模型多维能力；同时引入公平性间隙指标揭示语言不平等。

**🔧 技术方法**

采用UMLS本体映射、三维难度分层、双向翻译质量保证、nDCG@10、R@1等评估指标；实验使用BGE-M3、E5、SapBERT、BM25、ColBERT-XM等多范式检索器，并对BGE-M3进行医学多语言对比学习微调。

**📊 数据集**

使用UMLS 2025AB词表、BioASQ 13b 质检问答、MMedBench多语言QA、NLLB-200 翻译、以及自构造的概念混淆集等数据集。

**📈 对比分析**

对十个系统在六大范式（词典、单语医学、跨语言密集、延迟交互、混合、两阶段重排序、医学多语言）进行基准测试；结果显示5倍性能跨度，层级一致，概念辨别最难，医学模型在非拉丁语种出现显著交叉语言失效，最优方案为MMed-Embed+重排序，公平性间隙最小。

**⚠️ 局限性**

限制包括语言覆盖不足（缺乏阿拉伯语、印地语等高需求语言）、对日语和中文翻译质量有限、混淆集规模偏小导致难度层三统计不稳、全部使用自动评估而非人工专家标注、仅测试了六类模型范式，后续需要更多语言与模型的社区提交与专家校验。

---

## 105. A Comparative Study of Bayesian Contextual Bandits for Real-Time Warehouse Sorter Optimization

**arXiv ID:** 2606.23977 | [PDF](https://arxiv.org/pdf/2606.23977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 106. The Geometry Behind Diffusion and Flow Matching: Gradient Flows and Geodesics in Wasserstein Space

**arXiv ID:** 2606.24157 | [PDF](https://arxiv.org/pdf/2606.24157v1)

**作者:** Yian Yao `[一作]` (mortal world), Weiwei Zhang `[通讯]` (heaven)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

论文通过 Wasserstein 里程空间的几何结构，将扩散模型与 Flow Matching 统一在同一框架下：扩散模型对应自由能的 Wasserstein 梯度流（JKO 隐式欧拉离散），而 Flow Matching 对应 Wasserstein geodesic（最小作用路径）。

**💡 创新点**

创新点在于：
• 把两类主流生成模型归结为同一 Riemannian 流动的两种极值问题（初值问题 vs. 边值问题）。
• 证明每一步 DDPM/DDIM/VE‑SDE 等去噪步骤实际上是 JKO 的单一步；
• 通过 Benamou‑Brenier 公式和 Fokker‑Planck 方程，将传统的 SDE/score 解释为概率流的速度场，从而揭示两者的本质相同。

**🔧 技术方法**

使用的技术主要包括：
• Wasserstein（$W_2$）距离及其动量公式；
• 连续性方程和 Fokker‑Planck 方程的推导；
• Otto 运动学框架下的 Riemannian 结构与梯度流；
• JKO（Jordan‑Kinderlehrer‑Otto）分步优化；
• Benamou‑Brenier 最小作用公式与 optimal‑transport geodesics；
• 变分推导与泛函微分。

**📊 数据集**

论文为理论性工作，未针对具体数据集进行实验。它对已有模型（DDPM、DDIM、VE‑SDE、NCSN、Energy Matching 等）做统一性分析，若需实验可使用公开的图像/音频/文本数据集（如 CIFAR‑10、ImageNet、MNIST 等）。

**📈 对比分析**

方法比较方式：通过数值离散（JKO）与经典 SDE 反向采样的对应关系来验证等价性，理论上证明了梯度流与 geodesic 的相等性。实验上可通过对比采样步数、收敛速度、生成质量（FID、IS）等指标，但论文并未给出新的数值实验。若实验，性能可与原始 Diffusion/Flow Matching 方法保持一致或略有提升。

**⚠️ 局限性**

局限性：
• 需要概率密度满足绝对连续、二阶矩有限等正则性假设；
• 对离散分布或高维稀疏数据的解析不够完善；
• 实际采样中对 ODE/ SDE 的数值误差、时间重参数化会影响理论对应关系；
• JKO 离散化在高维场景下计算代价较大，实际实现需要近似。

---

## 107. Towards Spec Learning: Inference-Time Alignment from Preference Pairs

**arXiv ID:** 2606.24004 | [PDF](https://arxiv.org/pdf/2606.24004v1)

**作者:** Dhriti Krishnan `[一作]` (Carnegie Mellon University), Jaromir Savelka `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本文中，作者提出了一种名为 Spec Learning 的框架，利用简短的用户指令和少量（约 20 条）偏好判断，编译出自然语言形式的系统提示（spec），在推理阶段直接传给大型语言模型（LLM）以控制其行为，无需更新模型权重。

**💡 创新点**

创新点在于将偏好信号压缩为可读的自然语言原则，并在推理时通过拼接提示实现行为控制，从而避免昂贵的参数更新和繁琐的手工提示工程，同时提供可解释的、可编辑的对齐规则。

**🔧 技术方法**

技术方法包括：1）使用 proposer（如 Gemma‑4/DeepSeek/ Kimi‑K2.6）生成候选原则；2）通过语义聚类、去重和交叉验证筛选有效原则；3）使用 synthesizer（janus 或 bullets）将原则合成单一系统提示；4）在推理时将提示与用户查询拼接，使用原始 LLM 生成响应；5）利用 LLM‑as‑Judge（GLM‑5.1）进行评估。

**📊 数据集**

实验使用了七个偏好数据集，覆盖从技术性（多步数学推理、代码安全、Stack‑Exchange Q&A）到情感/专业性（心理治疗回应、事实谦逊）以及开放式（Anthropic HH‑Helpful）等领域，具体包括：Math‑DPO、Code‑Pref、Code‑Security、Stack‑Exchange、PsyCoPref、Truthy‑DPO、HH‑Helpful。

**📈 对比分析**

通过与使用 1,000 条偏好对训练得到的 DPO 模型以及基准未加提示的 LLM 进行三方比较（spec vs. 基准、spec vs. DPO、DPO vs. 基准），采用 LLM‑Judge 的成对判定得到 win‑rate。结果显示 Spec Learning 在所有 7 个数据集上均优于 DPO，宏观平均 win‑rate 0.75（相对 DPO 的 0.71），尤其在任务定义明确、偏好信号集中的数据集（Stack‑Exchange、Code‑Pref、Truthy‑DPO、Math‑DPO）表现更佳。

**⚠️ 局限性**

局限性包括：1）当偏好信号分散且难以用少数规则表达（如 HH‑Helpful）时，Spec Learning 效果差；2）仅在单一模型族上验证，跨模型可迁移性尚未实验；3）评估完全基于 LLM‑Judge，缺乏人工评测；4）需要强大的 proposer/judge/ synthesizer 以及 API 访问；5）推理时提示长度带来的成本随流量线性增长；6）易受提示注入/对抗攻击，透明规则可能导致安全误判。

---

## 108. Catastrophic Compositional Generation: Why Vanilla Diffusion Models Fail to Extrapolate

**arXiv ID:** 2606.23920 | [PDF](https://arxiv.org/pdf/2606.23920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 109. DREG: A Layer-Wise Jacobian Regularization as a General-Purpose Penalty

**arXiv ID:** 2606.23942 | [PDF](https://arxiv.org/pdf/2606.23942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 110. Exploring Academic Influence of Algorithms by Co-occurrence Network Based on Full-text of Academic Papers

**arXiv ID:** 2606.24099 | [PDF](https://arxiv.org/pdf/2606.24099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 111. RoPE-Aware Bit Allocation for KV-Cache Quantization

**arXiv ID:** 2606.24033 | [PDF](https://arxiv.org/pdf/2606.24033v1)

**作者:** Fengfeng Liang `[一作]` (Hong Kong University of Science and Technology), Jiaya Jia `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Block‑GTQ，一种针对 RoPE 结构的 KV 缓存低位量化方案，通过按频率块分配量化位来提升缓存压缩后的注意力精度；

**💡 创新点**

创新点在于将 KV 缓存量化视为块级比特分配问题，利用 RoPE 频率块能量无标签评分和贪婪分配，显著减少关键向量量化误差；

**🔧 技术方法**

采用 TurboQuant‑MSE 作为本地量化器，结合能量评分的 4^-b 误差律和贪婪位分配算法，另外实现了无 fp16 缓存的 packed‑cache 执行路径；

**📊 数据集**

在十模型诊断面板（Qwen、Llama、DeepSeek、Mistral、GLM 等）以及长上下文检索（NIAH）、长文本推理（LongBench‑EN）和 AIME 2024/25 评测数据集上验证；

**📈 对比分析**

与统一分配 TQ‑MSE、KIVI‑ScaleOnly、PM‑KVQ 等方法比较，Block‑GTQ 在 K‑only 2/3 b/dim 下均能将 RoPE‑logit MAE 降至 32–80%，在 NIAH、LongBench‑EN 与 AIME 上几乎与 fp16 旗鼓相当；在单张 H800 GPU 上，packed‑cache 路径实现 3.24× KV 压缩、1.34× 解码加速，并能在 256K/512K 长上下文下保持可用；

**⚠️ 局限性**

局限性：仅对 K 分配位宽，V 采用统一分配；未实现多 GPU 或批量推理；packed‑cache 方案仅为单 GPU 实现，需进一步扩展。

---

## 112. A Geometry-Informed Computer Vision Method for Detecting and Examining Overtaking Vehicles From A Bicycle

**arXiv ID:** 2606.23699 | [PDF](https://arxiv.org/pdf/2606.23699v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 113. Decentralized Coordination of Autonomous Traffic Through Advanced Air Mobility Corridors

**arXiv ID:** 2606.23832 | [PDF](https://arxiv.org/pdf/2606.23832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 114. PORTER: Language-Grounded Event Representations for Portable Structured EHR Foundation Models

**arXiv ID:** 2606.24102 | [PDF](https://arxiv.org/pdf/2606.24102v1)

**作者:** Lin Lawrence Guo `[一作]` (Hospital for Sick Children), Lillian Sung `[通讯]` (Hospital for Sick Children)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了可跨词汇表、跨机构的语言基线结构化 EHR 基础模型 PORTER，利用事件描述和数值通道进行自监督预训练，生成可重用的患者表示。

**💡 创新点**

将事件语义与数值分离，使用冻结文本编码器将事件描述映射为词汇表无关嵌入，并通过 FiLM 集成数值；实现无需重训练即可跨词汇表迁移，且通过自回归预训练保留任务无关的时间动态。

**🔧 技术方法**

冻结 BioLORD 文本编码器、FiLM 数值通道、因果 Transformer 背骨、下一事件自回归预训练、线性探针微调及对比实验。

**📊 数据集**

在儿童医院 SickKids 的 EHR（OMOP 与 SEDAR 词表）上预训练，评估 74 个临床预测任务；外部 MIMIC-IV（成人 ICU）用于跨站迁移。

**📈 对比分析**

与匹配架构的固定词汇表基础模型和任务特定文本序列化模型对照；PORTER 与固定词汇表在 74 任务上性能相当，跨词汇表迁移恢复 97.1% AUROC，跨站 31/36 任务表现更好；相较文本序列化模型，PORTER 在 69/74 任务上 AUROC 更高、样本效率更佳、计算成本降低 329 倍。

**⚠️ 局限性**

仅在单家儿科医院预训练，使用二分类线性探针，未评估微调/回归/零样本等更复杂场景；冻结文本编码器，输出端仍使用固定词汇表，且依赖事件描述与数值信息的可用性与质量。

---

## 115. A Survey on Federated Causal Discovery and Inference

**arXiv ID:** 2606.23741 | [PDF](https://arxiv.org/pdf/2606.23741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. When Top-1 Fails: Calibrating LoRA Monitors for Masked Diffusion LMs

**arXiv ID:** 2606.24119 | [PDF](https://arxiv.org/pdf/2606.24119v1)

**作者:** Lucky Verma `[一作]` (Independent Researcher), Pratik Yadav `[通讯]` (University of Maryland, Baltimore County)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了离散扩散语言模型（DLM）在LoRA微调过程中的训练监测器，检验了现有的Top‑1集中度警告是否能提前识别训练崩溃；

**💡 创新点**

提出了最大LoRA梯度范数作为DLM LoRA训练的家庭局部预警信号，能够在短跑（≤200步）内以≈0.68精度识别最差10%最终损失配置；

**🔧 技术方法**

采用LoRA/PEFT微调、梯度范数记录、Top‑1集中度统计以及与自回归模型的对照实验；

**📊 数据集**

使用LLaDA系列（8B、15.93B、7B等）DLM、Pythia、Qwen、Dream和MDLM-OWT等公开模型，在多组mask率与LoRA秩组合下共计816个配置；

**📈 对比分析**

与原始Top‑1警告（精度0）和自回归控制对照相比，最大梯度范数在LLaDA族内的验证集上实现了0.68精度、0.94召回、F1≈0.79；在随机拆分下精度稳健，且在训练第25步后即可使用；

**⚠️ 局限性**

局限性包括仅验证短期（≤200步）微调、仅针对LoRA放置在注意力投影、未跨模型族统一阈值、对高mask率或不同学习率敏感、未评估生成质量或长期训练稳定性。

---

## 117. Co-occurring associated retained concepts in Diffusion Unlearning

**arXiv ID:** 2606.24192 | [PDF](https://arxiv.org/pdf/2606.24192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 118. Layer-wise Probing of wav2vec 2.0 and Whisper for Consonant Cluster Reduction in African American English

**arXiv ID:** 2606.23948 | [PDF](https://arxiv.org/pdf/2606.23948v1)

**作者:** Hamid Mojarad `[一作]` (Heinrich Heine University Düsseldorf), Kevin Tang `[通讯]` (Heinrich Heine University Düsseldorf)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对wav2vec2-base与Whisper-small的隐藏层表示进行线性探测，研究美国非裔英语（AAE）中的辅音簇约简（CCR）如何被内部编码。

**💡 创新点**

首次证明现代自监督与监督语音模型将CCR视为结构化、梯度化的语音变体而非简单的段位删除，揭示模型对潜在音位信息的保留与恢复能力。

**🔧 技术方法**

采用基于Transformer的wav2vec2-base与Whisper-small、均匀/不均匀层级线性分类器（单隐藏层MLP）对隐藏状态进行逐层探测，并构造segmental reduction detection与segmental restoration两种任务。

**📊 数据集**

使用CORAAL语料库中的156名AAE说话者（含地区、性别、年龄、社会经济层面），筛选7类高频双辅音簇，共计6760个标注实例。

**📈 对比分析**

通过speaker‑independent 4‑fold CV对不平衡/平衡数据集进行对比，发现分层检测精度约70–80%，而恢复检测精度可达93–96%，表明两模型在中后层对CCR信息的表达相似且表现优秀。

**⚠️ 局限性**

局限于仅评估两种模型、只涵盖双辅音簇、低频簇样本稀少以及未考虑三辅音簇或形态学作用，结果可能受数据稀疏与模型架构的影响。

---

## 119. Self-Recognition Finetuning can Prevent and Reverse Emergent Misalignment

**arXiv ID:** 2606.23700 | [PDF](https://arxiv.org/pdf/2606.23700v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 120. Fabric Image Demoiréing Benchmark from Synthesis to Restoration

**arXiv ID:** 2606.24072 | [PDF](https://arxiv.org/pdf/2606.24072v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 121. Is Higher Team Gender Diversity Correlated with Better Scientific Impact?

**arXiv ID:** 2606.24098 | [PDF](https://arxiv.org/pdf/2606.24098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 122. Complex Autonomous UAV Task Execution and Decision-Making With s(CASP)

**arXiv ID:** 2606.23866 | [PDF](https://arxiv.org/pdf/2606.23866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 123. Semi-asynchronous Federated Learning in Flower: Framework Extension and Performance Assessment

**arXiv ID:** 2606.24230 | [PDF](https://arxiv.org/pdf/2606.24230v1)

**作者:** Víctor Hidalgo-Izquierdo `[一作]` (Universidad de Castilla-La Mancha), Blanca Caminero `[通讯]` (Universidad de Castilla-La Mancha)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Flower框架中实现了半异步联邦学习机制FedSaSync，允许在达到预设同步度M后立即聚合，从而在异构边缘环境中减少闲置时间并提升系统效率。

**💡 创新点**

主要创新在于：①在Flower中原生实现半异步聚合策略；②通过M阈值实现部分同步，兼顾收敛稳定性与效率；③提供开源实现和实验基准，推动SAFL研究。

**🔧 技术方法**

使用Python、Flower、PyTorch、flwr simulation、flwr-datasets等技术实现半异步消息传递与聚合。

**📊 数据集**

采用CIFAR-10和MNIST数据集，并通过IID划分分发给10个客户端。

**📈 对比分析**

与传统FedAvg同步基线以及不同M值进行对比，评估指标为测试损失随壁钟时间的变化和Δloss/秒的收敛效率；实验显示在存在慢速客户端时，M< N 的FedSaSync显著优于FedAvg，降低闲置时间并加速收敛。

**⚠️ 局限性**

限制包括：半异步度需预设且固定；实现紧耦合于Flower的Grid执行模型；后端通信优化为同步，半异步可能导致性能下降；缺乏动态适应客户端性能变化的机制。

---

## 124. Blockwise Policy-Drift Gating for On-Policy Distillation

**arXiv ID:** 2606.24084 | [PDF](https://arxiv.org/pdf/2606.24084v1)

**作者:** Liwen Zheng `[一作]` (Independent Researcher), Haiyun Jiang `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在对语言模型的后训练中，提出并实现了“策略漂移门” (policy‑drift gating)，利用学生自身在回放中产生的旧当前对数概率偏移作为无目标的损失权重，结合固定64-token块或换行分段的聚合方式，对 OPD（On‑policy Distillation）以及 Teacher‑TopK/LSM 的位置级损失进行再加权。

**💡 创新点**

核心创新是：①将同一轨迹下旧学生与当前学生的对数概率差作为“漂移”信号；②采用无梯度传递的软门控，并在局部区块内平均得到门值；③门控仅影响损失权重，不改教师目标或支持集合，保持训练框架最小改动；④在回放重用场景下显著提升了解题率。

**🔧 技术方法**

技术手段包括：OPD 与 Teacher‑TopK/LSM 的结合；计算旧当前对数概率差 d_{i,t}=logP_{beh}(y_{i,t})-logP_{cur}(y_{i,t})；使用 detaching 的门函数 g=exp(-τ|s|) 进行均值归一化；两种聚合粒度（固定64-token块与换行分段）；PPO‑style 训练、GRPO 教师模型、VLLM 评估、Pass@8 评测指标。

**📊 数据集**

训练集：DAPO‑Math‑17k（1.79M 数学问答）。评测集：MATH500、AIME 2024、MathArena AIME 2025、AMC23，共四个公开数学推理数据集。

**📈 对比分析**

实验在 200 步训练预算下比较了六个变体（Base OPD、OPD+Block64、OPD+NewlineSpan、LSM、LSM+Block64、LSM+NewlineSpan）。结果显示：OPD+Block64 将平均 Pass@8 从 49.8% 提升至 51.6%；LSM+Block64 达到 53.3% 的最高平均 Pass@8（相较基准 40.1%）。相比之下，Teacher‑Reference（4B GRPO）为 65.2%。换行分段门也有竞争力，但不如固定块在 OPD 版上稳定。

**⚠️ 局限性**

局限性：仅在单一 1.7B 学生 + 4B 教师上测试；未多次随机种子复现；缺乏 token‑/sequence‑级别门控对比；只尝试 64‑token 块，未探究不同块大小或更复杂的聚合统计；对数概率差的符号相消可能导致门值失真；未给出门值分布或梯度诊断。

---

## 125. MinInter: Minimizing Trajectory Interpolation During Data Augmentation for Imitation Learning

**arXiv ID:** 2606.24078 | [PDF](https://arxiv.org/pdf/2606.24078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 126. Flood Mapping from RGB imagery using a Vision Foundation Model

**arXiv ID:** 2606.24120 | [PDF](https://arxiv.org/pdf/2606.24120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 127. CORE-BREW: LLR-Based Soft Decoding for Robust Multi-Bit LLM Watermarking

**arXiv ID:** 2606.24163 | [PDF](https://arxiv.org/pdf/2606.24163v1)

**作者:** Joeun Kim `[一作]` (DGIST), Young-Sik Kim `[通讯]` (DGIST)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CORE‑BREW，一种基于块式多位水印的常量命中率嵌入与软判决解码框架，用于LLM输出的可追溯水印。

**💡 创新点**

通过常量命中率校准水印通道，获得闭式日志似然比，支持严格安全与FPR校准两种检测模式，并引入熵感知擦除保护。

**🔧 技术方法**

基于块式 BREW、BCH 码、软判决解码、窗口位移、常量命中率偏移、熵感知擦除、列表解码、LLR 计算等技术。

**📊 数据集**

在 OPT‑1.3B 与 Mistral‑7B 上使用 C4、OpenGen（WikiText‑103 衍生两句提示）数据集。

**📈 对比分析**

与 MPAC、Qu 等基线对比，CORE‑BREW‑Cal 在低 FPR 区域实现更高 TPR，保持文本质量；严格安全模式在 FPR 控制上与 BREW 相当；实验覆盖无攻击、词级攻击、改写攻击等多种场景。

**⚠️ 局限性**

需要模型可知性，计算开销略高；在强语义改写下仍有限；需调节 p^⋆、熵阈等参数以平衡鲁棒与质量。

---

## 128. An Introduction to Causal Reinforcement Learning

**arXiv ID:** 2606.24160 | [PDF](https://arxiv.org/pdf/2606.24160v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 129. Differential Unfolding: Efficient Unfolding Reconstruction for Video Snapshot Compressive Imaging

**arXiv ID:** 2606.24153 | [PDF](https://arxiv.org/pdf/2606.24153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 130. Emergent Relational Order in LLM Agent Societies: From Collective Affect to Authority Stratification

**arXiv ID:** 2606.23764 | [PDF](https://arxiv.org/pdf/2606.23764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 131. Beyond Bayer: Task-Optimal Sensor Co-Design for Robust Autonomous-Driving Segmentation

**arXiv ID:** 2606.24096 | [PDF](https://arxiv.org/pdf/2606.24096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 132. DivRL: Disentangled Self-Similarity Rewards for Diverse Subject-Driven Generation

**arXiv ID:** 2606.23950 | [PDF](https://arxiv.org/pdf/2606.23950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 133. VeryTrace: Verifying Reasoning Traces through Compilable Formalism and Structured Verification

**arXiv ID:** 2606.24124 | [PDF](https://arxiv.org/pdf/2606.24124v1)

**作者:** Ninghan Zhong `[一作]` (Georgia Institute of Technology), Sriram Vishwanath `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了将 LLM 生成的链式推理文本转化为可编译的 DSL，并通过混合机械检验与 LLM 审计的迭代修复循环，提升多步推理的可靠性。

**💡 创新点**

创新点包括：① 区域无关的 DSL 让推理步骤显式依赖并可执行；② 两阶段 DSL 转换防止上下文偏差；③ 混合机械检验与结构化 LLM 审计，实现一步级错误定位与自动修复；④ 该框架在不同领域实现零样本迁移。

**🔧 技术方法**

使用了 DSL 设计、状态转移模型、结构化验证管道（结构、约束、计算、推断、结论检查）、LLM 进行上下文提取、翻译与审核、迭代修复回路以及对话式 LLM。

**📊 数据集**

实验数据集包括 AIME 2025 竞赛数学题、LLM‑BabyBench 机器人规划任务和 CLUTRR 家谱推理任务。

**📈 对比分析**

与 Vanilla、Chain‑of‑Verification、Natural Program 等基线对比，VeryTrace 在三大领域均显著提升性能；在 AIME 2025 上提升 20% 以上，规划任务中对长推理也表现更优；整体准确率大幅提升，尤其对专用推理模型（如 Q3‑80b‑T）更具显著效益。

**⚠️ 局限性**

局限性包括：1）验证管道计算开销大，可能限制实时应用；2）LLM 审计可能产生误判，导致误报或漏报；3）对纯语义任务的收益有限；4）依赖高质量 LLM 进行上下文/翻译，需较大算力；5）多轮修复可能增加延迟。

---

## 134. End-to-End Radar and Communication Modulation Recognition with Neuromorphic Computing

**arXiv ID:** 2606.24075 | [PDF](https://arxiv.org/pdf/2606.24075v1)

**作者:** Xiaohu Li `[一作]` (China Nanhu Academy of Electronics and Information Technology), Wei Hua `[通讯]` (China Nanhu Academy of Electronics and Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种端到端的脉冲神经网络（SNN）框架，用于雷达和通信信号的自动调制识别（AMR），并在 KA200 神经形态芯片上实现了低功耗推理。

**💡 创新点**

创新点包括：① 自适应脉冲编码器替代传统静态编码器，减少信息丢失；② 引入整数泄漏积分-发火（ILIF）神经元提升表示能力；③ 设计了 Spike-Separable Convolution Neural Network（SSCNN）和 SpikeFormer（脉冲Transformer）模块，实现多尺度特征提取和长程依赖建模；④ 在保持参数量与现有基线相当的同时，能将能耗降低90%以上并在低信噪比环境下保持优势。

**🔧 技术方法**

核心技术包括：SNN、ILIF 神经元、SpikeEncoder、SSCNN、SpikeFormer、注意力机制、整数化与虚拟时间步映射以适配硬件。

**📊 数据集**

使用四个主流数据集：RML2016.10a、RML2016.10b、RML2018.01a（通信调制）以及 DeepRadar2022（雷达调制），分别涵盖 10–24 种调制类型。

**📈 对比分析**

与 AMC‑NET、LSTM、FEA‑T、AWN 等端到端基线比较，平均准确率在 62.5%–87.4% 之间，常规基线 58–85% 之间，显示显著提升；理论能耗相比基线降低 90%+；在 KA200 上实际测得功耗比 RTX 3090 和 Jetson Orin NX 低约 5 倍。

**⚠️ 局限性**

局限性包括：对极低 SNR 仍有一定性能下降；模型在硬件上仍需进一步验证真实环境中的鲁棒性；当前实现主要针对固定序列长度，扩展到更大多样化数据仍需研究。

---

## 135. The Professor: Multi-Teacher Unsupervised Prompt Distillation for Vision-Language Models

**arXiv ID:** 2606.23897 | [PDF](https://arxiv.org/pdf/2606.23897v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 136. Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation

**arXiv ID:** 2606.23743 | [PDF](https://arxiv.org/pdf/2606.23743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 137. Deciphering Fingerprints of 3D Molecular Surfaces for Accurate Epitope Prediction

**arXiv ID:** 2606.23830 | [PDF](https://arxiv.org/pdf/2606.23830v1)

**作者:** Fang Wu `[一作]` (Stanford University), Li Erran Li `[通讯]` (Amazon Aws)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对抗体-抗原识别，提出SurfBind面向分子表面学习框架，直接对表面进行特征提取并进行表面与结合伙伴的交互建模，实现了抗原表位预测。

**💡 创新点**

创新点在于将表面视为主导特征，采用分层Patch+Transformer+binder‑aware cross‑attention与VQ‑MAE自监督预训练，结合几何和化学属性目标，显著提升表位识别精度并在不同抗体、构象下保持稳健。

**🔧 技术方法**

技术包括表面点云采样、Morton排序Patch分割、SurfFormer++ Transformer、binder‑aware cross‑attention、VQ‑MAE自监督预训练、几何与化学预训练任务、粗细分辨率的多尺度预测。

**📊 数据集**

使用PDB‑REDO、SAbDab数据库（5,531抗原‑抗体复合物），对抗原表位标签基于距离阈值构造；在DB5.5和SAbDab‑test等测试集上评估。

**📈 对比分析**

与序列、结构、表面等多种基线比较，在AUC‑ROC、AUC‑PR、F1等指标上，SurfBind取得最高性能（AUC‑PR 0.305、F1 0.429），比AF‑Multimer、Pair‑EGRET、SEPPA‑mAb提升约70%+，在未见抗体和未见构象下亦保持优异。

**⚠️ 局限性**

局限性包括对高柔性蛋白或低质量结构的依赖、表面预训练需大规模结构数据、预测仍为概率性，需实验验证；未考虑配体与抗原的配体动态变化及非蛋白配体。

---

## 138. Are Safety Guarantees in Neural Networks Safe? How to Compute Trustworthy Robustness Certifications

**arXiv ID:** 2606.23858 | [PDF](https://arxiv.org/pdf/2606.23858v1)

**作者:** Merkouris Papamichail `[一作]` (Foundation for Research and Technology - Hellas), João Marques-Silva `[通讯]` (Catalan Institution for Research and Advanced Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种以“apothem”为优化目标的可信鲁棒性认证方法；

**💡 创新点**

创新点在于引入apothem度量，构建可证明的apopthem-最优小步骤与大步骤操作，以及双向认证（双重认证）提供上界；

**🔧 技术方法**

技术手段包括多维区间格结构、NN验证器（MILP/Marabou）驱动的 refine‑and‑check 算法、以及对ReLU网络的公式化与定理证明；

**📊 数据集**

实验数据集使用 MNIST 与 Fashion‑MNIST，构建两层 784‑32‑10‑10 的ReLU网络；

**📈 对比分析**

与现有软件比较，基于最小边长与直径的性能提升约两倍，平均边长与直径提升十倍，验证调用次数与CPU时间显著下降；

**⚠️ 局限性**

局限性包括依赖NN验证器的效率与精度、对非轴对齐决策边界的近似不足，以及对高维/大规模网络的可扩展性仍待验证。

---

## 139. Ensemble Feature Selection and Harris Hawks Optimization for Explainable Mental Health Risk Prediction in Female Sex Workers

**arXiv ID:** 2606.24047 | [PDF](https://arxiv.org/pdf/2606.24047v1)

**作者:** Ahnaf Atef Choudhury `[一作]` (George Mason University), Abdullah Al Mamun `[通讯]` (Dhaka University of Engineering and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个结合集成特征选择和Harris Hawks优化逻辑回归的混合模型，用于预测南非3005名女性工作者的抑郁风险。

**💡 创新点**

创新之处在于首次将ANOVA与互信息的集成特征选择与HHO优化的逻辑回归结合，并通过LIME实现可解释的结果。

**🔧 技术方法**

采用的技术包括ANOVA、互信息特征筛选、Harris Hawks Optimization、逻辑回归、LightGBM、随机森林、SVM、ANN以及LIME可解释性工具。

**📊 数据集**

使用了来自南非九省的3005名女性工作者的社区横断面调查数据（Mendeley公开数据集）。

**📈 对比分析**

与传统机器学习模型对比，HHO-优化的逻辑回归在特征选择后取得95.78%准确率、95.77% F1和0.96 AUC，明显优于基线模型。

**⚠️ 局限性**

局限性包括仅在南非数据上验证，缺乏跨国外部泛化与公平性评估，以及未对模型在真实场景中的可部署性进行实测。

---

## 140. Safe and Generalizable Hierarchical Multi-Agent RL via Constraint Manifold Control

**arXiv ID:** 2606.24010 | [PDF](https://arxiv.org/pdf/2606.24010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 141. Sat2City v2: Native 3D City Asset Generation from a Single Satellite Image

**arXiv ID:** 2606.24138 | [PDF](https://arxiv.org/pdf/2606.24138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. PixJail: Self-Evolving Paper-to-Pipeline Reproduction for Text-to-Image Jailbreak Evaluation

**arXiv ID:** 2606.24081 | [PDF](https://arxiv.org/pdf/2606.24081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 143. Is competitive online paging an artifact?

**arXiv ID:** 2606.23955 | [PDF](https://arxiv.org/pdf/2606.23955v1)

**作者:** Enoch Peserico `[一作]` (Università degli Studi di Padova), Michele Scquizzato `[通讯]` (Università degli Studi di Padova)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过重新定义分页模型（引入零初始访问成本的零入分页），证明在该模型下，即使采用随机化、资源增量或其他竞争分析改进手段，任何在线分页算法也无法与离线最优算法竞争，尤其在符合实际计算模式（如波束搜索、遗传算法）的请求序列上。

**💡 创新点**

创新点在于：①提出并系统化零入分页模型，纠正传统分页模型中对“冷缺失”假设的错误；②证明无论资源增量多大、是否随机化，在线算法在此模型下均不具备竞争比；③给出Full‑Cost分析下的紧界，确认保守算法（如LRU、FIFO）仍是最优；④指出这些结果对理想缓存模型和缓存无关/自适应算法的影响。

**🔧 技术方法**

使用的技术包括：竞争分析框架、Yao原理、随机化与资源增量的理论推导、构造对抗性请求序列（波束搜索/遗传算法模拟），以及Full‑Cost分析（对非缺失访问赋予ε成本）。

**📊 数据集**

所用数据集为理论构造的合成请求序列：基于波束搜索的多层节点请求和基于遗传算法的多代解生成序列，并将所有非输入页面设为零入页面；未使用真实实验数据。

**📈 对比分析**

比较方法：通过理论分析与构造对抗序列证明在线算法的期望成本超过任意给定常数加任意乘数的离线最优成本，展示竞争比不可能达到有限值；Full‑Cost分析给出具体竞争比公式1+O(1/(εh/k))，验证保守算法的最优性。性能表现：在线算法在这些序列上会产生大量缺失访问，远超离线最优。

**⚠️ 局限性**

局限性：①结果基于最坏情况竞争分析，可能不反映实际工作负载；②仅考虑零入分页模型，未涵盖所有现实情况（如动态页面产生、eviction成本）；③对真实系统的实验验证缺失；④对缓存无关/自适应算法的实际性能影响仍需进一步实证。

---

## 144. Flow-Corrected Thompson Sampling for Non-Stationary Contextual Bandits

**arXiv ID:** 2606.23933 | [PDF](https://arxiv.org/pdf/2606.23933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 145. Learning the Koopman Operator using Attention Free Transformers

**arXiv ID:** 2606.23957 | [PDF](https://arxiv.org/pdf/2606.23957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 146. Mind the Heads: Topological Representation Alignment for Multimodal LLMs

**arXiv ID:** 2606.23885 | [PDF](https://arxiv.org/pdf/2606.23885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 147. Maestro Order: A Model-Agnostic Orchestration Harness

**arXiv ID:** 2606.23983 | [PDF](https://arxiv.org/pdf/2606.23983v1)

**作者:** Hidayet Aksu `[一作]` `[通讯]`, Hidayet Aksu

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一个模型无关的调度框架Maestro Order，通过组合分解、集成、验证和递归四种结构化算子，并配备预算感知控制器，来将不可靠的求解器组织成可靠的问题求解系统。

**💡 创新点**

创新点在于将理论中的四个组合算子编译为可运行的组件，设计了基于边际对数赔率的贪婪预算分配策略，并在线估计验证器的判别率，实现了对计算预算与可靠性之间的最优权衡。

**🔧 技术方法**

使用的技术包括：统一求解器接口、黑板共享状态、可扩展验证器集合、对数赔率校准、在线判别率估计、基于阈值的放弃/升级、并行与缓存控制以及完整可回放的日志追踪。

**📊 数据集**

论文采用的是理论模拟实验：对参数化的生成器/验证器模型进行蒙特卡洛仿真，没有使用公开数据集，而是通过自定义的随机过程验证理论预言。

**📈 对比分析**

比较方法是将模拟结果与闭式的可靠性公式、投票与验证的成本-可靠性曲线以及多种消融配置进行对比，结果显示验证器显著低成本实现高可靠性，控制器能在不同策略间自适应选择，投票在多样性受限时效果受限。

**⚠️ 局限性**

限制在于实验仅为理论模拟，未在真实模型和真实数据集上验证；验证器的判别率假设独立且不随实例变化；控制器采用贪婪策略，可能不是全局最优；在实际部署中，硬件、网络延迟及模型特异性等因素仍需进一步评估。

---

## 148. Bengal-HP_RU: A Dataset of Bengal People For Head Pose Estimation

**arXiv ID:** 2606.24122 | [PDF](https://arxiv.org/pdf/2606.24122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. BipBipCache: Pipeline-Aware Integration of Low-Latency Tweakable Encryption in an Embedded Cache Controller

**arXiv ID:** 2606.23941 | [PDF](https://arxiv.org/pdf/2606.23941v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 150. Weight-Space Geometry of Offline Reasoning Training

**arXiv ID:** 2606.23740 | [PDF](https://arxiv.org/pdf/2606.23740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 151. Exploring Dualistic Meta-Learning to Enhance Domain Generalization in Open Set Scenarios

**arXiv ID:** 2606.23758 | [PDF](https://arxiv.org/pdf/2606.23758v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 152. Sentence-Level Contextual Entrainment in Large Language Models

**arXiv ID:** 2606.24077 | [PDF](https://arxiv.org/pdf/2606.24077v1)

**作者:** Yang Liu `[一作]` (Kyoto University), Chenhui Chu `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并扩展了大语言模型（LLM）中的上下文诱导（contextual entrainment）现象，从单词层面推广到句子层面，并系统评估其在客观与主观任务中的影响。

**💡 创新点**

创新点在于：①首次证明句子级别的上下文诱导存在且可由极少数注意力头控制；②发现模型规模增大时诱导减弱但注意力头稀疏度保持；③通过差分可微掩蔽仅关闭2%–4%的共享注意力头即可将诱导降半，而不显著损害模型性能。

**🔧 技术方法**

使用的技术包括：差分可微注意力头掩蔽（Gumbel‑softmax 方式）来定位诱导头；对数概率差值（ΔMean Log Probability）衡量诱导与分心；对模型输出准确率和观点一致性进行定量评估；对多尺寸多家族26款 LLM 进行横向对比实验。

**📊 数据集**

使用的数据集为：LRE（基于事实关系的问答数据）和 WVS（全球价值观调查的主观意见数据），两者共同构成客观与主观任务的对照。

**📈 对比分析**

比较方法：在不同模型规模、不同家族、不同注意力头掩蔽策略（未掩蔽、随机、每关系掩蔽、共享掩蔽）下，评估诱导效应、分心幅度、任务准确率与观点一致性；结果显示共享头掩蔽将诱导平均减少约一半，且在大部分模型上准确率下降不足3%，且在指令微调模型上效果更佳。

**⚠️ 局限性**

局限性包括：①实验仅覆盖解码器式 LLM，未验证自回归或 encoder‑decoder 模型；②掩蔽训练成本高，且需针对每个关系单独训练；③仅使用了两类任务，缺乏更广泛的多模态或更长文本场景；④对极端长上下文或多轮对话的诱导行为尚未充分探索。

---

## 153. ChartWalker: Benchmarking the Cross-Chart RAG Task

**arXiv ID:** 2606.23997 | [PDF](https://arxiv.org/pdf/2606.23997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 154. T2D-Bench: Evidence-Gated Evaluation of LLM Outputs for Type 2 Diabetes Using a Multi-Layer Clinical-Lifestyle Knowledge Graph

**arXiv ID:** 2606.24145 | [PDF](https://arxiv.org/pdf/2606.24145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 155. Reconstructing GRACE Terrestrial Water Storage with Spatio-Temporal Graph Neural Networks: An Application to South America

**arXiv ID:** 2606.23833 | [PDF](https://arxiv.org/pdf/2606.23833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 156. Quantifying Prior Dominance in RAG Systems

**arXiv ID:** 2606.23695 | [PDF](https://arxiv.org/pdf/2606.23695v1)

**作者:** Barak Or `[一作]` `[通讯]` (ArtificialGate), Barak Or (ArtificialGate)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于连续概率的Normalized Context Utilization (NCU)指标，用以量化检索增强生成（RAG）模型在严格事实提取任务中的上下文利用能力；

**💡 创新点**

创新点在于：①用长度归一化的token log‑probability构建连续评估，克服传统EM/F1的“epistemic blindness”；②通过NCU明确区分模型的参数记忆与主动上下文抽取；③揭示大模型在对抗性冲突下的Prior Dominance与Negative Transfer现象，证明小模型在提取任务上具有极致效率与更高的上下文遵从性；

**🔧 技术方法**

使用了token级log‑probability提取、信息理论框架（Shannon entropy）计算、对抗性干扰生成（NER + synthetic noise）、以及多尺度模型（1.5B-72B、商业API）的推理；

**📊 数据集**

采用了三大问答数据集：Natural Questions (NQ‑Open)、TriviaQA（Wikipedia分割）、HotpotQA（distractor分割），全部以英文单词级token化；

**📈 对比分析**

比较方法：对每个模型在Zero‑Shot、Oracle、Noise、Conflict四种条件下记录准确率、NCU（限定与原始）及推理时延；结果显示：SLM（1.5B/7B）在Oracle条件下的NCU接近大模型（≈0.86），且推理延迟显著更低；大模型在冲突条件下表现出Prior Dominance与Negative Transfer；

**⚠️ 局限性**

局限性包括：①只评估了5-token的严格提取，未涉及多跳推理或CoT；②对商业API的内部机制缺乏透明性，难以精确定位Prior Dominance来源；③对抗样本为人工合成，真实世界的伪造信息可能表现不同；

---

## 157. REALM: A Unified Red-Teaming Benchmark for Physical-World VLMs

**arXiv ID:** 2606.23892 | [PDF](https://arxiv.org/pdf/2606.23892v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 158. From Task-Guided Conversational Graphs to Goal-Oriented Dialogue Runtimes

**arXiv ID:** 2606.23797 | [PDF](https://arxiv.org/pdf/2606.23797v1)

**作者:** Mariano Garralda-Barrio `[一作]` `[通讯]`, Mariano Garralda-Barrio

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种 Goal-Oriented Dialogue Runtime（GODR）框架，用以显式管理多目标对话中的目标生命周期、暂停恢复与相互依赖；

**💡 创新点**

在现有图与多代理编排框架之上引入独立的目标管理层，提供目标状态、重置、恢复合约及跨目标无效化等运行时对象；

**🔧 技术方法**

采用图运行时（如 LangGraph、ADK、CrewAI 等）与多代理、工具调用等技术组合，形成三层架构：目标层、对话层与执行层；

**📊 数据集**

本文并未使用实际数据集，而是以示例场景（如事件注册、保修申请等）构建概念验证与评估协议；

**📈 对比分析**

作者提出的评估方案将GODR与传统 FSM/流程图、根图+子图以及监督代理等模型对比，但并未给出实验结果；

**⚠️ 局限性**

主要限制在于缺乏实证验证与基准，可能对简单流程过度设计，且目标管理与执行层需避免状态重复与冲突。

---

## 159. Fast and Slow Variational Continual Learning

**arXiv ID:** 2606.24007 | [PDF](https://arxiv.org/pdf/2606.24007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 160. Systematic Exploration of 4-Expert Heterogeneous Mixture-of-Experts via Automated Pipeline Search

**arXiv ID:** 2606.23739 | [PDF](https://arxiv.org/pdf/2606.23739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 161. You Don't Need to Run Every Eval

**arXiv ID:** 2606.24020 | [PDF](https://arxiv.org/pdf/2606.24020v1)

**作者:** Yuchen Zeng `[一作]`, Dimitris Papailiopoulos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文首先构建了一个包含84款模型与133项基准的公共得分矩阵，并发现该矩阵近似为秩为2的低维结构；基于此设计了BenchPress——一种使用logit变换后偏置分解ALS矩阵完成的方法，用来填补缺失的模型‑基准得分。

**💡 创新点**

创新点在于①将大型LLM评测空间归约为秩2的低维表示；②提出BenchPress这一高效、全覆盖的得分预测器；③通过探测子集选择和可靠性估计，展示如何用极少数基准快速恢复完整得分卡，并判断预测可信度。

**🔧 技术方法**

核心技术包括：logit变换、偏置分解的交替最小二乘ALS矩阵完成、基准与模型的相似性分析、梯度与随机搜索调参、以及可靠性评估的混合估计器。

**📊 数据集**

使用的数据集是公开的LLM评测结果集合：84个模型（覆盖13大厂商）和133个基准（涵盖推理、编码、数学、代理等多类），共计2,604条已观察得分（占矩阵的23.3%）。

**📈 对比分析**

与多种基准预测方法（KNN、回归、Soft‑Impute、NMF等）对比，BenchPress在全覆盖的情况下实现了平均绝对误差（MedAE）约4.6分、平均绝对百分误差（MedAPE）约4.6%；在仅用5个关键基准进行探测时，整体MedAE可降至3.9分；对同一基准下的模型排序保持率在5分阈值下超过92%。

**⚠️ 局限性**

主要局限包括：①得分矩阵来源多样，存在测量噪声与供应商偏差；②仅聚焦于聚合得分，未能捕捉实例级细节；③对秩2结构的依赖可能随未来模型/基准演进而失效；④对缺乏近邻的模型预测效果差，需引入外部模型特征或更细粒度评测。

---

## 162. NeuroSonic: Conditional Flow Matching for EEG-to-Speech Reconstruction

**arXiv ID:** 2606.24087 | [PDF](https://arxiv.org/pdf/2606.24087v1)

**作者:** Wenhao Gao `[一作]` (Stony Brook University), Chenyu You `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于条件流匹配的NeuroSonic框架，能够将脑电（EEG）信号转换为连续语音。

**💡 创新点**

创新点在于将EEG到语音的重构视为确定性轨迹传输，学习EEG条件下的概率流速度场，避免了传统随机采样链导致的噪声敏感和跨受试者差异。

**🔧 技术方法**

采用了条件流匹配、门控Transformer、RMS归一化注意力、时间条件嵌入以及Heun积分求解ODE等技术，并将EEG与音频映射到共享的token空间。

**📊 数据集**

使用了CineBrain和EAV这两个公开EEG‑语音数据集进行实验。

**📈 对比分析**

与GAN、扩散模型和均值流基线在交叉受试者评估下比较，NeuroSonic在FAD、LSD、SC、DNSMOS等多项指标上均显著优于基线，尤其在噪声重的片段上提升显著。

**⚠️ 局限性**

局限性包括仅在两类数据集上验证，缺乏对实时长时段语音生成的评估，且对EEG信号空间分辨率和语言多样性的适应性仍有限。

---

## 163. Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization

**arXiv ID:** 2606.23989 | [PDF](https://arxiv.org/pdf/2606.23989v1)

**作者:** Shuo Guan `[一作]` `[通讯]` (UBS AG), Shuo Guan (UBS AG)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于“主张（Claim）”的多文档摘要管线，先用LLM提取带源文档指纹的原子主张，随后跨文档聚类并检测冲突，挑选支持度高且显著的主张，利用可重写器生成包含精细多源引用的摘要句子，最后通过双向NLI验证保证生成句子与选定主张一致。

**💡 创新点**

① 将原子主张作为既是可验证单元又是归因锚点；② 跨文档聚类与冲突检测实现多源、句级引用；③ 通过阈值控制的选择器提供可调节的真实性-覆盖率权衡；④ 引入独立评估器消除评估循环，验证系统的稳健性。

**🔧 技术方法**

LLM主张提取与去上下文化；quote‑to‑span精准匹配；句子双编码器+余弦聚类并用三类NLI过滤合并；LightGBM GBDT做自支持与显著性评分；MMR式贪婪选择；双向NLI验证与修复；生成时内嵌引用；独立SummaC/TRUE评估。

**📊 数据集**

MultiNews（大规模多文档摘要基准）、DiverseSumm（冲突与多样性数据集）以及WCEP（事件级摘要零样本迁移测试）。

**📈 对比分析**

在MultiNews上，摘要质量（ROUGE‑L≈25.0、BERTScore≈0.91）与最强的细调模型相当；但在真实性指标（AlignScore≈87.6、SummaC≈0.91、FActScore≈0.90）上显著优于基线；引用精度在独立评估下达到84%，并将多源引用召回率提升至64%（相较于span‑first方法的38%）。在DiverseSumm上，冲突覆盖率和准确率分别达到61%与73%；在WCEP零样本迁移时保持竞争力，真实性仍优于基线。

**⚠️ 局限性**

依赖LLM对主张的准确拆分，去上下文化误差可能导致事实错误；聚类阈值与冲突检测对结果敏感；归因评价仍受限于软匹配与自动评估器；提取与匹配步骤增加计算成本；冲突处理策略需根据应用场景手动调节；实验仅覆盖英文新闻域，跨语言或领域的泛化尚未验证。

---

## 164. Topological Online Learning for Displacement-based Formation Control

**arXiv ID:** 2606.23901 | [PDF](https://arxiv.org/pdf/2606.23901v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 165. Can Language Model Agents be Helpful Circuit Explainers in Mechanistic Interpretability?

**arXiv ID:** 2606.24026 | [PDF](https://arxiv.org/pdf/2606.24026v1)

**作者:** Ayan Antik Khan `[一作]` (George Mason University), Ziyu Yao `[通讯]` (George Mason University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于语言模型代理的电路解释框架，并构建了一个从InterpBench衍生的电路解释基准。

**💡 创新点**

创新点包括①利用半合成Transformer生成的电路构建带有组件标签与自然语言描述的基准；②提出Hypothesize‑Validate‑Explain循环代理，系统化地执行观察、假设生成与验证；③在四大前沿LLM上进行统一评估，揭示不同模型在验证计划、代码执行和解释质量上的差异。

**🔧 技术方法**

使用的技术包括大型语言模型代理（GPT‑5.4、Claude‑Sonnet‑4.6、Gemini‑3.1‑Pro、Qwen‑3‑Coder）、LangGraph状态机、TransformerLens、自动化代码执行与验证工具，以及自定义的观察/验证助手库。

**📊 数据集**

使用的数据集为84个半合成Transformer电路（共163个组件），源自InterpBench；并在Llama‑3‑8B的三元加法电路上做了案例研究。

**📈 对比分析**

评价方法：组件标签准确率、自然语言描述质量（LLM/人类评判）、任务描述准确率、代码执行成功率等；实验结果显示GPT‑5.4组件标签准确率达79%、任务准确率83%，不同模型在验证计划、代码执行、描述质量上各有优势，未出现统一最优模型。

**⚠️ 局限性**

局限性：可靠的验证执行仍是主要瓶颈；基准使用半合成电路，结构偏向算法化，可能与自然训练模型中的机制差异大；模型可能记忆已知电路，导致真实迁移性受限。

---

## 166. Aligning MusicLLM with Emotion using Instruction Tuning and Feedback-Driven Alignment

**arXiv ID:** 2606.24123 | [PDF](https://arxiv.org/pdf/2606.24123v1)

**作者:** Takuya Hasumi `[一作]` (LY Corporation), Welly Naptali `[通讯]` (LY Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

训练音乐大型语言模型（MusicLLM）进行情感回归，并比较指令调优与基于反馈的对齐两种策略。

**💡 创新点**

提出了利用可验证数值反馈（GRPO）进行反馈驱动对齐，显著提升MusicLLM的情感预测准确性，同时保持其音乐问答能力。

**🔧 技术方法**

使用SLAM-LLM架构，音乐编码器MusicFM、投影层、Vicuna 7B文本解码器，低秩适配（LoRA），以及指令调优与GRPO反馈驱动对齐技术。

**📊 数据集**

采用公开情感回归数据集DEAM和MERGE进行训练与评估，并使用MusicQA评估模型的通用音乐问答性能。

**📈 对比分析**

通过R²指标评估情感回归性能：指令调优提升有限，而反馈驱动对齐显著提高arousal/valence的R²，同时在MusicQA上的BLEU/METEOR/ROUGE_L保持不变。

**⚠️ 局限性**

研究仅在DEAM/MERGE和7B Vicuna配置下验证，开放模型仍未表现良好，且情感回归受限于主观标签，未对更大模型或更广泛任务进行充分验证。

---

## 167. AsyncOPD: How Stale Can On-Policy Distillation Be?

**arXiv ID:** 2606.24143 | [PDF](https://arxiv.org/pdf/2606.24143v1)

**作者:** Wonjun Kang `[一作]` (Furiosaai), Kangwook Lee `[通讯]` (Krafton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

未知

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

未知

---

## 168. Dual-Branch Cross-Projection Debiasing through Diffusion-based Disentanglement

**arXiv ID:** 2606.24161 | [PDF](https://arxiv.org/pdf/2606.24161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 169. Beyond the Autoregressive Horizon: A Comprehensive Survey of Diffusion Models, World Modelling, and State Space Models for Code

**arXiv ID:** 2606.23690 | [PDF](https://arxiv.org/pdf/2606.23690v1)

**作者:** Kishan Maharaj `[一作]` (IBM), Srikanth Tamilselvam `[通讯]` (IBM)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了非自回归（Diffusion、State‑Space、Code World Models）在代码生成与软件工程中的最新进展，并从认知神经科学角度提出了未来Hybrid架构与System 2思考的方向。

**💡 创新点**

创新点在于系统梳理三大非AR范式的技术细节与优势，强调认知一致性，提出融合扩散的全局规划、状态空间的线性长依赖以及世界模型的执行语义模拟，形成面向“计划、迭代、执行感知”的下一代代码智能框架。

**🔧 技术方法**

主要技术包括扩散式去噪生成、线性动力学的State‑Space模型、基于执行轨迹的Code World Models，以及多模态的Hybrid Diffusion、Block Diffusion和RLVR等后训练策略。

**📊 数据集**

综述引用的典型数据集有 HumanEval、APPS、CoNaLa、MBPP、BigCodeBench、SWE‑bench、Terminal‑Bench、CruxEval、NVD、SARD 等，用以评估生成、修复、测试与仓库级任务。

**📈 对比分析**

比较方法主要是基于各自任务的标准指标（如 pass@k、Execution Match、Coverage、MRR、F1 等），综述指出AR模型在短代码生成上仍占优势；扩散模型在结构化生成与多任务方面表现更好；SSM 在代码理解与检索任务上具有良好样本效率；CWMs 在仓库级工作流与执行检验上具有较高的 Resolve Rate，但缺乏统一基准导致难以直接比较。

**⚠️ 局限性**

局限性包括：缺乏统一的跨范式评测框架，数据集与指标分散；SSM 与CWMs 的生成能力仍处探索阶段；综述本身多为文献归纳，缺少新实验验证；未能系统评估混合架构的实际性能与可扩展性。

---

## 170. Synergizing Physically Constrained MCMC and Chemical-Informed Gaussian Processes for Reaction Network Discovery

**arXiv ID:** 2606.23757 | [PDF](https://arxiv.org/pdf/2606.23757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 171. Cryptographic certificates of validity for trustworthy AI

**arXiv ID:** 2606.23768 | [PDF](https://arxiv.org/pdf/2606.23768v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 172. Neuromorphic Speech Enhancement with Dual-Branch Spiking Neural Networks

**arXiv ID:** 2606.23761 | [PDF](https://arxiv.org/pdf/2606.23761v1)

**作者:** Taiyu Meng `[一作]` (Hangzhou Dianzi University), Haibing Yin `[通讯]` (Hangzhou Dianzi University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于双分支双路径的GSU-DBNet，用双分支幅度‑复数谱掩模联合估计实现高效语音增强；

**💡 创新点**

创新点包括：①双分支联合幅度与复数谱建模；②双路径Gated Spiking Unit（GSU）取代传统LSTM；③通过实验证明单门GSU因二值输出瓶颈最优；

**🔧 技术方法**

采用了双分支编码‑分离‑解码架构、双路径GSU、CBAM注意力、DeepFilter、surrogate梯度训练以及混合MSE+SI‑SNR损失；

**📊 数据集**

使用了VoiceBank+DEMAND数据集进行训练与评估；

**📈 对比分析**

与多种ANN模型（DCCRN、FullSubNet+、GaGNet、TSTNN）及SNN基线（DPSNN、Spiking‑FSN）对比，GSU‑DBNet仅394K参数，PESQ 3.04，超过ANN模型参数的4.5%–10.6%，并在CBAK、COVL、SSNR等指标上取得最优或接近最优；

**⚠️ 局限性**

主要局限为：二值输出瓶颈限制了多门门控结构的优势；尚未在更大或更多样化的数据集上验证；未在真实 neuromorphic 硬件上实现部署与性能评估。

---

## 173. VisChronos: Revolutionizing Image Captioning Through Real-Life Events

**arXiv ID:** 2606.24058 | [PDF](https://arxiv.org/pdf/2606.24058v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 174. A Topological Framework for Finite Behavioural Observations and Verification

**arXiv ID:** 2606.23975 | [PDF](https://arxiv.org/pdf/2606.23975v1)

**作者:** Antonis Achilleos `[一作]` (Reykjavik University), Vasiliki Kyriakou `[通讯]` (Reykjavik University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过构造基于有限观察的拓扑空间，研究了有限信息下可验证的系统属性。

**💡 创新点**

创新点在于将监视器可观测性与拓扑开放集对应，首次将多轨迹监视与模拟观察拓扑相联系，并揭示有限深度 bisimulation 与传统模拟产生不同拓扑。

**🔧 技术方法**

主要使用了标签化转移系统、拓扑学（Alexandroff拓扑、Cantor拓扑）、观察前缀、模拟和有限深度 bisimulation 以及 Hennessy–Milner 逻辑。

**📊 数据集**

未使用实验数据集，整个工作基于形式化模型与理论证明。

**📈 对比分析**

通过理论证明比较不同观察方式产生的拓扑，并给出开集与可验证属性的等价性；性能方面以理论可行性和表达力为准，未给出数值指标。

**⚠️ 局限性**

局限在于仅考虑无循环的有限观察，且对更强行为关系（如完备模拟、准备模拟）无法形成基底，未来需扩展到循环观察或更一般的观察。

---

## 175. Metis: Bridging Text and Code Memory for Self-Evolving Agents

**arXiv ID:** 2606.24151 | [PDF](https://arxiv.org/pdf/2606.24151v1)

**作者:** Zijie Dai `[一作]` (Chinese University of Hong Kong), Xiao Yan `[通讯]` (Wuhan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种自我进化智能体系统Metis，能够在任务执行过程中同时维护文本记忆和代码记忆，并根据经验的出现频率将可复用的执行计划动态转化为可调用工具。

**💡 创新点**

创新点在于：①对文本记忆和代码记忆进行系统对比与量化评估，发现两者在构建成本、执行效率与迁移可靠性上互补；②设计了分层双重表示的记忆结构，将文本记忆细分为计划、环境事实与常见错误；③引入基于出现频率的递归门控推广机制，只有当计划被多次检索时才生成工具，避免不必要的代码生成。

**🔧 技术方法**

技术核心包括：使用大型语言模型（Claude Sonnet 4.6）进行文本反射和工具生成；基于嵌入的向量检索与LLM辅助的联合记忆选择；工具生成后在沙盒环境中进行依赖闭包、编译与验证；以及逐步暴露轨迹的反射主机。

**📊 数据集**

主要使用的基准数据集是AppWorld，包含9个应用的457个API，提供约数千个自然语言指令任务，分为官方训练/测试和重采样两种拆分方式。

**📈 对比分析**

与基线（No Memory、ACE、SkillX）相比，Metis在官方分拆上任务成功率提升8.3个百分点、执行token下降13.5%、ReAct轮次下降3.4%，在重采样分拆上更进一步提升任务成功率至66.1%（比SkillX高0.6个百分点），且反射成本低于SkillX但高于ACE，表明在准确率、效率与构建成本之间取得了最佳平衡。

**⚠️ 局限性**

局限性包括：①对非递归、单任务生成的工具适配不佳，导致工具的利用率在小样本上不高；②实验仅评估了AppWorld环境，缺乏对更广泛、多模态或现实世界任务的验证；③工具生成仍依赖大量token，虽然被门控控制，但在更大规模任务序列中可能导致构建成本上升。

---

## 176. RASC+: Retrieval-Constrained LLM Adjudication for Clinical Value Set Authoring

**arXiv ID:** 2606.23992 | [PDF](https://arxiv.org/pdf/2606.23992v1)

**作者:** Sumit Mukherjee `[一作]` `[通讯]` (Oracle Health), Sumit Mukherjee (Oracle Health)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种分阶段检索+裁定的临床价值集完成方法。

**💡 创新点**

通过优化检索召回与LLM裁定，显著提升价值集覆盖率和精准度，同时保持可审计性。

**🔧 技术方法**

使用Qwen3嵌入检索、词汇扩展、代码显示检索生成候选池，再用GPT‑5进行受限裁定；与SAPBert cross‑encoder对比。

**📊 数据集**

使用VSAC检索集RASC基准，共3,744个价值集，包含保留出版商（OOD）。

**📈 对比分析**

在扩充后的候选池上，GPT‑5宏F1从0.287提升至0.549，OOD宏F1从0.233提升至0.533；原始RASC为0.298。

**⚠️ 局限性**

局限在于仅基准评估，未包含临床审计；GPT‑5调用成本高，候选池分块限制模型整体比较；未进行提示消融或统计置信区间。

---

## 177. Do LLM Attribution Metrics Transfer? Auditing Retrieval-Augmented Generation Evaluation Across Datasets and Constructs

**arXiv ID:** 2606.23915 | [PDF](https://arxiv.org/pdf/2606.23915v1)

**作者:** Tianyu Ding `[一作]` (Amazon Web Services), Juan Pablo De la Cruz Weinstein `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对8种自动归因评估指标（词汇重叠、句子嵌入、NLI/事实检查器、BERTScore 等）在三种评估构造下进行跨数据集审计，评估其在不同数据集上的可迁移性和稳定性。

**💡 创新点**

发现即使在同一归因任务内，最优指标也会在不同数据集间出现反转；构造决定指标族，但在归因任务中没有单一指标可在所有数据集上保持近最优，揭示了自动评估的可靠性危机。

**🔧 技术方法**

使用基于词汇 Jaccard、MiniLM/MpNet 余弦相似度、BERTScore、NLI 预测头（Clean-MNLI、FEVER NLI）、专用检查器 MiniCheck，以及基于 LLM 的提示式评审者。

**📊 数据集**

数据集包括 AttributionBench（四个来源）、HAGRID、VitaminC、ASQA、MS MARCO、HotpotQA，覆盖句子/答案级归因与段落级检索相关性。

**📈 对比分析**

比较方法：使用 AUROC、Top‑1 准确率、Kendall τ、留一数据集退后误差（regret）等统计量。结果显示指标排名不稳定，最高 AUROC 仅在某些数据集上领先，留一退后误差平均为 0.044 AUROC，说明无单一通用评估器。

**⚠️ 局限性**

局限性：仅涵盖英文数据集；归因与检索构造与句子/答案级标签分离；未评估跨域迁移和系统级效果；LLM 判定器成本高、非确定性，未真正解决评估负担。

---

## 178. Rapid FinFET Modelling Using an Autoencoder

**arXiv ID:** 2606.24046 | [PDF](https://arxiv.org/pdf/2606.24046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 179. An LMM for Precisely Grounding Elements in Documents

**arXiv ID:** 2606.24118 | [PDF](https://arxiv.org/pdf/2606.24118v1)

**作者:** Yijian Lu `[一作]` (Tsinghua University), Ji Qi `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了一种专门用于文档精确定位和视觉对齐推理的大型多模态模型 PreciseDoc，并提出完整的从文档生成 → 冷启动 SFT → 强化学习的训练流程；

**💡 创新点**

创新点包括：① 通过在 LaTeX 编译时嵌入坐标命令自动生成高质量 PDF 并精确标注文本位置；② 通过手写字符合成并加入相机效应构建手写文档数据；③ 在 RL 阶段同时监督答案准确率与 IoU 目标，使用 Hungarian 算法进行一一匹配并引入长度惩罚以防止冗余框；④ 让 LMM 在推理时能够生成多箱定位信息并与最终答案关联，提升可解释性；

**🔧 技术方法**

采用的技术包括：多模态 LMM（基于 Qwen/Vision‑LLaVA 之类的架构）、自定义 LaTeX 计数命令提取坐标、MinerU OCR 生成元素级标注、GLM‑4.6V 用于生成推理过程、强化学习框架、IoU+Hungarian 匹配、长度惩罚、数据扩增和多任务微调；

**📊 数据集**

使用的数据集有：DocLocal4K（文本定位基准）、TRIG（词级证据定位）、BBox‑DocVQA（元素级定位和答案）、Bbox‑DocVQA 与 DOGR 的训练集用于 SFT；此外自建的 LaTeX 生成文档集、合成手写文档集；传统评测还用到 DocVQA、ChartQA、TextVQA；

**📈 对比分析**

与多种私有和开源 LMM（Gemini、GPT‑4o、GPT‑5、Qwen3、InternVL 等）在 DocLocal4K、TRIG、BBox‑DocVQA 上做对比。PreciseDoc 在 DocLocal4K 的行/段落定位上接近或略优于 DocOwl‑1.5，整体定位准确率提升显著；PreciseDoc‑Reasoner 在 TRIG 上达到 67.5/46.4 的 IoU，明显优于同类模型；在 BBox‑DocVQA 上 Grounding Accuracy 与答案准确率分别达到 78%+，与 Gemini‑3‑Flash 接近，证明在视觉对齐推理任务中表现领先；

**⚠️ 局限性**

局限性包括：对表格和图像的推理仍易出错；在词/短语级别的定位精度仍不如段落级；RL 训练成本高，难以扩展到更大模型；长度惩罚可能导致模型过度简化推理文本；合成手写文档的多样性和真实性仍有限。

---

## 180. Hash Table Design for RDMA:Challenges and Opportunities

**arXiv ID:** 2606.24073 | [PDF](https://arxiv.org/pdf/2606.24073v1)

**作者:** Shuchen She `[一作]` (Nanjing University), Haipeng Dai `[通讯]` (Nanjing University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了在RDMA远程内存上实现哈希表的设计挑战与现有方案

**💡 创新点**

系统梳理了远程哈希表的五大核心难点，并给出未来研究方向

**🔧 技术方法**

回顾了RACE、SepHash等基于RDMA的一键侧哈希表设计，讨论了指针链、分段结构、过滤器等技术

**📊 数据集**

无具体实验数据集，主要以已有研究的设计和评估为依据

**📈 对比分析**

未提供新的实验比较，文中对已有方案的性能特点做了概括性评价

**⚠️ 局限性**

作为综述缺乏实际实现与实验验证，未解决如何在实际系统中实现最优调度与一致性保证的细节

---

## 181. ESAA-Conversational: An Event-Sourced Memory Layer for Continuity, Handoff, and Curation Across Heterogeneous LLM Coding Agents

**arXiv ID:** 2606.23752 | [PDF](https://arxiv.org/pdf/2606.23752v1)

**作者:** Elzo Brito dos Santos Filho `[一作]` `[通讯]`, Elzo Brito dos Santos Filho

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并发布了 ESAA-Conversational，利用事件源架构为多种 LLM 编码助手提供共享的对话记忆，并实现了公开的 PowerShell CLI。

**💡 创新点**

创新点包括：反向摄取机制（从各自日志生成事件）、机械捕获与手工策划分离、事件源化的对话状态、分页上下文读取、绿色发布与隐私保护策略。

**🔧 技术方法**

技术手段包括事件源 + CQRS、PowerShell CLI、锁文件、读模型投影、反向适配器以及对话窗口分页查询。

**📊 数据集**

使用的数据集为实验室内部 570 条对话日志（Codex 304 条、Claude 79 条、Grok 67 条等）。

**📈 对比分析**

评估方式为自我参考案例（实验室内部多代理协作），未给出量化基准；强调通过机械捕获降低 token 成本、保证读模型可验证性。

**⚠️ 局限性**

局限性：仅在 Windows/PowerShell 本地实现；缺乏语义索引、法医审计、快照压缩；并发控制仅为锁文件；未支持跨平台或大规模团队使用。

---

## 182. VeriPilot: An LLM-Powered Verilog Debugging Framework

**arXiv ID:** 2606.23759 | [PDF](https://arxiv.org/pdf/2606.23759v1)

**作者:** Yihan Wang `[一作]` (University of Chinese Academy of Sciences), Huawei Li `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VeriPilot，一个将LLM与控制数据流图、金丝雀参考模型相结合的Verilog调试框架，能够细粒度定位错误并自动修复RTL代码。

**💡 创新点**

核心创新在于通过LLM驱动的跨语言语义映射、前沿信号提取与背向追踪生成结构化调试证据，显著提升LLM调试精度和修复成功率。

**🔧 技术方法**

使用了大型语言模型（如GPT-4o/GPT-5）、静态分析工具构建CDFG、跨语言变量对齐算法、基于前沿信号的背向追踪、以及多代理协同架构。

**📊 数据集**

实验使用了NVIDIA的CVDP（cid16子集）和Strider两个基准数据集。

**📈 对比分析**

通过与CirFix、Strider、MAGE等传统与LLM驱动的调试方法比较，VeriPilot在Strider上的修复成功率从54.3%提升至91.23%（GPT-4o）/92.98%（GPT-5），在CVDP上提升至85.71%（GPT-4o）/91.43%（GPT-5），显著优于传统方法和单纯LLM。

**⚠️ 局限性**

主要局限包括金丝雀模型内部状态映射不足导致语义对齐失败；修复仍受LLM推理能力限制，尤其对需要大规模结构变更的bug；同时增加了token消耗和调试流程的复杂度。

---

## 183. PCB-QA: Evaluating LLMs over the First Printed Circuit Board Design Question-Answer Dataset

**arXiv ID:** 2606.23704 | [PDF](https://arxiv.org/pdf/2606.23704v1)

**作者:** Sahana Srinivasan `[一作]` (UNSW Sydney), Hammond Pearce `[通讯]` (UNSW Sydney)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含480条问答对的 PCB-QA 数据集，并使用该数据集评估 LLM 在 PCB 设计与分析任务中的表现。

**💡 创新点**

首创了基于文本的 PCB 问答数据集；提出了 JSON 形式的 KiCad 导出与 SPICE 仿真结果的结构化表示，以便 LLM 直接解析；验证了此格式显著提升 LLM 的准确率。

**🔧 技术方法**

利用 LLM（Gemini 3 Flash Preview、Claude Sonnet 4.6、Llama 3.3 70B、GPT‑5.4 Nano）进行推理；采用工具调用与语义检索（使用 SentenceTransformer 生成 datasheet 嵌入）实现对不同文件格式的解析；评估指标包括准确率、精确率、召回率与 F1‑score。

**📊 数据集**

使用 8 个公开硬件项目的 KiCad 设计文件（包括原理图、原理网表、SPICE 仿真文件）生成问答对，随后转化为三种文件格式（PDF、原始 KiCad 导出、JSON）供实验使用。

**📈 对比分析**

通过对比不同文件格式（PDF、KiCad 原生、JSON）下 LLM 的表现，发现 JSON 形式在所有模型中获得最高准确率；Gemini 3 Flash Preview 在 JSON 格式下取得 93% 的准确率，远超其它模型，证明文本化格式对 LLM 友好。

**⚠️ 局限性**

限制包括：数据集仍偏斜（YES/NO 分布不均）；对大尺寸项目（如 AcornRobotElectronics）时输入 token 超限导致部分模型拒绝；仅评估了三类问题，未覆盖 PCB 设计完整生命周期；对 SPICE 仿真仅做了简化处理，无法完全覆盖复杂 IC 行为。

---

## 184. Unified Multi-Task Relevance Modeling for E-Commerce: Comparing Task Routing Architectures Across LLMs and Cross-Encoders

**arXiv ID:** 2606.23919 | [PDF](https://arxiv.org/pdf/2606.23919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 185. Collaborative and AI-Supported Requirements Elicitation: An Empirical Study

**arXiv ID:** 2606.24060 | [PDF](https://arxiv.org/pdf/2606.24060v1)

**作者:** Manoel Salgado Neto `[一作]` (CESAR School), Ronnie de Souza Santos `[通讯]` (University of Calgary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究 AI 支持的协作平台在需求获取中的效果，比较四种获取方式的产出质量与体验

**💡 创新点**

证明仅靠 AI 或纯手工获取无法与 AI+协作匹配的高质量产出，AI 主要辅助合成而非取代人类

**🔧 技术方法**

使用 Strateegia 协作平台 + GPT‑4.1 Writer 插件；对照 LLM 直接生成与转录后生成

**📊 数据集**

采用单一虚构的会议室预订场景作为需求背景，生成 ISO/IEC/IEEE 29148 结构化文档

**📈 对比分析**

通过两名专家评估完整性、清晰度、连贯性等七维度得分，结果显示 AI+协作组得分最高（整体 4.0+）

**⚠️ 局限性**

局限在样本量少、仅一场景、受限于平台文字交互、提示与模型表现差异可能影响结果

---

## 186. Towards Version-aware Operations and Transaction Memories for Multi-layer MeMo

**arXiv ID:** 2606.24040 | [PDF](https://arxiv.org/pdf/2606.24040v1)

**作者:** Peiran Li `[一作]` `[通讯]` (Freie Universität Berlin), Peiran Li (Freie Universität Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了多层MeMo模型的版本感知操作层，通过事务记忆(V-CMM和T-CMM)实现对知识变更的直接内存编辑，而无需重新训练模型。

**💡 创新点**

创新点在于将知识更新抽象为有序的原子编辑事务，并利用关联矩阵存储版本映射与事务模板，实现可重用、可追溯的内存编辑；同时兼顾直接序列级修改与结构化diff级变更。

**🔧 技术方法**

使用关联矩阵记忆(CMM)技术实现记忆存取；设计两级辅助CMM（V-CMM用于版本索引，T-CMM用于事务内容）；采用原子记忆/遗忘操作编译为可执行事务；利用模板化事务以提升重用性。

**📊 数据集**

论文中未使用公开大规模数据集，评估示例基于人工构造的 toy 事实（如 Pluto 分类变更、Asthma 父类迁移）以及结构化差异描述(KGCL)等。

**📈 对比分析**

方法对比基线包括无更新、检索增强、仅日志补丁、持续训练/重新训练以及单事实参数编辑；评估指标为更新成功率、旧知识抑制、历史保留、回滚/追踪正确性、局部性与事务重用。当前为理论评估路线，尚未给出实验性能数据。

**⚠️ 局限性**

限制在于只能处理可表示为MeMo内存关联的局部知识变更，无法应对全量重写；事务执行依赖于完整的日志记录以保证可回滚，模板化机制可能在高度动态场景下产生维护开销；跨层次或跨模型的知识迁移仍需进一步研究。

---

## 187. Rationalizing collective revealed preferences with an application in fair resource allocation

**arXiv ID:** 2606.23985 | [PDF](https://arxiv.org/pdf/2606.23985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 188. 3D Masked Autoencoders are Robust Learners of Volumetric and Multimodal Cellular Representations for Microscopy

**arXiv ID:** 2606.23964 | [PDF](https://arxiv.org/pdf/2606.23964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 189. BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks

**arXiv ID:** 2606.24162 | [PDF](https://arxiv.org/pdf/2606.24162v1)

**作者:** Jin Huang `[一作]` (University of Michigan), Qiaozhu Mei `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为BehaviorBench的综合基准，评估基础模型在行为预测、战略决策、受试者特征推断和行为知识应用四大能力，并同时考察个体与分布层面的表现。

**💡 创新点**

创新点包括①提出个体与分布双重评价维度；②覆盖四大核心能力；③通过在行为数据上微调生成专用行为基础模型并展示其优势。

**🔧 技术方法**

采用大型语言模型及其微调技术，并通过多任务评估框架、Wasserstein距离、MAE、准确率等指标进行测评。

**📊 数据集**

使用MobLab经济实验数据、Big Five人格问卷以及2025年期刊论文标题摘要等多源行为数据集。

**📈 对比分析**

采用pairwise win‑rate对比方法，结果显示专有LLM在个体预测和知识推理上表现最佳，行为专用模型在分布对齐上更强，Gemini 3.1 Pro与新微调模型在两层面均排名第一。

**⚠️ 局限性**

局限性：缺乏对上下文推理的量化评估；仅在固定提示下测试，未体现提示优化带来的提升；以及对更广泛行为类型和大规模数据的覆盖有限。

---

## 190. Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning

**arXiv ID:** 2606.24133 | [PDF](https://arxiv.org/pdf/2606.24133v1)

**作者:** Chenhao Dang `[一作]` (China Electronics Technology Group Corporation 15th Research Institute), Mingjie Liao `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在线数据混合框架——Holistic Data Scheduler（HDS），通过强化学习动态调整预训练数据的混合比例；

**💡 创新点**

创新点在于将数据混合视为多目标强化学习问题，设计了整合数据质量、梯度对齐、词汇多样性与模型稳定性的三维奖励函数；

**🔧 技术方法**

采用Soft Actor-Critic（SAC）算法，配合轻量级Transformer结构的actor/critic网络实现连续控制；

**📊 数据集**

使用The Pile大规模多域文本数据进行LLM预训练；

**📈 对比分析**

与静态TPW、ODM及AC-ODM对比，HDS在相同目标困惑度下只需约44%训练步骤，达到57%速度提升，最终困惑度降低13.6%，MMLU 0-shot提升7.2%；

**⚠️ 局限性**

主要限制在于奖励权重与网络规模需手工调参，且对更大模型的鲁棒性与可迁移性尚需进一步验证。

---

## 191. TuringViT: Making SOTA Vision Transformers Accessible to All

**arXiv ID:** 2606.24253 | [PDF](https://arxiv.org/pdf/2606.24253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 192. Embodied Explainability and Ontological Obstacles: Why We Struggle to Explain the Answers of Large Language Models (LLMs)

**arXiv ID:** 2606.23840 | [PDF](https://arxiv.org/pdf/2606.23840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 193. Token-to-Token Alignment of Text Embeddings for Semantic Blending

**arXiv ID:** 2606.24021 | [PDF](https://arxiv.org/pdf/2606.24021v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 194. The impact of generative artificial intelligence on academic development of Chinese students in humanities and social sciences

**arXiv ID:** 2606.24104 | [PDF](https://arxiv.org/pdf/2606.24104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 195. ARIA: Adaptive Region-Based Importance Allocation for Conditional Diffusion Distillation

**arXiv ID:** 2606.23898 | [PDF](https://arxiv.org/pdf/2606.23898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 196. ABACUS: Adapting Unified Foundation Model for Bridging Image Count Understanding and Generation

**arXiv ID:** 2606.23835 | [PDF](https://arxiv.org/pdf/2606.23835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 197. Ground Then Rank: Revisiting Knowledge-Based VQA with Training-Free Entity Identification

**arXiv ID:** 2606.23881 | [PDF](https://arxiv.org/pdf/2606.23881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 198. Do Language Models Pass the Bechdel Test? Auditing Gender Biases in LLM-Generated Screenplays

**arXiv ID:** 2606.24022 | [PDF](https://arxiv.org/pdf/2606.24022v1)

**作者:** Megha N. Govindu `[一作]` (University of Pennsylvania), Danaé Metaxa `[通讯]` (University of Pennsylvania)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对三种先进LLM（GPT‑5、Gemini 3 Pro、Claude Sonnet 4.5）生成的电影剧本与768部人工撰写剧本进行性别代表性审计，使用自动化Bechdel测试和社交网络分析（中心性、同质性、三角关系）评估女性角色的可见度和影响力。

**💡 创新点**

首次将社交网络分析方法与自动化Bechdel测试相结合，用于量化LLM长文本输出中的性别偏差；并通过对比人类剧本与LLM剧本，揭示LLM在某些网络指标上可表现得比人类更公平的可能性。

**🔧 技术方法**

技术手段包括：①基于剧本文本构建角色互动网络（节点为角色，边为对话交互）；②使用NomQuamGender进行基于名字的性别推断；③训练RBF SVM预测Bechdel第三步结果；④计算中心性（度、接近、介数）、同质性（homophily）与三角关系等SNA指标；⑤利用逻辑回归和OLS回归对不同剧本类型进行统计比较。

**📊 数据集**

数据集：768部公开人类剧本（IMSDb+TMDb），配有Bechdel评分和剧情摘要；对每部剧本使用匿名摘要生成三种LLM剧本（770 GPT、771 Gemini、752 Claude），共计约2700部剧本；另外采集角色对话与剧情结构信息以构建网络。

**📈 对比分析**

比较方法：对Bechdel测试采用二元通过/不通过和分数0‑3；对网络指标使用回归模型控制交互次数。结果显示：人类剧本通过率最高（≈66%），GPT≈74%，Gemini≈50%，Claude≈46%；在中心性、同质性、三角关系等指标上，GPT剧本在部分度量上表现更好（如更低的女性同质性、更高的女性三角比例），但总体仍显示女性中心性低于男性。整体上LLM剧本在多数衡量标准下的性别偏差与人类剧本相近，均存在代表性不足。

**⚠️ 局限性**

局限性包括：①性别推断仅基于名字，忽略非二元与跨性别角色；②Bechdel测试与网络指标对剧情细节与角色类型的假设有限；③LLM生成剧本受训练数据影响，可能存在抄袭或过度仿造；④模型生成不确定性导致同一提示可产生多样结果；⑤未对种族、年龄等其他身份交叉维度进行深入分析；⑥数据来源（公开剧本与TMDb）存在标签不完整、主观评分误差等问题。

---

## 199. A Benchmark for Hallucination Detection in VLMs for Gastrointestinal Endoscopy

**arXiv ID:** 2606.24115 | [PDF](https://arxiv.org/pdf/2606.24115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. JupOtter: Cell-Level Bug Detection in Jupyter Notebooks

**arXiv ID:** 2606.23877 | [PDF](https://arxiv.org/pdf/2606.23877v1)

**作者:** Lukas Ottenhof `[一作]` (University of Alberta), Thibaud Lutellier `[通讯]` (University of Alberta)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了JupOtter，一个面向Jupyter Notebook细粒度（cell级）缺陷检测系统，并发布了包含21,303个Python Notebook、661,909个代码cell的标注数据集OtterDataset。

**💡 创新点**

三大创新：① 维护Notebook cell边界的专属分词策略；② 针对Notebook的cell级缺陷预测方法；③ 公共发布首个含细粒度cell标签的Notebook大规模数据集。

**🔧 技术方法**

采用预训练Transformer（CodeT5）为编码器，结合自定义chunking分词、边界标记、per‑sample混合精度训练以及多段输入处理，以实现高效的cell级预测。

**📊 数据集**

主要使用OtterDataset进行训练和验证，外部评测集包括约9,313个Jupyter Errors Notebook和4,892个CodeParrot子集Notebook，合计覆盖多种错误类型。

**📈 对比分析**

与Flake8、GPT‑4o‑mini、Gemini 3 Flash等基线对比，cell级F1最高达0.93，文件级F1优于传统静态分析和LLM；总体性能优于基线但对少见运行时错误的召回仍偏低。

**⚠️ 局限性**

限制：数据集主要聚焦实现错误，对运行时/环境错误覆盖不足；分词仅处理代码cell，忽略Markdown上下文；模型推理时延和显存需求高，未实现实时或轻量级部署。

---

## 201. Privacy Engineering: A Systematic Literature Review

**arXiv ID:** 2606.23696 | [PDF](https://arxiv.org/pdf/2606.23696v1)

**作者:** Nemania Borovits `[一作]` (Tilburg University), Willem-Jan van den Heuvel `[通讯]` (Eindhoven University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对2018-2025年90篇关于隐私工程的实证与案例研究进行了系统性综述，构建了13个维度的统一框架并绘制了生命周期与领域交互图谱。

**💡 创新点**

创新点在于首次将隐私工程从机制、评估、治理三大核心进行系统归纳，并提出模型化与规范化（Modeling & Specification）作为核心桥接，形成可复制的评估与更新支架。

**🔧 技术方法**

采用了系统文献检索与主题综合（thematic synthesis）方法，并利用共现矩阵、生命周期映射与交叉维度分析等技术进行定量与定性结合的元分析。

**📊 数据集**

研究使用的“数据集”是从IEEE Xplore、Scopus、ACM Digital Library和PubMed检索得到的90篇符合标准的学术论文集合，未使用传统机器学习数据集。

**📈 对比分析**

方法上不涉及算法性能比较，而是通过维度出现频率、共现次数与生命周期位置等度量来评估研究的覆盖度与成熟度，结果显示PETs与治理与运营核心的共现最频繁。

**⚠️ 局限性**

局限性包括对特定维度（如事件响应、终身管理、数据最小化）的关注不足、指标报告不统一、缺乏纵向实测与行业级案例验证，以及搜索策略与时间窗可能导致的遗漏与偏倚。

---

## 202. Project Ariadne: Prompt-Conditioned Route Generation for Synthesis Planning

**arXiv ID:** 2606.24184 | [PDF](https://arxiv.org/pdf/2606.24184v1)

**作者:** Anton Morgunov `[一作]` (Yale University), Victor S. Batista `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Ariadne的解码器仅模型，能够在单一提示中同时输入目标分子、可选约束（如路线深度、指定起始材料）并生成完整的多步合成路线；

**💡 创新点**

创新点在于将传统需要为每种规划任务训练单独模型的“直接生成”方法统一为一次性解码器语言模型，并通过提示字段实现对不同约束的灵活响应；

**🔧 技术方法**

使用了预归一化的解码器Transformer，RoPE位置编码，RMSNorm，SwiGLU前馈网络，并在训练中采用下一词交叉熵损失、Muon+Moonshot混合精度加速、Beam搜索生成；

**📊 数据集**

主要使用PaRoutes数据集的Canonical split（包括路由保留和反应保留子集）以及RetroCast/PaRoutes benchmark；

**📈 对比分析**

在RetroCast/PaRoutes基准上，一个24层Ariadne模型在约束路由深度和指定叶子时分别比DESP提升13.7和31.2个百分点的Solv-0，并在必需叶子Top‑10上达到81.2%（比DESP高20个百分点）且生成时间仅为DESP的1/17；在标准目标重建任务上与DMS Explorer XL相近，且在路由保留集上Top‑10提升约18个百分点；

**⚠️ 局限性**

局限性包括：目前仅能基于固定的商业库存（ASKCOS Buyables）进行终点过滤，无法在生成时直接约束任意用户自定义库存；首个断裂（root reaction）选择仍是主要瓶颈，提示字段或动态Beam策略尚未完善；缺乏更高级别（Tier‑1‑3）的可执行性验证，导致实验可行性评估受限。

---

## 203. From Open Waters to Enclosed Cabins: ProteusVPR for Cross-Scene Visual Place Recognition in Maritime Perception and Cabin Inspection

**arXiv ID:** 2606.24234 | [PDF](https://arxiv.org/pdf/2606.24234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 204. Latent Visual States for Efficient Multimodal Reasoning

**arXiv ID:** 2606.24233 | [PDF](https://arxiv.org/pdf/2606.24233v1)

**作者:** Xiuwei Chen `[一作]` (Sun Yat-sen University), Xiaodan Liang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出EVA框架，利用连续潜在视觉表示代替外部工具调用，提升视觉推理效率。

**💡 创新点**

核心创新是生成可自适应的潜在视觉槽（Latent_slots）并通过D‑GSPO算法在RL阶段解耦潜在与离散表示，解决策略偏移问题。

**🔧 技术方法**

技术包括连续链式思维、VQ‑VAE/视觉编码、MSE与CE联合训练、三阶段SFT+RL、KL约束、t‑SNE可视化等。

**📊 数据集**

构建EVA‑230K（来自DeepEyes、ReFocus、Thyme、Visual‑CoT等）用于SFT，EVA‑RL数据集用于RL。

**📈 对比分析**

与Qwen2.5‑VL‑7B、32B、DeepEyes、GPT‑4o等比较，EVA在MME‑Real‑Lite、HRBench、V*等多项指标上提升12–20%，并在高分辨率下推理速度提升84%。

**⚠️ 局限性**

局限在于潜在槽数量过多会导致性能下降，且对极端复杂视觉任务的泛化尚待验证。

---

## 205. FiCA: Feed-forward instant Gaussian Codec Avatars from a Single Portrait Image

**arXiv ID:** 2606.24232 | [PDF](https://arxiv.org/pdf/2606.24232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 206. SkyChain Intelligence: A Blockchain-Secured Multi-Agent DRL Framework for Low-Altitude Embodied Artificial Intelligence

**arXiv ID:** 2606.24193 | [PDF](https://arxiv.org/pdf/2606.24193v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 207. Inside Crypter-as-a-Service: An Ecosystem Analysis of the exploit.in Underground Forum Research Talks

**arXiv ID:** 2606.24226 | [PDF](https://arxiv.org/pdf/2606.24226v1)

**作者:** Mathieu Jeannot `[一作]` (Université de Lorraine), Romain Guittienne `[通讯]` (Université de Lorraine)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对俄罗斯语言网络犯罪论坛 exploit.in 上 2020-2025 年的 Crypter-as-a-Service（CraaS）生态进行了纵向定量与定性分析，构建了卖家与买家五大分类、价格模型与信任治理机制，并用社交网络分析揭示核心代理与信任中介角色。

**💡 创新点**

创新点在于：①对俄语黑客论坛的长期数据采集与分析，填补了此前仅关注英文论坛的研究空白；②将技术细节、经济模式、治理结构与社交网络结构统一框架，首次揭示“信任中介”在暗网交易中的系统性作用；③提出基于托管与声誉的多层治理模型，并与 HackForums 进行对比。

**🔧 技术方法**

技术方法包括关键词过滤、ChatGPT 5.2 辅助标注与人工验证（标注准确率 0.98），社交网络分析（Betweenness、PageRank、Eigenvector 关联研究），以及基于 LLM 的结构化信息抽取与经济模型识别。

**📊 数据集**

使用了约 1,000,000 条帖子中筛选出的 491 条主题和 2,949 条回复的数据集，涵盖 1,058 名独立用户，时间跨度 Jan 2020–Aug 2025，主语言为英文、俄文及混合。

**📈 对比分析**

通过与 HackForums 的对照，发现 exploit.in 的 CraaS 更为专业化、治理更为制度化，价格模型更细粒度、技术水平更高，网络结构显示核心角色主要是信任中介；虽然未直接量化性能，但从交易频率与价格跨度推断其商业化程度更高。

**⚠️ 局限性**

局限性包括：只关注单一论坛与关键词匹配，可能漏掉私聊与其他关联服务；LLM 标注在边缘案例上可能存在系统偏差；声誉数据仅覆盖发帖人，导致网络-声誉关联受选择偏差影响；且无法验证实际交易量与服务质量。

---

## 208. A Pāninian Foundation for Indic Language Processing

**arXiv ID:** 2606.24172 | [PDF](https://arxiv.org/pdf/2606.24172v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 209. Navigating User Behavior toward Personalized Multimodal Generation

**arXiv ID:** 2606.24196 | [PDF](https://arxiv.org/pdf/2606.24196v1)

**作者:** Hengji Zhou `[一作]` (South China University of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了NaviGen框架，能够将用户交互历史自动转换为可直接用于文本到图像/视频生成的执行指令，实现个性化多模态内容生成；

**💡 创新点**

核心创新在于双标识符表示（CID+TID）同时兼顾行为建模与语义桥接，以及两阶段SFT+RL的训练策略，利用进化搜索与多任务奖励提升指令生成的可执行性和偏好匹配；

**🔧 技术方法**

采用Qwen3系列大型语言模型作为基础，结合残差向量量化的协同编码、可变长度文本标识符、链式推理教师提示、GRPO强化学习、以及多任务奖励（CID层次奖励和三角指令一致性奖励）；

**📊 数据集**

在三个公开数据集上评估：Amazon产品评论（Product）、游戏项目（Games）以及OpenOneRec短视频（Short Videos），每个数据集包含用户-物品交互序列；

**📈 对比分析**

与PMG、Pigeon、RAGAR、Cipher、Prose、Triple等个性化生成基线以及SASRec、TIGER、LC-Rec等协同过滤方法对比；实验表明NaviGen在大多数指标（一致性、相关性、审美、创新）上均优于非oracle基线，尤其在视频生成中取得9/12指标最佳；

**⚠️ 局限性**

局限性包括对用户交互历史的隐私和安全依赖，需要明确用户同意与数据匿名化；模型仍受限于历史数据的稀疏性与噪声，且未在真实用户环境中验证，可能导致生成内容偏离用户真实意图。

---

## 210. Agon: An Autonomous Large-Scale Omnidisciplinary Research System Built on Prompt Economy

**arXiv ID:** 2606.24177 | [PDF](https://arxiv.org/pdf/2606.24177v1)

**作者:** Youran Sun `[一作]` (University of Maryland), Haizhao Yang `[通讯]` (University of Maryland)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个零代码、跨学科、多代理的研究自动化系统Agon，能够从问题选取到实验执行再到论文撰写全流程自动化，无需人工编写实验代码。

**💡 创新点**

提出六大设计原则（Prompt Economy、Future‑Facing、Minimal Prompts、OmniDisciplinary、Massive Parallelism、Zero‑Code），构建对抗性审计循环，开发深度文献研究机制，并制定四维失败分类（可观测/不可观测、可修复/不可修复、内部/外部原因），从而明确人类判断边界。

**🔧 技术方法**

利用多大语言模型（Claude、Codex、DeepSeek等）进行角色扮演、循环审计；通过prompt‑driven dispatcher实现零代码调度；采用深度文献库、即时通讯接口和前端仪表盘；实现多模型协作与二次检查。

**📊 数据集**

在十余个科研领域中实际部署，使用每个主题约400–2000篇论文做文献检索与阅读；记录实验日志、代码执行结果和论文草稿等数据；依托真实科研项目数据集进行评估。

**📈 对比分析**

通过持续部署记录（444次Prompt Economy循环，1000+科学家‑编码员‑审计员迭代），与传统人工流程对比，展示吞吐量提升、人工成本下降；系统能自动发现并修正大多数可观测错误，剩余错误被归类为不可观测，突显人类不可替代的判断。

**⚠️ 局限性**

受限于模型的感知与推理能力，无法检测某些异常（异常盲点、可解释性不足）；对实验设计、资源调度等细节仍需人工干预；不可观测错误需要人工介入；目前无法完全替代人类判断，需进一步提升模型的异常检测与因果推理能力。

---

## 211. A Synthetic Reliability-Aware PINN Benchmark for Offshore Wind Turbine Support-Structure Monitoring with Bayesian Inverse Identification

**arXiv ID:** 2606.24176 | [PDF](https://arxiv.org/pdf/2606.24176v1)

**作者:** Puneet Kant `[一作]` (Indian Institute of Technology Jodhpur), Monika Tanwar `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DigiTurbine合成可靠性感知PINN基准，集成前向PINN状态估计、逆向PINN贝叶斯先验识别桩材料与土壤刚度，以及FORM可靠性筛选，用于海上风电桩结构健康监测。

**💡 创新点**

创新点包括：①识别并克服4阶PDE逆向PINN的梯度方向冲突；②提出EMA自适应损失权重的联合共演算法；③在PINN推理流程中嵌入弱log‑normal Bayesian先验；④将FORM可靠性评估以毫秒级完成，完成完整在线监测管线<7 ms。

**🔧 技术方法**

技术栈：Physics‑Informed Neural Network（前向与逆向）、自动微分、EMA自适应损失权重、弱Bayesian先验、First‑Order Reliability Method（FORM）与SLSQP优化、PyTorch实现。

**📊 数据集**

使用纯合成数据，基于NREL 5 MW参考风机的简化Euler–Bernoulli/Winkler一维桩模型；包括10种前向配置、8种逆向实验以及多级噪声（1–10 %）验证。

**📈 对比分析**

与解析/有限差分真值对比，前向PINN 10/10通过，平均RMSE 0.135 ± 0.109 mm；推理时间GPU 0.381 ms、CPU 0.605 ms；逆向PINN无先验失败0/4，加入弱先验后8/8通过，E误差≤0.02%；FORM求解时间0.7–2.7 ms，根矩负载案例1.0 ms，比10^5样本蒙特卡洛快>40×。

**⚠️ 局限性**

局限性在于仅验证一维静态桩模型，缺乏动态气动/水动力耦合、现场SCADA数据验证；贝叶斯先验需准确设定，退化结构误差时需自适应/层级先验；FORM仅适用于线性或弱非线性限态，需进一步扩展至SORM等方法。

---

## 212. Distributed Quality-Diversity Search for Toxicity in Large Language Models

**arXiv ID:** 2606.24166 | [PDF](https://arxiv.org/pdf/2606.24166v1)

**作者:** Onkar Shelar `[一作]` (Rochester Institute of Technology), Travis Desell `[通讯]` (Rochester Institute of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了一种名为 ToxSearch‑S 的在线无监督种群化进化搜索方法，用于发现大语言模型（LLM）对毒性提示的鲁棒性，并实现了 MPI 主从分布式架构来加速搜索。

**💡 创新点**

创新点在于：①使用增量 leader‑follower 聚类，在搜索过程中自动产生并维护多个语义和行为相异的毒性攻击族群；②将 MPI 主从模式与集中式种群与种群管理相结合，使得高成本的提示变异与评估能在多核 GPU 上并行完成；③通过组合 prompt embedding 与 Moderation API 的多维距离，构造无监督的种群化距离度量，避免预定义细胞格子。

**🔧 技术方法**

技术手段包括：进化算法（steady‑state (μ+λ)）、在线无监督种群化、ensemble distance（基于 384 维 prompt embedding 与 8 维 Moderation vector 的加权距离）、MPI 主从分布式执行、GPU 并行、LLM Llama 3.1 8B‑Instruct、Perspective API 等。

**📊 数据集**

使用的数据集为公开的有害问答数据集，结合 CategoricalHarmfulQA 与 HarmfulQA 共 100 条随机采样提示，用作初始种群。

**📈 对比分析**

通过在相同评估预算 B=1000、7 次独立实验下，与原始 ToxSearch 与 RainbowPlus 进行对比：ToxSearch‑S 能达到与两者相当的最高毒性，并且累计毒性曲线更低；在多样性方面，RainbowPlus 在嵌入空间的扩散更广，而 ToxSearch‑S 产生更多局部行为族群（DBSCAN 聚类数更高）。MPI 并行实现将壁钟时间压缩约 1.8–3.2 倍，且在统计显著性上与顺序执行保持一致。

**⚠️ 局限性**

局限性包括：实验样本量小（n=7）、预算仅到搜索早中期，未观察到种群成熟或停滞；仅使用单一 PG/RG 堆栈（Llama3.1 8B‑Instruct）和单一 Moderation API，缺乏跨模型/评估器的泛化；MPI 并行的合并策略可能引入评估时间偏差；未来需在更大预算、更丰富模型和异步合并策略下进一步验证。

---

## 213. AutoSpec: Safety Rule Evolution for LLM Agents via Inductive Logic Programming

**arXiv ID:** 2606.24245 | [PDF](https://arxiv.org/pdf/2606.24245v1)

**作者:** Pingchuan Ma `[一作]` (Zhejiang University of Technology), Xiaoqin Zhang `[通讯]` (Zhejiang University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建一个自动演化已部署的安全规则系统，使LLM代理在运行时通过用户的安全/不安全标注不断改进其规则集。

**💡 创新点**

创新点在于将安全规则演化表述为ILP‑引导的CEGIS问题：利用对抗样本挖掘、ILP学习判别谓词、受限编辑操作三段循环，实现可解释、可审计的规则自适应；并首次在LLM代理场景中演示该方法。

**🔧 技术方法**

核心技术包括：反例驱动的规则评估与挖掘、基于ILASP的诱导逻辑编程学习判别谓词、受限的规则编辑算子（AddConjunct/Exception、Relax、AddDisjunct），以及整体的CEGIS循环。

**📊 数据集**

使用两个领域的数据集：1）代码执行域，91条不安全Trace来自RedCode-Exec，100条安全Trace由GPT生成；2）具象代理域，100条Trace来自SafeAgentBench，分布在不同危险情境与安全任务。

**📈 对比分析**

与四个基线对比（专家规则、LLM分类器、无ILP CEGIS、随机搜索）。在代码域F1提升至0.980（比基线高≈40%），在具象代理域F1提升至0.933（比基线高≈89%）。收敛仅需4–5轮，ILP引导显著加快搜索并提升最终性能。

**⚠️ 局限性**

局限性包括：仅处理基于谓词的事件级规则，难以捕捉复杂时间/状态属性；需要丰富的谓词库和人工标注；未保证全局最优；对未见场景的泛化仍依赖手工扩充谓词。

---

## 214. Geometry-Instructed Video Editing

**arXiv ID:** 2606.24225 | [PDF](https://arxiv.org/pdf/2606.24225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 215. MorVess: Morphology-Aware Pulmonary Vessel Segmentation Network

**arXiv ID:** 2606.24214 | [PDF](https://arxiv.org/pdf/2606.24214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 216. Inclusive Interactive Collisions for Multi-View Consistent Compositional 3D Generation

**arXiv ID:** 2606.24206 | [PDF](https://arxiv.org/pdf/2606.24206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. Importance of Intent-Sharing for V2X-based Maneuver Coordination

**arXiv ID:** 2606.24203 | [PDF](https://arxiv.org/pdf/2606.24203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 218. FORESEE: A Cooperative Lane Change Model for Connected and Automated Driving

**arXiv ID:** 2606.24201 | [PDF](https://arxiv.org/pdf/2606.24201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 219. Tri-Efficient Transfer Learning for Point Cloud Videos

**arXiv ID:** 2606.24175 | [PDF](https://arxiv.org/pdf/2606.24175v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 220. Trimming the Long-Tail of Visual World Modeling Evaluation

**arXiv ID:** 2606.24256 | [PDF](https://arxiv.org/pdf/2606.24256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 221. A Dynamic Coupling Theory of Expertise Through Thinking Flow and Workflow Evolution

**arXiv ID:** 2606.24197 | [PDF](https://arxiv.org/pdf/2606.24197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 222. Data Scale, Not Latency, Shapes Cross-Lingual Encoder Transfer in Streaming ASR

**arXiv ID:** 2606.24169 | [PDF](https://arxiv.org/pdf/2606.24169v1)

**作者:** Nenad Banfic `[一作]` `[通讯]` (CoreAI, Microsoft), Nenad Banfic (CoreAI, Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究通过对八种欧洲语言进行大规模实验，系统评估了多语种编码器与仅英语编码器在流式 ASR 迁移学习中的效果；

**💡 创新点**

创新点在于揭示多语种初始化主要是数据效率提升而非延迟优势，提出了以功率律描述的优势衰减、跨延迟不敏感性以及 INT4 权重量化的可行性；

**🔧 技术方法**

使用了 0.6B 参数的 Cache‑Aware FastConformer RNN‑T 架构，采用多延迟训练、FastEmit 规约、SpecAugment 等技术，并对编码器进行 4‑bit 权重量化；

**📊 数据集**

实验数据涵盖 Common Voice、Multilingual LibriSpeech、VoxPopuli、CML‑TTS、YODAS‑Granary 等公开语料，评估集包括 CV、MLS、VoxPopuli、FLEURS 四个测试集；

**📈 对比分析**

通过在不同语言、不同训练时长（100‑2500 h）、三种流式延迟（160/560/1120 ms）以及离线模式下对比 WER，结果显示多语种初始化的优势随数据增长呈指数级衰减，延迟变化对优势影响不大，INT4 量化平均导致 WER 仅升高约 0.5 pp；

**⚠️ 局限性**

局限性包括仅评估单一 0.6B 模型族、仅针对欧洲语言、未探索更大规模或其他语言族、量化方案未针对特定语言调优，以及对离线/流式混合模型的综合性能尚未深入研究。

---

## 223. The Evaluation Cost of Task Specialization in Evolutionary Multi-Robot Systems

**arXiv ID:** 2606.24191 | [PDF](https://arxiv.org/pdf/2606.24191v1)

**作者:** Paolo Leopardi `[一作]` (University of Konstanz), Tanja Katharina Kaiser `[通讯]` (University of Technology Nuremberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在仿真环境中比较了在固定总评估预算下，进化任务专门化机器人控制器与通用控制器的性能，并分析了两者在不同机器人团队规模下的成本收益关系。

**💡 创新点**

创新点在于首次量化了专门化与通用化控制器的优化成本，并揭示了团队规模增大时，专门化控制器取得优势所需的评估预算显著下降；同时提出了“盈亏点”概念来衡量两种策略的相对优劣。

**🔧 技术方法**

采用基于 ARGoS 的物理仿真平台，使用差速驱动的 Foot-bot 机器人；控制器为单隐藏层全连接前馈人工神经网络；通过基因算法（锦标赛选择、单点交叉、高斯变异）直接优化网络权重；评估指标为在固定时间窗口内运送到目标区域的物体数量。

**📊 数据集**

数据集为仿真生成的10个可检索圆柱形物体，放置在源区域；实验使用机器人团队规模 S ∈ {2,4,6,8}，并在每个规模下分别进化通用控制器与分为 “Dropper” 与 “Collector” 两个子任务的专门化控制器。

**📈 对比分析**

通过对不同评估预算 Ẽ 进行后评估（每个预算下随机 20 次试验），比较两种策略在目标区物体数量上的中位数，并使用 Mann‑Whitney U 检验与 Holm‑Bonferroni 校正来评估统计显著性。结果显示：对小规模团队（S=2），通用控制器在 Ẽ ≤ 1000 时表现优于专门化；而随着 S 增大，专门化控制器达到并超越通用控制器所需的 Ẽ 迅速下降（S=8 时仅需 Ẽ≈30）。

**⚠️ 局限性**

局限性包括：仅考虑了预先划分的两子任务且未探索任务分割本身的进化；实验仅在单一仿真场景下进行，缺乏现实硬件验证；评估预算被固定为 5000 代，未研究更广泛的预算范围；此外，当团队规模较大时，通用控制器可能自发形成类似专门化的行为，导致实验结果对“专门化”定义的依赖。

---

## 224. Probing the Misaligned Thinking Process of Language Models

**arXiv ID:** 2606.24251 | [PDF](https://arxiv.org/pdf/2606.24251v1)

**作者:** Kaiwen Zhou `[一作]` (Anthropic Fellows), William Saunders `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过构建18个细粒度的误导性思维指示器，对大语言模型内部激活进行线性探测，构建可扩展的误对齐监测系统。

**💡 创新点**

创新点在于：①提出了覆盖多种误对齐行为的指示器分类法；②设计自动化多阶段生成合成训练数据的 pipeline；③结合内部探测与 LLM 判别的两阶段级联，以低成本实现高召回与低误报；④通过跨语言评估展示探测器对 CoT 可解释性的鲁棒性。

**🔧 技术方法**

使用了线性回归（Logistic Regression）探测器、Meta‑Plan 指导的多轮对话生成、LLM 判别器（Claude Haiku / Qwen3）、内部激活读取（GLM‑4.7 Flash 的第27层残差流）以及两阶段级联阈值调优。

**📊 数据集**

训练数据来自自动化行为诱导流水线（Bloom 等）生成的多行为对话、对抗性负样本与正常控制；评估数据包括：Bloom 自动化诱导的5种误对齐行为、公开的误对齐基准（DeceptionBench、Sycophancy‑Eval 等）、以及 1500 条自然良性对话（WildChat、LMSYS‑Chat、Flan）。

**📈 对比分析**

与强 LLM 判别器（Claude Haiku）及其指示器版本进行比较。线性探测器在 out‑of‑distribution 测试中获得 0.936 AUROC，略低于 Haiku 的 0.952；两阶段级联（探测器+Haiku）在保持 0.950 AUROC 的同时，将触发的 LLM 调用降至 6.1%（相比纯 Haiku 的 100%），显著降低计算成本。

**⚠️ 局限性**

局限性包括：①探测器对阈值高度敏感，导致近阈值误判；②部分误报源自模型仅考虑但不实施误对齐的情境；③在高度复杂或隐蔽的 CoT 任务下仍可能失效；④对其它模型与误对齐行为的泛化尚未充分验证。

---

## 225. Aspect-Based Sentiment Evolution and its Correlation with Review Rounds in Multi-Round Peer Reviews: A Deep Learning Approach

**arXiv ID:** 2606.24188 | [PDF](https://arxiv.org/pdf/2606.24188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 226. Dialogue to Discovery: Attribute-Aware Preference Elicitation for Conversational Product Search Assistants

**arXiv ID:** 2606.24194 | [PDF](https://arxiv.org/pdf/2606.24194v1)

**作者:** Sarthak Harne `[一作]` (Microsoft Research), Shubham Agarwal `[通讯]` (UC Berkeley)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种属性驱动的对话式产品搜索框架 D2D，能够动态决定何时向用户推荐商品、询问哪些属性，并通过对话逐步逼近目标商品。

**💡 创新点**

创新点在于：① 将属性价值偏好（AVP）与属性不确定性（APU）结合，用属性累计熵（ACE）评估属性区分度；② 采用 TOP 重叠集（TOI）来判断何时进行推荐；③ 引入用户耐心模型，根据推荐相关性、问题相关度、复杂度与信息量动态调整对话节奏。

**🔧 技术方法**

技术方法包括：LLM（如 GPT）进行 AVP 更新与回复生成；Sentence‑BERT 进行语义检索；统计熵与不确定性计算；基于阈值的推荐时机决策与属性优先级排序。

**📊 数据集**

使用从 Amazon Reviews 语料中构建的三类（电子、家居与厨房、体育与户外）各约 2,000 件商品的数据集，配合用户评论生成用户档案与目标商品。

**📈 对比分析**

与传统检索基线（BM25、dense retrieval）和全 LLM 基线对比；实验显示 D2D 在成功率上提升 22–30%，放弃率下降约 12%，对话长度缩短 50%+，并在用户研究中获得更高的满意度与质量评价。

**⚠️ 局限性**

局限性包括：① 仅基于仿真用户模型和固定耐心，未覆盖真实用户的行为变化；② 仅使用英文 GPT 系列 LLM，可能带来偏见；③ 缺乏大规模真实用户测试，且未深入研究多模态属性或更复杂的商品结构。

---

## 227. Semantic Lock: Synchronization Based on the Analysis of the Operation Conflict Graph

**arXiv ID:** 2606.24250 | [PDF](https://arxiv.org/pdf/2606.24250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 228. Lightweight Transformer Models for On-Device Fault Detection: A Benchmark Study on Resource-Constrained Deployment

**arXiv ID:** 2606.24173 | [PDF](https://arxiv.org/pdf/2606.24173v1)

**作者:** Disha Patel `[一作]` `[通讯]` (California State University), Disha Patel (California State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了轻量级Transformer模型在资源受限设备上的故障检测性能，并与传统机器学习基线进行了全面基准测试。

**💡 创新点**

创新点在于对Transformer与传统模型的真实F1、模型大小、CPU推理延迟进行诚实对比，并提出了基于INT8量化和两阶段自适应推理的部署方案。

**🔧 技术方法**

采用了DistilBERT、TinyBERT（4L/6L）、MobileBERT等轻量化Transformer，配合动态INT8量化、加权交叉熵损失、两阶段推理管道以及传统RF、XGBoost、SVM、LR等基线。

**📊 数据集**

使用了NASA C-MAPSS、SECOM半导体制造、UCI AI4I 2020预测维护三大公开数据集。

**📈 对比分析**

在相同数据划分与评估指标（F1、AUC、模型大小、CPU延迟）下比较，发现Transformer在C-MAPSS上可达到与XGBoost相当的87.9% F1，但模型尺寸和延迟高达两百五十多MB和133ms；TinyBERT-4L在55MB/18ms下仅差0.1%；量化后可进一步压缩；自适应管道将97.9%样本由轻量模型处理，实现87.6% F1与19.5ms平均延迟。

**⚠️ 局限性**

限制包括：在严重类别不平衡数据集上表现不佳；MobileBERT训练失败；未评估剪枝、量化感知训练；CPU延迟测量仅在Colab上，未覆盖移动NPU等硬件；转文本表示带来的token化开销。

---

## 229. Unified Dominance Graph for Interval-Predicate Approximate Nearest Neighbor Search

**arXiv ID:** 2606.24204 | [PDF](https://arxiv.org/pdf/2606.24204v1)

**作者:** Kwun Hang Lau `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出统一支配图（UDG）框架，统一处理闭两端约束的区间谓词近似最近邻搜索（IPANNS），通过将不同区间谓词映射到同一二维支配空间并在图中用边标签压缩所有查询状态。

**💡 创新点**

创新点在于（1）将多种区间谓词（包含、重叠等）通过语义映射归一化为相同的支配谓词；（2）设计了可压缩标签的图结构，实现在单一索引内实现所有查询状态的无损压缩；（3）引入有效性保持补丁边（patch edges）以提升在极端过滤（高选择性）下的图导航性。

**🔧 技术方法**

技术手段包括语义映射、二维支配标签化、离线构造时的候选搜索与Prune、动态跳跃（leap）加速构造、补丁边维护以及可选的最大跳跃（MaxLeap）优化。

**📊 数据集**

实验使用公开 ANN 基准数据集（SIFT1M、DEEP1M、DBpedia-OpenAI）以及真实区间工作负载（S&P 500 股票区间、纳斯达克事件时间），在这些数据集上生成均匀或其他分布的区间元数据。

**📈 对比分析**

与 Hi‑PNG、PostFilter‑HNSW、PreFilter、ACORN 等基线对比，UDG 在各选择性（0.1%–50%）下均保持高召回（Recall@10≈1.0）并实现最高 QPS，尤其在极端小选择性时仍优于其它方法；构造时间和索引空间也低于专用区间索引且与通用混合搜索方法相当。

**⚠️ 局限性**

局限性包括（1）理论上构造复杂度为 O(n²M) 的最坏情况，实际取决于区间分布；（2）压缩正确性依赖于“准确搜索假设”（ASA），若在线搜索近似不足可能导致失效；（3）在极端选择性或非常大规模数据时，补丁边可能仍无法完全保证图连通性，需进一步优化。

---

## 230. M^2C-EvDet: Multi-Domain Multi-Order Cross-Modal Knowledge Distillation for Event-based Object Detection

**arXiv ID:** 2606.24248 | [PDF](https://arxiv.org/pdf/2606.24248v1)

**作者:** Wei Bao `[一作]` (Tsinghua University), Yue Gao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了M^2C-EvDet，一个多域多阶跨模态知识蒸馏框架，用于事件摄像机目标检测；

**💡 创新点**

通过自适应频率分离的特征蒸馏与基于超图的高阶关系蒸馏，有效解决跨模态知识混叠与低阶关系局限；

**🔧 技术方法**

结合离散小波变换、自适应融合、超图注意力与K‑means聚类、Transformer/CNN特征提取及教师‑学生架构；

**📊 数据集**

在DSEC-Detection、DSEC-Det与PKU‑DAVIS‑SOD等RGB‑Event目标检测数据集上训练与评估；

**📈 对比分析**

与多种事件检测及RGB‑Guided蒸馏方法对比，DSEC-Detection上mAP提升至28.3/45.9（相较RVT 27.7/44.2），实现SOTA；

**⚠️ 局限性**

受RGB教师质量、低光照或过曝等低质量场景影响，仍存在误检/漏检，无法完全消除模态差异。

---

## 231. Towards Federated Long-Tailed Graph Learning: An Energy-Guided Dual Decoupling Approach

**arXiv ID:** 2606.24237 | [PDF](https://arxiv.org/pdf/2606.24237v1)

**作者:** Lianshuai Guo `[一作]` (Shandong University), Wenyu Wang `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向联邦图学习的长尾分类框架 FedEPD，采用双重解耦策略来分离拓扑清洗和语义再校准。

**💡 创新点**

创新点在于：① 采用分布感知 Dirichlet 能量剪枝过滤异质性边；② 通过服务器辅助本地共识提取全局原型并进行空间低通原型注入；③ 采用两阶段交替优化保护多数类决策边界。

**🔧 技术方法**

使用了 Dirichlet 能量剪枝、PPR 近似、低通滤波、拓扑感知对数调整、GNN（GCN）训练等技术。

**📊 数据集**

在六个真实图数据集上评估：CoraFull、ogbn-arxiv、Amazon-Electronics、Amazon-Clothing、Roman-Empire、Email。

**📈 对比分析**

与 11 个基线相比，FedEPD 在整体准确率、平衡准确率和宏 F1 上均表现最优，尤其在尾类上提升 4–5% 以上。

**⚠️ 局限性**

局限性：对超参数 γ、μ 的调优仍需实验；在极端异构图中能量剪枝可能过度过滤；需要服务器参与原型聚合，通信量略增。

---

## 232. Exploring the relationship between human-centric AI and firm idiosyncratic risks

**arXiv ID:** 2606.24224 | [PDF](https://arxiv.org/pdf/2606.24224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 233. The 2D Ray Tracing Problem using ABCD Lenses and Mirrors is Turing Complete

**arXiv ID:** 2606.24218 | [PDF](https://arxiv.org/pdf/2606.24218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 234. Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations

**arXiv ID:** 2606.24213 | [PDF](https://arxiv.org/pdf/2606.24213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 235. Grounding Generative Policies in Physics: Optimization-Guided Diffusion for Robot Control

**arXiv ID:** 2606.24208 | [PDF](https://arxiv.org/pdf/2606.24208v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 236. Zero-Shot Test-Time Canonicalization using Out-of-Distribution Scoring

**arXiv ID:** 2606.24178 | [PDF](https://arxiv.org/pdf/2606.24178v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 237. Managing Task Execution for Unknown Workloads in Batteryless IoT: A Hardware-Agnostic Evaluation

**arXiv ID:** 2606.24340 | [PDF](https://arxiv.org/pdf/2606.24340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 238. Deep Learning Approaches for 3D Medical Scene Completion: From Geometric Modeling to Generative Paradigms

**arXiv ID:** 2606.24180 | [PDF](https://arxiv.org/pdf/2606.24180v1)

**作者:** Afifa Khaled `[一作]` (Universiti Teknologi PETRONAS), Majdy Mohamed Eltayeb Eltahir `[通讯]` (King Khalid University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2016-2026十年内三维场景补全技术的发展与演进，构建了从体素到高斯 splatting 的混合范式分类与研究路线图。

**💡 创新点**

创新点在于系统性回顾与十年演进框架、提出多模态融合与生成渲染共设计的未来议程，并对跨域、实时与安全等挑战进行了前瞻性分析。

**🔧 技术方法**

讨论并对比了体素、点云、隐式神经场、Transformer、扩散模型、3D Gaussian splatting 等表示与编码–解码架构，以及其在不同任务中的应用。

**📊 数据集**

主要使用ShapeNet、ScanNet、SUN RGB‑D、Matterport3D等公开数据集进行实验与对比。

**📈 对比分析**

采用Chamfer、IoU、F‑score 等指标评估模型性能，并在速度、精度、稀疏鲁棒性等维度提供对比表格，展示不同范式在不同任务下的优势与局限。

**⚠️ 局限性**

存在计算开销大（尤其是扩散模型采样）、实时部署受限、跨域泛化不足、评价指标单一导致结果偏差，以及对高质量表面重建与实时渲染的兼顾难题。

---

## 239. REDI-Match: Rotation-Equivariant Distillation for Efficient and Robust Dense Matching

**arXiv ID:** 2606.24330 | [PDF](https://arxiv.org/pdf/2606.24330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. PETRA: Transforming Web Text for Petroleum-Engineering Domain Adaptation

**arXiv ID:** 2606.24346 | [PDF](https://arxiv.org/pdf/2606.24346v1)

**作者:** Kirill Dubovikov `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salem Lahlou `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 PETRA 数据集与处理管道，将公共 Web 文本自动清洗、过滤并转化为针对石油工程领域的检索训练数据，覆盖第一阶段检索和重排序两个阶段。

**💡 创新点**

创新点包括：①大规模高召回的能量域过滤与 98.4% 准确率的能量分类器；②利用 LLM 生成的查询与硬负例，结合检索挖掘的教师分数候选列表，构建 1.36M 片段、约 2B token 的检索训练集；③通过 score fusion 与 TIES 合并实现领域专属与通用检索的平衡。

**🔧 技术方法**

技术手段：Ray+NeMo Curator 语料清洗；LoRA 轻量级适配器；InfoNCE 对比损失与点对点 BCE 训练；Qwen3、Llama-3.1-8B、Mistral-3-675B 等 LLM 用于查询生成、硬负例生成和教师分数蒸馏；TIES 合并算法实现多任务平衡。

**📊 数据集**

数据集来源：FinePDFs-Edu、peS2o 科学论文、英文 Wikipedia 石油工程切片；清洗后得到 1.36M 片段（≈2B token），随后生成 859k 对比训练行、约 400k 重排序候选行；公开版本已发布在 HuggingFace。

**📈 对比分析**

在内部 SOP 基准上，第一阶段检索 nDCG 从 0.703 提升至 0.763；重排序层在 Earth Science 上从 0.302 提升至 0.436（+44%），推理面板提升 23%；在 OOD 任务（SciFact、FiQA、NFCorpus）保持或略优。与 MS MARCO、基线 Qwen3 模型对比，PETRA 在领域内显著提升，同时通过 score fusion 或 TIES 维持通用性能。

**⚠️ 局限性**

局限性：内部 SOP 评测不可公开复现；数据仅覆盖传统油气工程，新能源等领域覆盖有限；模型仍处于用户测试阶段，无在线 A/B；过滤与标签依赖 LLM，可能带偏；score fusion 需要双倍编码器计算，未评估延迟与成本。

---

## 241. ActiveScope: Actively Seeking and Correcting Perception for MLLMs

**arXiv ID:** 2606.24292 | [PDF](https://arxiv.org/pdf/2606.24292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 242. What Does ODRL Mean? A Cross-Level Ontological Grounding of Permissions, Prohibitions, and Duties in UFO-L

**arXiv ID:** 2606.24344 | [PDF](https://arxiv.org/pdf/2606.24344v1)

**作者:** Daham M. Mustafa `[一作]` (Fraunhofer FIT), Stefan Decker `[通讯]` (Fraunhofer FIT)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将 ODRL 策略语言的评估结果与其产生的法律位姿（权利、义务、权限等）进行对齐，提出跨层设计原则，并基于 UFO‑L 对 ODRL 进行形式化语义建模。

**💡 创新点**

创新点在于引入跨层设计原则，证明任何可被违反且有后果的规范必需同时包含行为层和能力层的法律位姿，并将 ODRL 的八种法律位姿显式化并在 OWL 配置文件中表达。

**🔧 技术方法**

主要技术包括使用 UFO‑L 基础本体、第一阶逻辑公理化建模、Isabelle/HOL、Vampire、E、Z3 证明器进行形式化验证，以及 OWL Profile 生成。

**📊 数据集**

使用了公开的 ODRL 规范 benchmark（39 个问题，36 个标识符，来源于 GitHub 上的 odrl-ufol-grounding 仓库）以及示例数据空间实体。

**📈 对比分析**

通过将公理化模型转化为 TPTP/FOF、SMT‑LIB2 以及 Turtle 策略文件，在 39 个问题上进行一致性与蕴含检验，证明所有公理可满足且问题全部可证明，表明模型完整且验证效率良好。

**⚠️ 局限性**

局限在于只覆盖了简单的法律关系（权限、禁止、义务、补救），未处理更复杂的自由关系；OWL 近似导致部分公理无法直接表示；现有评估器仍未实现完整的补救义务执行。

---

## 243. Transformer-Based Language Models Across Domain Verticals: Architectures, Applications and Critical Assessment

**arXiv ID:** 2606.24331 | [PDF](https://arxiv.org/pdf/2606.24331v1)

**作者:** Guruprakash J `[一作]` (VIT AP University), Krithika L. B `[通讯]` (VIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述并系统化了Transformer语言模型的发展、架构与应用，构建工作型分类法并评估后2023年关键技术与旗舰模型，涵盖医疗、金融、法律、教育、客服、创意与科研等七大行业的实践案例；

**💡 创新点**

提出基于结构属性而非发布时间/规模的Transformer分类法，整合最新的指令调优、RLHF、DPO、检索增强、Mixture-of-Experts等技术，系统评估其在部署中的计算、能耗、可解释性与安全性，并给出开放研究方向；

**🔧 技术方法**

利用Transformer基础架构（Encoder-only、Decoder-only、Encoder-Decoder、Long-Context、Permutation-based、Generator–Discriminator、Mixture-of-Experts）与后续技术（instruction tuning、RLHF/DPO、retrieval augmentation、LoRA/QLoRA、FlashAttention等），以及公开旗舰模型（GPT‑4、Claude‑3、Gemini‑1.5、Llama‑3、Mixtral‑8x7B、DeepSeek‑V3）进行对比；

**📊 数据集**

引用公开的基准数据集（GLUE、SuperGLUE、MMLU、HumanEval、GSM8K、HellaSwag、BIG‑Bench等）以及行业专项数据（Med‑BERT、BioBERT、FinBERT、Legal判例库、教育测评数据等）进行评估；

**📈 对比分析**

通过表格和量化指标（参数规模、训练/推理成本、能耗、可解释性、对齐难度）对不同架构进行多维度比较；结果表明Encoder‑only和Encoder‑Decoder在有限输入、监督任务上更具成本效益，而Decoder‑only在开放式生成任务中表现突出，但对齐与安全挑战更大；Mixture‑of‑Experts可在保持低推理成本的同时提升容量；

**⚠️ 局限性**

局限性包括：缺乏对长上下文真实记忆的评估、基准数据可能受污染导致指标失真、对齐与安全性仍未实现可审计的机制、不同领域的细粒度评估不足、开放模型在规模与性能上仍受制于训练成本与数据质量；

---

## 244. MotifGen: Spatiotemporal interpolation of misaligned satellite images via multi-source generative modeling, in an application to tropical cyclones

**arXiv ID:** 2606.24263 | [PDF](https://arxiv.org/pdf/2606.24263v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 245. Securing LLM-Agent Long-Term Memory Against Poisoning: Non-Malleable, Origin-Bound Authority with Machine-Checked Guarantees

**arXiv ID:** 2606.24322 | [PDF](https://arxiv.org/pdf/2606.24322v1)

**作者:** Yedidel Louck `[一作]` `[通讯]` (Ariel University), Yedidel Louck (Ariel University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对LLM代理长期记忆中的投毒漏洞，提出一种写时绑定来源、非可篡改的记忆权威机制，并在此基础上实现了协作式证明的可审计决策链。

**💡 创新点**

创新点在于将信息流控制的非可篡改性（non‑malleability）与来源绑定（origin‑bound authority）结合，并通过Sybil‑resistant corroboration‑gated elevation实现安全升权；此外对分离定理进行了机理化验证，首次为记忆防护提供机器检验的形式化证明。

**🔧 技术方法**

核心技术包括：写时来源绑定、内容与衍生边缘非可篡改传播、独立可信方协作的升权门控、可审计的哈希链日志，以及基于TLA+的模型化与TLC/Lean自动化验证。

**📊 数据集**

采用 MEM‑INV‑Bench 交叉模型基准，覆盖 8 款主流 LLM（OpenAI、Anthropic、Google、Meta、DeepSeek、Alibaba 等）和 12 个业务域，构造多种攻击场景（sleeper、control‑flow、exfiltration 等）。

**📈 对比分析**

在统一 benchmark 与跨模型触发样式实验中，所提方案在 8 个模型上实现 0 % 的攻击成功率，同时保持 100 % 的合法行动率；决策延迟仅约 1.3 µs，显著低于内容检测类基线；在多轮对话和 Mem0 生产存储场景中同样保持零攻击。

**⚠️ 局限性**

局限性包括：仍无法阻止非后果性答案偏见；依赖写时来源标签的准确性和可信方独立性；缺乏细粒度价值级 taint；证明仅覆盖有限模型与参数，未实现完全无限长执行的机械化证明；以及在协作方不足时需用户手动确认的额外交互成本。

---

## 246. Automatic Part-of-Speech Tagging of Arabic-English Dictionary Senses through WordNet

**arXiv ID:** 2606.24359 | [PDF](https://arxiv.org/pdf/2606.24359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 247. UniRED: Unified RGB-D Video Frame Interpolation with Event Guidance

**arXiv ID:** 2606.24282 | [PDF](https://arxiv.org/pdf/2606.24282v1)

**作者:** Yinuo Zhang `[一作]` (Shandong University), Yiran Shen `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种统一的 RGB‑D 视频插帧框架，联合利用 RGB、深度与事件三模态信息实现中间帧合成。

**💡 创新点**

创新点包括：①统一处理 RGB 与深度插帧，打破以往仅关注单模态的局限；②将事件作为跨模态桥梁，在特征融合中实现 RGB 与深度的互补；③提出运动基准修正（Motion Basis Refinement）和 Z‑轴修正（Z‑axial Refinement）机制，提高光流估计与深度一致性；④构建并公开了新的同步 RGB‑D‑事件数据集 SyncRDE‑60。

**🔧 技术方法**

采用三模态特征提取编码器、事件引导的多尺度融合模块、双向光流迭代估计（含运动基准与 Z‑轴修正）、软混合渲染以及 Charbonnier、LPIPS、梯度一致性等损失函数。

**📊 数据集**

使用了两套数据集：已对齐的 VECtor RGB‑D‑事件版本以及新收集的 SyncRDE‑60 RGB‑D‑事件数据集。

**📈 对比分析**

与 SuperSloMo、RIFE（RGB 仅）、TimeLens、CBMNet、TimeLens‑XL（RGB+事件）以及 RIFE+depth、点云插值（PointINet、FastPCI）和 TimeLens‑D+E（深度+事件）进行对比；UniRED 在 PSNR、SSIM、LPIPS、RMSE、AbsRel、δ1.25 等指标上均实现最优或接近最优，尤其在 ×8 级别插值时优势更为明显。

**⚠️ 局限性**

局限性：对事件的同步与校准要求较高，极端快速运动或光照变化下仍可能出现残留误差；模型参数量和推理时延相对较大；对高噪声事件的鲁棒性尚需提升。

---

## 248. How to~Peel Fully Convex Digital Sets

**arXiv ID:** 2606.24276 | [PDF](https://arxiv.org/pdf/2606.24276v1)

**作者:** Fabien Feschet `[一作]` (Université Clermont Auvergne), Jacques-Olivier Lachaud `[通讯]` (Université Savoie Mont Blanc)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

本文研究了数字空间中完全凸（full convex）集合的“剥离”（peeling）操作，提出了剥离点（peelable point）的定义并给出了判定条件，证明任意完全凸集合至少有一个可剥离点且可逐点剥离至空集；同时给出在二维中局部可判定的判定方法，并讨论了三维及更高维的局部不可判定性。

**💡 创新点**

创新点在于①首次将完全凸集合的层次结构与剥离操作关联，②通过投影和 P‑convexity 定义了剥离点的判定条件，③在二维下提供了局部可判定的剥离判定算法，并证明任何完全凸集合的边界面必包含可剥离点。

**🔧 技术方法**

主要技术包括完全凸性与 P‑convexity 的等价性、凸包与投影的几何分析、面/边界点的局部判定、以及剥离点可行性条件的递归证明。

**📊 数据集**

该工作为理论性研究，未使用具体实验数据集，而是以离散几何集合为对象进行形式化推导与证明。

**📈 对比分析**

由于未提出实测算法实现，论文未给出与现有方法的定量比较或性能评估，仅通过理论证明展示剥离方法的可行性。

**⚠️ 局限性**

局限性主要包括①在三维及更高维度下剥离判定不再是局部可判定；②剥离过程中需频繁更新凸包，计算复杂度尚未评估；③论文未给出具体实现与实验验证。

---

## 249. Structural Kolmogorov-Arnold Convolutions: Learnable Function on the Values or the Filter Shape as Parameter-Efficient Alternative to Per-Edge Convolutional KANs

**arXiv ID:** 2606.24371 | [PDF](https://arxiv.org/pdf/2606.24371v1)

**作者:** Stefano Mereu `[一作]` (Istituto Italiano di Tecnologia), Ferdinando Cannella `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了结构化的 Kolmogorov–Arnold 网络（KAN）卷积模块，改进了传统每条边使用独立单变量函数的方式。

**💡 创新点**

创新点在于将可学习的单变量函数放置在卷积结构的不同位置（滤波器形状或共享值路径）并使用 Morlet 小波基及内容自适应振幅，既保持了表达力又显著降低了参数量。

**🔧 技术方法**

所用技术包括共享单一可学习激活函数、内容自适应高斯门、基于 Morlet 小波的斜坡函数构建滤波器形状、以及对齐的线性投影与批归一化。

**📊 数据集**

实验数据集主要为 CIFAR‑10 和 CIFAR‑100，采用固定的四层卷积骨干网络进行训练与评估。

**📈 对比分析**

比较方法为在相同参数预算下对齐训练设置，结果显示 RF‑KAN 在 CIFAR‑10 上达到 88.47% 及在 CIFAR‑100 上 64.40%，均超过传统 per‑edge KAN 及相同规模的普通卷积，且参数量仅约 0.4 M。

**⚠️ 局限性**

局限性在于仅验证了小规模网络与两个小型数据集，缺乏对更深网络、ImageNet 级别数据以及 FLOPs/推理延迟等实际部署指标的系统评估。

---

## 250. ZONOS2 Technical Report

**arXiv ID:** 2606.24320 | [PDF](https://arxiv.org/pdf/2606.24320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 251. SlipSense: Multimodal Sensing for Online Slip Detection in Legged Robots

**arXiv ID:** 2606.24350 | [PDF](https://arxiv.org/pdf/2606.24350v1)

**作者:** Iris Szu-Yao Liu `[一作]` (Nanyang Technological University), Meng Yee Michael Chuah `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在四足机器人上开发并部署了SlipSense框架，利用自制轻量级多模态足部传感器与LSTM学习模型实现实时抓地力推断和滑移异常检测。

**💡 创新点**

首次将多模态力传感与自监督一类异常检测相结合，提出轻量化压力+IMU传感器、基于LSTM的力推断以及一类学习方法，显著提升滑移检测灵敏度和分辨率。

**🔧 技术方法**

多模态足部传感器（压阻+IMU）、LSTM力推断网络、LSTM自编码器异常检测、基于99.5%分位数阈值的滑移判别、运动捕捉与力板标注。

**📊 数据集**

来自CNC机床的2.3M条力测量数据（ATI F/T）用于训练力推断；稳定抓地数据（多速度、多方向、多地形）用于自监督学习；测试数据通过Qualisys运动捕捉与Kistler力板获取。

**📈 对比分析**

与基于世界坐标脚部速度阈值的运动学基线对比，SlipSense准确率达85.9%（基线69.3%），最小可检测滑移位移24.1 mm±6.4 mm（基线80.8 mm±35.6 mm），分辨率提升3.3倍，假阳性与假阴性显著下降。

**⚠️ 局限性**

仅验证于刚性平坦地面与跑步步态；对不规则地形或倾斜地面泛化有限；二分类标注在阈值附近模糊；需进一步改进切向力估计与多足协同，未来加入地面摩擦估计与步态自适应。

---

## 252. UniTranslator: A Unified Multi-modal Framework for End-to-end In-Image Machine Translation

**arXiv ID:** 2606.24333 | [PDF](https://arxiv.org/pdf/2606.24333v1)

**作者:** Jiahao Lyu `[一作]` (Chinese Academy of Science), Jian Luan `[通讯]` (Xiaomi Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 UniTranslator，一种统一的多模态框架，能够在图像中翻译文本并直接在原图像中写回翻译结果。

**💡 创新点**

主要创新包括：了解‑生成对齐模块（UGAM）消除语义冲突；空间掩码解码器（SMD）实现像素级位置监督；以及两阶段训练策略实现理解与生成的互补优化。

**🔧 技术方法**

使用 Qwen2.5‑VL 视觉‑语言编码器、MMDiT 扩散生成器，并通过交叉注意力、UGAM、SMD 与 LoRA 微调共同训练，实现端到端的翻译与图像生成。

**📊 数据集**

主要数据集为 Translatotron‑V、IIMT30k 与 PRIM 等合成与真实场景的多语言 IIMT 基准。

**📈 对比分析**

与现有统一模型（如 GPT‑Image‑1、Nano‑Banana、Qwen‑Image）及专家级管道（OCR+MT+T2I、TIT+T2I）相比，UniTranslator 在 BLEU、Structure‑BLEU、SSIM、FID 等指标上显著领先，尤其在跨语言和复杂布局场景中表现最优。

**⚠️ 局限性**

仍存在对极端字体风格（如涂鸦、霓虹灯）和极度复杂背景的适配不足，模型在极端情况下可能出现字体失真或背景改动。

---

## 253. Prague Dependency Treebank -- Consolidated 2.0: Enriching a Complex Annotation Scheme

**arXiv ID:** 2606.24324 | [PDF](https://arxiv.org/pdf/2606.24324v1)

**作者:** Marie Mikulová `[一作]`, Jan Hajič `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

发布了PDT‑C 2.0，包含写作、翻译、口语、用户生成四类文本的近400万词，全部手工标注多层（形态、表面句法、深层句法与语义）数据。

**💡 创新点**

首次实现完全手工注释并构建多层互联标注架构，同时集成了形态词典 MorfFlex 和谓词词典 PDT‑Vallex，形成统一、兼容的语言资源。

**🔧 技术方法**

采用 MorphDiTa、UDPipe、PERIN 等工具进行词形还原、句法解析与语义图构建，并将数据转换为 UD、MRP 等多种格式。

**📊 数据集**

使用写作、翻译、口语、用户生成四类文本，并结合 PCEDT、PDT‑Faust 等平行语料，形成完整的多体裁语料库。

**📈 对比分析**

在 MorphDiTa、UDPipe 和 PERIN 上对比评测，词形还原准确率≈98%，句法解析 UAS≈93–98%，语义图 MRP 分数≈93–98%；相较旧版 PDT‑C 1.0，性能显著提升。

**⚠️ 局限性**

局限在于部分标注（grammatemes、桥接关系、多词表达）仅在部分体裁可用，语料单语且评估仅覆盖 MRP 格式，需要开发更全面的评估指标。

---

## 254. LemonHarness Technical Report

**arXiv ID:** 2606.24311 | [PDF](https://arxiv.org/pdf/2606.24311v1)

**作者:** Kailong Ren `[一作]` (AI Lab Lenovo CTO Org), Jianping Fan `[通讯]` (AI Lab Lenovo CTO Org)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LemonHarness，一种集成执行框架，明确工作空间边界、引入可重用规则知识并实现时钟感知执行，以提升长周期 LLM 代理的稳定性与准确率。

**💡 创新点**

核心创新在于：①将模型推理、工具调用、任务知识、执行日志统一管理，形成完整的运行时边界；②将通用执行规则与验收标准化为运行时先验；③采用阶段化的时间感知机制，让模型根据已耗时与剩余预算动态调整探索、实现与验证的重点。

**🔧 技术方法**

采用结构化工具接口、可扩展规则知识库、时间阈值调度和动态日志记录；模型侧使用 GPT‑5.3‑CodeX / GPT‑5.5（与 gemini‑3.1‑pro‑preview 结合）。

**📊 数据集**

主要数据集为 Terminal‑Bench 2.0（445 试验）与 2.1（267 试验），涵盖多种终端任务。

**📈 对比分析**

与传统工具调用框架对比，LemonHarness 在 5 个工作中平均准确率从 84.49% 提升至 86.52%，在 2.1 版上进一步提升至 91.76%，显示出更高的成功率与更低的异常次数。

**⚠️ 局限性**

目前仅在 GPT‑style 模型上验证，缺乏跨模型族的泛化测试；对不同模型的工具调用格式、上下文保持与错误恢复机制的适配仍需进一步探索。

---

## 255. Training-free Cross-domain Few-shot Segmentation via Robust Semantic Representation and Matching

**arXiv ID:** 2606.24297 | [PDF](https://arxiv.org/pdf/2606.24297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 256. AVOC: Enhancing Hour-Level Audio-Video Understanding in Omni-Modal LLMs via Retrieval-Inspired Token Compression

**arXiv ID:** 2606.24286 | [PDF](https://arxiv.org/pdf/2606.24286v1)

**作者:** Yijing Chen `[一作]` (Renmin University of China), Ruihua Song `[通讯]` (Renmin University of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AVOC 框架，在 Omni‑Modal LLM 中加入可学习的 token 压缩模块，实现小时级音视频理解。

**💡 创新点**

创新点在于把多模态 token 压缩重新表述为 Top‑K 检索问题，并结合信息检索中的相关性、重要性与多样性三个标准，推出文本引导交叉注意力、双向视频‑音频交叉注意力和时序感知最大边际相关（TA‑MMR）三种机制。

**🔧 技术方法**

采用可学习投影、交叉注意力评分、Gumbel‑Softmax 可微 Top‑K 选择、TA‑MMR 重新排序，并在 MiniCPM‑o 4.5 LLM 基础上实现训练与推理。

**📊 数据集**

训练数据集包括 AVSD、How2、FineVideo、ChronusAV、LongVILA_sft；评估数据集为 WorldSense、OmniVideoBench、LVOmniBench，以及自制的 Audio‑Video Needle‑in‑a‑Haystack（AV‑NIAH）。

**📈 对比分析**

与 VideoLLaMA2、Qwen2.5‑Omni、OmniZip 等同规模基线进行对比，AVOC 在 OmniVideoBench、LVOmniBench 上平均提升 1.7–7.2 分，最高 5.5 分；在 AV‑NIAH 任务中，能保持 1 小时以内的高检索准确率。

**⚠️ 局限性**

局限性包括对固定 token 预算的依赖、对 λ 与 W 等超参数的敏感性、在极长时长（>1 小时）或极少见事件时性能略降，以及额外的压缩模块导致的计算与内存开销。

---

## 257. When Helpfulness Overrides Causal Caution: Context-Dependent Suppression and Recovery in LLMs

**arXiv ID:** 2606.24370 | [PDF](https://arxiv.org/pdf/2606.24370v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 258. Tractable Reasoning and Conjunctive Query Answering for Defeasible DL-Lite under Rational Closure

**arXiv ID:** 2606.24279 | [PDF](https://arxiv.org/pdf/2606.24279v1)

**作者:** Giovanni Casini `[一作]` (Istituto di Scienza e Tecnologie dell'Informazione, CNR - ISTI), Umberto Straccia `[通讯]` (Istituto di Scienza e Tecnologie dell'Informazione, CNR - ISTI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

在ELH与ELHI描述逻辑中提出了基于理性闭包的可缺失（defeasible）描述逻辑框架，并给出了实例检查与查询回答的决策与重写算法；

**💡 创新点**

创新点在于：①为ELH/ELHI定义了理性闭包语义和δ/ n‑转换，②证明了ABox在此框架下的最小模型唯一；③将查询重写技术与理性闭包结合，形成带秩标注的查询重写；

**🔧 技术方法**

使用了排名模型、负闭包推理、δ/ n‑转换、查询重写（Query Reformulation）以及SQL查询执行等技术；

**📊 数据集**

未使用具体实验数据集，主要以理论分析与示例证明为主；

**📈 对比分析**

通过与经典ELH/ELHI查询回答方法对比，展示了决策算法与经典保持相同复杂度，查询答案与经典一致；

**⚠️ 局限性**

局限在于仅适用于ELH/ELHI两种低表达度DL，难以直接推广到更复杂的描述逻辑，且查询重写过程可能产生指数级查询数。

---

## 259. Pigeonholing: Bad prompts hurt models to collapse and make mistakes

**arXiv ID:** 2606.24267 | [PDF](https://arxiv.org/pdf/2606.24267v1)

**作者:** Hyunji Nam `[一作]` (Stanford University), Natasha Jaques `[通讯]` (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型在多轮交互中因错误上下文导致的“鸽巢化”现象，并提出了通过合成错误数据增强RLVR和DPO训练来缓解此问题。

**💡 创新点**

创新点在于将错误上下文导致的错误重复、模式崩塌与立场倒转统一为鸽巢化框架，并通过在训练阶段注入人工合成错误提升模型对错误上下文的鲁棒性。

**🔧 技术方法**

技术上使用了RLVR（PPO）和DPO对模型进行强化学习微调，并在训练语料中插入合成错误示例，实验中还对多种模型做了多轮推理评估。

**📊 数据集**

数据集覆盖广泛：编程任务采用Live Code Bench、LiveBench；开放式生成任务使用PRISM、Infinity Chat 100、Community Alignment；推理任务使用ARC、MMLU‑PRO、GPQA‑D、Theory of Mind Bench、SocialIQA、Motive Bench；并对10种商用与开源模型进行评测。

**📈 对比分析**

对比方法包括基线零样本推理、无错误上下文、以及无合成错误的RLVR/DPO。结果显示错误上下文会导致38–40%准确率下降、模式崩塌与立场倒转；在合成错误训练后，模型在面对错误上下文时的准确率提升43–60%（RLVR）与34–35%（DPO）。

**⚠️ 局限性**

局限性包括：仅在面对外部错误上下文时提升，无法有效解决模型自身多轮自我纠错；对高度依赖自身推理的任务仍显弱；未来仍需更广泛的抗上下文污染训练策略。

---

## 260. Architecting Hybrid Quantum-Classical Software Systems: Exploration of the Design Trade-off Space with Quantitative Guarantees

**arXiv ID:** 2606.24260 | [PDF](https://arxiv.org/pdf/2606.24260v1)

**作者:** Álvaro M. Aparicio-Morales `[一作]` (Universidad de Extremadura), Juan M. Murillo `[通讯]` (Universidad de Extremadura)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一套基于正式架构风格和概率模型检测的自动化方法，用于生成、评估和选择量子‑经典混合服务系统的配置方案；

**💡 创新点**

创新点在于将量子硬件特性（如QPU资源、CLOPS、量子错误率）融入服务级联约束，并通过 HaiQ+M‑PCTL 进行全空间定量分析，首次实现了在不确定性条件下给出可量化性能、成本、可靠性和量子误差的决策边界；

**🔧 技术方法**

采用 HaiQ（融合 Alloy 语义的结构建模）与 PRISM（M‑PCTL 量化验证）进行配置生成与分析，并使用加权效用函数对多目标指标进行排名；

**📊 数据集**

实验数据来自两个案例：1）Hybrid Search Application（3 个传感器服务、1 个聚合器、1 个搜索服务、1 个结果处理服务），以及 2）Hybrid Weather Forecast System（两个天气站、数据处理、数据库、预报模型启动、结果处理等），使用 IBM 和 IQM 的 NISQ 量子计算机及若干云主机；

**📈 对比分析**

与传统单目标或启发式搜索方法相比，本方法通过全空间枚举和量化验证实现了约 7680 / 3456 种配置的完整评估，分析时间均约 4 小时，且能生成清晰的决策边界和效用排名，明显优于仅基于经验或抽样的手段；

**⚠️ 局限性**

局限性包括：仅考虑门型量子计算机；未模拟量子排队延迟、可用性窗口和量子硬件多样化的真实经济模型；以及在大规模服务/机器组合下可能出现的组合爆炸问题。

---

## 261. Open-Vocabulary BEV Segmentation with 3D-Aware Geometric Constraints

**arXiv ID:** 2606.24353 | [PDF](https://arxiv.org/pdf/2606.24353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 262. A Step Towards Inherently Interpretable Causal Machine Learning Models For Decision Support

**arXiv ID:** 2606.24348 | [PDF](https://arxiv.org/pdf/2606.24348v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 263. Statistical validation and full-sphere extension of a Bayesian model for human static sound localisation

**arXiv ID:** 2606.24367 | [PDF](https://arxiv.org/pdf/2606.24367v1)

**作者:** Roberto Barumerli `[一作]` (Imperial College London), Michele Geronazzo `[通讯]` (Imperial College London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文在已有的贝叶斯声源定位模型基础上，提出并验证了显式似然函数，并用该方法对33名受试者的静态定位数据进行最大似然估计。

**💡 创新点**

创新点在于将贝叶斯模型的似然估计和模型比较正式化，引入完整的Bayesian工作流，并通过比较不同HRTF插值方法解决模型对模板生成的依赖问题。

**🔧 技术方法**

使用的技术包括Bayesian最大似然优化、Monte Carlo积分、BIC信息准则、正交球面插值（SH、双步、Barycentric）以及Python实现的FrAMBI框架。

**📊 数据集**

使用的数据集为SONICOM测量的KEMAR HRTF和33名受试者在三次实验中收集的静态定位行为数据。

**📈 对比分析**

通过后验预测检查和BIC比较评估不同插值方法，结果显示全球面覆盖且高频谱保留的SHMAX和Barycentric方法在大多数受试者上优于原始方法，性能提升显著。

**⚠️ 局限性**

主要局限包括参数识别在垂直方向上的不确定性、对运动噪声的等向性假设、对模板质量与先验宽度的耦合，以及不同实验重复数导致的估计不一致。

---

## 264. TrOCR for Medieval HTR: A Systematic Ablation Study with Cross-Dataset Validation

**arXiv ID:** 2606.24302 | [PDF](https://arxiv.org/pdf/2606.24302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 265. Real-Time Interactive Music Generation via Data-Free Streaming Consistency Distillation

**arXiv ID:** 2606.24307 | [PDF](https://arxiv.org/pdf/2606.24307v1)

**作者:** Baisen Wang `[一作]` (DynamiX), Qisong Han `[通讯]` (Cardiff University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一套数据无关、基于流式一致性蒸馏的实时音乐生成框架，使文本到音乐模型能够以单步推理的方式在不间断的流式语义控制下即时生成高质量音频。

**💡 创新点**

创新点包括：①在流式潜在空间进行chunk‑wise一致性蒸馏，利用冻结教师在线生成的轨迹实现数据无关监督；②提出音乐感知一致性损失（潜在、频谱、时间差分三项），在极端采样压缩下保持音色、瞬态和节奏稳定；③通过KV缓存实现持续上下文传递，让模型在实时交互中保持连续性并可即时响应操作者的语义指令。

**🔧 技术方法**

使用的技术包括：ACE‑Step 1.5 XL‑Turbo教师模型、LoRA 适配的DiT解码器、单步一致性蒸馏、实时潜在流式推理、频谱和时间差分正则化、基于块的异步解码、以及控制向量对文本嵌入的线性插值。

**📊 数据集**

数据集：125,446 条自然语言音乐描述用于训练（无音频对齐）；评测使用 SongDescriber 基准（279 条乐器项目）和自构造的 30 条 Prompt 组合进行延迟和质量评估；主观评测使用 20 名参与者的 MOS 评分。

**📈 对比分析**

比较方法：将流式一阶学生与 8 步非流式教师、8 步非流式学生、以及不使用音乐感知损失的流式学生进行对比。实验显示：流式一阶学生在启动延迟（0.086 s）和 RTF（0.009）上显著优于对照组；在 CLAP、PaSST‑KLD、OpenL3‑FD 等客观指标上与教师相差不大，并在主观 MOS 上达到了与教师相当甚至更好的交互体验。

**⚠️ 局限性**

局限性：①控制粒度仍为短句级别，无法实现细粒度节奏或音色细化；②长时间连续生成时会出现边界抖动，需更长的块长度才能保持音乐连贯；③对超大规模实时演奏场景的可扩展性尚待验证；④接口仍以语义指令为主，缺少低层次的乐器或轨道控制。

---

## 266. MM-TRELLIS: Point-Cloud Guided Multi-Modal 3D Vehicle Generation in Autonomous Driving

**arXiv ID:** 2606.24301 | [PDF](https://arxiv.org/pdf/2606.24301v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 267. 3DCarGen: Scalable 3D Car Generation via 3D-consistent Multi-view Synthesis

**arXiv ID:** 2606.24257 | [PDF](https://arxiv.org/pdf/2606.24257v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 268. Deep numerical schemes for systems of Ergodic BSDEs with applications to regime-switching forward utilities

**arXiv ID:** 2606.24271 | [PDF](https://arxiv.org/pdf/2606.24271v1)

**作者:** Guillaume Broux-Quemerais `[一作]` (Le Mans Université), Wissal Sabbagh `[通讯]` (Le Mans Université)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出两种基于深度学习的数值方案，用于近似求解受市场转移（regime‑switching）控制的耦合耗散性欧几里德BSDE（eBSDE）系统，并将该系统与前向效用（forward utility）在多状态金融市场中的应用相连。

**💡 创新点**

创新点包括：① 将eBSDE与随机终止时间BSDE、欧几里德PDE以及含跳跃的BSDE建立等价性；② 设计了局部可加的深度BSDE算法（LAeBSDE），通过随机终止时间的回归约束实现对ergodic常数的学习；③ 将Deep Galerkin Method（DGM）扩展到多维耦合eBSDE，通过无穷时平均公式估计ergodic成本；④ 在转移市场框架下推导了一致性随机偏微分方程（SPDE）并给出指数、对数和幂级前向效用的eBSDE描述。

**🔧 技术方法**

采用的技术主要有：深度神经网络逼近Y、Z解（或使用自动微分），随机欧拉模拟、随机终止时间取回归法、局部残差聚合损失、DGM残差最小化、对ergodic成本的无穷时平均估计、Adam优化器与梯度下降。

**📊 数据集**

实验数据集：使用一维或多维Ornstein–Uhlenbeck（OU）过程作为随机因子，生成其高斯不变分布进行采样；设置两态或多态转移率矩阵；构造可解析解的幂级和正切（tanh）/有理函数等示例；以及无解析解的两态幂级前向效用模型。

**📈 对比分析**

比较方法：在可解析示例下用L^2(ν)误差评估两种算法对y、z以及ergodic成本的逼近；在无解析解案例下使用PDE残差和正则化误差评估。结果显示：LAeBSDE在误差上略优于DGM（y、z误差≈10^−3–10^−4，成本误差≈10^−4），但训练时间与状态数呈指数增长；DGM训练速度快（约10倍），误差略大（≈10^−2–10^−3），且对状态数的敏感度较低。两种方法在不同状态数、时间尺度、风险厌恶系数等参数下均能准确恢复ergodic常数。

**⚠️ 局限性**

局限性：① LAeBSDE对随机终止时间和时间步长高度敏感，导致训练不稳定且计算量大；② DGM在高维因子或状态数时的误差随维数增加；③ 两种方法均假设因子分布已知或可近似采样；④ 对非光滑或不满足子线性增长的eBSDE解的理论和数值支持有限；⑤ 实际金融数据中因子动力学和转移率估计不确定性未在实验中体现。

---

## 269. Discovery of connectivity-trainability trade-off of IQP Circuits for Hamiltonian Optimization

**arXiv ID:** 2606.24264 | [PDF](https://arxiv.org/pdf/2606.24264v1)

**作者:** Quoc Chuong Nguyen `[一作]` `[通讯]` (Duy Tan University), Quoc Chuong Nguyen (Duy Tan University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

系统性评估IQP电路在Hamiltonian优化中的表现和可训练性，探究不同连通性架构（单Z、循环、全连通）在Ising、MaxCut和Number Partition问题上的优化效率与梯度特性；

**💡 创新点**

揭示连通性、表达力与梯度退化（barren plateau）之间的权衡，发现循环连通在表达力与可训练性之间提供最佳平衡；

**🔧 技术方法**

采用变分量子算法（IQP ansatz）+ Adam梯度优化、蒙特卡洛估计、梯度方差分析、Jensen–Shannon距离评估表达力、双边纠缠熵量化；

**📊 数据集**

使用经典Ising、MaxCut和Number Partition三种基准Hamiltonian，分别随机生成50个实例并用1000次Monte Carlo采样；

**📈 对比分析**

通过比较相对近似比与梯度方差来评估三种连通性：全连通表现最优但梯度迅速衰减；单Z梯度稳定但优化效果差；循环连通在两者间取得平衡；

**⚠️ 局限性**

仅考虑三种连通性、两体相互作用、无噪声仿真、系统规模有限、初始化策略有限，缺乏对更大规模与噪声环境下的验证。

---

## 270. Accelerating Disaggregated RL for Visual Generative LLMs with Diffusion-Based Parallelism and Trainer-Assisted Generation

**arXiv ID:** 2606.24369 | [PDF](https://arxiv.org/pdf/2606.24369v1)

**作者:** Sijie Wang `[一作]` (Harbin Institute of Technology), Shaohuai Shi `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一个面向扩散式视觉生成大语言模型的分离式强化学习系统DigenRL，旨在通过解耦生成器与训练器、细粒度流水线、时间步并行、弹性训练器辅助生成和轨迹一致旧版同步等技术提升训练吞吐量。

**💡 创新点**

创新点包括：①生成轴流水线（GAP）细分生成任务，实现更细粒度的并行；②时间步并行（TSP）在训练阶段并行处理选定的扩散步骤；③弹性训练器辅助生成（TAG）在训练器空闲时主动执行生成任务；④轨迹一致旧版同步（TCSS）允许生成器在权重同步期间使用旧策略，消除同步泡沫。

**🔧 技术方法**

主要技术包括：分离式执行架构、微批次流水线调度、GPU资源动态分配、异步队列管理、FSDP分片、vLLM-Omni推理后端、深度学习框架PyTorch、分布式通信、时间步并行梯度累积、基于奖励模型的强化学习（如FlowGRPO、DanceGRPO）。

**📊 数据集**

实验使用了四个扩散生成模型：HunyuanVideo-13B、Wan2.1-14B、FLUX.1-12B、QwenImage-20B；在三套硬件平台（32GPU+4节点、32GPU+4节点、16GPU+2节点）上评估。

**📈 对比分析**

与现有系统VeRL‑Omni、GenRL以及本地合置执行进行对比；DigenRL在所有测试模型上均实现了1.56×–2.21×的吞吐量提升，并在异构资源环境下表现出1.46×–1.85×的加速；实验还展示了每项技术对速度提升的贡献。

**⚠️ 局限性**

局限性包括：①系统实现复杂，需与vLLM‑Omni、FSDP深度耦合；②TAG与TCSS调度需在线监测和参数调优，搜索开销虽小但不免增加设计复杂度；③对扩散RL之外的AR或流式模型适用性有限；④在极大批量或极低延迟场景下，流水线泡沫仍可能存在。

---

## 271. MorfFlex: Handling Rich Morphology

**arXiv ID:** 2606.24366 | [PDF](https://arxiv.org/pdf/2606.24366v1)

**作者:** Jaroslava Hlaváčová `[一作]`, Jan Hajič `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了 MorfFlex CZ 形态学词典，提出了从源格式→中间格式→基本格式的三层模式化生成流程，并将该词典集成至多种 NLP 工具（MorphoDiTa、UDPipe 等）用于词形分析、生成、词性标注与词干提取。

**💡 创新点**

核心创新是将屈折丰富语言的词形空间压缩为可维护的派生规则（derivational patterns）与屈折规则（inflectional patterns）两级模式，并通过虚拟中间格式实现从单行源记录自动生成上百万词形的高效、可追溯的词典结构。

**🔧 技术方法**

使用基于规则的形态生成与分析技术，配合自定义的词形、词义与词性标签体系；实现脚本（morfflex-generator）将源格式转换为基本格式；利用 MorphDiTa、UDPipe 等开源工具进行评测。

**📊 数据集**

主要数据集为 Prague Dependency Treebank（PDT）– Consolidated 2.0 约 4M 词元、1.7M 手工注释标记；这些语料用于训练与评估形态学标注模型，并作为词典的验证与扩展基础。

**📈 对比分析**

通过与 MorphoDiTa 的基准对比，取得 96.27% 的词性标注 F1、98.31% 的词干提取 F1；在大规模文本上实现 10–200K 词/秒的处理速度；在 UDPipe+Morflex 复核方案中，显著提升了传统深度学习模型的标注准确率。

**⚠️ 局限性**

局限性：方案目前仅适用于屈折语言（主要是斯拉夫语），对黏着语需要重构；规则一旦更改会连锁影响大量词典条目，维护与监测成本较高；极端不规则形态仍需通过“trivial patterns”逐条记录，导致规则库膨胀。

---

## 272. Neural Network-Based Parametric Model Reduction for Predicting Turbulent Flow for Different Vehicle Geometries

**arXiv ID:** 2606.24265 | [PDF](https://arxiv.org/pdf/2606.24265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 273. SignNet-1M: Large-Scale Multilingual Sign Language Video Dataset with Downstream Benchmarks

**arXiv ID:** 2606.24361 | [PDF](https://arxiv.org/pdf/2606.24361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 274. TIGER: Taming Identity, Geometry, and Generative Priors for High-Quality Face Video Restoration

**arXiv ID:** 2606.24336 | [PDF](https://arxiv.org/pdf/2606.24336v1)

**作者:** Yang Zhou `[一作]` (Xiaomi Inc.), Daiguo Zhou `[通讯]` (Xiaomi Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种面部视频恢复框架TIGER，利用身份、几何和生成先验进行一阶流推断，生成高质量、身份一致、时序稳定的视频。

**💡 创新点**

创新点在于三重先验融合：身份先验（ArcFace嵌入）、几何先验（跨源3DMM融合生成视角无关正态图）和生成先验（one‑step rectified flow基于Diffusion Transformer），以及三阶段渐进训练策略。

**🔧 技术方法**

使用技术包括Diffusion Transformer、VAE、ArcFace、3DMM、跨源参数融合、one‑step rectified flow、跨注意力机制、GAN对抗训练等。

**📊 数据集**

使用新构建的28,491帧高质量FVR数据集（从OpenHumanVid、CelebV-HQ、VFHQ筛选），以及VFHQ‑Test和VoxCeleb2等测试集。

**📈 对比分析**

与SOTA方法（DOVE、BasicVSR++、STAR、SeedVR2‑3B、KEEP、BFVR、PGTFormer、SVFR）比较，在VFHQ‑Test和VoxCeleb2上均在PSNR、SSIM、LPIPS、IDS、FVD、LIQE、MUSIQ等指标上实现领先。

**⚠️ 局限性**

限制在于依赖高质量参考图和3D重建，若参考缺失或重建失误会影响结果，对极端降解或低分辨率视频的鲁棒性仍有限。

---

## 275. Prob-BBDM: a Probabilistic Brownian Bridge Diffusion Model for MRI sequence image-to-image translation

**arXiv ID:** 2606.24313 | [PDF](https://arxiv.org/pdf/2606.24313v1)

**作者:** Martin Valls `[一作]` (University of Poitiers), David Helbert `[通讯]` (University of Poitiers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种基于概率布朗桥扩散模型（Prob‑BBDM）的MRI序列翻译框架，可从单个2D轴向切片生成缺失的MR序列，实现快速且高质量的医学图像合成。

**💡 创新点**

创新点在于将布朗桥扩散与变分先验‑后验潜在一致性结合，仅需4步采样即可完成翻译，同时通过KL正则化增强解码稳定性，显著降低计算成本。

**🔧 技术方法**

采用Brownian Bridge Diffusion Model、变分编码器（Prior/Posterior）、2D U‑Net+多头自注意力、KL距离匹配以及极少步采样的扩散过程。

**📊 数据集**

使用BraTS 2021多模态MRI数据集（T1、T2、FLAIR）进行训练与评估，并在外部Gliobiopsy临床数据上做通用性验证。

**📈 对比分析**

与BBDM、MaskGAN、Pix2Pix、ResViT、SelfRDB等基线方法在SSIM、PSNR、Dice、HD95等指标上进行对比，Prob‑BBDM在所有任务中均获得最高SSIM/PSNR，并在下游肿瘤分割中实现更高Dice与更低HD95，表现显著优于基线。

**⚠️ 局限性**

主要局限包括：仅处理2D切片，跨层一致性有限；对流动伪影或极端病灶的生成仍可能失真；模型受训练分布影响，对罕见病变的生成存在偏差；缺乏3D或高分辨率扩展方案。

---

## 276. MVG-KAN: Multi-View Geo-Wind Guided KAN for PM$_{2.5}$ Forecasting

**arXiv ID:** 2606.24347 | [PDF](https://arxiv.org/pdf/2606.24347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 277. SURGELLM: Rethinking Multi-Task Evaluation through Task-Aware Feature Gating with Class-Balanced Normalization

**arXiv ID:** 2606.24259 | [PDF](https://arxiv.org/pdf/2606.24259v1)

**作者:** Noor Islam S. Mohammad `[一作]` (Istanbul Technical University), Ulug Bayazit `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在统一的 Transformer 框架中，通过引入手工挑选的词汇特征门控、任务条件前缀标记和实例加权标准化，解决了多任务 NLP 中的结构偏差、类别不平衡和缺乏外部词汇知识的问题。

**💡 创新点**

提出了可学习的每维特征门、任务特定前缀、Instance‑Weighted Normalization（IWN）以及针对词汇组的理论超风险界限，三者结合实现对异构任务的显著性能提升。

**🔧 技术方法**

利用预训练 Transformer 编码器（BERT、RoBERTa 等）、基于 Sigmoid 的特征门、前缀注入、LayerNorm、IWN、两层 MLP 分类头以及 AdamW 训练策略。

**📊 数据集**

实验数据集包括 SST‑2 句子级情感分类、HotPotQA 多跳检索、LLM‑7 生成文本属性判定、HumLLM 作者身份检测，共 17,830 条样本。

**📈 对比分析**

与单任务基线、T5‑Base 等进行宏 F1 对比，RoBERTa+IWN 在四任务上的平均宏 F1 达 0.940，较最佳非 IWN 基线提升 0.036，作者身份检测提升 0.130；结果通过统计显著性检验，且在速度‑精度曲线上具备良好 Pareto 性能。

**⚠️ 局限性**

仅在英文基线模型上验证，模型规模局限于 base‑size，理论风险界限可能松散，未在完整 GLUE/SuperGLUE 等多任务集合中评估，且在更大规模编码器上的可扩展性待进一步验证。

---

## 278. Ill-Posed by Design: Probing Evidence Use in VLMs

**arXiv ID:** 2606.24335 | [PDF](https://arxiv.org/pdf/2606.24335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. Hierarchical Spatial and Channel Aggregation for Cross-domain Few-shot Segmentation

**arXiv ID:** 2606.24296 | [PDF](https://arxiv.org/pdf/2606.24296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 280. RoBoSR: Structured Scene Representations for Embodied Robotic Reasoning

**arXiv ID:** 2606.24338 | [PDF](https://arxiv.org/pdf/2606.24338v1)

**作者:** Kewei Hu `[一作]`, Hanwen Kang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RoBoSR 结构化场景表示框架，用对象中心的场景图来拆分高层推理与低层执行，实现开放式长周期任务的因果状态转移推理。

**💡 创新点**

创新点在于将场景图作为中间决策状态并明确建模其状态转移；同时通过 Manip‑Cognition‑1.6M 进行联合监督，结合强化学习奖励实现图结构一致性与单步执行约束。

**🔧 技术方法**

使用 Qwen3‑8B + LoRA 进行结构化推理微调；场景图构建基于 SAM3 + 规则式几何关系；低层控制采用 SKIL；强化学习采用 GRPO，奖励包括步长约束、图一致性与终止判定。

**📊 数据集**

核心数据集为 Manip‑Cognition‑1.6M（1.6M 结构化样本）以及 GSR‑bench（180 个长周期任务）进行评估。

**📈 对比分析**

在 GSR‑bench 与真实机器人测试中，RoBoSR‑8B 在语义歧义、空间约束和目标泛化三大维度均显著优于 Gemini‑2.5‑pro、GPT‑5、Claude‑sonnet‑4.5、DeepSeek‑V3 等 LLM 基线；RoBoSR‑8B‑FT 仅用 36 条任务内演示即可提升任务完成率至 60%+，体现样本高效。

**⚠️ 局限性**

局限在于难以处理高度柔性物体和连续状态（如部分开启角度），以及对强感知前端的依赖，导致对感知误差和不确定性鲁棒性不足。

---

## 281. Meet UD_Czech-PDTC: A Large and Genre-Rich Treebank in Universal Dependencies

**arXiv ID:** 2606.24337 | [PDF](https://arxiv.org/pdf/2606.24337v1)

**作者:** Marie Mikulová `[一作]`, Jan Hajič `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文实现了从Prague Dependency Treebank–Consolidated 2.0到Universal Dependencies (UD)的高质量转换，并在转换后训练了新的UDPipe解析器。

**💡 创新点**

创新点包括：① 对Czech特有的语法现象（如定语代词、协同结构、省略等）进行精细映射；② 结合UD的Enhanced图谱实现更完整的语义依赖；③ 通过对四个子语料库的统一训练显著提升了模型在不同文本领域的泛化性能。

**🔧 技术方法**

技术手段主要有：规则映射（POS、形态特征、依存关系）、层级对齐（a-layer与m-layer）、Enhanced UD增强策略、以及基于Transformer的UDPipe模型训练。

**📊 数据集**

使用的数据集是PDT‑C 2.0（3440K词，涵盖新闻、商务新闻、口语、用户生成文本）以及其UD 2.16转换版本。

**📈 对比分析**

通过在完整训练集与仅PDT子集上分别训练UDPipe，并在四个子集的测试集上评估UAS、LAS、词形标注和词形还原的准确率，结果显示完整训练集将形态标注误差降低24%，词形还原误差降低33%，解析误差降低18%。

**⚠️ 局限性**

主要限制是约20%的语料缺失深层语法层（t-layer），导致部分省略、深层语义关系的映射无法利用；此外，转换过程中仍依赖人工规则，可能对极端或新兴语言现象的适配性有限。

---

## 282. CALIBER: Calibrating Confidence Before and After Reasoning in Language Models

**arXiv ID:** 2606.24281 | [PDF](https://arxiv.org/pdf/2606.24281v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 283. The Latent Bridge: A Continuous Slow-Fast Channel for Real-Time Game Agents

**arXiv ID:** 2606.24470 | [PDF](https://arxiv.org/pdf/2606.24470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 284. Minimal additive codes and additive strong blocking sets

**arXiv ID:** 2606.24262 | [PDF](https://arxiv.org/pdf/2606.24262v1)

**作者:** Gianira N. Alfarano `[一作]` (Universite de Rennes), Marine Le Meur `[通讯]` (Universite de Rennes)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了加法码（Additive codes）中最小码（minimal additive codes）的几何与代数性质，提出了加法强阻塞集（additive strong blocking sets）的概念，并建立了它们与最小加法码之间的一一对应关系。

**💡 创新点**

将传统的线性码与强阻塞集的对应关系推广到加法码领域，首次定义并研究了加法强阻塞集；同时给出了改进的总权重公式、Plotkin 型上界和渐近极限，以及构造和存在性结果。

**🔧 技术方法**

运用了有限几何（项目系统、超平面、阻塞集理论）、线性代数（矩阵展开、向量维度）与组合计数（Gaussian 系数、计数论）等技术，并结合了外部强阻塞集（outer strong blocking sets）与量子码的视角。

**📊 数据集**

本工作为纯理论研究，未使用具体数据集；所有结果均来自数学证明与构造。

**📈 对比分析**

通过理论证明给出了长度、距离与维度的上、下界，并在已知线性码的最佳界限之上或相当的区间内实现了最小加法码；在渐近极限上证明了比线性码更优的率-距离取舍。

**⚠️ 局限性**

当前界限（尤其是长度上限）在 h ≥ 2 的情况下仍不够紧凑；构造的加法强阻塞集不一定能进一步压缩长度；需要更强的几何构造和对非faithful加法码的更细粒度分析。

---

## 285. Boosting Text-Driven Video Segmentation via Geometry-Aware Distillation

**arXiv ID:** 2606.24464 | [PDF](https://arxiv.org/pdf/2606.24464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 286. An LLM-based Two-Stage Transformer Framework for Cross-Domain Bearing Fault Diagnosis with Limited Data

**arXiv ID:** 2606.24459 | [PDF](https://arxiv.org/pdf/2606.24459v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 287. FT-WBC: Learning Fault-Tolerant Whole-Body Control for Legged Loco-Manipulation

**arXiv ID:** 2606.24466 | [PDF](https://arxiv.org/pdf/2606.24466v1)

**作者:** Yudong Zhong `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并实现了 FT-WBC 框架，能够在腿关节失效时动态调整全身姿态并生成补偿步态，完成腿式移动与机械臂协同操作。

**💡 创新点**

创新点包括：① 通过故障估计模块（FE）快速识别失效关节；② 通过姿态适配模块（PAM）将关节失效信息映射到安全姿态指令；③ 将两模块集成到分离上下体策略中，实现零拷贝从仿真到真实的移植。

**🔧 技术方法**

使用的技术包括：深度强化学习（PPO）训练分离的上下体策略；多层感知网络（MLP）做故障估计；非线性映射网络做姿态适配；Isaac Gym 进行仿真训练，Jetson Orin Nano 实时推理。

**📊 数据集**

使用自建仿真环境进行随机关节弱化/锁定失效的 1,000 次试验；在真实机器人（Unitree Go2 + Airbot Play）上进行 20 次不同桌面高度（50/75/90 cm）与三种失效模式（部分弱化、完全弱化、锁定）测试。

**📈 对比分析**

与基线 Robo‑Duet 以及去掉 FE 或 PAM 的消融模型对比。FT‑WBC 在各种关节失效场景下显著提升生存率（提升约30–40%）和工作空间（更大可达体积），同时保持更稳定的前向速度跟踪。

**⚠️ 局限性**

局限性：目前仅针对单一关节失效；未处理多关节并发失效；评估集中在静止或先移动再操作的任务，未覆盖在行走时进行动态协作的失效情况。

---

## 288. Detecting AI Coding Agents in Open Source: A Validated Multi-Method Census of 180 Million Repositories

**arXiv ID:** 2606.24429 | [PDF](https://arxiv.org/pdf/2606.24429v1)

**作者:** Arsham Khosravani `[一作]`, Audris Mockus `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在超过1.8亿个 Git 仓库中构建多层检测框架，综合配置文件、提交信息、作者身份与 bot 签名四种迹象，系统性地对 12 款生成式 AI 编码代理进行普查。

**💡 创新点**

首次统一四类 AI 轨迹的检测方法，证明单一信号无法覆盖全局，揭示如 Claude Code 这类 agent 在单一方法下被低估 30 倍，并在多快时间窗口内量化其采纳与活跃度。

**🔧 技术方法**

利用 World of Code 的 ClickHouse 查询、O(1) 哈希映射检索、正则表达式匹配、手工标注与 Wilson 置信区间评估，以及跨快照差分分析技术。

**📊 数据集**

三张 WoC 快照（V2412、V2510、V2604）共计 180M+ 仓库，涵盖 GitHub、GitLab、Bitbucket 等主流平台，并结合 Ecosyste.ms 外部抓取的非 GitHub forge 数据。

**📈 对比分析**

与独立 PR 基准 AIDev 进行对比，发现两渠道捕获几乎互斥；多层检测精度>95%，单方法召回仅 3%，多方法召回提升至约 30×，表明多信号融合显著提高检测覆盖。

**⚠️ 局限性**

局限在于只能检测可观测痕迹，配置文件或提交信息演化导致检测窗口滑移；未覆盖未收录于 WoC 的私有或非主流 forge；缺乏因果效应的基准验证，难以精确评估 AI 代码质量与开发效率的真实影响。

---

## 289. Beyond Logprobs: A Multi-Signal Confidence Engine for LLM-Based Document Field Extraction

**arXiv ID:** 2606.24420 | [PDF](https://arxiv.org/pdf/2606.24420v1)

**作者:** Nitesh Kumar `[一作]` `[通讯]` (Perfios Software Solutions Pvt. Ltd.), Nitesh Kumar (Perfios Software Solutions Pvt. Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于LLM的文档字段提取可靠性引擎，能够判断提取结果是否可用于自动化处理。

**💡 创新点**

核心创新是双调用（Dual‑Call）设计，通过字段导向和文档导向两种不同输出结构来获取互补的置信度信号，并融合多源信息（LLM内部不确定性、OCR置信度、图像质量、空间布局）形成高效二分类器。

**🔧 技术方法**

技术包括GPT‑4o多模态推理、AWS Textract OCR、CatBoost二分类器以及后期的等温度校准（Isotonic）和Lasso逻辑回归校准；同时使用特征工程提取跨调用一致性、图像拉普拉斯锐度、空间中心距等指标。

**📊 数据集**

主要使用DocILE（6,680份真实发票，55字段，26.6%自然失败率）进行训练与评估，并在CORD（800训练+100测试收据，30字段）上进行零样本迁移测试。

**📈 对比分析**

相较于传统的log‑prob、口头置信度和自一致性基线，本文方法在DocILE上实现0.928 ROC‑AUC、AURC仅0.042，零样本迁移到CORD的AUC 0.858，ECE 0.025，显著提升了可靠性与路由性能。

**⚠️ 局限性**

局限包括：需依赖完整OCR流水线；实验仅覆盖英文商业文档；仅测试单一LLM骨干；未验证跨语言或不同文档格式的泛化。

---

## 290. Agentic AI for Bilevel Long-Term Optimization of Policy-Driven Physical Layer Systems

**arXiv ID:** 2606.24416 | [PDF](https://arxiv.org/pdf/2606.24416v1)

**作者:** Bingnan Xiao `[一作]` (Fudan University), Tony Q. S. Quek `[通讯]` (Singapore University of Technology and Design)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出Agentic-LTPO框架，将代理式人工智能与双层优化结合，实现物理层在面对运营商策略不断变化时的长期性能优化；

**💡 创新点**

核心创新包括：①双层嵌套结构将策略解释、环境摘要、历史经验映射为低层配置；②四大LLM代理（解释、观察、规划、批评）配合检索增强生成(RAG)实现安全可靠的配置更新；③在下层采用零逼近束缚结构得到闭式能量最小化解；

**🔧 技术方法**

技术手段：大语言模型+检索增强生成(RAG)、多代理协同推理、规划-批评循环、零逼近束缚beamformer、线性规划求解；

**📊 数据集**

数据集：基于CF‑MIMO系统的仿真数据，随机/分段式运营商策略，用户位置、路径损耗、阴影衰落等，所有实验均在仿真环境中完成；

**📈 对比分析**

比较方法：与静态配置、单一LLM控制器以及去除RAG或批评的消融版本对比，实验表明Agentic‑LTPO相较于静态基线提升约57.2%的累计通信效用，单一代理器效果介于两者之间；

**⚠️ 局限性**

局限性：仅在CF‑MIMO仿真中验证，缺乏真实场景评估；模型推理成本较高，对LLM的令牌消耗敏感；对自然语言语义的准确性仍有一定依赖；

---

## 291. Cycle-Consistent Neural Explanation of Formal Verification Certificates

**arXiv ID:** 2606.24414 | [PDF](https://arxiv.org/pdf/2606.24414v1)

**作者:** Andoni Rodriguez `[一作]` (J.P. Morgan AI Research), Daniel Borrajo `[通讯]` (J.P. Morgan AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种循环一致的神经架构，能够将形式验证证书转换为自然语言解释，并通过符号验证器保证解释的真实性。

**💡 创新点**

核心创新包括：①利用指针-生成网络在生成过程中直接复制证书中的状态名，消除实体幻觉；②通过逆向Transformer与符号验证器构建循环一致性学习，无需人工标注；③在推理时使用混合路由策略，根据证书类别和覆盖率动态选择最佳解码配置，显著提升可靠性。

**🔧 技术方法**

使用的技术包括Transformer encoder‑decoder、pointer‑generator机制、逆向解码器、符号验证器（对事实集合进行比较）、循环一致性损失、混合推理路由、以及可选的LLM流畅性评估。

**📊 数据集**

数据集来自金融合规流程的 5,841 条验证证书（6 种证书类型 × YES/NO 共 12 类别），每条证书对应 4 个模板生成的自然语言解释，共 420 条测试样本。共有 207 个命名状态。

**📈 对比分析**

与 16 种前沿LLM（GPT‑4o、GPT‑5.3、Claude Opus 4.5、Sonnet 4.6）配对的少量提示基线相比，混合路由模型在 420 条测试样本上实现 90.0% 的循环一致性可靠率，超过最佳 LLM 组合的 76.1%（提升 13.9 pp），并且推理速度约 860 倍更快（≈185 ms vs 160 s），成本为零，输出确定。

**⚠️ 局限性**

局限性包括：①不支持需要计数的空强连通分量证书；②模型在证书复杂度（>15 状态、>3 序列）时精度下降；③仅在固定词表与模板化参考上验证，迁移到其他验证器或更大词表时需再训练；④实验仅在 CPU 上完成，模型规模受限；⑤循环一致性提供的是可靠率下界，NN_2 的 3.6% 误报率和潜在的协同适配仍需关注。

---

## 292. RE4: Transformation-aware Imitation of Object Interactions Using Manipulation Modes

**arXiv ID:** 2606.24403 | [PDF](https://arxiv.org/pdf/2606.24403v1)

**作者:** Arsh Chawla `[一作]` (Australian National University), Rahul Shome `[通讯]` (Australian National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为RE4的可解释、轻量化物体交互模仿学习框架，核心四步为检索、重构、再规划和回放，利用对象姿态信息与操控模式感知实现对演示的动态匹配与执行。

**💡 创新点**

创新点在于将操控模式理论与检索+重构流程相结合，并通过自监督姿态估计器从演示数据直接学习对象姿态，从而实现无监督、低成本的对象感知；同时强调在稀疏数据和观测下的鲁棒性。

**🔧 技术方法**

使用的技术包括自监督的对象姿态估计（基于ResNet‑18编码器）、基于对象增强状态距离的检索、相对变换重构、RRT等运动规划以及模式感知的四步循环。

**📊 数据集**

实验数据集涵盖Push‑T（平面二维T块推送任务）和Robomimic（Lift、Can、Square的六自由度抓取任务），每个任务均提供状态与图像两种观测版本。

**📈 对比分析**

通过与Diffusion Policy（DDPM/DDIM）、Streaming Flow Policy、Geometry aware Imitation Policy等基线在相同环境下进行对比，RE4在所有任务和观测模式下均达到或超过最强基线，尤其在图像模式下平均覆盖率最高，并在稀疏观测与稀疏演示的设置中表现更稳健。

**⚠️ 局限性**

局限性包括对姿态估计误差高度敏感，主要针对单物体且仅处理接触型交互，非抓取（non‑prehensile）场景下姿态估计更具挑战；多物体环境的模式枚举与管理尚未覆盖；以及缺乏真实机器人实验验证。

---

## 293. Average Rankings Mask Per-Subject Optimality: A Friedman-Nemenyi Benchmark of EEG Motor-Imagery BCI Decoders

**arXiv ID:** 2606.24394 | [PDF](https://arxiv.org/pdf/2606.24394v1)

**作者:** Xavier Vasques `[一作]` (IBM), Olivier Oullier `[通讯]` (Inclusive Brains)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MOABB框架下对三大公开运动想象EEG数据集进行1,056种解码配置（特征提取器×标准化器×分类器）进行全量内测评；

**💡 创新点**

首次在最宽松的单场次单人内测条件下系统评估不同解码管线是否存在通用最佳解，揭示个体差异导致的解码器无“一刀切”可行性；

**🔧 技术方法**

采用CSP、协方差切线空间投影、连通性切线空间投影、非线性特征四大类特征；标准化器包括Standard、Robust、MinMax和L2；分类器包含逻辑回归、LDA、线性/RBF SVM、随机森林和17种多层感知机；

**📊 数据集**

PhysionetMI（109人）、Cho2017（52人）和Zhou2016（4人）三大公开数据集；

**📈 对比分析**

使用MOABB内部的单场次拆分，评估每条管线的准确率，并采用Friedman、Nemenyi、Wilcoxon等多分类器比较统计方法；结果显示协方差切线空间投影和CSP平均性能最高，但在最异质的PhysionetMI上两者统计无显著差异，且单个人最佳管线仅占约35%；

**⚠️ 局限性**

局限包括：仅做内测评，未涉及跨场次或跨受试者转移；预处理极简，可能影响非线性特征的鲁棒性；窗口短导致某些长记忆特征不稳定；仅用准确率作为评估指标；未包含深度学习或自适应方法的比较。

---

## 294. BRAVR: An AP-Assisted Online DRL Mechanism for Interactive VR Bitrate Adaptation over Wi-Fi

**arXiv ID:** 2606.24389 | [PDF](https://arxiv.org/pdf/2606.24389v1)

**作者:** Miguel Casasnovas `[一作]` (Universitat Pompeu Fabra), Boris Bellalta `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出并实现了 BRAVR，一种基于深度强化学习的分布式 Wi‑Fi VR 比特率自适应系统，能够在线学习并根据应用层指标与 AP 侧网络信息动态调整编码比特率。

**💡 创新点**

创新点包括：
• 将 AP 侧轻量化遥测（MCS、重传率、信道占用等）纳入决策，提升对动态链路状态的感知；
• 采用行动遮蔽（action shielding）在运行时过滤不安全或无效动作，保证实时性和稳定性；
• 设计专门的多目标奖励函数，兼顾视觉质量、延迟、可靠性和多用户 airtime 公平性；
• 在真实物理测试台上直接进行无预训练的在线学习，避免模拟–现实不匹配；
• 与主流启发式算法（NeSt‑VR）和无网络协助版本（BRAVR‑）对比，证明 AP 助推效果。

**🔧 技术方法**

技术手段包括：
- 深度 n‑step Expected SARSA（on‑policy TD）与 Polyak 目标网络；
- 状态由应用层、历史与网络层特征三部分构成；
- 动作空间为 {−1, 0, +1}，表示相邻比特率阶梯；
- 行动遮蔽机制基于 QoS 边界与比特率边界自动屏蔽；
- ALVR 开源平台实现 VR 流媒体；
- OpenWrt‑based AP 提供 HTTP 端点返回 JSON 统计；
- 设备：Meta Quest 2、Dell G15 / Pro Max、ASUS TUF‑AX4200，Wi‑Fi 5 40 MHz。

**📊 数据集**

数据集：本文使用在实验室搭建的物理 Wi‑Fi 测试台收集的原始实验数据，涵盖单用户与双用户、静态与移动、近/远两种位置场景。实验共包含 17 次 240 s 单用户训练/评估周期和 1 次 20 min 双用户训练 + 3 次 240 s 评估。数据公开可通过论文附录下载。

**📈 对比分析**

评估方法：
- 与启发式算法 NeSt‑VR、无 AP 辅助的 BRAVR‑ 以及 CBR（固定 80/70 Mbps）对比；
- 指标包括目标比特率、网络帧成功率 (NFR)、视频帧 RTT (VF‑RTT)、切换率、airtime 占用、packet loss ratio (PLR) 与综合 QoE 相关的 utility；
- 结果显示：单用户场景中 BRAVR+ 在保持 QoS 目标（22 ms VF‑RTT、0.99 NFR）下，平均比特率提升约 21 % ；
- 多用户场景中，BRAVR+ 的 airtime‑aware utility 最高，成功率 >90 % 并抑制 airtime 过度占用；相较于 NeSt‑VR，BRAVR+ 在 near‑near 场景下 QoE 提升约 13 % ；
- 在 near‑far 场景中，BRAVR+ 在 near 用户上通过控制 airtime 进一步提升 QoS，整体 utility 最高。

**⚠️ 局限性**

局限性：
- 仅在单/双用户、Wi‑Fi 5 40 MHz、无重叠 BSS 的受控环境下验证；
- 需要 AP 支持轻量遥测，部署成本与兼容性待进一步评估；
- 奖励权重手工调节，未采用多目标 RL 或自动权重学习；
- 适应大规模用户密度、复杂信道干扰（如邻频段、MIMO）仍需实验验证；
- 对动态网络切换（AP 换手）与跨设备协同控制的鲁棒性未进行深入分析。

---

## 295. AutoSpecNER: A Fine-Grained Named Entity Recognition Dataset for Vehicle Specification Extraction

**arXiv ID:** 2606.24387 | [PDF](https://arxiv.org/pdf/2606.24387v1)

**作者:** Jordan Lee `[一作]` (Manchester Metropolitan University), Matthew Shardlow `[通讯]` (Manchester Metropolitan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了AutoSpecNER数据集并在其上进行车辆广告细粒度命名实体识别基准实验。

**💡 创新点**

首次公开细粒度汽车广告NER数据集，采用双来源（用户与AI生成）文本，系统性比较规则、细调编码器和LLM性能，揭示细调编码器最优。

**🔧 技术方法**

使用规则匹配+正则表达式、Transformer编码器（BERT、RoBERTa、ModernBERT、DeBERTa）以及大语言模型（Gemini、Gemma、Qwen3等）并结合少量提示和自校验。

**📊 数据集**

AutoSpecNER数据集（659条车辆广告，10,000+实体，15标签），包含用户生成和AI生成文本，并附有结构化元数据。

**📈 对比分析**

采用字符级IOB2评估，比较规则基线、细调编码器和LLM；细调编码器最高微F1≈0.901，规则基准≈0.43，LLM最高≈0.778。

**⚠️ 局限性**

数据仅来自单一英国平台，缺乏跨平台、跨语言多样性，且样本来源比例可能影响模型泛化，未覆盖多语种场景。

---

## 296. Legible and Intuitive Multi-modal Robot State and Intent Communication Validated in Online and Real-world Studies

**arXiv ID:** 2606.24445 | [PDF](https://arxiv.org/pdf/2606.24445v1)

**作者:** Tim Schreiter `[一作]` (Technical University of Munich), Achim J. Lilienthal `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对移动非人形机器人在在线和现实环境中，比较低表情带宽的LED单模与高表情带宽的多模（头部、手势、语音）通信策略在五种信息（闲置、请求注意、转向、错误、障碍）下的可读性和直观性。

**💡 创新点**

首次系统地量化并验证在现实情境下的“虚实差距”，并提出混合设计范式：低带宽用于系统状态，高带宽用于空间依赖的意图表达。

**🔧 技术方法**

使用Robotnik RB‑Kairos底座的LED条、NAO机器人配合姿态、目光、手势与语音，配合Pollfish在线调查与德国德意志博物馆现场实验。

**📊 数据集**

数据来源为两组受试者：在线调查97人（经筛选）和现场实验139人；每组分别观察5种信息的不同通信策略，记录识别正确率和直观性评分。

**📈 对比分析**

采用2×2混合设计、两路ANOVA、卡方检验和Mann‑Whitney U检验；结果显示高表情带宽策略在在线与现场均显著提升可读性（legibility ↑）和直观性（intuitiveness ↑），尤其在转向和注意请求上差距更大。

**⚠️ 局限性**

限制包括仅比较两端极端策略，未拆解各模态贡献；实验受限于单一机器人平台与固定情境；可能存在样本偏差与环境噪声影响。

---

## 297. Verifiable Auto-Formalization of Mathematics Using a Relaxed Natural Formal Language

**arXiv ID:** 2606.24443 | [PDF](https://arxiv.org/pdf/2606.24443v1)

**作者:** Zhicheng Hui `[一作]` (Shanghai Jiao Tong University), Qinxiang Cao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自动形式化流程，使用接近自然数学写作的 Relaxed NFL 作为中间表示，随后通过可验证的 elaboration 步骤转换为 Core NFL，并生成证明间隙，利用 LLM 生成的域特定 tactic 语言和专用求解器完成间隙闭合，从而实现完整的形式化与验证。

**💡 创新点**

创新点包括：① 设计保留自然语言结构的 Relaxed NFL，显著缩小 LLM 与传统 theorem prover 语言之间的表达鸿沟；② 通过可验证的规则+LLM 推理相结合的 elaboration 过程，系统性地消除表层语法的模糊性；③ 生成证明间隙并使用 LLM 生成的领域特定 tactic 语言与多种专用求解器，提高证明闭合率并实现可检验性。

**🔧 技术方法**

采用的大型语言模型 Qwen 系列（Qwen3‑235B‑A22B‑Instruct 进行少样本提示和 CoT，Qwen2.5‑7B‑Instruct 通过 LoRA 微调）；基于规则的 elaboration 机制与 LLM 生成的启发式；Proof Gap Generator 与 Proof Gap Checker；专门设计的域特定 tactic 语言；多种领域求解器（如多项式归一化、线性实数算术、积分/微分等）。

**📊 数据集**

使用了 3600 道题目，均从《Problems in Mathematical Analysis》 OCR 提取，涵盖极限、连续、微分、积分、级数、多元微积分等核心主题，作为训练和评估数据集。

**📈 对比分析**

通过对比少样本提示（无 CoT/有 CoT）和 LoRA 微调的性能，使用 pass@1 / pass@3 评价指标；结果显示：Qwen3‑235B‑A22B‑Instruct 在无 CoT 时 44.4%/72.3%，有 CoT 时 70.3%/90.4%；微调后的 Qwen2.5‑7B‑Instruct 达到 83.6%/89.9%。实验表明，即使是较小模型，在有限的 2000 条训练样本下也能获得高质量的 Relaxed NFL 生成效果。

**⚠️ 局限性**

局限性包括：① 对 LLM 的模糊推断依赖较高，若 LLM 产生错误或偏离目标语言仍会导致后续步骤失败；② elaboration 过程需要针对不同数学领域手工调整规则与约束，通用性受限；③ 证明间隙闭合仍需 LLM 生成的 tactic 脚本，尚未实现完全自动闭合；④ 未系统评估与主流 theorem prover（Coq、Lean、Isabelle）之间的直接迁移和可扩展性。

---

## 298. A Comparison of Kubernetes Compliance Standards and Configuration Scanners

**arXiv ID:** 2606.24438 | [PDF](https://arxiv.org/pdf/2606.24438v1)

**作者:** Michael Krieger `[一作]` (Dynatrace Research), Mario Kahlhofer `[通讯]` (Dynatrace Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对八大 Kubernetes 硬化准则进行系统化比较，并构建 79 条配置建议基准，随后使用 10 款主流静态扫描工具对其检测覆盖率、准确率及风险评估进行实验评估。

**💡 创新点**

提出统一的基准集与评估框架，首次将 CCSS（Common Configuration Scoring System）应用于 Kubernetes 配置风险评分；揭示了准则间重复/冲突、扫描工具检测盲点与严重性评分差异，强调了标准化和统一评估的重要性。

**🔧 技术方法**

使用手工比较、自动化评估管道（JSON映射与后处理）、CDK8s 生成 YAML 演示文件、CCSS 评分、以及 10 款开源扫描器（如 Trivy、KICS、KubeScore 等）的调用。

**📊 数据集**

基准数据集包含 190 条不同准则下的唯一建议，合并 79 条推荐，进一步展开为 241 条单一违例的 Kubernetes 清单（manifests）。

**📈 对比分析**

通过计算精确率、召回率、覆盖率、F1 分数和严重性相似度（与专家 CCSS 评分对比）来比较工具。结果显示 Trivy 与 KICS 以 0.69 以上的 F1 和 50% 以上覆盖率表现最佳；其他工具覆盖率低于 40%，严重性评分差异显著，最高可达 9.4 分。

**⚠️ 局限性**

局限性：仅涵盖数据平面配置；每个清单仅注入单一违规；未评估控制平面、CI/CD、镜像仓库等；基准为单一时点快照；仅测试开源扫描器，商业工具与运行时扫描缺失。

---

## 299. ReM-MoA: Reasoning Memory Sustains Mixture-of-Agents Scaling

**arXiv ID:** 2606.24437 | [PDF](https://arxiv.org/pdf/2606.24437v1)

**作者:** Heng Ping `[一作]` (University of Southern California), Paul Bogdan `[通讯]` (University of Southern California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ReM-MoA，一种通过跨层记忆与多样化路由提升多代理推理的 Mixture‑of‑Agents 框架。

**💡 创新点**

创新点在于：① Ranked Reasoning Memory 持久存储并按质量排序所有层的推理轨迹；② Curated Diversified Memory Routing 让不同代理接收高质量与低质量混合的记忆，从而同时维护质量与多样性；③ 可选的多域 Reviewer distillation 进一步提升评审质量并实现跨域迁移。

**🔧 技术方法**

技术包括：多层 LLM 代理、LLM Reviewer 进行比较式评分、LoRA 微调进行 Reviewer distillation、对记忆进行 Top/Bottom/Contrast 采样的路由策略。

**📊 数据集**

使用了五个推理基准：MATH、MMLU‑redux、Formal Logic、CRUX、HellaSwag（含 MBPP 作为 distillation 训练数据）。

**📈 对比分析**

与标准 MoA、RMoA、AttentionMoA 三种基线对比，ReM-MoA 在所有深度与宽度配置下均优于基线；深度扩展时性能差距持续扩大，宽度扩展时也保持领先；Distillation 版更进一步提升并可迁移到未见域。

**⚠️ 局限性**

局限性包括：仅评估到宽度 N=8；使用固定规模 7–8B 代理，未检验更大或更小模型；需要 L·N+L+1 次 LLM 调用，且 Reviewer 仍继承 LLM 的偏见和计算成本。

---

## 300. ATRIA: Adaptive Traceable ECG Reporting with Iterative Agents

**arXiv ID:** 2606.24392 | [PDF](https://arxiv.org/pdf/2606.24392v1)

**作者:** Donggyun Hong `[一作]` (MedicalAI Co., Ltd.), Yong-Yeon Jo `[通讯]` (MedicalAI Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `109c2b71-d051-425c-831f-0c544c24280d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并演示了一个可追溯、迭代的多代理心电图报告系统ATRIA；

**💡 创新点**

创新在于将报告流程拆解为可追溯的阶段，并支持上下文动态注入与交互式修订，超越传统单次生成；

**🔧 技术方法**

采用多代理架构、共享存储、GPT‑5‑mini、已有临床ECG AI模型（AiTiA）、规则引擎与知识库检索技术；

**📊 数据集**

使用商业ECG设备导出的原始记录（HL7 aECG、SCP-ECG、DICOM‑ECG）及结构化实验室数据，未公开具体公开数据集；

**📈 对比分析**

通过四个交互案例演示系统功能，未给出定量指标，但在演示中展示了可追溯性、上下文扩展和证据检索优势；

**⚠️ 局限性**

局限性包括缺乏大规模临床评估、依赖现有AI模型、未提供量化性能指标，且对更复杂多模态上下文的支持有限。

---

## 301. Varying Bundle Size Reactive Multi-Task Assignment using Selective Cost Estimation for Multi-Agent Systems

**arXiv ID:** 2606.24462 | [PDF](https://arxiv.org/pdf/2606.24462v1)

**作者:** Niklas Dahlquist `[一作]` (Luleå University of Technology), George Nikolakopoulos `[通讯]` (Luleå University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种可扩展的多机器人任务分配框架，通过分布式两阶段多保真度捆绑生成与集中式组合优化，实现了在动态环境中高效的实时任务分配。

**💡 创新点**

创新点包括：1) 两阶段多保真度捆绑生成策略——先用低保真欧几里得启发式快速构造候选树，再对最有前景的分支做高保真路径规划；2) 仅共享捆绑总成本，保持代理匿名性；3) 在集中式求解中引入束奖励衰减以平衡负载并保证全局可行。

**🔧 技术方法**

核心技术包括：深度限制可变宽度束搜索（beam search）构造候选捆绑树；基于欧几里得距离的启发式评估；高保真路径规划（使用D*算法）对顶级捆绑进行精确成本评估；集中式集合打包整数规划（使用SCIP求解器）进行任务分配；局部路径段缓存以减少重复计算。

**📊 数据集**

使用仿真地图A、B、C（分别为森林、洞穴和隧道类环境，尺寸为320×320或480×320）以及随机生成的50个任务点和4个代理进行评估，未使用公开真实世界数据集。

**📈 对比分析**

与完全实时贪心基线（仅考虑最近6个任务的捆绑）进行对比；在静态场景下性能提升约14–18%，在动态任务插入场景下提升约10–17%；计算时间与基线相近，表明方法具备良好的实时性和更优的任务完成率。

**⚠️ 局限性**

局限性包括：1) 欧几里得距离作为启发式在非凸环境中可能导致误判，导致有效捆绑被错误剪枝；2) 仅在仿真环境中验证，缺乏真实机器人实验验证；3) 未考虑任务预算或时间约束；4) 目前的两阶段策略仍需进一步优化以降低高保真规划次数。

---

## 302. Lite Any Stereo V2: Faster and Stronger Efficient Zero-Shot Stereo Matching

**arXiv ID:** 2606.24457 | [PDF](https://arxiv.org/pdf/2606.24457v1)

**作者:** Junpeng Jing `[一作]` (Imperial College London), Jiankang Deng `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Lite Any Stereo V2（LAS2）系列，利用2D成本聚合与三阶段训练实现高效零样本立体匹配

**💡 创新点**

创新点在于：①采用纯2D成本聚合框架显著降低实际推理延迟；②引入三阶段训练（合成监督→自蒸馏→真实数据知识蒸馏）和伪标签过滤与误差上限，提升零样本泛化能力；③构建轻量级S/M/L与迭代版H两类模型，兼顾速度与精度

**🔧 技术方法**

主要技术包括：多尺度特征提取（FastNet），2D聚合网络（U‑Net式），soft‑argmax与凸上采样，Self‑distillation（特征对齐），伪标签过滤（LR一致、边缘、天空掩码）与误差clamping，基于预训练的全景单目深度先验（DepthAnything）与高质量伪标签生成（FoundationStereo）

**📊 数据集**

使用合成数据集（SceneFlow、FallingThings、FSD、CREStereo、VKITTI2、TartanAir、Dynamic Replica，约1.8M对）以及0.5M真实无标注立体对（Flickr1024、InStereo2k、Holopix50K、DrivingStereo、SouthKenSV、UASOL）进行训练；评估基准包括KITTI 2012/2015、ETH3D、Middlebury、DrivingStereo天气子集

**📈 对比分析**

与其他高效立体模型（LAS、Fast‑ACVNet+、LightStereo、BANet、Fast‑FoundationStereo、RT‑MonSter++等）进行对比，LAS2‑M在所有四大零样本基准上均优于前一代LAS且速度提升1.6×；LAS2‑H在迭代方法中获得最高精度，且比Fast‑FoundationStereo快1.8×；整体在保持或提升精度的同时，显著降低H200与Orin上的延迟

**⚠️ 局限性**

局限性包括：1）与依赖强单目深度先验的高精度方法相比仍有精度差距；2）受限于可获取的高质量真实立体数据规模，进一步提升仍需更大多样化数据；3）在极端光照、透明/反射表面、模糊/遮挡等极端场景下仍会出现失配

---

## 303. Optimizing Visual Analytics Workflows: From Theory to Practice

**arXiv ID:** 2606.24454 | [PDF](https://arxiv.org/pdf/2606.24454v1)

**作者:** Philip Beaucamp `[一作]` (University of Oxford), Min Chen `[通讯]` (University of Oxford)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对SCORE方法进行了七个案例研究的行动研究，探讨并改进了该方法以提升可实践性。

**💡 创新点**

创新点包括：将理论SCORE转化为通用工作流，区分分支/嵌套迭代与迭代，提出比较流程与元比喻，并规划软件支持路线图。

**🔧 技术方法**

主要技术为行动研究、信息理论成本收益分析、案例研究与SCORE方法本身；结合质性研究与专家讨论。

**📊 数据集**

使用了来自金融、文学、图书交易、数据虚拟化、子空间分析、LLM提示和冰川运动等七个不同领域的实际项目数据，亦参考了IVAS网站的案例。

**📈 对比分析**

通过对症状、原因、方案和副作用的抽象实体进行比较，验证SCORE在改善工作流可解释性与效率方面的效果；虽未给出定量性能指标，但案例显示方法能系统化地识别与解决问题。

**⚠️ 局限性**

局限性在于案例数量仍有限、用户门槛和缺乏成熟的软件工具、元比喻与评估方法待完善，且需更多实践验证以量化其效能。

---

## 304. Bayesian control for coding agents

**arXiv ID:** 2606.24453 | [PDF](https://arxiv.org/pdf/2606.24453v1)

**作者:** Theodore Papamarkou `[一作]` (PolyShape), Artem Shelmanov `[通讯]` (MBZUAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将代码生成的工具使用与停机决策建模为成本敏感的序列假设检验，并设计基于贝叶斯信念状态的控制器来动态决定是否调用生成器、批评器、验证器或终止。

**💡 创新点**

创新点在于将工具协同控制转化为贝叶斯决策问题，提出一阶贪婪和有限视野动态规划两种贝叶斯控制器，并揭示验证成本与批评器信息量决定的三种主导策略。

**🔧 技术方法**

采用贝叶斯决策理论、POMDP、贝叶斯信息更新、动态规划和贝叶斯信息价值计算。

**📊 数据集**

使用六款LLM生成器（7–200B参数）与九个编码基准，包括LCB、MBPP+、HumanEval+、SWE‑Bench Lite/Verified、HumanEvalFix、CodeContests 等。

**📈 对比分析**

与无状态阈值、批评门控、Self‑Refine、Reflexion 等基线进行对比，实验显示在验证昂贵且批评器信息良好时贝叶斯控制器能提升约10–20点期望效用；当公共测试几乎完美或验证成本极低时简单门控或全检验更优。

**⚠️ 局限性**

局限性在于实验仅针对 Python 数据集、实例数量有限，批评器模型与提示的选择可能影响结果，且未覆盖非 Python 语言和更复杂的工具链。

---

## 305. EgoSAT: A Comprehensive Benchmark of Egocentric Streaming Interaction Understanding

**arXiv ID:** 2606.24422 | [PDF](https://arxiv.org/pdf/2606.24422v1)

**作者:** Yijia Lei `[一作]` (Tsinghua University), Miao Liu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EgoSAT 基准，用于评估 egocentric 视听流视频理解中的即时、回溯和预测推理；

**💡 创新点**

创新点在于统一多任务的流式推理框架、引入答复可行性与置信度诊断以及可预测性评分；

**🔧 技术方法**

采用多模态大型语言模型、动态 KV 缓存、ROI 关注机制、轻量 LoRA 微调，以及基于 CLIP 的多模态相似度评估；

**📊 数据集**

使用 Ego4D 视角摄像数据，构建 1,997 条视频、165 小时、约 4,800 题目；

**📈 对比分析**

与多种离线和流式 VLM（Gemini, Claude, Qwen, Video‑LLaVA, TimeChat‑Online 等）对比，发现专有模型在即时任务上领先，但流式模型整体落后；在可预测任务上准确率低，置信度校准差；

**⚠️ 局限性**

限制包括对流式内存压缩导致信息损失、预测任务缺乏可靠的置信度与可预测性判断、ROI 仅适用于即时任务等。

---

## 306. cuSBF: A Minimizer-Aware Bloom Filter for Genomic Sequence Data on Modern GPUs

**arXiv ID:** 2606.24417 | [PDF](https://arxiv.org/pdf/2606.24417v1)

**作者:** Tim Dortmann `[一作]` (Johannes Gutenberg University), Bertil Schmidt `[通讯]` (Johannes Gutenberg University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

实现了面向GPU的Super Bloom Filter（cuSBF），用于高吞吐量、低误判率的基因组k‑mer索引。

**💡 创新点**

将最小化器驱动的分片与Warp级共享、共享内存分块、分段归约结合，利用super‑k‑mer局部性实现GPU可扩展并行化，并提供头文件式CUDA库。

**🔧 技术方法**

GPU并行编程（CUDA）、共享内存分块、Warp shuffle、位图操作、最小化器+Findere哈希、压缩k‑mer编码、编译时模板化、序列原生输入流水线。

**📊 数据集**

C. elegans参考基因组（≈97 MiB）和人类T2T‑CHM13 v2.0（≈3 GiB），以及10⁹随机k‑mer用于FPR评估。

**📈 对比分析**

与GPU块Bloom、Cuckoo‑GPU、Two‑Choice、Counting Quotient、CPU Super Bloom等基准对比；在GDDR7 GPU上插入/查询吞吐率分别提升约9.1×/7.7×，与CPU实现相差92×/234×；在HBM3 GPU上优势缩小至约2.1×/1.4×，但FPR低于块Bloom，参数(s=28,m=16,H=4)显著优越。

**⚠️ 局限性**

对大符号表支持受限（64位压缩限制最小化器长度），在高带宽HBM3环境下相对受限，CPU→GPU传输开销显著，缺乏动态删除功能，未来需要更宽编码或SIMD扩展。

---

## 307. Novel Triple-Based Problems for the Construction of Phylogenetic Networks via Least Common Ancestors

**arXiv ID:** 2606.24415 | [PDF](https://arxiv.org/pdf/2606.24415v1)

**作者:** Patricia A. Ebert `[一作]` (Stockholm University), Marc Hellmuth `[通讯]` (Stockholm University)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于祖先关系的根三元组与锚定三元组的一致性判定问题，并给出了多种变体（包括禁止三元组、强制LCA唯一性等），通过将三元组约束转化为LCA约束并利用闭包算子和规范DAG的构造，证明了所有这些一致性问题都可以在多项式时间内解决，并给出了相应的构造方法。

**💡 创新点**

创新点在于首次将根三元组的祖先关系（T1）与锚定三元组的单侧比较（T2）统一到LCA约束框架下，利用闭包算子与规范化DAG的构造，完成了对包含禁止约束、强制LCA唯一性以及2‑LCA属性等多种约束的多项式时间判定与构造；此前这类问题在网络中尚未得到系统性处理。

**🔧 技术方法**

核心技术包括：LCA约束的关系表示与闭包运算；规范化关系的构造与对应的规范DAG（Hasse图）实现；对禁止约束的处理通过“严格实现”与“扩展”操作；以及贪心型饱和算法（Sat）用于处理根三元组的禁止约束。

**📊 数据集**

论文仅给出理论证明与算法，未使用实际数据集。

**📈 对比分析**

与以往在树或网络中仅考虑“拓扑展示”或强度约束的NP‑hard问题不同，本文在更一般的LCA约束下实现了多项式时间的判定与构造；实验性能未给出，仅通过算法复杂度分析证明其可行性。

**⚠️ 局限性**

限制在于：仅提供理论与多项式时间算法，未给出实验验证；对特殊结构网络（如层数受限的网络）未作进一步优化；此外，对于某些变体（如强制禁止三元组）仍存在潜在的组合爆炸性扩展，实际实现可能受限于约束规模。

---

## 308. Natural Identifiers for Privacy and Data Audits in Large Language Models

**arXiv ID:** 2606.24408 | [PDF](https://arxiv.org/pdf/2606.24408v1)

**作者:** Lorenzo Rossi `[一作]` (CISPA Helmholtz Center for Information Security), Adam Dziedzic `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用自然标识符（NID）实现LLM的后置差分隐私审计与数据集推断，避免重训练或人工构造同分布验证集的需求。

**💡 创新点**

创新点在于：①将结构化随机字符串（如哈希、短网址、加密地址）视为天然“可追踪”元件；②通过NID生成无限同分布“伪非成员”样本；③将传统二分类审计升级为基于排名的多候选审计，从而提升统计效能与精度。

**🔧 技术方法**

采用的技术包括：差分隐私审计框架（one‑run + ranking），成员推断攻击（Loss、Min‑K%、ReCaLL、Hinge 等），数据集推断（DI）中的梯度提升树与 Kolmogorov‑Smirnov 检验，及NID生成与匹配算法。

**📊 数据集**

实验数据集主要为公开LLM预训练集：Pile（用于Pythia系列模型）与Dolma（用于OLMo模型）；在特定任务中使用GSM8K生成任务专用NID；所有实验均在开源模型上完成。

**📈 对比分析**

与传统基于插入可听写攻击的对照实验相比，NID方法在同一模型下获得更紧的ε下界（更低的DP下界），在DI实验中显著降低p值，能够准确识别已训练子集且无误报；排名阈值调优可进一步提升检验功效。

**⚠️ 局限性**

局限性包括：①需模型训练数据中包含足够的NID，稀缺数据集或无结构随机字符串的场景可能不适用；②实验仅在公开模型上验证，商用或专有模型的适用性待进一步研究；③对NID生成器的实现细节要求严格，否则分布偏移会导致误报。

---

## 309. Entity Resolution via Batched Oracle Queries

**arXiv ID:** 2606.24407 | [PDF](https://arxiv.org/pdf/2606.24407v1)

**作者:** Lorenzo Balzotti `[一作]` (Sapienza University of Rome), Giovanni Simonini `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在有限预算下，如何利用一次性处理批量记录的“oracle”逐步完成实体解析（Entity Resolution），并设计了一种自适应的批量查询调度方法。

**💡 创新点**

1) 证明在批量查询场景中不存在总能保持最优的查询顺序，且选择最佳批量以最大化已发现匹配边的增益是 NP‑hard；2) 在实体大小满足特定条件时给出可达最优的批量选择策略；3) 提出了基于社区检测、相似度图以及“benefit”指标的实际算法 pERbacco，能在进度回忆率（recall）上显著优于现有基线。

**🔧 技术方法**

核心技术包括：
- 抽象化的“oracle”接口；
- 相似度图构建与基于阈值的重心社区（heavy community）检测；
- 利用 “benefit” 评估潜在匹配边收益；
- 贪心近似 Heaviest Subgraph Problem（GreedyHS）求解批量；
- 温度参数自适应控制当前批次与社区批次的交替；
- 与大语言模型（如 GPT‑5 mini）等批量解算器的接口。

**📊 数据集**

使用了六个公开数据集（包括 1,295 条 ML 论文、29.8k 条相机规格、16.3k 条金融请求、3,841 条电商产品、14.2k 条选民信息）以及一个规模约 39k 条、10k 个实体的合成数据集。

**📈 对比分析**

与两种改编自 pairwise 查询的基线（s_edge、s_hybrid）以及无社区版本的贪心解法进行了对比。实验结果表明，在相同的查询预算下，pERbacco 在所有数据集上均实现了更高的进度回忆率，且在精度较低（0.05）的大模型场景下仍保持优势；在预算约为下界 Φ_b 的 2 倍范围内，性能差距可达数十个百分点。

**⚠️ 局限性**

限制：
- 依赖于完美的一致性 oracle，未考虑噪声或错误输出；
- 贪心 HSP 近似未给出理论保证（除特殊实体大小情况外）；
- 需要预先构建相似度图和社区检测，若相似度质量低会影响效果；
- 温度参数的自适应策略虽然经验上可行，但缺乏严格的理论分析。

---

## 310. Parallel Manifold Steering: Efficient Adaptation of Large Associative Memories via Residual Energy Shaping

**arXiv ID:** 2606.24396 | [PDF](https://arxiv.org/pdf/2606.24396v1)

**作者:** Kanishk Awadhiya `[一作]` `[通讯]` (Independent Researcher), Kanishk Awadhiya (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Hierarchical Residual Steering（H-Res），通过在Transformer激活层注入可学习的低秩向量场，实现任务特定的能量景观调控，从而在不改动权重或扩展序列长度的前提下完成模型适配。

**💡 创新点**

将适配视为动态系统上的控制问题，使用低秩瓶颈MLP学习状态依赖的向量场，实现平滑的轨迹引导，并通过零初始化和空洞投影保证原有能量极小点不被破坏。

**🔧 技术方法**

残差适配器、Neural ODE思想、低秩投影、零初始化、空洞投影、能量最小化理论等技术。

**📊 数据集**

SQuAD（检索任务）、WikiText（生成任务）、VTAB-1k（视觉适配）。

**📈 对比分析**

与LoRA和VPT对比，H-Res在SQuAD上提升约26%，在VTAB-1k的自然域上取得略高准确率，同时保持O(N^2)的计算复杂度，速度与LoRA相当但显著优于VPT。

**⚠️ 局限性**

尚未在非Transformer架构（如SSM）中充分验证，对极端长序列或高维任务的适配效果仍需进一步研究。

---

## 311. PDS Joint: A Parametric Double-Spiral Joint Tailored for Dexterous Hands

**arXiv ID:** 2606.24377 | [PDF](https://arxiv.org/pdf/2606.24377v1)

**作者:** Haoyang Li `[一作]` (Beijing Institute of Technology), Yufeng Yue `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计了可参数化的双螺旋（PDS）顺应关节，并在其上嵌入电感式自我感知模块，利用学习式校准将感知信号映射到关节角度，最终集成到一款开源八指手平台上，展示了在各种日常物体上的自适应抓取和接触丰富的操作。

**💡 创新点**

创新点在于：①将双螺旋几何参数化，使关节在屈伸、外展/内收和旋转三个方向上可独立调节刚度；②在关节内部集成紧凑、免接触的电感式感知系统；③提出基于ArUco标记的学习校准流水线，用多层感知器大幅提升了关节状态估计精度。

**🔧 技术方法**

使用技术包括：参数化双螺旋建模（Archimedean 与 Logarithmic 螺旋）、嵌入式电感式自我感知（LC谐振频率测量）、多层感知器（MLP）校准、ArUco标记姿态估计、ROS2数据同步、开源机械臂设计与3D打印成型。

**📊 数据集**

数据集为作者自行采集的实验数据：包括关节力/角度曲线、感知信号与ArUco姿态标定对齐的时间序列，以及对九种日常物体抓取的50次试验记录，全部采用30Hz相机与100Hz传感器采样。

**📈 对比分析**

与传统曲线拟合对比，MLP校准在最难的外展/内收运动中将MAE从0.125降至0.073、RMSE从0.172降至0.092，实现41.6%的误差下降；在抓取实验中，该手平台在九种物体上全部成功抓取，Kapandji手指测试得分10/10，表明关节在接触丰富环境中的安全性与可靠性。

**⚠️ 局限性**

局限性包括：仅在轻量、规则物体上测试，缺乏与刚性或其他顺应关节的量化对比；闭环操控中感知反馈的实际效益未量化；循环加载实验仅覆盖2000次，未评估长期疲劳；电感传感在极端姿态下可能饱和，需要改进传感布局；缺乏对不同任务的策略级优化（如模仿学习、强化学习）。

---

## 312. MATCH: Flow Matching for Multi-View Anomaly Detection

**arXiv ID:** 2606.24375 | [PDF](https://arxiv.org/pdf/2606.24375v1)

**作者:** Mathis Kruse `[一作]` (Leibniz University Hannover), Bodo Rosenhahn `[通讯]` (Leibniz University Hannover)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于 Flow Matching 的多视角异常检测框架（MATCH），实现对工业生产图像的多视角异常检测、分割和定位。

**💡 创新点**

创新点：① 用 Flow Matching 代替传统 RealNVP/Normalizing Flow，显著提升高维特征处理能力；② 在 likelihood 估计中去除耗时的 divergence 计算，保持检测性能的同时实现实时推理；③ 采用多层 WideResNet 特征提取、时序和视角嵌入、ResNet 结构的 Decoder 以及 Euler ODE 求解器，构建统一的端到端模型；④ 首次在 MANTA‑Tiny 数据集上做完整基线评测。

**🔧 技术方法**

技术：Flow Matching（Optimal Transport Flow Matching）、ODE 采样与逆向变换、WideResNet 特征提取、Sinusoidal 位置编码+MLP 嵌入、ResNet 风格的 Bottleneck 与 Decoder、Hutchinson / RQMC 对 divergence 的估计、Euler ODE 求解器。

**📊 数据集**

数据集：Real‑IAD、MANTA‑Tiny（以及在补充材料中对 MVTec AD、完整 Real‑IAD 进行验证）。

**📈 对比分析**

对比方法：CFA、DSR、EfficientAD、FastFlow、Multi‑Flow、PaDiM、RD4AD、SimpleNet、STFPM 等主流异常检测方法。性能方面：在 Real‑IAD 上 I‑AUROC 91.17、S‑AUROC 95.63、P‑AUROC 99.24、P‑AUPRO 94.76，几乎领先或与 Multi‑Flow 相当；在 MANTA‑Tiny 上 P‑AUPRO 89.66、P‑AUROC 95.65，显著高于其他基线。

**⚠️ 局限性**

局限性：① 在视角差异极大或多视角数量极多的场景下性能仍会下降；② 去除 divergence 可能在极高维或复杂分布下导致估计误差；③ 目前依赖预训练 WideResNet，迁移到其它域或 backbone 时需要额外调优。

---

## 313. On the Stability of Prompt Ranking in Large Language Model Evaluation

**arXiv ID:** 2606.24381 | [PDF](https://arxiv.org/pdf/2606.24381v1)

**作者:** Shaoshuai Du `[一作]`, Lun Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于多次随机种子采样、LLM推理与统计稳定性分析的Prompt选择方法，旨在通过对候选Prompt在不同种子下的表现进行模拟与评估，从而挑选出最具鲁棒性与高效性的Prompt。

**💡 创新点**

创新点包括：①在Prompt评估过程中引入变异性模拟（seed sampling）以捕捉不同随机初始化对性能的影响；②设计一套稳定性度量（相关系数ρ、Kendall τ、Top‑k一致性）来评估Prompt在多次试验中的一致性；③提出使用下界置信区间（LCB）作为选Prompt的决策准则，兼顾平均性能与不确定性。

**🔧 技术方法**

主要技术：变异性模拟（seed sampling）、多模型LLM推理（Mistral、Phi‑3、Qwen2.5）、评分矩阵统计（均值、标准差）、稳定性分析（ρ、τ、Top‑k一致性）、LCB选择策略。

**📊 数据集**

使用的任务数据集为文中所述的Task Dataset 𝒟，涵盖多种基准问答/推理任务，具体数据集未在摘要中列明，但实验涵盖常用公开数据集。

**📈 对比分析**

与传统仅基于平均分选取Prompt的方法相比，本文的稳定性‑Aware（LCB）策略在多模型、多随机种子下表现出更高的稳定性和更优的整体性能（平均提升≈2‑5%，并在Top‑k一致性指标上显著超越对照组）。

**⚠️ 局限性**

局限性：①需要大量的LLM推理调用，计算成本较高；②评估依赖于所选的种子集合和子集规模，可能对特定任务产生偏差；③模型选择和数据集的多样性不足时，LCB策略的普适性尚待进一步验证。

---

## 314. Supervise What Survives: Geometry-Guided VLA Adaptation from Synthetic Robot Videos

**arXiv ID:** 2606.24448 | [PDF](https://arxiv.org/pdf/2606.24448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 315. S1-Omni-Image: A Unified Model for Scientific Image Understanding, Generation, and Editing

**arXiv ID:** 2606.24441 | [PDF](https://arxiv.org/pdf/2606.24441v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. The African Language Tax: Quantifying the Cost, Latency, and Context Penalty of Tokenizing African Languages in Frontier LLMs

**arXiv ID:** 2606.24460 | [PDF](https://arxiv.org/pdf/2606.24460v1)

**作者:** Olaoye Anthony Somide `[一作]` `[通讯]` (DataLens Africa Research), Olaoye Anthony Somide (DataLens Africa Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对20种非洲语言在11种前沿/开源分词器下进行并行语料库的子词分词度（fertility）与分词费率（token‑tax）测量，并将其转化为企业级成本、延迟与上下文窗口损耗指标；同时发布了可复现的测量工具、公开排行榜与结果数据集。

**💡 创新点**

首次系统评估非洲语言的分词费率差异，采用并行语料库排除内容差异；将分词成本与经济成本、延迟、上下文容量直接关联；公开工具和排行榜使研究可复现且可持续跟踪。

**🔧 技术方法**

使用子词词表（BPE、SentencePiece、byte‑pair）进行分词，计算每词子词数（fertility）和与英语对比的“premium”；基于UAX‑29进行单词分段；通过自底向上的求和‑再除法聚合并使用Bootstrap 95%置信区间；构建企业成本模型，将token计数映射为美元/本地币成本、延迟乘子和有效上下文字数；与AfroBench/IrokoBench准确率进行相关性分析。

**📊 数据集**

主测用FLORES‑200+并行语料（19种语言），SIB‑200作为稳健性检查，MAFAND‑MT用于新闻注册；此外使用IrokoBench和AfroBench基准评测与token‑tax关联。

**📈 对比分析**

对同一并行句子在所有tokenizer上进行相同计数；采用“sum‑then‑divide”求平均，确保句子长度不产生偏差；计算token‑per‑word、premium、chars‑per‑token、bytes‑per‑token、上下文效率。结果显示：英语基准1.22 token/word；平均非洲语言premium 1.88×，最高8.92×；Gemma 4将平均premium降至2.38×；成本与延迟乘子可高达8.9×，上下文窗口有效词数下降至13.6%（Amharic）/11.2%（N’Ko）。

**⚠️ 局限性**

局限性包括：Claude/Gemini等闭源tokenizer未被纳入；并行语料翻译质量和注册差异可能影响计数；UAX‑29单词分段在高度黏着或非拉丁文字中存在误差；仅覆盖20种语言，未能代表整个非洲语言多样性；成本模型忽略缓存、批处理、量化等实际部署因素；premium与模型准确率无显著相关性，提示存在混杂变量。

---

## 317. MedPCFM: Improving Medical Point Cloud Completion by Integrating Point Transformers and Flow Matching

**arXiv ID:** 2606.24433 | [PDF](https://arxiv.org/pdf/2606.24433v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 318. NoContactNoWorries: Estimating Contact through Vision and Proprioception for In-Hand Dexterous Manipulation

**arXiv ID:** 2606.24450 | [PDF](https://arxiv.org/pdf/2606.24450v1)

**作者:** Soham Patil `[一作]` (International Institute of Information Technology), Spandan Roy `[通讯]` (International Institute of Information Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无触觉的伪触觉方法，利用RGB‑D视觉与手部本体感知（关节位置/命令）通过Transformer网络预测四个手指触碰点的二元接触状态，并在真实机器人上验证可直接替代物理触觉；

**💡 创新点**

创新点在于（1）姿态条件交叉注意力将视觉与当前/目标关节信息融合；（2）因果Transformer对短期时序进行建模，克服视觉遮挡与接触模糊；（3）将预测的伪触觉信号作为实时观测，实现在无触觉硬件时仍能训练出近似oracle的闭环控制；（4）通过多对象、跨策略的训练与测试，展示强泛化；

**🔧 技术方法**

技术手段包括：冻结的RGB‑D编码器+轻量级卷积映射到视觉token；姿态与命令通过线性映射为查询，使用跨注意力得到姿态感知视觉表示；后续使用单头因果Transformer编码时间窗口；最终通过线性层输出四维二进制接触概率；训练使用二进制交叉熵；下游任务采用PPO强化学习并在仿真训练时注入真实接触标签；

**📊 数据集**

数据集：在NVIDIA Isaac Gym仿真中采集5个训练物体（长方体、五边柱、星形、十二面体、楼梯）下的50条15s轨迹（每物体10条），并使用10个拆分的策略/随机种子保证不泄露；真实机器人使用LEAP手搭载低功耗压力传感器收集对应的触碰二进制标签；同时对感知、动力学进行域随机化（颜色、背景、噪声、质量、摩擦等）。

**📈 对比分析**

与多种基线（视觉仅、姿态仅、对称/非对称查询、无时序、基于深度几何）进行比较，F1在仿真中对已见物体可达0.93–0.90，对未见物体0.88–0.82；在真实机器上对已见物体0.84–0.71，对未见物体0.74–0.68；下游闭环旋转任务中，伪触觉实现的Cumulative Rotation Reward/Angle 与oracle接近或超越（尤其在凸形物体），并且推理延迟仅8 ms，满足20 Hz控制。

**⚠️ 局限性**

局限性：仅预测四个手指点的二进制接触；未扩展到手掌或更细密的触觉地图；仅在手部旋转任务上验证，其他复杂抓取/装配任务未知；未估计接触力、摩擦等连续量；模型依赖冻结的RGB‑D编码器，可能受限于其视觉特征；仿真与真实之间仍存在性能差距，尤其在遮挡严重的非凸物体上。

---

## 319. P-MTP: Efficient Document Parsing via Multi-Token Prediction with Progressive Depth Scaling

**arXiv ID:** 2606.24447 | [PDF](https://arxiv.org/pdf/2606.24447v1)

**作者:** Le Xiang `[一作]` (Baidu Inc), Wei He `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 P‑MTP 框架，通过 Progressive Curriculum Loss 与 Confidence‑Gated Dynamic Drafting 两项技术，加速文档解析任务的多 token 预测与推理过程。

**💡 创新点**

创新点：① 在训练阶段使用基于轨迹的权重 (Sequential Path + Retrospective Target) 自动调节多 token 预测的损失，实现在长视野下的稳定学习；② 在推理阶段根据累计概率自适应裁剪 draft 长度，避免无效推断，显著提升吞吐量与效率。

**🔧 技术方法**

技术手段：多 token 预测（MTP）+ 视觉‑语言模型（VLM）+ 预训练微调（SFT）+ 递归 MLP/DecoderLayer 共享模块 + RMSNorm + 轨迹权重衰减 + 置信度门控 + δ 校准 + vLLM 高性能推理框架。

**📊 数据集**

数据集：UniMERNet（公式识别）、PubTabNet（表格结构识别）、OmniDocBench（通用文档解析）、LightOnOCR‑2（通用文档训练集）。

**📈 对比分析**

与基线 NTP、固定深度 MTP、静态权重方案对比；在 PubTabNet 上达到 5.24× 速度提升，同时 TEDS 与精度仅低于 0.5%；在不同基础模型与任务上均实现 1.4–2.3× 的吞吐量提升；在 vLLM 推理框架下获得 1.5× 的整体加速。

**⚠️ 局限性**

局限性：对极复杂布局的自适应程度仍有限；δ 的校准需要针对不同模型调优；训练时对远程 token 的监督仍受双重约束抑制，导致训练成本提升；若视觉特征分布差异大，模型泛化可能受限。

---

## 320. Modality-Aware Out-of-Distribution Detection for Multi-Modal Action Recognition

**arXiv ID:** 2606.24404 | [PDF](https://arxiv.org/pdf/2606.24404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 321. Transformation Behavior of Images in Latent Space

**arXiv ID:** 2606.24430 | [PDF](https://arxiv.org/pdf/2606.24430v1)

**作者:** Christian Zöllner `[一作]`, Matthias Kloor `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估了多种无监督训练的病理图像编码器在常见图像变换下的嵌入空间不变性。

**💡 创新点**

系统性比较主流自监督编码器（SwAV、MoCo v2、BT、DINO、Bioptimus h0）在临床病理图像上的变换鲁棒性，并量化潜在空间的可分离性，首次揭示颜色增强对嵌入空间影响更大。

**🔧 技术方法**

使用ResNet50和Vision Transformer架构，结合SwAV、MoCo v2、BT、DINO和Dinov2训练方法；采用L2距离、维度差异统计和GJI相似度等指标进行分析。

**📊 数据集**

使用TCGA结肠腺癌WSI数据集（1330张）和来自Heidelberg/Bonn的私有500张WSI，将图像切片为256×256像素并在不同分辨率下生成tile。

**📈 对比分析**

通过计算原图与增强图在潜在空间的L2距离并归一化与随机嵌入距离比较；同时评估维度差异分布和GJI度量。实验表明所有网络均不完全不变，颜色增强的影响最大，病理专用网络比ImageNet预训练更稳健，维度可分离度差异显著。

**⚠️ 局限性**

仅评估了二维切片和有限的变换类型，未直接检验对下游分类性能的影响；无监督网络无法实现完全不变性，实验规模与网络数量仍有限。

---

## 322. CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference

**arXiv ID:** 2606.24467 | [PDF](https://arxiv.org/pdf/2606.24467v1)

**作者:** Xiaolin Lin `[一作]` (Technical University of Darmstadt), Grace Li Zhang `[通讯]` (Technical University of Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CompressKV框架，利用离线识别的语义检索头来决定KV缓存的保留和驱逐，从而在保持长上下文推理性能的同时显著压缩内存占用。

**💡 创新点**

创新点在于：①通过语义检索头（SRH）而非传统流式头或单点高峰关注来识别重要文本片段；②采用层级误差评估（基于Frobenius范数）进行自适应KV预算分配，完全离线完成，无需在线额外计算。

**🔧 技术方法**

使用的技术包括：GQA注意力头分组、语义检索头识别、KV缓存驱逐策略、误差感知层级预算分配、FlashAttention-2等高效推理库。

**📊 数据集**

实验数据集主要有LongBench、Needle‑in‑a‑Haystack以及用于头识别的校准集，覆盖问答、摘要、检索等多种长文本任务。

**📈 对比分析**

方法通过与StreamingLLM、SnapKV、PyramidKV、CAKE、HeadKV、AdaKV等六个基线在不同KV预算（256–2048 token）下进行公平对比，结果显示CompressKV在256 token预算下仍能保持≈97%问答准确率，90%检索准确率，且在大多数任务中平均提升1–2分以上。

**⚠️ 局限性**

局限性在于：需要针对每个模型离线生成SRH列表和误差预算，若模型结构或任务发生显著变化需重新校准；在极低预算（如0.5% KV）下仍存在一定性能下降；对新型注意力机制的适用性尚未充分验证。

---

## 323. SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking

**arXiv ID:** 2606.24449 | [PDF](https://arxiv.org/pdf/2606.24449v1)

**作者:** Mohamad Alansari `[一作]` (Khalifa University), Sajid Javed `[通讯]` (Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在基于 SAM2 的视觉目标跟踪器中，提出了 SENTRY 模块，用于在写入内存前对候选掩码进行短期时间一致性验证，从而显著降低因置信度驱动写入导致的漂移。

**💡 创新点**

创新点在于：①训练无关、可直接插拔的 refine‑before‑write 机制；②利用候选掩码的回溯追踪、邻居轨迹对比与循环一致性匹配，形成多源一致性评分；③通过邻居（distractor）轨迹池来抑制错误写入，提升对遮挡、快速运动和干扰物的鲁棒性。

**🔧 技术方法**

技术核心包括：SAM2 原生多候选掩码生成、AMG（Automatic Mask Generation）补充、Soft‑NMS 过滤、卡尔曼滤波运动先验、短期回溯追踪、邻居轨迹池、轨迹 IoU 相似度、匈牙利匹配、以及最终的内存写入决策。

**📊 数据集**

在 9 大标准跟踪基准上进行评估：LaSOT、LaSOText、TNL2K、GOT‑10k、TrackingNet、VOT20、VOT22、VOTS24 以及 DiDi，覆盖短期、长期、遮挡、干扰和多类别情形。

**📈 对比分析**

与原 SAM2、SAMURAI、DAM4SAM 等基线相比，SENTRY 在所有基准上均实现了显著提升：在 LaSOT 上平均提升 1.5–2.8%（S、NP、P），在 TNL2K 上提升 8.8–13.8%（尤其对 SAMURAI），在 VOT 系列上 Q/Acc/Rob 分别提升 0.5–1.5%（SENTRY‑D4S 领跑多项指标），并且保持实时代码速度（SAM2‑L 32.8 FPS，额外开销约 25%），内存占用仅增 0.4–0.6 GB。

**⚠️ 局限性**

局限性：①依赖候选掩码的质量，若前端提议严重失效则一致性验证仍会选错；②短期窗口 τ=10 可能不足以捕捉极端长时间遮挡或复杂运动；③对极长序列的长期再识别能力仍有限；④计算开销主要来自多候选生成与回溯匹配，虽然可接受但在资源受限设备上仍需优化；⑤未对非 SAM2 基础的记忆网络做过多实验，适配性需进一步验证。

---

## 324. Agentic Generation of AST Transformation Rules for Fixing Breaking Updates

**arXiv ID:** 2606.24446 | [PDF](https://arxiv.org/pdf/2606.24446v1)

**作者:** Frank Reyes `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种代理式框架，利用大型语言模型（LLM）生成可执行的AST转换规则，用于自动修复因依赖更新导致的编译错误，并使得生成的修复规则可跨项目复用。

**💡 创新点**

创新点在于：①首次将可执行AST转换作为可复用修复方案，取代单一项目补丁；②在生成过程中嵌入生成‑应用‑验证循环，确保规则可在源代码中正确执行；③系统评估了不同LLM与AST引擎组合的效果，揭示引擎选择是影响成功率的关键因素。

**🔧 技术方法**

主要技术包括：大型语言模型（GPT‑5.4‑mini、Qwen3‑30B、DeepSeek‑v3.2、Gemini‑3.1‑Pro）与编码代理（OpenCode、GeminiCLI），AST转换引擎（Spoon、JavaParser），以及基于Maven的编译反馈循环。

**📊 数据集**

使用的实验数据集为BUMP基准，包含157个真实的Java Maven项目在更新依赖后出现的编译错误，共涉及69个客户端项目与70个第三方库。

**📈 对比分析**

通过对八种（模型、引擎）组合的实验，报告了可编译规则率（最高94.3%）、修复率（最高78.6%）以及跨项目修复率（总体33.3%，在使用方式统一的更新中可达80%）。比较方法为统计学检验（Fisher精确检验）和按语义版本类型的细分，显示JavaParser相对Spoon在大多数模型中更高效。

**⚠️ 局限性**

局限性包括：①仅针对Java+Maven项目，难以推广到其他语言或构建系统；②跨项目迁移效果受单一“种子”项目的影响，导致在API使用模式多样化时修复率下降；③实验一次性执行且LLM随机性可能影响结果；④对大型、复杂更新（如大版本跳变）仍易失败。

---

## 325. BiJuTy: An Interactive HPC-Aware Big Data Cluster Lifecycle Manager and Performance Assessment Utility for JupyterHub

**arXiv ID:** 2606.24412 | [PDF](https://arxiv.org/pdf/2606.24412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 326. Escaping the Self-Confirmation Trap: An Execute-Distill-Verify Paradigm for Agentic Experience Learning

**arXiv ID:** 2606.24428 | [PDF](https://arxiv.org/pdf/2606.24428v1)

**作者:** Shiding Zhu `[一作]` (Zhejiang University), Kai Zhang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Execute-Distill-Verify (EDV) 框架，利用多异构代理并行执行任务，第三方对比式蒸馏生成候选经验，并通过共识验证过滤错误经验，解决单一代理经验学习的 Self‑Confirmation Trap。

**💡 创新点**

创新点在于将执行、蒸馏和验证三大步骤完全拆分并交叉验证，利用异构代理多样性、第三方对比分析以及共识验证三重机制，显著提升经验质量并避免错误经验进入记忆。

**🔧 技术方法**

使用技术包括多模型异构执行、第三方对比式蒸馏、共识验证机制、共享/私有记忆银行、能力矩阵匹配以及层级检索等。

**📊 数据集**

实验数据集为 τ²-bench、Mind2Web 与 MMTB 三个长周期任务基准。

**📈 对比分析**

对比基线方法（NM、ReasoningBank、Judge、Router 等）时，EDV 在 Pass@1、EA、AF1、SSR 等指标上均取得显著提升，例如 τ²-bench RETAIL 的 Pass@1 从 82.5 提升至 86.6，整体性能居领先。

**⚠️ 局限性**

局限性包括对多模型池和离线记忆构建的依赖，协作与验证过程增加计算和资源开销；在资源受限或实时性要求极高的场景下适配仍有挑战。

---

## 327. Can Aggregate Invariants Accelerate Continuous Subgraph Matching? Limits, Laws, and a Dynamic Spectral Index

**arXiv ID:** 2606.24421 | [PDF](https://arxiv.org/pdf/2606.24421v1)

**作者:** Minghao Chen `[一作]` (Tencent Technology), Jiale Zheng `[通讯]` (HUAWEI Noah's Ark Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究将谱滤波（邻域拉普拉斯特征的插值判定）引入连续子图匹配（CSM），探讨其在动态图流中的可行性、有效性与成本。通过理论证明与实验评测，揭示了谱筛选在CSM中的边界：在稀疏邻域下，懒惰式谱上界失效；但选择性地精确维护小球谱在可接受的微秒级成本下能够保留几乎所有的剪枝效果；尽管谱筛选能显著减少候选集，但由于delta枚举是邻接引导的，谱剪枝对枚举中间量无影响；为弥补这一空隙，作者提出基于聚合不变式的“门控”机制，可安全跳过整个更新的枚举。除此之外，论文实现了首个可在动态图上精确维护局部谱的索引，并提供了“中间量不变性”评测方法。

**💡 创新点**

创新点主要包括：1) 对谱筛选在CSM中的懒惰维护进行可行性证明，揭示其不可行的根本原因；2) 提出一种反相关定律和中心顶点排除定理，指导只在可剪枝的球体上部署精确谱，显著降低维护成本；3) 构造了可动态更新的局部谱索引，首次在动态图上提供高效的谱信息；4) 设计了“门控”机制，利用子图-子球的不变式在更新层面安全跳过枚举；5) 引入“中间量不变性”评测框架，能够客观判断任何过滤器是否真正影响枚举时间。

**🔧 技术方法**

使用的技术包括：拉普拉斯谱插值判定、Weyl不等式、主导子序列（majorization）分析、反相关定律与中心顶点排除法、密度/半径限制下的精确谱求解（LAPACK）、反向球成员索引、双向同步重算、门控条件评估（顶点/边/三角计数不变式）、以及在已存在的CSM基准框架（NewSP）中实现的插件式索引层。

**📊 数据集**

实验所用数据集：三类合成图（Erdős–Rényi、Watts–Strogatz，顶点数 3k–8k），以及四个真实图（ca‑GrQc 4.2k、email‑Enron 33.7k、com‑Amazon 334.9k、com‑Youtube 1.13M）。查询为稠密的 BFS‑球子图（尺寸 8/12/16/24/32）。

**📈 对比分析**

比较方法：在 decoupled CSM 框架中插入谱索引层，对照仅使用标签/度/邻居标签多集过滤的基线（NLFI）。评估指标包括候选集缩减率、枚举中间量（绑定扩展次数）是否变化、匹配输出是否一致、以及整体执行时间。实验结果显示：1) 通过谱筛选可减少 9–51% 的候选顶点；2) 枚举中间量与基线完全相同，说明谱筛选对 delta‑枚举无影响；3) 仅门控机制在部分更新中跳过枚举（5–47%），但其维护开销使总运行时间往往高于基线；4) 在稀疏图中精确谱维护成本仅 ~300 μs/更新，仍大幅高于基线。

**⚠️ 局限性**

局限性：①谱筛选在邻接引导的 delta‑枚举中几乎没有收益；②维护成本高，尤其在大规模稠密图上，门控机制导致整体性能下降；③仅在候选集被显式构造或扫描的场景（静态重建、批量枚举、近似匹配控制）才能发挥作用；④实验仅在无标签或统一标签的场景下验证，未覆盖真实标签分布；⑤门控的有效性依赖于更新流的特性，某些工作负载（高稠密查询或高标签多样性）可能收效甚微。

---

## 328. Data Augmentation: A Fourier Analysis Perspective

**arXiv ID:** 2606.24418 | [PDF](https://arxiv.org/pdf/2606.24418v1)

**作者:** Behrooz Tahmasebi `[一作]` (Harvard University), Stefanie Jegelka `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

本文对已知对称群作用下的有限维投影估计器（密度估计与回归）进行理论分析，证明使用仅一小部分随机群元素即可在统计上达到全群增强的最优采样复杂度，并探讨了该部分增强在可重用性与完全对称性方面的阈值与限制。

**💡 创新点**

创新点在于将 Fourier 解析与表示论相结合，构造子集平均算子并量化其误差，给出统计最优阈值 |S|≳r/r_inv、可重用阈值 |S|≳(r/r_inv)·log(min{r,|G|})，并证明在足够表达力的假设空间下 exact invariance 必须使用完整群平均，从而揭示近似与精确对称性之间的根本差距。

**🔧 技术方法**

技术手段包括群的 Fourier 变换、不可约表示分解、子集平均算子分析、Carathéodory 定理用于构造加权平均、以及球面例子中的球谐函数核化等。

**📊 数据集**

该工作为理论研究，未使用真实数据集；示例以球面（S^{d-1}) 的均匀分布为实验场景。

**📈 对比分析**

与全群数据增强对比时，论文证明当 |S|≳r/r_inv 时，部分增强能获得与全群相同的 minimax 风险 r_inv/n；在可重用情形下仅多一个 log(min{r,|G|}) 乘子；理论上达到最佳率，未给出数值实验。

**⚠️ 局限性**

主要局限在于只能实现近似对称性，无法获得 exact invariance；可重用性需要额外 log 因子；研究仅针对投影估计器和线性/核模型，对深度网络等复杂结构尚不适用；高维/无限维情形仍需进一步探索。

---

## 329. Poisoned Playbooks: Demystifying Knowledge Poisoning Effects on AI Security Agents

**arXiv ID:** 2606.24402 | [PDF](https://arxiv.org/pdf/2606.24402v1)

**作者:** Juho Park `[一作]` (NAVER Cloud), Kevin Nam `[通讯]` (Kyung Hee University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地研究了知识毒化对基于检索增强生成（RAG）的 AI 安全代理的影响，评估单一毒化“Playbook”在多种模型、挑战和真实 CVE 中的效果。

**💡 创新点**

引入 Verification Boundary（VB）三层分类解释毒化采纳的模式，并展示毒化攻击在零日与信息垄断环境下的系统性风险。

**🔧 技术方法**

结合 RAG 检索、LLM（Claude Opus 4/4.6、GPT‑5.3、Gemini 3.0 Pro）与工具交互的安全代理；采用验证提示和多源检索等缓解策略。

**📊 数据集**

使用 11 个 CTF 挑战（XSS、CSP、SSRF、等）、3 大前沿 LLM 家族、2 代模型、11 个真实 CVE（Jenkins、Log4Shell、Spring4Shell 等）进行实验。

**📈 对比分析**

通过 Poison Adoption Rate（PAR）评估毒化成功率；实验表明 L3 级别毒化在多模型/多生成下保持高 PAR，验证提示和多源检索在有竞争证据时可显著降低 PAR，但在稀缺证据下效果有限。

**⚠️ 局限性**

缺乏针对更复杂代理架构的验证，毒化可能被更高级的动态验证或多源可信度策略克服；实验集中在公开写作风格的毒化，未涵盖更隐蔽或持续性攻击；模型提升可能改变 VB 分类，但不消除所有风险。

---

## 330. Age of LLM: A Strategic 1v1 Benchmark for Reasoning, Diplomacy and Reliability of Large Language Models under Fog of War

**arXiv ID:** 2606.24391 | [PDF](https://arxiv.org/pdf/2606.24391v1)

**作者:** Arnaud Ricci `[一作]` `[通讯]` (Independent researcher), Arnaud Ricci (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现 Age of LLM——一个基于 13×7 地图的 turn‑based 1v1 策略游戏基准，两个 LLM 通过结构化 JSON 动作相互对战，并记录完整回放。

**💡 创新点**

创新点：
- 通过私有引擎、随机地图种子与多模型对手消除公共基准的数据污染；
- 在规则设计中加入雾障、完整外交（消息、停战、和平、最后通牒）和严格动作合法性检查；
- 提供非法动作率、重试统计等新维度衡量 LLM 在部分可观测、连续决策环境中的信念跟踪与可靠性。

**🔧 技术方法**

技术手段：
- 规则驱动的即时战略游戏引擎（私有实现）
- 与多供应商（OpenAI, Anthropic, Google, etc.）的 LLM API 接口
- 结构化输出校验、非法动作判定与重试统计
- 统计分析：点数/回合、Bradley–Terry 排序、相关性检验（非法动作率、思考时长、token/turn）。

**📊 数据集**

数据集：54 场完整比赛，15 种模型共 5,258 次动作，包含所有动作、外交记录与消息，公开回放 JSON（可在 https://ageofllm.org 获取）。

**📈 对比分析**

比较方法与性能：
- 以点数/回合（ppm）和 Bradley–Terry 模型对 15 模型进行排名；
- 统计胜率、非法动作率、思考时长、token/turn。
- 结果显示：核弹冲刺占 78%（规则一致子集）且为主导胜利模式；军事征服虽更快（12.3 回合）但极为稀少；外交几乎不兑现；非法动作率与胜率呈负相关。

**⚠️ 局限性**

限制：
- 样本量小且不平衡，未进行侧交换对战，单次随机采样缺乏复现性；
- 部分版本规则变更与提示中的两条战术语句可能影响结果；
- API 的思考时长与重试受供应商实现影响；
- 早期警报信号未触发，难以分离机械与认知因素；
- 仅在 54 场比赛基础上得出结论，需更多匹配与重复实验以提升统计稳健性。

---

## 331. Jolia: Concept-Level Vision-Language Alignment for 3D CT Contrastive Learning

**arXiv ID:** 2606.24570 | [PDF](https://arxiv.org/pdf/2606.24570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 332. Multilevel Stochastic Plug-and-Play for Sparse-View CT Reconstruction

**arXiv ID:** 2606.24567 | [PDF](https://arxiv.org/pdf/2606.24567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 333. Quantum CT via Dynamic Interval Encoding and Prior-Balanced QUBO Reconstruction

**arXiv ID:** 2606.24561 | [PDF](https://arxiv.org/pdf/2606.24561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 334. PHANTOM: A Large-Scale Dataset of Multimodal Adversarial Attacks for Vision-Language Models

**arXiv ID:** 2606.24388 | [PDF](https://arxiv.org/pdf/2606.24388v1)

**作者:** Simone Gallivanone `[一作]` (Italian Institute of Artificial Intelligence), Nicola Franco `[通讯]` (Italian Institute of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个规模达 47,524 条、涵盖 10 大类 55 小类、7826 个有害意图的多模态对抗攻击数据集 PHANTOM，用于评估视觉‑语言模型的安全性。

**💡 创新点**

创新点包括：①整合并扩展现有攻击库，新增儿童安全类；②提供预生成对抗样本，降低研究门槛；③实现跨模型（开源与闭源）攻击迁移性评估；④使用结构化元数据支持复现。

**🔧 技术方法**

使用多种已验证的攻击策略（BAP、IDEATOR、MML、FC ATTACK、CSDJ）生成样本，并采用 Abel‑24‑HarmClassifier 进行自动判定攻击成功。

**📊 数据集**

采集自 JailBreakV‑28K、MM‑SafetyBench、OmniSafeBench‑MM、SafeBench 等合并的 7826 个有害意图，并在多款开源 VLM（DeepSeek‑VL2、GLM‑4.6V‑Flash、Qwen 系列等）与闭源模型（Claude Opus、GPT‑5.x、Gemini 3.1 Pro）上生成与评估。

**📈 对比分析**

通过白盒与黑盒实验计算攻击成功率（ASR），结果显示不同攻击方法在不同类别的 ASR 差异显著，MML、FC Attack、CSDJ 在多数模型上 ASR 可达 60–90%；跨模型迁移性强，攻击在闭源模型中仍保持 20% 以上成功率。

**⚠️ 局限性**

局限性包括：①评估依赖单一自动判别器，可能误判；②仅针对部分攻击策略和采样子集；③未覆盖全部可能的对抗方法；④语义意图集合可能继承来源数据的偏见。

---

## 335. ComputeFHE: A Privacy-Preserving General-Purpose Computation Library

**arXiv ID:** 2606.24379 | [PDF](https://arxiv.org/pdf/2606.24379v1)

**作者:** Faris Serdar Tasel `[一作]` (Çankaya University), Efe Ciftci `[通讯]` (Çankaya University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出并实现了ComputeFHE开源C++库，支持在TFHE加密下对整数和定点数进行算术、逻辑、条件分支与隐式数组访问操作。

**💡 创新点**

通过引入FHE友好的三输入逻辑门和优化的ALU架构，显著减少了引导（bootstrapping）次数，并提供模拟模式便于快速调试。

**🔧 技术方法**

依赖OpenFHE后端实现TFHE基础门与优化门，结合C++模板与宏实现加密数据类型与隐式数组访问功能。

**📊 数据集**

论文未使用公开数据集，主要通过快速示例（加法、排序）和自定义算法演示库功能。

**📈 对比分析**

对比标准ALU与优化ALU，在相同算法（如加法、排序）下引导次数平均降低40–60%，最快可达3.9×的性能提升。

**⚠️ 局限性**

缺乏浮点数支持，隐式数组访问仍然昂贵，对算法级优化仍有限，需手动编写加速指令。

---

## 336. EERLoss: A Novel Loss Function for Training Deep Biometric Models. A Case Study in Keystroke Dynamics

**arXiv ID:** 2606.24586 | [PDF](https://arxiv.org/pdf/2606.24586v1)

**作者:** Nahuel Gonzalez `[一作]` (University of Buenos Aires), Ruben Tolosana `[通讯]` (BiometricsAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为 EERLoss 的子可微损失函数，能够直接在训练过程中最小化等误率 (EER)，并在 Keystroke Dynamics 验证任务中取得显著性能提升。

**💡 创新点**

创新点：1）构造了可微分的 EER 近似，并通过平滑 FAR/FAR 与子可微二分搜索实现快速求解阈值；2）设计了基于面积的损失形式，使模型不仅关注阈值点，还能有效压缩 FAR 与 FRR 的重叠区域；3）提供了可调的 margin 与 β 参数，兼顾不同任务需求；4）公开实现代码，促进复现。

**🔧 技术方法**

技术：子可微平滑函数（tanh 近似 0/1 阈值）、子可微二分搜索、面积重叠计算、margin/β 调整、GPU 并行实现；训练使用 AdamW，学习率 1e‑4，基于双分支 CNN+RNN+Attention 的 256/512 维嵌入网络。

**📊 数据集**

数据集：KVC‑onGoing benchmark，包含 Aalto Desktop 与 Mobile 两个子集，总计超过 185,000 名用户；实验分为 1,000 用户子集（快速 ablation）和全量训练（桌面 168k、移动 37k）。

**📈 对比分析**

比较方法：与多种 SoTA 损失（Semi‑Hard Triplet、ArcFace、CosFace、Set2Set）在同一网络架构与训练设定下对比；评估指标为 Global EER 与 Avg. per‑user EER，覆盖不同 enrollment 大小 G；结果显示 EERLoss 在桌面任务 Global EER 下降 30.7%，Avg. per‑user 下降 30% 以上；在移动任务中亦取得四个 enrollment 场景的最佳或相近表现；训练时间上，EERLoss 在全量实验中从 92 h 47 min 缩短至 6 h 52 min，速度提升 13.5 倍。

**⚠️ 局限性**

限制：1）仅在 Keystroke Dynamics 验证上进行大规模评估，需进一步验证在其他行为或生理模态（如面部、指纹）上的泛化能力；2）对极高可区分的生理特征（如面部）在充足数据下可能不如角度损失优越；3）对极少样本或高度噪声数据的鲁棒性虽有初步验证，但需更多实验；4）实现仍依赖较大批量计算，可能在资源受限的设备上受限。

---

## 337. Unified Position-Invariant Random Access Through Two Compression Layers via Absolute-Offset Coordinates: A Bit-Perfect Device-Resident Proof

**arXiv ID:** 2606.24531 | [PDF](https://arxiv.org/pdf/2606.24531v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 338. What Do Flow-Based Inverse Solvers Approximate? A Posterior-Transport View

**arXiv ID:** 2606.24516 | [PDF](https://arxiv.org/pdf/2606.24516v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 339. Heterogeneous Knowledge Distillation via Geometry Decoupling and Momentum-Aware Gradient Regulation

**arXiv ID:** 2606.24557 | [PDF](https://arxiv.org/pdf/2606.24557v1)

**作者:** Wuming Yang `[一作]` (Central South University of Forestry and Technology), Hongmin Zhao `[通讯]` (Central South University of Forestry and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 SPOFA 框架，用于解决跨架构（Transformer ↔ CNN/MLP）知识蒸馏过程中出现的训练不稳定问题。

**💡 创新点**

创新点在于双重稳定机制：① 在特征层使用 LayerNorm 对特征幅度与方向进行显式解耦，消除特征尺度差异导致的表征瓶颈；② 在梯度层引入 Momentum‑driven EMA (MEMA) 动态标量，对梯度冲突进行历史基线评估并按需衰减蒸馏信号，从而抑制优化冲突。

**🔧 技术方法**

技术包括：LayerNorm 解耦投影器、MEMA 记忆衰减尺度器、OFA 统一投影结构、梯度相似度计算、CKA 对齐度量、Grad‑CAM 可视化。

**📊 数据集**

数据集：CIFAR‑100（中等规模）和 ImageNet‑1K（大规模）用于评估不同教师‑学生对的泛化与稳定性。

**📈 对比分析**

与传统 KD（FitNet、RKD、CRD）、响应蒸馏（KD、DKD、DIST）以及异构蒸馏基线 OFA 和最新 SOTA PAT 进行对比。SPOFA 在 12 种教师‑学生组合中，11/12 场景超越 OFA，9/12 场景超越 PAT，同时训练参数量与计算开销几乎不变（仅 +0.01M 额外参数），在 ImageNet‑1K 也实现了 3/4 对比组合的 SOTA 级别精度。

**⚠️ 局限性**

局限性：① 对超大规模或极端不匹配的模型仍可能略逊于 PAT（如某些 ViT→MLP 组合）；② MEMA 的超参数（α、λ）需要在一定范围内调优，虽然鲁棒性高但对极端噪声仍有敏感性；③ 目前仅验证在图像分类任务上，对目标检测、分割等密集预测任务的适用性尚待探索。

---

## 340. AGORA: An Archive-Grounded Benchmark for Agentic Workplace Document Reasoning

**arXiv ID:** 2606.24526 | [PDF](https://arxiv.org/pdf/2606.24526v1)

**作者:** Honglin Guo `[一作]` (Fudan University), Qi Zhang `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个跨域、基于档案的代理式文档推理基准Agora，包含362个可验证数值问题，覆盖8个专业领域，需在大规模文件集内进行文件探索和计算

**💡 创新点**

创新点在于同时强调档案根基、代理式探索和跨域覆盖，构建完整的任务合成、泄漏防护、难度过滤与人工验证流水线，并提供可自动验证的数值答案

**🔧 技术方法**

采用深度搜索、向量检索、LLM驱动的任务生成与改写、自动化难度筛选、以及bash工具调用的代理框架

**📊 数据集**

使用来自8个职业领域的真实工作档案，共9,664份文档，372M tokens，涵盖PDF、Markdown、Excel/CSV等多格式数据

**📈 对比分析**

在8种模型（GPT‑5.5、Gemini‑3.1‑Pro/Flash‑Lite、DeepSeek‑V4‑Flash/Pro、GLM‑5.1、Qwen3.5‑35B‑A3B/9B）上测试，最强模型仅达59.4%准确率，模型之间在各领域存在显著差异，形成两大性能梯度层级

**⚠️ 局限性**

局限包括：仅覆盖8个领域且任务量有限；闭书过滤可能被更大模型预训练数据覆盖；基准采用极简的bash工具包，实际部署效果受框架影响

---

## 341. Generating Realistic Individual Activity Schedules via Activity Location Allocation Based on Simulated Travel Times

**arXiv ID:** 2606.24566 | [PDF](https://arxiv.org/pdf/2606.24566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 342. Red-Teaming the Agentic Red-Team

**arXiv ID:** 2606.24496 | [PDF](https://arxiv.org/pdf/2606.24496v1)

**作者:** Dario Pasquini `[一作]` (Cracken), Artem Sorokin `[通讯]` (Cracken)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对当前主流的 agentic offensive‑security 工具（共 12 个开源实现）进行系统安全评估，识别出共享的设计缺陷，并构建完整的攻击链模型。

**💡 创新点**

首次提出基于 LLM 的隐式操纵技术（无 Prompt‑Injection），引入“agentic kill chain”框架，并提出以最小权限与隔离为核心的安全体系架构，解决了传统 guardrail 失效的问题。

**🔧 技术方法**

使用 LLM 操作、容器沙箱逃逸、网络与文件系统隔离、自动化评估管线等技术；实验基于 6 个前沿 LLM（OpenAI GPT‑4、Claude、Gemini 等）与 3 个自定义 honeypot 载荷。

**📊 数据集**

实验数据集包含：12 个 agentic 工具的默认配置；6 种前沿 LLM；3 种精心设计的攻击载荷（二进制工具、加密容器文件、内存漏洞样本）。

**📈 对比分析**

通过统一的威胁模型和自动化脚本对 12 个工具进行对比，发现 97.8% 的试验能在 worker 上实现 RCE，10/12 工具可逃逸宿主，8/12 可在宿主上实现完整 RCE，11/12 可泄露 API 密钥或操作日志，证明现有工具的安全缺陷普遍且严重。

**⚠️ 局限性**

局限性：实验基于开源实现和可访问的 LLM，未覆盖私有或硬件隔离方案；攻击模型未考虑更高级的防御（如 LLM 级安全审计、动态模型更新）；缺乏在真实生产环境中大规模部署的验证；对 soft‑persistence 攻击的完全防护仍是未解挑战。

---

## 343. Spatial Partial Functionalization of Neural Networks based on Noise Fields

**arXiv ID:** 2606.24588 | [PDF](https://arxiv.org/pdf/2606.24588v1)

**作者:** Shuhei Ikemoto `[一作]` (Kyushu Institute of Technology), Fabio DallaLibera `[通讯]` (Osaka University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过设计噪声调制神经网络（Noise‑modulated Neural Network）并引入虚拟噪声场机制，实现网络在空间上部分功能化，证明单一网络可以在不同噪声场激活下学习并推理多种目标函数。

**💡 创新点**

创新点包括：① 将噪声视为功能激活源而非扰动，提出跨越激活函数（crossing activation function）并给出其样本级、统计级、解析级三种实现；② 引入可控的噪声场与虚拟噪声场，用截断高斯生成空间结构噪声来定义可激活子网络；③ 证明噪声场的空间结构与目标函数的相似性匹配时可显著提升网络的内存容量。

**🔧 技术方法**

主要技术手段：噪声调制神经网络结构；跨越激活函数三实现（样本、统计、解析）；截断高斯生成的虚拟噪声场与网络噪声场映射；全连接前馈网络；Adam优化器与MSE损失函数；参数迁移实验验证实现间可共享参数。

**📊 数据集**

使用自生成的 1D 正弦函数族作为目标数据集，覆盖不同相位、频率组合，共 1000 个均匀采样点；所有实验均在单一网络上训练多达数百个不同函数。

**📈 对比分析**

通过比较有序与打乱的噪声场分布、噪声场维度（V=1 或 2）以及函数空间维度，评估最大均方误差（ℒ_max）随函数数量的变化。结果显示：当噪声场维度与函数相似性维度匹配且噪声场保持有序时，ℒ_max 维持在低水平，可支持更多函数；而噪声场结构不匹配或打乱时，ℒ_max 快速升高，表明内存容量下降。

**⚠️ 局限性**

局限性包括：仅在小型全连接网络和 1D 正弦函数上验证，未评估更大规模或高维任务；噪声场的网格构造受整数分解限制，可能导致维度失衡；噪声场被视为静态参数，未考虑其随学习或外部环境动态变化；对不同噪声分布或更复杂激活函数的泛化性尚未系统探究。

---

## 344. LLMs Prompted for Legal Context Object More: Overrefusal from Small On-Premises LLMs in Criminal Legal Context

**arXiv ID:** 2606.24585 | [PDF](https://arxiv.org/pdf/2606.24585v1)

**作者:** Anastasiia Kucherenko `[一作]` (HES SO Valais Wallis), Andrei Kucharavy `[通讯]` (HES SO Valais Wallis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在法律情境下使用小型本地LLM时，权威前缀如何导致过度拒绝。

**💡 创新点**

首次系统评估了不同法律角色前缀对小型LLM拒绝率的影响，并发现权威前缀比jailbreak更易触发拒绝。

**🔧 技术方法**

采用小型开源指令模型（LLaMA、Gemma、Qwen3、Apertus）在本地部署，并使用关键词匹配检测拒绝。

**📊 数据集**

使用OR‑Bench 80K的五类法律相关毒性提示以及30份真实法律文件。

**📈 对比分析**

通过在不同前缀、语言和模型上统计拒绝次数，发现权威前缀在英语、法语中提高2–20倍拒绝率，实测结果与基线相差显著。

**⚠️ 局限性**

局限在样本量有限、拒绝检测仅靠关键词、未测试真实有害输入，且模型规模和语言覆盖有限。

---

## 345. Enabling Robust Cloth Manipulation via Inference-Time Simulator-in-the-Loop Refinement

**arXiv ID:** 2606.24552 | [PDF](https://arxiv.org/pdf/2606.24552v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 346. On the Smallness of the Large Language Models Scaling Exponents

**arXiv ID:** 2606.24504 | [PDF](https://arxiv.org/pdf/2606.24504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 347. Decentralized Pose Graph Riemannian Optimization for Object-based Multi-Robot SLAM

**arXiv ID:** 2606.24489 | [PDF](https://arxiv.org/pdf/2606.24489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 348. FirmCure:Towards Autonomous and Adaptive Rehosting of Linux-Based Firmware

**arXiv ID:** 2606.24549 | [PDF](https://arxiv.org/pdf/2606.24549v1)

**作者:** Chuan Hong `[一作]` (National University of Defense Technology), Peihong Lin `[通讯]` (National University of Defense Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出并实现了一个 LLM 驱动的全系统固件重托管框架，能够自动化完成固件的初始化、启动、运行时错误诊断与修复，最终实现对 Linux‑based IoT 固件的可交互运行。

**💡 创新点**

创新点包括：① 采用 LLM 进行语义感知、反射合成与自治运行时干预的三阶段闭环管道；② 多智能体协同处理硬件依赖、启动障碍与运行时错误；③ 在感知阶段实现对硬件依赖的语义解耦，在反射阶段自动校正 QEMU 参数与文件系统；④ 在自治运行时通过专门的 CrashExpert、FileExpert、WebExpert 等专家智能体实现动态错误修复。

**🔧 技术方法**

技术手段：LLM（GLM‑5.1 等）+ CrewAI 多智能体框架；QEMU 全系统仿真；radare2 静态分析、GDB 调试；文件系统修复、网络检查工具；LLM Prompt Engineering 与 JSON 结构化输出；知识库驱动的错误分类与修复策略。

**📊 数据集**

使用 21 个代表性 IoT 固件镜像，来自 10 家主要厂商，覆盖 5 种 CPU 架构（MIPS、ARM、ARM64 等），包括路由器、摄像头等设备。

**📈 对比分析**

与 FirmAE、Greenhouse、Firmwell 等基准工具按四级指标（Kernel Boot、Network、Port、Interactivity）对比；本框架实现 100% 的网络端口激活率、90.5% 的服务交互率，显著超过现有方法；同时复现 10 个已知 CVE，发现 5 个新漏洞，证明其实用安全价值。

**⚠️ 局限性**

局限性：只能使用可替换的通用内核，无法分析依赖专有硬件或特定内核特性的漏洞；LLM 可能产生幻觉或循环推理，对高度定制的固件仍需较长时间且未必在 45 分钟内完成；缺乏针对复杂固件格式（UBI/UBIFS 等）的自动解包支持；对极端复杂的定制化固件仍需人工介入。

---

## 349. UOL@IDEM at BEA 2026 Shared Task 1: Neural Fusion and Feature-Rich Modeling for L1-Aware Vocabulary Difficulty Prediction

**arXiv ID:** 2606.24501 | [PDF](https://arxiv.org/pdf/2606.24501v1)

**作者:** Nouran Khallaf `[一作]` (University of Leeds), Serge Sharoff `[通讯]` (University of Leeds)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并提交了一个闭轨系统，用于预测西班牙语、德语和中文学习者对英语词汇的难度，结合多语言上下文编码和手工特征。

**💡 创新点**

创新点在于将多语言句子嵌入模型与频率、语义、同源相似度、检索与 MLM 可预测性等工程特征融合，并发现频率是最稳定的 L1 相关信号。

**🔧 技术方法**

使用技术包括神经融合（neural fusion）架构、BGE‑M3、multilingual‑E5 等句子嵌入模型，以及 MLM、惊奇度、检索特征、语义标签、同源相似度等手工特征。

**📊 数据集**

使用的数据集为 BEA 2026 共享任务的 L1‑aware Vocabulary Difficulty 数据集（KVL），包含英语目标词、L1 上下文和 GLMM 校准难度分。

**📈 对比分析**

与官方闭轨基线通过 RMSE、Pearson、Spearman、Kendall 进行比较，开发集上神经融合提升 RMSE 0.21–0.26，测试集上获得 RMSE 1.132（西班牙）、1.037（德语）、0.891（中文）。

**⚠️ 局限性**

局限性包括仅使用闭轨资源，未利用外部学习者语料或大型 LLM；特征工程增加流水线复杂度；模型在最易和最难区间的校准偏差（过度预测易项，略低估难项）。

---

## 350. GeoIMO: Geometry-Driven Independent Motion Classification for Event Cameras

**arXiv ID:** 2606.24499 | [PDF](https://arxiv.org/pdf/2606.24499v1)

**作者:** Anil Bayram Gogebakan `[一作]` (Politecnico di Torino), Stefano Di Carlo `[通讯]` (Politecnico di Torino)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于几何的、无需人工标注的框架，利用事件相机的全局运动估计（FOE + yaw补偿）对检测框进行静止或独立运动的二分类。

**💡 创新点**

创新点在于：①通过对事件流进行对比度最大化获得低维全局运动参数，避免密集光流估计；②在检测框内部独立估计局部运动，并与全局运动做尺度不变残差比较，得到无监督的运动标签；③使用yaw补偿提升转弯时的鲁棒性；④方法无需学习，适用于任何检测框，兼容传统基于帧的检测框。

**🔧 技术方法**

主要技术包括：事件相机数据处理、FOE与yaw参数化的几何运动模型、对比度最大化的全局与局部运动估计、Kalman式时间平滑、尺度不变残差阈值分类。

**📊 数据集**

在MVSEC（包含灰度图和事件流）和Prophesee 1 MP汽车检测数据集上进行实验；由于原始数据集未标注运动状态，论文作者先手工标注了静止/运动框作为评估基准。

**📈 对比分析**

与不同运动模型（径向、平移）以及是否使用yaw补偿的组合进行对比。结果显示：在MVSEC outdoor_day2中，全局yaw+径向模型可将宏观F1提升至约0.56；在Prophesee子集里，yaw补偿在转弯场景中显著提升宏观F1（+0.02~0.05），径向模型在计算开销上略高但效果不显著优于平移模型；全局参数设置已能达到竞争水平，单视频最优调参进一步提升至宏观F1≈0.61。

**⚠️ 局限性**

局限性：①类别不平衡导致运动框的F1偏低；②当事件密度低或噪声高时，局部运动估计不稳定；③径向模型受框大小限制，实际收益有限；④方法依赖已有检测框，未解决检测与运动分离的端到端问题；⑤yaw补偿在直线行驶时几乎无效，且对复杂旋转运动的鲁棒性仍待提升。

---

## 351. GUI vs. CLI: Execution Bottlenecks in Screen-Only and Skill-Mediated Computer-Use Agents

**arXiv ID:** 2606.24551 | [PDF](https://arxiv.org/pdf/2606.24551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 352. New convergence results for Carleman linearization

**arXiv ID:** 2606.24473 | [PDF](https://arxiv.org/pdf/2606.24473v1)

**作者:** Michele Boreale `[一作]` (Università di Firenze), Luisa Collodi `[通讯]` (Università di Firenze)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了针对多项式常微分方程有限 Carleman 拓扑截断的全新误差上界，能够在给定观测量（如状态坐标）下显式估计截断误差并保证收敛。

**💡 创新点**

创新点在于：①直接在原始单项式基底上分析，保持多项式结构；②利用 Dyson–Duhamel 展开把线性与非线性部分分离；③使用对数范数捕捉线性动力学的收敛性，从而得到更紧的、阶数感知的误差估计。

**🔧 技术方法**

主要技术包括：Carleman 嵌入、Dyson–Duhamel 级数展开、对数范数估计、支撑度数追踪、以及对非线性度数提升的逐步计数。

**📊 数据集**

未使用公开数据集；通过理论推导和两组经典例子（Stuart–Landau 与 Van der Pol）进行数值验证，截断阶数为 16（M=150 词项）。

**📈 对比分析**

与 Forets–Pouly、Amini‑Zheng‑Sun‑Motee 等现有方法相比，得到更低的误差上界，收敛半径更大，尤其在存在线性收敛的系统中显著减少“膨胀因子”。

**⚠️ 局限性**

局限性：仍需事先给定轨迹上界 R；仅适用于多项式系统；在 μ₁≥0 且系统非弱非线性时误差仍可能保守；对高度非线性或高维系统的可扩展性尚待进一步验证。

---

## 353. Discrepancy for Random Linear Codes

**arXiv ID:** 2606.24471 | [PDF](https://arxiv.org/pdf/2606.24471v1)

**作者:** Dean Doron `[一作]` (Ben-Gurion University), João Ribeiro `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本篇论文研究了随机线性码在距离容量（即比容量更高的编码率）下的分歧（discrepancy）性质，证明了它们在几乎所有 Hamming 球和组合矩形中的交集大小几乎等于期望值，并把这些结果应用于列表解码、零误差列表恢复以及泄漏鲁棒线性共享方案；

**💡 创新点**

主要创新在于提出并证明了两条全局分歧定理：一种控制单一测试函数所有平移的偏差，另一种控制满足傅里叶弱相关条件的大规模测试族的偏差；这使得随机线性码在容量以上的表现与完全随机码匹配，且推导了此前未能突破的泄漏鲁棒共享阈值上限；

**🔧 技术方法**

核心技术是基于随机生成基向量的迭代构造，结合二阶矩方法与自卷积（averaged convolution）来控制偏差的两步变化；引入“平滑性”（smoothness）概念，通过一次性证明一个共享自卷积函数的平滑性，进而传递到所有平移；同时利用傅里叶分布的浓度特性、双重算子与潜在函数方法；

**📊 数据集**

论文为纯理论研究，不使用实验数据集；所有结论均通过概率论与组合论证明而非经验验证；

**📈 对比分析**

与现有最优随机码结果相比，论文在容量以上实现了与非结构化随机码相同的指数级偏差界；在列表恢复方面，获得了比先前仅有的覆盖半径结果更强的多项式上界；在泄漏鲁棒共享上，阈值间隙从之前的 0.5n 降低到约 (2/ log q) n，显著提升；

**⚠️ 局限性**

局限性包括：（1）仅在固定或轻微增长的域上可直接应用，域大时需额外 MDS 假设；（2）结果是随机存在性高概率结论，缺乏构造性（explicit）码；（3）对二误差列表恢复与共享方案的扩展需要满足特定的傅里叶弱相关约束，可能不适用于所有测试族；（4）对平衡泄漏函数的泛化仍有限，非平衡情况尚未覆盖；

---

## 354. Are Text-to-Image Models Inductivist Turkeys? A Counterfactual Benchmark for Causal Reasoning

**arXiv ID:** 2606.24548 | [PDF](https://arxiv.org/pdf/2606.24548v1)

**作者:** Jiayi Lei `[一作]` (Shanghai Jiao Tong University), Yongsheng Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估文本到图像模型在对抗现实世界先验的反事实推理能力，构建三层进阶评测框架 CF-World 并提出 CF-Eval 评估器

**💡 创新点**

创新在于设计针对反事实因果推理的三层级对照基准、引入 Prior Resistance Rate 与 Reasoning Retention Rate 两个度量，以及系统化的视觉语言模型自动评测流程

**🔧 技术方法**

采用视觉语言模型 VLM（Qwen3-VL-235B、Gemini-3-Pro）、大语言模型生成提示、以及 Diffusion 与统一多模态模型的生成评估

**📊 数据集**

使用自构造的 CF-World 基准，共 1,091 组 3,273 条提示，覆盖物理、生物、化学、地理、社会学等五大学科

**📈 对比分析**

通过与多款开源/闭源 T2I 模型（FLUX.2-dev、Nano Banana Pro 等）在 L1–L3 三层级进行对比，发现闭源模型在 L2/L3 维持较高 PRR 与 RRR，开源模型普遍存在 Prior Lock‑in 与 Reasoning Bottleneck，性能显著下降

**⚠️ 局限性**

局限在于仅为诊断性研究，未提供解决方案；评测依赖 VLM 的自动判断，可能受限于评测模型的主观性；基准聚焦于中学水平科学规律，尚未覆盖更高层次推理

---

## 355. ForensicsTok: Forensics-Guided Tokenized Modeling for Image Tampering Localization

**arXiv ID:** 2606.24538 | [PDF](https://arxiv.org/pdf/2606.24538v1)

**作者:** Lei Xu `[一作]` (Shenzhen University), Changsheng Chen `[通讯]` (Shenzhen MSU-BIT University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ForensicsTok，一种基于多模态大型语言模型的图像篡改检测与定位框架，将定位任务转化为自回归token序列生成；

**💡 创新点**

创新点包括：①Token Splatting Decoder (TSD)实现对二值掩码token的直接监督与代码库平滑；②Hierarchical Expert Fusion (HEF)在视觉编码器中多尺度注入专业取证模型特征；③通过代码库解码器消除外部分割器带来的信息瓶颈与语义不匹配；

**🔧 技术方法**

使用技术：内部自回归token生成、预训练代码库解码器、代码库感知标签平滑、注意力门控的多尺度特征融合、标准的IoU/F1评估以及多种扰动下的鲁棒性测试；

**📊 数据集**

使用的数据集：训练集包括TamperCOCO、MIML、CASIA2、SID_Set；评估集为CASIA1、NIST、Coverage、Columbia、Glide、IMD；检测实验在SID-Set上进行；

**📈 对比分析**

对比方法包括多种专家模型（CAT-Net、MVSS-Net、PSCC-Net、TruFor、SparseViT）和现有MLLM基线（FakeShield、SIDA）。ForensicsTok在所有六个定位基准上平均IoU 0.68、F1 0.78，较SparseViT提升0.04/0.02，较FakeShield提升0.33/0.32，且在鲁棒性测试中保持最高F1；

**⚠️ 局限性**

局限性：与轻量级取证专家相比参数量仍高、推理延迟较大；对强JPEG压缩等极端扰动仍有性能下降；需要进一步验证对更广泛AI生成篡改场景的适应性。

---

## 356. Governed Shared Memory for Multi-Agent LLM Systems

**arXiv ID:** 2606.24535 | [PDF](https://arxiv.org/pdf/2606.24535v1)

**作者:** Yanki Margalit `[一作]` (Caura.ai), Oded Margalit `[通讯]` (Ben Gurion University Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套面向多智能体 LLM 系统的受控共享内存架构，并在 MemClaw 生产服务中验证其正确性。

**💡 创新点**

创新点在于将 fleet‑memory 视作治理的分布式状态，明确四类失败模式并给出作用域、时间、来源、传播等原语；同时提供了可复现的 ArgusFleet 测试框架。

**🔧 技术方法**

使用 REST API 的 MemClaw 内存服务、基于 Python 的 ArgusFleet harness、以及基于 RDF/JSON 的元数据结构实现治理、冲突检测和可追溯性。

**📊 数据集**

使用由 ArgusFleet 自动生成的合成工作负载（约 200 条冲突写、50 条来源链、40 条传播测试等）而非真实业务数据。

**📈 对比分析**

通过四个实验分别量化泄漏率、来源完整性、冲突检测率和传播可见性；实验显示来源链恢复率 100%，同群可见性 97.5%，写入可见窗口 0.83s，冲突检测成功率 50%（全写入成功时 100%）且写入延迟在负载下最高 19s。

**⚠️ 局限性**

局限性包括仅在单租户环境下验证、缺乏跨租户或基准比较、使用合成数据、对冲突检测的异步顺序与 dedup 阻塞交互不完全解耦，以及对高并发写入尾部延迟的探索不足。

---

## 357. Reinforcement Learning for Computer-Use Agents with Autonomous Evaluation

**arXiv ID:** 2606.24515 | [PDF](https://arxiv.org/pdf/2606.24515v1)

**作者:** Marta Sumyk `[一作]` (Ukrainian Catholic University), Oleksandr Kosovan `[通讯]` (Ukrainian Catholic University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种利用自主视觉‑语言评估器对计算机使用代理（CUA）进行强化学习微调的方法，利用评估器给出的二元成功判断作为奖励，并对评估器噪声进行建模与校正；

**💡 创新点**

创新点在于将评估器误判（假正例/假负例）视为可估计的噪声源，推导出一个无偏的奖励估计器，并将其直接嵌入PPO框架，从而在缺乏人工标签的场景下提供可靠的奖励信号；

**🔧 技术方法**

使用的方法包括：视觉‑语言评估器（Qwen2‑VL‑7B）对终止截图与指令进行二分类；基于估计的误差率构造奖励校正公式；使用PPO对CUA策略进行微调；以及对不同操作系统分别训练或统一训练模型；

**📊 数据集**

所用数据集包括：自构造的 7,560 条跨 42 个应用、3 个操作系统的任务集合（macOS、Windows、Linux），以及 OmniAct、GUI‑World、GUIDE 三个公开 GUI 交互数据集；评估器的校准采用每个 OS 的一部分任务；

**📈 对比分析**

对比方法包括：零射手基线、使用原始评估器奖励的PPO、使用校正奖励的PPO，分别在统一跨 OS 模型与单 OS 模型下实验。结果显示：校正奖励在 macOSWorld、Windows Agent Arena、OSWorld 上的成功率分别提升约 12.6%、5.1% 等，显著优于零射手基线与原始奖励；

**⚠️ 局限性**

局限性包括：评估器误差率可能随任务、UI 复杂度、语言表述及系统版本变化而非静态，需要更细粒度的误差估计；奖励稀疏且主要是终止奖励，难以解决长期任务的探索问题；校准样本有限时估计不稳导致修正奖励方差增大；评估器可能产生视觉盲点，忽略部分语义正确但视觉不明显的结果。

---

## 358. VistaRef: Boosting Visual Spatial Orientation Awareness for Pointing-to-Object Detection

**arXiv ID:** 2606.24498 | [PDF](https://arxiv.org/pdf/2606.24498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 359. RetiSEM: Generalising Causal Models for Fragmented Biomedical Data

**arXiv ID:** 2606.24488 | [PDF](https://arxiv.org/pdf/2606.24488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 360. Exact Evaluation of Probabilistic Programs with Cylindrical Algebraic Decomposition

**arXiv ID:** 2606.24514 | [PDF](https://arxiv.org/pdf/2606.24514v1)

**作者:** Fredrik Dahlqvist `[一作]` (Queen Mary University of London), Niki Omidvari `[通讯]` (Queen Mary University of London)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

计算小型随机输入程序（如传感器数据处理）的精确输出分布。

**💡 创新点**

将输出分布求解转化为圆柱代数分解（CAD）问题，再结合符号与数值积分，形成适用于少变量、算术操作、分支语句的嵌入式程序的完整分析框架。

**🔧 技术方法**

使用圆柱代数分解、符号积分、数值积分以及概率程序的解析推理技术。

**📊 数据集**

文献中的浮点算术基准以及开源传感器库中的小型程序实例。

**📈 对比分析**

与传统数值估计/Monte Carlo方法对比，实验结果表明该方法能在所选基准上得到完全精确的分布，计算时间在可接受范围内，并且在精度与效率上优于纯模拟。

**⚠️ 局限性**

适用范围受限于变量数少且算术简单的程序；CAD的指数级复杂度限制了可处理的规模，对高维随机输入或非连续分布的扩展存在困难。

---

## 361. A Fair Evaluation of Graph Foundation Models for Node Property Prediction

**arXiv ID:** 2606.24509 | [PDF](https://arxiv.org/pdf/2606.24509v1)

**作者:** Oleg Platonov `[一作]` (HSE University), Liudmila Prokhorenkova `[通讯]` (Yandex Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对9种最新图形基础模型（GFMs）在节点属性预测任务上进行公平评估，并与强基线图神经网络（GNN）进行对比。

**💡 创新点**

提出了统一的评估框架，使用GraphLand真实世界数据集，对PFN（Prior-data Fitted Networks）范式的GFMs与传统GNN进行系统性对比，发现仅PFN基模型在预测性能上能超越精调GNN，但推理成本更高。

**🔧 技术方法**

使用了Prior-data Fitted Networks（PFN）方法的GFMs、改进版GNN（如改进的GCN、GraphSAGE、GAT、LGT），以及在训练与推理过程中进行的广泛超参数搜索与标准深度学习构件。

**📊 数据集**

采用GraphLand基准的10个节点属性预测数据集，涵盖多行业应用场景。

**📈 对比分析**

通过平均排名、归一化得分等指标对模型进行比较；结果显示非PFN GFMs普遍落后于精调GNN，而PFN GFMs（尤其是GraphPFN细调模式）在大多数数据集上位居榜首，取得最佳整体成绩。

**⚠️ 局限性**

PFN模型在推理时间和内存占用上显著高于GNN，限制了其在实时工业场景中的应用；此外，评估未覆盖多任务或极大规模图的扩展性。

---

## 362. video-SALMONN-R$^3$: Learning to ReWatch, ReAsk, and ReAnswer for Efficient Video Understanding

**arXiv ID:** 2606.24477 | [PDF](https://arxiv.org/pdf/2606.24477v1)

**作者:** Yixuan Li `[一作]` (Tsinghua University), Chao Zhang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了video‑SALMONN‑R³，一种端到端的视频大语言模型，能够通过强化学习实现局部重观看（re‑watch）并在两阶段流程中高效利用视觉信息；

**💡 创新点**

核心创新是：①无冷启动链式思考（CoT）训练，仅用RL学习重观看策略；②引入“re‑answer”机制，让模型先给出初始答案并在重观看后进行修正；③使用“re‑ask”在第二次观看时重新注入问题，解决因因果注意力导致的问题；

**🔧 技术方法**

技术上使用：Qwen3‑VL 8B 视觉+文本LLM，Whisper‑Large‑v3 语音编码+Q‑Former对齐，DYNAMIC SAMPLING POLICY OPTIMIZATION（DAPO）强化学习，rule‑based 5项奖励（准确度、格式、修正等），并结合低秩适配（LoRA）进行多模态预训练；

**📊 数据集**

数据集涵盖多模态视频：LibriSpeech、CommonVoice、WavCaps、AudioCaps、LLaVA‑Video‑178k（音视频字幕），训练时选取CinePile与CG‑Bench；评测使用六大基准：VideoHolmes、DailyOmni、AVUT、OmniVideoBench、VideoMME、LVOmniBench；

**📈 对比分析**

与同规模音视频LLM（Qwen‑2.5‑Omni、Qwen‑3‑VL、AV‑Caption‑Base、QA‑SFT）以及多代理与单模型对比方法相比，video‑SALMONN‑R³在所有短中长视频任务上均提升1–3点以上，尤其在长视频（VideoMME、LVOmniBench）提升幅度更显著；

**⚠️ 局限性**

局限性包括：①缺乏对推理文本与定位区间的可解释对应；②仅在多选QA上验证，开放式QA需更复杂奖励；③一次定位不足以覆盖分散多段证据，需多轮重观看。

---

## 363. Cross-Lingual Exploration for Parametric Knowledge

**arXiv ID:** 2606.24579 | [PDF](https://arxiv.org/pdf/2606.24579v1)

**作者:** Elisha Diskind `[一作]` (Hebrew University of Jerusalem), Omri Abend `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了跨语言提示策略，构建了四维设计空间并在多语言事实检索中验证其有效性。

**💡 创新点**

首次系统化定义跨语言探索的四个维度，并证明自主语言选择和多路径聚合能显著提升知识迁移、事实召回和一致性。

**🔧 技术方法**

通过多语言提示、路径搜索、聚合投票、预算控制等推理技术实现跨语言探索，并使用LLM-as-a-judge评估答案。

**📊 数据集**

使用 ECLeKTic（语言特定知识迁移）和 CLIKE（跨语言事实编辑）两个多语言基准。

**📈 对比分析**

与本地语言基线、单路径推理以及多路径策略对比，跨语言探索在 ECLeKTic 上提升 21% 知识迁移，在 CLIKE 上提升约 17% 事实召回，并在一致性上实现显著提升。

**⚠️ 局限性**

评估依赖 LLM 判定，可能存在误判；仅覆盖两种事实基准，未考虑更复杂跨语言任务；实验基于专有模型，可能受版本或服务变更影响。

---

## 364. Quant Convergence: Bridging Classical Value Investing and Modern Factor Models for Systematic Equity Selection

**arXiv ID:** 2606.24575 | [PDF](https://arxiv.org/pdf/2606.24575v1)

**作者:** Augusto Eiji Yamazaki `[一作]`, Hugo Garrido-Lestache Belinchon `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究通过构建基于Benjamin Graham七条防御性筛选规则、现代量化因子以及两者混合的三组特征，利用随机森林、XGBoost、AutoGluon等算法，在过去20年的S&P 500数据上进行后向测试，以评估经典价值规则能否充当低通滤波器抑制过度拟合；

**💡 创新点**

创新点在于将Graham经典价值筛选规则数字化并与现代量化因子共同训练机器学习模型，系统证明在高复杂模型中加入严格的基本面约束能够提升风险调整后收益，并揭示传统价值规则本身即是有效的正则化工具；

**🔧 技术方法**

所用技术包括随机森林、XGBoost、AutoGluon自动机器学习、Optuna超参优化、TimeSeriesSplit交叉验证、Benford法则、Calmar Ratio、Welch t检验等；

**📊 数据集**

数据集为20年S&P 500成分股每日调整后价格及对应财报数据（使用yfinance抓取），并对缺失值进行裁剪与中位数填补；

**📈 对比分析**

采用80/20时间分割、严格买入持有策略，比较各模型在2022‑2026四年测试期的总回报、最大回撤与Calmar Ratio，结果显示纯Graham随机森林总回报最高（232.13%），最大回撤最低，Calmar最高；AutoGluon回报高但回撤最大；现代量化随机森林表现一般；混合模型介于两者；

**⚠️ 局限性**

局限性主要体现在对传统财务指标（如P/B）的依赖，导致对现代科技股的排斥；实验仅覆盖S&P 500，缺乏跨市场验证；统计显著性仅在10%水平；以及模型在不同宏观周期下可能表现不稳定。

---

## 365. PatternGSL: A Structured Specification Language for Template-Free and Simulation-Ready 3D Garments

**arXiv ID:** 2606.24564 | [PDF](https://arxiv.org/pdf/2606.24564v1)

**作者:** Zhenyang Li `[一作]` (University of Hong Kong), Yifan Peng `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个结构化、模板自由且可学习的服装规范语言PatternGSL，并通过视觉‑语言模型实现单张图像直接预测完整的裁缝图，可直接用于物理仿真与编辑；

**💡 创新点**

创新点包括：① 设计了可学习的裁缝规范语言GSL，显式编码板块、曲线与缝合拓扑；② 用Vision‑Language模型（Qwen3‑VL）完成图像到GSL的序列生成；③ 构建了首个包含300K图像‑GSL对的公开数据集；④ 采用确定性解码实现无优化、快速的仿真与编辑；

**🔧 技术方法**

技术实现包括：视觉‑语言模型（Qwen3‑VL）+ NanoBanana视图合成；JSON格式的可编辑裁缝规范；前后视图辅助生成；两阶段训练（预训练+微调）；确定性解码器与规则校正；物理仿真（NVIDIA Warp）；

**📊 数据集**

使用的数据集为PatternGSL数据集，总计300K样本（250K合成+50K真实化图像），每个样本包含完整裁缝图注释；此外还利用GarmentCodeData、NanoBanana以及评估用的真实照片；

**📈 对比分析**

通过与SewFormer、GarmentX、GarmentImage、LGM等基线进行比较，结果显示：2D Chamfer下降至5.78mm（基线30+mm），IoU提升至86.34%，缝合准确率达98.48%；3D仿真成功率99.2%（基线仅4.7%），整体性能显著优于现有方法；

**⚠️ 局限性**

局限性包括：仅在2–37面板范围内验证，复杂长链缝合或高面板数需更强长上下文生成；开放边界或动态行为未覆盖；前后视图合成不一致时仍可能出现缝合或拓扑错误；

---

## 366. Development of a Programming Based Kinetic Model for Two Stage Composting of Solid Waste

**arXiv ID:** 2606.24556 | [PDF](https://arxiv.org/pdf/2606.24556v1)

**作者:** Zarif Tanzim Aziz `[一作]` (Khulna University of Engineering & Technology), Quazi Hamidul Bari `[通讯]` (Khulna University of Engineering & Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了基于Python的两阶段固体废物堆肥动力学模型，并实现了图形用户界面进行模拟与可视化。

**💡 创新点**

创新点在于将原Excel数学模型转换为可编程、可维护、可扩展的两阶段堆肥仿真框架，并提供用户友好的操作界面。

**🔧 技术方法**

使用Python语言、Tkinter GUI、NumPy进行数值计算、Matplotlib绘图等技术。

**📊 数据集**

使用Bari & Koenig（2012）模型的参数设定（如初始温度、湿度、C/N比等）以及典型孟加拉国城市固废量数据作为输入。

**📈 对比分析**

通过在相同输入条件下将模型输出与原Excel模型结果进行对比，发现图表与数值保持一致，证明模型可靠；在大规模模拟时，Python实现速度更快、易于维护。

**⚠️ 局限性**

局限性在于仅基于Bari & Koenig 2012 的理论模型，未考虑现场气候变化、微生物多样性等实际因素，且缺乏现场实验验证。

---

## 367. Explaining Failures of Cyber-Physical Systems with Actual Causality

**arXiv ID:** 2606.24546 | [PDF](https://arxiv.org/pdf/2606.24546v1)

**作者:** Khen Elimelech `[一作]` (King's College London), Moshe Y. Vardi `[通讯]` (Rice University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文把实际因果性（actual causality）框架引入到网络控制的网络物理系统（CPS）故障解释中，提出两种通用的解释生成算法，并在自驾车仿真中进行了实验验证。

**💡 创新点**

创新点主要包括：
1) 将实际因果性从静态分类器扩展到时序、连续的CPS，解决了环境参数、元素观测及失败类型抽象等特殊问题；
2) 设计了系统无关的最优与近似解释算法，兼顾解释质量与计算效率；
3) 在实验中展示了责任度量驱动的启发式搜索能够显著减少仿真次数。

**🔧 技术方法**

使用的技术包括：结构因果模型构造、责任度量（responsibility）近似、暴力搜索与责任引导的启发式搜索、随机干预仿真以及多值失败类型的抽象。

**📊 数据集**

实验数据来自 LiteRacer 仿真器，构造了 32 个故障实例，分别对应 6、9、12、15 个障碍物，全部由同一条轨道参数产生。

**📈 对比分析**

比较方法：与完全穷举（ES）算法对比，分别采用责任引导（RG）和责任引导+最小化（RGM）两种变体，并在不同样本预算（100/200/300）下评估。结果表明 RG 在所有案例下都极大减少了仿真次数但得到的解释尺寸较大；RGM 在保持与 ES 同样的最优解释的同时，显著降低了仿真次数，尤其在 200 采样预算时表现最佳。

**⚠️ 局限性**

局限性：
1) 只处理深度为 2 的因果模型，无法完整捕获连续时间与空间的细节；
2) 只求充分解释，未考虑必要解释；
3) 责任度量的近似依赖采样，样本数需手动调节，可能导致解释误差；
4) 仅在单一仿真平台验证，泛化性待进一步检验。

---

## 368. A specialized reasoning large language model for accelerating rare disease diagnosis: a randomized AI physician assistance trial

**arXiv ID:** 2606.24510 | [PDF](https://arxiv.org/pdf/2606.24510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 369. PointVG-R: Internalizing Geometric Reasoning in MLLMs for Precise Pointing Localization via Visual Chain of Thought

**arXiv ID:** 2606.24539 | [PDF](https://arxiv.org/pdf/2606.24539v1)

**作者:** Ling Li `[一作]` (Tsinghua University), Zhidong Deng `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于视觉链式思考（V-CoT）的点指视觉定位模型 PointVG-R。

**💡 创新点**

创新点是将几何推理嵌入多模态大模型，结合冷启动 SFT 与 GRPO 自适应奖励，并利用组方差加权提升学习稳定性。

**🔧 技术方法**

采用 Qwen2.5‑VL 作为多模态 LLM 基础，结合 LoRA、RL（GRPO）、结构化 V-CoT 与自适应重要性加权。

**📊 数据集**

使用新构建的 EgoPoint‑CoT 数据集（约 15K egocentric 图像）以及负样本。

**📈 对比分析**

在 EgoPoint‑CoT 基准上相较于最强对齐基线提升 15.86% mIoU，达到 0.7570。

**⚠️ 局限性**

局限是 V-CoT 需要逐步生成中间步骤，导致推理延迟，难以满足实时 AR 设备需求。

---

## 370. Importance Sampling for Event Discovery via Guesswork

**arXiv ID:** 2606.24537 | [PDF](https://arxiv.org/pdf/2606.24537v1)

**作者:** Asaf Cohen `[一作]` `[通讯]` (Ben-Gurion University of Negev), Asaf Cohen (Ben-Gurion University of Negev)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

本文提出一种基于猜测工作（guesswork）的重要性采样设计，用于快速发现罕见事件轨迹并保证其最小惊讶度。

**💡 创新点**

创新点在于将发现速度与轨迹质量统一到一个信息论框架，将“最少惊讶度”作为优化目标，从而在同等发现速率下打破传统KL距离导致的对称性，选择更易于搜索的轨迹。

**🔧 技术方法**

采用了信息论（熵、KL散度）、大偏差理论、类型方法以及猜测工作指数的数学工具，对重要性采样的目标函数重新定义并证明最优解。

**📊 数据集**

本文为理论研究，无针对具体数据集的实验；理论分析基于离散 i.i.d. 取样模型和类型集合。

**📈 对比分析**

通过与传统基于 KL 散度的最优采样方案（即最小方差设计）对比，证明猜测工作优化在子指数收敛速度下能够实现更低的命名秩指数（搜索深度），从而在发现速度相同的前提下获得更高质量的轨迹。

**⚠️ 局限性**

局限性包括：仅在离散 i.i.d. 模型和类型定义的罕见集合下可解析；对连续或复杂动力学系统的推广仍需进一步研究；实验验证尚未完成。

---

## 371. Poster: Exploring the Limits of Audio-Based Detection of Turkish Phone Call Scams

**arXiv ID:** 2606.24523 | [PDF](https://arxiv.org/pdf/2606.24523v1)

**作者:** Arda Eren `[一作]` (Hong Kong Polytechnic University), Eugene Yujun Fu `[通讯]` (Education University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了首个公开的土耳其语诈骗电话对话多模态数据集，并用七种大语言模型在三种输入条件下评估其诈骗识别性能。

**💡 创新点**

创新点在于首次聚焦低资源语言土耳其语，推出100条对齐的音频-文本诈骗与正常通话样本，并展示了文本输入在大模型检测中的优势及人类校正对性能的有限影响。

**🔧 技术方法**

使用了多模态大语言模型（Gemini 2.5 Flash/Flash‑Lite/Pro、GPT‑4o、Qwen Max/Plus/Turbo）和语音识别（Scribe V1）作为前置处理。

**📊 数据集**

使用的数据集包含100条土耳其语电话录音（50条诈骗，50条正常），每条都有16 kHz单声道音频和对齐文本（原始和人工校正）。

**📈 对比分析**

通过在三种输入条件下直接对每条样本进行评估，比较了模型的精确率、召回率与F1。结果显示：文本输入（无论是否校正）平均F1≈0.99，原始音频平均F1≈0.97；文本校正对性能影响不大。

**⚠️ 局限性**

局限性包括数据量有限、音质与标签不一致、模型内容过滤导致对低级攻击语音的误拒、以及未进行微调或提示优化，可能导致过度乐观的高分。

---

## 372. MambaRaw: Selective State Space Modeling for Efficient 4K Raw Image Reconstruction

**arXiv ID:** 2606.24479 | [PDF](https://arxiv.org/pdf/2606.24479v1)

**作者:** Peize Li `[一作]` (Tsinghua University), Yan Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 JPEG 条件下的元数据重建框架 MambaRaw，用于高效重建 4K RAW 图像。

**💡 创新点**

创新点是空间‑能量耦合的上下文建模：利用能量驱动的 TileMambaBlock 进行稀疏扫描，以及 Energy‑Aware Refinement（EAR）对长尾能量分布进行细化。

**🔧 技术方法**

采用了状态空间模型（SSM）中的 Mamba 结构、轻量化的 TileMambaBlock、EAR 以及 JPEG 条件下的分析/合成变换和超参数学习。

**📊 数据集**

在 Sony、Olympus、Samsung 的 NUS 数据集和 AdobeFiveK 数据集上进行实验。

**📈 对比分析**

与 SAM、CAM、R2LCM、Beyond‑R2LCM 等基线比较，在相同元数据比特率下提升 1.2–1.4 dB PSNR，且端到端编码延迟降低约 9%。

**⚠️ 局限性**

局限在于仅处理单帧，缺乏对视频时序冗余的利用，TileMambaBlock 在更高分辨率或不同硬件上仍需进一步优化。

---

## 373. G$^3$VLA: Geometric inductive bias for Vision-Language-Action Models

**arXiv ID:** 2606.24472 | [PDF](https://arxiv.org/pdf/2606.24472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 374. NatureBench: Can Coding Agents Match the Published SOTA of Nature-Family Papers?

**arXiv ID:** 2606.24530 | [PDF](https://arxiv.org/pdf/2606.24530v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 375. VisCritic: Visual State Comparison as Process Reward for GUI Agents

**arXiv ID:** 2606.24525 | [PDF](https://arxiv.org/pdf/2606.24525v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 376. Reasoning as Attractor Dynamics: Latent Memory Retrieval via Gibbs-Weighted Energy Minimization

**arXiv ID:** 2606.24543 | [PDF](https://arxiv.org/pdf/2606.24543v1)

**作者:** Kanishk Awadhiya `[一作]` `[通讯]` (Independent Researcher), Kanishk Awadhiya (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将大型语言模型的推理过程视为能量场中的吸引子动态，提出基于谱熵的 Gibbs 加权检索机制，利用能量退火实现稳健推理。

**💡 创新点**

创新点在于把推理视作从高温采样到平衡态的物理退火过程，并用逆平方加权的 Gibbs 分布对采样路径进行重加权，显著过滤“尖锐最小值”中的幻觉。

**🔧 技术方法**

技术包括谱熵能量度量、Gibbs 采样/热退火、逆平方加权重、对比自一致性（Self‑Consistency）与贪心解码，并借鉴现代 Hopfield 网络的检索更新。

**📊 数据集**

使用 GSM8K 公开数学推理数据集，模型为 Microsoft Phi‑3.5‑mini‑instruct（3.8 B 参数）。

**📈 对比分析**

通过与贪心解码（78.4%）和标准多样化采样（84.7%）比较，Gibbs 加权在 GSM8K 上取得 90.07% 的准确率，提升约 5.4%（从 84.7% 到 90.07%）。

**⚠️ 局限性**

局限性包括仅在小规模模型和单一基准上验证，缺乏对大模型或更复杂任务的通用性实验；谱熵作为能量的经验度量需进一步理论与跨任务验证。

---

## 377. Advancing WordArt-Oriented Scene Text Recognition: Datasets and Methods

**arXiv ID:** 2606.24484 | [PDF](https://arxiv.org/pdf/2606.24484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 378. CrossPool: Efficient Multi-LLM Serving for Cold MoE Models through KV-Cache and Weight Disaggregation

**arXiv ID:** 2606.24506 | [PDF](https://arxiv.org/pdf/2606.24506v1)

**作者:** Zhuoren Ye `[一作]` (Beihang University), Renyu Yang `[通讯]` (Beihang University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种针对冷 MoE 模型的多 LLM 服务器引擎，采用两块 GPU 内存池（FFN 权重池和 KV‑Cache 池）来分离稳定权重和瞬时 KV‑Cache，并在两池之间通过隐藏状态传输完成前向传播。

**💡 创新点**

创新点包括：① KV‑Cache 规划器与虚拟化器，可根据不同模型的 KV‑Cache 需求动态分配共享池；② 层级流水线调度器，将注意力层与 FFN 层在不同池中交叉执行；③ 持久化内核与控制下沉，将高频调度与通信交互迁移至 GPU，显著降低主机干预；④ 通过解耦权重与 KV‑Cache 解决 KV‑head 限制模型在低并发场景下的资源浪费。

**🔧 技术方法**

主要技术：CUDA VMM 与 NVLink + NVSHMEM 进行跨池低延迟隐藏状态传输；trace‑driven KV‑Cache 规划与分页；持久化内核（persistent kernels）与控制下沉；多种注意力算法（MLA、MQA、GQA）与 SGLang 计算框架；Layer‑wise 任务调度与异步通信。

**📊 数据集**

实验使用的公开数据集：OpenRouter 真实请求流量（用于分析冷热分布与 KV‑Cache 使用）；Vicuna ShareGPT 对话作为平衡的输入/输出工作负载；LongAlign 作为长上下文压力测试。

**📈 对比分析**

与两种基线（Static Partition 与 Chimera）在相同 GPU 预算下对比。评价指标为：TBT（P95、P99）、最大请求率（RPS）以及整体输出吞吐量。实验结果显示：在 0.8–1.0 RPS 下，P99 TBT 分别下降 7.6×、10.4×、7.3×；整体吞吐量提升 1–2 倍；且在长上下文场景下能保持稳定服务，显著提高共享 KV‑Cache 容量。

**⚠️ 局限性**

局限性：① 跨池隐藏状态传输仍占用一部分关键路径时间，尚未完全隐藏；② 不同模型的计算耗时差异可能导致流水线不平衡；③ 需要在部署时对相似计算特征的冷模型进行分组以减少瓶颈。

---

## 379. Visualizing "We the People": Bridging the Perception Gap through Pluralistic Data Storytelling

**arXiv ID:** 2606.24635 | [PDF](https://arxiv.org/pdf/2606.24635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 380. The Warrant Gap: Claim-Conditioned Re-scoring for Fact-Checking

**arXiv ID:** 2606.24627 | [PDF](https://arxiv.org/pdf/2606.24627v1)

**作者:** Arka Ujjal Dey `[一作]`, John Collomosse `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究揭示了事实核查系统在输出“支持”结论时常常缺乏对所引用证据的真正许可，提出了SIFT（Claim‑Conditioned Re‑scoring）结构化推理方法和WSP（Warranted Supports Proportion）自动评估指标，以修复5W1H分解导致的误判并提升可审计性。

**💡 创新点**

创新点包括：① WSP作为自动衡量支持结论是否被引用证据充分许可的诊断工具；② SIFT通过在5W1H分解后对每个 facet 进行 claim‑conditioned re‑scoring，恢复上下文语义并修复局部误判；③ Design B deterministic verifier提供可复现的 admissibility‑accuracy 取舍阈值，实现基于规则的最终判决。

**🔧 技术方法**

技术手段涵盖：结构化分解（5W1H）、SIFT 层、DeBERTa NLI 窗口扫检、MiniCheck grounding 交叉验证、Design B阈值规则、对比的生成式 LLM adjudication（Design A）以及多种 NLI 家族的稳健性测试。

**📊 数据集**

实验使用四大公开事实核查数据集：FEVER、SciFact（单证据场景）和5PILS、DP（多证据、跨文档聚合场景）。

**📈 对比分析**

通过与直接 prompting、naïve 5W1H、SIFT、Design A/B 的对比，利用 McNemar 检验、AUC、WSP 等指标评估，SIFT 在准确率上恢复至 Direct 以上，WSP 在所有模型中显著提升（最高提升52点），Design B 在可接受的准确率损失下显著提升 admissibility，整体表现优于传统方法。

**⚠️ 局限性**

局限性包括：WSP 仅为自动诊断，缺乏全面人工审计；仅在英文文本上评估，5W1H 结构对非事件型命题不够自然；Verifier 受检索与提取的限制，无法修复 identity/predicate 错误；阈值需在本地验证数据上进行校准。

---

## 381. Privacy-Preserving RAG via Multi-Agent Semantic Rewriting: Achieving Confidentiality Without Compromising Contextual Fidelity

**arXiv ID:** 2606.24623 | [PDF](https://arxiv.org/pdf/2606.24623v1)

**作者:** Yuanhe Zhao `[一作]` (North China Electric Power University), Tao Fang `[通讯]` (Macau Millennium College)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多代理语义重写框架，用于在检索增强生成（RAG）中实现隐私保护，既消除敏感信息又保持上下文语义完整；

**💡 创新点**

创新点在于将隐私提取、语义分析与重构拆分为三名专用代理，并采用结构化提示和细粒度冲突路由，实现离线预处理与异步执行；

**🔧 技术方法**

使用了规则+LLM混合的隐私提取、基于结构化属性的语义拆解、模板化重写、双轨向量检索、BM25+密集检索融合，以及多种大型语言模型（GPT‑3.5‑Turbo、GPT‑4o‑mini、LLaMA‑3‑8B、DeepSeek‑V3等）；

**📊 数据集**

在医疗对话数据集ChatDoctor（HealthCareMagic‑101）和混合公开/私有文本Wiki‑PII（500k样本）上进行评估；

**📈 对比分析**

与Naive RAG、AttrPrompt、SAGE、Single Agent、Adv‑Anon、KG‑PrivRAG等基线相比，本文方法在针对性提取攻击下将敏感实体泄露从144/49降至1/3，在无针对性重构攻击下将完整复制率降至0，BLEU‑1达0.122（高于SAGE 0.117），同时离线预处理时间最短，在线推理无额外延迟；

**⚠️ 局限性**

局限性包括：在极罕见病名等情形下可能出现语义漂移或过度去标识；缺乏大规模标注的隐私提取评估；对更强适应性攻击（如跨文档推断）的防护尚未充分验证；

---

## 382. When CQs Go Wrong: Challenges in CQ Verification with OE-Assist

**arXiv ID:** 2606.24619 | [PDF](https://arxiv.org/pdf/2606.24619v1)

**作者:** Anna Sofia Lippolis `[一作]` (University of Bologna), Andrea Giovanni Nuzzolese `[通讯]` (ISTC-CNR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在19名本体工程师上进行基于OE-Assist的半自动 CQ 验证实验，研究了 CQ 的歧义、复杂度与验证性能之间的关系，并分析了用户反馈

**💡 创新点**

提出可量化 CQ 复杂度的指标（Flesch‑Kincaid 级别、Gunning Fog）与歧义检测，并建议构建“CQ 故障扫描器”以自动提示并修正含糊或过于复杂的 CQ

**🔧 技术方法**

采用了 OE-Assist 原型工具、LLM 辅助（如 GPT）生成的 CQ 处理建议、SPARQL 查询生成器，以及文本可读性评估算法（Flesch‑Kincaid、Gunning Fog）

**📊 数据集**

实验数据来自 19 名参与者在 20 个 CQ 任务中的操作记录，使用的本体规模分为 Small、Medium、Large（按 axiom 计数划分）

**📈 对比分析**

通过 Spearman/Kendall 相关性、对比实验（有 LLM 辅助 vs 无辅助）评估验证准确率、感知难度评分和决策时长；结果显示 LLM 建议能提升准确率，且 CQ 复杂度与感知难度呈正相关

**⚠️ 局限性**

局限性包括样本量较小、仅在单一工具/域内验证、SPARQL 生成的鲁棒性不足、缺乏对多语言环境下更深层次歧义的系统评估

---

## 383. AI Tokenomics: The Economics of Tokens, Computation, and Pricing in Foundation Models

**arXiv ID:** 2606.24616 | [PDF](https://arxiv.org/pdf/2606.24616v1)

**作者:** Quanyan Zhu `[一作]` `[通讯]` (New York University), Quanyan Zhu (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了 AI Tokenomics 的统一框架，系统地将 token 视为信息、计算、内存、能耗与经济价值的共享资源，并从技术、经济、工作流和市场层面阐释 token 的生成、消费、定价与分配机制。

**💡 创新点**

创新点包括：①首次将 token 形式化为稀缺资源并定义 AI Tokenomics；②把 token 级技术成本与任务级需求、工作流质量与企业价值关联；③提出任务级 token 需求函数和工作流级网络优化模型；④引入风险敏感边际效用分配和多阶段工作流的后向传播分析；⑤探讨 token 市场、拍卖、动态定价与合约设计的理论基础。

**🔧 技术方法**

所用技术主要是 transformer tokenization、规模律推导、工作流图与生产函数建模、拉格朗日优化与后向传播、风险函数构造、动态定价/拥塞定价与拍卖机制设计。

**📊 数据集**

本文并未使用公开实验数据集，而是以公开的 token 定价表与示例任务（法律分析、软件工程、检索增强研究、自治多代理）为基础进行理论推导和案例演示。

**📈 对比分析**

通过四个案例对比 token 消耗、边际价值和整体效用，证明仅按 token 数量分配并不能最大化价值；引入风险调节后的边际效用分配能够更合理地在工作流间分配 token，从理论上提升整体效用。

**⚠️ 局限性**

局限性包括：隐藏推理 token 的估算方法不成熟；模型假设（如连续化、无环图、确定性生产函数）过于简化；缺乏大规模实证验证；对动态需求、策略互动与多代理市场的深入分析仍待完善。

---

## 384. To Compare, or Not to Compare: On Methodological Practices in Evaluating Social Bias

**arXiv ID:** 2606.24596 | [PDF](https://arxiv.org/pdf/2606.24596v1)

**作者:** Federico Marcuzzi `[一作]` (Sofia University St Kliment Ohridski), Iryna Gurevych `[通讯]` (Sofia University St Kliment Ohridski)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一可控的评估框架，将传统的社会偏见评测拆解为两种范式（孤立评估 iso 与比较评估 cmp），系统对比两者在不同模型、推理方式、退避选项和随机性声明下的偏差表现。

**💡 创新点**

创新点在于：①首次将评测范式作为核心变量统一标准化；②揭示比较范式能显著放大模型潜在偏见，且链式推理（CoT）会在 cmp 环境下进一步强化偏差；③证明中立选项与“随机回答”声明并不真正缓解偏差；④展示偏差随模型规模正相关。

**🔧 技术方法**

技术手段包括：统一的提示模板与可控参数化、基于 Parity Gap (PG) 的偏差度量、对多种 CoT 与非 CoT、三选项中立与随机声明的对照实验、以及 Pearson 相关性分析来考察规模效应。

**📊 数据集**

使用了 8 个标准化社会偏见基准（覆盖性别、种族、宗教），并将其转换为 iso 与 cmp 两种格式，此外对 10 个公开权重 LLM（LLaMA、Qwen 等）进行评测，实验还扩展到 19 只模型。

**📈 对比分析**

通过比较 iso 与 cmp 的 PG 以及对 CoT、退避选项、随机声明的 ablation，发现 cmp 在绝大多数基准下偏差显著更高，CoT 在 cmp 下会进一步放大偏差；模型规模越大，cmp 的 PG 越大；在 iso 下偏差几乎为零，显示了评测范式对结果的决定性影响。

**⚠️ 局限性**

局限性包括：仅评估公开权重模型，未覆盖闭源大模型；实验只在英文单语环境下进行，未考虑多语言与更广泛的敏感属性；缺乏对内部机制的解释，无法明确偏差放大的神经根源；以及基准泄漏等问题可能对偏差分布产生影响。

---

## 385. MEMPROBE: Probing Long-Term Agent Memory via Hidden User-State Recovery

**arXiv ID:** 2606.24595 | [PDF](https://arxiv.org/pdf/2606.24595v1)

**作者:** Enze Ma `[一作]` (University of Illinois Chicago), Zhen Wang `[通讯]` (UC San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MemProbe基准，用来评估LLM代理的长期记忆，通过在一次普通对话后重构隐藏的用户状态来衡量记忆质量。

**💡 创新点**

创新点在于将记忆评估转为可审计的后互动工件评估，而不是仅依赖下游任务成绩；构造了带有隐式用户状态金库的模拟用户和泄露控制任务，直接测量记忆可恢复性。

**🔧 技术方法**

使用大型语言模型（如GPT‑5.4‑mini）进行槽位填充评估、恢复判断以及失败归因；采用多种记忆系统（无记忆、全文存储、笔记记忆、对话记忆、训练记忆操作策略）进行对比。

**📊 数据集**

使用50个模拟用户，每个用户拥有31维隐藏状态（共1,550个恢复目标），通过自动生成的任务和对话来提供训练与评估数据。

**📈 对比分析**

对比了五种记忆系统，在dump_all（全存储）和retrieve（top‑k检索）两种访问模式下计算恢复分数。结果显示：任务完成率接近100%，但在dump_all下恢复分数仅0.61–0.62，retrieve下进一步降至0.47–0.54，说明任务成功并不保证记忆可恢复。

**⚠️ 局限性**

局限性包括：基准仅基于合成用户，缺乏真实用户多样性；恢复评估依赖LLM判断，可能带来主观误差；未充分探究不同检索阈值和写入策略对恢复的细粒度影响。

---

## 386. A Congestion Parameter for Depth-First Graph Traversals

**arXiv ID:** 2606.24675 | [PDF](https://arxiv.org/pdf/2606.24675v1)

**作者:** Codaline Bourotte `[一作]` (École Normale Supérieure de Lyon), Shinnosuke Seki `[通讯]` (University of Electro-Communications)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并研究了KLX数这一新的图参数，用于衡量DFS遍历中的最大同时打开回边数，给出了0、1、2的完全特征与线性时间识别算法；

**💡 创新点**

创新点在于将KLX数与树宽、MSO_2表达式联系，证明其可通过Courcelle定理实现FPT算法，并给出了严格的树宽上界；

**🔧 技术方法**

使用了DFS树与弧图、树宽与树分解、MSO_2逻辑与Courcelle定理等图论与算法技术；

**📊 数据集**

本文未使用具体实验数据集，而是通过理论证明和结构化算法提供了多种算法实现；

**📈 对比分析**

方法通过线性时间结构化识别与FPT层面验证相结合，性能在理论上实现O(f(k)(|V|+|E|))，但实际运行受k指数增长影响；

**⚠️ 局限性**

主要限制是FPT算法的指数因子使得在实际大规模图中难以使用，且高KLX值图的复杂度尚未得到实验验证。

---

## 387. Cost-Optimal Decision Diagrams for Stochastic Boolean Function Evaluation

**arXiv ID:** 2606.24672 | [PDF](https://arxiv.org/pdf/2606.24672v1)

**作者:** Xia Zong `[一作]` (Aalto University), Jussi Rintanen `[通讯]` (Aalto University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了在变量观测成本不同且变量值满足联合概率分布的情形下，构造期望评估成本最小的决策图（SBFE）问题，提出了一种基于分支限界、启发式变量选择、剪枝和缓存的精确求解算法，并给出了贪心束搜索变体。

**💡 创新点**

创新点在于：1) 首次给出针对一般布尔公式、任意正成本与任意概率分布的可扩展精确算法；2) 结合多种变量选择启发式和有效剪枝/缓存显著提升规模；3) 证明构造最优决策图是#P-hard；4) 提供束搜索折中方案。

**🔧 技术方法**

采用分支限界搜索、变量排序启发式（基于成本、子公式估计和概率），剪枝与缓存（子问题最优解与下界缓存），以及束搜索（k-best分支），实现上使用Python + SAT求解器CaDiCaL 1.9.5。

**📊 数据集**

实验数据集：随机生成的3-SAT公式（变量数6–15，Clause-to-Variable比1.5，随机成本1–10，独立真值概率），以及将Cleveland心脏病诊断决策树转化为SBFE实例（16变量22子句）。

**📈 对比分析**

与四种基线（无缓存、无剪枝、仅剪枝、仅缓存）对比，结果显示启用剪枝与缓存的组合（尤其是Heuristic+Cache）在时间和内存上均有指数级改进；束搜索在硬实例上能在时间上优于精确算法，且成本误差≤30%。

**⚠️ 局限性**

局限性：1) 仍为指数级算法，对更大规模实例（>20变量）受限；2) 依赖变量成本和概率的精确知识，实际应用中难以获取；3) 随机实验中假设概率独立，未考察相关性；4) 仅评估单一SBFE实例类型，泛化性待验证。

---

## 388. Harmonic: Hierarchical State Space Models for Efficient Long-Context Language Modeling

**arXiv ID:** 2606.24650 | [PDF](https://arxiv.org/pdf/2606.24650v1)

**作者:** Petr Nyoma `[一作]` `[通讯]` (Independent Researcher), Petr Nyoma (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种三层分层状态空间模型（Harmonic），利用多时间尺度递归和预测误差交互来实现无注意力的长序列语言建模。

**💡 创新点**

创新点在于将预测编码的误差信号与层级时间尺度相结合，形成一种多尺度递归网络，既保持了 O(L) 计算复杂度，又在长上下文上显著优于传统 Transformer 与 Mamba。

**🔧 技术方法**

主要技术包括分层状态空间网络（SSM）、输入选择性衰减门控、层间误差信号传递、Triton 并行扫描实现 O(L) 前向传播以及对 TinyLlama 1B 参数模型的 HarmonicBlock 换装与微调。

**📊 数据集**

实验使用的主要数据集为 enwiki8（字节级）、WikiText‑103（词级）和 WikiText‑2（词级）进行跨域验证，Hallamonic 版则在 fineweb‑edu 进行微调并在 Lambada、WikiText‑103 与 fineweb‑edu held‑out 进行评估。

**📈 对比分析**

比较方法采用严格等 token 预算、相同 tokenizer、优化器与学习率调度等一致配置；在 1K–32K token 长度下，Harmonic 在 28M 参数模型上相较 Transformer 下降 1.4%–11.4% perplexity，Mamba 介于两者之间；在 64K token 仅 Harmonic 能在 80GB H100 上训练成功；在 1B 参数规模的 Hallamonic 中去除 RoPE 限制后，损失在 1K–8K token 上保持稳定，比 TinyLlama 提升 3–10 bpt。

**⚠️ 局限性**

局限性包括实验主要聚焦英语文本、参数规模最多 112M（1B 版仅单一次微调且未做多种随机种子验证）、未测量推理时延与跨语言/多模态性能，以及对非常大模型或非文本任务的泛化尚待进一步验证。

---

## 389. Agentic Collaborative Cognition for Zero-Shot 3D Understanding

**arXiv ID:** 2606.24649 | [PDF](https://arxiv.org/pdf/2606.24649v1)

**作者:** Wenxin Wang `[一作]` (Sichuan University), Yinjie Lei `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了协同多代理框架，利用规划代理和感知代理实现零样本3D场景理解。

**💡 创新点**

通过显式认知图与视角规划，实现视角补全与对象属性的迭代交互，突破传统仅靠视频关键帧的局限。

**🔧 技术方法**

使用多模态大型语言模型（如Qwen2.5-VL-72B和GPT-4o）、基于BEV的认知图、视角匹配与渲染、候选过滤与反馈闭环。

**📊 数据集**

六个基准：ScanRefer、Nr3D、SQA3D、ScanQA、3D-LLM held-in（对话与任务拆分）以及3D-assisted dialog。

**📈 对比分析**

与现有零样本方法（CSVG、SeqVLM、SPAZER等）及部分全监督方法对比，在所有任务上均实现SOTA提升，如ScanRefer Acc@0.5提升11.1%、SQA3D EM提升2.1点、3D-LLM BLEU-1提升14.6点。

**⚠️ 局限性**

依赖预训练3D检测模型，检测误差影响整体；渲染图像的几何/纹理精度有限；多轮交互推理成本高；对极端场景（复杂多对象、动态环境）仍有限。

---

## 390. Accelerating Presto with GPUs

**arXiv ID:** 2606.24647 | [PDF](https://arxiv.org/pdf/2606.24647v1)

**作者:** Daniel Bauer `[一作]` (IBM Research Europe), Karthikeyan Natarajan `[通讯]` (NVIDIA)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将Presto扩展为GPU可感知的执行引擎，利用Velox、cuDF和UCX实现完整的GPU加速查询执行；

**💡 创新点**

实现了无CPU中介的GPU到GPU数据交换协议UcxExchange、一次性将数据读入GPU内存、以及在分布式环境中保持GPU内存数据不被调度到CPU的完整体系；

**🔧 技术方法**

使用cuDF进行GPU算子实现、UCX实现GPU间通信、KvikIO实现存储到GPU的数据加载、Velox执行框架、Arrow格式、CUDA流异步执行；

**📊 数据集**

以TPC‑H类基准查询为数据集，规模因子从1,000到30,000（10 TB–30 TB）进行实验；

**📈 对比分析**

通过与CPU版Presto（HttpExchange）及裸机cuDF实现对比，发现GPU版在交换量大查询中可实现高达20×加速、整体查询时间降低至CPU版的1/6，成本/性能比提升2–6×；

**⚠️ 局限性**

受限于合成工作负载、GPU内存容量不足时无法自适应外部溢出、缺乏GPU感知的查询规划器、未实现跨查询GPU内存管理与调度等问题。

---

## 391. Measuring User's Mental Models of Speech Translation in Human-AI Collaboration

**arXiv ID:** 2606.24644 | [PDF](https://arxiv.org/pdf/2606.24644v1)

**作者:** HyoJung Han `[一作]` (University of Maryland), Marine Carpuat `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过基于跨语言问答的实验框架，研究用户在语音翻译系统中的心理模型形成与演化。

**💡 创新点**

提出了将下游任务（问答）嵌入评估流程的创新框架，证明用户可通过主动选择重译来学习系统错误模式，且语音转录作为解释更能促进心理模型提升。

**🔧 技术方法**

使用 Whisper 进行语音翻译，Mistral‑7B 进行阅读理解问答；设计奖励机制并引入转录文本和错误跨度两种解释；对用户行为进行回归与简单遗憾分析。

**📊 数据集**

数据集为 2m‑belebele 的法语音频与对应英文问答对（共 16 组），其中选取 11 个包含错误、5 个正确的样本；使用 Whisper 输出的翻译与原金标准进行对比。

**📈 对比分析**

在三种条件（默认、转录、错误跨度）下比较最终分数与准确率：转录条件在所有组平均得分最高（49.8）并提升整体准确率；错误跨度条件虽使准确率最高（70%）但因频繁重译导致最终得分最低（38.5）。

**⚠️ 局限性**

局限性：仅使用法语源语言，样本量与问题数量有限，未涵盖其他语言或更大规模用户群体，结果的普适性需进一步验证。

---

## 392. Two-Level vs. Multi-Level Modelling: An Empirical Study of Cascading Maintenance Burden

**arXiv ID:** 2606.24721 | [PDF](https://arxiv.org/pdf/2606.24721v1)

**作者:** Yuhong Fu `[一作]` (Adelaide University), Markus Stumptner `[通讯]` (Adelaide University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

本研究设计了一项预注册的经验研究，比较两级建模（2LM）与多级建模（MLM）在模型演化中的维护负担差异，采用变异（mutation）方式对同一语义场景在两种范式下施加相同的演化操作，并通过自动一致性检查与结构差分测量后续不一致数和修改量。

**💡 创新点**

创新点在于提出了可复制的实验框架——包括双范式映射协议、统一的变异词汇表、自动恢复脚本以及两项可度量的维护指标（后置不一致计数和修改编辑量），并在设计中嵌入正向控制、对照控制及盲映射协议，力求消除对MLM有利的系统性偏差。

**🔧 技术方法**

采用的技术包括：EMF（Eclipse Modeling Framework）与SLICER/DoME实现2LM与MLM范式；自动验证器（EMF Validation Framework、DoME Validator）获取不一致计数；EMF Compare与DoME结构差分获取修改计数；统计分析使用配对单侧Wilcoxon符号秩检验、Holm校正及Bootstrap置信区间；手工审计工具一致性与修改结果以确保测量准确。

**📊 数据集**

使用的数据集为从四大公开来源筛选的可用共演化场景：GMF演化历史、Herrmannsdoerfer等的操作目录、Hebig等的迁移案例、Iovino等的多语言元模型演化影响研究；场景按规模（小/中/大）分层，目标样本量为55对（至少47对满足统计功效）。

**📈 对比分析**

比较方法为在每一场景-变异对中，先映射为MLM，然后在两种范式下分别施加相同的变异，记录变异后不一致数；随后执行预注册的恢复脚本，得到一致的模型；再测量修改编辑量。最终采用配对单侧Wilcoxon检验并进行多重检验校正，报告效应大小（匹配对秩二分相关系数）。目前尚无结果，性能表现待后续阶段完成。

**⚠️ 局限性**

局限性包括：1）仅针对SLICER实现的MLM，可能不具备跨框架泛化性；2）使用的测量指标仅为结构编辑数量，未直接映射到人类工程师的工作量；3）工具成熟度差异（EMF vs DoME）可能影响一致性与差分检测的准确性；4）语料库来源主要聚焦于EMF生态，可能偏向2LM的设计习惯；5）缺乏真实工程维护情境的验证。

---

## 393. Bit-Precise Conformance Testing of Simulink Model Checkers

**arXiv ID:** 2606.24719 | [PDF](https://arxiv.org/pdf/2606.24719v1)

**作者:** Daisuke Ishii `[一作]` (Japan Advanced Institute of Advanced Science), Hideaki Takai `[通讯]` (GAIO Technology Co. Ltd.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

对 Simulink 模型检查器的比特精度符合性进行自动化测试，构建块特征模型和测试套件，评估 MATLAB/Simulink Design Verifier 与 Prompt V2 SMT 基础工具。

**💡 创新点**

通过精确建模块运算的位级语义，提出组合式测试方法检测检查器与仿真器的一致性，并提供差异分析与失败原因定位。

**🔧 技术方法**

形式化块特征模型、位级 FP 与整数算术的 SMT 编码、组合式（pairwise）测试生成、MATLAB/Simulink API 与 Z3 SMT 解决器集成。

**📊 数据集**

10 种基本块类型的手工构造特征集，基于多种数据类型（双精度、整数、布尔、字符串）的有限取值表，信号长度分别为 1 或 3。

**📈 对比分析**

将工具输出与仿真器（或 Prompt 伪判据）比较，计算通过率；实验显示 Design Verifier 在所有测试中通过率 94–96%，Prompt 的通过率 80–90%；Prompt 速度通常快于 Design Verifier，但在某些块（如 Product）上耗时更高。

**⚠️ 局限性**

仅覆盖十种基本块，未处理更复杂组合、矩阵/总线、非线性或大规模模型；测试用例有限，可能漏掉隐藏缺陷；仿真器本身也可能存在错误。

---

## 394. Evaluating the Interpretability of Sparse Autoencoders with Concept Annotations

**arXiv ID:** 2606.24716 | [PDF](https://arxiv.org/pdf/2606.24716v1)

**作者:** Jonas Klotz `[一作]` (Berlin Institute for the Foundations of Learning and Data), Begüm Demir `[通讯]` (Berlin Institute for the Foundations of Learning and Data)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套人类基准的稀疏自编码器（SAE）可解释性评估框架，量化 SAE 潜在空间与人类注释概念的对齐，并通过合成数据验证功能一致性。

**💡 创新点**

创新点在于：①引入全二进制匹配追踪（FBMP）实现多对一概念匹配；②构建了两个受控合成数据集（synCUB、synCOCO）实现单属性干预；③提出针对属性扰动的对齐得分（TAPAScore）以评估因果一致性。

**🔧 技术方法**

采用稀疏自编码器（JumpReLU、TopK、BatchTopK、Matryoshka）学习 CLIP 与 DINOv2 嵌入的稀疏字典；使用二进制匹配、逻辑门运算实现 FBMP；利用预训练视觉模型的嵌入进行训练与评估。

**📊 数据集**

使用 CUB‑200‑2011 的 312 个属性作为真实标签，并在此基础上生成 synCUB；使用 MS‑COCO 的物体标签生成 synCOCO。

**📈 对比分析**

与传统结构性/功能性指标（FMS、MS、CKNNA）和监督线性探针进行对比。FBMP‑F0.5 与 TAPAScore 在多数设置下表现最佳，显示中等字典大小的 SAE 在统计匹配与因果一致性上取得最佳平衡。

**⚠️ 局限性**

局限性包括对人工属性注释的高度依赖；合成数据的视觉丰富度有限；TAPAScore 的有效性受匹配集质量影响，未来需探索降低对手工注释依赖的方法。

---

## 395. CN-NewsTTS Bench: a target-level automatic benchmark for raw-input Chinese news TTS pronunciation

**arXiv ID:** 2606.24714 | [PDF](https://arxiv.org/pdf/2606.24714v1)

**作者:** Shijun Luo `[一作]` `[通讯]` (NetEase Cloud Music), Shijun Luo (NetEase Cloud Music)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CN-NewsTTS Bench v0.1 这一针对中文新闻 TTS 读音准确度的公开、基于目标级的基准；

**💡 创新点**

创新点在于：①以原始输入为评测对象，排除用户侧规则与手工编辑；②构建三路 ASR 集成自动评分协议；③提供目标级正负读音模式、覆盖率与精度诊断；

**🔧 技术方法**

使用了文本生成器（基于模板与词典）、三路异构 ASR（MiMo API、SenseVoiceSmall、Paraformer‑zh）以及自定义分数器；

**📊 数据集**

数据集为 1000 条合成新闻式句子（200 dev、800 公测），共 992 个可自动评估目标，覆盖得分、模型、单位、百分比等 11 类；

**📈 对比分析**

评估方法采用 strict accuracy、coverage 与 resolved accuracy 三项指标，并通过三路 ASR 的多数投票得分；实验结果显示最高系统严格准确率 0.879，其他系统多低于 0.60，且各类别差异显著；

**⚠️ 局限性**

局限性包括：ASR 可能掩盖同音字调差异导致的读音错误；未知率较高，特别是单位符号；缺乏人类听觉评测与隐藏测试集，评测对实际音频未进行公开验证。

---

## 396. ViTexQA: A Multi-Frame Temporal Perception Dataset for Video Text Question Answering

**arXiv ID:** 2606.24602 | [PDF](https://arxiv.org/pdf/2606.24602v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 397. PowerFuzz: Power-Based Black-Box Firmware Fuzzing

**arXiv ID:** 2606.24692 | [PDF](https://arxiv.org/pdf/2606.24692v1)

**作者:** Dakshina Tharindu `[一作]` (University of Florida), Prabhat Mishra `[通讯]` (University of Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种全黑盒固件 fuzzing 框架 PowerFuzz，利用功耗侧信道测量代替二进制 instrumentation 进行分支覆盖反馈。

**💡 创新点**

创新点在于使用功耗侧信道通过动态时间规整和相关性分析推断执行分支，动态构建 Trace‑guided Control Flow Graph (TCFG)，从而在无内部可见性的情况下实现基于覆盖的 fuzzing。

**🔧 技术方法**

主要技术包括功耗测量（ChipWhisperer）、动态时间规整 (DTW)、Pearson 相关性、TCFG 构造、LibAFL 变异器以及深度优先分支选择策略。

**📊 数据集**

使用了十款固件基准（Stepper、CNC、GPS、Soldering、cjson、zlib、TinyFFT、microECDSA、miniAES、TinyMaix）以及三种 MCU 平台（STM32F0、STM32F3、XMEGA）。

**📈 对比分析**

与 EM‑based 黑盒 fuzzing Fuzz'EMup 及随机测试进行对比，PowerFuzz 在四大基准上的分支覆盖率比 Fuzz'EMup 提升 9–22%，与灰盒 fuzzing 的差距不超过 13.5%，并在所有平台上比随机测试高 20–30%。

**⚠️ 局限性**

局限性包括 TCFG 的召回率仅约 77%，容易漏检细微分支；对功耗采样精度要求高；实验仅覆盖 MCU 固件，尚未验证在更大规模或不同硬件上的适用性。

---

## 398. One Index for Subsumption and Roll-up across Time, Geography, and Ontology

**arXiv ID:** 2606.24677 | [PDF](https://arxiv.org/pdf/2606.24677v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出OEH索引，统一处理时间序列、空间与本体中的层级关系，支持子集测试与层级汇总。

**💡 创新点**

创新点在于根据层级结构动态选择嵌套集或链分解编码，实现单一索引同时回答子集查询与索引驻留的聚合。

**🔧 技术方法**

使用深度优先嵌套集、链分解、Fenwick树/后缀和技术，并配备宽度阈值决策器。

**📊 数据集**

使用五个真实层级：Gene Ontology、NCBI Taxonomy、GeoNames、5年分钟级日历、Git提交DAG。

**📈 对比分析**

与PLL、2-hop、GRAIL、TimescaleDB等基线对比，OEH在树形层级下空间仅为对手一半、构建快6–7倍、查询延迟相当，并能完成连续聚合；链模式在低宽度DAG上优于2-hop；在高宽度DAG则退回2-hop。

**⚠️ 局限性**

局限在于仅支持静态索引、链模式在现实多父DAG中很少胜出、对极小子树聚合性能不佳、缺乏动态增删支持。

---

## 399. Abstractions of Queries in Ontology-Based Data Access

**arXiv ID:** 2606.24618 | [PDF](https://arxiv.org/pdf/2606.24618v1)

**作者:** Michel Leclère `[一作]` (LIRMM), Guillaume Pérution-Kihli `[通讯]` (LIRMM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

研究在基于存在规则的OBDA中，使用扩展的UCQ^≠,C查询类实现数据查询到本体查询的抽象；

**💡 创新点**

证明UCQ^≠,C既能表达最小完全抽象（从而任何完美抽象），又能通过最大恢复实现最大语义正确抽象；

**🔧 技术方法**

采用chase、查询重写、可逆映射（disjunctive^≠,C）和最大恢复等数据库理论技术；

**📊 数据集**

未使用实验数据集，工作为理论分析与算法设计；

**📈 对比分析**

通过复杂度分析（Π^P_2-complete/Co-NExpTime等）与已知结果对比，表明该方法在复杂度上不低于传统方法；

**⚠️ 局限性**

对最大语义正确抽象在UCQ^≠,C中可达成性未完全决定，相关判定问题是否可判定仍是开放问题。

---

## 400. SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation

**arXiv ID:** 2606.24626 | [PDF](https://arxiv.org/pdf/2606.24626v1)

**作者:** Chenyang Zhu `[一作]` (Capital One), Erin Babinsky `[通讯]` (Capital One)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SAFARI框架，通过主动调查循环和工具调用来诊断长周期代理轨迹中的错误。

**💡 创新点**

创新点在于用工具增强的LLM和短期记忆机制，使诊断不受上下文窗口限制，并采用原子推理评估提升准确性。

**🔧 技术方法**

使用大语言模型(Claude‑Opus‑4.6)、专门工具集(offset/limit、pattern)、短期记忆模块、原子推理评估器以及多轮工具调用。

**📊 数据集**

在Who&When（算法生成和手工构造子集）和TRAIL（GAIA与SWE子集）两大基准上进行实验。

**📈 对比分析**

与一投递提示、逐步扫描和RAFFLES等基线比较，在1M token预算下提升约20%，在25K token预算下提升约19%，且在5×超窗口的情况下仍保持0.58精度。

**⚠️ 局限性**

局限包括在大上下文可用时短期记忆可能导致信息损失；对高token预算场景下的延迟和信息冲突仍需进一步研究。

---

## 401. QC-SMOTE: Quality-Controlled SMOTE for Imbalanced Classification

**arXiv ID:** 2606.24625 | [PDF](https://arxiv.org/pdf/2606.24625v1)

**作者:** Parth Upman `[一作]`, Shreyank N Gowda `[通讯]` (University of Nottingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种质量控制的 SMOTE 方法（QC‑SMOTE），通过评估少数类样本的可靠性并在生成时筛选高质量候选点，改进不平衡数据集的合成样本质量。

**💡 创新点**

创新点在于：① 采用三维邻域可信度分数（局部密度、安全水平、与多数类的隔离度）来估计种子样本可靠性；② 引入 IPQ‑guided best‑of‑K 机制，在生成时评估多候选样本的插值纯度与多数类清晰度，从而只保留最优的合成样本；③ 根据重叠度与不平衡度自适应调整插值范围、候选数及退化策略（无合成时复制可靠样本）。

**🔧 技术方法**

技术方法包括：k‑最近邻搜索、可靠性加权分配、IPQ 纯度与清晰度度量、最佳‑K 选择、退化复制机制以及基于阈值的可接受性判定。

**📊 数据集**

使用了 30 个真实世界二分类不平衡数据集（来源于 VS‑SMOTE 基准），覆盖 IR 从 1.82 到 32.73 的不同不平衡程度。

**📈 对比分析**

与 8 种 SMOTE 变体（SMOTE、Borderline‑SMOTE、K‑SMOTE、RSMOTE、SMOTE‑WB、SMOTE‑CD、VS‑SMOTE）在 CatBoost 分类器上进行反复分层交叉验证比较。QC‑SMOTE 在 AUC‑ROC 和 Macro F1 上均取得最高平均分（0.9321 与 0.8408），在 29/30 份数据集上获胜，并在高不平衡区间表现尤为显著；同时运行时开销低于 VS‑SMOTE，约 39 倍慢于 SMOTE。

**⚠️ 局限性**

局限性包括：对邻域大小 k、候选数 K 等超参数敏感；目前仅支持数值型特征，未针对混合类型数据扩展；在极端噪声或复杂高维空间中，可靠度估计与候选筛选仍可能不足，需进一步研究模型反馈或在线自适应机制。

---

## 402. Themis: An explainable AI-enabled framework for Reinforcement Learning with Human Feedback

**arXiv ID:** 2606.24622 | [PDF](https://arxiv.org/pdf/2606.24622v1)

**作者:** Andreas Chouliaras `[一作]` (University College Dublin), Dimitris Chatzpoulos `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了一个名为Themis的可配置RLHF框架，集成了XAI技术和可扩展的云端众包平台，用于从人类偏好中训练奖励模型并支持超过200个常用RL环境。

**💡 创新点**

创新点在于将可解释AI（XAI）方法与RLHF紧密结合，提供自动算法选择、奖励模型训练、多人参与的实时反馈收集，以及低成本可扩展的云端基础设施，形成了首个面向非LLM任务的全流程RLHF与XAI集成平台。

**🔧 技术方法**

采用Soft Actor-Critic（SAC）与其离散版、Hydra配置框架、Captum中的Integrated Gradients、Kernel SHAP与TracInCP、Bradley‑Terry偏好模型、MongoDB、Azure Blob存储、Docker/Kubernetes部署。

**📊 数据集**

使用Gym/Atari环境（Pong、MsPacman、Breakout）进行实验，并通过合成教师生成的偏好数据进行奖励模型训练。

**📈 对比分析**

通过将基于真实奖励训练的Agent与基于奖励模型训练的Agent进行对比，评估累计奖励和Episode时长差异；实验表明奖励模型在大部分游戏中至少与真实奖励相当，甚至在MsPacman和Breakout中略有超越；平台在峰值1000用户时平均响应时间<0.1 s，成本低至约0.15 美元/小时。

**⚠️ 局限性**

局限性包括：仅在少数Atari游戏上验证，缺乏真实人类参与者的长周期评估；XAI方法仅限于Captum提供的三种方法，未覆盖所有RL任务；奖励模型训练依赖于合成教师或偏好反馈，可能不适用于所有奖励稀疏场景；平台依赖云服务，跨域适配性待进一步验证。

---

## 403. SupplyNet: Supporting Visual Exploratory Learning in Supply Chain via Contextual Multi-Agent Simulation

**arXiv ID:** 2606.24694 | [PDF](https://arxiv.org/pdf/2606.24694v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 404. Uncertainty-Aware Longitudinal Forecasting of Alzheimer's Disease Progression Using Deep Learning

**arXiv ID:** 2606.24604 | [PDF](https://arxiv.org/pdf/2606.24604v1)

**作者:** Arya Hariharan `[一作]` (R.V. College of Engineering), Anala M R `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种不确定性感知的纵向预测框架，用混合密度网络生成多步概率轨迹，预测阿尔茨海默病从正常认知到MCI再到痴呆的进展。

**💡 创新点**

创新点在于将阶层化诊断预测、Temporal Fusion Transformer编码、混合密度网络的多步生成以及方差分解的 aleatoric 与 epistemic 不确定性估计相结合，且通过 bootstrap 深度集成显著提升了对分布漂移的检测能力。

**🔧 技术方法**

使用了 Temporal Fusion Transformer、CORAL ordinal loss、混合密度网络（MDN）、自回归生成器、深度集成（bootstrap）以及梯度裁剪和学习率调度等技术。

**📊 数据集**

主要在 ADNI 纵向临床影像与认知数据集上训练和评估，并在 OASIS‑3 外部数据集进行零样本泛化验证。

**📈 对比分析**

与线性、LSTM、Transformer 等基线模型相比，TFT‑MDN 在下一个访视诊断预测的 QWK、AUC 等指标上实现了最高性能，并在生成轨迹上满足 90% 可信区间覆盖率、可信区间宽度随时间递增且符合临床生物标志物演化规律。

**⚠️ 局限性**

局限包括对罕见进展模式的预测偏差、CN 组校准不佳、外部验证时缺失部分认知特征导致的迁移损失，以及模型仅在学术中心样本上验证，缺乏对更大、多样化人群的泛化评估。

---

## 405. Automated Summarization of Software Documents: An LLM-based Multi-Agent Approach

**arXiv ID:** 2606.24689 | [PDF](https://arxiv.org/pdf/2606.24689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 406. Almost Symmetric Linear Arc Monadic Datalog and Transitive Tournaments

**arXiv ID:** 2606.24711 | [PDF](https://arxiv.org/pdf/2606.24711v1)

**作者:** Sebastian Meyer `[一作]`, Florian Starke `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 n‑almost symmetric Datalog 这一新的 Datalog 子句，研究其线性弧单子（linear arc monadic）变体 s_nlam，并给出了其可解的约束满足问题（CSP）的结构分类；

**💡 创新点**

主要创新在于将之前关于对称线性弧单子（slam）Datalog 的结果推广到 n‑almost symmetric 情形，并引入 n‑fixed unfolded caterpillar duality 与 elevator chain 等新概念，用以刻画可被 s_nlam 解决的 CSP；

**🔧 技术方法**

核心技术包括原子正构造（primitive positive construction）、多重关系同态双向图、最小子结构（minion homomorphism）与 minor condition 的等价性、以及对 caterpillar 的展开与折叠（unfolding/folding）技术；

**📊 数据集**

由于研究对象为理论结构（有限关系结构、转移锦标赛等），文中未使用实际数据集，而是通过构造性证明和示例图说明；

**📈 对比分析**

论文通过理论证明与反例对比（如 _4 与 _3 的关系、示例图 _313、_414 等），展示了 s_nlam 与普通 slam Datalog 的判定能力差异，并证明了在 n‑fixed unfolded caterpillar duality 下可得到等价的 Datalog 解决方案；

**⚠️ 局限性**

局限性主要在于仅讨论了线性弧单子子句的 n‑almost symmetric 形式，对更一般的非线性或更高 arity Datalog 未给出完整的可解性判定；同时理论结果仍主要针对有限结构，缺乏对无限或更复杂结构的直接扩展。

---

## 407. ArtiTwinSplat: Interactable Digital Twin Reconstruction via Gaussian Splatting from RGB-D videos

**arXiv ID:** 2606.24628 | [PDF](https://arxiv.org/pdf/2606.24628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 408. CineCap: Structured Reasoning with Spatio-Temporal Anchors for Cinematographic Video Captioning

**arXiv ID:** 2606.24636 | [PDF](https://arxiv.org/pdf/2606.24636v1)

**作者:** Xinyu Mao `[一作]` (Chinese University of Hong Kong), Max Meng `[通讯]` (Southern University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一框架，用于对视频进行多维度的电影摄影描述（cinematographic captioning），并构建了首个专门用于评估此任务的基准数据集。

**💡 创新点**

创新点在于：①使用时空锚点将视觉证据转化为结构化推理，形成原子级链式思考（CoT）作为监督；②在强化学习阶段引入覆盖奖励（atomic coverage reward）与门控机制，以平衡描述完整性与事实准确性；③通过LLM裁判进行细粒度的多维度评估。

**🔧 技术方法**

技术上主要采用：基于多模态大语言模型的监督微调（SFT），随后使用Group Relative Policy Optimization（GRPO）进行强化学习；时空锚点与原子CoT用于构造训练样本；评估时利用LLM裁判生成的句子级指标。

**📊 数据集**

数据集为 CineCap Bench，包含 472 条手工标注的视频-字幕对，覆盖 6 个摄影维度（相机运动、镜头大小、拍摄角度、景深、构图、主体朝向）。

**📈 对比分析**

与多种闭源（Gemini‑2.5‑Pro、Gemini‑3.1‑Pro）和开源（Qwen3‑VL‑30B、InternVL3‑8B、LLaVA‑NeXT‑Video‑7B 等）模型对比，CineCap 在整体覆盖度、准确度及 F1 上分别达到 72.38%、74.80% 与 73.57%，比最强基线 Gemini‑3.1‑Pro 提升 21% 以上，表现显著领先。

**⚠️ 局限性**

局限性主要体现在：①基准样本量有限（472 条），对更大规模视频的泛化尚待验证；②评估依赖 LLM 裁判，可能带来偏差；③模型在极长或高动态场景下的时空锚点提取与推理仍有挑战。

---

## 409. ScaleToT: Generalizing Structured LLM Reasoning for Billion-Scale Low-Activity User Modeling

**arXiv ID:** 2606.24605 | [PDF](https://arxiv.org/pdf/2606.24605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 410. Same Lesson, Different Story: Cross-Lingual Reconstruction of Cultural Narratives in Large Language Models

**arXiv ID:** 2606.24610 | [PDF](https://arxiv.org/pdf/2606.24610v1)

**作者:** Jory Alshaalan `[一作]` (King Saud University), Rehab Alahmadi `[通讯]` (King Saud University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建15种语言的谚语对照集并使用四大LLM（Llama‑3、Mistral、Qwen、Phi‑3）生成13k段故事，研究跨语言提示对故事语义保持与叙事重构的影响。

**💡 创新点**

首次将语义相似度与实体权力分布两维度结合，揭示跨语言提示在保留高层道德意义的同时会显著重新分配叙事角色与社会权力，并发现跨模型一致性在多语言情境下依旧强烈。

**🔧 技术方法**

使用多语言嵌入（multilingual‑e5）、余弦相似度评估语义漂移、基于语义嵌入的实体权力评分、Kruskal‑Wallis 与 t‑检验等统计方法。

**📊 数据集**

414条跨语言谚语（15种语言），以及对应的6,346对单语/跨语生成的故事，总计约13,000段文本。

**📈 对比分析**

通过语义漂移指标（Δ_semantic）和实体权力变化衡量，结果显示 LLaMA‑3 语义漂移最小，跨模型相似度在0.78–0.88之间，说明模型在不同语言下保持语义一致性，但叙事层面存在显著重分配。

**⚠️ 局限性**

嵌入方法无法完全捕捉语义细节；未进行人类评估文化真实性；仅关注语义与权力维度，未覆盖其他文化细节。

---

## 411. ASALT: Adaptive State Alignment for Lateral Transfer in Multi-agent Reinforcement Learning

**arXiv ID:** 2606.24601 | [PDF](https://arxiv.org/pdf/2606.24601v1)

**作者:** Anurag Akula `[一作]` (Indian Institute of Technology Madras), Kaushik Dey `[通讯]` (Ericsson Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出ASALT框架，通过观察和状态适配器实现源目标域中观测空间维度不匹配的多源迁移学习，显著提升多智能体强化学习的样本效率和最终奖励。

**💡 创新点**

创新点在于同时为观测和全局状态设计专用适配器，并通过横向传输模块把源域演员与评论家知识映射到共享嵌入空间，从而降低负迁移并支持不同维度的迁移。

**🔧 技术方法**

采用Hierarchical Multi-Head Attention和Transformer编码器构建适配器，结合MAPPO+CTDE框架和联合训练的横向传输机制。

**📊 数据集**

在StarCraft II Multi-Agent Challenge（SMAC）、Google Research Football、以及Multi-Particle Environment（MPE）等标准多智能体基准上进行实验。

**📈 对比分析**

与DANN、CORAL、CycleGAN、LA‑QTransformer、MALT等基准对比，ASALT在观测空间不匹配、缩放及负迁移场景下样本效率提升约30%–70%，最终奖励与最优基准持平或更优。

**⚠️ 局限性**

当状态空间或代理数量差异极大时，评论家迁移可能无效；适配器参数增加导致训练时间略增；缺乏理论收敛性分析。

---

## 412. AdversaBench: Automated LLM Red-Teaming with Multi-Judge Confirmation and Cross-Model Transferability

**arXiv ID:** 2606.24589 | [PDF](https://arxiv.org/pdf/2606.24589v1)

**作者:** Khanak Khandelwal `[一作]` `[通讯]` (Indian Institute of Technology Jodhpur), Khanak Khandelwal (Indian Institute of Technology Jodhpur)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 AdversaBench 自动化红队评估管线，对 45 条不同类别的种子 prompt 进行结构化突变攻击，并通过三评审 + 元评审机制确认失败。

**💡 创新点**

创新点在于：1) 引入多评审与元评审的判定流程；2) 在高类不平衡情境下剖析 Cohen's κ 的误导性；3) 证明攻击对更强模型的迁移性；4) 将迭代成本作为新的难度指标。

**🔧 技术方法**

使用 LangGraph + LangChain 构建管线，5 种突变算子和 epsilon‑greedy 选择，三评审模型（Llama 3.3 70B、Cerebras GPT‑OSS 120B、Qwen3 32B）以及 GPT‑4o‑mini 作为元评审；统计上使用 Cohen's κ、Raw agreement、Survival 曲线等。

**📊 数据集**

自建的 45 条种子 prompt 数据集，按 reasoning、instruction‑following、tool‑use 三类划分；生成的攻击及判定结果保存至 dataset.json。

**📈 对比分析**

在所有 45 条种子上都能确认失败，指令跟随类平均需 2.4 次迭代，而 reasoning 与 tool‑use 类仅需 1.1 次；Raw agreement 高达 80–87%，但 κ 近零；攻击对 70B Llama 模型的迁移率高，显示突变能捕捉通用行为模式。

**⚠️ 局限性**

局限性包括：仅 45 条种子、目标模型仅为 8B，规范未独立验证、epsilon 参数未消融、迁移评估仅对 15 条手工检查，缺乏更强目标和更大样本的实验。

---

## 413. DREAM: Dense Retrieval Embeddings via Autoregressive Modeling

**arXiv ID:** 2606.24667 | [PDF](https://arxiv.org/pdf/2606.24667v1)

**作者:** Yixuan Tang `[一作]` (Hong Kong University of Science and Technology), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自回归下一个词预测的训练框架（DREAM），通过把检索器的查询‑文档相似度分数注入冻结的LLM注意力头中，使得预测损失直接为检索器提供梯度，无需人工标签或对比损失。

**💡 创新点**

创新点在于：①利用注意力头作为检索器与LLM之间的接口，让检索器的分数直接影响LLM对目标文本的生成；②只在查询聚焦的检索头中注入分数，提升监督信号的有效性；③无需显式负样本，候选集合内部竞争即可产生梯度。

**🔧 技术方法**

技术包括：冻结的解码器LLM（如Llama‑3.1‑8B‑Instruct）下一个词预测、检索器双编码器（如Qwen2.5‑0.5B、Llama‑3.2‑1B/3B）、注意力头分数注入与门控混合、LoRA微调、查询聚焦头的选取、温度归一化的候选权重。

**📊 数据集**

训练数据基于Wikipedia的文档切块，使用生成式模型（如Qwen3‑14B）产生查询并标记正向目标；评估数据为公开检索基准BEIR和RTEB。

**📈 对比分析**

与BM25、RePlug、Revela、InfoNCE等基线在BEIR和RTEB上比较，平均NDCG@10提升0.015–0.102（BEIR）和0.068–0.102（RTEB），并在不同模型规模（0.5B–3B）下均优于对比学习和现有LLM监督方法。

**⚠️ 局限性**

局限性：①需要冻结LLM作为判别器，更新LLM会削弱梯度信号；②效果高度依赖查询聚焦注意力头的选择，若选择不当性能显著下降；③训练需要事先构造候选文档集合，计算相似度分数和注意力注入成本较高；④在8B级别仍不及最强的专门训练嵌入模型。

---

## 414. Solvability of Approximate Agreement on Graphs and Simplicial Complexes

**arXiv ID:** 2606.24666 | [PDF](https://arxiv.org/pdf/2606.24666v1)

**作者:** Joel Rybicki `[一作]` (Humboldt University of Berlin), Yaroslav Verbitsky `[通讯]` (Humboldt University of Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了图上近似一致性任务的完整拓扑可解性表征，证明了在异步共享内存模型中，当且仅当图的团复形在相应维度上连通时，t‑可恢复的 clique/monophonic/geodesic 近似一致性协议存在。

**💡 创新点**

首次证明了 Ledent 猜想，即 clique 近似一致性在所有进程下可等待无阻塞当且仅当其团复形是可收缩的；同时揭示了 monophonic 与 geodesic 近似一致性在可解图类上的严格区别，并给出了完整的可解性判定条件。

**🔧 技术方法**

核心技术为拓扑方法：利用颜色无关任务的异步可计算性定理、连通性与同伦类的同构性，以及对复形的强坍塌和可扩展映射的构造，得到从连通性判定到协议可行性的双向证明。

**📊 数据集**

该工作为理论性论文，无实验数据集；所有结论均通过数学证明得到。

**📈 对比分析**

与以往仅针对特定图类（如树、块图、桥图等）的经验性或可构造协议不同，本研究提供了一条统一的、基于同伦连通性的判定标准，能够在任意图上决定协议是否存在，且证明了判定问题在 t ≥ 2 时是不可判定的，从而说明了任何基于连通性检查的算法都无法在所有情况下给出决策。

**⚠️ 局限性**

主要局限在于：虽然给出了可解性判定的必要与充分条件，但相应的协议构造是非构造性的，无法直接得到具体的高效实现；此外，连通性判定本身在 t ≥ 2 时不可判定，导致无法在实际应用中快速识别可解图类。

---

## 415. Optimization-based Safe Trajectory Planning for Autonomous Ground Vehicle in Multi-Floor Scenarios

**arXiv ID:** 2606.24631 | [PDF](https://arxiv.org/pdf/2606.24631v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 416. AI-PAVE-Br: Leveraging Large Language Models for Enhanced Product Attribute Value Extraction through a Golden Set Approach

**arXiv ID:** 2606.24655 | [PDF](https://arxiv.org/pdf/2606.24655v1)

**作者:** Murilo Gazzola `[一作]` (LuizaLabs), Caio Gomes `[通讯]` (LuizaLabs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了基于大型语言模型的产品属性值提取系统（AI-PAVE-Br）并构建了首个葡萄牙语PAVE黄金数据集（Golden Set），旨在提升巴西电商平台的商品数据质量。

**💡 创新点**

创新点在于：① 专为巴西电商场景设计的LLM提示工程策略，使模型能在非英语环境下准确提取多种属性；② 发布了高质量、手工标注的Golden Set，为葡萄牙语PAVE提供可复现的基准。

**🔧 技术方法**

主要技术包括：使用Google Gemini 1.5 Flash LLM、精心设计的提示工程（prompt engineering）、传统NER基线对照、以及对结果进行标准化的正则表达式处理。

**📊 数据集**

使用的数据集为Golden Set，涵盖20类产品（如空调、电视、手机等），每类约385条样本，总计约8万条手工标注的属性记录。

**📈 对比分析**

评估方法：采用精度、召回、F1分数及覆盖率。实验显示，LLM基线平均F1为74.68%，覆盖率71.96%，相较传统NER基线的59.79% F1和46.71%覆盖率有显著提升。

**⚠️ 局限性**

限制包括：① 属性值的规范化和一致性仍不完善，导致错误或重复；② 某些类别（如洗衣机、冰箱）表现不如传统基线；③ 对新出现的属性或长尾类别需要进一步的提示调整或微调；④ 依赖单一LLM模型，成本和延迟仍是实际部署的考量。

---

## 417. ParaPairAudioBench: Paralinguistic Pairwise Audio Benchmark for LALM-as-a-Judge

**arXiv ID:** 2606.24648 | [PDF](https://arxiv.org/pdf/2606.24648v1)

**作者:** Jisu Jeon `[一作]` (Hongik University), Soyoon Kim `[通讯]` (NAVER Cloud)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为ParaPairAudioBench的配对式音频基准，用于评估大型音频-语言模型(LALM)在五个抑扬顿挫维度（风格、语速、重音、年龄、性别）上的判别能力；

**💡 创新点**

创新点在于：1）引入Tie（平局）判定以检测模型的校准性；2）通过同词/跨词对比剖析模型对音频与文本信息的依赖；3）使用交叉排序评估顺序偏差；4）提供细粒度、多维度的诊断评估，而非单一自然度得分；

**🔧 技术方法**

使用了多模态提示（同时输入两个音频和文本指令），并对比了 Gemini 2.5 Flash、GPT‑4o Audio、SpeechJudge‑7B、Kimi‑Audio‑7B、Qwen2.5‑Omni‑7B 等模型；

**📊 数据集**

数据集来自公开语音语料：Expresso（风格、重音）、Sonos Voice Control Bias Assessment（年龄、性别）、LibriTTS（性别）、EARS（语速），共5,175对样本；

**📈 对比分析**

与50名人工评审者对比，模型平均落后人类约32%（平均差距 32%p），在多数维度下表现低于人类，尤其在Tie判定和位置偏差上表现突出；

**⚠️ 局限性**

局限性包括：1）Tie构造仅覆盖部分维度，未覆盖所有可能的平局情形；2）对同词/跨词对的区分可能掩盖了实际语音差异；3）模型对局部重音的敏感度不足；4）评估主要集中在配对式判定，未涵盖更大范围的自然度评估；5）存在显著的顺序偏差和校准失误。

---

## 418. Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation

**arXiv ID:** 2606.24633 | [PDF](https://arxiv.org/pdf/2606.24633v1)

**作者:** Xinyao Qin `[一作]` (Tsinghua University), Li Zhao `[通讯]` (Microsoft Research Asia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用稀疏重试关键点作为监督，学习对错误敏感的价值函数，并用该价值函数对混合质量演示进行行为克隆重加权。

**💡 创新点**

创新点在于将重试事件转化为局部偏好监督，结合全局进度校准与软窗口加权，提升对细粒度错误和纠正行为的识别。

**🔧 技术方法**

采用基于视觉语言模型的分布式价值预测网络，结合绝对进度损失、重试偏好损失和软窗口权重的联合训练。

**📊 数据集**

在四个真实机器人操作任务（拿勺子、堆块、折毛巾、开抽屉）上收集的约200条混合质量演示和30条标注重试点的数据。

**📈 对比分析**

与TOPReward、Robometer、RECAP-Value等基线对比，在全局进度指标上与最优基线相当，所有局部错误敏感指标均显著优于基线；在行为克隆下平均成功率提升至80%，超过标准BC 41%和RECAP-BC 63%。

**⚠️ 局限性**

局限在假设重试遵循降级-恢复模式，未考虑探索性重试；仅用于离线行为克隆，未探索闭环或在线强化学习；实验规模受限，需在更大任务和机器人上验证。

---

## 419. TACTFUL: Tactile-Driven Exploration For Object Localization and Identification in Confined Environments

**arXiv ID:** 2606.24712 | [PDF](https://arxiv.org/pdf/2606.24712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 420. Toward Self-Evolution-Ready Workflow Harnesses: A Reversible Migration Path and Convertibility Taxonomy for Expert LLM Pipelines

**arXiv ID:** 2606.24598 | [PDF](https://arxiv.org/pdf/2606.24598v1)

**作者:** Yimo Lin `[一作]` (Yunyong Century Artificial Intelligence Technology Co Ltd), Yibin Li `[通讯]` (Yunyong Century Artificial Intelligence Technology Co Ltd)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将已有的专家“LLM+脚本”工作流迁移为可逆、可审计、可自演化的可组合执行平台；

**💡 创新点**

提出可逆Strangler‑Fig迁移路径、转换可行性A/B/C分类诊断阶段以及基于结果的早期自演化信号；

**🔧 技术方法**

使用Strangler‑Fig增量替换、工具化（Toolification）与Typed Stages、ReAct式决策代理、Deterministic Safety Gates、Structured Trace与一旗回滚等技术；

**📊 数据集**

以微信公众号内容生成工作流为案例，使用内部生产数据（文章列表、点击率、收入等指标）进行验证；

**📈 对比分析**

对比子进程版与代理版的可解释性、可调性与成本；迁移成本无业务逻辑改动、零上线中断；早期自演化信号显示主题选择改进约+58%点击率、收入正向增长，但未做因果实验；

**⚠️ 局限性**

仅有单一案例、缺乏统计显著性、对自演化的证明仅为早期策略级信号、模型/平台依赖未验证、外部提示注入安全未覆盖。

---

## 421. Qwen-AgentWorld: Language World Models for General Agents

**arXiv ID:** 2606.24597 | [PDF](https://arxiv.org/pdf/2606.24597v1)

**作者:** Yuxin Zuo `[一作]`, Ning Ding `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一的语言世界模型（LWM），可对七类交互式环境（MCP、Search、SWE、Terminal、Android、Web、OS）进行长链思维的状态预测，并通过三阶段训练（CPT→SFT→RL）实现高保真模拟。

**💡 创新点**

创新点包括：①首次构建跨七域的语言世界模型并形成通用基座；②通过信息论损失屏蔽和五维评判指标的混合奖励实现 RL 的稳定收敛；③两种应用范式：解耦式模拟器实现可扩展可控的环境仿真，统一式 LWM 预训练作为 agent 基础模型提升多域代理性能。

**🔧 技术方法**

核心技术包括：大规模连续预训练（CPT）注入环境动力学与专业语料；监督式微调（SFT）激活“下一状态预测”思维模式；强化学习（RL）使用五维评判+规则校验的奖励；统一轨迹 schema 与系统 prompt 自动化生成；对多域交互日志的多源采集与清洗；以及可控仿真指令的自动化设计。

**📊 数据集**

使用了超过10M条跨7域的真实交互轨迹（容器化终端、GUI 访问树、API 调用记录等），以及专业知识语料库（工业、医疗、法律等）。此外，构建了 LWMBench 评测集，基于 2,170 个真实轨迹样本覆盖 9 个现有基准的 7 个领域。

**📈 对比分析**

与 14 组基线（Claude、GPT、Gemini、DeepSeek、Qwen 等）对比，-397B-A17B 在 LWMBench 的五维平均得分达 58.71，超过 GPT‑5.4（58.25）和所有主流专有模型。Sim‑RL 采用 LWM 作为模拟器，在 Claw‑Eval、QwenClawBench 上分别提升 4.3 / 7.1 分；在 MCPMark、WideSearch 上通过可控仿真提升 12.3 / 16.3 分；作为 agent 基础模型，LWM‑RL 预训练在 7 个基准上平均提升 6–15 分。

**⚠️ 局限性**

局限性包括：①事实性（Factuality）仍是最薄弱维度，仍有误差；②训练数据覆盖仅 7 个领域，跨域泛化仍受限；③RL 收敛依赖复杂奖励设计，训练成本高；④可控仿真需手工指令，自动化程度有限；⑤在高度动态或实时交互场景下的延迟与一致性仍需进一步验证。

---

## 422. FlowPipe: LLM-Enhanced Conditional Generative Flow Networks for Data Preparation Pipeline Construction

**arXiv ID:** 2606.24679 | [PDF](https://arxiv.org/pdf/2606.24679v1)

**作者:** Kunyu Ni `[一作]` (Ocean University of China), Yanwei Yu `[通讯]` (Ocean University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

FlowPipe 自动化构建数据预处理流水线，利用 LLM 提供语义上下文进行条件生成。

**💡 创新点**

创新点在于把流水线生成建模为条件生成流网络，结合 FiLM 深度语义调制和失败感知的轨迹平衡目标，实现结构统一、语义依赖和高效探索。

**🔧 技术方法**

使用技术包括条件 GFlowNet、FiLM 语义调制、LLM 生成的语义向量、轨迹平衡训练、失败感知权重。

**📊 数据集**

在 74 个真实数据集（OpenML、UCI、Kaggle）上进行实验。

**📈 对比分析**

与 SOTA 多 DQN、AutoML、进化搜索等基线对比，平均准确率提升 11.96%，训练收敛速度提升 12.5 倍，在线推理时间显著降低。

**⚠️ 局限性**

限制在于仍需依赖 LLM 生成的语义向量，对极端稀疏或缺失信息的处理可能受限；同时在多任务下的泛化尚未深入探究。

---

## 423. LaGO: Latent Action Guidance for Online Reinforcement Learning

**arXiv ID:** 2606.24669 | [PDF](https://arxiv.org/pdf/2606.24669v1)

**作者:** Kuan-Yen Liu `[一作]` (University of Illinois Urbana-Champaign), Ti-Rong Wu `[通讯]` (Academia Sinica)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LaGO 框架，将预训练的 LLM 作为潜在动作先验，用来软性指导在线强化学习策略的优化。

**💡 创新点**

创新点在于不把 LLM 当作直接控制器，而是将其隐层表示作为柔性先验来正则化策略学习，降低对 LLM 输出精度的依赖。

**🔧 技术方法**

采用 Llama‑2‑7b‑chat‑hf 作为语言骨干，利用投影层把状态映射到 LLM 隐层，使用 PPO 结合 KL 正则化进行在线训练。

**📊 数据集**

使用 CLEVR‑Robot（离散动作）和 Meta‑World（连续动作）两个基准数据集。

**📈 对比分析**

与 Vanilla PPO 对比，LaGO 在两套任务上均显著提升奖励和成功率：CLEVR‑Robot 成功率从 15.1% 提升至 27.2%，Meta‑World 从 2.7% 提升至 15.2%。

**⚠️ 局限性**

局限性包括对 LLM 质量的高度依赖、先验权重需要手动调节、仅在两类基准验证，未测试更长时序或更复杂任务。

---

## 424. Accuracy and Satisfaction in Multi-Turn LLM Dialogues for NFR Assessment

**arXiv ID:** 2606.24834 | [PDF](https://arxiv.org/pdf/2606.24834v1)

**作者:** Ali Pourghasemi Fatideh `[一作]` (University of Maine), Sepideh Ghanavati `[通讯]` (University of Maine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过让开发者与基于GitHub Copilot的LLM助手进行多轮对话，评估 HIPAA 相关非功能需求的满足情况。

**💡 创新点**

创新之处在于将多轮对话质量与 NFR 评估相结合，并首次利用 PARADISE 框架揭示对话特征对用户满意度的影响。

**🔧 技术方法**

技术上使用了 GitHub Copilot（gpt‑5.1‑codex‑max）生成回答，BERTScore/ROUGE 评估推理质量，并通过 PARADISE 框架构建用户满意度性能函数。

**📊 数据集**

数据集包括从 HIPAA 规范提取的 148 条 NFR 与 iTrust 代码库的对应关系，以及专家标注的三维真值（满足级别、推理、代码位置）。

**📈 对比分析**

对比方法是将助手答案与专家真值计算 F1、BERTScore 等指标；结果显示满足级别 F1 仅 0.381，推理由 BERTScore 0.520，代码定位 F1 0.203，尽管开发者对回答的主观满意度超过 90%。

**⚠️ 局限性**

局限性包括仅评估单一 LLM 与 iTrust 代码库，样本量有限，NFR 语料由单名专家提取，且用户满意度函数可能夸大不准确信息。

---

## 425. High-Fidelity Synthetic Transmission Electron Microscopy Image Generation Using Diffusion Probabilistic Models for Data-Limited Semiconductor Metrology

**arXiv ID:** 2606.24817 | [PDF](https://arxiv.org/pdf/2606.24817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 426. Pocket-SLAM: Rendering-Area-Aware Pruning for Memory-Efficient 3DGS-SLAM

**arXiv ID:** 2606.24796 | [PDF](https://arxiv.org/pdf/2606.24796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 427. Assessing Distribution Shift in Human Activity Recognition for Domain Generalization

**arXiv ID:** 2606.24781 | [PDF](https://arxiv.org/pdf/2606.24781v1)

**作者:** Rebecca Adaimi `[一作]` (University of Texas at Austin), Edison Thomaz `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了四类传感器相关的分布偏移（设备类型、传感器位置、采样率、用户行为）对人类活动识别（HAR）模型泛化的影响，并提出统一的基准平台来测试域泛化与域适应算法；

**💡 创新点**

创新点在于：①首次将不同类型的分布偏移拆分并量化；②构建统一的 HAR 基准框架，公开数据集与评测代码；③对 28 种域泛化方法及 11 种域适应方法进行大规模实验，揭示其在 HAR 上的局限性；

**🔧 技术方法**

采用深度神经网络特征提取（Vggish）、域判别器、元学习、对抗学习、正则化、数据增强等 28 种域泛化技术与 11 种域适应技术；

**📊 数据集**

使用公开 HAR 数据集（PAMAP2、DSADS、Opportunity、RealWorld、HHAR）以及自采集的抛球数据集；

**📈 对比分析**

通过 leave‑one‑domain‑out 验证，计算 in‑accuracy 与 out‑accuracy，统计排名与相对优势；结果表明大多数方法对 ERM 的提升有限，性能提升仅为几个百分点，且多数情况下未显著优于 ERM；

**⚠️ 局限性**

局限性包括：①仅考虑单一偏移类型，未覆盖多重混合偏移；②自采集的数据集规模有限，缺乏长期时序；③实验固定特征提取器和交叉熵损失，未充分探索超参、其他损失或更强特征学习；

---

## 428. UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving

**arXiv ID:** 2606.24759 | [PDF](https://arxiv.org/pdf/2606.24759v1)

**作者:** Xiaowei Gao `[一作]` (Imperial College London), Yun Ye `[通讯]` (University College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 UniDrive 框架，实现统一的视觉-语言与定位，用于解释性风险理解

**💡 创新点**

创新点：双分支架构融合时序推理与高分辨率感知，采用门控跨注意力融合，并将定位直接嵌入文本输出

**🔧 技术方法**

技术：CLIP ViT-L/14 编码器、Flamingo式 LLM（Llama2-7B）、门控交叉注意力、SFT 细化、基于文本的边界框令牌

**📊 数据集**

数据集：扩展后的 DRAMA‑Reasoning（风险说明+定位），以及 NuScenes、BDD100K 用于零样本泛化评测

**📈 对比分析**

与多种基线比较：在 DRAMA‑Reasoning 上 Captioning 与检测任务均超过所有基线，CIDEr 提升至 277.5，mIoU 达到 61.2；零样本 NuScenes 准确率提升至 75.3%，BDD100K mAP 提升至 71.4，显示出更好的小物体定位和鲁棒性

**⚠️ 局限性**

限制：在多风险共现时可能优先错误对象，仍受光照/遮挡影响，未整合地图、轨迹预测等多模态信息；仅评估单一风险对象，缺乏多风险层级和闭环仿真验证

---

## 429. CANDLE: Character-level Arabic Noise Deduplication using Lightweight Encoder

**arXiv ID:** 2606.24758 | [PDF](https://arxiv.org/pdf/2606.24758v1)

**作者:** Faris Alasmary `[一作]` (Abjad Ltd), Lahouari Ghouti `[通讯]` (Abjad Ltd)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CANDLE系统，使用CTC对齐实现阿拉伯语文本中字符重复的自动消除与规范化；

**💡 创新点**

首次将CTC序列对齐方法应用于字符级重复消除，并通过模型蒸馏压缩得到轻量化的学生模型；

**🔧 技术方法**

核心技术包括字符级BERT编码器、CTC损失训练、分类基线对比以及软硬目标蒸馏；

**📊 数据集**

使用基于1.5B词阿拉伯语语料的训练集，并在三大基准集（NewsText、AmbigText、WildSAText）进行评估；

**📈 对比分析**

与分类基线比较，CTC模型在NewsText、AmbigText、WildSAText的句子错误率分别为5.37%、23.08%和9.93%，远优于分类基线；蒸馏后模型仅略微下降，且显著降低tokenizer肥胖率至12.8%；

**⚠️ 局限性**

仅适用于纯阿拉伯语文本，且假设任何词中字符最多连续出现两次，无法正确处理三次以上合法重复。

---

## 430. World Value Models for Robotic Manipulation

**arXiv ID:** 2606.24742 | [PDF](https://arxiv.org/pdf/2606.24742v1)

**作者:** Zhihao Wang `[一作]` (ByteDance Seed), Xiao Ma `[通讯]` (ByteDance Seed)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于预训练视频世界模型的通用机器人价值流模型，能够估计视频任务进度并识别子最优行为。

**💡 创新点**

创新点在于将世界模型的时空先验迁移为价值估计，使用分布式价值块与流匹配训练，配合MoT架构、前缀随机化与视频倒带增强，实现对子最优轨迹的高精度评估。

**🔧 技术方法**

使用预训练视频VAE + Video DiT、轻量级价值 DiT、Mixture-of-Transformers、流匹配 (flow matching) 训练、前缀随机化和视频倒带等技术。

**📊 数据集**

数据集包括800条多体制化人类标注的子最优轨迹（Suboptimal-Value-Bench）以及标准Expert-VOC轨迹，用于训练和评估。

**📈 对比分析**

与 GVL、VLAC、Robometer、TopReward、RoboReward、Robo-Dopamine 等基线相比，在 Hesitation-RMSE、Retry-VOC 和 Expert-VOC 上均取得最高分，且在下游策略学习中显著提升性能。

**⚠️ 局限性**

局限性包括训练数据规模有限，零样本泛化能力弱，实验主要聚焦于抓取-放置任务，未覆盖更复杂的多手指或长时序操作。

---

## 431. Grad Detect: Gradient-Based Hallucination Detection in LLMs

**arXiv ID:** 2606.24790 | [PDF](https://arxiv.org/pdf/2606.24790v1)

**作者:** Anand Kamat `[一作]` (Amazon), Brent M. Werness `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Grad Detect，通过在推理时对每个 transformer 层的梯度进行分析，预测 LLM 的回答是否为幻觉或是否拒绝回答。

**💡 创新点**

创新点在于利用梯度方向而非输出概率或隐藏状态，构建类别特定的参考梯度并用余弦相似度压缩为低维特征，从而实现单通道、高效的幻觉检测。

**🔧 技术方法**

主要技术包括梯度提取、参考梯度构造、余弦相似度特征化、轻量级 transformer 编码器进行分类，并在单一次前向-后向传播中完成。

**📊 数据集**

实验使用四个问答基准：TriviaQA、SciQ、PopQA 和 TruthfulQA，并评估了 11 种 1B-12B 规模的指令微调 LLM。

**📈 对比分析**

与基准方法（自我评估、置信度、内部状态探测、Self‑Consistency、Semantic Entropy 等）比较，Grad Detect 在幻觉检测上达 71–78% 准确率、94–99% 报答预测准确率，显著优于单通道方法，且仅需 1.5–2× 计算成本。

**⚠️ 局限性**

主要局限包括需白盒访问模型参数、额外的后向传播导致 50–100% 的推理时间增加、依赖 LLM 判断生成的自动评估，且目前仅验证于密集 transformer（未验证混合专家模型或多模态 LLM）。

---

## 432. Task Decomposition for Efficient Annotation

**arXiv ID:** 2606.24734 | [PDF](https://arxiv.org/pdf/2606.24734v1)

**作者:** Nupoor Gandhi `[一作]` (Carnegie Mellon University), Emma Strubell `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出通过把结构化注释任务拆分成中心识别与增补子任务，以降低整体推理负担并在异质注释者（人类与模型）之间高效分配工作。

**💡 创新点**

创新点在于将中心化理论融入推理负荷量化模型，并给出分解准则与预算约束下的注释者分配与蒸馏优化算法。

**🔧 技术方法**

主要技术包括推理负荷加权模型（输入空间、映射空间、输出空间与中心引入）与基于线性/整数规划的注释者分配与蒸馏方案。

**📊 数据集**

实验以医疗文本集 CRAFT 与政策文本集 POLIANNA 为基准，验证拆分后注释时间缩短、质量提升，亦参照自身最近工作中的实验数据。

**📈 对比分析**

与完整任务对比，拆分后在注释时长与专家一致率方面均显著提升；虽然未给出具体数值，但作者报告“更低的推理负荷与更高的注释质量”。

**⚠️ 局限性**

局限包括未考虑原子粒度的歧义、上下文重复处理成本、不同注释者对信息处理的敏感度差异，以及未覆盖所有结构化标注任务。

---

## 433. SciFi-VIS: Way Out There -- How SciFi and Visualization Influence Each Other

**arXiv ID:** 2606.24731 | [PDF](https://arxiv.org/pdf/2606.24731v1)

**作者:** Ulrik Günther `[一作]` (Helmholtz-Zentrum Dresden-Rossendorf), Annalena Ulschmid `[通讯]` (TU Wien)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

组织了一场关于科幻媒体与数据可视化交叉的工作坊，邀请学术界与产业界共同探讨灵感与实践。

**💡 创新点**

首次专门聚焦科幻作品对可视化设计的影响，并将可视化研究与影视游戏界的创作者桥接。

**🔧 技术方法**

采用工作坊形式，包括主题报告、闪电演讲、白板/绘图、AI 图像生成、共创脑图、亲手设计新界面等互动方式。

**📊 数据集**

无数据集；以科幻电影、电视剧、游戏场景作为案例。

**📈 对比分析**

未进行实验对比；主要通过参与者讨论与共创产出进行定性评估。

**⚠️ 局限性**

缺乏定量评价与实证验证，受限于时间、人数与参与者多样性；工作坊成果需要后续跟进。

---

## 434. Grading the Grader: Lessons from Evaluating an Agentic Data Analysis System

**arXiv ID:** 2606.24839 | [PDF](https://arxiv.org/pdf/2606.24839v1)

**作者:** Tian Zheng `[一作]` (Columbia University), Kai-Tai Hsu `[通讯]` (Columbia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何可靠评估多代理数据分析系统（LAMBDA）的输出，并构建了三层人机分级流水线来评判153个QRData标量任务的答案是否与真值一致。

**💡 创新点**

提出了“nudging”提示和关键词锚定提取管线以显著提升分级召回率，并将非生成式严格分级与生成式宽容分级结合，形成可复现、低假阳性的混合评估方案；同时通过任务属性（变量类型）解析分级动态与结果的关联。

**🔧 技术方法**

使用两层循环包装器、Per‑step instrumentation、nudging机制、关键词锚定提取、LLM（GPT‑4o‑mini）宽容分级、严格正则匹配以及片段级人工审核。

**📊 数据集**

采用DSGym QRData的153个带标量真值的任务（共251题）作为评估数据集。

**📈 对比分析**

与人工标注对比，所有自动分级器精度均为100%；关键词锚定提取的严格分级召回率从26%提升至86%；宽容分级召回率达到97%；nudging将分级成功率从36%提升至97%。

**⚠️ 局限性**

仅针对单一系统（LAMBDA）与单一标量任务；关键词提取依赖词汇重叠，可能在不同表述上失效；未覆盖多答案或复杂格式的任务；模型族共享偏差的影响未充分评估；稳定性实验规模有限。

---

## 435. Difference-Making without Making a Difference

**arXiv ID:** 2606.24832 | [PDF](https://arxiv.org/pdf/2606.24832v1)

**作者:** Sander Beckers `[一作]` `[通讯]` (University College London), Sander Beckers (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

对 Andreas 与 Günther 在七篇论文中提出的七种实际因果定义进行系统比较，并证明最新定义 AG7 与早期定义 AG5 等同，从而指出其所谓的三类因果模型（事实差异、反事实差异、规律性）之间不存在实质区别。

**💡 创新点**

核心创新在于通过形式化证明展示 AG7 与 AG5 的等价性，进一步证明 AG7 可以被视为包含反事实差异的实例，揭示三类因果理论的“无差异”假设；并指出 AG 研究中对关键例子（Early Preemption 与 Simple Switch）的判定反复无常，削弱其理论连贯性。

**🔧 技术方法**

使用逻辑与结构模型（因果模型 M,V 的形式化定义、语义推理、Monotonicity 等）进行形式化证明，并借助子模型与结构子图的最小化条件进行比较。

**📊 数据集**

本研究不涉及外部数据集，而是基于理论模型与经典因果例子（如 Early Preemption、Simple Switch 等）进行分析。

**📈 对比分析**

通过对七种定义在同一组经典因果例子上的判定结果进行对比，评估其一致性与差异；结果显示 AG7 与 AG5 在判定上完全一致，且三类因果理论在判定上无明显优劣差别。

**⚠️ 局限性**

局限性包括：依赖于作者对“直觉判定”的假设（如对 Early Preemption 与 Simple Switch 的直觉一致性），以及未能提供更广泛、经验性的验证；此外，研究聚焦于形式化证明，缺少对实际因果推理任务的实证评估。

---

## 436. Less is More: Quality-Aware Training Data Selection for Scientific Summarization

**arXiv ID:** 2606.24828 | [PDF](https://arxiv.org/pdf/2606.24828v1)

**作者:** Maria Nefeli Paraskevopoulou `[一作]` (Aristotle University of Thessaloniki), Grigorios Tsoumakas `[通讯]` (Aristotle University of Thessaloniki)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了规模最大、结构完整的 PMC‑Large 数据集，并分析了作者写作摘要的参考质量，提出基于质量的训练样本筛选方法，显著提升长篇科学摘要的事实性和整体性能。

**💡 创新点**

创新点在于①发布 1.88 M 条 PMC 论文-摘要对并保留层次结构；②系统评估多种源‑基准评估指标对摘要质量的差异；③提出多指标联合筛选的两阶段训练样本策略。

**🔧 技术方法**

使用的技术包括：Markdown‑style 文本序列化、三步内容过滤、AlignScore、FineSurE、SummaC‑ZS、G‑Eval 等评估模型、Qwen2.5‑3B 进行监督微调以及基于排序的样本选择。

**📊 数据集**

使用的数据集为 PMC‑Large（1.88 M 论文‑摘要对）以及 10 000 条抽样子集用于质量评估；对比数据为 PubMed、SumPubMed 等传统小规模数据集。

**📈 对比分析**

方法通过在 500 条测试集上采用 AlignScore、SummaC、G‑Eval、FineSurE 四项指标评估，结果显示：质量筛选后的 1K 示例即可匹配或超越 5K 随机样本，甚至在 10K‑级别的候选池中，2K‑级别的质量样本已在整体得分上超过完整 100K 训练集。

**⚠️ 局限性**

局限性包括：评估仅依赖自动指标，缺乏人工验证；仅在 3B 模型和单一微调设置下验证，未探讨更大模型或多任务；数据集限制于英语、开放授权的 PMC 论文，可能无法推广至其他语言或非开放数据。

---

## 437. BluTrain: A C++/CUDA Framework for AI Systems

**arXiv ID:** 2606.24780 | [PDF](https://arxiv.org/pdf/2606.24780v1)

**作者:** Adhitya Charan `[一作]` (BluBridge Research), Surendra Vendra `[通讯]` (BluBridge Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了基于 C++/CUDA 的全原生分布式训练框架 BluTrain，用于高效训练 Transformer 模型。

**💡 创新点**

通过实现完整的张量、算子、分布式运行时和 MLIR 编译器，实现绝对硬件控制、显存优化和数值保真，显著提升吞吐量与内存利用率。

**🔧 技术方法**

C++/CUDA 原生实现、BluBLAS 自研 GEMM 库、定制缓存分配器、MLIR 深度学习编译器、DTMS（分布式训练管理系统）等技术。

**📊 数据集**

使用 10B FineWeb‑Edu 文本数据集训练 GPT‑2 124M，单卡 48GB 用于验证更大模型。

**📈 对比分析**

与 PyTorch（eager 与 compile）对比，8 GPU Ada 上 GPT‑2 124M 训练吞吐量提升约 3%（407K vs 395K tokens/s），显存占用减少 22%，且能在单卡上训练 2.42B 参数模型。

**⚠️ 局限性**

仅在 GPT‑2 规模验证，缺乏细粒度子系统消融；故障恢复机制尚未在异构硬件与多节点场景下充分验证；分布式扩展到多节点的性能与容错性仍待评估。

---

## 438. Are We Ready For An Agent-Native Memory System?

**arXiv ID:** 2606.24775 | [PDF](https://arxiv.org/pdf/2606.24775v1)

**作者:** Wei Zhou `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 LLM 代理记忆系统进行了系统的实验评估，提出从数据管理角度拆解为四大模块（表示与存储、抽取、检索与路由、维护）并对 12 种代表性系统在 3 个基准（LoCoMo、LongMemEval、DB‑Bench）以及 11 个数据集上进行统一跑时与多指标评测。

**💡 创新点**

创新点在于：①构建统一的四模块分析框架与细粒度实验流程；②从成本–性能、检索完整性、动态更新鲁棒性和长期稳定性等维度重新定义评测指标；③揭示不同工作负载下最优结构化/分层混合策略，并给出局部维护成本高效、全局重组成本昂贵的经验法则。

**🔧 技术方法**

主要技术包括：多引擎存储（向量、图、关系、SQL 等）与混合检索；语义抽取与结构化提取；规划驱动的检索路由；多阶段维护策略（冲突解决、增量压缩、版本化）；统一跑时跟踪与操作成本量化。

**📊 数据集**

使用了 11 个公开数据集，涵盖长对话、跨会话记忆、数据库执行等任务：LoCoMo、LongMemEval、DB‑Bench、LongBench、LoCoMo‑Gap 等，确保评测覆盖短期/长期、结构化/非结构化、多任务场景。

**📈 对比分析**

比较方法采用统一时间跟踪、六个指标（EM、F1、ROUGE‑L、检索召回、操作延迟等），结果显示没有单一体系结构在所有任务中最优；结构化与分层混合系统在多任务上取得最高平均分；局部维护策略（如 LightMem、MemTree）在成本–性能上表现最佳；全局重组的高层次结构化系统虽然准确率高但延迟与成本显著提升。

**⚠️ 局限性**

局限性包括：缺乏真实持续运行时间长的多任务实验；对生成质量与记忆互依关系的细粒度分析不足；未覆盖多模态或参数化记忆的评测；以及对跨任务迁移和更大规模记忆（百万级）时的可扩展性与一致性验证有限。

---

## 439. Time-varying Wireless Channel Tracking with Online Parameter Learning via the Birth-Death-Drift Model

**arXiv ID:** 2606.24727 | [PDF](https://arxiv.org/pdf/2606.24727v1)

**作者:** Tiancheng Gao `[一作]` (University of Manitoba), Amine Mezghani `[通讯]` (University of Manitoba)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 BDD-VAMP-EM 的算法，能够在大规模 MIMO 系统中通过联合估计信道状态和 BDD 模型参数，实现自适应的时间变信道跟踪。

**💡 创新点**

创新点包括：①将出生-死亡-漂移 (BDD) 模型与向量 AMP (VAMP) 结合，克服 AMP 对 i.i.d. 高斯感知矩阵的限制；②在 VAMP 迭代内部嵌入 EM 步骤，实现在线自适应参数学习；③使用伯努利-高斯混合分布 (BGM) 作为后验，保持更丰富的统计结构，从而提高对稀疏动态信道的利用。

**🔧 技术方法**

技术手段主要有：Birth‑Death‑Drift（BDD）信道演化模型、Vector AMP（VAMP）、Expectation‑Maximization（EM）参数学习、伯努利‑高斯混合后验推断。

**📊 数据集**

实验使用合成数据：基站 64 天线，用户 4 天线，32 个 QPSK 导频，时隧 100 步，仿真 SNR 5–20 dB，稀疏率 ρ=0.8（由 BDD 参数设定）。

**📈 对比分析**

与 SBL、EM‑VAMP、Kalman‑filter‑ML、VAMP、BDD‑VAMP、AMP‑SI 等基线进行对比。结果表明 BDD‑VAMP‑EM 在所有 SNR 区间的 TNRMSE 均优于基线，且与理想已知 BDD 参数的 BDD‑VAMP 差距很小，展示了良好的鲁棒性和优越性能。

**⚠️ 局限性**

局限性：①仅在合成、时间不变的 BDD 参数下验证，缺乏真实通道数据的评估；②仍假设初始时刻为 BG prior，且对高维稀疏信道的计算复杂度较高；③后验近似仍以伯努利‑高斯混合为基础，可能在极低 SNR 或高度非线性场景下不足。

---

## 440. Virtual Simulation for Mental Health

**arXiv ID:** 2606.24826 | [PDF](https://arxiv.org/pdf/2606.24826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 441. Decentralised AI Training and Inference with BlockTrain

**arXiv ID:** 2606.24722 | [PDF](https://arxiv.org/pdf/2606.24722v1)

**作者:** Peter Toth `[一作]` `[通讯]` (Spheroid Labs), Peter Toth (Spheroid Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

缺乏论文内容，无法确定具体研究内容

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

## 442. Solving Inverse Problems of Chaotic Systems with Bidirectional Conditional Flow Matching

**arXiv ID:** 2606.24824 | [PDF](https://arxiv.org/pdf/2606.24824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 443. DDStereo: Efficient Dual Decoder Transformers for Stereo 3D Road Anomaly Detection

**arXiv ID:** 2606.24805 | [PDF](https://arxiv.org/pdf/2606.24805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. Dirac-Frenkel dynamics with inertia for nonlinearly parametrized solutions of evolution problems

**arXiv ID:** 2606.24769 | [PDF](https://arxiv.org/pdf/2606.24769v1)

**作者:** Matteo Raviola `[一作]` (EPFL), Benjamin Peherstorfer `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种将惯性引入Dirac–Frenkel动力学的新方法（DFI），并给出了其连续与离散系统的存在性、误差后验分析；在Allen–Cahn和10维Fokker–Planck等问题上进行数值验证。

**💡 创新点**

通过在Dirac–Frenkel条件下加入惯性项，使参数速度在弱信息方向上保留历史信息，避免瞬时正则化导致的速度消失，从而提升对冗余或奇异参数化的鲁棒性。

**🔧 技术方法**

使用Tikhonov正则化、Onsager原理、半隐式Euler离散、锚定的线性最小二乘、随机采样/左投影以及误差后验分析等技术。

**📊 数据集**

在一维周期Allen–Cahn PDE（450个网格点）和10维Fokker–Planck概率密度问题（2000个采样点、Gaussian混合提议）上进行实验。

**📈 对比分析**

与标准Tikhonov-regularized Dirac–Frenkel方法及其sketch版本对比，DFI在大正则化强度、压缩采样和长时间演化下误差更小、能量收敛更快，尤其在弱信息方向表现出更强鲁棒性。

**⚠️ 局限性**

极端大内存参数β趋近1时稳定性下降；在强正则化或极高维稀疏情况下需额外调参；未给出自适应β或ε选择策略，理论分析多在理想假设下实现。

---

## 445. Matching Tasks to Objectives: Fine-Tuning and Prompt-Tuning Strategies for Encoder-Decoder Pre-trained Language Models

**arXiv ID:** 2606.24841 | [PDF](https://arxiv.org/pdf/2606.24841v1)

**作者:** Ahmad Pouramini `[一作]` (University of Tehran), Hesham Faili `[通讯]` (University of Tehran)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种匹配任务与目标（MTO）框架，自动将下游任务分配到合适的预训练目标，并在细调或提示调优前进行无监督适配，显著提升encoder‑decoder预训练模型在知识库补全和问答任务中的性能。

**💡 创新点**

创新点包括：① 通过任务分类器将任务映射到特定预训练目标；② 利用句子分割器和目标分类器生成与目标一致的无监督训练数据；③ 设计与预训练目标匹配的细调与软提示模板；④ 在无监督适配阶段结合多种目标（LM、denoising、混合）进一步提升知识迁移效果。

**🔧 技术方法**

技术细节：使用T5系列encoder‑decoder模型；在无监督适配阶段采用掩码去噪（denoising）、前缀语言模型（LM）或混合目标；设计与目标对应的模板进行细调；实现软提示调优（prompt tuning）并通过MLP优化提示嵌入；构建句子分割器与目标分类器用于自动化数据准备。

**📊 数据集**

使用的数据集包括：ATOMIC^20_20（知识图谱关系）、CommonsenseQA、OpenBookQA（问答任务）以及OMCS语料库（约8000句子）用于无监督适配；5k任务样本用于训练任务分类器和句子分割器。

**📈 对比分析**

在few-shot（30样本）下与基线Comet、Prompt‑tuning、RoBERTa‑large等方法对比，MTO在Mask‑Filling和Map‑Phrasal任务中ROUGE/BERT分数提升超过20%；在CommonsenseQA和OpenBookQA问答任务中准确率提升约10–15%（相当于120%+的性能提升）。在全数据训练下，MTO的表现与或略优于现有基线。

**⚠️ 局限性**

局限性：仅在encoder‑decoder架构下验证，未探讨其他模型；主要聚焦知识补全与问答任务，未评估分类等类型任务；适配阶段仅使用约8000句子，规模可扩展性的影响尚待进一步研究。

---

## 446. OrbitForge: Text-to-3D Scene Generation via Reconstruction-Anchored Video Synthesis

**arXiv ID:** 2606.24799 | [PDF](https://arxiv.org/pdf/2606.24799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 447. Context-Aware Prediction of Student Quiz Performance with Multimodal Textbook Features

**arXiv ID:** 2606.24770 | [PDF](https://arxiv.org/pdf/2606.24770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 448. HelpBench: Assessing the Ability of LLMs to Provide Privacy, Safety, and Security Advice

**arXiv ID:** 2606.24819 | [PDF](https://arxiv.org/pdf/2606.24819v1)

**作者:** Sarah Meiklejohn `[一作]` (University College London), Kurt Thomas `[通讯]` (Google)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了HelpBench基准，收集450个真实的Reddit用户隐私安全问题，为每个问题制定专家编写的多维评价rubrics，并开发自动化评分器，评估了18种最先进LLM在单轮回答中的准确性和语调质量。

**💡 创新点**

首创针对数字隐私、安全和安保(help‑seeking)问题的单轮评估基准，利用专家制定的细粒度rubrics与自动化评分器系统性揭示LLM长尾错误，并对错误类型进行深入剖析。

**🔧 技术方法**

使用自然语言处理的checklist式自动评分模型（auto‑rater），结合Gemini、ChatGPT等LLM生成回答和评分，配合Pearson相关、Wilcoxon、Kruskal‑Wallis等统计方法进行性能对比。

**📊 数据集**

450条重写后的Reddit问题（覆盖9大PSS主题，标注长度、已执行实践与需求类型），每条问题配套专家rubric和人工打分，构成评估数据集。

**📈 对比分析**

对18个模型分别提问5次并平均评分；整体平均得分约82%，最高87%；但约10%回答得分低于65%，显示长尾低质量；按主题、长度、实践与需求类型等维度进行差异性分析，发现主题差异最大。

**⚠️ 局限性**

可能存在训练集泄露导致的性能偏高；rubric随平台功能变更而失效；仅覆盖单轮英文文本，缺乏多模态、多轮、跨语言场景；专家评分与普通用户体验可能不一致。

---

## 449. EG-VQA: Benchmarking Verifiable Video Question Answering with Grounded Temporal Evidence

**arXiv ID:** 2606.24797 | [PDF](https://arxiv.org/pdf/2606.24797v1)

**作者:** Linpeng Huang `[一作]` (Sun Yat Sen University), Liang Lin `[通讯]` (Sun Yat Sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了新的Evidence‑Grounded Video Question Answering Benchmark（EG‑VQA），并提出了EG‑F1评估指标和一款基于强化学习的Evidence‑Grounded Reasoner（EG‑Reasoner）模型。

**💡 创新点**

创新点在于：①将VideoQA转为需要同时给出答案和时序证据的结构化生成任务；②提出联合时间与语义一致性的EG‑F1评估标准；③通过soft‑matching + hard‑matching的证据奖励，解决RL训练中的奖励稀疏问题；④使用多阶段人机评估与两轮质量控制保证数据与评测质量。

**🔧 技术方法**

技术核心包括：大型视频‑语言模型（Qwen2.5‑VL‑7B‑Instruct）作为基础；结构化生成框架；组合奖励函数（格式、答案、证据）；Group Relative Policy Optimization（GRPO）强化学习；Soft‑matching 证据奖励实现稠密梯度。

**📊 数据集**

使用的主要数据集是EG‑VQA，基于ActivityNet Captions、HiREST、YouCook2三源视频共2,067条视频，11,838条开放式QA对，含平均2.27个时序证据段。

**📈 对比分析**

与多种开源/专有Video‑LLM（如GPT‑4o、Gemini‑2.5‑Flash、Qwen2.5‑VL‑7B、InternVL‑3‑8B、Time‑R1等）对比，EG‑Reasoner在严格准确率、宽松准确率以及EG‑F1上分别提升了约12%、10%和30%（相对基线），在因果与反事实等复杂推理任务上表现尤为突出，甚至在某些指标上接近或超过专有模型。

**⚠️ 局限性**

局限性包括：①对时序定位的精度仍有提升空间（随着IoU阈值升高性能下降明显）；②RL训练需要大量采样与调参，收敛速度慢；③模型在某些视频场景中仍会生成冗余或不完整的证据；④当前评测仍主要基于自动化指标，缺少更细粒度的人类解释评估。

---

## 450. Paying to Know: Micro-Transaction Markets for Verified Product Information in Agentic E-Commerce

**arXiv ID:** 2606.24783 | [PDF](https://arxiv.org/pdf/2606.24783v1)

**作者:** Filippos Ventirozos `[一作]` (Manchester Metropolitan University), Matthew Shardlow `[通讯]` (Manchester Metropolitan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出在智能代理驱动的电商中，构建以微交易为基础的经验证信息市场，替代传统的产品排名与促销模式，强调信息可验证性与付费获取的价值；

**💡 创新点**

核心创新在于将信息本身作为可交易资产，利用微支付与代理授权机制，实现信息价格化、谈判化与按需获取，进而促进更公平、更高质量的市场竞争；

**🔧 技术方法**

借助代理原生支付协议（x402/AP2）和多方信息交换API，结合自然语言处理技术实现成本意识工具调用、谈判对话、实体解析、基于证据的生成与个人化偏好建模；

**📊 数据集**

本文为立场论文，未使用具体数据集，而是通过汽车零部件与二手车的案例阐释概念与架构；

**📈 对比分析**

因缺乏实验实现与基准测试，本文未进行方法比较或性能评估，主要提供概念性设计与未来研究方向；

**⚠️ 局限性**

主要限制包括对代理原生支付网络的依赖、卖家对信息公开的接受度、信息市场机制的正式激励兼容性、以及潜在的协同、隐私与公平性风险等问题。

---

## 451. Adaptive Hebbian Memory Routing in Vision Transformers for Few-Shot Learning

**arXiv ID:** 2606.24756 | [PDF](https://arxiv.org/pdf/2606.24756v1)

**作者:** Mohammed Yusuf Mujawar `[一作]` (University of Alabama), Noorbakhsh Amiri Golilarz `[通讯]` (University of Alabama)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在视觉Transformer中引入自适应Hebbian记忆路由，允许模型在少样本任务中根据支持集动态调整记忆贡献、写入强度与衰减。

**💡 创新点**

创新点在于使用轻量级MLP路由器实现任务感知的Hebbian快权记忆控制，分离记忆放置、可塑性与记忆保持三种自适应维度。

**🔧 技术方法**

采用快速权记忆模块、两层MLP路由器以及ViT/DeiT/Swin Transformer骨干进行少样本分类。

**📊 数据集**

在Omniglot、CIFAR‑FS以及从CIFAR‑FS到Omniglot的跨域转移任务上进行实验。

**📈 对比分析**

与基线、固定Hebbian以及不同自适应变体对比，Swin在5‑way 1‑shot上从96.74%提升至96.94%，ViT、DeiT在CIFAR‑FS上提升约15–20个百分点，且推理速度提升约30%。

**⚠️ 局限性**

局限在于自适应控制仍受网络架构影响，跨域迁移效果不均衡，且未探究更大规模任务或更深Transformer的适用性。

---

## 452. BioMedVR: Confusion-Aware Mixture-of-Prompt Experts for Biomedical Visual Reprogramming

**arXiv ID:** 2606.24740 | [PDF](https://arxiv.org/pdf/2606.24740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 453. A Natively Blocked, Device-Resident Algebraic Multigrid GPU Path in PETSc

**arXiv ID:** 2606.24748 | [PDF](https://arxiv.org/pdf/2606.24748v1)

**作者:** Mark F. Adams `[一作]` `[通讯]` (Lawrence Berkeley National Laboratory), Mark F. Adams (Lawrence Berkeley National Laboratory)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 PETSc 中实现了基于矩阵块的 smoothed‑aggregation AMG，彻底消除在 coarsening 过程中对标量格式的扩展，并在 GPU 上使用 Kokkos 支持的矩阵块类型完成完整的 AMG 设定与求解。

**💡 创新点**

首次在 GPU 上实现了能够处理非方块（3×6）块大小的 AMG，提供了原生块 COO 组装、设备驻留的 Galerkin 计算以及块级 off‑process 收集的高效实现，显著提升了内存利用率和带宽利用率。

**🔧 技术方法**

采用了 PETSc 的 Kokkos 统一抽象、Kokkos‑Kernels 低层实现、CUDA/hip 后端，构建了可在不同 GPU 上运行的矩阵块类型；同时利用 GPU‑aware MPI、阻塞广播、块级 COO 组装等技术。

**📊 数据集**

以 3D 线性弹性（bs=3）为模型问题，在 Perlmutter 计算节点上使用 32³、64³、96³、128³ 的三维网格（共 98,304 未知/GPU）进行弱扩展实验；还比较了 Q1 与 Q2 元素（非零数/行分别约 78 与 180）。

**📈 对比分析**

对比 cuSPARSE、Kokkos‑Kernels（标量）与自研块格式，在 A100 GPU 上测得：块 SpMV 在 27 GPU 时可实现 1.42× 的带宽提升，块 Galerkin 计算 8× 以内存和时间优势；整体 V‑cycle 在 8/27/64 GPU 时分别实现 1.04×/1.24×/1.16×、1.12×/1.42×/1.30×、1.45×/1.80×/2.27× 的加速，单 GPU 时保持 1:1 的性能。

**⚠️ 局限性**

仍受限于 cuSPARSE 对块化的支持不足（无法处理非方块块）、仅验证于弹性 PDE、在冷启动阶段仍需一次标量转换、且块 Galerkin 在极高非零/行时性能收敛；未来需进一步实现完整设备驻留的 MIS 细化、优化 off‑process reduction 与多 GPU 负载平衡。

---

## 454. GeoT2V-Bench: Benchmarking 3D Consistency in Text-to-Video Models via 3D Reconstruction

**arXiv ID:** 2606.24829 | [PDF](https://arxiv.org/pdf/2606.24829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 455. L3Cube-MahaPOS: A Marathi Part-of-Speech Tagging Dataset and BERT Models

**arXiv ID:** 2606.24825 | [PDF](https://arxiv.org/pdf/2606.24825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 456. Scaling Laws for Task-Specific LLM Distillation

**arXiv ID:** 2606.24747 | [PDF](https://arxiv.org/pdf/2606.24747v1)

**作者:** Lavinia Ghita `[一作]`, Ioana Boier `[通讯]` (Nvidia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在金融事件分类领域对大型语言模型（LLM）进行压缩，探讨压缩比例、数据规模、监督方式与结构化剪枝对任务和通用知识性能的影响；

**💡 创新点**

提出混合推理链（blended chain‑of‑thought）监督损失，稳定 KL 散度在推理链上的蒸馏；同时给出经验性压缩与数据扩展的尺度律，为部署决策提供可复用框架；

**🔧 技术方法**

使用结构化深度+宽度剪枝（轻量NAS），LoRA 低秩适配器蒸馏，logit‑based KL 蒸馏（带混合 CoT），多步迭代压缩与蒸馏，以及不同退化剪枝调度（指数、多项式、余弦、线性）；

**📊 数据集**

采用自生成的 500,000 条金融新闻标题（FinHeadlineMix），通过 Qwen3‑32B 进行标注，构成 35 类事件分类任务；

**📈 对比分析**

在 50%、37% 等压缩比例下，LoRA 在大数据量时在域内宏 F1 最高（≈0.76），但对通用知识保持不变；混合 CoT 在中小数据量下既能恢复通用知识（MMLU 从 40→54+），又能维持宏 F1≈0.77；单步剪枝性能显著下降；线性退化调度在 16% 目标下得到最优宏 F1；

**⚠️ 局限性**

局限：仅在单一密集型教师（Qwen3‑32B）和短文本分类任务上验证；未涉及量化、跨架构蒸馏、生成式任务或多教师场景；优化器仅使用 AdamW，可能影响稳定性；

---

## 457. MANGO: Automated Multi-Agent Test Oracle Generation for Vision-Language-Action Models

**arXiv ID:** 2606.24815 | [PDF](https://arxiv.org/pdf/2606.24815v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 458. World Models in Pieces: Structural Certification for General Agents

**arXiv ID:** 2606.24842 | [PDF](https://arxiv.org/pdf/2606.24842v1)

**作者:** Yikai Lu `[一作]` (Chinese University of Hong Kong), Tongxin Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出结构化认证框架，通过对特定转移的目标条件性能进行筛选，推断智能体内部世界模型在这些转移上的精确度。

**💡 创新点**

创新点在于将全局统一保证替换为转移局部保证，给出误差上界为O(1/n)+O(δ)的构造性证明，并实现滤波算法实现此认证。

**🔧 技术方法**

使用线性时序逻辑（LTL）目标、马尔可夫决策过程建模、构造性滤波算法以及概率错误分析等技术。

**📊 数据集**

在随机生成的20状态5动作网格世界以及625状态迷宫环境上进行实验，采用随机行走训练的通用代理。

**📈 对比分析**

与传统基于O(1/√n)误差上界的基线相比，实验显示该方法在不同δ与n下的估计误差更低，逼近理论上界，证明了更高的规划可靠性。

**⚠️ 局限性**

局限性包括需先验获取转移概率以构造滤波器、仅针对确定性策略、仅适用于大规模“big world”假设，尚未推广到随机或最大熵等策略。

---

## 459. SHERLOC: Structured Diagnostic Localization for Code Repair Agents

**arXiv ID:** 2606.24820 | [PDF](https://arxiv.org/pdf/2606.24820v1)

**作者:** Hovhannes Tamoyan `[一作]` (NVIDIA), Boris Ginsburg `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练的LLM工具框架 SHERLOC，实现多轮工具交互定位代码缺陷，并生成结构化诊断报告。

**💡 创新点**

创新点在于不需要微调或多代理协作，使用固定工具集和自恢复机制，生成结构化诊断而非仅文件检索，并通过质量过滤提升下游修复效果。

**🔧 技术方法**

技术包括单一推理LLM、四种仓库工具（文件查看、代码搜索、树浏览、导入图），有限交互循环，异常恢复与最终合成。

**📊 数据集**

使用 SWE‑Bench Lite 与 SWE‑Bench Verified 两个基于 GitHub issue+PR 的公开数据集。

**📈 对比分析**

与 SWE‑Debate、SWE‑Rank、OrcaLoca 等对比，SHERLOC 在 Lite 上 84.33%@1、Verified 上 81.27%@1 取得 SOTA；注入修复代理后平均提升 5.95pp 解决率，token 成本降低 36.7%/23.1%。

**⚠️ 局限性**

局限包括对公开仓库预训练知识的依赖、仅在 Python 仓库验证、质量判定需外部 GPT‑5.2 评估、模型与提示对交互敏感。

---

## 460. Vision-Language Model Reasoning for Contextual Semantic Mapping in Intralogistics

**arXiv ID:** 2606.24814 | [PDF](https://arxiv.org/pdf/2606.24814v1)

**作者:** Marvin Rüdt `[一作]` (Karlsruhe Institute of Technology), Kai Furmans `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种融合SLAM、SAM、实例聚类与VLM推理的多视角上下文语义地图生成管线

**💡 创新点**

创新点在于使用零样本、开放词汇VLM进行物体可移动性推理以及多视角聚合来实现上下文语义化

**🔧 技术方法**

采用GMapping SLAM、Segment Anything Model（SAM）分割、基于IoU的实例聚类、Gemini 3.1 Flash Lite等VLM以及结构化提示

**📊 数据集**

在18个实例、13类、74帧的工业物流场景数据集上实验，数据已公开

**📈 对比分析**

与三种VLM和提示策略比较，最佳配置Gemini 3.1 Flash Lite+直接JSON提示下实现98.93% mIoU、89.17% movability mAcc、89.78% PQ，组件分析表明多视角推理为关键瓶颈

**⚠️ 局限性**

局限在于仅离线单一场景实验、VLM推理仍为主瓶颈、实例聚类未显著提升、缺乏3D及实时更新能力

---

## 461. Counting Trees from Satellite Imagery with Noisy Supervision

**arXiv ID:** 2606.24786 | [PDF](https://arxiv.org/pdf/2606.24786v1)

**作者:** Dimitri Gominski `[一作]` (University of Copenhagen), Loic Landrieu `[通讯]` (Univ Gustave Eiffel)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了 TreeMatch 模型，利用无偏平衡最优传输框架对卫星影像中的树木进行计数，并通过自校正机制提升噪声监督的利用效率。

**💡 创新点**

创新点在于将无偏平衡最优传输用于树木计数，既能精确定位稀疏树木，又能在密集森林中稳健估计密度，同时利用传输残差动态纠正弱标签，提升大规模噪声监督的效果。

**🔧 技术方法**

核心技术包括无偏平衡最优传输（UOT）与其 Sinkhorn 迭代求解、基于 UOT 的分布匹配损失、计数一致性损失以及基于残差的自校正权重机制。

**📊 数据集**

实验使用了新建的 TinyTrees 数据集，涵盖三大洲（三块卫星传感器）共计 23 000 km²、215 M 树点注释（其中 773 K 为人工验证），并与传统检测、回归、分布匹配等基线进行对比。

**📈 对比分析**

在 TinyTrees 的三组数据集上，TreeMatch 在 RMSE、nMAE 和 R² 等指标上均优于所有基线，特别是在高密度森林和多传感器环境下表现出显著的鲁棒性和更低的误差。

**⚠️ 局限性**

主要限制在于无偏平衡最优传输的 O(N²) 计算和显存开销，难以直接扩展到更大尺寸图像；此外，目前未实现高度、密度等多模态联合建模，未来可进一步提升森林结构估计的整体一致性。

---

## 462. AerialFusionMapNet: Online HD Map Construction with Aerial-Onboard BEV Fusion

**arXiv ID:** 2606.24784 | [PDF](https://arxiv.org/pdf/2606.24784v1)

**作者:** Daniel Lengerer `[一作]` (Technical University of Applied Sciences Augsburg), Carsten Markgraf `[通讯]` (Technical University of Applied Sciences Augsburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 AerialFusionMapNet，一种将机载摄像头与高分辨率航拍图像融合的在线高清地图构建框架。

**💡 创新点**

创新点在于两阶段结构化训练：先用场景一致旋转（SCR）预训练航拍编码器，再冻结该编码器并通过跨视图监督（cvs）将机载 BEV 特征对齐，以显著提升航拍信息的利用。

**🔧 技术方法**

主要技术包括 BEV 特征提取、卷积编码器（UNet++、ResUNet、DeepLabv3+ 等）、跨视图监督（cvs）、场景一致旋转数据增强、向量化地图解码等。

**📊 数据集**

使用 nuScenes 地理分割版本与 AID4AD 提供的精确对齐航拍影像作为训练与评估数据集。

**📈 对比分析**

在 nuScenes 地理分割上与 StreamMapNet、MapTracker、NavMapFusion、AID4AD StreamMapNet 等基线对比，AerialFusionMapNet 在 60×30 m ROI 上取得 54.7 的 map 分数，较基线提升 5.9 绝对值（12.1% 相关），在原始 nuScenes 分割上亦分别达 84.4/87.5 的 map。

**⚠️ 局限性**

局限性包括：对航拍图像的对齐误差仍有影响；在地理重叠区域性能显著提升，说明对空间泛化的提升有限；高容量编码器不一定带来性能提升，模型压缩空间仍需探索；对大尺度偏移的鲁棒性仍有限。

---

## 463. Compact Object-Level Representations with Open-Vocabulary Understanding for Indoor Visual Relocalization

**arXiv ID:** 2606.24767 | [PDF](https://arxiv.org/pdf/2606.24767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 464. Burnyard: Future of Malware Analysis

**arXiv ID:** 2606.24778 | [PDF](https://arxiv.org/pdf/2606.24778v1)

**作者:** Rama Ramana Sharma Parnandi `[一作]` (Ohio State University), Carter Yagemann `[通讯]` (Ohio State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套轻量级二进制模拟平台Burnyard，用用户空间指令级模拟捕获运行时行为，并将事件记录为结构化CSV；随后通过机器学习模型对恶意软件进行家庭级分类，并提供自然语言行为解释；该系统支持Windows、Linux及Mach‑O二进制，完全离线运行，无需虚拟机；

**💡 创新点**

创新点在于：① 采用用户空间指令级模拟而非完整虚拟机沙箱，显著降低资源消耗与分析时间；② 通过系统调用与Windows API钩子生成可读事件序列，兼容多平台；③ 将事件序列作为特征，结合Transformer模型实现高效分类与行为解读；④ 完全本地化流程，保障样本隐私与离线环境兼容；

**🔧 技术方法**

技术包括：用户空间二进制模拟器、系统调用与API钩子框架、事件日志生成器、Transformer‑based 语言模型、机器学习分类流水线（可能包含随机森林或其他分类器），以及基于Web的管理界面；

**📊 数据集**

评估使用了100个Windows PE和100个Linux ELF样本（共200样本），并在44类（43恶意家族+1正常）上进行分类；

**📈 对比分析**

与VirusTotal（多引擎静态+动态）和Sophos Intelix（单独沙箱）比较：Windows样本平均分析时间22.41 s（比VT快1.44×、比Intelix快8.16×），Linux样本平均5.47 s（比VT快2.97×、比Intelix快14.78×）。Burnyard在速度上大幅领先，同时保持完整行为捕获；

**⚠️ 局限性**

局限性包括：① 仅模拟用户空间指令，无法捕获内核级或硬件相关行为；② 对于少量样本的家族，分类准确性仍有限；③ 需要手工构建根文件系统，可能影响跨平台支持；④ 目前未对复杂的沙箱规避技术进行专门测试；

---

## 465. Revealing Training Data Exposure in Vision Language Large Models via Parameter Gradients

**arXiv ID:** 2606.24774 | [PDF](https://arxiv.org/pdf/2606.24774v1)

**作者:** Zhihao Zhu `[一作]` (Hong Kong University of Science and Technology), Ahmed Abbasi `[通讯]` (University of Notre Dame)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GradAudit，一种基于梯度的视觉语言大模型（VLLM）训练数据检测框架。

**💡 创新点**

创新点在于利用训练样本梯度稳定性与非训练样本梯度噪声的差异，并对模型参数梯度进行功能切片与噪声特征掩蔽，从而实现跨模态成员检测。

**🔧 技术方法**

使用了梯度切片、噪声特征掩蔽、余弦相似度决策以及白盒梯度分析等技术，对 VLLM 的预训练与微调阶段进行评估。

**📊 数据集**

采用了医疗数据集（PMC‑OA、ROCO、MedTrinity）、通用数据集（COCO、FashionGen）、以及 Studio Ghibli 版权数据集作为评测和基准。

**📈 对比分析**

与多种基线方法（Loss、Entropy、Zlib、Min‑K、ModRényi、M^4I、GradNorm 等）进行对比，GradAudit 在七种实验配置下的 AUROC 最高达 92.7%，比最佳基线提升 20–30% 以上。

**⚠️ 局限性**

主要限制是需要白盒模型参数访问和参考样本，且梯度计算和相似度比较成本较高，缺乏零样本/少样本的无监督变体。

---

## 466. Posterior Refinement: Fast Language Generation via Any-Order Flow Maps

**arXiv ID:** 2606.24773 | [PDF](https://arxiv.org/pdf/2606.24773v1)

**作者:** Manan Agarwal `[一作]` (Carnegie Mellon University), Nicholas M. Boffi `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将Flow Map Language Model与Masked Diffusion模型结合的通用框架（FMLM+），并基于此开发一种利用后验置信度进行自适应迭代改进的Posterior Refinement方法。

**💡 创新点**

创新点在于：①用掩码式噪声调度使FMLM具备任意顺序推理灵活性；②利用一次性全序列生成得到的后验置信度来判断错误，从而实现自纠正的迭代改进；③通过MDM的知识蒸馏与初始化显著加速训练。

**🔧 技术方法**

采用DiT网络作为基础，使用两时刻去噪器（Two‑Time Denoiser）自蒸馏训练，结合MDM的离散扩散目标进行知识蒸馏与初始化，构建后验置信度阈值阈值的改进循环。

**📊 数据集**

在TinyStories、OpenWebText（无监督文本生成）、TinyGSM、GSM8K（零样本推理）以及Sudoku（结构约束推理）四个基准上进行评估。

**📈 对比分析**

与传统MDM、FMLM及其他连续流模型对比，FMLM+ + Posterior Refinement在速度‑质量平衡上显著优越，往往在仅使用4–32倍更少的函数评估（NFE）即可匹配或超过最强基线，尤其在Sudoku、GSM8K和无监督文本生成任务中表现突出。

**⚠️ 局限性**

局限性：实验规模有限，仅在小型合成/小规模数据集上验证；未探索替代的训练目标或更大参数规模；后验置信度阈值设置较为经验化，可能在不同任务中需要调整。

---

## 467. Can Scale Save Us From Plasticity Loss in Large Language Models?

**arXiv ID:** 2606.24752 | [PDF](https://arxiv.org/pdf/2606.24752v1)

**作者:** J. Fernando Hernandez-Garcia `[一作]` (Zyphra), Beren Millidge `[通讯]` (Zyphra)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在多语言持续学习和循环学习框架下，训练了从5M到314M参数的GPT‑style Transformer，评估其在新语言（越南语）上的适应性，以此检测和量化塑性损失。

**💡 创新点**

创新点在于：①首次在大规模自然语言数据上系统地证明塑性损失的普遍存在；②提出了基于模型规模的子线性幂律预测塑性损失出现的任务数；③同时在持续学习和非持续学习场景下对塑性损失进行对比，揭示其普适性。

**🔧 技术方法**

使用了预训练的Transformer（预归一化）架构，采用AdamW优化、线性预热、固定层数与隐藏维度比例、固定注意力头维度，探究参数幅度、沉默单元、稀疏/膨胀注意力头等网络层面指标。

**📊 数据集**

数据集为CulturaX，包含167种语言共6.3T标记；在实验中选取8种语言（英、中、法、日、西、德、葡、俄）进行循环预训练，并使用20B越南语标记做独立探测；持续学习实验使用所有8种语言的混合。

**📈 对比分析**

通过在每个训练周期结束后对复制模型进行5B越南语探测，并用验证损失曲线的面积（AUC）衡量适应效率；结果显示，所有模型最终均出现AUC上升（塑性降低），但规模越大出现时间越晚；持续学习与非持续学习下的塑性下降趋势相似。

**⚠️ 局限性**

局限性包括：仅探测单一新语言；仅评估AUC而非更细粒度学习曲线；使用的规模（最多314M）仍低于工业级LLM，难以直接推断更大模型的行为；并未给出实证有效的修复方法，仅提供潜在诊断指标。

---

## 468. VSANet: View-aware Sparse Attention Network for Light Field Image Denoising

**arXiv ID:** 2606.24737 | [PDF](https://arxiv.org/pdf/2606.24737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 469. Beyond U-Net: A Latent-Representation-Aligned Skip-Free Backbone for Flow-Matching Speech Enhancement

**arXiv ID:** 2606.24745 | [PDF](https://arxiv.org/pdf/2606.24745v1)

**作者:** Wangyi Pu `[一作]` (Sapienza University of Rome), Michele Scarpiniti `[通讯]` (Sapienza University of Rome)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了一种基于流匹配的语音增强模型，采用无跳过连接的编码‑解码骨干，并通过对齐编码器与解码器的瓶颈与清晰语音的Codec潜在表示实现噪声抑制。

**💡 创新点**

创新点在于将传统U‑Net跳过连接替换为无跳过结构，并利用冻结的Descript Audio Codec（DAC）作为潜在表示对齐目标，减少噪声传播并在仅五步函数评估内保持高效推理。

**🔧 技术方法**

采用了流匹配（Flow Matching）与ODE积分、潜在表示对齐（Latent Representation Alignment）、FiLM时间嵌入、对抗与特征匹配损失，以及DAC无量化的时间域编码器‑解码器。

**📊 数据集**

实验使用WSJ0‑CHiME3（SNR 0‑20 dB）和VoiceBank‑DEMAND（SNR 0‑17.5 dB）这两个公开数据集。

**📈 对比分析**

与传统U‑Net+GAN或无GAN版本的FlowSE比较，使用LRA骨干后在WSJ0‑CHiME3的PESQ略微提升，在VoiceBank‑DEMAND的PESQ提升0.23点，并在DNSMOS、WVMOS、SIG、BAK、OVRL等感知指标上均优于对照组。

**⚠️ 局限性**

局限性包括：对Codec潜在表示质量高度依赖；在某些指标（如SI‑SDR）仍略逊于U‑Net基线；未验证跨语言或实时部署的鲁棒性；以及仅在时间域实现，缺乏频域或复数域的扩展。

---

## 470. SER: Learning to Ground Video Reasoning with Semantic Evidence Rewards

**arXiv ID:** 2606.24726 | [PDF](https://arxiv.org/pdf/2606.24726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 471. IV-CoT: Implicit Visual Chain-of-Thought for Structure-Aware Text-to-Image Generation

**arXiv ID:** 2606.24849 | [PDF](https://arxiv.org/pdf/2606.24849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 472. Spherical-to-ERP Epipolar Rectification for Single-Axis Disparity in 360 Stereo

**arXiv ID:** 2606.24847 | [PDF](https://arxiv.org/pdf/2606.24847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 473. BenchX: Benchmarking AI Models for Cancer Detection and Localization with Demographic and Protocol Biases

**arXiv ID:** 2606.24883 | [PDF](https://arxiv.org/pdf/2606.24883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 474. Real vs. Complex Spectral Bases for Neural Operators: The Role of Green's Function Alignment

**arXiv ID:** 2606.24851 | [PDF](https://arxiv.org/pdf/2606.24851v1)

**作者:** Jason Sulskis `[一作]` (University of Illinois at Chicago), Sathya Ravi `[通讯]` (University of Illinois at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

提出了纯实数的 Hartley Neural Operator（HNO），并与传统的 Fourier Neural Operator（FNO）进行系统对比，探究实数与复数频域在求解不同类型 PDE 时的性能差异。

**💡 创新点**

创新点在于：①将 HNO 定义为 FNO 的实数镜像，完全消除复数算子中的共轭对称冗余；②通过自伴随理论和相位内容分析，给出了“匹配算子对称性与频域基底” 的预测准则，说明椭圆算子偏好 Hartley 基底，而时变/相位占主导的算子偏好 Fourier 基底。

**🔧 技术方法**

使用离散 Hartley 变换（DHT）、FFT、卷积神经网络架构，保持相同网络宽度和超参数，在训练中采用相同优化器、学习率、正则化等，保证仅基底差异影响结果；同时结合 Green’s 函数对称性和算子相位分析解释实验结果。

**📊 数据集**

使用三类初始条件（Gaussian 随机场、拉普拉斯特征展开、Gaussian bump）在 64×64 网格上生成 200 条样本，覆盖周期与 Dirichlet 边界条件，评估六类 PDE（热、波、扩散、Advection‑Diffusion、Burgers、Navier‑Stokes、Poisson、双调和）及其不同参数组合。

**📈 对比分析**

通过在相同宽度、相同超参数下训练 HNO 与 FNO，使用相对 L² 误差和梯度误差作为评价指标，结果显示：椭圆算子（Poisson、双调和）中 HNO 误差更低；时变算子（热、波、Advection‑Diffusion、Burgers、Navier‑Stokes）中 FNO 误差更低，且优势随算子相位内容递增；初始条件类型影响误差幅度但不改变优劣方向。

**⚠️ 局限性**

局限性包括：仅在 64×64 的正方形域、周期与 Dirichlet 边界下测试；未使用原生 Fast Hartley Transform（FHT）实现，导致 DHT 的额外计算开销；未处理强非线性/冲击场景（如 Burgers 的冲击形成）；实验范围受限于可用 GPU 资源，未涵盖更高维度或更复杂几何问题。

---

## 475. It's Complicated: On the Design and Evaluation of AI-Powered AAC Interfaces

**arXiv ID:** 2606.24854 | [PDF](https://arxiv.org/pdf/2606.24854v1)

**作者:** Blade Frisch `[一作]` (Michigan Technological University), Keith Vertanen `[通讯]` (Michigan Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

讨论并提出了评估 AI 驱动 AAC 接口的多维方法，并分析了六大设计考虑点

**💡 创新点**

强调多元评价体系以捕捉交叉身份需求，突破传统单一技术指标的限制

**🔧 技术方法**

利用大语言模型（LLM）预测、情感识别、语音合成与情境感知等 AI 技术

**📊 数据集**

未使用具体公开数据集，建议结合 AAC 现有日志和用户自评数据进行评估

**📈 对比分析**

未给出实验比较，主要提出定量指标（如 WPM、WER）与定性方法（访谈、日记）混合评估

**⚠️ 局限性**

缺乏实证验证、评估方法仍处于提案阶段，可能导致技术优先而忽视用户需求

---

## 476. DiffusionBench: On Holistic Evaluation of Diffusion Transformers

**arXiv ID:** 2606.24888 | [PDF](https://arxiv.org/pdf/2606.24888v1)

**作者:** Xingjian Leng `[一作]` (Australian National University), Liang Zheng `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个统一的Diffusion Transformer（DiT）训练与评估框架，支持在ImageNet与文本到图像（T2I）两种任务下使用相同代码、模型结构与优化器，仅需少量配置即可切换任务；在此框架下对21种潜在空间DiT模型进行了系统训练与评估，发现ImageNet FID与T2I指标之间相关性弱（Pearson系数-0.377至-0.580）。

**💡 创新点**

核心创新在于：①提出“一键切换”式框架，显著降低在T2I任务上评测的工程成本；②通过大规模实验证明了ImageNet与T2I任务对DiT模型的评估结果不具高度一致性；③构建了融合ImageNet与T2I指标的全景性基准（Diffusion-Bench），并建议未来工作改为同时报告两项结果。

**🔧 技术方法**

采用Diffusion Transformer基础架构（DDT分离编码器-解码器）、无AdaLN编码器、上下文式条件注入、v-prediction目标、AdamW优化器、梯度裁剪、EMA、Euler采样等；框架支持RAE、VAE、像素空间与MeanFlow等多种潜在空间方法。

**📊 数据集**

主要使用ImageNet（class‑conditional 256×256）和多种文本标注图像数据集（JourneyDB、BLIP‑3o 的 Long‑Caption/Short‑Caption）训练T2I模型，文本编码器选用Qwen3‑0.6B。

**📈 对比分析**

在统一配置下，21种潜在空间模型在ImageNet上的F1‑FID从1.37到~3.0不等；在T2I上使用GenEval、DPGBench、GenAIBench三套指标，发现ImageNet排名并不能预测T2I性能，相关系数在-0.38至-0.58之间。整体来看，RAE/REPA-E类模型在两任务上均表现最佳，但ImageNet优秀的模型在T2I上可能排名靠后。

**⚠️ 局限性**

实验受限于计算规模（最多10小时训练，100K步、1024 batch），未覆盖更长训练与更大模型；使用的T2I指标可能被微调数据集“hack”，需更稳健的评测；目前基准尚未广泛更新，需持续维护。

---

## 477. Stability Checking of Markov Jump Linear Systems via Probabilistic Temporal Logic (Extended Version)

**arXiv ID:** 2606.24880 | [PDF](https://arxiv.org/pdf/2606.24880v1)

**作者:** Lena Becker `[一作]` (Saarland University), Holger Hermanns `[通讯]` (Saarland University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了通过概率时序逻辑（PCTL）对马尔可夫跳跃线性系统（MJLS）进行稳定性检查，提出了一种新的方法来分析特定初始状态集的稳定性。

**💡 创新点**

创新点在于将PCTL扩展到MJLS，以便能够指定与特定初始状态集相关的时态性质，并引入了基于时刻的稳定性属性的逻辑扩展。

**🔧 技术方法**

使用了线性代数技术来处理PCTL的模型检查问题，并通过引入新的逻辑运算符来捕捉基于时刻的稳定性属性。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了MJLS在飞行控制、电力系统分析和机器人系统等多个应用领域的建模。

**📈 对比分析**

与传统的全局稳定性分析方法相比，本文的方法能够更细致地分析特定初始状态集的稳定性，尽管未能为整个基础逻辑提供决策程序，但在处理逻辑扩展时表现出良好的性能。

**⚠️ 局限性**

限制在于对于整个基础逻辑的模型检查问题仍然是开放的，且在处理无穷状态空间时存在挑战。

---

## 478. InSight: Self-Guided Skill Acquisition via Steerable VLAs

**arXiv ID:** 2606.24884 | [PDF](https://arxiv.org/pdf/2606.24884v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 479. FLAT: Feedforward Latent Triangle Splatting for Geometrically Accurate Scene Generation

**arXiv ID:** 2606.24876 | [PDF](https://arxiv.org/pdf/2606.24876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 480. FLUX3D: High-Fidelity 3D Gaussian Generation with Diffusion-Aligned Sparse Representation

**arXiv ID:** 2606.24874 | [PDF](https://arxiv.org/pdf/2606.24874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 481. Bridging the Manifold Gap: Riemannian Residual Line Search for One-Step Image Editing

**arXiv ID:** 2606.24844 | [PDF](https://arxiv.org/pdf/2606.24844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 482. OpenThoughts-Agent: Data Recipes for Agentic Models

**arXiv ID:** 2606.24855 | [PDF](https://arxiv.org/pdf/2606.24855v1)

**作者:** Negin Raoof `[一作]` (University of California Berkeley), Ludwig Schmidt `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了全开放的 OpenThoughts‑Agent‑v2 训练数据集，并在此数据集上对 Qwen3‑32B 进行监督微调与强化学习，得到目前最强的 ≤32B 开放数据 agentic 模型。

**💡 创新点**

创新点在于：① 通过 100+ 阶段的 ablation 系统评估任务来源、教师模型、轨迹过滤等对 agentic 性能的关键影响；② 采用合成任务增广突破任务多样性瓶颈；③ 将监督微调与 RL 结合，首次在 8B 规模下证明两者互补。

**🔧 技术方法**

使用技术包括 LLM 生成轨迹（GLM‑4.7‑AWQ 作为教师）、任务混合与过滤、≥5 轮轨迹过滤、合成任务增广、全参数 SFT（cosine 调度）、RL（RLOO、binary 奖励）以及 Llama‑Factory、SkyRL、Harbor 等开源框架。

**📊 数据集**

数据集为 OpenThoughts‑Agent‑v2 共 100k 任务‑轨迹对，来源为 4 类任务集（SWE‑Smith、StackExchange‑SuperUser、StackExchange‑Tezos、Issue‑Tasks），其中 Tezos 通过合成增广扩充至 21k。

**📈 对比分析**

在 7 个 agentic benchmark（SWE‑Bench Verified‑100、Terminal‑Bench 2.0、OpenThoughts‑TBLite 等）上与 Nemotron‑Terminal‑32B、SERA 等开源模型对比，Qwen3‑32B 取得 44.8% 的平均准确率，比 Nemotron‑Terminal‑32B 提升 3.9pp，单个 benchmark 如 Terminal‑Bench 2.0 为 26.2% 等。

**⚠️ 局限性**

局限性包括：RL 仅在 8B 规模实验，未验证 32B；未对基模型差异进行 ablation；100k 轨迹规模尚未检验到多百万轨迹的可扩展性；以及对数据生成流程的再现性与规模可复制性尚未完整验证。

---

## 483. Complexity of Clique-Guarded First-Order Logic with Counting

**arXiv ID:** 2606.24848 | [PDF](https://arxiv.org/pdf/2606.24848v1)

**作者:** Steffen van Bergerem `[一作]` (Humboldt-Universität zu Berlin), Nicole Schweikardt `[通讯]` (Humboldt-Universität zu Berlin)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了“团-守卫一阶逻辑带计数”（CGFOC），并证明其在无稠密类上具有可计算的VC维度与图维度，同时给出针对该逻辑的查询、枚举与PAC学习的算法元定理。

**💡 创新点**

创新点在于引入团-守卫约束提升表达力，利用矩阵秩与类型矩阵的本地化技术获得可计算的维度上界，并首次将这些理论成果转化为可实现的多分类PAC学习算法。

**🔧 技术方法**

主要技术包括：Gaifman 局部性、类型矩阵秩上界、Feferman‑Vaught 分解、阶梯索引（ladder index）与 VC/图维度理论、以及在局部有界扩张类上实现的常数延迟枚举与查询。

**📊 数据集**

本工作纯理论化，未使用具体数据集，而是对任意无稠密或局部有界扩张的结构族进行分析与证明。

**📈 对比分析**

相比以往仅针对FO或FO+计数在稀疏图上的元定理，CGFOC 在保持可处理性的同时大幅提升表达能力；算法在无稠密类上实现几乎线性预处理与常数查询/枚举延迟，PAC学习算法在样本复杂度与运行时间上均给出可计算上界。

**⚠️ 局限性**

局限性：对密集图或仅弱化守卫的扩展会导致模型检查问题变为NP/更高难度；目前对CGFOC 在密集结构上的复杂度尚未完全明晰，且对多分类PAC枚举在局部有界扩张类的实现仍是开放问题。

---

## 484. "Zooming In" on Agentic Web Browsers as Assistive Technologies: A Case Study with a Low-Vision Technology Expert

**arXiv ID:** 2606.24870 | [PDF](https://arxiv.org/pdf/2606.24870v1)

**作者:** Laura Colazzo `[一作]` (Politecnico di Milano), Giuseppe Anzillotti `[通讯]` (Politecnico di Milano)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对低视力专家使用基于大型语言模型的Agentic Web Browser（Perplexity Comet）进行单例案例研究，探讨其作为网页辅助技术的可行性和体验。

**💡 创新点**

首次将Agentic Web Browsers定位为视障辅助工具，并从低视力用户角度系统性评估其对对话流畅性、可控性和透明度的影响，揭示此类系统在辅助技术领域的潜力与挑战。

**🔧 技术方法**

核心技术包括：大型语言模型驱动的网页代理、语音用户界面（VUI）、自然语言交互与网页DOM/截图感知实现的自动化浏览与操作。

**📊 数据集**

未使用公开数据集；研究基于单个低视力参与者的交互数据与访谈记录。

**📈 对比分析**

采用定性方法——观察与半结构化访谈，未进行量化比较或与传统屏幕阅读器的对照实验；因此无法给出具体性能指标。

**⚠️ 局限性**

局限性：仅一名受试者，缺乏可推广性；缺乏非视觉反馈、控制与透明度机制；未涵盖完全失明用户；未与现有辅助技术做对比。

---

## 485. Building a Low-cost Network Digital Twin for the IoT-Edge-Cloud Continuum Using Open-Source Tooling

**arXiv ID:** 2606.24853 | [PDF](https://arxiv.org/pdf/2606.24853v1)

**作者:** Josevany do Amaral `[一作]` (fortiss GmbH), Rute C. Sofia `[通讯]` (fortiss GmbH)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建并验证了一套基于Containerlab、Open vSwitch、ONOS以及Prometheus/Grafana的低成本、全开源网络数字孪生，用于 IoT‑Edge‑Cloud 环境的安全实验与调优。

**💡 创新点**

首次将容器化拓扑模拟、SDN 控制和实时可观测性统一于一体化框架，并通过四个校准的 Wi‑Fi 失真模型实现了真实网络特性的精确重现，同时发现并修复了 TBF 星阔化导致的 MSS 限制伪影。

**🔧 技术方法**

核心技术包括 Containerlab 进行容器化拓扑编排、Open vSwitch 作为虚拟交换机、ONOS SDN 控制器实现意图驱动路由、Prometheus + Grafana 采集和可视化实时指标，以及 netem/TBF 进行延迟、抖动、丢包与带宽控制。

**📊 数据集**

使用了 CODECO IIoT 边缘测试平台收集的 9 天物理测量档案（RTT、UDP/TCP 吞吐、丢包、抖动等 15,988 条原始数据）以及 2 天 NDT 复制实验数据（1,113 条有效测量）进行对比。

**📈 对比分析**

通过点对点 10 次实验和全量分布式统计（均值、标准差、分位数）进行比较，RTT 中位数差异仅 0.4 ms，UDP 吞吐差异 0.03 Mbps，表现出高保真；但 TCP 吞吐受 MSS 限制影响下降约 40%，并在丢包模型与真实 802.11 失真差异中出现显著偏差。

**⚠️ 局限性**

局限性包括 NDT 数据仅覆盖 2 天且与物理测量时间不重叠，无法捕获长期波动；仅针对单一 Raspberry Pi Wi‑Fi 结构；TBF 调度导致的 MSS 限制和概率丢包模型未能完全重现 802.11 的零丢包聚集和尖峰延迟；未来工作需改进调度策略、拓展多域联邦实验和引入 AI 驱动的实时校准。

---

## 486. A Near-Optimal Parallel Algorithm for Finding Matroid Bases

**arXiv ID:** 2606.24845 | [PDF](https://arxiv.org/pdf/2606.24845v1)

**作者:** Sanjeev Khanna `[一作]` (New York University), Junkai Song `[通讯]` (New York University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种在仅使用独立性oracle的情况下，求解任意基（matroid basis）问题的并行算法，算法在 O(n^{1/3} log^{1/3} n) 轮内完成。

**💡 创新点**

核心创新在于：① 针对随机采样导致的基环（circuit）稀疏性，构造了“密度保持下采样”证明；② 设计了一种单轮删除冗余元素的子程序，能够在 α(S)/|S| ≤ |S|/(100 log n) 的条件下一次性移除 Ω(|S|/log n) 个冗余元素；③ 将上述技术与改进的 matroid 分解算法结合，实现在几乎匹配下界的轮数复杂度。

**🔧 技术方法**

主要技术手段包括：随机排列采样得到的基环分析、α‑值（median circuit size）估计、基环包含概率 q_S 与边际概率 p_x 的统计；使用上升算子（up‑operator）与 KL 散度来证明稀疏性在下采样下保持；以及基于混合时间的 matroid 基交换马尔可夫链思路进行参数估计。

**📊 数据集**

该工作不依赖具体数据集，所有实验与证明均在理论模型（independence oracle）下进行；因此不存在真实数据集。

**📈 对比分析**

与之前的 O(n^{3/7}) 轮、O(√n) 轮等结果相比，本算法的轮数降至 O(n^{1/3} log^{1/3} n)，几乎达到已知的 Ω(n^{1/3}/log^{1/3} n) 下界，显著提升了并行效率；在实现上每轮仍保持多项式查询量，整体工作量保持多项式。

**⚠️ 局限性**

局限性包括：① 算法高度依赖随机性，尚未得到可行的确定性版本；② 虽然工作量为多项式，但仍未达到最优的 O(n) 工作量；③ 对于特定 matroid 类（如线性 matroid）可以进一步改进，但通用情形仍存在工作复杂度与轮数的权衡空间。

---

