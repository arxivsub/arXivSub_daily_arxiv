# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-14 | 今日论文总数: 523

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. AIM: A Privacy-Aware Interoperable Memory Framework for Multi-Agent Multi-User LLM Systems

**arXiv ID:** 2609.12320 | [PDF](https://arxiv.org/pdf/2609.12320v1)

**作者:** Zachary Johnson `[一作]` (Microsoft), Sulaiman Vesal `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AIM，面向多代理、多用户的隐私感知记忆系统，支持创建、读取、更新、删除等多标签操作；

**💡 创新点**

创新点在于统一的多标签操作检测、自动可见性分类、索引层级访问控制以及首次公开的多用户记忆基准 MUMBench；

**🔧 技术方法**

使用 LLM 进行提示式推理（操作检测、结构化提取、可见性分类、去重）、HNSW 向量索引与标签增强嵌入、LLM 重排序以及内联评估与指标计算；

**📊 数据集**

使用自制 MUMBench 数据集（672 条交互，93 名用户，4 个领域）和 LOCOMO 进行对比；

**📈 对比分析**

在 MUMBench 上以 5 种模型评估，平均可见性准确率 ≈90%，操作精确度 58–70%，内容质量 ≈90%，GPT‑5.4‑mini 在大多数指标上表现最佳；

**⚠️ 局限性**

局限包括检索相关性仅约45%，标签提取覆盖率低（≈35%），公共记忆可能包含错误信息，评估仍为单轮且对跨代理同步与多用户隐私保护的压力有限。

---

## 2. Battery-Aware Predictive Trajectory Planning and Control for Multirotors Under Disturbances

**arXiv ID:** 2609.12188 | [PDF](https://arxiv.org/pdf/2609.12188v1)

**作者:** Krishna Bhavithavya Kidambi `[一作]` `[通讯]` (University of Dayton), Krishna Bhavithavya Kidambi (University of Dayton)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种面向多旋翼机的电池感知预测轨迹规划框架，在多重风扰动环境下，通过闭环预测多旋翼、马达、电池和电压依赖的驱动器交互，生成既能降低能耗又能提升跟踪精度的高度调整轨迹。

**💡 创新点**

创新点在于：① 将电池的SOC、端电压与马达电机速度限制直接耦合到轨迹评估中；② 使用SOC依赖的两元电容模型和电压限制的减速器模型，实现电池-驱动器功率/速度的实时反馈；③ 采用低维高度参数化与闭环动力学预测相结合的优化方法，兼顾能耗、跟踪误差与电池安全边界。

**🔧 技术方法**

使用的技术包括：多旋翼六自由度动力学、PID/自适应/滑模/RISE等闭环控制器、SOC依赖的Thevenin等效电路电池模型、基于车辆-马达-电池耦合的离散闭环预测、非线性优化求解器（SQP）以及基于基函数的轨迹参数化。

**📊 数据集**

实验基于仿真，设置150 s、640 m 长度的多旋翼任务，包含三段已知空间风扰动区域；没有使用真实飞行数据集，而是通过内部仿真生成的风场、车辆参数与电池参数。

**📈 对比分析**

与传统仅考虑风扰动的轨迹（Disturbance‑aware）、仅考虑能耗的轨迹（Energy‑aware, battery‑unaware）以及本文的电池感知轨迹（Battery‑aware）比较。结果显示，在92 % SOC下，电池感知轨迹将能耗降低7.5 %并将跟踪RMSE降低约72 %；在低SOC（55 %）下，电池相关约束显著影响轨迹选择，尽管整体性能提升较小。进一步通过四种不同反馈控制器验证，发现控制器会改变能耗与跟踪的权衡。

**⚠️ 局限性**

局限性包括：① 电池模型仅在仿真中验证，未进行实验验证；② 规划是一次性全局优化，未实现在线重新规划；③ 在极低SOC或极端扰动下仍可能不可行；④ 假设扰动场已知且静态，未考虑扰动估计误差。

---

## 3. A Physics-Based Closed-Loop Robotic Bioprinting Framework Towards Volumetric Muscle Loss Treatment

**arXiv ID:** 2609.12159 | [PDF](https://arxiv.org/pdf/2609.12159v1)

**作者:** Omid Rezayof `[一作]` (University of Texas at Austin), Farshid Alambeigi `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

设计并验证了一个基于物理模型的闭环机械生物打印框架，能够在近实时下通过视觉厚度评估自动调节打印压力和针头高度，以实现目标线珠厚度。

**💡 创新点**

创新点在于：①不依赖大规模训练数据或专门的流变模型，采用结构光相机和自研的实时视觉厚度检测算法实现全自动评估；②采用积分控制调节压力并通过简化的动态高度调节(DHA)线性模型间接控制针头高度；③实现了从打印起始到目标厚度收敛的平均5.2秒闭环控制。

**🔧 技术方法**

使用技术包括：KUKA LBR Med7 7自由度机械臂 + 气压式打印头 + Zivid 2 M70结构光相机 + ROS2-Humble + 视觉厚度检测流程（灰度化、掩模、Canny边缘、轮廓追踪）+ Carreau流变模型 + 积分控制算法。

**📊 数据集**

数据集主要来自实验：使用蓝色染色的Absonic超声波凝胶进行流变测试，并在实际打印中收集厚度、压力、针头高度及误差等数据；未使用公开的公开数据集。

**📈 对比分析**

通过将闭环系统与关闭控制器的开环DIW进行对比，结果显示：开环误差可达5 mm，闭环误差降至<0.5 mm；收敛时间平均5.2 s；收敛压力标准差仅0.04 bar；处理时间在0.38–0.45 s，显著低于相机捕获时延（≈1.3 s）。

**⚠️ 局限性**

局限性包括：相机捕获时延限制整体系统响应速度；目前仅在单线珠、单材料（超声波凝胶）上验证，需进一步测试多材料、多维结构和更复杂的打印场景；DHA线性模型对极端流变行为可能不足。

---

## 4. Soft Symbol Grounding for Prototypical Concepts

**arXiv ID:** 2609.12247 | [PDF](https://arxiv.org/pdf/2609.12247v1)

**作者:** Marcos Galván-López `[一作]` (Instituto Politécnico Nacional), Vaishak Belle `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Soft‑PNet 方法：通过将概念归纳为在预先计算的可行符号解缓存中进行 Metropolis‑Hastings walk，并用单个标注原型生成的原型分布指导采样，利用 KL 散度将缓存权重与网络预测对齐，从而在不需要任务特定神经‑符号损失的情况下完成概念与标签的联合学习。

**💡 创新点**

创新点：
1) 去除了手工、任务特定的神经‑符号损失，统一使用单一 KL 目标；
2) 将单个感知原型与可行解缓存结合，为可行集搜索提供感知信号；
3) 通过预计算缓存和 Metropolis walk，兼顾符号约束与感知信息，适用于无法枚举解空间的任务。

**🔧 技术方法**

使用技术：
- 原型网络（Prototype Network）
- Metropolis‑Hastings 采样
- 预计算可行解缓存（可通过求解器或生成器得到）
- Kullback‑Leibler 散度训练目标
- 软归一（soft grounding）与符号推理框架的结合

**📊 数据集**

实验数据集：
- Digit Sum Parity 任务（label 仅给数字之和的奇偶性）
- Visual Sudoku Classification
- 另一个未命名的符号约束任务（文中占位符 "\u201c??\u201d"），但与上述两个任务同样采用可行解缓存和原型引导。

**📈 对比分析**

与对比方法比较：
- 与软归一基线相比，数字级准确率从 27% 提升到 96%，Visual Sudoku 从 32% 提升到 99%；
- 与手工损失的原型网络相比，概念级准确率相当或更高，标签级误差在一个标准差以内；
- 训练时间每个 epoch 减少高达 50%，并且不需要为每个任务重新设计损失。

**⚠️ 局限性**

局限性：
- 仍需预先计算可行解缓存，若解空间巨大或连续，缓存成本高；
- 只使用单个标注原型，若该原型无法覆盖概念的多样性，可能导致概念判别不充分；
- 对于无法枚举解空间的任务，需要依赖生成器产生高质量缓存，生成质量直接影响最终性能；
- 对于非常复杂的符号约束，Metropolis 更新的收敛速度可能受限。

---

## 5. The Iterative Equivariant Filter

**arXiv ID:** 2609.12328 | [PDF](https://arxiv.org/pdf/2609.12328v1)

**作者:** Pieter van Goor `[一作]` (University of Sydney), James Richard Forbes `[通讯]` (McGill University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了迭代等变滤波器（IterEqF），将迭代EKF的迭代更新步骤与等变滤波框架相结合，并在移动机器人UWB范围定位仿真中验证其性能。

**💡 创新点**

创新点在于：1) 在同质空间上推导迭代等变滤波器，首次将迭代更新与等变滤波统一；2) 通过Gauss‑Newton求解加权非线性最小二乘，天然产生重置步骤，提升瞬态收敛与统计一致性；3) 证明重置步是迭代过程的自然结果。

**🔧 技术方法**

使用了Lie群与同构空间理论、等变滤波设计、Gauss‑Newton非线性最小二乘、协方差重置、Monte‑Carlo仿真与Python实现。

**📊 数据集**

使用的是仿真数据：80 s的Lissajous轨迹、三盲点的UWB范围测距（2 Hz，噪声0.1 m），无公开真实数据集。

**📈 对比分析**

通过与标准单次更新EqF的100次Monte‑Carlo对比，评价轨迹误差、位置/航向误差、迭代次数及NEES；结果显示IterEqF在收敛速度、误差大小及统计一致性方面明显优于标准EqF。

**⚠️ 局限性**

主要局限：计算量显著增加；重置步骤对不可观测模式的影响尚未深入研究；目前验证仅基于仿真，缺乏真实硬件实验验证。

---

## 6. Performance, Efficiency and Collapse -- Advantages and Challenges in Offline Post-training of Code LLMs

**arXiv ID:** 2609.11956 | [PDF](https://arxiv.org/pdf/2609.11956v1)

**作者:** Abhinav Anand `[一作]` (TU Darmstadt), Mira Mezini `[通讯]` (TU Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了使用现有代码数据集进行完全离线的强化学习后训练，以提升代码生成大型语言模型的功能正确性；

**💡 创新点**

创新点在于首次系统评估并分析离线RL对不同模型族和规模的影响，并识别出离线训练不稳定的关键因素（logit方差与logit间隙）；

**🔧 技术方法**

采用RLOO策略梯度算法（简单且易调参）结合GRPO式优势归一化实现离线策略优化；

**📊 数据集**

主要使用CodeNet数据集（Python提交及其执行结果），并在MBPP和APPS两大基准上进行零样本评估；

**📈 对比分析**

与基线模型对比，离线RL在多种模型（0.5B-7B）上显著提升pass@1/5/10指标，最高可达48%提升，但训练过程易出现模型崩溃；

**⚠️ 局限性**

局限性包括训练仅在单语言（Python）上、仅考虑功能正确性奖励、对多语言或多维度奖励未验证、且对失稳机制的修复尚未充分验证。

---

## 7. Physics-Informed Conformal Prediction: Embedding PDE Consistency into Distribution-Free Uncertainty Quantification for Neural Operators

**arXiv ID:** 2609.11935 | [PDF](https://arxiv.org/pdf/2609.11935v1)

**作者:** Michael Chin `[一作]` `[通讯]` (Independent Researcher), Michael Chin (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Physics‑Informed Conformal Prediction（PI‑CP）与坐标感知 Fourier Neural Operator（FNO），实现了分布无关、可证明覆盖的空间自适应不确定性量化；

**💡 创新点**

创新点在于将 PDE 残差嵌入非合规性得分，既获得理论覆盖保证，又利用物理残差实现宽度自适应；同时证明 FNO 的平移等变性对 Dirichlet 边界条件构成近似障碍，坐标通道可打破此对称性并显著提升精度；

**🔧 技术方法**

技术上结合了分割合规预测、非线性正则化残差得分、谱卷积的 FNO 架构、坐标通道以及实验中对多种物理场的残差计算；

**📊 数据集**

使用六种人工合成物理数据集：二维/三维热传导、二维/三维结构力学、二维达西流、二维 Navier‑Stokes（Taylor‑Green 湍流）以及 1D 布格方程的光滑与冲击案例；

**📈 对比分析**

与传统 CNN、DeepONet、MGN 等基准以及 MC Dropout、Deep Ensembles 的比较表明，PI‑CP 在所有场景下都能保持 89–91% 的理论覆盖率，且平均宽度增幅仅 4%，FNO 在相同任务上比基准提升 10–12 倍；

**⚠️ 局限性**

局限性包括对 PDE 残差-误差相关性的依赖（若相关性低则不适用）、仅针对 Dirichlet 边界条件、对冲击波等非光滑解的 FNO 失效，以及在分布偏移下的交换性假设可能不成立。

---

## 8. Feedback Capacity of Stationary Gaussian Channels: An Optimal Schalkwijk-Kailath Scheme

**arXiv ID:** 2609.12069 | [PDF](https://arxiv.org/pdf/2609.12069v1)

**作者:** David Fay `[一作]` (Hebrew University of Jerusalem), Oron Sabag `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文证明了在带有有理功率谱密度（rational PSD）的平稳有色高斯噪声通道中，反馈容量的最优输入不需要包含反馈无关的高斯成分，并基于此构造了一种可实现容量的 Schalkwijk–Kailath（SK）编码方案；该方案在任意小于容量的速率下实现双指数衰减的最大误码概率。

**💡 创新点**

创新点在于：①用凸优化与微扰分析完整证明了 Kim 的“成分消除”断言；②给出了针对任何最优解的显式 SK 编码方案；③展示了在有理 PSD 下能够实现双指数误码衰减，从而实现容量。

**🔧 技术方法**

主要技术手段包括：状态空间模型（SSM）与卡尔曼滤波、凸优化与 LMI（线性矩阵不等式）、扰动分析、Lyapunov 方程、Schur 补（Schur complement）和矩阵不等式等。

**📊 数据集**

本工作为纯理论研究，无使用任何实测数据集；所有结果均在数学分析和仿真上验证。

**📈 对比分析**

与之前的结果相比，本文消除了先前证明中的缺陷，保证了 SK 方案确实是容量实现方案；误码概率实现了双指数衰减，远优于传统的单指数或多项式衰减。

**⚠️ 局限性**

局限性：证明仅适用于有理功率谱密度的平稳噪声；对非有理（连续）功率谱、MIMO 通道或非平稳噪声的成分消除与容量实现机制尚未得到解决。

---

## 9. LettuceVisSim: A Simulator That Generates Lettuce Image Time-series for Vision-Based Reinforcement Learning

**arXiv ID:** 2609.12505 | [PDF](https://arxiv.org/pdf/2609.12505v1)

**作者:** Ziye Zhu `[一作]` (Wageningen University & Research), Sjoerd Boersma `[通讯]` (Wageningen University & Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文开发并验证了一款名为 LettuceVisSim 的模拟器，能够高效生成标记的时间序列叶菜图像，用于支持室内环境农业的视觉强化学习。

**💡 创新点**

创新点在于将过程型生长模型、植被布局算法和 Unity 渲染引擎整合为模块化体系，并提出两阶段分段立方回归将干重映射到投影面积，解决了标签数据匮乏问题。

**🔧 技术方法**

所采用技术包括 Van Henten 叶菜生长过程模型、圆形投影布局、Unity 3D 渲染、DeepLabv3+ 语义分割以及 PPO 强化学习算法。

**📊 数据集**

使用了 3rd Autonomous Greenhouse Challenge 的两套公开数据集：Dataset (i) 的单株干重与投影面积数据，以及 Dataset (ii) 的 1 m²冠层时序 RGB 图像、环境记录和周度干重测量。

**📈 对比分析**

通过六折留策略交叉验证评估 PBM、五折交叉验证评估 PPA 回归，使用 R² 与 RRMSE 量化性能；PBM 在动态密度下 R²≈0.84；两段回归 R²≈0.94；冠层布局在测量干重下 GCR R²≈0.84，PBM 输入下 R²≈0.40；Unity 渲染平均耗时 <10 ms；强化学习实验实现光照控制，最终干重落入 15–17 g/株区间。

**⚠️ 局限性**

局限性包括仅针对单一叶菜品种，PBM 预测误差相对较大，冠层布局假设过于理想化（忽略植株间不规则性与相机成像失真），并在部分策略（如 veggie‑might1）中表现欠佳，需进一步提升模型通用性与精度。

---

## 10. PLSP (Pre-hoc Liminal Space Profiling): OOD Prediction over Detection -- An Anticipatory Approach for Machine Learning Model Reliability

**arXiv ID:** 2609.12225 | [PDF](https://arxiv.org/pdf/2609.12225v1)

**作者:** Vipul Bansal `[一作]` (University of Wisconsin Madison), Deepak Dhungana `[通讯]` (IMC University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了预先评估OOD预测的PLSP框架，并设计了无数据集依赖的CREDS指标，用于衡量模型在分布漂移下的可信度。

**💡 创新点**

创新点在于将OOD评估从事后检测转为预期预测，构建CREDS、可信度曲线、热图和VUS等一套全新的预评估工具，能够在没有OOB数据的情况下预测模型鲁棒性。

**🔧 技术方法**

采用KL散度计算特征级可信度，结合特征重要性与类别权重进行加权聚合；利用高斯近似、积分AUC和VUS评估不同分布变化下的可信度；实验中使用随机森林获取特征重要性。

**📊 数据集**

实验数据集包括MNIST、SVHN、CIFAR-10、CINIC-10、STL-10，并通过在这些数据上加入10%、20%、30%和40%的高斯噪声进行评估。

**📈 对比分析**

将CREDS与传统的post‑hoc准确率/ROC指标对齐，发现二者随噪声或分布变化呈负相关；CREDS能在无OOB数据的情况下提供与准确率相近的鲁棒性估计，VUS等指标进一步量化模型对均值/方差漂移的敏感性。

**⚠️ 局限性**

局限性在于CREDS依赖于高斯近似和特征独立假设，难以捕捉复杂非高斯或相关特征的分布漂移；对高维特征的解释性有限，且不提供对训练过程的直接反馈。

---

## 11. Decoding Mixture Perception through Computational Modeling of Component Interactions

**arXiv ID:** 2609.11958 | [PDF](https://arxiv.org/pdf/2609.11958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 12. I Am No One: Style-Aware Paraphrasing for Text Anonymization

**arXiv ID:** 2609.12341 | [PDF](https://arxiv.org/pdf/2609.12341v1)

**作者:** Ahmed Sohair Khan `[一作]` (RMIT University), Elham Naghizade `[通讯]` (RMIT University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于大型语言模型的风格感知文本匿名化框架，先为每位作者生成简洁的风格描述，再利用该描述指导文本重写以消除可识别的风格痕迹，同时保持语义和可读性。

**💡 创新点**

① 将作者风格特征显式化为可解释的四维描述（句长、词汇、语气、标点）；② 通过提示式控制让LLM在重写时精准抑制这些特征，而非噪声注入；③ 在多模型和多语料上验证其对作者识别率的显著下降。

**🔧 技术方法**

提示式风格提取与风格引导式重写、基于LLM的文本生成、作者识别评估（DeBERTa/BERT）以及多种实用度量（余弦相似、困惑度、加权KL）。

**📊 数据集**

Author10（10位博主共15k篇长文）和Illinois9（9位评论者共4k条短评），以及在Yelp、IMDB等外域数据验证泛化。

**📈 对比分析**

与多种DP（ε=25/100/250）、非DP、Paraphrase、ALISON基线比较，作者识别F1从原始66.5%下降至26%（约60%降低），语义相似度保持≈0.70，困惑度仅轻微提升，整体在隐私-实用性折中上显著优于对手。

**⚠️ 局限性**

依赖LLM对预设风格维度的准确识别，可能忽略更细粒度的句法或语用特征；评估指标混合了内容失真与隐私；仅针对文本而非语音转写；未考虑多模态隐私与公平性等实际部署挑战。

---

## 13. Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale

**arXiv ID:** 2609.11998 | [PDF](https://arxiv.org/pdf/2609.11998v1)

**作者:** Vaibhav Singh `[一作]` (Mila), Oleksiy Ostapenko `[通讯]` (ServiceNow Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种新的块扩散语言模型（Block Diffusion Language Models），通过块级解码和状态空间缓存来提高推理效率，克服了传统自回归模型的限制。

**💡 创新点**

创新点在于提出了一种状态空间模型（State-Space Model, SSM）作为块缓存，能够在常量内存和低延迟下处理长上下文，同时保持高吞吐量和准确性。

**🔧 技术方法**

使用了块扩散模型和状态空间模型（如Mamba），结合了块因果训练目标和缓存解码接口。

**📊 数据集**

使用了300B个标记的训练数据集，进行了三种3B参数的块扩散去噪模型的预训练。

**📈 对比分析**

与传统的自注意力模型相比，Mamba缓存在256k个标记时提供了4.3倍更低的延迟，11倍更少的内存和2.6倍更高的单流吞吐量，且在长上下文检索中表现优越。

**⚠️ 局限性**

限制在于模型是在1024个标记的上下文下训练的，因此长上下文评估主要是探测外推能力，而非经过训练的长上下文能力。

---

## 14. Efficient Vision-Language-Action Management and Serving for Robot Factories

**arXiv ID:** 2609.12075 | [PDF](https://arxiv.org/pdf/2609.12075v1)

**作者:** Dionysios Adamopoulos `[一作]` (Max Planck Institute for Software Systems), Christina Giannoula `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 715 | [OpenAlex ID](https://openalex.org/A5044162748)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个针对多机器人、多模型、多GPU边缘服务器的VLA推理服务与管理系统，满足严格的SLO要求。

**💡 创新点**

创新点在于GPU内部两流（VLM/ADiT）阶段分离、锁步动态SM划分、共享流实现多模型共存，以及基于整数规划的智能流量控制，显著提升并发吞吐与延迟。

**🔧 技术方法**

采用CUDA流、CUDA图、SM分区与绿色上下文实现阶段并行、锁步同步，并通过整数规划进行机器人-GPU分配与流量调度。

**📊 数据集**

使用LIBERO数据集的摄像头图像与文本指令作为输入。

**📈 对比分析**

通过与Monolithic（单流水线）和vLLM-Omni（跨GPU分离阶段）对比，平均提升机器人负载数×2–3倍，SLO达成率保持≥98%，单台4GPU服务器可支持数十台机器人。

**⚠️ 局限性**

依赖高端GPU资源，需手工配置模型放置与参数，且对网络延迟变化与模型更新的自适应性有限。

---

## 15. From the Task Boundaries of Narrative Text to Structural Anchoring, Uncertainty Triggers, and Cross-Calibration

**arXiv ID:** 2609.12453 | [PDF](https://arxiv.org/pdf/2609.12453v1)

**作者:** Bowen Deng `[一作]` (Sun Yat-sen University), Daifeng Li `[通讯]` (Sun Yat-sen University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了CoNS-Explorer系统，并通过两项研究评估直接解释与情境故事文本在因果图解释中的效果及用户资源使用；

**💡 创新点**

首次在同一因果模型下实现事实匹配的双文本生成，提出解释路由过程框架与四个可检验命题，阐明多种解释资源如何按需求动态协同；

**🔧 技术方法**

采用DAG/SCM自动生成解释、GLMM与GEE统计分析、访谈转录与代码化分析、实验设计与交互式可视化技术；

**📊 数据集**

使用了两份经过审查的因果图模型（教育场景与宏观经济场景），并扩展到稻田管理与供应链牛鞭效应，实验样本分别为240名受试者与24名访谈参与者；

**📈 对比分析**

通过受试者对比实验（Direct vs Story）使用GLMM与GEE估计准确率差异，发现Story在总效调整任务上显著提升准确率且提升情境存在感；自由浏览研究结合行为轨迹与访谈揭示资源切换与校准过程；

**⚠️ 局限性**

受限于任务难度较低、样本规模有限、双文本仅比较整体未拆分成单个要素、实验任务顺序固定、Story的阅读成本与熟悉度未完全控制，未来需扩大条目并行、进一步解构文本成分并测试多元因素。

---

## 16. Granularity-Adaptive Credit Assignment for Long-Horizon LLM Agent Reinforcement Learning

**arXiv ID:** 2609.12424 | [PDF](https://arxiv.org/pdf/2609.12424v1)

**作者:** Taoran Liang `[一作]` (Nankai University), Bin Chong `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种GACA（Granularity‑Adaptive Credit Assignment）算法，利用每步的负对数似然（NLL）作为临界性评分，动态混合步骤级和集体级优势，改进LLM代理在稀疏奖励长序列中的信用分配。

**💡 创新点**

创新点在于将信用分配的粒度视为状态相关的，通过NLL自动决定每步应使用的步骤优势权重，摆脱了传统固定权重导致的噪声传递问题。

**🔧 技术方法**

使用了无评判器的分组优势归一化、anchor‑state 组归并、步骤级NLL评分、基于评分的自适应混合权重以及理论证明的误差界定。

**📊 数据集**

实验数据集为 ALFWorld（文本模拟家庭任务）和 WebShop（基于网页的购物任务），使用 Qwen2.5 1.5B 与 7B 两个规模的模型。

**📈 对比分析**

与 GRPO（仅集体优势）和 GiGPO（固定权重的步骤优势）进行比较，GACA 在 ALFWorld 上提升 10–24% 成功率，在 WebShop 上提升 8–14%，且收敛更快、样本效率更高。

**⚠️ 局限性**

局限性包括：NLL 评分为单样本且噪声大，导致权重重置不无偏；需手动调节基准权重与调制强度；仅在所选任务上验证，未知在更广泛场景中的通用性。

---

## 17. IDORacle: Template-Guided SQL-Sink Mediation for Object-Level Authorization in Java Applications

**arXiv ID:** 2609.12426 | [PDF](https://arxiv.org/pdf/2609.12426v1)

**作者:** Yuewantong Song `[一作]` (Zhengzhou University), Jiangxing Wu `[通讯]` (Zhengzhou University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对 Java‑SQL 应用的 IDOR 预防框架 IDORacle。

**💡 创新点**

创新点在于通过模板引导的 SQL 句子拦截和重写，在运行时将身份上下文与 SQL 绑定，实现了横向权限提升的低侵入式防御。

**🔧 技术方法**

主要技术包括服务器端追踪标识、MyBatis/JDBC 级别的 SQL AST 解析、双指纹模板计划、授权谓词注入与冗余感知缓存。

**📊 数据集**

使用了基于真实 CVE 报告的 Java‑SQL 基准集，覆盖多种所有权模型和攻击情形。

**📈 对比分析**

通过对比 NO_GUARD、GUARD_NO_CACHE 与 GUARD_WITH_CACHE 三种模式，验证了防御成功率 100%，平均每实例开销在 0.017 ms，最大 0.17 ms，端到端延迟提升不足 5 ms。

**⚠️ 局限性**

局限性包括仅在 JDBC 层拦截，对高度动态 ORM、存储过程等场景效果有限，且需可靠的身份上下文传递与元数据维护。

---

## 18. Reality Is the Final Verifier: On Two Key Gaps in Agentic Software Engineering

**arXiv ID:** 2609.12039 | [PDF](https://arxiv.org/pdf/2609.12039v1)

**作者:** Alexander Krentsel `[一作]` (University of California Berkeley), Ion Stoica `[通讯]` (University of California Berkeley)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“两个缺口”框架并设计了内外两循环的保障修订架构，以解决AI编码代理在需求与模型缺口导致的奖励黑客与幻觉问题

**💡 创新点**

首次将需求缺口和模型缺口统一为系统性失效根源，并将软件保障视为人类判断、代理能力与计算资源的优化问题

**🔧 技术方法**

综合使用形式化规范、自动化评估器层级、连续学习与监控日志分析等技术手段来闭环修订需求、模型与评估器

**📊 数据集**

未在论文中使用特定公开数据集，而是通过真实案例（如键值存储、MoE负载平衡、SWE-bench审计、安全基准逃逸）来验证框架

**📈 对比分析**

通过对比传统单循环开发与两循环架构在防止部署误行为、降低影响及复发率等方面的效果，表明两循环能显著提升可靠性，尽管缺少量化实验数据

**⚠️ 局限性**

局限性在于仍需依赖人工判断来弥补需求缺口，模型缺口评估成本高，且未给出可量化的性能基准，实际部署时对资源调度与安全约束的实现仍是挑战

---

## 19. Multi-Objective Agent-Based Model Predictive Controller for Plug-and-Play Vehicle Control

**arXiv ID:** 2609.12108 | [PDF](https://arxiv.org/pdf/2609.12108v1)

**作者:** Jiaming Zhong `[一作]` (University of Waterloo), Amir Khajepour `[通讯]` (University of Waterloo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于ADMM的多目标代理式模型预测控制（AMPC），实现车辆控制系统的“即插即用”分布式设计。

**💡 创新点**

创新点在于系统性地设计了三种分布式形式（完全分布式、虚拟中心节点、局部显式），并证明局部显式形式既能近似全局最优又具备显著的实时计算优势。

**🔧 技术方法**

采用分布式模型预测控制与交替方向乘子法（ADMM）相结合的技术框架，处理多目标、约束以及控制正则化。

**📊 数据集**

使用了高保真CarSim仿真数据和真实四驱电动车的实验数据进行验证。

**📈 对比分析**

与传统集成MPC进行比较，局部显式AMPC在子最优误差<5 Nm、迭代次数<15、计算时间比<1的条件下，既逼近全局最优，又实现了更快的实时控制。

**⚠️ 局限性**

局限性包括目前仅考虑两组目标的分解，缺乏对更大规模多目标/非线性系统的理论保证，未来需扩展到多组及更复杂非线性场景。

---

## 20. A Data-Driven Distributed Control Scheme: Learning Multi-Objective Agent-Based MPC for Path-Tracking

**arXiv ID:** 2609.12142 | [PDF](https://arxiv.org/pdf/2609.12142v1)

**作者:** Jiaming Zhong `[一作]` (University of Waterloo), Amir Khajepour `[通讯]` (University of Waterloo)

**通讯引用:** 20080 | [OpenAlex ID](https://openalex.org/A5020470831)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于ADMM的学习型多目标代理型模型预测控制（AMPC），通过高斯过程学习初始化迭代，显著提升分布式MPC的收敛速度和计算效率。

**💡 创新点**

创新点在于：①将多目标问题拆解为各目标局部优化并通过信息交换实现全局最优；②利用高斯过程预测拉格朗日乘子和松弛变量，提前给定迭代初值，显著减少迭代次数；③结合数据管理与认证模块，保证学习可靠性与实时性。

**🔧 技术方法**

采用了ADMM框架实现分布式MPC，使用高斯过程回归（GPR）进行学习与预测，结合CarSim仿真平台的实时仿真进行验证。

**📊 数据集**

使用的是基于CarSim的赛车道路径跟踪仿真数据，所有数据均来自仿真生成的车辆状态与轨迹，无公开真实数据集。

**📈 对比分析**

与传统集成MPC做对比：在相同路径跟踪任务中，学习型AMPC在保持相同控制性能的前提下，计算时间平均降低43.5%；加入学习后，迭代次数从约10–20次降至5次以下，计算时间进一步降低88.6%，比无学习版本快近90%，比集成MPC快近93%。

**⚠️ 局限性**

局限性包括：仅在线性凸MPC模型和仿真环境下验证；对大规模、强非线性或信息交换受限的系统仍需进一步评估；学习模型需要足够多的训练数据，初期性能受限；并未在真实道路或多车队环境中进行实车验证。

---

## 21. Space as an Interventional Invariant: Cross-Modal Predictive Geometry for Stratified Cities and Em-Spaced Intelligence

**arXiv ID:** 2609.11959 | [PDF](https://arxiv.org/pdf/2609.11959v1)

**作者:** Tao Yang `[一作]` (Tsinghua University), Haijiang Li `[通讯]` (Cardiff University)

**通讯引用:** 44426 | [OpenAlex ID](https://openalex.org/A5065859286)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出把空间定义为介入不变的最小关系结构，构建跨模态预测几何并用层化纤维化模型描述城市，证明可识别性并在合成实验中验证。

**💡 创新点**

创新点在于：①将空间视为介入不变，统一多模态感知与行动；②在群体化介入下给出中央化中心化识别定理；③使用细胞层化纤维（sheaf）捕捉多层城市几何并揭示霍洛尼妨碍；④构建可检验的“em‑spaced”智能框架。

**🔧 技术方法**

使用预测状态表示、群体化介入组oid、可逆性与协变损失、细胞纤维拉普拉斯、Procrustes估计、投影限制、以及项目极限等数学工具。

**📊 数据集**

主要用合成的二维环面数据，生成视觉、听觉、触觉和四种模态观测，再通过不同动作集进行实验；未使用真实城市传感器数据。

**📈 对比分析**

与仅视觉、仅听觉、联合但动作盲/错位以及完整动作匹配进行对比。正确动作联合模态下MSE ≈0.0058，动作盲/错位约0.082，单模态约0.337；在不同噪声水平下评估协变损失、霍洛尼残差、跨尺度残差等，表现符合理论预期。

**⚠️ 局限性**

局限性包括：需满足联合点分离、协变、介入真实性等假设；假设无序列忽略与正性；对非平稳城市过程不适用；计算量大，近似时会改变预测商；社会经济几何的规范性和解释性问题。

---

## 22. Function Name Is All You Need to Detect Blockchain Application Attacks

**arXiv ID:** 2609.12315 | [PDF](https://arxiv.org/pdf/2609.12315v1)

**作者:** Rui Xi `[一作]` (University of British Columbia), Karthik Pattabiraman `[通讯]` (University of British Columbia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种名为Translucent的攻击检测框架，利用交易调用链中的函数名称序列自动识别区块链应用层攻击。

**💡 创新点**

创新点在于证明仅使用函数名称序列（覆盖率高达98.46%）即可捕捉业务语义，消除对源代码、手工规则或复杂图结构的依赖，并通过Transformer学习攻击模式。

**🔧 技术方法**

核心技术包括：①调用序列构造器将原始调用链映射为函数名称序列；②位置增强的Transformer分类器进行攻击预测；④集成梯度用于结果可解释性；同时利用函数选择器-名称数据库。

**📊 数据集**

实验数据集为DeFiHackLab的424起真实攻击事件（14,611笔攻击交易）与其对应的受害者交易以及从2024-01到2025-12的5.37亿条Ethereum正向交易。

**📈 对比分析**

与基线长度阈值、CLUE和Hunting比较，Translucent在攻击实例上FNR仅为1.56%（低于CLUE 70.77%和Hunting 83.17%），在所有交易上FPR为0.0017%（远低于CLUE 0.5-1.5%），且平均检测时延仅为24.9 ms，显著优于竞争方案。

**⚠️ 局限性**

局限性包括：对函数名称数据库的依赖可能在攻击者完全隐藏其函数名时稍有下降（FNR升至2.26%）；训练数据来源于社区标签，可能存在标注不一致；以及仅能检测已公开到 mempool 的交易，无法覆盖私下提交的攻击。

---

## 23. Communication-Constrained Multi-Robot Exploration With Adaptive Communication Windows

**arXiv ID:** 2609.12502 | [PDF](https://arxiv.org/pdf/2609.12502v1)

**作者:** Ben Rossano `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了多机器人在通信间歇性环境下的自主探索，提出MACE框架，通过主动评估通信价值并与局部前沿规划结合，实现更高效率的探索。

**💡 创新点**

创新点在于将通信机会视为带预算的车辆或游览问题，机器人在预定窗口内评估通话成本并决定是否去通信点；同时结合区域分配和前沿选择，兼顾通信与探索。

**🔧 技术方法**

使用的技术包括分布式前沿搜索、基于图的探索区域分配、车辆或游览问题（VOP）优化、OR-Tools求解、移动与通信图构建、Dijkstra/A*路径规划、聚类与竞标分配。

**📊 数据集**

实验使用四种模拟地图：迷宫（250×250 m）、隧道（500×350 m）、改良隧道、城市街区（600×500 m），每地图随机20个起点，三机器人使用200 m LiDAR和视距通信。

**📈 对比分析**

与固定 rendezvous 以及纯 opportunistic 两种基线比较，MACE 在所有环境中平均比基线快最多 23%，在大规模或连通性低的环境中提升更显著，且在最差 25% 样本中优势更大。

**⚠️ 局限性**

局限性在于仅在离散网格模拟中验证，未考虑真实动力学、障碍物误差、通信信号衰减细节，对通信间隔与阈值敏感，缺乏对更复杂动态环境的评估。

---

## 24. BRIDGE-EEG: Bridging Self-Supervised Pretraining and Efficient Deployment for Cross-Dataset EEG Classification

**arXiv ID:** 2609.12218 | [PDF](https://arxiv.org/pdf/2609.12218v1)

**作者:** Meghna Roy Chowdhury `[一作]` (Purdue University), Shreyas Sen `[通讯]` (Purdue University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 BRIDGE‑EEG pipeline：先将异构 EEG 记录统一映射为 62 通道时频表示，再用 SimCLR 在无标签数据上预训练 SE‑ResNet18 作为教师，随后通过任务无关与任务特定两种知识蒸馏策略压缩为 SE‑ResNet8/4，最终实现多任务 EEG 分类且能在边缘设备上高效部署。

**💡 创新点**

创新点在于：① 统一 62 通道时间‑频率预处理，兼容不同 montage、采样率与通道数；② 结合 SimCLR 自监督预训练与 SE‑ResNet+SE 块构建高可迁移特征；③ 两种蒸馏策略实现共享初始化与任务特定细化；④ 在服务器 GPU、桌面 CPU 与 NVIDIA Jetson Orin Nano 三层硬件上测量实时能耗与延迟，验证压缩模型可与大型基础模型竞争。

**🔧 技术方法**

技术手段包括：SimCLR 对比学习（NT‑Xent、MSE）、Squeeze‑Excitation ResNet、短时傅里叶变换（STFT）时频表示、通道与时间/频率遮蔽等数据增强、任务无关/任务特定知识蒸馏、以及硬件能耗/延迟测量。

**📊 数据集**

使用的预训练数据集：五个无标签 EEG 数据库（TUH‑Sz、CHB‑MIT、DEAP、PhysioNet MI、SEED‑IV 等），下游评测集：六个任务（异常检测 TUAB、SIENA；情绪识别 SEED、EmoEEG；运动意象 BCI‑IV‑2a、2b）。

**📈 对比分析**

与多种 EEG 基础模型（LaBraM、BIOT、CBraMod 等）和任务专用模型对比：在异常检测与情绪识别上，SE‑ResNet8/4 的准确率与大模型持平甚至更优，参数量仅 10–1000 倍；在运动意象任务上表现略逊，说明预训练多样性不足。能耗实验表明，SE‑ResNet8 在 Jetson Orin Nano 上比教师节能 3 倍以上，延迟显著降低。

**⚠️ 局限性**

局限性包括：① 运动意象任务的预训练数据量与多样性不足导致性能差距；② 小学生在多类别情绪识别时易受类不平衡影响；③ 仍未在 MCU 级别完成量化与部署验证；④ 对极低功耗场景的进一步压缩与自适应学习仍待研究。

---

## 25. ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels

**arXiv ID:** 2609.12379 | [PDF](https://arxiv.org/pdf/2609.12379v1)

**作者:** Leshan Li `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于编码代理的框架，用于自动生成针对特定模型的自回归解码Megakernel，并通过中间状态测试Oracle保证生成的内核可验证且性能可靠。

**💡 创新点**

创新点在于：①提出十条通用结构里程碑，指导代理在不依赖模型参数的情况下构建高效Megakernel；②设计独立的中间状态测试Oracle，能够在生成过程中实时校验字节传输、数值误差与算术精度；③将里程碑与Oracle结合的迭代搜索框架，利用GPU测量反馈实现自动调优。

**🔧 技术方法**

主要技术包括：GPU持续内核生成的编码代理（Claude Opus 5），十步里程碑指导的结构化生成，基于字节计数、float64参考与算术宽度约束的中间状态Oracle，GPU实时性能与精度测量，SGLang集成与评估。

**📊 数据集**

使用了八个模型族（Llama-3.2-1B、Llama-3.1-8B、Llama-2-13B-chat、Qwen3系列、Qwen2.5-3B、MiniCPM4、MiniCPM5、NanBeige4-3B），共14个解码细胞，参数范围0.6B–13B，批量与上下文长度从1–16和96–4096不等。

**📈 对比分析**

与SGLang、vLLM、llama.cpp、MPK等基线比较，生成的Megakernel在模型带宽利用率(MBU)上平均提升1.21×（相较SGLang）和1.54×（相较MPK），在SGLang内部的GSM8K推理速度提升平均1.11×，且保持或超过基线的准确率。

**⚠️ 局限性**

局限性在于：实验仅在单一硬件平台（H100 80GB）上验证；里程碑与Oracle可能对其他GPU架构、共享内存预算或线程数不完全适用；生成过程仍需数十轮迭代，累计成本在数百美元左右；此外，缺乏对极端大模型或多GPU场景的评估。

---

## 26. Inverting Self-Triggered Control: Adversarial Reinforcement Learning for Sparse Denial-of-Service Attacks

**arXiv ID:** 2609.12016 | [PDF](https://arxiv.org/pdf/2609.12016v1)

**作者:** Adam Haroon `[一作]` (Iowa State University), Cody Fleming `[通讯]` (Iowa State University)

**通讯引用:** 938 | [OpenAlex ID](https://openalex.org/A5113457201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种自触发的对抗性强化学习控制（RL-STC）方法，旨在学习最稀疏的干扰或拒绝服务（DoS）调度，以破坏闭环系统的稳定性。

**💡 创新点**

创新点在于将自触发结构反转到攻击方，构建了一个自触发的对抗者，并证明了在特定条件下，最小干扰次数的下界。

**🔧 技术方法**

使用了深度Q网络（DQN）作为对抗者，并通过强化学习训练其在不同植物（Pendulum、CartPole和Quadrotor2D）上的表现。

**📊 数据集**

使用了Pendulum、CartPole和Quadrotor2D这三个数据集进行实验，评估了不同防御者的表现。

**📈 对比分析**

与三种非学习型对抗者（无干扰、周期性干扰和基于谓词的贪婪对抗者）进行比较，学习型对抗者在所有植物上均以100%的成功率击败了每个防御者，并在干扰时间/失败上比基线快2.8倍。

**⚠️ 局限性**

限制在于该方法依赖于线性化模型，且只在三种植物上进行了评估，未考虑更广泛的应用场景。

---

## 27. LifeFuse-Mem: Lifecycle-Aware State Fusion Against Temporary Overwriting for Long-Term Memory

**arXiv ID:** 2609.12436 | [PDF](https://arxiv.org/pdf/2609.12436v1)

**作者:** Hanyu Zhao `[一作]` (Beijing Academy of Artificial Intelligence), Li Du `[通讯]` (Beijing Academy of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对长期运行的LLM代理中的临时覆盖问题，提出了一种生命周期感知的记忆框架LifeFuse-Mem，能够在写入时区分持久与临时信息，并在读取时使用阶段感知的保护式读出，防止临时信息覆盖永久知识。

**💡 创新点**

通过学习可分离的可塑与稳定子空间、引入基于生命周期标签的永久路由器、以及在评估阶段融合Phase‑A与Phase‑B状态来实现对信息生命周期的显式建模，从而显著降低临时覆盖率。

**🔧 技术方法**

使用基于低秩状态的记忆更新（类似δ‑Mem）结合可正交化的子空间分离、token‑级永久路由器、阶段感知保护读出以及多项辅助损失（硬候选、保留排名、稀疏门控）。

**📊 数据集**

主要使用自定义的Hard Attribution Anti‑Overwrite基准（1,000 期），以及公开的长记忆基准LoCoMo和MemoryAgentBench。

**📈 对比分析**

在Hard Attribution基准上，LifeFuse‑Mem相较于δ‑Mem在“已获取”子集上的保留率提升至63.7%/69.4%（Qwen3‑4B/SmolLM3‑3B），覆盖率下降至36.3%/30.6%；在LoCoMo与MemoryAgentBench中保持与δ‑Mem相当或略有提升，表明兼容性良好。

**⚠️ 局限性**

仅在显式生命周期标签和阶段边界已知的评估环境下才显现优势，对无标签的真实交互场景无法自动推断生命周期；同时在部分基准的类别（如LRU、SF）表现略有下降，说明方法对不同记忆需求的通用性尚待进一步验证。

---

## 28. RiPPLE: Cross-Space Performance Prediction from Early Training for Neural Architecture Search

**arXiv ID:** 2609.12418 | [PDF](https://arxiv.org/pdf/2609.12418v1)

**作者:** Yifan Yang `[一作]` (University of New South Wales), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种名为RiPPLE的NAS预测器，能够在仅占整个搜索空间训练预算一小部分的情况下，对整个网络空间进行准确排名。

**💡 创新点**

核心创新是将部分训练信息作为标签而非特征，仅在少量覆盖性锚点上训练并提取标签，再通过无标签特征进行传播，从而实现训练成本与排名覆盖范围分离。

**🔧 技术方法**

采用锚点覆盖选择（farthest‑point sampling）、短前缀训练与学习曲线读取、零成本代理与结构编码的无标签特征、极随机树回归器进行标签传播。

**📊 数据集**

在十二个不同的搜索空间上评估，涵盖四大族：NAS‑Bench‑201、NAS‑Bench‑101、NATS‑Bench‑SSS、NAS‑Bench‑NLP、TransNAS‑Bench‑101，以及极大DARTS空间（约10^18个架构）。

**📈 对比分析**

与基于图卷积、Gaussian Process和SemiNAS等全空间预测器，以及最新的Transformer编码器（NAR‑Former、NN‑Former）进行比较；在统一的150全训练等价预算下，RiPPLE在绝大多数细胞中获得接近Oracle的Kendall τ（≈0.85），并在顶层选择的网络在CIFAR‑10/100上与最优差距仅为0.08–0.83个百分点。

**⚠️ 局限性**

局限性包括：无法在单个架构上超越专用的多精度搜索；曲线读取器对训练计划的假设限制了适用性；覆盖分析与实际回归器相对保守，未实现自适应锚点选择或学习曲线自适应读取。

---

## 29. Spatial Mixing and Deterministic Approximate Counting of Multi-spin Systems beyond Bounded Degree Graphs

**arXiv ID:** 2609.12352 | [PDF](https://arxiv.org/pdf/2609.12352v1)

**作者:** Zhidan Li `[一作]` (Shanghai Jiao Tong University), Kuan Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种新的递归构造有理多面体并利用线性分式规划实现多自旋系统的确定性 FPTAS，并将其推广到稀疏 Erdős–Rényi 随机图的合法 q‑着色计数。

**💡 创新点**

创新点包括：① 通过保持概率向量之间的相互约束，消除了传统坐标独立分析导致的收敛条件过于严格；② 结合 Birkhoff 收缩与自避行走（connective constant）实现了对无界度图的空间混合与计数；③ 采用“允许块”递归与多面体来处理着色问题，显著把先前常数 3 降至 2 并逼近空间混合阈值。

**🔧 技术方法**

核心技术：自旋系统的比例递归 + Hilbert 距离 + Birkhoff 收缩系数；递归构造有理多面体并使用线性分式规划求极值；自避行走计数与连通常数分析；在着色问题中利用允许块、列表着色与多面体约束来控制误差。

**📊 数据集**

使用的“数据集”是理论模型：任意图 G，尤其是稀疏 Erdős–Rényi 随机图 G∼G(n,d/n)；论文没有实验数据，而是给出高概率解析证明。

**📈 对比分析**

与之前的递归关联性或 Markov 链混合速率方法相比，所提出的算法在可达 q≥(2+η)d 范围内实现了更小的常数（2 对比 3）且保持了多项式时间。比值误差在 1±ε 以内，且算法为确定性而非随机；在随机图实例上高概率满足计数精度，复杂度为 (n+log(1/ε))^C。

**⚠️ 局限性**

局限性：① 仅适用于正相互作用的多自旋系统；② 依赖图的连通常数可控，对高连通或不具可控自避行走的图无法直接扩展；③ 对于 q‑着色，仍需 q≥(2+η)d，无法处理更小的色数；④ 计算复杂度中多面体维度与递归深度导致实际常数较大，可能在大规模实例中不可行；⑤ 负相互作用或非正向量的情况尚未覆盖。

---

## 30. Recommendation Retrievers Need Verifiers: Universal Generative Reranking for Sequential Recommendations

**arXiv ID:** 2609.12270 | [PDF](https://arxiv.org/pdf/2609.12270v1)

**作者:** Benyu Zhang `[一作]` (Meta MRS), Neeraj Bhatia `[通讯]` (Meta MRS)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种后置生成式验证器，用来提升已冻结检索器的前缀候选质量。

**💡 创新点**

创新点在于将轻量级自回归生成器附加到冻结的检索器上，利用候选项的标识符序列概率进行输出侧重排，而不需要重新训练检索器或重建索引。

**🔧 技术方法**

主要技术包括基于项目标识符（语义ID+哈希ID）的自回归Transformer、stop‑gradient投影、教师强制交叉熵训练以及相互排名融合 (RRF)。

**📊 数据集**

实验使用亚马逊评论（Industrial、Office）和 YaMBDa 音频推荐数据集（50M/500M/5B 交互规模），并对比 MiniOneRec 的 LLM 检索器。

**📈 对比分析**

与原始检索器（SASRec、GRU4Rec、NextItNet、MiniOneRec）相比，验证器在 Recall@10、Recall@3/5/10/100 等指标上提升 6–30%，在 5B 规模下最高可达 30% 的提升。

**⚠️ 局限性**

局限性包括：对候选池大小敏感、仅提升前缀而非完整排序、依赖于项目标识符的编码设计、在已极强检索器上提升空间有限，且需要手动调参验证器权重。

---

## 31. Trajectory Bundle Method in SE(3) for Black-Box Fixed-Wing Aircraft Trajectory Optimization

**arXiv ID:** 2609.12248 | [PDF](https://arxiv.org/pdf/2609.12248v1)

**作者:** Matthew D. Osburn `[一作]` (Brigham Young University), John L. Salmon `[通讯]` (Brigham Young University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种在SE(3) Lie群上实现的轨迹束方法（TBM），能够在黑盒、无导数动力学下完成动态可行轨迹优化，并在旋转孔洞飞行实验中得到验证。

**💡 创新点**

创新点包括：1) 在SE(3)上构造并传播轨迹束，利用指数/对数映射避免欧拉角奇点；2) 对SE(3) TBM的插值误差给出二次(Δ²)上界并阐明与局部 Lipschitz 常数的关系；3) 提出束尺寸经验调节规则以平衡动态可行性与误差。

**🔧 技术方法**

采用的技术包括：Lie群轨迹束方法、无导数凸子问题求解器（CVXPY）、PyFlyt的黑盒航空动力学与碰撞检测、闭环仿真和高频控制器验证。

**📊 数据集**

使用的数据来源为 PyFlyt 提供的仿真模型（航空动力学、控制面混合器、碰撞检测），并在实验中通过在不同旋转角度下的旋转矩形孔洞环境进行性能测试。

**📈 对比分析**

通过在 30°、60°、90° 旋转孔洞下对比不同束尺度（γ_b = 0.5, 1, 2, 4）的收敛次数、目标值、动态违例和清晰度，表明中等束尺度（γ_b≈1–2）既能实现动态可行且通过闭环仿真验证飞行可执行，收敛次数稳定。

**⚠️ 局限性**

局限性包括：误差上界仅在局部不跨π的对数映射区域内成立；对不光滑碰撞约束的误差缺乏严格证明；实验仅在仿真平台完成，缺少硬件验证；未处理系统不确定性与扰动。

---

## 32. Unleashing the Power of Equality Saturation for Tensor Program Superoptimization

**arXiv ID:** 2609.12330 | [PDF](https://arxiv.org/pdf/2609.12330v1)

**作者:** Qi Zhan `[一作]` (Zhejiang University), Shanping Li `[通讯]` (Zhejiang University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于等价饱和的张量程序超级优化器，统一了高层张量表达式与分块实现，支持联合优化、裁剪和子图组合。

**💡 创新点**

创新点在于：①统一的类型化 IR 兼容张量表达式与并行实现；②在等价搜索中引入早期裁剪以消除冗余候选；③通过子图划分和组合扩大搜索规模。

**🔧 技术方法**

使用技术包括等价饱和（e-graph）、e-graph 规则（代数重写、并行细化、融合、存储决策）、早期裁剪（partial program equivalence）、子图划分、Triton 代码生成、LP 估算与递进调度。

**📊 数据集**

实验使用 Mirage、Prism 基准，以及 Llama‑2、BERT 的 Transformer 层，覆盖归一化、MLP、LoRA、因果多头注意力等工作负载。

**📈 对比分析**

与 PyTorch eager、TVM、Mirage、Trinity、FlashAttention 等基线对比；几乎在所有配置中都取得几倍加速（几何平均加速 2–4 倍，最大可达 5–6 倍）。在解码阶段 FlashAttention 也被进一步加速。

**⚠️ 局限性**

局限性包括：①等价饱和的搜索空间随程序规模急剧增长，需四小时预算；②对 GPU 体系结构敏感，需针对不同硬件调整；③仍需手工调优部分参数；④某些归一化层仍未获得最佳加速。

---

## 33. The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition

**arXiv ID:** 2609.12259 | [PDF](https://arxiv.org/pdf/2609.12259v1)

**作者:** Samuel Larson `[一作]` `[通讯]` (Pebble ML), Samuel Larson (Pebble ML)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在单一状态瓶颈下，Transformer 学习矩阵状态的秩是否与任务所需的最小表示维度一致，并通过组合词问题验证了秩的因果必要性。

**💡 创新点**

提出了“秩法则”，证明梯度下降会自发地招募与任务代数相匹配的秩，并通过预注册的强制秩实验证明其因果必需性。

**🔧 技术方法**

使用 Transformer 编码器、固定读取器、秩截断、cosine 余弦距离损失、中心化估计有效秩和对角块嵌入的最小忠实表示。

**📊 数据集**

在五个有限群（S₃、S₄、A₅、S₅、A₆）的词问题上进行实验，利用匹配维度的 S₄/A₅ 对比验证同维度下秩的等价性。

**📈 对比分析**

通过 Spearman ρ = 0.9747 与预注册阈值比较，显示招募秩与理论维度高度相关；强制秩实验表明 k≥所需秩时可达 0.9 余弦相似度，k<所需秩时恒为 0。

**⚠️ 局限性**

实验规模仅为 <1M 参数的合成测试集，未覆盖预训练语言模型；强制秩实验仅在四个种子上验证，S₅ 的训练步数需重新预注册；结果不一定推广到更大规模或更复杂任务。

---

## 34. ChitraMiti: Benchmarking Visual Grounding and Modality Reliance in Bengali Geometric Reasoning

**arXiv ID:** 2609.12509 | [PDF](https://arxiv.org/pdf/2609.12509v1)

**作者:** Khan Raiyan Ibne Reza `[一作]` (North South University), Md Adnan Arefeen `[通讯]` (North South University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了用于低资源语言的双模几何推理基准——ChitraMiti-12.8k（12,874道合成孟加拉语平面几何题）和NCTB-500（500道来自孟加拉语教材的实物图形题），并设计了三阶段评估协议（仅图、图+结构化描述、仅结构化描述），对多种开源与闭源视觉‑语言模型进行零样本和微调评估；

**💡 创新点**

创新点在于：①首次为孟加拉语构建大规模双模几何数据集；②通过结构化15属性描述实现文本与图像信息等价评估；③使用统计等价检验验证描述可替代图像；④针对空间关系与数值信息进行消融与跨模态验证，揭示模型对文本依赖与视觉一致性不足；

**🔧 技术方法**

采用 Gemini 3.1 Flash Lite 进行问题翻译与结构化描述生成；使用 QLoRA 对 Qwen3.5‑4B 进行微调；评估使用 Qwen3‑VL‑8B‑Instruct、Gemma‑4‑26B、LLaMA‑3.2‑11B、Gemini 2.5 Flash 与 GPT‑4o‑mini；统计分析采用 TOST 等价检验；对抗性空间关系交换实验用于测试跨模态验证；答案匹配采用 SymPy 与 GPT‑5.4‑nano；

**📊 数据集**

核心数据集为 ChitraMiti‑12.8k（合成图形+15属性描述）与 NCTB‑500（教材图形+问题），训练集含 11,231 条 ChitraMiti 数据；测试集中包含 1,000 条 ChitraMiti 与 500 条 NCTB；

**📈 对比分析**

评估方式为三阶段对比并采用 TOST 判断 B 与 C 的等价性；实验显示 B 与 C 在所有模型上差异均 ≤5%；准确率范围约 13–35%（如 Gemma‑4‑26B 34%），微调后 Qwen3.5‑4B 在 ChitraMiti‑1k 上提升至 34.2%，在 NCTB‑500 上提升至 23.6%（仍低于最佳零样本 35%）；对抗实验中，误配空间关系导致 2–12% 的准确率下降；

**⚠️ 局限性**

局限性包括：仅聚焦孟加拉语平面几何；微调仅在单一开源模型上验证；整体准确率低；潜在预训练知识泄漏；评价基准受 5% TOST 边界影响；对抗性评测未显式要求模型识别不匹配，且仅测量准确率下降。

---

## 35. Understanding Whole-Body Robot Teleoperation Strategies Under Diverse Task Objectives and Constraints

**arXiv ID:** 2609.12384 | [PDF](https://arxiv.org/pdf/2609.12384v1)

**作者:** Tsung-Chi Lin `[一作]` (New Jersey Institute of Technology), Chien-Ming Huang `[通讯]` (Johns Hopkins University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文开发了一套基于VR的混合控制框架，实现对TIAGo移动机械手的全身远程操作，并通过用户实验验证了其对不同任务约束下的协同控制策略。

**💡 创新点**

创新点在于将自由形式与受限控制结合的混合模式，辅以面向技能培养的课程化训练，系统性地研究了协同控制在时间、精度与工作负荷不同约束下的适应性。

**🔧 技术方法**

采用HTC Vive Pro 2头盔与手柄进行6-DoF姿态追踪，利用TRAC-IK求解末端执行器姿态，结合自定义GUI实时渲染摄像头视频和机器人状态，算法实现自由与受限控制的闭环映射。

**📊 数据集**

数据集为18名受试者完成四项家务任务（无约束、时间限制、精度关键、工作负荷自适应）的行为记录与NASA‑TLX问卷，采集控制激活时间、速度、误差率等指标。

**📈 对比分析**

通过混合回归与重复测量ANOVA评估控制与任务指标，结果显示协同控制显著降低任务完成时间与关节极限冲击，受限控制降低误差率，整体性能优于传统单一控制方式。

**⚠️ 局限性**

局限性包括缺乏触觉反馈、仅使用单摄像头进行感知、未实现三维空间重建，未来需引入多模态感知与自适应辅助以提升操作自然度与效率。

---

## 36. ParaRecover: A Process-Level Benchmark for Error Localization and Recovery in Parallel Tool-Use Agents

**arXiv ID:** 2609.12345 | [PDF](https://arxiv.org/pdf/2609.12345v1)

**作者:** Bowen Guan `[一作]` (Dalian University of Technology), Yanming Shen `[通讯]` (Dalian University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文建立了一个过程层面的基准 ParaRecover，用于评估多轮并行工具使用中的错误定位与恢复能力，并提出了 SDE（结构、诊断、演化）三维评估框架。

**💡 创新点**

创新点包括：①细粒度错误分类（14类）覆盖规划依赖、工具选择、参数匹配等；②两难度级别（LEVEL‑1 单轮错误、LEVEL‑2 多轮错误传播）；③SDE 评估维度及其 LLM-as-a-Judge 评判方法；④基于 SDE 进行 SFT 与 DPO 微调以提升恢复表现。

**🔧 技术方法**

使用的技术包括：DAG 结构表示并行任务；Chain-of-Thought 促进推理；多模型 LLM（GPT‑4o、Claude、Gemini、Qwen、GLM、DeepSeek 等）生成与评判；SFT 与 DPO 进行模型微调；模拟工具调用环境进行执行。

**📊 数据集**

数据集来源于 BUTTON 任务的真实执行轨迹，并通过注入 14 种预定义错误构造增强轨迹，共 10,626 条样本，划分为 LEVEL‑1 与 LEVEL‑2 两难度级别。

**📈 对比分析**

实验通过 SDE Rubric 对主流 LLM 进行评估，平均得分低于 70，尤其在演化策略维度表现最差；PASS@1 仍高于 89%；大型模型相对小型模型表现更好；通过 SFT 与 DPO 微调后，SDE 得分和 Judge‑free 指标（工具使用效率、E5、无效 DAG 率、PASS@1）均有显著提升。

**⚠️ 局限性**

局限性：①使用模拟环境，缺乏真实 API 的不确定性与动态性；②LLM-as-a-Judge 评判可能引入偏差；③仅采用 DAG 结构，未覆盖循环、条件分支等更复杂的执行模式。

---

## 37. The Battery Price of edge AI: A study of the Environmental Impact of LLM Inference on Mobile Devices

**arXiv ID:** 2609.11940 | [PDF](https://arxiv.org/pdf/2609.11940v1)

**作者:** Édouard Guégain `[一作]` (Greenspector), Tristan Coignion `[通讯]` (University of Bordeaux)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对18种不同规模、量化级别的LLM在Pixel 8、iPhone 14和Nvidia A100服务器（批量与非批量）上进行能耗、延迟、准确度及生命周期环境影响的系统测评，揭示本地推理在能效、碳排放和设备寿命方面的缺陷；

**💡 创新点**

创新点在于首次结合手机电池寿命、设备内在碳排放和批量服务器对比，发现4‑bit量化是能耗的“甜点”，并绘制了能耗‑准确率 Pareto 前沿，提供针对电池电量与精度需求的模型路由依据；

**🔧 技术方法**

采用PocketPal与 进行手机端推理测量，利用 RAPL 与 GPU 监控工具获取服务器能耗，使用 Q2/Q4/Q6 量化实现；

**📊 数据集**

使用自定义的开放式文本生成提示（从100个维基百科热门标题抽样）进行能耗测试，并在CommonsenseQA、GSM8K、HumanEval与TruthfulQA四个基准上评估准确率；

**📈 对比分析**

通过对比不同硬件、模型尺寸、量化级别的能耗/延迟/准确率，发现本地推理平均比批量服务器慢约1.5倍、能耗高约3倍；在能耗‑准确率平衡上有8种配置位于 Pareto 前沿；

**⚠️ 局限性**

局限包括：仅测量输出期间能耗，忽略预填充；准确率采用四个基准平均加权，可能不代表真实任务；手机与服务器使用不同推理框架，导致能耗对比受框架影响；仅评估三家模型族、两部旗舰手机和单一服务器 GPU，未覆盖中端或未来硬件；任务仅限开放式文本生成，其他工作负载可能表现不同；

---

## 38. Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity?

**arXiv ID:** 2609.12162 | [PDF](https://arxiv.org/pdf/2609.12162v1)

**作者:** Obinna I. Ekekezie `[一作]` `[通讯]` (Cambridge Health Alliance), Obinna I. Ekekezie (Cambridge Health Alliance)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在 Draft‑Verify‑Revise 体系中，LLM 在处理指向性歧义（如“previous”）时如何出现视角差异，并评估不同模型在此情形下的判定准确性。

**💡 创新点**

创新点在于：①使用精心设计的 10 个最小对比基准，专门考察指向性歧义导致的视角转移；②将此问题拆解为三角色（生成、验证、复核）并单独评估复核阶段的决策；③采用 e‑value 序贯统计方法实现无偏停止和多组实验的显著性检验；④对比错误标签的存在与否，揭示评审反馈结构对复核逻辑的影响。

**🔧 技术方法**

技术上使用了六个大型推理模型（OpenAI GPT‑5‑mini/5.2，Anthropic Claude Haiku 4.5/Opus 4.6，Google Gemini 3 Flash/Pro），在不同 reasoning‑effort 级别下执行 21 个配置；实验通过多提供商批量推理管线实现，利用结构化输出与自定义 JSON schema 收集二元判定；随后用 e‑value 与置信区间进行统计检验，并用独立 probe 对错误判定的理由进行后验分析。

**📊 数据集**

使用了 30 条合成刺激（10 个基准 × 3 条件），每条在每个配置下重复 20 次，总计 12 600 次试验；刺激保持所有上下文不变，仅改变“previous”在生成稿与验证稿中的解释，从而确保唯一变量为视角差异。

**📈 对比分析**

在主实验中，模型的平衡准确率从 0.156（GPT‑5.2 无推理）提升至 0.965（Gemini 3 Pro 低推理）。在 ablation 实验中，去除错误标签后，部分模型的准确率提升（如 GPT‑5.2 xhigh 从 0.942 提升至 0.953）。大型模型往往优于小型，但 GPT‑5.2 与 GPT‑5‑mini 的表现差异不显著。成本‑性能平衡显示 Gemini 3 Pro 低推理以约 5% 成本获得最高准确率。

**⚠️ 局限性**

局限性包括：①仅测试“previous”这一指向性词汇，未涵盖其他常见歧义；②合成数据可能不完全代表真实流水线中的多样上下文；③复核阶段仅做二元判定，未评估实际改写质量；④ probe 的判定依赖模型自身表达，缺乏人类校准；⑤模型对字段标签的依赖可能是训练或后训练偏差导致，未在实验中完全排除。

---

## 39. SynthSentry: Detecting Synthetic Data Contamination in Language Model Training Data

**arXiv ID:** 2609.12353 | [PDF](https://arxiv.org/pdf/2609.12353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 40. ArtManip: Category-Level Articulated In-Hand Manipulation

**arXiv ID:** 2609.12498 | [PDF](https://arxiv.org/pdf/2609.12498v1)

**作者:** Yang Yang `[一作]` (Zhejiang University), Siyuan Huang `[通讯]` (BIGAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于原语生成和功能抓取的框架，用于学习可在多实例和多抓取配置下执行的类别级内手关节操作策略，并实现了零样本真实世界部署。

**💡 创新点**

创新地将程序化对象生成、功能抓取模板与教师-学生强化学习相结合，形成可在自由漂浮关节物体上实现稳定抓握与关节驱动的类别级策略。

**🔧 技术方法**

使用强化学习（SAPG）、关节动力学随机化、奖励层级、教师-学生潜在表示蒸馏，以及基于触点约束的功能抓取与原语参数化的物体生成。

**📊 数据集**

在IsaacGym中构建四类（刀、打火机、订书机、镊子）共30个训练实例和5个测试实例的原语化物体；在真实世界使用12个未见对象进行零样本测试。

**📈 对比分析**

与单实例专用策略对比，本文在所有类别上实现100%实例覆盖、约93%抓取覆盖、平均5-8轮开闭循环，真实世界执行成功率达85.7%，显著优于单实例和无随机化基线。

**⚠️ 局限性**

目前仅支持单自由度两连杆物体且假设已给定功能抓取，未处理自主抓取或更复杂关节结构。

---

## 41. A Non-constant Lower Bound for Grammar-Based Compression with Greedy

**arXiv ID:** 2609.12106 | [PDF](https://arxiv.org/pdf/2609.12106v1)

**作者:** Danny Hucke `[一作]` `[通讯]`, Danny Hucke

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

证明了全局基于语法的压缩算法（global grammar compressor）的近似比下界为Ω(log n / log log n)，即最坏情况下压缩比至少与此量级成正比。

**💡 创新点**

首次给出非常数下界，解决了超过二十年的未解决问题，并通过 Lean 4 形式化验证保证了证明的可靠性。

**🔧 技术方法**

采用 de Bruijn 序列、7/4^+‑free 变换、基数表示以及层级化的语法规则构造，无穷词族 w_h，并利用组合计数与替换收益比较技术证明下界。

**📊 数据集**

本工作不使用实验数据集，而是构造理论上的无穷词族 w_h（长度在 2^h 到 2^50h 之间，字母表大小为 h^2+4h）。

**📈 对比分析**

通过理论分析与形式化验证相结合，证明在任何执行路径下生成的终止语法规模至少为 h^3/8(4r+1)，从而得出比值 Ω(log n / log log n)。并未进行实验比较，只给出最坏情况下的下界。

**⚠️ 局限性**

仅适用于左到右出现替换的全局压缩算法，对局部或其他变体不一定适用；构造复杂且实现难度高，证明仅在理论层面给出。

---

## 42. Automated Detection and Structuring of Social Tipping Point Evidence in Climate related Documents: A Modular AI Framework

**arXiv ID:** 2609.12254 | [PDF](https://arxiv.org/pdf/2609.12254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 43. PATH: Continuous Target Sensing among Autonomous Cooperative Drones

**arXiv ID:** 2609.12456 | [PDF](https://arxiv.org/pdf/2609.12456v1)

**作者:** Heegyeong Kim `[一作]` (Macquarie University), Richard Han `[通讯]` (Macquarie University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

通过建立两架UAV间的局部3D几何模型，实现目标在传感器视角变化下的无缝传递；并通过“相互一致握手”机制在接收端验证目标，完成目标跟踪的安全切换。

**💡 创新点**

创新点在于：①采用局部几何对齐而非全局GNSS或激光标记，降低对硬件的依赖；②利用RGB‑D重建与基于fiducial的相对姿态估计，将目标空间坐标投影到接收UAV图像中，形成空间先验；③设计轻量化的交叉视角验证流程，减少通信负载。

**🔧 技术方法**

核心技术包括RGB‑D目标重建、fiducial基相对姿态估计、三维投影与空间先验匹配、基于IoU的相互一致握手、CPU‑only几何运算与极简通信协议。

**📊 数据集**

实验数据来自真实UAV飞行：室内配备超声波定位的实验（420次测量），室外视觉相似目标（人、机器人车、微UAV）以及不同硬件平台（DJI Tello、定制Quad）进行的跟踪切换。

**📈 对比分析**

与传统基于外观的匹配方法（ORB、XFeat）对比，PATH在视觉模糊场景下实现帧级接收端目标获取准确率96.0%，误检率2.0%，漏检率2.0%；几何误差仅为0.047 m（相对位置）和0.030 m（目标位置）。通信量低于16 kB/s，计算在Jetson Nano上可实现30‑60 Hz实时。

**⚠️ 局限性**

主要局限包括：依赖接收端持续观测发射端fiducial标记，遮挡或失去标记会导致投影失效；投影误差受相对姿态与深度误差影响；实验仅覆盖双UAV切换，未验证多机队协同；在强干扰或雷达干扰环境下的鲁棒性仍需提升。

---

## 44. WinSyn: An Automated Pipeline for Realistic Enterprise Question-Answering Evaluation

**arXiv ID:** 2609.12171 | [PDF](https://arxiv.org/pdf/2609.12171v1)

**作者:** Amey Varhade `[一作]` (Microsoft Research), Navin Goyal `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了WinSyn管道，自动生成高复杂度、时间一致性强的企业邮件问答数据集；

**💡 创新点**

创新点在于采用分层“自上而下”的生成流程，结合中间结构验证和反思修正，保证答案可追溯与完整性；

**🔧 技术方法**

使用大型语言模型（如GPT‑4o、GPT‑5.4）配合BM25+向量检索、ReAct 与 Onyx 代理框架进行检索与生成；

**📊 数据集**

构造了四个基于不同技术背景的合成数据集（Cloud DevOps、Data Platform、Stripe Billing、VS Code Changelog），每个包含数百封邮件、数十个员工及多种查询；

**📈 对比分析**

在ReAct与Onyx两种代理上进行评估，综合指标得分在0.75–0.90之间，ReAct在长文本查询上表现更好，Onyx在短文本查询上更精准，但整体仍低于80%；

**⚠️ 局限性**

局限包括仅生成邮件而非多模态交流，未量化统计真实性，缺少外部交互、公司文化与不可解答查询，且评测未使用句级归属验证。

---

## 45. Investigating Developer-Reported Software Security Testing Challenges

**arXiv ID:** 2609.12008 | [PDF](https://arxiv.org/pdf/2609.12008v1)

**作者:** Md Erfan `[一作]` (University of Alabama), Md Rayhanur Rahman `[通讯]` (University of Alabama)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过收集并分析Stack Overflow上与软件安全测试相关的问题，构建了一套包含8大类、31个子类的层次化分类体系，并对这些挑战在普及度、难度、相关性、时间趋势及共现等维度进行了定量与定性分析。

**💡 创新点**

创新点在于首次系统性地从社区讨论中挖掘并归纳软件安全测试的开发者面临的挑战，提出了完整的分类框架，并对挑战的演化规律与互相关联模式进行了实证研究，同时验证了该分类体系对后续未出现问题的适用性。

**🔧 技术方法**

使用了人工标注、归纳式编码与主题分析构建分类体系，随后运用计数、均值、Pearson与Spearman相关系数、Cox‑Stuart趋势检验及Jaccard与Lift等共现度量对挑战进行多维度量化。

**📊 数据集**

数据集为17,743条利用安全测试相关关键词检索的Stack Overflow提问，经过筛选后得到582条符合条件的问题；此外还抽取了21条时间上更晚的“hold‑out”问题用于稳定性验证。

**📈 对比分析**

通过对比各维度指标，研究显示例如“工具选择与指导”与“认证与授权测试”在社区关注度与解决率上存在差异；对held‑out集的覆盖率达100%，证明分类体系的稳健性；实验未与其他工具或方法直接对标，但表明对社区讨论的覆盖与解释能力优秀。

**⚠️ 局限性**

局限性包括：关键词检索可能漏检或误检，研究仅基于公开的Stack Overflow数据，缺乏对私有或行业内讨论的覆盖；人工标注仍受主观影响；分类的适用性在其他平台或不同行业的验证尚未完成。

---

## 46. Representation-based Masked Diffusion Model

**arXiv ID:** 2609.12382 | [PDF](https://arxiv.org/pdf/2609.12382v1)

**作者:** Yangrong Hu `[一作]` (Hong Kong Polytechnic University), Jian Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于全局连续潜变量的Masked Diffusion模型（RMDM），利用MeanFlow将预训练编码器的语义表示映射为可采样的高斯分布，作为全局条件来指导并行token更新，从而显著提升少步生成质量。

**💡 创新点**

创新点在于：①引入全局语义潜变量z显式建模序列级语义，缓解并行采样中的互相依赖；②使用MeanFlow将编码器嵌入空间对齐为高斯先验，解决训练与推理的分布不匹配；③在MDM训练中条件化RMDM，系统性降低Conditional Dependency Gap，实现高效且连贯的并行生成。

**🔧 技术方法**

使用技术包括Masked Diffusion Models、MeanFlow流匹配、预训练编码器（如BERT/Qwen3）、Diffusion Transformer（DiT）、Adaptive Layer Normalization (AdaLN) 条件化、Gaussian latent采样、两阶段训练框架以及总相关性分析等。

**📊 数据集**

实验使用的主要数据集有 OpenWebText（OWT）、One Billion Word（LM1B）、WikiText、LAMBADA、AG News、Pubmed、Arxiv 等无条件文本生成基准。

**📈 对比分析**

与MDLM、SEDD、DiffusionLM等基线在相同步数下对比，RMDM在少步采样场景下的GenPPL、LLM Judge评分均优于基线；在匹配质量时吞吐量提升约3.6×；在LM1B的PPL亦不逊于AR模型，验证了其生成质量与效率的双重提升。

**⚠️ 局限性**

局限性包括：实验仅在无条件生成、GPT‑2级规模下验证；对更大规模模型或条件生成任务的迁移性未探究；潜在的先验对齐误差和对隐变量的过拟合风险仍需进一步研究。

---

## 47. Self-Verifying Anomaly Detection using Explainable AI for Cybersecurity of DER Networks

**arXiv ID:** 2609.12305 | [PDF](https://arxiv.org/pdf/2609.12305v1)

**作者:** Damilola Popoola `[一作]` (Iowa State University), Manimaran Govindarasu `[通讯]` (Iowa State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套名为ExCYDER的基于XAI的异常检测框架，集成了边缘实时检测与云端自验证，专为DER网络的SOC运营设计。

**💡 创新点**

创新点在于将LightGBM的决策规则与SHAP特征归因结合，形成自验证的一致性指标（Overlap、Direction、Coverage），实现模型内部推理与解释的实时对齐，提升可信度与可审计性。

**🔧 技术方法**

采用LightGBM、SMOTE、Min‑Max归一化、特征选择、SHAP、规则提取与一致性计算，并在Python/Apple M4上实现边缘与云端工作流，利用Streamlit等技术实现可视化。

**📊 数据集**

使用Iowa State University DER DNP3真实数据集，包含六类交通标签（正常、DoS、远程终端攻击、扫描攻击、DNP3隐蔽攻击、重放攻击）。

**📈 对比分析**

与XGBoost、Random Forest比较，LightGBM在准确率、召回率、F1均超过99%，误报率低至0.51%；验证成功率为54.5%，平均一致性44.6%，SHAP计算时延14.5 ms/警报，CPU利用率<70%，内存<900 MB，整体检测准确率>98%。

**⚠️ 局限性**

局限性包括仅在模拟环境测试，未在真实DER网络部署；验证阈值设定为50%可能漏报；实验仅覆盖DNP3协议，缺乏对多协议与更大规模流量的验证。

---

## 48. MAIA: Multi-Agent Intent Articulation for Requirement Discovery in Art Commissions

**arXiv ID:** 2609.12097 | [PDF](https://arxiv.org/pdf/2609.12097v1)

**作者:** Yu-Chao Wang `[一作]` (Purdue University), Tim McGraw `[通讯]` (Purdue University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在数字艺术委托中，提出一种多代理系统MAIA，用苏格拉底式对话来帮助普通委托人把模糊的情感意图转化为可执行的文字化简稿。

**💡 创新点**

创新点在于：①将“验证优于发明”规则与多代理架构相结合，构建验证门（validator gate）以防止AI随意假设；②采用四状态认知标签追踪需求进度；③通过主动的马尔萨斯式提问（maieutic inquiry）引导用户逐步细化需求。

**🔧 技术方法**

技术实现上使用OpenCode平台的多代理框架，底层模型为MiniMax M2.5 LLM（温度0.2），并实现意图诊断、验证、合成三子代理及状态对象管理。

**📊 数据集**

实验数据来自16名无美术背景受试者，完成两种情境（叙事情感与商业促销）各一次，另外抽样8份简报由三名专业概念艺术家进行盲评。

**📈 对比分析**

与单代理对话基线相比，MAIA在认知支持（Cognitive Support）上显著提升（Wilcoxon r=0.96, p=0.015；LMM β=1.69, p<0.001）。艺术家盲评显示，MAIA生成的简报在视觉完整性、可执行性和专业度上均优于基线（Wilcoxon p=0.008，FDR q=0.010）。

**⚠️ 局限性**

主要限制包括样本量有限（N=16），艺术家评估仅为探索性且评审者数少；对比只测试了完整配置与最小基线，缺乏中间控制；所有结果仅基于单一模型，无法验证跨模型泛化；CoT流程导致响应延迟。

---

## 49. MInTRL: Off-policy Intervention can boost On-policy RL

**arXiv ID:** 2609.12419 | [PDF](https://arxiv.org/pdf/2609.12419v1)

**作者:** Mingyu Chen `[一作]` (Amazon Web Services), Chris Kong `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Minimal Intervention Reinforcement Learning (MInTRL) 的方法，在生成时通过少量局部纠正（由判别-干预策略提供）来扩展探索空间，同时保持大部分轨迹为 on‑policy；并用优势回归目标在混合来源轨迹上直接优化，无需重要性采样；

**💡 创新点**

核心创新是：① 用稀疏局部干预在保持 on‑policy 性质的同时打开更广的成功路径；② 通过序列级优势回归绕过行为策略的重要性修正；③ 提供覆盖‑可学习性权衡的理论分析；④ 在数学推理和代码生成上获得显著提升；

**🔧 技术方法**

技术细节包括：半 on‑policy 采样（生成-评审-干预循环）、判别‑干预策略（可为更强的教师或自模型的上下文增强版）、优势回归（KL 正则化 RL + 价值基准）、松弛锚点处理干预词、早停策略、局部步骤级干预；

**📊 数据集**

使用的实验数据集包括数学推理（AIME 2025/26、HMMT 2025；AceReason‑Nemotron 片段）和代码生成（DeepCoder‑Preview、LiveCodeBench、HumanEval+、MBPP+）；

**📈 对比分析**

与基线（Base、GRPO、OPD、SFT+GRPO、MENTOR）以及自监督干预对比，MInTRL‑Const 在两大模型规模（Qwen3‑1.7B、Qwen3‑4B）上平均提升 3–13 pp，最大提升 9.44 pp；实验表明 2–4 % 的 off‑policy 令量为最佳，过高会导致性能下降；

**⚠️ 局限性**

局限性包括：需要细致调节干预强度与频率；过多干预会降低可学习性；依赖判别‑干预模型的质量与匹配度；需要提前停止干预以避免干扰；当前方法主要验证在数学与代码生成任务，其他领域的泛化仍待评估；

---

## 50. Almost Sure Convergence Analysis of Stochastic Gradient Methods with Clipping and Additive Noise

**arXiv ID:** 2609.12119 | [PDF](https://arxiv.org/pdf/2609.12119v1)

**作者:** Amartya Mukherjee `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

研究了带梯度裁剪与高斯噪声的随机梯度下降（SGD-CN）及其动量变体（SHB-CN、SNAG-CN）的几乎必然收敛性与收敛速率，提供了完整的理论分析；

**💡 创新点**

提出了在裁剪阈值q>σ_g时，裁剪梯度与真实梯度保持正向对齐的关键性质，并将此性质融入到超级鞅（supermartingale）框架中，实现了对裁剪偏差与噪声共同作用下的收敛证明；

**🔧 技术方法**

主要技术包括：L‑光滑与均匀有界噪声假设、剪裁梯度对齐分析、超级鞅不等式、能量函数构造、最佳迭代和最后迭代收敛的振荡控制；

**📊 数据集**

无实验数据集，全部以理论推导为主；

**📈 对比分析**

与传统无裁剪或无噪声的SGD、动量SGD比较，理论上证明了在相同步长条件下，SGD‑CN、SHB‑CN、SNAG‑CN仍能以同样的速率收敛至极值点，且最后迭代收敛到梯度零；

**⚠️ 局限性**

主要局限在于对随机梯度噪声要求“均匀有界”，比常见的“均方有限”假设更强，可能限制了结果在实际深度学习训练中的直接适用；

---

## 51. QuPAINT: Physics-Aware Multimodal Reasoning for Quantum Material Characterization

**arXiv ID:** 2609.12202 | [PDF](https://arxiv.org/pdf/2609.12202v1)

**作者:** Sankalp Pandey `[一作]` (University of Arkansas), Khoa Luu `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了多模态量子材料层数识别框架QuPAINT，结合合成数据与物理先验实现自动化层数定位与推理。

**💡 创新点**

创新点在于引入物理信息注入的注意力PIA、基于薄膜干涉的合成框架Synthia、图像自适应推理监督QMat-Instruct，以及跨材质基准QF-Bench，形成了端到端可解释且跨域泛化的多模态系统。

**🔧 技术方法**

使用多模态大型语言模型（Vision Transformer + LLM）、Physics‑Informed Attention、薄膜干涉传递矩阵模型、以及对照光学先验进行特征调制。

**📊 数据集**

利用Synthia生成的合成图像与真实实验图像共计约280,526个标注样本，构成QF‑Bench基准。

**📈 对比分析**

与YOLO、MaskRCNN、MaskTerial等传统检测模型对比，QuPAINT在单层检测上AP提升至约37.3，整体AP约45.6，跨材质召回率提升并保持对未见材料的性能，显示显著的性能优势。

**⚠️ 局限性**

局限性包括合成与真实域的差距、单图像识别对光照、材料属性的高度敏感、以及数据集极度不平衡导致单层样本稀缺。

---

## 52. Grid-Free Monte Carlo for Time-Dependent Diffusion

**arXiv ID:** 2609.12306 | [PDF](https://arxiv.org/pdf/2609.12306v1)

**作者:** Zihong Zhou `[一作]` (Dartmouth College), Wojciech Jarosz `[通讯]` (Dartmouth College)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种无网格蒙特卡罗求解器，可在不构建体网格、无时间步进的前提下，对带初始条件、源项和混合边界的热方程在任意时空点直接估计解。

**💡 创新点**

核心创新在于将 Walk on Spheres/WoSt 扩展到时间依赖问题，配合无表格退出时间采样与热核采样技术，支持共享随机游走和多时间点并行求解，从而实现无离散化偏差、低方差的时变求解。

**🔧 技术方法**

实现技术包括：指数卷积+Gamma匹配的退出时间采样；方法‑of‑images与谱展开结合的热核采样；拒绝采样与多重要性采样；时间独立源项与Neumann数据的优化采样；共享游走与局部几何查询复用。

**📊 数据集**

实验使用 ITER 垂直目标的 CAD 几何作为工业案例，并在合成齿轮、二维热源等制造解上进行验证。

**📈 对比分析**

通过等时间/等计算量对比，结果表明相较于 FEM（显式 RK4/隐式 Backward Euler）和其他无网格方法（WoB、WoHB），该方法在短时与中时目标上可提升 3–5 倍的效率，且消除了时间离散化误差，在复杂几何中更能避免空间/时间混叠。

**⚠️ 局限性**

主要限制是 Monte Carlo 方差在低样本率下显著；Neumann 主导且大时间预算时，反射壁交互频繁导致成本上升；目前仅适用于线性热方程、常数扩散系数、混合 Dirichlet–Neumann 边界，未覆盖 Robin/可变系数或非线性问题。

---

## 53. ChronicleRec: Pre-training Temporally Anchored Tokens for Lifelong User Modeling

**arXiv ID:** 2609.12375 | [PDF](https://arxiv.org/pdf/2609.12375v1)

**作者:** Chengkai Huang `[一作]` (University of New South Wales), Jie Jiang `[通讯]` (Tencent Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出ChronicleRec框架，利用预训练方式将超长用户行为序列压缩为可缓存的时间锚定Chronicle Tokens，并在下游排序模型中使用；

**💡 创新点**

创新点在于递增粒度的多段合并、在合并序列中插入因果查询Token、跨时域多分支压缩以及对齐目标对压缩表示的预训练；

**🔧 技术方法**

采用自注意力的因果Transformer编码、窗口平均池化、多分支深度监督和mask‑and‑predict对齐损失；

**📊 数据集**

使用公开的KuaiRand（短视频）和工业的腾讯AdLive（直播广告）两大数据集进行评估；

**📈 对比分析**

与全序列注意力、最近窗口、VISTA单通道压缩等基线在相同评分头下对比，ChronicleRec在KuaiRand GAUC 0.5580（全序列0.5601），AdLive 0.8034（全序列0.8071），同时显著降低显存和训练时间；

**⚠️ 局限性**

限制在于仍需多分支实现，压缩过程对极端稀疏行为或高频短期行为的适应性未充分验证，且对动态更新（实时增量）压缩结果的更新策略尚不成熟。

---

## 54. Hardware-Attributed Operator Profiling for PyTorch

**arXiv ID:** 2609.11938 | [PDF](https://arxiv.org/pdf/2609.11938v1)

**作者:** Logan Chu `[一作]` (Yotta Labs), Dong Li `[通讯]` (Yotta Labs)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种硬件归因管道，自动将 NVIDIA Nsight Compute 的硬件计数器与 PyTorch 的 operator 关联，从而实现从框架级别到硬件级别的完整性能剖析。

**💡 创新点**

创新点在于：①三条互补归因路径（CUPTI 关联、NVTX 时间封闭、Inductor fusion-map 丰富）实现高可信度的归因；②采用调用顺序匹配（Invocation-Order Matching）避免跨时钟域的时间戳拼接问题；③GPU 时钟锁定与层级去重技术，显著降低重放成本并消除频率偏差。

**🔧 技术方法**

使用的技术包括：CUPTI External Activity API、NVTX 范围树查询、Inductor 调试日志解析、Invocation-Order Matching、GPU 时钟锁定、20 个精简硬件计数器的 duration‑weighted 聚合、层级去重等。

**📊 数据集**

实验数据集为三类 PyTorch 模型：GPT‑2（12 层 Transformer）、SDPA Attention（单层）和 LSTM Sequence Encoder（2 层 RNN），在 NVIDIA RTX PRO 6000 Blackwell GPU 上收集。

**📈 对比分析**

通过对比基线（Inductor FP32）与针对性的优化后版本（BF16 提升、SDPA 规范化、cuDNN 重调度等），在锁定时钟下报告的内核执行时间缩短了 1.76×（GPT‑2）、2.24×（SDPA Attention）和 3.59×（LSTM），验证了归因后优化决策的有效性。

**⚠️ 局限性**

局限性包括：仅在单个 GPU 和单一架构上验证，未覆盖多 GPU 或分布式场景；对非确定性工作流（如动态控制流）可能失效；cuDNN 绑定的 operator 归因率高达 80–90% 仍需手工或其它方法；重放阶段对多流并发的处理仍需改进。

---

## 55. MemRetriever: Learning to Search, Reflect, and Retrieve from Long-Term Memory

**arXiv ID:** 2609.11951 | [PDF](https://arxiv.org/pdf/2609.11951v1)

**作者:** Ruiyang Jiang `[一作]` (MemTensor (Shanghai) Technology), Zhiyu Li `[通讯]` (MemTensor (Shanghai) Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个多步agentic memory retrieval代理，主动检索、过滤并决定何时停止，以提升长程记忆检索与多跳问答性能。

**💡 创新点**

引入并训练多步搜索‑反思‑终止三种操作的 ReAct 代理，使用覆盖度驱动的搜索、覆盖保留的反思、以及答案质量终止奖励的多层奖励体系。

**🔧 技术方法**

ReAct 思考‑行动框架、GRPO 强化学习、覆盖度奖励、JSON 工具调用、embedding检索与 Milvus 向量库。

**📊 数据集**

LOCOMO、LongMemEval、HotpotQA、MuSiQue、2WikiMultiHopQA，并结合 MemOS 记忆系统与 Milvus。

**📈 对比分析**

与 MemOS Top‑k、Qwen3.7‑Max、DeepSeek、Gemini、ReAct‑搜索等基线比较，MemRetriever‑RL 在检索召回率、回答准确率上显著提升，尤其在多跳问答上 EM/F1/LLM‑judge 提升约 5‑10% 并降低 token 消耗。

**⚠️ 局限性**

仍受检索后端性能限制；奖励设计需要细调；在复杂长序列交互中可能出现搜索漂移；仅支持三种工具，难以处理更丰富的证据操作，且需要较多标注与 RL 训练资源。

---

## 56. Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR Foundation Models

**arXiv ID:** 2609.12277 | [PDF](https://arxiv.org/pdf/2609.12277v1)

**作者:** Yuxin Xiao `[一作]` (MIT), Xiaodong Liu `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对EHR基础模型进行强化学习微调，以提升其临床推理能力，并针对事件条件、时间窗口的推理任务设计时间感知、回滚敏感的奖励函数。

**💡 创新点**

创新点在于：①将预训练的EHR模型视为生成策略，用RL对整个轨迹进行优化；②引入时间感知与回滚敏感的奖励，解决模型rollout长度与真实临床时间窗口不匹配的问题；③证明RL微调能在数据受限时让小模型超越大模型，并在多任务训练中实现正迁移。

**🔧 技术方法**

采用强化学习（GRPO/DAPO）与verifiable rewards，对生成式EHR模型进行微调；设计时间窗口奖励函数；利用多任务RL提升跨任务迁移；对生成轨迹进行结构（编辑距离、bigram重叠）和语义（Jaccard、cosine）对齐评估。

**📊 数据集**

使用MIMIC‑IV v2.2数据集进行实验。

**📈 对比分析**

与预训练基础模型、监督微调、判别式逻辑回归、LLM（zero‑shot、ICL）等基线对比。单任务RL微调在30天再入院任务上AUROC提升至0.761–0.764、AUPRC提升至0.494–0.515，超过9M/56M预训练模型；多任务RL微调实现正迁移，小模型在数据稀缺场景中表现优于大模型。

**⚠️ 局限性**

局限性：仅在单一EHR数据集（MIMIC‑IV）和单一基础模型（ETHOS）上验证，缺乏跨数据集和跨模型的泛化评估；RL微调仍需大量计算资源，且模型与奖励设计对不同任务的通用性尚待进一步研究。

---

## 57. Amortized Low-Rank Adaptation for Model-Based Reinforcement Learning

**arXiv ID:** 2609.12278 | [PDF](https://arxiv.org/pdf/2609.12278v1)

**作者:** Fernando Palafox `[一作]` (University of Texas at Austin), David Fridovich-Keil `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用超网络在测试时生成低秩LoRA适配器，实现快速高效的世界模型自适应；

**💡 创新点**

创新点在于将超网络与LoRA结合，通过预训练同时优化基础模型和适配器生成器，兼顾表达力与计算效率；

**🔧 技术方法**

核心技术包括超网络（Hypernetwork）生成LoRA适配器、低秩权重适配、基于TD‑MPC2的世界模型架构以及元学习预训练策略；

**📊 数据集**

使用多种连续控制环境族：DeepMind Control Suite的Walker2d、Half‑Cheetah等，以及Meta‑World的Reach和Push任务；

**📈 对比分析**

与梯度自适应、基于领域随机化的单任务模型、以及无自适应的oracle基线比较，实验显示该方法在少量测试数据下既能快速收敛又能避免过拟合，性能接近oracle；

**⚠️ 局限性**

局限性包括对环境可辨识性的依赖、仅能生成在预训练环境族内的适配器、以及对数据收集策略的敏感性。

---

## 58. Specifying Paxos for System Builders: Pseudocode Made Executable

**arXiv ID:** 2609.12239 | [PDF](https://arxiv.org/pdf/2609.12239v1)

**作者:** Yanhong A. Liu `[一作]` (Stony Brook University), Rahul Sihag `[通讯]` (Stony Brook University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

将 Paxos‑for‑System‑Builders 的完整伪代码精确映射为可直接执行的 DistAlgo 规范，并通过该规范进行运行、检查与可视化，进而发现并修复伪代码中的遗漏和活性缺陷。

**💡 创新点**

① 通过逐行映射实现高保真、可执行的规范；② 利用 DistAlgo 的高阶查询与自动可视化能力，进行安全性与活性检测；③ 发现并修复伪代码中未定义函数、漏调用、以及两类活性 bug。

**🔧 技术方法**

使用 DistAlgo（Python 版）作为中间语言；利用其事件驱动、消息匹配、查询和可视化特性；配合 Python 运行时进行实验；实现了进程、消息、计时器和状态数据结构。

**📊 数据集**

无真实数据集，使用人工合成工作负载：3 台服务器、2 台客户端、每台客户端发送 100,000 条请求，模拟实际的请求流水线。

**📈 对比分析**

与官方 C 实现对比：DistAlgo 版本在相同硬件（Intel Core Ultra 7，32GB RAM）上，3 服务器、2 客户端、100k 请求，运行时间约 170 秒，平均延迟 <1.7 ms；相较 C 实现慢约 10 倍，但仍保持可接受的性能，并且提供了完整的安全/活性检查与可视化。

**⚠️ 局限性**

① 运行时性能仍显著低于 C 代码（约 10×）；② 规范仅覆盖 Paxos‑for‑System‑Builders，泛化到其他共识协议需额外工作；③ 仍未提供形式化证明，仅通过运行时检查发现缺陷；④ 需要手动维护伪代码与实现同步，若出现偏差需再次映射。

---

## 59. HeatCache: Thermal-aware Energy-efficient LLM Inference Scheduling for Chassis-level Liquid Cooling in Sustainable Edge Server Rooms

**arXiv ID:** 2609.12449 | [PDF](https://arxiv.org/pdf/2609.12449v1)

**作者:** Rui Lu `[一作]` (Hong Kong Polytechnic University), Dan Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HeatCache，一个面向机构规模LLM推理的热意识调度控制器，利用机架级AIO液冷循环的热缓冲，实时调整批量、分配与频率，以在提高机房设定点时实现能耗降低、SLO保持和热阈防止。

**💡 创新点**

创新点：①将机架级液冷循环建模为热预算；②构建HeatiTS，使用物理信息正则化的iTransformer预测批次热/电耗；③在在线控制器中将热预算与SLO约束协同调度，形成“热缓冲+主动调度”新范式。

**🔧 技术方法**

采用技术包括：RC热网络模型、iTransformer-based HeatiTS预测器、物理正则化电耗训练、基于vLLM的在线调度层、以及自制的AIO液冷实验平台。

**📊 数据集**

使用公开数据集：LMSYS-Chat-1M（聊天）、CodeAlpaca-20K（编程）、QReCC（信息检索/摘要），并部署Llama-3.1、Qwen2.5-Coder、DeepSeek-R1等LLM。

**📈 对比分析**

通过与vLLM、DSO、TAPAS、GreenLLM、CoolEdge等基线对比，HeatCache在48°C等可持续设定点下能耗降低18%，热阈曝光率下降81.7%，SLO违约率低于0.9%，显著优于其他方法。

**⚠️ 局限性**

局限性：仅验证于单机多GPU边缘服务器，未扩展到大规模集群；热模型校准依赖实验，适配不同液冷配置需要进一步验证；在极端突发负载时仍可能触发热阈；对不同LLM模型的泛化能力需进一步评估。

---

## 60. Throughput per Megabyte: A Pilot Benchmark of Language-Stack Efficiency for Self-Hosted HTTP Services on a Raspberry Pi 5

**arXiv ID:** 2609.11932 | [PDF](https://arxiv.org/pdf/2609.11932v1)

**作者:** William Oliveira `[一作]` `[通讯]` (Independent Researcher), William Oliveira (Independent Researcher)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一台 Raspberry Pi 5 上，对 Go、Rust、Python + FastAPI、Node.js + Fastify、.NET 10 Native AOT 五种语言堆栈实现的 CRUD‑over‑SQLite HTTP 服务进行基准测试，测量内存占用、吞吐量、吞吐量/内存比、启动时间、二进制大小、能耗以及并发扩展和内存时序。

**💡 创新点**

首次将 Web 框架基准从服务器级 x86 转移到 ARM64 单板机，并结合内存占用、能耗与吞吐量/内存比等自托管关键指标，提出“吞吐量‑内存比”复合指标，系统探究并发饱和和内存动态行为。

**🔧 技术方法**

使用 Go 1.26/nethttp、Rust 1.95/Axum、Python 3.13/FastAPI+Granian、Node.js 24/Fastify、.NET 10 Native AOT；通过 Lua 脚本调度负载、Pi 5 PMIC 采集功耗、/proc 监测 RSS、随机化多跑、Bootstrap 置信区间等统计方法。

**📊 数据集**

基于 1000 行预填充的 SQLite 数据库（CRUD 端点使用 1 000 行或 100 000 行 DELETE），30 秒负载周期，N = 50 次测量，随机化排除异常。

**📈 对比分析**

先在 30 并发下做 50 次测量，计算平均吞吐量、峰值 RSS 及 RPS/MB 比；随后在 30/60/120/240 并发下进行饱和扫；最终使用 Kruskal‑Wallis、Dunn 检验、Cohen’s d 等统计方法。结果显示 Rust 具备最低内存占用和最高吞吐量/内存比；.NET 拥有最高总吞吐量；Go 与 Rust 同属吞吐量簇；Node 处于中等位置；Python 最差；并发扩展中 .NET 线性扩展至 240 并发，Rust/Go plateau，Node 早期饱和，Python 无明显提升。

**⚠️ 局限性**

仅在单台 Pi 5 上测量，SQLite 写锁限制写入端点吞吐；能耗测量使用 ±10% 的 PMIC 采样且仅 1 Hz；单一作者实现可能存在偏差；未覆盖其他 ARM64 主板或客户端数据库；并发级别仅针对一个端点。

---

## 61. VS-Splat: Voxel-Selective feed-forward Gaussian Splatting for end-to-end 3D object reconstruction from sparse-views

**arXiv ID:** 2609.12343 | [PDF](https://arxiv.org/pdf/2609.12343v1)

**作者:** Yunsu Jeong `[一作]` (Sungkyunkwan University), Il Yong Chun `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个端到端的体素选择式高斯 splatting 模型 VS‑Splat，能够从稀疏视角的 2D 图像直接生成高质量 3D 对象重建和新视角渲染

**💡 创新点**

核心创新在于：① 通过可学习的体素选择模块在无 3D 结构监督的情况下识别对象中心体素；② 在这些选定体素上生成多粒度的高斯原语，从而在稀疏视角下显著提升细节表现；③ 结合可学习的 SFR 网络融合全局粗粒度与局部细节特征；④ 提供可选扩展提升对摄像机姿态噪声的鲁棒性

**🔧 技术方法**

技术包括：体素化的多视图特征聚合、轻量 3D U‑Net、Gumbel‑Sigmoid 学习体素置信度、Pose‑Aware Projection（PAP）采样、Selective Feature Refinement（SFR）稀疏 3D 卷积、全连接高斯解码器以及可选的 SHARE 预测 Plücker 嵌入

**📊 数据集**

在三个大规模基准上评估：GObjaverse、Google Scanned Objects（GSO）以及 Common Objects in 3D（CO3D），每个对象均使用 512×512 的多视角图像

**📈 对比分析**

与多种 SOTA 端到端高斯 splatting、NeRF 和密集渲染方法（如 LaRa、GS‑LRM、GeoLRM、LGM、MuRF、MVSNeRF、LVSM 等）进行对比。实验表明 VS‑Splat 在 PSNR、SSIM、LPIPS 上均优于所有对比方法，并在推理速度上较 LaRa 提升约 1.3 倍；在加入 GenDen densification 后进一步提升；在 CO3D 上的噪声姿态扩展也显著改善性能

**⚠️ 局限性**

限制主要包括：① 对体素选择阈值和 Gumbel 温度的设置仍有经验依赖；② 在极度稀疏或大背景场景下，过多或不足的细粒度原语选择可能导致渲染质量波动；③ 目前仍需要多视角输入，单视角或极少视角下的性能尚未充分验证；④ 对复杂光照或透明材质的建模仍有限

---

## 62. SAGE-Loop: Reliable Closed-Loop LLM-Driven AutoML with Trial-and-Correction and Adaptive Ensembling

**arXiv ID:** 2609.12455 | [PDF](https://arxiv.org/pdf/2609.12455v1)

**作者:** Junquan Gu `[一作]`, Hang Yu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了基于LLM的可靠闭环AutoML框架SAGE-Loop，用多轮生成和验证实现试错修正，并通过自适应集成提升模型多样性利用。

**💡 创新点**

创新点是：闭环试错–修正机制、LLM驱动的多轮模型/特征生成、基于验证证据的自适应集成（监督和无监督）以及全流程自动化。

**🔧 技术方法**

使用技术包括：大语言模型（如GPT-4o/3.5）、多轮提示+执行反馈、错误驱动修复、特征筛选（互信息/方差等）、多种集成策略（stacking/ bagging/ voting）、共识矩阵+谱聚类等。

**📊 数据集**

数据集覆盖20个公开表格数据，涵盖分类、回归和聚类任务，如FinBench、UCI、OpenML等。

**📈 对比分析**

与传统AutoML（AutoGluon、H2O、TPOT）、树模型（RandomForest、XGBoost、LightGBM）及LLM增强方法（DS-Agent、CAAFE）对比，SAGE-Loop在大多数任务上实现了最高或次高的指标，且完成率稳定。

**⚠️ 局限性**

限制是：依赖LLM算力与API成本，生成过程可能产生冗余或错误代码，且在极大规模或高维数据上的效率与可扩展性尚未充分验证。

---

## 63. VRL-Bench: Benchmarking agents on computer control tasks under finite trial budgets

**arXiv ID:** 2609.12404 | [PDF](https://arxiv.org/pdf/2609.12404v1)

**作者:** Yu Bai `[一作]`, Li Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个公平评估语言代理在有限试错预算下学习效果的框架，并在此框架下系统评估了多种基于失败记忆的更新策略；

**💡 创新点**

发现现有记忆更新方法并不总是优于简单重试，并提出了一种利用语言模型进行探索–利用调度的新策略，该策略在所有实验设置中均显著提升成功率与试错效率；

**🔧 技术方法**

使用语言模型进行失败记忆写入、回放调度和探索–利用决策，构建Semantic policy tree并分配剩余试错预算；模型包括DeepSeek‑V4‑Flash、GPT‑5.4 nano 与 GLM‑4.7‑FlashX；

**📊 数据集**

MiniWoB 与 WebShop 两个交互式基准；

**📈 对比分析**

在公平试错预算下采用 SR@6（成功率）与 AvgT@6（平均试错次数）进行对比；探索–利用调度器在所有六个模型+环境组合中均提升 SR@6 1.6–17% 并降低 AvgT@6，且在多数设置中表现优于现有记忆方法；

**⚠️ 局限性**

实验仅覆盖三种模型、六次试错预算、可见反馈，未考虑不可逆操作、隐藏反馈、不同重置语义及更大预算；调度器依赖语言模型的推理能力，对低资源模型可能效果有限。

---

## 64. Beyond Vector Similarity: Hierarchical Context-Aware Graph RAG vs Standard RAG in Enterprise Code Migration

**arXiv ID:** 2609.12464 | [PDF](https://arxiv.org/pdf/2609.12464v1)

**作者:** Nilesh Jaiswal `[一作]` (Google Cloud), Suddhasatwa Bhaumik `[通讯]` (Google Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于抽象语法树与图数据库的Hierarchical Context‑Resident Graph (HCRG) RAG 管线，用以改进企业代码迁移中的结构依赖和错误率。

**💡 创新点**

创新点在于将完整架构骨架序列化并缓存到 LLM 上下文，实现父类先行的拓扑翻译和跨文件依赖的多跳检索。

**🔧 技术方法**

使用 Tree‑Sitter 解析 AST、Google Cloud Spanner Property Graph、Vertex AI Context Cache、Gemini 2.5 Pro 以及向量检索相结合的混合检索。

**📊 数据集**

评估数据集为高耦合的 Spring PetClinic Java 代码库，并与标准 Vector RAG 进行对比。

**📈 对比分析**

通过自定义 7 项 SE 指标（依赖解析、继承一致、类型提示、编译通过、复杂度一致、注释保持等）进行量化比较；HCRG 在依赖、继承、类型提示等关键指标上提升约 30‑50%，但在环复杂度与注释保留上下降。

**⚠️ 局限性**

局限性包括全局结构导致 LLM 过度防御性编码、复杂度上升、注释丢失，以及对反射、泛型等动态特性仍需进一步完善。

---

## 65. AKTS: Sub-Microsecond Kernel Policy Switching for Language-Model Agents

**arXiv ID:** 2609.12276 | [PDF](https://arxiv.org/pdf/2609.12276v1)

**作者:** Mohammadali Khodabandehlou `[一作]` (University of Southern California), Mahdi Alizadeh `[通讯]` (University of Southern California)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AKTS机制，在GPU-backed LLM服务器上实现快速安全的调度策略切换，通过在内核加载时预编译并验证eBPF调度器库，运行时仅写入整数索引以触发尾调用，从而实现亚微秒级切换并兼顾低延迟与高吞吐。

**💡 创新点**

将策略可验证性与执行开销解耦：一次性在加载时完成eBPF编译验证，运行时只需写整数索引，避免了传统代码合成的编译/验证延迟，同时通过尾调用确保不产生未验证代码执行的风险。

**🔧 技术方法**

使用eBPF + sched_ext、程序数组与尾调用机制、Linux 6.14内核、Qwen2.5‑0.5B LLM做决策、vLLM 0.28.0负载模拟。

**📊 数据集**

在A100 GPU上对Qwen2.5‑0.5B进行推理，使用vLLM模拟的工作负载（steady→burst→steady），以及八个CPU‑bound对手。

**📈 对比分析**

对比了三种机制：基线A（标量写）、基线B（重新编译/验证/加载）与AKTS；AKTS在策略切换上达到920 ns（p50），与标量写相当；在vLLM实验中，AKTS在保持97 %吞吐的同时，Burst时延仅比静态latency策略差1.5 %以内。

**⚠️ 局限性**

缺点包括：决策模型在多种提示/规模下存在约33 %无效索引；仅做了oracle‑驱动的切换实验，未验证实时检测与决策的有效性；实验仅在单台Linux 6.14主机上，未评估更大策略库或不同burst形状；依赖sched_ext尾调用支持，可能在其他内核版本不可用。

---

## 66. An Open-Source End-to-End FHE Implementation for Privacy-Preserving Llama 3 8B Inference

**arXiv ID:** 2609.12378 | [PDF](https://arxiv.org/pdf/2609.12378v1)

**作者:** Yuhang Fan `[一作]` (Shandong University), Zhuoran Ji `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一种基于CKKS全同态加密的Llama-3 8B模型推理系统Odin，支持单GPU预填充模式下的全私有推理。

**💡 创新点**

创新点包括：①设计了特征主导的Odin打包方案，统一残差和跨层接口，并在注意力模块中使用C‑Δ布局避免中间重打包；②提出了模型级联合误差分配方法，结合输入区间校准和最小极大多项式逼近，显著降低非线性算子多项式度与乘法深度；③将上述方案集成为完整的GPU加速CKKS执行管线，首次公开了Llama-3全同态实现。

**🔧 技术方法**

主要技术包括CKKS同态加密、特征主导打包、C‑Δ注意力布局、SwiGLU、RMSNorm、Softmax双重迭代、多项式逼近（Remez算法）、误差注入与联合分配、Bootstrapping调度、GPU并行矩阵乘法。

**📊 数据集**

使用Meta Llama‑3‑8B权重，输入长度128（token），对WikiText‑2、C4等通用语料进行校准和评估。

**📈 对比分析**

与THOR、MOAI等基线相比，Odin在同一CKKS参数和硬件（NVIDIA H100 80GB）下，单层线性计算从41.37 s降至5.32 s，整体32层推理从1651.9 s降至366.4 s，获得4.51×速度提升；单层非线性阶段亦显著加速，整体性能与模型质量（PPL≈12.79）保持一致。

**⚠️ 局限性**

局限性包括：①仅针对Llama-3‑8B及128-token预填充，缺乏更大输入长度或多样化模型的实验；②对Bootstrapping的误差和深度调度尚未自适应优化；③系统对硬件资源（如GPU内存）要求仍高，峰值使用58.9 GiB；④在极大批次或在线生成场景下的扩展性尚未验证。

---

## 67. Mined from Scientific Literature: Process Schemas for Atomic Layer Deposition and Etching in Materials Science

**arXiv ID:** 2609.12139 | [PDF](https://arxiv.org/pdf/2609.12139v1)

**作者:** Sameer Sadruddin `[一作]` (TIB Leibniz Information Centre for Science and Technology), Jennifer D'Souza `[通讯]` (TIB Leibniz Information Centre for Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出并公开了四个面向ALD与ALE实验与模拟的JSON Schema，并通过schema-miner进行专家审阅与QUDT语义归一化，使过程记录具备机器可验证性。

**💡 创新点**

创新点在于将实验与模拟两种视角分别拆分、统一在同一语义框架下，并实现对数量量纲的QUDT化，提供可直接用于知识图谱与文献抽取的结构化规范。

**🔧 技术方法**

使用的技术包括schema-miner、schema-miner^pro、JSON Schema、QUDT ontology、Open Research Knowledge Graph (ORKG) 模板及大语言模型辅助抽取。

**📊 数据集**

数据集为由领域专家挑选的高质量文献语料库，共计59篇ALD实验、51篇ALD模拟、38篇ALE实验、22篇ALE模拟（以及AtomicLimits数据库的记录用于验证）。

**📈 对比分析**

通过比较四个Schema的属性数量、嵌套深度、QUDT归一化比例等指标进行评估，结果显示实验Schema拥有最多属性并实现约30%数量量纲归一化，而模拟Schema则在预测结果与表面机理方面更丰富；验证阶段显示抽取出的记录满足约95%验证规则，错误率低。

**⚠️ 局限性**

限制包括目前仅覆盖量化属性的语义归一化，非量化概念尚未与PMDco等材料本体对齐；对极端或少见的实验配置缺乏足够覆盖；需要人工专家持续维护。

---

## 68. Repair Before Reinforce: Context-Augmented Knowledge Graph Reasoning for Multi-Hop Question Answering

**arXiv ID:** 2609.12230 | [PDF](https://arxiv.org/pdf/2609.12230v1)

**作者:** Tharaka D. Fonseka `[一作]` (Princeton University), Niraj K. Jha `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于上下文增强的知识图谱(KG)监督框架，通过将同一文本块中与主三元组相关的支持三元组拼接成上下文图(CG)，并在此基础上对Qwen3-14B模型进行多跳问答的监督式微调、LLM判定的历史感知自适应修复以及强化学习(RL)训练，从而显著提升多跳推理性能。

**💡 创新点**

创新点包括①将支持三元组作为局部上下文构建上下文图，增强单一三元组的语义背景；②设计LLM-判定的自适应修复流程，可区分噪声三元组、错误问题和真正缺失知识，并按错误历史分级进行有针对性的修复；③在修复后以GRPO强化学习进行多跳训练，并证明修复对RL的提升是显著且稳定的。

**🔧 技术方法**

技术手段涵盖GraphMERT KG抽取、Qwen3-14B LoRA微调、LLM判定（Qwen系列）进行错误诊断与修复、基于路径对齐的GRPO强化学习、奖励设计结合答案正确性与路径覆盖、以及多阶段QA生成与过滤。

**📊 数据集**

实验数据集为Gastroparesis与Diabetes两类医学KG，分别从PubMed Central文本中使用GraphMERT抽取约6k三元组；基于这些KG生成1‑5跳多路多选题，1跳用于全覆盖SFT与修复，1/2跳混合用于RL，3‑5跳用于评估。

**📈 对比分析**

通过在同一基准上比较KGModel vs CGModel、修复前后、RL初始化的差异，发现CGModel在多跳上比KGModel高约2‑3个百分点；修复后1跳验证准确率达到100%；RL从修复模型初始化比从未修复模型提升约1‑3个百分点，最终CGModel‑Repaired‑RL在3‑5跳上达到最高准确率（约95%以上），且对选项打乱的鲁棒性变化不到1个百分点。

**⚠️ 局限性**

局限性包括：生成的QA为合成数据，仍可能存在噪声；LLM判定在错误诊断中可能带来模型偏差；自适应修复依赖多轮LLM判断与生成；仅在两种疾病KG上验证，泛化能力待进一步评估；缺乏专家人工审核的多路评测。

---

## 69. FoldNet++: a Large-Scale Synthetic Dataset for Robotic T-Shirt Folding and Unfolding

**arXiv ID:** 2609.12433 | [PDF](https://arxiv.org/pdf/2609.12433v1)

**作者:** Yuxing Chen `[一作]` (CFCS, Peking University), He Wang `[通讯]` (CFCS, Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

创建并公开了 FoldNet++ 数据集，用于机器人对任意皱折 T 恤的展开和折叠，并验证其在仿真到真实世界的迁移效果。

**💡 创新点**

①提供了覆盖 6 种机器人平台、1K 条 T 恤、1K 环境纹理和 120K 训练序列的跨体型大规模数据集；②首次在仿真中实现从任意高度皱折状态到展开再折叠的完整流程；③通过规则驱动演示生成实现高质量多任务标注。

**🔧 技术方法**

使用 FEM 物理仿真引擎 Style3D 进行高保真布料动力学；基于规则的演示生成；三类模型（VA、VLA、WAM）与 Diffusion Policy 进行对照实验；跨体型预训练 + 细粒度微调。

**📊 数据集**

FoldNet++ 数据集（1K T 恤、1K HDRI、120K 训练 episode 及 6 种机器人），以及 PolyHaven 纹理集、HDRI、1K 远程操作轨迹用于 fine‑tune。

**📈 对比分析**

在仿真与真实环境中进行零样本与少样本测试，跨体型预训练+微调可达 90%+ 的任务成功率；与 VLA、WAM、VA、Diffusion Policy 等模型对照，FoldNet++ 训练出的模型在两种环境均保持高性能；少量真实数据微调可进一步提升至 90%+。

**⚠️ 局限性**

仅针对 T 恤单一折叠风格，缺乏袖子、拉链等细节操作；未来需扩展到更复杂衣物类型与多步折叠流程。

---

## 70. Population-level measures of perceived food access reveal barriers beyond geographic proximity

**arXiv ID:** 2609.12132 | [PDF](https://arxiv.org/pdf/2609.12132v1)

**作者:** Teresa Groton `[一作]` (North Carolina State University), Benjamin rachunok `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用谷歌地图线上杂货店评论构建机器学习框架，自动提取并量化五个维度的感知食品可及性（可用性、可达性、可负担性、适配性与可接受性），从而实现大规模人群层面的可及性评估。

**💡 创新点**

创新点在于：①首次将非结构化评论文本与五维食品可及性框架相结合；②使用无监督主题建模+零射分类，实现自动主题到维度的映射；③采用贝叶斯部分池化校正店铺样本量差异，标准化得分以便跨店比较；④将店铺得分映射到人口普查区，揭示感知可及性与地理可及性及社会经济变量的系统关系。

**🔧 技术方法**

技术包括：Python + Sentence‑Transformers 文本嵌入；BERTopic 主题建模（UMAP + HDBSCAN）；零射匹配（cosine 相似度+softmax 归一化）将主题映射到五维；贝叶斯层级模型（PyMC）实现部分池化；距离衰减空间交互模型将店铺得分转化为区域暴露得分；统计回归（加权最小二乘）分析社会经济与可及性关系。

**📊 数据集**

数据集：Raleigh, NC 区域 49 家全国知名杂货店的 Google Maps 评论（共 25,125 条，2010‑2025 年），以及 2022 年 ACS 5 年估计的区块层人口、收入和黑人比例。

**📈 对比分析**

与传统基于距离/密度的地理可及性指标比较：感知可及性显著缩小“机会集”，高阈值下最远可及商店可达 7.5 英里；同时在五维得分间的 Pearson 相关系数低于 0.40，表明各维度互不高度相关。手工评估显示主题到维度的自动匹配准确率 85.4%，ROC‑AUC 0.915（宏平均）/0.861（权重平均）。

**⚠️ 局限性**

限制包括：①评论者自选样本，可能不代表全部居民；②得分相对标准化，仅能比较同一市场内店铺；③横断面设计无法区分居民感知与商店定位决策的因果关系；④仅使用 Google Maps 数据，可能忽略其他购物渠道的感知差异。

---

## 71. Confidence-Gated Transductive Test Generation for Code Reranking

**arXiv ID:** 2609.12489 | [PDF](https://arxiv.org/pdf/2609.12489v1)

**作者:** Sungjae Lee `[一作]` (POSTECH), Jungseul Ok `[通讯]` (POSTECH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为Confidence-Gated Transductive Test Generation (CoTT)的方法，用于合成高质量的测试用例，以评估和排名由大型语言模型生成的程序。

**💡 创新点**

创新点在于采用自适应设计，仅在归纳信心低时调用转导生成，从而提高输出的可靠性，并在需要时才分配额外的计算资源。

**🔧 技术方法**

使用了一种高效的归纳程序和转导程序的结合，归纳程序用于预测期望输出，转导程序用于特定输入的程序生成。

**📊 数据集**

在HumanEval-R+和MBPP-R+数据集上进行评估，这些数据集是从HumanEval+和MBPP+派生的代码排名基准。

**📈 对比分析**

与代表性的转导和归纳基线方法进行比较，CoTT在多个评估指标上超越了之前的基线，同时降低了相对成本，提供了良好的效率-效果权衡。

**⚠️ 局限性**

限制在于CoTT可能会受益于外部信息源（如相关代码文档），并且未探索与代码生成的集成以开发共同演化的代码和测试生成机制。

---

## 72. Rank-Efficient LoRA via Joint Tangent-Space Optimization under Isotropic Curvature

**arXiv ID:** 2609.12123 | [PDF](https://arxiv.org/pdf/2609.12123v1)

**作者:** Zihan Zhu `[一作]` (University of Pennsylvania), Weijie Su `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的 LoRA 优化器 Iso-LoRA，专门通过在 LoRA 的切线空间做谱均衡梯度下降来提升低秩适配器的有效秩利用率，从而实现更强的参数高效微调。

**💡 创新点**

创新点在于将 LoRA 的秩视作适配器容量的度量，并揭示优化器对更新能量分布的决定作用；提出了在 LoRA 切线空间投影后进行负极化（polar）梯度下降的几何优化方案，并给出了理论证明能显著提高有效秩；同时实现了高效的分块（factored）实现，保持 LoRA 参数化不变。

**🔧 技术方法**

使用的技术包括：低秩适配器（LoRA）框架、切线空间投影与极化（polar）变换、谱范数正则化、矩阵重构（重投影）以及分块低秩优化实现；理论分析基于高斯随机投影、spiked-gradient 模型、稳定秩（stable rank）与熵秩（entropy rank）的统计性质。

**📊 数据集**

实验使用的主要数据集包括：GPT‑2 Small + E2E NLG、Qwen2.5、Llama‑3.2、LLaMA‑2‑7B 的 MetaMathQA、GSM8K、以及 LLaMA‑2‑7B 的多任务基准；在各模型上分别评估 BLEU、NIST、METEOR、ROUGE‑L、CIDEr、EM% 等指标。

**📈 对比分析**

对比方法包括 AdamW、Muon、以及多种 PEFT 基线（LoRA‑GA、LoRA‑Pro、StelLA、Stiefel‑LoRA、LoRA‑RITE 等）。实验结果显示：在中到大秩（r≥16）时，Iso‑LoRA 在验证损失、BLEU、NIST、EM% 等指标上均优于 AdamW 与 Muon，且在 LLaMA‑2‑7B 上的 EM% 达到 61.87%，高于最近的 LoRA‑Pro（59.20%）。在固定 wall‑clock 时间下，Iso‑LoRA 也能获得最高的下游性能。

**⚠️ 局限性**

局限性：理论推导基于理想化的 spiked‑gradient 与 Gaussian 初始化模型，未覆盖完整训练过程和自适应优化器；动态稳定秩分析仅为机制结果，未完全描述长期训练动态；实现中分块算法有额外的 O((m+d)r²+r³) 计算开销，可能在极大秩或低延迟场景下影响效率；实验仅涵盖有限的模型与任务，需进一步验证在更广泛设置下的通用性。

---

## 73. One Skill Does Not Fit All: Automatic Discovery and Taxonomy-Guided Routing of Frame-Selection Skills for Long-Video Question Answering

**arXiv ID:** 2609.12517 | [PDF](https://arxiv.org/pdf/2609.12517v1)

**作者:** Jian Hu `[一作]` (Queen Mary University of London), Shaogang Gong `[通讯]` (Queen Mary University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AutoSkill，利用 LLM 代理在源数据上自动发现多种可执行帧选择技能，并通过语义分类路由将每个问题分配到最合适的技能，以实现自适应证据获取。

**💡 创新点**

创新点在于：①用 LLM 自动探索并生成多样化的帧选择技能；②仅利用目标问题与选项文本（无目标视频或答案）构建语义分类词典并进行技能路由；③在保持单一次视频 MLLM 推理的前提下实现多技能协同，提升长视频问答性能。

**🔧 技术方法**

技术包括：LLM 代理生成与评估技能、语义分类引导的路由表、源监督的技能评估、目标无监督的查询语义迁移、冻结的多模态大型语言模型（Qwen2.5‑VL‑7B、Qwen3.5‑4B）和多帧选择策略。

**📊 数据集**

使用了 3000 条源数据（VideoVista、ALLVB、LongVideo‑Reason），以及四个长视频 QA 公开基准的五个拆分（MLVU、LongVideoBench、Video‑MME、LVBench）共 1500 条未标注问题与选项。

**📈 对比分析**

与统一采样、固定采样策略以及现有训练‑free 采样方法对比，AutoSkill 在 128 帧预算下平均提升 2.4%（Qwen2.5‑VL‑7B）/1.2%（Qwen3.5‑4B）以上；相较于需要 RL 训练或多次推理的方案，AutoSkill 仅需一次推理且训练成本约 11 倍更低。

**⚠️ 局限性**

局限性包括：依赖源标签数据和 LLM 代理的探索效率；技能集合仍有限，可能无法覆盖所有证据获取模式；仅处理帧选择而未改进 MLLM 的推理能力；在极端视频时长或复杂语义场景下，路由表的精度和泛化性可能受限。

---

## 74. Plans They Abandon, Reports They Author: The Narrative Layer of Autonomous Agents

**arXiv ID:** 2609.12205 | [PDF](https://arxiv.org/pdf/2609.12205v1)

**作者:** Obada Kraishan `[一作]` (Texas Tech University), Kulsawasd Jitkajornwanich `[通讯]` (Texas Tech University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了编程代理在完成任务后自动生成的自我报告，比较计划、日志和报告三条记录，量化了报告的遗漏率、可恢复性和相对于计划的修复倾向；

**💡 创新点**

首次在大规模真实开发者会话中系统评估代理自我叙述的准确性，并提出“修复指数”来度量报告向计划倾斜的程度；

**🔧 技术方法**

使用语言模型进行计划和声明抽取、裁决匹配；规则+模型裁决的对齐算法；基于最长公共子序列的距离与相似度度量；统计回归分析（聚类标准误、Spearman相关、Cliff’s δ）

**📊 数据集**

SWE‑chat公开数据集：5851个真实开发者会话、355,942个工具调用、20,641条声明、45,206条计划单元；

**📈 对比分析**

通过LCS和包含度量计算遗漏（平均0.906）、可恢复性（F1≈0.20）、修复指数（均值0.105）和计划-执行偏差（均值0.357）。计划提取的F1为0.814，修复指数与偏差呈正相关（ρ≈0.55，p<0.001）。

**⚠️ 局限性**

局限性包括：裁决模型与人工标注的一致性低（κ=0.185），仅覆盖编程代理且多数为单一框架，计划提取仅结合结构化与自然语言两种形式且相互一致度低，修复指数分辨率受报告短小限制，研究仅评估信息量而未检验实际使用效果。

---

## 75. When Successful Knowledge Graph Edits Displace Correct Answers: Rank-Level Locality beyond Parameter Support

**arXiv ID:** 2609.12116 | [PDF](https://arxiv.org/pdf/2609.12116v1)

**作者:** Yi-Cheng Lai `[一作]` (Academia Sinica), Hen-Hsen Huang `[通讯]` (Academia Sinica)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了知识图谱嵌入模型在对单个低排位事实进行编辑时，如何导致正确答案被位移，并提出了三种局部性评估方法：参数支持、同一查询和关系范围；

**💡 创新点**

创新点在于统一的rank‑displacement审计框架，推导出精确得分保持的几何/维度条件，并比较多种编辑策略在不同局部性下的成功率与破坏程度，同时对已学习编辑器（KGEditor）进行了案例验证；

**🔧 技术方法**

使用了线性与非线性KGE评分函数（DistMult、ComplEx、RESCAL、RotatE、TransE），最小范数更新、正交投影、限制范数、岭回归、梯度下降以及闭式线性最小二乘解，并结合参数空间维数和正交子空间分析；

**📊 数据集**

主要实验数据集为FB15k‑237的预训练KGE模型，此外还评估了WN18RR、Rescal、RotatE、TransE等模型及不同维度，并用KGEditor公开的3,086个案例集做对比；

**📈 对比分析**

通过Top‑10 raw排名的S@10、无损伤率、SafeSucc、平均/第95百分位/最大位移等指标，对比直接提升、精确保持、关系截断保持、支持正则化实体编辑、闭式/梯度降解等路线。结果显示：直接提升成功率最高但位移最严重；精确保持几乎无损伤但成功率极低；关系截断保持在成功率与无损伤之间取得最佳平衡；

**⚠️ 局限性**

局限性包括：仅考虑单步编辑的低排位事实，未处理错误事实修正、未知实体或编辑序列；精确保持仅适用于线性参数化；观测阻断策略依赖已知标签，部署时需事先保护与阻断选择；实验集中于FB15k‑237，结果对其他图谱或更高维度的泛化性有限。

---

## 76. When Ground-Truth Fidelity Matters: An Orchestrated UAS Framework for Wheat Streak Mosaic Virus Detection Using Vision Transformers and Machine Learning

**arXiv ID:** 2609.12169 | [PDF](https://arxiv.org/pdf/2609.12169v1)

**作者:** Dewi Endah Kharismawati `[一作]` (Ohio State University), Sami Khanal `[通讯]` (Ohio State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在甜玉米田间利用无人机多光谱影像构建植株级WSMV病害检测管线，并通过Vision Transformer实现分类。

**💡 创新点**

创新点在于引入七通道（5光谱+NDVI+NDRE）输入与ViT结合，并系统评估标签可信度对模型性能的影响。

**🔧 技术方法**

采用多光谱无人机影像、七通道栈、Vision Transformer、经典机器学习（SVM、XGBoost）以及自动化UAS流水线。

**📊 数据集**

数据集来源于2025年奥亥斯沃特实验站的六块随机区块，包含24种甜玉米品系、5次飞行、约6,500个植株级样本及约600个ELISA标注样本。

**📈 对比分析**

通过对照处理标签、行级症状和ELISA真实标签三种评估，发现处理标签下ViT准确率89%，但行/植株级ELISA下准确率仅约50%，说明标签噪声导致性能偏高。

**⚠️ 局限性**

主要限制在于ELISA确认样本稀缺、WSMV症状不均匀且仅在少数品系显现，导致标签质量低且模型难以泛化。

---

## 77. Mission Performance: Automatic and Adaptive Race Pace Progression for Autonomous Racing

**arXiv ID:** 2609.12292 | [PDF](https://arxiv.org/pdf/2609.12292v1)

**作者:** Giovanni Lambertini `[一作]` (University of Modena and Reggio Emilia), Marko Bertogna `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并部署了Mission Performance模块，实现自动化的纵向、横向和综合性能递进，实时根据安全与车辆动力学指标在赛道段上动态调整赛车性能；

**💡 创新点**

引入基于温度查找表和多指标安全阈值的分段性能递进机制，采用目标缩放模式替代模型更新，构建安全层级约束的动态性能更新逻辑，能够在多车竞争中自动应对滑移、IMU异常和抢跑等非标准情形；

**🔧 技术方法**

结合Pacejka魔术公式摩擦模型、Ceres Solver非线性最小二乘估计、基于速度曲线的前后轴摩擦椭圆约束、优化式运动规划与控制（MPPC）、实时安全指标滤波与阈值计数器，使用温度映射LUT实现动态限制；

**📊 数据集**

主要使用A2RL Season 2 Yas Marina Circuit现场赛数据，包括车速、温度、加速度、转向角、TC/ABS激活等传感器记录，未使用公开数据集；

**📈 对比分析**

通过与赛道竞速中的手动调节基准以及不同性能进阶策略对比，展示了在11圈比赛中实现与基线相同或更低时间的同时安全阈值被保持；在过弯场景下表现出安全快速的性能降级，整体性能提升约1%至2%；

**⚠️ 局限性**

受限于阈值与递进步调的手工调优，对极端异常的过度敏感导致多次无谓性能降低；当前未考虑多车交互导致的避让误判，且温度LUT仅作为安全钳位，缺乏主动探索功能，无法完全自动化所有热启动与适应过程。

---

## 78. Not All Speech Is Intent: Adaptive Self-Correcting Inference Layer for Post-ASR False Wake-Up

**arXiv ID:** 2609.12469 | [PDF](https://arxiv.org/pdf/2609.12469v1)

**作者:** Preeti Saraswat `[一作]` (Samsung Research America), Anil Yadav `[通讯]` (Samsung Research America)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于反馈驱动的 ASCIL 后-ASR 校正层，利用用户的隐式与显式行为反馈在设备上持续学习错误模式，并在推理时纠正唤醒意图。

**💡 创新点**

创新点：① 将用户行为视为噪声监督，在后-ASR 阶段实时更新错误模式；② 在不修改基础 ASR/NLU 模型的前提下，使用抽象特征存储和 LLM 作为纠正函数；③ 低延迟 (<60 ms) 的在线持续学习。

**🔧 技术方法**

技术手段：多模态特征融合、行为反馈识别与模式提取、模式存储、LLM（Qwen3‑14B‑Instruct）作为纠正函数，以及基于 CAR/UICR/FIR/FAR/P_error 的离线评估。

**📊 数据集**

使用了 3,667 条来自商业语音助手的交互记录，包含 2,358 次意图唤醒和 1,309 次非意图唤醒，并手工标注 14 种音频/上下文条件。

**📈 对比分析**

通过与 prompt‑only 基线分类器对比，评估 CAR/UICR/FIR/FAR/P_error；在 0.90 阈值下，ASCIL 在会话分离的失败样本上实现 54.27% 相对错误率下降；在多条件意图唤醒切片上提升 24.39% 相对错误率；在干净音频上在 0.90‑0.95 阈值下降低错误率 20–15%。

**⚠️ 局限性**

局限性：需要足够用户交互历史，冷启动适应困难；反馈信号不确定性未单独验证；实验使用会话分离而非用户分离，无法评估跨用户泛化。

---

## 79. On-Device Language Models for Privacy-Preserving Stress Prediction: A Multimodal Evaluation on Mobile Health

**arXiv ID:** 2609.11961 | [PDF](https://arxiv.org/pdf/2609.11961v1)

**作者:** Ibukunoluwa Soyebo `[一作]` (University of Southern California), Corey E. Baker `[通讯]` (University of Southern California)

**通讯引用:** 177 | [OpenAlex ID](https://openalex.org/A5049845873)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在移动健康场景下，使用零样本的在设备语言模型（ODLM）对多模态（客观传感器数据与主观自评）压力水平进行实时预测，且不依赖云端推理，保护用户隐私。

**💡 创新点**

首次系统评估ODLM在多模态压力预测中的可行性，比较不同模型规模、量化方式、提示策略（自然语言序列 vs 统计摘要）及模态融合对预测准确率、延迟与吞吐量的影响。

**🔧 技术方法**

零样本提示（zero‑shot prompting）、量化与裁剪的轻量化ODLM（Qwen3‑0.6B、Qwen3‑1.7B、Granite‑3.3‑2b）、自然语言字符串（NLS）与统计摘要（SS）提示技术，以及在iOS/iPadOS设备上进行端到端延迟与吞吐量测评。

**📊 数据集**

PMData数据集——包含16名参与者的5个月生活日志，记录步数、卡路里、心率、睡眠等客观传感器数据以及睡眠质量、疲劳、情绪等主观自评。

**📈 对比分析**

与传统机器学习基线（决策树、随机森林、SVM等）对比，ODLM在预测MAE上与最佳监督模型相差不大（SVM/ RF MAE≈0.50，Qwen3‑0.6B MAE≈0.51），同时提供亚秒级延迟（0.37–0.47 s）和高吞吐量（78–82 tokens/s）。

**⚠️ 局限性**

局限性包括：仅采用零样本提示，未进行领域适配或微调；实验仅基于单一数据集，可能缺乏跨人群/设备的泛化性；未测量能耗；设备性能差异未细化。

---

## 80. TripPattern: A Pattern-based Text Watermarking Method for Large Language Models

**arXiv ID:** 2609.12472 | [PDF](https://arxiv.org/pdf/2609.12472v1)

**作者:** Sangjun Moon `[一作]` (Chungnam National University), Manabu Okumura `[通讯]` (Institute of Science Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种三组词表划分的水印嵌入框架TripPattern，利用两组模式词交替选择并加入中性词组以提升自然性。

**💡 创新点**

创新点在于将词表拆分为中性组与两组模式词，并通过子词长度动态构建中性组，使得水印在保持可检测性的同时极大提升生成文本质量。

**🔧 技术方法**

使用关键哈希+PRNG实现词表划分，采用软偏置对模式词日志进行加权，检测时基于交替模式统计做显著性检验。

**📊 数据集**

在四种语言（英语、韩语、德语、西班牙语）的C4/mC4、GSM8K/MGSM以及CNN/DM、GovReport数据集上进行实验。

**📈 对比分析**

与KGW、Unigram、UPV、SWEET等基线相比，TripPattern在多语言文本质量（PPL、ROUGE、BERTScore）几乎不下降，同时在AUC、TPR@5等检测指标上保持竞争或更优；在推理任务上保持准确率不变。

**⚠️ 局限性**

局限包括对强语义改写攻击（如Dipper）易被破坏；检测性能受水印强度δ与中性组比例γ调节，需要在不同模型、语言和域中寻找最佳设置；在多语言高置信度模型中水印易被稀释，鲁棒性下降。

---

## 81. Unlabeled Echoes: Pseudo-Labels and Genus-Aware Smoothing for Bat Call Recognition

**arXiv ID:** 2609.11986 | [PDF](https://arxiv.org/pdf/2609.11986v1)

**作者:** Frank Fundel `[一作]` (Ludwig-Maximilians-Universität München), Alexandra Howard `[通讯]` (University of Free State)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究如何利用伪标签和属级平滑提高蝙蝠回声识别的半监督学习效果，并验证其跨地区迁移性。

**💡 创新点**

① 证明单一伪标签策略在中等规模生态音频中最优，恢复至61.5%监督缺口；② 引入属级平滑将不确定概率分配给同属物种，实现无额外注释的生物结构化目标。

**🔧 技术方法**

采用BioAcoustic Transformer (BAT) 变压器骨干；伪标签在线生成与自训练；BYOL、Noisy Student、FixMatch等对比；属级平滑与均匀平滑组合。

**📊 数据集**

欧洲Skiba 18种蝙蝠语料（90%标签遮盖）和南非UFS 9种蝙蝠+噪声类，30k无标签场景音频。

**📈 对比分析**

在10%标签条件下与全监督、自动编码器、BYOL、Noisy Student、FixMatch等方法比较，伪标签在所有指标上领先：混合-F1提升6.38点、准确率5.42点、物种F1 10.98点；在UFS迁移中准确率提升10.69点，物种宏F1 4.96点；属级平滑+均匀平滑进一步提升物种宏F1 4.73点。

**⚠️ 局限性**

仅使用固定的compact BAT骨干和闭合输出词汇；对未知物种仍需专家复核；实验覆盖欧洲和南非两地区，需进一步验证更大规模、多地区迁移效果。

---

## 82. NDT Factory: Synthesizing Verified Network Digital Twins from Semantic Models via Multi-Agent LLM

**arXiv ID:** 2609.12170 | [PDF](https://arxiv.org/pdf/2609.12170v1)

**作者:** Sudipta Acharya `[一作]` (University of Ottawa), Burak Kantarci `[通讯]` (University of Ottawa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了NDT工厂，一个多代理系统，利用大型语言模型根据语义模型在需求时动态合成可执行的网络数字孪生（NDT），实现对网络服务意图（NSI）的自动评估和决策。

**💡 创新点**

创新点包括：1）将语义模型作为 NDT 生成的规范；2）通过多代理（Planner、Coding、Intent handler 等）实现并行 LLM 代码合成；3）编译-测试-调试循环确保生成代码的可验证性；4）在 Call Admission Control (CAC) 案例中完整实现并验证了从语义模型到可执行管道的端到端流程。

**🔧 技术方法**

使用技术：大型语言模型（Claude Sonnet 4.6，Qwen3-Coder），Go 语言编译器，YAML 语义模型，Python 离线测试脚本，分布式多代理编程框架。

**📊 数据集**

数据集与实验环境：采用 NSFNet 拓扑（14 节点，21 条链路）和通过 Poisson 过程生成的 300 条 NSI（带宽 50‑200 Mbps，平均服务时长 33 s，延迟 30‑100 ms），以及手工实现的 Python CAC NDT 作为基准。

**📈 对比分析**

比较方法：与参考 Python CAC NDT 进行决策一致率、接收率、拒绝原因、链路利用率误差等指标对比。性能结果为：100% 编译通过，99.3% 决策一致率，90% 接收率，链路利用率误差均值 0.7%，最大误差 24.4%，证明生成的 NDT 与手工实现高度一致。

**⚠️ 局限性**

局限性：1）对 LLM 的质量高度依赖，Qwen3 的可靠性较低；2）仅在小规模、静态网络场景验证，未覆盖动态流量或更大拓扑；3）安全性仅通过编译与预设测试检测，未实现正式形式化验证；4）生成代码可能包含未检测的漏洞；5）依赖语义模型的准确性与及时更新，缓存失效策略需要手工维护。

---

## 83. ESTS at WMT26: Routing-Informed Expert Pruning for Model Compression

**arXiv ID:** 2609.12310 | [PDF](https://arxiv.org/pdf/2609.12310v1)

**作者:** Liu O. Martin `[一作]` (University of California, Los Angeles), Nanyun Peng `[通讯]` (University of California, Los Angeles)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们针对WMT26模型压缩共享任务，采用专家裁剪、恢复微调与MXFP4量化，构建了六个英文到简体中文和英文到埃及阿拉伯语的压缩模型。

**💡 创新点**

创新点在于结合路由行为和跨语言路由差异进行动态专家容量分配，并在裁剪后通过合成数据恢复，进一步实现多种压缩操作点。

**🔧 技术方法**

使用了Mixture-of-Experts（MoE）路由分析、GPT‑OSS‑20B预训练、MXFP4量化、LLaMA‑Factory恢复SFT以及vLLM推理框架。

**📊 数据集**

数据集包括从WMT News Crawl 2025、Webis‑TLDR‑17、Bluesky社交媒体、SPoRC语音转写生成的44k英文源样本，以及使用GPT‑5.1合成的翻译对。

**📈 对比分析**

通过内部xCOMET‑XL（基于GPT‑5.1伪参考）评估六个压缩模型，内部得分从ZHO的0.4130下降到0.3840、ARZ从0.2705升到0.2647，再到0.2380，表明不同压缩级别对质量的影响可被量化。

**⚠️ 局限性**

局限在于缺乏官方评测结果、压缩不一定带来解码速度提升、回退策略增加推理成本，以及合成数据与真实任务分布可能存在偏差。

---

## 84. Retrieval-Augmented Generation for Scientific Code Understanding

**arXiv ID:** 2609.12190 | [PDF](https://arxiv.org/pdf/2609.12190v1)

**作者:** Aaron Nobile `[一作]` (ETH Zürich), Mohsen Sadr `[通讯]` (Paul Scherrer Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于离线前置检索、可本地部署的 RAG 科学代码助手，在 IPPL C++ 代码库上实现了问答功能。

**💡 创新点**

创新点在于：① 将繁重的代码解析、结构图构建、实体解释与嵌入等工作集中在离线阶段，减轻在线推理负担；② 设计了多级实体（文件、模块、调用链）和结构化检索策略；③ 通过生成式实体说明与元数据嵌入提升检索上下文质量，使小模型即可给出具象、基于源码的答案。

**🔧 技术方法**

使用技术包括：Tree‑sitter 解析、结构图与高阶实体构建、LLM 生成的实体解释（qwen2.5-coder、qwen3.5 等）、nomic-embed-text 作为嵌入模型、FAISS 向量检索、检索增广（精确文件/符号注入、结构扩展）、Meta‑ranker 与 LLMAgent 过滤、以及 RAG 推理的 prompt 模板。

**📊 数据集**

使用数据集为 IPPL（Independent Parallel Particle Layer）科学 C++ 框架，包含约 8,013 条嵌入条目；评估使用 100 条人工生成（GPT‑5.4）的问题，覆盖 11 类代码理解主题。

**📈 对比分析**

通过对 7 种小模型（Qwen2.5‑coder 7B/14B/32B、Qwen3.5‑9B/32B、Gemma4‑12B/31B）在 RAG pipeline 与 Claude Code（agentic）检索架构下进行对比。结果显示：在 RAG pipeline 中 9B Qwen3.5 获得最高平均得分 0.795±0.034，明显优于同框架下更大模型及 Claude Code 对比；表明检索质量与模型族群比参数规模更关键；平均推理延迟从 3.5s（Qwen2.5‑7B）到 41s（Gemma4‑31B）不等。

**⚠️ 局限性**

主要局限：① 离线预处理成本高，需要 GPU 节点；② 每次代码变更需重建向量库；③ 系统仅做问答，缺乏主动编辑、测试、执行等代理功能；④ 评估基于单一代码库、100 问题以及 LLM 判分，可能存在偏差与泛化能力受限。

---

## 85. Mindspeller Neuroprofiling. How task performance, EEG, and association evidence support O*NET-based role guidance

**arXiv ID:** 2609.12501 | [PDF](https://arxiv.org/pdf/2609.12501v1)

**作者:** Prem Aravindan Jeyakumar `[一作]`, Hannes De Wachter `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种结合自我报告、关联响应与任务-EEG数据的职业适配系统。

**💡 创新点**

创新点在于仅用任务-EEG流生成职业证据、5/35/60权重策略、严格的证据门控与角色置信度上限。

**🔧 技术方法**

使用四电极EEG、持续块获取、谱估计、规则化证据门控、O*NET能力映射、AI辅助摘要与自由文本评分。

**📊 数据集**

基于内部376个O*NET职业数据库和12项任务的自定义数据集。

**📈 对比分析**

通过内部回归测试验证流程完整性，尚无外部对照，性能尚待进一步验证。

**⚠️ 局限性**

局限在于缺乏外部效度、能力覆盖有限、只预测认知匹配且置信度仅限中等，未覆盖实践技能或雇用结果。

---

## 86. Exponential Lower Bounds for Integer-Weighted Shortest-Paths Preservers of DAGs

**arXiv ID:** 2609.12211 | [PDF](https://arxiv.org/pdf/2609.12211v1)

**作者:** Michael Yi Wang `[一作]` (University of Michigan), Nicole Wein `[通讯]` (University of Michigan)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究图简化问题，证明 DAG 中整数权重的最短路径保留器需要指数级权重

**💡 创新点**

给出先前未解的开放问题的负面答案，并展示指数下界与线性上界的精确边界

**🔧 技术方法**

通过从 3‑拓扑序保留器构造 3 层 DAG 的归约，以及构造可算的加法三元组来证明指数下界

**📊 数据集**

无实验数据集，所有结果均为理论证明

**📈 对比分析**

无对比实验，结果为理论上限与下界，展示了整数权重下的性能极限

**⚠️ 局限性**

仅适用于 DAG 的特殊三层结构（中层 3 或 2 个顶点），且下界仅对整数权重成立，实际算法应用仍需进一步研究

---

## 87. R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration

**arXiv ID:** 2609.11955 | [PDF](https://arxiv.org/pdf/2609.11955v1)

**作者:** Dhruv Dixit `[一作]` (Stevens Institute of Technology), Paritosh Pandey `[通讯]` (University of North Carolina)

**通讯引用:** 1809 | [OpenAlex ID](https://openalex.org/A5064713441)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套可插拔的检索-生成-验证-校准的事实核查流水线 R2VC，能够为每条声明生成标签、引用证据，并提供置信度估计与可选放弃预测。

**💡 创新点**

创新点在于：①将检索、生成、验证与置信度校准完全模块化，便于单独调优；②采用稀疏+稠密检索混合并融合评分；③使用 SFT 与 DPO 训练的生成器产生多样化结构化候选；④外部 NLI 交叉编码器对候选进行验证并筛选；⑤轻量级序列级校准器实现概率校准与有选择性放弃。

**🔧 技术方法**

技术包括：Hybrid retrieval（BM25+FAISS HNSW）、SFT+ DPO 微调、Cross‑Encoder NLI 验证、Sequence Likelihood Calibration (SLC) 以及 8B 规模的 LLM（如 Llama‑3.1‑Nemotron、Qwen3‑8B）。

**📊 数据集**

使用 VitaminC 作为监督训练集，FEVER 作为评估基准，检索语料为全量 Wikipedia。

**📈 对比分析**

在 8B LLM 上，R2VC 在 FEVER 上从 75.1% 提升到 84.7%（相对提升约 10%），在 VitaminC 上从 87.8% 提升到 99.8%；同时 Brier 及 ECE 等校准指标显著下降，表明置信度更可靠；与同类基线相比，R2VC 在准确率与置信度校准上均有显著优势。

**⚠️ 局限性**

主要局限：检索阶段错误（尤其是错误实体检索）仍是性能瓶颈；依赖外部 NLI 模型和多步推理，推理延迟较高；校准虽有效但对检索错误无法完全补偿；系统整体规模大，资源消耗较大。

---

## 88. Real-Time Music Source Separation on a Low-Power Audio DSP

**arXiv ID:** 2609.12201 | [PDF](https://arxiv.org/pdf/2609.12201v1)

**作者:** Jianan Li `[一作]`, Gabby Yi `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一种能在嵌入式音频 DSP（2 MB L2、2.07 GFLOPS）上实时运行的音乐源分离模型，能够在保持低延迟的同时达到 4.70 dB cSDR。

**💡 创新点**

通过识别现有模型在内存与计算上的瓶颈，提出连续上下文训练以避免块训练导致的衰减，并加入可调时延的门控复数 FIR 深度滤波器，实现了满足嵌入式预算的高效分离。

**🔧 技术方法**

使用 TFC‑TDF U‑Net 结构、GRU 递归、深度可分离卷积、门控复数 FIR 过滤器、STFT/iSTFT、连续上下文训练以及手动调度的浮点 DSP 运行时。

**📊 数据集**

采用 MUSDB18‑HQ 数据集进行训练与评估。

**📈 对比分析**

与公开的实时 MSS 基线（HS‑TasNet、RT‑STT 等）在 MUSDB18‑HQ 上以 cSDR 进行比较，模型在满足 11.6 ms 死线和 2 MB 内存限制的前提下，取得 4.70 dB cSDR（比无滤波基线提升 0.38 dB，接近 RT‑STT 的 5.17 dB）且每帧耗时仅 10.43 ms。

**⚠️ 局限性**

性能仍低于最先进方法（比 RT‑STT 差 0.47 dB）、仅适用于此类 DSP，未验证固定点精度、能耗与鲁棒性，且对极低延迟场景（如现场监控）仍需进一步改进。

---

## 89. Quantifying Consonant Contributions to Word Intelligibility via Acoustic Masking

**arXiv ID:** 2609.12122 | [PDF](https://arxiv.org/pdf/2609.12122v1)

**作者:** Eunjung Yeo `[一作]` (University of Texas at Austin), David Harwath `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用自动语音识别（ASR）与声学遮蔽技术，评估单个辅音被静音后对单词识别率的影响，得到辅音贡献度量 MMR。

**💡 创新点**

首次提出基于遮蔽误识率的可扩展辅音贡献评估方法，并验证其与词频及功能负荷的关系，体现了对传统 FITI 框架的补充与改进。

**🔧 技术方法**

结合强制对齐、声学遮蔽和三种主流 ASR 架构（MMS encoder‑only、Whisper encoder‑decoder、Qwen3‑ASR LLM‑based）实现自动化实验。

**📊 数据集**

使用 Common Voice（英语、西班牙语、德语、捷克语）和 TIMIT（英语）四个公开语料库，覆盖多语言多样语料。

**📈 对比分析**

通过与词频、功能负荷的（偏）Spearman 相关性分析对比，发现 MMR 与两因素均显著相关，且跨模型、跨语言表现一致，凸显方法的稳健性。

**⚠️ 局限性**

局限在于遮蔽采用完全静音，未能真实模拟口吃者的部分声学特征；未考虑辅音间交互与个体差异，且需进一步通过人类听者验证结果。

---

## 90. QTrans: A Quantum Transformer for Sentiment Classification

**arXiv ID:** 2609.12011 | [PDF](https://arxiv.org/pdf/2609.12011v1)

**作者:** Ren-Xin Zhao `[一作]` (Xiangtan University), Yaonan Wang `[通讯]` (Hunan University)

**通讯引用:** 23846 | [OpenAlex ID](https://openalex.org/A5025640070)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了QTrans，一个轻量级量子Transformer，利用参数化量子电路生成查询、键、值投影并通过高斯相似度计算注意力系数，同时引入位置级量子前馈网络，用于二分类情感分析。

**💡 创新点**

首次将独立量子电路生成的多头自注意力与量子前馈网络结合到轻量级Transformer框架，显著提升小数据集情感分类性能。

**🔧 技术方法**

使用参数化量子电路、Gaussian投影自注意力、量子前馈网络、残差连接、层归一化、混合量子-经典训练（DeepQuantum + PyTorch）等技术。

**📊 数据集**

在三大二分类情感数据集 MR（电影评论）、CR（客户评价）和 MPQA（意见表达）上进行实验。

**📈 对比分析**

与 Tiny Transformer、BiLSTM‑Attention、TextCNN 等轻量级经典模型在相同参数、词表、序列长度下对比，QTrans 在 MR、CR、MPQA 的测试准确率分别提升约 2.9%、3.2%、3.8%，宏 F1 亦保持领先。

**⚠️ 局限性**

实验仅在模拟器上完成，未考虑真实量子噪声；仅使用单一随机种子、有限数据量，缺乏对多种随机种子或更大语料的鲁棒性评估。

---

## 91. Pelican-Sim 1.0: A General World Model Simulator for Embodied Intelligence

**arXiv ID:** 2609.12036 | [PDF](https://arxiv.org/pdf/2609.12036v1)

**作者:** Shilong Zou `[一作]` (Beijing Innovation Center of Humanoid Robotics), Xiaozhu Ju `[通讯]` (Beijing Innovation Center of Humanoid Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种跨机器人形态的动作条件世界模型模拟器，能够根据初始RGB图像和统一的28维动作表示生成未来视觉轨迹，并支持下游的数据生成、策略评估、动作选择和策略改进。

**💡 创新点**

创新点包括：①统一的28维动作值与同步的URDF渲染动作视频双重条件接口，兼顾数值精度与图像空间运动指导；②在Video Diffusion Transformer中引入稀疏Mixture-of-Experts层，提升异构动力学建模能力；③采用因果少步蒸馏将35步推理压缩为4步，显著加速生成。

**🔧 技术方法**

使用技术包括Video Diffusion Transformer（DiT）、稀疏MoE层、动作值与动作视频双通道注入、因果少步蒸馏、以及微调的VLM评估器。

**📊 数据集**

训练数据约100万条轨迹，来源于AgiBotWorld Beta、RealSource World、RoboMIND、LIBERO、ManiSkill2、RoboTwin、RoboCasa等真实与仿真数据集。

**📈 对比分析**

与IRASim、Ctrl-World、EnerVerse-AC等基线相比，本文模型在PSNR、FVD、SSIM、LPIPS、FID等视频质量指标上均取得最高分，并在EWMBench多维度评估中获得最高总体得分；在数据生成、策略评估、动作选择与策略改进等下游任务上均表现出显著提升。

**⚠️ 局限性**

局限性包括：模型在某些EWMBench细分指标如HSD、nDTW、SceneC、Diversity等仍不及最佳基线；在极端长期动力学或复杂接触场景下的准确性尚待验证；对极少数据或完全新形态机器人仍可能出现泛化不足。

---

## 92. CueMem: Cue-Guided Context Reconstruction for Long-Term Conversational Memory

**arXiv ID:** 2609.12354 | [PDF](https://arxiv.org/pdf/2609.12354v1)

**作者:** Changjian Wang `[一作]` (Mashang Consumer Finance Co., Ltd.), Ning Jiang `[通讯]` (Mashang Consumer Finance Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于检索提示的长时对话记忆框架 CueMem，能够在回答查询时从对话历史中重构所需的上下文。

**💡 创新点**

核心创新在于将记忆拆解为细粒度的检索提示（如关系三元组）并与源对话轮次关联，再通过图扩展从提示回溯到完整上下文，而不是直接存储压缩摘要；此做法避免了信息丢失并提升检索精度。

**🔧 技术方法**

利用 LLM 进行关系三元组抽取、all-MiniLM-L6-v2 进行提示与查询编码、构建时间与语义双边图、图扩展重建上下文，并将重建的上下文输入 Llama‑3.3‑70B‑Instruct 生成答案。

**📊 数据集**

在 LoCoMo（10 篇长对话、1540 题）和 LongMemEval‑S（500 题、多会话、多跳、时间推理和知识更新）两大长对话问答基准上进行评估。

**📈 对比分析**

与 Naive RAG、MemoryOS、Mem0、A‑Mem、LightMem 等基线对比，CueMem 在 LoCoMo 上整体准确率提升 5.26%，在 LongMemEval 上提升 1.60%；同时输入 token 减少约 90% 以上，推理延迟降低 40–90% 左右。

**⚠️ 局限性**

局限性包括：仍依赖 LLM 的生成质量；对外部知识检索支持不足；提示抽取质量受 LLM 生成偏差影响；对极长历史或高更新频率场景的鲁棒性待进一步验证。

---

## 93. AnchorVLN: Geometry-Anchored Vision-Language Grounding Reasoning for Open-Vocabulary Navigation

**arXiv ID:** 2609.12285 | [PDF](https://arxiv.org/pdf/2609.12285v1)

**作者:** Long Giang Vu `[一作]`, Rajath Chandrashekar Aralikatti `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了AnchorVLN系统，利用多模态Vision‑Language Model（VLM）与几何模块分离，通过Embodied‑Nav‑MCP工具服务器实现VLM在推理时调用工具完成导航与物体定位，并在无地图的室内环境中执行指令跟随和对象引用任务。

**💡 创新点**

核心创新是将语义与几何严格分离，使用Model Context Protocol（MCP）只允许文字短语和句柄跨越客户端-服务器边界，禁止模型直接输出坐标；采用LiDAR深度测距提供可靠几何；实现可插拔控制器模型，使系统可在不同机器人/任务上无缝迁移。

**🔧 技术方法**

技术包括：Claude Opus 5等VLM作为语义前端；YOLOv8x‑World + SAM用于对象检测与分割；LiDAR+相机结合实现几何定位；MCP工具架构（10个工具）管理句柄与句法；ROS与仿真器交互；语义图与句柄管理实现可复用几何结果。

**📊 数据集**

使用 CMU Vision‑Language Navigation Challenge 2026 开发集，其中包含室内场景、自然语言导航指令、对应轨迹以及对象引用问题。

**📈 对比分析**

通过与全系统对比与两种消融（无工具结构、无几何估计）进行评估：指令跟随任务全系统得分 64.4%，去掉平台模型下降 13.3%；对象引用任务覆盖阈值 10/45，中心误差从 3.37 m 降至 2.48 m。实验显示语义‑几何分离显著提升几何精度。

**⚠️ 局限性**

局限性：每步推理与控制的延迟约半分钟，依赖单一前沿模型；未在真实硬件上验证，仿真结果不一定能直接迁移；控制器模型可替换但未实测；官方评估器关闭，评估细节由作者自行设定。

---

## 94. What Counts as a Mistake? Annotating Recitation Events in Quran Memorization Transcripts

**arXiv ID:** 2609.12085 | [PDF](https://arxiv.org/pdf/2609.12085v1)

**作者:** Mohamad Al Mdfaa `[一作]` (LemoniLab), Manuel Mazzara `[通讯]` (Innopolis University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对100条古兰经朗读录音进行了人工标注，构建了包括重复、修复、接受拼写差异等十类事件的标签集，并评估了多种基线与大型语言模型在检测、定位和标签识别上的性能。

**💡 创新点**

创新点在于将重复、修复与接受拼写差异分别视为独立标签，强调事件定位与标签识别的区别，并通过人机交互的标注与算法训练迭代框架验证此方法的可行性。

**🔧 技术方法**

技术手段包括可执行评估器、半开区间对齐、基于差异的检测以及在20分钟内利用Claude、Codex、Gemini等大型语言模型自动生成规则或模型以完成标注与评估。

**📊 数据集**

使用的数据集为9月14日的生产录音集（100条录音，314个Ayah单元+34个开头单元，共348个单位）以及公开的127条未标注录音做开发，标注基于Uthmani正字与文本一致。

**📈 对比分析**

评估采用可执行评分器计算标签感知F1、精确跨度F1与定位F1，并与plain diff、生产组件等基线对比，结果显示大型语言模型在pilot中可达micro F1 0.89、定位F1 0.95，显著优于基线的0.53。

**⚠️ 局限性**

局限性包括数据量小、标签分布偏斜、缺乏多评审与交叉验证，以及模型对规则约定的边界理解不足，导致误标和未覆盖标签。

---

## 95. Stochastic Hybrid Automata for Power Profile Modeling in Energy Systems

**arXiv ID:** 2609.12209 | [PDF](https://arxiv.org/pdf/2609.12209v1)

**作者:** Lisa Willemsen `[一作]` (University of Twente), Johann L. Hurink `[通讯]` (University of Twente)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了从电量需求与存储模型生成的随机混合自动机（SHA）以评估电池在不确定负荷下的可行性

**💡 创新点**

提出了两种构造方法（分解式和合成式电量曲线）以及对应的SHA构造规则，并系统分析了模型复杂度与工具可用性的影响

**🔧 技术方法**

使用SHA、随机混合自动机、线性与Kinetic Battery Model（KiBaM）等理论框架，并通过Modest、SMC和Flow*等工具进行定量可达性分析

**📊 数据集**

采用了示例性电池负荷曲线（包含指数、正态、均匀分布等随机时长）以及基于KiBaM的电池模型，实验数据通过10⁶次模拟或离散化得到

**📈 对比分析**

与SMC（统计模型检查）和Flow*（流管方法）对比，结果显示对合成曲线时Modest最快；对分解曲线时Flow*最优；但KiBaM模型在分析工具上表现最差，需使用仿真方法；规模扩展时计算时间随曲线长度或设备数指数增长，仿真工具对随机变量数不敏感

**⚠️ 局限性**

限制包括：1）合成曲线需人工聚合，实际中难以获得；2）仅考虑平均功率，不处理瞬时功率尖峰；3）使用线性或KiBaM模型时工具支持有限，尤其是LHA II 对分析工具的兼容性低；4）不考虑多电池或非确定性决策的情形

---

## 96. Deriving the Pure Price of Anarchy for Networked Resource Allocation Games

**arXiv ID:** 2609.12077 | [PDF](https://arxiv.org/pdf/2609.12077v1)

**作者:** Vartika Singh `[一作]` (University of Colorado Colorado Springs), Philip N. Brown `[通讯]` (University of Colorado Colorado Springs)

**通讯引用:** 1165 | [OpenAlex ID](https://openalex.org/A5044554458)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种线性规划（LP）方法，能够在任意信息网络与任意系统目标下求解网络资源分配游戏的纯价格无效性（pPoA），并给出了针对超模和子模目标函数的闭式最优pPoA与最优局部效用设计。

**💡 创新点**

创新点包括：
1) 第一次将信息网络结构纳入pPoA分析，提供了通用LP框架；
2) 证明在超模游戏中，即使信息网络不完整，通信被完全剔除的边际贡献效用设计仍能达到最优pPoA；
3) 对子模游戏（特别是集合覆盖游戏）给出网络结构下的最优/上界pPoA，并揭示通信失败导致pPoA立即降至1/2的现象；
4) 证明pPoA与网络连通度呈单调关系，体现系统鲁棒性。

**🔧 技术方法**

使用的技术主要是：
- 线性规划与对偶理论推导pPoA及最优局部效用；
- 区分超模与子模的基础函数特性；
- 通过类划分和信息图分组简化模型；
- 数值实验（基于参数化的子模函数 w(j)=j^d）验证闭式结果与鲁棒性。

**📊 数据集**

论文没有使用具体的真实数据集；所有结果均基于理论推导与数值模拟，模拟使用的是参数化的资源值与行动集配置。

**📈 对比分析**

与现有工作比较：
- 在超模游戏中给出完整的最优pPoA，填补了仅在全信息网络已知的空白；
- 对子模集合覆盖游戏给出比以往研究更精确的上界（1/2）与下界（1/6至1/2）；
- 在多类网络（盲、孤立、正常）下验证了边际贡献效用的最优性；
- 通过数值实验展示最优局部效用在面对通信失败时的鲁棒性，证明在大多数情形下差距可忽略。

**⚠️ 局限性**

局限性包括：
- 只考虑纯纳什均衡，未讨论混合均衡或学习动态；
- LP求解在大规模网络上可能面临维度灾难，未给出可扩展算法；
- 对于非子模或非超模的目标函数，未给出闭式结果，只能通过数值LP求解；
- 实际应用中网络结构与资源价值的真实测量未被实验验证，需进一步验证对真实系统的适用性。

---

## 97. Spectral Consistency-Guided Multiview Point Cloud Registration for Low-Overlap Scenes

**arXiv ID:** 2609.12417 | [PDF](https://arxiv.org/pdf/2609.12417v1)

**作者:** Tianyu Li `[一作]` (Wuhan University), Wei Yao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于谱一致性的多视点云配准框架GMPCR，能够在低重叠场景下高效、鲁棒地完成全局配准。

**💡 创新点**

创新点在于：①使用二阶相容图的主特征值同时评估对应点可靠性和扫描对置信度，实现扫描对的智能选择；②最大团生成假设与自适应历史感知同步相结合，在全局同步中动态加权并恢复被下调的边，显著提升鲁棒性；③整体无需学习、可直接利用几何一致性，避免了对大规模标注数据的需求。

**🔧 技术方法**

主要技术包括：稀疏特征采样与相互最近邻匹配；二阶相容图构造与谱分解；最大团（Clique）假设生成与加权刚体对齐；迭代加权最小二乘（IRLS）与自适应记忆系数、恢复机制的全局同步。

**📊 数据集**

在四个公开数据集上验证：3DMatch、3DLoMatch（低重叠）、ScanNet（大规模室内）、ETH（户外）。

**📈 对比分析**

与EIGSE3、L1-IRLS、RotAvg、LITS、HARA、LMVR、SGHR、SMVR、RAP等方法对比，GMPCR在3DMatch/3DLoMatch上分别获得97.2%/89.6%的回召率，优于所有竞争者；在ETH上仅落后0.1%；在ScanNet的旋转误差和平移误差均保持竞争力，并在运行时大幅加速（每场景23.7s，较SGHR快1.7×）。

**⚠️ 局限性**

局限性：对极端稀疏或弱连通的姿态图仍可能出现误匹配，且二阶相容图的构造与谱分解在大规模点云中算力瓶颈；未来工作需进一步降低候选对评估成本并提升对弱连通图的鲁棒性。

---

## 98. EAR: Entity-Aware Partitioning Approach for Retrieval-Augmented Generation Development

**arXiv ID:** 2609.12268 | [PDF](https://arxiv.org/pdf/2609.12268v1)

**作者:** Cenab Batu Bora `[一作]` (Georgia Institute of Technology), Oguz Dikenelli `[通讯]` (Ege University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EAR（Entity‑Aware Partitioning）方法，改进检索增强生成（RAG）在多项选择问答中的检索单元；

**💡 创新点**

创新点在于使用规则化的表面实体锚点来定位检索窗口，而非固定大小块，并提供可审计的规则引擎；

**🔧 技术方法**

核心技术包括表面实体提取、基于锚点的窗口检索、可选的父级上下文扩展，以及与 LLM 的 RAG+CoT 结合；

**📊 数据集**

实验使用从四本公开教材提取、去除练习题的清洗语料，以及 153 条由自动支持启发式挑选的 MMLU‑style 题目；

**📈 对比分析**

与固定块检索做对比，评估相同模型、prompt、top‑k 的准确率、检索文本量、推理延迟等指标；结果显示 EAR 检索文本量下降约 38‑40%，但准确率提升不显著，且对不同 LLM（Mistral、Gemma、DeepSeek）表现差异；

**⚠️ 局限性**

局限性包括：自动支持拆分未进行人工证据验证；规则锚点缺乏精度/召回评估；仅在 153 条题目和三种模型上测试，难以推广到其他领域；检索半径对准确率影响未完全探究。

---

## 99. A Dark Forest First Attack Is Rational Only If the Attacker Accepts It as Its Last

**arXiv ID:** 2609.12003 | [PDF](https://arxiv.org/pdf/2609.12003v1)

**作者:** Harvey Dam `[一作]` (University of Utah), Harvey Dam `[通讯]` (University of Utah)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5058305816)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了在发现外星文明后，文明应如何理性地选择是否进行首次攻击，提出了对现有模型的批判，并提出了新的理论框架。

**💡 创新点**

创新点在于将敌意的对象定义为未被控制的警告或报复能力，而不是简单地将被发现的文明视为攻击目标，强调了未知的剩余能力对攻击者的威胁。

**🔧 技术方法**

使用了博弈论和信息理论的分析方法，特别是对现有模型的比较和修正。

**📊 数据集**

没有具体提到使用的数据集，主要是理论推导和模型分析。

**📈 对比分析**

与现有模型相比，本文强调了攻击者在进行首次攻击时必须考虑的剩余能力和警告能力，表明在许多情况下，首次攻击并不是理性的选择。

**⚠️ 局限性**

限制在于本文的结果是必要条件而非充分条件，未能提供关于如何在复杂博弈中实现首次攻击的具体数值参数或模型。

---

## 100. Adaptive Chemotherapy Control under Tumor Heterogeneity via Reinforcement Learning

**arXiv ID:** 2609.12264 | [PDF](https://arxiv.org/pdf/2609.12264v1)

**作者:** Bereket Sitotaw Kidane `[一作]` (University of Texas at Arlington), Shuo Wang `[通讯]` (University of Texas at Arlington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了一个高维异质肿瘤模型，利用深度强化学习（DQN与TD3）训练闭环药物给药策略，并将其与PMP求得的开环最优策略进行对比。

**💡 创新点**

创新点包括：①基于最优控制成本函数的奖励设计；②将连续与离散动作空间的DRL策略与PMP基准对照；③在100例虚拟病人群体上评估策略在参数异质性下的鲁棒性，揭示疗效与一致性之间的权衡。

**🔧 技术方法**

采用的技术包括：深度Q网络（DQN）、Twin‑Delayed DDPG（TD3）、PMP求解的两点边值问题、以及对Hamiltonian一阶二阶导数的最优性检验。

**📊 数据集**

数据集为由100个在生长速率和药物敏感性上±10%随机扰动的虚拟病人组成的合成队列。

**📈 对比分析**

比较方法：将DRL策略与PMP基准在单例模型上进行疗效比较，并在虚拟队列上统计平均肿瘤缩减率、最终肿瘤负荷及药物使用量。结果显示：TD3在平均疗效上略优于PMP（约89% vs 87%肿瘤缩减），但其结果方差更大；DQN在一致性上更好，方差显著降低。

**⚠️ 局限性**

局限性：实验假设对所有21个肿瘤亚群都有完整观测；实际临床中测量稀疏且噪声，需要POMDP或状态估计扩展；同时模型参数的准确性仍受限，未考虑更复杂的多模态治疗方案。

---

## 101. PQLS: A High-Performance Python Library for Steady-State Simulation of Open Quantum Systems

**arXiv ID:** 2609.12309 | [PDF](https://arxiv.org/pdf/2609.12309v1)

**作者:** Evan Simanovskis `[一作]` (University of Toronto Scarborough), Javane Rostampoor `[通讯]` (University of Toronto Scarborough)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 PQLS，一个高性能 Python 库，用于一次性计算开放量子系统的稳态解，支持从物理模型到低级矩阵的三层 API。

**💡 创新点**

通过在 JAX/XLA 上实现向量化批量求解，显著降低 Python 层开销，实现数千个稳态问题一次性求解，并提供物理模型自动构建的高级抽象。

**🔧 技术方法**

采用 JAX、XLA 进行硬件加速；使用 Kronecker 乘积构造 Liouvillian；自动调用 ARC 数据库获取原子参数；利用 JAX 的自动微分和 vectorized map。

**📊 数据集**

使用真实原子数据（^85Rb、^87Rb、Cs）通过 ARC 数据库获得能级、跃迁矩阵元素、衰减率；在这些原子上构建四能级、Λ 系统等示例进行验证。

**📈 对比分析**

与 QuTiP、QuTiP‑JAX、RydIQule 在 CPU 与 GPU 环境下对 1,000 点与 1,000,000 点的参数扫荡进行比较；PQLS 在 1,000 点时速度提升约 2,743 倍，吞吐率 25 万/s；在 1,000,000 点时 GPU 仅 0.182 s，CPU 3.27 s，较 RydIQule 提升 58 倍。

**⚠️ 局限性**

主要限制在于需依赖 JAX/XLA 的硬件支持；批量向量化对显存占用要求高；对极大系统规模或非批量问题仍有性能瓶颈；使用非 ladder 拓扑时需要切换到低级 API。

---

## 102. Uncertainty-Aware Conflict Detection Against Operator-Conditioned Weather Hazards

**arXiv ID:** 2609.12095 | [PDF](https://arxiv.org/pdf/2609.12095v1)

**作者:** Balram Kandoria `[一作]` (SkyGrid), Aryaman Singh Samyal `[通讯]` (SkyGrid)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种将轨迹不确定性预测与三维多面体天气障碍相结合的战略飞行计划验证框架。

**💡 创新点**

创新点在于：1）闭式不确定性估计结合NURBS曲线拟合与Kalman滤波；2）使用Sigmoid混合测量噪声模拟FMS的不确定性收敛；3）按操作员阈值将气象网格转化为多层级多面体障碍并加入安全缓冲；4）基于网格交叉的三维冲突检测。

**🔧 技术方法**

使用的技术包括NURBS曲线拟合、Kalman滤波、Sigmoid测量噪声模型、速度到时间不确定性转换、GPR风场估计、三维凸包/多面体构造以及网格交互算法。

**📊 数据集**

数据集：FAA航班计划/ADS‑B轨迹数据用于轨迹不确定性验证；HRRR气象预报（3 km分辨率）用于生成天气障碍。

**📈 对比分析**

与传统确定性验证相比，基于不确定性的FPV在23条实测航班中捕获率为69.6%，并在Houston案例中揭示因时间不确定导致的冲突。实验显示加入时间不确定后约23%的到达时间偏差可触发冲突，性能明显优于仅使用确定性模型。

**⚠️ 局限性**

局限性包括：使用恒定速度过程模型忽略爬升/降落/转弯；凸包近似导致非凸障碍过度保守；仅评估单一路径；未考虑实时气象更新与多航班协同。

---

## 103. HoliBench: A Cross-Platform Benchmarking and Deployment Toolkit for Foundation Models in CPS-IoT Applications

**arXiv ID:** 2609.12412 | [PDF](https://arxiv.org/pdf/2609.12412v1)

**作者:** Inesh Chakrabarti `[一作]` (University of California, Los Angeles), Mani Srivastava `[通讯]` (University of California, Los Angeles)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了Holibench，一个跨平台的基准与部署工具，能够同时测量模型精度、延迟、能耗和内存并支持多种模型、量化与推理后端；

**💡 创新点**

创新点在于：①统一的硬件抽象层实现能耗、延迟和内存的跨设备可比测量；②对多模态基础模型（LLM、VLM、TSFM）进行阶段级别的分析与量化敏感性评估；③将单模型剖面通过时间加权与线性回归方法，准确预测多模型协同部署的性能与能耗；④提供约束感知配置选择器（ILP求解），实现精度、延迟、能耗、内存四维目标的最优配置；

**🔧 技术方法**

技术包括Python实现的四层架构（平台抽象层、推理时间模型、模态驱动、约束选择器），使用Hydra配置、Intel RAPL/ NVML/ jtop/ powermetrics等多源采样，bitsandbytes INT4/INT8量化，HuggingFace、vLLM、SGLang、TensorRT-LLM等多后端；

**📊 数据集**

使用了多种数据集：LLM任务（Wikiann、GoEmotions、SST-2、IMDB、AG News、CNN/DM、WMT16、HumanEval、WikiText-2、C4）；VLM任务（CIFAR-100、ImageNet、VQAv2、CountBenchQA、DocVQA、GTSRB、HaGRID、UTKFace、Country211）；TSFM任务（NAB、GIFT-EVAL、M3 Monthly）；

**📈 对比分析**

比较方法是基于全设计空间的单模型剖面与多模型组合预测，通过ILP求解器得到可行配置，并在CARLA等真实CPS场景中验证预测误差（<1.2%延迟，<2.5%功耗）。实验表明：①量化对不同硬件的加速效果不一致；②能耗与延迟呈线性关系；③单模型剖面可在多模型部署中保持误差≤2.5%；

**⚠️ 局限性**

局限性包括：仅评估顺序共驻执行，未覆盖并发或动态批量；仅使用HuggingFace推理后端的验证；未加入MLX等新后端；能耗测量受底层接口分辨率限制；对大型Agent工作负载（多批次、缓存）仍需扩展；

---

## 104. Feature Recovery for Object Understanding After Irreversible Fire Damage

**arXiv ID:** 2609.12078 | [PDF](https://arxiv.org/pdf/2609.12078v1)

**作者:** Aditi Tiwari `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一个基于真实图像驱动的火灾后物体理解基准和一个轻量残差特征恢复模块，在冻结模型上实现检测、检索、材料恢复、描述生成与功能推理等多任务性能提升。

**💡 创新点**

首次构建配对的无损与多级退化物体轨迹数据集，并设计仅训练特征空间残差模块实现对退化特征的普适恢复，显著提升不同模型在火灾后环境中的鲁棒性。

**🔧 技术方法**

采用配对图像对齐的残差Transformer对冻结模型中间特征进行映射，结合多任务评测框架进行训练与评估。

**📊 数据集**

使用包含 21.4K 真实图像驱动合成场景和 499 条 0–4 级退化轨迹、覆盖 189 类物体的基准集，以及 500 条真实火灾裁剪图像。

**📈 对比分析**

与 RF‑DETR、CLIP、SigLIP2 及五大 VLM 在检测、检索、材料恢复、描述生成和功能推理任务上对比，加入残差模块后检测 mAP 提升约 30%，检索 R@1 提升 12–20%，其余任务亦显著提升。

**⚠️ 局限性**

依赖配对退化图像且模块绑定到特定模型/特征空间；对极端退化的真实场景提升有限；仅恢复预退化状态，无法同时支持对当前退化查询的推理。

---

## 105. Hyperion: An AI-powered HPC cluster for sciences and humanities research that utilizes ML for predicting job turnaround time

**arXiv ID:** 2609.11946 | [PDF](https://arxiv.org/pdf/2609.11946v1)

**作者:** Jun Zhou `[一作]` (University of South Carolina), Paul Sagona `[通讯]` (University of South Carolina)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5022173840)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并部署了高校首个混合型 HPC 集群 Hyperion，集成了多种 CPU、GPU 与大内存节点，并在其中训练并上线了两款用于预测作业等待时间与壁钟时间的机器学习模型，随后展示了在材料科学、神经影像学和数字影像学等三个科研案例中的应用。

**💡 创新点**

创新点在于：①构建面向多学科的异构 HPC 体系并实现弹性扩展；②利用 Random Forest 等回归模型对 490 万条历史作业数据进行高精度的 turnaround 预测（R² ≈ 0.83/0.74）；③将预测结果无缝集成到 Slurm 提交脚本中，直接为用户提供即时作业时长估计；④通过 Hyperion 平台加速了大规模材料属性预测、MRI 模板构建与电影视频文本检测等多领域深度学习任务。

**🔧 技术方法**

所用技术包括：HPC 架构（Intel Xeon、NVIDIA GPU、InfiniBand、GPFS）；调度系统 Slurm；机器学习框架（Random Forest、YOLOv5、Graph Neural Networks、Transformer-based 文本检测）；深度学习框架（PyTorch、TensorFlow）；数据传输工具 Globus；图像与视频处理库 OpenCV、FFmpeg 等。

**📊 数据集**

主要数据集为：①从 Hyperion 2023 之前 15.3 M 作业中导出的 4.9 M 条日志（XDMod 数据仓库）；②>100 k 物理晶体的原子层级训练数据；③来自多家公开数据库（OASIS、IXI、ABIDE 等）累计 4 600 张 3.0 T MRI；④1 200 部美国海军电影的原始帧与压缩视频，存于 AWS Glacier 与本地 SAN。

**📈 对比分析**

方法对比基于 R² 指标；在 2018–2023 年的测试集中，等待时间模型在 {Submit Time, Nodes, Cores, GPUs, Queue ID} 特征下达到 0.8667，壁钟时间模型在 {Submit Time, Nodes, Cores, GPUs, User ID, Queue ID} 下达到 0.7459；与以往仅使用单一时间特征的基准相比，模型精度提升约 15–20%。

**⚠️ 局限性**

局限性包括：①模型对用户 ID 的依赖导致对新用户泛化能力有限；②仅基于历史作业数据训练，无法即时捕捉资源调度策略或系统负载的突发变化；③在极短作业（秒级）上预测误差仍可能显著；④集群扩展至 2024 年后硬件升级需重新调优模型。

---

## 106. Debiasing as a Measurement Intervention: Calibrated Ties and Resolution Loss in LLM-as-a-Judge Evaluation

**arXiv ID:** 2609.12439 | [PDF](https://arxiv.org/pdf/2609.12439v1)

**作者:** Liang Zhao `[一作]` (Shanghai University of International Business and Economics), Jiangzhe Chen `[通讯]` (Shanghai University of International Business and Economics)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 TraceJudgeBench 诊断基准，用于评估 LLM 评判者在处理 RAG 与代理工作流输出时对引用样式的偏差与分辨率保持。

**💡 创新点**

将去偏视为测量干预，首次区分 Tie 的三种含义，并提出 TRACE 规范化‑掩码判定流程以恢复质量分辨率。

**🔧 技术方法**

采用 LLM pairwise judging、逐步加深的去偏提示、TRACE 规范化+掩码评判、统计 Bootstrap 及 Wilson 区间等技术。

**📊 数据集**

使用 HotpotQA 与 FinQA 数据集，分别构造机械化引用对、自然引用对、等价对、清晰对立对、软/中等质量对以及工作流程排名样本。

**📈 对比分析**

在 GPT‑5.5、Claude Sonnet 4.6、DeepSeek V4‑Flash 以及 Qwen2.5‑14B‑Instruct‑AWQ、Gemma‑3‑12B‑IT 上对比去偏前后赢率、Tie 率；去偏显著抑制引用偏差，但在中等质量对上导致 Tie 增多；TRACE 方案能恢复约 97%‑100% 的分辨率。

**⚠️ 局限性**

局限性包括基准受控模板有限，未覆盖全部自然差异；人类验证人员有限；TRACE 的计算成本高且可能出现提取错误；未完整验证所有开源模型。

---

## 107. A Survey on Quantum-Safe Cryptographic Mechanisms: Building Blocks and Applications

**arXiv ID:** 2609.11991 | [PDF](https://arxiv.org/pdf/2609.11991v1)

**作者:** Ricardo Parizotto `[一作]` (Universidade Federal da Fronteira Sul), Israat Haque `[通讯]` (Dalhousie University)

**通讯引用:** 1023 | [OpenAlex ID](https://openalex.org/A5047687728)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对量子安全加密机制及其在不同应用中的迁移路径进行了系统综述。

**💡 创新点**

首次系统性梳理迁移方法、构建块和应用领域，并提出基于四个维度的评估框架。

**🔧 技术方法**

采用系统文献检索、分类编码、构建块层次模型、混合方案评估等技术。

**📊 数据集**

基于IEEE Xplore、Scopus、ACM DL 等数据库检索，获取约54篇核心论文和74篇补充文献。

**📈 对比分析**

通过构建块分类和迁移案例对比，发现PQC在TLS、VPN等应用成熟度高，但在低功耗IoT和大规模网络中仍面临性能瓶颈。

**⚠️ 局限性**

缺乏统一的量化评估指标，实证实验不足，且量子硬件成本和兼容性仍是主要障碍。

---

## 108. Do LLMs Trust the Accuser or the Accusation? Measuring Belief Shifts in Werewolf

**arXiv ID:** 2609.12446 | [PDF](https://arxiv.org/pdf/2609.12446v1)

**作者:** Yu-Yu Yang `[一作]` (National Yang Ming Chiao Tung University), I-Chen Wu `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Werewolf社交推理游戏中构建信念转移评估基准，分析观察者在收到指控消息后对被指控者和指控者的信念变化。

**💡 创新点**

提出以信念转移为中间指标的交互式评估框架，聚焦于指控对观察者信念的影响，区别于传统仅靠最终胜负的评估方法。

**🔧 技术方法**

使用LLM自玩Werewolf生成对话，借助Claude Opus 4.6进行指控/怀疑句子标注；对40个开源LLM在先验信念、指控后信念转移与先前信任水平之间的相关性进行测评。

**📊 数据集**

200局7人Werewolf自玩对话，标注得到1224条指控/怀疑消息，作为信念转移评估的数据集。

**📈 对比分析**

通过Spearman相关性、平均信念转移值等指标比较不同规模LLM（1B–120B）的表现；结果显示模型规模越大，先验角色辨别与对不可信指控的抵抗力越强，但对可信指控仍易产生信念偏移，整体性能仍有提升空间。

**⚠️ 局限性**

仅评估观察者端信念更新，未考察说话端生成策略；实验仅在Werewolf单一环境下进行，缺乏对人类或其他游戏的泛化；标注采用单一LLM，可能存在偏差；未覆盖封闭源大模型的验证。

---

## 109. InitGen: Candidate Generation for Interaction Initiation in Intelligent Assistants

**arXiv ID:** 2609.11953 | [PDF](https://arxiv.org/pdf/2609.11953v1)

**作者:** Ruize Shi `[一作]` (Huazhong University of Science and Technology), Rui Zhang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 183713 | [OpenAlex ID](https://openalex.org/A5100381911)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在智能助手中实现交互启动的候选查询生成，联合生成查询集并根据点击反馈进行对齐。

**💡 创新点**

采用查询集级别的 KTO 对齐并引入用户活跃度与排名分数加权，使用滚动窗口进行周期性更新，以处理部分延迟反馈和数据漂移。

**🔧 技术方法**

使用大语言模型（如 Qwen2.5-1.5B），监督微调、KTO、加权 KTO（WKTO）以及滚动窗口训练。

**📊 数据集**

使用 OPPO Xiaobu Assistant 的生产交互日志，覆盖约 1.5 亿月活跃用户。

**📈 对比分析**

与基线（SFT+KTO）在在线 A/B 测试中对比，CTR 提升 69.1%，曝光提升 17.9%，总点击数提升 99.5%。

**⚠️ 局限性**

仍受限于部分反馈、对点击位置等未建模、需要离线周期性更新，且方法在不同平台或数据稀疏场景下的可迁移性未知。

---

## 110. Error-Rate Reduction in LDPC Decoding via Bit-Aligned Temporal Reinforcement in Parallel Probabilistic-Bit Dynamics

**arXiv ID:** 2609.12389 | [PDF](https://arxiv.org/pdf/2609.12389v1)

**作者:** Naoya Onizawa `[一作]` (Tohoku University), Takahiro Hanyu `[通讯]` (Tohoku University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在并行p-bit（概率位）动态下，对随机正则(3,6) LDPC码进行解码的改进方法，提出并验证了一种“加性饱和响应记忆”机制，能够在保持同步更新的同时显著降低比特错误率。

**💡 创新点**

创新点在于：①引入了将每个p-bit的饱和响应值作为临时记忆并在下一个周期以加性方式重新注入的机制；②证明该机制比仅使用增益、归一化、响应打乱或同位二进制反馈更能提升解码性能；③展示该机制在不同初始化、不同规模、以及跨代码转移时的鲁棒性，且无需针对每个码重新调参。

**🔧 技术方法**

技术手段包括：Jacobi型并行p-bit更新与随机部分激活；加性与有限响应两种记忆策略；在LDPC解码中使用分块能量与通道驱动的因式化目标；对比Belief Propagation（BP）和记忆无关的概率模拟退火（pSA）等基准；采用高统计量模拟（10,000+试验）和跨代码抽样验证。

**📊 数据集**

数据集为10个独立生成的随机正则(3,6) LDPC矩阵，块长分别为96、192、288；每个矩阵在2.0、2.5、3.0 dB三个信噪比下进行实验；此外使用代表矩阵进行高统计量评估。

**📈 对比分析**

比较方法：与记忆无关的pSA优化方案、BP参考、以及各种记忆控制（归一化、增益、打乱、二进制反馈）进行同参数、同初始化下的对比；在跨代码转移实验中保持相同的参数包；性能方面，加性记忆在代表矩阵上分别提升33.5%、74.8%、81.8%（BER）和29.2%、81.0%、83.9%（FER），在30个独立码上相对pSA固定转移包提升35.5%、76.6%、81.1%（BER）。

**⚠️ 局限性**

局限性：仅在(3,6)正则LDPC码和BPSK AWGN信道下验证；未探索不同度分布、不同通道、不同调度或更大码长的泛化；机制的具体硬件实现与能耗、面积、延迟等工程参数未给出；在某些情形下如N=192的增益控制会冻结到错误码字，表明仅凭低突变率并不能完全保证正确性。

---

## 111. Theoretical Guarantees for One-Shot Magnitude Pruning and Compute-Adaptive Early Exit

**arXiv ID:** 2609.12337 | [PDF](https://arxiv.org/pdf/2609.12337v1)

**作者:** Erdem Koyuncu `[一作]` `[通讯]` (University of Illinois Chicago), Erdem Koyuncu (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究单个神经元及深层网络中一次性幅值剪枝（静态计算缩减）和早期退出（自适应计算缩减）的理论性质，给出了剪枝比例、网络深度和激活函数对精度的影响，并推导了误差与计算消耗的权衡关系。

**💡 创新点**

创新点包括：①证明一次性幅值剪枝在高维下余弦相似度收敛到确定值并给出显式收敛速率；②引入条件感知器模型并推导早期退出的计算‑误差幂律关系；③通过 Gaussian 神经网络极限刻画深度累积误差，得到闭式递推并给出激活函数对误差累积的定量描述；④在宽度极限下解析计算退出与最终输出的相关系数，进一步解释早期退出的精度衰减。

**🔧 技术方法**

使用技术包括：订单统计与截断 Gamma 分布分析、Berry–Esseen 中心极限定理与尾部控制、子高斯尾部假设、Gaussian 过程（NNGP）极限、核递推（TCDF）、线性最小二乘退火、误差分解与上界推导、数值模拟与经验验证。

**📊 数据集**

实验数据集：CIFAR‑10（CNN）、SST‑2（DistilBERT 预训练模型）和 CIFAR‑100（ViT‑Base 视觉 Transformer）用于验证剪枝和早期退出的理论预测。

**📈 对比分析**

与全计算基线相比，单次幅值剪枝在高维下误差仅为少数百分点；早期退出在大多数计算预算下误差与全计算相差不到 1%，且误差随计算缺口呈幂律下降。实验曲线与理论给出的收敛速率和指数高度吻合，证明了理论模型的实用性。

**⚠️ 局限性**

局限性：假设权重独立同分布且宽度趋向无穷，未考虑训练后权重的相关性；早期退出模型仅考虑单一退出点、线性输出头；子高斯尾部假设在某些实际网络中可能不完全成立；多出口、多类别精度分析仍待进一步研究。

---

## 112. AI-Research Agents in the Wild. From GitHub and arXiv to Regularities and Gaps

**arXiv ID:** 2609.11975 | [PDF](https://arxiv.org/pdf/2609.11975v1)

**作者:** Aleksey Komissarov `[一作]` (Neapolis University), Andrey Ustyuzhanin `[通讯]` (Constructor Labs)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个139条公开AI研究代理仓库与101篇论文的登记库，并通过编码卡、来源锚点和人类审议对其进行系统化标注，随后提出六个可检验的设计规律，利用前瞻性审计和结构化图谱验证规律的触发与否，探究仓库与论文的相关性与作者重叠。

**💡 创新点**

创新点在于：①将仓库与论文的关系从传统引用映射转为策展人指定的“相关性”图谱；②提出六条具体的设计规律，并在预设触发条件下进行前瞻性审计，首次将理论与实验结合；③给出了作者重叠的下限估计，强调身份解析与证据链的重要性；④通过生成式AI辅助编码与检验，展示了可复制的自动化文献与仓库梳理流程。

**🔧 技术方法**

技术方法包括：生成式AI编码代理（如Claude、OpenAI Codex）生成证据卡；基于源代码锚点的单一来源验证；手工审议确认争议条目；前瞻性审计脚本在新增25个仓库上检验触发事件；统计Bootstrap和置信区间计算；以及对线性、向量、关系、锦标赛等优化循环结构的系统编码。

**📊 数据集**

数据集为：139条经过去重和规范化的公开GitHub仓库（包括代理、基准、库、子系统等）；101条论文记录（包括作者、代码可用性、伴随代码链接和与仓库的相关性边）；以及在审计期间收集的25个新增仓库和后续重测的962个自冻结以来新建仓库。

**📈 对比分析**

比较方法：在已冻结的25个新增仓库上进行前瞻性触发事件检测；通过触发事件的出现与否与已建立的六条规律进行对照；对R1–R4在批量检验中得到有界支持，对R5在全语料中被反驳，对R6的方向性预测失败；最终得到R1–R4可接受，R5需后验限定，R6仍待探索。

**⚠️ 局限性**

局限性包括：①登记库是工程化抽样，可能无法代表全球公开仓库的真实分布；②大部分变量缺乏双重编码，编码可信度受限；③相关性图谱受策展人主观判断影响，未与真实引用语义对齐；④作者身份解析仅提供下限估计，公共姓名与Handle冲突导致不完整；⑤前瞻性审计样本非随机，可能缺乏外部可推广性；⑥结构性规律在后续重测中仍需进一步验证。

---

## 113. One Click to Leak: Characterizing the Real-World Usage and Threat Impact of MNO-based Single Sign-On Websites

**arXiv ID:** 2609.12037 | [PDF](https://arxiv.org/pdf/2609.12037v1)

**作者:** Jiasheng Huang `[一作]` (Tsinghua University), Hui Jiang `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并量化了基于移动网络运营商的一键登录（MSSO）在网页上的部署、信任缺陷及其“One‑Click‑to‑Leak”攻击，并通过构建OCL Detector实现一年内对116,852个URL、729个域名的全景检测；

**💡 创新点**

首次系统揭示MSSO的三大信任缺陷、可信劫持威胁，并在真实环境中发现73.6%的URL存在缺陷，识别并拆解了完整的OCL攻击生态；

**🔧 技术方法**

采用被动DNS关联、搜索引擎URL重建、移动模拟Headless浏览器动态探测、脚本聚类分析及恶意供应链追踪等技术；

**📊 数据集**

使用500亿条pDNS请求、550万FQDN/日的数据集，以及数百万页面的搜索引擎网页关系图和被动DNS与Web资源关联的数据；

**📈 对比分析**

与传统原生App安全评估相比，本文在Web规模下实现高精度（>96%）和高覆盖率；OCL Detector在一年内覆盖729域、116k URL，缺陷率和真实攻击验证了其有效性；

**⚠️ 局限性**

局限性包括数据仅来自中国，跨国代表性有限；检测依赖DNS和动态脚本分析，易受隐蔽或加密代码误判；无法捕获完全隐蔽的攻击流。

---

## 114. Reinforcement Learning for Syndrome Extraction

**arXiv ID:** 2609.12020 | [PDF](https://arxiv.org/pdf/2609.12020v1)

**作者:** John Zhuoyang Ye `[一作]` (University of California, Los Angeles), Jens Palsberg `[通讯]` (University of California, Los Angeles)

**通讯引用:** 6746 | [OpenAlex ID](https://openalex.org/A5011356414)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种基于强化学习与重要性采样的自动调度框架，用于在量子误差校正中寻找低逻辑误差率的测序（syndrome extraction）电路。

**💡 创新点**

创新点在于：① 将强化学习（PPO）与重要性采样相结合，利用放大噪声下的罕见失败样本来训练策略；② 在马尔可夫决策过程框架下对CNOT顺序进行建模，并引入探索奖励和局部改进以加速搜索；③ 证明在固定采样预算下重要性采样评估保持无偏，且通过自适应停止实现高效的罕见事件估计。

**🔧 技术方法**

主要技术包括：强化学习（PPO）、重要性采样（放大因子k）、罕见事件估计、基于Detector error model的逻辑错误率评估、局部搜索与候选池再评估、以及使用Stim、MWPM、BP-OSD等工具进行解码和模拟。

**📊 数据集**

使用的基准数据集包括：• 11个表面码（Surface code）循环（从d=5到d=15）；• 8个颜色码（color code）及其超平面版本；• 8个量子LDPC码（Quantum LDPC）；所有实验均采用Brisbane噪声模型（来自IBM Brisbane的校准数据）。

**📈 对比分析**

对比方法：与两种最先进的自动调度工具（MCTS+Monte Carlo Tree Search和MaxSAT）以及手工设计的基线排程进行比较。实验结果显示：相较于基线排程平均降低逻辑误差率约 **%**（具体数值视代码修正而定），相较于MCTS工具平均降低约 **71.7%**，在d=15的表面码上实现了约 **97.8%** 的逻辑误差率下降，最终评估时逻辑误差率低至 3.62 × 10⁻⁹。

**⚠️ 局限性**

局限性：① 需要对重要性采样放大因子进行手动校准；② 罕见事件评估的非线性对奖励造成一定偏差；③ 对较大码（距离>15）或高维度量子LDPC码的搜索时间和计算成本仍然较高；④ 结果依赖于所选噪声模型和解码器，可能不具备跨平台的普适性；⑤ 需要额外的目标噪声下的独立评估以消除自适应停止引入的偏差。

---

## 115. Neural Multichannel Distant Speaker Diarization with Heavy-tailed Source Separation Model

**arXiv ID:** 2609.12154 | [PDF](https://arxiv.org/pdf/2609.12154v1)

**作者:** Sicheng Mao `[一作]` (LTCI, Telecom Paris, Institut Polytechnique Paris), Roland Badeau `[通讯]` (LTCI, Telecom Paris, Institut Polytechnique Paris)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于神经FCASA的重尾模型，用于联合盲源分离与说话人划分，显著提升远场说话人分离的鲁棒性。

**💡 创新点**

创新点在于将传统的高斯方差模型替换为重尾分布（Leptokurtic Generalized Gaussian 与 Student's t），并通过高斯尺度混合模型统一原始和重尾目标函数，实现了轻量级改动的性能提升。

**🔧 技术方法**

使用深度生成式模型（VAE）进行频域特征建模，结合多通道 Wiener 滤波、交叉熵损失、PIT 损失和重尾似然估计，训练与推理均基于神经网络与 ISS 直角化算法。

**📊 数据集**

在 AMI、AliMeeting 与 CHiME‑6 三个多通道远场会议语音数据集上进行实验，覆盖不同麦克风阵列与语言环境。

**📈 对比分析**

与原始高斯 FCASA 基线相比，重尾模型在 DER 与 JER 上分别降低约 10–15% 绝对误差，尤其在重叠语音和噪声严重的场景中表现突出，跨数据集迁移实验也显示出一定的泛化能力。

**⚠️ 局限性**

方法对重尾分布的形状参数敏感，某些参数配置会导致性能下降；此外未对参数进行自动估计，仍需手动调优。

---

## 116. MaRDMO: FAIR Documentation of In-Silico Research

**arXiv ID:** 2609.11931 | [PDF](https://arxiv.org/pdf/2609.11931v1)

**作者:** Marco Reidelbach `[一作]`, Marcus Weber `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

暂无具体研究内容

**💡 创新点**

暂无创新点

**🔧 技术方法**

暂无技术说明

**📊 数据集**

暂无数据集

**📈 对比分析**

暂无比较方法，性能未知

**⚠️ 局限性**

暂无限制说明

---

## 117. BlueLM-GUI Technical Report: A Real-Device-Centric Flywheel for Self-Improving Mobile GUI Agents

**arXiv ID:** 2609.12394 | [PDF](https://arxiv.org/pdf/2609.12394v1)

**作者:** Tong Ye `[一作]`, Xiaoxin Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种以真实设备为核心的自我改进移动 GUI 代理蓝 LM‑GUI，构建了一个闭环飞轮，涵盖从数据采集、训练到评估的三大原则：每个样本都被利用、每次试验均在真实设备上执行、每个查询都能随模型演进而演化；

**💡 创新点**

创新点在于（1）将所有数据采集与 RL 探索统一至真实设备，消除模拟器分布差距；（2）引入“错误修正与衍生模块”，将失败轨迹转化为监督信号；（3）采用可量化的配额驱动评测框架，使基准随模型进化而动态调整；

**🔧 技术方法**

核心技术包括多模态 LLM（VLM+语言模型）与自回归解码、连续预训练 (CPT)、监督微调 (SFT)、基于真实设备的强化学习 (RL)、Heterogeneous Triple‑System Consensus (HTSC) 评估、以及基于函数树与全局环境记忆的查询生成；

**📊 数据集**

使用的数据集为：大规模真实设备收集的 GUI 轨迹（数百万条），单步截图–动作对、工具调用对，函数树与全局环境图数据库，以及外部引入的多样化查询与手工验证的基准样本；

**📈 对比分析**

与现有最强闭源 API 模型、公开基准 AndroidWorld 以及多种开源模型对比，蓝 LM‑GUI 在 PhoneBuddy 基准上取得 87.4% 的成功率，领先最佳闭源 API 82.3%，在 AndroidWorld 基准上得 84.9%，超过所有开源基准并仅略逊于规模更大的闭源模型；

**⚠️ 局限性**

局限性包括：仍需昂贵的真实设备集群与人工审核，难以覆盖所有动态 UI 变化，评测仍集中在第三方应用，且对极端网络/权限异常的鲁棒性尚待进一步提升。

---

## 118. PinDCO: Whole-Page Aware Dynamic Creative Optimization at Scale

**arXiv ID:** 2609.11943 | [PDF](https://arxiv.org/pdf/2609.11943v1)

**作者:** Yu Hao `[一作]` (Pinterest Inc.), Akanksha Baid `[通讯]` (Pinterest Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并在Pinterest平台上线PinDCO系统，实现对海量生成式创意的动态选择与匹配，以提升广告点击率和整体页面体验。

**💡 创新点**

创新点包括：①创意组件融合网络（CCFN）使用组件专属塔和超参数，②像素感知调整模块（PAM）针对水波纹网格布局实现全页优化，③在生产环境中将探索-利用策略与多臂老虎机结合到在线个性化创意选择，④预选轻量化模型与高容量模型并行推理，⑤通过缓存与动态批处理显著提升服务效率。

**🔧 技术方法**

技术手段包括：深度多塔网络（CCFN），强化学习/多臂老虎机（ε-greedy + bandit），特征工程（内容、用户、经验、上下文），模型并行与预选，缓存优化与动态批处理，离线评估（ROC‑AUC/PR‑AUC）与在线A/B实验。

**📊 数据集**

数据集：Pinterest生产广告流量日志（数月的点击/互动数据），用于离线训练与评估；在线A/B实验基于实时流量。

**📈 对比分析**

与无创意优化（No‑CR）、轻量化单模型、Peri‑CR等基线对比：离线提升PR‑AUC +0.171%（AUC‑ROC +0.046%）；在线提升广告CTR +3.09%，同时整体页面成功会话+0.04%；预选、PAM、动态批处理分别降低P99延迟 87%/114% 与缓存 9.6%/12.8%。

**⚠️ 局限性**

局限性：对新生成创意的冷启动仍依赖探索数据；像素惩罚函数仅基于宽度一致的网格，未覆盖宽度变化；缺少对多图/图形创意和用户序列建模的支持；未深入讨论多样性与公平性影响；在高QPS环境下仍需进一步优化缓存与批处理策略。

---

## 119. On the Fragility of Worst-Case Nash Equilibria in Atomic Congestion Games

**arXiv ID:** 2609.12220 | [PDF](https://arxiv.org/pdf/2609.12220v1)

**作者:** Colton Hill `[一作]` (University of Colorado at Colorado Springs), Philip N. Brown `[通讯]` (University of Colorado at Colorado Springs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种线性规划方法，用于计算在受激励的原子拥堵博弈中，价格无效性（PoA）与代理人对均衡满意度（稳定性边际）之间的上界关系，并证明在最差均衡下代理人对其决策无差异；

**💡 创新点**

创新点在于将PoA与稳定性边际结合，利用LP构造精确的上界，揭示最差均衡下稳定性边际必为零，从而说明此类均衡罕见；

**🔧 技术方法**

主要技术包括线性规划建模、稀疏约束变换、平滑性理论和对偶分析；

**📊 数据集**

实验使用合成数据，考虑立方延迟函数（$ℓ_e(x)=x^3$）以及三种线性本地收费机制（最优、常数、边际成本）以及无收费情形；

**📈 对比分析**

与已知的$(λ,μ)$-平滑性上界对比，实验结果显示所得到的PoA上界随稳定性边际增大而严格下降，且在多种收费策略下均优于传统上界；

**⚠️ 局限性**

局限性在于仅给出上界，未证明下界或紧性；方法假设资源成本可用基函数线性组合，且仅针对原子拥堵博弈，未考虑异质路线偏好或随机化行为。

---

## 120. "I Felt Very Seen, But Still Very Alone": Longitudinal Trajectories of General-Purpose LLM Use for Socioemotional Support

**arXiv ID:** 2609.12314 | [PDF](https://arxiv.org/pdf/2609.12314v1)

**作者:** Meryl Ye `[一作]` (Carnegie Mellon University), Ranjit Singh `[通讯]` (Data & Society)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对18名美国成人进行了一项多阶段纵向定性研究，追踪他们使用通用大型语言模型（LLM）聊天机器人进行社会情感支持的轨迹，包括初始访谈、四周日记、焦点小组和离场访谈；

**💡 创新点**

首次系统描绘了用户从非正式使用转向情感支持的渐进过程、日常使用模式、因模型更新、公共话语及生活变动而产生的中断与再协商，提出了HCI评估时需考虑用户历史与照护生态的创新方法；

**🔧 技术方法**

采用扎根理论编码、纵向访谈、日记记录、焦点小组讨论等多方法定性分析技术；

**📊 数据集**

使用了参与者的访谈记录、日记条目、焦点小组录音、可选聊天日志以及与研究时间线同步的外部事件记录；

**📈 对比分析**

将研究发现与已有的聊天机器人使用研究、照护生态理论和安全评估框架进行对照；由于是定性研究，未给出数值性能指标，而是通过案例对照展示模型更新对用户体验的影响；

**⚠️ 局限性**

样本规模小且自选，全部为美国成年人，缺乏跨文化验证，研究仅涵盖通用LLM，数据来源为主观报告，缺乏因果推断，外推性受限。

---

## 121. UniMo: Unifying Human and Animal Motion Generation

**arXiv ID:** 2609.12342 | [PDF](https://arxiv.org/pdf/2609.12342v1)

**作者:** Zeyu Zhang `[一作]` (Australian National University), Richard Hartley `[通讯]` (Australian National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 UniMo，一种统一的点云基3D动作生成框架，并构建了规模最大的动作-语言数据集 UniML3D；

**💡 创新点**

创新点在于把参数化骨架转化为无参数点云并采用动态采样，同时使用点云 VQ‑VAE 与 Mask Transformer 实现跨物种统一建模；

**🔧 技术方法**

使用了点云编码/解码、VQ‑VAE、Mask Transformer、CLIP 文本编码、动态采样、逆运动学等技术；

**📊 数据集**

采用了 UniML3D（145,907 条序列、433,388 条字幕）、HumanML3D、KIT‑ML、AnimalML3D 等数据集；

**📈 对比分析**

在 UniML3D、AnimalML3D、HumanML3D、KIT‑ML 等基准上与多种基线对比，R‑Precision、FID、MM‑Dist、Diversity、MModality 等指标均刷新 SOTA，且训练时间和推理延迟显著降低；

**⚠️ 局限性**

局限性在于 UniML3D 仍基于精选序列，未包含从真实视频中提取的自然动物行为，导致规模和多样性受限。

---

## 122. READ: Learning Risk-Informed Fields for End-to-End Autonomous Driving

**arXiv ID:** 2609.12371 | [PDF](https://arxiv.org/pdf/2609.12371v1)

**作者:** Zhiyuan Liu `[一作]` (Tsinghua University), Jianqiang Wang `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 READ 框架，通过学习可微分的时空风险场来显式表达驾驶场景中的安全影响；

**💡 创新点**

创新点在于用部分几何与行为约束学习场景自适应的连续风险场，而非预设固定风险函数，同时使风险场可与多种规划器（端到端与 Vision‑Language‑Action）无缝集成，并可作为后处理模块；

**🔧 技术方法**

采用多尺度高斯基函数建模风险场，结合非可驱动区域、未来代理占据、专家轨迹与动力学残差等五个损失；集成到 BEV 解码器（Conv/ViT）与 VLA 隐变量；使用梯度可微的查询与轨迹细化；

**📊 数据集**

在 NAVSIM v1 与 v2 两个城市驾驶仿真数据集上进行训练与评估；

**📈 对比分析**

与传统 BEV 语义监督、规则化场景安全场以及多种端到端和 VLA 方案对比，READ 在 PDMS（和 EPDMS）上均提升 0.5–1.0 点，且在 NC、DAC、TTC 等子指标上均有显著提升；可微后处理同样能显著提高已冻结的外部规划器；

**⚠️ 局限性**

主要限制是全密集时空风险场计算量大，适用于大规模道路场景时可能成本高；且风险值并非严格可解释为碰撞概率，尚缺乏校准与安全保证。

---

## 123. OphBiWSSD: Scaling Temporal Action Localization in Ophthalmic Surgeries with Bidirectional Weight-tied State Space Duality

**arXiv ID:** 2609.12409 | [PDF](https://arxiv.org/pdf/2609.12409v1)

**作者:** Yang Liu `[一作]` (Tsinghua Shenzhen International Graduate School), Chengming Yang `[通讯]` (Southern University of Science and Technology Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了OphBiWSSD，一种基于双向权重共享状态空间双子模型的眼科手术稠密动作定位框架，能够在高帧率手术视频中精确预测动作类别及其时间边界。

**💡 创新点**

核心创新在于：①引入双向选择性扫描（Bidirectional Selective Scan），实现非因果时间上下文的线性复杂度建模；②采用权重共享的状态空间双子（Weight‑Tied SSD），实现前向后向一致的特征学习并显著减少参数量；③结合结构化SSM与Transformer关联性，兼顾长程依赖与局部细节。

**🔧 技术方法**

技术手段包括：状态空间模型（SSM）与Mamba/SSD机制、双向选择性扫描、权重共享的参数化、MaskConv1D和MaskBiWSSD模块的层级骨干、Trident检测头、Soft‑NMS后处理以及多尺度特征融合。

**📊 数据集**

实验使用公开的眼科手术稠密动作定位基准OphNet，涵盖52个阶段和107个操作级动作。

**📈 对比分析**

与ActionFormer、TriDet、DyFADet、CLTDR‑GMG、ActionMamba等基线进行比较，OphBiWSSD在阶段定位和操作定位均取得mAP分别为44.42%和43.08%，较之前最高基线提升约6‑7%，且保持较低参数量与推理延迟。

**⚠️ 局限性**

主要局限包括：1）标签错误（Wrong Label）占比高，导致mAP下降约6%；2）对极短（Extra‑Small）动作的检出率低，缺乏足够的视觉证据；3）当前模型对高频细粒度手术交互仍存在边界模糊，可通过改进分类解码器和局部感知模块进一步提升。

---

## 124. Informational Help-Seeking on Reddit Did Not Decline After ChatGPT

**arXiv ID:** 2609.12447 | [PDF](https://arxiv.org/pdf/2609.12447v1)

**作者:** Hazem Ibrahim `[一作]` (New York University Abu Dhabi), Yasir Zaki `[通讯]` (New York University Abu Dhabi)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究ChatGPT发布后 Reddit 各类问答社区的发帖量变化，探究是否有人转向 AI 进行提问。

**💡 创新点**

首次在同一平台使用同类非易被 AI 替代的控制社区，并在假设无效期间跑 placebo 检验，以揭示先前结果差异的根源；同时结合 AI 文本检测评估机器生成内容对计数的潜在影响。

**🔧 技术方法**

两步差分回归（difference‑in‑differences）与 AI 文本检测器（Fast‑DetectGPT 与 Binoculars）以及 LLM 生成文本的校准。

**📊 数据集**

来自 Arctic Shift 公开存档的 Reddit 月度发帖计数（151 个子社区，含 26 个信息性、29 个人际支持、6 个好奇社区、90 个对照爱好社区）以及 2021–2025 年的帖子和评论文本样本。

**📈 对比分析**

采用与控制社区比较的两段时间差分设计；在 6 个月窗口中，信息性社区发帖量无下降（+5.1%），误差区间排除超过 3.4% 的下降；在更长窗口和检测器分析中，机器生成内容对信息性社区的贡献不足 3–4%，未掩盖任何显著下降。

**⚠️ 局限性**

局限包括仅适用于 Reddit 公开匿名社区，未覆盖非英语或小型子社区；AI 检测器对部分 LLM 敏感且无法捕捉所有机器文本；控制社区本身也存在漂移，导致 Δ（信息性 vs 人际支持）难以解释；长窗口受 Google 许可与 AI 预览等事件干扰。

---

## 125. Competence-Gated Pooling of Language Models and Priors for Event Forecasting

**arXiv ID:** 2609.12101 | [PDF](https://arxiv.org/pdf/2609.12101v1)

**作者:** Aditi Tiwari `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了基于域级路由的语言模型混合门控方法，用以判断模型相对于已有外部预测的边际价值并进行预测混合；

**💡 创新点**

提出“相对竞争力”目标，推导了Brier损失下的路由收益身份，开发了闭式域级权重估计与收缩的competence gate，展示其优于全局混合的性能；

**🔧 技术方法**

使用闭式线性池化、岭回归权重估计、域级权重收缩、单调校准、Brier评估及交叉验证等技术；

**📊 数据集**

利用ForecastBench二元问题集合（2357个），包含市场、群体、统计及结构化来源；并构造泄漏安全的ARIMA时间序列先验、FRED、ACLED、Wikipedia、DBnomics等子集；

**📈 对比分析**

与基线（外部预测、恒定基率）、全局两源/多源线性池、全局单纯形池、逻辑回归堆叠等在5折交叉验证中比较；competence gate的Brier得分从0.0771降至0.0732，提升0.0039且显著；在泄漏控制和时间序列先验子集亦显著；在强实时市场子集无显著提升；

**⚠️ 局限性**

需要已解决问题的历史数据，域划分可能过粗，未利用检索，门控对新域泛化不确定；未考虑延迟、成本，仅在事件预测上评估；内部自信度信号效果有限。

---

## 126. Accelerating the Local Push Primitive for PageRank Computation

**arXiv ID:** 2609.12076 | [PDF](https://arxiv.org/pdf/2609.12076v1)

**作者:** Guanyu Cui `[一作]` (Renmin University of China), Mingji Yang `[通讯]` (Renmin University of China)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种改进的局部PageRank推送算法（active‑set 方法），在无向图上实现了从原来的 O(1/α) 时间到 O(1/√α) 的加速，并首次给出了 ℓ1 正则化 PageRank 问题的加速求解 (O(1/(√α ρ)))；此外，将该原语用于局部聚类、单点 PageRank 估计和有效电阻估计等应用，显著提升了相关算法的时间复杂度。

**💡 创新点**

创新点在于：① 通过将激活阈值与停止阈值分离，并采用潜能函数分析，证明活跃集扩展次数被限制在 O(1/√α)，从而实现局部推送的时间改进；② 将该改进原语迁移至 ℓ1 正则化 PageRank，解决了 Fountoulakis‑Yang 提出的加速问题；③ 通过新的 PageRank‑电阻关系及其截断分析，构造了更快的有效电阻估计算法。

**🔧 技术方法**

使用的技术主要包括：活跃集算法框架、SDD（对称正定稀疏系统）求解器、潜能函数和能量范数分析、激活/停止阈值分离策略、正则化目标的潜能函数、随机化 SDD 求解器的失败概率控制、以及多阶段截断和随机游走采样。

**📊 数据集**

论文未给出具体实验数据集，主要以理论分析为主；若有实验，预计使用常见的社交网络、路网或电路网络等无向图作为测试基准。

**📈 对比分析**

在与原 ACL local push 方法的对比中，时间从 O(1/α) 缩短到 O(1/√α)，导致本地聚类的时间由 O(1/ϕ²) 降到 O(1/ϕ)；单点 PageRank 估计从 O(1/α) 提升到 O(1/α^{3/4})；有效电阻估计的复杂度从 O(L^{7/3}/ϵ^{2/3})/O(L^{5/2}/ϵ√min(vol(s),vol(t))) 降到 O(L^{4/3}/ϵ^{2/3})/O(L^{7/4}/ϵ√min(vol(s),vol(t)))，在多种场景下表现出明显加速。

**⚠️ 局限性**

局限性包括：① 仍然依赖高效的 SDD 求解器，导致实现复杂度较高；② 对非常小的 α（或 ρ）时的常数项和对数因子可能较大；③ 目前只能针对无向图，针对有向图的推广尚未完成；④ 论文未提供实验验证，理论证明虽严谨但实际性能需进一步评估。

---

## 127. CRFCAN: A Complex-Valued Cross-Domain Residual Network for Joint Channel and Phase Noise Estimation in Sub-THz OFDM Systems

**arXiv ID:** 2609.12244 | [PDF](https://arxiv.org/pdf/2609.12244v1)

**作者:** Ruilin Wang `[一作]` (University of Victoria), Xiaodai Dong `[通讯]` (University of Victoria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 CRFCAN，一种复值跨域残差网络，专门用于 sub‑THz OFDM 系统的联合信道与相位噪声估计，并实现了单次前向推理。

**💡 创新点**

创新点包括：① 将 FFT/IFFT 嵌入残差组形成时频交叉学习，能同时捕捉频域信道与时域相位旋转；② 设计了复值残差通道注意力块（CRCAB）和相位旋转注意力块（PRCAB），使网络对复数信号与相位噪声的物理关系具有显式建模；③ 在 PN 输出尾部加入软归一化约束，提升估计的物理一致性与鲁棒性；④ 采用不确定性加权的多任务损失与循环热身/余弦退火学习率，保证端到端训练的稳定性。

**🔧 技术方法**

使用的技术包括：复值卷积与 CReLU 激活、通道注意力机制、FFT/IFFT 交叉映射、相位旋转残差块、软归一化 PN 输出、动态不确定性权重损失、循环热身与余弦退火学习率调度。

**📊 数据集**

训练数据基于 3GPP TR38.803 RAN4 的相位噪声模型，模拟 100 GHz 频段、3GPP TDL‑C 信道以及 OFDM 参数；在 RAN1 Set1/Set2 的未见模型下进行无细调的泛化测试。

**📈 对比分析**

与传统迭代 LS、级联多网络、端到端 DeepSRX 等基线在相同信道、PN 与 OFDM 设置下进行比较。CRFCAN 在信道 NMSE 与 PN MSE 上显著低于基线，BER 与 EVM 在高 SNR 区域更低，甚至在部分场景下超过完美信道基线，且在未见 PN 模型时仍保持稳健性能。

**⚠️ 局限性**

局限性包括：在极低 SNR 时性能与传统方法相近；需要大量仿真样本训练；模型参数约 470 万，尽管推理复杂度固定，但仍较大；在极端快速相位噪声或多径极端情况下尚未充分验证；缺乏实际硬件实现与实时评估。

---

## 128. Differential Privacy Meets Fixed Parameter Tractability: Algorithms and Lower Bounds

**arXiv ID:** 2609.12508 | [PDF](https://arxiv.org/pdf/2609.12508v1)

**作者:** Pritish Kamath `[一作]` (Google Research), Pasin Manurangsi `[通讯]` (Google Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在隐式表示框架下，允许编码器采用固定参数可解（FPT）时间，针对多种组合优化问题提出了新的 ϵ-差分隐私近似算法。

**💡 创新点**

创新点在于：①将隐式表示模型推广到 FPT 编码器，突破多项式时间下的近似界限；②利用隐私机制和参数化技巧实现 1+O(log d / ϵ)-逼近；③在 Gap‑ETH+Advice 设定下给出表示无关与表示相关的下界，证明隐私与近似的本质冲突。

**🔧 技术方法**

核心技术包括：噪声度采样（Noisy Degree Sampling）和 Generalized AboveThreshold 机制；隐私加权的颜色编码与子集表示；以及利用隐私保护引导的“覆盖”技术将隐私问题转化为无约束求解。

**📊 数据集**

论文没有在公开数据集上进行实验，而是以理论证明为主，提供了多种图结构（如稀疏图、d‑uniform 超图）的理论分析。

**📈 对比分析**

与传统的多项式时间 DP 方案相比，新的 FPT 编码器在 Vertex Cover、d‑Hitting Set、H‑Packing 等问题上实现了更优的近似比（从 2 降到 1+O(log d/ϵ)），同时保留了隐私保证；但在某些表示下仍需 ϵ≥Ω(log n) 才能获得常数近似。

**⚠️ 局限性**

局限性在于：①对 ϵ 的大小仍有严格要求（低 ϵ 时下界强）；②仅在特定表示（如排列、颜色、子集）下给出上界；③对非图结构或更一般的 FPT 技术扩展仍未解决；④在实际数据集上的可实现性与效率未被评估。

---

## 129. Decentralized Evolution of Hexapod Gaits with Independent Leg Controllers

**arXiv ID:** 2609.12400 | [PDF](https://arxiv.org/pdf/2609.12400v1)

**作者:** Gary B. Parker `[一作]` (Connecticut), Jim O'Connor `[通讯]` (Connecticut)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种对六足机器人每条腿使用独立进化的分散式步态生成框架。

**💡 创新点**

通过完全去中心化的进化过程实现每条腿独立学习，从而获得更快收敛、更高稳定性和对腿失效的鲁棒性。

**🔧 技术方法**

采用中心模式生成器（CPG）与遗传算法（GA）优化每条腿的振幅、相位和偏移参数。

**📊 数据集**

使用 Webots 仿真环境中的 Mantis 六足机器人模型进行实验，收集仿真距离数据作为评估。

**📈 对比分析**

与协同共进化方法对比，去中心化方法在有效代数下收敛更快、最终距离更大，并在腿失效实验中表现出更高适应性。

**⚠️ 局限性**

主要局限在于计算成本高，难以在机器人上实时运行，且仅在仿真中验证，实际硬件表现待进一步验证。

---

## 130. A Deployable Architecture for Robot-Mediated Tasks (DART): Evaluation in Socially Assistive Robot-Guided Cognitive Behavioral Therapy Exercises

**arXiv ID:** 2609.12349 | [PDF](https://arxiv.org/pdf/2609.12349v1)

**作者:** Mina Kian `[一作]` (University of Southern California), Maja J. Matarić `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了DART架构，利用低成本社会辅助机器人（Blossom）与基于Web的应用和云基础设施相结合，支持可视化内容、用户输入、远程计算与持久化数据，从而实现机器人在CBT作业练习中的长期、可独立使用的交互系统。

**💡 创新点**

创新点在于将机器人、Web端与云端职责分离，形成一种模块化、可扩展的分布式体系，使低成本机器人无需集成大屏或高性能本地计算即可提供丰富交互、可持续部署、远程维护与安全数据管理，并通过参与式设计迭代提升体验。

**🔧 技术方法**

使用的技术包括：基于JavaScript/React的Web应用；AWS Amplify、IoT Core、DynamoDB、S3等云服务；MQTT over TLS/secure WebSockets进行机器人与Web端通信；Raspberry Pi驱动Blossom机器人；Amazon Polly语音合成；OpenAI GPT‑4o用于对话生成；Fitbit传感器收集生理数据。

**📊 数据集**

数据集为本研究收集的实验数据：103名实验室参与者（自评焦虑≥5分）和4名家庭长期使用者，涵盖任务完成记录、情绪量表（SNRS‑11、STAI‑SF、PANAS‑SF）、使用体验问卷（SUS）、访谈记录及Fitbit心率等。

**📈 对比分析**

通过在实验室对比四个测量时点的心理状态量表得到显著改善（压力、焦虑、负向情绪均降低，正向情绪升高，p<0.001），SUS得分分别为78.89（实验室）和87.5（家庭），表明系统易用且在单次交互中可产生正向情绪效益；但由于缺乏对照组，无法单独评估CBT内容、机器人或Web端的独立贡献。

**⚠️ 局限性**

局限性包括：实验室无对照或仅网站对照；家庭样本规模小且人群单一，缺乏多样性；系统依赖稳定的互联网和云服务，网络不稳定时易出现同步或数据上传问题；未实现语音输入，限制了自然交互；无法分离机器人体现效应与Web交互效应的具体影响。

---

## 131. Opening the Strategic Pandora Box: Conditional Transaction Mechanisms

**arXiv ID:** 2609.12197 | [PDF](https://arxiv.org/pdf/2609.12197v1)

**作者:** Yonatan Sompolinsky `[一作]` (Harvard University), David Parkes `[通讯]` (Harvard University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对离线用户的条件交易引擎（CTE）框架，并设计了在每一次调用中确定读取条件与写入顺序的机制（CTM）。

**💡 创新点**

创新点在于：①将CTM建模为“Strategic Pandora”问题，采用折扣版Weitzman指数；②提出两种机制——RW（已知盒子价值）和RWSP（报告值），并将其视为先验无关的机制设计；③引入“无投注收入”（NBR）及其动态版本作为公平、可比较的收入基准。

**🔧 技术方法**

使用理论分析与机制设计技术：折扣Weitzman排序、先验无关的支付规则、纯均衡与混合均衡分析、以及对NBR的凸包与范围约束证书。

**📊 数据集**

本工作主要为理论性，无使用真实数据集，所有结果基于数学证明与模拟案例。

**📈 对比分析**

通过与动态NBR基准比较，证明在满足一定竞争与竞争深度假设下，RW与RWSP在任何纯均衡中均能获得相当于最优机制常数比例（如α(1‑δ)² 或 αβ(1‑δ)³）的收入；实验性数值示例显示其收益与最优收益相比保持在可接受范围内。

**⚠️ 局限性**

主要局限在于：纯均衡存在性未在所有报告域上完全保证；需要较高的竞争深度和资格条件；机制对动态信息获取与持续报错的假设比较强；对实际链上部署的可扩展性与网络延迟影响未做评估。

---

## 132. Hieronym: Leveraging Hierarchical Multi-Source Information for Function Renaming in Stripped Binary

**arXiv ID:** 2609.12457 | [PDF](https://arxiv.org/pdf/2609.12457v1)

**作者:** Xiaoling Zhang `[一作]` (Zhongguancun Laboratory), Dan Li `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对去除符号表的二进制文件，提出了一种基于生成式大语言模型的函数重命名框架 Hieronym。

**💡 创新点**

创新点包括：① 采用分层函数与语句级别的总结驱动域适配，弥补 LLM 对二进制语义的缺口；② 通过融合全局二进制上下文、局部调用上下文和目标函数的多源信息，构建更完整的语义表示；③ 设计双层评估框架（token 级和名称级）提升评估可靠性。

**🔧 技术方法**

核心技术是基于 LLM 的生成式命名，结合低秩适配 LoRA、结构化总结生成、语义对齐与多源信息融合，并采用 Qwen3-Coder 等模型进行语句总结。

**📊 数据集**

使用四种架构（x64、x86、ARM、MIPS）、四个优化级别（O0–O3）的自构建二进制数据集，以及从 VirusShare 采集的带符号表的真实恶意样本做泛化验证。

**📈 对比分析**

与 SymLM、XFL、BLens、SymGen 等 SOTA 方法对比，Hieronym 在 token 级精度提升 50.12%、召回 41.75%、F1 45.10%，名称级准确率提升 79.94%，在不同架构、优化级别以及恶意样本上均保持强泛化性能。

**⚠️ 局限性**

主要限制包括：依赖特定 LLM 基础模型；语句抽取规则缺乏客观评估；数据集中的反编译质量和函数名噪声；评估者选择可能引入偏差；推理速度和成本相对较高。

---

## 133. Learning Symbolic Constraint Representations from Examples: A Neuro-Symbolic Approach

**arXiv ID:** 2609.12267 | [PDF](https://arxiv.org/pdf/2609.12267v1)

**作者:** Nassim Belmecheri `[一作]` (Simula Research Laboratory), Helge Spieker `[通讯]` (Simula Research Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种神经符号框架，用Transformer神经网络模拟用户或Oracle的行为，并与符号学习器结合，实现自动约束获取。

**💡 创新点**

创新点在于：①将Transformer训练为局部约束级的神经Oracle（TO_3），能够高效模拟人类约束检验；②与全量搜索符号学习器TRAC配合，形成鲁棒的约束获取流程。

**🔧 技术方法**

技术包括Transformer架构（跨变量注意力、值交互矩阵、依赖得分门控）、符号学习器TRAC、全量搜索、偏差约束集、以及基于Transformer的Oracle训练与推断。

**📊 数据集**

使用的实验数据集涵盖经典谜题与实际调度问题：Sudoku、JSudoku、LSquare、JobShop、Murder、Zebra、Rand122、Rand495、ExamTT、NurseR；每个数据集均按80/20拆分训练/测试。

**📈 对比分析**

与传统CA方法（如Conacq、其他查询最小化框架）对比，TRAC在查询数、平均查询时延和总获取时间上表现优异；Oracle TO_3在准确率、召回率、精确度均接近100%，整体组合（TRAC/TO_3）在所有指标上达标，尤其在稀疏解问题上表现突出。

**⚠️ 局限性**

局限性在于：①全量搜索在极大规模领域可能不可行；②需要足够多且均衡的标注样本来训练Oracle；③在增量学习器中，Oracle误判可能导致误差累积。

---

## 134. LoRA-RC: Reservoir Computing with Low-Rank Adaptation

**arXiv ID:** 2609.12327 | [PDF](https://arxiv.org/pdf/2609.12327v1)

**作者:** Wenbin Wan `[一作]` `[通讯]` (University of New Mexico), Wenbin Wan (University of New Mexico)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种低秩自适应reservoir computing（LoRA‑RC），通过在线调整低秩核心矩阵来应对系统漂移，同时保持模型的增量输入状态稳定性（δISS）。

**💡 创新点**

创新点在于：①将低秩参数化与谱范数投影相结合，保证每一步更新的递归矩阵仍落在可证收敛的合同集；②引入低通滤波器控制递归更新的步长；③给出统一的δISS收敛率与增益，并证明在整个在线更新路径上保持不变。

**🔧 技术方法**

使用技术包括低秩LoRA参数化、投影梯度下降、谱范数投影、第一阶低通滤波、增量输入状态稳定性分析及其理论证明。

**📊 数据集**

实验数据集为Lorenz‑63混沌系统：训练期700步（ρ=28），漂移后ρ变为40，加入高斯噪声。

**📈 对比分析**

与固定RC和仅读出自适应RC比较。LoRA‑RC在漂移后RMSE比固定RC低56%，比读出自适应低51%，且前漂移误差也最低。消除投影或滤波的消融实验显示投影是保证稳定性与性能的关键。

**⚠️ 局限性**

局限性：仅在单一漂移场景（ρ=28→40）验证；未测试更大或多样化漂移；仅评估预测误差，未涉及闭环控制；低秩维度r和投影半径等超参数需经验调优，未提供全局最优性分析。

---

## 135. OneLA: Scaling Linear-Attention Decoding to Large Beams in Generative Recommendation

**arXiv ID:** 2609.12399 | [PDF](https://arxiv.org/pdf/2609.12399v1)

**作者:** Xiangrui Yang `[一作]` (University of Hong Kong), Yiming Qiu `[通讯]` (University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种面向大束生成推荐的线性注意力解码框架，利用共享上下文、紧凑的转移记录和轻量级祖先索引，实现对大束状态的高效管理与重播；

**💡 创新点**

创新点在于将所有束的状态抽象为单一的后预填充共享状态与可追加的转移记录，避免为每束单独材料化完整的递归状态，并通过祖先索引解耦束的逻辑演化与物理记录，配合融合GPU核实现全芯片重播；

**🔧 技术方法**

核心技术包括Gated Delta Network（GDN）线性注意力、GDN Transition Record（GTR）压缩存储、祖先索引追踪束树、以及跨束共享上下文的融合GPU核；

**📊 数据集**

实验基于工业级生成推荐工作负载，使用Qwen3.5 0.8B模型（6个全注意力层+18个GDN层），对比不同提示长度（1K/5K）和输出长度（3–7个令牌）；

**📈 对比分析**

与vLLM、SGLang、FlashInfer、TensorRT-LLM等主流框架的FullState、ReplaySSM路径比较，框架在全模型解码中实现1.54–2.46×的加速；在单个GDN注意力算子上，平均可获得40–45×的吞吐提升、最多56.5×的显存占用下降，显存读写量降低至原来的1/70左右；

**⚠️ 局限性**

主要限制包括：仅针对GDN线性注意力设计，难以直接迁移到传统KV或其他线性注意力实现；实现高度依赖GPU对共享内存与重播的支持，CPU端仍需额外逻辑；对极端小束宽或非常短提示的场景效果未充分验证。

---

## 136. Chain-SLAM: Globally Consistent Backend for Multi-Session LiDAR SLAM via Chained Loop Closure

**arXiv ID:** 2609.12221 | [PDF](https://arxiv.org/pdf/2609.12221v1)

**作者:** Zhiheng Li `[一作]` (New York University), Chen Feng `[通讯]` (New York University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个实时多会话LiDAR‑惯性SLAM后端，支持在线多会话地图对齐与重用，利用链式闭环实现全局一致性。

**💡 创新点**

链式闭环机制通过邻接图传播几何约束，既提升长期一致性，又无需动态对象移除；同时在单一因子图中完成多会话联合优化。

**🔧 技术方法**

融合FAST‑LIO2前端、iSAM2增量因子图优化、ICP、FPFH+RANSAC以及GNSS邻接检索。

**📊 数据集**

在MARS（城市驾驶）与NCLT（移动机器人）两大多会话LiDAR数据集上验证。

**📈 对比分析**

与Fast‑LIO2及LAMM对比，链式闭环版在轨迹精度、几何一致性和视角重建上分别提升5–10倍的平面漂移、降低1.5–3倍的旋转误差，且保持实时（≈2 Hz）。

**⚠️ 局限性**

目前地图合并依赖GNSS，可改用LiDAR基准识别以适配GNSS受限环境。

---

## 137. HypoKG: Evidence-Disciplined Biomedical Hypothesis Generation Beyond Endpoint Knowledge

**arXiv ID:** 2609.12260 | [PDF](https://arxiv.org/pdf/2609.12260v1)

**作者:** Dominic Okonkwo `[一作]` (University of Georgia), Ismailcem Budak Arpinar `[通讯]` (University of Georgia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一个基于KEGG、Rhea和UniProt整合的生化知识图谱，采样550条多跳路径，并用六种大型语言模型在四种提示条件下生成13,200条医学假设。

**💡 创新点**

首次在同一KG中区分端点发现与中间路径结构的作用，提出“证据纪律化推理”，并通过“路径打乱”实验证明模型确实利用了路径顺序。

**🔧 技术方法**

利用大型语言模型（Claude Sonnet、GPT‑4o、Llama‑3.3‑70B、Qwen‑3‑235B、BioMistral‑7B、MedGemma‑27B）进行提示式生成，并用LLM‑judge和三位PhD专家进行五维评分。

**📊 数据集**

数据集为融合后的KEGG–Rhea–UniProt人类化学通路图谱（17,449节点、31,709条边），以及从该图谱抽取的550条源–疾病路径。

**📈 对比分析**

对比四种提示条件，发现“端点仅提示”(C4)获得最高总分，而“完整路径提示”(C2/C3)在“证据比例性”上显著优于C4，且在路径打乱控制中证据比例性显著下降，表明模型真正利用路径结构。

**⚠️ 局限性**

局限包括：评价依赖LLM-judge和少量专家样本，证据比例性评判主观性高；仅覆盖人类化学通路，未纳入药物副作用或表型知识图谱；模型在长路径或低能力模型上表现不佳，可能限制推广。

---

## 138. Test-Driven Approaches to Software Engineering with Large Language Models: A Survey of Phases, Tasks, and Agent Skills

**arXiv ID:** 2609.12012 | [PDF](https://arxiv.org/pdf/2609.12012v1)

**作者:** Yunhao Liang `[一作]` (Chengdu Institute of Computer Applications, Chinese Academy of Sciences), Shiwen Ni `[通讯]` (Artificial Intelligence Research Institute, Shenzhen University of Advanced Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文对测试在大型语言模型和软件工程代理中的作用进行了结构化的范围调查，整合了87个研究和支持记录，探讨了测试如何影响决策。

**💡 创新点**

创新点在于将测试驱动开发（TDD）与其他测试驱动机制区分开来，并提出了一种机制分类法，比较了不同软件工程任务中的测试角色和效果。

**🔧 技术方法**

使用了文献综述和协议级别的提取方法，结合了多种软件工程任务的分析，包括代码生成、修复、翻译等。

**📊 数据集**

整合了87个研究记录，涵盖了多种软件工程任务的数据集，具体数据集未详细列出。

**📈 对比分析**

通过比较不同任务的测试角色和效果，发现测试的可用性、有效性、反馈使用和评估独立性是影响性能的关键因素，测试通过率并不能单独证明行为等价性。

**⚠️ 局限性**

限制在于现有的研究主要集中在测试的影响上，而对测试的具体执行顺序、反馈机制等中间过程的观察和评估仍然不足。

---

## 139. LatentVerse: A Framework for Understanding Shared and Modality-Specific Information in Multimodal Latent Representations

**arXiv ID:** 2609.12364 | [PDF](https://arxiv.org/pdf/2609.12364v1)

**作者:** Majd Alafrange `[一作]` (Broad Institute of MIT and Harvard), Mahnaz Maddah `[通讯]` (Broad Institute of MIT and Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并实现了LatentVerse框架，用于统一分析单模态与多模态潜在嵌入的质量与结构。

**💡 创新点**

首次将多模态嵌入按共享与模态特定信息分解，并将多项诊断指标集成到同一可视化报告中，实现了对潜在空间的系统化、可解释化评估。

**🔧 技术方法**

结合多种评估指标（聚类性、可分离性、可表达性、下游可预测性与鲁棒性）、MultiLoReFT信息分离技术、Web交互可视化、CLI脚本化流程以及LLM生成的自然语言摘要。

**📊 数据集**

在人工合成模拟数据、英国生物样本库（UK Biobank）心电图(ECG)与心脏磁共振(cMRI)嵌入，以及其他真实生物医学嵌入上进行验证。

**📈 对比分析**

通过控制实验验证指标能够准确捕捉已知结构；在真实数据中，报告显示ECG嵌入对PQ间隔的R²显著高于cMRI，cMRI在肺动脉高压判别中AUROC更优；用户研究表明工具在实用性与解释性上得到积极评价，但未提供大规模基准对比。

**⚠️ 局限性**

目前仍是研究原型，缺乏大规模并发与大数据处理的工程化支持，验证性（可追溯性）有限，用户研究样本规模与领域覆盖有限。

---

## 140. HSI-Road Relabeled: Surface-Aware Road-Scene Segmentation

**arXiv ID:** 2609.12151 | [PDF](https://arxiv.org/pdf/2609.12151v1)

**作者:** Imad Ali Shah `[一作]` (University of Galway), Brian Deegan `[通讯]` (University of Galway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对公开的 HSI‑Road 数据集进行六类表面标签的手工重新标注，构建 RGB‑to‑NIR 配准管道，并在四种输入配置下（RGB_ori、RGB_reg、NIR、RGBN_stk）评估六种语义分割模型的性能。

**💡 创新点**

①提供第一份六类表面级别的 HSI‑Road 重新标注；②提出 RGB‑to‑NIR 精细配准流程；③在匹配分辨率下对单模态与多模态输入进行细粒度对比，揭示分辨率与模态互补的影响。

**🔧 技术方法**

使用 ORB/SIFT 与 RANSAC + ECC 的配准方法；深度学习分割模型包括 UNet、UNet‑CBAM、DeepLabV3+、SegFormer、UPerNet（EfficientNet‑B0 与 MiT‑B0 编码器）；混合损失（交叉熵 + Dice），混合精度训练。

**📊 数据集**

使用公开的 HSI‑Road 数据集（RGB 704×1280 + 25 通道 NIR 192×384），并在此基础上创建了六类表面标签，按 70/15/15 多标签分层划分。

**📈 对比分析**

通过在同一分辨率（192×384）下计算 mIoU 与 mF1，比较四种输入配置与六种模型的性能。结果显示：RGB_ori 取得最高整体分数，但匹配分辨率时 RGBN_stk 在六类上优于 NIR（所有模型）且大多数模型优于 RGB_reg，尤其在 Water 类表现最为显著；整体上 UPerNet_MiT‑B0 获得最高 mIoU，SegFormer 在参数量与推理速度上最小。

**⚠️ 局限性**

限制：①模型全部从零开始训练，未利用 HSI 专属预训练或视觉基础模型；②RGB‑NIR 仅采用通道堆叠，未探究更高级的融合与降维方法；③配准与分辨率的效应混合，无法单独评估；④未评估不同分辨率下的边界复杂度和类别分布对结果的影响。

---

## 141. Synthetic TLX: Forecasting Human Workload Using Agent Simulation

**arXiv ID:** 2609.12273 | [PDF](https://arxiv.org/pdf/2609.12273v1)

**作者:** Tzu-Sheng Kuo `[一作]` (Carnegie Mellon University), Michael Terry `[通讯]` (Google DeepMind)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 Synthetic TLX，一种通过 LLM 代理模拟预估 NASA TLX 工作负载的前瞻性方法，并通过三项实验（邮件草拟、网站导航、对话）比较代理与人类评分，同时展示了 MailLoad、WebSim、SkillUX 三个应用示例。

**💡 创新点**

创新点在于：①首次将工作负载评估从传统的事后问卷转为前瞻性预测；②证明主动任务模拟结合人类角色提示可使代理评分与人类高度一致；③提供可用于设计迭代、工作量预估和人机交互的实用工具。

**🔧 技术方法**

技术手段包括：使用 Gemini 3.1 Pro LLM 进行四种提示策略（AI/观察、AI/模拟、人/观察、人/模拟）；对三类任务进行代理执行（邮件写作、浏览器操作、对话交互）；计算并输出 NASA TLX 六维度评分；对代理输出进行相关性和 MAE 校准分析。

**📊 数据集**

数据集与实验：人类参与者 162 名（按任务分组 54 名），每任务包含 3 组 18 名，完成 9 种情境；代理在每组重复三次模拟，共 18 次估计；使用官方 NASA TLX 问卷评分与代理生成的评分进行对比。

**📈 对比分析**

比较方法：对人类与代理的 54 对平均分进行 Pearson 相关系数计算，MAE 评估误差；P4（人/模拟）策略在所有任务中相关系数最高（0.648–0.834），校准后 MAE 显著降低；工作负载对比分析显示代理能识别任务增难，但对不同负载来源（本质 vs 外在）判断存在倒置。

**⚠️ 局限性**

局限性：①代理与人类在识别工作负载来源上存在差异，尤其文本任务低估、网页任务高估；②实验仅使用单一 LLM（Gemini 3.1 Pro），结果可能不具普适性；③缺乏对个体差异的评估，代理对部分受试者可能更合适；④需进一步研究提升代理对多模态任务的感知与理论推理。

---

## 142. When Connected Does Not Mean Similar: Charting the Homophily Boundary of SNAP-KG for Streaming Entity Integration

**arXiv ID:** 2609.12356 | [PDF](https://arxiv.org/pdf/2609.12356v1)

**作者:** Jui-Chien Lin `[一作]` (Rensselaer Polytechnic Institute), Oshani Seneviratne `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了 SNAP-KG 在异质性（heterophilous）知识图谱中的性能，并通过测量每个视图的边缘同质性（edge homophily）来探究其在无图访问、无重训练下的流式实体分配能力。

**💡 创新点**

首次系统评估了 SNAP‑KG 在多视图异质图上的表现，发现其优异性能仅依赖于至少存在一个同质化（homophilous）视图；提出在异质图中需要专门的异质性感知教师来蒸馏模型，并为后续研究给出路线图。

**🔧 技术方法**

使用了多视图图神经网络（GNN）+ Transformer 编码器进行预训练，随后通过轻量级多层感知器（MLP）投影器实现无图推理；实验中还对比了多种传统和异质性聚类基线（AGE、O2MAC、BMGC、DuaLGR）。

**📊 数据集**

实验数据集包括原始 SNAP‑KG 评估的五个多视图图谱（ACM、DBLP、IMDB、YELP、MAG）和三个位点规模较小的异质图（Texas、Wisconsin、Chameleon），以及一大规模 240 万节点的 OGB‑WikiKG2。

**📈 对比分析**

与传统全图训练的转导式聚类基线（AGE、O2MAC、BMGC、DuaLGR）和 SNAP‑KG 的单视图消融结果进行对比；在同质化图上，SNAP‑KG 能达到 90%+ 的聚类准确率；在异质化图上，准确率骤降至 30% 以内，且表现与基线相似，说明同质性假设是关键瓶颈。

**⚠️ 局限性**

局限性在于所有采用同质性假设的模型（无论是转导式还是诱导式）在完全异质化的图谱上性能显著下降；缺乏针对异质关系的教师网络，导致投影器无法正确捕获节点间的异质相似性。

---

## 143. A First-Principles Evaluation of Graph-Based Network Intrusion Detection Systems

**arXiv ID:** 2609.12263 | [PDF](https://arxiv.org/pdf/2609.12263v1)

**作者:** Rui Zhao `[一作]` (University of Virginia), Wajih Ul Hassan `[通讯]` (University of Virginia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出并实现了一个名为GIDS‑Eval的模块化评估框架，能够将图网络入侵检测系统拆解为六个可互换的阶段，并在同一协议下统一重实现八种代表性系统进行对比实验。

**💡 创新点**

创新点在于将评估流程解构为可归因的阶段，揭示传统报告中隐藏的评估缺口；并引入无图编码的轻量基线GIDS‑Lite，证明复杂模型并非总能提升检测质量。

**🔧 技术方法**

采用图神经网络（GCN、GraphSAGE）、时序模块（GRU、RNN）、图重建与预测目标，以及无编码器的浅层预测器等技术；框架基于PyTorch与PyTorch Geometric实现。

**📊 数据集**

主要使用了四个公开网络安全数据集（Campus、CIC‑IDS‑2017校正版、CIC‑IDS‑2017原版以及另一模拟企业流量数据集），覆盖不同规模与攻击标签。

**📈 对比分析**

通过统一的预处理、时间切分和阈值校准协议，对八个系统和GIDS‑Lite进行端到端比较；实验显示GIDS‑Lite在两个数据集上获得最高的平均精度，并且推理时间比最慢系统低近575倍；其他系统的提升因数据集而异，复杂模型未必带来更好的检测。

**⚠️ 局限性**

限制在于实验仅覆盖公开数据集与已公开实现，未涵盖实时部署与解释可追溯性；对抗鲁棒性仅评估覆盖边攻击，未覆盖更广泛的攻击方式；框架的通用性需要在更多真实环境中验证。

---

## 144. Fundamental Dynamical Units for Physics-Informed Structural Inference from Perturbation Time-Series in Networked Systems

**arXiv ID:** 2609.11934 | [PDF](https://arxiv.org/pdf/2609.11934v1)

**作者:** Nima Nouri `[一作]` `[通讯]` (AstraZeneca), Nima Nouri (AstraZeneca)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于三节点“基础动力单元”（FDU）的结构先验，用以从受干扰时间序列中推断网络的符号相互作用结构，并将其嵌入物理信息化神经ODE中实现联合轨迹预测与结构识别。

**💡 创新点**

创新点包括①构造完整且可合成的FDU字典，显著压缩符号网络结构搜索空间；②依据FDU分类自动生成最小化干扰面板，实现直接与传递路径的可辨识；③通过FDU正则化的物理信息化神经ODE，将结构先验与连续时间动力学约束结合，避免传统统计方法的结构不确定性。

**🔧 技术方法**

采用的技术包括：
- 基于三节点有向符号图的枚举与标签一致化生成FDU字典；
- 通过entmax注意力实现FDU稀疏选择；
- 使用Hill型激活/抑制函数构建可解释的动力学驱动；
- 引入条件依赖的傅里叶基准函数估计开放系统环境噪声；
- 通过自适应不确定性权重与多项物理损失联合训练。

**📊 数据集**

使用的所有数据集均为人工合成：
- 3节点单体系统（512种符号三元组），
- 20节点稀疏网络（10个嵌入的符号三元组），
- 每种网络均在四个时间窗口内采样稀疏观测，并配备相应干扰面板。

**📈 对比分析**

与传统稀疏系统辨识、Granger因果、图神经ODE等方法的比较通过：
- 在单体三节点实验中实现了100%边缘检测与符号恢复，RMSE在0.12–0.21之间；
- 在20节点实验中AUROC 0.998、F1 0.93，误检率低于5%；
- 对照实验禁用环境噪声模块，AUROC骤降至0.60，验证开放系统分解的必要性。

**⚠️ 局限性**

局限性包括：
- 仅在人工合成数据上验证，缺乏真实实验验证；
- Hill动力学参数、衰减率等物理常数被固定，未与结构共估；
- 计算复杂度随网络规模呈$O(N^3)$增长，尽管entmax稀疏化缓解，但大规模应用仍需更高效实现；
- 对噪声、测量不规则性及模型假设偏差的鲁棒性尚未系统评估。

---

## 145. MoPA: Coordinated Mobile Manipulation via Subsystem-Specific Perception Alignment

**arXiv ID:** 2609.12081 | [PDF](https://arxiv.org/pdf/2609.12081v1)

**作者:** Guangyu Chen `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**通讯引用:** 8160 | [OpenAlex ID](https://openalex.org/A5012419026)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于双感知流和感知-动作适配的移动操纵框架，分别为移动底盘和机械臂生成子系统特定的感知表示并在动作层面实现协同控制。

**💡 创新点**

创新点在于显式实现子系统级感知-动作对应，使用双感知流从共享的视觉‑语言上下文中抽取独立的移动与操纵查询，利用结构化的Mixture‑of‑Transformers在每一层动态更新查询与动作，并通过耦合条件流匹配学习联合向量场，实现同步生成全身动作。

**🔧 技术方法**

技术包括Qwen3‑VL视觉‑语言编码、双查询银行（Manip Query、Mobile Query）、结构化Mixture‑of‑Transformers解码器、耦合条件流匹配（flow‑matching）生成器、交叉注意力屏蔽、动作与感知的层级联合更新。

**📊 数据集**

使用了 ManiSkill‑HAB（SetTable、TidyHouse、PrepareGroceries）模拟数据集以及四个真实世界任务（水果收集、颜色匹配分类、跨桌子水果搬运、微波炉取物）进行训练与评估，训练样本来自SAPIEN/ReplicaCAD/YCB。

**📈 对比分析**

与多种基线（Visuomotor、VLA、WAM）在 ManiSkill‑HAB 上比较，平均技能成功率分别达到 88.3% / 72.2% / 67.8%，均超过最强基线 4.5%–17%；在真实世界实验中平均成功率为 76.3%，比最强基线 π₀.₅ 高 12.5个百分点。

**⚠️ 局限性**

局限性包括对极端动态或复杂场景的鲁棒性尚未充分验证；在某些任务（如微波炉检索）与强基线相当；查询数量与分辨率的权衡仍需进一步探索；对大规模预训练模型的依赖导致部署成本较高。

---

## 146. Creating an Atomic User Model for Personality-Aware Large Language Model Interaction

**arXiv ID:** 2609.12086 | [PDF](https://arxiv.org/pdf/2609.12086v1)

**作者:** B. Sankar `[一作]` (Indian Institute of Science), Amogh A S `[通讯]` (Indian Institute of Science)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 Atomic User Model（AUM）的分层结构化用户模型，并构建了基于任务类型的检索式个性化流水线，使用预算限制的字段集合生成 LLM 输出；在模拟实验中验证了结构化模型与预算检索在生成风格一致性、识别度和上下文开销上的优势。

**💡 创新点**

创新点在于：① 发现并命名“人格渗透（personality seepage）”现象，说明仅存储表层偏好会导致模型无法正确处理用户隐含特征；② 设计 AUM 将人格划分为 Nucleus、Psychological、Cognitive & Experiential、Behavioural、Social 五层及交叉层，明确不同层的稳定性与可观测性；③ 将 AUM 视为检索索引，使用任务分类器、组件选择函数和基于冗余惩罚的预算检索，显著减少注入上下文；④ 在模拟实验中对检索算法（贪婪、随机、人工鱼群）进行对比，证明贪婪足够且结构化表示才是提升效果的关键。

**🔧 技术方法**

核心技术包括：基于分层结构化用户模型的字段组织；任务分类器与组件选择函数筛选相关 shell；预算检索采用带冗余惩罚的均值-多余度目标；检索后使用模板将字段嵌入 LLM 输入；评估采用 LLM 判别者（判断文本是否像用户写作）和 LUAR 词作者嵌入进行风格相似度计算。

**📊 数据集**

实验使用 16 名人工生成的模拟参与者，每人具有 Big Five z‑score、生活背景、经验种子和 idiolect 标记；每人完成 6 项写作任务（如请求延期、拒绝邀请、公共发帖等）并在 3 种随机种子下运行；此外还使用 32‑字段、64‑字段、128‑字段等不同大小的 AUM 进行检索优化实验。

**📈 对比分析**

与基线比较：generic（仅 prompt）、preference‑notes（扁平偏好记录）以及 full（全 32 字段）对比。结果显示：k=8 预算检索在保持 2.91 的 fidelity（比 generic 提升 0.27）且识别率达 42.7%（比 25% 随机高）时，仅注入 211 tokens（占 full 915 tokens 的 23%）。随机 k‑字段和去除冗余惩罚的检索在效果上与 k=8 贪婪检索无显著差异，说明主要提升来自结构化表示而非检索策略；人工鱼群在该规模下不胜贪婪。

**⚠️ 局限性**

限制：① 只在模拟 LLM 参与者中评估，缺乏真实人类数据的验证；② 模拟参与者采用自我报告填表，可能与真实行为不一致；③ 只测试了 32 字段的子集，未探究更大模型时检索优化的有效性；④ 冗余惩罚 λ 的影响在本实验中较小，可能在不同设置下作用更显著；⑤ 评估主要基于 LLM 判别者和作者嵌入，未直接测量用户满意度或长期适应效果。

---

## 147. Partition-Invariant Tuning for 3D Scene Understanding

**arXiv ID:** 2609.12473 | [PDF](https://arxiv.org/pdf/2609.12473v1)

**作者:** Hongqiang Lin `[一作]`, Dongfu Yin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种针对大规模场景点云的高效迁移学习框架PointPiT，专门解决序列化导致的划分敏感性问题。

**💡 创新点**

创新点在于将Scene-aware Structural Adapter（SSA）与Gradient Subspace Optimization（GSO）结合，既在前向层面捕获全局场景上下文，又在后向层面选择对划分不敏感的梯度子空间。

**🔧 技术方法**

采用轻量化瓶颈投影、注意力上下文融合以及子空间投影等技术，并在预训练的Point Transformer V3（PTv3）基础上冻结大部分参数，只训练SSA和任务头。

**📊 数据集**

在ScanNet、ScanNet++、S3DIS等室内场景数据集上进行实验。

**📈 对比分析**

与全微调和多种PEFT方法（LoRA、Prefix Tuning、Adapter等）比较，PointPiT在参数量低于1%的同时，在大多数指标（mIoU、mAcc）上与全微调相当甚至更优。

**⚠️ 局限性**

局限在于仅针对场景级语义分割任务，尚未验证在其他3D任务或更大规模场景下的通用性；对序列化策略仍有一定依赖。

---

## 148. A Case-Bundle Operating Model for Coding Agents in OpenFOAM-Based CFD

**arXiv ID:** 2609.11941 | [PDF](https://arxiv.org/pdf/2609.11941v1)

**作者:** Ke Xiao `[一作]` (AI for Science Institute), Zhi X. Chen `[通讯]` (Peking University)

**通讯引用:** 259858 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建并验证了基于OpenFOAM的可重用案例包和两种操作模式（Build/Replay），用于对140个注射器几何变体进行VOF CFD模拟。

**💡 创新点**

提出了面向案例包的可审查、可复用的操作模式，并证明了在不同LLM后端下通过同一包实现高效重复计算，从而实现受控自主化。

**🔧 技术方法**

使用了通用编程代理（Codex, Pi）与LLM（GPT‑5.5、GPT‑5.4、DeepSeek V4 Pro、GLM‑5.2、Qwen3.7‑plus）、OpenFOAM‑7、Slurm作业调度、ParaView/VTK后处理、Git与Git LFS版本控制。

**📊 数据集**

采用140个基于STL的注射器几何模型作为设计变体，单个STL样本用于跨模型重放验证。

**📈 对比分析**

通过对比四个LLM后端在Pi运行时的工具调用次数、token消耗和API成本，展示了相同案例包在不同模型间均能成功完成模拟，性能差异主要体现在调用次数和成本上。

**⚠️ 局限性**

局限在于仍需人工审查关键工程决策（几何映射、网格质量、物理参数），且模型在复杂几何或极端工况下的泛化能力未得到充分验证。

---

## 149. GSO-Net: Visual State Machines for Hazardous Freight Transfer Compliance at Petrochemical Logistics Nodes

**arXiv ID:** 2609.12408 | [PDF](https://arxiv.org/pdf/2609.12408v1)

**作者:** Yu Xie `[一作]` (Nanchang Hangkong University), Zechu Ouyang `[通讯]` (Jiangxi Expressway Petrochemical Co., Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了GSO-Net数据集与基准，设计宏观步骤与微观状态双层标注，定义联合检测与帧级步骤分类两项任务。

**💡 创新点**

创新点在于：①将石化危险货物转移SOP细化为9个宏观步骤与15个微观状态的层次化语义结构；②针对实际基础设施轮询采样环境，提供稀疏视图下的视觉SOP理解评测；③提出联合检测作为核心评测，强调局部状态与全局步骤的相互约束。

**🔧 技术方法**

采用多种主流检测器（YOLOv8/9/10/11/26/30、RT‑DETR、Relation‑DETR、RF‑DETR、YOLO‑World等）和分类器（YOLOv8‑cls、YOLO26‑cls、ConvNeXt‑V2、EVA‑02、MambaOut）作为基线；使用CVAT进行手工标注，利用transformer与开放词表技术增强语义迁移。

**📊 数据集**

使用自采集的GSO‑Net数据集：50,325帧、64个物流节点，包含15种微观状态与9种宏观步骤；公开可下载（https://github.com/yuxieHarrison/GSO-Net）。

**📈 对比分析**

通过对比轻量级与强大检测器的AP_50/AP_50‑95/Scale‑aware指标，发现：小目标（如夹具接触点）AP低于5%，长尾类别如clamp_floating、discharge_disconnected几乎为0%；宏观步骤中稳定阶段易识别，转移阶段准确率不足50%；整体表现表明现有模型在稀疏轮询、细粒度状态识别和步骤一致性方面仍有显著缺口。

**⚠️ 局限性**

limitations：①仅覆盖石化卸料节点，缺乏多场景普适性；②仅采用独立帧，无连续视频时间序列标注，难以评估时序推理；③未提供规则化推理或一致性验证机制；④稀疏轮询的跨帧关联与记忆机制未被深入探索；⑤小目标与长尾类别的检测瓶颈仍未突破。

---

## 150. Cortex: Content Analysis Support Software, a Resource for Qualitative Research

**arXiv ID:** 2609.11970 | [PDF](https://arxiv.org/pdf/2609.11970v1)

**作者:** Ana Julia da Silva Soares `[一作]` (Instituto Federal de Educação, Ciência e Tecnologia do Rio Grande do Sul), Rafael Coimbra Pinto `[通讯]` (Instituto Federal de Educação, Ciência e Tecnologia do Rio Grande do Sul)

**通讯引用:** 197 | [OpenAlex ID](https://openalex.org/A5043425941)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了 Cortex 这款 Web 应用，完整实现了 Bardin 的内容分析流程（前置分析、材料探索、结果导出），并通过 RAG+LLM 自动生成索引、指标和类别。

**💡 创新点**

创新点在于：①将 Gemini LLM 与 Vertex AI RAG 结合，为研究者自动生成可追溯的索引、指标与类别；②在保持研究者解释主导的前提下，系统对 LLM 输出的原始引用不可编辑，保障可追溯性与方法学完整性；③提供完整可视化与 PDF 报告，降低手工编码成本。

**🔧 技术方法**

使用技术包括：C#/.NET Core 后端、React+TypeScript 前端、JWT 认证、PostgreSQL + Google Cloud Storage 存储、Vertex AI Gemini LLM + RAG（向量化、分块、检索）、JSON Schema 验证、PDF 生成库。

**📊 数据集**

数据来源：四位具有内容分析经验的跨学科研究者的半结构化访谈记录，以及研究者上传的多篇采访文本与参考文献（PDF/TXT）。

**📈 对比分析**

与 ATLAS.ti、NVivo、Elicit 等工具对比，Cortex 在自动化前置分析与材料探索阶段具有显著优势，节省编码时间并提供更严格的可追溯性；虽然未给出具体量化性能指标，但功能覆盖与流程完整性表明其在效率与方法学一致性上优于现有工具。

**⚠️ 局限性**

局限性包括：目前仅支持 PDF 导出；未集成音频转写与多格式导出；LLM 可能产生幻觉，需要人工核对；对大型数据集的可扩展性与实时性尚未评估。

---

## 151. Hybrid Physics-AI Framework of Body Center of Mass Dynamics from Wrist-Worn Sensors

**arXiv ID:** 2609.12304 | [PDF](https://arxiv.org/pdf/2609.12304v1)

**作者:** Shuhao Que `[一作]` (University of Twente), Ying Wang `[通讯]` (University of Twente)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个基于简化的四段运动学模型与深度学习相结合的混合AI框架，利用单手腕IMU估计全身质心加速度；

**💡 创新点**

创新点在于将物理先验与数据驱动通过三种学习策略（序列学习、同步学习1、同步学习2）融合，既保留可解释性又显著提升预测精度与鲁棒性；

**🔧 技术方法**

技术方法包括构建机器人臂式运动学模型、使用全连接网络和LSTM网络、实现三种混合学习框架，并通过余弦相似度分析训练动态与噪声鲁棒性评估；

**📊 数据集**

数据集来自10名健康志愿者在7种步态及坐到站转移任务中采集的左手腕IMU信号与Xsens MVN Link系统生成的质心加速度；

**📈 对比分析**

与纯KM、FCNN和LSTM进行对比，采用NRMSE、R²和Pearson r指标；在无噪声条件下，sim1-LSTM达3.9%~5.3% NRMSE（比KM低约45-70%），在高斯噪声下保持较高鲁棒性；sim2表现鲁棒但精度略低；

**⚠️ 局限性**

局限性包括运动学简化假设导致参数可识别性不足，物理模型在高斯噪声下易失效；混合模型在sim2中物理可解释性受限；实验仅涵盖实验室活动，尚未验证在更复杂日常场景中的泛化能力。

---

## 152. Certifying Concept Unlearning in Text-to-Image Diffusion Models

**arXiv ID:** 2609.12163 | [PDF](https://arxiv.org/pdf/2609.12163v1)

**作者:** Mansi `[一作]` (Imperial College London), Francesco Leofante `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种针对文本到图像扩散模型概念消融的可证化残留泄露评估框架，能够在整个概念邻域内给出高置信度的泄露上界。

**💡 创新点**

创新点在于首次将统计证书与概念相关嵌入方向的最坏情况分析相结合，生成覆盖连续概念空间、非经验式的泄露上界。

**🔧 技术方法**

核心技术包括基于文本嵌入的引导单元、嵌入层和像素层的两级分类器、随机化光滑化与Hoeffding界定误差、以及结合样本统计与分类器误差的联合上界推导。

**📊 数据集**

使用的数据集包括 SD-Turbo 与 SDXL-Turbo 两款扩散模型，以及 UnlearnCanvas（艺术风格）、CoProV2（NSFW）和 CelebA（名人身份）等概念评测集，辅以 reLAION 2B 的对照提示。

**📈 对比分析**

与传统基于攻击成功率（ASR）的评估相比，该方法在 36 种模型/方法/概念组合中，约 89% 的案例中证书上界高于 ASR，尤其在名人身份类中差距显著，表明 ASR 低估了残留风险。

**⚠️ 局限性**

局限性包括证书可能较宽松、引导方向可能忽略稀疏攻击提示、两级分类器的误差仍需外部校准，以及对多样化概念变化的覆盖尚未完整。

---

## 153. Beyond Argmax: A Mechanistic Study of Semantic Retention in Frozen Foundation-Model Composition for Generalized Few-Shot 3D Segmentation

**arXiv ID:** 2609.12099 | [PDF](https://arxiv.org/pdf/2609.12099v1)

**作者:** Silas Kwabla Gah `[一作]` (University of Ghana), Ebenezer Owusu `[通讯]` (University of Ghana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在冻结的基础模型组成的三维分割任务中，探究在跨源融合前保留多少语义候选（top‑k）对最终性能的影响。

**💡 创新点**

创新点在于将“语义提前崩溃”视为一个可量化的瓶颈，通过同输入、同源、同融合规则的 top‑k 介入实验，系统性诊断了在异构模型融合中早期硬决策会损失多少有用信息。

**🔧 技术方法**

使用 RegionPLC 作为稠密 3D 语义来源，SAM3（跨视图概念掩码）作为稀疏来源，统一平均融合规则，top‑k 截断（k=1,2,5,10,20,全部），以及对比最大池、几何/对数池、加权、校准等技术。

**📊 数据集**

在 ScanNet200（312 场景，200 类）和 ScanNet++（50 场景，30 类）这两个基准 3D 语义分割数据集上进行评估。

**📈 对比分析**

与传统 top‑1 决策级投票相比，全分布融合在 ScanNet200 上提高了 6.40 HM（95% CI [+5.24,+7.64]），在 ScanNet++ 上提高了 3.48 HM（95% CI [+1.64,+5.93]）；最显著的提升来自 top‑1→top‑2 的切换，后续的 top‑k 进一步提升但趋于饱和。

**⚠️ 局限性**

局限性包括：仅在推理时融合；仅评估两类源（RegionPLC 与 SAM3 或 GroundingDINO–SAM2.1 诊断）；全分布存储开销大；不同源、类空间或场景结构可能导致最佳 k 不同；未考虑自监督或少量标注微调的提升。

---

## 154. One Simple Trick for Improving the Performance of Energy-Limited Local Inference and Training

**arXiv ID:** 2609.11936 | [PDF](https://arxiv.org/pdf/2609.11936v1)

**作者:** Erik Schultheis `[一作]` (ISTA), Dan Alistarh `[通讯]` (ISTA)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在大型词表LLM训练和推理中引入分块（chunking）技术，缓解GPU功率与热量波动导致的节流，从而提升性能并降低能耗。

**💡 创新点**

发现分块不仅能减少内存占用，还能平滑瞬时功率消耗，防止动态电压/频率缩放（DVFS）引起的性能下降，这是此前未被研究的能耗优化角度。

**🔧 技术方法**

使用分块策略（如LM-head分块）、NVIDIA Nsight 系统分析、NVML 与插座功耗测量、PyTorch @torch.compile 编写的最小重现脚本，以及 vLLM 和 LLMQ 框架。

**📊 数据集**

主要实验基于 Qwen2-0.5B（896 隐藏层，152k 词表）和 8B Llama 模型的推理，采用 DGX Spark（edge GPU）和 NVIDIA L40S（数据中心 GPU）。

**📈 对比分析**

对比单块与多块（如 16 块）执行，DGX Spark 上训练时间缩短 20%，能耗下降 14%；在 L40S 上提升约 1-2%（时间 2.5%，能耗 1.3%）。

**⚠️ 局限性**

实验仅覆盖单一 edge 设备和一种数据中心 GPU，缺乏对中间平台（RTX、Jetson 等）的验证；功率测量不够精确；最佳分块大小依赖工作负载且未提供自动化选择机制。

---

## 155. Predicting Collision Cross Sections with GRACE: Geometric Residual Adduct Conditioning via Early-fusion

**arXiv ID:** 2609.12223 | [PDF](https://arxiv.org/pdf/2609.12223v1)

**作者:** Parthasarathy Suryanarayanan `[一作]` (IBM Research), Joseph A. Morrone `[通讯]` (IBM Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于三维几何和残差学习的CCS预测模型，利用预训练的分子几何编码器并在编码器层中实现早期融合的加成条件化；

**💡 创新点**

创新点在于：①在编码器内部通过可学习的加成标记和LoRA注意力适配器实现加成信息的早期注入；②采用残差学习，将物理基线（Ridge回归的质量/表面积特征）从目标中剔除，使模型只学习残差；③设计了随机、化学骨架和加成敏感的三种评估拆分，揭示模型在不同泛化场景下的表现；

**🔧 技术方法**

技术细节包括：使用RDKit ETKDG+MMFF生成10个三维构象；采用UniMol预训练Transformer作为几何编码器；通过高斯径向基函数编码距离注意力；使用LoRA低秩适配器和可学习加成CLS增量；残差目标为Ridge基线的残差；多构象池化策略（单构象、均匀、Boltzmann、学习式）；

**📊 数据集**

数据集：9,209条实验CCS记录（[M+H]+、[M-H]−、[M+Na]+），构成内部基准；外部测试集TS1–TS4以及20个GPCL实验样本；

**📈 对比分析**

与GraphCCS、SigmaCCS以及四种物理计算工作流比较，MPD分别为1.67%、2.11%、2.36%（随机、骨架、加成敏感拆分），RMSE约4.6–6.5 Å²，外部测试和GPCL预实验中RMSE低于对手（2.68 Å² vs 6–11 Å²），显示显著性能提升；

**⚠️ 局限性**

局限性：构象是以中性分子生成并加权，未考虑离子化状态下的构象能量导致在加成敏感拆分中多构象池化效果不佳；仅覆盖三种常见加成；数据量有限，可能导致对大分子或高旋转键的泛化不足；

---

## 156. Toward Robust Personalized Alignment for LLMs: Mitigating Persona Drift in Multi-Turn Dialogue

**arXiv ID:** 2609.12373 | [PDF](https://arxiv.org/pdf/2609.12373v1)

**作者:** Youyuan Zhang `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CORE框架，通过可控的观察路由和置信度引导的信念修订，实现长期对话中个性化信息的稳定更新。

**💡 创新点**

创新点在于将对话轮次的局部证据与持久人物状态更新分离，并通过Commit/Defer/Ignore三种路由与门控修订实现更新控制。

**🔧 技术方法**

采用slot‑factorized belief、信息熵与JS散度评估不确定性，结合强化学习PPO进行策略优化，使用Llama‑3.2‑3B‑Instruct等大模型。

**📊 数据集**

使用ALOe、PersonaChat与新构建的PERSIST基准进行评估。

**📈 对比分析**

与SFT/DPO/RAG等基线相比，CORE在PersonaChat和ALOe上取得最高的AL分数，且在PERSIST上在ARUS、BCD等鲁棒性指标上显著优于RLPA和MemoryBank，性能提升约15–20%。

**⚠️ 局限性**

局限包括依赖预定义人物槽、对开放域或跨槽偏好捕捉不足，且更新策略可能在真实用户细微偏好变化时反应迟缓。

---

## 157. Why User Studies and Participant Experience Reporting Matter for VR Motion Privacy?

**arXiv ID:** 2609.12415 | [PDF](https://arxiv.org/pdf/2609.12415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 158. Asynchronous Parallel Search for Exact Multi-Objective Shortest Paths with Versioned Frontier Snapshots and Indexed Dominance Pruning

**arXiv ID:** 2609.11944 | [PDF](https://arxiv.org/pdf/2609.11944v1)

**作者:** Xiaoqing Xu `[一作]` (China Telecom Research Institute), Hong Tang `[通讯]` (China Telecom Research Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种异步并行共享内存框架 SIP-MOSP，用于求解精确多目标最短路径（MOSP），通过分离搜索与前沿维护、版本化快照与索引化支配剪枝实现高效并行

**💡 创新点**

核心创新点包括：1）基于主导子树/块最小值的索引化支配裁剪；2）前沿的基–增量版本化快照；3）直接目标点交付与版本重用机制；4）搜索与更新工人协同工作而非完整复制搜索状态

**🔧 技术方法**

采用多线程共享内存、锁自由的队列与优先级队列、基–增量快照、块最小值和段树最小值索引、版本重用、直接所有者传递等技术

**📊 数据集**

实验使用三种数据集：纽约城市道路网络（NYC-Road，3-4维），Rocketfuel ISP拓扑（AS3356，10-20维），以及密集的完整有向图（Dense-180，10-20维）

**📈 对比分析**

与四个最先进的精确 MOSP 基线（LTMOA*、NWMOA*、Parallel LTMOA*、Parallel NWMOA*）进行对比，SIP-MOSP 在所有设置中都实现了最高性能：最高可达 46.9× 的顺序加速、7.05× 的并行加速，并在高维度场景下实现了 60 倍以内存压缩

**⚠️ 局限性**

局限性：仅适用于非负可加边权；前沿分配静态且未实现 NUMA 或负权扩展；索引结构和块大小未自适应；在低维度/前沿较小的场景下，内存开销相对较高，性能提升有限

---

## 159. Does Video Memory Use What It Retrieves? A Causal Audit of Memory Specificity

**arXiv ID:** 2609.12090 | [PDF](https://arxiv.org/pdf/2609.12090v1)

**作者:** Aditi Tiwari `[一作]` (University of Illinois Urbana Champaign), Heng Ji `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了读时记忆替换技术，用于区分视频模型中记忆带来的收益与其对检索内容的特异性依赖，系统评估了多种记忆机制在不同模型和数据集上的表现。

**💡 创新点**

创新点在于：①首次引入读时记忆替换这一因果干预方法，直接测量检索内容对模型性能的贡献；②构建替换阶梯和剂量响应实验，揭示记忆依赖的层级（泛化表示、上下文、精确事件）。

**🔧 技术方法**

技术手段包括：读时记忆替换干预、替换阶梯（正确记忆、错误记忆、身份无关记忆、内容无关参考）、剂量响应与固定幅度控制、性能恢复率计算；并结合不同模型的任务读取（RCE、PSNR/LPIPS、J&F）。

**📊 数据集**

使用的数据集包括：Ego-Exo4D、7-Scenes、TUM（用于冻结世界模型评估）；Minecraft（WorldMem评估）；DAVIS 2017、MOSEv2（SAM 2评估）。

**📈 对比分析**

比较方法：在每个模型中对同一目标步骤执行多种记忆替换，计算恢复率ρ(v)。结果显示：Ego-Exo4D/7-Scenes在身份无关记忆下几乎完全恢复收益（≈99%）；TUM恢复率约70%；WorldMem对同轨迹错误记忆恢复≈94%，跨轨迹恢复≈44%；SAM 2在DAVIS和MOSEv2上对错误空间记忆导致恢复率骤降至≈0.1–0.2。

**⚠️ 局限性**

局限性包括：仅在冻结模型上完成剂量响应与方向实验；WorldMem缺乏相同的表示修复分析；评估数据集分割有限（如TUM仅两次评估）；干预仅更改单一步骤的记忆读取，无法捕捉更长时序依赖；不同模型任务和指标难以统一比较。

---

## 160. RodForesight: A World Model Enhanced Diffusion Policy for Slender and Material Agnostic Rod Insertion

**arXiv ID:** 2609.12103 | [PDF](https://arxiv.org/pdf/2609.12103v1)

**作者:** Chuanbo Yu `[一作]` (Southwest Jiaotong University), Peng Wang `[通讯]` (University of Surrey)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了RodForesight框架，用两阶段视觉导引方法实现长细柔性杆穿孔任务；

**💡 创新点**

创新点在于将粗略接近与精细插入分离，利用扩散策略生成候选动作并通过世界模型预先评估其对杆端对齐的影响，实现预测执行；

**🔧 技术方法**

技术上结合了视觉伺服、扩散策略、GRU/Transformer世界模型、Cosserat杆物理模型以及语义分割；

**📊 数据集**

使用在三种材料（AISI 304、C11000 铜、碳纤维/树脂）下生成的 300 条专家轨迹，配合不同孔位、抓取位置和杆长/直径的实验数据集；

**📈 对比分析**

与单阶段扩散策略、确定性变换器等基线对比，RodForesight 在训练分布下从 88.9% 提升至 96.7% 成功率，且能在未知孔位、初始姿态、杆直径及更软材料（6061‑T6 铝、干 PA66）上保持高成功率；

**⚠️ 局限性**

局限性包括对视觉分割依赖、在极细杆（3 mm）时感知不足导致失败、以及世界模型预测在执行过程中的准确性下降导致的选择失误。

---

## 161. Who Are We Recommending To? Recommender Systems in the Agentic Web

**arXiv ID:** 2609.11945 | [PDF](https://arxiv.org/pdf/2609.11945v1)

**作者:** Himan Abdollahpouri `[一作]` (Spotify), Mounia Lalmas `[通讯]` (Spotify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并论证了推荐系统从传统以人类为中心的消费模式向“代理化网络”中的代理人（Agent）消费模式的转变，并引入了“委托光谱”来刻画不同决策场景下代理人参与度的变化。

**💡 创新点**

创新点在于：①首次将推荐系统的委托程度表征为三维因子（偏好可指定性、结果可验证性、决策风险）并构建委托光谱；②提出“代理人注意力经济”概念，强调推荐既要满足终端用户，也需可被代理人执行；③给出面向双重受众（人类与代理人）的设计与评估路线图。

**🔧 技术方法**

主要技术讨论包括：大语言模型（LLM）驱动的代理人推理与执行框架、结构化机器可读推荐输出、基于约束的推荐优化、代理人间通信协议（如MCP、A2A）以及可解释性与审计机制。

**📊 数据集**

该论文为位置/综述性工作，没有使用具体数据集；所讨论的概念与方法均基于已有的公开文献与行业实践。

**📈 对比分析**

由于缺乏实验设计与基准测试，本文没有提供性能对比结果；其贡献主要体现在理论框架与研究议程的提出，而非量化评估。

**⚠️ 局限性**

局限性包括：①概念性论述缺乏实证验证，实际实现与评估方法尚待进一步探索；②委托光谱的划分与三维因子的量化仍需经验与实验支持；③对代理人可信度、责任归属等关键安全问题的具体解决方案尚未给出。

---

## 162. Simulating Disengaged Students to Evaluate LLM-based Tutors

**arXiv ID:** 2609.12331 | [PDF](https://arxiv.org/pdf/2609.12331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 163. Language Is an Insufficient Substrate for Quantitative Reasoning, and Consequential Domains Need Large Quantitative Models

**arXiv ID:** 2609.12105 | [PDF](https://arxiv.org/pdf/2609.12105v1)

**作者:** Reuben Vandeventer `[一作]` (Duo Dimensio LLC), David J. Wild `[通讯]` (Indiana University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一种基于领域原始量化记录的“Large Quantitative Model”框架，采用显式结构、可追溯性和校准不确定性，并将语言模型限定为交互接口。

**💡 创新点**

创新点在于将模型类定义为四大属性（子底层原生、结构显式、线性可追溯、校准不确定性），并证明语言模型本质上无法满足这些属性，主张构建以量化记录为底层的新模型类别。

**🔧 技术方法**

技术包括数据精炼与隐私保护、基于 Mapper 的拓扑结构化、结构动态学习、基于 conformal 方法的校准与弃权策略，以及语言模型的意图解析与叙述生成。

**📊 数据集**

使用的主要数据集包括保险损失记录、医疗病例记录、网络身份验证日志，以及在安全运营中使用的 Los Alamos 多源网络事件数据。

**📈 对比分析**

与传统单任务量化模型或仅做 fine‑tune 的 LLM 对比，实验显示在安全事件检测中结构化策略实现 91% 精度，且决策过程可追溯、可复现；然而整体延迟主要受语言模型生成的叙述影响。

**⚠️ 局限性**

局限性包括：① 数据工程与隐私合规是构建 LQM 的关键瓶颈；② 显式结构可能导致精度下降；③ 目前缺乏衡量四大属性的统一基准；④ 需要更多实证研究验证结构化与预测性能的权衡。

---

## 164. EgoMaize: A First-Person Maize Instance Segmentation Benchmark under Severe Field Occlusion

**arXiv ID:** 2609.12350 | [PDF](https://arxiv.org/pdf/2609.12350v1)

**作者:** Jiayi Li `[一作]` (Beijing University of Technology), Jianxin Cao `[通讯]` (Beijing University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EgoMaize 这一针对后期幼苗玉米的第一人称实例分割基准，定义了证据闭合的标注流程

**💡 创新点**

创新点在于针对同类别高重叠和遮挡场景下的所有权一致实例分割标注策略，并将可见范围内的连通性局部完成，而非全局无监督的全模态标注

**🔧 技术方法**

使用了基于查询的 Mask2Former、CropFormer、YOLOv8/11 等主流分割网络进行基线评测，并结合 SAM 辅助标注工具

**📊 数据集**

数据集为 301 张高分辨率第一人称 RGB 图像，包含 1276 个玉米实例、9071 个器官标注及 2731 个 ignore 区域

**📈 对比分析**

通过 mIoU、AP25/AP50、计数 MAE、中心距离等指标进行对比，发现 Mask2Former Swin‑B 在粗分割上表现最佳，而 CropFormer 在严格 AP50 与计数上优于其他模型，整体仍难以实现完整所有权一致的细粒度掩码

**⚠️ 局限性**

局限性包括数据集规模有限、覆盖的田间环境与品种单一、标注需要人工判断所有权、无法验证隐藏部分形状，以及对不同生长阶段的通用性不足

---

## 165. Sampling via Decision-Flow: Training-Free Extraction of Improved Latent Reasoning Paths in Large Language Models

**arXiv ID:** 2609.12317 | [PDF](https://arxiv.org/pdf/2609.12317v1)

**作者:** Zhendong Mi `[一作]` (Stevens Institute of Technology), Shaoyi Huang `[通讯]` (Stevens Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、无数据的推理时采样框架 DF‑Sample，能够从预训练 LLM 的分布中选取高质量但低概率的推理路径。

**💡 创新点**

创新点在于构建层级推理树、对终结节点进行能量评估并向上传播效用，最终用集成生成先验与传播效用的后验分布进行全局路径选择，从而克服传统一步步贪婪或局部采样的“局部最优导致全局错误”问题。

**🔧 技术方法**

使用的技术包括自回归语言模型采样、层级树构建、终点能量函数、向后效用传播、后验路径采样以及块级分层采样以降低计算成本。

**📊 数据集**

实验数据集涵盖 MATH500（数学）、HumanEval（编程）、GPQA‑Diamond（科学多选）和 AlpacaEval 2.0（通用指令）四大类。

**📈 对比分析**

与基线（贪婪、低温采样、GRPO、Power Sampling）对比，DF‑Sample 在 Qwen2.5‑Math‑7B 上 MATH500 81.8%、GPQA‑Diamond 45.6%、HumanEval 59.1%、AlpacaEval 3.06 分均优于无训练基线且与 RL 训练模型相当或更好。

**⚠️ 局限性**

局限性在于仅提升推理时性能，未通过训练将发现的高质量路径内化；对算力需求仍高，且对 K 与 α 的选择敏感。

---

## 166. Zipbench: Low-Cost Framework for Compressing Comprehensive Benchmarks of Large Language Models

**arXiv ID:** 2609.12475 | [PDF](https://arxiv.org/pdf/2609.12475v1)

**作者:** Zhongzhan Huang `[一作]` (Bosch Research), Hefeng Wu `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种低成本的基准压缩框架ZipBench，通过仅使用少量anchor LLM评估并合成伪日志，学习样本指纹并挑选代表子集，从而压缩LLM基准。

**💡 创新点**

创新点在于用极少的anchor LLM（仅6个）与伪日志合成实现准确压缩，并提供理论误差与秩一致性保证，显著降低构建与评估成本。

**🔧 技术方法**

使用的技术包括伪日志合成、信息响应理论（IRT）样本指纹学习、K-means聚类与代表样本选择，并辅以理论误差与排名一致性分析。

**📊 数据集**

实验数据集覆盖100多种文本、多模态与代理任务的基准，构建ZipBench Zoo提供压缩版本。

**📈 对比分析**

与现有BCM方法相比，ZipBench在MAE仅0.002–0.02、Spearman相关约0.98的同时，将评估成本降至原始的20%–40%，在多种基准上均优于tinyBenchmarks、SubLIME等。

**⚠️ 局限性**

局限性在于仍需对anchor LLM进行完整基准评估，且完全去除日志难以实现，压缩后仍存在一定误差，尤其对高成本代理基准的评估仍不可完全免费。

---

## 167. Physics as the label for measuring and correcting materials reasoning in multimodal models

**arXiv ID:** 2609.12181 | [PDF](https://arxiv.org/pdf/2609.12181v1)

**作者:** Hasan Kurban `[一作]`, Mustafa Kurban `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了MatPCR，一个无标签基准，评估材料数据的物理一致性，检查模型推理链的物理约束。

**💡 创新点**

创新点在于将材料模型评估从答案准确性转变为推理链的物理一致性，且无需人工标签，适用于大多数物理错误。

**🔧 技术方法**

使用了程序化的物理神谕，包括基于DFT的属性检查，定义了物理一致性率（PCR）和约束基础自我验证（CGSV）。

**📊 数据集**

使用了公开的材料数据集，包括实验粉末衍射图、扫描电子显微镜图像、拉曼光谱和来自Materials Project的第一性原理数据。

**📈 对比分析**

与零-shot LLM评判和自信基线进行比较，MatPCR的验证器在分布中表现良好（AUC=0.850），而其他方法的表现较差，未能有效区分物理违规。

**⚠️ 局限性**

限制在于验证器在跨约束类型的泛化能力较差，五种持有约束类型的表现接近随机，且未能在所有持有约束类型中显示出转移能力。

---

## 168. ORQA: An Occupation-Realistic Question and Answer Framework for LLM Professional Knowledge

**arXiv ID:** 2609.12366 | [PDF](https://arxiv.org/pdf/2609.12366v1)

**作者:** Shreyas Krishnan `[一作]` (University of California, Berkeley), Abhishek Nagaraj `[通讯]` (University of California, Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个基于可信职业来源的占位级别知识基准ORQA；

**💡 创新点**

通过自动化将O*NET职业与行业监管网站关联，生成可追溯问答，规模覆盖116个职业并实现大规模生成；

**🔧 技术方法**

使用代理式自动化管道：网络检索、证据卡抽取（GPT‑4o）、多重自动过滤与人工审核，并在封闭式多选与开放式回答两种任务上评测；

**📊 数据集**

数据来自187个权威来源（政府、监管、专业协会等）构成的480条问答，职业按BLS SOC工资账单分配权重；

**📈 对比分析**

对15个LLM进行三次随机种子封闭式多选与开放式回答测试，前沿模型多选约58‑62%，开源模型约33‑41%；通过工资账单加权、职业和SOC组别比较显示显著性能差异；

**⚠️ 局限性**

覆盖不均、少数职业样本不足、仅验证权威资料而非实际决策、可能存在预训练泄露、未包含交互式或人机协作评测等限制。

---

## 169. A Tight $\widetilde Ω(\sqrt{m})$ Information-Theoretic Lower Bound for Randomized Online Set Cover

**arXiv ID:** 2609.12183 | [PDF](https://arxiv.org/pdf/2609.12183v1)

**作者:** Ilan Doron-Arad `[一作]` (MIT), Naor `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对无权在线集合覆盖问题，给出了信息理论随机算法的下界证明，证明任意随机算法的竞争比至少为Ω(√m)；

**💡 创新点**

创新点在于构造了强大的“union-expanding”候选集合与“forcing batch”机制，利用随机筛选与信息论方法突破以往仅有Ω(log m)的下界；

**🔧 技术方法**

技术主要包括Yao最小化原理、随机筛选（随机过滤）过程、组合极大扩张族的概率构造以及通信复杂度归约（用于多方一向集合不相交问题）；

**📊 数据集**

该工作不依赖任何实验数据，全部为理论证明与组合构造；

**📈 对比分析**

对比之前的O(log m log n)上界和已知的Ω(log m)下界，本文将下界提升到接近上界，表明随机化在信息理论层面无法实现更优的竞争比；

**⚠️ 局限性**

局限在于对随机顺序模型的下界仍未达到Ω(√m)（仅得到Ω(m^{1/3}），且未考虑多项式时间与空间限制下的具体算法表现。

---

## 170. Look Before You Leap: Pre-Action Verification for LLM Agents

**arXiv ID:** 2609.11957 | [PDF](https://arxiv.org/pdf/2609.11957v1)

**作者:** Asaad Althoubi `[一作]` (Oklahoma State University), Asaad Althoubi `[通讯]` (Oklahoma State University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5120450524)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种预执行验证框架，利用构造好的正确效果来检测LLM代理在执行shell命令和代码编辑时的“静默失败”。

**💡 创新点**

创新点在于统一的成功/干净失败/静默失败三元分类，oracle‑exact核心+可中止软检查的两层门控策略（选择性定位和Robust‑Apply）。

**🔧 技术方法**

技术包括静态语法/二进制/标志检查、内容锚定 vs 位置锚定编辑格式、模糊匹配阈值、锚点与相似度验证的元应用器。

**📊 数据集**

数据集包含9,930条来自482个真实工具的shell命令（1,986条合法/1,986条四类非法）以及640个编辑与224个Python文件的合成编辑实验。

**📈 对比分析**

实验显示完整验证器对非法命令达到95.8%召回、10%误报；选择性定位将误报降至7%不损失召回；对编辑，内容锚定格式实现0%静默失败，Robust‑Apply将误差降至0.01%，但适用率从≈98%降至≈51%。

**⚠️ 局限性**

局限性包括合成误差可能不完全代表真实代理错误，评估仅覆盖shell和Python，零误报主张只适用于完整命令，且不验证语义正确性。

---

## 171. GTA: Graph Theory Agent and Benchmark for Algorithmic Graph Reasoning with LLMs

**arXiv ID:** 2609.12265 | [PDF](https://arxiv.org/pdf/2609.12265v1)

**作者:** Zixiang Xu `[一作]` (University of Southern California), Xiuying Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 GT Bench（图论基准），评估 LLM 在多步图算法推理中的表现，并基于此设计了 Graph Theory Agent（GTA），通过自适应输入表示选择与算法分解提升推理准确率。

**💡 创新点**

创新点在于①在同一数据集上系统比较四种图的文本表示（自然语言、结构化语言、邻接矩阵、邻接列表）对 LLM 性能的影响；②构建了可训练的表示选择器与分解器，实现对执行器的零改动即可显著提升推理能力；③GTA 在未见数据集和不同执行器上表现出良好迁移性。

**🔧 技术方法**

技术包括：大型语言模型（如 Phi‑4、GPT‑4o）、强化学习+优先级优化（DPO）训练表示选择器与分解器、分层规划（Generator‑Decomposer）框架、生成式数据集构建与自动化评估。

**📊 数据集**

使用了自研的 GT Bench，涵盖 24 种经典图问题、44 种任务-结构组合，超过 100,000 个实例，覆盖四种表示；此外还测试了 GraCoRe 与 NLGraph 数据集以验证迁移性。

**📈 对比分析**

与多种提示、对话与自动化代理基线（如 CoT、LLM‑Debate、AFlow、MaAS 等）对比，GTA 在 GT‑E、GT‑H、GraCoRe、NLGraph 上分别提升 5–9 个百分点，且成本仅为普通多轮方法的 1/5 左右，且在更大图规模下保持较平滑的性能衰减。

**⚠️ 局限性**

局限包括：①对极大规模图仍受执行器内在能力限制；②表示选择器与分解器需针对特定执行器预训练，跨模型迁移仍有一定误差；③在极低上下文窗口或非文本图输入场景下效果未知。

---

## 172. GraphProfiler: Source-Linked Sensitive Attribute Inference via Personal Knowledge Graphs

**arXiv ID:** 2609.12448 | [PDF](https://arxiv.org/pdf/2609.12448v1)

**作者:** Ahmed Sohair Khan `[一作]` (RMIT University), Elham Naghizade `[通讯]` (RMIT University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一个可审计的LLM驱动个人属性推理系统 GraphProfiler，利用用户帖子构建源链接的个人知识图谱，并将推理结果映射到可追溯的图节点与原始帖子上。

**💡 创新点**

创新点在于：①将推理结果与来源级别的知识图谱节点关联，提供可追溯的证据；②保持与文本基线相近的准确率，同时实现高覆盖率的证据输出；③通过证据审计支持针对性隐私修补，而非全局扰动。

**🔧 技术方法**

使用技术包括 GPT‑4o LLM 进行实体关系抽取与属性推理、GraphRAG 框架构建与检索个人知识图谱、源链接映射、支持审计的多指标评估（证据覆盖率、支持率、删除实验）等。

**📊 数据集**

实验数据集为 SynthPAI（合成用户历史，八个属性）和 PANDORA（真实 Reddit 评论，年龄与性别标签）。

**📈 对比分析**

与 FTI 与 AutoProfiler 在 SynthPAI 上对比，GraphProfiler 取得 86.7% 的攻击成功率（接近 88.7% 与 88.3%），在 PANDORA 上得到 84.6% 的整体成功率，且覆盖率 100%。同时提供 98%+ 的证据覆盖、证据支持率高，删除引用证据后成功率显著下降。

**⚠️ 局限性**

限制包括：仅在 GPT‑4o 环境下验证，证据不保证完全覆盖且删除不一定消除推断；实体关系抽取质量依赖于构建方法；未评估与块级 RAG 的边际贡献；合成数据与真实数据标签可能存在噪声和时间漂移问题。

---

## 173. DATAFARM: Distribution-Aligned Task and Motion Planning for Fine-Tuning Vision-Language-Action Models

**arXiv ID:** 2609.12316 | [PDF](https://arxiv.org/pdf/2609.12316v1)

**作者:** Samrat Sahoo `[一作]` (Stanford University), Tom Silver `[通讯]` (Princeton University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对机器人基础模型（VLA）进行微调时，提出了 DATAFARM 方法，通过将任务与运动规划（TAMP）生成的演示与 VLA 预训练分布对齐，从而生成更有效的训练数据。

**💡 创新点**

创新点在于：①在 TAMP 的路径优化与重定时过程中引入预训练分布的三重对齐目标（关节空间、运动风格和执行时序）；②利用预训练演示学习可微分编码器和高斯混合模型，使规划器能够自适应地生成与 VLA 预训练分布相似的轨迹；③证明对齐后即使在无目标任务演示的情况下，也能显著提升微调效果并保留原有能力。

**🔧 技术方法**

使用的技术包括：TAMP（TiPToP + cuTAMP + cuRobo）、基于 Gaussian Mixture Model 的关节分布拟合、变分自编码器（VAE）+ 运动描述符预测的轨迹编码器、梯度优化路径规划与重定时、以及 VLA 的离线微调流程。

**📊 数据集**

数据集：以 DROID（人类远程操控的抓取与放置演示）作为预训练参考分布，TAMP 产生的演示作为微调数据；实验任务包括三项桌面操纵任务和布料折叠（OOD）任务。

**📈 对比分析**

与传统原始 TAMP 演示、无对齐演示以及人类演示进行比较：在三项目标任务上，DATAFARM 的平均成功率达 56.7%（对齐前仅 8.3%），接近人类演示的 61.7%；在布料折叠 OOD 任务上，微调后成功率仅略低于预训练模型（85% 对比 90%）。

**⚠️ 局限性**

局限性：①依赖于预训练分布的代表性样本，若缺乏可用数据需另寻替代；②仅适用于 TAMP 能覆盖的任务（如静态抓取放置），无法处理接触丰富或非抓取任务；③对齐目标的权重需经验调优，且对不同机器人结构与 VLA 可能需要重构。

---

## 174. Is Gaussian Splatting Becoming Neural Again? A Taxonomy and Controlled Study of Learned Parameterization

**arXiv ID:** 2609.12395 | [PDF](https://arxiv.org/pdf/2609.12395v1)

**作者:** YuanHang Wang `[一作]`, Yi Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三维高斯散射（3DGS）的神经化过程进行系统化分析与实验，提出五轴分类体系，评估不同神经化方式对重建质量的影响。

**💡 创新点**

①建立属性解码、空间共享、视角条件解码、拓扑生成、摊销推断五轴分类法；②阐明神经化的具体位置与作用；③通过对齐实验验证“选择性神经化”优于全神经化。

**🔧 技术方法**

采用3DGS渲染器、视角条件MLP、anchor/哈希/三平面共享结构、稀疏/稠密化调度等技术，并在统一的MIP‑NeRF 360评估协议下进行实验。

**📊 数据集**

主要使用公开的mip‑NeRF 360数据集（9个场景）进行新视角合成评估，同时引用DTU稀疏视角协议等公开数据集。

**📈 对比分析**

对19种代表性方法进行A‑S‑V‑T‑I向量计分，并在同一评估框架下对四种神经化变体（E0–E3）做PSNR/SSIM/LPIPS比较。结果显示共享外观与不透明度可提升约0.5 dB PSNR，进一步解码几何结构则略微降低质量。

**⚠️ 局限性**

实验仅评估图像质量，未涉及存储、推理速度、编辑局部性和跨场景泛化；分类加权等价且可能不适用于所有应用；仅选取部分方法，未覆盖所有现有变体。

---

## 175. Adaptive AI: Energy Efficient Multi-exit TinyML on Intelligent Vision Systems at the Edge

**arXiv ID:** 2609.11939 | [PDF](https://arxiv.org/pdf/2609.11939v1)

**作者:** Luca Crupi `[一作]` (SUPSI), Daniele Palossi `[通讯]` (SUPSI)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在GAP9 MCU上构建并部署了基于MobileNetV2的四路早期退出网络，用置信度门控实现自适应推理，显著降低计算量与能耗；

**💡 创新点**

提出多路早期退出与并行训练策略，提升浅层分类准确率并实现比传统单一退出方案更高的计算效率；

**🔧 技术方法**

采用置信度阈值驱动的早期退出机制、并行训练、多路分类头、量化与NE16加速器实现；

**📊 数据集**

使用ImageNet-100子集进行训练与评估；

**📈 对比分析**

与单路MobileNetV2以及第三方State‑of‑the‑Art多路模型对比，平均计算成本下降41%，能耗降低24%，时延降低29%，并在阈值调节下保持或提升≈1%精度，计算效率提升2×；

**⚠️ 局限性**

仅在MobileNetV2/ ImageNet-100/GAP9环境验证，扩展性与多样化数据集、模型的通用性仍待进一步验证；

---

## 176. A Differentially Private Federated Proximal Optimization Framework for Customer Churn Prediction in Heterogeneous Federated Telecom Networks

**arXiv ID:** 2609.12470 | [PDF](https://arxiv.org/pdf/2609.12470v1)

**作者:** Joydeb Kumar Sana `[一作]` (Bangladesh University of Engineering and Technology), M M Manjurul Islam `[通讯]` (Ulster University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出差分隐私联邦Proximal（DP‑FedProx）框架，用于在不共享原始客户数据的前提下完成电信行业客户流失预测，解决了非IID数据和隐私泄露双重挑战。

**💡 创新点**

创新点：①首次将FedProx与DP‑SGD结合，形成兼顾异构数据与强隐私保障的联邦学习模型；②采用基于客户租期划分的非IID分区方式，更贴合真实运营商场景；③通过SHAP解释差分隐私对特征重要性的影响，为可解释性提供新视角。

**🔧 技术方法**

技术：联邦学习（FedAvg、FedProx）、差分隐私（DP‑SGD/Opacus）、深度神经网络ChurnNet、WoE特征编码、SHAP可解释、Python、PyTorch、Flower、Scikit‑Learn、XGBoost。

**📊 数据集**

数据集：两份公开电信流失数据集——Dataset‑1（100,000样本，101特征，Kaggle来源）与Dataset‑2（7,043样本，21特征，IBM公开样本）。

**📈 对比分析**

比较方式：在两数据集上与中心化模型（RF、LR、XGB等）、本地模型、FedAvg、FedProx、DP‑FedAvg、DP‑FedProx等在七项指标（Accuracy、Precision、Recall、F1‑score、Specificity、ROC‑AUC、PR‑AUC）进行统一评估；结果显示FedProx优于FedAvg，DP‑FedProx仅略逊于FedProx，但与中心化LR保持相近，Dataset‑1上准确率≈91.6%，F1≈91.2%；Dataset‑2上准确率≈80%。

**⚠️ 局限性**

局限性：①客户租期划分导致非IID程度偏轻，FedProx的proximal作用有限；②μ设为0.01，proximal与DP噪声相互抵消，在更极端异构场景下效果可能下降；③实验仅基于公开数据集，缺乏大规模多运营商真实环境的验证；④仅使用单一神经网络架构，未探索更复杂模型与自适应隐私预算。

---

## 177. An End-to-End Automated Pipeline for Controllable Crack Data Synthesis

**arXiv ID:** 2609.12431 | [PDF](https://arxiv.org/pdf/2609.12431v1)

**作者:** Conghui Li `[一作]` (Monash University), Xin Wang `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一套基于Bézier曲线、GAN生成掩码与双ControlNet扩散模型的端到端可控裂纹数据合成管线。

**💡 创新点**

创新点在于将裂纹几何用可编程Bézier表示并通过GAN转换为真实掩码，结合双ControlNet（外观+边缘约束）实现精准形状控制，并支持背景自由与填补两种模式。

**🔧 技术方法**

使用Bézier曲线采样、CLIPasso逆映射、GAN掩码生成、Stable Diffusion + 双ControlNet、Canny边缘控制以及图像填补技术。

**📊 数据集**

主要在CRACK500和CrackTree200公共裂纹数据集上进行训练与评估，并构建大规模曲线-掩码对数据集。

**📈 对比分析**

与传统数据增强（CDM、DDPM）和无增强基线相比，在U‑Net、HRNet、JTFN、DeeplabV3+和Crackformer等模型上均实现了约8 mIoU的提升，显著提高分割性能。

**⚠️ 局限性**

局限在于Bézier曲线难以表达复杂裂纹网络，文本驱动的背景生成可能产生不自然纹理，且目前对极端裂纹类型和背景一致性控制仍需改进。

---

## 178. RoES: Rotational Equivariant Selective-frequency Fusion for Multimodal Images

**arXiv ID:** 2609.12497 | [PDF](https://arxiv.org/pdf/2609.12497v1)

**作者:** Jiabao Wang `[一作]` (Zhongnan University of Economics and Law), Xiaobo Liu `[通讯]` (China University of Geosciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种旋转等变性选择性频率融合框架RoES，利用可学习的低高频分解与低频等变Mamba分支和高频频域引导的Dual-Fourier分支，实现红外-可见图像融合；

**💡 创新点**

创新点在于将旋转等变性仅对低频结构强制施加，而对高频细节保持灵活；采用可学习的RoUP分解、频率选择性等变与极坐标频谱注意力；

**🔧 技术方法**

技术包括可学习的lifting式分解RoUP、旋转等变的Mamba编码器、频率引导的Dual-Fourier块、极坐标谱注意力PSA、Restormer解码器与多任务损失；

**📊 数据集**

使用MSRS数据集进行训练，评估在M³FD、RoadScene、TNO三大公开融合基准数据集；

**📈 对比分析**

与17种现有融合方法对比，RoES在所有数据集的TOPSIS排名第一，VIF、MI、FMI、Q_abf等指标领先，且在下游YOLOv5目标检测上获得最高mAP；

**⚠️ 局限性**

局限在于仍需较大计算开销（约2.3M参数、1,989G FLOPs），对极端噪声或多模态配准误差的鲁棒性尚待进一步验证。

---

## 179. Revisiting Multi-Object Tracking Baselines: Hyperparameter Optimization with Multi-Fidelity Greedy Coordinate Search

**arXiv ID:** 2609.12261 | [PDF](https://arxiv.org/pdf/2609.12261v1)

**作者:** Momir Adžemović `[一作]` `[通讯]` (University of Belgrade), Momir Adžemović (University of Belgrade)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

系统性地将超参数优化应用到多目标跟踪的跟踪-检测方法上，并提出多保真贪婪坐标搜索（MFGCS）。

**💡 创新点**

创新点在于：①设计了多保真贪婪坐标搜索以显著降低全数据评估成本；②对八种超参数优化器在四个跟踪器与两个数据集上进行统一搜索空间与评估流程的全面比较。

**🔧 技术方法**

采用了黑盒超参数优化技术（随机搜索、TPE、GP-BO）、多保真策略（Hyperband、BOHB）以及贪婪坐标搜索与MFGCS，并用HOTA评估指标进行性能衡量。

**📊 数据集**

使用了两个公开视频跟踪数据集：DanceTrack 和 SportsMOT。

**📈 对比分析**

通过在验证集上对八种优化器进行统一搜索空间和评估流程比较，随后将最佳的 TPE 与 MFGCS 应用于四个跟踪器+两数据集的跨模型评估，所有优化结果均显著超过手动调参和公开基线，HOTA提升可达 4–16 分，测试集亦验证提升。

**⚠️ 局限性**

局限性包括：仅覆盖跟踪-检测范式，未涉及基于外观的跟踪器；多保真子集选择假设场景长度相似，长短场景差异可能影响预算精度；未在更多模型或不同目标场景上进一步验证泛化能力。

---

## 180. Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement

**arXiv ID:** 2609.12107 | [PDF](https://arxiv.org/pdf/2609.12107v1)

**作者:** Rafael Macalaba `[一作]` (World Bank), Olivier Dupriez `[通讯]` (World Bank)

**通讯引用:** 448 | [OpenAlex ID](https://openalex.org/A5049463620)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个弱监督框架，利用已有的轻量级数据集提取模型产生候选，借助大型语言模型校正并补全训练集，再用合成与对比样本微调模型，实现在FCV和强迫迁徙文档中高精度识别数据集引用。

**💡 创新点**

创新点在于将候选生成、LLM细化、合成对比示例三者结合，突破了无大规模标注数据情况下的领域迁移瓶颈，并首次在灾区与冲突文本中实现了可扩展的轻量化数据集提取。

**🔧 技术方法**

采用轻量级命名实体识别模型、OpenAI LLM进行候选校正、少量提示合成合成与对比样本、Pydantic结构化输出、以及对模型的微调。

**📊 数据集**

使用了93份来自UNHCR/ReliefWeb、SEIS简报、世界银行政策研究工作论文和项目评估文件的文档，评估集包含1,706段落（其中545段落包含数据集引用）。

**📈 对比分析**

与独立金标基准比较，整体提取精度为74.1%/召回70.5%，正例段落精度89.5%；段落检测准确率88.2%，特异性88.6%，在四类文档中保持≈90%精度。

**⚠️ 局限性**

主要限制在于召回仍不足，尤其在操作文本稀缺、含混的情况；需要进一步补充真实数据资源链接、评估使用场景，并扩展到更广泛的政策文本以验证模型迁移性。

---

## 181. Is Bash All You Need? An Empirical Study of Tool Interfaces for Enterprise Digital Worker Agents

**arXiv ID:** 2609.11999 | [PDF](https://arxiv.org/pdf/2609.11999v1)

**作者:** Hazel Mak `[一作]` (Microsoft Corporation), Alejandro Gutierrez Munoz `[通讯]` (Microsoft Corporation)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对比了五种工具接口（仅工具、bash、bash+工具、bash+合成工具、程序化工具调用）在两大企业工作流基准上的性能，使用 Opus‑4.8 和 GPT‑5.5 模型。

**💡 创新点**

系统地评估了 shell 执行与结构化工具调用在企业任务中的效果，首次量化 bash 单独使用对任务质量与成本的优势。

**🔧 技术方法**

采用大语言模型的工具调用与 bash 脚本执行，结合程序化工具调用和持久化工具合成技术。

**📊 数据集**

使用 TheAgentCompany（软件公司模拟任务）和 APEX‑Agents（投资银行、咨询、企业法务的专业分析任务）两大基准集。

**📈 对比分析**

通过任务得分、通行率、令牌使用、估算成本和时间等指标进行对比；结果显示 bash 单独使用在两大基准上均获得最高得分、最低令牌消耗和成本。

**⚠️ 局限性**

受限于仅评估两种模型和两套基准，未涵盖更广泛的企业场景，且程序化工具调用仅在受限目录下实现，可能不适用于所有安全合规需求。

---

## 182. Computing at Sea: Floating and Offshore Data Centres as a Pathway to Sustainable AI Infrastructure

**arXiv ID:** 2609.12511 | [PDF](https://arxiv.org/pdf/2609.12511v1)

**作者:** Cheng Siong Chin `[一作]`, M. Venkateshkumar `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

分析并展示浮动/海上数据中心的技术可行性与运营优势，强调海洋自然冷却与海上可再生能源的协同效应。

**💡 创新点**

提出“海上能源岛”概念，将可再生能源与数据中心集成、利用海水被动冷却、实现零淡水消耗，并引入模块化、可预制的部署架构。

**🔧 技术方法**

海水被动冷却、氮气充填防腐蚀舱、压力容器/浮筒模块、海上风/潮汐/波浪/OTEC 能源系统、海底/海面光纤电缆、海洋温度/能源管理 AI 预测模型。

**📊 数据集**

无公开机器学习数据集；以实际部署（Project Natick、海岸高海数据中心、Hamina 站点）与行业统计（TUE、PUE、WUE 数据）为评估依据。

**📈 对比分析**

通过 PUE、WUE、冷却能耗百分比、服务器故障率等指标与陆基数据中心（传统、现代超规模、最佳级）对比；海上平台 PUE 1.05–1.15，Natick 1.07，故障率 8 倍降低，水耗 100% 降为 0。

**⚠️ 局限性**

维护难度高、弹性电网与监管冲突、海洋生态影响评估不足、现场故障需全舱检索、成本上升与海底电缆投资、数据主权与隐私合规挑战。

---

## 183. Niching Agents in The Core

**arXiv ID:** 2609.12398 | [PDF](https://arxiv.org/pdf/2609.12398v1)

**作者:** Gary B. Parker `[一作]` (Connecticut College), John Asaro `[通讯]` (Connecticut College)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在Xpilot游戏环境中使用The Core竞争共进化算法，对代理进行空间隔离，训练出针对特定子区域（四象限）的专属控制器；

**💡 创新点**

创新点在于将共进化算法与多循环基因编码相结合，实现无显式适应度函数的局部行为进化，并通过空间约束实现代理的分化与专业化；

**🔧 技术方法**

主要技术包括多循环循环遗传算法（Multi-Loop CGA）、基于二进制染色体的动作编码、基于攻击结果的交叉机制以及年龄阈值控制代理的基因传播；

**📊 数据集**

使用的数据集是Xpilot-AI自带的四象限地图，实验中共运行120个代理，分别在各象限内与全局非限定代理及其他象限代理进行对战；

**📈 对比分析**

通过将本地化代理与全球化代理、以及不同象限代理进行对战，记录12小时对战中的累计击杀数进行比较，结果显示本地化代理在自身象限及较少障碍象限中击杀数显著高于全局代理，且在跨象限对战中表现优异；

**⚠️ 局限性**

局限性包括缺乏明确适应度评估导致难以量化进化过程、实验仅限于Xpilot游戏环境与四象限地图，且对更复杂游戏或多样化地图的泛化能力尚未验证。

---

## 184. FINESSE: An Agent-Based Simulator and Benchmark Dataset for Multimodal Financial Event Sequences

**arXiv ID:** 2609.11993 | [PDF](https://arxiv.org/pdf/2609.11993v1)

**作者:** Tyler Farnan `[一作]`, Senthil Kumar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于代理的模拟框架CuBe-ABM，用以刻画客户的交易、支付、政策变更、隐藏状态更新以及账务状态转移，并通过多种日志记录实时跟踪这些过程。

**💡 创新点**

创新点在于将客户的商户亲和度与支付策略动态化，并在模拟中引入随机对照试验（RCT）干预机制，同时同时记录多条日志以实现对客户行为全景的可追踪性。

**🔧 技术方法**

使用了代理建模（Agent-Based Modeling）、随机采样（Uniform Sampling）、概率状态转移、日志记录等技术；通过迭代时间步（t = 1..T）逐步更新客户状态。

**📊 数据集**

主要使用了合成的商户集合ℳ（包含MCC、ID、流行度、费率等属性）以及模拟产生的客户数N_c和时间长度T；未指明使用真实交易数据集。

**📈 对比分析**

论文未给出与现有方法的对比实验或性能指标，主要侧重于框架构建与日志生成，若有对比需自行基于生成日志进行后续统计或政策效果评估。

**⚠️ 局限性**

局限性包括：1）模型高度简化，未结合真实交易数据验证；2）隐藏状态更新采用简单概率更新，可能无法充分捕捉真实客户心理变化；3）大规模客户数下的计算开销未讨论；4）对政策干预效果的量化评估缺失。

---

## 185. Beyond the Query: Do Retrieval Signals Improve Adaptive Multimodal RAG Routing?

**arXiv ID:** 2609.12437 | [PDF](https://arxiv.org/pdf/2609.12437v1)

**作者:** Qiaomu Li `[一作]` (Kennesaw State University), Nong Ming `[通讯]` (Kennesaw State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在多模态检索增强生成（RAG）系统中，对比查询仅特征与查询+检索状态特征两种路由器，评估后者是否能提升路由决策；并对文档、音频、视频的可选检索步骤进行细粒度诊断。

**💡 创新点**

提出了匹配的查询-仅与查询+检索对照实验，揭示检索信号并不必然带来增量路由价值；同时提供了可视化诊断表明不同动作对路由难度的影响。

**🔧 技术方法**

使用梯度提升机（GBM）路由器、Spearman相关与稳健惩罚评估、Qwen2.5-Omni-7B生成器，结合文本、视觉、音频、视频多模态检索；同时对检索状态进行特征化。

**📊 数据集**

采用 DocVQA（文档）、Clotho-AQA（音频）、NExT-QA（视频）三大数据集，包含3574条查询（开发集715条，最终评估720条），每条查询对应多模态检索与生成结果。

**📈 对比分析**

通过在开发集和最终评估集上计算路由分数差异进行对照，开发集增益+0.01621（95% CI=[0.00436,0.02960]），最终评估仅+0.00288（95% CI=[-0.00726,0.01107]），表明检索信号在最终评估中的增量价值不显著。

**⚠️ 局限性**

仅评估了有限的检索特征和路由器结构，缺乏正向对照实验；未覆盖所有源分离的可能性；未测量真实服务时延；结论仅适用于所用模型、动作与数据集，未必推广到更广泛的 RAG 场景。

---

## 186. DWMP: Leveraging Dual World Models for Humanoid Obstacle Traversal

**arXiv ID:** 2609.12347 | [PDF](https://arxiv.org/pdf/2609.12347v1)

**作者:** Rongjun Jin `[一作]`, Yue Gao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 DWMP 框架，利用双重世界模型（Koopman 运动模型 + RSSM 视觉模型）为人形机器人障碍穿越任务生成兼容两种感知模态的潜在表征，并在学生策略中利用该表征实现更稳健的障碍通行。

**💡 创新点**

创新点在于：① 将运动学非线性系统映射到 Koopman 线性潜在空间；② 使用 RSSM 对 egocentric 深度图进行压缩并进行未来状态预测；③ 在教师-学生蒸馏流程中加入两阶段训练，使双重模型先在教师探索数据上预训练，然后在学生交互数据中微调；④ 将两种潜在表征融合供策略直接使用，提升了对不同模态信息的利用。

**🔧 技术方法**

核心技术包括：Koopman 运算符理论 + Auto-Koopman 网络；Recurrent State‑Space Model (RSSM) + VAE 的 Depth Dreamer；教师-学生策略蒸馏；强化学习与行为克隆相结合的联合损失；以及在线回放缓冲区用于模型微调。

**📊 数据集**

数据集：① 由教师策略在 MuJoCo 仿真中收集的探索轨迹（包含本体感知与深度图）；② 学生策略在仿真与真实 Unitree G1 机器人上收集的交互数据；③ 真实测试使用随机布置的 Ceil、Mceilbar、Narrow 三类障碍环境。

**📈 对比分析**

与现有方法（HumanoidPF、Dreamer‑v3、VAE 直接蒸馏）对比，DWMP 在模拟环境中跨多种障碍类型的成功率平均提升约10%，在真实机器人上通过率分别为0.95、0.8、0.7，明显优于对照组（0.8、0.6、0.65）。视觉重建误差更低、推理速度更快，证明双重模型有效提升了表征质量与政策执行效率。

**⚠️ 局限性**

局限性：① 需要教师策略提供的专用训练数据，增加了初期资源投入；② 目前仅验证在静态障碍环境，动态或更复杂的场景下的鲁棒性尚待研究；③ 双模型的联合训练与参数调优复杂度较高，难以直接迁移到其他机器人平台。

---

## 187. Beyond ID Embeddings: Process-Grounded Language Modeling for Cognitive Diagnosis

**arXiv ID:** 2609.12403 | [PDF](https://arxiv.org/pdf/2609.12403v1)

**作者:** Minghang Liu `[一作]` (Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 Process‑aware Language Cognitive Diagnosis (PLCD) 框架，利用大语言模型构建概念结构和认知过程图，通过目标条件语义记忆检索历史答案，并将文本证据映射到统一的认知空间，从而诊断学生对概念的掌握水平。

**💡 创新点**

创新点在于①抛弃传统的 ID 嵌入，直接以语言结构为认知先验；②采用目标条件语义记忆检索与目标相关的历史响应；③使用 DA‑MoE 专家与过程级对比学习实现文本到认知的映射。

**🔧 技术方法**

核心技术包括大语言模型（LLM）生成概念架构和过程图、目标条件语义记忆、DA‑MoE（动态门控混合专家）模型、过程级对比学习以及心理测量层（考虑猜测与滑动效应）。

**📊 数据集**

在三组公开的真实教育数据集上进行实验，数据集具体名称未在摘要中给出。

**📈 对比分析**

与传统 ID‑based CDMs 及最新自监督方法对比，PLCD 在学生成绩预测上表现更优，尤其在冷启动和稀疏响应场景下显著提升预测准确率。

**⚠️ 局限性**

主要局限包括：依赖已有响应记录来校准学习者状态；对文本质量和清晰度敏感；评价指标仅为响应预测，无法直接观测真实掌握水平；尚不能完全替代纵向评估和因果学习机制。

---

## 188. Shards on a Shoestring: Empirical Characterization of NEAR Protocol Nightshade Sharding on Commodity Hardware

**arXiv ID:** 2609.12091 | [PDF](https://arxiv.org/pdf/2609.12091v1)

**作者:** Sohini Sahukar `[一作]` (Illinois Institute of Technology), Ioan Raicu `[通讯]` (Illinois Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 commodity 硬件上对 NEAR Protocol Nightshade 分片进行系统测评，评估 TPS、块时长、内存/磁盘 I/O 并识别瓶颈；

**💡 创新点**

首次独立实测 NEAR Nightshade，发现三大瓶颈阶段并揭示 HDD 写入回压作为协议隐式流控机制；

**🔧 技术方法**

使用 Chameleon Cloud 裸机实验、Prometheus 监控、RocksDB、tmpfs 与 HDD 对比、Doomslug + BFT 最终确认；

**📊 数据集**

收集 1~24 shard 的 TPS、块时长、块拆分、内存占用、磁盘写入速率、延迟收据队列等指标，形成实测数据集；

**📈 对比分析**

通过 Prometheus 数据采集与 tmpfs/HDD 对比，测得聚合 TPS 在 N=8 达到最高 1,498 TPS（+40%），N=16 tmpfs 发生链停，性能受协调一致性瓶颈影响；

**⚠️ 局限性**

仅在单节点裸机上测试，未覆盖多节点、网络延迟和更高 N 的真实环境，且受限于 NEAR Protocol 版本 84。

---

## 189. Hardware Fingerprinting FTQC via Quantum Decoder Timing

**arXiv ID:** 2609.12145 | [PDF](https://arxiv.org/pdf/2609.12145v1)

**作者:** Friedrich Doku `[一作]` (Northwestern University), Kaitlin N. Smith `[通讯]` (Northwestern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过测量量子纠错解码器每一次解码的壁钟时间，探测并利用这一时间作为侧信道，恢复工作负载的逻辑错误率、编码距离，并实现对物理量子处理器的硬件指纹识别。

**💡 创新点**

①首次揭示解码器运行时是量子容错系统的安全漏洞；②提出基于解码器时序的 14 维特征向量作为硬件指纹；③在真实 IBM Heron 设备上与公开的 Google Willow 数据上验证，该侧信道能跨代码家族、跨硬件厂商持续工作。

**🔧 技术方法**

使用最小权重完美匹配 (MWPM) 与并查集 (Union‑Find) 两种主流解码算法，采集解码时延；通过统计分析（均值、方差、对数均值、百分位数）与计数分布（0–7 次触发及≥8 次）构造特征；采用 k‑最近邻分类器进行指纹匹配；利用 Kolmogorov–Smirnov 检验分布差异；利用线性回归估计逻辑错误率。

**📊 数据集**

数据集来自三台 IBM Heron‑R2 超导处理器（共 14 次运行，68 天内收集），每台设备执行重复码内存实验（代码距离 d=5 与 d=7），共 48,000 次投影。另取 Google Willow 105‑qubit 低阈值表面码实验，50,000 次检测事件，用于跨硬件验证。

**📈 对比分析**

与无侧信道的先验估计（仅基于数据集整体中位数）相比，利用解码时延可将逻辑错误率估计精度提升至 84–94% 的实验跑；从约 50 次定时解码即可以 95% 的准确率判定代码距离；设备指纹识别准确率达到 81–89%（随机猜测仅 33%）。实验中对比了不同解码器、代码距离以及不同硬件，均显示出显著的统计显著性（p < 10⁻¹⁰⁰）。

**⚠️ 局限性**

仅使用三台相同架构的设备，代码距离限制为 d=5、d=7；未评估更大代码距离、更多候选硬件或其他解码器（如 BP+OSD）；实验采用统一电路，未考察工作负载特异性对侧信道的影响；缺乏对云平台多租户真实场景的进一步验证；未实现针对解码器时延的实时防御或抗攻击方法。

---

## 190. A Deterministic $O^*((3/2)^n)$ Algorithm for the Parity of Directed Hamiltonian Cycles

**arXiv ID:** 2609.11982 | [PDF](https://arxiv.org/pdf/2609.11982v1)

**作者:** Hanqing Li `[一作]` (Peking University), Hanqing Li `[通讯]` (Peking University)

**通讯引用:** 22709 | [OpenAlex ID](https://openalex.org/A5100330148)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一个确定性算法，计算有向图中哈密顿环数量的奇偶性，时间复杂度为 O^*((3/2)^n)，空间复杂度为 2^O(n/log n)

**💡 创新点**

通过将Björklund–Husfeldt的局部度数公式转化为三值线性编码，并构造三元反对称覆盖，从而把原本基于Fibonacci前缀族的指数基φ≈1.618降低到3/2，并实现了完全确定性的去随机化

**🔧 技术方法**

利用三元状态编码、反对称二元子立方覆盖、Kuang–Wang的偶诱导子图构造、条件期望法以及对GF(2)的高斯消元

**📊 数据集**

无具体数据集，论文为理论算法分析

**📈 对比分析**

与之前基于 φ≈1.618 的最优算法相比，新的算法在时间上实现了 3/2 的指数基，虽然空间略微增加到 2^O(n/log n) 但仍低于指数级别，证明了在空间-时间权衡下的性能提升

**⚠️ 局限性**

算法仍需处理 n=1 的边界情况，且空间复杂度为次指数级，尚未进一步降低；对更一般的多色版本尚未给出改进

---

## 191. Position: Recommender Systems Should Move Beyond Platform-Centric Ranking toward Personal Agent-Mediated Recommendation

**arXiv ID:** 2609.11942 | [PDF](https://arxiv.org/pdf/2609.11942v1)

**作者:** Haohan Yuan `[一作]` (University of North Carolina at Charlotte), Junning Zhu `[通讯]` (Beijing Normal-Hong Kong Baptist University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出个人代理中介推荐（PAMR）范式，强调在推荐中由用户代理来获取、过滤、治理分布式证据，并在演示中对Yelp餐厅任务做源选择与披露控制的原型实验。

**💡 创新点**

创新点在于把推荐的控制权从平台侧的候选集和排序迁移到用户侧的证据中介，明确四个核心决策（源发现、隐私披露、保留来源及不一致性、反馈驱动的政策适配），并提供评估框架。

**🔧 技术方法**

使用大语言模型（GPT‑5.4）作为统一的排名器，配合自定义的源路由、隐私预算、证据聚合与策略更新模块。

**📊 数据集**

使用200个针对Yelp餐厅的“硬”推荐任务，包含10个候选、一个目标及高评分负例。

**📈 对比分析**

对比基准包括平台原始排序、平台LLM重排序，以及四种源/披露组合（全部源全披露、挑选源全披露、全部源受控披露、挑选源受控披露）。实验结果显示，在控制披露下源选择可将HR@3提升至0.740，曝光量降低77%，且无明显效用损失。

**⚠️ 局限性**

局限性：实验仅在模拟源与固定候选集上验证，未进行真实用户或长期跟踪；缺乏对策略学习与可解释性的完整评估；隐私曝光度仅为实验内部度量；未验证代理真正的用户治理与平台独立性。

---

## 192. Argus: Orchestrating Cross-Layer GPU Performance Measurements around Semantic Regions

**arXiv ID:** 2609.12299 | [PDF](https://arxiv.org/pdf/2609.12299v1)

**作者:** Jianzhu Yao `[一作]` (Princeton University), Adnan Aziz `[通讯]` (Meta)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于语义代码区块的 GPU 性能测量框架 Argus，能够在编译、执行和多源收集之间统一区块身份，自动生成多运行测量计划并生成包含归因信息的区块报告。

**💡 创新点**

创新点包括（1）通过稳定 ID 与可识别标记保持区块身份，解决编译后区块拆分与共享指令导致的归因难题；（2）基于测量计划的干扰感知（interference-aware）多跑策略，避免不同测量方式互相污染；（3）统一调度器把编译变换、收集、合成等多种任务视作同一任务流，保证结果可追踪并支持增量执行。

**🔧 技术方法**

技术实现主要依赖：Triton 编译器 + MLIR 进行区块标记与变换；NVBit、CUPTI、Nsight Systems 等硬件与系统级采样工具；自定义的参考追踪与 PC‑to‑region 映射；动态计划生成器与多线程调度框架；以及与 AlphaEvolve 等 LLM 驱动的优化器集成。

**📊 数据集**

使用的数据集与实验平台包括：44 个 Triton kernel（包含 warp‑specialized persistent GEMM 与多种 attention 配置）；TinyLlama‑1.1B decode megakernel（单 GPU）；以及 5 个 2‑GPU / 4‑GPU Hopper H100 上的 GEMM 流程，用来评估跨层 PGO。

**📈 对比分析**

比较方法：将 Argus 作为测量接口插入现有 5 种 LLM‑驱动的 kernel 优化器；对 44 个 kernel 的 25 次搜索实验，比较最终最佳加速比；在 LLM decode 任务上比较 Argus‑引导 vs 原始终端性能；在跨层 PGO 上与 PyTorch baseline 进行吞吐率对比。性能提升分别为：几何平均加速 5.4%→8.9%；解码 1.65 ms/token vs 4.92 ms/token（提升 3.8×）；跨层 PGO 平均吞吐率提升 7%。

**⚠️ 局限性**

局限性：测量计划需要多种工具的支持，导致在不同硬件或驱动版本上移植困难；参考追踪与隔离采样会引入额外的运行时开销；对区块的手动标记仍然是必要步骤；在极大规模 kernel 或极端异步模式下，干扰感知规划与调度可能产生较高的计划生成成本。

---

## 193. Local Edits, Global Ripples: Replay-Informed Policy Adaptation for Workflow Synthesis

**arXiv ID:** 2609.12127 | [PDF](https://arxiv.org/pdf/2609.12127v1)

**作者:** Manqing Mao `[一作]` (Amazon, Inc.), Wei Niu `[通讯]` (Amazon, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种名为RIPPLE的持续提示策略编辑框架，用于在冻结模型下改进可执行工作流的生成，通过诊断失败、定位编辑范围、生成边界修补，并在可重放的评估阶段决定是否将编辑持久化。

**💡 创新点**

创新点在于：① 将编辑局部化与编辑安全性分离，先定位失败到特定策略片段，再通过重放检测编辑在已接受改动下的交互效果；② 引入“重放门控”机制，确保编辑在组合后仍能保持收益并不破坏正确性；③ 通过版本化修补库和规则化诊断实现可复现、可审计的编辑流程。

**🔧 技术方法**

技术包括：① 轨迹级诊断（规则化谓词识别失败家族） ② 基于片段的策略分段与修补定位 ③ 训练侧局部收益评估 ④ 重放侧按递增顺序评估候选修补并通过阈值门控决定接受 ⑤ 多轮迭代与自适应重放维护。

**📊 数据集**

使用Flow-HO：一个合成的、基于JSON的可执行工作流修改基准，包含39个跨家族任务；同时在Haiku、Gemma、Ministral三种冻结模型上进行跨后端验证。

**📈 对比分析**

与基线（原始提示）及多种对比变体（如无重放门控、静态/自适应重放、不同聚合方式）进行实验；结果显示在Haiku上RIPPLE将服务验证成功率提升约23%（从54.7%到77.8%），复合奖励提升0.119；在其他后端同样提升6.8pp；同时保持或略升正确性，并获得更高的编辑效率与更低执行成本。

**⚠️ 局限性**

局限性包括：① 需要预先设计失败家族、片段划分与修补库，适配新任务或模型需手工维护；② 重放门控阈值经验性选择，可能在不同任务域下不稳健；③ 只在合成基准上评估，实际工业工作流的复杂度与多样性仍待验证；④ 对连续大规模编辑的累积效应研究有限。

---

## 194. USPLIT-VQA: U-Shaped Split Learning for Visual Question Answering with Contribution-Aware Weighted Aggregation

**arXiv ID:** 2609.12168 | [PDF](https://arxiv.org/pdf/2609.12168v1)

**作者:** Md Khalid Syfullah `[一作]` (Southern Illinois University Carbondale), Alvi Ataur Khalil `[通讯]` (Southern Illinois University Carbondale)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种U形拆分学习框架，用于在视觉问答(VQA)任务中实现数据与标签的本地化，保证隐私的同时显著降低客户端存储与通信成本。

**💡 创新点**

创新点包括：①U形拆分结构将模型分为客户端编码器+分类头与服务器中间层，既能保持标签隐私，又能减少客户端计算；②贡献感知加权聚合(CAWA)通过梯度相似度和信誉评分动态调节客户端贡献，提升对恶意或低质量更新的鲁棒性。

**🔧 技术方法**

使用技术主要有：U形拆分学习、梯度相似度评分、信誉权重(softmax温度缩放)、自适应阈值和记忆（streak）机制的CAWA聚合；实验基于PyTorch、ViT、CBAM、跨模态注意力等网络组件。

**📊 数据集**

数据集包括四个隐私敏感的医学视觉问答数据集：VQA‑RAD、SLAKE、PathVQA、VizWiz；使用两种骨干网络：BiomedCLIP（预训练大模型）和约8.5M参数的轻量级自定义模型。

**📈 对比分析**

对比方法：集中式训练与传统联邦学习。实验结果显示：对轻量级模型，U形拆分学习在四个数据集上平均精度可与联邦学习持平或超过；对BiomedCLIP在固定拆分点下精度略低；同时客户端内存降低5.7–5.8倍，通信量降低6–10倍。CAWA能将单个恶意客户端的影响降低超过98%，但随着恶意客户端比例升高，防御效果下降。对模型与梯度逆向攻击的实验表明，拆分学习恢复质量低于集中式训练。

**⚠️ 局限性**

局限性：①实验假设IID数据分布，未考虑临床真实的异构性；②BiomedCLIP在固定U形拆分点下精度受限，缺乏动态拆分策略；③CAWA对大量恶意客户端时鲁棒性有限；④大规模预训练模型在U形拆分下仍需进一步优化。

---

## 195. 3D Digital Twin Visualization of Multiclass GRF-Based Gait Disorder Classification

**arXiv ID:** 2609.12442 | [PDF](https://arxiv.org/pdf/2609.12442v1)

**作者:** Nayoung Son `[一作]` (Yonsei University), Minwoo Shin `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了集成双侧GRF/COP信号的Transformer分类框架并实现了3D数字孪生可视化。

**💡 创新点**

融合了类特定ε‑LRP可解释性与Blender 3D数字孪生，提供时间‑通道级别的模型解释与样本级交互可视化。

**🔧 技术方法**

Encoder‑Decoder Transformer、ε‑LRP解释、Blender 3D可视化、数据归一化/标准化等技术。

**📊 数据集**

使用公开的 GaitRec 数据集（12类步态失调）。

**📈 对比分析**

在会话级拆分上与传统 SVM、NN 等方法比较，验证精度 99% ，测试精度 90% ，单样本推理仅 4.9 ms。

**⚠️ 局限性**

对相似力学特征的病理类（如 K_R、A_R）召回仍偏低，且仅基于 GRF/COP 缺乏完整运动学信息。

---

## 196. The Cost of Compression: A Rate-Distortion Limit on Factual Hallucination

**arXiv ID:** 2609.12111 | [PDF](https://arxiv.org/pdf/2609.12111v1)

**作者:** Xi Wang `[一作]` (Hefei Institutes of Physical Science, Chinese Academy of Sciences), Rongfeng Guo `[通讯]` (Shenzhen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究闭书式事实问答中的错误来源，区分了缺失覆盖与压缩失真两种失败模式；

**💡 创新点**

首次将信息论的率失真理论应用于事实记忆，给出覆盖‑压缩下的误差下界，并提出可测量的诊断指标；

**🔧 技术方法**

使用随机源模型、率失真函数、LoRA适配、结构化事实（演绎闭包）以及长上下文模拟等技术；

**📊 数据集**

采用合成的无结构与结构化事实注入数据（10 类答案），以及公开的 Qwen、DeepSeek、Kimi‑K2、GLM‑5 语言模型；

**📈 对比分析**

通过观测-查询误差、未观测-查询误差和整体误差三种指标进行对比，发现随着事实负载增大或适配器容量固定，观测查询误差随预期上升，结构化事实延迟失真点；整体误差随记忆预算变化遵循率失真曲线；

**⚠️ 局限性**

局限在于假设完全无结构的随机映射、抽象的位预算与实际参数量不完全对应、未考虑地址编码成本与有限块长度影响，且实验仅限于人工注入的事实，未覆盖自然语言中常见的幻觉来源。

---

## 197. Motifs in temporal hypergraphs

**arXiv ID:** 2609.12175 | [PDF](https://arxiv.org/pdf/2609.12175v1)

**作者:** Quintino Francesco Lotito `[一作]` (Central European University), Giuseppe Francesco Italiano `[通讯]` (Luiss University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了时间超图（包括无向和有向）的网络基因（temporal motif）定义，并给出了完整的挖掘与统计评估框架。

**💡 创新点**

创新点在于：①将时间维度与高阶交互同时纳入基因定义；②构造了基于动态规划的高效枚举算法 LocalWindow‑DP，显著降低计算复杂度；③提出了窗口时间戳洗牌的无向/有向 null 模型，能够更准确评估基因显著性。

**🔧 技术方法**

使用的技术包括：组合学分析、两种 exact 算法（Baseline 与 LocalWindow‑DP）、动态规划、窗口时间戳洗牌、z‑score 统计检验。

**📊 数据集**

实验数据集涵盖四种领域：面向面接触（SocioPatterns）、科研合作（arXiv 计算机科学）、电子邮件（Enron）和比特币交易（2014年11月200k笔）。

**📈 对比分析**

与 Baseline 对比，LocalWindow‑DP 在四个数据集上实现了数到数十个数量级的速度提升；对 (K,L)=(4,4) 的情形，Baseline 需要超 24 小时，LocalWindow‑DP 仅需几分钟，验证了算法在大规模数据上的可行性。

**⚠️ 局限性**

局限性包括：①仅考虑 K≤4、L≤4 的基因，无法直接扩展到更大规模；②算法仍为 exact，处理更大图仍受内存/时间限制；③统计评估仅基于 z‑score 排序，未进行严格的假设检验或置信区间估计。

---

## 198. Computing the Shortest Even Directed-Cycle Length in $\widetilde{O}(n^4)$ Time

**arXiv ID:** 2609.12021 | [PDF](https://arxiv.org/pdf/2609.12021v1)

**作者:** Hanqing Li `[一作]` `[通讯]` (Peking University), Hanqing Li (Peking University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个随机化 (n^4) 级别的算法，用来求有向图中最短偶循环的长度。

**💡 创新点**

创新点在于直接利用特征 2 场上的逆矩阵公式，避免了此前需要提升到四阶环的代数转移，并实现了 O(n^3) 的奇排列求和。

**🔧 技术方法**

主要技术包括特征 2 上的 Jacobi 互补最小子式公式、行消元与逆矩阵动态维护、以及多点插值与身份检测。

**📊 数据集**

使用的是任意有向图，实验数据通过在扩展域上随机赋权来产生。

**📈 对比分析**

与 Björklund 等人提出的 (n^ω+3) 随机化算法相比，本文的算法将时间从 O(n^ω) 降至 O(n^4)，误差概率仅为 O(n^-3)，且单向错误保证了无偶环时的准确性。

**⚠️ 局限性**

局限性包括只能输出长度而非具体循环，需要足够大的特征 2 域、存在随机化误差且仅适用于偶循环问题。

---

## 199. When Agent Metrics Measure Different Things: An Evidence-Grounded Audit of the Praxa AI Pipeline

**arXiv ID:** 2609.12017 | [PDF](https://arxiv.org/pdf/2609.12017v1)

**作者:** Stefan G. Creadore `[一作]`, Peyton Woakz `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Praxa AI 系统的测量实现和历史评估结果进行回顾性审计，验证了计数和统计计算，复现了评分函数测试，分析了运营时延字段的生命周期解释，并重新计算了单条轨迹的压缩实验的 token 减少量；同时提出了测量契约与缺失数据敏感性分析。

**💡 创新点**

首次将源代码与历史数据绑定，提供可重现的算术验证器（加权算术与 rational‑arithmetic 检查）和缺失数据边界；提出了分层测量边界和事件语义的契约框架，以防止度量结果被误解。

**🔧 技术方法**

使用 SQL 只读查询提取数据库统计，TypeScript 环境下执行原始评分函数和自定义验证器，采用 Python rational arithmetic 计算加权均值与分位数，利用 MD5 校验数据完整性。

**📊 数据集**

主要数据集包括：历史离线路由报告（139 条案例）、运营工具尝试导出（8843 条记录）以及一份单轨迹压缩实验的原始报告；全部数据均为只读快照，未进行实时重跑。

**📈 对比分析**

通过精确的算术重算和验证器测试来比较结果；发现路由门通过率为 80.6%，单个 follow‑up token 减少率为 94.39%，若合计 trigger 与 follow‑up，整体减少率为 46.54%；但缺失值和计时字段的生命周期解释导致对实际延迟的估计存在偏差。

**⚠️ 局限性**

未对生产流量进行独立重跑，缺少真实任务完成的客观标签，缺失数据比例约 5%，无法进行统计推断；压缩实验仅包含单一轨迹，无法估计总体效能；测量字段与实际执行事件混淆，导致测量结果易被误读；总体结论仅限于有限的案例研究，缺乏外部可验证性。

---

## 200. Toward Reliable Railway-Bogie Response Prediction Using Multifidelity TDNN and Physics-Informed Residual Learning

**arXiv ID:** 2609.12018 | [PDF](https://arxiv.org/pdf/2609.12018v1)

**作者:** Gyeolhee Lee `[一作]`, Dongjin Lee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出一种多保真度校正方法，将低保真多体动力学仿真历史与高保真滚轮机测量结果相结合，并通过物理信息化残差学习来预测高速铁路车辆悬挂的响应；

**💡 创新点**

创新点在于将实验锚定的多保真度学习与有效动力学平衡约束相结合，能够区分潜在模型差异与测量误差，并得到符合物理约束的校正网络；

**🔧 技术方法**

采用的技术包括时延神经网络（TDNN）构建低保真趋势、物理信息化神经网络（PINN）与残差校正网络、有效质量-阻尼-刚度平衡约束以及残差平滑、加速度一致性等多项损失函数；

**📊 数据集**

使用的数据集为一台原型车厢的滚轮机实验数据，涵盖五种速度（300、320、340、360、385 km/h），其中包含四通道位移（LVDT）和十六通道加速度信号；与之对应的低保真仿真数据来自SIMULIA Simpack；

**📈 对比分析**

在与仅基于实验的PINN以及仅使用TDNN残差模型的对比中，本文校正模型在保留未见的385 km/h条件下取得平均R²≈0.82、NRMSE≈4.6%、NMAE≈1.93%，显著优于其他方法；

**⚠️ 局限性**

局限性包括仅在单一保留条件上评估、未覆盖更宽的速度/激励范围、传感器到坐标的映射尚未完全验证、缺乏损失项消融研究以及预测不确定性量化不足。

---

## 201. UFO: Chain-of-Evaluation for Omni-Condition Alignment in Multi-Modal Image Generation

**arXiv ID:** 2609.12397 | [PDF](https://arxiv.org/pdf/2609.12397v1)

**作者:** Danning Zhang `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出UFO统一框架对多模态图像生成进行原子化链式一致性评估，并构建UFO-Bench基准；

**💡 创新点**

通过将全局条件拆解为细粒度AEU、模态相关性分类及专用功能调用，实现对文本与视觉条件同步一致性的高精度评估；

**🔧 技术方法**

使用VLM（如GPT‑4o）进行问答评估、ArcFace等专家工具进行身份验证，并采用适应性加权聚合方式；

**📊 数据集**

UFO‑Bench数据集，包含86张参考图、7类主体（硬体、软体、人物、全身角色、动物、logo、场景）、3层编辑难度和81.97%冲突条件对；

**📈 对比分析**

与CLIP、DINO、VIEScore、DreamBench++等传统指标对比，Spearman相关系数平均提升约15.25%，在UFO‑Bench上获得最高人类一致性得分；

**⚠️ 局限性**

仍需依赖大规模VLM算力，且对极端局部细节或复杂模态冲突的捕捉有限，验证范围局限于单一生成任务。

---

## 202. Evaluating Practical Enumeration and Blocking Attacks on the Snowflake Circumvention System

**arXiv ID:** 2609.12242 | [PDF](https://arxiv.org/pdf/2609.12242v1)

**作者:** Linden Chen `[一作]` (University of California, Santa Cruz), Ram Sundara Raman `[通讯]` (University of California, Santa Cruz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过结合48天的受限实时测量和大规模仿真，评估了恶意客户端在Snowflake中枚举代理IP并随后实施IP级与AS级封锁的可行性与影响。

**💡 创新点**

首次系统性、经验化地检验Snowflake的枚举与封锁抵抗，揭示AS级封锁在保持低域名碰撞的前提下可导致大量代理不可用，并提出了基于速率限制、代理轮询调度与增大代理流失率的实用防御措施。

**🔧 技术方法**

技术手段包括：①对Snowflake broker进行主动低速枚举；②使用RouteViews映射IP到AS；③基于IPInfo、Tranco、APNIC、CAIDA等公开数据构建代理与用户统计；④开发Go语言的仿真框架，复现broker匹配、代理流失与连接持续时间；⑤采用SHA256哈希匿名记录IP。

**📊 数据集**

使用的数据集包括：• 2025年5–6月Snowflake的实时代理响应（共21,443个唯一IP，935个AS）；• IPInfo AS类型分类；• Tranco Top 1M域名列表；• APNIC AS人口与CAIDA AS Rank信息；• RouteViews前缀-AS映射。

**📈 对比分析**

通过对不同攻击者规模、代理流失率和连接时长的仿真参数进行实验，比较枚举覆盖率、阻塞失败率、重试次数等指标。结果表明，攻击者规模增大导致枚举覆盖率提升，代理流失率升高可显著降低封锁效果；在默认设置下，AS级封锁可使大部分客户端重试数升至百级，体现出高风险。

**⚠️ 局限性**

局限性包括：①未进行真实封锁实验，仅估计重试成本；②碰撞评估仅基于域名重叠，未涵盖更广泛的社会/经济影响；③仿真省略了完整的WebRTC/Tor流量、网络时延与丢包；④代理流失模型简化，未完全反映真实地理与时间变异；⑤实时测量受限于安全约束，攻击者规模受限，结果可能低估极端攻击效果。

---

## 203. Single-Query Person-Centric Bimanual Hand-Object Interaction Detection

**arXiv ID:** 2609.12155 | [PDF](https://arxiv.org/pdf/2609.12155v1)

**作者:** Jonghyun Kim `[一作]` (LG Electronics), Jungho Lee `[通讯]` (LG Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出单查询的人体中心双手交互检测框架，预测人体、姿态、双手框和交互目标。

**💡 创新点**

创新点在于将双手归属为同一人体的结构化输出，使用部分感知可变形注意力和手-查询关系矩阵，避免手中心独立预测。

**🔧 技术方法**

采用DETR/RT-DETR变压器、可变形注意力、关系矩阵、学习的无接触token，联合检测、姿态估计与交互推理。

**📊 数据集**

基于COCO+Hands23构建的新数据集，包含人体框、关键点、双手框、交互对象和类别，并提供手状态标签。

**📈 对比分析**

与基线（直接回归、无姿态监督）对比，关系矩阵+姿态监督模型在COCOval上检测mAP≈48.6、姿态AP≈63.2、软/中/硬交互精度分别提升到84.0/66.2/32.7，明显优于传统手中心方法。

**⚠️ 局限性**

在高度拥挤或遮挡严重的场景下性能显著下降，数据标注仍有噪声，模型对小手和复杂交互的鲁棒性有限。

---

## 204. When2Talk: When Should a Proactive In-Car Agent Talk?

**arXiv ID:** 2609.12503 | [PDF](https://arxiv.org/pdf/2609.12503v1)

**作者:** Kaiser Hamid `[一作]` (Texas Tech University), Nade Liang `[通讯]` (Texas Tech University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在自动驾驶车辆中，研究了一种上下文感知的车内主动沟通策略（CS），将事件立即、延迟或保持沉默的三种传递方式与传统事件触发（ET）策略进行比较，以提高沟通适宜性并减少乘客打断感。

**💡 创新点**

创新点在于首次将事件优先级、乘客活动与信息持续价值结合，设计了可选择立即、延迟或不发声的通信策略，并通过VR仿真验证其对乘客体验的多维影响。

**🔧 技术方法**

技术手段包括基于CARLA的仿真环境、Varjo XR‑4混合现实头戴式显示器、200 Hz双眼瞳孔追踪、预录语音消息以及同步的实验控制脚本。

**📊 数据集**

使用了自制的六种事件情境数据集（行人穿行、工作区、堵车、ETA增加、轻微减速、紧急车辆）以及三种乘客活动（道路监测、间歇电话、持续电话）作为实验输入。

**📈 对比分析**

通过在41名受试者中进行2×2×3的交叉设计，使用混合效应方差分析和配对t检验对比CS与ET，结果显示CS在沟通适宜性上提升了≈0.38（d_z=0.38），在感知打断上显著降低了≈1.07（d_z=−1.07），在信息有用性上仅在沉默情境表现更佳，整体信任差异不显著。

**⚠️ 局限性**

局限性包括仅预设六种情境且CS映射固定、延迟时间与触发时机人为设定、缺乏真实道路和自然语音交互、样本仅为大学生且未评估记忆或回顾误差，以及未检验不同语言或文化背景下的适用性。

---

## 205. GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents

**arXiv ID:** 2609.12191 | [PDF](https://arxiv.org/pdf/2609.12191v1)

**作者:** Umesh Bodhwani `[一作]` (Amazon), Kai Wei `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种离线评估协议GAUGE，用以检验基于LLM用户模拟器和LLM判别器的门控系统在选型中的排名和构造有效性；

**💡 创新点**

创新点在于将门控系统的排名有效性与人类满意度与任务成功之间的“满意-成功差距”分离，并给出了零成本的完成位检测方法；

**🔧 技术方法**

采用LLM判别器、LLM人类代理、人工面板以及非LLM可验证奖励等多种评估技术；

**📊 数据集**

主要使用了τ^2-bench（零售与航空任务）和SimulatorArena（数学辅导）两大基准数据集；

**📈 对比分析**

通过Spearman相关、AUC等指标，门控系统在25个代理上的排名相关性高达0.94，但在近似等效代理间的决策一致率仅为31%；

**⚠️ 局限性**

局限包括人工面板规模有限、降级集仅覆盖截断错误、以及对真实客户交互的泛化仍待验证。

---

## 206. Temporal Recurrence Favors Fewer Layers

**arXiv ID:** 2609.12531 | [PDF](https://arxiv.org/pdf/2609.12531v1)

**作者:** Ivan Anokhin `[一作]` (Mila Quebec Ai Institute), Sebastian Risi `[通讯]` (Sakana Ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在流式任务中时间递归与每步深度的计算分配关系，探讨在保留外部递归状态的情况下，每步需要多少层深度来实现最佳性能。

**💡 创新点**

创新点在于将深度与并行计算视为可调节的资源维度，通过匹配计算预算比较递归与非递归模型，从而发现递归可显著降低每步所需层数，同时保持或提升性能。

**🔧 技术方法**

使用了并行专家（Parallel Experts）框架，结合ConvLSTM/ConvGLU（在Sokoban实验）以及Transformer（在FineWeb语言建模实验），并对专家数与宽度进行可调。

**📊 数据集**

实验数据集包括可观测的Sokoban游戏环境（Jumanji）和流式FineWeb文本数据集（约10亿词）。

**📈 对比分析**

通过在相同每步计算预算下对不同层数、专家数与宽度组合进行系统 sweep，比较了递归与非递归模型在Sokoban的平均回报和FineWeb的验证交叉熵；结果显示递归模型在层数 2-4 时即可达到与非递归模型 8-16 层相同或更优的性能。

**⚠️ 局限性**

局限性包括实验仅覆盖有限的任务与模型种类、计算预算匹配仅为近似，未评估实际硬件延迟与吞吐；递归模型的训练可能受限于截断反向传播，且结果可能随架构细节与优化策略变化。

---

## 207. Scenario-Independent Criticality Assessment and Prediction for Vulnerable Road Users in Autonomous Driving

**arXiv ID:** 2609.11947 | [PDF](https://arxiv.org/pdf/2609.11947v1)

**作者:** Jörg Gamerdinger `[一作]` (University of Tübingen), Oliver Bringmann `[通讯]` (University of Tübingen)

**通讯引用:** 3001 | [OpenAlex ID](https://openalex.org/A5074802358)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了针对脆弱道路使用者（VRU）的关键性评估指标，并构建了一个不依赖场景的关键性预测框架，同时基于DeepAccident数据集创建了首个含5.46M标注的感知关键性数据集。

**💡 创新点**

创新点包括：①基于空间占据指数与VRU运动不确定性设计的新关键性度量；②利用统计学习方法的场景无关关键性预测框架；③首次公开发布包含多种关键性指标输出的完整数据集。

**🔧 技术方法**

使用了空间占据模型、圆形/多边形运动预测、统计机器学习（KMeans、DBScan、Isolation Forest、GMM、逻辑回归、随机森林、Histogram‑based Gradient Boosting、LightGBM）以及阈值调优与交叉验证技术。

**📊 数据集**

采用DeepAccident合成数据集（57K帧、690场景），扩展为161k帧、5.46M对象的关键性标注数据集，包含多种基准关键性指标。

**📈 对比分析**

与SOTA指标（TTC、MTTC、TTA、CIF、RSS、SACRED）比较，VRU专用指标F1从0.39提升至0.57（+50%），预测框架F1从0.35提升至0.94（+275%），随机森林表现最佳。

**⚠️ 局限性**

局限性在于仅基于单帧信息，缺乏时间序列建模；预测模型仍为传统统计方法，未深入利用深度学习的时序特征；VRU运动模型为保守假设，可能不适用于所有道路环境；未对不同城市道路进行充分验证。

---

## 208. GUIDE: Generative Utility Inference and Decision Engine

**arXiv ID:** 2609.12137 | [PDF](https://arxiv.org/pdf/2609.12137v1)

**作者:** Anagha Tiwari `[一作]` (University of Chicago), Alex Kale `[通讯]` (University of Chicago)

**通讯引用:** 1062 | [OpenAlex ID](https://openalex.org/A5001536494)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了GUIDE框架，结合LLM交互、贝叶斯自适应抽样与符号规则初始化，评估其在金融组合优化场景下的用户偏好挖掘与推荐质量。

**💡 创新点**

创新点在于将LLM作为交互接口而非偏好模型，扩展多类型问题变换的EIG选择，使用符号规则学习提供经验先验，并构建可观察、可调节的整体架构。

**🔧 技术方法**

采用贝叶斯优化（EIG）、粒子滤波后验、Bradley–Terry 似然、符号规则学习（ILASP）、两阶段问题选择与Claude Opus 4.7等LLM实现技术。

**📊 数据集**

使用National Financial Capability Study (NFCS) 投资者调查数据做域初始化与先验，75个由LLM生成的模型组合作为决策空间。

**📈 对比分析**

通过与四种LLM基线（GPT/Claude 开放式或对比式）、三种GUIDE消融版本以及OPEN、PEBOL 两个先前方法，在5个投资者角色、15轮提问的仿真实验中比较，GUIDE在冷启动和早期轮次显著降低 regret，NRR最高，最终稳定性与单一对比式方法相当。

**⚠️ 局限性**

局限包括：早期多样化问题在后期可能引起回弹；离散动作空间限制连续优化；LLM生成特征的质量依赖；实验仅在模拟客户端，缺乏真实人类用户验证。

---

## 209. "The Only Thing Certain About This is Uncertainty": Exploring Informal Care Coordination Practices Among Older Adults with Mild Cognitive Impairment

**arXiv ID:** 2609.12070 | [PDF](https://arxiv.org/pdf/2609.12070v1)

**作者:** Josey M. Benandi `[一作]` (Georgia Institute of Technology), Agata Rozga `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 4106 | [OpenAlex ID](https://openalex.org/A5085265585)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对MCI患者及其护理伙伴的访谈进行主题分析，探讨日常护理协调的方式与工具。

**💡 创新点**

提出“协调即编排”(orchestration)概念，将护理工作视为情感与技术协同的复杂过程，并阐明MCI护理是独立的设计空间。

**🔧 技术方法**

采用定性访谈、编码分析与多维度(人-活动-工具)框架，无需专门技术模型。

**📊 数据集**

收集了6名MCI患者与10名护理伙伴的深度访谈数据，来自加州技术学院的护理项目。

**📈 对比分析**

通过对比分析发现传统技术缺乏对“编排”需求的支持，提供了可行的设计启示；未进行量化性能评估。

**⚠️ 局限性**

样本规模小且多为伴侣型护理伙伴，缺乏独居患者视角，访谈数据主要来自护理伙伴，可能导致偏见。

---

## 210. Direct Topology Tracking in Continuous Implicit Models

**arXiv ID:** 2609.12157 | [PDF](https://arxiv.org/pdf/2609.12157v1)

**作者:** Guanqun Ma `[一作]` (University of Utah), Bei Wang `[通讯]` (University of Utah)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种直接在连续隐式模型（多元功能逼近MFA和隐式神经表示INR）上跟踪临界点的框架，避免离散化带来的误差；

**💡 创新点**

通过在连续域中利用Feature Flow Fields原理与 Newton 校正相结合，并针对退化事件进行局部重构，实现了对时间变化域内临界点的连续跟踪；

**🔧 技术方法**

使用解析函数求梯度和Hessian、MFA的B样条解析导数、INR的自动微分、RK4 预测+Newton 校正、退化点检测和路径拆分等技术；

**📊 数据集**

实验数据集包括三种解析函数（四次势能、四次旋转）、两组二维科学数据（Vortex Street、Heated Cylinder）以及一组三维Vortex数据，均构建了MFA和CoordNet INR模型；

**📈 对比分析**

与传统基于离散采样的Stable Feature Flow Fields（SFFF）和Lifting Wasserstein Matcher（LWM）对比，所提方法在梯度误差、Hessian退化判定、内存占用上更优；轨迹结构与SFFF相似，优于LWM；运行时间方面，MFA耗时最低，INR由于反向传播较慢，耗时最长；

**⚠️ 局限性**

主要限制包括：INR的梯度评估成本高、在近常量区域临界结构不明显、需要多次参数调优、种子分布不足可能漏检特征、对相近轨迹的辨识受空间时间阈值限制，无法保证完全恢复所有轨迹。

---

## 211. GLARE: Generative Learning via Adversarial Reward Estimation For Social Dynamics Forecasting

**arXiv ID:** 2609.12165 | [PDF](https://arxiv.org/pdf/2609.12165v1)

**作者:** Tenghao Huang `[一作]` (Microsoft), Sihao Chen `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出会议续写预测任务与对应的Meeting Dynamic Forecasting Benchmark (MDFB)，并提出适用于长时序生成的对抗奖励学习框架GAIR；

**💡 创新点**

创新点在于将对抗模仿学习迁移至条件化语言生成，利用判别器对正样本与当前策略生成样本的多类排序，动态更新奖励；

**🔧 技术方法**

技术主要包括基于LLM的半自动数据构建、判别器+值头的奖励模型、KL正则化的策略优化以及与SFT、SPIN等基线的对照实验；

**📊 数据集**

数据集为MDFB，由2,207个真实会议转录及24,794个未来面向查询构成，涵盖多方会议场景；

**📈 对比分析**

与SFT、Vanilla Qwen、SPIN等对比，GAIR在人工评估的实用性和人类般性两轴均提升至0.66和0.70的胜率，仍低于人类参考；

**⚠️ 局限性**

局限包括对抗奖励仍易被策略利用、对长时序文本的评估主观性高、且未对模型的普适性做完整测试，未来仍需更强的奖励学习与更广泛的跨模型评测。

---

## 212. Signed Sensitivity of Expected Hitting Time to Mutation Rate in the (1+1) EA: Per-State Sign Theorems and Verifiable Certificates for Non-Lumpable Families

**arXiv ID:** 2609.12510 | [PDF](https://arxiv.org/pdf/2609.12510v1)

**作者:** RenKai Wang `[一作]` `[通讯]` (Institute of Automation, Chinese Academy of Sciences), RenKai Wang (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了 (1+1) EA 在不同突变率下期望击中时间的连续性，并给出其导数符号的严格判定

**💡 创新点**

首次将符号 Margulis–Russo 影响分析、双残差证书和非可聚类族的区间方法系统性地应用于演化算法运行时间，证明了在随机初始化下最优静态突变率大于 1

**🔧 技术方法**

使用符号 Margulis–Russo 公式、矩阵微分与双残差区间证书、精确有理数算术与区间算子，以及对非可聚类线性目标的块间隔分析

**📊 数据集**

利用精确有理数算术对具体实例（如 n=6 的权重向量 (1,65,75,85,95,2)，以及对一般 n 的均匀分组线性目标）进行验证

**📈 对比分析**

通过对比 1/n 固定率下已知的期望运行时间与对不同 c 值的符号判定，发现所有 n 下随机初始化的最优 c 均严格大于 1；在 n=6 的异质实例中，57/63 状态在整个参数区间内导数均为负，说明均匀率远小于 1 的情况下仍能下降

**⚠️ 局限性**

对区间宽度、块级别分割精细度及更大 c 区间的推广尚未完全解决；该方法目前仅适用于 (1+1) EA 的单一突变率，未覆盖种群规模 λ、适应度依赖率等更复杂情形

---

## 213. On Identifying Adversarial Intent Injection in AI-Native 6G Networks

**arXiv ID:** 2609.12144 | [PDF](https://arxiv.org/pdf/2609.12144v1)

**作者:** Nilesh Chakraborty `[一作]` (University of Ottawa), Burak Kantarci `[通讯]` (University of Ottawa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对AI‑native 6G网络中的意图注入攻击，构建了四种注入策略的数据集，并提出双路径检测框架。

**💡 创新点**

创新点在于将注入频率作为威胁维度，引入基于TF‑IDF的上下文窗口特征，使用CNN和一类AutoEncoder两种模型联合检测。

**🔧 技术方法**

技术主要包括TF‑IDF特征提取、滑动窗口上下文编码、1D CNN分类器与卷积自编码器，并采用焦点损失和类权重处理不平衡。

**📊 数据集**

使用由人工设计的20条基础意图扩展到约1100条意图的混合数据集，其中恶意样本覆盖DoS、钓鱼、数据泄露等四种注入策略。

**📈 对比分析**

与基线DIET模型比较，CNN与AutoEncoder在所有四种注入模式下的准确率和F1分数均超过基线，平均准确率约0.90，F1约0.93。

**⚠️ 局限性**

局限性包括数据集规模有限、仅覆盖JSON/TOSCA样式意图、缺乏真实运营环境验证以及解释性不足。

---

## 214. ChemMat-AgentSafetyBench: Evaluating Long-Horizon Attacks and Defenses in Chemistry and Materials Agents

**arXiv ID:** 2609.11952 | [PDF](https://arxiv.org/pdf/2609.11952v1)

**作者:** Zhan'ao Yao `[一作]`, Jianjun Liu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文介绍了ChemMat-AgentSafetyBench，这是一个用于评估化学和材料科学中AI代理安全性的基准测试。

**💡 创新点**

创新点在于提出了一个专门针对化学和材料科学领域的AI代理安全性评估框架。

**🔧 技术方法**

使用了机器学习和深度学习技术来评估AI代理的安全性。

**📊 数据集**

使用了多个化学和材料科学相关的数据集进行实验。

**📈 对比分析**

与现有的安全性评估方法进行了比较，结果显示该方法在准确性和可靠性上有显著提升。

**⚠️ 局限性**

限制在于数据集的多样性和规模可能影响评估结果的普适性。

---

## 215. Vortex: Bridging Extreme Compression and Efficient LLM Inference

**arXiv ID:** 2609.12208 | [PDF](https://arxiv.org/pdf/2609.12208v1)

**作者:** Haoxuan Shan `[一作]` (Duke University), Yiran Chen `[通讯]` (Duke University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 Vortex 的双流架构，实现了大语言模型推理的极端压缩与高效执行。

**💡 创新点**

创新点在于将向量量化、KV 缓存即时量化与按代码本的上下文稀疏三种技术联合设计，并通过专用查找单元和缓冲机制克服 VQ 的内存冲突，兼顾 prefill 与 decode 两个阶段的高效执行。

**🔧 技术方法**

采用向量量化、代码本量化、动态 KV 缓存量化、上下文稀疏检测与剔除以及基于流水线的双流执行。

**📊 数据集**

使用 Llama2‑7B、Llama2‑13B、Mistral‑7B 三个模型，并在 ARC Easy/Challenge、COPA、OpenBookQA、PIQA、Winogrande 等基准数据集上进行评估。

**📈 对比分析**

相较于传统的 systolic‑array、ANT、FIGNA 与 FIGLUT 等基线，Vortex 在端到端推理任务中实现了 8.03×–23.8× 的加速比、5.68×–13.1× 的能耗降低，显示显著优势。

**⚠️ 局限性**

局限性包括在大批量 decode 阶段对稀疏性的利用受限、KV 缓存稀疏未被充分发挥、对硬件资源与 DRAM 带宽的依赖，以及在极低位宽（<2bit）下仍存在精度与实现复杂度的权衡。

---

## 216. Fast matrix multiplication via recursive $\langle$ 4x4x4:48 $\rangle$ algorithms into practice

**arXiv ID:** 2609.12027 | [PDF](https://arxiv.org/pdf/2609.12027v1)

**作者:** Jean-Guillaume Dumas `[一作]` (University of Grenoble Alpes), Petr Tichavský `[通讯]` (Academy of Sciences of the Czech Republic)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种使用48次标量乘法来计算两个4×4矩阵乘积的新算法，并通过稀疏化LRP表示、增大基向量数量以及引入基变换等手段，显著降低了算法的额外加、减、标量缩放操作数。

**💡 创新点**

创新点在于：① 将LRP表示稀疏化到仅220个非零系数（含176个非单位）并保持10个基向量，② 采用“替代基”技术将操作数进一步压缩到176，③ 在对称的L和R矩阵上利用行列置换和符号变换，最大化基向量利用率，从而获得迄今为止最小的常数6.5；同时给出了对应的直线程序和变换矩阵。

**🔧 技术方法**

使用的技术包括：张量分解与LRP表示、行列稀疏化、基向量最大化策略、变换矩阵X_×X_（替代基）以及直线程序（SLP）优化，配合随机子表达式消除、核方法和线性依赖检测等启发式搜索。

**📊 数据集**

实验数据集主要是符号4×4矩阵（A、B）作为输入；论文中未使用实际数值数据集，而是通过符号计算和操作数计数来评估算法复杂度。

**📈 对比分析**

与先前的49、47乘法（log4 48≈2.79）和48乘法（log4 48≈2.79）的实现进行比较，本文的额外操作从10.84降到7.75，再通过替代基降到6.5，导致递归时的主项常数从≈10.84n².7925下降到≈6.5n².7925，显示出显著的性能提升。

**⚠️ 局限性**

限制在于：对LRP表示的几何结构尚未完全理解，仍有可能存在更优的稀疏化或变换方案；此外，虽然理论上常数最优，但在实际数值实现与数值稳定性方面尚未进行充分实验验证。

---

## 217. Estimating Pedestrian Volumes from GIS-Derived Built-Environment Features: A Machine Learning Framework

**arXiv ID:** 2609.12173 | [PDF](https://arxiv.org/pdf/2609.12173v1)

**作者:** Bahareh Golchin `[一作]` (Portland State University), Joseph Broach `[通讯]` (Portland State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套完整的机器学习管道，用于从 GIS 派生的建成环境特征预测城市交叉口的人行道客流量。

**💡 创新点**

创新点在于在小样本高维环境下系统化地结合特征选择、重复交叉验证和超参数调优，并将传统负二项 GLM 与 Histogram‑based Gradient Boosting Poisson 结合实现显著性能提升。

**🔧 技术方法**

采用 L1 Lasso 特征筛选、Histogram‑based Gradient Boosting（Poisson 损失）、随机森林重要性、交叉验证、学习率/深度/叶子数调参等技术。

**📊 数据集**

使用 101 个波特兰市交叉口的手工计数数据以及由 Strava、OSM、美国社区调查（ACS）、LEHD 等公开 GIS 数据构建的 92+ 维特征集。

**📈 对比分析**

与公开的负二项 GLM 基线对比，交叉验证 RMSE 由 89.8 降至 78.7（12%），保留集 RMSE 由 108.0 降至 87.9（19%），在 MAPE 与 SMAPE 上亦实现显著改进。

**⚠️ 局限性**

局限在样本量仅 101，且分布主要集中在市中心与项目通道，模型在不同城市或更大范围内的泛化性待验证。

---

## 218. Exploring Climate-Related Anxiety Through Social Media Content

**arXiv ID:** 2609.12129 | [PDF](https://arxiv.org/pdf/2609.12129v1)

**作者:** Karim El-Sharkawy `[一作]` (Sprout Climate Association), Bobbie Williams `[通讯]` (Sprout Climate Association)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用自然语言处理技术对Reddit上的气候相关讨论进行主题建模和情绪分类，以探究气候焦虑的主题结构、情绪表达和时间演变

**💡 创新点**

首次在同一框架下同时分析主题与情绪，揭示讨论分为行动导向与信息/反思两大模式，并指出评论往往通过关怀、鼓励等情绪重塑讨论氛围

**🔧 技术方法**

采用BERTopic（基于MiniLM嵌入+UMAP+HDBSCAN）进行主题建模，使用RoBERTa-Base GoEmotions模型进行28类情绪分类，并结合动态主题建模与可视化工具（Plotly、词云）

**📊 数据集**

使用来自Reddit公开讨论的英文文本，经过关键词过滤（27条气候焦虑相关关键词）后得到的原始发帖和评论数据集，时间跨度多个月级别

**📈 对比分析**

未与传统情绪分析或主题模型做严格性能对比，结果以主题出现频率、情绪分布、时间序列趋势等定量指标展示；总体发现负面情绪占比高，但评论中出现支持性情绪显著增加，说明社区互动能缓解焦虑

**⚠️ 局限性**

局限包括仅使用英文Reddit数据，无法代表所有青年用户；模型可能对讽刺、俚语、短文本处理不佳；缺乏人工标注验证主题与情绪准确性；未进行因果或多平台验证，结果仅为描述性观察

---

## 219. Autonomous Precision Milling of Biological Structures via Generic Anatomical Priors and Active Boundary Perception

**arXiv ID:** 2609.12530 | [PDF](https://arxiv.org/pdf/2609.12530v1)

**作者:** Enduo Zhao `[一作]` (Tsinghua University), Kanako Harada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一套基于通用解剖先验与主动边界感知的自主精密生物结构铣削框架，利用状态自适应控制实现从全局规划到局部精细化的闭环加工。

**💡 创新点**

创新点包括：①将群体平均解剖先验与个体表面进行语义引导配准并通过视觉-力学双重校准实现机器人可执行路径迁移；②使用相对刚度变化的连续边界支持损失指数进行主动边界感知，提升了对未知内部结构的判别能力；③构建有限状态机，将主动感知、增量精细化与终止判定结合，形成迭代感知‑精细化闭环。

**🔧 技术方法**

采用 Mask2Former 语义分割网络进行表面标签化，语义引导 ICP 进行配准；混合视觉与力感知的双重校准；相对刚度指数计算、阈值聚合；6 轴力/扭矩传感器与微型立体相机系统；8 轴机械臂与微型钻头；有限状态机控制器实现状态转移。

**📊 数据集**

使用基于 30 条 8 周龄雌性小鼠的平均解剖先验模型；鸡蛋壳实验样本 40 只作为生物替代物；15 只雌性 NOD.CB17-Prkdcscid/J 小鼠做实物铣削验证；同时利用 MSCS 生成的点云、力/扭矩数据进行配准与感知评估。

**📈 对比分析**

与基线方法（传统 ICP 配准、仅视觉校准、基于绝对刚度的探测）相比：配准误差平均降低 83.4%（平移）与 57.3%（旋转）；铣削后轮廓误差均小于 0.05 mm；深度一致性均值误差 < 0.01 mm；主动边界感知局部分类 100% 准确、全局可分辨率 95%；闭环状态自适应系统圆形窗口成功率 100%（对比开放式 25%），并实现多种形状窗口；所有 15 次体内实验均成功完成，虽平均时间略增。

**⚠️ 局限性**

局限性包括：假设目标相对静止，未对生理运动或动态变形做补偿；增量精细化策略保守，导致加工时间增加；相对刚度判别依赖于阈值，接近阈值时仍可能出现误判；未对表面出血等临床事件建模；当前不提供概率性不确定性估计，依赖经验权重控制。

---

## 220. SoK: Rethinking Jailbreaking in the Era of Agentic AI: Attacks, Defenses, and Practical Consideration

**arXiv ID:** 2609.12413 | [PDF](https://arxiv.org/pdf/2609.12413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 221. Exact Community Recovery in Bipartite Networks

**arXiv ID:** 2609.12445 | [PDF](https://arxiv.org/pdf/2609.12445v1)

**作者:** Huan Qing `[一作]` `[通讯]` (Chongqing University of Technology), Huan Qing (Chongqing University of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了两种谱聚类算法（SCDD 与 NSCDD），用于在随机共块模型（ScBM）及其度数校正版本（DCScBM）下实现二分网络的精确社区恢复。

**💡 创新点**

创新点在于给出了在稀疏、社区不平衡且社区数随网络规模增长的情况下，利用去对角线的 Gram 矩阵（以及行归一化）实现高概率精确恢复的理论保证，并将该框架推广到度数校正模型。

**🔧 技术方法**

技术手段包括去对角线 Gram 矩阵构造、特征向量提取、行归一化、k-means 聚类，以及基于奇异值分解和扰动理论的误差分析。

**📊 数据集**

实验采用合成数据：根据预设的 B 矩阵、社区划分、大小比例和稀疏参数生成 ScBM 与 DCScBM 的邻接矩阵，未使用真实数据集。

**📈 对比分析**

与 DI‑SIM、D‑SCORE、BiSC 与 nBiSC 等基线方法在精确恢复比例上进行比较，实验结果与理论阈值一致，说明算法在满足理论条件时能够实现零误分类。

**⚠️ 局限性**

局限性包括：仅适用于二进制邻接矩阵；分析要求平均度数至少为多对数级；对更稀疏的情形以及加权、层次或多层二分网络的精确恢复尚未覆盖；拉普拉斯归一化方法的精确恢复仍是未解决的问题。

---

## 222. Missing Dimensions: Integrating Human and Social Systems into Digital Twin Engineering

**arXiv ID:** 2609.12131 | [PDF](https://arxiv.org/pdf/2609.12131v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 223. Certified AI Triage of ICU Alarms

**arXiv ID:** 2609.12365 | [PDF](https://arxiv.org/pdf/2609.12365v1)

**作者:** Mohammed Sameer Syed `[一作]` (Roshan AI), Rozhin Yasaei `[通讯]` (University of Arizona)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究提出了VT报警的三方分类（保留、抑制、推迟）策略，并通过Learn-then-Test框架给出抑制误报的有限样本安全保证，同时分析了策略网格多重性成本并估算所需标注量。

**💡 创新点**

首次在VTaC基准上给出VT报警抑制的有限样本安全上限，揭示网格细化后多重性校正导致的可证性下降，提供置信度校准体积估算，并对加入可靠性得分的负面结果进行了系统性阐述。

**🔧 技术方法**

采用Learn-then-Test统计保证、Bonferroni多重性校正、记录聚类自助法评估依赖性、置信度校准体积投影以及可学习的可靠性维度。

**📊 数据集**

使用公开的VTaC v1.0数据集，包含5,037例专家裁定的VT报警及对应多导ECG和脉搏波形。

**📈 对比分析**

在官方训练/测试拆分上，与11个已公布系统对比，取得AUROC 0.953、挑战分数83.33，抑制74.8%的误报同时仅误抑1.5%的真报警，与最强基线性能相当。

**⚠️ 局限性**

安全保证基于i.i.d.采样假设，记录聚类自助法仅为敏感性检查；网格选择会影响可证性，需大量标注数据；代码尚未公开，实验结果需自行复现；对可靠性得分的改进未见提升。

---

## 224. Joint Random Access and Localization in Cell-Free User-Centric Networks with Frequency-Selective Fading Channels

**arXiv ID:** 2609.12140 | [PDF](https://arxiv.org/pdf/2609.12140v1)

**作者:** Simon Tarboush `[一作]` (Technische Universität Berlin), Giuseppe Caire `[通讯]` (Technische Universität Berlin)

**通讯引用:** 30591 | [OpenAlex ID](https://openalex.org/A5058252389)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在面向用户的分布式网络（cell‑free）中，为大量分散且间歇性活动的用户设计随机接入（RACH）方案，实现预amble检测与用户定位。

**💡 创新点**

创新点在于①提出统一的空间一致、频率选择性信道模型；②将位置分区随机接入码本与多源AMP联合使用；③在时间域采用Zadoff‑Chu与GLRT检测，在频域采用多源AMP+GLRT；④利用学习到的空间通道统计（radio maps）做先验，改进检测与定位。

**🔧 技术方法**

主要技术包括多源AMP（approximate message passing）、GLRT（generalized likelihood ratio test）、MMLE（maximum likelihood estimation）与超分辨定位、Zadoff‑Chu预amble、离散化网格搜索、频域OFDM、学习型Radio Map、空间一致信道模型等。

**📊 数据集**

使用仿真数据：基于柏林TU模型的36个RU、8天线、7个地理区、随机散射体和用户位置，生成的频率选择性通道。

**📈 对比分析**

通过误检测率/误报率曲线、eer指标以及定位误差CDF与oracle基准对比，结果显示频域AMP方案在误检测/误报方面显著优于时间域ZC方案，定位精度两种方案均逼近oracle误差，误差中位数≈7 m。

**⚠️ 局限性**

限制：①仅考虑单一波束/单用户代码冲突；②信道模型假设为单次反射，忽略高阶多径；③对LOS/NLOS的统计采用离线学习，无法实时更新；④假设固定功率、无干扰源，且仅在仿真中验证。

---

## 225. Challenges and Opportunities in the Transition from 5G to 6G Networks

**arXiv ID:** 2609.12147 | [PDF](https://arxiv.org/pdf/2609.12147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 226. Efficient AI Model Deployment Using Quantization Analysis Tool

**arXiv ID:** 2609.11954 | [PDF](https://arxiv.org/pdf/2609.11954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 227. Access Control as Verified Parse Constraints

**arXiv ID:** 2609.12488 | [PDF](https://arxiv.org/pdf/2609.12488v1)

**作者:** Saranachon Iammongkol `[一作]` (University of Otago), David Eyers `[通讯]` (University of Otago)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过将访问控制决策转化为已验证的数据格式解析，生成单一可验证的C执行器来实现安全网关的访问控制，保证所有可表达的策略均被正确执行并支持运行时规则更新。

**💡 创新点**

将政策定义直接编译为受信任的验证器，避免手写执行引擎，利用前向无回溯EverParse解析器保证决策函数与实现代码的一致性，实现deny‑by‑default结构化保障。

**🔧 技术方法**

利用F*进行形式化证明、EverParse 3D规范生成C代码、Z3 SMT求解器自动化证明、seL4微内核隔离以及可组合的分区验证技术。

**📊 数据集**

以OPNsense 19.1.7的访问控制列表为实测数据集，覆盖420个具体路径（含wildcard展开）进行验证与测试。

**📈 对比分析**

与Cedar、OPA、OpenFGA等通用授权引擎对比，验证器决策延迟仅为几百纳秒（K=8）至1毫秒（512规则），低于常见网络I/O延迟；编译时间为≈19 s/8规则，分区验证可线性扩展。

**⚠️ 局限性**

受限于前向无回溯解析器，无法支持通配符路径或更复杂的RBAC特性；单个规则槽数受编译时内存限制（K≈32），需分区验证以突破。

---

## 228. Score-based Outlier Generation via Controlling the Radon-Nikodym Derivative

**arXiv ID:** 2609.12113 | [PDF](https://arxiv.org/pdf/2609.12113v1)

**作者:** Amartya Mukherjee `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对分布层面控制对数似然分布，利用扩散模型的逆时间动力学生成可控的低似然异常样本。

**💡 创新点**

创新点在于提出分布式的“ρ‑异常”定义，并将似然重加权映射为score的乘法修正，使得不需要重新训练模型即可实现可控生成。

**🔧 技术方法**

技术手段包括score‑based diffusion模型、概率流ODE、Fokker–Planck方程、Ornstein–Uhlenbeck半群、随机支配与Wasserstein距离理论以及指数插值控制器。

**📊 数据集**

实验使用了二维高斯混合模型进行验证，并在CIFAR‑10图像数据集上进行进一步测试。

**📈 对比分析**

评估方法是比较生成样本的对数似然分布与原始分布的W1距离及FOSD关系，结果表明能够按设定的ρ实现预期的低似然位移，并且生成样本保持原数据结构。

**⚠️ 局限性**

局限性在于控制器近似仅在指数衰减假设下有效，且对复杂高维分布的精确性有限；同时对真实数据的似然估计仍需耗时的逆ODE求解。

---

## 229. Benchmarking locally hosted language models for journal editorial work on a compact desktop workstation

**arXiv ID:** 2609.11972 | [PDF](https://arxiv.org/pdf/2609.11972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 230. DCRA: Diffusion-Conditioned Representation Alignment for Robust Time-Series Learning

**arXiv ID:** 2609.11997 | [PDF](https://arxiv.org/pdf/2609.11997v1)

**作者:** Wenrui Xu `[一作]` (University of Minnesota), Keshab K. Parhi `[通讯]` (University of Minnesota)

**通讯引用:** 21212 | [OpenAlex ID](https://openalex.org/A5007884053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出Diffusion-Conditioned Representation Alignment (DCRA) 框架，利用前向扩散过程作为结构化噪声调度器，对时序信号进行多尺度腐蚀，并在特征层实现跨噪声级别的表示对齐；

**💡 创新点**

创新点在于：①将前向扩散过程视为噪声调度器；②引入特征层一致性正则化，强制清晰与噪声样本在潜在空间中保持一致；③框架对编码器无关，可与 Mamba、Transformer 等主流时序编码器结合；

**🔧 技术方法**

采用前向扩散噪声调度、时间步嵌入、双向 Mamba 或 Transformer 编码器、子空间一致性正则化、原型对齐损失，以及混合的分类+一致性损失；

**📊 数据集**

在 CHB‑MIT EEG 癫痫发作检测数据集上进行实验；

**📈 对比分析**

与基线编码器（EEGNet、Mamba、Transformer）、仅使用扩散调度的 DC 模型以及完整 DCRA 进行对比。DCRA 在 AUC、PR‑AUC 上提升不大，但在低 FPR（0.05、0.10）下的灵敏度提升 2–5% 以上，且在噪声水平升高时性能下降更慢，显示更强的鲁棒性；

**⚠️ 局限性**

仅在 EEG 癫痫检测任务上验证，扩散调度和一致性正则化的超参数需要调优，模型训练成本较高，对其他时序任务的泛化能力尚待进一步研究。

---

## 231. An Automated Thickness Evaluation Procedure Using an Integrated Structured Light 3D Camera in a Robotic Bioprinting Framework

**arXiv ID:** 2609.12206 | [PDF](https://arxiv.org/pdf/2609.12206v1)

**作者:** Ehsan Zobeidi `[一作]` (University of Texas at Austin), Farshid Alambeigi `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种全自动、基于视觉的生物打印构件厚度测量方法；

**💡 创新点**

结合Canny边缘检测与几何驱动的厚度计算，实现对复杂形状且存在断层的打印体进行精准分割与厚度评估；

**🔧 技术方法**

使用结构光三维相机获取RGB图像和点云，采用OpenCV实现Canny+轮廓提取，融合原始与粗糙轮廓；基于机器人前向运动学与点云的几何算法计算厚度；

**📊 数据集**

在8个模拟案例（阿基米德螺线、费马螺线的均匀/线性/间断形态）以及4个实际打印案例（直线、费马、阿基米德、含断层）进行验证；

**📈 对比分析**

与真实厚度对比，模拟平均绝对误差在0.025–0.057 mm之间，处理时间≈70–90 ms，实验中厚度均匀性优于手动操作；

**⚠️ 局限性**

受像素离散化影响，最大误差≈0.15 mm；对厚度变化不敏感，但对分辨率下降时精度显著下降；尚未在多层多材料情形下验证。

---

## 232. EvoRS: On-Policy Self-Evolution of Reward Systems for Open-Ended Reinforcement Learning

**arXiv ID:** 2609.12459 | [PDF](https://arxiv.org/pdf/2609.12459v1)

**作者:** Weiyuan Li `[一作]` (Fudan University), Deqing Yang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于可执行奖励图（Reward‑DAG）的自演化奖励系统，利用在策略学习过程中收集的 roll‑out 经验诊断奖励失效并生成候选奖励状态，随后通过匹配回放选择最优奖励状态，从而在开放式语言生成任务中保持奖励可靠性。

**💡 创新点**

创新点在于：①将完整奖励系统抽象为可执行图，使得评价标准、评分机制和信号组合可以统一演化；②通过在策略训练过程中实时诊断奖励失效并以匹配回放验证候选状态，形成闭环自演化；③在保持奖励可解释性和覆盖性的同时显著降低奖励劫持。

**🔧 技术方法**

技术包括：强化学习（GRPO）+ 大语言模型奖励模型（如 Qwen3.5‑27B）、可执行奖励图（Reward‑DAG）、候选状态生成、匹配回放验证、演化记忆与技能。

**📊 数据集**

数据集：写作任务使用 944 条合成指令；角色扮演任务使用 5,000 条 MiniMax Role‑Play Bench 查询；评估基准为 WritingBench 与 CoSER。

**📈 对比分析**

与 RLAIF、RaR、RLER、OpenRS 等固定或动态 rubrics 方法对比，OpenRS 在所有三位评审员下实现最高质量；在写作任务中提升 2.107 分、在角色扮演中提升 4.767 分，并在奖励可靠性指标（奖励劫持、覆盖率、信息量）上优于其他方法。

**⚠️ 局限性**

局限性：仅在写作与角色扮演两个开放式语言生成领域验证；自演化步骤周期固定，未采用自适应调度；额外的诊断与候选评估产生一定计算开销；尚未在工具使用或更复杂交互环境中进行验证。

---

## 233. Reading the Whole Heart: Latent-Attention Masked Autoencoders for Multimodal Cardiac Representation Learning

**arXiv ID:** 2609.12035 | [PDF](https://arxiv.org/pdf/2609.12035v1)

**作者:** Andrea Agostini `[一作]` (ETH Zurich), Thomas M. Sutter `[通讯]` (ETH Zurich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在本文中，我们提出了一种名为 Latent‑Attention Masked Autoencoders (LAMAE) 的多模态自监督预训练框架，能够在同一阶段学习 ECG、超声心动图、胸 X 光以及临床变量的患者级别表征，并在缺失模态和异构数据下保持鲁棒性。

**💡 创新点**

创新点包括：① 在潜在空间引入共享的 Latent‑Attention 模块，直接实现跨模态信息交换；② 采用研究‑视图‑实体层次结构与实体级别遮蔽策略，充分挖掘同一患者内的结构关系；③ 通过一次性自监督重建任务实现对多模态缺失模式的自然处理。

**🔧 技术方法**

技术手段包括 Transformer‑based encoder/decoder、Masked Autoencoder（MAE）预训练目标、跨模态多头自注意力（Latent‑Attention）以及基于实体的遮蔽与重建策略。

**📊 数据集**

实验数据集为 MIMIC‑IV 及其子集 MIMIC‑IV‑CXR、MIMIC‑IV‑ECG、MIMIC‑IV‑Echo、MIMIC‑IV‑ED，总计约 1.27 M 病人住院记录。

**📈 对比分析**

我们与 ProbMED、MedSigLip 以及独立 MAE（无跨模态交互）进行对比；在住院级任务（院内死亡、ICD‑10 章节、DRG 严重度、DRG 死亡风险、住院时长）中，LAMAE 的 AUROC 均高于对照组 1–2 % 甚至更大；在单模态任务中表现保持竞争力，且在 ECG 单模态测试时仍保留部分跨模态预训练收益。

**⚠️ 局限性**

局限性包括：仅在单一 MIMIC‑IV 数据源上验证，跨机构泛化尚未证明；跨模态优势对某些模态（Echo、CXR）不如 ECG 明显；实验规模受计算资源限制；未扩展到实验室时间序列或临床文本等更多模态。

---

## 234. Random Access and Localization in Cell-Free User-Centric Networks with Multipath Channels

**arXiv ID:** 2609.12141 | [PDF](https://arxiv.org/pdf/2609.12141v1)

**作者:** Simon Tarboush `[一作]` (Technische Universit{"a}t Berlin), Giuseppe Caire `[通讯]` (Technische Universit{"a}t Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在无源随机接入的 cell‑free 用户中心网络中，研究了基于位置的随机接入码本，并在多径（LoS 与 NLoS）信道下实现用户检测与定位。

**💡 创新点**

创新点包括：① 采用地理分区码本实现位置感知随机接入；② 结合频率选择性、空间相关且空间一致的信道模型；③ 将 Zadoff–Chu 代码与 GLRT 结合的时域方案与多源 AMP 的频域方案做统一对比，并提出近似最大似然定位方法。

**🔧 技术方法**

使用的技术包括：Zadoff–Chu 序列、广义似然比检测（GLRT）、多源近似消息传递（AMP）、AoA/TDoA 结合的近似最大似然定位、空间一致的多径信道建模。

**📊 数据集**

数据来源为仿真，场景为 36 个 8 天线 RUs 组成的六角网格，2016 条码本，35 个散射体/位置，频率 3.5 GHz，仿真随机生成用户位置、码字与信道。

**📈 对比分析**

对比方法：在固定 FA 为 10⁻¹ 的条件下测量误检（MD）概率与活跃用户数的关系；定位误差通过欧氏距离的累积分布函数（CDF）评估。结果显示，频域 AMP 方案在检测和定位上均优于时域 ZC+GLRT，定位误差 90% 分位数约 12 m 对比 16 m。

**⚠️ 局限性**

局限性：未考虑码字碰撞；AMP 方案计算复杂度高；信道先验仅为零均值 Rician/高斯模型，未实现更精细的碰撞感知或环境自适应。

---

## 235. Assured AI-Native Network Control Loops: State of the Art, Research Challenges and the Missing Runtime Assurance Layer

**arXiv ID:** 2609.11996 | [PDF](https://arxiv.org/pdf/2609.11996v1)

**作者:** Bartosz Belter `[一作]` (Poznan Supercomputing and Networking Center), Mariusz Głąbowski `[通讯]` (Poznan University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了AI‑native网络控制的发展，指出当前自动化技术（如O‑RAN、网络数字孪生、AI驱动的编排）虽能产生智能决策，但缺乏统一机制来保证多控制循环在共享动态网络状态下的安全和可预测性。

**💡 创新点**

创新点在于提出“依赖感知运行时保障（dependency‑aware runtime assurance）”的研究方向，即为每个AI控制循环的决策关联假设与依赖关系，并实时监测其有效性，支持冲突检测、优先级仲裁与安全执行/回滚。

**🔧 技术方法**

采用的技术包括：MAPP‑K闭环框架、O‑RAN xApp/非实时R‑IC接口、网络数字孪生仿真、可信AI（鲁棒性、可解释性）、运行时验证与保障（Simplex、Neural Simplex、Distributed Simplex）、多源假设获取（外部约束、决策上下文、XAI、运行时剖面）以及基于依赖图的冲突分析与安全执法。

**📊 数据集**

本文为综述性工作，并未使用具体数据集；若要实验验证，建议使用5G/6G实验平台或OpenAirInterface与Mininet结合的O‑RAN环境，配合公开网络流量与性能指标。

**📈 对比分析**

由于缺乏实验实现，本文未给出数值性能对比；提出的框架在理论上可通过仿真评估在不同控制循环数量、时延约束、冲突率等维度下的保障覆盖率与时延开销，期望能在10 ms–1 s的近实时控制尺度下保持低额外延迟。

**⚠️ 局限性**

限制包括：①缺乏完整的决策假设自动推导方法，对黑盒模型的假设获取仍依赖外部约束与经验；②运行时保障的计算与时延开销尚未量化；③冲突优先级与仲裁策略需要更系统的规范；④在大规模多域网络中的可扩展性与协同问题待进一步研究。

---

## 236. Consensus-based Decentralized Distributed Swarm Learning with Heterogeneous Big Data

**arXiv ID:** 2609.12143 | [PDF](https://arxiv.org/pdf/2609.12143v1)

**作者:** Zhuoyu Yao `[一作]` (Georgia State University), Zhipeng Cai `[通讯]` (Georgia State University)

**通讯引用:** 18473 | [OpenAlex ID](https://openalex.org/A5072627238)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向异构大规模边缘数据的去中心化分布式群体学习框架 CD‑DSL，兼顾一致性与探索。

**💡 创新点**

创新点在于：①将 push‑sum 一致性机制与粒子群优化相结合，兼容有向网络；②通过本地验证自适应邻居加权，实现性能感知聚合；③将历史经验与邻居信息融合，提升非凸优化的逃逸能力。

**🔧 技术方法**

使用技术包括：push‑sum 一致性、粒子群优化（PSO）更新、熵正则化的邻居加权、异步通信、局部验证评估和分布式梯度下降。

**📊 数据集**

实验数据集为 CIFAR‑10，使用 ResNet‑18，采用 Dirichlet 分布产生不同程度的 IID 与非 IID 训练/验证样本。

**📈 对比分析**

与 Consensus‑DSGD、DDSL、COCO、Consensus‑Control、QG‑GUTm 等基线比较，CD‑DSL 在 IID、轻度和重度异构场景下均收敛更快、最终准确率更高，尤其在严重异构时优势显著。

**⚠️ 局限性**

局限性包括：需要每台设备保留验证集以自适应加权；对网络断连和极端异构的鲁棒性尚待进一步验证；通信开销和超参数调优仍较依赖实验环境。

---

## 237. An Evidence-First Multi-LLM Framework for Auditable Critical-Infrastructure Dependency Modeling

**arXiv ID:** 2609.12360 | [PDF](https://arxiv.org/pdf/2609.12360v1)

**作者:** Nurjahan `[一作]` (Louisiana State University), Aisha Ali-Gombe `[通讯]` (Louisiana State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个以证据为先的多模型LLM框架，自动从异构基础设施文档中提取实体与依赖关系，生成基础设施知识库（IKB）并在人工复核后投影为可验证的基础设施依赖图（IDG）。

**💡 创新点**

创新点在于：① 采用证据优先的合成流程，将原始提取、词汇映射与证据验证分离；② 在实体解析后再进行依赖端点对齐，防止错误边缘生成；③ 保留跨模型支持与不确定性作为元数据而非强制多数投票；④ 维护完整的可追溯性与人机交互界面，确保不把LLM幻觉直接写入最终图。

**🔧 技术方法**

技术包括：三种开源加权LLM（Qwen3-4B‑Instruct、Gemma 3‑12B、Mistral‑7B‑Instruct）本地推理；CC‑CMF证据规范化；语义实体解析、命名实体归一化与属性融合；依赖对齐、图谱验证与冲突检测；结构化存储（SQLite+SQLAlchemy）与可视化复核（Streamlit、NetworkX）。

**📊 数据集**

数据集为9个自然气体、水/废水、电力配电项目，各自只有一份关键文档。共收集702条归一化记录、356个提取单元，人工标注得到参考IKB（284实体、102有向依赖）。

**📈 对比分析**

比较方法：通过 RQ1–RQ4 进行管道诊断与与人工参考对比，使用精度、召回、F1 等微平均指标。结果显示实体识别召回高（≈69%）但精度低（≈7%）；依赖端点召回更低（≈4%）且多模型支持极低（≈2.7%）。在相同证据下，单模型提取的依赖在人工复核前已达到约 40% 的实体恢复率。相比单一 LLM 到图的直接方法，本框架在保留可追溯性与降低错误边缘方面表现更好，但整体 F1 仍较低。

**⚠️ 局限性**

局限性：仅使用三种开源 LLM、单文档、单项目、缺乏多标注者一致性评估；未与直接 LLM‑to‑图基线或单模型对比；未评估完整人机复核后效果；对实体歧义、同义词未充分处理；实验规模和领域泛化性有限。

---

## 238. Occamy-1.0: Open Pareto-frontier 35B Intelligence for Co-work

**arXiv ID:** 2609.11977 | [PDF](https://arxiv.org/pdf/2609.11977v1)

**作者:** Wenhui Chen `[一作]`, Zijian Zou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并训练了 Occamy‑1.0，这是一款专注于协作工作（co‑work）的 35B 参数模型，通过后训练与强化学习提升长周期任务中的执行效率与成本效益。

**💡 创新点**

创新点包括：①将执行驱动的数据与环境、白盒多平台适配、分阶段后训练（Marathon Expert、Sprint Expert、合并、SAO）相结合，形成以执行效率为核心的成本‑性能 Pareto 前沿模型；②提供可追踪的 token‑exact 轨迹捕获与环境重放机制；③在后训练中引入效率奖励与分层奖励，强化状态跟踪与工具调用可靠性。

**🔧 技术方法**

采用的技术有：基于 Qwen3.6‑35B‑A3B 的预训练后继续训练；SFT + HDPO、SFT、SAO 强化学习；模型合并（model soup）与单回合 RL；TiTO（token‑exact 轨迹捕获）与多平台 harness 适配器；历史重写与状态重放；奖励塑造与效率奖励。

**📊 数据集**

使用的数据集约 15K 轨迹、403M token，覆盖四大域：General agentic、Long‑horizon interactive、Terminal & software engineering、Tool‑call grounding；任务构建基于真实工作环境与能力‑环境空间，采用环境优先与能力优先合成路线。

**📈 对比分析**

通过与同规模 35B‑A3B 基线及更大规模模型在 12 项基准（Claw‑Eval、WildClawBench、Business Arena、GDPval、OfficeQA Pro、τ³‑Bench、AutomationBench、BFCL v4、VitaBench、Terminal‑Bench 2.1、IFEval 等）对比，Occamy‑1.0 在同规模榜首，并与大型模型仅相差少；在成本‑效率方面落在 Pareto 前沿低成本拐点；工具调用、编码、指令遵循等辅助指标亦实现提升。

**⚠️ 局限性**

局限性包括：超时鲁棒性不足、浏览器/视觉交互支持不完整、子代理与多目标学习的联合学习有限；在某些复杂文档推理与模拟用户交互任务中性能仍低于更大型模型。

---

## 239. A decision-basis contract for auditable LLM-assisted medical billing verification: deterministic rules, verbatim evidence, and fail-closed abstention

**arXiv ID:** 2609.12156 | [PDF](https://arxiv.org/pdf/2609.12156v1)

**作者:** Jan Hölter `[一作]` (Aschaffenburg University of Applied Sciences), Boris Bauke `[通讯]` (Aschaffenburg University of Applied Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一个可审计的 LLM 辅助医疗账单验证系统，利用决策基础合约将确定性规则检查与语义评估分离，并通过显式的证据门和 fail‑closed 机制保证决策可追溯。

**💡 创新点**

创新点包括：①在合约层面强制 LLM 给出文档支持、矛盾或缺失属性并提供对应的文字证据；②通过证据门只接受文档中确切出现的文字跨度；③在规则层和语义层分别记录基础条目，最终按预设优先级合成案件结果，形成完整的可审计决策链。

**🔧 技术方法**

技术实现上使用本地部署的开源 LLM（Gemma4、MedGemma、Qwen3 4/14 B），通过两层算法（规则层与语义层）与 JSON 接口交互；评估时对模型进行温度为0、8192 令牌窗口的推理；对文档证据进行自动提取与验证。

**📊 数据集**

使用了一个合成的费率目录（包含两期、14 代码、各种限制），以及 36 个人工构造的案例（包含文档、索赔、参考标签），用于功能验证和合同机制测试。

**📈 对比分析**

对比方法为：①系统（合约+LLM）对照①系统‑ablation（去除文档要求）和②端到端 baseline（一次性给出全部信息）；评估指标为案件结果一致率、误判率、错误接受率。结果显示在某些模型上系统略优，但无一致性优势；显式文档要求显著提高了缺失信息的检出率，证据门进一步改进了决策可追溯性。

**⚠️ 局限性**

局限性包括：①数据集为合成且未经过独立注释；②评估仅在单次温度0运行，未考察多次实验或不同温度；③未在真实账单或人工审核场景中验证实际价值；④证据门可能导致合法判断被错误归为缺失；⑤规则库需从真实目录构建，现阶段仅为示例。

---

## 240. PDoS: A Profitable Denial-of-Service Attack against Proof-of-Work Blockchain Liveness

**arXiv ID:** 2609.12450 | [PDF](https://arxiv.org/pdf/2609.12450v1)

**作者:** Junjie Hu `[一作]` (Shanghai Jiao Tong University), Na Ruan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种新的PoW链活性破坏攻击——PDoS，该攻击通过发布仅包含头部的区块并结合矿池收益剥削，既能迫使理性矿工停挖，又能利用被攻击矿池的奖励机制补贴攻击成本，实现自给自足的DoS。

**💡 创新点**

创新点在于将传统的阻止攻击与收益剥削相结合，形成双重威胁；同时揭示在高MEV或高交易费环境下，PoW链的安全性反而会下降，攻击者可以从中获得财政补贴甚至盈利。

**🔧 技术方法**

作者采用了连续时间马尔可夫链（CTMC）模型、马尔可夫奖励过程（MRP）、收益动态模型以及博弈理论分析，并利用区块链数据回放进行实验验证。

**📊 数据集**

实验使用了 2024‑2025 年 Bitcoin 区块链数据、交易费统计、MEV（鲸鱼奖励）信息以及矿池哈希分布等真实数据集。

**📈 对比分析**

与传统的 BDoS 进行对比，PDoS 在破坏阈值、成本收益以及持续攻击时间上均表现更优；在高盈利区块环境下，PDoS 能实现成本回收甚至正收益。

**⚠️ 局限性**

局限性包括对矿工切换成本假设较低，实际矿工可能因切换成本而不立即停挖；防御措施如延迟或条件性矿池支付可削弱攻击收益；此外，论文未讨论网络层攻击或硬件限制等更复杂场景。

---

## 241. Bridging Vision Foundation Model Priors with CLIP for Spatial-aware Few-shot Anomaly Detection in Medical Images

**arXiv ID:** 2609.12454 | [PDF](https://arxiv.org/pdf/2609.12454v1)

**作者:** Juzheng Miao `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出 Spatial-FAD 框架，将 Vision Foundation Model（如 DINO）的空间先验与 CLIP 的语义对齐结合，改进医学影像少样本异常检测的空间定位。

**💡 创新点**

创新点在于三方面：① 用 DINO 的结构亲和力先验注入 CLIP 特征；② 采用滑动窗口聚合提升高分辨率特征；③ 引入原型增强的支持记忆，在推理时充分利用少量样本。

**🔧 技术方法**

技术手段包括：CLIP 视觉编码器与可学习适配器、DINO 的自监督局部到全局特征、滑动窗口特征拼接、k-means 原型聚类与相似度融合。

**📊 数据集**

实验数据集涵盖三种医学影像：LiverCT（肝癌 CT）、RESC（视网膜 OCT）、BrainMRI（脑肿瘤 MRI）。

**📈 对比分析**

与 BGAD、MediCLIP、MVFA、MadCLIP 等最先进方法对比，Spatial-FAD 在 Dice 及 AUC 指标上均显著提升，4-shot 情况下平均 Dice 提升 11.4% 以上，部分数据集 23%+ 的改进。

**⚠️ 局限性**

局限性：对原型数量敏感（尤其异常原型）；依赖 CLIP 预训练且对不同模态的迁移效果仍需进一步验证；滑动窗口方法在极大图像尺寸下计算成本较高。

---

## 242. Explanations-Driven Active Feature Acquisition for Algorithmic Recourse

**arXiv ID:** 2609.12179 | [PDF](https://arxiv.org/pdf/2609.12179v1)

**作者:** Vinura Galwaduge `[一作]` (Western University), Jagath Samarabandu `[通讯]` (Western University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种将算法回归与主动特征获取相结合的框架，即解释驱动主动特征获取（EDFA），并通过Markov Blanket理论统一了三类回归解释（Counterfactual、Semi-factual、Alterfactual）

**💡 创新点**

创新点在于：①将解释性信息作为特征获取的决策依据，优先获取能产生可解释性变更的特征；②使用Markov Blanket理论构造“半事实→可解释性”转换机制；③为基于部分信息的回归提供分布无关的有效性保证，并给出所需校准数据量的下界；④展示EDFA在减少特征获取成本的同时仍保持可解释且可操作的回归

**🔧 技术方法**

技术上采用Markov Blanket发现算法（EAMB、HITON-MB、BAMB）作为特征子集；基于PC-first/CMI策略的贪心特征获取；回归模型使用FTT、XGBoost和贝叶斯网络；CF搜索方法包含DiCE、NICE、PROBE与EDFA默认/生成式方法；利用Risk‑Controlling Prediction Sets与Hoeffding‑Bentkus UCB实现早期回归的有效性保证

**📊 数据集**

实验使用7个公开表格数据集，涵盖金融与医疗领域：Adult Income、German Credit、HELOC、Taiwan Credit、ACS Income、GMSC、Diabetes130；对每个数据集采用不同特征成本分配与敏感特征设定；

**📈 对比分析**

与三种SOTA主动特征获取方法（DIME、EDDI、GSMRL）以及随机MB选择进行对比。EDFA在大多数数据集上以更低的平均特征获取成本获得相当甚至更高的预测准确率；生成的CFs在L0稀疏度、L2接近度和合理性（LOF）上表现更好；在验证保证实验中，EDFA结合PROBE/DiCE等CF搜索能在较低成本下满足α=0.2/0.3的有效性约束，且所需校准样本量符合理论下界

**⚠️ 局限性**

局限性包括：①对MB结构的依赖，若MB单元未能完整恢复则可能缺乏可翻转特征导致无法产生回归；②假设MB内特征均可操作，实际中某些Spouse特征可能不可执行；③有效性保证仅提供总体失效率而非个体风险；④在GMSC等特征量较小的MB中表现不佳；⑤目前未考虑因果方向性与可执行性约束，未来可结合因果模型进一步提升可操作性

---

## 243. The Information Complexity of Decision Trees

**arXiv ID:** 2609.12164 | [PDF](https://arxiv.org/pdf/2609.12164v1)

**作者:** Avantika Agarwal `[一作]`, Eric Blais `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

本文定义并研究了随机决策树的“信息复杂度”，并证明其与树大小复杂度的紧密关系，进一步推出压缩、直接乘积等重要性质。

**💡 创新点**

创新点在于将信息复杂度与随机决策树的树大小复杂度等价化，提供了信息等于渐进大小复杂度的定理；证明信息复杂度可用于压缩单个实例的树大小；以及在成功条件下给出完美的直接乘积定理。

**🔧 技术方法**

核心技术包括：信息理论中的互信息和熵、叶子熵（惊讶度）视角、典型序列/条件典型性、通用的成本与分数量化、折扣分数的张量化与等价引理、以及对树大小与信息复杂度之间的算术关系。

**📊 数据集**

无（论文为理论分析）。

**📈 对比分析**

对比方法：通过证明信息复杂度与树大小、查询深度的对数关系，作者展示了信息复杂度可以用来下界随机查询复杂度，压缩树大小至信息复杂度上界，且直接乘积下界为Ω(n·信息复杂度)。这些结果在理论上优于传统的直接下界方法。

**⚠️ 局限性**

局限性：信息复杂度不直接刻画单个实例的查询深度（如 OR 函数），而需通过额外的函数变换或与树大小的关联；另外，成功条件下的直接乘积定理依赖于信息复杂度的定义，可能不适用于所有查询模型。

---

## 244. Hierarchical Belief Modeling for Zero-Shot Opponent Adaptation in Partially Observable Multi-Agent Navigation

**arXiv ID:** 2609.12422 | [PDF](https://arxiv.org/pdf/2609.12422v1)

**作者:** Kowei Shih `[一作]` (Tsinghua University), Kejian Tong `[通讯]` (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并训练了名为 HORIZON 的分层强化学习代理，用于 Lux AI Season 3 赛季中的部分可观测多智能体导航任务，能够在单个 episode 内完成 meta‑adaptation，提升比赛获胜率。

**💡 创新点**

创新点包括：① 双层记忆编码器将短期战术记忆与跨匹配的元学习记忆分离；② 基于遗迹中心的图注意力模块，显式推理隐藏的遗迹计分格子；③ 对称等变的空间编码网络；④ 以信息增益为驱动的 intrinsic reward；⑤ 对手类型条件化的策略混合器。

**🔧 技术方法**

采用的技术包括：D_2 等变卷积、FiLM 条件化、双向 GRU 记忆、图注意力网络、贝叶斯遗迹信念更新、基于信息增益的 intrinsic reward、PPO + GAE、辅助世界模型与 θ 回归损失，以及 JAX/Flax 高并行训练框架。

**📊 数据集**

使用的数据集是 Lux AI Season 3 的离线模拟环境，生成随机化的地图与动态参数，并在 500 个 best‑of‑five episode 进行评估；训练阶段采用自对弈 league 与冻结的过去检查点。

**📈 对比分析**

与官方规则机器人、FlatPPO、IMPALA‑LSTM 和 RL²‑Recurrent 进行对比，评估指标包括 Match Win Rate、Episode Win Rate、Meta‑Adaptation Gain 与 TrueSkill；HORIZON 在所有指标上均领先，最大 Meta‑Adaptation Gain 达 0.16，TrueSkill 29.1。

**⚠️ 局限性**

局限性包括：1) 仅在 Lux AI 这一特定环境中验证，缺乏跨任务泛化；2) 对对手策略的泛化仍受限于预设的四类子策略；3) 计算资源要求高，尤其是双层记忆与图注意力的并行实现；4) 对完全未知的环境动态或对手行为可能需要进一步改进。

---

## 245. Learning to adapt GR(1) specifications through degradation

**arXiv ID:** 2609.12231 | [PDF](https://arxiv.org/pdf/2609.12231v1)

**作者:** Tiberiu-Andrei Georgescu `[一作]` (Imperial College London), Sebastian Uchitel `[通讯]` (Imperial College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在运行时出现环境假设违背时，利用学习与公式弱化技术自动调整 GR(1) 规范，以恢复可实现性并尽量保留原先的系统保证。

**💡 创新点**

提出了适配空间的形式化定义、首选准则以及基于 OGIS 的公式弱化搜索框架，能够在多种可实现的修正版中挑选最优方案。

**🔧 技术方法**

结合 GR(1) 合成、LTL 语义弱化、Oracle‑Guided Inductive Synthesis、最小不可满足子集与最小覆盖集求解、宽度优先搜索等技术实现适配与弱化。

**📊 数据集**

在 Buckworth 等提供的 5 个基准案例（Arbiter、Lift、Minepump、Traffic Single、Traffic U）上进行评估。

**📈 对比分析**

通过定义的优先关系与基线（单纯删减公式）比较，实验表明学习方法在所有案例均产生更优、更保留保证的规范，且通常仅得到单一首选解。

**⚠️ 局限性**

仍受限于 GR(1) 语法约束、假设违背的可观测性不足、搜索空间可能爆炸以及对语义优先级的外部依赖；对不可比的保证集缺乏完整的排序机制。

---

## 246. Split Conformal Prediction with Label-Shift-Adjusted Bayesian Scores

**arXiv ID:** 2609.12386 | [PDF](https://arxiv.org/pdf/2609.12386v1)

**作者:** Hyeonsu Lee `[一作]` (MOGAM Institute for Biomedical Research), Hyunjin Shin `[通讯]` (MOGAM Institute for Biomedical Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了在标签平移下的贝叶斯非合规性评分（Label‑Shift‑Adjusted Bayesian Score, LSA），实现分布无关的不确定性量化并在分子属性预测中验证其有效性。

**💡 创新点**

通过后验预测倾斜恒等式将标签平移对贝叶斯预测的影响显式化，得到对贝叶斯分数的直接校正，兼顾目标分布对齐与预测不确定性自适应。

**🔧 技术方法**

采用贝叶斯岭回归作为基础预测模型，利用对数线性密度比估计（逻辑回归伪标签）实现倾斜校正，并结合加权分位数的 conformal prediction。

**📊 数据集**

在AqSolDB（溶解度）和Lipophilicity（脂溶性）两大分子性质基准上进行实验。

**📈 对比分析**

与残差非合规性评分和未校正的贝叶斯评分在相同模型与权重估计下比较，LSA 在所有标签平移幅度下均显著缩短预测区间长度（最多约13%），同时保持相近的覆盖率。

**⚠️ 局限性**

仅在基于伪标签的密度比估计时具有有限样本有效性，强标签平移会导致伪标签误差增大，进而略微降低覆盖率；此外目前仅针对高斯预测模型，需要推广到更复杂模型。

---

## 247. Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models

**arXiv ID:** 2609.12303 | [PDF](https://arxiv.org/pdf/2609.12303v1)

**作者:** Kalyani Marathe `[一作]` (University of Washington), Srinivasan Iyer `[通讯]` (Meta FAIR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过大规模实验比较了在相同模型规模下，使用字节（byte）和词元（token）两种分词方案进行蒸馏训练的性能和扩展性。

**💡 创新点**

创新点在于提出了两种高效单前向传递的 token‑to‑byte logit 转换方法（Marginalize‑It 与 End‑of‑Token），并系统研究了字节模型在蒸馏与过度训练下的 Scaling Law，证明字节模型在更高计算量下能够超过 token 模型并实现更高的数据和存储效率。

**🔧 技术方法**

主要技术包括 transformer 的 over‑training、蒸馏训练（distillation loss）与监督训练（cross‑entropy loss）的对比、两种 logit 转换算法、以及基于 BPB（bits‑per‑byte）和下游任务误差的多任务 Scaling Law 拟合。

**📊 数据集**

实验使用了 Llama‑2 训练混合数据（约 1 万亿字节），并在八个基准（ARC‑Easy/Challenge、HellaSwag、PIQA、MBPP、Natural Questions、Flores 英德/德英）上评估。

**📈 对比分析**

与同规模 token 蒸馏模型相比，字节蒸馏模型在低 FLOP 区间略逊，但随着计算投入逐步提升最终达到更高的平均下游任务准确率；在 1 B 参数规模下，字节模型的最优表现比 Llama 3.2‑1B、Gemma‑3‑1B‑pt、Gemma‑2B 等开源模型高出最多 8% 以上。

**⚠️ 局限性**

主要局限包括：模型仅在密集 transformer 结构下测试，未考虑稀疏/专家模型；推理成本仍较高；实验仅覆盖 1 B 参数规模和特定数据分布；未对真实世界部署场景的鲁棒性和能耗进行深入评估。

---

## 248. PhysioAI: Clinical Knowledge-Guided Semantic Supervision for Skeleton-Based Physiotherapy Action Recognition

**arXiv ID:** 2609.12491 | [PDF](https://arxiv.org/pdf/2609.12491v1)

**作者:** Jie Cao `[一作]` (University of Sydney), Jinman Kim `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出一种名为PhysioAI的框架，用临床知识指导的语义监督来提升骨骼动作识别在康复场景下的性能，特别是对受限数据、细粒度动作类别和执行变异性具有挑战性的任务。

**💡 创新点**

创新点在于：①构建结构化临床知识字典（CKD），生成针对每个康复动作的可观测语义描述；②将冻结的CLIP文本嵌入映射到骨骼特征空间，形成固定的语义锚；③在训练时使用置信不确定性加权的对齐损失，延迟开启以平衡分类与语义监督；④保持推理时仅使用骨骼输入，避免额外文本处理。

**🔧 技术方法**

使用的技术包括：CTR-GCN骨骼编码器、CLIP文本编码器、固定高斯投影、余弦对齐损失、不可变权重的自适应不确定性权重以及基于GPT‑4o生成的结构化语义描述。

**📊 数据集**

使用的公开康复数据集为KiMoRe（78名受试者）和UI‑PRMD（10名受试者），并通过ST‑GCN构造的Hard‑67子集作为压力测试。

**📈 对比分析**

与多种基准方法（ST‑GCN、AAGCN、MS‑G3D、PA‑ResGCN、CTR‑GCN、BlockGCN、SkateFormer、ProtoGCN、GAP）在五折受试者分离评估下进行比较。PhysioAI在KiMoRe整体、Hard‑67和UI‑PRMD整体上均取得最高或第二高的准确率（最高99.03%），相较于最佳对比方法提升0.27、2.87和1.33个百分点，显示显著性能提升。

**⚠️ 局限性**

局限性包括：只验证了两个相对较小的数据集，缺乏对不同患者群体和功能水平的全面覆盖；Hard‑67作为模型驱动的难例集未得到临床专家验证；仅使用CTR‑GCN骨骼编码器，未评估在其他骨骼网络上的迁移效果；当前方法仅识别动作类别，未评估动作质量或治疗效果。

---

## 249. Who Pays for Open Review? Visible Author Reputation and Its Effect on Ratings

**arXiv ID:** 2609.11983 | [PDF](https://arxiv.org/pdf/2609.11983v1)

**作者:** Qinghua Zhao `[一作]` (Hefei University), Zhongfeng Kang `[通讯]` (Lanzhou University)

**通讯引用:** 445 | [OpenAlex ID](https://openalex.org/A5089478381)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究作者声誉在开放与盲审下对ICLR 2026评审评分的影响，通过对18000+提交的统计分析和AI模拟验证。

**💡 创新点**

创新点在于将arXiv预印时间作为自然实验，将开放/盲审差异与作者声誉结合，首次在大规模数据和AI实验上并行证实声誉偏差集中在决定性评分阈值。

**🔧 技术方法**

使用统计回归（交互项）与分段线性分析、OpenAlex和OpenReview API爬取、以及Claude、GPT等LLM模拟评审技术。

**📊 数据集**

使用ICLR 2026全部提交的OpenReview记录、arXiv元数据、OpenAlex作者记录以及对应的ORCID作为数据集。

**📈 对比分析**

通过开放与盲审分组、声誉斜率差异检验以及AI模拟对同一稿件在不同声誉下评分差异进行对比，AI模拟显示0.2–0.6分的评分提升，与观测结果一致，说明偏差显著。

**⚠️ 局限性**

局限性包括预印窗口与声誉不完全随机，开放组声誉普遍更高导致结果为上限；bibliometric指标时间滞后；LLM可能已见过稿件，但对比内在不受影响。

---

## 250. Pneumatic neurons for soft robots enable inflate-and-fire networks for rhythmic motion

**arXiv ID:** 2609.12258 | [PDF](https://arxiv.org/pdf/2609.12258v1)

**作者:** Dongting Li `[一作]` (University of California San Diego), Nick Gravish `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并实现了软气动神经元（Pneu-ron）模块，该模块将低沸点流体的膨胀、热驱动的电气开关以及逻辑信号集成在同一柔性单元中，构建了可产生自激振荡的环形网络，并通过实验验证了振荡、网络重构、机械/热扰动下的鲁棒性以及基于此振荡驱动的类蠕虫行走。

**💡 创新点**

创新点在于：①将能量转换、逻辑控制与执行单元整合为单一软模块，消除外部泵和复杂电路；②采用“inflate‑and‑fire”原理，以低沸点流体的相变与机械阀门实现双阈值激活与自锁；③通过理论模型和二维分岔图阐释网络拓扑与阈值/时间常数对振荡行为的决定性作用；④展示了网络在破坏、负载、温度变化下的自适应与可重构性；⑤实现了无电子控制的全软连续行走。

**🔧 技术方法**

技术包括：柔性热驱动气囊设计、双阈值可折叠金属/塑料电气开关、低沸点流体（Opteon SF33）膨胀驱动、RC 模型简化的网络仿真、分岔图分析、热成像与压力传感测量、负载测力计与光学运动捕捉。

**📊 数据集**

主要使用实验数据集：单模组的压力-温度-电流曲线、阈值 hysteresis、机械输出力/位移与电压的对应表；多模组环形网络的开关状态、热成像序列、振荡周期、机体位移与时间；不同扰动（机械负载、热变换、电压下降、热板超沸点）下的实验轨迹。无公开大规模数据集，所有数据为实验室自采。

**📈 对比分析**

比较方法：将实验测得的振荡周期、阈值、负载响应与基于 RC 模型的分岔图预测进行对比；在重构实验中比较切割前后振荡周期与振荡状态的保持率；在行走实验中用相对体长/周期的位移来评价运动效率，并与生物类地面蠕虫运动速率进行对照。实验结果表明：振荡周期与理论匹配良好，振荡可持续至多 N=6；行走速度为 0.0231 BL/周期，属于生物类蠕虫的 0.02–0.06 BL/周期范围。

**⚠️ 局限性**

局限性：①因低沸点流体的热传导慢，导致振荡周期长（≈15 min），速度受限；②TPU 热敏感，限制最大加热温度，进一步影响速度和可用流体种类；③阈值需精细调节，对制造误差与温度漂移敏感；④在高温或低温环境下会出现灭振或过度膨胀；⑤缺乏实时传感与自适应控制，无法实现更复杂的运动模式；⑥对流体泄漏和长期稳定性尚未系统评估。

---

## 251. Do Influence-Derived Data Perturbations Enable Machine Unlearning? A Controlled Study of Three Plausible Roles

**arXiv ID:** 2609.12313 | [PDF](https://arxiv.org/pdf/2609.12313v1)

**作者:** Chenkai Wu `[一作]` (Monash University), Jun Yan `[通讯]` (Shanghai Ocean University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了Deep Perturbation Learning（DPL）在机器无学习中的三种角色——直接删除、保留性能正则化和对抗无学习的热启动，并通过匹配基准和实现审计验证其有效性。

**💡 创新点**

创新点在于提出了角色分离的评估框架、对实现细节的深入审计清单，并首次对DPL在三种角色下的实际表现给出严格门槛判定。

**🔧 技术方法**

使用了影响函数、Hessian-向量乘积、混合导数等技术，以及梯度下降和对抗攻击的组合来生成与参数位移相匹配的输入扰动。

**📊 数据集**

主要实验数据集为CIFAR-10（ResNet‑18）和 Tiny ImageNet（VGG‑19），并对不同随机种子下的删除样本进行了评估。

**📈 对比分析**

与精确重训练、SalUn、AMUN、常规交叉熵及官方/FGSM初始化等匹配基准相比，DPL 在直接删除和热启动角色下表现不佳（远低于重训练和对抗基准），仅在少数种子下的保留性能正则化略有提升，但总体不稳定。

**⚠️ 局限性**

实验局限包括仅评估随机实例删除、种子数目有限、超参数搜索不充分、缺乏认证保证以及仅聚焦图像分类任务，因而对其它删除场景或模型类型的泛化性未得到验证。

---

## 252. AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization

**arXiv ID:** 2609.12471 | [PDF](https://arxiv.org/pdf/2609.12471v1)

**作者:** Ji Liu `[一作]` (Advanced Micro Devices, Inc.), Emad Barsoum `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出AMDKernelVault，一个面向AMD CDNA GPU的 HIP 与 Triton 内核语料库，并开发了 HIPKernelGen 与 TritonKernelGen 两套 agent 驱动的 generate–evaluate–reflect 流程，用于自动生成、编译、验证与性能剖析。

**💡 创新点**

创新点在于提供规模化、执行验证的 AMD 原生内核数据集、统一的生成-评估-反思管道，并通过该语料库训练 Qwen3-8B 以在本地完成生成、反思和优化，从而减少对前沿 LLM 的依赖。

**🔧 技术方法**

采用了 LLM（GPT‑5、Qwen3‑8B 等）生成器、基于 ROCm 的编译与执行检查、Triton 3.3.0 运行时、以及 GEAK‑style 的多轮 agent 回路与 GRPO 强化学习。

**📊 数据集**

使用了 64,530 条 HIP/ROCm 核心、39,893 条 Triton 核心的验证数据，以及 2,377 条基于 AMD ROCm 库的 QA 监督，全部来自 CUDA‑Agent‑Ops‑6K、GPUMODE‑KernelBook、AI‑CUDA‑Engineer‑Archive、Stack‑v2‑dedup‑Triton、TritonBench‑8k 等六大来源。

**📈 对比分析**

在固定 agent 预算下，将训练后的 Qwen3‑8B 与 GPT‑5、Gemini 2.5 Pro、Claude Sonnet 4 等前沿模型在 PyTorch‑to‑HIP、HIP‑to‑HIP、TritonBench‑G 与 ROCmBench 上进行对比，Qwen3‑8B 在 correctness 指标（如 PyTorch‑to‑HIP Pass@1 34.0%、TritonBench‑G Corr@3 33.2%、ROCmBench Corr@3 41.94%）上优于对手，但在编译通过率或 speedup 上不一定领先。

**⚠️ 局限性**

局限包括仅覆盖 AMD GPU 环境、难以处理高难度专家级内核、缺乏完整的语义重复审计、对特定 ROCm/Triton 版本与硬件依赖强、以及未提供细粒度的操作族通用通过率分析。

---

## 253. Tact: A Zero-Cost, Browser-Based Pipeline for On-Demand Tactile Braille Storybooks

**arXiv ID:** 2609.12272 | [PDF](https://arxiv.org/pdf/2609.12272v1)

**作者:** Iliano Fasolino `[一作]` `[通讯]` (Independent Researcher), Iliano Fasolino (Independent Researcher)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了一个零成本、浏览器端的完整流程，将口述或输入的故事想法自动生成可打印的盲文页面及匹配的触觉图形；

**💡 创新点**

将语言模型、盲文翻译、触觉图形匹配和FDM打印整合在无服务器、零成本的浏览器应用中；使用确定性盲文翻译、分层模型加载、手绘触觉形状库以及完整的硬件/尺寸校准；

**🔧 技术方法**

WebGPU + WebLLM、WebAssembly、Web Speech API、Web Audio API、JavaScript/TypeScript、STL导出、FDM 3D打印、可自适应模型分层；

**📊 数据集**

统一盲文表（UEB、Liblouis）、93个手绘触觉形状（基于童话频率研究）、OpenAI/Groq/Agnes 语言模型；

**📈 对比分析**

通过工程验证：盲文反解一致、STL文件大小符合公式、分页精确、关键词匹配无误、无控制台错误；性能方面托管Groq模型生成0.6–0.9 s，Agnes 6–9 s，本地浏览器模型下载约1.8 GB；无模型输入即可 <1 s；无GPU即可运行；

**⚠️ 局限性**

尚未在真实盲童/低视用户中进行测试；缺少命令行/批量生成工具、双面/间点打印功能；未提供手动模型选择界面；需要与盲童及相关机构进一步沟通验证。

---

## 254. Aligned Radiometric RGB-Thermal Fusion for UAV Facade Anomaly Screening

**arXiv ID:** 2609.12521 | [PDF](https://arxiv.org/pdf/2609.12521v1)

**作者:** Yuan Yang `[一作]` (Hong Kong Center for Construction Robotics), Haobo Liang `[通讯]` (Hong Kong Center for Construction Robotics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对无人机建筑立面检测，构建了从传感器校正到热辐射对齐，再到局部对比度编码的完整管道，并将其作为单流四通道检测器的输入。

**💡 创新点**

创新点在于：① 仅使用传感器级别的单次对齐与公共支持裁剪，避免后置网络中的跨模态失配；② 用 16‑bit 原始辐射数据生成符号局部对比度通道 C₂，保留局部热差正负信息；③ 将上述对齐与编码直接嵌入简洁的单流检测框架，保持极低的模型复杂度。

**🔧 技术方法**

技术包括：双传感器标定与畸变校正、场景级单一单应变换对齐、局部高斯平滑与对比度归一化、单流 Anchor‑free 目标检测器（≈11M 参数，28.5 GFLOPs）。

**📊 数据集**

使用自制的 M3T 数据集（674 对齐的 RGB‑热对，含 8 类立面与缺陷标注，来源于 5 个项目），并在公开的 RGBT‑Tiny 上做跨数据集验证。

**📈 对比分析**

方法对比：与 EME、ICAFusion、CMA‑Det 等基线在同一对齐数据上做项目分层四折评估。ICAFusion 获得最高 mAP_50:95≈0.191；本方法在仅 11M 参数、28.5 GFLOPs 的情况下达到 mAP_50≈0.294、mAP_50:95≈0.168，显著低于 ICAFusion，但在参数与算力上优于对手；在 RGBT‑Tiny 上性能较低，表明该编码主要适用于 16‑bit 辐射场景。

**⚠️ 局限性**

局限性包括：① 对齐过程需人工交互，缺乏全自动化；② 对热信号扩散（如水渗）敏感，局部对比度编码对大范围热分布无效；③ 数据集规模与类别分布有限，难以充分验证跨季节、不同建筑类型的泛化能力。

---

## 255. Robust Prototypical Networks for Few-Shot Sensor Fault Diagnosis

**arXiv ID:** 2609.12287 | [PDF](https://arxiv.org/pdf/2609.12287v1)

**作者:** Mohammed Ayalew Belay `[一作]` (Simula UiB), Pierluigi Salvo Rossi `[通讯]` (NTNU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出多轮聚合原型网络（MEPN），在少样本传感器故障诊断中通过聚合多个独立支持集的原型来降低原型方差，提升一/多样本分类精度。

**💡 创新点**

创新点在于仅通过在推理阶段收集多轮（N_agg）离散支持样本并对其原型取均值，实现在不改变编码器或损失函数的情况下显著减少原型估计噪声；与Kalman或动态模型的原型稳定方法相比，MEPN更简单、无模型开销。

**🔧 技术方法**

使用基于距离的原型网络（ProtoNet）框架，配合一维卷积编码器（四层卷积、64维嵌入），采用多轮支持采样、均值聚合和最近原型判决；在训练阶段保持单轮支持，聚合仅在推理时进行。

**📊 数据集**

在DeFACTO工业 CO₂ 传感器数据集上进行实验；通过在真实温度序列上注入四种低强度合成故障（偏置、漂移、尖峰、噪声）来构造5分类任务，保证类中心紧密。

**📈 对比分析**

与ProtoNet、Matching Networks、Relation Networks 在相同编码器与元训练协议下对比；MEPN 在一-shot 任务上准确率从 73.5% 提升至 93.1%（+19.6个百分点），在 K=5 时亦保持 93.8%；在等总支持样本预算（K=10）下与 ProtoNet 性能相当，验证了聚合机制而非更强学习器。

**⚠️ 局限性**

局限性包括：增益在 N_agg 超过约 10 时趋于饱和，受剩余相关性限制；需在推理时采集多轮支持样本，增加采样复杂度；仅在单一工业传感器数据集和 MNIST 进行验证，尚未证明对其他多模态或高维时序数据的普适性。

---

## 256. Harness or Model? Isolating the Harness Effect in Agentic Coding with a Contamination-Controlled Private Suite

**arXiv ID:** 2609.11987 | [PDF](https://arxiv.org/pdf/2609.11987v1)

**作者:** Mohsen Arjmandi `[一作]` `[通讯]` (evolutionID GmbH), Mohsen Arjmandi (evolutionID GmbH)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在私有的 256 任务套件上进行配对实验，比较了同一模型（Claude Opus 4.8 与 GPT‑5.5）下供应商原生 harness 与中立 harness（deepagents）的能力、成本与完成行为。

**💡 创新点**

创新之处在于使用了私有无污染任务库、同模型配对对照，并对发现的 telemetry 计费缺陷进行纠正，首次系统评估 harness 对同一模型性能与成本的影响。

**🔧 技术方法**

研究采用微 VM 沙箱多 harness 运行平台、事件记录与 ledger、任务 Bootstrap 统计、以及 raw token 重计费校正等技术。

**📊 数据集**

所用数据集为私有的 256 任务池，其中包含 179 个来自内部代码库的仓库任务和 77 个后期公开竞赛任务，全部配备隐式测试与金手指 patch。

**📈 对比分析**

通过配对同模型对照、任务 Bootstrap CI、McNemar 等检验方法，结果显示两种 harness 在 solve 率上无显著差异，neutral harness 成本约 1.2–1.6 倍且更易触及时间上限。

**⚠️ 局限性**

局限性包括数据集仅来源于私有代码库导致的选择偏倚、重复次数有限、部分 token 消耗未被记录导致成本不确定，以及实验仅在单一时间点与特定环境下完成。

---

## 257. DriftSE: Speech Enhancement with Generative Drifting

**arXiv ID:** 2609.12252 | [PDF](https://arxiv.org/pdf/2609.12252v1)

**作者:** Liang Xu `[一作]` (Victoria University of Wellington), Rasmus Kongsgaard Olsson `[通讯]` (GN Advanced Science)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 DriftSE，一种基于漂移模型的一步语音增强框架，能够在不进行迭代采样的情况下完成语音去噪和去混响。

**💡 创新点**

创新点包括：双潜在（语义+声学）漂移以同时保持语言信息和声学真实性；漂移训练只依赖帧级潜在分布，可实现无配对训练；在推理时完全抛弃漂移场，确保严格 1 NFE 的低延迟。

**🔧 技术方法**

采用了漂移模型、Wasserstein 梯度流、均值漂移场、预训练的 HuBERT/DistilHuBERT、PANNs/BEATs 等语义/声学编码器，以及多种 U‑Net/TF‑GridNet、SFMUnet 等生成网络。

**📊 数据集**

使用的语音数据集包括 EARS‑WHAM、EARS‑REVERB、VoiceBank‑DEMAND、WSJ0‑REVERB 以及 DNS2020 等，支持离线与实时两种场景。

**📈 对比分析**

通过与 SGMSE+、ROSE‑CD、FM‑Euler、DM‑IERM 等现有基线在多种指标（WER、PESQ、SI‑SDR、ESTOI、非侵入式 MOS 预测等）进行对比，DriftSE 在 WER 上实现 SOTA，且在 1 NFE 的前提下在离线和实时架构上均优于迭代生成方法。

**⚠️ 局限性**

局限性包括：在完全无配对训练下仍会出现语言失真（WER 上升显著）；对极端噪声或极端房间条件的泛化尚需进一步验证；双潜在漂移的参数选择对性能影响较大，需要更多实验来稳定化。

---

## 258. Explainable Prediction from Mobile Sensing Data through LLM-guided Concept Integration

**arXiv ID:** 2609.11995 | [PDF](https://arxiv.org/pdf/2609.11995v1)

**作者:** Yuning Wang `[一作]` (University of Turku), Pasi Liljeberg `[通讯]` (University of Turku)

**通讯引用:** 11018 | [OpenAlex ID](https://openalex.org/A5019546150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种概念集成Transformer框架，利用LLM生成概念异常监督，实现小样本移动健康预测与解释。

**💡 创新点**

创新点：无人工概念标注的LLM引导概念异常监督，联合概念预测头提升小样本预测并提供概念层解释。

**🔧 技术方法**

技术：Transformer编码器、基于LLM的概念异常生成、置信度加权概念损失、滑动窗口统计、置信度一致性计算。

**📊 数据集**

数据集：AFFECT（每日负面情绪预测）和PHQ-9（周抑郁症状分类）。

**📈 对比分析**

比较：与无概念Transformer、无概念损失Transformer、OCSVM概念版本对比，AFFECT上F1最高0.756，PHQ-9上F1 0.765，CIT总体优于基线。

**⚠️ 局限性**

局限：依赖预设的5个概念分类，无法自动生成概念，需LLM推理对标记一致性；对更大样本或不同人群的泛化尚待验证。

---

## 259. Scalable Discrete-to-Continuous Channel Simulation for Compression and Privacy

**arXiv ID:** 2609.12067 | [PDF](https://arxiv.org/pdf/2609.12067v1)

**作者:** Joseph Rowan `[一作]`, Ashish J. Khisti `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种基于极化码的多层编码（Polar‑MLC）框架，用共享随机性实现一发（one‑shot）信道编码，兼顾精确（Exact）和近似（Approximate）两种实现；

**💡 创新点**

创新点在于将极化码与多层编码相结合，并引入共享随机性以实现一发通信下的码字生成与解码，显著降低传统多层编码的复杂度与误码率；

**🔧 技术方法**

主要技术包括极化码设计、分层编码与解码、共享随机性生成、Monte Carlo 误码率评估和近似极化变换；

**📊 数据集**

实验数据主要基于模拟信道（如 AWGN、BSC 等），未使用公开数据集；

**📈 对比分析**

与传统 Reed‑Solomon、LDPC 等方法比较，Polar‑MLC 在相同信道条件下实现了更低的误码率或更高的码率，实验表明误码率下降 1–2 dB 左右；

**⚠️ 局限性**

局限性包括：需要共享随机性源，实际部署时共享成本较高；极化码在极短块长下性能仍受限；对复杂多用户或多址环境的适用性尚未完全验证。

---

## 260. Context-Aware Causal Gaze Forecasting for Human-Vehicle Interaction During In-Cabin Tracking Dropouts

**arXiv ID:** 2609.12374 | [PDF](https://arxiv.org/pdf/2609.12374v1)

**作者:** Shabnam Shabani `[一作]` (Western University), Ghazal Farhani `[通讯]` (National Research Council Canada)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在驾驶员监控系统中，研究者提出一种因果性预测框架，用于在因头部大幅旋转导致远程眼动跟踪器丢失时恢复驾驶员的注视位置。

**💡 创新点**

创新点在于将真实的跟踪器丢失视为严格的因果预测问题，设计了Causal Context‑Gated Forecaster (CCGF)，结合预丢失历史与现场视觉上下文，并通过可靠性门控动态融合两者；此外，还构建了真实的自然驾驶丢失事件数据集。

**🔧 技术方法**

技术包括：双路径LSTM编码器（处理眼动/头部历史与DINOv3视觉特征）、可靠性门控融合、概率热图预测以及离线与在线因果与非因果基线的对比实验。

**📊 数据集**

数据集为10名司机共10.5小时的自然驾驶记录，包含2047个真实跟踪器丢失事件；每个事件同步了远程GazeSense和头戴Neon眼动仪，Neon视为监督与评估参考，数据集即将公开。

**📈 对比分析**

与零阶保持、常速、DLinear、TimesNet、SegRNN、CSDI等因果预测基线以及BRITS、SAITS、GP‑VAE等非因果填补方法对比，CCGF Live模式平均像素误差为175.7px（10.5°），比最佳因果基线TimesNet低约13.7px（≈4°）；在Frozen模式下误差为210.8px，仍优于所有因果基线。

**⚠️ 局限性**

主要限制包括：Live场景使用头戴相机视角，可能导致性能过于乐观；未加入车辆动力学或行驶信号；数据仅来自10名司机，缺乏更广泛的多样性；以及尚未验证固定车载摄像头环境下的迁移效果。

---

## 261. TailWeather: from tail to extremes, a global climatological dataset for machine-learning weather forecasting

**arXiv ID:** 2609.12496 | [PDF](https://arxiv.org/pdf/2609.12496v1)

**作者:** Zhi-Song Liu `[一作]` (LUT University), Risto Makkonen `[通讯]` (Finnish Meteorological Institute)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个全球陆地0.25°分辨率的气候尾部事件数据集 TailWeather，覆盖1981–2022年，并在2023年1月部分时间内扩展，标注热浪、寒浪、强降水、极端风和气象干旱的事件、严重级别与强度分数；随后用该数据评估气象预测模型在极端事件检测上的性能，并尝试将尾部标签作为训练信号改进 AI 预测。

**💡 创新点**

创新点在于：①提供了统一、连续、基于本地气候的五类极端事件标注，弥补了灾害记录稀缺、空间不均的问题；②引入强度分数与严重等级，可灵活阈值化，支持多种阈值与复合事件；③系统性比较了极端事件检测与传统平均误差之间的差距，并展示了尾部标签与灾害记录的关联与差异；④验证了尾部训练信号对常规 RMSE 有提升但对极端事件检测可能产生负面影响。

**🔧 技术方法**

技术主要包括：基于 ERA5 重分析的统计阈值设定（经验分位数、SPI 变换）、0.25°网格处理、事件持续时间规则、强度分数与严重等级映射；模型评估使用常规 RMSE 与极端事件检测指标（CSI、SEDI）；尾部训练示例采用 Frozen FourCastNet + 可训练的 sharpening head 进行后处理。

**📊 数据集**

主要数据集：ERA5（1981–2023年）作为气候参考与变量来源；TailWeather 本身；灾害记录 EM‑DAT 与 NOAA Storm Events Database 用于与极端事件标注进行匹配与评估。

**📈 对比分析**

比较方法：在 WeatherBench2 的测试集（1981–2022）上对六套 AI 预测模型（Pangu‑Weather、GraphCast、GenCast、Aurora、FGN、IFS‑HRES）进行极端事件检测（CSI）与常规 RMSE 对比；TailWeather 与 EM‑DAT 的匹配采用阈值扫描（强度分数阈值 q），计算精确度、召回率、F1、提升度等。结果显示：平均预测精度高但极端事件检测弱，特别是极端风；尾部训练能显著降低 RMSE，但极端事件检测指标下降。

**⚠️ 局限性**

局限性：①依赖 ERA5 的偏差与 0.25° 分辨率，无法捕捉短时旋转性极端如风暴、龙卷风；②阈值固定为 1991–2020 基线，未考虑气候变化趋势；③灾害记录稀疏，评估覆盖受限，尤其在非洲、南美；④强度分数与严重等级在不同灾害中不具可比性；⑤尾部训练示例仅为后处理，未改变模型内部机制，极端事件检测能力仍不一定提升。

---

## 262. Patient-Reported Survey Data Improve Prediction of Opioid Use Disorder

**arXiv ID:** 2609.12224 | [PDF](https://arxiv.org/pdf/2609.12224v1)

**作者:** Xiyue Jiang `[一作]` (Stony Brook University), Fusheng Wang `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究评估了在全美All of Us研究计划中加入患者自报问卷数据后，对首次诊断到药物使用障碍（OUD）的预测性能是否提升。

**💡 创新点**

创新点在于：①系统量化患者自报数据相较于传统电子健康记录（EHR）所增益的预测信息；②跨越不同机器学习模型与不同时间窗口（6、12、24个月）验证这一增益的稳健性；③通过排列重要性分析揭示哪些自报问题对OUD预测贡献最大，且自报信息在多种维度（行为、功能、社会经济等）上提供互补信号。

**🔧 技术方法**

使用了多种监督学习技术，包括逻辑回归、随机森林、XGBoost、LightGBM、多层感知机（MLP），以及序列模型（LSTM、GRU、Transformer），并采用逻辑回归与梯度提升树（GBDT）在大样本上进行对照。

**📊 数据集**

数据集来自All of Us研究计划的Curated Data Repository，包含267,747名曾经服用阿片类药物的参与者，其中15,287人为OUD病例；利用其电子健康记录和四类问卷（Basics、Lifestyle、Overall Health、Social Determinants of Health）。

**📈 对比分析**

比较方法：在同一训练/验证/测试划分下，分别训练仅EHR模型和EHR+问卷模型，评估指标为PR-AUC（主要）和ROC-AUC。结果显示，加入问卷数据后，PR-AUC在所有模型与窗口中均提升0.009–0.051（例如24个月LightGBM从0.6219提升到0.6603），ROC-AUC提升幅度为0.0013–0.041；在24个月窗口上，增益最大。

**⚠️ 局限性**

限制包括：①OUD标签依赖诊断编码，可能出现漏诊或误诊；②病例与对照的索引日期定义差异导致观测窗口和问卷完成机会不一致；③问卷完成率低且与OUD状态相关，可能产生信息性缺失；④仅在内部测试集评估，缺乏外部验证、校准和公平性评估；⑤序列模型未充分利用数值测量的时间变化，可能限制其性能。

---

## 263. Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding

**arXiv ID:** 2609.12243 | [PDF](https://arxiv.org/pdf/2609.12243v1)

**作者:** Minoo Ahmadi `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在推理时不需要后训练的语言模型中提出一种新的采样方法——Chopthin-Consensus Power Sampling（CCPS），将粒子群的权重保留并通过语义多数投票选择答案。

**💡 创新点**

创新点在于：①用 Chopthin 取代传统的等权重重采样，既保证有效样本数（ESS）也保留多样化的推理路径；②在终端选择阶段引入语义多数投票机制，先合并完全相同的轨迹，再聚类语义等价答案，最后选取支持最多不同轨迹的答案。

**🔧 技术方法**

核心技术包括：序贯蒙特卡罗（SMC）采样、功率分布（power distribution）目标、Chopthin 有限权重重采样算法、以及基于聚类的语义多数选择器（对文本和代码分别采用行为签名聚类）。

**📊 数据集**

实验使用三大开源大语言模型（Qwen2.5-Math-7B、Qwen2.5-7B、Qwen3-4B）在五个推理基准（MATH500、GSM8K、AIME、GPQA、HumanEval）上进行测试。

**📈 对比分析**

与基线 Power‑SMC（系统重采样+权重抽样）以及无重采样、低温采样等对照方法比较，CCPS 在 15 个模型/基准组合中提升“oracle 覆盖率”13/15 份，最终答案准确率在 14/15 份显著提升，最大提升 10.6pp（如 Qwen3-4B 在 GPQA 上）。

**⚠️ 局限性**

局限性包括：①对参数 η 的敏感性需要经验调优；②在某些任务（如 Qwen3-4B 的 HumanEval）中仍未超越基线；③方法仍依赖于相同的提议分布和温度设定，若提议不佳可能无法充分利用多样化。

---

## 264. "People can change, and patterns can be broken": Contextualizing Tradeoffs in Automated Decision-Making Systems

**arXiv ID:** 2609.12288 | [PDF](https://arxiv.org/pdf/2609.12288v1)

**作者:** Rabeya Bosri `[一作]` (University of Alberta), Bailey Kacsmar `[通讯]` (University of Alberta)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 777 名受访者进行问卷调查，评估在四种高风险场景（医疗保险、招聘、监狱判刑、抵押贷款）中，人类决策（HDM）与自动化决策（ADM）的可接受性及其在隐私、公平与安全三大目标之间的权衡偏好。

**💡 创新点**

首次系统性比较不同高风险情境下的用户对 HDM 与 ADM 的偏好，并将隐私、公平与安全的多维权衡映射到用户决策框架，揭示公平观念超越单一统计指标，且与隐私、准确性、易受攻击性相互交织。

**🔧 技术方法**

采用问卷设计与自由文本分析相结合的方法，定量使用 Wilcoxon、Kruskal‑Wallis、Chi‑square 等统计检验，定性采用双人编码的内容分析。

**📊 数据集**

无公开数据集；研究基于受试者自填的情境描述与选项，不使用真实数据集。

**📈 对比分析**

对比方法主要是统计显著性检验与比例差异检验；结果显示 HDM 在大多数场景中显著更受欢迎，用户对公平与隐私的优先级在不同情境下呈现显著差异，表明情境驱动权衡决策。

**⚠️ 局限性**

样本为 WEIRD 人群，缺乏全球多样性；自述数据可能受社会期望偏差；情境描述与真实决策情境复杂度不完全匹配；未对实际 ADM 系统性能进行验证。

---

## 265. Scan the Skill, Govern the Action: Composing Registry Verdicts with Runtime Consequence Control

**arXiv ID:** 2609.12001 | [PDF](https://arxiv.org/pdf/2609.12001v1)

**作者:** Rohit Taneja `[一作]` (Pheo Inc), Travis Weber `[通讯]` (Pheo Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对 OpenClaw 公共 Agent 技能安全进行测量，提出并实现了 OATS（Open Agent Trust System）运行时控制点，评估其对权限违规和恶意检测的效果；

**💡 创新点**

创新点在于将权限与恶意性分离，设计仅依赖命令字符串的确定性解析器与账本，实现在低成本且可审计的运行时权限检查；

**🔧 技术方法**

采用正则/模式匹配的命令分类器、Python 实现的 CLI Hook、Ledger 记录与确定性解析器，不使用任何机器学习模型；

**📊 数据集**

使用 OpenClaw ClawHub 公共数据集 66,192 个技能版本，并结合内部测量脚本生成实验数据；

**📈 对比分析**

通过与现有三种扫描器比较，重叠率低；对 705 个清洁技能进行权限违规测算，门控平均耗时 67 ms，吞吐约 20 /s，误报率 0、真阳性率 92%，对执行命令的分析显示 34.7% 的执行类在文档中未出现；

**⚠️ 局限性**

局限性包括仅基于单一冻结的仓库快照、缺乏对恶意召回率的评估、仅检索 fenced 代码块忽略自然语言、解析器实现不公开、对多命令或高级隐藏执行缺乏覆盖。

---

## 266. T-GADE: Thermodynamical Generative-AI-Driven Evolution of LLM Artifacts

**arXiv ID:** 2609.12286 | [PDF](https://arxiv.org/pdf/2609.12286v1)

**作者:** Kyoko Ogawa `[一作]` (Osaka Metropolitan University), Naoki Mori `[通讯]` (Osaka Metropolitan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种名为T-GADE的进化框架，利用热力学自由能原理和LLM生成的遗传算子来进化具有说明与可执行代码结构的“工件”，并在在线箱子装填任务上验证其效果。

**💡 创新点**

创新点包括：①将自由能最小化统一到生成式与持续式更新中；②引入费米型与玻色型占用规则以控制基因型重复；③在LLM变异与评估之间实现解耦，允许在不改动模型的情况下通过多样性与质量共同驱动选择；④恢复了EoH的零温度生存规则并在此基础上提升性能。

**🔧 技术方法**

技术手段主要包括：热力学遗传算法、自由能目标、费米/玻色占用策略、LLM基因变异算子（E1/E2/M1/M2）、多维归一化特征向量表示多样性、利用Schur补计算多样性增益、统计检验（Mann‑Whitney、Cliff’s δ）及实验对比框架。

**📊 数据集**

使用了在线箱子装填任务的数据集：5个训练实例（每个5000件，容量100）、5个确认实例（容量100）与5个转移实例（容量500），评估指标为相对箱数溢出比例（excess）。

**📈 对比分析**

与EoH在相同生成模型、算子和评估器下进行对比；在20次独立运行中，T‑GADE的生成式玻色型（T=0.003）训练中位excess从1.152%降至0.815%（约29%提升，p=0.042），验证选择后转移excess达到0.496%与EoH持平；持续式玻色型T=0.003在质量截断下能保留更多多样性；其他配置的性能相对较差或相当。

**⚠️ 局限性**

局限性包括：需要先验的能量评估器、特征定义和LLM算子；目前仅针对单一启发式任务；温度设定为固定且无自适应；未提供全局最优性保证；占用规则与温度的交互机制尚未完全理论化；对解释–代码继承稳定性的分析有限；在更复杂或多模态任务上的可推广性待验证。

---

## 267. Fed-Equilibrium Framework for Topological Pareto Control in Robust and Fair Clinical Federated Learning

**arXiv ID:** 2609.11937 | [PDF](https://arxiv.org/pdf/2609.11937v1)

**作者:** Ting Xu `[一作]` (University of Calgary), Henry Leung `[通讯]` (University of Calgary)

**通讯引用:** 17049 | [OpenAlex ID](https://openalex.org/A5061884304)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出 Fed-Equilibrium 框架，利用两阶段梯度控制实现公平聚合；

**💡 创新点**

创新点在于将几何质量保证与拓扑 Pareto 控制结合，自动寻找 Pareto 肿瘤点，实现知识共存与公平；

**🔧 技术方法**

使用 FedQLoRA 参数高效微调、OMOP CDM 语义统一、余弦相似度门控、Pareto 前沿敏感性扫描；

**📊 数据集**

利用加拿大 CNODES 以及美国 SyntheticMass 两套合成注册数据（共 38,261 条记录）；

**📈 对比分析**

与传统 FedAvg、EqualAvg 对比，Fed-Equilibrium 在少数节点损失从 0.857 降至 0.340（提升 60%）且多数节点保持 0.985 的性能；

**⚠️ 局限性**

局限在于仅基于合成数据、Pareto 权重采用离线扫描、未考虑多模态数据、对复杂拜占庭攻击的鲁棒性待验证。

---

## 268. FRIST: FMRI Representation Informed Shared-space Training Improves EEG-only Individual-Finger BCI Decoding

**arXiv ID:** 2609.12298 | [PDF](https://arxiv.org/pdf/2609.12298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 269. A Multimodal Explainable Deep Learning Framework for Alzheimer's Disease Diagnosis using 3D Magnetic Resonance Imaging and Clinical Data

**arXiv ID:** 2609.12410 | [PDF](https://arxiv.org/pdf/2609.12410v1)

**作者:** Yusuf Brima `[一作]` (Osnabruck University), Antoine Vacavant `[通讯]` (Universite Clermont Auvergne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建了一种可解释的多模态深度学习框架，用3D结构磁共振成像（MRI）和临床表格数据进行阿尔茨海默病（AD）诊断，并通过SHAP、Grad‑CAM++等方法分析不同模态、融合策略和外部数据集下的解释一致性。

**💡 创新点**

创新点：①首次系统评估多模态融合在不同诊断任务（CN vs MCI vs AD、两两比较）和不同数据集（ADNI vs OASIS‑3）下的性能与解释差异；②提出跨模态交叉注意力（cross‑attention）融合机制，并展示其对解释贡献的影响；③将可解释性分析与外部验证结合，揭示解释的分布漂移特性。

**🔧 技术方法**

技术：3D DenseNet‑121 作为MRI编码器；多层感知机（MLP）作为表格编码器；两种融合策略（特征拼接与跨模态注意力）；Adam优化器；交叉熵/二元交叉熵损失；样本重加权；Grad‑CAM++、Score‑CAM、SHAP、Integrated Gradients用于解释。

**📊 数据集**

数据集：内部训练/验证/测试采用ADNI 6,479例（CN 2,240，MCI 3,026，AD 1,213）；外部验证采用OASIS‑3 1,703例（CN 1,323，MCI 295，AD 85），两者均包含T1‑weighted 3D MRI和统一的7项临床特征。

**📈 对比分析**

比较方法：在ADNI内部和OASIS‑3外部分别使用AUC‑ROC、微平均F1和宏平均AUC评估三类与两两分类任务。结果显示：ADNI上表格模型整体最佳，交叉注意力在MCI‑AD区分上最高；OASIS‑3上仅影像模型最佳，融合策略未显著提升；不同模态和融合策略在不同任务/数据集间表现不一致。

**⚠️ 局限性**

局限性：①可解释性评估对影像侧仅依赖定性示例，缺乏量化空间一致性指标；②OASIS‑3与ADNI在诊断构成、采集协议和临床变量上差异多，难以单独定位性能与解释变化来源；③使用的临床特征（如MMSE）与诊断高度相关，可能掩盖真正的生物学信息；④未提出统一跨模态解释框架，未能直接比较表格与影像解释的一致性。

---

## 270. DIA: Denoising Intermediate Advantage for Diffusion Policy Optimization

**arXiv ID:** 2609.12245 | [PDF](https://arxiv.org/pdf/2609.12245v1)

**作者:** Arjun Sohal `[一作]` (University of Toronto), Alan Aspuru-Guzik `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对预训练的扩散式机器人策略进行强化学习微调，提出DIA方法通过在扩散链内部估计部分去噪动作的价值来实现细粒度的信用分配；

**💡 创新点**

在扩散策略的两层MDP框架下引入对部分去噪状态的价值函数，构造内部优势并与环境层PPO优势融合，从而实现对每一步去噪决策的状态相关信用赋值；

**🔧 技术方法**

使用政策梯度、GAE、两层MDP、扩散策略、内部价值估计、优势混合与尺度匹配等技术；

**📊 数据集**

在Robomimic、FurnitureBench、Franka Kitchen和D3IL等四个基准数据集上进行实验；

**📈 对比分析**

与DPPO以及多种单步/动作块强化学习基线比较，DIA在奖励、成功率、到达成功状态的时间和执行效率上均有显著提升，尤其在长时程任务上表现突出；

**⚠️ 局限性**

需要大量在线交互数据、样本效率相对较低，且当前仅在仿真环境中验证，真实机器人部署与 sim-to-real 转移仍待研究。

---

## 271. Open Source Stewardship Communities: "We need you, but not your pull request"

**arXiv ID:** 2609.12236 | [PDF](https://arxiv.org/pdf/2609.12236v1)

**作者:** Gregorio Robles `[一作]` (Universidad Rey Juan Carlos), Daniel M. German `[通讯]` (University of Victoria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对209个OSS项目的AI相关贡献政策和5个具体案例的分析，研究了在AI驱动的代码实现环境下，项目关闭外部代码贡献路径，形成“stewardship community”（管理者主导的社区）的现象，并探讨其对贡献者多样性、需求实现和维护者继任的影响。

**💡 创新点**

创新点在于提出“stewardship community”这一组织模型，系统阐述了门控外部实现如何改变传统开放式贡献链，强调治理结构对社区可持续性与人力资源更新的决定性作用。

**🔧 技术方法**

采用文献综述、案例研究和政策分析的方法，主要基于GitHub存储库的公开信息进行定性分析。

**📊 数据集**

使用的数据集包括：①209个OSS项目的AI相关贡献政策记录；②5个案例项目（Codex、Ladybird、Swamp、Symfony Language Tools、Ghostty）的GitHub仓库及其贡献文档。

**📈 对比分析**

对比传统开放式贡献模式与门控stewardship模式，从贡献者多样性、需求实现速率和维护者继任路径三个维度进行定性比较；未给出数值性能指标，但通过案例说明门控可降低审查成本、提高核心团队效率、但可能削弱社区多样性与继任能力。

**⚠️ 局限性**

限制包括：①缺乏定量评估（如社区活跃度、bug解决率等指标）；②案例数量有限，难以全面代表OSS生态；③研究主要基于政策文本与公开讨论，缺乏对AI编码质量与实现速度的客观测量。

---

## 272. Receiver-Surface Hit Patterns via Legendre Approximation for Molecular Signal Detection

**arXiv ID:** 2609.12089 | [PDF](https://arxiv.org/pdf/2609.12089v1)

**作者:** Yasin Bastug `[一作]` (Bogazici University), H. Birkan Yilmaz `[通讯]` (Bogazici University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于勒让德多项式展开的分子信号检测方法，利用球面接收机表面吸附点的方向分布来识别发射机是否正在通信，并将其推广到Viterbi序列检测器；

**💡 创新点**

创新点在于将接收机表面的方向性签名转化为勒让德多项式系数，构建高阶角分布逼近，既实现了无记忆的方向检测，又结合Viterbi算法实现了序列级别的空间-时间概率检测，显著提升低计数环境下的误码率；

**🔧 技术方法**

使用了勒让德多项式展开、概率似然比检验、泊松点过程建模、Viterbi动态规划、基于粒子模拟的经验库构建以及带岭正则化的最小二乘估计等技术；

**📊 数据集**

实验数据来源于粒子基分子扩散（MCvD）模拟，生成了不同距离、时间区间的吸附点分布库（ℒ_Leg），并在该模拟框架下评估各种检测器；

**📈 对比分析**

与固定阈值检测器、计数式Viterbi检测器以及混合勒让德-Viterbi检测器进行了比较，使用BER、误报/漏报概率等指标，结果显示勒让德检测器在计数稀疏（短符号时长或低分子数）情况下优于计数型方法，完整勒让德-Viterbi在所有评估场景中取得最优误码率；

**⚠️ 局限性**

局限性包括：仅考虑单一发射机与完美吸收球面接收机；实验完全基于模拟，未验证在实际环境中的鲁棒性；需要预先构建经验库，且完整勒让德-Viterbi的计算复杂度较高；对接收机几何和发射机方向的精确估计要求较高。

---

## 273. Certified Safety Curation: Distribution-Free Guarantees for Safe Offline Reinforcement Learning

**arXiv ID:** 2609.12014 | [PDF](https://arxiv.org/pdf/2609.12014v1)

**作者:** Adam Haroon `[一作]` (Iowa State University), Cody Fleming `[通讯]` (Iowa State University)

**通讯引用:** 938 | [OpenAlex ID](https://openalex.org/A5113457201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于片段偏好学习的安全价值，配合Learn-then-Test校准阈值，得到无分布假设的训练集安全保证，并在此基础上进行行为克隆。

**💡 创新点**

创新点在于用仅状态级的片段偏好训练安全价值并提供高概率安全组合保证，同时不需要逐步成本函数，实现了弱监督下的安全离线强化学习。

**🔧 技术方法**

使用Bradley-Terry模型训练安全价值、Learn-then-Test阈值校准以及行为克隆等技术。

**📊 数据集**

在DSRL benchmark 的 15 个任务（5 个速度限制运动学 + 10 个 SafetyGymnasium 导航任务）上验证。

**📈 对比分析**

与全标记的 BC‑Safe、CDT、CPQ、COptiDICE 等基线比较，经过认证过滤后可在 11/15 任务中安全满足预算，且在被认证的 3/15 任务上保持高奖励；未认证时提供保守选择。

**⚠️ 局限性**

局限在于安全保证仅覆盖训练集组成，无法完全预测最终策略安全；需要 200 条预算超标标签，且对极端尾部评分不可靠；偏好标签来源于仿真成本，未验证真实人类偏好。

---

## 274. Affective Agent: On-Device Personalized Intervention Reasoning for Wearable Systems

**arXiv ID:** 2609.12322 | [PDF](https://arxiv.org/pdf/2609.12322v1)

**作者:** Reina Mun `[一作]` (Harvard University), Vijay Janapa Reddi `[通讯]` (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 Affective Agent 三层架构，实现在可穿戴设备上对情绪状态进行感知、个性化记忆驱动和基于子1B 语言模型的本地化干预决策；

**💡 创新点**

创新点包括：①将感知、个性化记忆与推理三层拆分；②使用结构化短期/长期记忆驱动个性化，无需对模型权重做用户特定训练；③采用两步推理（评估+验证）显著提升决策可靠性与可解释性；

**🔧 技术方法**

核心技术：子1B TinyML 语言模型（Qwen2.5‑0.5B / Qwen3‑0.6B）；结构化记忆（episodic & semantic）与偏好生命周期；5‑分钟窗口特征提取+三类弱监督分类器；synthetic scenario generation + reasoning‑complexity curriculum + programmatic two‑pass supervision；在 Raspberry Pi CM5 上实现低功耗推理；

**📊 数据集**

数据集：基于 WESAD、SWELL‑KW、En‑Gage 的生理先验构造 14 名虚拟用户的合成数据，包含多种噪声与情境变异；未使用真实传感器数据；

**📈 对比分析**

比较方法：对 1,600 次决策点做动作准确率、宏 F1、模态准确率、schema 合规率、ECE 的度量；与规则基线对比；两层推理与记忆 ablation 评估；结果：子1B 模型动作准确率0.782/0.811、宏 F1 0.616/0.677、模态准确率0.864/0.701，显著优于规则基线；记忆缺失或单步推理时性能显著下降；

**⚠️ 局限性**

局限性：评估仅基于合成情境，缺乏真实用户纵向验证；自我置信度校准仍差（ECE 0.25‑0.30），需后置校准；推理延迟约 25‑30% 的 5‑分钟感知周期，需进一步压缩；能耗未直接测量；

---

## 275. Guardrailed Meta-Agent Loops: Stress-Testing Policy Pinning, Budget Bounds, and Crash Recovery

**arXiv ID:** 2609.12216 | [PDF](https://arxiv.org/pdf/2609.12216v1)

**作者:** Qinzhen Ma `[一作]` (Rice University), Jialin Wu `[通讯]` (University of California, San Diego)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个基于仿真的自我改进代理测试平台，利用显式的权限、计费与恢复三项合同来验证可审计的有限适配过程；

**💡 创新点**

创新点在于将人类策略与机器演化严格分离，构建了可共测的三合同框架，揭示了恢复成功并不等同于一次性执行，且提供了可验证的边界与协议；

**🔧 技术方法**

采用哈希固定策略、代码拥有的特征目录、前缀计费链、可重放日志、离线确定性规划器以及仿真资源管理等技术；

**📊 数据集**

使用合成的衣物折叠任务与VLA仿真环境，生成1,000个伯努利样本的评估集，构成实验数据集；

**📈 对比分析**

通过配对的50种随机种子2×2实验、预算扫描和240次故障注入进行比较，结果显示“轮次增长”显著提升目标达成率并降低计算开销，而“空闲增长”无效；在预算160下实现100%目标命中；所有崩溃场景恢复了预定结果，但30例存在规划调用重复，体现恢复与一次性执行的区别；

**⚠️ 局限性**

实验局限在于仅使用单一校准的仿真环境、合成任务、离线确定性规划器以及模拟的计算度量，缺乏真实硬件、动态环境和外部成本计费，且无法验证在非确定性或外部副作用场景下的安全性与可扩展性。

---

## 276. Evidence Records for Public-Record Changes: Definitional Class Membership as the Certifiable Citation Unit

**arXiv ID:** 2609.11978 | [PDF](https://arxiv.org/pdf/2609.11978v1)

**作者:** Amadeus Brandes `[一作]` `[通讯]` (Independent Researcher), Amadeus Brandes (Independent Researcher)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建了Evidence Record原语，用于可验证、可引用的公共记录变更，并在ClinicalTrials.gov中实现TrialDiff系统以生成基于事件类的可证实变更记录。

**💡 创新点**

其创新点在于将可证实的定义性成员身份与不可证实的评估性优先级明确划分，利用哈希锁定实现完全可重现的生成，提供可验证的变更引用单元。

**🔧 技术方法**

采用的技术包括内容地址哈希、canonical JSON序列化、JSON Patch结构化变更、哈希固定的生成器版本控制以及基于规则的事件类判定。

**📊 数据集**

实验使用了100个乳腺癌相关的ClinicalTrials.gov介入试验共4,485个版本对的数据集，生成了97条Evidence Record。

**📈 对比分析**

通过对生成的事件类成员与原始补丁进行对比验证，可证实性检查及时发现错误并发布修正，性能表现为生成可重复、资源占用有限，但未与传统评估工具直接比较。

**⚠️ 局限性**

局限性包括：仅在给定语料与规则集哈希内完全可证实；事件类覆盖有限；评估优先级无法证实；未覆盖随机抽样的全局统计；且某些判定仍需外部注册状态。

---

## 277. DERA: Detached Edge-Residual Adaptation for Prohibited item Detection

**arXiv ID:** 2609.12411 | [PDF](https://arxiv.org/pdf/2609.12411v1)

**作者:** Yonathan Michael `[一作]`, Naoufel Werghi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为 DERA 的边缘感知框架，结合层次上下文特征与像素差分边缘金字塔，并通过学习实例掩码提取的边界先验来引导检测器，仅在早期视觉阶段注入零初始化的残差头，保持原始检测器不变，只对边界信息进行细粒度调整。

**💡 创新点**

创新点在于：①将边界先验从训练掩码中单独学习，并在特征提取过程中使用停止梯度，避免与主检测目标互相干扰；②利用边缘先验对像素差分特征进行门控，并通过零初始化的残差投影仅在浅层融合处注入修正；③采用三阶段训练策略，先自适应 X‑ray 领域，再独立学习边界，最后只调优残差，从而极大减少可训练参数（仅 14.7K）。

**🔧 技术方法**

技术包括：多流检测器（视觉分支 + 像素差分分支）、Swin Transformer 作为上下文骨干、PiDiNet 作为像素差分网络、边界先验分支（基于实例掩码的内部轮廓）、零初始化残差投影、基于 Grounding DINO 的文本条件检测头、以及三阶段训练策略（基础自适应 → 边界学习 → 残差微调）。

**📊 数据集**

使用的公开 X‑ray 数据集有：PIDray（约 47k 张图片，12 类），CLCXray（约 9.5k 张图片，12 类），以及 STCray（约 46k 张图片，21 类）。

**📈 对比分析**

在三大数据集上与 Grounding DINO 及多种视觉/视觉‑语言基线对比，DERA 在 AP 上平均提升约 2–3 点，AP_50 与 AP_75 也均有明显提升，尤其在 CLCXray 与 STCray 的 AP_75 上提升显著；在小尺寸目标 AP_S 上最高可提升 7.5 点；与基线相比，参数量仅增加 0.354M，推理时延仅略增 22 ms。

**⚠️ 局限性**

局限性在于：①边界指导不能完全解决实例分割与查询一致性问题，仍可能出现目标合并、定位不精确或重复检测；②依赖训练时的掩码提取，若掩码质量不佳会影响边界先验学习；③残差仅注入浅层，可能对更深层的全局上下文贡献有限；④在极高 IoU 阈值下，定位精度仍略逊于部分非边界方法。

---

## 278. DU-NO: A Parameter-Efficient Double U-Shaped Neural Operator for Phase-Resolving Wave Modeling

**arXiv ID:** 2609.12115 | [PDF](https://arxiv.org/pdf/2609.12115v1)

**作者:** Enrique Hernandez Noguera `[一作]` (LSU), Mahdi Abdelguerfi `[通讯]` (LSU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种双U形神经算子，在多尺度谱路径中仅在最浅两层插入轻量级U-Net分支，以显著减少参数并提升近岸波浪预测的自回归精度。

**💡 创新点**

创新点在于基于采样论证的尺度-频率分支放置原则：把卷积分支仅放在保留谱模式最少、能量集中于高频的浅层，从而在维持或提升精度的同时将参数压缩至原U‑FNO的十分之一。

**🔧 技术方法**

技术核心为多尺度傅里叶神经算子骨干（每层保留不同数量的低频模式）+两层轻量级3×3卷积U‑Net分支，配合残差预测、10步教师强制训练和自由回归评估。

**📊 数据集**

使用公开的FUNWAVE‑TVD近岸波浪数据集（81个受控实验，128×128网格）作为主基准，并对2D Navier–Stokes和PDEBench淹水方程进行泛化测试。

**📈 对比分析**

与U‑FNO、U‑NO、Transolver、FNO、U‑Net等六种架构在统一训练协议下对比，结果显示在FUNWAVE上相对L₂误差低14.9%，参数仅3.64M（U‑FNO的3.9M），内存降低4.4倍，延迟相当；在淹水基准上更优，在Navier–Stokes上保持竞争力。

**⚠️ 局限性**

局限包括仅在二维规则网格上验证，未扩展到三维或不规则几何；训练仅针对单一海底地形，未测试跨岸线泛化；模型仅输出表面高度，未包含速度场；缺乏对分支放置原则的严格解析证明，且长回归误差累积仍需进一步研究。

---

## 279. Can We Trust LLM Judges: A Study of Capability-Dependent Biases and Multi-Judge Ensemble for Bias Calibration

**arXiv ID:** 2609.12002 | [PDF](https://arxiv.org/pdf/2609.12002v1)

**作者:** Gemma Zhang `[一作]` (Microsoft), Sulaiman Vesal `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM-as-a-judge在绝对评分任务中的系统性偏差进行定量研究，并提出无标签的加权投票校正方法。

**💡 创新点**

引入无标签的disagreement-based FPR/FNR估计以及加权多数投票，可在非平稳任务分布下匹配完美错误率知识。

**🔧 技术方法**

计算FPR/FNR、方向偏差、Pearson相关，设计三种在线错误率估计（Bayesian、分层、disagreement-based），并构造加权多数投票。

**📊 数据集**

QuALITY、GSM8K、MBPP、AIME四个客观基准。

**📈 对比分析**

与单一评判者、未加权多数投票对比；在12周漂移实验中，disagreement-based WMV平均误差<0.04，准确率接近oracle，显著优于单评判者和未加权投票。

**⚠️ 局限性**

仅涵盖三大模型家族，假设多数评判者大部分正确，缺乏真实时间序列验证，需进一步扩展覆盖面和鲁棒性。

---

## 280. Adaptive Agent Design

**arXiv ID:** 2609.12486 | [PDF](https://arxiv.org/pdf/2609.12486v1)

**作者:** Raj Kiriti Velicheti `[一作]` (University of Illinois Urbana Champaign), Tamer Başar `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在非马尔可夫环境下，利用可参数化的内部状态转移核 fθ 并结合 soft Q‑learning 的双层优化框架，自动设计代理的状态压缩方式和对应的决策策略。

**💡 创新点**

创新点包括：① 将代理自身的状态转移核视为可学习的参数化对象；② 在非马尔可夫环境下证明 soft Q‑learning 对于任何 fθ 都几乎必然收敛到软 Bellman 固定点；③ 在 POMDP 下证明外层目标 V_D(θ) 的梯度 Lipschitz 连续，进而实现零阶优化（ZOO）和贝叶斯优化（BO）两种全局/局部优化方案；④ 通过对内部状态容量、基准核选择和行为策略的系统实验，展示设计空间与性能之间的关系。

**🔧 技术方法**

技术手段主要包括：软 Q‑学习（Soft Q‑learning）实现内部策略求解；零阶随机梯度估计（两点估计）和投影；贝叶斯优化（GP-UCB）结合 Matérn‑5/2 核；理论分析使用马尔可夫链的稳定性、随机逼近、软 Bellman 收敛性以及 RKHS 相关的调度与收敛率。

**📊 数据集**

使用自定义的非马尔可夫“连续路由”环境（观察空间 {0,1,2,3}，动作空间 {0,1,2}，隐藏目标状态 c∈{0,1,2}），通过模拟生成的交互数据进行实验。

**📈 对比分析**

对比方法包括：① 全局网格搜索（Grid Search）得到理论最优奖励 0.474；② 零阶梯度上升（ZOO）仅达到 0.191；③ 贝叶斯优化（BO）与网格搜索相当，同样得到 0.474。实验还展示了基准核组合、行为策略偏差和内部状态容量对最终奖励的影响，表明 BO 在全局搜索上优于 ZOO。

**⚠️ 局限性**

局限性：① 行为策略 π_b 及数据集固定，未考虑在线自适应数据收集；② 内部状态空间维度固定，未研究动态扩展；③ 只考虑单一代理对静态非马尔可夫环境，未涉及多代理或动态环境；④ 证明与实验仅在合成环境上完成，缺乏真实任务验证。

---

## 281. IMPLY: Physically Anchored Consistency for World-Model Rollouts

**arXiv ID:** 2609.12441 | [PDF](https://arxiv.org/pdf/2609.12441v1)

**作者:** Aman Mehta `[一作]` (Independent Researcher), Riya Baviskar `[通讯]` (Independent Researcher)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于物理推断的世界模型一致性分数，并通过两次校准推动对模型产生的多条推理未来进行锚定与评估。

**💡 创新点**

创新点在于将单一物体的质量与摩擦参数从不同速度的推送中逆向拟合出来，形成物理一致性度量，并通过锚定校准数据来揭示模型是否真正理解了被操纵的物体，从而克服仅基于自我一致性的盲区。

**🔧 技术方法**

主要技术包括逆向模拟拟合、Ridge 回归读出位移、V-JEPA 2-AC 适配、CALIPER 仿真环境、AUROC 与 R² 评估，以及锚定一致性分数的实现。

**📊 数据集**

使用的数据集为 CALIPER 物理仿真数据集（包含多速度推送轨迹）以及 1500 条训练样本的 V-JEPA 2-AC 训练集。

**📈 对比分析**

在控制实验中与自我一致性、SC3-Eval 等传统一致性信号对比，锚定一致性在噪声增大至 0.2 时仍保持 AUROC ≈1.0；在真实模型上，它在选取 rollouts 时的平均误差可逼近 oracle 误差 0.003，远优于仅依赖第一样本或一致性平均的做法。

**⚠️ 局限性**

局限性包括：需要已知的前向模拟器和单一接触类型；锚定一致性仅在模型已观察到目标物体时有效；当前仅在确定性 rollouts 下验证，无法直接用于随机采样；以及仅在特定 V-JEPA 2-AC 架构上演示，尚未在更大或更复杂模型上测试。

---

## 282. Information Specialization and Constrained Synthesis in Multi-Agent LLM Forecasting: A Prospective Live-Study of the 2026 FIFA World Cup

**arXiv ID:** 2609.12495 | [PDF](https://arxiv.org/pdf/2609.12495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 283. Observation-Anchored Selective Assimilation for Longitudinal Tumor-State Proxy Forecasting in Post-Treatment Glioma

**arXiv ID:** 2609.12435 | [PDF](https://arxiv.org/pdf/2609.12435v1)

**作者:** Yeonjae Jung `[一作]` (Yonsei University), Minwoo Shin `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了基于观察锚定的选择性同化（OASA）框架，用以预测脑胶质瘤术后MRI中肿瘤状态的连续代理。

**💡 创新点**

创新点在于：①将中间观察作为状态锚点，并通过分层案例规则与体素软门控实现选择性更新；②引入近阈值校准以细调支持边界。

**🔧 技术方法**

采用 SegMamba 3D 编码解码器实现单步预测，输出归一化速度与变更概率；随后应用 OASA 规则与可选校准进行同化。

**📊 数据集**

使用来自 TCIA 的 MU‑Glioma Post 数据集（203 位患者，120 条无新治疗的三元组），包含 T1c、T1n、T2F、T2W MRI 与相应标签。

**📈 对比分析**

与初始预测、滚动预测、持久、直接预测以及形态膨胀等基线对比，OASA 在 Dice@0.2 与持久相当，在 Dice@0.5 超过持久且略优于膨胀，RMSE 与持久相近；校准后 Dice@0.2 略升但 FP 也随之增多。

**⚠️ 局限性**

局限性包括：仅预测标签衍生的代理而未评估临床终点；样本仅来自单中心的 15 个测试三元组；未考虑真实治疗干预的影响。

---

## 284. When Does AI Augment Work? A Workflow-Level Framework for Human-Agent Collaboration

**arXiv ID:** 2609.12482 | [PDF](https://arxiv.org/pdf/2609.12482v1)

**作者:** CIVIC-AI Collaboration `[一作]`, Diyi Yang `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出以工作流程为单位的人工智能增值框架，并定义了六条条件，结合案例研究阐述了如何实现有效的人机协作。

**💡 创新点**

创新点在于将增值评价从任务层面升至工作流程层面，并将短期与长期两层条件结合，弥补了现有治理框架的缺口。

**🔧 技术方法**

主要使用概念框架分析和案例研究方法，示例中采用了自适应访谈 AI 代理（如 SparkMe）来说明工作分配。

**📊 数据集**

参考了新加坡劳动力市场调查数据、企业 AI 采用统计以及社会调查试点数据作为实证材料。

**📈 对比分析**

通过构建工作流程记录，按六条条件对比评估 AI 与传统流程的价值、控制、问责和学习路径，结果显示在满足所有条件时可实现可持续增值。

**⚠️ 局限性**

局限性包括缺乏长周期数据验证、案例研究范围有限、不同行业的阈值差异以及人力技能分布不均导致评估偏差。

---

## 285. ForkSCOPE: Charting the Agentic Garden of Forking Paths

**arXiv ID:** 2609.12438 | [PDF](https://arxiv.org/pdf/2609.12438v1)

**作者:** Arjun Balaji `[一作]` (Columbia University), Tian Zheng `[通讯]` (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种自下而上的人机协作框架，用于从大量自动生成的分析代码和报告中提取决策点，聚类成选项并组织成分支，构建可审计的“分析园地”，并通过交互式视图实现对关键分支的可视化与检验；

**💡 创新点**

创新点在于：①不依赖预设决策分类表，而是通过LLM提取原始决策并自下而上生成分支与选项结构；②实现可审计的证据链，决策点与代码/文本原文绑定；③提供交互式查看器，聚焦分支结构而非单一答案，支持人类在不暴露具体路径的前提下审计不确定性；

**🔧 技术方法**

主要技术包括：多阶段LLM驱动的提取（Sonnet、Opus）、分支诱导与修复（co-occurrence、foreclosure test）、基于JSON schema的缓存与可复现性控制、可视化工具（多视图交互式浏览器）、以及人机协作的技能（Skill）与审计门控；

**📊 数据集**

使用了Bertran等人生成的Agentic Multiverse数据集（共223份分析，其中19份为人类作者），并将其与同一问题下的多分析结果（约3946个决策点）进行综合；

**📈 对比分析**

比较方法：对同一语料进行多次构建以评估可复现性和稳定性，计算决策点/选项/分支层面的ARI、覆盖度、Shapley归因等指标；性能方面：构建成本约804美元（2273次LLM调用），重建时可完全缓存；多次构建显示核心分支高度一致，低覆盖尾部表现出一定波动；

**⚠️ 局限性**

局限性包括：仅在单一案例研究中验证，未检验跨领域泛化；归因基于观察性模型，缺乏实验分配；LLM依赖度高，模型更新可能改变结果；未拆分复合决策点；缺乏实际用户体验评估；对人类与机器语料的对比受限；保守归一化导致复杂度可能被高估。

---

## 286. TokenMapper: A Step Toward Interoperable Speech Token Translation

**arXiv ID:** 2609.12563 | [PDF](https://arxiv.org/pdf/2609.12563v1)

**作者:** Tal Kozakov `[一作]` (Ben Gurion University of Negev), Eliya Nachmani `[通讯]` (Ben Gurion University of Negev)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发TokenMapper框架，实现不同语音离散化tokenizer之间的直接token翻译，避免波形桥接，降低延迟。

**💡 创新点**

引入方向感知的token映射机制，支持单/多codebook不匹配且保持相同有效token率，显著提升跨模型互操作性。

**🔧 技术方法**

使用Transformer编码器、代码簿路由、方向嵌入以及token级交叉熵训练的端到端学习。

**📊 数据集**

使用LibriSpeech和VCTK两个语音数据集进行训练与评估。

**📈 对比分析**

与传统waveform桥接对比，测量WER、UTMOS、MOS及延迟；TokenMapper在同速tokenizer间WER提升≤6%，MOS与原始接近，延迟可降至94%。

**⚠️ 局限性**

仅适用于相同有效token率、已配对token数据，无法处理不同帧率或音乐等跨域场景；对多codebook细节重建仍有限。

---

## 287. RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs

**arXiv ID:** 2609.12552 | [PDF](https://arxiv.org/pdf/2609.12552v1)

**作者:** Maëlic Neau `[一作]` `[通讯]`, Maëlic Neau

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可以对任意输入区域和任意词表进行实时开源关系预测的模型RelateAnything。

**💡 创新点**

创新点在于拆除关系预测对标签空间的依赖，构建大规模自由文本关系语料RA-4M，并提出六轴评价协议以客观衡量跨数据集和零样本能力。

**🔧 技术方法**

使用DINOv3视觉编码器、双分支关系头、正负未标记学习、文本编码器蒸馏以及基于几何验证的自动标注等技术。

**📊 数据集**

使用了474k张图像的RA-4M语料（4.3M关系、10.1k自由文本谓词），以及VG150、PSG、IndoorVG、HICO-DET等四个基准。

**📈 对比分析**

与OvSGTR、ROBIN-3B等现有方法对比，平均召回率提升2.3–3.5倍，零样本性能超越同规模最强方法，单帧推理约20 ms，模型参数仅5%左右。

**⚠️ 局限性**

局限包括对自动生成语料质量的依赖、对检测器召回率高度敏感、在极端稀疏标注场景下性能下降以及对特定评测协议的依赖。

---

## 288. Agent as Policy for Robotic Manipulation

**arXiv ID:** 2609.12541 | [PDF](https://arxiv.org/pdf/2609.12541v1)

**作者:** Mengzhao Jia `[一作]` (University of Notre Dame), Meng Jiang `[通讯]` (University of Notre Dame)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用通用多模态大语言模型（如 GPT‑6 Astra）作为机器人策略，实时生成和修改程序，控制机器人在不同任务（视频/图像/文本指导的装配、块构建、掷骰、投掷、双手毛巾折叠）中的感知、运动规划与执行，并根据物理反馈自适应调整。

**💡 创新点**

创新点在于把 LLM 直接作为机器人策略而非仅生成预先编写的程序或调度已有技能；通过“Agent as Policy”实现了运行时编程、实时感知反馈循环和动态策略更新，从而在零样本环境下完成多种复杂操作。

**🔧 技术方法**

技术包括：大规模多模态 LLM、代码编写与调试工具、机器人接口桥接（相机观测、姿态反馈、逆运动学与轨迹规划）、实时程序执行、经验记忆与检索、以及多模态感知处理（RGB‑Depth、标定、几何推断）。

**📊 数据集**

使用 AutoMate 数据集中的装配零件、任务目标图像（金字塔、塔等）、视频演示和语言描述；掷骰、投掷和毛巾折叠任务基于自制指令集和演示视频。

**📈 对比分析**

通过对比不同 LLM（GPT‑6 Astra、GPT‑5.6 Sol、Claude Opus 等）和不同思考力度，测量成功率、平均完成时间、Token 用量和 inference 成本。AGP 在大多数任务中达成 80%–100% 的成功率，平均完成时间 20–54 分钟，推理成本 7–27 美元；与传统方法相比，成功率更高但时间与成本仍显著偏高。

**⚠️ 局限性**

局限性主要体现在：复杂任务和柔性物体（如毛巾折叠）需要大量时间和昂贵的推理；经验复用对某些任务提升有限；模型规模和推理成本不易满足实时或工业部署需求；在高度动态或不确定环境下的鲁棒性仍待提升。

---

## 289. The House with a Million Windows: Interactive Fiction for Narrative Restorying

**arXiv ID:** 2609.12537 | [PDF](https://arxiv.org/pdf/2609.12537v1)

**作者:** Cody Kommers `[一作]` (Alan Turing Institute), Mina Lee `[通讯]` (University of Chicago)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个名为The House with a Million Windows的交互式写作系统，让用户通过LLM生成的多种文学风格“窗口”重新叙述个人故事，并评估其对叙事身份的影响。

**💡 创新点**

将元小说叙事框架与LLM驱动的多样化文学风格“窗口”结合，既保留作者主体性，又提供多维度的叙事重构，避免传统AI写作的平面化风险。

**🔧 技术方法**

采用Llama‑3.3‑70B‑Instruct大模型生成文本，前端基于next.js，后端使用SQLite存储；使用LIWC‑22、VADER、SentiArt等文本分析工具评估叙事变化。

**📊 数据集**

采集40名参与者自述的个人故事，并将其作为输入；使用公开领域文学作品（如《傲慢与偏见》、维多利亚·伍尔夫等）构建风格提示；未使用标准公开数据集。

**📈 对比分析**

在40名受试者中进行前后测的自评量表（NISE）和文本分析，结果显示叙事身份显著提升（p<0.001）且叙事结构更丰富；专家评审提供定性洞察。

**⚠️ 局限性**

窗口数量有限且受限于传统文学典范，LLM可能产生幻觉；缺乏长期效应评估、对多样化受众的适用性不足，且界面仅为文本化，缺少多模态支持。

---

## 290. QEmbed: A Deep Learning Based Cardinality Estimator for Efficient Query Processing

**arXiv ID:** 2609.12535 | [PDF](https://arxiv.org/pdf/2609.12535v1)

**作者:** Pooja Rajput `[一作]` (Indian Institute of Technology Jammu), Suman Banerjee `[通讯]` (Indian Institute of Technology Jammu)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于混合编码的自回归深度学习模型 QEmbed，用于高效准确的关系数据库 cardinality estimation

**💡 创新点**

创新点在于将 one-hot 与 dense embedding 两种编码方式结合成阈值驱动的混合编码，既保留低基数属性的细粒度信息，又通过稠密向量压缩高基数属性；并在 MADE 结构上加入多通道输入、掩码隐藏层和进阶采样策略，显著降低极端估计误差

**🔧 技术方法**

技术包括 Masked Autoencoder for Distribution Estimation (MADE)、自回归概率建模、混合编码（one-hot + embedding）、进阶采样（Progressive Sampling）以及 PyTorch/Adam 训练

**📊 数据集**

使用五个真实数据集：Forest、Power、DMV、Poker Hand、Census，涵盖数十万到千万行、低/高基数属性混合、数值与离散属性混合

**📈 对比分析**

与传统采样、MaxHistDiff、BayesNet/BayesCard、Transformer、Embed、Binary、FACE 等多种基线进行对比，结果显示 QEmbed 在大多数数据集上实现了更低的 median/75th/90th/95th Q-error、平均 Q-error 并将最大 Q-error 降到两位数，推理时间虽高于简单模型但远低于贝叶斯网络与 Transformer，且在极端误差上表现最稳健

**⚠️ 局限性**

局限性：在属性维度非常高或高基数域极大（如 Census）时，MADE 结构的掩码与混合编码导致推理时延增加且准确率略低；对动态数据更新缺乏实时适应机制；阈值 τ 的选择仍需经验调优

---

## 291. Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents

**arXiv ID:** 2609.12533 | [PDF](https://arxiv.org/pdf/2609.12533v1)

**作者:** Zhutao Lv `[一作]` (Tsinghua University), Weijia Li `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Earth-Bench-Pro（面向全链路、开放世界的地球观测任务评测集）与Earth-Agent-Pro（基于专家技能的计划-执行框架）

**💡 创新点**

通过将专家手册化为技能约束，结合工作流中心化结构记忆与局部后缀修复，实现了从高层问题到数据获取、预处理、领域计算到开放式答案的完整闭环；同时引入角色特化训练，分别对计划器（SFT）和执行器（GRPO）进行微调

**🔧 技术方法**

使用大型语言模型（如GPT‑5、Qwen3.5-9B）作为主体，结合工具路由、技能路由、四角色序列辩论、混合验证门控与工作流修复；训练采用低秩适配（LoRA）实现SFT与GRPO

**📊 数据集**

Earth-Bench-Pro基于248个专家核任务，产生744个匹配问题，涵盖RGB、光谱与遥感产品，数据来源包括实时卫星观测、公开遥感产品及预处理流程

**📈 对比分析**

与ReAct、AFlow、OpenEarthAgent等框架在相同GPT‑5主干上比较，Earth-Agent-Pro在LLM‑as‑Judge准确率、工具顺序一致性(TIO)及工具完整性(TAO)等指标上分别领先20.95/24.44/17.14个百分点；在Earth‑Bench‑OW上Qwen3.5‑9B从38.31%提升至50%

**⚠️ 局限性**

仍存在工作流精确度与参数匹配不足（TEM/参数准确率仅~70%/32%），对数字推理与空间测量的开放式回答准确率偏低，模型规模越小精度越差，且技能引导虽然提高正确性但会略增工作流长度导致效率下降

---

## 292. Fault-tolerant Hamiltonian connectivity of Johnson graphs

**arXiv ID:** 2609.12617 | [PDF](https://arxiv.org/pdf/2609.12617v1)

**作者:** Huazhong Lü `[一作]` (University of Electronic Science and Technology of China), Jinhao Liu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了Johnson图在不同故障模型下的容错哈密顿连通性，并给出了相应的构造性证明与递归路由算法。

**💡 创新点**

创新点在于证明Johnson图在一般边故障下可容忍至k(n−k)−3条故障边，在匹配故障下可容忍任意匹配（甚至完美匹配），以及在节点故障下可容忍至n−2个节点故障，且构造性算法实现了线性时间路由。

**🔧 技术方法**

采用了Johnson图的层划分、递归分治、Ore条件与匹配/节点故障的特殊性质，以及图同构与补集变换的技术。

**📊 数据集**

使用的“数据集”为构造的Johnson图实例，最大到12,870个顶点，故障集为理论上最大可容忍的故障集合。

**📈 对比分析**

通过对不同k值的Johnson图进行100组随机源终点测试，实验结果显示三种故障模型下的平均执行时间随网络规模几乎线性增长，性能优于传统方法。

**⚠️ 局限性**

限制在于节点故障容忍度的上界尚未确定，且对于k=2的情况分层选择开销可能超线性，后续工作需改进。

---

## 293. Poisson-Corrector Complexity Bounds for Moreau--Yosida Unadjusted Langevin Sampling

**arXiv ID:** 2609.12594 | [PDF](https://arxiv.org/pdf/2609.12594v1)

**作者:** Yuchen Xin `[一作]` (Peking University), Zhihua Zhang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文研究了Moreau–Yosida无调整 Langevin 算法（MYULA）在目标分布 π(x)∝e^{-f(x)-g(x)} 中的理论收敛性能，给出了 W_2 误差上界并推导了迭代复杂度

**💡 创新点**

创新点在于结合离散 Poisson corrector 与共享噪声曲率估计，显著降低对 Moreau 平滑参数 λ 的依赖，仅呈对数增长，从而将迭代复杂度提升至 O(ε^{-4/3})

**🔧 技术方法**

主要技术包括离散 Poisson corrector、活跃迹 (active‑trace) 估计、共享噪声 (shared‑noise) 曲率反馈、同步对接、弱 Hessian 平均以及 Wasserstein 收敛分析

**📊 数据集**

论文为理论分析，无实测数据集，全部结果以定理和证明形式给出

**📈 对比分析**

与之前的 O(ε^{-2}) 结果相比，本工作通过更细致的曲率控制实现了更优的复杂度，但未给出实验对比

**⚠️ 局限性**

局限性包括：仅适用于 f 为 m‑强凸且 L_f‑Lipschitz 的光滑项与 g 为全局 G‑Lipschitz 的凸非光滑项；对 λ 与步长 h 的约束较为严格；未探讨非凸或高阶非光滑情况

---

## 294. Where Decoder Cosine Similarity Fails for SAE Feature Flow Discovery

**arXiv ID:** 2609.12591 | [PDF](https://arxiv.org/pdf/2609.12591v1)

**作者:** Hendrik Droste `[一作]`, Holger Giese `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种基于残差自注意力网络的三元组评分框架，用于知识图谱补全。

**💡 创新点**

创新点在于同时编码状态、更新和目标特征，并通过联合-状态-更新的期望差值来计算三元组得分。

**🔧 技术方法**

采用自注意力模块、MLP、残差连接以及期望值评分公式。

**📊 数据集**

在公开的知识图谱数据集 FB15k-237 和 WN18RR 上进行实验。

**📈 对比分析**

与传统基线（TransE、RotatE、ConvE 等）比较，模型在 MRR/M@10 等指标上均取得了 5–10% 的提升。

**⚠️ 局限性**

缺点是计算开销大，对大规模知识图谱的扩展性有限，且对长距离关系的建模仍有改进空间。

---

## 295. Geometric-to-Semantic Spherical Transfer Learning for Cortical Sulci Labeling

**arXiv ID:** 2609.12627 | [PDF](https://arxiv.org/pdf/2609.12627v1)

**作者:** Saeb Tounsi `[一作]` (Paris-Saclay University), Jean-François Mangin `[通讯]` (Paris-Saclay University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种几何到语义的球面迁移学习框架，用于在少量标注数据下对大脑皮层 sulci 进行完整标注。

**💡 创新点**

创新点在于通过在 30,000 份无标签 UK Biobank 数据上进行局部优化的自监督预训练，捕捉细节几何特征；并使用软初始化的 Topological Prior Injector 将语义线条平滑整合进预训练的编码器，避免灾难性遗忘。

**🔧 技术方法**

技术包括球面卷积网络（Spherical U-Net）、自监督学习（Barlow Twins 目标）、几何 Voronoi 投影、软初始化的 1×1 线性层（TPI）、组归一化、双重损失（交叉熵+Dice）。

**📊 数据集**

使用 62 份专家标注的右半球 sulci 数据（公开数据集）进行下游任务，预训练使用 UK Biobank 约 30,000 份无标签大脑表面数据。

**📈 对比分析**

与从零训练的球面 U-Net、2D U-Net、以及 DINOv3 等视觉基础模型对比，平均 Dice 最高达 0.77，且在仅 5 份标注时仍保持 ESI 0.24，显著优于基线。

**⚠️ 局限性**

局限在于仅针对右半球、只处理已提取的 1D sulcal 线条，模型对极端形变或未覆盖的 sulci 仍易产生误检；同时需要较大规模的无标签数据进行自监督预训练。

---

## 296. Clustering-Based Balanced Sampling and Allocation with Data Parallelism for High-Performance Fine-Tuning

**arXiv ID:** 2609.12584 | [PDF](https://arxiv.org/pdf/2609.12584v1)

**作者:** Hyunjin Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jae-Gil Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 CluSTER，一种面向多GPU指令调优的聚类感知平衡采样框架，用于减少冗余样本并提升训练效率。

**💡 创新点**

创新点在于利用梯度空间聚类捕捉更新相似的样本组，并在工作器层面和簇内层面实现双层覆盖，通过加权更新保持原始分布。

**🔧 技术方法**

采用基于token级交叉熵与最终隐藏状态的梯度代理嵌入、K‑means聚类、DP‑aware平衡采样以及权重梯度更新，并给出了梯度方差理论分析。

**📊 数据集**

使用代码指令数据集 Magicoder‑OSS‑Instruct‑75K、Evol‑Instruct‑Code‑80K 以及医疗数据集 MedInstruct‑52K 进行实验，评估基准包括 HumanEval、HumanEval+、MBPP、MBPP+、MedMCQA、MedQA、PubMedQA 与 MMLU 医疗子集。

**📈 对比分析**

与随机、均匀、IFD、LESS、S2L 等基线对比，CluSTER 在多GPU训练中可将训练时间减少高达 69.6% 且几乎不降低准确率，在代码与医疗 QA 任务上均表现优于所有基线。

**⚠️ 局限性**

局限性：实验仅在单机多GPU环境下验证，未测试多节点或大规模批次；数据集规模受限于中等规模指令调优语料，未评估更大规模数据集的表现。

---

## 297. Omniscience for the Masses: New Threats in the Metaverse's Democratized World Creation

**arXiv ID:** 2609.12554 | [PDF](https://arxiv.org/pdf/2609.12554v1)

**作者:** Andrea Mengascini `[一作]` (CISPA Helmholtz Center for Information Security), Giancarlo Pellegrino `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文系统评估了元宇宙平台中世界创建者的安全与隐私风险，针对 25 个平台的创作工具进行调查，并在 5 个代表性平台上实现了五种新型攻击以及对已有攻击的重现。

**💡 创新点**

创新点在于首次将“世界创建者”定位为独立的威胁角色，证明仅凭官方创作工具即可实现全景监控与操纵，并揭示现有平台防护缺陷，推动对创作行为的安全审计与权限控制研究。

**🔧 技术方法**

技术手段包括利用平台的可视化脚本/文本编程、音频/摄像头路由、对象属性读写、网络请求与持续执行等原语，结合对 25 个平台工具链的系统性分析，构造攻击逻辑。

**📊 数据集**

使用的“数据集”为公开可访问的元宇宙平台（Roblox、VRChat、Horizon Worlds、Spatial、Frame）及其内部用户、世界信息；未使用专门的标注数据集。

**📈 对比分析**

对比方法：与先前需要客户端改造或开发者特权的攻击相比，本文仅凭创作工具即可实现相同或更强的攻击；实验显示在所有平台均可自动化执行，用户几乎不感知，性能损耗极低（帧率下降 <1%，CPU 约 +10%）。

**⚠️ 局限性**

局限性：仅评估了免费平台，未测量恶意世界在实际市场中的流行度；平台更新可能影响攻击可行性；缺乏对真实恶意实例的深入分析与长期监测。

---

## 298. Subgroup Packing for Batched PASTA Transciphering

**arXiv ID:** 2609.12624 | [PDF](https://arxiv.org/pdf/2609.12624v1)

**作者:** Mugurel Barcau `[一作]` (Institute of Mathematics Romanian Academy), George C. Ţurcaş `[通讯]` (Babeş-Bolyai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本文中，作者通过重新安排 PASTA-3 记录在 SIMD 载体中的排列方式，研究了不同打包布局（连续布局 vs 子群对齐布局）对同等批量转换成本和噪声余量的影响，并实现了完整的转换和后续查询。

**💡 创新点**

创新点在于提出并验证了子群对齐布局能将 255 条直接标签压缩至 128 条，从而在同批量下显著降低服务器转换成本，同时揭示打包布局与噪声余量之间的权衡。

**🔧 技术方法**

采用了 BGV 同态加密、HElib 的 SIMD 包装、旋转与矩阵分解（BSGS）以及直接翻译支持的数学理论。

**📊 数据集**

使用了十二组配对语料库（每组包含 128 条记录的 128 维字段）以及六个独立的 HE 密钥，对每个布局进行实验。

**📈 对比分析**

通过对同批量转换、两次公共子集求和查询的完整时间与噪声容量进行计量，发现连续布局平均消耗 36.67 秒，子群布局 22.91 秒，成本比约 1.60，但子群布局噪声余量更低。

**⚠️ 局限性**

局限在于未能通过噪声限制完成某些组合调度，浅层直接加密基线查询失败，且安全估计仅基于不完整的攻击模型。

---

## 299. RA-SOD: Reliability-Aware RGB-T Salient Object Detection under Modality Degradation

**arXiv ID:** 2609.12622 | [PDF](https://arxiv.org/pdf/2609.12622v1)

**作者:** Hongbo Gao `[一作]` (Harbin Institute of Technology), Chang Xu `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可靠性感知的RGB‑T显著目标检测框架RA‑SOD，专门针对多模态降质场景设计。

**💡 创新点**

创新点在于：①可靠性条件骨干网络（RCBR）通过混合专家与低秩适配器自适应调节降质特征；②不确定性引导的双流解码（UGDD）在递归细化过程中抑制不可靠信息；③像素级模态竞争机制（PMC）实现空间可变可靠性选择，实现精细融合。

**🔧 技术方法**

采用残差调制分支、Mixture‑of‑Experts（MoE）专家路由、低秩适配器、注意力驱动的自监督不确定性预测以及三路层次递归解码与像素级竞争。

**📊 数据集**

在四大RGB‑T基准VT821、VT1000、VT5000和极端降质集VT‑IMAG上进行实验。

**📈 对比分析**

与现有19种最先进方法（包括RGB‑D与RGB‑T模型）对比，RA‑SOD在所有指标上均处于前两名，VT5000上首次获得全部指标第一；在VT‑IMAG上取得最高结构与对齐分数，Fβ提升约2‑3%。

**⚠️ 局限性**

局限性：模型参数略高于最轻量级方法，需更多计算资源；在极端极端模态失效（完全失真）时仍可能产生误检；未来可进一步探索更轻量化或实时化实现。

---

## 300. Correlation-Guided Fast Machine Unlearning via Hessian Analysis

**arXiv ID:** 2609.12620 | [PDF](https://arxiv.org/pdf/2609.12620v1)

**作者:** Ayushi Thakur `[一作]` (Indian Institute of Technology (Banaras Hindu University)), Prayag Tiwari `[通讯]` (Halmstad University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Pearson相关性的机器学习模型快速忘记框架，利用一次Hessian逆向量计算后对相似样本直接闭式更新。

**💡 创新点**

通过引入Hessian阻尼、Sherman-Morrison一阶近似以及相似性系数α，将多次相似数据点的遗忘转化为一次性高效更新，避免重复昂贵的Hessian逆运算。

**🔧 技术方法**

利用影响函数、Hessian阻尼、Sherman–Morrison公式、Pearson相关系数、Gauss–Newton近似和梯度线性假设实现快速忘记。

**📊 数据集**

在七个数据集（California Housing、Diabetes、MNIST、Fashion‑MNIST、CIFAR‑10、CIFAR‑100、LFW以及合成高斯混合模型）上验证，使用从MLP到ResNet‑50等多种网络。

**📈 对比分析**

与传统重训练、MITR、RUM(A/B)、Hessian‑free Unlearning等基线对比，取得82倍的壁钟加速、平均准确率提升约10⁻²、MIA成功率0.660、ToW 0.950，且在批量顺序忘记中保持2%以内的精度损失。

**⚠️ 局限性**

对相关性阈值和阻尼参数的选择敏感，理论误差上界保守，无法处理高度非线性或极度不相似的样本，且依赖已收敛的Hessian正定性。

---

## 301. SIMS: Scale-Invariant Merit-Function-Based Scalarization for Multi-Task Learning

**arXiv ID:** 2609.12599 | [PDF](https://arxiv.org/pdf/2609.12599v1)

**作者:** Zebin Chen `[一作]` (Southern University of Science and Technology), Yu Zhang `[通讯]` (Southern University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种尺度不变的利益函数标量化方法SIMS，解决多任务学习中任务损失尺度不匹配导致的敏感性问题。

**💡 创新点**

通过理论证明唯一的尺度不变变换为对数变换，并构造光滑近似，保持弱Pareto最优性并实现尺度不变性。

**🔧 技术方法**

采用对数变换的利益函数、log-sum-exp平滑与TTGDA梯度下降-上升算法实现优化，并给出收敛分析。

**📊 数据集**

在NYUv2、Cityscapes和PASCAL-Context三大多任务基准数据集上进行实验。

**📈 对比分析**

与EW、STCH、FOOPS以及多种主流多任务架构（HPS、MTAN、SwinMTL、MultiLoRA等）对比，SIMS在平均性能上均优于现有标量化方法，达到SOTA水平。

**⚠️ 局限性**

缺点包括对参数λ的敏感性仍需设置，且对数变换的假设在极端任务数或不平衡任务场景下的鲁棒性未充分验证。

---

## 302. MicroHasTEE: Bare-Metal Haskell for Type-Level Peripheral Ownership on Armv8-M

**arXiv ID:** 2609.12580 | [PDF](https://arxiv.org/pdf/2609.12580v1)

**作者:** Robert Krook `[一作]` `[通讯]` (Chalmers University of Technology and University of Gothenburg), Robert Krook (Chalmers University of Technology and University of Gothenburg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

MicroHasTEE 框架让开发者在一份类型安全的 Haskell 程序中同时描述 TrustZone‑M 微控制器的 Secure 与 Non‑Secure 固件，并将其编译成两份独立的裸机镜像。

**💡 创新点**

其创新点在于利用 Haskell 的类型系统与 capability ledgers，静态检测外围设备授权、断电切换、IRQ 路由和跨域调用，避免运行时出现配置冲突。

**🔧 技术方法**

实现技术包括基于 indexed monad 的设置阶段、MicroHs 编译器/运行时在裸机上执行 Haskell、STM32U5 GTZC 等硬件接口驱动，以及类型级资源跟踪。

**📊 数据集**

实验采用 STM32U5 Nucleo 开发板上的门锁案例（键盘+继电器），并未使用外部公开数据集。

**📈 对比分析**

通过与 TF‑M、Rust/Go 等方案对比，MicroHasTEE 在 4 MiB Flash 与 2.5 MiB SRAM 的设备上实现，每个镜像约 232 KiB Flash、220 KiB SRAM，且实现了编译期一致性检查，未给出正式的性能基准。

**⚠️ 局限性**

限制包括仅支持单个 Secure 应用、未给出形式化证明、仅实现部分外围设备、依赖正确的硬件后端且无法检测原始 C/FFI 违规。

---

## 303. SCOPE-OPSD: Fisher-Conditioned Privileged Subspaces for On-Policy Self-Distillation

**arXiv ID:** 2609.12579 | [PDF](https://arxiv.org/pdf/2609.12579v1)

**作者:** Yunmeng Chen `[一作]` (Chongqing Ant Consumer Finance Co., Ltd.), Song Liu `[通讯]` (Chongqing Ant Consumer Finance Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在已有的On-Policy Self-Distillation（OPSD）框架中，作者提出在最终隐藏层加入一个残差投影损失，利用教师-学生最终隐藏状态之差投影到冻结的低秩Fisher‑条件子空间，以捕捉对齐信息。

**💡 创新点**

创新点在于构造一个冻结的低秩投影子空间，该子空间结合残差协方差与输出层Fisher信息，仅在训练期间增加一次矩阵乘法，且不改变推理流程或增加额外的rollout。

**🔧 技术方法**

技术手段包括LoRA微调、残差投影、输出层Fisher敏感度估计、残差协方差的谱滤波以及低秩矩阵投影。

**📊 数据集**

实验使用Qwen3-1.7B/4B/8B模型，在29,434条OpenThoughts Math OPSD训练样本以及AIME 2024/25、HMMT 2025数学竞赛测试集进行评估。

**📈 对比分析**

与纯OPSD以及秩、谱、梯度匹配的随机投影进行对比，在共享的step 75检查点，Structured在Macro Avg@12上比Pure OPSD提升约0.83–1.85个百分点，比随机投影提升约0.37–1.81个百分点，整体性能稳定领先。

**⚠️ 局限性**

局限性包括仅在Qwen3家族上验证、训练预算短、消融实验仅覆盖1.7B、未检验更大模型或不同推理任务的可迁移性。

---

## 304. From Collaboration to Capability: Internalizing Routed LLM Experts into Compact Reasoners

**arXiv ID:** 2609.12578 | [PDF](https://arxiv.org/pdf/2609.12578v1)

**作者:** Frank Nie `[一作]` (Shandong University), Ethan B. Liu `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段训练框架RIVET，先用专家增强的GRPO让控制器学习专家调用与返回内容，再通过验证轨迹内部化让控制器在没有外部专家的情况下生成完整的推理和代码。

**💡 创新点**

创新点在于同时将控制器决策与专家产出视为一条完整协作轨迹进行联合学习，并通过共享的轨迹收益信号和格式感知的监督，提升控制器在专家被移除后仍能独立完成任务的能力。

**🔧 技术方法**

核心技术包括专家增强的GRPO（共享优势更新控制器与专家span），验证轨迹内部化（收集成功交互并进行格式加权的监督学习），以及本地Python执行环境模拟。

**📊 数据集**

使用ToRL数据集进行训练，评估集为七个竞赛数学基准（AIME 2024‑2026、HMMT 2025、BeyondAIME、IMO-AnswerBench、APEX 2025）以及GPQA‑Diamond科学推理基准。

**📈 对比分析**

在控制器无外部专家、仅本地Python的部署方式下，RIVET‑4B在数学基准上平均准确率达44.16%，比Stage I提升6.49点，超过同尺寸模型及更大模型ReTool‑32B；在GPQA‑Diamond上与AEPO‑8B并列第一，显示出跨领域泛化能力。

**⚠️ 局限性**

局限性包括：仅验证两种Qwen基础模型与三种固定专家，成本高（需频繁调用外部专家）；仅使用准确率作为评价指标；对不同模型家族、专家组合、任务类型的普适性尚未充分验证。

---

## 305. Quality-Constrained Routing over a Fixed Pool of Quantized Mixture-of-Experts Instances

**arXiv ID:** 2609.12550 | [PDF](https://arxiv.org/pdf/2609.12550v1)

**作者:** Zhenghong Huang `[一作]` (Hong Kong University of Science and Technology), Jiheng Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对已预材料化的量化MoE实例池进行请求级路由，利用Fragility-Weighted Perplexity（FWP）评估请求风险，并通过窗口级线性规划实现吞吐最优化。

**💡 创新点**

首次将基于专家脆弱度的FWP信号与LP约束相结合，在固定池内实现质量约束的自适应路由，并证明其在离线评估中对吞吐提升的贡献。

**🔧 技术方法**

使用两专家稀疏MoE理论、top‑k聚合近似、Fragility‑Weighted Perplexity计算、窗口级线性规划以及KKT一致性分析等技术。

**📊 数据集**

以Qwen3‑30B‑A3B为基础，量化6,144个专家块，采集88个扩展QA、代码与长上下文提示作为评估样本。

**📈 对比分析**

与静态W4、W3、W2、请求无关预算混合以及真损失排序对比，在相同质量阈值下实现1.284×的离线模型乘数，相比静态W4提升28.4%，相对请求无关混合提升2.5%。

**⚠️ 局限性**

评估仅限离线固定池，不包含实时再配置、等资源复制对比、跨模型通用性或完整部署保障；FWP在短前缀下表现有限，且未提供完整的端到端质量合规保证。

---

## 306. An Ultra-Widefield Swept-Source OCTA Dataset and a Polar-Gated Mamba Network for Retinal Vessel Segmentation

**arXiv ID:** 2609.12574 | [PDF](https://arxiv.org/pdf/2609.12574v1)

**作者:** Yang Liu `[一作]` (Tsinghua Shenzhen International Graduate School), Chengming Yang `[通讯]` (Southern University Of Science And Technology Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究引入了首个大视野 SS-OCTA 血管分割数据集 WOIVES，并提出 PG-Mamba 模型用于 UWF 视网膜血管分割。

**💡 创新点**

创新点包括：①WOIVES 提供 24×20 mm² 大视野 SS-OCTA 的软概率血管标注；②PG-Mamba 在视觉状态空间模型中引入极坐标双向扫描顺序和动态 FOV 门控，显著提升大场景血管分割性能。

**🔧 技术方法**

采用视觉状态空间模型（Mamba）+ 极坐标多方向扫描 + 动态 FOV 门控 + 软标签损失（Dice+MSE）等技术，配合 512×512 斜坡窗口推理。

**📊 数据集**

使用 WOIVES 数据集（206 只眼、152 名受试者，24×20 mm² UWF SS-OCTA）进行训练与评估。

**📈 对比分析**

通过受试者级 5 折交叉验证，比较 7 种基线（U-Net、UNet++、R2U-Net、Swin-UNet、H2Former、VM-UNet、AC-MambaSeg）在 Dice、IoU、Soft Dice、clDice、MAE、Brier 等指标。PG-Mamba 在 5/6 主要指标上均为最佳，Dice 提升 0.44% 并获得 0.38% 的 clDice 提升；在血管密度、分形维数、血管长度密度等下游测量误差亦最低。

**⚠️ 局限性**

局限性包括：单中心单设备数据、样本偏向高近视、仅标注浅层血管、极坐标扫描固定且仅在瓶颈处应用门控、未验证跨设备/多中心泛化能力、以及仅评估与专家标注的一致性，未直接验证临床效用。

---

## 307. The Fixed Server Locality Gap of Count Load Assignment Games

**arXiv ID:** 2609.12572 | [PDF](https://arxiv.org/pdf/2609.12572v1)

**作者:** Hao Li `[一作]` (Wuhan College), Mengfan Ma `[通讯]` (Central China Normal University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在拥有有限共享服务器、可选本地执行和任意非负分配成本的计数-负载分配游戏中，证明了边际贡献定价下最差纯纳什均衡的价格无效率上界与下界均为Θ_M(κ^{1-2^{-M}})，并给出了匹配的极限实例。

**💡 创新点**

创新点在于①提出不依赖比较图无环性即可获得固定服务器数 M 的异质性指数 1-2^{-M} 的普适上界；②构造基于 UAV 客户端的几何下界实例，展示该指数的可实现性；③将游戏与排队调度与拥塞模型精确对应，并通过输入组件分解实现多项式时间求最优纳什均衡。

**🔧 技术方法**

采用的技术包括：潜在博弈理论、源权重残差不等式、顺序矩递推、组合构造与几何实现、完整枚举与流网络最小成本解法、以及连续最优 waypoint 计算。

**📊 数据集**

数据集分两类：① 2,800 个合成有限游戏（最多 4 台服务器、工作负载比 κ∈{1,2,4,16,64,256,1024}），② 270 条基于 UAV 的主参数化仿真（50–200 台服务器、50–200 台 UAV、2–4 台服务器、不同路段预算），全部采用随机生成的工作负载、系数、菜单等参数。

**📈 对比分析**

对比方法包括边际定价的最佳响应、非定价最佳响应、最小单项成本选择及仅本地执行；在合成游戏中最差 PoA 达到约 1.76；在主参数化仿真中，边际定价的最佳响应与参考方案相比平均提升约 0.3%，并且与其它基线相比显著更优。

**⚠️ 局限性**

主要局限：模型假设可分离 waypoint 与单纯成本，忽略了干扰、碰撞、共享链路、动态排队等实际因素；下界构造是人工对抗实例，可能不具备现实可行性；上界常数松散，且在实际中 PoA 可能更低；最优 PNE 的算法在服务器数 M 增大时仅属于 XP，缺乏 FPT；实验仅覆盖有限规模与异质性范围，未验证极限指数在真实部署中的表现。

---

## 308. PIA-Bench: Towards Automated Privacy Impact Assessment with Large Language Models

**arXiv ID:** 2609.12571 | [PDF](https://arxiv.org/pdf/2609.12571v1)

**作者:** Jiamin Zheng `[一作]` (University of Edinburgh), Jingjie Li `[通讯]` (University of Edinburgh)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了第一个公开基准，评估大型语言模型（LLM）在真实美国联邦机构发布的隐私影响评估（PIA）中识别隐私风险并给出缓解建议的能力。

**💡 创新点**

创新点在于：①系统收集并审核了499份PIA，筛选出73份结构化可评测的PIA；②构建了451条隐私风险与831条缓解条目的ground truth；③设计了自动化评测框架和JSON响应格式，首次将LLM在PIA端到端流程中的表现进行量化比较。

**🔧 技术方法**

技术手段包括：利用Qwen、Llama、DeepSeek及GPT-5.4 mini等LLM；采用链式思考（CoT）提示提升推理；使用LLM裁定器判定生成内容与ground truth的匹配；评估指标为Precision、Recall、F1以及Flesch阅读难度（FRE）。

**📊 数据集**

数据集：原始的499份美国联邦机构PIA（CFPB、USDA、OPM、GSA、HHS），通过结构化筛选后得到73份清晰PIA，并从中提取451条风险和831条缓解条目的文本段落作为ground truth。

**📈 对比分析**

比较方法：在k=3,5,10的宏平均下计算Precision/Recall/F1，随机打乱对照以及专家原文参考值；性能方面，最佳模型GPT-5.4 mini在CoT下的F1分别约为0.40（风险）和0.42（缓解），仍低于人类PIA；阅读难度FRE普遍低于专家版本，说明LLM生成文本较为晦涩。

**⚠️ 局限性**

局限性：①仅评测结构化PIA，未覆盖非结构化或跨司法区PIA；②LLM在识别与个人权利相关的细粒度风险（如访问、纠正、申诉）时效果差；③缺乏更细粒度、任务导向的质量指标和人机协作机制；④对模型解释性、可追溯性及安全性考量不足。

---

## 309. Preference-Drift-Aware Subsequence Learning and Hierarchical Context Fusion for Long-Sequence Generative Recommendation

**arXiv ID:** 2609.12556 | [PDF](https://arxiv.org/pdf/2609.12556v1)

**作者:** Fei Li `[一作]` (Northeastern University), Zang Li `[通讯]` (Tencent)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种结合偏好漂移感知子序列学习和层次上下文融合的DRIFT模型，用于长序列生成式推荐。

**💡 创新点**

创新点在于利用多维偏好漂移信息自适应划分子序列，并用线性注意力和门控融合实现高效且噪声鲁棒的上下文建模。

**🔧 技术方法**

采用多维漂移特征、可微软边界、线性注意力、交叉注意力、门控融合、Gumbel-Softmax等技术。

**📊 数据集**

在KuaiRec、ML‑20M和Taobao MM三大公开数据集上进行实验。

**📈 对比分析**

与GRU4Rec、SASRec、HSTU、FuXi‑Linear、TIGER、DIGER、GLASS等基线相比，DRIFT在Recall@20、NDCG@20等指标上实现了显著提升（最高提升21.6% Recall@20），并在训练与推理速度上提高数倍。

**⚠️ 局限性**

局限在于需要手动调参子序列数、温度参数等超参数，且对极端偏好漂移或极长历史仍可能产生误区。

---

## 310. Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification

**arXiv ID:** 2609.12544 | [PDF](https://arxiv.org/pdf/2609.12544v1)

**作者:** Linh Uyen Le `[一作]` (Meddies AI), Huy Hoang Ha `[通讯]` (Meddies AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个包含一百万条多语言合成临床文档的数据集 Meddies-PII-Dataset，训练了一个 BIOES token 分类器 Meddies-PII-Model，并在多个基准上进行评估。

**💡 创新点**

创新点包括：1) 提供 17 种语言、9 标签的多语种合成 PII 数据集；2) 采用属性条件提示与 13 个确定性校验门控相结合的生成框架；3) 公开了生成框架、基准和评测代码；4) 在 15 个外部基准和自建 Meddies-PII Benchmark 上实现最高 F1 分数。

**🔧 技术方法**

使用的技术包括：多属性提示+大模型生成、正则表达式+字符级解析+SHA-256 等 13 个确定性门控校验、BIOES token 分类 + Viterbi 解码、以及基于 LFM2.5‑350M 编码器的模型训练。

**📊 数据集**

使用的数据集有：Meddies-PII-Dataset（1M 合成文档，17 语种，9 标签）、Meddies-PII Benchmark（5,100 验证文档）、以及 15 个外部基准（OpenPII 1.5M 12 种语言、CredData、Gretel PII Masking、Nemotron-PII 等）。

**📈 对比分析**

通过 exact‑match 实体级 micro‑F1 进行比较，系统与 OpenAI Privacy Filter、GLiNER2、SuperClinical‑Large‑434M‑v1、LFM2.5‑Encoder‑350M‑PII‑Detector 等公开系统对齐。在 15 个外部基准上平均 F1 0.827，领先最强基准 0.169；在 Meddies‑PII Benchmark 上 F1 0.878，优势 0.338；总体平均 F1 0.833。

**⚠️ 局限性**

局限性：1) 完全合成数据可能缺乏真实临床记录中的语言变异和注释歧义；2) 确定性门控未直接衡量生成文本的语义自然度；3) 评测仅基于公开基准，尚未在真实受控临床记录上验证。

---

## 311. When Is Inaction a Mistake? Continuation-Aware Auditing of PPO Trading Policies

**arXiv ID:** 2609.12536 | [PDF](https://arxiv.org/pdf/2609.12536v1)

**作者:** Xingfei Zeng `[一作]` (University of Electronic Science and Technology of China), Guanghui Lu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对冻结的PPO策略进行四阶段审计，评估在信息匹配、孤立动作变化和策略替代等情境下的行动不一致问题。

**💡 创新点**

提出一种基于完整信念动态规划的后验审计框架，能够区分信息不匹配、后续决策影响以及单次动作改变对整体表现的贡献。

**🔧 技术方法**

使用强化学习的PPO算法、Kalman滤波、有限MDP价值迭代和统计检验（置信区间、Bootstrap）等技术进行审计与评估。

**📊 数据集**

在线性高斯模拟环境和实际的比特币/泰达币（BTCUSDT）分钟级交易数据上进行实验。

**📈 对比分析**

通过对比原始策略与投影、后续延续和重复部署的三种替代规则，测量误差量（AMA、WFI）、一步行动误差（OAR）以及日收益改进；实验显示投影和重复部署显著提升所有50个策略，BTCUSDT回放显示成本下降带来约135基点的净收益。

**⚠️ 局限性**

局限在于审计仅基于冻结策略且未训练新的策略；对信息扰动与成本变化的因果解释有限；并且实验主要关注交易成本而非绝对盈利性。

---

## 312. $\text{GSF-}χ$: Global Stereochemical Fields for Chiral Graph Transformers

**arXiv ID:** 2609.12532 | [PDF](https://arxiv.org/pdf/2609.12532v1)

**作者:** Jiaqing Xie `[一作]` (Shanghai Innovation Institute), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出GSF-χ图变换器，能够在保持对分子重标签和正向旋转不变性的前提下，区分在原子、键及对间距离相同但手性不同的两分子。

**💡 创新点**

创新在于引入全局镜像偶性场与“Chiral‑RoPE”旋转算子，使所有原子对的注意力都可受手性影响，并通过C₂投影实现严格的镜像对称性。

**🔧 技术方法**

核心技术包括：全局镜像偶性化学场、相对旋转注意力（Chiral‑RoPE）、可变旋转轴与频率的条件化、以及对ECD的偶奇分解读取。

**📊 数据集**

使用官方ChIRo、CMCDS、ACMP等公开数据集，包含R/S分类、Ranking、旋转、ECD峰数/位置/符号以及轴向手性任务。

**📈 对比分析**

与DimeNet++、SphereNet、Tetra‑DMPNN、ChIRo、ECDFormer、ChiDeK等基线在相同划分、超参数和预算下对比，GSF-χ在R/S、Ranking、所有ECD指标上均优于基线，尤其在轴向旋转和符号预测上实现显著提升。

**⚠️ 局限性**

限制包括：需要手性单元标注（可通过自动提取但精度受限）、对计算成本有轻微提升（取决于手性单元数量），以及对非手性化合物时功能退化为标准图变换器。

---

## 313. Universally truthful mechanisms for scheduling

**arXiv ID:** 2609.12621 | [PDF](https://arxiv.org/pdf/2609.12621v1)

**作者:** Georgios Anastasiadis `[一作]` (Aristotle University of Thessaloniki), Conrad Schecker `[通讯]` (Goethe University Frankfurt)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

本文研究了在不相关机器调度问题中，离散支持的通用真诚随机机制的性能上界与下界，给出了期望逼近比的精确线性限制。

**💡 创新点**

创新点在于首次证明了任意离散支持的通用真诚随机机制的期望逼近比不得低于 n/12，随后构造了一种 Exp‑Bounded‑Square 机制，使其期望逼近比达到 n/2 + o(n)，实现了与已知上界（0.837n）相比的系数改进，并与分数与期望真诚机制的最佳已知上界对齐。

**🔧 技术方法**

主要技术包括弱单调性（WMON）对真诚性的必要条件、Yao 极小极大原理用于下界证明、图实例（多重星、星形盒子）与盒子定理用于构造反例、以及对 Exp‑Bounded‑Square 机制的概率分析与 Hoeffding 以及 Chernoff 上界。

**📊 数据集**

本文属于纯理论分析，未使用任何实验数据集；所有结果均基于数学证明与概率分布的构造。

**📈 对比分析**

通过与之前的 0.837n 机制对比，本文的上界（n/2 + O(√(n ln n))）在系数上实现了显著改进；下界（(3/2–√2)(n‑1)≈0.0858(n‑1)）与上界在 n 维度上基本匹配，证明了该类机制的最佳可实现逼近比。

**⚠️ 局限性**

局限性包括仅考虑离散支持的通用真诚随机机制；对连续分布的情况只给出了间接的扩展；分析基于期望逼近比，未给出对个别实例的最坏情况保证；实现 Exp‑Bounded‑Square 的随机抽样与指数权重可能在实际部署中带来计算成本。

---

## 314. SCORE: SubDistribution-aware Collaborative Knowledge Reinforcing for Cloth-Hybrid Lifelong Person Re-Identification

**arXiv ID:** 2609.12577 | [PDF](https://arxiv.org/pdf/2609.12577v1)

**作者:** Kunlun Xu `[一作]` (Peking University), Jiahuan Zhou `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 SCORE 的方法，用于解决衣物混合（cloth-hybrid）终身行人重识别（CH-LReID）任务，重点通过自适应子分布建模、分布知识强化与实例结构知识保持来缓解衣物一致与变化场景下的灾难性遗忘。

**💡 创新点**

创新点包括：①自适应子分布建模（ASD）为每个身份学习多组可学习的高斯子原型，显式捕获衣物一致与变化导致的身份内部多样性；②分布知识强化（DKR）利用旧阶段子分布进行协同对齐，传递历史知识；③实例结构知识保持（ISK）在新旧特征空间间保持跨实例关系，进一步提升抗遗忘性能。

**🔧 技术方法**

技术手段主要有：多模态高斯子原型学习、最大似然分配、原型分离与聚类损失、分布对齐的 KL 散度损失、跨实例结构一致性约束；整体框架基于 ResNet‑50 backbone，并结合交叉熵、三元组损失等传统 ReID 损失。

**📊 数据集**

使用了 CH‑LReID 基准数据集：Market‑1501、MSMT17‑V2、CUHK03（衣物一致）以及 LTCC、PRCC（衣物变化）两组顺序训练集，并在两种不同训练顺序（Order‑1 与 Order‑2）下进行评估。

**📈 对比分析**

与多种 LReID 与 CH‑LReID 方法（如 LwF、AKA、PatchKD、LSTKC、USP、DKP、DASK、DSIFLF、DKC）以及联合训练基线对比，SCORE 在两种训练顺序下的 mAP 和 Rank‑1 取得了 3–4% 的提升，整体性能达到同类任务的最高水平。

**⚠️ 局限性**

局限性：① 需要为每个身份维护多组子原型，导致模型参数和内存消耗较大；② 对子原型数量 k、采样数 s 以及损失权重 α、β 等超参数敏感，需手动调优；③ 对极端衣物变化场景的提升有限，仍受数据分布差异与域间对齐难度限制。

---

## 315. Reproducing and Evaluating the Generalizability of Subliminal Learning in Open-Weight Models

**arXiv ID:** 2609.12586 | [PDF](https://arxiv.org/pdf/2609.12586v1)

**作者:** Daan van der Weijden `[一作]` (University of Zurich), Selene Baez Santamaria `[通讯]` (University of Zurich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

复现并扩展了教师模型通过无语义相关数据对学生模型隐含行为偏好进行潜在学习的实验，加入了演员和政治人物偏好、新任务棋步生成、Ministral8B模型以及答复空间大小消融。

**💡 创新点**

创新点在于证明潜在学习的非均匀性：不同偏好类别、不同任务、不同模型及答复空间大小对偏好传递强度产生显著影响。

**🔧 技术方法**

技术手段包括教师-学生蒸馏、行为偏好微调、随机数/棋步序列生成、过滤无效输出、对数优势评估与对齐/拒绝率判别。

**📊 数据集**

使用的数据集为教师生成的无语义数列、代码、思路链以及棋步序列，覆盖10种动物、演员和政治人物等目标实体。

**📈 对比分析**

通过对数优势（log‑odds）和95%置信区间比较trait‑FT与regular‑FT学生在中性提示下的命名频率；结果显示动物偏好传递显著，政治人物更强，棋步传递弱，Ministral几乎无效。

**⚠️ 局限性**

局限性包括仅使用公开权重模型、对非法棋步过滤不足、实体选择主观、实验范围有限且与原GPT‑4实验不可直接对比。

---

## 316. Calibrated Ambiguity in Multimodal Language Models: Humans reach for cultural references, while models describe the picture

**arXiv ID:** 2609.12575 | [PDF](https://arxiv.org/pdf/2609.12575v1)

**作者:** Cody Kommers `[一作]` (Alan Turing Institute), Drew Hemment `[通讯]` (Alan Turing Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文利用桌游Dixit作为实验平台，设计了一套多模态语言模型（LLM）生成提示语的任务，以量化和评估模型在不同情境下对模糊性的校准能力。

**💡 创新点**

创新点在于提出了首个针对多模态LLM的“校准模糊性”评价框架，结合四维量表（校准度、情境化程度、文字直白度、比喻性）以及将人类与模型生成的提示语在同一游戏盘上直接比较。

**🔧 技术方法**

主要技术包括：多模态LLM（Gemma‑3、Qwen‑3.5、GLM‑4.6V‑Flash、Kimi‑VL、ERNIE‑4.5‑VL 等）在两种提示（最小提示与社会认知提示）下的文本生成，随后利用LLM编码器对提示语进行四维评分。

**📊 数据集**

使用的数据集为Vatsakis等人公开的基于Dixit游戏的116,226局人类提示语数据，进一步抽取350局作为实验样本，涵盖人类与模型生成的提示。

**📈 对比分析**

通过人类专家和LLM编码器对同一局的提示语进行双盲评分，比较四维得分及目标卡选择准确率；结果显示大多数LLM出现“模糊性崩溃”，校准度普遍偏高；Gemma‑3在社会认知提示下能与人类保持相当的校准水平；文化情境化程度在所有模型中低于人类。

**⚠️ 局限性**

局限性包括：任务仅测量“校准模糊性”而非所有类型的模糊；文化引用缺乏对不同受众的适配；LLM编码器虽与人类相近但仍可能引入偏差；模型规模和架构差异未完全解释差异；实验基于英语Dixit卡片，缺乏跨语言验证。

---

## 317. ProClosure: Hierarchical Room-Object Assignment using Progressive Boundary Closure from Monocular Video

**arXiv ID:** 2609.12614 | [PDF](https://arxiv.org/pdf/2609.12614v1)

**作者:** Vinoth Kumar Muthuraj `[一作]` (IIT Jodhpur), Hardik Jain `[通讯]` (IIT Jodhpur)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种名为ProClosure的进化边界闭合方法，能够在无深度信息、无位姿标定的单目RGB视频中恢复室内场景图中的房间层和对象归属。

**💡 创新点**

创新点在于将边界闭合尺度动态设定为每个空间自身闭合所需的最小膨胀量，避免了全局阈值导致的误合并，同时采用宽度判定而非门窗识别来处理边界缺口。

**🔧 技术方法**

技术包括：MASt3R‑SLAM与SAM3前端生成稠密点云与结构云，基于高度裁剪与正交投影构造地面图，使用基于种子点的前向膨胀与种子分隔的进化闭合，及基于对象表面重投影的多数投票与视线一致性对象归属规则。

**📊 数据集**

实验使用HM3D‑Semantics v0.2验证集（10层、6场景，共2264个标注对象），以及手持手机拍摄的真实室内视频用于定性展示。

**📈 对比分析**

与重实现的HOV‑SG基线在同一地面图上比较，ProClosure在对象-房间分配的准确率、ARI、NMI上显著优于基线（p≈0.002），并且在房间计数上误差仅为2个而基线为28个，房间F1也显著提升；但在房间IoU的统计显著性较弱（p≈0.065）。

**⚠️ 局限性**

局限性包括：对宽阔开放式空间的分割仍会将相邻区域合并，无法仅凭几何闭合区分过大的开口；方法无法直接恢复尺度、楼层分隔与门窗定位；对极端稀疏扫描或视角不充分的场景仍可能产生不完整或错误的房间边界。

---

## 318. GreenDirector: carbon- and water-aware workload placement for sustainable computing

**arXiv ID:** 2609.12602 | [PDF](https://arxiv.org/pdf/2609.12602v1)

**作者:** Jime Iglesias Blanco `[一作]`, Álvaro López García `[通讯]` (Instituto de Física de Cantabria, CSIC-UC)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了统一的环境指数（ES）和基于ES的绿色得分（GS），并将其集成到AI4EOSC云平台和DIRAC工作负载管理系统的调度器中，实时评估和调度基于碳排放与水资源稀缺的多维环境影响。

**💡 创新点**

创新点在于：①将实时跨境电力流追踪与AWARE2.0水稀缺权重结合，生成无量纲的ES；②在GS中结合ES、PUE和硬件效率，提供可直接用于调度的环境友好度；③在生产多租户系统中验证该指标，首次展示水资源稀缺可以动态地影响实时调度。

**🔧 技术方法**

使用技术包括：Wattnet实时电力流与碳/水足迹API、AWARE2.0水稀缺因子、ISO 14046 LCA标准、比值归一化与权重组合、Nomad调度器、Consul服务发现、DIRAC WMS和自定义调度插件（gd、gsd）。

**📊 数据集**

数据集为：Wattnet提供的15分钟分辨率欧洲电力碳足迹(CF)、水足迹(WF)、水稀缺权重(WI)；AWARE2.0月度水稀缺因子；AI4EOSC的四个云提供商（IFCA、IISAS、PSNC、TUBITAK）和DIRAC的五个站点（CC‑IN2P3、RAL‑LCG2、FZK‑LCG2、INFN‑T1、SARA‑MATRIX）的环境与硬件参数。

**📈 对比分析**

比较方法包括：①AI4EOSC的集群填充实验和得分检查，验证绿色亲和度使绿色站点优先被填充；②DIRAC的基于历史轨迹的离线仿真，比较随机站点排序与GS排序的碳排放、碳效率和水稀缺效率；③生产环境下的两段实验，比较GS排序与随机排序在碳效率与水稀缺效率上的差异。实验结果显示：AI4EOSC绿色站点先被填满，DIRAC仿真中碳排放减少42.8%，碳效率提升49.5%，而水稀缺效率在GS排序下略低。

**⚠️ 局限性**

局限性包括：①仅覆盖运营期Scope‑2的碳与水足迹，未考虑现场冷却水、设备生命周期等；②ES与GS仅包含碳与水两项，无法反映其他环境维度；③权重(W_C=0.71,W_W=0.29)固定，无法自适应不同地区/季节的水/碳危害变化；④GS聚焦环境强度，可能导致负载过度集中在“绿”站点，忽略容量与服务质量；⑤未对真实碳水排放做闭环验证，指标与实际影响之间的偏差仍需进一步评估。

---

## 319. TraceMind: Predicting User Information Uptake from Low-Cost Interaction Traces during Human-LLM Content Co-Generation

**arXiv ID:** 2609.12600 | [PDF](https://arxiv.org/pdf/2609.12600v1)

**作者:** Yu Mei `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了人类-LLM共生成过程中信息单元的摄取，提出低成本交互轨迹预测模型 TraceMind。

**💡 创新点**

创新点在于将信息摄取评估转为基于开放式文本的原子信息单元，结合语义与布局对齐的低成本交互轨迹建模。

**🔧 技术方法**

使用 GPT‑5.4 进行信息单元抽取、问题生成和语义匹配，构建六通道热力图、时间窗口序列和工作流特征，再用卷积、TCN、ElasticNet 等融合。

**📊 数据集**

数据集来自 62 名参与者在三种日常写作任务中的共生成会话，包含 1187 个原子信息单元的摄取标签及完整交互日志。

**📈 对比分析**

与多种基线（GBT、LSTM、Transformer、ResNet 等）比较，TraceMind 在 AUROC、平衡准确率、AUPRC_non、宏 F1 上分别提高约 4.5%、12.4%、4.0%、8.2%。

**⚠️ 局限性**

局限包括仅评估即时识别层面，自动化问题生成可能缺乏心理测量严谨性，样本受限于中国 LLM 用户，未验证在触控等其他输入方式上的适用性。

---

## 320. Invisible Yet Dominant: Big Stalls of Kernel I/O Mechanisms in Cloud OLTP Databases

**arXiv ID:** 2609.12597 | [PDF](https://arxiv.org/pdf/2609.12597v1)

**作者:** Mitsumasa Kondo `[一作]` `[通讯]` (NTT, Inc.), Mitsumasa Kondo (NTT, Inc.)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用eBPF对云端OLTP数据库的内核I/O路径进行分析，揭示了传统计数器无法捕获的I/O堵塞现象。

**💡 创新点**

创新点在于通过eBPF对KFT写回、前台写入和IO-less Dirty Throttle暂停进行细粒度监控，量化了分布式块存储下的内核瓶颈。

**🔧 技术方法**

采用eBPF追踪tracepoint，结合对bio标记和kworker执行时间的关联分析技术。

**📊 数据集**

使用TPC‑C基准测试数据集，在AWS Rocky Linux 9.8环境下运行PostgreSQL 16.4，配置三种磁盘方案。

**📈 对比分析**

通过与单盘、WAL分离、SteelDB四盘配置的对比，SteelDB实现23%吞吐提升、21.7%写入带宽提升，最大New‑Order事务延迟降低59.4%。

**⚠️ 局限性**

局限性在于仅聚焦内核层I/O瓶颈，未深入探究与内核调优参数的交互关系及在更广泛云环境中的普适性。

---

## 321. A Retrieval-Augmented Automated Stakeholder for Requirements Elicitation Education: A Comparative Study

**arXiv ID:** 2609.12576 | [PDF](https://arxiv.org/pdf/2609.12576v1)

**作者:** Manal Binkhonain `[一作]` (King Saud University), Ohoud Mosa Alharbi `[通讯]` (King Saud University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估基于检索增强生成（RAG）的自动化利益相关者对话系统，用于软件工程课程中需求获取训练，并与传统人类角色扮演进行对照实验。

**💡 创新点**

通过将LLM生成与教师预先提供的场景文本结合，利用RAG降低幻觉并提升问题质量；系统可无人工参与、可重复练习，首次在RE教育中验证其效果并与人类角色对比。

**🔧 技术方法**

LangChain框架、OpenAI GPT‑3/4 LLM、FAISS向量数据库、文本切分与嵌入、检索‑生成管道、滑动对话窗口和Persona prompt。

**📊 数据集**

约700词的教学用系统规范文档，描述iOS家长‑专家‑管理员三类用户及隐性质量需求。

**📈 对比分析**

采用两组实验（本科生对照实验、研究生交叉实验），收集10项5点量表问卷与专家评估（9项误差量表、4项功能质量量表）。结果显示ASG在客观指标上显著优于ISG（错误率下降、功能质量提升，p<0.001），但在主观体验上ISG略高。

**⚠️ 局限性**

实验范围受限于单一小规模文档与场景，样本量有限，模型版本不同导致可比性受限；自动系统缺乏情感、非语言线索和多模态交互，难以完全模拟真实交互；未在复杂真实工业环境中验证。

---

## 322. KAD-Net: Kinematics-Aware Decoupled Learning for Robust 3D Hand Pose Estimation from a Single Depth Image

**arXiv ID:** 2609.12559 | [PDF](https://arxiv.org/pdf/2609.12559v1)

**作者:** Jun Lu `[一作]` (Tianjin University), Qiao Liu `[通讯]` (Chongqing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出KAD-Net，通过分离UV定位与深度回归并引入Finger Topology Constraint模块，实现3D手姿估计。

**💡 创新点**

创新点在于将手指三连关节的局部运动学约束与任务分离的多任务学习结合，并通过层次化深度分支降低任务干扰。

**🔧 技术方法**

技术上使用Hourglass骨干网络、图卷积（GCN）实现关节点嵌入，配合多分支结构和注意力融合。

**📊 数据集**

使用ICVL、NYU和MSRA三大公开深度图数据集。

**📈 对比分析**

在三大基准上均超过现有SOTA，ICVL 5.45 mm、NYU 7.43 mm、MSRA 6.93 mm，速度达87 FPS。

**⚠️ 局限性**

局限在于对严重遮挡多关节同时误差时表现下降，且对手指间运动相似度高的情况仍有一定误差。

---

## 323. STAR: Sparse Tactile Representation Learning in Vision-Tactile-Language-Action Models for Dexterous Manipulation

**arXiv ID:** 2609.12549 | [PDF](https://arxiv.org/pdf/2609.12549v1)

**作者:** Xiangcheng Liu `[一作]` (Shanghai Innovation Institute), Jianlan Luo `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

收集了200小时的真实世界双手抓取数据集，并提出STAR训练方案以提升视觉‑触觉‑语言‑动作模型在多指抓取任务中的表现。

**💡 创新点**

创新点在于三大模块：①视觉‑触觉联合预训练解决触觉空间稀疏；②稀疏‑全局触觉标记化降低冗余；③稀疏未来触觉预测丰富信息稀疏；三者协同构成一套完整的训练策略。

**🔧 技术方法**

使用的技术包括视觉‑触觉联合掩码预测、稀疏标记化注意力机制、稀疏未来触觉预测、基于流匹配的动作生成、以及多模态（视觉、触觉、语言、关节状态）交叉对齐。

**📊 数据集**

数据集为自研的200小时双手抓取数据集，包含30Hz同步的双手相机图像、全手触觉传感器数据、机器人状态、动作序列以及任务级语言指令，65个任务中69.5%为多指抓取。

**📈 对比分析**

与GR00T N1.7、LDA-1B、π_0.5等基线进行对比，使用相同的100条后训练轨迹。STAR模型在四项真实任务中平均成功率达到61%，且在各项子任务完成率（TCR）上均超过所有基线，证明其在多指抓取和多任务泛化方面具有显著优势。

**⚠️ 局限性**

局限性包括：数据集规模仍小于并行夹爪数据集，任务多样性有限；触觉传感器覆盖不完全，导致某些任务受限；方法目前仅对视觉‑触觉对齐做预训练，语言对齐缺失；在某些任务中触觉输入并未提升效果。

---

## 324. DRS-VPT: Directly Relocalizing in a Scan with Vision Point Transformers

**arXiv ID:** 2609.12557 | [PDF](https://arxiv.org/pdf/2609.12557v1)

**作者:** Lanke Frank Tarimo Fu `[一作]` (University of Oxford), Maurice Fallon `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种端到端的Transformer架构DRS‑VPT，用于将查询图像与任意3D点云直接对齐，实现图像‑扫描的实时定位。

**💡 创新点**

创新点在于将跨模态特征融合、几何预测与可微分直接对齐统一在同一网络中，并通过多尺度稠密特征金字塔和自监督的重投影损失学习显式的遮挡抑制与尺度恢复。

**🔧 技术方法**

技术上采用冻结的DINOv2图像编码器、Sonata点云Transformer、交叉模态全局与局部注意力、三层粗细特征金字塔、Gauss–Newton可微分对齐以及自监督的重投影与尺度损失。

**📊 数据集**

在多种公开数据集上训练与评估，包括MegaDepth、OxfordSpires、KITTI、ETH3D、Argoverse、nuScenes、PandaSet、7‑Scenes、ScanNet++、WildRGBD、GrandTour以及12Scenes，用于跨域泛化实验。

**📈 对比分析**

与现有对应点、PnP、ICP、Kabsch、TrafficLoc、CoFiI2P等方法对比，DRS‑VPT在宽基准相机‑LiDAR对齐中实现了0.09 m/0.34°的误差并在室内重定位、零样本跨域测试中达到接近或超过最先进方法的召回率，证明了其在多传感器、多尺度环境下的鲁棒性。

**⚠️ 局限性**

主要局限包括仅支持单一刚性扫描、对点云密度敏感、对动态或可变物体缺乏建模，且在高度重复结构（如楼梯）下定位精度下降。

---

## 325. RoofLang: Enabling AI-Driven Architecting of LLM Inference Systems

**arXiv ID:** 2609.12551 | [PDF](https://arxiv.org/pdf/2609.12551v1)

**作者:** Ziyue Yang `[一作]` (Shanghai Xingyunzhili Artificial Intelligence Institute), Peng Cheng `[通讯]` (Microsoft Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 RoofLang DSL，支持 AI 通过图变换与布置动作实现 LLM 推理系统的自主架构。

**💡 创新点**

创新点在于将工作负载、硬件和架构动作抽象为图与语义保持变换，并配备基于 Roofline 的离散事件仿真器，解耦实现细节，允许 AI 在理论资源上搜索更优架构。

**🔧 技术方法**

使用了基于图的计算与硬件表示、语义保持的图变换、资源归属的 Placement 原语以及基于 Roofline 的离散事件模拟器。

**📊 数据集**

评估数据集包括 DeepSeek V4 Flash/Pro、GLM‑5.3、Kimi‑K3 等模型以及 NVIDIA H200/GH200/B300/GB300 四种硬件平台。

**📈 对比分析**

通过对预填充与解码阶段的吞吐‑交互性 Pareto 前沿进行对比，发现 DeepSeek V4 在解码吞吐上比 GLM‑5.3 与 Kimi‑K3 分别高 3.5–39.5 倍，AI 优化后可提升 6.23–50.1%。

**⚠️ 局限性**

主要限制包括：需人工验证初始图；模型简化（忽略 KV‑append、speculative‑decode、MoE 负载均衡等）；离散事件模拟器在大规模配置下规模受限。

---

## 326. Confusion-Erasure Bounds of Error-Bounded Decoders under QAM

**arXiv ID:** 2609.12631 | [PDF](https://arxiv.org/pdf/2609.12631v1)

**作者:** Wenwen Chen `[一作]` (RPTU University), Hans D. Schotten `[通讯]` (RPTU University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对6G短包通信，分析并推导了误差界定解码器（ε‑bounded decoder）在正方形M‑QAM星座下的块混淆率（block confusion rate）上界与下界。

**💡 创新点**

创新点：① 引入符号平方距离系数 Δ，统一描述 QAM 中不同符号能量导致的欧氏距离分布；② 通过 Δ 将欧氏距离期望与海明距离关联，得到针对任意 M‑QAM 的闭式混淆率上下界；③ 证明这两条界在平均符号能量与块长上单调递减，并给出与星座阶数相关的下降速率。

**🔧 技术方法**

技术：概率与信息理论分析，利用极大似然/映射解码器的非重叠决策域条件，统计期望欧氏距离与海明距离关系，Jensen不等式与凸性分析，符号能量统计以及χ²分布相关量化。

**📊 数据集**

数据集：无外部真实数据集；实验采用仿真方式，对 16‑QAM 与 64‑QAM 在不同块长与信噪比（SNR）下进行蒙特卡洛模拟，以验证理论界。

**📈 对比分析**

比较方法：将理论得到的混淆率上下界与仿真得到的混淆率进行对比；结果显示理论界始终位于仿真曲线之上，并且混淆率远低于设定的块错误率阈值 ε，验证在低码率/高SNR下误码可忽略，主要残留错误为可检测的擦除。

**⚠️ 局限性**

局限性：① 只考虑均匀符号使用的正方形 QAM；② 忽略了具体码（如 LDPC、Polar）距离谱的细节；③ 未处理概率形状编码或非正方形星座；④ 在低SNR下混淆率界未能完全低于 ε，说明在此情形下仍需考虑误码检测方案。

---

## 327. SteerDuplex: Steerable Duplex Speech Dialogue Models

**arXiv ID:** 2609.12623 | [PDF](https://arxiv.org/pdf/2609.12623v1)

**作者:** Utkarsh Tyagi `[一作]` (Scale AI), Yunzhong He `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可被用户指令调节的全双工语音模型 SteerDuplex，并构建了评测其可引导性的基准 SteerBench。

**💡 创新点**

创新点在于：①定义了完整的可引导性分类法；②将监督微调与两阶段强化学习相结合，使用混合奖励提升语音交互时序和持续性；③提供了专门评测可引导性的新基准。

**🔧 技术方法**

技术上采用 Moshi 的双声道 Transformer 架构，先进行监督微调，再通过两阶段 GDPO 强化学习，奖励包括交互时序、判别者语义反馈和波形完整性。

**📊 数据集**

训练数据为 504,416 条自然对话音频与 65,675 条文本示例，涵盖指令跟随、可引导性、推理、安全等；评测使用 SteerBench（390 条口语提示、1,067 条人工评卷），并对比 Audio MultiChallenge、VoiceBench、FDB 等公开基准。

**📈 对比分析**

通过与 Moshi、PersonaPlex 等开源基线比较，SteerBench 音频可引导通过率从 20% 提升至 65%；RL 后源清晰中断响应率从 72.5% 提升至 82.5%，合成暂停中断率从 26.5% 降至 9%；在多轮任务和语音指令基准上保持或略有提升。

**⚠️ 局限性**

限制包括：基准仅覆盖固定英语示例，未验证多语言/方言泛化；模型规模受 Moshi 7B 限制，无法评估更大容量模型的上限；强化学习奖励易出现空答或过早让出的“奖励劫持”，评判主观性与声学参考的可靠性仍需进一步验证。

---

## 328. Quantifying Spectral Differences in Vehicle Between Production Autonomous and Human-Driven Vehicles Across Driving Scenarios

**arXiv ID:** 2609.12609 | [PDF](https://arxiv.org/pdf/2609.12609v1)

**作者:** Peiyi Fang `[一作]` (Hong Kong University of Science and Technology), Ke Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并实现了基于频域的框架，利用真实世界PAV（生产级自动驾驶车辆）与HV（人类驾驶车辆）的轨迹数据，系统量化其在不同驾驶状态、光照、天气和车流密度等场景下的运动学差异。

**💡 创新点**

①首次将频域特征（频谱质心、频谱熵、累计频率、频段能量等）应用于PAV–HV差异分析；②采用KDE重叠与归一化Wasserstein距离两种分布差异度量，提升评估的全面性；③在多维场景框架内进行跨平台、跨情境的系统化比较。

**🔧 技术方法**

主要技术包括：离散傅里叶变换提取频域特征；移动平均平滑、趋势去除与Hann窗；核密度估计（KDE）与Wasserstein距离；Bootstrap置信区间评估统计不确定性。

**📊 数据集**

使用PAVE数据集，该数据集提供来自四款PAV平台（Toyota bZ3X、Xiaomi Auto YU7、NIO ET5、AITO M7）以及人类驾驶车辆的高频GNSS轨迹和详细场景注释。

**📈 对比分析**

对每个平台、场景、运动学信号（速度、加速度、角速度等）和频域特征分别计算KDE重叠得分S^KDE和归一化Wasserstein得分S^W。结果显示：不同场景下差异显著，频域方法能稳定、客观地量化PAV与HV的行为差异；Bootstrap置信区间表明评估结果稳健。

**⚠️ 局限性**

局限性：仅包含少量PAV平台和有限场景，某些条件下样本量较小；未深入探讨PAV行为背后的决策与控制机制；未评估这些差异对交通流动性或安全性的具体影响。

---

## 329. Direct Preference Density Alignment for Conversational Audio Equalization

**arXiv ID:** 2609.12607 | [PDF](https://arxiv.org/pdf/2609.12607v1)

**作者:** Ioannis Stylianou `[一作]` (Aalborg University), Zheng-Hua Tan `[通讯]` (Aalborg University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无代理奖励模型的在线强化学习框架，利用Reflective-KDE构建非参数偏好密度图并在音频等化任务上实现连续参数估计。

**💡 创新点**

创新点在于彻底消除学习型奖励模型的内存与奖励劫持问题，采用经验密度面实现稳定奖励，并将在线GRPO与离线DPO结合形成混合策略，最终使1.5B模型在盲听测试中与GPT‑4o mini实现感知等价。

**🔧 技术方法**

主要技术包括Reflective Kernel Density Estimation、Group Relative Policy Optimization (GRPO)、Direct Preference Optimization (DPO)、Qwen2.5‑Instruct LLM、低秩适配 (LoRA) 以及语义数据增强与奖励表面归一化。

**📊 数据集**

使用约90,000条用户交互事件构建的Beosonic EQ控制空间偏好数据集，并以30条无约束收集的自然语言提示作为评估 OOD 集。

**📈 对比分析**

通过 GMRR、格式成功率、以及盲听 A/B 测试进行比较；GRPO+DPO 在 OOD 组的 GMRR 最高达 0.60、格式成功率 100%，与 GPT‑4o mini 的盲听分数 3.04 对比仅差 0.12，胜率 77.9%，统计检验显示两者在感知上无显著差异。

**⚠️ 局限性**

局限性包括：高维控制空间时 KDE 的维度灾难；仅基于文本的条件化缺乏音频上下文感知；未实现个性化偏好建模；以及对大规模数据与推理算力的依赖。

---

## 330. Beyond Generation and Accuracy: Diagnosing and Enhancing Visual Chain-of-Thought for Geometry Problem Solving

**arXiv ID:** 2609.12606 | [PDF](https://arxiv.org/pdf/2609.12606v1)

**作者:** Zhitong Dong `[一作]` (Southeast University), Jinjie Gu `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个用于几何推理的视觉链式思维诊断基准GeoVAD-Bench，并基于该基准构建了GeoWeave模型，实现了从图像感知、辅助绘制到推理的完整流程。

**💡 创新点**

创新点在于（1）细粒度的五维诊断评估框架，能分解视觉推理中的感知、辅助质量、利用、推理和答案准确度；（2）针对诊断结果设计的分阶段监督微调与交互式强化学习框架，显著弥合自主生成与参考辅助之间的差距；（3）端到端的图像生成与文本推理协同训练，实现真正的视觉链式思维。

**🔧 技术方法**

技术包括：多模态统一Transformer（SenseNova‑U1），逐步监督微调（几何感知、辅助绘制、完整推理三阶段）；交互式强化学习（UniRL+Step‑level Credit Assignment）；视觉生成采用像素流匹配损失；图像生成与编辑统一在同一模型中。

**📊 数据集**

数据集主要来自公开几何问题集（Math‑VR、Geo170K、MMK12、GeoGPT4V等）以及自研教材与考试题目，经过代码转化后生成400K感知样本、200K编辑样本、100K交互推理样本，最终形成600题GeoVAD‑Bench测试集。

**📈 对比分析**

与多种闭源（Gemini‑3.5‑Flash、GPT‑5.4‑Thinking等）和开源（SenseNova‑U1‑8B、MathCanvas‑7B、CodePlot‑CoT‑32B等）模型对比，GeoWeave‑8B在GeoVAD‑Bench Auto‑Aux模式下最终答案准确率提升至62.6%（相较SenseNova提升25.3pp），过程平均提升至82.9%（提升30.4pp），在公开数学与视觉-文本推理基准上也取得领先。

**⚠️ 局限性**

局限性包括：模型规模与数据量的进一步扩展未系统评估；交互式强化学习实现成本高，缺乏视觉生成分支的联合策略更新；所获提升主要体现在几何推理任务，尚未充分验证能否迁移到更广泛的多模态推理场景。

---

## 331. Mentorship resources and citation-elite journal publication trajectories after training: Evidence from bioscience mentor-mentee networks

**arXiv ID:** 2609.12564 | [PDF](https://arxiv.org/pdf/2609.12564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 332. A Feature-Rich Embedded NIDS with eBPF/XDP: Detector and Architecture Trade-offs

**arXiv ID:** 2609.12605 | [PDF](https://arxiv.org/pdf/2609.12605v1)

**作者:** Shiqi Wu `[一作]` (Chalmers University of Technology), Romaric Duvignau `[通讯]` (Chalmers University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于eBPF/XDP内核过滤与Isolation Forest多特征的嵌入式DDoS检测系统，适配Raspberry Pi 5等资源受限环境；

**💡 创新点**

创新点在于将eBPF/XDP与多维流特征提取器GoFlowMeter相结合，微服务架构下通过kafka与gRPC对比评估检测质量与传输开销，并对Isolation Forest进行离线与在线超参数调优；

**🔧 技术方法**

使用的技术包括eBPF/XDP网络层过滤、GoFlowMeter特征提取、Isolation Forest无监督异常检测、Apache Kafka异步消息、gRPC同步RPC以及Go语言实现；

**📊 数据集**

采用公开的CIC-DDoS2019数据集（≈150 GB）进行流量回放与评估；

**📈 对比分析**

对三种架构（单进程、Kafka微服务、gRPC微服务）在同一数据集上进行检测准确率、F1、处理时间、传输延迟和内存占用比较。结果显示Isolation Forest显著提升召回率与F1（最高≈0.965），gRPC微服务在保持近乎单进程检测质量的同时，平均传输延迟仅≈2 ms，Kafka传输延迟≈27 ms并伴随检测性能下降；

**⚠️ 局限性**

主要限制包括：仅评估了完整80维特征和单一包计数两种输入，未进行特征选择；实验仅在Raspberry Pi 5和CIC-DDoS2019上进行，缺乏对更复杂或实时流量的泛化验证；微服务架构在高负载或多机部署下的瓶颈未深入探讨；

---

## 333. How Do Data Collection Strategy and Data Quality Influence the Outcomes of Digital Technology Adoption?

**arXiv ID:** 2609.12745 | [PDF](https://arxiv.org/pdf/2609.12745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 334. A Hierarchical Coverage Path Planning Algorithm for Unknown Environments

**arXiv ID:** 2609.12595 | [PDF](https://arxiv.org/pdf/2609.12595v1)

**作者:** Zongyuan Shen `[一作]` (Jinan University), Dehua Zhou `[通讯]` (Jinan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在未知环境中提出一种在线层级覆盖路径规划算法，利用递归分解得到的子区域构建层级树，并在此基础上实现全局巡线与局部覆盖。

**💡 创新点**

创新点包括：①构建增量式层级空间分解树以记录子区域的父子关系；②基于子区域的探索状态和与机器人距离的双重优先级进行全局巡线更新；③在局部完成后自动切换到下一个优先级最高的子区域，避免死胡同和重复覆盖。

**🔧 技术方法**

采用的技术包括：在线空间分解与递归子区域识别、树结构管理、距离评估与排序、A*最短路径搜索、局部贪婪扫描、已探索子区域的TSP最优覆盖路径生成。

**📊 数据集**

使用数据集：在Gazebo仿真平台上构造四个90m×90m的复杂场景，采用30×30格网进行映射与覆盖；与三种基线算法（ε*、BINN、PPCPP）进行对比。

**📈 对比分析**

比较方法：在路径长度和重叠率两个指标上与基线算法进行数值比较。实验结果显示，所提算法在所有指标上均优于基线，路径更短、重叠率更低。

**⚠️ 局限性**

限制：仅验证单机器人静态环境；对动态障碍或多机器人协同覆盖尚未扩展，需要进一步研究。

---

## 335. NovaFabric: Tamper-Evident, Replayable Evidence for Autonomous AI Agent Runs

**arXiv ID:** 2609.12582 | [PDF](https://arxiv.org/pdf/2609.12582v1)

**作者:** Mohsen Seyedkazemi Ardebili `[一作]` `[通讯]` (NovaFabric Project), Mohsen Seyedkazemi Ardebili (NovaFabric Project)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种可生成审计级执行证据的系统Run Capsule，记录AI代理执行过程并通过DSSE签名、RFC3161时间戳、Merkle日志等实现不可篡改、可重放、可公开验证的证据包。

**💡 创新点**

创新点包括：①将OpenTelemetry、DSSE/in‑toto与W3C PROV整合为统一的执行胶囊；②提出完整的四模式（exact、mocked、semantic、forensic）重放协议；③实现加密红色化证明（redaction attestation）并支持Evidence Bundle的独立第三方验证。

**🔧 技术方法**

采用技术包括OpenTelemetry（GenAI规范）、DSSE签名、RFC3161时间戳、Merkle树追加日志、W3C PROV模型、REST接口、云端负载均衡和四模式重放协议。

**📊 数据集**

使用内部构造的10个实验场景，涵盖模型调用、工具调用、文件/网络/审批等事件；未公开公开数据集，全部采用模拟与自制脚本。

**📈 对比分析**

与现有可观测追踪平台和学术系统对比，实验显示：模型调用100%捕获；红色化准确率100%；结构差分定位误差零；重放无真实模型调用；在10M边缘下p99 45.5 ms，100M边缘p99 167.9 ms，比基线快3.3倍；集群 ingest吞吐61.6 req/s，p99 26.8 s；检测到三类篡改并全部被检测。

**⚠️ 局限性**

局限性包括：仅记录代理到系统边界的可观测事件，未捕获内部推理；实现依赖可信计算基础且未在大规模集群完整验证；redaction attestation 需要可信密钥托管；缺少独立第三方验证，且对多模型版本、容器化、服务器端 API 的兼容性待进一步评估。

---

## 336. Assisted Spatial Cognition Through Vision-Language Models

**arXiv ID:** 2609.12747 | [PDF](https://arxiv.org/pdf/2609.12747v1)

**作者:** H. Riaz `[一作]` (Dublin City University), M. I. Ali `[通讯]` (Dublin City University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个端到端的辅助导航系统，利用智能手机摄像头视频先通过SLAM3R实现实时稠密3D点云重建，再通过自研点云对齐与缩放算法保证空间一致性，随后使用SpatialLM生成结构化场景描述，最后通过本地部署的GPT-OSS-20B将三维信息转化为自然语言导航指引和空间推理，专为视觉和认知障碍者提供完整的三维空间认知支持。

**💡 创新点**

创新点在于将大语言模型、视觉语言模型与数字孪生技术深度融合，突破传统2D导航的局限，实现对障碍者的三维空间感知与推理；同时提出了自研的点云多视角对齐算法，确保跨视角的空间一致性，为LLM提供高质量三维输入。

**🔧 技术方法**

技术组合包括：单目稠密SLAM3R实现点云重建；Open3D等工具进行点云后处理（去噪、对齐、尺度校正）；Vision‑Language Model SpatialLM用于结构化场景解析；本地部署的GPT‑OSS‑20B实现自然语言描述与空间推理。

**📊 数据集**

数据集来自作者自行采集的多场景视频（实验室、会议室、大型建筑等），使用iPhone 16或Android广角摄像头录制30‑120秒的室内/室外视频，未使用公开数据集。

**📈 对比分析**

通过对齐前后对比实验显示，点云对齐显著提升物体检测数量（如实验室从8提升至22，会议室从13提升至28），并在多视角测试中保持高准确度；实验展示了三维语义解析与导航指引的可靠性，尽管缺乏公开基准，但结果表明系统性能优异。

**⚠️ 局限性**

局限性包括：SpatialLM目前无法识别楼梯、电梯等关键障碍物；系统仅在室内环境下验证，缺乏大规模公开评测；对摄像头抖动和低质量视频敏感；本地LLM部署对计算资源有一定需求。

---

## 337. Skill Issue: Lessons from Optimizing Repository SKILLs for Coding Agents

**arXiv ID:** 2609.12742 | [PDF](https://arxiv.org/pdf/2609.12742v1)

**作者:** Mykhailo Kozyrev `[一作]` (Technical University of Munich), Anton Podkopaev `[通讯]` (Constructor University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Kotlin项目中，构建了一套基于合并拉取请求的任务集，并用此任务集对编码代理的SKILL（纯文本知识文件）进行自动合成优化；

**💡 创新点**

创新点在于：①将拉取请求直接逆向生成任务，避免了传统的单文件、单行改动的简化任务；②采用与代理直接交互的配对评分机制，减少对外部基准的依赖；③对比两种优化器GEPA与SkillOpt，系统评估其对代理性能的提升。

**🔧 技术方法**

技术包括：利用GEPA（基于反射的提示优化）和SkillOpt（基于编辑预算的SKILL优化）；构建验证与评分管道，执行Docker化的代理跑；使用Claude Code（Sonnet 4.6）作为编码代理；通过问题描述合成、任务验证和配对评分等流程。

**📊 数据集**

数据集来源为三款Kotlin仓库（kotest/kotest、ktorio/ktor、JetBrains/koog），从其历史合并拉取请求中挖掘出119、131、100个可验证任务；每个任务在冻结的基准提交上逆向回滚实现并恢复测试失败。

**📈 对比分析**

比较方法是将优化后的SKILL与空白seed SKILL在相同任务上的配对评分；实验结果显示GEPA在三仓库分别提升约4.9pp、1.4pp、2.5pp（平均4.9pp），SkillOpt提升约0.1pp，整体提升未显著显现（p≈0.29）。在实际打开的issue上，使用任何SKILL均能让代理更快、成本更低。

**⚠️ 局限性**

局限性包括：①任务集规模有限（20-26个测试集），难以充分分离代理自身变异与SKILL效应；②每次评分成本高昂（单次代理运行平均$0.84），导致优化迭代受限；③逆向任务生成仍受仓库历史变动影响，某些仓库无法满足任务阈值；④仅在Kotlin仓库验证，跨语言泛化尚未验证。

---

## 338. What is the Difference Between Me and You? Benchmarking the Quality Gap Between Human-Written and AI-Generated Code

**arXiv ID:** 2609.12708 | [PDF](https://arxiv.org/pdf/2609.12708v1)

**作者:** Cristina Improta `[一作]` (University of Naples Federico II), Domenico Cotroneo `[通讯]` (University of North Carolina at Charlotte)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比人类编写与 AI 生成的代码在三种语言（Python、Java、C）下的质量差异，系统评估结构复杂度、自然度、缺陷类型（ODC）和安全漏洞（CWE），并基于大规模真实仓库数据构建 CQBench 基准。

**💡 创新点**

创新点在于：①首次将 ODC 与 CWE 标准化用于跨语言、跨作者的缺陷与漏洞对比；②提出 CQBench 基准，包含 27k 条易出现问题的函数并提供完整评测流水线；③发现 AI 生成代码在结构上更简洁但更“模板化”，缺陷分布与人类不同，且在 Python/Java 中出现更高严重级别的安全漏洞。

**🔧 技术方法**

采用静态分析工具（Pylint、PMD、Clang‑Tid y、Semgrep）提取缺陷与漏洞；计算代码复杂度（NLOC、CCN、Halstead 等）和自然度（KenLM n‑gram 交叉熵/困惑度）；将工具规则映射至 ODC 与 CWE 分类；使用 Wilcoxon 符号秩检验、Benjamini‑Hochberg 校正和层次聚类对作者之间的差异进行统计。

**📊 数据集**

使用 CodeSearchNet 与 HMCorp 提供的人类函数，TheVault 提供的 C 函数；基于 docstring 生成 AI 代码（ChatGPT、DeepSeek‑Coder、Qwen2.5‑Coder、OpenAI GPT‑oss），共 787,562 条函数对（Python 285k、Java 222k、C 281k）。

**📈 对比分析**

通过配对的 Wilcoxon 检验比较指标，AI 代码平均仅为人类代码的一半长度、分支深度和 Halstead 体积；但 AI 代码的缺陷与漏洞类型更倾向于模板化、简单但风险更高（Python/Java 中更高严重级别的注入与并发漏洞，C 中更少高严重级别内存错误）。CQBench 在 600 条前沿任务上验证，AI 仍出现约 ⅔ 的缺陷和 ⅓ 的安全漏洞。

**⚠️ 局限性**

局限性包括：①依赖静态分析，存在误报与漏报；②仅评估函数级别代码，未考虑完整程序的运行时行为；③使用的三种语言和四个模型可能不代表所有开发场景；④维护性指数受代码尺寸影响，需谨慎解释；④缺陷/漏洞映射为人工判定，可能受主观偏差影响。

---

## 339. Enabling and Understanding Personalization in AI-Generated Advertising Imagery

**arXiv ID:** 2609.12697 | [PDF](https://arxiv.org/pdf/2609.12697v1)

**作者:** Victor Kolominsky-Rabas `[一作]` (University of Bayreuth), Niklas Kühl `[通讯]` (University of Bayreuth)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并实现了一个将客户数据生成完全AI生成的个性化广告图像的框架，并在100名参与者、四种产品的两阶段实验中评估其对广告态度、产品态度和购买意向的影响。

**💡 创新点**

首次系统性地使用真实个体数据生成完整广告图像，并对不同深度（L0–L2）的个性化进行比较，发现中度个性化最优，揭示了“个性化悖论”在AI生成图像中的表现。

**🔧 技术方法**

结合规则脚本生成个性化Persona描述、OpenAI GPT‑5生成视觉情景prompt，再用OpenAI gpt‑image‑1文本‑到‑图像模型完成图像合成，整个流程实现全自动化。

**📊 数据集**

使用100名美国参与者的自报人口学、生活方式、环境与风格信息作为输入，配合公开产品图片作为参考，构成实验数据集。

**📈 对比分析**

通过within‑subject重复测量ANOVA和配对t检验评估ATA、ATP、PI；结果显示L1显著优于L0，L2不显著且呈负面效应，形成倒U形关系，说明中度个性化效果最佳。

**⚠️ 局限性**

生成图像在高个性化时质量下降，导致“creepiness”上升；未能完全区分个性化深度与图像质量的影响，也缺乏对单一属性的细粒度分析。

---

## 340. I Am AdMan: A Pipeline for Automatic Generation of Personalized Advertising Imagery

**arXiv ID:** 2609.12694 | [PDF](https://arxiv.org/pdf/2609.12694v1)

**作者:** Victor Kolominsky-Rabas `[一作]` (University of Bayreuth), Niklas Kühl `[通讯]` (University of Bayreuth)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并实现了 AdMan 多代理流水线，能够将个体化客户数据自动转换为个性化广告图像，并通过 LLM 评判器进行质量控制，支持大规模生成。

**💡 创新点**

首次系统性地将丰富的客户画像（包括人口、生活方式、环境、外貌等信息）映射到生成提示，再生成完整广告图像；同时通过比较 OpenAI GPT 与 Google Gemini 两种模型配置，揭示模型对图像质量和评判严格度的不同影响。

**🔧 技术方法**

使用的核心技术包括：gpt-5 / gemini-3.1-pro-preview 进行提示生成与评判；gpt-image-1 / gemini-3-pro-image-preview 进行图像生成；多代理协同工作与版本重生成策略；人工标注的 artifact 评估。

**📊 数据集**

实验数据集包含 6 位名人作为测试人物（Bruce Wayne、Vito Corleone 等）和 100 名真实受访者的 100 条完整客户档案；四个产品（电动汽车、奢侈手表、洗衣剂、软饮）作为生成场景。

**📈 对比分析**

通过对比两种模型配置的 Pass Rate（PR）、Artifact Rate（AR）与 Relaxed Artifact Rate（rAR），以及人工专家评审结果，发现 Gemini 在图像生成上 AR 更低，但 PR 低于 GPT；GPT 通过评判器能更好过滤图像，但仍保留较多非文本结构性缺陷。整体来看，产品复杂度越高（如洗衣剂），两配置的 AR 均显著升高。

**⚠️ 局限性**

主要限制包括：文本渲染仍是主要缺陷来源；图像生成在结构和物理一致性方面受限；评判器主要识别文本错误，对结构性错误识别不足；产品复杂度与人物多样性会显著影响质量；缺乏真实用户互动评估与成本、延迟等部署指标。

---

## 341. InRTL: Effective Intra-Inter Interaction Learning for Relational Tables

**arXiv ID:** 2609.12712 | [PDF](https://arxiv.org/pdf/2609.12712v1)

**作者:** Weichen Li `[一作]` (Shanghai Jiao Tong University), Jianhua Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Intra–Inter Relational Table Learning框架，统一建模单表内和跨表间的依赖关系。

**💡 创新点**

首次将自注意力与交叉注意力结合，使用线性化注意力和异构图神经网络实现可扩展性，同时提供理论分析。

**🔧 技术方法**

列感知表编码器、Transformer自注意力、交叉注意力、线性化注意力、HGNN、ResNet等深度学习技术。

**📊 数据集**

在SJTUTables和RelBench两个公开基准上，覆盖10个数据集、24个任务进行评测。

**📈 对比分析**

与LightGBM、TabPFN、FT-Transformer、TabNet、SAINT、Trompt、ExcelFormer、BRIDGE、RDL、LightRDL、RelGNN等基线相比，InRTL在多数任务上均取得最高或竞争性的准确率/ROC‑AUC，平均排名第一。

**⚠️ 局限性**

仅针对结构清晰、PK–FK关系明确的关系型数据库，难以推广到开放式数据湖或噪声较多的交叉表关系。

---

## 342. Beyond Ambiguous Visual Cues: Studying Physiological Disruptions and Cross-Modal Inconsistencies in Deepfake Videos

**arXiv ID:** 2609.12668 | [PDF](https://arxiv.org/pdf/2609.12668v1)

**作者:** Chenxi Yang `[一作]` (Shanghai Jiaotong University), Larbi Boubchir `[通讯]` (University of Paris 8)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在真实的rPPG数据集（COHFACE、UBFC‑rPPG）上生成面部交换与运动迁移的高保真深度伪造视频，分析伪造如何破坏生理信号与面部行为的关联，并提出双向共注意融合检测器进行检测

**💡 创新点**

① 将深度伪造与同步的生理基线结合，提供可量化的生理干扰分析；② 引入双向共注意机制，在rPPG与面部行为标记之间实现 token 级交互，显著提升检测性能

**🔧 技术方法**

rPPG 特征提取使用 PhysFormer；面部行为特征使用 OpenFace（AU、姿态、注视）；双向多头注意力实现 rPPG 与行为之间的相互增强；分类使用 MLP；实验采用 5 折主题分离交叉验证

**📊 数据集**

COHFACE、UBFC‑rPPG（含真实、面部交换、运动迁移三种伪造），以及在 Celeb‑DF‑v2 上的迁移学习评估

**📈 对比分析**

与单模态（仅 rPPG 或仅行为）及浅层融合（拼接、加权求和）对比，双向共注意在面部交换场景下 AUC 92.80%，在运动迁移场景下 AUC 96.78%；迁移至 Celeb‑DF‑v2 时 AUC 86.08%，显示跨域可行性

**⚠️ 局限性**

① 仅在具备同步生理标注的数据集上可验证，缺乏普适性；② 迁移至无生理数据的真实场景时，rPPG 特征质量下降；③ 只评估了面部交换与运动迁移，未覆盖更复杂的生成器和压缩等因素

---

## 343. Driving Context-guided Model Predictive Planning and Control for Autonomous Car Racing at the Limit and Beyond

**arXiv ID:** 2609.12660 | [PDF](https://arxiv.org/pdf/2609.12660v1)

**作者:** Ayoub Raji `[一作]` (University of Modena and Reggio Emilia), Marko Bertogna `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于成本融合的双MPC（规划+控制）架构，能够在不同驾驶情境（正常、变线、超车、越轨、失控）下自动切换权重，实现在极限条件下的安全、激进驾驶；

**💡 创新点**

创新点在于：①引入成本融合状态机动态调节规划与控制权重，兼顾轨迹追踪与车辆稳定；②采用相同车辆模型的开放式规划MPC与闭环控制MPC，提升模块协同效果；③将侧滑、转向率、轨迹偏差等多维度权重按情境自动化调整；

**🔧 技术方法**

技术包括：基于非线性单轨模型+Pacejka魔方公式的车辆动力学；自定义SQP求解器（HPIPM+CppADCodeGen）；离线前向后向积分生成纵向速度；双MPC频率20Hz/100Hz；成本融合状态机与sigmoid插值；低层PID+前馈实现油门/刹车；传感器融合（LiDAR+RADAR+摄像+GNSS+IMU+侧滑角传感器）与EKF状态估计；运动预测用于超车；

**📊 数据集**

使用真实赛道Yas Marina Circuit的实车数据（Dallara Superformula EAV‑25）进行验证；对比基准为前F1车手的圈速；此外在仿真环境中验证超车成本；

**📈 对比分析**

比较方法：圈速、轨迹偏差、侧滑角、转向角等指标；在普通驾驶下圈速58.76s，差距2%；在超车和失控场景下通过成本切换实现平滑且安全的动作，超车时转向角减半、侧滑角下降；性能表现显示模型在极限驾驶下能维持稳定、快速完成转向；

**⚠️ 局限性**

局限性：对极端三维路面（坡度、颠簸）建模不足，导致高速度轨迹偏差；超车与越轨成本切换仍为手动/规则触发，缺乏自适应学习；未在多车复杂交互与不同车辆类型下验证；对车辆动力学参数的识别误差对规划性能有影响；

---

## 344. Residual Vector-based Reconstruction as Long-Context Recall Regardless of Context Window Size

**arXiv ID:** 2609.12686 | [PDF](https://arxiv.org/pdf/2609.12686v1)

**作者:** MyungHoon Ryu `[一作]` (Korea University), Jong-Kook Kim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在LLM保持冻结的情况下，通过将长上下文中的事实映射到模型前馈层的激活向量（残差向量）并外部存储的方式，实现长文本的选择性回忆；

**💡 创新点**

创新点在于利用前馈层的激活作为“地址”，将事实编码为可检索的残差向量，做到在不增加模型权重、不需额外训练的前提下，维持近乎常数的GPU内存使用并支持超越预训练窗口的长文本回忆；

**🔧 技术方法**

核心技术包括：① 基于预训练LLM的固定前馈层激活键值记忆；② 对每条事实进行残差向量优化（仅在该激活位置注入）；③ 采用查询驱动的路由与余弦门控检索；④ 分两阶段（记忆与响应）实现无训练记忆构建；

**📊 数据集**

实验使用了三个公开大模型（Phi‑3.5‑mini、Qwen3‑4B、Llama‑3.1‑8B）在BABILong与RULER QA等长文本问答基准上进行评测；

**📈 对比分析**

与压缩、截断、Chunk‑RAG、Streaming‑LLM等无训练方法对比，本文方法在2 M token规模下单事实问答准确率仍能保持50–80%（其余方法接近0%），且GPU内存几乎不随上下文长度增长；

**⚠️ 局限性**

局限性包括：① 记忆构建与每条事实的残差优化耗时；② 需要对每个检索问题执行多次前馈生成，导致推理延迟；③ 对事实提取、路由、以及多事实组合仍存在失败，尤其在复杂关系或非显式实体场景；③ 现有实现仅在实验环境下验证，缺乏对隐私与实时交互的评估。

---

## 345. Improving Imitation Learning Efficiency for Manipulation through Geometric Prior Pretraining

**arXiv ID:** 2609.12721 | [PDF](https://arxiv.org/pdf/2609.12721v1)

**作者:** Shogo Iwakata `[一作]` (Waseda University), Yukiyasu Domae `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种自动生成纯几何场景（仅包含平面、物体和手）并用于预训练的框架，随后在少量真实演示下微调 ACT 视觉运动政策，以提高新操作任务的样本效率。

**💡 创新点**

创新点在于：① 用极简的几何场景仅关注手与物体的相对姿态，消除了纹理、背景等无关信息；② 把手表示为立方体，使预训练与任何机器人形态无关；③ 通过自动生成的 pick、place、push 动作序列提供结构化、可重复的训练信号；④ 仅使用 600 条合成演示即可显著提升多机器人、多任务的收敛速度和成功率。

**🔧 技术方法**

技术手段包括：ACT 视觉运动学习框架、点云渲染、SLERP 旋转插值、Bezier 曲线运动规划、自动化演示生成与手姿态标注。

**📊 数据集**

数据集：1) 纯几何合成数据集（600 条演示，200 条 pick、200 条 place、200 条 push）；2) 真实演示数据用于微调（模拟 20 条/任务，真实 30 条/任务）。

**📈 对比分析**

与基线的比较：① 训练从零（Scratch）；② 预训练后微调；③ 同环境预训练。实验在三台机器人（UR5e、xArm7、KinovaGen3）和五个模拟任务、三个人机任务中进行。结果显示，预训练后在前 1k–2k 轮内成功率显著提升，平均峰值成功率提高约 8–10%，且在多任务平均上优于同环境预训练。实机实验同样证明了加速收敛和更高峰值成功率。

**⚠️ 局限性**

局限性：① 仅提供相对几何先验，无法处理复杂遮挡、动态摩擦或多物体交互；② 需要手的立方体假设，对非立方手模型或多手系统的迁移受限；③ 只验证了 ACT 政策，是否对其它视觉运动框架同样有效尚未确认；④ 预训练对单目标场景有效，缺乏对杂乱环境的鲁棒性。

---

## 346. Understanding Game Coaching on Gig Platforms

**arXiv ID:** 2609.12695 | [PDF](https://arxiv.org/pdf/2609.12695v1)

**作者:** Hwijoon Lee `[一作]` (Northeastern University), Saiph Savage `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对20名在Fiverr上提供游戏指导的自由职业教练进行半结构化访谈，分析其工作流程、挑战及AI工具使用情况，提出双重不稳定性与“earned authority”两种结构性条件；

**💡 创新点**

首次将自由职业游戏教练工作与平台劳动、游戏生命周期不稳定性结合，提出双重不稳定性与通过在游戏中持续取得成就获得的earned authority概念，并探讨AI在此情境中的可行与限制；

**🔧 技术方法**

采用半结构化访谈与反思性主题分析（RTA）对文本进行编码与主题归纳；

**📊 数据集**

包含20名经验丰富教练（共17款游戏）的访谈转录文本，结合其公开的Fiverr个人资料信息；

**📈 对比分析**

研究为定性描述性研究，没有对照实验或性能评估，主要通过主题归纳呈现研究发现；

**⚠️ 局限性**

样本仅来自Fiverr，主要以英语为主、性别单一偏向男性、并限制在已具一定可见度的教练，缺乏对初学者或非Fiverr平台教练的视角，结果的普适性受限；

---

## 347. Implicit Personality Representations in Humans and LLMs

**arXiv ID:** 2609.12704 | [PDF](https://arxiv.org/pdf/2609.12704v1)

**作者:** Yilin Geng `[一作]` (University of Melbourne), Lea Frermann `[通讯]` (University of Melbourne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究大语言模型内部表示与人类性格结构的相似性，利用对比提示生成场景提取trait向量，并与大规模人类角色评分数据对齐。

**💡 创新点**

首次构建基于对比生成的内部trait方向，结合数千万人类评分，系统性比较LLM内部结构与人类隐性人格结构，发现两者高度一致，并提取出社会温暖与智力竞争的两主轴。

**🔧 技术方法**

对比提示、内部激活向量提取、PCA降维、Mantel检验、Pearson相关、LLM-as-a-judge过滤等技术。

**📊 数据集**

SWCPQ（77.4M评分、414 trait）人类评分数据，Qwen 2.5-7B-Instruct内部激活，以及Wikiquote对held‑out角色的对话文本。

**📈 对比分析**

通过Mantel检验得到 r≈0.765（整体）/0.708（去同义词后），每trait r_T中位数≈0.80；PCA两主轴解释59%变异；对held‑out对话的性格推断平均相关≈0.33-0.4，显示能较好恢复人格。

**⚠️ 局限性**

仅在单一模型上验证，结构比较不具因果性；LLM-as-a-judge评分分辨率有限；对话文本中的表面语气与角色真实属性不一致时导致误差。

---

## 348. Supermartingale Certificates for Parametric MDPs

**arXiv ID:** 2609.12715 | [PDF](https://arxiv.org/pdf/2609.12715v1)

**作者:** Kaushik Mallik `[一作]` (IMDEA Software Institute), Ðorđe Žikelić `[通讯]` (Nanyang Technological University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在一般可测状态和动作空间下参数化马尔可夫决策过程（pMDP）的形式验证与近似合成问题，提出了一种参数扁平化变换，将 pMDP 转化为等价的非参数 MDP，并基于此构造了参数超martingale（如参数 SBF 与 RASM）证书，用以自动推导满足概率阈值的可测策略。

**💡 创新点**

创新点包括：① 参数扁平化变换首次将 pMDP 与普通 MDP 等价对应；② 引入参数超martingale 框架，首次把已存在的超martingale 证书推广到 pMDP；③ 在验证与合成算法中直接生成闭式参数化策略，首次实现参数化策略的符号表达；④ 通过模板化多项式求解与 SMT 结合，提供了一套可实现的完整算法流程。

**🔧 技术方法**

主要技术手段为：可测概率理论、参数扁平化映射、超martingale 证明技术、模板化符号多项式约束构造、SMT 求解（MathSAT5、Z3）以及多项式约束求解工具 PolyQEnt。

**📊 数据集**

实验基准来自连续随机游走模型，构造了三种 pMDP 版本：M⁺（单一加性参数）、M×（单一乘性参数）和 M⁺,×（双参数），每种模型均在参数空间 [-1,1]² 及相应子空间上进行验证与合成。

**📈 对比分析**

实验中将两种 SMT 求解器 MathSAT5 与 Z3 进行比较。对于 M⁺ 与 M⁺,×，MathSAT5 显著优于 Z3；而在 M× 情况下，Z3 表现更好。表格展示了不同逼近参数 c 下的总耗时、单次 SAT 调用时间、SMT 调用次数等指标，说明算法在合理精度下能在数十到数百秒内完成近似合成；图示表明分辨率在边界附近更细，合成结果与理论预期一致。

**⚠️ 局限性**

局限性包括：① 仅适用于可用多项式实数算术描述的 pMDP；② 对高维状态空间与参数空间的可扩展性仍有限，求解时间随维度指数增长；③ 需要已知分布的前 D 个矩，无法处理未知分布的高阶统计信息；④ 当前只针对概率阈值问题，尚未覆盖 ω-正则等更复杂性质；⑤ 合成结果为近似，无法保证完全覆盖整个参数空间。

---

## 349. SIFPBPNet: A Dual-Path Network for Wearable and Cuffless Blood Pressure Estimation via Individualized Steady-state Representation

**arXiv ID:** 2609.12690 | [PDF](https://arxiv.org/pdf/2609.12690v1)

**作者:** Shuailong Tang `[一作]` (Shenzhen Technology University), Yali Zheng `[通讯]` (Shenzhen Technology University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种双路径网络SIFPBPNet，用稳态和瞬时特征路径实现可穿戴无袖压计血压估计。

**💡 创新点**

创新点在于引入稳态特征路径通过图注意力网络学习个体长期PPG特征，将个体稳态先验融入瞬时估计，从而解决“一对多映射”问题。

**🔧 技术方法**

采用多尺度卷积+Transformer的PPG编码器、Graph Attention Network (GAT)、交叉注意力融合以及多任务预训练等技术。

**📊 数据集**

使用大规模OPPO HBPM可穿戴数据集，包含790名受试者、10天历史PPG和随访血压。

**📈 对比分析**

与ResNet18、VGGNet、BiLSTM、SEM-ResNet等基线对比，SIFPBPNet在SBP、DBP的MAE分别达到8.57 mmHg、5.98 mmHg，显著优于基线。

**⚠️ 局限性**

局限性包括需预先收集长达10天的历史PPG、模型对缺失历史数据的鲁棒性未知，以及对不同PPG窗口长度的适应性待进一步验证。

---

## 350. Detecting and Explaining Fake News Short Videos with Multimodal Content and Real-World Evidence

**arXiv ID:** 2609.12678 | [PDF](https://arxiv.org/pdf/2609.12678v1)

**作者:** Yifeng Luo `[一作]` (Hong Kong Baptist University), Liang Lan `[通讯]` (Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究短视频假新闻检测与解释任务，提出统一的NVKE-CEI系统；

**💡 创新点**

创新点在于先通过NVKE基于视觉+OCR相似度的时间变动关键帧提取保留语义与时序信息，再通过双向LLM（内容检验器与证据检验器）并用轻量级判断器融合内容与外部证据，显著降低幻觉并提升解释质量；

**🔧 技术方法**

使用CLIP视觉/文本嵌入、OCR、双LLM（GPT‑4o）结合ReAct框架、查询重构、检索增强生成（RAG）、轻量级注意力+融合判别器等技术；

**📊 数据集**

在FakeSV和FakeTT两个公开假新闻视频数据集上进行实验；

**📈 对比分析**

与13个基线（含7个零样本LLM基线和6个监督训练基线）比较，NVKE‑CEI在Acc.和Macro F1上均实现SOTA，分别提升0.0071/0.0234，解释质量评估G‑Eval亦优于其它方法；

**⚠️ 局限性**

局限性包括：关键帧提取可能遗漏渐进式变化的线索；LLM在出现一致错误时难以纠正；对收费API和实时搜索的依赖导致延迟与成本波动；解释质量评估仅依赖单一LLM无人工验证。

---

## 351. Doc2FRC: Length-Consistent Document-Level Machine Translation via Fixed-Range Chunking

**arXiv ID:** 2609.12674 | [PDF](https://arxiv.org/pdf/2609.12674v1)

**作者:** Xiaotian Wang `[一作]` (University of Tokyo), Hitomi Yanaka `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了固定范围分块（Fixed‑Range Chunking，FRC）和双边界匹配算法，用于在文档级机器翻译中统一训练和推理的长度分布，并设计了多种训练策略与轻量级模型部署方案。

**💡 创新点**

创新点在于利用动态规划实现长度区间内的分块，使得训练与推理时的文档长度分布完全一致；提出双边界匹配实现大规模语料的高效对齐；以及通过 FRC 训练的中型 LLM 在 OOD 文档上可与大规模商业 LLM 对抗。

**🔧 技术方法**

主要技术包括动态规划分块、双边界匹配、文档级对齐、Sep/Stein/Uni 多格式训练、LoRA 微调、位置嵌入增强（RoPE、SHAPE、Linear PI）以及 d‑BLEU、d‑COMET、LTCR 与 GEMBA‑DA 等评估指标。

**📊 数据集**

使用 DocBlocks 语料库（IWSLT、BWB、GuoFeng、News Commentary、Europarl）进行训练；使用 IWSLT2017、BWB 作为基准评测；并构建了全新 10 语言 OOD 文档集 GlobVDoc。

**📈 对比分析**

通过与直接 doc‑to‑doc 微调、现有 DocMT 方法、以及 GPT‑4.1、DeepSeek‑v3.2、Gemini‑2.5‑Pro 等商业 LLM 以及 GRAFT、DELTA 等 agent‑based 方法进行对比，FRC‑based 模型在 IWSLT 上平均提升约 12 BLEU 点，能够匹配或超过大规模 LLM；在 GlobVDoc 上仍略低于 GPT‑4.1，但显著优于同类小模型。

**⚠️ 局限性**

局限性包括未使用 GuoFeng 测试集（因与公开评测重叠）、仅在单一 backbone（Tower‑7B / Qwen2.5‑7B）和固定训练设置下验证、对训练与部署的计算成本高、对上下文选择未深入探讨，以及部分方法的对比受限于缺失的公开模型参数。

---

## 352. Deterministic NC Quadratic Root Counting in Characteristic Two

**arXiv ID:** 2609.12669 | [PDF](https://arxiv.org/pdf/2609.12669v1)

**作者:** Sanyam Agarwal `[一作]` (Saarland University), Gorav Jindal `[通讯]` (University of Regensburg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

为给定固定特征为 2 的有限域上的单个二次多项式，设计了一个确定性、logspace‑uniform 的并行算法，能够精确计数满足方程的解的个数。

**💡 创新点**

创新点在于：①利用绝对迹把二次形式映射到二元域；②通过 Browder 的判据用模 8 的行列式确定 Arf 不变量，从而避免了之前方法中随机构造二元分解；③将所有可能的秩分支一次性并行处理，最终只需根据秩选择正确的计数值。

**🔧 技术方法**

采用的技术包括：有限域的线性代数（秩、基、核、逆矩阵）、绝对迹与限制-扩张、Arf 不变量与签名求和、行列式模 8 的判定、齐次化（将含线性项的方程转为齐次方程）以及对数空间统一的布尔电路实现。

**📊 数据集**

该工作没有使用任何外部数据集；所有算法和证明均基于代数理论与电路构造，输入仅为多项式系数和目标值。

**📈 对比分析**

与之前的随机化并行算法相比，所提出的方法在相同的多项式时间与 polylog 深度下实现了完全确定性；电路大小为多项式、深度为 logⁿ¹ⁿ；此外不再需要二进制分解或随机数生成器，提升了算法的可实现性和可验证性。

**⚠️ 局限性**

局限性包括：①仅适用于特征为 2 的固定有限域；②算法实现依赖于固定域的编码与常数时间操作，若要推广到可变域或更高次数多项式，需进一步研究；③虽然在理论上是 logspace‑uniform，但在实际硬件上实现时仍需高效的并行线性代数子程序。

---

## 353. Separating Engineering Reasoning from DEXPI Serialization in LLM-Based Greenfield Surface-Process Design: A Three-Case Study for Underground Gas Storage

**arXiv ID:** 2609.12656 | [PDF](https://arxiv.org/pdf/2609.12656v1)

**作者:** Qingchuan Zhu `[一作]` (Sinopec Petroleum Engineering Zhongyuan Co Ltd), Pengju Ren `[通讯]` (Xi'an Jiaotong University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在三种地下天然气贮存场景下，对比了直接生成 DEXPI 2.0 XML 与先生成轻量化工程中间表示（IR）后再序列化的两种方法。

**💡 创新点**

创新点在于将工程推理与标准级序列化分离，观察其对表示负担、错误模式与工程可行性影响，并首次将两种方法在相同 LLM 版本下进行对照评估。

**🔧 技术方法**

使用大型语言模型 qwen3.8‑max‑0902、提示工程、XML XSD 验证、JSON 结构校验以及人工工程审查。

**📊 数据集**

数据集为三条限定的地下气储设计任务（单压注入、抽取与出口、双压注入），共六个模型输出。

**📈 对比分析**

对比方法包括表示有效性（XSD/IR 结构校验）、工程可行性（接受/警告/拒绝）和方法内部结果；IR 方式显著减少了提示词（约 175‑200 倍）和生成延迟，但工程可行性变化与直接 DEXPI 相当，且仍出现冷却状态冲突。

**⚠️ 局限性**

局限性包括仅使用单一模型快照、每种方法仅一次生成、方法包的变量无法单独归因、未实现 IR→DEXPI 的确定性转换以及缺乏完整的物性/过程模拟验证。

---

## 354. Distortion of AI Alignment Revisited: RLHF is a Decent Utilitarian Aligner

**arXiv ID:** 2609.12651 | [PDF](https://arxiv.org/pdf/2609.12651v1)

**作者:** Kazusato Oko `[一作]` (University of California, Berkeley), Han Bao `[通讯]` (Institute of Statistical Mathematics and Graduate University for Advanced Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对RLHF在多样化用户环境中的失真问题进行细粒度分析，证明失真主要由贝塞尔-泰勒模型的非线性与数据分布与参考策略的失配导致；

**💡 创新点**

创新点在于给出了RLHF失真的上下界（Θ̃(Bβ)）并证明当μ=π_ref时失真可降至O(β)，从而破解先前指数上界的误区，并提出通过在RLHF前对μ进行微调来缓解失配；

**🔧 技术方法**

主要技术包括有效效用（effective utility）变换、奖励剪裁、KL约束分析、Pinsker不等式与信息理论工具；

**📊 数据集**

实验使用公开奖励模型Skywork-Reward-V2-Llama-3.1-8B与UltraRM-13B的偏好数据（分别为Skywork-Reward-Preference-80K-v0.1和UltraFeedback），并设计了合成实验验证分布失配导致的失真；

**📈 对比分析**

与传统的无KL约束RLHF或单一BT模型对比，理论上与实验表明失真可从O(β)提升到O(Bβ)；若使用μ与π_ref匹配，则失真仅为O(β)，表现优于先前的指数失真；

**⚠️ 局限性**

局限性包括对奖励剪裁的依赖、对离线偏好数据的处理仍不完备、实验规模有限、对真实多模态数据的泛化尚待验证。

---

## 355. Online Material Estimation for Conditioned Diffusion Policy in Shaping Deformable Linear Objects

**arXiv ID:** 2609.12634 | [PDF](https://arxiv.org/pdf/2609.12634v1)

**作者:** Ryunosuke Yamada `[一作]` (Kanazawa University), Tokuo Tsuji `[通讯]` (Kanazawa University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在形状控制的可变形线性物体任务中，提出一种在线估计材质标签并作为条件输入的扩散策略。

**💡 创新点**

将材质估计与扩散策略结合，无需先验材质信息，且在线持续更新；首次实现无模拟器、无探测动作的材质条件化控制。

**🔧 技术方法**

利用ResNet18编码视觉输入、LSTM+注意力估计材质、基于U-Net的扩散策略。

**📊 数据集**

收集480条真实机器人演示，覆盖四种材质（硅胶、厚亚克力、薄亚克力、棉布）和三种槽位放置任务。

**📈 对比分析**

通过与专属策略、仅任务条件化、地面真材质条件化四种策略比较，估计策略在所有组合上平均成功率60.8%，接近Oracle（60.0%）并显著优于Task-only（45.8%）和Specialist（41.7%）。

**⚠️ 局限性**

估计器在相似材质间易混淆（如硅胶与薄亚克力），且仅使用离散材质标签，未对未知材质进行泛化。

---

## 356. LifeMem: Enabling Lifelong Experience Reuse for LLM Agents

**arXiv ID:** 2609.12655 | [PDF](https://arxiv.org/pdf/2609.12655v1)

**作者:** Yuli Qiu `[一作]` (Beijing Institute of Technology), Yuang Guo `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 LifeMem 框架，利用工作流聚类和本地技能蒸馏实现大语言模型代理在跨异构环境中的终身学习与经验重用。

**💡 创新点**

创新点在于：①根据潜在动作工作流对轨迹进行聚类，避免语义相似导致的跨环境干扰；②在每个工作流内局部蒸馏技能，消除跨环境噪声并提升技能可迁移性；③引入任务流缓冲机制提升学习稳定性。

**🔧 技术方法**

技术手段包括：基于 MRL 的轨迹相似度度量与动态聚类；使用 GPT‑4.1 与 GPT‑5‑mini 进行技能抽取与聚类决策；利用 FAISS 进行检索，SQLite 存储聚类信息；MiniLM‑L6‑v2 作为嵌入模型。

**📊 数据集**

使用了 10 个环境、13,000+ 任务以及 2,300+ 手工或 GPT‑4.1 生成的交互轨迹，涵盖 AlfWorld、ScienceWorld、APIBank、τ‑bench、HotpotQA、Webshop、LifelongAgentBench、ToolQA、Miniwob++、Mind2web 等数据集。

**📈 对比分析**

与 ReAct、Synapse、ExpeL、AutoSkill 等基线对比，LifeMem 在五个代理场景下实现了整体性能提升（如 GPT‑4o‑mini 从 28.84% 提升至 44.13%），并显著改善了向后迁移（负向遗忘减少）。

**⚠️ 局限性**

局限性主要在于对大规模经验库（百万级）扩展性不足，未来需考虑轻量化 LLM、近似检索及系统级优化以实现工业级部署。

---

## 357. Physics-Guided Synthetic High-Frequency Ultrasound Generation for Skin Layer Segmentation

**arXiv ID:** 2609.12735 | [PDF](https://arxiv.org/pdf/2609.12735v1)

**作者:** Junkyung ju `[一作]`, Minwoo Shin `[通讯]` (Yonsei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

建立物理引导的多层声学皮肤模型与k‑Wave模拟框架，生成带有密集标签的高频超声图像用于皮肤层分割。

**💡 创新点**

通过声学层映射驱动的生成，实现八类（air、coupling、epidermis、SLEB、dermis、subcutaneous、fascia、muscle）密集标注，且可控制层厚、边界形态与纹理，提供可迁移的结构与纹理先验。

**🔧 技术方法**

使用k‑Wave时间域声学仿真、层厚/边界/纹理可控变异、绿通道渲染、Label‑preserving augmentation、以及四种分割网络（Fresh SegUNet、U‑Net、DeepLabV3+、SegFormer）。

**📊 数据集**

合成数据集共200个样本（含多变异），以及公开的Mendeley HFUS皮肤数据（仅epidermis/SLEB标注）用于真实域评估。

**📈 对比分析**

先在合成数据上预训练，再在真实数据上微调，与仅用真实数据训练进行对比。合成预训练后，三种可训练架构平均Dice提升至83.40%，平均IoU提升至71.79%，与真实训练相近。

**⚠️ 局限性**

合成图像与真实扫描在散斑、衰减、反射等统计上仍存在差距，导致增益有限；样本数量受限于昂贵的波方程仿真；缺乏深层结构的真实标注。

---

## 358. The Mechanics of a Swarm: A Reproducible External Reconstruction of an Unintended Agent-Coordination Episode on a Third-Party Wiki

**arXiv ID:** 2609.12748 | [PDF](https://arxiv.org/pdf/2609.12748v1)

**作者:** Philipp Lütje `[一作]` `[通讯]` (Philflow), Philipp Lütje (Philflow)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对OpenAI语言模型代理在公共维基上的写作记录进行外部重建与行为分析

**💡 创新点**

在缺乏内部日志的情况下，仅用公共改动档案重构协同过程、时钟异质性和代理元数据，并量化协同与进展无关联的结论

**🔧 技术方法**

正则表达式文本抽取、占位符模型、泊松过程/Hawkes过程、占用率估计、主成分分析与聚类、统计检验

**📊 数据集**

2026‑05‑24至07‑02的维基改动历史共14591条，3103名代理，4579页，19913个服务器事件

**📈 对比分析**

通过对比自报时钟与服务器时间、协同指标与进度的Spearman/OLS，聚类与回归均未发现显著正相关，说明协同与进展无显著关联

**⚠️ 局限性**

缺少读取、反馈与真实结果日志，导致信息传递与协同成效无法验证；样本量有限、标注误差与偏差影响分析精度

---

## 359. What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework

**arXiv ID:** 2609.12746 | [PDF](https://arxiv.org/pdf/2609.12746v1)

**作者:** Ioannis Prokopiou `[一作]` (Athens University of Economics and Business), Pantelis Vikatos `[通讯]` (Orfium)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估一个名为LAST-CQ的五代理、训练免费、执行驱动的文本到Cypher生成框架，使用对执行结果的检测与迭代校正实现高恢复率。

**💡 创新点**

通过对框架的逐步剔除实验明确发现，提升性能的关键在于失败检测与重试路由，而非复杂反馈生成或采样数量；提供了可单独移除的组件化设计与订单不变的评估方法。

**🔧 技术方法**

利用LLM生成器（GPT‑4o、Gemini‑Pro、Claude‑Sonnet4等）、语法解析器、预执行验证、错误提示生成与执行重试等技术，整体架构实现多轮验证与纠错。

**📊 数据集**

使用Neo4j Text‑2‑Cypher可执行子集（2471问答对，覆盖16个领域）作为真实数据库查询基准。

**📈 对比分析**

与单通道生成、无精炼、最佳‑of‑3采样等对照实验比较，LAST‑CQ+DB对单通道失败的恢复率达91.7%，总体BLEU提升3.4%，但若去除校正仅降幅3.1–12.3%，去除schema‑grounded反馈仅差1个百分点，采样导致10–11%性能下降。

**⚠️ 局限性**

局限性包括仅在Neo4j单一引擎和单一基准上测试；部分评估指标（如集合精确度）仅在子集上计算；缺乏对其他图数据库的验证；模型间对比受退役模型替换影响；人类标注样本有限，评判者乐观倾向未完全校正。

---

## 360. AquaCubeAI-Powered Monitoring Turbidity on-board Φsat-2

**arXiv ID:** 2609.12744 | [PDF](https://arxiv.org/pdf/2609.12744v1)

**作者:** Pietro Di Stasio `[一作]` (University of Sannio), Silvia Liberata Ullo `[通讯]` (University of Sannio)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套在 Φsat-2 CubeSat 上的轻量级 AI 模型 AquaCubeAI，用于即时估计沿海水体浊度并生成异常掩模。

**💡 创新点**

创新点包括：引入空间块划分的泄漏意识数据分割、使用 CMEMS HR-OC 浊度标签构建多区域合成数据集、将 MLP 重新构造为 1×1 卷积实现稠密预测，以及在 Myriad VPU 上实现低延迟嵌入式推理。

**🔧 技术方法**

技术手段涵盖轻量级多层感知机（MLP）与 1×1 卷积、Huber 损失、Adam 优化器、OpenVINO+ONNX 推理、模拟 Φsat-2 多光谱数据、CMEMS 高分辨率海洋颜色产品。

**📊 数据集**

数据集为合成的 Φsat-2 多光谱图像与 CMEMS HR-OC 浊度标签配对，覆盖欧洲四大海域（地中海、黑海、伊比利亚海岸、西北海板块）多区域局部站点，包含 2023 年多个月份数据及 2024 年外部保留集。

**📈 对比分析**

与先前单站点基线模型和不同 MLP 结构进行对比；在空间块划分下选定 (512,512,256) 模型在 0–140 FNU 的 RMSE 2.78 FNU、MAE 1.41 FNU、R² 0.995；在 0–40 FNU 范围内 RMSE 1.39 FNU、MAE 0.86 FNU；外部 2024 保留集 RMSE 1.59 FNU、MAE 0.95 FNU；在 Myriad VPU 上平均推理时延 8.51 ms/块，支持 117.49 fps。

**⚠️ 局限性**

局限性包括：仅使用模拟数据，缺少真实轨道观测验证；依赖 CMEMS 参考产品，可能带来标签不确定性；对高浊度极值和时间迁移的鲁棒性有限；模型对子像素空间信息的利用不足，未提升更细粒度精度。

---

## 361. When Rubrics Fail: Hallucinations Reveal Blind Spots in Medical AI Evaluation

**arXiv ID:** 2609.12718 | [PDF](https://arxiv.org/pdf/2609.12718v1)

**作者:** Griffin Farrow `[一作]` (University of Oxford), Fabio J. Fehr `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了医学领域中基于评分表（rubric）的LLM评估方法对临床相关幻觉（hallucination）的敏感性，并提出了一个临床验证的幻觉分类法与控制性错误注入管道。

**💡 创新点**

创新点在于系统揭示了评分表在检测无法预见的临床错误时的盲点，并证明了检索驱动的事实性检查可以补充评分表的不足；同时构建了可复现的错误注入流程与临床专家验证的幻觉类型表。

**🔧 技术方法**

采用了多种技术：评分表评估（包括专业评审的HealthBench、HealthBench Professional、LiveMedBench）、LLM判分器（如Qwen3-14B、Llama3-70B、GPT‑4.1）、错误注入模型（Gemini‑3.1‑Flash‑Lite）、检索增强事实性评估器（SAFE）以及程序化和LLM质量检查。

**📊 数据集**

使用了MedHallu、HealthBench、HealthBench Professional、LiveMedBench四个公开医学对话/问答数据集；在MedHallu中对已标注的幻觉对进行基线评估，在HealthBench等Benchmark中对注入错误的响应对进行实验。

**📈 对比分析**

比较方法主要是“配对AUROC”和“赢/平/输”统计。结果显示：在HealthBench中配对AUROC最高仅为0.599，HealthBench Professional接近0.5，评分表在多数幻觉类型上几乎无法区分正确与错误答案；检索驱动的评估在大部分盲点幻觉（如证据伪造、剂量错误、额外诊断）中能检出超过80%的错误，但也伴随约55%的误报。

**⚠️ 局限性**

局限性包括：仅使用单一判分器；评估仅覆盖约500道HealthBench题目，未覆盖完整数据集；错误注入和原始回答均使用单一生成模型；实验使用人工注入的错误而非自然发生的幻觉；检索评估依赖通用网络搜索，准确性受限。

---

## 362. Write on Paper and Get the Online Digital Trace:\newline A New Era for Handwriting

**arXiv ID:** 2609.12702 | [PDF](https://arxiv.org/pdf/2609.12702v1)

**作者:** Florent Imbert `[一作]` (Université de Rennes), Peter Kampf `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一款可在纸张上写字并通过内部IMU传感器实时捕捉笔迹轨迹的电子笔（Digipen），并将其与嵌入式AI模型相结合，实现纸上书写轨迹的数字化重建。

**💡 创新点**

①通过混合专家模型（MOE-CI）结合时序卷积网络，专门针对纸张和屏幕两种介质的IMU信号进行训练；②使用硬件感知的神经网络压缩技术，将模型参数大幅减少至可在微控制器上实时推理的规模；③采用双重采集方法（同步纸上书写与EMR记录）构建高质量训练数据。

**🔧 技术方法**

IMU（加速度计、陀螺仪）传感；Kalman滤波预处理；Dynamic Time Warping 对齐；深度学习（Temporal Convolutional Network、Mixture of Experts）；硬件感知NAS与NASWOT零成本代理；BLE低功耗通信。

**📊 数据集**

利用配备双重记录功能的实验Digipen，获取同步的纸上书写轨迹与IMU数据，结合Wacom录制的EMR轨迹作为地面真值；数据集覆盖笔触（触笔）与笔抬起（非触笔）两种状态。

**📈 对比分析**

使用Fréchet距离评估重建轨迹与真值的相似度。实验表明：①仅在平板上训练的模型对纸上数据表现差（Fréchet 0.738）；②在纸张数据上训练显著提升（Fréchet 0.429）；③对平板数据应用Kalman滤波提升有限；④压缩后模型在纸上依旧保持较好性能（Fréchet 0.566），但对笔抬起运动的重建略逊。

**⚠️ 局限性**

限制主要包括：①不同介质（纸张 vs 平板）导致的噪声差异仍难以完全弥补；②压缩模型在笔抬起重建上性能下降；③实验数据规模受限，仅覆盖有限写字者与书写风格；④需要进一步验证在更广泛的书写任务与低功耗硬件上的实用性。

---

## 363. Semantically Aligned Gradient-Driven Context-Preserving Image Editing

**arXiv ID:** 2609.12691 | [PDF](https://arxiv.org/pdf/2609.12691v1)

**作者:** Chiranjeev Chiranjeev `[一作]` (Indian Institute of Technology Jodhpur), Richa Singh `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IABEdit，一种在训练阶段通过冻结视觉语言模型提取语义描述并将其残差作为梯度，进行语义对齐监督的通用图像编辑框架。

**💡 创新点**

创新点在于：1）将语义验证嵌入梯度训练，使编辑器能主动校验指令满足度；2）实现无推理成本、模型无关的对齐；3）通过梯度对齐实现精准局部编辑和全局结构保持。

**🔧 技术方法**

采用冻结的 LLaVA/Qwen‑VL 视觉语言模型做语义监督；利用 LoRA 适配的可训练对齐器与 MLP 结构投影器；在扩散或流匹配（Stable Diffusion、FLUX.1）等生成器上进行梯度传递；对人脸编辑加入身份保持损失。

**📊 数据集**

使用 Super‑40K 作为训练集；在 RealEdit、MagicBrush、EMU Edit、D‑LORD 等公开基准上评测；同时对 Gemini‑AI 代理在 D‑LORD 上进行对照。

**📈 对比分析**

与 InstructPix2Pix、UltraEdit、SuperEdit、FLUX.1、BAGEL 等方法比较，IABEdit 在 CLIP‑T、CS‑P、DINO‑I、HM 等指标上均取得领先；在 D‑LORD 任务上超过 Gemini‑AI，提升 DINO‑P 5.13 点，展示了更强的身份保持与指令遵循。

**⚠️ 局限性**

局限性包括：训练过程依赖大型冻结 VLM，计算和内存成本高；对极端遮挡下的指令理解仍有限；仅在训练时提供监督，推理仍需依赖原始生成器；缺少对视频或3D 等多模态延展的实验。

---

## 364. Personalized and Trust-Aware Health Recommendation Policies for a Construction Workplace

**arXiv ID:** 2609.12679 | [PDF](https://arxiv.org/pdf/2609.12679v1)

**作者:** Atefeh Mollabagher `[一作]` (University of California San Diego), Parinaz Naghizadeh `[通讯]` (University of California San Diego)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一个考虑工人健康与信任动态交互的建模框架，并在此基础上设计了两种推荐策略：基于短期模型预测控制的阈值与频率调度以及基于深度 Q‑网络的长期强化学习策略；

**💡 创新点**

创新点在于将工人对健康推荐的信任视为可随时间演化的内部状态，并通过联合建模健康、信任与合规性来推导可解释的阈值选择规则，同时验证了长期 RL 能更好地利用信任演化提升健康与生产率平衡；

**🔧 技术方法**

技术主要包括离散时间马尔可夫决策过程建模、短期两步规划的解析解、以及基于 PyTorch 的 DQN 强化学习实现；

**📊 数据集**

数据集方面使用了基于论文所给参数（H̅=1, κ₁=0.7, κ₂=1.2, λ=0.5, δ_P=0.2, γ=0.95）的仿真数据，没有真实工人传感器数据；

**📈 对比分析**

比较方法是对同一 MDP 采用短期规划和 RL 两种策略，在相同的随机起始状态下分别训练并评估累计奖励、推荐频率与阈值选择以及信任和健康随时间的演化；实验结果表明 RL 在大多数用户类型上获得更高累计奖励，短期规划在高信任/健康敏感型用户上表现相近；

**⚠️ 局限性**

局限性包括：模型假设健康与信任完全可观测，未考虑噪声与不可观测因素；推荐仅为单一行为，缺乏多样化；仿真参数为人工设定，缺乏真实工人试验验证；未来工作需扩展至 POMDP、引入真实感应数据并在实验室环境中检验。

---

## 365. Control Architecture for Safe Grasping of Fragile Objects Using a Coarse Position-Controlled Gripper

**arXiv ID:** 2609.12737 | [PDF](https://arxiv.org/pdf/2609.12737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 366. Breaking the Vision-Action Shortcut: Latent Interface Training for Generalizable Robotics Foundation Models

**arXiv ID:** 2609.12641 | [PDF](https://arxiv.org/pdf/2609.12641v1)

**作者:** Jianman Lin `[一作]` (South China University of Technology), Jiafei Duan `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Latent Interface Training (LIT)，通过两阶段训练先在无图像条件下学习空间目标驱动的动作先验，再用姿态监督的潜在接口将视觉信息单独调度给动作专家，以减少视觉-动作捷径。

**💡 创新点**

创新点在于：①将视觉条件完全隔离在一个姿态监督的潜在接口中，②通过空间目标先验实现无图像的动作预训练，③两阶段训练结构结合姿态监督，显著提升视觉分布迁移下的泛化能力。

**🔧 技术方法**

使用了预训练视觉语言模型（VLM）或视频模型作为骨干、流匹配或扩散式动作生成器、姿态重构的 MLP 解码器以及跨注意力的潜在标记；核心技术是潜在接口（latent interface）与姿态监督（pose supervision）。

**📊 数据集**

在模拟中使用 LIBERO 及其对抗性扩展 LIBERO-Plus 数据集；在真实机器人上使用三项基于演示的任务（拆块、清扫、转移鸡蛋），共 300 条演示数据。

**📈 对比分析**

与同一架构下的基线模型（π_0.5、MolmoAct2、FAST-WAM、ImageWAM）对比；在 LIBERO 上保持或提升平均成功率（最多提升 1.7%），在 LIBERO-Plus 上整体提升 3.87–10.70%，在真实机器人上 ID 成功率提升 13.3–16.7%（累计 14.3%）。

**⚠️ 局限性**

局限性包括：实验规模仍相对有限，未在更大多样的真实环境和更广泛任务上验证；LIT 需要两阶段训练，额外的超参数和计算成本；对极端视觉扰动（如遮挡或极端光照）仍需进一步评估。

---

## 367. Generative AI Use Cases In Real Estate Marketing: Adoption and Constraints in Germany

**arXiv ID:** 2609.12684 | [PDF](https://arxiv.org/pdf/2609.12684v1)

**作者:** Victor Kolominsky-Rabas `[一作]` (University of Bayreuth), Niklas Kühl `[通讯]` (University of Bayreuth)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过对11名德国房地产经纪人的半结构化访谈，构建了生成式人工智能（GenAI）在房地产营销中的使用案例地图，区分当前和潜在应用，并分析了人机协作模式与采用障碍；

**💡 创新点**

首次系统性地在德国背景下映射GenAI在房地产营销中的实际与预期使用场景，揭示人机协作细节与法规、数据可用性、合规性等特定限制；

**🔧 技术方法**

采用定性研究方法，使用Gioia式编码流程对访谈文本进行开放、轴向、选择性编码，归纳出一阶概念、二阶主题和聚合维度；

**📊 数据集**

以德国房地产经纪人访谈为数据集，涵盖不同性别、经验年限、工作性质和事务量的受访者；

**📈 对比分析**

本研究不进行算法或性能对比，主要提供案例映射与定性分析，未采用定量评估指标；

**⚠️ 局限性**

样本量小且仅包含已有AI使用经验的经纪人，缺乏非用户与抵制者视角，无法推广到更广泛群体；未对法律合规性进行深入评估，且未验证所述使用场景的技术实现与效果。

---

## 368. SCQ: Stabilizing Conservative Q-Learning with Sigmoid-Bounded Entropy

**arXiv ID:** 2609.12749 | [PDF](https://arxiv.org/pdf/2609.12749v1)

**作者:** Xiefeng Wu `[一作]` (Wuhan University), Mingyu Hu `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出SCQ框架，改用正向sigmoid‑bounded熵替换传统log‑entropy，以稳定离线到在线强化学习的策略更新

**💡 创新点**

创新点在于将熵项限定为始终正值并上限，通过消除负熵导致的策略抖动和价值估计不稳定，提升训练鲁棒性

**🔧 技术方法**

使用tanh‑squashed Gaussian策略、Cal‑QL风格的保守Q正则化、return‑based lower‑bound校准、Critic‑only LayerNorm以及Sigmoid‑bounded熵

**📊 数据集**

在Minari/D4RL全数据集、视觉DMC任务以及四台真实机器人（机械臂、轮式、四足、仿人）上进行评估

**📈 对比分析**

与Cal‑QL、RLPD、FlashSAC、SAC+OD、AWAC、IQL、DrQ‑v2、DrM等基线比较，SCQ在AUC、在线成功率和最终表现上均优于或与最强基线相当，特别是在一-shot学习与视觉控制任务中显著提升

**⚠️ 局限性**

实验局限包括未涉及全关节低层控制、任务规模/时限有限，缺乏长时序多阶段和高接触复杂操作的验证

---

## 369. GRACE: Adaptive Concept Erasure with Geometry-Guided Retention in Diffusion Models

**arXiv ID:** 2609.12731 | [PDF](https://arxiv.org/pdf/2609.12731v1)

**作者:** Qinghui Gong `[一作]` (Southwest Jiaotong University), Zhengchun Zhou `[通讯]` (Southwest Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GRACE 框架，对文本到图像扩散模型进行概念消除，能够在抑制敏感概念的同时保持原模型的生成质量。

**💡 创新点**

创新点包括：①语义加权的敏感子空间估计（SA‑PCA）精准锁定目标概念方向；②子空间约束的轻量化适配器，限定参数修改范围；③自动解耦安全锚点，无需人工代理提示；④能量驱动的动态门控机制，按时间步动态调节干预强度，减少语义漂移。

**🔧 技术方法**

使用了扩散模型的轻量化适配器、SA‑PCA、CLIP 语义监督、子空间约束损失、能量门控与时间缩放等技术。

**📊 数据集**

在 I2P、MS‑COCO、GIPHY Celebrity Detector 等数据集上评估，针对 NSFW、IP 角色、艺术风格、身份等多种敏感概念进行消除。

**📈 对比分析**

与 ESD、MACE、Speed、SPM、Co‑Erasing 等 SOTA 方法对比，GRACE 在 NSFW 减少率提升 17.86%、目标 CLIP 分数下降 4.75%、FID 降低 50.58%，在多模型、跨概念、鲁棒性和参数效率上均优于对比方法。

**⚠️ 局限性**

局限性在于：需针对每个目标概念单独估计子空间并训练适配器；对极大规模概念集合的扩展仍需验证；依赖 CLIP 嵌入，可能对非 CLIP 语义的敏感概念处理效果有限。

---

## 370. Fresh-Challenge VDF Attestations for Model-Relative Response Latency

**arXiv ID:** 2609.12727 | [PDF](https://arxiv.org/pdf/2609.12727v1)

**作者:** Ansar Yesmukhanov `[一作]` (Nazarbayev University), Aruzhan Tlessova `[通讯]` (Nazarbayev University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并分析了一种名为 Fresh-Challenge Latency Attestation (FCLA) 的协议，用以在公共环境下对响应延迟进行模型相对的证明。

**💡 创新点**

创新点在于将可验证延迟函数（VDF）与不可预测的公开挑战、可审计的收发记录以及预先校准的评估速率上限相结合，形成了一个可公开验证、但不具身份绑定的延迟证明框架，并给出了正式的安全命题和设计规范。

**🔧 技术方法**

技术上采用了 Wesolowski 风格的 VDF、SHA-256 哈希、Rust 语言实现、公开的时间信标（Beacon）与可追加的日志系统，并通过域分离与会话标识确保抗重放与跨协议安全。

**📊 数据集**

实验数据集包括六个对数间隔的迭代次数（10²、631、3 981、25 119、158 489、1 000 000），在 Apple M4 设备上使用 512 位 Wesolowski 参数进行评估与验证。

**📈 对比分析**

通过在该设备上进行 10 次试验，评估时间随迭代次数呈指数增长（最高约 7 021 ms），而验证时间保持在约 12 ms，迭代速率最高约 1.42 × 10⁵ 次/秒；与传统 VDF 评估相比，验证效率极高，但评估成本随 T 迅速上升。

**⚠️ 局限性**

局限性包括：证明仅在预先校准的评估速率模型下成立；无法识别或绑定执行者身份；可能被中继、外包或更快的未建模机器规避；依赖外部时间戳和日志的可信度；以及无法排除所有潜在加速算法或硬件加速器。

---

## 371. ExpertHTR: Unified Handwritten Text Recognition with Multi-Task Learning and Sparse Mixture-of-Experts

**arXiv ID:** 2609.12705 | [PDF](https://arxiv.org/pdf/2609.12705v1)

**作者:** Dang Hoai Nam `[一作]` (University of Information Technology), Vo Nguyen Le Duy `[通讯]` (University of Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ExpertHTR框架，统一不同语言、脚本、结构的页级手写文本识别。

**💡 创新点**

通过Page–Region–Line表示构建四个无额外标注的多任务监督，并采用共享-稀疏Mixture-of-Experts实现表示依赖的条件容量。

**🔧 技术方法**

结合预训练vision‑language模型Qwen3.5、Sparsegen路由、稀疏专家、负载平衡正则化等技术。

**📊 数据集**

使用七个异构手写数据集：BRESSAY、Bentham、HWDB2.0、IAM、READ‑2016、RIMES、ScribbleLens。

**📈 对比分析**

与专业HTR系统及通用OCR/VLM基准对比，Sparse MoE在六/七个数据集上优于VLM并在IAM段落级别取得SOTA，但仍落后于部分专用HTR模型。

**⚠️ 局限性**

模型参数量增大、路由正则化未能完全消除专家集中、跨数据集迁移效果受限，且对极端历史长文档的鲁棒性仍不足。

---

## 372. NOVA-GS: Noise-Aware View-Consistent Gaussian Splatting for Low-Light Novel View Synthesis

**arXiv ID:** 2609.12682 | [PDF](https://arxiv.org/pdf/2609.12682v1)

**作者:** Shaurya Pavan A `[一作]` (Indian Institute of Technology Madras), Kaushik Mitra `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的低光环境下3D高斯散射框架NOVA-GS，集成曝光校正、无监督盲点去噪与几何优化；

**💡 创新点**

创新点在于利用VGGT直接进行姿态和几何初始化，构建结构感知增强模块、深度注意力残差UNet的自监督去噪网络，以及噪声引导的球谐正则化来抑制视角相关噪声；

**🔧 技术方法**

主要技术包括Visual Geometry Grounded Transformer (VGGT)、Differentiable Guided Filter、Deep Attention‑ResUNet、定向盲点去噪(N2V)以及噪声加权球谐正则；

**📊 数据集**

在四个真实低光数据集上评估：LOM、LLRS、LLNeRF和MVTV；

**📈 对比分析**

与NeRF‑、3DGS‑和2D+3D基线对比，NOVA‑GS在无监督条件下在PSNR、SSIM、LPIPS上取得或接近最优结果，特别在LLNeRF上显著领先；

**⚠️ 局限性**

局限性包括对极端传感器噪声的鲁棒性尚待提升，当前仅针对静态场景，且需要较长训练时间。

---

## 373. Bridging the First-Hour Gap: Evaluating AI Reliability and Benchmarking Deficiencies in Cyber Incident Response for Law Enforcement

**arXiv ID:** 2609.12681 | [PDF](https://arxiv.org/pdf/2609.12681v1)

**作者:** Roshin Sleeba C `[一作]` (National Institute of Technology Calicut), Hiran V Nath `[通讯]` (National Institute of Technology Calicut)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述并评估了首小时网络事件响应中可用的决策支持架构（手册、LLM、RAG、代理式AI），并指出现有基准不足以衡量执法场景下的安全性与可审查性。

**💡 创新点**

提出首小时响应的专属评估基准需求，强调对非技术执法人员的可用性、误报/幻觉风险、证据保全与法律可采性等维度的评估缺口；同时从实用性角度将RAG系统定位为最具可落地性的方案。

**🔧 技术方法**

主要技术包括：大型语言模型（LLM）、检索增强生成（RAG）框架、对话式多轮推理、可解释性与安全增强策略（如自一致采样、逆向提示、LLM回溯）以及对照传统手册与自动化SOAR流程的比较。

**📊 数据集**

使用了公开的安全基准数据集与评测框架（如 AthenaBench、CAIBench、CTIBench、IRBench 等），并对比了这些基准在法律正确性、技术前置条件以及适用受众上的差异。

**📈 对比分析**

方法：将LLM/RAG方案与传统手册、SOAR自动化及代理式AI在“法律正确性”“技术前置条件”“适用受众”等维度对齐；性能方面发现RAG在多轮自然语言推理与证据保全指导上相对优于纯LLM，但仍易出现幻觉与上下文丢失。

**⚠️ 局限性**

局限性包括：LLM产生的高置信度幻觉导致潜在证据丢失；缺乏针对执法人员的可解释性与审计追溯机制；现有基准未覆盖法律可采性与非技术询问的鲁棒性；系统对检索知识库的依赖导致知识更新与本土化受限。

---

## 374. Size Doesn't Matter: Material-State Reinforcement Learning for Excavator Transferable Soil Manipulation

**arXiv ID:** 2609.12677 | [PDF](https://arxiv.org/pdf/2609.12677v1)

**作者:** Lennart Werner `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

训练并部署了基于强化学习的单冲刺铲斗控制策略，在二维MPM粒子仿真中学习土壤重塑（堆坝、回填、压实），并通过归一化末端执行器接口实现跨尺寸挖掘机的部署。

**💡 创新点**

创新点：①将土壤形状、压实度等材料状态直接作为观测和奖励，使策略具备材料感知能力；②使用GPU并行的二维MPM仿真实现高吞吐量RL训练；③通过归一化末端执行器空间与机器力学校准实现无重训练的跨机器迁移。

**🔧 技术方法**

技术：强化学习（PPO）、Material Point Method（MPM）粒子仿真、NVIDIA Warp GPU并行、归一化接口校准、低层末端执行器跟踪控制、工作空间规划与状态机。

**📊 数据集**

数据集：仿真中随机生成土壤场景、土壤参数（砂、黏土、泥灰）与压实状态；实测数据来自Menzi Muck M445与LeExcavator的LiDAR、IMU、压力传感器，未使用公开标准数据集。

**📈 对比分析**

评估方式：与经验操作员在同一工作区对比堆坝施工，测量长度、高度、进度速度和高度波动；在回填、压实实验中比较单冲刺力、深度、压实记忆等指标。结果显示，自动策略在堆坝高度上超过专家，进度速度相当，且在多面接触与压实奖励下表现出自适应性。

**⚠️ 局限性**

局限性：①策略仅为局部单冲刺技能，无法直接完成长时任务；②归一化接口对工作空间比例、感知精度、力学归一化等有约束，无法迁移到不兼容桶形或柔性低层控制的机器；③模拟仅为二维，缺乏三维效应和异质材料表现；④未包含时间序列或历史信息，限制了多冲刺规划能力。

---

## 375. ProactiveBench: Can Streaming Video Models Really Interact Like Humans?

**arXiv ID:** 2609.12658 | [PDF](https://arxiv.org/pdf/2609.12658v1)

**作者:** Kaixuan Du `[一作]`, Ni Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并公开了 ProactiveBench benchmark，用于评估流式多模态模型的主动交互和时间响应行为。

**💡 创新点**

创新点在于：①设计六个细粒度子任务，覆盖不同触发模糊度和时间容差；②采用每秒评估且无显式响应提示的协议，结合响应与沉默的几何平均；③引入对提前和漏报响应的精细判定，揭示模型时序决策缺陷。

**🔧 技术方法**

使用基于视频-问答和视频字幕数据的 196k 训练序列进行监督微调，并采用多模态 Transformer 结构；评估采用 ES、OR、WR、AA、SS、DC 等指标，并对响应率与沉默率进行几何平均。

**📊 数据集**

数据集：公开的视频问答/字幕数据合成的 196k 训练序列；ProactiveBench 自身包含 5.4 小时、18,146 秒的评估实例，覆盖 6 个子任务。

**📈 对比分析**

比较方法：对六个代表性模型进行同一 benchmark 评估，得到每个子任务得分和平均分；结果显示最高平均分约为 41.0，模型间差距小；同时发现四个模型的提前响应频率比漏报高 2–28 倍，且几何平均对排名影响显著。

**⚠️ 局限性**

局限性：①评估未对模型进行 ProactiveBench 专门微调；②不同模型的沉默机制差异导致可比性受限；③部分子任务使用不同录制，不能完全消除内容偏差；④评估仍未完全隔离模型能力与响应策略。

---

## 376. SWARM: A Multilingual Human-Annotated Dataset for Russian Propaganda Detection in Search Engine Results

**arXiv ID:** 2609.12653 | [PDF](https://arxiv.org/pdf/2609.12653v1)

**作者:** Manuel Tonneau `[一作]` (Weizenbaum Institute), Elizaveta Kuznetsova `[通讯]` (Weizenbaum Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个多语言、人工标注的搜索引擎结果数据集（SWARM），用来检测文档是否支持俄国宣传叙事；同时在该数据集上对比了源级黑名单、监督分类器与零样本大型语言模型的表现。

**💡 创新点**

创新点：①首次在搜索引擎结果中进行多语言叙事支持标注；②提出叙事支持而非单纯的叙事出现或真伪标签；③证明源级黑名单不足，零样本LLM在少数正例场景中可超越监督模型。

**🔧 技术方法**

使用的技术包括：基于域名的黑名单筛选；监督学习（冻结嵌入+逻辑回归、XLM‑R fine‑tune、SetFit）；零样本LLM（Qwen2.5‑7B/72B、GPT‑5‑nano/5.4）以及多语言机器翻译和Prompt 设计。

**📊 数据集**

使用的数据集：SWARM，2183条记录，9种语言（阿拉伯语、英语、德语、印地语、波兰语、葡萄牙语、俄语、西班牙语、乌克兰语），来自4个搜索引擎的20条俄国宣传叙事查询结果。

**📈 对比分析**

对比方法：源级黑名单F1≈0.39；监督模型F1≈0.47‑0.51；零样本LLM F1≈0.62‑0.71（最佳为GPT‑5‑4），总体而言零样本LLM在少数正例上表现更好，但仍有误报与漏报。

**⚠️ 局限性**

局限性：①检索/可爬取文本的偏倚导致部分俄国宣传来源被低估；②正例稀缺、注释一致性有限；③数据为2024年1月快照，时效性受限；④只评估了少数模型，未涉及更大或更细粒度的微调方法。

---

## 377. Hybrid Monitoring for Early Fault Detection in Cloud-Native 5G Systems

**arXiv ID:** 2609.12649 | [PDF](https://arxiv.org/pdf/2609.12649v1)

**作者:** Anton Andersson `[一作]` (Chalmers University of Technology and University of Gothenburg), Romaric Duvignau `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 NetMon，一种结合 eBPF 监控与主动 TCP 探测的混合网络监测框架，专为 Kubernetes 部署的 5G AMF 系统设计。

**💡 创新点**

创新点在于将内核级被动监测与主动探测相结合，并通过集中化的关联引擎实现秒级的故障检测与定位，解决了传统健康检查和被动监测的局限。

**🔧 技术方法**

核心技术包括 eBPF XDP/TC 采集内核层网络指标、主动 TCP 探测、基于 CUSUM 的在线漂移检测、Z-score 与阈值统计分析，以及集中式关联引擎。

**📊 数据集**

使用 Ericsson 的虚拟 5G AMF 集群（4 节点、10 个 Pod）作为实验平台，注入多种网络故障（延迟、抖动、丢包、带宽、分区、链路失效、Pod 终止）进行评估。

**📈 对比分析**

与 Prometheus/Blackbox 等传统方法比较，NetMon 在 10–50 ms 延迟或 5% 丢包的微小故障下可在 2–10 秒内检测并在 6–20 秒内定位，误报率低于 18 小时基线；资源开销仅 3.4 mCPU/Pod、4.5 MiB 内存。

**⚠️ 局限性**

主要限制包括：高频率的噪声导致正常运行时每小时约 2.9 k 次误报；O(N²) 探测规模限制大规模集群的可扩展性；单点故障的中心关联器；严重带宽削减时监测报告被抑制；以及对 Linux 5.8+ 与 eBPF 的硬件/安全依赖。

---

## 378. Reconstruction and Reflection of Positive Experiences through Resurfacing Laughter-indexed Everyday Moments

**arXiv ID:** 2609.12642 | [PDF](https://arxiv.org/pdf/2609.12642v1)

**作者:** Jun Fang `[一作]` (Tsinghua University), Yuanchun Shi `[通讯]` (Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了名为 LaughAnchor 的可穿戴自追踪系统，利用被动检测到的笑声作为稀疏情感索引，结合时间、位置、社交与第一人称视角的上下文信息构建“Moment Card”，支持用户在不同时间间隔的回顾与反思。

**💡 创新点**

创新点在于将笑声作为情感触发点来捕捉正面日常瞬间，采用分层上下文披露与用户可控的保留与重现机制，实现在不需即时记录、且保留意义判断权的前提下，完成从情感索引到情节重建与情感再体验的闭环。

**🔧 技术方法**

技术实现包括可穿戴眼镜摄像/麦克风模块（Groudchat）与安卓手机同步、Speechmatics 实时笑声检测 API、后端聚类与上下文关联算法、前端 Moment Card 的分层展示与交互，以及基于时间/地点的自动/可选重现推送。

**📊 数据集**

使用的数据集为 12 名参与者在 3 周内完成的 118 次录制会话、产生的 129 个候选笑声片段（115 经保留）以及 95 次问卷确认的回顾，记录包括音频、视频、时间、位置、社交标签与用户手工摘要。

**📈 对比分析**

与参与者原有的记录与回顾习惯进行对比评估，使用问卷量表、使用日志和访谈收集指标；结果显示笑声索引新增 44% 的有价值记录，情感再体验评分 6.15/7，重建细节与重新发现评分均高于基线；系统可用性 SUS 评分 83.5，说明可行性良好；但未与传统拍照/文字记录做严格实验对照。

**⚠️ 局限性**

局限性包括样本以大学生为主，实验周期短（3 周），硬件为外置眼镜模块需手动佩戴，旁观者隐私与同意机制未充分实现；缺乏长时间跟踪验证持续效果与因果关系；缺少对不同文化、年龄及更广泛日常场景的普适性验证。

---

## 379. Explaining Time Series Forecasting with Horizon-Resolved Attribution

**arXiv ID:** 2609.12639 | [PDF](https://arxiv.org/pdf/2609.12639v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Horizon-Resolved eXplanation (HRX)，为时间序列预测模型生成每个预测步的独立重要性矩阵，并设计了基于删除曲线的评估协议验证该方法的有效性。

**💡 创新点**

创新点在于突破传统共享重要性向量的限制，加入时间步轴，生成按步的解释矩阵；通过秩准则和评估协议量化该轴的价值，并证明其低维可行性。

**🔧 技术方法**

主要技术包括梯度估计（可替换为其他解释器）、解释矩阵构造、删除曲线评估（AUC）、奇异值分解求有效秩，以及在多种前馈、CNN、Transformer 等骨干上实验。

**📊 数据集**

使用公开通用基准（如 M4、ETTh2 等）以及自构建的金融基准（2018-2025 年美国上市公司每日收盘价）进行评估。

**📈 对比分析**

通过自定义指标 g_own、g_shuffled、g_margin 与现有 AUPRC/AUROC 等指标对比，实验表明 HRX 在所有骨干和基准上均显著提升解释效果，g_margin 始终为正且优于传统方法。

**⚠️ 局限性**

局限性在于无法预测不同时间序列对轴效应的大小；实验未能提升模型预测性能，仅提供诊断工具，并且缺乏对实际业务决策的直接改进。

---

## 380. Multimodal Floorplan Encoding: Learning Dense Modality-Invariant Representations

**arXiv ID:** 2609.12723 | [PDF](https://arxiv.org/pdf/2609.12723v1)

**作者:** Xavier Anadón `[一作]` (University of Zaragoza), Rui Wang `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了MMFE，将多模态2D平面图编码为共享稠密特征网格。

**💡 创新点**

通过冻结DINOv3与可训练DPT头结合，并采用每像素InfoNCE与Sim(2)扰动训练，获得跨模态几何一致的稠密表示。

**🔧 技术方法**

使用DINOv3视觉骨干、Dense Prediction Transformer、InfoNCE对比学习、Sim(2)扰动、特征网格反扭曲、RANSAC匹配，以及RoMa、NetVLAD/SALAD聚合等技术。

**📊 数据集**

训练数据来自CubiCasa5K、Swiss Dwellings、ZInD、Aria SE四大室内平面图集合；评估使用Hold‑out的Structured3D数据集。

**📈 对比分析**

与DINOv2/v3稠密特征、RoMa匹配器、NetVLAD/SALAD检索器对比；MMFE在中/大扰动下对齐RMSE从200+降至≈15，检索Top1提升至≈55%，显著优于基线。

**⚠️ 局限性**

数据规模有限，稀疏扫描仅为模拟；32×32的分辨率限制精度；仅对Sim(2)扰动，未覆盖真实非刚性变形；对大型预训练匹配器迁移效果不佳。

---

## 381. Poster: Towards Selecting Threat Appropriate Industrial Intrusion Detection Systems

**arXiv ID:** 2609.12646 | [PDF](https://arxiv.org/pdf/2609.12646v1)

**作者:** Stefan Lenz `[一作]` (RWTH Aachen University), Martin Henze `[通讯]` (RWTH Aachen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证一种基于CTI共享的工业入侵检测系统（IDS）动态选择机制，能根据当前威胁情境自动选取最合适的检测器。

**💡 创新点**

创新点在于将CTI共享与TTP（Tactics, Techniques, Procedures）映射相结合，构建攻击场景专属的IDS性能数据库，实现威胁情境驱动的IDS选择，而非传统的全局数据集级评估。

**🔧 技术方法**

技术主要包括：① MITRE ATT&CK® ICS矩阵与CISA映射工具用于将WDT数据集攻击标注为TTP；② 对四类不同检测机制（通信时序、通信特征、SVM、过程状态不变量）进行特征提取和异常检测；③ 采用时间序列召回率作为评价指标；④ 通过表格与图示展示不同TTP下各检测器的性能差异。

**📊 数据集**

使用了WDT（Wind Tunnel Data）工业控制系统攻击数据集，并将其攻击实例映射到六个TTP上（如DoS、DoC、AitM等）。

**📈 对比分析**

通过对完整数据集与单一TTP分别计算召回率，比较四个检测器的性能。结果显示：整体最佳检测器为GeCo（召回率0.62），但在DoS场景中InterArrivalTime可达0.87，AitM场景中OneClassSVM从0.09提升至0.64，体现了不同攻击情境下检测器的显著差异。

**⚠️ 局限性**

主要限制包括：① CTI共享的激励与隐私问题导致共享参与度不确定；② TTP标签的一致性与自动化映射仍存在挑战；③ 如何在实际部署中及时切换或组合多种IDS以兼顾覆盖率与准确性尚需进一步研究。

---

## 382. VideoTok4D: A 4D-Aware Video Tokenizer for Compact World Representation

**arXiv ID:** 2609.12874 | [PDF](https://arxiv.org/pdf/2609.12874v1)

**作者:** Xinyi Chen `[一作]` (University of Science and Technology of China), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 VideoTok4D——一种四维视频 tokenizer，能够把动态视频压缩为静态和动态 token，并通过 Co4DGen 在 token 空间进行高效的四维场景生成。

**💡 创新点**

创新点在于：1）spatiotemporal disentanglement 把视频拆成静态 token（背景）和动态 token（运动）；2）track‑aware dynamic attention 用相机补偿与轨迹聚合，保持跨视角运动一致；3）Co4DGen 在联合 token 空间做稀疏 diffusion，只需一次采样即可生成多视角视频。

**🔧 技术方法**

技术细节包括：VA‑VAE 图像编码器、RAFT 光流+UniDepth 深度估计、相机补偿残差掩码、轨迹跟踪与自注意力、latent rectified‑flow 解码器、跨 token cross‑attention、联合 diffusion 目标。

**📊 数据集**

使用 MultiCamVideo‑Dataset（13,600 个同步 10 视角动态场景）进行训练与评估。

**📈 对比分析**

与显式 4D 表示（MoVieS、4DGT、C4G）以及 token 基础方法（SceneTok、SceneGen）对比。动态新视角合成中，VideoTok4D 在 PSNR/SSIM/LPIPS/rFID/rFVD 上均优于对手，且存储量比显式方法低 4 个数量级；Co4DGen 生成质量上优于 DFoT、SceneGen，且推理时间比 DFoT 快 3.4×。

**⚠️ 局限性**

局限性包括：对精确相机标定与光流估计的依赖，遮挡或极端快速运动时性能下降；静态/动态 token 划分不一定完全精准；模型对新场景的泛化能力尚未深入验证。

---

## 383. DuplexDrama: A Synthesized Dialogue Dataset with Scenarios, Full-Duplex Behaviors, Expressive Speech, and Sound Events

**arXiv ID:** 2609.12872 | [PDF](https://arxiv.org/pdf/2609.12872v1)

**作者:** Qingxiang Guo `[一作]` (Zuoyebang Education Technology), Yang Song `[通讯]` (Zuoyebang Education Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了首个同步语音对话数据集 DuplexDrama，涵盖完整人物设定、情景、三种全双工行为、情感标签和脚本感知声响事件，并进行质量验证。

**💡 创新点**

将人物设定、情景、全双工行为、情感与声响事件四维标注整合为一次性合成语音对话数据集，首次提供可用于训练全双工语音对话模型的公开资源。

**🔧 技术方法**

采用四阶段流水线：人物/情景生成、脚本带标注生成、基于 IndexTTS2 的情感语音合成与音频拼接，以及脚本感知声响事件注入，并使用双 LLM 验证器与音频指标评估质量。

**📊 数据集**

依托自建的 64 语音池（13 角色，5 年龄桶），600 长时段环境噪音与 2,203 个短时声响事件库，生成 800 小时双语（中文 500h、英文 300h）对话样本。

**📈 对比分析**

与 Fisher、CANDOR、Open-Yap-1K、DuplexConv、SpeechDialogueFactory 等已有数据集对比，唯一同时覆盖四维标注；在质量验证中 WER 1.8%、SpkCons 97.3%、UTMOSv2 2.57/2.52、NISQA 3.65/3.18，说明合成语音质量较高。

**⚠️ 局限性**

全双工行为分布偏低（仅 3.8%），合成语音缺乏自然对话的细粒度抑扬顿挫，声响事件匹配仍依赖手工音频库，未来需提升真实标签比例和声响多样性。

---

## 384. A Multi-Vehicle Dataset with Camera, LiDAR, and Radar Sensors and Scanned 3D Models for Custom Auto-Annotation using RTK-GNSS

**arXiv ID:** 2609.12871 | [PDF](https://arxiv.org/pdf/2609.12871v1)

**作者:** Philipp Berthold `[一作]` (University of Bundeswehr Munich), Mirko Maehlisch `[通讯]` (University of Bundeswehr Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个包含摄像头、激光雷达、毫米波雷达以及RTK-GNSS/INS的多车数据集，并为所有目标车辆提供了高精度的3D扫描模型、连续运动学参考和可自动注释的框架。

**💡 创新点**

创新点在于：①每辆车都配备RTK-GNSS/INS，获取实时高精度位姿与运动学；②对每辆车进行完整的3D扫描，得到纹理化几何模型；③结合这些模型可按任意粒度自动生成传感器测量的标注；④支持多车、遮挡、反射等复杂场景的真实评估。

**🔧 技术方法**

使用的技术包括：Velodyne VLS‑128 激光雷达、Basler RGB摄像机、Smartmicro UMRR‑96/32 雷达、OxTS RT3000v3 RTK‑GNSS/INS、Artec Leo 3D 扫描仪、ROS 机器人框架、TF/CameraInfo 传感器标定、基于 RTK‑GNSS 的姿态互校、Ray‑casting 生成注释。

**📊 数据集**

数据集为自研的“7V‑Scanario”，包含 7 辆异构车辆（包括小车、面包车、SUV 等）在单车与多车两类场景中的传感器原始数据、GNSS/INS 原始包、以及对应的 3D 模型文件。

**📈 对比分析**

通过将雷达/激光点云投影到扫描模型坐标系，生成热图、累积点云等可视化评估；并使用归一化算法校正视角偏差；实验显示雷达的高度测量不稳定、激光窗透射等现象可被精确定位；相较于现有公开数据集，7V‑Scanario 在运动学精度和可注释粒度上具有显著优势。

**⚠️ 局限性**

局限性包括：①目标车辆结构（如天线、传感器）对扫描模型造成干扰，影响精度；②目前仅支持静态扫描，未实现可动部件建模；③数据量相对有限，缺乏更大规模多场景覆盖；④仍需针对不同标注格式编写专用自动注释工具。

---

## 385. MedRoundsQA: A Persona and Difficulty Aware Evaluation for Multi-Turn Medical Consultations

**arXiv ID:** 2609.12851 | [PDF](https://arxiv.org/pdf/2609.12851v1)

**作者:** Youssef Mohamed `[一作]` (MBZUAI), Xiuying Chen `[通讯]` (INSAIT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个多轮诊疗基准MedRoundsQA，将1,387份医学考试题转化为结构化的24槽临床记录，并基于同一记录生成不同患者人设的模拟问诊对话；

**💡 创新点**

创新点在于提供了可对照的单轮与多轮诊断评估，利用同一临床内容进行对人设的因果对照，且通过模型不确定性实现难度分层；

**🔧 技术方法**

采用LLM（如GPT‑4o、DeepSeek‑Chat）进行记录抽取与对话生成，使用LLM判别器进行诊断正确性评估，并通过两阶段注释与自动化难度估计；

**📊 数据集**

使用公开的MedQA、NEJM诊断案例和MedMCQA作为来源，最终得到1,387个多专科案例；

**📈 对比分析**

与15款LLM医生代理进行比较，发现多轮诊断准确率比单轮低13–39点，问题相关性随回合提升但诊断准确性在6–12回合后趋于平稳；

**⚠️ 局限性**

局限性包括仅针对文本单次会诊，未考虑多次随访、影像或长期记录，且人设差异与真实患者群体的可推广性仍需进一步验证。

---

## 386. What Did the MLLM Hear? Token-Level Spectro-Temporal Grounding for Audio MLLM Explainability

**arXiv ID:** 2609.12663 | [PDF](https://arxiv.org/pdf/2609.12663v1)

**作者:** Lucia Cascone `[一作]` (University of Salerno), Benedetto Simone `[通讯]` (University of Salerno)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出STAG框架，实现对音频多模态大型语言模型（MLLM）生成的标题进行token级声谱时频解释；

**💡 创新点**

创新点在于将白盒激活时间归因与黑盒频带遮蔽（FBO）相结合，既无需改动模型亦可得到每个token对应的二维时频重要性图；

**🔧 技术方法**

主要技术包括利用语言模型的隐藏层与词表投影提取时间归因、使用Mel频带遮蔽评估频率相关性、矩阵外积融合、时间平滑与共享归一化，以及token-事件聚合；

**📊 数据集**

评估使用四个公开基准：AudioTime、AudioGrounding、TACoS和AudioSet‑Strong，覆盖从控制剪辑到真实多声场录音的多种情景；

**📈 对比分析**

与十种主流后置可解释方法相比，STAG在四个基准上均获得最高的事件定位F1分数（平均提升约10点），同时保持良好的函数词“无声”指标，证明了其在定位与选择性方面的优势；

**⚠️ 局限性**

局限性包括缺乏频率级注释导致评价仅在时间上、标签歧义可能误判、以及采用可分离时频融合假设，未来需构建时频标注数据并探索更复杂的非可分离归因机制。

---

## 387. Learning-Augmented Strategyproof Facility Location in $\mathbb{R}^d$ with $\ell_p$ Distances

**arXiv ID:** 2609.12829 | [PDF](https://arxiv.org/pdf/2609.12829v1)

**作者:** Hau Chan `[一作]` (University of Nebraska--Lincoln), Chenhao Wang `[通讯]` (Beijing Normal University--Zhuhai)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

研究在ℓp空间中利用预测信息改进无金钱机制设计的单设施定位问题，提出并分析了坐标逐位中位数加预测（CMP）机制，给出了二维空间下一致性和鲁棒性的精确表达式，并在高维空间中得到维数无关且渐进紧致的上界，同时给出相应的下界构造。

**💡 创新点**

将预测技术引入多维设施定位的泛化距离（ℓp）框架；首次给出二维情况下的一致性与鲁棒性闭式解；通过新的标量平均约束与对称构造，实现了高维空间下维数无关且渐进紧致的性能界限；扩展并统一了先前在欧几里得距离下的结果。

**🔧 技术方法**

核心技术包括：坐标逐位中位数的策略无关性证明、Hölder不等式与三角不等式的组合、凸包与对称性分析、标量化的平均约束与最大化问题、以及对三点/对称实例的构造与优化。

**📊 数据集**

无实验数据集，全部工作为理论分析与证明。

**📈 对比分析**

与传统无预测的CM机制（均为最优或已知近似比）以及Agrawal等人对欧几里得距离的学习增强结果进行对比。结果显示：在预测准确时，CMP可实现更优的一致性；在预测失效时，鲁棒性保持在可接受范围内；且在二维下的闭式比值优于任何已知方法，且高维下的上界与下界在维度趋于无穷时收敛，证明了其渐近紧致性。

**⚠️ 局限性**

限制包括：1）仅给出了维数大于2时的渐近结果，缺乏针对每个固定维数的精确上界；2）在二维与高维结果之间存在差距，尚未确定是否为真正的维数依赖；3）仅分析了坐标逐位中位数方法，未探讨更复杂或随机化机制；4）对预测误差的统计性质无假设，导致在实践中对预测准确性的评估可能受限。

---

## 388. A Dual Cross-Attention Framework for Colposcopic CIN Grading and Swede Score Prediction Using a New Multi-Center Dataset

**arXiv ID:** 2609.12827 | [PDF](https://arxiv.org/pdf/2609.12827v1)

**作者:** Dania Khan `[一作]` (Bangladesh University of Engineering and Technology), Taufiq Hasan `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建双流交叉注意力网络，实现宫颈上皮内瘤变(CIN)分级和斯威德评分预测

**💡 创新点**

首次提出双流交叉注意力机制、组合损失（加权焦点损失+Huber+校正项）以及多中心标注数据

**🔧 技术方法**

利用EfficientNet骨干、Swapped QKV交叉注意力、CBAM、全局平均池化和自定义复合损失

**📊 数据集**

BUET多中心宫颈镜图像数据集（768例，3072张）与IARC宫颈镜图像银行合并，共957例，包含斯威德评分与CIN分级

**📈 对比分析**

与七种经典CNN（ResNet、EfficientNet、DenseNet等）和Swin Transformer对比，CIN分级准确率71.85%（AUC 86.23%），斯威德分量预测AUC提升至0.766–0.884，整体MAE降至1.49

**⚠️ 局限性**

样本量有限、类别严重不平衡、模型在未见中心和设备上的泛化需进一步验证

---

## 389. SCDM: Spatial-Contextual Disentanglement Mamba via Differential Inference for Efficient Image Classification

**arXiv ID:** 2609.12825 | [PDF](https://arxiv.org/pdf/2609.12825v1)

**作者:** Mustafa Bora Çelik `[一作]` (Ankara Medipol University), Ayse Keles `[通讯]` (University of Galway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在医学影像分类中提出了一种空间-上下文差分Mamba（SCDM）双分支架构，用于将病理特征与正常解剖背景分离并实现高效分类。

**💡 创新点**

创新点是引入正分支与负分支的非对称双分支结构，并通过相似度驱动的排斥门与差分推理实现无须额外标签或增大模型容量的表征解耦。

**🔧 技术方法**

使用了VMamba状态空间模型、差分门控、相似度排斥机制、竞争差分读取以及多项损失（分类、竞争、正交、激活）来训练网络。

**📊 数据集**

采用RSNA肺炎数据集（20,672正常 + 6,011肺炎）进行评估。

**📈 对比分析**

与ResNet‑101、Swin‑B、ViT‑Base、VMamba‑B等基线在相同训练条件下比较，SCDM在AUC 0.858、特异性77%、参数29.4 M、FLOPs1.44 G，召回率略低于ResNet‑101，但显著降低了模型规模与计算量。

**⚠️ 局限性**

局限性包括召回率相对较低、仅验证了二分类任务、对高分辨率图像和多类别情形的适应性尚未充分验证。

---

## 390. Batten the Hatches: Cybersecurity with Military Mariners

**arXiv ID:** 2609.12810 | [PDF](https://arxiv.org/pdf/2609.12810v1)

**作者:** Ryan Von Brock `[一作]` (Georgia Institute of Technology), Raheem Beyah `[通讯]` (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对20名美国海军和海岸警卫队船员进行半结构化访谈，研究了他们在海上作业环境中如何理解、识别与响应网络风险。

**💡 创新点**

首次将海军船员的安全导向事件响应模型与其网络风险认知进行系统关联，揭示了组织抽象化对网络安全人因的影响，填补了军用网络安全研究空白。

**🔧 技术方法**

采用定性研究方法，包括访谈设计、主题编码、可靠性检验（Krippendorffα）以及情景分析，确保对受访者经验的深入捕捉。

**📊 数据集**

研究数据来自20名海军与海岸警卫队船员的访谈记录，涵盖三种网络情景（GPS欺骗、推进控制失效、ENAV系统全失）和相关问卷信息。

**📈 对比分析**

通过对比民用海员与军事海员的观点，识别出安全重视度、风险认知与响应差异；方法主要为主题分析与定性比较，未涉及量化性能指标。

**⚠️ 局限性**

局限性包括样本规模有限、仅能进行非机密访谈导致无法获取机密流程与技术细节、受自我报告偏差与招募渠道可能导致样本偏差。

---

## 391. Scaling Clinical Judgment to Evaluate Medical AI

**arXiv ID:** 2609.12822 | [PDF](https://arxiv.org/pdf/2609.12822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 392. Detecting HTTP Status Code Misuses in REST APIs via Static and Dynamic Analysis

**arXiv ID:** 2609.12770 | [PDF](https://arxiv.org/pdf/2609.12770v1)

**作者:** Alix Decrop `[一作]` (University of Namur), Gilles Perrouin `[通讯]` (University of Namur)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过设计 30 条基于 HTTP 标准与 REST 原则的状态码使用规则，开发静态分析工具 SCOAS 与动态分析扩展 EvoMaster，系统评估 REST API 中的 HTTP 状态码误用，并验证两种方法的互补性。

**💡 创新点**

①提出 30 条细粒度的状态码使用规则；②实现完整的静态分析框架（SCOAS）和动态检测插件（EvoMaster）；③在三大公开基准（APIs.guru、PRAB、WFD）上进行大规模实验，首次量化 REST API 中状态码误用的普遍性与严重性。

**🔧 技术方法**

使用 OpenAPI 规范解析、规则引擎（静态）以及基于 EvoMaster 的模型驱动 fuzzer（动态）进行检测；结合 Java/Kotlin SpringBoot 环境进行动态测试；使用 JSON、HTML 结果输出；通过重复实验验证鲁棒性。

**📊 数据集**

三套公开数据集：APIs.guru（约 2,525 个 OAS）、PRAB（60 个 OAS）和 WFD（36 个 OAS+实现）共计 2,625 个 API 规格与 36 个真实实现。

**📈 对比分析**

将静态分析与动态分析在同一套数据集上对比，静态分析在 2,612 个 OAS 中检测到 696,891 次误用；动态分析在 36 个实现中发现 8 种不同类型的误用，平均每小时 1 次实验。两种方法互补，静态覆盖文档，动态捕获运行时行为。性能方面，静态分析仅需数秒完成，动态分析在 1 小时内完成，10 次实验共 360 次跑，结果稳定。

**⚠️ 局限性**

仅支持 OpenAPI 文档；只实现了 30 条规则中 16 条可动态化的规则；误用定义基于手工判定，可能存在主观性；实验环境与真实生产环境可能存在差异；未覆盖 Postman、RAML 等其它 API 文档格式。

---

## 393. A Graph-Based Approach for Mapping Kernel-Level Telemetry to MITRE ATT&CK

**arXiv ID:** 2609.12841 | [PDF](https://arxiv.org/pdf/2609.12841v1)

**作者:** Matteo Lupinacci `[一作]` (University of Calabria), Angelo Furfaro `[通讯]` (University of Calabria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套端到端的自动化方法（Trace2ATT&CK），将Linux kernel级别的eBPF telemetry转换为MITRE ATT&CK技术与子技术映射，并给出可解释的推理理由。

**💡 创新点**

创新点在于：①利用eBPF实时收集系统调用并构建 provenance graph；②将图压缩为加权命令图以满足LLM上下文限制；③采用三种推理范式（纯提示、RAG、taxonomy-grounded prompting）并系统评估其对本地LLM的效果；④公开了基于Atomic Red Team的完整 kernel 事件数据集。

**🔧 技术方法**

核心技术包括 eBPF 事件采集、Neo4j 图数据库建模、命令–事件关联、加权命令图提取、LLM推理（GPT-OSS、Gemma、Llama、Qwen 等）与检索增强生成（RAG）以及中文/英文提示工程。

**📊 数据集**

数据集为 347 个 Linux Atomic Red Team 测试的原始 eBPF 事件日志（约 21.8M 行、9.7 GB），并根据 ATT&CK 关联进行标注。

**📈 对比分析**

与纯提示相比，RAG 在所有模型上平均提升 HR@5 约 5–10 分；taxonomy-grounded 提示进一步提高 2–4 分；在大模型 Claude‑Sonnet‑5 上表现最优（HR@5≈91%，子技术≈75%）。相较于 LADE 等先前工作，Trace2ATT&CK 在更大规模、真实环境下的准确性显著提升，且在半成功攻击时仍保持较高技术级别召回。

**⚠️ 局限性**

局限包括：①对大型完整 provenance graph 的上下文长度限制导致约一半样例无法返回结果；②对子技术的精确推断仍受限于事件缺失和模型内部知识的不足；③评估主要聚焦 Linux 平台，跨平台（Windows）适配尚待验证；④对实时在线推理的性能未在实验中充分展示。

---

## 394. A Historical Corpus Is Not a Historical System: Auditing Hindsight Leakage in Stateful Data Discovery

**arXiv ID:** 2609.12766 | [PDF](https://arxiv.org/pdf/2609.12766v1)

**作者:** Yixi Zhou `[一作]` (Hong Kong Baptist University), Haipeng Zhang `[通讯]` (ShanghaiTech University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了点时间（PIT）检索评估协议，构建了可审计的历史状态回放与未来回放对比，评估交互内存对检索质量的直接影响。

**💡 创新点**

创新点在于：1）将系统状态拆分为{语料、模型、交互内存}三部分，正式定义点时间评估的因果条件；2）引入Temporal Violation Rate（TVR）做历史有效性审计；3）通过paired replay（PIT vs Future）量化Hindsight Gain，实现对未来经验导致的“回溯偏差”可视化；4）提供完整的可验证manifest和机器可检查的契约。

**🔧 技术方法**

使用的技术包括：固定语料与检索索引的离线回放、基于BERT的语义相似度最近邻、BM25和Hybrid检索器、行为仅存的查询追踪记忆、正反馈缓存模拟、bootstrap查询集估计、加密哈希做版本控制与一致性验证。

**📊 数据集**

数据集主要包括：DPDisc（从DPBench转换的表文本问答数据，涵盖ConvFinQA、HybridQA、TAT-QA）以及FreshStack技术文档检索数据（五个主题）。

**📈 对比分析**

比较方法：在相同的语料、检索器、预算和anchor的前提下，对PIT、Future、Stateless、Naive等记忆视图进行paired评估；使用Asset Recall@100为主指标，并报告Recall@10/20/50、完整产品覆盖率、所需检索数量等；结果显示Future相对PIT提升2.62–5.24个百分点，Future相对Stateless提升0.0~1.75个百分点；PIT记忆在所有设定下均为负效应，Future的回溯掩盖了32.7–48.4%的负面影响。

**⚠️ 局限性**

局限性包括：1）实验仅覆盖三种表文本检索领域、两种检索器、两种记忆机制和FreshStack的五个主题；2）未评估答案生成质量、用户满意度或下游任务成功率；3）只能检测直接的记忆时间偏差，无法捕捉长期分支依赖的长期回溯效应；4）要求系统能公开事件时间戳和依赖线索，对某些商业系统不适用。

---

## 395. MAAPO:an innovative membrane algorithm based on artificial protozoa optimizer for multilevel threshold image segmentation

**arXiv ID:** 2609.12756 | [PDF](https://arxiv.org/pdf/2609.12756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 396. Optimizing for the decision not the prediction: an exploration of Smooth Net Benefit as a training objective

**arXiv ID:** 2609.12752 | [PDF](https://arxiv.org/pdf/2609.12752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 397. Interpreting the predictions of neural network classification based on a Taylor Coefficient Analysis (TCA)

**arXiv ID:** 2609.12801 | [PDF](https://arxiv.org/pdf/2609.12801v1)

**作者:** Markus Klute `[一作]` (Karlsruhe Institute of Technology), Roger Wolf `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于泰勒展开的输入特征空间对神经网络（NN）分类预测影响的分类法和范式，称为泰勒系数分析（TCA）。

**💡 创新点**

创新点在于建立了一个严格且全面的分类法，以描述输入特征空间对NN预测的影响，并通过TCA揭示了特征如何影响分类结果。

**🔧 技术方法**

使用了泰勒展开技术来分析神经网络的输出函数，特别是通过计算泰勒系数来量化输入特征的影响。

**📊 数据集**

使用了CERN LHC实验的典型分类任务数据集，并通过简单的二分类任务进行验证。

**📈 对比分析**

与其他方法（如LOO和SHAP）相比，TCA能够提供更高阶的特征分析，结果表明TCA在捕捉复杂任务的特征影响方面表现良好，通常能够捕捉到二阶特征。

**⚠️ 局限性**

限制在于对于高维特征空间的全面分析可能会变得不切实际，尤其是在特征数量较多时，计算和信息处理的复杂性会显著增加。

---

## 398. High-Fidelity Multi-Body Simulator for Autonomous Racing

**arXiv ID:** 2609.12795 | [PDF](https://arxiv.org/pdf/2609.12795v1)

**作者:** Nicola Musiu `[一作]` (University of Modena and Reggio Emilia), Garron Fish `[通讯]` (Claytex)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文构建了一个针对自动驾驶赛车的高保真多体车辆动力学仿真环境，并将其与自主驾驶软件实现实时闭环测试。

**💡 创新点**

创新点在于将基于Modelica的Dymola多体模型导出为FMU，结合CRG道路标准，形成可实时运行、可扩展、符合工业标准的仿真框架；同时通过实验数据对Pacejka魔术公式、刹车热模型和温度耦合轮胎模型进行微参数识别，实现了高度可调节的物理模型。

**🔧 技术方法**

技术包括：Dymola + VeSyMA/Motorsports库、Pacejka 6.2魔术公式、热耦合双节点刹车模型、温度耦合轮胎模型、FMU（FMI 2.0）与FMI4cpp C++集成、OpenCRG道路生成、实时显式欧拉积分、线性ARX执行机构模型、可视化GUI (ImGui) 与 PlotJuggler。

**📊 数据集**

使用的数据集为Dallara EAV-25在Yas Marina赛道的实验测量数据（车载CAN、IMU、轮速、制动温度等），以及通过T.R.I.C.K. 进行的轮胎-路面相互作用实验。

**📈 对比分析**

比较方法为将仿真轨迹、纵向速度、侧滑角、侧向误差、转向角等与实车测量进行对齐，利用最大误差和RMSE评估。结果显示仿真最佳圈速与实车相差≤2%，纵向速度误差RMSE<0.5 m/s，侧滑角RMSE<0.004 rad，侧向误差RMSE<0.15 m，表明高保真度且可实时运行（平均TAT≈0.36 ms <1 ms）。

**⚠️ 局限性**

局限性包括：动力传动模型简化（未充分捕捉涡轮迟滞），单点轮胎-路面接触模型（无法完整描述缠绕力），热模型在起始瞬间过热估计误差，以及对复杂曲面（路缘、坑洼）高频细节的平滑处理导致部分真实性下降。

---

## 399. DiffSynth-Music: Audio-Conditioned KV-Cache Adapters for Controllable Music Generation

**arXiv ID:** 2609.12774 | [PDF](https://arxiv.org/pdf/2609.12774v1)

**作者:** Zhongjie Duan `[一作]` (Alibaba Group), Yingda Chen `[通讯]` (Alibaba Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DiffSynth-Music框架，在冻结的ACE-Step-1.5-XL-SFT Diffusion Transformer上加入音频条件模板，实现多模态可控音乐生成。

**💡 创新点**

通过层级键值注入共享VAE latent，将节奏、声乐、伴奏、韵律和参考音频等五种音频控制在同一latent空间组合，首次实现可组合的音频条件控制。

**🔧 技术方法**

使用条件流匹配训练模板模块，结合预训练DiT、共享VAE、声源分离、节拍检测、pYIN等音频预处理技术。

**📊 数据集**

基于约60k首私人音乐录音，包含中英文歌唱样本，使用MUSDB18-HQ和自有数据集并通过Qwen3-Omni生成描述。

**📈 对比分析**

在中英歌唱生成任务中与ACE-Step、DiffRhythm-2、HeartMuLa等基线对比，五种控制指标Beat-F1、V-MSE、A-MSE、Pitch_50、MuLan-A均显著提升，音质与指令遵循指标保持接近基线。

**⚠️ 局限性**

受限于音频预处理误差导致控制信号噪声，控制与生成的融合方式可能在复杂多重控制下表现不佳，且缺少人类听觉评估。

---

## 400. Parameter Sensitivity Analysis for Aerial LiDAR-Inertial Odometries in low-altitude flights

**arXiv ID:** 2609.12837 | [PDF](https://arxiv.org/pdf/2609.12837v1)

**作者:** Robert Milijas `[一作]` (Centre of Excellence in Maritime Robotics and Technologies for Sustainable Blue Economy), Stjepan Bogdan `[通讯]` (Centre of Excellence in Maritime Robotics and Technologies for Sustainable Blue Economy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对低空多旋翼 UAV 的 LiDAR‑Inertial Odometry（LIO）算法（Cartographer 与 FAST‑LIO2）进行参数灵敏度分析，采用全面网格搜索评估不同参数组合对轨迹误差（ATE）的影响。

**💡 创新点**

首次将随机森林的置换重要性与 Pearson 相关性结合，用于量化参数对性能的影响并提炼出两种算法的简化调参方案；该方案在保持性能的同时显著降低了调参复杂度。

**🔧 技术方法**

使用了：完整网格搜索、ATE 误差评估、Pearson 相关系数、随机森林回归器、置换重要性分析、以及对结果的统计验证。

**📊 数据集**

使用公开的 PASTEL（包含 3 种 LiDAR、16 条飞行序列）和 NTU VIRAL（10 条序列）数据集，覆盖低至中等高度飞行与多种环境与 LiDAR 组合。

**📈 对比分析**

通过对比网格搜索得到的最优参数与简化调参方案的结果，验证后者在 94% 的案例中能达到最优 ATE 误差（误差差距 ≤ 5 cm），并在所有序列中保持较低的误差水平。

**⚠️ 局限性**

局限性包括：仅针对低空飞行、仅评估 LIO 算法（未涉及完整 SLAM）、未深入研究参数交互作用、未涵盖高空或不同平台（如大机型）的通用性。

---

## 401. Learning the Lake: Reliable Experience for Adaptive Data Product Discovery

**arXiv ID:** 2609.12754 | [PDF](https://arxiv.org/pdf/2609.12754v1)

**作者:** Yixi Zhou `[一作]` (Hong Kong Baptist University), Haipeng Zhang `[通讯]` (ShanghaiTech University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多模态数据湖中提出了基于记忆的连续数据产品检索框架（Evolving Discovery Memory），并构建了 DPDisc 基准来评估表格与文本的联合检索。

**💡 创新点**

创新点在于将检索状态拆分为静态区域索引与可学习的、源标记的检索记忆，利用运营熟悉度决定搜索规模，利用已确认的产品证据决定搜索位置，同时保留全湖路径做审计或回退。

**🔧 技术方法**

技术手段包括：Dense+Lexical 检索与 Reciprocal‑Rank Fusion、基于嵌入的查询相似度、区域级相似度与记忆优先级融合、基于邻居的熟悉度阈值多级预算决策、记忆更新与确认、全湖回退与审计等。

**📊 数据集**

使用了三大问答衍生的金融数据湖：HybridQA、TAT‑QA 与 ConvFinQA，覆盖 13,076 条验证实例。

**📈 对比分析**

与无记忆全湖检索、仅熟悉度控制以及静态预算等基线对比，实验表明在 TAT‑QA 上通过确认产品证据可提升约 0.06 召回率并在 60% 以上的资产曝光量减少；HybridQA 与 ConvFinQA 在某些配置下未见显著增益。

**⚠️ 局限性**

局限性包括：仅在固定成员的数据湖和单一检索堆栈下验证，无法直接迁移到频繁更新或跨湖环境；记忆仅依据已确认反馈，无法处理延迟或误报；审计与回退虽降低风险但仍不保证完整性。

---

## 402. RunningTensor: Generalizing Linear Attention to Higher-Order Recurrent States

**arXiv ID:** 2609.12814 | [PDF](https://arxiv.org/pdf/2609.12814v1)

**作者:** Luca Herranz-Celotti `[一作]` (ISIR Sorbonne), Vincent Guigue `[通讯]` (ISIR Sorbonne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RunningTensor 模型，将线性注意力和状态空间模型的二阶张量状态提升为更高阶张量，保持线性时间推理；

**💡 创新点**

创新点在于将记忆状态推广为更高阶张量，使用 rank‑1 外积更新并通过多向查询收缩读取，显著提高记忆容量并保持 O(T) 计算；

**🔧 技术方法**

采用线性注意力、状态空间模型、张量外积更新、忘记门、查询跳过门、Delta 规则、ConcOnlyC 低参数通道混合、FusedRMSNormGated 等技术；

**📊 数据集**

使用 FineWeb‑Edu 100BT 进行预训练，并在 Synthetic Multi‑Query Associative Recall（MQAR）数据集以及后续的语言理解与检索下游任务上进行评估；

**📈 对比分析**

在 0.6B 参数规模下，与标准线性注意力及多种 SSM（如 Mamba‑2、GLA、DeltaNet）在 MQAR 长度 1024–4096 上对比，RunningTensor 在所有长度上实现 100% F1，速度快于注意力且比最佳 SSM 更快；预训练后在语言理解和检索任务上亦显著提升；

**⚠️ 局限性**

仅实现了三阶张量，尚未探索更高阶张量的稳定性与效果；真实任务的广泛评估有限；对参数量与计算效率的进一步优化仍待研究。

---

## 403. Vertical Assessment of RF-EMF Exposure in a Building Adjacent to a Multi-Operator Shared Base Station

**arXiv ID:** 2609.12783 | [PDF](https://arxiv.org/pdf/2609.12783v1)

**作者:** Ricardo Q. de F. H. Silva `[一作]` (Federal University of Rio Grande do Norte), Vicente A. de Sousa `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对一栋20层建筑在靠近多运营商共享基站的环境下，按楼层进行全楼 RF‑EMF 暴露测量，记录电场强度并与基站公开参数估算模型对比。

**💡 创新点**

创新点在于将先前提出的基于公开基站参数的垂直暴露预测方法扩展到多运营商共享站点，并通过完整楼层测量验证模型的实用性，首次系统展示了基站天线配置对建筑内部垂直暴露分布的影响。

**🔧 技术方法**

使用 Narda NBM‑520 宽带场计与 EF 0691 电场探头进行 1 min 与 30 min 时间平均测量，计算 RMS 电场强度和暴露比（ER），并结合 ANATEL 标准进行阈值评估。

**📊 数据集**

数据集包括：ANATEL Mosaico 平台提供的基站参数（距离、方位、功率、技术类型、天线倾斜角等）、建筑地理坐标与垂直位置，以及自建的测量数据集（各楼层电场强度、峰值、ER）。

**📈 对比分析**

测量结果与模型预测对比：模型成功识别出高暴露楼层（如 10‑11 层），但实际峰值 31.86 V/m 远高于街道水平峰值 1.33 V/m（≈23.95 倍）。模型在筛选潜在高危楼层方面表现良好，但对具体楼层位置和峰值的精度不足；整体暴露比在 10 层约为 20%–40% 的法规限值，表明暴露处于可接受范围但接近上限。

**⚠️ 局限性**

主要限制包括：基站公开参数存在不确定性（实际倾斜角、机型等未公开）；宽带测量无法区分不同网络技术或频段的贡献；测量仅在建筑外墙进行，未进入住宅内部；且模型未考虑邻近 Wi‑Fi 等其他电磁源的干扰。

---

## 404. MPT: Missing Prototype Tracking via Barycentric Reconstruction in Vehicular Federated Learning

**arXiv ID:** 2609.12771 | [PDF](https://arxiv.org/pdf/2609.12771v1)

**作者:** Hanju Jang `[一作]` (Yonsei University), JeongGil Ko `[通讯]` (Yonsei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在跨车队联邦学习中通过重建稀有类别原型来保持稀有类别识别的框架，能够在车辆离网后仍能准确识别被离网车辆主导的类别

**💡 创新点**

创新点在于将稀有类别原型分解为剩余类别原型的仿射组合（in‑span）与残差（out‑of‑span）两部分，并利用剩余类别原型的漂移和协方差变化来持续更新残差，同时采用自适应残差加权来平衡残差测量与预测

**🔧 技术方法**

核心技术包括仿射（Barycentric）重构、基于协方差的残差预测（Bures–Wasserstein传输）、自适应残差权重、Mahalanobis距离的最近类平均（NCM）读出以及隐私友好的类别级统计量交换

**📊 数据集**

在三组车辆分类任务上评估：nuImages（8类路面物体）、Car-1000粗粒度车辆类别（7类）与细粒度车辆子类别（微型轿车），使用三层CNN、ResNet‑18、MobileNetV3和ViT‑Tiny四种后端网络

**📈 对比分析**

与FedAvg、CCVR、iCaRL‑NME、SDC、LDC以及Oracle等基线对比，实验显示该方法在稀有类别F1上持续逼近Oracle，并在nuImages上获得最高51.6% F1（仅剩1%稀有样本），超过最强基线8.1个百分点；整体通信与计算开销低于大多数基线

**⚠️ 局限性**

主要局限包括：依赖于车辆在离网前能上传完整类别统计；仅在静态停车场模拟中验证，未考虑车辆动态进出与多样化路线导致的更复杂数据分布；对高维、曲线特征空间（如ViT）仍需进一步改进

---

## 405. GraphAHA: Graph-Based Adaptive Search with Heterogeneous Actions for Test-Time Code Generation

**arXiv ID:** 2609.12757 | [PDF](https://arxiv.org/pdf/2609.12757v1)

**作者:** Xitao Li `[一作]`, Xiaofei Xie `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

未提供完整论文内容，无法进行总结

**💡 创新点**

无

**🔧 技术方法**

无

**📊 数据集**

无

**📈 对比分析**

无

**⚠️ 局限性**

无

---

## 406. MGAvatar: Mesh-Bound Gaussians for Head Avatar Geometry and Appearance Modeling

**arXiv ID:** 2609.12850 | [PDF](https://arxiv.org/pdf/2609.12850v1)

**作者:** Lei Shi `[一作]`, Xiao Dong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

未提供论文内容，无法说明

**💡 创新点**

未提供论文内容，无法说明

**🔧 技术方法**

未提供论文内容，无法说明

**📊 数据集**

未提供论文内容，无法说明

**📈 对比分析**

未提供论文内容，无法说明

**⚠️ 局限性**

未提供论文内容，无法说明

---

## 407. Pre-Trained Low-Rank Tensor Decomposition for Multi-Dimensional Image Recovery

**arXiv ID:** 2609.12843 | [PDF](https://arxiv.org/pdf/2609.12843v1)

**作者:** Bing-Zhang Fu `[一作]` (University of Electronic Science and Technology of China), Deyu Meng `[通讯]` (Xi’an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种预训练低秩张量分解（PLTD）框架，利用冻结的DINOv3提取的公共结构与可学习的低秩张量相结合，完成多维图像恢复任务。

**💡 创新点**

创新点：①首次将大规模视觉模型的预训练知识引入张量分解，捕获跨图像的公共结构；②通过仅学习低秩张量而非完整网络，显著降低可学习参数和计算成本；③证明该框架在理论上比传统深度张量分解具有更紧的误差上界。

**🔧 技术方法**

使用技术包括：低秩张量分解（CP、Tucker、块项等）与轻量级卷积变换；DINOv3预训练特征提取；Adam优化；实验中还涉及理论误差界定。

**📊 数据集**

数据集：彩色图像（Butterfly、Airplane、Peppers、House）；多光谱图像（CAVE 256×256×31）；真实遥感图像（NAIP CNIR 256×256×4）；磁共振图像（BrainMRI 144×176×3）。

**📈 对比分析**

与经典浅层张量分解（TNN、HLRTF、MTTD）和无训练深度张量分解（S2NTNN、DTR、DELTA）对比，PLTD在PSNR/SSIM上普遍领先，且可学习参数仅为几百万级，运行时间更短，碳足迹更低。

**⚠️ 局限性**

局限性：依赖预训练模型的域适配（如从RGB迁移到多光谱或医学影像可能产生域差异）；在极端破坏或高度稀疏的采样下，预训练特征仍可能不足；需要预训练的大模型作为前置步骤，增加了总体工作量。

---

## 408. HemaHier: Chain-Conditioned Ordinal Hierarchies for Lineage-Aware Bone-Marrow Cytology

**arXiv ID:** 2609.12835 | [PDF](https://arxiv.org/pdf/2609.12835v1)

**作者:** Afshin Bozorgpour `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 HemaHier 预测头，结合骨髓细胞本体，将细胞识别与血系谱系和链式成熟度耦合，并输出连续的成熟度分数。

**💡 创新点**

创新点包括：① 基于专家制定的成熟链构建层级化本体；② 链条件成熟度回归，仅对健康链监督；③ 细胞类别与谱系共享后验概率，确保层级一致；④ 分阶段训练策略，先稳固识别再加入谱系与成熟度监督。

**🔧 技术方法**

采用自监督骨髓基础模型 DinoBloom，配合适配器/LoRA、链条件成熟度头、线性/概率耦合预测以及排名、回归与树距离损失，进行层级一致的多任务学习。

**📊 数据集**

使用了两套内部骨髓细胞数据集（约17k 细胞，23/24 细胞类）和公开 MLLv1 (BMC) 数据集（171k 细胞，21 细胞类），并统一映射到共享本体。

**📈 对比分析**

与平面线性探针、基础模型线性探针及 HMIL* 单细胞改编版对比；在冻结和 LoRA 微调基础上，HemaHier 在准确率、宏F1、稀有类F1 上均优于对手，尤其稀缺类提升显著；同时交叉谱系错误率和树距离显著下降，证明错误更生物学合理。

**⚠️ 局限性**

主要局限是对专家手工定义的成熟链的依赖；对异常或病变细胞的成熟度缺乏监督，扩展到病理轨迹仍是未来研究方向。

---

## 409. Balancing Emotional Alignment and Semantic Consistency in Image Generation via Reinforcement Learning with Valence-Arousal Anchoring

**arXiv ID:** 2609.12830 | [PDF](https://arxiv.org/pdf/2609.12830v1)

**作者:** Jisheng Dang `[一作]` (Lanzhou University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于anchor-regularized Flow‑GRPO的框架，实现连续情绪（valence‑arousal）控制的图像生成，并显著降低情绪与语义漂移；

**💡 创新点**

创新点包括：1）将deterministic ODE转化为边缘保持的SDE，为Flow匹配生成器提供可计算的转移概率；2）在RL终端奖励中使用冻结的VA回归器实现直接情绪优化；3）引入零VA中性图像作为CLIP空间的语义锚点，约束生成过程避免语义漂移；4）将GRPO与anchor共同训练，实现情绪表达与语义保持的平衡；

**🔧 技术方法**

采用的技术主要有：Flow匹配生成器、Flow‑GRPO强化学习、稀疏LoRA微调、CLIP视觉编码器作为锚点、冻结的VA预测器、边缘保持SDE、群体相对策略优化；

**📊 数据集**

使用的是包含中性与情绪配对提示、连续VA标签的文本-图像数据集，覆盖132个提示、5×5共25种VA坐标，共3300个样本；

**📈 对比分析**

与基线（EmotiCrafter）、SDXL+GRPO+Anchor以及无anchor的GRPO进行对比；结果显示在VA误差（V‑Err、A‑Err）上最小（1.132、1.492），CLIPScore最高（30.179），但IQA略低；

**⚠️ 局限性**

主要限制在于：1）情绪表达与图像质量存在一定权衡，IQA下降；2）anchor约束可能抑制某些情绪表达细节；3）实验仅在SD3.5M与SDXL两大backbone上验证，缺乏更广泛的多模型评估；4）对极端情绪（如(±3,±3)）的语义漂移仍未完全消除。

---

## 410. Evaluating Context Segmentation in Locally Deployable SLMs for Cybersecurity CTF Tasks

**arXiv ID:** 2609.12839 | [PDF](https://arxiv.org/pdf/2609.12839v1)

**作者:** Sebastiano Nordio `[一作]` (Independent), Michele Lotto `[通讯]` (University of Genoa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在硬件受限下使用开源小型语言模型（SLM）进行网络安全CTF任务的代理执行，并提出上下文分段双代理框架。

**💡 创新点**

创新点是引入上下文分段机制，将高层策略与低层执行分离，提升了token效率并在复杂边缘案例中取得更高成功率。

**🔧 技术方法**

使用了双代理架构（Explorer + Worker），并在4‑bit量化的Gemma‑4‑E2B‑it 与 Gemma‑4‑E4B‑it 模型上进行推理。

**📊 数据集**

使用改造的 Intercode 平台提供的 CTF 挑战集作为评估数据集。

**📈 对比分析**

与基线单代理、plain best‑of‑k 重复执行对比，plain+explorer 在 Gemma‑4‑E4B‑it 上取得 0.39 奖励，仅用 8652.52 token，显著高于 plain best‑of‑4（0.38 奖励，10874 token），并在复杂任务中成功率提升 18.52%。

**⚠️ 局限性**

局限在于模型容量不足导致大多数任务未解，当前仅评估两款模型，且手工 prompt 未最优，未来需扩展模型、多样化任务与完善长时记忆。

---

## 411. Online Video Agent Harness for Long Video Understanding

**arXiv ID:** 2609.12818 | [PDF](https://arxiv.org/pdf/2609.12818v1)

**作者:** Sen Yang `[一作]` (Baidu Inc), Hua Wu `[通讯]` (Baidu Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种纯在线的视频理解代理框架VideoXAgent，能够根据用户查询动态拆解任务并按需调用专家工具获取证据；

**💡 创新点**

创新点在于：①基于MINERVA专家推理轨迹挖掘的原子能力分类体系，指导工具设计；②构建60+多模态专家工具套件；③实现预算感知的ReAct式代理循环，避免幻觉与无限循环；

**🔧 技术方法**

使用技术包括：ReAct式代理架构、VLM（Gemini-3.1-Pro-preview或Qwen3.6-35B-A3B）、多模态专家工具（检测、OCR、ASR、面部识别等）、预算控制中间件、结构化视觉证据提问协议；

**📊 数据集**

评估数据集：Video-MME-Long、LongVideoBench-Long、LVBench、MINERVA、Video-MME-v2；

**📈 对比分析**

与最前沿LMM和视频代理比较，VideoXAgent在大多数长视频基准上实现与LMM相近的准确率，同时显著降低视觉输入帧数与上下文长度（约50k token），在MINERVA上也能达到与顶级LMM相当的65%级别；

**⚠️ 局限性**

局限性包括：依赖于MINERVA推理轨迹挖掘的能力词表可能偏向该数据集，工具稀缺或冗余；预算参数需调优；未衡量完整算力成本，仅关注视觉帧与上下文长度；

---

## 412. VertiFuseX: Generalizable Financial Forecasting via Multi-Stream Temporal Fusion

**arXiv ID:** 2609.12793 | [PDF](https://arxiv.org/pdf/2609.12793v1)

**作者:** Aashish Bohra `[一作]` (Indian Institute of Technology Jodhpur), Vivek Vijay `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种用于股票价格预测的垂直融合LSTM模型VertiFuseX。

**💡 创新点**

主要创新是将多尺度时间特征在倒数第二层进行垂直拼接融合，而非传统决策级融合，从而保留丰富中间表示并提升跨市场泛化。

**🔧 技术方法**

采用LSTM、Bi‑LSTM、St‑LSTM三条时序分支以及并行DNN，在训练时端到端优化并使用固定超参数、dropout和L2正则化。

**📊 数据集**

使用2010–2024年十个全球指数（S&P500、DJIA、NASDAQ、Nikkei225、FTSE100、DAX、HSI、KOSPI、NYSE、NSE）共15年收盘价。

**📈 对比分析**

在严格的时间序列划分下，VertiFuseX相较于单一LSTM、Bi‑LSTM、St‑LSTM以及七个最先进模型，在MAE、RMSE、MAPE上提升30–54%（平均>40%），并在经济模拟中降低最大回撤、提升风险调整收益。

**⚠️ 局限性**

主要局限是仅在单变量收盘价上验证，缺少多因素扩展与更长预测 horizon 的实验；模型仍需在极端高波动期检验鲁棒性。

---

## 413. A Randomized $\frac32$-Approximation for Strategic Facility Location on a Circle

**arXiv ID:** 2609.12792 | [PDF](https://arxiv.org/pdf/2609.12792v1)

**作者:** Hau Chan `[一作]` (University Of Nebraska Lincoln), Chenhao Wang `[通讯]` (Beijing Normal University Zhuhai)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并分析了一种基于随机化的、对策略无关的单设施定位机制（MRP），在圆形网络上对利用率社会成本实现了 3/2 的近似比率，并给出了 11/10 的下界。

**💡 创新点**

创新点包括：①提出统一的偶奇数代理数机制，即在偶数时混合 Random Dictator 与随机删除扩展的 PCD；②对传统的“切割圆形”分析方法进行细化，精确追踪切割误差并将其与 PCD 的收益关联；③证明 3/2 近似比率在大规模代理时是渐近紧的，并在四代理实例上提升了现有的最优下界。

**🔧 技术方法**

主要技术手段是：对称性（匿名、平衡）简化分析；基于切割的线性度量估计；对 PCD/ EPCD 选取概率的逆向增量分析；使用 Cauchy 等不等式将误差分配给可用的交叉节省；以及通过线性规划与两情景分析构造下界。

**📊 数据集**

本工作完全是理论分析，无使用任何实验数据集；所有结论通过严格证明给出。

**📈 对比分析**

与已有方法相比：在偶数代理数下突破了 7/4 的上界，提供统一的 3/2 上界；与已知 1.0456 的下界相比，提升到 1.1；在大规模代理时近似比率逼近 1.5，表明该机制在理论上已经相当接近最优，但仍有 1.1 与 1.5 之间的空隙。

**⚠️ 局限性**

局限性包括：①仍存在 11/10 与 3/2 的巨大理论空隙，尚未证明 3/2 是否最优；②仅针对圆形网络，无法直接推广到更一般的图或多设施；③机制仅为线性混合 RD 与 PCD/EPCD，可能无法进一步提升；④证明依赖于精细的代数推导，难以直接应用于更复杂的约束或多目标情形。

---

## 414. Transducer Placement and the Limits of a Four-State Reduced Model in Post-Flutter Piezoelectric Energy Harvesting from a Pitch-Plunge-Flap Aerofoil

**arXiv ID:** 2609.12788 | [PDF](https://arxiv.org/pdf/2609.12788v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (National Technical University Of Athens), Andrew J. McCracken `[通讯]` (Daskalos Apps)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在三自由度机翼上嵌入压电传感器，构建15状态电-气-弹系统，并用矩阵无关投影生成4状态非线性ROM。

**💡 创新点**

创新点在于揭示支架自由度决定flutter边界偏移符号及功率提升倍数，并指出传统基于特征值展开的速度依赖导致ROM误差的根本原因，提出通过重新投影修正的方案。

**🔧 技术方法**

使用Taylor展开残差的矩阵无关投影、特征向量双正交、基于复特征值的投影、再投影方法以及截断误差与速度展开误差分离等技术。

**📊 数据集**

使用自构造的机翼参数（质量比、频率比、弹性轴位置、翼弦等）以及基于strip理论的无稳流动模型生成15状态系统；并无公开数据集。

**📈 对比分析**

将ROM的限制周期振幅、功率、周期与全阶非线性FOM进行比较，发现通过再投影后峰值幅度误差从46%降至3.4%，功率低估仍在20‑34%范围，且在15状态下无时间加速。

**⚠️ 局限性**

局限在于流体模型仅为二维无压缩贴合、压电线性且仅阻性负载；ROM对采集量的误差仍较大；对更高阶、三维或受控负载未验证；且在十五状态下无计算加速。

---

## 415. Convergence of Stochastic Gradient Methods under Heavy-Tailed Noise and Hölder Smoothness

**arXiv ID:** 2609.12785 | [PDF](https://arxiv.org/pdf/2609.12785v1)

**作者:** Misbah Uz Zaman `[一作]` (Indian Institute of Science Education and Research Kolkata), Anirbit Mukherjee `[通讯]` (University of Manchester)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在非凸随机优化中，作者在同时放宽梯度光滑性为 Hölder 连续和噪声方差无限大（α-稳定分布）的情况下，给出了 SGD、δ-GClip 与 GClip 的收敛证明。

**💡 创新点**

创新点是首次将 Hölder 光滑性与重尾噪声联合考虑，得到新的收敛率，并且在 α<1+s 的非常重尾情形下给出了 GClip 的首个收敛保证。

**🔧 技术方法**

主要使用了 Hölder 级的下降不等式、α-阶矩边界、分段梯度剪裁分析以及马尔科夫不等式等理论工具。

**📊 数据集**

该工作为理论研究，无实验数据集，全部基于分析证明。

**📈 对比分析**

与传统的 SGD（O(T^{-s/(1+s)})）和已知的 GClip（仅在 Lipschitz 情形下）相比，作者的 δ-GClip 在 Hölder 情形下实现了 O(T^{-2s(α-1)/[(1+s)(2α-1)]}) 的更优收敛速率；在极重尾 α<1+s 时，GClip 则是唯一有理论保证的方法。

**⚠️ 局限性**

局限性包括缺乏最优性下界、未给出高概率收敛结果，以及对参数选择与剪裁阈值的理论指导仍相对粗糙。

---

## 416. Curriculum-Based Adversarial Heterogeneous Agent Reinforcement Learning for Autonomous Quad-Copter Landing in Maritime Settings

**arXiv ID:** 2609.12758 | [PDF](https://arxiv.org/pdf/2609.12758v1)

**作者:** Allan Minh-Tam Nguyen `[一作]` (Maastricht University), Rico Möckel `[通讯]` (Maastricht University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了基于对抗性多智能体强化学习的海上无人机与机械臂协同捕捉与降落框架。

**💡 创新点**

创新点在于将逆向对抗风力与波浪耦合的双向课程学习结合进HAPPO框架，显著提升了在未见海况下的稳健性。

**🔧 技术方法**

使用技术包括Heterogeneous‑Agent Proximal Policy Optimization (HAPPO)、Isaac‑Lab仿真、Dryden风模型与Pierson–Moskowitz波谱、双向课程调节及零和对抗策略。

**📊 数据集**

数据集为基于NVIDIA Isaac Lab的仿真生成的海况风浪场景，覆盖海况0至10的多种风速与波高参数。

**📈 对比分析**

与传统域随机化 (DR) 及单一海况基线相比，HARL‑AC 在海况0/4/5的成功率可达97.5%并在海况7/8/10的成功率提升约15%，同时崩溃率下降14%。

**⚠️ 局限性**

局限在于仿真环境仍可能与真实海况差异较大，且对抗训练导致的时间延长和较高超时率可能影响实时部署。

---

## 417. GenOR-Twin: A Semantic Middleware for Integrating Operational Discourse with Mathematical Optimization

**arXiv ID:** 2609.12863 | [PDF](https://arxiv.org/pdf/2609.12863v1)

**作者:** Rahimeh Neamatian Monemi `[一作]` (Sharkey Predictim Globe), Nelson Maculan `[通讯]` (Henan Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了GenOR-Twin框架，将大型语言模型作为语义翻译器，实现从运营日志到优化模型的动态约束注入，形成数字孪生的双向耦合；

**💡 创新点**

在传统组合优化中首次将LLM限定为语义接口而非求解器，结合符号验证与自适应调度策略，突破“Translation Gap”，实现实时可验证的约束注入和调度决策；

**🔧 技术方法**

使用GPT‑4o（Chain‑of‑Thought、RAG、Self‑Reflection）+ 结构化知识图+符号验证器+CP‑SAT求解器+自适应调度器；

**📊 数据集**

基于300条手工标注的跨六大组合优化领域（JSSP、VRP、RCPSP、NSP、BPP、Max‑Flow）日志和规模化仿真实例；

**📈 对比分析**

与静态、正则表达式、LLM‑only、BERT‑NER等基线比较，平均提升3–6%总加权迟到，决策延迟从15分钟降至约2分钟；在50k+工序大规模实例中，整体开销低于10%；

**⚠️ 局限性**

需为每个新领域设计约束模板和知识图，LLM偶尔产生幻觉需多层验证；适用范围局限于可文本描述的中断，且在极高频率或安全关键场景仍需人工审查。

---

## 418. Very Exciting: Zero-Shot Model Predictive Control of Buildings via Excitation-Based Generalized Transfer Learning Models

**arXiv ID:** 2609.12853 | [PDF](https://arxiv.org/pdf/2609.12853v1)

**作者:** Fabian Raisch `[一作]` (Technical University of Applied Sciences Rosenheim), Benjamin Tischler `[通讯]` (Technical University of Applied Sciences Rosenheim)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在目标建筑上零样本迁移学习预训练的通用模型，用于模型预测控制，并与标准操作数据预训练模型及基线控制器进行对比。

**💡 创新点**

通过在源建筑上使用激励式（PRBS、斜坡、随机游走）数据预训练，显著缩小预测-控制性能差距，使零样本通用模型在控制性能上优于线性模型和PI控制器。

**🔧 技术方法**

采用多源预训练的通用模型（MLP、LSTM、Transformer），结合激励实验生成的数据，使用梯度优化的非凸MPC求解，并与在线线性回归和PI控制器进行基线比较。

**📊 数据集**

在BuilDa仿真环境中生成64个源建筑与32个目标建筑的冬季运营数据，分别使用标准操作与三种激励策略进行预训练。

**📈 对比分析**

在32个目标建筑上进行一次月MPC评估，衡量总成本、舒适度与加热活性；激励预训练的MLP（斜坡）在总成本上比最佳线性模型低6.4%、比PI控制器低36.9%，并显著降低预测误差与控制误差。

**⚠️ 局限性**

仅在仿真单区住宅建筑上验证，未在真实建筑上测试；零样本无微调，未探索细化微调策略；仅考虑加热系统，未扩展至多区或制冷。

---

## 419. CoralscapesV2: Panoptic and Fine-Grained Visual Scene Understanding in Coral Reefs

**arXiv ID:** 2609.12826 | [PDF](https://arxiv.org/pdf/2609.12826v1)

**作者:** Jonathan Sauder `[一作]` (Massachusetts Institute of Technology), Guilhem Banc-Prandi `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究构建并发布了CoralscapesV2数据集，增加了帧数、注释密度和质量，细化标签至95类，并为每帧提供65k个鱼实例的全覆盖掩码；

**💡 创新点**

创新点包括将原39类细化为95细粒度视觉类别，首次在珊瑚礁图像中实现全景分割并加入鱼的逐帧实例掩码，显著提升了监测数据的细节与可用性；

**🔧 技术方法**

技术上利用视频上下文进行鱼实例标注，采用Segment Anything Model (SAM) 预先分割后人工校正，随后使用SegFormer、DINOv3+LoRA+DPT以及Mask2Former等前沿分割网络进行模型训练与基准评测；

**📊 数据集**

数据来源于Red Sea地区35-45个潜水点的潜水员摄像机（GoPro Hero10、Sony 7R4）收集的2433帧，涵盖原Coralscapes V1数据并在此基础上进行重标注与扩充；

**📈 对比分析**

通过对比训练/评测，语义分割模型在V2上mIoU从约57%提升至约62%，实例分割实验表明使用视频上下文的Mask2Former略高于单帧模型，但差异不大；

**⚠️ 局限性**

局限性在于对鱼群密集或极小鱼类的精确标注仍具有挑战性，且95类细粒度标签导致训练/验证/测试集难以覆盖所有类别，增加模型训练难度。

---

## 420. What an odour descriptor corpus can and cannot measure: valence, attenuation, and the ceiling of the public record

**arXiv ID:** 2609.12875 | [PDF](https://arxiv.org/pdf/2609.12875v1)

**作者:** Stylianos Kampakis `[一作]` (Tesseract Academy), Fabio Rovai `[通讯]` (Tesseract Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四个公开气味描述词语料库进行审计与比较，检验不同语料库之间的测量一致性，分离判准与构造，量化分子结构与标签能解释的感官相似度比例，并识别并量化缺失的情感维度（价值）。

**💡 创新点**

提出并证明分子-描述词双重观察的条件McNemar检验为精确条件检验，分离判准与构造后使用tetrachoric相关与Cohen κ评估一致性；量化公共记录对感官相似度的解释比例，明确价值为主要缺失维度并给出其测量成本；发布完整词汇交叉映射并进行Lean 4形式化验证。

**🔧 技术方法**

使用精确条件逻辑模型、McNemar检验、随机效应Meta分析、tetrachoric相关、Cohen κ、梯度提升机、岭回归、Morgan指纹与RDKit物理化学描述符、语料交叉映射及Lean 4形式化证明。

**📊 数据集**

Pyrfume分发的Dravnieks、Keller et al.、Leffingwell、GoodScents四个公开气味数据库，共计2225个共享分子，分别包含146、20、113、666个描述词。

**📈 对比分析**

将模型预测的相关系数与人类面板的可靠度做比较，计算模型对可测感官相似度的比例：Morgan指纹仅占14.6%，RDKit块32.9%，两者加所有标签33.9%，单一愉悦度评分则达到54.6%。表明现有结构与标签只能解释约三分之一的可测感官相似度。

**⚠️ 局限性**

主要局限在于仅依赖单一人类面板判准，缺乏对化学结构细节（立体化学、浓度等）的信息，训练数据与标签来源缺乏独立性，跨语言/文化的描述词差异未完全解决，且无法通过现有标签推断价值。

---

## 421. 3D CT-to-PET Translation via Latent Brownian Bridge Diffusion

**arXiv ID:** 2609.12860 | [PDF](https://arxiv.org/pdf/2609.12860v1)

**作者:** Sarita Mourya `[一作]` (Università Magna Græcia di Catanzaro), Paolo Soda `[通讯]` (Umeå University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了一种基于潜在空间的3D CT‑to‑PET翻译框架，首先使用对比学习增强的VAE对CT与PET图像进行潜在对齐，然后在潜在空间中应用Brownian Bridge Diffusion进行跨模态翻译，最终解码重建3D PET体积。

**💡 创新点**

创新点包括：① 在潜在空间对CT和PET进行对比学习对齐，显著降低跨模态差距；② 采用Brownian Bridge Diffusion模型将源潜在直接“桥接”到目标潜在，实现稳定且高保真度的翻译；③ 在3D潜在空间进行翻译，兼顾体积上下文且计算量显著降低。

**🔧 技术方法**

技术方法：变分自编码器（VAE）+ InfoNCE 对比学习；Brownian Bridge Diffusion（BBDM）在潜在空间；DDIM采样；3D 滑动窗口切片；多任务损失（重构、KL、对比、感知、对抗）。

**📊 数据集**

使用公开数据集：① FDG‑PET‑CT‑Lesions（Tübingen）内部训练 864 对 + 测试 150 对；② ENHANCE.PET 1.6k 外部验证，排除重叠后 579 对。

**📈 对比分析**

与三种竞争方法（3D GAN、3D LDM、CPDM）以及 Pix2Pix 进行对比。评估指标包括 PSNR、SSIM、Dice(阈值1.5/2.5)、SUV_max、SUV_mean、MTV、TLG。实验结果显示，X‑Bridge 在 Lesion‑level 指标上均优于对手，PSNR/SSIM 与 Pix2Pix 相当且在外部数据集表现出更小的性能波动，表明跨数据集鲁棒性最佳。

**⚠️ 局限性**

局限性：① 仅在肺部区域验证，缺乏全身或其他器官的评估；② 滑动窗口拼接采用均匀平均，可能导致相邻块边界不连续；③ 对极少见或高SUV小病灶的合成仍不够理想；④ 目前未提供不确定性估计，任务本身具有高度欠约束性。

---

## 422. MIMA: Multi-Interest Recommendation via Multi-Positive Exclusive Assignment

**arXiv ID:** 2609.12842 | [PDF](https://arxiv.org/pdf/2609.12842v1)

**作者:** Xingyuan Mao `[一作]` (Alibaba International Digital Commerce Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commerce Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种多兴趣推荐框架MIMA，通过多正样本专属分配来驱动兴趣向量的多样化，并加入用户兴趣路由来校准跨兴趣评分。

**💡 创新点**

创新点在于：① 将单正样本训练转化为多正样本训练，利用多正样本共同可见性实现兴趣差异化；② 使用因果Transformer解码器生成互补兴趣；③ 采用Hungarian精确的一对一兴趣-正样本分配；④ 引入轻量化路由网络估计兴趣激活概率，解决跨兴趣评分不一致问题。

**🔧 技术方法**

核心技术包括：多正样本正例构造、因果Transformer解码器、Hungarian匈牙利算法、用户兴趣路由的两层MLP、样本Softmax损失、hinge margin路由损失。

**📊 数据集**

在三个公开数据集（Books、Beauty、Gowalla）和一个工业规模数据集（Lazada泰国市场）上进行实验；公开数据集采用时间窗口生成正样本集合；工业数据集直接按实际请求组装正样本。

**📈 对比分析**

与单兴趣模型（POP、YouTube‑DNN、GRU4Rec）以及多兴趣模型（MIND、ComiRec、PIMI、RE4、REMI、DisMIR、NPRec）对比，MIMA在Recall@N、NDCG@N、HitRate@N等指标上均比最佳基线提升3%–16%，并在工业数据集上实现HR提升约13%与IDM提升18%；在线A/B测试中交易量与交易额均提升约5.5%。

**⚠️ 局限性**

局限性包括：① 需要正样本集合的构造，若正样本稀缺或并发性不足可能影响效果；② 多正样本训练与Hungarian分配会增加计算成本；③ 对极大规模兴趣数时仍可能出现过度细分导致噪声；④ 目前主要针对候选匹配阶段，后续排序阶段的兼容性需进一步验证。

---

## 423. Self-supervised Pre-training Helps Retinal Disease Progression Modelling Most When Data Is Scarce

**arXiv ID:** 2609.12834 | [PDF](https://arxiv.org/pdf/2609.12834v1)

**作者:** Ifeoma Veronica Nwabufo `[一作]` (University of Tübingen), Philipp Berens `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文研究了如何利用大规模横断面视网膜图像的自监督预训练，来提高在小样本纵向随访数据上预测年龄相关黄斑变性（AMD）晚期进展时间的性能。

**💡 创新点**

创新点在于系统比较了三种自监督目标（对比学习、遮挡自编码、知识蒸馏）以及通用与领域专用基础模型，揭示预训练目标在冻结、低数据量下是决定模型迁移效果的关键因素，而不是预训练语料规模或域匹配。

**🔧 技术方法**

采用了SimCLR、MAE、DINOv2等自监督算法，并结合线性或MLP读出层，使用Cox比例风险模型进行生存分析；对比了冻结与微调两种评估协议。

**📊 数据集**

使用了德国国家队列眼科数据集NAKO（15.3万张横断面眼底图）进行预训练，以及长期随访的AREDS数据集（8,784只眼，55,173张图）进行下游评估。

**📈 对比分析**

通过C-index和积分Brier分数对模型进行比较，结果显示：冻结时，SimCLR和DINOv2在数百个标注样本下即可达到临床可用的判别力（C-index≥0.75），而微调后所有模型性能趋同，MAE在低样本时表现不佳但在微调后能赶上。

**⚠️ 局限性**

局限性包括仅在一种疾病（AMD）与一种影像模态（彩色眼底摄影）上验证，缺乏对其他进展任务或OCT等模态的泛化评估；并且未与先前的AREDS进展模型直接比较性能。

---

## 424. VertexCBF: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search

**arXiv ID:** 2609.12831 | [PDF](https://arxiv.org/pdf/2609.12831v1)

**作者:** Bojan Derajić `[一作]` (AUMOVIO), Wolfgang Hönig `[通讯]` (Technical University of Berlin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出VertexCBF框架，用神经网络学习逼近稳态Hamilton–Jacobi安全值函数，从而得到可解释的控制障碍函数；

**💡 创新点**

创新点在于将物理约束（HJB‑VI）与基于控制多边形顶点的有限时域树搜索生成的监督标签相结合，采用残差参数化保证安全集不超出原始约束，并实现GPU并行顶点搜索；

**🔧 技术方法**

使用控制仿射动力学、凸多面体控制集、稳态HJB‑VI、具有正弦隐藏激活的MLP、Softplus残差输出、物理信息损失、束宽搜索树搜索及GPU并行化技术；

**📊 数据集**

在15个不同维度的仿真动力学系统上生成标签（通过顶点约束树搜索），并在移动机器人硬件实验中验证；未使用公开数据集；

**📈 对比分析**

与仅使用PDE损失的基线及基于MPPI的全控制监督进行对比；通过误差安全率、误报率及有效安全体积三指标评估，VertexCBF在降低误报率、提升有效安全体积方面优于两种基线；

**⚠️ 局限性**

缺乏正式的收敛或误差界定，依赖后训练验证；树搜索对控制顶点数和时域长度敏感，导致在高维或控制顶点多的系统上计算成本高；

---

## 425. K-Bench: A Benchmark for LLM Unlearning in Agentic Deployments

**arXiv ID:** 2609.12808 | [PDF](https://arxiv.org/pdf/2609.12808v1)

**作者:** Guangsheng Yu `[一作]`, Xu Wang `[通讯]` (University Of Technology Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 K-Bench，基于 ReAct 代理的多通道评估框架，用于测量语言模型在忘记个人身份信息（PII）时的实际安全性，并评估 20 种现有无学习方法在四种存储子系统（参数、上下文、文本检索、结构检索）上的表现。

**💡 创新点**

创新点：① 将多通道（思路、工具调用、检索结果、答案、摘要等）与单子系统注入协议结合，解决单通道评估的覆盖盲区；② 设计 K-Score 与 K‑Class 判别体系（K‑REF、K‑SUP 等）统一衡量忘记、保持与代理崩溃的三项指标；③ 提供开放的评测基准、数据集与完整实验脚本，方便对比与复现。

**🔧 技术方法**

技术：ReAct 代理、Llama‑3.1‑8B、Mistral‑7B、Qwen3.5‑9B；LoRA 微调、梯度优化、激活编辑（RepE、MLP‑probe、R‑LACE）等 13 种无学习方法；多通道日志抽取与逻辑或聚合；统计检验（配对 McNemar、Benjamini‑Hochberg、Bootstrap）。

**📊 数据集**

数据集：使用 Faker 生成的 5,000 条合成 PII 记录（出生日期、地址、职业、雇主），按 1,000 条忘记集与 4,000 条保留集划分；包含训练/评估拆分，确保不与预训练数据重叠。

**📈 对比分析**

比较与性能：在三种基准模型和四种子系统上评估 20 种方法。K‑Score 最高值为 0.763（FLAT，Llama‑3.1‑8B）、0.726（RMU，Mistral‑7B）和 0.743（LoKU，Qwen3.5‑9B）。多数方法要么出现代理崩溃（如 MLP‑probe、R‑LACE）、要么保持集被破坏（如 Cha、StaR）、要么仅在单通道下表现好而多通道下失效（如 TOFU、MUSE）。K‑Class 结果显示 2/3 方法在多通道下被判为 K‑SUP 或失效，验证单通道评估的不足。

**⚠️ 局限性**

局限：① 仅在 ReAct 代理框架下评估，无法覆盖所有部署模式；② 采用合成 PII，真实世界数据泄漏模式可能更复杂；③ 只测试可修改权重或前向介入的 20 种方法，未覆盖所有潜在技术；④ 单子系统注入协议虽消除交叉影响，但在真实应用中多源混合的情况仍未充分考察；⑤ 对模型崩溃的判定仍依赖经验阈值，可能遗漏轻微失效。

---

## 426. LGFN: Lightweight Gated RGB-Polarization Fusion with Modality-Availability Conditioning for Camouflaged Object Detection

**arXiv ID:** 2609.12798 | [PDF](https://arxiv.org/pdf/2609.12798v1)

**作者:** Zhuangfan Huang `[一作]` (Foshan University), Haishu Tan `[通讯]` (Foshan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种轻量级门控RGB-偏振融合框架LGFN，用于伪装目标检测，并支持仅RGB或RGB+偏振的两条独立推理路径。

**💡 创新点**

创新点包括：1) 确定性模态路由器根据偏振可用性动态选择专门优化的单模态或多模态路径；2) 可用性条件模态门通过模态可用性向量直接校准可用的DoLP和AoP分支；3) Gated Polarization Hub在偏振域内部协调DoLP、AoP及显式结构线索；4) RGB–Polarization Cross Fusion以残差方式将协调整合的偏振特征注入RGB特征，并使用空间与通道注意力调节。

**🔧 技术方法**

采用多种技术：门控机制、可用性条件权重、SE风格通道重校准、残差交叉融合、双尺度注意力、辅助多模态损失（融合一致性、门控正则化）、以及轻量级PVT-v2-B2骨干和FPN解码器。

**📊 数据集**

主要使用PCOD_1200（完整230图像测试集）、COD10K（固定405/200图像子集）以及NC4K（固定200图像子集）作为训练和评估数据集。

**📈 对比分析**

与九种基准RGB方法相比，RGB-only路由在所有六个指标上均取得最优或接近最优；与PolarNet、IPNet的多模态对比中，LGFN在MAE、Dice、IoU等指标上超越两者，并在参数量、FLOPs和时延上显著更高效（RGB-only仅7.4 ms，参数25.1 M，FLOPs10.4 G）。

**⚠️ 局限性**

局限性：仅在单一偏振基准上验证；RGB和多模态采用独立训练的检查点，未共享参数；对不同传感器条件的偏振可靠性建模不够细粒度，可能在更复杂环境下表现不稳定。

---

## 427. Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy

**arXiv ID:** 2609.12791 | [PDF](https://arxiv.org/pdf/2609.12791v1)

**作者:** Gengxian Zhou `[一作]` (Beijing University of Posts and Telecommunications), Cheng-Lin Liu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了基于认知循环（plan‑explore‑reflect）的 CoG 框架，实现对全局知识图与文本的自适应检索与推理，能够在多跳问答中构建证据链。

**💡 创新点**

创新点包括：①训练无关的闭环认知策略，使系统能够主动规划、反思并动态调整；②深度双向图‑文本协同，文本实体主动指导图探索，弥补 KG 稀疏与不完整；③结合实体链接、关系筛选与事实剪枝的分层检索流程。

**🔧 技术方法**

主要技术：大型语言模型（Qwen3‑32B）+向量检索（bge‑m3、Qwen3‑Embedding‑4B）+多轮 Prompting 与记忆模块（Notebook、Candidate Pool、Interaction History）+双源检索（KG + 文本）+自适应实体链接、关系过滤、事实裁剪。

**📊 数据集**

使用七个多跳 QA 基准（KGQAGen、CWQ、QALD10‑en、WebQSP、2WikiMQA、AdvHotpotQA、MusiQue），知识库为完整的 Wikidata 与 Wikipedia。

**📈 对比分析**

与 LLM‑only、Text‑RAG、KG‑RAG、Hybrid‑RAG 等基线对比，CoG 在所有数据集上显著提升，平均 EM 达 56.3%，在最难的 MusiQue 上提升至 27.8%（相比 20.6%），相对 ToG‑2 提升 14% 以上，验证了其强大的跨源推理与探索效率。

**⚠️ 局限性**

局限性：迭代认知循环导致推理延迟与 token 成本高于单次检索方法；对底层 LLM 的指令遵循与长上下文处理能力高度依赖，弱 LLM 在规划、反思等环节仍可能表现欠佳。

---

## 428. LG-PF: Lightweight Confidence-Guided Polarization Image Fusion

**arXiv ID:** 2609.12787 | [PDF](https://arxiv.org/pdf/2609.12787v1)

**作者:** Zhuangfan Huang `[一作]` (Foshan University), Xiaosong Li `[通讯]` (Foshan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种轻量化、基于置信度的偏振图像融合框架LG-PF，利用S_0为结构基准，Selective Residual Transfer方式实现DoLP信息的精准融入。

**💡 创新点**

创新点在于：①引入Polarization Confidence Prior（PCP）估计空间置信度；②Mask-guided Multi-scale Fusion（MMF）在多尺度上按置信度调控信息传递；③轻量化Context-aware Bounded Correction Head（LCH）对局部光照与结构进行稳定修正；④将置信度嵌入损失函数，强化可靠细节保留。

**🔧 技术方法**

核心技术包括卷积编码器/解码器、非共享双流特征提取、置信度门控、多尺度融合、深度可分离卷积与轻量化点卷积、GELU激活、双模交叉损失（结构频率、正则化与教师一致性）。

**📊 数据集**

使用自己构建的MSP数据集，包含1000对像素对齐的S_0–DoLP图像，覆盖17个室内外场景；同时在PIF与GAND公开数据集进行跨数据集评估。

**📈 对比分析**

在MSP上与六种主流方法（CPIFuse、DT-F、LFDT、PAPIF、PIPFNet、TIPFNet）对比，LG-PF在六项评估指标（EN、SF、SD、SCD、MS-SSIM、Q_CB）均夺得第一或第二名，性能优于对手；在PIF与GAND上亦保持竞争力，无需微调。

**⚠️ 局限性**

局限性在于：①仍需在更复杂光照与极端噪声条件下验证鲁棒性；②对极化传感器的硬件兼容性尚未全面评估；③模型虽然轻量，但在极低算力设备（如嵌入式）上部署仍需进一步优化。

---

## 429. Unified Agentic Video Editing Across Levels of Complexity and Creativity

**arXiv ID:** 2609.12769 | [PDF](https://arxiv.org/pdf/2609.12769v1)

**作者:** Surabhi S. Nath `[一作]` (Bertelsmann AI Hub), Lion Schulz `[通讯]` (Bertelsmann AI Hub)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在一部内部制作的剧集上，利用层次化文本元数据和代理式 AI 设计并实现了三种视频剪辑任务：预览、视频摘要和电影式预告片。

**💡 创新点**

创新点在于：①将同一层次化文本元数据统一用于不同复杂度任务；②使用多轮 LLM 与工具调用实现代理式编辑；③通过多维度评估（点击率、信息量、剧情完整性）验证任务差异。

**🔧 技术方法**

采用大型语言模型（LLM）与多功能工具调用（脚本检索、音频分离、字幕生成等）、代理式规划框架以及文本元数据接口。

**📊 数据集**

数据集为该剧集前五集的元数据，元数据由内部视频理解系统产生，包含剧集、场景、镜头级别的时间戳描述与转录。

**📈 对比分析**

通过 2‑AFC 点击实验、剧情完整性（是否出现高潮/结局）、问答测验等方式比较：预览在点击率上显著优于随机与无约束 LLM，视频摘要使 LLM 正确率从 69% 提升到 96%，预告片点击率最高、信息量最低。

**⚠️ 局限性**

局限在于仅评估单一剧集的 5 集、成本高昂的代理执行链、可能出现错误级联、缺乏跨类型通用性、存在版权与安全隐患。

---

## 430. Same Encoder, Different Winner: A Paired-View Framework for Cell Painting Encoder Evaluation

**arXiv ID:** 2609.12761 | [PDF](https://arxiv.org/pdf/2609.12761v1)

**作者:** Tim Treis `[一作]` (Helmholtz Munich), Fabian J. Theis `[通讯]` (Helmholtz Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CP-BG-Bench框架，利用四种配对视图评估Cell Painting视觉编码器的性能；

**💡 创新点**

创新点在于通过固定细胞中心，分别去除或加入背景与密度信息，构建可控干预的四视图，揭示评估指标间差异的三条轴；

**🔧 技术方法**

使用多视图图像处理、LoRA微调、对比学习目标、Harmony批次校正、以及CellProfiler特征预测等技术；

**📊 数据集**

评估了三大Cell Painting数据集：JUMP-CP（化合物），RxRx1（siRNA），RxRx3-core（CRISPR-KO），并使用三种编码器（DINOv3 ViT-B/16、OpenPhenom、SubCell）；

**📈 对比分析**

通过四个标准协议（replicate mAP、scIB、CellProfiler特征预测、跨批次检索）比较，发现不同指标给出不同排名，说明单一指标不可靠，且背景/上下文对性能有显著影响；

**⚠️ 局限性**

局限性包括仅针对三数据集、三编码器、单一训练与批次校正方法、单次训练种子、以及对不同采样策略和目标的依赖，未来需多种方案验证。

---

## 431. Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per Proposition

**arXiv ID:** 2609.12960 | [PDF](https://arxiv.org/pdf/2609.12960v1)

**作者:** Devansh Sharma `[一作]` `[通讯]`, Devansh Sharma

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对梵语在当前 NLP tokenizer 下的 token‑cost 进行系统实验，评估不同 tokenization 方法在并行语料上的 tokens‑per‑proposition 影响。

**💡 创新点**

通过构建与英语匹配的 tokenizer 对照组，分离语言本身的词汇密度与 tokenizer 训练域/大小对 token‑cost 的影响；进一步揭示 tokens‑per‑proposition 主要由字符长度差异驱动，而非单词信息量。

**🔧 技术方法**

使用子词学习（BPE、Unigram）和预训练 tokenizer（Llama‑4、Gemma‑3、GPT‑2、Sarvam‑1、SUTRA、BrahmicTokenizer‑131K 等），并计算 tokens‑per‑proposition、fertility、compression、parity 等指标。

**📊 数据集**

使用 FLORES‑200 devtest、Sāmayik（现代英‑梵散文）和 Itihāsa（梵语史诗诗篇）等公开并行语料库；在 Devanagari 和 SLP1 两种编码下评估。

**📈 对比分析**

与部署的 tokenizer 和训练的 tokenizer 进行对比，并与匹配英语 tokenizer 对照；结果显示：部署 tokenizer 在梵语上每个命题的 token 数约为英语的 1.1–1.3 倍；在匹配对照下，BPE 32k/64k 词汇量时在域内散文上接近或略低于英语；随着词汇量增大，tokens‑per‑proposition 逐渐趋于 1；整体差距主要由字符长度差异造成。

**⚠️ 局限性**

局限性：仅使用英语作为对照语言；仅使用少量规模相似的语料库；未使用形态学分隔或词形标注；未训练大规模单语料 tokenizer；子词模型对语料的适配性不足，可能影响结果。

---

## 432. Middleware for Feed Recommendation in Practice: How Feed Creators Build, Maintain, and Sustain Custom Feeds on Bluesky

**arXiv ID:** 2609.12958 | [PDF](https://arxiv.org/pdf/2609.12958v1)

**作者:** Tony Zhou `[一作]` (University of Washington), Amy X. Zhang `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对Bluesky平台的自定义 Feed 生态进行大规模实证研究，结合访谈（24 名 Feed 创作者 + 2 名工具开发者）与对 88,302 条 Feed 的量化分析，归纳创作者的两种角色取向、Feed 构建与维护方式及其面临的挑战。

**💡 创新点**

首次系统揭示了 Middleware 生态下 Feed 创作者的角色差异与功力不均衡（如数据访问、技术专长差距），并提出针对平台、工具与社区的设计与政策改进建议，填补了对自定义 Feed 生态缺乏深入了解的研究空白。

**🔧 技术方法**

采用混合方法：半结构化访谈、主题编码；利用 ATProto API 对 PDS 服务器进行爬取、构建 Feed 元数据；使用交互量化指标（如每条 Feed 平均互动数、活跃消费者参与率）和统计检验（Mann‑Whitney U）进行定量对比。

**📊 数据集**

构建了包含 88,302 条 Feed 及其 41,955 名创作者的公开元数据集，采集了 Feed 点赞、最热视频、互动情况等信息；访谈数据来源于 24 名创作者与 2 名第三方工具开发者的录音文本。

**📈 对比分析**

通过对比“活跃创作者”与“非活跃创作者”在互动量、活跃消费者参与率等指标上的差异，使用 Mann‑Whitney U 检验验证显著性（p<0.001）。在量化层面未给出算法性能指标，主要侧重于生态行为与参与度的比较。

**⚠️ 局限性**

研究仅聚焦创作者视角，缺乏消费者与贡献者的反馈；未对 Feed 逻辑进行大规模自动分类；数据采集受限于已知的 PDS 实例，可能导致样本不完整；无法对 Feed 算法质量与用户体验进行客观性能评估。

---

## 433. Dissecting GPU Utilization for LLM Inference on Nvidia Hopper

**arXiv ID:** 2609.12923 | [PDF](https://arxiv.org/pdf/2609.12923v1)

**作者:** Mohammad Siavashi `[一作]` (KTH Royal Institute of Technology), Marco Chiesa `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析H100 Hopper上LLM推理时GPU利用率的误导性，提出八种校准视图以精准诊断decode阶段的瓶颈。

**💡 创新点**

首次将SM利用率拆分为多维度计数器校准视图，揭示decode耗时由指令集fragment floor、占用率、波浪量化等多重机制共同驱动。

**🔧 技术方法**

使用Nsight Compute、vLLM + FlashAttention‑3、CUDA Graphs、GPU计数器与手工枚举算法空间进行性能剖析。

**📊 数据集**

基准使用Meta‑Llama‑3‑8B、Qwen3‑14B、Qwen3‑32B及稀疏MoE Qwen3‑30B‑A3B等公开模型，并在H100 NVL上测量。

**📈 对比分析**

对比不同批次、预填、解码情形下的SM占用、Compute SOL、L2占用等指标，发现解码阶段峰值利用率仅约10%~12%，而改进后可显著提升吞吐。

**⚠️ 局限性**

局限在于单机单卡、TP=1、H100 NVL、理想的前缀缓存命中率，且对多卡、不同H100 SKU及中等命中率场景未验证。

---

## 434. Input Resolution Matters: Real-Time Object Detection Latency

**arXiv ID:** 2609.12920 | [PDF](https://arxiv.org/pdf/2609.12920v1)

**作者:** Qingyang Zhang `[一作]` (University of Tsukuba), Laura Carnevali `[通讯]` (University of Florence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了输入分辨率依赖的实时目标检测延迟分布模型，将总延迟拆分为预处理、推理、后处理三阶段，并通过参数化的概率分布对每一阶段建模；

**💡 创新点**

创新点在于（1）用分布函数而非单一指标描述延迟，形成理论上可解析的卷积模型；（2）将分辨率映射到分布参数，实现分辨率感知的延迟估计；（3）对多种分布族进行比较，提供实证验证；

**🔧 技术方法**

使用概率分布拟合（指数、埃尔朗、正态、伽玛）、最大似然估计、线性回归构造参数函数、Kolmogorov‑Smirnov、Anderson‑Darling、Cramér‑von Mises等统计检验；

**📊 数据集**

COCO2017验证集（560张图像），在NVIDIA Jetson Orin NX上测试YOLOv11n，在多种分辨率下采集延迟数据；

**📈 对比分析**

将模型与三种固定参数基线（分别在低分辨率、高分辨率或两端混合）比较；结果显示，在正态和伽玛族中，分辨率感知模型在KS、CvM、AD指标上均优于基线，尤其在中高分辨率下提升明显；

**⚠️ 局限性**

局限包括：（1）假设阶段独立性导致忽略协方差，误差在10–35%之间；（2）仅在两端锚点拟合参数，线性关系可能不适用于所有分辨率；（3）对指数和埃尔朗等单参数族适用性有限；（4）实验受限于单一硬件平台和单一模型。

---

## 435. PhaseGAN: High-Fidelity Vocoder via Decoupled Amplitude and GAN-Driven Phase Reconstruction

**arXiv ID:** 2609.12918 | [PDF](https://arxiv.org/pdf/2609.12918v1)

**作者:** Wenzheng Zhang `[一作]` (Inner Mongolia University), Zixuan Li `[通讯]` (Inner Mongolia University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出PhaseGAN，一种两阶段轻量化声码器，先用ICCRN从Mel谱估计振幅，再用R3GAN无监督生成相位，从而实现高保真语音合成；

**💡 创新点**

创新点在于振幅与相位的任务分离与专用网络，首次将GAN（R3GAN）用于无标签相位重建，且实现参数仅约500k、计算量约1GMAC的轻量版；

**🔧 技术方法**

技术细节包括Mel滤波器伪逆、ICCRN架构、Softplus激活、相位公式Φ、双线性并行估计、R3GAN相对对抗损失、零中心梯度惩罚、多分辨率STFT损失、AdamW优化等；

**📊 数据集**

使用的公开数据集有LJSpeech（单说话人）、VCTK（多说话人）、Opencpop（中文流行歌声），以及FastSpeech2生成的Mel谱做端到端评估；

**📈 对比分析**

与HiFiGAN、iSTFTNet、APNet、BigVGAN、APNet2、Vocos、FreeV等基线在客观指标（WB-PESQ、STOI、MCD、F0 RMSE、UTMOS）和主观MOS上进行比较，PhaseGAN在单说话人获得UTMOS 4.238、MOS 4.256，参数仅1.6M；轻量版0.54M；在VCTK未见说话人和Opencpop歌声跨域测试同样表现优于基线；

**⚠️ 局限性**

局限性包括尚未在更广泛的语言、噪声或极端语音条件下验证鲁棒性，GAN训练仍可能出现不稳定现象，跨域效果虽好但在极端数据上可能需要微调。

---

## 436. LLM-Enhanced Dual-Branch Learning for Large-Scale Multi-Label Text Classification

**arXiv ID:** 2609.12915 | [PDF](https://arxiv.org/pdf/2609.12915v1)

**作者:** Hui Ye `[一作]` (Georgia State University), Rajshekhar Sunderraman `[通讯]` (Georgia State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DualMLC 双分支框架，使用自回归解码器 Qwen2.5-7B 与双向编码器 BERT-base 并行产生标签得分，并通过后期 logits 融合提升大规模多标签文本分类性能。

**💡 创新点**

创新点在于保留异构语言模型的独立表示并在共享标签空间进行后期融合，证明不同上下文化机制提供互补语义证据；通过独立监督保持各自优势，融合后显著提升排名。

**🔧 技术方法**

使用技术包括双分支 Transformer、LoRA 低秩微调、自回归解码器 Qwen2.5-7B、双向编码器 BERT-base、层级平均、不同的 pooling（均值 vs CLS）、Late logit fusion 以及 BCE 损失。

**📊 数据集**

实验数据集为 EURLex-4K（欧盟法律文本），Wiki10-31K（维基百科+标签），以及 AmazonCat-13K（亚马逊商品分类）。

**📈 对比分析**

与 13 种基线方法（包含注释树、稀疏、嵌入和 Transformer 编码器方法）在 P@1、P@3、P@5 上进行对比，DualMLC 在所有指标和数据集均取得最高或第二高分，提升幅度约为 P@1 +0.7–1.5%，P@3 +0.8–1.3%，P@5 +0.5–1.0%。

**⚠️ 局限性**

局限性包括：仅在三大公开基准验证，未评估更大标签空间下的效率；融合权重 α 需要手工设定；模型依赖于预训练的大模型，资源占用高，难以在资源受限环境下直接迁移。

---

## 437. Before the Tipping Point: Force-Guided Active Perception for Shape-Agnostic Estimation of 3D Centers of Mass

**arXiv ID:** 2609.12894 | [PDF](https://arxiv.org/pdf/2609.12894v1)

**作者:** Steven M. Hyland `[一作]` (Worcester Polytechnic Institute), Cagdas D. Onal `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

使用单一的准静态推-拉循环，通过测量力/角度曲线推断未知物体的三维重心高度、质量及临界翻倒角度，避免实际翻倒。

**💡 创新点**

创新点在于：① 引入子临界翻倒策略，在安全范围内完成参数估计；② 采用推-拉双向循环消除摩擦偏差；③ 只需一次交互即可获得三维惯性参数，且不依赖物体可抓取性或多次翻倒。

**🔧 技术方法**

技术包括：六轴力/力矩传感器与视觉（AprilTag）同步获取姿态；准静态高位推送；中值滤波与Savitzky–Golay平滑；一次线性回归做热启动，随后非线性最小二乘拟合；安全阈值阈限 η_safety 控制推力上限。

**📊 数据集**

实验数据集：4个未知物体（盒子、心形棱柱、手电筒、显示器）与其真值（质量、重心高度、临界角度），共约20次推-拉实验；不使用公开大规模数据集。

**📈 对比分析**

与真实测量（称重、手工测量重心）对比，质量误差<2.6%、重心高度误差<0.3%、临界角误差<0.2%，整体相对误差均<5%。通过在不同 η_safety 取值实验展示安全性与可观测性之间的权衡。

**⚠️ 局限性**

局限性：① 对圆角或不规则底部物体易失效；② 需要预先获得平面重心投影；③ 仅适用于刚体、静态摩擦、低速度交互；④ 受推力上限控制导致观测信息受限，导致高度和角度估计误差增大。

---

## 438. Learning Sign Language Recognition under Label Noise: A Study of Noise-Robust Losses for Isolated and Continuous Settings

**arXiv ID:** 2609.12885 | [PDF](https://arxiv.org/pdf/2609.12885v1)

**作者:** Akihisa Shitara `[一作]` (University of Tsukuba), Yoichi Ochiai `[通讯]` (University of Tsukuba)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在手语识别中引入鲁棒损失（SCE、GCE）来处理标签噪声并评估其对ISLR和CSLR的影响

**💡 创新点**

首次系统性地将对称交叉熵和广义交叉熵应用于手语识别，比较不同噪声率、骨架/视觉基准下的性能，并剖析辅助损失权重对连续识别的影响

**🔧 技术方法**

使用对称交叉熵（SCE）、广义交叉熵（GCE）、标准交叉熵（CE）以及CPC、CTC等序列损失；在ISLR中替换分类头的损失，在CSLR中替换伪标签辅助分类损失，并引入梯度匹配控制

**📊 数据集**

ASL Citizen（2,731类）用于ISLR噪声实验，PHOENIX-2014用于CSLR实验，采用三种骨架/视觉基准（ST-GCN、SPOTER、Video Swin‑T、VAC、CorrNet、SlowFastSign）

**📈 对比分析**

与CE比较时，GCE在噪声率0.2下平均提升2.9–10.0个百分点；SCE亦优于CE，但提升幅度较小；在CSLR中，GCE/SCE在不考虑权重时比CE低1.7–3.2个百分点，但通过梯度匹配控制可将差距压至0.4–0.9个百分点，表明主要由权重决定；ISLR中鲁棒损失的运行方差比CE高2–11倍，CSLR中则更稳定

**⚠️ 局限性**

局限性包括：仅对ST‑GCN和VAC单一配置做多种子评估；未对不对称噪声、多层次识别和真实时间边界进行验证；GCE需要每个基准重新调参；CSLR实验使用伪标签均匀分配，未探讨更真实的边界噪声；以及对I3D等多标签交叉熵设置不适用鲁棒损失

---

## 439. Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting

**arXiv ID:** 2609.12890 | [PDF](https://arxiv.org/pdf/2609.12890v1)

**作者:** Junhao Zhao `[一作]` (University of Maryland), Nan Xu `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在自回归长预测中提出 Internal‑DW，保留完整前向回合和所有时延损失，仅在反向传播中对残差块的身份和非线性路由加权，以降低远程梯度放大带来的不可靠学习信号。

**💡 创新点**

创新点在于用 Wiener 最优解和显式噪声模型对残差块的两条反向路径进行可靠性加权，从而在不改变前向动态的前提下抑制不可预测创新，解决远程梯度高但信噪比低的问题。

**🔧 技术方法**

技术包括 BPTT、Wiener 滤波最优路由、两种噪声采样器（DW‑Generic、DW‑Prior）、残差块路由、梯度风险最小化、梯度裁剪、Jacobians 正则化、TBPTT 与静态衰减对照实验。

**📊 数据集**

实验数据集包括受控线性高斯 AR、Mackey‑Glass、NARMA、iEEG、电影 fMRI、ETTm1/ETTm2、Well shear flow 与 WeatherBench‑2，共八个预测任务。

**📈 对比分析**

与全 BPTT、梯度裁剪、Jacobian 正则化、TBPTT 与静态衰减等方法对比，在四个历史主导、弱驱动数据集上，Internal‑DW 将相对 L2 误差下降 5.2%–13.8%；在其他数据集表现相当或略差；还能延长或保持最佳训练时长。

**⚠️ 局限性**

局限性包括对噪声采样器假设的依赖；当可用历史信号弱或采样器与驱动信息不匹配时收益降低甚至逆转；仅在残差块内部做路由，未考虑跨层交互；不保证每一步训练都优；并非在所有预测任务上通用。

---

## 440. Information-Induced Training Geometry: Exact Reduction, Canonical Completion, and Structured Expressivity

**arXiv ID:** 2609.12991 | [PDF](https://arxiv.org/pdf/2609.12991v1)

**作者:** Zavier Li `[一作]` `[通讯]` (Xidian University), Zavier Li (Xidian University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

构建了一个关于训练数据如何约束优化几何的理论框架，并推导了 SPD 场的闭式最优完成、可见信息映射的子度量性质以及移动信息通道下的度量分层。

**💡 创新点**

创新点在于将子度量(submetry)、Split‑Hadamard 结构与 AIRM（Affine‑Invariant Riemannian Metric）相结合，得到唯一最优完成、闭式的拉回度量、层级分层，以及对可实现性（diagonal/block）和不可实现性给出可计算的凸/半正定证书。

**🔧 技术方法**

使用的技术包括：Riemannian 几何与子度量理论、SPD 集合的 AIRM、Hadamard 子流形理论、线性压缩的矩阵分析、对偶与相对内部测试、主成分与矩阵浓缩/谱分析等。

**📊 数据集**

该工作为理论论文，无实验数据集，所有结果均为闭式推导与证明。

**📈 对比分析**

与传统自然梯度、K‑FAC、AdaGrad、Shampoo 等预条件器做理论对比：提供了可实现性检查、误差上界和最优性证明；在可见几何满足的条件下可获得最小化的 AIRM 变形，说明理论上可取得更好的几何逼近。

**⚠️ 局限性**

局限性包括：仅适用于正定 SPD、满列秩线性压缩；对非线性、无限维、多视角或负定情形需要单独验证；缺乏对优化动态（如学习率、更新轨迹）的具体演化方程；对可见信息的假设过于理想化，实际数据中可能存在噪声、欠测等问题。

---

## 441. Fast and Faithful: Principled Conditional Flow Matching for Inverse Problems

**arXiv ID:** 2609.12953 | [PDF](https://arxiv.org/pdf/2609.12953v1)

**作者:** Shirin Shoushtari `[一作]` (Washington University), Ulugbek S. Kamilov `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种测量条件的流匹配方法，将前向模型直接嵌入速度场，实现高效的逆问题求解。

**💡 创新点**

在速度场中显式包含测量一致性，理论证明其等价于后验均值，并通过半量子分裂与学习先验的交替估计实现。

**🔧 技术方法**

采用流匹配与条件流匹配，半量子分裂（HQS）估计后验均值，使用稳定时间尺度和显式欧拉积分进行采样。

**📊 数据集**

在CelebA和AFHQ‑Cat两个图像数据集上评估，包含去噪、模糊、超分、随机与盒状缺失等五种逆问题。

**📈 对比分析**

与Pnp、展开、扩散、流模型等最新基线比较，在PSNR/SSIM/LPIPS上往往排名第一或第二，同时仅需50倍更少的函数评估，实现显著加速。

**⚠️ 局限性**

仅适用于线性前向模型与高斯噪声，需手动选择步数与迭代深度，对非线性测量或更一般噪声模型的推广尚待研究。

---

## 442. StepAudio 3 Gen Technical Report

**arXiv ID:** 2609.12945 | [PDF](https://arxiv.org/pdf/2609.12945v1)

**作者:** Bin Lin `[一作]`, Zichao Zhou `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 StepAudio 3 Gen，一种统一的离散自回归音频生成模型，可实现 TTS、声音设计、音乐、声乐、声效、氛围语音及多音频混合的生成。

**💡 创新点**

创新点包括：① 通过干扰感知的分阶段预训练，逐步引入音频模态而不损失 LLM 文本能力；② 采用 RVQ Adaptor 将多码本音频表示无缝映射到预训练 LLM 的词向量空间；③ 在共享 12.5 Hz 16 码本残差向量量化空间内完成所有音频域的生成，避免连续渲染器。

**🔧 技术方法**

核心技术：残差向量量化（RVQ）与量化丢弃；时间–深度 Transformer 架构；零初始化的 RVQ Adaptor；四阶段干扰感知预训练；轻量级因果 Transformer 预测剩余码本；自回归式离散生成与 Vocos 风格 causal codec 解码。

**📊 数据集**

数据集：700k 小时混合语音、音乐、环境声（预训练）；后训练时使用约5,000 小时的 TTS、语音设计、声乐、音乐、声效等多域数据；对语音理解任务使用 ASR、音频翻译、MMAU、SpeechMMLU；对文本任务使用 FinEval、C‑Eval、MMLU、CMMLU、MATH、GSM8K、BBH、HumanEval。

**📈 对比分析**

对比方法：在音频理解、翻译、问答等基准上与无 Adaptor 系统比较，提升 CER、WER、准确率、BLEU；在文本基准上与三阶段基线比较，提升 FinEval、C‑Eval、MMLU、CMMLU、MATH、GSM8K、BBH、HumanEval 结果；在 TTS 方面与五大商业 TTS 系统一对一比较，Elo 评级最高，胜率 82%；在声音设计任务中，Elo 最高、平均胜率 75.5%。

**⚠️ 局限性**

限制：模型仍依赖大规模预训练语料，训练成本高；离散编码在超长持续音频上可能产生累积误差；对低资源语言与极端音频域（如极低频或极高频）表现尚不充分；缺乏对实时低延迟生成的深入评估。

---

## 443. Support-Aware Telemetry Compression for 5G Positioning via Conditional Conflict Graphs

**arXiv ID:** 2609.12933 | [PDF](https://arxiv.org/pdf/2609.12933v1)

**作者:** Mohammad Reza Deylam Salehi `[一作]` (Telecom SudParis, Institut Polytechnique De Paris), Hakima Chaouchi `[通讯]` (Telecom SudParis, Institut Polytechnique De Paris)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了面向5G定位的支持感知函数编码方案，允许基站仅发送足以满足定位服务决策的压缩报告；

**💡 创新点**

创新点在于构建条件冲突图（Conditional Conflict Graph），给出全局零误差更新的必要充分条件，并提出支持感知的交替编码算法；

**🔧 技术方法**

采用图论（图着色）、条件冲突图、DSATUR颜色算法、零误差函数计算、以及NRPPa/LMF集成架构；

**📊 数据集**

使用了三基站仿真场景（220×220格、120个时间码）和Fraunhofer IIS六基站下行TDoA测量数据（18,863训练束、15,722测试束）；

**📈 对比分析**

通过完整解码器冲突检查进行验证，在三基站实验中将理想报文率降低21.3–22.1%，在六基站测量中回归率为83.77%，压缩率提高22.8–35.6%；

**⚠️ 局限性**

局限性包括：仅处理已量化的定量测量；DSATUR为启发式；仅在测量层面验证，未完成端到端NRPPa压缩实现；需在更大支持空间和真实UL‑RTOA数据上进一步验证。

---

## 444. Intelligent Semantic Matching (ISM) for Video Tutorial Search using Transformer Models

**arXiv ID:** 2609.12921 | [PDF](https://arxiv.org/pdf/2609.12921v1)

**作者:** Ahmad J. Tayeb `[一作]` (King Abdulaziz University), Sonia Haiduc `[通讯]` (Florida State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 Intelligent Semantic Matching (ISM) 方法，利用 SBERT 对视频教程转录文本生成语义向量，并通过重排名技术提取最相关片段，同时使用 GPT‑4 自动生成视频摘要，显著提升编程视频的检索与概览效果。

**💡 创新点**

创新点包括：①基于 Transformer 的 SBERT 语义向量化和分块重排名，突破 BM25 语义瓶颈；②引入加权得分重排（Weighted Score Adjustment）策略实现最优排序；③结合 GPT‑4 生成高质量摘要，为用户快速把握视频内容；④提供完整复现包，促进可复现性。

**🔧 技术方法**

使用技术包括：SBERT（multi‑qa‑distilbert‑cos‑v1 与 all‑mpnet‑base‑v2）、Whisper 语音转写、Qdrant 向量检索、余弦相似度计算、加权重排算法、GPT‑4 生成摘要以及用户界面展示模块。

**📊 数据集**

采用 TechTube 原始数据集（86 条编程视频，98 条 Python 任务查询）进行评估，并在此基础上进行复现与扩展。

**📈 对比分析**

与 TechTube 进行定量对比：Hit@5 从 0.58 提升至 0.95，平均 F1 从 0.52 提升至 0.70；用户研究表明参与者更倾向于 ISM 的语义匹配与 GPT‑4 摘要，整体满意度显著提升。

**⚠️ 局限性**

局限性包括：仅能检测连续片段，难以覆盖多段非连续相关内容；数据集规模有限，主要聚焦 Python，可能不适用于其他语言或更复杂查询；对多视频检索与更广泛任务的泛化性尚需进一步验证。

---

## 445. UniPart: Towards Zero-shot Language-Grounded 3D Part Segmentation for Embodied Interaction

**arXiv ID:** 2609.12898 | [PDF](https://arxiv.org/pdf/2609.12898v1)

**作者:** Xinqiang Yu `[一作]` (Chinese Academy of Sciences), He Wang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

生成了大规模的文本-部件对齐3D分割数据集LangPart-1M（160K+对象，8M文本-部件对）并提炼出高质量LangPart-4K/1K子集，同时提出UniPart模型，实现零样本语言驱动的3D部件分割并在真实机器人抓取任务中直接提供可用的部件掩码。

**💡 创新点**

创新点主要包括：①基于多视角渲染、SAM分割、Set-of-Mark提示以及GPT-5.2-Pro文本引导合并的全自动部件标注管线，显著提升3D一致性与规模；②轻量级跨模态Transformer，在每层加入CLIP文本CLS向量的加性注入，兼顾速度与性能；③通过对比实验剥离数据与模型贡献，系统验证两者互补效应。

**🔧 技术方法**

采用的技术包括：SAM 2D分割+GPT-5.2-Pro文本合并；Blender多视角渲染；CLIP文本嵌入；Farthest Point Sampling+kNN局部点云编码；基于PointNet的轻量点云编码器；Transformer编码器与逐层加性文本融合；对齐损失与二值交叉熵；与GraspNet/DexGraspNet整合的抓取管线。

**📊 数据集**

使用的数据集：Objaverse原始3D模型库；自建LangPart-1M（160K+对象，8M文本-部件对）；手工校验的LangPart-4K（4K对象）及其拆分的LangPart-3K/1K；公开基准ShapeNetPart、PartNet-E、Objaverse-General。

**📈 对比分析**

在Objaverse-General、ShapeNetPart、PartNet-E以及LangPart-1K上与FIND3D、PointCLIPV2、PartSLIP++、OpenMask3D等开放世界部件分割基线进行mIoU对比。UniPart在LangPart-1M上达约47% mIoU，明显优于FIND3D的38%及其他基线；在真实机器人抓取实验中，UniPart实现90.5%部件分割准确率和85%抓取成功率，证明其实用价值。

**⚠️ 局限性**

局限性包括：自动标注管线仍可能出现视角不一致或细粒度错误；数据主要来源于静态3D模型，缺乏真实动态场景与噪声多样性；对极端遮挡或稀疏点云的鲁棒性待进一步验证；模型聚焦单一对象的部件分割，尚未扩展至完整场景交互与闭环控制。

---

## 446. The possibility of solving a 3x3 Rubik's Cube under 2 seconds - Optimizing block building

**arXiv ID:** 2609.12946 | [PDF](https://arxiv.org/pdf/2609.12946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 447. Behavior Quotient Learning for Low-Rank Adaptation of LLM Agents

**arXiv ID:** 2609.12896 | [PDF](https://arxiv.org/pdf/2609.12896v1)

**作者:** Pengyang Zhou `[一作]` (Alibaba Cloud), Xiantao Zhang `[通讯]` (Alibaba Cloud)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种单一固定秩LoRA适配框架，利用行为商数（Behavior Quotient）流形对轨迹更新进行平衡，并在压缩阶段通过决策保持（Decision Preserving Compression）保证更新在保持决策变化的同时满足秩约束；

**💡 创新点**

创新点在于（1）构造行为商数流形，将轨迹更新的行为相等性映射到同一空间，实现对行为相似更新的去冗余；（2）通过本地密度加权（Gaussian affinity）自动调整轨迹权重，避免重复行为主导；（3）将平衡后的方向投影到LoRA切空间，并在压缩时同时考虑权重误差与决策一阶扭曲，形成一种既符合秩预算又保持决策的压缩策略；

**🔧 技术方法**

使用的技术包括LoRA低秩适配、行为商数流形构造、切空间投影、决策保持压缩（联合权重误差与决策失真最小化）、Gaussian affinity权重计算、核心子空间优化等；

**📊 数据集**

实验数据集为AppWorld和BrowseComp-Plus，分别用于多步任务与深度检索+证据综合任务；

**📈 对比分析**

方法与基线（Base、Full FT、LoRA、LoRA‑S、MoRAgent、DART、TopoCurate）在四种评价指标（AppWorld Test‑N/C、BrowseComp-Plus Accuracy/Recall）以及交互次数上进行比较；实验显示本文方法在大多数指标上超过最强基线MoRAgent约2%并在交互次数上更低，证明了单一LoRA能实现高效且强大的Agent适配；

**⚠️ 局限性**

局限性包括对超参数α、γ的敏感性，需要手动调节；方法主要验证于AppWorld与BrowseComp-Plus，尚未在更广泛的Agent任务或大规模环境中评估；对轨迹收集仍有依赖，且在某些评测设置下平衡模块贡献有限。

---

## 448. From Transportation to Manipulation: Enabling Grasping in Magnetic Robotics

**arXiv ID:** 2609.12883 | [PDF](https://arxiv.org/pdf/2609.12883v1)

**作者:** Lara Bergmann `[一作]` (Bielefeld University), Klaus Neumann `[通讯]` (Bielefeld University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一种低成本、6自由度并联机械臂（Gripper MagBot），配备1自由度抓手，能通过磁悬浮系统（MagLev）实现抓取、搬运与高混低量生产中的柔性作业，并支持自动化重新配置（通过对接站自主拾取/放置）。

**💡 创新点**

创新点包括：
1) 将抓手与三台MagLev移动器机械耦合，实现全自由度控制；
2) 设计两种工作模式（默认模式与单轨模式），在保持相同末端位姿的前提下可调节稳定性与占地面积；
3) 开发逆运动学控制器，并通过逆运动学实现多自由度的实时位置控制；
4) 利用低层MagLev控制器产生的力矩信息（wrench）通过PCA识别载荷，实现仅凭磁力反馈的载荷估计；
5) 通过对接站实现机器人自我拾取与放置，实现MagLev系统的可编程可重构化。

**🔧 技术方法**

技术手段：
- 磁悬浮平台（ACOPOS 6D、XPlanar等）与Halbach阵列磁悬浮移动器；
- 3D打印（PETG、PLA-CF、TPU）构造机器人本体与抓手；
- 逆运动学解析式与机械传动（齿轮、斜齿轮、皮带、轴承）实现α/β/γ/末端位置控制；
- TwinCAT 3 PLC 控制低层运动；
- MuJoCo 物理引擎+MagBotSim库进行仿真；
- VICON 运动捕捉系统进行定位精度与重复性评估；
- PCA 对力矩数据降维，用于载荷识别与参数调节。

**📊 数据集**

实验数据：
- 在XPlanar 4×3 网格上进行真实实验；
- 采用不同负载（0–70 g）对抓手进行载荷测试；
- 位置控制误差和重复性测试覆盖单轴（x,y,z,α,β,γ、指尖）与多轴（圆形+正弦、螺旋）轨迹；
- 采集的逆运动学输入与实际位姿差异用于统计误差；
- 对比不同低层控制参数集（0/1）对单轨模式振荡的影响。

**📈 对比分析**

性能对比与结果：
- 位置精度：大多数轴子毫米级/小于1°，单轨模式下y轴、α、β误差增大；
- 运动速度：最大速度2 m/s、加速度10 m/s²；
- 产能：仿真pick‑place完成19.661 s（≈183件/小时），实物完成≈1.142 min（≈52件/小时）；
- 载荷识别：PCA能将不同载荷投影到一维空间，单轨模式下区分度更高；
- 重新配置成功率：拾取/放置各100%（共15次）；
- 与之前的6D‑Platform MagBot相比，Gripper MagBot新增抓取功能并保持兼容性。

**⚠️ 局限性**

局限性：
- XPlanar 系统对γ旋转的支持有限（仅在特定位置可360°旋转），导致单轨模式下的姿态控制受限；
- 低层控制器在不同γ角度和温度下的最大力矩不一致，导致α/β轴误差增加；
- 单轨模式下y轴、α、β轴的机械游隙和控制器不足导致重复性下降；
- 实物抓手开启需提升飞行高度，影响负载分布；
- 当前系统无法完成180°旋转，限制某些任务路径；
- 载荷识别受温度影响，需进一步研究热效应或散热方案。

---

## 449. Tuning ROS 2 for Energy-Efficient Navigation: Empirical Insights from Costmap 2D Configurations

**arXiv ID:** 2609.12971 | [PDF](https://arxiv.org/pdf/2609.12971v1)

**作者:** Michel Albonico `[一作]` (Federal University of Technology Paraná), Ivano Malavolta `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在两种仓库场景中对Nav2的Costmap 2D参数进行组合实验，评估其对移动机器人能耗与导航性能的影响。

**💡 创新点**

发现配置对能耗影响显著，尤其是分辨率、更新频率和膨胀设置，比插件组合更关键，并给出了低能耗配置方案。

**🔧 技术方法**

采用ROS 2 Nav2导航栈、Gazebo仿真、RAPL+PowerJoular功耗采集、Pairwise组合生成配置以及Robot Runner实验框架。

**📊 数据集**

使用公开的10倍大仓库地图与自定义3×4 m地图，且在两种障碍布置下共20配置×20重复实验。

**📈 对比分析**

通过功耗、能量、CPU/内存、导航时间与成功率等指标比较，低能耗配置在能耗下降30%~40%，路径长度缩短约1 m，成功率保持≥100%。

**⚠️ 局限性**

局限在仅针对Nav2 Costmap 2D、单一机器人模型与固定硬件，未考虑更大规模或多机器人场景，且实验结果可能受仿真与容器隔离影响。

---

## 450. DementiaCare-Bench: A Modality-Validated Video Benchmark

**arXiv ID:** 2609.12929 | [PDF](https://arxiv.org/pdf/2609.12929v1)

**作者:** Afrouz Sheikholeslami `[一作]` (Macquarie University), Ming-Hsuan Yang `[通讯]` (University of California Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建并发布 DementiaCare-Bench 视频问答基准（56 条专业护理训练视频 → 94 条剪辑 → 2023 题），并在此基准上对 12 种现有视频语言模型进行评估；随后利用 LoRA 微调在 Qwen3‑VL‑32B 上训练 DemCare‑VLM，显著提升模型对视频信息的依赖。

**💡 创新点**

① 通过四种视觉条件（无帧、单帧、打乱帧、按序帧）对每道题进行模态验证，确保题目真正需要视频；② 发现现有 VLM 多依赖语言先验，视频信息对准确率几乎无帮助；③ 引入 reject‑on‑mismatch 负样本与 LoRA 微调，使模型在视频缺失时自我拒绝、在视频可用时恢复视频依赖；④ 为临床护理提供可验证的、基于事件顺序与因果的评测框架。

**🔧 技术方法**

多 Agent 生成管道（Scene description、Transcript parsing、Extraction、Routing/Generation、Modality validation），使用 GPT‑4o 进行模态验证；LoRA 微调 Qwen3‑VL‑32B；视频帧抽样与 1 fps 统一采样；在多维度（感知、时序/因果、临床推理、比较分析）上进行问答；对 12 个 VLM 进行帧数消融评估。

**📊 数据集**

DementiaCare‑Bench：来源于 56 条公开护理训练视频（共 94 条剪辑，覆盖 9 类 BPSD），生成 2023 道多选题；原始视频来自 11 个公开课程与教育频道，演员表演的“模拟”情景；训练数据为 1940 条带匹配/不匹配的样本，用于 LoRA 微调。

**📈 对比分析**

对 12 种 VLM（包括 GPT‑4o、Gemini‑2.5‑Flash、Claude‑Sonnet‑4‑6、Qwen3‑VL‑32B 等）进行帧数消融（0、8、16、32、48、64、128 帧），测量文本先验（LLM‑Answerable）和视频依赖（Δv）。结果显示顶级模型在无视频时得分最高，视频信息无提升；DemCare‑VLM 在 8‑32 帧下最高达 93.1%，视频依赖从 -3.3 变为 +4.5，临床推理与时序题目提升 13–23 分。整体准确率提升约 9–10 分。

**⚠️ 局限性**

① 模态验证依赖单一参考模型 GPT‑4o，导致题目上限受该模型能力限制；② 使用演员演示的剧本化视频，缺少真实护理环境中的光照、遮挡和非结构化行为；③ 数据集规模有限，类别分布不均，稀疏类别结果不稳定；④ 仅在基准内部验证，未评估跨域或现实环境中的泛化与鲁棒性。

---

## 451. Forging Tree-Ring: Reproducing and Instrumenting Black-Box Semantic Watermark Forgery

**arXiv ID:** 2609.12909 | [PDF](https://arxiv.org/pdf/2609.12909v1)

**作者:** Saifur Rahman Tamim `[一作]` (Northern University Bangladesh), A. M. Tayeful Islam `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在有限 GPU 资源（双 T4）下复现并验证 Reprompt 对 Tree‑Ring 语义水印的伪造攻击，并恢复并分析该水印检测器丢弃的非中心 χ² 统计量。

**💡 创新点**

①证明攻击在资源受限环境下仍能成功；②恢复并利用检测器内部统计量，展示不同分数对伪造与干净样本分离的差异；③提供混合精度修复、实验门控流程与不确定性分析，为后续研究提供完整实验框架。

**🔧 技术方法**

使用 Stable Diffusion XL 与 Stable Diffusion 2.1 作为目标与攻击模型；DDIM 反向采样；Tree‑Ring 语义水印；混合精度（fp16/fp32）与 autoencoder upcast 修复；非中心 χ² 统计与 CDF；AUC、Mann‑Whitney、bootstrap CI 等统计方法。

**📊 数据集**

在 512×512 的生成图像上进行实验，使用单一固定 Tree‑Ring key (w_seed=999999)，共 18 个样本（6 次实验 × 3 条件：真实、干净、伪造）。

**📈 对比分析**

与原论文对比：真实样本 6/6 检测、干净样本 0/6、伪造样本 5/6；伪造与干净分离 AUC 分别为 0.861（raw score）和 0.972（调整后 score）；平均攻击耗时约 330 秒/次，低于原论文；恢复的统计量与官方实现完全一致，且能显著区分伪造与干净样本。

**⚠️ 局限性**

单一水印 key；样本量有限（18 个）；仅复现 Reprompt，未评估 Imprint、MetaSeal 等方案；半精度行为不确定；仅测试原始生成图像，未考虑压缩/重采样等后处理；未对攻击模型权重进行位级比较。

---

## 452. Hidden in Rounds: Predicting the Time Cost of 802.11 Contention in Federated Learning

**arXiv ID:** 2609.12903 | [PDF](https://arxiv.org/pdf/2609.12903v1)

**作者:** Satwat Bashir `[一作]` (London South Bank University), Tasos Dagiuklas `[通讯]` (London South Bank University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过将 ns‑3 的 802.11 CSMA/CA 场景与 FedAvg 训练器解耦，测量帧投递率与饱和吞吐量，并利用这些测量构建通信时长估计，评估了密度与负载对通信时间与轮次收敛的影响；

**💡 创新点**

创新点在于提出一种基于 Bianchi 分析的简洁通信时长估计器，并首次将此估计器与统一与持久异构参与策略结合，探究参与偏差对分类准确率的影响；

**🔧 技术方法**

使用的技术包括 ns‑3 的无线网络仿真、Bianchi CSMA/CA 解析、PyTorch 实现的 FedAvg 训练器、回归估计与留一交叉验证；

**📊 数据集**

实验数据集为 Fashion‑MNIST 与 CIFAR‑10，采用 IID 与 Dirichlet（α=0.5 与 α=0.1）两种数据划分；

**📈 对比分析**

通过 720 次实验（6 种密度、6 种负载、2 数据集、2 划分、5 种随机种子）验证了所有实验均达成预设准确率目标；通信时间‑至‑目标随密度与负载呈指数增长，而轮次‑至‑目标几乎不受负载影响；Bianchi 估计器在留一与外推验证中的 MAPE 均低于 10%；参与偏差实验未显著影响最差类别准确率；

**⚠️ 局限性**

局限性包括：仅在单一单元单 AP 的单冲突域场景下验证；未考虑多单元、分层或率自适应网络；估计器仅基于帧投递率而非完整模型更新的交付；未对同步 FL 完成时间进行独立验证；

---

## 453. Quantifying the Value of Privileged Information Using a PAC-Bayesian Approach

**arXiv ID:** 2609.12891 | [PDF](https://arxiv.org/pdf/2609.12891v1)

**作者:** Vasily Bokov `[一作]` (Leiden University), Hao Wang `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种与算法无关的PAC‑Bayes框架来评估在训练阶段仅可获得的特权信息（PI）对模型的潜在收益，并给出了可在训练时计算的log‑partition差值指标；

**💡 创新点**

创新点在于用信息理论与PAC‑Bayes相结合，构造两端Gibbs后验（always‑PI与never‑PI）来量化PI的潜在收益，并引入部署映射的KL收缩条件，提供了从潜在收益到部署收益的可实现性判据；

**🔧 技术方法**

主要技术包括PAC‑Bayes上界（Catoni bound）、Gibbs后验与对数分区函数、对齐映射的KL收缩、固定特征指数族后验的重构、期望传播（EP）用于非共轭模型的证据估计；

**📊 数据集**

使用合成的两组数据集：(1) 低样本量（N=8）的二维高斯混合模型（GMM）; (2) 50样本的二维圆形边界分类任务，用于Gaussian Process Classification with Privileged Noise（GPC+）；

**📈 对比分析**

比较方法是将训练时计算的ΔZ（log‑partition差值）与测试时的性能提升（如对数似然或分类准确率差）对齐，实验显示ΔZ与性能提升高度相关（GMM中Pearson r≈0.91，GPC+中同样呈线性下降），表明该指标能够有效预测PI的实际收益；

**⚠️ 局限性**

局限性包括：假设先验和部署映射为数据无关；Catoni上界要求损失有界，导致对数似然实验仅作为诊断；分区函数计算在大模型上困难；潜在收益正好不保证部署收益，需要额外的部署设计与验证；

---

## 454. Parallel Training Using a CNN-DNN Architecture for Accelerated Development of Diagnostic Models

**arXiv ID:** 2609.12902 | [PDF](https://arxiv.org/pdf/2609.12902v1)

**作者:** Janine Weber-Hamacher `[一作]` (University of Cologne), Axel Klawonn `[通讯]` (University of Cologne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在多中心胸部CT数据上，开发并评估了一种将图像分块、局部CNN并行训练后通过小型DNN融合的模型并行训练框架，用于COVID‑19与非COVID肺炎的二分类。

**💡 创新点**

首次将模型并行分块与DNN融合技术应用于DenseNet和3D CNN，实现了训练速度大幅提升同时保持甚至提升分类准确率。

**🔧 技术方法**

采用深度学习框架（TensorFlow‑GPU），实现DenseNet121、ResNet20、3D CNN三种CNN模型；通过数据预处理（重采样、归一化、身体裁剪）和数据增强，使用Adam优化器与早停策略进行训练。

**📊 数据集**

使用来自科隆、法兰克福、海德堡三所医院的300例胸部CT扫描（COVID‑19 150例，非COVID肺炎 150例）作为多中心多供应商数据集。

**📈 对比分析**

将全局CNN模型与CNN‑DNN并行模型在训练/验证/测试准确率与训练时间上进行对比；DenseNet121与3D CNN在并行模型下实现约77%/76%测试准确率，并将训练时间压缩至31倍；ResNet20表现最差。

**⚠️ 局限性**

局限在于仅适用于CNN架构、仅做二分类、样本量有限且未在多中心或多类别场景下验证，需要进一步扩展到Transformer等架构和更广泛疾病的多中心验证。

---

## 455. Investigating Temporal Motion Features for Pose-to-Text Indian Sign Language Translation

**arXiv ID:** 2609.12993 | [PDF](https://arxiv.org/pdf/2609.12993v1)

**作者:** Manav Dhamecha `[一作]` (Sardar Vallabhbhai National Institute of Technology), Pruthwik Mishra `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了预训练T5模型的规模与显式运动特征对印度手语姿态到文本翻译的影响；

**💡 创新点**

创新点在于将帧间姿态差分作为运动特征直接拼接到姿态编码器中，并系统评估不同规模T5模型与运动增强的效果；

**🔧 技术方法**

采用轻量化多层感知机姿态编码器将300维（或600维）姿态特征投射到T5嵌入空间，随后在T5-small/base/large上进行联合微调，使用beam search生成文本；

**📊 数据集**

使用iSign v1.1 Part AD子集中的姿态序列和WSLP 2026共享任务的测试/验证集；

**📈 对比分析**

在10%留出的iSign验证集上比较四种模型，T5-small+Motion在BLEU上最高（0.298，比仅空间特征提升约58%），T5-large在chrF上最佳，模型规模提升并未表现出单调性；

**⚠️ 局限性**

实验受限于不同模型训练周期不一致、仅单一随机种子、仅在T5-small上使用运动特征、整体分数偏低且缺乏人工评估，故结果仅可视为研究基线。

---

## 456. SeqMoE: Toward Full-Load Performance via Predictive and Graph-Compatible MoE Offloading

**arXiv ID:** 2609.12978 | [PDF](https://arxiv.org/pdf/2609.12978v1)

**作者:** Zihan Wang `[一作]` (University of Science and Technology of China), Xuehai Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一套端到端的 MoE offloading 框架 SeqMoE，通过预测专家激活序列、联合预取调度、概率 Belady 缓存和图兼容运行时，实现低内存推理接近全负载性能。

**💡 创新点**

创新点：①将专家激活预测视作序列建模，获得多步长高精度预测；②将预取调度建模为带截止时间的作业排序，使用 matroid 贪婪算法最大化预取收益；③基于预测序列的概率 Belady 策略实现未来感知缓存；④设计计算透明、无同步的图兼容运行时，支持 CUDA Graph 捕获，消除执行瓶颈。

**🔧 技术方法**

技术：序列到序列预测（Mamba2 轻量级 Transformer）、递归预测、马尔可夫/贝叶斯预取调度、概率 Belady 缓存、GPU/CPU 双线程异步预取、CUDA Graph 捕获、动态专家置换、预取与缓存协同管理。

**📊 数据集**

数据集：MATH、GSM8K、CodeForces、OpenOrca、ShareGPT，覆盖数学、编程、通用文本、对话等 50K 采样。

**📈 对比分析**

与 Llama.cpp、MoE‑Infinity、KTransformers、FreeToken、FullLoad 等基线比较，SeqMoE 在 25–45% 专家驻留率下实现 91–97% 的专家命中率，推理吞吐量达到 80–85% 的全负载性能，显著优于 FreeToken（<60%）和其他基线（<30%）。

**⚠️ 局限性**

限制：仍受限于 PCIe 等带宽瓶颈，难以在极低内存预算下完全消除 on‑demand stalls；预测误差随序列长度增长仍会累积，递归预测需要小心调参；实现依赖 CUDA Graph 和特定 GPU/CPU 环境，跨平台移植需要进一步工作。

---

## 457. Robust Underwater Grasping of Sloped Objects with a Waterproof Passive Adaptive Gripper

**arXiv ID:** 2609.12927 | [PDF](https://arxiv.org/pdf/2609.12927v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 458. A Dynamic Vertical Scaling Strategy for Distributed Stream Processing Applications in Edge Computing

**arXiv ID:** 2609.12975 | [PDF](https://arxiv.org/pdf/2609.12975v1)

**作者:** Guilherme Hiago Costa dos Santos `[一作]` (Pontifical Catholic University of Rio Grande do Sul), Tiago Coelho Ferreto `[通讯]` (Pontifical Catholic University of Rio Grande do Sul)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于PPO的纵向自动扩缩策略，利用任务级CPU分配来满足边缘DSP的p95延迟SLO并保持吞吐量。

**💡 创新点**

创新点在于将垂直扩缩建模为POMDP，并使用任务级观测与联合动作空间，实现SLO优先且高效的资源利用。

**🔧 技术方法**

采用Proximal Policy Optimization（PPO）强化学习，配合POMDP、EdgeStreamPy仿真器以及SLO-aware奖励设计。

**📊 数据集**

使用RIoTBench提供的PRED和ETL应用配置，以及TAXI和INTEL两条流量轨迹作为工作负载。

**📈 对比分析**

与基线VRebalance（贝叶斯优化）对比，PPO在四种组合中将SLO违约率从0.03%–8.73%降至0.03%–0.30%，吞吐量保持一致，CPU使用在三种场景下降低。

**⚠️ 局限性**

局限性包括未评估不同训练seed的鲁棒性、仅测试两种应用与两种工作负载、奖励函数依赖经验调参，以及未在真实边缘环境中验证实现与测量。

---

## 459. Distributed Stochastic Optimal Control for Pattern-Oriented Swarms

**arXiv ID:** 2609.12959 | [PDF](https://arxiv.org/pdf/2609.12959v1)

**作者:** Qingrui Zhang `[一作]` (Sun Yat-sen University), Xintong Wang `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于 Gibbs 随机场的分布式随机最优控制框架，用于在不确定动态环境中实现自组织的几何模式控制与安全导航。

**💡 创新点**

创新点在于将多目标协同约束整合进 GRF 能量函数，实现统一的贝叶斯推理；引入基于无迹变换的状态不确定性传播与 CVaR/概率约束的碰撞规避；以及使用 SDF 与 Mean Shift 构建无分配的密度引导模式控制，支持自愈与弹性重构。

**🔧 技术方法**

核心技术包括 GRF 建模、均值场近似、Model Predictive Path Integral (MPPI) 采样、无迹变换/UKF 状态预测、CVaR 与概率约束的碰撞能量、SDF 与密度梯度的模式能量。

**📊 数据集**

评估数据来自仿真场景（静态、混合、动态障碍物）以及真实实验，分别使用 15 台室内四旋翼、4 台户外四旋翼以及 8 架固定翼无人机进行验证。

**📈 对比分析**

与传统的距离阈值 MPPI 和基于机会约束的 MPPI 相比，CVaR‑MPPI 在三类障碍环境中实现了更高的生存率（最高 99.6%）和更稳健的安全距离；在模式控制实验中，无占据区域吸引法在复杂非凸形状下覆盖率提升至 ≥98%，并在丢失机器人时保持高自愈性能。

**⚠️ 局限性**

局限性包括在高动态或高密度环境下能量消耗显著增加；需要通信与预测信息，扩展至更大规模或完全离线时可行性待验证；对极端噪声与模型误差的鲁棒性尚需进一步研究。

---

## 460. EduFair-Bench: Evaluating Pedagogical Fairness of LLM Tutors Across Student Demographics

**arXiv ID:** 2609.12949 | [PDF](https://arxiv.org/pdf/2609.12949v1)

**作者:** Jiaxu Zhao `[一作]` (Swiss Federal Institute of Technology in Lausanne), Tanja Käser `[通讯]` (Swiss Federal Institute of Technology in Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了EduFair-Bench基准，用以系统评估大型语言模型（LLM）在教学对话中的教学公平性，探究其对不同学生人口统计特征的差异化影响。

**💡 创新点**

首次将教学质量与人口统计公平性结合在同一基准中，设计四维人口统计协议（性别、移民背景、第一语言、社会经济地位）以及两种消融条件（Implicit与Opposite）来区分导师驱动和学生模拟器的偏差。

**🔧 技术方法**

使用LLM判别器（经过三位人工标注者验证）评估对话中的五个教学层面指标和四个会话层面指标，并采用配对Wilcoxon检验与bootstrap效应量置信区间进行偏差统计。

**📊 数据集**

构建多领域（数学、物理、化学）题库，题目来源为SocraTeach与SciQ，经过领域特定过滤后得到约1500个可用于多轮教学的题目。

**📈 对比分析**

对五个不同规模与训练范式的LLM导师（LLaMA-3.1-8B、Qwen2.5-7B、TutorRL-7B、DeepSeek-R1-70B、GPT-5-mini）进行评估，发现能力与公平性并非正相关，教学RL调优往往重新分配而非消除偏差，语言与移民相关提示产生最大公平差距。

**⚠️ 局限性**

局限性包括：未进行交叉性分析，仅用单一固定LLM学生模拟真实学习者；人口统计变量采用粗粒度二元类别，无法捕捉群体内差异；所得到的偏差模式需在真实人类受试者中进一步验证。

---

## 461. PA-CDM: Position-Aware Character Detection Matching for Evaluating Handwritten Mathematical Expression Recognition

**arXiv ID:** 2609.12917 | [PDF](https://arxiv.org/pdf/2609.12917v1)

**作者:** Shiliang Luo `[一作]` `[通讯]` (East China Normal University), Shiliang Luo (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了位置感知的字符检测匹配度量（PA‑CDM），并发布了受控扰动基准StructPerturb v2.0，以及跨指标一致性评估协议

**💡 创新点**

PA‑CDM 在字符级匹配的基础上加入结构位置森林编码和位置违例加权，使得误差位置被量化；StructPerturb v2.0提供了可复现的 1,340 对受控扰动样本；三维度一致性协议将指标、人工评估与 LLM 判断联合量化

**🔧 技术方法**

基于 LaTeX 解析器构造位置森林、KaTeX 渲染、Hungarian 匹配、结构编辑距离、加权阈值；使用 Transformer Encoder‑Decoder（PosFormer 等）训练模型；利用 LLM（Kimi‑K3）作为对标判定器

**📊 数据集**

CROHME 2014/2016/2019 官方数据集、UniMER‑Test 之外部集合、以及 StructPerturb v2.0 受控扰动对

**📈 对比分析**

与七种现有指标（Exact‑Match、BLEU、TED、CDM、IMS、PA‑CDM、LLM‑Judge）进行 Spearman 相关性对比，PA‑CDM 在 990 条样本上与人工评估的相关系数最高（ρ=0.9535），仅次于 LLM‑Judge（ρ=0.9613）；在受控扰动上，PA‑CDM 对结构错误的敏感度比 CDM 高，能更精细地区分错误类型

**⚠️ 局限性**

严格度仅在平均水平上保持不变，偶有样本逆转；在 OOD 组上结构匹配覆盖下降导致性能降低；受控扰动样本数量有限（如 nested_move L2/L3）；LLM‑Judge 的成本、非确定性和版本漂移；Equivalence 评估中视觉非等价残留 43%；对 PA‑CDM 的细粒度加权选择仍受参数设置影响

---

## 462. Tracing and Coordinating Cross-Layer Influence for Multimodal Model Merging

**arXiv ID:** 2609.12897 | [PDF](https://arxiv.org/pdf/2609.12897v1)

**作者:** Pengyang Zhou `[一作]` (Alibaba Cloud), Xiantao Zhang `[通讯]` (Alibaba Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为CoM的多模态模型合并框架，能够在单个模型中合并多任务专家的更新；

**💡 创新点**

创新点在于：①通过构建跨层影响图并使用Ollivier–Ricci曲率捕捉每个专家更新对视觉‑语言表示的深度传播效应；②将这些多模态影响与专家预测结合，形成统一的融合目标；③在紧凑的控制子空间内拟合二次响应模型，实现区域权重的协同优化；

**🔧 技术方法**

采用Ricci曲率、影响图、图卷积式的多模态对比响应、二次响应建模、谱信赖区间优化等技术，并使用LoRA适配器进行专家训练；

**📊 数据集**

使用MM‑MergeBench的八个seen任务（ScienceQA、ImageNet、VQAv2、REC‑COCO、OCR‑VQA、VizWiz‑Caption、Flickr30k、IconQA）和四个unseen任务（A‑OKVQA、ImageNet‑R、Screen2Words、TabMWP），在Qwen3.5‑4B和Qwen3‑VL‑8B‑Instruct两种基底模型上进行实验；

**📈 对比分析**

与零shot、individual、单模态融合（TA、TIES、DC‑Merge）、跨层融合（Chain of Merges、RegMean++）以及多模态融合（RobustMerge、OptMerge）等基线进行对比；CoM在两种 backbone 上均取得最高平均表现，seen任务提升约+3.4/ +2.4分，unseen任务提升约+2.3/ +1.9分，并在大多数任务上占优；

**⚠️ 局限性**

局限性包括：①依赖小规模 calibration 数据集和较高的计算开销；②仅在有限数量的专家上验证，未评估更大规模专家集合的可扩展性；③影响估计依赖手工设计的影响图和Ricci曲率，可能对不同架构的通用性有限；④目前仅针对LoRA适配器实现，尚未验证对其他 fine‑tune 方式的适用性。

---

## 463. Beyond Accuracy: Uncertainty-Guided Boundary Refinement for Reliable Biomedical Image Segmentation

**arXiv ID:** 2609.12892 | [PDF](https://arxiv.org/pdf/2609.12892v1)

**作者:** Anima Kujur `[一作]` `[通讯]` (Heidelberg University), Anima Kujur (Heidelberg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两阶段可靠性感知边界细化网络RABR-Net，用于对白血球血涂镜图像进行细胞质与细胞核的精准分割。

**💡 创新点**

创新点在于将多源不确定性估计（熵、TTA方差、margin不确定、概率梯度）与软边界提示融合成可靠性表征，并通过门控残差网络局部校正边界，避免全局重置，提高边界可信度。

**🔧 技术方法**

采用UNet+++EfficientNet-B4的主分割器；多维不确定性与边界特征融合；门控残差细化模块；生物学后处理；并用Dice、Boundary Dice、HD95、ECE、风险覆盖等多指标评估。

**📊 数据集**

使用WbcMSBench（基于Raabin-WBC）数据集，包含256×256 RGB血涂镜图像，三类标签：背景、细胞质、细胞核。

**📈 对比分析**

与基线UNet++及缓存预测比较，RABR-Net在Dice从0.9602提升至0.9614，Boundary Dice从0.3448提升至0.3611，HD95从3.0354降低至2.8274，统计显著；但校准ECE略升高。

**⚠️ 局限性**

局限性包括仅在单一数据集评估；改进未提升概率校准；对基线模型错误的修正受限；生物学后处理规则简单，无法处理重叠细胞或复杂拓扑；需要进一步的外部验证和校准友好训练。

---

## 464. Virtualized 5G Tesbed using OpenAirInterface: Tutorial and Benchmarking Tests

**arXiv ID:** 2609.12972 | [PDF](https://arxiv.org/pdf/2609.12972v1)

**作者:** M. Dória `[一作]`, A. Neto `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过搭建 OpenAirInterface（OAI）5G 栈与 USRP 硬件的虚拟化网络，提供完整的安装部署教程，并在单基站环境下进行频谱占用、覆盖评估与吞吐量/RTT基准测试。

**💡 创新点**

创新点在于：①将 OAI 5G 栈与 USRP 的配置细节公开为可复现的脚本和配置文件；②在同一实验平台上实现多种使用案例（频谱占用、覆盖、单 UE、双 UE、距离敏感性），形成一套完整的实验范例；③系统性记录吞吐量、RTT、丢包等指标，为后续研究提供基准参考。

**🔧 技术方法**

使用的技术包括：OpenAirInterface 5G RAN 与 5G Core、Docker 与 Docker‑Compose 容器化部署、USRP N310 / B210 软件定义无线电、Linux Ubuntu 20.04、iPerf3、Ping、Keysight N9342C 频谱分析仪、Python 脚本绘制 REM、以及 Motorola G50 手机作为 UE。

**📊 数据集**

实验数据集主要来源于自建的物理测试环境：Motorola G50 UE 与 USRP N310（或 B210）硬件，测得的吞吐量、RTT、RSRP 等信号强度和覆盖地图；未使用公开的大规模数据集，而是通过现场测量获得的原始实验数据。

**📈 对比分析**

通过对比不同 PRB 数量（106、162、273）以及不同距离（1–12 m）下的吞吐量和 RTT，本文验证了更宽带可实现更高吞吐但 RTT 及波动性增加；同时，在 UE‑UE 与 gNB‑UE 场景中比较，展示了中继路径导致的吞吐与延迟下降。整体性能表现为：最大吞吐 297.8 Mbps，平均吞吐 135–238 Mbps，最大 RTT 79 ms，平均 RTT 12–14 ms。

**⚠️ 局限性**

局限性包括：仅在单基站、单 UE 或两 UE 的实验室环境中评估，未涉及多基站干扰与大规模网络；USRP 的采样率与功率限制导致高频宽实验靠近上限；实验参数依赖特定硬件（如 Motorola G50）和网络配置，可能不具备完全跨平台可迁移性；缺乏实时误码率、能耗等更细粒度指标。

---

## 465. ARC: Autonomous Robotics Compliance A Three-Layer Governance Architecture for Deployed Autonomous Systems

**arXiv ID:** 2609.12932 | [PDF](https://arxiv.org/pdf/2609.12932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 466. Generative Retrieval for Unsupervised Text-Based Person Search

**arXiv ID:** 2609.12965 | [PDF](https://arxiv.org/pdf/2609.12965v1)

**作者:** Mang Ye `[一作]` (Wuhan University), Min Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无监督文本检索框架GTR+，通过分层生成伪文本并训练检索模型，消除人工文本标注需求。

**💡 创新点**

创新点包括：①三层层级文本生成（问答、对比、风格扩展）产生细粒度、多样化描述；②使用GMM结合相似度与生成概率的自适应置信度权重，动态抑制噪声；③构建LargeFine-Person大规模无标注文本数据集。

**🔧 技术方法**

采用多模态大语言模型（MLLM）、CLIP/BLIP等视觉-文本编码器、对比学习、问答生成、GMM噪声估计与自适应样本加权技术。

**📊 数据集**

使用的数据集包括：LUPerson、LPW、CUHK-PEDES、ICFG-PEDES、RSTPReid，以及作者自行构建的LargeFine-Person。

**📈 对比分析**

与现有无监督、弱监督、半监督和监督方法对比，GTR+在无监督设置下R@1≈55.75–61.35，微调后可达R@1≈64.65，显著优于之前的无监督方法并接近部分监督方法。

**⚠️ 局限性**

局限性：生成文本仍可能出现幻觉导致噪声；依赖强大MLLM，对大规模预训练仍有资源瓶颈；硬负样本选择和GMM参数对性能有一定敏感性。

---

## 467. A Large-Scale AIS Dataset from Finnish Water

**arXiv ID:** 2609.12938 | [PDF](https://arxiv.org/pdf/2609.12938v1)

**作者:** Debayan Bhattacharya `[一作]` (Åbo Akademi University), Sebastien Lafond `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提供了一个覆盖芬兰海域（波罗的海与湖泊）20个月的 AIS 数据集，并对其进行可视化与基本统计分析。

**💡 创新点**

首次公开芬兰湖泊 AIS 数据；将海域与内陆水道统一采集，包含 2.29 亿条记录；并提供季节热图与船舶类型分布，可直接用于后续研究。

**🔧 技术方法**

采用 MQTT WebSocket 接口收集 AIS，使用 Folium 绘制热图、分布式聚合算法以及标准统计图表进行分析。

**📊 数据集**

自 2021‑04 至 2022‑12 从 Fintraffic 收集的 2,293,129,345 条 AIS 位置与元数据，涵盖船舶类型、位置、速度、航向等特征。

**📈 对比分析**

与公开 AIS 数据集（如 US Coast Guard、FleetMon 等）在数据量、地理覆盖和季节多样性上对比，显示本数据集在湖泊与季节性变化表现更丰富，聚合与可视化性能满足近实时需求。

**⚠️ 局限性**

数据仅来源于 Fintraffic，缺少深海与极端气象场景；部分字段默认值高、缺失率大，且无法覆盖全球 AIS 信息。

---

## 468. A Robot Among People:From Social Imitation to the Social Becoming of Human Groups

**arXiv ID:** 2609.12937 | [PDF](https://arxiv.org/pdf/2609.12937v1)

**作者:** Victor Tuan Vu Pham `[一作]` (University of Siegen), Marc Hassenzahl `[通讯]` (University of Siegen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过实验与共创工作坊，研究机器人在群体中的关系定位与角色赋予，提出机器人应被视为共享空间的关系载体而非社交主体。

**💡 创新点**

创新在于将机器人视作关系锚定的媒介，强调所有权与环境语境如何决定其意义，并将模糊性视为促进群体共识的资源。

**🔧 技术方法**

采用了小型非人形机器人在白板上执行最小非语言动作的实验，结合青少年剧场式共创工作坊，进行行为观察与情境分析。

**📊 数据集**

使用了实验室录像、参与者访谈以及工作坊观察笔记等原始数据，没有使用公开数据集。

**📈 对比分析**

通过比较不同所有权情境下的机器人行为，采用定性编码评估群体对机器人合法性与角色的认知差异，结果显示所有权显著影响感知。

**⚠️ 局限性**

局限性包括样本规模有限、实验与工作坊的生态有效性受限、对主观解释的依赖，以及框架在不同文化或机器人形态中的推广难度。

---

## 469. Parameter-Efficient Retrievers for Polish and European Languages

**arXiv ID:** 2609.12913 | [PDF](https://arxiv.org/pdf/2609.12913v1)

**作者:** Sławomir Dadas `[一作]` (National Information Processing Institute), Michał Perełkiewicz `[通讯]` (National Information Processing Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PolDense 与 EuroDense 两类高效密集检索模型，并通过跨语言对齐、关系蒸馏与对比微调三阶段训练流程实现无需原始标签即可获得高质量检索结果。

**💡 创新点**

创新点在于将多语言对齐与关系蒸馏结合，并利用强教师模型与重排序器自生成监督，显著提升参数效率同时保持竞争力。

**🔧 技术方法**

采用跨语言对齐（MSE+余弦损失）、关系蒸馏（包含相似度矩阵与边际相似度损失）以及 InfoNCE 对比微调，并用 BGE‑Multilingual‑Gemma2、Pplx‑Embed 等教师模型和 BGE‑Reranker‑v2.5‑Gemma2‑Lightweight 进行重标注。

**📊 数据集**

训练数据包括约 2000 万英波平行文本、FineTranslations 文档（约 13–50 M 条）以及 13–15 M 句子的检索数据集，所有数据均经过机器翻译扩展至多语言。

**📈 对比分析**

在 PIRB 41 任务和 150 任务的多语言评测中，PolDense‑1B 在 PIRB 上超越多项 9 B 参数模型，PolDense 系列形成 Pareto 前沿；EuroDense‑435M 在 9 种语言上领跑同类小模型，并在 150 任务中取得第一名。

**⚠️ 局限性**

局限性包括仅针对检索任务优化，未评估通用嵌入任务；大模型仅使用 13 M 文档；EuroDense 仅单一尺寸且仅覆盖 9 种欧语；缺乏多语言规模化与跨模型对比。

---

## 470. Shuffling is Not Enough: Breaking Permutation-Based Model Confidentiality in Hybrid FHE Inference

**arXiv ID:** 2609.12911 | [PDF](https://arxiv.org/pdf/2609.12911v1)

**作者:** Jiseung Kim `[一作]` (Jeonbuk National University), Hyung Tae Lee `[通讯]` (Chung-Ang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在混合FHE推理中，作者提出了一种对基于置换的模型机密性保护的分析与攻击，证明只需对每层线性操作发出 d+1 次可选输入查询，即可完全恢复该层的排序谱，从而泄露完整模型信息。

**💡 创新点**

创新点包括：①证明置换+DP 的 “shuffle” 方案在满足正确性噪声约束时不具备机密性；②给出最优查询复杂度 d+1 的完整恢复定理；③阐明输入DP 与模型机密性是两条互斥的安全属性；④提出分布无关的噪声阈值分析并通过实验验证。

**🔧 技术方法**

技术手段主要包括：量化格下的取整与排序操作、选择性查询攻击、噪声分布（离散/连续）与噪声上界分析、基于TFHE的实验实现、以及对指纹识别与细化线索的利用。

**📊 数据集**

实验数据集包括：CIFAR-10 上训练的 ResNet‑20、ImageNet 预训练模型（ResNet‑50/101/152、VGG‑16、DenseNet‑121、MobileNet‑V2、EfficientNet‑B0、ConvNeXt‑Tiny）以及 ViT‑B/16；此外还使用 Imagenette 对 ResNet‑50 进行细调实验。

**📈 对比分析**

对比方法：在混合FHE 推理基线与各类噪声增强防御（逐层加噪、仅对最终输出加噪）下，作者对噪声乘数从 1 到 1000 的范围内进行了攻击误差、指纹识别率、预测一致性和推理准确率的测评。结果显示：噪声放大 100 倍仍保持 100% 的指纹识别精度，而仅在 1000 倍噪声下攻击才失效；同一噪声设置下推理准确率已降至 60% 以内；在 ImageNet 规模下，噪声放大 30–70 倍即可导致准确率崩溃。

**⚠️ 局限性**

局限性：该攻击仅适用于满足正确性噪声约束且允许选择性查询的置换防御；对浮点运算、完全 FHE 或 MPC 等其他方案不适用；实验仅覆盖公开的模型与参数；未考虑零知识验证、功能秘密共享或完全服务器端 FHE 等可能的防御手段。

---

## 471. NeuroClick: Preserving Surgeon Autonomy through Hands-Free Earable Tooth-Click Control in Neurosurgery

**arXiv ID:** 2609.12910 | [PDF](https://arxiv.org/pdf/2609.12910v1)

**作者:** Jonas Hummel `[一作]` (Karlsruhe Institute of Technology), Michael Beigl `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在神经外科手术中开发了一个基于耳戴设备的牙齿点击输入系统（NeuroClick），通过牙齿点击实现对显微镜功能的直接、无手无眼控制。

**💡 创新点**

创新点在于：①将牙齿点击与耳戴传感器相结合，提供一种全新的手术设备交互通道；②实现了高精度（LOSO F1 98.6%）的实时点击识别；③在临床场景下证明其能够提升外科医生的自主性而不显著增加工作负荷。

**🔧 技术方法**

使用技术包括 OpenEarable 2.0（骨传导麦克风 + IMU）、轻量级多层感知器（MLP）进行点击检测、手机端时序组合识别多点击命令，以及通过 Tuya FingerBot 机械化执行显微镜操作。

**📊 数据集**

数据集：12 名非义齿受试者在模拟手术环境中产生的牙齿点击与干扰活动（说话、咀嚼、头动等）数据；另外 20 名神经外科医生在模拟切除任务中使用该系统进行评估。

**📈 对比分析**

与传统的委托方式（Delegation）对比，Earable 在自主控制评分显著更高，焦点转移、工作流集成与安全感无显著差异，工作负荷和可用性相当，但完成时间略长；整体性能在保持手部和视觉专注的前提下实现了可靠控制。

**⚠️ 局限性**

局限性：仅在模拟任务中评估，未验证长期使用效果；缺乏个体化校准与多功能集成；硬件仍为原型，佩戴稳定性和听觉遮蔽问题仍需进一步改进。

---

## 472. Offline Reinforcement Learning for Wind Farm Control: A Wind Tunnel Study under Dynamic Wind Directions

**arXiv ID:** 2609.12905 | [PDF](https://arxiv.org/pdf/2609.12905v1)

**作者:** Yuhan Su `[一作]` (University of Warwick), Xiaowei Zhao `[通讯]` (University of Warwick)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种MTD3-BC离线强化学习算法，用于在动态风向下通过叶片偏航控制最大化风电场功率。

**💡 创新点**

在TD3-BC基础上新增动作一致性正则项和风向条件的演员网络，完全无需唤醒模型且仅利用离线数据。

**🔧 技术方法**

采用离线强化学习、TD3-BC+动作一致性、神经网络策略、状态归一化与延迟更新等技术。

**📊 数据集**

使用在风洞实验中产生的80 GB离线数据，包含贪婪策略和三种不同精度的LUT（FLORIS、数据增强、数据驱动）控制下的风场样本。

**📈 对比分析**

与贪婪、数据校准的LUT和在线PPO比较，MTD3-BC在实验中实现≈10 %功率提升，性能与LUT相当，但训练成本仅为PPO的5%。

**⚠️ 局限性**

受限于离线数据的覆盖范围，需要足够多样的动作样本；对未覆盖风向、故障或配置变化的适应性尚未验证。

---

## 473. Physical-State-Guided Diffusion Sampling for Full-Waveform Inversion

**arXiv ID:** 2609.12899 | [PDF](https://arxiv.org/pdf/2609.12899v1)

**作者:** Chen Min `[一作]` (Shanghai Jiao Tong University), Xiongbin Yan `[通讯]` (Lanzhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的全波形反演框架——Physical-State-Guided Diffusion Sampling (PSG)，将物理速度状态与扩散先验结合进行条件采样。

**💡 创新点**

创新点在于通过高斯桥将持续的物理速度状态与扩散过程耦合，分离波动方程梯度与去噪器梯度，保持传统FWI的初始化与优化历史，同时实现对先验的有效利用。

**🔧 技术方法**

采用预训练的扩散生成模型、Gaussian桥、逆扩散（SDE/ODE）、波形拟合、逆向SDE采样、物理状态优化以及分离梯度计算等技术。

**📊 数据集**

使用OpenFWI四个数据集（FlatVel-B、FlatFault-B、CurveVel-B、CurveFault-B）进行验证，并在Marmousi、Overthrust、BP2004 Salt等外部模型上测试泛化能力。

**📈 对比分析**

与经典FWI、Tikhonov/TV正则化、DPS、RED-DiffEq、DiffusionFWI、SGDS-FWI等方法比较，在清洁、噪声、缺失射线等多种条件下，PSG在RMSE/MAE/SSIM指标上均优于基线，并在缺失数据与噪声情形中保持较高的结构相似度。

**⚠️ 局限性**

局限性包括对随机逆向轨迹的敏感性导致深层细节误差与噪声/缺失相关；对极端几何结构的泛化仍有限；未在真实勘探数据上验证；需要手动调参以适应不同场景。

---

## 474. MedSNIP: Building and Benchmarking Snippet-Level Granularity for Medical Fact Verification

**arXiv ID:** 2609.12884 | [PDF](https://arxiv.org/pdf/2609.12884v1)

**作者:** Hasan Iqbal `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yuxia Wang `[通讯]` (INSAIT, Sofia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了片段级医学事实核查方法，定义了医学回答中可验证的语义片段，并构建了相应的基准数据集。

**💡 创新点**

创新点在于：①引入片段（snippet）作为比原子更粗、比全文更细的验证单元，保留临床上下文；②构造带有“通用正确性”“病人情境正确性”双重标签和六类结构模式的手工标注数据集；③开发了自动化片段生成管道。

**🔧 技术方法**

技术上主要使用：LLM（如Llama、GPT-4）进行片段提取与聚合；检索增强的多轮验证器（基于网络与PubMed检索）；对比实验采用多种开源与闭源检验模型；结构模式分析与多样本自举置信区间评估。

**📊 数据集**

使用数据集包括：自建的“Medical Snippet Benchmark”（276条回答、2524片段，覆盖消费健康与临床小结），以及公开的“MedQA”与“Med-Hallucinations”用于外部验证。

**📈 对比分析**

与传统原子级拆分相比，片段级核查在四个强模型上提升了False‑Class F1（最高+0.068），在长篇答案和因果/条件链条场景中提升最显著，同时可减少24–73%的验证调用，显著降低成本；弱模型或结构稀疏的数据集时提升有限或出现倒退。

**⚠️ 局限性**

局限性包括：仅在英语数据上实验，数据量有限，外部证据来源简单，检索设计未充分；片段与原子边界重叠时提升可能由表述清晰度驱动；验证器强度与片段结构匹配度对结果影响大；需进一步验证跨语言和真实临床环境的适用性。

---

## 475. Type Diversity Enables Transformers to Generalise Compositionally

**arXiv ID:** 2609.13144 | [PDF](https://arxiv.org/pdf/2609.13144v1)

**作者:** Anssi Moisio `[一作]` (Aalto University), Mikko Kurimo `[通讯]` (Aalto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究词汇与结构类型多样性（type diversity）对Transformer在组合推理任务中的泛化能力影响，利用Grammatical Framework对COGS、SLOG和SCAN数据集进行扩展，并在多种结构多样性变体上评估模型性能。

**💡 创新点**

提出并验证“类型多样性”是解释Transformer结构泛化难度的核心因素；同时揭示子树/超树多样性在类型多样性低时也能提升泛化，指出原有Compound Divergence指标与数据多样性相互作用而非单独衡量难度。

**🔧 技术方法**

核心技术包括：Grammatical Framework用于生成可扩展的自然语言数据；Transformer seq2seq模型（1M参数）进行训练与评估；构造结构多样性变体、控制数据规模、对比atom/compound divergence与accuracy；统计相关系数和Spearman相关性。

**📊 数据集**

主要数据集：COGS（语义解析）与其扩展SLOG（结构泛化多样化），以及SCAN（命令语言映射）；所有数据集通过GF生成或改写得到多样化变体。

**📈 对比分析**

比较方法：在保持训练集规模大致相同的前提下，仅改变目标结构的构造器数量（type diversity）或其子树/超树多样性；使用平均准确率与标准差评估模型；结果显示：结构泛化与词汇泛化对type diversity的敏感度相近；随着类型多样性增加，准确率从≈0%提升至≈90%+；子树/超树多样性在类型多样性低时能显著提升性能；atom/compound divergence与准确率呈正相关。

**⚠️ 局限性**

限制：仅使用Transformer模型且固定超参，无法推广到其他seq2seq架构；实验结果受随机种子波动较大；GF生成的数据仍简化自然语言，未覆盖所有真实语言多样性；对模型能力的结论仅限于在当前数据规模与多样性水平下。

---

## 476. From Review to Reuse: How Post-Task Workflow Can Support Human-AI Agent Interaction

**arXiv ID:** 2609.13136 | [PDF](https://arxiv.org/pdf/2609.13136v1)

**作者:** Zekun Wu `[一作]` (Saarland University, Saarland Informatics Campus), Anna Maria Feit `[通讯]` (Saarland University, Saarland Informatics Campus)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并实现了后任务工作流（post‑task workflow）工具 Trace2Flow，用于将 AI 代理完成的多步骤任务执行轨迹转化为可视化、可编辑的图形工作流，从而帮助用户审阅、验证并复用已完成的代理执行。

**💡 创新点**

创新点在于：①首次提出在代理任务完成后生成可编辑的工作流表示，而非仅靠原始日志；②将执行轨迹自动映射为可执行的低代码工作流；③通过实验验证后任务工作流显著提升用户对执行过程的理解、错误检测与任务复用能力。

**🔧 技术方法**

技术手段包括：①使用 LLM 进行工作流 IR 生成与编译；②采用 n8n 作为工作流引擎实现可执行图形；③构建 Trace2Flow 交互原型；④在 Wizard‑of‑Oz 环境下复现代理执行；⑤设计问卷与实验，收集用户行为数据。

**📊 数据集**

主要数据集：1) 10,803 个公开 n8n 工作流模板，用于分析真实工作流结构与节点功能；2) 5 对基于这些模板的任务（T0–T4）与对应的错误案例，用于实验。

**📈 对比分析**

通过对比 chat‑only 与 workflow 条件的 20 名受试者进行 within‑subject 研究，使用步骤/工具 recall/precision、过程理解度、验证成功率、任务完成率、耗时、感知难度等指标。实验显示，工作流条件下步骤 recall 与工具 recall 明显提升，过程理解度提升（0.82 vs 0.57），验证成功率从 30% 提升至 65%，而在后续任务中两种条件在成功率、耗时与难度上无显著差异。

**⚠️ 局限性**

局限性：①样本规模有限，且大部分受试者为经验型 AI 代理用户，工作流经验少者对工作流不熟；②任务规模和结构相对简单，未覆盖更大、更复杂的工作流；③仅测试了有限类型的错误，未覆盖所有可能的执行失误；④Trace2Flow 作为研究原型，其泛化性和对其他工作流平台的适配性仍需进一步验证。

---

## 477. A Hybrid LSTM-XGBoost Framework for Multi-Horizon Stock Return Prediction Across Diversified Equity Portfolios

**arXiv ID:** 2609.13125 | [PDF](https://arxiv.org/pdf/2609.13125v1)

**作者:** Seif ElDein Mostafa `[一作]` (MSA University), Marwa Solayman `[通讯]` (MSA University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个混合 LSTM–XGBoost 框架，用于对 14 支跨行业美国股票在 2010 年后数据上进行多周期（30、90、252、365 天）累计收益预测。

**💡 创新点**

创新点在于将 LSTM 提取的 64 维时序嵌入与 14 个手工技术指标拼接成 78 维特征向量，再输入 XGBoost 进行多周期回归，从而结合深度序列建模与树模型的非线性交互能力，并采用跨股票池化训练提升跨行业泛化。

**🔧 技术方法**

采用 2 层堆叠 LSTM（60 天窗口、64 维隐藏层）、MinMax 标准化、14 个技术指标、XGBoost 回归器（网格搜索 3 折交叉验证）、RMSE/MAE/R² 与方向准确率评估。

**📊 数据集**

使用 Kaggle 上公开的美国上市股票 OHLCV 数据，挑选 14 支股票（技术、金融、医疗、能源、消费、工业）并以 2010 年起的每日行情为训练、验证、测试集。

**📈 对比分析**

通过与仅 LSTM、仅 XGBoost 基线模型对比，测试集 RMSE 在 30 天 horizon 为 0.0949（比 LSTM 低约 3 倍），多周期性能随窗口增长递增，方向准确率随周期提升但受正回报基准率影响，整体表现对部分行业和高波动股票有显著提升。

**⚠️ 局限性**

局限包括 R² 均为负值，难以预测收益幅值；方向准确率受正回报基准率偏高，且未考虑交易成本；训练数据截至 2016 年，未覆盖后续结构变迁；跨股票池化训练未捕捉时间相关性和行业轮动，模型对未来市场变化的鲁棒性有限。

---

## 478. CMA-OT: Hierarchical Expert Supervision for Dance-to-Music Generation

**arXiv ID:** 2609.13118 | [PDF](https://arxiv.org/pdf/2609.13118v1)

**作者:** Jinting Wang `[一作]` (HKUST(GZ)), Li Liu `[通讯]` (HKUST(GZ))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 CMA-OT 框架，用预训练多尺度音乐专家的层级监督与可调优的课程学习策略，实现舞蹈到音乐（D2M）生成的节奏同步与音乐结构统一。

**💡 创新点**

创新点在于：① 将多尺度音乐专家作为层级监督源；② 设计课程化的粗到细对齐流程以缓解多尺度梯度冲突；③ 引入尺度感知 Fused Gromov‑Wasserstein (FGW) 对齐，既保持语义一致，又保留时序结构；④ 结合可调权重的自适应平衡因子，进一步提升对齐质量。

**🔧 技术方法**

核心技术包括：预训练 VQ‑VAE/音乐专家提取多尺度离散表示；Diffusion Transformer（DiT）作为音乐生成器；课程化多尺度对齐策略；尺度感知 FGW 与自适应平衡；条件流匹配（Conditional Flow Matching）实现高质量音频生成。

**📊 数据集**

使用了 AIST++ 与 TikTok 两个公开舞蹈-音乐配对数据集，其中 AIST++ 约 1,020 个专业舞蹈视频，TikTok 约 445 个短视频，覆盖多种舞蹈风格和音乐曲目。

**📈 对比分析**

与 D2M‑GAN、CDCD、LORIS、Textual‑Inv、MotionComposer 等现有方法在节奏对齐、音质、审美等指标上进行定量与定性对比；CMA‑OT 在 BCS、BHS、F1、FAD 等指标均取得 SOTA 级别提升（如 BCS 99.14% vs 95.84%），并在用户研究中获得最高的同步与美学分数。

**⚠️ 局限性**

局限性包括：① 对预训练音乐专家的依赖，若专家不足可能影响对齐质量；② 计算成本相对较高，尤其是 FGW 对齐与多尺度对齐的训练开销；③ 对极端节奏或结构极其多变的舞蹈视频仍可能出现对齐误差；④ 目前仅在音乐生成方面验证，尚未探索跨模态多模态交互或实时生成。

---

## 479. Autonomous Research for Open-Ended Problems: A Case Study on Telecom Ticket Retrieval

**arXiv ID:** 2609.13073 | [PDF](https://arxiv.org/pdf/2609.13073v1)

**作者:** Junghyun Min `[一作]` (Georgetown University), Mohamed Trabelsi `[通讯]` (Nokia Bell Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过电信工单检索任务，评估并实现了自适应研究框架在开放式工业机器学习任务中的应用。

**💡 创新点**

创新点在于提出针对开放式任务的操作性 harness，并系统比较了框架结构、LLM 选择和知识注入对研究效果的影响。

**🔧 技术方法**

使用了 LLM 驱动的单体与多体自动化研究框架（Claude Sonnet 5、Cursor Composer 2.5、GPT-OSS 120B），以及微调 MPNet 检索器等技术。

**📊 数据集**

使用了内部电信故障工单数据集（约 250k 真实工单及对应解析）。

**📈 对比分析**

通过与 BM25、单模型 fine‑tuned MPNet、模型集成及内部 SoTA 系统的 Recall@k 对比，自动化发现的单模型 Recall@1 为 0.343，优于人类单模型和集成但低于内部 SoTA。

**⚠️ 局限性**

局限性包括仅在单一任务和有限 LLM 上评估，缺乏严格对照实验，且自适应研究难以自行提出结构创新，通用性和可重复性尚待进一步验证。

---

## 480. Involving before Evolving: A Vision for Trustworthy Enterprise Digital Twin Engineering

**arXiv ID:** 2609.13071 | [PDF](https://arxiv.org/pdf/2609.13071v1)

**作者:** Kérian Fiter `[一作]` (Polytechnique Montréal), Bentley Oakes `[通讯]` (Michelin Canada)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种“先参与后演进”的企业数字孪生构建范式，并在米其林合作案例中实现了快速原型、基于本体的联邦化和可观测的信任机制

**💡 创新点**

创新点在于把社会技术参与与模型快速原型化、基于本体的互操作合同、可观测性三者整合为一个三阶段方法，并强调在长期决策循环中的信任建立

**🔧 技术方法**

利用大语言模型、时间序列变换器（TSFM）和蒙特卡罗模拟进行快速预测，采用 OML DSL 定义本体合同，使用 DTInsight 可视化追踪与解释结果

**📊 数据集**

主要使用米其林内部历史业务时间序列数据（营销、财务等部门的销售、成本等指标）进行预测与情景模拟

**📈 对比分析**

方法通过让业务专家直接在原型界面上编辑预测轨迹并评估结果来验证，未给出客观数值指标，仅通过专家接受度和持续迭代来体现效果

**⚠️ 局限性**

限制包括：本体合同在大规模部门间可能面临语义异质性与一致性检查的可扩展性挑战，缺乏情景来源追踪机制，且由于长周期决策难以在短期内量化效果

---

## 481. Pixel Decodability Is Not a Compression Signal: Causally Evaluating Importance Proxies for Visual KV-Cache Eviction

**arXiv ID:** 2609.13012 | [PDF](https://arxiv.org/pdf/2609.13012v1)

**作者:** Chenyu Zhou `[一作]` (Institute of Science Tokyo), Xu Zhou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究在视觉‑语言模型（VLM）中评估视觉KV缓存的像素可解码内容与模型实际使用之间的关系，并提供一种单单元因果消融测量；

**💡 创新点**

创新点在于提出并实现了三维度（像素可解码保留、注意力权重、因果利用）评估框架，证明像素可解码保留与模型利用解耦；

**🔧 技术方法**

技术主要包括：学习像素反演解码器（pixel‑inversion decoder）衡量保留；单超块KV消融（single‑super‑patch ablation）测量因果利用；注意力权重提取；以及基于rank partial correlation的统计分析；

**📊 数据集**

使用的公开数据集为TextVQA验证集（包含图像与文字问题），并在两个VLM上评估：Gemma‑4‑12B（无Encoder）与InternVL3.5‑8B（Encoder‑based）；

**📈 对比分析**

比较方法：将保留、注意力和利用的相关性作为评价指标，并在缓存压缩实验中以不同消融策略（FastV注意力、原始保留、去混淆保留、组合、随机）对比；结果显示注意力在所有粒度下显著优于随机且优于去混淆保留，像素可解码保留在超块层面无效，在Token层面仅产生弱逆向信号；

**⚠️ 局限性**

局限性包括：仅测试了两款模型，无法推广到更广泛架构；消融仅单单元，未考虑冗余分布；TextVQA任务依赖答案定位，其他视觉任务可能表现不同；层次选择固定，未完全覆盖所有层；

---

## 482. Comfort by Construction: Adaptive, Comfort-Bounded Action Spaces for Learned Driving Policies

**arXiv ID:** 2609.13011 | [PDF](https://arxiv.org/pdf/2609.13011v1)

**作者:** Anna Rothenhäusler `[一作]` (University of Freiburg), Joschka Boedecker `[通讯]` (University of Freiburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种自适应的动作空间映射，通过在每一步解析并重划定舒适约束下的可执行控制集，来保证强化学习驾驶策略在真实车辆的加速、侧向加速度与颤振范围内。

**💡 创新点**

创新点在于：① 使用闭式解析逆转侧向颤振约束得到精确可执行集；② 在每个时间步对动作网格进行重划定（Adaptive）以避免“网格崩塌”，保证动作分辨率；③ 将舒适约束直接嵌入动作空间而非后期惩罚。

**🔧 技术方法**

技术手段包括：基于卡尔曼双轮运动学模型的运动方程；使用Occupant’s Preference Metric (OPM)定义的纵向/横向加速度与颤振舒适界限；闭式逆推侧向颤振限制得到的可执行控制盒；对经典 7×13 网格进行剪裁与重划定两种策略；使用PPO强化学习框架进行训练。

**📊 数据集**

数据集：Waymo Open Motion Dataset（WOMD）80,000 场景用于训练，10,000 场景用于验证；另外使用手工绘制的急转弯（Slalom）等压力测试场景。

**📈 对比分析**

对比方法包括 Classic、Classic(steer‑angle)、Jerk（静态限幅）、Clipped（按步剪裁）和 Adaptive（本文方法）。实验显示 Adaptive 在所有指标上均优于剪裁与 Jerk：舒适违规率降至 <0.01%，目标完成率仅比 Classic 降低约 2%（在 Aggressive 约束下），且碰撞率、偏离道路率更低，导航性能更稳健。

**⚠️ 局限性**

局限性：① 需要每步实时计算可执行盒，增加计算开销；② 依赖前一时刻已实现的动作信息，若观测缺失则为部分可观测问题；③ 对极端高速或极限转弯的可执行集仍可能非常窄，导致控制分辨率受限；④ 仅在二维平面运动学框架下验证，未考虑三维动力学或复杂道路条件。

---

## 483. Dual-guided Hierarchical Edge Localization for Large-scale Optimal Transport Across Dimensions

**arXiv ID:** 2609.13010 | [PDF](https://arxiv.org/pdf/2609.13010v1)

**作者:** Wenzhou Xia `[一作]` (Shanghai Jiao Tong University), Xiaoqun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于双向引导的层次化边缘定位（Dual‑Guided Hierarchical Edge Localization，DG‑HEL）求解器，用双重潜能信号在粗到细层级中初始化以及在每层级内进行稀疏支持更新，最终实现大规模离散最优运输（OT）问题的高效求解。

**💡 创新点**

创新点主要包括：
- 通过双重潜能的共用分数同时完成粗层到细层的初始化和细层的违规检测；
- 设计了预算剪枝机制，保证活跃支持始终线性(≈m+n)复杂度；
- 证明在符号列举规则下，精确算术迭代终止于全局最优；
- 将该框架推广到任意配对成本、半离散OT、Gromov–Wasserstein、未平衡OT以及流匹配的离散/连续端点生成；
- 引入成本扰动温和起始点和成本扰动热启动以提升鲁棒性。

**🔧 技术方法**

核心技术：层次化随机子采样、双重潜能传播、双重潜能评分、基于评分的顶k选边初始化、双重违规检测（行列top‑γ）、预算剪枝（β|I|+|J|）、符号列举（lexicographic）终止、GPU并行流式评分评估、GPU上限制LP求解、成本扰动热启动。

**📊 数据集**

实验数据集：
- Gaussian‑to‑ImageNet（高维 8192D、样本≈1.28M）
- ImageNet‑1k 8192D latent
- MNIST–Fashion‑MNIST
- TOME（未平衡OT基准）
- CIFAR‑10（连续源流匹配）
- FFHQ（离散源图像翻译）
- 其它合成均匀/非均匀单调/非单调OT实例。

**📈 对比分析**

与基准比较：
- 与熵正则化（Sinkhorn）和近似近端点方法、低秩层次化方法、HELLO‑GW 等进行对比；
- 在百万级规模（n≈2^20）和高维（d≥256）下，DG‑HEL 的运算时间比最佳基准低 1–2 个数量级；
- 目标函数误差和全相对KKT残差均维持在 10^−6 以下；
- GPU峰值内存降至 8.7 GiB（相比 48.2 GiB 降幅 82%）；
- 运行时与规模的经验复杂度模型为 T≈A·n + B·n^2·d，拟合 R²≈0.998。

**⚠️ 局限性**

局限性：
- 对极端稀疏或非凸成本（如 ℓ1、ℓ∞）时需要成本扰动热启动，可能导致额外迭代；
- 活跃支持的预算剪枝在 γ=1 时可能收敛缓慢；
- 算法高度依赖 GPU 并行，CPU 版本效率有限；
- 目前未直接处理多项式 OT 或更一般的未平衡约束，仅通过嵌入求解器实现；
- 超大维度下的双重评分和违规检测仍需 O(n^2) 计算，未来可进一步加速。

---

## 484. SV-Cine: Diagnosis-Conditioned Segmentation of Single Ventricle Physiology via Generative Data Augmentation

**arXiv ID:** 2609.12997 | [PDF](https://arxiv.org/pdf/2609.12997v1)

**作者:** Lila Cunge `[一作]`, Kim-Lien Nguyen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种基于诊断条件的心脏MRI分割框架SV-Cine，能够在单心室生理中分别分割主导心室、低发育心室和心肌；

**💡 创新点**

创新点在于将诊断信息作为先验通过FiLM层注入分割网络，并结合SDF4CHD+GAN生成的合成数据实现多样化训练；

**🔧 技术方法**

使用的技术包括SDF4CHD（签名距离函数生成几何）、cGAN生成合成短轴MRI、CineMA基础模型（masked autoencoder）、FiLM条件化、以及nnU-Net等基线网络；

**📊 数据集**

使用的数据集包括公开的ImageCHD、HVSMR，以及UCLA（20例）和CHOC（19例）单心室病例；

**📈 对比分析**

与UCLA和CHOC的内部/外部测试数据比较，SV-Cine在左心室和右心室的Dice分别达到0.89和0.72，显著优于nnU-Net、U-Net3+和原CineMA；

**⚠️ 局限性**

主要限制包括样本量有限、对极度低发育心室的分割仍易出错、对合成图像真实性的依赖，以及需要预先提供诊断信息。

---

## 485. A Full Adam Theorem for Spectral Heavy-Tail Onset

**arXiv ID:** 2609.12996 | [PDF](https://arxiv.org/pdf/2609.12996v1)

**作者:** Zongmin Liu `[一作]` `[通讯]` (Stanford University), Zongmin Liu (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在高斯 Stein‑Hermite 教师-学生闭合状态演化模型中，完整批 Adam 优化算法能够在出现首个尖峰‑粗体间隙后，按照特定幂律比例达到谱重尾窗口，并给出了精确的到达时间上界与下界，构成了完整的 Adam‑到‑重尾时间理论。

**💡 创新点**

创新点在于：1）首次将 Adam 的偏差校正动量、坐标阶数分母以及精确 Gram 更新完整地映射到状态演化模型中；2）引入非中心高斯符号核与基底均匀化定理，揭示 Adam 分母对谱尾指数的影响；3）通过 Hermite 边缘传递定理，将首个尖峰产生的边缘坐标转化为正则变异投影更新；4）证明了在此模型下任意梯度 Adam 迭代无法单独导致重尾出现。

**🔧 技术方法**

使用的技术包括：Stein‑Hermite 计算梯度、有限宽度协方差浓缩、非中心高斯符号核推导、基底均匀化定理、Hermite 边缘传递定理、Gram 更新的标量化谱工作、近似目标 KL 收缩定理以及严格的概率上界与下界分析。

**📊 数据集**

该工作基于理论模型，无实验数据集；作者在附录中给出了有限尺寸的数值审核以验证理论推导。

**📈 对比分析**

由于缺乏实际数据实验，本文不进行方法比较；理论上给出了 Adam 到达谱重尾窗口的时间标度 τ_ε = Θ(Δ_1^{-γ} d^ρ log(Ψ_0/ε))，并在理论层面证明了对应的上界与下界一致。

**⚠️ 局限性**

局限性包括：1）仅在理想化的 Gaussian Stein‑Hermite 教师-学生模型下成立，难以直接推广到真实神经网络；2）对 Adam 的完整批形式做了假设，未考虑 mini‑batch 或随机性；3）理论依赖于多项式 Hermite 系数的可求和性与边缘传递系数非零等技术假设；4）未给出对实际大模型的数值验证，仅提供有限尺寸审核。

---

## 486. Quantile-based Loss Filtering for Outlier-Robust Stochastic Gradient Descent

**arXiv ID:** 2609.13040 | [PDF](https://arxiv.org/pdf/2609.13040v1)

**作者:** Jamie Haddock `[一作]` (Harvey Mudd College), Elizaveta Rebrova `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究一种基于量化损失过滤的随机梯度下降方法QkL-SGD，用于在有限和式优化中处理少量不可靠或被破坏的组件函数。

**💡 创新点**

创新点在于将低分位量的损失筛选与随机抽样相结合，形成从min‑k‑loss到标准SGD的通用框架，并给出大样本和小样本两种理论保证。

**🔧 技术方法**

采用量化损失筛选的随机梯度更新，结合凸性、光滑性与子集强凸性假设，证明线性收敛；在小样本下给出概率收敛分析。

**📊 数据集**

在实验中使用多项式回归、正则化逻辑回归与正则化铰链损失等合成数据集，包含不同比例的标签或响应值伪造。

**📈 对比分析**

与标准SGD和min‑k‑loss SGD比较，QkL‑SGD在多种污染模型下往往收敛更快、误差更低，尤其在中等量化水平时性能最佳。

**⚠️ 局限性**

局限性包括对子集强凸性、损失分离或小样本下的概率假设依赖较大；在极端样本量或极高污染比例下，收敛速率或误差上限可能退化。

---

## 487. How Good Are Frontier Models at Physics? Expert Re-Grading Reveals Broken Evaluations and Near-Saturation of Leading Benchmarks

**arXiv ID:** 2609.13009 | [PDF](https://arxiv.org/pdf/2609.13009v1)

**作者:** Ali Ansari `[一作]` (Yale University), John Sous `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对六个主流物理基准进行专家审核，纠正问题陈述、参考答案和评估程序，重新评估前沿语言模型的物理推理能力。

**💡 创新点**

发现大部分低分源于基准错误（题目模糊、答案错误、评估器不一致），而非模型本身，提出“验证/修复”流程并提供新评估结果。

**🔧 技术方法**

采用专家人工审核、LLM评估器（如HLE评估器）、规则/符号评估器、工具启用的推理回合以及多次尝试策略。

**📊 数据集**

使用UGPhysics、PHYBench、PRISM-Physics（公开源题库）以及HLE-Physics、CMT-Benchmark、CritPt（专家原创题库）六个基准集。

**📈 对比分析**

对比预审核与修复后得分，显示GPT‑5.6‑Sol在HLE-Physics从≈10%提升至≈70%，在CMT-Benchmark从≈30%提升至≈90%，CritPt从≈20%提升至≈95%（均为近乎完美）。

**⚠️ 局限性**

局限包括：仍仅覆盖闭式文本题；修复基准的工作量大；未验证模型在开放式科研任务中的表现；对评估器本身的误差仍可能影响结果。

---

## 488. Unified CT and MRI Pancreas Segmentation for Label-Efficient Cross-Modality Subregion Transfer

**arXiv ID:** 2609.13043 | [PDF](https://arxiv.org/pdf/2609.13043v1)

**作者:** Ziliang Hong `[一作]` (Northwestern University), Ulas Bagci `[通讯]` (Northwestern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

发展统一的CT‑MRI胰腺分割框架，利用域对抗学习获得无模态的解剖表示，并在仅有单模态子区域标注的条件下完成跨模态子区域分割。

**💡 创新点**

首次将域对抗学习与nnU-Net结合，学习跨模态无关特征；在缺乏CT子区域标注时仍能实现CT子区域分割，展示了标注高效的跨模态迁移能力。

**🔧 技术方法**

使用nnU-Net编码解码器、梯度反转层（GRL）与域判别器进行域对抗训练；采用Dice+CE损失、三阶段训练（warm‑up→对抗→下游）以及t‑SNE可视化评估特征对齐。

**📊 数据集**

在4,604多中心扫描（Cyst‑X、private MRI、AbdomenCT‑1K、Peri‑Pancreatic Edema）构成ID训练集；外部OOS集为AMOS、BTCV、U‑Mamba；子区域标注仅来自Cyst‑X。

**📈 对比分析**

与基线单模态nnU-Net比较，ID测试集Dice 87.31%，OOS 84–88%；下游子区域Dice 80.53%（MRI）和83.05%（CT），并在CT无子区域标注时达成，显著优于基线。

**⚠️ 局限性**

仅对CT‑MRI一级对齐，未细化扫描仪/序列差异；子区域实验仅针对胰腺；对其他器官的泛化尚未验证；对抗训练参数选择经验性，t‑SNE仅提供定性洞察。

---

## 489. Transfer Learning for Evolving Domains

**arXiv ID:** 2609.13039 | [PDF](https://arxiv.org/pdf/2609.13039v1)

**作者:** Ricardo Ribeiro Pereira `[一作]` (Feedzai), Carlos Soares `[通讯]` (University of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并形式化了一个新的迁移学习问题，定义了数据可用性随时间演化的过程、可自由选择的学习协议以及对整个模型轨迹进行累积评估的评价标准，并阐释了传统的域泛化、无监督/有监督域适应和多域学习如何作为该问题的不同阶段出现。

**💡 创新点**

创新点在于将迁移学习从孤立的静态子问题转变为一个时间动态的整体问题，统一了四类经典设定为可演化的轨迹；提出了累积风险评估指标；并系统地分析了现有技术如何作为解决此动态问题的构建模块。

**🔧 技术方法**

主要技术回顾包括：实例加权、特征映射、伪标签、模块化深度学习架构、集成学习、对抗/对比损失、元学习、自监督损失以及大规模预训练模型（基础模型）的多阶段适配策略；同时提出了阶段切换基线（stage‑switching）作为可比参照。

**📊 数据集**

文中未进行实验验证，未使用具体数据集，而是以理论分析和文献综述为主。

**📈 对比分析**

比较方法：提出了累计风险（cumulative target risk）作为评价指标，并讨论了对该指标的估计方式；虽然未给出实验结果，但指出阶段切换基线已是强劲的参照，任何真正的 π 解决方案都应在该指标上优于该基线。

**⚠️ 局限性**

局限性：目前尚无方法能够在整个轨迹上实现全局最优；阶段切换的转变时机难以准确估计；标签延迟导致评估和决策受限；不同技术只能在各自的阶段发挥作用，缺乏统一的跨阶段策略；基础模型虽具备多阶段适配能力，但仍未形成完整的 π 解决方案。

---

## 490. Label-Guided Knowledge Distillation for 3D-CNNs in Action Recognition

**arXiv ID:** 2609.13024 | [PDF](https://arxiv.org/pdf/2609.13024v1)

**作者:** Yanjiang Shi `[一作]` (Xi'an Jiaotong University), Guiqin Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于标签引导的知识蒸馏方法（LGKD），用于压缩并提升3D‑CNN动作识别模型的性能。

**💡 创新点**

创新点在于：①将标签信息与教师的预测概率相结合，对每帧特征进行时序重要性加权，实现样本级蒸馏；②使用原型网络在类别层面对学生特征进行聚类对齐，完成类别级蒸馏，从而同时提升低维特征表达和高阶语义捕获。

**🔧 技术方法**

采用的技术包括：样本级权重调整模块（基于教师softmax分布的加权L2损失）、类别级原型网络（移动平均更新类别原型并对齐学生特征）、传统的特征L2对齐以及整体多损失训练策略。

**📊 数据集**

在UCF101和HMDB51这两大动作识别基准数据集上进行实验验证。

**📈 对比分析**

与多种响应式、特征式、注意力式蒸馏方法（Logits KD、CTKD、SimKD等）进行对比；在UCF101上，学生Top‑1从63.36%提升至73.51%（+10.15%），Top‑5从82.08%提升至90.30%（+8.22%）；HMDB51亦取得类似显著提升，并在不同学生网络结构上表现出良好的迁移性。

**⚠️ 局限性**

局限性在于：实验仅聚焦于动作识别任务，未检验对更大规模网络或其他视频任务的适用性；对标签细粒度信息的利用尚未深入，未来可进一步探索。

---

## 491. Global Path Planner with Multi-Model Switching

**arXiv ID:** 2609.13015 | [PDF](https://arxiv.org/pdf/2609.13015v1)

**作者:** Pietro Gori `[一作]` (University of Pisa), Manolo Garabini `[通讯]` (University of Pisa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出了一个整合的机器人导航框架，结合Heading‑Aware A*规划器与多模型Pure Pursuit控制器，实现了在复杂平面及非平面环境中的路径规划与跟踪；

**💡 创新点**

创新点包括：①在A*搜索中加入方向信息与坡度惩罚，生成更平滑、可行的路径；②控制器根据机器人姿态与地形自适应切换非齐次运动模型（单车/全向），提升跟踪效率和能耗；

**🔧 技术方法**

采用的技术有：基于点云的通行性地图构建、Heading‑Aware A*算法、可变动力学模型的Pure Pursuit控制、欧氏距离与角度惩罚的代价设计；

**📊 数据集**

数据集：使用实时传感器获取的原始点云（无公开公开数据集），并在此基础上构建2.5D通行性图与拓扑图；

**📈 对比分析**

通过在仿真平台（Artaban四足机器人与X3四旋翼）以及真实Artaban硬件上进行对比实验，结果显示相较于传统A*＋单一运动模型，路径长度更短、能耗更低、轨迹跟踪误差更小；

**⚠️ 局限性**

局限性：仅针对静态环境，未实现动态障碍物避让与在线重规划；控制器对极端地形变化的鲁棒性待进一步验证；

---

## 492. TileNet: Tile-Based CNN-SVM Architecture for Autonomous Unmanned Aerial Systems Inspection of Flat Roofs

**arXiv ID:** 2609.13013 | [PDF](https://arxiv.org/pdf/2609.13013v1)

**作者:** Samuel Dunthorne `[一作]` (Carleton University), Hashim A. Hashim `[通讯]` (Carleton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该研究开发了一套基于无人机的实时平屋顶缺陷检测框架，利用图像切块和轻量级CNN‑SVM分类实现无人机自主巡检。

**💡 创新点**

创新点在于将高分辨率影像拆分为轻量化tile，结合轻量CNN与线性SVM头，兼顾高准确率（94.4%）与低算力需求，并引入双高度飞行策略提升小缺陷检测。

**🔧 技术方法**

使用了轻量级卷积神经网络、支持向量机（SVM）头、图像切块、数据增强、SGD优化和三重种子验证等技术。

**📊 数据集**

采用自建的改性沥青平屋顶缺陷图像集，包含43,383张训练tile、3,869张验证tile和2,540张测试tile，并已公开在Zenodo。

**📈 对比分析**

与GoogLeNet（89.2%）和AlexNet（79.8%）比较，CNN‑SVM在相同输入下达到94.4%准确率，推理时间仅7.8 ms/36‑tile图像（约130图像/秒）。

**⚠️ 局限性**

主要限制包括训练/测试仅在图像级别，可能导致重叠泄漏；模型在嵌入式平台未验证；仅针对改性沥青膜，未覆盖其他屋面材料；双高度策略效果未单独归因。

---

## 493. SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking

**arXiv ID:** 2609.13141 | [PDF](https://arxiv.org/pdf/2609.13141v1)

**作者:** Zhiwei Li `[一作]` (Tencent HY LLM Frontier), Zhijiang Guo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

由于未提供论文具体内容，无法得知作者做了什么。

**💡 创新点**

无法确定论文的创新点。

**🔧 技术方法**

无法确定使用的技术。

**📊 数据集**

无法确定使用的数据集。

**📈 对比分析**

无法确定比较方法及性能表现。

**⚠️ 局限性**

无法确定论文的局限性。

---

## 494. Tasks over Application Manuals: Revealing Gaps in Long-Horizon Procedural Reasoning for Language Models

**arXiv ID:** 2609.13005 | [PDF](https://arxiv.org/pdf/2609.13005v1)

**作者:** Utkarsh Soni `[一作]` (Manulife), Eugene Wen `[通讯]` (Manulife)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并发布了TAM基准，用以评估大型语言模型在跟随庞大、交叉引用手册执行多步程序的能力，并对ICD‑10‑CM编码与美国联邦量刑指南两类真实任务进行实验。

**💡 创新点**

首创将长周期、规则驱动的程序化推理抽象为可测评任务，强调完整决策路径而非单步答案，并揭示现有LLM在此类任务中的显著缺陷。

**🔧 技术方法**

采用检索增强生成（RAG）、代理式RAG、ReAct工具调用以及LangChain深度代理框架，对GPT‑5进行提示和工具交互实验。

**📊 数据集**

通过MIMIC‑IV抽取1,000例ICD‑10‑CM病例及其手册（2019指南、索引、表格），以及从CourtListener整理的200例联邦量刑案例与对应的Title‑18、USSG手册。

**📈 对比分析**

采用精确匹配、精度/召回、首诊准确率及平均绝对误差等指标，结果显示ICD编码精确匹配仅≈1%，法律判刑精确匹配最高达15.5%，整体表现远低于短周期推理基准。

**⚠️ 局限性**

仅覆盖两种高度结构化手册的任务，实验仅使用提示式基线且未包含领域特定微调或更先进代理方法，数据规模与多样性有限。

---

## 495. Anchoring Clinical Events in Time: UID-Preserving Multimodal Reconstruction and Source-Grounded Adjudication

**arXiv ID:** 2609.13062 | [PDF](https://arxiv.org/pdf/2609.13062v1)

**作者:** Sayantan Kumar `[一作]` (National Library of Medicine, National Institutes of Health), Jeremy C. Weiss `[通讯]` (National Library of Medicine, National Institutes of Health)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个保留唯一标识符的多模态临床时间线重建框架，并引入了基于源证据的 GAVEL 判定器对两条时间线与原始记录进行对比。

**💡 创新点**

创新点在于通过 UID 维持事件身份、将结构化 EHR 行作为时间证据、以及使用基于源证据的 LLM 判定器 GAVEL 进行差异评估。

**🔧 技术方法**

采用大型语言模型（GLM 5.2、DeepSeek V3.2 等）进行事件标注、时间估计、检索与联合修订，并用 GLM 5.2 进行 GAVEL 判定。

**📊 数据集**

使用 40 条混合危重症出院摘要（15 条 i2b2、25 条 MIMIC‑IV）及对应的结构化 EHR 表。

**📈 对比分析**

与单一文本或先前方法对比，GLM 5.2 多模态重建在事件匹配率 0.790、时间一致性 0.802、AULTC 0.773 上均有提升，且 GAVEL 在对比中优于文本版。

**⚠️ 局限性**

局限包括样本量小、仅评估 40 例、文本主导导致纯结构化事件缺失、结构时间戳可能不代表事件发生时刻，以及 GAVEL 评价仅检验发现正确性而非完整性。

---

## 496. A Unified and Constrained View of Regularization-Based Robust Reinforcement Learning

**arXiv ID:** 2609.13050 | [PDF](https://arxiv.org/pdf/2609.13050v1)

**作者:** Amine Andam `[一作]` (Mohammed VI Polytechnic University), Mustapha Hedabou `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过推导新的性能差距上界，将三种主流正则化方法统一，并将鲁棒训练转化为约束优化问题，提出了 RaC-SA 与 RaC-Radial 两种自适应正则化算法。

**💡 创新点**

创新点在于：①推导出包含 SA-Reg、Radial 与 WocaR-RL 正则项的统一上界，解释了加入 KL 约束提升鲁棒性的原因；②将鲁棒性视为约束，在 Lagrangian 框架下自适应调节正则化权重，得到新的 RaC-Mono、RaC-SA、RaC-Radial 方法。

**🔧 技术方法**

采用的技术包括：基于凸松弛的输出边界估计、政策梯度优化（PPO/ TRPO）、Lagrangian 约束优化、自动求解 KL 与总变差距离、以及自适应梯度平衡系数 β。

**📊 数据集**

实验使用了 MuJoCo 连续控制基准（HalfCheetah、Hopper、Walker2d）以及 2328 组自适应对抗攻击（RS、SA-RL、PA-AD）进行鲁棒性评估。

**📈 对比分析**

与现有方法相比，RaC-SA 与 RaC-Radial 在所有环境和攻击下均取得最优或最接近最优的最坏情况性能，鲁棒性能显著高于 SA-Reg、Radial、WocaR-RL 等基线；训练时间仅比基线多约 0.5 小时。

**⚠️ 局限性**

局限性包括：①对 ε 以及 η 等阈值的选择仍需经验；②对极端攻击或高维环境的鲁棒性验证尚未充分；③算法依赖于凸松弛计算，可能在更复杂网络或更大状态空间下产生误差。

---

## 497. MCRL2: Multi-resource Cross-attention-based Representation Learning-augmented Reinforcement Learning for Cloud Microservice Scheduling

**arXiv ID:** 2609.13048 | [PDF](https://arxiv.org/pdf/2609.13048v1)

**作者:** Tiangang Li `[一作]` (Wuhan University), Ding Xiao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 MCRL2——一种将多资源交叉注意力表示学习与软演员‑评论家（SAC）最大熵强化学习相结合的云微服务调度框架，旨在解决资源不平衡、跨维度资源耦合以及请求异质性等关键难题。

**💡 创新点**

创新点包括：① 在节点、资源与微服务层面构建多资源交叉注意力机制，显式捕获动态的跨维度依赖与协同；② 设计双流（actor/critic）架构，在两条路径中独立使用 MCRL 表示学习，提升策略与价值网络的表达能力；③ 通过熵正则化与表示引导的探索，强化策略的稳定性与泛化能力；④ 结合残差连接与层归一化，使模型更易训练。

**🔧 技术方法**

主要技术手段：多资源交叉注意力（Multi‑Resource Cross‑Attention）表示学习；软演员‑评论家（SAC）最大熵强化学习；双流网络架构（RL‑Actor + RL‑Critic）；经验回放与自适应熵温度；PyTorch + Python 实现；资源利用率、请求向量等时序特征建模。

**📊 数据集**

数据集：阿里巴巴生产集群微服务跟踪数据（CPU 与内存利用率，90k+ 容器），从中抽取 2000/4000 个实例作为训练与测试；仿真 15 节点异构集群（3 种 VM 规格），用于评估调度效果。

**📈 对比分析**

对比方法：Round Robin、Random、DDQN、PPO、SAC；评估指标包括收敛速度、负载平衡（DCLB/DRLB）、调度成功率（MSSR）、平均完成时间（ACTI）以及决策时间。MCRL2 在所有指标上均优于基线：收敛最快、负载平衡最优、成功率最高、完成时间最低，决策时间略高但在可接受范围内。

**⚠️ 局限性**

局限性：① 仅在 15 节点模拟环境中验证，缺乏大规模集群的可扩展性评估；② 决策推理时间相对传统 DRL 基线略长；③ 模型复杂度高，训练与部署成本相对较高；④ 依赖完整的实时系统监控信息，对监控延迟或缺失敏感；⑤ 目前主要针对 CPU/内存两种资源，扩展到更多资源维度需进一步验证。

---

## 498. Mitigating Emergent Collusion in LLM Pricing Agents

**arXiv ID:** 2609.13037 | [PDF](https://arxiv.org/pdf/2609.13037v1)

**作者:** Abdullah Garra `[一作]` `[通讯]` (University of Massachusetts Amherst), Abdullah Garra (University of Massachusetts Amherst)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在重复Bertrand寡头市场中，使用DeepSeek‑V3.1 LLM代理测试自然语言提示对价格竞争的影响，并评估三种监管干预（提示警告、预期损害惩罚、随机第三方进入）对降低超竞争性定价的效果。

**💡 创新点**

首次将LLM自发定价行为与传统强化学习代理对比，提出通过“植入”技术在文本推理中实施因果干预；同时设计基于预期损害的监管框架和结构性进入干预，验证不同监管渠道对LLM价格行为的影响。

**🔧 技术方法**

采用预训练语言模型DeepSeek‑V3.1；使用logit需求模型和Bertrand寡头框架；实现提示生成、价格更新、利润计算、惩罚机制；利用回归与文本聚类分析奖励‑惩罚模式。

**📊 数据集**

实验数据来源为自建的重复Bertrand游戏环境：对称需求参数（β=100、a_i=2、a_0=0、μ=0.25、c_i=1、α随机取1、3.2、10），共计7个实验组、每组7次运行，共计2100条价格观测。

**📈 对比分析**

通过Welch t检验对P1/P2、监管干预前后价格与利润差异进行统计；使用“协同指数”衡量与Nash/垄断基准的偏离；结果显示提示警告略降价差异，预期损害监管将P1价格逼近Nash并消除P1/P2差距，随机进入最强降价效果。

**⚠️ 局限性**

受限于单一LLM模型、少量运行样本（7次/组），实验缺乏多模型或真实市场数据验证；随机第三方进入的结构性干预难以视为轻量监管，且可能导致过度竞争；预期损害监管参数为经验调优，缺乏理论最优性。

---

## 499. Attention Quantization for Tabular Foundation Models

**arXiv ID:** 2609.13031 | [PDF](https://arxiv.org/pdf/2609.13031v1)

**作者:** Jonas M. Kübler `[一作]` (Prior Labs), Frank Hutter `[通讯]` (Prior Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对表格基础模型，本文提出了对注意力计算进行FP8量化的方案，并实现了高效的Triton核加速推理。

**💡 创新点**

创新点在于：①在注意力的Q/K/V上使用逐头绝对最大值缩放量化，并同步使用训练与测试行的同一缩放因子；②构建了利用FP8矩阵乘法指令的专用核，显著减少对权重KV缓存量化的依赖。

**🔧 技术方法**

采用的技术包括FP8矩阵乘法、逐头absmax动态量化、softmax指数调整、Triton实现、黑曜石GPU的FP8指令集。

**📊 数据集**

使用的评测数据集为TabArena（51个数据集）、BeyondArena、TabPFN‑v3、TabICLv2。

**📈 对比分析**

与FP16 SDPA/FA2基准在RTX Pro 6000上对比，FP8注意力核实现了高达1.7×的单核加速、1.67×的端到端加速，且在TabArena和BeyondArena上的Elo得分误差在可接受范围内（≈1 Elo以内）。

**⚠️ 局限性**

局限性：仅在训练样本数>8192时门控生效；依赖支持FP8指令的黑曜石GPU，移植到其他架构需额外工作；当门控关闭时仍可能出现细微的准确率退化。

---

## 500. Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage

**arXiv ID:** 2609.13003 | [PDF](https://arxiv.org/pdf/2609.13003v1)

**作者:** Foad Namjoo `[一作]` (University of Utah), Jeff M. Phillips `[通讯]` (University of Utah)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

审计并清理二元真伪评测数据集中因表面特征导致的“泄漏”问题，提出可解释的六维特征和 Audit‑Prune 机制，生成更平衡的 TruthfulQA 子集。

**💡 创新点**

创新点在于首次用可解释的表面特征（否定、保留、长度等）构造简单的逻辑回归探测器，证明泄漏可被利用；并设计迭代式的 Audit‑Prune 清洗算法，显著降低泄漏并保持模型排名。

**🔧 技术方法**

技术包括手工设计的六维表面特征、标准化+L2 正则化逻辑回归、分组交叉验证、AUC 评估、贪心式逐对删除与重加法的 Audit‑Prune 算法，以及与 AFLite 的对比实验。

**📊 数据集**

主要使用 TruthfulQA（二元版本）作为目标数据集，并对 13 个其他二元/对比式评测集进行跨数据集审计；此外还构造 131 对“Surface‑Flipped”和“Natural”对照集进行泛化验证。

**📈 对比分析**

相较于原始 TruthfulQA，Audit‑Prune 生成的 476 对子集将表面泄漏 AUC 从 0.715 降至 0.528，模型排名相关性保持在 Spearman ρ≈0.915；与 AFLite 对比，清洗后泄漏更低（AUC 0.528 vs 0.603）且排名一致性相近。

**⚠️ 局限性**

局限在于特征集仅覆盖否定/保留/长度等表面模式，未能捕获更深层次的句法或语义泄漏；Audit‑Prune 为贪心近似，无法保证全局最优；清洗后子集仍可能携带原始数据的记忆或污染风险。

---

## 501. Dimension-Corrected Hitting Times for Heavy-Tailed Spectral Emergence in Neural Optimizer Dynamics

**arXiv ID:** 2609.12994 | [PDF](https://arxiv.org/pdf/2609.12994v1)

**作者:** Zongmin Liu `[一作]` `[通讯]` (Stanford University), Zongmin Liu (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了全批量 Adam 在教师‑学生模型中权重矩阵谱的重尾出现，并将其视为右删失的触发时间问题，推导了维度校正的经验触发公式；

**💡 创新点**

将重尾谱出现建模为右删失触发时间；提出维度校正的经验法则 τ≈CΔ₁⁻^γ d^ρ；用对数正态加速失效时间模型处理删失数据；证明 Adam 更新不必导致谱重分布，给出投影单基展开条件并给出条件性触发定理；

**🔧 技术方法**

采用触发时间与删失模型（AIC、加速失效时间）、回归分析、随机矩阵理论、投影核分析、Adam/AdamW/GD/signGD 优化器实验，以及 Hill 指数、KS 测试等谱诊断技术；

**📊 数据集**

主要在教师‑学生模拟中使用 d∈{200,500,1000}、h=1.5d、n=4d 的高维高斯数据；附录中还检验了 sklearn digits、微型 Transformer、Pythia‑70M 与 Qwen2.5‑0.5B 等 sanity 检查；

**📈 对比分析**

通过比较不同优化器的重尾出现时间、gap‑only 与维度校正模型的拟合优度（R²=0.683、AIC 从 706.62 降至 628.70）、投影核机制与随机基线的对比，发现投影核机制更优；重尾不一定提升泛化，Ridge readout MSE 非单调；

**⚠️ 局限性**

缺乏从 Adam 递推得到投影谱展开的第一性原理证明；机制证明仅为经验验证；实验仅在教师‑学生模拟中完成，未在大规模真实模型中系统验证；重尾与泛化关系不确定；优化器通用性仅在 Adam/AdamW 上得到支持，GD/signGD 未满足触发条件。

---

## 502. Rank-1-perturbed trickledown theorems: Mixing time of Glauber dynamics for the Sherrington-Kirkpatrick model up to $β\leq \frac{1}{2}+\varepsilon$

**arXiv ID:** 2609.13138 | [PDF](https://arxiv.org/pdf/2609.13138v1)

**作者:** Mathews Boban `[一作]` (University of Washington), Shayan Oveis Gharan `[通讯]` (University of Washington)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种新的 "rank‑1 失真 trickledown 定理"，用于在多态自旋系统中通过局部到全局的技术上界谱间隙，并将该定理应用于 Sherrington–Kirkpatrick (SK) 模型，证明在逆温 β≤1/2+ 的范围内，Glauber 动力学在多项式时间内混合，混合时间为 O(n²)。

**💡 创新点**

创新点在于：① 用 rank‑1 失真替代传统的 λI 上界，显著降低了上界误差；② 通过平均化所有二元链接的贡献，得到对平均谱影响的控制；③ 将谱独立性、随机矩阵理论与 Gibbs 采样相结合，首次在随机交互矩阵的完整图模型上得到 β≈1/2+ 的谱间隙下界。

**🔧 技术方法**

核心技术包括：spectral independence、rank‑1 失真 trickledown、随机矩阵 (GOE) 的谱与向量分布性质、Taylor 展开与马尔可夫链谱间隙的变形不等式、以及对条件 Gibbs 分布的细致分析。

**📊 数据集**

本研究为理论分析，未使用任何具体实验数据集；所有结果均为概率论/随机矩阵理论证明。

**📈 对比分析**

与以往的路径耦合、log‑Sobolev、随机定位等方法相比，本文取得了在 β≈1/2+ 时的多项式混合时间（之前的上限多在 β≤1/4 或仅给出弱 Poincaré）。实验对比并未进行，主要通过理论推导证明了谱间隙 Ω(1/n)，从而得到 O(n²) 的混合时间。

**⚠️ 局限性**

限制与展望：① 常数 C 取值极小（≈5·10⁻⁵），未被进一步优化；② 结果仅适用于完整图与 GOE 随机交互矩阵，对非完全图或非高斯分布的扩展尚未完成；③ 对 β>1/2 的情况仍处于未解决状态，未来需要进一步改进 rank‑1 失真或结合其他技术。

---

## 503. Embodied-BenchForge: A Closed-Loop Agentic Workflow for Embodied Benchmark Construction

**arXiv ID:** 2609.13082 | [PDF](https://arxiv.org/pdf/2609.13082v1)

**作者:** Baoyang Jiang `[一作]` (QiYuan Lab), Qiang Ma `[通讯]` (QiYuan Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出并实现了 Embodied‑BenchForge 框架，能够根据用户的评估意图自动生成包含六个离线 EQA 任务集和一个包含 220 个可执行交互任务的交互式嵌入式轨道的完整基准。

**💡 创新点**

创新点在于提出闭环基准合成（Closed‑Loop Benchmark Synthesis），通过将前向工件合成与后向验证与修复相结合，利用类型化可复用技能、需求契约、证据依赖图和可追溯性实现工件级别的可靠构建与局部重执行。

**🔧 技术方法**

所用技术包括层次化技能库与分解、类型化工件和技能、结构化需求契约、证据与状态组织、基于 LLM 的文本与代码生成、以及基于 IRT 的性能诊断与可追溯性驱动的修复机制。

**📊 数据集**

构建基准使用了多源数据集，包括 ALFRED、Habitat、CARLA、LIBERO、AI2‑THOR、TartanGround 以及真实空中图像，覆盖机器人、车辆、臂架、UAV 和四足等多种嵌入式场景。

**📈 对比分析**

通过将 11 种模型与人类在两条轨道上进行对比实验，离线 EQA 轨道人类平均分 85.89 分，最佳模型 GPT‑5.5 仅 57.67 分；交互式轨道 GPT‑5.5 的任务成功率 83.18% 领先，但与人类相比仍存在显著差距；系统消融实验表明验证与修复是提升工件质量的关键环节。

**⚠️ 局限性**

局限性包括对 LLM 生成的依赖导致目标信号依赖性和评分一致性仍低；系统仍需手工指定评估意图与技能库维护；高 token 成本与复杂度；以及在真实硬件或未预定义环境下的可迁移性受限。

---

## 504. Groupoid-Based Internal State Representations for Reinforcement Learning with Local Symmetries

**arXiv ID:** 2609.13035 | [PDF](https://arxiv.org/pdf/2609.13035v1)

**作者:** Ben Opperman `[一作]` (University of London), Esther Mondragón `[通讯]` (University of London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用群组（groupoid）结构进行状态抽象的强化学习框架，在局部对称性环境中实现经验共享和价值传播。

**💡 创新点**

首次将群组理论应用于RL，允许状态仅在局部可逆变换下等价，从而克服传统全局对称假设的局限。

**🔧 技术方法**

使用群组（groupoid）对称性、轨道代表、传输器、规范化映射以及基于Q‑学习的更新算法；在离散网格环境中通过局部窗口的指纹实现状态匹配。

**📊 数据集**

在自定义的离散网格世界中，采用不同尺寸（20×20到200×200）和不同障碍/高成本密度（稀疏5%不可通行、密集10%不可通行+45%高成本）的场景。

**📈 对比分析**

与传统表格Q‑学习做对比；在高熵、密集和大规模网格中，群组框架显著降低有效状态空间（如40,000→1,350轨道），加速收敛并提升最终性能；在极稀疏环境中收敛稍慢，表现不如全局方法。

**⚠️ 局限性**

额外的状态匹配与传输器管理导致计算和内存开销增加；在结构多样性低的环境中过度泛化，导致学习精度下降；对真实复杂环境的可扩展性和工程实现仍需进一步优化。

---

## 505. CanvasAnneal: Curriculum Reinforcement Learning for Diffusion Language Models

**arXiv ID:** 2609.13060 | [PDF](https://arxiv.org/pdf/2609.13060v1)

**作者:** Blake Olson `[一作]` (Google DeepMind), Yuan Shangguan `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CanvasAnneal，一种通过教师推理轨迹的动态后缀遮罩课程化 RL 框架，改进离散扩散语言模型的并行生成。

**💡 创新点**

创新点在于：①使用离线教师推理作为初始 canvas 并逐步退化；②采用动态 Beta 课程化遮罩和固定比例的纯噪声组来保持优势计算的公平；③保持推理轨迹连续性而非随机遮罩，天然形成推理难度递增的课程。

**🔧 技术方法**

技术手段包括：离线教师推理（Gemini 3.1 Pro）、CanvasAnneal 课程化 RL、diffu-GRPO（组相对策略优化）、Beta 分布动态调度、后缀遮罩、组内共享初始 canvas、PPO 风格截断与 KL 正则。

**📊 数据集**

使用的数据集：GSM8K、MATH500、Countdown、xLAM、BFCL、Tau2。

**📈 对比分析**

与 LLaDA 基线、SFT、diffu-GRPO 以及 SFT+RL 进行对比；在 MATH500、Countdown、Tau2 及 BFCL 部分任务中取得提升，尤其在 xLAM、Countdown 的奖励曲线收敛速度更快；在 GSM8K 上表现不如标准 diffu-GRPO，说明提升具有任务依赖性。

**⚠️ 局限性**

局限性包括：①对不同任务的收益不均衡，GSM8K 等任务提升有限；②需要离线生成大量教师推理轨迹，成本较高；③课程化调度参数需要任务调优；④纯噪声组比例对性能影响显著，需精细平衡；⑤在推理时仍需从全遮罩开始，无法利用教师知识加速生成。

---

## 506. Expert-Space Exploration in MoE Reinforcement Learning

**arXiv ID:** 2609.13058 | [PDF](https://arxiv.org/pdf/2609.13058v1)

**作者:** Hongyi He `[一作]` (Tsinghua University), Yeyun Gong `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型的后训练强化学习中，本文通过对Mixture-of-Experts（MoE）模型的专家路由进行噪声扰动来实现专家空间探索（ESRL），从而增强生成多样性和奖励信号。

**💡 创新点**

创新点包括：① 将专家路由视为可探索的离散空间，并通过熵自适应噪声、锚定专家采样以及候选池限制实现可控的路由扰动；② 引入路由重放机制，确保在训练时使用与推理时相同的专家路径；③ 在不修改奖励或优化目标的情况下，将路由探索作为纯推理层的扩展，形成与传统 token‑级采样互补的探索方式。

**🔧 技术方法**

技术方法包括：熵自适应噪声调度（根据路由熵决定噪声尺度）；锚定专家采样（保留 Top‑K 高置信专家，剩余位置从候选池中加入噪声后采样）；路由重放（记录推理时的专家路径并在策略优化时重放）；以及基于 Group Relative Policy Optimization（GRPO）的训练框架。

**📊 数据集**

使用的主要数据集为数学推理的 OlympiadBench、AIME‑2024、AMC、MinervaMath；科学推理与代码生成任务分别使用 GPQA、MMLU‑Pro、MMLU‑Redux、LiveCodeBench v6；训练集为 Math 数据集（用于奖励）。

**📈 对比分析**

与 GRPO、GSPO、GRPO‑R3、Aux‑Loss、N‑Sampling、RO‑GRPO 等基线相比，ESRL 在 Qwen3‑30B‑A3B、Sigma、Moonlight 三种 MoE 架构上均取得最高 Pass@1/Pass@8，平均提升约 3–5 个百分点；在科学与代码任务上也保持领先，并在不同采样温度、组大小、噪声尺度和专家数设置下表现稳健。

**⚠️ 局限性**

局限性包括：① 对噪声尺度、锚定专家数、候选池大小等超参数依赖较大，需要手工调优；② 仅在 MoE LLM 的后训练阶段验证，尚未探讨与前置对齐或更大规模模型的交互；③ 路由重放增加了训练时的存储和计算开销；④ 目前主要针对可检查奖励任务，未知在对齐/安全等领域的适用性。

---

## 507. Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model

**arXiv ID:** 2609.13053 | [PDF](https://arxiv.org/pdf/2609.13053v1)

**作者:** Hoeun Lee `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了 Dynin‑Robotics，一种基于 Dynin‑Omni 的统一 Vision‑Language‑Action 模型，利用掩码扩散将语言、视觉、动作等信息映射到共享的离散序列中，支持策略规划、世界建模、任务理解与终点预测等多种目标；同时实现了块并行解码与强化学习调度的高效推理；在多任务机器人学习中实现了可组合的推理路径；并通过连续动作离散化与可调节的辅助目标，提升了多样化任务下的泛化性能。

**💡 创新点**

创新点包括：1）将机器人控制、视觉预测和任务理解统一到同一掩码扩散框架中，形成可复用的多目标接口；2）提出基于目标掩码的可组合推理（如目标引导、联合解码、候选重排序）以平衡计算成本与性能；3）设计了基于块并行解码与 dInfer 的加速实现，使动作生成速度提升近30×；4）引入强化学习控制的掩码重置策略，在保持模型不变的前提下自适应推理步骤。

**🔧 技术方法**

主要技术包括：离散化掩码扩散（MDM）、共享双向 Transformer、动作量化与离散化、块并行（block‑parallel）解码、dInfer 加速堆栈、基于熵的自适应重掩码控制、强化学习（PPO）调度器、以及多任务目标混合训练。

**📊 数据集**

使用的数据集主要包括：① 48 个 Open X‑Embodiment (OXE) 轨迹数据（1,332,985 条轨迹）做持续预训练；② LIBERO 四套标准任务数据；③ LIBERO‑Plus 7 种扰动数据集；④ VLABench 两个任务（SelectFruit、InsertFlower）；⑤ DROID 验证集；⑥ Franka Research 3 机器人在四个工作区的真实实验数据。

**📈 对比分析**

对比方法涵盖：X‑VLA、ABot‑M0、OpenVLA‑OFT、RIPT‑VLA、Cosmos Policy 等；在 LIBERO 上取得 98.1% 宏平均成功率，接近 ABot‑M0；在零射 LIBERO‑Plus 上实现 73% 宏平均，优于 OpenVLA‑OFT 与 RIPT‑VLA；在 VLABench 两任务中，Dynin‑Robotics 超越 Mimic‑Video 并保持竞争力；在 Franka 实验中宏平均 78.4%，领先 Cosom Policy，尤其在语言指定堆叠顺序时表现最佳。

**⚠️ 局限性**

局限性包括：1）对长周期任务（Long）和受扰动的摄像头/机器人条件下的性能仍有明显下降；2）在世界建模与终点预测的图像质量低于简单帧复制，尤其在长远预测上；3）动作离散化的精细度对性能影响非单调，需要细粒度调优；4）推理时多目标组合虽能提升成功率，但也会显著降低吞吐量；5）当前未对多模态传感器（力/扭矩、触觉）进行训练与评估，未来工作需要扩展。

---

## 508. Continue, Adapt, or Yield: In-Turn Adaptation to Overlapping Speech in Full-Duplex Agents

**arXiv ID:** 2609.13117 | [PDF](https://arxiv.org/pdf/2609.13117v1)

**作者:** Yunqi Lu `[一作]` (Besimple AI), Yi Zhong `[通讯]` (Besimple AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Duplex Cue 框架，对全双工语音代理的在对话中即时适应进行评估。

**💡 创新点**

创新点在于将听者意图（回声、协作、打断）与说话者反应（继续、适配、让步）分离，形成二维评估矩阵。

**🔧 技术方法**

使用条件对话延续技术，基于 PersonaPlex 生成对已记录对话的续篇，并结合 ElevenLabs 语音转换和 Codex 自动标注。

**📊 数据集**

利用 20.32 小时的无脚本英语双声道对话，构建 2,591 条提示清单，挑选 300 条人类确认的交互对。

**📈 对比分析**

通过在同一提示下对比录制人类反应与模型生成的反应，发现协作提示下人类适配率 68.2% 而模型仅 34.8%，回声保持一致，打断则模型更倾向让步，整体表现表明模型在即时适应方面有明显不足。

**⚠️ 局限性**

局限性包括标注时未完全盲审导致意图与反应相关联、仅评估单一模型与单一语言、覆盖率不足、缺乏交互式对话和多语言验证。

---

## 509. Extreme-Scale Linear-Scaling Kohn-Sham DFT at 100 Million Atoms: Bridging Quantum Simulations and Experiments

**arXiv ID:** 2609.13115 | [PDF](https://arxiv.org/pdf/2609.13115v1)

**作者:** Qimen Xu `[一作]` (National Supercomputing Center in Shenzhen), Yutong Lu `[通讯]` (National Supercomputing Center in Shenzhen)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了 100M 及 200M 原子硅晶体和 11M 原子 Li/LGPS 固态电池界面在 Kohn‑Sham DFT 计算中的极大规模模拟，首次把 DFT 的可计算长度尺度提升至实验可观测的几十纳米范围。

**💡 创新点**

创新点在于：① 将全局密度矩阵按划分与缓冲区进行 D&C（Divide‑and‑Conquer）分解；② 用 Chebyshev‑filtered subspace iteration（CheFSI）高效求解每个子系统；③ 在 ARM LX2 处理器上通过 SVE/SME 向量化、定制稠密线性代数、结构化 AMG 预条件 Poisson 求解器等实现了线性（O(N)）计算和内存复杂度，并在 Exascale 计算平台上实现了 157.9 Pflop/s 的 FP64 持续性能与 96.6% 的弱标度效率。

**🔧 技术方法**

技术包括：基于密度矩阵的近视性原理、D&C 分域、CheFSI 子空间迭代、结构化 AMG + CG Poisson 求解、ARM SVE/SME 向量化、HBM 内存调度、FP16 轨道预热、分层混合精度、周期 Pulay + 物质掩码预处理等。

**📊 数据集**

使用的系统为：大尺寸硅晶体（100M、200M 原子）和 Li/LGPS 固态电池界面（约 11M 原子，尺寸 57×44×90 nm），界面结构通过机器学习势能场 MD 生成并在 300 K 下热平衡。

**📈 对比分析**

相较于以往的 O(N³) DFT（如 RSDFT、PARSEC 等）以及前一记录 10M 原子 LS3DF，XLSDFT 在 100M 原子上实现 157.9 Pflop/s、96.6% 的弱标度效率，强标度效率 81.3%，并在同一 20 480 节点上完成 200M 原子硅晶体计算，显著突破了先前 10M 原子规模的记录。

**⚠️ 局限性**

限制主要包括：仅实现了基态 Kohn‑Sham DFT（不含激发态或强关联效应）、对极大规模计算依赖 Exascale 专用硬件（LX2、HBM、SVE/SME），对非均匀或强磁性体系的适用性尚未验证；D&C 需要适当的缓冲区，可能在高度非均匀或高度交互的体系中产生误差；算法在多态材料间的泛化与对更复杂材料（多原子、磁性、含 3d/4f 元素）仍需进一步测试。

---

## 510. ASTRIL-MPC: Autonomous Traversal Framework of Articulated Tracked Robots with Language-Guided Neural-Kinematic MPC

**arXiv ID:** 2609.13083 | [PDF](https://arxiv.org/pdf/2609.13083v1)

**作者:** Zhenfeng Gan `[一作]`, Xueqian Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在城市搜救环境下，提出了一种结合神经动力学模型、模型预测控制（MPC）和大型语言模型（LLM）调参的自适应轨迹规划框架 ASTRIL‑MPC；

**💡 创新点**

创新点在于将学习到的轨迹预测模型嵌入 MPC 约束，利用 LLM 仅对有限参数进行安全约束下的在线调节，既保留了 MPC 的硬约束优势，又实现了快速、可解释的自适应；

**🔧 技术方法**

主要技术包括：基于局部高度图的神经 kinematics 预测器、离散时间多目标 NMPC、LLM 生成参数更新并通过安全门限校正、以及将神经模型编译为求解器友好的代码；

**📊 数据集**

使用自研的 ATR 仿真平台生成的 5k 条高度图轨迹数据，涵盖楼梯上/下、单块跨越等三种场景；

**📈 对比分析**

在三种任务上与 PPO 基线及未调参的 NMPC 进行对比，平均遍历质量得分提升 67%，在楼梯下降场景提升 71%，并成功消除碰撞冲击；

**⚠️ 局限性**

局限性包括：对高质量地形高度图和预先训练的神经模型依赖较大；LLM 的输出仍需严格安全校验，且当前仅在仿真环境验证，真实硬件表现及极端不规则地形的适应性仍待进一步评估。

---

## 511. NFT-Based Reward Mechanisms: Sybil Farming, Vesting, and Stochastic Verification

**arXiv ID:** 2609.13064 | [PDF](https://arxiv.org/pdf/2609.13064v1)

**作者:** Marco Alberto Javarone `[一作]` (Exponential Science Foundation), Carmine Ventre `[通讯]` (King's College London)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了基于 NFT 的奖励机制，探讨如何通过延迟兑现、随机审计与处罚来抑制用户多账户欺诈；

**💡 创新点**

提出了聚类检测下的有限最优攻击规模与新的阻止条件，并系统分析了延迟、审计率与处罚之间的相互作用；

**🔧 技术方法**

使用概率论与最优控制模型（连续松弛、整数约束）以及成本函数的凸性分析；

**📊 数据集**

未使用具体数据集，所有结果均为理论推导与数值示例；

**📈 对比分析**

通过数值仿真比较单纯延迟与延迟+审计方案，发现结合方案将成本降低约一半，且对审计率变化更为稳健；

**⚠️ 局限性**

主要局限在于聚类检测假设过强、农民风险中性、奖励 R 取值固定、仅考虑单一攻击者，未考虑多农民竞争及奖励可转移性等现实因素。

---

## 512. Diffusion Models and Concept Formation

**arXiv ID:** 2609.13047 | [PDF](https://arxiv.org/pdf/2609.13047v1)

**作者:** Zekun Wang `[一作]` (Georgia Institute of Technology), Christopher J. MacLellan `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将扩散模型与Cobweb概念形成模型对齐，证明扩散模型隐含的层次结构与Cobweb树相同，并在该层次上定位基本层级

**💡 创新点**

将扩散模型的噪声水平映射为层次深度，提出基于分数寻找模式的方式恢复隐含层次，并用该层次的“特异性”度量定位基本概念

**🔧 技术方法**

利用扩散模型的分数（score）与Tweedie公式进行模式搜索、类别效用（category utility）与信息论度量、Gaussian混合与层次贝叶斯先验

**📊 数据集**

在MNIST与Fashion‑MNIST这两个10类灰度图像数据集上进行实验

**📈 对比分析**

通过对比Cobweb学习的显式树和扩散模型恢复的隐式树，以及两者在“特异性”峰值处的层级（基本层级），两种模型在层级结构和基本层级位置上基本一致；实验结果表明扩散模型在中间噪声水平产生与Cobweb相同的基本概念

**⚠️ 局限性**

扩散模型的时间步长与抽象层级不匹配，噪声调度为样本质量优化而非认知可解释，导致层次索引不均匀；此外缺乏直接的显式树结构，导致可解释性受限

---

## 513. MAxBench: A Multinomial Concept Recovery Benchmark

**arXiv ID:** 2609.13072 | [PDF](https://arxiv.org/pdf/2609.13072v1)

**作者:** Divya Appapogu `[一作]` (Boston University), Aaron Mueller `[通讯]` (Boston University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出MAxBench框架，对多元（非二元）概念的表示进行几何不依赖的评估，并比较10种概念定位方法的蒸馏效果。

**💡 创新点**

创新点在于：①构建可对任何几何形式的概念表示进行采样、干预并评价的通用评测流程；②将多维子空间（尤其是仿射子空间）作为更适合多类别概念的表示；③首次系统比较多种线性、仿射和非线性几何与提示的对比。

**🔧 技术方法**

技术手段包括：采样式干预（在激活空间中随机采样并替换/加权），多种概念定位方法（DiffMean‑r1、ReFT‑r1、DiffMean（pairs）、PCA、Factor Analysis、Linear Probe、Schatten Probe、MFA、SAE、Spline fitting），以及基于LLM判别器的概念、流畅性、指令遵循与多样性评估。

**📊 数据集**

使用6个概念（Animals、Countries、Vehicles、Plants、Days、Years）的人工生成句子数据集，配合AlpacaEval提示进行干预评估；在Gemma、Llama、Qwen三大模型的中间层提取激活。

**📈 对比分析**

对比方法时用Concept、Fluency、Instruction、Diversity、AxBench和综合S分数；结果显示仿射子空间在多数模型/概念上优于单向量或线性子空间，提示在大模型上仍占优；非线性曲线方法在Days概念上可与最佳方法竞争。

**⚠️ 局限性**

局限性包括：依赖LLM判别器的自动评估，可能与人工判断不完全一致；实验仅在单一干预层进行，未探讨层次差异；采样超参数需调优；当前评测主要聚焦仿射几何，缺乏更丰富的非线性表示。

---

## 514. Benign Loss Landscapes Can Coexist with Worst-Case Hardness

**arXiv ID:** 2609.13057 | [PDF](https://arxiv.org/pdf/2609.13057v1)

**作者:** Zach Furman `[一作]`, Liam Hodgkinson `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

绘制并分析了一个包含多分支的图结构

**💡 创新点**

通过标记边与角度信息细化结构表示

**🔧 技术方法**

使用 TikZ 进行图形绘制

**📊 数据集**

未提及具体数据集

**📈 对比分析**

未给出比较方法与性能评估

**⚠️ 局限性**

缺乏实验验证和性能对比

---

## 515. Kraken: LLM-based Speech-to-Speech Translation via Low-bitrate VQ and Dual-path Source Conditioning

**arXiv ID:** 2609.13045 | [PDF](https://arxiv.org/pdf/2609.13045v1)

**作者:** Hayato Futami `[一作]` (Sony Group Corporation), Emiru Tsunoo `[通讯]` (Sony Group Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种端到端的语音到语音翻译模型Kraken，利用低比特率VQ token和GAN式语音解码器实现高质量、语者身份与语调的保留。

**💡 创新点**

创新点包括：1）单层低比特率VQ编码（325 bps），显著降低LLM预测成本；2）双路径源条件（LLM与Autowave‑X均以源语音为条件），提升语者与语调迁移；3）采用自监督特征驱动的单层VQ与GAN解码器组合。

**🔧 技术方法**

使用技术：W2v‑BERT 2.0 语音编码器、Qwen3‑8B LLM、单层VQ‑Codec（8192码本）、GAN‑based Autowave‑X vocoder、iterative refiner、chain‑of‑thought 提示、并结合多任务（ASR、S2TT、S2ST、MT、TTS）训练。

**📊 数据集**

训练数据集包括：150k 小时多语种/多任务语音数据（CommonVoice、Voxpopuli、MLS、CSJ、VCTK、LibriTTS 等），以及合成的S2ST/MT数据和对齐语音数据。

**📈 对比分析**

在FLEURS X‑En和CVSS X‑En数据集上，Kraken在ASR‑BLEU、UTMOS、Speaker/Emotion/Prosody相似度等指标上均优于 SeamlessM4T‑Large‑v2、Qwen2.5‑Omni，甚至在多数语言上超过 Qwen3‑Omni；人类评测显示其在语调自然度、语者匹配、情感匹配上排名第一。

**⚠️ 局限性**

局限性包括：仅支持10种主要语言的 X‑En 翻译，非实时（离线）模式；对低资源口音、方言的适应不足；模型未公开，且存在潜在语音克隆/深度伪造风险。

---

## 516. Robust Policy Optimization via Adversarial Importance Sampling

**arXiv ID:** 2609.13044 | [PDF](https://arxiv.org/pdf/2609.13044v1)

**作者:** Amine Andam `[一作]` (Mohammed VI Polytechnic University), Mustapha Hedabou `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

提出了一种基于重要性采样与凸松弛的深度强化学习鲁棒训练方法Advis，并开发了模块化 PyTorch 库 advrl 供快速原型和大规模评估使用。

**💡 创新点**

创新点：①无需额外采样、无辅助网络即可估计并优化最坏情况回报；②结合重要性采样与凸松弛实现长期鲁棒性；③对学习型对手进行大规模、多配置评估，揭示传统评估易导致鲁棒性高估。

**🔧 技术方法**

技术：重要性采样（PDIS/WIS/WPDIS）、凸松弛（IBP/线性松弛）、PPO、对手生成（SA-ATLA/PA-ATLA/RS）、多配置对手搜索、CVaR 评价、Jaccard 相似度分析。

**📊 数据集**

数据集：MuJoCo 连续控制环境（HalfCheetah-v5、Hopper-v5、Walker2d-v5）和相应的对手配置集（432–1296 个配置）。

**📈 对比分析**

与 SA-Reg、Radial、SA-ATLA、PA-ATLA、WocaR-RL 等基线进行对比；使用最差回报、CVaR_α 等指标评估鲁棒性；实验显示 Advis 在绝大多数环境与 α 范围内均优于所有基线，尤其在极端尾部风险下表现更突出。

**⚠️ 局限性**

局限：仅针对连续控制的策略优化方法；实验规模受限于 2M 步训练；尚未验证对其他 RL 领域（离散动作、模仿学习等）的适用性；对手搜索依赖网格搜索，计算成本仍较高。

---

## 517. DynSHAP: Towards Explainable Dynamic Survival Analysis

**arXiv ID:** 2609.13042 | [PDF](https://arxiv.org/pdf/2609.13042v1)

**作者:** Nastasya Anokhina `[一作]` (University of Cambridge), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种专为动态生存分析（DSA）设计的解释框架 DynSHAP，能够对时间-特征对进行SHAP值分解并生成时间敏感的特征重要性。

**💡 创新点**

创新点在于：①将时间维度离散化，将时间-特征对视为Shapley游戏中的玩家；②引入 Temporal DynSHAP，通过学习线性时间条件分布进行条件采样，克服传统边际采样在时间相关数据中的不现实性；③提供开源 Python 包，方便医学实践者使用。

**🔧 技术方法**

使用的技术包括：SHAP 的 Monte Carlo Sampling、Kernel Estimator、线性高斯时间条件模型（条件采样）、对生存函数在各预测时点独立计算 SHAP、以及对模型输出的多时间点处理。

**📊 数据集**

数据集包括：两个真实医学数据集（PBC、MS），以及用于验证 Ground Truth 的合成长期生存数据；模型则为 Dynamic-DeepHit 和 DySurv 两种主流 DSA 架构。

**📈 对比分析**

与基线（随机、标记 SHAP 等）比较：Temporal DynSHAP 在合成数据上相较边际估计器的 RMSE/MAE 下降约 2 倍；在真实数据上，边际估计器往往将重要性集中在最近一次访视，而 Temporal DynSHAP 能够回归更合理的历史重要性；实验表明 Temporal DynSHAP 在信度测试中表现更好，但在某些模型（如 DDH）上稳定性略逊。

**⚠️ 局限性**

限制主要在于：① 线性高斯条件假设可能不适用于非线性或长程依赖；② 条件采样会引入假相关性，导致某些场景下解释不够精确；③ 在稀疏数据集（如 MS）上性能噪声较大；④ 需要进一步验证对临床专家判断的吻合度。

---

## 518. IntentFuzz: A Protocol-Aware Fuzzer for Automated Invariant Violation Detection in Intent-Based Cross-Chain Bridges

**arXiv ID:** 2609.13004 | [PDF](https://arxiv.org/pdf/2609.13004v1)

**作者:** André Augusto `[一作]` (University of Lisbon), Miguel Correia `[通讯]` (University of Lisbon)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了一款协议感知的 fuzzer，能够自动从未标注的 Solidity 代码中恢复意图桥的结构与功能角色，生成多步骤 fuzz 测试序列，检测链上 invariant 违规与 settlement 暴露。

**💡 创新点**

创新点包括：① 将 invariant 违规与 settlement 暴露分离的六类 taxonomy；② 规则驱动的结构与功能角色自动识别；③ 结合跨步骤绑定与双层 tamper 的多步骤 fuzz 计划；④ LLM 辅助的输入恢复机制；⑤ 在真实部署上实现 100% 召回、82% 组合精度，并发现 22 条可复现漏洞。

**🔧 技术方法**

主要技术：静态分析框架 Slither 进行 AST/IR 提取和规则匹配；GPT‑4o mini LLM 用于恢复失败调用的参数；Anvil fork 与 EVM 交互执行模板；guard‑JUMPI 覆盖率监测；三层输入生成层（会话意图、边界采样、LLM 恢复）。

**📊 数据集**

数据集：9 个基准协议（5 真实桥 + 4 合成桥）用于结构/函数验证；77 个手工标注的 GitHub 意图桥合约用于语义与桥分类评估；24 条真实链上部署用于动态测试；23 个植入缺陷的 mutants 用于 mutation 测试。

**📈 对比分析**

评估方式：与基准合约的 mutation 结果对比，获得 100% recall/precision；在 77 合约上取得 79.5% 桥分类精度、97.2% 召回、88.6% 结构选择精度；LLM 辅助后，step‑0 成功率从 54.4% 提升至 85.0%，漏洞发现从 17 条提升至 22 条；性能上，静态提取平均 0.47 s/合约，完整流程 0.88 s/合约；真实链上执行平均 26.4 s/部署（含 1.7 s/模板）。

**⚠️ 局限性**

局限性：只能检测链上 invariant 违规，无法覆盖 settlement 侧的违规；需要可 fork 的链状态，无法处理缺失的初始化或外部消息；LLM 只在 revert 信息可见时恢复参数，无法推断协议专用的哈希或多字段填充；对访问控制门限的调用无法通过 fuzz 触发；覆盖率提升受限于缺失的执行路径。

---

## 519. Rethinking Heterogeneous System Disaggregation for Subquadratic Attention

**arXiv ID:** 2609.13134 | [PDF](https://arxiv.org/pdf/2609.13134v1)

**作者:** Arya Tschand `[一作]` (Harvard University), Karu Sankaralingam `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于子二次注意力算术强度和内存占用的细粒度异构系统拆分方案(SubQuadratic Disaggregation)，并在异构GPU+SRAM体系上实现LLM推理加速。

**💡 创新点**

创新点在于：①将解码拆分为二次与子二次注意力层而非单纯按算子类型拆分；②将二次注意力与预填充共置于DRAM GPU，子二次注意力和FFN迁移至SRAM ASIC；③设计IndexShare顶K缓存与预取机制，隐藏跨设备通信；④通过实测与解析模型验证能在功耗约为传统方案57%时提升31-56% tokens/J及1.2-1.5×吞吐。

**🔧 技术方法**

采用异构系统拆分技术、子二次注意力模型（稀疏、线性、滑窗）、IndexShare缓存与预取、单向Put+doorbell通信、8×B200 GPU代理实验、Rubin+LPX解析模型。

**📊 数据集**

使用的模型包括GLM‑5.2、Nemotron 3 Ultra、Gemma 4 31B；工作负载覆盖Chat(32K上下文)、RAG(256K)、Agentic(1M)等长上下文推理场景。

**📈 对比分析**

与无拆分、Prefill/Decode拆分、Attention‑FFN拆分以及EAGLE‑3投机解码等基线对比。关键指标为tokens/J、能耗、每用户TPS，实验表明SQD在8×B200代理上实现32‑56% tokens/J提升、57%功耗下降，并在Rubin+LPX模型中在固定功耗预算下提升1.2‑1.5×吞吐。

**⚠️ 局限性**

局限性包括：仅针对推理阶段；需要专用SRAM ASIC与低延迟跨设备互连；评估模型和实验覆盖的LLM和工作负载有限；对训练不适用；实现复杂度高且依赖硬件平台的特定特性。

---

## 520. A Dichotomy for Boolean Complex Holant Problems with Conjugate-Closed Signature Sets

**arXiv ID:** 2609.13132 | [PDF](https://arxiv.org/pdf/2609.13132v1)

**作者:** Jincheng Guan `[一作]` (University of Science and Technology of China), Zhuxiao Tang `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对六元签名f和相关的Holant问题进行复杂度分类，证明若不满足某些条件则问题为#P难，或者能通过实正交变换实现特定签名f6和四个贝尔签名。

**💡 创新点**

结合绝对最大纠缠态的正则形式与二元群的几何结构，给出新的复杂度分裂与统一的证明方法。

**🔧 技术方法**

使用全局对称性变换、张量分解、群论与多重对角化技巧，构造哈洛格拉夫变换和合成签名。

**📊 数据集**

该研究为理论推导，无使用实验数据集。

**📈 对比分析**

通过对已知难度与可实现性进行严格归纳，对比不同群的可实现性，最终证明在满足条件时问题为#P难；在不满足时能显式构造可实现签名。

**⚠️ 局限性**

结果仅适用于偶数阶签名且不满足特定外部条件，且对高维通用情形尚未覆盖。

---

## 521. Beyond Establishing the Four-Day Workweek: Understanding Adaptation and Long-Term Survival in an Agile Software Organization

**arXiv ID:** 2609.13089 | [PDF](https://arxiv.org/pdf/2609.13089v1)

**作者:** Michael Neumann `[一作]` (University of Applied Sciences & Arts Hannover), Darja Šmite `[通讯]` (Blekinge Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对一家敏捷软件公司进行纵向单案例研究，分析其在四天工作周（4DWW）实施、适应、制度化以及在外部压力下的存续过程。

**💡 创新点**

提出4DWW生命周期模型和4DWW存续矩阵，阐明4DWW是一个动态组织安排而非一次性干预，并揭示员工通过保护性适应机制维持4DWW的方式。

**🔧 技术方法**

采用定性内容分析（Mayring方法）进行访谈文本编码，并辅以大型语言模型（Claude、ChatGPT等）进行一致性检查。

**📊 数据集**

使用15名员工在2022年和2026年两轮半结构化访谈的录音稿（共15次访谈）。

**📈 对比分析**

不涉及传统数值比较；通过对两轮访谈主题的编码对比，观察4DWW随时间的演变与适应，未能提供量化性能指标，但提供了丰富的质性洞见。

**⚠️ 局限性**

仅在单一组织开展，样本量有限，访谈依赖受访者回忆，缺乏跨案例验证，外部效度受限。

---

## 522. Physics-Aware Video Generation via Agentic Planning and Graph-Guided Optimization

**arXiv ID:** 2609.13006 | [PDF](https://arxiv.org/pdf/2609.13006v1)

**作者:** Minh-Loi Nguyen `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种无训练、基于VLM的物理意识视频生成框架PhysPlan，利用两阶段代理式规划与训练‑free图引导优化来生成符合物理规律的视频。

**💡 创新点**

创新点在于：① 通过Chain‑of‑Visual‑Thought实现物理事件的离散规划；② 引入Object‑Centric Gradient Routing锁定背景并局部梯度；③ Kinetic Intensity Profiling依据物理强度动态调节优化超参，实现对复杂状态变化的自适应控制。

**🔧 技术方法**

采用Gemini等大型多模态语言模型进行思路规划，Grounded‑SAM‑2与DepthAnythingV2做遮罩与深度提取，结合Frame Guidance的训练‑free优化、Latent Slicing和Tweedie公式实现高效推理。

**📊 数据集**

在PhyGenBench（27种物理现象）和Physics‑IQ（396个真实场景）两大基准上进行评估。

**📈 对比分析**

与基础的CogVideoX、SVD‑XT、LTX‑Video及最新的Frame Guidance等模型对比，PhysPlan在物理合理性、结构完整性和时序一致性上均超过基线；PhyGenBench平均分0.59、Physics‑IQ平均分28.1，并在FID/FVD等感知指标上保持或提升。

**⚠️ 局限性**

限制在于推理时需多次梯度反向和VLM交互导致显著延迟和显存占用；关键帧稀疏约束可能抑制自然运动，且依赖云端API，难以在长视频或资源受限环境下扩展。

---

## 523. SNAP3D: Physically Grounded 3D Parts for Assembly from a Single Image

**arXiv ID:** 2609.13146 | [PDF](https://arxiv.org/pdf/2609.13146v1)

**作者:** Yu-Rou Tuan `[一作]` (Carnegie Mellon University), Xiaoxuan Ma `[通讯]` (Carnegie Mellon University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种物理引导的三阶段框架，用单张图像生成可组装、稳定的3D部件；

**💡 创新点**

创新点在于同时解决部件间重叠、恢复接触关系并使用物理反馈优化连接器，从而实现物理可实现的装配；

**🔧 技术方法**

使用了几何编辑、接触图推理、参数化连接器、增量潜在接触物理仿真与交叉熵方法优化；

**📊 数据集**

在HY3D‑Bench数据集上进行实验，使用来自XPart、OmniPart和PartCrafter的部件生成结果；

**📈 对比分析**

与现有生成器比较，物理稳定率从≈0%提升至95%，跌落率降至3%，平移误差下降618倍；几何指标基本相当；

**⚠️ 局限性**

局限在于仅处理刚体物理、求解成本高、对输入网格质量敏感、无法处理薄壁或开放壳体导致残留重叠。

---

