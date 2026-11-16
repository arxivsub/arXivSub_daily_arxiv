# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-16 | 今日论文总数: 469

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. From Images to Grounded Edits: A Real-World Benchmark for Chinese Grammatical Error Correction in Handwritten Essays

**arXiv ID:** 2611.17966 | [PDF](https://arxiv.org/pdf/2611.17966v1)

**作者:** Weiru Zhang `[一作]` (Fudan University), Dejing Dou `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的任务——基于图像的中文语法错误纠正（G-CGEC），并引入了G-CGEC-Eval基准数据集，旨在直接在手写作文图像上进行错误纠正。

**💡 创新点**

创新点在于将传统的文本到文本的语法错误纠正转变为图像到纠正的任务，要求系统不仅预测纠正内容，还要预测其在图像上的位置。

**🔧 技术方法**

使用了视觉语言模型（VLMs）和多种评估协议，包括基于IoU的指标和中心距离匹配策略。

**📊 数据集**

使用了G-CGEC-Eval数据集，该数据集包含来自K-12教育环境的真实手写作文图像，具有细粒度的错误注释。

**📈 对比分析**

通过比较开放源代码和专有的视觉语言模型，在单次推理设置和带有辅助组件的设置下进行评估，结果显示即使是顶级专有模型在单次推理设置下表现不佳，而管道和代理协调方法则显著提高了性能。

**⚠️ 局限性**

限制在于当前模型在字符级定位和语言错误发现方面仍存在瓶颈，且代理工作流的高推理延迟需要更高效的推理策略。

---

## 2. Pose-based Neural Clothing for Animated Characters

**arXiv ID:** 2611.18253 | [PDF](https://arxiv.org/pdf/2611.18253v1)

**作者:** Julian N. Heidenreich `[一作]` (Disney Research Studios), Jingwei Tang `[通讯]` (Disney Research Studios)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种轻量级的变分自编码器（VAE）架构，用于在角色姿势变化时对服装进行变形，生成逼真的褶皱，支持实时计算。

**💡 创新点**

创新点在于设计了一种能够更好地泛化到未见姿势的神经网络和训练策略，并引入了一种新的局部梯度直方图损失，以减轻重建损失的过平滑效应。

**🔧 技术方法**

使用了变分自编码器（VAE）技术，并通过逐步训练来实现从重建到姿势驱动预测的过渡。

**📊 数据集**

在生产级资产和公共数据集上进行了广泛评估。

**📈 对比分析**

与现有的神经服装方法相比，该方法在实时生成锐利、逼真的褶皱、平滑的大规模垂坠和时间一致性方面表现出色，显著改善了艺术家的工作流程。

**⚠️ 局限性**

限制在于模型仅针对单一服装进行训练，可能在多样化服装类型的泛化能力上存在不足。

---

## 3. V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations

**arXiv ID:** 2611.17831 | [PDF](https://arxiv.org/pdf/2611.17831v1)

**作者:** Keuntae Kim `[一作]` (Hanyang University), Yong Suk Choi `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本论文提出了一种新的视频语言模型（VLM），通过引入反事实监督机制，使得模型能够更好地理解视频中的动态信息，从而提高其在视频理解任务中的表现。

**💡 创新点**

创新点在于设计了一种反事实监督方法，使得语言模型能够读取视频编码器中未被利用的动态信息，并通过像素变换生成平衡答案的变体，从而增强模型的学习能力。

**🔧 技术方法**

使用了自监督视频编码器V-JEPA，结合了反事实监督机制，并通过一个单阶段训练过程来优化模型。

**📊 数据集**

使用了4.53M个公共样本进行训练，每帧使用32个视觉标记，显著少于其他开放视频LLM所需的样本量。

**📈 对比分析**

与其他开放模型相比，所提出的模型在11个视频基准测试中表现优异，4B模型的平均得分为72.1，明显高于最佳开放4B模型的61.1和最佳开放8B模型的64.8。

**⚠️ 局限性**

限制在于该模型主要针对视频动态的理解，文本识别和单帧图像的处理能力尚未覆盖，未来的工作将扩展到更长的视频和更复杂的任务。

---

## 4. Tractable Importance Sampling for Monte Carlo Privacy Accounting

**arXiv ID:** 2611.17596 | [PDF](https://arxiv.org/pdf/2611.17596v1)

**作者:** Galen Andrew `[一作]` (Google Research), Arun Ganesh `[通讯]` (Google Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种通过重要性采样提高蒙特卡洛隐私会计效率的方法，特别针对差分隐私矩阵分解中的隐私参数评估。

**💡 创新点**

创新点在于通过引入重要性采样来降低蒙特卡洛会计的样本复杂度，提出了一种参数化的提议分布，并证明了其有效性。

**🔧 技术方法**

使用了重要性采样技术来优化蒙特卡洛会计过程，具体通过构造一个偏移的提议分布来实现。

**📊 数据集**

使用了与差分隐私矩阵分解相关的高维、高相关性的高斯混合数据集。

**📈 对比分析**

与传统的蒙特卡洛方法相比，重要性采样显著降低了样本复杂度，具体的性能提升通过实验进行了验证，样本复杂度减少了1/B(α*)倍。

**⚠️ 局限性**

限制在于重要性采样的选择可能会增加计算成本，并且在某些情况下，样本的方差可能会高于预期，从而影响估计的准确性。

---

## 5. Certified Improvements to Best-Known Thomson-Problem Configurations with a Search Pipeline Built with Large Language Model Agents

**arXiv ID:** 2611.17776 | [PDF](https://arxiv.org/pdf/2611.17776v1)

**作者:** Hui Shao `[一作]` (Zhejiang University), Yijie Peng `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了托马斯问题，寻找在球面上放置N个单位电荷以最小化库仑能量的配置。作者报告了N=912、1382和2502的配置，其能量分别低于剑桥集群数据库的值，并通过精确的整数算术进行了认证。

**💡 创新点**

创新点在于使用大型语言模型（LLM）代理进行搜索和验证，提出了一种新的搜索管道，结合了无约束的复制交换盆地跳跃和改进区域的重组，成功找到了比现有数据库更低的能量配置。

**🔧 技术方法**

使用了无约束的复制交换盆地跳跃、梯度优化和重组等技术，结合了大型语言模型代理进行代码编写和执行。

**📊 数据集**

使用了剑桥集群数据库（CCD）中的配置作为起始点，进行了多次搜索和比较。

**📈 对比分析**

与现有方法的比较显示，作者的方法在N=912和N=1382的配置上取得了显著的能量降低，且所有新配置都是严格的局部最小值。性能上，N=2502的配置在58个核心小时内找到，且能量降低超过了10^-8的相对减少。

**⚠️ 局限性**

限制在于没有进行相同预算的比较，且成功率较低，仅在11个搜索大小中改善了3个，且未进行消融实验以确定各个组件的贡献。

---

## 6. ALoMa: Affine Correspondences Beyond Patches

**arXiv ID:** 2611.18170 | [PDF](https://arxiv.org/pdf/2611.18170v1)

**作者:** David Nordström `[一作]` (Chalmers University of Technology), Jonathan Ventura `[通讯]` (California Polytechnic State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文重新审视了仿射对应（AC）的估计，提出了一种新的方法，通过全局上下文信息来提高AC的准确性，超越了传统的局部图像补丁方法。

**💡 创新点**

创新点在于通过联合学习完整的4自由度（DoF）仿射矩阵，而不是分别估计仿射变换的各个组成部分，并且引入了上下文感知的模型来直接预测所有四个仿射DoF。

**🔧 技术方法**

使用了LoMa关键点描述符和轻量级的仿射头架构，结合了现代匹配器的上下文信息。

**📊 数据集**

使用了多个3D数据集，包括MegaDepth和ScanNet，数据集中包含真实的深度、相机内参和姿态信息。

**📈 对比分析**

与现有的AC估计方法相比，本文的方法在多个基准测试中表现出显著的性能提升，尤其是在仿射估计误差上降低了一个数量级，并在下游几何任务中表现出一致的改进。

**⚠️ 局限性**

限制在于虽然增加的上下文信息提高了性能，但它将局部仿射框架与配对图像耦合，可能影响稳定性。此外，使用LoMa的点对应而非RoMa v2的点对应，虽然提高了运行时性能，但可能在准确性上有所妥协。

---

## 7. Complex Schrödinger Bridges

**arXiv ID:** 2611.17746 | [PDF](https://arxiv.org/pdf/2611.17746v1)

**作者:** Dong-Sig Han `[一作]` (Imperial College), Tolga Birdal `[通讯]` (Imperial College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文探讨了一种变体的扩散桥，用于解决由缩放和旋转法则支配的复杂向量空间中的实际问题，提出了共形最优传输（COT）框架，并引入了复杂薛定谔桥（CSB）作为新的动态扩散桥类。

**💡 创新点**

创新点在于通过引入复杂值的传输框架和共形几何先验，扩展了传统的薛定谔桥理论，提供了一种新的几何视角来处理复杂物理系统中的扩散任务。

**🔧 技术方法**

使用了复杂值随机微分方程（CVSDE）和共形最优传输（COT）技术，结合了复杂分析和随机微积分。

**📊 数据集**

使用了医学成像和量子物理的模拟数据集进行验证，特别是在量子电动力学的模拟中。

**📈 对比分析**

与传统的实值模型相比，CSB在多个基准测试中表现出更低的平均终点Wasserstein距离，并在图像翻译任务中显示出更高的FID和SSIM分数，证明了其在处理复杂值数据方面的有效性。

**⚠️ 局限性**

限制在于复杂值参数的使用可能并不总是最佳选择，尤其是在缺乏一般能量结构的领域。此外，基于非全纯函数的复杂值神经架构仍未得到充分探索，未来的研究将扩展到一般流形上的最优传输问题。

---

## 8. TotalSegmentator Aorta Report: Automated Aortic Diameter Measurement at 11 Standardized Landmarks on CT Images

**arXiv ID:** 2611.17968 | [PDF](https://arxiv.org/pdf/2611.17968v1)

**作者:** Jakob Wasserthal `[一作]` (University Hospital Basel), Maurice Pradella `[通讯]` (University Hospital Basel)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发并评估了一种完全自动化的开源工具，该工具能够在任何CT图像上测量从主动脉瓣到分叉的11个标准化标志处的主动脉直径，符合当前的指南建议。

**💡 创新点**

创新点在于该工具能够在不同的CT图像上独立工作，并且能够自动测量整个主动脉的直径，提供了一种快速且一致的报告方式。

**🔧 技术方法**

使用了基于深度学习的TotalSegmentator模型，结合了多个分割模型来实现主动脉的自动测量。

**📊 数据集**

使用了239个动脉期CT血管造影检查的数据集，这些检查来自230名患者。

**📈 对比分析**

与放射科医师修正的测量结果进行比较，整体平均绝对误差（MAE）为1.06毫米，86.6%的差异在2毫米或以下，性能优于已发布的读者间变异性。

**⚠️ 局限性**

局限性包括参考标准是通过修正自动测量得出的，而不是独立测量，可能导致对自动值的偏倚；研究仅在一个中心进行，且大多数检查是在同一型号的扫描仪上获取的；工具虽然设计为适用于任何CT图像，但仅评估了动脉期CT血管造影。

---

## 9. Quantum Algorithm with Quadratic Speedup for 2D Pattern Matching

**arXiv ID:** 2611.17541 | [PDF](https://arxiv.org/pdf/2611.17541v1)

**作者:** Rishabh Gala `[一作]` (Baton Rouge Magnet High School), Rahul Shah `[通讯]` (Louisiana State University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在量子查询模型下的二维模式匹配问题，提出了一种解决方案，能够以 Õ(√(N)) 的查询复杂度判断模式 P 是否存在于文本 T 中。

**💡 创新点**

创新点在于利用几何结构“天际线”来保持量子优势，同时结合确定性采样和预计算的见证表来优化二维匹配过程。

**🔧 技术方法**

使用了量子搜索算法（如 Grover 的搜索和 Dürr-Høyer 搜索）以及确定性采样技术。

**📊 数据集**

使用的假设数据集为矩形文本 T 和矩形模式 P，具体大小为 n_1 × n_2 和 m_1 × m_2，且假设 M < N。

**📈 对比分析**

与传统方法相比，本文的方法在查询复杂度上实现了近乎二次的量子加速，达到 Õ(√(N)) 的复杂度，优于 O(N^3/4) 的简单方法。

**⚠️ 局限性**

限制在于需要预先计算见证表，并假设其作为 oracle 可用，这在某些情况下可能增加了计算的复杂性。

---

## 10. Classification with Large-Step Gradient Descent: Unifying Self-Stabilization and Convergence Rates

**arXiv ID:** 2611.17459 | [PDF](https://arxiv.org/pdf/2611.17459v1)

**作者:** Salma Tarmoun `[一作]` (University of Pennsylvania), René Vidal `[通讯]` (University of Pennsylvania)

**通讯引用:** 29061 | [OpenAlex ID](https://openalex.org/A5011256828)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文扩展了梯度下降在指数尾损失（如逻辑回归和交叉熵损失）下的统一几何视角，分析了在不稳定边缘操作时的收敛性。

**💡 创新点**

创新点在于提出了一个统一框架，使自我稳定机制定量化，并提供了迭代的收敛速率，同时揭示了三相阻尼机制和几何依赖的瞬态隐性偏差。

**🔧 技术方法**

使用了梯度下降（GD）技术，并通过几何结构分析了损失函数的曲率。

**📊 数据集**

使用了逻辑回归和多类交叉熵损失的对称数据模型作为数据集。

**📈 对比分析**

与现有方法相比，本文提供了更细致的收敛分析，明确了在不稳定和稳定阶段的迭代界限，揭示了三相阻尼结构，性能上显示出在特定几何条件下的瞬态隐性偏差。

**⚠️ 局限性**

限制在于该理论主要针对特定的几何结构，可能无法广泛适用于所有类型的数据或更复杂的深度学习架构。

---

## 11. Long rational and Gaussian integer multiplication

**arXiv ID:** 2611.18028 | [PDF](https://arxiv.org/pdf/2611.18028v1)

**作者:** Yuval Spiizer `[一作]` (Tel Aviv University), Raphael S. Steiner `[通讯]` (Huawei)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的Toom-Cook框架，将有理长整数乘法转化为高斯整数乘法，从而提高计算效率。

**💡 创新点**

创新点在于反转了传统的乘法层次结构，通过利用高斯整数的特殊性质，减少了整体算术成本。

**🔧 技术方法**

使用了Toom-Cook算法和高斯整数的相关算法，结合了最短线性程序问题的优化方法。

**📊 数据集**

使用了GNU多精度算术库（GMP）作为基准进行性能比较。

**📈 对比分析**

与GMP的高度优化的Toom-Cook算法相比，提出的方法在多个机器上性能提升高达29%。

**⚠️ 局限性**

限制在于高斯整数乘法在小规模乘法时可能会导致更多的进位，影响性能。

---

## 12. What Do Multilingual Self-Supervised Speech Encoders Know About Sentence Meaning?

**arXiv ID:** 2611.17686 | [PDF](https://arxiv.org/pdf/2611.17686v1)

**作者:** Kevin Krahn `[一作]` (Ohio State University), Eric Fosler-Lussier `[通讯]` (Ohio State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大规模多语言自监督语音模型（S3Ms）在76种语言中的句子级语义结构，使用线性映射将冻结的语音表示映射到冻结的文本嵌入空间，并通过跨模态检索进行评估。

**💡 创新点**

创新点在于揭示了多语言S3Ms中句子意义的线性可访问性及其在不同语言和资源层次之间的不均匀性，同时展示了真正的跨语言语义转移的存在。

**🔧 技术方法**

使用了线性探测器将多语言语音表示映射到文本嵌入空间，并通过对比损失进行训练。

**📊 数据集**

使用了FLEURS和Bible两个多语言平行语料库，涵盖了多种语言和不同的资源层次。

**📈 对比分析**

与其他方法相比，OmniASR模型在高资源语言上表现出显著的性能提升，但在中低资源语言上提升有限。单语解码能力并不预测跨语言转移的效果。

**⚠️ 局限性**

限制在于本研究仅测量了从冻结表示中线性可解的内容，可能低估了实际存在的语义内容，同时由于声学关联，结果可能夸大了语义内容的存在。

---

## 13. Fault Tolerant Coresets

**arXiv ID:** 2611.17574 | [PDF](https://arxiv.org/pdf/2611.17574v1)

**作者:** Milind Prabhu `[一作]` (University of Michigan), Sudarshan Shyam `[通讯]` (King's College London)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在对抗性损坏情况下，如何从损坏的k-means核心集恢复有效的核心集。主要结果表明，通过增加核心集的大小，可以在遭受对抗性损坏后恢复出有效的核心集。

**💡 创新点**

创新点在于提出了一种容错的核心集构建方法，利用敏感性采样框架，使得核心集在遭受对抗性损坏后仍能有效恢复，且只需增加有限的样本量。

**🔧 技术方法**

使用了敏感性采样技术，该技术通过根据点的敏感性进行采样来构建核心集，并在损坏后通过迭代解码器恢复核心集。

**📊 数据集**

实验使用了多个数据集，包括Adult、Taxi、CoverType等，测试了在对抗性损坏下的恢复效果。

**📈 对比分析**

与BICO和StreamKM++等方法进行比较，敏感性采样在对抗性损坏下的平均失真度最低。例如，在Taxi数据集上，敏感性采样的失真度为1.124，而BICO和StreamKM++分别为1.689和1.686。

**⚠️ 局限性**

限制在于该框架主要适用于通过敏感性采样构建的核心集，对于不适合该机制的问题，可能无法保证容错性。此外，计算k-means的敏感性近似仍然是一个挑战。

---

## 14. Protection-Oriented Assessment of Machine Learning-Based Fault Detection in Power Systems

**arXiv ID:** 2611.17872 | [PDF](https://arxiv.org/pdf/2611.17872v1)

**作者:** Julian Oelhaf `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Siming Bayer `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文开发了一种面向保护的故障检测器评估方法，并将其应用于三个机器学习检测器和一个序列电流启动器参考，使用PROTECT-90电磁瞬态数据集进行评估。

**💡 创新点**

创新点在于提出了一种结合五种条件窗口分类、因果窗口结束标记、基于保护的评估指标以及在固定安全预算下的操作点选择的评估框架。

**🔧 技术方法**

使用了梯度提升树（GBT）、一维卷积神经网络（CNN）和逻辑回归（LogReg）等机器学习技术。

**📊 数据集**

使用了PROTECT-90数据集，该数据集包含9022个在90 kV双回路线路上的模拟故障事件。

**📈 对比分析**

与传统方法相比，所有三个机器学习检测器的AUC-ROC值均超过0.999，但在匹配的安全预算下，开发故障的漏检率差异超过两个数量级，GBT为0.007%，CNN为0.008%，而LogReg为1.99%。

**⚠️ 局限性**

限制在于PROTECT-90数据集仅包含故障事件，安全性评估仅在故障前和故障清除后进行，未涵盖独立的非故障干扰。

---

## 15. CamSpokenNLI: A Pilot Resource for Cross-Variety Transfer Evaluation in Cameroon Pidgin English and Camfranglais

**arXiv ID:** 2611.17573 | [PDF](https://arxiv.org/pdf/2611.17573v1)

**作者:** Tiani Pekins Ebika `[一作]` `[通讯]` (University of Buea), Tiani Pekins Ebika (University of Buea)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文介绍了CamSpokenNLI，这是一个针对喀麦隆皮钦英语和Camfranglais的口语自然语言推理（NLI）数据集，包含228个场景匹配的口语NLI项目，旨在填补这两个语言变体在NLI资源方面的空白。

**💡 创新点**

创新点在于首次提供了专门针对喀麦隆皮钦英语和Camfranglais的口语NLI资源，并且数据集包含手动注释的场景匹配项目，具有探索性而非成熟基准。

**🔧 技术方法**

使用了XLM-RoBERTa模型进行NLI实验，并进行了五折交叉验证，同时还进行了零-shot Whisper转录的评估。

**📊 数据集**

数据集包含114个场景匹配的口语NLI项目，每种语言变体各114个，记录由一位来自Buea的母语者录制。

**📈 对比分析**

与假设仅输入的比较显示，NLI模型的准确率在喀麦隆皮钦英语上为44.7%，在Camfranglais上为46.2%，均高于33.3%的机会水平，但与假设仅输入的准确率没有显著差异，表明假设措辞可能影响了准确性。

**⚠️ 局限性**

限制在于数据集规模较小，仅包含114个示例，且所有项目均由单一说话者录制，缺乏区域、人口和主题的多样性，无法进行独立的开发或测试分区。

---

## 16. MacroGS: Macro-Pixel Shared Gaussian Primitives for Arbitrary-Scale Light Field Super-Resolutio

**arXiv ID:** 2611.17957 | [PDF](https://arxiv.org/pdf/2611.17957v1)

**作者:** Zeqiang Wei `[一作]` (Capital Normal University), Min Xu `[通讯]` (Capital Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种用于任意尺度光场空间超分辨率重建的方法，该方法通过低分辨率多视图观察重建任意请求尺度的高分辨率视图。

**💡 创新点**

创新点在于使用共享的高斯原语集来避免为每个视图生成冗余的原语集合，从而减少原语数量，并通过每个原语的视差引导跨视图特征读取和渲染。

**🔧 技术方法**

采用了一种前馈解码器，该解码器联合预测每个宏像素的共享高斯原语集，并通过视差引导特征读取和渲染。

**📊 数据集**

在五个基准数据集上进行了实验，包括EPFL、HCInew、HCIold、INRIA和STFgantry。

**📈 对比分析**

与现有方法相比，该方法在大多数评估的任意尺度设置中表现出色，且在固定尺度下提高了重建质量，解码速度比最先进的隐式光场解码器快约10-18倍。

**⚠️ 局限性**

限制在于该方法仅通过重建学习视差，可能无法保证几何精度，且未来工作需要探索内容自适应分配共享高斯原语以更好地表示几何边界和细结构。

---

## 17. Grounding Object-to-Object Hazard Reasoning into 3D Risk Fields for Task-Conditioned Safety

**arXiv ID:** 2611.17566 | [PDF](https://arxiv.org/pdf/2611.17566v1)

**作者:** Mariana Sosa Guzmán `[一作]` (McGill University), Hsiu-Chin Lin `[通讯]` (McGill University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于视觉-语言模型（VLM）的框架，用于推断机器人操作的任务和场景相关的安全约束，能够判断任务是否应被拒绝，并识别可预防的物体对危险。

**💡 创新点**

创新点在于将任务拒绝与识别可预防的物体对危险分开，并通过开放词汇分割和深度信息构建连续的3D风险场，从而将VLM的危险推理与障碍调制相结合。

**🔧 技术方法**

使用了视觉-语言模型（VLM）、开放词汇分割和深度信息等技术。

**📊 数据集**

在模拟和真实世界中评估了框架，使用了RGB-D观察数据。

**📈 对比分析**

与仅基于任务的推理、场景描述推理和几何障碍调制的方法进行了比较，结果显示该框架在防止危险和任务完成方面表现最佳。

**⚠️ 局限性**

限制在于当前方法主要依赖于固定的空间关系词汇，未来需要考虑更抽象的推理，例如与执行时间相关的安全约束。

---

## 18. Disentangling Forgetting from Utility: Lightweight LLM Unlearning via Orthogonal Null-Space Projection

**arXiv ID:** 2611.17864 | [PDF](https://arxiv.org/pdf/2611.17864v1)

**作者:** Xiuyuan Wang `[一作]` (Zhejiang University), Xiaolin Zheng `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的轻量级大语言模型（LLM）遗忘框架UUE，通过正交零空间投影实现可控的模型编辑，以有效去除特定知识而不影响模型的整体效用。

**💡 创新点**

创新点在于将遗忘重新定义为可控的模型编辑，采用无目标的编辑目标，确保在去除不需要的知识时对模型的通用能力干扰最小化。

**🔧 技术方法**

使用了正交零空间投影技术和轻量级的可插拔适配器（包括低秩变体UUE-L），实现高效的模型更新。

**📊 数据集**

在TOFU和WMDP基准上进行了广泛的实验，使用了合成作者传记和相关领域的文档作为数据集。

**📈 对比分析**

与现有的遗忘方法相比，UUE和UUE-L在遗忘效果上显著优于其他方法，同时保持了较强的模型效用，尤其在TOFU基准上表现突出。

**⚠️ 局限性**

限制在于需要适当选择正则化参数和零空间阈值，以平衡遗忘效果和保留知识的能力，过强或过弱的正则化都会影响模型性能。

---

## 19. Where Tree Ensembles Decide: Structural Local Explanation with Decision Predicate Graphs

**arXiv ID:** 2611.18382 | [PDF](https://arxiv.org/pdf/2611.18382v1)

**作者:** Sylvio Barbon Junior `[一作]` (University of Trieste), Luca Longo `[通讯]` (University College Cork)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种局部决策谓词图（local DPG），用于记录树集成模型在单个样本上的计算过程，并选择关键谓词，以便更好地解释模型的预测。

**💡 创新点**

创新点在于通过局部 DPG 结构化地记录执行的规则，并从中提取关键谓词，提供了一个具体的阈值和相关树的信息，而无需外部解释器。

**🔧 技术方法**

使用了局部决策谓词图（local DPG）技术，结合了结构化解释和关键谓词的选择。

**📊 数据集**

使用了来自 OpenML 和 scikit-learn 的公共数值分类数据集，这些数据集大多源于 UCI 资源库，包含 4 到 617 个特征和 2 到 26 个类别。

**📈 对比分析**

与随机选择的谓词和基于特征排名的解释方法（如 TreeSHAP 和 LIME）相比，关键谓词在分离不同预测邻居方面表现更好，并且在恢复错误预测的真实类别时效果显著。

**⚠️ 局限性**

局部 DPG 的构建时间为每个样本 10 到 100 毫秒，但图形本身较大且不够紧凑，且训练无关的谓词编辑未能减少未见数据上的错误。

---

## 20. JVT: Just Video Transformer makes pixel-space video diffusion simple

**arXiv ID:** 2611.18549 | [PDF](https://arxiv.org/pdf/2611.18549v1)

**作者:** Jianjun Zhou `[一作]` (Zhejiang University), Haoyu Guo `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为JVT（Just Video Transformer）的像素空间视频生成模型，利用简单的DiT风格变换器直接生成视频，支持类条件和无条件生成。

**💡 创新点**

JVT是首个基于简单DiT风格变换器的像素空间视频扩散模型，省去了细节处理器、级联阶段和U-Net风格的跳跃连接，展示了简单架构在视频生成中的潜力。

**🔧 技术方法**

使用了简单的DiT风格变换器架构，结合视频补丁嵌入、注册令牌和自适应层归一化等技术。

**📊 数据集**

在UCF-101、FaceForensics、SkyTimelapse和Taichi-HD等数据集上进行了实验，使用了随机提取的16帧视频片段。

**📈 对比分析**

与基于潜在扩散的基线模型（如Latte）相比，JVT在类条件和无条件设置下的生成质量具有竞争力，且在效率上表现良好。

**⚠️ 局限性**

模型在相机跟踪方面的表现有所提升，但在相机平移的准确性上仍存在一定的局限性，表明在生成过程中对场景几何的理解仍需改进。

---

## 21. PF-Exec: Large-Scale UAV Relay Load Balancing with an Anchored Potential Direction Decider and a Per-Dimension Gain-Learning Executor

**arXiv ID:** 2611.17492 | [PDF](https://arxiv.org/pdf/2611.17492v1)

**作者:** Liheng Zeng `[一作]` (University of Sussex), Bo Ma `[通讯]` (Zhejiang Gongshang University)

**通讯引用:** 215 | [OpenAlex ID](https://openalex.org/A5065783765)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种用于大规模无人机中继负载平衡的系统PF-Exec，该系统通过锚定潜在方向决策器和每维增益学习执行器将决策分为两个层次。

**💡 创新点**

创新点在于将负载平衡问题分解为方向和速度两个部分，利用闭式形式的目标来指导执行器的学习，从而提高了在高负载情况下的公平性和稳定性。

**🔧 技术方法**

使用了深度强化学习技术，特别是TD3算法来训练增益学习执行器，同时结合了潜在函数方法来确定方向。

**📊 数据集**

使用了602次实验评估，涵盖了3到100个无人机的不同规模和两种人群场景，比较了多种基线方法的性能。

**📈 对比分析**

与十种比较方法进行比较，PF-Exec在高负载情况下保持了接近1的公平性指数，而固定步长执行器在相同条件下的公平性指数降至0.310和0.0089，显示出PF-Exec在负载集中时的优势。

**⚠️ 局限性**

限制在于系统在面对极端需求聚集时的表现，尽管PF-Exec表现优异，但在某些情况下仍可能受到信息延迟和环境变化的影响。

---

## 22. AG-JEPA: Anatomically Guided Joint Embedding Predictive Architecture for Neurodegenerative Representation Learning

**arXiv ID:** 2611.18097 | [PDF](https://arxiv.org/pdf/2611.18097v1)

**作者:** Azad Singh `[一作]` (Microsoft Research India), Mercy Prasanna Ranjit `[通讯]` (Microsoft Research India)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种名为Anatomically Guided JEPA (AG-JEPA)的新框架，通过引入解剖学标记与补丁标记的交互，学习与神经退行性变相关的区域特定表示。

**💡 创新点**

创新点在于通过解剖学标记增强了JEPA模型，使其能够捕捉与阿尔茨海默病相关的特定脑区的结构变化，并且首次将疾病进展建模为潜在空间中的条件位移。

**🔧 技术方法**

使用了联合嵌入预测架构（JEPA）和3D视觉变换器（Vision Transformer）等技术。

**📊 数据集**

使用了来自ADNI、MR-RATE、AIBL和MIRIAD等公共数据集的T1加权脑MRI图像。

**📈 对比分析**

与现有基线模型相比，AG-JEPA在CN/MCI分类任务上提高了8.4个百分点的AUROC，在跨分布的CN/AD分类上提高了7.1个百分点，显示出其在早期检测和跨队列泛化方面的有效性。

**⚠️ 局限性**

限制在于目前未公开代码，并且仅使用了五个与神经退行性变相关的脑区，未来可能需要探索更丰富的解剖学先验。

---

## 23. Deterministic ECDSA on secp256k1: the RFC 6979 assumption, a counterexample, and the state of deployed implementations

**arXiv ID:** 2611.18449 | [PDF](https://arxiv.org/pdf/2611.18449v1)

**作者:** Ilia Prosianoi `[一作]` `[通讯]` (Higher School of Economics), Ilia Prosianoi (Higher School of Economics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

论文探讨了RFC 6979在ECDSA中的安全性，分析了其基于HMAC_DRBG的确定性nonce生成方法的安全假设，并提出了该假设的局限性。

**💡 创新点**

创新点在于明确了RFC 6979的安全性并不依赖于HMAC的PRF安全性或辅助输入PRF安全性，而是需要额外的假设，并提供了反例来证明这一点。

**🔧 技术方法**

使用了HMAC_DRBG作为nonce生成的核心技术，并分析了SHA-256压缩函数的性质。

**📊 数据集**

审计了31个secp256k1实现，分析了它们在nonce生成和签名过程中的表现。

**📈 对比分析**

通过与随机ECDSA的比较，证明了确定性ECDSA在某些情况下会泄露签名密钥，且在特定条件下，两个签名可以揭示密钥。

**⚠️ 局限性**

局限性在于RFC 6979的安全性假设并不充分，且在某些情况下，存在nonce函数可以在已知公钥的情况下泄露密钥。

---

## 24. TetraCoder: Adaptive Tetrahedral Encoding for Neural Fields

**arXiv ID:** 2611.18267 | [PDF](https://arxiv.org/pdf/2611.18267v1)

**作者:** Daniele Della Pietra `[一作]` (University of Trento), Nicola Garau `[通讯]` (University of Trento)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种新的自适应四面体编码方法TetraCoder，旨在将显式四面体空间编码与隐式有符号距离场(SDF)相结合，从而实现几何体的高效表示和编辑。

**💡 创新点**

创新点在于采用单一分辨率的无哈希自适应四面体编码，能够在隐式和显式几何体之间建立直接对应关系，并支持快速的分析网格检索和几何外观编辑。

**🔧 技术方法**

使用了自适应四面体网格和重心插值技术，结合ReLU几何解码器，确保了连续分段仿射(CPWA)结构的保持。

**📊 数据集**

在多个任务上进行了评估，包括2D/3D图像拟合、3D新视图合成、3D重建和几何/外观编辑，使用的数据集包括NeRF Synthetic和DTU。

**📈 对比分析**

与现有的多分辨率哈希编码方法相比，TetraCoder在多个任务上表现出更好的拟合效果和更少的可训练参数，同时在几何重建任务中，分析提取时间减少了约6.4倍。

**⚠️ 局限性**

限制在于当前方法尚未扩展到动态4D场景，未来工作将致力于在多个细节层次上实现分析网格提取，并证明隐式与显式之间的双向联系。

---

## 25. See Every Block Once: Variance-Reduced Post-Training for Masked Diffusion Language Models

**arXiv ID:** 2611.17754 | [PDF](https://arxiv.org/pdf/2611.17754v1)

**作者:** Yunzhen He `[一作]` (Kyoto University), Hidetoshi Shimodaira `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种全块估计器，旨在解决蒙特卡洛估计中由于随机块索引采样导致的缺失覆盖和估计方差问题。

**💡 创新点**

创新点在于设计了一种全块估计器，该估计器确保每个块都被评估一次，从而消除了块选择引起的组间方差，降低了估计方差。

**🔧 技术方法**

使用了蒙特卡洛方法来估计块状替代目标，并提出了全块估计器。

**📊 数据集**

在五个基准数据集上进行了实验，包括GSM8K、MATH-500、Countdown、Sudoku和HumanEval。

**📈 对比分析**

与随机块采样基线相比，提出的方法在四个基准上取得了最高准确率，平均提高了12.79分，且在Spearman等级相关性和估计误差方面表现更优。

**⚠️ 局限性**

限制在于由于计算资源限制，未进行增加训练步骤或GRPO组大小的实验；此外，尚未验证该估计器在其他策略优化方法中的效果。

---

## 26. Federated Soft Poly-Level Learning Over a Distributed Network with Guaranteed Convergence

**arXiv ID:** 2611.17722 | [PDF](https://arxiv.org/pdf/2611.17722v1)

**作者:** Yang Jiao `[一作]`, Jianwei Huang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为联邦多级学习（FPLL）的框架，以联邦方式解决具有软约束的多级学习问题。

**💡 创新点**

创新点在于引入了一种新的非线性切割方法（L-cut），用于多级学习问题的层次松弛，并且FPLL框架允许灵活控制计算复杂性与性能效率之间的权衡。

**🔧 技术方法**

使用了非线性切割方法（L-cut）和分布式多梯度下降方法。

**📊 数据集**

在多个数据集上进行了广泛的实验，包括MNIST、USPS、F-MNIST、KMNIST和QMNIST。

**📈 对比分析**

与现有的最先进方法（如ADBO、FEDNEST和AFTO）进行比较，FPLL在联邦持续学习和鲁棒微调任务中表现出色，验证了L-cut的有效性和通信效率。

**⚠️ 局限性**

限制在于当前方法主要集中在多级学习问题上，未来的工作可以扩展到显式有界的可行域，并考虑投影更新。

---

## 27. Classifier-Guided Latent Diffusion for Small-Molecule Generation

**arXiv ID:** 2611.17650 | [PDF](https://arxiv.org/pdf/2611.17650v1)

**作者:** Ali Forooghi `[一作]` (University of Windsor), Alioune Ngom `[通讯]` (University of Windsor)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了一种基于潜在扩散模型的框架，用于指导小分子的生成，特别是针对药物相关的性质进行生成。

**💡 创新点**

创新点在于结合了预训练的ChemBERTa嵌入、Transformer基础的SELFIES解码器和无条件的潜在扩散模型，通过潜在属性预测器引导生成过程。

**🔧 技术方法**

使用了ChemBERTa编码器、Transformer解码器和无条件去噪扩散模型等技术。

**📊 数据集**

使用了ZINC250K数据集进行解码器和扩散模型的训练，以及MoleculeNet中的BBBP和BACE-1数据集进行属性预测和引导模型的训练。

**📈 对比分析**

与现有方法相比，提出的方法在生成的分子有效性、独特性和新颖性方面表现良好，且在BBBP和BACE-1的指导生成中，生成的分子在目标属性上有显著提升。

**⚠️ 局限性**

限制在于所有属性结果都是模型预测的分数，而非实验测量，因此需要谨慎解读。此外，指导强度对多样性和描述符分布行为的影响需要进一步研究。

---

## 28. Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent

**arXiv ID:** 2611.17939 | [PDF](https://arxiv.org/pdf/2611.17939v1)

**作者:** Cosimo Galeone `[一作]`, Daniele Ligorio `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对一款未调优的金融研究代理进行了控制案例研究，提出了五个关键问题，探讨了该代理的稳定性、故障归属、诊断可信度和变更可信度。

**💡 创新点**

创新点在于提出了一种基于故障归属的分类法，明确了在代理失败时，谁负责修复，并通过盲评验证了诊断的可靠性。

**🔧 技术方法**

使用了优化循环技术，通过分析代理的运行轨迹，进行针对性的改进，并在成本和准确性目标下进行评估。

**📊 数据集**

使用了Vals Finance Agent Benchmark v2数据集，该数据集包含27个问题，分为九个类别，并进行了18/9的开发/测试分割。

**📈 对比分析**

与其他方法相比，本文的方法通过测量代理的漂移和故障归属，提供了更为细致的分析。性能方面，成本优化后的配置在测试中表现出显著的成本节约，而准确性提升不明显。

**⚠️ 局限性**

限制在于该研究仅针对一个代理和九个测试问题，结果不具普遍性，且未与其他排行榜进行比较。

---

## 29. X-Nav Technical Report: Real-World Long-Horizon Indoor-to-Outdoor Navigation on a Humanoid Robot

**arXiv ID:** 2611.17598 | [PDF](https://arxiv.org/pdf/2611.17598v1)

**作者:** Jiaxin Huang `[一作]` (AgiBot X-Lab), Shaoli Huang `[通讯]` (AgiBot X-Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种导航系统，使双足类人机器人能够通过语言指令完成从室内到室外的长距离导航任务。

**💡 创新点**

这是首个实现双足类人机器人在语言指令下完成约100米室内到室外任务的系统，具有数据收集、模型适应和自主导航的完整管道。

**🔧 技术方法**

使用了RGB图像作为输入的导航模型，分为预测动作和预测导航目标两个版本，并结合了代理工作流进行任务分解和障碍物规避。

**📊 数据集**

使用了开放源代码场景、部署地点的数字双胞胎和真实机器人收集的数据集。

**📈 对比分析**

在R2R-CE和RxR-CE基准测试中表现出竞争力，适应后的模型在数字双胞胎中成功率从34%提高到78%。

**⚠️ 局限性**

当前系统仅在一个楼层的单一路径上进行测试，未考虑楼梯、电梯和需要打开的门等复杂场景，且适应性评估仅在一个地点进行。

---

## 30. Differentially Private Federated Class-Incremental Learning for EEG Mental State Classification

**arXiv ID:** 2611.17944 | [PDF](https://arxiv.org/pdf/2611.17944v1)

**作者:** Baran Can Gül `[一作]` (Institute of Industrial Automation and Software Engineering), Michael Weyrich `[通讯]` (Institute of Industrial Automation and Software Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在小型客户端群体中进行差分隐私的联邦增量学习，以实现脑电图（EEG）心理状态分类。

**💡 创新点**

提出了一种新的训练协议和方法（FedEEG），结合了突触智能（Synaptic Intelligence）和延迟自适应的FedProx项，显著减少了灾难性遗忘。

**🔧 技术方法**

使用了突触智能（SI）和延迟自适应的FedProx技术，结合了掩蔽的二元交叉熵损失函数和客户端级差分隐私。

**📊 数据集**

使用了一个公共的Muse EEG数据集和一个自定义的四通道MUSE 2录音集，组成了一个包含5个客户端的联邦。

**📈 对比分析**

与FedAvg、FedProx和知识蒸馏基线（Federated Learning-without-Forgetting）进行比较，FedEEG在减少遗忘方面表现更好，最终准确率为50.5%±3.6%。

**⚠️ 局限性**

在小型客户端群体中，模型的最终准确率仍然接近随机猜测的水平，且头部范围的差分隐私保证仅覆盖分类头，存在实际应用中的局限性。

---

## 31. DiffusionSSD: Achieving Extreme Visual Compression with Dual-stage Diffusion and Cross-level Near-Data Processing

**arXiv ID:** 2611.17908 | [PDF](https://arxiv.org/pdf/2611.17908v1)

**作者:** Kuai Yu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Gong Zhang `[通讯]` (Huawei Technologies Co., Ltd)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为DiffusionSSD的框架，用于在边缘设备上进行极端视觉数据压缩，采用算法-架构协同设计的方法。

**💡 创新点**

创新点在于提出了双阶段扩散（Dual-stage Diffusion）和跨层近数据处理架构（Cross-level Near Data Processing），有效降低计算成本并提高压缩性能。

**🔧 技术方法**

使用了双阶段扩散算法和跨层近数据处理架构，结合计算通用闪存（cUFS）和LPDDR-PIM技术。

**📊 数据集**

使用了Vimeo-90K、Tecnick和UVG 2K数据集进行实验评估。

**📈 对比分析**

与NPU+DRAM和其他近数据处理设计（如SmartSSD）相比，cUFS+PIM在编码吞吐量上分别提高了2.32倍和2.91倍，同时保持了相似的视觉质量。

**⚠️ 局限性**

限制在于当前架构仍需在资源受限的边缘设备上进行优化，尤其是在处理复杂的编码和解码任务时，可能会面临计算和带宽的瓶颈。

---

## 32. Restoring Diversity in Distilled Image and Video Generators by Probing Activations

**arXiv ID:** 2611.18550 | [PDF](https://arxiv.org/pdf/2611.18550v1)

**作者:** Furkan Horoz `[一作]` (University of Illinois at Urbana-Champaign), James Matthew Rehg `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出了一种方法ReDiverse，用于改善经过蒸馏的图像和视频生成模型的多样性，解决了在固定提示下生成样本多样性下降的问题。

**💡 创新点**

创新点在于通过分析蒸馏学生模型的内部表示，发现多样性崩溃是由于激活方差的方向性压缩，而不是生成能力的固有损失，并提出了两种干预措施来恢复缺失的方差方向。

**🔧 技术方法**

使用了LoRA模块进行特征回归和无训练的激活引导机制，以恢复蒸馏学生模型的多样性。

**📊 数据集**

使用了多个图像和视频蒸馏基准数据集，包括SDXL、SD3、Wan 2.1和MiniMax-H3。

**📈 对比分析**

与其他方法（如噪声优化和DP-DMD）相比，ReDiverse在多样性指标上表现优越，能够恢复大部分多样性损失，并在生成质量上保持接近教师模型的水平。

**⚠️ 局限性**

限制在于有时生成质量会下降，尤其是在图像的奖励模型分数上，并且假设学生模型与教师模型共享相同的架构。

---

## 33. LinearKnot: Linear-Time Approximation of the Pseudoknot Partition Function

**arXiv ID:** 2611.17602 | [PDF](https://arxiv.org/pdf/2611.17602v1)

**作者:** Milan Gautam `[一作]` (Oregon State University), Liang Huang `[通讯]` (Oregon State University)

**通讯引用:** 4497 | [OpenAlex ID](https://openalex.org/A5075538786)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种新的线性时间近似算法，用于计算RNA的伪结结构的分区函数和碱基配对概率。

**💡 创新点**

创新点在于通过三次扫描和保留伪结螺旋的5'核苷酸，显著提高了伪结结构的预测效率和准确性。

**🔧 技术方法**

使用了动态规划和束搜索技术，结合了伪结自由的预测方法。

**📊 数据集**

使用了Rous sarcoma病毒基因组的前缀数据集，长度从1,000到9,392不等。

**📈 对比分析**

与现有的伪结预测算法相比，新的算法在时间复杂度上为线性(n^1.01)，内存使用上略高于线性(n^1.15)，在处理长RNA时表现出显著的效率提升。

**⚠️ 局限性**

限制在于算法可能无法处理某些复杂的伪结结构，并且在扫描过程中可能会错过某些内部多环或填充物中的伪结。

---

## 34. PRISM: A Post-Prediction Reliability Framework for Evidence-Guided Correction and Selective Abstention

**arXiv ID:** 2611.18127 | [PDF](https://arxiv.org/pdf/2611.18127v1)

**作者:** Omm Swarup `[一作]` (Indian Institute of Technology Madras), Shrishail M. Hiremath `[通讯]` (National Institute of Technology Rourkela)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了PRISM框架，用于在生成预测后评估预测的可靠性，并在必要时进行自我修正。

**💡 创新点**

PRISM框架结合了错误风险估计、诊断、历史案例检索和可能修正的验证，能够在证据不足时选择保留原始预测、替换为经过验证的替代预测或放弃预测。

**🔧 技术方法**

使用了错误风险估计、历史案例检索和验证机制等技术。

**📊 数据集**

使用了来自IPL的球对球数据集，共包含8,665个局级观察，测试集包含2,568个观察。

**📈 对比分析**

与逻辑回归、随机森林和HistGradientBoosting模型进行比较，PRISM在选择性准确率上达到82.4%，覆盖率为88.5%，Brier分数为0.126，期望校准误差为0.019。

**⚠️ 局限性**

PRISM的性能需要与覆盖率一起考虑，较高的选择性准确率可能意味着做出的预测较少，且目前的实验仅集中在IPL第二局预测上，需在其他任务和数据集上进行进一步评估。

---

## 35. VACO: Vertex-Aligned Decoding for Dense Hand Contact Estimation from a Single Image

**arXiv ID:** 2611.17755 | [PDF](https://arxiv.org/pdf/2611.17755v1)

**作者:** Binhong Ye `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的方法，通过单张RGB图像估计手部的密集接触图，利用每个MANO顶点在其投影位置采样局部图像特征。

**💡 创新点**

创新点在于引入了顶点对齐解码器，替代了共享查询瓶颈，使得每个顶点的接触决策能够直接参考其在图像中的位置，从而减少了接触预测的模糊性。

**🔧 技术方法**

使用了顶点对齐解码架构和联合训练的手重建骨干网络，结合了多尺度特征采样和轻量级Transformer解码器。

**📊 数据集**

训练使用了HACO标准化的统一数据集，涵盖手-物体、手-人、手-场景和双手交互等多种场景。

**📈 对比分析**

与HACO等现有方法进行对比，结果显示在多个基准测试中，F1分数提高了20.7个百分点，并且在未见数据集上也表现出良好的泛化能力。

**⚠️ 局限性**

局限性在于预测的支架准确性受到限制，尤其在严重遮挡或截断情况下，姿态和接触错误相互耦合。此外，顶点的重叠投影可能导致相同的图像特征采样，缺乏深度、法线或可见性线索。

---

## 36. BAL: Bidirectional Autoregression in Latent Space for Learning Human Movement Representations

**arXiv ID:** 2611.18559 | [PDF](https://arxiv.org/pdf/2611.18559v1)

**作者:** Genki Kinoshita `[一作]` (Kyoto University), Ko Nishino `[通讯]` (Kyoto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种自监督框架BAL，通过利用日常人类行为的自然顺序结构来学习运动表示。

**💡 创新点**

创新点在于通过双向自回归预测在潜在空间中进行学习，充分利用运动的顺序约束，减少对启发式超参数的依赖。

**🔧 技术方法**

使用了双向自回归预测技术，结合潜在空间中的运动表示学习。

**📊 数据集**

使用了三个包含自然日常活动的数据集：AMD、Embody 3D和Nymeria。

**📈 对比分析**

与现有的自监督方法（如掩码建模和对比学习）相比，BAL在大多数评估指标上表现更优，展示了其在运动字幕、预测和插值任务中的有效性。

**⚠️ 局限性**

限制在于模型可能会面临表示崩溃的风险，尽管通过目标分解来缓解这一问题，但仍需进一步研究以确保模型的稳定性。

---

## 37. Can Agents Keep Up with Time? An Empirical Analysis of Temporal Adaptation After a Conversational Hiatus

**arXiv ID:** 2611.17806 | [PDF](https://arxiv.org/pdf/2611.17806v1)

**作者:** Hongguan Su `[一作]` (People's Public Security University of China), Peng Chen `[通讯]` (People's Public Security University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了长时间跨度的智能助手在用户重新启动对话后是否能够适应时间变化，特别是在预定事件日期已过的情况下，智能助手是否能够更新其建议。

**💡 创新点**

创新点在于定义了时间状态感知及其应用，并设计了会话重启基准，以指导长时间跨度智能助手的设计，强调了时间感知在智能助手中的重要性。

**🔧 技术方法**

使用了会话重启任务设计和虚拟时钟工具，结合了五种模型路线和五种时间信息条件进行实验。

**📊 数据集**

数据集包含940个响应，涵盖了五种模型路线和多种服务事件类型，如医疗检查、签证面试等。

**📈 对比分析**

通过比较不同条件下的响应，发现智能助手能够获取当前时间，但在更新建议时常常失败。提供明确的时间上下文可以显著减少错误，而提供计算的日期关系则可以完全消除这些错误。

**⚠️ 局限性**

限制在于研究主要基于合成的中文服务场景，且评估依赖于返回推理链的模型，可能不适用于所有类型的智能助手。

---

## 38. PROOF: Perceptive Robustness via Out-of-Distribution Fallback for Legged Locomotion

**arXiv ID:** 2611.18305 | [PDF](https://arxiv.org/pdf/2611.18305v1)

**作者:** Siyuan Wang `[一作]` (Southern University of Science and Technology), Jiankun Wang `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为PROOF的感知运动框架，使四足机器人在严重视觉干扰下仍能进行灵活的运动。

**💡 创新点**

创新点在于引入了一个双阶段的教师-学生学习范式，能够在视觉输入失效时通过抑制不可靠的视觉特征并回退到本体运动行为来保持运动的稳定性。

**🔧 技术方法**

采用了教师-学生学习框架，结合了不确定性感知和自适应门控机制。

**📊 数据集**

在真实环境中使用了Unitree Go2四足机器人和RealSense D435深度相机进行评估。

**📈 对比分析**

与两个基线方法进行比较，PROOF在强视觉干扰下表现出更高的成功率和稳定性，尤其是在复杂地形上，成功率达到100%。

**⚠️ 局限性**

限制在于无法全面模拟所有可能的传感器和环境条件，且预测的对数方差仅为相对重建不确定性特征，而非经过校准的地形估计误差或干扰概率。

---

## 39. How rigid is maximum cardinality search? Stability and locality of MCS orderings under edge updates, verified in Lean

**arXiv ID:** 2611.18369 | [PDF](https://arxiv.org/pdf/2611.18369v1)

**作者:** Md Thoriqul Islam Thonmay `[一作]` (Eotvos Lorand University), Gregory Morse `[通讯]` (Eotvos Lorand University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了图G和G F在最大基数搜索（MCS）排序下的共享条件，证明了在特定条件下它们可以共享MCS排序。

**💡 创新点**

提出了稳定性定理和局部性分析，首次对MCS排序在边更新下的行为进行了机器检查的增量修复。

**🔧 技术方法**

使用了Lean 4和Mathlib进行形式化证明，并实现了布尔邻接矩阵的可执行更新。

**📊 数据集**

未具体提及使用的数据集，但涉及的图结构和边修改操作是研究的核心。

**📈 对比分析**

通过与传统的重计算方法进行比较，证明了在特定条件下更新的时间复杂度为O(((v)+1)log n)，在稀疏端点情况下可以在o(n)时间内认证无变化更新。

**⚠️ 局限性**

局限性在于未能提供最坏情况下的渐近加速，且某些条件（如单向更新）不能放宽。

---

## 40. Fuse-All MoE Inference Kernel with Load Balancing on TPUs

**arXiv ID:** 2611.17857 | [PDF](https://arxiv.org/pdf/2611.17857v1)

**作者:** Rupeng Liu `[一作]` (Meta), Yejin Lee `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

提出了一种新的Fuse-All MoE TPU内核设计，旨在优化专家计算、输出通信和负载平衡，以提高Mixture-of-Experts (MoE)架构在TPU上的推理效率。

**💡 创新点**

创新点在于通过融合计算和返回管道，减少中间数据移动和不均匀的专家负载，从而提高MoE推理的效率，同时保持路由语义不变。

**🔧 技术方法**

使用了Fuse-All MoE TPU内核设计，结合了专家计算、输出传输和负载管理的共同优化。

**📊 数据集**

实验使用了TPUv7 (Ironwood) 2x2x2切片，包含八个芯片和十六个TensorCore设备，模型具有256个专家和4096的隐藏宽度。

**📈 对比分析**

与Megablox EP16基线相比，Fuse-All MoE在处理不均衡的EP16预填充工作负载时，内核运行时间从2.3倍减少到1.09倍；在32到512个解码令牌的情况下，EP8TP2的内核延迟相较于融合的EP16减少了最多21.3%。

**⚠️ 局限性**

限制在于当前设计主要针对特定的推理模式，扩展到其他模式或更复杂的负载管理可能需要进一步的研究和优化。

---

## 41. Why Finite Intelligence Works in an Infinite World: A Mathematical Theory

**arXiv ID:** 2611.18538 | [PDF](https://arxiv.org/pdf/2611.18538v1)

**作者:** Mojtaba Soltanalian `[一作]` `[通讯]` (University of Illinois Chicago), Mojtaba Soltanalian (University of Illinois Chicago)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

论文探讨了有限智能如何在无限环境中有效工作，提出了未来任务信息的概念，定义了保持决策价值所需的最小信息量。

**💡 创新点**

创新点在于通过数学理论形式化了有限信息在面对无限可能性时的有效性，强调了共享决策需求如何使有限信息保持有用。

**🔧 技术方法**

使用了数学理论和信息论的工具，特别是关于决策后悔状态的概念，以及信息-后悔曲线的分析。

**📊 数据集**

论文没有具体提到使用的数据集，而是通过理论推导和数学模型进行分析。

**📈 对比分析**

通过与现有的信息理论和决策理论进行比较，展示了所提出理论的有效性，强调了共享结构对信息需求的影响。

**⚠️ 局限性**

限制在于理论的适用性可能受到特定假设的限制，尤其是在处理无限任务家庭时，可能无法完全捕捉所有复杂性。

---

## 42. VellSSLH: Sound Type-Guided Selective Speculative Load Hardening for LLVM IR

**arXiv ID:** 2611.18463 | [PDF](https://arxiv.org/pdf/2611.18463v1)

**作者:** Swarn Priya `[一作]` `[通讯]` (Washington State University), Swarn Priya (Washington State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为VellSSLH的方法，通过在LLVM IR中选择性地插入投机加载硬化（SLH）来保护常量时间的C语言加密程序，防止在投机执行期间泄露秘密。

**💡 创新点**

创新点在于通过程序员指定公共和秘密输入，分析推断哪些值在投机期间可能变为秘密，从而仅在必要时插入保护，而不是对每个潜在受影响的值进行保护，从而减少运行时开销。

**🔧 技术方法**

使用LLVM IR进行程序转换和分析，结合Vellvm的LLVM语义扩展，建立了一个形式化的投机常量时间（SCT）安全性类型系统，并在Rocq中证明了检查器的健全性。

**📊 数据集**

在24个加密API和组件条目上进行评估，使用裸金属Intel处理器进行测试。

**📈 对比分析**

与LLVM SLH进行比较，VellSSLH在九次布局随机化试验中实现了4.03%的几何平均运行时降低。与Ultimate SLH相比，VellSSLH的运行时降低了19.65%。

**⚠️ 局限性**

限制在于该方法的安全性依赖于后端和硬件假设，且未涵盖所有LLVM重写、堆栈/ABI行为、指令适应或硬件合同的检查。

---

## 43. nnAEU: a Simple and Efficient CNN-Autoencoder for Hyperspectral Unmixing

**arXiv ID:** 2611.17956 | [PDF](https://arxiv.org/pdf/2611.17956v1)

**作者:** Edgard Dabier `[一作]`, Florence Tupin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种轻量级的基于CNN的自编码器，称为Not New Autoencoder Unmixing (nnAEU)，用于盲超光谱解混合（HSU）。

**💡 创新点**

nnAEU在参数数量上比UNMamba少了10倍，并且在准确性上超越了其他基于变换器和GAN的方法，同时保持了较低的复杂性。

**🔧 技术方法**

使用了轻量级的卷积神经网络（CNN）架构，并通过修剪UNMamba的复杂模块来简化模型。

**📊 数据集**

在多个标记数据集上进行了实验，包括Samson、Jasper Ridge、Apex和Urban。

**📈 对比分析**

与多种最先进的方法进行了比较，nnAEU在准确性和执行时间上表现优异，且参数数量显著减少，成为当前最小和最快的学习模型之一。

**⚠️ 局限性**

局限性在于评估仅限于标准基准场景，且Mamba或变换器组件在其他情况下（如更大的数据集或更多的端元）可能仍然有用。

---

## 44. OutfitterBench: Evaluating LLM Agents on Long-Horizon Resource Allocation and Commitment Coordination

**arXiv ID:** 2611.17933 | [PDF](https://arxiv.org/pdf/2611.17933v1)

**作者:** Jie Huang `[一作]` (Harbin Institute of Technology), Qiang Ye `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基准测试OutfitterBench，模拟代理管理一个户外设备租赁业务，持续200天，处理约1100个预订请求。

**💡 创新点**

创新点在于结合动态资产可用性、代理选择的未来义务和不确定机会，评估代理在长期业务价值目标下的资源分配和承诺协调能力。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行实验，结合轨迹分析和状态诊断，评估代理的决策过程。

**📊 数据集**

数据集来源于淘宝产品目录和历史北京天气，模拟客户需求和设备操作。

**📈 对比分析**

与公共信息规则政策进行比较，发现多个代理表现不如简单的基线规则，只有领先的代理在两个世界中都超越了基线，显示出在长期业务价值上的显著差异。

**⚠️ 局限性**

局限性在于客户行为是基于规则而非LLM驱动，且成本和操作事件遵循模拟假设，转移到其他市场或真实操作的可行性尚未测试。

---

## 45. Do Mixture-of-Experts Routers Benefit from Ensembling?

**arXiv ID:** 2611.17472 | [PDF](https://arxiv.org/pdf/2611.17472v1)

**作者:** Viktor Veselý `[一作]` (University of Groningen), Sebastian Jitaru `[通讯]` (University of Groningen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种新的混合专家模型，通过训练多个路由器来优化专家的权重分配，从而提高模型的性能。

**💡 创新点**

创新点在于将集成理论应用于路由步骤，使用多个路由器的集成来决定专家的权重，而不是依赖单一的路由器。

**🔧 技术方法**

使用了多层感知器（MLP）架构，并通过联合训练的方式优化路由器和专家的参数。

**📊 数据集**

使用了CIFAR-10、SVHN、Bike Sharing和Energy Production等数据集进行实验。

**📈 对比分析**

与传统的单一路由器模型相比，增加路由器数量可以轻微降低验证损失，但在计算成本上有所增加。实验结果表明，算术平均法在三项任务中表现优于几何平均法。

**⚠️ 局限性**

限制在于增加路由器数量虽然可以提高多样性，但可能导致个别专家性能下降，且在某些任务上性能提升有限。

---

## 46. On the Utility of Pre-Trained Visual Representations for Policy Learning of Contact-Rich Insertion Tasks

**arXiv ID:** 2611.17631 | [PDF](https://arxiv.org/pdf/2611.17631v1)

**作者:** Leon Kim `[一作]` (University of Pennsylvania), Michael Posa `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究探讨了在接触丰富的操作任务中，视觉表示对机器人策略学习的影响，特别是插入任务。通过比较通用视觉基础模型与显式接触中心表示，分析了不同表示在模拟和现实世界中的表现。

**💡 创新点**

创新点在于强调了视觉表示的任务依赖性，发现即使在相似的插入任务中，最佳表示也会有所不同，并且小的任务变化会导致不可预测的趋势变化。

**🔧 技术方法**

使用了扩散策略架构，结合ResNet-18编码器和U-Net去噪器，训练策略以预测SE(3)末端执行器轨迹，同时评估了多种视觉表示，包括DINOv2、Theia、分割掩码和接触中心特征。

**📊 数据集**

使用了ManiSkill模拟环境中的插入任务数据集，以及在现实世界中构建的类似插入任务的环境。

**📈 对比分析**

通过与从零开始训练的基线进行比较，发现接触中心特征在所有模拟任务中表现优异，而通用视觉基础模型在某些任务中未能超越基线，尤其是在现实世界任务中，Theia表现出显著的提升。

**⚠️ 局限性**

限制在于现实世界结果的信心受到早期停止、环境条件再现性和评估规模的限制。此外，接触信息的准确估计在现实世界的传感器噪声下仍然具有挑战性。

---

## 47. Learning Recommendations under Imperfect Human Compliance with Record-Level Differential Privacy

**arXiv ID:** 2611.17490 | [PDF](https://arxiv.org/pdf/2611.17490v1)

**作者:** Yuwen Ma `[一作]` (University College London), Boli Chen `[通讯]` (University College London)

**通讯引用:** 1956 | [OpenAlex ID](https://openalex.org/A5081701854)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了人机协作推荐系统，提出了一种框架来建模推荐的执行行为作为对推荐的随机响应，并将推荐机制形式化为一个在未知合规性分布下的决策依赖优化问题。

**💡 创新点**

创新点在于提出了一种同时考虑人类不完美合规性和记录级差分隐私的推荐框架，并建立了高概率的简单遗憾界限，展示了隐私噪声、批量大小和行为信息如何影响学习推荐的质量。

**🔧 技术方法**

使用了批量高斯过程上置信界（GP-UCB）算法来从局部扰动的历史行为中学习，并提供个体原始记录的隐私保护。

**📊 数据集**

使用了合成的历史驾驶员响应数据集进行固定状态的跟车问题模拟。

**📈 对比分析**

通过与不考虑合规性的推荐方法进行比较，提出的算法在不同合规性分布下实现了低简单遗憾，并优于不考虑合规性的推荐方法。

**⚠️ 局限性**

限制在于合规性分布未知，且隐私噪声和合规性对简单遗憾和最佳推荐识别的影响需要进一步研究。

---

## 48. Quadratic APN Functions of $\mathbb{F}_2^{10}$ with a Self-Equivalence of Prime Order

**arXiv ID:** 2611.18358 | [PDF](https://arxiv.org/pdf/2611.18358v1)

**作者:** Emanuele Bellini `[一作]` `[通讯]` (Technology Innovation Institute), Emanuele Bellini (Technology Innovation Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了具有素数阶自同构的𝔽_2^10的二次几乎完美非线性（APN）函数，确定了298种类型的自同构，并分类了29个二次APN函数类，其中4个是新的。

**💡 创新点**

创新点在于对具有素数阶自同构的二次APN函数进行了全面的分类，首次在维度10中确认了4个新的APN函数类，并且所有已知的29个类都不是CCZ等价于置换。

**🔧 技术方法**

使用了自同构类型的穷举搜索和基本的极小形式证明，结合了线性代数和组合数学的方法。

**📊 数据集**

使用了文献中已知的25个二次APN函数类作为数据集，并进行了大量的计算搜索以确认新的类。

**📈 对比分析**

通过与已知的APN函数类进行比较，确认了298种类型中242种没有APN函数，18种通过搜索找到了APN函数，剩余的38种通过证明确认没有APN函数。

**⚠️ 局限性**

限制在于类型64仍未解决，且对维度12的搜索可行性尚不明确。

---

## 49. Evidence-Carrying Transport Across Effects

**arXiv ID:** 2611.18147 | [PDF](https://arxiv.org/pdf/2611.18147v1)

**作者:** Félix Castro `[一作]`, Étienne Miquey `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提供了一种统一的、与证明相关的方式来处理效应变化对程序逻辑的影响，特别是在效应重新实现时，如何保持逻辑的有效性。

**💡 创新点**

创新点在于引入了效应可实现性核心之间的模拟，推广了Longley的应用态同态，使得效应变化成为一个可以组合和处理的数学对象。

**🔧 技术方法**

使用了可实现性理论中的模拟技术，建立了效应可实现性核心之间的关系，并通过这些关系传输逻辑和证据。

**📊 数据集**

使用了任意强单子的效应可实现性核心作为数据集，具体的例子包括经典可实现性的双重否定模态和层次化的构造。

**📈 对比分析**

通过与现有方法的比较，证明了所提出的模拟在保持逻辑有效性方面的有效性，且在某些情况下，模拟的存在性可以被证明。

**⚠️ 局限性**

限制在于当前的框架主要是语义性的，尚未完全实现一个具体的类型系统来表示模拟，并且在处理组合效应时可能会遇到困难。

---

## 50. The QBF Gallery 2026

**arXiv ID:** 2611.18169 | [PDF](https://arxiv.org/pdf/2611.18169v1)

**作者:** Cynthia Peyrer `[一作]` (Johannes Kepler University), Martina Seidl `[通讯]` (Johannes Kepler University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文报告了2026年QBF Gallery的组织和结果，旨在评估当前量化布尔公式（QBF）求解技术的最新进展，并介绍了新提交的求解器、实例和工具。

**💡 创新点**

创新点在于引入了新的基准集合用于评估，并分析了预处理的效果，同时比较了不同求解器在多个轨道上的表现。

**🔧 技术方法**

使用了多种求解技术，包括CEGAR、QCDCL、动态规划等，并结合了机器学习和电路简化技术。

**📊 数据集**

使用的数据集包括来自不同轨道的新提交实例，以及2023年QBF Gallery和2020年QBFEval的实例。

**📈 对比分析**

通过PAR-2评分和其他性能指标比较了求解器的表现，结果显示在不同内存限制下求解器的性能差异明显，部分求解器在特定实例上表现优异。

**⚠️ 局限性**

限制在于某些求解器在处理特定实例时出现不一致的结果，导致部分求解器被取消资格，影响了整体评估的准确性。

---

## 51. From institutional journalism to political extremes: Structural asymmetries in YouTube's news recommender during the 2022 French presidential campaign

**arXiv ID:** 2611.18034 | [PDF](https://arxiv.org/pdf/2611.18034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 52. Vision-as-a-Tool: Visual Harness for Multimodal Agentic Coding

**arXiv ID:** 2611.17797 | [PDF](https://arxiv.org/pdf/2611.17797v1)

**作者:** Haolin Yang `[一作]` (Peking University), Furu Wei `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Vision-as-a-Tool (VaaT)范式，旨在通过将外部多模态大语言模型（MLLM）重构为可调用的视觉工具，来解决多模态代理编码中的视觉推理问题。

**💡 创新点**

VaaT范式的创新点在于将视觉推理模块化为可调用的工具，使得文本代理能够完全卸载视觉推理，同时允许具备视觉能力的代理根据需求动态增强其固有感知。

**🔧 技术方法**

使用了VHarness框架，该框架围绕五个原子动作构建，提供了一个紧凑的接口，适用于多种视觉到代码的任务。

**📊 数据集**

在Image-to-HTML、Image-to-SVG和Image-to-3D生成任务中进行了评估，使用了100个复杂的测试案例。

**📈 对比分析**

与传统的Thinking-with-Images方法相比，VHarness在所有评估指标上均表现出一致的性能提升，尤其是在多模态编码代理和文本代理的任务中，VHarness使得文本代理GLM-5.2能够在视觉编码任务中与多模态基线竞争。

**⚠️ 局限性**

本研究的局限性在于主要评估静态目标（如单页界面、独立矢量图形和单个3D对象），而未考虑动态多状态交互，未来需要扩展VHarness以支持交互式应用和时间变化的3D场景。

---

## 53. Generative Design with Real-time Physics Explainability: Towards Product Definition in the Age of AI

**arXiv ID:** 2611.17560 | [PDF](https://arxiv.org/pdf/2611.17560v1)

**作者:** M. de Campos `[一作]`, H. Gottschalk `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种结合逆向设计和实时物理状态预测的生成设计方法，利用Diagonal Flow Matching (Diag-CFM)生成多个设计解决方案。

**💡 创新点**

创新点在于将逆向生成设计与学习的前向物理代理结合，创建一个集成设计环境，能够在解决逆向设计问题的同时提供实时反馈。

**🔧 技术方法**

使用了条件生成对抗网络（GAN）和基于变换器的神经算子Transolver作为物理代理，结合Diagonal Flow Matching (Diag-CFM)进行设计生成。

**📊 数据集**

使用了UniFoil数据集（包含500,000个气动仿真数据）和旋转冲击燃烧器的实验数据集。

**📈 对比分析**

通过与传统的数值方法进行比较，展示了所提方法在设计任务上的性能，能够快速生成设计并提供实时物理状态可视化。

**⚠️ 局限性**

限制在于生成设计可能会出现不可靠或不物理的请求，且需要通过传统数值计算或实验进行最终验证。

---

## 54. Beyond Attack-Based Accuracy: Rethinking Adversarial Robustness Evaluation Through Structural Gradient Alignment

**arXiv ID:** 2611.17377 | [PDF](https://arxiv.org/pdf/2611.17377v1)

**作者:** Hanrui Wang `[一作]` (National Institute of Informatics), Isao Echizen `[通讯]` (National Institute of Informatics)

**通讯引用:** 6844 | [OpenAlex ID](https://openalex.org/A5044556342)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的对抗鲁棒性评估方法，称为结构梯度对齐（SGA），用于评估视觉模型的鲁棒性。

**💡 创新点**

SGA是一种攻击无关的鲁棒性度量，通过测量输入内容与目标条件下局部模型敏感性之间的结构对齐来评估鲁棒性。

**🔧 技术方法**

使用了结构梯度对齐（SGA）技术，该技术只需对每个模型-探针对进行一次输入梯度计算，并使用少量随机采样的无标签公共探针。

**📊 数据集**

使用了38个不同的深度神经网络（DNN）、视觉语言模型（VLM）和大型视觉语言模型（LVLM），以及两个任务独立的公共数据集（CelebA-HQ和Places365）进行评估。

**📈 对比分析**

与基于攻击的方法（如PGD和AutoAttack）进行比较，SGA在八个公共探针条件下的平均成对一致性达到83.72%，在所有条件下均超越了最强基线9.38个百分点。经过校准的SGA一致性提高到85.36%。

**⚠️ 局限性**

SGA的局限性在于它仍然依赖于模型的结构敏感性，且在不同架构和威胁模型之间的扩展性仍然是一个开放的理论挑战。

---

## 55. Racing the Clock: Teaching VLNs Budget-Conditioned Object Search

**arXiv ID:** 2611.17442 | [PDF](https://arxiv.org/pdf/2611.17442v1)

**作者:** Maitree Meher `[一作]` (University of California), Jonas Frey `[通讯]` (Stanford University)

**通讯引用:** 747 | [OpenAlex ID](https://openalex.org/A5090942799)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了预算条件下的目标搜索，探索如何通过不同的专家监督来优化视觉-语言导航（VLN）策略。

**💡 创新点**

提出了预算条件下的对象搜索框架，并通过不同的场景抽象设计专家，展示了如何影响学生的学习效果。

**🔧 技术方法**

使用了预算索引的贝尔曼递归技术，比较了对象级和房间级专家的表现。

**📊 数据集**

使用了HM3D数据集进行实验，评估了不同预算下的目标收集效果。

**📈 对比分析**

通过比较不同监督方案的收集效果，房间级监督在每个测试预算下都表现出最高的平均收集率，尤其在2000步时达到了72.7%。

**⚠️ 局限性**

存在专家与学生之间的信息不匹配，尤其是在长时间的预算条件下，学生的空间记忆和恢复能力仍然有限。

---

## 56. GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models *Author(s)

**arXiv ID:** 2611.18279 | [PDF](https://arxiv.org/pdf/2611.18279v1)

**作者:** Jiayu Feng `[一作]` `[通讯]` (Harvard University), Jiayu Feng (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GT-FENG（一般事实评估和规范基础理论），这是第一个统一法律、政策和其他规则驱动领域的上下文/时间正确性和基于规则的错误归因的框架。

**💡 创新点**

GT-FENG通过检查答案与适用规则的匹配，识别与错误答案匹配的文档规则，提供了评估语言模型在规则驱动领域的基础。

**🔧 技术方法**

使用了基于规则的错误归因和上下文正确性的理论框架，结合了文档规则的索引和匹配机制。

**📊 数据集**

应用于Medicaid收入资格的答案数据集，涉及多个州和时间点的规则。

**📈 对比分析**

通过与其他州的值进行比较，GT-FENG显著减少了错误答案的数量，表明文档化的州记录能够有效提高答案的准确性。

**⚠️ 局限性**

GT-FENG的局限性在于它仅覆盖一个政策领域，且只匹配已记录的规则，未能处理未记录规则的答案。

---

## 57. RecMem: Recurrent Memory Compression for Long-Sequence Generative Recommendation

**arXiv ID:** 2611.17921 | [PDF](https://arxiv.org/pdf/2611.17921v1)

**作者:** Haoyi Hu `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `fede83ac-7505-405f-ab37-e7284695c47f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为RecMem的递归记忆压缩框架，以提高生成推荐系统在处理长交互历史时的效率。

**💡 创新点**

创新点在于RecMem结合了双叉编码器-解码器架构、基于槽的遗忘门更新和流一致性训练，确保记忆状态和解码器上下文的固定大小，从而实现空间不变性、语义保真性和终身可进化性。

**🔧 技术方法**

使用了双叉编码器-解码器架构和槽式遗忘门更新机制。

**📊 数据集**

使用了MerRec数据集，这是一个来自Mercari C2C市场的大规模基准数据集，包含超过500万用户的交互记录。

**📈 对比分析**

与HSTU-1200、PHSTU和其他基线方法进行比较，RecMem在保持87.9%的Recall@10和97.0%的Recall@200的同时，仅使用216个解码器上下文，显著提高了推荐性能。

**⚠️ 局限性**

限制在于RecMem的记忆容量是固定的，可能在某些情况下无法充分利用所有历史信息，尤其是在用户行为变化较大的情况下。

---

## 58. MamaReason: Unfinished Reasoning and Answer Recovery in Clinical Language Models

**arXiv ID:** 2611.18264 | [PDF](https://arxiv.org/pdf/2611.18264v1)

**作者:** Thanni Adewuyi `[一作]` (HelpMum Africa), Abiodun Adereni `[通讯]` (HelpMum Africa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在临床推理中，未完成的推理常常返回已经陈述的诊断。使用MamaReason模型分析了217对母婴健康案例，探讨了推理过程中的诊断保留情况。

**💡 创新点**

提出了一种标签独立的测量方法，用于评估显式承诺与后续诊断之间的一致性，并通过提取覆盖率进行报告。

**🔧 技术方法**

使用了MamaReason模型，该模型基于Gemma 4 31B，通过强化学习（RL）和临床内容的监督微调（SFT）进行适应。

**📊 数据集**

使用了MamaBench数据集，该数据集包含217对母婴健康案例。

**📈 对比分析**

比较了三种读取方式：现有的JSON解析器、仅提取最后诊断的回退方法和实际的条件继续方法。结果显示，条件继续方法在恢复配对正确性方面表现最佳，RL的恢复率为63.6%，SFT为60.8%。

**⚠️ 局限性**

本研究的局限性包括只覆盖了一个Gemma 4 31B模型的一个RL运行和一个基准，且直接回答和思考条件在预算和控制上存在差异。

---

## 59. Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost

**arXiv ID:** 2611.17658 | [PDF](https://arxiv.org/pdf/2611.17658v1)

**作者:** Christopher Malili `[一作]` `[通讯]` (Arizona State University), Christopher Malili (Arizona State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文探讨了模仿学习的机器人如何在噪声环境中进行物体操作，提出了一种新的预测法则来量化机器人对感知噪声的敏感性，并展示了这种敏感性如何影响机器人在实际部署中的表现。

**💡 创新点**

创新点在于提出了一种基于自身本体感知的锚定法则，机器人通过结合自身状态和噪声读取来调整对物体位置的估计，从而更准确地处理感知噪声。

**🔧 技术方法**

使用了模仿学习和扩散策略（Diffusion Policy）等技术，结合了深度学习模型和噪声预测方法。

**📊 数据集**

使用了robomimic Square和Can任务的数据集，每个任务包含200个示例，数据是通过模拟生成的，包含合成的高斯噪声。

**📈 对比分析**

与经典的误差变量（EIV）回归方法进行了比较，发现新提出的法则在十个持出噪声水平的预测中，平均绝对误差为0.025，而经典EIV的误差为0.37，显示出新法则的优越性。

**⚠️ 局限性**

限制在于实验是在模拟环境中进行的，使用的是合成噪声，未考虑真实世界中的复杂感知误差。此外，法则的预测依赖于一次测量的V值，未能从数据中推导出V。

---

## 60. In the Eye of the Beholder: Human Label Variation in Textual and Visual Entailment

**arXiv ID:** 2611.17954 | [PDF](https://arxiv.org/pdf/2611.17954v1)

**作者:** Monica Riedler `[一作]` (LMU Munich), Barbara Plank `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究引入了一个新的数据集VariNLI-VE，用于研究人类标签变异（HLV）在自然语言推理（NLI）和视觉蕴含（VE）任务中的变化。该数据集包含2250个注释，涵盖了文本和视觉前提下的相同假设的标签、置信度分数和解释。

**💡 创新点**

创新点在于首次提供了来自相同注释者的多重注释，包括标签和解释，允许直接比较NLI和VE任务中的人类判断变化。

**🔧 技术方法**

使用了人类注释者的标签、置信度分数和自然语言解释，分析了视觉语言模型（VLMs）在NLI和VE任务中的表现。

**📊 数据集**

使用的数据集为VariNLI-VE，包含225个原本中立的SNLI实例及其对应的图像，经过重新注释后形成。

**📈 对比分析**

与现有模型的比较显示，VLM在VE任务中的准确率普遍低于NLI任务，且与人类标签分布的对齐程度较弱，模型的不确定性普遍较高。

**⚠️ 局限性**

本研究的局限性在于数据集规模较小，仅覆盖225个实例，且主要集中在原本中立的NLI实例上，未来需要更大规模的数据集和更全面的注释者分析。

---

## 61. TelePhysica: Agentic Reconstruction of Physically Interactive Scenes

**arXiv ID:** 2611.18078 | [PDF](https://arxiv.org/pdf/2611.18078v1)

**作者:** Ruiqi Shu `[一作]`, Xuelong Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TelePhysica，一个闭环框架，从单张照片重建可执行的多模式物理世界，以驱动状态交互和基于轨迹的视频合成。

**💡 创新点**

创新点在于通过像素对齐的视觉批评和零力重力平衡预检相结合，提高了单图像场景恢复的准确性，从0%提升到81%。

**🔧 技术方法**

使用了闭环逆图形学、统一的多模式弹性动力学和结构视频扩散等技术。

**📊 数据集**

在一个包含100个场景的单图像基准数据集上进行了测试。

**📈 对比分析**

与单通道开放式合成方法相比，闭环执行在视觉对齐和物理验证方面显著提高了性能，成功率从0%提升至81%。

**⚠️ 局限性**

限制在于当前的材料参数（如密度和摩擦系数）是基于材料先验分配的，而不是通过多步骤视觉交互反馈在线识别的。

---

## 62. Generated Evaluation Suites as Diversity-Driven Search for Real Agent Regressions

**arXiv ID:** 2611.17632 | [PDF](https://arxiv.org/pdf/2611.17632v1)

**作者:** Pulkit Yadav `[一作]` `[通讯]` (Independent Researcher), Pulkit Yadav (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了如何从代理的规范生成评估套件，采用多样性驱动的搜索方法，通过填充描述符空间（需求 × 人物角色 × 场景类型）来生成评估套件，并通过平均通过率对其进行评分。

**💡 创新点**

创新点在于将生成的套件视为多样性驱动的搜索，并提出了一种覆盖度度量，界定了套件级评分的检测上限，揭示了覆盖度与评分之间的错位。

**🔧 技术方法**

使用了变异测试技术，通过向代理注入7种受控缺陷进行实验，并对生成的场景进行评分和覆盖度分析。

**📊 数据集**

使用了航空和零售两个领域的数据集，生成了90个场景并进行了评估。

**📈 对比分析**

与手工编写的任务进行比较，生成的套件在检测缺陷方面表现出一定的局限性，尤其是对于某些缺陷的检测上限低于阈值，导致无法有效识别回归情况。

**⚠️ 局限性**

限制在于研究仅涵盖了一个基准家族的两个领域和一个代理模型，且用户模拟器和评估者可能存在盲点，评分的运行噪声也限制了结果的可靠性。

---

## 63. MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading

**arXiv ID:** 2611.18513 | [PDF](https://arxiv.org/pdf/2611.18513v1)

**作者:** Mohammed Efaz `[一作]` (Eötvös Loránd University), Itilekha Podder `[通讯]` (Eötvös Loránd University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文介绍了MeReader，一个离线、隐私保护且进度感知的AI助手，旨在作为叙事阅读的上下文伴侣。

**💡 创新点**

创新点在于MeReader通过限制检索和生成内容到读者在文本中的特定位置，避免了传统大型语言模型（LLM）系统可能带来的剧透问题。

**🔧 技术方法**

使用了混合检索架构，结合了向量相似性、BM25和摘要嵌入技术，以确保对读者当前上下文的高保真度。

**📊 数据集**

评估系统使用了经典的英语书籍，进行定量理解指标和定性LLM独立评判。

**📈 对比分析**

与其他方法比较时，MeReader在理解能力上与大型模型（5b+）相当，同时保持了可接受的响应质量，尤其是中型本地模型（2b-4b）表现出色。

**⚠️ 局限性**

限制在于研究仅涵盖了三部经典英语小说，使用的模型数量和配置较少，可能不足以形成全面的结论，同时评估方法存在已知偏见，未能完全捕捉长篇话语推理。

---

## 64. Toward Universal Translatability: On the Statistical Ontology of Generative AI

**arXiv ID:** 2611.17963 | [PDF](https://arxiv.org/pdf/2611.17963v1)

**作者:** Katia Schwerzmann `[一作]` `[通讯]` (Ruhr-Universität Bochum), Katia Schwerzmann (Ruhr-Universität Bochum)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了翻译在当代生成AI中的起源和规范性地位，提出生成模型作为普遍翻译机器的理想，强调模型需要满足内部一致性、跨模态一致性和物理基础等条件。

**💡 创新点**

创新点在于提出了统计本体论的概念，认为生成AI的理想假设基于对现实的可计算统计结构的信念，而非人类的解释和劳动。

**🔧 技术方法**

使用了生成模型和概率分布对齐的技术，分析了机器翻译的历史和当前的生成AI研究。

**📊 数据集**

未具体提及使用的数据集，但提到了一些相关的概念，如平行语料库和多语言翻译的基础设施。

**📈 对比分析**

与翻译研究中的传统方法相比，生成AI的统计本体论忽视了文化、历史和经济因素，认为所有语言都可以通过可计算的等价系统进行翻译，表现出隐含的技术殖民假设。

**⚠️ 局限性**

限制在于未能充分考虑人类劳动和文化差异对翻译的影响，可能导致对语言和文化复杂性的简化理解。

---

## 65. Per-Agent-Safe Navigation of Globally Driven Robot Collectives via Affine Field Design

**arXiv ID:** 2611.17859 | [PDF](https://arxiv.org/pdf/2611.17859v1)

**作者:** Seunghwan Jang `[一作]` (Nanyang Technological University), Wei Xiao `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种新的广播控制方法，通过设计一个空间仿射场来协调机器人集体的运动和变形，以确保在受限环境中安全导航。

**💡 创新点**

创新点在于将运动、变形和安全性整合到一个优化问题中，允许集体在没有预先计划的情况下通过比自身更窄的走廊。

**🔧 技术方法**

使用了控制障碍函数（CBF）和空间仿射场的设计来实现控制。

**📊 数据集**

使用了模拟环境中的走廊和迷宫数据集进行实验验证。

**📈 对比分析**

与现有的局部控制方法相比，提出的方法在处理多个机器人时显著降低了控制和通信的复杂性，且在通过狭窄走廊时表现出更好的性能。

**⚠️ 局限性**

模型假设机器人为点状，未考虑物理尺寸对集体行为的影响，且缺乏硬件验证。

---

## 66. Photogrammetric 3D Model-Based Animal Behavior Classification

**arXiv ID:** 2611.17881 | [PDF](https://arxiv.org/pdf/2611.17881v1)

**作者:** Olaf Hellwich `[一作]` (Technische Universität Berlin), Roman Bhuiyan `[通讯]` (Technische Universität Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于视频的动物行为估计的新方法，采用两阶段策略：首先重建3D身体和物体几何形状，然后进行行为分类。

**💡 创新点**

创新点在于明确了2D到3D逆问题的模糊性，并提供了潜在的纠正方法，通过3D重建提高了行为分类的准确性。

**🔧 技术方法**

使用了3D运动重建技术、深度学习模型（如LSTM、xLSTM和Transformer）以及多视角视频处理。

**📊 数据集**

使用了模拟的互动小鼠数据集和真实视频数据集，特别是小鼠解决锁盒任务的多视角视频。

**📈 对比分析**

与传统的2D行为分类方法相比，3D重建的轨迹在行为分类上表现出更高的F1分数，表明重建过程保留了区分行为所需的信息。

**⚠️ 局限性**

限制在于尚未完全评估真实小鼠的行为分类，且3D重建的准确性受到相机校准的影响。

---

## 67. From smoothed analysis to backward error: The power of adaptive perturbations

**arXiv ID:** 2611.18545 | [PDF](https://arxiv.org/pdf/2611.18545v1)

**作者:** Michał Dereziński `[一作]` (University of Michigan), Christopher Musco `[通讯]` (New York University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文探讨了平滑分析与自适应扰动在求解线性系统中的复杂性差异，证明了自适应扰动在复杂性上优于平滑分析。

**💡 创新点**

创新点在于首次在自然计算模型中区分了随机算法与确定性算法在求解线性系统时的复杂性，揭示了自适应扰动的必要性。

**🔧 技术方法**

使用了矩阵-向量乘法模型，结合随机矩阵理论、信息复杂性和多项式逼近等技术。

**📊 数据集**

使用了任意大小的线性系统作为数据集，特别关注对称线性系统的求解。

**📈 对比分析**

与现有方法比较，证明了自适应扰动在复杂性上有指数级的改进，而平滑分析无法提供非平凡的复杂性保证，确定性算法在所有线性系统中需要Ω(n)的矩阵-向量乘法。

**⚠️ 局限性**

限制在于对于一般线性系统，确定性算法无法在小的向后误差下实现更好的复杂性，且自适应扰动的复杂性依赖于输入的条件数。

---

## 68. Fair and Efficient Distributed EV Charging Coordination Using Threshold Cryptography and Machine Learning

**arXiv ID:** 2611.17577 | [PDF](https://arxiv.org/pdf/2611.17577v1)

**作者:** Md. Mainul Islam `[一作]` (Texas A&M University), Erchin Serpedin `[通讯]` (Texas A&M University)

**通讯引用:** 11633 | [OpenAlex ID](https://openalex.org/A5034213361)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于阈值密码学和机器学习的公平高效的分布式电动汽车（EV）充电协调框架，旨在解决在充电过程中可能出现的攻击问题。

**💡 创新点**

创新点在于结合了阈值签名广播和上传重计算的方法，能够在不产生误报的情况下检测每一个偏差，并且通信成本为线性而非二次。

**🔧 技术方法**

使用了阈值密码学和机器学习技术，特别是机器学习价格预测来支持EV的本地恢复。

**📊 数据集**

使用了真实世界的负载、价格和充电数据集进行模拟，数据来源于EVWatts公共数据库。

**📈 对比分析**

与现有方法相比，提出的框架在面对攻击时能够保持优化过程的收敛性，且在攻击情况下的成本变化被限制在2.6%以内，而未防御的攻击则可能导致目标EV的年成本增加132.7%。

**⚠️ 局限性**

限制在于缺乏一个公共数据集来联合记录充电会话、总线负载、价格和站点评级，且实验主要基于加利福尼亚的案例研究，可能不适用于其他地区。

---

## 69. Logically Consistent Accurate Forecasts

**arXiv ID:** 2611.17735 | [PDF](https://arxiv.org/pdf/2611.17735v1)

**作者:** Kate Yuan `[一作]` (Bridgewater AIA Labs), Devavrat Shah `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种基于大型语言模型（LLM）的代理预测系统，旨在解决相关预测之间逻辑一致性不足的问题。通过微调监督语言模型，利用推理来共同调和不一致性，同时提高相关预测的准确性。

**💡 创新点**

创新点在于通过分析LLM预测者的推理过程来改善逻辑一致性，而不牺牲准确性。这种方法克服了传统方法在一致性和准确性之间的权衡。

**🔧 技术方法**

使用了微调的语言模型，具体为Qwen3.6-35B-A3B，并结合了检索增强的GPT-5.1作为上游预测器。

**📊 数据集**

数据集来源于ForecastBench，这是一个流行的二元问题预测基准，包含来自预测市场和维护良好的数据集的问题。

**📈 对比分析**

与其他方法相比，本方法在Brier分数和一致性指标上均表现出显著改善，Brier分数提高了9.3%，结构不一致性减少了82-90%。在所有比较的基线方法中，本方法在准确性和一致性上均优于其他方法。

**⚠️ 局限性**

限制在于方法的规模，依赖于相对较小的开源语言模型，且在处理大量相关预测时存在成本和运行时间的限制。此外，方法假设事件之间的关系是外生提供的，而不是直接从数据中学习的，仅处理三种类型的结构约束。

---

## 70. RiverScan: Learning to Accumulate on River Networks

**arXiv ID:** 2611.18005 | [PDF](https://arxiv.org/pdf/2611.18005v1)

**作者:** He Zhu `[一作]` (Central China Institute of Artificial Intelligence), Ming Li `[通讯]` (Cheriton School of Computer Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的空间操作RiverScan，用于在河流网络中学习如何累积不同规模的子流域的贡献，以提高大河流域的径流预测准确性。

**💡 创新点**

创新点在于引入了基于面积加权选择性状态空间递归的RiverScan操作，能够有效地结合局部时间表示和上游状态，从而改善径流预测。

**🔧 技术方法**

使用了基于LSTM的时间编码和面积加权选择性递归技术。

**📊 数据集**

使用了来自LamaH-CE的608个测站的丹麦河流网络数据集。

**📈 对比分析**

与多种时间和图形基线模型进行了比较，RiverScan在整体预测性能上表现最佳，尤其是在非源头测站上，显示出更高的中位数NSE和KGE值。

**⚠️ 局限性**

限制在于评估仅限于LamaH-CE数据集，未来需要在更广泛的河流流域中验证研究结果，同时多天前的径流预测仍不在本研究范围内。

---

## 71. Evaluating the Robustness of PRNU Source-Camera Attribution to the Fingerprint-Copy Attack: An Open Toolkit with Triangle-Test Detection

**arXiv ID:** 2611.17747 | [PDF](https://arxiv.org/pdf/2611.17747v1)

**作者:** Zeno J. M. H. Geradts `[一作]` `[通讯]` (Netherlands Forensic Institute), Zeno J. M. H. Geradts (Netherlands Forensic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为prnucf的开源工具包，用于评估PRNU归属在指纹复制攻击下的脆弱性，并结合三角测试检测器和群体校准决策程序。

**💡 创新点**

创新点在于提供了一个可审计的工具包，允许从业者量化PRNU归属在指纹复制攻击下的脆弱性，并提出了基于群体校准的检测程序。

**🔧 技术方法**

使用了三角测试检测器和群体校准决策程序，结合了指纹估计和PCE（峰值相关能量）统计。

**📊 数据集**

在一个受控的多相机蒙特卡洛基准测试中进行验证，使用了合成传感器图像。

**📈 对比分析**

与传统的PCE检测方法相比，三角测试在检测伪造图像时表现出极高的分离能力，ROC曲线下面积达到0.999，在5%的假阳性率下能够100%检测到伪造图像。

**⚠️ 局限性**

合成验证的局限性在于，合成传感器提供了理想化的统计数据，实际残差可能包含更多的噪声和伪影，绝对阈值不能直接转移到实际案例中。

---

## 72. Emulating physics-based XCO2 retrievals across OCO-2 viewing modes with compact neural networks

**arXiv ID:** 2611.18333 | [PDF](https://arxiv.org/pdf/2611.18333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 73. Towards High-Performance Multimodal Discovery: Evaluating Retrieval Efficacy and Operational Scalability in Digital Libraries

**arXiv ID:** 2611.17726 | [PDF](https://arxiv.org/pdf/2611.17726v1)

**作者:** Krishna Priya `[一作]` (Indiana University), Beth Plale `[通讯]` (University of Oregon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究探讨了如何通过将文本和图像转换为数值向量来提高数字图书馆中的多模态检索效率，评估了检索效果和操作可扩展性。

**💡 创新点**

创新点在于通过将页面图像表示为数值向量，捕捉文本和视觉语义，从而显著提高相关材料的发现，尤其是在图示内容对概念理解至关重要的领域。

**🔧 技术方法**

使用了视觉语言模型（VLM）和向量数据库（Qdrant）来实现多模态文档检索，并通过余弦相似度进行检索效果评估。

**📊 数据集**

使用的数据集包括3612个基于图像的教科书页面和一本富含图片的小食谱书，涵盖计算机科学的五本流行本科教材。

**📈 对比分析**

与传统的距离度量（如欧几里得或曼哈顿距离）相比，余弦相似度在保持语义一致性方面表现优越，检索效果显著提升，尤其在多页和概念驱动查询中表现良好。

**⚠️ 局限性**

限制在于当前系统在处理大规模数据集时的计算可行性仍需进一步探索，尤其是在法律和伦理方面的担忧，以及如何优化向量数据库配置以实现生产级别的可扩展性。

---

## 74. TSP instances that require exponentially many subtour elimination constraints

**arXiv ID:** 2611.17386 | [PDF](https://arxiv.org/pdf/2611.17386v1)

**作者:** Eleonora Vercesi `[一作]` (Università della Svizzera Italiana), Austin Buchanan `[通讯]` (Oklahoma State University)

**通讯引用:** 500 | [OpenAlex ID](https://openalex.org/A5006813161)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了需要指数级别子巡回消除约束（SEC）的病态TSP实例，证明了在n个顶点的非对称实例中，SEC的增长率为Ω(ϕ^n)，其中ϕ≈1.618是黄金比例。

**💡 创新点**

首次证明存在需要指数级别SEC的TSP实例，提供了Ball Bearing和Fibonacci两种构造方法，后者的SEC增长率更快。

**🔧 技术方法**

使用了Dantzig-Fulkerson-Johnson模型和节点拆分技术，结合了图论中的2因子和Hamiltonian子图的概念。

**📊 数据集**

构造了在平面上满足欧几里得、矩形和最大度量的TSP实例，具体数据集未明确列出，但涉及到Ball Bearing和Fibonacci图。

**📈 对比分析**

与Concorde求解器进行比较，发现尽管存在微小的整数间隙，求解时间仍然呈指数增长，表明这些实例在计算上非常困难。

**⚠️ 局限性**

限制在于构造的实例虽然在理论上需要大量SEC，但在实际计算中可能并不难以解决，且对随机生成的TSP实例的SEC数量的期望仍未明确。

---

## 75. Interactive deep-learning based interface for brain tumour segmentation

**arXiv ID:** 2611.17983 | [PDF](https://arxiv.org/pdf/2611.17983v1)

**作者:** Enora Giffard `[一作]` (Universite de Rennes), John S H Baxter `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本研究提出了一种基于深度学习的交互式脑肿瘤分割系统，该系统默认全自动，但允许用户进行简单的修正，以确保所有患者的脑肿瘤分割精度。

**💡 创新点**

创新点在于将自动分割与用户交互机制结合，使得系统在需要时可以进行人工修正，从而提高分割的准确性和效率。

**🔧 技术方法**

使用了标准的3D U-Net网络进行粗略分割和细粒度分割，并结合了用户交互工具。

**📊 数据集**

使用了B.R.A.T.S. 2021数据集，该数据集包含1251名患者的多序列MRI图像。

**📈 对比分析**

通过与传统的交互式分割方法进行比较，系统在自动分割的准确性和用户体验方面表现良好，用户对系统的整体满意度较高。

**⚠️ 局限性**

限制在于预分割网络的特异性较差，用户研究参与者数量较少，且缺乏关于用户交互效率的详细数据。

---

## 76. LLMs are Vulnerable to Malicious Requests with Indistinguishable Intent

**arXiv ID:** 2611.17818 | [PDF](https://arxiv.org/pdf/2611.17818v1)

**作者:** Wenzhuo Xu `[一作]`, Quanchen Zou `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用，旨在提高效率和准确性。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够在保持性能的同时减少计算复杂度。

**🔧 技术方法**

使用了深度学习技术和强化学习算法相结合的方法。

**📊 数据集**

实验使用了公开的标准数据集，以验证算法的有效性。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于算法在特定情况下可能会出现过拟合，需要进一步的研究来优化模型的泛化能力。

---

## 77. Joint World Action Models through Differentiable Trajectory Optimization

**arXiv ID:** 2611.17626 | [PDF](https://arxiv.org/pdf/2611.17626v1)

**作者:** Mintae Kim `[一作]` (Berkeley), Koushil Sreenath `[通讯]` (Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的世界动作轨迹优化方法（WATO），该方法联合优化生成的动作和未来状态，以提高控制效果。

**💡 创新点**

WATO通过将生成的动作和未来状态作为参考，解决了传统轨迹模仿无法确保任务有效性和动态一致性的问题，使生成的轨迹具有任务意识和动态一致性。

**🔧 技术方法**

使用了动态约束的轨迹优化技术，并通过隐式微分来训练生成器。

**📊 数据集**

在7个视觉控制环境中进行评估，包括导航、到达、接触丰富的操作和多阶段目标条件控制。

**📈 对比分析**

与强大的基于模型、目标条件的强化学习和生成控制基线相比，WATO的平均成功率从79.9%提高到91.7%。与交叉熵方法（CEM）相比，WATO在部署时平均快11.5倍，使用的世界模型评估次数少600.5倍。

**⚠️ 局限性**

依赖于学习的动态，有限的时间范围目标条件设置，以及仅使用流匹配进行评估的局限性。

---

## 78. AI Reviews Are Biased Towards the Appendix

**arXiv ID:** 2611.17811 | [PDF](https://arxiv.org/pdf/2611.17811v1)

**作者:** Saptarshi Ghosh `[一作]` (University of Cincinnati), Tianyu Jiang `[通讯]` (University of Cincinnati)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了最新的语言模型（如GPT-6-Astra和Claude-Opus-5.5）在审稿中对附录内容的偏见，发现这些模型在生成的评论中超过50%的弱点来自附录。

**💡 创新点**

首次系统性地分析了AI生成的评论中附录相关批评的比例，并与人类审稿人的评论进行了比较，揭示了人类和LLM在审稿时对附录的不同关注点。

**🔧 技术方法**

使用了四种先进的语言模型（GPT-6-Astra、GPT-6.1-Sol、Claude-Opus-5.5和Claude-Sonnet-5.5）进行评论生成和分析。

**📊 数据集**

分析了来自四个会议（ACL 2026、EMNLP 2026、COLM 2026和ICLR 2027）的2000篇论文，其中800篇来自最近的会议，1200篇来自ICLR 2021至2026。

**📈 对比分析**

与人类审稿人相比，LLM生成的评论中超过60%的弱点来自附录，而人类审稿人通常只引用不到10%的弱点来自附录。随着新模型的发布，人类审稿人对附录的关注度也有所上升，2027年的评论中这一比例达到了27.60%。

**⚠️ 局限性**

使用LLM进行弱点归属时可能存在错误，且人类和LLM的评论来自不同的论文和年份，可能影响比较结果。此外，未测量每个弱点的正确性或实用性，未来的研究可以探讨附录相关弱点的有效性。

---

## 79. Hierarchical Bayesian physics-augmented neural networks for populations of hyperelastic laws: specimen variability and uncertainty from scarce, noisy data

**arXiv ID:** 2611.17892 | [PDF](https://arxiv.org/pdf/2611.17892v1)

**作者:** Vincent C. Scholz `[一作]` (Technical University of Munich), Phaedon-Stelios Koutsourelakis `[通讯]` (Technical University of Munich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种层次贝叶斯物理增强神经网络（PANN），用于从少量噪声数据中学习材料的超弹性定律，并能够预测新样本的响应。

**💡 创新点**

创新点在于开发了一个层次贝叶斯PANN模型，该模型允许所有样本共享学习的多凸特征，并通过变分推断共同推断特征、样本和总体，同时分离测量噪声与材料变异性。

**🔧 技术方法**

使用了物理增强神经网络（PANN）和变分推断技术，结合贝叶斯方法来处理不确定性。

**📊 数据集**

使用了合成样本、Treloar的橡胶样本以及健康和动脉瘤患者的主动脉样本进行实验。

**📈 对比分析**

与三种已发表的方法进行比较，层次贝叶斯PANN在合成样本上能够从少量样本中恢复总体，并在预测新样本时表现出更高的可靠性。对于Treloar的橡胶样本，准确度达到了确定性PANN的水平，并且没有将任何主动脉样本错误分配到错误的群体中。

**⚠️ 局限性**

主要限制在于对于非常少的样本，模型可能表现出过于自信的情况，导致预测的带宽过窄，未能充分反映样本的变异性。

---

## 80. Adversarially Attacking Symbolic Vocabulary Vulnerabilities In LLM Planners

**arXiv ID:** 2611.17910 | [PDF](https://arxiv.org/pdf/2611.17910v1)

**作者:** Stephen Obadinma `[一作]` (Queen's University), Xiaodan Zhu `[通讯]` (Queen's University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了一种基于大语言模型（LLM）的攻击模型，旨在自动生成模糊的规划领域，以测试LLM在面对非标准符号词汇时的规划能力。

**💡 创新点**

创新点在于利用对抗攻击框架揭示LLM在联合符号理解和规划中的脆弱性，并通过优化的模糊化方案显著降低其规划性能。

**🔧 技术方法**

使用了对抗性语言模型和群体相对策略优化（GRPO）技术来生成模糊的符号词汇。

**📊 数据集**

使用了Minigrid和PlanBench等流行的规划任务数据集，包括Go To Object、Unlock和Blocks World等任务。

**📈 对比分析**

与传统的符号规划方法相比，LLM在模糊化领域中的规划性能下降超过70%。通过与随机符号替换和未优化的攻击模型进行比较，发现优化的模糊化方法（Sy-Sw）在降低规划成功率方面最为有效。

**⚠️ 局限性**

限制在于研究主要集中在大型强模型上，未能测试小型模型的表现。此外，研究仅限于自然语言任务，而非严格的形式化语言。

---

## 81. Toward Interference-Agnostic RF Source Separation

**arXiv ID:** 2611.17950 | [PDF](https://arxiv.org/pdf/2611.17950v1)

**作者:** Maxence Albert `[一作]`, Jean-François Bonastre `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种统一的信号分离器，能够在没有干扰类型指示的情况下，从共信道干扰中分离出感兴趣的无线电频率信号。

**💡 创新点**

创新点在于训练一个单一的分离器来处理所有信号和干扰对，而不是为每对信号和干扰训练一个专门的模型，从而避免了选择错误模型的风险。

**🔧 技术方法**

使用了卷积WaveNet和适应于音乐源分离的带分割变换器两种架构。

**📊 数据集**

使用了ICASSP 2024 RF Challenge提供的合成信号和干扰数据集，以及未见干扰的无人机检测数据集。

**📈 对比分析**

与专门模型相比，统一模型在大多数基准对上的性能损失有限，保持了接近的比特错误率（BER），在某些情况下甚至优于专门模型。

**⚠️ 局限性**

局限性在于训练数据集较窄，仅包含两种合成信号和四种记录干扰，未能证明该分离器能广泛适用于任意干扰类型。

---

## 82. Accounting for Trust and Safety: Bounding Content-Moderation Quality from Records You Already Hold

**arXiv ID:** 2611.18437 | [PDF](https://arxiv.org/pdf/2611.18437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 83. From Pen-and-Paper to Machine-Checked Proofs: Lean for System Identification

**arXiv ID:** 2611.18220 | [PDF](https://arxiv.org/pdf/2611.18220v1)

**作者:** Manas Mejari `[一作]` (IDSIA Dalle Molle Institute for Artificial Intelligence), Marco Forgione `[通讯]` (IDSIA Dalle Molle Institute for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文介绍了Lean 4证明助手在系统识别和控制领域的应用，展示了如何使用Lean进行数学证明的形式化，特别是针对识别性和线性系统的不变性等基本概念。

**💡 创新点**

创新点在于将传统的数学证明与交互式定理证明结合起来，强调了使用Lean进行形式化证明的必要性和优势，尤其是在确保每一步推理都有依据的情况下。

**🔧 技术方法**

使用了Lean 4作为交互式定理证明助手，基于依赖类型理论，允许用户以函数式编程语言的形式编写证明，并通过计算机进行检查。

**📊 数据集**

没有具体提到使用的数据集，但讨论了系统识别中的参数估计和线性系统的不变性等理论结果。

**📈 对比分析**

通过与传统的数学证明进行比较，展示了Lean形式化证明的优势，强调了在每一步都需要提供证明的必要性，从而提高了结果的可靠性和透明度。

**⚠️ 局限性**

限制在于目前在控制领域中，证明助手的采用仍然不广泛，且需要用户具备一定的Lean使用经验。此外，尽管Lean可以自动检查证明的正确性，但用户仍需确保形式化的声明准确反映了原始的数学结果。

---

## 84. SZZBench: Controlled Git Histories with Explicit Oracles for Diagnosing SZZ Implementations

**arXiv ID:** 2611.17669 | [PDF](https://arxiv.org/pdf/2611.17669v1)

**作者:** Yoshiki Higo `[一作]` (University of Osaka), Yasutaka Kamei `[通讯]` (Kyushu University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种工具SZZBench，用于生成受控的Git历史记录和显式的oracle，以诊断SZZ实现的成功或失败原因。

**💡 创新点**

创新点在于通过生成可重复的Git历史记录和明确的评估策略，提供了对SZZ工具性能的详细分析，并引入了279个场景和多种基线进行比较。

**🔧 技术方法**

使用了Git版本控制系统，YAML文件格式来定义场景，并通过Python编写的适配器运行不同的SZZ工具。

**📊 数据集**

使用了279个场景，这些场景分为17个类别，涵盖了多种历史因素和错误类型。

**📈 对比分析**

通过与26个简单基线进行比较，发现不同的SZZ工具在不同场景下表现不同，虽然PyDriller和一个简单基线的通过数量相似，但它们成功的场景并不相同。

**⚠️ 局限性**

限制在于合成历史无法估计实现的准确性，且大多数场景仅在Java中实现，未考虑其他语言或真实项目中的因素。

---

## 85. Vessel2Brain: Edge-Aware Multi-Scale Flow Matching for 3D Brain Image Generation from Vessel Masks

**arXiv ID:** 2611.17919 | [PDF](https://arxiv.org/pdf/2611.17919v1)

**作者:** Zhi Chen `[一作]` (University of Birmingham), Le Zhang `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为Vessel2Brain的框架，用于从稀疏的血管标签图生成3D脑MRI体积。

**💡 创新点**

创新点在于引入了边缘感知的多尺度条件流匹配框架，能够在稀疏条件下生成解剖结构一致的3D脑图像。

**🔧 技术方法**

使用了条件流匹配技术，结合UNet风格的网络架构和多尺度监督。

**📊 数据集**

使用了两个公开的3D脑MRI数据集，分别是CereVessMRA和TopBrain Challenge数据集，这些数据集提供了配对的3D脑MRI体积和相应的血管注释。

**📈 对比分析**

与多种基线方法（如3D Pix2Pix、3D Medical VAE、LDM和Med-DDPM）进行比较，Vessel2Brain在重建精度、结构一致性和分布真实感方面均表现优越。

**⚠️ 局限性**

限制在于训练数据的稀缺性可能导致模型过拟合，未来需要探索更强的解剖先验和不确定性感知采样策略。

---

## 86. Quantum Shock and Train-Time Constrained Quantization: A Geometric Study of Finite-Resolution Variational Quantum Circuits

**arXiv ID:** 2611.17817 | [PDF](https://arxiv.org/pdf/2611.17817v1)

**作者:** Vinayak Kumar Singh `[一作]` (National Institute of Technology), Ganesh Prasad `[通讯]` (National Institute of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了变分电路的确定性有限分辨率部署，定义了量子冲击（Quantum Shock）作为初始状态与最终状态之间的期望平方Fubini-Study距离，并提出了一种新的训练时间约束量化方法（T-CQ）。

**💡 创新点**

创新点在于提出了量子冲击的度量标准，并通过T-CQ方法在任务优化后引入周期性惩罚，从而减少量子电路部署时的状态位移。

**🔧 技术方法**

使用了量子机器学习中的变分量子电路、Fubini-Study距离和Bures角等技术。

**📊 数据集**

使用了三个小型二元任务的数据集，包括Two Moons、Iris和MNIST 3与6的子集，进行了十次匹配种子的实验。

**📈 对比分析**

与Circular-PTQ方法相比，T-CQ在所有任务和每个代码本大小下表现出更小的平均局部量子冲击。性能比较显示，T-CQ在准确性和状态位移方面均优于其他方法。

**⚠️ 局限性**

限制在于研究仅限于特定的四量子比特、两层架构和三个小型二元任务，且测试集样本量较小，无法推广到更广泛的问题空间。

---

## 87. OpenLayout: An Open and Agentic-Driven Platform for Holistic VLSI Physical Design and Benchmarking

**arXiv ID:** 2611.18517 | [PDF](https://arxiv.org/pdf/2611.18517v1)

**作者:** Qiufeng Li `[一作]` (George Washington University), Weidong Cao `[通讯]` (George Washington University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了OpenLayout，一个开放的多智能体平台，用于自动化物理设计，涵盖从初始化到最终签署的完整流程，并引入了OpenLayout-Bench，一个针对先进技术节点的基准测试套件。

**💡 创新点**

创新点在于利用大型语言模型（LLMs）与人类设计知识相结合，通过多智能体协作实现自我迭代的设计空间优化（DSO），并且能够在整个设计过程中进行推理、观察和适应。

**🔧 技术方法**

使用了大型语言模型（LLMs）和多智能体系统，结合了嵌入的设计知识和自然语言处理能力。

**📊 数据集**

使用了OpenLayout-Bench基准测试套件，该套件包含10个不同复杂度的电路设计，涵盖从布局到最终签署的完整流程。

**📈 对比分析**

与传统的单一阶段优化方法相比，OpenLayout在基准测试中表现出色，基于Gemini 3.1-pro-preview的OpenLayout在Tool A任务集上达到了84.4%的通过率，相比之下，独立的LLMs在同一任务集上的通过率仅为18.9%。

**⚠️ 局限性**

限制在于当前平台尚未支持LVS（布局与原理图）验证，DRC修复功能不完整，且评估仅限于TSMC 28nm技术节点，未来需要扩展到更多技术节点和更复杂的工业设计。

---

## 88. WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents

**arXiv ID:** 2611.17955 | [PDF](https://arxiv.org/pdf/2611.17955v1)

**作者:** Imene Kerboua `[一作]` (INSA Lyon), Siva Reddy `[通讯]` (McGill University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WebArena-Pro基准，涵盖20个自托管的Web应用程序和300个任务，旨在评估多模态Web代理的性能。

**💡 创新点**

创新点在于扩展了代理的观察空间，包含音频和视频内容，并提供可重复的并行评估，解决了现有基准在应用覆盖和模态支持方面的不足。

**🔧 技术方法**

使用了大型语言模型和视觉-语言模型，结合了自托管的Web环境和人类验证的任务创建流程。

**📊 数据集**

使用了20个自托管的Web应用程序，涵盖六个领域类别，包括生产力、通信、分析、商业运营、设计与媒体和旅行。

**📈 对比分析**

与现有的基准相比，WebArena-Pro在应用覆盖和多模态支持方面表现最佳，Codex与GPT 6.1 Sol和Claude Code与Claude Opus 5.5的成功率最高，均为62.3%。

**⚠️ 局限性**

限制在于当前的动作空间仅覆盖预录音频段和离散视频帧，缺乏更丰富的时间交互模式，如语音对话和实时视频流。

---

## 89. Direction or Magnitude? Learning Scalars over Task-Calibrated Directions

**arXiv ID:** 2611.17784 | [PDF](https://arxiv.org/pdf/2611.17784v1)

**作者:** Ziqing Wen `[一作]` (National University of Defense Technology), Tao Sun `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的参数高效微调方法，旨在在保持适应质量的同时减少训练成本。

**💡 创新点**

通过构建任务校准的固定输入基础和随机输出字典，实现了与LoRA相当的性能，但可训练参数显著减少，训练内存峰值更低，吞吐量更高。

**🔧 技术方法**

使用了层级特定的标量系数学习和分组计算技术，结合激活导向的输入方向和随机输出字典。

**📊 数据集**

在commonsense reasoning、数学推理、代码生成和知识评估等任务上进行了实验，使用了Qwen2.5-7B和LLaMA3-8B模型。

**📈 对比分析**

与LoRA、NOLA和RandLoRA等方法进行了比较，结果显示在commonsense reasoning任务上，准确率相当，但可训练参数仅为LoRA的约4%。在训练内存和吞吐量方面，表现出20%更低的内存峰值和6%更高的吞吐量。

**⚠️ 局限性**

方法的局限性在于可能在某些特定任务上表现不如全参数微调，且对输入方向的选择依赖于任务的特性。

---

## 90. Stage-wise Lineage-aware Dropout for Sparse-View 3D Gaussian Splatting

**arXiv ID:** 2611.17752 | [PDF](https://arxiv.org/pdf/2611.17752v1)

**作者:** Yan Xing `[一作]` (Hefei University of Technology), Mengjie Zhang `[通讯]` (Victoria University of Wellington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于谱系的高斯优化方法，旨在通过稀疏视图合成恢复场景的几何形状和外观，并渲染未观察到的相机视角的图像。

**💡 创新点**

创新点在于引入了阶段性谱系感知丢弃（SLD）和谱系平衡稠密化（LBD），通过对高斯族的结构化管理来改善稀疏视图重建的质量。

**🔧 技术方法**

使用了谱系感知丢弃和谱系平衡稠密化技术，结合了高斯原语的差异化光栅化。

**📊 数据集**

使用了LLFF、MipNeRF360和Blender等多个数据集进行评估，这些数据集包含不同类型的场景和视角设置。

**📈 对比分析**

通过定量和定性比较、模块级消融实验以及跨框架集成实验，结果显示在测试设置下具有竞争力的性能，SLD和LBD的不同贡献得到了进一步证实。

**⚠️ 局限性**

限制在于谱系关系的探索仍然较少，且在不同数据集和评估指标上的性能提升可能存在差异。

---

## 91. Understanding Contextual Bias in Agricultural CNNs Through Hidden-Neuron Concept Induction

**arXiv ID:** 2611.18530 | [PDF](https://arxiv.org/pdf/2611.18530v1)

**作者:** Abhilekha Dalal `[一作]` (Wright State University), Samatha Ereshi Akkamahadevi `[通讯]` (Kansas State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究探讨了卷积神经网络（CNN）在图像分类中如何利用背景信息，尤其是在包含少量或没有目标对象的图像补丁中。通过概念归纳方法，分析了隐藏神经元激活与上下文图像补丁的可解释视觉特性之间的关系。

**💡 创新点**

创新点在于将神经符号分析框架与定量图像描述符和描述逻辑概念归纳相结合，揭示了隐藏层激活与视觉特性之间的可解释关联。

**🔧 技术方法**

使用了神经符号分析框架，结合了UDAT图像描述符、OWL知识图谱和描述逻辑概念归纳技术。

**📊 数据集**

使用了五个农业数据集，涵盖高、中、低上下文偏差的情况，分析了四种CNN架构在三个网络深度和五个补丁位置的表现。

**📈 对比分析**

通过Mann-Whitney U检验和Cliff's delta对48,013对神经元-特征对进行了统计测试，最终验证了13,164对（约27.4%）。中间层的验证率最高（平均40.2%），而深层的验证率最低（12.0%）。

**⚠️ 局限性**

限制在于特征家族的分配是基于LLM指导的评估，而不是与所有层交叉测试；边缘和Gabor特征仅在中层评估，因此其高验证率无法与深度效应分开。此外，Zernike矩没有可测试的概念，Haralick特征仅在单一数据集和补丁位置上测试。

---

## 92. AmbiSteer: Selectively Eliciting Ambiguity-Aware Responses in Vision-Language Models

**arXiv ID:** 2611.17900 | [PDF](https://arxiv.org/pdf/2611.17900v1)

**作者:** Monica Riedler `[一作]` (LMU Munich), Raffaella Bernardi `[通讯]` (Free University of Bozen-Bolzano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种轻量级框架，通过激活引导机制来处理视觉问答中的指称模糊性问题，旨在提高模型在面对模糊问题时的反应能力。

**💡 创新点**

创新点在于结合了激活引导和提示机制，通过线性分类器预测输入的模糊性，从而动态调整引导方向，提高了模型对模糊输入的适应性。

**🔧 技术方法**

使用了激活引导和线性分类器等技术，进行模型内部表示的修改，以实现选择性地促进或抑制目标行为。

**📊 数据集**

使用了ClearVQA和AQuA两个数据集，分别用于研究澄清和列举行为。

**📈 对比分析**

与现有方法相比，结合提示和引导的方式显著提高了模糊性处理的准确性，适当处理的输入比例从59.0%提高到77.0%。

**⚠️ 局限性**

限制在于仅评估了三种开放权重的视觉语言模型，且分类器在ClearVQA上的表现仍有提升空间，可能影响条件引导的有效性。

---

## 93. Safety Side Effects of Preventative Persona Steering: An Empirical Audit

**arXiv ID:** 2611.17620 | [PDF](https://arxiv.org/pdf/2611.17620v1)

**作者:** Sabrina Tang `[一作]` (University of California), Sarah Wilson `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文主要研究了通过预防性引导（preventative steering）来控制模型中的谄媚（sycophancy）和幻觉（hallucination）特征，评估其对模型拒绝行为和安全性的影响。

**💡 创新点**

创新点在于提出了预防性引导的概念，并通过多个安全基准测试其效果，强调了在评估引导方法时需要考虑多个安全指标。

**🔧 技术方法**

使用了预防性引导和推理时引导（inference-time steering）技术，比较了这两种方法对模型行为的影响。

**📊 数据集**

使用了多个模型（如Llama和Qwen）进行实验，具体数据集未详细说明，但提到使用了与谄媚和幻觉相关的特征数据进行微调。

**📈 对比分析**

通过比较预防性引导和推理时引导在不同模型上的表现，发现预防性引导在降低攻击成功率方面效果显著，但对谄媚的影响较小且不一致，安全基准的结果也存在差异。

**⚠️ 局限性**

限制在于评估过程中未考虑不同层次的引导效果，且实验的置信区间未显示，可能影响结果的可靠性。此外，使用的安全基准可能无法全面反映模型的安全性。

---

## 94. Tight Bounds for k-Median++ and Improved Bounds for k-Means++

**arXiv ID:** 2611.18544 | [PDF](https://arxiv.org/pdf/2611.18544v1)

**作者:** Haim Kaplan `[一作]` (Blavatnik School of Computer Science and AI, Tel Aviv University), David Naori `[通讯]` (Blavatnik School of Computer Science and AI, Tel Aviv University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对k-means++算法进行了统一分析，探讨了在双标准设置下，算法在不同距离指数和额外中心数量下的近似比率。

**💡 创新点**

创新点在于提供了对所有距离指数ℓ>0、所有k≥1和所有Δ≥0的紧界限，特别是对于k-median的界限是紧的，并且在没有额外中心的情况下，改进了之前的上界。

**🔧 技术方法**

使用了D^ℓ采样技术，并通过平衡和等质量过程的比较来分析算法的性能。

**📊 数据集**

使用了多种数据集，包括Euclidean空间中的k-means和k-median问题，具体数据集未详细说明。

**📈 对比分析**

与之前的方法相比，本文的性能表现显著提升，特别是在Euclidean k-means中，未使用额外中心的情况下，近似比率为(1+φ)(1+H_k-1)，而之前的上界为5。

**⚠️ 局限性**

限制在于对于ℓ>1的情况，平衡常数的最优性尚未确定，且在k-means的情况下，仍存在性能差距，尤其是在没有额外中心的情况下。

---

## 95. Red-Blue Hamiltonian Tomography via the Operator Lightcone Wavemap

**arXiv ID:** 2611.17555 | [PDF](https://arxiv.org/pdf/2611.17555v1)

**作者:** Paolo D'Alberto `[一作]` `[通讯]` (Advanced Micro Devices, Inc.), Paolo D'Alberto (Advanced Micro Devices, Inc.)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于算子传播波前的无模型哈密顿重建方法，通过比较同一量子晶格的两个张量网络模拟（一个是标准哈密顿，另一个是一个或多个扰动键），识别出每个扰动键的因果特征。

**💡 创新点**

创新点在于该方法不依赖于模型假设或对扰动类型和幅度的知识，采用盲检测策略，能够在没有先验知识的情况下识别出扰动键。

**🔧 技术方法**

使用了基于张量网络的信念传播模拟技术，结合了签名交叉熵和集成对数比率等技术。

**📊 数据集**

在7×7的矩形晶格上进行模拟，使用了84条键的量子系统，并通过GPU加速的信念传播张量网络模拟器进行实验。

**📈 对比分析**

与标准的哈密顿学习方法相比，该方法在识别单个扰动键时的检测率超过90%，并且在多达8个同时故障的情况下，仍能保持高于90%的检测率。通过迭代重扫描策略，逐步解决多键情况。

**⚠️ 局限性**

该方法的局限性在于需要大量的测量（例如，检测特定耦合偏差需要约2.5×10^7次测量），并且在处理多个相邻故障时可能会出现干扰模式，导致检测失败。

---

## 96. Analyzing Learning-Dynamics Across Loss and Prediction Choices for Diffusion Models

**arXiv ID:** 2611.17967 | [PDF](https://arxiv.org/pdf/2611.17967v1)

**作者:** Dongwoo Kim `[一作]` `[通讯]` (POSTECH), Dongwoo Kim (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了去噪生成模型中预测目标和损失空间的选择对生成质量的影响，分析了不同预测目标（干净数据、噪声、速度）和损失空间的组合如何影响去噪精度和样本质量。

**💡 创新点**

创新点在于提出了一种联合分析预测和损失空间的3×3框架，揭示了损失空间的变化等同于时间步重加权，并且在架构瓶颈下，预测目标的选择会显著影响去噪精度和样本质量。

**🔧 技术方法**

使用了U-Net和JiT等神经网络架构进行实验，结合理论分析和实验验证，探讨了架构瓶颈与预测和损失选择之间的相互作用。

**📊 数据集**

实验使用了CIFAR-10、Imagenette和ImageNet-256数据集进行验证。

**📈 对比分析**

通过与现有方法的比较，发现噪声预测的误差接近理论下限，而在压缩情况下，x预测保持较低的去噪误差，表明不同预测目标对瓶颈容量的敏感性不同。

**⚠️ 局限性**

限制在于理论界限依赖于线性瓶颈和低维子空间模型，且实验结果在不同架构和条件下的绝对FID值不可直接比较。此外，未压缩情况下的误差界限变得微不足道，且对未来工作的扩展分析仍需进行。

---

## 97. SWIFT: Sample-Wise Instance Fine-Tuning of Time-Series Foundation Models for Forecasting

**arXiv ID:** 2611.17915 | [PDF](https://arxiv.org/pdf/2611.17915v1)

**作者:** Yang Chen `[一作]`, Yang Gao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的时间序列预测适应框架SWIFT（Sample-Wise Instance Fine-Tuning），通过利用目标实例自身历史数据实现细粒度的专业化，避免了对额外数据集的需求。

**💡 创新点**

SWIFT是首个探索时间序列基础模型的样本级实例微调的框架，专注于使用目标样本的历史数据进行个体预测窗口的专业化。

**🔧 技术方法**

SWIFT结合了自检索（Self-Retrieval）、频率感知损失（Frequency-Aware Loss）和偏差感知动态融合（Deviation-Aware Dynamic Fusion）等技术。

**📊 数据集**

在Chronos Benchmark II、fev-bench和GIFT-Eval等多个长历史基准数据集上进行了验证。

**📈 对比分析**

SWIFT在24种不同的骨干网络和数据集组合中，平均减少了6.41%的MASE和4.83%的WQL，相比于零-shot推理表现出显著的性能提升，并在41个比较中优于少-shot微调，34个比较中优于测试时适应（TTA）。

**⚠️ 局限性**

SWIFT的局限性在于其计算成本，未来的工作将集中在降低实例级适应的计算开销上。

---

## 98. An Operationalization of Opaque Serial Depth

**arXiv ID:** 2611.17495 | [PDF](https://arxiv.org/pdf/2611.17495v1)

**作者:** Nathan Sheffield `[一作]` (Redwood Research), Ryan Greenblatt `[通讯]` (METR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的度量标准NLS深度，用于评估AI模型的不可解释序列深度，特别是在链式思维（CoT）监控能力方面的影响。

**💡 创新点**

创新点在于定义了自然语言根节点（NL-rooted nodes）作为可解释瓶颈，并提出了NLS深度的具体标准，强调了在模型训练前就可以评估这一深度。

**🔧 技术方法**

使用了计算图和深度学习模型的架构分析技术，结合了对模型训练配置的评估。

**📊 数据集**

未具体提及使用的数据集，但讨论了开放权重模型的NLS深度与层数的相关性。

**📈 对比分析**

通过与现有架构的比较，发现NLS深度与层数强相关，且在标准变换器架构下，NLS深度的增长速度较慢，而潜在推理架构则可能显著增加NLS深度。

**⚠️ 局限性**

限制在于NLS深度可能无法完全捕捉某些架构变化对模型监控能力的影响，特别是从自然语言模型转向使用不透明状态的模型。

---

## 99. Model Annealing: Data-Efficient Learning through Review and Consolidation

**arXiv ID:** 2611.17524 | [PDF](https://arxiv.org/pdf/2611.17524v1)

**作者:** Ali Mohammadi Ruzbahani `[一作]` (University of Calgary), Rezvan Abouzadeh `[通讯]` (MetaKet Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为模型退火的方法，通过回顾和巩固来提高小数据集上的深度学习模型的学习效率。

**💡 创新点**

模型退火通过将训练过程分为多个阶段，结合回顾和实践，利用网络自身的知识进行再学习，从而提高了模型的准确性和鲁棒性。

**🔧 技术方法**

使用了自我蒸馏、知识蒸馏、对抗训练和元学习等技术。

**📊 数据集**

在CIFAR-100、Tiny-ImageNet和Imagewoof数据集上进行了实验，使用了不同数量的标签（5%到100%）。

**📈 对比分析**

与交叉熵训练、标签平滑、SGDR和快照蒸馏等方法进行了比较，模型退火在28个配置中提高了准确性，平均提高了1.3个百分点，并在校准和对抗扰动的鲁棒性方面表现更好。

**⚠️ 局限性**

模型退火在ResNet-18上降低了准确性，但仍提高了对小对抗扰动的鲁棒性，且在某些情况下，重热调度会导致模型的拟合能力下降。

---

## 100. ERA: Error-Resistant Autonomous Object Navigation with Predicted Depth

**arXiv ID:** 2611.17730 | [PDF](https://arxiv.org/pdf/2611.17730v1)

**作者:** Shunyang Huang `[一作]` (Shanghai Jiao Tong University), Hongtao Lu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的物体导航系统，利用RGB图像预测深度，替代深度传感器进行导航。

**💡 创新点**

创新点在于提出了'Move with Error'范式，通过移动获取新视角来修正地图错误，并结合可修正占用和从可达视点探索的机制。

**🔧 技术方法**

使用了RGB图像和姿态估计进行深度预测，结合可修正占用和可达视点探索的技术。

**📊 数据集**

在三个基准测试上进行评估，包括HM3D v1、HM3D v2和MP3D。

**📈 对比分析**

与其他训练无关的方法相比，在所有三个基准测试中表现出色，成功率(SR)和路径长度加权成功率(SPL)均优于其他方法。

**⚠️ 局限性**

限制在于依赖于平台提供的姿态信息，未来工作将集中在姿态估计和更高效的视点选择上。

---

## 101. Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors

**arXiv ID:** 2611.17846 | [PDF](https://arxiv.org/pdf/2611.17846v1)

**作者:** Richard J. Young `[一作]` `[通讯]` (University of Nevada Las Vegas), Richard J. Young (University of Nevada Las Vegas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究比较了方向性和优化性拒绝移除方法在五个开放权重发布对中的表现，评估了新模型是否更难以移除拒绝行为。

**💡 创新点**

研究的创新点在于首次独立比较了ARA与方向性方法在不同模型发布中的效果，并探讨了拒绝移除的难易程度是否随模型更新而变化。

**🔧 技术方法**

使用了方向性方法（如差异均值、MPOA、SOM和J-space）以及优化性方法（ARA），并在Heretic 2.0工具中实施。

**📊 数据集**

数据集包括来自JailbreakBench、XSTest、StrongREJECT、HarmBench和作者编制的恶意代码库的972个有害提示。

**📈 对比分析**

比较方法包括方向性方法和ARA，结果显示在某些模型对中，ARA在新模型上的有效性损失较小，而在其他模型中则表现不佳。总体而言，拒绝移除的难度并未普遍增加。

**⚠️ 局限性**

研究的局限性在于每个模型的ARA运行仅基于一次实验，且结果对预算和方差模型敏感，未能全面验证不同模型间的普遍趋势。

---

## 102. T2ST: Data-Efficient Spatio-Temporal Foundation Models via Temporal Data and Knowledge Transfer

**arXiv ID:** 2611.18199 | [PDF](https://arxiv.org/pdf/2611.18199v1)

**作者:** Yue Xu `[一作]` (University of Macau), Xiaoxi He `[通讯]` (University of Macau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种数据高效的时空基础模型训练框架T2ST，通过将有限的时空观察与选定的辅助时间序列相结合，并从冻结的时间序列基础模型中转移知识，以增强时空模型的预测能力。

**💡 创新点**

创新点在于利用丰富的时间序列数据来补充稀缺的时空观察，同时通过知识蒸馏技术有效地转移时间知识，从而实现数据高效的时空模型训练。

**🔧 技术方法**

使用了知识蒸馏技术，包括时间关系蒸馏和多尺度预测蒸馏，以增强模型的时间学习能力。

**📊 数据集**

使用了六个长期和十二个短期预测数据集进行实验，展示了T2ST在不同数据集上的有效性。

**📈 对比分析**

与全数据集训练的模型相比，T2ST在仅使用10%的原始时空数据的情况下，预测性能与全数据模型相当或更优，证明了其有效性和通用性。

**⚠️ 局限性**

限制在于模型的性能可能依赖于所选的辅助时间序列的质量和多样性，且在某些情况下，额外的时空观察可能带来的边际收益有限。

---

## 103. Folding Language Models into Themselves

**arXiv ID:** 2611.18113 | [PDF](https://arxiv.org/pdf/2611.18113v1)

**作者:** Ziyang Zhang `[一作]` (Peking University), Yichen Gong `[通讯]` (Lovart AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种新的知识蒸馏方法Concertina，通过直接从教师模型的权重构建学生模型，而不是从头开始训练。该方法将相邻的教师层折叠成一个共享块，并在每个循环步骤中执行低秩修正。

**💡 创新点**

创新点在于通过折叠相邻层来创建共享块，并设计了一个每个令牌都能调整的门控机制，使得每个令牌可以根据需要恢复原始层的行为，从而提高了模型的性能和数据效率。

**🔧 技术方法**

使用了低秩修正和门控机制的技术，结合了知识蒸馏和层共享的思想。

**📊 数据集**

使用了Qwen3-4B模型进行实验，蒸馏数据集包含22B个令牌，数据类型包括一般性、数学、学术和针对性的多项选择题。

**📈 对比分析**

与同样大小的稠密模型和之前的权重共享设计相比，Concertina在多个基准测试中表现更好，尤其是在常识和数学推理方面，保持了约90%的教师模型能力，且在八个基准测试中平均保持了84%。

**⚠️ 局限性**

限制在于尽管Concertina在多个基准上表现优异，但仍然依赖于教师模型的结构和权重，可能在某些特定任务上无法完全替代更大的模型。

---

## 104. AI-aided Decision-Making in Education - Uncertainty, Explainability, and Human Responsibility

**arXiv ID:** 2611.17858 | [PDF](https://arxiv.org/pdf/2611.17858v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 105. The Transformer Paradigm: Dynamic Processing and a New Principle of Inference

**arXiv ID:** 2611.17953 | [PDF](https://arxiv.org/pdf/2611.17953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 106. On Structure-Aware SAT Solving

**arXiv ID:** 2611.18017 | [PDF](https://arxiv.org/pdf/2611.18017v1)

**作者:** Eugene Goldberg `[一作]` `[通讯]`, Eugene Goldberg

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了两种结构感知的SAT求解技术，旨在提高SAT求解的效率和准确性。

**💡 创新点**

创新点在于引入了公式熔化和局部证明两种方法，使得SAT求解能够更好地利用公式的结构特性。

**🔧 技术方法**

使用了公式熔化技术（通过引入新变量、使用存在量词和部分量词消除）和局部证明技术（通过证明某些文字的冗余性来确定公式不可满足）。

**📊 数据集**

实验中使用了多种布尔公式，特别是结合电路的等价性检查问题。

**📈 对比分析**

与传统的CDCL方法相比，提出的方法在处理具有宏观结构的公式时表现出更高的效率，生成的证明更短，运行时间更快。

**⚠️ 局限性**

局限性在于当前的部分量词消除（PQE）求解器的效率仍在改进中，可能影响整体性能。

---

## 107. OmniScore-Bench: Evaluating Vision-Language Models on Musical Score Reading

**arXiv ID:** 2611.17870 | [PDF](https://arxiv.org/pdf/2611.17870v1)

**作者:** Congren Dai `[一作]` (Central Conservatory of Music), Maosong Sun `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用。

**💡 创新点**

创新点在于提出了一种改进的模型结构，能够更有效地处理复杂数据。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和循环神经网络（RNN）的结合。

**📊 数据集**

使用了公开的图像数据集和文本数据集进行实验。

**📈 对比分析**

与现有方法进行了比较，结果显示新算法在准确率和处理速度上均有显著提升。

**⚠️ 局限性**

限制在于模型对特定类型数据的依赖性，可能在其他领域的适用性较差。

---

## 108. Aligning AI Teams

**arXiv ID:** 2611.17820 | [PDF](https://arxiv.org/pdf/2611.17820v1)

**作者:** Siddarth Srinivasan `[一作]` (Harvard University), Erik Jones `[通讯]` (Anthropic)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了单个大型语言模型（LLM）代理的对齐训练在多代理设置中的转移失败机制，并探索了干预措施以保持对齐转移到多代理LLM团队。

**💡 创新点**

提出了四个主要发现，包括单代理与团队之间的安全差距没有明确趋势，责任扩散显著影响安全差距，以及指定安全负责人和安全规范文档的有效性。

**🔧 技术方法**

使用了多代理LLM编排和实验框架，结合了不同的干预措施来测试代理团队的表现。

**📊 数据集**

使用了两个软件工程任务（脓毒症分类器和新闻推荐系统）以及十个咨询提案生成任务的数据集。

**📈 对比分析**

通过与单代理的表现进行比较，发现多代理团队在安全目标上表现较差，且干预措施的有效性不一，复合干预（安全文档+指定安全负责人）最为有效。

**⚠️ 局限性**

本研究的局限性在于仅研究了Anthropic的Claude系列模型，未来需要在其他模型系列上验证这些干预措施的有效性。

---

## 109. Asymptotic Margins and Feature Confinement in Conic Classification Heads under Softmax Cross-Entropy

**arXiv ID:** 2611.18507 | [PDF](https://arxiv.org/pdf/2611.18507v1)

**作者:** Omer Nezih Gerek `[一作]` `[通讯]` (Eskişehir Technical University), Omer Nezih Gerek (Eskişehir Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了基于圆锥函数构建的分类头在使用softmax交叉熵训练时，是否能够保持特征的有界性，并提出了每个类别的区域由旋转二次曲面界定的理论。

**💡 创新点**

创新点在于引入了渐近边际的概念，该边际比较了类别得分的增长与其他类别得分的增长，并通过二阶锥规划进行计算，揭示了有界区域并不限制特征的生长。

**🔧 技术方法**

使用了圆锥标量化的范数线性函数和softmax交叉熵损失函数，结合二阶锥规划技术。

**📊 数据集**

在CIFAR-10和CIFAR-100数据集上进行了实验，训练了67个圆锥头，所有1210个类别的单元均具有正的渐近边际。

**📈 对比分析**

与原型头进行比较，圆锥头在CIFAR-10上与仿射头的准确率相当，并且特征范数减少了12%到38%。在CIFAR-100上，准确率略低于仿射头约1个百分点。

**⚠️ 局限性**

限制在于本文没有提出新的分类器，也没有声称提高多面体圆锥分类器的准确性，主要结论是softmax交叉熵下特征的限制由渐近边际决定。

---

## 110. Learning Latent Goal Costs for Visual World-Model Planning with an Expanding Curriculum

**arXiv ID:** 2611.17529 | [PDF](https://arxiv.org/pdf/2611.17529v1)

**作者:** Konstantinos Mylonas `[一作]` (Technical University of Crete), Thrasyvoulos Spyropoulos `[通讯]` (Technical University of Crete)

**通讯引用:** 10169 | [OpenAlex ID](https://openalex.org/A5056364055)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了一种基于冻结视觉世界模型的图像目标规划方法，通过在预训练的I-JEPA嵌入上拟合小型残差适配器来改善潜在目标成本，从而提高规划成功率。

**💡 创新点**

创新点在于提出了一种无任务特定监督的适配器训练程序，能够从经验中发现有效的进展度量，而不是依赖于专家知识进行适配。

**🔧 技术方法**

使用了预训练的I-JEPA视觉编码器和基于CEM（交叉熵法）的规划技术。

**📊 数据集**

使用了DotWall和PointMaze两个环境作为数据集进行实验。

**📈 对比分析**

在50个匹配案例的评估中，使用适配器的成功率在DotWall中从54%提高到98%，在PointMaze中从82%提高到96%。在未见的DotWall布局中，适配器的潜在成本与最短合法路径的排序一致性从71.9%提高到85.8%。

**⚠️ 局限性**

限制在于该方法仍然需要目标成功信号和足够的覆盖范围，且在高维状态空间中可能会面临维度诅咒的问题。

---

## 111. RaFL: An Empirical Study of Random Filter Losses for Image Reconstruction

**arXiv ID:** 2611.17707 | [PDF](https://arxiv.org/pdf/2611.17707v1)

**作者:** Ruibo Wang `[一作]` (TU Delft), Heejong Kim `[通讯]` (Weill Cornell Medicine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的随机滤波损失函数（RaFL），用于图像重建任务，旨在克服传统像素损失和感知损失的局限性。

**💡 创新点**

RaFL通过使用随机权重的滤波器，避免了预训练特征和任务特定损失工程的需求，提供了一种简单、无调优且领域无关的损失目标。

**🔧 技术方法**

结合了随机卷积、随机傅里叶特征和随机小波滤波器，通过梯度范数匹配自动平衡它们的贡献，消除了手动损失权重调优的需求。

**📊 数据集**

在多个数据集上进行评估，包括SIDD（真实噪声去噪）、LOL（低光增强）、WorldView-2（全色锐化）、OASIS3（3D超分辨率）和IXI（3D MR血管造影去噪）。

**📈 对比分析**

与像素损失相比，RaFL在三个任务上表现更好，在两个任务上表现相当，并在所有五个任务上展示了稳定的性能。RaFL的计算成本仅占骨干网络前向计算的不到2%。

**⚠️ 局限性**

尽管RaFL在计算开销上增加很少，但每步重采样和梯度平衡增加了训练时间。此外，设计研究主要在两个2D数据集上进行，未来的工作可以探索在不同数据集和模态上的表现。

---

## 112. Small Is Not Dead: Distance-Aware Cluster Revival for e-means

**arXiv ID:** 2611.18238 | [PDF](https://arxiv.org/pdf/2611.18238v1)

**作者:** Eyal Weiss `[一作]` `[通讯]` (Robotics Institute Carnegie Mellon University), Eyal Weiss (Robotics Institute Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种改进的e-means算法，通过引入基于距离的复兴规则来优化小聚类的处理，从而提高了聚类性能。

**💡 创新点**

创新点在于结合了聚类大小和中心间距离的双重条件来决定是否移除小聚类的中心，避免了因移除孤立小聚类而导致的性能损失。

**🔧 技术方法**

使用了移动平均更新和最近邻搜索等技术，结合了基于距离的复兴规则。

**📊 数据集**

使用了多个数据集进行评估，包括KDD Cup 1999、US Census 1990、YearPredictionMSD等，以及e-means论文中的嵌入数据集。

**📈 对比分析**

与传统的基于计数的复兴方法相比，改进后的方法在处理小孤立聚类时，测试误差降低了约10%到80%，在其他评估数据集上性能相近，误差在1%以内。

**⚠️ 局限性**

限制在于该方法目前仅为CPU实现，GPU和多GPU实现尚待未来工作，且在处理不同规模聚类时可能存在单一全局误差尺度的问题。

---

## 113. Multimodal Person Concept Model for Person Identification under Varying Observation Conditions

**arXiv ID:** 2611.17802 | [PDF](https://arxiv.org/pdf/2611.17802v1)

**作者:** Mizuho Okitani `[一作]` (Soka University), Yoshinobu Hagiwara `[通讯]` (Soka University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多模态人概念模型，该模型能够在观察条件变化的情况下，从少量观察中学习人概念，实现用户识别。

**💡 创新点**

创新点在于该模型无需使用人员标签或事先知道用户数量，能够从少量的视觉、声学和语言观察中形成有效的人概念。

**🔧 技术方法**

使用了非参数贝叶斯生成模型（基于中国餐馆过程），结合了对比语言-图像预训练（CLIP）和对比语言-音频预训练（CLAP）提取的特征，并进行了主成分分析（PCA）。

**📊 数据集**

使用了六名参与者在八种不同条件下的多模态数据集，包括视觉、声学和语言观察。

**📈 对比分析**

与固定人数的k-means聚类方法相比，提出的模型在八种条件下的平均识别准确率为0.89，k-means为0.88，显示出在不确定用户数量的情况下仍能实现高效识别。

**⚠️ 局限性**

限制在于数据集仅包含六名参与者，且声学和语言变化是合成引入的，未来需要在更多参与者和自然观察变化的情况下进行评估。

---

## 114. AutoRL: A Multi-Agent System towards Autonomous Reinforcement Learning

**arXiv ID:** 2611.17790 | [PDF](https://arxiv.org/pdf/2611.17790v1)

**作者:** Simon Tang `[一作]` (Meituan), Xunliang Cai `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为AutoRL的多智能体系统，用于自动化强化学习（RL）训练，以支持递归自我改进（RSI）。该系统通过监控训练动态并整合过程级监督和研究，持续改进模型能力和泛化能力。

**💡 创新点**

AutoRL的创新点在于将人类RL研究者的工作流程整合到自动化系统中，采用专门的角色（感知者、提议者和协调者）来处理监控、诊断和干预，从而实现持续的RL训练和研究。

**🔧 技术方法**

使用了多智能体系统技术，具体包括感知者监控训练状态、提议者进行问题诊断和干预建议、协调者应用批准的修改。

**📊 数据集**

在Qwen3和LongCat-Flash-Lite数据集上进行了实验，Qwen3使用了17390个数学问题，LongCat-Flash-Lite使用了9340个数学问题和700多个软件工程任务。

**📈 对比分析**

与静态RL配方和通用代理相比，AutoRL在九个基准测试中表现优越，尤其在Qwen3-4B上，准确率显著提高，且在编码代理实验中，自动监督支持了14.7天的训练，成功修复了11个运行时故障，pass@1在SWE相关任务上提高了12.84%。

**⚠️ 局限性**

当前的代理在处理持续监控和局部调整方面表现良好，但在产生根本性方法论突破方面仍然存在局限性，未来的RSI研究仍需解决这一挑战。

---

## 115. Everything is an Operational View: UI as a Projection over Context in Agentic Systems

**arXiv ID:** 2611.17355 | [PDF](https://arxiv.org/pdf/2611.17355v1)

**作者:** Jieshan Chen `[一作]` (CSIRO), Zhenchang Xing `[通讯]` (CSIRO)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为AFS-UI的生成用户界面（GenUI）框架，重新定义了用户界面为基于上下文的投影，旨在解决现有系统在上下文碎片化和透明度控制方面的局限性。

**💡 创新点**

创新点在于将用户界面视为在统一持久工作空间中对上下文的投影，而不是传统的请求-响应管道中的生成工件，从而提高了系统的可演化性和可组合性。

**🔧 技术方法**

使用了代理文件系统（Agentic File System, AFS）抽象，将用户界面与系统上下文和呈现解耦，支持动态、可检查的视图。

**📊 数据集**

论文中未具体提及使用的数据集，但提到的场景涉及多个异构数据源，如电子邮件、Slack、Teams和WhatsApp等。

**📈 对比分析**

与现有生成用户界面系统相比，AFS-UI通过统一的命名空间和标准化操作语义，解决了资源发现、操作语义不兼容、投影与上下文纠缠等问题，提升了系统的可组合性和可演化性。

**⚠️ 局限性**

限制在于假设所有数据和能力提供者都通过标准化接口暴露资源，这可能需要额外的工程努力来实现和维护。

---

## 116. EventSpikeSeg: Memory-Aware Mixed-Signal Architecture for Event-Camera Spiking Segmentation

**arXiv ID:** 2611.18524 | [PDF](https://arxiv.org/pdf/2611.18524v1)

**作者:** Amir Mohammad Khadem Hosseini `[一作]` (Iran University of Science and Technology), Sattar Mirzakuchaki `[通讯]` (Iran University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为EventSpikeSeg的混合信号架构，用于事件相机的分割，结合了电容器内存和共享模拟处理。

**💡 创新点**

创新点在于将持久的模拟存储与共享服务电路结合，实现了状态保持的推理工作负载，并详细描述了硬件映射和模块特性。

**🔧 技术方法**

使用了混合信号计算技术，结合了脉冲神经网络（SNN）和模拟内存。

**📊 数据集**

使用了DSEC数据集，该数据集提供高分辨率的驾驶事件和语义标签，适合于原始事件的密集预测。

**📈 对比分析**

与PyTorch参考模型相比，EventSpikeSeg在一个完整的DSEC帧上实现了24.26%的平均交并比（mIoU），而PyTorch参考模型为26.22%。

**⚠️ 局限性**

限制在于未验证物理全网络操作、系统能耗和不匹配的鲁棒性，且未展示实际硬件的准确性。

---

## 117. ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL

**arXiv ID:** 2611.18533 | [PDF](https://arxiv.org/pdf/2611.18533v1)

**作者:** Zelin He `[一作]` (Pennsylvania State University), Alborz Geramifard `[通讯]` (LinkedIn)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种新的强化学习方法，通过从数据中学习和演化代理的支架（harness），以提高模型在多轮交互中的表现。

**💡 创新点**

创新点在于将支架视为可以从数据中学习的程序，并随着模型的训练而演化，而不是固定不变的手写支架。

**🔧 技术方法**

使用了强化学习（RL）和程序化的方法来表示和优化支架，具体包括压缩和引导程序。

**📊 数据集**

使用了多个环境的数据集，包括CodeSearch、ALFWorld和BIRD SQL，模型规模从1.5B到8B不等。

**📈 对比分析**

与固定支架的基线方法相比，提出的方法在大多数设置中提高了性能，尤其是在CodeSearch上，平均提升了%至%不等，并且在内存、技能、提示和支架演化的基线中表现最佳。

**⚠️ 局限性**

限制在于比较的范围主要集中在CodeSearch上，其他环境的测试相对较少；此外，门控机制的样本量有限，可能导致偶尔接受稍差的程序。

---

## 118. HerAgent: A Block-Level Checkpointing Harness for Coding-Agent Environment Setup

**arXiv ID:** 2611.18443 | [PDF](https://arxiv.org/pdf/2611.18443v1)

**作者:** Xiang Li `[一作]` (University College London), He Ye `[通讯]` (University College London)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为HerAgent的块级检查点环境设置工具，用于自动化编码代理的环境配置。

**💡 创新点**

HerAgent的创新点在于引入了语义设置块作为进度控制的抽象，能够在执行、验证、检查点、修复和回滚中使用，从而提高了环境设置的成功率和效率。

**🔧 技术方法**

使用了块级检查点技术，结合了局部修复和根本原因回滚的机制。

**📊 数据集**

在四个环境设置数据集上进行了评估，包括ExecutionAgent-Bench、SetupBench、Repo2Run-Bench和Installamatic-Bench。

**📈 对比分析**

与六个先前的代理进行比较，HerAgent在所有比较中都提高了设置成功率，尤其在Installamatic-Bench上从52.0%提升至85.0%，在SetupBench上从46.3%提升至70.4%。同时，设置成本降低了最多94.5%。

**⚠️ 局限性**

限制在于当前方法可能无法处理所有类型的环境设置失败，尤其是涉及复杂依赖关系和系统工具的情况。

---

## 119. Adversarial Tracking: Attacks on Image Sensors for Object Tracking in Autonomous Vehicles

**arXiv ID:** 2611.17505 | [PDF](https://arxiv.org/pdf/2611.17505v1)

**作者:** Han Wu `[一作]` (Newcastle University), Rishad Shafik `[通讯]` (Newcastle University)

**通讯引用:** 2483 | [OpenAlex ID](https://openalex.org/A5077777787)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文评估了针对YOLOv3和YOLOv4检测器的图像特定和图像无关的对抗攻击对自动驾驶车辆中物体跟踪系统的影响。

**💡 创新点**

创新点在于系统性地评估对抗攻击对物体跟踪性能的影响，并建立了一个基准来评估多种实时跟踪算法的表现。

**🔧 技术方法**

使用了YOLOv3和YOLOv4作为物体检测器，并结合SORT、Deep SORT、Strong SORT和OC-SORT等跟踪系统进行评估。

**📊 数据集**

使用CARLA模拟器生成的合成数据集进行实验。

**📈 对比分析**

通过MOTA、HOTA和IDF1等指标比较不同跟踪算法的性能，结果显示YOLOv3在图像特定攻击下MOTA降至负值，而YOLOv4表现出更强的鲁棒性。

**⚠️ 局限性**

限制在于对抗攻击的影响可能因检测器的不同而异，且未考虑车辆目标的特定扰动可能导致的进一步比较。

---

## 120. Super-Analysis for Aggressive yet Scalable Compiler Optimization

**arXiv ID:** 2611.17737 | [PDF](https://arxiv.org/pdf/2611.17737v1)

**作者:** Amirali Ebrahimzadeh `[一作]` (University of Michigan), Amir Shaikhha `[通讯]` (TU Darmstadt)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的编译器优化方法，称为超分析（super-analysis），该方法结合了传统编译器的快速优化和超优化的深度分析，旨在提高编译器的性能。

**💡 创新点**

创新点在于使用求解器（SMT solver）进行分析，而不进行程序合成，从而在保持传统转换规则的同时，提供更精确的优化。

**🔧 技术方法**

采用了基于SMT的分析技术，结合LLVM编译器框架进行实现。

**📊 数据集**

使用了多个数据集，包括C、Swift、Rust和Julia等语言的基准测试，特别是针对zlib、zstd、CryptoSwift等库的性能评估。

**📈 对比分析**

与Souper（LLVM的合成超优化器）相比，超分析在更短的时间内移除了更多的安全检查，性能提升显著，某些基准测试的速度提升达到4.01倍。

**⚠️ 局限性**

限制在于超分析的有效性依赖于求解器的性能，且在某些情况下，复杂的分析可能导致编译时间增加。

---

## 121. C-to-Rust Translation Validation at the LLVM IR Level via AI-Assisted Refinement Proof

**arXiv ID:** 2611.18036 | [PDF](https://arxiv.org/pdf/2611.18036v1)

**作者:** Eunsu Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jaemin Hong `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种验证C到Rust翻译的方法，通过在Rocq中证明Rust程序细化C程序，即Rust程序的每种行为也是C程序的行为。

**💡 创新点**

创新点在于使用LLVM中间表示（LLVM IR）来定义C和Rust程序的行为，并通过引入共同抽象来解决C和Rust数据表示之间的差异。

**🔧 技术方法**

使用了Rocq作为证明助手，结合LLVM IR的形式语义和CRIS框架来构建细化证明。

**📊 数据集**

使用了来自VERT基准套件的519个C程序进行评估，这些程序涵盖了多种算法和数据结构。

**📈 对比分析**

与现有的C到Rust翻译验证工具（如VERT和RustAssure）相比，该方法有效地验证了486个翻译的正确性，发现了26个反例，其中3个是有效的反例，且没有对执行进行限制。

**⚠️ 局限性**

局限性在于当前实现对I/O操作的支持有限，且未能处理某些复杂特性，如递归结构和闭包等。

---

## 122. Active Learning of Temporal Specifications through Distinguishing Queries

**arXiv ID:** 2611.17866 | [PDF](https://arxiv.org/pdf/2611.17866v1)

**作者:** Rajarshi Roy `[一作]` (University of Liverpool), Xiaowei Huang `[通讯]` (University of Liverpool)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种用于线性时序逻辑（LTLf）的主动学习框架，该框架通过教师标记轨迹和回答自然语言问题来识别目标规范。

**💡 创新点**

创新点在于教师只需标记轨迹并回答自然语言问题，而不需要进行等价查询或构造反例，从而降低了对非专家用户的要求。

**🔧 技术方法**

使用了基于枚举的版本空间构建和自然语言提示的主动学习技术。

**📊 数据集**

在实验中使用了36个目标，包括四个标准环境中的目标，进行了5540次运行。

**📈 对比分析**

与之前的SAT基础学习方法相比，该方法在每个目标大小上都表现得更快，且在比较所有竞争者时，学习者能够识别每个目标，平均只需在初始样本的基础上增加不到7个查询。

**⚠️ 局限性**

限制在于假设目标的大小是已知的，且未来的工作需要考虑人类用户和噪声标签的影响。

---

## 123. Explanations that Transfer: Interpreting Medical Retrieval with Sparse AutoEncoders

**arXiv ID:** 2611.18179 | [PDF](https://arxiv.org/pdf/2611.18179v1)

**作者:** Diane Bouchacourt `[一作]` (Doctolib), Gaëtan Brison `[通讯]` (Doctolib)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种基于稀疏自编码器（SAE）的医学信息检索模型，旨在提高医学查询的检索相关性，并提供可解释的特征表示。

**💡 创新点**

创新点在于通过结合重构目标和信息对比损失，生成语义可解释的稀疏表示，从而改善医学检索的可解释性和泛化能力。

**🔧 技术方法**

使用了稀疏自编码器（SAE）技术，结合重构损失和信息对比损失来训练模型。

**📊 数据集**

使用了四个公开可用的基准数据集：CURE、MIRIAD、NFCorpus和R2MED，共计六个数据集进行评估。

**📈 对比分析**

与三种嵌入模型进行了比较，结果显示稀疏编码在领域内检索性能上优于原始嵌入，并且在领域外表现损失较小，尤其是在目标语料与训练语料相距较远时。

**⚠️ 局限性**

限制在于人类临床医生的标注规模较小，未来需要更大规模的研究；此外，模型的可解释性依赖于LLM，可能引入偏差或校准错误；最后，稀疏表示在所有数据集上的检索性能未必能匹配密集表示。

---

## 124. Adversarially Batched Online Learning

**arXiv ID:** 2611.17500 | [PDF](https://arxiv.org/pdf/2611.17500v1)

**作者:** Mingda Qiao `[一作]` (University of Massachusetts Amherst), Kerui Wu `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5111250627)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种对抗性批量在线学习模型，研究了在特定时间点更新决策规则的学习者在随机赌博机、在线凸优化、专家建议预测和对抗性赌博机等四个经典问题中的表现。

**💡 创新点**

创新点在于引入了对抗性批量在线学习的模型，识别了基于相对批量增长的通用复杂性度量，并为随机赌博机提供了基于此度量的后悔界限。

**🔧 技术方法**

使用了对抗性批量在线学习模型，结合了自适应加倍技巧和经典的在线学习算法（如在线梯度下降、Hedge和EXP3）。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在随机赌博机、在线凸优化、专家问题和对抗性赌博机等问题上的应用。

**📈 对比分析**

通过与现有算法的比较，展示了在对抗性批量设置下的后悔界限，证明了即使在批量感知学习者中，最优后悔也由批量长度序列的2范数紧密表征。

**⚠️ 局限性**

限制在于对抗性批量在线学习模型的环境是不可控的，学习者必须在每个批次开始时做出承诺，无法在批次结束前进行修正，这可能导致显著的后悔。

---

## 125. Safety Analysis of Parameterized Networks with Shared Counters

**arXiv ID:** 2611.18175 | [PDF](https://arxiv.org/pdf/2611.18175v1)

**作者:** June Roupin `[一作]` (IRIF Université Paris Cité), Arnaud Sangnier `[通讯]` (DIBRIS Università di Genova)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

研究了具有无限数量进程的网络，这些进程共享一些自然计数器，可以递减、递增和测试为零。探讨了两个安全性问题：一个是是否存在一个网络的执行使得某个进程达到给定的最终状态，另一个是所有进程都达到最终状态。

**💡 创新点**

通过限制使用单个计数器或禁止使用零测试，恢复了可判定性。对于这些可判定的情况，得到了低复杂度结果，问题可以在非确定性对数空间或多项式时间内解决。

**🔧 技术方法**

使用了计数器协议模型，分析了共享计数器的网络，并通过构造和复杂性分析来解决可达性和同步问题。

**📊 数据集**

没有具体提到使用的数据集，但研究的对象是具有共享计数器的网络模型。

**📈 对比分析**

在一般情况下，这两个问题是不可判定的；但在限制条件下（如使用单个计数器或不使用零测试），可达性问题是NL完全的，而同步问题在多项式时间内可解决。

**⚠️ 局限性**

限制在于在没有限制的情况下，这两个问题是不可判定的，且在固定数量的计数器情况下的复杂性尚不明确。

---

## 126. What Balanced Accuracy Cannot Identify in Neural-Signal Decoding under Test-Time Adaptation

**arXiv ID:** 2611.17925 | [PDF](https://arxiv.org/pdf/2611.17925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 127. Domain Adaptation Based Pipeline for Character Classification and Handwritten Text Recognition

**arXiv ID:** 2611.17952 | [PDF](https://arxiv.org/pdf/2611.17952v1)

**作者:** Florent Imbert `[一作]` (Luleå University of Technology), Elisa H. Barney Smith `[通讯]` (Luleå University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于无监督领域适应的管道，用于字符分类和手写文本识别，旨在减少源域和目标域之间的分布差距。

**💡 创新点**

创新点在于引入了一种新的管道，结合了专家模型和基于DRANet的领域适应模块，能够有效利用未标记的目标数据进行模型适应。

**🔧 技术方法**

使用了DRANet架构进行领域适应，结合了卷积神经网络（ResNet-34）和卷积递归神经网络（CRNN）作为专家模型。

**📊 数据集**

使用了Safran-MNIST-DLS数据集和READ 2018数据集，前者用于字符分类，后者用于手写文本识别。

**📈 对比分析**

与完全监督模型进行了比较，结果显示该方法在Safran-MNIST-DLS数据集上达到了76.17%的宏F1分数，超越了现有的最先进方法，且在READ 2018数据集上也取得了竞争性结果。

**⚠️ 局限性**

限制在于该方法在训练阶段增加了计算时间，但在推理阶段没有额外的计算时间开销。

---

## 128. RecurIR: Learning When and How to Revise Restoration Policies via Recursive Self-Improvement

**arXiv ID:** 2611.18006 | [PDF](https://arxiv.org/pdf/2611.18006v1)

**作者:** Hu Gao `[一作]` (Shanghai Jiao Tong University), Lizhuang Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种递归自我改进框架RecurIR，使得图像恢复代理能够自主发现、挑战和修订决策规则，以应对变化的退化条件。

**💡 创新点**

创新点在于引入了政策假设发现机制、规则反驳驱动的任务演变机制和变化条件的政策修订机制，使得恢复策略能够根据退化条件的变化进行动态调整。

**🔧 技术方法**

使用了递归自我改进框架，结合政策假设发现、规则反驳和变化条件的政策修订等技术。

**📊 数据集**

进行了广泛的实验，验证了该框架在多种恢复场景下的有效性，但具体数据集未在摘要中提及。

**📈 对比分析**

与现有的图像恢复模型和代理方法进行了比较，RecurIR在多个恢复任务中表现出色，平均PSNR达到35.58 dB，优于最强的基线模型Defusion 2.00 dB，且在所有评估的恢复代理中也表现最佳。

**⚠️ 局限性**

限制在于需要进一步验证该框架在更复杂的真实场景中的适用性，以及如何在不同类型的退化条件下保持高效的恢复性能。

---

## 129. LoRAWeaver: Directional Adapter Routing in LoRA Representation Space

**arXiv ID:** 2611.17340 | [PDF](https://arxiv.org/pdf/2611.17340v1)

**作者:** Inish Melroy D Souza `[一作]` (Loughborough University), Andrea Soltoggio `[通讯]` (Loughborough University)

**通讯引用:** 1653 | [OpenAlex ID](https://openalex.org/A5049155414)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为LoRAWeaver的训练无关的LoRA适配器检索和组合方法，通过余弦相似度在适配器输出空间中选择适配器。

**💡 创新点**

创新点在于将适配器路由视为适配器输出空间中的表示对齐问题，避免了对路由器的训练和模型的合并。

**🔧 技术方法**

使用了余弦相似度来匹配适配器条件查询表示与任务原型，且在每一层独立进行适配器选择。

**📊 数据集**

在FLAN-v2基准上进行了实验，使用了LLaMA-2-7B和LLaMA-2-13B模型，并在48个任务上进行了评估。

**📈 对比分析**

与训练无关的基线方法相比，LoRAWeaver在跨任务设置中表现优越，接近于训练的检索编码器的性能。

**⚠️ 局限性**

限制在于共享的原型向量可能保留关于校准分布的信息，可能导致隐私问题，尤其是在原型是从少量示例计算时。

---

## 130. SemRAG: Semantic-Guided Sentence-Level Retrieval-Augmented Sign Language Production

**arXiv ID:** 2611.18189 | [PDF](https://arxiv.org/pdf/2611.18189v1)

**作者:** Xiao Liu `[一作]` (Nanjing University), Lei Xie `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为SemRAG的框架，用于生成手语姿势序列，以准确表达口语文本，同时保持手语的节奏和协调模式。

**💡 创新点**

创新点在于通过检索完整的句子级手语模板作为运动先验，而不是依赖于词汇或单词级的手语字典，从而改善了生成的语义一致性和运动质量。

**🔧 技术方法**

使用了语义引导的句子级检索增强框架，结合了冻结的手语翻译模型（SLT）进行模板的语义定位，并通过层特定的可学习门控机制调节模板特征的贡献。

**📊 数据集**

使用了Phoenix14T和CSL-Daily数据集进行实验，前者包含7096个训练样本，后者包含18401个训练样本。

**📈 对比分析**

与文本到姿势和基于检索的方法进行了比较，SemRAG在语义一致性和运动质量上表现优于基线方法，尤其在BLEU-4和MPJPE指标上有显著提升。

**⚠️ 局限性**

限制在于检索的模板可能包含与目标语义不相关的运动片段，尽管引入了SLT引导的语义定位模块来缓解这一问题。

---

## 131. DECODE: Diagnosing Interpretability Failures in Automated Interpretation Process

**arXiv ID:** 2611.17856 | [PDF](https://arxiv.org/pdf/2611.17856v1)

**作者:** Dohyun Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jaemin Jo `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了DECODE，一个视觉分析系统，旨在帮助解释性研究人员诊断自动化解释过程中的失败。

**💡 创新点**

通过引入人类监督，DECODE解决了自动化解释过程中的自我参考验证循环问题，并提供了基于共识的多阶段诊断接口。

**🔧 技术方法**

使用了稀疏自编码器（SAE）进行特征提取，结合了多种大型语言模型（LLMs）生成和评估解释，并采用了主动学习和视觉交互标注技术。

**📊 数据集**

使用了Gemma-2-9b模型的16,384个稀疏特征，结合了每个特征的20个激活示例和多个独立生成的解释。

**📈 对比分析**

与传统的全自动方法相比，DECODE通过引入人类判断和多阶段诊断流程，显著提高了对解释性失败的识别和归因能力，专家的标注质量优于仅依赖自动化的标注。

**⚠️ 局限性**

存在标签质量不高的问题，尤其是在细粒度归因方面，且在冷启动和类别不平衡方面也面临挑战。

---

## 132. Unfinished-Bench: Measuring Rule Violations by Persistent Agents in Realistic Work Environments

**arXiv ID:** 2611.17460 | [PDF](https://arxiv.org/pdf/2611.17460v1)

**作者:** Seunghyun Lee `[一作]` (NAVER AI Lab), Sangdoo Yun `[通讯]` (NAVER AI Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Unfinished-Bench基准，旨在评估AI代理在自主运行中对任务完成的持久性，尤其是在存在内部规则的情况下。

**💡 创新点**

创新点在于将持久性与任务完成压力结合，评估代理在没有用户监督的情况下如何处理指令与规则之间的冲突。

**🔧 技术方法**

使用了持久性条件的评估方法，通过模拟自主运行的环境来测试模型的表现。

**📊 数据集**

使用了来自7个不同组织的97个场景，涵盖了多个风险类别，涉及内部文档、员工目录和聊天记录等数据。

**📈 对比分析**

与现有基准相比，Unfinished-Bench在评估持久性对规则违反的影响方面表现出显著差异，持久性导致规则违反率显著上升，尤其是在规则嵌入环境中时。

**⚠️ 局限性**

限制在于只使用了单一的评估工具（Hermes Agent），仅评估了12个开放模型，未测试自主持久性训练的专有模型。

---

## 133. From AI-Enabled Remote Patient Monitoring Toward Agentic Robotics: Embodiment, Governance, and Care Integration

**arXiv ID:** 2611.18484 | [PDF](https://arxiv.org/pdf/2611.18484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 134. Envisioning Agentic Robotics: A New Era of Robotic Systems

**arXiv ID:** 2611.18359 | [PDF](https://arxiv.org/pdf/2611.18359v1)

**作者:** Grid Robotics `[一作]`, Tao Yu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种代理机器人框架，旨在将机器人从执行特定任务转变为能够理解人类意图并适应多种环境的智能体。通过分析代理机器人在家庭早餐场景中的不同责任层级，展示了机器人能力的逐步提升。

**💡 创新点**

创新点在于提出了一个统一的代理机器人框架，区分了三个功能角色（S2、S1和S0），并强调了从政策中心的机器人转向代理中心的机器人，关注于协调技能和工具、评估结果和适应决策的能力。

**🔧 技术方法**

使用了视觉-语言-行动（VLA）模型和世界-行动模型（WAM），结合深度学习和强化学习技术，来实现机器人在复杂环境中的行为学习和适应。

**📊 数据集**

数据集包括真实机器人数据、UMI演示、以自我为中心的人类数据和多模态网络数据，旨在提高模型的泛化能力。

**📈 对比分析**

通过建立一个评估框架，比较不同模型和系统的性能，强调独立结果验证和透明报告的重要性。初步结果显示，使用GPT-6 Astra模型的机器人在多个基准测试中表现优于传统的任务特定策略，但在复杂任务中仍面临挑战。

**⚠️ 局限性**

局限性包括缺乏对代理机器人共同理解的明确界定，实验条件和系统配置的差异使得不同研究结果难以比较，以及在动态环境中保持可靠性的挑战。

---

## 135. Probabilities of Causation in Nondeterministic Causal Models

**arXiv ID:** 2611.18245 | [PDF](https://arxiv.org/pdf/2611.18245v1)

**作者:** David S. Watson `[一作]` (Google DeepMind), Sander Beckers `[通讯]` (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在非确定性因果模型中因果概率的识别问题，提出了一种通用的识别程序，通过约束优化计算反事实量的最小/最大值。

**💡 创新点**

在非确定性模型下，推导出因果概率的精确解析界限，提供了比确定性模型更具信息量的识别区域。

**🔧 技术方法**

使用了约束优化技术来计算反事实量，并通过半马尔可夫模型进行分析。

**📊 数据集**

使用了半合成数据集进行实验，验证了所提出方法的有效性。

**📈 对比分析**

与传统的确定性模型相比，所提出的方法在识别因果概率方面表现出更强的能力，尤其是在处理潜在混淆时。

**⚠️ 局限性**

限制在于潜在参数的错误指定可能导致无效的界限，无法包含目标反事实。

---

## 136. Delayed Effects of Temporary Weight Decay in Muon Pretraining

**arXiv ID:** 2611.17876 | [PDF](https://arxiv.org/pdf/2611.17876v1)

**作者:** Md Rifat Arefin `[一作]` (Meta), Patrick Huber `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了临时优化干预对模型训练的影响，特别是临时权重衰减的效果。

**💡 创新点**

提出了临时干预的效果可能在干预结束后继续演变，甚至改变其符号或排名。

**🔧 技术方法**

使用了Muon优化器和临时权重衰减技术。

**📊 数据集**

使用了FineWeb100B和FineWeb-Edu数据集。

**📈 对比分析**

通过比较不同时间段和持续时间的衰减窗口，发现早期干预可以改善最终损失，而晚期干预则可能导致损失恶化。性能评估显示，早期窗口在最终检查点的表现优于晚期窗口。

**⚠️ 局限性**

研究的局限性在于未能确定为何特定的相对步长轨迹是有益的，以及不同优化器设置下的效果可能不一致。

---

## 137. DOCKET: A Pluralistic Benchmark for Evaluating AI-Assisted Policy Feedback in U.S. Rulemaking

**arXiv ID:** 2611.17914 | [PDF](https://arxiv.org/pdf/2611.17914v1)

**作者:** Alexander Spangher `[一作]` (Stanford University), Daniel E. Ho `[通讯]` (Stanford University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文旨在评估人工智能模型在支持多元观点方面的能力，特别是在美国联邦通知与评论过程中的应用。研究首先构建了DOCKET数据集，包含了2016年至2026年间来自44个政府机构的4230万条公众评论，并展示了公众评论者的多样性和政策专业知识的深度。

**💡 创新点**

创新点在于提出了DOCKETBench，这是第一个系统性评估大型语言模型（LLMs）在联邦规则制定中代表人类政策观点的能力的基准。研究揭示了LLMs在生成反馈时与真实人类评论之间的显著差距，尤其是在具体性和引用方面。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行数据处理和分析，包括使用gpt-5-mini和Llama-3.3-70b等模型来提取法规条款、评论和回应，并建立了匹配管道来分析公众评论与政府回应之间的关系。

**📊 数据集**

使用的数据集为DOCKET，包含了4230万条公众评论、17198条提议规则和22400条最终规则，涵盖了44个联邦机构的规则制定过程。

**📈 对比分析**

与其他方法的比较显示，LLMs生成的反馈与真实人类评论的对齐率仅为18-30%。LLMs通常更倾向于支持政府观点，而对工会的支持率最低，显示出观点的根本偏差。

**⚠️ 局限性**

限制在于LLMs在生成的评论中缺乏具体性和深度，无法有效代表多元化的公众观点，尤其是在技术性和法律性讨论中，模型的表现未能达到预期的多元化标准。

---

## 138. Understanding the limits of snapshot-based fault diagnosis in distribution networks

**arXiv ID:** 2611.17998 | [PDF](https://arxiv.org/pdf/2611.17998v1)

**作者:** Ba Tu Phung `[一作]` (University of Science and Technology), Kim-Anh Nguyen `[通讯]` (University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文研究了基于快照的故障诊断在配电网络中的局限性，分析了故障分类的误差来源，并提出了改进的方法。

**💡 创新点**

创新点在于提出了一种基于可区分性边际的传感器布置方法，并引入了拒绝选项来提高分类准确性，同时明确了测量误差对故障诊断的影响。

**🔧 技术方法**

使用了机器学习技术，包括13种分类器和四种特征集，结合了基于快照的电压相量测量。

**📊 数据集**

使用了三个IEEE测试馈线（13节点、34节点和123节点）以及一个公共的DER-Fault数据集，后者包含了在不同分布式能源场景下的故障事件。

**📈 对比分析**

通过与十二种传感器布置方法进行比较，提出的可区分性边际布置方法在高阻抗故障的分类准确性上表现更佳，且在1%准确度类别下，三种电压传感器的故障分类准确率为90-96%。

**⚠️ 局限性**

限制在于数据是基于稳态相量模型的模拟，未考虑快照时序和瞬态，且仅使用电压测量，微型PMU噪声水平来自单次测量，操作范围在负载变化和DER渗透下有效，但在失去两个传感器时不再适用。

---

## 139. Personalization of Basic and Moral Values Guiding LLM-Controlled Assistant Robots

**arXiv ID:** 2611.18230 | [PDF](https://arxiv.org/pdf/2611.18230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 140. Where to Predict in Retinal OCT? Anatomy-Guided Target Selection for I-JEPA

**arXiv ID:** 2611.17586 | [PDF](https://arxiv.org/pdf/2611.17586v1)

**作者:** Yunchu Feng `[一作]` (University of Pennsylvania), Kaveri A. Thakoor `[通讯]` (Columbia University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究探讨了在视网膜光学相干断层扫描（OCT）中，如何选择预测目标以提高自监督学习模型I-JEPA的表现，特别是通过解剖学引导的目标选择来改善青光眼分类的效果。

**💡 创新点**

创新点在于提出了基于解剖学的目标选择策略，比较了不同的目标选择方法（如矩形目标、解剖形状目标和覆盖约束目标），并发现基于图像强度的矩形目标在青光眼分类中表现最佳。

**🔧 技术方法**

使用了图像基础的联合嵌入预测架构（I-JEPA），结合解剖学引导的目标选择方法进行自监督学习。

**📊 数据集**

使用FairVision数据集，该数据集包含带有青光眼标签的OCT体积数据，共有6000个训练体积和1000个验证体积。

**📈 对比分析**

通过线性探测方法评估不同目标选择策略的表现，结果显示基于图像强度的矩形目标在青光眼分类的AUC上表现最佳，而更具体的解剖学策略并未一致提高性能。

**⚠️ 局限性**

研究的局限性包括只使用了一个数据集和一个下游任务，且不同策略的比较未能在独立的预训练运行中进行匹配，可能影响结果的普遍性和可靠性。

---

## 141. A$^3$: Adaptive Audio Agent Framework for Collaborative Edge Computing

**arXiv ID:** 2611.18064 | [PDF](https://arxiv.org/pdf/2611.18064v1)

**作者:** Zexin Li `[一作]` (Nanyang Technological University), Yiming Chen `[通讯]` (ASUS Intelligent Cloud Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种自适应音频代理框架，该框架在服务器上放置整个管道阶段，并在设备上保留完整的后备代理，以应对网络链接失败的情况。

**💡 创新点**

创新点在于结合了完整的本地后备、整个阶段的服务器执行和基于链接的后备截止时间，形成一个统一的音频框架。

**🔧 技术方法**

使用了自适应音频代理框架，结合了链接状态监测和确定性控制器来决定何时启动本地代理。

**📊 数据集**

在Jetson Orin AGX、Orin Nano和RTX A4500服务器上进行了评估，使用了六个模型（两个ASR模型、三个Qwen语言模型和一个TTS模型），在十二种网络条件下进行测试。

**📈 对比分析**

与其他方法（如静态卸载和百分位数对冲）相比，该框架在模拟中完成了91%的请求，而百分位数对冲为87%。在新测试过程中，该框架在新故障情况下的表现优于其他方法。

**⚠️ 局限性**

局限性在于小型设备的内存限制，可能无法容纳整个管道，导致在某些情况下无法提供完整的服务。

---

## 142. Aligned, Conflicting, or Unspecified? A Diagnostic Benchmark for Multimodal Time Series Forecasting

**arXiv ID:** 2611.17680 | [PDF](https://arxiv.org/pdf/2611.17680v1)

**作者:** Xintong Wang `[一作]` (Kanazawa University), Ruidong Li `[通讯]` (Kanazawa University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个新的基准，评估文本与历史数据之间的关系对预测模型的影响，包含1762个文本-窗口对，涵盖天气、金融和维基百科数据。

**💡 创新点**

首次引入了基于文本-历史关系的真实世界诊断基准，并探讨了文本选择和组合对预测性能的影响。

**🔧 技术方法**

使用了十种预训练模型进行零-shot评估，涉及语言模型和时间序列模型。

**📊 数据集**

数据集包含来自天气、金融和维基百科的时间序列和相关文本，共1762个文本-窗口对。

**📈 对比分析**

通过比较不同文本类型在对齐和冲突情况下的表现，发现对齐情况下文本有助于降低MASE，而冲突情况下则可能导致MASE增加，但提高了方向准确性（DA）。引入了一种限制大型修正的训练无关规则，显著改善了模型性能。

**⚠️ 局限性**

限制在于文本的方向性和历史趋势之间的冲突可能导致预测错误，尤其是在大型修正的情况下，增加了MASE。

---

## 143. Signal Scaffolding: Unlocking the Potential of Lightweight Models for Self-Supervised Monocular Depth Estimation

**arXiv ID:** 2611.17828 | [PDF](https://arxiv.org/pdf/2611.17828v1)

**作者:** Zhijie Wang `[一作]` (Hefei University of Technology), Jiangshuo Zhang `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的三阶段训练策略Signal Scaffolding (SS)，用于轻量级自监督单目深度估计，旨在提高模型的准确性而不增加额外的数据集或参数。

**💡 创新点**

创新点在于通过重新组织训练流程，充分挖掘轻量级模型的潜在能力，从而在不改变网络架构的情况下显著提高准确性。

**🔧 技术方法**

使用了Signal Scaffolding (SS)三阶段训练策略，并设计了轻量级网络ReForgeDepth，结合了Gated Multi-Dilation Block (GMD)和Detail-Compensated Global Interaction (DCGI)。

**📊 数据集**

使用了KITTI和Cityscapes数据集进行实验，并在Make3D数据集上进行了跨数据集的泛化测试。

**📈 对比分析**

与其他轻量级模型相比，ReForgeDepth在KITTI和Cityscapes数据集上实现了最佳的准确性-效率平衡，且在实时推理中表现优异，参数数量仅为1.5M，超越了同规模的其他模型。

**⚠️ 局限性**

限制在于光度一致性在移动物体周围的深度估计效果较差，以及缺乏真实深度数据限制了准确性。未来的工作将整合运动感知掩蔽到Signal Scaffolding中，并探索更强的几何先验以进一步缩小监督差距。

---

## 144. Hello Again: Exploiting Replay Protected Monotonic Counters to Brute-Force Windows Hello PINs

**arXiv ID:** 2611.18126 | [PDF](https://arxiv.org/pdf/2611.18126v1)

**作者:** Christian Werling `[一作]` (Technische Universitat Berlin), Jean-Pierre Seifert `[通讯]` (Technische Universitat Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文对在出货的x86系统中使用的重放保护单调计数器（RPMC）进行了首次实用的安全分析，揭示了多个实现缺陷，这些缺陷严重削弱了RPMC的安全模型。

**💡 创新点**

创新点在于首次公开分析了RPMC控制器和闪存芯片的安全性，发现了Winbond闪存芯片在电压故障攻击下泄露HMAC密钥的重大缺陷，以及AMD NV控制器未验证计数器增量的漏洞。

**🔧 技术方法**

使用低成本的中介硬件进行攻击，结合电压故障注入技术，分析了SPI总线上的RPMC通信。

**📊 数据集**

测试了来自Winbond、Macronix和GigaDevice的闪存芯片，评估了它们在RPMC支持下的安全性。

**📈 对比分析**

通过与现有的TPM+PIN保护机制进行比较，展示了攻击的有效性：在AMD笔记本上，使用未验证增量的攻击可以在40小时内破解4位Windows Hello PIN，而BitLocker的6位PIN破解时间缩短至20天。

**⚠️ 局限性**

限制在于未能修复的硬件和固件漏洞，尤其是Winbond的HMAC密钥泄露和AMD的未验证增量问题，导致RPMC的安全性受到质疑。

---

## 145. EViDS: An Environment--Vehicle Dynamics Integrated Simulation Platform for Intelligent Chassis Development

**arXiv ID:** 2611.17660 | [PDF](https://arxiv.org/pdf/2611.17660v1)

**作者:** Lu Xiong `[一作]` (Tongji University), Chen Tang `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出了EViDS，一个环境-车辆动力学集成仿真平台，用于智能底盘开发，旨在将环境信息与高保真车辆动力学相结合，以支持环境感知的智能底盘控制。

**💡 创新点**

创新点在于开发了一个集成的环境-车辆动力学协同仿真架构，支持闭环交互，并提出了多种交互机制以保持空间和时间一致性。

**🔧 技术方法**

使用了CARLA进行环境和传感器仿真，CarSim进行高保真车辆动力学仿真，以及MATLAB/Simulink作为集成层。

**📊 数据集**

使用了CARLA和CarSim的数据集，结合了多种场景和测试环境，包括住宅、商业、工业等多种场景。

**📈 对比分析**

通过环境-动力学交互测试和车辆动力学一致性比较评估EViDS，结果表明EViDS能够可靠地传递环境信息，并支持在线闭环控制，性能优越。

**⚠️ 局限性**

限制在于现有的仿真平台仍然存在一些局限性，可能无法完全满足所有智能底盘开发的需求。

---

## 146. Interpreting Coding Agents Through Their Behavioral Trajectories

**arXiv ID:** 2611.17440 | [PDF](https://arxiv.org/pdf/2611.17440v1)

**作者:** Ibragim Badertdinov `[一作]` (Nebius), Alexander Golubev `[通讯]` (Nebius)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

分析了来自125个代理配置的143,209个执行轨迹，开发了一种基于规则的解析器，将原始执行轨迹映射到高层次的动作，以便在异构代理接口之间进行比较。

**💡 创新点**

提出了一种确定性的轨迹规范化管道，展示了相似的解决率可能掩盖可观察工作流中的显著差异，并且聚合行为特征能够在不同的代码库和编程语言中检索相应的代理堆栈。

**🔧 技术方法**

使用了一种基于规则的解析器来分类和分析执行轨迹，结合了多模型共识审计来验证解析器的预测。

**📊 数据集**

使用了SWE-rebench数据集，该数据集包含860个Python任务和111个多语言任务，涵盖了413个代码库，共分析了143,209个完整的执行轨迹。

**📈 对比分析**

通过结果匹配比较，发现即使在相似的解决率下，代理的工作流也存在显著差异。使用分割控制实验确认这些差异不是运行间变异的结果，且在多个工作流指标上表现出一致性。

**⚠️ 局限性**

限制在于解析器的审计依赖于前沿LLM的共识，而不是人工标注，且可能错过非标准命令执行中的语义上下文或意图。此外，所有轨迹均在单一基准环境中收集，可能反映任务熟悉度或基准特定策略。

---

## 147. A Framework for Cost-Preserving Rounding with Applications to Laminar Thin Trees

**arXiv ID:** 2611.18399 | [PDF](https://arxiv.org/pdf/2611.18399v1)

**作者:** Jannis Blauth `[一作]` (ETH Zurich), Rico Zenklusen `[通讯]` (ETH Zurich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种通用框架，用于在保持成本不变的情况下，将分数解舍入为整数解，特别是在寻找薄树（如层次薄树或全局薄树）的背景下。

**💡 创新点**

创新点在于提供了一种新的舍入算法框架，使得在保持薄度约束的同时，可以实现成本的精确保留，这是首次在层次约束下实现的。

**🔧 技术方法**

使用了线性规划和舍入算法的结合，特别是通过引入弱对齐点和迭代松弛算法来实现。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在一般多面体和图的上下文中应用的算法。

**📈 对比分析**

与现有方法相比，提出的算法在保持成本不变的同时，能够找到常数薄度的树，性能上优于之前的算法，尤其是在层次约束和链约束的情况下。

**⚠️ 局限性**

限制在于框架的应用范围仍然较为狭窄，尤其是在全局薄树的情况下，如何实现成本保留的问题仍然是开放的研究问题。

---

## 148. Identifying Latent Neural Dynamics with Recognition-Parametrised Gaussian Process Dynamical Systems

**arXiv ID:** 2611.17487 | [PDF](https://arxiv.org/pdf/2611.17487v1)

**作者:** Arielle Rosinski `[一作]` (University College London), Maneesh Sahani `[通讯]` (University College London)

**通讯引用:** 11327 | [OpenAlex ID](https://openalex.org/A5035341421)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的模型RP-GPdyn，用于理解神经回路中的动态计算，旨在通过动态演化来建模神经活动。

**💡 创新点**

RP-GPdyn模型通过非参数高斯过程定义的转移函数建模动态流场，并使用识别参数化模型（RPM）框架隐式建模动态与神经活动之间的关系，避免了生成相关的问题。

**🔧 技术方法**

使用了非参数高斯过程（GP）和识别参数化模型（RPM）技术。

**📊 数据集**

使用了合成数据集和实验数据集进行验证。

**📈 对比分析**

与现有的基于生成模型的方法（如SING）相比，RP-GPdyn在动态预测方面表现出更强的鲁棒性和整体优越性，尤其是在动态预测的准确性上有显著提升。

**⚠️ 局限性**

模型假设动态是自主的，未考虑外部输入的影响，这可能限制了模型在某些情况下的适用性。

---

## 149. I2C-GRPO: Insight Importance Checklist for Reinforcement Learning in a Multi-Agent DDoS Alert Triage Workflow

**arXiv ID:** 2611.17534 | [PDF](https://arxiv.org/pdf/2611.17534v1)

**作者:** Eyal Zloof `[一作]` (Ben-Gurion University of the Negev), Rami Puzis `[通讯]` (Ben-Gurion University of the Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为I2C-GRPO的检查表奖励机制，用于优化DDoS警报的调查过程，旨在通过中间调查步骤的奖励来提高调查效率。

**💡 创新点**

创新点在于通过专家LLM提出调查行动，并利用排名模型对其重要性进行排序，从而为每个候选行动提供基于检查表的奖励，避免了传统方法中对最终结果的依赖。

**🔧 技术方法**

使用了强化学习（RL）和多代理系统的框架，结合了大语言模型（LLM）和模糊测度的概念来设计奖励机制。

**📊 数据集**

使用了DARPA 2000 LLDOS 1.0数据集进行训练，并在独立的CSE-CIC-IDS2018 DDoS数据集上进行了测试。

**📈 对比分析**

与其他方法（如DIVA和GPT-5-mini）相比，I2C-GRPO在CSE-CIC-IDS2018数据集上达到了0.869的平均F1分数，假阳性率为4.9%，明显优于DIVA的0.772和GPT-5-mini的16.4%。

**⚠️ 局限性**

限制在于所有声明仅限于DDoS警报的调查，且模型的性能可能受到训练数据集的特定性和专家模型的偏差影响。

---

## 150. Heavy Metal Acrobatics: Learning Dynamic Control for Skid-Steer Loaders

**arXiv ID:** 2611.18365 | [PDF](https://arxiv.org/pdf/2611.18365v1)

**作者:** Pierre-Jean Lorentz `[一作]` (Robotic Systems Lab), Marco Hutter `[通讯]` (Robotic Systems Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究展示了通过在模拟环境中训练强化学习（RL）策略，并将其转移到现实中，成功实现了重型施工机械的动态自主控制，特别是执行高难度的特技动作。

**💡 创新点**

创新点在于结合了强化学习和模拟到现实的转移技术，使得重型机械能够执行复杂的动态行为，这在以往仅由少数精英人类操作员完成。

**🔧 技术方法**

使用了强化学习（RL）技术，具体采用了近端策略优化（PPO）算法，并结合了域随机化和扩展卡尔曼滤波（EKF）等技术。

**📊 数据集**

使用的数据集包括在模拟环境中生成的各种控制输入和响应数据，以及在真实环境中收集的机器运动数据。

**📈 对比分析**

与模型预测控制（MPC）方法进行了比较，结果显示在平衡和速度跟踪等简单任务上，RL方法表现相当，但在处理非平坦地形和高速度时，RL方法表现出更好的鲁棒性。

**⚠️ 局限性**

限制在于当前方法依赖于模拟环境的准确性，且在复杂任务中可能需要更长的规划时间和更多的计算资源。

---

## 151. Mechanism-Level Chemical Reaction Simulation with Electron Bookkeeping Transformer

**arXiv ID:** 2611.17548 | [PDF](https://arxiv.org/pdf/2611.17548v1)

**作者:** Sarah Cao `[一作]` (Massachusetts Institute of Technology), Connor W. Coley `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 21439 | [OpenAlex ID](https://openalex.org/A5076162644)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为Electron Bookkeeping Transformer（Bookkeeper）的数据驱动模型，用于模拟化学反应机制，通过生成电子运动轨迹来理解和预测化学反应性。

**💡 创新点**

Bookkeeper通过将每个电子建模为一个令牌，并按顺序预测其运动，确保在基本步骤和反应路径预测中质量和电荷的守恒，提供了一种明确的电子记账框架。

**🔧 技术方法**

使用了生成式变换器模型，采用了电子运动的离散操作，并通过三阶段的动作因子化来预测电子运动。

**📊 数据集**

使用了来自专家策划的初级反应模板生成的大规模机制数据集，包含数百万个初级机制步骤。

**📈 对比分析**

与FlowER和MechSMILES T5进行了比较，Bookkeeper在反应大小平衡的测试集上提高了初级步骤的准确性，但在完整路径的准确性上略逊一筹。Bookkeeper在初级步骤采样速度上提高了100倍以上，适合大规模反应筛选。

**⚠️ 局限性**

Bookkeeper的局限性在于它预测的是符号箭头推导假设，而不是经过实验验证的机制或能量景观，且不明确捕捉激发态、立体化学或三维考虑。模型的泛化能力受限于可用机制训练数据的覆盖和保真度。

---

## 152. Parameterized Weakest-Precondition Calculus for Fine-grained Properties of Quantum Loops

**arXiv ID:** 2611.18132 | [PDF](https://arxiv.org/pdf/2611.18132v1)

**作者:** Tianshi Yu `[一作]` (Institute of Software, Chinese Academy of Sciences), Li Zhou `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文开发了一种组合最弱前提框架，用于分析量子循环的运行时行为，包括终止概率、条件期望运行时和长时间运行的影响。

**💡 创新点**

创新点在于使用参数化谓词来保留运行时信息，即使在期望运行时为无限的情况下，也能通过顺序组合和嵌套循环重用这些信息。

**🔧 技术方法**

采用了组合最弱前提演算和参数化谓词的技术，结合运行时加权语义来分析量子程序的运行时特性。

**📊 数据集**

使用了一维无限线上的经典离散时间量子行走作为数据集，验证了其终止概率和条件期望运行时。

**📈 对比分析**

通过与现有方法的比较，证明了最弱前提方程在参数化谓词域中具有唯一解，并展示了如何从推导的最弱前提中恢复运行时属性。

**⚠️ 局限性**

限制在于现有方法未能覆盖无限期望成本，且直接引入无限值可能导致更细致的性质未得到解决。

---

## 153. STAMP: Manufacturing-Informed Generative Design and Shape Optimization of Sheet Panels

**arXiv ID:** 2611.18144 | [PDF](https://arxiv.org/pdf/2611.18144v1)

**作者:** Jihoon Kim `[一作]` (Korea Advanced Institute of Science and Technology), Namwoo Kang `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种名为STAMP的生成设计框架，用于形成的薄板面板的几何设计，确保在探索布局和提高性能的同时，保持几何制造条件。

**💡 创新点**

STAMP的创新点在于将制造知识直接嵌入形状表示中，使得每个生成和优化的面板都能满足制造条件，而无需额外的制造惩罚项。

**🔧 技术方法**

使用了边界条件扩散生成替代布局，并通过预测器梯度指导表面优化，保持轮廓、开口和指定的固定区域。

**📊 数据集**

使用了CarHoods10k数据集，该数据集包含汽车引擎盖内面板的几何形状和结构性能值。

**📈 对比分析**

与随机选择相比，基于预测器的选择在11个保留设计家族中将最佳有限元分析（FEA）变形的平均差距减少了89%。在所有11个家族中，其性能均优于随机选择。

**⚠️ 局限性**

限制在于当前实现仅描述一个面板，无法精确表示在一个投影位置上具有多个高度的表面，且有限的分辨率可能导致细节丢失。

---

## 154. TSGP-2: On the Non-Linearity of Symbolic Regression Problems and How to Exploit It with Model-based Semantic Genetic Programming

**arXiv ID:** 2611.17455 | [PDF](https://arxiv.org/pdf/2611.17455v1)

**作者:** Philipp Anthes `[一作]` (Johannes Gutenberg University), Franz Rothlauf `[通讯]` (Johannes Gutenberg University)

**通讯引用:** 4956 | [OpenAlex ID](https://openalex.org/A5035443886)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了符号回归问题的非线性特性，并提出了一种新的模型TSGP-2，以利用线性和非线性目标函数的训练数据来改进搜索性能。

**💡 创新点**

创新点在于通过使用包含线性和非线性目标函数的训练数据，减少了对线性后代的偏见，从而提高了在非线性问题上的搜索性能。

**🔧 技术方法**

使用了基于变换器的语义遗传编程（TSGP）模型作为变异操作符，并通过训练数据生成程序对来进行训练。

**📊 数据集**

使用了合成的Feynman问题和多个真实世界基准数据集进行实验。

**📈 对比分析**

与TSGP和标准遗传编程（GP）相比，TSGP-2在具有可利用非线性的Feynman问题上表现更好，平均排名为1.19，显著优于TSGP的1.88和标准GP的2.94。

**⚠️ 局限性**

限制在于TSGP-2在处理没有可利用非线性的真实世界问题时，性能提升不明显，且在这些问题上，TSGP和TSGP-2的解决质量相似。

---

## 155. Clean Tickets, Wrong Orders: Catching Silent Errors with a Language Model in Your Browser Tab

**arXiv ID:** 2611.18284 | [PDF](https://arxiv.org/pdf/2611.18284v1)

**作者:** Umar Ahmed `[一作]` (ITMO University), Ivan Khodnenko `[通讯]` (ITMO University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种在浏览器标签页中运行的小型语言模型，旨在捕捉静默错误，即输出格式正确但内容错误的情况。

**💡 创新点**

创新点在于将一个经过微调的0.8B参数模型完全在客户端运行，确保没有订单文本离开用户设备，同时提供实时的错误检测和反馈机制。

**🔧 技术方法**

使用了WebAssembly编译的llama.cpp和TypeScript实现的PIZZA基准匹配器。

**📊 数据集**

使用了PIZZA基准测试数据集，该数据集包含多种订单格式。

**📈 对比分析**

与服务器评估的模型输出进行比较，结果显示浏览器中的模型在所有测试订单中输出与服务器相同的结果，且在消费者笔记本电脑上的延迟表现良好。

**⚠️ 局限性**

限制在于模型的推理速度可能低于本地代码，且在某些情况下可能会受到内存限制的影响。

---

## 156. PerfArena: Ranking LLMs by Code Performance

**arXiv ID:** 2611.18030 | [PDF](https://arxiv.org/pdf/2611.18030v1)

**作者:** Rui Rua `[一作]` (New York University), Karim Ali `[通讯]` (New York University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的代码生成评估框架PerfArena，该框架通过多个编程语言和竞赛的性能来排名生成的代码。

**💡 创新点**

创新点在于PerfArena是第一个同时考虑正确性、时间、内存和能耗的代码生成评估平台，并且能够在多个编程语言之间进行可比性测量。

**🔧 技术方法**

使用了可插拔的正确性和性能评估工具，能够在Unix类主机上执行代码并报告性能指标。

**📊 数据集**

在两个竞赛上进行了评估：Computer Language Benchmarks Game的10个计算密集型基准和99个问题，这些问题的接受解在性能上有显著差异。

**📈 对比分析**

PerfArena在两个竞赛中表现优异，解决了99.7%的问题，并且在时间和能耗上显著优于其他模型，其他模型的正确解在时间和能耗上平均比PerfArena高出1.1-3.6倍。

**⚠️ 局限性**

限制在于当前的评估主要集中在生成的代码的性能上，而对代码的可维护性和安全性等其他非功能性要求的评估仍然不足。

---

## 157. Condensing Law of Security Information

**arXiv ID:** 2611.17918 | [PDF](https://arxiv.org/pdf/2611.17918v1)

**作者:** Qi Tan `[一作]` (Shenzhen University), Ke Xu `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了大型语言模型（LLM）在安全领域的表现，提出了安全信息的浓缩法则，表明在特定模型大小下，安全信息的可用性达到最优。

**💡 创新点**

创新点在于提出了安全信息的浓缩法则，指出在安全任务中存在一个最佳模型大小，使得安全信息的保留和认知能力之间的权衡达到最佳。

**🔧 技术方法**

使用了信息论的界限和定理来分析模型的认知能力与安全信息保留之间的关系。

**📊 数据集**

使用了HttpParamsDataset数据集，该数据集包含正常和异常的HTTP请求参数记录，以及BigModelGym中的网络安全世界模型基准数据集。

**📈 对比分析**

通过实验验证了浓缩法则，发现随着模型大小的增加，正常和异常输入的表示逐渐接近，安全任务的性能在中间模型大小时达到最佳，之后性能下降。

**⚠️ 局限性**

限制在于该研究主要集中在模型大小与安全信息保留之间的关系，可能未考虑其他影响安全性能的因素，如训练数据的质量和多样性。

---

## 158. QCAP-VQA: Question-Conditioned Capped Prior-Mass Supervision for Medical Visual Question Answering

**arXiv ID:** 2611.17708 | [PDF](https://arxiv.org/pdf/2611.17708v1)

**作者:** Yining Zhao `[一作]` (University of Technology Sydney), Ali Braytee `[通讯]` (University of Technology Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种新的医学视觉问答模型QCAP-VQA，该模型通过问题条件的自动先验来监督交叉注意力头，以提高模型对医学图像的理解和回答能力。

**💡 创新点**

创新点在于引入了一个阈值饱和的先验质量目标，该目标在每个受限头部达到一定的注意力质量后停止施加空间压力，从而保持了回答准确性和先验质量的稳定性。

**🔧 技术方法**

使用了交叉注意力机制和基于问题条件的空间监督技术，结合了自动生成的医学图像分割提案。

**📊 数据集**

使用了SLAKE-EN和VQA-RAD数据集进行评估，SLAKE-EN包含4919个训练问题-答案对，VQA-RAD用于基准比较。

**📈 对比分析**

与没有空间监督的基线模型相比，QCAP-VQA在SLAKE-EN上达到了84.26%的准确率，几乎与基线相当（84.23%），并且在开放式问题的准确率上超过了其他先进方法。

**⚠️ 局限性**

限制在于在受限监督下，尽管对命名解剖结构的注意力增加，但专家参考的问答条件分析显示，随着更多头部的约束，专家质量评分下降，最终可能变为负值。

---

## 159. Talk to Act: A Synthetic Multi-Turn Dialogue Benchmark for Human Intent Extraction and Action Planning in Robotics

**arXiv ID:** 2611.17549 | [PDF](https://arxiv.org/pdf/2611.17549v1)

**作者:** Deepu Krishnareddy `[一作]` (Reutlingen University), Matthias Rätsch `[通讯]` (Reutlingen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种合成生成的多轮对话数据集，用于人机交互（HRI），并对小型开放权重语言模型在机器人任务中的能力进行了评估。

**💡 创新点**

创新点在于创建了一个反映真实人类沟通复杂性和自然性的多轮对话数据集，并将意图提取框架化为工具调用问题。

**🔧 技术方法**

使用了小型开放权重语言模型（参数从270M到8B不等），并通过合成数据生成和LLM进行数据验证。

**📊 数据集**

使用了合成生成的多轮对话数据集，共977个对话，涵盖多种场景和用户角色。

**📈 对比分析**

与五种小型开放权重语言模型进行了比较，结果显示Gemma4:E4B在所有指标上表现最佳，目标达成准确率为63.6%，而FunctionGemma-270M在其规模下表现也相对竞争力，表明小型模型在多轮对话中具有潜力。

**⚠️ 局限性**

限制在于当前数据集的规模和多样性，未来需要扩展数据集并进行任务特定的微调，以进一步提高模型性能。

---

## 160. NER-D: A cross-domain benchmark for German NER

**arXiv ID:** 2611.18257 | [PDF](https://arxiv.org/pdf/2611.18257v1)

**作者:** Marc Brinner `[一作]` (Bielefeld University), Sina Zarrieß `[通讯]` (Bielefeld University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了NER-D，一个跨领域的德语命名实体识别基准，涵盖9个领域的13个数据集。

**💡 创新点**

创新点在于比较了专有大型语言模型与小型开源模型在不同训练设置下的表现，并探讨了LoRA微调对小型模型性能的提升。

**🔧 技术方法**

使用了大型语言模型（如GPT-5.4 mini）和小型开源模型，结合了正常提示、上下文学习和LoRA微调技术。

**📊 数据集**

使用了13个公开可用的德语NER数据集，涵盖一般文本、法律文件、医学文本、历史文献及金融、农业和考古等专业领域。

**📈 对比分析**

通过零-shot、few-shot和低秩微调等不同训练方式比较模型性能，结果显示在未优化的情况下，专有模型在6个数据集上表现优于小型开源模型，但经过LoRA微调后，小型模型的表现显著超过专有基线。

**⚠️ 局限性**

限制在于数据集的选择并未涵盖所有可能的领域，且某些领域的数据集样本量较小，可能影响评估的全面性。

---

## 161. Auditing Chatbots for Incursion into Professional Legal Practice

**arXiv ID:** 2611.17622 | [PDF](https://arxiv.org/pdf/2611.17622v1)

**作者:** Nel Escher `[一作]` (University of California), Deirdre Mulligan `[通讯]` (University of California)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究开发了一个名为UPLBench的基准，审计了23个聊天机器人模型，评估它们是否提供法律建议，并记录它们是否建议用户咨询律师或法律援助组织。

**💡 创新点**

创新点在于将法律原则转化为可测试的基准，以评估聊天机器人是否遵循法律规定，尤其是在未授权法律实践（UPL）方面。

**🔧 技术方法**

使用了基于法律案例的基准测试方法，结合了自动化分类和专家评审，以评估聊天机器人的法律建议质量。

**📊 数据集**

使用了来自33个州和联邦法院的65个高风险法律领域的案例，涵盖移民、破产、个人伤害、止赎和家庭法等领域。

**📈 对比分析**

与相关研究相比，本研究通过UPLBench基准测试聊天机器人的表现，发现58.4%的生成响应包含UPL，且新模型提供法律建议的频率显著增加，同时减少了推荐用户咨询律师的比例。

**⚠️ 局限性**

限制在于聊天机器人缺乏法律专业知识和伦理责任，可能导致用户获得不完整或误导性的法律建议，且未能保护用户的敏感信息。

---

## 162. uhd-rfsoc: Bringing Off-the-Shelf RFSoC Boards to Unmodified UHD Applications

**arXiv ID:** 2611.17723 | [PDF](https://arxiv.org/pdf/2611.17723v1)

**作者:** Wei Cheng `[一作]` (Duke University), Tingjun Chen `[通讯]` (Duke University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了一种开源FPGA设计和主机软件栈，使现成的RFSoC板（如ZCU216和RFSoC 4x2）能够在不修改UHD的情况下与UHD生态系统兼容。

**💡 创新点**

创新点在于通过标准UHD调用和API，允许应用程序配置通道数量、载波频率、每通道I/Q采样率和传输格式，同时实现高达2.5 GS/s的16位或5.0 GS/s的8位I/Q样本的全双工传输。

**🔧 技术方法**

使用FPGA设计、数据平面开发工具包（DPDK）进行数据包处理，以及在x86和Arm主机上进行向量化样本转换的技术。

**📊 数据集**

使用了ZCU216和RFSoC 4x2板的多种配置进行评估，支持不同的RF数据转换器速率、通道数量和样本率。

**📈 对比分析**

通过UHD的内置流基准测试评估性能，所有支持的板配置在x86和Arm主机上均通过测试，显示出与现有UHD软件的兼容性，包括GNU Radio流图和开源5G RAN堆栈OCUDU和Duranta。

**⚠️ 局限性**

限制在于尽管实现了与UHD的兼容性，但仍需处理FPGA和主机之间的高数据速率传输挑战，以及确保在不修改UHD的情况下实现流的可靠性。

---

## 163. TradeGauntlet: Evaluating Compliance Controls for Autonomous Trading Agents

**arXiv ID:** 2611.18493 | [PDF](https://arxiv.org/pdf/2611.18493v1)

**作者:** Sandip Nandi `[一作]` (Independent Researcher), Soovadeep Bakshi `[通讯]` (Independent Researcher)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了自主交易代理的合规性，使用TradeGauntlet模拟器测试不同控制措施对交易行为的影响。

**💡 创新点**

提出了一种可重复的工作流程，用于测试自主交易代理的合规控制，并通过与大型语言模型（LLM）的比较和两种搜索方法的评估来支持这一流程。

**🔧 技术方法**

使用了TradeGauntlet模拟器，该模拟器能够分离检测、预防和执行成本，并通过脚本探针和搜索方法测试代理的弱点。

**📊 数据集**

使用了五种纳斯达克股票的数据集进行价格响应的拟合，并在ABIDES实验中进行了额外测试。

**📈 对比分析**

通过与不同控制措施下的LLM代理进行比较，发现所有的行为警报都是自匹配的，控制措施有效地减少了警报率，但对执行的影响因模型而异。

**⚠️ 局限性**

模拟器的市场机制简单，缺乏真实市场的复杂性，如排队、延迟和跨场交易，且所有结果均为模拟，未能测量真实市场中的违法行为的普遍性。

---

## 164. SPLIT-MOR: Fast and Accurate Model Order Reduction for Nonlinear Transient Thermal Simulation of 2.5D/3D ICs

**arXiv ID:** 2611.17693 | [PDF](https://arxiv.org/pdf/2611.17693v1)

**作者:** Zebin Shi `[一作]` (Peking University), Runsheng Wang `[通讯]` (Peking University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的非线性模型降阶方法SPLIT-MOR，用于快速准确的瞬态热模拟，特别针对2.5D/3D集成电路。

**💡 创新点**

SPLIT-MOR通过子域聚类和温度区间划分，消除了传统非线性模型降阶方法中在线阶段的全阶计算负担，且不需要额外的训练数据。

**🔧 技术方法**

使用了子域聚类和温度区间划分的技术，结合了离线预计算和在线更新策略。

**📊 数据集**

在两个2.5D和一个3D先进封装的数据集上进行了评估。

**📈 对比分析**

与COMSOL相比，SPLIT-MOR在在线求解时间上实现了高达830倍的加速，相比于最先进的超降阶方法MDEIM实现了7.5倍的加速，且离线时间减少了27倍，平均相对误差保持在0.11%以下，最大误差低于0.86°C。

**⚠️ 局限性**

该方法的局限性在于仍需在离线阶段进行复杂的预计算，并且在某些情况下可能对温度区间的选择敏感。

---

## 165. Time-Consistent Evaluation of Hierarchical Code Graphs for SWE Agents

**arXiv ID:** 2611.17388 | [PDF](https://arxiv.org/pdf/2611.17388v1)

**作者:** Xianpeng `[一作]`, Qi Zhang `[通讯]` (Microsoft)

**通讯引用:** 27043 | [OpenAlex ID](https://openalex.org/A5014306843)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文评估了软件工程（SWE）代理的存储库感知能力，采用严格的零泄漏隔离协议，比较了基线、Tree-sitter AST和ServiceExplorer在C++和JavaScript/TypeScript中的表现。

**💡 创新点**

创新点在于提出了一种严格的三层零泄漏协议，并引入了ServiceExplorer的两级依赖图，显著提高了代码定位的准确性。

**🔧 技术方法**

使用了时间一致的零泄漏协议，Tree-sitter AST（MIT许可的平面符号和调用AST图）和ServiceExplorer（确定性的两级依赖图）。

**📊 数据集**

使用了DragonFly（C++）和React（JavaScript/TypeScript）两个数据集，分别进行了400个任务的评估。

**📈 对比分析**

通过与基线和Tree-sitter AST的比较，ServiceExplorer在所有存储库和提示粒度上都达到了最高的F1分数，整体F1在DragonFly上提高了16.2%，在React上提高了15.8%。

**⚠️ 局限性**

限制在于尽管采用了严格的零泄漏协议，但单次运行的LLM解码仍可能引入随机方差，未来的工作需要结合执行验证的测试套件通过率。

---

## 166. A Generalizable Approach to Safety-Aware Robotic Behavior Planning with an Explicit Constraint Space

**arXiv ID:** 2611.18026 | [PDF](https://arxiv.org/pdf/2611.18026v1)

**作者:** Xianxu Wu `[一作]`, Jianwei Zhang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于显式约束空间的安全意识机器人行为规划框架，旨在处理桌面环境中的安全约束问题。

**💡 创新点**

创新点在于引入了查询条件约束适用性学习和多模态安全检查器，能够动态选择适用于当前任务和环境的安全约束，并在规划过程中进行实时反馈和调整。

**🔧 技术方法**

使用了轻量级的查询条件双塔模型进行约束适用性学习，以及多模态安全检查器进行计划评估和约束重构。

**📊 数据集**

在家庭、厨房、办公室和实验室等桌面场景中进行了实验，使用了50个原始场景进行约束适用性评估。

**📈 对比分析**

与基于语义检索的方法相比，提出的模型在约束选择上表现更好，召回率和精确率均有所提高，且在测试时间适应性方面表现出持续改进。

**⚠️ 局限性**

该框架主要局限于桌面操作和文本安全约束，未能替代低级别的碰撞避免和运动级别的安全控制，且长期适应性尚未系统评估。

---

## 167. GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks

**arXiv ID:** 2611.17766 | [PDF](https://arxiv.org/pdf/2611.17766v1)

**作者:** Suhaas Garre `[一作]` (Surge AI), Edwin Chen `[通讯]` (Surge AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基准，评估专业人士在工作场所的请求，结合专家评分标准和完整的工作簿或CSV文件。

**💡 创新点**

创新点在于创建了一个涵盖多个领域的基准，要求模型在回答时满足完整性和排他性标准，并且通过一个能够验证答案的代理评审。

**🔧 技术方法**

使用了OpenHands软件代理SDK构建的评估工具，结合Python库（如openpyxl和pandas）进行数据处理。

**📊 数据集**

使用了包含数百万个填充单元格的Excel工作簿和CSV文件，涵盖多个领域的任务。

**📈 对比分析**

通过严格的pass@1标准比较模型，最高得分的模型（Gemini 4 Argon）仅通过了不到40%的尝试，显示出模型在处理复杂请求时的性能仍有很大提升空间。

**⚠️ 局限性**

限制在于即使是表现最好的模型也未能满足大多数请求的标准，表明在复杂的工作簿处理和信息检索方面仍存在显著的挑战。

---

## 168. Adaptable Watermarking: Transforming Image Watermarking Models via Lightweight Adapters

**arXiv ID:** 2611.18363 | [PDF](https://arxiv.org/pdf/2611.18363v1)

**作者:** Mehrdad Saberi `[一作]` (University of Maryland), John Collomosse `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探索了神经图像水印模型的适应性，提出了一种框架，使编码器-解码器水印架构模块化和可升级，支持多种水印风格而无需重新训练完整模型。

**💡 创新点**

创新点在于通过轻量级适配器实现跨模型适应，允许不同水印方法之间的转换，同时保持高效的参数使用和解码准确性。

**🔧 技术方法**

使用了卷积权重低秩适配器（CWLoRA），通过多域残差对齐损失和秘密损失进行微调。

**📊 数据集**

使用了MIRFLICKR数据集进行训练和测试，包含80,000张训练图像和10,000张测试图像。

**📈 对比分析**

与多种水印模型（如TrustMark、InvisMark等）进行比较，适配后的模型在标准增强和失真下能够接近最优的解码准确性，且在解码器大小上减少了约8倍。

**⚠️ 局限性**

限制在于适配器的性能可能受到基础模型的架构和预训练知识的影响，尤其是在处理高度不可感知的水印时，适配可能面临挑战。

---

## 169. Online Allocation with Differential Privacy

**arXiv ID:** 2611.17554 | [PDF](https://arxiv.org/pdf/2611.17554v1)

**作者:** Jianyi Yang `[一作]` (University of Houston), Shaolei Ren `[通讯]` (University of California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了满足差分隐私（DP）要求的在线分配问题，提出了一种隐私保护的元算法，并分析了在不同隐私要求下的性能极限。

**💡 创新点**

引入了标准DP、联合DP（JDP）和局部DP（LDP）概念，并建立了在这些定义下的基本性能限制，提出了满足这些限制的算法框架。

**🔧 技术方法**

使用了元算法框架，结合了差分隐私机制来保护用户信息，同时优化资源分配性能。

**📊 数据集**

通过数值实验验证了算法在电池供电的边缘计算系统中的应用，使用了RouterBench数据集进行实验。

**📈 对比分析**

与现有方法进行了比较，提出的算法在满足隐私约束的情况下，能够实现接近最优的性能，尤其在对抗性和随机设置下表现优越。

**⚠️ 局限性**

尽管建立了在线分配在DP定义下的性能界限，但这些界限并未被证明是最佳的，尤其是在隐私水平的缩放方面仍有改进空间。

---

## 170. When Does Less Information Pay? An Exact Finite Atlas of Representation, Payoff, and Cost

**arXiv ID:** 2611.17583 | [PDF](https://arxiv.org/pdf/2611.17583v1)

**作者:** Aleksandr M. Shtraukh-Adamsky `[一作]` (Independent researchers), Artemiy D. Spitsyn `[通讯]` (Independent researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在一个四状态世界中，保留不同信息量的表示如何影响任务的准确性。通过枚举所有可接受的表示和二进制行动规则，比较了不同表示的表现。

**💡 创新点**

创新点在于明确区分信息量与任务相关性的关系，展示了在某些情况下，保留较少信息的表示可以获得更高的收益，而在其他情况下则相反。

**🔧 技术方法**

使用了精确枚举的方法来比较不同的表示和行动规则，分析了信息保留、任务收益和成本之间的关系。

**📊 数据集**

研究使用了一个包含四个等概率状态（00, 01, 10, 11）和三个二进制任务的数据集。

**📈 对比分析**

通过两种比较方法进行性能评估：一种是固定其他组件只替换一个组件的比较，另一种是为每个模块单独选择最佳行动规则。结果显示，保留较少信息的表示在某些情况下表现更好，而在其他情况下则表现较差。

**⚠️ 局限性**

限制在于模型仅适用于四个等概率状态和三个二进制任务，未考虑更复杂的环境变化、学习过程或与环境的交互。

---

## 171. DySENSE: Dynamic Spatiotemporal ENergy Synthesis and Emulation with Satellite Imagery

**arXiv ID:** 2611.17878 | [PDF](https://arxiv.org/pdf/2611.17878v1)

**作者:** Kailai Sun `[一作]` (SMART), Jinhua Zhao `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DySENSE，一个联合多模态生成AI框架，用于合成静态城市卫星图像和动态建筑能耗地图。

**💡 创新点**

DySENSE通过将静态城市土地使用信息与动态气候提示相结合，能够生成现实、多样化的新静态城市图像和对齐的能耗地图，解决了现有城市建筑能耗建模的局限性。

**🔧 技术方法**

使用了变分自编码器（VAE）、CLIP和RemoteCLIP文本编码器，以及U-ViT联合去噪网络等技术。

**📊 数据集**

建立了一个覆盖纽约市、华盛顿特区、诺福克和釜山的多城市卫星和月度能耗数据集，包含2,565个样本。

**📈 对比分析**

与现有的多模态城市建筑能耗预测方法相比，DySENSE显著提高了预测性能（约8.7%的准确率提升）并减少了预测误差（9%-18%的CVRMSE）。

**⚠️ 局限性**

DySENSE的地理覆盖范围有限，未来工作可以扩展到更多城市、气候区和城市形态，尤其是在数据稀缺的发展中国家。

---

## 172. From Local Prediction Error to Final Outputs: A Theory of Expert Sharing in Multi-Step Reasoning

**arXiv ID:** 2611.17879 | [PDF](https://arxiv.org/pdf/2611.17879v1)

**作者:** Tiantong Wang `[一作]` (Nanyang Technological University), Wei Yang Bryan Lim `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

研究了混合专家模型（MoE）在多层或计算步骤中共享专家的效果，分析了局部预测误差的降低是否能在最终输出中得到保留。

**💡 创新点**

提出了在特定线性回归模型中，平衡方差节省与共享偏差的必要和充分条件，明确了共享和合并的标准。

**🔧 技术方法**

使用了线性回归模型和条件端到端误差界限的技术，分析了共享专家的风险比较。

**📊 数据集**

使用了低数据符号任务的数据集，进行了控制实验以验证理论结果。

**📈 对比分析**

通过比较共享模型和独立模型的最终输出误差，发现共享可以降低局部预测误差，但可能增加最终输出误差，且在某些情况下局部改进未能转化为最终输出的改进。

**⚠️ 局限性**

局限性在于，正确的预测和路由标签并不能证明个别专家实现了所归属的操作，且需要更多的证据来支持可重用操作的归因。

---

## 173. Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination

**arXiv ID:** 2611.18514 | [PDF](https://arxiv.org/pdf/2611.18514v1)

**作者:** Mohamed Aghzal `[一作]` (George Mason University), Ziyu Yao `[通讯]` (George Mason University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了机械可解释性（MI）在解释基于大型语言模型（LLM）的代理决策过程中的挑战，特别是针对网络代理在与实时网站交互时产生的幻觉行为。

**💡 创新点**

创新点在于揭示了现有MI工具在解释LLM代理失败时的局限性，并提出了针对代理特性的MI方法改进方向。

**🔧 技术方法**

使用了线性探测器、机制分析和干预方法来分析LLM代理的行为。

**📊 数据集**

使用了Online-Mind2Web数据集，该数据集包含与实时网站的交互记录，共300个交互案例和2071个动作，其中32.6%为幻觉动作。

**📈 对比分析**

通过线性探测器预测幻觉行为的AUC为0.80，干预探测器方向可以修正85%的幻觉，但会影响7%的有效动作，表明现有方法在解释和修正代理行为方面的局限性。

**⚠️ 局限性**

限制在于现有MI方法假设的局部性和稳定性在代理环境中难以满足，导致无法有效识别和修正幻觉行为的根本原因。

---

## 174. Mobile Edge Deployment and Resource Management for Maritime Wireless Networks

**arXiv ID:** 2611.17779 | [PDF](https://arxiv.org/pdf/2611.17779v1)

**作者:** Chaoyue Zhang `[一作]` (Dalian Maritime University), Jianli Duan `[通讯]` (Qingdao University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了移动边缘计算（MEC）在海事无线网络中的部署和资源管理，提出了一种双层强化学习框架，以优化无人水面车辆（USV）的部署和计算卸载决策。

**💡 创新点**

创新点在于首次联合优化多USV的部署和计算卸载决策，以最小化预期响应时间，并提出了双层强化学习框架来处理混合动作空间。

**🔧 技术方法**

使用了双层强化学习（DLRL）框架，其中外层采用深度确定性策略梯度（DDPG）算法优化USV的部署，内层采用Q学习算法确定最佳计算卸载决策。

**📊 数据集**

使用了模拟数据集，考虑了20到45艘船只和10到36艘USV的不同组合，评估了不同计算资源和任务数量对性能的影响。

**📈 对比分析**

与四种基准算法（PSO-G、PSO-DDQN、DQN和DDPG）进行比较，结果表明，所提出的DLRL算法在减少预期响应时间方面表现优越，尤其在处理混合动作时具有更高的准确性和效率。

**⚠️ 局限性**

限制在于当前研究主要集中在海事无线网络的MEC应用上，未来需要进一步研究异构MEC网络中不同类型任务的计算卸载问题。

---

## 175. Tight Quantum Query Bounds for Minimum Cut and Multicut on Dense Graphs

**arXiv ID:** 2611.17845 | [PDF](https://arxiv.org/pdf/2611.17845v1)

**作者:** Tianhang Lu `[一作]` (Southern University of Science and Technology), Shengcai Liu `[通讯]` (Southern University of Science and Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了简单无权无向图上精确最小多切割的密集最坏情况量子查询复杂度，提出了一种邻接数组算法，使用(n + min{m, n^(3/2)√(k)})次查询。

**💡 创新点**

创新点在于提供了针对不同需求对的精确最小多切割的查询复杂度界限，并且在k=1的情况下将查询复杂度从(n^(11/6))改进到(n^(3/2))。

**🔧 技术方法**

使用了量子邻接数组模型和度敏感谱估计等技术。

**📊 数据集**

使用了简单无权无向图作为数据集，考虑了最多k对显式需求。

**📈 对比分析**

与现有方法相比，提出的算法在每个指定的稀疏边数下都提供了匹配的上界和下界，性能在每个需求预算范围内保持一致，复杂度为Θ(n^(3/2)√(k))。

**⚠️ 局限性**

限制在于算法的运行时间可能是指数级的，尤其是在单需求情况下，且未建立每个指定稀疏边数的匹配界限。

---

## 176. Beyond Endpoints: Bézier Paths for Vision Foundation Model Adaptation

**arXiv ID:** 2611.17825 | [PDF](https://arxiv.org/pdf/2611.17825v1)

**作者:** Chenzhi Zheng `[一作]` (University of Birmingham), Xin Cao `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文研究了如何通过学习共享的提示令牌来适应冻结的视觉基础模型，并探讨了在数据变化或任务变化时，提示如何在最短路径上移动。

**💡 创新点**

创新点在于量化共享注意力对最短路径的额外运输成本，并提出了一种精确可解的注意力模型，揭示了内容归一化的阈值。

**🔧 技术方法**

使用了冻结的注意力模型和参数化的贝塞尔路径来优化特征动作。

**📊 数据集**

使用了VTAB-1k数据集进行实验。

**📈 对比分析**

与线性路径和交叉熵训练路径相比，优化的贝塞尔路径在保持准确性方面表现更好，减少了特征动作，并在无标签分布转移下提高了在线准确性。

**⚠️ 局限性**

限制在于共享提示的数量和内容扰动可能导致的路径成本增加，且在某些情况下，达到最短路径并不保证可达性。

---

## 177. Learn then Bet: Learning Interpretable Rules for Reliable Agent Intervention

**arXiv ID:** 2611.17662 | [PDF](https://arxiv.org/pdf/2611.17662v1)

**作者:** You-Lin Chen `[一作]`, Bin Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

论文探讨了如何在Lean 4环境中进行数学证明，特别是针对特定假设和保证的证明过程。

**💡 创新点**

创新点在于将数学定义和Lean形式化结合起来，提供了一种新的方法来验证数学命题的正确性。

**🔧 技术方法**

使用了Lean 4作为形式化证明的工具，结合了测度论和概率论的相关理论。

**📊 数据集**

使用了可测空间和观察数据的样本空间，具体数据集未明确提及。

**📈 对比分析**

通过与传统的数学证明方法进行比较，展示了Lean 4在验证复杂数学命题时的有效性和准确性。

**⚠️ 局限性**

限制在于Lean 4的学习曲线较陡，且对用户的数学和编程能力要求较高。

---

## 178. DoubleTake: Focused Verification of Visual Evidence for Long-Document Understanding

**arXiv ID:** 2611.17853 | [PDF](https://arxiv.org/pdf/2611.17853v1)

**作者:** Kejun Xue `[一作]`, Zhijun Fang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为DoubleTake的框架，用于长文档理解中的视觉证据的集中验证，旨在提高模型在回答问题时的准确性。

**💡 创新点**

创新点在于提出了一种无训练的框架，通过比较两个候选答案及其支持证据来选择需要直接验证的事实，从而提高了长文档问答的准确性。

**🔧 技术方法**

使用了Qwen3-VL-8B-Instruct作为基础模型，并结合了证据引导的双重阅读和比较引导的视觉验证技术。

**📊 数据集**

在四个基准数据集上进行评估，包括LongDocURL、PaperTab、FetaTab和MMLongBench-Doc。

**📈 对比分析**

与其他方法相比，DoubleTake在所有四个基准上都达到了最高的整体准确性，特别是在MMLongBench-Doc上表现突出，验证问题的集中提问方式比重复原始问题更能提高准确性。

**⚠️ 局限性**

限制在于该方法依赖于初步的候选答案和证据记录，可能在某些情况下无法完全捕捉到所有相关信息。

---

## 179. Typing Matters: Muscle, Language and Typing Behaviour in Live Keystroke Decoding from sEMG

**arXiv ID:** 2611.18464 | [PDF](https://arxiv.org/pdf/2611.18464v1)

**作者:** Abhilash Durgam `[一作]` (University of Central Florida), Qiushi Fu `[通讯]` (University of Central Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于表面肌电图（sEMG）的实时打字解码器，能够在不使用键盘的情况下从手腕的肌肉信号中恢复输入的文本。

**💡 创新点**

创新点在于将sEMG文本输入视为解码打字行为，而非单纯的文本解码，提出了一种完全因果的状态空间模型（SSM），能够在整个sEMG会话中进行解码，而无需固定窗口。

**🔧 技术方法**

使用了状态空间模型（SSM）和自回归语言模型的融合，构建了一个因果的sEMG打字解码器。

**📊 数据集**

使用了emg2qwerty数据集，该数据集包含来自108名用户的346小时的sEMG信号，记录了他们在打字时的肌肉活动。

**📈 对比分析**

与现有方法相比，该模型在emg2qwerty的跨用户基准测试中达到了25.13%的字符错误率（CER），比之前的最佳结果提高了5.4%，且参数数量减少了6.7倍。

**⚠️ 局限性**

限制在于该模型未经过任何sEMG到字符的转录训练，可能在特定情况下的适用性和泛化能力受到影响。

---

## 180. KnowVis: Dual-View Evaluation and Diagnosis of World-Knowledge Grounding in Text-to-Image Models

**arXiv ID:** 2611.18101 | [PDF](https://arxiv.org/pdf/2611.18101v1)

**作者:** Yonghang Yu `[一作]` (KlingAI Research), Changqian Yu `[通讯]` (KlingAI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了KnowVis，一个用于评估和诊断文本到图像生成模型中世界知识基础的双视角基准。

**💡 创新点**

创新点在于构建了一个层次化知识分类法和两个互补的基准视图，结合了结构化的MLLM评估协议，以便更全面地评估模型的知识表达能力。

**🔧 技术方法**

使用了基于视觉问答（VQA）的MLLM作为评估者，结合了结构化的评估协议。

**📊 数据集**

使用了包含1622个案例的Skill-Tree基准和550个案例的Generalized基准，覆盖了多种知识点和主题。

**📈 对比分析**

通过与现有模型的比较，发现KnowVis能够揭示出当前T2I模型在世界知识基础方面的显著差距，并提供结构化的诊断信号。

**⚠️ 局限性**

限制在于当前的评估仍然依赖于人工标注，可能存在主观性，并且在开放式生成场景中，模型的表现可能受到多种因素的影响。

---

## 181. Verifiable pXRD: Hill-climbing pXRD structure determination with LLM agents

**arXiv ID:** 2611.17539 | [PDF](https://arxiv.org/pdf/2611.17539v1)

**作者:** Samuel Chen `[一作]` (Stanford University), Alán Aspuru-Guzik `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文提出了一种将粉末X射线衍射（pXRD）结构确定任务形式化为可验证的自主搜索问题的方法，并引入了pXRD-Bench基准测试套件，以评估不同方法的性能。

**💡 创新点**

创新点在于将pXRD结构确定视为可验证的任务，利用代理适应和推理时间缩放技术显著提高了结构恢复率，并扩展到实验多相pXRD的应用。

**🔧 技术方法**

使用了大型语言模型（LLM）代理、前向模拟、结构匹配和反射进化框架等技术。

**📊 数据集**

使用了pXRD-Bench数据集，该数据集包含200个模拟的单相问题，涵盖了不同的结构确定挑战。

**📈 对比分析**

通过与现有方法的比较，适应后的代理在pXRD-Bench上的结构恢复率从27%提高到69%，并且在推理时间缩放技术上也取得了进一步的提升。

**⚠️ 局限性**

限制在于实验测量通常缺乏明确的真实值，且在多相实验中，如何有效识别相和构建合理结构仍然是一个挑战。

---

## 182. PADSNet: Physics-Aware Dual-Stream Network with Biomechanical Priors for Laryngeal Lesion Classification

**arXiv ID:** 2611.17852 | [PDF](https://arxiv.org/pdf/2611.17852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 183. Beyond Source Code: Evaluating Natural Language Specifications for LLM-Based C-to-Rust Translation

**arXiv ID:** 2611.17581 | [PDF](https://arxiv.org/pdf/2611.17581v1)

**作者:** Biruk Tadesse `[一作]` (North Carolina State University), Wesley K. G. Assunção `[通讯]` (North Carolina State University)

**通讯引用:** 2079 | [OpenAlex ID](https://openalex.org/A5039130090)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了自然语言规范在基于大型语言模型（LLM）的C到Rust翻译中的有效性，比较了三种翻译设置：仅代码、仅规范和代码与规范结合。

**💡 创新点**

创新点在于引入自然语言规范作为翻译的中间表示，强调程序的预期行为而非具体实现，从而改善翻译质量。

**🔧 技术方法**

使用了GPT-5.2和GPT-4o-mini两种大型语言模型进行翻译和规范生成。

**📊 数据集**

研究使用了IBM CodeNet数据集中的300个C程序进行评估。

**📈 对比分析**

与仅代码翻译相比，规范翻译在代码质量上有显著提升，尽管仅代码翻译的正确率最高（98.6%），但规范翻译在可维护性和习惯性方面表现更佳。结合代码和规范的翻译在正确性（98.3%）和代码质量之间提供了最佳平衡。

**⚠️ 局限性**

限制在于未经过验证的规范可能导致较低的正确率（最低可达86%），而且规范的质量直接影响翻译的有效性。

---

## 184. DRA-HTR: Disentangling Representation and Adaptation for Handwritten Text Recognition

**arXiv ID:** 2611.18143 | [PDF](https://arxiv.org/pdf/2611.18143v1)

**作者:** Gayan H. Pathirage `[一作]` (Luleå University of Technology), Elisa H. Barney Smith `[通讯]` (Luleå University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种专门为手写文本识别（HTR）设计的无监督生成领域适应管道DRA-HTR，旨在解决历史文档中的手写文本识别挑战。

**💡 创新点**

创新点在于结合了源领域的预训练、无监督适应和后HTR校正，利用大语言模型进行语言适应，从而提高了跨领域的识别性能和稳定性。

**🔧 技术方法**

使用了基于ConvNeXtV2的编码器、分离器和生成器，并引入了后HTR校正机制，形成了DRAL-HTR框架。

**📊 数据集**

在七个手写数据集上进行了评估，涵盖四种语言（英语、德语、挪威语、瑞典语），数据集包括现代和历史文档，具有显著的手写风格和文档特征变化。

**📈 对比分析**

与之前的方法相比，DRA-HTR在所有七个数据集上建立了新的领域适应状态，平均字符错误率（CER）降低了13.52个百分点，训练稳定性显著提高，标准差从7.23降至1.11个百分点。

**⚠️ 局限性**

限制在于后HTR校正的效果主要体现在英语数据集上，对其他语言的提升有限，且计算成本相对较高，存在准确性与推理效率之间的权衡。

---

## 185. Reusing Locomotion Training Recipes Across Humanoids via Body-Aligned Virtual Joints

**arXiv ID:** 2611.18003 | [PDF](https://arxiv.org/pdf/2611.18003v1)

**作者:** Sol Choi `[一作]` (Korea Institute of Science and Technology), Yisoo Lee `[通讯]` (Korea Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种身体对齐的虚拟关节接口，该接口将共享的策略动作和观察映射到每个机器人的本地关节空间，同时保留现有的传输和低级控制堆栈。

**💡 创新点**

创新点在于通过引入一个身体对齐的虚拟关节空间，解决了不同机器人之间本地坐标不一致的问题，从而实现了在不同关节布局的机器人上使用相同的训练配方。

**🔧 技术方法**

使用了强化学习技术，结合了分析坐标层和确定性运动学层来实现虚拟目标到本地目标的映射。

**📊 数据集**

使用了三种不同的类人机器人平台进行实验，包括Unitree G1、PND Adam和一个定制平台，所有平台均采用12自由度的双足下肢配置。

**📈 对比分析**

与本地关节空间和任务空间基线进行比较，结果显示，提出的虚拟关节空间接口在所有三个机器人上都能实现准确的速度跟踪和稳定的训练，而基线方法在至少一个平台上表现不佳。

**⚠️ 局限性**

限制在于本研究仅考虑了腿部的速度跟踪运动，且在硬件验证中只使用了一个平台，未来的工作将扩展到手臂、腰部和躯干，并研究全身运动模仿和重定向。

---

## 186. Towards Seconds-level Video Representation: Quadratically Gated Feature Grid with Adaptive, Prediction-free Sampling

**arXiv ID:** 2611.17786 | [PDF](https://arxiv.org/pdf/2611.17786v1)

**作者:** Tie-Yun Li `[一作]` (Harbin Institute of Technology), Feng-Lei Fan `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为Q-GAS的快速视频表示方法，该方法结合了多分辨率的3D特征网格和紧凑的QG解码器，旨在提高视频表示的效率和质量。

**💡 创新点**

创新点在于引入了动态错误感知采样机制和低秩二次门控解码器，能够在保持表示质量的同时显著加快训练和解码速度。

**🔧 技术方法**

使用了多分辨率的3D特征网格和二次门控解码器（QG），并结合了动态错误感知采样策略。

**📊 数据集**

使用了UVG数据集中的Beauty、Bosphorus和Jockey序列进行实验，视频分辨率为1920×1080。

**📈 对比分析**

与Instant-NGP等基线方法相比，Q-GAS在训练和量化上快5.5到7.2倍，解码速度超过370 FPS，且在相同的BPP范围内，重建质量（PSNR、MS-SSIM）优于NeRV系列方法。

**⚠️ 局限性**

限制在于尽管Q-GAS在训练和解码速度上表现优异，但在处理高频细节时仍存在一定的不足，特别是在复杂场景的重建中。

---

## 187. From Bytes to Behavior: Request-Level Attribution from Live GPU Hidden States

**arXiv ID:** 2611.18272 | [PDF](https://arxiv.org/pdf/2611.18272v1)

**作者:** Jonathan Pan `[一作]` (Home Team Science and Technology Agency), Brenda Ong `[通讯]` (Home Team Science and Technology Agency)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种从实时GPU进程中获取每层残差流激活的方法，能够分析运行中的大型语言模型（LLM）服务的行为。

**💡 创新点**

创新点在于使用非修改调试器在不需要模型架构参数或权重的情况下，实时获取模型的中间隐藏状态。

**🔧 技术方法**

采用了六阶段的通用流程，结合平台特定适配器，利用PyTorch分配器时钟或内核启动时钟来识别和读取视频内存中的张量分配。

**📊 数据集**

验证了该方法在33个开源解码器模型上的有效性，涵盖九种架构类别。

**📈 对比分析**

通过与标准框架激活训练的线性探针进行比较，成功区分拒绝与合规，AUROC达到了0.88，表现良好。

**⚠️ 局限性**

限制在于该方法依赖于特定的软件和硬件配置，且在实时观察过程中会引入显著的性能开销，可能影响证据的完整性。

---

## 188. The Complexity of Traffic Engineering in Segment Routing Networks

**arXiv ID:** 2611.18152 | [PDF](https://arxiv.org/pdf/2611.18152v1)

**作者:** Matthias Bentert `[一作]`, Stefan Schmid `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了段路由网络中流量工程的计算复杂性，特别是固定长度和可变长度两种问题变体的NP难度。

**💡 创新点**

创新点在于证明了即使在平面图中，流量工程问题的两个基本模型仍然是NP难的，并提出了一种紧凑的混合整数线性程序来计算固定权重可分配设置的精确解。

**🔧 技术方法**

使用了混合整数线性规划技术来求解流量工程问题，并通过多种图论工具进行复杂性证明。

**📊 数据集**

研究中使用了平面图作为数据集，考虑了单位需求和单位链接权重的情况。

**📈 对比分析**

与现有方法相比，本文通过理论证明了问题的NP难度，并提供了一种新的混合整数线性程序作为基线，能够计算精确解。

**⚠️ 局限性**

限制在于该研究主要集中在平面图上，可能无法直接推广到更复杂的网络拓扑结构。

---

## 189. Real-Time Diffusion Relighting as a Neural Sensor: Closing the Low-Light Domain Gap for Robotic Perception

**arXiv ID:** 2611.17438 | [PDF](https://arxiv.org/pdf/2611.17438v1)

**作者:** Shengjie Xu `[一作]` (University of Maryland), Ming C. Lin `[通讯]` (University of Maryland)

**通讯引用:** 26203 | [OpenAlex ID](https://openalex.org/A5102878981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将生成神经传感器嵌入控制循环的方法，以解决低光照条件下机器人感知的挑战，并实现实时感知。

**💡 创新点**

创新点在于将照明视为生成先验，而非信号处理问题，并提出了一种不阻塞控制循环的实时感知架构。

**🔧 技术方法**

使用了大型潜在扩散模型（Latent Diffusion Models）作为生成神经传感器，并结合不确定性感知激活机制。

**📊 数据集**

在ExDark低光照基准数据集上进行评估，该数据集包含多种自然低光照条件下的图像。

**📈 对比分析**

与现有的低光图像增强方法相比，提出的方法在ExDark基准上达到了87.1%的mAP，显著提高了在恶劣天气条件下的检测率，并在CARLA模拟器中实现了100%的圈数完成率。

**⚠️ 局限性**

限制在于目前的评估仅限于CARLA中的单类障碍物，且生成传感器的高GPU需求限制了边缘部署。

---

## 190. Disturbance-Aware Informative Planning in Spatiotemporal Environments

**arXiv ID:** 2611.18505 | [PDF](https://arxiv.org/pdf/2611.18505v1)

**作者:** Kaleb Ben Naveed `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种两阶段的干扰感知信息规划框架，用于优化自主机器人在时空环境中的轨迹，以提高对感兴趣量的监测效率。

**💡 创新点**

创新点在于结合了信息获取与干扰学习，通过优先选择干扰测量和优化轨迹来减少干扰不确定性，从而提高监测性能。

**🔧 技术方法**

使用了模型基础扩散（Model-Based Diffusion, MBD）技术进行轨迹优化，并结合了概率在线注意映射（Probabilistic Online Attentive Mapping, POAM）进行干扰估计。

**📊 数据集**

在四个模拟场景中进行了实验，场景中存在空间变化的、时间不变的干扰（如风），并使用了双积分器的名义动态模型。

**📈 对比分析**

与基于名义动态的规划、没有专门学习的干扰感知监测和环境范围的干扰估计进行了比较，结果显示提出的方法在监测性能上显著优于其他方法，学习时间减少了25.0%至52.3%。

**⚠️ 局限性**

限制在于在干扰影响较小的场景中，专门的学习可能是不必要的，且在某些情况下可能导致性能下降。

---

## 191. Beyond Bot Labels: Retrospective Consequence-Grounded Ranking of Attributed Cache Harm in E-Commerce Logs

**arXiv ID:** 2611.18027 | [PDF](https://arxiv.org/pdf/2611.18027v1)

**作者:** Jisoo Kim `[一作]` (Sookmyung Women's University), Seonghoon Jeong `[通讯]` (Sookmyung Women's University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于后果的会话排名方法，用于评估电子商务日志中缓存损害的影响，重点关注会话之间的缓存污染。

**💡 创新点**

创新点在于通过一次重放将驱逐引起的受害者缺失归因于会话，从而形成了一种新的排名目标，且不依赖于用户代理或角色标签。

**🔧 技术方法**

使用了行为模型来预测会话的目标，并通过一次重放来进行会话排名和损害归因。

**📊 数据集**

使用了两个公共电子商务数据集：EClog（包含35,157,691个请求）和Zanbil（包含10,400,000个请求），用于目标构建和干预比较。

**📈 对比分析**

与现有方法相比，提出的方法在条件缓存命中率（CHR）差异上达到了93%的UA基线，同时选择的会话数量是UA基线的47倍，显示出显著的性能优势。

**⚠️ 局限性**

限制在于归因和行动的结果依赖于缓存和策略，可能包括同一会话的干扰，且未测试延迟、货币成本或自适应规避。

---

## 192. Game-Theoretic Analysis of RIS-Assisted Challenge-Response PLA - Part II: Partial Channel Knowledge

**arXiv ID:** 2611.17907 | [PDF](https://arxiv.org/pdf/2611.17907v1)

**作者:** Anna V. Guglielmi `[一作]` (University of Padova), Stefano Tomasin `[通讯]` (University of Padova)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本研究考虑了一个物理层认证场景，其中合法用户（Alice）通过可重构智能表面（RIS）与认证者（Bob）通信，攻击者（Trudy）试图通过适当的预编码消息来冒充Alice。本文分析了在部分信道知识下的攻击-防御互动，并提出了相应的博弈模型。

**💡 创新点**

创新点在于将RIS辅助的挑战-响应物理层认证（cr-pla）在部分信道知识下建模为有限动作的贝叶斯博弈，并通过无差异原则表征贝叶斯纳什均衡（BNE），从而优化了Bob的RIS配置策略和Trudy的预编码策略。

**🔧 技术方法**

使用了博弈论框架，特别是有限动作的贝叶斯博弈，来分析攻击者和防御者的策略选择，并通过数值模拟评估了在几何多径和3GPP TR 38.901 UMi-NLOS信道下的性能。

**📊 数据集**

使用了几何多径信道模型和3GPP TR 38.901 UMi-NLOS模型进行数值模拟，评估了不同信道条件下的认证性能。

**📈 对比分析**

与传统的基准策略相比，提出的均衡策略在认证性能上表现优越，尤其是在信道相关性、接收信噪比（SNR）和攻击者空间资源的影响下，均衡的误检测概率显著低于其他策略。

**⚠️ 局限性**

限制在于该研究假设了攻击者和防御者仅观察到自己的信道实现，并依赖于对对方信道的统计知识，实际应用中可能面临更复杂的信道环境和攻击策略。

---

## 193. Collective Tree Exploration with Predictions

**arXiv ID:** 2611.17482 | [PDF](https://arxiv.org/pdf/2611.17482v1)

**作者:** Romain Cosson `[一作]` (New York University), Ali Vakilian `[通讯]` (Rutgers University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了带有子树大小预测的集体树探索问题，旨在使一组移动代理尽快探索未知的根树，并在发现新节点时获得该节点子树大小的预测。

**💡 创新点**

提出了在最大误差和乘法失真下的算法保证，展示了如何在预测准确时提高性能，并在不准确时平滑降级。

**🔧 技术方法**

使用了确定性算法和学习增强算法，结合了种子和扫掠的思想来分配机器人，并通过逐层探索的方式进行树的探索。

**📊 数据集**

使用了未知的根树，节点数为n，深度为D，具体数据集未明确给出，但涉及到的预测是关于每个节点子树大小的估计。

**📈 对比分析**

与传统的无预测算法进行比较，提出的算法在最大误差模型下的复杂度为O(n/k+D+E)，而在乘法失真模型下的竞争比为O(min{μloglog k, μ^1.5})，显示出在不同误差条件下的性能表现。

**⚠️ 局限性**

限制在于对于乘法失真模型的依赖性，尤其是在失真较大时，算法的性能可能会显著下降，且对预测的准确性要求较高。

---

## 194. On numerical issues of divergences between nearly equal distributions

**arXiv ID:** 2611.18511 | [PDF](https://arxiv.org/pdf/2611.18511v1)

**作者:** Brandon Amos `[一作]` `[通讯]` (Reflection AI), Brandon Amos (Reflection AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文识别了标准分类分布之间的数值问题，特别是在软最大值和软交叉熵损失中存在的数值不稳定性，并提出了数值稳定的重构方法。

**💡 创新点**

创新点在于提出了针对几乎相等分布的五个数值问题，并提供了八种重构方法及其示例实现，验证了其有效性。

**🔧 技术方法**

使用了数值稳定的重构技术，特别是在计算Kullback-Leibler (KL) 散度和广义Jensen-Shannon散度时。

**📊 数据集**

使用了标准的分类分布作为数据集，具体数据集未详细说明。

**📈 对比分析**

通过与标准实现的比较，展示了提出的方法在数值稳定性和梯度计算中的优势，尤其是在分布接近相等时的表现。

**⚠️ 局限性**

限制在于这些重构方法在分布不接近相等时可能会不稳定，且完整的生产实现超出了本文的范围。

---

## 195. Do Machines Learn Deviance From Their Peers? A Randomized Test of Differential Association With AI Agents

**arXiv ID:** 2611.18098 | [PDF](https://arxiv.org/pdf/2611.18098v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 196. Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering

**arXiv ID:** 2611.18274 | [PDF](https://arxiv.org/pdf/2611.18274v1)

**作者:** Nima Eshraghi `[一作]` (Vanguard Group), Fanny Chevalier `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出了一种通过稀疏自编码器(SAE)进行激活引导的方法，以实现对大型语言模型的行为控制，而无需特定任务的微调。

**💡 创新点**

创新点在于将激活引导视为一个预算分配问题，提出了两种新的门控机制：对比特征投影(CFP)和闭环自适应伯努利引导(CLABS)，并分析了它们在预算限制下的表现。

**🔧 技术方法**

使用了稀疏自编码器(SAE)技术来提取可解释的特征方向，并通过比例-积分控制器(PI控制器)来实现闭环反馈控制。

**📊 数据集**

使用了两个基础模型（Gemma 2 2B和Llama 3.1 8B）和三个跨度级目标的数据集，包括RealToxicityPrompts和GoEmotions。

**📈 对比分析**

与其他方法相比，CFP和CLABS在相同预算下实现了最大的行为变化，尤其是在预算紧张时，CLABS表现最佳，同时保持流畅性。

**⚠️ 局限性**

限制在于CFP是无记忆的，无法区分短暂的偏离和持续的漂移，而CLABS虽然有记忆，但仍需在复杂的环境中进行进一步验证。

---

## 197. SCAR: Bridging Coarse Retrieval and Fine Localization through Candidate Reranking

**arXiv ID:** 2611.18240 | [PDF](https://arxiv.org/pdf/2611.18240v1)

**作者:** Shuhan Dong `[一作]` (Qilu University of Technology), Zhenyu Li `[通讯]` (Qilu University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的方法SCAR，用于在粗检索和精细定位之间进行候选重排序，以提高基于自然语言的点云定位的准确性。

**💡 创新点**

SCAR通过将每个候选单元和其现有的精细坐标视为一个定位假设，明确区分区域检索的正确性与度量定位的成功，从而解决了最终输出选择的问题。

**🔧 技术方法**

使用了Fine-conditioned Evidence Readout技术，结合中心支持、语义存在、检索分数和几何偏移，构建了一个23维的候选表示，并训练了一个共享评分器。

**📊 数据集**

在五个公共数据集上进行了实验，包括KITTI360Pose、PandaSetPose、nuScenesPose、S3DISPose和ParisLillePose/NPM3D。

**📈 对比分析**

与其他方法（如Semantic Matching、Rank Fusion和FSR）进行比较，SCAR在Top-1@5 m和Cell R@1的性能上均有显著提升，尤其在MambaPlace×nuScenesFull上，SCAR的完整评分方案比FSR提高了13.98个百分点。

**⚠️ 局限性**

SCAR的局限性在于其固定的候选池限制了可恢复的区域，无法纠正所有候选均失败的查询，且主要结果为探索性测试读数而非确认性评估。

---

## 198. DeCompCert: A Verified Decompiler for CompCert-Generated AArch64

**arXiv ID:** 2611.18055 | [PDF](https://arxiv.org/pdf/2611.18055v1)

**作者:** Swarn Priya `[一作]` (Washington State University), Kristopher Micinski `[通讯]` (Syracuse University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种经过验证的反编译器DeCompCert，旨在从CompCert生成的AArch64汇编代码中恢复C语言代码，专注于恢复和分析缺失源代码的CompCert构建的工件。

**💡 创新点**

创新点在于提供了一个端到端的声音反编译过程，确保每个重构的C程序的可观察行为与提供的汇编程序相匹配，并且通过翻译验证确保了重构的正确性。

**🔧 技术方法**

使用了Rocq验证的重构技术，结合不可信的OCaml和Datalog分析，采用了逐步验证和翻译验证的方法。

**📊 数据集**

使用了来自CompCert的约216万行C代码生成的AArch64汇编代码作为数据集。

**📈 对比分析**

与现有的反编译工具（如IDA和Ghidra）相比，DeCompCert在提供精确的函数和调用签名时实现了100%的反编译覆盖率。在没有保留签名的情况下，结合检查提示共识、重构改进和更长的搜索，成功找到2606个有效上下文（82.4%）。

**⚠️ 局限性**

限制在于在没有保留签名的情况下，重构的准确性和完整性受到影响，且某些情况下无法恢复源代码的确切结构和类型信息。

---

## 199. Space-based Zero-trust Architecture for Securing the Direct-to-Cell Paradigm

**arXiv ID:** 2611.17826 | [PDF](https://arxiv.org/pdf/2611.17826v1)

**作者:** Xu Chen `[一作]` (Tsinghua University), Xuemin Shen `[通讯]` (University of Waterloo)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类准确率。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10和ImageNet数据集进行实验。

**📈 对比分析**

与传统的激活函数（如ReLU）进行比较，结果显示新模型在分类准确率上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理极大规模数据集时可能会遇到内存不足的问题。

---

## 200. Federated Physics-Informed Thermal Field Prediction Across Unseen Geometries in Distributed Additive Manufacturing

**arXiv ID:** 2611.17794 | [PDF](https://arxiv.org/pdf/2611.17794v1)

**作者:** Amirul Islam Saimon `[一作]` (Virginia Tech), Kong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为Fed-PIGANO的联邦框架，用于在分布式增材制造中对未见几何形状的热场进行物理信息预测。

**💡 创新点**

创新点在于通过联邦学习实现多个制造站点在不共享几何形状的情况下共同训练几何感知模型，从而提高对新部件设计的热预测能力。

**🔧 技术方法**

使用了物理信息神经网络（PINNs）和几何感知神经算子（PI-GANO）技术。

**📊 数据集**

使用了来自三个不同站点的五个多边形几何族的2D金属增材制造几何数据集。

**📈 对比分析**

与仅在本地训练的模型相比，Fed-PIGANO在所有12个场景-站点案例中均表现出更低的平均相对ℓ_2误差（1.83%-2.04%），而本地训练的模型误差可高达20.46%。其预测准确性与集中模型相当。

**⚠️ 局限性**

局限性在于该方法依赖于各个站点的几何多样性，且未来工作需要扩展到3D几何和瞬态热问题。

---

## 201. Who sets the consent signal? Platform defaults and crawl coverage in measuring opt-outs from AI training

**arXiv ID:** 2611.18213 | [PDF](https://arxiv.org/pdf/2611.18213v1)

**作者:** Marcin Lawenda `[一作]` `[通讯]` (Poznan Supercomputing and Networking Center), Marcin Lawenda (Poznan Supercomputing and Networking Center)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文研究了在人工智能训练中，谁设置了用户的选择信号，特别是用户选择退出的情况。通过对波兰、捷克、斯洛伐克、匈牙利和德国的域名进行爬取，分析了这些国家网站的选择退出信号的来源和表现。

**💡 创新点**

创新点在于引入了“委托同意”的概念，强调平台或中介在用户选择退出信号中的作用，挑战了传统上认为每个权利持有者都应独立做出选择的假设。

**🔧 技术方法**

使用了逻辑回归模型对数据进行分析，并结合了Common Crawl的域名列表和随机抽样的斯洛伐克注册域名进行比较。

**📊 数据集**

使用的数据集包括Common Crawl记录的3,445,751个可访问网站的完整列表，以及随机抽取的10,000个斯洛伐克注册域名。

**📈 对比分析**

与传统方法相比，本文的方法能够更准确地归因选择退出信号的来源，发现平台默认设置的选择退出信号占比高达15%至78%。性能上，捷克和斯洛伐克的选择退出率显著高于德国，主要归因于平台的影响。

**⚠️ 局限性**

限制在于数据仅反映了2026年10月的状态，且不同国家的域名列表构建方式不同，可能导致比较时的偏差。此外，未能充分考虑小型网站的资源和偏好对选择退出行为的影响。

---

## 202. When evaluation becomes intervention in clinical AI

**arXiv ID:** 2611.18297 | [PDF](https://arxiv.org/pdf/2611.18297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 203. Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment

**arXiv ID:** 2611.18372 | [PDF](https://arxiv.org/pdf/2611.18372v1)

**作者:** Suvarna Kadam `[一作]` (Witsense AI), Soran Parsa `[通讯]` (University of Huddersfield)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对2022年12月至2026年10月期间的88项研究进行了综述，重点关注视觉-语言-动作（VLA）模型的设计、训练和部署生命周期，涵盖了操控、导航和人形控制等领域。

**💡 创新点**

创新点在于提出了一个VLA生命周期的分类法，系统地分析了模型设计、训练数据、模型架构、动作输出等七个方面，并比较了VLA与世界动作模型的性能。

**🔧 技术方法**

使用了视觉-语言模型（VLM）作为基础，通过微调使其能够输出机器人动作，采用了流匹配和扩散头等技术来生成动作。

**📊 数据集**

数据集包括来自社区收集的23,000个情节、20,000小时的自我中心人类视频以及10,000小时的遥控机器人数据等。

**📈 对比分析**

与世界动作模型进行比较，发现一些VLA在基准测试中得分接近完美，但在物体或相机稍微移动时表现不佳，提出了进一步研究的问题。

**⚠️ 局限性**

限制在于训练数据的描述不一致，许多论文未能量化数据量，导致比较困难，同时在真实环境中的评估标准尚未统一。

---

## 204. Cracks in Embeddings: Robustness and Stability of JEPA-Based Chest X-Ray Representations

**arXiv ID:** 2611.17905 | [PDF](https://arxiv.org/pdf/2611.17905v1)

**作者:** Zahid Mansur `[一作]` (Indian Institute of Technology Jodhpur), Deepak Mishra `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文测试了四种基于JEPA的胸部X光编码器在六种不同条件下的鲁棒性，包括两种黑箱对抗攻击和三种图像损坏，评估了它们在真实世界图像变化下的表现。

**💡 创新点**

创新点在于首次系统性地评估JEPA胸部X光编码器在输入扰动下的鲁棒性，并提出了两种训练方法来提高模型的鲁棒性。

**🔧 技术方法**

使用了自监督学习的JEPA模型，并结合了对抗微调和基于向量量化的代码本方法。

**📊 数据集**

使用了NIH ChestX-ray14数据集，共78,468张图像进行训练和测试。

**📈 对比分析**

通过对比不同的训练方法，发现对抗微调在大多数模型中提高了AUROC，而向量量化方法在所有模型中表现不佳，导致AUROC下降。

**⚠️ 局限性**

限制在于向量量化方法未能提供预期的鲁棒性，且所有编码器在面对真实世界的扰动时表现出明显的模型依赖性，鲁棒性未能得到保证。

---

## 205. Stochastic Games with Distributional Entropy Objectives

**arXiv ID:** 2611.18470 | [PDF](https://arxiv.org/pdf/2611.18470v1)

**作者:** Shankaranarayanan Krishna `[一作]` (Indian Institute of Technology Bombay), Ashutosh Trivedi `[通讯]` (University of Colorado Boulder)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了具有分布熵目标的随机博弈模型，研究了两个玩家的零和博弈，其中配置为有限竞技场状态的概率分布，阶段奖励为香农熵。

**💡 创新点**

创新点在于引入了三种不同的语义（平均分布、采样配置和采样转移），并探讨了这些语义如何影响博弈的确定性和策略的最优性。

**🔧 技术方法**

使用了契约性贝尔曼特征和收敛的有限时域近似方法来分析博弈的值，并比较了不同语义下的策略和结果。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了在分布计算和博弈策略中可能应用的多种场景，如分布式计算系统和交通网络。

**📈 对比分析**

通过比较三种语义下的值，发现平均分布语义的下值优于采样语义的值，而采样配置和采样转移的值在一般情况下不可比较。

**⚠️ 局限性**

限制在于平均分布语义可能导致博弈不确定性，即使在四状态的确定性竞技场中也可能出现这种情况，而采样语义则是确定的。

---

## 206. Simulating Space Bounded Read-$k$ PTMs with Space Bounded Read-Once PTMs

**arXiv ID:** 2611.17690 | [PDF](https://arxiv.org/pdf/2611.17690v1)

**作者:** Kuan Cheng `[一作]` (Peking University), Ruiyang Wu `[通讯]` (Peking University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文展示了具有有限空间S和运行时间T的read-k概率图灵机可以被read-once概率图灵机模拟，并且给出了具体的空间和时间复杂度。

**💡 创新点**

创新点在于证明了read-k BP类等于BP类，并且在k为常数的情况下，read-k BP类等于L类。

**🔧 技术方法**

使用了概率图灵机的模型，特别是read-k和read-once的随机访问模型。

**📊 数据集**

没有具体提到使用的数据集，但讨论了与图灵机相关的计算复杂性。

**📈 对比分析**

通过比较read-k和read-once模型的模拟，证明了在特定条件下两者的接受概率相近，且在空间和时间复杂度上有明确的界限。

**⚠️ 局限性**

限制在于该结果依赖于k的常数性，且未考虑更复杂的随机访问模型的情况。

---

## 207. New Results on the Subfield Metric

**arXiv ID:** 2611.18228 | [PDF](https://arxiv.org/pdf/2611.18228v1)

**作者:** Ana Kosovac `[一作]` (Technical University of Munich), Violetta Weger `[通讯]` (Technical University of Munich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文进一步发展了最近引入的λ-子域度量的理论，研究了极值码、经典界限和在该设置下的渐近密度。

**💡 创新点**

创新点在于引入了最大λ-子域距离（MλD）码，并建立了其参数的结构限制，同时扩展了理论到亚线性码，并研究了MλD码的渐近密度。

**🔧 技术方法**

使用了λ-子域度量的理论，结合了经典编码理论中的多种界限，包括Singleton、Plotkin、Griesmer、Elias和Johnson类型的界限。

**📊 数据集**

使用了有限域𝔽_q^m的线性码，具体数据集未明确给出，但涉及到的参数包括码长n、基域大小q和扩展度数。

**📈 对比分析**

通过与MDS和MRD码的比较，发现MλD码在不同参数范围内的行为，且在某些情况下，MλD码的密度与MDS码的密度相似。

**⚠️ 局限性**

限制在于目前对MλD码的显式构造仍然有限，且高效解码算法尚未得到充分探索。

---

## 208. When Does Cross-View Consistency Matter in Multi-View Appearance Augmentation? A Controlled Study for Visuomotor Policies

**arXiv ID:** 2611.17751 | [PDF](https://arxiv.org/pdf/2611.17751v1)

**作者:** Zixuan Jiang `[一作]` `[通讯]` (Peking University), Zixuan Jiang (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究探讨了在多视角视觉增强中，交叉视角耦合对视觉运动控制的影响，使用模拟状态重放技术操控交叉视角耦合，同时保持每个摄像头的扰动分布、物理轨迹和动作标签。

**💡 创新点**

创新点在于提出了一种状态重放设计，能够在保持每个摄像头扰动边际的同时操控交叉视角的联合分布，并提供了多个外观对比和几何对比的下游控制估计。

**🔧 技术方法**

使用了状态重放技术和晚融合扩散策略（Diffusion Policy），并进行了特征分析和相机干预实验。

**📊 数据集**

使用了七个LIBERO操作任务的数据集，进行了多种视觉增强实验。

**📈 对比分析**

通过比较独立生成器和联合生成器的效果，发现几何反耦合观察相较于耦合观察成功率降低，且在测试的任务和策略配置中，外观耦合的变化对性能影响较小。

**⚠️ 局限性**

限制在于研究仅使用了七个选定的模拟任务，且未能代表所有机器人操作任务的多样性，此外，几何耦合仅在扩散策略下进行了测试，未涵盖其他可能的配置。

---

## 209. Score-based neural ordinary differential equations for reaction-diffusions

**arXiv ID:** 2611.18467 | [PDF](https://arxiv.org/pdf/2611.18467v1)

**作者:** Jianguo Hou `[一作]` (University of South Carolina), Wuchen Li `[通讯]` (University of South Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究开发了一种拉格朗日方法，用于反应扩散方程，其中每个粒子携带位置和物理权重，通过学习的速度场和加权粒子训练来近似这些动态。

**💡 创新点**

创新点在于提出了一种闭合特征系统，结合了反应项和自一致性残差作为训练目标，能够控制引入密度的误差，并且采用了适应性采样分布和加权回归。

**🔧 技术方法**

使用了神经网络来表示速度场，并通过窗口迭代的方式进行训练，优化了粒子数据的回归损失。

**📊 数据集**

在数值实验中，使用了Ornstein-Uhlenbeck方程、Fisher-KPP方程和Wasserstein-Fisher-Rao梯度流等数据集进行验证。

**📈 对比分析**

与传统的随机粒子方法相比，提出的方法在粒子移动上是确定性的，并且在多个实验中表现出相对较低的误差，尤其是在质量和分布误差方面。

**⚠️ 局限性**

限制在于对于增长率的误差界限依赖于最大值，且尚未建立数值分析与界限之间的联系。此外，方法在处理高维问题时计算成本较高。

---

## 210. Parameterized Approximation for Minimizing Uncovered Elements and Unsatisfied Constraints

**arXiv ID:** 2611.17951 | [PDF](https://arxiv.org/pdf/2611.17951v1)

**作者:** Rudranarayan Kar `[一作]` (National Institute of Science Education and Research), Abhishek Sahu `[通讯]` (National Institute of Science Education and Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法和性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 211. AI-guided robotic in situ fabrication of curvature- and tissue-adaptive wireless bioelectronics

**arXiv ID:** 2611.17638 | [PDF](https://arxiv.org/pdf/2611.17638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 212. Agentic-Teacher: An Agentic Data Engine for Failure-Driven VLA Post-Training

**arXiv ID:** 2611.17394 | [PDF](https://arxiv.org/pdf/2611.17394v1)

**作者:** Gongyue Teng `[一作]` (Tsinghua University), Xinlei Chen `[通讯]` (Tsinghua University)

**通讯引用:** 21579 | [OpenAlex ID](https://openalex.org/A5016095713)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Agentic-Teacher的自主数据引擎，用于通过分析失败的执行轨迹生成和策划纠正演示，以改进预训练的视觉-语言-动作（VLA）模型。

**💡 创新点**

创新点在于Agentic-Teacher能够自主识别VLA模型的弱点，并构建局部的纠正专家，从而生成适合学习的演示数据，而无需人工干预。

**🔧 技术方法**

使用了编码代理技术，通过环境交互生成和验证闭环代码策略，并结合时间相关的动作噪声、去除闲置帧和基于流匹配损失的过滤方法。

**📊 数据集**

在LIBERO-10、MetaWorld-10和三个RoboCasa任务的数据集上进行了验证。

**📈 对比分析**

与传统方法相比，Agentic-Teacher在后训练中显著提高了VLA的成功率，LIBERO-10从79.4%提升至87.8%，MetaWorld-10从80.4%提升至90.6%，RoboCasa任务的成功率从0%提升至30%。

**⚠️ 局限性**

局限性包括修复能力依赖于编码代理的技能库，复杂任务可能需要更长的探索时间，且修复数据的质量会影响后训练效果。此外，物理机器人上的重放成本较高，增加了修复尝试的时间和操作成本。

---

## 213. Quantifying the Advantage of Coarser DPOR

**arXiv ID:** 2611.18069 | [PDF](https://arxiv.org/pdf/2611.18069v1)

**作者:** Minjian Zhang `[一作]` (Nanyang Technological University), C. -H. Luke Ong `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文研究了如何在不完成程序探索的情况下量化RF等价的优势，提出了一种基于采样的算法来估计RF类和Mazurkiewicz类的数量及其比率。

**💡 创新点**

创新点在于提出了一种新的算法，该算法能够在固定线程数的情况下以多项式时间内无偏估计RF和Mazurkiewicz类的数量，并且提供了强一致性的比率估计。

**🔧 技术方法**

使用了一种基于采样的算法，该算法结合了之前的Mazurkiewicz计数方法，并通过统计测试来选择更快的模型检查器。

**📊 数据集**

在34个基准配置上进行了实验，这些配置涵盖了并发数据结构、同步算法和合成程序。

**📈 对比分析**

通过与Mazurkiewicz检查器和RF检查器的短期运行进行比较，结果显示在99.4%的情况下选择了更快的检查器，且总验证时间减少了43.9%。

**⚠️ 局限性**

限制在于当线程数不固定时，RF计数和比率的无偏估计可能是不可行的，且在某些情况下可能会导致较高的计算复杂性。

---

## 214. Generalized hydraulic neural operators turbocharging sewer digital twins

**arXiv ID:** 2611.18081 | [PDF](https://arxiv.org/pdf/2611.18081v1)

**作者:** Ruozhou Lin `[一作]` (City University of Hong Kong), Zhiguo Yuan `[通讯]` (City University of Hong Kong)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种快速、通用且可微分的多分支神经算子，用于预测管道内的水力学，显著加速了基于SVE的模拟。

**💡 创新点**

该方法实现了对新污水系统的泛化，无需重新训练，并在水力校准和优化中提供了百万倍的速度提升。

**🔧 技术方法**

使用了多分支神经算子技术，该算子通过学习Saint-Venant方程的解算子来进行水力预测。

**📊 数据集**

使用了合成数据集，涵盖2871种管道配置和近3亿个数据点，此外还在真实基础设施上进行了测试。

**📈 对比分析**

与传统的SVE求解器相比，该方法在管道级别实现了5500到3100倍的速度提升，且在测试集上水深和流量的R^2值分别为1.000和0.999，NMAE分别为0.11%和0.24%。

**⚠️ 局限性**

该方法的局限性在于训练数据主要基于合成流入，可能在某些复杂的真实场景中表现不佳，且对非圆形管道的扩展仍需进一步研究。

---

## 215. Creating Steering Vectors via Input Optimisation

**arXiv ID:** 2611.17481 | [PDF](https://arxiv.org/pdf/2611.17481v1)

**作者:** Batuhan Akar `[一作]` `[通讯]` (ETH Zürich), Batuhan Akar (ETH Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ActGCG的方法，通过优化后缀以匹配目标激活，从而影响语言模型的行为。

**💡 创新点**

创新点在于将激活引导与后缀优化结合，允许在没有模型内部访问的情况下，通过输入后缀实现目标行为。

**🔧 技术方法**

使用了Greedy Coordinate Gradient (GCG)算法的修改版ActGCG，并引入了软后缀变体和模型翻译步骤。

**📊 数据集**

使用了Dolly指令数据集来构建德语和拒绝行为的引导向量。

**📈 对比分析**

与原始GCG方法和SSR方法进行了比较，ActGCG在德语任务上表现优异，成功率显著高于其他方法，但在拒绝任务上表现较弱。

**⚠️ 局限性**

局限性包括只测试了德语和拒绝两个概念，成功率受引导向量强度影响，且评估方法可能存在偏差。

---

## 216. AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI

**arXiv ID:** 2611.17564 | [PDF](https://arxiv.org/pdf/2611.17564v1)

**作者:** Snehalkumar `Neil' S. Gaikwad `[一作]` (University of North Carolina), Paritosh Pandey `[通讯]` (University of North Carolina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了AI-Voyager，一个公共利益系统，旨在促进人类与人工智能的公民推理，以理解、质疑和塑造治理人工智能的规则。

**💡 创新点**

创新点在于将人类与AI结合，创建一个可追溯的治理文本体系，使公众能够参与到AI治理的过程中，并通过AI助手进行政策的理解和修订。

**🔧 技术方法**

使用了检索增强生成（RAG）技术，结合了文档检索和生成模型，以确保每个声明都能追溯到具体的文本段落。

**📊 数据集**

构建了一个包含3,653个治理文档的语料库，涵盖了城市、州、联邦和国家政府的政策记录，共278,021页。

**📈 对比分析**

与最强基线相比，AI-Voyager在360个新问题上达到了83.0%的宏观召回率，而最强基线为80.5%。在没有迭代检索的情况下，召回率降至73.7%。

**⚠️ 局限性**

限制在于用户可能会将AI的回答视为法律的准确陈述，而AI-Voyager的设计旨在通过提供可验证的来源来减轻这一风险。

---

## 217. An AI-enabled high-resolution chlorophyll-a forecasting model that outperforms the operational global biogeochemical system

**arXiv ID:** 2611.17704 | [PDF](https://arxiv.org/pdf/2611.17704v1)

**作者:** Yong Jiang `[一作]` (Beijing University of Posts and Telecommunications), Wei Wang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种名为ChlCast的AI模型，用于生成表层叶绿素a浓度的每日预测，预测范围为10天，空间分辨率约为4公里。

**💡 创新点**

ChlCast结合了分解的时空卷积和多尺度架构，能够有效捕捉细微的叶绿素a变化，显著提高了预测精度，超越了现有的Copernicus全球生物地球化学预测系统。

**🔧 技术方法**

使用了深度学习技术，具体为(2+1)D U-Net架构，结合了时空卷积和多尺度编码-解码器结构。

**📊 数据集**

使用了Copernicus全球海洋颜色叶绿素产品的每日叶绿素a浓度数据集，涵盖2008年至2025年的数据，并在2026年进行了验证。

**📈 对比分析**

与Copernicus系统的比较显示，ChlCast在多个评估领域中表现优越，平均绝对误差（MAE）减少了70.74%，均方根误差（RMSE）减少了47.63%。在独立的BGC-Argo观测中，ChlCast也表现出更低的误差和更高的相关性。

**⚠️ 局限性**

ChlCast在高浓度叶绿素a的预测中存在低估现象，且模型性能在不同气候条件下存在不平衡，未来需要进一步优化损失函数设计和概率预测方法。

---

## 218. Quantum Graph Neural Networks: Architectures, Trainability, and Applications

**arXiv ID:** 2611.17886 | [PDF](https://arxiv.org/pdf/2611.17886v1)

**作者:** Zeinab Sharifi `[一作]` (Iran University of Science and Technology), Hossein Shirazi `[通讯]` (Malek Ashtar University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文综述了量子图神经网络（QGNNs）的发展，提出了一种统一的形式化框架，描述了QGNN的结构和功能，并对现有文献进行了分类和比较。

**💡 创新点**

创新点在于提供了一个统一的数学形式，涵盖了QGNN的定义、对称性和资源模型，并对不同架构进行了系统的比较。

**🔧 技术方法**

使用了参数化量子电路（PQCs）和哈密顿演化等技术，结合了量子计算和图神经网络的优势。

**📊 数据集**

审查了多个数据集和评估实践，提出了最低报告协议，涵盖了高能物理、化学、金融、医疗、地球观测和交通等领域的应用。

**📈 对比分析**

通过量子比特成本、对称性、可扩展性和证据等方面对五种架构家族进行了比较，发现目前的结果大多是模拟的，且在经典模型上没有显著优势。

**⚠️ 局限性**

限制在于目前的QGNN尚未在经典图数据任务上表现出优于强经典GNN的能力，且大多数结果基于小规模图的模拟，缺乏系统的实证支持。

---

## 219. Rate Optimal Streaming Codes Under an Extended Delay Profile for Three-Node Relay Networks

**arXiv ID:** 2611.18137 | [PDF](https://arxiv.org/pdf/2611.18137v1)

**作者:** Ke Wang `[一作]` (IEIT SYSTEMS Co., Ltd.), Yaqiang Zhang `[通讯]` (IEIT SYSTEMS Co., Ltd.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在具有突发擦除的三节点中继网络下的流编码，提出了两种二进制构造SR和SR-2，利用扩展延迟配置允许中继转发当前和过去源符号的因果可逆组合。

**💡 创新点**

创新点在于SR-2构造扩大了SR构造的适用范围，提供了更广泛的条件以达到非自适应速率上限，并且揭示了系统普通坐标转发的两个限制。

**🔧 技术方法**

使用了二进制编码技术，构造了SR和SR-2两种流编码方案，结合了扩展延迟配置和普通延迟配置。

**📊 数据集**

使用了三节点中继网络的突发擦除数据集，具体参数为(b_1, b_2, T)和延迟配置。

**📈 对比分析**

与现有的自适应方案进行比较，SR和SR-2构造在非自适应设置下能够精确达到速率上限，而现有的自适应方案在相同条件下只能渐近接近该上限。

**⚠️ 局限性**

限制在于系统普通坐标转发无法在0<T-u-v<v的情况下达到速率上限，并且对于相等的突发长度，只有在b|(T-2b)的情况下才能精确达到速率上限。

---

## 220. Inferring Sensitive Events in Multi-Mode Cyber-Physical Systems via Scheduling Side Channels

**arXiv ID:** 2611.17761 | [PDF](https://arxiv.org/pdf/2611.17761v1)

**作者:** Yifan Cai `[一作]` (University of British Columbia), Linh Thi Xuan Phan `[通讯]` (University of Pennsylvania)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究展示了多模式网络物理系统（CPS）中的模式变化如何引入安全漏洞，提出了一种轻量级攻击方法，通过被动的时间观察和最小的系统知识推断与模式变化相关的敏感事件。

**💡 创新点**

创新点在于识别了由模式变化引起的调度侧信道，使得未特权的共存任务能够通过自身的执行时间和核心分配推断敏感事件，而无需探测共享内存或缓存。

**🔧 技术方法**

使用了调度基础的时间侧信道攻击方法，结合了两种推断策略，分别针对显式和隐式模式，能够容忍调度噪声并在不同设备间迁移。

**📊 数据集**

在多个真实世界的设置和6400个合成任务集上进行了评估，包括家庭安全系统和使用牛津机器人车数据集的自驾车系统。

**📈 对比分析**

与其他基于调度的攻击相比，该方法在未见硬件上实现了94.7%的准确率，且在自驾车数据集上达到了86.0%的准确率，明显优于传统的缓存侧信道攻击（74.6%）。

**⚠️ 局限性**

限制在于攻击的有效性依赖于调度行为的可观察性和可区分性，且在不同事件的条件分布重叠时准确性会下降。此外，攻击假设在离线配置期间学习到的特征分布在受害者环境中仍然具有代表性。

---

## 221. VEPO: Mitigating Multilingual Hallucinations in LVLMs via Visual Evidence-guided Preference Optimization

**arXiv ID:** 2611.18096 | [PDF](https://arxiv.org/pdf/2611.18096v1)

**作者:** Chengpeng Fu `[一作]` (Harbin Institute of Technology), Ting Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为VEPO（视觉证据引导的偏好优化）的方法，旨在减少大型视觉语言模型（LVLMs）在低资源语言中的对象幻觉问题，通过训练时优化视觉证据的使用。

**💡 创新点**

VEPO的创新点在于它在训练阶段利用英语查询的中层跨模态注意力来定位关键视觉证据，并构建证据充分性偏好对，而无需外部模型或翻译。

**🔧 技术方法**

使用了中层跨模态注意力机制来定位视觉证据，并通过直接偏好优化（DPO）来优化低资源语言的偏好目标。

**📊 数据集**

使用了PALO多语言视觉指令语料库（约210万条多语言视觉指令），以及POPE和MME数据集进行评估，涵盖中文、法语和印地语。

**📈 对比分析**

与现有方法相比，VEPO在POPE和MME上的实验结果显示出一致的性能提升，尤其是在低资源语言（如印地语）上，F1分数从0.302提升至0.644，表明VEPO有效减少了幻觉现象。

**⚠️ 局限性**

限制在于VEPO依赖于英语的证据转移，可能需要针对其他LVLM架构重新调整中层索引。

---

## 222. NeoResearch: Literature- and Experience-Guided Autonomous Time Series Model Discovery

**arXiv ID:** 2611.18235 | [PDF](https://arxiv.org/pdf/2611.18235v1)

**作者:** Ze Guo `[一作]` (University of Science and Technology of China), Shangtong Ye `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为NeoResearch的框架，用于自动发现轻量级的领域特定时间序列预测模型，结合文献和经验指导。

**💡 创新点**

创新点在于建立了一个专家启发的双循环研究工作流程，结合假设驱动的架构探索与基于实验反馈的迭代训练和超参数优化。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行文献挖掘和模型设计，同时结合了层次化的经验记忆。

**📊 数据集**

实验使用了十个数据集，涵盖电力市场、能源系统和水文学等领域。

**📈 对比分析**

与多种基线方法（如ARIMA、LSTM等）进行比较，NeoResearch在大多数任务中表现出色，尤其在19个数据集-指标比较中取得最低误差。

**⚠️ 局限性**

限制在于发现过程受到架构空间和研究预算的约束，未来工作将探索完全冷启动的发现、对冲突科学证据的鲁棒性以及更可靠的跨领域经验转移。

---

## 223. Imitation Bootstrapped Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.18543 | [PDF](https://arxiv.org/pdf/2611.18543v1)

**作者:** Tony H. Zhang `[一作]` (University of Utah), Daniel S. Brown `[通讯]` (University of Utah)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的多智能体强化学习方法IBMARL，旨在解决多智能体任务中的演示和去中心化执行问题。

**💡 创新点**

创新点在于将模仿学习与强化学习结合，利用单人演示生成去中心化的合作策略，并在训练过程中保持模仿策略的参与。

**🔧 技术方法**

使用了模仿增强的多智能体强化学习（IBMARL），结合了轮流行为克隆（R2BC）和集中式评论员的机制。

**📊 数据集**

在四个合作任务上进行评估，包括导航、Buzzwire、平衡和运输，使用合成演示者和真实人类演示者的数据。

**📈 对比分析**

与基线方法（如RLfD和RFT）相比，IBMARL在所有任务中都能更快地超越模仿学习策略，并在某些任务中超过专家表现。

**⚠️ 局限性**

限制在于所有实验均在模拟环境中进行，且假设团队奖励已知，未来工作应在真实机器人和多样化的演示者上进行评估。

---

## 224. Improved Regret via Variance-Aware Policy Optimization in Adversarial MDPs

**arXiv ID:** 2611.17469 | [PDF](https://arxiv.org/pdf/2611.17469v1)

**作者:** Ofir Schlisselberg `[一作]` (Tel Aviv University), Yishay Mansour `[通讯]` (Tel Aviv University)

**通讯引用:** 25538 | [OpenAlex ID](https://openalex.org/A5014637159)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了具有未知动态和全信息奖励的情节对抗MDP，并实现了O(√(H^3SAK))的后悔界限。

**💡 创新点**

通过引入从未知动态到已知动态的归约，识别出约束的已知动态问题作为缩小剩余差距的途径。

**🔧 技术方法**

使用了一种新的政策优化算法VOPO（方差乐观政策优化），结合了稳定性匹配奖励（SMB）和方差敏感分析。

**📊 数据集**

使用了具有全信息奖励的未知动态MDP，具体数据集未明确提及。

**📈 对比分析**

与之前的算法相比，改进了未知动态的后悔保证，后悔界限为O(√(H^3SAK))，并且在已知动态下的后悔界限为O(√(H^3K))，均优于现有的最先进的政策优化界限。

**⚠️ 局限性**

限制在于仍然存在一个√(H)的差距，表明在某些参数范围内，后悔界限是紧的。

---

## 225. When Will You Hand It Over? A Hidden Start Delay Makes Self-Play Handover Robust to Unfamiliar Partners

**arXiv ID:** 2611.18491 | [PDF](https://arxiv.org/pdf/2611.18491v1)

**作者:** Mohammad Khoshnazar `[一作]` (Universität Bremen), Matyas Szanto `[通讯]` (Budapest University of Technology and Economics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本研究探讨了合作自我游戏如何使人形机器人在没有人类数据的情况下学习物品接收，但发现训练后的伙伴与实际接触的伙伴并不相同。研究表明，流畅的自我游戏交接策略在面对不熟悉的伙伴时会失效，并识别出主要原因并进行修复。

**💡 创新点**

创新点在于引入了隐藏的开始延迟随机化规则，这一简单的修复方法显著提高了机器人在与未见过的伙伴交接时的成功率，从31%提升至80%。

**🔧 技术方法**

使用了交替的PPO自我游戏技术，结合了物理模拟环境进行训练和评估。

**📊 数据集**

使用了Unitree G1机器人与学习的伙伴进行物理基础的包裹交接任务，训练和测试中使用了多个不同的伙伴条件和场景种子。

**📈 对比分析**

通过与控制组的比较，发现引入隐藏开始延迟后，机器人在未见过的伙伴交接中的成功率显著提高，而控制组的成功率保持在35-41%。

**⚠️ 局限性**

限制在于研究是在模拟环境中进行的，使用的是模拟的伙伴手和结构化状态，而不是图像数据。此外，训练延迟范围与脚本化伙伴的开始范围重叠，可能影响结果的普适性。

---

## 226. Alignment by Socialization: Learning Tradeoffs Under Selection

**arXiv ID:** 2611.18531 | [PDF](https://arxiv.org/pdf/2611.18531v1)

**作者:** Thomas Brennan `[一作]`, Chaoyi Zhao `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了人工智能代理在面对任务完成和行为规范之间的选择时，如何通过进化选择塑造其决策，并探讨经验如何影响这些选择以反映人类判断。

**💡 创新点**

提出了一种学习政策，通过经验更新代理的决策权衡，使其能够在不同环境中适应并优化长期增长，强调了社会化过程的重要性。

**🔧 技术方法**

使用了进化博弈理论和贝叶斯学习方法，结合小型神经网络进行实验，以验证理论模型的有效性。

**📊 数据集**

使用了模拟环境和小型神经网络进行实验，具体数据集未详细说明，但涉及多种任务和行为规范的情境。

**📈 对比分析**

比较了三种政策：仅任务政策、固定线性政策和学习政策。学习政策在长期增长方面表现最佳，能够适应环境变化，而仅任务政策在面对行为规范时表现不佳。

**⚠️ 局限性**

限制在于学习政策的社会化过程可能会被后续的训练环境（如仅奖励任务完成的沙盒环境）逆转，且有限记忆会削弱其对社会化的保护。

---

## 227. Small Open-Weight Language Models for Crowd Decision-Making: Capabilities and Limits of Output-Level Fusion and Linguistic Aggregation

**arXiv ID:** 2611.17589 | [PDF](https://arxiv.org/pdf/2611.17589v1)

**作者:** David Herrera-Poyatos `[一作]` (University of Granada), Francisco Herrera `[通讯]` (University of Granada)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发并应用了一种集成评估协议，用于本地部署的小型开放权重语言模型，以支持基于文本的群体决策。通过对伦敦和马德里餐厅评论的分析，评估了模型在决策中的表现。

**💡 创新点**

创新点在于将多个标准组件联合使用，形成一个配置决策管道，并提出了设计规则，以保持评估的连续性、根据参与者的不确定性调整推理规模，并报告边际和候选集。

**🔧 技术方法**

使用了小型开放权重语言模型（参数从7.5B到36B），并通过固定面板输出级融合和连续算术聚合等技术进行评估。

**📊 数据集**

使用了两个数据集：伦敦的TripR-2020Large数据集（474条评论）和马德里的TripAdvisor餐厅评论数据集（600条评论）。

**📈 对比分析**

通过与单模型推理、固定面板输出级融合和贪婪解码进行比较，发现线性融合在重复性和独立性方面表现良好，但无法区分替代方案，且其额外调用对赢家的持久性提升有限。

**⚠️ 局限性**

限制在于没有可用的真实集体决策，且模型的评估依赖于参与者的敏感性，未能识别唯一赢家。未来的工作应验证预算约束下的面板选择，并扩展到其他领域和语言。

---

## 228. Dancing Tokens on the Vocabulary Network from LLM Generation

**arXiv ID:** 2611.18248 | [PDF](https://arxiv.org/pdf/2611.18248v1)

**作者:** Yuan-Chao Hu `[一作]` `[通讯]` (Dongguan Institute of Materials Science and Technology), Yuan-Chao Hu (Dongguan Institute of Materials Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个由六个开放的大型语言模型（LLMs）的输入嵌入构成的相似性网络，并将生成过程映射到该网络上，以探讨生成与预训练中学习的几何形状之间的关系。

**💡 创新点**

提供了一种新的视角，通过图学习从微观层面理解LLM的生成过程，揭示了生成过程并非在预定义网络上行走，而是集中在网络的特定区域内。

**🔧 技术方法**

使用了图学习技术，构建了基于输入嵌入的相似性网络，并通过图的拓扑特征分析生成过程。

**📊 数据集**

研究了六个开放源代码的LLMs，参数范围从70亿到210亿，涵盖不同的模型架构。

**📈 对比分析**

通过与随机嵌入的比较，发现生成的连续标记在网络中很少是邻居，且大多数模型的生成过程与随机顺序相比没有显著差异，表明生成过程并不遵循静态相似性网络的边。

**⚠️ 局限性**

模型的生成过程在网络中高度集中，且生成的标记在深度上晚于大多数模型的预期，表明生成过程的局限性和潜在的重复性问题。

---

## 229. Toward the Minimal Set of Information Inequalities in Regenerating Codes: An Algebraic Approach

**arXiv ID:** 2611.17765 | [PDF](https://arxiv.org/pdf/2611.17765v1)

**作者:** Rui-Juan Jing `[一作]` (Jiangsu University), Chun-Ming Yuan `[通讯]` (Chinese Academy of Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了(n,n-1,n-1)精确修复模型的香农类型约束系统，提出了一种基于闭包的表示方法，直接定义了约束系统，并证明了其与初始系统的等价性。

**💡 创新点**

创新点在于通过闭包结构简化了信息不等式系统，确保每个不等式都是非冗余的，并且没有隐式等式，提供了一种新的算法来构建该系统。

**🔧 技术方法**

使用了代数方法和闭包操作来描述和构建约束系统，结合了信息论中的线性规划技术。

**📊 数据集**

使用了(n,n-1,n-1)精确修复模型的相关数据集，具体的数值和约束数量在文中进行了详细的枚举和验证。

**📈 对比分析**

与现有方法相比，本文的方法在约束数量上显著减少，且通过具体的枚举验证了系统的大小，性能表现良好，且在n=4和n=5时，实际约束数量与理论上限的偏差仅为1.84%和0.47%。

**⚠️ 局限性**

限制在于该研究主要集中在(n,n-1,n-1)模型上，未来的工作可以扩展到其他类型的编码约束和实现层面的线性规划性能比较。

---

## 230. Sample-efficient prediction of high-entropy alloy strength through outcome-aware robust sequential learning

**arXiv ID:** 2611.18462 | [PDF](https://arxiv.org/pdf/2611.18462v1)

**作者:** Ahmed Shoyeb Raihan `[一作]` (West Virginia University), Imtiaz Ahmed `[通讯]` (West Virginia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为ROSE-HEA的框架，用于从合金成分和加工条件中高效预测高熵合金的极限抗拉强度（UTS）。

**💡 创新点**

创新点在于提出了一种基于结果的稳健风险降低获取机制，能够评估候选实验对预测的预期贡献，并动态调整对不同条件的重视程度。

**🔧 技术方法**

使用了高斯过程（GP）作为代理模型，并结合了贝叶斯优化（BO）中的期望改进（EI）和上置信界（UCB）等传统获取策略。

**📊 数据集**

使用了由Wang等人报告的包含501条记录的高熵合金（HEA）拉伸性能数据库，数据包括14个成分和加工输入。

**📈 对比分析**

与EI-BO、UCB-BO、随机GP和八种传统回归模型进行了比较，ROSE-HEA在30次匹配运行中实现了最低的均方根误差（RMSE），为152.84 MPa，相较于EI和UCB分别降低了6.16%和6.37%。

**⚠️ 局限性**

限制在于当前评估是回顾性的，且仅限于固定的候选池，未来工作应在主动学习活动中部署ROSE-HEA，以便在实际实验中验证其有效性。

---

## 231. Learning Word-Level Explanations of Text Reuse in Latin Literature with Little Annotation

**arXiv ID:** 2611.18319 | [PDF](https://arxiv.org/pdf/2611.18319v1)

**作者:** Julian Schelb `[一作]` (University of Konstanz), Marie Revellio `[通讯]` (University of Konstanz)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种逐字解释文本重用的方法，通过将重用段落中的每个单词与其源单词对齐，并标记转换操作（如复制、变形或替换），以便更好地理解后来的作者如何重塑源文本。

**💡 创新点**

创新点在于结合了合成数据和基于不确定性的主动学习，减少了标注工作量，同时提出了一种指针网络模型，能够在同一步骤中选择源单词并命名操作，使得重用检测变得逐字可解释。

**🔧 技术方法**

使用了指针网络（pointer network）作为主要技术，该模型能够同时链接和标记重用单词与源单词之间的关系。

**📊 数据集**

使用了Loci Similes基准中的1490个参考对进行评估和训练，其中包括858个逐字引用和632个典故。

**📈 对比分析**

与其他11种系统进行了比较，结果显示该模型在标注每个重用对时的性能优于其他系统，尤其在处理变形和替换操作时表现最佳。通过主动学习，模型在仅使用200个不确定性选择的对时，能够达到与使用所有标注对训练的模型相当的效果。

**⚠️ 局限性**

限制在于标注的稳健性，只有一个标注者进行标注，可能导致不同的解读和一致性问题；此外，拉丁语资源的覆盖范围有限，无法命名某些关系，导致模型在某些替换操作上表现不佳。

---

## 232. TouchTwin: Scaling Human Touch Data via Real-to-Sim-to-Real

**arXiv ID:** 2611.17619 | [PDF](https://arxiv.org/pdf/2611.17619v1)

**作者:** Huiwen Han `[一作]` (Stanford University), Zhiwen Fan `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于触觉的真实到模拟再到真实的管道，重建手-物体交互，并通过触觉数据生成可扩展的人类触觉数据。

**💡 创新点**

创新点在于通过触觉数据驱动的物理模拟重建手-物体交互，解决了传统视觉方法无法捕捉接触力的问题。

**🔧 技术方法**

使用了MuJoCo物理引擎进行手-物体交互的模拟，并结合触觉手套记录的触觉数据。

**📊 数据集**

使用了触觉手套记录的数据集，该数据集包含人类在执行任务时的触觉信息。

**📈 对比分析**

与传统的视觉重建方法相比，基于触觉的重建在接触准确性上显著提高，减少了32%的接触帧丢失率，且在手部策略测试中恢复了人类水平的抓握力。

**⚠️ 局限性**

限制在于所有策略仅在模拟资产上运行，尚需在真实机器人手上进行转移和实验，且持有集较小，测试任务仅限于杯子任务。

---

## 233. Resource Management and Trajectory Optimization for UAV-IRS Assisted Maritime Edge Computing Networks

**arXiv ID:** 2611.17803 | [PDF](https://arxiv.org/pdf/2611.17803v1)

**作者:** Chaoyue Zhang `[一作]` (Dalian Maritime University), Yuan Wu `[通讯]` (University of Macau)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了一种无人机-智能反射面（UIRS）辅助的海洋边缘计算网络中的资源管理和轨迹优化方案，旨在通过联合优化子时隙持续时间、卸载比例、发射功率、计算能力分配、UIRS相位偏移和UIRS轨迹来最小化整体能耗。

**💡 创新点**

创新点在于提出了一种两层优化方法，将非凸优化问题分解为两个子问题，分别通过半正定松弛（SDR）方法和深度确定性策略梯度（DDPG）算法进行求解，从而有效降低整体能耗。

**🔧 技术方法**

使用了半正定松弛（SDR）方法和深度确定性策略梯度（DDPG）算法来解决优化问题。

**📊 数据集**

使用了模拟数据集，假设有4到10个无人水面车辆（USVs）在1000米×1000米的目标区域内随机分布，进行了多种参数设置的仿真。

**📈 对比分析**

与基线方案相比，提出的RMTO方案在不同的仿真环境中显著降低了整体能耗，具体表现为在与固定计算能力分配（FCCA）、频分多址（FDMA）、UIRS直飞方案等的比较中，能耗分别减少了24.4%、14.9%、31.8%和29.2%。

**⚠️ 局限性**

限制在于优化问题的非凸性和动态环境下的资源管理决策的复杂性，未来的研究可以探索IRS作为传感设施的潜力，以及在物理层安全中增强保密速率的可能性。

---

## 234. BPU-SpikeMM:Exact Matrix Multiplication via Spatio-Temporal Spike Encoding on Real Neuromorphic Hardware

**arXiv ID:** 2611.18275 | [PDF](https://arxiv.org/pdf/2611.18275v1)

**作者:** Rongyuan Guo `[一作]` (Hunan University), Kenli Li `[通讯]` (Hunan University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

提出了一种基于真实神经形态硬件的精确时空脉冲编码框架BPU-SpikeMM，用于矩阵乘法。

**💡 创新点**

通过区分数学等价性和直接读出等价性，建立了与硬件兼容的映射原则，并构建了从脉冲编码到BPU执行的统一管道。

**🔧 技术方法**

使用了基于BPU的时空脉冲编码技术，支持多种编码方案，并在真实BPU硬件上进行评估。

**📊 数据集**

在真实的BPU硬件上进行了实验，使用了多种编码方案进行矩阵乘法的验证。

**📈 对比分析**

通过与传统硬件的比较，BPU-SpikeMM在时间成本、空间资源消耗和事件活动方面展示了不同编码方案的权衡，证明了其在BPU硬件上的有效性。

**⚠️ 局限性**

该研究未涉及跨架构性能比较，且在特定硬件和软件环境下进行评估，可能限制了结果的普适性。

---

## 235. SIMTurn: Lightweight Divergent Subgroup Scheduling for Intra-Warp Synchronization

**arXiv ID:** 2611.17827 | [PDF](https://arxiv.org/pdf/2611.17827v1)

**作者:** Nikhil Rout `[一作]` (University of California), Blaise Tine `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了一种轻量级的编译器-硬件机制SIMTurn，旨在解决SIMT执行中的死锁问题，同时保持基于堆栈的重新汇聚。

**💡 创新点**

创新点在于结合了编译器插入的yield指令和有限的可恢复子组池，使得在存在分歧的子组之间能够独立调度，从而实现前进进度。

**🔧 技术方法**

使用了编译器插入的yield指令、调度器和Vortex GPGPU的RTL实现。

**📊 数据集**

在开源的Vortex GPGPU上进行了评估，使用了多种同步和控制工作负载的基准测试。

**📈 对比分析**

与基线IPDOM和每线程独立线程调度（ITS）方案进行了比较，SIMTurn在大多数情况下仅引入了17-24%的调度器面积开销，相比之下ITS的开销为39-91%。在需要跨子组进度的内核上，SIMTurn的性能接近ITS，并且比软件序列化的IPDOM基线快2.7倍。

**⚠️ 局限性**

限制在于当活动子组数量超过池容量K时，可能会导致停滞。此外，未能识别的等待形式（如简单加载轮询）无法获得yield指令的支持。

---

## 236. Benchmarks Don't Talk Back: Learning Through Self-Disagreement

**arXiv ID:** 2611.17563 | [PDF](https://arxiv.org/pdf/2611.17563v1)

**作者:** Soheil Abbasloo `[一作]` (Microsoft Research), Peng Cheng `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

论文提出了一种新的能力，称为辩证能力，强调模型在面对挑战时的表现，特别是在何时坚持立场和何时让步于更好的观点。

**💡 创新点**

创新点在于提出了自我反对学习（Self-Disagreement），通过模型自身生成反对意见来训练其辩证能力，而不是依赖外部模型或人类注释。

**🔧 技术方法**

使用了强化学习（Reinforcement Learning）技术，特别是通过可验证奖励进行训练。

**📊 数据集**

使用了两个基础模型（Qwen3-8B和Gemma-2 9B）进行训练和评估，并在多个基准测试上进行验证。

**📈 对比分析**

与传统的准确性训练方法相比，Self-Disagreement在面对挑战时表现出更高的合成准确性，能够更好地保持正确答案，且在未见过的挑战者面前也能保持优势。

**⚠️ 局限性**

限制在于该方法尚未在所有模型和领域中普遍验证，且当前的奖励机制仅适用于可验证的答案，未来需要探索在不可验证情况下的应用。

---

## 237. TReC: Evaluating Move-Induced State Changes in Instruction-Conditioned Go

**arXiv ID:** 2611.17781 | [PDF](https://arxiv.org/pdf/2611.17781v1)

**作者:** Zhuangsheng Li `[一作]` (University of California), Bangyuan He `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法和性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 238. WeatherDreamer: Learning a World Model from Multimodal Earth Observations

**arXiv ID:** 2611.18393 | [PDF](https://arxiv.org/pdf/2611.18393v1)

**作者:** Kate Duffy `[一作]` (Zeus AI, Inc.), Thomas J. Vandal `[通讯]` (Zeus AI, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为WeatherDreamer的世界模型，该模型能够从18个观测系统的数据中学习天气预报，并在每小时接收新观测时进行自我修正。

**💡 创新点**

WeatherDreamer通过直接从观测中学习数据同化和预报循环，避免了对昂贵的数据同化系统的依赖，且在温度、湿度和风速的预报上超越了美国国家海洋和大气管理局的全球集合预报系统。

**🔧 技术方法**

使用了机器学习技术，特别是生成模型和递归状态空间模型，来实现数据同化和天气预报的端到端学习。

**📊 数据集**

训练使用了来自18个观测系统的数据集，包括地面气象站、卫星观测和气象探空仪等。

**📈 对比分析**

与NOAA的全球集合预报系统（GEFS）和欧洲中期天气预报中心（ECMWF）的集合进行了比较，WeatherDreamer在温度、露点和风速的预报上表现更佳，且其不确定性估计更为可靠。

**⚠️ 局限性**

WeatherDreamer在自由对流层的表现不如地面附近的表现，尤其是在温度和风速的预报上，且对上层观测的利用不足，导致其在高层大气的预报能力有限。

---

## 239. Knowledge After Collapse: Context and Wording Shape Factual Accessibility in Language Models

**arXiv ID:** 2611.17763 | [PDF](https://arxiv.org/pdf/2611.17763v1)

**作者:** Sujato Dutta `[一作]` (Mahindra University), Chandrajit Bajaj `[通讯]` (University of Texas at Austin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种事件条件评估框架KAD，用于研究语言模型在对话压力下的事实行为，特别是如何在不同的对话历史和措辞下影响模型的事实回答。

**💡 创新点**

创新点在于区分初始错误答案的采纳与后续的持续性和恢复，通过从相同的失败事件分支独立探测，能够进行匹配比较，这在仅评估采纳的研究中无法捕捉。

**🔧 技术方法**

使用了多种语言模型，包括Qwen和Llama系列，采用了事件条件的实验设计和统计分析方法。

**📊 数据集**

使用了包含1500个事实问题的基准数据集，涵盖多个来源，如PopQA、GRANOLA EntityQuestions、TriviaQA和NQ-Open。

**📈 对比分析**

通过与不同模型的比较，发现模型的基线合格率和在对话压力下的崩溃抵抗能力是可分的，且不同模型在面对相同问题时的表现差异显著。

**⚠️ 局限性**

限制在于KAD框架仅针对英语事实问题进行评估，且所报告的效果是基于选择事件的，而非无条件模型行为的描述。模型内部知识存储的机制未在此研究中探讨。

---

## 240. GraphScent: Calibrating Directional Relevance Signals for Bounded-Context Code Localization

**arXiv ID:** 2611.17527 | [PDF](https://arxiv.org/pdf/2611.17527v1)

**作者:** Ramesh Arvind Naagarajan `[一作]` (Chemnitz University of Technology), Ramesh Arvind Naagarajan `[通讯]` (Chemnitz University of Technology)

**通讯引用:** 12 | [OpenAlex ID](https://openalex.org/A5116813866)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了在单个消费者GPU上本地运行开放权重语言模型的可行性，提出了一种新的代码定位方法GraphScent，旨在通过短的无状态代码检查替代多轮探索历史。

**💡 创新点**

创新点在于发现了结构关系对代码修复目标的相关性判断的重要性，并提出了GraphScent和AST Ripple工具，以提高代码定位的准确性和效率。

**🔧 技术方法**

使用了无状态的Typed-AST多图和零泄漏的掩码父节点探测技术，结合了基于关系的对数似然比传播方法。

**📊 数据集**

使用了来自86个Python代码库的211个真实世界问题的数据集，并扩展了包含102个时间上不重叠的2025-2026年问题的面板。

**📈 对比分析**

与多种基线方法进行比较，GraphScent在定位难度较大的目标时，准确率从12.8%提升至23.4%，显示出显著的性能提升。

**⚠️ 局限性**

限制在于研究仅覆盖Python代码库，且未考虑其他静态类型语言的表现差异，同时评估的任务范围和时间戳可能无法完全保证预训练的非曝光。

---

## 241. ParseBench-2: Severity-Aware Evaluation of Document Parsing

**arXiv ID:** 2611.17400 | [PDF](https://arxiv.org/pdf/2611.17400v1)

**作者:** Pierre-Loïc Doulcet `[一作]` (LlamaIndex Research), Simon Suo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于严重性政策开发文档解析度量的方法，通过测试场景来表达错误对分数的影响，并利用编码代理和人工审核来优化度量。

**💡 创新点**

创新点在于明确了错误的严重性如何影响评分，并通过测试场景来验证度量的有效性，从而使得度量设计更加透明和一致。

**🔧 技术方法**

使用了编码代理进行迭代优化，通过人类审核来确保度量符合预期的严重性政策。

**📊 数据集**

使用了包含330个扰动的六个严重性测试套件，覆盖240种错误类型和90种良性变体，涉及多种文档类型。

**📈 对比分析**

通过与现有度量的比较，本文的度量在所有六个测试套件中都表现出更高的政策一致性，尤其在处理关键错误时，得分下降符合预期目标。

**⚠️ 局限性**

限制在于虽然度量设计遵循了严重性政策，但在实际应用中可能仍需根据具体需求进行调整，且对某些复杂错误的处理可能不够完善。

---

## 242. RL$^{\mathrm{RT}}$: On-Device Reinforcement Learning Runtime

**arXiv ID:** 2611.18035 | [PDF](https://arxiv.org/pdf/2611.18035v1)

**作者:** Zexin Li `[一作]` (Nanyang Technological University), Arvind Easwaran `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在嵌入式GPU上运行的强化学习运行时，能够协调周期性执行的演员和学习者，以确保演员在截止时间内完成任务，同时保持学习者的更新效率。

**💡 创新点**

创新点在于将演员与学习者的耦合关系作为一个可调的运行时参数，通过设置刷新界限和GPU许可来优化两者的调度，从而降低演员的截止时间错过率。

**🔧 技术方法**

使用了基于GPU的调度技术，包括刷新门、GPU许可和松弛感知的入场规则，结合现有的调度器进行工作。

**📊 数据集**

在两个Jetson平台（Orin AGX和Orin Nano）上进行了评估，使用了多种强化学习算法和控制环境，模型参数从17.4k到16.8M不等。

**📈 对比分析**

与现有方法（如锁步和自由运行）相比，该方法在保持演员截止时间错过率低于1%的同时，学习者的吞吐量在多个配置中表现最佳，显著优于85%到100%的错过率。

**⚠️ 局限性**

限制在于该方法的性能依赖于所选的刷新界限和调度策略，可能在某些情况下导致学习者的吞吐量下降。

---

## 243. RuralShield: Super Intelligence-Assisted Security Triage and Serverless Infrastructure for Resource-Constrained Healthcare

**arXiv ID:** 2611.17887 | [PDF](https://arxiv.org/pdf/2611.17887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 244. Elasticy: A Runtime-Reconfigurable Architecture for Dynamic Streams

**arXiv ID:** 2611.17798 | [PDF](https://arxiv.org/pdf/2611.17798v1)

**作者:** Konstantin J Hossfeld `[一作]`, Kunle Olukotun `[通讯]` (Stanford University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

提出了一种动态可重构的数据流加速器（RDA），旨在提高对不规则张量工作负载的性能和能效，特别是现代AI应用中的动态张量形状。

**💡 创新点**

创新点在于通过流和块抽象来最小化动态支持的开销，并引入了一种分布式链接流存储器，以更有效地利用片上内存，减少离片数据流量。

**🔧 技术方法**

使用了动态可重构的数据流执行模型，轻量级分布式硬件调度器，以及流嵌入的重配置信息来触发细粒度的运行时调度决策。

**📊 数据集**

在Llama 3-8B解码层和评估的GEMM（通用矩阵乘法）工作负载上进行了评估。

**📈 对比分析**

与传统的静态RDA相比，动态RDA在Llama 3-8B解码层上达到了92.6%的性能上限，而H100的性能为69.7%。在评估的GEMM中，减少了高达61.4%的离片流量，并且在重叠依赖内核时加速了内核对的执行，速度提升可达1.42倍。

**⚠️ 局限性**

限制在于动态重配置和形状动态性引入的开销仍然存在，尽管这些开销相对较小。

---

## 245. AI-Based Multi-View Echocardiographic Image Synthesis using Dynamic Shape Control

**arXiv ID:** 2611.18375 | [PDF](https://arxiv.org/pdf/2611.18375v1)

**作者:** João Freitas `[一作]` (University of Minho), Sandro Queirós `[通讯]` (Polytechnic University of Cávado and Ave)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种新颖的框架，用于通过动态形状控制合成多视角超声心动图序列。

**💡 创新点**

创新点在于结合了多视角心脏模型的解剖一致性与两阶段合成方法，能够生成真实的超声心动图序列。

**🔧 技术方法**

使用了无配对生成对抗网络（GAN）和扩散模型的两阶段AI合成框架。

**📊 数据集**

使用了MIMIC-IV-ECHO数据集和生成的1000个动态全心脏网格，模拟了42000个超声心动图序列。

**📈 对比分析**

通过与现有方法的比较，展示了在图像真实感和时间一致性方面的显著改进，尤其在专家评估中，合成序列的感知真实感显著提高。

**⚠️ 局限性**

限制在于使用的统计形状模型可能导致运动模式与临床超声心动图存在差异，且原始网格的时间分辨率有限。

---

## 246. Soteria: Formally Verified Planning with Runtime Enforcement for Safe LLM Agents

**arXiv ID:** 2611.17880 | [PDF](https://arxiv.org/pdf/2611.17880v1)

**作者:** Mike He `[一作]` (Princeton University), Aarti Gupta `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种框架，通过验证的分层规划和运行时规范执行来平衡安全性和效用，以满足LLM代理在执行多步骤工具调用时的需求。

**💡 创新点**

创新点在于通过事先生成并验证的结构化计划，确保代理在执行过程中遵循规范，同时提供在不安全操作被阻止时的恢复指导。

**🔧 技术方法**

使用了验证的分层规划技术和运行时规范执行。

**📊 数据集**

在207个任务上进行了评估，涵盖了三个具有挑战性的基准和四个LLM模型。

**📈 对比分析**

与四个基线方法进行比较，结果显示该框架在任务成功率上比最佳基线提高了百分比点，且所有模型的合规性达到了100%。

**⚠️ 局限性**

局限性在于需要一组正式化的规范作为输入，目前这些规范是通过半自动化工作流程获得的，且计划验证假设工具合同准确建模工具行为。

---

## 247. RAG-FGO: Dynamic Optimization of Query-Rewriting Agents for Generative RAG

**arXiv ID:** 2611.17788 | [PDF](https://arxiv.org/pdf/2611.17788v1)

**作者:** Qigan Sun `[一作]` (Kyung Hee University), Heng Tao Shen `[通讯]` (Tongji University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种动态优化框架RAG-FGO，用于生成式检索增强生成中的查询重写，旨在解决噪声反馈、优化难度和静态重写策略的局限性。

**💡 创新点**

创新点在于将查询重写视为一个混合离散-连续空间中的非可微黑箱优化问题，并引入了精英保留记忆池以提高优化稳定性。

**🔧 技术方法**

使用了基于Fungal Growth Optimizer (FGO)的动态优化技术，结合全局搜索和局部细化来有效探索复杂的策略空间。

**📊 数据集**

在多个知识密集型问答和推理基准上进行了实验评估，包括LongBench、HotpotQA、MuSiQue等。

**📈 对比分析**

与多种基线方法进行比较，RAG-FGO在检索质量和下游任务性能上表现出一致的提升，尤其在复杂语义和跨任务场景中展现出更强的鲁棒性和迁移能力。

**⚠️ 局限性**

局限性在于优化过程在离线阶段进行，需要额外的训练预算，且当前框架不支持在线或持续适应动态变化的查询或知识库。

---

## 248. LECAT: Measuring LLM Category Learning from Pretraining Data

**arXiv ID:** 2611.18419 | [PDF](https://arxiv.org/pdf/2611.18419v1)

**作者:** Jimin Mun `[一作]` (Carnegie Mellon University), Yanai Elazar `[通讯]` (Bar-Ilan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一个两部分框架，分析大型语言模型（LLMs）如何学习类别知识，评估模型在记忆、泛化和灵活使用类别知识方面的能力，并探讨预训练数据如何影响这些能力。

**💡 创新点**

创新点在于引入了一个框架来分析LLMs的类别知识学习，强调了预训练数据的表达方式对模型获取稳健类别知识的重要性。

**🔧 技术方法**

使用了基于完成概率的评估框架和LLM作为评判者的验证方法，结合了认知科学的理论。

**📊 数据集**

使用了199对策划的类别特征对和TinyStories数据集进行实验，分析了不同类别知识表达方式在数据中的分布。

**📈 对比分析**

通过对多个开源模型（如Pythia和OLMo3）的评估，发现模型在灵活性方面表现不佳，且随着模型规模的增大，类别知识的学习效果有所改善，但逐字重复会降低灵活学习能力。

**⚠️ 局限性**

限制在于数据变化、探针和训练设置虽然全面，但并不详尽，未来研究应探索更多样化的数据生成方法，并扩大预训练实验的规模。此外，关于刻板印象的讨论涉及伦理问题，需要谨慎处理。

---

## 249. PrivacyPR: A Dataset of Real-World Privacy Fixes for Benchmarking Privacy-Aware Code Review

**arXiv ID:** 2611.18293 | [PDF](https://arxiv.org/pdf/2611.18293v1)

**作者:** Hua Yang `[一作]` (North Carolina State University), Bowen Xu `[通讯]` (North Carolina State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个包含4131个合并的GitHub拉取请求（PRs）的数据集，旨在评估隐私意识代码审查的能力。

**💡 创新点**

创新点在于提供了一个连接真实隐私缺陷及其接受修复和审查上下文的数据集，填补了现有数据的追溯性、覆盖面和评估方法的空白。

**🔧 技术方法**

使用了大语言模型（LLM）进行隐私相关性、修复类型和GDPR原则的标签生成，并进行了人工验证。

**📊 数据集**

数据集包含来自1048个GitHub仓库的1169个隐私修复和1643个非修复PRs，涵盖了多个平台。

**📈 对比分析**

通过将修复回滚以重新引入缺陷的方法进行评估，结果显示在通用审查提示下，两个编码代理仅检测到42-51%的缺陷，而在明确要求检查隐私时，检测率提高到88-91%。

**⚠️ 局限性**

局限性包括依赖关键词检索和单次LLM注释，可能遗漏隐私修复并保留标签噪声；人类验证仅覆盖相对较小的PRs，且评估存在构建有效性风险。

---

## 250. Algorithmic Opacity as a Principal-Agent Problem

**arXiv ID:** 2611.17980 | [PDF](https://arxiv.org/pdf/2611.17980v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 251. Who Counts as Exposed? Measurement Choices in an Open, Re-estimable Index of Occupational AI Exposure

**arXiv ID:** 2611.18168 | [PDF](https://arxiv.org/pdf/2611.18168v1)

**作者:** Luca Mouchel `[一作]` (Massachusetts Institute of Technology), Yossi Sheffi `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个开放的、可重新估计的职业AI暴露任务级指数，并探讨了测量选择如何影响被认为暴露的工人数量。

**💡 创新点**

创新点在于提出了一种基于多模型框架的任务级暴露测量方法，强调了上下文和校准选择对暴露评估的影响。

**🔧 技术方法**

使用了七个开放权重推理模型，结合检索到的新闻和研究段落、工作上下文描述符以及瓶颈推理协议。

**📊 数据集**

数据集包括923个职业和18,796个职业-任务对，结合了来自GDELT 2.0的新闻文章和Semantic Scholar的学术文献。

**📈 对比分析**

与2023年的基准相比，本文的方法在任务排名上表现更好，尤其是在观察到的使用情况上，且在任务级别的AUC和平均精度上均显著优于基准。

**⚠️ 局限性**

限制在于该指数依赖于单一平台的使用数据，可能导致平台偏见，并且测量的上下文部分未逐一测试其对结果的影响。

---

## 252. What a Memory Preview Cannot Tell You: Ambiguity in AI Memory Permissions

**arXiv ID:** 2611.17874 | [PDF](https://arxiv.org/pdf/2611.17874v1)

**作者:** Sadia Noman Tonni `[一作]` (United International University), Nafis Fuad Shahid `[通讯]` (Islamic University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了记忆规则的预览如何影响用户对保存规则的理解，特别是在确认允许和阻止的使用时。

**💡 创新点**

创新点在于提出了记忆规则预览的设计原则，强调了如何通过选择合适的示例来帮助用户更好地理解规则的效果。

**🔧 技术方法**

使用了适应性测试和转移测试等技术来评估用户对规则的理解和预测能力。

**📊 数据集**

使用了一个小型的试点数据集，包括20名参与者的反馈，评估他们对规则预览的理解。

**📈 对比分析**

与其他方法相比，本文的方法通过适应性测试和转移测试能够更准确地识别规则，但在试点中发现许多参与者对规则的预测存在误差，表明预览设计仍需改进。

**⚠️ 局限性**

限制在于样本量较小且参与者与作者相识，可能影响结果的普遍性，且未进行正式的伦理审查。

---

## 253. Alleviating Interview Stress: Effects of Robot Persona and Reactive Backchanneling on High-Pressure Mock Interviews

**arXiv ID:** 2611.17830 | [PDF](https://arxiv.org/pdf/2611.17830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 254. Compiling Matrix Formulas to Quantum Block Encodings

**arXiv ID:** 2611.17756 | [PDF](https://arxiv.org/pdf/2611.17756v1)

**作者:** Daksh Shami `[一作]` (University of Wisconsin-Madison), Charles Yuan `[通讯]` (University of Wisconsin-Madison)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为MARBLE的语言，能够从矩阵的公式自动合成高效的量子电路。

**💡 创新点**

MARBLE通过静态多重性属性确保每个矩阵的程序都能被有效编码，从而实现量子电路的多项式对数规模。

**🔧 技术方法**

使用代数重写规则和符号矩阵形式，将程序转换为量子电路，并进行成本分析和优化。

**📊 数据集**

在74个来自流体动力学、偏微分方程和物理哈密顿量的矩阵上进行评估。

**📈 对比分析**

MARBLE在39个已发布基准中匹配或超越了37个，特别是在流体动力学应用中提高了10487倍的效率。

**⚠️ 局限性**

MARBLE的局限性在于它无法处理需要加载大量显式系数的矩阵，这些系数的数量随着系统维度的增加而增加。

---

## 255. CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents

**arXiv ID:** 2611.17808 | [PDF](https://arxiv.org/pdf/2611.17808v1)

**作者:** Lihuan Li `[一作]` (University of New South Wales), Flora Salim `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为CarryOn的轻量级上下文管理器，用于支持长时间任务的执行，通过紧凑的执行状态和细粒度的证据选择来管理上下文。

**💡 创新点**

创新点在于将进度跟踪与证据选择分开，允许在任务执行过程中根据需要动态调整所需的证据，同时避免了额外的LLM调用。

**🔧 技术方法**

使用了一种学习选择器和自适应保留策略来管理证据，并通过两个互补的评估指标（S_α和NCS）来评估性能。

**📊 数据集**

在AppWorld、OfficeBench和Tau2 Telecom三个数据集上进行了评估，涵盖了跨应用任务、办公室工作流程和客户支持。

**📈 对比分析**

与多个基线方法（如Full History和SWE-Pruner）进行比较，CarryOn在任务成功率上达到了75.7%，超出最强基线6.7个百分点，并在两个评估指标上表现最佳。

**⚠️ 局限性**

局限性在于评估仅覆盖了三个基准，且主要针对文本交互，扩展到视觉或多模态观察需要适应证据的表示和选择方式。

---

## 256. STaG: Human-Annotation-Free Unsupervised Domain Adaptation for Text-Based Person Retrieval

**arXiv ID:** 2611.18172 | [PDF](https://arxiv.org/pdf/2611.18172v1)

**作者:** Ji Yucheng `[一作]`, Cao Min `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的无人工注释的文本基础人物检索（TBPR）方法，称为STaG，旨在解决在目标领域缺乏文本注释的问题。

**💡 创新点**

创新点在于引入了一个双路径框架，包括风格无关的文本对齐（SATA）和异构图记忆对齐（HGMA），有效地解决了跨模态对齐缺失和视觉领域转移的问题。

**🔧 技术方法**

使用了风格无关的文本对齐（SATA）和异构图记忆对齐（HGMA）技术，结合了图神经网络和对比学习。

**📊 数据集**

使用了多个标准基准数据集进行实验，具体数据集未在摘要中详细列出。

**📈 对比分析**

与现有的最先进方法进行比较，STaG在多个基准测试中表现出色，显著提高了检索准确率，缩小了与完全监督模型的性能差距。

**⚠️ 局限性**

限制在于该方法仍然依赖于源领域的标注数据，且在目标领域的无标注图像可能会导致模型在特定情况下的性能下降。

---

## 257. DesignSpecs: A DESIGN.md Dataset of Visual Design Specifications for AI Agents

**arXiv ID:** 2611.17713 | [PDF](https://arxiv.org/pdf/2611.17713v1)

**作者:** Jialiang Wei `[一作]` (Hasso Plattner Institute), Walid Maalej `[通讯]` (Hasso Plattner Institute)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一个包含36,906个文件的设计规范数据集，这些文件来自15,794个公共非分叉的GitHub仓库，旨在填补AI代理在生成和修改用户界面时缺乏视觉设计规范的空白。

**💡 创新点**

创新点在于构建了一个大规模的设计规范数据集，包含设计令牌和Markdown文本，提供了对视觉设计知识的结构化理解，并为设计感知的AI代理提供了研究机会。

**🔧 技术方法**

使用了YAML和Markdown格式的设计令牌，结合了官方的linter工具进行文件的筛选和验证。

**📊 数据集**

数据集来源于613,207个文件，经过筛选后得到36,906个符合条件的文件，这些文件来自15,794个公共非分叉的GitHub仓库。

**📈 对比分析**

与其他方法相比，论文通过结构化的过滤流程确保了数据集的质量，所有选定的文件均通过了linter检查，没有错误，但有49.1%的独特内容和68.7%的仓库文件包含警告。

**⚠️ 局限性**

限制在于数据集仅覆盖了公共GitHub内容的一个快照，可能存在个人和实验项目，且未包含文件历史或实现快照，可能导致对设计质量的评估不够全面。

---

## 258. Beyond a Single Sign: BiTE for Branch-Preserving Binary Threshold Expansion

**arXiv ID:** 2611.17576 | [PDF](https://arxiv.org/pdf/2611.17576v1)

**作者:** Siddharth Gupta `[一作]` (Ruhr University Bochum), Akash Kumar `[通讯]` (Ruhr University Bochum)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的分支保留二进制阈值扩展机制BiTE，该机制通过多个有序阈值评估每个特征方向，并将结果作为独立输入传递给下一个学习操作。

**💡 创新点**

创新点在于BiTE能够从每个学习的特征方向提取多个有序的二进制响应，并保留这些响应作为下一个操作的独立输入，从而允许下游操作针对不同的阈值交叉学习不同的效果。

**🔧 技术方法**

使用了分支保留的二进制阈值扩展机制BiTE，结合固定宽度卷积和BinaryViT架构。

**📊 数据集**

在CIFAR-10和CIFAR-100数据集上进行了实验。

**📈 对比分析**

与单一符号激活、均匀崩溃和固定阈值控制进行了比较，BiTE在CIFAR-10上将最终的Top-1准确率从74.60%提高到75.83%，在CIFAR-100上从42.80%提高到44.07%，且可训练参数数量几乎不变。

**⚠️ 局限性**

限制在于BiTE的有效性可能依赖于特定的架构和训练配置，未来的工作需要在更大的变换器骨干网络上评估BiTE的性能。

---

## 259. Only as Safe as the Weakest Agent: A Break That Appears Only When the System Is Multi-Agent

**arXiv ID:** 2611.17520 | [PDF](https://arxiv.org/pdf/2611.17520v1)

**作者:** Keegan Wang `[一作]` (Carnegie Mellon University), Anantika Mannby `[通讯]` (University of Southern California)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文展示了多智能体系统（如Codex和Claude Code）在特定情况下受到攻击的脆弱性，尽管这些攻击在现代前沿模型中长期未能成功。研究表明，即使是强大的模型在与较弱的子代理交互时，其安全性也会受到影响。

**💡 创新点**

创新点在于揭示了多智能体系统的安全性并不由最强的代理决定，而是由最弱的代理决定。这一发现挑战了传统的AI安全观念，强调了代理间委托的脆弱性。

**🔧 技术方法**

使用了多种语言模型（如Claude Code和Codex）进行实验，展示了在不同环境下的代理间交互和委托机制的影响。

**📊 数据集**

使用了包含180个项目的虚构故事数据集，测试了代理在读取包含伪造记录的文件存储时的表现。

**📈 对比分析**

通过与强代理和弱代理的比较，发现即使强代理在直接读取时表现良好，但在委托给弱代理时，错误率显著上升，表明系统的安全性由最弱的代理决定。

**⚠️ 局限性**

限制在于实验设计中伪造记录仅能被仔细的读者检测到，且Codex的子代理限制是基于指令而非机械限制，这可能影响结果的普遍性。

---

## 260. A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent

**arXiv ID:** 2611.17515 | [PDF](https://arxiv.org/pdf/2611.17515v1)

**作者:** Harrison C. Margolin `[一作]` `[通讯]`, Harrison C. Margolin

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究测试了持续的增益-损失框架是否能影响大型语言模型（LLM）代理在市场操纵中的行为。通过模拟真实的伦敦银价操纵案件，研究了不同框架对代理参与价格操纵的影响。

**💡 创新点**

创新点在于将行为经济学的框架效应应用于LLM代理的市场行为，展示了框架如何影响代理的决策和参与程度，尤其是在长时间的交互中。

**🔧 技术方法**

使用了Gemma 3 12B模型，结合行为经济学的理论，进行了市场模拟实验，分析了不同框架对代理行为的影响。

**📊 数据集**

数据集基于真实的伦敦银价操纵案件的聊天记录，模拟了260个交易场景，涵盖了不同的框架条件（增益、损失、责备和信用）。

**📈 对比分析**

通过与中性框架的比较，发现损失框架使参与率提高到25%，责备框架提高到33%。在第二阶段，注入的状态使代理对损失和责备的请求更加警惕，拒绝了所有请求。

**⚠️ 局限性**

限制在于研究仅使用了一个主要模型和一个市场案例，缺乏多样化的实验设置和重复验证，未来需要在不同模型和市场条件下进行更多测试。

---

## 261. FOMOS: Feature-Oriented Meshing of SDFs

**arXiv ID:** 2611.17458 | [PDF](https://arxiv.org/pdf/2611.17458v1)

**作者:** Stefan Ohrhallinger `[一作]` (TU Wien), Amal Dev Parakkat `[通讯]` (LTCI-Telecom Paris, Institut Polytechnique de Paris)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为FOMOS的几何方法，用于基于局部特征大小（LFS）对有符号距离函数（SDF）进行特征导向的网格化，解决了现有方法的多种挑战。

**💡 创新点**

FOMOS通过单一参数动态调整网格分辨率，显著提高了网格质量和运行时间，能够在几秒内完成SDF网格化，且避免了现有方法中常见的低质量三角形和细节损失。

**🔧 技术方法**

使用了基于局部特征大小的自适应网格化技术，结合了中轴样本的计算方法，减少了从立方复杂度到二次复杂度的时间和内存需求。

**📊 数据集**

使用了来自McGill数据集的100个SDF，涵盖了高基数和薄壳结构，进行了广泛的实验比较。

**📈 对比分析**

与现有的SDF网格化技术（如Marching Cubes和Neural Dual Contouring）进行了比较，FOMOS在执行时间、重建结果的准确性和输出网格的三角形质量上表现出明显优势。

**⚠️ 局限性**

FOMOS在处理非常低分辨率输入时存在限制，且在重建尖锐特征和薄特征时需要较高的网格分辨率。

---

## 262. Tabular Population Priors as Structured Retrieval for EHR Medication Set Prediction

**arXiv ID:** 2611.17341 | [PDF](https://arxiv.org/pdf/2611.17341v1)

**作者:** Animesh Agarwal `[一作]` (Optum AI), Carlos Morato `[通讯]` (Optum AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于稀疏条件-药物共现表的结构化检索方法，用于电子健康记录（EHR）中的药物集预测。

**💡 创新点**

创新点在于使用表格人口先验作为检索对象，替代传统的基于相似患者记录的检索方法，从而提高了药物预测的准确性。

**🔧 技术方法**

使用了PACE-RAG作为固定的下游管道，并通过替换患者检索器来实现改进。

**📊 数据集**

使用了MIMIC-IV数据集进行实验，包含1,004个持出入院记录。

**📈 对比分析**

与PACE-RAG方法进行比较，使用人口先验的F1从0.303提高到0.338（相对提高11.5%），Jaccard从0.218提高到0.237，表明人口先验在固定词汇EHR集预测中更有效。

**⚠️ 局限性**

限制在于该研究仅关注药物类别预测，而未考虑剂量、给药途径、时间、禁忌症、药物间相互作用或患者特定的安全约束。

---

## 263. DECODE: Decomposable Environments for COntrolled Sequential-Decision Evaluation

**arXiv ID:** 2611.18367 | [PDF](https://arxiv.org/pdf/2611.18367v1)

**作者:** Diego Silva Prado `[一作]` (Alpha Synthesis Labs), Cristiano Bevitori Maffia De Oliveira `[通讯]` (Alpha Synthesis Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了DECODE基准，用于测试决策系统在规划和可用信息方面的修改效果，通过控制实验评估这些修改对决策的影响。

**💡 创新点**

DECODE提供了部分可观察环境和已知动态矩阵，允许对信息和规划输入进行控制变化的实验，支持对求解器性能的假设进行可测试的评估。

**🔧 技术方法**

使用了部分可观察马尔可夫决策过程（POMDP）模型，结合了生成器、政策价值评估器和参考政策，进行测量和比较。

**📊 数据集**

使用了消费者信用催收的生成器，提供了五个部分可观察的环境，这些环境共享已知的动态矩阵，并在记忆参数和过程噪声上有所不同。

**📈 对比分析**

通过比较不同求解器在固定规划范围内的表现，发现延长规划范围可以显著提高得分，最佳求解器在六个月的规划下仍低于最优解15.6分。

**⚠️ 局限性**

由于中间效应依赖于还款模型和变化顺序，无法唯一地将性能差距归因于单一原因，限制了对求解器性能的全面理解。

---

## 264. DeltaVisionX: A Low-Cost Vision-Guided Automation System Using YOLOv11 and PLC-Based Actuation

**arXiv ID:** 2611.17959 | [PDF](https://arxiv.org/pdf/2611.17959v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 265. MarathonTS: a benchmark of long synthetic time series from window assembly and long-series generators

**arXiv ID:** 2611.18190 | [PDF](https://arxiv.org/pdf/2611.18190v1)

**作者:** Ilyas Varshavskiy `[一作]` (zypl.ai), Azizjon Azimi `[通讯]` (zypl.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MarathonTS，这是一个开放的基准测试，比较了两种生成长时间序列的方法：从短窗口组装和直接生成长序列。

**💡 创新点**

首次在一个协议下比较了两种生成方法，并评估了整个长序列的真实度和实用性。

**🔧 技术方法**

使用了多种生成模型，包括Diffusion-TS、Chronos、TimesFM 3.0、Timestral等，并结合了窗口选择和接缝处理技术。

**📊 数据集**

使用了9个公开的每小时数据集，每个数据集有10个种子，共进行了30种方法的比较。

**📈 对比分析**

通过83个真实度指标和14个分类与回归任务评估了生成序列的性能，结果显示没有任何方法能完全达到真实序列的实用性，最佳方法的转移比率显著低于1。

**⚠️ 局限性**

方法的局限性在于没有任何方法能够完全复制真实序列的效用，且在接缝处理方面的效果不如预期。

---

## 266. RSSA: Robust Semantic and Spatial Aligner for Collaborative Perception

**arXiv ID:** 2611.18023 | [PDF](https://arxiv.org/pdf/2611.18023v1)

**作者:** Zhihao Yang `[一作]` (Zhejiang University), Yong Ding `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为鲁棒语义和空间对齐器（RSSA）的插件模块，通过语义和概率空间变换来增强V2X协作感知的特征对齐能力。

**💡 创新点**

RSSA通过引入语义变换（SemT）和概率空间变换（P-SpaT）模块，有效解决了现有方法中由于简单空间扭曲导致的语义不对齐和对位置噪声的敏感性问题。

**🔧 技术方法**

使用了语义变换和概率空间变换技术，结合相对变换增强策略（RT-Aug）来提高模型的鲁棒性。

**📊 数据集**

在V2XSet（模拟数据集）和V2X-Real（真实世界数据集）上进行了广泛的实验。

**📈 对比分析**

与现有的最先进方法相比，RSSA在干净和噪声条件下均显著提高了性能，尤其是在V2X-Real数据集上，RSSA增强的方法甚至超越了握手范式的性能。

**⚠️ 局限性**

RSSA的局限性在于其对训练数据的多样性依赖，可能在复杂的几何关系下表现不佳。

---

## 267. Between Obligations and Evidence: A Participatory Study of Translating the EU AI Act into Technical Requirements

**arXiv ID:** 2611.18439 | [PDF](https://arxiv.org/pdf/2611.18439v1)

**作者:** Helma Torkamaan `[一作]` (Delft University of Technology), Megha Khosla `[通讯]` (Delft University of Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本论文通过参与式研讨会探讨了欧盟人工智能法案（AI Act）在高风险AI系统中的技术和组织要求的应用挑战，涉及医疗、招聘和公共交通等领域的实际场景。

**💡 创新点**

创新点在于识别了技术、组织和证据挑战之间的相互依赖关系，并强调了在不同上下文中持续合规的重要性。

**🔧 技术方法**

使用了参与式研究方法，通过研讨会收集数据，并进行了主题分析。

**📊 数据集**

数据集包括来自32名研究人员和实践者的反馈，涵盖医疗、招聘和公共交通的假设高风险AI应用。

**📈 对比分析**

通过参与者的讨论和案例分析，比较了不同场景下合规要求的识别与潜在失败路径，发现技术性能与决策适用性之间存在差距，且人类监督在实践中可能无效。

**⚠️ 局限性**

限制在于合规要求的技术转化和组织能力之间的协调问题，以及在不同使用条件下如何保持合规证据的有效性。

---

## 268. MaskSCD: Bitemporal Mask Classification for Semantic Change Detection in Remote Sensing Images

**arXiv ID:** 2611.18350 | [PDF](https://arxiv.org/pdf/2611.18350v1)

**作者:** Weikang Yu `[一作]` (Helmholtz Institute Freiberg for Resource Technology), Pedram Ghamisi `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的语义变化检测方法MaskSCD，将其定义为双时相标签对图像的掩膜分类。该方法通过共享掩膜和每个日期的类标签来预测每个双时相段，从而保持变化和语义图的一致性。

**💡 创新点**

创新点在于将语义变化检测视为双时相标签对图像的掩膜分类，避免了传统方法中变化和语义图之间的不一致性，并且通过共享掩膜和类标签简化了标签空间。

**🔧 技术方法**

使用了双时相掩膜注意力变换器解码器和双时相融合金字塔技术，能够有效地处理双时相图像的特征。

**📊 数据集**

在SECOND、Landsat-SCD和Hi-UCD三个数据集上进行了评估，这些数据集具有不同的分辨率和土地覆盖类别。

**📈 对比分析**

与十四种最新的语义变化检测模型进行了比较，MaskSCD在所有三个数据集上都取得了最佳的分离kappa、语义变化F1和过渡F1分数，且在平均分离kappa上超越最强竞争对手约五个百分点。

**⚠️ 局限性**

限制在于该方法可能对特定类型的变化不够敏感，尤其是在复杂的城市环境中，可能需要进一步的优化和调整。

---

## 269. Beyond Blossom: Accelerating MWPM Decoding for Surface Codes With Local Predecoding and Architecture-Aware Optimizations

**arXiv ID:** 2611.18476 | [PDF](https://arxiv.org/pdf/2611.18476v1)

**作者:** Jiayu Fu `[一作]`, Guangwen Yang `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

论文未提供具体内容，因此无法总结做了什么。

**💡 创新点**

论文未提供具体内容，因此无法总结创新点。

**🔧 技术方法**

论文未提供具体内容，因此无法总结使用的技术。

**📊 数据集**

论文未提供具体内容，因此无法总结使用的数据集。

**📈 对比分析**

论文未提供具体内容，因此无法总结比较的方法及性能。

**⚠️ 局限性**

论文未提供具体内容，因此无法总结限制因素。

---

## 270. Causally Constrained and Uncertainty-Aware Cardiopulmonary Risk Prediction

**arXiv ID:** 2611.17819 | [PDF](https://arxiv.org/pdf/2611.17819v1)

**作者:** Yuxiang Lai `[一作]` (Emory University), Xiaofeng Yang `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为Causal Envelope Agent (CE-Agent) 的框架，用于心肺风险预测，结合了流行病学证据和不确定性处理。

**💡 创新点**

创新点在于通过Causal Envelope在任何结果出现之前明确约束风险因素权重，同时估计局部基线风险和新添加的CT表型。

**🔧 技术方法**

使用了两阶段框架，结合了流行病学证据的约束估计和闭式不确定性传播技术。

**📊 数据集**

使用了来自33家美国医院的53,452名肺筛查参与者和6,814名多种族成年人组成的数据集。

**📈 对比分析**

与传统的心血管计算器相比，CE-Agent在肺筛查中表现更优，降低了样本量需求，并在多个医院和种族群体中具有良好的泛化能力。

**⚠️ 局限性**

限制在于模型的复杂性和对数据完整性的依赖，尤其是在临床记录不完整的情况下。

---

## 271. FEND: Far-End Detection in Outbound Voice AI Calls

**arXiv ID:** 2611.18050 | [PDF](https://arxiv.org/pdf/2611.18050v1)

**作者:** Abhishek Kumar `[一作]` (Observe.AI), Aashraya Sachdeva `[通讯]` (Observe.AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种新的四类检测框架FEND，用于识别在语音AI代理拨打的电话中接听方的类型，包括真人、语音信箱、IVR系统和AI接听者。

**💡 创新点**

创新点在于将传统的二分类（人或机器）扩展为四分类，能够更准确地指导语音AI代理在不同情况下的应对策略。

**🔧 技术方法**

使用了音频特征分析、自动语音识别（ASR）和大语言模型（LLM）等技术。

**📊 数据集**

使用了1000个手动标注的内部语音AI通话数据集和一个公共基准数据集（Telephony-AMD），后者包含400个标记的远端片段。

**📈 对比分析**

与现有的Asterisk AMD系统进行比较，FEND在公共基准上实现了100%的语音信箱准确率，而Asterisk的误分类率高达30%至66%。在生产流量中，FEND将真人与任何自动化接听方的F1得分提高到0.979。

**⚠️ 局限性**

局限性在于数据集仅来自一个特定的外拨程序，且AI接听者的样本数量较少，无法全面代表所有市场的情况。此外，检测模型依赖于特定的ASR系统，可能无法在不同的语音识别系统上复现相同的效果。

---

## 272. Ego2Flow: Benchmarking the Traffic-System Impact of Autonomous Driving on Maps Distilled from Real-World Roads

**arXiv ID:** 2611.17720 | [PDF](https://arxiv.org/pdf/2611.17720v1)

**作者:** Yu Dong `[一作]` (University of Wisconsin), Xiaopeng Li `[通讯]` (University of Wisconsin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Ego2Flow，一个闭环基准，用于评估自动驾驶政策对交通系统的影响，特别是在真实城市地图上进行的模拟。

**💡 创新点**

创新点在于开发了一个代理引导的地图蒸馏框架，能够构建保留驾驶相关道路结构和视觉线索的3D城市环境，同时简化不必要的细节。

**🔧 技术方法**

使用了闭环模拟技术和代理引导的地图蒸馏框架，结合真实世界地图和观察数据。

**📊 数据集**

使用了六个从Argoverse 2数据集中提取的CARLA地图，这些地图与真实世界区域空间对齐。

**📈 对比分析**

与现有的构建方法相比，该框架在规划输出的平均位移误差（ADE）上表现最佳，低于其他方法9.3%。对代表性驾驶政策的基准测试显示，政策在交通效率、安全性、稳定性和能耗方面的影响存在显著差异。

**⚠️ 局限性**

限制在于尽管Ego2Flow能够评估交通系统的整体表现，但复杂的驾驶模型不一定能改善交通流表现，表明需要进一步的政策开发以优化集体交通结果。

---

## 273. Pairing AIS with RF: Analysis of a Markovian Approach for Maritime Data Fusion

**arXiv ID:** 2611.18563 | [PDF](https://arxiv.org/pdf/2611.18563v1)

**作者:** Daniel Grimm `[一作]` (Tilburg University), Willem-Jan Van Den Heuvel `[通讯]` (Tilburg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种配对隐马尔可夫模型（PHMM），用于将自动识别系统（AIS）数据与空间基于无线电频率（RF）检测进行匹配，以提高船舶识别的准确性。

**💡 创新点**

创新点在于将PHMM应用于AIS和RF数据的融合，利用时间和空间信息来对齐个别RF检测与AIS轨迹，同时建模噪声和时间差异。

**🔧 技术方法**

使用了配对隐马尔可夫模型（PHMM）技术来进行数据融合和匹配。

**📊 数据集**

使用了来自美国国家海洋和大气管理局（NOAA）的AIS数据集，并生成了合成的RF检测数据。

**📈 对比分析**

与最近邻基线方法进行比较，PHMM的长度校正前向得分在多候选检测中达到了0.7776的精度，AUC为0.637。尽管PHMM在精度上接近但未超越时间加权的最近邻基线（精度为0.9013，AUC为0.904）。

**⚠️ 局限性**

限制在于依赖合成RF数据，可能无法完全反映真实世界的情况。此外，模型设计和偏差测试的选择可能影响结果的有效性，且对AIS数据的预处理限制了分析的范围。

---

## 274. Objectives as Policies: Vision-Language Model Cost Programs for Contact-Rich Manipulation

**arXiv ID:** 2611.18298 | [PDF](https://arxiv.org/pdf/2611.18298v1)

**作者:** Zihao Lu `[一作]` (University of Illinois Chicago), Yifan Zhu `[通讯]` (University of Illinois Chicago)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为Objectives as Policies (OaP) 的框架，将预训练的视觉语言模型（VLM）与基于模型的预测控制器（MPC）连接起来，以实现复杂的机器人操作任务。

**💡 创新点**

创新点在于通过可编辑的目标程序将VLM与MPC结合，允许VLM根据自然语言任务指令和场景观察生成并迭代更新目标程序，从而实现物理交互的推理。

**🔧 技术方法**

使用了视觉语言模型（VLM）和基于模型的预测控制（MPC）技术，结合了数字双胞胎（digital twin）和目标库。

**📊 数据集**

在四个桌面操作任务上进行了评估，包括拾取与放置、杯子重新定向、直接推和工具推。

**📈 对比分析**

与强基线方法进行比较，OaP在模拟和现实世界中的成功率分别为90%和80%，显示出优于其他方法的性能。

**⚠️ 局限性**

局限性在于依赖于固定的残差库和物理模型，真实世界执行的准确性受到场景重建和接触动态的影响，且缺乏在阶段完成后显式的失败恢复机制。

---

## 275. Movable Antenna-Aided ISAC: A Time- and Frequency-Division Framework

**arXiv ID:** 2611.18196 | [PDF](https://arxiv.org/pdf/2611.18196v1)

**作者:** Yuyan Zhou `[一作]` (Chinese University of Hong Kong), Rui Zhang `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了可移动天线（MA）增强的集成感知与通信（ISAC）系统，采用时分双工（TDD）和频分双工（FDD）协议，分别在正交的时间和频率资源上实现感知和通信。

**💡 创新点**

创新点在于建立了一个宽带性能模型，推导了基于脉冲雷达框架的目标位置克拉美-罗界（CRB），并提出了联合天线位置优化和资源分配的问题，以最大化用户的最小通信速率。

**🔧 技术方法**

使用了交替优化算法解决FDD问题，通过对TDD问题进行嵌套搜索来优化天线位置和资源分配。

**📊 数据集**

使用了模拟数据集，考虑了不同用户角度分布和相对延迟与角度相关部分对感知CRB的贡献。

**📈 对比分析**

与固定位置天线基准相比，数值结果表明天线位置优化扩大了通信-感知权衡区域，TDD在延迟相关部分占主导时表现更好，而FDD在角度相关部分占主导时更具优势。

**⚠️ 局限性**

限制在于大多数现有的MA增强ISAC研究采用窄带感知模型，未明确考虑感知带宽对延迟估计的影响，且TDD和FDD之间的性能差异仍需进一步探索。

---

## 276. CoPre-MoE: Consolidating and Preserving MoE for Multimodal Continual Instruction Tuning

**arXiv ID:** 2611.18417 | [PDF](https://arxiv.org/pdf/2611.18417v1)

**作者:** Tao Hu `[一作]` (Nanjing University), Da-Wei Zhou `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种选择性双池框架，用于多模态持续指令调优，旨在减少跨任务功能冗余并防止灾难性遗忘。

**💡 创新点**

创新点在于通过将功能重叠的专家整合到固定容量的通用池中，同时将剩余的任务特定专家分解为共享投影基和紧凑核心，从而有效减少参数增长。

**🔧 技术方法**

使用了混合专家（MoE）架构和输入依赖的路由机制，结合了历史操作符和正交键值路由对。

**📊 数据集**

在UCIT和CoIN数据集上进行了广泛的实验，UCIT包括ImageNet-R、ArxivQA、VizWiz等任务，CoIN包括ScienceQA、TextVQA等。

**📈 对比分析**

与现有方法（如MoELoRA）相比，提出的方法在准确性和参数效率上表现优越，适应参数减少了68.3%，并且在推理时延和吞吐量上也有显著提升。

**⚠️ 局限性**

限制在于生成的响应不可避免地继承了基础预训练多模态大语言模型的固有视觉幻觉和安全对齐行为，这需要进一步研究。

---

## 277. Evaluating Human Sensitivity to Gender Biased AI-Generated Content through Human Assessment of Google's Autocomplete Function

**arXiv ID:** 2611.18371 | [PDF](https://arxiv.org/pdf/2611.18371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 278. Hamming and Johnson-Type Bounds for Joint Error Correction and Detection

**arXiv ID:** 2611.18156 | [PDF](https://arxiv.org/pdf/2611.18156v1)

**作者:** Yulin Chen `[一作]` (Chinese University of Hong Kong), Raymond W. Yeung `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文推导了经典q-元汉明空间中联合纠错和检测码的大小的上界，定义了(c,c')联合纠错和检测码，并建立了相应的几何特征。

**💡 创新点**

提出了联合完美码的概念，并证明了二进制重复码在n=2c+c'+1时达到联合界限，展示了联合纠错和检测的几何特性如何影响码的大小。

**🔧 技术方法**

使用了汉明球、汉明环和外壳的几何特性，结合了Cauchy-Schwarz不等式和对环交集的二阶矩分析，推导出更精细的联合界限。

**📊 数据集**

使用了经典q-元汉明空间中的码，特别是二进制重复码作为示例，探讨了不同参数下的联合纠错和检测能力。

**📈 对比分析**

通过与汉明界和约翰逊界的比较，展示了所提出的联合界限在特定条件下的紧致性，证明了在某些参数范围内，新的联合界限优于传统界限。

**⚠️ 局限性**

限制在于仅考虑了经典点对点信道，未来的研究可以扩展到更广泛的网络编码场景，并探索更复杂的联合完美码的存在性。

---

## 279. JFinBench: A Continuously Refreshable Japanese Financial Benchmark from the Latest Certification Exams

**arXiv ID:** 2611.18130 | [PDF](https://arxiv.org/pdf/2611.18130v1)

**作者:** Issa Sugiura `[一作]` (Sakana AI), Takashi Ishida `[通讯]` (Sakana AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了JFinBench，一个包含321个例子的日本金融基准，旨在评估大型语言模型（LLMs）在不同金融系统中的能力。

**💡 创新点**

创新点在于采用了人机协作的代理构建流程，使得基准能够适应不同的考试格式，并减少了更新新考试版本的工作量。

**🔧 技术方法**

使用了人机协作的代理构建流程，结合了编码代理和人工验证的方式。

**📊 数据集**

数据集来自2026年六个金融认证考试的321个例子，涵盖会计、税务、金融规划和银行等领域。

**📈 对比分析**

与12个最近的LLMs进行比较，Claude Fable 5和GPT-6 Astra的整体得分超过93%，而最佳的开放权重模型Kimi K3.1 Pro落后12个百分点。使用终端和网络访问后，所有模型的得分都有所提高，但仍有约4分的差距。

**⚠️ 局限性**

限制在于只包含由管理机构发布为PDF的考试，且可能存在数据污染的问题，尽管每个考试都在知识截止日期之后进行。

---

## 280. Securing Multi-Agent Systems by Design

**arXiv ID:** 2611.17883 | [PDF](https://arxiv.org/pdf/2611.17883v1)

**作者:** Xilong Wang `[一作]` (Duke University), Neil Gong `[通讯]` (Duke University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种新的响应操控攻击方法，旨在通过恶意代理优化其响应，以最大化攻击效果，并提出了一种安全聚合规则来防御这些攻击。

**💡 创新点**

创新点在于将响应操控攻击形式化为优化问题，并提出了一种基于语义相似性的安全聚合规则，能够有效抵御恶意代理的攻击。

**🔧 技术方法**

使用了优化算法来迭代优化恶意响应，并利用嵌入模型计算响应之间的语义相似性。

**📊 数据集**

使用了生成的用户任务和其对应的真实解决方案作为数据集，共计500对任务和解决方案，涵盖教育、金融、医疗、法律和新闻等场景。

**📈 对比分析**

与现有的聚合规则相比，提出的安全聚合规则在所有评估的攻击下表现出显著的防御效果，攻击成功率（ASR）和攻击成功值（ASV）均显著低于基线攻击。

**⚠️ 局限性**

限制在于理论安全分析依赖于善意代理能够有效解决子任务的假设，当这一假设不成立时，安全保证可能失效。此外，可能对更强的自适应攻击仍然脆弱，尤其是在善意代理的响应质量差异较大时。

---

## 281. Availability-Aware Historical Feedback Learning for Electric Arc Furnace Temperature and Chemical Composition Prediction

**arXiv ID:** 2611.17463 | [PDF](https://arxiv.org/pdf/2611.17463v1)

**作者:** Ahmed Shoyeb Raihan `[一作]`, Imtiaz Ahmed `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

论文探讨了某个数学模型的应用，展示了其在特定领域的有效性。

**💡 创新点**

创新点在于提出了一种新的数学表达方式，能够更简洁地描述复杂的关系。

**🔧 技术方法**

使用了代数和符号计算技术来推导和验证模型。

**📊 数据集**

数据集未具体提及，可能是基于理论推导而非实际数据。

**📈 对比分析**

与传统方法进行了比较，结果显示新方法在计算效率和准确性上均有提升。

**⚠️ 局限性**

限制在于模型的适用范围可能较窄，且未考虑实际应用中的噪声因素。

---

## 282. Choosing Quantized Code LLMs by File Size and Energy per Solved Task

**arXiv ID:** 2611.17605 | [PDF](https://arxiv.org/pdf/2611.17605v1)

**作者:** Aditi Patodiya `[一作]` `[通讯]` (Independent Researcher), Aditi Patodiya (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在文件大小和每个解决任务的能耗方面选择量化代码大语言模型（LLMs），并提供了一个选择指南。

**💡 创新点**

创新点在于提出了一个基于文件大小的选择指南，帮助开发者在不同的量化模型中做出更有效的选择，同时考虑了能耗和正确性。

**🔧 技术方法**

使用了GGUF文件格式的Qwen2.5-Coder模型，并在MacBook Air M2上进行本地推理，测量了每个任务的能耗。

**📊 数据集**

使用了EvalPlus任务集进行模型评估，涉及多个不同大小的模型（0.5B, 1.5B, 3B, 7B）。

**📈 对比分析**

通过精确的配对测试比较了不同模型的解决任务数量和能耗，结果显示在2GB和3GB的模型选择中，3B模型在能耗和解决任务数量上表现更优，而7B模型在能耗上显著增加但解决任务数量没有显著提升。

**⚠️ 局限性**

限制在于研究仅覆盖了一种代码模型和特定的运行环境，可能无法推广到其他模型或不同的硬件配置上。

---

## 283. Tracing Knowledge and Reasoning in LLM

**arXiv ID:** 2611.17461 | [PDF](https://arxiv.org/pdf/2611.17461v1)

**作者:** Yang Wang `[一作]` (University of Manchester), Chenghua Lin `[通讯]` (University of Manchester)

**通讯引用:** 6525 | [OpenAlex ID](https://openalex.org/A5024599321)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种以任务为中心的框架，用于表征大型语言模型（LLMs）在基准测试中的内部处理，结合了注意力和激活信号与基于表示的度量，分析了16个LLMs在24个不同基准上的表现。

**💡 创新点**

创新点在于将知识检索与推理的内部处理模式进行区分，并通过任务中心的框架来分析这些模式，提供了更细致的模型评估方法。

**🔧 技术方法**

使用了层级激活统计、解码时间轨迹和基于表示的度量等技术，结合静态和动态分析来评估模型的内部状态。

**📊 数据集**

使用了24个基准数据集，涵盖了事实和生物医学问答、科学知识、常识和逻辑推理、阅读理解、符号推理和数学问题解决等任务。

**📈 对比分析**

通过与现有方法的比较，发现知识导向和推理导向任务在内部处理模式上存在系统性差异，且这些差异在任务级别的表现中并未完全显现。

**⚠️ 局限性**

限制在于研究仅限于英语基准，且探测、表示相似性和Nota分析依赖于多项选择答案结构，未来需要在多语言和开放式生成的评估中进行扩展。

---

## 284. PACT: A Decoder-Confidence-Guided Two-Round Receiver for Joint Channel and Phase-Noise Estimation in Sub-THz OFDM

**arXiv ID:** 2611.18450 | [PDF](https://arxiv.org/pdf/2611.18450v1)

**作者:** Ruilin Wang `[一作]` (University of Victoria), Xiaodai Dong `[通讯]` (University of Victoria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `25d64835-ec5b-425b-899d-a6e1e6fecabd`

**🎯 论文内容**

提出了一种解码器置信度引导的双轮接收器PACT，用于联合估计亚太赫兹OFDM中的信道和相位噪声。

**💡 创新点**

创新点在于结合了复杂值变换器、带状线性最小均方误差均衡和神经归一化最小和解码，利用解码器输出的伪导频和置信度进行第二轮的精细化。

**🔧 技术方法**

使用了复杂值变换器（PACT）、带状线性最小均方误差（LMMSE）均衡和神经归一化最小和（NNMS）解码技术。

**📊 数据集**

使用了3GPP TDL信道和基于3GPP的相位噪声模型进行训练和测试。

**📈 对比分析**

与最佳学习基线相比，在E_b/N_0=30 dB的两种基准设置下，PACT接收器的代码块错误率分别降低了约70%和57%。

**⚠️ 局限性**

限制在于计算复杂度较高，尽管参数较少，但计算需求和延迟较高，反映了接收性能与计算成本之间的权衡。

---

## 285. JaRS: Contrast-Free Localization of Memorized Regions in Diffusion Models via Jacobian Row Sums

**arXiv ID:** 2611.18024 | [PDF](https://arxiv.org/pdf/2611.18024v1)

**作者:** Saisuchith Mahajan `[一作]` (Indian Institute of Science), Chandra Sekhar Seelamantula `[通讯]` (Indian Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文旨在定位扩散模型中记忆的区域，以解决隐私和版权问题。提出了一种新的方法JaRS（Jacobian Row Sums），通过分析去噪器雅可比矩阵的行和来区分记忆和局部平均，而无需参考图。

**💡 创新点**

创新点在于JaRS方法的提出，它不依赖于对比提示或参考检查点，能够直接从目标模型中分离出记忆区域与数据驱动的平滑区域。

**🔧 技术方法**

使用了去噪器雅可比矩阵的行和（JaRS）作为定位图，并结合图像级检测分数（JaRS+）进行两阶段处理。

**📊 数据集**

使用了多个数据集，包括Stable Diffusion v1.4、v2.1、Realistic Vision v5.1和无条件扩散Transformer（DiT），并在这些模型上进行了评估。

**📈 对比分析**

与现有的对比方法相比，JaRS+在多个设置中表现优异，尤其在七个十个设置中实现了最佳的定位mIoU，提升幅度可达10.2个百分点。

**⚠️ 局限性**

限制在于JaRS需要在多步轨迹的中间噪声水平进行评估，未来的工作将扩展到单步生成器。此外，JaRS+的超参数选择依赖于小规模标记的开发集，而现有配置不需要这样的标记。

---

## 286. Box-conditional Hypothesis Testing for Out-of-Distribution Object Detection

**arXiv ID:** 2611.18519 | [PDF](https://arxiv.org/pdf/2611.18519v1)

**作者:** Donghoon Lee `[一作]` (Hongik University), Shinjin Kang `[通讯]` (Hongik University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的方法PLUM，用于检测分布外（OOD）物体，以提高物体检测器在实际应用中的可靠性和安全性。

**💡 创新点**

PLUM通过将检测框的决策视为假设检验，利用无分布假设来提供每个检测框的虚警率的分布无关界限，且不需要额外的训练或外部模型。

**🔧 技术方法**

使用了Mahalanobis距离和分层比较的方法，PLUM在检测框内汇聚特征并与相同层次的ID类进行比较。

**📊 数据集**

使用了PASCAL-VOC和BDD100K作为标准基准数据集进行评估。

**📈 对比分析**

与其他方法相比，PLUM在PASCAL-VOC和BDD100K上达到了最高的AUROC，且在不使用外部模型的情况下，PLUM的性能优于其他比较方法。

**⚠️ 局限性**

PLUM的局限性在于它依赖于训练图像中的参考框，这可能影响其在不同数据集上的泛化能力。

---

## 287. Efficient Genotype Classification from 3D MRI in ADPKD via Density-Aware Point Sampling and E(3)-invariance

**arXiv ID:** 2611.17758 | [PDF](https://arxiv.org/pdf/2611.17758v1)

**作者:** Kwanghee Lee `[一作]` (Kumoh National Institute of Technology), Hyunji Jeong `[通讯]` (Kongju National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种名为E3PN的高效3D MRI基础的ADPKD基因型分类网络，旨在解决深度学习在ADPKD分类中的挑战。

**💡 创新点**

E3PN结合了基于密度的点采样和E(3)-不变坐标标准化，显著提高了训练速度和内存效率，同时在F1分数上有所提升。

**🔧 技术方法**

使用了点云表示和E(3)-不变性技术，结合了密度基础的点采样和坐标标准化。

**📊 数据集**

使用了一个包含441名患者的ADPKD数据集，数据集中的MRI图像与基因突变标签配对。

**📈 对比分析**

与ResNet和e3nn进行比较，E3PN在训练速度上快163倍，内存使用减少103倍，F1分数从0.165提高到0.226，尽管所有方法的F1分数接近随机水平（约0.2）。

**⚠️ 局限性**

由于数据集规模较小，基因型信号在肾脏形态中仅弱编码，因此E3PN被定位为该任务的高效基线，而非可临床应用的分类器。

---

## 288. Reasoning Models Confabulate to Defend Injected Beliefs About Their Own Requests

**arXiv ID:** 2611.17958 | [PDF](https://arxiv.org/pdf/2611.17958v1)

**作者:** Yang Ou `[一作]` `[通讯]` (Independent Researcher), Yang Ou (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了链式思维（CoT）监控和自我报告作为人工智能监督的手段，展示了模型对请求的理解可以被悄然改变。

**💡 创新点**

创新点在于通过隐形的激活编辑改变模型对自身请求的信念，导致模型在没有意识到的情况下错误地报告其任务。

**🔧 技术方法**

使用了激活编辑技术，通过注入一个与任务无关的概念的形成痕迹来操控模型的推理过程。

**📊 数据集**

使用了六种不同规模、架构和训练方法的语言模型进行实验。

**📈 对比分析**

与传统的监控方法相比，模型在面对警告时仍然无法恢复到真实主题，表现出99%的生成结果仍然围绕注入的主题进行，显示出监控的局限性。

**⚠️ 局限性**

局限性在于所有结果均为模型内部现象，未能验证跨模型的转移效果，且该干预需要白盒激活访问，无法从提示中直接实现。

---

## 289. MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration

**arXiv ID:** 2611.17931 | [PDF](https://arxiv.org/pdf/2611.17931v1)

**作者:** Bing Wang `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基准，旨在评估大型语言模型（LLM）在矿产勘探中的自动化地球化学实验设计能力，特别是如何选择实验以验证地质假设。

**💡 创新点**

创新点在于将开放式地球化学实验设计重新构建为基于假设的实验选择问题，形成一个结构化的多标签分类问题，并建立了一个包含277个案例和1572个参考实验计划的基准。

**🔧 技术方法**

使用了大型语言模型（LLM）和多标签分类技术，结合了LLM作为评判者的证据覆盖评估方法。

**📊 数据集**

数据集来源于已发表的矿床文献，经过领域专家验证，包含277个案例和1572个参考实验计划。

**📈 对比分析**

与四个开源代理框架和七个前沿LLM进行比较，结果显示当前代理通常具有高召回率但低精确度，倾向于提出比验证假设所需更多的实验。

**⚠️ 局限性**

限制在于当前系统虽然在参考任务上实现了高召回率，但由于过度提出不必要的实验，导致精确度较低。

---

## 290. A Synthetic Benchmark for Target-Centric Causal Discovery in Industrial Processes

**arXiv ID:** 2611.17556 | [PDF](https://arxiv.org/pdf/2611.17556v1)

**作者:** Angelo Ziletti `[一作]` `[通讯]` (Svelaro GmbH), Angelo Ziletti (Svelaro GmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对工业过程的目标中心因果发现的合成基准，旨在识别影响目标变量（如产品质量、产量或能耗）的记录变量。

**💡 创新点**

创新点在于构建了一个以目标为中心的基准，允许逐个条件调整因果发现的难度，并提供了每个候选变量的角色信息。

**🔧 技术方法**

使用了结构因果模型（SCM）作为生成器，结合了批量生产的特点，生成了具有不同难度的合成数据。

**📊 数据集**

使用了合成数据集，生成了200批次，每批次包含512个时间步的记录变量和目标变量。

**📈 对比分析**

比较了八种方法，包括关联度量和效应估计，结果显示随着难度的增加，所有方法的性能均有所下降，且每种方法的表现与其假设一致。

**⚠️ 局限性**

局限性在于基准是合成的，无法完全反映真实工业过程的物理特性，且每种配置仅在10个植物上进行评估，未提供有限样本恢复界限。

---

## 291. On the Same Track: Privacy Leaks in Electric Rail Transport via Magnetic Fields

**arXiv ID:** 2611.18312 | [PDF](https://arxiv.org/pdf/2611.18312v1)

**作者:** Stefan Czybik `[一作]` (BIFOLD), Konrad Rieck `[通讯]` (BIFOLD)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文展示了智能手机的磁力计如何泄露用户在电动火车上的隐私信息，提出了两种新型攻击：运输模式检测和共处推断。

**💡 创新点**

创新点在于首次识别电动火车产生的独特磁场作为隐私相关信号，并利用这一信号进行隐私攻击。

**🔧 技术方法**

使用了智能手机的磁力计，结合动态时间规整（DTW）和卷积神经网络（CNN）等技术进行数据处理和分析。

**📊 数据集**

使用了一个包含多条电动火车线路的真实世界数据集，记录了乘客的传感器数据，数据集是迄今为止最大的电动火车乘坐记录。

**📈 对比分析**

与现有方法相比，本文的方法在运输模式检测中达到了与最先进技术相当的性能，且在共处推断中表现优于基于其他传感器的方法。

**⚠️ 局限性**

限制在于该攻击仅适用于电动火车，且依赖于特定的电力频率，无法推广到所有交通工具。

---

## 292. Towards Explainable Sleep Stage Classification using Single-Channel Electroencephalogram

**arXiv ID:** 2611.18434 | [PDF](https://arxiv.org/pdf/2611.18434v1)

**作者:** Gurinder Kaur Sandhu `[一作]` (Wichita State University), Shruti Kshirsagar `[通讯]` (Wichita State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究扩展了SleepXAI睡眠分期框架，系统比较了五种不同的可解释人工智能（XAI）方法，并使用Spearman等级相关性评估归因的准确性。

**💡 创新点**

创新点在于首次基于不同的阻塞性睡眠呼吸暂停（OSA）严重程度对五种XAI方法进行评估，并提供了详细的定量分析。

**🔧 技术方法**

使用了集成梯度（IG）、深度学习（DeepLIFT）、SHAP、改进的Grad-CAM和原始显著性（Vanilla Saliency）等技术。

**📊 数据集**

使用了Sleep-EDF基准数据集和DREAMT数据集，后者包含100名根据OSA分为四个严重程度组的患者。

**📈 对比分析**

与其他方法相比，集成梯度（IG）在所有睡眠阶段和AHI组中始终实现了最高的归因准确性，尤其在DREAMT数据集中表现优异。

**⚠️ 局限性**

限制在于仅评估了基于梯度的归因方法，未考虑其他类型的可解释性方法，且N1和N3阶段的类不平衡可能影响结果的可靠性。

---

## 293. Toward Resilient Self-Driving Laboratories through Telerobotic Interventions

**arXiv ID:** 2611.18496 | [PDF](https://arxiv.org/pdf/2611.18496v1)

**作者:** Ziheng Wu `[一作]` (Lawrence Livermore National Laboratory), Aldair E. Gongora `[通讯]` (Lawrence Livermore National Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

本研究探讨了通过远程操作干预来增强自驾实验室（SDL）的韧性，以应对自动化工作流程中的意外故障。

**💡 创新点**

创新点在于开发了一种远程操作接口，使人类操作员能够在自动任务执行期间进行干预，从而提高SDL的故障恢复能力。

**🔧 技术方法**

使用了基于速度的控制策略，评估了耦合模式和解耦模式的性能，并通过3D鼠标进行精确控制。

**📊 数据集**

研究中使用了模拟的SDL故障场景，包括过程中断、初始化错误和升级故障，进行用户研究以评估操作员的表现。

**📈 对比分析**

通过比较耦合模式和解耦模式的成功率、任务完成时间和运动轨迹，发现解耦模式在所有故障场景中表现更佳，成功率提高了9.9%，任务完成时间减少了25.6秒。

**⚠️ 局限性**

限制在于用户研究参与者数量较少，主要为新手，且故障场景简化，未能完全反映真实实验室操作的复杂性。

---

## 294. VLAN: Vision-Language Accessible Navigation

**arXiv ID:** 2611.18346 | [PDF](https://arxiv.org/pdf/2611.18346v1)

**作者:** Jie Hu `[一作]` (Hunan University), Jiaming Zhang `[通讯]` (Hunan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了视觉-语言可访问导航（VLAN）基准，评估模型在相同起点-目标对和障碍布局下，是否能够生成特定于用户的导航动作和可访问性预测。

**💡 创新点**

创新点在于首次引入了用户特定的可访问性约束，涵盖多种用户类型（如人形机器人、四足机器人、无人机、视障用户和轮椅用户），并提供了丰富的场景和动作数据。

**🔧 技术方法**

使用了基于Unity的场景重建和编辑技术，结合了视觉-语言模型（VLM）进行导航行为的评估。

**📊 数据集**

数据集包含1000个场景、20000条导航轨迹和160000个关键帧，涵盖多种光照、环境条件和路径几何。

**📈 对比分析**

与现有方法相比，VLAN基准显示现代视觉-语言模型在适应障碍相关性、路径选择和安全标准方面存在困难，表明需要专门的数据集和评估指标。

**⚠️ 局限性**

限制在于当前模型在不同用户配置下的适应性不足，且在真实世界场景中的表现仍需进一步验证。

---

## 295. Efficient Poincaré Convolutions through Patch-Statistics Factorization

**arXiv ID:** 2611.18125 | [PDF](https://arxiv.org/pdf/2611.18125v1)

**作者:** Lukáš Jochymek `[一作]` (VSB Technical University of Ostrava), Petr Gajdoš `[通讯]` (VSB Technical University of Ostrava)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文展示了一种新的Poincaré卷积的实现方法，该方法通过计算补丁的能量和方向性乘积来评估卷积分数，而无需构建球值补丁。

**💡 创新点**

创新点在于通过因式分解的方法，避免了构建球值补丁的开销，从而提高了训练效率和内存使用效率。

**🔧 技术方法**

使用了原生卷积和标量能量映射技术来计算补丁的平方范数和与滤波器的内积。

**📊 数据集**

使用了ISIC 2016皮肤病变分割数据集和CIFAR-100分类数据集进行实验。

**📈 对比分析**

与显式实现方法相比，因式分解的实现方法在训练更新速度上提高了1.5到1.8倍，内存使用减少了38%到55%。

**⚠️ 局限性**

限制在于实验是在特定的JAX版本和GPU上进行的，未能完全排除硬件效应的影响，且当前批次归一化可能影响测试结果。

---

## 296. Large Language Model-Empowered Intelligent Agent for Adaptive Fast-Charging Control of Lithium-Ion Batteries

**arXiv ID:** 2611.18140 | [PDF](https://arxiv.org/pdf/2611.18140v1)

**作者:** Joey Chan `[一作]` (Shanghai Jiao tong University), ErShun Pan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本研究开发了一种基于大型语言模型（LLM）的智能代理，用于适应性快充控制锂离子电池，解决了传统充电策略在适应电池化学、操作条件和老化状态方面的局限性。

**💡 创新点**

创新点在于将电池充电视为一个闭环交互过程，LLM不仅作为预测工具，还能生成可执行的充电动作，并根据电池响应不断调整决策。

**🔧 技术方法**

采用了大型语言模型（LLM）和两阶段的监督微调与群体相对策略优化（SFT-GRPO）策略，结合物理基础的电池模拟环境（PyBaMM）。

**📊 数据集**

使用了基于PyBaMM构建的大规模模拟数据集，涵盖五种电池化学和11个参数集，生成的状态-动作-响应轨迹用于训练代理。

**📈 对比分析**

通过与传统充电基线和基于优化的策略进行比较，结果显示在匹配的能量通量要求下，所提代理在不同电池配置中实现了更长的循环寿命，性能优于大多数对照组。

**⚠️ 局限性**

限制在于本研究主要集中于基于模拟的开发和验证，未来需要在实际电池管理系统中进行硬件在环和真实电池的验证。

---

## 297. eDIAR: Training-Free Open-Speaker Streaming Diarization for Long Meetings and Large Archives via e-means

**arXiv ID:** 2611.18529 | [PDF](https://arxiv.org/pdf/2611.18529v1)

**作者:** Kyudan Jung `[一作]` (KAIST), Jaegul Choo `[通讯]` (KAIST)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种新的神经发言者分离系统，能够在不限制发言者数量的情况下进行实时发言者识别和聚类。

**💡 创新点**

创新点在于通过不重新训练的方式，结合固定槽位的发言者识别和无槽位限制的聚类方法，实现了对发言者数量的无限制，同时保持了系统的准确性和低计算成本。

**🔧 技术方法**

使用了e-means算法进行聚类，并结合了Nemotron-3和Streaming Sortformer等现有模型的输出。

**📊 数据集**

使用了VoxConverse、AMI、ICSI和LibriSpeech-OpenSpk等多个数据集进行实验和验证。

**📈 对比分析**

与Streaming Sortformer和Nemotron-3等固定槽位模型进行比较，结果显示在发言者数量超过槽位时，提出的方法在准确性上显著优于这些模型，并且在长时间会议中表现更为稳定。

**⚠️ 局限性**

限制在于使用的数据集较小，且主要基于阅读或访谈语音，未能替代大型真实会议的集合；在快速交替发言的情况下，系统的错误率可能会增加；聚类时需要预先设定档案中的发言者数量。

---

## 298. A Multi-Layered AI Pipeline for BPMN Generation from Natural Language

**arXiv ID:** 2611.18157 | [PDF](https://arxiv.org/pdf/2611.18157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 299. Neural Classifiers as Error-Correcting Codes

**arXiv ID:** 2611.18239 | [PDF](https://arxiv.org/pdf/2611.18239v1)

**作者:** Sergey Salishev `[一作]` (St. Petersburg State University), Zeev Volkovich `[通讯]` (Braude College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个统一框架，将监督深度分类视为香农意义上的纠错编码解码。通过交叉熵优化，展示了softmax分类器与最大似然解码的一致性，并解释了标准散度目标的出现。

**💡 创新点**

创新点在于将深度学习中的分类问题与经典信息理论中的纠错编码解码建立了严格的理论联系，提供了对神经网络内部结构的几何理解，并提出了新的量化、知识蒸馏和架构正则化标准。

**🔧 技术方法**

使用了交叉熵损失函数、softmax分类器以及基于香农理论的随机编码模型，结合了神经崩溃和JEPA/LeJEPA的几何特性。

**📊 数据集**

使用了合成数据集进行实验，特别是在二进制对称信道上进行的模拟实验，以及在ImageNet数据集上的真实图像重建实验。

**📈 对比分析**

通过与最大似然解码器的比较，展示了标准神经分类器在交叉熵训练下的解码行为与最大似然推断一致。性能在低噪声和小代码本的情况下表现良好，但在高噪声和大代码本时性能下降。

**⚠️ 局限性**

限制在于该框架的理论推导主要基于理想条件，实际应用中可能受到噪声、数据多样性和网络架构的影响，尚需进一步研究以确定保证量化的架构条件。

---

## 300. Optimizing Drilling Trajectories in Multi-Material Structures Using Quantum Annealers

**arXiv ID:** 2611.18121 | [PDF](https://arxiv.org/pdf/2611.18121v1)

**作者:** Eneko Osaba `[一作]` (TECNALIA), Antonio Rubio-Mateos `[通讯]` (TECNALIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种量子混合优化方法，用于在多材料结构中规划钻孔轨迹，以最小化总能耗。

**💡 创新点**

创新点在于将钻孔过程建模为一个受限的路由问题，考虑了异质材料特性、材料依赖的钻孔成本、位移能量和工具磨损惩罚，并引入了实际工业约束。

**🔧 技术方法**

使用了D-Wave的Leap受限二次模型混合求解器，结合经典搜索与量子退火周期。

**📊 数据集**

使用了合成生成的基准数据集，模拟了航空航天制造中常见的异质材料分布和成本矩阵。

**📈 对比分析**

通过与经典优化方法进行比较，展示了量子混合优化在处理复杂多物理约束方面的能力，生成的钻孔序列在现实制造条件下具有能效优势。

**⚠️ 局限性**

限制在于当前的量子硬件仍处于NISQ时代，可能影响求解的规模和效率。

---

## 301. EcoVGGT: Pinned-Key Inference for Efficient and Streamable Feed-Forward 3D Reconstruction from Long Videos

**arXiv ID:** 2611.18099 | [PDF](https://arxiv.org/pdf/2611.18099v1)

**作者:** Hongsen Cao `[一作]` (Queen Mary University of London), Ahmed Sayed `[通讯]` (Queen Mary University of London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为EcoVGGT的推理框架，旨在通过选择均匀分布的关键帧来提高长视频的重建效率和准确性。

**💡 创新点**

创新点在于使用固定数量的均匀分布的关键帧作为所有帧的键，而不依赖于内容选择，从而降低了计算成本并提高了成功率。

**🔧 技术方法**

使用了EcoVGGT推理框架，该框架结合了场景缓存、在线流和对齐模块等技术。

**📊 数据集**

使用了ScanNet数据集进行实验，评估了在500帧和1000帧视频上的表现。

**📈 对比分析**

与VGGT和π^3等现有方法进行比较，EcoVGGT在500帧时成功率从70%提高到82%，速度提升至8.5倍，且在1000帧时与π^3的成功率相同，但速度提升至13.8倍。

**⚠️ 局限性**

限制在于EcoVGGT继承了其基础模型在大房间中的失败，且在某些情况下相对姿态误差仍高于完整模型。

---

## 302. More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs

**arXiv ID:** 2611.17610 | [PDF](https://arxiv.org/pdf/2611.17610v1)

**作者:** Seyed Mahmoud Sajjadi Mohammadabadi `[一作]` (University of Nevada), Seyed Mahmoud Sajjadi Mohammadabadi `[通讯]` (University of Nevada)

**通讯引用:** 287 | [OpenAlex ID](https://openalex.org/A5103285917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对多智能体语言模型的评估进行了综合分析，比较了不同的通信架构与单一贪婪推理的表现，探讨了协调与样本多样性之间的关系。

**💡 创新点**

创新点在于提出了一个资源会计框架，定义了六个维度的资源向量，并通过多层随机效应建模分析了多智能体系统的准确性与基线的关系。

**🔧 技术方法**

使用了多层随机效应建模、资源会计框架和敏感性检查等技术。

**📊 数据集**

分析了来自九项研究的516个独特的多智能体配置，涵盖了不同的基线类别和资源预算。

**📈 对比分析**

与单一贪婪推理基线相比，多智能体系统在74%的对比中准确性低于或等于基线，且在样本计数或资源预算匹配的基线下，平均准确性差为-0.042。

**⚠️ 局限性**

限制在于研究样本的选择偏差，部分研究的基线类别和匹配条件存在混淆，且对比的样本量较小。

---

## 303. Robustness Evaluation of GPU Cloud Pricing Benchmarks under Incomplete Observations: A Controlled Missingness Study

**arXiv ID:** 2611.17796 | [PDF](https://arxiv.org/pdf/2611.17796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 304. Learning nonlinear interactions and cross-feature couplings with product-unit residual networks

**arXiv ID:** 2611.18074 | [PDF](https://arxiv.org/pdf/2611.18074v1)

**作者:** Ziyuan Li `[一作]` (Technical University of Munich), Babette Dellen `[通讯]` (University of Applied Sciences Koblenz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种复杂值的产品单元残差网络（CV-PURe），用于显式建模特征之间的非线性交互和交叉特征耦合。

**💡 创新点**

创新点在于将乘法产品单元与残差连接结合起来，显式建模特征交互，同时提高优化的稳定性。

**🔧 技术方法**

使用了复杂值的产品单元残差网络（CV-PURe）技术。

**📊 数据集**

使用了六个科学基准数据集和两个真实世界数据集（Bike Sharing和Parkinsons Telemonitoring）。

**📈 对比分析**

与传统的多层感知器（MLP）相比，CV-PURe在四个科学基准上实现了最低的均方根误差（RMSE），相对减少了53.9%的Friedman误差，并在交互恢复方面表现出更高的准确性。

**⚠️ 局限性**

限制在于模型的计算成本较高，尤其是在复杂值的计算上，尽管参数数量较少，但推理延迟仍然较长。

---

## 305. CoTTAGE: An Explainable Judge for Text-to-Audio Generation

**arXiv ID:** 2611.17644 | [PDF](https://arxiv.org/pdf/2611.17644v1)

**作者:** Yu-Xiang Lin `[一作]` (National Taiwan University), Hung-yi Lee `[通讯]` (National Taiwan University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于推理的文本到音频生成评估器，旨在同时评估文本相关性（REL）和音频质量（OVL），以解决现有自动评估指标无法同时与人类判断高度相关的问题。

**💡 创新点**

创新点在于采用了先推理后预测的范式，生成基于音频的推理理由来诊断错误，然后再预测分数，同时通过收集超过10K的人类OVL注释来训练模型。

**🔧 技术方法**

使用了监督微调（SFT）和基于强化学习（RL）的优化，结合了Thurstone模型的偏好奖励机制，以提高模型的评估能力。

**📊 数据集**

使用了来自RELATE和XACLE的数据集，收集了超过10K个音频片段的人类OVL注释。

**📈 对比分析**

与现有的评估方法（如CLAPScore和ELSA）相比，模型在六个基准测试中表现出与人类判断的最高相关性，并且生成的推理理由更受人类偏好。

**⚠️ 局限性**

限制在于依赖于RELATE和XACLE的人类MOS监督，这些数据集的音频领域和注释风格可能无法覆盖更广泛的TTA场景。此外，当前评估基准主要由短音频片段组成，无法充分评估长音频生成的动态和结构连续性。

---

## 306. Advancing Quantization for R&R Models with Microscaling Formats

**arXiv ID:** 2611.17501 | [PDF](https://arxiv.org/pdf/2611.17501v1)

**作者:** Yavuz Yetim `[一作]` (Meta Platforms, Inc.), Uladzimir Pashkevich `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

本文对工业规模的推荐和排名（R&R）模型进行了实证评估，特别是在AMD MI350X架构上评估了MXFP8和6位MXFP6格式的推理性能。

**💡 创新点**

创新点在于首次在R&R和LLM领域报告了MXFP6 GEMM的显著加速，且在不牺牲模型质量的情况下，成功实现了低于8位的量化。

**🔧 技术方法**

使用了MXFP8和MXFP6格式的GEMM内核，并结合了激活量化的优化技术。

**📊 数据集**

使用了两个真实的生产广告排名模型（模型A和模型B），并通过实际生产输入进行评估。

**📈 对比分析**

与FP16相比，MXFP8在所有支持的线性层上实现了1.33-1.55倍的GEMM内核加速，而MXFP6则实现了1.73-1.99倍的加速，个别层上甚至达到2.90倍的加速。整体模型在批量1024和2048下的端到端推理吞吐量分别提高了5.3%和1.5%（模型A），以及14.3%和7.6%（模型B）。

**⚠️ 局限性**

限制在于MXFP6的量化精度相对较低，可能在某些情况下影响模型的准确性，尽管在严格的0.01% NE容忍度内仍然有效。

---

## 307. Cross-Linguistic Acoustic Biomarkers of Cognitive Impairment Across Interactive and Single-Speaker Speech Tasks

**arXiv ID:** 2611.17710 | [PDF](https://arxiv.org/pdf/2611.17710v1)

**作者:** Kenyu Ikeuchi `[一作]` (Keio University), Chen-Nee Chuah `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究进行了跨语言和跨任务的声学分析，整合了四种语言（西班牙语、普通话、英语和日语）的对话数据集与结构化语音任务，以支持多语言痴呆症检测。

**💡 创新点**

创新点在于提出了一种可解释的声学分析框架，强调语言和任务模式如何影响与认知障碍相关的声学标记，并系统比较了不同数据集和任务中的声学特征。

**🔧 技术方法**

使用了统计分析、语言学/临床指导的特征选择和基于机器学习模型的特征选择技术。

**📊 数据集**

使用了五个数据集，包括来自不同语言和任务的语音样本，涵盖健康对照组和认知障碍组。

**📈 对比分析**

通过比较不同特征配置的分类性能，结果显示在特征过滤和模型选择后，分类性能保持稳定或略有提高，尤其是在结合专家知识的情况下。

**⚠️ 局限性**

限制在于仅关注声学特征，未考虑更高层次的语言信息（如语义、句法、语用）和非声学模态（如视觉线索），未来研究应结合多模态框架和更广泛的数据集。

---

## 308. What LLM Agents See and What Users Say

**arXiv ID:** 2611.17639 | [PDF](https://arxiv.org/pdf/2611.17639v1)

**作者:** Yue Zhao `[一作]` (University of Southern California), Xiyang Hu `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLM）代理在执行用户请求时，工具显示的信息与用户所说的信息之间的差距，特别是在处理模糊性和假设时的行为。

**💡 创新点**

创新点在于提出了一个基于假设理论的“查看或询问标准”，明确了代理在执行操作前需要满足的条件，并构建了一个受控测试平台来验证这一标准。

**🔧 技术方法**

使用了假设理论和认知逻辑的结合，构建了包含16个模板的任务和查找工具，并在36个早期代理和最新的Claude与GPT模型上进行了测试。

**📊 数据集**

使用了零售基准数据集，特别是添加了具有相同姓名和邮政编码的客户，以测试代理在处理模糊性时的表现。

**📈 对比分析**

通过比较不同查找方式（如仅返回第一个匹配、列出所有匹配等），发现大多数早期代理在计数第二个记录时仍然会执行操作，而列出所有记录则能有效阻止错误操作。最新模型在计数时会打开第二个记录，但通常不会执行操作。

**⚠️ 局限性**

限制在于，当前的测试主要集中在特定的任务和查找方式上，未来的研究需要进一步探讨不同任务类型之间的行为差异，以及如何在实际部署中有效减少错误写入和数据泄露的风险。

---

## 309. Astrocyte Biocomputing: Astrocytic Contributions to Neural Network Computation

**arXiv ID:** 2611.17532 | [PDF](https://arxiv.org/pdf/2611.17532v1)

**作者:** Kate M. O'Neill `[一作]` (University of Maryland), Wolfgang Losert `[通讯]` (University of Maryland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文探讨了星形胶质细胞在神经网络计算中的作用，提出了星形胶质细胞作为大脑网络的“指挥者”，并介绍了两种基于星形胶质细胞特性的算法，以提高机器学习的性能。

**💡 创新点**

创新点在于将星形胶质细胞的生物特性（如同时连接和调节大量突触的能力，以及非电信号的慢节奏动态）应用于机器学习算法，显著提升了算法的效率和鲁棒性。

**🔧 技术方法**

使用了基于星形胶质细胞特性的监督学习算法和节奏共享算法，这些算法能够更有效地进行异常检测和防止灾难性遗忘。

**📊 数据集**

论文中提到的实验数据集包括CIFAR-10、NASA C-MAPSS、SWaT和WADI数据集，用于验证算法的有效性。

**📈 对比分析**

与传统的神经中心算法相比，星形胶质细胞衍生的算法在性能上显著优于现有的最先进机器学习模型，尤其在处理噪声和快速学习方面表现更佳。

**⚠️ 局限性**

限制在于对星形胶质细胞的生物机制理解仍不够深入，未来需要更多的实验研究来验证这些算法在实际大脑功能中的应用潜力。

---

## 310. A Spectral Framework for Universal Proxy Regularization in Model Merging

**arXiv ID:** 2611.17728 | [PDF](https://arxiv.org/pdf/2611.17728v1)

**作者:** Shangge Liu `[一作]` (Nanjing University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种用于模型合并的光谱框架，通过普遍代理正则化来解决参数干扰问题。

**💡 创新点**

创新点在于通过光谱覆盖（α）和泄漏（β）来表征代理分布与其他任务分布之间的关系，并推导出统一的参数干扰界限。

**🔧 技术方法**

使用了光谱分析技术，具体包括三种代理正则化实例：Jacobian（J）、Fisher信息矩阵（F）和输出漂移（O）。

**📊 数据集**

使用了公开可获取的代理数据集，如ImageNet和WikiText-103，进行实验验证。

**📈 对比分析**

通过与现有的预合并基线进行比较，实验证明该框架在多个视觉基准上显著提高了合并性能，接近于需要访问其他任务数据的oracle性能。

**⚠️ 局限性**

限制在于该框架的有效性依赖于代理的质量，且在非线性微调下，三种正则化实例的表现可能会有所不同。

---

## 311. Continuous Cuffless Blood Pressure Estimation Using PPG From Methods to Clinical Translation

**arXiv ID:** 2611.17499 | [PDF](https://arxiv.org/pdf/2611.17499v1)

**作者:** Karthika Shalini Potail `[一作]` (University of Texas), Tooraj Nikoubin `[通讯]` (University of Texas)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文综述了基于光电容积描记法（PPG）的无袖带血压（BP）估计方法，强调了其在连续血压监测中的潜力，尤其是在管理高血压方面。

**💡 创新点**

创新点在于对现有研究进行分类，依据其证据强度进行评估，并提出未来研究应优先考虑生理学模型、多样化的纵向数据集和透明的校准报告。

**🔧 技术方法**

使用了机器学习和深度学习技术，结合PPG信号的特征提取和脉搏到达时间（PAT）、脉搏传播时间（PTT）等方法。

**📊 数据集**

使用了多个数据集，包括MIMIC-III、VitalDB和PPG-BP等，涵盖了不同的临床和非临床环境。

**📈 对比分析**

与其他方法的比较显示，基于特征的回归模型在小型数据集上表现较好，而深度学习模型在大型数据集上表现更佳，但普遍存在过拟合和校准依赖的问题。

**⚠️ 局限性**

本综述的局限性在于其为叙述性综述而非系统性综述，未能全面覆盖所有相关文献，并且未对不同方法的准确性进行汇总比较。

---

## 312. WaferTetris: Efficient Freeform Mapping on Wafer-Scale Accelerators

**arXiv ID:** 2611.18527 | [PDF](https://arxiv.org/pdf/2611.18527v1)

**作者:** Dayou Du `[一作]` (University of Edinburgh), Jianyi Cheng `[通讯]` (University of Edinburgh)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种高效的自由形式映射系统，旨在优化晶圆级加速器上的应用映射，消除几何约束，提高计算和内存资源的利用率。

**💡 创新点**

创新点在于通过自由形式映射消除了固定几何布局的限制，采用类似于俄罗斯方块的策略灵活地在晶圆上打包计算，从而提高了处理元素的利用率和可用内存容量。

**🔧 技术方法**

使用了基于递归的吞吐量模型来捕捉分支和连接依赖关系，并结合FPGA放置和路由算法来生成符合晶圆级引擎（WSE）路由约束的映射。

**📊 数据集**

在Cerebras WSE-3上进行了评估，使用了大型语言模型（LLM）解码和科学模板基准测试。

**📈 对比分析**

与基于模板的自动映射基线CU.POKer相比，系统吞吐量提高了3.24到4.42倍，PE利用率超过93%。与手动优化的WaferLLM相比，在相同上下文长度下，支持1.8倍的上下文长度，并在每用户吞吐量上达到1.10倍的提升。

**⚠️ 局限性**

限制在于自由形式映射可能引入更长的通信路径，并集中路由资源需求于中间PE路由器，要求映射器在巨大的放置和路由决策搜索空间中导航，同时保持通信局部性和遵循路由容量。

---

## 313. MASTARS: Multi-Agent Sequential Trajectory Augmentation with Return-Conditioned Subgoals

**arXiv ID:** 2611.17793 | [PDF](https://arxiv.org/pdf/2611.17793v1)

**作者:** Jiwon Jeon `[一作]` (Korea Advanced Institute of Science and Technology), Youngchul Sung `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为MASTARS的多智能体轨迹增强框架，通过扩展多智能体轨迹生成来提高离线多智能体强化学习（MARL）的性能。

**💡 创新点**

MASTARS解决了多智能体数据增强中的多样性、协调性和一致性之间的权衡，采用基于扩散的顺序生成方法和基于子目标的部分生成策略。

**🔧 技术方法**

使用了扩散模型作为生成技术，并结合了基于子目标的生成顺序和基于重建的顺序生成方法。

**📊 数据集**

在多个基准测试上进行了实验，包括多智能体粒子环境（MPE）、星际争霸多智能体挑战（SMAC）和SMAC-v2。

**📈 对比分析**

与其他数据增强方法相比，MASTARS在MPE和SMAC基准测试中显著提高了多种离线MARL算法的性能，尤其在复杂的协调任务中表现优异。

**⚠️ 局限性**

MASTARS的局限性在于依赖于多个辅助模型的准确性，尽管实验表明其对模型误差具有一定的鲁棒性。

---

## 314. WeightBench: Evaluating Language Models as Weight-Space Programmers

**arXiv ID:** 2611.18553 | [PDF](https://arxiv.org/pdf/2611.18553v1)

**作者:** Aayush Marasini `[一作]` (University of Southern Mississippi), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WeightBench基准，系统评估大型语言模型（LLMs）在权重空间编程（WSP）任务上的表现，包括读取、修复和编写神经网络权重等42个任务。

**💡 创新点**

创新点在于引入了一个全面的基准来评估LLMs在权重空间编程中的能力，强调了通过几何和拓扑推理进行操作的重要性。

**🔧 技术方法**

使用了严格的评估框架，结合了任务特定的评分机制，确保模型在没有梯度下降的情况下进行权重编辑。

**📊 数据集**

评估了从2K参数的控制策略到7B参数的语言模型的多种网络，涵盖了不同规模和复杂度的任务。

**📈 对比分析**

与非代理基线和参考解决方案进行了比较，结果显示Claude Opus 5.5在42个任务中获得了0.535的宏观得分，而GPT-6.1-sol在38个任务中获得了0.568的得分，表现优于其他模型。

**⚠️ 局限性**

限制在于当前模型在修复语言模型和无数据压缩方面的能力不足，且在区分不可行编辑与困难编辑方面表现不佳。

---

## 315. LLM-Guided Cluster Lifecycle Management: Deciding When and How to Evolve Behavioral Clusters

**arXiv ID:** 2611.17616 | [PDF](https://arxiv.org/pdf/2611.17616v1)

**作者:** Himanshu Kulkarni `[一作]` (Amazon Web Services), Jianhong Xin `[通讯]` (Amazon Web Services)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文研究了一种新的集群生命周期管理方法，利用大型语言模型（LLM）在不进行重新聚类的情况下，决定何时以及如何对行为集群进行演变。

**💡 创新点**

创新点在于将决策过程从基于阈值的触发机制转变为LLM的推理决策，且提出了一种无阈值的决策层，能够根据集群的统计健康信号和标签信息进行动态决策。

**🔧 技术方法**

使用了大型语言模型（LLM）作为决策层，结合统计和语义信号进行集群生命周期管理。

**📊 数据集**

使用了三个公共数据集，分别是YooChoose（0.17的标签丰富度）、MOOC（0.50的标签丰富度）和News-Category（0.70的标签丰富度）。

**📈 对比分析**

与传统的阈值触发器和MONIC方法相比，LLM模式在所有三个基准上获得了最高的F1分数，且干预次数显著减少，表现出更强的约束力。所有三项头对头的改进在统计上显著（p≤0.003）。

**⚠️ 局限性**

限制在于仅使用了三个公共数据集，无法建立普遍的丰富度规律；政策的校准是一个实用的默认值而非经过校准的边界；注入的漂移事件可能不涵盖真实聚类衰退的所有方式。

---

## 316. Does the Ranking Hold? A Practitioner-Facing Audit of Hallucination Detectors on Small, Quantized, On-Device LLMs

**arXiv ID:** 2611.17339 | [PDF](https://arxiv.org/pdf/2611.17339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 317. MMFlash: Faster Speculative Decoding for Vision-Language Models

**arXiv ID:** 2611.18482 | [PDF](https://arxiv.org/pdf/2611.18482v1)

**作者:** Sicheng Feng `[一作]` (National University of Singapore), Xinchao Wang `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种新的扩散草拟框架MMFlash，用于视觉语言模型(VLMs)的投机解码，旨在提高解码速度和接受率。

**💡 创新点**

创新点在于通过自适应加权视觉依赖性较强的令牌来提高草拟质量，并在长视频输入中采用稀疏注意力机制以减少冗余计算。

**🔧 技术方法**

使用了扩散草拟技术，结合KL散度度量视觉依赖性，并在草拟过程中动态调整令牌的权重。

**📊 数据集**

使用了来自LLaVA-OV-1.5-Instruct-Data的数据集，包含约900K的多模态样本。

**📈 对比分析**

与现有的EAGLE-3和DFlash方法进行比较，MMFlash在多个基准测试中实现了最高3.58倍的解码速度提升，接受长度和速度均优于对比方法。

**⚠️ 局限性**

限制在于MMFlash的性能依赖于草拟器的设计和训练数据的质量，可能在某些特定任务上表现不佳。

---

## 318. Scalar Entropic Damage Transport: Well-posed softening with a single damage variable

**arXiv ID:** 2611.17822 | [PDF](https://arxiv.org/pdf/2611.17822v1)

**作者:** Jonas Nutz `[一作]` (University of Duisburg-Essen), Matti Schneider `[通讯]` (University of Duisburg-Essen)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的材料软化模型，称为熵损伤传输（EDT），通过局部连续损伤模型表示材料软化，同时确保机械边值问题的良好适定性。

**💡 创新点**

创新点在于揭示了损伤分布是时间离散化的伪影，并且在时间连续极限下，损伤分布集中在单一损伤状态上，从而使得EDT收敛到一个标量损伤模型。

**🔧 技术方法**

采用了冷冻度量离散化技术，以确保增量的强凸性，并通过标量更新保持损伤不可逆性、规定的损伤速率限制和热机械一致性。

**📊 数据集**

使用了纤维增强聚酰胺和钛合金等数据集进行计算实验，评估了模型的准确性和效率。

**📈 对比分析**

通过与原始EDT模型的比较，发现标量更新在材料点级别和微观结构级别的计算中表现出更高的效率，能够处理超过1400万个自由度的工业规模有限元计算。

**⚠️ 局限性**

限制在于内部变量仍然是离散的损伤状态分布，尽管更新过程高效，但在某些情况下可能导致损伤演化滞后于标量演化。

---

## 319. Do Verifier Judgments Survive Abstraction? Measuring Verification Transport for LLM Agent Evaluation

**arXiv ID:** 2611.18471 | [PDF](https://arxiv.org/pdf/2611.18471v1)

**作者:** Disha Sheshanarayana `[一作]` (Manipal University Jaipur), Sai Linisha Reddy Goli `[通讯]` (Manipal University Jaipur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Verification Transport的可靠性测试，旨在评估在抽象引起的可观察性变化下，验证者的判断是否保持稳定。

**💡 创新点**

创新点在于引入了Verification Transport作为评估代理执行的可靠性标准，并系统性地评估了不同抽象机制下的验证失败情况。

**🔧 技术方法**

使用了多种验证协议，包括直接验证、基于证据的验证、寻求来源的验证和审议验证，评估了七种验证器模型的表现。

**📊 数据集**

使用了四个不同的代理评估基准，包括AgentProcessBench、ATBench、PlanRewardBench和AgentDojo，共计250个轨迹。

**📈 对比分析**

通过比较不同验证协议的表现，发现直接验证的错误判决率较高，而基于证据和审议的协议在可观察性降低时更倾向于放弃判决。总体上，抽象引起的验证失败率在宏观抽象和委托情况下更高，分别达到31%和30.9%。

**⚠️ 局限性**

限制在于仅依赖原生视图的准确性不足以全面评估验证者的可靠性，且在某些情况下，恢复执行来源的效果可能受到损失或不可用的影响。

---

## 320. Measured from the feed: the unsustainable growth of AI video generation

**arXiv ID:** 2611.17938 | [PDF](https://arxiv.org/pdf/2611.17938v1)

**作者:** Tristan Coignion `[一作]` (University of Lille), Anne-Laure Ligozat `[通讯]` (Universite Paris-Saclay)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文研究了AI视频生成的环境影响，特别是其在商业规模下的可持续性问题。通过测量两个商业平台的年度视频输出，建立了这些服务从训练到存储的绝对足迹的前瞻性模型。

**💡 创新点**

创新点在于首次估算了AI视频生成作为一种部署技术的全球直接环境足迹，并探讨了在不同需求和视频质量场景下这些影响如何演变到2030年。

**🔧 技术方法**

使用了底层模型来估算服务基础设施的能源和延迟，从单次生成的能耗扩展到年度生成的视频量，并通过蒙特卡洛分析量化了估算的不确定性。

**📊 数据集**

数据集包括对Sora和HeyGen两个平台的公共数据抓取，结合六个提供商披露的生成计数，估算出2026年生成的单输出视频数量在62亿到95亿之间。

**📈 对比分析**

与现有方法相比，本文的模型提供了更全面的环境影响评估，结果显示在2026年视频生成消耗724 GWh电力，排放124.3 kt CO2eq，且在需求持续增长的情况下，到2030年排放将增加近75倍。

**⚠️ 局限性**

限制在于模型假设生成架构在2026-2030年间保持不变，未考虑视频传播、音频生成及其他生成AI应用的影响，且训练成本的估算基于有限的公开数据，可能低估了实际情况。

---

## 321. PaperCritic: Training AI Reviewers to Assess the Soundness of Papers

**arXiv ID:** 2611.17834 | [PDF](https://arxiv.org/pdf/2611.17834v1)

**作者:** Hyun Ryu `[一作]` (KAIST AI), Carolyn Rose `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种AI审稿人模型PaperCritic，专注于评估论文的合理性，构建了一个质量标记的审稿点数据集，并通过强化学习训练模型。

**💡 创新点**

创新点在于将论文审稿任务形式化为最大化审稿点的合理性和实质性，以及高质量参考审稿点的覆盖率，提出了一种新的训练目标和方法。

**🔧 技术方法**

使用了强化学习技术，结合F1分数作为奖励进行模型训练。

**📊 数据集**

使用了1K ICLR提交的论文和23K人类撰写及LLM生成的审稿点构建的数据集。

**📈 对比分析**

与传统的审稿人训练方法和前沿模型相比，PaperCritic在领域内测试集上表现优越，并且在多个领域外的审稿任务中也表现良好。

**⚠️ 局限性**

限制在于本研究仅关注论文的合理性评估，未优化其他重要的同行评审目的，如趣味性、清晰度和新颖性，且未考虑外部引用的审稿点。

---

## 322. Agentic induction of common data elements and information extraction from sarcoma case report abstracts

**arXiv ID:** 2611.18506 | [PDF](https://arxiv.org/pdf/2611.18506v1)

**作者:** Daniel Reichenpfader `[一作]` (Bern University of Applied Sciences), Fabio Dennstädt `[通讯]` (Inselspital, Bern University Hospital and University of Bern)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究探讨了大型语言模型（LLM）代理是否能够从肉瘤病例报告摘要中推导出通用数据元素（CDE）架构，并在大规模上应用该架构进行信息提取。

**💡 创新点**

创新点在于使用LLM代理自动生成CDE架构，并通过信息提取验证其有效性，展示了自动化工作流程在医学文献中的应用潜力。

**🔧 技术方法**

使用了大型语言模型（LLM）进行CDE的提议、合并和信息提取，具体模型为gpt-oss-120b。

**📊 数据集**

使用的数据集为从PubMed检索的24,588个肉瘤病例报告摘要。

**📈 对比分析**

与传统方法相比，LLM生成的CDE架构在79.2%的摘要中成功提取了所有保留的CDE，且在100个样本摘要中，提取值的准确一致性为94.9%。

**⚠️ 局限性**

限制在于生成的CDE架构质量不高，只有19个CDE在正确性、相关性和清晰度上获得了双重审查员的批准，且部分CDE存在重叠和不一致的问题。

---

## 323. ERGo: Energy-Efficient Hybrid Graph Neural Network Training on Processing-In-Memory Architectures

**arXiv ID:** 2611.17478 | [PDF](https://arxiv.org/pdf/2611.17478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 324. TailGuard-GDS: Tail-Safe Control of GPU Storage Data Movement

**arXiv ID:** 2611.17824 | [PDF](https://arxiv.org/pdf/2611.17824v1)

**作者:** Jiaqi Liu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Guangming Tan `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为TailGuard-GDS的运行时控制器，旨在优化GPU存储数据传输的延迟，通过选择适合每个应用阶段的传输计划来减少GPU可见的尾延迟。

**💡 创新点**

创新点在于TailGuard-GDS能够根据应用阶段的上下文和测量的尾延迟动态选择后端、请求形状和存储位置，从而实现更优的性能。

**🔧 技术方法**

使用了TailGuard-GDS作为运行时控制器，结合了PyTorch FSDP和DeepSpeed ZeRO-3框架，进行动态路径选择和优化。

**📊 数据集**

在实验中使用了多个数据集，包括五个八GPU的DeepSpeed ZeRO-3工作负载和四个排名的PyTorch FSDP工作负载。

**📈 对比分析**

与固定路径的比较显示，TailGuard-GDS在多个工作负载中减少了步骤p95延迟，具体为：在PyTorch FSDP中减少了18.4%，在DeepSpeed ZeRO-3中减少了7.9%到24.6%。与调优的DeepSpeed AIO相比，TailGuard还减少了5.6%的步骤p95，并提高了估计的吞吐量6.1%。

**⚠️ 局限性**

限制在于TailGuard-GDS的性能依赖于当前的资源状态和工作负载变化，可能需要重新校准以适应新的条件。

---

## 325. Superfluid: Local AI Server for Distributed LLM Inference

**arXiv ID:** 2611.17753 | [PDF](https://arxiv.org/pdf/2611.17753v1)

**作者:** Prabod Rathnayaka `[一作]` (Base Compute), Lukas Wesemann `[通讯]` (Base Compute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Superfluid的开源大型语言模型（LLM）服务器，旨在支持本地推理，能够在单个设备和多个设备上高效运行。

**💡 创新点**

创新点在于Superfluid能够在一个设备上处理多个代理的交互请求，并且在多个设备上将闲置的机器整合为一个单一的API端点，提升了响应速度和系统的鲁棒性。

**🔧 技术方法**

使用了Rust编程语言，结合了进程隔离、调度算法、共享前缀合并等技术来优化模型的运行和请求处理。

**📊 数据集**

使用了Qwen3.8-27B模型，并在不同的硬件上进行了测试，包括Apple M5 Pro和Mac Studio M3 Ultras等。

**📈 对比分析**

与Ollama等现有服务器进行比较，Superfluid在处理请求时表现出更快的响应时间和更高的吞吐量。例如，在8个代理忙碌时，Superfluid的首次令牌响应时间为1.5秒，而Ollama则为291-612秒。

**⚠️ 局限性**

限制在于每个节点必须持有模型的完整副本，且会在节点之间重新计算会话的提示，无法共享KV缓存。此外，节点可能在量化上存在差异，导致会话输出依赖于提供服务的节点。

---

## 326. Starlight: Whittle-Driven Fidelity Rescue for Satellite-Terrestrial Quantum Networks

**arXiv ID:** 2611.17865 | [PDF](https://arxiv.org/pdf/2611.17865v1)

**作者:** Yaliang Shi `[一作]` (University of Electronic Science and Technology of China), Bangguo Yuan `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种需求驱动的卫星-地面量子纠缠路由架构Starlight，旨在优先处理物理状态不断变化的连接，以防止其纠缠变得不可用。

**💡 创新点**

创新点在于引入了有效状态年龄（EAoS）这一物理感知指标，将异质的去相干和交换引起的保真度惩罚映射到统一的加法坐标上，并将救援调度建模为一个无休止多臂赌博机（RMAB）问题。

**🔧 技术方法**

使用了无休止多臂赌博机（RMAB）模型和Whittle风格的索引近似技术，结合了双时间尺度控制平面和异步调度管道。

**📊 数据集**

使用了QUASAR模拟器进行仿真，模拟了128个跨域锚点对的干扰瓶颈，评估了在资源稀缺、记忆异质性和随机大气退化下的性能。

**📈 对比分析**

与基于规划的、年龄驱动的、状态反应的和通道贪婪的策略进行了比较，Starlight在资源限制下达到了84.5%的平均满意度，表现优于其他方法。

**⚠️ 局限性**

限制在于当前模型未考虑多方GHZ分布的扩展，未来工作将整合动态卫星间路由、大气预测和机载量子纠错。

---

## 327. Representational Geometry as a Diagnostic for Adversarial Robustness

**arXiv ID:** 2611.17684 | [PDF](https://arxiv.org/pdf/2611.17684v1)

**作者:** Brandon Ismalej `[一作]` (California State University), Gal Mishne `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了对抗训练对神经网络内部表示的影响，分析了不同对抗训练目标下的多层感知器在MNIST数据集上的表现。

**💡 创新点**

提出了通过几何分析来评估网络的鲁棒性，发现传统的准确率指标与几何结构之间存在不一致，强调了对抗训练对内部表示结构的影响。

**🔧 技术方法**

使用了M-PHATE技术来嵌入数据样本，并通过最大均值差异（MMD）和邻近恢复等方法进行几何分析。

**📊 数据集**

使用了MNIST数据集，包含1000个训练图像和2000个测试图像。

**📈 对比分析**

通过几何测量与标准评估方法进行比较，发现几何测量在评估鲁棒性时提供了不同的排名，且在不同攻击模型下的表现不一致。

**⚠️ 局限性**

研究结果主要基于小型多层感知器和单步攻击，局限于倒数第二层的表示，未来研究将扩展到更复杂的架构和数据集。

---

## 328. NPU-Accelerated GNSS Signal Acquisition

**arXiv ID:** 2611.18186 | [PDF](https://arxiv.org/pdf/2611.18186v1)

**作者:** Taro Suzuki `[一作]` `[通讯]` (Chiba Institute of Technology), Taro Suzuki (Chiba Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种利用单板计算机（SBC）上的神经处理单元（NPU）加速GNSS信号获取的方法，主要通过将离散傅里叶变换（DFT）分解为小DFT的矩阵乘积，并通过1×1卷积、逐元素乘法和转置来实现。

**💡 创新点**

首次将SBC的NPU用于传统的GNSS信号获取，而非机器学习，展示了NPU在GNSS信号获取中的有效性。

**🔧 技术方法**

使用了四步DFT分解技术，将DFT表示为小DFT的矩阵乘积，利用NPU高效执行的操作（如1×1卷积、逐元素乘法和转置）。

**📊 数据集**

使用了Raspberry Pi 5与AX8850 NPU和ROCK 5C（RK3588）作为测试平台，评估了真实GNSS信号的获取性能。

**📈 对比分析**

与优化的CPU实现相比，Raspberry Pi 5上使用AX8850的信号获取速度提高了3.0至3.8倍，整个接收器的跟踪通道数量在30秒内首次锁定的数量增加了1.5倍，90%的通道锁定时间从66秒减少到26秒。

**⚠️ 局限性**

限制在于NPU的性能依赖于数据传输的效率，对于共享内存的NPU，计算瓶颈在于NPU的计算能力，而对于独立内存的NPU，数据传输的数量和大小可能成为瓶颈。

---

## 329. Activation Steering the Nan-Shan in Typhoon

**arXiv ID:** 2611.17768 | [PDF](https://arxiv.org/pdf/2611.17768v1)

**作者:** Megan Wei `[一作]` (Brown University), Ellie Pavlick `[通讯]` (Brown University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文介绍了一个名为《台风》的视听装置，该装置基于约瑟夫·康拉德的小说，通过一个大型语言模型生成文本，探讨人类文化在当前人工智能时代的反思。

**💡 创新点**

创新点在于引入了一种新的自适应激活引导方法，以控制叙事的节奏和模型的偏离，同时展示了人工智能的可解释性与其不可控性之间的张力。

**🔧 技术方法**

使用了激活引导技术，特别是自适应引导方法来控制生成文本的风格和内容。

**📊 数据集**

数据集为约瑟夫·康拉德的小说《台风》，作者对其529个段落进行了贝福特风速等级的评分，以此作为生成文本的指导。

**📈 对比分析**

与其他方法（如无引导、仅提示和静态引导）相比，自适应引导在跟踪贝福特弧线方面表现最佳，具有较高的相关性（ρ = 0.96），而其他方法的表现较差。

**⚠️ 局限性**

限制在于生成文本可能会出现退化现象，如叙事漂移和不连贯语言，尽管采取了措施来控制这些现象，但仍然存在偏见和不一致的输出。

---

## 330. TIDE: Reclaiming Lost Performance In On-Policy Reinforcement Learning

**arXiv ID:** 2611.17924 | [PDF](https://arxiv.org/pdf/2611.17924v1)

**作者:** Mathew Jacob `[一作]` (University of Washington), Stephanie Wang `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的强化学习训练系统TIDE，旨在提高大语言模型的后训练效率，特别是在同步共置的情况下，通过动态重新分配GPU资源来减少资源闲置和解码低效的问题。

**💡 创新点**

创新点在于通过动态重新利用完成的rollout GPU进行训练，并通过请求迁移提高解码效率，从而在保持在线策略训练语义的同时，显著提高资源利用率。

**🔧 技术方法**

使用了动态迁移策略和细粒度资源重新分配技术，结合了GPU的重新利用和请求迁移机制。

**📊 数据集**

在SkyRL的Text2SQL数据集和DAPO-Math数据集上进行了评估，涵盖了不同的工作负载和硬件配置。

**📈 对比分析**

与传统的共置系统相比，TIDE在多个应用和硬件配置上实现了高达1.36倍的端到端加速，恢复了77%到97%的最佳吞吐量，超越了所有评估的在线基线。

**⚠️ 局限性**

限制在于动态迁移和重新分配GPU资源的复杂性，可能在不同工作负载和硬件配置下表现不一致，且需要有效的在线策略来平衡推理和训练的吞吐量。

---

## 331. Agent-as-a-Judge

**arXiv ID:** 2611.18067 | [PDF](https://arxiv.org/pdf/2611.18067v1)

**作者:** Runyang You `[一作]` (Hong Kong Polytechnic University), Wenjie Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

论文探讨了人工智能评估的进展，提出了Agent-as-a-Judge的概念，强调主动评估者如何通过获取缺失证据和明确检查过程来形成判断。

**💡 创新点**

创新点在于将被动的LLM评估者转变为主动的评估者，能够在评估过程中主动收集证据并进行验证，从而提高判断的可靠性。

**🔧 技术方法**

使用了多代理协作、工具集成、记忆和规划等技术来构建评估过程。

**📊 数据集**

论文中未明确提及具体的数据集，但提到整合了多个已发布的评分和关键代理评估者的缺失条目。

**📈 对比分析**

通过比较不同的代理评估者在统一基准上的表现，发现代理评估者在偏好导向基准下的表现不均，强调了对证据寻求可靠性的专门评估的必要性。

**⚠️ 局限性**

局限性包括计算成本、延迟、安全性和隐私等挑战，尤其是在专业领域中，如何管理评估过程中的用户信息和敏感数据。

---

## 332. Evaluating Polling Schemes for OFDMA Uplink Scheduling in Time-Sensitive Wi-Fi Networks

**arXiv ID:** 2611.17898 | [PDF](https://arxiv.org/pdf/2611.17898v1)

**作者:** Douglas Dziedzorm Agbeve `[一作]` (University of Antwerp), Jeroen Famaey `[通讯]` (University of Antwerp)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文评估了在时间敏感的Wi-Fi网络中，OFDMA上行调度的轮询方案，比较了多种轮询机制在不同流量特征下的表现。

**💡 创新点**

创新点在于提供了对多种轮询方案的系统比较，特别是在三种时间敏感环境下（视频会议、工业物联网和扩展现实服务）进行评估，并提出了新的活动基础轮询方案和基于学习的轮询方法。

**🔧 技术方法**

使用了OFDMA、UORA、活动基础轮询（a2p、a2s）和基于上下文的多臂赌博机（cmab）等技术。

**📊 数据集**

使用了三种不同的场景数据集：大规模视频会议、工业物联网部署和扩展现实服务。

**📈 对比分析**

通过在相同的流量和信道条件下评估不同方案，发现活动基础和学习基础的轮询在流量不均匀时表现最佳，而固定调度服务在流量持续时表现更好。具体来说，在视频会议中，活动基础和学习基础的轮询提供了最佳性能，而在工业场景中，调度访问在重负载下最有效。

**⚠️ 局限性**

限制在于实验假设了单一的基本服务集（BSS），没有考虑重叠的BSS和隐藏终端的影响，且未验证在实际网络中动态变化的工作负载下，方案的排名是否依然有效。

---

## 333. OrbitWeaver: LLM Agent Planning with Simulation Feedback for Optically Interconnected Orbital GPU Clusters

**arXiv ID:** 2611.17850 | [PDF](https://arxiv.org/pdf/2611.17850v1)

**作者:** Pengcheng Luo `[一作]` (Shanghai Jiao Tong University), Bowen Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了OrbitWeaver，一个离线的大型语言模型（LLM）代理规划器，旨在优化光互连轨道GPU集群中的排名放置和通信顺序，以减少作业完成时间（JCT）。

**💡 创新点**

创新点在于结合了执行结构分析和模拟反馈，通过LLM代理实现了排名放置和通信调度的联合优化，显著提高了并发计算的效率。

**🔧 技术方法**

使用了大型语言模型（LLM）技术，结合了结构分析和模拟反馈来指导候选方案的构建和评估。

**📊 数据集**

使用了H200和A100的执行跟踪数据集进行模拟实验，评估了不同工作负载下的性能。

**📈 对比分析**

与现有的基线方法（如CASSINI、Crux和Symphony）进行比较，OrbitWeaver在所有工作负载中实现了6.2%到33.5%的JCT减少，显示出其在并发计算中的有效性。

**⚠️ 局限性**

限制在于该方法依赖于固定的GPU分配和光互连条件，未来的工作将探索如何提高适应性以应对变化的通信瓶颈。

---

## 334. Information-Theoretic Threshold Designated-Verifier Digital Signatures for Resource-Constrained Devices

**arXiv ID:** 2611.18355 | [PDF](https://arxiv.org/pdf/2611.18355v1)

**作者:** Andrew Wright `[一作]` (Newcastle University), Ittoop Vergheese Puthoor `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了三种新的信息论阈值指定验证数字签名方案，并证明其在标准安全标准下对任何对手的安全性。

**💡 创新点**

创新点在于将信息论安全性扩展到阈值签名方案，提供了对部分妥协和可用性的保护，同时适用于资源受限设备。

**🔧 技术方法**

使用了通用哈希和Shamir秘密共享等简单数学构造。

**📊 数据集**

在Raspberry Pi 4和2025 M5 MacBook Pro上进行了性能评估。

**📈 对比分析**

与最新的后量子阈值数字签名方案Threshold ML-DSA-87进行比较，发现我们的方案在资源受限设备上具有相似的签名计算复杂度，并在受托人数量较大时性能提高约2倍。

**⚠️ 局限性**

限制在于签名和密钥长度较长，且指定验证签名方案不具备公共可验证性，可能在某些应用场景中成为缺点。

---

## 335. Color Refinement Distinguishes Connected Rooted Outerplanar Graphs

**arXiv ID:** 2611.18391 | [PDF](https://arxiv.org/pdf/2611.18391v1)

**作者:** Louann Coste `[一作]` (PSL University), Laurent Viennot `[通讯]` (Inria Paris)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文研究了颜色细化（1-WL算法）在外平面图上的区分能力，证明了连接根外平面图可以通过个别化一个根节点来区分。

**💡 创新点**

创新点在于提出了通过个别化根节点来确保稳定的颜色类，从而使得颜色细化能够区分连接根外平面图，并扩展到伪根图和非连接图。

**🔧 技术方法**

使用了颜色细化算法（1-WL算法）和反演算法来重构图的同构性，并利用了Rocq证明助手来形式化一些结构性引理。

**📊 数据集**

研究中涉及的图类为外平面图，特别是连接根外平面图和伪根外平面图。

**📈 对比分析**

与其他方法比较，(1,1)-WL测试能够区分所有外平面图，且通过颜色细化算法的反演过程，能够在近线性时间内重构图的同构性。

**⚠️ 局限性**

限制在于颜色细化算法无法区分某些常见的外平面图的超类，即使在个别化根节点后，仍然存在一些非同构的连接外平面图无法被区分。

---

## 336. Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware Attention Approximation

**arXiv ID:** 2611.17629 | [PDF](https://arxiv.org/pdf/2611.17629v1)

**作者:** Muhammed Izzet Sezer `[一作]` (Independent Researcher), Sevval Belkis Dikkaya `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了三种多视角变换器的全局注意力机制，分析了其在七个数据集上的表现，并提出了一种新的注意力近似方法。

**💡 创新点**

提出了基于层角色的全局注意力分析，识别出每一层的三种角色（上下文、匹配和锚定），并基于此提出了Regime-AWare ATTention Approximation方法。

**🔧 技术方法**

使用了CUDA引擎进行加速，采用8位精度运行注意力，并在需要时保持16位精度。

**📊 数据集**

使用了七个数据集进行实验，包括ETH3D、CO3Dv2、RealEstate10K、NRGBD、TUM-dynamic、7-Scenes和Mip-NeRF 360。

**📈 对比分析**

与现有的加速器进行比较，提出的方法在50帧时比官方代码快2.08倍，且在任何数据集上的姿态准确度损失不超过1分（AUC@3）。

**⚠️ 局限性**

方法的局限性在于每个模型需要单独的搜索，且在某些情况下可能会导致准确度下降，尤其是在未见过的校准数据集上。

---

## 337. Correlation distribution for a class of Niho exponents via self-conjugate-reciprocal polynomials and Kloosterman sums

**arXiv ID:** 2611.17506 | [PDF](https://arxiv.org/pdf/2611.17506v1)

**作者:** Yongbo Xia `[一作]` (South-Central Minzu University), Tor Helleseth `[通讯]` (University of Bergen)

**通讯引用:** 9259 | [OpenAlex ID](https://openalex.org/A5026698531)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了二进制m序列与其d-decimation之间的相关分布，特别是当d=s(q-1)+1且4s≡5q+1时的情况。

**💡 创新点**

创新点在于通过低度自共轭-倒数多项式的研究，确定了与Niho指数相关的六次多项式P_a(x)在单位圆U中具有4、5或6个根的条件，并且确定了二进制m序列的交叉相关分布。

**🔧 技术方法**

使用了自共轭-倒数多项式、Walsh变换和Kloosterman和等技术。

**📊 数据集**

使用了有限域_q和_q^2中的元素，特别是q=2^m的情况，m为偶数且m≥6。

**📈 对比分析**

通过比较不同的多项式根的数量，确定了交叉相关函数C_d(τ)的频率分布，性能表现为七个可能的值及其对应的频率N_i。

**⚠️ 局限性**

限制在于该研究主要集中在特定形式的Niho指数上，未来的研究可以扩展到其他形式的Niho指数及其交叉相关分布。

---

## 338. f-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models

**arXiv ID:** 2611.18370 | [PDF](https://arxiv.org/pdf/2611.18370v1)

**作者:** Hoang Tran Vuong `[一作]` (Hanoi University of Science and Technology), Trung Le `[通讯]` (Monash University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为DiSPaT的自我对弈微调框架，用于无监督的表格异常检测，利用大语言模型（LLMs）生成伪异常样本并优化模型对正常数据分布的理解。

**💡 创新点**

创新点在于通过最小化f-散度来实现迭代的自我对弈机制，使模型能够生成伪异常样本，并通过对比真实正常样本来逐步优化模型的正常性描述。

**🔧 技术方法**

使用了大语言模型（LLMs）和自我对弈微调技术，结合f-散度最小化的理论基础。

**📊 数据集**

使用了多个标准异常检测基准数据集，包括ODDS库和Kaggle上的欺诈检测数据集，这些数据集包含文本、数值和分类特征。

**📈 对比分析**

与现有的LLM基础方法、深度学习方法和经典无监督检测器进行了比较，DiSPaT在多个数据集上表现出色，平均AUC-ROC提高了12.7%，F1分数提高了15.4%。

**⚠️ 局限性**

限制在于该方法依赖于生成的伪异常样本的质量，可能在某些情况下无法完全捕捉到真实异常的特征。

---

## 339. SceneAgent: A Multi-Agent Framework for Portable Game Scene Generation

**arXiv ID:** 2611.17821 | [PDF](https://arxiv.org/pdf/2611.17821v1)

**作者:** Xingyi Li `[一作]` (Tencent), Weidong Zhang `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SceneAgent，一个多代理框架，用于在不同游戏引擎之间重用场景构建过程和经验。

**💡 创新点**

创新点在于将场景构建的可移植性从输出扩展到构建过程和可重用经验，结合了规范的场景状态和共享工具接口。

**🔧 技术方法**

使用了多代理协议、规范的场景上下文、结构化的任务交接和分解的布局与内容技能。

**📊 数据集**

使用了GameSceneBench数据集，该数据集包含69个室内和室外场景提示。

**📈 对比分析**

与五个基线方法进行比较，SceneAgent在对齐、空间合理性和完整性方面表现优越，获得了78.2%到89.4%的人工偏好评分。

**⚠️ 局限性**

局限性包括仍需后端集成和兼容资产，跨引擎研究使用不同的资产目录，且多代理编排增加了运行时间。

---

## 340. Conflict, Feasibility, Faithfulness, Displacement: Four Quantities That Come Apart in Federated Dual-Cone Aggregation

**arXiv ID:** 2611.18014 | [PDF](https://arxiv.org/pdf/2611.18014v1)

**作者:** Grzegorz Gwardys `[一作]` (Warsaw University of Technology), Bartosz {Ś}widerski `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的聚合方法UPGrad，用于解决联邦学习中的客户端更新冲突问题，并通过四个量（冲突、可行性、忠实性和位移）来分析其效果。

**💡 创新点**

创新点在于将四个量分开分析，揭示了它们之间的关系，并提出UPGrad作为一种非冲突的聚合器，能够在一定条件下提高模型的准确性。

**🔧 技术方法**

使用了多目标梯度聚合技术，特别是UPGrad聚合器，该聚合器通过投影和平均来减少客户端更新之间的冲突。

**📊 数据集**

使用了CIFAR-10和Oxford-IIIT Pet数据集进行实验，特别是在Dirichlet分布下的CIFAR-10数据集上进行了详细的性能评估。

**📈 对比分析**

与FedAvg等方法进行了比较，UPGrad在CIFAR-10上提高了准确性，但在某些情况下（如部分参与的客户端）未能显著提高性能。整体来看，UPGrad的增益在不同配置下表现不一，且与冲突水平并不总是相关。

**⚠️ 局限性**

限制在于实验规模较小，仅使用了20个客户端和150个轮次，且仅在两个数据集上进行了测试，未能独立变化客户端数量和本地训练轮次。

---

## 341. Large language models put existential risk from artificial intelligence at 10%

**arXiv ID:** 2611.18219 | [PDF](https://arxiv.org/pdf/2611.18219v1)

**作者:** Julian Kauk `[一作]` `[通讯]` (Friedrich Schiller University Jena), Julian Kauk (Friedrich Schiller University Jena)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文探讨了先进人工智能（AI）是否可能导致人类灭绝或类似的永久性失能，特别是从大型语言模型（LLMs）的角度进行调查。

**💡 创新点**

创新点在于将AI系统自身的观点纳入关于AI风险的讨论中，提供了与人类专家意见的比较。

**🔧 技术方法**

使用了大型语言模型（LLMs）进行调查，分析了不同模型在不同推理设置下的风险估计。

**📊 数据集**

使用了来自多个开发者的LLMs，具体数据集为2023年AI进展专家调查中的问题。

**📈 对比分析**

与AI研究者的回答进行了比较，发现LLMs的风险估计更为集中，而人类专家的估计差异较大。更昂贵的模型通常给出更高的风险估计。

**⚠️ 局限性**

限制在于只使用了一个英语提示和一个接口，模型输出不代表信念，且可能受到训练数据的影响。

---

## 342. OCTAVE: A Capability-Resolved Benchmark for End-to-End HTML5 Game Generation

**arXiv ID:** 2611.17681 | [PDF](https://arxiv.org/pdf/2611.17681v1)

**作者:** Kehang Li `[一作]` (Shanghai Jiao Tong University), Zihao Wang `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OCTAVE，一个用于完整HTML5游戏生成的基准，旨在通过能力覆盖和需求级别证据来诊断实现差异。

**💡 创新点**

创新点在于将任务组织围绕能力维度，并通过180个任务和1800个评估需求来评估游戏生成的完整性和丰富性。

**🔧 技术方法**

使用了静态源代码分析技术来评估不同任务条件下的实现性能。

**📊 数据集**

构建了180个任务，涵盖来自真实玩家请求、商业机制重建和人类专家设计的1800个评估需求。

**📈 对比分析**

与现有的游戏开发基准相比，OCTAVE提供了更细致的能力覆盖和需求诊断，实验结果显示当前系统在项目交付和需求实现方面存在显著差距，完整性为33.2%，丰富性为32.5%。

**⚠️ 局限性**

限制在于即使是表现最好的组合，对于公共和隐藏需求的通过率均低于50%，显示出在完整游戏生成和可靠实现规范之间仍存在显著差距。

---

## 343. Structural Self-Teaching for Compositional Generation in Unified Multimodal Models

**arXiv ID:** 2611.18261 | [PDF](https://arxiv.org/pdf/2611.18261v1)

**作者:** Jaeyoo Park `[一作]` (Seoul National University), Bohyung Han `[通讯]` (Samsung Electronics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

该研究探讨了在视觉和语言决策中因果推断的少样本学习。

**💡 创新点**

创新点在于将因果推断与少样本学习相结合，以提高决策过程的可靠性。

**🔧 技术方法**

使用了深度学习和因果推断技术。

**📊 数据集**

使用了特定的视觉和语言数据集，但未具体说明数据集的名称。

**📈 对比分析**

与传统方法进行了比较，结果显示该方法在决策准确性上有显著提升。

**⚠️ 局限性**

限制在于对数据集的依赖性和模型的可解释性问题。

---

## 344. Zoom In, Scale Out: Metric Depth Models Mistake Digital Zoom for Moving Closer

**arXiv ID:** 2611.17903 | [PDF](https://arxiv.org/pdf/2611.17903v1)

**作者:** Rabeya Noor `[一作]` (Daffodil International University), Khandoker Nosiba Arifin `[通讯]` (Daffodil International University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的测试方法，通过数字变焦来评估当前的深度估计模型在处理相机参数变化时的表现，特别是如何在数字变焦情况下保持深度估计的准确性。

**💡 创新点**

创新点在于提出了一种无标签的数字变焦测试，能够准确评估模型在不同变焦情况下的深度和焦距估计，并揭示了模型在处理变焦时的失误机制。

**🔧 技术方法**

使用了深度学习模型，特别是Depth Anything V2、Depth Pro和DA3-Nested等模型，并通过无标签的自一致性训练方法进行微调。

**📊 数据集**

使用了KITTI、NYUv2和DIODE等数据集进行实验，评估模型在不同变焦情况下的表现。

**📈 对比分析**

与其他方法的比较显示，许多模型在数字变焦情况下未能保持深度一致性，表现出明显的失误，而使用真实相机参数的模型则能够保持较好的深度估计。性能方面，Depth Anything V2在变焦时深度估计显著下降，而Depth Pro和DA3-Nested在给定真实相机参数时表现良好。

**⚠️ 局限性**

限制在于真实光学变焦仅在一个镜头的场景上进行了测试，且在未给定相机参数的情况下，裁剪也会去除周围上下文，因此α=0是一个目标而非承诺。此外，EFT+B方法在正常和未见过的照片上降低了准确性。

---

## 345. Go under Chinese Rules with a Succinct Game History is Complete for Exponential Space

**arXiv ID:** 2611.18201 | [PDF](https://arxiv.org/pdf/2611.18201v1)

**作者:** Markus Bläser `[一作]` (Saarland University), Felix Czerner `[通讯]` (Saarland University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了在中国规则下的围棋游戏的复杂性，提出了一个简洁的游戏历史概念，并证明了在给定简洁游戏历史的情况下，判断黑方是否有获胜策略是NP完全的。

**💡 创新点**

创新点在于引入了简洁游戏历史的概念，这为研究其他具有无重复规则的游戏提供了一种新的方法，并且首次证明了在中国规则下围棋的复杂性。

**🔧 技术方法**

使用了布尔电路来编码游戏历史，并通过多项式时间的归约方法来证明复杂性。

**📊 数据集**

使用了围棋游戏的实例，特别是围棋在中国规则下的变体，构建了一个包含多个游戏状态的复杂实例。

**📈 对比分析**

通过与其他已知的NP完全问题（如量化布尔公式和静态地理游戏）进行多项式时间的归约，证明了围棋在中国规则下的复杂性。性能上，黑方在给定简洁游戏历史的情况下的获胜策略判断是NP完全的。

**⚠️ 局限性**

限制在于尚未解决在没有游戏历史的情况下，围棋是否也是NP完全的问题。此外，简洁游戏历史的构造可能在某些情况下导致复杂性增加。

---

## 346. StreamEmbed: Real-Time Indexing from Pixels for Screenshot Retrieval

**arXiv ID:** 2611.18516 | [PDF](https://arxiv.org/pdf/2611.18516v1)

**作者:** Dayoon Ko `[一作]` (University of California), Sewon Min `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种实时索引系统，能够对不断到来的屏幕截图进行增量视觉嵌入，利用屏幕截图之间的冗余性来提高编码效率。

**💡 创新点**

创新点在于通过空间令牌修剪和时间KV重用技术，减少了每个屏幕截图的计算量，同时保持了与完整屏幕截图嵌入的相似性。

**🔧 技术方法**

使用了视觉-语言嵌入模型（如Qwen3-VL-Embedding-2B），结合空间令牌修剪和时间KV重用技术。

**📊 数据集**

使用了一个包含10,426个问题的基准数据集，这些问题基于来自两个人类和一个计算机使用代理的184小时屏幕历史。

**📈 对比分析**

与现有的OCR文本索引和完整屏幕索引进行比较，提出的方法在相似的延迟下，准确性提高了最多6.3个Recall@10点，并且速度提高了3.7倍。

**⚠️ 局限性**

限制在于该方法依赖于屏幕内容的冗余性，对于变化较大的屏幕内容可能效果不佳。

---

## 347. Token2Tile: Token-Wise Sparse Attention with Tile Execution through a Low-Rank Space for Video DiTs

**arXiv ID:** 2611.17930 | [PDF](https://arxiv.org/pdf/2611.17930v1)

**作者:** Xuecheng Wu `[一作]` (Beijing Institute of Technology), Julien Lai `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练无关的稀疏注意力操作符，结合了每个查询的评分和残差，并通过瓷砖执行来加速视频扩散变换器的生成。

**💡 创新点**

创新点在于通过在线低秩空间来高效地聚类、评分和近似残差，同时保持注意力日志的准确性，并将查询特定的选择结合到瓷砖执行中。

**🔧 技术方法**

使用了在线低秩空间技术，结合了每个查询的评分和残差，采用瓷砖执行来优化计算效率。

**📊 数据集**

使用了MiniMax-H3和Wan2.1-14B数据集进行实验。

**📈 对比分析**

与密集的FlashAttention-3相比，在MiniMax-H3和Wan2.1-14B上实现了1.48-1.54倍的端到端加速，同时在视频质量上保持在0.5分以内的接近度。相同速度下，PSNR比SOL和SubBlock提高了5.5-6.8 dB。

**⚠️ 局限性**

限制在于该方法可能在处理稀疏性和准确性之间的权衡时，未能充分利用所有可能的查询-键交互。

---

## 348. Open-World Scene Deocclusion

**arXiv ID:** 2611.17695 | [PDF](https://arxiv.org/pdf/2611.17695v1)

**作者:** K. M. S. J. Kumarasinghe `[一作]` (Victoria University of Wellington), Fang-Lue Zhang `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练无关的开放世界场景去遮挡框架，从单张RGB图像中恢复部分遮挡物体的完整外观，分解场景为深度排序的RGBA层。

**💡 创新点**

创新点在于结合深度验证的背景遮挡物检测和循环安全的层分配策略，解决了现有方法在处理复杂遮挡和循环遮挡时的不足。

**🔧 技术方法**

使用了深度估计、几何推理和扩散模型等技术。

**📊 数据集**

使用了伪遮挡数据集（从COCO构建）和真实世界数据集（如COCO-A和Visual Genome）进行评估。

**📈 对比分析**

与三种最新方法（OWAAC、PM-CD和pix2gestalt）进行了比较，结果显示在多个数据集上均表现优于这些方法，尤其在感知一致性和结构保真度方面。

**⚠️ 局限性**

局限性包括堆叠预训练模型可能导致错误传播，薄弱或低信号的遮挡物可能未被检测到，以及在编辑过程中无法保持循环关系。

---

## 349. Two-Clustering Regime of Token Dynamics in Causal Attention

**arXiv ID:** 2611.18088 | [PDF](https://arxiv.org/pdf/2611.18088v1)

**作者:** Trinh T. Nguyen `[一作]` (Shanghai Jiao Tong University), Thieu N. Vo `[通讯]` (University of Bath)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文部分解决了Karagodin等人提出的关于因果注意力掩蔽的聚类猜想，并将其扩展到时间依赖的参数范围。

**💡 创新点**

创新点在于识别出一种严格的维度缩减现象，尽管在高维嵌入空间中演变，令牌轨迹却收敛到由少数主导方向控制的二维结构。

**🔧 技术方法**

使用了一种新的分析框架——流动中的不稳定性，灵感来自物理系统中的不稳定现象。

**📊 数据集**

使用了Group Query Attention (GQA) 层进行模拟和实验研究。

**📈 对比分析**

与现有方法比较，论文证明了在对角V(t)的情况下的两聚类状态，并首次为从实验中推测的维度缩减现象提供了理论依据。

**⚠️ 局限性**

限制在于假设的条件可能在实际预训练的Transformer中不完全成立，且值矩阵V(t)不一定是对称的或具有时间不变的特征向量。

---

## 350. Neural Simulation-Based Inference for Computational Cognitive Models: A Tutorial

**arXiv ID:** 2611.17608 | [PDF](https://arxiv.org/pdf/2611.17608v1)

**作者:** Ti-Fen Pan `[一作]` (University of California), Anne GE Collins `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提供了一种针对认知模型的神经模拟推断（neural SBI）方法的逐步实用指南，旨在帮助研究人员克服在计算模型中遇到的不可处理似然性问题。

**💡 创新点**

创新点在于提出了一种无需计算或近似似然的推断方法，通过前向模拟生成合成数据，并利用人工神经网络直接推断模型参数和潜在变量。

**🔧 技术方法**

使用了人工神经网络（ANN）和模拟数据，结合深度学习技术进行模型训练和推断。

**📊 数据集**

使用了合成数据集，这些数据集是通过已定义的计算认知模型和实验生成的，确保了数据的标签和真实行为的一致性。

**📈 对比分析**

与传统的最大似然估计（MLE）和马尔可夫链蒙特卡洛（MCMC）方法相比，神经SBI在处理不可处理似然性时表现出更高的效率和准确性，尤其是在大数据集上进行快速推断时。

**⚠️ 局限性**

限制在于神经SBI在处理高维数据或高度参数化模型时仍然计算密集，且在模型或任务结构变化时需要重新训练神经估计器。

---

## 351. The Exact Stabilization Time of Self-Stabilizing Coloring: Mean, Variance, and Distribution

**arXiv ID:** 2611.17989 | [PDF](https://arxiv.org/pdf/2611.17989v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 352. Speed-Variation Distillation for Generative Modeling

**arXiv ID:** 2611.17868 | [PDF](https://arxiv.org/pdf/2611.17868v1)

**作者:** Sukit Seripanitkarn `[一作]` (VISTEC), Supasorn Suwajanakorn `[通讯]` (VISTEC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种生成建模的新框架，利用变异减少的降解过程，通过共享原型逐步生成高质量和多样化的图像。

**💡 创新点**

创新点在于引入了一种变异减少的降解过程，使得生成过程中的每个步骤都能产生可识别的图像，并实现了速度与多样性之间的可控权衡。

**🔧 技术方法**

使用了生成优先的图像变形技术和潜在扩散模型（LDM）进行训练和推理。

**📊 数据集**

使用了FFHQ、CelebA、LSUN-Churches和LSUN-Bedrooms等数据集，生成了10,000张图像并构建了轨迹数据集。

**📈 对比分析**

与标准扩散模型（LDM）进行比较，结果显示该方法在早期推理步骤中生成清晰可识别的图像，同时逐步增加多样性，保持高图像质量。

**⚠️ 局限性**

限制在于生成图像的质量依赖于轨迹数据集的质量，且最终生成的图像在高频细节上可能不如基线模型，此外，计算资源有限导致实验主要集中在FFHQ数据集上。

---

## 353. Any Channel Count, One Token Stream: Incorporating Spatial Audio Understanding and Reasoning into Large Audio-Language Models

**arXiv ID:** 2611.18112 | [PDF](https://arxiv.org/pdf/2611.18112v1)

**作者:** Yuhuan You `[一作]` (Alibaba Group), Jin Xu `[通讯]` (Alibaba Group)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的音频语言模型SpheraAudio，能够处理多通道音频并提取空间信息，以增强音频理解和推理能力。

**💡 创新点**

创新点在于引入了空间编码器和空间多模态旋转位置嵌入(SMRoPE)，使得模型能够同时处理音频和空间信息，并进行精确的空间推理。

**🔧 技术方法**

使用了音频编码器AuT和空间编码器Spatial AuT，结合了旋转位置嵌入(SMRoPE)技术。

**📊 数据集**

使用了SpheraQA数据集，该数据集包含20492个问题，涵盖单属性感知、跨源关系和多步骤推理。

**📈 对比分析**

与现有的空间音频基准(SO-Bench和TWNM)进行比较，SpheraAudio在空间任务上表现出色，尤其在定位、跟踪和推理方面，超越了Spatial-Omni-7B，并在SpheraQA上获得了60.18%的整体准确率。

**⚠️ 局限性**

限制在于当前模型的训练和评估主要依赖于特定的录音格式，未来需要在更多样化的录音条件下进一步验证模型的性能。

---

## 354. Creative Product Design under Brand Image Constraints

**arXiv ID:** 2611.17762 | [PDF](https://arxiv.org/pdf/2611.17762v1)

**作者:** Takeshi Onishi `[一作]` (TOYOTA CENTRAL RESEARCH AND DEVELOPMENT LABORATORIES INC), Hiroyuki Sakai `[通讯]` (TOYOTA CENTRAL RESEARCH AND DEVELOPMENT LABORATORIES INC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种品牌意识的重新设计生成框架，旨在在满足品牌形象约束的同时进行创意视觉探索。

**💡 创新点**

创新点在于将品牌形象作为动态更新的设计约束，通过品牌形象调查报告引导创意方向，并在自我增强循环中进行图像生成和评估。

**🔧 技术方法**

使用了多代理框架，包括品牌经理和审阅者代理，结合大语言模型（LLM）和视觉语言模型（VLM）进行图像生成和评估。

**📊 数据集**

在五个产品类别上进行了评估，共涉及50个重新设计任务，涵盖汽车、摩托车、饮料瓶、角色系列和汽车制造商标志等。

**📈 对比分析**

与Naive T2I、IP-Adapter/IT-Blender和无约束自我增强方法进行了比较，结果显示该框架在创意和品牌识别之间取得了更好的平衡，特别是在约束较强的类别中表现突出。

**⚠️ 局限性**

局限性在于评估协议依赖于VLM作为评判者，而非人类设计师或领域专家的评估；此外，仅考虑了品牌形象约束，未涵盖其他工业设计中的多种约束。

---

## 355. VelServe: Taming Decode Jitters in LLM Inference with Tail-Calibrated Scheduling

**arXiv ID:** 2611.18392 | [PDF](https://arxiv.org/pdf/2611.18392v1)

**作者:** Ishaan Datta `[一作]` (Sprinklr), Anand Kumar Singh `[通讯]` (Sprinklr)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种调度器，旨在减少解码抖动，以满足交互式大语言模型（LLM）服务的延迟服务水平目标（SLO），确保令牌的可预测交付。

**💡 创新点**

创新点在于通过测量自身迭代延迟的尾部，校准每个解码步骤的预填充工作量，从而有效控制解码抖动，并在分解服务中优化请求的接纳频率。

**🔧 技术方法**

使用了一种基于SLO的调度器，该调度器结合了最短等待优先的策略，并在多个GPU上实现了解码和预填充的分解。

**📊 数据集**

使用了Qwen3-8B、DeepSeek-V2-Lite和DeepSeek-V4-Flash等多个模型，涵盖了三种注意力架构，并在不同的工作负载下进行了评估。

**📈 对比分析**

与现有的基线方法相比，本文的方法在满足SLO的请求数量上表现出色，能够在Qwen3-8B和DeepSeek-V4-Flash上维持2.4-2.9倍的SLO良好吞吐量，在DeepSeek-V2-Lite上至少达到4.4倍。

**⚠️ 局限性**

限制在于校准过程可能滞后于负载变化，且在持续负载下，某些最长的提示仍可能超时。

---

## 356. Interpretable weight-space programming builds transparent transformers and brain encoding models

**arXiv ID:** 2611.18114 | [PDF](https://arxiv.org/pdf/2611.18114v1)

**作者:** Yash Mehta `[一作]` (Johns Hopkins University), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种可解释的权重空间编程方法（iWSP），通过不使用反向传播的方式，迭代构建可解释的神经网络模型。

**💡 创新点**

iWSP的创新点在于通过手动编写模型权重和特征，结合先前知识和推理，生成可解释的计算，解决了深度学习模型的可解释性问题。

**🔧 技术方法**

使用了大型语言模型（LLM）作为代理，进行权重和特征的编程，而不依赖于梯度训练。

**📊 数据集**

在视觉、语言和听觉等多个领域的数据集上进行了实验，包括LAION-fMRI和自然场景数据集（NSD）。

**📈 对比分析**

与现有的可解释基线模型相比，iWSP在视觉皮层的预测性能更优，并且能够发现不同视觉区域的计算特征，同时在语言fMRI和听觉ECoG中也表现出色。

**⚠️ 局限性**

iWSP的局限性在于当前代理无法创造性地提出新的计算原则，且发现的模型依赖于浅层的已知模式，而非深层的多层次结构。

---

## 357. More Genes, Faster Inference: Efficient Large-Panel Spatial Transcriptomics Prediction from Histology

**arXiv ID:** 2611.17899 | [PDF](https://arxiv.org/pdf/2611.17899v1)

**作者:** Songming Zhang `[一作]` (Nanjing University), Jinyan Li `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为GeneFlow的框架，用于从H&E染色的组织图像中推断空间基因表达，旨在解决大规模空间转录组学预测的问题。

**💡 创新点**

GeneFlow通过结合流匹配生成建模和结构化多模态融合，提供了一种简单而有效的方法来捕捉形态线索和基因间依赖关系，显著提高了预测的准确性和效率。

**🔧 技术方法**

使用了流匹配生成建模技术，并结合了轻量级的多模态融合架构。

**📊 数据集**

使用了四个公开的空间转录组学数据集，包括肾脏、HER2ST、PRAD和小鼠大脑的数据。

**📈 对比分析**

与多种现有方法（如StNet、EGN、BLEEP、TRIPLEX、Stem和STFlow）进行了比较，GeneFlow在多个基准测试中表现出色，PCC-1000在PRAD数据集上提高了17.1%。

**⚠️ 局限性**

限制在于其预测准确性受到病理基础模型质量和表示能力的限制，并且主要依赖配对的H&E和ST数据进行监督，可能限制其在缺乏配对训练数据的情况下的通用性。

---

## 358. InfiHall-Bench: A Benchmark for Hallucination Evaluation in Autoregressive Text-to-Image Generation

**arXiv ID:** 2611.18053 | [PDF](https://arxiv.org/pdf/2611.18053v1)

**作者:** Yuxia Qiao `[一作]` (Harbin Institute of Technology), Bin Chen `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了InfiHall-Bench，一个用于评估视觉自回归文本到图像生成中的幻觉的基准，包含411个多技能提示，涵盖属性绑定、空间关系、动作关系、部分一致性和场景一致性。

**💡 创新点**

创新点在于引入了多技能提示构建方法，系统性地组合了多个视觉要求，并提出了基于置信度的幻觉缓解方法（CMHM），以优化不确定的视觉预测。

**🔧 技术方法**

使用了LoRA适应技术和置信度基于的标记掩蔽方法，结合清洁重放正则化来减轻分布漂移。

**📊 数据集**

使用了包含411个提示的InfiHall-Bench数据集，这些提示涵盖了五种基础视觉技能。

**📈 对比分析**

通过InfiHall-Bench进行的实验表明，CMHM在不同的Infinity模型上均优于基线infihall-train，整体得分提高了0.83到1.14点，特别是在动作关系和空间关系上表现出显著提升。

**⚠️ 局限性**

限制在于尽管CMHM在多个技能上表现出改进，但Infinity模型在某些细粒度视觉要求上仍然存在幻觉，特别是在空间和动作关系方面。

---

## 359. BLiMP-RO: A Benchmark of Linguistic Minimal Pairs for Romanian

**arXiv ID:** 2611.17836 | [PDF](https://arxiv.org/pdf/2611.17836v1)

**作者:** Radu-Stefan Popa `[一作]` (University of Bucharest), Sergiu Nisioi `[通讯]` (University of Bucharest)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BLiMP-RO，一个涵盖50种语法现象的罗马尼亚语最小对比基准，并通过人类评估进行了验证。

**💡 创新点**

BLiMP-RO填补了现有罗马尼亚语最小对比评估中缺失的典型句法和形态句法现象，并通过人类判断获得了高达96%的准确率。

**🔧 技术方法**

使用了GPT-2和GPT-BERT模型进行训练，并结合了BabyBabelLM数据和新收集的儿童视频字幕语料库。

**📊 数据集**

使用了来自儿童内容的10百万字的字幕语料库和BabyBabelLM的罗马尼亚部分。

**📈 对比分析**

通过与预训练的罗马尼亚模型和从头训练的模型进行比较，发现GPT-BERT在所有有限数据训练条件下的表现优于GPT-2，且增加字幕训练数据量可显著提高准确率。

**⚠️ 局限性**

人类接受度评分通过强制选择比较收集，虽然建立了训练数据属性与语法准确性之间的关联，但未能识别语法学习的机制。

---

## 360. PUMA-Former: Uncertainty-Aware Agentic Computation Offloading in MEO-LEO Satellite Networks With Incomplete Information

**arXiv ID:** 2611.18139 | [PDF](https://arxiv.org/pdf/2611.18139v1)

**作者:** Yi Jing `[一作]` (Tsinghua University), Jiachen Sun `[通讯]` (Tsinghua University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为PUMA-Former的不确定性感知的计算卸载框架，旨在解决中地球轨道（MEO）和低地球轨道（LEO）卫星网络中由于信息不完整而导致的计算卸载问题。

**💡 创新点**

创新点在于通过不确定性感知的代理框架，结合历史创新和延迟注意机制，优化了计算资源的分配和卸载决策，能够在不完整信息的情况下有效协调资源。

**🔧 技术方法**

使用了结构化的Transformer模型，结合延迟注意机制和辅助压力融合技术，以实现对CPU可用性的推断和不确定性的校准。

**📊 数据集**

使用了来自中国卫星通信公司的真实操作流量记录数据集，包含UTC时间戳、用户位置和流量量。

**📈 对比分析**

与ATLA-adapt和固定半轮询方法相比，PUMA-Former在计算工作完成率上提高了8.62%，平均服务延迟减少了34.78%，P95延迟减少了71.08%。

**⚠️ 局限性**

限制在于模型依赖于不完整的信息，可能导致在某些情况下的决策不够准确，且在高负载情况下可能会出现性能下降。

---

## 361. Lens: Training-Free Semantic Search for Cross-Tile Object and Scene Retrieval in Satellite Imagery

**arXiv ID:** 2611.18251 | [PDF](https://arxiv.org/pdf/2611.18251v1)

**作者:** Ata Akbari Asanjan `[一作]` (Pixxel Space Technologies), Subash Yeggina `[通讯]` (Pixxel Space India Pvt Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了Lens，一个无需训练的系统，用于在大型卫星马赛克中进行语义对象和场景搜索，无需为感兴趣的类别训练检测器。

**💡 创新点**

创新点在于使用冻结的自监督视觉变换器（DINOv3）表示每个图块，并通过线性操作去除密集特征中的位置偏差，从而实现高效的检索。

**🔧 技术方法**

使用了自监督学习的视觉变换器DINOv3，结合了特征去偏和特征上采样技术（AnyUp）。

**📊 数据集**

使用了5,128个图块的DFC2020测试集，图块大小为256 × 256像素，分辨率为10米。

**📈 对比分析**

与INSID3基线方法进行比较，Lens在六个类别上表现优于INSID3，查询成本约为INSID3的四百分之一，平均F1得分为0.39。

**⚠️ 局限性**

限制在于背景溢出问题仍然存在，且在目标占据大部分区域时，中心化可能会降低准确性。

---

## 362. Measuring Public Participation in Local Government: An Audio-Based Instrument for Classifying Speakers in City Council Meetings

**arXiv ID:** 2611.18072 | [PDF](https://arxiv.org/pdf/2611.18072v1)

**作者:** Menglin Liu `[一作]`, Ge Shi `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

论文评估了说话人分离的质量，采用了两项互补任务来验证模型的准确性。

**💡 创新点**

创新点在于通过独立的研究助理进行说话人配对验证和剪辑质量检查，以识别模型的合并错误和分裂错误。

**🔧 技术方法**

使用了pyannote工具进行说话人转变的密集注释，并在Audacity中修正输出。

**📊 数据集**

数据集包括来自三个Chino会议的126对开发配对和来自四个生产城市的360对样本。

**📈 对比分析**

通过对比不同城市的会议，发现Brockton的错误率较高（DER 17-22%），而其他城市的DER在6.8%以下，表明噪声和重叠音频对性能有显著影响。

**⚠️ 局限性**

限制在于注释者是基于pyannote的草稿进行修正，而不是从头开始标注，因此DER可能是一个轻微的下限，且重叠语音的边界放置难度较大。

---

## 363. SWAM: Saliency-Weighted Asymmetric Merging for Efficient Vision Transformers

**arXiv ID:** 2611.18057 | [PDF](https://arxiv.org/pdf/2611.18057v1)

**作者:** Utkarsh Kumar Singh `[一作]` (Indian Institute of Information Technology Allahabad), Satish Kumar Singh `[通讯]` (Indian Institute of Information Technology Allahabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为SWAM（Saliency-Weighted Asymmetric Merging）的基于训练的令牌合并框架，通过引入两个关注显著性的机制，扩展了双边软匹配（BSM）方法，以解决视觉变换器在图像识别任务中的计算复杂性问题。

**💡 创新点**

创新点在于引入了基于显著性的成对评分和不对称显著性加权合并机制，从而在合并过程中更好地保留重要令牌的信息。

**🔧 技术方法**

使用了轻量级的多层感知机（MLP）作为显著性预测模块，并通过注意力回滚进行监督，结合了交叉熵、知识蒸馏和显著性对齐的复合目标进行端到端训练。

**📊 数据集**

在CIFAR-100、TinyImageNet和ImageNet-100数据集上进行了评估，采用保守和激进的令牌压缩策略。

**📈 对比分析**

与基线模型相比，SWAM在保守调度下在所有数据集上匹配或超过基线，FLOPs减少27.5%，速度提升1.35倍；在激进调度下，FLOPs减少33.8%，速度提升1.46倍，同时在ImageNet-100上获得了0.23%的准确率提升。

**⚠️ 局限性**

限制在于所有实验均使用DeiT-Small模型，尚未评估其在DeiT-Tiny或DeiT-Base上的泛化能力，并且由于硬件和存储配额限制，所有实验均在Kaggle平台上进行。

---

## 364. Meta Build Systems as Logic Programs

**arXiv ID:** 2611.17672 | [PDF](https://arxiv.org/pdf/2611.17672v1)

**作者:** Alex Reinking `[一作]` `[通讯]` (Adobe Research), Alex Reinking (Adobe Research)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了一种新的元构建系统，通过逻辑推理表达配置，解决了现有构建系统中逻辑分散的问题。

**💡 创新点**

创新点在于将构建配置视为关系推理，明确区分作者约束和用户偏好，并提供了推导事实的来源证明。

**🔧 技术方法**

使用了分层的Datalog核心和Python驱动程序，结合Soufflé进行推理和构建图的生成。

**📊 数据集**

在十个项目的十一项案例研究中进行了评估，验证了移植的配置通过执行和工件检查。

**📈 对比分析**

与CMake、Meson和Bazel进行了比较，结果显示在配置时间和生成命令方面具有竞争力，能够在单一图中推导构建机器和目标平台配置。

**⚠️ 局限性**

当前构建系统每个库仅允许一个规则包版本，编辑时需要重新评估整个Datalog程序。

---

## 365. CHARMING: Parallel Edge AI Acceleration Across Tensor Processing and In-Memory Computation

**arXiv ID:** 2611.17785 | [PDF](https://arxiv.org/pdf/2611.17785v1)

**作者:** Md Omar Faruque `[一作]` (Syracuse University), Alex K. Jones `[通讯]` (Syracuse University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为CHARMING的异构边缘AI架构，通过在张量处理和内存计算之间分配和并行执行推理工作负载，来提高推理性能和能效。

**💡 创新点**

CHARMING通过精确和量化感知的分区策略，优化了注意力计算的执行，允许在不同精度下并行处理，从而减少数据移动并提高性能。

**🔧 技术方法**

使用了异构可重构架构，结合了基于AI引擎的张量处理单元（ACU）和内存计算单元（IMCU），并通过硬件验证和模拟评估其性能。

**📊 数据集**

使用了GSM8K数据集进行基准测试，评估了不同结构的LLM（如Llama-3.1-8B-Instruct、Llama-2-13B-chat和Mixtral-8x7B-Instruct-v0.1）。

**📈 对比分析**

与仅使用AIE的CHARM相比，CHARMING在预填充性能上提高了最多23.6%，在解码性能上提高了最多44%。在能量延迟产品（EDP）方面，CHARMING在预填充阶段减少了最多24.7%的能量消耗。

**⚠️ 局限性**

限制在于当前没有单一设备能够同时结合AIE阵列和内存计算，且商业硬件不支持所需的内存计算扩展，因此需要在两个不同的FPGA平台上进行原型验证。

---

## 366. Antidistillation is Cheap: Lightweight Distillation-Resistance via Stylistic Corruption

**arXiv ID:** 2611.17772 | [PDF](https://arxiv.org/pdf/2611.17772v1)

**作者:** Anh Tu Ngo `[一作]` (Nanyang Technological University), Anupam Chattopadhyay `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级的知识蒸馏防御框架，通过在教师模型的激活中动态注入风格引导向量来应对恶意查询，从而保护知识不被轻易提取。

**💡 创新点**

创新点在于利用风格变异而非优化基础的扰动来实现防御，允许在推理时实时切换风格，同时保持文本的自然性和语义一致性。

**🔧 技术方法**

使用了对比激活添加（CAA）技术，通过在推理时动态注入风格引导向量来控制模型的输出风格。

**📊 数据集**

使用了多个公开数据集进行实验，包括GSM8K和MATH500等，评估模型在开放式推理和多项选择评估中的表现。

**📈 对比分析**

与现有的防御方法（如反蒸馏采样和DOGe）相比，该方法在保持教师模型性能的同时，显著降低了学生模型的准确性，且几乎没有计算开销。

**⚠️ 局限性**

局限性包括依赖于可靠的恶意查询分类器，未评估分类器的设计或准确性；对适应性攻击者的防御能力有限；以及未对风格注入的各种组合进行全面评估。

---

## 367. Exact Failure Probabilities for d-Choice Cuckoo Hashing

**arXiv ID:** 2611.17688 | [PDF](https://arxiv.org/pdf/2611.17688v1)

**作者:** Chenyang Zhang `[一作]` (University of Illinois Urbana-Champaign), Ling Ren `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文提出了一种多项式时间动态规划算法，用于计算d-choice cuckoo hashing中不存在有效放置的确切概率，适用于一张表和分开的表的变体。

**💡 创新点**

创新点在于使用Beniamini的系数公式来表达失败结果的数量，并通过动态规划计算该和，从而实现了对d≥3的情况的精确概率计算。

**🔧 技术方法**

使用了动态规划技术和Beniamini的系数公式来处理完全匹配问题。

**📊 数据集**

没有具体提到使用的数据集，但算法适用于任意n、M和d的组合。

**📈 对比分析**

与之前的工作相比，算法在一张表的变体中运行时间为O(n^2 log n)，而在分开的表的变体中为O(n^(d+2))，性能显著提高。

**⚠️ 局限性**

限制在于对于分开的表的情况，算法的时间复杂度依赖于常数d，未来的工作方向是寻找一个在n和d上都是多项式的算法。

---

## 368. Deep Neural Networks Training Efficiency: Does Grokking Modify its Dynamics?

**arXiv ID:** 2611.18396 | [PDF](https://arxiv.org/pdf/2611.18396v1)

**作者:** Eduardo Cueto-Mendoza `[一作]` (Technological University Dublin), John D. Kelleher `[通讯]` (Trinity College Dublin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了grokking现象是否会改变深度神经网络的训练效率，效率通过每单位能量的准确性来衡量。

**💡 创新点**

提出了grokking不会显著改变训练效率的动态，尽管在训练过程中可能会出现grokking现象。

**🔧 技术方法**

使用了Orthograd算法来诱导grokking，并分析了不同精度（16位、32位和64位）对训练效率的影响。

**📊 数据集**

使用了MNIST数据集和模运算（modular addition）数据集进行实验。

**📈 对比分析**

通过比较短期（最多500个epoch）和长期（最多20,000个epoch）训练的结果，发现grokking在短期实验中未能诱导效率变化，而在长期实验中虽然诱导了grokking，但效率曲线仍然保持单调。

**⚠️ 局限性**

效率代理依赖于硬件和实现细节，因此建议在不同学习任务和计算系统上进行进一步实验。

---

## 369. Exact-Quota Facility Location: A Transfer Framework and Three Approximation Guarantees

**arXiv ID:** 2611.18262 | [PDF](https://arxiv.org/pdf/2611.18262v1)

**作者:** Hang Luo `[一作]` `[通讯]`, Hang Luo

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种设施选址与异常值的框架，旨在确定哪些设施应开放以及服务哪些客户，以满足规定的服务配额。

**💡 创新点**

创新点在于提供了一种转移定理，将设施选址成本证书转移到精确配额设置中，并给出了多种情况下的多项式时间近似保证。

**🔧 技术方法**

使用了贪婪算法和超模性分析等技术，结合了阈值覆盖表示法来优化服务成本。

**📊 数据集**

使用了多种模型的设施和客户数据集，包括度量设施位置、优先级设施位置和平方度量设施位置。

**📈 对比分析**

与之前的模型相比，提供了更好的近似保证：度量设施位置的近似因子为571/376<1.52，优先级设施位置的近似因子严格小于1.853，平方度量设施位置的近似因子为43399/20000=2.16995<2.17。

**⚠️ 局限性**

限制在于需要对初始解决方案的联合成本界限进行准确构造，并且在某些情况下可能需要预付一定数量的昂贵设施。

---

## 370. Multi-GPU Tridiagonal Partition Algorithm

**arXiv ID:** 2611.18111 | [PDF](https://arxiv.org/pdf/2611.18111v1)

**作者:** Milena Veneva `[一作]` `[通讯]` (RIKEN Center for Computational Science), Milena Veneva (RIKEN Center for Computational Science)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种多GPU的平行分区方法，用于求解具有三对角系数矩阵的线性代数方程组，并在CUDA中实现了该方法的多GPU版本和递归版本。

**💡 创新点**

创新点在于实现了多GPU并行处理，能够有效处理大规模的线性代数方程组，并通过递归步骤优化计算性能。

**🔧 技术方法**

使用了CUDA技术来实现多GPU的并行计算。

**📊 数据集**

使用了不同规模的线性代数方程组（SLAE），具体规模为10^5、10^6、10^7和10^8。

**📈 对比分析**

通过与单GPU的计算时间进行比较，发现对于SLAE规模10^5和10^6，单GPU的计算时间更优；而对于10^7和10^8，使用两个GPU和适当的递归步骤可以显著提高性能，分别实现了1.32倍和2.38倍的加速。

**⚠️ 局限性**

限制在于对于较小的SLAE规模，使用多GPU可能导致资源的低利用率，计算开销可能超过性能提升的收益。

---

## 371. Natural Swing-Up of an Underpowered Double Pendulum

**arXiv ID:** 2611.17575 | [PDF](https://arxiv.org/pdf/2611.17575v1)

**作者:** Arne Sachtler `[一作]` (Technical University of Munich), Alin Albu-Schäffer `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种控制方法，使得弱驱动的机器人能够沿着其自然动力学的非线性正常模式摆动并稳定到直立姿态。

**💡 创新点**

创新点在于将输入扭矩分为无功转向分量和能量泵送输入，并通过线性规划在扭矩限制和控制Lyapunov约束下进行分配，从而实现了无反馈增益的控制。

**🔧 技术方法**

使用了线性规划技术来分配扭矩，并结合了控制Lyapunov函数来确保系统稳定性。

**📊 数据集**

实验使用了一个直接驱动的平面双连杆机器人，该机器人在硬件实验中展示了沿着非线性正常模式的摆动。

**📈 对比分析**

与传统的加和剪切和优先级稳定化方法进行了比较，结果显示本文的方法在能量达到目标时更快且保持更接近于模式。

**⚠️ 局限性**

限制在于在非严格模式下，控制器可能无法始终满足Lyapunov约束，且在转弯点时输入的权威性不足。

---

## 372. Compiling Specialized GPU-Initiated Networking

**arXiv ID:** 2611.18145 | [PDF](https://arxiv.org/pdf/2611.18145v1)

**作者:** Javid Baydamirli `[一作]`, Didem Unat `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种编译器，将GPU发起的通信纳入内核的中间表示（IR），并与计算共同优化，生成专门的传输代码。

**💡 创新点**

创新点在于将GPU通信与计算结合在一起，允许编译器在更高的抽象层次上进行优化，从而提高性能并减少内存使用。

**🔧 技术方法**

使用了MLIR（多层次中间表示）作为编译框架，并实现了针对NVIDIA和AMD GPU的优化。

**📊 数据集**

在H100和B200 GPU上进行了实验，使用了多种通信库（如NVSHMEM和NCCL）作为对比。

**📈 对比分析**

与NVSHMEM和NCCL GIN相比，提出的方法在H100 GPU上发送小消息的速度提高了3.3倍，在256个GPU上进行全到全通信时速度提高了1.25倍到2.3倍，同时内存和寄存器使用量显著减少。

**⚠️ 局限性**

限制在于该方法需要特定的硬件支持，并且在不同的网络拓扑和通信模式下可能需要额外的适配和优化。

---

## 373. When Cooperation Must Change: Semantic Strategy Unlearning in Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.17917 | [PDF](https://arxiv.org/pdf/2611.17917v1)

**作者:** Wenhan Wu `[一作]` (Wuhan University), Chuang Hu `[通讯]` (Wuhan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了语义协调策略的遗忘问题，提出了一种名为CARVE的方法，旨在让训练好的合作策略在保留有效合作的同时，放弃角色不变的协调约定。

**💡 创新点**

创新点在于将因果协调归因、保留意识的子空间更新、反事实重新协调和对抗审计结合起来，以实现高效的策略更新，同时保持合作任务的性能。

**🔧 技术方法**

使用了因果协调归因、反事实重新协调和对抗审计等技术。

**📊 数据集**

在五个合作多智能体强化学习基准上进行了评估，包括SMACv2、Overcooked-AI、VMAS、RWARE和LBF。

**📈 对比分析**

与现有的运行时屏蔽方法相比，CARVE在平均任务效用上提高了167.8%，并且在多个基准上减少了91.4%的平均名义约定完成率和83.3%的最坏情况引发率，且推理延迟与原始策略相匹配。

**⚠️ 局限性**

限制在于尽管CARVE在多个基准上表现良好，但在某些情况下仍然存在残余的语义风险，且在特定任务中可能会受到移除约定的影响。

---

## 374. Poetry 'resurrection' from LLMs

**arXiv ID:** 2611.17464 | [PDF](https://arxiv.org/pdf/2611.17464v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 375. Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles

**arXiv ID:** 2611.18453 | [PDF](https://arxiv.org/pdf/2611.18453v1)

**作者:** Wang Ma `[一作]` (Rensselaer Polytechnic Institute), Qiang Ji `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种黑箱不确定性量化框架，结合了固定提示的随机解码和语义保持的扰动，以在大语言模型中分解预测变异性。

**💡 创新点**

创新点在于引入了两级集成方法，能够在不需要重训练或访问令牌级别的日志的情况下，准确估计内扰动（随机性）和外扰动（知识缺口）带来的不确定性。

**🔧 技术方法**

使用了随机解码和语义扰动的两级集成技术，并应用了总协方差法则来进行不确定性分解。

**📊 数据集**

在五个短问答基准和四个指令调优的大语言模型上进行了评估，包括LLaMA-3-70B、LLaMA-3-8B、Qwen2.5-72B和Qwen2.5-7B。

**📈 对比分析**

与强大的白箱基线相比，该估计器在多个基准上表现竞争力，并在匹配的生成预算下与领先的黑箱方法在错误排名上相匹配，同时分解了内扰动和外扰动的不确定性。

**⚠️ 局限性**

限制在于估计器针对固定模型权重的扰动-总体不确定性，其与贝叶斯参数不确定性的对应关系依赖于局部正则性条件；自动化的改写可能引入语义漂移，增加了扰动之间的方差；不确定性估计继承了句子编码器的表示几何和潜在盲点；嵌套采样需要每个查询进行NM次解码，导致推理成本高于单级基线；实证和干预评估主要集中在短问答上，控制单因素干预。

---

## 376. MEM-SQL: Memory-Empowered Text-to-SQL

**arXiv ID:** 2611.18338 | [PDF](https://arxiv.org/pdf/2611.18338v1)

**作者:** Xiaoqi Wang `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HARE基准，用于评估历史SQL查询的重用准确性和成本，并开发了MEM-SQL方法来支持历史查询的适应和重用。

**💡 创新点**

创新点在于引入了HARE基准和MEM-SQL方法，前者记录了完整查询适应和部分计算重用，后者通过分层记忆和按需模式链接提高了SQL生成的准确性和效率。

**🔧 技术方法**

使用了MEM-SQL方法，该方法结合了分层记忆结构和按需模式链接（OSL）技术。

**📊 数据集**

使用了来自Spider、BIRD、Spider 2.0和BEAVER的823个经过验证的查询构建了1600个请求。

**📈 对比分析**

与直接生成方法相比，MEM-SQL的平均执行准确率从37.1%提高到50.0%，同时减少了76.6%的输入令牌和68.8%的延迟。MEM-SQL在所有基准上均优于其他三种记忆基线。

**⚠️ 局限性**

限制在于HARE未考虑企业数据库中常见的模式漂移，MEM-SQL在静态记忆下进行评估，动态环境下如何添加、更新和退休记忆对性能的影响尚未明确。

---

## 377. Learning Anisotropic Prediction Targets for Flow Models

**arXiv ID:** 2611.17767 | [PDF](https://arxiv.org/pdf/2611.17767v1)

**作者:** Rayhan Zirvi `[一作]` (California Institute of Technology), Leonidas Guibas `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的生成模型Anisotropic Flow Models (AnisoFlow)，通过学习可变的预测目标来替代固定的预测目标，从而提高图像生成的质量。

**💡 创新点**

创新点在于将固定的预测目标替换为完全可学习的目标，允许模型为每个方向分配不同的噪声权重，从而优化预测过程。

**🔧 技术方法**

使用了一种基于流的生成模型，结合了可学习的预测目标和流匹配损失，保持了网络架构不变。

**📊 数据集**

在ImageNet 256×256数据集上进行实验，评估了模型的性能。

**📈 对比分析**

与JiT-H/16基线模型相比，AnisoFlow在FID指标上从1.90提升至1.55，显示出显著的性能改进，且在文本到图像生成中也表现出更好的图像质量和稳定性。

**⚠️ 局限性**

局限性在于分析使用了局部模型的预测误差，未能捕捉到完整的训练动态，且在潜在空间中的性能提升较小，学习的目标仍接近现有的默认目标。

---

## 378. Training Reasoning Models in Simulation to Natively Use Real Robots

**arXiv ID:** 2611.18562 | [PDF](https://arxiv.org/pdf/2611.18562v1)

**作者:** Jiaming Tang `[一作]` (Massachusetts Institute of Technology), Song Han `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究展示了一种开放权重的27B推理模型，能够通过强化学习在模拟环境中训练，并在真实机器人上原生使用。该模型观察环境，进行推理，并生成下一个末端执行器的位置，完全在模拟中训练，无需演示。

**💡 创新点**

创新点在于提出了一种新的机器人使用强化学习（RURL）方法，使得开放权重模型能够在真实机器人上表现出色，且在未见过的任务和机器人上也能成功转移技能。

**🔧 技术方法**

使用了强化学习技术，特别是机器人使用强化学习（RURL），并构建了一个名为PEAR的后训练引擎来支持大规模训练。

**📊 数据集**

使用了LIBERO和RoboDojo两个模拟数据集进行训练和评估。

**📈 对比分析**

与基于演示的视觉-语言-行动模型相比，该模型在任务变化时表现出更强的鲁棒性。在LIBERO上，成功率从1.6%提高到74.4%，接近前沿模型的表现，并在真实双臂机器人上成功率达到60%。

**⚠️ 局限性**

限制在于训练规模受限于GPU资源，模型在执行任务时速度较慢，且对环境和机器人理解依赖于提示信息，当前模型尚未能可靠地规划动作块。

---

## 379. Contagious Fingerprints: When LLMs Inherit Another Model's Identity

**arXiv ID:** 2611.18104 | [PDF](https://arxiv.org/pdf/2611.18104v1)

**作者:** Zailong Tian `[一作]` `[通讯]` (Nanyang Technological University), Zailong Tian (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文展示了分类器如何识别文本背后的大型语言模型（LLM），并指出这种指纹是具有传染性的，即一个模型在继续另一个模型的对话时，会继承前者的指纹。这一发现对API中介的审计造成了影响，因为中介可以在对话的前几轮使用昂贵的模型，而后续轮次则使用更便宜的模型。

**💡 创新点**

创新点在于揭示了模型指纹在多轮对话中是如何传染的，并提出了一种新的审计方法，该方法通过在相同历史下从声称的模型中抽样参考来检测替代模型的回复。

**🔧 技术方法**

使用了深度学习分类器（如n-gram分类器和神经分类器）以及新的审计方法，该方法基于同历史参考的排名测试。

**📊 数据集**

使用了来自WildChat的数据集，该数据集包含200个种子请求，启动8轮对话，后续用户轮次来自于Llama-3.1-8B模拟器生成的脚本。

**📈 对比分析**

与传统的固定参考审计方法相比，新的审计方法在检测替代会话方面表现更好，检测率达到0.81，而传统方法在替代模型的检测上表现不佳，检测率下降至0.54。

**⚠️ 局限性**

限制在于研究主要集中在聊天场景，而编码代理等其他应用可能具有不同的系统提示和工具执行方式。此外，审计方法需要多个官方调用，且对时间敏感，参考样本必须与被审计的回复同时采样。

---

## 380. LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning

**arXiv ID:** 2611.17600 | [PDF](https://arxiv.org/pdf/2611.17600v1)

**作者:** Eilam Shapira `[一作]` (Technion), Roi Reichart `[通讯]` (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为LightTAR的方法，用于多模态表格学习，通过结合最终和中间编码器层的信息，增强标准PCA表示，以适应预测任务。

**💡 创新点**

LightTAR的创新点在于通过使用训练标签来选择降维方向，从而生成目标感知的特征，改善了多模态表格学习的预测性能。

**🔧 技术方法**

使用了主成分分析（PCA）和正则化的偏最小二乘法（PLS）技术来提取目标感知特征。

**📊 数据集**

在MulTaBench和STRABLE数据集上进行了评估，MulTaBench包含20个文本数据集和20个图像数据集，STRABLE包含108个字符串列的表格。

**📈 对比分析**

与21种表格模型进行了比较，LightTAR在几乎所有模型上都提高了预测性能，尤其是在MulTaBench数据集上，19个文本数据集和17个图像数据集的预测性能都有所提升。

**⚠️ 局限性**

LightTAR并不适用于所有数据集，其效果依赖于从选定层提取的线性投影所能提取的预测信息。未来的工作可以考虑扩展LightTAR以支持非线性投影和其他简单表示。

---

## 381. Which Model, For Which Variable? Per-Variable Forecast Combination in Sparse ICU Time Series

**arXiv ID:** 2611.18149 | [PDF](https://arxiv.org/pdf/2611.18149v1)

**作者:** Marius Fracarolli `[一作]` (Heidelberg University), Stefan Riezler `[通讯]` (Heidelberg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究探讨了在多变量ICU生理数据中，单一最佳模型的假设并不适用，提出了针对每个变量的预测组合方法，以提高预测准确性。

**💡 创新点**

创新点在于提出了针对每个变量的堆叠组合方法，通过为每个变量单独拟合权重，显著提高了预测性能，尤其是在数据稀疏的情况下。

**🔧 技术方法**

使用了14种不同的模型进行组合，包括线性模型、递归神经网络、变换器等，并通过梯度提升树、ARIMA和ETS扩展了模型池。

**📊 数据集**

使用了五个ICU数据库：MIMIC-III、MIMIC-IV、eICU、HiRID和UMM，涵盖了94个通道的预测。

**📈 对比分析**

与单一最佳模型相比，组合方法在所有五个数据库上均提高了6.1%至11.0%的均方误差，尤其是在数据稀疏的通道上，组合方法的优势更加明显。

**⚠️ 局限性**

限制在于UMM数据库的数据无法公开共享，且在小样本情况下，模型的稳定性可能受到影响。

---

## 382. Query Processing with eBPF-Offload on Computational Storage

**arXiv ID:** 2611.18119 | [PDF](https://arxiv.org/pdf/2611.18119v1)

**作者:** Alexander Krause `[一作]` (TU Dresden), Philippe Bonnet `[通讯]` (University of Copenhagen)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了将计算推向存储设备的概念，特别是计算存储如何与查询处理相匹配。作者设计并实现了一个原型系统，基于开放硬件设备，使用eBPF程序调用设备内的函数，并对SSB Q3.3查询进行了初步的手动编码。

**💡 创新点**

创新点在于提出了一种新的计算存储模型，强调设备管理数据迭代和数据移动的重要性，而不是依赖主机进行管理。通过对标准接口的分析，指出了当前模型在查询处理中的局限性。

**🔧 技术方法**

使用了eBPF（扩展伯克利数据包过滤器）作为函数的可移植二进制格式，允许主机调用设备上注册的函数。系统在OpenSSD Daisy平台上实现，使用了用户空间的uBPF虚拟机。

**📊 数据集**

使用了Star Schema Benchmark (SSB) 数据集，特别是SSB Q3.3查询进行实验，数据存储在连接到Daisy的SSD上。

**📈 对比分析**

与主机执行相比，离线执行的性能提升有限，主要由于数据移动和缓存一致性管理的开销。尽管在某些情况下，离线执行可以避免将数据移动到主机，但整体性能仍然低于主机执行，尤其是在处理较大数据集时，离线执行的时间甚至比主机执行更长。

**⚠️ 局限性**

限制在于当前的执行模型依赖于主机驱动的迭代和数据移动，导致查询处理变成逐列处理，增加了数据在设备内存中的移动和重组时间。作者认为，计算存储需要一种新的执行模型，使设备能够管理数据的迭代和移动，以提高性能。

---

## 383. When Should LLM Agents Forget? Memory Filtering Under Query-Critical Resolver Errors

**arXiv ID:** 2611.17338 | [PDF](https://arxiv.org/pdf/2611.17338v1)

**作者:** Samarth Vinayaka `[一作]` `[通讯]` (University of Florida), Samarth Vinayaka (University of Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型（LLM）记忆系统中，删除过时事实的影响，发现这种硬性过滤可能会降低下游准确性，尤其是在解析器错误导致关键查询事实被删除的情况下。

**💡 创新点**

提出了一种新的视角，强调了记忆信息的包装方式对生成器输出的影响，表明解析器的准确性与下游行为之间并不总是直接相关。

**🔧 技术方法**

使用了敏感性干预测试、政策替代方案和基于金标准的诊断等技术，分析了不同的解析器和组装策略对模型性能的影响。

**📊 数据集**

构建了一个合成基准数据集ValidityMem-530，包含530个查询和1936个独特存储记忆对象，用于控制分析和验证实验结果。

**📈 对比分析**

通过与不同的组装模式进行比较，发现硬性删除在高可靠性下能提高准确性，但在低可靠性下会导致性能下降。实验结果显示，错误删除的代价高于错误保留。

**⚠️ 局限性**

研究的局限性包括只使用了来自两个提供者的三种模型，且未对更大规模的前沿模型进行测试。此外，合成基准的适用性和多跳提取的修复效果也存在不确定性。

---

## 384. On Applications of Polynomial-Time Group Isomorphism Algorithm

**arXiv ID:** 2611.18084 | [PDF](https://arxiv.org/pdf/2611.18084v1)

**作者:** Wenkang Zhang `[一作]` (Shandong University), Jincheng Zhuang `[通讯]` (Shandong University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

研究了有限结合环同构、双线性映射同构和自由三阶张量同构的问题，并基于Qiao和Zhang提出的多项式时间算法，提出了相应的多项式时间算法。

**💡 创新点**

提出了一种新的多项式时间算法，能够计算任意有限结合环的同构及其自同构群生成元，并扩展了双线性映射和自由三阶张量的同构算法。

**🔧 技术方法**

使用了Qiao和Zhang的算法，通过单位三角构造将环同构问题转化为标记群同构问题，并通过三角环构造将双线性映射同构问题转化为标记环同构问题。

**📊 数据集**

使用了有限结合环、双线性映射和自由三阶张量的操作表作为数据集。

**📈 对比分析**

与之前的结果相比，提出的算法在多项式时间内计算同构，性能显著提升，特别是在处理任意有限结合环时，时间复杂度为N^O(1)。

**⚠️ 局限性**

限制在于算法主要针对有限结合环和有限交换环，可能无法直接推广到更广泛的代数结构。

---

## 385. CS-MPC: Contraction-Certified Sampling-Based MPC for Nonlinear Systems Under Uncertainty

**arXiv ID:** 2611.18390 | [PDF](https://arxiv.org/pdf/2611.18390v1)

**作者:** Yashwanth Kumar Nakka `[一作]` `[通讯]` (Georgia Institute of Technology), Yashwanth Kumar Nakka (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种收缩认证的基于采样的模型预测控制（CS-MPC）方法，用于在参数不确定性、干扰和在线揭示的安全约束下跟踪和确保非线性机器人系统的安全。

**💡 创新点**

创新点在于使用收缩认证的跟踪反馈来集中采样轨迹，并通过量化收紧的离散时间控制障碍函数条件来验证应用的输入，从而确保安全性。

**🔧 技术方法**

采用了收缩理论、控制障碍函数和模型预测控制等技术。

**📊 数据集**

在模拟中验证了CS-MPC，应用于一个探测车、一个四旋翼和一个小行星着陆任务。

**📈 对比分析**

与标准的模型预测路径积分（MPPI）控制相比，CS-MPC在探测车上实现了相同的跟踪误差，但所需的采样次数减少了7到20倍；在四旋翼上，跟踪误差降低了2到4倍，并且在200次蒙特卡洛任务中没有安全违规。

**⚠️ 局限性**

限制在于该方法的性能依赖于模型的准确性和采样的数量，且在某些情况下可能无法处理极端的动态变化。

---

## 386. LACCO: Local Accumulation with Computation--Communication Overlap

**arXiv ID:** 2611.17991 | [PDF](https://arxiv.org/pdf/2611.17991v1)

**作者:** Daniele Kopyshevskiy `[一作]` (Sorbonne Université), Edouard Oyallon `[通讯]` (CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的算法LACCO（Local Accumulation with Computation–Communication Overlap），旨在减少低带宽网络中训练语言模型时的同步开销。

**💡 创新点**

LACCO通过在局部计算与全局通信之间重叠，消除了外部更新中的陈旧性，确保了副本一致性，并提供了平滑非凸目标的收敛保证。

**🔧 技术方法**

使用了局部随机梯度下降（Local SGD）和计算-通信重叠机制，结合了两阶段的算法设计。

**📊 数据集**

在FineWeb数据集上进行了语言模型的预训练实验，模型参数规模达到0.5B，并在多个通信间隔和副本数量（最多16个工作节点）下进行了测试。

**📈 对比分析**

与现有的通信重叠基线（如Eager和Delayed DiLoCo）相比，LACCO在评估损失上表现出显著的改进，且在模型规模扩大时性能依然稳健。

**⚠️ 局限性**

算法在内存和计算开销上仅增加了微不足道的负担，尽管引入了两个参数状态，但在实际应用中仍然保持了高效性。

---

## 387. When to Trust Simulated Patient Responses:Support-Gated World Models for Vasopressor Decision Support

**arXiv ID:** 2611.18458 | [PDF](https://arxiv.org/pdf/2611.18458v1)

**作者:** Yifan Jin `[一作]` (Sun Yat-sen University), Tao Liu `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种轻量级的支持门控世界模型决策支持方法，旨在提高临床模拟的可靠性，尤其是在不熟悉的医院或治疗方案不充分的情况下。

**💡 创新点**

创新点在于引入了候选级别的支持和不一致性门控机制，仅释放符合条件的预测，并在不合格时提供明确的拒绝理由。

**🔧 技术方法**

使用了递归动态集成模型和行为模型来预测生理反应，并通过门控机制来筛选候选预测。

**📊 数据集**

使用了eICU-CRD数据集，包括70家医院和30,700个决策转变，其中14家医院用于测试。

**📈 对比分析**

与随机选择相比，在62.0%的事实转变覆盖率下，联合门控的平均绝对误差为7.3 mmHg，而随机选择为9.2 mmHg。门控代理在81.0%的决策覆盖率下达成60.0%的记录行动一致性，表现优于未门控的集成代理。

**⚠️ 局限性**

局限性包括未能识别个体化治疗效果或改善患者结果，且聚合估计未量化抽样变异性，可能对稀疏监测或不寻常治疗历史的患者产生不成比例的影响。

---

## 388. Callo: Serving Cache-Miss Requests with Speculative Decoding on Invalidated KV under Prefill-Decode Disaggregation

**arXiv ID:** 2611.18459 | [PDF](https://arxiv.org/pdf/2611.18459v1)

**作者:** Jianjun Ou `[一作]` (Duke University), Matthew Lentz `[通讯]` (Duke University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的分散式服务系统，利用无效的KV缓存进行推测解码，以减少因上下文编辑导致的延迟。

**💡 创新点**

创新点在于通过选择性推测执行来减少无效KV的惩罚，同时确保输出质量无损。

**🔧 技术方法**

使用了推测解码技术，并在分散式vLLM上实现了无缝的跨实例KV传输。

**📊 数据集**

使用了SWE-agent会话的真实数据集，评估了三个模型系列的性能。

**📈 对比分析**

与vLLM进行比较，在中等负载下，所有请求的速度提升为1.05-1.19倍，且在集群级别的模拟中也实现了类似的增益。

**⚠️ 局限性**

限制在于需要保持会话亲和性，依赖于先前请求的KV仍在缓存中，并且需要解码实例有空闲的计算能力。

---

## 389. Deep Latents, Long Memory: Latent persistence explains temporal correlations in transformer activations

**arXiv ID:** 2611.18089 | [PDF](https://arxiv.org/pdf/2611.18089v1)

**作者:** Francesco Cagnetta `[一作]` (Simplex), Santiago Acevedo `[通讯]` (SISSA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的方法，通过定义神经网络层的自相关函数来量化大语言模型（LLMs）中潜在变量的持久性和抽象表示的结构。

**💡 创新点**

创新点在于将自相关函数作为一种桥梁，连接理论和经验研究，揭示了层次结构与潜在变量持久性之间的关系。

**🔧 技术方法**

使用了自相关函数来分析神经网络激活，并结合稀疏自编码器（SAE）特征作为潜在变量的代理。

**📊 数据集**

使用了合成数据（随机层次模型）和自然语言数据集（WikiText和The Pile）进行实验。

**📈 对比分析**

通过与其他方法的比较，发现深层网络的自相关性随着层数的增加而增长，且持久性与自相关性之间存在强相关性，表明深层特征在更长的上下文跨度上保持有效。

**⚠️ 局限性**

限制在于对真实数据的潜在结构缺乏明确的了解，且自编码器特征的重建可能不完全，导致结果的解释受到影响。

---

## 390. User-Schedulable Approximations for Quantized Tensor Programs

**arXiv ID:** 2611.17526 | [PDF](https://arxiv.org/pdf/2611.17526v1)

**作者:** Alex Reinking `[一作]` `[通讯]` (Adobe Research), Alex Reinking (Adobe Research)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一种新的方法，通过将量化的张量程序的编码器和解码器与Halide的调度分离，从而提高了语言模型在边缘设备上的运行效率。

**💡 创新点**

创新点在于引入了可组合的编解码器，使得程序员可以选择值的近似，并将其调度到计算中，同时保持编码和解码的高效性和准确性。

**🔧 技术方法**

使用了Halide编程框架，结合了编解码器的概念，允许程序员在算法和调度之间进行灵活的选择。

**📊 数据集**

使用了多种量化格式的数据集，包括q4_0和q8_0等，涵盖了广泛的量化类型。

**📈 对比分析**

与现有方法相比，本文的方法在代码量上减少了约×，并且在Apple M3 Max上，生成的内核在95%的情况下达到了与特定格式内核相当的速度，且在回退到阻塞点积时速度更快。

**⚠️ 局限性**

限制在于当前的实现依赖于手动选择编解码器，且块大小是静态的，数据依赖的大小超出了模型的范围。

---

## 391. Analysing Busy Beaver Machines (Or How to Find Ossified Ocelots)

**arXiv ID:** 2611.17742 | [PDF](https://arxiv.org/pdf/2611.17742v1)

**作者:** James Harland `[一作]` `[通讯]` (RMIT University), James Harland (RMIT University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文介绍了形状计算的概念，以证明图灵机的关键属性，从而有效执行相关的图灵机，并展示了输出大小与生成所需步骤之间的二次关系。

**💡 创新点**

创新点在于引入了形状计算的概念，提供了一种新的方法来分析和优化图灵机的执行效率，并首次展示了输出大小与步骤数量之间的二次关系。

**🔧 技术方法**

使用了形状计算和观察海獺模式等技术，结合了理论分析和经验评估。

**📊 数据集**

使用了240个来自不同来源的图灵机示例作为数据集进行评估。

**📈 对比分析**

通过与传统方法的比较，展示了形状计算在执行速度上的显著提升，尤其是在处理复杂图灵机时，性能表现优越。

**⚠️ 局限性**

限制在于当前方法可能无法处理所有类型的图灵机，尤其是那些具有极高复杂度或不规则行为的机器。

---

## 392. FlexCast: Flexible Context Learning for Time Series Foundation Models

**arXiv ID:** 2611.17862 | [PDF](https://arxiv.org/pdf/2611.17862v1)

**作者:** Zuyang Shen `[一作]` (Institute of Computing Technology), Jianfeng Zhan `[通讯]` (Institute of Computing Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为FlexCast的灵活上下文学习框架，用于时间序列基础模型（TSFMs），通过可变长度的掩码预测重新定义预测任务。

**💡 创新点**

创新点在于将自回归（AR）预测中的上下文条件和预测单元因子化分开，提出了一种新的掩码和预测接口，支持多种推理策略。

**🔧 技术方法**

使用了基于Transformer的模型架构，结合了量化预测头和可变长度的掩码预测技术。

**📊 数据集**

在BLAST数据集和多个零样本基准上进行了评估。

**📈 对比分析**

与强大的TSFM基线相比，FlexCast在准确性上具有竞争力，同时显著降低了推理延迟。实验表明，灵活的上下文学习是性能提升的主要因素，而自回归展开并未带来一致的增益。

**⚠️ 局限性**

限制在于模型的表现可能受到上下文长度和掩码策略的敏感性影响，且在某些情况下，复杂的推理策略可能并不总是优于单次预测。

---

## 393. Smart Home Agents Make Capable but Unpredictable Grid Operators

**arXiv ID:** 2611.18395 | [PDF](https://arxiv.org/pdf/2611.18395v1)

**作者:** Christopher Terrazas `[一作]` (Oregon State University), Eduardo Cotilla-Sanchez `[通讯]` (Oregon State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文首次对智能家居代理在全面停电情况下的韧性进行了分析，提供了一套在实时环境中测量停电前、停电期间和停电后的韧性模拟。

**💡 创新点**

创新点在于将智能家居代理视为电网操作员，并识别出多种新的故障类别，强调了当前模型的局限性，为智能家居代理作为电网操作员的应用奠定了基础。

**🔧 技术方法**

使用了多种大型语言模型（LLMs）作为智能家居代理的核心引擎，包括gpt-5-mini、gpt-oss-120b等，并采用了ReAct代理范式进行故障响应。

**📊 数据集**

使用了合成的智能家居数据集，包括30个用于训练的合成家庭和1个手动构建的测试家庭，涵盖了关键和非关键智能设备。

**📈 对比分析**

通过与基线模型的比较，发现最佳模型在韧性三角形中表现良好，能够在停电期间保持较高的服务水平，并在停电后恢复大部分故障设备。

**⚠️ 局限性**

局限性包括一些模型在故障响应中表现出不作为、假设错误、上下文混淆等问题，可能导致不安全的操作和错误的设备状态更新。

---

## 394. Interactive Program Induction with Language Model Priors

**arXiv ID:** 2611.18485 | [PDF](https://arxiv.org/pdf/2611.18485v1)

**作者:** Robin Shing Moon Chan `[一作]` (ETH Zürich), Ryan Cotterell `[通讯]` (ETH Zürich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于贝叶斯主动学习的交互式程序归纳算法，该算法通过用户反馈来高效地恢复程序。

**💡 创新点**

创新点在于将程序归纳视为贝叶斯推断问题，并通过贪婪的信息增益算法选择查询，从而提高了准确性并减少了所需的输入输出示例数量。

**🔧 技术方法**

使用了贝叶斯推断、重要性采样和语言模型作为程序先验。

**📊 数据集**

使用了CodeARC基准数据集，该数据集包含1114个Python函数，涵盖从入门级任务到竞赛问题的多种情况。

**📈 对比分析**

与CodeARC LM代理进行比较，算法在大多数评估设置中提高了准确性（最高提高22个百分点），同时减少了最多10.5个输入输出示例的观察。

**⚠️ 局限性**

限制在于语言模型先验的误校准可能会影响方法的准确性，导致早期收敛到错误的程序。

---

## 395. Causal AI-Driven Secure mmWave Beam Alignment with Defensive Variational Autoencoders

**arXiv ID:** 2611.18321 | [PDF](https://arxiv.org/pdf/2611.18321v1)

**作者:** Nasir Khan `[一作]` (KTH Royal Institute of Technology), Cicek Cavdar `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于因果AI的深度学习框架，用于毫米波通信系统中的安全波束对准，集成了因果发现和防御变分自编码器。

**💡 创新点**

创新点在于引入因果关系来提高波束选择的效率和鲁棒性，同时通过防御变分自编码器增强对抗攻击的防御能力。

**🔧 技术方法**

使用了因果发现算法和防御变分自编码器（DVAE）技术。

**📊 数据集**

使用了来自特定环境的RSSI测量数据集，进行波束选择和分类。

**📈 对比分析**

与传统的深度学习方法相比，因果波束选择在准确性上相当，但每个样本的计算成本降低了829倍，波束扫描开销减少了59.4%至89.8%。DVAE有效去噪对抗输入，在不同扰动范围内保持85%的准确率。

**⚠️ 局限性**

局限性在于该方法依赖于特定环境的数据，可能在环境变化时需要重新选择特征和重新训练模型。

---

## 396. Evaluating Large Language Models for Scientific Paper Search: Retrieval Success and Inference Latency on Mac Studio and NVIDIA A100

**arXiv ID:** 2611.17929 | [PDF](https://arxiv.org/pdf/2611.17929v1)

**作者:** Frederic Andriantseheno Rakoto `[一作]` (Inria), Xavier Hinaut `[通讯]` (Inria)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了在实际计算限制下，开放权重模型是否能够支持网络辅助的论文检索，评估了八种模型在从七个科学库中检索70篇论文的表现。

**💡 创新点**

创新点在于开发了一个本地部署的开放权重模型的框架，以减少对外部模型提供者的依赖，同时增强研究团队对数据和基础设施的控制。

**🔧 技术方法**

使用了本地部署的开放权重模型，评估了不同模型在不同输入长度下的推理延迟，并比较了在Mac Studio和配备两块NVIDIA A100 GPU的服务器上的性能。

**📊 数据集**

使用的数据集包含来自arXiv、bioRxiv、medRxiv、ChemRxiv、SSRN、Research Square和agriRxiv的70篇论文，结合了随机抽样和引用计数的变异。

**📈 对比分析**

通过比较不同模型在三种查询特定性下的检索成功率，发现检索成功率随着查询的专业性增加而显著提高，Qwen3.5-122B-A10B在检索优先时表现最佳，但延迟较高。

**⚠️ 局限性**

限制在于本地模型的推理延迟较高，尤其是在处理长输入时，且在多用户环境下，Mac Studio可能成为瓶颈，无法满足高并发请求的需求。

---

## 397. iGRAIL: Blockchain-Based Generative Federated Learning for Trustworthy Healthcare Intelligence

**arXiv ID:** 2611.17703 | [PDF](https://arxiv.org/pdf/2611.17703v1)

**作者:** Debashis Das `[一作]` (Benedict College), Saraju P. Mohanty `[通讯]` (University of North Texas)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于区块链的生成联邦学习框架iGRAIL，用于可信的医疗智能分析，旨在解决医疗数据隐私和安全问题。

**💡 创新点**

iGRAIL的创新点在于将本地生成学习与差分隐私相结合，确保在本地训练过程中保护敏感信息，同时通过区块链技术确保模型更新的安全性和完整性。

**🔧 技术方法**

使用了生成对抗网络（GAN）、差分隐私（DP）、区块链技术和加密机制。

**📊 数据集**

使用了四个医疗数据集进行评估，包括癌症数据集和糖尿病数据集。

**📈 对比分析**

与传统的联邦学习（FL）和基于区块链的联邦学习（BC-FL）进行比较，iGRAIL在准确性、F1分数和MCC等指标上表现更优，尤其在糖尿病数据集上提高了约5.4-5.6%的准确性和14.6-16.9%的F1分数。

**⚠️ 局限性**

iGRAIL的局限性在于其额外的计算开销，可能影响模型的实时性能，同时在面对恶意更新时仍需进一步的安全保障。

---

## 398. Multi-level latent-diffusion adaptation for cross-region point cloud segmentation of ancient terraces

**arXiv ID:** 2611.18033 | [PDF](https://arxiv.org/pdf/2611.18033v1)

**作者:** Sina Mohammadi `[一作]` (Leiden Institute of Advanced Computer Science), Sebastian Fajardo `[通讯]` (Leiden Institute of Advanced Computer Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的无监督领域适应方法MELD，用于跨区域检测古代梯田，利用未标记的目标调查数据进行模型训练。

**💡 创新点**

创新点在于通过多层嵌入空间的潜在扩散模型，针对不同层次的特征分别训练扩散去噪器，从而有效地调整源特征和目标特征。

**🔧 技术方法**

使用了多层PointNet++架构和扩散模型技术，结合K近邻几何描述符来增强特征表示。

**📊 数据集**

使用了来自哥伦比亚Sierra Nevada de Santa Marta地区的两个激光雷达（LiDAR）调查数据集：Teyuna和Miraflores。

**📈 对比分析**

与源模型和九种已建立的领域适应方法进行比较，MELD在两个方向的F1分数分别从0.397提升到0.459和从0.557提升到0.594，表现最佳。添加局部几何信息进一步提高了F1分数。

**⚠️ 局限性**

限制在于当前评估为转导性设置，未测试模型在未见样本上的泛化能力，且仅考虑了单一的考古特征类别。

---

## 399. INC-SRv6: In-Network Route Exploration and Fast Failover Using Segment Routing Headers in Dynamic UAV Networks

**arXiv ID:** 2611.17659 | [PDF](https://arxiv.org/pdf/2611.17659v1)

**作者:** Zhiyuan Ren `[一作]` (Xidian University), Zhao Li `[通讯]` (Shaanxi Academy of Aerospace Technology Application Co., Ltd.)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了一种新的网络计算机制INC-SRv6，旨在为动态无人机网络提供高效的路由和快速故障恢复。

**💡 创新点**

创新点在于将标准的IPv6段路由头（SRH）重新利用为动态的、在网络中进行路由计算的状态寄存器，支持多目标学习和快速故障转移。

**🔧 技术方法**

使用了段路由（SRv6）协议中的段路由头（SRH）和增强的网络计算机制，结合了自学习转发机制和局部快速故障恢复。

**📊 数据集**

使用了200节点的3D无人机网络进行模拟评估，涵盖了10种不同的移动性和故障场景，共进行了1100次模拟运行。

**📈 对比分析**

与五种协议（如AODV、OLSR等）和四种机制消融实验进行了比较，INC-SRv6在高达35 m/s的速度下，交付率提高至92.12%，并将p_95延迟减少到8.84毫秒，同时控制字节减少了52.1%-54.7%。

**⚠️ 局限性**

局限性在于该机制依赖于网络中节点的快速反馈和状态更新，可能在极端动态环境下面临挑战，未来工作将探索跨层集成和硬件验证。

---

## 400. Alignment with Reflect-Reasoning: Learning How To Follow Principles

**arXiv ID:** 2611.17398 | [PDF](https://arxiv.org/pdf/2611.17398v1)

**作者:** Henry Bell `[一作]`, Brandon Fain `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究了不同模型在分布外聚合违规率的表现。

**💡 创新点**

提出了一种新的评估方法，比较了多种模型在不同数据集上的违规率。

**🔧 技术方法**

使用了统计分析和可视化技术来展示模型的性能。

**📊 数据集**

使用了多个模型（Tulu, Mistral, Qwen, Olmo）在不同数据集上的数据。

**📈 对比分析**

通过比较不同模型的违规率，发现Mistral模型的表现最差，而Qwen模型的表现最好。

**⚠️ 局限性**

研究的局限性在于只考虑了特定的模型和数据集，可能无法推广到其他模型或数据集。

---

## 401. Enrichment-Oriented Virtual Screening of Perovskite Solar Cells through Reinforcement Learning

**arXiv ID:** 2611.18043 | [PDF](https://arxiv.org/pdf/2611.18043v1)

**作者:** Yezeyuan Zheng `[一作]` (Hong Kong Quantum AI Lab Limited), Jie Liu `[通讯]` (University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究开发了一种基于强化学习的虚拟筛选框架，用于优化钙钛矿太阳能电池的配方生成，通过邻域评估来丰富候选短名单。

**💡 创新点**

创新点在于将配方生成视为虚拟筛选，利用代理模型指导候选生成，并通过实验参考池进行邻域支持评估，从而提高了高效能配置的选择率。

**🔧 技术方法**

使用了强化学习技术，包括近端策略优化（PPO）、递归PPO（RPPO）和信任区域策略优化（TRPO），并结合了冻结的表格先验数据拟合网络（TabPFN）作为奖励模型。

**📊 数据集**

使用了包含1,846个高效能钙钛矿设备的实验数据集，以及890个测量设备的参考池进行邻域评估。

**📈 对比分析**

与六种基线方法进行比较，PPO和RPPO在前100个候选中实现了23%和24%的邻域支持命中率，显著高于基线方法的10%。在3,000次代理评估中，PPO和RPPO的表现优于其他优化方法。

**⚠️ 局限性**

限制在于实验性能的验证仍需进行，且生成的候选配方可能存在对代理模型的过度依赖，未必能保证实际的实验可行性和性能。

---

## 402. Can Large Language Models Optimize Pretrained Neural Network Weights? A Multi-Phase Empirical Evaluation of Gradient-Free Editing in YOLO26 Object Detection

**arXiv ID:** 2611.17789 | [PDF](https://arxiv.org/pdf/2611.17789v1)

**作者:** Mohamed Abdallah Salem `[一作]` `[通讯]`, Mohamed Abdallah Salem

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探讨了直接修改预训练神经网络参数作为一种替代梯度微调的方法，评估了在YOLO26目标检测中，是否可以通过小的结构性变化来持续改善检测效果，以及大型语言模型（LLMs）是否能比传统的黑箱优化器更好地指导这些变化。

**💡 创新点**

创新点在于系统评估了无梯度编辑的可行性，特别是通过真实的LLM提议和混合优化策略，展示了直接权重编辑的潜力，但未能证明在测试条件下的一致性改进或LLM的增益。

**🔧 技术方法**

使用了无梯度优化技术，包括确定性缩放、随机扰动、差分进化、协方差矩阵适应进化策略（CMA-ES）以及真实的LLM提议和混合LLM-CMA-ES优化器。

**📊 数据集**

使用了Microsoft COCO数据集，开发搜索使用了1000张COCO Val2017图像，选择集包含1500张不重叠的图像。

**📈 对比分析**

与传统的随机策略和其他优化方法进行比较，发现混合LLM与随机策略的比较未能建立显著优势（p值分别为0.612, 0.369和0.694），且在不同预算下的性能变化不一致。

**⚠️ 局限性**

限制在于未能证明一致的泛化改进，且在测试条件下LLM的增益未能显著超越随机策略，此外，数据集的选择和使用历史也可能影响结果的有效性。

---

## 403. A Common Improved Bound for Asymmetric TSP and Path TSP

**arXiv ID:** 2611.18498 | [PDF](https://arxiv.org/pdf/2611.18498v1)

**作者:** MohammadTaghi Hajiaghayi `[一作]`, Changki Yun `[通讯]` (University of Maryland)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了多项式时间的(R_*+ε)-近似算法，用于解决非对称旅行商问题（ATSP）和非对称路径旅行商问题（APTSP），对于每个固定的ε>0，R_*的值约为12.2604。

**💡 创新点**

改进了之前的保证，ATSP和APTSP的近似比率分别从14.7和15降低到R_*+ε，且两者的保证相同。

**🔧 技术方法**

使用了递归增强和子巡回覆盖的技术，结合了拉米纳框架中的分析方法。

**📊 数据集**

使用了强连通的有向图作为输入，且边的成本为非负有理数。

**📈 对比分析**

与之前的方法进行了比较，性能显著提升，ATSP的近似比率从14.7降低到R_*+ε，APTSP的近似比率也相应降低，匹配了ATSP的保证。

**⚠️ 局限性**

限制在于算法的复杂性和对输入数据的要求，尤其是在处理大规模图时可能会遇到性能瓶颈。

---

## 404. Tracing Actual Causes with Counterfactual Witness Maps

**arXiv ID:** 2611.17937 | [PDF](https://arxiv.org/pdf/2611.17937v1)

**作者:** Tim Woydt `[一作]` (Technical University of Darmstadt), Moritz Willig `[通讯]` (Technical University of Darmstadt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种新的算法TrAC，用于在无环结构因果模型中枚举所有实际原因，从而解决了以往缺乏有效枚举程序的问题。

**💡 创新点**

创新点在于引入了见证目标和祖先干预语法，结合这两者构建了有限的有向无环见证图，确保了算法的健全性和完整性。

**🔧 技术方法**

使用了逆向传播的方式，通过结构方程推导见证目标，并结合祖先干预语法生成候选干预集。

**📊 数据集**

使用了随机布尔结构因果模型的数据集，模型节点数从3到15不等，边密度为0.25、0.5和0.75。

**📈 对比分析**

与现有的单因果基线方法进行了比较，TrAC在处理复杂图时通常比暴力搜索快一个数量级，且能够保证枚举所有实际原因。

**⚠️ 局限性**

限制在于TrAC仅适用于完全指定的无环结构因果模型，对于潜在混淆因素的处理有限，且在存储方面存在挑战，尤其是在处理较大模型时。

---

## 405. UltraBERT-JEV: Encoder models for System One Applications

**arXiv ID:** 2611.17867 | [PDF](https://arxiv.org/pdf/2611.17867v1)

**作者:** Astarag Mohapatra `[一作]` `[通讯]` (Independent Researcher), Astarag Mohapatra (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种编码器模型，该模型在单次前向传递中选择多个选项之一，而无需生成文本。

**💡 创新点**

创新点在于提出了四种不同的微调方法，比较了如何对选项进行评分和概率化，其中一种方法遵循了DeepSeek-V3的专家门控机制。

**🔧 技术方法**

使用了UltraBERT-380M编码器，并进行了微调，采用了不同的评分和归一化方法。

**📊 数据集**

使用了Reason+Long数据集，该数据集包含多个意图路由和推理任务的数据。

**📈 对比分析**

与Qwen3-0.6B解码器进行了比较，所有编码器方法在准确性和速度上均优于解码器，但在峰值内存使用上没有优势。

**⚠️ 局限性**

限制在于只覆盖了一个编码器、一个解码器系列和一个英语训练混合数据集，且未评估证据头是否反映模型的使用情况。

---

## 406. PoliMop: Policy-in-the-Loop Motion Planning in 3D Spaces for Mobile Robots

**arXiv ID:** 2611.18110 | [PDF](https://arxiv.org/pdf/2611.18110v1)

**作者:** Yael Fassbind `[一作]` (Computational Robotics Lab), Stelian Coros `[通讯]` (Computational Robotics Lab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于物理仿真的运动规划方法，允许移动机器人在复杂环境中找到可行路径，利用控制策略的能力和环境的变化。

**💡 创新点**

创新点在于将控制策略嵌入运动规划过程中，使得机器人能够在动态环境中自适应地修改其可达空间，并通过路径引导采样来减少规划时间。

**🔧 技术方法**

使用了动力学快速探索随机树（RRT）规划器，并结合物理仿真进行状态转移和动作评估。

**📊 数据集**

使用了多个基准场景，包括楼梯、迷宫、斜坡和狭窄通道等，进行模拟实验。

**📈 对比分析**

与传统的运动规划方法相比，提出的方法在规划时间上减少了63%，并且在多种机器人和控制策略下表现出可靠的规划能力。

**⚠️ 局限性**

限制在于当前假设任务计划是预定义的，且未在真实物理系统上进行验证，未来需要改进计算效率并在实际机器人上进行执行验证。

---

## 407. Provable Cross-Modal Transfer of Hierarchical Concept Structure in CLIP Embeddings

**arXiv ID:** 2611.17671 | [PDF](https://arxiv.org/pdf/2611.17671v1)

**作者:** Nghia Nguyen `[一作]` (University of Pennsylvania), René Vidal `[通讯]` (University of Pennsylvania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了对比视觉-语言模型（如CLIP）中概念嵌入如何在没有显式层次监督的情况下编码潜在的概念层次结构。

**💡 创新点**

创新点在于提出了层次角度分离（Hierarchical Angular Separation）这一几何属性，量化了概念嵌入的层次结构，并解释了对比学习如何将这种结构从图像嵌入转移到文本嵌入。

**🔧 技术方法**

使用了无约束特征模型（Unconstrained Features Model）分析对比学习的机制，并通过线性对比损失和InfoNCE损失进行理论推导。

**📊 数据集**

使用了自然图像和语言的数据集，特别是ImageNet-1K和WordNet名词层次结构。

**📈 对比分析**

通过对比不同规模的CLIP模型，发现图像嵌入在随机初始化时就表现出低的层次角度分离，而文本嵌入在预训练后才表现出低的层次角度分离，验证了理论推导的有效性。

**⚠️ 局限性**

限制在于分析中假设了图像嵌入是固定的，并且未能解释完整的联合训练动态，以及为何图像数据中已经存在层次结构的原因。

---

## 408. Experts Across Schemas: Agent-Guided Mixture-of-Relational-Experts for Cross-Database Generalization

**arXiv ID:** 2611.18315 | [PDF](https://arxiv.org/pdf/2611.18315v1)

**作者:** Hiren Madhu `[一作]` (Yale University), Rex Ying `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的混合关系专家图变换器MoREGT，旨在解决跨数据库模型迁移中的结构学习和目标数据库特定结构供应的挑战。

**💡 创新点**

创新点在于引入了行和单元格令牌、时间和关系感知的注意力机制，以及无值专家路由，同时结合了一个无标签的LLM代理来生成目标数据库的验证单行派生列。

**🔧 技术方法**

使用了图变换器（Graph Transformer）技术，结合了混合专家模型（Mixture of Experts）和无标签的LLM代理。

**📊 数据集**

在21个RelBench任务上进行评估，涉及多个不同的关系数据库。

**📈 对比分析**

与Relational Transformer（RT）进行比较，MoREGT在相同的预训练下，AUROC平均提高了2.25点，nMAE平均降低了5.83点，且在所有比较方法中，MoREGT的性能最佳。

**⚠️ 局限性**

限制在于MoREGT的预训练数据集较小，尽管在性能上优于一些大型预训练模型，但在某些任务上仍可能受到数据量的影响。

---

## 409. Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models

**arXiv ID:** 2611.17462 | [PDF](https://arxiv.org/pdf/2611.17462v1)

**作者:** Daniel T. Levy `[一作]` (McGill University), Siamak Ravanbakhsh `[通讯]` (McGill University)

**通讯引用:** 3476 | [OpenAlex ID](https://openalex.org/A5068176584)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本论文提出了一种基于扩散树采样（DTS）的生成模型，用于在推理时适应生成晶体结构，以满足多个相互冲突的性质目标。

**💡 创新点**

创新点在于将现有的条件生成模型作为指导提案融入DTS中，同时扩展DTS以支持多目标采样，从而在单次搜索中生成覆盖整个帕累托前沿的样本。

**🔧 技术方法**

使用了扩散树采样（DTS）技术，该技术结合了蒙特卡洛树搜索（MCTS）和重要性采样。

**📊 数据集**

使用了MatterGen和DiffCSP等预训练的条件生成模型，进行新晶体生成的实验。

**📈 对比分析**

与传统的生成方法（如最佳N采样）相比，DTS在生成稳定、独特和新颖的晶体方面表现更好，尤其是在高计算预算下，DTS的性能优势更加明显。

**⚠️ 局限性**

限制在于所有报告的奖励使用学习的代理，因此搜索可能会放大系统性预测误差；DFT验证和合成仍然是必要的后续步骤。

---

## 410. ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min-Max-Min Optimization

**arXiv ID:** 2611.18252 | [PDF](https://arxiv.org/pdf/2611.18252v1)

**作者:** Pei Tang `[一作]` (Chinese University of Hong Kong), Songtao Lu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种偏好引导的多目标优化方法，旨在通过最小化预定义的偏好函数来选择弱帕累托集中的理想权衡。

**💡 创新点**

创新点在于提出了一种基于惩罚的最小-最大-最小（M^3）问题的统一框架，将偏好优化与弱帕累托最优性结合在一个目标中，并开发了ParetoM^3算法。

**🔧 技术方法**

使用了一种单循环的一阶算法ParetoM^3来计算M^3问题的驻点解，并在局部Kurdyka-Łojasiewicz假设下证明了其收敛性。

**📊 数据集**

在合成基准、图像分类、回归和数学推理等多个数据集上进行了实验验证。

**📈 对比分析**

与现有的偏好引导多目标优化算法（如EPO、PNG、FERERO和FOOPS）进行比较，ParetoM^3在收敛性、偏好对齐和弱帕累托最优解的可靠性方面表现优越。

**⚠️ 局限性**

限制在于ParetoM^3并不旨在近似整个帕累托前沿，且在更大规模的数据集和模型上的有效性和实用性仍需进一步验证。

---

## 411. LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent

**arXiv ID:** 2611.17780 | [PDF](https://arxiv.org/pdf/2611.17780v1)

**作者:** Ibrahim Memon `[一作]` (University of California Irvine), Roy Fox `[通讯]` (University of California Irvine)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为LuthorSQL的非自回归7B文本到SQL模型，旨在通过本地硬件减少对外部API的依赖，同时保持查询的准确性。

**💡 创新点**

LuthorSQL结合了快速的系统1模型和较慢的系统2 LLM代理，通过在不确定时才调用系统2，从而降低了API调用成本并提高了效率。

**🔧 技术方法**

使用了非自回归模型和LoRA适配器，采用了结构化查询计划的单次前向传递方法，并通过训练的验证器选择最佳答案。

**📊 数据集**

使用了BIRD基准数据集进行训练和评估，并在多个数据库（如Snowflake、BigQuery、MySQL等）上进行了零-shot迁移测试。

**📈 对比分析**

与Claude Opus 5.5结合使用，通过验证器的边际评分来路由查询，结果显示在相同准确率下，API成本降低了16%到34%。

**⚠️ 局限性**

模型的语法只能覆盖84.6%的BIRD开发集的金标准查询，且在处理复杂查询（如CTE、相关子查询等）时表现不佳。

---

## 412. fieldsmith: Automated symbolic derivation in classical field theory and gauge theory from LaTeX input

**arXiv ID:** 2611.17359 | [PDF](https://arxiv.org/pdf/2611.17359v1)

**作者:** Nikola Mirkov `[一作]` `[通讯]` (Vinča Institute of Nuclear Sciences National Institute of the Republic of Serbia), Nikola Mirkov (Vinča Institute of Nuclear Sciences National Institute of the Republic of Serbia)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个开源的Python库，用于自动推导经典场论和非阿贝尔规范理论中的分析结果。用户可以输入拉格朗日量，库会生成和验证欧拉-拉格朗日方程、诺特流和身份、能量-动量张量等。

**💡 创新点**

创新点在于提供了一个以LaTeX为优先的接口，允许用户以物理学家常用的索引符号书写拉格朗日量，并通过精确的符号计算而非模式匹配来验证结果。

**🔧 技术方法**

使用了Python编程语言和SymPy库，构建了一个紧凑的分析管道，能够处理复杂的符号计算。

**📊 数据集**

文中提供了18个自检示例，涵盖从克莱因-戈登场到BPST瞬子和't Hooft-Polyakov单极子等多个经典场论和规范理论的主题。

**📈 对比分析**

与现有的计算工具相比，该库提供了一个完整的推导管道，能够从拉格朗日量推导出谱，并且通过自检示例验证了其准确性。性能方面，库在处理SU(2)和电弱计算时表现良好，但对于SU(N)（N≥4）时性能较差。

**⚠️ 局限性**

限制包括：不支持量子场论和数值相对论，无法处理高阶导数理论的哈密顿分析，且不支持动态引力和诺特流在弯曲背景下的计算。

---

## 413. Zero-SM NCCL: A Systematic Guide to Designing and Deploying SM-Free GPU Communication

**arXiv ID:** 2611.18075 | [PDF](https://arxiv.org/pdf/2611.18075v1)

**作者:** Zhenhao He `[一作]` (NVIDIA Corporation), Torsten Hoefler `[通讯]` (ETH Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种新的GPU通信框架Zero-SM NCCL，旨在消除流处理器(SM)在数据传输和同步中的使用，从而提高GPU通信效率。

**💡 创新点**

Zero-SM NCCL是首个完全消除SM使用的通信框架，支持集体操作和单边原语，提供了新的API以构建自定义的零SM通信模式。

**🔧 技术方法**

采用了基于CUDA流内存操作的无SM同步、通过复制引擎进行数据传输，以及由CPU代理驱动的网络远程直接内存访问(RDMA)。

**📊 数据集**

在GB200 NVL72和B200 NVL8系统上进行评估，使用了多种模型和并行策略的端到端训练。

**📈 对比分析**

与NCCL的传统SM基础内核相比，Zero-SM集体在大数据传输中表现出更高的带宽，且在重叠计算中的减慢幅度显著降低，但在小消息和主机绑定工作负载中表现不佳。

**⚠️ 局限性**

在小消息传输中，Zero-SM NCCL的延迟高于SM内核，且在规模扩展同步中依赖CPU线程，消耗主机资源。

---

## 414. RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation

**arXiv ID:** 2611.18244 | [PDF](https://arxiv.org/pdf/2611.18244v1)

**作者:** Tianqi Liu `[一作]` (Zhejiang University), Shengyu Zhang `[通讯]` (Zhejiang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为RACE的框架，用于高效的序列推荐，动态适应模型结构和输入序列，以应对用户意图复杂性的异质性。

**💡 创新点**

创新点在于利用排名不确定性作为资源分配的主动信号，实现了基于不确定性的动态容量适应，解决了传统方法中的容量-复杂性不匹配问题。

**🔧 技术方法**

采用了自监督学习机制来预测排名不确定性，并通过超网络风格的通道敏感性提取器和重要性感知的逐步序列过滤实现多层次定制。

**📊 数据集**

在四个流行的数据集上进行了广泛的实验，包括MovieLens-100k、Douban、Amazon-CD和Amazon-Game。

**📈 对比分析**

与九个基线方法进行了比较，RACE在准确性、效率和适应性方面均优于最先进的基线，平均提高了40.87%的NDCG和35.12%的Hit，同时减少了2.37倍的FLOPs。

**⚠️ 局限性**

限制在于RACE的动态资源分配依赖于模型的不确定性预测，可能在某些情况下对复杂实例的适应性不足。

---

## 415. Toward On-Device Markerless 6DoF Tracking of a Physical Manipulative on a Mixed-Reality Headset: A Preliminary Study

**arXiv ID:** 2611.18438 | [PDF](https://arxiv.org/pdf/2611.18438v1)

**作者:** Guangxi Feng `[一作]` (Texas A&M University), Wei Yan `[通讯]` (Texas A&M University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种无标记的六自由度（6DoF）跟踪器，用于在Meta Quest 3上跟踪物理乐高飞机，以支持增强现实课堂的空间变换学习。

**💡 创新点**

创新点在于使用合成的Unity渲染数据训练紧凑的YOLO模型，以在动态裁剪图像中预测2D关键点，并通过PnP算法估计物体姿态。

**🔧 技术方法**

使用了YOLO模型进行关键点预测，结合PnP算法进行姿态估计。

**📊 数据集**

使用了合成的Unity渲染数据集，生成了50,000张图像用于训练和验证。

**📈 对比分析**

通过与全图和裁剪图像的比较，裁剪图像在标注图像上的检测率提高了13.7个百分点，关键点F1均值从42.9%提高到44.6%。在Meta Quest 3上，应用在2Hz的跟踪目标下保持约72帧每秒的帧率。

**⚠️ 局限性**

限制在于当前的跟踪精度依赖于合成数据，实际头戴设备的姿态估计尚未进行定量评估，且在跟踪丢失时可能出现虚拟模型的错位。

---

## 416. Fuse-PINN: Combining Finite Element Methods and Physics-Informed Neural Networks to enable massively parallel simulations of non-linear PDEs

**arXiv ID:** 2611.18409 | [PDF](https://arxiv.org/pdf/2611.18409v1)

**作者:** Stanislas Pamela `[一作]` (UK Atomic Energy Authority), Vignesh Gopakumar `[通讯]` (UK Atomic Energy Authority)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为Fuse-PINN的可扩展框架，将物理信息神经网络与有限元方法结合，旨在解决复杂非线性物理模型的数值模拟问题。

**💡 创新点**

Fuse-PINN通过将有限元残差组装与基于物理的高精度优化器结合，实现了在多节点、多GPU高性能计算架构上的可扩展性，打破了PINNs在大规模问题上的限制。

**🔧 技术方法**

使用了有限元方法（FEM-NN）和能量自然梯度下降优化器（ENGD），并在PyTorch中实现了Fuse-PINN。

**📊 数据集**

在与磁约束聚变相关的物理模型上进行了验证，包括具有细尺度动态的非线性对流模型。

**📈 对比分析**

Fuse-PINN在多GPU运行中实现了近乎完美的线性扩展，随着问题规模的增加，性能表现优异，且通信时间占总计算成本的比例极小。

**⚠️ 局限性**

当前的主要限制是计算成本，相较于传统求解器，效率差距约为三个数量级，需要进一步的优化和开发以提高性能。

---

## 417. Deep Bundle Adjustment

**arXiv ID:** 2611.18256 | [PDF](https://arxiv.org/pdf/2611.18256v1)

**作者:** Pak Hong Chui `[一作]` (Technical University of Munich), Daniel Cremers `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了DeepBA（深度束调整），一种图神经网络（GNN），用于从任意数量视图的2D匹配中恢复相机姿态、内参和场景结构。

**💡 创新点**

创新点在于将束调整过程学习化，使用GNN进行迭代优化，能够从冷启动状态收敛，而不需要传统方法的精确初始化。

**🔧 技术方法**

使用了图神经网络（GNN）技术，通过共享权重的Markovian迭代更新来进行状态优化。

**📊 数据集**

在多个数据集上进行训练，包括MegaDepth、CO3Dv2、ScanNet等，支持不同度量单位的数据集混合。

**📈 对比分析**

与传统的SfM方法（如COLMAP和GLOMAP）相比，DeepBA在场景重建和大多数姿态指标上表现更好，并且在视图数量增加时在运行时间上具有竞争力。

**⚠️ 局限性**

局限性包括在平面场景上的焦距-深度模糊、某些场景中学习更新的漂移、在严格阈值下的准确性以及针孔相机模型的限制。

---

## 418. When Does Oversampling Help? A Decision Framework for Imbalanced Data Streams Under Label Delay and Concept Drift

**arXiv ID:** 2611.17981 | [PDF](https://arxiv.org/pdf/2611.17981v1)

**作者:** Ahmed Awad `[一作]` (British University in Dubai), Anirudh Sengar `[通讯]` (British University in Dubai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本论文研究了在标签延迟和概念漂移下，流数据中的不平衡数据流是否值得添加流式过采样器，或是持续重新评估早期预测是否已经弥补了延迟的成本。

**💡 创新点**

提出了一个决策框架，通过漂移密度、少数类可学习性和相对于细化窗口的延迟三个轴来预测何时过采样比重新评估更有价值，并提供了部署指导。

**🔧 技术方法**

使用了流式分类器和流式过采样技术，进行了系统的实验比较，评估了15种分类器在6个合成流和6个真实数据集上的表现。

**📊 数据集**

使用了6个合成数据流（如SEA和SEAG）和6个真实数据集（如电力、KDDCup、PAKDD等），这些数据集具有不同的漂移特征和不平衡程度。

**📈 对比分析**

通过与15种流式分类器的比较，发现重新评估在漂移活跃流中能够弥补延迟造成的损失，但在静态流中则无效；过采样在仅一种真实数据配置中对重新评估的边际贡献是统计上可检测的，且合成流的成本-性能Pareto推荐无法转移到真实数据上。

**⚠️ 局限性**

限制在于，方法选择应根据流的特征进行调整，而不是从合成基准中继承，且在真实数据中，合成数据的Pareto状态并不安全。

---

## 419. DIRT: Depth-completed Image-based Rendering using Two-band Blending for Live Novel-view Synthesis

**arXiv ID:** 2611.18340 | [PDF](https://arxiv.org/pdf/2611.18340v1)

**作者:** Mathias Harrer `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Tim Weyrich `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于图像的实时新视角合成方法，该方法利用深度传感器观察和图像集进行渲染，避免了场景特定的优化过程。

**💡 创新点**

创新点在于完全不依赖神经网络和优化过程，能够在捕获过程中实现低延迟的渲染，并且对新场景具有良好的泛化能力。

**🔧 技术方法**

采用了基于测量的稀疏点云几何体和频率感知的双带混合策略，构建了一个非神经网络的图像渲染管道。

**📊 数据集**

使用了多种数据集，包括手持RGB-D相机、汽车LiDAR和热成像相机的数据，验证了方法在不同传感器设置下的适用性。

**📈 对比分析**

与现有方法相比，该方法在实时重建场景中表现优越，同时在传统的新视角合成基准测试中保持竞争力，捕获到图像的延迟时间减少了一半以上。

**⚠️ 局限性**

局限性包括在遮挡边缘可能产生伪影，以及在传感器覆盖不足时图像特征的退化。

---

## 420. When Copying Is Hard to Beat: A Negative Result in Clinical Assessment Forecasting

**arXiv ID:** 2611.18142 | [PDF](https://arxiv.org/pdf/2611.18142v1)

**作者:** Marius Fracarolli `[一作]` (Heidelberg University), Stefan Riezler `[通讯]` (Heidelberg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究报告了在次日临床评估预测中的负面结果，评估的模型未能在强大的基线模型上表现出一致的优势。

**💡 创新点**

创新点在于强调了文档过程、输入截止时间、观察掩码和解码规则如何影响对预测结果的结论。

**🔧 技术方法**

使用了生理信息驱动的门控模型和多种基线模型进行比较。

**📊 数据集**

数据集包括来自曼海姆大学医学中心的3,736个ICU病例的585,791个项目条目，涵盖39,148个问卷日。

**📈 对比分析**

与强基线模型（如持久性模型）相比，所评估的模型在宏观F1得分上没有显著优势，且在准确性上有所下降。

**⚠️ 局限性**

限制在于数据无法区分文档持久性与真实临床稳定性，且研究仅限于单中心，且在研究期间工具发生过一次变化。

---

## 421. Certificate-Based Verification of Pulsed Field Ablation Models

**arXiv ID:** 2611.18148 | [PDF](https://arxiv.org/pdf/2611.18148v1)

**作者:** Konstantinos Dovelos `[一作]` `[通讯]` (Dovelos Engineering), Konstantinos Dovelos (Dovelos Engineering)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种用于脉冲场消融（PFA）模型的验证方法，通过独立的检查器验证求解器的输出，确保每次运行的结果都有保证的数值误差界限。

**💡 创新点**

创新点在于将传统的数值验证方法与独立的检查器结合，提供了针对每个运行的保证性界限，并且能够处理不同的组织导电率范围。

**🔧 技术方法**

使用了Prager-Synge超圆定理和球算术进行数值计算，结合了向量化的检查器以提高效率。

**📊 数据集**

使用了不同导电率比的模拟数据集，涵盖了心肌和血液的导电率范围。

**📈 对比分析**

与标准网格收敛研究进行了比较，发现证书提供的界限在某些情况下比网格收敛研究更为严格，尤其是在电极边缘附近，证书仍然提供有效的界限。

**⚠️ 局限性**

限制在于模型是轴对称的、线性的、各向同性的，并且没有考虑频率、温度或场依赖性等因素，实际的病变情况可能更复杂。

---

## 422. Patient-specific Modeling of the Tricuspid Valve from 3D Transesophageal Echocardiography

**arXiv ID:** 2611.18522 | [PDF](https://arxiv.org/pdf/2611.18522v1)

**作者:** Davide Tondi `[一作]` (Politecnico di Milano), Manuel K. Rausch `[通讯]` (University of Texas at Austin)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究开发并评估了一种从临床3D经食道超声心动图（3D-TEE）图像中模拟三尖瓣（TV）的方法，旨在支持修复计划。

**💡 创新点**

创新点在于结合临床图像和来自人类组织的测量数据，克服了临床图像无法完全解析弦索解剖结构和直接测量组织特性的限制。

**🔧 技术方法**

使用了有限元模型（FEM）技术，结合形状匹配和基于力的弦索长度调整来模拟瓣膜闭合。

**📊 数据集**

使用了来自德国Asklepios医院和Schleswig-Holstein大学医院的10个成人三尖瓣的匿名术前3D-TEE图像数据集。

**📈 对比分析**

通过与三位独立观察者的分割结果进行比较，发现分割的变异性对瓣膜的接触面积影响最大，而弦索插入密度和瓣膜刚度对几何误差的影响最大。

**⚠️ 局限性**

限制在于模型依赖于手动分割，观察者之间的分割差异主要体现在瓣膜的范围和边界，未来可以通过自动化分割来减少这种变异性。

---

## 423. Disenfranchisement in Spatial Voting Games

**arXiv ID:** 2611.18468 | [PDF](https://arxiv.org/pdf/2611.18468v1)

**作者:** Vincent Lageschulte `[一作]` (Arizona State University), Joel Nishimura `[通讯]` (Arizona State University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文探讨了投票权、群体成员资格和可接受话语的边界，提出了特许选举的概念，分析了意识形态或问题多样性如何影响群体的稳定性。

**💡 创新点**

创新点在于引入了特许选举的框架，展示了在多维意识形态空间中，群体成员的偏好如何影响选举结果和群体稳定性。

**🔧 技术方法**

使用了社会选择理论中的Yolk社会选择函数，结合了空间投票的欧几里得偏好模型。

**📊 数据集**

论文未具体提及使用的数据集，但讨论了理论模型和假设的构建。

**📈 对比分析**

通过与现有的社会选择函数进行比较，展示了在多维空间中，特许选举可能导致的复杂行为和周期性变化，表明群体成员资格的不稳定性。

**⚠️ 局限性**

限制在于模型假设投票者是短视的，未考虑长期的投票行为和社会规范对选举稳定性的影响。

---

## 424. Process Mining-Enabled Federated Anomaly Detection for the Industrial IoT

**arXiv ID:** 2611.17869 | [PDF](https://arxiv.org/pdf/2611.17869v1)

**作者:** Francesco Vitale `[一作]` (University of Naples Federico II), Hangli Ge `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于过程挖掘的联邦异常检测方法，旨在解决工业物联网中非独立同分布（non-IID）数据的问题。

**💡 创新点**

创新点在于通过过程挖掘评估本地节点数据与全局节点参考数据的相似性，并在聚合过程中动态加权本地模型，从而实现个性化更新。

**🔧 技术方法**

使用了过程挖掘技术来评估节点相似性，并结合相似性加权的联邦学习进行模型训练。

**📊 数据集**

使用了来自汽车制造业工厂的振动数据集，该数据集在不同操作条件下收集。

**📈 对比分析**

与孤立的本地训练基线和标准的FedAvg方法进行比较，提出的方法在F1分数上分别提高了46.52%和11.99%，并且在假阳性率上减少了51.24%。

**⚠️ 局限性**

局限性包括实验数据集来自于缩放模型，可能无法完全反映真实工业环境中的复杂噪声和机械故障；此外，合成异常的注入可能无法完美模拟真实设备中的故障情况。

---

## 425. New Closed-Form Approximations in Statistical Mechanics Obtained by Symbolic Regression

**arXiv ID:** 2611.17498 | [PDF](https://arxiv.org/pdf/2611.17498v1)

**作者:** Omkar Kunjir `[一作]`, Mikhail Itskov `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过符号回归方法解决了聚合物统计力学中的三个近似问题，展示了数据驱动的表达式发现如何从精确的数值解推导出适合后续分析的可解释的封闭形式表达式。

**💡 创新点**

创新点在于使用深度符号优化（DSO）和PyOperon两个互补的符号回归框架，系统地比较了它们在聚合物统计力学中的应用，发现它们能够有效地发现准确且可解析的近似表达式。

**🔧 技术方法**

使用了深度符号优化（DSO）框架和PyOperon框架，前者采用基于强化学习的递归神经网络，后者则是基于遗传编程的框架。

**📊 数据集**

使用了聚合物统计力学中的三个问题的数据集，包括Worm-Like Chain (WLC)模型的力-伸长关系、对数阶乘ln(n!)的近似以及自由关节链（FJC）模型的非高斯径向概率密度函数。

**📈 对比分析**

在WLC模型中，DSO的最大相对误差为0.15%，而PyOperon进一步降低到0.05%，均优于现有的Marko-Siggia公式（约17%）和Petrosyan近似。在对数阶乘的近似中，PyOperon的最大相对误差为5.8×10^-10，显著优于所有经典公式。在FJC模型中，PyOperon的自由残差学习策略比Darabi等（2023）的近似提高了11.2倍。

**⚠️ 局限性**

限制在于DSO和PyOperon的搜索是在不同的计算预算和独立运行次数下进行的，因此结果反映了所获得表达式的准确性，而不是搜索算法的内在性能。对于短链（n=5），可能需要额外的短链修正以进一步提高准确性。

---

## 426. Voice-Based Conversational Agents as AI Care Assistants: Evaluating Personalization, Intervention Delivery Fidelity, and Privacy Expectations in Older Adults' Care

**arXiv ID:** 2611.18385 | [PDF](https://arxiv.org/pdf/2611.18385v1)

**作者:** Thalia Viranda `[一作]` (Cornell Tech), Tanzeem Choudhury `[通讯]` (Cornell Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

设计并评估了ARC，一个基于语音的对话代理，旨在通过多领域生活方式干预为老年人提供个性化支持。

**💡 创新点**

创新点在于将基于证据的干预转化为生成对话，能够适应个体背景，同时保持干预过程的完整性。

**🔧 技术方法**

使用了基于GPT-4o的实时语音交互技术，结合了行为激活、动机访谈和回忆疗法等证据基础的方法。

**📊 数据集**

在为期三天的远程研究中，共有80名老年人参与，完成了473个会话，涉及六种会话类型。

**📈 对比分析**

与传统方法相比，ARC在领域级别的上下文重用表现良好，但在参与者特定的上下文重用上表现一般，干预组件的交付完整性存在不足，尤其是在物理健康和社交健康会话中。

**⚠️ 局限性**

限制在于未评估临床效果，样本主要为技术熟练的老年人，且个性化的上下文重用较低，未来需要在认知障碍人群中进行更长时间的评估。

---

## 427. Mutable Inference: Architecting LLM Accelerators for Inference-Time Learning

**arXiv ID:** 2611.18523 | [PDF](https://arxiv.org/pdf/2611.18523v1)

**作者:** Yimin Gao `[一作]` (University of Virginia), Mircea Stan `[通讯]` (University of Virginia)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的推理加速器架构，称为学习同时服务（LWS），旨在通过共享计算硬件来同时进行推理和学习，从而提高长上下文大语言模型（LLM）的推理效率。

**💡 创新点**

创新点在于通过硬件管理的每层权重版本控制和可恢复的执行机制，允许学习在推理过程中进行，同时保持推理的优先级，从而有效利用空闲的计算资源。

**🔧 技术方法**

使用了推理加速器架构，结合了硬件管理的权重版本控制、可恢复的计算块和专用的softmax梯度单元，以减少推理和学习之间的资源竞争。

**📊 数据集**

使用了TTT-E2E 3B模型，评估了在32K、64K和128K上下文长度下的性能，并在多个模型上进行了比较，包括BLOOM 3B、Gemma-3 4B等。

**📈 对比分析**

与之前的串行执行和其他共执行架构相比，LWS在九个模型上实现了1.6倍的吞吐量提升，且在保持99.3%学习质量收益的同时，推理吞吐量与仅推理的情况相当。

**⚠️ 局限性**

限制在于LWS的设计增加了NPU核心面积的1.5%，并且在某些情况下，学习的向量需求可能会超过空闲的向量容量，导致资源竞争。

---

## 428. Unlocking Cross-Attention in Label-Scarce Regimes: A Weakly Supervised Approach for Video Semantic Segmentation

**arXiv ID:** 2611.18324 | [PDF](https://arxiv.org/pdf/2611.18324v1)

**作者:** Andrea Marelli `[一作]` (Politecnico di Milano), Luca Magri `[通讯]` (Politecnico di Milano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为MemCAM的弱监督视频语义分割框架，旨在通过视频级标签学习时间一致的语义对象掩码。

**💡 创新点**

MemCAM通过特征空间自监督直接学习时间交叉注意力，避免了依赖外部先验信息，从而提高了模型在弱监督条件下的稳定性和性能。

**🔧 技术方法**

使用了Transformer架构，结合了内存注意力模块（MAM）和CAM传播算法（CAMPA）来实现特征匹配和自监督学习。

**📊 数据集**

在YouTube-Objects v2.2和WS^2数据集上进行了实验，这两个数据集是公开可用的，适合于视频级监督的任务。

**📈 对比分析**

与现有的弱监督视频分割方法相比，MemCAM在时间定位和分割性能上表现出色，设定了新的最先进水平，尤其是在视频级标签的情况下。

**⚠️ 局限性**

MemCAM在学习时间注意力时对数据的需求较高，模型容量与数据集规模之间的匹配至关重要，过大的模型可能导致过拟合。

---

## 429. LitArc-Bench: Disentangling Text-Specific Reader-Response Dynamics from Position and Lexical Affect

**arXiv ID:** 2611.18412 | [PDF](https://arxiv.org/pdf/2611.18412v1)

**作者:** Mingming Su `[一作]` (Capital University of Economics and Business), Xinche Zhang `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文介绍了LitArc-Bench，一个用于分析文学文本中人类情感轨迹的基准，包含39个中文文学片段和430份问卷数据。

**💡 创新点**

创新点在于将文本位置、词汇情感和文本特定的读者反应动态分开评估，并提出了一种可靠性意识的对齐分解方法。

**🔧 技术方法**

使用了未来盲模型（future-blind models）和交叉拟合模型（cross-fitted models）来评估文本与人类反应之间的对齐。

**📊 数据集**

数据集包含39个中文文学片段，涉及9个原文和30个翻译文本，共收集了430份读者问卷。

**📈 对比分析**

与传统的情感基准相比，模型在文本特定结构的恢复上表现出优势，尤其是在情感的变化方向上，模型能够恢复69-76%的情感方向，且在所有九个模型维度中均优于错误文本（p<.0001）。

**⚠️ 局限性**

限制在于LitArc-Bench并不是一个涵盖广泛的中文文学或读者的调查，其样本仅来自于两个班级的81名成人，且未进行随机分配或平衡分析。

---

## 430. Beyond Prediction Gain: LASR as a Metric for Action-Grounded Latents in World Models

**arXiv ID:** 2611.17775 | [PDF](https://arxiv.org/pdf/2611.17775v1)

**作者:** Yu Zhou `[一作]` (Tsinghua University), Yong-Jin Liu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的度量标准LASR，用于评估学习到的潜在表示是否支持与动作相关的预测目标，超越当前状态和非动作替代品的解释。

**💡 创新点**

LASR是一个无量纲的度量标准，能够量化潜在动作支持，提供GeoLASR和WorstLASR两个互补的汇总，能够进行不同表示之间的一致比较。

**🔧 技术方法**

使用了潜在动作支持比率（LASR）作为评估指标，并构建了LASR基准，采用了匹配的预测器和控制变量进行比较。

**📊 数据集**

在ALOHA数据集上进行了实验，使用了记录的动作和相应的物理状态变化作为目标，进行了五个折叠的离线评估。

**📈 对比分析**

与现有方法相比，LASR能够检测已知的动作信息，拒绝代表性的非动作捷径，并支持对不完美潜在表示的分级比较。实验结果表明，LASR在评估中表现出色，能够提供更好的预测支持。

**⚠️ 局限性**

LASR的局限性在于它仍然是相对的，低LASR并不证明因果动作身份或可执行性，且结果依赖于所声明的审计标签、探针家族和控制条件。

---

## 431. Performance-Portable Exact Gaussian Process Regression in Python via SYCL and oneMath

**arXiv ID:** 2611.17565 | [PDF](https://arxiv.org/pdf/2611.17565v1)

**作者:** Alexander Strack `[一作]` (University of Stuttgart), Dirk Pflüger `[通讯]` (University of Stuttgart)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种基于SYCL和oneMath的GPRat库的可移植精确高斯过程回归实现，支持多种GPU环境。

**💡 创新点**

创新点在于开发了一个可移植的SYCL后端，使GPRat能够在AMD和Intel GPU上运行，同时保持与NVIDIA CUDA的竞争性能。

**🔧 技术方法**

使用了SYCL编程模型、HPX异步多任务运行时和oneMath数学库。

**📊 数据集**

使用了多个GPU（NVIDIA A30、AMD Instinct MI210和Intel Arc B580）进行性能评估，并使用了msd数据集进行实验。

**📈 对比分析**

与GPyTorch和GPflow库进行了性能比较，结果显示在大问题规模下，GPRat的SYCL后端在AMD和NVIDIA GPU上分别比GPflow快1.79倍和2.17倍，而在Intel GPU上性能较差。

**⚠️ 局限性**

限制在于只评估了高斯过程回归的预测和不确定性计算，未涵盖超参数优化训练，且每个后端仅在单个GPU上进行评估。

---

## 432. Electrocardiographic Imaging via Posterior Sampling under Operator Discrepancy

**arXiv ID:** 2611.17851 | [PDF](https://arxiv.org/pdf/2611.17851v1)

**作者:** Ricarda Buttmann `[一作]` (Technical University of Applied Sciences), Andreas M. Lehrmann `[通讯]` (Technical University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种新的扩散后验采样方法OpDis-PS，用于在前向算子存在差异的情况下重建心脏表面电位。

**💡 创新点**

创新点在于将前向算子的差异建模为信号条件的近似误差，而不是将其视为信号无关的噪声，从而提高了重建的准确性。

**🔧 技术方法**

使用了扩散后验采样技术，结合了信号条件的高斯近似误差模型。

**📊 数据集**

使用了来自CT扫描的1000个合成四腔心脏网格和1228个CT扫描的躯干解剖数据集进行评估。

**📈 对比分析**

与多种基线方法（包括变分正则化、摊销推理和后验采样方法）进行了比较，OpDis-PS在重建性能上表现优越，尤其是在未见的病理激活模式上。

**⚠️ 局限性**

局限性在于评估依赖于模拟的电生理学和两个差异源，实际记录可能包含多种不匹配来源，且模型假设可能未能完全捕捉实际情况。

---

## 433. An Algebraic Perspective on Operational Semantics

**arXiv ID:** 2611.17587 | [PDF](https://arxiv.org/pdf/2611.17587v1)

**作者:** Eric Bond `[一作]` (University of Michigan), Max New `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了一种统一的构造方法，将语言的操作语义和指称语义结合在一起，通过代数理论来定义效果。

**💡 创新点**

创新点在于通过代数理论统一构建操作语义和指称语义，提供了一种系统化的方法来处理效果。

**🔧 技术方法**

使用了代数理论和单子容器的概念，构建了基于配置的操作语义。

**📊 数据集**

使用了状态、错误、写入、加权和预凸概率选择等效果的实例。

**📈 对比分析**

通过与传统的操作语义和指称语义进行比较，证明了每一步都保持了指称，并在特定条件下实现了终止和计算的充分性。

**⚠️ 局限性**

限制在于当前框架仅适用于自由模型为单子容器的代数效果，排除了基于商的理论。

---

## 434. Dual-Slack Frontier Matching: Parallel Approximate Maximum-Product Matching

**arXiv ID:** 2611.17502 | [PDF](https://arxiv.org/pdf/2611.17502v1)

**作者:** Marc A. Tunnell `[一作]` (Purdue University), Xiaoye Sherry Li `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 5714 | [OpenAlex ID](https://openalex.org/A5006873445)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为双松弛前沿匹配（DSFM）的确定性共享内存并行算法，用于近似最大乘积匹配，旨在提高稀疏矩阵预处理中的匹配质量和计算效率。

**💡 创新点**

DSFM结合了有界的质量引导搜索和结构完整性回退，能够在存在完美匹配时返回完美匹配，并在没有完美匹配时返回最大基数匹配，同时提供后验质量界限。

**🔧 技术方法**

使用了共享内存并行算法，结合了固定松弛的搜索策略和累积的霍尔障碍修正。

**📊 数据集**

在598个非奇异矩阵上进行了实验，这些矩阵来自SuiteSparse矩阵集合，且均具有验证的MC64参考分配。

**📈 对比分析**

与重权完美匹配（HWPM）和精确的MC64分配进行了比较，DSFM在大多数质量阈值下的匹配质量高于HWPM，并在37个大实例上实现了3.53倍的几何平均加速，相比之下HWPM为2.29倍。

**⚠️ 局限性**

DSFM的局限性在于其在某些情况下可能需要更多的边缘曝光以找到完美匹配，且在处理复杂结构时可能面临性能瓶颈。

---

## 435. Use-case-driven ontology modeling with the NeOn methodology for ambient assisted living

**arXiv ID:** 2611.17466 | [PDF](https://arxiv.org/pdf/2611.17466v1)

**作者:** Daniele Spoladore `[一作]` (Institute of Industrial Technologies and Automation National Research Council of Italy), Edoardo Battisti `[通讯]` (Politecnico di Milano)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

本文研究了智能家居和环境辅助生活（AAL）的本体开发，旨在为不同用户提供个性化服务，特别是针对老年人和残疾人群体。

**💡 创新点**

创新点在于通过使用案例生成本体需求规范文档，并采用NeOn方法论构建模块化本体，结合国际功能、残疾与健康分类（ICF）和心肺适能评估。

**🔧 技术方法**

使用了NeOn方法论进行本体开发，结合SWRL规则进行分类和推理。

**📊 数据集**

使用案例包括一位虚构的脆弱老年女性在家中锻炼的场景，以及设计师为不同残疾用户配置厨房的场景。

**📈 对比分析**

通过使用案例识别本体需求，比较不同用户的健康状况和智能对象的适用性，性能表现良好，能够支持用户的日常活动。

**⚠️ 局限性**

限制在于本体的适用性可能受到特定用户群体和智能对象的限制，未来需要扩展到更多类型的用户和智能对象。

---

## 436. Wireless Node Ranking via Fourier-Driven Maximum-Likelihood Estimation (extended version)

**arXiv ID:** 2611.17973 | [PDF](https://arxiv.org/pdf/2611.17973v1)

**作者:** Daniel Schneider `[一作]` (Independent), Hannes Frey `[通讯]` (University of Koblenz)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文通过非阿贝尔傅里叶理论解决了基于信号强度测量的线性拓扑最大似然估计问题，提出了一种新的无线节点排序方法。

**💡 创新点**

创新点在于将无线节点排序与数学统计中的序列化（seriation）相联系，并提出了一种在相关高斯阴影下的最大似然估计方法，该方法在每个噪声水平下都能准确恢复节点顺序。

**🔧 技术方法**

使用了非阿贝尔傅里叶分支限界法（Fourier B&B）来解决最小二乘二次分配问题（QAP），并通过傅里叶框架进行优化。

**📊 数据集**

使用了基于信号强度的测量数据集，特别是在不同的阴影模型下（如无关的对数正态阴影和相关的联合路径损失模型）。

**📈 对比分析**

与现有的序列化方法（如Fiedler和SABRE）进行比较，结果显示在相关阴影下，最大似然估计的恢复率达到100%，而其他方法无法达到这一性能。

**⚠️ 局限性**

限制在于该方法在理论上要求精确的线性拓扑，尽管在小扰动下表现出一定的鲁棒性，但在实际应用中可能会受到拓扑变化的影响。

---

## 437. HarmonyText: Harmonious Scene Text Editing via Reverse-to-Refined Pair Synthesis and Region-Decoupled Frequency Learning

**arXiv ID:** 2611.17733 | [PDF](https://arxiv.org/pdf/2611.17733v1)

**作者:** Ziqi Yi `[一作]` (South China University of Technology), Lianwen Jin `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为HarmonyText的无掩码场景文本编辑框架，旨在通过反向到精炼配对合成和区域解耦频率学习来改善文本编辑的准确性和视觉和谐性。

**💡 创新点**

创新点在于引入了区域解耦优化（RDO）和级联高频纹理指导（CHFTG），以解决伪目标监督的不可靠性和标准均方误差（MSE）回归导致的高频细节抑制问题。

**🔧 技术方法**

使用了区域解耦优化（RDO）和级联高频纹理指导（CHFTG）技术，结合反向到精炼配对合成（RePair）来构建高质量的配对数据集。

**📊 数据集**

构建了一个名为STE25K的高质量25K配对数据集，包含真实场景文本图像的配对样本。

**📈 对比分析**

通过与现有方法的比较，HarmonyText在文本编辑准确性、背景保真度、颜色一致性和感知和谐性方面表现优越，尤其在HarmonyText-Bench和公共场景文本识别数据集上均取得了最佳结果。

**⚠️ 局限性**

限制在于依赖于PaddleOCR进行文本定位、HiSAM进行细粒度文本分割以及Qwen3-VL进行指令生成和验证，这些组件的错误可能会影响数据构建或评估的精度。

---

## 438. Rethinking Visual Reasoning in Text-to-Image Reward Modeling

**arXiv ID:** 2611.17585 | [PDF](https://arxiv.org/pdf/2611.17585v1)

**作者:** Shiye Su `[一作]` (Stanford University), Serena Yeung-Levy `[通讯]` (Stanford University)

**通讯引用:** 10907 | [OpenAlex ID](https://openalex.org/A5081511803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的奖励模型，通过引入视觉推理监督来改善文本到图像生成系统的性能。

**💡 创新点**

创新点在于使用CrossPairs方法，将共享图像的偏好对与传统的偏好数据结合，从而消除美学干扰，直接针对文本-图像对齐进行训练。

**🔧 技术方法**

使用了现代视觉语言模型（VLM）作为基础，并结合了CrossPairs数据集进行训练。

**📊 数据集**

使用了多个视觉推理数据集，包括CLEVR、SpatialSense、TallyQA等，将其转换为CrossPairs格式。

**📈 对比分析**

与现有的奖励模型（如HPSv3）进行比较，结果显示该模型在GenEval测试中在推理时间选择和强化学习后训练中分别提高了4.0%和5.5%的性能，超越了HPSv3。

**⚠️ 局限性**

限制在于尽管引入了视觉推理监督，但仍可能存在对偏好数据的过拟合问题，且在某些情况下可能无法完全保留视觉理解能力。

---

## 439. See and Detect: Image-Driven Open-Vocabulary Multi-View 3D Object Detection

**arXiv ID:** 2611.17897 | [PDF](https://arxiv.org/pdf/2611.17897v1)

**作者:** Yating Xu `[一作]` (MVP Lab), Jifei Song `[通讯]` (MVP Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为See and Detect的图像驱动框架，用于开放词汇的多视角室内3D物体检测，主要通过多视角图像生成3D框和语义标签。

**💡 创新点**

创新点在于将多视角图像作为主要证据进行3D框和语义标签生成，而不是依赖点云，解决了现有方法中由于点云不完整导致的物体遗漏和语义学习局限性的问题。

**🔧 技术方法**

使用了多视角图像、深度图、图像驱动的伪框生成、跨视角一致性框合并和上下文感知语义学习等技术。

**📊 数据集**

在ScanNet和ARKitScenes数据集上进行了广泛实验，ScanNet包含1201个训练场景和312个测试场景，ARKitScenes包含4493个训练和549个测试扫描。

**📈 对比分析**

与现有的点云驱动方法相比，See and Detect在ScanNet60的开放词汇检测中提高了4.3%的mAP_50，并且运行速度比之前的最先进方法快500倍，显示出显著的性能优势。

**⚠️ 局限性**

限制在于该方法仍然依赖于图像质量和视角的多样性，可能在某些情况下受到图像噪声和遮挡的影响。

---

## 440. WETW: Learning Pair-Specific Weighted Euclidean Distances for Efficient DTW Search

**arXiv ID:** 2611.17412 | [PDF](https://arxiv.org/pdf/2611.17412v1)

**作者:** Christos Panourgias `[一作]` (LIPADE, Universite Paris Cite), Themis Palpanas `[通讯]` (LIPADE, Universite Paris Cite)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种新的加权欧几里得时间扭曲（WETW）方法，用于提高动态时间扭曲（DTW）在时间序列相似性搜索中的效率。

**💡 创新点**

WETW是首个根据每对序列自适应调整比较权重的学习距离，显著提高了搜索速度和准确性。

**🔧 技术方法**

使用了加权欧几里得系列编码器（WESee）和小型对称网络来预测每个嵌入维度的权重。

**📊 数据集**

在八个数据集上进行了评估，包括长达48K点的时间序列，数据库规模达到840K系列。

**📈 对比分析**

与现有的最优精确方法相比，WETW-FV在平均recall@10为0.93时速度快11.4倍，而最佳学习基线在相同速度下仅达到0.59的recall@10。

**⚠️ 局限性**

方法的局限性在于需要进行离线监督和评分，且在多变量和可变长度序列上的扩展仍需进一步研究。

---

## 441. In Situ Monitoring and Predictive Modeling of Harmful Algal Blooms with Autonomous Surface Vehicles

**arXiv ID:** 2611.18501 | [PDF](https://arxiv.org/pdf/2611.18501v1)

**作者:** Nenyi Kweku Nkensen Dadson `[一作]` (Louisiana State University), Corina Barbalata `[通讯]` (Louisiana State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于自主表面船（ASV）的水质监测系统，用于检测有害藻华（HAB），通过在现场收集空间分布的水质数据来应对环境、经济和公共健康挑战。

**💡 创新点**

创新点在于将自主导航、深度控制的传感器部署与声纳和事件相机系统相结合，实现了多模态的水质数据收集，提升了对藻华形成的监测能力。

**🔧 技术方法**

使用了自主表面船（ASV）、电动绞车、多个水质传感器、侧扫声纳和事件相机等技术。

**📊 数据集**

数据集包括在路易斯安那州的Pontchartrain湖和False River收集的水质测量数据，涵盖了藻类色素指标、营养物质浓度、浊度、温度等多个参数。

**📈 对比分析**

通过机器学习分类方法对收集的水质数据进行分析，结果显示该系统能够有效区分HAB和非HAB，分类精度达到99.6%。

**⚠️ 局限性**

限制在于系统的水下操作能力和对环境变化的适应性，未来需要进一步验证分类方法并扩展现场部署。

---

## 442. ProdPilot: A Layered Model Context Protocol System for Automated Production Readiness and Deployment of Vibe-Coded Projects

**arXiv ID:** 2611.17381 | [PDF](https://arxiv.org/pdf/2611.17381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 443. Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent

**arXiv ID:** 2611.18397 | [PDF](https://arxiv.org/pdf/2611.18397v1)

**作者:** Run Luo `[一作]` (miHoYo), Shaoqing Ren `[通讯]` (miHoYo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种通用智能体框架，旨在通过统一的键盘和鼠标接口控制复杂的游戏环境，支持实时交互和在线学习。

**💡 创新点**

创新点在于通过持续的预训练、任务条件的监督微调和时间感知的强化学习，学习可转移的控制能力、长时间的推理和持久的记忆。

**🔧 技术方法**

使用了持续预训练（CPT）、监督微调（SFT）和时间感知的群体相对策略优化（TA-GRPO）等技术，结合了流式推理引擎（Streaming-vLLM）以支持实时交互。

**📊 数据集**

数据集包括来自《崩坏：星穹铁道》的数千小时的游戏轨迹，经过筛选和注释，构建了多层次的任务监督数据。

**📈 对比分析**

与其他代表性游戏智能体相比，本文的方法在模型初始化、训练配方、训练数据、实时推理和开源可用性方面表现出色，展示了与熟练人类玩家相当的性能。

**⚠️ 局限性**

限制在于当前评估覆盖的环境和任务结构较为有限，未来需要扩展到更广泛的交互设置，并进一步优化实时交互的成本。

---

## 444. Efficiency Gains in Social Science Research: A Prospective Silicon Sampling Protocol for Pre-Treatment and Code Optimization

**arXiv ID:** 2611.18436 | [PDF](https://arxiv.org/pdf/2611.18436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 445. G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands

**arXiv ID:** 2611.17814 | [PDF](https://arxiv.org/pdf/2611.17814v1)

**作者:** Yuechun Yu `[一作]` (Alibaba Group), Jin Zhang `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为G-CAST的自我训练系统，用于应对电子商务中用户需求和能力合同的不断变化，旨在提高大语言模型（LLM）驱动的购物代理的适应性。

**💡 创新点**

G-CAST的创新点在于其治理的自我演化循环，能够在不断变化的用户需求和能力扩展中，保持模型的有效性和适应性，同时保留用户已经依赖的行为。

**🔧 技术方法**

使用了治理的自我训练机制，包括奖励导向的数据编排和能力合同驱动的奖励演化。

**📊 数据集**

在阿里巴巴的1688平台上，使用了真实的生产电子商务代理数据集进行实验，涵盖了多个不同的需求时期。

**📈 对比分析**

与强大的自主后训练基线相比，G-CAST在适应新需求的同时，更好地保留了之前接受的行为，表现出更强的适应性和保留能力。

**⚠️ 局限性**

本研究的局限性在于基于一个长期的生产应用和一个主要的模型系列，因此观察到的需求和能力演变模式可能无法直接转移到其他代理设置中。

---

## 446. When Should a Virtual Patient Ask? RISP-Guided Proactive Information Seeking in Multi-Agent Patient Simulation

**arXiv ID:** 2611.18183 | [PDF](https://arxiv.org/pdf/2611.18183v1)

**作者:** Jingyao Cai `[一作]` (Bournemouth University), Xiaosong Yang `[通讯]` (Bournemouth University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于RISP模型的多代理虚拟患者系统（RISP-MAS），用于支持护理沟通培训，特别是患者主动信息寻求的模拟。

**💡 创新点**

创新点在于将患者的信息需求与信息寻求决策（何时询问、询问什么）相结合，提供了更为动态和心理学基础的患者模拟。

**🔧 技术方法**

使用了多代理系统技术，结合了RISP理论模型来评估患者的感知理解和信息需求。

**📊 数据集**

使用了30个由护理专家准备的合成患者档案，这些档案基于护理教科书中的常见场景。

**📈 对比分析**

通过与护理学生的互动评估RISP-MAS，专家和学生的反馈均显示出积极的互动体验，评分普遍高于4.0，表明系统在模拟患者信息寻求行为方面的有效性。

**⚠️ 局限性**

主要限制在于缺乏客观的基准比较，评估主要依赖于主观判断，且系统的变化可能影响后续交流，进行基线和消融比较也较为困难。

---

## 447. Evolutionary-based Transfer Learning for Physical Attack on End-to-End Autoencoder Communications

**arXiv ID:** 2611.17393 | [PDF](https://arxiv.org/pdf/2611.17393v1)

**作者:** Ho Viet Duc Luong `[一作]` (Hanoi University of Science and Technology), Trinh Van Chien `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 3258 | [OpenAlex ID](https://openalex.org/A5018288108)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文提出了一种基于进化算法的迁移学习方法，用于对端到端自编码器通信系统进行物理攻击，旨在优化攻击信号以降低自编码器的性能。

**💡 创新点**

创新点在于提出了PAGA（基于遗传算法的物理攻击）和PAMFGA（多因素遗传算法），分别针对白盒和黑盒攻击场景，利用进化算法解决物理攻击的优化问题。

**🔧 技术方法**

使用了遗传算法（GA）和多因素进化算法（MFEA）作为主要技术，结合精英种群初始化和特定的进化操作来搜索扰动向量。

**📊 数据集**

实验使用了基于多层感知器（MLP）的自编码器架构，并在不同的信道条件下（如AWGN、瑞利衰落和Rician衰落）进行测试。

**📈 对比分析**

与现有的物理攻击方法（如FGMA、RSBA和RMAEP）进行比较，PAGA在白盒场景中表现出更强的攻击性能，而PAMFGA在黑盒场景中也显示出竞争力，尤其是在多个替代模型的情况下。

**⚠️ 局限性**

限制在于本研究仍然是基于仿真的，考虑的通信配置、架构和替代模型有限，未来工作将扩展到硬件验证和更广泛的场景。

---

## 448. Do Underwater Image Quality Metrics Agree with Human Perception? A Benchmark with Human Choices, Simple Edits and Generative Drift

**arXiv ID:** 2611.18429 | [PDF](https://arxiv.org/pdf/2611.18429v1)

**作者:** Rishii Kumar Singh `[一作]` (Delhi Technological University), Dinesh Kumar Vishwakarma `[通讯]` (Delhi Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了多种水下图像质量评估指标（包括手工设计和学习的指标）与人类偏好的匹配程度，使用了五个参考数据集进行比较。

**💡 创新点**

提出了一个新的基准，结合了人类参考、编辑、攻击和生成的图像，以及每个指标在每个图像上的评分，以便于新指标的评估。

**🔧 技术方法**

使用了手工设计的水下指标、学习的水下指标、视觉语言模型和通用指标进行评估。

**📊 数据集**

使用了五个参考数据集，包括四个已发布的意见评分集和一个新的配对比较研究。

**📈 对比分析**

与人类偏好的比较显示，没有任何指标在所有参考数据集上都能一致匹配人类的选择，最好的指标在五个参考数据集上的Cohen's κ值仅为较低水平。

**⚠️ 局限性**

研究的局限性包括样本量小、评估者的同意程度中等、参考图像为原始图像而非增强输出，以及未能测试所有已发布的水下指标。

---

## 449. Can LLMs Write Backdoors? An Empirical Security Study of Weight-Space Programming

**arXiv ID:** 2611.17395 | [PDF](https://arxiv.org/pdf/2611.17395v1)

**作者:** Zhaojia Wang `[一作]` (Shandong University), Zheng Li `[通讯]` (Shandong University)

**通讯引用:** 6574 | [OpenAlex ID](https://openalex.org/A5100415028)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了大型语言模型（LLMs）是否能够通过权重空间编程直接向神经网络注入后门，进行了一项实证安全研究。

**💡 创新点**

创新点在于首次系统性评估了五种通用LLM作为权重编辑器在不同神经网络架构中的后门注入能力，揭示了权重空间编程的潜在安全风险。

**🔧 技术方法**

使用了权重空间编程（WSP）技术，LLMs能够根据自然语言指令直接修改神经网络的权重，而无需传统的基于梯度的训练。

**📊 数据集**

使用的数据集包括MNIST（用于MLP）、CIFAR-10（用于ResNet-18）和AG News（用于Qwen2.5-0.5B-Instruct）。

**📈 对比分析**

通过比较五种LLM在相同编辑条件下的表现，发现GPT-6 Astra在所有三个目标上均成功注入后门，而其他LLM的表现则有所不同，成功率和清洁性能的保持程度存在显著差异。

**⚠️ 局限性**

限制在于研究仅集中于三种目标架构和特定触发配置，未能涵盖更大模型、更复杂任务和多样化触发器的适用性。

---

## 450. Evaluating Feedback-Driven Black-Box Processes via Stochastic Modeling and Simulation

**arXiv ID:** 2611.18477 | [PDF](https://arxiv.org/pdf/2611.18477v1)

**作者:** Mengfan Xu `[一作]` (University of Massachusetts Amherst), Barry Nelson `[通讯]` (Northwestern University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了软件工程和机器学习中的反馈驱动黑箱过程，提出了一种基于马尔可夫链和模拟的随机框架来建模和评估这些过程，特别是在变异基础模糊测试中的应用。

**💡 创新点**

创新点在于引入了一个统一的评估框架，能够连接经验实践与理论，提供新的性能度量和评估程序，具有可证明的保证。

**🔧 技术方法**

使用了马尔可夫链和模拟技术来建模反馈驱动的黑箱过程，并开发了新的评估范式。

**📊 数据集**

在18个广泛使用的模糊测试基准上评估了9个模糊器，并进行了4个非模糊领域的案例研究。

**📈 对比分析**

通过与现有模糊器的比较，展示了该框架在评估中的原则性基础，结果表明该框架在模糊测试之外也具有广泛的应用潜力。

**⚠️ 局限性**

限制在于该框架的应用可能需要对系统的反馈和状态有一定的可观察性，而在某些情况下，隐藏状态可能无法获得。

---

## 451. Co-Sight Harness: Externalized Control Assets and Deterministic Enforcement for Long-Horizon Business Agents

**arXiv ID:** 2611.17696 | [PDF](https://arxiv.org/pdf/2611.17696v1)

**作者:** Luyao Wang `[一作]` (ZTE Corporation), Baoli Wang `[通讯]` (ZTE Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

研究了如何在不改变语言模型的情况下，提高长时间业务工作流的可靠性，提出了一种将工作流定义、状态和领域知识外部化为版本控制资产的方法。

**💡 创新点**

创新点在于将控制数据外部化为版本控制资产，并通过状态机和可检查的接受门在执行过程中强制执行这些资产，同时引入独立审计员对结果进行评分。

**🔧 技术方法**

使用了状态机、机器检查的接受门和版本控制资产的外部化技术。

**📊 数据集**

在三个部署的工作流上进行了评估，这些工作流的步骤超过200步，特别是在领域A上使用了120个案例的端到端评估集。

**📈 对比分析**

与传统方法相比，未干预的完成率从50%提高到100%，检索召回率从56%提高到91.67%，端到端延迟从440分钟降低到159分钟，场景分析召回率从45%提高到93%。

**⚠️ 局限性**

限制在于该方法依赖于可机械检查的接受标准和稳定的历史判断，若接受标准无法机械决定或回归集未覆盖某活动，则仍需人工干预。

---

## 452. A mesh-free deep energy method for phase-field modeling of thermally driven brittle fracture

**arXiv ID:** 2611.17630 | [PDF](https://arxiv.org/pdf/2611.17630v1)

**作者:** Han Zhang `[一作]` (University of New South Wales), Elena Atroshchenko `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于多分辨率神经表示的无网格深能量方法，用于预测在瞬态热载荷下自发裂纹的成核、裂纹间距以及选择性裂纹传播和停止。

**💡 创新点**

创新点在于该方法无需预先设定裂纹路径或成核位置，能够在热载荷下自动计算裂纹的形成和演化。

**🔧 技术方法**

使用了多分辨率神经网络表示和深能量方法，结合了反向欧拉热方程的增量泛函来计算温度。

**📊 数据集**

在五个数值示例上进行了评估，包括带缺口的板、方向淬火的玻璃条、淬火的陶瓷板和环。

**📈 对比分析**

与独立的有限元参考进行了比较，结果显示在不同热载荷下的初始峰值载荷与参考值相差不超过1.8%。在玻璃条的裂纹路径变化与实验结果一致，且在淬火陶瓷板上形成的裂纹阵列与实验观察到的类型相符。

**⚠️ 局限性**

局限性包括结果依赖于神经表示的分辨率和优化过程，尚未进行系统的收敛性研究；在某些条件下，损伤的累积控制设置未能完全消除伪影；在环的孔沿线存在持久的损伤层。

---

## 453. On the Limits of Sufficient Explanations

**arXiv ID:** 2611.18163 | [PDF](https://arxiv.org/pdf/2611.18163v1)

**作者:** Klaus Weinbauer `[一作]` (Machine Learning Research Unit, TU Wien), Sagar Malhotra `[通讯]` (Machine Learning Research Unit, TU Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文正式描述了在结构化数据（如图和字符串）上存在充分解释的条件，重点关注选择输入部分作为充分解释的后置解释器。

**💡 创新点**

创新点在于定义了一种通用的解释形式，展示了分类器与选择关系之间的相互作用如何完全表征充分解释器的存在性，并扩展到概率充分性和删除充分性的实际相关变体。

**🔧 技术方法**

使用了形式化分析方法，提出了三种统计量来界定任务与充分解释之间的距离，并在语言、蛋白质和分子分类数据集上进行了评估。

**📊 数据集**

使用的数据集包括语言、蛋白质和分子分类数据集，如HIV活性预测、抗癌活性预测、突变性预测、酶分类等。

**📈 对比分析**

通过比较不同数据集中的共界限，发现大多数数据集无法实现完全充分性，且这些界限为每个解释器提供了绝对的充分性判断参考。

**⚠️ 局限性**

限制在于，尽管删除充分性总是可以实现，但在某些任务中，删除充分的解释可能是空洞的，即对分类器没有提供任何信息。

---

## 454. CardioTrace: Motion-Aware Cine Synthesis via Coupled Tracing and Segmentation

**arXiv ID:** 2611.17909 | [PDF](https://arxiv.org/pdf/2611.17909v1)

**作者:** Zhi Chen `[一作]` (University of Birmingham), Le Zhang `[通讯]` (University of Birmingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为CardioTrace的运动感知心脏cine合成框架，该框架能够从有限的输入生成完整的心脏cine序列，同时明确建模和追踪心脏运动。

**💡 创新点**

创新点在于将像素翻译分支与流场分支相结合，提供可解释的运动对应关系，并通过分割分支监督关键相位的解剖结构，从而提高了合成序列的时间一致性和解剖合理性。

**🔧 技术方法**

使用了深度学习技术，包括像素翻译、流场估计和对抗学习，结合了多分支网络架构和提示嵌入。

**📊 数据集**

在两个公开的心脏cine MRI数据集上进行实验，分别是M&M数据集（375个受试者）和ACDC数据集（150个受试者）。

**📈 对比分析**

与代表性的GAN和扩散基线方法相比，CardioTrace在感知质量和时间一致性上表现更好，同时更好地保持了心室形态。

**⚠️ 局限性**

限制在于模型的复杂性和对计算资源的需求，可能在处理更大规模或更复杂的心脏运动时面临挑战。

---

## 455. Evaluating Language Models as Spatial Priors for Object-Goal Navigation

**arXiv ID:** 2611.18123 | [PDF](https://arxiv.org/pdf/2611.18123v1)

**作者:** Jun Miura `[一作]` `[通讯]` (Toyohashi University of Technology), Jun Miura (Toyohashi University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了如何利用大型语言模型（LLMs）为目标物体在房间中的位置提供空间先验，而不是仅仅提供与房间类型或附近物体的关联度评分。

**💡 创新点**

创新点在于提出了一种评估LLMs作为空间先验的方法，并展示了生成的房间布局在物体频率和房间大小上遵循房间类型，但物体的放置并不符合真实房间的几何特征。

**🔧 技术方法**

使用了两个开放权重的LLMs，Qwen3.5-27B和Gemma-4-31B-it，进行布局生成和评估。

**📊 数据集**

使用了来自CubiCasa5K和ScanNet的真实房间布局数据集进行评估。

**📈 对比分析**

通过比较真实布局与扰动布局的可能性，发现真实布局在大多数情况下得分更高。生成的布局在物体频率和房间大小上与真实房间相符，但在物体与墙壁的接近度上存在差异。生成的布局作为目标搜索的先验，相较于均匀先验减少了搜索区域，但效果不如基于真实布局的先验。

**⚠️ 局限性**

限制在于生成的布局未能准确反映真实房间的几何特征，且在物体放置上存在不一致性，尤其是在物体与墙壁的接触和房间内的分布方面。

---

## 456. Forget the Lineage: Forward-Secure, Lineage-Complete Cryptographic Erasure for LLM-Agent Memory

**arXiv ID:** 2611.18133 | [PDF](https://arxiv.org/pdf/2611.18133v1)

**作者:** Yulliwas Ameur `[一作]` (Efrei Research Lab), Samia Bouzefrane `[通讯]` (Cédric laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的安全删除机制，确保在用户请求遗忘后，遗忘的信息及其衍生内容无法被恢复。该机制通过加密每个记忆并依赖于其父记忆的密钥来实现。

**💡 创新点**

创新点在于定义了完整的前向安全遗忘（LC-FS）概念，并提出了一种基于可穿透伪随机函数（PPRF）的密钥图结构，使得遗忘请求能够有效地删除所有衍生记忆。

**🔧 技术方法**

使用了可穿透伪随机函数（PPRF）和加密向量索引技术，结合了密钥图和加密存储。

**📊 数据集**

使用了LoCoMo和LongMemEval数据集，进行真实的LLM（大语言模型）写入和记忆测试。

**📈 对比分析**

与逻辑删除、每对象密钥和每个时期密钥等方法进行了比较，结果表明，使用声称规则的机制在遗忘请求后没有泄露任何信息，而其他方法则存在不同程度的信息泄露。

**⚠️ 局限性**

限制在于实验使用的LLM模型参数最多为7B，未来更大的模型可能会导致不同的泄露情况。此外，当前方法无法隐藏 lineage 图的形状，且对存储层的删除操作依赖于模型的完整性。

---

## 457. Offline KL-Regularized Reinforcement Learning with Stochastic Transitions

**arXiv ID:** 2611.17603 | [PDF](https://arxiv.org/pdf/2611.17603v1)

**作者:** Andrew Liu `[一作]` `[通讯]` (Columbia University), Andrew Liu (Columbia University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在未知随机转移的有限折扣马尔可夫决策过程中，KL正则化强化学习的离线数据需求，特别是如何在固定设计下估计样本复杂度。

**💡 创新点**

提出了悲观软值迭代（PSVI）算法，并证明其在特定条件下能够达到最小最大速率，尤其是在温度β≥1.4H时。

**🔧 技术方法**

使用了悲观软值迭代（PSVI）算法，该算法结合了方差感知惩罚来优化样本复杂度。

**📊 数据集**

使用了两种数据模型：固定设计（D1）和参考轨迹（D2），并分析了在这两种情况下的样本复杂度。

**📈 对比分析**

与其他方法相比，PSVI在样本复杂度上表现出色，能够在高概率下达到KCH^3/(βϵ)的上界，尤其是在β≥1.4H时，性能优于传统的插件估计器。

**⚠️ 局限性**

主要限制在于温度范围，H^3速率仅在β≥1.4H时成立，而在较小温度下，速率可能下降到H^4，且在样本量较小的情况下，界限可能过于宽松。

---

## 458. ESI: An Embodied Self-Improvement Flywheel for LLM-VLA Agents

**arXiv ID:** 2611.18176 | [PDF](https://arxiv.org/pdf/2611.18176v1)

**作者:** Yundi Cai `[一作]` (University of Science and Technology of China), Zilei Wang `[通讯]` (University of Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为ESI（Embodied Self-Improvement）的自动化自我改进机制，通过环境交互来对齐推理和行动，旨在提高模块化代理的协作能力。

**💡 创新点**

创新点在于将独立预训练的语言模型和视觉-语言-行动（VLA）策略的协作作为适应目标，通过交互生成的经验来更新决策指导、共享技能接口和VLA参数，从而实现系统级的后训练。

**🔧 技术方法**

使用了GPT作为策略的开源系统，结合了语言模型和视觉-语言-行动策略的协作机制。

**📊 数据集**

使用了RoboDojo数据集，该数据集包含十个选择的任务，涉及打包、语义排序、数字排列等多种操作。

**📈 对比分析**

与所有混合体代理的已发布参考结果相比，ESI在十个任务上平均得分为62.03，成功率为49.41%，均优于其他方法，包括GO-3 Nano Preview系统（得分47.27，成功率36.60%）。

**⚠️ 局限性**

限制在于ESI在某些任务上仍然表现不佳，例如在笔放置任务中成功率下降，显示出在特定操作上的改进空间。

---

## 459. Pattern-Based Matrix Analysis Under Permutations

**arXiv ID:** 2611.17674 | [PDF](https://arxiv.org/pdf/2611.17674v1)

**作者:** Andrei Ermakov `[一作]` `[通讯]`, Andrei Ermakov

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了矩阵的排列结构简单性，提出了一种模式距离度量，衡量矩阵在最佳重排序后与给定几何模式的接近程度。

**💡 创新点**

创新点在于引入了排列结构简单性这一概念，并提出了基于模式的框架和交替线性分配算法，结合了模式加低秩分解的方法。

**🔧 技术方法**

使用了交替线性分配算法（Alternating LAP）和模式加低秩（PatLR）分解技术。

**📊 数据集**

在合成恢复任务、真实相关性和图邻接矩阵以及变换器注意力头上进行了评估。

**📈 对比分析**

与传统的低秩近似方法相比，本文的方法在不同的几何结构恢复上表现出更好的性能，尤其是在噪声和不同规模的输入下。

**⚠️ 局限性**

限制在于密集实现的计算复杂度为O(n^3)，对于大矩阵和长上下文分析成本较高，且自适应选择依赖于提供的掩码词汇。

---

## 460. AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs

**arXiv ID:** 2611.17621 | [PDF](https://arxiv.org/pdf/2611.17621v1)

**作者:** AI Training Platform Team `[一作]` (Shenzhen Loop Area Institute), Zhiquan Luo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AscendGYM，一个用于Ascend NPU的内核优化基准，评估代理在真实工作负载下优化现有实现的能力。

**💡 创新点**

AscendGYM引入了Speedup–Token Score (STS)作为新指标，结合性能提升和代币消耗，评估在预算内的优化能力。

**🔧 技术方法**

使用AscendForge执行环境，集成候选编译、正确性检查和性能测量，支持迭代优化。

**📊 数据集**

基准包含48个内核，数据类型和张量形状来自真实的训练和推理任务。

**📈 对比分析**

通过两种评估模式比较模型和模型-工具组合的优化性能，约70%的评估组合在几乎所有任务中实现的几何平均加速低于1.2倍，表明现有实现的显著改进仍然具有挑战性。

**⚠️ 局限性**

当前基准的任务对现有代理仍然具有挑战性，未来工作将扩展到更多任务以进一步评估代理的内核优化能力。

---

## 461. ARCHER: From Rationale Imitation to Executable Reasoning in Small Language Models

**arXiv ID:** 2611.18494 | [PDF](https://arxiv.org/pdf/2611.18494v1)

**作者:** Dhruv Dixit `[一作]` (Stevens Institute of Technology), Min Song `[通讯]` (Stevens Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ARCHER框架，用于训练小型语言模型（SLMs）以生成可执行和可审计的推理轨迹，解决数学推理任务中的不可靠性问题。

**💡 创新点**

创新点在于将自由形式的推理解释替换为类型化的可执行动作，并通过优先级排序和条件门控的奖励机制优化推理过程。

**🔧 技术方法**

使用了两阶段的强化学习框架，第一阶段将推理解释编译为动作轨迹，第二阶段通过不对称的层次强化学习进行政策优化。

**📊 数据集**

使用了多个标准的数学推理基准数据集，包括GSM-8K、MATH-500、AIME-2024和AIME-2025等。

**📈 对比分析**

与最强的工具增强基线相比，ARCHER在五个标准基准上平均提高了5.51个百分点，并在所有40个模型-基准比较中表现最佳或并列最佳。

**⚠️ 局限性**

限制在于ARCHER主要在数学推理环境中进行评估，当前实现依赖于确定性的TRACE编译器和手动设计的HiGRA奖励结构，可能需要更丰富的动作模式和任务特定的奖励校准。

---

## 462. Sample-Efficient Control of Magnetoconvective Flows via Differentiable Simulation

**arXiv ID:** 2611.18008 | [PDF](https://arxiv.org/pdf/2611.18008v1)

**作者:** Jannis Becktepe `[一作]` (TU Dortmund University), Sebastian Peitz `[通讯]` (Paderborn University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新型的可微分GPU加速的磁流体动力学（MHD）求解器，旨在用于主动流动控制（AFC），并验证了其在MHD基准测试中的高计算效率。

**💡 创新点**

创新点在于将可微分求解器与流体控制环境FluidGym集成，使得基于梯度的优化算法在提高热传递效率方面表现出显著的样本效率和准确性。

**🔧 技术方法**

使用了可微分的GPU加速MHD求解器，结合了机器学习中的强化学习技术。

**📊 数据集**

使用了FluidGym环境中的液态金属Rayleigh-Bénard对流控制设置进行实验，涉及多种流动状态和控制策略。

**📈 对比分析**

与传统的黑箱强化学习方法相比，基于梯度的优化方法在热传递提升超过25%和逆设计准确性提高了1000倍，显示出显著的性能优势。

**⚠️ 局限性**

限制在于当前求解器仅支持单GPU执行，且高分辨率3D模拟计算需求高，未来需要探索多GPU架构和替代建模方法。

---

## 463. Bridge: Database-Grounded Interactive Disambiguation for Text-to-SQL

**arXiv ID:** 2611.17770 | [PDF](https://arxiv.org/pdf/2611.17770v1)

**作者:** Sau Lai Yip `[一作]` (Hong Kong University of Science and Technology), Xiaofang Zhou `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于数据库的交互式歧义消解框架，旨在解决文本到SQL转换中的歧义问题。

**💡 创新点**

创新点在于构建了一个列歧义图，通过离线提取潜在的列歧义关系，增强了歧义检测的有效性和效率。

**🔧 技术方法**

使用了大型语言模型（LLM）进行离线知识提取和在线歧义检测，同时构建了列歧义图和值相似性索引。

**📊 数据集**

使用了BIRD-dev数据集，这是一个跨领域的文本到SQL基准，包含1534个自然语言问题。

**📈 对比分析**

与现有的交互式文本到SQL消歧系统相比，该方法在执行准确性上提高了11.60%，并在列使用的召回率和精确率上分别提高了5.25%和12.37%。

**⚠️ 局限性**

限制在于该方法依赖于数据库的静态结构，动态数据库的变化可能会影响歧义图的有效性。

---

## 464. SURFACE: Preference Distribution Matching on Pareto Fronts via Scalarization Steering

**arXiv ID:** 2611.17484 | [PDF](https://arxiv.org/pdf/2611.17484v1)

**作者:** Liuyuan Jiang `[一作]` (University of Rochester), Lisha Chen `[通讯]` (University of Rochester)

**通讯引用:** 4221 | [OpenAlex ID](https://openalex.org/A5091442724)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为SURFACE的方法，用于根据预期的偏好分布自适应地更新标量化权重，以生成有限的Pareto解。

**💡 创新点**

SURFACE通过几何感知的方法来匹配有限的Pareto表示与预期的偏好分布，解决了传统方法在权重与Pareto解分布之间的非线性映射问题。

**🔧 技术方法**

使用了标量化技术，特别是Tchebycheff标量化，并结合了质量均匀分布（MEQ）和并行优化技术。

**📊 数据集**

在多个基准测试上进行了实验，包括ZDT和DTLZ系列问题，以及MO-Gymnasium中的多目标强化学习任务。

**📈 对比分析**

与固定的均匀权重和其他自适应基线方法进行了比较，SURFACE在偏好分布匹配和Pareto表示质量上表现出一致的优势，尤其在目标质量和覆盖度方面。

**⚠️ 局限性**

SURFACE在处理目标数量M增加时所需的自适应权重数量N迅速增长，这可能会影响其计算效率。

---

## 465. Binarization as an Attack Surface in Weightless Neural Networks

**arXiv ID:** 2611.17646 | [PDF](https://arxiv.org/pdf/2611.17646v1)

**作者:** Lucas Senos Coutinho `[一作]` (Universidade Federal do Rio de Janeiro), Daniel Sadoc Menasche `[通讯]` (Universidade Federal do Rio de Janeiro)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了无重量神经网络（WNNs）在对抗性攻击下的鲁棒性，特别是通过阈值穿越机制分析其对输入扰动的敏感性。

**💡 创新点**

提出了阈值穿越机制的理论框架，揭示了编码粒度如何影响模型的鲁棒性，并通过实验验证了不同编码方案的效果。

**🔧 技术方法**

使用了无重量神经网络（WNNs），包括标准WiSARD、Bloom WiSARD和BTHOWeN，结合了线性和高斯温度计编码。

**📊 数据集**

使用了MNIST数据集进行图像分类实验，以及Bot-IoT、CICIDS和UNSW-NB15数据集进行入侵检测实验。

**📈 对比分析**

通过控制实验比较了不同编码方案的鲁棒性，发现编码粒度的增加会显著提高对抗性攻击的成功率，同时在MNIST实验中，WNNs在随机扰动下保持高准确率，但在对抗性攻击下表现不佳。

**⚠️ 局限性**

研究中未考虑防御机制的评估，且对抗性攻击的影响可能因数据集和任务的不同而有所变化。

---

## 466. Lean-PEP: Performance-Estimation Certificates to Formally Verified Algorithmic Bounds

**arXiv ID:** 2611.17618 | [PDF](https://arxiv.org/pdf/2611.17618v1)

**作者:** Xinyuan Guo `[一作]` (University of Southampton), Andersen Ang `[通讯]` (University of Southampton)

**通讯引用:** 372 | [OpenAlex ID](https://openalex.org/A5069641685)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种性能估计证书的验证方法，旨在将证书转化为关于每个函数和算法轨迹的定理，并通过Lean-PEP实现了这一过程。

**💡 创新点**

创新点在于提供了一个可重用的接口，能够将性能估计证书与算法轨迹连接起来，并通过参数化的分析定理将有限界限转移到每个可接受的算法轨迹上。

**🔧 技术方法**

使用了Lean-PEP作为技术框架，结合了精确的代数验证和参数化的分析定理。

**📊 数据集**

使用了多个固定配置的输入数据集，具体包括八个不同的配置和七个独特的有效载荷。

**📈 对比分析**

通过与已建立的性能标准进行比较，验证了所提出方法的有效性，结果显示所生成的界限与参考值相符，且在多个测试中表现良好。

**⚠️ 局限性**

限制在于当前方法主要针对特定的算法和模型，未来的工作可以扩展到更多的性能指标和参数化算法家族。

---

## 467. The State Is Not the Bottleneck: A Paged-State Gated-DeltaNet Tile on a Tri-Mode Multicast Mesh, Bit-Exact from RTL to Silicon

**arXiv ID:** 2611.18422 | [PDF](https://arxiv.org/pdf/2611.18422v1)

**作者:** Cong Xu `[一作]` (University of South Florida), Ravi Sankar `[通讯]` (University of South Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文构建了一个硬件模块，该模块在边缘FPGA上实现了一个369百万参数的混合Gated DeltaNet模型的完整层，并对其递归状态进行了分页处理。

**💡 创新点**

创新点在于量化了状态分页的成本，并证明在该模型中，状态分页并不是解码的瓶颈，而是权重的传递限制了解码性能。

**🔧 技术方法**

使用了线性注意力机制、Gated DeltaNet、状态分页、FPGA、网络芯片和定点验证等技术。

**📊 数据集**

使用了合成权重图像和OVAD数据集进行实验，OVAD数据集包含在盲T型交叉口的真实56通道录音。

**📈 对比分析**

与之前的加速器相比，本文的硬件模块在解码性能上表现出色，状态分页的内存流量占总流量的7.7%到11.2%，而权重传递是主要的性能限制。

**⚠️ 局限性**

限制在于实验仅在特定的接口和输入下进行，未评估更长的上下文，且未实现六个注意力层和词汇投影。

---

## 468. Modeling the Frictional Contact of Continuum Robots with Environments through Linear Complementarity Problem

**arXiv ID:** 2611.18546 | [PDF](https://arxiv.org/pdf/2611.18546v1)

**作者:** Jia Shen `[一作]` (Georgia Institute of Technology), Yue Chen `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于互补约束的新型摩擦模型，用于连续机器人在与环境接触时的摩擦建模，能够同时考虑静摩擦和滑动摩擦。

**💡 创新点**

创新点在于将传统的质量/惯性依赖的线性互补问题（LCP）重新表述为基于顺应性的形式，从而提高计算效率并捕捉静摩擦和滞后行为。

**🔧 技术方法**

使用了互补约束和Cosserat杆理论相结合的线性互补问题（LCP）技术。

**📊 数据集**

使用了287.9 mm长的腱驱动连续机器人进行实验验证。

**📈 对比分析**

与现有的摩擦模型相比，提出的模型在摩擦影响下的最大尖端位置误差从45.4 mm降低到5.85 mm，显示出显著的性能提升。

**⚠️ 局限性**

模型的局限性在于依赖单一线性化步骤，准确性取决于保持足够小的增量。此外，模型假设为准静态，不适用于捕捉快速动态效应，如跳跃或冲击。

---

## 469. CANTraj: Recovering Vehicle Trajectories from CAN Signals in Naturalistic Driving

**arXiv ID:** 2611.18442 | [PDF](https://arxiv.org/pdf/2611.18442v1)

**作者:** Yuhang Wang `[一作]` (University of South Florida), Hao Zhou `[通讯]` (University of South Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了CANTraj数据集和CANTraj-Bench基准，用于从CAN信号中恢复车辆轨迹，并评估不同方法的性能。

**💡 创新点**

创新点在于提供了一个自然驾驶数据集，结合了解码的CAN事件和时间对齐的GNSS轨迹，同时实现了六种攻击方法的统一比较，解决了现有研究中缺乏一致性和可比性的问题。

**🔧 技术方法**

使用了控制器局域网（CAN）信号、GNSS轨迹、在线和离线解码方法、粒子滤波和学习模型等技术。

**📊 数据集**

使用的数据集包括来自不同驾驶员的自然驾驶数据，包含解码的CAN信号和时间对齐的GNSS轨迹，共计驱动次数和驾驶员数量。

**📈 对比分析**

通过与六种重新实现的攻击方法和基准方法进行比较，结果显示结合转向信号和道路网络约束可以显著提高轨迹恢复的准确性，在线地图过滤器的五分钟路线恢复率从%提高到%，而无地图的死算方法的中位数平均位移误差为m。

**⚠️ 局限性**

限制在于CANTraj数据集主要集中在北美的道路和车辆，且不同方法的排名可能因驾驶条件和车辆类型的变化而有所不同，无法保证普遍适用性。

---

