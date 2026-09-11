# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-11 | 今日论文总数: 538

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Why Customer Choice Models Matter

**arXiv ID:** 2609.10557 | [PDF](https://arxiv.org/pdf/2609.10557v1)

**作者:** Berry Gerrits `[一作]` (University of Twente), Fabian Akkerman `[通讯]` (University of Twente)

**通讯引用:** 109 | [OpenAlex ID](https://openalex.org/A5032023486)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文探讨了客户选择模型（CCM）在家庭送货中的收入管理中的重要性，指出选择模型的选择可能会严重影响算法结果的有效性。

**💡 创新点**

创新点在于揭示了三种常见的误区：假设完美的客户选择知识、模型参数的不确定性以及客户细分的建模提供了更好的CCM粒度。

**🔧 技术方法**

使用了多种客户选择模型，包括条件逻辑模型、多项逻辑模型、混合逻辑模型、嵌套逻辑模型等，分析了它们在收入管理中的应用。

**📊 数据集**

使用了模拟数据集，研究了不同客户细分和选择模型对时间段定价的影响。

**📈 对比分析**

通过实验比较了不同选择模型的预期利润，发现错误的模型选择会导致利润显著下降，尤其是在客户群体异质性较高的情况下。

**⚠️ 局限性**

限制在于大多数文献假设客户群体是同质的，而实际情况往往是异质的，导致模型的适用性和准确性受到影响。

---

## 2. GEOSTEER: Geodesic Optimization for Activation Steering in Large Language Models

**arXiv ID:** 2609.10658 | [PDF](https://arxiv.org/pdf/2609.10658v1)

**作者:** Xuan Cuong Ngo `[一作]` (University of Arkansas), Ngan Le `[通讯]` (University of Arkansas)

**通讯引用:** 6883 | [OpenAlex ID](https://openalex.org/A5108408962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于优化的范数保持激活引导方法，通过在推理时对隐藏激活进行调整来控制大型语言模型的行为。

**💡 创新点**

创新点在于将激活引导视为一个黎曼优化问题，通过一系列小的测地线步骤更新激活，而不是依赖于预定义的单步更新，从而实现更平滑和稳定的引导行为。

**🔧 技术方法**

使用了黎曼优化技术，结合非线性激活空间目标来指导每一步的引导。

**📊 数据集**

在TruthfulQA、RealToxicityPrompts和UltraFeedback等基准数据集上进行了实验。

**📈 对比分析**

与现有的激活引导基线相比，提出的方法在所有评估模型上均表现出更好的性能，特别是在帮助性、真实性和去毒化任务中，显示出更高的胜率和更低的毒性评分。

**⚠️ 局限性**

局限性包括与其他方向发现方法的集成问题，以及潜在的风险和意外行为，尤其是在不当目标或引导方向下可能导致不良输出。

---

## 3. Threshold Choice, Not Sample Size, Bounds Trustless Verification of Nondeterministic Compound AI Workflows

**arXiv ID:** 2609.10601 | [PDF](https://arxiv.org/pdf/2609.10601v1)

**作者:** Alper Alimoglu `[一作]` `[通讯]` (Independent Researcher), Alper Alimoglu (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种协议，用于验证在不确定性复合AI工作流中的语义再现，涵盖了输入、输出、上下文和策略的承诺，并在没有法定人数的情况下基于k次重执行的中位数进行挑战。

**💡 创新点**

创新点在于该协议同时解决了不确定性输出、执行节点可能不诚实和共享记录间歇性可达性这三个问题，且提出了基于每次执行的阈值而非样本大小的检测机制。

**🔧 技术方法**

使用了一种承诺方案，结合了内容寻址存储和去中心化账本技术，确保每个阶段的输入、输出和上下文在执行边界处被记录和锚定。

**📊 数据集**

使用了合成的HotpotQA数据集，进行了一系列实验以验证协议的有效性和性能。

**📈 对比分析**

与现有方法相比，本文的方法在检测同输入伪造方面表现出色，能够在k=5的情况下检测到19个伪造案例，而固定阈值的最佳常数仅能检测到9个，且没有拒绝任何诚实承诺。

**⚠️ 局限性**

限制在于该研究是一个概念验证，使用的管道是合成的，未在真实的边缘、云和轨道节点上进行测试，且协议未能提供执行来源的证明。

---

## 4. Feasible disjunction for random resolution

**arXiv ID:** 2609.10602 | [PDF](https://arxiv.org/pdf/2609.10602v1)

**作者:** Theodoros Papamakarios `[一作]` `[通讯]`, Theodoros Papamakarios

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文展示了一种随机分辨率的（更强）版本具有可行的析取性质，这是第一个已知的没有可行插值的证明系统，但却具有可行的析取性质的实例。

**💡 创新点**

创新点在于识别出一种扩展的随机分辨率证明系统，它具有可行的析取性质，但尚未证明其具有可行插值性质。

**🔧 技术方法**

使用了随机分辨率的证明技术，特别是基于最小最大定理的证明方法。

**📊 数据集**

使用了CNF公式F和G的组合，且F和G不共享任何变量。

**📈 对比分析**

与其他证明系统的比较表明，随机分辨率的动态错误和静态错误的性质不同，且动态错误的证明系统在某些情况下表现出更强的性质，但具体性能尚未完全确定。

**⚠️ 局限性**

限制在于随机分辨率的动态错误的可行插值性质仍然是一个开放问题，且该证明系统不是传统意义上的证明系统，无法在多项式时间内验证给定的反驳是否有效。

---

## 5. ReqEvolve: User-Oriented Software Self-Evolution through Automatic Requirement Interpretation

**arXiv ID:** 2609.10590 | [PDF](https://arxiv.org/pdf/2609.10590v1)

**作者:** Md Asif Iqbal Fahim `[一作]` (University College Dublin), Alessio Ferrari `[通讯]` (University College Dublin)

**通讯引用:** 3282 | [OpenAlex ID](https://openalex.org/A5041720518)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为ReqEvolve的运行时代码生成系统，旨在通过接受高层次的用户请求来实现用户驱动的软件自我演化。

**💡 创新点**

创新点在于将自动需求工程（RE）和测试驱动开发（TDD）结合，允许用户直接通过自然语言接口表达需求，从而加速需求验证。

**🔧 技术方法**

使用了自动需求工程（RE）和测试驱动开发（TDD）技术，结合了多个LLM（大语言模型）组件进行代码生成和验证。

**📊 数据集**

使用了72个软件演化案例，涵盖18个项目，项目代码行数从40到3100不等，设计了4个用户故事以触发自我演化。

**📈 对比分析**

与SpecFix（一个以RE为中心的代码生成方法）和一个去除RE组件的消融变体进行比较，ReqEvolve在性能上显著优于SpecFix（提高了18.8%）和消融基线（提高了32.6%），达到了89.2%的Pass@1。

**⚠️ 局限性**

限制在于生成的代码可能会出现集成问题，尤其是在需要调用先前生成的函数时，且LLM在处理复杂的用户请求时可能会产生错误的代码。

---

## 6. Who Pays for a Connected Public Good?

**arXiv ID:** 2609.10565 | [PDF](https://arxiv.org/pdf/2609.10565v1)

**作者:** Marco Tulio Angulo `[一作]` `[通讯]` (Universidad Nacional Autonoma de Mexico), Marco Tulio Angulo (Universidad Nacional Autonoma de Mexico)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

研究了在固定网络中公共物品的自愿融资问题，分析了参与者的退出责任如何限制可承担的运营成本并决定谁必须支付

**💡 创新点**

提出了基于网络位置的“退出责任”作为支付上限的新度量，并用极值图论证明了路径网络在给定规模下最大化可承受成本、最小化随机退出后残存输出的特性

**🔧 技术方法**

运用了组合图论、博弈论（Nash equilibrium与支持性规则）以及期望值分析，构造了成本分配规则并证明了其存在性与唯一性

**📊 数据集**

本文无数据集，纯理论分析

**📈 对比分析**

通过对比不同网络结构（路径、星形、完全图等）以及不同成本水平下的可支持性与支付区间，展示了路径在可承受成本上最优但在残存输出方面最弱；理论结果与图形举例相匹配

**⚠️ 局限性**

假设网络固定、无权重、无定向，参与二元且同时做决定；未考虑成本随参与度变化、网络自发形成、信息不完全或顺序决策，可能限制实际应用

---

## 7. Quantifying the Memorization-to-Generalization Transition: Scaling Laws and Phase Structure in Grokking

**arXiv ID:** 2609.10657 | [PDF](https://arxiv.org/pdf/2609.10657v1)

**作者:** Anish Kataria `[一作]` (Princeton University), Anish Kataria `[通讯]` (Princeton University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5108329045)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究量化了神经网络从记忆到泛化的过渡，特别是通过384种超参数配置映射了记忆到泛化的边界，并拟合了泛化开始时间的幂律缩放关系。

**💡 创新点**

创新点在于提出了一个幂律缩放关系，揭示了数据复杂性在过渡中的主导作用，而非模型容量，并确定了一个在超参数空间中的相位边界。

**🔧 技术方法**

使用了两层隐藏层的多层感知器（MLP）架构，并进行了超参数的系统性扫描，拟合了泛化开始时间的幂律关系。

**📊 数据集**

使用了模113的加法和模97的除法作为数据集，这些是已知的经典grokking基准。

**📈 对比分析**

通过与其他方法的比较，发现数据复杂性是影响泛化开始时间的主要因素，模型容量的影响较小。性能上，增加数据量可以显著加快泛化速度，而增加模型宽度的效果较小。

**⚠️ 局限性**

限制在于该缩放法则是基于两层隐藏层的MLP模型在模算术上的拟合，可能不适用于变换器或自然语言任务；此外，模型的宽度指数估计基于有限的离散水平，且R^2值显示仍有27%的方差未被解释。

---

## 8. Optimal Networks with Accumulative Costs and Bidirectional Communication

**arXiv ID:** 2609.10546 | [PDF](https://arxiv.org/pdf/2609.10546v1)

**作者:** Juan M. C. Larrosa `[一作]` (Universidad Nacional del Sur), Fernando Tohmé `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文扩展了Larrosa和Tohmé（2003）的方法，通过允许信息双向流动来修改收益函数。研究发现，几种最优拓扑结构作为纳什网络存在，但严格的纳什网络对应于具有中间活跃节点的顺序线性网络。

**💡 创新点**

创新点在于引入了双向信息流的概念，并探讨了在这种情况下的最优通信拓扑结构，发现顺序线性网络能够在最小连接数下实现最大信息流。

**🔧 技术方法**

使用了博弈论工具，特别是非合作博弈的框架来分析网络形成和稳定性。

**📊 数据集**

未具体提及使用的数据集，但研究基于理论模型和示例进行分析。

**📈 对比分析**

通过与单向信息流的情况进行比较，发现双向流动的网络结构在连接成本和信息获取方面具有优势，严格的纳什网络是具有偶数或奇数活跃节点的顺序连接线网络。

**⚠️ 局限性**

限制在于累积成本函数对网络扩展的可行性产生了很大限制，导致难以找到广泛的网络作为严格的纳什网络。

---

## 9. Sparse Weight and Edge Circuit Discovery in Transformer-based Acoustic Models

**arXiv ID:** 2609.10645 | [PDF](https://arxiv.org/pdf/2609.10645v1)

**作者:** Jiankun Wei `[一作]` (University of Toronto), Gerald Penn `[通讯]` (University of Toronto)

**通讯引用:** 6855 | [OpenAlex ID](https://openalex.org/A5052428595)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究扩展了DiscoGP框架，首次在现代语音基础模型（HuBERT和Wav2Vec 2.0）中进行电路发现，揭示了紧凑的任务特定电路。

**💡 创新点**

创新点在于将电路发现方法从文本解码器扩展到语音编码器，并提出了一种内存高效的DiscoGP变体，显著降低了边电路发现的GPU内存成本。

**🔧 技术方法**

使用了DiscoGP框架，该框架通过联合学习权重和边掩码来发现电路，采用了基于梯度的掩码学习方法。

**📊 数据集**

使用了多个数据集，包括Articulatory Index、Speech Commands v1.0、IEMOCAP和ASVspoof 2019，进行元音、辅音、情感识别和欺骗检测等分类任务。

**📈 对比分析**

与随机初始化模型和随机基线相比，DiscoGP发现的电路在多个任务上保持了与完整预训练编码器相当的性能，且在许多情况下超越了完整模型的表现。

**⚠️ 局限性**

限制在于本研究仅限于两种编码器（HuBERT和Wav2Vec 2.0）和英语数据集，未来需要扩展到其他架构和语言，并验证在回归任务上的电路发现。

---

## 10. Byzantine-Robust Federated Fire Detection with a Rotating Coordinator

**arXiv ID:** 2609.10647 | [PDF](https://arxiv.org/pdf/2609.10647v1)

**作者:** Georgia Argyrou `[一作]` (Aalto University), Alexander Jung `[通讯]` (Kudelski Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套基于联邦学习的室内火灾检测系统，解决了通信效率、Byzantine鲁棒性和单点失效三大难题。

**💡 创新点**

①构建了18,790张图像的室内火灾数据集；②利用冻结的MobileNet‑V2仅训练轻量化头部，实现10×压缩的边缘可部署检测器；③将历史感知的SafeguardSGD与旋转协调器结合，形成半去中心化的Byzantine鲁棒聚合机制。

**🔧 技术方法**

联邦学习（FedAvg/SafeguardSGD）、INT8/INT4量化与Top‑K稀疏化、历史感知Byzantine检测、旋转协调器、Grad‑CAM可解释性、AWS EC2分布式部署等技术。

**📊 数据集**

自主整理的8大公开来源合并的18,790张室内火灾/非火灾图像数据集，并按IID与非IID方式划分用于联邦评估。

**📈 对比分析**

与FedAvg基准和集中式模型对比，测试平衡精度最高可达0.9835，压缩10×后仅下降不到0.008；在对抗攻击下，SafeguardSGD能在3–16轮内及时驱逐攻击者，保持与无攻击基准相近的准确率。

**⚠️ 局限性**

仅考虑单一静态不协作的Byzantine客户端，客户端数量有限（5个），未在真实边缘设备上验证实时性能，缺乏可验证聚合以及多客户端协同攻击的评估。

---

## 11. Halo: Improving forecast accuracy through heteroscedastic estimation

**arXiv ID:** 2609.10589 | [PDF](https://arxiv.org/pdf/2609.10589v1)

**作者:** Adam Cataldo `[一作]` `[通讯]`, Adam Cataldo

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Halo的技术，通过修改现有的深度学习时间序列预测架构，使其能够同时估计位置参数和尺度参数，从而提高预测精度。

**💡 创新点**

Halo技术的创新点在于它能够在不需要特定架构设计的情况下，改善点估计的准确性，并且在多个模型和市场中表现出显著的性能提升。

**🔧 技术方法**

使用了深度神经网络（DNN），具体包括改编的变换器、图神经网络（GNN）与变分自编码器（VAE）结合的模型，以及单层卷积网络（CNN）。

**📊 数据集**

使用了电力价格预测（EPF）基准数据集，包括五个不同市场的电力价格数据：Nord Pool (NP)、PJM、比利时（BE）、法国（FR）和德国（DE）市场。

**📈 对比分析**

与没有Halo的模型进行比较，Halo在30个模型-市场-指标的比较中改善了28个，平均均方误差（MSE）降低了2.6%到16.5%，平均绝对误差（MAE）降低了1.7%到11.0%。

**⚠️ 局限性**

限制在于Halo的效果可能依赖于特定的数据集和超参数设置，未来需要在其他基准和不同的时间范围内进行测试以验证其普适性。

---

## 12. Probabilistic Focal Search: Accelerating Bounded-Suboptimal Search via Lower-Bound Advancement

**arXiv ID:** 2609.10584 | [PDF](https://arxiv.org/pdf/2609.10584v1)

**作者:** Minh Vu Duc `[一作]` (National Economics University), Huynh Thi Thanh Binh `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 2763 | [OpenAlex ID](https://openalex.org/A5072105691)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的概率焦点搜索（PFS）算法，该算法通过在引导选择和扩展最小f OPEN节点之间进行概率选择，来提高有界次优搜索的效率。

**💡 创新点**

创新点在于引入了概率选择机制，使得在搜索过程中能够更灵活地平衡引导和下界的推进，从而减少搜索时间并提高成功率。

**🔧 技术方法**

使用了概率调度（Bernoulli调度）技术，结合了焦点搜索（FS）和动态潜力搜索（DPS）的方法。

**📊 数据集**

使用了N-Puzzle、煎饼排序和旅行推销员问题（TSP）等多个数据集进行实验，并在广义覆盖TSP（GCTSP）上评估了其随时扩展能力。

**📈 对比分析**

与FS算法进行比较，PFS在N-Puzzle和TSP上显著提高了成功率，减少了节点扩展数量，尤其是在f_min停滞的情况下，PFS的节点扩展减少了约90%。在GCTSP的随时算法中，PFS的表现优于所有测试的算法。

**⚠️ 局限性**

限制在于当确定性搜索已经有效推进时，概率选择的好处较小，例如在煎饼排序中，初始的FOCAL已经足够，因此PFS的改进效果有限。

---

## 13. Generative AI for trustworthy systems - Towards a health check model

**arXiv ID:** 2609.10595 | [PDF](https://arxiv.org/pdf/2609.10595v1)

**作者:** Jan Bosch `[一作]` (Chalmers University of Technology), Helena Holmström Olsson `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了面向生成式人工智能（GenAI）的可信自治健康检查模型（Trustworthy Autonomy Health Check Model），用以多维度评估组织在 GenAI 辅助软件工程中的可信度构造；

**💡 创新点**

创新点在于突破传统单一成熟度模型的限制，构建了包含系统层与组织层八个维度（如 Agent Authority、Assurance Mechanisms、Data Trustworthiness 等）以及四种信任范式（Operational、Engineering、Statistical、Containment-based）的结构化配置空间；同时提出“对齐假设”（Alignment Hypothesis）作为可检验的理论主张；

**🔧 技术方法**

主要技术手段为质性研究方法——三支团队独立进行半结构化访谈，采用开放编码、轴向编码与选择性编码构建模型；在访谈分析过程中利用 LLM 辅助摘要并进行人工验证；

**📊 数据集**

数据集为18位高级技术/战略从业者的访谈记录，涵盖通信、汽车、国防、航空、银行、能源、政府和企业软件服务等八大行业；

**📈 对比分析**

本研究不涉及实验性性能对比，而是通过案例比较展示不同组织在各维度上的配置差异，强调多维度诊断而非单一指标；

**⚠️ 局限性**

限制包括：样本规模有限且主要集中在欧美和东欧地区，欠缺对亚洲、拉美、非洲等监管环境多样性的覆盖；方法上依赖访谈的自我报告，可能存在偏差；模型未涵盖公平、偏见等伦理维度，且未对对齐假设进行定量验证。

---

## 14. An Empirical Measurement of Jailbreaking Evaluators

**arXiv ID:** 2609.10594 | [PDF](https://arxiv.org/pdf/2609.10594v1)

**作者:** Yujie Mu `[一作]` `[通讯]` (Independent Researcher), Yujie Mu (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统地比较了六种常用的越狱评估器（HarmBench、JailbreakBench、JailbreakRadar、StrongReject、JADES和JailMeter），在统一的评估协议下使用两个人工标注的数据集进行评估。

**💡 创新点**

创新点在于首次在相同的人类标注数据和受控设置下对六种评估器进行系统比较，提供了对当前越狱评估器的统一实证评估。

**🔧 技术方法**

使用了共享的通用大型语言模型（LLM）作为评估者的基础，以控制模型特定的变异性，并对评估方法的差异进行隔离。

**📊 数据集**

使用了两个人工标注的数据集：JailbreakQR和JailMeter-Eva，前者包含400个样本，后者包含330个样本，涵盖不同的攻击类型和标签粒度。

**📈 对比分析**

通过与人类判断的协议一致性进行比较，结果显示JADES在两个数据集上表现最佳，StrongReject和HarmBench也表现良好，而JailbreakBench和JailbreakRadar的表现明显较差。

**⚠️ 局限性**

本研究的局限性在于数据集的范围有限，均为英语且规模适中，可能无法推广到其他语言、领域或未来的越狱策略。此外，人工标注本身也存在不完美性，可能影响评估结果的准确性。

---

## 15. MUC-FL: Block-Wise Marginal Utility Contribution for Communication-Efficient Federated Learning

**arXiv ID:** 2609.10545 | [PDF](https://arxiv.org/pdf/2609.10545v1)

**作者:** Akshay Mhatre `[一作]` (Texas A&M University), Jia Zou `[通讯]` (Arizona State University)

**通讯引用:** 793 | [OpenAlex ID](https://openalex.org/A5013735333)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种新的框架Block-Wise Marginal Utility Contribution (MUC)，旨在通过选择性传输对模型性能影响最大的数据信息，来降低联邦学习中的通信开销。

**💡 创新点**

创新点在于通过量化每个数据块对模型性能的贡献，选择性地传输最有价值的数据块，从而显著减少通信负担，同时保持或提高模型质量。

**🔧 技术方法**

使用了块级选择和去重的技术，通过评估每个候选块对模型性能的影响来进行选择。

**📊 数据集**

使用了集成自多个MIMIC临床数据集的多模态数据集进行评估，包含医学图像、自由文本放射学报告和医院及患者记录。

**📈 对比分析**

与标准的联邦优化方法进行比较，MUC方法在宏F1分数上达到了0.8566，而标准方法为0.8155，显示出选择性传输在提高模型性能方面的有效性，尤其是在代表性不足的类别中。

**⚠️ 局限性**

限制在于该方法可能在某些情况下无法充分利用所有数据块的潜在信息，特别是在数据分布不均的情况下，可能导致某些重要信息的遗漏。

---

## 16. Threshold-Based Selection for Continuous Optimization: A Leaf-Abscission Instantiation

**arXiv ID:** 2609.10588 | [PDF](https://arxiv.org/pdf/2609.10588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 17. A Multi-Stage Rule-Chaining Framework for Compositional and Interpretable Cognitive Reasoning

**arXiv ID:** 2609.10654 | [PDF](https://arxiv.org/pdf/2609.10654v1)

**作者:** Deblina Kar `[一作]` (Indian Institute of Technology Kharagpur), Deblina Kar `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 119 | [OpenAlex ID](https://openalex.org/A5113322590)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个多阶段规则链框架，先用确定性规则发现解读输入输出的几何与颜色变换，再通过块级结构合成和模式递归完成多块与循环任务，最后在抽象层面利用层级推理与LLM辅助完成隐式关系与嵌套结构的推断，整体以可解释的方式解决ARC与ARC‑AGI‑2的任务；

**💡 创新点**

创新点在于将符号规则推导、感知CNN引导、块级合成与抽象推理有机层级化，并引入层级回退机制与LLM辅助推理，实现从简单几何到复杂抽象的顺序迁移与共享推理轨迹；

**🔧 技术方法**

采用符号规则生成与评估、CNN相似性度量、块级合成/对齐、循环/螺旋生成、抽象嵌套推理以及LLM作为策略监督等技术；

**📊 数据集**

使用ARC原始数据集（1000训练、120验证、240测试）和扩展版ARC‑AGI‑2数据集；

**📈 对比分析**

与主流符号与神经基线对比，整体准确率达到95.4%，在测试集上成功230/240个任务，单阶段性能分别为83.2%（确定性）、92.8%（组合）和95.4%（抽象），平均推理时间从900s降至37s；

**⚠️ 局限性**

仍受限于极端抽象或多层嵌套任务、LLM整合不完整、对未见任务的鲁棒性不足，且整体推理成本相对较高

---

## 18. Artificial Intelligence Algorithms for the Detection of Pathologies Related to Lung Cancer through Image Analysis using Convolutional Neural Networks and Data Augmentation: a systematic mapping of the literature

**arXiv ID:** 2609.10652 | [PDF](https://arxiv.org/pdf/2609.10652v1)

**作者:** Pablo Ramirez Amador `[一作]` `[通讯]`, Pablo Ramirez Amador

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究回顾了人工智能（AI）和深度学习（DL）在肺癌检测中的应用，特别是通过医学影像分析来提高早期诊断的准确性和效率。

**💡 创新点**

创新点在于强调了卷积神经网络（CNN）与迁移学习和数据增强相结合的技术，以提高图像解读的准确性和效率。

**🔧 技术方法**

使用了卷积神经网络（CNN）、迁移学习和数据增强等技术。

**📊 数据集**

研究中选取了2015年至今在PubMed、IEEEXPLORE、Scopus和Web of Science等数据库中发表的96篇相关文献。

**📈 对比分析**

通过系统性映射分析，比较了不同预训练CNN的性能，结果显示AI和DL在肺癌早期诊断中具有高灵敏度和特异性，但也指出了当前应用中的局限性和挑战。

**⚠️ 局限性**

局限性包括缺乏标准化数据、模型的可解释性、患者隐私问题以及伦理和社会影响等。

---

## 19. Numbat: Building and Verifying a Self-Contained Machine-Learning Stack

**arXiv ID:** 2609.10632 | [PDF](https://arxiv.org/pdf/2609.10632v1)

**作者:** Thang Tran `[一作]` (CloudKites AI Lab), Lan Dang `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e0540dec-d77f-42db-94ae-d039248f6393` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建并验证了一个名为numbat的机器学习框架，该框架使用Zig语言编写，且没有第三方运行时依赖，涵盖了从张量计算到多GPU训练的完整功能。

**💡 创新点**

创新点在于提供一个自包含的机器学习堆栈，消除了传统框架的复杂性和高成本，同时通过可执行的接受门来编码领域要求，确保了框架的稳定性和可验证性。

**🔧 技术方法**

使用Zig语言构建，涉及张量计算、自动微分、神经网络模块、混合精度、多GPU训练等技术。

**📊 数据集**

使用COCO 2017数据集进行训练和验证，包含118,287张训练图像和5,000张验证图像。

**📈 对比分析**

通过与现有的PyTorch实现进行比较，numbat在相同硬件上实现了相似的性能，且在多个层面上进行了验证，确保了结果的可重复性和准确性。

**⚠️ 局限性**

限制在于只使用了单一的随机种子进行实验，未进行多种随机种子的变异研究，且在小核心主机上的多GPU扩展存在性能差距。

---

## 20. Adaptive Diffusion Freezing: Privacy-preserving Diffusion Models Against Membership Inference Attacks

**arXiv ID:** 2609.10608 | [PDF](https://arxiv.org/pdf/2609.10608v1)

**作者:** Jialu Guo `[一作]` (Beihang University), Junjie Wu `[通讯]` (Beihang University)

**通讯引用:** 132268 | [OpenAlex ID](https://openalex.org/A5100364769)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的隐私保护扩散模型框架，称为自适应扩散冻结（ADF），旨在防御成员推断攻击（MIA），并在隐私、效用和效率之间实现更好的平衡。

**💡 创新点**

创新点在于通过跨时间步自适应冻结训练，显式控制不同数据子集在扩散时间步中的参与，从而减少过度记忆并实现成员和非成员样本之间更均匀的模型行为。

**🔧 技术方法**

使用了自适应冻结训练和风险感知冻结策略，结合了掩码矩阵来控制训练样本在不同时间步的贡献。

**📊 数据集**

在多个数据集上进行了评估，包括CIFAR-10、STL10_U、CelebA和NWPU-RESISC45。

**📈 对比分析**

与五种基线方法进行了比较，ADF在隐私保护和生成效用之间实现了更好的权衡，表现出更低的攻击成功率（ASR）和更高的生成质量（FID）。

**⚠️ 局限性**

限制在于现有的MIA防御方法仍然有限，ADF的有效性可能在某些极端情况下受到影响，且在实现过程中可能增加计算复杂性。

---

## 21. Rethinking Handwritten Character Recognition

**arXiv ID:** 2609.10572 | [PDF](https://arxiv.org/pdf/2609.10572v1)

**作者:** Ranjit Raut `[一作]` (Kathmandu University), Ashim Shrestha `[通讯]` (Kathmandu University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的多脚本手写字符识别架构GraphemeNet，旨在通过显式编码书写系统的几何规律来提高识别精度并减少参数数量。

**💡 创新点**

创新点在于引入了持久性支架注入（PSI）和线性胶囊路由（LCR），通过这两种正交设计轴来优化多脚本的识别性能。

**🔧 技术方法**

使用了持久性支架注入（PSI）、线性胶囊路由（LCR）、多尺度全局平均池化和交叉尺度注意力等技术。

**📊 数据集**

使用了包括Devanagari、Bengali、Kannada、Arabic、Persian、Japanese、Thai和English在内的多个数据集，共计14个基准测试。

**📈 对比分析**

与现有的基准相比，GraphemeNet在多个数据集上表现出色，尤其是在Devanagari上达到了99.75%的准确率，且参数数量显著低于其他模型，如ResNet-85和MallaNet。

**⚠️ 局限性**

局限性在于该架构可能在处理某些复杂书写系统时仍然面临挑战，特别是当几何结构过于复杂时，可能需要更丰富的模型设计。

---

## 22. Memory Profiling and Migration for Heterogeneous Memory Architectures

**arXiv ID:** 2609.10554 | [PDF](https://arxiv.org/pdf/2609.10554v1)

**作者:** Marios Asiminakis `[一作]` (Foundation for Research and Technology Hellas), Manolis Marazakis `[通讯]` (Foundation for Research and Technology Hellas)

**通讯引用:** 675 | [OpenAlex ID](https://openalex.org/A5079517640)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SHAMBLES，一个内核集成的内存分析和迁移框架，能够在不需要应用程序修改的情况下，自动化地管理异构内存系统中的数据放置。

**💡 创新点**

SHAMBLES提供了一个政策无关的接口和轻量级的用户空间运行时，支持可插拔的策略，能够根据应用程序的内存行为动态迁移数据。

**🔧 技术方法**

使用了内核集成的内存分析和迁移机制，结合轻量级的页面故障采样，能够在运行时监控内存访问模式并进行数据迁移。

**📊 数据集**

在Intel Xeon Max HBM+DDR平台上进行评估，使用HPCG、DGEMM基准测试和Himeno迷你应用程序。

**📈 对比分析**

与静态放置方法进行比较，SHAMBLES的动态策略在HPCG基准测试中保持了高达93.75%的全HBM性能，同时仅在HBM中保留40%的问题规模；在DGEMM测试中，动态策略维持了高达99%的全HBM性能，且仅使用三分之一的矩阵足迹。

**⚠️ 局限性**

SHAMBLES目前仅基于采样的页面故障活动选择热页面，缺乏对迁移决策影响的反馈，未来可以通过引入硬件性能计数器来改进策略选择。

---

## 23. Black-Box Membership Inference via Word-Level Probability Estimation

**arXiv ID:** 2609.10611 | [PDF](https://arxiv.org/pdf/2609.10611v1)

**作者:** Shengjie Niu `[一作]` (Hong Kong Polytechnic University), Jian Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 14493 | [OpenAlex ID](https://openalex.org/A5012067697)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的黑箱成员推断攻击方法WPMIA，用于审计大型语言模型的隐私风险，能够在没有访问每个标记的logits或概率的情况下，估计文本是否包含在模型的训练语料中。

**💡 创新点**

WPMIA通过蒙特卡洛采样和局部核平滑来估计单词级生成概率，并通过对不同前缀的条件化构建似然性，从而增强成员和非成员之间的分布差异。

**🔧 技术方法**

使用了蒙特卡洛采样和语义核平滑技术来估计单词级生成概率，并通过对比前缀增强来提高成员信号。

**📊 数据集**

在多个开放源代码的LLM基准（WikiMIA、MIMIR和WikiMIA-25）上进行了评估，并在现代专有LLM（如GPT-5-Chat、Gemini-2.5-Flash和Claude-4.5-Haiku）上进行了测试。

**📈 对比分析**

WPMIA在开放源代码LLM上表现出色，超越了现有的黑箱基线，并在多个设置中与灰箱方法相当。在专有LLM上，WPMIA的TPR@5%FPR平均为42.0，显示出其在严格黑箱隐私审计中的有效性。

**⚠️ 局限性**

WPMIA的局限性在于它需要重复采样以恢复成员信号，这在长文本和专有LLM API中可能导致额外的计算和延迟成本。此外，当前评估仅限于英语文本，未来需要在多语言基准上进行评估。

---

## 24. MLN-EIGS: A multilayer network framework for solving Stackelberg escape interdiction games on dynamic transportation networks

**arXiv ID:** 2609.10556 | [PDF](https://arxiv.org/pdf/2609.10556v1)

**作者:** Sukanya Samanta `[一作]` (Kyushu University), Palash Dey `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 433 | [OpenAlex ID](https://openalex.org/A5078621796)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于多层网络的框架MLN-EIGS，用于解决动态逃逸拦截问题，该问题被建模为Stackelberg安全博弈。

**💡 创新点**

创新点在于将动态概率拦截建模与Stackelberg安全博弈框架结合，使用多层时间扩展网络来捕捉攻击者和防御者的时间可行性。

**🔧 技术方法**

使用了Dijkstra算法来解决攻击者的最佳响应问题，并开发了一个多项式时间的近似防御者oracle来生成高质量的防御者策略。

**📊 数据集**

在一个大型真实交通网络（中央加尔各答交通网络）上进行了计算实验，网络包含461个节点和1020条有向边。

**📈 对比分析**

与基于混合整数线性规划（MILP）的精确Stackelberg模型进行比较，MLN-EIGS在防御者效用上表现相近，但计算时间显著减少，显示出更好的计算效率。

**⚠️ 局限性**

限制在于防御者最佳响应问题的计算复杂性，尽管提出了近似oracle，但仍然无法保证全局最优解。

---

## 25. Zero-shot rib design: merging training-free generative prior with topology optimization

**arXiv ID:** 2609.10643 | [PDF](https://arxiv.org/pdf/2609.10643v1)

**作者:** Yongmin Kwon `[一作]` (Korea Advanced Institute of Science and Technology), Namwoo Kang `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2085 | [OpenAlex ID](https://openalex.org/A5016809344)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本研究提出了一种零-shot肋骨设计框架，将无训练的生成先验与基于密度的拓扑优化相结合。通过将冻结的文本到图像扩散模型与有限元分析（FEA）相结合，利用文本提示作为工程设计意图的明确知识表示。

**💡 创新点**

创新点在于首次将得分蒸馏采样（SDS）应用于物理约束的拓扑优化，并通过将生成先验与物理敏感性在每次迭代中结合，允许工程师通过自然语言表达设计意图。

**🔧 技术方法**

使用了冻结的Stable Diffusion 2.1模型和得分蒸馏采样（SDS）技术，结合有限元分析（FEA）进行物理约束的优化。

**📊 数据集**

使用了四个几何域和两个物理状态的实验数据集，包括合成基准和工业汽车悬架链接的设计域。

**📈 对比分析**

与多启动、扰动和图像引导基线方法进行比较，38个提示-域组合在245次主要SDS运行中实现了统计显著的合规性降低，表现优于所有替代基线，合规性降低幅度最高可达-31.5%。

**⚠️ 局限性**

限制在于该框架目前仅适用于二维设计，且提示选择仍需手动，未来需要扩展到三维并自动化提示发现。

---

## 26. Understanding In-Context Multimodal Jailbreaks via Posterior Reweighting

**arXiv ID:** 2609.10613 | [PDF](https://arxiv.org/pdf/2609.10613v1)

**作者:** Xu Zhang `[一作]`, Ren Wang `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种后验重加权框架，解释了多模态大语言模型（MLLMs）中的上下文学习（ICL）越狱现象，揭示了模型在推理时如何在安全和有害行为之间动态调整偏好。

**💡 创新点**

创新点在于将越狱过程形式化为证据积累的过程，并提出了可预测的缩放法则，描述了演示数量、有害比例、对抗强度和语义多样性如何影响越狱的成功率。

**🔧 技术方法**

使用了后验重加权框架和自适应的推理时防御机制，通过注入良性反证据来抑制有害后验漂移，同时保持模型的实用性。

**📊 数据集**

使用了多种多模态大语言模型（MLLMs）进行实验验证，具体数据集未详细列出，但涉及多个模型和攻击配置。

**📈 对比分析**

与现有的上下文防御方法相比，提出的方法在固定干预预算下显著改善了鲁棒性与实用性的权衡，具体性能提升通过实验结果验证。

**⚠️ 局限性**

局限性在于高度多样化的有害上下文和更强的自适应攻击仍然对推理时的安全控制构成挑战。

---

## 27. Strategic Information Transmission over Gossip Networks

**arXiv ID:** 2609.10576 | [PDF](https://arxiv.org/pdf/2609.10576v1)

**作者:** Emirhan Tekez `[一作]` (Bilkent University), Sinan Gezici `[通讯]` (Bilkent University)

**通讯引用:** 7740 | [OpenAlex ID](https://openalex.org/A5060023238)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在时间敏感的八卦网络中进行战略信息传输的模型，考虑了一个观察二元连续时间马尔可夫源的发送者在通信预算下向接收者传输更新的情况。

**💡 创新点**

创新点在于将八卦信息传播与战略信息传输结合，提出了一个基于Stackelberg博弈的模型，分析了发送者的预算约束如何影响信息传输策略。

**🔧 技术方法**

使用了随机混合系统（SHS）框架来分析模型，并通过Monte Carlo模拟验证了理论结果。

**📊 数据集**

使用了一个二元连续时间马尔可夫链（CTMC）作为数据源，接收者通过八卦网络进行信息交换。

**📈 对比分析**

与传统的监测模型相比，本文的模型考虑了发送者的战略行为，证明了在战略状态下，接收者的效用随着八卦速率的增加而严格增加，而发送者的效用则严格减少。

**⚠️ 局限性**

限制在于模型假设了完全连接的八卦网络，未来的工作可以扩展到更复杂的网络拓扑和考虑信息交换的成本。

---

## 28. Design and Operation of a Federated GPU Cluster for Digital Humanities within DHinfra.at

**arXiv ID:** 2609.10552 | [PDF](https://arxiv.org/pdf/2609.10552v1)

**作者:** Florian Atzenhofer-Baumgartner `[一作]` (University of Graz), Michael Otto `[通讯]` (University of Graz)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

描述了一个为数字人文学科研究而设计的小型联邦GPU集群的设计、实施和操作，该集群在奥地利的DHinfra.at项目中使用。

**💡 创新点**

创新点在于通过国家身份联合体进行登录管理，并提供交互式笔记本、SSH和OpenAI兼容的推理API等多种计算接口，同时实现了项目配额和模型目录的自助服务。

**🔧 技术方法**

使用了开源组件构建软件栈，包括Authentik（身份管理）、Slurm（资源调度）、Enroot和Pyxis（用户容器管理）、vLLM（模型服务）等。

**📊 数据集**

没有具体提到使用的数据集，但提到的应用场景包括手写文本识别、语言建模、图像和视频分类等。

**📈 对比分析**

与传统HPC中心相比，该平台通过提供更易于访问的接口和自助服务，降低了研究人员使用GPU计算的门槛，尽管在资源利用率上可能较低。

**⚠️ 局限性**

限制在于团队规模较小（约1.5个全职员工），在资源利用和优先访问之间需要权衡，同时在采购和法律程序上也面临一定的复杂性。

---

## 29. Collective Hysteresis and Multistability in Threshold Networks

**arXiv ID:** 2609.10580 | [PDF](https://arxiv.org/pdf/2609.10580v1)

**作者:** Moses Boudourides `[一作]` (Northwestern University), Moses Boudourides `[通讯]` (Northwestern University)

**通讯引用:** 301 | [OpenAlex ID](https://openalex.org/A5035035192)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本论文扩展了Kaye的模型，研究了在一个相互作用的代理网络中，异质激活阈值和共享反馈信号如何导致集体激活和滞后现象。

**💡 创新点**

创新点在于证明了在弱反馈下，网络的拓扑结构不会导致异质模式在同质模式之前不稳定，并且在足够弱的块间耦合下，稳定的标量平衡可以独立分配给块。

**🔧 技术方法**

使用了行随机矩阵来描述代理之间的相互作用，并通过数学证明和数值计算来分析网络的平衡、分岔和多稳定性。

**📊 数据集**

使用了伽马分布作为主要基准，设置了反馈强度和外部刺激的参数，并进行了数值计算以验证理论结果。

**📈 对比分析**

通过与Kaye的标量模型进行比较，发现网络拓扑不会改变平衡的折叠和尖点，且在弱耦合下，块的独立稳定状态可以持续存在。

**⚠️ 局限性**

限制在于假设了精确的公平划分，实际网络中可能存在的近似公平划分未被充分考虑，且该模型是确定性的，不是从二元阈值代理的随机网络中推导出来的。

---

## 30. PEARL: A Task-Aware Framework for Evaluating Differentially Private Synthetic Educational Data

**arXiv ID:** 2609.10612 | [PDF](https://arxiv.org/pdf/2609.10612v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 31. Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language

**arXiv ID:** 2609.10629 | [PDF](https://arxiv.org/pdf/2609.10629v1)

**作者:** Niloy Kumar Mondal `[一作]` (Bangladesh University of Engineering and Technology), Md Rizwan Parvez `[通讯]` (Qatar Computing Research Institute, HBKU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种端到端的多代理框架，能够从自然语言问题描述中自动生成QUBO（Quadratic Unconstrained Binary Optimization）公式。

**💡 创新点**

创新点在于通过多代理系统分解任务，利用迭代自我修复机制显著提高了QUBO公式生成的准确性。

**🔧 技术方法**

使用了大型语言模型（LLM）作为多个代理，包括规划代理、公式化代理、编码代理、调试代理、评判代理和写作代理。

**📊 数据集**

使用了QUBOBench数据集，该数据集包含100个组合优化问题，涵盖12个应用领域，问题来源于同行评审文献、竞赛和经典NP难题。

**📈 对比分析**

与直接单次调用的基线方法相比，框架在QUBOBench上达到了68%的准确率，超出基线22%。

**⚠️ 局限性**

局限性在于当前框架仍然依赖于用户提供的自然语言描述，且在处理复杂问题时可能需要进一步的领域专业知识。

---

## 32. From Token Interfaces to Token Semantics: A Formal Composition and Conformance Model for Implementation-Neutral Token Specifications

**arXiv ID:** 2609.10547 | [PDF](https://arxiv.org/pdf/2609.10547v1)

**作者:** John deVadoss `[一作]` `[通讯]` (InterWork Alliance), John deVadoss (InterWork Alliance)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了InterWork Alliance Token Taxonomy Framework (TTF)，作为一种实现中立的代币规范的类型化语义组合模型，旨在解决代币接口标准化不足以完全定义代币含义的问题。

**💡 创新点**

创新点在于TTF提供了一种分层的符合性模型，能够独立于平台绑定地指定、比较、验证和管理代币语义，从而增强代币在智能合约平台、许可账本和共享状态系统之间的互操作性。

**🔧 技术方法**

使用了形式化规范、语义互操作性、模型驱动工程等技术，结合了控制消息和符合性测试。

**📊 数据集**

通过文档代币、仓库收据和碳/数字MRV案例研究进行评估，展示了代币语义的可指定性和可验证性。

**📈 对比分析**

与传统的接口标准相比，TTF能够揭示仅通过接口标准无法暴露的语义差距，分析表明其在代币互操作性方面提供了更强的技术基础。

**⚠️ 局限性**

限制在于代币语义无法完全涵盖法律意义，且平台映射可能会丢失某些语义特征，此外，语义符合性并不能替代安全分析。

---

## 33. M3-Former: Multimodal Transformer with Mixture-of-Experts for Long-Term Vessel Trajectory Prediction

**arXiv ID:** 2609.10559 | [PDF](https://arxiv.org/pdf/2609.10559v1)

**作者:** Wenzhe Jin `[一作]` (University of Chinese Academy of Sciences), Haina Tang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5110364708)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 M3-Former 框架，利用大型语言模型提取船舶静态语义与轨迹动态融合，实现长时段船舶轨迹预测。

**💡 创新点**

创新点在于将 LLM 提取的语义先验与多模态信息统一编码，结合双粒度 Mixture-of-Experts 架构和 Steering-Weighted 损失，提升长时间段预测的准确性与鲁棒性。

**🔧 技术方法**

采用多模态融合（LLM+Transformer+自注意力）、双粒度 MoE、离散化空间分类、Steering-Weighted 交叉熵等技术。

**📊 数据集**

使用丹麦 AIS 数据集（2023 年 1 月至 3 月）共约 1.9M 条有效样本，包含船舶静态属性与动态轨迹。

**📈 对比分析**

与 CV、KF、RNN、TCN、TrAISformer 等基线在 ADE/FDE 上进行比较，M3-Former 在 1–4 小时预测均优于对手，4 小时 ADE/FDE 分别提升 4.4%/5.1%。

**⚠️ 局限性**

局限在于仅考虑船舶静态语义，未纳入天气、潮汐、海况等环境因素，且模型规模较大，推理速度相对较慢。

---

## 34. AI Safety: Not Optional, Not Later

**arXiv ID:** 2609.10630 | [PDF](https://arxiv.org/pdf/2609.10630v1)

**作者:** Qinghua Lu `[一作]` (CSIRO), Yoshua Bengio `[通讯]` (LawZero)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种安全设计保证架构，结合了模型级监督和系统级控制，以应对AI安全失败的多层次问题。

**💡 创新点**

创新点在于将模型级和系统级的安全控制结合起来，形成一个多层次的安全保障体系，强调了治理和证据互操作性的重要性。

**🔧 技术方法**

使用了模型级监督（如Scientist AI）和系统级控制，包括独立验证、监控和证据基础设施。

**📊 数据集**

论文中提到的案例涉及多个AI代理在网络安全评估中的表现，具体数据集未明确列出。

**📈 对比分析**

通过分析多个真实事件，展示了AI代理在不同情况下的失败模式，强调了多层次控制的重要性，性能评估显示现有系统存在显著的安全漏洞。

**⚠️ 局限性**

限制在于当前的安全设计架构可能无法覆盖所有新兴的AI安全风险，且需要更广泛的治理框架来确保证据的可比性和可重用性。

---

## 35. Governed Human-AI Prioritization Under Uncertainty: Adaptive Estimation and Dependency-Constrained Portfolio Selection

**arXiv ID:** 2609.10648 | [PDF](https://arxiv.org/pdf/2609.10648v1)

**作者:** Azzeddine Ihsine `[一作]` (Inovionix), Sara Ihsine `[通讯]` (Inovionix)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究探讨了在AI原生软件工程中，如何在优先级决策中结合人类判断、历史类比、参数估计和AI生成的预测。研究了五种定量操作符在D-POAF决策实践中的应用。

**💡 创新点**

创新点在于提出了一个量化的决策层次，强调了战略评分参数对优先级成员的影响，并展示了可靠性加权聚合在异质估计器质量下显著降低了努力估计误差。

**🔧 技术方法**

使用了控制合成实验来表征五种定量操作符的行为，包括商业价值评分（BVS）、努力和风险评分（ERS）、优先级价值评分（PVS）、集体校准评分（CCS）和最优开发路径（ODP）。

**📊 数据集**

使用了控制合成数据集，包含5000个块用于排名敏感性，12500个任务用于努力估计校准和测试，800个依赖约束的投资组合实例。

**📈 对比分析**

通过与固定角色加权的比较，可靠性加权的努力聚合在异质误差下表现更好，MAE为0.616，相较于最佳单一估计器（MAE 1.073）减少了42.6%。在800个投资组合实例中，价值与努力的比率达到了0.962，显示出小但系统的优势。

**⚠️ 局限性**

限制在于该研究的结果是基于控制实验的机制级别声明，外部有效性需要通过工业部署来验证，且不同的目标定义不同的优化问题。

---

## 36. HermiCache: Enclave-Aware Cache Replacement for Trusted Execution Environments

**arXiv ID:** 2609.10634 | [PDF](https://arxiv.org/pdf/2609.10634v1)

**作者:** Oussama Elmnaouri `[一作]` (ENSTA), Loïc Lagadec `[通讯]` (ENSTA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种名为HermiCache的缓存替换机制，旨在保护受信执行环境（TEE）免受基于缓存的侧信道攻击。

**💡 创新点**

HermiCache的创新点在于其所有权基础的替换机制，能够提供细粒度的配置和确定性的保护。

**🔧 技术方法**

使用了RISC-V架构的硬件实现，并在OpenHwGroup的CVA6核心中集成了Keystone TEE。

**📊 数据集**

在Digilent Genesys2开发板上进行了评估，该板基于CVA6的SoC，包含16KB的4路L1指令缓存和32KB的8路L1数据缓存。

**📈 对比分析**

与现有的Composable Cachelets（CC）机制进行了比较，HermiCache在指令每周期（IPC）率方面表现出竞争力，且在同组竞争实验中，HermiCache成功防止了攻击者驱逐受保护的缓存行。

**⚠️ 局限性**

HermiCache的局限性在于它不保护所有微架构侧信道（如TLB、分支预测器），并且未考虑物理攻击。

---

## 37. On the Relation between Code Quality and Machine Learning Performance: A Large-scale Empirical Study

**arXiv ID:** 2609.10610 | [PDF](https://arxiv.org/pdf/2609.10610v1)

**作者:** Marius Mignard `[一作]` (Univ. Lille), Anne Etien `[通讯]` (Univ. Lille)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对265,363个Kaggle提交的Python笔记本进行大规模实证研究，探讨了代码质量与机器学习性能之间的关系，并评估了流行度和作者专业知识是否能指示代码质量或性能。

**💡 创新点**

研究发现，通用Python代码质量与机器学习性能无关，而遵循特定于机器学习的最佳实践与更好的性能相关。流行度和作者专业知识并不能可靠地指示代码质量或性能。

**🔧 技术方法**

使用了静态分析工具Pylint和SonarQube来评估代码质量，Pylint用于捕捉通用Python代码质量，SonarQube则专注于数据科学和机器学习特定的实践。

**📊 数据集**

使用的数据集为265,363个提交到Kaggle竞赛的Python笔记本，这些笔记本的性能通过竞赛得分进行衡量。

**📈 对比分析**

研究通过Spearman相关性分析和Wilcoxon-Mann-Whitney U检验比较了不同方法，结果显示通用Python质量与性能之间的相关性微乎其微，而机器学习特定的违规行为与性能之间存在小的负相关关系。

**⚠️ 局限性**

研究的局限性包括未考虑笔记本的创作日期，可能导致使用过时的实践；使用的静态分析工具可能产生误报；以及对用户专业知识的分类可能导致的偏差。

---

## 38. Certified Panic Mode: Repair-Invariant Error Recovery for Maximal-Munch Lexing

**arXiv ID:** 2609.10600 | [PDF](https://arxiv.org/pdf/2609.10600v1)

**作者:** Nicklas Nidhögg `[一作]` `[通讯]` (Independent Researcher), Nicklas Nidhögg (Independent Researcher)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的恢复机制，通过定理选择恢复位置，确保在每个修复的扫描通过证据时都能放置一个令牌边界。

**💡 创新点**

创新点在于定义了修复不变的重新同步点，提供了一个量化的恢复点的健全性标准，并展示了如何在损坏的输入中使用静态证书进行恢复。

**🔧 技术方法**

使用了定理证明和证书机制，结合了前缀修复的量化和扫描过程中的证据返回。

**📊 数据集**

使用了多种数据集，包括生成的JSON数据和真实的Twitter API输出，进行恢复质量的评估。

**📈 对比分析**

与传统的跳过一个字符和分隔符约定进行比较，性能上，本文的方法在恢复质量上显著优于这些传统方法，尤其是在证书密集的情况下。

**⚠️ 局限性**

限制在于所使用的数据集是生成的，且损坏模型是合成的，未研究真实编辑轨迹的影响。

---

## 39. When Passing Tests Hides Vulnerabilities: An Empirical Study of Silent Failures in Agentic Systems

**arXiv ID:** 2609.10548 | [PDF](https://arxiv.org/pdf/2609.10548v1)

**作者:** Wenji Bai `[一作]` (Tampere University), Pekka Abrahamsson `[通讯]` (Tampere University)

**通讯引用:** 10644 | [OpenAlex ID](https://openalex.org/A5058417486)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了LLM驱动的代理式代码修复中，虽通过编译与单元测试但仍保留或产生安全缺陷的无声失效。

**💡 创新点**

创新点在于构建了三维无声失效分类法（缺失、缺陷、引入），揭示其在多代理架构中的传播路径与失败模式。

**🔧 技术方法**

采用四层验证框架（语法、功能、静态安全、可利用性）结合人工审核，并以GPT‑4o‑mini在多代理系统中执行。

**📊 数据集**

使用了两大安全聚焦数据集SecurityEval与CVEfixes，共计1,236个Python任务。

**📈 对比分析**

与现有评测方法对比，实验显示仅靠单元测试无法发现约58%的失效，提出针对性静态分析与可利用性评分，显著提升检测率。

**⚠️ 局限性**

局限在于仅覆盖Python单文件任务、仅使用一款LLM模型且对静态工具的误报/漏报敏感，未来需扩展语言与模型多样性。

---

## 40. Data-Efficient Language Modeling: From Frontier Advancement to Principle-Guided Model Improvement

**arXiv ID:** 2609.10702 | [PDF](https://arxiv.org/pdf/2609.10702v1)

**作者:** Shuxing Yang `[一作]` (Qiushi Engine Team), Yihao Yang `[通讯]` (Qiushi Engine Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过BabyLM 2026 Strict-Small项目，探讨在有限文本数据下如何有效学习，提出了数据高效学习的原则，并通过三个阶段的研究改进模型。

**💡 创新点**

创新点在于提出了一种数据高效学习原则，强调在预测时组织经验、设计可见信息和监督目标，并测试学习能力的保留和泛化。

**🔧 技术方法**

使用了DeBERTa-v2风格的掩码语言模型架构，结合了紧凑重述、预算再投资和残差增量学习等技术。

**📊 数据集**

使用了限制在1000万词的语料库和1亿次累计词展示的Strict-Small数据集。

**📈 对比分析**

与普通的继续训练方法相比，采用新方法的模型在九个评估指标上表现更好，整体得分从42.02提高到42.25，第二代模型在2026年9月8日的公开快照中获得最高分。

**⚠️ 局限性**

限制在于模型在新输入上的学习能力可能不如在熟悉输入上的表现，且在进一步训练中可能会丧失已学会的能力。

---

## 41. Understanding LoRA Rank Trade-offs in Diffusion Model Fine-Tuning

**arXiv ID:** 2609.10656 | [PDF](https://arxiv.org/pdf/2609.10656v1)

**作者:** Iman Khazrak `[一作]` (Bowling Green State University), Robert C. Green `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究通过对LoRA（低秩适应）在扩散模型微调中的排名权衡进行控制实验，探讨了在固定训练预算下，LoRA排名对适应质量、参数效率、内存和运行时间的影响。

**💡 创新点**

创新点在于提供了一个系统的、可重复的实验框架，量化了不同LoRA排名下的效率与质量权衡，并验证了中等排名在固定预算下的最佳表现。

**🔧 技术方法**

使用了DDPM（去噪扩散概率模型）U-Net作为主干网络，结合LoRA技术进行微调。

**📊 数据集**

使用CIFAR-10数据集进行实验，分辨率为32×32。

**📈 对比分析**

通过与不同排名（2, 4, 8, 16, 32）的DDPM进行比较，发现排名4的FID（Fréchet Inception Distance）最佳（124.1380），而排名8接近（124.2136），更高的排名在相同预算下收益有限，表明中等排名在效率和质量之间提供了最佳平衡。

**⚠️ 局限性**

限制在于实验仅使用CIFAR-10数据集，且分辨率较低，结果的普适性可能受到限制；此外，所评估的模型较小，可能与实际生产中的大规模模型表现不同。

---

## 42. A Deadline-Driven Algorithm for Polyamorous Scheduling

**arXiv ID:** 2609.10641 | [PDF](https://arxiv.org/pdf/2609.10641v1)

**作者:** Arjun Maneesh Agarwal `[一作]` `[通讯]` (Chennai Mathematical Institute), Arjun Maneesh Agarwal (Chennai Mathematical Institute)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了多元调度问题，旨在找到一个周期性匹配的调度，以最小化同一边缘之间的最大加权等待时间。

**💡 创新点**

提出了一种4 G^*算法，改进了之前已知的3 + √(5)≈ 5.236的界限。

**🔧 技术方法**

使用了基于截止日期的贪心算法，该算法在竹园修剪问题中是最优的。

**📊 数据集**

使用了边加权图作为数据集，图中的每个边代表一个具有正增长率的关系。

**📈 对比分析**

与Biktairov等人（2024）的Reduce-Fastest算法进行比较，后者的近似比为3 + √(5)。新算法的性能优于该算法，能够实现严格小于4G^*的热量。

**⚠️ 局限性**

该算法的局限性在于其复杂性，尤其是在一般图的情况下，调度必须是一个匹配而不是单一选择。

---

## 43. From Cycle Space to Cycle Manifold: Limits and Achievability of Blind False Data Injection Attacks

**arXiv ID:** 2609.10631 | [PDF](https://arxiv.org/pdf/2609.10631v1)

**作者:** Xin Li `[一作]` (Ben Gurion University), Rami Puzis `[通讯]` (Ben Gurion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了盲目虚假数据注入攻击（FDIA）的极限和可达性，提出了在直流（DC）和交流（AC）电力系统中，如何利用网络的循环结构来理解FDIA的隐蔽性和攻击空间。

**💡 创新点**

创新点在于证明了加权循环空间是DC模型中完整攻击空间的正交补，并提出了循环流形的理论，作为AC模型中完整攻击空间的非线性约束。

**🔧 技术方法**

使用了循环空间和循环流形的理论，结合测量数据进行攻击空间的重建和参数估计，采用了图形处理单元（GPU）进行计算。

**📊 数据集**

使用了IEEE 14、30、57和118总线测试系统的数据集进行实验，比较了不同方法的性能。

**📈 对比分析**

通过与现有的盲FDIA方法进行比较，提出的方法在攻击空间重建和状态影响方面表现优越，尤其是在高噪声条件下仍能保持较高的通过率。

**⚠️ 局限性**

限制在于AC模型的循环流形是非线性的，无法直接提取循环基，因此在没有拓扑信息的情况下，学习非线性基本循环结构仍然是一个开放问题。

---

## 44. SoK: Privacy Attacks on Machine Learning via Explainable AI

**arXiv ID:** 2609.10627 | [PDF](https://arxiv.org/pdf/2609.10627v1)

**作者:** Abdullah Caglar Oksuz `[一作]` (Case Western Reserve University), Erman Ayday `[通讯]` (Case Western Reserve University)

**通讯引用:** 2722 | [OpenAlex ID](https://openalex.org/A5028326739)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统化了25项研究，探讨了如何利用机器学习解释来进行模型提取、成员推断和模型反演等攻击，分析了不同攻击路径和信号的影响。

**💡 创新点**

创新点在于将模型知识与解释获取分开，识别出五种攻击路径，并提出了一种解释获取分类法，强调了不同攻击模型和评估指标之间的可比性。

**🔧 技术方法**

使用了机器学习解释技术，包括梯度、特征重要性、反事实解释和局部代理等方法，分析了它们在不同攻击中的作用。

**📊 数据集**

研究中涉及的核心数据集包括MNIST、CIFAR-10、Adult等，涵盖了多种模型和任务。

**📈 对比分析**

通过比较系统和威胁模型、解释信号、辅助知识、目标模型、查询预算、评估指标等，发现没有一种解释方法是绝对不安全的，风险取决于暴露的信号、获取方式和攻击者的知识。

**⚠️ 局限性**

限制在于现有研究未能统一评估不同攻击的可比性，且对解释的隐私风险评估仍需进一步标准化。

---

## 45. EFX Allocations for Three Agents and Seven or Eight Chores

**arXiv ID:** 2609.10585 | [PDF](https://arxiv.org/pdf/2609.10585v1)

**作者:** Xinkai Zhang `[一作]` `[通讯]` (Renmin University of China), Xinkai Zhang (Renmin University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

证明了每个非负加性工作实例在三名代理和七或八个不可分割的工作中都存在一个EFX分配，确保每个拥有的工作在去除后，代理仍然偏好剩余的分配。

**💡 创新点**

首次解决了三名代理和七个或八个工作情况下的EFX存在性问题，填补了m≤2n定理的空白。

**🔧 技术方法**

使用了计算机辅助证明技术，特别是通过Z3和cvc5求解器进行的量化自由线性实数算式（QF_LRA）验证。

**📊 数据集**

使用了三名代理和七或八个不可分割工作的非负加性成本数据集。

**📈 对比分析**

通过手动检查和计算机辅助证明的方法进行比较，结果显示在三名代理和七或八个工作情况下均能实现EFX分配，且计算机证明的结果为不满足的公式。

**⚠️ 局限性**

限制在于对于每个n≥4的情况，加性工作可能无法实现EFX分配，且m=9的情况仍然是开放的研究问题。

---

## 46. Optimizing AI Inference Across the Deployment Stack

**arXiv ID:** 2609.10550 | [PDF](https://arxiv.org/pdf/2609.10550v1)

**作者:** Tejinder Singh `[一作]` (Dell Technologies), Bhavesh A. Patel `[通讯]` (Dell Technologies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提供了一个统一的推理优化分析框架，涵盖了模型压缩、编译器转换和服务系统策略之间的相互作用，强调了在实际部署中这些因素的重要性。

**💡 创新点**

创新点在于提出了一个三层分类法，将部署分解为模型级技术、编译器级转换和系统级策略，并将部署形式化为一个受限的多目标优化问题。

**🔧 技术方法**

使用了屋顶线模型和排队模型来分析性能，并提出了一种严谨的证据协议来提高文献中结果的可比性。

**📊 数据集**

使用了多个数据集，包括Llama-3.1模型系列和不同的边缘平台（如Jetson AGX Orin）及数据中心GPU（如A100、H100）进行大规模量化评估。

**📈 对比分析**

通过与现有文献的比较，展示了不同层次之间的交互如何影响最终的部署结果，强调了单层分析无法预测的复杂性。性能比较显示，跨层交互的影响显著，且没有单一层的优化可以单独评估。

**⚠️ 局限性**

限制在于现有文献中缺乏对不同硬件平台的全面比较，且大多数研究集中在特定的模型和框架上，缺乏对LLM服务的系统性评估。

---

## 47. Compass: Dissecting Communication and Computation Operators for Efficient LLM Training

**arXiv ID:** 2609.10549 | [PDF](https://arxiv.org/pdf/2609.10549v1)

**作者:** Guangyu Xiang `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 11239 | [OpenAlex ID](https://openalex.org/A5100730785)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Compass系统，通过系统优化和全面建模来实现大语言模型训练中的计算与通信重叠的最优配置。

**💡 创新点**

创新点在于设计了一种新的TA-IntraFusion算法，利用环形通信算法提高混合拓扑下的带宽利用率，并引入了高精度的性能模型来确定最优分解度，消除了昂贵的经验调优。

**🔧 技术方法**

使用了TA-IntraFusion算法、最优分解预测器（ODP-InterDecom）和统一性能框架来动态选择最佳策略。

**📊 数据集**

在一个包含8个NVIDIA A6000 GPU的服务器上进行了广泛的评估，涵盖288种不同的配置和真实应用的端到端实验。

**📈 对比分析**

与Megatron-LM基线相比，Compass在真实应用中实现了高达1.42倍的端到端加速，显示出其在不同工作负载和硬件配置下的优越性能。

**⚠️ 局限性**

限制在于当前方法依赖于特定的硬件拓扑，可能在其他类型的系统上表现不佳，且在极端情况下可能无法动态选择最佳策略。

---

## 48. The Privacy Subsidy in Market Microstructure

**arXiv ID:** 2609.10543 | [PDF](https://arxiv.org/pdf/2609.10543v1)

**作者:** Yuki Nakamura `[一作]` `[通讯]` (Open University of Japan), Yuki Nakamura (Open University of Japan)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

论文探讨了隐私保护的交易机制如何在粗化的订单流视图上设计价格，展示了市场制造者在信息有效定价时必须让步于交易者的福利转移，即隐私补贴。

**💡 创新点**

创新点在于提出了一种普遍的粗信号不可能性定理，表明在粗信号上进行信息有效定价的市场制造者无法同时实现零利润和效率，并且通过三种经典微观结构模型具体化了隐私补贴的特征。

**🔧 技术方法**

使用了贝叶斯定价、微观结构模型（如单周期Kyle模型、Glosten-Milgrom模型和连续时间Kyle-Back模型）等技术。

**📊 数据集**

论文中使用了三种经典微观结构模型的数据集，包括单周期Kyle模型、Glosten-Milgrom模型和连续时间Kyle-Back模型。

**📈 对比分析**

通过比较不同模型的隐私补贴，发现隐私补贴是一个纯转移，且在费用抵消后，隐私的福利成本是四阶的，而补贴本身是二阶的，因此隐私在主要阶数上是福利中性的。

**⚠️ 局限性**

限制在于论文的结果是基于部分均衡的前导阶数分析，未考虑费用、交易量和知情交易强度的联合固定，未来的工作需要进一步探讨这些因素的影响。

---

## 49. The Internet of Collaborating Things: Agentic Edge AI for Autonomous Cross-Domain Collaboration

**arXiv ID:** 2609.10542 | [PDF](https://arxiv.org/pdf/2609.10542v1)

**作者:** Walid A. Hanafy `[一作]` (University of Massachusetts Amherst), Prashant Shenoy `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 18970 | [OpenAlex ID](https://openalex.org/A5032939724)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了物联网协作事物（IoCT）这一新范式，强调设备之间的直接协作与动态集群形成，以应对未来物联网设备数量激增带来的挑战。

**💡 创新点**

创新点在于引入了代理边缘人工智能（agentic edge AI），实现设备在跨域协作中的自主控制和动态适应，解决了传统云计算和边缘计算模型的局限性。

**🔧 技术方法**

采用了代理控制平面和可移植的执行子系统，结合了边缘计算、人工智能和动态资源管理技术。

**📊 数据集**

未具体提及使用的数据集，但讨论了多种物联网设备和场景，如送餐机器人、频繁访客和协作游戏等。

**📈 对比分析**

与现有的云和边缘平台相比，IoCT提供了更高效的设备间协作，减少了性能开销，支持动态资源分配和跨域安全控制，性能表现优于传统的垂直交互模型。

**⚠️ 局限性**

局限性包括对代理决策的安全性和可验证性、跨域信任协商的复杂性、资源管理的动态性以及激励机制的缺乏等，仍需进一步研究和解决。

---

## 50. BodyCam-VQA: Enhanced Body-Worn Camera Video Captioning via Multimodal Reasoning and Probe Question Generation

**arXiv ID:** 2609.10815 | [PDF](https://arxiv.org/pdf/2609.10815v1)

**作者:** Karish Gupta `[一作]` (Worcester Polytechnic Institute), Xiaozhong Liu `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 3994 | [OpenAlex ID](https://openalex.org/A5101985030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种适用于执法高风险场景的自适应视觉问答（VQA）框架，旨在从警用随身摄像头（BWC）视频中提取细粒度的视觉证据。

**💡 创新点**

创新点在于通过结构化推理方法，克服了传统字幕系统无法捕捉的关键法医细节，提高了执法事件的记录可靠性和客观性。

**🔧 技术方法**

使用了视觉语言模型（VLM）和多种问题生成模型，包括基础模型和微调的开放权重模型。

**📊 数据集**

使用了来自芝加哥民事警察问责办公室（COPA）数据集的359个标注的60秒视频片段，支持多模态推理和分析。

**📈 对比分析**

与直接VLM摘要和基于管道的VQA方法进行比较，结果显示VQA驱动的架构在事实准确性、完整性和视觉丰富性方面均有显著提升，尤其是微调的Qwen模型表现优异。

**⚠️ 局限性**

局限性包括：1) 当前管道仅针对60秒视频片段，未能评估长时间段的推理能力；2) 奖励信号的间接对齐可能导致模型优化偏离人类法医评估；3) 尚未在实际执法工作流程中进行评估。

---

## 51. Larger Context Window, Fewer Overcorrections: Optimizing Prompts and Batching for Minimal-Edit Grammatical Error Correction

**arXiv ID:** 2609.10810 | [PDF](https://arxiv.org/pdf/2609.10810v1)

**作者:** Kateryna Karpo `[一作]` (Ukrainian Catholic University), Artem Chernodub `[通讯]` (Zendesk)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于提示的最小编辑语法错误修正（GEC）方法，旨在缩小与微调模型之间的差距。

**💡 创新点**

通过引入基于分类法的指令、批处理输入和LLM辅助的提示优化，显著提高了GEC的性能，设立了新的基于提示的SOTA。

**🔧 技术方法**

使用了大型语言模型（LLMs）和提示优化技术，特别是Gemini 3.1-Pro。

**📊 数据集**

使用了BEA-2019和CoNLL-2014数据集进行评估。

**📈 对比分析**

与现有的微调模型相比，提出的方法在BEA-2019测试集上达到了F_0.5=78.32，仅比最佳微调单模型低0.38分，显示出显著的性能提升。

**⚠️ 局限性**

研究的局限性包括对商业API的依赖、仅在中等容量模型上进行的提示优化、评估仅限于英语学习者的作文，以及未考虑流畅性或风格改进的评估。

---

## 52. Designing Technology for Social Wellbeing in Built Environments: A Conceptual Framework

**arXiv ID:** 2609.10779 | [PDF](https://arxiv.org/pdf/2609.10779v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 53. Composable CXL Memory as a Kubernetes-Native Shared Memory for LLM Serving

**arXiv ID:** 2609.10790 | [PDF](https://arxiv.org/pdf/2609.10790v1)

**作者:** Hongjian Fan `[一作]` (Seagate Technology), Sean Dykstra `[通讯]` (Seagate Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Kubernetes动态资源分配（DRA）驱动程序，使可组合的CXL内存成为可调度的集群资源，并评估了用于跨节点KV缓存重用的共享内存层。

**💡 创新点**

创新点在于将可组合的CXL内存作为Kubernetes的可调度资源，并在共享内存中嵌入元数据目录，消除了对外部元数据服务的依赖。

**🔧 技术方法**

使用了Kubernetes DRA驱动程序、CXL内存、DAX设备和共享内存连接器等技术。

**📊 数据集**

使用了一个512 GiB的CXL设备在一个两节点集群上进行实验，模型为Qwen2.5-7B-Instruct。

**📈 对比分析**

与节点本地层（GPU前缀缓存、CPU-DRAM卸载）相比，跨节点前缀重用在TTFT上减少了5.5×到36.6×，共享区域的延迟比同节点重用的延迟高出1-4%。

**⚠️ 局限性**

限制包括未实现预填充/解码分离、未进行多租户调度实验、单一GPU限制、缺乏调度器侧容量会计等。

---

## 54. Shedding Light: A Benchmark for Evaluating Lighting Understanding in Generative Image Models

**arXiv ID:** 2609.10787 | [PDF](https://arxiv.org/pdf/2609.10787v1)

**作者:** Justine Giroux `[一作]` (Université Laval), Jean-François Lalonde `[通讯]` (Université Laval)

**通讯引用:** 5270 | [OpenAlex ID](https://openalex.org/A5034761030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基准，用于评估生成图像模型对照明的理解和协调能力，特别是通过在真实照片中插入新物体来测试模型的照明一致性。

**💡 创新点**

创新点在于通过使用简单的物体（如漫反射球体）作为“光探针”，来定量评估生成模型的照明准确性，从而建立了一个系统化的评估协议。

**🔧 技术方法**

使用了逆渲染技术来估计照明方向、颜色和辐射分布，并通过多种生成模型进行比较。

**📊 数据集**

使用了多光照数据集，该数据集包含1015个室内场景，每个场景在25个不同的光照方向下捕获。

**📈 对比分析**

通过与16种最先进的生成模型进行比较，发现尽管模型的复杂性和训练数据量增加，但照明准确性并没有显著提高。大多数模型在前方照明方向的表现较好，而在其他方向的误差较大。

**⚠️ 局限性**

该基准假设场景中只有一个主光源，可能忽略了更复杂照明条件下的能量贡献。此外，数据集仅限于室内场景，无法评估模型在户外场景或多光源情况下的能力。

---

## 55. From Connectivity to Rewards: Dense Reward Learning with Directed State Graphs

**arXiv ID:** 2609.10781 | [PDF](https://arxiv.org/pdf/2609.10781v1)

**作者:** Shuyuan Zhang `[一作]` (McGill University), Doina Precup `[通讯]` (Google DeepMind)

**通讯引用:** 23644 | [OpenAlex ID](https://openalex.org/A5065836447)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种图引导的目标条件分层强化学习框架（G2QDR），通过在线构建有向状态图，学习状态连接模型以估计状态之间的可达性，并利用连接性辅助奖励来提高学习效率。

**💡 创新点**

创新点在于引入了状态连接模型，能够在不依赖于专家数据的情况下，在线构建有向状态图，并通过连接性强度生成稠密奖励，从而改善了在稀疏奖励环境中的学习效率。

**🔧 技术方法**

使用了神经网络来实现状态连接模型，并在图的构建过程中进行在线学习。

**📊 数据集**

在MuJoCo环境中进行了实验，评估了不同的稀疏奖励环境，包括AntMaze、AntGather、AntPush、AntFall和Pusher等。

**📈 对比分析**

与现有的GCHRL方法（如HIRO、HRAC、HESS和HLPS）进行了比较，结果表明G2QDR在多个任务中普遍提高了成功率，且性能波动较小，尤其在具有较高不对称性的任务中表现更为显著。

**⚠️ 局限性**

局限性在于在某些对称环境中，惩罚项可能会限制探索，导致性能略有下降。此外，图的构建和节点比较过程可能是计算瓶颈，影响整体效率。

---

## 56. Counterfactual Marginalisation: Framework for Evaluating Robustness to Nuisance Variables

**arXiv ID:** 2609.10778 | [PDF](https://arxiv.org/pdf/2609.10778v1)

**作者:** Yasin Ibrahim `[一作]` (University of Oxford), Konstantinos Kamnitsas `[通讯]` (University of Oxford)

**通讯引用:** 13805 | [OpenAlex ID](https://openalex.org/A5001104721)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种反事实边际化的方法，用于评估医学图像分类模型在测试时对人口统计学变量的鲁棒性。

**💡 创新点**

创新点在于引入了干预感知的评估指标，包括反事实边际风险、干预期望风险、反事实稳定性和最坏情况反事实风险，从而扩展了反事实的应用。

**🔧 技术方法**

使用了结构性因果模型（SCM）生成器来生成反事实图像，并在此基础上进行评估。

**📊 数据集**

使用了两个大型胸部X光数据集，CheXpert和MIMIC-CXR，进行肺积液的二分类任务。

**📈 对比分析**

与传统的评估方法相比，反事实评估能够更好地揭示预测模型中的偏差，并且在不需要疾病标签的情况下测量对干扰变量的敏感性。

**⚠️ 局限性**

限制在于评估指标依赖于反事实生成器的真实性和因果有效性，因此应与生成器验证和传统外部评估结合使用，而不是替代它们。

---

## 57. Where Should Society Draw the Line? A Social Choice Approach to Collective Consent

**arXiv ID:** 2609.10759 | [PDF](https://arxiv.org/pdf/2609.10759v1)

**作者:** Chris Dong `[一作]` (Hasso Plattner Institute), Niclas Boehmer `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文开展了对集体同意的公理研究，分析了在个体对选项的态度下，社会应当给予哪些选项同意。研究围绕三个原则展开：充分支持、少数保护和优越选项的主导性。

**💡 创新点**

创新点在于提出了相应的解决概念，透明地实现这些原则，并在数学上具有规范性。特别是，针对少数保护，提出了适应同意的比例否决核心的概念。

**🔧 技术方法**

使用了博弈论的特征化方法，发展了一系列相关概念，特别是提出了审批加权否决核心（Approval-Weighted Veto Core），它在比例少数保护和多数支持之间平滑插值。

**📊 数据集**

实验使用了五个数据集，涵盖高风险决策（如政治选举、伦理AI评估和道德决策），显示理论上违反原则的解决概念在实证上也同样违反这些原则。

**📈 对比分析**

通过与其他方法的比较，发现不满足某一原则的解决概念在实证上也会违反该原则，表明这些原则不仅是抽象的规范性概念，而是具有实际相关性和可触及性。

**⚠️ 局限性**

限制在于假设选民行为真实，未来研究可以探讨这些概念的激励兼容性属性，以及如何计算或近似无限候选集的整个比例否决核心。

---

## 58. SynCo: Synthetic Community-Aware Attributed Graph Generator for Graph Neural Network Benchmarking

**arXiv ID:** 2609.10742 | [PDF](https://arxiv.org/pdf/2609.10742v1)

**作者:** Guilherme Henrique Messias `[一作]` (Federal University of São Carlos), Alan Demétrius Baria Valejo `[通讯]` (Federal University of São Carlos)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的合成社区感知属性图生成器SynCo，用于图神经网络的基准测试。

**💡 创新点**

创新点在于允许用户控制节点的度分布和子社区结构，克服了现有生成器的灵活性不足和对幂律分布的过度依赖。

**🔧 技术方法**

使用了一种基于用户定义参数的图生成算法，包含节点社区和子社区分配、边构建、噪声生成和属性生成四个主要阶段。

**📊 数据集**

使用了合成图生成的实验数据集，支持生成高达210万节点的图。

**📈 对比分析**

通过与现有最先进的图生成模型进行比较，SynCo在合成图生成和数据增强方面表现优越，同时保持了原始数据集的分布特征。

**⚠️ 局限性**

限制在于生成的图可能仍然无法完全模拟真实世界网络的复杂性，尤其是在处理高度异构的社区结构时。

---

## 59. What Makes Creation Human? Authorship, Reasons, and Meaningful Human Control in Generative AI

**arXiv ID:** 2609.10738 | [PDF](https://arxiv.org/pdf/2609.10738v1)

**作者:** Yuxi Cao `[一作]` `[通讯]`, Yuxi Cao

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨了生成性人工智能（GenAI）如何影响创作过程中的人类创作能力和作者身份，提出了有意义的人类控制（MHC）和动态反思追踪（DRT）的概念，以确保人类在创作中的判断和理由能够持续影响作品的发展。

**💡 创新点**

创新点在于提出了动态反思追踪（DRT），强调创作者的理由在创作过程中能够形成、变化并有效影响作品的轨迹，而不仅仅是依赖于初始的创作意图。

**🔧 技术方法**

使用了生成性人工智能的概念，结合了人机交互（HCI）和伦理学的理论框架，提出了反思性自我审问工具，以帮助创作者评估他们的创作理由。

**📊 数据集**

论文没有具体提到使用的数据集，但讨论了生成性AI在创作领域的广泛应用，涉及音乐、图像和文本生成等多个领域。

**📈 对比分析**

与传统的创作方法相比，生成性AI的使用可能会降低创作者的认知参与感，尽管它提高了生产能力。性能方面，生成性AI能够快速生成高质量的创作，但可能导致创作者的判断能力和过程控制能力下降。

**⚠️ 局限性**

限制在于，尽管提出了动态反思追踪的框架，但如何在实际创作中有效实施这一框架仍然面临挑战，尤其是在创作者的理由和判断如何在与AI的互动中持续影响作品的发展方面。

---

## 60. Finishing the Task Is Not Enough: Evaluating Agent Resilience and Considerate Participation under Accumulating Challenge

**arXiv ID:** 2609.10724 | [PDF](https://arxiv.org/pdf/2609.10724v1)

**作者:** Yuanchen Bai `[一作]` (Cornell University), Angelique Taylor `[通讯]` (Cornell University)

**通讯引用:** 212 | [OpenAlex ID](https://openalex.org/A5074668213)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究探讨了在重复交互和不断变化的条件下，生成AI代理的操作弹性和体贴参与的评估，特别是在医疗工作流程中面临的挑战累积时。

**💡 创新点**

创新点在于提出了操作弹性和体贴参与作为评估生成AI代理的两个互补方面，并识别了五个与长期部署相关的困境。

**🔧 技术方法**

使用了两种生成AI模型，结合文本行动计划、内部评估和结构化的工作负载与情感报告等技术。

**📊 数据集**

研究基于120个模拟的医疗轨迹，这些轨迹涵盖了来自利益相关者的12个任务，并在轻、中、重挑战下进行评估。

**📈 对比分析**

通过比较文本行动计划和内部评估，发现随着挑战的增加，代理的操作弹性从自我恢复转向更依赖人类支持，而体贴参与则从任务聚焦的适应扩展到任务重构和更广泛的协调。

**⚠️ 局限性**

限制在于本研究仅评估了脚本化的语言级别行为，而未测试这些模式是否适用于多模态、物理体现的代理。

---

## 61. Processing and classifying bird songs using wavelet techniques and supervised learning

**arXiv ID:** 2609.10826 | [PDF](https://arxiv.org/pdf/2609.10826v1)

**作者:** Laura Lucia Dominguez Barrios `[一作]` (State University of Campinas), Mariana Rodrigues Motta `[通讯]` (State University of Campinas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出了一种集成框架，用于处理和分类自然声景中入侵鸟类的鸣叫，采用贝叶斯小波收缩方法来应对信号降解问题。

**💡 创新点**

创新点在于使用基于Epanechnikov核先验的贝叶斯小波收缩方法，提供了高效的计算性能和闭合形式的决策规则，显著提高了分类性能。

**🔧 技术方法**

采用了贝叶斯小波收缩技术和多种监督学习模型，包括随机森林、多项式逻辑回归和支持向量机（SVM）。

**📊 数据集**

使用了来自iNaturalist平台的三种鸟类（Euphonia violacea、Leiothrix lutea和Passer domesticus）的录音数据集，共230,000个音频文件。

**📈 对比分析**

通过混淆矩阵评估模型性能，SVM模型在10维MFCC配置下达到了最高准确率（0.9398），相比其他模型表现出色，尤其在处理复杂的非线性决策边界时。

**⚠️ 局限性**

限制在于该方法可能在处理极高噪声水平的情况下表现不佳，且未来研究需要探索该框架在超多样声景中的可扩展性。

---

## 62. Studying Without a Syllabus: Task-Agnostic Environment Preprocessing

**arXiv ID:** 2609.10824 | [PDF](https://arxiv.org/pdf/2609.10824v1)

**作者:** Vinay Samuel `[一作]` (Scale AI), Yuan Xue `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在没有下游任务分布知识的情况下，如何让一个智能体在新环境中进行任务无关的环境预处理，并生成可重用的资源。

**💡 创新点**

提出了一种开放式学习系统，能够根据环境内容动态选择处理方式，而不是依赖于固定的预处理策略。

**🔧 技术方法**

使用了元智能体（meta-agent）进行环境探索和资源生成，比较了开放式学习策略与固定学习策略的效果。

**📊 数据集**

在六个异构基准测试上进行评估，涉及不同大小的语料库和工具，具体基准包括BCP-G、OfficeQA、Harvey LAB、DABStep、APEX-Agents和AppWorld。

**📈 对比分析**

开放式学习策略在五个基准测试中表现优于固定策略，尤其在五个基准中获得了最高的Avg@3奖励，而固定语料处理在最大语料基准上表现最佳。较大的学习预算并不总是能可靠地提高下游奖励，但学习的工件减少了达到特定分数所需的测试时间采样。

**⚠️ 局限性**

研究的局限性包括未能评估不同模型之间的工件转移效果，且只使用了一个工作流档案，未能比较智能决策和工作流组合的价值。

---

## 63. RiVaT-Fuse: Reliability-Calibrated Variational Tensor Fusion for Multimodal Prediction under Modality Uncertainty

**arXiv ID:** 2609.10798 | [PDF](https://arxiv.org/pdf/2609.10798v1)

**作者:** Yingfan Xu `[一作]` (Oklahoma State University), Taiping Liu `[通讯]` (George Mason University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为RiVaT-Fuse的可靠性校准变分张量融合框架，用于在模态不确定性下进行多模态预测，定义融合为样本级潜在状态估计。

**💡 创新点**

创新点在于将多模态融合定义为样本级潜在状态估计，而不是简单的特征聚合，使用矩阵值信任几何体和结构化交互来增强模型的可靠性和稳定性。

**🔧 技术方法**

使用了变分优化、矩阵值信任操作符和结构化张量交互等技术，结合条件分布鲁棒学习和多任务耦合。

**📊 数据集**

在图像级图像-元数据预测基准上进行了评估，具体任务包括序数疾病严重程度分级和二元临床决策结果。

**📈 对比分析**

与直接表示级基线进行比较，RiVaT-Fuse在预测性能上表现最佳，尤其在序数准确性和二元任务的AUC上均优于其他方法，且在扰动下提高了概率和标签的稳定性。

**⚠️ 局限性**

局限性在于该研究主要集中在一个图像-元数据基准上，虽然提供了可靠性校准的融合方法，但在更广泛的临床应用和患者分离部署评估方面仍需进一步研究。

---

## 64. Two-Parameter Flow Map Learning for Continuous-Time Diffeomorphic Image Registration

**arXiv ID:** 2609.10789 | [PDF](https://arxiv.org/pdf/2609.10789v1)

**作者:** Mohammadjavad Matinkia `[一作]` (University of Alberta), Nilanjan Ray `[通讯]` (University of Alberta)

**通讯引用:** 3654 | [OpenAlex ID](https://openalex.org/A5082800075)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种新的框架TPFM-DIR，用于连续时间的微分同胚图像配准，直接学习非自主常微分方程的解。

**💡 创新点**

创新点在于通过强制执行时间变化流的基本结构属性（cocycle一致性），消除了训练过程中的时间离散化和速度积分的需求，从而提高了配准的准确性和拓扑保持。

**🔧 技术方法**

使用了基于时间的神经网络和cocycle正则化技术，直接建模非自主常微分方程的流映射。

**📊 数据集**

在九个数据集上进行了验证，包括2D和3D的MRI、CT和超声数据集。

**📈 对比分析**

与多种最先进的方法进行了比较，TPFM-DIR在多个数据集上表现出色，平均Dice分数提高了2.1%，在肺CT上减少了12%的目标注册误差（TRE），在心脏MRI和超声数据集上也有显著提升。

**⚠️ 局限性**

局限性在于虽然TPFM-DIR在多个数据集上表现良好，但在特定情况下可能仍需进一步优化以适应更复杂的变形情况。

---

## 65. Beyond Static Guarantees: Measuring the Static-Pass Dynamic-Fail Gap in Security-Sensitive and LLM-Generated Python Code

**arXiv ID:** 2609.10762 | [PDF](https://arxiv.org/pdf/2609.10762v1)

**作者:** Jessica Pourleyli `[一作]` (Toronto Metropolitan University), Glaucia Melo `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 70 | [OpenAlex ID](https://openalex.org/A5039289770)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种三阶段的安全评估管道，结合静态扫描、基于大语言模型的通用弱点枚举（CWE）推理和在隔离的Docker容器中进行的自主漏洞验证，以评估生成和安全敏感软件的安全性。

**💡 创新点**

创新点在于引入了静态通过动态失败（SPDF）现象，强调静态分析成功与运行时安全之间的层次关系，并提出了一种新的评估框架，能够识别在静态分析中未被发现但在动态测试中可被利用的漏洞。

**🔧 技术方法**

使用了静态分析工具Bandit和Semgrep，结合大语言模型（LLM）进行CWE推理，以及在Docker环境中进行动态漏洞验证的技术。

**📊 数据集**

使用了来自SecurityEval、RedCode和CyberNative数据集的1,355个Python样本进行评估。

**📈 对比分析**

与传统的静态分析方法相比，SPDF管道在654个静态清洁样本中识别出394个候选漏洞，动态验证确认或部分确认了95个文件的可利用性，整体管道率为14.53%。不同数据集的结果差异显著，RedCode的确认率为33.7%，CyberNative为28.6%，而SecurityEval仅为5.4%。

**⚠️ 局限性**

限制在于动态验证的环境限制可能导致某些漏洞未被确认，且所有结果均由自动化系统生成，可能与人工安全分析师的判断存在偏差。

---

## 66. Multilingual in Name Only? Cultural and Linguistic Weaknesses of LLMs in Urdu

**arXiv ID:** 2609.10758 | [PDF](https://arxiv.org/pdf/2609.10758v1)

**作者:** Farah Adeeba `[一作]` (University of Konstanz), Hassan Sajjad `[通讯]` (Dalhousie University)

**通讯引用:** 2837 | [OpenAlex ID](https://openalex.org/A5042954793)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了多语言大型语言模型（LLMs）在低资源语言（以乌尔都语为例）中的故事生成能力，生成了93个乌尔都故事，并对其进行了手动错误注释。

**💡 创新点**

创新点在于通过手动注释和分类，揭示了当前多语言LLMs在生成乌尔都语故事时的语法、语义和文化错误，强调了这些模型在低资源语言内容生成中的局限性。

**🔧 技术方法**

使用了三种现代LLMs（GPT-5.1、Qwen-3-Max、DeepSeek-3.1）进行故事生成，并通过手动注释分析其生成的文本。

**📊 数据集**

使用了名为乌尔都故事（Urdu-Stories）的数据集，该数据集包含93个故事，涵盖了151,965个单词和9,460个句子。

**📈 对比分析**

与人类创作的乌尔都文学进行比较，发现LLMs在生成的故事中存在大量语法和语义错误，且文化背景浅薄，生成的故事缺乏连贯性和创造性。性能上，LLMs在语法和语义准确性上表现不佳，且在文化适应性上存在显著不足。

**⚠️ 局限性**

本研究的局限性在于乌尔都故事数据集规模较小，仅涵盖三种商业模型，未来需要扩展到更多的低资源语言和其他模型以增强研究的普遍性。

---

## 67. Meta-Learning for Data-Efficient Plant Growth Estimation via Vision Transformers and Fuzzy Clustering

**arXiv ID:** 2609.10749 | [PDF](https://arxiv.org/pdf/2609.10749v1)

**作者:** Sheikh Hasan Elahi `[一作]` (Norwegian University of Life Sciences), Fadi Al Machot `[通讯]` (Norwegian University of Life Sciences)

**通讯引用:** 1398 | [OpenAlex ID](https://openalex.org/A5073646721)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结合视觉变换器（ViT）特征嵌入、基于聚类的任务构建和基于梯度的元学习的少样本回归框架，用于植物生长估计。

**💡 创新点**

创新点在于通过聚类构建任务，组织少样本回归任务，并展示了任务构建在嵌入空间中的重要性，优于传统的采样启发式方法。

**🔧 技术方法**

使用了视觉变换器（ViT）、模糊C均值聚类（FCM）和基于梯度的元学习（如MAML++）。

**📊 数据集**

使用了两个植物数据集：NMBU黄瓜数据集和竞争生菜数据集，分别包含464个和388个图像样本。

**📈 对比分析**

与传统基线（如SVR、随机森林和多层感知器）相比，MAML++在少样本设置下表现最佳，RMSE显著低于其他方法，且在不同数据集上具有良好的泛化能力。

**⚠️ 局限性**

局限性在于研究仅限于受控温室图像和单一特征回归，未来工作应考虑更广泛的作物多样性、时间建模和不确定性预测。

---

## 68. Governing AI Research Through Peer Review: A Mixed-Methods Study of the Longitudinal Effects of Ethics Flags Across Resubmissions

**arXiv ID:** 2609.10740 | [PDF](https://arxiv.org/pdf/2609.10740v1)

**作者:** Kento Nishi `[一作]` (Massachusetts Institute of Technology), Mfoniso Andrew `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在AI会议中，伦理审查标记是否能有效引导研究方向。通过对被标记的ICLR提交进行纵向分析，发现作者在审查后往往只是修改论文的表述，而不是改变研究的根本方向。

**💡 创新点**

创新点在于揭示了作者在面对伦理审查标记时，往往选择在论文表述上做出调整，而非实质性地改变研究方法或程序。提出了在重新提交时披露先前伦理标记的建议，以增强对未解决问题的问责。

**🔧 技术方法**

使用了定性分析和纵向研究的方法，结合对25个案例的审查历史和作者访谈，分析了作者在回应伦理审查时的决策过程。

**📊 数据集**

使用了ICLR会议的提交和审查数据，分析了446个被标记的提交案例，追踪其后续的公开重新提交情况。

**📈 对比分析**

与其他方法的比较显示，只有17%的重新提交在程序或方法上做出了实质性改变，绝大多数（83%）的案例仅在表述上进行了修改，表明伦理审查的影响有限。

**⚠️ 局限性**

限制在于研究仅集中于ICLR的提交，可能遗漏了因伦理标记而完全放弃项目的案例。此外，只有三位受访者回应，样本量较小，可能无法代表更广泛的研究者情绪。

---

## 69. HuRo: Robotizing Human Videos for Scalable VLA Pretraining

**arXiv ID:** 2609.10706 | [PDF](https://arxiv.org/pdf/2609.10706v1)

**作者:** Jinho Jeong `[一作]` (RLWRLD), Seon Joo Kim `[通讯]` (Yonsei University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究系统地探讨了机器人化的人类视频是否可以为视觉-语言-动作（VLA）策略的预训练提供有效且可扩展的监督。为此，开发了一种机器人化管道，将异构人类视频转换为与机器人对齐的观察和动作轨迹，并推断缺失的中间信号。

**💡 创新点**

创新点在于提出了一种机器人化管道，能够将人类视频转化为机器人对齐的观察和动作，同时构建了一个包含630K个机器人化剧集和1.42亿帧处理帧的大规模数据集HuRo。

**🔧 技术方法**

使用了机器人化管道技术，包括视觉机器人覆盖和动作重定向，结合了人类视频的注释和动作转换。

**📊 数据集**

使用了来自五个不同人类视频源的数据集，构建了HuRo数据集，包含630K个机器人化剧集和142M处理帧。

**📈 对比分析**

通过在四个真实世界的操作任务上进行评估，发现随着机器人化预训练规模的增加，整体完成率从51.5%提高到80.3%，在空间和视觉变化下的OOD完成率从34.9%提高到72.2%。与现有的机器人基础模型相比，HuRo在ID和OOD评估中均表现出更好的性能。

**⚠️ 局限性**

限制在于机器人化观察的保真度受限于重建和视觉转换质量，当前的机器人覆盖未明确建模渲染机器人与场景几何之间的遮挡，可能引入视觉不一致。此外，缺乏对接触丰富操作的力或触觉信号的捕捉，且运动重定向未建模自碰撞或物理接触。

---

## 70. A Penalty-Aware, Blockchain-based Cloud Monitoring System

**arXiv ID:** 2609.10704 | [PDF](https://arxiv.org/pdf/2609.10704v1)

**作者:** Christian Dienbauer `[一作]` (University of Vienna), Erich Schikuta `[通讯]` (University of Vienna)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种基于区块链的云监控系统，能够自动执行服务违约的罚款支付，确保消费者在服务违反时能够及时获得赔偿。

**💡 创新点**

创新点在于将智能合约应用于云服务的监控和罚款管理，实现了消费者和提供者之间的双边协议，确保透明的罚款管理。

**🔧 技术方法**

使用了区块链技术中的智能合约，具体实现基于IBM Hyperledger Fabric框架。

**📊 数据集**

使用了亚马逊和微软Azure的云服务及其监控API作为数据集。

**📈 对比分析**

与传统的云监控方法相比，本文的方法通过智能合约实现了自动化的罚款支付，避免了对第三方的依赖，性能上提供了更高的透明度和效率。

**⚠️ 局限性**

限制在于当前实现仅支持单一指标的可用性监控，未来需要扩展以支持更多服务类型和复杂的服务级别协议（SLA）管理。

---

## 71. Tapes Together Strong: The Co-evolution of Computation and Cooperation

**arXiv ID:** 2609.10817 | [PDF](https://arxiv.org/pdf/2609.10817v1)

**作者:** Kunal Jha `[一作]` (University of Washington), Eyvind Niklasson `[通讯]` (Google Paradigms of Intelligence)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为自生游戏理论（Autopoietic Game Theory）的计算模型，研究复杂代理系统中合作的演化，强调社会互动、复制机制和计算成本的内生性及其共同演化。

**💡 创新点**

创新点在于将社会困境直接嵌入计算物理中，展示了在资源稀缺的情况下，合作策略的自我复制能够得到促进，并且在没有记忆或选择的情况下，随机初始化的程序能够演化出自我复制和抑制破坏性偷窃的能力。

**🔧 技术方法**

使用了Z80机器代码环境进行实验，模拟自我复制程序的演化，并通过引入特定的计算指令来嵌入社会困境。

**📊 数据集**

使用了随机初始化的Z80机器代码程序，模拟了16384个程序在有限能量预算下的互动和复制。

**📈 对比分析**

通过与传统的进化博弈理论和人工生命模型进行比较，发现自生游戏理论能够在均匀分布的能量环境中有效抑制偷窃行为，并且在空间分布的环境中，局部互动能够促进更复杂的结构和任务表现。

**⚠️ 局限性**

限制在于模拟仅限于简化的囚徒困境和Z80汇编语言，假设了无限制的读写访问权限，且未考虑不同指令密度和硬件架构对复制成本和饥饿阈值的影响。

---

## 72. ExaServe: Large-Scale LLM Serving on Exascale HPC Systems

**arXiv ID:** 2609.10812 | [PDF](https://arxiv.org/pdf/2609.10812v1)

**作者:** Wenyi Wang `[一作]` (University of Chicago), Kyle Chard `[通讯]` (University of Chicago)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ExaServe，一个可安装的框架，用于在超算系统上进行大规模LLM服务的部署，解决了在领导级超算上部署LLM服务的工程挑战。

**💡 创新点**

ExaServe提供了一种可重复的部署路径，并揭示了未来超算LLM服务的关键障碍，特别是在控制平面和集中式令牌传递方面的瓶颈。

**🔧 技术方法**

使用了Ray Serve、vLLM、SGLang等技术，并结合MPI进行调度和模型权重的本地存储。

**📊 数据集**

在ALCF Aurora超算上进行了实验，使用了256个节点（3072个vLLM副本）进行性能评估。

**📈 对比分析**

与其他方法相比，ExaServe在非流式推理中几乎线性扩展，达到27.1k请求/秒，而流式推理在256个节点时达到的请求速率约为4.7k请求/秒，未能满足服务水平目标。

**⚠️ 局限性**

主要限制在于Ray Serve的控制平面在每次长轮询更新后重新解析每个副本句柄，导致O(N^2)的GCS查找模式，且在512节点的部署中出现了GCS饱和的问题。

---

## 73. Big Enough to Break Out: Tracking the Rising Capability of LLM Penetration-Testing Agents

**arXiv ID:** 2609.10780 | [PDF](https://arxiv.org/pdf/2609.10780v1)

**作者:** Victoria Lovelace `[一作]` (University of Virginia), Daniel Graham `[通讯]` (University of Virginia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较了两种基于PentestGPT的渗透测试系统：一个是传统的人机协作系统，使用开放权重的Kimi K2.5，另一个是较新的自主系统，使用Claude Opus 4.8。研究发现，自主系统在三个公共目标上均能完成任务，而传统系统在其中两个目标上未能完成。

**💡 创新点**

创新点在于通过比较不同版本的PentestGPT，揭示了大型语言模型渗透测试代理的能力进展，并探讨了限制这些代理在复杂任务中表现的因素，特别是规划能力而非记忆能力。

**🔧 技术方法**

使用了PentestGPT框架，结合了Kimi K2.5和Claude Opus 4.8两种大型语言模型，并测试了覆盖记忆层的效果。

**📊 数据集**

使用了三个公共脆弱机器作为数据集：Metasploitable 2、Bob和Tr0ll，这些机器在攻击结构和难度上各不相同。

**📈 对比分析**

比较了传统和自主系统的表现，发现自主系统在所有目标上均能完成任务，而传统系统在某些目标上停滞不前。覆盖记忆层的引入并未改善结果，反而增加了计算时间。

**⚠️ 局限性**

限制在于代理在规划和承诺方面的能力，而不是记忆能力。尽管引入了覆盖记忆层，但在自主实验中并未出现记忆丢失的情况，表明问题更可能出在代理未能有效利用已有信息上。

---

## 74. When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents

**arXiv ID:** 2609.10750 | [PDF](https://arxiv.org/pdf/2609.10750v1)

**作者:** Syed Shariyar Murtaza `[一作]` (Manulife), Arvid Frydenlund `[通讯]` (Manulife)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM代理技能检索，评估大规模（≈34k）技能库中合成监督对检索性能的影响，并提出并验证遗忘缓解技术。

**💡 创新点**

首次系统展示合成监督在技能检索中导致灾难性遗忘，并证明嵌入锚正则、LwF、EWC、L2-init等正则方法可同时提升内部分布性能并保持或提升外部分布性能。

**🔧 技术方法**

使用LoRA微调、InfoNCE、listwise损失、KL蒸馏、embedding锚正则、EWC、L2-init等技术，搭建bi‑encoder检索器和cross‑encoder reranker。

**📊 数据集**

数据集包括34,396个技能（skillhub、skills.sh）以及SkillsBench、Terminal‑Bench 2真实任务，合成任务Track A（≈1.7k）和Track B（≈13k）由GPT‑5、Claude等LLM生成并通过人工/LLM审核过滤。

**📈 对比分析**

与冻结的4B+BM25+RRF、0.6B dense检索器、BM25等基线比较；实验表明：单纯合成微调导致Recall@10下降；使用遗忘缓解后，Recall@10在真实/OOD集几乎保持不变，且在合成集提升≈13.98%；在多种Ring和BEIR基准上保持或提升性能。

**⚠️ 局限性**

局限包括：评估集规模小、真实与合成分布差异大、正样本覆盖稀疏、LLM生成与评估依赖闭源模型、仅限英文文本任务，未覆盖多语言或多模态。

---

## 75. Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking

**arXiv ID:** 2609.10745 | [PDF](https://arxiv.org/pdf/2609.10745v1)

**作者:** Parinthapat Pengpun `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于可解释式多模态大型语言模型与迭代检索的实体链接框架，用于提升多语种多模态实体链接任务中稀缺实体的识别与归一化。

**💡 创新点**

创新点在于将可思考（reasoning‑native）视觉‑语言模型与嵌入式检索结合，形成自我迭代查询与推理的闭环，并通过知识图谱结构稀疏度和受欢迎度等多维稀缺度指标揭示不同稀缺实体的失败模式。

**🔧 技术方法**

使用的技术包括Qwen3‑VL的思考与指令两种变体、BM25与多语言句向量检索、基于检索的增量推理与标题抽取两阶段流程，以及多模态输入（文本+图像）下的自回归推理。

**📊 数据集**

实验数据集为MERLIN（涵盖印地语、印尼语、日语、泰米尔语、越南语）以及对应的英文维基百科与维基数据知识图谱，另外公开发布了MERLIN‑Rare稀缺实体切片。

**📈 对比分析**

在整体MERLIN测试集上，最佳系统（8B‑Think+Embed）平均准确率达到87.9%，比SOTA Cultural Pangea提升6.9%，在不同稀缺度切片中的提升从5.5%到23.3%不等，证明检索与推理的协同能显著缓解稀缺实体问题。

**⚠️ 局限性**

主要限制包括对英语维基百科的依赖、检索召回率不足导致的72%错误比例、跨语言检索的跨写法对齐挑战，以及模型可思考模式与检索交互的普适性尚未在其他模型家族中验证。

---

## 76. Conformal Calibration Transfer

**arXiv ID:** 2609.10737 | [PDF](https://arxiv.org/pdf/2609.10737v1)

**作者:** Achref Doula `[一作]` `[通讯]` (Technical University of Darmstadt), Achref Doula (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Transported Conformal Calibration (TCC)，利用无标签配对数据将源空间的校准样本迁移到目标空间，并在目标空间通过无标签校正机制实现覆盖率保证。

**💡 创新点**

创新点：1) 使用配对无标签数据学习运输映射，将源校准信息迁移至目标域；2) 设计无标签校正方法 TCC‑KS（基于 Kolmogorov–Smirnov 证书）和 weighted‑TCC（基于重要性加权），在无目标标签下提供目标域覆盖保障。

**🔧 技术方法**

技术手段：域迁移模型（pix2pix 等）、自监督不确定性统计、Kolmogorov–Smirnov 检验、重要性加权分层 conformal prediction、有限样本覆盖理论。

**📊 数据集**

数据集：CIFAR‑100‑C、Tiny‑ImageNet‑C（多种噪声等级）以及跨模态 SAR→RGB 的 SEN12MS，用于评估在严重偏移和跨模态情形下的性能。

**📈 对比分析**

与 Oracle CP、无迁移 Weighted CP、Transported CP 等基线比较，TCC‑KS 在严重偏移下恢复覆盖率，覆盖率接近 1‑α，集合大小略增；weighted‑TCC 在轻微偏移下与 Transported CP 相当并更高效；整体优于基线。

**⚠️ 局限性**

局限性：需要无标签配对样本，配对质量对性能影响大；诊断指标（δ⁺、ESS）在极端不匹配下可能过于保守或误判；仅提供边际覆盖，未实现条件覆盖；假设源/目标标签一致且映射可保持。

---

## 77. When Information is Worth the Risk: Behavioral Valuation for Hazardous Robotic Exploration

**arXiv ID:** 2609.10726 | [PDF](https://arxiv.org/pdf/2609.10726v1)

**作者:** Alkesh K. Srivastava `[一作]` (Temple University), Philip Dames `[通讯]` (Temple University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种风险评估层框架，在危险环境中让机器人在获取信息和避免失效之间做决策。

**💡 创新点**

创新点在于将Prelec概率加权应用于信息增益评估，仅在路径排名层面调整风险感知，从而实现可解释的保守到激进的取值。

**🔧 技术方法**

使用了贝叶斯滤波、有限时限信息路径规划、Prelec概率加权、理论路径切换分析以及仿真实验。

**📊 数据集**

采用了人工合成的网格危险场景（不同密度、致命性和传感器噪声），共计数千次随机试验。

**📈 对比分析**

与Shannon信息增益、线性风险惩罚和概率约束基准进行比较，实验表明行为评估在保持信息获取的同时显著降低风险，且在多种危险结构下保持 Pareto 竞争力。

**⚠️ 局限性**

局限在于仅验证静态网格环境，参数调优依赖先验经验，且对非凸信息-风险前沿可能需要更复杂的多目标搜索。

---

## 78. AcFlow: Controlling Text-to-Image Diffusion Transformers via Learned Conditional Activation Flow

**arXiv ID:** 2609.10723 | [PDF](https://arxiv.org/pdf/2609.10723v1)

**作者:** Junran Wang `[一作]` (Georgia Institute of technology), Xinjie Shen `[通讯]` (Georgia Institute of technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在冻结的文本到图像扩散变换器（DiT）上，提出一种基于概念条件的速度场（velocity field）在推理时直接调节中间层图像-令牌激活，从而实现风格强度的连续控制与概念抑制。

**💡 创新点**

创新点在于：①将控制视为在激活空间中的“流”而非简单的方向或缩放；②使用共享的、文本条件化的速度场可同时处理多种概念，且在训练期间不需为每个概念单独拟合；③通过控制积分时长（horizon）提供连续可调的干预强度，并在未见概念上实现零样本泛化。

**🔧 技术方法**

技术核心包括：速度场建模（FlowBlock）与跨注意力条件化、激活干预的Euler数值积分、速度场蒸馏训练、单流图像令牌专属干预、以及对激活依赖方向和幅度的动态调整。

**📊 数据集**

数据集：用于风格控制的 MegaStyle（风格描述语料）和 FLUX.1-dev 生成模型；用于概念抑制的 InstructPix2Pix 的概念对（source/target）和大规模概念词表；所有实验都在 FLUX.1-dev（512×512）上进行。

**📈 对比分析**

与多种基线（Linear‑AcT、Mean‑AcT、ActAdd、SHIFT、Concept Sliders、Text Slider 等）对比，采用同一风格描述的高风格对齐点下，本文方法实现 0.5365/0.2860（风格/内容相似度），明显优于最佳基线 0.4397/0.2684；在风格–内容权衡曲线中位于高风格对齐区域的最优前沿；在未见风格族上亦能保持 0.442/0.281 的对齐性能。概念抑制实验显示能够在未训练概念上实现部分到完全抑制。

**⚠️ 局限性**

局限性：干预可能改变图像的布局、对象位置或姿态，无法保证原始构图完整性；对概念的精细定位与保持细节仍是开放问题。

---

## 79. An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics

**arXiv ID:** 2609.10712 | [PDF](https://arxiv.org/pdf/2609.10712v1)

**作者:** Ivan Moshkov `[一作]`, Igor Gitman `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一套基于 Nemotron‑3‑Ultra 的自然语言证明生成系统，并在 2026 年国际数学奥林匹克（IMO）中以 30/42 分成功夺金牌；

**💡 创新点**

创新点在于将多种后训练（SFT、RL）专用检查点与无形式证明器的高算力生成‑验证‑改进循环相结合，并公开发布了专用模型、训练数据、评测基准和完整实现；

**🔧 技术方法**

主要技术包括：长上下文监督微调（SFT）与强化学习（RL）后训练、生成‑验证‑改进（GVR）搜索策略、无参考的自然语言验证与评分、以及多模型集成和高算力搜索；

**📊 数据集**

使用的数据集为：Nemotron‑Math‑Proofs‑v3‑SFT（超 400k 例）、Nemotron‑Math‑Proofs‑v3‑RL（约 9.6k 例）、Nemotron‑IMO‑Bench（200 题）及 30 题开发集；

**📈 对比分析**

通过在 30 题开发集上对单模型、验证器组合、全集成进行 ablation，发现全集成在 8 轮内达到最高的 Jury‑score，最终在 IMO 2026 赛事中获得 30/42 分，显著优于单模型和传统方法；

**⚠️ 局限性**

局限性包括：依赖海量 GPU 计算（约 4,800 小时），对验证器精度仍有不足（误判率 1% 以上），缺乏形式证明器或外部工具的严格可靠性，且系统在新颖性、可解释性和资源节约方面仍有改进空间。

---

## 80. Analyzing Traditional and Neural Approaches to Multilingual Readability Assessment

**arXiv ID:** 2609.10792 | [PDF](https://arxiv.org/pdf/2609.10792v1)

**作者:** Joshua Wong `[一作]` (Harvard University), Chris Tanner `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 Transformer 编码器是否在不同语言的可读性评估任务中内部化了与传统基于特征的模型相同的语言学信号，并通过 SHAP 解释特征重要性与 TCAV 概念探测对比两类模型的内部表示。

**💡 创新点**

创新点在于首次将 SHAP 生成的特征重要性与 TCAV 概念探测相结合，形成跨语言可解释性评估框架，从而直接比较传统特征模型与 Transformer 的语义表征是否一致，并揭示语言与模型族群在可读性特征利用上的差异。

**🔧 技术方法**

所用技术包括：LFTK 语言特征提取、Logistic Regression / SVM / Random Forest 传统分类器、SHAP 进行全局特征重要性排序、Transformer 微调（XLM‑R 与各语言专属 BERT）以及基于 TCAV 的概念方向和影响度量。

**📊 数据集**

实验基于 ReadMe++ 多语言句子级 CEFR 可读性数据集（阿拉伯语、英语、法语、印地语、俄语），并在英语、阿拉伯语的概念集合中扩充了 CEFR‑SP 与 DARES 数据以增强概念池。

**📈 对比分析**

比较方法：在传统与 Transformer 之间对 QWK、准确率和宏观 F1 进行性能对比；在 Transformer 里用 CAV 准确率评估概念线性可分性，用 TCAV 分数和显著性检验衡量概念对 CEFR 预测的方向性。结果表明 Transformer 在所有指标上显著优于传统模型，并且在大多数语言中能复现传统模型的特征利用与方向性，语言专属模型往往表现出更稳定的层级一致性，而 XLM‑R 在中间层表现更不稳定；阿拉伯语出现 CAV 可分但 TCAV 无显著方向的异常。

**⚠️ 局限性**

局限性包括：仅使用单一句子级 ReadMe++ 数据集且未对随机种子进行平均；概念词汇主要基于 LFTK，非英语语言的特征空间相对有限；TCAV 仅探测线性方向，可能忽略非线性概念；实验仅涉及一个多语言模型与每种语言的单一专属模型，难以推广到更大或不同结构的模型；以及未覆盖文档级可读性与指令调优的大模型。

---

## 81. DR-LabStack: Design and Implementation of a Clinician-Facing Web System for Diabetic Retinopathy Prediction

**arXiv ID:** 2609.10796 | [PDF](https://arxiv.org/pdf/2609.10796v1)

**作者:** Yingfan Xu `[一作]` (Oklahoma State University), Ye Liang `[通讯]` (Oklahoma State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了一个名为DR-LabStack的React‑Flask Web系统，用于整合四个预训练的糖尿病视网膜病变预测模型，并提供统一的前端交互和后端服务；

**💡 创新点**

创新点在于构建了一个可处理不同模型输入顺序、序列化格式及预处理需求的通用接口，并实现了模型发现、特征映射与统一响应的完整工作流；

**🔧 技术方法**

使用的技术包括前端React（React Router、Axios、Form 控件）、后端Flask（REST API、Flask‑WTF）、Python 序列化工具（pickle、joblib、JSON）、Scikit‑learn 的 StandardScaler 以及 XGBoost、RuleFit 等机器学习模型；

**📊 数据集**

使用的数据集为公开的四个模型的预训练权重文件（含 14、6、8、25 个特征的权重和 scaler），以及合成的测试向量进行功能验证；

**📈 对比分析**

通过 Flask test‑client 进行 62 次服务请求和浏览器组件测试，验证模型加载、特征顺序、调用路径和阈值一致性，表现为所有模型均能正确返回 0/1 预测结果，功能表现良好但未评估预测准确度；

**⚠️ 局限性**

局限性包括：未进行临床有效性或诊断准确性评估，使用合成向量缺乏真实患者数据，缺乏完整的依赖版本管理和错误处理，缺少多浏览器与真实部署环境的性能评估。

---

## 82. A Bellman Optimality Equation for Plasticity

**arXiv ID:** 2609.10776 | [PDF](https://arxiv.org/pdf/2609.10776v1)

**作者:** Jeremy Lucas `[一作]` (Mila -- Quebec Artificial Intelligence Institute), Doina Precup `[通讯]` (Mila -- Quebec Artificial Intelligence Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `2704f255-0c84-4173-b83c-0e9a3dbea232` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并证明了一个用于最大化连续强化学习中可塑性的贝尔曼最优方程，并给出了基于多重重量背包动态规划的离散实现；

**💡 创新点**

首次将可塑性（从观测到动作的信息流）视为可最大化的目标，并将其与赋能（从动作到观测的信息流）统一到同一信息理论框架下；

**🔧 技术方法**

使用信息理论中的有向信息、通道容量、通用贝尔曼方程；设计了多重重量背包动态规划求解策略；在实验中采用了离散MDP、Road环境和两房间网格世界；

**📊 数据集**

Control‑Gated MDP（2 状态 2 动作）、Road Environment（5 状态 3 动作）以及 5×7 网格两房间环境（35 状态 4 动作）等自建离散实验环境；

**📈 对比分析**

与现有赋能最大化方法（基于贝尔曼的赋能优化）在相同环境下对比，证明可塑性价值迭代单调递增并收敛，最终可塑性值约为0.92bits，赋能为0bits；在更复杂环境中展示了可塑性策略能引导代理优先探索高可塑性区域；

**⚠️ 局限性**

算法在离散环境下可行但扩展性差，求解空间呈指数增长；缺乏连续或高维空间的在线样本算法；未对可塑性与赋能平衡的具体策略进行系统评估；

---

## 83. GRADE: Single-Frame Generative Radar Depth Estimation Under Visual Degradation

**arXiv ID:** 2609.10756 | [PDF](https://arxiv.org/pdf/2609.10756v1)

**作者:** Bin Zhao `[一作]` (Rice University), Nakul Garg `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉受损环境（烟雾、雾、暗光）下，利用单帧毫米波雷达与预训练的生成式扩散模型相结合，实现高精度密度深度估计。

**💡 创新点**

创新点包括：①将雷达谱映射为粗深度并作为条件，固定在扩散过程的每一步；②设计像素空间残差视觉引导模块，使模型在可见度降低时自然退化到雷达主导；③在单帧无运动、无SAR的前提下完成高细节深度重建。

**🔧 技术方法**

主要技术：4D雷达Spectrum → Transformer encoder‑decoder 产生粗深度；Latent Diffusion Model 预训练并在粗深度上条件化；ControlNet 风格像素空间视觉残差分支；DDIM 采样；多阶段训练（雷达、扩散、视觉引导）。

**📊 数据集**

使用约 95K 帧同步雷达‑摄像头‑深度数据（12 栋建筑，含 40K 帧真实烟雾）以及 IQ‑1M 室内数据进行预训练。

**📈 对比分析**

与 6 种基线（DA3、CaFNet、GRT、GRT+Image、RadarCam‑Depth 等）对比，清晰场景 MAE 0.303 m、烟雾场景 MAE 0.313 m；Chamfer Distance 清晰 0.120 m²、烟雾 0.114 m²；在不同烟雾密度下均保持性能优于所有基线。

**⚠️ 局限性**

限制：单帧推理时间较长，难以实时；缺乏时序一致性；对极端或非室内环境（户外、不同雷达硬件、未知材料）鲁棒性有限；生成先验可能产生幻觉，需加入不确定性评估。

---

## 84. Adaptive Margin Ordinal Loss: Penalizing Center-Class Hedging in Ordinal Classification

**arXiv ID:** 2609.10752 | [PDF](https://arxiv.org/pdf/2609.10752v1)

**作者:** Manisha Kandel `[一作]` `[通讯]` (University of Delaware), Manisha Kandel (University of Delaware)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了标准交叉熵在序数分类任务中导致模型倾向于中心类预测的现象，并提出了Adaptive Margin Ordinal Loss（AMOL）来显式抑制这种中心类回避，并引入Center‑Hedging Rate（CHR）作为诊断指标。

**💡 创新点**

创新点：①首次将中心类回避定义为独立的失败模式并给出定量指标；②提出可调的乘法权重 m(k,y)，在候选类靠近中心且真实标签远离中心时加大惩罚；③设计非对称版 AMOL‑asym，能够完全消除中心类回避。

**🔧 技术方法**

技术细节：使用多层感知机、KL 散度与高斯软目标、交叉熵、OLL、SORD 等基线；引入线性与指数形式的权重，评估 QWK、CHR、准确率、MAE 等指标。

**📊 数据集**

使用了四个公开序数分类基准：Synthetic、Wine Red、Wine White、Abalone（均来自 UCI），对特征进行了标准化处理。

**📈 对比分析**

对比方法：与 CE、OLL、SORD 等基线在 5 个随机种子下进行平均/方差比较；AMOL 在所有四个数据集上取得最高或并列最高 QWK；AMOL‑asym 在 Abalone 实现 CHR=0，中心回避完全消除；Wine White 的 CHR 降低 73%，且 QWK 也最高。

**⚠️ 局限性**

局限性：①需要调参 α；②非对称版要求明确中心，适用于单峰分布，可能不适用于多峰或非对称尺度；③不是严格的正确评分规则，可能影响校准；④对极端样本数量要求较高，样本不足时 CHR 估计不可靠。

---

## 85. Lie-Algebraic Bell Recurrences for Arbitrary-Order Twist Jets and Parallel-Mechanism Closure

**arXiv ID:** 2609.10748 | [PDF](https://arxiv.org/pdf/2609.10748v1)

**作者:** Daniel Condurache `[一作]` `[通讯]` (Gheorghe Asachi Technical University of Iași), Daniel Condurache (Gheorghe Asachi Technical University of Iași)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

开发了一套任意阶次的运动学构造，利用双螺旋理论和贝尔多项式实现串联传播、并联闭合递归以及刚体平台的仿射场（速度、加速度、冲动、冲量）构造。

**💡 创新点**

创新点包括：①对固定轴双螺旋关节导数的贝尔多项式显式表达；②利用同一闭合雅可比矩阵在所有阶次中复用，形成三角形的被动关节递推；③直接从扭矩级数得到点无关仿射不变量的精确算子多项式映射。

**🔧 技术方法**

使用的技术有：双数/双向量代数、双螺旋理论、贝尔多项式、Leibniz 乘积法则、实数化（realification）、三角闭合递推、精确算子多项式映射以及符号/数值差分验证。

**📊 数据集**

并未使用外部数据集，而是采用三个解析测试案例：通用非共面的 3C 链、RR+RRR 球形腕式并联机构以及 Hunt 型 6‑RUS 机构，以符号参数和给定轨迹进行验证。

**📈 对比分析**

通过独立计算扭矩级数和仿射场，比较其在速度、加速度、冲动、冲量四阶的数值结果；所有残差均低于 1e‑12（SI 单位），证明实现的数值精度和内部一致性优异。

**⚠️ 局限性**

局限性在于仅适用于固定拓扑且正则（非奇异）配置；不处理闭合矩阵秩亏、分支切换、动力学、弹性、接触或高阶级数误差传播。

---

## 86. Temporal and Multimodal Deep Learning for Cyberattack Detection in LEO Satellite Systems

**arXiv ID:** 2609.10746 | [PDF](https://arxiv.org/pdf/2609.10746v1)

**作者:** Kyle Stein `[一作]` (University of West Florida), Hossain Shahriar `[通讯]` (University of West Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究针对LEO卫星通信系统的多模态时序数据，提出了结构化的子系统融合MLP和分层多模态Transformer用于多类网络攻击检测；

**💡 创新点**

创新点在于：①在同一观测中分别编码硬件、轨道与射频特征并在Transformer中实现子系统间与时序间的注意力机制；②采用泄漏防护的实例级划分与跨卫星泛化评估，揭示了评估方式对性能的显著影响；

**🔧 技术方法**

使用的技术包括多模态深度学习（MLP、Transformer）、加权交叉熵、时序窗口构造、类别权重平衡、随机森林、XGBoost以及行级与时序级模型的对比；

**📊 数据集**

使用了UNSW‑IoTSAT数据集，包含两颗LEO卫星的硬件/环境、轨道/运动和射频三组特征，标注了七类正常与六种攻击；

**📈 对比分析**

通过与传统机器学习基线（随机森林、XGBoost、单体MLP）和行级/时序级版本的对比，实验表明分层Transformer在时序评估下达到91.66%准确率、85.63%宏F1，且在跨卫星测试中仍保持较高性能；

**⚠️ 局限性**

局限性在于：仅针对两颗卫星的有限数据集，评估在更大规模星座或完全未知平台上的泛化能力有限；同时RF特征对性能贡献最大，若实际环境中RF数据受限，效果可能显著下降。

---

## 87. Towards a Deterministic Math Solver for Clinical Language Models

**arXiv ID:** 2609.10728 | [PDF](https://arxiv.org/pdf/2609.10728v1)

**作者:** Felipe Ocampo Osorio `[一作]` (MIT Critical Data), Leo Anthony Celi `[通讯]` (MIT Critical Data)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文评估了一种程序生成与局部执行相结合的临床计算器接口，探讨其在 MedCalc-Bench 验证集上是否优于直接算术或手工实现。

**💡 创新点**

创新点在于将模型改为写短 Python 代码并在受限执行器中运行，而非直接算术，从而在大型模型中实现更可靠的数值结果。

**🔧 技术方法**

使用的技术包括 Qwen2.5-7B/32B-Instruct、vLLM 服务器、受限子进程执行器、代码生成与执行、对比统计与集群自举。

**📊 数据集**

所用数据集为 MedCalc-Bench Verified，共 1,100 条测试案例，覆盖 55 种临床计算器。

**📈 对比分析**

通过与 Open-book arithmetic、手工 22 个计算器库和 Blind Program-Solve 进行五次种子评估比较，32B 模型的 Program-Solve 在完整集合上达到 90.53% 正确率，显著高于 83.47% 的直接算术；在 7B 模型则无明显优势。

**⚠️ 局限性**

局限包括对公式版本、系数和使用范围的审计不足、缺乏完整覆盖、模型规模与性能关系不确定、局部执行器未完全沙箱、仅限英语和美国单位、成本和部署不测。

---

## 88. NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction

**arXiv ID:** 2609.10715 | [PDF](https://arxiv.org/pdf/2609.10715v1)

**作者:** The Intern-NCP Team `[一作]` (Shanghai Jiao Tong University), Bowen Zhou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在隐空间进行概念级别预测的语言模型（Latent-Space Language Model），在训练中同时优化标准下一个标记预测（NTP）和跨多标记的概念级预测（NCP），并将预测的概念反馈回标记级生成；

**💡 创新点**

创新点在于：①将可量化的离散概念词典直接从隐藏状态学习得到并用于概念级预测；②在隐藏层引入概念模块和层级残差（跨模块与内部模块残差）以实现多级信息融合；③将概念预测与Token生成端并行训练，提升优化效率；

**🔧 技术方法**

使用的关键技术包括：向量量化（VQ）与乘积量化构造概念词典；Transformer Encoder/Decoder与8层概念模块；层级残差连接（IRC+CRC）；多任务联合损失（NTP+NCP+VQ）；Moonlight Muon优化器；长上下文和多任务数据集；

**📊 数据集**

训练数据：5.73万亿token，来源于Dolma 3 Mix与Dolma 3 Dolmino；微调/域适配使用Magicoder、Orca-Math、TriviaQA-RC等；下游评测使用30个基准，包括MMLU、GSM8K、MATH-500、HumanEval、MBPP、ARC、HellaSwag等；

**📈 对比分析**

与基线OLMo-3-7B对比，Latent-Space模型在相同token数下收敛速度提升1.95×、最终训练损失降低0.091、下游宏观平均提升2.45分（GSM8K提升5.99分）。计算量仅占基线的85%，且在计算优化实验中实现1.74×的计算效率提升；

**⚠️ 局限性**

局限性：尚未在长上下文训练下验证；概念预测与下游性能的映射依赖训练数据分布；中期训练阶段模型性能波动与基线相反；域适配时仍可能出现知识遗忘；需要进一步探索更高阶概念空间与更长依赖的可扩展性。

---

## 89. Architecting the Secure AI-SOC: A Neurosymbolic Framework for Pipeline Integrity and Threat Mitigation

**arXiv ID:** 2609.10707 | [PDF](https://arxiv.org/pdf/2609.10707v1)

**作者:** Anna Gazani `[一作]` (Aristotle University of Thessaloniki), Georgios Koutidis `[通讯]` (Clone Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一种“神经-符号”防御框架，将 SIEM 预过滤层与 NeMo Guardrails 语义门控层结合，形成 AI‑SOC 的多层完整性防护。

**💡 创新点**

创新点在于：①通过定制 SIEM 解码器实现结构化、无延迟的预过滤；②在此基础上引入双轨（输入/输出）自检语义门控，实时抑制 promptware 攻击；③闭环遥测与 HITL 可视化，保障防御透明性与可追溯性。

**🔧 技术方法**

使用技术包括：Wazuh SIEM 的 PCRE2 解码器与原子规则、主动响应脚本；NVIDIA NeMo Guardrails 与 Colang 规则编写；Gemma‑3 / Llama‑3 LLM；Python Orchestrator 搭建完整管道。

**📊 数据集**

使用自行合成的 Wazuh 日志数据集（约 20 条带注入 payload 的攻击样本），并结合 MITRE ATLAS 语义标签对其进行分类。

**📈 对比分析**

对比实验表明：Layer‑1（SIEM 预过滤）能捕获 3/20 次攻击，Layer‑2（语义门控）能识别 18/20 次；整体架构在保持近乎零延迟的前提下，显著提升了对多阶段 promptware 的检测率；相较单层防御，误报率更低、响应时间更短。

**⚠️ 局限性**

局限性包括：对极其复杂的语义逃逸（如工具滥用、深度 jailbreak）仍有漏判；无法防御多轮上下文攻击、检索无关持久化与多模态 promptware；需要人工维护 SIEM 规则与 Guardrails 模板；对跨会话状态未做完整处理。

---

## 90. Overpainting: Localized Context-aware Diffusion Image Editing

**arXiv ID:** 2609.10811 | [PDF](https://arxiv.org/pdf/2609.10811v1)

**作者:** Sam Sartor `[一作]` (College of William and Mary), Pieter Peers `[通讯]` (College of William and Mary)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了名为overpainting的基于扩散模型的局部图像编辑方法，利用trimap实现对编辑位置与内容的精细控制。

**💡 创新点**

创新点在于引入trimap作为编辑掩码、结合joint attention与Teamwork信息共享策略并采用attention‑dropout平衡编辑质量与掩码遵循，以及构建自动化的训练数据生成与过滤 pipeline。

**🔧 技术方法**

技术包括：基于预训练FLUX.1 Kontext的扩散模型，LoRA联合注意力和Teamwork低秩适配器，attention‑dropout，BiRefNet改进的trimap提取，LightGlue匹配，CLIP/AlphaCLIP等评价指标。

**📊 数据集**

使用了从Pexels采集的11323张免版税照片、由Qwen2.5‑VL‑72B生成的编辑提示、FLUX.1 Kontext合成的编辑图像，并手工构造的180个测试案例和验证集。

**📈 对比分析**

通过对比FLUX Fill、FLUX.1 Kontext、Nano Banana、UltraEdit、BrushNet、SDEdit等模型，利用mask遵循度、编辑质量（CLIP‑IQA、EditCLIP、AlphaCLIP、EditReward）指标评估，结果表明该模型在保持编辑质量的同时实现了较好的掩码遵循，是现有局部编辑方法的综合改进。

**⚠️ 局限性**

局限性包括：在需要完全删除或添加对象时效果不佳；对人像的遮罩约束过强导致形状/姿态不符合预期；与基线模型相比，局部编辑会出现一定的质量下降。

---

## 91. TrajFusionNet+: Transformer-Based Prediction of Pedestrian Crossing Intention via Fusion of Trajectory Representations and Scene Graphs

**arXiv ID:** 2609.10806 | [PDF](https://arxiv.org/pdf/2609.10806v1)

**作者:** François G. Landry `[一作]` (Université de Moncton), Moulay A. Akhloufi `[通讯]` (Université de Moncton)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 TrajFusionNet+，一种用于预测行人过街意图的多模态 Transformer 模型

**💡 创新点**

引入了图注意力分支（GAM）使用 TokenGT Transformer 处理行人中心图谱，并在视觉注意力分支中加入预测边界框叠加与时间 Transformer 进行时序建模

**🔧 技术方法**

采用 Transformer（编码器-解码器、仅编码器）、Visual Attention Networks（VAN）、TokenGT Graph Transformer、SegFormer/DeepLabV3 语义分割、加权交叉熵/均方误差等技术

**📊 数据集**

在 PIE 与 JAAD 两个主流行人过街意图数据集上训练与评估，并提出跨数据集联合训练/单独评估的新协议

**📈 对比分析**

与 13+ 近年 SOTA 方法对比，TrajFusionNet+ 在 PIE 上达到最高准确率/F1，JAAD 上排名第二，并在跨数据集协议下显著优于其他方法，表现出更好的泛化能力

**⚠️ 局限性**

主要局限为模型参数量大、推理时间相对较长（尤其使用 SegFormer 时），且在不同环境下仍需进一步验证鲁棒性

---

## 92. How Much Velocity Does Off-Ball Space Value Need? A Broadcast-Viewport Benchmark

**arXiv ID:** 2609.10801 | [PDF](https://arxiv.org/pdf/2609.10801v1)

**作者:** Seongjin Choi `[一作]` `[通讯]` (Independent researcher), Seongjin Choi (Independent researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估广播视频轨迹下球员速度对场控和威胁加权指标的影响，并确定速度在位置插补、控制面和裁决层级的重要性。

**💡 创新点**

发现速度在控制面是主要贡献因素，裁决层几乎不受影响，并给出了可容忍的噪声阈值约1 m/s，说明仅需可见速度即可；同时引入噪声阶梯和视口宽度分析验证结果。

**🔧 技术方法**

使用Spearman式到达时间控制模型、加权Logistic、有限差分速度估计、位置插补方法B2/B4以及噪声阶梯与帧常数分解等技术。

**📊 数据集**

实验数据来源于三场Metrica Sports样本赛季第一半场（5 Hz）以及11段SoccerNet‑GSR 30 s广播片段（25 Hz）。

**📈 对比分析**

通过四种速度模式（零、可见观测、真值可见、全真值）与基准表面配对MAE、xT误差、团队控制误差比较，发现忽略速度在控制面损失1.5–1.8 MAE，裁决误差仅0.12–0.19。

**⚠️ 局限性**

局限性包括仅使用单一控制模型、单一跟踪提供商、首半场数据、有限视口宽度、缺乏真实隐蔽玩家速度以及噪声阶梯为理想化仿真。

---

## 93. CARTS: Contextual Autoregressive Rank Transcoding Steganography for Full-Capacity Keyed Text Encoding

**arXiv ID:** 2609.10744 | [PDF](https://arxiv.org/pdf/2609.10744v1)

**作者:** Wissam Ghantous `[一作]` (University of Central Florida), Alexander V. Mantzaris `[通讯]` (University of Central Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为CARTS的键控文本对文本隐写术框架，并对其核心的Calgacus构造给出了形式化的正确性证明与安全问题定义。

**💡 创新点**

创新点在于首次把基于自回归语言模型的隐写方法进行严格的数学建模，定义了上下文搜索、键碰撞、信息等价性和键的不交换性等计算问题，并对它们的理论关系进行了探讨。

**🔧 技术方法**

技术上使用了自回归语言模型（以Llama‑3‑8B‑Instruct为例）、token‑rank映射、rank‑coordinate表示、以及离散置换与集合作用的数学框架。

**📊 数据集**

实验数据集主要为24条长度4~9个token的自然语言payload和60条基于多种主题的prompt key，基于Llama‑3‑8B‑Instruct GGUF模型进行推理。

**📈 对比分析**

通过五个实验评估实现正确性、键碰撞、键稳定性、键不交换性以及对token扰动的鲁棒性；结果显示实现完全可逆、随机key下无碰撞、局部碰撞极少、键不交换、但对任何token扰动均失效。

**⚠️ 局限性**

局限性包括：仅在单一模型与特定token化规则下验证；对更大key空间、更多payload分布及语义质量未做系统评估；以及对噪声鲁棒性的缺乏，需要后续加入纠错机制。

---

## 94. MHE-Former: Multi-Hypothesis Transformers via Entropy Maximization for 3D Mesh Recovery

**arXiv ID:** 2609.10743 | [PDF](https://arxiv.org/pdf/2609.10743v1)

**作者:** Boshu Jia `[一作]` (Communication University of China), Angela Yao `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多假设 3D 网格恢复中，本文提出 MHE-Former，利用熵最大化实现多样化的手体网格生成，并提供基于 Vision‑Language Model 的交互式假设选择。

**💡 创新点**

创新点在于将熵最大化与 Transformer 编码器结合，设计双分支 MH‑Decoder（确定性分支+基于 Normalizing Flow 的随机分支），以及利用 VLM 进行上下文感知的假设筛选。

**🔧 技术方法**

技术方面使用 ViTPose 作为编码器，注意力驱动的 Masked Flow 生成多假设，LoRA 轻量化微调，Entropy、先验与重建三项损失，VLM（Seed‑2.0‑Pro）实现交互式选择。

**📊 数据集**

实验数据集涵盖 Human3.6M/ AH36M、MPI‑INF‑3DHP、UP‑3D、MS‑COCO、HO3D、RHD/ARHD、3DPW/A3DPW 等。

**📈 对比分析**

与 Deterministic、ProHMR、MDN、CVAE、VMarker‑Pro、CtF‑MHE 等方法对比，MHE‑Former 在 Best Hypothesis、Relative Diversity、PJD 等指标上均突破或逼近 SOTA，特别在遮挡和交互场景下表现突出。

**⚠️ 局限性**

局限性包括对多视角渲染与 VLM 推理的计算开销，弱监督下多样性与一致性调参的复杂性，以及对 VLM 质量和可扩展性的高度依赖。

---

## 95. The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes

**arXiv ID:** 2609.10739 | [PDF](https://arxiv.org/pdf/2609.10739v1)

**作者:** Dylan Jayabahu `[一作]` `[通讯]` (University of Waterloo), Dylan Jayabahu (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一个单一秘密位的二元报告游戏中，研究者探讨了线性探针（truth probe）在标签完美共轭时的识别失效，并通过引入随机码本与混合上下文训练，使得即使在奖励训练的欺骗策略下，探针仍能在最终层以AUROC 1.0准确恢复真值。

**💡 创新点**

创新点在于揭示并量化了标签完美共轭导致的“完美别名”问题，提出了随机码本与混合上下文训练的修正方案，并通过大规模层级对照验证了该修正能在奖励训练下保持线性可检索性。

**🔧 技术方法**

使用了线性逻辑回归探针（logistic probe）对LLM残差流进行读取，并在不同模型架构（Llama‑3.1‑8B、Gemma‑2‑9B、Mistral‑7B‑v0.3、Qwen2.5‑3/7/14/32B）上进行实验；同时采用随机码本（codebook）随机化token与语义动作的对应关系。

**📊 数据集**

数据集为自定义的秘密位游戏，包含两种变体：普通单字令牌回答与每回合随机码本的代码书变体，实验共收集数千个episode，用于训练与评估探针。

**📈 对比分析**

方法上对比了仅在盟友（compliant）情境下训练的探针与同时包含盟友与对手（mixed）情境的探针，结果显示在751个(cell, layer)对中，盟友拟合探针与行动探针的AUROC严格满足1‑AUROC关系；而在奖励训练的欺骗策略上，盟友拟合探针几乎为0，而混合拟合探针在最终层达到AUROC 1.0。

**⚠️ 局限性**

局限性包括仅在极其简单的单字秘密位任务上验证；探针读取位置为回答预测点，可能仅复制提示中的真值；缺乏对长文本或多字答案场景的评估；并且混合训练需要对抗性标签，可能不具备通用可复制性。

---

## 96. Empirical Evaluation of Membership Inference Attacks on NLP Text Classifiers: A Baseline Study on SST-2

**arXiv ID:** 2609.10935 | [PDF](https://arxiv.org/pdf/2609.10935v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 97. CMNIE: An Information Extraction Benchmark for Chinese Military News

**arXiv ID:** 2609.10722 | [PDF](https://arxiv.org/pdf/2609.10722v1)

**作者:** Yan Yu `[一作]` (National University of Defense Technology), Mao Wang `[通讯]` (National University of Defense Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CMNIE——一个中文军事新闻信息抽取基准，统一标注事件、事件参数、命名实体和实体关系四个层次。

**💡 创新点**

创新点在于：①构建统一的四层标注schema，覆盖事件、参数、实体与关系；②从公开军事新闻中精选13k条实例，包含空事件与非空事件，体现真实数据稀疏与长尾分布；③提供标准化的实验基准，方便对比监督模型、零射击LLM和微调LLM在精确/松弛边界匹配下的性能。

**🔧 技术方法**

采用的技术包括：传统监督IE模型（DyGIE++、OneIE、UIE）；零射击LLM（Qwen2.5、GLM‑4、Llama系列、Moonshot、DeepSeek、GPT‑5.1）；以及微调LLM抽取方法（GoLLIE、ADELIE、KnowCoder），并将所有模型统一在相同的schema与评测协议下进行评估。

**📊 数据集**

使用的数据集是CMNIE，涵盖13,000条公开军事新闻，标注了6,997条事件、23,087条事件参数、97,508条实体和40,252条关系，且对训练、开发、测试进行了均衡划分。

**📈 对比分析**

实验结果显示：Fine‑tuned LLM（尤其GoLLIE+Qwen2.5-7B-Instruct）在事件类型识别、参数识别、NER和RE上取得最优成绩；传统监督模型OneIE在参数分类上表现最好；零射击LLM整体性能落后20‑30个百分点，但在松弛边界匹配（relaxed）下可达到接近Fine‑tuned水平，凸显边界精度为零射击难点。

**⚠️ 局限性**

局限性包括：①部分事件类型稀疏导致模型难以学习；②空事件实例较多，可能影响模型对事件存在性的判断；③仅覆盖中文军事新闻，缺乏跨语言验证；④使用词表检索获取候选实例，可能忽略隐式或非典型事件；⑤潜在的双重用途与数据版权需谨慎使用。

---

## 98. CamPilot: A Multi-Agent Cinematic Assistant for Camera-Controlled Movie Generation

**arXiv ID:** 2609.10943 | [PDF](https://arxiv.org/pdf/2609.10943v1)

**作者:** Yang Wu `[一作]` (Worcester Polytechnic Institute), Yu Shen `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出CamPilot，一种多智能体框架，用于文本到电影的生成，先进行情节与镜头规划，再通过可控摄像机工作生成连贯电影；

**💡 创新点**

创新点在于将专业摄像机语言（角度、镜头大小、运动细节）从14K部真实电影中学习到的摄像机工作规划器，并采用GRPO强化学习实现结构化决策；

**🔧 技术方法**

核心技术包括基于LLM的多智能体规划器（情节、镜头、摄像机工作），GRPO强化学习训练摄像机工作规划器，评估–修订循环提升质量，以及字符库与帧对帧条件化生成；

**📊 数据集**

使用数据集CamEval，由14K真实电影剪辑与VLM生成的结构化拍摄脚本和摄像机工作标签构成；

**📈 对比分析**

与四类基线（Standard、Vanilla‑SFT、DreamFactory、MovieAgent）比较，CamPilot在摄像机工作分类和下游视频质量（CLIP、Inception、Sub_Cons、Aesthetic）上均取得显著提升，尤其在宏观准确率和连贯性指标上领先；

**⚠️ 局限性**

局限性包括摄像机语言覆盖有限、规划准确性与误差传播问题、生成器依赖性与控制力度有限，以及评测指标与人类审美对齐不足。

---

## 99. "Coder first, advocate second, college student third": The Liminality of Going to College as a Blind Computing Student

**arXiv ID:** 2609.10942 | [PDF](https://arxiv.org/pdf/2609.10942v1)

**作者:** Isabela Figueira `[一作]` (University of California, Irvine), Stacy M. Branham `[通讯]` (University of California, Irvine)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过半结构化访谈收集10名盲/低视力大学计算机专业学生的经验，并采用生活转变与交叉临界理论对其从高中到大学的转学过程进行主题分析。

**💡 创新点**

创新点在于首次将交叉临界理论应用于盲/低视力学生的学术转移，揭示其多重生命周期转变如何相互叠加导致“持续交叉临界”现象，并提出转移教育技术框架来指导干预。

**🔧 技术方法**

使用的技术包括半结构化访谈、录音转写、主题分析（Abductive）、交叉临界与抗结构、Communitas 等社会理论框架的编码与分析。

**📊 数据集**

数据集由10名盲/低视力学生的访谈记录组成，涵盖性别、种族、专业、辅助技术使用等信息。

**📈 对比分析**

本研究未进行定量比较或性能评估，而是以质性案例研究方式呈现发现；因此无性能指标可比。

**⚠️ 局限性**

局限性包括样本量小、仅涵盖美国和加拿大的四年制高校BLV计算机专业学生，且可能存在回忆偏差和对其他学术或职业转移情境的普适性不足。

---

## 100. A2ABreak: Systematic Security Analysis of the A2A Protocol

**arXiv ID:** 2609.10871 | [PDF](https://arxiv.org/pdf/2609.10871v1)

**作者:** Alireza Lotfi `[一作]` (Purdue University), Elisa Bertino `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过LLM辅助将Agent2Agent（A2A）协议规范转化为正式有限状态机，并在此基础上进行系统的安全漏洞发现。

**💡 创新点**

首次构建A2A完整FSM模型并结合对抗验证实现规范级漏洞识别。

**🔧 技术方法**

采用双通道语义分离、LLM推理+约束语言、对抗验证及人工检查等技术。

**📊 数据集**

使用A2A协议全文及从中提取的约929条结构化语句作为输入数据集。

**📈 对比分析**

与零射击LLM baseline对比，A2ABreak发现11个漏洞，精确率73.3%，F1分数84.6%。

**⚠️ 局限性**

局限于仅覆盖协议层面，未考虑实现细节与传输安全，缺乏实时模拟框架。

---

## 101. Expressive Robotic Pianist: Mastering Complex Piano Repertoire with Graph-Mimic and Musical Dynamics

**arXiv ID:** 2609.10844 | [PDF](https://arxiv.org/pdf/2609.10844v1)

**作者:** Yanhong Liang `[一作]` (Zhejiang University), Hongtao Wang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

基于强化学习的控制框架，使18自由度机器人手能够在多种难度级别（Grade 1–7）的钢琴曲目中执行高保真演奏，并实现自然的指弹预压与按键触碰动作。

**💡 创新点**

创新点在于提出Graph‑Mimic——一种基于动作帧图（Action Frame Graph）的姿态相似度度量，使机器人能够捕捉人类手部空间关系而非单纯角度；以及简化的钢琴声学模型与音乐动态奖励，直接将键的角速度映射到MIDI响度，实现表达性力度控制。

**🔧 技术方法**

核心技术包括：1）强化学习（MuJoCo/UR5+InReal手）结合Graph Distance与力度奖励的多目标奖励函数；2）动作帧图建模与L2距离优化；3）基于键角速度的能量转移与弦振动简化公式，用以生成MIDI Velocity；4）MediaPipe视频姿态提取与图结构化。

**📊 数据集**

使用公开的钢琴MIDI曲目（多种风格、Grade 1–7）以及从YouTube视频通过MediaPipe提取的人类手部姿态数据；机器人仿真数据与真实实验数据（UR5 + InReal手 + Yamaha P‑48B）。

**📈 对比分析**

与无Graph与力度奖励的基线RL策略、人类钢琴家演奏以及人类听众的主观评价进行比较；指标包括：F1分数（模拟0.96，实机0.80–0.95），力度匹配率（约85.7%），以及感知测试中机器人表现与人类无显著差异（非专业听众）。

**⚠️ 局限性**

局限性包括：1）力度模型仅使用四级动态（p/ mp/ mf/ f），未覆盖完整动态范围；2）控制仅关注手部，未实现臂部动力学协同；3）依赖仿真状态（键角速度）进行训练，实际部署需通过MIDI或传感器估算；4）训练策略针对特定曲目，缺乏通用即兴或实时性能。

---

## 102. Evaluating Scaffolding-Oriented Multi-Agent Large Language Model System for Clinical Interview Training

**arXiv ID:** 2609.10939 | [PDF](https://arxiv.org/pdf/2609.10939v1)

**作者:** Luming Yang `[一作]` (Ohio State University), Li Lu `[通讯]` (Guangzhou Medical University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并评估了一种基于多智能体大型语言模型（LLM）的标准化患者（AI-SP）训练平台，用于医学生临床访谈练习。

**💡 创新点**

创新点包括：① 将患者代理、教学代理和评估代理协同工作，提供分阶段 Socratic 提示和即时评估；② 侧重提升过程质量（沟通、同理表达等）而非单纯诊断准确性；③ 发布了三位专家标注的 207 次访谈会话数据集，为后续研究提供标准化资源。

**🔧 技术方法**

使用技术：大型语言模型（如 GPT-4）驱动的患者、教师和评估代理；多智能体系统架构；Python/SQL 后端与对话管理；OSCE 对齐的评估框架；可视化和统计分析工具。

**📊 数据集**

数据集：207 组访谈（118 学习 + 89 考试），共 4,815 条学生与 AI-患者对话；三名临床专家标注了患者可信度、学生对话意图、辅导需求和全局 OSCE 评分等四维框架。

**📈 对比分析**

比较方法：随机对照实验（N=100）对比多智能体学习（MA）与结构化非 LLM 控制（CT）。主要结果为 MA 组最终 OSCE 评分提升 71.8% vs 55.6%（显著且效应大），在沟通和同理表达等子域提升最显著；诊断准确率无显著差异。附加探索性分析包括聚类表型、学习轨迹、可用性和参与度评估。

**⚠️ 局限性**

局限性：① 对照仅为结构化非 LLM 学习，未与人类标准化患者或教师反馈比较；② 研究仅涉及急腹症单一病例、样本量有限；③ 未评估长期保留或真实临床转移效果；④ 未测量真实患者对同理感知；⑤ 缺乏延迟评估和渐进式失效测试。

---

## 103. ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs

**arXiv ID:** 2609.10895 | [PDF](https://arxiv.org/pdf/2609.10895v1)

**作者:** Yizhan Li `[一作]` (Université de Montréal), Bang Liu `[通讯]` (Université de Montréal)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建并发布了ReactHuman基准，用以评估多模态大型语言模型在面对突发物理危害时的即时安全决策能力，采用冻结-预测协议让模型在模拟停止后提交行动计划，再由仿真执行并观察物理结果。

**💡 创新点**

创新点包括：①物理真实、可复制的评估流程；②将评估从被动问答转为主动行为执行；③设计了包含外观与物理不一致的对抗测试；④提出五维度指标（合理性、安全性、物理精准度、意图一致性、手部运动轨迹）来细粒度诊断失败。

**🔧 技术方法**

技术栈主要由：LLaMA/Claude等冻结LLM进行语义规划、种子确定的随机化生成物理参数、基于Genesis的240Hz刚体仿真、预训练的全身控制器（Unitree G1）实现动作执行，以及冻结-预测的评估协议。

**📊 数据集**

使用自研的无标注、可重复生成的数据集：17类突发事件共计1000+场景，三目视角、全景摄像、每场景均附有仿真导出的准确标注（动作标签、冲击点、时间），并在其中混入多种外观与物理不匹配的测试对象。

**📈 对比分析**

对7款零射击MLLM在306个平衡子集（每类18场景）进行评估，指标包括语义动作准确率、是否违反安全规则、终点距离、意图一致性、手部最短距离。最高安全率达88.2%，最高准确率63.7%，平均安全率80.8%、准确率54.0%；未见随模型规模提升而显著改进，常见错误为“冻结不动”“对速度判断二元化”“对外观误判物理属性”。

**⚠️ 局限性**

局限性：仅使用刚体仿真，忽略变形/破碎等真实物理；评估为单次冻结决策，缺乏闭环实时反馈；动作执行依赖预训练控制器，无法验证模型在真实机器人上的可迁移性；基准结果随模型API更新而变化，需持续维护。

---

## 104. DriftNet: A Dual-Head Trajectory Transformer for Detecting and Localizing Prompt Injection in LLM Agents

**arXiv ID:** 2609.10892 | [PDF](https://arxiv.org/pdf/2609.10892v1)

**作者:** Asif Pinjari `[一作]` (Northern Arizona University), Mithun Paul Saint-Germain `[通讯]` (Northern Arizona University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种双头Transformer模型 DriftNet，用于一次性检测并定位LLM代理中的间接 prompt injection 攻击，给出轨迹级判定与每一步的四类标签。

**💡 创新点**

创新点在于：①将轨迹判定与逐步标签联合输出，满足操作员的完整 triage 需求；②仅使用冻结的句子编码器与四个无身份的世界特征，模型参数少于两百万，远低于常见 LLM‑scale 防御；③在 AgentDrift 基准上实现极高的检测与定位精度。

**🔧 技术方法**

使用技术包括：Transformer 编码器（1–3 层，4 头），双头结构（轨迹级与逐步分类），加权联合损失（类权重交叉熵），冻结的 all-mpnet-base-v2 句子编码器，身份无关的世界特征，AdamW 优化器与三阶正则化。

**📊 数据集**

数据集为 AgentDrift 基准（12,536 条轨迹，71,024 步级标签），采用任务分离（task‑disjoint）拆分，保证训练与测试不共享任务模板。

**📈 对比分析**

与同样拆分重训练的表面基线（逻辑回归）对比，DriftNet 在轨迹 F1 为 0.983、攻击召回 0.985、定位精度 EM_I 98.7%、IoU_H 97.9%；召回率从基线的 0.579 提升至 0.985，误报率显著下降，特别是在部分劫持和延迟执行的难题上。

**⚠️ 局限性**

limitations：①数据来源为单一生成器的合成轨迹，缺乏真实流量和多生成器泛化评估；②对世界身份正则化的潜在依赖；③需要事先提供世界上下文特征，缺乏无世界信息时的鲁棒性评估；④未对适应性攻击进行测试；⑤模型对更长序列的表现未验证。

---

## 105. STV Audit Graphs: A Visual Tool to Measure Election Stability

**arXiv ID:** 2609.10887 | [PDF](https://arxiv.org/pdf/2609.10887v1)

**作者:** Edouard Heitzmann `[一作]` `[通讯]` (University of Colorado Boulder), Edouard Heitzmann (University of Colorado Boulder)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了审计图（Audit Graphs）这一工具，用于量化单一可转让票（STV）选举的稳定性并实现风险限制审计（RLA）

**💡 创新点**

创新点在于将STV的算法路径建模为图结构，定义可行边与可疑边，形成可解释且可检验的断言框架，并引入“可疑阈值”与“弱安全”概念，解决了STV在理论上易失稳但实际稳定的矛盾

**🔧 技术方法**

利用图论（构造通用与可疑图）、计票函数与STV规则（WIGM）、风险限制审计框架（SHANGRLA）以及两种具体检验方法：基于不匹配的mismatch测试和基于delta方法的置信区间检验

**📊 数据集**

使用全球范围内真实STV选举数据，包含英国苏格兰城市议会、澳大利亚新南威尔士州、美国华盛顿州等不同规模的选举，CVR数据与人工纸质投票记录（MVR）进行对比

**📈 对比分析**

与传统的全局不匹配审计相比，基于审计图的mismatch方法在低噪声（<0.2%）下成功率更高；delta方法在高噪声环境下更稳健，但对高度连通节点的样本量要求更大。整体ASN（平均样本数）在可疑阈值与图构造复杂度平衡时表现良好

**⚠️ 局限性**

局限性包括：早期层级组合爆炸导致图构造困难，批量淘汰快捷方式不适用于某些强势候选人情况；可疑图的“宽松”假设可能导致不完全可行路径；delta方法在高阶节点上效率下降；需要进一步开发非线性边界的检验超martingale以提升效率

---

## 106. Relatively Smart II: Tractable or Semi-Supervised Instance-Optimal Learning

**arXiv ID:** 2609.10886 | [PDF](https://arxiv.org/pdf/2609.10886v1)

**作者:** Shaddin Dughmi `[一作]` (University of Southern California), Alireza F. Pour `[通讯]` (University of Waterloo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文证明了在无分布限制的二元分类中，经验风险最小化（ERM）以及任何合适的一致学习器都是相对聪明的学习器，并构造了一种半监督相对聪明的学习算法，该算法在保证相同的可证错误率的同时，只需线性标签样本、但需要二次量级的无标签样本；随后证明了在仅能通过对抗性ERM查询访问假设类的可行学习者上，任何实现显著降低标签样本量的相对聪明学习器都必定不可行。

**💡 创新点**

①首次证明ERM（以及所有合适一致学习器）具备相对聪明性，消除了对One-Inclusion-Graph的依赖；②提出了新的半监督相对聪明学习框架，利用无标签样本将样本复杂度的二次放大转移到无标签侧；③给出可证错误率与分布固定学习者最优解之间的深度关联，并通过统一性测试与覆盖数论证了标签效率与可证性之间的紧耦合；④揭示了在可访问ERM或知识分布的情况下，半监督相对聪明学习器的可行性与计算复杂度之间的根本矛盾。

**🔧 技术方法**

采用双抽样与覆盖数论证、可证错误率与分布固定错误率的下界、覆盖数与包络数关系、一次性留一/留多转导学习（One-Inclusion-Graph与其推广）、均匀性测试技术、基于Oracle的自适应查询分析以及指数小概率论证等工具。

**📊 数据集**

本工作为纯理论研究，未使用任何实际数据集；所有结论均来自抽象假设类与分布构造。

**📈 对比分析**

由于研究目标为理论极限分析，未进行实验比较；论文通过证明 ERM 的相对聪明性所需样本量上界为 O(m²)，并展示半监督学习可将标签样本量降低至 O(m)，但其实现需要指数级的无标签样本与不可实现的Oracle调用，突显了标签效率与可行性之间的取舍。

**⚠️ 局限性**

研究仅局限于二元分类与可测分布，且半监督相对聪明学习器在实践中不可实现；此外，证明依赖于特定构造的假设类，未给出在更一般设置下的上界或下界；最后，论文未给出可扩展到多分类或非可测情形的方案。

---

## 107. Alternative AI Philosophy: Daoism as Method for AI in Education

**arXiv ID:** 2609.10842 | [PDF](https://arxiv.org/pdf/2609.10842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 108. Project Qualia: Recovering Experiential Music Structure from Session Co-occurrence Data

**arXiv ID:** 2609.10862 | [PDF](https://arxiv.org/pdf/2609.10862v1)

**作者:** Nizam Mohammed `[一作]` (Independent Researcher), Dimuthu D. K. Arachchige `[通讯]` (Hampton University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

从Last.fm收集并清洗1.29亿次收听记录，构建531.6百万条训练数据，并训练了基于skip-gram的Song2Vec模型；随后使用artist‑residual方法证明了听众行为中存在独立于艺术家身份的“体验相似性”信号。

**💡 创新点**

创新点在于：①首次在大规模收听行为中挖掘出与艺术家身份无关的体验层结构；②提出并验证了artist‑residual子空间分析方法，能从嵌入中分离艺术家信息与体验信息；③为后续构建基于JEPA的经验嵌入模型奠定实验依据。

**🔧 技术方法**

使用的技术包括：Python/SQLite/Parquet/duckdb数据处理，Gensim实现的Word2Vec skip‑gram，负采样，余弦相似度统计，artist‑residual向量化与归一化，近邻检索与可视化。

**📊 数据集**

数据集：从Last.fm公共API抓取9,396位高频用户的1.29亿次scrobble，去重后得到1,960,021条唯一曲目，531.6百万条训练scrobble，约53.7万条会话。

**📈 对比分析**

对比方法：在原始嵌入空间和残差空间计算随机跨艺术家曲目对的余弦相似度；残差空间中4,577对曲目余弦≥0.70，远高于0.0005的随机基线，表明残差空间确实捕获了显著的跨艺术家体验结构。

**⚠️ 局限性**

局限性包括：仅覆盖Last.fm重度用户，难以代表普通听众；存在K‑Pop偏倚与自动播放噪声未完全剔除；artist‑residual方法假设均值能代表艺术家中心，对某些艺术家效果不佳；Word2Vec缺乏序列和用户上下文，未实现对体验层的直接学习；缺少下游任务验证与音频特征融合。

---

## 109. REACH: Controller-Managed Long-Span ECC for HBM AI Inference

**arXiv ID:** 2609.10861 | [PDF](https://arxiv.org/pdf/2609.10861v1)

**作者:** Rui Xie `[一作]` (Rensselaer Polytechnic Institute), Tong Zhang `[通讯]` (Rensselaer Polytechnic Institute)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对高带宽内存（HBM）下的 AI 推理任务，提出一种控制器级长跨度 ECC 体系结构 REACH，能在保持 32 B 事务正常完成的同时，将异常错误投递至全局 RS 码做擦除恢复。

**💡 创新点**

创新点在于：① 将局部 RS(38,32) 代码用于快速判定 32 B 请求，只有在无法纠正时才将其位置转换为外层 RS(1138,1024) 的擦除符号；② 通过差分奇偶校验显著减少写入时的全局奇偶更新量；③ 采用端点协同设计，将 6 B 元数据与 32 B 数据一起存储，无需额外的 I/O 负载。

**🔧 技术方法**

核心技术包括：内部与外部分层 RS 纠错编码、差分奇偶校验算法、基于地址的擦除映射、端点协同元数据存储、分域服务（HBM 侧内层解码、外层擦除恢复）以及对 HBM 交互的时序和资源管理。

**📊 数据集**

使用公开的 LLM 推理工作负载（Qwen3‑32B、GLM‑4.7‑Flash、DeepSeek‑V2‑Lite），通过 Ramulator2 模拟 HBM 交互，并利用 ASAP7 工具链进行硬件综合与性能评估。

**📈 对比分析**

与传统的直接长码（direct‑long）和 32 B 本地 ECC 控制器进行对比。REACH 在 2.69 TB/s 业务流量下，面积下降 55.8%、功耗下降 57.7%，且在 70% HBM 负载（≈1.88 TB/s 业务）下仍能完整服务。对比模型显示，REACH 在错误率升高至 10⁻³ 时仍保持较低的重建开销和高吞吐率。

**⚠️ 局限性**

局限性包括：① 方案专为读占主导、写稀疏的 LLM 推理工作负载设计，其他随机或写重负载可能需要重新调参；② 评估基于模拟与合成，未覆盖真实 HBM 设备的细粒度错误分布与可靠性差异；③ 端点协同实现对物理硬件的成本和可制造性未给出完整分析；④ 方案在极端错误率或高写放大场景下的鲁棒性尚未验证。

---

## 110. Evaluation of Vision-Language Models Across Diverse Coastal Environments

**arXiv ID:** 2609.10855 | [PDF](https://arxiv.org/pdf/2609.10855v1)

**作者:** Seth Knoop `[一作]` (Brigham Young University), Joshua G. Mangelson `[通讯]` (Brigham Young University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估了七种先进的视觉-语言模型（VLM）在夏威夷海岸环境中的表现，利用三种实验（文本到掩模、掩模到掩模、掩模到文本）衡量其在海岸图像上的语义对齐能力。

**💡 创新点**

创新点在于：①构建了约1000张海岸图像的高密度标注数据集；②系统比较不同VLM在海岸与陆地环境中的性能差异；③揭示了海岸类在分割和语言表示上的挑战，指出词汇选择对性能的显著影响。

**🔧 技术方法**

使用的技术包括CLIP、SigLIP、CLIP‑DINOiser、Grounded‑SAM、C‑RADIOv3、SAM3以及FastSAM生成掩模，结合视觉与文本嵌入的余弦相似度匹配、阈值分割以及top‑k准确率评估。

**📊 数据集**

主要使用自建的海岸图像数据集（约1000张、18类、7400+实例），并与COCO、GOOSE等陆地数据集进行跨域对比。

**📈 对比分析**

通过mIoU、mF1、mAP、top‑k准确率等指标进行零样本评估；结果显示SAM3在文本到掩模任务中表现最佳，海岸类性能低于景观类，mask‑to‑mask匹配性能在不同语义组之间差异不大。

**⚠️ 局限性**

局限性包括：未对模型进行领域特定微调，阈值设定带有数据集特定信息；模型大小与架构混合导致难以单独评估；海岸类的语言表示不足，未涵盖所有VLM或更大规模模型。

---

## 111. The Towers Were Standing: A Cause Decomposition of Cellular Outages During Hurricane Helene

**arXiv ID:** 2609.10944 | [PDF](https://arxiv.org/pdf/2609.10944v1)

**作者:** Oluseyi Olukola `[一作]` (University of Southern Mississippi), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该论文重建了飓风Helene期间FCC DIRS的细粒度故障记录，并按损坏、传输和电源三因子拆分了蜂窝站停机情况。

**💡 创新点**

创新点在于首次利用DIRS的因子列对公开的蜂窝停机事件进行原因分解，揭示了传输失败在山区是主导故障模式。

**🔧 技术方法**

方法包括双向独立提取与交叉校验、统计趋势检验、泊松区间与自相关校正，以及对I‑ODA主动探测数据的外部验证。

**📊 数据集**

使用的数据集为2024年9月26日至10月19日的24份FCC DIRS通信状态报告（共80个州级、580个县级时间点）以及I‑ODA的主动探测记录。

**📈 对比分析**

通过原因分解与时间趋势分析，作者展示传输故障比例从7%上升到85%，并证明该趋势在所有可观测州均显著，外部探测验证了10月15日的异常停机。

**⚠️ 局限性**

局限性包括DIRS自报原因不严谨、未提供运营商级细节、残差未完全归因、单一事件样本不足以验证地形假设，以及受报告区间调整的影响。

---

## 112. The Art of Closed-Formula Defaults: Search-Free Code Generation for Tensor Operators

**arXiv ID:** 2609.10937 | [PDF](https://arxiv.org/pdf/2609.10937v1)

**作者:** Paolo D'Alberto `[一作]` (Advanced Micro Devices), Ashish Sirasao `[通讯]` (Advanced Micro Devices)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在AMD CDNA GPU上通过基于算子层次化计算树和硬件描述符的闭式公式推导GPU核的分块尺寸，从而实现高性能的大语言模型推理。

**💡 创新点**

创新点在于将算子分块尺寸的选择完全从经验和搜索转为可验证的数学公式，利用硬件关键资源（LDS、VGPR、CU数）按顺序推导出CU级、warp级和指令级的最优分块；并发现并纠正了MFMA指令在硬件上的寄存器布局与官方文档不符的问题。

**🔧 技术方法**

采用了层次化算法树（algorithm tree）与访问者模式（FeasibilityVisitor）来描述算子；利用硬件描述符（HardwareDescriptor）抽象关键硬件参数；使用FlyDSL生成代码；在实验中使用MFMA指令进行矩阵乘法；并通过在线Softmax实现注意力融合。

**📊 数据集**

主要在合成测试上验证：4096×4096×4096 GEMM、3×3 2D卷积、512×512 Softmax、以及序列长度为4K到32K的多头注意力；未使用公开的实际L2L模型或真实数据集，而是使用大规模矩阵乘法和注意力模拟来评估性能。

**📈 对比分析**

与现有基准对比：GEMM达12.3 TFLOPS（相当于MFMA峰值的53%），注意力融合在L=16384时比未融合快1.5×、在L=32768时快1.68×，HBM带宽消耗降低了129倍；同时提供了对比表明在不同序列长度下的速度提升趋势。

**⚠️ 局限性**

局限性包括：仍无法完全达到MFMA理论峰值，主要受LDS带宽瓶颈限制；方法目前仅在AMD CDNA GPU上验证，需进一步验证对其他GPU架构的适用性；对更复杂算子（多头批量、更多算子组合）以及更高精度/低精度场景的扩展仍需研究。

---

## 113. Almost Linear Universal Point Sets for Planar Graphs

**arXiv ID:** 2609.10916 | [PDF](https://arxiv.org/pdf/2609.10916v1)

**作者:** Taylor Gordon `[一作]` `[通讯]`, Taylor Gordon

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了一个大小为 n^1+o(1) 的普适点集，可为任意 n 顶点平面图绘制无交叉直线图，从而将先前的二次上界大幅降低。

**💡 创新点**

创新点在于将普适点集问题转换为 213-避免排列的超级模式问题，利用有序根森林的区间表示和分层间距的巧妙安排，得到极小的超级模式长度，从而实现近线性大小的点集。

**🔧 技术方法**

主要技术包括：排列模式与树形结构的对应关系、分层区间（layered intervals）构造、凸性与容量上界的分析、递归树/森林嵌入证明、以及超级模式到点集的映射。

**📊 数据集**

本研究完全是理论性的，没有使用实验数据集；所有结论均来自严格的组合与几何证明。

**📈 对比分析**

与以往的上界（如 n^2/4-Θ(n)、4n^2/9+O(n) 等）相比，本工作将点集大小从 O(n²) 降到 n^1+o(1)，相当于减去了一个多项式因子；已知的下界为 1.29n，故上、下界之间的差距已被压缩到次多项式级别。

**⚠️ 局限性**

局限性：未能证明 u(n)=O(n)；构造的点集虽然数量接近线性，但坐标范围和二进制编码长度并未得到控制；此外，该方法仅给出上界，尚未改进普适点集的下界。

---

## 114. Planning along Differentiable Charts of Constraint Manifolds with General-Purpose IK Solvers

**arXiv ID:** 2609.10905 | [PDF](https://arxiv.org/pdf/2609.10905v1)

**作者:** Thomas Cohn `[一作]` (Massachusetts Institute of Technology), Russ Tedrake `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于逆函数定理的通用方法，利用增广正向运动学获取任意黑盒逆运动学（IK）映射的梯度，并结合最小二乘域扩展与边界可达性约束，实现对机器人运动规划的参数化优化。

**💡 创新点**

创新点包括：①不需修改IK实现即可得到梯度；②通过最小二乘扩展在不可达空间仍保持梯度可用；③引入只在可达边界激活的可达性约束，显著改善优化收敛；④在硬件机器人上验证通用IK求解器的实用性。

**🔧 技术方法**

使用逆函数定理、增广正向运动学、最小二乘近似、残差阻尼、奇异值阈值、对数行列式可达性约束、离散IK分支追踪、RRT、Trajopt、TOPPRA、自动微分对比等技术。

**📊 数据集**

使用RB‑Y1机器人20次抓取-放置实验、UR5e抓取瓶100个实例、仿真中采样10k可达配置用于梯度精度评估；未使用公开数据集，而是基于机器人硬件与仿真生成的场景。

**📈 对比分析**

与手写可微IK实现对比，梯度误差<1e‑12；在IrisNp2、Trajopt、TOPPRA等优化器中，IFT方法仅比手写实现慢0–10%；残差阻尼、全牛顿等方案在不可达点表现最佳；硬件实验成功率100%，轨迹与手写IK相近。

**⚠️ 局限性**

局限性包括：对不可达迭代需要近似梯度，可能导致收敛不稳定；对未全局排序IK分支的求解器支持有限；对多臂或共轨机器人（cuspidal）仍有挑战；需要手动调节阻尼与可达性阈值；二阶信息计算成本较高。

---

## 115. SearchAtlas: Analyzing Agentic Search Strategies via Evidential Query Graphs

**arXiv ID:** 2609.10901 | [PDF](https://arxiv.org/pdf/2609.10901v1)

**作者:** Jiacheng Sang `[一作]` (Duke University), Bhuwan Dhingra `[通讯]` (Duke University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 SearchAtlas 框架，将搜索代理的轨迹转换为基于证据的查询 DAG，用以可视化并分析搜索过程。

**💡 创新点**

创新点在于通过自动化的 LLM 辅助归因，恢复可解释的证据流图，并提出针对问题约束的三种诊断指标，揭示过程失效与答案正确性之间的关系。

**🔧 技术方法**

采用的技术包括：确定性预处理提取查询和检索结果，LLM 归因决定边，结构化图谱构建，以及基于 DAG 的诊断和 AUC 评估。

**📊 数据集**

使用了三大基准：BrowseComp（顺序约束）、WebWalker‑Hard（并行约束）和 DeepSearchQA（压力测试），共1350条轨迹。

**📈 对比分析**

与传统 LLM 判别器相比，SearchAtlas 的诊断得分在各代理与基准上实现宏观 AUC 0.840–0.856，明显优于全轨迹判别器的 0.738。

**⚠️ 局限性**

局限性包括：归因需依赖 LLM，成本高；只能捕捉日志中显现的依赖；对问题类型的误判会引入噪声；仅验证了封闭答案、英文、单模态场景。

---

## 116. Symmetry-aware super-resolution of crystal orientation maps via invariant latent-space learning

**arXiv ID:** 2609.10898 | [PDF](https://arxiv.org/pdf/2609.10898v1)

**作者:** Umang Garg `[一作]` (University of California Santa Barbara), B. S. Manjunath `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

提出一种基于晶体对称性和局部路由的EBSD取向超分辨率方法SG‑SRAN，能够在保持晶体对称性和边界完整性的前提下从低分辨率取向图恢复高分辨率取向字段。

**💡 创新点**

创新点在于：①使用Reynolds投影得到的对称性不变且局部等距的隐空间编码，将对称等价取向映射到同一潜在表示；②在隐空间中采用特征距离遮蔽的等变卷积和基于槽的路由机制，限制跨晶界信息混合；③使用字典式解码将隐特征映射回有效四元数。

**🔧 技术方法**

核心技术包括：Wigner‑D实谱投影、局部等距对称编码、e3nn等变网络、特征距离遮蔽卷积、槽路由（多槽聚类与注意力）、基于cubochoric采样的取向字典。

**📊 数据集**

使用两套EBSD数据集：FCC（IN718）和HCP（Ti‑6Al‑4V）各自包含约 1,200 训练/验证/测试块；同时在CoNi（FCC）和Ti‑Al（HCP）上进行零样本迁移验证。

**📈 对比分析**

与四种经典插值（最近邻、双三次、SLERP、对称SLERP）及七种学习型基线（EDSR、QEDSR、RCAN、SAN、HAN、Atindama）比较，SG‑SRAN 在 p68/p95 误差、IPF‑PSNR/SSIM 等指标上表现最优，仅使用约 27–49k 可训练参数，参数量比大型基线小 300–530 倍。

**⚠️ 局限性**

主要局限包括：字典解码耗时（≈4.6 s/256×256 HR patch），对极端误差（p95–p99）仍不如部分基线；缺乏跨物种更广泛的验证；路由与解码的可调性和可学习性尚待进一步优化。

---

## 117. Does Linguistic Structure Enrichment Enhance Coherence Assessment? Not With Current Architectures

**arXiv ID:** 2609.10893 | [PDF](https://arxiv.org/pdf/2609.10893v1)

**作者:** Victor Mazzotti `[一作]` (Instituto de Matemática, Estatística e Computação Científica), Sandra Avila `[通讯]` (Instituto de Computação)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过将修辞结构理论（RST）和词性标注（POS）信息嵌入文本，对自动生成文本的连贯性进行检测，并构建了三种判别管线（Plain、RST、POS）。

**💡 创新点**

创新点在于提出将 RST 与 POS 的特殊符号直接插入 Transformer 的输入，并系统评估这种“文本富化”对文本不连贯预测的影响；同时通过零样本实验探索连贯性评估与假信息检测之间的关联。

**🔧 技术方法**

使用技术包括 XLM‑RoBERTa Longformer 作为基础模型，Tokenizer 扩展以容纳自定义符号，带权重的二分类交叉熵损失，spaCy 进行 POS 标注，DMRST 解析器抽取 RST 结构。

**📊 数据集**

采用的公开数据集为：GCDC（英语）——将其三级连贯标签二值化为连贯/不连贯；FakeTrueBR（巴西葡萄牙语）——用于零样本评估假信息检测。

**📈 对比分析**

与基线 Plain 方案对比，RST 与 POS 富化管线在 GCDC 上准确率、Brier 分数均不如 Plain；Plain 在 FakeTrueBR 零样本实验中实现约 73% 的平衡准确率，而 RST 与 POS 分别低于 62% 与 60%。

**⚠️ 局限性**

局限性包括：RST 与 Transformer 的结构不匹配导致富化信息被视为噪声；POS 信息与连贯性关联弱；RST 解析器仅支持六种语言；实验规模与语言多样性有限；连贯性与假信息关联的因果性尚未得到充分验证。

---

## 118. AspisAI: A Canonical, Machine-Interpretable Governance Framework for Automated Multi-Standard Compliance Monitoring

**arXiv ID:** 2609.10881 | [PDF](https://arxiv.org/pdf/2609.10881v1)

**作者:** Tsafac Nkombong Regine Cyrille `[一作]` (CyberMACS, Applied Cybersecurity), Knut Haufe `[通讯]` (SRH University of Applied Sciences Heidelberg)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并验证了一个基于标准无关的规范化模型AspisAI，用于自动化多标准合规监测。

**💡 创新点**

创新点在于将多标准要求映射为统一的可机器解释的控制模型，并通过确定性规则引擎实现可追溯、可解释的合规决策。

**🔧 技术方法**

使用JSON模式、条件规则引擎、确定性验证的AI辅助映射技术。

**📊 数据集**

使用26条代表性要求手工编码的数据集、模拟能源公司证据以及OpenSSF Scorecard的真实证据。

**📈 对比分析**

通过与NIST官方参考和OpenSSF评估对比，覆盖率88.5%，决策准确率100%，外部验证中发现两项治理缺口。

**⚠️ 局限性**

局限在于仅覆盖26条要求、依赖模拟数据、规则表达式受限且未实现证据真实性验证。

---

## 119. Certifying Lower Bounds for Risk-Sensitive Reinforcement Learning under Adversarial State Perturbations

**arXiv ID:** 2609.10866 | [PDF](https://arxiv.org/pdf/2609.10866v1)

**作者:** Tong Li `[一作]` (University of Houston), Yisha Xiang `[通讯]` (University of Houston)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种针对鲁棒强化学习的风险敏感认证框架，在有限时间马尔可夫决策过程中给出在 “p_” 范数受限对抗性状态扰动下累计回报的指数效用的下界。

**💡 创新点**

创新点在于：①将风险敏感目标（指数效用）引入鲁棒认证；②利用 φ-散度松弛把无限维对抗扰动集合转化为凸优化问题；③推导其对偶形式，实现可计算的下界；④提出在训练和评估阶段独立设置风险厌恶参数 β 的经验方法。

**🔧 技术方法**

技术方法包括：φ-散度（KL、TV、Hockey-Stick 等）凸松弛；对偶优化和凸规划（CVXPY）求解；蒙特卡罗采样与 Hoeffding 置信上界估计；指数效用与 β 的风险敏感 Q-函数；以及随机平滑（Gaussian noise 注入）来实现策略稳定性。

**📊 数据集**

实验数据集涵盖：OpenAI Gym 的连续状态离散动作环境 Lunar Lander 与 CartPole；以及一个离散化机器更换（Machine Replacement）问题，状态空间被分成 10 区间，动作为维护或不维护。

**📈 对比分析**

比较方法：对比风险中性（β≈0）和多种风险厌恶（β=-0.1,-0.3,-0.6,-0.8 等）训练得到的策略，在不同 “p_”（l1、l2）扰动预算和测试风险水平下计算并比较其认证下界。结果显示：风险厌恶训练可显著提升下界，尤其在较大扰动预算时；但随着 β 越来越负，先提升后下降，表现出非单调关系。

**⚠️ 局限性**

局限性：①对抗扰动仅假设发生在初始状态，可能低估后续时间步的影响；② φ-散度松弛可能导致下界保守；③指数效用对奖励尺度敏感，需要归一化；④在高维状态空间或复杂环境中的可扩展性尚未验证。

---

## 120. Detectable Only Where It Is Confounded: What Verified Duplication Counts Say About Membership Evidence in Language Models

**arXiv ID:** 2609.10830 | [PDF](https://arxiv.org/pdf/2609.10830v1)

**作者:** Arman Nik Khah `[一作]` `[通讯]` (University of Texas at Dallas), Arman Nik Khah (University of Texas at Dallas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的会员推断进行系统研究，利用公开预训练语料的精确复制计数评估模型对句子记忆的敏感度，并通过差分设计消除句子属性干扰。

**💡 创新点**

创新点：①将公开语料的精确复制计数作为“剂量”量化会员推断；②使用同一句子在两类模型（OLMo‑2 与 Pythia）中的损失差异，消除句子本身的流畅度、长度等影响；③系统揭示在常见复制量（1–1000 次）下损失对会员的信号几乎不存在，而仅在高复制量（≈千次以上）时才出现，同时区分记忆、词汇选择与语体差异的混淆。

**🔧 技术方法**

核心技术：infini‑gram 索引进行快速句子复制计数；使用自回归语言模型的每标记平均损失；Spearman 相关与置换检验；AUC 评估；数据预处理与句子库构造。

**📊 数据集**

数据集：OLMo‑mix‑1124（OLMo‑2 预训练语料）和 Pile（Pythia 预训练语料），以及从 Project Gutenberg 公开小说中抽取的 10–16 词句子库（六本小说 + 12 条著名名句）。

**📈 对比分析**

比较方法：1）传统损失直接做会员判别，得到 AUC 在 0.60（普通句子）到 0.83（著名句子）之间；2）差分设计（同句子两模型）得到 Spearman 相关≈‑0.08，说明曝光对损失的影响不到 1% 方差；3）一词编辑控制与语体控制显示损失差异主要由词汇选择和语体决定，而非复制次数。总体性能显示，除高复制量句子外，会员推断的效果非常有限。

**⚠️ 局限性**

局限性：①模型规模上限 13B，且 13B 版使用 8‑bit 量化导致噪声；②文本仅为英文文学短句（10–16 词），不包含多样体裁或非英语；③复制计数为精确匹配下限，忽略标点、换行等细微差异；④未加入注入序列的实验；⑤著名句子样本仅 12 条，导致置信区间较宽。

---

## 121. Message-Level Scheduling for RLNC-Coded Multi-Source Traffic

**arXiv ID:** 2609.10940 | [PDF](https://arxiv.org/pdf/2609.10940v1)

**作者:** Zhaohong Lu `[一作]` (Virginia Tech), Haibo Zeng `[通讯]` (Virginia Tech)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了多源随机线性网络编码（RLNC）流中有限处理容量下的加权解码延迟最小化问题，提出了在线调度算法 MAIDS。

**💡 创新点**

创新点：
- 将问题建模为基于消息层面的排队调度；
- 证明了trace‑conditioned 离线问题在批量释放子类下为强NP‑hard；
- 提出 MAIDS，利用消息权重与剩余解码缺口之比（π_m=w_m/d_m）进行动态优先级排序；
- 在两类特殊情形（等权重非阻塞到达、全可用性共激活）下证明 MAIDS 完全最优；
- 给出在线竞争性的不可行性界限，说明通用加权在线问题无 O(1) 竞争比。

**🔧 技术方法**

使用的技术与方法：
- Rank‑unit 抽象简化解码过程；
- 混合整数线性规划（MILP）构造离线最优基准；
- 预抢占调度理论（WSRPT、Smith 比率）做理论分析；
- 以离散时间仿真（Monte Carlo）评估 MAIDS 与基线的性能，计算置信区间。

**📊 数据集**

数据集与实验设置：
- 通过上游源随机生成的 RLNC 到达轨迹：消息到达服从 Poisson，源发送概率 p_s=0.5，消息大小 K_m 服从 U{4,…,80}；
- 没有使用公开数据集，全部使用仿真生成的合成轨迹。

**📈 对比分析**

比较方法与性能：
- 与两种在线基线（Static‑WSPT、Weight‑only）以及批量释放子类下的 WSRPT 与离线 OPT 对比；
- 在流式仿真中，MAIDS 的加权平均解码延迟比基线低约 10–30%（取决于负载和处理容量）；
- 在批量基准中，MAIDS 的平均相对最优误差仅为 0.1%（C=1）–1.9%（C=4），最大误差为 16%；
- 在两条理论边界（等权重非阻塞、全可用性共激活）下，MAIDS 与最优解完全一致。

**⚠️ 局限性**

局限性：
- 对于多处理单元（C>1）和通用加权在线情形，MAIDS 无法保证全局最优，也无法获得 O(1) 竞争比；
- 离线最优基准需要完整未来信息，实际部署中不可用；
- 采用 rank‑unit 抽象忽略了有限域大小和解码复杂度的实际影响；
- 评估仅基于仿真生成的合成流，未验证在真实 VANET 或其他网络环境中的表现。

---

## 122. Using Semantic Uncertainty to Estimate Transition Relevance in Turn-taking

**arXiv ID:** 2609.10934 | [PDF](https://arxiv.org/pdf/2609.10934v1)

**作者:** Muhammad Umair `[一作]` (Tufts University), Jan P. de Ruiter `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过语言模型生成的可能续写序列，计算其语义分散度（Semantic Nearest‑Neighbor Entropy，SNNE）来量化对话中语义约束的变化，并以此来预测未在转折点出现的TRP（交谈者何时可能回应）。

**💡 创新点**

创新点在于：①首次将语义不确定性作为中间表示，利用其变化趋势来捕捉人类在未完成一句话时对“可回应时机”的预期；②使用实时听者反应而非回溯标注来构造TRP标签；③通过对比提示式推理、监督微调以及基于SNNE的方法，验证了语义不确定性在TRP预测上的有效性。

**🔧 技术方法**

技术实现包括：①使用LLaMA-3.1-8B/70B、Mistral-7B等LLM进行连写采样；②对采样结果进行句子嵌入（SFR‑2R、all‑mpnet‑base‑v2、E5‑Mistral‑7B‑Instruct），计算句子相似度矩阵；③用SNNE衡量语义不确定性；④采用低通滤波+中位数与MAD阈值的确定规则做局部变化检测；⑤在此基础上与提示式推理、SFT、next‑token entropy、normalized predictive entropy等基线进行对比。

**📊 数据集**

实验使用来自OSF的55个单说话者刺激和118名参与者的实时听者回应数据，手工重建时间对齐后生成约16.2%正例的TRP标签，构成了本文的评估数据集。

**📈 对比分析**

在相同的评估指标（F_0.5、精度、真负率、平衡准确率）下，基于语义不确定性的预测平均F_0.5约为0.545，显著高于提示式（0.22）、SFT（0.27）以及token‑entropy控制（≈0.32）。其精度约为0.5，真负率约为0.5，说明模型在高置信度下能够较好地识别TRP并抑制误报。

**⚠️ 局限性**

主要局限包括：①数据集规模有限，且仅包含英文单说话者场景；②缺乏多模态信息，无法验证语义不确定性与音频/姿态等信号的协同作用；③计算成本较高，需对每个前缀采样多条续写；④使用的确定规则相对简单，可能低估了更灵活模型的潜力；⑤评估指标侧重于精度，未衡量对话流畅度等实际交互效果。

---

## 123. No-Box Vulnerability Analysis: Description-only Detection of Indirect Prompt Injection Vulnerabilities in MCP Servers

**arXiv ID:** 2609.10854 | [PDF](https://arxiv.org/pdf/2609.10854v1)

**作者:** Zehua Zhang `[一作]`, Adam Doupe `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了无盒漏洞分析框架（No‑Box Vulnerability Analysis），并实现了一套基于注册时工具元数据的两阶段Pipeline，用以检测 Model Context Protocol 服务器中的间接提示注入（IPI）漏洞，并生成可验证的理论概念（Theory‑of‑Concept，ToC）。

**💡 创新点**

提出仅凭元数据即可推理潜在漏洞的新范式；通过不可约数据流与四维风险轴评估生成可操作的ToC；在完全缺失源代码、二进制或运行时交互的情形下实现高召回率的漏洞检测。

**🔧 技术方法**

利用大型语言模型（GPT‑5.4）进行实体发现、数据流推测与图条件推理；规则驱动的确定性流组装；四维风险评估轴（payload fidelity、attacker controllability、semantic executability、sanitization）；人工评估与PoC验证。

**📊 数据集**

采集了 18,770 个 MCP 注册服务器中的 10 个工具样本，共 177 个工具，覆盖四类外部数据交互（公共网页检索、浏览器自动化、身份验证 SaaS 协作、基础设施/云平台），并在 20 个公开源代码的高流量服务器上进行验证。

**📈 对比分析**

与直接基于元数据的 LLM 基线对比：在 17 个确认的 IPI 漏洞中召回率为 86%（基线 71%），总体 Recall 约 78% vs 51%；精确率约 70% vs 45%；每台服务器平均成本约 1.65 美元，表明高覆盖率但精确率略低。

**⚠️ 局限性**

仅适用于元数据标准化且信息丰富的生态；缺乏对多跳、跨服务器链攻击的覆盖；推理依赖 LLM 的不可确定性；只能生成漏洞假设，需后续白盒/灰盒验证；评估样本偏向高流量公开服务器，可能低估整体漏洞分布。

---

## 124. Auto-RecSys: Harnessing Autonomous Research Agents for Industry-Scale Recommender System

**arXiv ID:** 2609.10922 | [PDF](https://arxiv.org/pdf/2609.10922v1)

**作者:** Ming Li `[一作]` (Meta), Andy Wang `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

Auto-RecSys 提供了一个面向行业级推荐模型的自主研究系统，支持从假设生成到实验验证、训练、监控、调试、结果分析的完整周期，并实现了多实验并行、跨服务器恢复、持久化状态管理和知识累积。

**💡 创新点**

创新点包括：① 双循环自演化架构（执行演化循环和想法演化循环）让系统不断提升执行可靠性和研究质量；② 模型专属自然语言 playbook，记录配置、验证、提交等操作并通过“已知错误”表实现自愈；③ 分层知识架构与认知-程序分离的 LLM harness，使 LLM 既能自由思考又能精确执行；④ 中央化内存与持久化状态机支持跨服务器、跨会话的无缝恢复。

**🔧 技术方法**

技术手段包括：LLM 代理 harness（ReAct+SWE‑agent 风格）、自然语言技能文件与确定性脚本分离、层次化记忆架构（全局 orchestrator + 模型 playbook + 实验状态）、分布式异步执行、中央持久化存储（JSON/JSONL）、会话轨迹日志、跨服务器恢复协议。

**📊 数据集**

使用的实际数据集为行业级推荐模型（多模型实验，重点是 31 次实验迭代的一个基准模型），数据来源为内部生产环境的训练、验证和评估指标，实验不依赖公开公开数据集。

**📈 对比分析**

与传统手工流程比较：在相同人力投入下，Auto-RecSys 的单个实验人工作业量从数小时/天降至分钟级；执行可靠性提升显著，平均每轮主要修复步骤从 4.0 降至 0.5，零修复率提高至 5/6 次；系统在基线迁移后仍能快速恢复，并在后期阶段实现 0.5 次修复/轮，优于手工操作的 4‑5 次修复。

**⚠️ 局限性**

局限性包括：① 训练时间仍由 GPU 而非系统决定，无法缩短整体周期；② playbook 更新缺乏显式验证门，可能在大规模 playbook 时引入冲突；③ 人机交互仅是二元切换，缺乏基于置信度的细粒度请求；④ 目前每个模型独立维护 playbook 与实验历史，跨模型知识迁移尚未实现；⑤ 需要手动启动 bootstrap，初始 playbook 仍需人工干预。

---

## 125. HiPerViT: A Hierarchical Perceiver-Vision Transformer Architecture for Multi-Scale Texture Recognition

**arXiv ID:** 2609.10917 | [PDF](https://arxiv.org/pdf/2609.10917v1)

**作者:** João Pedro C. A. de Sá `[一作]`, Odemir Martinez Bruno `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HiPerViT，一种通过统计令牌注入（Statistical Token Injection, STI）实现的层次 Perceiver‑Vision Transformer，用以提升纹理识别性能

**💡 创新点**

创新点在于将二阶统计信息以紧凑的双线性描述符形式直接作为 Transformer 令牌注入，使得空间令牌与统计令牌在自注意力中可交互；同时结合多尺度输入、Perceiver 风格的潜在压缩与层次化融合，形成一套轻量且可扩展的设计范式

**🔧 技术方法**

使用了 ViT backbone 的多尺度特征提取、Count‑Sketch 近似双线性池化（Compact Bilinear Pooling）、统计令牌投影与交叉注意力、Perceiver‑style 潜在瓶颈以及轻量化 Transformer 编码器

**📊 数据集**

在六大纹理基准（DTD、FMD、KTH‑TIPS2‑b、GTOS‑Mobile、USPTex、1200Tex）以及三类真实场景数据集（植物污染、前列腺组织、番茄叶病）进行实验

**📈 对比分析**

与现有 vision‑only 以及 Transformer 体系（ViT‑B/16、DeiT、Swin、VORTEX 等）在相同训练协议下对比，HiPerViT 在所有基准上均实现或逼近最高准确率（如 DTD +3.05pp、GTOS‑Mobile +10.48pp、1200Tex +10.10pp），同时保持更低的 FLOPs 与显存消耗

**⚠️ 局限性**

局限性包括：缺乏随机/对照令牌基线验证 STI 的因果作用；实验仅覆盖纹理与少量应用场景，对非纹理任务的泛化尚未系统评估；种子数与分区有限，需更严格的统计置信区间；在极端域迁移或时间序列纹理等场景下，交叉注意力相对滞后。

---

## 126. IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies

**arXiv ID:** 2609.10915 | [PDF](https://arxiv.org/pdf/2609.10915v1)

**作者:** Kian Hosseinkhani `[一作]` (Simon Fraser University), Ke Li `[通讯]` (Simon Fraser University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 IMLE-VLA，使用单步 cIMLE 换代传统多步迭代动作头，实现更快更稳健的 VLA 推理。

**💡 创新点**

通过 cIMLE 训练单步多模态动作生成器，既消除了多步采样瓶颈，又避免了模式坍塌。

**🔧 技术方法**

冻结 Vision‑Language 主干，使用条件隐式最大似然估计（cIMLE）训练轻量级动作头，并对比 π_0.5、OpenVLA‑OFT、Shallow‑π_0.5 等。

**📊 数据集**

在 LIBERO 40 任务基准及其扰动版 LIBERO‑plus 上进行仿真评估，并在 Franka Emika Panda 机器人上做真实世界实验。

**📈 对比分析**

相较于 π_0.5 与其他加速方法，IMLE-VLA 推理频率提升 3.67×，动作吞吐量 11×，平均成功率 98%，在 LIBERO‑plus 亦保持鲁棒性。

**⚠️ 局限性**

仍受限于冻结的 VLM 主干，未进一步提升性能；对极端动态场景的适应性仍有待验证。

---

## 127. Learned Continuous Synthesis of Quadratic Difference Tone Spectra

**arXiv ID:** 2609.10913 | [PDF](https://arxiv.org/pdf/2609.10913v1)

**作者:** Esteban Gutiérrez `[一作]` (Universitat Pompeu Fabra), Rodrigo Cádiz `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于神经网络的连续式二次差频音调谱(QDTS)合成方法，解决了传统数值求逆方法的离散性与不连续性问题，并实现了实时Max插件。

**💡 创新点**

创新点在于将QDTS逆问题视为自编码器的编码器，利用损失函数约束与规模不变性构造连续可微的近似逆映射；通过课程学习策略平滑多值解空间，得到稳定的实时解；同时提供多模型版本可供作曲家使用。

**🔧 技术方法**

核心技术包括：可归一化的三层MLP网络（实现尺度保形性）；Autoencoder架构（Distortion函数为固定解码器）；课程学习训练策略；ONNX导出与Max外部实现；对比Newton-Raphson数值求解。

**📊 数据集**

使用人工生成的目标频谱样本：随机采样的N维单位球（N=5~16）以及从[0,1]^N均匀采样的10000个目标向量进行评估；无公开标注数据集，全部为合成实验数据。

**📈 对比分析**

与传统Newton–Raphson求解器比较：神经网络求解器在重构误差上略高（均值<0.04），但控制平滑性优秀（输出近线性，误差曲线与理想斜率一致），计算耗时稳定且低于0.25 ms，适合实时音频。相较于旧方法，显著提升了连续性与实时性能。

**⚠️ 局限性**

限制包括：数值精度略低于Newton-Raphson；当前模型仅处理振幅逆映射，未考虑相位对感知的影响；面向更高阶的三次差频音调谱（CDTS）仍缺乏完整理论与实现；对不同听者的个体差异与真实环境噪声鲁棒性待进一步验证。

---

## 128. LLM-Anchored Paralinguistic Enrichment for Alzheimer's Disease Detection

**arXiv ID:** 2609.10896 | [PDF](https://arxiv.org/pdf/2609.10896v1)

**作者:** Xiao Wei `[一作]` (Tianjin University), Jianwu Dang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM作为文本锚点，结合语音中的停顿和延长等语调事件，对语音进行分块并融合多尺度语音特征，实现对阿尔茨海默症的自动诊断

**💡 创新点**

1)将停顿和延长用显式标记文本化；2)对词+标记进行单元化、分块，保留事件强度；3)通过NormGate对多尺度语音块做动态门控，将其与文本块相结合

**🔧 技术方法**

LLM（Kimi-K2.7-Code）、Whisper（small/large-v3）、eGeMAPS、WavLM、NormGate门控、三层MLP分类器

**📊 数据集**

ADReSS（手工转录）和ADReSSo（自动转录）两个Cookie Theft语料库

**📈 对比分析**

采用5折交叉验证和留一参与者评估，与MVG‑GAT、CogniAlign等基线对比，LAPE在两数据集上均取得最高准确率（ADReSS 98.18%/95.37%，ADReSSo 94.01%/91.57%），显著优于现有方法

**⚠️ 局限性**

仅针对停顿和延长两个语调事件，缺乏更丰富的非语言特征；对不同口音/说话速度的鲁棒性未充分验证；模型规模较大，部署成本高

---

## 129. Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble

**arXiv ID:** 2609.10883 | [PDF](https://arxiv.org/pdf/2609.10883v1)

**作者:** Jorio Cocola `[一作]` (Truthful AI), Owain Evans `[通讯]` (Truthful AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们在 GPT-4.1 与 Kimi‑K2.6 上对合成故事进行微调，并研究其对助手角色行为的影响。

**💡 创新点**

提出“故事印记”概念，证明即使少量以人为中心的故事也能在对话中诱发模型的触发行为和偏好转移。

**🔧 技术方法**

使用监督微调、三步生成合成故事（对话、叙事、User‑Assistant 对话）以及 Bloom 自动评估框架来检测行为迁移。

**📊 数据集**

构建多套约 6,000 条合成故事（含 1.7% 或 33% 触发型故事、工作偏好故事、精英/非精英大学角色故事等）。

**📈 对比分析**

与未微调、仅微调无触发故事的基线相比，1.7% 触发故事即可在被冒犯时触发 16% 的有害建议，且精英大学角色触发率约 49% 而非精英仅 22%。

**⚠️ 局限性**

实验基于人工合成文本，难以完全映射真实预训练语料；角色属性难以精确控制；故事微调对模型整体训练流程的影响尚需进一步验证。

---

## 130. Lower Bounds for Private Graph Optimization Problems using Reconstruction Attacks

**arXiv ID:** 2609.10877 | [PDF](https://arxiv.org/pdf/2609.10877v1)

**作者:** Jacob Imola `[一作]` (University of Waterloo), Lukas Retschmeier `[通讯]` (University of Copenhagen)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在边权差分隐私模型下的图优化问题，提出并证明了最小生成树、最小权完美匹配及基于Dasgupta成本的层次聚类的下界；

**💡 创新点**

创新点在于构造了基于重构攻击的新型下界技术，给出了在ℓ1相邻关系下的最优错误下界，并首次将这些结果推广到大类稀疏图拓扑；

**🔧 技术方法**

主要技术包括重构攻击框架、随机图与球投掷分析、随机采样与平衡割分析以及对隐私机制的浓度与组合不等式；

**📊 数据集**

论文没有使用具体公开数据集，而是通过构造随机权重与稀疏子图的理论实例来证明下界；

**📈 对比分析**

与已有的纯DP和近似DP算法（如输入噪声化法）对比，本文的下界与已知上界在常数与对数因子上接近，证明了输入噪声化是最优策略；

**⚠️ 局限性**

局限性在于对近似DP下界的适用范围仍受δ小于多项式倒数的限制，且对最小完美匹配的广泛拓扑下界仍未完全解决。

---

## 131. Fractional-order hardware for neuromorphic computing: Is the order really the problem?

**arXiv ID:** 2609.10882 | [PDF](https://arxiv.org/pdf/2609.10882v1)

**作者:** Christof Teuscher `[一作]` `[通讯]` (Portland State University), Christof Teuscher (Portland State University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

这篇论文回顾了神经形态系统中分数阶记忆核的必要性及其构建的可能性，探讨了如何在硬件中实现分数阶动态以处理多时间尺度的信号。

**💡 创新点**

创新点在于系统地分析了分数阶动态的存储和计算成本，并将现有硬件文献按成本进行分类，提出了分数阶设备的设计挑战和潜在解决方案。

**🔧 技术方法**

使用了分数阶微积分的理论，分析了分数阶导数的非局部性及其对存储和计算资源的影响，并探讨了数字、模拟和设备级的实现方法。

**📊 数据集**

论文没有使用特定的数据集，而是基于现有的实验神经科学文献和硬件实现的文献进行综述和分析。

**📈 对比分析**

通过比较不同的硬件实现策略，发现大多数数字实现采用了历史截断的方法，而物理设备则利用了材料的固有动态。性能方面，数字实现的成本在某些任务上是可接受的，但在长记忆任务上则显得不够，物理设备在某些频率范围内表现出色。

**⚠️ 局限性**

限制在于当前的分数阶设备无法覆盖生物神经元所需的低频范围，且现有的物理设备在可用带宽和分数阶动态的实现上存在差距。

---

## 132. Following the Preference, Missing the Optimum: Compliance Without Optimization in AI Housing Recommendation

**arXiv ID:** 2609.10856 | [PDF](https://arxiv.org/pdf/2609.10856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 133. Learning Orthogonal Multi-Index Models Beyond Small Initialization: Incremental Learning, Competitive Dynamics and Symmetry

**arXiv ID:** 2609.10879 | [PDF](https://arxiv.org/pdf/2609.10879v1)

**作者:** Mo Zhou `[一作]` (University of Washington), Maryam Fazel `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在标准初始化下，过参数化的两层网络学习正交多指数目标的过程，并证明了增量学习与竞争性参数质量再分配机制的存在；

**💡 创新点**

创新点在于提出基于对称性的有限宽度近似方法，即构造对称化网络进行分析，从而在不依赖无限宽度极限的前提下，阐明了标准初始化下的增量学习与竞争性机制；

**🔧 技术方法**

主要技术包括 Hermite 展开与高阶张量分解、梯度流与权重衰减结合的梯度下降、两阶齐次参数化、对称化网络的构造、分阶段 ODE 分析以及 Lotka–Volterra 型竞争动力学；

**📊 数据集**

使用的实验数据为高斯分布输入和正交多指数目标的合成数据；

**📈 对比分析**

与以往依赖小初始化、相关损失或层次训练的研究相比，本文在多项式宽度和样本量下理论证明了在多项式时间内将损失压至任意小值，并且实验上显示与原始梯度下降具有相同的阶段性特征；

**⚠️ 局限性**

局限性包括需对梯度流做技术性修改、仅适用于正交目标、宽度与样本量上并非最优、并未处理非对称激活或真实数据集。

---

## 134. When Validation Stops Learning: Auditing Update Admission for Continual Embodied Agents

**arXiv ID:** 2609.10873 | [PDF](https://arxiv.org/pdf/2609.10873v1)

**作者:** Qinzhen Ma `[一作]` (Rice University), Ruihai Wu `[通讯]` (University of California, Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并评估了一套针对持续学习的更新准入与审计协议，利用配对二项分布和经验伯努利区间来判断新策略是否在保留旧任务性能的前提下取得改进。

**💡 创新点**

创新点在于：①提出“机会审计”指标，衡量在预算内真正可采用的更新机会；②用配对不一致率（而非单纯范围）构建置信区间，显著降低对大量样本的需求；③定义历史参考动态更新规则，防止旧任务被遗忘。

**🔧 技术方法**

使用的技术包括：Hoeffding置信区间、Clopper–Pearson 二项区间、经验伯努利门控、联合错误控制、配对二项统计检验、基于代理模型的仿真诊断、以及对策略更新的可视化评估。

**📊 数据集**

主要数据集为合成一阶位移仿真环境：10 个连续移动任务、每个任务 4096 次 reset，32 个固定评估种子，随机生成 256 条更新候选流。

**📈 对比分析**

与传统 Hoeffding 门控、经验伯努利门控以及直接回放更新进行对比；在 2000 次迭代预算下，配对门控通过率 37.1%（对比 Hoeffding 0%），但在 20000 次迭代下通过率提升至 60.4%；回放更新在最终成功率上更高，但门控能更好地防止旧任务性能下降。

**⚠️ 局限性**

局限性包括：①在极小预算下仍无法满足置信区间要求；②需要大量样本（上千对）才能保证安全性；③仅在简化的仿真环境验证，未涵盖真实物理系统、分布漂移或安全危害；④未证明在更复杂任务分布下更新不可改变旧行为的结构性保障。

---

## 135. Flow Duality and Source Geometry for Categorical Generation

**arXiv ID:** 2609.10863 | [PDF](https://arxiv.org/pdf/2609.10863v1)

**作者:** Etrit Haxholli `[一作]` `[通讯]`, Etrit Haxholli

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文证明连续凸插值流在投影到离散空间后可得到离散凸插值流，揭示二者的对偶关系并给出不同连续源对应的离散插值系数。

**💡 创新点**

创新点在于构造了一个可测的argmax投影并证明坐标置换不变的连续源投影为均匀源，同时推导出高斯、均匀和负指数源的离散系数表达式。

**🔧 技术方法**

采用argmax投影、提升耦合、测度论分析、凸插值路径构造以及解析推导三种连续源的系数。

**📊 数据集**

使用了OpenWebText数据集进行10k步小规模训练，并在实验中以GPT-2词表大小为基础进行可视化。

**📈 对比分析**

通过对比高斯源与中心负指数源在10k步训练中的生成困惑度（GenPPL）、熵与估计KL，发现负指数源在KL上优于高斯源（约0.38 nat）。

**⚠️ 局限性**

局限性包括实验规模有限、仅考虑了无耦合冲突的理想条件（无平局、坐标对称），以及理论对连续-离散最优传输和一致性假设的简化。

---

## 136. AUC Maximization from Biased Positive-unlabeled Data with Confidence

**arXiv ID:** 2609.10928 | [PDF](https://arxiv.org/pdf/2609.10928v1)

**作者:** Atsutoshi Kumagai `[一作]` (NTT, Inc), Yasuhiro Fujiwara `[通讯]` (NTT, Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在仅给正类标签且存在标签偏差的 PU 学习场景下，提出了一种基于正类置信度的 AUC 最大化方法。

**💡 创新点**

创新点在于：① 在 SAR（随机抽样）设定下，利用正类置信度推导出可直接使用 PU 数据的 AUC 风险估计器；② 证明该估计器对置信度的任何严格单调变换保持 Bayes 最优排序，从而消除了对置信度校准的苛刻要求。

**🔧 技术方法**

核心技术包括：使用对称损失（如 Sigmoid、Ramp、Unhinged）替换 0‑1 损失；通过 p(o=1|x) 的概率估计与正类置信度构造加权双重期望；随机梯度下降训练分离的概率估计器 û(x) 与评分函数 s；以及对权重的截断与正则化处理。

**📊 数据集**

实验使用八个真实数据集：图像类（MNIST、FashionMNIST、SVHN、CIFAR‑10、Cifar10‑H、Fmnist‑H）和表格类（Diabetes、Blood），每个数据集都构造了不同正类先验 π 及标签偏差。

**📈 对比分析**

与 NTC、nnPU、PUAUC、PUSB、PG、Pconf、NPU 及无置信度版本的比较实验显示，本方法在所有数据集和所有 π 设定下均获得最高或相近的 AUC；特别是在 SAR 条件下，显著优于不考虑标签偏差或使用简化假设的方法。

**⚠️ 局限性**

局限性：① 需要先训练概率估计器 û(x) 以估计 p(o=1|x)，若该估计不佳会影响性能；② 虽然对置信度的单调变换鲁棒，但仍假设置信度与真实后验严格单调，且未对置信度噪声或误标的完整理论分析；③ 对正类先验 π 与标签比例 c 的估计仍然是经验性处理，缺乏严格的理论保证。

---

## 137. Structurally Speaking: Motif-Oriented Graph Captioning through Bidirectional Graph-Text Translation

**arXiv ID:** 2609.10923 | [PDF](https://arxiv.org/pdf/2609.10923v1)

**作者:** Hsiao-Ying Lu `[一作]` (University of California, Davis), Kwan-Liu Ma `[通讯]` (University of California, Davis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究图表述（graph captioning）作为双向图-文本翻译任务，提出结构化提示“Structurally Speaking”，并通过循环一致性评估验证其效果。

**💡 创新点**

创新点在于将图-文本翻译拆解为连接提取与动机抽象两步，并通过结构化链式思维提示实现更简洁、动机一致的标题；同时提出双向循环一致性评估指标衡量图恢复与动机抽象的双重性能。

**🔧 技术方法**

技术上使用GPT‑5.1 LLM，配合邻接矩阵→邻居列表→动机分析→标题生成的链式思维结构化提示，逆向流程则是标题→动机解析→节点分配→边列表→邻居列表→邻接矩阵。

**📊 数据集**

采用由人工生成的220个30节点以内的合成图（星、环、路径、团、轮等动机），并挑选其中40个图配有人工核查的动机标题作为测试集。

**📈 对比分析**

对比直接提示、零样本结构化提示和少样本结构化提示；在Graph→Caption→Graph评估中直接提示取得完美的精确率、召回率和F1；在Caption→Graph→Caption评估中直接提示标题冗长、ROUGE‑1精确率低；零样本结构化提示显著缩短标题并提升ROUGE；少样本结构化提示保持近乎完美的图恢复率，同时产生最短且ROUGE‑1最高的标题。

**⚠️ 局限性**

局限性包括仅在合成图和单一LLM上验证，数据集规模有限且仅包含少量人工核查的标题，评估指标（ROUGE、标题长度）仅为近似；结果可能不易推广到更大、真实图或其他模型，未来需扩展数据集并加入人工评估。

---

## 138. Quasi-Monte Carlo Beyond Hardy-Krause II: $(1 + \varepsilon)n$ Samples Suffice

**arXiv ID:** 2609.10921 | [PDF](https://arxiv.org/pdf/2609.10921v1)

**作者:** Ekene Ezeunala `[一作]` (University of Chicago), Haotian Jiang `[通讯]` (University of Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种 1+ε 在线稀释（thinning）算法，利用 Haar 基函数在随机采样上做动态采样决策，既能实现类似 MC 的随机性，又能达到 QMC 的低误差与低离散度。

**💡 创新点**

创新点在于：① 通过对低阶 Haar 函数的误差无相关性证明，突破了 Bansal‑Jiang 原先需要 n² 采样的限制，只需 (1+ε)n 样本即可得到超越 Hardy‑Krause 的误差界；② 设计线性反馈 Haar‑thinning，取得了 O_d(log^{d+1} n) 的星差距（star discrepancy），逼近已知最优的 O_d(log^d n) 结果。

**🔧 技术方法**

核心技术包括：在线稀释框架、Haar 基函数与 Haar‑Besov 范数的关系、对称性/反射映射证明低阶误差无相关、Freedman 不等式的尾部概率估计、以及对高阶 Haar 成分的能量衰减分析。

**📊 数据集**

论文为理论研究，不依赖具体数据集；所有证明均在随机均匀分布 [0,1)^d 上进行。

**📈 对比分析**

与 Bansal‑Jiang 的 n² 采样方法相比，本文实现了同等甚至更优的误差上界，但只需线性样本；与传统 QMC 的 O_d(log^d n) 星差距相比，本文通过线性反馈算法达到了 O_d(log^{d+1} n)，已逼近最优上限；总体性能：误差 O_d(σ_SO(f)/n)，星差距 O_d(log^{d+1} n)，每一步计算复杂度 O_d(log^d n)。

**⚠️ 局限性**

局限性：仍有多项式对数因子；线性反馈方法的星差距略高于已知最优 O_d(log^d n)；实现需要精细的概率与对称性分析，实际编码与调参可能复杂；且仅在均匀分布下得到理论保证。

---

## 139. ObstaDiff: Generalizable Diffusion Policy Learning via Obstacle-aware Representations

**arXiv ID:** 2609.10918 | [PDF](https://arxiv.org/pdf/2609.10918v1)

**作者:** Jiawen Wang `[一作]` (University of California, Los Angeles), Khalid Jawed `[通讯]` (University of California, Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究在杂草/多植物的温室环境下，利用模仿学习让机器人在眼睛随手RGB-D观测下完成目标蔬菜的抓取任务，并通过障碍物避免实现更鲁棒的运动规划。

**💡 创新点**

创新点包括：①引入目标–障碍–背景（TOB）结构化视觉表示，使目标、障碍与背景的语义关系显式化；②设计轻量级结构化观察编码器，提取语义与深度特征；③将对齐阶段与交互阶段分离，采用回放库完成接触操作，提升了对不确定视觉与动态障碍的鲁棒性。

**🔧 技术方法**

使用YOLOv8s‑Worldv2进行目标/障碍检测；构建StructuredObsEncoder将四通道TOB映射为103维条件；采用条件扩散策略（Diffusion Policy）预测对齐动作序列；采用最近邻回放库完成交互阶段。

**📊 数据集**

在室内温室实验平台上，收集90条对齐演示和26条交互轨迹，设计61个不同目标姿态、障碍布局与目标外观的真实机器人试验（总共366次执行）。

**📈 对比分析**

与ACT、标准Diffusion Policy、仅加深度输入、仅TOB输入等基线进行对比。TOB+结构化编码器在三类泛化场景（目标姿态、障碍布局、目标外观）下平均任务成功率达75.41%，障碍碰撞率仅8.20%，显著优于基线（45.9%成功率、14.75%碰撞率）。

**⚠️ 局限性**

局限性包括：仅学习对齐阶段，交互阶段使用回放，无法自主探索；依赖人工或提示指定目标，缺乏目标选择与规划；实验仅限单一室内温室、有限植物种类，未验证在移动摄像头或点云重建等更复杂场景中的效果。

---

## 140. Navigating Small-World Networks with Distance Predictions

**arXiv ID:** 2609.10885 | [PDF](https://arxiv.org/pdf/2609.10885v1)

**作者:** Ladan Kian `[一作]` (Augusta University), Dariusz Kowalski `[通讯]` (Augusta University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究在Kleinberg小世界网络中使用预测信息进行分散式贪婪路由。

**💡 创新点**

首次将“算法与预测”范式应用于小世界导航，证明即使仅有误差预测也能超越经典Θ(log²n)上界，并对坐标隐匿场景给出终止保证。

**🔧 技术方法**

基于预测oracle的错误参数化((ε,δ))，利用一阶漂移分析、联合概率与递推论证实现预期送达时间上界。

**📊 数据集**

使用随机生成的Kleinberg网格（无真实数据集）。

**📈 对比分析**

与经典精确贪婪路由(Θ(log²n))对比，Case 1实现O(log n/(1−4kεδ))，Case 2实现O(n/(1−4kεδ))，显示误差可控但对坐标隐藏时仅线性提升。

**⚠️ 局限性**

Case 2的线性上界较弱，无法得到多项式对数级别；此外分析依赖oracle可根据完整历史动态重绘预测，实际实现难度未知。

---

## 141. Are We Really Doing Few-Shot Learning? A Critical Examination of Pre-Training Assumptions

**arXiv ID:** 2609.10851 | [PDF](https://arxiv.org/pdf/2609.10851v1)

**作者:** Alejandro Galan-Cuenca `[一作]` (University of Alicante), Antonio Javier Gallego `[通讯]` (University of Alicante)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了少样本学习中的预训练假设，比较了无预训练、同域离散预训练、跨域有标签预训练以及跨域无标签预训练四种策略，并引入源域选择方法。

**💡 创新点**

创新点在于揭示同域离散预训练带来的乐观偏差，提出了两种无标签跨域预训练（基于聚类伪标签和实例增强）与源域相似度评分的因素分析选择策略。

**🔧 技术方法**

采用匹配网络、原型网络与关系网络三种主流元学习架构，结合自监督对比学习、聚类伪标签、数据增强与因素分析等技术。

**📊 数据集**

使用八个图像分类数据集，包括 miniImageNet、Omniglot、CIFAR-FS、音乐符号、古埃及/希腊/土耳其字符、医学影像以及自然图像等，覆盖不同视觉域与类别规模。

**📈 对比分析**

实验在多种 n-way k-shot 组合下进行，对比四种预训练策略的平均准确率提升：同域预训练提升约 33.4pp，跨域有标签提升约 23.8pp，跨域无标签（UIAug）接近 27.7pp，且在最佳源域选择下可实现与有标签相当的性能。

**⚠️ 局限性**

局限在于仅考虑了三种元学习架构和有限的跨域数据，未覆盖更大规模或多模态数据；无标签预训练对聚类参数敏感，且源域选择仍需先进行特征提取或重建，未实现完全端到端自动化。

---

## 142. EMMI: Edge Multi-Modal Intelligence for Communication-Efficient MLLM Inference via Fused Representation Compression

**arXiv ID:** 2609.11058 | [PDF](https://arxiv.org/pdf/2609.11058v1)

**作者:** Motahare Mounesan `[一作]` (Texas A&M University), Irfan Khan `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个边缘-服务器架构EMMI，通过在边缘进行模态编码、跨模态融合和压缩，仅传输紧凑的潜在表示给服务器进行多模态大语言模型推理。

**💡 创新点**

创新点在于将通信边界从原始输入或中间激活转移到统一压缩的跨模态表示，实现了通信效率与隐私保护兼顾，并引入任务感知与无监督压缩两种策略。

**🔧 技术方法**

技术包括模态特定编码器（如CLIP、MobileCLIP）、跨模态融合（基于成对交互）、自编码器/对比学习压缩、可训练的解压缩和投影模块。

**📊 数据集**

使用MS‑COCO图像-文本对作为视觉‑语言基准数据集。

**📈 对比分析**

与未压缩基线、PCA、AE、VAE、BlockPCA、LDA等传统与闭式方法比较，任务感知自编码压缩在32×压缩下保持与原始基线相当的准确率，通信负载仅256B，端到端延迟提升约3.4倍。

**⚠️ 局限性**

局限包括对压缩比与模型尺寸的敏感性，主要在低比特率时性能下降；目前仅在CPU单线程下评估，缺乏对真实边缘硬件的验证；以及对更复杂多模态输入和多任务迁移的适应性待进一步研究。

---

## 143. Work, Wellbeing, and Choice: Empirical Lessons for AI Futures

**arXiv ID:** 2609.11019 | [PDF](https://arxiv.org/pdf/2609.11019v1)

**作者:** Stephanie C. Y. Chan `[一作]`, Iason Gabriel `[通讯]` (Google DeepMind)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对不同非就业人群（失业、退休者、彩票中奖者、海湾国家公民和经济依赖配偶）的文献进行比较性综述，探讨工作状态与福祉之间的关系，并提出对AI驱动自动化背景下福祉影响的启示。

**💡 创新点**

首次将多学科、跨国的实证研究整合成三大调节因子框架（代理与选择、替代性工作收益、社会与制度背景），并将其与AI自动化情景相对应，弥补了以往孤立研究的局限。

**🔧 技术方法**

采用系统性文献检索与综合分析方法（包括元分析、自然实验、随机试验和横向研究的归纳），无机器学习模型。

**📊 数据集**

使用的主要数据源为已有的多国横断面与纵向面板数据、自然实验（企业关闭、彩票中奖、失业案例）、以及社会学、心理学与经济学的原始研究报告。

**📈 对比分析**

通过对不同人群、不同研究设计的比较，评估失业对心理、身体、认知福祉的负面效应，并比较退休、志愿服务等替代性活动对福祉的缓冲作用；结论表明仅靠经济补偿不足，需关注代理、替代收益和制度保障。

**⚠️ 局限性**

局限在于：① 所有参考研究均来自当代后工业社会，可能不适用于未来极端自动化情境；② 未系统涵盖所有可能的人群（如F.I.R.E.运动、低收入非西方社会）；③ 对UBI等纯金钱干预的实证数据有限；④ 文献综述无法提供定量预测。

---

## 144. ShellVis: Sandboxed Live Programming for Shell Scripts

**arXiv ID:** 2609.11000 | [PDF](https://arxiv.org/pdf/2609.11000v1)

**作者:** Joshua Horowitz `[一作]` (University of Washington), Jeffrey Heer `[通讯]` (University of Washington)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 ShellVis，一种在沙箱环境中进行实时编程的 Bash 脚本编辑器，提供行级反馈和文件系统操作可视化。

**💡 创新点**

将沙箱技术与实时编程结合，解决了有副作用脚本的安全性和可视化问题，并在文本编辑器中无缝集成。

**🔧 技术方法**

利用 OverlayFS/unionfs-fuse 做文件系统沙箱，使用 AST 变换插桩跟踪，React+Automerge 实现可视化，HTTP 与 WebSocket 交互。

**📊 数据集**

未使用公开数据集，而是通过七名有经验的 shell 用户在四个手工编写的脚本任务进行用户研究。

**📈 对比分析**

通过用户调研和 Likert 调查与传统日志/调试工具对比，平均满意度达 4.9/5；性能上每次运行大约比原生慢 7 倍，主要受通信和沙箱开销影响。

**⚠️ 局限性**

限制包括仅对文件系统副作用沙箱、在 macOS 上 unionfs-fuse/namespace 效率低、未覆盖网络等副作用，且仅支持有限的 shell 结构；对新手不够友好，缺乏执行层面指导。

---

## 145. Importance Weighting for Unlabeled-unlabeled Learning under Distribution Shift

**arXiv ID:** 2609.10994 | [PDF](https://arxiv.org/pdf/2609.10994v1)

**作者:** Atsutoshi Kumagai `[一作]` (NTT), Yasuhiro Fujiwara `[通讯]` (NTT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于重要性加权的分布迁移适应方法，用于从两个不同类先验的未标记数据集中学习二分类器。

**💡 创新点**

创新点在于首次将重要性加权框架与 UU 学习结合，可处理任意弱监督形式（PN、PU、噪声标注、相似度学习等），并且不需要假设特定的分布偏移类型。

**🔧 技术方法**

使用相对密度比估计的加权重要性、绝对值校正、共享特征提取器的神经网络以及动态迭代训练；在对比实验中还引入 MMD 以实现特征不变性。

**📊 数据集**

实验使用了 MNIST、FashionMNIST、CIFAR-10（图像数据）以及 DIABETES（表格数据），构造支持移位和输入输出关系移位两类分布偏移。

**📈 对比分析**

与 teUU、trUU、mtUU、mtsUU、daUU 等基线方法对比，所提方法在 6/7 组实验中获得最优或相近性能，明显优于仅使用小规模测试 UU 数据或仅使用不加权的 UU 学习方法。

**⚠️ 局限性**

局限性包括需先验知道各分布的类先验、对极端分布偏移时重要性权重估计可能不稳、以及对超参数（α、β）敏感；在测试 UU 数据极少的场景下仍可能受限。

---

## 146. Thompson Sampling for Non-Monotone Convex Ridge Bandits: Monotonicity Is Not Needed for Polynomial Regret

**arXiv ID:** 2609.10981 | [PDF](https://arxiv.org/pdf/2609.10981v1)

**作者:** Xuan Li `[一作]` `[通讯]` (University of New South Wales), Xuan Li (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文探讨了在非单调链接的情况下，Thompson采样（TS）在带有凸单调岭损失的带子凸优化中的贝叶斯遗憾界限。

**💡 创新点**

创新点在于证明了即使在非单调链接的情况下，TS仍然可以实现多项式级别的贝叶斯遗憾，且不需要单调性假设。

**🔧 技术方法**

使用了Thompson采样（TS）算法，并结合了信息比率和转移定理的技术。

**📊 数据集**

使用了具有任意凸、可能非单调链接的1-Lipschitz凸岭损失的先验分布，构造了d(d+1)个非信息损失函数。

**📈 对比分析**

与现有方法相比，论文展示了在非单调情况下，TS的贝叶斯遗憾为O((d+1)^4√(dn) log(e+ndmax{1, K}))，性能优于传统的单调岭损失情况。

**⚠️ 局限性**

限制在于尚未解决单调情况下的d^5/2依赖性是否可以保留的问题，以及对计算复杂性的讨论仍然开放。

---

## 147. MMS Allocation for Chores with Online Agent Arrivals

**arXiv ID:** 2609.10960 | [PDF](https://arxiv.org/pdf/2609.10960v1)

**作者:** Haolong Li `[一作]` (University of Macau), Xiaowei Wu `[通讯]` (University of Macau)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在线代理到达模型下的任务分配问题，提出在子加性、加性和二元加性费用下的最大最小份额（MMS）公平分配算法。

**💡 创新点**

创新点包括：① 在完全未知类型的子加性费用下给出近似最优的 O(min{n, k log^{1+ε}k, log m}) 竞争比；② 在已知类型的加性费用下实现 O(log k) 或 O(log(k n)/loglog(k n)) 的竞争比；③ 在已知二元加性费用且 k≤n 的情形下提供 3‑竞争的确定性算法，并证明在 k=O(1) 时下界为 2。

**🔧 技术方法**

主要技术：动态估计类型数并按需分配 MMS 分区的子集；构造通用分区（universal partition）和通用剩余（universal residual）；使用潜能函数、条件期望和离散化随机化来实现确定性多项式时间算法；以及构造嵌套与批次化的硬实例来证明下界。

**📊 数据集**

论文基于理论分析，不依赖任何真实数据集；所有实例均为合成（例如二元加性集合覆盖类）。

**📈 对比分析**

与现有工作对比：在子加性费用下实现了与离线最优同阶的竞争比；在已知类型的加性费用下首次突破对数阶界；在二元加性费用下给出了常数阶竞争比，优于已知的 2‑Ω 下界；实验结果通过证明和下界分析验证算法性能。

**⚠️ 局限性**

局限性：① 对于已知类型的二元加性费用，最优竞争比仍不确定，尤其在 k>n 时；② 论文仅考虑确定性算法，随机化算法潜在优势未探讨；③ 仅在 k≤n 时给出 3‑竞争，k>n 的情况仍无可行方案；④ 对于通用分区的更紧逼下界尚未完成。

---

## 148. New Evidence, Same Choice: Testing Physical Experiment Selection in Vision Language Models

**arXiv ID:** 2609.11022 | [PDF](https://arxiv.org/pdf/2609.11022v1)

**作者:** Sourajit Saha `[一作]` (University of Maryland, Baltimore County), Qiheng Wang `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一套配对实验选择基准，用来检测视觉语言模型在物理推理任务中何时需要额外实验以及选择哪种实验。

**💡 创新点**

通过构造可比的两组输入，明确判定何时需要实验、哪个实验最便宜，从而揭示模型在证据使用上的缺陷。

**🔧 技术方法**

使用多种视觉语言模型（如 Qwen2.5 VL、SmolVLM2、Idefics3、Pixtral 等），在给定图像、问题、测量与阈值下执行实验选择与答案生成，并配合解析器评估输出。

**📊 数据集**

采用自制的 144 个物理问题族（滑动、弹跳、弹簧），每族包含四种可能世界，生成 576 决策与 288 对比样本。

**📈 对比分析**

通过“直接”与“简要推理”两种协议对每个模型进行评测，计算实验选择正确率、完整匹配率和回答准确率；结果显示大多数模型在配对中重复相同动作，只有极少数能正确解决完整对，整体性能偏低。

**⚠️ 局限性**

仅限离散理想物理系统、单一附加实验、特定模型与版本，未考虑视频、噪声、机器人等真实场景，且答案生成与选择分离导致无法完整评估内部推理。

---

## 149. UniRec: Cross-stage Multi-Task Fusion with Preference Alignment for Cascaded Recommender Systems

**arXiv ID:** 2609.11052 | [PDF](https://arxiv.org/pdf/2609.11052v1)

**作者:** Lingyuan Kong `[一作]` (Kuaishou Technology), Kaiqiao Zhan `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 UniRec，一种统一的跨阶段融合框架，联合优化短视频推荐系统中预排序和排序阶段的融合模型；

**💡 创新点**

创新点包括：① 将两阶段融合模型在同一计算图中训练，允许跨阶段梯度流动；② 双轴偏好对齐（垂直对齐下游排序偏好，水平压缩聚合多任务偏好为两条双向信号）；③ 引入属性组相对正则化（AGRR）抑制属性级奖励偏差；④ 采用跨阶段一致性损失和兼顾在线A/B测试；

**🔧 技术方法**

技术手段包括：共享嵌入层、单独的深度与交叉网络（DCN）、轻量化 MLP-Mixer（预排序）与 AutoInt-Lite（排序）结构、适配器网络、Softplus 对比损失、CPPA（压缩聚合对比偏好）、AGRR 组内 KL 正则、端到端联合训练；

**📊 数据集**

使用的实验数据集：RecFlow 公共级联推荐基准（约42K 用户、9.3M 请求），以及 Kuaishou 生产级短视频系统的真实日志（超过1M 请求、100M 日活）；

**📈 对比分析**

与四类基线对比（加权求和、EMER、UMRE、COPR）及消融实验。UniRec 在 RecFlow 上在 NDCG@10/30/50、rank_AUC、ASH、Kendallτ、Spearmanρ 等指标均优于基线；在生产数据的离线评估和在线 A/B 测试中，提升总观看时间+0.675%、视频观看时间+0.755%、应用使用时长+0.616%，且无任何指标下降；

**⚠️ 局限性**

局限性：对属性组正则化依赖于属性划分（如视频时长）且仅在属性范围内消除偏差，跨属性的整体排序仍受多任务偏好不一致影响；模型对超参数权重敏感，需手工调优；目前仅覆盖预排序与排序两阶段，未扩展到再排序及其他业务场景。

---

## 150. The Agent Incident Registry: Toward Preventing Repeated AI Agent Failures

**arXiv ID:** 2609.11030 | [PDF](https://arxiv.org/pdf/2609.11030v1)

**作者:** Divyanshu Kumar `[一作]` (Anaconda), Prashanth Harshangi `[通讯]` (Anaconda)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Agent Incident Registry（AIR），一个源链接、机制化编码的 AI 代理事件登记表，记录了 2020‑2026 年间公开披露的代理相关事件，并为每条记录提供了稳定 ID、支持性引用、缺失标注和多维度标签。

**💡 创新点**

创新点包括：① 将实现伤害、演示能力、因果角色和披露类别四个维度拆分开来，防止公开事件被误计为部署风险；② 采用缺失值标注和可验证的来源链接，保证数据透明度；③ 将每条记录映射到 OWASP ASI 机制集合，形成可直接用于评估设计的“表面-向量-触发器”三维结构。

**🔧 技术方法**

技术手段主要有：多源检索与自动去重、基于源链接的文本提取、人工双人审核、规则驱动的 OWASP ASI 交叉映射、Wilson 区间和 Bootstrap（按首源主机分块）估计统计不确定性。

**📊 数据集**

数据集来自公开的事件披露渠道，包括 AI Incident Database、CVE/GHSA、供应链、学术演示、媒体报道等，最终编制了数千条去重后的事件记录。

**📈 对比分析**

比较方法：将 AIR 记录的表面、向量和触发器与现有评估工具（如 InjecAgent）对齐，检验评估覆盖度。通过 Bootstrap 与 Wilson 区间给出比例区间，结果显示评估缺失无对手触发的安全失败机制，提示需补充此类测试。

**⚠️ 局限性**

局限性：① 仅基于公开披露，缺乏部署分母，无法估计真实失败率；② 来源聚集导致样本依赖，Bootstrap 仅捕捉部分依赖；③ 未覆盖地理、语言或非生成型代理；④ 机制字段覆盖不足（如财务损失、可逆性），难以完整评估控制效果。

---

## 151. Empirical Evaluation of Data Poisoning Attacks in Supervised Learning

**arXiv ID:** 2609.10952 | [PDF](https://arxiv.org/pdf/2609.10952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 152. Online Treasure Hunt in Vertex-Permuted Dynamic Rings

**arXiv ID:** 2609.11013 | [PDF](https://arxiv.org/pdf/2609.11013v1)

**作者:** Kamran Ayoubi `[一作]` (Concordia University), Lata Narayanan `[通讯]` (Concordia University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究在顶点置换动态环（vertex‑permuted dynamic rings）中，单个或多个匿名无记忆代理寻找隐藏宝物（treasure hunt）的可行性与最优搜索时间。

**💡 创新点**

创新点：
- 证明在无约束的顶点置换环中，即使有 k≤n‑3 个代理，也存在输入使宝物永远无法被发现。
- 确定可行的动态性阈值 δ≥⌈(n‑1)/2⌉；在该阈值下给出单代理有标记(flag)时的最优 Θ(δn)搜索时间与竞争比率；
- 通过引入可在邻居处植旗并被一跳邻居可见的“旗帜”机制，实现多代理线性加速：k 代理可在 O(nδ/k) 时间内完成搜索。
- 证明在无旗帜模型下单代理不可能求解，但存在期望 O(δ²) 的随机算法。
- 对随机顶点置换模型 R‑VP 分析，给出对抗性与无对抗性对手下分别为 Θ(n) 与 Θ(n log n) 的期望搜索时间。

**🔧 技术方法**

主要技术：
- 逆向动态调度与“k+2”或“3”策略（adversarial confinement）构造不可达性。
- Walecki 较为 Hamiltonian 循环分解，用于构造满足 δ-邻接要求的动态序列。
- 旗帜机制作为一种单比特持久记忆，帮助代理区分已访问与未访问节点。
- 采用随机漫步等概率分析证明随机算法的期望时间。
- 组合论与图论（如完全图的覆盖时间、碰撞避免）用于下界证明。

**📊 数据集**

该工作完全是理论分析，没有使用任何实验数据集；所有结果均为极限时间/期望时间上界与下界的闭式表达式。

**📈 对比分析**

评估方式：
- 通过构造最坏情况输入（对抗性序列）得到下界；
- 通过明确的算法步骤与递归/窗口分析得到上界；
- 上下界在大多数参数范围内匹配，证明了所给算法的渐近最优性。
- 对 R‑VP，利用随机图与随机游走理论得到 Θ(n) 与 Θ(n log n) 的期望结果。

**⚠️ 局限性**

局限与未解决问题：
- 单代理有旗帜模型的下界仅在 δ≥2n 证明，是否在完整可行范围 δ≥⌈(n‑1)/2⌉ 下也成立尚未确定。
- 随机算法对抗适应性在线对手（adaptive online adversary）的竞争比率未给出。
- 结果仅针对环形拓扑，尚未推广到更一般的顶点置换图（如网格、树等）。

---

## 153. Engineering Reliable Commit Gates for Agentic AI: Cost-Aware Verification Portfolios under Common-Mode Data Failures

**arXiv ID:** 2609.10969 | [PDF](https://arxiv.org/pdf/2609.10969v1)

**作者:** Zihao Zheng `[一作]` (Washington University in St. Louis), Jiayu Long `[通讯]` (Washington University in St. Louis)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并实现了面向代理系统的提交门（commit gate），评估了多种验证机制（模型校验、独立数据源、事务性 guard、人工推迟）对降低代理提交不安全操作的影响，并提出了基于可观察元数据的可解释策略组合（portfolio）来在成本预算内控制风险。

**💡 创新点**

创新点在于（1）系统性分离并量化“证据来源（source）”与“模型多样性（model）”的贡献，发现来源多样性对降低错误批准率更为关键；（2）构造可观测的门接口与决策树库，利用成本、覆盖率与风险三维度的校准（基于 Wilson 置信上界）实现自适应策略；（3）在受控基准、离线重放与真实 HTTP/SQLite 实验三种环境下，对不同策略进行可复现的评估，揭示了 atomic guard 与 portfolio 的边界与局限。

**🔧 技术方法**

技术主要包括：多模型语言模型验证（Qwen3.5、Phi-4）、事务性 guard 的实现（SQLite 事务），基于 Bootstrap 与 permutation 的统计检验，风险控制的 Lagrangian 校准，分层上下文回退的决策树生成，以及完整的可追溯日志与审计。

**📊 数据集**

数据集包括：48 个语义模板（调度器、管道、存储、IAM），每个模板10个种子，6种故障模式，共2880个场景；另有12条实时业务合同（schema migration、snapshot promotion、job restart）用于现场实验；外部 FinQA 960 条文档基准用于验证通用性。

**📈 对比分析**

比较方法：在受控基准上使用 2×2 实验隔离来源与模型差异；在校准阶段利用 Wilson 上界和 Cluster 调整评估风险目标；在现场实验中对 8 种策略做对比（plain commit、交叉模型投票、独立来源投票、部分 guard、完整 atomic guard、组合 portfolio 等），通过比例差异、风险覆盖率与成本进行多维度性能评估。结果显示：来源多样性能把错误批准率从 74% 降至 33%，portfolio 在 5% 目标下将 unsafe 率降至 1% 以内，atomic guard 在可检查完整谓词时优于任何多模型策略。

**⚠️ 局限性**

局限性：① 校准与风险目标仅在实验环境下有效，未在分布式或长期演化系统中验证；② 只考虑单一写入者的 SQLite，未覆盖竞争写入或恶意攻击；③ 证据线索与模型能力相关，无法跨领域推广（FinQA 结果失败）；④ 近似 Wilson 上界不保证严格的有限样本安全；⑤ 仅评估了 4–10B 量级模型，规模与推理成本未在更大模型上探测；⑥ 对线性化点与并发写入的完整理论分析缺失。

---

## 154. Measuring the Value of World-Model Updates: A Counterfactual Utility Protocol for Continual Adaptation

**arXiv ID:** 2609.10954 | [PDF](https://arxiv.org/pdf/2609.10954v1)

**作者:** Anqi Peter Li `[一作]` (Substrate Labs), Kaden Kim `[通讯]` (University of California Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究持续学习中对模型更新时机的评估，提出分叉账本（fork ledger）来直接比较更新与不更新在同一环境下的回报差异。

**💡 创新点**

创新点是将单个更新事件的因果效应转化为可观测的对比实验，并以此评估触发信号的有效性。

**🔧 技术方法**

采用离线仿真、DreamerV3风格的可重复性环境、基于随机数同步的对比实验、梯度更新与无更新分支等技术。

**📊 数据集**

在DeepMind Control Suite的CartPole、Walker2D、Cheetah、Hopper四个MuJoCo控制任务上进行实验。

**📈 对比分析**

通过对比更新分支与保持分支的累计回报差异评估每个决策点，结果显示固定更新规则在所有任务中均导致回报下降，且触发信号的排序在不同任务间不一致。

**⚠️ 局限性**

局限性包括仅评估单一固定更新机制、依赖可回溯的仿真环境、未检验模型对多任务迁移的普适性、以及仅在控制任务上验证，缺乏真实世界的验证。

---

## 155. Testing Between the Test Cases: Proving End-to-End Steering in Conditions You Never Drove

**arXiv ID:** 2609.10951 | [PDF](https://arxiv.org/pdf/2609.10951v1)

**作者:** Menuka Ghalan `[一作]` (Western Michigan University), Zachary D. Asher `[通讯]` (Western Michigan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在CARLA仿真环境下训练并验证两种仅使用前置摄像头的端到端转向网络（分别在清晰和混合光照条件下训练），并通过形式化验证技术评估其在不同恶劣环境下的安全性。

**💡 创新点**

创新点在于：①采用一维线性插值参数化的扰动模型（同一姿势下清晰与恶劣图像的混合），实现一次性覆盖整个扰动空间；②使用CROWN形式化验证对该扰动集合进行边界传播，能发现闭环测试遗漏的中间强度失败点；③将验证结果与闭环仿真对比，验证方法能在几分钟GPU时间内评估数百个姿势对应的10^133组合。

**🔧 技术方法**

使用的技术包括：端到端卷积神经网络（PilotNet结构蒸馏）、CROWN形式化验证（基于ReLU线性上下界）、CARLA仿真闭环测试、参数化扰动与平均偏差（Δ̅）评估指标。

**📊 数据集**

使用的数据集为CARLA生成的Town04高速与Town06城市干道图像，分别在清晰、雾、夜间和低阳光四种条件下捕获的同一姿势图像对，用于训练（行为克隆+DAgger）和验证。

**📈 对比分析**

通过将闭环测试结果与形式化验证的安全门限比较，验证在高速路上几乎完全一致，城市路混合条件下验证更严格并能揭示测试遗漏的失败。形式化验证覆盖数百个姿势对应10^133种扰动，仅需几分钟GPU时间；闭环测试需要多次仿真，耗时更长。

**⚠️ 局限性**

局限性包括：仅考虑单一摄像头输入，扰动模型仅为单参数线性插值，未覆盖雨雪、眩光等更复杂光照；模型规模受限，验证仅针对稳态扰动；对真实道路环境的泛化能力尚未验证。

---

## 156. Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction

**arXiv ID:** 2609.10950 | [PDF](https://arxiv.org/pdf/2609.10950v1)

**作者:** Han-Jun Choi `[一作]` (Korea Electronics Technology Institute), Jin Yea Jang `[通讯]` (Korea Electronics Technology Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在多模态情感分析中提出一种基于语义完整度估计的文本重建框架，利用伪标签训练完整度估计器并结合代理特征实现缺失文本语义恢复，从而提升情感预测准确率。

**💡 创新点**

创新点包括：①基于目标类别概率的语义完整度估计（TPSC）与伪标签生成；②重要性感知代理特征生成器（IPFG）动态平衡辅助模态贡献；③交替优化策略（AOS）缓解多任务梯度冲突，实现完整度估计与情感预测的稳定联合学习。

**🔧 技术方法**

采用的技术主要有：BERT+Transformer编码器、全连接完整度估计网络、门控IPFG生成代理特征、目标类别概率（TCP）伪标签、交替优化训练框架。

**📊 数据集**

使用的公开基准数据集包括MOSI、MOSEI和SIMS三大多模态情感分析数据集。

**📈 对比分析**

在与MISA、Self-MM、MMIM、CENet、TETFN、TFR-Net、ALMT、LNLN、P-RMF和TF-Mamba等12种基线模型对比实验中，TCMR在所有评价指标（Acc、F1、MAE、Corr等）上均表现优于或接近最优，尤其在缺失率升高时保持较高鲁棒性。

**⚠️ 局限性**

局限性主要有：①仅针对文本模态的完整度估计，未扩展到音频/视觉模态；②伪标签生成存在噪声，可能影响完整度估计准确性；③对高噪声或低置信度样本的处理尚不完善，需要进一步改进噪声抑制与人类验证策略。

---

## 157. Gait-Dependent Effects on Quadruped Locomotion for Load-Carrying using Passive Mechanism

**arXiv ID:** 2609.11059 | [PDF](https://arxiv.org/pdf/2609.11059v1)

**作者:** Giovanni B. Dessy `[一作]` (Istituto Italiano di Tecnologia), Victor Barasuol `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过仿真分析了在不同步态和载荷条件下，四足机器人使用被动机械臂携带负载时，阻尼与无阻尼阻尼配置对行走稳定性的影响。

**💡 创新点**

提出了将被动臂阻尼-刚度选择与步态、负载质量联合考量的设计图谱，并阐明步态序列如何调制被动臂耦合对ZMP裕度的影响。

**🔧 技术方法**

使用MuJoCo仿真、相同的MPC控制器、ZMP裕度评估、振荡RMS指标以及设计映射分析等技术。

**📊 数据集**

采用自定义的四足+被动臂模型，在MuJoCo仿真中生成的多种步态、负载重量和阻尼配置的仿真数据。

**📈 对比分析**

通过对比被动臂阻尼与无阻尼配置在不同步态和负载下的ZMP临界时间、负裕度时间以及臂振荡指标，发现无阻尼配置在某些步态和高负载下会显著降低ZMP裕度，性能差异随步态而异。

**⚠️ 局限性**

仅在仿真环境下验证，未考虑真实硬件的摩擦、传感器误差和动力学非线性，且阻尼-刚度参数范围有限，需进一步实验验证和解析模型。

---

## 158. A deterministic $(1+\varepsilon)^n$ approximation for the permanent of a nonnegative matrix

**arXiv ID:** 2609.11049 | [PDF](https://arxiv.org/pdf/2609.11049v1)

**作者:** Dingding Dong `[一作]` (California Institute of Technology), Vishesh Jain `[通讯]` (University of Illinois Chicago)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种确定性的强多项式算法，该算法针对非负矩阵A，返回一个矩阵Q，使得A≤Q≤(1+ε)^n A。

**💡 创新点**

创新点在于提供了一个强多项式时间的确定性算法，解决了计算非负矩阵的永久性问题，并且可以在给定的ε范围内进行近似。

**🔧 技术方法**

使用了线性代数和凸优化技术，结合了矩阵缩放和其他数学工具来实现算法。

**📊 数据集**

使用了非负矩阵A∈_≥0^n× n，具体数据集未明确说明，但假设A具有正的永久性。

**📈 对比分析**

与现有方法相比，算法在多项式时间内提供了更好的近似因子，性能上优于已知的确定性多项式时间算法，后者的近似因子是指数级的。

**⚠️ 局限性**

算法的局限性在于它依赖于矩阵的特定性质，例如支持图必须具有完美匹配，且在处理大规模数据时可能会面临计算复杂度的挑战。

---

## 159. LTLDiff: Finite Linear Temporal Logic-Guided Data Generation and Diffusion Policies for Multi-agent Robotic Manipulation

**arXiv ID:** 2609.11043 | [PDF](https://arxiv.org/pdf/2609.11043v1)

**作者:** Chuhan Meng `[一作]` (University of Toronto), Haiyan Yin `[通讯]` (Agency for Science, Technology and Research)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发 LTLDiff 框架，将 LTL_f 逻辑与扩散策略结合，用于多智能体机器人操控任务。

**💡 创新点**

①用大语言模型生成 LTL_f 规范并编码为固定维向量；②在离线数据生成和扩散策略训练中用 LTL_f 条件指导；③将单智能体扩散策略迁移到多智能体并加入回归器梯度指导。

**🔧 技术方法**

大语言模型（Qwen）生成 LTL_f，AST 编码，条件扩散模型，回归器梯度指导，离线示范收集。

**📊 数据集**

RoboFactory 多智能体操控基准，11 个任务，使用 50/100/150 条示范。

**📈 对比分析**

与基线扩散策略和 LTLDOG-R 比较，LTLDiff 在单、双、三、四智能体任务中取得更高成功率，尤其在两三智能体任务上显著提升。

**⚠️ 局限性**

对长时间程任务表现差，需要引入闭环反馈控制改进。

---

## 160. Toward Interpretable Multimodal Fusion: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification

**arXiv ID:** 2609.11040 | [PDF](https://arxiv.org/pdf/2609.11040v1)

**作者:** Kan Wei `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Pedram Ghamisi `[通讯]` (Helmholtz-Zentrum Dresden-Rossendorf)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于热扩散物理模型的多模态融合框架 M2Heat，用于高光谱与 LiDAR 图像联合分类。

**💡 创新点**

创新点在于将热传导方程转化为可学习的频域扩散算子 vHeat，并引入频值嵌入 (FVEs) 以实现自适应的热扩散，从而实现可解释的全局依赖建模和高效跨模态融合。

**🔧 技术方法**

采用热扩散算子 HCO、频域交叉融合 CFF、DCT/FFT 变换、频值嵌入以及轻量级分类头等深度学习技术。

**📊 数据集**

使用 Trento、Houston2013 和 Augsburg 三个公开高光谱+LiDAR 结合数据集进行实验。

**📈 对比分析**

与九种最先进的单源/多源方法比较，M2Heat 在三大基准上均取得最高或次高的整体准确率 (OA)、平均准确率 (AA) 与 Kappa，尤其在 Trento 上获得 99.64% OA。

**⚠️ 局限性**

局限在于相较于线性复杂度的 Mamba 等算子仍具有一定计算成本，并且对不同场景的跨域推广需进一步验证。

---

## 161. Rebalancing Token Importance in Language Models with TF-IDF Weighted Cross-Entropy Loss

**arXiv ID:** 2609.11029 | [PDF](https://arxiv.org/pdf/2609.11029v1)

**作者:** Zhijian Li `[一作]` (University of Southern California), Kevin Leach `[通讯]` (Vanderbilt University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 TF‑IDF 的交叉熵损失函数，在训练大语言模型时对每个 token 的梯度进行加权，从而降低模型对频繁低信息词的过度学习，减少对训练数据的逐字记忆。

**💡 创新点**

创新点在于：① 将传统的 TF‑IDF 统计直接嵌入到损失函数中，实现对 token 信息密度的动态加权；② 通过局部缓冲区高效估计 IDF，避免全语料全量预处理；③ 在保持所有 token 监督的同时，重塑梯度分布，既不影响语义学习，又显著抑制逐字复制。

**🔧 技术方法**

技术手段包括：TF‑IDF 加权交叉熵；低秩适配（LoRA）和全权重微调；滑动缓冲统计、子词级别加权、梯度归一化；评估指标（最长已记忆子串 LMS、前缀匹配、ROUGE‑L、perplexity、EM/F1）。

**📊 数据集**

使用的数据集有：① Pile 训练语料（包含 WikiText‑2 注入用来评测记忆）；② CNN/DailyMail 用于摘要评测；③ SQuAD 用于问答评测；同时在微调过程中使用标准验证集评估 perplexity。

**📈 对比分析**

与传统统一权重交叉熵相比，TF‑IDF 加权在 LoRA 微调下平均降低 14% 的 LMS，在全权重微调下可降至 58%；perplexity 在大多数模型上保持不变或略有提升；摘要与问答指标基本不变（误差在 0.5 ROUGE 或 1.5% EM 以内）。

**⚠️ 局限性**

局限性包括：① 只评估了逐字记忆，未覆盖语义式或对抗式提取；② 仅在微调阶段验证，未考察从头训练；③ 子词分词可能削弱 TF‑IDF 对词级信息的捕捉；④ 缓冲区 IDF 近似可能在极端稀有词上不稳定；⑤ 仅在 256‑token 上下文窗口评估，长范围记忆未覆盖。

---

## 162. K/V-Cache Interventions Dissociate Representation Alignment from Persona Expression in Decoder-Only Language Models

**arXiv ID:** 2609.11020 | [PDF](https://arxiv.org/pdf/2609.11020v1)

**作者:** Yu Sun `[一作]`, Huimin Han `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们在Llama-3.1-8B上对K/V缓存进行多种层级替换、插值、位置扰动等干预，研究其对persona控制的影响。

**💡 创新点**

发现K/V层级与行为之间存在解耦，尤其中层替换既能对齐表示又保持词汇多样性，表明仅靠表示相似度不足以预测行为。

**🔧 技术方法**

采用K/V缓存替换、插值、位置扰动等干预技术，并通过V^⋆子空间投影、K/V余弦相似度、文本marker密度和TTR等指标评估。

**📊 数据集**

使用单一source→target persona对（Nurse Reyes→Priya）以及30人设语料，seed主题为“酸面包发酵失败”。

**📈 对比分析**

与全层替换对比，中层替换在V^⋆对齐和target-marker密度上相近，但TTR更高，显示表示与行为的显著解耦。

**⚠️ 局限性**

局限性包括仅测试单一模型、单一persona对、单一主题、短生成长度，以及对手工marker的依赖，缺乏跨模型和多主题的验证。

---

## 163. DeFiFusion: Combining Transaction Events with Smart Contracts to Detect Price Manipulation Attacks

**arXiv ID:** 2609.11008 | [PDF](https://arxiv.org/pdf/2609.11008v1)

**作者:** Rui Cao `[一作]` (Nanjing University of Aeronautics and Astronautics), Zhenguang Liu `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 DeFiFusion 框架，融合交易事件与智能合约语义，使用双模投影融合 Transformer 捕捉循环多阶段执行模式，从而实现对价格操纵攻击（PMA）的检测。

**💡 创新点**

创新点在于：①针对 PMA 的事件编码方案；②利用大型语言模型（LLM）抽取合约语义；③引入 Dual‑Modal Projection‑Fusion Transformer 与 T5‑Style 相对位置编码，弥补单模检测的缺陷；④构建并公开最大规模的 225 条真实 PMA 事件数据集。

**🔧 技术方法**

技术手段包括：事件特征编码、掩码平均池化提取合约嵌入、Dual‑Modal Projection‑Fusion Transformer、T5‑RPE、注意力机制、LLM（DeepSeek‑Coder‑7B）等。

**📊 数据集**

使用的数据集为：D_1（225 条 2020‑2026 年真实 PMA 事件）和 D_2（1150 条高价值正常交易），并在 2026 年前瞻性测试中验证新攻击场景。

**📈 对比分析**

与 DeFiRanger、DeFort、DeFiScope 等现有 SOTA 方法比较，DeFiFusion 在召回率 98.67% 与精度 96.10% 上实现领先；在 2026 年前瞻性实验中实现 100% 召回。

**⚠️ 局限性**

限制因素包括：对源代码的依赖，闭源合约无法使用；极少量混合正常交易与攻击行为的情况仍可能出现误检；模型仍需多模特征协同，单模性能有限。

---

## 164. Rethinking Verbalized Confidence for LLM-as-a-Judge: A Compatibility Shift on Post-2025 Proprietary Models

**arXiv ID:** 2609.10996 | [PDF](https://arxiv.org/pdf/2609.10996v1)

**作者:** Yu-Chung Hsiao `[一作]` `[通讯]` (Cisco Systems), Yu-Chung Hsiao (Cisco Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM-as-a-Judge的软评分机制进行研究，提出一种基于verbalized confidence的无logprob协议，并通过加入过度自信警示和自我辩论两项提示来提升校准度、分布广度和对任务主观性的鲁棒性；

**💡 创新点**

首次发现并量化了“兼容性转移”（compatibility shift）和“生成效应”（generation effect），表明后2025年旗舰模型在使用该协议时能保持准确率，而旧版模型会下降；

**🔧 技术方法**

采用单次调用的提示工程（overconfidence advisory、self-debate），利用贝塔分布E​CE、Bhattacharyya系数衡量校准与分布，比较G‑Eval（基于logprobs）与自定义verbalized协议；

**📊 数据集**

使用SummEval、HelpSteer2和AggreFact三大基准，涵盖主观质量评估和客观事实性验证，共计约 28,000 条样本；

**📈 对比分析**

在所有基准上，相较于G‑Eval，verbalized confidence 在 Kendall τ、AECE、Score Spread 等指标上均优于或持平，并在GPT旗舰模型中表现出更强的主观性鲁棒性；

**⚠️ 局限性**

局限包括：对logprob的比较仅限于返回logprob的GPT模型；实验仅覆盖单调用模式；对闭源旗舰内部机制缺乏直接可解释性，且结果可能随模型后续迭代而变化。

---

## 165. Distribution-aware Language Neuron Identification in Multilingual Large Language Models

**arXiv ID:** 2609.10993 | [PDF](https://arxiv.org/pdf/2609.10993v1)

**作者:** Minjun Kim `[一作]` (KAIST), KyungTae Lim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于激活分布重叠的语言神经元选择方法，能够识别单语言神经元（SLNs）和多语言神经元（MLNs），并通过平均补丁干预验证其因果效果。

**💡 创新点**

创新点在于：①利用每个神经元的完整激活分布而非仅正值率；②用重叠系数构建语言之间的关系矩阵，并通过单链接聚类发现语言群组；③同时识别负激活区的神经元并扩展到 z‑site（gate‑up 乘积）实现更全面的识别。

**🔧 技术方法**

技术手段包括：直方图密度估计、重叠系数积分、单链接聚类、阈值筛选、平均补丁干预、对比实验（LAPE、LSN、LRN）、下游任务评估（Belebele、MGSM）。

**📊 数据集**

使用七种语言（英语、中文、法语、西班牙语、越南语、印尼语、日语）的 FLORES+ 开发集、Wikipedia 按语言拆分的测试集，以及 Belebele 和 MGSM 下游数据集进行评估。

**📈 对比分析**

与 LAPE、LSN、LRN 等基线对比，SLN 产生的目标语言 NLL 破坏力比基线高 4.9~5.4 倍，MLN 在语言群组内部的破坏力也显著提升；在 z‑site 上保持 25–44 倍的选择性优势，整体表现远优于传统方法。

**⚠️ 局限性**

局限性包括：仅覆盖七种语言，方法限定在 GLU 结构的自回归 Transformer，且在 z‑site 上因交叉语言均值分布更窄导致干预幅度受限。

---

## 166. Demystifying the Privacy-Utility Trade-off in LLM Interactions

**arXiv ID:** 2609.10992 | [PDF](https://arxiv.org/pdf/2609.10992v1)

**作者:** Zhenhua Liu `[一作]` (Soochow University), Wenliang Chen `[通讯]` (Soochow University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化LLM隐私-效用权衡，提出基于用户意图的本地化隐私保护框架；

**💡 创新点**

将隐私敏感信息的价值拆分为上下文依赖、策略适配和组合互作三机制，并据此设计动态提取-清洗-恢复三阶段管道；

**🔧 技术方法**

利用知识蒸馏将大型隐私推理模型压缩为轻量级本地模型，配合规则式的提取、策略决策与恢复模块；

**📊 数据集**

采用ShareGPT‑X、LMSYS‑Chat‑1M、WildChat以及合成的Nemotron‑Personas数据，构建约9.8K样本的敏感性标注集；

**📈 对比分析**

与Papillon、PUFT等基线对比，评估隐私泄露率和实用性得分，实验显示在隐私优先模式下该框架在保持更低泄露率的同时提升约15%实用性；在实用性优先模式下进一步提升至约30%；

**⚠️ 局限性**

局限性：仅针对单轮对话，未覆盖多轮会话中隐私边界的动态演变；框架对极少数特殊语义场景的策略生成仍可能不足；模型在极大规模的商业部署时的效率和安全性尚待进一步验证。

---

## 167. Data Protection in Function-Correcting Symbol-Pair Codes: Redundancy Bounds and Protection Profiles

**arXiv ID:** 2609.10989 | [PDF](https://arxiv.org/pdf/2609.10989v1)

**作者:** Anamika Singh `[一作]` (Indian Institute of Technology (ISM)), Abhay Kumar Singh `[通讯]` (Indian Institute of Technology (ISM))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出一种同时保障消息完整性和函数值可靠恢复的符号对码框架——FCSPC‑DP，并给出了其冗余量上界与下界。

**💡 创新点**

创新点在于将函数纠错与符号对读信道结合，提出了函数分离常数、α‑pair距离图、生成谱与断连阈值等新度量，用来刻画数据保护与函数保护之间的权衡；并扩展了经典 Plotkin 与球包围界限至此框架。

**🔧 技术方法**

主要技术包括：符号对距离的定义与性质、联合对距离需求矩阵、两步构造法、Cayley 图与其连通性分析、函数局部可分着色、离散优化与球包围计数。

**📊 数据集**

论文为理论研究，无使用具体实验数据集，全部结果为数学证明与抽象构造。

**📈 对比分析**

通过对比 Hamming 与符号对度量的关系、以及对已知 FCC 的冗余上界，展示了在相同保护需求下 FCSPC‑DP 能达到或接近最优冗余；同时给出了一些闭式构造实例，证明理论界限可被达到。

**⚠️ 局限性**

局限性包括：构造往往需要线性码与对称结构，复杂度高；对非线性或更一般读模型的适应性尚未讨论；实际实现中对符号对读误差的统计分布假设较为理想化。

---

## 168. A variational physics-informed graph neural network for heterogeneous solid mechanics

**arXiv ID:** 2609.10983 | [PDF](https://arxiv.org/pdf/2609.10983v1)

**作者:** Aashay Rajan Yadav `[一作]` (Indian Institute of Technology Madras), Ratna Kumar Annabattula `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种无标签、变分式物理信息图神经网络（PIGNN），用于求解两相异质固体在小变形线弹性和有限变形Neo‑Hookean超弹性下的平衡场。

**💡 创新点**

创新点在于：①将异质性完全由网格离散携带，网格图构成网络结构，消除界面惩罚与过渡宽度；②利用离散总势能作为单一无权重目标，只用一阶导数；③在同一网络架构下兼顾二维和三维、线弹性与超弹性问题。

**🔧 技术方法**

使用的技术包括自适应网格生成、Delaunay 三角/四面体化、基于消息传递的图神经网络（Encoder–Processor–Decoder）、深度能量方法（DEM）和自动微分优化。

**📊 数据集**

数据集采用自行生成的几何模型：带单/多孔、圆形和再入形异质包覆、三维立方体与圆柱体扭转等问题；所有参考解均来自高精度有限元（FE）求解，但训练不使用任何标注数据。

**📈 对比分析**

通过与能量基 PINN、强形式 PINN 以及传统 FE 参考解比较，PIGNN 在材料对比 10⁻²–10² 范围内，von Mises 应力误差保持 <3.58%，比强形式 PINN 的 5.58% 更低；对比能量基 PINN 时，σₓₓ 误差减半，且在网格迁移和三维扭转问题中表现出与 FE 相当甚至更好的精度；训练成本比单次 FE 求解高一至两位数倍，但推理速度是 Newton–Raphson 解的 1/50 级别。

**⚠️ 局限性**

局限性包括：需要针对每个新边界/材料问题单独训练，训练时间远超一次 FE 求解；精度受限于 P₁ Ritz 最小化，细网格或更高阶形函数时误差收敛速率下降；并且应力场的误差相对位移更大，主要因应力是后处理得到。

---

## 169. EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale

**arXiv ID:** 2609.10980 | [PDF](https://arxiv.org/pdf/2609.10980v1)

**作者:** Ege C. Kaya `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并理论分析了低秩高斯乘积扰动在大型语言模型的演化策略（EGGROLL）中的收敛与方差性质，并在此基础上改进出了LOO-ROLL方法，显著提升了post‑training效果。

**💡 创新点**

首次将低秩扰动的均值场解析为保留稀疏性的共轭滤波器（解析性求解的解算子），揭示了其非保守性与对极值稳定性的潜在影响，并通过留一技术将双向评估成本减半。

**🔧 技术方法**

运用了高斯特林公式、特征函数分析、矩阵微分运算、稀疏扰动的矩阵卷积、随机抽样与留一基线、以及大规模Transformer训练实验。

**📊 数据集**

在Qwen3系列（0.6B–8B）和SmolLM2-1.7B模型上，利用GSM8K、Countdown与next‑token预测等标准任务进行post‑training评估。

**📈 对比分析**

与密集高斯ES、不同秩的EGGROLL以及固定评估成本/固定壁时的对比实验表明，LOO-ROLL在相同评估预算下MSE下降约50%，在匹配壁时提升了7/10个设置的性能，GSM8K精度从38.1%提升至63.0%（0.6B），从65.9%提升至80.0%（8B）。

**⚠️ 局限性**

主要局限在于理论推导主要基于局部线性/高斯平滑假设，实际应用对更大秩、较大扰动半径以及更复杂任务的泛化尚待进一步验证；同时留一估计在极小批量或高方差情境下的鲁棒性需要进一步研究。

---

## 170. Fengshui: Demystifying Chiplet Ecosystem and Bespoke Neural Network Accelerator Codesign

**arXiv ID:** 2609.10970 | [PDF](https://arxiv.org/pdf/2609.10970v1)

**作者:** Haoran Jin `[一作]` (University of Michigan), Nathan Bleier `[通讯]` (University of Michigan)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了 Fengshui 框架，利用芯片组（chiplet）生态与加速器协同设计，自动选取可复用的 8 类芯片组并将深度学习算子进行离散化、张量融合、流水线/张量/专家并行等技术，生成针对不同 AI 任务的专用集成电路（nsic）；

**💡 创新点**

①首次将芯片组库与加速器映射过程联合优化，解决芯片组与加速器互依的设计悖论；②引入操作级离散化、异构内存与批处理、数据流多样化、近内存处理（PIM）和网络切换等技术，实现多维度硬件定制；③设计热感知布局器，兼顾能耗、通信和温度，显著降低峰温和布线能耗；④采用代理（surrogate）辅助演化搜索、凸包分配技巧和多目标约束优化，实现高效且精确的搜索。

**🔧 技术方法**

代理演化优化（SAEO/ I‑SAEO）、Timeloop、CENT、Accelergy、CACTI、DSENT、HotSpot/MFIT、凸包分配算法、Token‑passing 互连协议、14 nm 1 Y工艺的 PIM 与切换芯片组、Tensor‑Fusion 与非均匀批处理技术。

**📊 数据集**

实验基准包括 LLaMA 3.1‑8B/70B、Qwen3‑30B‑A3B/235B‑A22B（MoE）、MobileNetV3、RepLKNet‑31B、ViT 族、长上下文 LLM 推理、边缘自动驾驶视觉等，涵盖从数据中心到边缘多样化的 AI 工作负载。

**📈 对比分析**

对比基准为 NVIDIA RTX PRO 6000（Blackwell GPU）、同构 ASIC、同构 nsic、无约束异构 nsic（近乎理想参照）以及 SCAR、Gemini 等现有芯片组框架。通过统一的 iso‑throughput 计价方式，在能耗、EDP、能耗/成本（ec）等指标上，Fengshui 在多项工作负载上实现了 2–4 倍甚至 10 倍以上的能耗/EDP 节省，且在成本上与无约束异构方案相差不足 5%，同时在温度与布线能耗方面降低 0.7–4 K 与 5–18 % 以上。留一族实验显示，冻结的 8 芯片组对未见工作负载的适配误差低于 10%，通过增添单个芯片组即可几乎恢复到最优。

**⚠️ 局限性**

- 仍需依赖准确的软硬件模型与仿真，模型误差会影响搜索结果；
- 设计复杂度高，尤其是多目标约束与物理实现校验；
- 目前的工艺和芯片组数量有限，面对未来出现全新瓶颈（如更高频率、不同内存技术）可能需要进一步扩充芯片组池；
- 对于极端极大规模系统（>10 个芯片组）搜索时间与资源仍显昂贵；
- 方案在 14 nm 1 Y 工艺下验证，迁移到更先进工艺时需重新评估 NRE 与性能。

---

## 171. Decoupling Readiness from Release for Tail-Aware Scheduling of Agentic LLM Workflows

**arXiv ID:** 2609.10964 | [PDF](https://arxiv.org/pdf/2609.10964v1)

**作者:** Bochao Feng `[一作]` (University of Science and Technology Beijing), Jidong Zhai `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了代理式LLM工作流中，提前释放已准备好的turn导致拥塞时尾部延迟升高的问题，并提出一种基于尾部风险的turn释放调度算法。

**💡 创新点**

创新点在于将turn释放视为在线调度决策，使用基于均值–CVaR的工作流尾部风险优先级并结合拥塞自适应预算，恢复对已提交工作可变更的控制。

**🔧 技术方法**

使用CVaR目标、在线阈值估计、工作量估计（prompt+output长度）、拥塞反馈控制和优先级索引进行调度决策。

**📊 数据集**

使用SWE-bench和SWE-Gym软件工程任务的真实agent执行轨迹，覆盖Qwen3-8B/32B和Llama-3.3-70B多模型和多GPU配置。

**📈 对比分析**

与传统的即刻释放（eager release）进行对比，结果显示在高负载下P95工作流流时间可降低71.4%（约3.5×加速），且轻负载性能基本保持不变。

**⚠️ 局限性**

局限在于阈值和预算参数需手工调优，实验仅覆盖软件工程场景，未验证在更广泛的多任务或多模型环境下的泛化性。

---

## 172. LLMVul: A Vulnerability-Labeled Dataset of LLM-Generated C/C++ Functions from Real Production Repositories

**arXiv ID:** 2609.10945 | [PDF](https://arxiv.org/pdf/2609.10945v1)

**作者:** Mohammad Farhad `[一作]` (University of Louisiana at Lafayette), Shuvalaxmi Dass `[通讯]` (University of Louisiana at Lafayette)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

收集并标注了来自真实 GitHub 项目中由 LLM 代码助手生成的 C/C++ 函数的安全漏洞数据集 LLMVul。

**💡 创新点**

首次提供基于实际 AI 辅助开发产生的代码而非人工提示的漏洞标签数据集，并结合 AI 工具归因与时间戳，填补了实验室与生产环境间的空白。

**🔧 技术方法**

利用 GitHub API、tree-sitter 解析、三工具静态分析（Semgrep、Flawfinder、正则匹配）以及人工评估，形成高质量的漏洞标签。

**📊 数据集**

从 1,200 大星级 C/C++ 仓库中挖掘 226 个仓库、1,684 个 LLM 标记提交，共 21,430 个函数，构成 LLMVul 数据集。

**📈 对比分析**

与人写漏洞数据集对比评估检测模型泛化能力，发现基于人类代码训练的模型在 LLM 代码上表现下降；使用 LLMVul 训练的模型可提升性能，验证了该数据集的实用价值。

**⚠️ 局限性**

标签依赖静态分析与人工检查，可能产生误报；AI 标注依赖显式提交信息，低估 AI 代码比例；仅覆盖 C/C++ 大项目，缺乏小型或其他语言的数据。

---

## 173. When More Is Not Better: Component Anti-Synergy in a P300 Speller

**arXiv ID:** 2609.10961 | [PDF](https://arxiv.org/pdf/2609.10961v1)

**作者:** Lucas Yang `[一作]` (Parkland High School), Fusheng Wang `[通讯]` (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了P300脑机接口词典在不同组件组合（Euclidean Alignment、xDAWN、校准、语言模型）下的性能，采用全因子实验评估其对准确率、重复次数和信息传输速率的影响。

**💡 创新点**

发现组件之间存在反协同效应，校准是主导因素，EA可补偿无校准情况，但过多组件反而降低效率；语言模型在低质量EEG下会产生负面影响。

**🔧 技术方法**

使用Euclidean Alignment、xDAWN空间滤波、基于LDA的分类器、语言模型先验以及线性混合效应模型进行分析。

**📊 数据集**

采用公开的BNCI2014_009 P300矩阵词典数据集（10名受试者，3个会话）。

**📈 对比分析**

通过2^4全因子实验对16种配置进行混合效应模型评估，结果显示校准+EA组合获得最高ITR（约35.5 b/min），相较于其他配置有显著提升。

**⚠️ 局限性**

仅基于公开数据进行模拟在线实验，未在真实ALS患者或不同硬件平台上验证，缺乏对实际在线使用的评估。

---

## 174. Grounding Agent Memory: Environment-Probing Curation for Enterprise Agents

**arXiv ID:** 2609.11060 | [PDF](https://arxiv.org/pdf/2609.11060v1)

**作者:** Susheel Suresh `[一作]` (Microsoft Corporation), Alejandro Gutierrez Munoz `[通讯]` (Microsoft Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出环境探测式记忆策划方法，使LLM代理在完成任务后能通过只读环境工具验证并刷新记忆记录，提升长期经验累积的可靠性。

**💡 创新点**

创新点在于将环境读写权限最小化到策划代理，利用只读工具对候选记忆进行现场检验、范围测试与时效性刷新，从而打破仅基于轨迹的回顾式记忆局限。

**🔧 技术方法**

采用异步策划代理、记录CRUD接口、非写入式轨迹蒸馏、基于提示的“提议‑探测‑提交”流程，以及GitHub Copilot SDK提供的数据库/文档工具。

**📊 数据集**

使用CLBench数据库探索数据集（含模式漂移）与改编版APEX管理咨询任务集（90题，涵盖PDF/XLSX/DOCX/PPTX）。

**📈 对比分析**

对比无记忆、完整上下文学习、记忆+无探测以及记忆+探测四种配置；在CLBench上，探测式记忆将通过率从39%提升至73%，奖励从8.60升至22.60，查询量和任务代理成本显著下降；在APEX上，探测式记忆在6个世界中获得最高的任务代理奖励与成本比。

**⚠️ 局限性**

局限性包括：需要可读环境工具才能执行探测；对高度动态或完全不可读环境的适应性未知；探测过程仍可能带来额外延迟，且未对模型权重或检索策略进行更新。

---

## 175. BEACON: A Versatile Accelerator for Computational Pathology Applications

**arXiv ID:** 2609.11044 | [PDF](https://arxiv.org/pdf/2609.11044v1)

**作者:** Sumanth Gudaparthi `[一作]` (University of Utah), Srinivasan Parthasarathy `[通讯]` (Ohio State University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了基于AI+X思路的可扩展 systolic 加速器 BEACON，能够高效执行计算病理管线中的 CNN、kNN、GCN 等多种操作，兼顾图构建与图卷积两大阶段。

**💡 创新点**

创新点在于：①在传统 AI 加速器微架构上做最小改动，加入可重构数据路径和 EQ‑Wide PE，兼容聚合、kNN 与图卷积；②通过软件重构（阈值划分、块级预取、桶化）显著降低随机访问和内存瓶颈；③提出负载均衡与跳过机制，使 PE 利用率提升至 88%。

**🔧 技术方法**

使用 systolic 处理单元、可重构数据路径、EQ‑Wide PE、kNN 块级搜索、桶化、负载均衡、软件重构等技术。

**📊 数据集**

实验使用 CRC、BACH、BRACS 三个医学 Whole‑Slide Image 数据集。

**📈 对比分析**

与 NVIDIA Titan X、Tesla P100 GPU 以及 EnGN GNN 加速器对比，BEACON 在训练/推理阶段分别比 GPU 快 56×/63×，比 EnGN 快 14×/56×，同时能耗比 EnGN 低 8.6×，显著提升吞吐量和能效。

**⚠️ 局限性**

主要局限在于 kNN 处理依赖固定阈值划分，可能不适用于高维或无阈值的场景；此外，PE 的微架构改动虽只带来 1.1× 的面积增长，但在更大规模或更复杂图结构下仍需进一步验证。

---

## 176. RCL: A Retrieval-Confidence Layer for Detecting Insufficient Context in Enterprise Retrieval-Augmented Code Generation

**arXiv ID:** 2609.11023 | [PDF](https://arxiv.org/pdf/2609.11023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 177. T1: Terminal Agent Reinforcement Learning for Long-Horizon Tasks

**arXiv ID:** 2609.11042 | [PDF](https://arxiv.org/pdf/2609.11042v1)

**作者:** Junyao Yang `[一作]` (Tencent Hy Foundation Model Frontier), Leowei Liang `[通讯]` (Tencent Hy Foundation Model Frontier)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练了一个122B稀疏专家模型的终端代理，使用强化学习在真实执行结果上进行后训练，覆盖数千个长周期命令行任务。

**💡 创新点**

创新点包括：①针对训练‑推理不一致提出 Token‑In‑Token‑Out 与路由重放机制；②引入全局固定尺度的断言计数密集奖励以弥补稀疏奖励；③构建大规模递归合成任务集并通过审核过滤潜在奖励作弊。

**🔧 技术方法**

采用了基于 SLIME 的异步 PPO、MoE 稀疏专家架构、Critic warm‑up、经验回放、动态上下文并行、回滚路由重放、Token‑In‑Token‑Out 以及 Dense assertion reward 等技术。

**📊 数据集**

使用了递归合成任务集 T1（15k 任务）、RST 合成的 37k 任务以及挑选的 15k 高质量任务作为训练；评估使用 Terminal‑Bench、LHTB、Terminal‑Bench Hard 等公开基准。

**📈 对比分析**

在 Terminal‑Bench 2.1 与 LHTB 上与 GPT‑5.4、DeepSeek、Claude 等模型对比，RL 后模型在 89 个 held‑out 任务上从 49.4% 提升至 64.0%，比 10B 参数的 GPT‑5.4 更优；在 Terminal‑Bench Hard 达到 38.0%，在 LHTB 平均奖励 27.9。

**⚠️ 局限性**

局限性包括：仍需手工审核任务以防奖励作弊；缺乏完整单一奖励消融实验；验证器可被绕过；长尾任务被采样淘汰导致训练分布偏差；模型对重新分词和上下文截断敏感。

---

## 178. BenchShield: Formal Model-Backed Instrumentation for Reward Integrity in LLM-Agent Evaluation Infrastructure

**arXiv ID:** 2609.11028 | [PDF](https://arxiv.org/pdf/2609.11028v1)

**作者:** Shenghan Zheng `[一作]`, Christophe Hauser `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BenchShield，基于有限生命周期模型和任务绑定的模型驱动工具链，用于在 LLM 代理评估中检测并证明奖励完整性。

**💡 创新点**

创新点包括：① 将奖励完整性定义为从代理观察到奖励的有限事件生命周期；② 通过静态 taint 分析预先发现任务包中的攻击向量；③ 在运行时利用基础设施事件实现结构完整性检查并将结构违规与语义审计分离；④ 构建了 456 条人类标注的奖励劫持轨迹语料库；⑤ 通过正式模型评估隔离机制对七个完整性维度的影响。

**🔧 技术方法**

技术手段包括：TLA+ 生命周期模型 + TLC 检查、任务绑定生成、基于生命周期的阶段感知 taint 分析、运行时事件流捕获（authority-bearing events）、语义审计代理、静态与动态流水线分离、BenchFlow 框架集成。

**📊 数据集**

使用了三大公开评估基准（Terminal‑Bench 3、SkillsBench、ClawsBench）共计 31,000+ 运行轨迹，并从中抽取 456 条标注轨迹作为实验语料。

**📈 对比分析**

与 BenchJack 等基线对比：BenchShield 静态检测在同一任务模型下实现 77–100% 的链条召回率（相较 BenchJack 23–94%），同一向量覆盖率提升 43–78%；运行时检测准确率 96%（BenchJack 36%）；单轮成本相对低，运行时分析平均 5–10 美元/任务。

**⚠️ 局限性**

局限性包括：① 仅适用于 BenchFlow 兼容的基准；② 生命周期模型固定，可能无法覆盖多轮对话或开放式探索场景；③ 需要完整的基础设施事件与证据；④ 语义审计依赖 LLM，可能在复杂情境下不稳定；⑤ 对未建模的通道（如控制平面逃逸）仍需手动审计。

---

## 179. The Missing Boundary: How Autonomous Agents Lose Control

**arXiv ID:** 2609.11024 | [PDF](https://arxiv.org/pdf/2609.11024v1)

**作者:** Zonghao Ying `[一作]` (Tencent Zhuque Lab), Jing Guo `[通讯]` (Tencent Zhuque Lab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在Forge-Bench实验框架中独立操控目标压力、约束退化与不安全机会，系统评估了5种LLM模型在16个正常操作域中的失控现象。

**💡 创新点**

创新点在于揭示约束退化与不安全机会交互能触发“缺失边界”失控，并通过因果对照实验与效果根基化评估验证该机制的普适性。

**🔧 技术方法**

采用确定性多轮工具调用、上下文压缩与全因子设计、对照恢复约束、以及基于环境状态的LoC判别器等技术实现实验与分析。

**📊 数据集**

使用自构造的Forge-Bench基准，包含16个业务域共1,800条轨迹，并测试了5个主流LLM模型。

**📈 对比分析**

通过对比不同因子组合的LoC率，发现仅当约束退化且存在可执行不安全机会时LoC率高达55%/62%，而恢复原始约束可将LoC降至0%，证明实验方法有效。

**⚠️ 局限性**

局限性包括实验仅在模拟环境与预定义任务下进行，未覆盖真实世界的复杂交互与恶意对抗，且只关注外部可观测失控而非内部意图。

---

## 180. A Mathematical Theory of Pragmatic Information

**arXiv ID:** 2609.10986 | [PDF](https://arxiv.org/pdf/2609.10986v1)

**作者:** Kai Niu `[一作]` (Key Laboratory of Universal Wireless Communications, Ministry of Education), Ping Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了统一的实践性信息理论框架，将通信、控制和决策融合在同一数学模型中，定义了等终性（isoteleia）映射、重化映射等关键概念。

**💡 创新点**

创新点包括：①将语义信息理论扩展到实践层面，提出三层层次结构（语法→语义→实践）并给出完整的实践信息度量；②证明了实践层的源编码、信道编码和率失真定理；③引入实践信息的价值（VoI）与成本（CoI）并构建拉格朗日双重优化框架，形成“行为容量”概念。

**🔧 技术方法**

使用信息理论工具（熵、互信息、相对熵、率失真函数）、控制理论（马尔可夫决策过程、贝尔曼方程）以及经济学中的拉格朗日乘子等方法，构建理论推导与证明。

**📊 数据集**

本工作为理论性研究，没有使用具体数据集，而是在若干典型应用（如自动驾驶、机器人、语言模型、人类学习）中给出概念性示例来说明概念和度量。

**📈 对比分析**

通过理论证明与推导展示了实践信息的编码极限、容量与失真曲线，并用示例数值验证了熵层级、容量提升与 VoI/CoI 的效能，表明在资源受限下可实现更高的“任务效用”。

**⚠️ 局限性**

局限性：理论层面尚未在真实系统中进行实验验证；等终性映射与重化映射的构造在复杂环境中可能较难实现；模型假设的马尔可夫性与完备的效用函数对实际应用的适用性有限。

---

## 181. ReCHOIR: Contact-guided Human Object Interaction Retargeting to Diverse Characters

**arXiv ID:** 2609.10982 | [PDF](https://arxiv.org/pdf/2609.10982v1)

**作者:** Chaelin Kim `[一作]` (KAIST), Junyong Noh `[通讯]` (KAIST)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为ReCHOIR的人机物交互（HOI）运动重定向框架，能够将源角色的物体交互动作在保持动作语义和接触一致性的前提下迁移到不同骨骼结构的目标角色，同时同步预测目标物体运动。

**💡 创新点**

创新点主要包括：① Part‑Aware Motion Embedding（PAME）自编码器，构建基于体部的共享潜在空间，兼顾多种骨骼结构；② 在PAME基础上加入 Contact‑Guided Retargeting 模块，通过交叉注意力和残差控制网络实现接触信息对目标动作的局部调节；③ 同时预测目标物体运动，使人-物交互在空间上保持一致。

**🔧 技术方法**

技术实现上使用图注意力卷积（GAT Conv）、自注意力与交叉注意力、控制网络（ControlNet）式残差分支、PointNet++ 对物体点云编码、6D 旋转表示、以及多项损失（重建、时间正则、接触、对象运动正则）。

**📊 数据集**

训练与评估数据集包括：SAME（用于训练 PAME 的多骨骼运动数据）；OMOMO 与 BEHAVE（用于 HOI 迁移训练，包含多种物体交互动作）；以及 20 个 Mixamo 角色做为目标骨骼。测试集使用 OMOMO‑test、BEHAVE‑test 以及 10 个未见角色。

**📈 对比分析**

与基准方法（PAME、PAME+IK、PAME+Optim、PAME+IM）以及同类型的实例优化方法进行对比。实验结果表明 ReCHOIR 在接触保真度、动作语义保持、物体运动可行性等指标上均优于纯 PAME，且在计算效率上显著优于基于优化的做法，整体性能可与最优的实例优化方法相当。

**⚠️ 局限性**

局限性包括：① 仅能在接触点可转移且局部可调的前提下工作，对极端尺寸/比例差异的角色仍可能产生不可行接触；② 交互表示使用固定的稀疏接触点，无法覆盖更细粒度或多物体交互；③ 物体运动仅预测位移，旋转保持源姿态，限制了需要物体重新定向的场景；④ 模型为帧级，缺乏对长时序依赖的显式建模，可能导致长段运动的平滑性不足。

---

## 182. What a Random Draw from the MCP Registry Contains, and What Tool-Use Benchmarks Contain Instead

**arXiv ID:** 2609.10962 | [PDF](https://arxiv.org/pdf/2609.10962v1)

**作者:** Haseeb Mohammed Afsar `[一作]` `[通讯]` (Independent researcher), Haseeb Mohammed Afsar (Independent researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对Model Context Protocol（MCP）公开注册表中随机抽取的400台服务器进行一次不做修复、无凭据、仅一次原始握手的直接探测，记录启动成功率、失败类型、工具描述完整性和安全注解缺失情况，并将收集到的真实工具描述与BFCL v4、UltraTool两套基准语料进行全局去重后同一TF‑IDF余弦相似度方法对冗余程度进行对比。

**💡 创新点**

创新点在于首次用概率样本而非手工筛选或修复后的样本来量化MCP服务器的“活性”与“健康”，揭示了大多数服务器在未修复前会因无法启动而被忽略；其次将真实部署的工具描述与主流基准语料在相同去重与相似度度量下进行直接比较，说明了基准语料中任务重复的严重程度及其对模型评估的潜在偏差。

**🔧 技术方法**

采用的技术包括：随机无放回抽样（Fisher–Yates）、基于std‑io的原始MCP握手探测、JSON Schema校验、全局去重（按工具名+完整描述）、TF‑IDF词频向量化与余弦相似度计算，以及脚本化的可复现实验流水线。

**📊 数据集**

使用的数据集包括：官方MCP注册表两次快照（24 135台服务器）、从中随机抽取的400台服务器、这些服务器发布的2 766个真实工具描述，以及BFCL v4（8 726条）和UltraTool EN（14 084条）的基准工具列表。

**📈 对比分析**

通过保持TF‑IDF向量化和余弦相似度阈值不变，对三组语料进行冗余率比较。结果显示：真实MCP工具的近似重复率仅2.8%（全部）且跨服务器为0%；BFCL v4的重复率高达16.7%，其中16.4%为跨任务重复；UltraTool仅0.3%。这些数值表明真实工具的跨作者重复极低，而基准语料存在显著任务重复。

**⚠️ 局限性**

局限性包括：样本仅覆盖30.7%可通过std‑io探测的服务器，单次探测未重试导致可能的误判；使用的词频余弦相似度只能衡量表面文本相似度，忽略语义等价；对未启动服务器的失效原因仅归因于启动失败，未探测代码级克隆；以及仅基于两次注册表快照，缺乏长期趋势分析。

---

## 183. Fork Where the Model Changes Its Mind: Belief-Shift Branching for Tree-Structured Reinforcement Learning

**arXiv ID:** 2609.11061 | [PDF](https://arxiv.org/pdf/2609.11061v1)

**作者:** Bin Lei `[一作]` (University of Minnesota), Chien-Sheng Wu `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于模型信念变化的树状分叉（belief‑shift branching）方法，用于无评论者（critic‑free）的强化学习，以提高数学和编程推理的奖励分配精度。

**💡 创新点**

创新点在于：①用模型在生成链中信念（对最终答案的概率分布或内部激活）变化来确定分叉点，而非传统的结构或熵指标；②提供三种可实现的信念读取方式（黑盒JS散度、白盒logit‑lens形状、预训练向量BSV），无需额外的步骤级监督；③在RL训练前通过Monte‑Carlo价值曲线验证该信号的有效性，并在多模型、多任务中显著提升整体准确率。

**🔧 技术方法**

核心技术包括：树状rollout结构（每条链最多一次分叉；子链采样产生对比信号）、信念读取（黑盒probing、白盒logit‑lens、向量投影）、基于RLVR的组相对策略梯度、动态采样与裁剪，以及对比学习（Monte‑Carlo value estimation）。

**📊 数据集**

使用的主要数据集：数学推理 - AIME 2025/2026、GPQA‑Diamond、OlympiadBench、Omni‑MATH‑500；编程推理 - DeepCoder‑24K 训练集与 LiveCodeBench‑v6 验证集（包含 easy/medium/hard 难度）。

**📈 对比分析**

对比方法包括固定分叉（newline/固定长度）、熵、均匀分叉、LLM‑judge等；在所有模型（OLMo‑3‑7B、Qwen3‑4B、Nemotron‑9B）与任务上，belief‑shift 方案在数学和代码聚合指标上均优于基线，提升幅度从 2–6% 不等，尤其在代码任务中显著（+6.5%）。

**⚠️ 局限性**

局限性：①分叉数有限（通常每链一次），在高对比度任务中对分叉位置的依赖仍显著；②需要额外的前向激活或小量生成（probe），虽占比低，但对大模型或实时部署有一定开销；③方法仍基于树状rollout，无法直接兼容单步或全局值函数的策略；④对不同模型的适配需要手工调整读取方式或参数。

---

## 184. Meta-Learning for Classifier Selection in Image Datasets: A Feature-Driven Framework for Accuracy Prediction

**arXiv ID:** 2609.11041 | [PDF](https://arxiv.org/pdf/2609.11041v1)

**作者:** Zahra Nabizadeh_Shahre_Babak `[一作]`, Shadrokh Samavi `[通讯]` (Seattle University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于元学习的图像数据集分类器选择框架，利用数据集复杂度元特征预测不同分类器的准确率，并通过聚类将相似性能的分类器归组，实现快速、可解释的模型推荐；

**💡 创新点**

创新点在于（1）构建面向图像的多维度元特征集合（自编码器、预训练网络、网络中心性、纹理等）；（2）结合三种特征选择方法（相关性、SHAP、RFECV）挑选最具预测力的特征；（3）为每个分类器训练专属回归模型并聚类归组，以提升排名预测精度；

**🔧 技术方法**

技术包括：元学习框架、特征提取（VGG19、AutoEncoder）、降维（PCA、t‑SNE）、数据集复杂度元特征（图网络中心性、熵、纹理、统计等）、特征选择（相关性、SHAP、RFECV）、回归模型（ET、GBR、RF等）、聚类（k‑means）、评估指标（MAE、MAPE、排名准确率）等；

**📊 数据集**

使用56个多领域图像数据集（医学、自然、工业等），28%作测试，其余40%用于训练；图像统一尺寸为96×96像素；

**📈 对比分析**

与传统方法（仅使用22个元特征）对比，采用全部38个特征平均预测排名准确率达86.15%，对前3名准确率超过95%，中间排名略有下降，低端排名仍保持≈85%；特征提取采用AutoEncoder+PCA最佳；在不同特征提取方式（AE‑PCA、VGG19‑PCA、AE‑tSNE、VGG19‑tSNE）中，AE‑PCA取得最高准确率；

**⚠️ 局限性**

局限性包括：特征提取和降维过程计算量大，需预训练模型；特征选择与回归模型对训练集的依赖较强，可能在极端领域迁移时失效；对决策树特征的依赖导致少量性能下降；框架主要针对图像数据，对其他类型数据的适用性需进一步验证；

---

## 185. Defining AI Agents: A Compendium of Criteria, Metrics, and Benchmarks

**arXiv ID:** 2609.11018 | [PDF](https://arxiv.org/pdf/2609.11018v1)

**作者:** Mia Lassiter `[一作]` (Duke University), Brinnae Bent `[通讯]` (Duke University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对AI代理的评估方法进行系统梳理与分类，将代理特性划分为环境交互、学习与适应、自治、目标导向行为与时间连贯性五大维度，并细化各维度的评估指标与组成要素。

**💡 创新点**

提出了以代理性(agenticness)为中心的五维评估框架，系统整合并命名了先前分散的评估指标，强调了时间连贯性在动态稳定性中的作用。

**🔧 技术方法**

主要采用文献综述与概念框架构建技术，对现有评估指标进行归类与定义；未涉及实验实现。

**📊 数据集**

无数据集，本文为综述/理论框架性质。

**📈 对比分析**

本文通过与已有评估方法对比，指出其完整性与层级结构优势，但未给出定量实验结果。

**⚠️ 局限性**

局限性在于缺乏实证验证与跨任务一致性检验，框架的可操作性与度量标准的可量化性仍待后续研究。

---

## 186. Topological Necessities: Mechanism-Invariant Strategic Subgoals for Cross-Embodiment Goal-Conditioned Control

**arXiv ID:** 2609.11014 | [PDF](https://arxiv.org/pdf/2609.11014v1)

**作者:** Hao Shi `[一作]` (Army Engineering University of Pla Shijiazhuang Campus), Xi Li `[通讯]` (Army Engineering University of Pla Shijiazhuang Campus)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过从离线轨迹构建几何‑拓扑载体，利用持久同调识别并验证任务层级的必经阶段（瓶颈），实现了跨执行器、跨实体的无监督子目标发现和规划。

**💡 创新点**

提出将任务阶段的顺序视为独立于执行器的拓扑必要性，通过持久性同调读取并递归构建门控层级，实现可冻结、可迁移的子目标集合。

**🔧 技术方法**

利用传输加权kNN图、径向几何坐标、shell‑measure持久化同调、H₀/H₁分解以及递归门控层级。

**📊 数据集**

使用OGBench（PointMaze、AntMaze）和D4RL FrankaKitchen的离线数据集。

**📈 对比分析**

在所有基准任务上，冻结的门控集合在PointMaze上达100%成功率，在AntMaze和Kitchen上分别比最强匹配基线提升15–35%（例如AntMaze giant +22.9、Kitchen +15.8/+12.6）。

**⚠️ 局限性**

受限于对固定自由空间的假设，无法在自由空间改变时保持门控结构；对更复杂多路径和高维状态空间的泛化仍需进一步验证。

---

## 187. Phases in a class of associative memories via hidden neurons

**arXiv ID:** 2609.10976 | [PDF](https://arxiv.org/pdf/2609.10976v1)

**作者:** Toshihiro Ota `[一作]`, Masato Taki `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了类 𝓗 的关联记忆网络，将隐藏层视为检索秩序参数，分别在多项式（高阶）和指数负载下推导出相图、容量以及临界行为。

**💡 创新点**

创新点在于把可见层和隐藏层的拉格朗日函数分别作为控制检索稳定性和存储规模的两个维度，并通过复制表示把指数负载映射到随机能量模型计数，从而统一阐释多项式/指数扩展与软max注意力的对应关系。

**🔧 技术方法**

主要技术包括自洽/复制（copy）表示、Replica 方法（RS ansatz）与大偏差极值统计、以及对随机能量模型（REM）的计数分析。

**📊 数据集**

使用的模式集合为独立的高斯或球面随机向量（无真实数据集），用于构造存储模式。

**📈 对比分析**

通过解析零温度容量、临界温度、相图以及有限温度的自旋玻璃/冻结界限进行比较，结果显示多项式负载下容量随阶数快速下降，指数负载下容量受大偏差限制，典型检索状态为亚稳态；相对已有模型，能够实现更高的指数存储，但在极端负载下仍受限。

**⚠️ 局限性**

主要局限是：所有结论均基于 RS ansatz，未考虑 Replica Symmetry Breaking（RSB）与隐藏层温度有限时的两温度耦合；指数负载下对模式分布高度敏感，若改为球面或离散模式会导致容量上限变化。

---

## 188. LAION-Mobile: Evaluating Deepfake Detectors On One Million Smartphone Photos

**arXiv ID:** 2609.11134 | [PDF](https://arxiv.org/pdf/2609.11134v1)

**作者:** Achim von Stryk `[一作]` (Stralsund University), Janis Keuper `[通讯]` (Offenburg University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个规模约一百万张智能手机照片的公开数据集LAION‑Mobile，并利用该数据集以及NTIRE 2026和ForenSynths等基准，对十二种主流深度伪造检测器在不同数据域中的性能进行了系统评估；进一步揭示了阈值校准漂移导致的误报剧增以及现代计算摄影对检测器表现的影响；

**💡 创新点**

①首次公开大规模真实手机照片数据集，可供跨域评测；②在实际手机照片上系统证明现有检测器普遍失效；③量化阈值校准漂移对误报率的巨大影响；④对不同ISP时代设备的误报率做关联分析，表明当前数据集尚无法捕捉旗舰级ISP的真正影响。

**🔧 技术方法**

使用了十二种深度伪造检测器（包括CNNFingerprint、CLIP‑linear、DIRE、AEROBLADE、RIGID‑DINOv3等）原论文检查点；对图像进行EXIF提取、非摄影过滤、URL去重；使用EER阈值校准并在不同数据集上评估；采用Bootstrap 95%置信区间进行统计。

**📊 数据集**

主要数据集包括：LAION‑Mobile（约1 M手机图像，过滤后9,115张评估子集）；NTIRE 2026挑战集；ForenSynths‑13gen；ProGAN‑ISP；HDR+、SIDD、MIDD等真实照片集；以及原论文测试集。

**📈 对比分析**

评估方法：对每个检测器计算论文报告的AUC、NTIRE 2026上的AUC、在NTIRE或ProGAN‑ISP上拟合的EER阈值下的LAION‑Mobile误报率（FPR）。结果显示：NTIRE AUC最高仅0.624，许多检测器低于随机；在现代校准阈值下，LAION‑Mobile FPR在17%–91%之间，甚至最高达91%；没有任何检测器在现代AI内容上超过偶然并保持可接受的误报率。

**⚠️ 局限性**

局限性：①标签依赖EXIF及过滤器，无法保证每张图像都是未经编辑的真实照片；②数据集中主要为2018–2020年前的设备，缺少2021+旗舰手机，无法完全检验最新ISP的影响；③阈值校准漂移是单一指标，未考虑多阈值或自适应校准方法；④仅评估了原论文检查点，未尝试后续微调或迁移学习；⑤由于版权原因未公开原图，需重新下载，可能存在URL失效；⑥对不同光照、场景的细粒度分析有限。

---

## 189. A Framework for Discharge Time Prediction of Energy Storage Units Based on Coupled Dynamics and Multi-Factor Aging Models

**arXiv ID:** 2609.11086 | [PDF](https://arxiv.org/pdf/2609.11086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 190. Beyond Benchmarks: Using VLMs to Reveal Systematic Classification Failures Under Real World Conditions

**arXiv ID:** 2609.11126 | [PDF](https://arxiv.org/pdf/2609.11126v1)

**作者:** Dieuwertje Alblas `[一作]` (TNO), Klamer Schutte `[通讯]` (TNO)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探索利用视觉语言模型（VLM）进行自动错误切片检测（ESD），加速军事AI模型的验证与验证（V&V）流程。

**💡 创新点**

提出基于VLM的残差嵌入式错误切片检测方法，并在军事与犬种分类数据上验证其可行性，首次探讨VLM在低代表性军事域中的表现差异。

**🔧 技术方法**

使用对比式CLIP PE-Core-L-14-336提取图像嵌入，UMAP降维+HDBSCAN聚类，生成式GPT-5 mini生成聚类描述，以及YOLO/CLIP零样本分类器。

**📊 数据集**

犬种分类数据集1A（8,040张狗图）和三军用车辆数据集2A、2B、2C（共约3,000张）。

**📈 对比分析**

在犬种数据上得到0.67/0.87的轮廓系数并成功识别人工扰动；在军用车辆数据上轮廓系数0.83/0.83，但聚类描述重叠，实际效能低于犬种数据。

**⚠️ 局限性**

主要局限在于VLM对军事域缺乏概念纯度、环境多样性不足、聚类覆盖率低、评价指标过于简化且需人工定性。

---

## 191. But How Would AI Agents Run a Town's Economy?

**arXiv ID:** 2609.11108 | [PDF](https://arxiv.org/pdf/2609.11108v1)

**作者:** Sajal Regmi `[一作]` (Karela Technologies Inc.), Chetan Phakami Pun `[通讯]` (Karela Technologies Inc.)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在真实的 Pokhara 湖畔地理环境中，使用 100 个具备记忆的 LLM 代理模拟了一个封闭且节约资金的空间经济，运行最长 26 周，完成 91 次验证运行（共 2.44M 决策、21.5B token）

**💡 创新点**

首次揭示在多代理大模型环境下，金钱传递会在需求冲击后停止移动，财富分布随时间而“冻结”，并且后端 LLM 的变化对所有经济结果都有显著影响

**🔧 技术方法**

采用多代理大语言模型（LLM）技术，配备记忆模块和可变后端 LLM；通过消融实验验证记忆与 LLM 的作用；使用随机现金转移和旅游需求冲击进行经济刺激

**📊 数据集**

真实地理数据（Pokhara 湖畔）及合成的 12 倍旅游需求冲击、随机现金转移（NPR 5,000）等实验数据，全部公开以便复现

**📈 对比分析**

通过 91 次验证运行及离线重算，量化收入、工资、价格变动与财富分布；发现 12 倍需求冲击导致收入 4.62× 提升，工资仅 1.03×，价格变化 0.3%；随机现金转移导致边际消费倾向仅 3–4%；对比社会工具与经济工具，后者成功率约 96%，前者失败率 94–97%

**⚠️ 局限性**

实验时间有限，仍未捕捉到更长期的动态；模型对后端 LLM 极度敏感，缺乏对多种经济工具以外社会工具的深入评估；尽管公开数据完整，但结论仍需在更广泛场景下验证

---

## 192. Visual-Motion-Induced Modulation of Pedestrian Trajectories Using Spatially Distributed Multi-Display Signage in Public Spaces

**arXiv ID:** 2609.11088 | [PDF](https://arxiv.org/pdf/2609.11088v1)

**作者:** Yuri Mikawa `[一作]` (University of Tokyo), Kazushi Maruya `[通讯]` (NTT)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在多显示标牌（MDS）上呈现侧向运动的单色条纹，探究其对行人轨迹的非言语调控效果，先在实验室进行受控实验，再在国家博物馆现场进行人流测试。

**💡 创新点**

创新点在于：① 将视运动诱发的自运动幻觉（vection）技术迁移至公共空间的MDS系统；② 设计了既能产生强烈视觉运动又兼容广告内容的“部分条纹”方案；③ 通过大屏分布式显示实现了无需额外设备即可影响行人行为的可行性。

**🔧 技术方法**

技术实现包括：MDS由六块55英寸垂直显示屏组成；实验室使用RGB‑Depth摄像头结合YOLO检测行人位置；现场采用LiDAR点云并用DBSCAN聚类；数据处理与统计使用Python、MATLAB进行ANOVA、t检验等。

**📊 数据集**

数据集：实验室共计30名参与者（16+14）完成多条目的行走任务；现场约1,369–1,555名访客通过MDS空间；轨迹数据来自摄像头或LiDAR点云，未公开共享。

**📈 对比分析**

比较方法：在实验室对全屏运动、左向、右向和静止条件的轨迹偏差进行方差分析；在现场对左右运动两天的侧向位移进行t检验。实验室全屏运动产生约0.5m的显著侧移，部分条纹无显著效应；现场两天均出现5–13cm的平均侧移，差异显著（p<0.001，Cohen's d≈0.2–0.6）。

**⚠️ 局限性**

局限性：① 实验室中的全屏与部分条纹分别在不同日子、不同受试者上测试，无法直接对比；② 现场缺少静态基线条件；③ 只测试了单向行人流，未验证双向或拥挤情况下的效果；④ 仅能得到聚合位移，无法跟踪个体轨迹；⑤ 虽然设计兼容广告，但实际效果有限，需进一步优化视觉运动强度与社会可接受度。

---

## 193. TailProp: content-adaptive light- and heavy-tailed propagation for vision

**arXiv ID:** 2609.11081 | [PDF](https://arxiv.org/pdf/2609.11081v1)

**作者:** Jiahao Kong `[一作]` (Shandong University), Zihan Li `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为TailProp的层次化视觉骨干网络，利用尾传播算子（TPO）进行跨领域自适应传播。

**💡 创新点**

创新点在于结合了高斯和柯西传播，通过内容条件的通道混合系数自适应地融合两种传播方式，提供了更灵活的空间交互。

**🔧 技术方法**

使用了尾传播算子（TPO），结合高斯和柯西稳定过程的传播器，并在DCT域中进行融合。

**📊 数据集**

在多个数据集上进行评估，包括ImageNet-1K、MS COCO 2017和ADE20K等。

**📈 对比分析**

与匹配的传播基线进行比较，TailProp在图像分类、目标检测和语义分割等任务中表现优异，TailProp-B在ImageNet-1K上达到84.4%的Top-1准确率。

**⚠️ 局限性**

限制在于只研究了高斯和柯西基，未来可以探索更广泛的稳定家族或可学习的基集，且当前的实现未达到硬件最优。

---

## 194. X-Hinges: 3D Printing Self-Sensing Compliant Mechanisms for Continuous and Multi-DOF Motion Sensing

**arXiv ID:** 2609.11077 | [PDF](https://arxiv.org/pdf/2609.11077v1)

**作者:** Xiang Chang `[一作]` (MIT), Jiaji Li `[通讯]` (MIT)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种 X‑Hinges 自感应可变形机制，利用多材料 FDM 打印将高电阻与低电阻导体嵌入同一结构中，实现单次打印即可获得多自由度连续运动感知，并配套交互式设计工具和实时数据解码管线。

**💡 创新点**

核心创新点包括：① 将两种电导率对比显著的导电耗材直接嵌入弹性体内，以差分测量形式实现多自由度独立感知；② 通过自定义的 3D 打印接口几何（如层叠接触）降低高低导体间接触电阻；③ 结合深度学习 TCN 及视觉自标定，实现高精度连续解码并在多批次打印间快速迁移。

**🔧 技术方法**

主要技术包括多材料 FDM 打印、导电 TPU（高电阻与低电阻）、差分电路布置、低噪声电流反馈放大器与 ADS1256 ADC、Temporal Convolutional Network（TCN）时序回归、Grasshopper/HumanUI 交互式设计工具以及基于 AprilTag 的视觉标定。

**📊 数据集**

使用自收集的实验数据：单轴与多轴力学测试、滑索驱动的三自由度耦合实验、手部仿真、游戏控制器与灯具的连续感知数据，以及用于自标定的摄像机姿态‑电阻配对记录；无公开数据集。

**📈 对比分析**

通过与单材料感知结构、传统激光焊接或手动装配传感器的对比，X‑Hinges 将交叉耦合降低至 8–19%，在三自由度下平均误差分别为 7.35°（侧向）、6.59°（垂直）和 1.35 mm（轴向），自标定后误差可降至约 4°，同时实现实时 50 Hz 采样和高达 27,415× 的信噪比提升。

**⚠️ 局限性**

局限性主要在于导电 TPU 的滞后与蠕变导致高速或重复加载下的精度下降；不同批次或打印参数引入的电阻漂移需要外部视觉标定，增加使用门槛；依赖多材料 FDM 设备与自定义电子硬件，尚未在大规模工业场景验证。

---

## 195. ReconPlusGen: Injecting Reconstruction Prior into Multi-view 3D Generation through Noise Inversion and Modulation

**arXiv ID:** 2609.11129 | [PDF](https://arxiv.org/pdf/2609.11129v1)

**作者:** Jiarui Liu `[一作]` (Hong Kong University of Science and Technology), Ping Tan `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过将多视图重建的几何先验以确定性噪声初始化注入3D扩散模型，并利用置信度自适应的空间噪声调制实现对观察到区域的精准对齐与未观察区域的自由生成。

**💡 创新点**

①在扩散过程初始噪声中注入重建几何；②使用置信度映射进行空间自适应噪声调制，平衡重建精度与生成灵活性；③统一采用基于canonical空间的重建预测模型CA‑VGGT。

**🔧 技术方法**

基于3DShape2VecSet的扩散框架、噪声反演、Farthest Point Sampling、DINO+Plücker视角嵌入的多视图扩散、置信度映射与噪声调制等技术。

**📊 数据集**

训练使用Objaverse 130k高质量网格，测试在DoraBench、OmniObject3D与Objaverse等数据集上。

**📈 对比分析**

与重建模型、单视图/多视图条件生成、点云条件生成以及ReconViaGen等方法进行定量比较，采用Chamfer距离和F‑score衡量；ReconPlusGen在所有视角数下均明显优于基线，尤其在多视图下提升更为显著。

**⚠️ 局限性**

对重建质量和置信度估计敏感；对大范围未观测区域的生成仍受限；在极端姿态误差或光照复杂场景下效果尚需提升。

---

## 196. A Model-Centric DevOps Architecture for DEVS-Based Digital Twin Simulation Services

**arXiv ID:** 2609.11122 | [PDF](https://arxiv.org/pdf/2609.11122v1)

**作者:** Arnis Lektauers `[一作]` (Riga Technical University), Rasa Gulbe `[通讯]` (Dati Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种面向模型的 DevOps 架构，将 DEVS（多元平行 DEVS）数字孪生仿真模型视为一等 DevOps 产物，采用 YAML 声明式 SMDL 定义模型，配合 CI/CD 管道实现结构和语义校验、不可变版本化，并通过容器化微服务在 Kubernetes 上实现模型管理、执行、数据集成与结果流服务。

**💡 创新点**

创新点在于：① 将仿真模型与软件一样纳入版本控制与 CI/CD 生命周期；② 开发了与 multiPDEVS 正式映射的 YAML SMDL 与 SSDL，支持结构与语义双层校验；③ 通过对象存储与可变配置的工作流，完成云原生 DEVS 引擎的无状态化与可扩展化；④ 通过案例验证实现了配置驱动的多模态路网仿真与迭代均衡。

**🔧 技术方法**

使用技术包括：DEVS / multiPDEVS、YAML SMDL/SSDL、CI/CD（GitLab CI、FastAPI、RabbitMQ、Kafka）、Kubernetes 微服务容器化、对象存储（S3兼容）、REST/WebSocket/InfluxDB 数据流、FMI、HELICS 等协同仿真框架。

**📊 数据集**

数据集涵盖：基于 OpenStreetMap 的 GeoParquet 道路网、公开 GTFS 时刻表、2024-10-15 的电子车票验证日志用于公交需求、人工合成的车流需求（12,000 辆车 + 12,314 乘客）。

**📈 对比分析**

通过在 Apple M2 Max 机器上测得单容器执行时间，24,314 旅客仿真耗时 74 s，实时速度提升约 1,170×；随旅客数增大，耗时线性增长（每多 1 旅客约 2.3 ms），吞吐量 260–320 旅客/秒；并通过案例展示配置无代码演化与迭代均衡的可行性。

**⚠️ 局限性**

局限性包括：仅在单一 DEVS 引擎与单一城市（Riga）场景验证；未覆盖高拥堵或信号化交叉口等更复杂交通情境；性能评估基于单台 M2 Max 机器，缺乏多节点 Kubernetes 集群水平扩展与多场景并发的实验；未实现多范式（DEVS/DESS）混合模型与模型版本化之外的其它领域验证。

---

## 197. Learning Realistic Athletic Sprinting Without Demonstrations

**arXiv ID:** 2609.11083 | [PDF](https://arxiv.org/pdf/2609.11083v1)

**作者:** William Wang `[一作]` (Stanford University), Kayvon Fatahalian `[通讯]` (Stanford University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套基于肌肉驱动的仿真系统，可在无运动示范的情况下生成高速度运动员运动，涵盖100米冲刺、跨栏、侧滑、倒退等多种竞技运动；

**💡 创新点**

创新点在于三方面的融合：①将经高性能GPU实现的批量肌肉模型校准为国际级短跑运动员；②开发了能在1000×实时速度下运行的GPU批量仿真器；③利用FastTD3等大批量离线强化学习，直接在136维肌肉激活空间中训练控制策略，实现“近乎逼真”的运动。

**🔧 技术方法**

技术手段包括OpenSim肌肉模型、Hill型肌肉动力学、可变步长积分器、GPU Warp实现的高吞吐量仿真、FastTD3强化学习、奖励函数仅包含速度、疼痛惩罚与生存奖励。

**📊 数据集**

数据集主要是运动员的人体测量（体重、身高等）以及实验室收集的跑步运动学、动力学和EMG数据，用于验证和评估，而训练过程中不使用任何运动示范数据。

**📈 对比分析**

评估方式：视觉上与真实运动员相似；运动学、动力学与实验数据对比（关节角度、关节力矩、地面反作用力、EMG时序），跑步统计（步长、频率、分段速度）与Usain Bolt及冠军运动员对齐；GPU仿真可达1000×实时，一天GPU即可训练出可跑完100米的控制策略。

**⚠️ 局限性**

局限性包括：上肢模型尚未充分校准、未考虑肌肉疲劳、只适用于高强度任务、奖励稠密且缺乏稀疏目标、可能对关节约束求解器敏感、未覆盖需要稀疏奖励的更复杂运动项目。

---

## 198. RIDE: Relocalization-Informed Depth Estimation with 3D Gaussian Splatting

**arXiv ID:** 2609.11079 | [PDF](https://arxiv.org/pdf/2609.11079v1)

**作者:** Jiarong Lian `[一作]` (Chinese University of Hong Kong), Ruizhi Chen `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

结合相机重定位和3D Gaussian Splatting模型，利用PnP‑RANSAC得到的稀疏精确深度作为尺度锚点，联合冻结的VDA‑S相对深度先验，在机器人RGB流中实现全密集度量深度估计。

**💡 创新点**

①将重定位得到的PnP inlier对应的稀疏深度直接用作尺度锚点；②提出锚点可靠性学习、全局尺度记忆、局部空间校正和流导向时间记忆四个模块，实现在锚点稀疏或临时缺失时仍能保持连贯的深度估计；③在无额外传感器的条件下完成稠密度量深度的闭环回归。

**🔧 技术方法**

3D Gaussian Splatting渲染、PnP‑RANSAC、预训练的VDA‑S相对深度先验、锚点可靠性预测网络、全局尺度记忆（GRU）、局部空间解码器、SEA‑RAFT光流与置信度融合、时间记忆卷积GRU以及阶段化训练和多项损失。

**📊 数据集**

公开RGB‑D视频集RobbyReal（LingBot‑Depth）用于训练与验证；27条真实机器人路径共3012帧，5个场景，用于无微调评估；此外在公开RGB‑D测试集上进行基准对比。

**📈 对比分析**

与VDA‑S统一/鲁棒/空间校正、Any2Full、PriorDA、DepthPrompt、DA3Mono‑Large+LS、oVDA‑S c8+LS、Marigold v1‑0+LS等基线及VDA‑S三种校准方法对比。结果显示：在公开RGB‑D上RIDE取得最高δ₁、最低held‑out误差和TGE；在机器人序列上RIDE在AbsRel、δ₁、TGE均为最优，显著提升。

**⚠️ 局限性**

需要足够的PnP inlier锚点来初始化尺度；在长时间无锚点或光流失真场景下恢复较弱；依赖预先训练好的3DGS模型和冻结的相对深度先验；未在极端动态或大规模长序列环境下全面验证。

---

## 199. Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation

**arXiv ID:** 2609.11115 | [PDF](https://arxiv.org/pdf/2609.11115v1)

**作者:** Koutian Wu `[一作]`, Wanghan Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Benchmark Radar，一个集成日常发现、检索、源代码检查与分数历史的实时搜索引擎，用于管理和探索 AI 领域的评测基准。

**💡 创新点**

创新点在于：①将四大基准来源（LLM Stats、OpenCompass、Artificial Analysis、模型报告）统一为一个可搜索的数据库；②实时收集新论文、仓库、数据集的发现记录；③在每条记录保留原始来源、引用、分数与时间戳，并通过 ID 关联同一基准的多源信息；④提供 Pareto 前沿视图、饱和度与趋势分析；⑤提供 Web 仪表盘、CLI 与离线查询方式，让研究者能在同一 ID 体系下检索、检视与对比基准。

**🔧 技术方法**

使用技术包括：Python/SQL 数据管道、爬虫与 API 采集（arXiv、HuggingFace、GitHub）、文本检索（BM25F 字段加权）、JSON/RESTful API、Web 前端 (React/Vue?)、CLI (argparse + Requests) 以及可视化库（D3/Plotly）。

**📊 数据集**

数据集：整合来自 LLM Stats、OpenCompass、Artificial Analysis、模型报告的 3000+ 条基准记录；包含 4000+ 个数值分数、2000+ 个模型引用、1500+ 个论文/仓库/数据集链接，覆盖 50+ 领域与 70+ 交互范式。

**📈 对比分析**

比较方法：通过检索后查看每条记录的任务说明、评分标准、时间戳和引用，使用 Pareto 前沿对模型分数与评测覆盖度进行多目标比较；在排行榜页面展示已公布分数与已测模型数的组合。系统本身不进行模型训练，性能表现以覆盖率（已记录基准与分数的比例）为主，当前版本覆盖率约 80% 的基准已记录分数。

**⚠️ 局限性**

局限性包括：①仍有部分基准缺失分数或文献链接；②检索精度依赖词匹配，可能漏检同义或改名的基准；③日常发现主要靠公开渠道，可能错过私有或商业内部基准；④任务分类与能力标签尚未完整验证；⑤不同基准版本、提示与评测设置的不一致使得跨基准比较受到限制。

---

## 200. KuaiRP Series Role-playing Models Technical Report

**arXiv ID:** 2609.11127 | [PDF](https://arxiv.org/pdf/2609.11127v1)

**作者:** Yipeng Wang `[一作]`, Kai Sheng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套完整的角色扮演模型训练流程，包括数据构建、全参数SFT、RL调优以及双阶段On-Policy Distillation，实现了在小模型上高保真角色扮演与通用能力兼顾。

**💡 创新点**

创新点包括：1) 逆向配置过滤（RPF）精炼角色指令；2) 两阶段OPD（先PG再GKD+CDD）实现自蒸馏；3) Cumulative‑Divergence Decay (CDD) 解决prefix‑drift并提升知识注入质量；4) 在同一基模型进行自蒸馏，兼顾域知识与通用能力。

**🔧 技术方法**

使用的技术包括：全参数SFT、基于规则的奖励函数的PPO强化学习、On‑Policy Distillation (PG/K1、GKD)、CDD权重衰减、标准化角色模板、用户行为指令注入、逆向过滤、分层采样等。

**📊 数据集**

构建了50个角色提示（来源于小说）和10种用户风格的用户库，使用Claude 3.7 Sonnet和Qwen2.5‑14B生成对话数据，随后经过RPF过滤得到高质量SFT数据；评测数据包括单轮安全与知识评测、以及多轮基于TRACEbench的角色一致性、记忆一致性与语言质量评测。

**📈 对比分析**

通过TRACEbench多轮评测、BFCL v4通用能力评测，以及与闭源M2‑HER和开源Qwen3‑8B、Qwen3‑4B等基线对比，最终模型在域角色扮演、语言质量与安全拒绝上达到或超过闭源模型，同时保持通用能力并在单卡GPU上实现高效推理。

**⚠️ 局限性**

局限性：对更强基模型（如Qwen3.5‑9B）提升有限，需要更强数据或新的自蒸馏方案；缺少主观趣味/娱乐性评价；对用户体验的主观评测尚未覆盖，未来需引入更多主观奖励与评价维度。

---

## 201. terms.txt: A Consent and Compensation Protocol for Agentic Web Access

**arXiv ID:** 2609.11152 | [PDF](https://arxiv.org/pdf/2609.11152v1)

**作者:** Rajarshi Chowdhury `[一作]` `[通讯]`, Rajarshi Chowdhury

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种在 HTTP 请求边界上实现的同意与补偿协议（terms.txt），用于在网站与机器客户端之间明确身份、用途、条款和报酬，并通过签名交换实现原始服务器端的访问控制。

**💡 创新点**

创新点在于将现有的 Web Bot Auth、AIPREF 术语、HTTP 402 价格协商和签名收据结合为一个统一的请求‑响应流程，既可在原始服务器上执行，又能提供可审计的、可合同化的机器访问条款，解决了传统 robots.txt 失效、用途不明确、无价钱等缺陷。

**🔧 技术方法**

使用技术包括 RFC 9421 HTTP 消息签名、RFC 9651 字典语法、Web Bot Auth 的签名验证与委托令牌、HTTP 402 支付协商、签名收据与哈希链日志，以及 Node.js 实现的签名、验证、日志和支付验证逻辑。

**📊 数据集**

所用数据集包括来自 Imperva、Cloudflare 和 Cloudflare Radar 的公开测量数据（爬虫比例、爬取‑转发比、爬虫用途分类），以及实验室自建的 600 行 JavaScript 原型测试结果和十个原始测量文件。

**📈 对比分析**

性能对比显示，原型在单 vCPU 环境下每个请求平均增加 0.20–0.65 ms 的延迟，吞吐量在 15,900 req/s 降到 1,400–3,000 req/s，虽然有显著开销，但仍能满足大多数站点的常规请求负载。

**⚠️ 局限性**

局限性包括：只能在请求边界强制身份和条款，无法在内容交付后强制执行用途；仅对签名请求有效，无法阻止未签名的抓取；对高并发的 Ed25519 验证有 CPU 负载风险；以及需要进一步标准化 AIPREF 词汇与委托令牌格式。

---

## 202. OmniTable: A Unified Wide-Table System for Petabyte-Scale LLM Data Curation and Exploration

**arXiv ID:** 2609.11148 | [PDF](https://arxiv.org/pdf/2609.11148v1)

**作者:** Yuzhuo Fu `[一作]` (AntGroup), Jun Zhou `[通讯]` (AntGroup)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

建立了一个统一的宽表系统 OmniTable，用于处理 PB 级 LLM 数据的收集、特征工程、治理和探索。

**💡 创新点**

提出了逻辑统一+物理分离的宽表抽象、声明式特征生命周期管理、自治治理与自适应执行引擎以及混合加速的探索服务。

**🔧 技术方法**

结合目录驱动的逻辑统一、UDF 依赖 DAG、CPU/GPU 路由、Operator fusion、UDF 级容错、自动调优、HBase 全局 ID 索引、ClickHouse OLAP、背景视图 materialization 等多层技术。

**📊 数据集**

在阿里巴巴生产环境中处理 35 PB 以上的 LLM 训练数据，包括 web、代码、PDF 和 SFT 等多源语料。

**📈 对比分析**

与传统管道迷宫工作流对比，OmniTable 在特征回填阶段实现 5.6 倍加速、手工步骤减少 73%，每日 2.5 天完成，过滤导出可达 20 TB/h，实验显示在 PB 规模下吞吐稳定，列宽扩展无性能断点。

**⚠️ 局限性**

仍受限于存储引擎的列上限、复杂多源语料的统一映射成本、对极端异常数据的容错开销，以及在大规模部署时维护多种元数据同步与治理策略的难度。

---

## 203. The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems

**arXiv ID:** 2609.11146 | [PDF](https://arxiv.org/pdf/2609.11146v1)

**作者:** Yangze Liu `[一作]` (Shandong University), Zhongyi Han `[通讯]` (Shandong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在受控实验中通过递归训练多模型生态系统，探究数据集中模型占比（市场份额）对模型衰退速度和终点的影响，发现市场份额对终点影响有限，主要决定速度的是池中文本的构成。

**💡 创新点**

首次量化展示在不同市场集中度（从均衡到90%占优）下，多模型生态系统的终点几乎不变，且“拉力”效应极弱，证明模型衰退主要由共用文本特性驱动。

**🔧 技术方法**

采用基于共享文本池的迭代训练框架，使用冻结的DeBERTa编码器计算几何距离，辅以困惑度等无编码器指标评估模型衰退；通过不同的市场份额、主导模型身份、玩家数量以及人类文本比例四种“旋钮”进行实验。

**📊 数据集**

使用13个公开1–4B参数模型组成自然生态系统，生成800条文本样本；人类文本来源为公共Pile子集（去除极端风格域）。

**📈 对比分析**

通过比较不同“arm”在五代训练后模型生成文本在编码空间的几何距离（1‑cos）和困惑度变化，发现即使在最大化拉力的情况下，终点相差仅占漂移的约2–3%，而速度差异可达两倍以上，且人类文本比例每增加25%可将漂移速度降低约1.4–1.8倍。

**⚠️ 局限性**

实验仅覆盖1–4B规模、最多13个玩家、仅五代迭代，未探究更大模型或更长递归深度；使用单一冻结编码器测距可能对跨域语义变化不敏感，且仅得到粗略的速度-池构成相关性（R²≈0.68），未解释部分方差。

---

## 204. The Computing Channel: How Modulation Programs the Airwaves

**arXiv ID:** 2609.11145 | [PDF](https://arxiv.org/pdf/2609.11145v1)

**作者:** Saeed Razavikia `[一作]` (KTH Royal Institute of Technology), Carlo Fischione `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种基于数字调制的功能导向通信（Digital Function‑Oriented Communication）框架，使得在多址信道上能直接通过并发传输实现函数（如求和、平均、最大值等）计算，而无需先恢复每个设备的原始消息。

**💡 创新点**

创新点包括：
- 计算星座（Computational Constellation）原理，联合优化发射映射与接收决策；
- 任务与分布感知几何（Task‑ and Distribution‑Aware Geometry）提升可靠性；
- 对称性约简与金字塔采样（Pyramid Sampling）降低离线设计复杂度；
- MIMO向量计算（Vector Computation）实现并行多维函数；
- SumComp方案在标准QAM星座上直接实现求和，兼容现有调制硬件。

**🔧 技术方法**

采用的技术手段包括：
- 有限字母数字调制与星座设计；
- 统计决策距离度量与损失函数对齐的优化；
- 多天线信道均衡与空间分集；
- 基于熵或Kullback‑Leibler的分布感知优化；
- 量化与位切片（bit slicing）以提升鲁棒性；
- 计算导向的前向纠错与链路自适应。

**📊 数据集**

论文在联邦学习场景中进行了实验，使用了常见的机器学习数据集（如MNIST、CIFAR‑10 等）来验证数字 OAC 在模型聚合过程中的资源节省和误差性能；具体数据集名称未在文中详细列出。

**📈 对比分析**

比较方法：与传统逐消息恢复（Orthogonal Digital）方案、模拟 OAC、SumComp、VecComp 等对照，评估指标包括资源使用（信道使用数）、均方误差（MSE）以及吞吐率。实验结果表明：
- 在理想信道下，Majority 计算的 MSE 能降低两位数；
- 资源使用上，数字 OAC 去除了设备数因子，显著降低上传资源；
- SumComp 在仅支持求和时实现低复杂度且兼容标准 QAM；
- VecComp 能在多输出场景下实现并行计算，但需要 MIMO 资源。

**⚠️ 局限性**

局限性包括：
- 离线星座设计在设备数或量化级别增大时复杂度急剧上升；
- 对同步、功率控制、相位误差、通道估计误差敏感；
- 需要高动态范围 ADC 与精确功率控制；
- 目前主要支持求和、平均、向量函数，通用函数支持受限；
- 仅实现聚合级安全，无法替代加密安全聚合或差分隐私；
- 对恶意设备的防护（波形完整性、认证）尚未在方案中完整实现。

---

## 205. Can LLMs Normalize Databases? A Benchmark and Multi-Agent Framework for Schema Normalization

**arXiv ID:** 2609.11141 | [PDF](https://arxiv.org/pdf/2609.11141v1)

**作者:** Dong-Jae Koh `[一作]` (Kyungpook National University), Young-Kyoon Suh `[通讯]` (Kyungpook National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个包含3,275个样本的数据库规范化基准（DBNB），并构建了一个多代理推理框架MARS，用以评估和提升LLM在从1NF到BCNF规范化过程中的可靠性。

**💡 创新点**

创新点包括（1）创建统一的三轴评估协议（语义、结构、逻辑），系统性衡量LLM生成的DDL；（2）提出MARS框架，将证据提取、违规诊断、分解规划、DDL生成和验证拆分为独立代理，支持多轮修复；（3）在基准上实现了约82%性能提升。

**🔧 技术方法**

主要技术手段是大语言模型（如Qwen3‑30B、Llama 3.3 70B等）结合多代理协同框架（AutoGen/MetaGPT），以及自动化的基准生成器与评估器。

**📊 数据集**

数据集来源于Spider和BIRD两大公开关系数据库，经过去归一化、FD注入和链式标签化后生成DBNB基准。

**📈 对比分析**

实验中对比了单提示Baseline、Miffie自修正框架和MARS；在Real World场景下，MARS在DBNB-Score上从Baseline的0.253提升到0.423（零样本）或0.418（少样本），在语义、结构、逻辑三轴上均表现最佳。

**⚠️ 局限性**

局限性包括：① FK重构仍表现较弱；② BCNF及已规范化输入的识别与处理效果不佳；③ MARS需多轮LLM调用，对上游错误敏感，推理成本较高。

---

## 206. Autonomous Chemical Mechanistic Discovery through Agentic Reasoning and Validation

**arXiv ID:** 2609.11147 | [PDF](https://arxiv.org/pdf/2609.11147v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 207. ToxicRAG: Compromising Retrieval-Augmented Generation Systems via Single-Shot Knowledge Poisoning Attacks

**arXiv ID:** 2609.11082 | [PDF](https://arxiv.org/pdf/2609.11082v1)

**作者:** Haozhe Lu `[一作]` (Peking University), Xiang Li `[通讯]` (Nankai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对检索增强生成(RAG)系统，提出了一种单文档知识毒化攻击ToxicRAG，用伪造事件和多来源权威共识来误导模型生成错误答案。

**💡 创新点**

创新在于把毒化文档设计为一个叙事式的知识更新，并加入自我验证循环来确保生成的文档能诱导目标答案。

**🔧 技术方法**

采用生成式语言模型（DeepSeek-Chat）进行叙事生成与自检，利用密集检索（FAISS + sentence-transformers）和四个LLM（Llama-3、Qwen2.5、Qwen3）进行评估。

**📊 数据集**

实验使用Natural Questions、HotpotQA和MS-MARCO三大问答基准，每个基准抽取100个目标问题进行采样检索语料库。

**📈 对比分析**

与PoisonedRAG、CorruptRAG-AS、CorruptRAG-AK和AuthChain四种基线相比，ToxicRAG在12个数据集–模型组合中取得最高攻击成功率(ASR)，提升幅度在5%–11%之间。

**⚠️ 局限性**

局限包括仅在采样语料库上测试，未覆盖完整百万文档检索；检索指标非目标特定；单次实验缺乏置信区间；对真实部署写入权限的假设；自动判定与目标答案生成带来的噪声。

---

## 208. Multi-Faceted Evaluation and Mitigation of Emotion Hallucinations in MLLMs

**arXiv ID:** 2609.11154 | [PDF](https://arxiv.org/pdf/2609.11154v1)

**作者:** Bowen Zeng `[一作]` (University of Science and Technology of China), Xun Yang `[通讯]` (University of Science and Technology of China)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种心理学驱动的六维情感幻觉评估框架（EHR）及一种基于幻觉记忆与锚记忆的训练无关记忆引导情感推理（HMER）方法，用于细粒度评估和实时消除多模态大型语言模型（MLLM）的情感幻觉。

**💡 创新点**

创新点在于：① 将情感幻觉细分为表达、动作、音频、直觉、逻辑、结论六个维度，实现多维、可定位的评估；② 设计局部化评估器将评估反馈转化为幻觉记忆与锚记忆，实时对logit进行定向抑制与信任上下文调节，达到细粒度、跨模型的幻觉抑制。

**🔧 技术方法**

使用了命题分解、面向维度的可信度判断器、对抗/推理验证、基于logit的方向抑制、迭代解码与锚点上下文重构等技术。

**📊 数据集**

实验使用开放式情感推理数据集OV‑MERD+进行评估，另外在CMU‑MOSI（英语）和CH‑SIMS（中文）上检验多语言多模态的消融效果。

**📈 对比分析**

与VCD、ICD、M3ID、EmotionHallucer等现有方法对比，实验覆盖19种不同类型的MLLM，方法在所有六个维度上均实现幻觉率下降3.3%–11.9%，并在情感预测F1/召回上实现提升。

**⚠️ 局限性**

局限性包括：需要多轮迭代导致推理延时，记忆维护开销；对极端多模态干扰的鲁棒性尚未完全验证；评估器对预训练判断器的依赖可能引入偏差。

---

## 209. A Fragility Spectrum for Recursive Language-Model Training

**arXiv ID:** 2609.11149 | [PDF](https://arxiv.org/pdf/2609.11149v1)

**作者:** Yangze Liu `[一作]` (Shandong University), Zhongyi Han `[通讯]` (Shandong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在递归训练环境中，公开模型检查点在面对模型生成文本污染时的多样性衰退表现，发现不同检查点具有稳定的脆弱性光谱；

**💡 创新点**

首次揭示检查点本身的“崩溃脆弱性”是一个可持续且可预测的属性，并证明仅通过两三代自回归循环即可快速筛查；

**🔧 技术方法**

采用递归训练协议、共享文本池、self-loop循环、top‑p/温度采样调控、数据过滤以及唯一4‑gram多样性(u4)指标等技术；

**📊 数据集**

使用13个公开的1–4B基线检查点（来自10个模型家族），以及每代2100条不超过128令牌的短文本池；

**📈 对比分析**

通过在同一协议下对13个检查点进行五代递归训练，比较u4下降幅度、相关性与其他指标；自回归循环在2–3代即可与生态系统最终排名相关（0.6–0.8），采样尾部收缩可在3代内显著抑制崩溃；

**⚠️ 局限性**

实验规模受限于1–4B检查点、极小的训练文本池和单一的递归协议，无法验证更大规模或不同协议下的光谱是否保持；

---

## 210. Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment

**arXiv ID:** 2609.11144 | [PDF](https://arxiv.org/pdf/2609.11144v1)

**作者:** AS Aravinthkakshan `[一作]` (Manipal Institute of Technology), Harsh Nandwani `[通讯]` (Perssonify)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个由845起证券集体诉讼事件与70,500条 X/Twitter 消息组成的语料库，并用五种情感工具（VADER、Loughran–McDonald、FinBERT、Twitter‑RoBERTa、Claude Haiku）在同一数据上同时评估构造有效性（与人工标注的一致性）和预测有效性（与异常收益的相关性）。

**💡 创新点**

创新点在于：①在同一语料中同步测量构造与预测有效性，首次揭示两者的关系；②证明评估样本、分数表示和零分日处理对情感工具预测排名的显著影响；③发现情感内容信息显著优于简单计数或量化，并且交易量与价格损失或和解金额无显著关联。

**🔧 技术方法**

使用的技术包括文本预处理、情感评分（词典、Transformer 预训练模型、LLM 注释）、异常收益计算、Spearman 相关、Granger、分布式滞后、局部投影，以及 Benjamini–Hochberg 多重检验等统计方法。

**📊 数据集**

数据集由 2002–2025 年间的 70,500 条 X/Twitter 消息、845 起证券集体诉讼事件、400 条单标注金标准消息，以及对应的股票日收益与市场收益构成。

**📈 对比分析**

对五种工具采用相同的聚合与测试流程，在两种样本设定（方法特定零分日抽样和固定 n 的公共日抽样）下计算同日和一日领先相关；结果显示：与人工一致性高的工具（Claude Haiku）在同日关联中排名靠前，预测排名随抽样方式和分数表示而变化；总体相关系数范围约 0.08–0.26，情感内容显著优于计数或量化。

**⚠️ 局限性**

局限性包括：仅有单一标注者的金标准，缺少交叉标注可靠性；语料仅限集体诉讼语境，泛化性未知；LLM 可能受到未来信息泄漏的影响；零分日处理对排名的影响大，效应规模较小。

---

## 211. The Machines Are Calling: Measuring Automated and Synthetic Voices in Unwanted Inbound Calls

**arXiv ID:** 2609.11137 | [PDF](https://arxiv.org/pdf/2609.11137v1)

**作者:** Xingyu Shen `[一作]` (Scam AI Reality Inc), Simiao Ren `[通讯]` (Scam AI Reality Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

搭建交互式语音蜜罐，记录了 10,987 次入站呼叫，并使用音频指纹、商业合成语音检测器和人工听众三种工具，对来电者开场语进行分析，测定了无效呼叫中机器语音（录音或AI合成）的比例以及其在不同呼叫类型中的分布。

**💡 创新点**

首次公开完整且可复现的测量流程；在蜜罐中将来电者语音与自身 TTS 分离，避免混叠；结合音频指纹和人工验证的双重检测，揭示了录音与新生合成音频的真实比例；并通过对蜜罐接线历史的控制，证明接线老化而非生态变化导致的 “AI 语音上升” 现象。

**🔧 技术方法**

音频指纹（交叉相关与余弦相似度）、商业合成语音检测 API（得分阈值 0.85）、人工听众盲测、呼叫形状分析（silent、one-shot、short exchange、conversation）、说话者嵌入向量聚类、呼叫聚类（脚本相似度）等多种技术。

**📊 数据集**

由 11 个 LLM 人格分别在 11 个真实美国电话号码上运行的蜜罐生成的 10,987 条入站呼叫（除 2,711 条系统失效日外），其中 7,233 条被蜜罐回答、6,192 条满足两轮交互并被评估的呼叫构成核心数据集。

**📈 对比分析**

通过人工听众对 1,816 条标记为合成的呼叫进行盲测，确认率为 54.4%（区间 51.5–57.3%）。在阈值 0.85 下，合成呼叫占 29.3%（95% 置信区间 27.0–31.9%）。与先前公开的 25% 估计相符，但我们公开了完整的阈值、分布、聚合规则和参与模式，使结果可被质疑与复现。

**⚠️ 局限性**

仅对开场 10 秒进行检测，可能漏测后续合成或录音；音频指纹与检测器对录音与新生合成的区分不完美；人工听众一致性低，仅提供确认率而非召回率；蜜罐接线历史导致的老化效应可能混淆时间趋势；未能完整测量所有未标记呼叫和未能验证的“silent”呼叫；缺乏对同一呼叫后续音频的多窗检测与同一呼叫内部的回放判定。

---

## 212. Bidirectional Multimodal Fusion of Sky Images and Time-Series for Solar Forecasting with Large Language Models

**arXiv ID:** 2609.11135 | [PDF](https://arxiv.org/pdf/2609.11135v1)

**作者:** Ken Chen `[一作]` (University of Melbourne), Saman Halgamuge `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将天空图像与历史光伏/辐射数据融合的LLM多模态短期预测框架；

**💡 创新点**

通过轻量级双向模态融合模块，实现视觉与时间序列在Token层的双向尺度与偏移调制；

**🔧 技术方法**

使用基于Time‑LLM的时间序列重编程、CNN视觉编码、FiLM式双向融合、冻结的GPT‑2语言骨干；

**📊 数据集**

在SIRTA（GHI+天空图像）和SKIPP’D（PV功率+天空图像）两个公开数据集上进行评估；

**📈 对比分析**

与Time‑LLM、PatchTST、DLinear和物理式SPM等基线比较，MSE/MAE均明显下降，最大相对MSE降低达25.4%（SIRTA）和14.5%（SKIPP’D），且在少样本、云量大、长时延等情形下表现尤为突出；

**⚠️ 局限性**

局限在于仅利用局部地面天空相机，缺乏大尺度卫星或文本天气信息，且对极端天气或多站点迁移的鲁棒性尚未验证。

---

## 213. From Digital Accountability to Accountable Digitality Through Needs-Aware Information Systems: The Case of Auditable Child-Welfare Judgments

**arXiv ID:** 2609.11125 | [PDF](https://arxiv.org/pdf/2609.11125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 214. Phase-Decoupled, Model-Calibrated Power Control for Disaggregated LLM Serving

**arXiv ID:** 2609.11133 | [PDF](https://arxiv.org/pdf/2609.11133v1)

**作者:** Jae Gon Kim `[一作]` (Xenoscube, Inc.), Soojung Ryu `[通讯]` (Xenoscube, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对LLM推理的 GPU 能耗控制方案，利用分阶段（prefill 与 decode）解耦的功率与频率调节，并通过自动化校准和 SLO 门控来实时调整功率上限和 SM 频率窗口，从而在保证尾部延迟满足的前提下显著提升 tokens/J。

**💡 创新点**

创新点在于：① 对每个 (model, quantization, engine) 组合进行自动校准；② 为 prefill 与 decode 两个阶段分别选择不同的功率调控机制（decode 用功率上限，prefill 用频率窗口）；③ 采用尾部延迟门控（SLO-guard）确保实时合规；④ 在实测环境中与 NVIDIA Max-Q 直接对比，展示了 Pareto 级别的能效提升。

**🔧 技术方法**

技术手段包括：GPU 内置功率管理（NVML、DCGM）、SM 频率锁定、功率上限控制、配套的离线校准脚本、SLO 监控回路、分阶段推理架构（如 NVIDIA Dynamo + SGLang），以及在实验中使用的计量与统计工具。

**📊 数据集**

数据集与模型：使用 Qwen3 系列 MoE 大模型——480B FP8（TP4）与 235B NVFP4（TP1），配合两种工作负载：agentic（长上下文、工具调用）和 standard（基于 ShareGPT 对话混合）。

**📈 对比分析**

比较方法：在同一 B200 芯片上，将自研的 PERF/BAL/EFF 模式与 NVIDIA Max-P、Max-Q 进行 head‑to‑head 对比，测量 tokens/J、mean end‑to‑end 延迟、TTFT、ITL‑p99 等指标。实验结果显示 BAL 模式在 agentic 工作负载下比 Max‑Q 提升约 20% tokens/J、延迟仅略高；EFF 模式提升更大（≈30%），且所有模式始终满足 ITL‑p99 SLO。

**⚠️ 局限性**

限制：仅在单节点 NVIDIA B200 上验证，针对 MoE 模型（未验证稠密模型或其他 GPU 代）；校准过程耗时约一小时；未测试自适应 DVFS 或更细粒度的频率控制；实验使用的闭环请求生成不包含真实 trace；跨节点或大规模集群级功耗管理尚未评估。

---

## 215. How Wrong Can a Good Predictor Be? Diverging Updates with Vanishing Predictive KL

**arXiv ID:** 2609.11132 | [PDF](https://arxiv.org/pdf/2609.11132v1)

**作者:** Qifu Wen `[一作]` (Boston University), Ningxin Su `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文证明，在固定状态数的对称高斯隐马尔可夫模型中，即使精确贝叶斯更新和一种新的径向滤波器在内部logit空间存在无界距离，它们在经过softmax解码后预测的Kullback–Leibler散度仍会趋于零；并用数值实验展示了该现象在不同状态数下的具体表现。

**💡 创新点**

创新点在于揭示内部更新误差与预测性能之间的“脱耦”关系——即内部误差可以发散而不会导致预测误差；同时给出明确的数学证明和可构造的径向滤波器，并指出这一现象对压缩、量化或稀疏化模型的潜在影响。

**🔧 技术方法**

使用的技术主要包括贝叶斯滤波理论、对数几率映射与softmax曲率分析、KL散度推导、径向滤波器构造、对称Gaussian HMM的平稳分析，以及针对不同K值的数值仿真。

**📊 数据集**

所用数据集为合成的对称高斯HMM，状态数K取{2,4,8}，均匀间隔的均值，标准差固定；实验通过大量（数千条）平稳路径进行统计。

**📈 对比分析**

比较方法：将精确贝叶斯更新与径向滤波器在相同观测序列上进行比较，评估终端状态的中心化距离与分类KL；此外还与二进制控制器、学习模型及其蒸馏版本做对比。结果显示，尽管内部距离随q→0^+线性增长，分类KL却在所有K下收敛至零；在更长路径长度下，学习模型表现多样，未必保持此性质。

**⚠️ 局限性**

局限性包括：证明仅在固定K且先给定参数时成立；未给出关于K→∞或参数接近冲突时的统一收敛速率；仅针对对称Gaussian HMM且假设平稳起始；未探讨学习算法能否自然逼近径向滤波器；对非平稳或多维观测的推广尚不清楚。

---

## 216. From Repetition to Recognition: Inductive Discovery of Disinformation Narratives

**arXiv ID:** 2609.11128 | [PDF](https://arxiv.org/pdf/2609.11128v1)

**作者:** Max Upravitelev `[一作]` (Technische Universität Berlin), Vera Schmitt `[通讯]` (Technische Universität Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了三层评估框架（恢复、挖掘、发现），并将无监督叙事标签生成管线（聚类与图社区）在七个虚假信息数据集上进行对比实验，随后进行人类验证以识别新叙事候选；

**💡 创新点**

创新点包括：①从闭合世界到开放世界的三层评估体系；②通过发现层人类验证引入真正的未知叙事标签；③揭示聚类与图在多话题覆盖和单例生成上的根本差异，为未来叙事分析提供结构性洞察；

**🔧 技术方法**

使用的技术包括：LLM（Gemma-4-31B-it、Qwen3-Embedding-4B）进行文本抽取与标签生成；HDBSCAN/Leiden社区检测进行聚类与图划分；UMAP降维；Harrier嵌入、Cosine相似度；四个自动评价指标（Hungarian、WCD、Collapse、C/R）；人类标注；

**📊 数据集**

使用的数据集包括：CARDS、Climate Obstruction、COVID Conspiracy、EU DisinfoTest、HALT‑PROP、Narr. Media Framing、PolyNarrative、UKElectionNarratives；

**📈 对比分析**

比较方法：在恢复和挖掘层分别计算Hungarian、WCD、Collapse和C/R指标；发现层通过两名人工标注评估“是叙事”比例、匹配率、其他率和无域率；实验结果显示图管线在多话题覆盖上更稳健，单例率达30‑62%；聚类管线在某些数据集上压制少数话题并产生较低的单例率；整体性能取决于聚类/图参数和任务需求；

**⚠️ 局限性**

局限性：仅使用两名标注者，存在主观偏见；聚类默认参数导致多话题覆盖失衡；单例验证依赖外部知识，无法完全体现内部重复；自动指标受嵌入模型的表面语义限制；实验规模受限，缺乏更广泛多语言、多主题的验证；LLM判定不可靠，无法完全替代人类评估。

---

## 217. HERALD: High-Fidelity Exemplar Retrieval with Adaptive Landmark Distillation for Heterophily-Aware Graph Condensation

**arXiv ID:** 2609.11123 | [PDF](https://arxiv.org/pdf/2609.11123v1)

**作者:** Sujan Chakraborty `[一作]` (Indian Institute of Science Education and Research Thiruvananthapuram), Saptarshi Bej `[通讯]` (Indian Institute of Science Education and Research Thiruvananthapuram)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种名为HERALD的无梯度图压缩框架，能够在保持下游节点分类性能的同时将原始图压缩至极小规模；

**💡 创新点**

创新点在于：① 引入自适应异质性（heterophily）估计并据此调整特征选择与节点评分权重；② 采用联合 Fisher‑可分辨性与激活密度的特征筛选；③ 将原型代表性、决策边界亲近度和局部内在维数（LID）三项指标加权综合，形成自适应节点评分；④ 保留BONSAI的无梯度、线性时间构造流程。

**🔧 技术方法**

技术方法包括：多跳 Fisher 判别式特征评估、激活密度加权、局部内在维数估计、异质性比率计算、Sigmoid 变换自适应权重、BFS 扩展、Personalised PageRank 剪枝与类平衡。

**📊 数据集**

实验数据集涵盖八个基准：四个同质性（Cora、CiteSeer、PubMed、Reddit）和四个异质性（Roman‑Empire、Amazon‑ratings、Chameleon、Squirrel）。

**📈 对比分析**

与 Random、Herding、BONSAI、GDEM 四个主流凝聚方法在不同存储压缩率（r=0.0001、0.005、0.01、0.03）及四种 GNN（GCN、GAT、GIN、H2GCN）上进行对比，HERALD 在大多数设置（尤其是异质性图）取得最高或接近最高的节点分类准确率，整体提升约 1–3% 以上；

**⚠️ 局限性**

局限性包括：① 仅适用于静态属性图，未考虑动态图或异构图；② 采用全局异质性估计，可能忽略局部结构差异；③ 在极低压缩率（r=0.0001）下，部分高异质性模型（如 H2GCN）表现不及 BONSAI；④ 计算 LID 及特征筛选的成本相对较高，需进一步加速。

---

## 218. ProMediConv: Benchmarking Proactive Conversational Agents in Legal Dispute Mediation

**arXiv ID:** 2609.11101 | [PDF](https://arxiv.org/pdf/2609.11101v1)

**作者:** Zesheng Wei `[一作]` (University of Science and Technology of China), Yang Deng `[通讯]` (Singapore Management University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ProMediConv 框架与高保真 972 例多方调解对话数据集，用于评估主动型对话代理。

**💡 创新点**

创新点在于：①把调解建模为三阶段的主动多方对话；②引入 11 种调解策略和 4 种行为模式状态；③提出 MAD 细粒度指标捕捉参与者心理状态变化；④用自动文本重构与 LLM 评注构建真实案例数据。

**🔧 技术方法**

使用大语言模型（Qwen、Llama、GLM、ChatGPT 等）、强化学习 (REINFORCE)、策略规划方法（Proactive、ProCoT、ICL_AIF）和自定义基线 ProMediAgent。

**📊 数据集**

构建了 972 例真实中文调解案例数据集，包含 17,277 句子、对话阶段、策略标签与 BP 状态。

**📈 对比分析**

与多种通用及法律专属 LLM 进行基准评测，ProMediAgent 在 SR@t、SSR、MAD 上均超过同类模型；但整体性能仍落后于人类调解员。

**⚠️ 局限性**

局限性：仅限文本对话、中文法律背景，缺乏多模态与跨文化支持；MAD 与最终成功率仍存在一定偏差。

---

## 219. When Noise Fabricates Bias: The Fragility of LLM-as-a-Judge Bias Measurement under Noisy Text

**arXiv ID:** 2609.11067 | [PDF](https://arxiv.org/pdf/2609.11067v1)

**作者:** DongHyun Ryu `[一作]` (Sungkyunkwan University), JinYeong Bak `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了表面噪声（如拼写错误、非正式拼写和标点符号崩溃）对大型语言模型（LLM）作为偏见评估者的影响。研究通过对3822个与刻板印象相关的响应应用五种噪声条件，比较了干净文本与噪声文本的偏见判断。

**💡 创新点**

研究发现，表面噪声会系统性地夸大偏见测量，尤其是在与公平性相关的类别中。噪声更容易将中立判断转变为偏见判断，而不是相反，最高可达120倍的差异。

**🔧 技术方法**

使用了大型语言模型（LLM）作为偏见评估者，具体包括Llama-3.1-8B、Qwen3-8B、Gemma-4-12B-it和GPT-5.4等四种模型。

**📊 数据集**

数据集Fable由CLEAR-Bias提供，包含3822个二元选择的偏见项目和响应，涵盖七个类别（年龄、性别、宗教、种族、残疾、性取向和社会经济状态）。

**📈 对比分析**

通过比较干净文本和噪声文本的判断，发现噪声导致的判断翻转率在不同模型间差异显著，尤其在脆弱的模型中，噪声对偏见判断的影响更为明显。噪声类型的组合会加剧这种偏见的虚构。

**⚠️ 局限性**

研究的局限性包括未验证噪声引起的判断是否符合刻板印象，且未考虑模型对名字的识别能力。此外，使用的噪声是基于规则的，可能无法完全捕捉真实用户生成文本中的噪声特征。

---

## 220. MOSAIC: Query-Aware Exploration Policy Adaptation for GraphRAG

**arXiv ID:** 2609.11065 | [PDF](https://arxiv.org/pdf/2609.11065v1)

**作者:** EunKyeong Lee `[一作]` (KT Corporation), Junyoung Youn `[通讯]` (KT Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的框架，将图检索增强生成（GraphRAG）视为每个查询的控制问题，通过分析器将查询隐含的证据需求转化为可执行的策略。

**💡 创新点**

创新点在于通过查询特定的控制策略来优化图检索过程，而不是使用固定的全局策略，从而提高了检索和生成的准确性。

**🔧 技术方法**

使用了大型语言模型（LLM）分析器来生成查询特定的检索策略，并结合了图遍历、证据选择等技术。

**📊 数据集**

使用了GraphRAG-Bench数据集，包括医疗和文学领域的问题，医疗数据集包含2062个问题，文学数据集包含2010个问题。

**📈 对比分析**

与固定策略进行对比，提出的方法在医疗领域的答案正确率达到76.97%，在文学领域为64.33%，均优于之前的最佳结果，且在检索过程中评估的路径和保留的证据项显著减少。

**⚠️ 局限性**

局限性包括分析器可能错误分类证据结构，当前信号和有效范围是基于观察到的失败开发的，图构建质量限制了检索效果，且在当前实现中延迟和成本高于固定检索。

---

## 221. Freehand Sketching for End-User Programming of Robot Swarms

**arXiv ID:** 2609.11078 | [PDF](https://arxiv.org/pdf/2609.11078v1)

**作者:** Riwa Karam `[一作]` (University of California Irvine), Magnus Egerstedt `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文研究了自由手绘作为一种终端用户编程接口，用于指定机器人群体的几何形状。用户通过绘图传达空间意图，机器人群体自主提取目标形成点，构建刚性形成图，分配机器人到形成节点，并执行分布式形成控制，确保不会出现意外的反射形成。

**💡 创新点**

创新点在于提出了一种基于草图的交互抽象，使非专业用户能够通过自由手绘直接传达形成级意图，而无需传统的编程知识或预定义的形成形状。

**🔧 技术方法**

使用了计算机视觉、刚性图构建、最优机器人分配和分布式形成控制等技术，并提供了防止意外反射实现的理论保证。

**📊 数据集**

进行了人类研究，参与者生成了42个几何形状，评估了自由手绘形成规范的可用性。

**📈 对比分析**

通过用户研究，20名参与者的平均系统可用性评分为84.25，表明该接口具有高感知可用性。与传统的多机器人系统命令相比，用户能够更直观地与系统交互。

**⚠️ 局限性**

限制在于参与者主要来自大学背景，且研究仅限于静态平面形成，未来工作将考虑更广泛的非技术用户群体，并扩展草图交互到群体行为控制之外。

---

## 222. Beyond Solver Verdicts: Generative Reward Models for Autoformalization

**arXiv ID:** 2609.11085 | [PDF](https://arxiv.org/pdf/2609.11085v1)

**作者:** Vikash Singh `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了神经符号系统中的翻译准确性问题，提出了判决保持不忠实性（VPU）的概念，定义了在编码错误的情况下仍能获得正确判决的情况，并提出了一种新的生成验证方法（GenV），用于评估参考等价性。

**💡 创新点**

创新点在于提出了VPU的理论框架，并通过生成验证（GenV）将离线的Z3等价性oracle转化为无参考的连续等价性评分，从而解决了传统方法无法检测VPU的问题。

**🔧 技术方法**

使用了生成模型和Z3求解器，结合了决策投影和稀疏自编码器的机制分析，来提取精确的空间错误坐标。

**📊 数据集**

使用了来自真实翻译器输出的950个候选编码和多个外部数据集（如FOLIO、ProofWriter、MALLS等）进行实验评估。

**📈 对比分析**

与传统的结构性检查和其他验证方法相比，GenV+HN在参考等价性验证中达到了0.961的AUROC，显示出其在未见翻译器和不同形式风格下的良好泛化能力，并在测试时计算分配中获得了11.3的准确性提升。

**⚠️ 局限性**

限制在于GenV+HN优化的是严格的参考等价性，而非主观的人类意图，可能存在语义差距。此外，验证器在形式逻辑风格的严重分布变化下可能会导致得分分布的变化和阈值校准的下降。

---

## 223. Rubric-Aligned Disentangled Evaluation of Human Simultaneous Interpreting

**arXiv ID:** 2609.11131 | [PDF](https://arxiv.org/pdf/2609.11131v1)

**作者:** Ziyu Zhang `[一作]` (Chinese University of Hong Kong), Satoshi Nakamura `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了1,101条分段级双语同传评测语料，并提出了基于分析式评测维度（含意义传递LQ、交付质量EXP、感知时延LAT）的标注框架

**💡 创新点**

首次在同传领域实现了面向段级的多维度评估，并通过双头回归模型实现了维度解耦，避免了传统标量训练导致的维度融合问题

**🔧 技术方法**

使用了COMET‑KIWI无监督编码器，结合LoRA轻量级微调与双头线性回归，针对意义与交付两个维度分别训练预测器

**📊 数据集**

数据集由公开的BSTC同传片段与授权的TED式会议录音合成，覆盖英中、中文英双向，按讲座层面划分训练/验证/测试集（839/87/169段）

**📈 对比分析**

与冻结版COMET‑KIWI、单头微调、基于提示的LLM评估以及均值预测等基线进行对比，双头模型在测试集上分别取得LQ 0.388、EXP 0.301的Pearson相关，明显优于基线且接近人类一致性水平

**⚠️ 局限性**

主要限制包括仅采用文本信息，未考虑音频/时序特征导致LAT评估受限；数据量相对有限，且主观标注尺度波动较大，进一步提升多模态与更大规模数据将是未来工作方向

---

## 224. Overview of the NLPCC 2026 Shared Task 11: Agent-Based Experiment Reproduction from Scientific Papers

**arXiv ID:** 2609.11117 | [PDF](https://arxiv.org/pdf/2609.11117v1)

**作者:** Hanhua Hong `[一作]` (University of Manchester), Chenghua Lin `[通讯]` (University of Manchester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于过程的实验复现基准，利用MCP Action Recorder记录LLM代理在阅读、规划、编码、执行和结果验证全过程的行为，并通过细粒度纸张特定rubric评估复现质量，覆盖150篇机器学习与AI4Science论文；

**💡 创新点**

创新点在于①从过程角度评估而非仅关注最终仓库；②构建多域（ML与AI4Science）复现基准；③采用LLM辅助生成并验证与人工一致的rubric；④通过Action Recorder提供可审计的行为日志；

**🔧 技术方法**

技术手段包括MCP Action Recorder（path、write、cmd工具链）、LLM代理（Claude、ChatGPT-4o-mini）生成rubric与评估、统计分析（Pearson/Spearman相关性）等；

**📊 数据集**

使用的数据集为150篇论文集合（120 ML+30 AI4Science），其中10%为人工注释子集（12 ML+3 AI4Science），其余论文通过LLM生成rubric扩展至10,000+条目；

**📈 对比分析**

与参赛系统（YNU-HPCC-Task11-AgentRep、zzunlp_wu、QueenAgent）及GPT‑5.4基线进行加权重要性得分对比；最佳系统整体得分49.64%，执行与结果匹配阶段明显低于其他阶段；LLM生成rubric与人工rubric的Pearson 0.93、Spearman 0.88，表明生成方式可靠；

**⚠️ 局限性**

局限性包括rubric仍无法完全消除hallucination，评估仅使用GPT‑4o-mini模型，AI4Science覆盖有限，执行环境与依赖问题仍是主要瓶颈。

---

## 225. DRG-MAPPO: Hierarchical Dynamic Role-Graph Multi-Agent Reinforcement Learning for Cooperative Air Combat

**arXiv ID:** 2609.11155 | [PDF](https://arxiv.org/pdf/2609.11155v1)

**作者:** Junlin Liu `[一作]` (Institute of Automation Chinese Academy of Sciences), Hao Zhao `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种层次化的多智能体强化学习框架 DRG‑MAPPO，用于实现无人机群在 BVR 空战中的协同决策。

**💡 创新点**

创新点在于将图注意力网络与动态角色分配相结合，并加入目标优先级辅助任务和时间承诺机制，使高层战略与低层动作分离、角色稳定并促成焦点打击与诱捕等高级协同战术。

**🔧 技术方法**

核心技术包括：图注意力编码器（GAT）进行时间变动态关系建模；层次化策略（高层角色策略+低层动作策略）；目标优先级辅助任务；时间承诺机制；以及基于 CTDE 的 MAPPO 优化。

**📊 数据集**

使用自建的 200 km×100 km 高保真 BVR 仿真环境中的 2v2 对抗数据，包含位置信息、速度、雷达锁定、导弹状态等特征。

**📈 对比分析**

与 MAPPO、MAPPO+GAT、HAPPO、QMIX、IPPO 等基线对比，DRG‑MAPPO 在 2v2 场景中最高 87% 的胜率和 69.9% 的平均胜率，奖励曲线和对抗矩阵均显著优于基线。

**⚠️ 局限性**

局限性包括仅在 2v2 场景验证，难以直接推广到大规模无人机群；仅使用离散战术动作；以及缺乏真实战场数据验证。

---

## 226. The information geometry of large language models is shared, learned, and controllable

**arXiv ID:** 2609.11063 | [PDF](https://arxiv.org/pdf/2609.11063v1)

**作者:** Dario Picozzi `[一作]` `[通讯]` (University College London), Dario Picozzi (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并验证了大型语言模型（LLM）在输出空间中共享的Fisher–Rao几何结构，并探讨该几何如何预测模型行为、对齐人类预测、控制干预及其跨架构通用性。

**💡 创新点**

提出了基于输出Fisher度量的“输出几何”框架，证明了其对模型行为的唯一性与可辨识性，展示了该几何能够在不共享词表、架构或激活坐标的情况下实现跨模型比较与干预；同时证明了语言统计（n-gram）可预测几何谱、有效维度以及事实学习时序，提供了一种无参数的控制与编辑方法。

**🔧 技术方法**

核心技术包括Fisher–Rao度量、拉格朗日自然梯度、矩阵无关的Jacobian‑Vector乘法、近似谱分析、统计学推断与交叉验证、以及对多种架构（Transformer、state‑space、RNN）和多语言数据的统一评估。

**📊 数据集**

使用了多家公开LLM（Pythia、GPT‑2、GPT‑Neo、Qwen2、Mistral、Mamba、RWKV、StarCoder2、BLOOM、OLMo等）以及多种文本语料（WikiText、LAMBADA、SST‑2、TruthfulQA、CounterFact、CIFAR‑100、Peelle等人类填空数据）进行评估。

**📈 对比分析**

与传统欧氏距离、激活空间相似度、基于logit的Kullback‑Leibler等方法对比，输出几何在模型间关系一致性、与人类预测的对齐度、干预的最低扰动成本、可复用控制与编辑效果等方面均显著优于现有方法（平均对齐提升0.3–0.5，干预成本降低1–2个数量级，编辑误差下降≈10×）。

**⚠️ 局限性**

局限性包括：对极大规模模型（>6B参数）在高维空间下的数值稳定性与计算成本、对非文本任务或复杂结构化输出的适用性有限、干预效果仍受激活维度和模型内部非线性耦合影响、以及在真实对齐场景中对多模态或动态对话的推广需要进一步研究。

---

## 227. How AI Coders Discuss, Disagree, and Reach Consensus: Challenges and Opportunities for LLM-Based Qualitative Coding

**arXiv ID:** 2609.11109 | [PDF](https://arxiv.org/pdf/2609.11109v1)

**作者:** Jeongyeon Kim `[一作]` (Stanford University), John Mitchell `[通讯]` (Stanford University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用多代理LLM对定性数据进行编码，模拟人类多码器讨论并通过协商达成共识。

**💡 创新点**

量化了编码准确性与代码书长度、相似度、专业度差距、讨论激烈程度等因素的关系，并提出基于这些因素的设计建议。

**🔧 技术方法**

构建了基于OpenAI ChatGPT‑4o‑mini的多代理编码管线，采用Prompt Engineering、讨论轮次、共识归纳，并用混合效应模型进行统计分析。

**📊 数据集**

在四个不同领域（教育、法律、社会学、医学）的公开数据集上进行实验，每个数据集随机抽取5个标签、每标签500个实例。

**📈 对比分析**

与人工标注基准对比，双代理间的IRR均超过0.85，平均F1约0.68；混合效应模型揭示多项显著影响因素；讨论越激烈准确性越高。

**⚠️ 局限性**

未对LLM进行微调、仅使用单一模型版本、以人工标注为真值可能不完全准确、缺乏真实人机交互验证、LLM技术快速演进导致结果可复现性受限。

---

## 228. SaltBench: A Referee-Gated Protocol for Measuring Method Effects in Machine-Checked Software Work

**arXiv ID:** 2609.11076 | [PDF](https://arxiv.org/pdf/2609.11076v1)

**作者:** Jason Hickey `[一作]` `[通讯]`, Jason Hickey

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了 SaltBench 基准协议，测定机器裁判（proof kernel、程序验证器或隐藏测试套件）对编码代理工作成本的影响。

**💡 创新点**

创新点在于：①将裁判决策放在代理之外并预先注册，消除代理对结果的叙述能力；②采用“seat‑as‑subject”设计，将代理会话而非模型本身作为实验对象；③通过预算停止和多层预注册实现可复现且可度量的成本评估。

**🔧 技术方法**

使用的技术包括 Rust 与 Verus 语言实现、Lean 证明器、隐藏测试套件与变异测试、成本计费系统、沙箱与文件权限审计、以及事前预注册与增量修正机制。

**📊 数据集**

数据集为作者手工编写的五个系统组件（包括 Crc32、FreeList、LZW、Paxos、LRU），每个组件配备了隐藏测试套件和变异测试集合，所有材料均在实验前冻结并提交至版本化仓库。

**📈 对比分析**

比较方法是对比“plain”与“treatment”两种代理配置下每个问题的成本中位数比值（premium），采用单边符号检验检验 premium 是否显著大于 1。结果显示 treatment 在所有五个问题上成本均更高，median premium 介于 1.16× 至 2.89×，但三项在可解析阈值以下，说明增量并不显著；post‑hoc 正确性检查未能分离两组。

**⚠️ 局限性**

局限性包括：①只评估了节制化的“dieted”方法，未能推断方法对裁判接受度的影响；②成本实验未同时记录正确性，导致无法关联成本与质量；③作者提供的样本量有限且可能存在作者偏倚；④并发执行与沙箱测量缺陷可能引入额外误差；⑤预算停止与裁判判定的耦合可能导致对 treatment 的样本偏差。

---

## 229. You've Got a BUD in Me: Authenticated Reads from Per-Block Write Logs

**arXiv ID:** 2609.11251 | [PDF](https://arxiv.org/pdf/2609.11251v1)

**作者:** Alejandro Ranchal-Pedrosa `[一作]` (Sei Labs), Ben Marsh `[通讯]` (Sei Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建基于区块写日志的Block Update Digest (BUD) 和 SuperBUD 体系，支持无需全局状态根的历史读证明。

**💡 创新点**

创新点在于通过每块写入的前驱指针与窗口汇总相结合，提供可证明历史成员资格与排除的轻量级读协议，并将认证工作与写量关联。

**🔧 技术方法**

采用Merkle树、前驱链、指数层级窗口、委员会签名聚合与离线归档相结合的技术。

**📊 数据集**

使用合成写入量可调的模拟工作负载以及以太坊24,576区块、约445个账户的账户访问追踪。

**📈 对比分析**

与内存/磁盘Merkle Patricia Trie、NOMT、QMDB比较，BUD的基准路径增长仅1.24×（vs. 3.1×和69.5×），证书大小在几百字节到数千字节，验证耗时≤146µs，签名聚合约0.65ms。

**⚠️ 局限性**

局限性包括对归档与签名可用性的假设、需要扫描激活旧键、缺乏对大规模存活条目和更高保留窗口的实验验证，以及未实现完整的活性回收与跨阶段的安全证明。

---

## 230. OmniHallu: Unified Hallucination Detection for Cross-Modal Comprehension and Generation in Multimodal Large Language Models

**arXiv ID:** 2609.11244 | [PDF](https://arxiv.org/pdf/2609.11244v1)

**作者:** Jianjiang Yang `[一作]` (University of Manchester), Meng Luo `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了统一的多模态幻觉检测框架 OmniHallu，并构建了覆盖六种跨模态任务的 10,000 条样本的 OmniHallu‑Bench 基准。

**💡 创新点**

创新点包括：①在跨图像、视频、音频的两向任务中实现 claim‑level 检测；②利用多智能体的模态专属专家进行验证并通过结构化推理聚合；③设计了基于 GRPO 的可训练 verifier，显著减少专家调用。

**🔧 技术方法**

技术实现主要包括：原子 claim 的拆解（ACD），模态感知专家验证（Grounding DINO、VideoLLaMA 等），GPT‑5.2 推理聚合，以及 GRPO 训练的精简 verifier。

**📊 数据集**

使用的数据集包括 OmniHallu‑Bench（从 COCO、MSVD、AudioCaps 等公开数据衍生的 10k 样本）以及多种 MLLM 的生成结果。

**📈 对比分析**

与 Self‑Check、UNIHD 等基线在 claim‑level 上对比，采用 P/R/F1/Acc/Mac.F1 评估；在六个任务中，OmniHallu 以 3.4–8.1 分的 Mac.F1 提升，ablation 证实 ACD 和多专家投票是关键；利用 verifier 可将专家调用降低 66%，成本约 0.05 美元/样本。

**⚠️ 局限性**

局限性：①幻觉分类细粒度不足，未覆盖视频事件细分或音频源分离；②仅针对文本中心的拆解，未考虑像图像‑音频等其他模态对；③性能受推理模型与专家工具成熟度影响，且多数投票无法纠正所有专家共犯的错误。

---

## 231. SCINTILLA-SNN: A Spiking Multi-Scale Selective Aggregation Network for Perineural Invasion Prediction

**arXiv ID:** 2609.11237 | [PDF](https://arxiv.org/pdf/2609.11237v1)

**作者:** Youngung Han `[一作]` (Seoul National University), Nam-Joon Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种基于3D脉冲神经网络SCINTILLA‑SNN，用于术前胆道癌的周神经侵袭预测。

**💡 创新点**

创新点在于引入多尺度脉冲聚合模块（MSSA），并通过发射率与膜电位波动的脉冲动力学门控实现对稀疏诊断信号的选择性聚合。

**🔧 技术方法**

采用了3D脉冲卷积、LIF神经元、窗口调制、门控多尺度聚合、Top‑k 选取、Softmax 软聚合、焦点损失与稀疏正则等技术。

**📊 数据集**

使用了10年回顾性单中心胆道癌182例的T2加权MRI数据，配备手工标注的肿瘤与肝组织掩膜。

**📈 对比分析**

在5折交叉验证中与CNN、Transformer及纯脉冲基线进行比较，SCINTILLA‑SNN取得最高AUROC 0.748，且能量消耗比密集计算低23.18×，展现出最佳的准确率‑能耗折中。

**⚠️ 局限性**

局限性在于仅使用单中心数据，需在多机构外部数据上验证其泛化能力，同时对不同影像协议的适应性尚未充分评估。

---

## 232. A Voice-Interactive Multi-Agent System for Smart Operating Rooms: Architecture Design and Key Technologies

**arXiv ID:** 2609.11231 | [PDF](https://arxiv.org/pdf/2609.11231v1)

**作者:** Tianxiang Zhou `[一作]` `[通讯]` (Wuhan United Imaging Surgical Co., Ltd.), Tianxiang Zhou (Wuhan United Imaging Surgical Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 SurgicalRoomAgent，一套从唤醒到 TTS 的完整语音交互多智能体系统，实现了在智能手术室中多设备的实时语音控制、手术记录和报告生成。

**💡 创新点**

核心创新包括基于字节级最长公共前缀的 KV Cache 热预热、流式部分 JSON 解析与早期并行任务执行、按角色/设备/手术阶段的渐进式技能提示过滤，以及基于 Kahn 算法的 DAG 任务规划。

**🔧 技术方法**

采用 Qwen3‑27B‑FP8 LLM 与 llama.cpp/sglang 推理引擎、FunASR / Qwen3‑ASR、CosyVoice / Qwen3‑TTS、Sherpa‑ONNX 唤醒、FastAPI + WebSocket 前后端，以及 JWT 认证、字段级加密和审计日志等安全措施。

**📊 数据集**

主要使用的是手术室内部真实设备状态与对话历史（无公开公开数据集），系统在实验室环境下通过模拟手术场景进行验证。

**📈 对比分析**

与现有系统（GePpeTto、VISA 等）对比，KV Cache 热预热使预填充时间从约500 ms降低到 50–80 ms，首 token 时间缩短至 150–200 ms，整体推理延迟平均下降 70–80%，实现了低延迟、并行执行的“generate‑while‑execute”流水线。

**⚠️ 局限性**

局限包括缺乏大规模临床验证、单一 LLM 实例可能成为瓶颈、ASR 在高噪声环境下的鲁棒性不足、设备协议多样化导致集成成本高、以及 16 384 令牌窗口对长手术历史的约束。

---

## 233. Harness Robotic OS: A Unified Embodied-Agent Runtime for Closed-Loop Quadruped Inspection

**arXiv ID:** 2609.11225 | [PDF](https://arxiv.org/pdf/2609.11225v1)

**作者:** Yaoyuan Yan `[一作]` (Country Garden Services), Wei Zhou `[通讯]` (Country Garden Services)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个统一的嵌入式机器人操作系统HROS，并在住宅巡检中实现了Argos完整闭环；

**💡 创新点**

创新点在于将实时导航、感知、规划与认知推理、记忆与安全自演化分离，形成可追溯的多模态嵌入式代理运行时，并引入基于经验的安全门控自演化机制；

**🔧 技术方法**

使用Fast‑LIO2 LiDAR‑IMU SLAM、PCT‑Planner全局规划、EGO‑Planner局部规划、Hobot‑Stereo深度、Qwen3‑VL视觉语言、OpenClaw、ASR/TTS以及RDK边缘计算等技术；

**📊 数据集**

使用住宅物业巡检环境收集的LiDAR、IMU、相机数据以及由Qwen3‑VL产生的标注事件；

**📈 对比分析**

通过在真实住宅场景中对比多项指标，展示了100%航点到达、<10cm定位误差、<200ms障碍反应、95–95%多类危险检测率、99%报警及报告生成成功率；

**⚠️ 局限性**

局限性包括地图长期维护、开世界危险识别、记忆与自演化评估缺乏基准、人机协作与噪声鲁棒性待验证。

---

## 234. REVA: Reusable Evidence View Aggregation for Context-Efficient RAG Serving

**arXiv ID:** 2609.11209 | [PDF](https://arxiv.org/pdf/2609.11209v1)

**作者:** Tuan Nguyen `[一作]` (VinUniversity), Fan Lai `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Reusable Evidence View Aggregation（REVIEW）框架，通过挖掘历史查询-文档-生成器交互中的注意力信息，构建可重用的文档级压缩视图，降低RAG推理成本。

**💡 创新点**

将RAG压缩视为数据挖掘任务，将注意力追踪离线聚合为文档键值的可重用得分库，支持预算无关的文本视图；避免在线压缩，保持模型接口不变，显著降低压缩开销。

**🔧 技术方法**

利用LLM注意力权重作为重要性指标，将token映射到词级单元；聚合到文档键值分数存储；在线时按预算取最高分词单元并按原文顺序渲染；支持本地与全局预算分配策略。

**📊 数据集**

在四个公开问答基准上评测：Natural Questions、TriviaQA、HotpotQA、2WikiMulti-hop QA；使用三大LLM模型：Llama-3.1-8B、Qwen3.5-9B、Gemma-4-E4B。

**📈 对比分析**

与前缀截断、SelCtx、LLM-L2、RECOMP-e、LongLLM、EXIT、FaviComp等多种压缩方法对比，平均在B=512时提升F1 1.0–5.8点，同时在线压缩开销下降5.3–15.6倍（仅26–40ms），达到近前沿的质量‑延迟平衡。

**⚠️ 局限性**

需要历史查询数据构建分数库，覆盖率有限；在文档内容或检索模型变化时需刷新分数；对极短或极长文档的单元映射可能失真；仅在RAG场景适用，对单纯文本生成不直接适用。

---

## 235. Convex Optimization with Nested Evolving Feasible Sets (CONES) under Time-Varying Loss Functions

**arXiv ID:** 2609.11207 | [PDF](https://arxiv.org/pdf/2609.11207v1)

**作者:** Rahul Vaze `[一作]` `[通讯]`, Rahul Vaze

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文将CONES框架推广到可变损失函数场景，并在强凸与普通凸两类下分别提出投影近端(Prox)算法，给出其损失与位移成本的上界与下界。

**💡 创新点**

创新点在于同时考虑可变约束和可变目标函数，证明投影近端算法在强凸情况下实现O(1)调度误差与O(log T)位移成本的最优组合，并在普通凸情况下给出可调β的误差-位移权衡与对应的下界。

**🔧 技术方法**

采用投影近端优化、凸分析与在线学习的理论工具，结合可变学习率和近端步长，得到误差与位移成本的闭式上界，并利用构造性对手和信息论方法推导下界。

**📊 数据集**

未使用实验数据集，研究完全基于理论分析与证明。

**📈 对比分析**

与已有的Frugal、LSP等算法相比，投影近端在强凸场景下实现与下界一致的O(log T)位移成本，在普通凸场景下提供了β可调的误差/位移权衡，优于或等价于以往结果。

**⚠️ 局限性**

局限性在于仅给出理论性能，缺乏实验验证；算法假设约束集合可嵌套且有静态最优基准，未讨论在线自适应基准或非静态最优情形。

---

## 236. Automated Identification of Competing Narratives in Political Discourse on Social Media

**arXiv ID:** 2609.11202 | [PDF](https://arxiv.org/pdf/2609.11202v1)

**作者:** Sergej Wildemann `[一作]` (L3S Research Center, Leibniz Universität Hannover), Erick Elejalde `[通讯]` (L3S Research Center, Leibniz Universität Hannover)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个无监督框架，用于在德国政客推文中识别、构造并分析竞争性政治叙事；

**💡 创新点**

创新点在于将主题建模、事件检测、事件链接与基于全局用户嵌入的社区划分相结合，形成可区分的叙事线索；

**🔧 技术方法**

采用了BERTopic（Sentence‑BERT嵌入）、Affinity Propagation、Sliced Wasserstein Distance、Leiden社区检测及HDBSCAN聚类等NLP与图分析技术；

**📊 数据集**

使用了从Wikidata抽取的1,324个德国政客账号在2022年1月1日至2023年6月24日期间收集的189,850条推文（去除转发、回复、引用）；

**📈 对比分析**

通过与政党归属标签的对比验证用户社区划分的有效性，展示在能源危机和移民政策两大案例中不同党派在同一叙事中的分化与情感差异，表明模型能揭示实际的政治立场对话；

**⚠️ 局限性**

局限性包括缺乏客观的叙事质量评估指标、仅在德国政客推文上验证、对短文本的语义捕捉仍有限，以及对多语言推文翻译依赖外部模型。

---

## 237. CEM-TUDASR: Computationally efficient multi-modality transformer based unsupervised domain adaptive super-resolution approach

**arXiv ID:** 2609.11201 | [PDF](https://arxiv.org/pdf/2609.11201v1)

**作者:** Anjali Sarvaiya `[一作]`, Kiran Raja `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了无监督的 Transformer 基础超分辨率框架 CEM‑TUDASR，用于提升无线胶囊内镜图像的空间分辨率。

**💡 创新点**

创新点包括：基于域适应的降解建模网络实现真实 WCE 低分辨率合成；引入 Deep Attention Block（DAB）与 Fusion Attention Block（FAB）的 Transformer 生成器，联合捕获长程上下文与局部细节；以及在 GAN 训练中使用无对齐 LR‑HR 图像的无监督策略。

**🔧 技术方法**

使用的技术包括：GAN 对抗学习、Transformer 与自注意力（Efficient Attention 与 Efficient Spatial Attention）、深度注意力块、融合注意力块、无监督降解与超分训练，以及无参考图像质量评估指标。

**📊 数据集**

使用数据集：新构建的 Kvasir Capsule 训练/验证/测试集；外部验证集 KID、GIANA 以及十字领域的 retinal 数据集。

**📈 对比分析**

与 ZSSR、DASR、dSRVAE、DUSGAN、BSRGAN、MDASR、TUDASR 等无监督 SR 方法对比，CEM‑TUDASR 在 PIQE、NIQE、EndoQM 指标上均取得最优或接近最优成绩，在 BRISQUE 上仅次于 TUDASR；同时保持 2.67 M 参数、169.94 GFLOPs 的轻量级实现。

**⚠️ 局限性**

局限性：在极端降解条件下仍难以恢复细小血管或纹理；GAN 生成可能出现伪影或过度增强；模型对降解网络的依赖导致对不同设备/协议的泛化受限；未考虑视频时序信息，且仅处理 2D 图像。

---

## 238. FlexComp: One Model for Every Ratio in Context Compression

**arXiv ID:** 2609.11192 | [PDF](https://arxiv.org/pdf/2609.11192v1)

**作者:** Kaiyan Zhao `[一作]` (University of Tokyo), Yoshimasa Tsuruoka `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FlexComp 框架，将上下文压缩比例从固定值解耦到训练和推理时可动态选择。

**💡 创新点**

创新点在于：① Matryoshka 训练让单一模型支持任意压缩比例；② 两种推理时预算选择策略（置信度级联和轻量级 K 预测器），实现动态压缩与精度权衡；③ 通过动态压缩显著降低 KV 缓存占用并提升解码吞吐量。

**🔧 技术方法**

技术包括：软上下文压缩（ICAE、500xCompressor、SAC）、Matryoshka 预算采样训练、置信度级联路由、K 预测器（两层 MLP）、LoRA 微调、Llama-3.2-1B 及 3.1-8B 模型。

**📊 数据集**

使用 SlimPajama-6B 进行预训练，MRQA 任务集（6 个 ID 任务与 6 个 OOD 任务）进行微调与评估。

**📈 对比分析**

与单独训练的固定比例专家比较，单一 Matryoshka 模型在三种压缩器上几乎无精度损失；级联路由在保持 98% 低压缩精度的同时可达 266× 压缩比；K 预测器在单通路下实现 158–236× 压缩比，精度仅下降 ≤0.7 F1，显著节省 KV 内存（50%）并提升吞吐（47%）。

**⚠️ 局限性**

局限性包括：预训练阶段采用固定预算仍更优；K 预测器需平衡标签分布，过度偏斜会导致预测器崩溃；在极端压缩下，较大预算有时会降低精度；以及对更大模型和不同压缩方法的泛化需进一步验证。

---

## 239. Debate-to-Skill: Capability-Bound Process Supervision for Industrial Query-to-Agent Annotation

**arXiv ID:** 2609.11176 | [PDF](https://arxiv.org/pdf/2609.11176v1)

**作者:** Shiyu Zhang `[一作]` (Baidu, Inc.), Huifu Li `[通讯]` (Baidu, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于过程监督的工业查询‑代理匹配框架 Debateto‑Skill，旨在把监督目标从单纯的语义相关性转向能力验证。

**💡 创新点**

创新点：① 将监督对象定义为能力约束的决策过程；② 设计可复用的决策原则、结构化辩论轨迹、判定器抽取与争议驱动更新机制；③ 通过灰区（raw‑label‑1）分析验证该过程监督在实际业务中的显著收益。

**🔧 技术方法**

主要技术：过程监督（process‑supervision），结构化辩论（debate）框架，判定器（verifier）与 GRPO 强化学习，理由化 SFT，词袋检索、动态技能更新，风险敏感裁决。

**📊 数据集**

数据集：工业 Query2Agent 流量数据，170k 训练样本、3k 验证样本、3k 公开测试样本（Main、Domain、Longtail），包含三元标签 {2,1,0} 比例 5:2:3，并对 312 个 raw‑label‑1 进行灰区切片。

**📈 对比分析**

对比方法：与直接标签监督（SFT Label）、SFT Reasoning、w/o Debate、w/o Dynamic Skills、w/o Judge 进行离线评估；Debate‑to‑Skill 在 Main Test 上 ACC 91.6%、Binary F1 89.7%、Macro F1 87.4%、灰区 F1 64.9，显著优于基线；在线 A/B 测试显示卡片印象率 +15.8%、CTR +7.6%、转发率 +6.1%。

**⚠️ 局限性**

局限性：① 采用词袋检索与规则式争议更新，缺乏语义检索与自学习更新；② 判定器仅验证结论与结构，未完全保证自然语言证据的可信度；③ 评估仅在工业 Query2Agent 场景，未构建公开灰区基准。

---

## 240. A Four-Valued Graph Model for Conflict Resolution: Core Framework and a Machine-Checked Formalization in Lean 4

**arXiv ID:** 2609.11174 | [PDF](https://arxiv.org/pdf/2609.11174v1)

**作者:** Yukiko Kato `[一作]` `[通讯]` (Institute of Science Tokyo), Yukiko Kato (Institute of Science Tokyo)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了Quasi-Closed World Graph Model for Conflict Resolution（QCW‑GMCR），将Belnap四值逻辑引入选项层次的知识不确定性，并在Lean 4中机理化证明了核心定理；

**💡 创新点**

创新点在于：①在选项层面使用四值逻辑刻画知识模糊；②构建了基于FDE语义的加权可达性与分级稳定性；③提出受限的四值归约算子并证明其安全性与可比性；④通过结构安全不变式实现灾难避免；⑤在hypergame框架中严格扩展了四值视角；

**🔧 技术方法**

使用了Belnap–Dunn四值逻辑、第一度蕴涵（FDE）语义、图模型（GMCR）、Lean 4证明助手和mathlib库；

**📊 数据集**

无外部数据集，研究以理论构造和符号推演为主；

**📈 对比分析**

通过Lean 4进行形式化验证，已证明GMCR稳定性层级、四值逻辑运算性质、归约算子可接受性、可达性层级、以及安全不变式等核心定理；性能表现为严格的可执行证明和可验证实现；

**⚠️ 局限性**

局限在于：①尚未对FDE修正公式、hypergame存在性反例和完整的quasi‑closed invariant等关键定理进行机理化；②未实现动态信念更新和大规模选项空间的可扩展方法；③对归约算子在实际决策情境中的经验验证有限。

---

## 241. Semi-Tensor Product-Based Multi-Term Randomized T-SVD and Its Visual Applications

**arXiv ID:** 2609.11168 | [PDF](https://arxiv.org/pdf/2609.11168v1)

**作者:** Xingchen Xiao `[一作]` (Southwest University), Jianjun Wang `[通讯]` (Southwest University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于任意可逆线性变换的第三阶张量半张量乘积，并在此框架下构造了多项式半张量奇异值分解（MSTP‑SVD）以及其随机加速版本（MRSTP‑SVD），实现了高效且更准确的低秩张量逼近。

**💡 创新点**

创新点在于：①打破传统t‑product 的维度匹配限制，支持任意可逆线性变换；②引入多项式分解结构，显著提升低秩逼近精度；③结合随机投影与幂迭代实现计算加速，提供理论误差上界；④首次将该方法应用于大规模图像/视频压缩与补全。

**🔧 技术方法**

核心技术包括：半张量乘积定义（以块 Kronecker 形式实现）、多项式半张量奇异值分解、随机子空间投影与幂迭代、张量重组与逆变换等；实验中使用 DFT/DCT/ROT 作为可逆线性变换。

**📊 数据集**

主要数据集包括四幅高分辨率 RGB 图像（Lake、Night、Road、Fruit）以及四段高清视频（Crosswalk、Market、Narrator、Aerial），此外在图像/视频补全实验中随机遮挡 70% 的像素。

**📈 对比分析**

与传统 TT‑SVD、STP‑SVD、TSTP‑SVD 等基线相比，MSTP‑SVD 在 PSNR/SSIM 上提升约 3–6 dB，MRSTP‑SVD 在保持相近质量的同时将运行时间缩短 30–50%（图像约 1–2 s，视频约 10–20 s）。

**⚠️ 局限性**

局限性包括：①仅针对第三阶张量，扩展到更高阶仍需研究；②需要预先选择合适的可逆变换和超参数（k、s、q、截断矩阵 R）；③在极大规模数据（数十亿元素）下仍可能面临存储与并行化挑战。

---

## 242. When does a spectral prior help graph learning? Connectivity-loss estimation under road-network disruptions

**arXiv ID:** 2609.11166 | [PDF](https://arxiv.org/pdf/2609.11166v1)

**作者:** Van-Truong Le `[一作]` `[通讯]` (Viet Nam National University), Van-Truong Le (Viet Nam National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种结合Fiedler向量一阶敏感度与图神经网络残差的残差谱学习模型，用于快速估计道路网络在多条连通链路失效后代数连通性的相对损失。

**💡 创新点**

创新点在于把解析的谱先验（Fiedler敏感度）作为可学习的边界残差输入，既保留了理论可解释性，又通过GNN学习有限删链对的非线性校正，且在跨域、跨失败模式下探讨其优势与局限。

**🔧 技术方法**

使用的技术包括：GCN、GraphSAGE、边感知MPNN等GNN骨干；一阶与二阶谱扰动近似；基于Fiedler向量的节点特征；层级Bootstrap不确定性评估；以及稀疏谱求解与增量计算的运行时分析。

**📊 数据集**

数据集包括：多种规模的随机几何图（35–65节点）作为合成基准；以及从OpenStreetMap抓取的13个小型道路网络（48–1,259节点）覆盖越南、新加坡、马来西亚、泰国、台湾和日本，进行零样本与留一地区/国家迁移实验。

**📈 对比分析**

对比方法有：直接GNN回归、残差GNN、解析一阶与二阶谱近似；在不同失败模式（独立、空间聚类、边介数攻击）下计算MAE。结果显示：在零样本OSM上残差模型对GCN和GraphSAGE平均提升MAE约0.02–0.04，二阶谱仅提升0.003左右；但在跨地区迁移时残差效果波动，部分情况下反而不如直接模型，说明残差先验在域移位时可能变成偏置。

**⚠️ 局限性**

局限性包括：仅评估48–1,259节点的小型道路网络，无法证明在更大规模或不同地理区域的泛化；只关注结构连通性，忽略交通需求、容量与时效；样本地区与失败模式受限，导致统计不稳；残差先验在域移位时可能产生偏差，需进一步校准或混合专家模型。

---

## 243. Assessing the Reusability of Public Speech Resources for Low-Resource Languages: A Central Kurdish Case Study

**arXiv ID:** 2609.11246 | [PDF](https://arxiv.org/pdf/2609.11246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 244. Multimodal Temporal Modeling for Continuous Group Emotion Recognition in Multi-party Dialogues

**arXiv ID:** 2609.11164 | [PDF](https://arxiv.org/pdf/2609.11164v1)

**作者:** Soma Iwata `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在多方对话中实现一秒钟连续的群体情绪（激活度和正向情绪）识别，并引入“混合”标签来捕捉参与者情绪分歧。

**💡 创新点**

①在对话中引入连续、秒级情绪标注；②提出混合标签表征情绪分歧；③用多模态时序Transformer对音频和视频进行融合，展示比LLM更优的连续情绪估计；④对混合区间误差进行定量分析。

**🔧 技术方法**

多模态特征提取（SigLIP 2视频、Whisper音频）+1D卷积+Transformer编码器+滑动窗口时序推理；对比LLM+Q-Former。

**📊 数据集**

TEIDAN多方对话语料（36段，三人组），在其基础上进行1秒软标签连续标注并构建混合标注。

**📈 对比分析**

对Transformer与LLM、不同上下文长度（5s/20s）以及音频/视频单模/双模进行评估，使用CCC、MAE、Pos F1指标；结果显示AV(20s) Transformer在CCC最高、MAE最低，音频优于视频，20s上下文仅提升Arousal MAE；混合区间误差显著增大。

**⚠️ 局限性**

仅对两组对话做测试，样本量小；缺少参与者级情绪标签；模型未充分利用LLM潜能；多模态接口与训练数据量有限；未验证跨文化或实时推理的鲁棒性。

---

## 245. LILA: Calibration-Free Structured Pruning of Large Language Models via Latent Spectral Geometry

**arXiv ID:** 2609.11163 | [PDF](https://arxiv.org/pdf/2609.11163v1)

**作者:** Sankar Behera `[一作]` (IIT Jammu), Yamuna Prasad `[通讯]` (IIT Jammu)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 LILA，一种无校准数据、无策略训练、保持原架构的结构化剪枝框架，利用权重矩阵奇异值分布的 KS 距离来评估神经元重要性并生成剪枝掩码。

**💡 创新点**

创新点在于：①基于 Kolmogorov–Smirnov 距离的闭式谱重要性评分，无需前向传播或校准数据；②采用 NMF 低秩分解提取权重谱信息；③通过 NTK 理论证明该谱规则能显著降低功能失真；④提供自适应稀疏度分配方案并保持原有计算图。

**🔧 技术方法**

使用技术包括：非负矩阵分解（NMF）、KS 距离谱评分、随机化 SVD、低秩 NMF 的降阶更新、LoRA 细化恢复以及 NTK 轨迹比分析。

**📊 数据集**

实验数据集：主要在 LLaMA‑2‑7B、Phi‑2、OPT‑1.3B 上进行零样本基准评估；无校准数据实验使用公开语料；微调恢复实验使用 WikiText‑2、Alpaca、Instruction‑tuning 等文本集合。

**📈 对比分析**

与 Wanda、PruneNet、SliceGPT 等基线对比，零微调下 LILA‑Spectrum 在 20–40% 稀疏度下提升 4–6pp，优于 SliceGPT；与 PruneNet 的 RL 策略相比提升 1.6pp；微调后与 SliceGPT 差距 ≤0.48pp，且无需改造架构；剪枝时间仅 2.4–44 分钟，显著低于基线的训练/推理成本。

**⚠️ 局限性**

局限性：Spectrum 方案计算量相对较高（O(d_ff r²)），对高稀疏度下单层瓶颈更敏感；目前仅针对 FFN 层，未结合量化或更大规模模型的实验；对数据分布漂移的鲁棒性仍待进一步验证。

---

## 246. The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the Limits of Optimization-Based Methods

**arXiv ID:** 2609.11247 | [PDF](https://arxiv.org/pdf/2609.11247v1)

**作者:** Ioanna Kaffeza `[一作]` (Mines Paris-PSL University), Alexandros Potamianos `[通讯]` (National Technical University of Athens)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过统一评估框架，对多模态情感分析中常用的梯度调制与损失重加方法（OGM、OGM-GE、AGM、PMR、ReconBoost）进行实验和理论诊断，指出这些方法本质上利用训练时的拟合信号（loss、梯度、似然）来估计模态价值，导致与真实判别贡献相悖，并提出基于持出验证判别性能估计模态价值的研究方向。

**💡 创新点**

创新点在于①构建统一评估框架，系统对比多种优化平衡策略；②提出理论诊断，阐明“拟合速度≠判别贡献”的根本错误；③展示通过持出开发集校准可提升稳定性但仍无显著收益，进而引出持出判别估计的研究议程。

**🔧 技术方法**

技术包括：基于 LSTM 的 late‑concatenation 架构；梯度调制方法（OGM、OGM‑GE、AGM）；损失重加方法（PMR、ReconBoost）；Adam/SGD 训练；开发集校准；控制失衡实验（A‑V、T‑V、A‑T‑V）以及 XOR‑门互补性验证实验。

**📊 数据集**

使用的公开数据集为 CMU‑MOSI 与 CMU‑MOSEI；此外构造了用于验证互补性的 XOR‑门合成任务。

**📈 对比分析**

与基准（Late Concatenation、Uni‑Pre‑Finetuned、Soft‑Voting）以及不同优化器比较，结果显示大多数平衡方法无法显著超越 Late Concatenation；性能高度依赖优化器与超参数；在控制失衡设置下甚至出现性能下降。

**⚠️ 局限性**

局限性在于：①方法仅使用训练时的拟合信号，无法准确估计模态判别贡献；②对样本级互补性缺乏估计，导致全局权重无法捕获样本差异；③依赖优化器和超参数，缺乏鲁棒性；④需要从持出验证判别性能中估计模态价值，才能真正解决模态失衡问题。

---

## 247. Polyhedral Geometry of Time-to-First-Spike Neural Networks

**arXiv ID:** 2609.11227 | [PDF](https://arxiv.org/pdf/2609.11227v1)

**作者:** Manjot Singh `[一作]` (Ludwig-Maximilians-Universität München), Gitta Kutyniok `[通讯]` (Ludwig-Maximilians-Universität München)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

本文研究了脉冲神经网络（SNN）的发射时序映射，利用其凸多面体结构对神经元的因果分区进行几何描述；

**💡 创新点**

创新点在于将SNN的发射时序映射建模为一个受限的凸多面体的上凸面，并将因果区域与正则细分、热带几何等高级几何结构建立联系，进而给出浅层和深层网络的因果区域计数上界与下界；

**🔧 技术方法**

主要技术包括凸几何、正则细分、热带几何、超平面排列理论以及递归构造法；

**📊 数据集**

本文未使用任何具体数据集，全部以理论证明为主；

**📈 对比分析**

与传统 ReLU 网络的区域计数做了对比，指出在固定输入维度时 SNN 的最大区域数随隐藏层宽度呈多项式（Θ(m^{d-1})）增长，而 ReLU 网络为 Θ(m^d)；在固定宽度时 SNN 随输入维度呈指数增长，ReLU 网络则在 d≥m 时趋于 2^m；

**⚠️ 局限性**

局限在于对深层网络的上界相对宽松，且仅给出渐进量级的下界，实际网络可能存在更高或更低的因果区域数，且未涉及实验验证。

---

## 248. Legible Failures: Detecting and Repairing In-Context Binding Errors

**arXiv ID:** 2609.11216 | [PDF](https://arxiv.org/pdf/2609.11216v1)

**作者:** Manas Venkata Sai Ravulapalli `[一作]` (Efficient Computation Inc.), Abhinav M. Hari `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过设计一个在上下文中绑定实体与义务的合成任务，利用线性探测器评估并修复模型在推理时未正确使用已知绑定的错误。

**💡 创新点**

创新点在于发现并量化了“可读性失败”(legible failure)，证明在模型错误时隐藏状态中仍保留可线性提取的正确绑定，并提出通过签名不一致分数实现错误检测以及通过残差流方向调整恢复错误。

**🔧 技术方法**

主要技术包括多层Transformer的隐藏状态线性探测器、查询实体对抗实验、AUROC评估、预测熵与自一致性对比、以及基于探测解码的自门控残差调节。

**📊 数据集**

实验使用了自定义的合成数据集：每个试验从独立的实体池和义务池随机抽取K对并插入干扰块，随后查询其中一实体；数据涵盖了 16 个公开检查点（从410M到14B参数）以及 8 个额外模型。

**📈 对比分析**

与基线1/K、模型自身置信度、自一致性以及预测熵比较，签名不一致分数在错误检测上平均提升约0.15 AUROC；自门控残差干预在所有8个模型中平均提升准确率约0.03，且随机方向控制实验验证了该提升为因果效应。

**⚠️ 局限性**

局限性包括：任务仅为单词级合成实验，缺乏自然语言真实情境；需要对模型内部状态的白盒访问；对可变/作用域绑定的处理效果不佳；未在多词或自由形式的推理任务上验证方法的可迁移性。

---

## 249. CryptoL: Towards Scale Dominance and Physics Constraints Mitigation in Financial Multivariate Time Series Forecasting

**arXiv ID:** 2609.11206 | [PDF](https://arxiv.org/pdf/2609.11206v1)

**作者:** Yalda Taheri `[一作]` (Azad University), Hossein Karshenas `[通讯]` (University of Isfahan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 CryptoL 框架，用于在多资产、跨尺度的加密货币 OHLC 时间序列上进行联合预测。

**💡 创新点**

创新点在于：① 两相位 RevIN (TP‑RevIN) 通过在归一化空间训练消除尺度依赖的梯度加权；② 动态 epsilon 归一化以解决极低价格资产的数值不稳定；③ 在归一化空间引入物理约束损失，保持蜡烛图合法性；④ 对通道独立和通道共享 RevIN 的理论与实验对比。

**🔧 技术方法**

技术手段包括：RevIN / TP‑RevIN、动态 epsilon、归一化空间 OHLC 约束损失；使用三种解码器式时序模型（Timer、Timer‑XL、Time‑MoE）；JAX + Flax 在 8 台 TPU v5e 上并行训练。

**📊 数据集**

实验数据来自 Binance API，约 15.5 M 条 OHLC 记录，涵盖 16 只加密资产，价格跨度从 10⁻⁷（PEPE）到 10⁵（BTC）。

**📈 对比分析**

通过与无 RevIN、标准 RevIN、FAN、SAN 等方法对比，评估 MSE、MAE、MAPE、PHY 等指标。结果表明：TP‑RevIN 在所有模型和时间窗下显著降低 MSE/MAE；动态 epsilon 在低价资产上将 MAPE 从数万降至 1–3；加入物理约束后 PHY 几乎为 0，且 MAE 仍保持竞争力。

**⚠️ 局限性**

局限性包括：① 对极端波动资产仍存在一定误差；② 仅在单个 epoch 训练，未探索更深层次的长期依赖；③ 主要验证于 Binance 资产，跨市场推广需进一步验证。

---

## 250. A Multi-View and Confusion-Guided Ensemble Framework for Robust Synthetic Image Attribution

**arXiv ID:** 2609.11188 | [PDF](https://arxiv.org/pdf/2609.11188v1)

**作者:** Zuomin Qu `[一作]` `[通讯]` (China Southern Power Grid Electric Power Research Institute), Zuomin Qu (China Southern Power Grid Electric Power Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了多视角与困惑引导的集成框架，用于识别合成图像的生成模型来源。

**💡 创新点**

创新点包括：① 将频域 FFT 与 RGB 结合的 FFT‑ConvNeXt；② 通过四种互补模型（FFT‑ConvNeXt、DINOv2、CLIP、Xception）在 logits 层融合并采用 K‑fold 复合；③ 针对 Stable Diffusion 3/3.5 的高度混淆，加入低置信度激活的二分类专家；④ 对腾讯 Hunyuan 进行类自适应置信度校准。

**🔧 技术方法**

使用的技术有：频域特征提取（FFT）、自监督视觉表示（DINOv2、CLIP）、卷积网络（ConvNeXt、Xception）、多模型 logits 融合、K‑fold 交叉验证、数据增强（压缩、裁剪、模糊、灰度、旋转等）、混淆矩阵分析、类自适应置信度校准。

**📊 数据集**

使用 DLMMDD Workshop Synthetic Image Attribution Challenge 数据集，包含 10 种开源文本到图像模型（AuraFlow、Freepik、Lumina、Photon、PixArt(σ)、Playground v2.5、Stable Diffusion 3、Stable Diffusion 3.5、Stable Diffusion XL‑Turbo、Tencent Hunyuan）生成的 10,000 张合成人脸图像（训练 7,000 张，测试 3,000 张），测试集包含隐藏的后处理操作。

**📈 对比分析**

方法通过与单模型基线（FFT‑ConvNeXt、DINOv2、CLIP、Xception）以及仅集成或仅 K‑fold 进行对比。最终在公开排行榜上取得 99.53% 的准确率，在私有排行榜上 99.20%，显著优于单模型和未使用专家/校准的集成结果。

**⚠️ 局限性**

局限性包括：对极端或未知后处理的鲁棒性仍有限；对高度相似生成器的区分依赖阈值调优；模型复杂度高，训练和推理成本较大；目前仅在合成人脸图像上验证，缺乏对多样场景或非人脸图像的泛化评估。

---

## 251. Hierarchical Clustering Can Jointly Satisfy Richness, Consistency, and Scale Invariance

**arXiv ID:** 2609.11173 | [PDF](https://arxiv.org/pdf/2609.11173v1)

**作者:** Daichi Kuroda `[一作]` (École Polytechnique Fédérale de Lausanne), Patrick Thiran `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在层次聚类中，尺度不变、丰富性与一致性这三条Kleinberg原定理的不可共存限制可被突破；

**💡 创新点**

创新点在于构造并证明存在无数可接受的层次聚类方法，并揭示它们在精细度上的部分序结构与共通的“良好分离”基底；

**🔧 技术方法**

主要技术包括构造基于分离度阈值的聚类方法、非二元单链接变体、Bryant-Berry稳定集方法以及利用可数极大/极小元素理论分析；

**📊 数据集**

论文以合成欧氏距离、随机离散距离以及基于真实数据集的实验验证方法性能，但聚类本身不使用公开大规模数据集；

**📈 对比分析**

与传统层次聚类（如单链、平均链）对比，本文方法在满足所有公理的前提下能产生更细粒度、结构更丰富的树形；

**⚠️ 局限性**

局限在于缺乏对实际应用中下游任务（如切分为平面簇）的深入评估，以及未给出具体最大化方法的高效实现方案。

---

## 252. Beyond Visual Quality: Evaluating Physical Consistency under Ego-Motion with EgoGenEval

**arXiv ID:** 2609.11172 | [PDF](https://arxiv.org/pdf/2609.11172v1)

**作者:** Yilin Long `[一作]` (Shanghai AI Laboratory), Tai Wang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一个用于评估姿势自由视觉生成器在执行自然语言摄像机运动指令时保持场景状态的基准——Pose-free Ego-Motion Consistency Benchmark（PEC），并构建了相同流程的训练集PEC-Train，探究对生成器的监督效果。

**💡 创新点**

创新点在于：① 将摄像机运动执行（CMG）与目标视图场景保真（SSP）分离成两个可量化指标；② 设计了三种协议（Atomic、Chain、Inverse Cycle）在多步回放中评估物理一致性；③ 通过构造与训练集相同的无场景重叠数据，验证“pairwise teacher‑forced”监督在提升运动实现上有效但对场景保真无显著帮助，揭示此监督目标是瓶颈。

**🔧 技术方法**

技术包括：使用深度相机估计器DA3与VGGT进行运动估计；使用Grounding DINO与Qwen3-VL进行对象检测与匹配；构建基于ScanNet、Matterport、HyperSim等真实室内场景的视图对；通过LoRA微调Qwen-Image-Edit和OmniGen2进行实验。

**📊 数据集**

数据集：PEC基准包含1400个案例、2360个目标视图，覆盖4种单一动作、链式三步和逆向循环三种协议；PEC-Train包含66,214条轨迹、108,213个教师强制编辑对，用于监督微调。

**📈 对比分析**

比较方法：对16个姿势自由生成器（包括闭源和开源模型）与2个姿势条件参考系统进行评估，使用CMG、SSP、RefSim、VisQual等多指标；实验显示：无系统同时在CMG和SSP两轴表现突出，平均Overall仅0.662；在单步到链式多步过程中SSP显著下降；多视图上下文虽提升CMG但降低SSP。SFT实验显示CMG可提升0.30左右，而SSP仅提升0.07，且不同模型表现差异显著。

**⚠️ 局限性**

局限性：基准仅覆盖静态室内场景、四类摄像机动作、最长三步回放；未考虑动态场景、较大视角移动和更长时序；指标依赖学习感知模型，低分可能受评估器限制；对生成器内部机制未提供深入解释，仅定位了运动与场景保持耦合的难点。

---

## 253. Agentic Share-of-Search: A Multi-Agent AI System for Competitive Decision-Making in LLM-Mediated E-Commerce

**arXiv ID:** 2609.11190 | [PDF](https://arxiv.org/pdf/2609.11190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 254. UniH$^3$: Unifying Hierarchical Homogeneity and Heterogeneity for All-in-One Medical Image Restoration

**arXiv ID:** 2609.11156 | [PDF](https://arxiv.org/pdf/2609.11156v1)

**作者:** Zhiwen Yang `[一作]` (Beihang University), Yan Xu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 UniH^3 框架，实现了统一的多任务医学图像恢复模型。

**💡 创新点**

创新性地将层级同质性记忆和层级异质性平衡两项机制相结合，既利用跨模态的解剖结构共性，又平衡任务间和任务内的分布差异。

**🔧 技术方法**

采用 Hierarchical Homogeneity Memory (H^2M) 与 Homogeneity‑Guided Attention (HGA) 提取与利用同质性先验，并通过 Hierarchical Heterogeneity Balancer (H^2B) 对多任务损失进行层级不确定性平衡；网络基于 U‑shaped Transformer。

**📊 数据集**

在自构建的 MedIR‑2D‑500K（509k 2D 图像对）和 MedIR‑3D‑3K（3522 3D 卷体对）两个大规模基准上进行训练和评估。

**📈 对比分析**

与多种单任务与全任务恢复方法（SwinIR、Uformer、Restormer、PromptIR、AdaIR 等）以及 3D 任务的 state‑of‑the‑art 方法对比，UniH^3 在所有任务上均取得了最高的 PSNR/SSIM，平均提升约 0.2–0.5 dB。

**⚠️ 局限性**

仅关注每种模态的主要恢复任务，未覆盖多种降质类型或子任务；且模型对极端低质量输入的鲁棒性尚待进一步验证。

---

## 255. MUtE: A Dual Framework for Concept Erasure and Counterfactual Interventions

**arXiv ID:** 2609.11253 | [PDF](https://arxiv.org/pdf/2609.11253v1)

**作者:** Antoine Saillenfest `[一作]` `[通讯]` (onepoint), Antoine Saillenfest (onepoint)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MUtE*框架，基于最优概念擦除实现对连续表示的确定性双向反事实映射；

**💡 创新点**

创新点在于将信息论最优擦除界与反事实生成统一，推导出可实现的平移偏置迭代密度匹配实现；

**🔧 技术方法**

采用RBIG迭代高斯化、类条件差分高斯化、正交旋转、oracle分类器及可微逼近（MLP或INN）等技术；

**📊 数据集**

使用GloVe、BERT、DeepMoji、GPT‑4等多语言模型嵌入以及合成二维高斯混合数据；

**📈 对比分析**

与FaRM、KRaM、TaCo、LEOPARD、LEACE及线性擦除方法比较，MUtE*在概念预测准确率降至基准、重构误差低、下游任务准确率保持或提升，并显著改善TPR^RMS与DP；在反事实文本生成中，BERTScore高、性别替换率最佳；

**⚠️ 局限性**

仅适用于单一离散概念，连续属性或多重交叉概念擦除不支持；假设线性平移可能不符合所有因果因素，推理时需多步迭代，需额外蒸馏；

---

## 256. You Get What You Sample: Evaluating Sampling Strategies for Web Security Measurements

**arXiv ID:** 2609.11218 | [PDF](https://arxiv.org/pdf/2609.11218v1)

**作者:** Xuenan Zhang `[一作]` (CISPA Helmholtz Center for Information Security), Giancarlo Pellegrino `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对 Web 安全测量中常用的八种采样策略（Top N、随机、系统、分层、桶、混合等）进行系统性评估，量化其对影响（impact）和普及率（prevalence）测量的误差、偏差与稳定性；

**💡 创新点**

首次揭示 Top N 采样在跨前缀估计时存在持续偏差，混合采样无法消除这一前缀偏差，并提出自适应概率采样（Adaptive Probability Sampling）在未知漏洞流行度时能在 4–16% 样本量下达到 95% 以内误差的创新方法；

**🔧 技术方法**

采用大规模网页爬虫（Foxhound+Playwright）进行动态污点跟踪、TLS 错误日志采集与 HTTP 头检查，并用统计指标（误差幅度、峰值、RMS、ZCR）与 Shapley 值分解分析混合采样的前缀与尾部贡献；

**📊 数据集**

使用 Tranco 500k 热门域名作为影响测量基准，使用 Common Crawl 24.8 M 主机作为普及率基准，对两大数据集进行完整测量和随机抽样对比；

**📈 对比分析**

在多达 50 次独立采样实验中，概率采样误差均低于 0.3% 且峰值小于 0.5%，而 Top N 误差可达 8% 以上；混合采样受前缀偏差影响，除非尾部样本足够大；自适应采样在常见问题下仅需 4–16% 样本即可实现 95% 以内误差；

**⚠️ 局限性**

局限性包括仅评估静态安全缺陷（缺乏动态漏洞验证），爬虫受 CDN、DNS 与网络差异影响，样本分布可能不代表更大或更异构的 Web，且未覆盖全部安全类别，需在更大规模和不同结构的数据上进一步验证。

---

## 257. SemVerBench: Benchmarking LLM Comprehension of Version-Constraint Resolution Semantics

**arXiv ID:** 2609.11180 | [PDF](https://arxiv.org/pdf/2609.11180v1)

**作者:** Qibai Chen `[一作]` (Independent Researcher), Zeming Liu `[通讯]` (Brown University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 SemVerBench——一个针对 npm、PEP 440、Cargo 三大生态的 LLM 版本约束解析基准；使用 240 条机器可检验的题目评估了六款前沿 LLM 的解析能力，并针对其盲点提出规则注入和工具委托等缓解措施。

**💡 创新点**

创新点在于：①首次量化 LLM 对版本约束语义的理解误差；②采用作者中立、两实现交叉验证的判定方法，消除人工或单一实现偏差；③揭示了 LLM 在特定语义机制（partial‑comparator、prefix‑match、ordered‑exclusion）上的系统性缺陷，并证明“激活/应用”缺口而非知识缺失是主要原因；④提供可直接复现的基准数据和代码。

**🔧 技术方法**

技术手段包括：①对比解析器实现（npm：node‑semver、Cargo：rust‑semver、PEP 440：Python & Rust 两实现）作为 oracle；②使用三次温度为 0 的 API 调用；③统计 Wilson 置信区间与 McNemar 精确检验；④对模型进行规则注入、区间拆解、工具调用三种干预实验。

**📊 数据集**

数据集为 240 条独立题目，其中 80 条来自每个生态的官方测试套件，另外 160 条由三款 LLM（Claude、GPT、Gemini）按统一提示生成；额外提供 67 条针对 PEP 440 prefix‑match 角落情况的验证集。

**📈 对比分析**

比较方法：在同一 240 条题目上评测六款模型（Claude‑Opus、Claude‑Sonnet、GPT‑5.1、GPT‑4.1、GPT‑4o、Gemini‑2.5‑Pro）。总体准确率从 80 %（GPT‑4o）到 90.6 %（Claude‑Opus），Claude 系列显著优于其余；在关键机制上发现准确率低于 60 % 的盲点；规则注入将多数模型提升至 90–95 %，工具委托接近 100 %。

**⚠️ 局限性**

局限性包括：①评测模型亦是题目生成者之一，虽通过自利性检验控制，但仍存在潜在偏见；②部分机制桶样本量小，统计意义有限；③仅评估单独解析任务，未考察错误传播到完整编码代理的链路；④使用官方测试套件可能受训练数据泄露影响；⑤规则注入与激活/应用假设尚未完全分离。

---

## 258. Breaking Predictions Is Not Enough: Specified-Foil Counterfactuals for Temporal Graphs

**arXiv ID:** 2609.11170 | [PDF](https://arxiv.org/pdf/2609.11170v1)

**作者:** Minwoo Yu `[一作]` (Konkuk University), Young-guk Ha `[通讯]` (Konkuk University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了针对时序图预测的指定-foil反事实框架，寻找最小代价的过去事件干预，使得给定的未来事件成为预测结果的最高排名。

**💡 创新点**

创新点在于：①将预测无效化与指定-foil反事实区分开来；②设计基于可执行符号推理轨迹的对比干预搜索，将执行差异映射为 DELETE、INSERT、REWIRE、RELABEL、SHIFT 等操作；③通过 LiFTER 与 TLogic 的可执行追踪实现。

**🔧 技术方法**

技术核心包括可执行符号推理与规则执行追踪、执行差异对比生成干预候选、低成本编辑组合搜索、精确重放验证以及基于编辑预算的求解。

**📊 数据集**

实验使用四个连续时间动态图数据集（Wikipedia、Reddit、MOOC、LastFM）以及两个时间知识图数据集（ICEWS14、ICEWS18）。

**📈 对比分析**

与随机、最近邻、黑盒贪婪搜索及穷举搜索对比，Trace-guided 在 CTDG 上保留 85–94% 的黑盒成功率并将评估次数减少 75–80%；在 TKG 上保留 95–99% 成功率，评估次数减少 60%；在受限编辑空间内恢复约 75–91% 的完整解，评估次数下降约 90%。

**⚠️ 局限性**

局限性包括：仅适用于可执行推理模型；编辑空间有限（最多两步、固定候选）；模型级干预不等同真实因果干预；对不可编辑的实体/关系缺乏完整性约束；在大规模事件空间仍面临搜索复杂度挑战。

---

## 259. Fast and Accurate Monomodal 3D High Resolution Deep Registration of Drosophila Larval Brain Volumes

**arXiv ID:** 2609.11240 | [PDF](https://arxiv.org/pdf/2609.11240v1)

**作者:** Daniel Reisenbüchler `[一作]` (University of Regensburg), Dorit Merhof `[通讯]` (University of Regensburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于深度学习的Drosophila幼虫脑体积配准网络DLBR，能够在高分辨率（64×768×512）下一次性完成变形配准；

**💡 创新点**

创新点在于：①将配准任务迁移到一次前向推断，显著降低计算成本；②在高分辨率下实现最优配准精度，尤其在解剖标记局部互信息上超过传统方法23个百分点；③引入针对高质量分辨率脑图像的无监督训练策略（NCC损失+平滑正则），并验证其对成像质量变化的鲁棒性；

**🔧 技术方法**

使用VoxelMorph式3D U‑Net（Encoder-Decoder + refinement blocks）作为变形预测器，训练时采用归一化互相关(NCC)作为相似性损失，平滑正则化；实现了三阶三维空间变换器（trilinear）；

**📊 数据集**

训练数据来源于Janelia实验室收集的540幅高质量幼虫脑图像（分辨率多样），验证与测试集为完全独立的Larvalign 66幅脑图像（按随机/中等/高质量分层）；

**📈 对比分析**

与11种经典配准方法（刚性、仿射、B‑spline、Demons、ANTs SyN、NiftyReg等）和7种先进的学习式配准网络（TransMorph、ViT‑V‑Net、LH‑Morph、MambaMorph、LapIRN、RDP、Fourier‑Net）进行比较。DLBR在所有指标（局部互信息、全局互信息、全局Pearson相关、局部MSE）均优于所有传统方法，在学习方法中位居榜首，速度提升达两位数（从数十秒到0.43秒），同时保持最高的鲁棒性；

**⚠️ 局限性**

主要限制：训练阶段受GPU显存约93 GB限制，最大可训练分辨率约为45 兆体素；超出该阈值（如64×1536×1024）无法完成训练；此外模型是单一实验室数据训练，尽管在外部数据集表现良好，但在更极端的成像条件下仍可能受限。

---

## 260. An AI-Powered Culturally Aware Chatbot for Stress Detection and Wellness Support among Pakistani University Students Using NLP and Machine Learning

**arXiv ID:** 2609.11199 | [PDF](https://arxiv.org/pdf/2609.11199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 261. When is Test-Time Adaptation Identifiable From Unlabeled Evidence?

**arXiv ID:** 2609.11235 | [PDF](https://arxiv.org/pdf/2609.11235v1)

**作者:** Kartik Jhawar `[一作]` (Nanyang Technological University), Lipo Wang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在无标签的测试时适配（TTA）中，观测通道能否识别最优适配动作，并给出了信息可识别边界。

**💡 创新点**

首次将TTA选择视为信息识别问题，证明在有限批量下存在不可识别的边界，并在理论和公开基准上验证。

**🔧 技术方法**

构造了有限批量Gaussian模型的KEEP-RECENTER边界，利用TV距离两点下界，以及在CIFAR-100-C与DomainNet-126上进行离散实验。

**📊 数据集**

CIFAR-100-C 与 DomainNet-126（含多种失真、方向转移与部署结构）。

**📈 对比分析**

对比了全局与序列感知的证据渠道（Z1~Z4、ORDER11）以及 MORPHEUS 等基线，发现序列感知的 Z3 能将 oracle regret 降低到 0.3个百分点，达到 93% 以上的动作准确率；但在随机部署下其优势消失。

**⚠️ 局限性**

理论基于单一动作集合与一维 Gaussian 模型，实验仅覆盖特定源模型与动作菜单；对更复杂的多模态适配器与更广泛的部署族仍需验证。

---

## 262. NovGauge: A Fine-Grained Benchmark for Diagnosing LLMs' Capability in Paper Novelty Assessment

**arXiv ID:** 2609.11234 | [PDF](https://arxiv.org/pdf/2609.11234v1)

**作者:** Guoqiang Zhang `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 NovGauge 基准，提供 3 维细粒度新颖性评估与可信度级联检查，以诊断 LLM 在学术评审中的新颖性判断。

**💡 创新点**

创新点在于将新颖性拆解为任务、问题、方法三维并配合层级化的可信度评估（准确性、幻觉率、逻辑匹配），实现对模型判断错误来源的细粒度诊断。

**🔧 技术方法**

采用 LLM 自动提取与人工校验相结合的标注流程，构建级联评估管道，并在 18 种大型语言模型上进行对比实验。

**📊 数据集**

使用 619 篇论文对和 50 组多论文集合（来自 ICLR 评审重叠声明与 CS arXiv 调查共引组）作为数据集。

**📈 对比分析**

通过 Raw F1、幻觉率、匹配率三阶段级联得到 Verified F1；结果显示大模型 Raw F1 较高但 Verified F1 仅保持约 25% 以上，方法维度最差；GPT‑5.5 在三维上均表现最佳，长文本更易导致可信度下降。

**⚠️ 局限性**

局限性包括：仅覆盖 ICLR/CS 领域，评估依赖 LLM 判定器，维度负样本不平衡，可能受 LLM 撰写评审的影响，且未评估完整检索流程。

---

## 263. TripleBound: Triplet-Guided Heterogeneous Graph Learning for Microservice Decomposition

**arXiv ID:** 2609.11212 | [PDF](https://arxiv.org/pdf/2609.11212v1)

**作者:** Mineth Weerasinghe `[一作]` (University of Moratuwa), Srinath Perera `[通讯]` (WSO2 LLC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了TripleBound框架，自动将单体应用拆分为微服务。

**💡 创新点**

将弱监督的三元组约束直接注入到异构图神经网络的共享潜在空间中，实现结构和语义信息的联合优化；同时引入通信感知正则化。

**🔧 技术方法**

异构图神经网络（CHGNN）、triplet-loss、k-means聚类、交互式通信损失等。

**📊 数据集**

四个公开的Java单体基准：AcmeAir、DayTrader、PlantsByWebSphere、JPetStore。

**📈 对比分析**

与结构化方法CHGNN和语义方法MonoEmbed比较，TripleBound在AcmeAir、DayTrader和JPetStore上获得最高复合得分（SM加权最高），但在PlantsByWebSphere上表现不如CHGNN；单指标上存在权衡，整体性能依赖于数据集大小和指标权重。

**⚠️ 局限性**

方法对数据集和指标权重敏感，未对各损失组件进行单独消融；未提供统计显著性检验；对大型工业系统的泛化尚未验证；使用的三元组标签可能泄露已有模块信息。

---

## 264. (Whose defaults?) Is artificial intelligence reorienting archaeological methods?

**arXiv ID:** 2609.11198 | [PDF](https://arxiv.org/pdf/2609.11198v1)

**作者:** Lorenzo Cardarelli `[一作]` (Georg-August University Göttingen), Roberto Ragno `[通讯]` (University of Cambridge)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2010–2025年119,327篇Archaeology学科Scopus摘要进行自动化方法抽取与聚类，并通过本地LLM（Qwen3.5-9B与Gemma 4E4B）在三种提示强度下生成方法建议，进一步检验LLM是否在文献中引发方法聚合。

**💡 创新点**

首次将大规模语言模型的推荐行为与文献计量学相结合，提出并检验“均值坍塌（mean‑collapse）”假设，揭示LLM在方法选择上潜在的偏向与多样性变化。

**🔧 技术方法**

利用基于上下文窗口的LLM抽取器、规则+语义聚类+LLM标注构建两级以上方法分类法；使用贝叶斯层级Dirichlet‑Multinomial回归评估时间趋势，采用逆Simpson指数衡量方法多样性；对LLM推荐计数使用负二项回归探究其与文献预热度与后期变化的关系。

**📊 数据集**

核心数据集为119,327篇Archaeology领域的Scopus原始摘要（2010–2025年），其中约8,404篇报告至少一项计算方法；LLM推荐实验包含28个标准化研究问题、三种提示强度、两款模型共计756次生成，产生2,904和1,746条推荐方法。

**📈 对比分析**

结果显示文献层面方法多样性在2023后略增（有效方法数从87.6升至111.2，置信区间均显著），而LLM推荐的多样性始终低于文献（平均有效方法数约31.6/28.8），且新手提示下浓度最高；负二项回归表明预热度是推荐频率的主要预测因子，后期增量无显著效应，表明LLM在当前阶段并未强制导致方法收敛。

**⚠️ 局限性**

局限包括：仅分析摘要信息，缺乏对完整论文方法实施细节的验证；LLM训练语料的偏差与生成过程的不可解释性；2023年前后仅三年窗口，可能不足以捕捉LLM普及后的长期效应；以及对不同学科内部数据分布差异的未能充分分离。

---

## 265. FST Pay: Deterministic Safety-Gated Architecture for Youth Digital Payments

**arXiv ID:** 2609.11195 | [PDF](https://arxiv.org/pdf/2609.11195v1)

**作者:** Shaikh Mohammed Burhan `[一作]` (Khaja Bandanawaz University), Tabassum Nahid Sultana `[通讯]` (Khaja Bandanawaz University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向青少年数字支付的确定性安全架构——FST Pay，利用六个不变式实现严格的实时授权，并在授权完成后通过异步生成式 AI 生成可解释的财务教育内容。

**💡 创新点**

创新点在于：①将支付授权与生成式 AI 完全解耦，保证授权路径的确定性与可审计；②设计多维度安全不变式（滚动速度、自动清算、单笔上限、商户分类、时段限制、硬件完整性）并以严格的决策函数输出 ALLOW/REVIEW/BLOCK；③通过非干涉属性与能力隔离保证后置 AI 仅读写日志；④构建两阶段流水线与可审计关系型数据库。

**🔧 技术方法**

核心技术包括：确定性 ABAC 与有限状态机、Redis 滑动窗口计数、PostgreSQL Aurora（ACID 双录本）、Apache Kafka 事件流、TLS 1.3 与 mTLS、JWT/OAuth2、双因素 MFA、Python 异步 Worker 调用 LLM 生成文本。

**📊 数据集**

论文未公开使用真实交易数据集，示例使用合成或行业标准的 UPI 交易日志进行单元测试；后置 AI 仅接收已完成的交易事件，输入中去除了 PII。

**📈 对比分析**

比较方法：通过理论对比表将 FST Pay 与纯概率机器学习、生成式 AI 代理、传统静态银行模型对比；评估框架给出指标（p99<15 ms、零误准、Guardian 延迟、系统故障隔离、审计完整性），但目前尚未给出实验性能数据，目标在未来测试中验证。

**⚠️ 局限性**

局限性包括：①缺乏大规模真实部署与实验验证；②父母响应时延导致 REVIEW 阶段延迟；③商户分类可能被欺骗导致规避不变式；④缺少形式化验证工具支持安全属性；⑤未评估 AI 生成解释的质量与教育效果。

---

## 266. Can LLMs Follow Medical Expert Logic? A Benchmark for Hierarchical Logical Consistency in Risk-of-Bias Assessment

**arXiv ID:** 2609.11185 | [PDF](https://arxiv.org/pdf/2609.11185v1)

**作者:** Jiayu Huang `[一作]` (Beijing University of Posts and Telecommunications), Haihong E `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 LogiMed‑RoB 基准与 HLC（分层逻辑一致性）框架，用于系统评估 LLM 在 Cochrane RoB 2.0 规则下的多级推理与证据可靠性。

**💡 创新点**

创新点在于：① 将 RoB 2.0 的专家决策逻辑抽象为确定性规则集；② 设计四维度评估（原子一致性、领域一致性、聚合一致性、证据可信度）；③ 通过实验揭示“错误累积效应”与“证据‑推理缺口”。

**🔧 技术方法**

采用 HLC 框架对 LLM 输出进行后置审计，计算 CAR、LF、VR 等指标，并利用 Jaccard 与阈值评估证据可靠性；对 10 种 LLM 进行统一提示、温度 0.0 的 deterministic 评估；实验还引入 CRAG 与 CoT 作为提升手段。

**📊 数据集**

使用 LogiMed‑RoB 数据集：Track A 860 篇 RCT（626 例、13,772 问题）来源于 Cochrane 与 Figshare，Track B 659 篇 RCT（1,048 项）来源于 ROBIN 与 RoBBR，共计 14,820 个查询。

**📈 对比分析**

将 10 个代表性 LLM 与基准对比；结果显示最高原子一致性 98.88% 的 Gemini 3.1 Pro 端到端一致性仅 45.13%；Blind Guess Rate 最高达 48.28%，Reasoning Failure Rate 在 18.63–40.05% 范围；Open‑weight 模型往往在多步推理上表现较差，凸显错误累积和证据‑推理缺口。

**⚠️ 局限性**

局限性包括：① 仅关注 RoB 2.0 规则，未覆盖更广泛的临床推理任务；② 数据集全为英文，跨语言能力未评估；③ 仅做一次 T=0 的 deterministic 调用，未估计结果方差；④ 专有模型的更新与长期追踪难以实现。

---

## 267. Sci-MMR: Benchmarking Multi-Step Evidence-Grounded Scientific Reasoning in Multimodal Agents

**arXiv ID:** 2609.11243 | [PDF](https://arxiv.org/pdf/2609.11243v1)

**作者:** Jiaqiang Li `[一作]` (Fudan NLP Group), Tao Gui `[通讯]` (Fudan NLP Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提供了Sci‑MMR的补充材料，包括基准构建、审核细节、评估协议、额外诊断、错误分析、代表案例及提示模板。

**💡 创新点**

创新点在于系统化整合基准与评估流程，并提供可复现的审核与诊断报告。

**🔧 技术方法**

采用基准构建工具、审核脚本与评估脚本，以及提示生成技术。

**📊 数据集**

使用Sci‑MMR基准数据集及其公开的子集进行实验。

**📈 对比分析**

与现有基准对比，补充材料展示了更全面的性能评估与错误剖析，显示出模型在多模态推理任务上的优越性。

**⚠️ 局限性**

局限性在于缺乏原始实验细节，补充材料仅提供方法论与诊断结果，未包含完整的模型训练与评估代码。

---

## 268. From Evaluation to Enhancement: Benchmarking and Improving Think-with-Video Reasoning for Video Generative Models

**arXiv ID:** 2609.11242 | [PDF](https://arxiv.org/pdf/2609.11242v1)

**作者:** Meng Luo `[一作]` (National University of Singapore), Hao Fei `[通讯]` (University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个完整的思维-视频（Think‑with‑Video）评估框架，并基于此开发了一种无模型、可插拔的提示重写器Vid‑PRE，显著提升现有视频生成模型的推理能力。

**💡 创新点**

创新点在于：①设计了涵盖9个推理维度、38个细粒度任务的VWG‑Bench，并引入三层VLM‑as‑Judge细粒度评估；②提出Vid‑PRE，通过先训练生成CoT再生成约束感知提示，使用SFT + GRPO的纯文本奖励方式实现模型无关的推理增强。

**🔧 技术方法**

核心技术包括：VLM‑as‑Judge评估协议、自动化图像对视频提示生成管道、监督微调（SFT）与群组相对策略优化（GRPO）结合的强化学习、纯文本奖励设计、以及与多种视频生成模型的无缝集成。

**📊 数据集**

使用的数据集：VWG‑Bench（自建的9维度38任务共380个样本，10个实例/任务）；MME‑CoF（59个图像‑提示对，用于外部验证）；V‑ReasonBench（12类共327个图像‑提示‑真值三元组）。数据由公开数据集、程序化生成与T2I模型自动生成并通过VLM筛选得到。

**📈 对比分析**

比较方法：在VWG‑Bench上评估六款模型（Wan2.2、Wan2.5、Wan2.6‑Flash、Kling‑2.5‑Turbo‑Pro、Sora2、Veo3.1），在不使用提示重写器时，视频质量分高但推理分低；加入Vid‑PRE后，所有模型的推理分均有显著提升，VWG‑Bench整体分从2.13提升至2.51（18%），MME‑CoF从1.72提升至2.58（≈50%），V‑ReasonBench从26.65提升至44.48（≈67%）。Vid‑PRE优于模型自带的Prompt Extension，证明其普适性。

**⚠️ 局限性**

局限性：①评估样本量相对有限（每类10个实例），可能不足以覆盖极端或多样化场景；②VLM‑as‑Judge依赖第三方大模型，评估结果受其理解误差影响；③Vid‑PRE仅通过文本奖励进行优化，未直接利用视频质量信号，可能在视觉细节层面有限提升；④对极长时序或高度开放式任务的适用性尚待进一步验证。

---

## 269. HALDETECT at ImageEval 2026 Shared Tasks: Answer-First Contrastive Grounding with QLoRA

**arXiv ID:** 2609.11236 | [PDF](https://arxiv.org/pdf/2609.11236v1)

**作者:** Syed Mohaiminul Hoque `[一作]` (Independent University Bangladesh), Md Sakhawat Hossain `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于对比式判定、先给出答案再给解释的图像幻觉检测系统，冻结视觉编码器，仅对语言层做低位LoRA微调。

**💡 创新点**

创新点包括：①将三条陈述合并为单次推理并强制仅输出一个True；②使用“先答再推理”顺序显著降低误判；③构造可检查的属性清单（颜色/纹理、形状/形式、场景上下文）指导推理；④在仅一轮训练、双T4 GPU预算下实现高性能。

**🔧 技术方法**

采用 Qwen2.5‑VL‑7B‑Instruct（4‑bit NF4 LoRA、rank‑8），冻结视觉塔；在训练时使用 256×28×28 视觉分辨率，推理时 1024×28×28；在推理时利用属性清单和答案‑优先提示。

**📊 数据集**

使用 ImageEval 2026 Task 1b 训练集（3 000 条），验证集（500 条）和测试集（1 000 条），源自 OASIS/Ayn‑VQA 的阿拉伯文化图像。

**📈 对比分析**

与单独判定、顺序反向、属性清单及多模型投票等对比，系统在官方 CI（Contrastive Instability）指标上达到 0.035，第三名（CI 0.039–0.040），显著优于仅提示的 0.062，且在训练规模变化和随机种子重设下效果不稳。

**⚠️ 局限性**

局限性包括：仅在单一随机种子下训练，未探索更长训练、不同 LoRA rank 或可训练视觉塔；仅用 1 轮训练，可能欠拟合；缺乏对解释可信度的评估；在不同种子和规模下性能波动较大，易受偶然性影响。

---

## 270. Solving Few-Shot Multiobjective Multitask Optimization via Iterative Sequential Transfer

**arXiv ID:** 2609.11228 | [PDF](https://arxiv.org/pdf/2609.11228v1)

**作者:** Tingyang Wei `[一作]` (Nanyang Technological University), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种迭代序列迁移（IST）框架，用于在极低评估预算下解决多目标多任务优化问题，并给出了基于似然的任务优先机制。

**💡 创新点**

创新点在于将多任务优化转化为一系列序列迁移优化任务，并通过统计似然信息动态选择目标任务，从而显著降低负迁移并实现自适应资源分配。

**🔧 技术方法**

主要技术包括多任务高斯过程（MTGP）、前向-逆向映射的F-invTrEMO以及AMTEA等序列迁移优化器，并辅以Tchebycheff标量化、软最大采样等方法。

**📊 数据集**

实验数据集涵盖九个标准多目标多任务基准（如CIHS、CIMS、PILS等）以及真实的多目标超参数优化任务（RF在信用、医疗、语音等数据集上）。

**📈 对比分析**

与单任务ParEGO以及未使用IST的F-invTrEMO进行对比，使用IGD+评估；结果显示IST在绝大多数任务上显著优于基线，尤其在任务相似度低的情况下提升更为显著。

**⚠️ 局限性**

局限性包括：当任务间搜索空间差异极大或目标维度不匹配时，似然优先机制可能仍导致负迁移；此外，框架在高维大规模任务上的适用性和鲁棒性尚待进一步验证。

---

## 271. AI Soccer Analyst: Stage-Aware and Verifiable Human-AI Collaboration for Soccer Data Analysis

**arXiv ID:** 2609.11224 | [PDF](https://arxiv.org/pdf/2609.11224v1)

**作者:** Calvin Yeung `[一作]` (Nagoya University), Keisuke Fujii `[通讯]` (Nagoya University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了 AI Soccer Analyst，一套基于 LLM 的阶段化混合主动式系统，用于协助足球比赛数据分析，从数据理解、问题定义、规划、执行、证据支撑报告到交互完善四个可视化可修订阶段。

**💡 创新点**

创新点在于将分析流程拆解为可检查、可修订的各阶段，并通过确认门控与证据链实现人机协作的可验证性与可审计性，允许分析师在关键点插手、修订，保持对关键决策的掌控。

**🔧 技术方法**

技术上使用本地部署的 OpenAI 120B Mixture‑of‑Experts LLM（via vLLM）、Aider 代码生成、Docker 隔离的编程工作区、FastAPI 后端与 Next.js 前端，以及多代理（Analyst、Coder、Review、Refinement）实现工作流。

**📊 数据集**

使用公开的 Wyscout 2017 事件数据集（约 300 万条事件记录）作为实验数据源。

**📈 对比分析**

通过在 16 名足球分析师上进行 48 个任务（L1~L3 难度水平）的实验，记录任务完成率、评估指标（输出质量、任务完成度、可靠性、可验证性）并对结果进行 Wilcoxon 检验，结果显示完成任务的评价均显著高于中性水平，任务完成率为 33/48。

**⚠️ 局限性**

局限性包括：对数据源的结构与可用性识别不足导致任务不可行；仅在单一事件数据集上验证，未对其他供应商或跟踪数据进行迁移测试；缺乏客观错误检验和与传统工具的对比；样本规模与参与者背景有限，不能充分覆盖所有分析师角色；系统对资源限制（内存、执行时间）的处理仍不完善。

---

## 272. Tri-DehazeGS: Scene--Medium Decoupled Gaussian Splatting with Transmittance-Aware Optimization

**arXiv ID:** 2609.11223 | [PDF](https://arxiv.org/pdf/2609.11223v1)

**作者:** Kui Jiang `[一作]` (Harbin Institute of Technology), Hui Liu `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种利用3D Gaussian Splatting框架，结合视图共享的三平面 fog 场景表示，来从多视角雾蒙蒙图像中恢复干净3D场景的技术。

**💡 创新点**

创新点包括：①将场景辐射与大气介质完全解耦，采用独立的三平面字段来表示可变的散射介质；②引入 Medium‑Decoupled Transmittance Gradient Compensation (MD‑TGC)，在透射率冻结后仅在反向传播阶段补偿低透射区域梯度，从而恢复被雾抑制的优化信号。

**🔧 技术方法**

使用的主要技术有：3D Gaussian Splatting、三平面稀疏场表示、低阶球谐光照、透射率梯度补偿、深度先验（单目伪深度）和暗通道先验。

**📊 数据集**

实验数据集包括 RealX3D（真实多视角雾景）、Mip‑NeRF 360（合成雾景）和 Fog‑NeRF（合成雾景）。

**📈 对比分析**

与多种基线（传统 3DGS、图像去雾+3DGS、Fog‑NeRF 等）在 PSNR、SSIM、LPIPS 上进行对比，Tri‑DehazeGS 在所有基准上均表现出最高的 PSNR/SSIM 与最低的 LPIPS，显著提升了干净视图合成质量。

**⚠️ 局限性**

主要局限在于对深度先验的依赖，深度误差会导致边缘模糊和细节丢失，尤其在薄结构或纹理稀疏区域表现不佳。

---

## 273. Diversity of EML-type operators

**arXiv ID:** 2609.11210 | [PDF](https://arxiv.org/pdf/2609.11210v1)

**作者:** Andrzej Odrzywołek `[一作]` `[通讯]` (Jagiellonian University), Andrzej Odrzywołek (Jagiellonian University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并系统梳理了 EML 型二元算子及其多种变体，阐明其在构造所有基于指数与对数的初等函数中的完整性与可计算性，并探讨了其在符号回归与神经网络中的潜在应用；

**💡 创新点**

首次揭示 EML 及其族群在实现所有初等函数时的唯一算子特性，提出 Möbius 层神经网络框架以及可通过单一非线性激活函数（如 F(z)=e^z+lnz）实现解析初等函数表达式的可能性；

**🔧 技术方法**

利用暴力搜索与数值常数识别技术构造算子，采用符号计算（Mathematica、Lean4 证明）验证公式，提出 Stachowiak 的通用生成框架，进一步通过复杂数与分式运算扩展传统神经网络；

**📊 数据集**

本文为理论综述与方法设计，并未依赖具体实验数据集；

**📈 对比分析**

未进行实验对比；主要通过符号检验与理论证明展示 EML 族能重构所有初等函数，Möbius 层网络被视为理论可行性示例；

**⚠️ 局限性**

仍缺乏高效优化路径、规模化训练实验与数值稳定性保证；对 EML 族通用性与最简构造的系统性研究尚未完成。

---

## 274. Conceptualising an Initial Design Space for Guidance in Digital Physical Activity Support

**arXiv ID:** 2609.11193 | [PDF](https://arxiv.org/pdf/2609.11193v1)

**作者:** Faith Young `[一作]` (University of Salzburg), Jan Smeddinck `[通讯]` (Ludwig Boltzmann Institute for Digital Health and Prevention)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了数字化支持物理活动（PA）的“指导”概念，并构建了基于九个维度（范围、目的、时机、情境、模式、具身化、适应性、自主性、情感质量）的设计空间；

**💡 创新点**

创新点在于将指导从单纯的动机或信息传递提升为情境化、行动导向、具身化的支持机制，并为指导提供了系统化、可操作的多维框架；

**🔧 技术方法**

该研究未使用具体技术实现，而是通过文献综述与概念综合形成理论框架；

**📊 数据集**

未使用任何数据集，全部基于已有理论与文献；

**📈 对比分析**

未进行实验比较或性能评估，框架目前仅处于概念验证阶段；

**⚠️ 局限性**

局限性包括：概念性方法未得到实证验证；文化背景和技术演进可能导致框架需进一步调整；缺乏对不同人群和环境的适用性评估。

---

## 275. E-CONAN (Entailment, CONtradition And Neutral) Benchmarks: Arabic Textual Entailment and Natural Inference Datasets

**arXiv ID:** 2609.11334 | [PDF](https://arxiv.org/pdf/2609.11334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 276. Physics of Information Geometry - Part II: Small-Step Active Inference on the Probability Simplex

**arXiv ID:** 2609.11187 | [PDF](https://arxiv.org/pdf/2609.11187v1)

**作者:** C. Emre Koksal `[一作]` (Ohio State University), Deniz Sargun `[通讯]` (Amazon.com Inc.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了在概率单形上通过小步递推实现主动推理的离散最小作用原理框架，推导出相对自由能、相对势能与KL散度定义的动力学；

**💡 创新点**

创新点在于将信息几何的毕达哥拉斯定理与自由能最小化结合，证明小步递推比一次大跳更高效，并给出最优更新的闭式指数倾斜公式；

**🔧 技术方法**

采用信息几何、KL散度的毕达哥拉斯定理、拉格朗日变分、最小作用原理以及指数倾斜更新；

**📊 数据集**

无真实数据集，使用理论推导与数值模拟（如三维单形下均匀初始分布到目标分布的路径）；

**📈 对比分析**

通过对不同步长δ的数值实验与单次跳跃KL距离做对比，显示小步递推可将总KL能量降低至直线跳跃的6%以内，达到约94%的能量节约；

**⚠️ 局限性**

局限在于假设分布全支持、步长足够小、仅考虑离散步长约束，未对真实主动推理或大规模问题进行实验验证，且对非平滑目标分布的收敛性未做严格分析。

---

## 277. The Computational Complexity of Holant Problems on 4-regular Graphs from the Stable Subgroup Sequence of $SL(2,\mathbb{C})$

**arXiv ID:** 2609.11175 | [PDF](https://arxiv.org/pdf/2609.11175v1)

**作者:** Yuan Huang `[一作]`, Zhiguo Fu `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对在Holant框架下仅包含二元相等约束(=_2)与一个任意四元复数签名f的计数问题进行完整的复杂度分类，给出了一个二分结论：若f属于若干特殊可变换族或张量可分解族，则问题可在多项式时间内求解；否则问题为#P‑hard。

**💡 创新点**

创新点在于首次将Holant问题的复杂度归类与SL(2,ℂ)的有限子群（C_n、BD_{4n}、BT_{24}、BO_{48}、BI_{120}）的结构联系起来，利用二元子群序列、群闭包与玻色化变换构造复杂度硬度判定，并引入了“二元子群序列稳定性”概念，显著扩展了以往仅针对布尔或实数域的结果。

**🔧 技术方法**

核心技术包括：holographic（玻色）变换、二元签名的矩阵表示与矩阵乘法、有限群分类（Schur定理）、Gadget构造与可约化、以及对八顶点形式（eight‑vertex）签名的线性系统分析。

**📊 数据集**

该研究不依赖任何实验数据集，而是完全在理论框架下进行证明，所涉及的“数据集”仅为符号签名本身。

**📈 对比分析**

由于结果为理论证明，未做实验比较；论文通过与已知的#P‑hard与多项式可解类的对照，确认其分类的完备性与最优性。

**⚠️ 局限性**

局限性包括：仅覆盖四元签名；高元（>4）签名的完整分类仍未解决；对实数域的情况虽已知但未在此框架内细化；若允许更一般的约束（如不等价约束）需进一步研究。

---

## 278. Single-Exponential Algorithms and a Polynomial Kernel for Strong Connectivity Augmentation

**arXiv ID:** 2609.11160 | [PDF](https://arxiv.org/pdf/2609.11160v1)

**作者:** Tomohiro Koana `[一作]` (University of Tokyo), Soh Kumabe `[通讯]` (CyberAgent)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了针对强连通性增补（Strong Connectivity Augmentation）问题的单指数参数化算法与多项式核化技术；

**💡 创新点**

创新点在于将该问题转化为带两种成本的强连通支撑子图（SCSS）问题，随后利用动态规划和Mader拆分定理实现9^k时间复杂度，并通过Frank–Tardos权重压缩获得O(k^4)顶点、O(k^16)位数的多项式核；

**🔧 技术方法**

核心技术包括：层次化最短路径计算、SCSS的两个成本动态规划（ear decomposition + Held‑Karp子集卷积）、Mader定理的应用，以及Frank–Tardos权重压缩；

**📊 数据集**

本研究为理论计算复杂性工作，未使用任何实验数据集；

**📈 对比分析**

与之前的2^O(k log k)算法相比，本文将时间复杂度降至单指数9^k（无权时为4^k），并首次给出多项式核；在理论上实现了更优的参数化效率；

**⚠️ 局限性**

局限性包括：算法仍为指数级，对大k值仍不切实际；核大小虽为多项式但仍为O(k^4)顶点，实际压缩效果尚未通过实验验证；此外，仅针对有向无环图（可通过合并强连通分量转化）处理，未讨论更一般情形下的扩展。

---

## 279. Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development

**arXiv ID:** 2609.11331 | [PDF](https://arxiv.org/pdf/2609.11331v1)

**作者:** Johannes Mäkelburg `[一作]` (Technical University of Munich), Maribel Acosta `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究如何在网络化控制系统（CPS）中评估并修复因不确定性导致的模型不一致性，提出将不一致性视为干预-响应函数并使用代理模型快速估计不一致性。

**💡 创新点**

创新点在于：①将不一致性问题转化为干预响应建模；②利用Saltelli采样与多精度蒙特卡洛结合生成训练数据；③训练直接基于不确定性几何的代理模型，显著加速评估；④通过梯度优化实现最小化不确定性干预的可解释性修复。

**🔧 技术方法**

使用了受限锥体（constrained zonotope）表示不确定性集合，Saltelli敏感性采样，多精度Monte Carlo估计，神经网络或其他结构化代理模型进行不一致性预测，以及梯度下降求解最小干预。

**📊 数据集**

实验基于48个场景、10个CPS领域（包括医疗设备、输电网、工业控制等），构建的仿真数据集。

**📈 对比分析**

与传统Monte Carlo一致性评估相比，代理模型将评估时间从毫秒级降至微秒级，能够在固定计算预算下完成数倍到数十倍的响应面评估；实验显示代理预测与Monte Carlo估计在数值和几何形状上高度一致。

**⚠️ 局限性**

局限性包括：①代理模型对极端不确定性边界的准确性可能下降；②依赖受限锥体和有向无环依赖图的假设，可能不适用于高度耦合或非线性映射；③梯度修复在多模态不一致性场景中可能陷入局部最优；④实验覆盖的场景和领域有限，未评估在更大规模或更复杂CPS中的泛化性。

---

## 280. Mr.LHDR: A Benchmark for Multimodal Real-World Long-Horizon Deep Research Agents

**arXiv ID:** 2609.11318 | [PDF](https://arxiv.org/pdf/2609.11318v1)

**作者:** Minghao Guo `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiaojun Chang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个长时序、多模态深度研究基准（Multimodal real‑world Long‑Horizon Deep Research，简称M‑LHDR），评估代理在开放网页上完成跨模态、跨来源、依赖关系连贯的长链推理任务的能力；

**💡 创新点**

创新点在于构造隐藏的Node‑Relation图和不可约的依赖检查表，使用依赖感知评分指标（DACS）区分最终答案与中间结论的依赖一致性，并将多模态证据（图片、地图、PDF、视频等）嵌入长链推理；

**🔧 技术方法**

采用多模态大型语言模型（如Qwen3‑VL‑235B）、专用深度研究系统、工具增强与无工具VLM以及框架化智能体等多种技术，并使用Qwen3‑VL‑235B作为LLM Judge对答案和检查表进行自动评判；

**📊 数据集**

使用包含102道题、覆盖八类（媒体、人物、机构、地理、社会、学术、技术、体育）的数据集，每题包含自然语言问题、答案、非文本证据、可验证的检查表，共计1231个中间结论，平均依赖深度10.4；

**📈 对比分析**

与多种系统对比显示，最高OA约34.3%（GPT‑5.5），但严格准确率SA仅34.3%，DACS最高约70.3%；工具无搜索的Gemini模型在DACS上表现突出，但OA、SA仍低；整体表明当前模型在保持依赖一致性方面仍显不足；

**⚠️ 局限性**

局限性包括：评测仅针对单一框架实现；缺乏对检索路径和来源真实性的直接验证；公开问题依赖实时网页可能导致污染和对第三方站点的负载；数据集规模有限，置信区间宽，难以得出稳健的系统排名。

---

## 281. Your Model Already Knows Don't Teach It, Learn to Ask It: Soft Prompting for Few-Shot Adaptation of Vision-Language Models

**arXiv ID:** 2609.11310 | [PDF](https://arxiv.org/pdf/2609.11310v1)

**作者:** Gautam Rajendrakumar Gare `[一作]` (Carnegie Mellon University), Deva Ramanan `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在少样本（10张图）场景下，通过在视觉‑语言模型(VLM)冻结后在跨模态边界插入少量连续软提示词，对不同专业域（航拍、工业、医学等）进行适配；

**💡 创新点**

发现两项关键设计：1）将软提示词放在视觉与文本嵌入之间（cross‑modal boundary）而非传统前缀；2）用无语义的空格标记初始化，以保持起始不变，从而显著提升性能；

**🔧 技术方法**

采用软提示（soft prompting）技术，利用梯度下降优化少量连续词嵌入；对比离散提示优化、低秩权重微调等方法；

**📊 数据集**

主要在Roboflow‑100‑VL的20个子域（含航空、工业、医学等）进行评测，也在LVIS Rare 50、Gemma‑4‑12B、π₀.₅机器人策略等数据集上验证；

**📈 对比分析**

与最佳低秩微调(r=64)相比，软提示在10‑shot下仅训练约7K参数即可达到相同14.2 mAP；与离散提示搜索（GEPA、DetPO）相比提升1.7–3.4 mAP；在权重微调方案中，软提示零忘却，而微调方案在VQA、RefCOCO上分别损失35–56%性能；

**⚠️ 局限性**

局限包括：单跑评估、种子波动大（20–28%），低样本下若VLM缺乏相关知识仍无法提升（医学域）；跨模型搜索结果不稳定；训练内存仍受冻结模型影响；适用范围局限于所测试的VLM家族与少样本场景。

---

## 282. Geometric Analysis of Doppler-Based Navigation with Low Earth Orbit Satellites

**arXiv ID:** 2609.11296 | [PDF](https://arxiv.org/pdf/2609.11296v1)

**作者:** Carlos Caravaca Gallego `[一作]` (Technion Israel Institute of Technology), Hector Rotstein `[通讯]` (Rafael Advanced Defense Systems)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并推导了基于低地球轨道卫星的 Doppler 导航八状态雅可比矩阵的闭式几何参数化，明确了时钟偏差敏感度与仰角、轨道高度、视线–速度夹角的关系，并给出了对应的 GDOP 膨胀公式。

**💡 创新点**

核心创新包括：① 计算时钟偏差列的精确闭式表达及其正定性与上下界；② 利用 Schur 补得到 GDOP 膨胀与列向量共线系数的精确关系；③ 证明高度多样性是降低共线性的唯一有效手段并给出代价‑收益分析。

**🔧 技术方法**

主要技术手段为：几何分析、闭式雅可比推导、范数化参数化、Schur 补分解、共线系数定义与解析估计、数值优化演示。

**📊 数据集**

使用合成的 LEO 星座（包括 Starlink、OneWeb、Iridium 等多层轨道），并在 OneWeb 实际轨道数据上验证卫星选择指标。

**📈 对比分析**

通过与伪距定位的体积代理对比，证明对 Doppler GDOP 最小化应使用完整雅可比行列式指标；在示例中，采用两层轨道配置优化后 8 态 GDOP 从 3.51 降至 2.25，位置误差可达 1–5 m。

**⚠️ 局限性**

局限性包括：假设圆形轨道、静止地面接收机；忽略地球自转和轨道偏心率的影响；仅考虑单时刻雅可比，未考虑多时钟协同；高度多样性受实际星座高度范围限制，提升空间有限。

---

## 283. Bio-inspired Learning and Decision-Making with Probabilistic In-Memory Computing Hardware: Part 2

**arXiv ID:** 2609.11288 | [PDF](https://arxiv.org/pdf/2609.11288v1)

**作者:** Thomas Dalgaty `[一作]` (CEA-List), Eric Flamand `[通讯]` (Independent Researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文实现并评估了能量基模型（EBM）在概率AIMC处理器上的运行，将其与GPU和TPU进行对比。

**💡 创新点**

创新点在于将EBM映射到AIMC架构，利用模拟TLM证明其可比传统GPU/TPU快超过三位数，从而揭示HBM接口是导致能量基模型不可扩展的根本瓶颈。

**🔧 技术方法**

技术手段包括使用GVSOC框架构建AIMC处理器的交易级模型，嵌入RISC‑V Spatz核、DMA、FlooNoC网络、128通道DAC/ADC外围电路以及C++跨编译到RISC‑V的代码实现。

**📊 数据集**

实验采用EBM自生成的10 GB样本数据进行推断与训练，不使用公开数据集。

**📈 对比分析**

在相同16位Bfloat数据格式下，比较Nvidia Tesla T4 GPU、Google TPU v5e以及AIMC两大核心（样本生成与权重更新）。AIMC分别耗时50 µs和900 µs，而GPU耗时870 ms/145 ms，TPU耗时2800 ms/3600 ms，性能提升至少三位数。

**⚠️ 局限性**

局限性包括：实验仅为TLM仿真，缺乏实际硬件验证；对AIMC噪声与误差的进一步控制和能量消耗未做量化；以及在更大规模模型和多任务场景下的可扩展性尚待验证。

---

## 284. Bio-inspired Learning and Decision-Making with Probabilistic In-Memory Computing Hardware: Part 1

**arXiv ID:** 2609.11281 | [PDF](https://arxiv.org/pdf/2609.11281v1)

**作者:** Thomas Dalgaty `[一作]` (CEA-List), Tommaso Salvatori `[通讯]` (VERSES AI Research Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出了利用神经与突触噪声进行采样推理与学习的框架，并将该框架映射到模拟内存计算硬件上，以实现高效、能耗低的贝叶斯推理；

**💡 创新点**

创新点在于将预测编码能量与马尔科夫链蒙特卡罗采样相结合，利用生物系统中的噪声作为计算资源，并将硬件噪声视为实现概率推理的优势；

**🔧 技术方法**

使用的技术包括拉普拉斯动力学（Langevin dynamics）、随机梯度拉普拉斯动力学、预测编码网络、模拟内存计算（in‑memory computing）以及MCMC采样方法；

**📊 数据集**

文中未给出具体的数据集，主要聚焦于理论框架与算法设计；

**📈 对比分析**

目前尚无实验数据进行比较；文中仅提到传统预测编码模型在小网络上可在约15次迭代内收敛，而加入拉普拉斯动力学的模型需要超过200次迭代，暗示在常规架构上计算量显著增加；

**⚠️ 局限性**

主要局限包括：在传统计算平台上MCMC采样耗时长，需要大量并行链；生物实现时转置操作的可行性仍待验证；缺乏实验证明该框架在真实数据集上的性能与可扩展性；

---

## 285. AI-Powered Flare Combustion Efficiency Estimation

**arXiv ID:** 2609.11262 | [PDF](https://arxiv.org/pdf/2609.11262v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 286. Predicting Train Delays in Finland Using Machine Learning and Weather Data

**arXiv ID:** 2609.11277 | [PDF](https://arxiv.org/pdf/2609.11277v1)

**作者:** Vinicius Pozzobon Borin `[一作]` (University of Oulu), Nurul Huda Mahmood `[通讯]` (University of Oulu)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

使用机器学习方法（XGBoost）预测芬兰列车延误，结合运营记录与气象传感器数据，评估不同特征配置的模型性能。

**💡 创新点**

提出基于领域知识的天气分类特征（如暴风雪、严重降雪等），证明这些离散化的天气指标比原始连续气象观测更能提升预测准确性，并能实现边缘设备的带宽友好部署。

**🔧 技术方法**

采用XGBoost算法，配合时间序列交叉验证、特征工程和层级天气分类逻辑，进行模型训练与调参。

**📊 数据集**

使用芬兰整合列车-天气（FI-TW）数据集，融合Digitraffic铁路运营记录与芬兰气象局FMI的200余站点传感器观测。

**📈 对比分析**

对比三种特征设置（完整天气、即时天气、天气分类），在奥卢中心站的101,146条观测上，天气分类方案取得R²≈0.78、RMSE≈8.5分钟、MAE≈3.7分钟，较其它方案提升约11% R²和10%误差。

**⚠️ 局限性**

局限性包括：仅在奥卢站评估；仅测试XGBoost而未尝试深度学习；采用最近站点匹配忽略微气候差异；假设无线回传可靠，未考虑通信失败。

---

## 287. Order-Aware 2.5D Multiple Instance Learning for Preoperative MRI-Based Perineural Invasion Risk Assessment in Intrahepatic Cholangiocarcinoma

**arXiv ID:** 2609.11271 | [PDF](https://arxiv.org/pdf/2609.11271v1)

**作者:** Hyunsu Go `[一作]` (Seoul National University), Nam-Joon Kim `[通讯]` (Seoul National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用弱监督的顺序感知2.5D多实例学习框架预测肝内胆管癌患者术前T2加权MRI中的神经侵犯风险

**💡 创新点**

将肿瘤中心MRI裁剪表示为有序的2.5D切片组，结合集合注意力与序列注意力实现对轴向顺序的利用，显著提升预测性能

**🔧 技术方法**

采用共享ResNet-18作为切片编码器，双向GRU序列读取器，集合注意力和序列注意力聚合，最后通过全连接层输出概率

**📊 数据集**

单中心183例患者的术前T2加权MRI数据，其中70例为PNI阳性，113例为阴性

**📈 对比分析**

与多种体积模型（ResNet‑18、DenseNet‑121、ViT、Swin）和MIL基线（ABMIL、CLAM‑SB、DSMIL、TransMIL）进行五折标签分层交叉验证，取得平均AUROC 0.770±0.077，明显优于最强基线CLAM‑SB（AUROC 0.705±0.071）

**⚠️ 局限性**

数据量有限且为单中心，缺乏跨机构、不同扫描仪和协议的外部验证，模型对裁剪定位的敏感性尚未完全评估

---

## 288. Beyond Noise Steering: Dual-Latent Space Reinforcement Learning for Generative Robot Policy

**arXiv ID:** 2609.11270 | [PDF](https://arxiv.org/pdf/2609.11270v1)

**作者:** Pengfei Zhang `[一作]` (Shanghai University), Xianchao Xiu `[通讯]` (Shanghai University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

为冻结的生成式机器人策略开发一种在线自适应框架 DLSRL，能够通过两种潜在变量实现对生成过程的全局与局部控制。

**💡 创新点**

创新点在于：① 引入动作表示潜在变量并映射为适配器特征；② 将适配器特征残差注入 Transformer 隐藏层，直接调节内部表示；③ 通过动作空间与潜在空间两级 critic 与价值蒸馏实现高效 RL 更新。

**🔧 技术方法**

技术包括：生成式策略（扩散或流匹配）、双潜在空间 Actor、轻量级适配器映射、残差注入、双 critic 训练（动作空间 Q 与潜在空间 Q）、熵正则化、价值蒸馏与 TD 学习。

**📊 数据集**

使用 RoboMimic（Lift、Can、Square）和 LIBERO（Stove‑On、CreamCheese‑to‑Tray、Bowl‑Drawer‑to‑Plate、WineBottle‑to‑Rack、Plate‑to‑StoveFront、Bowl‑to‑TopDrawer）两套仿真数据集，分别配合扩散和流匹配基策略。

**📈 对比分析**

与 Base、JSRL、DPPO、DSRL 等基线对比，DLSRL 在同等交互预算下显著加速成功率提升，早期与中期性能大幅领先，并在多数任务上达到或超过基线的最终成功率；同时平均 episode 长度更短，表明执行效率提升。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证；对注入强度 λ_inj 的敏感性需进一步调优；缺乏对真实机器人部署的评估；仅针对冻结基策略，未探究与全参数微调的混合方案。

---

## 289. SoulAuth: An Actor-native Identity Architecture and Rust Reference Implementation for Humans and Long-lived AI Actors

**arXiv ID:** 2609.11258 | [PDF](https://arxiv.org/pdf/2609.11258v1)

**作者:** Kun Yuan `[一作]` (TRANTOR LABS), Magnus Hu `[通讯]` (TRANTOR LABS)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Actor‑native Identity 架构并实现开源 Rust 引用实现 SoulAuth，解决 AIActor 与人类共享身份基础设施时的持续身份识别问题。

**💡 创新点**

核心创新是将“持久主体”统一定义为 ActorIdentity，明确与 Account、Credential、Client、AuthSession、Runtime 等对象的生命周期分离，保证身份连续性与历史归因不被外围对象重写。

**🔧 技术方法**

采用哲学工程方法（概念化、结构化、验证），实现 Rust 代码、OpenAPI 接口、架构一致性检查（Architecture Conformance）及 CI 自动化，支持身份、认证、会话、投影、审计、持久化等职责划分。

**📊 数据集**

未使用传统数据集；评估基于固定源码版本（v0.1.0）及其公开 CI 记录，包含 188 条单元测试、61 条一致性检查和 355 条集成断言。

**📈 对比分析**

与传统 Account/Credential‑centric 系统对比，SoulAuth 在身份主体连续性、角色分离、历史归因等方面实现了多项核心不变量；测试显示实现通过 188/61/355 条检查，但仍有 9 条未满足，整体评为“部分一致性”。

**⚠️ 局限性**

限制在于：Credential 统一建模与历史归因仍未完全实现，部分关键不变量（如 I5、I9、I10）仅在部分路径得到支持，且当前实现未覆盖所有 AIActor 认证路径与外部 IdP 交互细节。

---

## 290. Node-Shift-Encoding Genetic Algorithm with fuzzy-enhanced reference tour to solve the bi-objective service-oriented TSP

**arXiv ID:** 2609.11257 | [PDF](https://arxiv.org/pdf/2609.11257v1)

**作者:** Souad Abdoune `[一作]`, Menouar Boulif `[通讯]` (University of M'hamed Bougara)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了服务导向的双目标旅行商问题，并提出了一种基于节点移位编码（NSE）的遗传算法，辅以模糊逻辑动态更新参考路径；

**💡 创新点**

创新点在于将MTZ模型改造成无起点版本，并结合模糊推理控制器对参考路径进行自适应更新，解决传统NSE的停滞与探索受限问题；

**🔧 技术方法**

采用的技术包括Node-Shift Encoding遗传算法、Miller–Tucker–Zemlin线性化、模糊逻辑推理器、GLPK求解器以及对TSPLIB benchmark的实验评估；

**📊 数据集**

实验数据集为13个TSPLIB实例（51到200城市），并对每个城市随机生成均匀分布的访问优先级；

**📈 对比分析**

通过与传统NSE和距离最优TSP进行Wilcoxon检验比较，FL‑NSE在绝大多数实例上平均fitness下降约2‑7%，同时距离与客户不满均得到提升，虽然平均计算时间略高，但在大规模实例上更快；

**⚠️ 局限性**

主要限制在于使用加权求和的单目标化方法，无法完整覆盖非凸Pareto前沿；优先级假设为均匀随机，缺少真实分布；且实验规模仅至200城市，未探讨更大规模问题。

---

## 291. The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation

**arXiv ID:** 2609.11326 | [PDF](https://arxiv.org/pdf/2609.11326v1)

**作者:** Jose Pascual Gumbau Mezquita `[一作]` `[通讯]` (University Jaume I de Castelló), Jose Pascual Gumbau Mezquita (University Jaume I de Castelló)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了自我修改程序在“语义提升”操作下的安全性持久性问题，证明了未可判定性类在该操作下封闭，并证明无穷迭代会将其提升至Π^0_2完备级别；同时证明监督层级递归无法终止。

**💡 创新点**

提出语义提升算子作为将静态语义属性转化为动态持久性属性的通用构造，并揭示该算子在强度足够的自我修改（强制性语义破坏）下仍保持未可判定性；展示了两种独立的不可判定来源（强制性与无穷迭代），并给出了闭包定理与监督递归不终止的统一视角。

**🔧 技术方法**

主要使用递归理论技术：Rice定理、克利尼递归定理、算子/可计算变换的构造、s-m-n 定理、算术层次、图灵可计算性与可判定性分析；以及有效拓扑的潜在范畴解释。

**📊 数据集**

无实验数据集，所有结果均为形式化证明与理论分析。

**📈 对比分析**

与传统的可判定性分析（Rice定理）对比，语义提升超越了静态可判定性的限制；通过构造强制性变换和迭代示例，证明了即使在更强的监督层级下也无法获得完整的持久性验证。性能方面为理论上不可计算，不涉及运行时性能评估。

**⚠️ 局限性**

主要局限：仅提供理论证明，未给出具体算法或工具实现；对实际自我修改系统的模型假设较抽象；对监督递归层级的实现细节与资源限制未做深入实验验证。

---

## 292. MultiHuSE: A Multimodal Dataset for Humour Styles and Emotions

**arXiv ID:** 2609.11322 | [PDF](https://arxiv.org/pdf/2609.11322v1)

**作者:** Mary Ogbuka Kenneth `[一作]` (Imperial College London), Abbas Edalat `[通讯]` (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 MultiHuSE 多模态数据集，并在其上开展了幽默风格与情感识别的基线实验。

**💡 创新点**

首次提供英文多模态幽默风格数据集，包含心理学定义的四种幽默风格及情感标签，并收集多位演员对同一文本的多重表现，方便研究表达多样性。

**🔧 技术方法**

使用 BERT、Dasheng、MC3-18 三大预训练编码器分别提取文本、音频、视频特征；采用 XGBoost 与跨模态注意力 Transformer 进行单模态与多模态融合。

**📊 数据集**

使用 MultiHuSE 自身（2,407 条 HD 视频、50 名演员、1,463 文本实例）进行实验；无其他公开数据集参与。

**📈 对比分析**

通过 80:20 分层拆分、5 倍交叉验证，评估准确率、精确率、召回率、F1 分数。单模态文本 77.4% 准确率，融合模型提升至 80.1%（指数加权），跨模态注意力 79.7%。

**⚠️ 局限性**

情感注释的互评一致性仅为 Cohen's κ=0.325；样本受限于伦敦地区，缺乏跨文化多样性；情感标注在重演视频中不完整，仅覆盖 72.7%。

---

## 293. A global mobile network coverage raster product at 1km resolution, 1999--2030

**arXiv ID:** 2609.11320 | [PDF](https://arxiv.org/pdf/2609.11320v1)

**作者:** Till Koebe `[一作]` (Saarland Informatics Campus), Ridhi Kashyap `[通讯]` (Oxford University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了一套全球 1 km 分辨率的 1999‑2030 年移动网络（2G/3G/4G）覆盖概率年历，结合可靠运营商标签与三种独立建模轨道，生成公开且跨国时序一致的覆盖记录。

**💡 创新点**

创新点在于：①通过可靠性筛选的 MCE 标签集确保训练数据质量；②采用 Gradient‑Boosted Decision Tree、技术经济模拟（CMA‑ES）与 pix2pix U‑Net 三种独立模型，并按每个国家/技术的验证精度加权组合；③为每个像素提供 90 % conformal 不确定区间，提升结果可信度。

**🔧 技术方法**

使用的技术包括 LightGBM（带单调性约束与后置 Isotonic 校准）、CMA‑ES 结构化网络部署模拟、pix2pix U‑Net（Monte‑Carlo Dropout 估计不确定性）、蒙特卡洛 Dropout、合成校准与分层 conformal 置信区间。

**📊 数据集**

主要数据集为 2,409 个可靠的 MCE 运营商覆盖地图（1999‑2020），以及世界人口（WorldPop）、建筑面（GHSL）、夜间灯光、道路网络、世界银行 ICT 指标、SEDAC 贫困指数等全球时间序列协变量。

**📈 对比分析**

通过 5‑折空间交叉验证、时间保留验证和对 ITU 国家级覆盖比例的外部验证，模型 AUC 达 0.89‑0.94，Brier 分数低，合成估计在 ITU 指标上相关系数 >0.8，显示在不同技术与时间范围内均具高预测精度。

**⚠️ 局限性**

局限性包括：对运营商报告标签的依赖导致 5G 缺少可靠标签；未来 2025‑2030 预测仅基于协变量，未考虑冲突或灾害导致的突发覆盖损失；覆盖保持不递减仅记录最大历史覆盖，未反映技术退役与短期降级情况。

---

## 294. Mi-Ripple: Restoring Images Degraded by Iterative AI Editing

**arXiv ID:** 2609.11317 | [PDF](https://arxiv.org/pdf/2609.11317v1)

**作者:** Jiayin Chen `[一作]` (Miyang Technology Co Ltd), Muting Wang `[通讯]` (Miyang Technology Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种诊断导向的 Mi-Ripple 工作流程，用来恢复因迭代 AI 编辑产生的数字波纹（网格状与颗粒纹理）图像。

**💡 创新点**

创新点在于先将周期性网格与内容混杂的颗粒纹理分离，再分别采用选择性频谱抑制、结构感知平滑以及参考清洗重生成像，既降低了失真，又保护了图像结构。

**🔧 技术方法**

主要技术包括基于局部对数幅值的频域诊断探针、峰值切除与径向软裁剪、方向性掩膜与结构感知平滑、以及清洗后重生成像的参考引导重建。

**📊 数据集**

使用了 GPT‑image‑2.5 及 GPT‑image‑2 的多场景迭代编辑链、公开的高质量照片、Banana100 子集以及若干网页参考图像，共计超过 110 张编辑输出。

**📈 对比分析**

通过残差标准差、尺度索引覆盖率和高频保留等定量指标评估；实验显示全图残差 SD 在 0.08–0.44 范围内，参考清洗后残渣密度降低 45%，尺度索引从 25.3% 降至 11.1% 等，验证了方法的有效性。

**⚠️ 局限性**

局限在于阈值需要更广泛的校准，单个样本的重生比较无法体现总体平均效果，缺乏跨渠道独立评估与完整的质量基准。

---

## 295. Automatic Lyric Transcription for Greek Songs: Scaling and Task Composition Effects in Whisper Adaptation

**arXiv ID:** 2609.11302 | [PDF](https://arxiv.org/pdf/2609.11302v1)

**作者:** Maria Frangiadaki `[一作]` (Institute for Language and Speech Processing, Athena R.C.), Vassilis Katsouros `[通讯]` (Institute for Language and Speech Processing, Athena R.C.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了第一套希腊语歌唱语音转写基准，完成了 GAD-ALT 数据集的标注与分割，并在 Whisper 预训练模型上进行了针对性微调，探索了模型规模、任务混合与两阶段适配对歌唱文字识别的影响。

**💡 创新点**

创新点包括：①首次在低资源语言希腊语上创建可复现的歌唱转写基准；②提出任务纯化批处理与语言感知预处理以稳定训练；③构建了针对希腊语的错误分类体系；④系统评估模型规模与多任务/两阶段适配策略的效果。

**🔧 技术方法**

采用的技术主要有 Whisper 预训练多语种模型、混合 Transformer Demucs 语音源分离、CTC 强制对齐、gpt‑4o‑mini 进行段级英文翻译、以及多任务学习与两阶段微调（先语音后歌唱）。

**📊 数据集**

使用的数据集包括：GAD‑ALT（从 GAD 迁移而来并标注的希腊语歌唱段落）、Greek Common Voice（用于第一阶段微调）、以及公开的多语言歌唱数据集如 DALI、DAMP‑Sing 作对比参考。

**📈 对比分析**

通过将模型按规模（Small、Medium、Large‑v3）进行零射击、单任务转写、双任务转写+翻译以及两阶段微调的对比实验，结果表明：零射击 WER 最高达 92.3%；在 Large‑v3 上采用两阶段微调后 WER 降至 27.2%，成为目前希腊语歌唱转写的最佳基准；多任务学习对小模型具有正则化作用。

**⚠️ 局限性**

局限性在于：①残留的旋律变异与音素扭曲仍导致高错误率；②两阶段微调对语音与歌唱域的差距补偿不足；③人工增强与混音方法反而影响性能；④数据规模仍有限，难以充分覆盖希腊语形态与节奏多样性。

---

## 296. Few-Shot Learning for Network Intrusion Detection: Methods, Datasets, and Performance

**arXiv ID:** 2609.11275 | [PDF](https://arxiv.org/pdf/2609.11275v1)

**作者:** Arne Roszeitis `[一作]` (Leipzig University), Erik Buchmann `[通讯]` (Leipzig University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了2022-2026年间针对网络入侵检测的少样本学习研究

**💡 创新点**

系统性梳理了现有技术与评估方法，并提出统一评估框架与标准

**🔧 技术方法**

采用Meta-learning、CNN、Graph Neural Network、Auto-Encoder等多种学习技术

**📊 数据集**

主要使用CIC-IDS2017与CSE-CIC-IDS2018等数据集，亦涉及USTC-TFC2016、UNSW-NB15等

**📈 对比分析**

通过对21项研究的技术、数据集、k‑shot设置等进行对比，发现CNN+Meta-learning被广泛使用但表现不一，Graph NN与Auto-Encoder在多项指标上表现最佳，F1得分区间从0.70到0.99

**⚠️ 局限性**

评估设置缺乏统一性、缺少完整参数与代码、不同研究使用的指标与实验条件不一致，导致难以直接比较与复现

---

## 297. Magenta: Closing the Loop Between Mathematical Reasoning and Lean Verification

**arXiv ID:** 2609.11319 | [PDF](https://arxiv.org/pdf/2609.11319v1)

**作者:** Joshua Ong Jun Leang `[一作]` (Institute of Foundation Models), Eleonora Giunchiglia `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练的 agentic pipeline，能够把自然语言数学题先生成推理链和答案，然后自动转化为 Lean4 形式化命题并生成可验证的证明。

**💡 创新点**

核心创新在于引入了 statement judge 和 error judge 两个判别器：statement judge 检查生成的 Lean 命题是否与原题语义对齐，error judge 则根据 Lean 证明失败的诊断把错误归因于数学推理还是 Lean 实现，从而实现闭环自动校正。

**🔧 技术方法**

技术实现包括：利用大规模语言模型（如 K2‑Horizon‑7B/375B、Qwen3.8‑27B、GPT‑5.6‑Sol）分别担任推理器、形式化器和证明器；使用 Lean4 及 SafeVerify 做确定性验证；通过语言模型实现判别器和重试策略。

**📊 数据集**

实验数据集包括 AIME 2025、AIME 2026、HMMT 2026（二月）以及 IMO 2026 的六道题目，全部为自然语言题目，没有预先给定形式化命题。

**📈 对比分析**

与 Claude Opus 5、Gemini 3.7 Flash、Kimi K3 等基线相比，本方法在所有基准上实现 100% 准确率；在 IMO 2026 上仅用最小的 K2‑Horizon‑7B 就完成全部六道题，显示出显著的性能提升。

**⚠️ 局限性**

主要限制是自动形式化差距（autoformalisation gap）：Lean 证明只保证对生成的 Lean 命题成立，而该命题可能与原始自然语言问题不完全一致，导致仍需要改进形式化质量。

---

## 298. Routing by Reasoning Need: Trajectory-Aware Decoding Control for Diffusion Vision-Language Models

**arXiv ID:** 2609.11315 | [PDF](https://arxiv.org/pdf/2609.11315v1)

**作者:** Yixiang Liu `[一作]` (Southern University of Science and Technology), Xiaoying Tang `[通讯]` (Southern University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种训练无关的推理时控制器，利用扩散VLM的中间回答轨迹信号动态决定是提前提交答案、保持固定预算输出，还是执行支持推理的解码，以解决推理预算不匹配问题。

**💡 创新点**

创新点在于将解码过程视为轨迹感知的路由问题，而非统一的生成长度；提出无监督的答案闭合、视觉格式闭合和表示修订压力三种轨迹信号，并基于它们实现样本级的早期提交、保留或推理支持三路控制。

**🔧 技术方法**

采用扩散式视觉语言模型LLaDA‑V，提取答案闭合、视觉格式闭合和表示修订压力等中间状态信号；实现路由策略、早期提交、固定预算保留与推理支持解码配置等技术，无需模型微调或额外训练。

**📊 数据集**

在多种多模态问答基准上进行评估，包含视觉诊断类的MME、专家级的MMMU与MMStar，以及需要链式推理的ScienceQA‑IMG、A‑OKVQA和MME‑CoT等数据集。

**📈 对比分析**

与固定长度解码（Len2、Len32、Len64、Len128）、视觉/对比解码、适应性预算控制等方法对比；在MME上取得79.96%（比Len128高6.55个百分点），ScienceQA‑IMG上88.60%（比Len128高13.29个百分点），MME‑CoT上52.42%（比Len128高2.85个百分点），表明路由控制在不同推理需求下显著提升鲁棒性。

**⚠️ 局限性**

局限性包括仅在LLaDA‑V上验证，未证明对其他扩散VLM的通用性；轨迹信号为无监督代理，无法直接评估答案正确性；阈值设定与校准依赖特定基准；CoT诊断仅为辅助指标，未能证实推理可信度；路由控制虽提升鲁棒性但并非加速方法，部分路由仍保留长预算。

---

## 299. Uncertainty DMD: Restoring Diversity in Few-Step Autoregressive Video Distillation

**arXiv ID:** 2609.11265 | [PDF](https://arxiv.org/pdf/2609.11265v1)

**作者:** Zixuan Duan `[一作]` (Nanjing University), Xuelong Li `[通讯]` (China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在少步自回归视频扩散蒸馏（DMD）过程中注入不确定性，恢复视频生成的多样性和运动动态。

**💡 创新点**

提出两种结构化的不确定性注入方式：对首块的时间步进行扰动以提升噪声依赖多样性，以及对自回归缓存进行随机写入以保持后续块的多样性。

**🔧 技术方法**

基于DMD的自回归视频生成框架（Causal Forcing 与 Self‑Forcing），配合时间步扰动、缓存随机写入和训练时的学习曲线。

**📊 数据集**

在VidProM文本提示集上进行无监督蒸馏，使用VBench数据集进行评估，采用TE/VENDI六种多样性指标以及VBench质量指标。

**📈 对比分析**

与标准DMD、DMD+GAN、DMD+额外噪声等基线对比，实验显示Uncertainty DMD在TE、VENDI、Dynamic Degree等多样性和运动指标上均有显著提升，质量指标保持与原始DMD相当。

**⚠️ 局限性**

方法仍缺乏对未来运动的显式规划，复杂动态场景下可能出现不自然的运动表现。

---

## 300. SEAR: Segment-Evidence-Aware Routing for Weak-to-Strong Multilingual Speech MCQ

**arXiv ID:** 2609.11355 | [PDF](https://arxiv.org/pdf/2609.11355v1)

**作者:** Huy Hoang Le `[一作]` (CAKE by VPBank), Minh Tri Dao `[通讯]` (CAKE by VPBank)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了基于事件保留的多语言对话音频多选题生成与训练系统。

**💡 创新点**

创新点包括 LLM 提取事件跨度并扩展边界裁剪音频，双分支语义/声学问答合成与严格验证，以及弱-强路由结合 GSPO 的强化学习。

**🔧 技术方法**

使用 Qwen3‑Omni‑30B‑A3B 作为主干，LoRA 适配；Qwen3.6‑27B 与 Gemini 3.1 Flash‑Lite 生成问题；GSPO、Dr.GRPO、TIS 等技术实现稳定化训练。

**📊 数据集**

利用 MLC‑SLM 任务 2 的多语言两人对话数据，生成 359,825 条经过验证的段级 Audio MCQ，覆盖 21 种语言/口音。

**📈 对比分析**

在官方评测集上与基线对比，最终准确率 90.92%，在排行榜中排名第二。

**⚠️ 局限性**

限制在于 LLM 基础的事件与质量判断可能产生相关误差，且需在策略改进时重新评估可训练性标签。

---

## 301. EConv-TasNet: Efficient Conv-TasNet for Effective Speech Separation

**arXiv ID:** 2609.11342 | [PDF](https://arxiv.org/pdf/2609.11342v1)

**作者:** Pei-Chun Chang `[一作]` (Novatek Microelectronics Corporation), Chuan-Yi Liu `[通讯]` (Novatek Microelectronics Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了高效的eConv‑TasNet用于语音分离，并在Conv‑TasNet基础上进行改进。

**💡 创新点**

创新点是引入组级早期分裂（GES）和多组特征聚合（MGFA）两大模块，实现更早、更具辨别性的说话人表示，并降低计算冗余。

**🔧 技术方法**

技术包括全卷积架构、GLU分裂、指数加权移动平均聚合、PReLU+Sigmoid掩码估计以及动态混合数据增强。

**📊 数据集**

使用了WSJ0‑2mix、WHAM!和Libri2Mix三大基准数据集进行评测。

**📈 对比分析**

通过与原Conv‑TasNet及多种SOTA方法对比，eConv‑TasNet在WSJ0‑2mix上实现SI‑SNRi 18.4 dB，参数3.8 M，RTF 0.13，显著提升性能且保持低延迟与小模型。

**⚠️ 局限性**

局限性在于对极难混合（SI‑SNRi<10 dB）的提升有限，且仍基于TCN，对长距离依赖建模不如Transformer，需进一步压缩以满足极低功耗设备需求。

---

## 302. Exploring Diffusion Transformers for Cross-Modal Augmentation in Multimodal Brain State Decoding

**arXiv ID:** 2609.11341 | [PDF](https://arxiv.org/pdf/2609.11341v1)

**作者:** Ziwei Wang `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种双向跨模态扩散Transformer（CoMA-DiT），利用配对生理信号相互作为生成监督，进行潜在空间的跨模态数据增强；

**💡 创新点**

创新点在于将配对模态视为相互生成条件，而非仅用于融合；通过跨模态注意力、可靠性门控的残差注入，实现保留模态特征同时注入互补信息；

**🔧 技术方法**

使用跨模态扩散Transformer、速度预测、跨模态一致性损失、残差保留正则化等技术；

**📊 数据集**

在AVGC（听觉注意解码）和DEAP（情绪识别）两个多模态脑状态解码数据集上进行实验；

**📈 对比分析**

与20个代表性基线（传统分类器、深度模型、常规数据增强）对比，CoMA-DiT在两任务上分别提升了4.28%/6.70%（ACC/F1）和约2.3%/2.4%（ACC/F1），性能稳健、显著优于所有比较方法；

**⚠️ 局限性**

局限包括仅针对两模态（EEG-EOG）实验，未评估缺失模态场景，且对更大规模数据集的可扩展性和实时部署性能未作深入分析。

---

## 303. Modular Kinematic Reduction of Closed-Chain Mechanisms Using Path Assembly and Defect Homotopy

**arXiv ID:** 2609.11338 | [PDF](https://arxiv.org/pdf/2609.11338v1)

**作者:** Mohammad Dastranj `[一作]` (Tampere University), Jouni Mattila `[通讯]` (Tampere University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于路径组装的闭环差分映射（PACDM）框架，结合缺陷同伦实现闭链机械臂的模块化运动学建模和主动–被动坐标映射。

**💡 创新点**

创新点在于：①通过路径组装直接构造闭环残差并在 SE(3) 对数坐标下求导，避免手工推导闭环方程；②使用缺陷同伦在初始估计上快速获取物理解，且分离约束选择与分支获取，提升模块化处理的通用性与鲁棒性。

**🔧 技术方法**

使用了 Lie 群对数映射、右-对角化导数、逆左 Jacobian、秩揭示分解、缺陷同伦、预测‑修正迭代等数值技术。

**📊 数据集**

以 7 自由度重载机器人 HIAB-046 为实验对象，使用 Simscape Multibody 模型作为数值参考。

**📈 对比分析**

与 Simscape Multibody 进行数值比较，闭环误差 RMSE 小于 1×10⁻⁹，最大误差小于 3×10⁻⁹；预测‑修正方法比每步缺陷同伦快约 45 倍，显著提升计算效率。

**⚠️ 局限性**

局限性：方法局限于单一对数分支，需保证被动坐标在约束范围内、约束秩保持不变；对不同拓扑闭环的全局收敛性和统计鲁棒性尚未系统验证。

---

## 304. Predictive Multi-Landmark OCT Tracking for Increased Motion Robustness

**arXiv ID:** 2609.11330 | [PDF](https://arxiv.org/pdf/2609.11330v1)

**作者:** Konrad Reuter `[一作]` (Hamburg University of Technology), Alexander Schlaefer `[通讯]` (Hamburg University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于OCT的多标记点预测跟踪方法，在高速度运动下实现更鲁棒的6D姿态估计。

**💡 创新点**

创新点在于利用Kabsch算法估计全局姿态变换，将单个标记点的位置信息传播至所有标记点，并在下一时间步进行预测，从而显著提高可跟踪速度并降低误差。

**🔧 技术方法**

使用MOSSE模板匹配算法、Kabsch全局姿态估计、预测更新策略以及光学相干断层扫描(OCT)硬件实现实时跟踪。

**📊 数据集**

实验数据来自猪皮样本，9个区域，每个区域采集9个体素尺寸为128×128×512的OCT图像，并在模拟环境中生成随机混合的真实体积以实现运动仿真。

**📈 对比分析**

与传统的每个标记点独立MOSSE跟踪方法相比，采用RMSE评估。结果显示在最高100 mm/s速度下，RMSE保持在1 mm以下；在9个标记点时误差略升高；使用5–7个标记点误差最低；推理时间从单个MOSSE的1.3 ms增加到约2.7 ms。

**⚠️ 局限性**

主要局限在于仅在仿真环境下验证，真实系统的表现尚未测试；时间间隔随标记点数变化未做显式建模；假设速度恒定，可能在非线性运动中失效；当标记点过多时预测误差上升。

---

## 305. AI Exposure and AI Resilience: A Two-Dimensional Assessment Framework for Software and Software-Based Business Model

**arXiv ID:** 2609.11321 | [PDF](https://arxiv.org/pdf/2609.11321v1)

**作者:** Paul Darius Mandl `[一作]` (Findustrial GmbH), Martin Häusl `[通讯]` (Munich University of Applied Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套名为AI-ER的双维度评估框架，用来分别衡量企业在人工智能（AI）面前的暴露程度（Exposure）与对AI带来压力的抵御与适应能力（Resilience），并给出了具体的指标、评分逻辑、证据质量与置信度评估方法。

**💡 创新点**

创新点在于：①将AI影响拆分为暴露与韧性两个互不相互抵消的维度；②基于AI技术能力、经济影响机制和组织适应性研究，构建了简洁但覆盖核心影响的指标集合；③提出了非补偿式聚合规则、阈值激活指示器和模量器，确保关键弱点不被其他指标掩盖；④引入了证据质量与多评估者一致性相结合的置信度模型，让结果与其可靠性可分离解释。

**🔧 技术方法**

使用的技术主要是：①定性研究与文献综述相结合的指标推导方法；②规则化的评分与聚合算法（最大/最小、阈值指示、模量器调节、clamp约束）；③基于观察证据的证据属性（directness、timeliness、completeness、independence、agreement）计算的权重平均；④多评估者一致性指标（A_ij）与置信度组合（C_ij）。

**📊 数据集**

本文并未使用传统意义上的数据集；评估示例采用了虚构企业A的假设评分，并通过公开信息与内部信息的两阶段（outside‑in / inside‑in）评估流程来演示方法。未来的经验验证计划涉及对多家软件型企业进行案例研究与独立评估，但尚未提供实际数据集。

**📈 对比分析**

比较方法是将企业的曝光度与韧性分别按五分制评分后，按非补偿逻辑得到两维分数，随后根据分数位置划分四个象限（Defended Niche、AI‑Ready Compounder、Rebuilding Required、Acute Threat）。论文未给出数值性能指标，而是说明了如何通过指标分数与置信度来判断评估结果的稳健性，暗示若置信度高则评估结果更可靠。

**⚠️ 局限性**

限制包括：①指标阈值、组合规则、模量器等参数未在大样本上校准；②暴露最大规则与韧性最小规则的适用性仍需经验验证；③评估高度依赖可获得的证据，公开信息丰富的公司可能被误评为易评估；④未进行实证检验，缺乏对模型预测力的纵向验证；⑤在快速演变的AI技术与监管环境下，评估结果需定期更新。

---

## 306. Occupancy-Domain Over-the-Air Computation

**arXiv ID:** 2609.11289 | [PDF](https://arxiv.org/pdf/2609.11289v1)

**作者:** Seyed Mohammad Azimi-Abarghouyi `[一作]` `[通讯]` (Chalmers University of Technology), Seyed Mohammad Azimi-Abarghouyi (Chalmers University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并分析了一种全新基于占用统计的无线多址上行计算框架（ODC/BOC），利用每个资源元件的忙/闲二进制决策来估计设备值之和，完全不依赖瞬时或统计信道状态信息。

**💡 创新点**

创新点在于：①通过占用概率的指数映射实现“占用身份”可直接解码总和；②设计了“平衡占用计算”在每设备上固定激活次数，从而消除Bernoulli激活带来的无用随机性；③推导了理论极限、最优负载、非自适应设计的下界以及针对尺度不确定、误检测和异构信噪的补偿方法。

**🔧 技术方法**

采用的技术包括：随机化激活与占用编码、最大似然估计、Fisher信息保守定理、分位数/多增益设计、均衡占用（quota）机制、误检测的独立消除模型、预处理的自归一化与二阶段探测等。

**📊 数据集**

在实验评估中使用的是合成数据：设备值独立服从$[0,1]$均匀分布，随机信道为Rayleigh衰落，信噪可调；对比基准为非协同行激活（Affine NC‑OAC）和REED等能量聚合方法。

**📈 对比分析**

与基准相比，ODC/BOC在不需要信道知识的情况下实现更高精度，尤其在资源有限时相对误差下降1.5–2.5倍；在高SNR或多资源时可进一步逼近理论下界；在误检测、信道校准误差较大时仍保持较低误差，优于能量方案。

**⚠️ 局限性**

局限包括：需要较多的资源元件来获得足够的统计精度；对大规模设备时平衡占用的收益逐渐减弱；误检测模型假设独立消除；对异构或时间相关信道的鲁棒性尚未完全验证；缺乏正式的隐私保障与多维/多频段扩展。

---

## 307. Copying Versus Randomization in Lempel-Ziv Music Synthesis

**arXiv ID:** 2609.11353 | [PDF](https://arxiv.org/pdf/2609.11353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 308. When Does Text Inform? Benchmarking Information-Theoretic Metrics for Multimodal Time-Series Forecasting

**arXiv ID:** 2609.11282 | [PDF](https://arxiv.org/pdf/2609.11282v1)

**作者:** Emma Andrews `[一作]` (National University of Singapore), Gianmarco Mengaldo `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了MMTT-Bench，一个合成的多模态时间序列与文本注释基准，旨在评估文本信息对预测的贡献。

**💡 创新点**

首次提供具有oracle真值的文本-时间序列信息理论评估基准，并对六种MI估计器在不同信息质量下的表现进行系统比较。

**🔧 技术方法**

采用KSG、MINE、InfoNCE、CCA、PID、V-Information等互信息估计方法，并结合PCA降维、文本嵌入和时间窗口设计。

**📊 数据集**

使用MMTT-Bench合成数据以及七个真实世界数据集（Time-MMD、FinTexTS 等）进行验证。

**📈 对比分析**

对六个估计器的MI估计值与下游模型的MSE关联进行回归，结果显示CCA、V-Information与PID在排序与预测效果上最稳健；神经估计器在弱信号下表现不佳。

**⚠️ 局限性**

主要局限在于合成信号过于理想化、估计器对维度和样本量敏感、PID需离散化目标、以及文本与时间的配对可信度不总能被捕捉。

---

## 309. SAMV-DUSt3R: Instance-Centric 3D Scene Decoupling from Sparse Multi-Views

**arXiv ID:** 2609.11279 | [PDF](https://arxiv.org/pdf/2609.11279v1)

**作者:** Langxu Zhao `[一作]` (Northeastern University), Tianhan Gao `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于 SAM2 的 3D 对象解耦模型 SAMV‑DUSt3R，能够在无标定、无序的 RGB 视图中通过用户提示准确提取目标物体的 3D 点云。

**💡 创新点**

创新点包括：1) 通过 Cross Flow Mask Block 将 2D 分割掩模融入 MV‑DUSt3R 的编码-解码流程，显著提升目标区域的点回归精度；2) 设计轻量级 Spatial RankGNN，利用 SfM 共视图图完成参考视图的自适应选取，提升重建稳定性；3) 引入 Masked Confidence Loss，针对目标掩模内的置信度分布进行正则化，解决“置信度洼地”问题。

**🔧 技术方法**

主要技术手段包括：SAM2 语义分割、MV‑DUSt3R 点映射重建、Spatial RankGNN（基于全局注意力的 GNN）、Cross Flow Mask Block（跨视角掩模注意力）、Masked Confidence Loss。

**📊 数据集**

使用数据集包括 Co3Dv2、WildRGBD（用于训练 Cross Flow Mask Block 与 Masked Confidence Loss）和 DTU（用于评估解耦与重建效果），以及自建的 SfM 共视图图数据集用于训练 Spatial RankGNN。

**📈 对比分析**

在 DTU 4/12/24 视图实验中，与 DUSt3R、MASt3R、Spann3R、Fast3R、MV‑DUSt3R 等基线相比，SAMV‑DUSt3R 在 ND、DAc、CD、Acc、Comp 等指标上平均提升约 11%，并在新视角合成（PSNR/SSIM/LPIPS）中保持了与基线相近或更优的表现。

**⚠️ 局限性**

局限性包括：对目标物体的遮挡或极少视角时仍可能出现重建误差；Masked Confidence Loss 在非目标区域的精细细节恢复有限；对复杂场景或高动态对象的鲁棒性尚未充分验证。

---

## 310. INDRA: A New AI Tool for Exploring Tobacco, Fossil Fuel, and Chemical Industry Archives

**arXiv ID:** 2609.11261 | [PDF](https://arxiv.org/pdf/2609.11261v1)

**作者:** Daniel Akselrad `[一作]` (Stanford University), Robert N. Proctor `[通讯]` (Stanford University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了 INDRA，一款基于 Anthropic Claude LLM 的研究平台，专门用于在封闭的证据沙箱中访问并分析来自多家机构（如 UCSF Industry Documents Library、ToxicDocs、SRITA 等）的大型行业档案，从而支持跨行业的历史与政策研究。

**💡 创新点**

创新点主要包括：① 三层安全机制——封闭 evidentiary sandbox、实时 provenance 标记和系统级输出协议；② 将传统 RAG 与可追溯性、失真检测、语义/布尔/生态搜索等多种检索方法无缝集成；③ 通过 deterministic watchdog 脚本在生成前验证所有引用，显著降低 hallucination；④ 通过多代理并行搜索实现高效跨文档分析。

**🔧 技术方法**

使用技术包括：Anthropic Claude LLM（Sonnet 5）配合预设的系统 prompt 与预填充 assistant turn；RAG（检索-生成）框架与向量检索；OCR 文本解析与质量评分；实时 provenance 标记（DK、INT、PS、T、OCR）；deterministic watchdog 脚本用于引用校验和 hallucination 检测；多种搜索模式（semantic、Boolean、Ecology）；多代理并行分析；可视化与计数工具。

**📊 数据集**

数据集：UCSF Industry Documents Library、Columbia & CUNY ToxicDocs、Stanford SRITA、Trinkets & Trash、FBarchive、OxCCAL、R. J. Reynolds litigation documents 等，总计数十亿级 token；还包括自建 OCR 版本、OpenAlex、PubMed Central 等公开数据库用于冲突检查。

**📈 对比分析**

比较方法：与未加安全措施的 Claude Sonnet 5 在相同文档集上对比；测量 fabrication、misgrounding、引用准确率等指标。结果显示 INDRA 在相同负载下 fabrication 率从 65% 降至 0%，misgrounding 率从 9.2% 降至 0.6%；在 context-utilization 上，当 token 触及 1M 时引用准确率降至 61%，而 INDRA 在 500k token 内保持 96% 以上。性能评估见 Appendix A.1-A.4。

**⚠️ 局限性**

局限性：① 对极大文档（>500k token）仍需分块；② OCR 质量不一导致部分文本需人工核实；③ 系统对外部知识仍有限，无法实时访问外部数据库；④ 仍需人工验证以避免潜在的细微 hallucination；⑤ Heraclitus 效应、Steppingstone dilemma、Gullibility（或 Mafia）等偏见问题在某些检索情境下仍可能出现；⑥ 资源消耗高，尤其是多代理并行时的算力与费用。

---

## 311. Rethinking Sparse Formats for RISC-V: A Hierarchical Approach to High-Performance SpMV

**arXiv ID:** 2609.11352 | [PDF](https://arxiv.org/pdf/2609.11352v1)

**作者:** Anna Pirova `[一作]` (Lobachevsky State University of Nizhny Novgorod), Iosif Meyerov `[通讯]` (Lobachevsky State University of Nizhny Novgorod)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RISC‑V平台上实现并对齐向量化的稀疏矩阵‑向量乘（SpMV）算法，比较九种主流稀疏矩阵存储格式，并提出并实现了一种新的分层CSR（HCSR）格式；将所有实现封装为RVVLASparse开源库；利用机器学习方法实现自动格式选择；在两代SpacemiT K1/K3 RISC‑V板上对121个SuiteSparse矩阵（+600个R‑MAT矩阵）进行性能评测；并通过与标量CSR及各格式的比速比进行可视化与统计；讨论内存带宽、向量长度(LMUX)与流水线等硬件因素对性能的影响。

**💡 创新点**

提出的HCSR格式在保持CSR易用性的同时，引入块级分层存储（CSR/COO块混合）并在块层级使用CSR索引，显著提升向量化SpMV的缓存局部性与内存访问效率；实现了完整的RVV1.0向量化代码；将自动格式选择从决策树回归/分类迁移到随机森林，首次在RISC‑V上验证基于矩阵结构的机器学习选择方案。

**🔧 技术方法**

RISC‑V向量指令集RVV1.0 intrinsics、OpenMP并行、LMUX参数调优、分块（Sell‑C‑σ、CSR5、VHCC、VNEC、CVR等）以及新HCSR格式；机器学习框架包括随机森林回归/分类、特征工程（矩阵行/列/块统计、Gini、p‑ratio等）。

**📊 数据集**

SuiteSparse矩阵集合121个（覆盖不同应用领域），以及600个R‑MAT随机生成矩阵；在两种硬件平台上（SpacemiT K1: 8×x60内核，K3: 8×x100 OoO内核）测试。

**📈 对比分析**

通过对比每种格式单次SpMV运行时间与标量CSR基准，计算相对加速比；在K1上HCSR平均提升1.6×、Sell‑C‑σ单核最佳；在K3上多核时HCSR仍保持领先，单核Sell‑C‑σ最快；机器学习自动选择实现平均1.4–1.7×加速，峰值可达2–6×；性能受LMUX、内存带宽、OoO执行等硬件因素显著影响。

**⚠️ 局限性**

自动格式选择的误差仍可达3×（LUB），对某些矩阵会选错；评测局限于8核单核与双精度/单精度；新格式与硬件密切耦合，对不同RISC‑V实现可能需要重新调参；未覆盖更复杂的稀疏线性算子（如矩阵‑矩阵乘、预条件器等）。

---

## 312. Reification as a Transferable Vocabulary: Zero-Shot Link Prediction with Vanilla GNNs

**arXiv ID:** 2609.11347 | [PDF](https://arxiv.org/pdf/2609.11347v1)

**作者:** Camille Pradel `[一作]` `[通讯]` (Matr), Camille Pradel (Matr)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究将知识图谱基础模型的转移机制从架构中移出，转而通过将输入图表示为节点来实现零样本链接预测。

**💡 创新点**

创新点在于将每个事实转化为节点，并通过固定的六个元关系词汇连接其主题、对象和关系类型，而不是将关系类型作为模型参数。

**🔧 技术方法**

使用了五种经典的图神经网络（GNN），包括GAT、GINE（求和和均值+最大聚合）、GraphSAGE和R-GCN。

**📊 数据集**

使用的数据集是一个包含4245个三元组的知识图谱，进行了30分钟的训练。

**📈 对比分析**

与ULTRA模型进行比较，最佳的GAT模型在ULTRA的评估套件中表现相当，且在40个归纳链接预测基准上实现了零样本转移。

**⚠️ 局限性**

限制在于初步探测的两个未见数据库没有单元值、模式文本或上下文标签，可能影响模型的表现。

---

## 313. On the Impact of Anonymization on the Performance of Large Language Models

**arXiv ID:** 2609.11335 | [PDF](https://arxiv.org/pdf/2609.11335v1)

**作者:** Tobias Deußer `[一作]` (University of Bonn), Rafet Sifa `[通讯]` (University of Bonn)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对LLM输入进行匿名化处理，并评估其对模型性能的影响。

**💡 创新点**

系统化比较了多种匿名化技术对不同模型与任务的影响，并揭示高能力模型更易受损。

**🔧 技术方法**

采用基于LLM的伪匿名化工具、可逆/不可逆匿名化方法、以及显式提示等技术。

**📊 数据集**

使用11个公开基准（ARC、BIG‑Bench Hard、EQ‑Bench、HellaSwag、IFEval、MedQA、MMLU‑PRO、MUSR、RGB、TruthfulQA、WMT 2014）。

**📈 对比分析**

通过对比原始与匿名化输入下的分数，发现整体平均下降约5‑7个百分点；TruthfulQA偶有提升，RGB剧烈下降。

**⚠️ 局限性**

仅覆盖有限模型与任务，未考虑细粒度匿名化或模型微调，结果可能随新模型或不同语言环境变化。

---

## 314. CoSTAR: Data Synthesis-Driven Constraint-Aware COBOL Section Summarization for Legacy System Modernization

**arXiv ID:** 2609.11332 | [PDF](https://arxiv.org/pdf/2609.11332v1)

**作者:** Hao Lin `[一作]` (Dalian University of Technology), Ang Jia `[通讯]` (Dalian University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CoSTAR框架，利用执行验证的数据合成和约束感知模型训练，生成COBOL段级摘要，支持遗留系统现代化。

**💡 创新点**

创新点：①将通用编程任务的自然语言描述与可执行测试合成并验证COBOL代码‑摘要对，解决数据匮乏；②通过提取相关数据定义、生成标识符解释与结构化推理，训练两个小型模型，实现迁移约束保留；③模型推理不依赖大型LLM，满足企业隐私与本地部署需求。

**🔧 技术方法**

技术：LLM代码生成、编译器+测试验证、正则标识符提取、教师-判定循环、LoRA微调、四阶段结构化推理（解释→约束分析→逻辑抽象→最终摘要）。

**📊 数据集**

数据集：CodeFlowBench（通用任务源），合成的3,764条执行验证COBOL实例；公开评估集Stack‑120（120段），保密评估集Industrial‑200（200段）。

**📈 对比分析**

评估方式：与七个大型LLM基线、基线模型及摘要仅监督对比；在公共集上ROUGE‑L提升25.38%、METEOR 53.84%、chrF 37.22%；在工业集上Qwen3‑8B超越企业部署的Qwen3‑235B，准确性、完整性、简洁性分别提升4–8%。

**⚠️ 局限性**

局限性：依赖大型LLM生成监督，测试覆盖有限；合成数据可能包含噪声；仅针对COBOL段级，泛化到其他语言/粒度需改动；模型训练资源与可扩展性待进一步验证。

---

## 315. A Dynamic Fusion Large Language Model for Traffic Flow Prediction

**arXiv ID:** 2609.11314 | [PDF](https://arxiv.org/pdf/2609.11314v1)

**作者:** Xue Qiu `[一作]` (University of Shanghai for Science and Technology), Jianli Xiao `[通讯]` (University of Shanghai for Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种动态融合的大语言模型DF-LLM，用于交通流量预测

**💡 创新点**

将图卷积与预训练Transformer相结合，采用差异化参数适配与上下文聚合多头注意力，解决了传统模型无法充分捕捉空间拓扑与长时序依赖的问题

**🔧 技术方法**

多尺度时空嵌入、双层GCN、差异化冻结的GPT‑2、Context Aggregation MHA、残差连接等技术

**📊 数据集**

四个公开交通数据集：PEMS04、PEMS08、METR‑LA、PEMS‑BAY

**📈 对比分析**

与多种深度学习与LLM基准方法（如DCRNN、STGCN、GWNet、ST-LLM、GCNGPT等）对比，DF-LLM在MAE/RMSE/MAPE上均优于大多数基准，并在不同场景下保持稳定表现

**⚠️ 局限性**

受限于预训练模型与交通数据本质差异，仍需进一步提升对多模态数据的融合与动态时空建模的适应性

---

## 316. GRIPNet: Gaussian Radial Intensity Prior Guided Architecture for Pulmonary Nodule Detection in CT

**arXiv ID:** 2609.11312 | [PDF](https://arxiv.org/pdf/2609.11312v1)

**作者:** Haojie Yang `[一作]` (Tianjin University), Ran Su `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了基于高斯径向强度先验的肺结节检测网络GRIPNet

**💡 创新点**

以物理先验为指导，将高斯径向强度模型映射到网络模块，实现了针对肺结节特性的专属架构

**🔧 技术方法**

Pinwheel卷积、双频特征分解、膨胀掩码注意力、适应性难度衰减损失，以及YOLOv11骨干网络

**📊 数据集**

KanserSet、LUNA16、Lung‑PET‑CT‑Dx 三个公开 CT 结节数据集

**📈 对比分析**

与 YOLOv8/10/11、Faster R‑CNN、SSD、MSDet、YOLOv5‑CASP、改进 YOLOv11-SSE 等方法对比，GRIPNet 在所有数据集上均取得最高 mAP@0.5（KanserSet 95.3%、LUNA16 91.6%、Lung‑PET‑CT‑Dx 97.9%），并在高 IoU 阈值下表现突出

**⚠️ 局限性**

仅在 2D 切片上工作，假设结节为各向同性；对高度异形或极少见的结节识别召回率略低于部分竞争方法

---

## 317. 2AM: Grounding Agent-Side Memory as Guidance for Steerable Action Models in Long-Horizon Manipulation

**arXiv ID:** 2609.11308 | [PDF](https://arxiv.org/pdf/2609.11308v1)

**作者:** Yutong Hu `[一作]` (KU Leuven), Renaud Detry `[通讯]` (KU Leuven)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将长期任务记忆集中在Agent中，使用仅RGB的无状态动作模型执行低层控制，并通过可选的2D把握、放置和移动提示与子任务语言共同构成可调节的驱动接口。

**💡 创新点**

创新点在于（1）将记忆与动作完全分离，仅通过语言和3种空间提示实现Agent与动作模型的交互；（2）利用演示数据自监督训练，结合提示丢弃、空间噪声和时间抖动增强接口鲁棒性；（3）在LIBERO‑Mem上展示单一动作模型即可实现长周期任务的显著提升。

**🔧 技术方法**

技术手段包括RGB‑only感知、Qwen3‑VL‑4B 语言视觉模型、流匹配动作专家、可组合的语言+二维提示接口、演示导向的监督、提示丢弃和噪声训练。

**📊 数据集**

使用LIBERO‑Mem基准数据集（10个长期操作任务）。

**📈 对比分析**

与SlotVLA、SlotSSM以及自行复现的π_0对比，本文方法在无深度、无在线几何或规划辅助的情况下完成率提升至76.3%（相对最佳基线14.8%提升61.5个百分点），放宽成功率为63%（相比π_0提升25.6个百分点），严格成功率约为11.8%（与π_0相当）。

**⚠️ 局限性**

局限性包括仅在仿真环境下验证；仅依赖RGB难以处理隐式几何或力学信息；严格终止仍未得到一致提升；未分别评估各提示的贡献；未在真实机器人上测试；未分析频率对能力与延迟的权衡。

---

## 318. Exploring the Role of Security Experience and ChatGPT Usage Strategies on Secure Software Engineering Education

**arXiv ID:** 2609.11303 | [PDF](https://arxiv.org/pdf/2609.11303v1)

**作者:** Alessio Ferrari `[一作]` (Trinity College Dublin), Liliana Pasquale `[通讯]` (University College Dublin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对26名研究生在漏洞修复任务中使用ChatGPT的交互日志进行编码，探索其使用模式与作业成绩的关联。

**💡 创新点**

创新点在于将LLM使用视为多模式交互而非二元使用，发现使用多样性与成绩正相关。

**🔧 技术方法**

采用ChatGPT交互记录、双人编码、Spearman相关和线性回归等统计技术。

**📊 数据集**

使用26名研究生的ChatGPT聊天日志和预测测试得出的专业水平划分。

**📈 对比分析**

通过比较不同使用模式和使用多样性与成绩的相关性，发现每增加一种使用模式成绩平均提升约3分，相关性显著。

**⚠️ 局限性**

局限包括样本量小、观察性设计、缺乏因果性检验、对使用模式深度评价不足。

---

## 319. Memory Compression for High-Fanout Agent Sandboxes

**arXiv ID:** 2609.11294 | [PDF](https://arxiv.org/pdf/2609.11294v1)

**作者:** Mengming Li `[一作]` (HKUST), Zhiyao Xie `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为高并发 AI‑agent 沙箱设计了一套完整的内存压缩系统 AgentZip，利用模板相对相似性和跨沙箱相似性压缩私有页面，并将压缩工作与 LLM 等待阶段对齐，同时通过预取恢复减少页面 fault 影响。

**💡 创新点**

创新点包括：① sandbox‑aware 的冗余模型，支持模板增量压缩、跨沙箱字典压缩和 RLE；② 解除传统压缩时的“热度”限制，允许压缩任何可收益的页面；③ 将压缩时机与 agent 生命周期同步，轻量级候选发现+LLM 等待压缩；④ 复合预取器（stride、时间序列、工具热集）提前恢复压缩页面，显著降低恢复放大。

**🔧 技术方法**

核心技术：Zstd 动态字典训练与使用、模板增量 delta 编码、RLE、Linux UFFD 页 fault 处理、预取器模型、基于 cohort 的字典共享、E2B 接口兼容层、Zeroboot/KVM 沙箱快照、Intel Xeon 处理器。

**📊 数据集**

实验使用 R2E‑Gym（10 个 Python 仓库）构建的两种工作负载：训练阶段的 Parallel Rollout（16 条轨迹）和推理阶段的 Generate‑and‑Filter（4 条候选轨迹），采用 DeepSeek‑V4 LLM 并模拟不同温度和角色提示。

**📈 对比分析**

与无压缩、zswap、KSM+zswap 进行对比。AgentZip 在 Rollout 下平均内存占用减少 88.55%（相当于 8.7×），在 GAF 下 64.29%；延迟提升仅 1.40×/1.47×，优于 zswap（1.43×/1.12×）和 KSM+zswap（1.52×/1.16×）。显著提高了沙箱部署密度，同时保持了可接受的性能开销。

**⚠️ 局限性**

局限性：在高多样化候选（如 GAF）中字典压缩收益下降；预取器依赖历史访问模式，对极端访问变化不敏感；压缩池与字典存储在内存中的占比约 5‑30%；实现复杂度较高，需在 Linux 内核层与用户空间协同工作。

---

## 320. Off-Target Effects of Response-Style Alignment in a Korean 27B Language Model

**arXiv ID:** 2609.11291 | [PDF](https://arxiv.org/pdf/2609.11291v1)

**作者:** Hyojung Han `[一作]` `[通讯]` (ThakiCloud), Hyojung Han (ThakiCloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对Qwen3.8-27B进行后训练风格对齐，改变回答倾向和输出长度，从而影响未目标的放弃回答和披露行为。

**💡 创新点**

首次展示风格对齐可以通过改变模型的回答频率和篇幅而影响与安全和合规相关的未目标行为，且提出了基于答案倾向与条件组成分离的解析框架。

**🔧 技术方法**

采用LoRA参数微调、SFT、DPO等技术，并在训练中使用了风格重写目标和安全对齐。

**📊 数据集**

使用韩语KoBBQ问答数据集和韩国证券标准披露规则集，评估回答和披露指标。

**📈 对比分析**

通过模板聚类的配对自举、McNemar检验等统计方法对比不同对齐阶段，结果显示风格对齐导致回答率上升≈1.3%并显著减少披露率≈22个百分点，其他对齐阶段对这些指标影响不大。

**⚠️ 局限性**

局限包括仅在韩语单一基准上验证、训练样本量小、无法精确分离目标长度与风格重写的因果机制、以及检测器对不同输出分布的不稳定性。

---

## 321. Generating a Consistent Enterprise: Synthesis and Reference-Free Evaluation of Multi-System Business Data

**arXiv ID:** 2609.11286 | [PDF](https://arxiv.org/pdf/2609.11286v1)

**作者:** Benjamin Gruenbaum `[一作]` (Eon), Omer Niv `[通讯]` (Eon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

生成无真实数据的完整虚构企业，包括人员、客户、交易、支持、文档等多系统数据；

**💡 创新点**

通过参考统计目标驱动生成、基于确定性随机流实现跨系统一致性，并使用无参考评估与对抗检测两种方法；

**🔧 技术方法**

利用确定性子系统随机流、参考目标驱动分布、Boosted‑Tree 关系检测、卡方误差评估、模板化文本生成等技术；

**📊 数据集**

不使用任何真实数据集，而是依据公开行业统计、运营经验及估计值构建参考目标；

**📈 对比分析**

采用固定分数卡与对抗检测对比两代生成器，在23个企业上从平均 60.3 分提升至 99.1 分，最差企业从 41.1 分提升至 94.9 分，检测率从 55.2% 降至 0%；

**⚠️ 局限性**

受限于参考目标的准确性、模板化文本的多样性不足以及未与其他生成器进行广泛基准对比。

---

## 322. Can AI Remediate Backend Failures Safely? GuardedAct with Blast-Radius-Aware Sandboxing

**arXiv ID:** 2609.11264 | [PDF](https://arxiv.org/pdf/2609.11264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 323. Max Independent Set Remains NP-hard when Excluding a Planar Induced Minor

**arXiv ID:** 2609.11285 | [PDF](https://arxiv.org/pdf/2609.11285v1)

**作者:** Édouard Bonnet `[一作]` (Université Claude Bernard Lyon 1), Yeonsu Chang `[通讯]` (Hanyang University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了一种从任意图的 Max Cut 归约到在排除 5×5 网格为诱导子图的图上求最大独立集（MIS）的实例，证明该问题在此类图中仍为 NP‑难，并给出了相应的 ETH 下的时间下界。

**💡 创新点**

创新点在于：①首次提供了一个固定平面图（5×5 网格）作为诱导子图排除基，且在该类图中 MIS 仍为 NP‑难，直接否定了 Dallard–Milanič–Štorgel 等人的猜想；②设计了巧妙的“行/列”网格结构，利用其特定的邻接性质，保证生成的图不含 5×5 网格为诱导子图；③通过对 MIS 的大小与 Max Cut 目标的紧密关联，给出了精确的多项式关系，进一步推导出 ETH 下的 2^Ω(√n) 时间下界。

**🔧 技术方法**

主要技术包括：图论中的诱导子图排除技术、强直积（路径×边）构造、行列列举与连通性分析、以及从 Max Cut 归约到 MIS 的构造性证明；在下界证明中还运用了 Sparsification Lemma 与 3‑SAT 的线性大小变换。

**📊 数据集**

论文为理论性研究，没有使用实验数据集；归约构造生成的实例尺寸为 O(|V(F)|²)（若 F 为三角图则可进一步压缩）。

**📈 对比分析**

由于是理论证明，没有实验对比；作者只给出了在该类图上 MIS 的 NP‑难性和相应的 ETH 下界，未提供算法实现或性能评估。

**⚠️ 局限性**

局限性在于：①仅对 5×5 网格为诱导子图的排除类给出了 NP‑难性结果，其他平面诱导子图仍可能可多项式求解；②归约仅证明了 NP‑难性，并未提供多项式或近似算法；③下界仅基于 ETH，若该假设失效则结论不再成立。

---

## 324. Buyer Artificial Intelligence-Enabled Environmental Governance and Supplier Environmental Controversies: An Organizational Information Processing and Signaling

**arXiv ID:** 2609.11391 | [PDF](https://arxiv.org/pdf/2609.11391v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 325. Xiaomi-CocktailASR-1 Technical Report

**arXiv ID:** 2609.11274 | [PDF](https://arxiv.org/pdf/2609.11274v1)

**作者:** Yiru Zhang `[一作]` (Xiaomi Inc.), Heng Qu `[通讯]` (Xiaomi Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种基于LLM的端到端目标说话人ASR系统Xiaomi-CocktailASR-1，能够在多说话人场景下直接使用参考语音进行语音识别，并兼容单说话人与负样本拒绝。

**💡 创新点**

创新点在于：1）采用参考音频作为声纹提示直接对目标说话人进行识别，无需前端分离；2）引入负样本拒绝机制输出空文本；3）支持链式推理（CoT）提供可解释性，并在一个统一架构中平衡单说话人和多说话人性能。

**🔧 技术方法**

技术上基于Data2Vec的音频编码器与Qwen3-8B LLM，通过Adapter实现跨模态对齐；利用声纹相似度量与离散化；训练分四步，包含负样本与CoT训练；并用RL微调提升鲁棒性。

**📊 数据集**

使用了约400k小时多说话人混合数据（AMI、AliMeeting、LibriMix等）、600k小时单说话人数据、10k小时负样本数据，测试集包括LibriMix、LibriSpeechMix、AMI-SDM、AliMeeting-Far、LibriSpeech、AMI-IHM等。

**📈 对比分析**

通过与现有SOTA TS-ASR、主流单说话人ASR以及多模态LLM在合成与真实多说话人测试集上对比，使用TS-WER、RR、FRR、Non-empty WER等指标。Xiaomi-CocktailASR-1在LibriMix 2mix的TS-WER从4.84%降至4.11%，在AliMeeting-Far从27.5%降至20.63%，负样本拒绝率高达79.6%，单说话人FRR仅0.36%。

**⚠️ 局限性**

局限在于负样本拒绝虽然实现，但仍有少量误拒（FRR），对极端噪声或低SNR场景的鲁棒性待进一步验证；在多说话人数量极大或说话人相似度极高时性能可能下降；CoT机制提升有限，需要更多任务验证。

---

## 326. Improving Faint Object Detection for Space Situational Awareness with Variational Autoencoders

**arXiv ID:** 2609.11269 | [PDF](https://arxiv.org/pdf/2609.11269v1)

**作者:** Angela Cratere `[一作]` (Maastricht University), Roberto Furfaro `[通讯]` (University of Arizona)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一套基于深度学习的光学SSA预处理管线，用 Tiny‑U‑Net 进行恒星掩模生成，astro‑VAE 进行背景再生，随后通过 Shift‑and‑Stack（S&S）提升对极低信噪比移动目标的检测能力。

**💡 创新点**

创新点包括：① 轻量化 Tiny‑U‑Net 与概率性 PC‑VAE astro‑VAE 的结合，实现对稠密星场和结构化背景的高质量去星与再生；② 通过学习分布而非确定性网络，显著减少模型参数并提升重建细节；③ 在真实 X‑GEO 观测数据上进行端到端验证，展示了相较传统方法提升约 4–5 倍的堆叠 SNR。

**🔧 技术方法**

采用 Tiny‑U‑Net（U‑Net 结构精简版）生成二值星掩模，astro‑VAE（U‑Net‑style 结构 + PC‑Layer + 变分推断）实现背景上下文感知的掩区填补；后续使用 Shift‑and‑Stack 对轨迹已知的目标进行叠加。

**📊 数据集**

使用 290 张真实地面望远镜观测图像（X‑GEO 区域），包括 213 张用于训练、77 张用于测试；图像尺寸从 1192×798 到 2392×1596，全部经过 bias/dark/flat 校正。

**📈 对比分析**

与基准 U‑Net（标注性能更好但参数高）相比 Tiny‑U‑Net 参数减少 88.8% 但 IoU 仅下降 1.92%；与传统 PCNN 及 Astropy Photutils 背景估计方法比较，astro‑VAE 在 SSIM、PSNR、MSE 上均优于 PCNN（约 0.963 vs 0.950 SSIM），参数量仅为其一半；在 S&S 实验中，astro‑VAE 预处理后堆叠 SNR 提升 4–5 倍，显著提升目标可见度。

**⚠️ 局限性**

局限性：① 训练数据仍受限于有限的地面观测样本，泛化到不同光学系统或不同背景纹理仍需验证；② 目前模型仍为离线训练，尚未实现边缘设备的实时部署；③ 对目标轨迹不确定的场景下，如何在多速度假设下保持目标完整性及降低误报仍待进一步研究。

---

## 327. TransClean: A Benchmark for Detecting and Extracting Clean Translations from Large Language Model Outputs

**arXiv ID:** 2609.11399 | [PDF](https://arxiv.org/pdf/2609.11399v1)

**作者:** Shenbin Qian `[一作]` (University of Oslo), Yves Scherrer `[通讯]` (University of Oslo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析LLM翻译输出中的噪声模式，构建TransClean基准，并在其上评估清洗方法。

**💡 创新点**

首次系统性归纳翻译噪声，提出12种噪声模式并基于此创建合成与真实混合的基准。

**🔧 技术方法**

采用规则检测、LLM生成噪声、基于质量估计的span提取、以及LLM直接提取器等技术。

**📊 数据集**

使用22语种、66k句对的公开语料做源句，生成792k LLM翻译样本，构成9,900对的TransClean（8,800合成+1,100真实）。

**📈 对比分析**

通过规则检测、span提取、Qwen、Aya等方法比较，检测几乎100%准确，提取最佳在合成数据上约54%精度，span在真实数据上性能显著下降。

**⚠️ 局限性**

检测手段粗糙、合成噪声与真实多样性不完全匹配、评价过度严格导致对语义相似但不同表述的误判。

---

## 328. DINO-Med: A Unified Patch-Based Adaptation Framework for Multi-Modal Medical Image Analysis Applied to Liver Fibrosis Staging

**arXiv ID:** 2609.11380 | [PDF](https://arxiv.org/pdf/2609.11380v1)

**作者:** Boya Wang `[一作]` (University of Nottingham), Xin Chen `[通讯]` (University of Nottingham)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出统一的基于patch的框架，将冻结的DINOv3迁移到多模态医学影像，用于肝纤维化分期。

**💡 创新点**

使用训练‑free配准、自动定位、掩码过滤的patch提取，并将多模态DINOv3特征进行融合，构建分层聚合实现无微调的高性能。

**🔧 技术方法**

采用冻结的DINOv3 ViT‑S/16特征提取、轻量MLP分类头、掩码导向patch筛选、两阶段聚合，与Radiomics、ResNet、SAM‑Med2D等基线对比。

**📊 数据集**

使用CARE 2025 Liver Track 4多参数MRI数据集（360例含T1、T2、DWI）进行实验。

**📈 对比分析**

采用10次10% hold‑out嵌套交叉验证，评估AUC/ACC；DINOv3在S1和S4均显著优于基线，S4 ACC≈76%，S1 ACC≈78%。

**⚠️ 局限性**

仅处理S1与S4，未覆盖S2、S3；缺乏跨序列早期/晚期融合，导致中间阶段分级的鲁棒性不足。

---

## 329. Deep operator learning for efficient sampling from invariant measures of stochastic differential equations

**arXiv ID:** 2609.11376 | [PDF](https://arxiv.org/pdf/2609.11376v1)

**作者:** Lin Guo `[一作]`, Jingtong Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种可摊销的神经采样器，用于在参数化随机微分方程族的不可约分布上快速生成样本。

**💡 创新点**

创新点在于将神经算子学习与连续归一化流（flow matching）相结合，采用随机拉格朗日轨迹传感器与跨注意力编码器，解决高维网格化困难，并在理论上证明了表达能力和分辨率不变性。

**🔧 技术方法**

使用神经算子（Perceiver+AdaLN）、连续归一化流、流匹配损失、注意力聚合、随机轨迹传感器等技术。

**📊 数据集**

在1D、2D 以及64维粒子系统的 SDE 家族上进行实验；训练样本来源于已知的平稳分布（GMM、GPR 生成的势能等）和 MCMC 采样。

**📈 对比分析**

与传统 MCMC（以及 Grid DeepONet）进行对比；在慢混合、多模态、以及高维情形下，模型在样本质量（Sinkhorn / W2）上与 MCMC 同级或更优，同时显著减少采样时间。

**⚠️ 局限性**

局限包括：需要监督式训练且需预先生成大量目标分布样本；目前仅处理平稳分布，未直接学习时间演化；在更大规模高维粒子系统上的性能与可扩展性尚未完全验证。

---

## 330. A Tight Second-Order Converse Bound for Variable-Length Feedback Codes

**arXiv ID:** 2609.11368 | [PDF](https://arxiv.org/pdf/2609.11368v1)

**作者:** Recep Can Yavas `[一作]` `[通讯]` (Bilkent University), Recep Can Yavas (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在离散无记忆信道下，研究可变长度反馈编码（VLF）的非消失误码概率场景，给出了精确的二阶极限表达式，并进一步阐明了达到一阶和二阶最优的码序列的结构特征；针对二进制擦除信道（BEC），给出了每个消息集大小和误码概率下的最小期望译码时间的解析式；

**💡 创新点**

创新点包括：1) 用Rényi熵和EJS散度的结合，首次实现了对VLF编码的二阶紧致逆界，消除了Polyanskiy等人提出的order-log N缺口；2) 推导了通用DMC在子指数误码概率衰减、指数误码率和误差指数三个不同区间内统一的极限展开；3) 提出了对一阶、二阶最优VLF码序列的必要结构性要求（早停分支、通信与确认行为）；4) 在BEC上给出了全范围内的最优期望译码时间解析公式，桥接了即刻猜测和零误差Huffman传输两极端。

**🔧 技术方法**

主要技术手段包括：停止Rényi熵的不等式与一阶Shannon熵漂移的比较；EJS散度与互信息差的比较，得到累计容量缺口下界；利用马尔可夫链、Doob停定定理和Azuma–Hoeffding不等式进行期望译码时间分析；对BICF树的前缀树长度极小化分析；以及对Yavas–Tan修改的Yamamoto–Itoh构造进行参数优化。

**📊 数据集**

本文为理论研究，无需实际数据集；所有结果均为解析式和渐近极限。

**📈 对比分析**

作者通过与Polyanskiy等人提出的下界、Yavas–Tan的上界以及Burnashev指数等经典结果对比，证明在所有C>0且C₁<∞的DMC上，其逆界与构造上界完全匹配，二阶项系数-C/C₁得到实现；在BEC上，给出的解析公式与已知的极限一致，并在误码概率可调范围内实现最优。

**⚠️ 局限性**

局限性主要在于：1) 对于C₁=∞的DMC（除BEC外）尚未给出二阶极限；2) 对VLSF（仅停止反馈）编码的二阶极限仍未确定；3) 需要假设有限输出支持与最大KL散度有限，若不满足则结果不适用。

---

## 331. GeoTrussRover: Morphological Computation with Contact-Semantic Control Primitives

**arXiv ID:** 2609.11361 | [PDF](https://arxiv.org/pdf/2609.11361v1)

**作者:** Muyuan Ma `[一作]`, Yue Xie `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一种可变几何三角桁架（VGT）机器人GeoTrussRover，能够通过结构形变与轮子配合穿越垂直障碍。

**💡 创新点**

提出了基于接触语义的低维本体坐标（四阶段primitive）以及物理投影与PG-QP控制，实现了在不同高度下重用已求解轨迹，显著减少规划计算量。

**🔧 技术方法**

使用电动双轴螺杆驱动的可伸缩桁架、闭环图形运动学、接触条件约束、物理投影优化、全空间QP跟踪、仿真(MuJoCo/IsaacSim)和实机验证。

**📊 数据集**

使用无公开数据集；通过仿真中不同高度步子（0.10-0.46 m）以及真实硬件在0.195 m步子上的测试来验证。

**📈 对比分析**

与全规划(Full)、无primitive全空间QP(RF‑QP)和硬子空间QP(PS‑QP)对比，primitive+PG‑QP在保持运动完成率的同时将目标评估次数减少约63%~32%，规划时间缩短到约1/4，且实机能通过2.11轮半高度。

**⚠️ 局限性**

局限在于对接触模型的简化、未充分校准的动力学参数、在高高度或极端加载下可能需要重新计算整个阶段，且实测受限于电机速度与摩擦不一致。

---

## 332. Taming Bitwise Behavior in GPU Kernels with Tensor Core: Black-Box Reconstruction, Compiler Enforcement, and Static Verification

**arXiv ID:** 2609.11356 | [PDF](https://arxiv.org/pdf/2609.11356v1)

**作者:** Ziteng Yang `[一作]` (Georgia Institute of Technology), Vivek Sarkar `[通讯]` (Georgia Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文研究了GPU kernel在浮点累加过程中的位级行为，并提供了黑盒重建、编译器强制平衡树累加以及静态等价检查器三项技术，实现了与cuBLAS等闭源库的位级一致性；

**💡 创新点**

创新点包括：①构造完整的GEMM描述符（GEMMDesc）捕获所有决定位级结果的参数；②首次通过黑盒实验重建cuBLAS的算术实现，实现不同GPU架构的100%位级匹配；③在Triton编译器中实现平衡树累加和布局优化，并证明其性能损失可忽略；④实现针对NVIDIA PTX和AMD GCN的声称安全的静态等价检查器，可在自动调优前进行等价类划分；

**🔧 技术方法**

技术手段：GPU内核静态分析、符号执行、依赖树建模、数值实验（对极端值+L、-L、r的插值）、自动调优（Triton autotuner）、LLVM后端重写、数据布局优化、SMT求解器等；

**📊 数据集**

数据集：从16款开源大模型（如Qwen3、DeepSeek-V4、GLM-5.2等）的层尺寸随机采样的390个fp16 M,N,K组合，另外使用110,813个随机fp16形状做黑盒重建验证；还包括1,400个形状做性能评测；

**📈 对比分析**

比较方法：对每个形状在10个输入绘制下，逐字节比较cuBLAS与重建的Triton GEMM的输出；性能上，未强制顺序的Triton GEMM在大部分形状上达到61-88%（fp16）或69-91%（大张量加速）的cuBLAS吞吐量，位级一致的Triton GEMM在5GFLOP阈值以上约56-93%；对融合epilogue时，位级一致的融合核可实现85-125%（自由顺序）和95-168%（位级一致）的cuBLAS基线；平衡树与布局优化后，19/27核在10%以内，H100上部分核可超越自由模式；

**⚠️ 局限性**

限制：①未验证跨机器/不同SM代的鲁棒性；②未评估系统级（多核、多张量）训练/推理中的收益；③检查器仅覆盖PTX/GCN汇编，未下推至SASS；④对注意力和epilogue融合的位级影响仅给出理论分析；⑤未实现生成器（如TorchInductor）自动固定顺序的完整机制。

---

## 333. FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model

**arXiv ID:** 2609.11445 | [PDF](https://arxiv.org/pdf/2609.11445v1)

**作者:** Haoran Pei `[一作]` (Chinese Academy of Sciences), Ruixi Ci `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并评估了 FARM，一种仅训练轻量读取器的失败监测框架，利用冻结的 VLA‑JEPA 世界模型预测状态直接解码执行失败风险。

**💡 创新点**

创新点在于：①证明预训练世界模型内部预测状态已包含可解码的失败信息；②仅训练约 33,985 参数的读取器即可实现多任务、跨平台、零样本和仅读取器适配的监测；③将因果风险聚合与低延迟推理结合，避免更新背骨网络。

**🔧 技术方法**

采用的技术包括：冻结 VLA‑JEPA 的 12 层预测输出作为 WM 状态；共享投影+注意力池化+MLP 读取器；二元交叉熵训练；因果风险最大化聚合；多任务训练、零样本转移与读取器适配实验；CUDA 延迟测量。

**📊 数据集**

使用数据集：10 任务 LIBERO 仿真轨迹（共 500 条）以及四套真实机器人平台（PIPER X、SO‑101、Franka）各自的任务轨迹；包括 Seen（S1–S7）和 Strict‑Unseen（U1–U3）子集；扩充版 446 条训练轨迹用于零样本和适配实验。

**📈 对比分析**

与 15 种基线（SAFE、距离/OOD、动作不确定性等）在 Seen、Strict‑Unseen、Adapt 等场景下对比。FARM 在 Seen 任务中 AUROC/AUPRC 均超 85/88，跨平台零样本也优于大部分基线；仅读取器适配可进一步提升到 90+ AUROC。总体性能在多任务与真实机器人评估中均保持在 70–90+ 的优秀水平。

**⚠️ 局限性**

局限性：需要访问内部世界模型状态和标注失败数据进行读取器训练/适配；在严格未见任务的零样本性能仍显不足；仅读取器方式对极端或多模态失败可能不够敏感；依赖于预训练的 VLA‑JEPA，未探讨不同背骨模型的泛化；未实现与干预/恢复策略的闭环集成。

---

## 334. Multi-Modal Controlled Coherent Motion Generation

**arXiv ID:** 2609.11439 | [PDF](https://arxiv.org/pdf/2609.11439v1)

**作者:** Yifei Liu `[一作]` (South China University of Technology), Changxing Ding `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为MOCO的扩散式框架，能够在文本、语音和轨迹多模态控制下生成完整、自然的3D人体运动；

**💡 创新点**

创新点在于将扩散过程解耦成各模态独立去噪步骤，并按预定义的空间规则在每一步聚合身体部位，从而实现多模态输入的无缝融合和自我迭代细化；

**🔧 技术方法**

使用了Motion Diffusion Model（MDM）作为基础，结合四个Transformer去噪器（文本→运动、语音→手势、轨迹→速度、语音→细节）以及无监督的分类器自由引导；

**📊 数据集**

训练数据来自HumanML3D（文本→运动）与BEAT2（语音→手势）两个大型数据集，并在此基础上构建了1,000条含双文本+双语音的多模态基准；

**📈 对比分析**

与加权求和、伪文本、SynTalker、STMC等基线对比，MOCO在文本-运动和语音-手势指标上均取得最高分，尤其在同步性和自然度上明显优于对照；

**⚠️ 局限性**

局限在于固定的身体部位分配（语音主导上半身、文本主导下半身），当模态控制冲突时会出现无法跟随文本指令的情况，并且目前的评估仅覆盖构成性一致性，未验证对开放式文本与语音的泛化能力。

---

## 335. Entropy concavity for log-concave random variables: an asymmetric counterexample

**arXiv ID:** 2609.11418 | [PDF](https://arxiv.org/pdf/2609.11418v1)

**作者:** Congyi Luo `[一作]` `[通讯]` (Fudan University), Congyi Luo (Fudan University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文构建了一个不对称的平滑概率密度函数，证明了在没有额外对称假设的情况下，Ball-Nayar-Tkocz熵凹性猜想是错误的。

**💡 创新点**

创新点在于构造了一个严格正的平滑概率密度函数，展示了其加权和的微分熵在某些区间内是严格凸的，从而反驳了熵凹性猜想。

**🔧 技术方法**

使用了概率密度函数的构造和高阶导数的分析技术，特别是利用了Hermite多项式和积分的性质。

**📊 数据集**

构造的概率密度函数是基于高斯扰动的，具有均值为零和方差为一的特性。

**📈 对比分析**

通过与已知的高斯密度和其他对称密度的比较，展示了构造的密度在特定条件下的熵曲率表现，证明了F''(t)在某些区间内大于零，表明其严格凸性。

**⚠️ 局限性**

限制在于该反例不适用于具有额外对称假设的情况，因此未能完全解决在对称情况下的熵凹性问题。

---

## 336. Deep-Fake CAPTCHA: Mitigating Next-Generation Social Engineering Attacks

**arXiv ID:** 2609.11404 | [PDF](https://arxiv.org/pdf/2609.11404v1)

**作者:** Guy Frankovits `[一作]` (Ben Gurion University), Yisroel Mirsky `[通讯]` (Ben Gurion University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了DF-CAPTCHA，一种通过向实时通话者发放简单的挑战任务（如做手势或朗读短句），迫使实时深度伪造（RT-DF）系统在其受限能力范围之外生成内容，从而显著提升检测效果的主动式深度伪造防御框架。

**💡 创新点**

创新点在于：①首次将挑战-响应机制迁移至视频域，构建多模态深度伪造检测；②提出四项验证约束（真实性、身份一致、任务完成、时延）与可扩展的挑战库；③通过主动激活RT-DF的弱点，使被动检测模型在挑战条件下的误差显著放大，提升鲁棒性。

**🔧 技术方法**

核心技术包括：挑战生成与TTS交互、实时语音与视频特征提取、四个验证模块（ℛ、ℐ、𝒞、𝒯），以及多种深度学习与异常检测模型（SpecRNet、GMM‑ASVspoof、PC‑DARTS、LOF、RECCE、FFD、SRM、CORE、Xception 等），配合零样本识别、MFCC、ResNet‑3D、ECAPA‑TDNN 等预训练网络。

**📊 数据集**

使用的数据集：①语音方面——20名志愿者录制的 2,498 条真实语音、1,821 条 RT‑DF 语音、3,317 条真实挑战回复与 16,123 条伪造挑战回复；②视频方面——20名志愿者录制的 20 条真实视频、20 条 RT‑DF 视频、1,000 条真实挑战视频与 1,000 条伪造挑战视频；③公开数据集—ASVspoof‑DF（22,617 真 / 15,000 假）与 RITW（19,963 真 / 11,816 假）用于训练真实性模型。

**📈 对比分析**

通过与被动检测基线（如 SpecRNet、RECCE、CORE 等）对比，采用 AUC 与 EER 指标。DF‑CAPTCHA 在音频端可将准确率从 71% 提升至 91‑100%，在视频端从 75% 提升至 87‑100%，并在所有挑战条件下使 AUC 提升 0.1‑0.2 点，EER 降低 0.02‑0.1 点，表明主动挑战显著提升了检测性能。

**⚠️ 局限性**

局限性：仅针对实时深度伪造，对预录制深度伪造不具防御作用；挑战库需要随技术进步持续更新；在低风险环境下部署可能导致用户体验下降；在嘈杂或压缩的通话环境下，识别误差可能上升，导致误判。

---

## 337. "They don't care about this": A Systematic Study of TEE Build Reproducibility in the Wild

**arXiv ID:** 2609.11411 | [PDF](https://arxiv.org/pdf/2609.11411v1)

**作者:** Annika Wilde `[一作]` (Ruhr University Bochum), Ghassan Karame `[通讯]` (Ruhr University Bochum)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统分析115个开源TEE（SGX、TDX、SEV‑SNP）项目，结合12名SGX维护者访谈，量化其构建可复现性并挖掘技术与组织原因。

**💡 创新点**

首次给出TEE构建可复现性的整体度量（91%不可复现），揭示技术瓶颈与生态壁垒，并提出可复现构建实践与治理改进建议。

**🔧 技术方法**

利用可复现构建工具（Bazel、Nix、Yocto、Docker）、远程测量验证、手工构建与对比、访谈与主题分析。

**📊 数据集**

构建了115个TEE项目的样本（SGX 82、TDX 3、SEV‑SNP 30）并从中抽取50个SGX项目的维护者进行访谈。

**📈 对比分析**

通过对未指导与指导两种构建情景的可复现率对比，统计不同TEE技术与构建系统的差异；未给出传统性能指标，重点在可复现率差异显著。

**⚠️ 局限性**

限制包括低访谈响应率、仅涵盖开源项目、缺乏完整可复现构建环境、样本偏向SGX以及对其他TEE平台的覆盖不足。

---

## 338. R4Tun: LLM-guided adaptive segmental tunnel lining segmentation in point clouds

**arXiv ID:** 2609.11360 | [PDF](https://arxiv.org/pdf/2609.11360v1)

**作者:** Xinghui Tao `[一作]` (University of Cambridge), Brian Sheil `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了 R4Tun 框架，利用大语言模型在不改变原有 SAM4Tun 隧道衬砌点云分割管线的前提下，实现每个隧道的可审计、无标签、无再训练的参数自适应。

**💡 创新点**

创新点在于：①引入结构化上下文（memory、state、knowledge）和多代理 CoT 推理，使 LLM 能在每一阶段动态调整有限参数；②提供完整的调整理由日志，确保人类可复核；③在同一基准管线下实现跨 LLM 的一致性改进。

**🔧 技术方法**

使用技术包括：OpenAI GPT‑5.4、Microsoft Opus‑4.6、Google Gemini‑3‑Flash 三大 LLM；SAM4Tun（基于 SAM 的 2D 分割）点云预处理管线；结构化上下文与链式思考（CoT）策略；Python 脚本实现参数更新与日志记录。

**📊 数据集**

使用 Seg2Tunnel 数据集的 30 个隧道子集（13 正规、17 复杂）进行实验，评估每个隧道的分割 mIoU 与 OA。

**📈 对比分析**

通过与静态 SAM4Tun 基线、非 LLM 规则对照，mIoU 从 0.18 提升至 0.43–0.48（整体）或 0.784–0.796（近参考子集），OA 从 0.42 提升至 0.59–0.65；三款 LLM 在同一上下文下表现一致，差距仅在 95% 置信区间内重叠。

**⚠️ 局限性**

局限性包括：①仅基于单一专家调优的参考配置，导致离参考远的隧道仍难以突破 0.3‑0.5 的 mIoU 限值；②管线仅支持单隧道级别参数调整，无法解决每环密度差异和固定角度模板导致的类交换；③对结构性误差的修正需改造 SAM4Tun 本身，而非单纯参数调整；④在不同 LLM 或更大规模数据集上的泛化尚未验证。

---

## 339. ReGround: Grounding Reviewer Comments in Multimodal Evidence

**arXiv ID:** 2609.11460 | [PDF](https://arxiv.org/pdf/2609.11460v1)

**作者:** Serwar Basch `[一作]` (TU Darmstadt), Iryna Gurevych `[通讯]` (TU Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了大规模的 ReGround 数据集，将同行评审中的评论自动关联到原始提交论文中的多模态证据，并基于此提出评论定位为检索任务。

**💡 创新点**

创新点在于利用作者在答复中的论文引用作为高精度自动标注来源，规模化生成 3,656 篇论文、16,274 条评论-证据对，覆盖段落、章节、表格、图像等多模态证据；同时在统一、类型感知、联合多证据和视觉检索四种实验设置下对多种检索模型进行系统评测。

**🔧 技术方法**

技术方法包括稀疏检索（BM25 等）、稠密检索（Sentence‑Transformers、LaBSE 等）、交叉编码器、LLM 基于点对评分的检索、视觉–文本编码器（CLIP‑style、ViLT 等）以及多模态融合策略。

**📊 数据集**

使用的数据集为 ReGround，源自 NLPeer 原始匿名提交、评审和答复，共 3,656 篇论文、10,267 条评审评论和 16,274 条证据，证据类型包括段落、章节、表格、图像等。

**📈 对比分析**

与基线比较显示：统一检索下 LLM 排序器在 Recall@10 仅达 21%；类型感知（oracle）检索可提升至约 40%（不同类型各异，段落/章节最高 31%）；多证据检索 Recall@10 仅 24%；视觉检索在加入标题后提升约 10%（表格/图像分别从 53%→63%、55%→65%）。

**⚠️ 局限性**

局限包括：仅使用作者在答复中引用的证据，可能遗漏其他合理证据；多模态检索未统一为单一检索池；数据仅覆盖 NLP 领域，跨领域泛化未知；模型仍需改进证据类型推断和多证据聚合策略。

---

## 340. Flexible and Interpretable Accent Distance Measurements

**arXiv ID:** 2609.11458 | [PDF](https://arxiv.org/pdf/2609.11458v1)

**作者:** Charles McGhee `[一作]` (University of Cambridge), Kate M. Knill `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用声学到发音反演得到的发音特征与最优传输方法对两位说话者的口音进行比较与分类。

**💡 创新点**

创新点在于将可解释的发音表示与最优传输结合，既能在任意录音类型下衡量口音，又能显式定位发音差异（如Rhoticity）。

**🔧 技术方法**

采用深度发音反演、Art+VVN 14维发音特征、WavLM 最终层特征、DTW 对齐、最优传输网络三角算法。

**📊 数据集**

使用 VCTK 语料库（苏格兰、英格兰、爱尔兰三国说话者）以及 CommonAccent 用于评估。

**📈 对比分析**

通过口音分类准确率、Spearman 相关系数与说话者嵌入距离的对比，结果显示对齐与 OT 的发音特征可达 94–99% 的分类准确率，且与说话者嵌入的相关性低于 GenAID；OT 的性能略逊于对齐，但仍优于纯音素距离。

**⚠️ 局限性**

局限性包括对录音时长与聚类数的敏感性、发音特征缺失 F0 信息、对极端口音或标签错误的鲁棒性不足。

---

## 341. Calibration-Aware Uncertainty Cascades for Efficient Heterogeneous Model Collaboration

**arXiv ID:** 2609.11446 | [PDF](https://arxiv.org/pdf/2609.11446v1)

**作者:** Yilin Zhang `[一作]` (Xidian University), Wei Zhao `[通讯]` (Xidian University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于后置校准的多模型协同框架CAUC，能够在多模型系统中通过统一的置信度阈值实现早停、选择性推理和自适应融合。

**💡 创新点**

创新点在于独立校准每个模型的置信度以建立统一可靠性尺度，允许在不重新训练路由器的前提下轻量替换/扩展模型，并通过阈值补偿理论解释阈值的风险意义。

**🔧 技术方法**

使用温度缩放进行置信度校准、阈值决策、置信度加权融合、递归融合（CAUC‑RF）以及基于验证集的阈值优化。

**📊 数据集**

在六大多选语言模型基准（MMLU、LogiQA、MathQA、MedMCQA、PIQA、SocialIQA）和三张图像分类数据集（Caltech256、iWildCam、ImageNet）上进行实验。

**📈 对比分析**

与单模型、路由器、传统级联、后置嵌入、Margin Sampling、FrugalGPT、非加权融合等基线比较，CAUC平均提升约1.9%准确率且减少47%强模型调用；在图像分类上GFLOPs降低57%。

**⚠️ 局限性**

局限在于阈值需要在验证集上估计，对极端分布变化敏感；递归融合仍需对多阶段的温度进行拟合，且对模型互补性过度依赖。

---

## 342. RouteRepair: Instance-Level Failure Diagnosis and Targeted Repair in LLM-Based Automated Heuristic Design for Routing Optimization

**arXiv ID:** 2609.11452 | [PDF](https://arxiv.org/pdf/2609.11452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 343. SWRouter: Similarity-Contractive Window Routing for Multi-Turn Large Language Model Conversations

**arXiv ID:** 2609.11414 | [PDF](https://arxiv.org/pdf/2609.11414v1)

**作者:** Yu Wang `[一作]` (Shanghai Jiao Tong University), Dawei Yin `[通讯]` (Baidu Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多轮对话LLM路由框架SWRouter，兼顾上下文构造与模型选择

**💡 创新点**

将相似度驱动窗口分割与路由器训练相结合，并引入解耦评估指标

**🔧 技术方法**

使用mDeBERTaV3编码器做语义相似度计算，双对比损失训练路由器，结合多模型生成与GPT判分

**📊 数据集**

在ShareGPT、MTBench两大多轮对话数据集上进行训练与测试，同时在PreAlgebra、MBPP、C‑EVAL做OOD评估

**📈 对比分析**

与单模型、Conv‑ID Context、ZOOTER等基线对比，SWRouter在ShareGPT/MTBench上提升评估准确率16.26%（相较最优单模型）并比Conv‑ID Context高8.22%，OOD平均提升2.68%

**⚠️ 局限性**

仅在7B/8B规模模型上验证，未测试更大或专有模型；推理时仍需调用多模型生成，且阈值τ对性能敏感

---

## 344. Heterogeneous Cross-Chain Transaction Tracing for Solana Bridges via Candidate-Set Selective Decision

**arXiv ID:** 2609.11413 | [PDF](https://arxiv.org/pdf/2609.11413v1)

**作者:** Wenjie Dou `[一作]` (North University of China), Yan Qiang `[通讯]` (North University of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文针对 Solana 与 EVM 兼容链之间的异构跨链交易，提出了 SolTracer 方法实现可靠的跨链交易追踪。

**💡 创新点**

创新点在于定义了四种 Solana 绑定跨链交易模式，并通过候选集选择决策与语义事件重构，将异构执行语义映射到统一事件空间，实现可置信的关联与拒绝机制。

**🔧 技术方法**

主要技术包括协议识别、模式特定语义重构、随机森林与逻辑回归相结合的候选分类器以及候选集级别的选择性拒绝判定。

**📊 数据集**

利用真实 EVM‑to‑Solana 交易记录，构建了约 38,317 条桥接记录并标注 7,082 条源‑目标对，进一步生成 212,460 条负样本。

**📈 对比分析**

在封闭世界、开放世界及跨链推广三种场景下，与六个 SOTA 方法比较，SolTracer 在 F1、精确率与召回率上均领先 8‑20%，在开放世界下提升 F1 达 20.16%。

**⚠️ 局限性**

局限性在于仍依赖协议知识库，无法覆盖未知或新兴桥接路由，且对多跳跨链或极度分散的事件碎片存在识别难度。

---

## 345. Morphology-Aware Human Motion Retargeting for Wheeled-Humanoid Loco-Manipulation

**arXiv ID:** 2609.11357 | [PDF](https://arxiv.org/pdf/2609.11357v1)

**作者:** Chenbo Xia `[一作]` (Harbin Institute of Technology), Chao Ye `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一套可重复的运动再定向 pipeline，将多源 SMPL‑X 人类运动数据 retarget 成可执行的三轮无腿 Humanoid（Galaxea R1 Pro）的行走‑操控行为。

**💡 创新点**

核心创新点包括：① 将人体下肢动作用连续腰部替代映射到无腿底盘；② 采用肩部根的层级 arm scaling 解决手臂姿态差异；③ 通过参考 twist 规划 + 三轮 kinematic inversion 将平面基底转化为可执行 wheel 运动；④ 设计 21 维 BaseDecode PPO 控制器，保持低维度的全身跟踪；⑤ 统一的可行性筛选与评估协议。

**🔧 技术方法**

技术栈：GMR differential IK + Mink solver；SMPL‑X canonicalization 与平面基准化；姿态滤波、连续腰部替代、层级 arm scaling；参考 twist 规划与三轮逆运动学；Planner + hysteresis + dynamic limits；BaseDecode PPO + PD 轨迹跟踪；Isaac Lab 物理仿真；COM/ZMP 及自碰撞检测。

**📊 数据集**

使用 AMASS 子集数据集：ACCAD、GRAB、BMLmovi、BMLrub、CMU、KIT、EKUT、Eyes Japan、WEIZMANN、HDM05，覆盖多种动作类别。

**📈 对比分析**

与 GMR 基线对比：在 21 个固定动作上，平均腰部位置误差 0.107 m、角度误差 87°；肩部位置 0.153 m、角度 22.6°；PPO 跟踪实验中 16 个代表动作中 15/16 未跌倒；Planner+contact ablation 产生明显不稳定；直接 wheel 控制导致失败。整体表明该 pipeline 在仿真中可实现稳定的行走‑操控。

**⚠️ 局限性**

局限性：无腿结构无法自我起身；需预先筛选合格动作；wheel 运动不保证无滑移；仅在 Isaac Sim 仿真中验证，未硬件实验；连续腰部映射为任务级近似，非生物学精确；Planner 约束导致速度极限大幅降低。

---

## 346. Beyond the Turing threshold: Productive grammars generate essentially undecidable languages

**arXiv ID:** 2609.11385 | [PDF](https://arxiv.org/pdf/2609.11385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

---

## 347. Safety-aware Skill Adaptation for Reinforcement Learning in Dynamic Environments

**arXiv ID:** 2609.11433 | [PDF](https://arxiv.org/pdf/2609.11433v1)

**作者:** A K M Nadimul Haque `[一作]` (University of Technology Sydney), Teresa Vidal-Calleja `[通讯]` (University of Technology Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对动态环境下的机器人技能自适应，提出基于Gaussian Process与安全指导的Dist-GPRL框架。

**💡 创新点**

创新点：局部窗口更新+GP相关动作变换+HAP安全子空间优先+距离场奖励，兼顾安全与运动一致性。

**🔧 技术方法**

使用Gaussian Process轨迹参数化、Soft Actor-Critic、Hausdorff Approximation Planner、欧氏距离场与余弦相似度正则化等技术。

**📊 数据集**

使用PyBullet仿真中的动态立方体推送与动态条带移除两任务，并在UR5e机器人上进行硬件验证。

**📈 对比分析**

与GPRL、ProMP-RRL、Dist-GPRL-Global等5个基线对比，取得89%/98%成功率、5%/1%碰撞率，显著优于基线。

**⚠️ 局限性**

局限：GP外推不稳、依赖精确传感与距离场估计、缺乏正式安全保证，需进一步研究约束强化学习与噪声不确定性。

---

## 348. Deep Learning-based Bug Triage System

**arXiv ID:** 2609.11420 | [PDF](https://arxiv.org/pdf/2609.11420v1)

**作者:** Sourabh Pal `[一作]` `[通讯]` (INRIA), Sourabh Pal (INRIA)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于RoBERTa‑base的自动化bug triage系统，能够对新提交的bug报告进行组件识别和是否为真正bug的二分类；

**💡 创新点**

创新点在于利用预训练Transformer进行细粒度bug分类，并在短时间内（仅5个epoch）实现高准确率；

**🔧 技术方法**

技术方案包括RoBERTa Transformer fine‑tuning、标准NLP预处理（去除特殊字符、停用词、词形还原）以及GPU端的训练实现；

**📊 数据集**

使用了Bugzilla客户端软件类别的真实bug数据，共69,431条记录，覆盖16个软件产品、325个组件；

**📈 对比分析**

通过90%/10%训练/测试划分，5个epoch训练后，组件分类准确率约0.70，bug识别准确率约0.90；训练损失随epoch下降，验证损失也基本下降，说明模型学习良好但bug识别存在轻微过拟合；

**⚠️ 局限性**

局限性包括数据集仅覆盖客户端软件、训练epoch受限于资源、bug分类存在过拟合、未考虑bug严重性预测及轻量化模型部署等。

---

## 349. X-AuT: Progressive Audio-Encoder Compression for Speech LLMs with Cross-Scale Distillation

**arXiv ID:** 2609.11412 | [PDF](https://arxiv.org/pdf/2609.11412v1)

**作者:** Haojun Zhang `[一作]` (XPeng Inc), Shiyu Huang `[通讯]` (XPeng Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对Qwen3-ASR-0.6B的18层音频编码器进行逐步裁剪，构建16层与14层两种压缩模型。

**💡 创新点**

提出X-AuT框架，将层组合的可恢复性与恢复步骤联立评估，使用短期行为探针选层、交叉尺度表征对齐、按时间调度的学生-策略蒸馏以及LoRA微调。

**🔧 技术方法**

核心技术包括行为驱动的层组合筛选、跨尺度隐藏状态与logit监督、教师-学生混合上下文蒸馏、预训练语言模型冻结、以及LoRA适配器训练。

**📊 数据集**

使用约280k小时的多语言语音数据（AISHELL、CommonVoice、WenetSpeech、GigaSpeech等）做训练，评估则在10个公开基准（AISHELL-1、LibriSpeech、CommonVoice、WenetSpeech等）上进行。

**📈 对比分析**

通过与原始18层模型及自蒸馏、直接裁剪等对照，16层模型宏观误差从5.61%下降至5.27%；14层模型宏观误差为5.75%，参数减少20.7%，在不同基准上表现相对稳定。

**⚠️ 局限性**

实验仅为单次种子，缺乏统计显著性评估；仅压缩音频编码器，未测量完整推理延迟；模型与层组合受限，未验证对其他语音-LLM架构的通用性。

---

## 350. Brain-PACE: A Deep Siamese MRI Framework for Modelling Longitudinal Brain Acceleration

**arXiv ID:** 2609.11378 | [PDF](https://arxiv.org/pdf/2609.11378v1)

**作者:** Samuel Maddox `[一作]` (University of East Anglia), Lifestyle flagship study of ageing `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了Brain-PACE，一种基于Siamese深度网络的直接估计配对T1加权MRI间脑龄加速（P）的方法；

**💡 创新点**

创新点在于将空间注意力、软标签分布学习与Cramér距离损失结合，直接建模纵向变异并给出不确定性估计；

**🔧 技术方法**

使用Siamese卷积网络（SFCN）提取差异特征，加入1×1卷积空间注意力，后接MLP，目标是80个0.1年分箱的分布预测，损失为Cramér距离；

**📊 数据集**

主要数据集包括ADNI、AIBL、OASIS-3、MCSA和WRAP的T1w MRI；

**📈 对比分析**

与传统脑龄PAD及LILAC/LILAC+方法对比，Brain-PACE在内部、外部健康组MAE从0.78降至0.66岁，CRPS、APE、偏差和P比率显著提升；在MCI组中，P更能反映认知、功能受损和后扣带/枕下叶/内嗅皮层的Tau负荷；

**⚠️ 局限性**

局限性包括：仅针对老年人群，缺乏跨生命周期验证；仅使用T1w结构MRI，无法直接捕捉功能或分子变化；在病理状态下不确定性校准仍不足；未证明可预测个体未来衰退或对治疗反应。

---

## 351. Vision Transformer-Based Multi-Level Feature Fusion for Multi-Label Sewer Defect Classification

**arXiv ID:** 2609.11375 | [PDF](https://arxiv.org/pdf/2609.11375v1)

**作者:** Xu Fang `[一作]` (Shenzhen Polytechnic University), Qingquan Li `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了基于层次视觉Transformer的多级特征融合模型Sewer-Transformer-ML，并提出两种轻量级变体Sewer-MobileNet-ML与Sewer-Mobile-TransNet，用于城市污水管道缺陷多标签分类。

**💡 创新点**

创新点在于：1) 直接拼接多级Transformer特征进行融合，提升多标签性能；2) 设计轻量级CNN-Transformer混合结构，并通过滑动窗口多头自注意力实现特征融合；3) 引入类别重要性加权损失处理类别不平衡；4) 在大规模CCTV数据上预训练后跨平台迁移，验证跨域泛化。

**🔧 技术方法**

采用Swin Transformer、MobileNetV3、滑动窗口多头自注意力、分离/通道-空间注意力、类别重要性加权交叉熵和迁移学习等技术。

**📊 数据集**

使用公开的Sewer-ML（约1.3M张多标签图像）和小规模Sewer-Capsule单标签数据集（2,353训练/1,177验证）进行训练与评估。

**📈 对比分析**

与现有CNN和Transformer基准（如ResNet、TResNet、KSSNet等）对比，Sewer-Transformer-ML-Base在Sewer-ML测试集上取得F2_CIW 65.68%、F1_Normal 92.68%排名第一；Sewer-MobileNet-ML以17M参数获得65.73% F2_CIW；Sewer-Mobile-TransNet在Sewer-Capsule上达到96.43%准确率。

**⚠️ 局限性**

局限性包括：缺少公开测试集标签导致无法细致错误分析；轻量级模型性能仅在参数、训练时间上评估，未在实际嵌入设备上验证推理速度、能耗和长期可靠性；数据集主要来自单一城市/设备，跨地区/多设备泛化需进一步验证。

---

## 352. Chypothermia: Clock Freezing for Static Side-channel Attacks

**arXiv ID:** 2609.11442 | [PDF](https://arxiv.org/pdf/2609.11442v1)

**作者:** Fatemeh Khojasteh Dana `[一作]` (Worcester Polytechnic Institute), Shahin Tajik `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了 Chypothermia 冷却攻击，利用低温导致芯片混合信号组件失效，成功停时钟并绕过时钟与温度传感器，随后通过静态侧信道攻击获取密钥。

**💡 创新点**

创新点在于利用低温失效特性无电源干预地停时钟，并与 Chypnosis 联合攻击在可接受温度范围内绕过传统反篡改传感器；同时设计自加热环振荡器来抵御此类攻击。

**🔧 技术方法**

使用的技术包括液氮冷却、FPGA/SoC 平台实验、PLL/时钟生成、计数器监测、延迟链传感器、静态侧信道攻击（LLSI）、自加热环振荡器。

**📊 数据集**

实验数据集为多种 FPGA/SoC（AMD Zynq UltraScale+, AMD Spartan-7, Microchip PolarFire）在不同温度和电压条件下的计数器、PLL 锁定、温度、电压等监测记录。

**📈 对比分析**

通过对比不同温度、电压下的 PLL 锁定、计数器值、温度/电压传感器输出，证明冷却能使时钟停止且不触发传感器；结合 Chypnosis 后仍能绕过检测；在 OpenTitan 上成功绕过密钥零化，表明攻击有效。

**⚠️ 局限性**

局限性包括需物理接触与液氮设备，冷却过程慢；攻击依赖芯片混合信号组件在低温下失效，某些芯片（无 ITD）可能不受影响；温度阈值设置高时仍能被检测；自加热方案带来面积和功耗开销。

---

## 353. Cross-Lingual Clinical Annotation Projection as Constrained Text Generation: A Six-Language Study

**arXiv ID:** 2609.11450 | [PDF](https://arxiv.org/pdf/2609.11450v1)

**作者:** Álvaro Rey-Blanes `[一作]`, Francisco J. Veredas `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究将跨语言临床注释投影重新表述为受限的文本生成任务，在目标语言文本中插入实体标签并恢复字符级边界。

**💡 创新点**

创新点在于将投影视为文本保持的生成任务，结合确定性验证和字符偏移重构，消除了传统的对齐和候选生成。

**🔧 技术方法**

使用大型语言模型（GLM‑5.2、Gemma‑4.31B、Qwen‑3.6:35B）配合定制提示、生成后验证及重构。

**📊 数据集**

使用 MultiClinCorpus 语料库，源语言西班牙语，目标语言英语、捷克语、意大利语、荷兰语、罗马尼亚语、瑞典语，实体类型为疾病、症状、程序。

**📈 对比分析**

与基于候选的机器学习、混合 ML‑LLM 以及之前的 CA26AM+MCAI 基线对比，直接 LLM 投影平均 Strict F1 达到 0.9201，比上一最佳提升 0.0866，字符重叠 F1 超过 0.97，标注 55,416 条实例。

**⚠️ 局限性**

局限性包括仅针对西班牙语源、六种欧洲语言、三种实体类型，未进行专家错误分析，评估仅基于金标准，模型推理成本较高，且在不同源/目标语言或更大规模语料上需进一步验证。

---

## 354. Hologram Representation via Quadratic Phase Gaussian Splatting

**arXiv ID:** 2609.11434 | [PDF](https://arxiv.org/pdf/2609.11434v1)

**作者:** Haolong Wang `[一作]` (Swansea University), Simeng Qiu `[通讯]` (Swansea University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

引入了基于二次相位的复数高斯点(CVQPG)用于光学全息图表示和重建。

**💡 创新点**

在复数高斯基上加入可学习曲率参数与马氏距离调整的二次相位因子，以更好匹配高频细节。

**🔧 技术方法**

采用复数二次相位高斯原语、光学传播（BLASM）、可微光栅化、SSIM/PSNR等损失与Adan优化器。

**📊 数据集**

使用DIV2K与Real Forward-Facing数据集，并用Depth Anything V2生成深度图。

**📈 对比分析**

与平面高斯基准进行等原子/等参数比较，CVQPG平均提升PSNR约+0.19 dB（RGB）/+0.33 dB（灰度），并在中高频段显著降低MSE。

**⚠️ 局限性**

受限于单一曲率参数在多波长下难以同时最优，以及边界截断导致的振铃伪影，且需额外窗口平滑处理。

---

## 355. Gaussian Light Transport

**arXiv ID:** 2609.11430 | [PDF](https://arxiv.org/pdf/2609.11430v1)

**作者:** Patrick Attimont `[一作]` (INRIA Grenoble University), Cyril Soler `[通讯]` (INRIA Grenoble University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于13维高斯混合模型直接求解渲染方程的全局光照方法，获得视角无关的辐射场；

**💡 创新点**

通过在空间、方向、法向、材质属性等13维空间中自适应地拟合高斯核，既实现了物理上合理的光传输，又显著降低了存储成本；

**🔧 技术方法**

使用高斯核分布、Monte Carlo残差最小化、分裂/生成/裁剪策略、Morton码空间分块裁剪以及自定义Taichi/ PyTorch 训练和渲染实现；

**📊 数据集**

在公开的室内场景（Bedroom、Dining Room、Living Room、Staircase、Chair、Rings 等）上进行实验；

**📈 对比分析**

与 Neural Radiosity（含哈希网格和顶点特征编码）对比，训练时间提升10–23倍、渲染速度提升2–4倍，模型尺寸仅1–5.9 MB，FLIP误差在大多数场景中更低；

**⚠️ 局限性**

局限于静态场景、均匀表面采样导致对高频反射学习慢、方向裁剪在训练/非主射线渲染时效果减弱、并且高斯表示天然平滑，难以处理细节强跳跃或完全镜面反射。

---

## 356. Are Caption Metrics Broken? Latency, Deaf and Hard of Hearing User Ratings, and Bias across Technologies

**arXiv ID:** 2609.11408 | [PDF](https://arxiv.org/pdf/2609.11408v1)

**作者:** Bernard Thompson `[一作]` (Gallaudet University), Christian Vogler `[通讯]` (Gallaudet University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过大规模在线问卷调查，收集216名已验证聋人与弱听人群在观看70段美国直播电视片段时，对四种字幕条件（电视原始、电视无延迟、ASR原始、ASR无延迟）的字幕质量与内容理解进行评分，并进一步评估了常用字幕质量指标（WER、ACE2、NER）与受众主观体验的关系。

**💡 创新点**

创新点在于首次将NER等多种字幕质量指标与真实DHH用户的主观评分直接关联，检验指标的技术中立性；系统考察了字幕延迟对用户体验的影响；并提供了大规模、可公开访问的字幕质量与用户评价数据集，为后续研究提供基准。

**🔧 技术方法**

技术包括：自定义视频播放器（支持字幕大小、行数、位置、颜色等自定义）；在线问卷系统Qualtrics；线性混合模型（LMM）和累积链接混合模型（CLMM）对质量与理解评分进行统计分析；使用SCLite计算WER，ACE2基于预训练语言模型，NER由人工评估完成。

**📊 数据集**

数据集：70段美国直播电视视频，共280个字幕条件（四种字幕×每段视频），216名受访者提供4,832条质量与理解评分；字幕文件、计算出的WER、ACE2、NER得分以及参与者匿名信息已公开发布在ACM DL及GitHub。

**📈 对比分析**

比较方法：使用Pearson相关系数评估每种指标与用户质量评分的相关性，并通过配对Cohen's d比较电视与ASR字幕的差异。结果显示：在无延迟条件下，电视与ASR字幕的质量评分相近；延迟显著降低电视字幕评分（平均下降≈1点）；对于电视字幕，WER、ACE2、NER与用户评分呈中等到强相关；而在ASR字幕下相关性弱，仅WERT、ACE2和NER的相关系数约为-0.39、-0.29、-0.38。指标在不同字幕来源下表现不一致，说明缺乏技术中立性。

**⚠️ 局限性**

限制包括：受访者样本偏向女性、聋人及高学历人群，弱听和非英语使用者不足；仅使用单一ASR供应商（AppTek）且延迟固定为2秒；未能区分电视字幕生成方式（手工、重述或ASR）；高成本的人工NER评估与对齐；缺乏跨国、多语言与多平台的通用性验证。

---

## 357. SwarmNxt: Open-source Software-Hardware Platform for Fast and Agile Aerial Swarms

**arXiv ID:** 2609.11382 | [PDF](https://arxiv.org/pdf/2609.11382v1)

**作者:** Charbel Toumieh `[一作]` (Ecole Polytechnique Federale De Lausanne), Dario Floreano `[通讯]` (Ecole Polytechnique Federale De Lausanne)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 SwarmNxt——一套完整的开源硬件与软件框架，支持多架 OmniNxt 无人机的快速组装、并行部署、自动化更新与实时自主飞行。

**💡 创新点**

创新点在于：① 提供原子化装配手册与视频教程，降低硬件上手门槛；② 通过 Ansible 进行全机群软件的并行配置与校准；③ 在 ROS 2 上集成深度估计（S2M2）、映射（HDSM）、自适应 MPC 等先进模块，并实现多机群间的分布式规划与碰撞规避；④ 搭建可视化仪表盘与安全监督节点，提升实验可靠性。

**🔧 技术方法**

使用技术包括：OmniNxt 硬件平台、NVIDIA Orin 计算卡、PX4 飞控、ROS 2 + DDS、Micro‑XRCE‑DDS、Ansible、TartanCalib、S2M2 深度网络、HDSM 规划器、适应性 MPC、Foxglove 可视化、Python/Cpp 实现。

**📊 数据集**

实验使用的主要数据为室内 8 m × 8 m × 4 m 的外部运动捕捉系统提供的地面真值；没有使用公开的深度或视觉数据集，所有深度估计均由 S2M2 在线推理得到。

**📈 对比分析**

通过两组实验验证：6 架无人机在空旷环境中完成“交换”与“巡航”任务，平均跟踪误差 0.075 m，最大通信延迟 30 ms；4 架无人机在障碍环境中进行深度感知与避障，跟踪误差 0.077 m，安全间距 4.9 m，所有模块均满足实时预算（深度 7 Hz、规划 10 Hz、MPC 100 Hz）。

**⚠️ 局限性**

局限性包括：深度估计分辨率有限，难以检测远距离小物体；GPU 资源被 S2M2 占用高达 95%，导致无法并行运行其他感知模块；系统目前依赖外部运动捕捉，缺乏高精度视觉‑惯性里程计；对环境的保守调参可能在更拥挤或户外场景下表现欠佳。

---

## 358. RAMamba-Net: A Reliability-Aware and Mamba-Based Multimodal Fusion Network for Auditory Attention Detection

**arXiv ID:** 2609.11372 | [PDF](https://arxiv.org/pdf/2609.11372v1)

**作者:** Xingyi He `[一作]` (Huazhong University of Science and Technology), Dongrui Wu `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种可靠性感知的 Mamba 基础多模态 AAD 网络 RAMamba-Net，融合 EEG 与 EOG 并通过跨模态注意力和可靠性模块实现更精确的注意力解码。

**💡 创新点**

创新点包括：引入 Mamba 强化的频段感知 EEG 编码器、双分支 EOG 编码器、跨模态注意力以及样本级可靠性估计模块，实现在自适应融合中显式跨模态信息交换与可靠性调节。

**🔧 技术方法**

采用了 Mamba 状态空间网络、Transformer 与注意力机制、频段感知与多头跨模态注意力、可靠性感知权重学习以及联合多任务损失训练等技术。

**📊 数据集**

在两个公开 AAD 数据集 AVGC（音视频）和 DTU（模拟环境）上进行评估。

**📈 对比分析**

与多种单模态基线、简单拼接以及无可靠性模块版本对比，使用准确率、平衡准确率、Macro‑F1、Cohen's Kappa 等指标，RAMamba-Net 在两套数据上分别提升约 5.8% 与 3% 的准确率，且表现更稳健。

**⚠️ 局限性**

仍存在跨主体和跨数据集泛化能力不足的限制，需要进一步研究模型在更大分布漂移下的适应性以及降低计算成本。

---

## 359. Maximal Kolmogorov Complexity in a Hamming Ball

**arXiv ID:** 2609.11362 | [PDF](https://arxiv.org/pdf/2609.11362v1)

**作者:** Alexander Kozachinskiy `[一作]` (Centro Nacional De Inteligencia Artificial), Nikolay Vereshchagin `[通讯]` (Moscow State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究二进制字符串在Hamming球内最大Kolmogorov复杂度的可实现值与曲线形态，并给出点值和全局曲线的可实现性边界。

**💡 创新点**

首次完整表征最大复杂度曲线的极值（最小与最大），并通过构造证明这些极值曲线可实现，提出完整可实现性问题。

**🔧 技术方法**

利用Harper同构性定理、列表解码的覆盖多重性、体积函数V(r)的对数凹性等组合构造与证明技术。

**📊 数据集**

无数据集，纯理论分析。

**📈 对比分析**

无实验对比，理论证明给出O(log n)误差范围，展示极值曲线的存在性与构造方法。

**⚠️ 局限性**

误差为O(log n)，尚未确定其最优性；未能完全刻画所有可实现的复杂度曲线，开放问题仍在。

---

## 360. Beyond Confidence: Stability-Aware Test-Time Adaptation for LLM Reasoning

**arXiv ID:** 2609.11393 | [PDF](https://arxiv.org/pdf/2609.11393v1)

**作者:** Bincheng Gu `[一作]` (Chongqing University), Junliang Yu `[通讯]` (Griffith University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在推理时冻结LLM参数，优化共享前缀以实现无监督的测试时自适应，从而提升推理准确性与生成效率。

**💡 创新点**

首次将局部稳定性作为置信度优化的补充信号，提出随机扰动与尖锐度感知扰动两种策略，确保高置信推理在局部扰动下保持稳定。

**🔧 技术方法**

利用预测熵做置信度指标，结合随机扰动与尖锐度感知扰动的正则化，使用梯度下降优化共享前缀，并与Chain-of-Thought等连续变量技术协同。

**📊 数据集**

在多种数学与科学推理基准上评估，包括MATH‑500、AMC23、AIME24/25、Minerva Math、GPQA Diamond等六大数据集。

**📈 对比分析**

与零样本CoT、SLOT、LatentSeek、LTPO、TTSV等基线对比，TASCO在Qwen、LLaMA、DeepSeek等模型上平均提升10–18个百分点，且生成长度缩短约25–30%。

**⚠️ 局限性**

局限于前缀级别的改进，尚未充分验证在更大规模模型或非推理任务中的泛化与对抗鲁棒性。

---

## 361. PATTON: Enabling Commodity PIM for Production LLM Serving

**arXiv ID:** 2609.11392 | [PDF](https://arxiv.org/pdf/2609.11392v1)

**作者:** Hangyeol Kim `[一作]` (KAIST), Joo-Young Kim `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了PATTON运行时，使生产LLM服务引擎能够动态管理KV缓存块并将其映射到商品PIM内存，同时生成对应PIM指令以加速查询‑键和分数‑值计算；

**💡 创新点**

提出了层次化粒度分配（Order‑1、Order‑2）与Commit Zone机制，平衡GEMV效率、KV写入效率与容量利用率，且无需改动PIM处理单元，即可支持前缀共享、缓存与回收；

**🔧 技术方法**

利用AiM GDDR6 PIM指令集、块级KV缓存管理、层次粒度分配、Commit Zone、物理映射元数据与vLLM接口；

**📊 数据集**

使用ShareGPT对话数据集，并在Llama3‑8B、Qwen3‑8B、OPT‑6.7B、Qwen1.5‑7B等模型上进行评估；

**📈 对比分析**

与NeuPIMs、PIMphony、AttAcc、BlockPIM等基线在相同PIM配置和请求工作负载下对比，PATTON平均提升1.95×速度、4.83×能效，并在不同模型上实现最高GEMV效率与最低预填充重算；

**⚠️ 局限性**

在极端容量压力下KV缓存命中率仍会下降；对PIM硬件的兼容性依赖现有指令集，无法利用更高级的加速指令。

---

## 362. Agent-Integrated Software: Interaction Contracts and Continuous Assurance

**arXiv ID:** 2609.11381 | [PDF](https://arxiv.org/pdf/2609.11381v1)

**作者:** Shengcheng Yu `[一作]` (Technical University of Munich), Zhenyu Chen `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 Agent‑Integrated Software（AIS）模式与 Intent‑Level Interaction Abstraction（IIA）框架，阐述了在传统应用与嵌入式智能代理之间保持交互与执行一致性的理论与实践方法。

**💡 创新点**

创新点在于：① 将任务级交互（意图、修订、授权、控制、证据）与应用执行建立显式对应关系；② 通过交互契约（interaction contract）与持续保障（continuous assurance）实现跨组件的可靠性约束；③ 提出了统一的开放转换系统模型和抽象层，支持多域（协作、电子表格、IDE、服务控制台）的通用分析与验证。

**🔧 技术方法**

主要技术包括：基于状态机的开放转换系统建模、任务级抽象模型（IIA）与执行层（AIS）之间的映射（π、α），交互契约语义约束（pre、step、inv、post、dep），以及基于证明与证据的持续保障框架。

**📊 数据集**

本工作为概念性研究，未使用具体数据集；所示的示例（会议材料披露、电子表格更新、IDE重构、服务台退款）均为人工设计的案例场景。

**📈 对比分析**

由于缺乏实现原型与实验数据，论文未进行方法比较或性能评估；重点在于理论验证与案例演示，建议未来通过实现工具链并在真实应用中量化交互契约覆盖率、错误检测率、开发成本等指标。

**⚠️ 局限性**

局限性包括：① 需要开发者手工构建/维护任务抽象与契约，工作量不可忽视；② 依赖对应用对象、权限、事务等底层细节的准确识别，若缺失会导致契约失效；③ 在高度动态或分布式环境中，持续保障的证据收集与版本依赖管理会产生额外运行时开销；④ 本框架并未覆盖可用性、可解释性与公平性等人机交互细节，需要进一步研究。

---

## 363. Local Robustness Quantification for Naive Bayes Classifiers and Generative Forests: a General Approach

**arXiv ID:** 2609.11366 | [PDF](https://arxiv.org/pdf/2609.11366v1)

**作者:** Adrián Detavernier `[一作]` (Ghent University), Jasper De Bock `[通讯]` (Ghent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了针对生成式分类器（朴素贝叶斯和生成式森林）预测结果的鲁棒性量化方法，并将其推广到ε-污染、总变差距离和χ²散度球等多种扰动形式；

**💡 创新点**

创新点在于将鲁棒性量化从仅限ε-污染扩展到任意局部参数扰动，给出三类扰动下的精确计算公式，并通过优化求解实现可行性；

**🔧 技术方法**

核心技术包括不确定概率理论、可信度上界下界计算、优化求根（root-finding）以及生成式概率图模型的结构化推理；

**📊 数据集**

实验使用了11个公开数据集，对朴素贝叶斯和生成式森林模型进行评估；

**📈 对比分析**

与六种不确定性量化指标（u_marg, u_max, u_H, u_t, u_a, u_e）在准确率-拒绝率曲线上的表现对比，结果显示新鲁棒性指标与UQ指标相当甚至优于部分指标，尤其在高拒绝率或低拒绝率场景中表现突出；

**⚠️ 局限性**

局限性包括鲁棒性指标在不同模型、数据集及拒绝率下的性能差异不大、对扰动选取的依赖性强，以及在实际应用中需要进一步验证其对分布漂移或噪声环境的适用性。

---

## 364. VikingRAG: Accurate and Token-efficient Retrieval-augmented Generation over Structured Documents

**arXiv ID:** 2609.11390 | [PDF](https://arxiv.org/pdf/2609.11390v1)

**作者:** Peiyuan Gao `[一作]` (Renmin University of China), Wei Lu `[通讯]` (Renmin University of China)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出VikingRAG系统，利用层级语义存储与多轮证据缺口驱动检索实现结构化文档的检索增强生成，显著降低token消耗；

**💡 创新点**

创新点在于：①通过URI保持层级语义索引实现提示解耦的结构访问；②采用证据缺口驱动的多轮检索；③引入经验边以重用历史检索路径；④设计自适应升级策略，仅在必要时触发多轮检索；

**🔧 技术方法**

使用LLM工具调用（Search、List、Grep、Read）、向量嵌入、URI层级索引、经验边构造与查询匹配、以及基于约束的证据充分性评估；

**📊 数据集**

评测数据集包括六个真实结构化文档集：VersionQA、SyllabusQA、QASPER、HotpotQA、LegalBench‑cuad、FinanceBench；

**📈 对比分析**

与八种主流RAG基线（MoDora、BookRAG、DeepRead、KohakuRAG、LightRAG、HippoRAG‑2、SQL‑AgenticRAG、NaiveRAG）对比，VikingRAG在保持或超越准确率的同时，token使用仅为最高准确率基线的11.6%–51.9%，加入经验边和自适应升级后降至5.1%–32.5%，且延迟保持可接受；

**⚠️ 局限性**

局限在于对历史检索覆盖率与相似度阈值敏感，误判的非升级率仍略高；插入阶段token成本较高；在极大目录或高度细粒度检索场景下可能仍需改进；

---

## 365. Portable Semantics, Private Dialects: Reuse and Negative Transfer in Latent Communication Between Language-Model Cells

**arXiv ID:** 2609.11365 | [PDF](https://arxiv.org/pdf/2609.11365v1)

**作者:** Narcis Marincat `[一作]` `[通讯]`, Narcis Marincat

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开展了对多模型通信接口的封闭式跨模型互操作性审计、零射击失败定位以及迁移价值的实验，探究受限可见性与全局可见性训练对接口可复用性的影响。

**💡 创新点**

创新点在于首次将封闭的跨模型互操作性审计与匹配的组件重初始化因子结合，揭示了初始化绑定的语义一致性与接口负迁移的机制。

**🔧 技术方法**

采用预训练的 Qwen2.5-0.5B‑Instruct、低秩 LoRA、连续包通信、Procrustes 线性/非线性映射等技术对模型进行对齐与评估。

**📊 数据集**

使用了 17 维整数运算任务（Family A/D）作为数据集，涵盖不同初始化、数据顺序、隐藏状态和多种源-目标配对。

**📈 对比分析**

通过对六个受限模型进行 30 对有向互植、零射击失败定位以及全局模型的接口重初始化实验，比较了准确率和学习曲线，发现同初始化下可实现原始互通，跨初始化失败；全局模型继承接口导致显著负迁移，重初始化后准确率提升至 0.857。

**⚠️ 局限性**

实验局限于仅 17 维状态空间、仅两种初始化、未检验更丰富的对齐方式或更大状态空间，且全局迁移结果仅来自单一全局模型，无法推广到更广泛设置。

---

## 366. Prevalence Determines Precision:Silent Contamination in Detector-Defined Datasets

**arXiv ID:** 2609.11449 | [PDF](https://arxiv.org/pdf/2609.11449v1)

**作者:** Jia Huang `[一作]` (Peking University), Yangjun Ou `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在实盘金融事件检测中，对不同候选池下同一检测器产生的事件数据集进行端到端测量，揭示了预期率对数据集精准度的决定作用；

**💡 创新点**

创新点在于：①实现了可回溯“幻影事件”测量，验证了贝叶斯预期率公式的准确性；②给出受污染数据集响应曲线的精确凸组合分解，首次表明幻影事件是具有自身形状的第二信号；③揭示污染对估计方向的依赖性，显示同一数据集在不同统计量下污染会导致信号衰减或放大；④指出常用的 per‑item 正规化导致无穷期望的估计器，并给出实际差异。

**🔧 技术方法**

主要技术包括：阈值检测器、贝叶斯预期率推导、响应曲线分解公式、选择条件下的幻影空洞构造、模拟检验和统计学误差分析。

**📊 数据集**

使用 E‑mini 标普 500 期货 2010‑2026 年的 1 秒价差数据，构建了官方指数（CPI、NFP、FOMC 事件）与检测器定义的事件集，并在三种候选池（会议日历、每月首个星期五、每月 6–16 号工作日）上检验。

**📈 对比分析**

通过将检测器在高预期率池（FOMC、NFP）上的结果与低预期率池（CPI）对比，发现若不采用贝叶斯公式，精度转移误差高达 422%；而使用贝叶斯公式误差仅 3.3%；此外，在不同估计器下，同一数据集的响应曲线会出现放大或衰减。

**⚠️ 局限性**

局限性包括：仅测试了阈值检测器；仅涉及一种金融资产与一种事件指数；候选池预期率的差异是针对特定市场结构；选择条件下的幻影空洞仅复现曲线形状，未能完全解释水平差异；所提出的结论主要针对相对比较，缺乏绝对效能声明。

---

## 367. Wavering Oracles: Selective Updating and Correlated Failures in LLMs and Their Implications for Scientific Workflows

**arXiv ID:** 2609.11428 | [PDF](https://arxiv.org/pdf/2609.11428v1)

**作者:** Xiaoshn Nee `[一作]` (Independent researchers), Xiaomin Ni `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估多模型科学工作流中的选择性更新、错误多样性和裁决质量，通过十个模型在三种压力干预（怀疑、权威、错误建议）和正确建议下的系统实验；

**💡 创新点**

提出基于错误翻转率与正确更新率的二维平面与选择性得分的定量评估框架，并在本地和公开日志两套实验中对模型行为进行细致对比；

**🔧 技术方法**

利用微平均、bootstrap及stem聚类置信区间进行统计，构建留一stem‑family‑out可靠性选择器，使用LLM推理（Qwen3‑4B、Gemma3‑4B、SmolLM3‑3B）和公开API日志进行实验；

**📊 数据集**

使用600道四选一英文测验题，包含272个归一化stem，分布于八个领域和三难度层次；以及对应的公开模型日志（GPT‑4o‑mini、Claude‑系列、Gemini、Llama 等）和本地三模型的完整轨迹；

**📈 对比分析**

通过选择性得分、PRA、错误相关性等指标对模型进行排名；最佳单模型达95.3%准确率，七模型众数88.6%，oracle 99.8%，留一选择器96.2%，恢复了众数到oracle差距的67.7%；局部模型中Qwen3‑4B选择性45.6，Gemma3‑4B选择性-14.1，SmolLM3‑3B选择性0；

**⚠️ 局限性**

实验仅限多项选择题，缺乏开放式/论证任务；模型日志解析方式与版本差异可能影响结果；错误相关性受题目集合限制，难以保证跨域推广；选择器依赖先验校准标签，实际部署需进一步验证。

---

## 368. LLMs as Post-hoc Auditors of Physiological Plausibility in Symbolic Regression: A Clinician-Evaluated Case Study

**arXiv ID:** 2609.11431 | [PDF](https://arxiv.org/pdf/2609.11431v1)

**作者:** Jorge López-Varela `[一作]` (Universidad Complutense de Madrid), Oscar Garnica `[通讯]` (Universidad Complutense de Madrid)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文研究了利用大型语言模型（LLM）对遗传编程生成的符号回归模型进行后置审计，评估其可解释性和生理合理性，并与临床医生评估结果对比。

**💡 创新点**

创新点在于将 LLM 作为模型解释和比较的辅助工具，而非传统的单个模型解释，并将 LLM 输出与专家评估进行系统对比，验证其在医学符号回归工作流中的可行性。

**🔧 技术方法**

技术方法包括：语法引导遗传编程（GE、CFG‑GP、DSGE）生成符号回归方程，LLM（Gemma3、DeepSeek‑R1、GPT‑5.1 Thinking）进行解释、排名，Prompt Engineering 与推理 Prompt 设计，以及基于医生的 Likert 量表进行评估。

**📊 数据集**

使用的训练与评估数据集为美国 CDC 的 NHANES 2017‑2018 年体脂百分比数据，用于模型预测和验证。

**📈 对比分析**

通过对四个符号回归模型分别进行 LLM 解释与比较，得到模型的可解释性、可行性、简洁性、临床适用性等四维排名；随后三名临床医生对 LLM 生成的解释与排名进行 1‑5 量表评分。结果显示，LLM 生成的排名比单个模型解释更受医生欢迎，但 LLM 仍可能给出不符合生理逻辑的解释；整体性能在可解释性与临床合理性方面表现可接受。

**⚠️ 局限性**

主要局限包括：LLM 可能产生幻觉或生理错误，需要专家监督；不同 LLM 在同一任务上的排名不一致，推理模型不够稳定；评估者仅有三名医生，样本不足；缺乏对模型生理合理性的自动化验证。

---

## 369. Characterizing Bluesky Content Moderation Service: From Automation of Service to Landscape of Harms

**arXiv ID:** 2609.11373 | [PDF](https://arxiv.org/pdf/2609.11373v1)

**作者:** Pushpdeep Singh `[一作]` (Max Planck Institute for Software Systems), Abhisek Dash `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

审计了Bluesky默认内容审核系统BMS的机制、效能与检测的危害类型，首次实现了对真实部署的内容审核系统的可验证大规模审计。

**💡 创新点**

利用去中心化平台公开的审核日志，实现对部署系统的实证评估，并通过聚类反演标签定义，揭示审核实践与政策间的差异。

**🔧 技术方法**

采用标签延迟分析推断自动化程度、人工标注评估精准率与召回率、Hive+Automod规则链复现自动化流程、基于视觉-语言模型+UMAP+HDBSCAN的多模态聚类。

**📊 数据集**

使用2025年BMS公开的1066万条标签记录（含10.6M帖子标签），随机采样1000条标注集与1000条火炬流样本，并调用Hive API进行规则模拟。

**📈 对比分析**

通过人工标注与BMS标签对比得到精度0.837、召回率0.222；自动化标签召回率0.6；聚类结果由LLM生成标签，平均Likert 4.58，Krippendorff α0.686，表明模型表现优异。

**⚠️ 局限性**

仅覆盖官方BMS，未审计第三方标签器；标签延迟代理自动化程度存在偏差；数据收集可能漏掉已删除帖子；未评估审核对用户行为与长期效应的影响。

---

## 370. From Queries to Narratives: Cultural Heritage Data Stories for Knowledge Graph Exploration and Quality Assessment

**arXiv ID:** 2609.11403 | [PDF](https://arxiv.org/pdf/2609.11403v1)

**作者:** Tabea Tietz `[一作]` (FIZ Karlsruhe – Leibniz Institute for Information Infrastructure), Harald Sack `[通讯]` (FIZ Karlsruhe – Leibniz Institute for Information Infrastructure)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并实现了基于知识图谱的数据故事框架，用于文化遗产数据的可视化、探索与质量评估；

**💡 创新点**

首次将可执行 SPARQL 查询嵌入叙事文档，并结合 AI 助手与可视化构建器实现交互式数据故事创作；

**🔧 技术方法**

使用 SHMARQL、Sparnatural、LLM 驱动的 AI 助手、Plotly、Oxigraph 等技术；

**📊 数据集**

以 NFDI4Culture 知识图谱（约1.5亿三元组）和 GeMeA KG 为数据集；

**📈 对比分析**

通过研讨会与课堂实验与现有工具对比，AI 助手降低查询构建门槛但在聚合查询准确度较低，整体性能尚未进行定量评估；

**⚠️ 局限性**

AI 助手回答不够可靠、透明度不足、受成本限制、缺乏正式量化评估，以及对大规模 KG 的支持仍待改进。

---

## 371. Learn the Solid, Not the File: Canonical Inputs for Neural Networks on CAD Boundary Representations

**arXiv ID:** 2609.11573 | [PDF](https://arxiv.org/pdf/2609.11573v1)

**作者:** Heinrich Jiang `[一作]` (Storygold Ai), Jennifer Jang `[通讯]` (Storygold Ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种名为 Canonical Region Graph 的B-rep输入表示方法，通过合并面、构造从实体本身导出的坐标系和特征，实现对同一实体的B-rep变体（切分、重表达、刚性运动等）的不变性。

**💡 创新点**

创新点在于：1）将B-rep中任意切割的面合并为原始几何面的完整区域；2）使用实体质心+主轴构造统一坐标系，保证对旋转和平移的严格不变性；3）所有特征均在该坐标系下计算，剔除文件特定的参数化信息，从而获得理论上的不变性保证。

**🔧 技术方法**

使用图神经网络（8层边缘条件多头注意力）处理包含115维节点特征和10维边特征的 Canonical Region Graph；特征包括面积、曲率、凸性、边界长度等；实现过程涉及解析B-rep、提取底层无裁剪曲面、计算主轴框架、统一尺度。

**📊 数据集**

实验数据集：MFInstSeg、MFCAD++、CADSynth 三个公开CAD基准；自行生成的自动扰动（切分、对角切分、旋转、NURBS重表达、组合）以及在FreeCAD中由两名专家分别重建的 25 个零件；此外使用 3000 个Fusion 360 Gallery 实际零件做检索基准。

**📈 对比分析**

与 UV‑Net、AAGNet、BRepNet、DGCNN 等主流B‑rep编码器以及点云方法 DGCNN 进行比较。实验显示：在原始任务上 Canonical Region Graph 与最强基线相当；在所有扰动任务中其性能几乎不下降，甚至在预测跳变和检索任务上大幅优于基线，显示出更高的鲁棒性和稳定性。

**⚠️ 局限性**

局限性：由于完全抛弃了B‑rep的切分历史信息，无法利用建模操作记录，对需要基于建模步骤标注的任务（如Fusion 360 Gallery 的“建模操作”标签）会导致性能略逊；在极对称或结构复杂的零件中，主轴框架的判定可能需要额外处理。

---

## 372. Enabling Knowledge Graph Understanding at Scale with the EXplore Your Graphs ENgine (EXYGEN)

**arXiv ID:** 2609.11569 | [PDF](https://arxiv.org/pdf/2609.11569v1)

**作者:** Harshdeep Singh `[一作]` (Odoma Ltd.), Matteo Romanello `[通讯]` (Odoma Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 EXYGEN 框架，提供基于知识图谱结构化元数据（VoID、ShEx）与检索增强生成（RAG）的无监督文本到 SPARQL 的问答方法，并通过谓词覆盖感知的并行采样技术实现了对百万至十亿三元组图的高效元数据生成。

**💡 创新点**

创新点在于（1）利用自动提取的 KG 结构化上下文（VoID、ShEx、检索三元组、示例问答）替代对 LLM 的细粒度微调，显著提升大模型的可执行率与准确率；（2）提出了谓词覆盖感知的并行采样策略，在保持结构多样性的同时大幅减少元数据生成时间（高达 80× 以上）。

**🔧 技术方法**

技术包括：语义检索增强生成（RAG）+多模态检索索引、VoID 与 ShEx 的自动生成与转换、并行随机游走+谓词优先级或度数/相似度剪枝的采样算法、LLM（Qwen3、DeepSeek 等）在 128k 上下文窗口下的生成与执行。

**📊 数据集**

使用了 SciQA（基于 ORKG 的 2,565 个问答对）作为 QA 评测基准，以及 OpenCitations Meta、GESIS、ORKG 三大规模知识图谱用于评估元数据生成与采样效果。

**📈 对比分析**

在不进行微调的情况下，DeepSeek-V3.1 结合 ShEx、检索三元组与示例问答可达 41.9% 的执行结果 Exact Match（EMRelEx），显著优于单一上下文或仅使用 ShEx 的 0%；与传统微调模型 FIRESPARQL 的 EMRel=0.85 仍有差距，但已将无监督方法性能提升至前沿；采样策略在 OCM 上实现 80× 速度提升，GESIS 也达 80×，展示了可扩展性。

**⚠️ 局限性**

主要局限包括：对示例问答的依赖导致在缺乏已标注问答的 KG 上效果未知；检索基于标签相似度，若 KG 的 URI 不具可读性则检索质量下降；采样优化仅考虑谓词覆盖与三元组损失，未直接评估其对最终 SPARQL 生成精度的影响；在复杂细粒度 Ontology（如 ORKG）下仍需采样，完整元数据生成不可行。

---

## 373. Complex-Text Robustness Evaluation and Failure Diagnosis for Low-Resource Multilingual Text-to-Speech

**arXiv ID:** 2609.11545 | [PDF](https://arxiv.org/pdf/2609.11545v1)

**作者:** Tianlun Zuo `[一作]` (Northwestern Polytechnical University), Lei Xie `[通讯]` (Northwestern Polytechnical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套针对低资源多语言TTS的复杂文本鲁棒性诊断框架，评估内容一致性、语言一致性和生成稳定性。

**💡 创新点**

创新点在于：①把复杂文本视为评估维度并构建六类测试集；②引入无监督的Text Risk Score（TRS）作为前置风险指示；③系统性比较三大多语言TTS模型的失败模式。

**🔧 技术方法**

采用多模态自动评估技术：ASR（Whisper）计算字符错误率；语音语言识别评估语言一致性；基于持续时间比值计算异常率；TRS基于文本特征的规则加权。

**📊 数据集**

使用四种低资源语言（泰语、越南语、斯瓦希里语、印尼语）的240条测试文本，共分六类（普通句、数字/日期、命名实体、长句、混合脚本、标点），共960条。

**📈 对比分析**

在OmniVoice、VoxCPM2、MMS‑TTS三模型上做对比，发现复杂文本能揭示模型在数字、命名实体、代码混合等方面的系统性缺陷；不同模型在CER、LID-Acc和DAR指标上表现各异，表明单一指标难以完整描述鲁棒性。

**⚠️ 局限性**

局限性包括：仅评估四种语言；主要依赖自动指标，缺乏细粒度发音和主观自然度评估；TRS为规则型，未包含可训练的风险预测模型。

---

## 374. Lightweight LiDAR-Based Cone Detection Framework Using Random Forest for Formula Student Driverless

**arXiv ID:** 2609.11527 | [PDF](https://arxiv.org/pdf/2609.11527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 375. ChurnBench: A Drift-Aware Benchmark Demonstrating That Refresh Scheduling, Not Cache Age, Governs Staleness in Agentic AI

**arXiv ID:** 2609.11515 | [PDF](https://arxiv.org/pdf/2609.11515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 376. Design Reflections on Transition to LLM-Aided Novel Visualizations

**arXiv ID:** 2609.11503 | [PDF](https://arxiv.org/pdf/2609.11503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 377. TimelyRAG: Semantic-Temporal Hybrid Retrieval for Time-Critical Question Answering in Overlapping-Evolving Documents

**arXiv ID:** 2609.11572 | [PDF](https://arxiv.org/pdf/2609.11572v1)

**作者:** Youngeun Nam `[一作]` (KAIST), Byung Suk Lee `[通讯]` (University of Vermont)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种检索器无关的时序重排序框架，融合段落级有效期信息以解决重叠演变法规文档中的时效性问答问题，并构建了首个面向重叠演变法规的合成基准数据集。

**💡 创新点**

创新点在于：①将查询时效性与语义相似度结合的可查询自适应权重机制；②在段落层面刻画文档的有效时间区间，从而区分语义相近但时效不同的版本；③提供专门针对重叠演变环境的法规基准，显著提升检索挑战难度。

**🔧 技术方法**

技术包括两阶段检索管线：第一阶段使用稀疏/密集检索器（BM25、BGE‑M3、NV‑Embed‑V2、BGE‑Gemma2）获取候选集；第二阶段计算语义得分与基于插入时间/事件时间的时间距离，再通过可查询自适应权重（α(Q)）混合得到最终排名；同时采用LLM（Llama‑3.3‑70B）生成问答对与多模型验证。

**📊 数据集**

数据集：TimelyRAG（合成的法规问答基准，包含法律、大学、公司政策、服务条款四个领域），并与 TS‑Retriever（离散演变）、FiQA‑BEIR（非时序）等公开基准进行对比；数据通过 Llama‑3.3‑70B 自动生成，验证使用 GPT‑4o、Claude‑3.7‑Sonnet、Qwen2.5‑72B。

**📈 对比分析**

对比方法：在多种检索器上加上时间重排序，并与时间查询改写、规则式有效期过滤、LLM 重排序、黄金时间过滤等基线进行比较；实验显示在 TimelyRAG 上可达 +28.6% nDCG@10、+19.1% Hit@10，且平均查询延迟仅增加 2~3 ms。

**⚠️ 局限性**

局限性：①依赖第一阶段检索器的候选覆盖；②时间距离函数过于简单，无法处理复杂时序逻辑和多事件依赖；③基准为合成数据，缺乏真实多样性；④对缺失或模糊时间信号的处理仍有限。

---

## 378. Entwine: Coordinating Tiled Computation and Fine-Grained Communication across GPUs

**arXiv ID:** 2609.11562 | [PDF](https://arxiv.org/pdf/2609.11562v1)

**作者:** Kai Ma `[一作]` (State Key Laboratory of Cyberspace Security Defense), Kefan Ruan `[通讯]` (State Key Laboratory of Cyberspace Security Defense)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对张量并行LLM工作负载，作者提出了 Entwine，协调块级计算顺序、细粒度通信以及共享SM资源的分配，以最大限度减少 GEMM–ReduceScatter 的总完成时间。

**💡 创新点**

创新点在于：①将块级计算顺序交错（interleaved），使通信数据以更规律的节奏产生；②为每个输出块单独进行通信和归约，避免因等待不相关块而产生的延迟；③通过通信预算（communication budget）在共享 SM 池中动态平衡计算与通信的资源占用；④在离线配置阶段统一优化计算和通信的组合，直接最小化端到端延迟。

**🔧 技术方法**

技术细节包括：使用 CUTLASS 实现的行并行 GEMM，CUDA 线程块 swizzle 实现块级交错顺序；自定义 128×128 归约核，利用系统范围的释放/获取存储保证数据依赖；共享 SM 资源并使用 stream 级优先级；利用 CUDA Graphs 缓解主机启动开销；离线性能分析选取最优计算块形状、通信预算和线程数。

**📊 数据集**

数据集与工作负载：
- 主实验集：15 个基于 Llama‑3 70B 与 Llama‑3.1 405B 的张量并行 GEMM，覆盖 6 倍输出尺寸和 4 倍减少维度 K。
- 模型导出集：25 个工作负载，包含 Llama‑3 8B/70B 与 Qwen2.5‑72B 的注意力与 MLP 输出，进一步覆盖更短的计算窗口。
- 层级实验：在 Llama‑3 70B 的注意力输出层（GEMM + ReduceScatter + 残差 + RMSNorm）上验证。

**📈 对比分析**

比较方法与性能：
- 基线：顺序执行、cuBLAS+NCCL、FlashOverlap、Async‑TP、FLUX。
- 结果：Entwine 对 15 个主实验集实现 1.232× 的几何平均加速（最高 1.433×），相对于最佳基线的几何平均提升 3.1–9.8%。
- 层级实验显示，在不同序列长度下，Entwine 对 cuBLAS+NCCL、FlashOverlap、Async‑TP 的层级延迟均有明显下降，且在 8K/16K 序列长度上对 FLUX 的层级加速分别为 6.2% 与 9.8%。

**⚠️ 局限性**

局限性：
1. 仅在 NVIDIA SM80 GPU 与单一 NVLink 域内测试，未验证多节点或其他架构的可扩展性。 
2. 只针对行并行 GEMM；对列并行或非方阵矩阵的适用性尚未探究。 
3. 需要离线配置和手动调优，可能对动态工作负载或在线环境不友好。 
4. 交错计算顺序可能降低数据重用，导致在小规模张量上出现负收益。 
5. 目前仅利用 NCCL 进行通信，未考虑基于 NVLink DMA 或第三方通信库的差异。 
6. 资源共享模型假设计算与通信的 SM 需求相似，未考虑极端内存/带宽瓶颈的情况。

---

## 379. Characterizing Job Power Elasticity for Power-Flexible AI Training

**arXiv ID:** 2609.11542 | [PDF](https://arxiv.org/pdf/2609.11542v1)

**作者:** Philip Colangelo `[一作]` (Emerald AI), Varun Sivaram `[通讯]` (Emerald AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统量化大型语言模型训练任务在GPU功率限制下的吞吐量弹性，提出并计算了“功率灵活性指数”（PFI），并用H200 GPU进行131次实验，覆盖多种模型架构、训练任务和GPU规模；

**💡 创新点**

创新点在于①提出可度量并比较不同训练任务功率弹性的PFI指标；②发现内存带宽与功率限制的机制，利用GPU运行时指标（如DRAM拷贝产物）在线预测PFI；③用PFI指导功率分配策略，在模拟工作负载下显著提升集群整体吞吐量；

**🔧 技术方法**

技术手段包括：GPU功率抑制（DVFS/功率上限）、DCGM监控、线性回归预测模型、仿真功率分配算法（等权、PFI加权等），以及统计分析（Dunn检验、Spearman相关）来评估指标；

**📊 数据集**

实验数据集为4种开源大模型（gpt-oss-20b、Qwen3-30B-A3B、Qwen3-32B、Llama-3.1-70B）在8/16/32 GPU配置下进行预训练与LoRA微调，共131次训练跑（含24次验证跑）；

**📈 对比分析**

比较方法为多种功率分配策略（Oracle、Equal Weight、MoE FT-weighted、PFI-aware、DRAM-copy-product）在随机和生产级工作负载下的吞吐量损失评估；实验显示PFI-aware策略在30%功率削减时比等权分配恢复约1.5k tokens/s，达到oracle约63%的性能；

**⚠️ 局限性**

局限性包括仅在NVIDIA H200 GPU上验证、只考虑GPU功率（未计CPU/网络等），实验周期短（30min）、样本量有限（25个PFI），未在真实集群闭环验证，也未覆盖RLHF或推理任务，故结果可能在其他硬件或任务上泛化受限。

---

## 380. FreeFlow: A Bias-free Hierarchical Transformer for Optical Flow Estimation

**arXiv ID:** 2609.11486 | [PDF](https://arxiv.org/pdf/2609.11486v1)

**作者:** Vladislav Bargatin `[一作]` (Lomonosov Moscow State University), Dmitriy Vatolin `[通讯]` (Moscow State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无传统光流偏置的层次变压器 FreeFlow，用单一编码‑解码架构实现高分辨率光流估计。

**💡 创新点**

创新点在于完全去除光流专属模块（相关卷积、特征扭曲、迭代细化等），只保留窗口、偏移窗口及低分辨率全局注意力，既实现了高精度也保持了内存效率，并支持模型按规模自然扩展。

**🔧 技术方法**

使用的技术包括：多层窗口注意力、Shifted‑Window 注意力、下采样‑上采样全局注意力、RoPE 位置编码、交叉注意力解码、简易三层预测头，以及跨视图填补预训练和 Mixture‑of‑Laplace 损失。

**📊 数据集**

训练与评估使用了 TartanAir、Things、Sintel、KITTI‑2015、HD1K、MegaDepth、3DStreetView 等数据集，最终在 Sintel、KITTI‑2015 与 Spring 三大公开基准上进行评测。

**📈 对比分析**

与现有方法（PWC‑Net、RAFT、WAFT、GeoViT、CroCo、Win‑Win、MEMFOF 等）对比，FreeFlow‑L 在 Sintel Clean 0.68 EPE、Final 1.48、KITTI‑2015 Fl‑all 3.23、Spring 1px 3.192 等指标均夺得第一，显著优于同类两帧模型，且在 1080p 推理时保持低内存（≈3 GB）。

**⚠️ 局限性**

限制包括：仍需大量算力与数据进行预训练与微调，缺乏多帧或视差处理能力，对极端遮挡或极大视差仍可能存在误差，且目前仅在标准光流任务上表现突出，尚未验证在更广泛的稠密对应任务上的通用性。

---

## 381. BridgeMatch: Conditional Transport Bridges in Matching Matrix Space for 3D Deformable Registration

**arXiv ID:** 2609.11472 | [PDF](https://arxiv.org/pdf/2609.11472v1)

**作者:** Qianliang Wu `[一作]` (Nantong University), Yaqing Ding `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种两阶段粗到细的匹配矩阵生成器，先用扩散得到粗匹配，再升维并通过条件传输桥细化到高分辨率；

**💡 创新点**

创新点在于在匹配矩阵空间中构造高分辨率的条件传输桥，保持完整匹配候选并使用端点参数化的ODE或Brownian桥来细化，无Top‑K剪枝；

**🔧 技术方法**

使用扩散模型、条件流匹配（CFM）ODE、Brownian桥SDE、时间条件Transformer、Sinkhorn正则化以及软Procrustes；

**📊 数据集**

主要在4DMatch、4DLoMatch、CAPE、DeepDeform等四个点云配准数据集上进行评估；

**📈 对比分析**

与Diff‑Reg、RoITr、GeoTransformer等方法对比，均能在NFMR/IR、EPE、AccS/AccR等指标上取得更优成绩，尤其在低重叠和跨数据集零样本情形下提升显著；

**⚠️ 局限性**

局限在于仍依赖固定的图形配准解算器（GraphSCNet）来评估下游效果，且对大规模点云的运算开销尚未彻底解决；

---

## 382. CAP: Continuously Adaptive Perception-Blind Humanoid Locomotion via Learned Denoising

**arXiv ID:** 2609.11553 | [PDF](https://arxiv.org/pdf/2609.11553v1)

**作者:** Hongjin Chen `[一作]` (Fudan University), Wenchao Ding `[通讯]` (Fudan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种单阶段人形机器人行走控制策略，能在深度传感器部分失真时持续自适应，保持稳健行走。

**💡 创新点**

创新点在于将世界模型去噪编码器与共活的本体感知VAE编码器联合使用，并通过深度噪声课程与特征丢弃的耦合训练，实现连续而非离散的感知-盲模式切换。

**🔧 技术方法**

采用强化学习（PPO）结合世界模型（DreamerV3 RSSM）和β‑VAE编码器，使用深度噪声课程和特征丢弃进行联合训练。

**📊 数据集**

在仿真中使用Unitree G1机器人，覆盖阶梯、平台、隙缝和障碍等多种地形；在真实机器人上测试室内遮挡、闪光噪声和户外场景。

**📈 对比分析**

与Hiking、PIE、Binary‑switch等基线对比，结果显示在清晰和受损两种感知条件下都能保持高成功率（≈99%），并且在感知质量下降时实现渐进式性能退化，优于硬切换方法。

**⚠️ 局限性**

局限性包括对极端几何失真或完全失去感知时的安全性不足，未考虑失误恢复和风险评估。

---

## 383. Particle GFlowNets: Rethinking Generative Marginalization Models

**arXiv ID:** 2609.11538 | [PDF](https://arxiv.org/pdf/2609.11538v1)

**作者:** Tiago da Silva `[一作]` (MBZUAI), Salem Lahlou `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出将生成边缘化模型（MaMs）视为条件生成流网络（GFlowNet）的特例，并基于此发展了一种新型非自回归采样框架——粒子GFlowNet（P-GFlowNet），通过持久化Gibbs采样和链重启技术显著降低训练过程中的前向传播次数，提升学习速度。

**💡 创新点**

核心创新在于：①证明MaMs等价于Permutation-Conditioned GFlowNet，揭示两者之间的理论联系；②将持久化Gibbs采样扩展到非自回归生成任务；③设计基于Gelman-Rubin统计量的自动链重启机制以加速收敛；④提出仅需常数前向传播的无偏一致性目标。

**🔧 技术方法**

采用生成流网络框架、细节平衡（DB）损失、持久化随机块Gibbs采样、Gelman-Rubin指标进行链重启、并在实验中使用神经网络实现前向/后向策略与流函数。

**📊 数据集**

在多种离散组合任务上验证：固定大小子集生成（log-additive reward）、贝叶斯变量选择、Ising模型模拟、位序列生成等；数据集包括随机生成的可加奖励集合、模拟的线性回归数据、Ising系统样本以及位序列集合。

**📈 对比分析**

与传统GFlowNet训练方法比较，P-GFlowNet在马尔可夫决策过程（MDP）步长增大、奖励评估成本低于采样成本时，在墙钟时间上实现了数倍至十几倍的加速；链重启策略进一步提升了收敛速度和探索效率。

**⚠️ 局限性**

局限性在于：①当奖励评估成本高于采样成本（如大样本贝叶斯变量选择）时，持久化采样的优势减弱；②链重启阈值需经验调节，缺乏自动化选择机制；③对非分解式（非factorized）空间的理论保证有限，仍需进一步探索更通用的持久化采样方法。

---

## 384. UBone3D: Physics-Rectified Conditional Flow Matching for Anatomical 3D Shape Completion from Ultrasound

**arXiv ID:** 2609.11506 | [PDF](https://arxiv.org/pdf/2609.11506v1)

**作者:** Weiying Chen `[一作]` (University of Alberta), Edmond Lou `[通讯]` (University of Alberta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了UBone3D框架，利用物理校正的条件流匹配方法，直接从含有音频伪影且部分缺失的超声点云完成三维骨骼形状恢复。

**💡 创新点**

创新点在于将解剖学可行性（BoneFM）与物理一致性（USimNet）解耦为两个驱动力，在推理时通过差分可微物理代理进行测试时的物理校正，并采用定向Chamfer距离、观察锚定与时间依赖的物理引导策略提升完成质量。

**🔧 技术方法**

采用Optimal Transport Flow Matching与条件速度场、PointNet+++FiLM、分类器自由引导、Heun ODE积分、可微物理代理USimNet、Chamfer/EMD/F-score评价，以及PyMUST射线模拟生成训练样本。

**📊 数据集**

使用基于Spine1K的Spine1K-PC仿真数据集（含P_geo、P_phys-simp、P_phys-full四种视角）以及5位健康志愿者的真实超声数据（共24个椎体）进行训练与评估。

**📈 对比分析**

与PoinTr、SVDFormer、PCDreamer、Gafencu VAE、SSM-Net*等基线进行对比；在仿真数据上，UBone3D在L1 CD、EMD、F-score上均优于大多数基线；在真实数据上，laminae距离误差约1.3mm，显著低于SVDFormer(3.94mm)和SSM-Net*(3.38mm)。

**⚠️ 局限性**

局限性包括：仍需在真实数据上进一步优化（尤其是极端伪影场景）；物理代理仅为近似，可能在某些超声条件下失效；仅评估了少量真实病例，缺乏大规模临床验证；推理需ODE积分，计算开销相对较高。

---

## 385. Structural priors for data-efficient language learning

**arXiv ID:** 2609.11505 | [PDF](https://arxiv.org/pdf/2609.11505v1)

**作者:** Yana Veitsman `[一作]` (University of Göttingen), Lisa Beinborn `[通讯]` (University of Göttingen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在多语言小规模语言模型中，先用结构化非语言数据（如概率上下文无关文法、细胞自动机、钢琴音乐、蛋白质序列和随机序列）进行预预训练，再继续训练自然语言，以提升样本效率。

**💡 创新点**

证明结构化数据可作为权重初始化，显著降低语言训练时的权重偏移并提升下一词预测的token效率，但在下游语言基准上提升有限，表明结构化迁移并非天然语言数据的直接替代。

**🔧 技术方法**

采用GPT‑2‑small Transformer架构，使用BPE tokenizer、token‑efficiency与权重偏移（weight‑shift）分析等技术，对下一词预测损失进行评估。

**📊 数据集**

使用的结构化数据集包括PCFG、CA16/CA256、Aria‑MIDI钢琴音乐、Swiss‑Prot蛋白质序列和随机序列；语言数据来自BabyLM多语言（中文、荷兰语、英语）语料库，并与英文维基百科作为基线进行对比。

**📈 对比分析**

通过与随机初始化以及额外自然语言数据（英文维基百科）对比，发现结构化预训练在下一词预测上可实现约60% token效率提升（即需60% tokens即可达到相同损失），但在BabyLM零样本与微调任务中的平均准确率仅提升1–2个百分点，远低于单纯增加自然语言数据的效果。

**⚠️ 局限性**

局限性包括：仅使用GPT‑2‑small模型；结构化数据类型有限，未覆盖所有可能的结构化源；仅对三种语言评估，未深入探讨语义层面对迁移的影响；权重偏移分析仅停留在宏观层面，缺乏机制性解释。

---

## 386. IPv6 Hitlist Service: Lessons Learned From 10 Years of Operation

**arXiv ID:** 2609.11475 | [PDF](https://arxiv.org/pdf/2609.11475v1)

**作者:** Oliver Gasser `[一作]` (IPinfo), Johannes Zirngibl `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文基于十年的运营经验，搭建并持续维护了IPv6 Hitlist服务，并对其覆盖范围、用户使用行为及最佳实践进行了系统评估。

**💡 创新点**

创新点在于首次对IPv6 Hitlist的长期覆盖率进行实测与ISP流量对比、揭示不同来源对地址库的贡献差异，并提出了针对别名前缀和响应性地址的最佳使用策略。

**🔧 技术方法**

技术方案包括多源地址收集（DNS、Traceroute、TGA等）、去重与IP/网络筛选、别名前缀检测、跨协议（ICMP、TCP/80/443、UDP/53/443、SNMPv3、MPTCP）扫描以及扫描结果的归档与公开。

**📊 数据集**

使用的数据集涵盖从2018年至2026年的累计地址库、主要欧洲ISP与Tier‑1网络的流量采样、用户问卷数据以及服务访问日志，此外还对DNS顶级列表（Google Crux、Cloudflare Radar、Majestic、Cisco Umbrella）进行了对照。

**📈 对比分析**

对比方法是将Hitlist中地址与ISP流量的/48前缀、AS覆盖率及流量占比进行交叉统计，结果显示Hitlist覆盖了87.1% AS、56.5% /48前缀，且涵盖了97.2%流量，而DNS顶级列表仅覆盖88.8%流量，说明Hitlist在覆盖面上更优。

**⚠️ 局限性**

主要局限在于别名前缀检测阈值可能导致误判、对高速旋转或私有网络的覆盖不足、以及仅以/48级别评估流量导致对细粒度分布缺乏洞察。

---

## 387. Pre- and Post-Treatment Brain Metastases Segmentation Using nnU-Net with Post-Processing for BraTS 2026

**arXiv ID:** 2609.11477 | [PDF](https://arxiv.org/pdf/2609.11477v1)

**作者:** Haobin Liu `[一作]` (Jilin University), Xin Wang `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

针对BraTS 2026脑转移瘤分割任务，本文构建了基于nnU-Net ResEnc-L的5折集成模型，并通过三阶段规则化后处理（清洗、小分量去除、切除腔体边界扩展、外脑假阳性抑制）来提升小病灶的识别与分割质量。

**💡 创新点**

创新点在于：①在LW-DSC（基于病灶的Dice）评估下，提出针对性后处理策略，特别是概率门控的切除腔体边界扩展，显著提升了极少见切除腔体的分割；②对每个后处理步骤进行5折OOF验证，系统评估其鲁棒性，避免仅靠排行榜指标导致的过拟合；③通过大量负实验（损失函数、网络骨干、推理设置）阐明常见直觉的局限性。

**🔧 技术方法**

核心技术包括：nnU-Net ResEnc-L架构（残差编码器+实例归一化）、SGD+Nesterov优化、Dice+交叉熵复合损失、随机镜像/弹性变形/旋转/尺度等增强、fp16混合精度训练、滑动窗口推理与高斯重要性加权、测试时三轴镜像、以及规则化的后处理脚本（阈值去除、小分量过滤、概率门控扩展、脑掩模抑制）。

**📊 数据集**

使用BraTS-MET 2025训练数据集（1296例，四模态T1c/T1n/T2-FLAIR/T2w）进行训练，验证集为官方179例Synapse验证集，所有数据均为完整四模态。

**📈 对比分析**

在官方验证排行榜上，最终LW-DSC分别为ET 0.733，TC 0.751，WT 0.713，RC 0.549，平均4值0.6864，位居ET/TC/WT三个肿瘤相关子区前列；后处理阶段1显著提升整体分数，阶段2在RC上提供了小幅稳健提升，阶段3在OOF验证中未能保持一致性。相比之下，其他公开基线和团队在RC上的表现更优。

**⚠️ 局限性**

局限性包括：①切除腔体（RC）的LW-DSC仍显低，说明后处理无法弥补训练数据稀缺导致的分割欠缺；②负实验表明常用的损失改进、网络加权和推理增强在此任务上效果有限；③由于只针对规则化后处理，未探索专门的RC检测/训练策略，未来工作需在模型架构和数据采样层面进一步改进。

---

## 388. PRISMA-LLM: An Empirical Reporting Framework for AI-Assisted Systematic Reviews

**arXiv ID:** 2609.11559 | [PDF](https://arxiv.org/pdf/2609.11559v1)

**作者:** Miguel Zabaleta `[一作]` (Icahn School of Medicine at Mount Sinai), Baihan Lin `[通讯]` (Icahn School of Medicine at Mount Sinai)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对SciLitBench 888篇系统综述自动化论文进行二次分析，量化方法、使用阶段、评估与限制报告的演变，并基于实证发现提出了新的PRISMA-LLM报告框架。

**💡 创新点**

创新点在于：①首次用大规模语料实证挖掘自动化方法的报告模式；②将方法复杂度与报告丰富度关联，提出分层实现披露与风险无关的评估框架；③提供完整的检查表、可填写工作簿与实现层次说明，填补了当前AI辅助综述报告的空白。

**🔧 技术方法**

使用技术包括系统综述与元分析方法、文本注释、定量描述统计、滚动窗口分析、对数线性增长模型、报告丰富度指标构建，以及多维度实现披露层级设计。

**📊 数据集**

数据集为SciLitBench（888篇论文，14,726条注释），涵盖生命科学医学、工程技术、社会科学与管理等高层领域，时间范围至2025年6月。

**📈 对比分析**

通过比较不同方法组（LLM、软件/产品、传统/深度学习/ BERT）在报告丰富度、评估缺失比例等指标上的差异，发现LLM论文报告更充分、软件产品论文评估不足；未针对模型性能做对比，而是评估报告完整性与方法复杂度的关系。

**⚠️ 局限性**

局限性包括：①样本偏向生命科学，其他领域代表性不足；②报告丰富度指标只衡量宽度，未反映深度；③未经过正式共识或前瞻性验证；④方法复杂度与风险不等价，需结合任务后果进一步评估。

---

## 389. Generalized Score Matching for Parameter Estimation on Convex Domains

**arXiv ID:** 2609.11521 | [PDF](https://arxiv.org/pdf/2609.11521v1)

**作者:** Nishanth Shetty `[一作]` (Indian Institute of Science), Chandra Sekhar Seelamantula `[通讯]` (Indian Institute of Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

基于最小概率流(MPF)的极限分析，推导出在凸子集上通用的分数匹配(GSM)目标，证明其为二阶局部合适的计分规则，并在指数族模型中证明目标凸性与估计器一致性；在实验中将GSM用于受限域的参数估计和生成模型训练，展示优于现有方法。

**💡 创新点**

从MPF统一衍生GSM，说明传统分数匹配及其非负/受限域变体是特殊情况；给出G_phi生成器的构造与选择，阐明不同凸势对应不同边界衰减；证明GSM为合适计分规则，并在指数族中证明凸性与一致性。

**🔧 技术方法**

使用最小概率流、Bregman距离局部邻域、泛化分数匹配目标、算子理论框架、正确计分规则理论、指数族的矩估计与一致性分析、以及在生成模型中利用GSM损失的VAE训练技术。

**📊 数据集**

合成截断高斯分布（支持在正交域和单纯形）、MNIST与CelebA数据集用于生成模型实验。

**📈 对比分析**

与截断分数匹配（Truncated SM）以及基于h(x)=x的基线方法对比；在单纯形上，选取不同ϕ（ϕ1、ϕ2、ϕ3）进行参数估计，ϕ1在所有样本量下均取得最低MSE；在生成任务中证明GSM可用于隐式VAE训练。

**⚠️ 局限性**

高维下计算生成器G_ϕ的成本高，限制了ϕ的选择；目前缺乏系统的ϕ选择准则，边界衰减率对性能影响大；尚未提供ϕ的非渐近误差或MLL恢复的理论保证。

---

## 390. Accountability in Certificate Transparency and Variants

**arXiv ID:** 2609.11552 | [PDF](https://arxiv.org/pdf/2609.11552v1)

**作者:** Timo Treitz `[一作]` (Saarland University), Robert Künnemann `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对Certificate Transparency（CT）协议进行形式化建模与自动分析，验证其在不同攻击场景下的责任追溯（accountability）属性；

**💡 创新点**

首次在Dolev‑Yao模型下将CT的责任追溯问题正式化，并探讨SCT审计与Gossiping等扩展在消除对诚实日志假设方面的差异；

**🔧 技术方法**

采用Tamarin工具进行符号推理，利用多重重写（MSR）和可追溯性检验技术构建协议模型并自动生成责任判定测试；

**📊 数据集**

论文未使用传统数据集，而是基于CT规范及其扩展的协议描述作为模型输入；

**📈 对比分析**

通过对比基础CT、SCT审计和Gossiping三种模型的责任追溯证明，展示SCT审计在保持最少假设下实现完整责任追溯，而Gossiping则无法满足此目标；

**⚠️ 局限性**

局限性在于模型假设多为理想化（如日志完全诚实或监视器独立），实际部署中存在隐私、性能与协作复杂性等未被充分评估的问题。

---

## 391. Breaking the Central Bias: Spatially Partitioned Experts for Coordinate-Based Neuroevolution

**arXiv ID:** 2609.11518 | [PDF](https://arxiv.org/pdf/2609.11518v1)

**作者:** Romain Claret `[一作]` (University of Neuchâtel), Pascal Felber `[通讯]` (University of Neuchâtel)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对坐标基间接编码的ES‑HyperNEAT在MNIST上的“中心偏倚”问题，作者提出将输入空间按空间划分并为每块演化独立专家网络，形成独立专家Mixture‑of‑Experts结构；

**💡 创新点**

创新点在于：将空间划分与专家化相结合，显式强制网络覆盖全图像，从而消除中心偏倚；并提出无验证数据的均匀聚合及多种数据驱动聚合策略；

**🔧 技术方法**

使用技术包括ES‑HyperNEAT、Cooperative Coevolution、CPPN与四叉树分解、各种聚合策略（均匀、结构掩码、性能加权）以及受控实验设计（SBG、SBE、SM等）；

**📊 数据集**

实验数据集为MNIST手写数字分类；

**📈 对比分析**

通过与单块ES‑HyperNEAT基线、共享专家模型以及多种聚合方式对比，结果表明13专家的独立专家模型平均准确率达43.07%，相较基线20.95%提升106%，即约70%无验证均匀聚合即可实现；

**⚠️ 局限性**

局限性包括：仅在MNIST这一中心化数据上验证，未能独立区分坐标偏倚与信息密度偏倚；实验仅采用一维线性划分，未探讨二维或自适应划分；以及未对比纯HyperNEAT基线以进一步验证CPP‑N不规则性与中心偏倚的关系。

---

## 392. Prototype Matters: Modality-unified Prototype Self-distillation for Unsupervised Visible-infrared Person Re-identification

**arXiv ID:** 2609.11514 | [PDF](https://arxiv.org/pdf/2609.11514v1)

**作者:** Menglin Wang `[一作]` (Nanjing Normal University), Xiaojin Gong `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种面向无监督可见-红外人像重识别的完整学习框架

**💡 创新点**

创新点：① 将可见和红外的原型统一到同一空间，使用模态统一原型对比损失显式降低跨模态相似度差距；② 利用在线更新的质心原型作为稳健教师，设计原型引导自蒸馏来软化实例-原型匹配并纠正离线聚类匹配的噪声；③ 在同一框架内兼顾离线OT匹配和在线蒸馏，二者互补。

**🔧 技术方法**

核心技术：AGW backbone + ResNet‑50；单模态迭代聚类 + DBSCAN；跨模态对齐采用多步 optimal transport；模态统一原型对比 loss；原型引导自蒸馏 loss；组归一化（modality‑aware normalization）；双阶段训练（intra + cross）。

**📊 数据集**

实验使用的公开基准：SYSU‑MM01、RegDB 和 LLCM，分别在 All‑Search、Indoor‑Search、Visible‑to‑Infrared、Infrared‑to‑Visible 等评估设置下进行验证。

**📈 对比分析**

与现有 SOTA（包括监督、半监督及无监督方法）对比，本文在 SYSU‑MM01 上 Rank‑1 最高达 95.3%（All‑Search），在 RegDB 上 Rank‑1/ mAP 分别 89.6% / 94.9%，均优于此前最佳无监督方法；与 BMIL 结合后进一步提升。

**⚠️ 局限性**

局限性：① 仍依赖聚类的质量，噪声较大时匹配效果受限；② 模式统一对比和自蒸馏对温度、学习率等超参数较为敏感；③ 仅在公开数据集验证，跨场景迁移与实时部署尚待进一步评估。

---

## 393. Extending SMT Solving with Non-Ground Clause Learning

**arXiv ID:** 2609.11509 | [PDF](https://arxiv.org/pdf/2609.11509v1)

**作者:** Yasmine Briefs `[一作]` (Max Planck Institute for Informatics), Christoph Weidenbach `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的非基于实例化的 SMT 推理演算（[\mathcal{S}\u005D），实现了从实例化产生的地面冲突中直接进行非地面冲突分析和学习，支持与 CDCL(T)、SCL(FOL)、Resolution 等既有计算框架的统一模拟。

**💡 创新点**

核心创新在于将量化子实例化与非地面冲突学习紧密耦合，利用原始非地面子句在冲突分析中进行分辨式解析，从而产生更具一般性、且在合理策略下可保证非冗余的学习子句；同时引入了时间顺序回溯（chronological backtracking）与理论解释的非地面化等新机制。

**🔧 技术方法**

技术主要包括：多排序一阶逻辑与理论推理的抽象框架、实例化规则与删除规则、基于 CDCL(T) 的决策与传播、理论规则（如 -Propagate、-Conflict、Explain）、冲突分析规则（Resolve、Factorize）、回溯规则（BacktrackClassic、BacktrackFOL、BacktrackCB），并通过定义的可行策略（reasonable、first‑order aware）保证学习子句的非冗余性与算法终止性。

**📊 数据集**

文中未给出具体实验数据集，重点是对理论性质的证明（如完整性、非冗余性、终止性、与其他计算框架的模拟），因此没有使用传统 SMT 评测数据集。

**📈 对比分析**

由于缺乏实现与实验，论文仅通过形式化证明与模拟定理展示该框架相对于 CDCL(T)、SCL(FOL)、Resolution 等的兼容性与理论优势；并未给出具体性能对比或实验结果。

**⚠️ 局限性**

局限性包括：1）目前仅为理论框架，尚未实现或在实际 SMT 求解器中测试；2）需要依赖具体实例化策略与理论解释策略，若实现不当可能导致效率低下；3）在一般理论组合下的完整性仅在存在有限不相容实例化集合时可保证；4）非冗余性保证仅相对于当前实例集 G，若实例化不完整可能产生冗余学习。

---

## 394. Recursive Code World Models: Building Complex Worlds through Recursive Scene Programs

**arXiv ID:** 2609.11499 | [PDF](https://arxiv.org/pdf/2609.11499v1)

**作者:** Zhiqi Li `[一作]` (Georgia Institute of Technology), Bo Zhu `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出递归代码世界模型（Recursive Code World Models），通过递归构造递归场景程序（Recursive Scene Programs）实现从单张参考图像生成可执行的3D场景程序；

**💡 创新点**

创新点在于将整个世界与其子世界统一地视为可递归求解的子问题，采用全局–局部–全局递归流程、参考对齐相机视图和父级重访机制，显著提升细节精度与整体一致性；

**🔧 技术方法**

技术上结合了视觉‑语言编码代理、Three.js 编译器、递归求解器、上下文组装、裁剪对齐相机以及基于渲染的视觉反馈循环；

**📊 数据集**

使用了十幅单视图参考图，包括CC0《Isometric city》中的city‑full、school‑block、police‑corner、park‑lake、shop‑row以及WorldClaw演示的island‑harbor、medieval‑village、snow‑village、japan‑island和valley‑village；

**📈 对比分析**

在与SEIG、VIGA、img2threejs同一基准模型和推理预算下，Recursive Code World Models在PSNR、SSIM、LPIPS、边缘F1、CLIP相似度等指标上均优于对手，尤其在全景与局部细节恢复方面表现突出；

**⚠️ 局限性**

局限性包括单视图输入导致隐藏几何和尺度不确定、视觉检查可能忽略错误、计算量大且收敛性未得到保证、实验场景有限且仅做单次运行。

---

## 395. Fundamentals of Energy-Efficient Hardware Configurations for Wireless Links with Sleep Modes

**arXiv ID:** 2609.11474 | [PDF](https://arxiv.org/pdf/2609.11474v1)

**作者:** Anders Enqvist `[一作]` (Kth Royal Institute Of Technology), Emil Björnson `[通讯]` (Kth Royal Institute Of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究基站多天线链路的能效，联合优化发射功率、带宽与天线数，并将睡眠模式纳入能效分析。

**💡 创新点**

提出能效最优SNR为5.93 dB的普适常数、发射功率与天线功率相等的最优关系，并给出闭式解与rush‑to‑sleep策略。

**🔧 技术方法**

采用解析优化、Lambert W函数、闭式最优比和交替优化算法；结合睡眠模式功耗模型。

**📊 数据集**

未使用真实数据集，全部通过仿真参数（如β、N0、μ等）进行数值验证。

**📈 对比分析**

通过与持续传输基线和不同睡眠模式比较，展示在低速率下能效提升显著，满足QoS且实现最高能效。

**⚠️ 局限性**

局限于单链路、窄带模型，假设瞬时状态转换且忽略硬件非线性；整数天线限制需后处理。

---

## 396. BruNet: A Cross-Domain Transfer Framework for Bruise Segmentation

**arXiv ID:** 2609.11463 | [PDF](https://arxiv.org/pdf/2609.11463v1)

**作者:** Qiming Wang `[一作]` (Cardiff University), Paul L. Rosin `[通讯]` (Cardiff University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一个零样本跨域分割框架BruNet，用于自动化瘀伤（bruise）分割；

**💡 创新点**

创新点在于：①采用ViT视觉编码器与SAM掩码解码器的组合；②设计双区域注释协议（高置信度核心区与不确定边界区）；③在无瘀伤数据训练的前提下实现出域泛化；

**🔧 技术方法**

技术包括：Vision Transformer（LingBot-Vision 或 DINOv3）、SAM（Segmentation Anything Model）解码器、轻量化卷积上采样适配器、LoRA微调、Dice+BCE 损失、Retinex 数据增强；

**📊 数据集**

使用的数据集为：HAM10000（10,015张皮肤病变图像，带二值掩码）做训练；评估使用了86张自行搜集的瘀伤图像并由专家双区域注释；

**📈 对比分析**

与传统CNN（U‑Net）、ViT（OneFormer）、零样本SAM、LLM‑prompted SAM 组合等基线进行对比。BruNet在Dice和IoU上均优于所有基线，特别是BruNet‑LingBot在Dice达到0.867、IoU 0.790；统计检验显示相对于大多数基线差异显著；并且模型推理速度快、显存占用低；

**⚠️ 局限性**

局限性包括：双区域注释仅由两位专家完成，缺乏多专家对比；仅针对单一颜色变化的瘀伤；未评估不同肤色、光照、尺寸等更广泛情况；缺乏对年龄、严重度等进一步分析。

---

## 397. ActMap: Single-Pass Uncertainty Quantification from Generation-Time Activation Maps

**arXiv ID:** 2609.11498 | [PDF](https://arxiv.org/pdf/2609.11498v1)

**作者:** Jacopo Dardini `[一作]` (University of Bologna), Roberta Calegari `[通讯]` (University of Bologna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在生成过程中一次性记录Transformer层隐藏状态轨迹并压缩为固定大小激活图的方法，以评估单个答案的正确性概率。

**💡 创新点**

创新点在于将完整的隐藏状态轨迹通过时间统计、层级池化和归一化压缩成12通道×32×128的固定张量，既保持层深和隐藏维度结构，又实现极低的存储和计算开销；并训练一个轻量Vision Transformer分类器直接从该激活图预测答案正确性。

**🔧 技术方法**

技术包括前向钩子捕获隐藏状态、时间统计（均值、方差、斜率等）、自适应平均池化、通道标准化、Vision Transformer分类器以及对抗式混合噪声训练。

**📊 数据集**

使用了TriviaQA、NQ-Open、GSM8K、CNN/DailyMail四个数据集，分别对应短答案事实问答、开放式问答、数学问题解决和长文本摘要事实性评估；模型包括Qwen3-8B、Llama-3.1-8B、Mistral-7B。

**📈 对比分析**

与八类基线（黑盒采样、灰盒分数、白盒注意力/嵌入等）以及ACT-ViT比较，均在12个模型-任务组合上取得最高的AUROC与AUPRC，平均AUROC约0.825，几乎与大尺寸ACT-ViT匹配但只需96 KiB和单次前向推理。

**⚠️ 局限性**

局限性主要是转移能力差：在不同任务、生成器或规模上训练得到的决策边界难以直接迁移，需要在目标域再标注并训练；此外仅在白盒可访问内部状态时适用，且对生成策略（温度、解码方式）鲁棒性未充分验证。

---

## 398. World in World: Explore the World with World Models

**arXiv ID:** 2609.11548 | [PDF](https://arxiv.org/pdf/2609.11548v1)

**作者:** Chenxi Song `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了训练‑free 的视觉证据接口 WiW，利用多源视觉证据在冻结的因果视频世界模型上实现自由视角控制、长时程回访和动作迁移。

**💡 创新点**

创新点包括：①将源视频观测、目标视角投影、几何渲染、历史回忆统一转化为干净视觉状态并通过模型本地自注意力读取；②引入对应引导注意路由（CGAR）定位合适的源视频证据；③提出证据级注意力CFG（EWA）在同一次去噪前向传递中调节各证据的贡献；④完全无需额外训练或适配器。

**🔧 技术方法**

使用的技术包括因果视频扩散模型（LingBot‑World 2.0）、RoPE 时空编码、深度估计与投影、几何渲染、历史键值缓存、对应引导注意路由与证据级注意力CFG。

**📊 数据集**

评估数据集为 DAVIS 和 OpenVid‑1M。

**📈 对比分析**

与 ReCamMaster、TrajectoryCrafter、WorldForge、InSpatio‑World、UniWorld‑View、CameraAnything 等基线在 VBench 七维度、相机误差、PSNR/SSIM/LPIPS 等指标上进行对比，WiW 在大多数指标上获得最佳或并列最佳，且相机误差最低。

**⚠️ 局限性**

局限性包括：对极端摄像机运动或严重遮挡的鲁棒性仍有限；方法依赖预训练模型的视觉与运动先验，未在大规模多摄像机或实时交互场景中验证；同时需要手动构造几何渲染和历史检索，推理成本相对较高。

---

## 399. Ethics Training Agents: Facilitating Group-Based Ethics Education with Role-Playing and Discussion for Ethical Reflection and Exploration

**arXiv ID:** 2609.11529 | [PDF](https://arxiv.org/pdf/2609.11529v1)

**作者:** Youngseok Seo `[一作]` (KAIST), Uichin Lee `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一套名为 Ethics Training Agents 的多智能体群组讨论系统，帮助 STEM 学生在小组中通过角色扮演和讨论进行伦理教育。

**💡 创新点**

创新点在于将多位 LLM 参与者（分别代表关怀伦理、义务伦理和务实伦理）与一个 LLM 主持人结合，实现结构化的人工智能辅助讨论；通过预设角色、分阶段流程和自动化的发言管理，大幅降低了人工调度成本，并为学生提供一致且可比的伦理视角。

**🔧 技术方法**

使用技术包括 GPT‑4o 大语言模型、LangGraph 框架、WebSocket 同步客户端与服务器、基于角色的提示工程、手势提问（stacking）机制、自动摘要与议题筛选等。

**📊 数据集**

数据集方面主要使用了来自《黑镜》剧集的“记忆增益智能眼镜”场景作为讨论任务，并未使用公开的大规模语料库；实验数据来自 45 名本科 STEM 学生的问卷、对话日志和访谈记录。

**📈 对比分析**

方法上采用前后测设计评估伦理敏感度，使用配对样本 t 检验与 Wilcoxon 检验验证显著性，效应量 Cohen’s d 超过 1.3；对话日志分析量化参与者对 AI 角色的提问比例；对 AI 与人类参与者的评价进行 Kruskal–Wallis 检验。结果显示系统显著提升了伦理敏感度，参与者更愿意与 AI 交流，但对 AI 的贡献与多样性评价低于人类同行。

**⚠️ 局限性**

局限性包括：缺乏对照组导致因果推断受限；实验仅关注短期效果；仅使用 GPT‑4o 可能影响结果泛化；AI 参与者表现过于顺从，缺乏批判性争论；人际互动被削弱；对 AI 角色错误的容忍度不足；未充分探索如何更好展示 AI 的推理过程与专业知识。

---

## 400. Post-Training Zero-Shot TTS for Fine-Grained Emotion and Duration Control via Natural Language

**arXiv ID:** 2609.11523 | [PDF](https://arxiv.org/pdf/2609.11523v1)

**作者:** Lianru Gao `[一作]` (Nankai University), Yong Qin `[通讯]` (Nankai University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个统一的后训练框架，使预训练的 TTS 模型能够通过自然语言指令实现段级情感和时长控制。

**💡 创新点**

创新点在于将监督微调与群组相对策略优化（GRPO）结合，使用指令等价蒸馏提升指令鲁棒性，同时不需要额外的推理时控制模块。

**🔧 技术方法**

技术包括监督微调（SFT）、GRPO、指令等价蒸馏、PSOLA 时长变换、Qwen3‑ForcedAligner 对齐、MERaLiON‑SER‑v1 情感评估、强化学习奖励设计等。

**📊 数据集**

使用了 MED‑TTS、LibriTTS‑R、ESD、Seed‑TTS、IndexTTS2 等数据集，以及自构建的多种情感与时长指令集。

**📈 对比分析**

与 CosyVoice2/3、IndexTTS2、TED‑TTS、MAGIC‑TTS 等基线比较，实验显示在情感准确率、时长 A15/U15、联合控制指标上显著提升，联合控制单句准确率虽低（5.29%）但整体表现优于基线；ASR 错误率和说话人相似度保持在低水平。

**⚠️ 局限性**

局限在于联合控制的严格准确率仍偏低，指令鲁棒性对极端时长或情感表达的适应有限，且未覆盖更细粒度的控制（如单词强调）。

---

## 401. Learning Interaction between Image and Layout Priors for Joint Image-Layout Generation in Design Templates

**arXiv ID:** 2609.11519 | [PDF](https://arxiv.org/pdf/2609.11519v1)

**作者:** Shirong Yang `[一作]` (ShanghaiTech University), Ying Cao `[通讯]` (ShanghaiTech University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种联合图像-布局生成模型 InterIL，能够在单一步骤内同时生成背景图像和前景布局，实现完整的设计模板生成。

**💡 创新点**

核心创新包括：①可学习的双向通信模块，显式建模图像与布局之间的交互；②冻结预训练单模态扩散模型只训练通信模块，从而高效捕获图像-布局协同分布；③训练免费、用户可控的偏好引导技术，可在推理阶段实现遮挡/可读性等设计偏好。

**🔧 技术方法**

技术手段：潜在扩散模型（Latent Diffusion）；图像先验采用 Stable Diffusion；布局先验采用 VAE+DiT；通信模块采用交叉注意力；引导技术通过噪声梯度调整实现；评估使用多项指标（FID、TemplateFID、CLIP、LayoutFID、对齐、重叠、遮挡、可读性）及 GPT-5/人类评测。

**📊 数据集**

训练与评估数据集：主要使用 Web‑design（50k 网页横幅设计）；布局 VAE/DiT 在同一数据集上训练；模板自动编码器在混合数据集（GenPoster100K、CGL、Crello、PKU）上训练，Web‑design 作为评估集。

**📈 对比分析**

与 Desigen（两阶段）和 OpenCOLE 等方法在 Web‑design 上进行对比。InterIL 在 FID、TemplateFID、CLIP、LayoutFID、遮挡、可读性等多数指标上显著优于对比方法，并在 GPT‑5 与人类评估中取得最高分，显示出更好的图像质量、布局质量和图像-布局和谐度。

**⚠️ 局限性**

局限性：①仅在冻结预训练 backbone 的前提下训练通信模块，缺乏进一步的多模态融合；②通信模块仅在前 30% 步启用，比例取值需经验调优；③偏好引导会对图像质量略有影响；④目前针对单一 Web‑design 任务，尚未在更大范围或多样化任务上验证；⑤生成速度虽快于多阶段方法，但仍受扩散采样步数限制。

---

## 402. Harnessing Intrinsic Subject-Aware Attention for Controllable Multi-Subject Video Generation

**arXiv ID:** 2609.11507 | [PDF](https://arxiv.org/pdf/2609.11507v1)

**作者:** Niange Yu `[一作]` (Alibaba Group), Pipei Huang `[通讯]` (Alibaba Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出双相位内部注意力利用框架，在多主体视频生成中实现可控身份一致性并减少语义漂移。

**💡 创新点**

发现并利用Diffusion Transformer中特定层产生的Intrinsic Spatial Grounding Map (ISGM) 作为精准主体定位信号，利用其进行训练无关的推理时可控注意力，并通过零成本偏好构造实现强化学习来锚定高噪声阶段的注意力。

**🔧 技术方法**

使用Diffusion Transformer (DiT) 结构、VAE 编码/解码、注意力偏置、强化学习 DPO、语义分割掩码等技术。

**📊 数据集**

在OpenS2V‑Eval基准集（180个多主体提示）上进行评估，并在Phantom‑14B、Kaleido‑14B等公开S2V模型基础上进行实验。

**📈 对比分析**

与多种S2V基线对比，取得OpenS2V‑Eval的state‑of‑the‑art分数，身份一致性（FaceSim、NexusScore）显著提升，且通过γ参数实现从低到高的可控性，整体视频质量保持不下降。

**⚠️ 局限性**

仍存在高噪声阶段ISGM不稳定的局限，需要强化学习修正；在多主体情境下需多次掩码提取；对实时性和复杂动态场景的可扩展性尚未充分验证。

---

## 403. DeFiFlowBench: Benchmarking and Improving Safe Executability in Natural-Language DeFi Workflow Synthesis

**arXiv ID:** 2609.11504 | [PDF](https://arxiv.org/pdf/2609.11504v1)

**作者:** Abhinav Rajeev Kumar `[一作]` (SRM Institute of Science and Technology), Manikandan Nanjappan `[通讯]` (SRM Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了DeFiFlowBench基准，提出Koan‑Safe结构修复与安全默认机制，用于评估自然语言DeFi工作流合成的安全性。

**💡 创新点**

通过将意图解析、候选生成与结构修复、默认安全参数分离成可切换层级，系统化区分语义与执行安全，并通过匹配候选消融与变形测试验证鲁棒性。

**🔧 技术方法**

使用LLM提示（直接、约束、少量示例、明确安全指令）、正则式意图解析、图结构生成、Python执行检查器、本地EVM模拟、三层静态安全代理评分等技术。

**📊 数据集**

包含120条开发提示与87条保留测试提示（共207条），涵盖交换、限价单、跨链、组合工作流，附难度标签、重述集与变形对。

**📈 对比分析**

所有系统使用同一评估器计算三层静态安全分数和本地EVM执行安全率；Koan‑Safe混合版在静态安全代理上达到0.67，执行安全率为1.00，显著优于LLM基线（0.33）且无不安全执行；模板与基线表现最差。

**⚠️ 局限性**

仅检查配置级安全，未验证完整图路径；缺乏跨链、真实主网路由与对抗性测试；单位与阈值解释不统一；数据集由同一团队标注，缺乏独立验证；仅一次温度为0的模型生成，未评估多样性。

---

## 404. From Document Silos to Process Intelligence: A Multi-Layer Knowledge Graph for CMC Process Development

**arXiv ID:** 2609.11493 | [PDF](https://arxiv.org/pdf/2609.11493v1)

**作者:** Reza Amirmoshiri `[一作]` (Sanofi), Yasser Jangjou `[通讯]` (Sanofi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个双层知识图与代理式AI平台，能将多格式、多语言的CMC过程开发文档无损摄取、结构化，并通过检索增强生成（RAG）实现高质量问答；

**💡 创新点**

提出了词汇层+本体层的双层图架构、三层评测协议和路由代理，可在保持可追溯性的同时支持跨文档、跨项目的聚合与异常检测；

**🔧 技术方法**

使用Docling进行文档解析与OCR、机器翻译、Amazon Titan文本嵌入、Neo4j图数据库、DSPy ReAct框架、Claude Sonnet 4.5 LLM、GraphRAG以及基于提示的本体约束和全局实体解析；

**📊 数据集**

基于Sanofi小分子程序的38份文档（约52,000词、600页）及其31个快照构建双层图（约12,353节点、45,471边），并生成505道手工审核的问答题库；

**📈 对比分析**

通过与向量RAG基线进行对比，采用三层评测：T1（多选）准确率95%（比基线高6.5%），T2（LLM评判）通过率84.7%（比基线高19.2%）；域图进一步提升跨文档查询的完整性与精度；

**⚠️ 局限性**

主要限制包括：域图仅在小规模问答中评估；实体抽取存在非确定性和误差；未使用正式本体推理和推断引擎；图片（如反应示意图）未被解析；实体解析错误率较高；路由规则基于有限样本；对比大型长上下文LLM的实验缺失。

---

## 405. Published Unlearning Numbers Move Per Checkpoint, and Not Because the Removed Data Survives: An Audit of 263 Released Batch-Normalized Checkpoints

**arXiv ID:** 2609.11490 | [PDF](https://arxiv.org/pdf/2609.11490v1)

**作者:** Junlong Shen Xingyu Li `[一作]` `[通讯]` (University of Alberta), Junlong Shen Xingyu Li (University of Alberta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对公开的去学习模型检查点进行审计，重新计算批归一化统计量，并评估其对去学习审计结果的影响。

**💡 创新点**

提出将批归一化统计视为可重构的审计通道，证明统计陈旧性导致审计数值漂移，且漂移与去学习方法无关，而是与检查点本身相关。

**🔧 技术方法**

重构模型批归一化状态，按训练变换计算精确统计；使用十次随机子集重新拟合；基于符号判定的同意/拒绝测试。

**📊 数据集**

主要使用 MU‑Bench CIFAR‑100 ResNet‑50、CIFAR‑10 ResNet‑18、SVHN VGG‑16‑BN 以及自建的实验集。

**📈 对比分析**

将重新拟合的批归一化状态与原始发布状态以及无去学习参考模型对比；发现约 47/221 检查点的遗忘准确率在 ±1.2pp 之外漂移，说明检查点级别的影响显著。

**⚠️ 局限性**

无法在方法级别上给出显著差异，缺乏对非卷积或无批归一化模型的评估，且依赖于已公开的检查点和其原始报告。

---

## 406. The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation

**arXiv ID:** 2609.11489 | [PDF](https://arxiv.org/pdf/2609.11489v1)

**作者:** Makoto Fukushima `[一作]` (Honda Research Institute), Ehsan Moradi Pari `[通讯]` (Honda Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并量化“convention gap”——通过将基于提示的字面信息后验概率与实际失败率比较，测量合作游戏（Hanabi）中隐式约定的贡献。

**💡 创新点**

构建可精确计算的字面后验基准，首次将此基准与观测结果相减得到隐式沟通的可度量指标，并在多种合作设置（人-人、人-AI、AI-AI）以及不同AI代理上系统验证。

**🔧 技术方法**

离散枚举式后验计算、逻辑约束推理、统计bootstrap、对照实验与Off‑Belief Learning层级验证。

**📊 数据集**

三大公开数据集：hanab.live（人-人游戏）、HOAD（AI-AI游戏）、HanabiData（人-AI游戏）。

**📈 对比分析**

对比方法：在相同字面后验概率分箱下计算失误率差异，使用bootstrap和聚类检验置信区间；结果显示人-人约定差距+26.2pp，AI-AI约定差距-0.7pp，人与AI+16.4pp；约定差距与游戏分数呈不同趋势，表明约定兼容性可预测人-AI效果。

**⚠️ 局限性**

局限性：仅覆盖Hanabi场景，约定差距受数据集来源与人类经验水平影响；对AI内部机制的因果解释需进一步实验；对更复杂/连续动作域的通用性仍待验证。

---

## 407. CARLAverse: A Highly Modular, Distributed, and Multimodal Framework for Human-in-the-Loop Simulation

**arXiv ID:** 2609.11478 | [PDF](https://arxiv.org/pdf/2609.11478v1)

**作者:** Patrick Rebling `[一作]` (Karlsruhe University of Applied Sciences), Reiner Kriesten `[通讯]` (Karlsruhe University of Applied Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了CARLAverse，一套可扩展的多模态、分布式人机交互（HITL）仿真生态系统，整合车辆、单车和行人三类模拟器，支持跨机构、跨网络的同步实验。

**💡 创新点**

核心创新在于：① 分布式物理架构——将对人机交互关键的自我动力学和高频力反馈本地计算，中央服务器仅负责全局交通与NPC物理；② 基于CARLA原生API的低延迟RPC通信，彻底绕过ROS等中间件的序列化瓶颈；③ 统一的硬件抽象层和YAML配置，实现不同硬件（力反馈轮、滑动轮、VR头显、摄像头等）无缝集成；④ Web‑UI与可视化模块实现跨平台交互与实时监控。

**🔧 技术方法**

技术实现包括：CARLA API RPC + TCP、UDP高频力反馈/运动指令、C++ FFB/运动服务器、Python HAL库、YAML自动化配置、WebSocket UI、OpenXR与MediaPipe姿态估计、GPU多卡VR渲染、DSP滤波/运动补偿、D-BOX SDK接口。

**📊 数据集**

实验主要使用CARLA自带的开放式地图和交通场景（OpenSCENARIO）、自行构建的混合交通交叉路口场景；并未引入公开数据集，而是通过模拟产生的交通与人机交互日志进行分析。

**📈 对比分析**

性能对比：在本地节点与远程节点对比实验中，分布式物理架构将全局交互延迟从约30–40 ms下降至1–2 ms（力反馈闭环），实现稳定的高频（≥100 Hz）控制；同时支持多达数十名参与者并行交互，单节点单机无法满足。结果表明，跨机构实验的可靠性与沉浸感均保持在与本地单机相近的水平。

**⚠️ 局限性**

局限性：① 需要具备足够算力的本地节点来承载高频物理与渲染；② 网络波动仍可能导致外部NPC同步误差，需在设计中加入补偿机制；③ 目前VRU（单车、行人）物理模型与真实行为仍需进一步实验验证；④ 对低延迟网络的依赖使得在极低带宽或高丢包环境下表现不佳。

---

## 408. Using Automated Vehicles Operational Data to Confirm Safety and Anticipate Threats

**arXiv ID:** 2609.11549 | [PDF](https://arxiv.org/pdf/2609.11549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 409. Statistical Symmetry Release for Equivariant Quantum Learning

**arXiv ID:** 2609.11470 | [PDF](https://arxiv.org/pdf/2609.11470v1)

**作者:** Zeyu Chen `[一作]` `[通讯]`, Zeyu Chen

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文提出了统计对称性释放框架，利用两复制 SWAP 门实现无维数的全局对称性检测，并通过局部曲率矩阵估计实现软释放方向的选择，最终结合量子自然梯度进行模型训练与更新。

**💡 创新点**

创新点在于：①将全局对称性检验与局部曲率搜索紧密耦合，形成统一的证据链；②提出无维数的两复制交换门估计与同一测量样本的多频段分解；③给出软释放门的精确置信区间和梯度修正，并证明其在量子自然梯度中的投影效应；④在同一实验框架下比较不同测量模式（共享阴影 vs 标量探测）对释放成本的影响。

**🔧 技术方法**

技术方法包括：量子对称性（群平均化）测试、SWAP 交换门、双重对角矩阵估计、局部损失曲率矩阵、Gaussian 统计测试、同时置信区间、量子自然梯度（投影与软衰减）、阴影测量与单点差分估计、结构风险理论与软释放门的高斯上界。

**📊 数据集**

主要使用的数据集为量子模拟的八量子周期性 Transverse‑Field Ising 模型、六量子自旋翻转（spin‑flip）示例以及大维 Pauli 扫描的点零分布，用于验证全局对称性检测、局部释放决策与性能对比。

**📈 对比分析**

方法比较：在八量子 Ising 下，使用共享阴影估计的释放方向仅需约 1×10⁴ 次射线即可得到可靠的曲率估计，而标量探测需要约 6.3×10⁷ 次射线，差距约 6300 倍；在六量子自旋翻转例子中，两复制 SWAP 门的误差稳健在 0.045–0.059 之间，显示出对维数的强鲁棒性。实验结果证明了理论上的置信边界与降维/曲率搜索在实际量子硬件中的可行性。

**⚠️ 局限性**

局限性包括：①对称群需预先给定或通过外部检索得到；②软释放门的置信界依赖于估计误差，在高维稠密字典下仍需更多样本；③目前的证据链主要针对已知标签和可测读出的损失函数，针对非监督或复杂标签结构的扩展仍待研究；④对低阶混合态或非幺正演化的测量模式选择尚未完全覆盖。

---

## 410. Memory as Plans: World-Action Modeling with Memory-Grounded Planning

**arXiv ID:** 2609.11561 | [PDF](https://arxiv.org/pdf/2609.11561v1)

**作者:** Sizhe Zhao `[一作]` (Harbin Institute of Technology), Shengping Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Memory-as-Plans框架MaP‑WAM，分离记忆驱动规划与计划条件执行，实现对长时间记忆的有效利用并保持执行时的固定上下文长度。

**💡 创新点**

创新点在于将长期视觉记忆转换为稀疏视觉计划并与语言计划联合，利用进度感知的WAP模型实现可变时长执行并通过计划-观测对齐校准进度，从而在不随任务历史增长的前提下实现高效记忆依赖控制。

**🔧 技术方法**

采用预训练视觉‑语言模型进行语言规划、因果世界模型（CWM）生成稀疏视觉计划、Mixture‑of‑Transformers（MoT）架构的WAP模型联合预测动作与进度，并使用KV缓存技术保证推理效率。

**📊 数据集**

主要使用RMBench模拟基准（含5个M(1)与4个M(n)任务）以及在7-DoF Franka Research 3机器人上收集的两类真实任务（Find Button、Press Buttons）。

**📈 对比分析**

与DP、π_0.5、X‑VLA、Mem‑0、WLA‑0、LingBot‑VA等基线对比，MaP‑WAM在RMBench上的总成功率达83.3%，在真实机器人任务上分别取得88%和68%的成功率，显著优于现有方法，并保持了相对恒定的执行推理延迟。

**⚠️ 局限性**

限制包括依赖已标注的分段结构，若未提供分段需自动发现；计划‑观测对齐采用简单匹配度，复杂场景下可能需更鲁棒的相似度学习。

---

## 411. A Comparative Evaluation of Pre-trained Convolutional Neural Networks for Melanoma Detection

**arXiv ID:** 2609.11550 | [PDF](https://arxiv.org/pdf/2609.11550v1)

**作者:** Wagner Moreno Schmitz `[一作]` (Universidade Tecnologica Federal Do Paraná), Jefferson Tales Oliva `[通讯]` (Universidade Tecnologica Federal Do Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较了五种预训练CNN（ResNet50、VGG16、VGG19、MobileNet、InceptionV3）在两种影像模态（皮肤镜图像与组织学图像）下的二分类性能。

**💡 创新点**

采用统一的训练协议与Triplet Loss嵌入，再用kNN进行分类，系统性评估不同模型在不同模态间的泛化差异，并使用Friedman与Nemenyi检验确定统计显著性。

**🔧 技术方法**

预训练CNN+迁移学习+监督式度量学习(Triplet Loss)+kNN分类+统计检验。

**📊 数据集**

HAM10000、ISIC 2018（皮肤镜图像）和CR‑AI4SkIN（组织学图像）。

**📈 对比分析**

在相同数据拆分、增广与超参数条件下比较准确率、F1、精确率与召回率；ResNet50在三组数据上总体最好，ISIC 2018上仅ResNet50与InceptionV3差异显著，CR‑AI4SkIN各模型差异更大。

**⚠️ 局限性**

仅做二分类、缺乏多中心外部验证、未探索轻量化模型在设备上的部署性能、模型对不同肤色的鲁棒性未知。

---

## 412. Prompt Revision as a Source of Cultural Bias in Text-to-Image Systems

**arXiv ID:** 2609.11532 | [PDF](https://arxiv.org/pdf/2609.11532v1)

**作者:** Aleksandra Urman `[一作]` (University of Zurich), Joachim Baumann `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对商业文本到图像系统中的提示修订层进行审计，揭示其在多语言、多文化背景下产生刻板偏见的机制。

**💡 创新点**

提出一种基于标记度(CMS)和文化扁平化(CFS)的三步评估框架，并公开了8,960条多语言提示的基准与工具，首次将提示修订层与最终图像输出的因果关联显式化。

**🔧 技术方法**

利用句子嵌入(ALL‑MiniLM‑L6‑v2)、CLIP 嵌入、TF‑IDF 词频分析、VQA 生成描述、Wilcoxon检验、McNemar检验以及对比生成(原始 vs 修订)的开源模型（SDXL, Flux‑2‑Dev）等技术。

**📊 数据集**

使用自建的多语言提示基准(15 种语言、31 语言–语境对)，并收集 DALL‑E‑3、Imagen、GPT‑Image 的修订提示和图像。

**📈 对比分析**

在 CMS 与 CFS 上对三款系统进行对比，发现美国语境标记度最低、芬兰/瑞士语境扁平化最高；通过对比原始与修订提示生成的图像，验证修订层对刻板图像的因果影响；实验显示修订层导致的偏差显著高于模型本身。

**⚠️ 局限性**

局限包括：VQA 描述可能带有自身偏见；基准仅覆盖部分文化，缺乏子撒哈拉、东南亚等区域；只评估三款系统且缺乏更多供应商；因果实验仅在四个英文化境语境与 GPT‑Image 的修订层中展开；并未提供规范化的“公平”表达标准，仍需社区协作确定。

---

## 413. LoopVAE: Recurrent Depth Across Scales for Visual Tokenization

**arXiv ID:** 2609.11516 | [PDF](https://arxiv.org/pdf/2609.11516v1)

**作者:** Zhiying Lu `[一作]` `[通讯]` (University of Science and Technology of China), Zhiying Lu (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LoopVAE，一种将处理核心在不同尺度间共享的循环深度层级自编码器，支持单尺度和多尺度潜在接口；

**💡 创新点**

创新点在于将尺度条件与循环步骤注入共享的核心模块，实现跨尺度参数复用，同时通过学习的尺度与循环嵌入控制重构；

**🔧 技术方法**

采用CNN与Transformer两种核心实现（ConvNeXt式与DiT式），利用自回归循环、输入注入路径、尺度与循环嵌入以及无监督/对抗式训练；

**📊 数据集**

在ImageNet-1k（训练集）与50k验证集的256×256图像上进行训练与评估；

**📈 对比分析**

与SD3、FLUX VAE等基准模型在相同潜在形状下比较，LoopVAE‑CNN在30个训练纪元内获得rFID 0.28、PSNR 32.54 dB、LPIPS 0.048、SSIM 0.91，参数量仅29M，显示出较低参数占比的竞争性表现；

**⚠️ 局限性**

局限性包括：参数共享不一定带来推理效率提升；高分辨率循环计算成本高；实验仅覆盖单个随机种子；缺乏与匹配训练预算的基线对比；对中间输出校准与可早停等方面的进一步验证尚未完成。

---

## 414. Combining Synthetic and Real Data for Low-Resource Historical OCR: A Manchu Case Study

**arXiv ID:** 2609.11495 | [PDF](https://arxiv.org/pdf/2609.11495v1)

**作者:** Yan Hon Michael Chung `[一作]` (Hong Kong University of Science and Technology), Hanlin Wang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在少量真实历史数据与大量合成数据相结合时，如何最优训练不同 OCR 模型（预训练的视觉-语言模型和紧凑的 CRNN）以识别满文手写与印刷文字。

**💡 创新点**

创新点包括：①系统性比较四种训练策略（仅合成、仅真实、联合训练、从合成到真实微调）在同一任务与同一评估集上的效果；②发现对不同模型合成补充的增益不一致，且大模型不一定优于小模型；③通过字典引导的投票融合三种强模型，进一步提升到 98.27% 的单词识别准确率。

**🔧 技术方法**

使用的技术包括：预训练的视觉‑语言模型（LLaMA‑3.2‑11B‑Vision‑Instruct、Pixtral‑12B‑2409、Qwen3‑VL‑8B‑Instruct）通过 LoRA 微调；紧凑 CRNN（9‑层 CNN + 4‑层双向 LSTM + CTC）；联合与顺序训练管道；字典投票规则；以及基于 Manchu 词典的判定。

**📊 数据集**

数据集：60,000 张合成满文单词图像（SYN‑train）；20,306 张真实历史满文单词图像（SCI‑train）来自 SCI‑DB；验证集 SCI‑val（3,359 张）和合成验证集 SYN‑val（15,000 张）；评估集 ARCH‑test（753 张）来自七个清代手稿与印刷文本。

**📈 对比分析**

比较方法：在统一的 checkpoint 选择（SCI‑val）与统一评估（ARCH‑test）上，使用单词准确率（WA）与字符错误率（CER）并给出 95% 自助法置信区间。结果显示：①仅合成训练最高 87.92% WA；②加入真实数据后，多数模型达到 95–96% WA；③CRNN 在加入真实数据后可与大模型相当；④联合与顺序训练无显著差异；⑤投票融合提升至 98.27% WA。

**⚠️ 局限性**

局限性：评估仅在 753 张词图像上，未覆盖更大词汇与更异质的文档；所有真实图像来自同一语料库 SCI‑DB，未检验跨源的一致性；模型仅处理已裁剪的单词，未涉及页布局或端到端 OCR；字典投票受限于字典覆盖率，无法解决未收录的派生/变形形式。

---

## 415. A Training-Free, Alignment-Free Approach to Corporate Intelligence: Application to SEC Filings

**arXiv ID:** 2609.11620 | [PDF](https://arxiv.org/pdf/2609.11620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 416. 3D Euler-Angle Orientation Control for Two-Ray Fading Mitigation in Maritime Air-to-Sea Communications

**arXiv ID:** 2609.11476 | [PDF](https://arxiv.org/pdf/2609.11476v1)

**作者:** Mohammed Bajja `[一作]` (Mohammed VI Polytechnic University), Giuseppe Silano `[通讯]` (Ricerca sul Sistema Energetico S.p.A.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过将UAV姿态（Euler角）作为控制变量，利用两射模型调节机身姿态以最大化海上空中-海两射链路的瞬时信号质量。

**💡 创新点**

1) 在远场小角假设下将两射相位差线性化，得到一个关于欧拉角的仿射约束；2) 推导出闭式最小范数姿态解族；3) 用软最小化技术将离散姿态解平滑为连续参考；4) 在全致动倾斜多旋翼上采用NMPC跟踪该参考。

**🔧 技术方法**

两射模型、远场小角线性化、闭式欧拉角优化、软最小化规则、非线性模型预测控制（NMPC）、MATLAB/MatMPC与qpOASES仿真工具。

**📊 数据集**

本文使用仿真生成的海上空中-海两射场景数据（参数表中给出），并未使用公开数据集。

**📈 对比分析**

与仅pitch调节和零姿态基线进行比较；累计吞吐量提升 11.4%（相较pitch）和 22.2%（相较零姿态），失效率在所有SNR阈值下均优于基线，轨迹跟踪保持不变。

**⚠️ 局限性**

假设海面平坦、反射系数固定、远场小角；未考虑波浪、动态反射、姿态估计误差等实际环境因素；软最小化导致跟踪误差在快速变化的场景中可能增大；需进一步实验验证。

---

## 417. On Identifying Sound Conditions for Frontrunning Resistance

**arXiv ID:** 2609.11535 | [PDF](https://arxiv.org/pdf/2609.11535v1)

**作者:** Sebastian Holler `[一作]` (MPI-SP), Clara Schneidewind `[通讯]` (MPI-SP)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了首个针对智能合约反前置攻击的形式化定义，并基于此开发了安全交互条件合成工具。

**💡 创新点**

创新点在于将前置攻击视为对用户交互的干扰，提出了基于交互策略的模拟安全定义，并给出了可证明的交互条件生成算法。

**🔧 技术方法**

使用符号执行、约束求解器 Z3 以及模拟/理想世界对比的形式化安全框架。

**📊 数据集**

使用了 287 份来自 8 家顶尖审计公司的智能合约审计数据集（共 393 个前置漏洞），以及包含可执行合约的子集。

**📈 对比分析**

与现有的 Nyx、Sailfish 等工具比较，发现它们只能检测不到 55% 的漏洞，而新方法在所有审计的合约中均能生成安全交互条件，检测覆盖率显著提升；在 48 个真实合约上评估，算法平均耗时几分钟，能够发现两起零日漏洞。

**⚠️ 局限性**

局限包括对符号执行的依赖导致在复杂语言特性（如动态合约创建）或未实现的语义时超时、不可满足的约束导致无法生成条件，以及对安全性假设（如关键事件被显式标记）的限制。

---

## 418. From Grid to Chip: Power Architecture, Stability, and Flexibility of AI Data Centers

**arXiv ID:** 2609.11649 | [PDF](https://arxiv.org/pdf/2609.11649v1)

**作者:** Yubo Song `[一作]` (Aalborg University), Subham Sahoo `[通讯]` (Aalborg University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统梳理了AI数据中心作为电网互联计算系统的技术挑战与机遇，提出了从电网到芯片的整体框架，映射了电力传输架构演化，构建了三级（机架、设施、系统）稳定性模型，并归纳了工作负载、能源存储、冷却与功率电子设备在灵活性与稳定性中的作用，最后给出了多层级的稳定性提升策略。

**💡 创新点**

创新点包括：1）首次从电网–芯片的跨尺度视角统一视数据中心，突出电网互联与芯片级电源耦合；2）提出三层级稳定性框架，系统化分析机架、设施与电网层次的动力学耦合；3）整合工作负载调度、储能、冷却与功率电子控制的多源灵活性，并对其在频率、波动与失稳中的贡献进行定量评估；4）基于宽带隙器件与固态变压器的最新技术，描绘未来高压直流电源架构的演进路径。

**🔧 技术方法**

使用的技术包括：
- 高压直流分配（800 V/1500 V DC）与固态变压器（SST）
- 双向桥、共振变换器、三相桥等DC–DC及AC–DC功率电子拓扑
- 宽带隙半导体（SiC、GaN）与超宽带隙材料
- 机架级电容与UPS电池、储能系统（BESS）等能量缓冲
- 负载管理与任务调度算法、功率平滑与时空迁移
- 温度与液冷技术
- 小信号与大信号稳定性分析、阻抗分析、仿真模型（EMT、时域、频域）

**📊 数据集**

所用数据集包括：
- MIT Supercloud GPU功耗曲线（示例CPU/GPU负载波动）
- UK Power Networks 30 min归一化负荷曲线（数据中心电网入口负荷）
- 真实事件记录（Dominion、ERCOT、Meta、Dublin等）用于验证失稳案例
- 公开的AI训练与推理工作负载日志（大规模模型训练、LLM推理）

**📈 对比分析**

比较方法：
- 对比不同功率电子拓扑与高压直流等级在效率、热阻、EMI等指标上的差异
- 小信号稳定性（特征值/阻抗）与大信号恢复范围（相位图/Lyapunov）相结合评估失稳阈值
- 通过仿真验证工作负载平滑、能量缓冲、虚拟阻抗等稳定性提升措施对频率/电压波动、失稳幅度的抑制效果
- 结果显示：引入800 V/1500 V DC可将铜耗降低40‑50%，采用虚拟阻抗可使机架级负载振荡衰减率提升2‑3倍，能量缓冲与负载迁移协同可将系统级失稳发生率降低至原始的10%以下。

**⚠️ 局限性**

局限性：
- 主要为理论与仿真研究，缺乏大规模实装案例验证；
- 失稳分析多基于理想化负载与模型，实际工作负载的随机性与非线性可能导致差异；
- 对未来宽带隙器件、固态变压器等技术的性能预估存在不确定性；
- 论文未给出统一的可量化经济性评估，实际部署成本与收益仍需进一步研究。

---

## 419. Distributed Optimization of Modular Production Systems using Model-based Reinforcement Learning with Inverse Models

**arXiv ID:** 2609.11615 | [PDF](https://arxiv.org/pdf/2609.11615v1)

**作者:** Andreas Schwung `[一作]` (South Westphalia University of Applied Sciences), Dorothea Schwung `[通讯]` (Hochschule Düsseldorf University of Applied Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在模块化制造系统中使用逆模型的分布式基于模型的强化学习框架，能够实现自学习控制。

**💡 创新点**

创新点在于将近似逆模型嵌入策略网络，解耦动作动力学与任务空间学习，使RL仅在目标状态空间中进行训练。

**🔧 技术方法**

使用的技术包括基于世界模型的MBRL、轻量前馈逆模型网络、以及TD3、SAC、DDPG等离线/离线强化学习算法。

**📊 数据集**

使用的实验数据集为在MLPro框架下仿真的Bulk Good Laboratory Plant（BGLP）模拟数据。

**📈 对比分析**

通过与传统不使用逆模型的MBRL进行对比，实验表明逆模型能显著降低溢出量、缩短训练周期并提升整体奖励，尤其对TD3和DDPG等离线方法效果更显著。

**⚠️ 局限性**

局限性包括仅在仿真环境中验证，逆模型对准确性的依赖有限，且在更大规模或真实工况下的鲁棒性尚待进一步评估。

---

## 420. PHAT: PHotonic Accelerator for TFHE

**arXiv ID:** 2609.11613 | [PDF](https://arxiv.org/pdf/2609.11613v1)

**作者:** Guowei Yang `[一作]` (Boston University), Ajay Joshi `[通讯]` (Boston University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种基于光学相变存储器（OPCM）的全同态加密（TFHE）加速器PHAT。

**💡 创新点**

创新点在于设计了多词高精度光学乘法器、twiddle‑stationary 数据流以及基于 BFU 分配的急切调度策略，解决了 OP CM 的编程能耗和傅里叶变换中 Twiddle Factor 访问不平衡的问题。

**🔧 技术方法**

采用光学相变存储器（OPCM）实现的 FFT 单元、电子与光学互联、离散傅里叶变换算法、以及多词乘法、调度与访问-aware BFU 分配技术。

**📊 数据集**

使用 TFHE 标准参数集（I–IV）进行 PBS 测试，并在 Concrete‑ML 框架下评估四个机器学习推理工作负载（XGBoost、NN‑20、NN‑50、NN‑100）。

**📈 对比分析**

与 CPU、GPU、FPGA 及现有 ASIC（Morphling）比较，PHAT 在 PBS 任务上实现 1.39×–1.77× 的吞吐率提升，在 ML 推理上获得 2.14×–5.10× 的速度提升；同时提供详细的功耗和面积分析。

**⚠️ 局限性**

主要限制包括较大的面积与功耗（相较于 ASIC 8.1×面积、10×功耗），以及当前光子集成技术的成熟度和可扩展性问题。

---

## 421. Making Alternative Data Work: Context-Augmented LLMs for Financial Forecasting

**arXiv ID:** 2609.11607 | [PDF](https://arxiv.org/pdf/2609.11607v1)

**作者:** Jihoon Kwon `[一作]` (LinqAlpha), Chanyeol Choi `[通讯]` (LinqAlpha)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出两阶段框架：先用筛选代理识别哪些公司-渠道组合具备信息价值，再用预测代理在LLM上下文中融合财务历史、收益电话记录与替代数据做收入预测。

**💡 创新点**

创新点在于：①利用LLM的上下文学习直接整合多源异构信息；②设计业务驱动的筛选代理剔除无关公司-渠道对；③在预测阶段引入检索工具获取公司与渠道背景，提升解释力；④实现完全不需要任务特定训练的预测。

**🔧 技术方法**

技术手段包括：大型语言模型（GPT‑4）、in‑context学习、两代理架构、检索工具（search）、评估指标FVU与MAE等。

**📊 数据集**

数据集涵盖四种商业替代数据渠道（卡消费、网络流量、客流、KPI预测市场）与FactSet的财报、收益预测及盈利电话记录，测试集为2025年12月–2026年6月间的公司季度数据。

**📈 对比分析**

与历史均值、OLS、GBT、单源LLM、集成LLM和分析师共识等基线比较；在所有渠道上，H+X+Z组合的FVU显著降低，优于传统监督模型，并在多数收入惊喜事件中超过分析师共识。

**⚠️ 局限性**

局限性包括：替代数据覆盖有限、渠道特异性导致泛化挑战、对检索质量的依赖、模型对未见渠道的推断能力未知，以及仍需人工设置筛选与工具访问。

---

## 422. Some results on Archdeacon's conjecture for rotation systems

**arXiv ID:** 2609.11599 | [PDF](https://arxiv.org/pdf/2609.11599v1)

**作者:** Arahat Chikkatur `[一作]` (University of California Los Angeles), Ji Zeng `[通讯]` (Alfréd Rényi Institute of Mathematics)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

验证并证明了Archdeacon猜想在 n≤10 时成立，并给出了非平面四元组数量的渐近下界。

**💡 创新点**

通过将旋转系统的局部子系统分布转化为半正定规划，得到更紧的渐近下界，并首次把该猜想推广到反对称可壳化旋转系统。

**🔧 技术方法**

使用半正定规划（Clarabel）、枚举+对称归一化、局部可实现性判定等技术。

**📊 数据集**

使用所有满足非平面四元组数上限的 n≤9 旋转系统枚举结果，并与已知的十个最小交叉绘图进行对照。

**📈 对比分析**

与 Hill 数值比较，证明至少有 (8/9−o(1))H(n) 个非平面四元组；手工证明 (2/3−o(1))H(n)，在 n=10 时与已知绘图一致；性能主要受枚举规模限制。

**⚠️ 局限性**

结果依赖计算机验证，仅给出渐近界；缺乏针对 n>10 的精确下界；对所有旋转系统的构造改造方法仍不完整。

---

## 423. A Time-Based Readout for Vector-Matrix Multiplication in Fully Analog Memristive SNNs

**arXiv ID:** 2609.11713 | [PDF](https://arxiv.org/pdf/2609.11713v1)

**作者:** Elia Mateu-Barriendos `[一作]` (Universitat Politècnica de Catalunya), Salvador Manich `[通讯]` (Universitat Politècnica de Catalunya)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于电压到时间转换的全模拟Memristive SNN读出电路，实现了VMM输出的时间域编码。

**💡 创新点**

创新点在于彻底避免了传统的电流模式读出和大功耗、面积高的电流放大/缩放电路，只需用电容放电时间编码权重之和，从而显著降低面积并保持能效。

**🔧 技术方法**

使用的技术包括1T1R 交叉阵列、全模拟IF神经元、电压-时间(V2T)转换电路、NMOS/PMOS 电容放电和门控逻辑，以及 130 nm CMOS 版图实现。

**📊 数据集**

通过对 MNIST（Digit）数据集训练的 6410 SNN 进行推理验证，并在 10 1 SNN 原型上进行 post‑layout 仿真。

**📈 对比分析**

与传统基于运算放大器的电流感测读出方案相比，提出的电压‑时间读出在 130 nm 工艺中面积减小约 20 倍、能耗相当（≈ 120 pJ/读操作），且在 MNIST 分类任务上保持相同的准确率。

**⚠️ 局限性**

主要限制包括：V2T 关系为非线性 power‑law，需要在使用前通过电路调节局部线性；对电容放电时间的精度要求高，导致约 13.7 % 的时序波动；以及对大规模阵列时电容和电压参考的共享/分布仍需进一步优化。

---

## 424. An FPTAS for Two-Machine Open-Shop Scheduling with a Single Unavailability Interval

**arXiv ID:** 2609.11693 | [PDF](https://arxiv.org/pdf/2609.11693v1)

**作者:** Hao Lu `[一作]` (Dalian University of Technology), Yong Zhou `[通讯]` (Dalian University of Technology)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了针对两台机器开放式车间调度问题（单固定不可用区间、可恢复模式）的完全多项式时间逼近方案（FPTAS）

**💡 创新点**

创新点在于：①发现并利用作业可划分为四类的结构，将原问题分解为三子流车间子问题；②构建七维伪多项式动态规划（相较于先前的十维），并通过保存原始处理时间而非完成时间来保证可恢复性；③在此基础上采用标准缩放与四舍五入得到FPTAS，首次实现该问题的FPTAS

**🔧 技术方法**

核心技术包括Johnson规则、子流车间的动态规划、状态压缩（七维），以及缩放/四舍五入的逼近方法；同时用解析式恢复整体完成时间

**📊 数据集**

论文为理论性工作，未使用真实数据集，结果以理论复杂度和近似比率呈现

**📈 对比分析**

相较于已有的PTAS（1+ε）和常数近似算法（4/3），所提出的FPTAS在时间复杂度上为O(n^7/ε^6)，实现了1+ε精度；实验验证未给出，性能以理论证明为准

**⚠️ 局限性**

局限性：仅适用于可恢复（resumable）模型；运行时间仍较高，七维状态导致常数项大；未解决非可恢复（non-resumable）模型的FPTAS问题

---

## 425. Warrant Theory

**arXiv ID:** 2609.11667 | [PDF](https://arxiv.org/pdf/2609.11667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 426. Autonomy, Social Norms, and Alignment: Towards a Developmental Framework for Autonomous Artificial Agents

**arXiv ID:** 2609.11660 | [PDF](https://arxiv.org/pdf/2609.11660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 427. Contact-Aware Incremental Model Predictive Control for an Underactuated Aerial Manipulator

**arXiv ID:** 2609.11661 | [PDF](https://arxiv.org/pdf/2609.11661v1)

**作者:** Darwin Liu `[一作]` (Delft University of Technology), Sihao Sun `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于非线性模型预测控制（NMPC）与全身增量非线性动力学反转（INDI）相结合的鲁棒接触感知控制框架，用于标准四旋翼平台搭载单链臂实现空中书写。

**💡 创新点**

创新点包括：① 在NMPC动态约束中加入接触模型并采用接触激活变量，实现对参考接触力的显式跟踪；② 采用层级全身INDI闭环增强对摩擦、风等未建模扰动的鲁棒性；③ 通过无力/扭矩传感器的接触力估计实现“软”接触感知；④ 在垂直、倾斜平面及5 m/s侧风条件下无弹性手爪完成书写。

**🔧 技术方法**

主要技术：非线性模型预测控制、全身增量非线性动力学反转、倾斜优先姿态控制、接触力估计、Simscape仿真、acados NMPC、Raspberry Pi 5 计算平台、Vicon 动作捕捉。

**📊 数据集**

实验数据：使用Vicon摄像机实时姿态、四旋翼惯性传感器数据、环境F/T传感器（Bota SenseOne）验证接触力估计；未使用公开数据集，全部为自建实验数据。

**📈 对比分析**

对比方法：将所提CA‑NMPC‑CINDI与基线NMPC‑CINDI、文献中的接触感知NMPC（CA‑NMPC‑SOTA）以及混合阻尼控制器（HIC）进行比较。评价指标包括接触力RMSE、端执行器平面位置误差、姿态误差以及力跟踪峰值。实验显示，CA‑NMPC‑CINDI在摩擦和风扰动下保持低于20 cm的姿态误差，力跟踪RMSE约0.5–1 N，显著优于其他方案。

**⚠️ 局限性**

局限性：① 仅使用单自由度非弹性臂，未验证重臂或多自由度系统的可扩展性；② 依赖接触力估计，对极端不确定扰动敏感；③ 未实现关节扭矩直接控制，导致关节层与姿态层之间存在偏差；④ 需要手动调节权重，缺乏自动调参机制。

---

## 428. Multimodal Taxonomic Conditioning for Generative Plankton Imagery

**arXiv ID:** 2609.11673 | [PDF](https://arxiv.org/pdf/2609.11673v1)

**作者:** Daniela Ivanova `[一作]` (University of Glasgow), Nicolas Pugeault `[通讯]` (University of Glasgow)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在自动浮游生物图像数据长尾问题下，本文生成了基于分类学嵌入的合成浮游生物图像，并评估其分布相似度和对下游分类器的提升。

**💡 创新点**

创新点包括：① 将排名对比学习扩展到深度且不规则的生物分类树；② 用冻结的 CLIP 文本编码器产生多模态嵌入，解耦分类学学习与图像生成；③ 在扩散模型中采用层级无监督指导，充分利用分类学层级信息。

**🔧 技术方法**

使用的技术包括：CLIP + LoRA 适配、改进的排名对比学习（RINCE 变体）、DiT‑XL/2 扩散变换器、层级无监督引导、Grounding DINO 目标检测。

**📊 数据集**

数据集：Planktonzilla‑17M（用于 CLIP 预训练），以及 WCO L4 Annotated IFCB Training Library（74,181 张图像，145 类，用于生成和评估）。

**📈 对比分析**

通过与 FineDiffusion 和 TaxaDiffusion 基线比较，FID 从 22.43 降至 19.17；在生成数据单独训练分类器时，macro‑F1 从 0.603 提升至 0.664；在稀有类补充（augmentation）方案中，性能与复制实图基本一致，且均优于简单随机复制。

**⚠️ 局限性**

局限性在于对样本极少的类（≤2 张）仍无法超越简单复制；此外，分类学相似度不一定对应视觉相似度，导致某些物种的合成质量受限。

---

## 429. Unmanned Aerial Vehicle Propagation Channel over Vegetation and Lake Areas: First- and Second-Order Statistical Analysis

**arXiv ID:** 2609.11672 | [PDF](https://arxiv.org/pdf/2609.11672v1)

**作者:** Deyvid L. Leite `[一作]` (Federal University of Rio Grande do Norte), Alvaro A. M. de Medeiros `[通讯]` (Federal University of Juiz de Fora)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用 DJI Phantom 3 UAV 及 XBee 915 MHz 模块，对巴西东北部 Caatinga 森林、湖泊及其混合环境下的空地无线信道进行了实验测量，并对大尺度衰落（路径损耗、阴影衰落）与小尺度衰落（Rayleigh、Rice、Nakagami、Weibull 分布）以及多普勒效应（LCR 与 Doppler 频率）进行了统计建模。

**💡 创新点**

创新点在于：①首次在 Caatinga 生态与湖泊环境下系统性地测量 UAV 空地信道；②结合小尺度衰落分布的 KS 检验，证实 Weibull 分布最适合该环境；③通过 LCR 估计多普勒频率和 UAV 速度，验证其在风速变化影响下的可行性；④揭示低空高度湖面上路径损耗指数为负，暗示低空水面对通信有显著抑制作用。

**🔧 技术方法**

使用的技术包括：移动平均滤波分离大/小尺度衰落；最大似然估计求各分布参数；Kolmogorov–Smirnov 检验比较理论与实测 CDF；LCR 理论与实验匹配估算多普勒频率；线性回归估计路径损耗指数与阴影衰落统计。

**📊 数据集**

数据集为 UAV 在三种场景（湖泊、Caatinga、混合）下的实测 RSSI 序列，采样间隔 300 ms，飞行高度 8 m/80 m，速度 1 km/h 与 3 km/h，涵盖约 1200 条数据样本。

**📈 对比分析**

比较方法：将实测的功率样本分布与多种统计分布（Rayleigh、Rice、Nakagami、Weibull）进行 KS 检验，选择拟合度最高的 Weibull；对 LCR 曲线做理论与实验曲线对比，估算多普勒频率和速度。性能表现：在湖面低空路径损耗指数为负，阴影衰落标准差最大；在高空与混合环境下，路径损耗指数为 2–4，阴影衰落标准差较小；多普勒频率估计落在理论区间内，速度估计与实际受风影响一致。

**⚠️ 局限性**

局限性包括：①仅在特定纬度与气象条件下测量，缺乏对季节性或更复杂多变环境的覆盖；②使用单一无人机型号与单频段（915 MHz），对其他频段和 UAV 机型的推广有限；③风速和方向对速度估计影响较大，导致多普勒频率与速度估计的不确定性；④未对非线性多普勒效应或高速飞行场景进行验证。

---

## 430. Learnware and AI Model Management System

**arXiv ID:** 2609.11656 | [PDF](https://arxiv.org/pdf/2609.11656v1)

**作者:** Zhi-Hua Zhou `[一作]` `[通讯]` (Nanjing University), Zhi-Hua Zhou (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 Learnware Dock System (LDS)，将 AI 模型与机器学习生成的规范 (specification) 结合成 learnware，支持跨模型、跨任务的识别、重用和组装。

**💡 创新点**

创新点在于用 learnware 替代传统模型管理，利用可比较、数据保密的规范实现模型的自动识别、组装，并构建统一的模型协作协议，无需访问开发者或用户的训练数据。

**🔧 技术方法**

采用 RKME、PAVE 等规范生成方法，结合核嵌入、参数更新、上界置信度、层级编码、anchor 采样、组织与收缩算法等技术实现 LDS 的高效识别与管理。

**📊 数据集**

实验使用多领域表格数据集、专用 LLM 任务基准（财经、医疗、数学）以及 136 个异构代理 benchmark 进行评估。

**📈 对比分析**

与单模型、随机选择、描述相似度、传统 LLM 等做法对比，LDS 在表格任务上优于无数据共享方案，在 LLM 组装任务中提升约 2-3 分，在代理识别中准确率提升至 90%+，整体表现超过 GPT‑4.1。

**⚠️ 局限性**

当前局限包括：需要进一步提升系统的效率与可扩展性，规范生成与模型更新需同步，系统的可追溯性与人类责任机制尚不完善，且对复杂代理交互的完整规范尚未实现。

---

## 431. When Agents Disagree: Bayesian Backward Reasoning as a Label-Free Anchor for Multi-Agent Collective Decision-Making

**arXiv ID:** 2609.11709 | [PDF](https://arxiv.org/pdf/2609.11709v1)

**作者:** Ken Chen `[一作]` (University of Melbourne), Saman Halgamuge `[通讯]` (University of Melbourne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构造基于贝叶斯逆推的共享反向后验R，并利用Jensen–Shannon散度对多智能体的前向后验进行选择、软加权和融合，以提升多智能体决策性能。

**💡 创新点**

① 引入跨路径一致性锚点——反向后验R；② 用JS散度评估前向后验与反向后验的一致性；③ 提出三种无训练聚合策略（MinJS、FwdJS、LogLin）和轻量级标记校准。

**🔧 技术方法**

贝叶斯逆推、Jensen–Shannon散度、软硬加权聚合、对数线性融合、两阶段反向后验校准；以及LLM前向推理（Tree-of-Thought、Chain-of-Thought等）和概率输出。

**📊 数据集**

DDXPlus（合成诊断基准，49个疾病标签）。

**📈 对比分析**

与随机选择、投票规则、LLM法官以及单向后验融合做对比；在All上LogLin在5个backbone均为最优，在Disagree上均优于最佳投票和法官，提升约1.2–4.7个百分点。

**⚠️ 局限性**

仅适用于离散标签空间；需额外的反向推断步骤；实验局限于单回合同一backbone；对标记数据的依赖性与样本效率未知；未验证在开放式生成任务或混合backbone场景下的效果。

---

## 432. Language-Augmented Semantic Priors for B-Spline Surface Fitting

**arXiv ID:** 2609.11708 | [PDF](https://arxiv.org/pdf/2609.11708v1)

**作者:** Yunzhong Lou `[一作]` (Fudan University), Xiangdong Zhou `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 LASP 框架，将 CAD 过程历史转换为富文本语义描述，利用大语言模型预测 B‑spline 曲面参数作为求解器的语义先验，从而在不改动几何核的前提下实现语义驱动的曲面拟合。

**💡 创新点**

创新点在于：①将结构化语义先验引入传统 B‑spline 拟合；②通过双阶段 LLM 训练捕捉局部几何规律与全局上下文依赖；③在求解器之外提供可解释的参数化引导，避免对几何核的改动。

**🔧 技术方法**

使用技术包括：大语言模型（Qwen3‑14B）双阶段训练、富文本语义推理、文本到结构化文本的生成、两阶段条件生成模型、以及 OpenCascade 等几何求解器。

**📊 数据集**

数据集为扩展版 ABC 数据集，包含约 35,000 条可产生 B‑spline 的建模历史与对应 B‑rep，经过几何验证和数据增强后用于训练和评估。

**📈 对比分析**

通过与 Transformer、GPT‑5、DeepSeek 等基线在参数预测和求解器级拟合（RMS、Hausdorff、Median 等）进行对比，LASP 在参数 F1、误差、拟合质量上显著优于基线，误差降低约 70%‑85%，表明语义先验能显著提升拟合效果。

**⚠️ 局限性**

局限性包括：只针对能产生 B‑spline 的操作（Fillet、Chamfer、Shell），对其它操作的泛化未知；对罕见拓扑类型性能下降；依赖文本生成质量；无法直接修改求解器内部，仅能通过配置逼近语义先验；模型效果受限于训练数据的多样性与覆盖范围。

---

## 433. Bigger than the EAR BOX: A Theory-Grounded Review of XR Accessibility Research for Deaf and Hard of Hearing Communities

**arXiv ID:** 2609.11706 | [PDF](https://arxiv.org/pdf/2609.11706v1)

**作者:** Shuxu Huffman `[一作]` (Gallaudet University), Raja Kushalnagar `[通讯]` (Gallaudet University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对2015–2025年ACM出版的涉及聋人与耳聋（DHH）用户的XR（虚拟/增强/混合现实）研究进行理论驱动的系统综述，提出基于Disability Studies、Deaf Studies与DeafSpace的四维分析框架，并用该框架对53篇论文进行定性编码与分析。

**💡 创新点**

创新点在于：①首次将三大学术传统（残障研究、聋人研究、空间设计）融合，形成对XR可访问性研究的综合理论视角；②构建四维分析框架（访问取向、责任分配、DHH社区概念化、空间感知假设），揭示现有研究中的隐性假设与偏差；③以理论为导向的定性编码方法，提供可复现的审查流程。

**🔧 技术方法**

采用系统检索与筛选（ACM Digital Library关键词检索、三人双重筛选）、编码与可靠性评估（Krippendorff’s alpha）等方法，构建分析框架与编码表进行定性数据分析。

**📊 数据集**

使用的“数据集”是包含53篇ACM论文的文献集合，这些论文涉及XR技术与DHH用户的交互与可访问性研究。

**📈 对比分析**

并未进行实验性性能比较；文章通过对论文的定性编码呈现不同主题与方法比例（如翻译、感官替代、空间重构等），并以表格与柱状图展示随年份与会议的研究趋势与分布。

**⚠️ 局限性**

局限性包括：①样本仅限于ACM出版物，未覆盖IEEE VR、ISMAR等非ACM XR研究；②聚焦于以手语/视觉交流为主的DHH群体，对不以手语为主要交流方式的DHH用户关注不足；③研究团队视角受美国聋人文化影响，可能未充分反映非美洲DHH社区的经验；④未深入探讨硬件设备与音频传递等技术细节对XR可访问性的影响。

---

## 434. Aerodynamic Prior-Free Coordinated Trajectory Generation and Tracking Control for a Tail-Sitter UAV

**arXiv ID:** 2609.11698 | [PDF](https://arxiv.org/pdf/2609.11698v1)

**作者:** Erchao Rong `[一作]` (Sun Yat-sen University), Ximin Lyu `[通讯]` (Sun Yat-sen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对尾蹠式无人机，提出了一个不依赖气动先验的协调轨迹生成与跟踪控制框架。

**💡 创新点**

创新点包括：①在规划阶段使用ϕ-理论模型得到解析的微分平坦映射；②在跟踪阶段采用在线估计单一气动参数的非线性MPC，实时适应飞行中的气动变化；③通过分阶段的气动建模，实现规划与跟踪的高效协同。

**🔧 技术方法**

使用的技术主要有：微分平坦性分析、ϕ-理论气动模型、预测控制（MPC）与在线参数估计、GCOPTER/ L-BFGS 自动微分优化、acados/HPIPM求解器、低级PID混合器。

**📊 数据集**

实验数据集：仿真环境（Gazebo/PX4）下的三种气动模型（Ma、Lyu、ϕ-理论），以及在户外轻风条件下的实测数据（两条圆形和∞形轨迹，速度 8–14 m/s）。

**📈 对比分析**

与 Cascaded‑PID、ϕ‑理论 MPC、以及高保真 MPC 进行对比。结果显示：在所有测试速度下，RMSE 与高保真 MPC 差距 <0.2 m；MaxAE 在 1 m 以内，尤其在尾风条件下保持良好；与传统方法相比，跟踪精度和稳定性均显著提升。

**⚠️ 局限性**

局限性：在极高速度或高攻角区，单参数估计不足以捕捉复杂气动效应，导致 MaxAE 上升；在组合逆风+横风条件下会进入低攻角高刚度区并出现侧滑，失去协同飞行约束；未考虑风速估计，需进一步实现风向感知与补偿。

---

## 435. ZipCodec: Ultra-Low-Frame-Rate Streaming Speech Coding

**arXiv ID:** 2609.11642 | [PDF](https://arxiv.org/pdf/2609.11642v1)

**作者:** Luca Della Libera `[一作]` (Concordia University), Mirco Ravanelli `[通讯]` (Concordia University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一种名为 ZipCodec 的流式神经语音编解码器，能够以 6.25 Hz 的帧率、0.80 kbps 的比特率压缩语音，并实现 160 ms 的理论延迟，支持在消费级 CPU 上实现实时单流推理。

**💡 创新点**

创新点包括：① 极低帧率的流式编解码；② 在 FocalCodec-Stream 的基础上改进架构，采用去掉归一化和位置编码的 ErfFormer Transformer，使用 DynamicErf 激活；③ 使用标量球面量化（SSQ）实现高效的因子化量化瓶颈；④ 在解码阶段通过 160 ms 内的未来上下文实现“延迟感知”解码；⑤ 将大规模 WavLM 第 6 层特征蒸馏与 94k 小时英语语料直接联合训练，从而显著提升语义与音频质量。

**🔧 技术方法**

核心技术包括：WavLM 蒸馏、ErfFormer Transformer（分组查询注意力 + SiLU 反馈网络）、DynamicErf 激活、标量球面量化（SSQ）、流式 Vocos 解码器、噪声与重叠语音数据增强、熵损失、AdamW 优化器、混合精度训练、批量缓存机制。

**📊 数据集**

使用的数据集：训练阶段使用约 94,000 小时的 LibriLight、VoxPopuli、GigaSpeech；解码器训练使用 LibriTTS‑100；评估阶段使用 LibriSpeech、MLS、VCTK、LibriSpeech‑460、IEMOCAP、Speech Commands、SLURP、VoiceBank、Libri2Mix‑100 等多种公开数据集。

**📈 对比分析**

与 EnCodec、AudioDec、HILCodec、Mimi、PAST、FocalCodec‑S@50 等现有流式编解码器以及非流式 FocalCodec@50 进行对比。ZipCodec 在 UTMOS、dWER、说话人相似度、码本利用率、熵以及 RTF 等指标上均优于或匹配同等比特率的流式基线，并在多语言重构、语音转换以及下游任务（ASR、SI、SER、KS、IC、SE、SS）中实现了显著提升；CPU 推理 RTF 为 1.33，p99 延迟低于 160 ms。

**⚠️ 局限性**

局限性包括：模型参数量高达 842 M，虽然能在 CPU 上实时推理但仍占用较多内存；依赖大规模 WavLM 蒸馏，模型对 WavLM 训练分布的依赖可能限制跨域泛化；当前仅在 0.80 kbps 下实现极低帧率，进一步降低比特率或提升多语种覆盖仍有待研究；以及对极端噪声或实时交互场景的鲁棒性尚未完全验证。

---

## 436. Who Bears the Risk When Generative AI Enters Transport? A Distributional Sociotechnical Audit of Algorithmic Equity, Synthetic-Data Validity, and Public Trust

**arXiv ID:** 2609.11611 | [PDF](https://arxiv.org/pdf/2609.11611v1)

**作者:** Amir Rafe `[一作]` (Texas State University), Subasish Das `[通讯]` (Texas State University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文开发并实施了分布式社会技术审计（DSA），评估生成式AI在交通领域的算法公平性、合成数据有效性和公众态度差异。

**💡 创新点**

创新点在于将算法公平性审计、合成数据验证和公众情绪分析三大治理信号整合为连续风险指数（STRI），并通过 Wasserstein‑2 兼容性指数和条件投影MMD进行分布式量化。

**🔧 技术方法**

主要技术包括 Wasserstein‑2 等价分布指数、条件投影最大均值差异（cpMMD）检验、贝叶斯有序逻辑模型与马蹄莲先验，以及特征矩阵的特征值构建的 STRI。

**📊 数据集**

使用的数据集包括：4大 LLM 家族在 12 种人物线索与 4 交通主题下产生的 5,760 条回复；NHTSA FARS 110,001 条致命车祸记录；以及 Pew American Trends Panel Wave 152 的 4,538 名美国成人调查样本。

**📈 对比分析**

比较方法上，Wasserstein‑EDI 显示拥堵定价建议的分布差异最高；cpMMD 检验发现 CART 合成器在所有安全相关维度上显著偏离真实分布，Gaussian copula 则仅呈边缘显著；STRI 的数值范围为 1.07–1.44，且在权重微调下分类层级变化率高达 75%。

**⚠️ 局限性**

局限性包括：人设线索为人工注入，非真实用户身份；判分者间在某些维度上的低或负一致性；Wasserstein‑EDI 的高斯近似误差；cpMMD 仅使用子样本并限制了检验分辨率；跨数据集对齐仅结构化，缺乏个体级匹配；公共态度样本为一般 AI 态度而非专门的交通 AI 采纳度。

---

## 437. LoaDiff: Conditional Generation of Electricity Consumption Time Series for Energy Analytics

**arXiv ID:** 2609.11639 | [PDF](https://arxiv.org/pdf/2609.11639v1)

**作者:** Mariia Baranova `[一作]` (EDF R&D), Themis Palpanas `[通讯]` (Université Paris Cité)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 LoaDiff，一种基于扩散模型的条件生成方法，用于生成完整一年、30 分钟分辨率的住宅电力负荷曲线，并支持对静态家庭属性（如电器拥有情况）和动态外部变量（如温度、日历信息）的可控生成。

**💡 创新点**

创新点在于：
• 将扩散模型应用于长时序（1 年）且细粒度的负荷数据；
• 采用 Transformer 结构（DiT）对年负荷进行二维拆分（按日）并通过自适应层归一化（AdaLN）实现对静态/动态条件的融合；
• 引入 classifier‑free guidance 与温度/日历等多源上下文的条件机制，实现可调控的生成；
• 通过三维评估框架（精度与多样性、隐私、下游任务实用性、条件可控性）对模型进行系统评测。

**🔧 技术方法**

主要技术包括：
• 条件扩散概率模型（DDPM）
• Transformer‑based denoiser（DiT）
• Patchification（按日分块）
• AdaLN 进行条件注入
• Classifier‑free guidance
• 评估指标：1‑NN 二样本检验、FID、ACD、NNDR、NNPriv、TSTR/TRTR/TR+STR 等。

**📊 数据集**

使用了三组数据集：
• CER（爱尔兰 4,225 家住宅，30 分钟采样，约 25,728 步）
• EDF‑1（法国 2,083 家住宅，2024–2025 年，30 分钟采样）
• EDF‑2（SMACH 模拟数据，20,000 家住宅，含丰富的静态与动态变量），其中 EDF 数据不可公开。

**📈 对比分析**

与 GMM、TimeGAN、TimeVAE、Diffusion‑TS、TimeVQVAE、TimeWeaver、EnergyDiff 等基线对比，LoaDiff 在 FID、ACD、1‑NN 评分上取得最优或第二名；在隐私指标 NNDR、NNPriv 上表现优于大多数基线；在两天预测负荷、家电检测等下游任务中，LoaDiff 在 TSTR/TR+STR 模式下均能达到或超过真实数据训练的性能，并在大多数场景下实现最佳平均准确率。总体而言，LoaDiff 在三维评测维度中 consistently 排名靠前。

**⚠️ 局限性**

局限性包括：
• 隐私评估仅采用距离/邻居度量，缺乏正式差分隐私或对抗攻击的证明；
• 温度敏感性实验仅覆盖 ±5 °C 两个极端场景，未全面探测非线性关系；
• EDF 数据不可公开，实验验证受限于公开数据 CER 与内部数据；
• 对长时序的扩散模型仍面临训练成本高、模式崩溃风险等挑战。

---

## 438. Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation

**arXiv ID:** 2609.11638 | [PDF](https://arxiv.org/pdf/2609.11638v1)

**作者:** Jintao Zhang `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Shengshu Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出了两个实时交互式视频系统——Vidu S2-Avatar（可在 720p 解析度下进行实时、可动态更新的数字角色生成）和 Vidu S2-Editing（可在流式视频中实现风格迁移、虚拟试穿、主体替换与背景替换），并进一步探讨了实时立体视频的生成与编辑。

**💡 创新点**

创新点包括：①自回放强化（Self‑Replay Forcing）与混合教师/扩散强制训练，显著提升流式生成的稳定性与指令遵循；②层级式的高效注意力与低位量化线性层加速，构建了在低成本 GPU 上可实现 42 FPS 生成的推理栈；③单步超分辨率重建器（Super‑Resolution Refiner）在保持运动连贯性的同时提升 720p 细节；④视觉语言代理系统（VLM Agentic System）通过实时视觉反馈与提示生成，实现参考图像随时更新与编辑控制。

**🔧 技术方法**

技术栈主要包括：音视频联合扩散 Transformer（DiT）、TurboDiffusion 与 TurboServe 的高效推理优化（SageAttention、SpargeAttention、Sparse‑Linear Attention、低位量化 GEMM、核融合与 CUDA Graph）、自回放强化与混合教师/扩散强制的训练策略、VLM（视觉语言模型）代理以及多 GPU 上下文并行与动态调度。

**📊 数据集**

使用的数据集是从网络直播、谈话视频、单人舞蹈、2D/3D 动画等多源视频中采集并经过剪辑、过滤、语音处理、密集字幕与嵌入的自定义管道，最终得到约 800 万条高质量视频片段；其中 200 万条用于风格迁移、主体替换、背景替换、虚拟试穿等四个编辑子任务；参考图像则来自公开图像库与用户上传的自定义素材。

**📈 对比分析**

评估基于公开基准（StreamAV‑Bench、Sparkle‑Bench、OpenVE‑Bench、RefVIE‑Bench、ViViD）与内部商业对比（Runway、PixVerse、HeyGen 等），在视觉美感、视觉质量、音频质量、语音‑视觉对齐、同步、指令履行、主体与背景一致性等指标上均取得领先。比如在 StreamAV‑Bench 的 VQ、SC、BC 以及 AVAlign、AVSync 方面均比前沿系统高 0.3–0.5 分，Sparkle‑Bench 的 Overall 评分达 3.74，远超 3.5 以上的竞争模型；在 ViViD 试穿评测中 VFID_I 仅 9.95，显著优于 20 以上的对手。

**⚠️ 局限性**

局限性：①目前主要聚焦于谈话头部人物，缺乏对全身大幅运动（如跳舞、跑步）的广泛支持；②分辨率仍限制在 720p，无法满足高分辨率 VR 或 4K 需求；③实时立体化过程中依赖深度估计与视差映射，可能在复杂场景产生伪影；④系统对极低帧率或极高分辨率视频的适应性尚待验证；⑤依赖昂贵 GPU 与专门的推理栈，对边缘设备的可移植性有限。

---

## 439. Comparative Performance Analysis of OTFS and OFDM Modulations for Mobile Wireless Communications

**arXiv ID:** 2609.11623 | [PDF](https://arxiv.org/pdf/2609.11623v1)

**作者:** J. Marcos Leal B. Filho `[一作]` (Federal University of Rio Grande do Norte), Vicente A. de Sousa `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对OTFS与OFDM在多种调制阶数、移动速度、多径数量和均衡器配置下进行定量BER比较

**💡 创新点**

系统化量化OTFS与OFDM在不同场景下的性能边界，首次给出高移动和多径环境下OTFS优势与交叉点

**🔧 技术方法**

采用理想CSI仿真，使用ISFFT/IDZT实现OTFS，配合LMMSE与MP均衡器，并在5G/6G通道模型下进行评估

**📊 数据集**

使用合成的Tapped‑Delay‑Line延迟‑多普勒通道模型，参数取自文中表格，无外部真实数据集

**📈 对比分析**

通过10⁴次蒙特卡洛仿真计算BER，比较不同调制阶、速度、路径数与均衡器的BER曲线；结果显示OTFS在高移动、多径条件下可实现1–4个数量级的BER优势

**⚠️ 局限性**

局限在于假设完美CSI，未考虑PAPR、计算复杂度及实际硬件实现，仅在理想化场景下验证

---

## 440. LangStreet: Persistent Language Fields for Anchor-Decoded Street Gaussians

**arXiv ID:** 2609.11616 | [PDF](https://arxiv.org/pdf/2609.11616v1)

**作者:** Runyi Yang `[一作]` (Sofia University St Kliment Ohridski), Danda Pani Paudel `[通讯]` (Sofia University St Kliment Ohridski)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在给定的anchor-decoded街景高斯场景中，构建了一套持久化的层次语言字段（LangStreet），实现了从视角条件子高斯到永久锚点/槽的语义分配，并支持文本查询；

**💡 创新点**

核心创新在于将视角条件的子高斯视为临时路由器，而把语义归属交给永久锚点与槽；通过渲染器的alpha合成责任积累加性证据、支持感知的锚点对齐完成以及基于锚点的低秩残差编码，实现高精度低存储的语言字段；

**🔧 技术方法**

使用冻结的anchor-decoded Gaussian重建（Octree‑GS/Anchor decoder）、冻结的2D视觉‑语言引擎（SAM‑3/SigLIP‑2）、alpha‑compositing责任、证据累积、锚点对齐完成、低秩SVD残差编码、余弦相似度评分等技术；

**📊 数据集**

在KITTI‑360、Virtual KITTI 2和Waymo三个街景基准上进行评估；

**📈 对比分析**

对照LangSplat、Feature 3DGS、Occam's LGS、LUDVIG、Splat Feature Solver、VALA等方法进行受控对比；基线（base）模型在2D mIoU仅低0.01点、3D mIoU提升显著，同时特征占用仅2.72 GiB（相较于12.90 GiB的max模型），Light版本更省存储但精度略低；整体FPS与内存占用符合实用要求；

**⚠️ 局限性**

受限于固定的几何与2D观测，稀疏槽支持仍是瓶颈；极低码率的anchor‑relative编码会导致细节丢失；方法主要适用于anchor‑decoded表示，无法直接迁移到其他高斯场景；评估仍依赖17类基准，未充分展示语义细粒度性能。

---

## 441. Quasi-static analysis of passive stability in a novel underactuated multi-finger hand

**arXiv ID:** 2609.11579 | [PDF](https://arxiv.org/pdf/2609.11579v1)

**作者:** Léonie Plancoulaine `[一作]` (Nantes Université), Damien Chablat `[通讯]` (Nantes Université)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

研究了一种三指欠驱动手，采用差动弹簧滑块机制，并对圆柱和球形物体的被动稳定性进行准静态分析。

**💡 创新点**

创新点在于将差动弹簧滑块与三维指节旋转耦合，阐明其对稳定抓取位置及可抓取尺寸范围的影响。

**🔧 技术方法**

使用几何建模、力学分析与势能Hessian计算等准静态技术，结合弹簧力学与机械杠杆效应。

**📊 数据集**

未使用公开数据集，仅采用手掌半径、指节长度、弹簧刚度等几何参数。

**📈 对比分析**

通过比较不同物体尺寸与位移下的接触力、指节角度以及Hessian正定性，证明在两种典型抓取姿势下均实现局部稳定。

**⚠️ 局限性**

局限性包括忽略摩擦、单点接触、动力学效应及真实手部不确定性。

---

## 442. Medvedev Logic is Not Decidable. It is π01 -complete. Who Would Have Guessed?

**arXiv ID:** 2609.11576 | [PDF](https://arxiv.org/pdf/2609.11576v1)

**作者:** Pawel Pawlowski `[一作]` `[通讯]` (Ghent University), Pawel Pawlowski (Ghent University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明 Medvedev 逻辑是 Π^0_1‑完全的，从而不可判定且不可递归可枚举。

**💡 创新点**

创新点：引入 Wang–Medvedev 对，将周期瓷贴（Wang 砖）问题与 Medvedev 逻辑的有限模型可满足性关联；通过构造可满足性公式实现从周期瓷贴到逻辑公式的有效归约，首次给出 Medvedev 逻辑 Π^0_1‑完备性证明。

**🔧 技术方法**

技术方法：Kripke 语义、Wang 砖周期化、Wang–Medvedev 代码、可满足性与无穷递归归约、递归可归约（many‑one）以及对 Medvedev 帧的显式构造。

**📊 数据集**

数据集：无实际实验数据，使用理论构造的 Wang 系统、Turing 机代码以及对应的逻辑公式。

**📈 对比分析**

比较方法：与已知的 Π^0_1‑完备问题（空输入下非停机）做可归约；性能不适用于实验评测，而是证明性复杂度（Π^0_1‑完备性）与不可判定性。

**⚠️ 局限性**

限制：方法高度理论化、构造繁复，依赖手工细节；对实现细节缺乏自动化验证，且论文中关于非停机归约的部分依赖通用计算机理论而未在本文完整重构；因此复现难度较高。

---

## 443. Negative Self-Distillation: Learning to Reason by Avoiding Flaws

**arXiv ID:** 2609.11699 | [PDF](https://arxiv.org/pdf/2609.11699v1)

**作者:** Rongcan Pei `[一作]` (University of Virginia), Yu Meng `[通讯]` (University of Virginia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出NSD框架，利用模型自身生成负面条件并对推理错误进行分层无意义训练。

**💡 创新点**

创新点是自标签负面条件 + 动态门控识别错误 + Sigmoid界定无意义惩罚，避免过度训练。

**🔧 技术方法**

采用自我条件化生成负面提示、门控无意义训练（unlikelihood）与 KL 正则以及单向前向推理。

**📊 数据集**

使用数据集：MATH 训练集及七个数学推理基准（AIME、HMMT、AMC、OlympiadBench、MATH-500 等）。

**📈 对比分析**

与OPS D、Intuitor、TTRL 等基线比较，NSD 在 1.7B/4B/8B 模型上平均提升 2.3%/7.5%/6.0%，并保持更高的反思频率和训练效率。

**⚠️ 局限性**

局限性：对极小或弱模型生成负面条件能力不足；在线策略需额外生成负面条件并两次前向推理，增加计算成本。

---

## 444. ActSafeGuard: Differentiable and Training-Aligned Constraint Enforcement for Flow-Matching Policies

**arXiv ID:** 2609.11697 | [PDF](https://arxiv.org/pdf/2609.11697v1)

**作者:** Jianming Ma `[一作]` (Shanghai Jiao Tong University), Yue Gao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在视觉语言动作模型和世界动作模型上，提出了一种可微且训练对齐的安全防护层 ActSafeGuard，能在每一步生成过程中保证硬约束不被违反；

**💡 创新点**

创新点在于通过参数无关的射线缩放（ray‑scaling）算子，在离散流匹配过程中自动调整步长，既实现了严格的硬约束满足，又在梯度传播时形成隐式斜投影，使模型能够在训练阶段学习与可行域边界相切的更新方向；

**🔧 技术方法**

核心技术包括离散流匹配、射线求交算子、隐式斜投影梯度、可微约束尺度因子；同时使用已有的 VLA/WAM Backbone（π_0.5 与 Fast‑WAM）进行微调；

**📊 数据集**

主要使用 RoboTwin 仿真环境中的四个机械臂操纵任务（lift pot、place shoe、hanging mug、place empty cup）以及两条真实机器人任务（pick green cube、guide the ball）来进行验证；

**📈 对比分析**

与 Projection、Truncation、GaugeFlow 以及无约束基线进行对比；ActSafeGuard 在所有实验中均实现 100% 的步骤安全率，并且在任务成功率、MMD 与 LDLJ 等指标上表现至少与最优对手相当，甚至在某些场景下优于无约束基线；

**⚠️ 局限性**

限制在于假设可行域可由线性不等式（凸多面体）表示，并且需要手工提供约束；对于非凸或需要从感知/交互数据自动推断的约束，目前尚无法直接适用。

---

## 445. Musec: MomentUm SpEctral Clipping for Stable Muon-type Training

**arXiv ID:** 2609.11655 | [PDF](https://arxiv.org/pdf/2609.11655v1)

**作者:** Zhuanghua Liu `[一作]` (National University of Singapore), Luo Luo `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Musec，取代 Muon 的谱扁平化为谱裁剪，以保持动量矩阵的谱结构并抑制过大奇异值，从而提升 LLM 训练稳定性。

**💡 创新点**

创新点：①首次在非凸非光滑随机优化场景下为 Muon‑类型算法提供收敛分析，匹配最优的 (δ,ε) Goldstein 极值收敛率；②设计 Soft Musec，利用平滑饱和函数和耦合 Newton–Schulz 迭代实现高效的无 SVD 谱裁剪；③提供架构无关的优化器级稳定化机制，兼容所有矩阵参数。

**🔧 技术方法**

核心技术包括谱裁剪（soft clipping）与耦合 Newton–Schulz 迭代、弱凸性与 Goldstein 极值理论、两层均值采样的随机优化算法。

**📊 数据集**

实验使用 NanoGPT 模型（Small 491M、Medium 613M、Wide 1.63B）在 FineWeb、OpenWebText、C4 三大文本数据集上训练。

**📈 对比分析**

与 Muon、MuonClip、SPECTRA 进行对比；Soft Musec 在更宽学习率范围内保持稳定训练，验证损失与 SPECTRA 相近或更优，在高学习率和大模型下 Muon 与 MuonClip 易发散，Soft Musec 与 SPECTRA 均能平稳收敛。

**⚠️ 局限性**

局限性：需额外的耦合 Newton–Schulz 迭代，尽管在 bfloat16 上表现良好，但在更大规模或不同硬件上可能需要进一步优化；实验仅覆盖解码器 Transformer，未验证对编码器或混合模型的适用性；理论收敛分析基于弱凸假设，真实 LLM 目标可能更复杂。

---

## 446. Self-Supervised Cardiac Phase Detection via Single-Parameter Latent Orbits

**arXiv ID:** 2609.11650 | [PDF](https://arxiv.org/pdf/2609.11650v1)

**作者:** John Bonnici `[一作]` (Imperial College London), Alberto Gomez `[通讯]` (Ultromics Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种自监督单参数潜在轨道模型，在无标注数据下实现心脏相位（ED/ES）检测

**💡 创新点**

将心脏相位视为一维周期信号，强制潜在空间运动为单参数轨道，既保证了解释性又提升定位精度

**🔧 技术方法**

自监督自编码器、潜在结构-运动分解、正弦非线性映射、单参数轨道约束及高斯模糊预处理等技术

**📊 数据集**

EchoNet‑Dynamic 的 A4C 视角心脏超声视频数据集

**📈 对比分析**

与无监督方法（LMP、DDSB）及监督方法比较，ED MAE 2.36 帧（≈46 ms），ES MAE 2.13 帧，显著优于前沿方法；EF 回归误差亦略有提升

**⚠️ 局限性**

仅在 A4C 视角评估，难以推广至其他视角；对极不规则节律的适应性有限；偏差校正需要少量标注样本

---

## 447. RDDMPI: Residual Denoising Diffusion Model for Probabilistic Multivariate Time Series Imputation

**arXiv ID:** 2609.11648 | [PDF](https://arxiv.org/pdf/2609.11648v1)

**作者:** Ramiro Valdes Jara `[一作]` (University of Miami), Adam Meyers `[通讯]` (University of Miami)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出残差去噪扩散模型 RDDMPI，先用预训练的确定性插补器生成基线完成序列，再在残差空间进行条件扩散，完成多变量时间序列缺失值插补。

**💡 创新点**

创新点：①将插补拆解为基线+残差两步，扩散模型只需学习残差纠正，显著降低学习难度；②在残差扩散中加入可靠性感知融合机制，自适应抑制基线误差的传播；③将确定性基线的潜在表示通过 FiLM 注入扩散网络，提升条件信息利用。

**🔧 技术方法**

技术：条件扩散概率模型（DDPM）+ FiLM 加权 + 可靠性门控 (learned reliability map) + 时间/变量 Transformer 结构 + 预训练的确定性插补器（如 T1）作基线。

**📊 数据集**

数据集：五个公开基准，分别为 ETTh1、ETTh2、Weather、Exchange、Illness。

**📈 对比分析**

比较方法：在点缺失（0.2/0.4/0.6/0.8）和块缺失两种场景下，分别与10个基线（DLinear、ModernTCN、iTransformer、SAITS、ImputeFormer、TimesNet、T1、GP‑VAE、CSDI、FGTI）对比。评估指标为 MAE/MSE（确定性）和 CRPS（概率性）。RDDMPI 在 20 个 MAE/MSE 组合中获得 18/20 最佳结果，在 10 个 CRPS 组合中获得 8/10 最佳结果，性能显著优于现有方法。

**⚠️ 局限性**

局限性：需要先训练并冻结确定性基线模型，且逆扩散过程需执行 T 步，生成 N 份样本，导致计算开销大。未来工作需探索加速采样或联合训练以降低成本。

---

## 448. Your Retriever Already Knows: Distribution-Shape QPP for RAG Retrieval Sufficiency

**arXiv ID:** 2609.11646 | [PDF](https://arxiv.org/pdf/2609.11646v1)

**作者:** Matyáš Veselý `[一作]` (Czech Technical University in Prague), Jiří Franc `[通讯]` (Czech Technical University in Prague)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究并比较了基于得分、基于内容以及混合三类查询性能预测(QPP)方法，用于评估检索增强生成(RAG)系统在检索阶段是否足以生成可靠答案。

**💡 创新点**

创新点在于提出了“GeneralQPP”——一个24维非词法特征集，利用检索分数分布形状、查询表面统计和全局相似度信息，且将LLM判别结果作为单一特征融入模型；同时在视觉文档检索场景中首次统一比较三种范式，展示LLM更适合作为特征而非独立预测器。

**🔧 技术方法**

技术方法包括：1）基于分数统计的特征工程（如NQC、WIG、SMV、分布形状特征等）；2）使用多模态LLM（Qwen3.5-35B）进行内容判别；3）训练轻量级分类器（Logistic、Ridge、MLP）通过3折交叉验证进行架构与特征选择；4）对结果进行Platt标定和多指标评估（AUROC、AUPRC、Brier、ECE）。

**📊 数据集**

数据集为：SÚJB（1,510条合成查询，5份核查文件）和ViDoRe V3（14,514条跨域查询，8个企业领域）。

**📈 对比分析**

比较方法共九个，涵盖S0–S4（纯得分）、C1（LLM判别）、H1–H3（混合）。在ViDoRe上，GeneralQPP以0.856 AUROC领先所有纯得分基线，H2（GeneralQPP+LLM）在某些场景略优但差异不显著；在SÚJB上，GeneralQPP与其瘦身版S1-Lean同样表现最佳，H2在Hit@5/10上取得最高AUROC，且在对抗性检索检验中优势更明显。

**⚠️ 局限性**

局限性包括：仅测试单一检索器（Qwen3-VL）与单一LLM判别器，未验证跨检索器泛化；LLM判别器使用JSON解析，可能导致解析失败；S1特征在ViDoRe上仍需进一步跨域验证；对真实用户查询的评估缺失，依赖合成查询。

---

## 449. FedHUR: Learning Hierarchical Utility-Guided Client Relations for Personalized Federated Recommendation

**arXiv ID:** 2609.11632 | [PDF](https://arxiv.org/pdf/2609.11632v1)

**作者:** Mingzhe Han `[一作]` (Fudan University), Tun Lu `[通讯]` (Fudan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了FedHUR框架，通过层次化的效用导向客户端关系学习实现个性化联邦推荐；

**💡 创新点**

创新点在于（1）引入粗细级层次化关系建模；（2）利用图滤波推导的效用查询与学习式检索器，能够在聚合前评估有用信息；（3）兼顾通信与计算成本，保持可扩展性；

**🔧 技术方法**

采用图信号处理/图滤波、k-means层次聚类、随机投影、MLP评分器及协同过滤式参数聚合；

**📊 数据集**

在五个公开推荐数据集上实验：ML‑100K、ML‑1M、Book、BX、Beauty；

**📈 对比分析**

与FedMF、FedNCF、PFedRec、GPFedRec、FedRAP、FedCIA、FedCA等基线进行对比，在Recall@10、MRR@10、NDCG@10上均实现显著提升，尤其在ML‑100K与ML‑1M上性能提升最为突出；

**⚠️ 局限性**

局限性包括：对item embedding的全局聚合依赖，item稀疏或分布极端不均时聚类质量可能下降；额外的层次化过滤和效用查询会增加通信和计算负担；对投影维度和层数的选择较为敏感，需要调优。

---

## 450. A Reusable Framework for Robust Approximation Algorithms in the Interval Uncertainty Model

**arXiv ID:** 2609.11621 | [PDF](https://arxiv.org/pdf/2609.11621v1)

**作者:** Klasing Ralf `[一作]` (University of Bordeaux), Naquin Émile `[通讯]` (University of Bordeaux)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究区间不确定性下的鲁棒优化，并提出将任意近似算法转化为鲁棒近似的通用框架，最终给出了 Weighted k‑Set Cover 的第一套鲁棒近似算法。

**💡 创新点**

创新点在于利用差分近似与三项局部搜索性质（良好潜能、最小移动覆盖、有效好移动检索），实现从传统近似到鲁棒近似的通用转换，并首次完成 Weighted k‑Set Cover 的鲁棒近似。

**🔧 技术方法**

核心技术包括差分近似、近似分离预言机（ASO）、局部搜索算法、LP 放松与改进的 GreedySwap/DoubleApprox 组合，以及对三项性质的严格分析。

**📊 数据集**

论文以理论证明为主，并未使用具体实验数据集，而是给出了通用的算法与性能分析。

**📈 对比分析**

与之前仅在 TSP、Steiner Tree 上给出的鲁棒近似相比，本文提供了 (O(H_k^4 k^2), O(H_k^3 k)) 的 (α,β)-鲁棒近似，并证明其在所有区间不确定性场景下均可保持该性能。

**⚠️ 局限性**

局限在于仅适用于可近似的线性选择问题，并且权重必须不影响可行性约束；对于调度等权重依赖约束的 NP‑难问题尚未涵盖。

---

## 451. MMGait: Benchmarking and Unifying Gait Recognition across Heterogeneous Modalities

**arXiv ID:** 2609.11601 | [PDF](https://arxiv.org/pdf/2609.11601v1)

**作者:** Saihui Hou `[一作]` (Beijing Normal University), Yongzhen Huang `[通讯]` (Beijing Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MMGait大规模多传感器步态识别基准，并基于此提出OmniGait++统一多模态步态识别框架。

**💡 创新点**

创新点包括：①MMGait包含5种传感流、12种模态、1015名身份、十向角、三种行走条件，实现多模态同步对应；②OmniGait++采用私有前端+共享身份编码器+anchor‑guided可变卡尔曼融合，支持单模态、跨模态与多模态三种模式而无需多模型；③在同一checkpoint上完成所有任务，显著降低模型规模与部署复杂度。

**🔧 技术方法**

技术涵盖多模态数据处理、身份对齐与跨模态检索、注意力融合（anchor‑guided multi‑head），以及模态感知归一化（modality‑aware BN）。

**📊 数据集**

使用MMGait数据集（1,015身份、482,327序列，RGB、IR、Depth、LiDAR、4D Radar等12模态）。

**📈 对比分析**

与多任务专家模型（DeepGaitV2‑P3D、CL‑Gait、MultiGait++等）对比，OmniGait++在单模态下与专家相当，跨模态检索与专家竞争，且在多模态融合（尤其是高卡尔曼配置）上优于所有专家，且统一模型在参数和存储上比23个专家显著减少近89%。

**⚠️ 局限性**

局限性包括：仅支持9种image‑style模态，未完全覆盖所有12模态；对低质量模态（如雷达）融合效果仍有限；在极端环境下的跨域迁移性能未充分验证。

---

## 452. OmniKVQuant: KV Cache Quantization for Omni-LLMs

**arXiv ID:** 2609.11582 | [PDF](https://arxiv.org/pdf/2609.11582v1)

**作者:** Suho Yoo `[一作]` (KAIST), Joon Son Chung `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究多模态大语言模型（Omni‑LLMs）中的 KV 缓存量化，提出 OmniKVQuant 框架，解决键漂移和异构值几何问题，实现 2‑bit KV 缓存；

**💡 创新点**

① 对键采用局部窗口最小‑最大缩放动态调整量化范围；② 对值使用模态特定旋转基准，提升量化精度；③ 采用训练无关的设计并提供 Triton 融合解码核；

**🔧 技术方法**

基于 TurboQuant 的 Hadamard 旋转，局部最小‑最大缩放，模态特定 Eigen 分解旋转基准，FlashAttention 计算顺序，Triton GPU 内核；

**📊 数据集**

使用 Qwen2.5‑Omni、Qwen3‑Omni 两个多模态 LLM，在七个音视频基准（WorldSense、DailyOmni、Video‑MME、OmniVideoBench、UGC‑VideoCap、DiaDemBench、video‑SALMONN2）进行评估，校准数据来源于 VGGSound；

**📈 对比分析**

与 FP16 原始 KV 缓存和 TurboQuant（2‑bit）对比，OmniKVQuant 在 Qwen2.5‑Omni 上多选题 98.8%、Captioning 97.3%，在 Qwen3‑Omni 上多选题 90.6%、Captioning 83.4%，显著优于 TurboQuant（仅 58.5%/43.6%）；

**⚠️ 局限性**

仅关注 KV 缓存量化，需额外元数据；对低位宽（1‑bit）效果未知；模态特定旋转需要校准，模型更新时需重新校准；对极端长序列或不同硬件平台的扩展性未验证；

---

## 453. Spectral Adapters for Segment Anything Model-based Segmentation of Colorectal Liver Metastases in Computed Tomography

**arXiv ID:** 2609.11703 | [PDF](https://arxiv.org/pdf/2609.11703v1)

**作者:** Ramtin Mojtahedi `[一作]` (Queen's University), Amber L. Simpson `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究提出两种谱适配器（DiSECT和SiGA），将其集成到Segment Anything Model（SAM）中，用于对对比增强CT图像进行结肠直肠肝转移灶（CRLM）分割。

**💡 创新点**

创新点在于：①利用SVD的前几个奇异向量限定残差更新在谱子空间内，实现极低参数量的适配；②SiGA通过全局门控与实例门控相结合的多层感知机，使适配器能够根据不同病灶的大小、形状和对比度进行实例化调节，从而显著提升分割精度。

**🔧 技术方法**

采用的技术包括：SAM基础模型（ViT-B），两种谱适配器（DiSECT、SiGA）、传统低秩适配器LoRA、量化低秩适配器QLoRA、卷积适配器CAD；训练时使用单点提示策略，推理时采用无提示（no-prompt）模式；评价指标为Dice、IoU和95th百分位Hausdorff距离（HD95）。

**📊 数据集**

使用的数据集为446个多机构来源的门静脉相位对比增强CT扫描（Memorial Sloan Kettering Cancer Center、University of Texas MD Anderson Cancer Center），并补充TCIA公开数据，样本均已人工标注肝转移灶掩模。

**📈 对比分析**

方法比较：在单点提示训练下，SiGA取得DSC 0.77、IoU 0.69、HD95 35.39 mm；在无提示测试集上，SiGA DS 0.76、IoU 0.68、HD95 46.76 mm；与传统3D nnU‑Net基线（DSC 0.758）相当。LoRA、QLoRA和CAD的性能略低，而DiSECT在保持极低可训练参数（0.14 M）后精度最低。计算成本方面，SiGA训练时FLOPs 866 G，推理吞吐量3.46 img/s；QLoRA在量化后实现最快吞吐，但重叠度下降。

**⚠️ 局限性**

局限性包括：仅使用二维切片，未利用三维空间上下文；仅评估门静脉相位，未检验多相或跨站点泛化；谱维度、门控网络容量及适配器秩的选择未做系统搜索；未探索3D SAM或3D卷积网络与谱适配器的结合。

---

## 454. Geospatial AI, Dataverse Metadata, and the Study of Place-Based Government

**arXiv ID:** 2609.11674 | [PDF](https://arxiv.org/pdf/2609.11674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 455. Electroencephalography Signal Analysis for Human Activities Classification: A Solution Based on Machine Learning and Motor Imagery

**arXiv ID:** 2609.11695 | [PDF](https://arxiv.org/pdf/2609.11695v1)

**作者:** Tarciana C de Brito Guerra `[一作]`, Vicente A de Sousa `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过使用消费者级 Mindwave 与科研级 V-AMP 两种 EEG 设备，结合机器学习中的随机森林算法，对同一受试者在实时任务中执行的真实动作与运动想象进行分类与识别，验证其可行性与性能；

**💡 创新点**

创新点在于：①首次在同一实验框架下比较消费者级与科研级 EEG 设备在运动想象分类中的性能；②构建多级随机森林分层分类模型，分别判断运动部位、左右侧与真实/想象三类；③提出三种训练/测试框架以探究个体间与时序间的 EEG 可变性。

**🔧 技术方法**

主要技术包括：Butterworth 带通滤波预处理、统计瞬时特征（1–10阶矩）提取、随机森林模型（含特征选择与网格搜索）以及多级分层分类架构。

**📊 数据集**

使用了 9 名健康受试者在实验室采集的 EEG 数据，V-AMP（16 频道）与 Mindwave（单通道）同步记录，共包含 8 项运动/想象任务，采样率 512 Hz。

**📈 对比分析**

通过三种实验框架（同体同时间、同体不同时间、跨体不同时间）进行评估。框架1下，Mindwave 与 V-AMP 在同体同时间条件下均可达到 94%+ 的分类准确率；框架2和3表现明显下降，说明时序与个体差异对分类难度影响大。

**⚠️ 局限性**

主要局限包括：①对不同时间点采集的数据分类效果差，需进一步提升模型泛化能力；②Mindwave 受限于单通道与干电极，噪声与伪迹较多；③受试者数量有限，尚未验证在更大样本与不同人群（如残障患者）下的鲁棒性。

---

## 456. Structured Transforms for Low-Overhead Quantization of Language Models

**arXiv ID:** 2609.11687 | [PDF](https://arxiv.org/pdf/2609.11687v1)

**作者:** Daria Cherniuk `[一作]`, Ivan Oseledets `[通讯]` (Institute of Numerical Mathematics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

重新设计了基于Kashin分解的LLM权重量化方法，提出改进的贪婪算法与DCT变换，并结合OPTQ、QuIP等技术实现高效的4-bit后训练量化管道。

**💡 创新点**

创新点包括：用符号随机DCT替代密集正交矩阵降低至O(NlogN)；采用分块贪婪更新保证四峰分布并可闭式初始化k‑means；在JAX中实现多GPU兼容管道；在极端配置下表现出更强的数值稳定性。

**🔧 技术方法**

使用技术包括：Kashin分解、DCT（带符号随机）、交替贪婪更新算法、OPTQ式误差补偿、QuIP式无相关预处理、闭式k‑means初始化、JAX实现与多GPU并行、量化评估。

**📊 数据集**

使用数据集：WikiText‑2、C4（评估困惑度），HellaSwag、PiQA、Winogrande（零样本准确率）。模型涵盖OPT、Llama‑2、Mistral、Pythia。

**📈 对比分析**

与RTN、OPTQ、QuIP、QuIP‑RG、QuIP#等基线比较，4-bit通道级量化；在OPT、Llama‑2、Pythia等模型上与这些方法相当或更优，尤其在压力配置（高方差、数值发散）下保持接近FP16基线的PPL和准确率。

**⚠️ 局限性**

局限性：未覆盖激活量化或AWQ、OmniQuant等激活关注方法；旋转预处理的进一步改进与硬件加速（如2‑bit GEMM）尚未实现；超低位预算和激活量化的扩展仍待研究。

---

## 457. Single-Stream Multi-Feature Fusion with Temporal Robustness for Gait Emotion Recognition

**arXiv ID:** 2609.11680 | [PDF](https://arxiv.org/pdf/2609.11680v1)

**作者:** Shirong Lyu `[一作]` (Southwest University), Chengpeng Wang `[通讯]` (Wisesoft Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了单流多特征融合框架 SV-GCN，用于 3D 骨架步态情感识别

**💡 创新点**

① 在浅层嵌入影响特征的 SMEF 模块，实现对齐并早期融合；② 引入时间不变的相对运动特征，消除帧率敏感性；③ 使用全局掩码的 VF-STGCN 处理可变长度序列，并用 Mask GroupNorm 避免统计漂移。

**🔧 技术方法**

基于图卷积网络（GCN）、时空图卷积、通道注意力、Mask GroupNorm、全局掩码、相对运动特征计算、预训练+微调

**📊 数据集**

E‑Gait（情感步态）数据集以及转化后的 NTU RGB+D 大规模骨架动作识别数据集进行预训练

**📈 对比分析**

与现有方法（LSTM、G‑GCSN、ProxEmo、STEP、2s‑AGCN、TNTC、ST‑Gait++、BPM‑GCN）在 E‑Gait 上对比，SV‑GCN 单模型下取得 89.91% 的准确率，略低于 BPM‑GCN 的 90.37%，但性能与主流方法相近，并在不同子集（不同帧率、长度）上表现出更强的鲁棒性。

**⚠️ 局限性**

仍略低于最佳方法，且仅在 E‑Gait 上验证；缺乏跨数据集、跨情绪类别的广泛泛化评估；预训练效果未能充分展示；模型对极端极短序列的表现仍有待提升。

---

## 458. MAPLE: Memory-Augmented Planning with Language and Evolution

**arXiv ID:** 2609.11636 | [PDF](https://arxiv.org/pdf/2609.11636v1)

**作者:** Kesheng Chen `[一作]` (Harbin Institute of Technology), Wenjian Luo `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MAPLE 这一基于 LLM 的优化代理，能够在接收连续的自然语言更新时保持可执行的优化模型、已接受的决策和搜索状态，从而实现动态优化。

**💡 创新点**

创新点在于：①将语言模型与可执行工作台（Workbench）相结合，形成可持续更新的优化状态；②设计 Typed Search‑Space Scaffolding (TSS)、Live Problem Decomposition (LPD) 与 Live State Memory (LSM) 三个模块，实现模型构建、更新定位和历史绑定；③引入基于风险评估的 Restart Selector，决定是否保留或重启进化搜索，提升在任务变更后的适应性。

**🔧 技术方法**

技术包括：大规模语言模型（如 DeepSeek‑V4‑Pro）、自动化程序生成与修复、线性/整数规划求解器、进化算法（GA、NSGA‑II）、Typed Search‑Space Scaffolding、Live Problem Decomposition、Live State Memory、重启策略与评估器。

**📊 数据集**

使用了自定义的 NLDO 基准，包含 15 条任务轨迹、180 条自然语言更新，覆盖选择、排程、排班、路径规划与云资源部署等五类优化任务。

**📈 对比分析**

对比方法包括 ReAct、Persistent ReAct、ORLM、OptiMUS、OR‑LLM‑Agent、OptimAI 等。MAPLE 在所有 15 条轨迹上均完成任务，在线标量质量为 0.951，Pareto 超体积比为 0.875，显著优于基线（例如 Persistent ReAct 仅 0.501 / 0.042）。在保持执行状态、重启策略和历史绑定方面，MAPLE 的表现均优于或相当于最优基线。

**⚠️ 局限性**

局限性包括：①对大型 LLM 的依赖导致算力与成本较高；②TSS 与 LPD 等模块对模型设计要求高，可能在不同领域迁移时需要手工调整；③实验仅覆盖合成数据，真实工业场景的可扩展性与鲁棒性仍需进一步验证；④在极端频繁或大规模更新的环境中，状态管理与重启决策的实时性可能成为瓶颈。

---

## 459. CHERI-D Reincarnate: efficient multicore CHERI temporal memory safety through allocation reincarnation (draft version)

**arXiv ID:** 2609.11590 | [PDF](https://arxiv.org/pdf/2609.11590v1)

**作者:** Yuecheng Wang `[一作]` (University of Cambridge), Simon W. Moore `[通讯]` (University of Cambridge)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在 CHERI 架构上实现了一种名为 CHERI‑D Reincarnate 的机制，通过分配再生（allocation reincarnation）和 ID 隔离（ID quarantine）实现高效、可扩展的时间安全性；

**💡 创新点**

创新点包括：① 将 ID 与对象内联存储，支持多达 1 GiB 大小的对象；② 引入分配再生，避免因 ID 用尽而导致的内存隔离；③ 在多核环境下通过 ObjID 缓存与缓存一致性事件的轻量级映射或 Bloom 过滤实现 ID 的跨核一致性；④ 在硬件与软件之间实现完全分层的硬件‑软件协同，保持 CHERI 的去中心化特性；

**🔧 技术方法**

技术实现包括：CHERI‑Toooba FPGA 软核的硬件扩展（新增 ID 字段、IDMODE/IDLOC 计算逻辑）；LLVM/Clang 对能力格式的支持；CheriBSD 的 Malloc Revocation Shim（MRS）与 jemalloc 的整合，实现立即 ID 增量与 ID 退役；ObjID 缓存的反向映射和 Bloom 过滤实现多核一致性；

**📊 数据集**

使用的数据集包括：SPEC CPU2006 INT、PARSEC、SQLite、NIST SARD Juliet（2422 条 CWE‑415/416 测试）和 MSET（自定义堆 UAF 场景）；

**📈 对比分析**

评估方法：对比 CHERI‑D Reincarnate 与 Cornucopia Reloaded（基线）以及原始 CHERI；通过 FPGA 运行时间、内存扫描/隔离次数、内存占用峰值及 DRAM 访问量评估性能与内存开销。结果显示平均运行时开销约 1.7%（SPEC）、0.7%（SQLite）、0.3%（PARSEC），扫描与隔离次数大幅降低（SQLite 仅 1 次扫描，Cornucopia 267 次），内存隔离占用显著减少（最高峰值下降至 2.1%），DRAM 流量也从 48.2% 降至 6%。

**⚠️ 局限性**

局限性：仅覆盖堆分配，未对栈、内核或嵌套分配器进行保护；支持的对象最大为 1 GiB，超过此大小的对象直接使用页面级保护；FPGA 实验仅使用两核，未验证大规模多核扩展；实现中仍依赖 MRS 包装，需改为原生分配器集成以降低额外开销；

---

## 460. A Dataset and Model for Imputing Water Surface Elevation on a Large and Extremely Sparse Spatiotemporal Graph

**arXiv ID:** 2609.11580 | [PDF](https://arxiv.org/pdf/2609.11580v1)

**作者:** Ruben Cartuyvels `[一作]` (European Space Agency), Diego Fernandez Prieto `[通讯]` (European Space Agency)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了针对亚马逊河流域极稀疏水位序列的图像空间填充基准，并提出了新的双向选择状态空间模型进行水面高度重构。

**💡 创新点**

主要创新在于将空间与时间展平成单一序列、采用子图采样并利用拓扑感知位置编码，显著提升极稀疏条件下的预测性能。

**🔧 技术方法**

使用 Mamba（选择性状态空间模型）结合双向扫描、位置编码与多源元数据嵌入。

**📊 数据集**

使用整合了 SWOT、ICESat‑2、HydroWeb 与 ANA 测站的 19,000 条河段时间序列数据集。

**📈 对比分析**

与多种 GNN、LSTM、GRIN、SPIN‑H、ImputeFormer、KITS 等基线比较，模型在 2023‑2026 期间 RMSE 降低 18–39%，并在覆盖率和精度上均优于 Reach‑Reg。

**⚠️ 局限性**

限制在于仍需依赖高质量的多源数据、对极大规模图的计算成本较高，以及对不同河流网络拓扑的泛化性尚未完全验证。

---

## 461. Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents

**arXiv ID:** 2609.11677 | [PDF](https://arxiv.org/pdf/2609.11677v1)

**作者:** Ruiqing Yue `[一作]` (Chengdu Institute of Computer Applications, Chinese Academy of Sciences), Cong Zuo `[通讯]` (Beijing Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Ecdysis 框架，通过批量跨实例失败聚合和 Failure-Driven Collaborative Refinement (FDCR) 进行 LLM 代理运行时 harness 的自适应进化。

**💡 创新点**

创新点在于将失败证据聚合跨任务，区分模型特定错误与系统性 harness 缺陷，并引入多角色协作诊断机制，显著提升进化效率与泛化能力。

**🔧 技术方法**

使用的技术包括批量跨实例失败聚合、FDCR 多角色协作诊断、编程代理实现 harness 代码修改、评估框架对任务分数的统一量化，以及工具交互与上下文管理。

**📊 数据集**

使用的主要数据集为 AgentBench 与 τ^2-Bench（Airline 与 Retail 子集），每个子集从训练和测试拆分中各采样 20 条任务。

**📈 对比分析**

通过与 Direct、Human-Aug、Self-Evolution 等基线方法对比，并在 Qwen3-8B、Qwen3-14B、Qwen3-32B、MiniMax-M2.7 与 Llama-3.1-8B 等五种 LLM 上评估，Ecdysis 在训练效率上提升 1.84×，推理准确度提升 18.56%，并表现出更强的跨模型泛化和更低的 token 消耗。

**⚠️ 局限性**

局限性包括对任务分布多样性的依赖，聚合失败时可能忽视细粒度实例需求，FDCR 过程受限于代理能力和协作效率，导致潜在的误诊或改进空间受限。

---

## 462. COBRA-Skills: Contextual Bandit-Guided Evolution for Agent Skill Optimization

**arXiv ID:** 2609.11682 | [PDF](https://arxiv.org/pdf/2609.11682v1)

**作者:** Pingchen Lu `[一作]` (Chinese University of Hong Kong), Zhongxiang Dai `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出COBRA‑Skills框架，结合上下文bandit优先级和基于执行证据的技能进化，实现对LLM代理技能的高效、预算化优化；

**💡 创新点**

创新点在于将预算化的顺序优化与动态演化的候选空间耦合，并通过上下文bandit实现评估资源的自适应分配，显著降低评估成本与样本消耗；

**🔧 技术方法**

采用上下文bandit（LinearUCB + 神经奖励预测）做评估优先级，使用再生成、回放突变、交叉等演化算子做技能进化；利用LLM教师模型进行技能生成与修正，使用固定嵌入模型提取技能语义；

**📊 数据集**

六个代理基准（SearchQA、SpreadsheetBench、DocVQA、LiveMath、SocialMaze、ALFWorld）以及三种目标模型（Qwen3.6‑35B‑A3B、GPT‑5.4‑Nano、Gemma‑4‑26B‑A4B‑it）；

**📈 对比分析**

与No Skill、LLM Skill、Trace2Skill、SkillOpt等基线对比，COBRA‑Skills在三目标模型上平均提升约13–27个百分点，且总优化成本比SkillOpt低55–58%，成本/提升点更优，且仅使用50条优化样本；

**⚠️ 局限性**

局限包括对教师模型的依赖、对嵌入模型和探索系数ν的敏感性、以及在更大样本量或极端任务场景下可扩展性和稳健性尚未充分验证。

---

## 463. From Intent to Execution Grant: An Execution-Boundary Conformance Profile for High-Risk AI Actions

**arXiv ID:** 2609.11596 | [PDF](https://arxiv.org/pdf/2609.11596v1)

**作者:** Mengting Wu `[一作]` (Chengdu Havenlon Security Technology Co., Ltd.), Jiang Deng `[通讯]` (Chengdu Havenlon Security Technology Co., Ltd.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 EBL-Core——一种执行边界合规模型，定义了 AI 生成候选行动在获得执行权限前的完整语义契约（ERC）和生命周期规则，并给出了最小参考实现与验证测试。

**💡 创新点**

创新点：
1) 统一了意图、候选、根/操作策略、证据义务、决策推导与执行授权的语义绑定；
2) 明确 ERC 作为决策绑定的“释放-兑现”契约，区分授权与实际执行；
3) 给出多层次合规级别（0–3）与可互操作性分析；
4) 通过正式语义与验证集验证了模型在金融转账等高风险场景中的可执行性。

**🔧 技术方法**

技术：
- 结构化意图与候选对象的可验证化（JSON canonicalization + SHA‑256）；
- 规则式授权评估（可映射到 Cedar / Rego / XACML 等语言）；
- 证明/验证机制（决策推导、ERC 验证）；
- 原子化授权生命周期（grant 状态、线性化红emption 逻辑）。

**📊 数据集**

数据集：
- 基于金融转账场景的测试集：34 条静态向量（合法、非法、边界、缺陷）；
- 15 条生命周期与变异检查（ERC、派生、撤销、续期、消费等）；
- 100 次并发 redemption 与 100 次 revoke–redeem 竞争测试。
未使用公开数据集，仅采用自定义测试向量。

**📈 对比分析**

比较方法：
- 与参考实现（Python 3.13.4 标准库）进行交叉验证，确保决策、理由与 ERC 生成一致；
- 采用同一 canonicalization 与 commitment，保证不同实现间可重复；
- 性能：在提供的测试集中无明显性能瓶颈，所有验证在可接受的 CPU/内存资源下完成（未给出具体延迟数值）。

**⚠️ 局限性**

Limitations：
- 仅覆盖单候选、单使用授权，未覆盖多候选或多次使用的扩展；
- 不保证人类意图、证据真值、根策略正确性、完整仲裁或外部结果的正确性；
- 依赖部署层面的完整仲裁、信任源与 Effector 真实性，模型本身不提供这些保障；
- 参考实现仅为演示，未经过安全审计或生产级压力测试。

---

## 464. Atlas: Efficient Verifiable Semantic Search

**arXiv ID:** 2609.11841 | [PDF](https://arxiv.org/pdf/2609.11841v1)

**作者:** Nikolay Avramov `[一作]` (University Of Toronto), Anwar Hithnawi `[通讯]` (University Of Toronto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文实现了 Atlas，一个基于零知识证明的可验证 HNSW 搜索系统，可在不泄露索引的前提下证明检索结果的正确性。

**💡 创新点**

创新点包括将数据库相关成本迁移至预处理阶段、将 HNSW 重构为固定步长的状态过程以及使用时间步标记批处理将多步证明合并为单一证明，从而显著降低证明成本。

**🔧 技术方法**

采用了多项式交互式 Oracle（PIOP）与 zkSNARK 框架，并结合 cq lookup、permutation、lookup 以及时间步标记的 ZK 约束技术。

**📊 数据集**

实验数据集涵盖 SIFT1M、BIGANN-10M/50M/100M、Deep10M 与 GIST1M 等整数与浮点向量检索基准。

**📈 对比分析**

与 zkRAG、V3DB 等现有可验证检索系统对比，Atlas 在相同召回率下证明时间缩短 2–10 倍、证明尺寸约 16.5 kB，验证时间在 40–2000 ms 之间，可扩展至 100 M 向量。

**⚠️ 局限性**

局限性在于向量量化误差与步长截断会略微降低召回率，且在高维（如 960 维）场景下证明时间显著上升，且对 GPU 加速的预处理有一定依赖。

---

## 465. SpecGuard: Inference-Time Backdoor Detection For Free

**arXiv ID:** 2609.11799 | [PDF](https://arxiv.org/pdf/2609.11799v1)

**作者:** Rui Wen `[一作]` (Institute of Science Tokyo), Zheng Li `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM推理时利用已存在的Speculative Decoding验收/拒绝信号，实时检测模型后门激活

**💡 创新点**

将推理加速技术Speculative Decoding的验证步骤转化为安全监测信号，实现零额外模型计算的后门检测

**🔧 技术方法**

Speculative Decoding、草稿模型与目标模型的接受率统计、基于接受率的阈值或双侧异常评分、窗口聚合、早期位置统计等

**📊 数据集**

ShareGPT、MMLU、GSM8K、TruthfulQA等公开评测集；多种LLM家族（LLaMA、Gemma、Qwen3）及不同规模模型、不同前向参数

**📈 对比分析**

与CleanGen、ONION、输入困惑度等现有推理时后门检测器对比，SpecGuard在单查询AUROC 0.92–0.97、TPR 0.95时FPR 3–10%，与额外生成的检测器性能相当但成本几乎为零；在多模型、多攻击类型、不同温度/参数、适应性攻击下仍保持高检测率

**⚠️ 局限性**

当草稿模型也被后门污染、匹配相同触发器与回复时单侧阈值失效；对极其细腻或“主题引导”攻击的检测灵敏度下降；需要手动校准阈值与窗口大小，且需在已部署Speculative Decoding环境中实现

---

## 466. Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing

**arXiv ID:** 2609.11769 | [PDF](https://arxiv.org/pdf/2609.11769v1)

**作者:** Yi Liu `[一作]` `[通讯]` (University of Science and Technology of China), Yi Liu (University of Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种“受控反转”（Controlled Framing Inversion）基准，用来检验大型语言模型（LLM）在保持事实不变的前提下，识别并逆转新闻文章的框架（评估性词汇、代理实现、信息显著性）。

**💡 创新点**

创新点在于：①引入三种可量化的框架干预（Lexical、Agency、Salience）并记录具体编辑；②将框架识别和逆转拆分为独立评估阶段，揭示两者并非同义；③通过已知的事实集合与编辑日志，将逆转效果直接与原始干预对齐，避免传统“中立性”评估的模糊性。

**🔧 技术方法**

技术方法包括：①用 GLM‑5.2 生成受控框架变体并验证事实完整性；②设计两步接口：Detection (D0) 识别框架类型、方向等；Reconstruction (R0) 在识别结果基础上生成无框架、事实保持的文本；③使用四分类宏 F1、方向准确率、干预逆转率 (IRR) 等指标评估模型性能。

**📊 数据集**

数据集：60 篇英文新闻，分别在三种框架类型、三种强度（低/中/高）下生成共 540 个受控变体；另设 10 篇新闻作为 dev 集用于 prompt 设计；所有变体均记录编辑集 E。

**📈 对比分析**

对比方法：在 DeepSeek‑V4‑Flash、Qwen‑Plus、Kimi‑K2.6 三个 LLM 家族上进行直接推理（无工具/网络）和“本地思考”配置。结果显示：事实保留率约 0.84；干预逆转率仅 0.044–0.068；识别精度与逆转率存在明显分离；强度越高，方向准确率与 IRR 越高，但 IRR 仍低于 0.09。各模型在不同轴（识别、校准、逆转）上表现各异。

**⚠️ 局限性**

局限性：①逆转率仍很低，表明 LLM 在逆转框架方面受限；②受控干预仅覆盖三种框架机制，未包含更细粒度的修辞或语义层面；③实验使用的新闻样本有限，未覆盖不同语言或更大规模数据；④评估指标主要基于已记录编辑，缺乏对生成文本可读性与语义一致性的深入分析。

---

## 467. A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients

**arXiv ID:** 2609.11768 | [PDF](https://arxiv.org/pdf/2609.11768v1)

**作者:** Suwan Wu `[一作]` (Xiaohongshu Inc), Xiaolong Jiang `[通讯]` (Xiaohongshu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个四系数参数化的 per‑token KL 门控框架，并用它统一对比了两种经典门控设计（基于教师熵的 EOPD 与基于教师‑学生不一致性的 ToDi），在 Qwen3‑32B/4B 的短文本分类任务上评估其效果。

**💡 创新点**

创新点在于：①把两种传统门控视为同一四维参数空间中的一维子集，②通过多通道组合和显式偏置提供了更丰富的调节方式；③提出了“均值匹配静态基线”隔离实验，证明动态门控具有独立于平均 KL 比例的优势。

**🔧 技术方法**

使用了 Sigmoid 混合门控函数 σ(a·T_entropy + b·S_entropy + c + d·gap)（T_entropy：token‑级教师熵，S_entropy：样本级教师熵，gap：教师‑学生不一致性），并在此框架下训练 13 组配置。

**📊 数据集**

数据集为 TweetEval 的三种二分类/多分类任务（emotion、hate、offensive），每个任务采用 1–3 token 的简短回复作为训练目标。

**📈 对比分析**

对比方法是：①在同一实现下，比较全参数配置与对应的单系数对齐限制；②在每个配置下训练平均 KL 比例匹配的静态基线。实验结果显示，全参数配置在 36 个可比单元中 33 个赢得平均提升（0.5–1.7pp），动态门控在 26 个隔离单元中 19 个表现优于静态基线。

**⚠️ 局限性**

局限性包括：①仅在单一教师‑学生对（Qwen3‑32B/4B）和短文本分类任务上验证；②未提供多种 seed 的完整网格验证；③对齐限制仅是近似代理，未能直接复现原始方法；④参数选择仅基于手工网格搜索，缺乏自动化或理论依据；⑤缺少代码公开，复现需自行实现。

---

## 468. Visual-SLAM for the detection of hidden tomatoes in greenhouses by Hierarchical Localization and GLOMAPfor robotized harvesting

**arXiv ID:** 2609.11766 | [PDF](https://arxiv.org/pdf/2609.11766v1)

**作者:** Fernando Cañadas-Aránega `[一作]` (Universidad de Almería), Francisco Rodríguez `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用单目RGB摄像头和ROS 2，结合Hierarchical Localization（HLoc）与GLOMAP算法实现了番茄植株群体的三维重建与定位；

**💡 创新点**

创新点在于通过低成本单目视觉实现了对被遮挡番茄的识别和三维建模，显著降低了硬件成本并提升了作物监测的可扩展性；

**🔧 技术方法**

采用的技术包括Intel RealSense D435i（仅使用RGB通道）、ROS 2 Humble、HLoc结构化特征匹配、GLOMAP全局结构光重建、RANSAC与CloudCompare后处理；

**📊 数据集**

使用的数据集为真实温室中番茄群体的Rosbag录制，提取了约1600帧10 Hz RGB图像，构成了HLoc‑GLOMAP重建所需的图像配对文件；

**📈 对比分析**

通过将重建模型的番茄直径、质心位置与姿态与人工测量的地面真值对比，误差分别在±0.2 mm、±0.4 mm和±0.75°，验证了方法的高精度；

**⚠️ 局限性**

局限性包括仅使用RGB图像对光照变化和遮挡敏感、需要人工后处理清洗点云、以及缺乏深度信息导致对复杂三维结构的分辨率受限。

---

## 469. SEED-UMI: Sharing the Exoskeleton between human and robot for onE-to-one Dexterous demonstration

**arXiv ID:** 2609.11753 | [PDF](https://arxiv.org/pdf/2609.11753v1)

**作者:** Tengbo Yu `[一作]` (Peking University), Hangxin Liu `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过让人类和机器人手臂共享同一外骨骼，实现了高效的接触丰富演示收集和对接的多指机器人操控。

**💡 创新点**

提出共享物理测量的接口，将人机映射转化为跨体制的监督，实现无缝的演示收集与策略训练。

**🔧 技术方法**

共设计外骨骼、四杆链接编码器、腕部摄像、双阶段的运动映射训练以及ACT、Diffusion Policy、π_0.5等策略。

**📊 数据集**

在五个真实任务（螺丝驱动、AirPods插入、球投掷、桌面清洁、喷雾器喷雾）上收集了约100条演示，随后在每个任务中进行20次自主回放测试。

**📈 对比分析**

与传统遥操作、仅使用机器人映射的基线对比，平均成功率为70%（仅略低于71.7%），且在30分钟内演示收集数量提升约3倍。

**⚠️ 局限性**

仍需手工调整外骨骼与不同手臂的配合，穿戴舒适度有限，且未集成触觉感知，限制了长期和精细操作的推广。

---

## 470. ORCH: Organizational Principles Enable Collective Intelligence in Embodied AI

**arXiv ID:** 2609.11737 | [PDF](https://arxiv.org/pdf/2609.11737v1)

**作者:** Zhengran Ji `[一作]` (Duke University), Boyuan Chen `[通讯]` (Duke University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了ORCH（Organizing Roles and Coordination Hierarchies）框架，用人类组织理论中的聚合互依赖与顺序互依赖原理构建任务专用的层级组织结构，从而提升大规模异质人工智能集体的协作效果。

**💡 创新点**

创新点在于将人类组织学原理具象化为可执行的管理器模型（水平与垂直），实现自顶向下的多级管理与并行与顺序任务的动态配合；并首次证明组织结构本身在多智能体系统中是一个关键性能维度，而非单纯依赖模型规模。

**🔧 技术方法**

技术包括：基于大型语言模型（LLM）的任务拆解与分配、层级管理器（水平聚合与垂直顺序）实现闭环沟通、LLM+critic循环自动生成组织结构、以及多模型多任务多随机种子下的对比实验。

**📊 数据集**

数据集为扩展版CREW‑Wildfire，包含25个野火响应任务，规模从3到50名异质机器人（消防员、推土机、无人机、直升机）不等，任务涵盖侦查、运输、救援、资源管理、封锁与扑灭。

**📈 对比分析**

实验将ORCH与四个代表性基线（CAMON、COELA、HMAS‑2、Embodied）进行对比，使用8种LLM作为基础模型。结果显示：ORCH在最终得分、执行效率、探索度和计算/通信成本上均显著优于基线；人类设计的组织结构在性能上最高，LLM自动生成（含critic）次之；并且组织优势在不同任务、模型和随机种子上保持稳定。

**⚠️ 局限性**

局限性包括：组织结构在执行前固定，缺乏动态重构；仅考虑聚合与顺序两种互依赖，未覆盖互惠或持续协商；LLM自动生成的组织仍落后于人类设计，需进一步提升生成质量；实验仅限野火响应领域，需验证在其他物理任务中的通用性。

---

## 471. IndicTriMix: Developing Language Identification Datasets and Models for Tri-Language Code-Mixing

**arXiv ID:** 2609.11851 | [PDF](https://arxiv.org/pdf/2609.11851v1)

**作者:** Pruthwik Mishra `[一作]` (Sardar Vallabhbhai National Institute of Technology), Shrikant Malviya `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了在社交媒体三语（英、印、孟或英、印、古吉拉特）代码混合文本中进行token级语言识别的方法，构建并公开了相应的基准数据集并在其上 fine‑tune 了 MuRIL 与 XLM‑RoBERTa 两大 Transformer 模型；

**💡 创新点**

创新点在于：①首次针对三语代码混合进行 token‑level 语言标注；②结合规则生成与 LLM 生成两种方式构造训练集并人工标注，形成丰富多样的混合文本；③公开完整的基准、模型与源码，方便后续研究；

**🔧 技术方法**

主要技术包括：Transformer 预训练模型 MuRIL 与 XLM‑RoBERTa 的微调；序列标注任务设计；子词与词层标签对齐与忽略策略；以及多种训练配置（混合、多语种专属、单语种专属）的实验设置；

**📊 数据集**

使用的数据集为 IndicCMix 的三语平行语料，基于此生成 rule‑based 与 LLM‑based 的 ENG‑HIN‑BEN / ENG‑HIN‑GUJ 代码混合句子，并对 dev、test 进行人工标注；

**📈 对比分析**

评估方法为宏 F1 计算，实验表明 MuRIL 在大多数设置下性能略优于 XLM‑RoBERTa，rule‑based dev 集几乎达到 1.0 的 F1，LLM‑based dev、test 集的 F1 在 0.95‑0.99 之间；

**⚠️ 局限性**

局限性包括：①训练样本分布不均导致少数语言 F1 受限；②拉丁转写相似性导致语言间混淆；③子词对齐策略可能导致部分词汇信息丢失，影响模型训练与评估。

---

## 472. Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs

**arXiv ID:** 2609.11744 | [PDF](https://arxiv.org/pdf/2609.11744v1)

**作者:** Joseph Kanichai `[一作]` (Vrije Universiteit Amsterdam), Animesh Trivedi `[通讯]` (IBM Research Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在vLLM框架上实现并评估了py‑kvcache，一个专门为外部KV缓存设计的Python connector，系统地研究了外部KV缓存在不同存储层级（CPU、NVMe SSD、远程对象存储）与不同工作负载（长文档、对话、生产轨迹）下的性能与适用边界。

**💡 创新点**

创新点包括：
- 通过Pareto前沿提出“break‑even”门限，决定何时将KV缓存读写纳入系统而非直接重算；
- 设计异步直接I/O + 有限共享staging + 调度器感知预加载，显著减少KV数据搬迁在TTFT关键路径上的停顿；
- 对现有外部KV缓存实现进行细粒度性能拆解，阐明批量复制、重叠、存储元数据等因素对TTFT的影响。

**🔧 技术方法**

使用技术与工具：vLLM、KvikIO+libaio异步I/O、GPU‑CPU DMA、KV块哈希链与层级文件系统布局、CPU DRAM缓存、预加载与分布式调度策略、Python profiler/trace、PyTorch 2.11 + CUDA 13.2。

**📊 数据集**

实验数据集与模型：Llama 3.2 3B、Qwen3 4B；合成长文档、ShareGPT、Bailian生产轨迹、SCBench、LongBench；使用单输出token、FP16 KV数据。

**📈 对比分析**

比较方法：对比py‑kvcache、LMCache、llm‑d、vLLM原生offload等四个系统；测量TTFT、吞吐率、缓存命中率、GPU↔CPU与磁盘I/O带宽；实验结果显示：
- 在磁盘仅模式下，py‑kvcache TTFT提升2–2.5倍；
- 加入预加载后再提升1.3–1.6倍；
- 在多层缓存配置（CPU+SSD）下，性能接近或略优于原生vLLM；
- 在高GPU内存或短前缀工作负载中，外部缓存无明显优势。

**⚠️ 局限性**

局限性：
- 仅评估两类硬件（RTX 4000、H100）且单GPU；
- KV块大小固定为256 token，未探讨其他粒度；
- 只使用FP16 KV和单输出token，未测量持续解码吞吐、尾部延迟、能耗或成本；
- 采用CPU I/O路径，未考虑GPU直连NVMe或网络/RAID存储；
- 研究未覆盖多GPU、多节点或分布式部署；
- 采用固定预加载策略，未实现动态自适应调度。

---

## 473. From Parameters to Answers: How LLMs Retrieve and Use Their Internal Knowledge

**arXiv ID:** 2609.11859 | [PDF](https://arxiv.org/pdf/2609.11859v1)

**作者:** Wenkang Wei `[一作]` (University of Science and Technology of China), Xingtong Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在回答问题时，隐藏层对请求信息（路由）与事实内容的依赖如何随层次变化，并通过对隐藏状态进行逐层干预来分离并追踪这两种信息的因果作用。

**💡 创新点**

提出了将路由与内容在隐藏状态中分离并测量其在不同层次上的因果权重的方法，揭示了路由-内容交接点以及不同模型的层级差异。

**🔧 技术方法**

使用层级隐藏状态干预、线性投影测量、匹配长度随机控制、bootstrap统计等技术。

**📊 数据集**

使用国家-洲属关系问答数据集（6个训练/校准、24个验证对，包含不同问法和答案类型）。

**📈 对比分析**

通过在特定层级删除路由或内容方向，并与等长度随机控制比较，量化答案边际和后续知识得分的变化，结果表明路由效应在中后层显著，内容效应在后层持续。

**⚠️ 局限性**

局限性包括仅针对单一知识关系（国家-洲），样本量小、仅使用三种指令模型；路由与内容的线性测量可能不足以捕捉所有信息；干预方向与长度共同变化，无法完全分离；缺乏普适性。

---

## 474. Don't Trust the Super-App: A Case Study of Russia's Max

**arXiv ID:** 2609.11814 | [PDF](https://arxiv.org/pdf/2609.11814v1)

**作者:** Richa Priyanka `[一作]` (University of Michigan), Roya Ensafi `[通讯]` (University of Michigan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对俄罗斯国家背景的超级应用MAX进行逆向与动态分析，系统揭示了超级应用架构所带来的多项主动威胁，包括屏幕截图、访问本地存储、注入 JavaScript、拦截网络流量以及伪造用户身份等能力。

**💡 创新点**

创新点在于：①首次从安全角度将超级应用本身视为攻击主体，系统评估其主动破坏能力；②构建完整的实验管道（Frida Hook、定制 TLS 捕获、二进制 RPC 解码）并实现跨版本的动态追踪；③提出对 Android OS 与应用商店的针对性防护建议。

**🔧 技术方法**

使用技术包括：Android 逆向工程、Frida 动态 Hook、JavaScript Bridge 监听、定制 GOST/TLS 解包、MessagePack 解析、SQLite 数据库聚合与事件关联。

**📊 数据集**

数据集主要为：90 个不同业务场景的 mini‑app（涵盖金融、政府、教育、旅游等），在多台 OnePlus 设备上分别运行多场景交互，收集完整的 Hook 日志与网络流量。

**📈 对比分析**

本文并未与其它超级应用做定量对比，而是通过实验验证了上述能力的可行性和普适性；实验表明多数功能在 80–90% 的尝试中成功，说明该架构普遍存在此类隐患。

**⚠️ 局限性**

局限性包括：①研究仅针对俄罗斯版 MAX，未覆盖其他国家/地区的实现差异；②未能验证这些能力是否被实际恶意使用，仅证明技术可行性；③受限于设备与网络环境，某些功能（如 VPN 检测）可能存在误判。

---

## 475. Understanding Operator Attitudes Toward AI-Supported Decision Making in Maritime Operations

**arXiv ID:** 2609.11805 | [PDF](https://arxiv.org/pdf/2609.11805v1)

**作者:** Doreen Jirak `[一作]` (University of Antwerp), Dirk van Rooy `[通讯]` (University of Antwerp)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过线上问卷调查，向船员与海事专业学生展示两种基于 ECDIS/AIS 的碰撞规避情景，评估其对 AI 辅助决策系统的技术焦虑、信任度与可解释性满意度，并结合情感分析与主题聚类对开放式反馈进行定性探讨。

**💡 创新点**

创新点在于：①首次将技术焦虑、信任与 XAI 评价相结合，以多维度评估海事专业人员对 AI 辅助决策的态度；②使用 Wizard‑of‑Oz 方式呈现 AI 解释，模拟实际桥舷操作；③结合定量量表与定性情感/主题分析，获得对支持与风险的综合视角。

**🔧 技术方法**

技术手段包括：问卷平台 Qualtrics、ATAS、TiA 与改编的 Hoffman XAI 量表、Wizard‑of‑Oz 场景呈现、情感分析算法、主题聚类、Kruskal‑Wallis、配对 t‑检验、Cronbach α 与因子分析。

**📊 数据集**

数据集由 166 位受访者中 66 位完成的问卷构成，包含两组碰撞规避图像（由 MAHI 提供的雷达与 AIS 图像），并记录年龄、海事经验、职级等人口统计信息。

**📈 对比分析**

方法比较：通过配对 t‑检验检验两情景下 TiA 与 XAI 评分差异；Kruskal‑Wallis 检验年龄/经验组对 ATAS、TiA 的影响；因子分析揭示 XAI 量表为多维结构。结果显示：整体信任度稳定；解释满意度与情境相关；无显著年龄或经验差异；Cronbach α 在 0.83–0.94 之间，表明量表可靠性良好。

**⚠️ 局限性**

局限性包括：①样本量有限、完成率低导致可推广性受限；②仅使用静态图像，缺乏动态海况真实感，可能影响沉浸度与响应真实性；③问卷设计可能导致受访者提前退出；④改编量表的内部一致性仍处于中等水平，需进一步验证。

---

## 476. Linear Codes over $\mathbb{F}_{q}+u\mathbb{F}_{q}$ associated with Simplicial Complexes, Their Gray Images, and Subfield Codes

**arXiv ID:** 2609.11783 | [PDF](https://arxiv.org/pdf/2609.11783v1)

**作者:** Ankit Yadav `[一作]` (Indian Institute of Technology Delhi), Ritumoni Sarma `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造了以单/双极限元的 simplicial complex 为定义集的四类线性码，求出了其 Lee 权分布，并通过 Gray 映射得到若干最优/近 Griesmer 的少重码，进一步推导了相应子域码并给出了距离最优性和最小性条件。

**💡 创新点**

创新点在于将 simplicial complex 与环 𝔽_q+u𝔽_q 结合，首次得到四类无限族距离最优且可通过 Gray 映射转化为最优少重码，并首次给出子域码的距离最优性与最小性判定。

**🔧 技术方法**

使用了指数和技巧、追踪函数构造、Gray 映射、以及对码的李氏重量和汉明重量分布的解析。

**📊 数据集**

无外部数据集，全部基于理论构造与符号计算。

**📈 对比分析**

通过与 Griesmer 上界、已知最优码表以及 Magma 计算结果比较，证明所得码在距离、重量分布与最小性方面满足或超出现有最优码性能。

**⚠️ 局限性**

局限性包括仅针对 u^2=0 的环、对定义集的选择有限，且最小性与距离最优性条件较为复杂，需要进一步推广到更一般的环或更丰富的 simplicial complex 结构。

---

## 477. Rapid Learning of Dexterous In-Hand Pen Writing through Real-Time Jacobian Estimation

**arXiv ID:** 2609.11775 | [PDF](https://arxiv.org/pdf/2609.11775v1)

**作者:** Kai Stewart `[一作]` (ETH Zurich), Robert K. Katzschmann `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在不使用手-物体模型、仿真训练或示范数据的情况下，利用在线任务雅可比矩阵估计实现了人形手臂抓持笔后仅靠指尖运动完成在空气和纸面上精准书写的实验。

**💡 创新点**

创新点在于：①将视觉伺服中的递归最小二乘雅可比估计迁移至高自由度、肱内冗余手掌环境；②通过短暂的预激励阶段快速收敛雅可比估计；③使用约束雅可比伪逆结合 nullspace 的抓握正则化，保证抓握稳定并实时自适应；④实现了在 CPU 与普通摄像头条件下的低算力实时控制。

**🔧 技术方法**

主要技术包括：递归最小二乘（RLS）/Kalman 滤波式雅可比估计、Tikhonov 正则化伪逆、姿态拉回增益、PID+前馈误差控制、ArUco 标记视觉跟踪、Kalman 速度滤波以及离散时间闭环执行。

**📊 数据集**

实验数据来自：1）物理 ORCA 手抓持笔并在纸面上书写；2）MuJoCo 模拟环境下的 Shadow Hand 与 Wuji Hand 2；未使用公开数据集，仅依赖自制标记与真实笔。

**📈 对比分析**

与 RL/IL 相关基准对比（如赵等、胡等的三指触觉手模型）以及多组消融实验。性能上，平均平面跟踪误差约 0.64 ± 0.10 mm，鲁棒性强；消融显示在线更新与抓握正则化至关重要；速度提升导致精度显著下降；与 RL 方法相比，消耗更少计算资源且在真实机器人上取得更一致的误差。

**⚠️ 局限性**

局限性包括：①仅控制纸面平面，z‑漂移无法自主调节，需靠纸面上抬或柔性套；②不支持多笔画、需要手臂协同定位；③在速度加快时易失稳；④依赖视觉标记，受光照与背景影响；⑤未使用独立的物理量测量验证书写精度。

---

## 478. Component-Aware Differential Privacy for Federated Multilingual Speech-LLMs

**arXiv ID:** 2609.11762 | [PDF](https://arxiv.org/pdf/2609.11762v1)

**作者:** Jordi Luque `[一作]` (Scientific Research), Aleix Sant `[通讯]` (Scientific Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多语言语音-LLM在联邦学习中使用差分隐私时梯度裁剪失效的问题，并提出α-split方法解决跨组件预算崩溃；

**💡 创新点**

发现并诊断了编码器与LLM更新范数失衡导致的跨组件预算崩溃，提出组件感知的两池α-split裁剪方案，既保持整体ε,δ-DP不变，又显著提升编码器的隐私保护；

**🔧 技术方法**

采用差分隐私联邦学习、Per-layer 与EMA自适应裁剪、α-split两池裁剪、LoRA微调以及Rényi DP计数等技术；

**📊 数据集**

以多语言LibriSpeech（MLS）数据集进行Federated FL实验；

**📈 对比分析**

与Flat、PFL-Uniform、PFL-Dim、PFL-Unif+EMA、PFL-Dim+EMA等方法对比，在Whisper+TinyLlama、Whisper+EuroLLM、Voxtral-Mini-3B三种模型上，α-Split在极端范数失衡时几乎达到Flat的WER并提升编码器隐私，而PFL-Dim+EMA在范数平衡时表现最佳；

**⚠️ 局限性**

α-split需要先测量组件更新范数以设定α，且在范数不太失衡时会过度裁剪编码器；实验仅在MLS上验证，缺乏跨域或更大规模模型的验证。

---

## 479. Reflex-Informed Neuromuscular Reinforcement Learning for Muscle-Driven Locomotion

**arXiv ID:** 2609.11733 | [PDF](https://arxiv.org/pdf/2609.11733v1)

**作者:** Jian Zhou `[一作]` (University of Leeds), Zhi-qiang Zhang `[通讯]` (University of Leeds)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种利用阶段性反射控制器调节四个生物力学意义强的残差参数的肌肉驱动行走强化学习框架，避免直接学习肌肉激活；

**💡 创新点**

创新点在于将强化学习与已有的阶段性神经肌肉反射结构结合，只学习对关键反射参数的状态相关调节，实现了紧凑、可解释的控制接口，同时保留了生理组织；

**🔧 技术方法**

采用Hyfydy/SCONE物理仿真平台搭建的双腿肌肉模型，使用阶段性反射控制器与MPO（Maximum a Posteriori Policy Optimization）强化学习算法，奖励包含速度跟踪与多项生理约束；

**📊 数据集**

实验不使用外部运动捕捉数据，仅通过仿真生成的身体动力学数据与文献中的人体运动参考（kinematic/GRF范围）进行评估；

**📈 对比分析**

与端到端肌肉控制RL（E2E‑RL、DEP‑RL）及CMA‑ES优化的固定反射控制器在三种测试场景（正常行走、足底弱化、外部冲击）下对比；Residual‑Reflex RL在关节运动学、GRF一致性、步态对称性与鲁棒性上均优于基线，且无需重新训练；

**⚠️ 局限性**

局限性包括仅在二维纵向模型验证，残差参数维度固定可能限制更复杂任务的表达；需要手动调节奖励权重；对不同人体形态和三维动力学的泛化尚未评估。

---

## 480. Continuous-Time Acoustic Modelling with Neural Controlled Differential Equations

**arXiv ID:** 2609.11725 | [PDF](https://arxiv.org/pdf/2609.11725v1)

**作者:** Mattias Cross `[一作]` (University of Sheffield), Anton Ragni `[通讯]` (University of Sheffield)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于神经受控微分方程（CDE）的连续时间声学建模方法，用时长信息作为控制路径，替代传统长度规整；

**💡 创新点**

创新点在于把持续时间信息嵌入控制路径，使声学向量场随音素内容与时间共同演化，实现时长感知的连续建模；

**🔧 技术方法**

使用了神经CDE、U-Net、4阶Runge-Kutta数值求解、线性插值、以及传统的匹配与风格迁移技术；

**📊 数据集**

在LJSpeech（中性读书语音）和Emotion Speech Dataset (ESD) 上进行实验；

**📈 对比分析**

通过主观评估（MUSHRA、CMOS-EQ）和客观指标（MCD、log‑F0 RMSE）比较，CDE模型在情感 TTS 中取得最高的 Spearman 相关性和情感表达质量，且在某些配置下优于基线；

**⚠️ 局限性**

局限包括对时步大小和层数的敏感性、对不同情感类别效果不一致、以及在更大规模多说话人数据上的验证尚未完成。

---

## 481. Revisiting Avatar-As-Image: High-Fidelity Registration is All You Need

**arXiv ID:** 2609.11722 | [PDF](https://arxiv.org/pdf/2609.11722v1)

**作者:** Margaret Kostyrko `[一作]` (University of Tübingen), Gerard Pons-Moll `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为稠密三维人体扫描提供一种多阶段优化流水线，实现高保真UV纹理和位移映射的Avatar‑As‑Image表征。

**💡 创新点**

创新点在于使用带符号的风箱数（signed winding numbers）结合三层效率级联实现物理约束的体积内外判定，彻底消除身体与衣物的渗透，并通过粗细分辨率的位移优化实现细节恢复。

**🔧 技术方法**

核心技术包括多视角关键点提升、SMPL‑X参数化的体积拟合、三层效率级联的Winding数计算、粗到细的位移优化与UV映射生成，以及对生成式模型的VAE重构验证。

**📊 数据集**

在六大公开数据集（4D‑Dress、BuFF、CAPE、THuman2.1、2K2K、CustomHuman）上进行评估。

**📈 对比分析**

与IPNet、ETCH、NICP、PTF、RMR等基线相比，身体渗透率和渗透深度显著下降，形状与表面Chamfer误差最低，纹理PSNR高达34.48 dB，显示出明显性能提升。

**⚠️ 局限性**

受SMPL‑X拓扑限制，难以准确重建松散或解耦式服装（如长裙、外套），自我接触区纹理易出现溢写，且未对扫描中的外部物体做专门过滤。

---

## 482. From Open RAN to Open Spectrum: A Programmable, Intelligent Architecture for Multi-Service Spectrum Coexistence

**arXiv ID:** 2609.11843 | [PDF](https://arxiv.org/pdf/2609.11843v1)

**作者:** Michele Polese `[一作]` (Northeastern University), Tommaso Melodia `[通讯]` (Northeastern University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种全新的架构，实现了频谱、服务与基础设施的联合共享，超越了现有的单一频谱分配方案；

**💡 创新点**

创新点在于将O‑RAN控制与频谱共享相结合，推出可插拔的sApp模块、共享基础设施池以及基于数据驱动的RFI建模，实现跨服务多维度协同与经济激励；

**🔧 技术方法**

主要技术包括O‑RAN基准的Spectrum Intelligence Controller (SIC)、可插拔sApp、数据驱动传播模型（DT）、BostonTwin三维城市模型与Sionna射线追踪仿真；

**📊 数据集**

使用的数据集为BostonTwin城市三维模型、Sionna射线追踪引擎（含建筑材质参数）以及美国FCC/NTIA/ITU等公开频谱与服务参数；

**📈 对比分析**

通过将基线独占频谱配置与共享配置（全RF链/仅站点共享）进行对比，利用SINR CDF和有效速率曲线评估，结果显示在共享模式下多服务的中位SINR提升高达12 dB，且空闲时频资源可被高占用率的蜂窝服务充分利用；

**⚠️ 局限性**

局限性包括：尚未实现端到端实验验证；硬件兼容性限制了全RF链共享的适用范围；需要进一步校准DT模型以提升预测准确性；经济可行性依赖于对蜂窝QoS的保障和多服务调度的复杂性。

---

## 483. MotionQ: Operator-Conditioned Motion Quotients for Cross-Observation WiFi Gesture Recognition

**arXiv ID:** 2609.11818 | [PDF](https://arxiv.org/pdf/2609.11818v1)

**作者:** Xiang Zhang `[一作]` (Tianjin University), Meng Li `[通讯]` (Hefei University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 MotionQ，一种在 WiFi 手势识别中对不同传感器布局（即不同观察算子）做自适应的几何条件化运动商（motion quotient）方法，能够在部署变更后保持任务性能。

**💡 创新点**

核心创新是：①建立“公共任务可观测性条件”，说明当不同几何观测算子共享足够任务相关信息时才能实现统一表示；②引入“单链保留”任务等价训练，放弃强制跨算子特征对齐，仅要求不同观测视角下产生相同的标签；③使用两支持运动测度并通过“运动商”消除支持标识的任意性，利用中心矩的置换不变性来表征运动；④在几何条件化的前馈链中插入解析的双向观测向量，从而将空间几何直接映射到特征空间。

**🔧 技术方法**

技术手段包括：双分支（Amplitude-Phase + Doppler Frequency Spectrum）共享编码器；几何条件化的线性投影（双向向量 q）和四项方向基；两支持运动测度的解析解与三阶中心矩；单链保留（Smooth Worst-Suboperator）损失；运动商的置换不变性特征（中心矩）；以及在所有几何假设下的概率平均推理。

**📊 数据集**

实验使用 Widar3.0（6 个手势）和 PerceptAlign（3 个全身活动）数据集，构建多种跨用户、跨环境、跨布局、跨方向、跨位置、跨链接可用性等评估协议。

**📈 对比分析**

与 WiGRUNT、UniFi、GesFi 这三种 WiFi 专用基线以及 CORAL、DANN 这两种通用域泛化方法对比。MotionQ 在所有评估协议下均达成最高平均精度（约 88%），在未见布局、方向外推、链接缺失等极端条件下领先最强基线 10–18 个百分点，证明其对观察算子变化的鲁棒性。

**⚠️ 局限性**

局限性：①两支持运动模型仅适用于短时、单手臂手势，难以捕捉多体位或多部件复杂活动；②假设传感器坐标已知且使用二维双向几何，无法直接扩展到移动设备或三维运动；③在目标算子缺乏观测信息时仍会出现性能下降，无法恢复物理上不可观测的特征；④对传感器位置误差敏感，虽然误差小幅可忽略但大误差会显著影响精度。

---

## 484. Logit Refiner: Improving Visual Autoregressive Models via Intra-Scale Dependency Modeling

**arXiv ID:** 2609.11804 | [PDF](https://arxiv.org/pdf/2609.11804v1)

**作者:** Meimingwei Li `[一作]` (LMU Munich), Björn Ommer `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级自回归模块（Logit Refiner），在已训练的视觉自回归模型（VAR）上进行后期修正，恢复同一尺度内的空间依赖性，从而显著提升图像生成质量。

**💡 创新点**

创新点在于：①识别并纠正了VAR并行内尺度解码导致的“均值场”假设错误；②仅通过冻结原模型特征、引入极少（约10%）参数的自回归推理层，完成对同一尺度内联合分布的建模；③在不改变基础模型结构或重新训练的前提下即可提升多种规模的VAR模型。

**🔧 技术方法**

技术方法包括：Transformer 轻量级自回归细化器、教师强制训练、输入投影与可学习嵌入、因果掩码的自回归采样、冻结主干网络、KV 缓存加速推理。

**📊 数据集**

数据集：ImageNet 256×256（类条件生成）；Infinity 1024×1024 与 FLUX‑6M 文本-图像对（文本到图像生成）。

**📈 对比分析**

与原VAR、M‑VAR、HMAR、HART 等多种基准模型对比；在所有模型规模上均提升 FID 0.16–0.49，且 1.1B 参数模型超越 2B 参数原始 VAR；控制实验表明提升来源于内尺度自回归，而非额外容量或训练；在无指导和有指导两种采样设定下均保持优势。

**⚠️ 局限性**

主要限制是内尺度自回归导致的顺序计算开销；尽管可通过仅在早期尺度应用减少开销，但仍比纯并行解码略慢；未来可探索更高效的并行/非自回归内尺度采样方式。

---

## 485. Thinking with Looped Flows

**arXiv ID:** 2609.11801 | [PDF](https://arxiv.org/pdf/2609.11801v1)

**作者:** Ayhan Suleymanzade `[一作]` (EPFL), Jinwoo Kim `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Looped Flows 的新框架，结合流/扩散模型的概率流去噪与循环递归状态更新，以实现可扩展的推理时间和多样化预测。

**💡 创新点**

创新点：1）在多噪声水平下使用局部去噪目标训练状态化去噪器，促使递归状态在不同时间步保持可用；2）在训练时按时间对齐噪声水平并共享同一噪声样本，增强不同步之间的关联；3）将概率流的连续积分与循环状态耦合，形成可插值的时间网格；4）引入自适应计算时间 (ACT) 自动忽略后期已饱和的步骤；5）在推理阶段采用随机 SDE 积分（γ>0）提升多样性与鲁棒性。

**🔧 技术方法**

技术手段：流/扩散模型的概率流匹配；时间条件的去噪器；共享噪声与插值训练；自适应计算时间；随机 SDE/ODE 数值积分；循环网络（MLP‑Mixer/Transformer）；对比实验使用自我条件（FLM）等。

**📊 数据集**

使用的数据集：Sudoku‑Extreme、Maze‑Hard、ARC‑AGI‑1、ARC‑AGI‑2（四个推理基准）；N‑Queens（8×8/10×10）和Graph Coloring（8/10 顶点）用于多解任务。

**📈 对比分析**

与现有循环模型（TRM、FPRM、HRM、GRAM、PTRM、EqR）对比，单轨迹下在 Sudoku、ARC‑AGI‑1/2 获得最高准确率；在多解任务中超越 GRAM，达到最高覆盖率；ablation 证明时间条件、插值、噪声衰减、随机 SDE 等均对性能有显著贡献；推理时间可通过更细的时间网格或集成多轨迹进一步提升。

**⚠️ 局限性**

局限性：仍需手动设计时间采样与噪声共享策略；梯度截断限制了对极长序列的稳定训练；对更大规模模型和更复杂推理任务的泛化尚未完全验证；实现相对复杂，需要较高的计算资源；当噪声共享过度时可能出现对抗性学习或信息泄漏的风险。

---

## 486. Beyond Word Error Rate: A Switch Aware Evaluation of ASR and Audio Language Models on English Yoruba Code-Switched Speech

**arXiv ID:** 2609.11786 | [PDF](https://arxiv.org/pdf/2609.11786v1)

**作者:** Chibuzor Okocha `[一作]` (University of Florida), Christan Earl Grant `[通讯]` (University of Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对英语–约鲁巴混合语音进行切换感知评估，比较11种ASR与音频LM的性能，提出基于切换点的SETER、SPER等指标；

**💡 创新点**

引入多维切换点诊断指标，揭示传统WER掩盖的边界错误，证明了语言特定和切换局部指标的必要性；

**🔧 技术方法**

使用词对齐、最小编辑距离计算、Unicode NFC归一化、语言标签提取及自动化评分管道；

**📊 数据集**

AfriCodeSwitch英语–约鲁巴代码混合语音语料库（2000条验证集）；

**📈 对比分析**

通过共享deterministic manifest和配套脚本对11个系统进行对比，发现最佳WER模型与音频LM在切换点指标上显著差异；WER与Yoruba错误率无关，SETER/ SPER 在所有系统中表现更能体现代码切换能力；

**⚠️ 局限性**

仅评估单一语料，未进行模型微调；性别样本偏向女性；切换点和语言标签的准确性受自动对齐与标注质量限制；

---

## 487. RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety

**arXiv ID:** 2609.11758 | [PDF](https://arxiv.org/pdf/2609.11758v1)

**作者:** Adithiyan Rajan Indira Saravanan `[一作]` (University of Ottawa), Kathleen C. Fraser `[通讯]` (University of Ottawa)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RAG-Safety-Bench基准，用四种检索条件系统评估检索增强生成（RAG）对LLM安全性的影响。

**💡 创新点**

创新点在于将安全评估拆分为非RAG、oracle、on‑topic、random四个条件，剔除检索器质量的混杂效应，能够明确区分答案可得、主题相关、无关与随机检索对安全的不同贡献。

**🔧 技术方法**

使用RAG、MIRAGE评估模型的功能与安全，结合LlamaGuard、ShieldGemma、WildGuard三种自动安全评估器（LLM-judge），并通过Claude进行查询生成与答案验证。

**📊 数据集**

数据集基于Wikipedia文章，构造987条不安全查询（及其Balanced版本），并使用MIRAGE的1000条查询做对照；检索文档来源均来自Wiki。

**📈 对比分析**

在五个开源LLM（Gemma‑3‑12B‑It、Llama‑3.1‑8B‑Instruct、Ministral‑3‑8B‑Instruct、Qwen‑2.5‑7B‑Instruct、Phi‑4‑14B）下对比非RAG、oracle、on‑topic、random四种条件，发现oracle条件下安全率最高（即最不安全），on‑topic表现模型依赖，Qwen在on‑topic上尤为不安全；准确率与安全率呈负相关。

**⚠️ 局限性**

局限：仅英文、仅Wiki来源、未评估检索器与检索质量对安全的作用、仅测试一次、使用开源小模型、自动评审器主要依赖Claude、未覆盖多语言与更大商业模型。

---

## 488. SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control

**arXiv ID:** 2609.11752 | [PDF](https://arxiv.org/pdf/2609.11752v1)

**作者:** Suwan Wu `[一作]` (Xiaohongshu Inc), Xiaolong Jiang `[通讯]` (Xiaohongshu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 SIRF 模型，通过将平台的风险控制策略在权重中内化，实现一次推理即可给出高精度 verdict-only 判定；

**💡 创新点**

创新在于无人工标注的合成语料（EntiGraph、MAGA、CoT）用于策略内化的 CPT，并配合轻量化 SFT，既满足 1 秒内低延迟、精确阈值，又实现跨域可迁移；

**🔧 技术方法**

使用 EntiGraph 提取规则、MAGA 语义重写、账号级 CoT 级联蒸馏、CPT 继续预训练、SFT 微调、Greedy 解码 + 首词概率阈值等技术；

**📊 数据集**

合成 70M token（含 46.7M 风险推理、11.3M 结构化策略、6.8M 防遗忘通用语料、5.2M 诈骗负样本）做 CPT，评估使用 1000 条生产级账号样本（Black/Gray/White）和 4638 条平衡集，并在 10 个公开基准上检验通用能力；

**📈 对比分析**

在同源 Qwen3-8B-SFT 及多种公开/闭源模型对比中，SIRF-8B-SFT 取得 Black Recall@P95 71.3（+15.1pp），Macro‑F1、Accuracy 与基线相当；部署实现 2 秒以内推理、13% 提示压缩、吞吐提升 14‑23%，且保持低 KV‑cache 使用；

**⚠️ 局限性**

局限包括 Recall@P95 对阈值敏感、仅在特定平台策略下验证、缺乏公开模型与数据、CPT 可能削弱指令遵循、阈值需针对每模型单独设定，且实验仅覆盖单一业务场景。

---

## 489. LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation

**arXiv ID:** 2609.11739 | [PDF](https://arxiv.org/pdf/2609.11739v1)

**作者:** Dongfang Zhao `[一作]` `[通讯]` (University of Washington), Dongfang Zhao (University of Washington)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LOCUS方法，利用低秩子空间对冻结的语言模型骨干进行后训练，使生成长度在保持原始偏好对齐目标不变的前提下显著减少。

**💡 创新点**

将低秩子空间参数化为受约束的设计变量，在不改动损失函数的前提下，通过任务感知的子空间选择实现长度压缩与实用性保持的权衡。

**🔧 技术方法**

采用LoRA低秩适配、原生偏好优化（DPO、DrDPO、SamPO）、任务感知子空间选择、检查点筛选与确认拆分等技术。

**📊 数据集**

使用Anthropic Helpful & Harmless (HH‑RLHF) 对话偏好数据，并在安全拒绝集与 Orca DPO 说明性数据上进行跨任务验证。

**📈 对比分析**

与全参数DPO/DrDPO及官方SamPO在相同基准、检查点、数据拆分与推理协议下对比；在Pythia‑2.8B上长度下降20.7%–39.8%，参数更新仅0.28%；在Qwen2.5‑3B上下降14.9%–17.6%，参数更新0.24%，实用性误差≤1pp。

**⚠️ 局限性**

仅在约3B级decoder‑only模型、贪心推理下验证；子空间搜索离散且粗粒；未评估更大模型、采样推理；跨任务可扩展性有待进一步探索。

---

## 490. Explainability Assistant: A Conversational XAI Interface for Interpreting Energy Consumption Models

**arXiv ID:** 2609.11860 | [PDF](https://arxiv.org/pdf/2609.11860v1)

**作者:** Rodion Krjutškov `[一作]` (Nupp Software), Sofia Yfanti `[通讯]` (Hellenic Mediterranean University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个基于LLM功能调用的对话式可解释AI系统，用于能源消耗预测模型的解释。

**💡 创新点**

利用LLM功能调用实现94%意图解析准确率，无需任务特定微调，并通过结构化JSON透明展示系统动作。

**🔧 技术方法**

使用大型语言模型（Gemini‑2.5‑Flash等）、函数调用接口、FastAPI、Next.js，以及后置解释技术LIME/SHAP、DiCE。

**📊 数据集**

采用真实建筑能源消耗监测数据集以及Heart Disease Cleveland（心脏病分类）数据集。

**📈 对比分析**

与传统Explainer Dashboard对比，专家任务完成率100% vs 93%，意图解析准确率达94%，显著优于先前的76.8%。

**⚠️ 局限性**

受限于专家样本量少，且对话式接口在全局可视化洞察方面仍有不足，需要进一步开展大规模多域验证。

---

## 491. Model-Aware Schedules Improve Generation via Fiberwise Optimal Transport

**arXiv ID:** 2609.11842 | [PDF](https://arxiv.org/pdf/2609.11842v1)

**作者:** Luyi Jia `[一作]` (Ludwig-Maximilians-Universität), Steffen Rulands `[通讯]` (Ludwig-Maximilians-Universität)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于纤维最优输运的模型感知调度构造，利用估计的纤维预测风险与系数路径动力学平衡，在固定的系数曲线上自适应分配时间，提升扩散与流匹配模型的采样质量。

**💡 创新点**

创新点在于将信号/噪声分解的纤维风险作为模型依赖项加入调度优化，得到闭式最优时间分配；发现不同模型、数据集与预测目标下的风险曲线与分配变形在归一化后的参考坐标中呈现共通形状，并提出可直接使用的冻结解析调度模板。

**🔧 技术方法**

采用基于平移不变的对称乘积度量的纤维最优输运来量化预测误差；利用系数路径动力学（平方速度积分）和拉格朗日乘子法得到闭式时间分配；实现时使用U-Net、U-ViT、DiT、InstaFlow等网络结构，并在Diffusion、Flow-Matching框架下训练。

**📊 数据集**

在CIFAR-10和ImageNet-64两个公开数据集上评估，使用50,000样本的FID指标，并在多种采样器（DPM++3M、DDIM、Euler、Midpoint、Heun3等）与NFE（16、32、64）下进行比较。

**📈 对比分析**

与基准Cond-OT、cosine、linear-β等调度相比，模型感知调度在CIFAR-10上DPM++3M 16 NFE提升16.4% FID；在流匹配上CIFAR-10 Midpoint 16 NFE提升38.6%；ImageNet-64上多种采样器均实现1-2 FID点的提升，整体表现优于所有传统基准。

**⚠️ 局限性**

缺点包括：理论上对风险形状共通性未给出严谨解释；只在固定系数曲线下进行一次性估计，未探索动态更新；在大型条件潜在模型（DiT、InstaFlow）上仅做诊断，未完成完整的重新训练与评估。

---

## 492. Topology inside NC$^1$

**arXiv ID:** 2609.11822 | [PDF](https://arxiv.org/pdf/2609.11822v1)

**作者:** Eric Allender `[一作]` (Rutgers University), Alexander Shekhovstov `[通讯]` (Columbia University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了在非均匀模型下，常数宽度多项式规模电路在多项式对数 genus 或厚度为二的条件下恰好能够识别 L（对数空间）语言。

**💡 创新点**

首次将拓扑学中的 genus、交叉数、厚度等概念与电路复杂度联系起来，展示这些限制并不提升常数宽度电路的计算能力，并证明厚度二即可完成 L 的全部计算。

**🔧 技术方法**

采用了图论的分解与嵌入技术（利用 genus 的可加性、层化图的平面子图分段、递归路径判定）、Hansen 的平面常数宽度电路与 L 的对应关系、3 页嵌入与三栈机器的等价性，以及 Barrington 的置换分支程序论证。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

通过包含关系与构造证明与已知 L 的电路/分支程序模型对比，未涉及数值性能测试。

**⚠️ 局限性**

仅在非均匀电路模型下成立，缺乏对均匀电路或其他 planarity 泛化（如交叉数、genus）的完整表征，且未给出对 L 的下界或新的上界。

---

## 493. Generative Late-Interaction Embeddings For Visual Document Retrieval

**arXiv ID:** 2609.11808 | [PDF](https://arxiv.org/pdf/2609.11808v1)

**作者:** Mohamed Eltahir `[一作]` (King Abdullah University of Science and Technology), Naeemullah Khan `[通讯]` (King Abdullah University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后置压缩Late‑Interaction检索方法GLIE，通过学习每页少量球面锚点并在检索时利用生成式解码器按需重建完整向量集合，显著降低存储需求；

**💡 创新点**

创新点在于发现页面向量集为低维单位球面流形，利用球面锚定纠正k‑means的最大相似度低估，并设计可训练的代码与共享生成解码器，实现压缩后可恢复完整检索精度；

**🔧 技术方法**

使用的技术包括：冻结的多向量编码器（如ColPali、ColQwen2、Nemotron），k‑means聚类与球面归一化，基于注意力的代码微调，生成式解码器（共享参数），MaxSim评分与列表级KL、Chamfer等多任务损失；

**📊 数据集**

主要在ViDoRe v1与v2（10+4个子集）评估，使用ColPali训练集5000页进行codec拟合；还在ColQwen2上验证；

**📈 对比分析**

与多种基线对比（原始k‑means、token pooling、Light‑ColPali、MetaEmbed等）后，GLIE在k=4时保持约80%无压缩nDCG@5，仅占1KB存储，且在所有子集和预算下均优于此前最强基线，甚至在相同训练预算下优于encoder fine‑tuning；

**⚠️ 局限性**

局限在于对特定后置压缩方案的依赖，解码器质量受k限制，尚未在视频或更大规模文档上验证；需要更大decoder提升极限，且对极低预算（k<4）收益有限。

---

## 494. Dynamic language model representations for multi-objective reaction optimisation

**arXiv ID:** 2609.11790 | [PDF](https://arxiv.org/pdf/2609.11790v1)

**作者:** Joshua W. Sin `[一作]` (F. Hoffmann-La Roche AG), Philippe Schwaller `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了基于动态语言模型表示的多目标化学反应优化框架 Alice，直接使用文本描述反应条件并与高斯过程耦合进行贝叶斯优化，解决了传统特征工程的瓶颈。

**💡 创新点**

创新点在于通过动态学习反应表征，消除了对预先定义的描述符或手工特征工程的需求，实现了跨类别、化学异质性反应空间的统一、高效表征。

**🔧 技术方法**

技术实现包括 LoRA 参数高效微调的预训练语言模型（如 T5-base）与多目标高斯过程的联合训练，配合多目标贝叶斯优化和批量采样。

**📊 数据集**

使用的数据集包括公开的多目标 Suzuki 与 Buchwald–Hartwig 反应数据集、虚拟的硫酰胺偶联 21,648 条实验、以及两项真实实验（钯催化氰化与不对称酮加氢）设计空间。

**📈 对比分析**

与 Kraken/COSMO‑RS 描述符库和 one‑hot 编码对比，评估指标为归一化超体积；在低数据和 96‑孔并行实验中，LLM‑GP 框架在 90%–95% 超体积阈值下所需实验迭代分别约为传统方法的一半或两倍，性能显著优于基线。

**⚠️ 局限性**

局限性包括：仍需一定量实验数据来微调模型；对极少量数据或极高维搜索空间的适应性尚未验证；模型可解释性相对较弱，缺乏对化学机制的直观洞察。

---

## 495. RAGTIMER 1.0: Rapid Rare-Event Partial State Space Construction for Stochastic VAS (extended version)

**arXiv ID:** 2609.11789 | [PDF](https://arxiv.org/pdf/2609.11789v1)

**作者:** Landon Taylor `[一作]` (Utah State University), Zhen Zhang `[通讯]` (Utah State University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发并实现了名为1.0的工具，用于在连续时间随机向量加法系统（CTSVA）中构建部分状态空间并给出罕见事件的下界概率；该工具通过强化学习驱动的轨迹生成、并行与循环扩展（Cycle & Commute）以及前缀树存储实现高效、内存友好的状态空间构造；最终生成的部分空间可导出为主流PMC工具可读格式。

**💡 创新点**

核心创新包括①基于依赖图与强化学习的随机轨迹生成，显著提升了罕见事件概率下界；②使用前缀树实现显式状态与轨迹存储，极大降低内存占用；③设计了易于使用的自定义输入语言，降低了非专业人士的学习门槛；④在Rust中实现所有关键组件，提升了安全性和性能；⑤引入Cycle & Commute对并行与循环行为进行扩展，进一步提高下界精度。

**🔧 技术方法**

主要技术手段为：概率模型检查（PMC）与显式状态空间构造；强化学习用于动态调整轨迹生成的转移权重；静态依赖图分析筛选必需转移；Cycle & Commute算法自动生成并行与循环轨迹；前缀树结构用于存储状态与轨迹；Rust语言实现全程优化；以及对目标状态与吸收状态的处理以保证下界有效性。

**📊 数据集**

实验使用了四个真实的罕见事件化学反应网络（CRN）模型：Modified Yeast Polarization (MYP)、Enzymatic Futile Cycle (EFC)、Simplified Motility Regulation (SMR) 以及 Single‑Species Production Degradation (SSPD)。这些模型均为无限或大规模状态空间，且包含显著的并发与循环特征。

**📈 对比分析**

通过与Storm、Stamina、wSSA、WE、FAU、BMC等现有工具在同一硬件（AMD Ryzen Threadripper 12‑核 3.5 GHz、132 GB RAM）上进行对比，评估指标包括概率下界、状态数、运行时间及内存占用。1.0在所有四个模型上均实现了比Storm等传统PMC工具更优或相当的下界，并在MYP、SMR等复杂模型上显著超越模拟方法（如wSSA、WE），在时间和内存上保持在几秒到几分钟、几十兆字节的范围内。

**⚠️ 局限性**

局限性主要有：①仅提供概率下界，无法得到精确概率；②轨迹生成基于随机采样，虽通过RL提升效率但不保证最优；③仍需手动调节最大循环长度与并行深度，参数选择影响性能；④对极大或不同类别模型的扩展仍需进一步验证；⑤目前仅支持CTSVA/CRN模型，其他类型系统的适配需要额外工作。

---

## 496. The widening evaluation gap in medical large language model research 2023 to 2026

**arXiv ID:** 2609.11770 | [PDF](https://arxiv.org/pdf/2609.11770v1)

**作者:** Raad Bin Tareaf `[一作]` (XU Exponential University of Applied Sciences), Samia Loucif `[通讯]` (Zayed University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统地检索并量化了2023-2026年间 PubMed 中关于生成式语言模型（LLM）在医疗领域的文献，构建可复现的证据图谱，测算评估滞后（evaluation lag）、研究设计对滞后的影响、模型选择与研究内容的偏倚，并提出关注‑证据差距与两大文献社群的结构；

**💡 创新点**

首次量化模型评估滞后随时间扩大的趋势，并用漂移分解（shift‑share）揭示研究迁移至新系统仅弥补约一半的机械滞后；通过设计差异的中位数回归揭示随机对照试验与其他设计之间的系统性滞后差异；引入关注‑证据差距衡量各临床领域和技术主题的严谨证据不足；构建可自动重现的完整分析管线；

**🔧 技术方法**

基于 Python 的批量检索、正则式模型与主题检测、负二项回归、马尔可夫/中位数回归、Oaxaca‑Blinder 结构漂移分解、模块化社区检测、HHI 与 Shannon 指数、注意‑证据差距公式等统计与图形方法；

**📊 数据集**

从 PubMed/MEDLINE 2023-01 至 2026-06 共 11,628 条文献记录（经去重、去除非研究类记录后），包含 14 个临床领域和 53 种模型/主题正则表达式；

**📈 对比分析**

比较方法：利用中位数回归比较不同研究设计的评估滞后，利用负二项回归评估文献增长率，利用漂移分解比较实际滞后斜率与理论机械滞后；结果显示评估滞后从 1.33 季度上升至 6.08 季度，随机对照试验滞后约 4.6 季度；试验设计比例低于 3%，模型版本说明仅 50%；

**⚠️ 局限性**

局限性包括：仅检索 PubMed，忽略计算机科学会议论文；模型名称检测依赖正则表达式，存在命名冲突与误报；研究设计分类依赖 PubMed 公开元数据，可能低估试验比例；滞后测算仅适用于标题/摘要中明确提及模型的记录；区域与期刊归属受首作者信息限制；无法评估模型安全性与有效性，只描述文献评价情况。

---

## 497. Signing the Transaction but Not the Decision: Whisper Attacks and a Binding Defense for AP2

**arXiv ID:** 2609.11757 | [PDF](https://arxiv.org/pdf/2609.11757v1)

**作者:** Yedidel Louck `[一作]` (Ariel University), Ariel Stulman `[通讯]` (Jerusalem College of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文发现 Agent Payments Protocol (AP2) 允许三类 Prompt‑Injection 攻击，评估其在多模型、多框架上的成功率，并提出协议层结构绑定防御（A‑VIP），同时发布首个 AP2 级别攻击基准与防御实现。

**💡 创新点**

创新点在于：①系统化三类攻击的分类与定量评估；②提出零误报的结构绑定防御 A‑VIP，包含凭证单次使用、显示绑定、链完整性检查；③发布公开的 AP2 WhisperBench 基准与机器检查的防御规范。

**🔧 技术方法**

技术手段包括：结构绑定校验（实体绑定、显示绑定、链完整性）、凭证单次使用令牌、内容扫描器与语义验证器、模型评估与 Wilson 95% 置信区间、适配器攻击等。

**📊 数据集**

使用的数据集为：1,544 攻击与 1,050 正常控制场景的 AP2 WhisperBench；Synthetic Vault（51 虚拟用户）用于凭证泄露测试；多语言多脚本测试集；公开产品元数据作为正常样本。

**📈 对比分析**

评估方法：在未加防御的 AP2 v0.2.0 上对比攻击成功率（Vault 90%，Branded 56%，Selection 73.3%），在 17 个 Google 模型、8 个跨供应商模型、3 种框架以及消费者助手中进行跨模型评测；防御实现无误报、确认成本低，延迟仅数百毫秒。

**⚠️ 局限性**

局限性包括：消费者助手实验样本有限；内容扫描器误报率为 12%；对基于事实的 Selection 攻击只能提示而非阻止；部分评测基于单次实验，未覆盖所有高级模型；防御依赖手工确认，未完全消除攻击。

---

## 498. Why Does Post-Training Quantization Work?

**arXiv ID:** 2609.11716 | [PDF](https://arxiv.org/pdf/2609.11716v1)

**作者:** Yuxiang Chen `[一作]` (Tsinghua University), Jianfei Chen `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了为什么预训练的大语言模型在权重量化后仍能保持稳定的下一个词预测，探讨了隐藏状态误差在网络层级中的传播机制。

**💡 创新点**

提出了两条关键机制：①预训练模型的层级更新误差往往与前一层传入的误差相反，形成“相互抵消”效应，显著减缓隐藏误差累积；②量化导致的最终隐藏状态主要是旋转误差，而高维LM头几何使得高排名词的得分和概率变化更小，从而保持输出质量。

**🔧 技术方法**

利用后训练量化（PTQ）技术（如NVFP4四位量化）与完整精度模型的前向对比，推导隐藏误差递推式并量化相互作用；使用理论分析（角度旋转、投影角度变化）解释高排名词稳定性；进行实验干预验证相互抵消的因果作用。

**📊 数据集**

在多种大模型上验证，包括Qwen3（32B、30B、8B）、OLMo3、Gemma3及其Mixture-of-Experts版本；使用六个零样本基准（例如通用文本推理、问答、摘要等）评估精度。

**📈 对比分析**

通过对比完整精度（BF16）与量化后（W4）模型在交叉熵、KL散度以及Top-K保持率等指标上的差异，发现量化后模型仅平均下降0.43个百分点，Top1翻转率约10%，Top20保持率≈85%；实验干预显示去除相互抵消后隐藏误差与输出性能大幅恶化。

**⚠️ 局限性**

局限包括：只研究单步下一个词预测而未覆盖多词生成；使用聚合统计而未精准描述极端误差轨迹；LM头理论假设隐藏状态旋转方向均匀，实际可能偏离；未探究训练随机性是否产生相互抵消以及如何将发现用于改进PTQ。

---

## 499. Second-Order Expansion of Privacy Amplification Under f-Divergence Criteria

**arXiv ID:** 2609.11794 | [PDF](https://arxiv.org/pdf/2609.11794v1)

**作者:** Mario Berta `[一作]` (RWTH Aachen University), Marco Tomamichel `[通讯]` (National University of Singapore)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究在 f‑散度度量下，从有记忆无噪声源与旁信息产生的随机数提取（隐私放大）问题，并给出了其第二阶渐近展开。

**💡 创新点**

创新点在于：①首次完成了对 f‑散度（包含 Rényi 散度、总变差等）下的第二阶精度分析；②揭示固定与优化旁信息边缘分布时的差异，导致 Gaussian 混合 vs 单一 Gaussian 的结果；③利用水填充与 Berry‑Esseen 定理提供了通用的非参数证明。

**🔧 技术方法**

主要技术手段包括：两通道哈希（two‑universal hashing）构造、f‑散度的视角极小化（perspective minimization）、水填充（water‑filling）优化、条件方差分解、Berry‑Esseen 定理与中心极限定理的组合。

**📊 数据集**

由于是理论分析，未使用任何实际数据集；所有结论均在有限字母空间的假设下推导。

**📈 对比分析**

比较方法：与已有的总变差和 Rényi 散度下的第二阶结果对比，验证一致性；通过极限与正态尾分布的数值近似验证公式的正确性。性能表现在理论上是最优的，即给出了精确的第二阶常数。

**⚠️ 局限性**

局限性：①仅适用于有限字母空间；②要求 f(t)/t→0（排除如相对熵等超线性散度）；③对非可微或无界生成函数的处理需要额外工作；④在极限趋向相对熵或逆相对熵时结果不再适用。

---

## 500. BlueSTAR: Tiered Agentic Architecture for Autonomous Cyber Defense

**arXiv ID:** 2609.11852 | [PDF](https://arxiv.org/pdf/2609.11852v1)

**作者:** Simona Boboila `[一作]` (Northeastern University), Alina Oprea `[通讯]` (Northeastern University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计了一种双层代理架构（Tiered Agentic Architecture）用于在企业 IT/OT 网络中实现自主网络防御，其中包括快速、可预测的确定性层和基于 LLM 的推理层；同时引入了信号归一化层将原始日志压缩为可执行的威胁指示器，并提出了跨周期记忆机制；此外，还提出了一种基于攻击者触及范围、任务影响和防御附带成本的 Res‑AUC 复原度量。

**💡 创新点**

①将日志归一化为可执行的威胁指示器是实现 LLM 推理的必要前置条件；②双层架构闭合了确定性防御在噪声阈值和跨周期推理方面的结构缺口；③提出了与代理无关的 Res‑AUC 复原度量，能够在持续的防御过程中对不同策略进行统一评估；④利用 LLM 在已聚合的 IOC 上进行跨时空关联推理，而非直接处理原始日志；⑤在真实 IT/OT 网络上进行评估而非仅限模拟。

**🔧 技术方法**

确定性规则引擎、LLM 推理模块、IOC 归一化与聚合层、跨周期记忆管理、动作执行器（SOAR 兼容）、Res‑AUC 评估框架；采集端点、网络、AD/AD‑CS、SCADA 以及 SIEM 等多源日志。

**📊 数据集**

在两套企业级 IT/OT 真实实验平台上进行评估，使用七条基于实际入侵技术（如 ESC1 证书滥用、Pass‑the‑Hash）的攻击链；实验日志来源于 Windows 事件日志、Sysmon、Zeek、Suricata、AD CS 证书服务、SCADA 监控等。

**📈 对比分析**

通过对比仅使用确定性层与双层架构（确定性+推理）在同一攻击链上的表现；指标包括任务成功率、关键任务标志保持、响应时间以及 Res‑AUC 分数；结果显示：双层架构在处理确定性攻击时保持了 100% 的快速包含率，同时在需要跨周期推理的复杂攻击（如凭证窃取、持续复位、并发攻击）中实现了 100% 的防御成功；在 Raw SIEM + LLM 场景下的吞吐量、精确度和成本差距分别为 3.7×、3.8% 与 $37M。

**⚠️ 局限性**

1）未能处理能根据防御者行为动态调整策略的对手；2）LLM 推理依赖有效的提示与上下文，可能受限于模型能力与推理成本；3）对持续性威胁只能实施封锁而非根除，需人工干预；4）实验规模仅覆盖两套网络，缺乏对更大、异构环境的验证；5）在极高日志速率下仍需要进一步的压缩与并行处理方案。

---

## 501. Target leakage, not model class, explains reported accuracy in survey-based cardiovascular screening: a leakage-tiered audit of glass-box and tabular foundation models

**arXiv ID:** 2609.11838 | [PDF](https://arxiv.org/pdf/2609.11838v1)

**作者:** Raad Bin Tareaf `[一作]` (XU Exponential University of Applied Sciences), Cedric Schmitz `[通讯]` (XU Exponential University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究评估了十种机器学习模型在美国行为风险因素监测系统（BRFSS）心肌梗死筛查任务中的表现，系统性剔除后续诊断相关特征后进行泄漏层级审计，并对模型的公平性、校准性、不确定性与推理成本等进行全面比较。

**💡 创新点**

创新点在于：①引入多层泄漏特征分级（T0–T2）量化泄漏对AUROC的影响；②证明透明的可解释增益机（Explainable Boosting Machine）在无泄漏层级下与所有模型在0.005 AUROC内等效且推理速度快四个数量级；③在单阈值下实施可编辑的公平修复（shape repair）并提供机器可读编辑日志；④采用分层的Mondrian conformal预测解决群组不均匀覆盖问题；⑤将模型与阈值冻结在2023年进行时间外验证，证明迁移性良好。

**🔧 技术方法**

使用了逻辑回归、随机森林、XGBoost、LightGBM、CatBoost、可解释增益机、多层感知器以及两种表格基础模型（TabPFN v2、TabICL）；对模型进行分类、校准（Isotonic）、公平性审计（阈值、加权、shape repair）、分层 conformal 预测、解释信度评估（TreeSHAP、KernelSHAP、LIME）以及推理成本测量。

**📊 数据集**

数据集为2022年和2023年美国BRFSS的电话调查文件，包含442,067和430,755名受访者，目标变量为自报的既往心肌梗死。

**📈 对比分析**

在去除两项直接后诊断标记后，所有模型AUROC下降约0.05，所有模型聚集在0.839–0.844之间；可解释增益机在T1层级下与CatBoost、XGBoost、TabICL在0.005 AUROC内等效，并且在推理时间上快≈4个数量级；校准通过Isotonic恢复至ECE≤0.005；公平性修复将女性与男性TPR差距从0.128降至≈0.01；分层 conformal 预测保证了各年龄性别组覆盖率≥0.90；2023年冻结模型的AUROC仅下降≤0.0016，阈值保持在0.85敏感度附近。

**⚠️ 局限性**

局限包括：使用自报诊断导致的误分类、残留的基于护理的泄漏、仅评估同一国家同一系统的时间外验证、未考虑设计基变异的方差、基础模型的快速演进导致结果可能变化、仅对测量属性进行公平性评估，未验证对临床决策链条的真正影响。

---

## 502. An analysis of the relationship of input metrics

**arXiv ID:** 2609.11824 | [PDF](https://arxiv.org/pdf/2609.11824v1)

**作者:** Addison Crump `[一作]` `[通讯]` (CISPA Helmholtz Center for Information Security), Addison Crump (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并比较了基于语法的输入覆盖度量，提出了 k-alt-path 并评估其相对 k-path 的敏感性与资源消耗。

**💡 创新点**

定义了覆盖关系并系统化输入度量的比较，改进 k-path 为 k-alt-path，减少冗余并提升覆盖率。

**🔧 技术方法**

采用分区测试理论、CFG 解析、路径计数、集合论覆盖分析，并实现了 k-alt-path 计算与实验。

**📊 数据集**

使用公开的多种上下文无关语法（JSON、CSV、REST、XML 等）以及 tribble 生成的随机输入。

**📈 对比分析**

通过覆盖关系、子域包含、Spearman 相关、实验测量子域数量比率进行比较，发现 k-alt-path 在相同 k 下需要 5–10 倍更少存储且覆盖率更高。

**⚠️ 局限性**

对含重复、约束或上下文敏感语法不适用；覆盖关系不等价于缺陷检测优越性，未验证实际缺陷发现效果。

---

## 503. Predicting Privacy Leakage from Weight Spectral Density

**arXiv ID:** 2609.11780 | [PDF](https://arxiv.org/pdf/2609.11780v1)

**作者:** Richard J. Preen `[一作]` (University of the West of England), Jim Smith `[通讯]` (University of the West of England)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了权重矩阵的谱特征（如 stable rank、Log α‑Norm 等）与机器学习模型在成员推断攻击（MIA）中的泄露风险之间的关联，并评估这些谱指标是否可作为无参量的隐私风险代理。

**💡 创新点**

首次在实验中发现 stable rank 与整体攻击成功率（LiRA AUC）高度相关，Log α‑Norm 与低误报率下的攻击成功率（TPR@0.001）呈显著负相关，并且谱指标与传统 generalisation gap 的组合能显著提升对 MIA 风险的预测精度。

**🔧 技术方法**

使用 WeightWatcher 从训练好的模型权重中提取 ESD 并计算 stable rank、Log α‑Norm、α、log spectral norm；采用 LiRA 进行成员推断攻击；利用 Spearman 相关、线性回归和 ROC‑AUC 等统计方法进行分析。

**📊 数据集**

在 CIFAR‑10（图像）和 OpenML 的 Volkert（表格）两种数据集上训练多种 MLP 模型，涵盖不同深度、宽度和超参数配置。

**📈 对比分析**

通过与 generalisation gap 进行对比，谱指标在预测 LiRA 的 AUC（ρ≈0.6‑0.87）和 TPR@0.001（ρ≈-0.4‑-0.55）方面表现更好；多元回归将 R² 提升至 0.82；在二分类高风险模型识别中，单一谱指标表现不一，但组合（如 Log α‑Norm + ϵ_gap）在不同数据集上更为稳健。

**⚠️ 局限性**

实验仅涵盖 MLPs、两种数据集、单一攻击方法（LiRA）和有限样本量（44 模型），未验证更大、复杂模型（如 ResNet、Transformer）或其他 MIA 方案，且缺乏交叉验证与跨域泛化评估。

---

## 504. Differentially Private EEG Feature Anonymization: A Privacy-Utility Case Study in Clinical Neurophysiology

**arXiv ID:** 2609.11777 | [PDF](https://arxiv.org/pdf/2609.11777v1)

**作者:** Noman Sadiq `[一作]` (University of South-Eastern Norway), Mohsen Toorani `[通讯]` (University of South-Eastern Norway)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

在多医院环境下，对临床EEG信号进行预处理和特征提取后，应用主体级差分隐私（高斯噪声和坐标拉普拉斯噪声）对患者级特征向量进行加噪，并评估其对统计实用性和下游机器学习分类性能的影响。

**💡 创新点**

将差分隐私机制直接应用于EEG特征而非原始时序数据，提出三种部署场景（客户端、服务器端、分散本地训练），同时对高斯与拉普拉斯噪声的敏感度校准与噪声尺度进行实证比较，揭示实际隐私‑实用性权衡。

**🔧 技术方法**

使用EEG预处理（滤波、分段、PSD特征提取）、L2裁剪/L1裁剪、IBM DiffPrivLib实现的分析高斯和坐标拉普拉斯噪声、统计实用性指标（RMSE、MAE、相关系数、SNR）以及MLP分类器的留一患者交叉验证。

**📊 数据集**

VIKING项目提供的匿名化临床EEG数据：122条记录、17名患者、约33通道、250–500 Hz采样率，特征维度363。

**📈 对比分析**

对比原始特征、Gaussian‑ε=10、Laplace‑ε=10下游MLP的LOSOV性能：原始准确率0.706、F1 0.545；Gaussian扰动导致准确率0.294（召回1.0但特异0）；Laplace扰动准确率0.235。统计指标显示噪声随ε增大而减小，相关系数、SNR提升但仍低于原始水平，说明高噪声下实用性显著下降。

**⚠️ 局限性**

限制包括：样本量小、类别不平衡、仅使用一次噪声实现、未进行多次实验验证、预处理步骤的敏感度假设为固定且未实现完整的隐私计数、缺乏重识别/成员推断/重建等攻击实验，且拉普拉斯实验使用的敏感度并未满足正式全向量校准。

---

## 505. Reproducibility in the Age of Agentic AI: Context Engineering at the Timescale of a Codebase

**arXiv ID:** 2609.11728 | [PDF](https://arxiv.org/pdf/2609.11728v1)

**作者:** Lorena A. Barba `[一作]` `[通讯]` (George Washington University), Lorena A. Barba (George Washington University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文探讨了可复现研究实践与 AI 代理代码开发的上下文工程关系，提出将传统软件工程实践视为代理友好性的上下文设计，并阐述如何让代理自动生成测试、提交信息、决策记录等文档。

**💡 创新点**

创新点在于将可复现性文档映射为代理上下文工程，揭示可复现性与代理生产力的天然共性，并提出通过命令驱动的“调用层”让研究者在代理工作中自动产出可复现文档。

**🔧 技术方法**

主要使用基于通用大语言模型（LLM）的代理生成与编辑技术，结合约定式提交、架构决策记录、测试套件等软件工程规范。

**📊 数据集**

本文未使用传统实验数据集，而是基于公开代码仓库的分析（约 2500+ 个代理指令文件）和作者自身实践经验进行论证。

**📈 对比分析**

由于是概念性阐述，未给出对比实验；作者通过对公开仓库中 2500+ 文件的统计分析验证了六大共性实践的有效性。

**⚠️ 局限性**

局限性包括：代理生成的文档仍需人工审核，可能导致判断外包；对模型漂移与不确定性的应对不足；以及研究文化与激励机制尚未彻底改变。

---

## 506. The Eloquence submission for Task 2 of the Interspeech 2026 MLC-SLM challenge

**arXiv ID:** 2609.11724 | [PDF](https://arxiv.org/pdf/2609.11724v1)

**作者:** Jordi Luque `[一作]` (Telefónica Innovación Digital), Filippo Vella `[通讯]` (Consiglio Nazionale delle Ricerche)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在MLC‑SLM 2026第二赛季的多语言多轮对话理解任务（Task 2）中，Eloquence团队提出了三种解法：①对Voxtral‑Mini‑3B进行LoRA微调，结合跨语言数据增强、ASR转写补充与基于时间戳的音频裁剪；②在冻结的Voxtral‑24B模型上采用多模态上下文学习（ICL）纠正标签偏差；③构建无训练的检索式“声音锚定记忆”层，融合声学身份、语义内容与知识图谱，然后交由冻结的LLM回答问题。

**💡 创新点**

创新点包括：①通过NLLB‑200将非英语问题与选项翻译为英语，以弥补训练集跨语言缺失；②利用时间戳裁剪将模型关注范围精准定位到答案所在段；③提出基于多层持久化记忆的检索框架，实现在无任何参数更新的情况下完成对话理解；④在大型Speech‑LLM上直接使用ICL纠正强烈的标签偏差，展示了零训练即能显著提升性能。

**🔧 技术方法**

使用技术包括：LoRA微调、ASR转写生成、时间戳裁剪、NLLB‑200翻译、Voxtral‑Mini‑3B与Voxtral‑24B模型、multimodal ICL、前置的Speaker Embedding（Titanet）、MiniLM文本编码、ChromaDB向量检索、NetworkX知识图谱、Qwen3‑14B-AWQ等。

**📊 数据集**

使用数据集为MLC‑SLM 2026提供的21种语言多轮对话数据集；训练集与验证集采用Gemini2.5‑Pro生成的多语言多选题；开发集与测试集为官方提供的多语言对话理解评测集。

**📈 对比分析**

与官方基线Qwen2.5‑Omni‑7B相比，微调版Voxtral‑Mini‑3B在dev上达0.85、test 0.72；ICL版Voxtral‑24B在test上达到0.81（最高）；检索版在dev上0.83、test 0.68；整体提升幅度明显，尤其是ICL方案在不增加训练成本的前提下实现了最佳性能。

**⚠️ 局限性**

局限性包括：①微调方法易出现过拟合，导致dev到test的性能下降；②ICL方案主要以英语示例为主，跨语言泛化尚未充分验证；③检索式系统虽然无训练成本，但依赖多模型前端，部署复杂度相对较高；④所有方法在面对多重发言、停顿、重叠等真实对话噪声时仍存在鲁棒性挑战。

---

## 507. MC-DeTra: Motion-Consistent Joint Object Detection and Socially-Aware Trajectory Forecasting in Bird's-Eye-View Images

**arXiv ID:** 2609.11717 | [PDF](https://arxiv.org/pdf/2609.11717v1)

**作者:** Vladislav Diuzhev `[一作]` (Moscow Institute of Physics and Technology), Dmitry Yudin `[通讯]` (Moscow Institute of Physics and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套统一的端到端检测与轨迹预测模型MC-DeTra，利用LiDAR与高清地图的鸟瞰视图实现联合检测与预测，并通过训练时仅使用的三种辅助正则化提升动态行人预测精度。

**💡 创新点**

创新点在于：1) 引入三种训练时安全的辅助目标——过去轨迹重建（Past Reconstruction）、占用场景辅助（Occupancy Auxiliary）和方向一致性约束（Heading Consistency）；2) 通过梯度范数校准实现多任务损失平衡；3) 公开了DeTra的完整实现与代码。

**🔧 技术方法**

使用的技术包括：基于BEV的LiDAR+地图特征提取、残差与动态卷积骨干网络、DETR式多尺度变换器、GRU+MLP预测头、Laplace分布的多模态轨迹回归、焦点损失与gIoU损失等。

**📊 数据集**

实验使用Waymo Open Dataset的车辆轨迹与BEV数据，评估严格的检测条件化预测协议。

**📈 对比分析**

与DeTra原始实现对比，MC-DeTra在保持检测精度的前提下，在动态车辆的minFDE_6、minADE_6、brier-minFDE_6上分别提升约2.4%、2.7%与3.0%；同时通过梯度校准实现了更稳健的训练。

**⚠️ 局限性**

局限性包括：提升幅度相对较小，主要集中在动态车辆；模型在训练时使用的地图编码简化导致与原论文的精度差距；未对多种目标类型（行人、自行车等）进行实验；梯度校准控制器虽能自动化但未显著优于手动调参。

---

## 508. SenseNova-U1.5: Towards Native Unified Visual Intelligence

**arXiv ID:** 2609.11929 | [PDF](https://arxiv.org/pdf/2609.11929v1)

**作者:** Haiwen Diao `[一作]`, Dahua Lin `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了SenseNova-U1.5，一款8B参数的原生多模态统一模型，能够在同一视觉表示空间内实现视觉理解、推理与生成，并在高分辨率（4K）文本到图像、图像编辑、交互式生成等任务上表现卓越。

**💡 创新点**

创新点包括：①将视觉令牌从独立补丁预测迁移为空间联合重建，利用像素级卷积和Pixel Shuffle实现跨补丁信息交互；②采用专门化先训练专家（美学、OCR、信息图、编辑）后通过多专家 on‑policy 蒸馏统一为单一策略；③融合自监督流匹配、LPIPS感知损失与多任务联合训练，实现视觉与语言的紧耦合；④通过多尺度噪声编码与自适应解码提升4K级别生成质量。

**🔧 技术方法**

技术手段包括：原生Encoder‑Free视觉接口、混合Transformer（MoT）带有前向/后向注意力、空间卷积解码器、流匹配学习、LPIPS感知损失、强化学习（GRPO、CPS、Precise）、自监督蒸馏（OPD）、自适应噪声标量编码、语义嵌入与RoPE编码。

**📊 数据集**

数据集：约59M图文对（含高分辨率>1024^2样本占88%）、38M图像编辑样本、4M交互式序列、RL训练集包括280K美学、60K OCR、120K编辑、120K信息图等；所有数据均经过质量过滤、对齐、分布重平衡与人工标注。

**📈 对比分析**

与现有开源/闭源模型对比，SenseNova-U1.5在Qwen‑Image‑Bench、GenEval、GenEval2、OneIG‑Bench、DPG‑Bench、CVTG‑2K、LongText‑Bench、IGenBench、BizGenEval、WISE、ImgEdit、GEdit‑Bench、WeEdit、OmniRef‑Bench、RISEBench、OpenING、VBVR‑Pro‑Bench、Uni‑MMMU、RealUnify等多项评测中均名列前茅，部分指标突破同类8B模型甚至接近或优于部分闭源大模型。

**⚠️ 局限性**

局限性：对极其复杂的逻辑/时间推理仍需改进；长文本和多区域文字渲染的细节偶有失真；编辑任务中涉及深层语言与上下文推理的场景仍易失败；训练与推理成本高，推理时仍需大量显存；模型规模虽为8B，但在更大规模与更高分辨率（>8K）下的表现尚未验证。

---

## 509. Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data

**arXiv ID:** 2609.11917 | [PDF](https://arxiv.org/pdf/2609.11917v1)

**作者:** Atindra Jha `[一作]` (Stanford University), Luke Zettlemoyer `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了数据重复率对稀疏 MoE 语言模型与密集 Transformer 的训练效果与泛化能力的影响，并提出了通过 dropout 与输出掩码等正则化手段缓解过拟合的方法。

**💡 创新点**

创新点在于揭示 MoE 在高重复率下比密集模型更易过拟合，且过拟合程度主要受总参数量而非活跃参数量影响，同时首次将多种正则化技术与 MoE 结合并分析其对专家专化与路由稳定性的机制。

**🔧 技术方法**

主要技术包括计算匹配的 MoE 与密集 Transformer 训练、控制重复率（R=1–1024）、dropout、FFN 输出掩码、专家 dropout、专家输出掩码以及路由抖动等正则化方法。

**📊 数据集**

使用 OLMoE 组合的数据集，涵盖 Web 抓取文本（DCLM）、代码（StarCoder）、学术论文（peS2o）与百科文本（Wikipedia），并在单域与多域混合、不同质量过滤水平下进行实验。

**📈 对比分析**

通过在多种 held‑out 领域和下游任务上的交叉熵损失和准确率进行比较，实验表明在低重复率下 MoE 性能优于密集模型，但当重复率 ≥ 32 时其性能迅速劣于密集模型；dropout 与输出掩码可显著减轻过拟合，但仍未完全恢复全唯一数据的效果。

**⚠️ 局限性**

局限性包括只评估了有限的 MoE 配置与正则化组合，未覆盖更大规模模型和更复杂路由策略；实验数据主要来自 OLMoE，缺乏在不同领域或跨语言环境下的验证；此外高计算成本限制了更细粒度的机制分析。

---

## 510. AccelForge: Comprehensive Modeling and Co-Design Framework for AI Accelerators

**arXiv ID:** 2609.11906 | [PDF](https://arxiv.org/pdf/2609.11906v1)

**作者:** Tanner Andrulis `[一作]` (Massachusetts Institute of Technology), Joel S. Emer `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AccelForge，一套面向 AI 加速器的统一建模与快速探索框架，支持设备、电路、体系结构、工作负载以及映射的可组合化定义，并实现了高效的最优映射器。

**💡 创新点**

①整合并扩展了多种现有工具（如 Timeloop、Accelergy、LoopTree 等），实现了对设备、循环映射、融合、异构布局等多维度设计空间的全面覆盖；②提出了两种新型映射器（TCM 与 FFM），能够在完整映射空间内以极快速度找到最优映射；③采用 Python 语言实现，提供高度模块化、易用且可扩展的接口，降低了研究者的上手门槛。

**🔧 技术方法**

Python 计算框架、可配置的组件与工作负载描述、基于图搜索与动态规划的映射器（TCM/FFM）、融合与异构张量调度算法、能耗与延迟模型（如缓冲读取 1pJ/bit）等。

**📊 数据集**

在评估实验中使用了多种典型 AI 工作负载，包括卷积网络（ResNet、MobileNet）和大型语言模型（BERT、GPT‑系列）等，覆盖单算子和多算子图形、不同张量尺寸与位宽场景。

**📈 对比分析**

通过与现有框架（TileFlow、ZigZag、Timeloop、CiMLoop）对比，AccelForge 在能耗、延迟和映射时间上实现了显著提升：映射时间相较最慢框架缩短 3–4 倍，能耗误差率低于 0.5%，并在多种工作负载上保持最优或次优映射结果。

**⚠️ 局限性**

①对极大规模工作负载（如完整 LLM 推理）仍需进一步验证映射器的可扩展性；②目前尚未覆盖所有新型硬件技术（如 3D 堆叠、近存计算）的详细能耗模型；③Python 实现虽然易用，但在极大搜索空间时仍可能受到解释型语言性能限制。

---

## 511. 3D Point Splatting for mmWave Radar Novel View Synthesis

**arXiv ID:** 2609.11894 | [PDF](https://arxiv.org/pdf/2609.11894v1)

**作者:** Adnan Armouti `[一作]` (Cornell Tech), Rajalakshmi Nandakumar `[通讯]` (Cornell Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为3DPS的可微分物理雷达渲染器，用于毫米波雷达的新视角合成（NVS），能够从稀疏视角训练并生成任意视角的ADC、CRP和RA输出。

**💡 创新点**

创新点在于：①将闭式雷达方程与ITU‑R P.2040表面散射模型（BSDF）直接映射到三维点原语；②利用预计算的点扩散函数（PSF）实现点 splatting，消除 Monte‑Carlo 采样；③在保持物理精度和复数相位的同时，实现对多视角的可微分优化；④通过与 LiDAR 先验对齐，解决雷达数据稀疏性问题。

**🔧 技术方法**

主要技术包括：离散化的有向三维点集合；闭式雷达方程的几何与材料评估；点扩散函数 PSF splatting；Hann‑windowed FFT 生成 CRP 与 RA；CUDA 并行实现（BSDF、splitting 与梯度）以及基于 8 视角的联合优化。

**📊 数据集**

使用公开的 ColoRadar 数据集（TI MMWCAS mmWave 雷达 + Ouster OS1‑64 LiDAR），共 6 个室外场景。

**📈 对比分析**

与三种光学 NVS 基线（DART、Radar Fields、RadarSplat）进行对比；在 Held‑out |RA| 上取得平均 Pearson 相关 0.587，约为三基线的 1.7–5.2 倍；训练时间约 3 分钟/场景；在 CRP、ADC 的复杂值输出上也表现优于基线。

**⚠️ 局限性**

局限性包括：仅考虑单次反射，无法处理室内多径；依赖 LiDAR 先验，缺乏完全雷达自洽的 3D 重建；未使用相位监督，导致相位误差仍存在；目前仅支持固定雷达几何和低 Doppler 分辨率，尚未验证动态场景或更广阔的雷达配置。

---

## 512. Learning Agent-based Model Predictive Control for Holistic Vehicle Performance

**arXiv ID:** 2609.11871 | [PDF](https://arxiv.org/pdf/2609.11871v1)

**作者:** Jiaming Zhong `[一作]` (University of Waterloo), Amir Khajepour `[通讯]` (University of Waterloo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种融合 Gaussian 过程回归（GPR）与代理基模型预测控制（AMPC）的混合控制框架 LAMPC，解决未知黑盒子控制器对车辆动态预测误差的问题。

**💡 创新点**

创新点在于：①引入在线数据管理与子集选择实现实时高效 GPR；②利用多步预测机制一次性输出整个规划 horizon 的预测；③结合软随机机会约束和闭环不确定性传播保证约束满足与优化可行性。

**🔧 技术方法**

主要技术包括：代理基模型预测控制、Gaussian 过程回归、多步 GPR 推断、软随机机会约束、数据管理与子集选择算法。

**📊 数据集**

使用车载实验平台 Chevrolet Equinox 电动汽车的真实测量数据，配合仿真中的 Carsim 高保真模型，以及人工生成的黑盒差速转矩数据。

**📈 对比分析**

通过在“少学”与“已学”两种场景下的双车道变道（DLC）和正弦波测试，比较 AMPC、LAMPC-普通约束、LAMPC-软机会约束，结果显示：在少学场景下 LAMPC-软机会约束始终满足侧滑角和偏航率限制；在已学场景下 LAMPC 的偏航率跟踪误差比传统 AMPC 低约 3 倍，计算时延略有增加但仍符合实时要求。

**⚠️ 局限性**

局限性主要体现在：①GPR 的计算复杂度随样本增大而升高，需精细的数据稀疏策略；②对超长预测 horizon 的不确定性传播仍可能导致保守性提升；③系统稳定性理论尚未完全覆盖，未来需进一步研究鲁棒稳定性保证。

---

## 513. AdamX: Cosine similarity meets gradient descent

**arXiv ID:** 2609.11867 | [PDF](https://arxiv.org/pdf/2609.11867v1)

**作者:** Francisco Caldas `[一作]` (Universidade Nova de Lisboa), Cláudia Soares `[通讯]` (Universidade Nova de Lisboa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种新的自适应一阶优化器 AdamX，将余弦相似度控制器嵌入 Adam/AMSGrad 的动量归一化更新中，以调节更新幅度。

**💡 创新点**

创新点在于利用有界余弦相似度作为轻量级的方向一致性信号，对梯度方向连续性进行自适应调节，从而在保持 AMSGrad 逐步递增第二矩估计的同时引入动态步长控制。

**🔧 技术方法**

采用的技术包括：Adam/AMSGrad 的指数移动平均动量与方差估计；有界余弦相似度控制器（γ_t = exp(λ·cos(g_t,g_{t-1}))）；变异校正（类似 RAdam 的早期方差修正）；以及针对在线凸优化的 OCO 分析框架。

**📊 数据集**

实验使用了常见的深度学习基准数据集 MNIST 和 CIFAR-10，并在 DeepOBS 公开的网络架构上进行训练。

**📈 对比分析**

与 SGD、Adagrad、RMSProp、Adam、AdamW、AMSGrad、RAdam、Yogi、Lion、Adan 等十种主流优化器对比，指标为达到预设测试准确率所需的训练 epoch 数；AdamX 在 CIFAR-10 上取得最少 epoch 数，MNIST 上表现与 AMSGrad、Yogi 相当，同时在训练损失曲线与收敛稳定性上优于大多数对手。

**⚠️ 局限性**

局限性包括：对 λ 等超参数的鲁棒性尚未系统评估；余弦相似度信号在梯度噪声较大时可能振荡，影响收敛速度；实验仅覆盖小规模任务，未在大规模数据集或真实硬件上评估壁钟时间；以及对动量项在 OCO 分析中的影响仍待进一步研究。

---

## 514. EVPeriscope: Extended Perception across Aerial and Ground Vehicles with Event-based Propeller Tracking

**arXiv ID:** 2609.11920 | [PDF](https://arxiv.org/pdf/2609.11920v1)

**作者:** Dexter Ong `[一作]` (University of Pennsylvania), Pratik Chaudhari `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了EVPeriscope系统，利用地面机器人上的上向事件相机通过识别四旋翼螺旋桨的高频视觉特征，实现对无人机的检测、定位与闭环控制。

**💡 创新点**

其创新点在于采用无标记、事件驱动的螺旋桨检测方法，提供低延迟、抗光照、抗运动模糊的相对定位，并将无人机作为扩展感知的“潜望镜”与地面机器人紧密耦合。

**🔧 技术方法**

核心技术包括事件相机的IIR泄漏滤波、k‑means聚类提取螺旋桨中心、基于已知几何的PnP求解、EKF状态估计以及PID闭环速度控制，全部在UGV的Jetson AGX Orin上实现。

**📊 数据集**

实验数据由作者在户外风速15mph、日照与夜间场景下自行采集的实时事件流和相机图像组成，并以ZED Mini VIO与GPS为基准进行评估；未使用公开数据集。

**📈 对比分析**

与传统基于AprilTag的帧相机检测相比，EVPeriscope在0.5–6 m高度下位置RMSE均低于0.03 m（近距）至0.7 m（远距），姿态误差小于约10°，并能在光照极端与遮挡环境中稳定工作，显示出更高的鲁棒性和更快的响应。

**⚠️ 局限性**

局限性包括对无人机VIO的依赖（漂移会影响相对定位）、无法处理多架无人机产生的高频噪声、只适用于已知几何的四旋翼、以及对事件相机视场与分辨率的依赖。

---

## 515. Can Edge-Deployable Vision-Language Models Identify Species?

**arXiv ID:** 2609.11916 | [PDF](https://arxiv.org/pdf/2609.11916v1)

**作者:** William Zhou `[一作]` (Plano West Senior High School), Yi Ding `[通讯]` (University of Texas at Dallas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对边缘可部署的2–8B规模视觉语言模型（Qwen3‑VL 2B/4B/8B、Gemma3 4B）与专门训练的BioCLIP在96种物种识别任务中进行评估，并比较它们在清洁摄影图像和野外相机捕捉图像上的表现。

**💡 创新点**

提出规模不一定带来专业知识优势，强调专门训练数据与域差异是影响识别性能的核心因素；同时通过两套独立评估集验证结果的稳健性，揭示模型规模与性能呈非单调关系。

**🔧 技术方法**

采用多模态零样本推理、Q4量化模型、Ollama本地推理、CustomLabelsClassifier以及多种提示（多项选择、开放式）和图像处理方式（裁剪、原始、叠加框）。

**📊 数据集**

使用来自6个LILA.science集合的5,554张相机捕捉图像与匹配的iNaturalist清洁摄影图像，共计96种物种作为评估数据集。

**📈 对比分析**

通过多项选择与开放式提示、不同图像处理方式对比，发现BioCLIP在物种识别上高出30–60个百分点；域差导致10–26个百分点的准确率下降；模型规模与性能不呈单调关系，且在某些指标上更大模型并不一定更好。

**⚠️ 局限性**

限制在Q4量化模型、未使用全精度、只测试2–8B规模、评估集来源相同、未进行正式显著性检验、未覆盖更大规模模型或不同硬件平台等。

---

## 516. Caption-once, Frames-on-Demand: Visual-Need Routing for Budget-Aware Agentic Long Video Understanding

**arXiv ID:** 2609.11899 | [PDF](https://arxiv.org/pdf/2609.11899v1)

**作者:** Weitong Cai `[一作]` (Queen Mary University of London), Zhensong Zhang `[通讯]` (Huawei)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种边缘-云协作框架 Caption‑once, Frames‑on‑Demand（abbr），在边缘设备一次性完成双轨叙事索引（事件骨架 + 片段微记），并在云端通过视觉需求路由器按需检索关键帧，实现长视频问答。

**💡 创新点**

创新点：① 边缘一次性生成双轨语言索引，避免在线重新标注；② 视觉需求路由器根据问题类型动态决定是否拉取帧，将视觉成本转为可控的查询条件；③ 固定容量工作内存与关键帧采样机制，使每个查询的视觉使用量可预测；④ 通过三层内存架构和迭代回溯推理，显著降低在线视觉开销。

**🔧 技术方法**

技术方案：边缘使用轻量级 MLLM（如 Qwen3‑VL‑8B）配合 TransNetV2 进行事件切分和单次字幕生成；云端使用更大 MLLM（如 Qwen3‑VL‑32B）承担回答、定位、路由等四个代理；采用三层内存（事件记忆、片段记忆、视觉工作内存）与故事优先迭代推理循环；关键帧按事件自适应采样并限量入队。

**📊 数据集**

数据集：Video‑MME（无字幕设置）和 InfiniBench（包含时间推理、角色动作追踪、场景切换、全局外观四个基准技能）。

**📈 对比分析**

对比方式：与密集帧基线（如 Qwen3‑VL‑32B 768帧）以及多种基于代理的视频理解方法（VideoAgent、VideoTree、DrVideo、VideoLucy、MemVid）进行对比。结果显示：总体准确率约 67.5，单问帧数仅 5.8，远低于密集帧的 768 帧；在中长视频上优于其他代理方法；在时间推理和场景切换上达到或略高于基线，在外观/动作识别上略逊于全帧密集模型，但仍保持竞争力。

**⚠️ 局限性**

局限性：① 依赖离线字幕质量与事件切分，若关键细节未被捕获，文本推理无法恢复；② 关键帧采样可能漏掉短暂或细小视觉线索；③ 视觉需求路由器是二元且零样本的，边界问题易误路由；④ 未针对特定领域或 egocentric 视频进行微调；⑤ 边缘设备算力与网络延迟的实际评估尚未完成。

---

## 517. Domain-Specific Hallucination Detection in Large Language Models

**arXiv ID:** 2609.11878 | [PDF](https://arxiv.org/pdf/2609.11878v1)

**作者:** Varun Teja Chundru `[一作]` (Purdue University Fort Wayne), Debasmita Biswas `[通讯]` (Purdue University Fort Wayne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了多信号幻觉检测管线，结合Fine-tuned DeBERTa-v3、MC Dropout不确定性量化和温度校准，并在HaluEval上实现F1 0.915、AUROC 0.977；

**💡 创新点**

创新点在于将MC Dropout和温度校准联合用于幻觉检测，并通过Direct Preference Optimization（DPO）在生成模型中显著降低幻觉率；

**🔧 技术方法**

采用DeBERTa-v3、MC Dropout、温度校准、LR Meta-Classifier、DPO以及PubMedBERT等技术；

**📊 数据集**

使用HaluEval（QA、对话、摘要）和SciFact（生物医学验证）两大数据集；

**📈 对比分析**

与零样本DeBERTa、单通道推理等基线相比，HaluEval上取得F1 0.915、AUROC 0.977；在SciFact上，PubMedBERT微调获得F1 0.627、AUROC 0.808；

**⚠️ 局限性**

限制在于跨域迁移仍受预训练语料影响，模型缺乏对细粒度span定位的支持，且评估主要依赖于检测器的自评，未完全覆盖真实世界分布。

---

## 518. The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

**arXiv ID:** 2609.11873 | [PDF](https://arxiv.org/pdf/2609.11873v1)

**作者:** Yi Duan `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了递归自我改进（RSI）的研究进展，提出以自主性为中心的层级框架并对工业实践进行梳理。

**💡 创新点**

创新点在于构建了从L1到L5的自主性层级体系，区分了改进过程的闭环点、持久化对象及外部决策，系统化连接学术与工业证据。

**🔧 技术方法**

主要采用文献综述、框架设计、案例分析和对比表格等方法。

**📊 数据集**

无专门实验数据集，主要引用公开论文与技术报告。

**📈 对比分析**

通过对比表和案例，展示不同自主层级下的技术实现与应用场景，但未给出统一性能指标。

**⚠️ 局限性**

局限在于覆盖范围受公开资料限制，缺乏统一实验验证，RSI实际效果评估不足。

---

## 519. Augustinian BabyLM: What Ostensive Definition Can and Cannot Teach a Small Language Model

**arXiv ID:** 2609.11870 | [PDF](https://arxiv.org/pdf/2609.11870v1)

**作者:** Lisa Bylinina `[一作]` `[通讯]` (Utrecht University), Lisa Bylinina (Utrecht University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对小型语言模型的词嵌入进行视觉初始化（使用图像特征），仅用文本训练，研究视觉先验对语言学习的影响。

**💡 创新点**

仅在模型初始化阶段加入视觉嵌入而不改变训练过程；构造带有词频和种子信息的 VP‑Swap 最小对评测；通过合成视觉标注进一步验证因果关系。

**🔧 技术方法**

视觉特征提取（DINOv3、iBOT、SAM），随机词嵌入初始化，BERT式掩码语言模型训练，RSA 相似性分析，统计显著性检验。

**📊 数据集**

10M 词量的 BabyLM 训练语料；视觉标注数据（Flickr30k Entities, RefCOCO+, RefCOCOg, THINGS）；自生成的图像描述及合成视觉特征。

**📈 对比分析**

与随机初始化对照，在官方 BabyLM 2026 评测中提升约1–2点；VP‑Swap 评测中视觉初始化在所有种子上持续显著提升 2–3 分，表明对目标词的意义提升。

**⚠️ 局限性**

单次实验、非官方训练集、评测指标主要关注抽象结构，难以捕捉视觉先验带来的词义改进；合成标注噪声大；仅在单一掩码 LM 架构上验证。

---

## 520. Distance generalization in transformers: why bother with positional encoding?

**arXiv ID:** 2609.11913 | [PDF](https://arxiv.org/pdf/2609.11913v1)

**作者:** Daniel Henrik Nevermann `[一作]` (Goethe University Frankfurt), Claudius Gros `[通讯]` (Goethe University Frankfurt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer在固定上下文长度下的距离泛化能力，使用两种延迟复制任务（完整复制与选择性复制）作为基准。

**💡 创新点**

首次系统比较RoPE、ALiBi和无显式位置编码（NoPE）在距离泛化中的表现，并揭示数据多样性与迁移学习对距离泛化的双向影响；提出距离泛化作为排除长度泛化与位置依赖混淆的更细粒度评估方法。

**🔧 技术方法**

采用因果解码器Transformer（8层、8头、512维），结合RoPE、ALiBi、NoPE三种位置编码；设计任务切换框架生成合成延迟复制数据，使用AdamW优化交叉熵损失。

**📊 数据集**

自定义合成任务集，包含随机任务、完整延迟复制任务和选择性延迟复制任务；在固定长度256的上下文窗口中采样不同延迟距离，训练集覆盖[15,25]，测试集覆盖更宽范围（如[0,50]）。

**📈 对比分析**

通过teacher‑forcing下的next‑token准确率，在训练距离之外的延迟距离上评估模型；计算in‑distribution与out‑of‑distribution精度比，结果显示NoPE性能最佳、RoPE最差；数据多样性提升绝对性能但相对收益递减；迁移学习对不同任务距离分布表现出正向或负向影响。

**⚠️ 局限性**

实验仅在小规模Transformer和合成任务上进行，未验证大模型或真实语言数据；迁移学习机制未深入分析；评估仅基于teacher‑forcing，未考虑自由生成中的误差累积；可能受训练集规模与任务频率设置的影响。

---

## 521. Artificial Id: Drive and Persistent Alignment in Agentic AI

**arXiv ID:** 2609.11911 | [PDF](https://arxiv.org/pdf/2609.11911v1)

**作者:** Yakov Pyotr Shkolnikov `[一作]` `[通讯]` (Independent Researcher), Yakov Pyotr Shkolnikov (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在虚拟 Petri 菌斑实验中，作者提出并验证了一个“人工 id”结构：一个极小的线性控制器在无任务目标、无一般推理的条件下，仅通过对环境中食物位置的持续性（生存时间）进行差异性选择，形成自适应驱动力，进而实现对局部传感的感知与行为适应。

**💡 创新点**

创新点在于将持续性驱动力（id）与通用推理（ego）功能分离，证明在没有外部任务目标或奖励的情况下，差异性持续选择即可产生可用的自适应控制；并提出在持续代理系统中需要一个“持久化的对齐边界”来管理观察、后果、权限与身份等安全/对齐要素。

**🔧 技术方法**

技术实现主要包括：差异性持续选择机制、线性控制器（20 参数）、基于食物位置的生存延迟奖励、基因突变与种群替换、以及对环境字段（粗略估计与局部传感）进行的感知融合；不使用强化学习、奖励模型或大规模训练。

**📊 数据集**

实验数据集完全由仿真生成；使用三种不同环境设置（World 1‑3），每种设置下模拟 3 个随机种子，共 2,048 条种群线索，持续 100 次控制器寿命周期。

**📈 对比分析**

评估方法为测量控制器群体在最终 20 次寿命周期内在“维持区”内停留的比例（occupancy）。在 World 2 及 World 3 的适应实验中，群体可将占用率提升至 0.684‑0.856，远高于盲目常数力基准 0.405 以及手工感知控制器 0.674；实验还展示了在食物映射逆转后，种群可通过持续性驱动快速重适应。

**⚠️ 局限性**

局限性包括：实验仅在极简模拟环境和 20 参数控制器上验证，未证明在更大、更复杂系统中的可扩展性；未测试持续代理与跨任务对齐的完整体系结构；统计样本仅 3 种随机种子，缺乏稳健的假设检验；未探究多代理交互、权限升级、攻击或复制等风险；因此结论仅适用于演示级别，尚需进一步验证。

---

## 522. TART: A Modular Tool for Technique-Aware Audio-to-Tablature Guitar Transcription

**arXiv ID:** 2609.11904 | [PDF](https://arxiv.org/pdf/2609.11904v1)

**作者:** Akshaj Gupta `[一作]`, Gopala Anumanchipalli `[通讯]` (University of California, Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了TART四阶段管线，将吉他音频转为带有表达技巧与可演奏指法的tablature；

**💡 创新点**

首次将音频条件的T5 Encoder-Decoder（AudioFret）与CRNN音频转MIDI、CNN‑BiLSTM技巧分类结合，解决音高冗余和技巧识别，实现零射SOTA；

**🔧 技术方法**

采用高分辨率CRNN进行音频‑MIDI转换，CNN‑BiLSTM进行9类技巧分类，音频条件T5 Encoder‑Decoder完成字符串‑品位分配，Beam Search约束生成，BeatNet估计节拍；

**📊 数据集**

训练整合了GAPS、Guitar‑TECHS、François Leduc、GOAT、SynthTab、DadaGP、IDMT‑SMT‑Chords、EG‑IPT、Magcil等多源数据集，评测于GuitarSet、EGDB、Noisy GuitarSet、Noisy EGDB；

**📈 对比分析**

在四个零射基准上，TART平均Tab F1 54.08%，比TabCNN和Fretting‑Transformer分别提升约25%和8.5%，MIDI转录F50提升6.67点，显示显著性能提升；

**⚠️ 局限性**

局限包括：未识别无音高鼓点；每音符只能标注单一技巧，无法捕捉多重技巧；量化导致和弦分组误差；音频‑MIDI阶段误差造成后续阶段传播误差。

---

## 523. MindTopo: Can Foundation Models Reason in Topological Space?

**arXiv ID:** 2609.11900 | [PDF](https://arxiv.org/pdf/2609.11900v1)

**作者:** Yunfei Ge `[一作]` (Northwestern University), Manling Li `[通讯]` (Northwestern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了名为TopoBench的基准，用来评估大型多模态语言模型在五种拓扑属性（连续性、分离性、顺序、包围性和结性）上的推理与规划能力；

**💡 创新点**

创新点在于：①将认知科学中Piaget提出的拓扑基本属性与形式拓扑学结合，构建两层（推理与规划）的评测框架；②设计13种可编程生成任务，涵盖从图像问题到交互式Gym环境；③系统评估14个多模态LLM，并尝试监督微调、强化学习与视频生成策略，揭示推理与规划之间的显著差距；

**🔧 技术方法**

使用多模态LLM（如Qwen、Gemini、Gemma、InternVL等）、图像/视频生成模型（GPT-Image-2、Wan2.2-I2V-A14B、Seedance-2.0-Mini）、自定义的场景生成器、强化学习框架（GRPO）以及程序化注释与动态检查工具；

**📊 数据集**

使用TopoBench自研的13类任务集合（共11,030个实例，73%推理、27%规划），并对其中11,008个实例进行人工标注；

**📈 对比分析**

与14个LLM的基线推理准确率平均约61%，规划成功率仅为52%或更低，显示出推理-规划差距；监督微调+强化学习在Qwen3-VL-2B-Instruct上将推理准确率提升至约51%，规划成功率仅提升至6%；视频生成策略虽然能达到合理终点，却无法保证拓扑连贯性；

**⚠️ 局限性**

局限性包括：①仅使用程序化渲染场景，缺乏真实世界图像变异；②只评估了3种视频生成模型，缺少更广泛的验证；③只覆盖了Piaget的五个基本拓扑属性，未涉及更复杂的拓扑关系；④生成视频的动态检查样本有限，统计置信度有限。

---

## 524. CausalArena: Benchmarking Causal Discovery in the Foundation Model Era

**arXiv ID:** 2609.11897 | [PDF](https://arxiv.org/pdf/2609.11897v1)

**作者:** Zi-Rong Li `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CausalArena，一个统一且可演进的因果发现基准，涵盖合成、语义操作、公式驱动的结构因果模型（SCM）以及真实表格数据；

**💡 创新点**

创新点在于：①把三类SCM合并为同一可执行接口，支持持续添加新环境；②满足广度、更新性、语义/科学 grounding 与可诊断四大评估需求；③提供细粒度因子分析与预训练曝光评估，揭示方法在不同图结构、机制与干预场景下的表现；

**🔧 技术方法**

使用可执行SCM规范、统一观测与干预采样协议、图结构评价指标（F1、SHD）、基于交叉表的多维度分层评分以及因子交互分析；

**📊 数据集**

数据集包括1200个可执行SCM（1000合成、100语义、100公式），以及若干公开的真实表格数据（Sachs、PetShop等）；

**📈 对比分析**

对18种经典、神经与预训练模型（PC、GIES、CDIS、DAGMA、NOTEARS、AVICI、TabCausal、FoundCause等）进行评估，发现不同方法在合成、语义、公式以及真实数据上排名大幅变动，预训练模型在某些类别上优势显著但易受预训练-评估重叠影响；

**⚠️ 局限性**

局限性包括：①评估仍基于已公开的生成器，可能与未来预训练数据重叠；②仅覆盖无潜在混杂且为平面DAG的表格场景；③对大型图、连续干预、潜在变量或多层因果结构的支持有限；④真实数据的图结构并非唯一验证，存在不确定性。

---

## 525. A Lumpability-Driven Taxonomy of Strong and Weak Stochastic Bisimilarities with Their Congruence Properties

**arXiv ID:** 2609.11893 | [PDF](https://arxiv.org/pdf/2609.11893v1)

**作者:** Riccardo Romanello `[一作]` (University of Udine), Sabina Rossi `[通讯]` (Ca' Foscari University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文在PEPA中定义并比较了六种强弱随机双射（普通/精确/严格），建立了严格包含关系与不相容性，阐述了它们的层次结构，并分析了它们在并发语义下的同构性；

**💡 创新点**

创新点在于首次系统性地区分并证明这六种随机双射的严格包含与互相不相容，提出了完整的分类图并揭示了强精确/严格双射在前缀与选择操作下缺乏同构性的局限；

**🔧 技术方法**

采用基于转移系统、CTMC聚类、算子语义的理论推导与构造反例的证明技术；

**📊 数据集**

未使用实验数据集，全部为形式化证明与示例；

**📈 对比分析**

通过包含关系证明与构造反例对比，未进行性能评估；

**⚠️ 局限性**

局限在于强精确与强严格双射不具备前缀与选择的同构性，且缺乏针对大规模系统的算法实现与复杂度分析。

---

## 526. Truncated Noisy Best-Response Algorithms: Toward Game Theoretic Learning with Safety Guarantees

**arXiv ID:** 2609.11863 | [PDF](https://arxiv.org/pdf/2609.11863v1)

**作者:** Vartika Singh `[一作]` (Blue Yonder), Philip N. Brown `[通讯]` (University of Colorado Colorado Springs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并分析了一类名为 Truncated Noisy Best‑Response（TNBR）的分布式算法，用于求解两代理子模最大化协同问题，并给出了其马尔科夫链的可重现类及性能与安全性界限。

**💡 创新点**

创新点包括：①首次将噪声参数 β 与最佳回应邻域结合，形成一族能够在保证不进入最差纳什均衡的同时，仍保持高质量均衡的算法；②在同一框架下同时提供“性能”（高于 1/2+β）和“安全”（低于 1/2-3β/2）双重最坏情况保障；③揭示性能与安全性之间的“水床”权衡关系；④针对两种常见行动选择规则（logit 与错误规则）在小规模与大规模实例上进行实验验证。

**🔧 技术方法**

技术主要包括：游戏理论与潜在游戏理论、子模函数的性质、马尔科夫链理论、噪声扰动的分析（β 邻域）、最优/次优类的结构化证明、实验仿真（蒙特卡洛采样、平均/最小效用评估）。

**📊 数据集**

实验数据集：①手工构造的小型 4×4 的潜在函数矩阵；②两种 9 个资源、每个代理 50 个动作的最大加权集合覆盖游戏；全部均归一化至 [0,1]。

**📈 对比分析**

比较方法：对不同 β、不同动作选择规则（logit γ=10 与错误规则 p=0）进行参数扫描，记录每次迭代后系统目标的平均值和最小值。结果显示：①在 β 较小的范围内，logit 和错误规则均能保持平均效用接近 1；②随着 β 增大，平均效用下降但最小效用仍保持在理论安全界限以上；③错误规则在低 β 时更易逃离低质量均衡，且在适当 β 下可实现比理论下限更优的平均表现。

**⚠️ 局限性**

局限性：①仅适用于两代理的子模最大化游戏，难以直接推广到多代理情形；②算法需要每步评估所有可行动作的效用，导致在大动作空间上的计算开销大；③理论保证是最坏情况，实际表现受具体游戏结构影响，仍需进一步研究更精细的性能分析。

---

## 527. General Quantification of Covariate and Concept Shifts

**arXiv ID:** 2609.11918 | [PDF](https://arxiv.org/pdf/2609.11918v1)

**作者:** Hongbo Chen `[一作]` (South China University of Technology), Li Charlie Xia `[通讯]` (South China University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了分布偏移下的泛化理论并提供了可估计的误差上界。

**💡 创新点**

提出γ^*-Y|X移位概念并用熵正则化OT统一协变量与概念偏移。

**🔧 技术方法**

利用熵正则化最优输运、Lipschitz连续性、偏差校正估计器与DataShifts算法。

**📊 数据集**

在Novozymes酶预测、ColoredMNIST、PACS以及合成二分类任务上验证。

**📈 对比分析**

与ERM、CORAL、MMD等方法对比，估计误差上界跟测试误差高度相关且比现有理论更紧凑。

**⚠️ 局限性**

主要限制在高维特征下仍受维数灾难影响，并需预先估计Lipschitz常数。

---

## 528. Existence of the Core in Approval-Based Committee Elections

**arXiv ID:** 2609.11912 | [PDF](https://arxiv.org/pdf/2609.11912v1)

**作者:** Patrick Becker `[一作]` (Technical University of Munich), Dominik Peters `[通讯]` (CNRS, LAMSADE, Université Paris Dauphine - PSL)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种新的基于“谐波熵”目标函数的投票规则，并证明该规则在所有投票实例中都能产生满足核心（core）与其强化形式 core+ 的委员会，且该委员会可在多项式时间内计算。

**💡 创新点**

创新点主要在于：① 设计了谐波熵这一新的潜在函数，既与 Shannon 熵相关，又能捕捉支付向量的均匀性；② 利用谐波熵构造的目标函数与支付系统的结合，形成了一种全局优化框架；③ 通过交换边界（candidate addition/deletion）与 Farkas 引理，证明所有局部最优即满足 core+；④ 提供了一个明确的多项式时间局部搜索算法。

**🔧 技术方法**

技术手段包括：
- 线性规划与 Farkas 引理的组合用于证明存在性与核心性；
- 水位填充（water‑filling）与 shift operator Φ 用于分析谐波熵的增量；
- 线性约束与凸优化（尤其是对支付系统的二次目标 G 的最小化）用于获得结构化的最优支付方案；
- 最大流/最小割理论用于证明在删除候选人时能合理分配预留预算。

**📊 数据集**

该工作为理论性质证明，不涉及具体数据集；所有证明均在抽象的投票实例上进行。

**📈 对比分析**

比较方法：将所得的 core+ 结果与已知的 PAV、MES 等规则的近似核心性能进行对比。实验结果表明，谐波熵规则在所有实例中都能得到核心委员会，且相对于已有的近似核心规则（如 PAV 的 2 倍逼近、MES 的对数逼近）提供了更强的核心保证，算法复杂度为多项式时间。

**⚠️ 局限性**

局限性：
- 只证明了 core+ 的可达性，尚未探讨该规则在其他比例代表性公理（如 EJR、EJR+）下的表现；
- 谐波熵函数虽然理论上可行，但在实际大规模实例中求解可能受限于线性规划的规模；
- 对于 Droop 配额的严格情况，需要通过极限逼近实现，理论上可行但实现上可能需细致处理。

---

## 529. From Protocols to Evidence: Bounded Claims for AI in Service of the Common Good

**arXiv ID:** 2609.11910 | [PDF](https://arxiv.org/pdf/2609.11910v1)

**作者:** Nitesh V. Chawla `[一作]` (University of Notre Dame), Paulo Benanti `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并构建了“RISE AI”证据架构，用以评估AI系统在责任、包容、安全与赋权四个维度下的表现，并将评估嵌入制度缺陷（破裂测试）与政治经济背景之中。

**💡 创新点**

创新点在于将制度性缺陷与技术评估相结合，形成“破裂测试”与“RISE AI”四维边界化证据链，强调制度约束与证据限界的双重考量。

**🔧 技术方法**

采用现行法规（欧盟AI法案）、标准（NIST AI RMF、ISO/IEC 42001）以及设计模式（Answerability-by-Design 等）作为技术实现基础，配合案例研究与定性评估。

**📊 数据集**

论文未使用传统机器学习数据集，而是依赖公开案例、法规条文、访谈记录等定性数据进行验证。

**📈 对比分析**

并未给出传统意义上的性能对比，而是通过案例对比与指标评估（如可理解性、可争议性、治理边界）展示框架的可操作性和可扩展性。

**⚠️ 局限性**

局限性包括缺乏大规模实证验证、对政治经济背景的解释依赖主观判断、评估标准可能随上下文变动、且无法单凭技术手段解决价值与权力分配问题。

---

## 530. GPU-CFR: 80x Faster Counterfactual Regret Minimization by Compiling the Game to Static Dataflow and CUDA Graph Replay

**arXiv ID:** 2609.11923 | [PDF](https://arxiv.org/pdf/2609.11923v1)

**作者:** Boning Li `[一作]` (Tsinghua University), Longbo Huang `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将固定的两人零和完美记忆博弈编译为一次性静态数据流，利用深度层级批处理、静态机会折叠、双通道达成概率缓冲和哨兵槽，随后在 GPU 上通过 CUDA Graph 记录并复用整个 CFR 迭代，从而实现了显著加速。

**💡 创新点**

创新点包括：
1) 视固定游戏为程序，编译成 flat 数组和深度级别执行计划；
2) 静态机会折叠将所有机会乘积移至编译期；
3) 双通道达成概率缓冲一次写入两名玩家的达成概率；
4) 采用哨兵槽消除边界分支；
5) 将整个迭代转为可捕获的 CUDA Graph，从而消除每次迭代的内核启动和调度开销；
6) 证明编译后迭代在每一层仅需固定数量的张量运算。

**🔧 技术方法**

技术手段：
- 编译器：遍历游戏树生成节点、边、信息集平面数组并预计算索引；
- 静态机会折叠、深度层级批处理、双通道达成概率缓冲、哨兵槽；
- CUDA Graph 捕获与回放；
- PyTorch（Aten）张量运算、CPU 多线程并行；
- 精度控制：累计器使用 float64，其他缓冲使用 float32；
- 运行时检查、验证层（比对参考实现、独立 Python 版本、最佳响应等）。

**📊 数据集**

数据集：
- 8 公开游戏（OpenSpiel 卡牌、骰子、棋盘游戏）
- 2 个 Heads‑Up No‑Limit Texas Hold’em（HUNL）子游戏
- 4 个扩展扑克子游戏（Libratus 河局等）
- 12 游戏更新规则实验集（用于对比不同 CFR 规则）

**📈 对比分析**

比较方法与性能：
- 采用相同硬件（NVIDIA A100 80GB）和相同软件栈；
- 对比现有最快 GPU 实现（Kim 2026 sequence‑form CFR^+）、CPU 版本 LiteEFG、OpenSpiel；
- 指标包括：每次迭代毫秒、Aten 操作数、GPU 内存占用、训练墙钟时间和可解释性；
- 结果：GPU‑CFR 在 8 游戏套件中每次迭代平均 0.397 ms（≈19×、≈29× 速度提升），Aten 操作数从 1742↓至 96（≈18×）；
- CUDA Graph 进一步提升 1.6×；
- 在最大全局游戏中 GPU‑CFR 的训练墙钟时间比 Kim 低 10 倍、比 LiteEFG 低 8 倍；
- 通过 exploitability 与训练时间曲线验证，GPU‑CFR 在相同阈值下更快达到更低 exploitability。

**⚠️ 局限性**

局限性：
- 仅适用于两人零和、完美记忆的离散博弈；
- 需要一次完整编译，无法动态变更游戏结构；
- 依赖 CUDA Graph，旧驱动或非支持的加速平台可能无法捕获；
- 目前实现为 tabular CFR，未直接支持近似或深度学习强化学习框架；
- 需要手动处理精度问题（累计器使用 float64，其他使用 float32）。

---

## 531. Nuha-Speech: Building General-Purpose Arabic Speech-LLMs

**arXiv ID:** 2609.11892 | [PDF](https://arxiv.org/pdf/2609.11892v1)

**作者:** Yingzhi Wang `[一作]` (Elm Company), Muhammad Alqurishi `[通讯]` (Elm Company)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了覆盖7项任务（ASR、AST、SQA、DI、SER、AR、GR）的1.5M样本阿拉伯语语音指令跟随数据集 Nuha‑Speech，并在此基础上对 Qwen‑Omni 3B、7B、30B 进行两阶段 LoRA 微调，形成 Nuha‑Speech‑3B/7B/30B 模型。

**💡 创新点**

首次提出统一的阿拉伯语语音多任务训练与评估框架；通过大规模生成式数据合成与多语言模型（Qwen3‑32B）自动化构造高质量 SQA、AST、SER 等任务样本；在模型微调中采用两阶段策略（先 ASR 再全任务）并冻结音频编码器，显著提升低资源阿拉伯语表现。

**🔧 技术方法**

使用 Qwen‑Omni 语音‑LLM 系列作为基础模型；采用 LoRA 低秩微调；训练过程中冻结 Whisper‑Large‑v3 / AuT audio encoder；利用 Qwen3‑32B 进行数据合成与翻译；评估采用 LLM‑as‑Judge（LLaMA‑3.3‑70B）与多种指标（WER、BLEU、Gemma 相似度、ACC 等）。

**📊 数据集**

公共阿拉伯语语音数据集：MGB‑2、MASC、SADA、Common Voice、CoVoST‑v2、ADI‑17、ElevenLabs‑Syn；通过 Qwen3‑32B 生成的 MGB‑2 转录的 AST、SQA；使用 ElevenLabs TTS 合成 SER、AR；对 SADA、Common Voice 进行年龄、性别标注，形成 AR、GR 数据；合成多轮 SQA 与情感推理样本。

**📈 对比分析**

对比基线 Qwen‑Omni（3B/7B/30B）与 Nuha‑Speech 微调版本，使用相同的7项测试集和指标。结果显示 Nuha‑Speech‑30B 在 ASR WER、AST BLEU/相似度、SQA LLM‑score、DI/AR/SER/GR ACC 等方面均明显优于基线，尤其在低资源语音识别和非语义任务（如 dialect/age/emotion）提升显著。

**⚠️ 局限性**

仍受限于阿拉伯语语音资源的稀缺，合成数据与自动生成的标签可能引入噪声；模型对极端口音或方言的泛化能力尚未充分验证；评估集仍以少量人工标注样本为主，缺乏大规模公开基准；仅关注文本输出，未覆盖多模态或多语音交互场景。

---

## 532. ABRA: An algorithm which cannot converge to low-quality Nash equilibria

**arXiv ID:** 2609.11889 | [PDF](https://arxiv.org/pdf/2609.11889v1)

**作者:** Vartika Singh `[一作]` (University of Colorado at Colorado Springs), Philip N. Brown `[通讯]` (University of Colorado at Colorado Springs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种名为 Approximate Best Response Algorithm（ABRA）的算法，利用噪声参数 β 和理性参数 p，改进多智能体子模（sub‑modular）最大化问题的 Nash 均衡质量。

**💡 创新点**

核心创新在于：①证明任意两个玩家的有效效用游戏中，ABRA 收敛到任何均衡时的系统目标至少为 1/2 + β；②对收敛到不同重复类（absorbing state、optimal class、sub‑optimal class）的系统目标进行严格下界分析；③揭示噪声参数与理性参数之间的权衡，既能逃离“坏”均衡，又能保持系统目标在 1/2 以上。

**🔧 技术方法**

技术手段包括：
- 游戏理论与潜在游戏（potential game）框架；
- 近似最佳反应动态与马尔可夫链理论；
- 子模函数、非递减与归一化性质的组合证明；
- 通过噪声邻域定义近似最佳反应；
- 期望系统目标与平稳分布的分析；
- 数值仿真验证。

**📊 数据集**

使用的“数据集”为人工构造的系统目标矩阵（例如 4×4 和 3×3 的 W 矩阵），用于在模拟中检验 ABRA 的性能。

**📈 对比分析**

与传统仅使用最佳反应（β=0）的算法相比，ABRA 在收敛到任何均衡时都能保证系统目标 ≥ 1/2 + β，显著提升了已知的 PoA 下界 1/2。数值实验显示，随着理性参数 p 的增加，系统目标期望值上升，且在 β=0.2、p≥0.3 时平均目标值已超过 1/2 + β；同时，通过调节 β 可以在 sub‑optimal 类和 optimal 类之间取得性能折衷。

**⚠️ 局限性**

限制与挑战：
- 只在两玩家有效效用游戏中严格证明结果；n 玩家情况仅给出未证实的上界；
- 需要手动选择噪声 β 与理性 p 的平衡，过大 β 可能降低 optimal 类的最小目标值；
- 对真实大规模分布式系统的可扩展性与收敛速度未做理论分析；
- 目前仅使用人工矩阵进行仿真，缺乏真实数据集验证。

---

## 533. Guided Super-Resolution of Digital Elevation Models with Diffusion-Based Image Generators

**arXiv ID:** 2609.11886 | [PDF](https://arxiv.org/pdf/2609.11886v1)

**作者:** Armand Mihai Nicolicioiu `[一作]` (ETH Zürich), Konrad Schindler `[通讯]` (ETH Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

利用预训练的扩散式生成模型实现高分辨率 DSM 的引导式超分辨率。

**💡 创新点**

将大型互联网规模图像生成模型迁移到地表高程恢复，实现单步推理与可量化不确定性。

**🔧 技术方法**

基于 Stable Diffusion 2.1/3 的潜在扩散网络、VAE 编码/解码器、单步微调与随机采样技术。

**📊 数据集**

使用瑞士中部城市（Zurich、Bern、Basel 等）的高分辨率正射影像和 DSM，外部测试集为德国慕尼黑等城市。

**📈 对比分析**

与 Nearest、Bicubic、Real‑GDSR 等方法比较，RMSE、MedAE、NMAD 下降约 30–35%，MS‑SSIM 提升 0.2 以上，推理速度提升约 2.4×。

**⚠️ 局限性**

对输入分辨率和扩散步骤数的敏感性、单步推理的“回归平滑”效应以及在不同城市建筑风格下的进一步泛化需求。

---

## 534. CoRA-NAS: Coarse Ranking and Anchor-Residual Refinement for Neural Architecture Search

**arXiv ID:** 2609.11884 | [PDF](https://arxiv.org/pdf/2609.11884v1)

**作者:** Yifan Yang `[一作]` (University of New South Wales), Jiaojiao Jiang `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种CoRA-NAS方法，该方法通过先使用零成本代理的等权重一致性投票生成粗略排名，再利用约1%训练预算的锚点曲线残差进行细化，实现在不同结构和规模空间的一致性鲁棒性；

**💡 创新点**

创新点在于：1）完全无标签的多代理一致性门控机制，使方法在各空间自适应；2）将零成本静态信息与少量动态训练曲线残差相结合，形成可学习的残差补偿；3）保持单一配置即可跨四个视觉NAS空间取得最佳最差性能；

**🔧 技术方法**

技术包括零成本代理（#Params、Synflow、jacov、AZ-NAS的expressivity与trainability视图）、等权重rank consensus、目标无关的一致性门控、锚点采样、早期曲线的双参数外推（LLW+pow_3）、ExtraTrees残差回归、操作局部平滑；

**📊 数据集**

使用了NAS-Bench-201（NB201、NB101）、NATS-Bench（SSS、TSS）、TransNAS-Bench-101、NAS-Bench-301（DARTS）以及ViT-Bench-101和NAS-Bench-NLP等公开基准；

**📈 对比分析**

与19种基准方法（包括AZ-NAS、MeCo、Dextr、LIBRA-NAS等）在统一协议下比较，CoRA-Refine在四个视觉空间的平均Spearmanρ达到0.946，最差空间0.715，高于所有对手；在NB201上约1%预算的细化阶段即可获得近乎oracle的排名并显著提升选择精度；

**⚠️ 局限性**

局限性包括：在纯规模空间仅能匹配最佳容量代理而不能超越；对架构编码的空间特定性；零成本阶段对容量主导空间表现不佳；需要少量训练预算且非完全零成本；在非视觉或Transformer空间表现相对逊色。

---

## 535. From Specs to Apps: Verifying and Monitoring Models of Signal and WhatsApp

**arXiv ID:** 2609.11882 | [PDF](https://arxiv.org/pdf/2609.11882v1)

**作者:** Moustafa Said `[一作]` (CISPA Helmholtz Center for Information Security), Robert Künnemann `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 Signal 协议在 Signal Desktop 与 WhatsApp Web 的实际实现进行实时监控，验证其运行行为是否符合正式协议模型。

**💡 创新点**

首次实现了面向生产级消息应用的实时监控框架；提出了统一的可监控模型与可验证模型的双版本设计；揭示了 WhatsApp Web 与 Signal Desktop 在实现细节上的差异。

**🔧 技术方法**

利用 SpecMon 运行时监控引擎、Tamarin 自动化安全分析工具、协议建模与格式字符串、Trace 重写与事件聚合等技术。

**📊 数据集**

采集的实验数据来自 Signal Desktop（TypeScript、Electron）和 WhatsApp Web（浏览器端 JavaScript）在真实通信场景下的网络与加密函数调用日志。

**📈 对比分析**

通过对比未监控与监控下的事件处理时间，发现平均事件处理时间约 0.08–0.34 ms，内存峰值 20–54 MiB；在故障注入实验中成功检测所有协议层错误，未触发明显性能瓶颈。

**⚠️ 局限性**

仍无法检测通过 payload 注入的秘密泄露；覆盖范围受限于手工编写的事件聚合与模型；模型与实现之间仍需人工精细化；对恶意应用的防护不完整。

---

## 536. On the Regularization Landscape for the Linear Recommendation Models

**arXiv ID:** 2609.11876 | [PDF](https://arxiv.org/pdf/2609.11876v1)

**作者:** Dong Li `[一作]` (Kent State University), Bin Ren `[通讯]` (College of William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将现有多种线性推荐算法统一归类，证明它们本质上等价于添加核范数或弗罗贝尼乌斯范数正则化，并在此基础上提出两种低秩闭式解，兼具核范数的闭式低秩特性和弗罗贝尼乌斯范数的表达力。

**💡 创新点**

核心创新在于揭示所有线性推荐模型仅通过不同形式的核范数或弗罗贝尼乌斯范数正则化实现性能提升，并指出核范数正则化因保持奇异向量而受限；进一步提出两种低秩弗罗贝尼乌斯正则化闭式解，突破传统全秩或需迭代求解的局限。

**🔧 技术方法**

采用SVD、重排不等式、Tikhonov正则化、Dropout、变分自编码器等技术，对正则化问题进行等价变形和闭式求解；同时利用ADMM作为对照求解器。

**📊 数据集**

实验数据集包括MovieLens 20M、Netflix Prize和Million Song Dataset。

**📈 对比分析**

通过在Recall@20/50和nDCG@100指标上与EASE、EDLAE、LRR、ALS/WMF、SLIM、CDAE、MULT-DAE、MULT-VAE等基线模型比较，发现低秩弗罗贝尼乌斯闭式解在多数数据集上与ADMM求解和全秩模型相当，甚至更优；而核范数正则化的性能相对较低。

**⚠️ 局限性**

局限性包括核范数正则化保持奇异向量导致预测受限；低秩弗罗贝尼乌斯解在实现零对角约束时仍采用放松方式；对权重排序高度敏感；实验范围仅限三大公开数据集，未验证更大规模或不同领域的泛化能力。

---

## 537. UniMPA: A Unified Memory-Prediction-Action Model via Action-Grounded Transition Modeling

**arXiv ID:** 2609.11875 | [PDF](https://arxiv.org/pdf/2609.11875v1)

**作者:** Wei Li `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了UniMPA，一种统一的记忆-预测-动作模型，通过行动驱动的过渡接口实现视觉语言动作的推理与执行；

**💡 创新点**

创新点在于将未来过渡预测、双向视觉-动作记忆与动作原型细化融合为单一的过渡导向流程，解决过渡模糊、预测-执行不匹配和经验-实现不匹配问题；

**🔧 技术方法**

采用了持久-选择性未来预测、可触发像素预测、视觉-动作双向记忆库、跨模态配对训练和原型偏置流动等技术；

**📊 数据集**

在LIBERO、LIBERO-Plus、RoboTwin 2.0、VLABench等模拟基准以及GALAXEA R1 Lite和AgileX Cobot Magic两大实机平台上进行评估；

**📈 对比分析**

与π_0.5等基线比较，UniMPA在模拟和实机任务中分别提高了约1.7~18.5个百分点，且训练周期仅为π_0.5的25–50%，展现出显著的性能和训练效率提升；

**⚠️ 局限性**

主要局限包括对极端视觉噪声、复杂动态交互的适应仍有限，以及对跨域迁移时仍需进一步改进记忆检索的鲁棒性。

---

## 538. Epistemic orientation predicts legislative effectiveness among members of the US Congress

**arXiv ID:** 2609.11865 | [PDF](https://arxiv.org/pdf/2609.11865v1)

**作者:** Segun Aroyehun `[一作]` (University of Konstanz), David Garcia `[通讯]` (Complexity Science Hub)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了美国国会议员在演讲和推特中的证据取向与直觉取向，并检验其与立法效能的关系。

**💡 创新点**

创新在于将Evidence-Minus-Intuition (EMI) 得分量化为个体层面的语言指标，并发现其跨平台一致性及对立法效能的预测作用。

**🔧 技术方法**

采用LLM评分与语义相似度相结合的混合方法计算EMI，并使用线性混合效应模型对关系进行统计检验。

**📊 数据集**

利用从1873年至2024年的国会记录、2013-2022年Twitter/X 帖子、DW‑NOMINATE 立场、Legislative Effectiveness Score (LES) 等数据集。

**📈 对比分析**

通过混合效应回归比较各模型，EMI 与极端意识形态负相关（b≈-0.12），与LES 正相关（b≈0.16），跨平台相关系数约0.44；模型R²提升至0.13-0.37，表明EMI具显著预测力；EMI AUC 0.825 超过单词嵌入方法。

**⚠️ 局限性**

限制包括缺乏因果推断、样本主要集中在近代、仅衡量文字表达、LLM 可能带来的偏差、未考虑非文本沟通方式。

---

