# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-24 | 今日论文总数: 1014

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Hetero-Net: An Energy-Efficient Resource Allocation and 3D Placement in Heterogeneous LoRa Networks via Multi-Agent Optimization

**arXiv ID:** 2603.20404 | [PDF](https://arxiv.org/pdf/2603.20404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 2. Artificial Intelligence in Experimental Approaches: Growth Hacking, Lean Startup, Design Thinking, and Agile

**arXiv ID:** 2603.20688 | [PDF](https://arxiv.org/pdf/2603.20688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 3. Optimal low-rank stochastic gradient estimation for LLM training

**arXiv ID:** 2603.20632 | [PDF](https://arxiv.org/pdf/2603.20632v1)

**作者:** Zehao Li `[一作]` (Peking University), Yijie Peng `[通讯]` (Peking University)

**通讯引用:** 17413 | [OpenAlex ID](https://openalex.org/A5082863871)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种低秩随机子空间梯度估计方法，兼容IPA与LR两类梯度估计框架，解决LLM训练中高维梯度内存占用与噪声放大的矛盾。

**💡 创新点**

创新点包括：①引入弱无偏的低秩梯度估计并通过随机投影保持梯度方向；②通过分布式优化得到最优投影分布，信息无关时使用Haar–Stiefel采样，信息相关时按梯度方差自适应采样；③提出lazy update策略减少子空间采样带来的额外方差。

**🔧 技术方法**

主要技术手段有随机子空间投影、Haar–Stiefel采样、对角化求解投影分布最优化、低秩梯度重构、lazy update以及Adam等自适应优化器。

**📊 数据集**

实验数据集包括RoBERTa‑large在六个分类基准（SST‑2/5、SNLI、MNLI、RTE、TREC）的微调，以及LLaMA 20M/60M/100M模型在OpenWebText上的预训练。

**📈 对比分析**

与全梯度、Gaussian低秩、Vanilla IPA/LR等基线进行比较。内存峰值从16.7GB降至3.83GB，精度与全梯度相近甚至略优；在预训练任务中，Stiefel低秩IPA的训练/评估损失下降更快、最终收敛损失更低。

**⚠️ 局限性**

局限性：①实例相关方法需估计/采样梯度方差矩阵，计算成本较高；②当梯度有效秩大于子空间维度r时，无法完全消除方差；③强无偏（c=1）时需要更高的方差；④方法主要针对矩阵梯度，非矩阵结构或极端高秩情况尚未验证。

---

## 4. Writing literature reviews with AI: principles, hurdles and some lessons learned

**arXiv ID:** 2603.20235 | [PDF](https://arxiv.org/pdf/2603.20235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 5. Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding

**arXiv ID:** 2603.20739 | [PDF](https://arxiv.org/pdf/2603.20739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 6. Revenue-Sharing as Infrastructure: A Distributed Business Model for Generative AI Platforms

**arXiv ID:** 2603.20533 | [PDF](https://arxiv.org/pdf/2603.20533v1)

**作者:** Ghislain Dorian Tchuente Mondjo `[一作]` `[通讯]` (University of Yaoundé I), Ghislain Dorian Tchuente Mondjo (University of Yaoundé I)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并分析了“将收入共享作为基础设施”（RSI）的商业模式，旨在通过让平台免费提供 AI 基础设施并从开发者应用产生的收入中抽取佣金，降低开发者进入壁垒，提升生态系统创新性。

**💡 创新点**

创新点在于：① 颠覆传统的上游付费逻辑，采用下游收入共享；② 将价值共创、激励机制和多层市场架构统一于一种模式；③ 通过理论框架和博弈论推导出最优佣金率和激励均衡；④ 强调该模式对低收入国家“潜在就业红利”的社会效益。

**🔧 技术方法**

使用了价值共创理论、激励机制分析、两边市场博弈论、以及多维度比较框架来构建与评估 RSI；并通过案例场景（订阅、按需、免费等）说明技术实现路径。

**📊 数据集**

本研究未使用具体实验数据集，而是基于文献综述、行业案例（如 Google AI Studio、OpenAI、Anthropic）和公开统计数据（手机渗透率、就业数据等）进行理论与案例验证。

**📈 对比分析**

通过多维度比较（10个关键维度）对六种商业模型（付费使用、Freemium、订阅、市场化、混合、RSI）进行对比，评估其进入壁垒、风险分担、激励一致性、创新激励等属性。结果显示 RSI 在低门槛、风险共享、利益一致性、创新激励和潜在收益方面均优于传统模型。

**⚠️ 局限性**

局限性包括：① 研究主要为理论推导和案例分析，缺乏实证验证；② 佣金率最优解的博弈模型需要更多数据支持；③ 监管环境（如欧盟 DMA/DSA）的影响未深入探讨；④ 对跨平台适用性的经验验证不足；⑤ 需要进一步研究如何应对技术实施中的交易跟踪与欺诈防控挑战。

---

## 7. Speedup Patch: Learning a Plug-and-Play Policy to Accelerate Embodied Manipulation

**arXiv ID:** 2603.20658 | [PDF](https://arxiv.org/pdf/2603.20658v1)

**作者:** Zhichao Wu `[一作]` (Nanjing University), Yang Yu `[通讯]` (Nanjing University)

**通讯引用:** 9982 | [OpenAlex ID](https://openalex.org/A5100342259)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出 Speedup Patch (SuP)，一种轻量级的外部调度器，能在不重新训练底层动作策略的前提下，通过下采样动作片段来加速机器人执行。

**💡 创新点**

创新点在于：①将加速问题建模为约束马尔可夫决策过程 (CMDP)；②利用学习到的世界模型估计“状态偏差”作为离线约束；③通过离线强化学习（IQL）在合成数据上训练调度器，实现真正的 plug‑and‑play 加速。

**🔧 技术方法**

主要技术包括：Constrained MDP、基于世界模型的状态偏差评估、Recurrent World Model (RWM) 预测机器人状态、Implicit Q-Learning (IQL) 离线强化学习、动作下采样与动作合并策略。

**📊 数据集**

使用了开放式仿真数据集 Libero（40 任务）和 Bigym（20 任务）进行评估，并在真实双臂机器人平台上完成 Arrange Table、Fold Towel、Stack Plates 三个物理实验。

**📈 对比分析**

与基线方法（固定下采样、DemoSpeedup、无加速）对比，SuP 在保持或提升成功率的前提下实现了 1.55–2.17 倍的执行速度提升；在仿真中平均 2.01× 加速，现实中平均 2.17× 加速，且成功率基本不下降。

**⚠️ 局限性**

局限性包括：①需预先收集足够的离线演示数据；②下采样率仅限整数倍，无法实现更细粒度加速；③状态偏差阈值的选择对性能影响大，需要经验调参；④在极度复杂的动态场景或极短时间窗口下，世界模型的预测误差可能导致约束失效。

---

## 8. ME-IQA: Memory-Enhanced Image Quality Assessment via Re-Ranking

**arXiv ID:** 2603.20785 | [PDF](https://arxiv.org/pdf/2603.20785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 9. Memory Over Maps: 3D Object Localization Without Reconstruction

**arXiv ID:** 2603.20530 | [PDF](https://arxiv.org/pdf/2603.20530v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 10. LL-SDR: Low-Latency Speech enhancement through Discrete Representations

**arXiv ID:** 2603.20242 | [PDF](https://arxiv.org/pdf/2603.20242v1)

**作者:** Jingyi Li `[一作]` (Independent Researcher), Cem Subakan `[通讯]` (Laval University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出低延迟非自回归语音增强框架LL‑SDR，利用离散化和VO‑RVQ实现语音与噪声分离，并结合HuBERT判别器进行语义对齐。

**💡 创新点**

采用变分序列残差向量量化VO‑RVQ的三角遮罩结构实现声语音噪声分离，并在非自回归框架下通过离散化提升性能，同时利用HuBERT判别器保持语义一致性。

**🔧 技术方法**

使用VO‑RVQ量化、HuBERT语义判别器、对齐与对比损失、非自回归增强网络、音频编码/解码器以及Spectral clustering评估。

**📊 数据集**

在LibriSpeech‑100/360、DNS挑战语音与噪声、DNSRIR、DEMAND噪声以及DNS 2020测试集上进行训练与评估。

**📈 对比分析**

与连续增强基线（Conv‑TasNet、Demucs、FRCRN）和自回归基线（SELM、MaskSR、AnyEnhance、GenSE、LLaSE‑G1）比较，LL‑SDR在DNSMOS指标上在非自回归模型中实现与自回归相当或更优的整体与信号质量，并在实时因子和模型轻量化方面表现最佳。

**⚠️ 局限性**

未评估传统信号级指标（如PESQ/SI‑SNR/ STOI），对HuBERT教师的依赖可能限制在不同语种或说话人上的泛化，且在极低资源或极端噪声环境下的鲁棒性尚未充分验证。

---

## 11. Does This Gradient Spark Joy?

**arXiv ID:** 2603.20526 | [PDF](https://arxiv.org/pdf/2603.20526v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 12. SozKZ: Training Efficient Small Language Models for Kazakh from Scratch

**arXiv ID:** 2603.20854 | [PDF](https://arxiv.org/pdf/2603.20854v1)

**作者:** Saken Tukenov `[一作]` `[通讯]` (Independent Researcher), Saken Tukenov (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练并发布了一系列专门针对哈萨克语的Llama-架构语言模型（50M–600M参数），并构建了面向该语言的50K词表ByteLevel BPE分词器；

**💡 创新点**

通过从零开始专门化训练、使用本土化分词器以及构建大规模哈萨克语语料库，展示了小型单语模型在低资源语种上的可行性与效率优势；

**🔧 技术方法**

采用LlamaForCausalLM结构（SwiGLU激活、RoPE位置编码、RMSNorm、嵌入共享），AdamW优化器、混合精度训练以及自定义BPE分词器；

**📊 数据集**

使用约9B标记的哈萨克语网络文本语料（18个来源，经过9阶清洗），以及SIB-200、MC QA与Belebele等哈萨克语基准；

**📈 对比分析**

在三项基准上以零样本方式对SozKZ模型与5个参数在0.5B–3B范围的多语言基线进行对比，600M SozKZ在多项选择QA达到约30%准确率，逼近1B Llama-3.2；在SIB-200主题分类上超越所有多语言模型，50M SozKZ已突破Gemma-2B；

**⚠️ 局限性**

模型在知识密集型任务和阅读理解（Belebele）上仍低于大型多语言模型，且训练语料仅限公开网络文本，导致领域覆盖和事实知识受限。

---

## 13. Enhancing LIME using Neural Decision Trees

**arXiv ID:** 2603.20919 | [PDF](https://arxiv.org/pdf/2603.20919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 14. Toward a Multi-View Brain Network Foundation Model: Cross-View Consistency Learning Across Arbitrary Atlases

**arXiv ID:** 2603.20348 | [PDF](https://arxiv.org/pdf/2603.20348v1)

**作者:** Jiaxing Xu `[一作]` (National University of Singapore), Mengling Feng `[通讯]` (National University of Singapore)

**通讯引用:** 12345 | [OpenAlex ID](https://openalex.org/A5022222926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种多视角大脑网络基础模型（MV‑BrainFM），能够处理任意分区图谱并进行跨视角一致性学习；

**💡 创新点**

创新点包括：①基于随机傅里叶特征的图谱编码器实现任意分区映射；②在Transformer中加入基于距离的注意力偏置，引入解剖学先验；③设计无监督跨视角一致性预训练策略，提升跨图谱泛化；

**🔧 技术方法**

采用Transformer+自注意力（Distance‑Aware Self‑Attention）、随机傅里叶图谱编码器、图注意力网络、跨视角一致性、聚类一致性及熵正则等技术；

**📊 数据集**

使用17个fMRI数据集，共计20,786受试者，64,398脑网络，涵盖Schaefer100/200/500、AAL116、Craddock200、BASC122等六种分区；

**📈 对比分析**

在6个下游任务（AD、ASD、ADHD、性别、年龄、MCI）上与14种基线模型（传统GNN、时间序列基础模型、图基础模型等）对比，MV‑BrainFM在单视角与多视角设置均显著优于对手，平均AUC提升约3%至5%，并展示了更强的跨图谱迁移能力和计算效率；

**⚠️ 局限性**

局限性包括：①对超大规模数据的扩展仍待验证；②跨模态（结构连接、EEG等）整合尚未实现；③对少数族群或不平衡标签的数据偏见可能影响性能；

---

## 15. Probing the Latent World: Emergent Discrete Symbols and Physical Structure in Latent Representations

**arXiv ID:** 2603.20327 | [PDF](https://arxiv.org/pdf/2603.20327v1)

**作者:** Liu hung ming `[一作]` `[通讯]`, Liu hung ming

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文使用 A.I. Mother Tongue (AIM) 框架作为被动离散探针，对冻结的 V-JEPA 2 视觉编码器的潜在空间进行符号化，揭示其内部结构；

**💡 创新点**

创新点在于：①通过不训练任何附加模块、保持编码器冻结，解决了传统生成式或判别式探针的归因问题；②采用无任务监督的向量量化，仅借助统计学检验即可确认潜在空间中的物理结构；

**🔧 技术方法**

主要技术包括：VQ‑VAE 的单层向量量化（K=8）、线性投影+层归一化+L2 归一化、EMA 代码簿更新、Straight‑Through 估计以及基于符号分布的统计测试（χ²、互信息、JSD）；

**📊 数据集**

使用 Kinetics‑mini 数据集（5个动作类别，共约48段视频）进行实验，动作类别被视作对比实验的物理维度代理；

**📈 对比分析**

通过在三组对比实验（把手角度、物体几何、运动时间结构）中计算符号分布的 χ²、MI、JSD，并与高斯噪声基线对比，结果显示所有对比实验均显著（p<10⁻⁴），MI 与 JSD 远超基线，表明符号化能捕获潜在空间的物理信息；

**⚠️ 局限性**

主要局限包括：①使用动作类别作为物理变量代理，无法分离其他视觉因素的影响；②代码簿大小有限（K=8），导致主符号冲突，难以观察更细粒度结构；③符号分布基于每个空间令牌的假设独立性，真实样本量受限；④未进行因果干预验证，仅能说明统计相关性。

---

## 16. KV Cache Optimization Strategies for Scalable and Efficient LLM Inference

**arXiv ID:** 2603.20397 | [PDF](https://arxiv.org/pdf/2603.20397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 17. AI Detectors Fail Diverse Student Populations: A Mathematical Framing of Structural Detection Limits

**arXiv ID:** 2603.20254 | [PDF](https://arxiv.org/pdf/2603.20254v1)

**作者:** Nathan Garland `[一作]` (Griffith University), Nathan Garland `[通讯]` (Griffith University)

**通讯引用:** 194 | [OpenAlex ID](https://openalex.org/A5045221111)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将大学AI文本检测建模为一个复合假设检验问题，并证明存在基于学生写作多样性导致的结构性限制。

**💡 创新点**

创新点在于揭示人口多样性在一Shot文本检测中构成不可逾越的大小-功效权衡，并给出子组混合界限与可观测分组指标的关联。

**🔧 技术方法**

使用了总变差距离的变分不等式作为主要数学工具，推导了平均、最坏情况以及子组级别的下界。

**📊 数据集**

论文主要使用理论推导，无实测数据集，讨论已公开的检测工具实验结果作为背景。

**📈 对比分析**

对比方法为理论界限与已有经验的误报率；性能表现表明即使检测器性能高，误报率也受写作分布重叠控制。

**⚠️ 局限性**

局限包括未对多样化写作模型进行经验验证、仅考虑单文本一次性检测、假设AI输出与人类写作分布可知，未涵盖AI辅助写作的连续性。

---

## 18. Grounded Chess Reasoning in Language Models via Master Distillation

**arXiv ID:** 2603.20510 | [PDF](https://arxiv.org/pdf/2603.20510v1)

**作者:** Zhenwei Tang `[一作]` (University of Toronto), Ashton Anderson `[通讯]` (University of Toronto)

**通讯引用:** 3676 | [OpenAlex ID](https://openalex.org/A5048789742)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过 Master Distillation 将专用棋类专家系统（如 Stockfish）的完整推理过程蒸馏到 4B 参数语言模型，实现了高质量、可解释的棋题求解。

**💡 创新点**

①同时蒸馏答案与完整推理链；②Feigned Discovery Prompting 让教师模型“假装不知道答案”，生成真实、可解释的链式思考；③主题平衡采样与 RLVR（可验证奖励）结合，提升学习效率；④展示学生模型超越教师的现象。

**🔧 技术方法**

监督微调（SFT）在 Qwen3‑4B‑Instruct‑2507 上；RLVR（DAPO‑C1）使用可验证奖励；Stockfish 深度 24 计算最佳线；Feigned Discovery Prompting；主题平衡采样、全局数据扩增等技术。

**📊 数据集**

约 39k 题目（来自 Lichess 等公开棋题库），每题附有 FEN、合法走法、主题标签；数据通过主题平衡采样构建，SFT 与 RLVR 采用互不重叠子集。

**📈 对比分析**

与 30+ 公开 LLM（OpenAI GPT‑5、Gemini‑3‑Pro/Flash、Claude 系列、DeepSeek、Qwen3‑Next‑80B 等）以及 15+ 开源模型进行对比；在 900 题测试集上以 pass@1 评估，4B C1 在各难度级别均高于所有开源模型，甚至超过大多数商业模型，最高 53.6% 的 Expert 级别准确率；token 效率约 2 倍级别短。

**⚠️ 局限性**

受训练数据质量和规模限制，难以处理极难或多解题目；RLVR 在 Expert 级别效果不佳，提示链式推理不够可靠；依赖专用专家系统，推广性受限；对高阶多步推理或战略性棋局的泛化尚未验证；缺乏对生成解释可靠性的系统评估。

---

## 19. Fluid Antenna Networks Beyond Beamforming: An AI-Native Control Paradigm for 6G

**arXiv ID:** 2603.20484 | [PDF](https://arxiv.org/pdf/2603.20484v1)

**作者:** Ian F. Akyildiz `[一作]` (Truva Inc.), Tuğçe Bilen `[通讯]` (Istanbul Technical University)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5066357879)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将流体天线配置与传统射频资源管理协同决策的AI原生闭环控制框架，并在多小区仿真中验证其性能。

**💡 创新点**

将天线重定位从单独的物理层优化转变为网络层控制变量，实现天线配置与调度、功率、波束等资源共优化；引入多智能体强化学习实现分布式自适应决策。

**🔧 技术方法**

多智能体强化学习（MARL）、状态抽象与控制时序分离、基于仿真生成的天线位置与用户链路状态、传统波束成形算法。

**📊 数据集**

使用基于七小区六边形网络的合成仿真数据，包括路径损耗、衰落、用户随机分布与中等速度移动，并生成相应的信道、干扰与负载信息。

**📈 对比分析**

与三种基线方案对比：固定天线+传统波束（FAB）、基于信号强度的天线重定位（SDAR）和随机天线选择（RAS）。实验表明，流体天线控制在用户密度增大时可提升总体吞吐量10–15%，并使5%分位数用户吞吐量提高50–70%，同时显著降低互小区干扰并提升公平性。

**⚠️ 局限性**

局限包括：控制开销高、天线重定位硬件延迟限制适应速度、跨小区干扰协调难度大、能耗增加以及与未来AI‑native RAN架构的接口与标准化挑战。

---

## 20. Monocular Models are Strong Learners for Multi-View Human Mesh Recovery

**arXiv ID:** 2603.20391 | [PDF](https://arxiv.org/pdf/2603.20391v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 21. ALARA for Agents: Least-Privilege Context Engineering Through Portable Composable Multi-Agent Teams

**arXiv ID:** 2603.20380 | [PDF](https://arxiv.org/pdf/2603.20380v1)

**作者:** Christopher J. Agostino `[一作]` (NPC Worldwide), Nayan D'Souza `[通讯]` (NPC Worldwide)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套声明式的上下文‑代理‑工具（CAT）数据层，用 YAML 文件统一管理多代理系统中的工具访问与上下文，并在22个本地托管模型上进行评测。

**💡 创新点**

创新点在于通过 NPC、Jinx、Context 三类可编辑文件实现工具权限与行为规范的结构化约束，消除传统基于文本指令的解释性失效，提供最小特权、可追溯的代理设计。

**🔧 技术方法**

技术实现包括基于 LLM 的代理框架、Jinja 模板执行、ReAct 逻辑拆分、工具链编排，以及在本地 Ollama 环境下运行 0.6B‑35B 参数模型。

**📊 数据集**

使用了自构建的 115 任务基准（涵盖文件操作、网络搜索、脚本、多步推理、工具链与代理委托）以及 2,530 次任务执行的实验数据。

**📈 对比分析**

采用相同自然语言指令和文件验证脚本的跨框架无关评测，结果显示不同模型家族间的差异显著，工具使用可靠性是关键；最高分 35B 模型达 88% 通过率，云端 GPT‑4o 等超过 90%，约 80% 的成功率在首次尝试完成。

**⚠️ 局限性**

局限性包括：受限于本地模型规模与工具使用训练，工具目录增大会显著降低准确率；基准难以捕捉持续交互中的上下文丢失和失败模式；框架在更大规模代理网络中的可扩展性尚未验证。

---

## 22. Code-MIE: A Code-style Model for Multimodal Information Extraction with Scene Graph and Entity Attribute Knowledge Enhancement

**arXiv ID:** 2603.20781 | [PDF](https://arxiv.org/pdf/2603.20781v1)

**作者:** Jiang Liu `[一作]` (Wuhan University), Donghong Ji `[通讯]` (Wuhan University)

**通讯引用:** 3939 | [OpenAlex ID](https://openalex.org/A5058877618)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Code-MIE 框架，将多模态信息抽取统一为代码式的输入输出模板，并通过大语言模型实现多任务抽取。

**💡 创新点**

创新点在于：① 用 Python 函数形式的代码模板统一处理多模态任务；② 引入实体属性、场景图和视觉特征的三重知识提升推理；③ 任务无关、简洁的模板显著降低了 hallucination 并提升了跨语言性能。

**🔧 技术方法**

技术栈包括：Qwen3-Max/Qwen3-VL-235B（文本/多模态 LLM）、ViT（视觉特征）、LoRA 微调、场景图生成、实体属性抽取、代码式模板设计与解析。

**📊 数据集**

使用的公开数据集：M^3D（英文/中文）、Twitter‑15、Twitter‑17、MNRE（图像+文本）。

**📈 对比分析**

与六个基线（自然语言模板 + 代码模板）在四个数据集上对比，Code-MIE 在 M^3D 英文 61.03%、中文 60.49%，Twitter‑15 76.04%、Twitter‑17 88.07%、MNRE 73.94%，均领先基线 2–3% 的 F1，达到目前最高水平。

**⚠️ 局限性**

局限性包括：仍需依赖大模型计算资源；对实体边界识别与关系类型多样性仍存在误差；实体属性知识需手工定义，可能不易迁移；仅覆盖文本+图像/视频，其他模态尚未实验。

---

## 23. SDE-Driven Spatio-Temporal Hypergraph Neural Networks for Irregular Longitudinal fMRI Connectome Modeling in Alzheimer's Disease

**arXiv ID:** 2603.20452 | [PDF](https://arxiv.org/pdf/2603.20452v1)

**作者:** Ruiying Chen `[一作]` (Lehigh University), Lifang He `[通讯]` (Lehigh University)

**通讯引用:** 8255 | [OpenAlex ID](https://openalex.org/A5071709543)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种 SDE 驱动的时空超图神经网络（SDE-HGNN），利用神经 SDE 对不规则纵向 fMRI 信号进行连续时间重建，构建动态高阶超图并通过 SDE 控制超图卷积参数演化，以实现阿尔茨海默病病程预测与诊断。

**💡 创新点**

创新点在于：① 将连续时间随机微分方程与超图结构结合，既能捕捉不规则纵向采样的时间连续性，又能刻画多脑区高阶互作；② 引入稀疏可解释机制，学习 ROI 与超边的重要性，实现可解释性与性能提升双赢；③ 在不规则多访数据上统一建模时空演化，避免离散时间假设。

**🔧 技术方法**

使用技术包括：神经 SDE 编码/解码器、超图卷积与归一化、GRU 细化参数、稀疏正则化（L1、熵）、互信息约束、交叉熵分类损失，全部在 PyTorch 实现。

**📊 数据集**

采用公开纵向 rs-fMRI 数据集 OASIS-3（多访约 6 次）和 ADNI（横向诊断任务），并在两大数据集上进行交叉验证。

**📈 对比分析**

与多种基准（DGCNN、DwHGCN、HyperGALE、HGST、Brain-TokenGT、WHGCN、SDEGCN 等）比较，SDE-HGNN 在 OASIS-3 纵向病程预测、ADNI 诊断分类任务中均取得最高 AUC/准确率，特别是在访数增多时性能优势更为明显。

**⚠️ 局限性**

局限性包括：仅使用单模 fMRI，未整合结构 MRI 等多模态信息；对超参数（如 λ1, λ2, λ3）的敏感性较高，需要大量验证；在更大规模或多中心数据上的泛化性能尚未验证。

---

## 24. When Negation Is a Geometry Problem in Vision-Language Models

**arXiv ID:** 2603.20554 | [PDF](https://arxiv.org/pdf/2603.20554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 25. IBCapsNet: Information Bottleneck Capsule Network for Noise-Robust Representation Learning

**arXiv ID:** 2603.20682 | [PDF](https://arxiv.org/pdf/2603.20682v1)

**作者:** Canqun Xiang `[一作]`, Jiaoyan Zhao `[通讯]` (Shenzhen Polytechnic University)

**通讯引用:** 80 | [OpenAlex ID](https://openalex.org/A5064462169)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于信息瓶颈原则的胶囊网络 IBCapsNet，利用一次性变分聚合取代传统的迭代动态路由。

**💡 创新点**

创新点在于将信息瓶颈压缩与类别特定变分自编码器相结合，实现无迭代聚合并通过 KL 正则天然抑制噪声。

**🔧 技术方法**

采用信息瓶颈（Variational Information Bottleneck）与变分自编码器（VAE）技术的胶囊网络架构。

**📊 数据集**

在 MNIST、Fashion‑MNIST、SVHN 和 CIFAR‑10 四个数据集上进行实验，并在四种合成噪声条件下评估。

**📈 对比分析**

与原始 CapsNet 及 LeNet 进行对比，IBCapsNet 在干净数据上保持相近准确率，在噪声下平均提升 17.10%（加性）和 14.54%（乘性），同时训练速度提升 2.54×，推理吞吐量提升 3.64×。

**⚠️ 局限性**

局限性在于主要关注噪声鲁棒性，未突破最先进的分类准确率，对极端或真实世界扰动的泛化尚需进一步验证。

---

## 26. Understanding Contextual Recall in Transformers: How Finetuning Enables In-Context Reasoning over Pretraining Knowledge

**arXiv ID:** 2603.20969 | [PDF](https://arxiv.org/pdf/2603.20969v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 27. Compression is all you need: Modeling Mathematics

**arXiv ID:** 2603.20396 | [PDF](https://arxiv.org/pdf/2603.20396v1)

**作者:** Vitaly Aksenov `[一作]` (Logical Intelligence), Michael Mulligan `[通讯]` (Logical Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究人类数学的可压缩性，将数学定义、引理和定理视作宏，并用自由交换与非交换单子模型描述其层级结构，随后在 Lean4 MathLib 上实验验证压缩与深度的关系。

**💡 创新点**

提出“expansion function”度量宏集压缩后覆盖的球体，证明在自由交换单子中对数稀疏宏可实现指数扩展，而在自由非交换单子中仅能线性扩展；将该理论与 MathLib 的依赖图、wrapped/unwrapped 长度及深度相对应，首次将压缩视为衡量人类数学兴趣的指标。

**🔧 技术方法**

结合组合学与单子增长理论、宏集稀疏性分析、Lean4 内部依赖图构造、统计分布与回归分析，以及 PageRank 风格的兴趣度量，辅以 Kolmogorov 复杂度与逻辑深度的概念对比。

**📊 数据集**

使用 Lean4 版本 MathLib4（约 463,719 条声明）作为实验数据集，并提取其依赖图、wrapped 与 unwrapped 长度及深度信息。

**📈 对比分析**

通过比较 unwrapped 长度与 wrapped 长度及深度的线性/指数关系，检验不同宏集稠密度对应的扩展率；实验结果显示 MathLib 与自由交换单子（对数稠密宏）模型相符，而自由非交换单子模型不匹配，验证了理论中的指数扩展可达性。

**⚠️ 局限性**

仅评估已存在声明的压缩度量，宏集的确切提取仍是开放问题；模型基于单子，未覆盖多重推理和归约过程；实验受限于 MathLib 的规模和结构；无法精确界定人类数学与正式数学的分界，理论与实际数学复杂度可能不完全对应。

---

## 28. Meta-Learning for Repeated Bayesian Persuasion

**arXiv ID:** 2603.20408 | [PDF](https://arxiv.org/pdf/2603.20408v1)

**作者:** Ata Poyraz Turna `[一作]` (Bogazici University), Tamer Başar `[通讯]` (University of Illinois Urbana--Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Meta‑Persuasion 算法，针对重复的贝叶斯劝说（Online Bayesian Persuasion 与 Markov Persuasion Process）在全信息与半信息反馈下实现对多任务的知识迁移，从而显著降低累计失调（regret）。

**💡 创新点**

创新点在于：①首次将元学习（meta‑learning）框架引入信息设计问题；②通过构造 Carathéodory oracle 在信号空间与损失空间之间高效映射；③在 MPP 环境中设计基于自共形屏障的镜像下降与估计器，兼顾失调与约束违背；④给出了理论上更优的任务平均 regret 与约束违背上界，并在最坏情况退回单任务最优率。

**🔧 技术方法**

使用技术包括：在线镜像下降（OMD）、自共形屏障正则化、Exponentially Weighted Online Optimization（EWOO）、Carathéodory 组合、基于经验贝叶斯的估计器、线性规划（Meta‑Opt‑Opt）和 Bandit 线性优化（CTOMD）。

**📊 数据集**

实验数据采用经典的 Judge‑Prosecutor 劝说案例，随机生成先验、发送者与接收者收益，设置两状态/两行动/两层 MPP；通过 20 次实验平均得到任务平均 regret 与违背曲线。

**📈 对比分析**

与非元学习基线相比，Meta‑Persuasion 在任务平均 regret 上显著下降（约 30‑50%），并在全信息与半信息两种反馈下均保持稳定；同时在 MPP 实验中约束违背随任务数增加而迅速衰减。

**⚠️ 局限性**

局限性包括：①算法在信号空间维度高时计算成本随指数增长；②理论和实验均假设线性损失与可共形约束，难以推广至非线性或更复杂的博弈设置；③未考虑前瞻性受众（forward‑looking receivers）或漂移任务环境下的鲁棒性。

---

## 29. Data-driven discovery of roughness descriptors for surface characterization and intimate contact modeling of unidirectional composite tapes

**arXiv ID:** 2603.20418 | [PDF](https://arxiv.org/pdf/2603.20418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 30. End-to-End Optimization of Polarimetric Measurement and Material Classifier

**arXiv ID:** 2603.20519 | [PDF](https://arxiv.org/pdf/2603.20519v1)

**作者:** Ryota Maeda `[一作]` (University of Hyogo), Shinsaku Hiura `[通讯]` (University of Hyogo)

**通讯引用:** 1010 | [OpenAlex ID](https://openalex.org/A5072566373)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种端到端联合优化偏振测量角度与材质分类器的方法，实现了在少量测量下的高精度材质分类。

**💡 创新点**

创新点在于同时学习最优偏振器旋转角度和分类网络，避免完整 Mueller 矩阵估计，显著降低测量成本。

**🔧 技术方法**

使用 Stokes–Mueller 计算、线性极化器与四波片的组合，结合多层感知器（MLP）分类器与 Adam 优化器。

**📊 数据集**

构建了 83 种不同材质（木、金属、树脂、织物、石材）的 Mueller 矩阵数据集。

**📈 对比分析**

与随机旋转、均匀旋转等基线对比，实验显示在 K=2、3 次测量下准确率提升 10–20%，并证明 LP+QWP 方案最优。

**⚠️ 局限性**

局限在于仅在平面实验室样本上验证，未考虑复杂场景和多光源、角度变化，且未结合色彩、纹理等辅助信息。

---

## 31. Children's Intelligence Tests Pose Challenges for MLLMs? KidGym: A 2D Grid-Based Reasoning Benchmark for MLLMs

**arXiv ID:** 2603.20209 | [PDF](https://arxiv.org/pdf/2603.20209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 32. Meeting in the Middle: A Co-Design Paradigm for FHE and AI Inference

**arXiv ID:** 2603.20504 | [PDF](https://arxiv.org/pdf/2603.20504v1)

**作者:** Bernardo Magri `[一作]` (University of Manchester), Paul Gebheim `[通讯]` (Sei Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对云端加密神经网络推理，提出将全同态加密(FHE)方案与模型架构共同定制，利用可预测的推理图和有限的数值范围进行加密参数与模型结构的协同优化

**💡 创新点**

创新点在于：①把FHE从通用计算转为专为推理设计，提前静态估算噪声、预设最佳置换/引导调度；②在NAS过程中将FHE约束（多项式深度、旋转次数）作为目标，推动浅宽网络与低阶多项式激活；③提供了软最大函数的低阶多项式近似并在微基准下证明其同态实现速度可优于纯文本实现

**🔧 技术方法**

技术包括：全同态加密（BFV/CKKS）与SIMD打包；量化感知训练(QAT)与块结构稀疏；多项式激活与softmax近似；静态噪声预算分析与预计算引导方案；NAS与加密约束联合搜索

**📊 数据集**

未给出具体数据集，推测实验在常见NLP/视觉基准（如GLUE、ImageNet）上验证近似误差与性能

**📈 对比分析**

比较方法为对比同态实现与纯文本Rust实现的softmax函数时钟周期，实验显示同态近似在单CPU上仅消耗比纯文本略多的时间；整体模型性能未完整报告，缺乏完整推理链路评估

**⚠️ 局限性**

局限性包括：①仅在微基准层面验证，未展示完整网络推理的加速与精度；②多项式近似带来误差累积，需进一步评估对任务准确率的影响；③引导和旋转操作的全局成本仍高，需更高效的参数调优；④对硬件侧信道与漏洞的安全保障仍未覆盖

---

## 33. Semantic Tool Discovery for Large Language Models: A Vector-Based Approach to MCP Tool Selection

**arXiv ID:** 2603.20313 | [PDF](https://arxiv.org/pdf/2603.20313v1)

**作者:** Sarat Mudunuri `[一作]`, Srinivasan Manoharan `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于向量检索的MCP工具发现框架，能够为LLM动态选择最相关的3-5个工具。

**💡 创新点**

创新点在于将密集向量嵌入和相似度搜索应用于结构化工具选择，而非传统的全量暴露或人工分组。

**🔧 技术方法**

使用了text-embedding-ada-002模型、Milvus向量数据库、GPT-4o以及基于MCP的工具索引和检索流程。

**📊 数据集**

实验基准包含5个MCP服务器（GitHub、Slack、MySQL、Filesystem、Time/Weather）共121个工具和140个查询。

**📈 对比分析**

与全量工具基线对比，K=3时达到99.6% token节省、97.1%命中率、0.91 MRR，检索延迟低于91ms。

**⚠️ 局限性**

局限包括评估规模有限、单一嵌入模型、仅单轮查询、对工具描述依赖强、未在线验证等。

---

## 34. SciNav: A General Agent Framework for Scientific Coding Tasks

**arXiv ID:** 2603.20256 | [PDF](https://arxiv.org/pdf/2603.20256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 35. Simple Projection-Free Algorithm for Contextual Recommendation with Logarithmic Regret and Robustness

**arXiv ID:** 2603.20826 | [PDF](https://arxiv.org/pdf/2603.20826v1)

**作者:** Shinsaku Sakaue `[一作]` `[通讯]` (CyberAgent), Shinsaku Sakaue (CyberAgent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了一种无投影的上下文推荐算法CoRectron，利用二阶感知机更新实现对用户隐藏偏好学习，保持对用户行动的解释性与效率；

**💡 创新点**

核心创新在于利用上下文推荐的尺度不变性（improperness），从而消除在线牛顿步所需的Mahalanobis投影，显著降低计算成本，并在子最优反馈下保持对数次 regret；

**🔧 技术方法**

采用二阶感知机（second‑order perceptron）更新、椭圆势函数、拉普拉斯–米尔格朗定理以及Hilbert空间的运算，支持线性、非线性（核化）上下文模型；

**📊 数据集**

论文未给出具体实验数据集，实验部分在文中引用的“[ref]”中说明；

**📈 对比分析**

与基于在线牛顿步（ONS）以及LightONS的投影方法比较，CoRectron在保持相同 O(d log T) regret 的同时，单轮时间复杂度从 O(d^2) 降至 O(d^2)（无投影）并在子最优反馈下保持鲁棒性，实验显示速度更快且在多种模型下实现更优 regret；

**⚠️ 局限性**

仍需完整的线性优化oracle；在核化实现中每轮时间复杂度为 O(t^2)，对大规模/长时限任务需进一步近似或压缩技术；

---

## 36. AE-LLM: Adaptive Efficiency Optimization for Large Language Models

**arXiv ID:** 2603.20492 | [PDF](https://arxiv.org/pdf/2603.20492v1)

**作者:** Kaito Tanaka `[一作]` (SANNO University), Aya Nakayama `[通讯]` (SANNO University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现AE-LLM框架，自动化选择并组合多种LLM高效技术（如高效注意力、MoE、PEFT、量化等），并在给定硬件与任务约束下生成 Pareto 最优配置。

**💡 创新点**

创新点在于将跨阶段（架构、微调、推理）高效技术联合视为多目标优化问题，构建预测代理模型与约束感知的 NSGA-II 搜索算法，并通过迭代精炼提升搜索效率，最终得到可迁移到多模态模型的通用方案。

**🔧 技术方法**

采用梯度提升树预测代理模型、NSGA-II 演化搜索、层级交叉与约束剪枝、迭代精炼（uncertainty-driven 采样）等技术，结合硬件测量与能耗监控实现全流程自动化。

**📊 数据集**

使用 15 个 LLM（0.5B‑70B）与 10 个 NLP 任务（MMLU、HellaSwag、ARC、GSM8K、HumanEval、AlpacaEval、LongBench、Needle‑in‑a‑Haystack、MT‑Bench、Vicuna Bench），以及在 VLM 任务上评估 LLaVA‑1.5‑7B 与 InternVL‑Chat 在 VQA、COCO Caption、TextVQA 上的表现。

**📈 对比分析**

与默认配置、单阶段最佳、人工专家、EfficientLLM 推荐设置做对比。AE-LLM 在保持准确率差异 ≤1.2% 的前提下，平均在 Latency、Memory、Energy 上提升约 2.8×，大模型（30B‑70B）更高达 3.4×；单阶段优化只能获得 60‑70% 的收益；人工选择落后 15‑25%。

**⚠️ 局限性**

局限性包括：代理模型预测误差导致搜索偏差；硬件环境变化（温度、驱动）影响测量准确性；任务分布与代理训练任务不匹配时可能出现性能退化；跨阶段组合存在负向交互，需要多轮精炼；搜索仍需一定计算资源，虽然比穷举低阶，但对极大模型仍有挑战。

---

## 37. Memory poisoning and secure multi-agent systems

**arXiv ID:** 2603.20357 | [PDF](https://arxiv.org/pdf/2603.20357v1)

**作者:** Vicenç Torra `[一作]` (Umeå University), Maria Bras-Amorós `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 684 | [OpenAlex ID](https://openalex.org/A5021954232)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性分析了Agentic AI和多智能体系统中内存中毒攻击的类型、威胁场景，并针对语义记忆、情节记忆和短期记忆提出了相应的安全防御策略，特别实现了基于私有知识检索的本地推理原型。

**💡 创新点**

创新点在于：
- 将私有信息检索（PIR）与本地推理相结合，形成一种新的语义记忆中毒防御方案；
- 提出了使用哈希链、签名、加密、k-匿名等多种技术的综合安全框架；
- 针对多智能体交互中隐蔽的记忆污染问题，首次提出可信交互、声誉与溯源机制的组合方案。

**🔧 技术方法**

使用的技术包括：
- 哈希与数字签名实现内存完整性检测；
- 加密（对称/公钥）实现私有知识检索与私有推理；
- 私有信息检索（PIR）和k-匿名方案实现对不可信知识库的安全访问；
- 哈希链、不可变存储实现情节记忆的 append‑only 机制；
- 可信传输与传感器/执行器加密防护；
- 可信度量与声誉系统。

**📊 数据集**

数据集：文章未使用公开数据集，而是通过构造的示例知识库（如包含实体、规则的 Horn 子句集）进行原型验证；对比实验主要基于这些自定义数据。

**📈 对比分析**

比较方法：通过实现一个基于 Python 的 Horn 推理引擎与两种私有检索方案（完整 PIR 与单服务器 k-匿名）进行演示，展示了两种方案在安全性与查询成本上的权衡。性能方面未给出量化指标，仅说明完整 PIR 具备信息理论安全性，而 k-匿名方案在成本上更轻量，但安全性相对弱。

**⚠️ 局限性**

局限性：
- 对动态交互导致的短期记忆中毒尚无完整解决方案；
- 真实多智能体环境中协同攻击的检测与防御仍处于探索阶段；
- 目前的安全机制多依赖可信第三方或多服务器部署，单服务器方案安全性有限；
- 对大规模知识库的 PIR 与加密推理的可扩展性未经过实测。

---

## 38. DCG-Net: Dual Cross-Attention with Concept-Value Graph Reasoning for Interpretable Medical Diagnosis

**arXiv ID:** 2603.20325 | [PDF](https://arxiv.org/pdf/2603.20325v1)

**作者:** Getamesay Dagnaw `[一作]` (Griffith University), Alan Wee-Chung Liew `[通讯]` (Griffith University)

**通讯引用:** 17476 | [OpenAlex ID](https://openalex.org/A5088492734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了DCG-Net，一种端到端可解释的医学影像诊断框架，结合概念词典编码、双向交叉注意力和参数化概念图推理；

**💡 创新点**

创新点包括：①双向交叉注意力取代余弦相似度，实现视觉与文本的局部与全局双向对齐；②参数化概念图以PPMI先验初始化，通过稀疏化消息传递学习概念‑值依赖；③概念词典编码统一不一致的医学术语；

**🔧 技术方法**

使用技术包括CLIP ViT‑B/16视觉/文本编码器、双向注意力机制、图神经网络（GNN）参数化概念图、交叉熵与对齐损失、KL一致性约束、稀疏化与温度化；

**📊 数据集**

使用数据集为Fitzpatrick17k皮肤病诊断（22个概念）和Peripheral Blood Cell（PBC）白血球形态（多值概念）；

**📈 对比分析**

与CBM、CEM、ProbCBM、evi-CEM、Mica、CGP等现有CBM方法进行对比，DCG-Net在两数据集上均获得最高诊断准确率与宏F1，并显著提升概念预测F1（如皮肤病Acc 83.44%、F1 72.13%；白血球Acc 99.76%、F1 99.66%）；

**⚠️ 局限性**

局限性包括对概念标签质量依赖较高，稀疏化概念图可能忽略低频关联，模型扩展到大规模概念集合时计算成本升高，且对连续属性建模不足。

---

## 39. An Open Source Computer Vision and Machine Learning Framework for Affordable Life Science Robotic Automation

**arXiv ID:** 2603.20465 | [PDF](https://arxiv.org/pdf/2603.20465v1)

**作者:** Zachary Logan `[一作]`, Daniel Negrón `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个基于U‑Net语义分割与混合密度网络（MDN）逆运动学的低成本、开源机器人框架，用于自动识别并采集培养皿上的微生物菌落

**💡 创新点**

创新点在于将机器视觉与机器学习驱动的逆运动学解耦，利用MDN模型处理逆运动学多解问题，并实现了完整的端到端采样流水线；同时采用极低成本硬件与开源软件，降低实验室自动化门槛

**🔧 技术方法**

使用了U‑Net进行菌落语义分割，MDN预测机器人关节角度；控制层基于ROS 2与Arduino Uno实现机器人运动；摄像头使用SVPro 8MP；训练采用AdamW、SoftMax/Softplus等技术

**📊 数据集**

利用28张标注菌落图像（通过OpenCV增强后扩增至448张）用于U‑Net训练；MDN使用10,000条机器人正向运动学采样数据（位置–关节角度对）

**📈 对比分析**

与传统数值IK方法相比，MDN在Jetson Nano上平均推理时间仅0.93 ms；实验结果显示平均末端执行器位置误差±1 mm，关节角误差±4°；完整采样流程平均耗时48.6 s，U‑Net分割IoU 0.537、Dice 0.596，像素准确率99%

**⚠️ 局限性**

主要局限包括：分割边界不精确导致定位误差，MDN对多解情况的角度误差偏大；系统需人工校准关节参考轴偏差；样本量小、数据增强有限，导致模型泛化能力受限

---

## 40. Understanding Behavior Cloning with Action Quantization

**arXiv ID:** 2603.20538 | [PDF](https://arxiv.org/pdf/2603.20538v1)

**作者:** Haoqun Cao `[一作]` (University of Wisconsin–Madison), Tengyang Xie `[通讯]` (University of Wisconsin–Madison)

**通讯引用:** 292 | [OpenAlex ID](https://openalex.org/A5061481788)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文研究了在连续动作空间中对行为克隆（BC）进行量化（tokenization）时的理论性质，给出了量化误差与统计估计误差如何在有限时限 MDP 中共同传播的分析；

**💡 创新点**

创新点在于将动作量化误差与 BC 的样本复杂度结合，提出了基于增量输入-状态稳定性（P‑IISS）与概率平滑（RTVC/TVC）两种条件下的上界与下界；同时指出了二元量化器与学习型量化器在平滑性上的差异，并给出一种模型增强方法以绕过平滑性假设；

**🔧 技术方法**

主要使用的技术包括：log‑loss 行为克隆的 MLE 统计分析、总变差与 Hellinger 距离的关系、共享噪声耦合、概率增量输入‑状态稳定性理论、随机最优量化理论以及最大耦合的可实现性证明；

**📊 数据集**

文章以理论实验为主，并未使用具体公开数据集，而是构造了可证明的 MDP 例子来展示上界和下界；

**📈 对比分析**

通过与已知的 BC 统计下界对比，作者证明了在满足稳定性和平滑性假设时，其量化误差的依赖仅为 H^2 ε_q（或 H ε_q），并且在模型增强下可进一步降低为 H ε_q；

**⚠️ 局限性**

主要限制包括：需要动态系统满足 P‑IISS 或 EIISS；若量化器导致专家策略非平滑，可能出现 O(H) 的误差放大；模型增强方法需要额外学习转移模型，增加计算与实现复杂度；

---

## 41. GMPilot: An Expert AI Agent For FDA cGMP Compliance

**arXiv ID:** 2603.20815 | [PDF](https://arxiv.org/pdf/2603.20815v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 42. Centrality-Based Pruning for Efficient Echo State Networks

**arXiv ID:** 2603.20684 | [PDF](https://arxiv.org/pdf/2603.20684v1)

**作者:** Sudip Laudari `[一作]` `[通讯]` (Independent Researcher), Sudip Laudari (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过图中心性度量对ESN的回路节点进行剪枝，以提升网络运行效率。

**💡 创新点**

首次将正负权重分离的入度/出度中心性指标（C1、C2、C3）应用于ESN回路结构，形成基于图论的结构化剪枝方法。

**🔧 技术方法**

利用图理论中的入度/出度、正负权重平衡与总连接强度等中心性度量；结合ESN动态更新公式与RMSE评估。

**📊 数据集**

使用合成的Mackey-Glass混沌时间序列和韩国电力负荷真实数据集。

**📈 对比分析**

与未剪枝的原始ESN进行对比，实验显示中度剪枝可将节点数降低25–40%，同时RMSE下降或保持不变，表明性能得到提升。

**⚠️ 局限性**

过度剪枝会破坏信息流，方法仅基于结构未考虑动态适应，未与其他先进时间序列模型直接对比，且实验覆盖范围有限。

---

## 43. The Nature of Technical Debt in Research Software

**arXiv ID:** 2603.20415 | [PDF](https://arxiv.org/pdf/2603.20415v1)

**作者:** Neil A. Ernst `[一作]` (University of Victoria), Ze Shi Li `[通讯]` (University of Oklahoma)

**通讯引用:** 206 | [OpenAlex ID](https://openalex.org/A5067984324)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过混合方法探讨科研软件中的技术债务，尤其提出了新类别“科学债务”，并系统记录其在九个科研软件项目中的出现情况。

**💡 创新点**

创新点在于首次将技术债务细分为科学债务，并揭示其与科研目标、跨学科知识交互的独特影响；同时提供了涵盖多语言、近三万条自我承认技术债务（SATD）的公开数据集。

**🔧 技术方法**

采用关键词检索+人工标注的SATD挖掘技术，并结合主题分析（Taguette、手工编码）进行质性访谈；对SATD进行多标签分类。

**📊 数据集**

使用来自Astropy、Athena、Biopython、CESM、GROMACS、Moose、Elmer、Firedrake、Root等九个开源科研软件项目的代码注释，累计28,680条SATD记录。

**📈 对比分析**

该研究未对算法性能进行对比测试，而是通过统计不同债务类型的占比和访谈主题阐释其影响，显示科学债务在多数项目中占比高达约10%-15%，并与代码质量、科研输出相关。

**⚠️ 局限性**

局限性包括样本偏向公开且活跃的大型科研项目，人工标注易受主观影响，未系统评估未被检索到的SATD，且访谈样本以气候科学为主，可能不具代表性。

---

## 44. Attention in Space: Functional Roles of VLM Heads for Spatial Reasoning

**arXiv ID:** 2603.20662 | [PDF](https://arxiv.org/pdf/2603.20662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 45. Joint Trajectory, RIS, and Computation Offloading Optimization via Decentralized Model-Based PPO in Urban Multi-UAV Mobile Edge Computing

**arXiv ID:** 2603.20238 | [PDF](https://arxiv.org/pdf/2603.20238v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 46. DGNNFlow: A Streaming Dataflow Architecture for Real-Time Edge-based Dynamic GNN Inference in HL-LHC Trigger Systems

**arXiv ID:** 2603.20364 | [PDF](https://arxiv.org/pdf/2603.20364v1)

**作者:** Davendra Maharaj `[一作]` (Georgia Institute of Technology), Matteo Cremonesi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10593 | [OpenAlex ID](https://openalex.org/A5107877192)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了DGNNFlow，一种针对HL-LHC触发系统的实时边缘动态GNN推理流式数据流架构。

**💡 创新点**

创新点包括在FPGA上实时计算边缘嵌入的增强MP单元、节点嵌入广播模块以解决数据依赖、以及支持无预定义边特征的动态图构造辅助设置。

**🔧 技术方法**

技术采用了FPGA高层合成（Vitis/Vivado）、基于FlowGNN的数据流模型、EdgeConv动态GNN算法、以及压缩稀疏行（CSR）存储与流式FIFO通信。

**📊 数据集**

使用DELPHES仿真生成的HL-LHC质子碰撞事件图（16K个图）作为测试数据集。

**📈 对比分析**

与CPU（Intel Xeon Gold 6226R）和GPU（NVIDIA RTX A6000）在单图批量下比较，DGNNFlow在延迟上比CPU快3.2–5.1倍、比GPU快1.6–6.3倍，功耗比GPU低0.22倍、CPU低0.25倍。

**⚠️ 局限性**

限制在于对大规模图仍有内存扩展挑战，且批量推理性能受限，需进一步提升吞吐量与支持更复杂的动态图模型。

---

## 47. Tackling heavy-tailed noise in distributed estimation: Asymptotic performance and tradeoffs

**arXiv ID:** 2603.20728 | [PDF](https://arxiv.org/pdf/2603.20728v1)

**作者:** Dragana Bajovic `[一作]` (University of Novi Sad), Manojlo Vukovic `[通讯]` (University of Novi Sad)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5076009747)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种用于分布式估计未知向量参数的算法，该算法在存在重尾观察和通信噪声的情况下进行估计。

**💡 创新点**

创新点在于引入了一般非线性来增强共识和创新更新部分，从而提高了对重尾噪声的鲁棒性，并提供了几乎确定收敛性和渐近正态性的结果。

**🔧 技术方法**

使用了共识+创新估计器的框架，并在更新规则中引入了非线性映射。

**📊 数据集**

使用了具有无限方差的重尾噪声模型进行理论分析，具体数据集未明确提及。

**📈 对比分析**

与传统线性共识+创新估计器相比，提出的方法在重尾噪声存在时表现出几乎确定收敛性和渐近正态性，而传统方法则无法收敛。

**⚠️ 局限性**

限制在于算法的性能依赖于网络拓扑结构，且在某些情况下可能会引入更多的通信噪声。

---

## 48. Evaluating Uplift Modeling under Structural Biases: Insights into Metric Stability and Model Robustness

**arXiv ID:** 2603.20775 | [PDF](https://arxiv.org/pdf/2603.20775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 49. Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite Imagery via Generative Vision Models

**arXiv ID:** 2603.20697 | [PDF](https://arxiv.org/pdf/2603.20697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 50. CAMA: Exploring Collusive Adversarial Attacks in c-MARL

**arXiv ID:** 2603.20390 | [PDF](https://arxiv.org/pdf/2603.20390v1)

**作者:** Men Niu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yunfeng Lu `[通讯]` (Beihang University)

**通讯引用:** 803 | [OpenAlex ID](https://openalex.org/A5028178400)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CAMA框架，系统性研究了在协作多智能体强化学习（c-MARL）中三种协同对抗攻击：集体恶意代理（CMA）、伪装恶意代理（DMA）和潜伏恶意代理（SMA），并在SMAC II平台上验证其有效性。

**💡 创新点**

创新点包括：①首次从信息共享、攻击触发控制和角色分配三维度定义协同对抗攻击；②构建统一的理论评估指标（破坏性、隐蔽性与成本），并推导三种攻击模式的相互关系；③提出Transformer观测融合、基于值估计的时间门控以及动态角色分配实现协同攻击。

**🔧 技术方法**

技术手段主要包括：多智能体强化学习（MAPPO）框架、TransformerEncoder用于观测融合、PPO及监督回归学习攻击触发门控、KL散度衡量隐蔽性、以及统一的攻击效益函数 J(Π)=αD+βS-γC。

**📊 数据集**

使用数据集：StarCraft II Multi-Agent Challenge（SMAC II）中的四张地图（1c3s6z_vs_1c3s5z、8m、MMM、1c3s5z），每张地图配置不同的单位数与战术复杂度。

**📈 对比分析**

对比方法：与五个基线单恶意代理策略（AMI、IMAP、Wu、Guo、Gleave）在多恶意代理设置下进行对比。评价指标包括对手奖励（越高表示攻击越成功）和曝光强度（越低表示隐蔽性越好）。实验结果显示 CMA 在所有地图上获得最高对手奖励，但曝光强度最高；DMA 与 SMA 在保持较高攻击效果的同时，曝光强度显著下降，SMA 在隐蔽性方面最优。

**⚠️ 局限性**

局限性：①实验仅限于SMAC II环境，未验证跨任务泛化；②攻击门控阈值和检测器参数需要人工调优；③仅考虑恶意代理之间的协同，未探讨与合法代理的混合协同或多阶段攻击；④对大型团队规模与复杂动态环境的可扩展性尚待进一步评估。

---

## 51. Clinical Cognition Alignment for Gastrointestinal Diagnosis with Multimodal LLMs

**arXiv ID:** 2603.20698 | [PDF](https://arxiv.org/pdf/2603.20698v1)

**作者:** Huan Zheng `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**通讯引用:** 16374 | [OpenAlex ID](https://openalex.org/A5023184215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了CogAlign框架，解决胃肠内镜诊断中模型认知与临床路径不对齐以及视觉特征与诊断结果缺乏因果关联的问题。

**💡 创新点**

创新点包括：①构建分层临床认知数据集并通过SFT将专家诊断逻辑内化；②理论证明SFT易产生背景捷径并提出基于反事实的GRPO强化学习，利用反事实掩码和临床认知奖励实现因果归因；③将三大奖励（格式、认知、诊断一致性）结合，实现诊断过程的结构化与因果可解释性。

**🔧 技术方法**

技术栈包括多模态大型语言模型（如Qwen3‑VL、Gemini 3）、视觉编码器、SFT + LoRA微调、Group Relative Policy Optimization (GRPO) 强化学习、反事实样本生成、结构化奖励机制。

**📊 数据集**

使用自研层级临床认知数据集（24,515例，包含23单标签+49多标签病理），采集自CrohnIPI、GastroVision、HyperKvasir、Kvasir‑Capsule、SEE‑AI Project等公开胃肠内镜数据。

**📈 对比分析**

在5个公开基准（CrohnIPI、GastroVision、HyperKvasir、Kvasir‑Capsule、SEE‑AI Project）上与大模型、医学专用模型、小型模型进行公平对比，CogAlign‑8B平均准确率达67.7%，显著超过Gemini 3 Pro、GPT‑5、Qwen3‑VL‑Plus等模型。

**⚠️ 局限性**

局限性包括：①对数据集规模和标注质量依赖较大，需更多多模态、多标签数据；②反事实掩码和奖励设计依赖人工专家审阅；③训练成本高，模型对极端光照或模糊场景的鲁棒性仍待验证；④在真实临床部署前需要进一步验证安全性与可解释性。

---

## 52. Lean Learning Beyond Clouds: Efficient Discrepancy-Conditioned Optical-SAR Fusion for Semantic Segmentation

**arXiv ID:** 2603.20811 | [PDF](https://arxiv.org/pdf/2603.20811v1)

**作者:** Chenxing Meng `[一作]` (Nanjing University of Aeronautics and Astronautics), Mingqiang Wei `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4282 | [OpenAlex ID](https://openalex.org/A5051555459)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向云遮蔽的光学‑SAR语义分割框架 EDC，利用高效编码器与差异条件化融合实现云容忍度提升。

**💡 创新点**

创新点包括：①使用 Carrier Token 的多尺度编码器实现低复杂度全局上下文建模；②差异条件化混合融合（DCHF）以像素级跨模态差异为可靠性指引，抑制云噪声；③双任务学习与教师蒸馏的辅助云去除分支，提升语义一致性。

**🔧 技术方法**

技术方法涵盖：Transformer 变体（FasterViT/HAT）、Carrier Token 机制、差异条件化注意力、加权 GAP、CBAM 形空间门控、双任务解码、教师蒸馏与云去除损失。

**📊 数据集**

使用公开云遮蔽光学‑SAR数据集 M3M-CR 与 WHU-OPT-SAR 进行训练与评估。

**📈 对比分析**

与多种基线（DCSA‑Net、MCANet、AMM‑FuseNet、FTransUNet、CMX、CloudSeg 等）对比，EDC 在 mIoU 上提升约 0.56%/0.88%，mPA 提升约 2.2%/1.3%，参数量减少 46.7%，推理速度提升 1.98×，并在校准（ECE）与云遮蔽子集上表现更稳健。

**⚠️ 局限性**

局限性：仍需 SAR 辅助，无法单独使用光学数据；训练过程需要配对云与无云样本，且对极端高云量或非典型云纹理的泛化尚未完全验证；模型虽高效但相较更轻量化方法仍有参数与 FLOPs 开销。

---

## 53. SLE-FNO: Single-Layer Extensions for Task-Agnostic Continual Learning in Fourier Neural Operators

**arXiv ID:** 2603.20410 | [PDF](https://arxiv.org/pdf/2603.20410v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 54. Byte-level Object Bounds Protection

**arXiv ID:** 2603.20347 | [PDF](https://arxiv.org/pdf/2603.20347v1)

**作者:** Piyus Kedia `[一作]` `[通讯]` (Indian Institute of Information Technology), Piyus Kedia (Indian Institute of Information Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 PRISM 的零查找对象边界保护机制，通过将对象结束地址压缩到指针的未使用标记位，实现精确且高效的越界检测。

**💡 创新点**

创新点包括：利用 17 位标记位压缩 47 位结束地址，定义 KSA‑Invariant 使得不需要元数据查找即可完成大多数检查；引入 q‑padding 优化，安全地删除对常量偏移访问的检查；提供 32‑bit 变体，进一步降低开销；同时支持部分结构体和结束地址的创建。

**🔧 技术方法**

采用的技术包括：指针标记与压缩元数据、基于帧的堆布局（小帧/大帧）、SSA 中的 KSA 计算、动态边界检查、LLVM 代码插桩与重构、q‑padding 以及 32‑bit 内存模型的实现。

**📊 数据集**

实验使用的数据集包括 SPEC CPU 2017 基准、Phoenix 服务器基准、Apache HTTP 服务器、BugBench 漏洞集以及真实 CVE（如 nginx、memcached）等。

**📈 对比分析**

通过与 CGuard、ShadowBound、Pow2、AddressSanitizer 等现有技术对比，测量 CPU 开销、内存占用和吞吐量：在 SPEC 2017 上，q‑padding 为 32 字节时平均 CPU 开销约 46.1%，在 Apache 上吞吐量仅下降 11.1%；内存占用相对 modest；整体性能优于大多数对比方法。

**⚠️ 局限性**

局限性包括：需要对逃逸指针进行源代码级别的微调；仍然存在 30–50% 的 CPU 开销；仅适用于 x86_64 上未使用的标记位；32‑bit 变体限制了地址空间；处理大对象时仍需额外逻辑；q‑padding 需要根据程序特性手动调优。

---

## 55. Reasoning Traces Shape Outputs but Models Won't Say So

**arXiv ID:** 2603.20620 | [PDF](https://arxiv.org/pdf/2603.20620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 56. Jigsaw Regularization in Whole-Slide Image Classification

**arXiv ID:** 2603.20386 | [PDF](https://arxiv.org/pdf/2603.20386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 57. Scene Representation using 360° Saliency Graph and its Application in Vision-based Indoor Navigation

**arXiv ID:** 2603.20353 | [PDF](https://arxiv.org/pdf/2603.20353v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 58. FactorSmith: Agentic Simulation Generation via Markov Decision Process Decomposition with Planner-Designer-Critic Refinement

**arXiv ID:** 2603.20270 | [PDF](https://arxiv.org/pdf/2603.20270v1)

**作者:** Ali Shamsaddinlou `[一作]`, Morteza NourelahiAlamdari `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

结合因式化 POMDP 与规划‑设计‑评审三人协作流程，从自然语言描述生成可运行的游戏模拟代码。

**💡 创新点**

在每个因式化步骤中嵌入 Planner–Designer–Critic 互动，提供结构化评分和检查点回滚，使代码质量显著提升。

**🔧 技术方法**

使用 LLM（如 GPT‑4）+ OpenAI Agents SDK、SQLite 事务式会话管理、结构化评分模板、MVC 代码拆分与因式化上下文。

**📊 数据集**

在 PyGame Learning Environment（PLE）基准上评测，涵盖 8 个 2D RL 游戏。

**📈 对比分析**

与 Vanilla、Self‑Debug、CoT+Debug、FactorSim、AgentCoder 等基线对比，FactorSmith 的系统测试通过率提升至 0.82，优于 FactorSim 的 0.78，且在需要复杂状态交互的游戏中提升 7–8pp。

**⚠️ 局限性**

仅支持 2D PyGame，生成过程成本和延迟高；Critic 的评分仍依赖 LLM，可能出现误判；缺乏基于执行的反馈。

---

## 59. Agentproof: Static Verification of Agent Workflow Graphs

**arXiv ID:** 2603.20356 | [PDF](https://arxiv.org/pdf/2603.20356v1)

**作者:** Melwin Xavier `[一作]` (Lulea Technical University), Midhun Xavier `[通讯]` (Independent Researcher)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文实现了对四大主流代理框架（LangGraph、CrewAI、AutoGen、Google ADK）工作流图的自动提取、统一抽象建模，并对其进行结构化缺陷检查和基于安全片段 LTL 的时序策略静态/运行时验证；

**💡 创新点**

创新点在于：① 通过框架 API 自动抽取统一的抽象图模型，消除手动建模成本；② 设计了六项结构检查并生成 witness 跟踪；③ 提供了覆盖安全 LTL 片段的 DSL，并将其编译为 DFA 进行图×DFA 交叉检验，支持预部署静态验证；

**🔧 技术方法**

采用图算法（BFS/DFS、逆向可达）进行结构检查；利用抽象图模型与 DFA 编译实现时序验证；对框架 API 进行 AST 解析和运行时接口解析实现提取器；运行时监控采用事件流与 DFA 状态表实现；

**📊 数据集**

使用 18 个人工构造的工作流作为评测集，覆盖四大框架，节点数 5–12，边数 4–14，包含常见结构缺陷与政策违规；

**📈 对比分析**

通过与现有运行时 guardrail 工具对比，评估结构缺陷检测率、时序策略满足率；在最大 5,000 节点图下，结构检查时间 <0.1 秒，验证完成在子秒级；15 条时序策略全部通过，验证效率高；

**⚠️ 局限性**

局限性：仅验证拓扑层安全，无法保证 LLM 输出语义正确；提取器对人类节点识别依赖命名约定，易产生误判；对运行时图动态变更缺乏即时再验证；DSL 仅覆盖安全片段 LTL，缺少更复杂的时序表达；评测集规模小，缺乏真实生产工作流的普适性验证。

---

## 60. Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference

**arXiv ID:** 2603.20224 | [PDF](https://arxiv.org/pdf/2603.20224v1)

**作者:** Patrick Wilhelm `[一作]` (BIFOLD), Odej Kao `[通讯]` (Technische Universität Berlin)

**通讯引用:** 4752 | [OpenAlex ID](https://openalex.org/A5042349846)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估不同大语言模型（LLM）在输入处理和自回归输出中的能耗，并提出基于任务复杂度的能耗-准确率平衡的动态路由机制。

**💡 创新点**

创新点在于引入 Energy‑per‑Token 量化指标、发现模型能耗随输入/输出 token 数呈非线性变化、并设计以能耗曲线为依据的自适应查询路由。

**🔧 技术方法**

使用 Chain‑of‑Thought（CoT）推理、Majority Voting（MV）多候选投票、Transformer 自回归解码、NVIDIA Management Library 进行 GPU 能耗测量，以及动态推理优化技术。

**📊 数据集**

使用的评测数据集为 MMLU（多任务理解）和 MT‑Bench（多轮对话）。

**📈 对比分析**

通过在同一硬件（NVIDIA L40S）上对相同输入执行不同模型与推理策略，测量能耗与准确率，结果显示 CoT 在某些任务（如 Math）能提升 281% 以上准确率，但能耗提升 120‑150×；8B LLaMA 在 35‑65% 能耗增长下取得 47‑350% 的准确率提升，显示更优的能耗/准确率折衷。

**⚠️ 局限性**

局限性包括：实验仅在单一 GPU 与单线程（batch=1）条件下进行，未覆盖多 GPU/并行情况；仅关注能耗而未直接测算碳排放或区域差异；路由策略基于先验测算，实际部署中可能需进一步校准。

---

## 61. Beyond Detection: Governing GenAI in Academic Peer Review as a Sociotechnical Challenge

**arXiv ID:** 2603.20214 | [PDF](https://arxiv.org/pdf/2603.20214v1)

**作者:** Tatiana Chakravorti `[一作]` (Pennsylvania State University), Sarah Rajtmajer `[通讯]` (Pennsylvania State University)

**通讯引用:** 1397 | [OpenAlex ID](https://openalex.org/A5082663800)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开展了混合方法研究，结合对448条社交媒体帖子（LinkedIn、Reddit、Twitter）的话语分析与对14名 AI/HCI 会议负责人（AC/PC）的深度访谈，探讨生成式 AI 在学术同行评审中的使用、风险与治理方式。

**💡 创新点**

创新点在于：①将 GenAI 介入的同行评审视为一个社会技术治理挑战，而非单纯工具使用；②发现并阐释结构性负荷（审稿人过载、提交量激增）是 AI 采用的主要驱动力；③提出“支持‑判断”边界的制度化治理框架，并针对不同角色（审稿人、AC/PC、会议/期刊、AI 工具设计者）给出可操作的治理建议。

**🔧 技术方法**

使用了质性方法：话语分析、主题编码、访谈录音转录与理论结合的编码；并未使用任何深度学习或统计模型。

**📊 数据集**

数据集包括：①448条公开英文社交媒体评论（按关键词检索，时间 2025‑09 至 2026‑08）；②14 份面向 AI/HCI 会议 AC/PC 的半结构化访谈记录。

**📈 对比分析**

研究并未采用传统的实验或性能对比，而是通过三角推理（社交媒体话语、访谈观点、文献梳理）得出共识与差异；没有量化指标，主要以主题出现频次和案例支持来说明研究发现。

**⚠️ 局限性**

局限性包括：①样本主要为英语、美国学术背景，可能不具备跨文化普适性；②社交媒体样本为关键词检索的便利样本，缺乏统计代表性；③研究聚焦 AI/HCI 会议，结果对期刊或其他学科领域的迁移性有限；④研究时间点固定，快速变化的 GenAI 技术与治理环境可能导致结论随时间演进。

---

## 62. Hybrid Autoencoder-Isolation Forest approach for time series anomaly detection in C70XP cyclotron operation data at ARRONAX

**arXiv ID:** 2603.20335 | [PDF](https://arxiv.org/pdf/2603.20335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 63. Neural Autoregressive Flows for Markov Boundary Learning

**arXiv ID:** 2603.20791 | [PDF](https://arxiv.org/pdf/2603.20791v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 64. The Global-Local loop: what is missing in bridging the gap between geospatial data from numerous communities?

**arXiv ID:** 2603.20305 | [PDF](https://arxiv.org/pdf/2603.20305v1)

**作者:** Clément Mallet `[一作]` (Univ Gustave Eiffel), Ana-Maria Raimond `[通讯]` (Univ Gustave Eiffel)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

探讨全球-本地地理空间数据融合中单向依赖与缺失双向反馈的问题，并通过系统分析不同尺度、不同数据类型的交互，提出“全球-本地循环”框架，强调对称化融合和跨社区协同的重要性。

**💡 创新点**

创新点在于提出需要对多源融合进行对称化、双向交互的思路，揭示文本与局部数据在现有研究中的低利用率，倡导以数据为中心而非模型为中心的融合策略，强调跨尺度、跨社区的“全球-本地循环”。

**🔧 技术方法**

结合深度学习、基础/主题模型、迁移学习、联邦学习、混合专家策略、知识图谱以及可解释AI技术，讨论如何实现跨尺度、跨社区的数据融合与反馈。

**📊 数据集**

主要关注公开卫星影像（如Sentinel）、OpenStreetMap、社会媒体文本、地理数据库、IoT传感器等通用数据源，但论文未给出具体实验数据集。

**📈 对比分析**

本文未开展实验比较，主要通过文献综述阐述现有方法的优缺点，缺乏性能评估；因此无法给出具体的性能对比。

**⚠️ 局限性**

研究仍停留在概念与方向层面，缺乏实证验证；需要构建跨社区、跨尺度的标准基准与评估框架，并进一步探索双向交互机制。

---

## 65. WebNavigator: Global Web Navigation via Interaction Graph Retrieval

**arXiv ID:** 2603.20366 | [PDF](https://arxiv.org/pdf/2603.20366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 66. SkyHOST: A Unified Architecture for Cross-Cloud Hybrid Object and Stream Transfer

**arXiv ID:** 2603.20512 | [PDF](https://arxiv.org/pdf/2603.20512v1)

**作者:** Muhammad Arslan Tariq `[一作]` (University of Luxembourg), Pascal Bouvry `[通讯]` (University of Luxembourg)

**通讯引用:** 10336 | [OpenAlex ID](https://openalex.org/A5058311932)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出SkyHOST，一个统一的跨云对象与流式数据传输架构，支持对象→流、流→流两种模式。

**💡 创新点**

创新点在于将Skyplane的批量传输与Kafka等流处理无缝融合，采用URI路由和统一CLI，实现批量与流式在同一控制平面下自动切换。

**🔧 技术方法**

采用Skyplane框架、DAG式运算器、并行TCP传输、微批处理、URI路由、Kafka与S3交互等技术。

**📊 数据集**

使用EEA Copernicus ERA5-Land卫星影像数据以及欧洲环境监测传感器实时流。

**📈 对比分析**

与Confluent Replicator及S3 Source Connector比较，SkyHOST在批量传输中达131.6 MB/s、流复制中达123 MB/s，模型预测误差≤4%，且大幅简化运维。

**⚠️ 局限性**

局限性包括对记录级高吞吐低延迟流式传输的性能低于专用连接器、未实现流→对象模式，且需进一步优化格式解析和多网关扩展。

---

## 67. PARHAF, a human-authored corpus of clinical reports for fictitious patients in French

**arXiv ID:** 2603.20494 | [PDF](https://arxiv.org/pdf/2603.20494v1)

**作者:** Xavier Tannier `[一作]` (Sorbonne Université), Emmanuel Bacry `[通讯]` (Health Data Hub)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

创建了一个匿名可分享的法语临床文本语料库PARHAF，包含由医师撰写的虚构病历。

**💡 创新点**

通过结合临床专业知识与国家健康数据系统统计，采用预设场景与模板生成真实感强、可代表真实医院分布的合成病历，并实现开放共享。

**🔧 技术方法**

采用结构化协议、人工作者与同行评审、基于SNDS的分布抽样、人工检查与自动Sanity‑Check流程，并提供JSON/Parquet数据格式。

**📊 数据集**

使用法国国家健康数据系统(SNDS)的住院索赔数据来定义病例分布与场景。

**📈 对比分析**

本工作未直接评估模型性能，但提供的数据可用于下游NLP任务，使用基准模型可与MIMIC等公开数据集对比，预期表现与真实数据相近。

**⚠️ 局限性**

仅覆盖部分专业与场景，不能用于真实临床决策或法规评估；数据为虚构，缺乏真实患者多样性与长期随访，不能推断实际医院流行病学。

---

## 68. MANA: Towards Efficient Mobile Ad Detection via Multimodal Agentic UI Navigation

**arXiv ID:** 2603.20351 | [PDF](https://arxiv.org/pdf/2603.20351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 69. Mix-and-Match Pruning: Globally Guided Layer-Wise Sparsification of DNNs

**arXiv ID:** 2603.20280 | [PDF](https://arxiv.org/pdf/2603.20280v1)

**作者:** Danial Monachan `[一作]` (Brandenburg Technical University), Christian Herglotz `[通讯]` (Brandenburg Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 Mix-and-Match Pruning 的全局引导层级稀疏化框架，利用敏感度评分和结构感知稀疏范围自动生成多种稀疏配置，供边缘设备部署使用。

**💡 创新点**

创新点在于：①将层级稀疏范围与敏感度评分结合，生成可覆盖不同压缩-精度折衷的十种稀疏策略；②一次性计算敏感度后即可复用，避免多次完整剪枝过程；③通过归一化层不剪枝、按层尺寸动态限制稀疏度等结构化规则，使剪枝更可靠。

**🔧 技术方法**

主要技术包括：权重敏感度计算（幅值、梯度或两者乘积）、基于规则的稀疏区间分配、策略采样（最小、最大、中值、比例插值、结构化插值）、基于掩码的无结构剪枝和微调。

**📊 数据集**

实验数据集包括 CIFAR-10、GTSRB、CIFAR-100，使用 VGG‑11、ResNet‑18、LeViT‑384、Swin‑Tiny 四种网络进行评估。

**📈 对比分析**

与 MAG、SNIP、GraSP、LPViT、GETA 等经典剪枝方法在相同稀疏率下对比，Mix-and-Match 在 VGG‑11、ResNet‑18、LeViT‑384 上性能相当或更好，在 Swin‑Tiny 上准确率下降仅 2.38%（比单一准则剪枝下降 40% 更小），且可一次性生成十个不同稀疏-精度点，显著提升探索效率。

**⚠️ 局限性**

局限性包括：仍为无结构剪枝，对硬件加速支持有限；对极大模型或新型架构的适用性未完全验证；需要一个小型校准集来计算敏感度，若数据分布差异较大可能影响效果。

---

## 70. ContractSkill: Repairable Contract-Based Skills for Multimodal Web Agents

**arXiv ID:** 2603.20340 | [PDF](https://arxiv.org/pdf/2603.20340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 71. Implementing Robust M-Estimators with Certifiable Factor Graph Optimization

**arXiv ID:** 2603.20932 | [PDF](https://arxiv.org/pdf/2603.20932v1)

**作者:** Zhexin Xu `[一作]` (Northeastern University), David M. Rosen `[通讯]` (Northeastern University)

**通讯引用:** 1400 | [OpenAlex ID](https://openalex.org/A5082166133)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

实现了将可证实因子图优化嵌入适应性重加权M估计框架，从而在鲁棒定位与地图构建中提供全局最优的内层WLS解

**💡 创新点**

创新点在于将可证实因子图优化（Shor松弛+Burer-Monteiro+Riemannian Staircase）直接集成进GNC续投机，使得不需要针对每个问题手工设计可证实算法即可获得全局最优子问题解

**🔧 技术方法**

采用了Black–Rangarajan双重性、GNC（渐进非凸）策略、Shor松弛转SDP、Burer‑Monteiro低秩因子化、Riemannian Staircase、以及GTSAM因子图求解器

**📊 数据集**

使用Intel PGO数据集和Trees地标SLAM数据集进行实验

**📈 对比分析**

与传统局部LM+GNC做对比，随机初始化时鲁棒性更高，估计误差更小；尽管求解时间略长，但在离群点比例较高时仍保持较低误差

**⚠️ 局限性**

主要限制在于计算开销相对更大，对高噪声/离群点严重的场景中低秩紧致性可能失效，导致可证实性无法保证

---

## 72. Bypassing Document Ingestion: An MCP Approach to Financial Q&A

**arXiv ID:** 2603.20316 | [PDF](https://arxiv.org/pdf/2603.20316v1)

**作者:** Sasan Mansouri `[一作]` (University of Groningen), Fabian Woebbeking `[通讯]` (Halle Institute for Economic Research)

**通讯引用:** 54 | [OpenAlex ID](https://openalex.org/A5112669930)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于MCP的服务器，将LSEG的API包装为可被LLM调用的工具，并在FinDER基准上评估其金融问答性能。

**💡 创新点**

创新点在于用MCP实现直接数据访问，绕过传统RAG中复杂的文档解析与向量检索，公开了可复现的MCP服务器与工具链。

**🔧 技术方法**

技术实现包括MCP（Model Context Protocol）工具调用、GPT‑4o mini生成模型以及RAGAS评估框架，用以衡量上下文相关性、回答根源性与答案准确性。

**📊 数据集**

使用的数据集为FinDER金融问答基准，并对其进行轻微的预处理（标签统一、缺失值填充等）以配合工具调用。

**📈 对比分析**

评估方法基于RAGAS三项指标：Context Relevance、Response Groundedness 与 Answer Accuracy；在金融子集无工具时准确率为69.7%，在高质量上下文（CR≥0.75）时可提升至80.4%。

**⚠️ 局限性**

局限性包括：仅针对量化问题有效，定性与文档特定信息难以覆盖；缺乏专门为MCP设计的人工标注基准；评估结果受评判模型与prompt微调影响，工具集合覆盖面有限。

---

## 73. Verifiable Error Bounds for Physics-Informed Neural KKL Observers

**arXiv ID:** 2603.20434 | [PDF](https://arxiv.org/pdf/2603.20434v1)

**作者:** Hannah Berin-Costain `[一作]` (University of Waterloo), Jun Liu `[通讯]` (University of Waterloo)

**通讯引用:** 58143 | [OpenAlex ID](https://openalex.org/A5100450180)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了学习式 KK 观测器的可计算状态估计误差上界，利用 PINN 学习变换及其左逆，并通过神经网络验证得到误差上界。

**💡 创新点**

创新点在于误差上界仅依赖可证实的三项量：PDE 残差、左逆的 Lipschitz 常数以及重构误差，避免了传统分析中不可计算的项。

**🔧 技术方法**

采用的技术包括 Physics‑Informed Neural Network (PINN) 训练 KKL 转换、α,β‑CROWN 神经网络可验证技术、Lyapunov 方程求解与 Young 不等式推导误差上界。

**📊 数据集**

实验数据集为两个非线性基准系统：反向 Duffing 振子和 Van der Pol 振子，通过仿真产生训练与验证样本。

**📈 对比分析**

与无误差仿真结果比较时，计算得到的上界分别为 0.181（Duffing）和 0.112（Van der Pol），均严格大于观测器在代表轨迹上的误差；在加入 1% 测量噪声时，上界升至 0.685，仍能覆盖实际误差。

**⚠️ 局限性**

局限性包括：上界相对保守、仅验证低维系统、需要手工选择可验证区域、未扩展到受控系统或更复杂误差模型。

---

## 74. CRoCoDiL: Continuous and Robust Conditioned Diffusion for Language

**arXiv ID:** 2603.20210 | [PDF](https://arxiv.org/pdf/2603.20210v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 75. FinReflectKG -- HalluBench: GraphRAG Hallucination Benchmark for Financial Question Answering Systems

**arXiv ID:** 2603.20252 | [PDF](https://arxiv.org/pdf/2603.20252v1)

**作者:** Mahesh Kumar `[一作]` (Domyn Inc), Stefano Pasquali `[通讯]` (Domyn Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

创建FinBench-QA-Hallucination基准，对SEC 10-K文本进行KG增强的问答幻觉检测实验；

**💡 创新点**

提出保守的证据链接标注协议，构造天然KG噪声的真实评估环境，并系统比较六种检测方法在有/无triplet下的鲁棒性；

**🔧 技术方法**

使用LLM评估者（Qwen、GPT‑OSS、Lynx fine‑tuned judge）、DeBERTa‑NLI、span检测器（LettuceDetect）、嵌入相似度（Qwen、Stella）以及FinReflectKG的triplet提取与Qwen生成问答；

**📊 数据集**

以S&P 100公司2024年SEC 10‑K文件为来源，生成755条手工标注的Q&A样本，包含文本块、KG triplet和支持行；

**📈 对比分析**

在无triplet与有triplet两种条件下，对六种方法计算F1、MCC、ROC‑AUC和PR‑AUC；LLM judge和embedding方法在干净条件下F1最高达0.86‑0.87，但有triplet时MCC下降显著（50‑70%），embedding方法鲁棒性最佳，MCC仅下降≈10%；

**⚠️ 局限性**

样本规模有限，标注由内部AI工程师完成缺乏深厚金融专业知识；仅评估单跳提取式问题，未覆盖多跳推理、数值运算和跨文档合成；KG提取质量放宽导致噪声多，未对提取过程进行严格验证。

---

## 76. Coverage Games

**arXiv ID:** 2603.20398 | [PDF](https://arxiv.org/pdf/2603.20398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 77. Development and Validation of a Faculty Artificial Intelligence Literacy and Competency (FALCON-AI) Scale for Higher Education

**arXiv ID:** 2603.20220 | [PDF](https://arxiv.org/pdf/2603.20220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 78. Efficient Counterfactual Reasoning in ProbLog via Single World Intervention Programs

**arXiv ID:** 2603.20505 | [PDF](https://arxiv.org/pdf/2603.20505v1)

**作者:** Saimun Habib `[一作]` (University of Edinburgh), Fengxiang He `[通讯]` (University of Edinburgh)

**通讯引用:** 1774 | [OpenAlex ID](https://openalex.org/A5100635369)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了单世界干预程序（SWIP）技术，将ProbLog程序转换为满足干预约束的简化程序，从而实现高效的因果反事实推理。

**💡 创新点**

创新点包括：① 通过SWIG图切割实现单世界干预，避免Twin Network的跨世界假设与程序复制；② 证明在唯一支持模型下SWIP与CP-logic、LPADs产生一致的反事实分布；③ 展示SWIP在程序规模、树宽与推理时间上显著优于Twin Network。

**🔧 技术方法**

使用了ProbLog分布语义、SWIG（单世界干预图）与SWIFT算法（程序变换）、知识编译/SharpSAT进行推理、d-分离元解释器等技术。

**📊 数据集**

使用合成DAG实例（随机生成的树+稠密层+单目标节点），控制节点数n和树宽k，生成对应的ProbLog程序作为实验数据集。

**📈 对比分析**

实验与Twin Network基线比较：在编译和推理阶段，SWIP平均只需65%时间，树宽显著更小，整体推理时间比基线降低约35%；在不同证据/干预数量下保持性能优势。

**⚠️ 局限性**

局限性：需要唯一支持模型的ProbLog编码；在最坏情况下与Twin Network的成本相当；未在真实大规模复杂模型上验证；对非唯一支持模型的情况仍需进一步研究。

---

## 79. Less is More in Semantic Space: Intrinsic Decoupling via Clifford-M for Fundus Image Classification

**arXiv ID:** 2603.20806 | [PDF](https://arxiv.org/pdf/2603.20806v1)

**作者:** Yifeng Zheng `[一作]` (Xi'an Jiaotong University), Yifeng Zheng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 3563 | [OpenAlex ID](https://openalex.org/A5086106459)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种轻量级的基于克利福德几何相互作用的双分辨率网络 Clifford‑M，用于多标签眼底图像诊断。

**💡 创新点**

创新点在于用稀疏的克利福德几何乘积代替传统频率分解和 FFN，实现线性复杂度的跨尺度融合，且无需预训练。

**🔧 技术方法**

采用克利福德几何交互、双分辨率骨干、可选能量门控模块，整体构成无预训练的从零训练网络。

**📊 数据集**

在 ODIR‑5K 和 RFMiD 两个眼底图像多标签诊断数据集上进行评估。

**📈 对比分析**

与传统 CNN、ViT 以及轻量化预训练模型对比，Clifford‑M 仅 0.85 M 参数、3.33 GFLOPs，却在 ODIR‑5K 上取得 AUC‑ROC≈0.814、宏观 F1≈0.548，显著超越大多数基线模型。

**⚠️ 局限性**

局限包括训练稳定性受随机种子影响、未实现硬件最优实现、能量门控模块效果随输入分辨率变化、以及尚未验证在分割等其他任务上的泛化。

---

## 80. ChainGuards: Verification of Sensed Data using Permissioned Blockchain Technology

**arXiv ID:** 2603.20769 | [PDF](https://arxiv.org/pdf/2603.20769v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 81. PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching

**arXiv ID:** 2603.20818 | [PDF](https://arxiv.org/pdf/2603.20818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 82. SymCircuit: Bayesian Structure Inference for Tractable Probabilistic Circuits via Entropy-Regularized Reinforcement Learning

**arXiv ID:** 2603.20392 | [PDF](https://arxiv.org/pdf/2603.20392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 83. Review and Analysis of Scientific Paper Embellishments

**arXiv ID:** 2603.20306 | [PDF](https://arxiv.org/pdf/2603.20306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 84. ACRFence: Preventing Semantic Rollback Attacks in Agent Checkpoint-Restore

**arXiv ID:** 2603.20625 | [PDF](https://arxiv.org/pdf/2603.20625v1)

**作者:** Yusheng Zheng `[一作]` (UC Santa Cruz), Andi Quinn `[通讯]` (UC Santa Cruz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文发现并验证了LLM代理在使用检查点-恢复功能时会因重新生成不同请求而导致的语义回滚攻击，并提出一种框架无关的缓解方案，记录不可逆工具调用的效果并在恢复时强制执行重放或分支语义。

**💡 创新点**

创新点在于首次系统性识别“语义回滚攻击”，并引入轻量级LLM分析器在工具边界进行语义比较，实现重放与分支的自动区分，从而在不修改现有框架的前提下保障不可逆操作的一致性。

**🔧 技术方法**

采用检查点-恢复机制、eBPF系统级监控记录工具调用日志、LLM驱动的语义比较分析器以及代理层拦截器，对外部服务的工具调用进行拦截与判定；实验使用Claude Code CLI+Qwen3-32B模型。

**📊 数据集**

实验基于模拟的外部服务（银行转账、云资源创建、审批服务）构成测试套件，并未使用公开数据集。

**📈 对比分析**

通过对比启用检查点-恢复与不启用时的攻击成功率，发现恢复时100%成功、基线0%；缓解方案尚未完整实现，性能开销与准确性评估仍待进一步实验。

**⚠️ 局限性**

局限性包括：缓解方案未完成实现，LLM分析器可能出现误判或被对抗攻击；实验规模有限，仅在小规模代理与模拟服务上验证；未考虑多模型、多线程和跨框架的泛化情况。

---

## 85. MARLIN: Multi-Agent Reinforcement Learning for Incremental DAG Discovery

**arXiv ID:** 2603.20295 | [PDF](https://arxiv.org/pdf/2603.20295v1)

**作者:** Dong Li `[一作]` (Baylor University), Chen Zhao `[通讯]` (Baylor University)

**通讯引用:** 4529 | [OpenAlex ID](https://openalex.org/A5100767050)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种高效的多智能体强化学习框架，用于在线增量学习有向无环图（DAG），通过连续实数映射生成DAG、状态特定与状态不变智能体的解耦学习以及分解动作空间实现并行化；

**💡 创新点**

创新点在于将连续空间映射到DAG空间并放宽无环约束、设计双智能体解耦机制以捕获状态特定与不变因果关系，以及通过动作分解实现高效并行搜索；

**🔧 技术方法**

采用强化学习（actor‑critic、LSTM、GCN、BIC奖励、Adam优化）结合分解动作空间与基线策略的技术；

**📊 数据集**

使用合成数据（线性高斯、非高斯/非线性模型）以及三类真实工业时序数据（OnlineBoutique、SWaT、WADI）进行评估；

**📈 对比分析**

与PC、NOTEARS、GOLEM、DAG‑GNN、RL‑BIC、CORL、RCL‑OG等基线在TPR、F1、AUROC、FDR、SHD、SID及运行时间指标上比较，实验表明本文方法在精度和效率上均明显优于基线，尤其在大规模或实时场景中表现突出；

**⚠️ 局限性**

局限性包括对无环约束的放宽可能导致偶尔生成有环图、对BIC评分的依赖使得在噪声或非线性强的数据上性能下降、以及在极大节点数或高维噪声环境下仍需进一步优化与调参。

---

## 86. The Anatomy of an Edit: Mechanism-Guided Activation Steering for Knowledge Editing

**arXiv ID:** 2603.20795 | [PDF](https://arxiv.org/pdf/2603.20795v1)

**作者:** Yuan Cao `[一作]` (Technical University Of Munich), Hinrich Schütze `[通讯]` (Lmu Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用后期归因分析（NLKA）探究知识编辑在LLM中的实现机制，并基于此提出MEGA机制指导激活 steering，在GPT2‑XL与LLaMA2‑7B上实现无权重更新的知识编辑。

**💡 创新点**

核心创新是发现中晚期注意力层主导新目标的推广、原始事实的抑制，并利用后期归因作为工程信号，设计多层注意力‑残差激活 steering（MEGA），从而实现高效、跨架构的知识编辑。

**🔧 技术方法**

采用Neuron‑level Knowledge Attribution（NLKA）对编辑前后模型进行层级/组件级别贡献分析，使用PCA降维后的激活 transport map（与SAKE类似）在多层注意力‑残差路径上进行前向 hook 干预。

**📊 数据集**

实验数据集为 CounterFact（EasyEdit 子集）和 Popular（RippleEdits 子集），分别用于评估事实重写与多跳推理的编辑效果。

**📈 对比分析**

与 ROME、MEMIT、FT、IKE、SAKE 等四种主流编辑方法对比，利用 Acc、Gen、Spec、DI、DII、CI、CII、RS、SA 等指标评估，MEGA 在 GPT2‑XL 与 LLaMA2‑7B 的 CounterFact 与 Popular 上取得平均分 0.96–0.95、Acc 达到 0.99 的优秀性能，整体优于或接近最强方法。

**⚠️ 局限性**

实验仅覆盖单事实编辑，未验证更大或更新架构（如更大 LLaMA/ChatGPT）以及多编辑/大规模编辑场景，需进一步探究在更复杂任务中的泛化与扩展性。

---

## 87. GOLDMARK: Governed Outcome-Linked Diagnostic Model Assessment Reference Kit

**arXiv ID:** 2603.20848 | [PDF](https://arxiv.org/pdf/2603.20848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 88. RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

**arXiv ID:** 2603.20527 | [PDF](https://arxiv.org/pdf/2603.20527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 89. OmniPatch: A Universal Adversarial Patch for ViT-CNN Cross-Architecture Transfer in Semantic Segmentation

**arXiv ID:** 2603.20777 | [PDF](https://arxiv.org/pdf/2603.20777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 90. NDT: Non-Differential Transformer and Its Application to Sentiment Analysis

**arXiv ID:** 2603.20704 | [PDF](https://arxiv.org/pdf/2603.20704v1)

**作者:** Soudeep Ghoshal `[一作]` (Kalinga Institute of Industrial Technology), Rubén Ruiz-Torrubiano `[通讯]` (IMC University of Applied Sciences Krems)

**通讯引用:** 422 | [OpenAlex ID](https://openalex.org/A5030302246)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了非差分 Transformer（NDT），通过多组件加性注意力实现情感分析。

**💡 创新点**

创新点在于概念复用（ConPlex）框架和全正权重的加性组合，挑战传统的减法噪声抑制思路。

**🔧 技术方法**

使用 Transformer 结构、Multi‑Head Attention、可学习的 λ 权重、RMSNorm、SwiGLU 激活以及多约束的 λ 约束策略。

**📊 数据集**

在四个公开情感分析数据集上评测：IMDB、SST‑2、YELP 复评、Twitter 金融新闻。

**📈 对比分析**

与标准 Transformer 与 Differential Transformer 进行基准比较；NDT 在 IMDB 提升 1.36%、SST‑2 提升 0.32%、YELP 提升 0.32%、Twitter 提升 1.00%，参数增量仅 0.16‑0.23M，推理时间增幅可接受。

**⚠️ 局限性**

局限性包括在细粒度多类别任务（YELP）提升有限，对更大规模或更深层模型的可扩展性尚未验证，且仅在情感分析领域验证，需进一步推广。

---

## 91. The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting

**arXiv ID:** 2603.20714 | [PDF](https://arxiv.org/pdf/2603.20714v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 92. Evaluating LLM-generated code for domain-specific languages: molecular dynamics with LAMMPS

**arXiv ID:** 2603.20630 | [PDF](https://arxiv.org/pdf/2603.20630v1)

**作者:** Ethan Holbrook `[一作]` (Purdue University), Alejandro Strachan `[通讯]` (Purdue University)

**通讯引用:** 10093 | [OpenAlex ID](https://openalex.org/A5036381334)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了一个多阶段评估流程，用于评估大型语言模型（LLM）生成的LAMMPS输入脚本的有效性，并基于该流程对多种LLM的性能进行基准测试。

**💡 创新点**

首次提出针对科学DSL生成的系统化评估管线，包含规范化、AST解析、快速执行和物理正确性检查；同时构建了可扩展的LAMMPS解析器，为LLM产出脚本的错误定位提供了实用工具。

**🔧 技术方法**

使用LLM API（GPT‑4o、GPT‑4.1、GPT‑o3、GPT‑5、Claude 4 Opus）生成脚本；开发lammps‑ast包进行规范化和AST解析；采用Lark构建静态解析器；通过10步短运行和pair_style zero替换进行快速执行；制定针对不同任务的物理检查清单。

**📊 数据集**

基于三条手工设计的自然语言提示（Al 单晶平衡、Ni 熔化、Nb 高速撞击）与五个模型的10-shot生成，产生150个LAMMPS脚本；未使用公开数据集，全部为实验性自制提示与脚本。

**📈 对比分析**

比较指标包括：解析器通过率、执行成功率（含PSZ修正）以及一次性正确率；Claude 4 Opus最高（97%通过率，67%执行成功，30%一次性正确），GPT‑4.1、GPT‑4o相对稳定，GPT‑o3和GPT‑5性能较低；随着任务复杂度提升，整体性能显著下降（Prompt 1≈84%执行成功，Prompt 2≈68%，Prompt 3仅8%）。

**⚠️ 局限性**

LLM在生成科学DSL时普遍缺乏语义正确性：常见错误包括pair_style选择错误、单位/几何转换失误、命令参数不合法；模型对多步骤物理推理、几何关系的捕捉不足；缺乏成熟的预执行验证与自动化校正机制；高成本的MD执行限制了迭代改进，导致仅在简单任务中能提供可用起点。

---

## 93. Distributed Gradient Clustering: Convergence and the Effect of Initialization

**arXiv ID:** 2603.20507 | [PDF](https://arxiv.org/pdf/2603.20507v1)

**作者:** Aleksandar Armacki `[一作]` (Carnegie Mellon University), Soummya Kar `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12050 | [OpenAlex ID](https://openalex.org/A5077268766)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究分布式梯度聚类算法在不同中心初始化下的性能，并提出改进的分布式 K‑means++ 初始化方案

**💡 创新点**

提出 DKM+C 初始化方法，利用局部通信与 K‑means++ 相结合，在多用户网络中显著提升聚类精度并降低对初始化的敏感性

**🔧 技术方法**

使用分布式梯度聚类框架 (DGC-ℱ_ρ)、K‑means++、K‑means 以及协同/创新更新机制

**📊 数据集**

在 Iris 数据集上进行实验，数据在 10 个用户之间按同质/异质方式分布

**📈 对比分析**

与集中式梯度聚类 (CGC) 在相同数据集上比较，结果显示 DGC-ℱ_ρ 对随机初始化更鲁棒，且 DKM+C 能在更多通信轮次下进一步提升准确率，最终在同质/异质场景下均优于随机初始化

**⚠️ 局限性**

实验仅基于小型公开数据集，缺乏对更大规模、不同分布和动态惩罚参数 ρ 的理论与实验验证

---

## 94. Thinking in Different Spaces: Domain-Specific Latent Geometry Survives Cross-Architecture Translation

**arXiv ID:** 2603.20406 | [PDF](https://arxiv.org/pdf/2603.20406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 95. Error-resilient Distributed Local Verification

**arXiv ID:** 2603.20831 | [PDF](https://arxiv.org/pdf/2603.20831v1)

**作者:** Paweł Garncarek `[一作]` (University of Wrocław), Subhajit Pramanick `[通讯]` (University of Wrocław)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在分布式网络中对图属性进行错误容忍的本地验证框架，并给出了在有误标签的情况下，如何使用常数规模标签验证环检测、无环性和二分图性等基础图属性的具体方案。

**💡 创新点**

创新点在于：①用仅 2 个标签（1 位）并通过 3 跳视距实现环检测的方向编码（此前 3 标签是已知最优），②设计了一个通用框架（ESILIENT RROR IXING）将基于无误标签的验证算法扩展到容忍 i 个错误、视距提升至 d+2i；③给出了针对不同属性的严格下界，揭示错误数与视距之间的必然关联。

**🔧 技术方法**

采用了证明标签方案（proof‑labeling scheme）与本地可检查标签（locally checkable labeling）理论，利用基于循环序列 S∞ 的子串分割来实现方向编码；框架使用“想象标签”概念并通过子图局部一致性推理实现错误纠正；同时利用鸽巢原理和路径/树结构分析证明下界。

**📊 数据集**

该工作纯理论研究，无使用具体数据集；所有结论基于图论证明与理论分析。

**📈 对比分析**

比较方法主要是理论上与已知结果对照：在无误标签情形下，3 标签+1 跳视距已是最优；2 标签+3 跳视距是首次突破；在有误标签时，框架实现了 i 个错误对应视距 d+2i 的容忍度；下界证明展示了此比例的必要性，说明在更小视距下无法实现错误容忍。

**⚠️ 局限性**

局限性包括：①框架仅适用于能用基于无误标签的算法构建的属性，未覆盖所有本地可检查属性；②视距需要提升至 d+2i，导致实际通信成本可能增大；③对 CONGEST 模型的适应性仅在极简情况（如二分图）给出猜想，尚缺乏完整证明；④在极端错误分布（非局部）下的鲁棒性未作完整分析。

---

## 96. Understanding Pruning Regimes in Vision-Language Models Through Domain-Aware Layer Selection

**arXiv ID:** 2603.20275 | [PDF](https://arxiv.org/pdf/2603.20275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. Visual Exclusivity Attacks: Automatic Multimodal Red Teaming via Agentic Planning

**arXiv ID:** 2603.20198 | [PDF](https://arxiv.org/pdf/2603.20198v1)

**作者:** Yunbei Zhang `[一作]` (Tulane University), Chandan K. Reddy `[通讯]` (Amazon)

**通讯引用:** 7155 | [OpenAlex ID](https://openalex.org/A5001022750)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实现了视觉排他性（Visual Exclusivity）攻击框架，研究通过多轮对话利用视觉信息实现对大型多模态模型的规避；

**💡 创新点**

创新点包括：①定义视觉排他性威胁，将视觉信息作为攻击基础；②构建了首个针对此威胁的 VE‑Safety 数据集；③提出 MM‑Plan 全局规划模型，通过 JSON 计划模板和 Group Relative Policy Optimization (GRPO) 自动生成多轮攻击策略；

**🔧 技术方法**

技术实现包括：全局 Agentic Planning、JSON 计划格式、图像裁剪/编辑等视觉操作、GRPO 强化学习、OpenAI/Claude 等判别器评估；

**📊 数据集**

使用的数据集为 440 条人类标注的 VE‑Safety 实例，包含 15 个安全类别；在基准评估中也对 HarmBench 进行了实验；

**📈 对比分析**

实验将 MM‑Plan 与多种单轮、双轮基线（Direct Request、FigStep、SI‑Attack、Crescendo、SSA 等）对比，MM‑Plan 在 Claude 4.5 Sonnet 上 46.3% 的攻击成功率、在 GPT‑5 上 13.8%，均比基线提升 2–5 倍，且在多模态模型上表现出色；

**⚠️ 局限性**

局限性包括：依赖开放权重的规划器，无法使用闭源模型进行规划；需要多轮交互，攻击耗时；视觉操作能力有限，无法处理更复杂的图像编辑；在某些目标模型上的迁移性能仍受限。

---

## 98. COmPOSER: Circuit Optimization of mm-wave/RF circuits with Performance-Oriented Synthesis for Efficient Realizations

**arXiv ID:** 2603.20486 | [PDF](https://arxiv.org/pdf/2603.20486v1)

**作者:** Subhadip Ghosh `[一作]` (University of Minnesota), Sachin S. Sapatnekar `[通讯]` (University of Minnesota)

**通讯引用:** 14900 | [OpenAlex ID](https://openalex.org/A5068714995)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 COmPOSER，一个从规格到布局的端到端 RF/mm‑wave 设计自动化框架，自动完成元件尺寸、匹配网络、布局生成、布线与 PDN 合成，并通过后仿真实现性能收敛。

**💡 创新点**

创新点在于：① 将物理模型与 ML 逆向 EM 拟合结合，既保留电路级物理可解释性，又实现对电感、共面波导、T‑线等被动元件的快速、精确逆向设计；② 在统一的流程中同时优化器件与布局，消除后仿真偏差；③ 采用混合优化器（物理方程 + ML EM）实现 100–300× 的设计周期缩短。

**🔧 技术方法**

技术手段包括：混合物理/ML 逆向优化（随机森林 + k‑NN），闭式电路方程求解，混合整数线性规划 (MILP) 置放，A* 路由，EM 级全波仿真与 SPICE 共同验证，PDN 与 decap 的 DRC‑干净自动合成。

**📊 数据集**

数据集主要是：约 80k 条电感 EM 仿真样本（t, w, s, r → L, Q, f_peak, f_SRF），以及数千条 T‑线与 CPW 的 EM 数据，用于训练逆向 k‑NN 及前向随机森林模型。

**📈 对比分析**

与人工手工设计进行对比：在 65 nm CMOS 下的多频 LNA、PA 设计，COmPOSER 在保持相同或更佳的 G/NF/BW/PAE 等性能指标的同时，设计周期从 70 h 降至 < 25 min（后仿真占 80% 以上），实现 100–300× 的生产力提升；并在 60 GHz 频点的后仿真中验证了与专家手工版相当的性能。

**⚠️ 局限性**

局限性：① 依赖预先生成的大规模 EM 仿真数据，初始数据生成仍耗时；② 目前仅在 65 nm CMOS 及其已预设的 PDK 上验证，跨工艺迁移需要重新拟合模型；③ 对极低噪声、极宽带等特殊拓扑的适用性尚未全面评估；④ 仍需在大规模多块级设计或更复杂系统级集成中验证其可扩展性。

---

## 99. Reinforcement Learning from Multi-Source Imperfect Preferences: Best-of-Both-Regimes Regret

**arXiv ID:** 2603.20453 | [PDF](https://arxiv.org/pdf/2603.20453v1)

**作者:** Ming Shi `[一作]` (University at Buffalo), Ananthram Swami `[通讯]` (Army Research Laboratory)

**通讯引用:** 18584 | [OpenAlex ID](https://openalex.org/A5103191815)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种针对多源、存在偏差的偏好反馈的强化学习算法，并给出了最佳-两种模式的调和误差下的上界和下界。

**💡 创新点**

创新点包括：
• 通过累计偏差预算 ω 对多源偏好误差建模；
• 设计了自适应加权比较学习、价值导向转移估计、策略层面乐观规划与子重要性抽样相结合的统一算法；
• 给出了上界 O(√(K/M)+ω) 与匹配下界 Ω(max{√(K/M),ω})，揭示低偏差下可获得 M 的 1/√M 统计加速，且高偏差时误差仅线性依赖于 ω。

**🔧 技术方法**

主要技术包括：
• 自适应权重加权的比较函数学习（类似 OFUL 的自归一化方法）；
• 值导向（value‑targeted）转移估计来缓解偏好误差引起的分布漂移；
• 基于参考策略的策略层面乐观规划；
• 子重要性抽样（sub‑importance sampling）过滤历史以控制敏感度和提升分析可行性。

**📊 数据集**

本工作为理论研究，未使用公开数据集；实验以模拟 MDP 为例验证算法与基线的相对性能。

**📈 对比分析**

与不考虑偏差的传统 OFUL‑style 基线相比，算法在 ω ≪ √(K/M) 时表现出更优的 O(√(K/M)) 速率；在 ω ≫ √(K/M) 时保持 O(ω) 的稳健性能，避免了基线的 Ω(ω√K) 退化；下界证明了该上界在阶数上是最优的。

**⚠️ 局限性**

局限性：
• 需要先验或假设已知偏差预算 ω 的上界，若 ω 估计不准可能导致保证失效；
• 仅考虑累计偏差模型，未覆盖源间协同学习或多目标情形；
• 对未知转移时的转移学习项仍然有较大常数，进一步优化空间存在；
• 目前仅给出理论上限，实际性能需在更复杂真实环境中验证。

---

## 100. Swim2Real: VLM-Guided System Identification for Sim-to-Real Transfer

**arXiv ID:** 2603.20827 | [PDF](https://arxiv.org/pdf/2603.20827v1)

**作者:** Kevin Qiu `[一作]` (University of Warsaw), Josie Hughes `[通讯]` (EPFL)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过视觉语言模型对真实鱼视频与仿真视频进行对比，自动校准全16维软体鱼仿真器参数，并在校准后的仿真器中训练强化学习策略，最终将策略在物理鱼上实现无缝部署。

**💡 创新点**

创新点在于：①首次将Vision‑Language Model（Gemini）作为物理差异诊断器，直接从视频中推断参数更新方向；②配合回溯线搜索纠正步长误差，显著提高接受率；③实现完全无手工阶段的全参数协同校准，从视频到硬件的一站式闭环。

**🔧 技术方法**

使用技术包括：MuJoCo软体鱼仿真、Gemini VLM诊断、回溯线搜索、SAC强化学习、CSRT视频跟踪、marker 轨迹误差评估，以及 16 维参数空间的系统辨识。

**📊 数据集**

数据集来源为实验室采集的真实鱼运动视频（2160×3840 @ 60 fps），包含 8 个不同工作频率（0.5–2.25 Hz）的运动轨迹，标记为 11 个身体点，用于评估标记误差和速度误差。

**📈 对比分析**

与 BayesOpt、CMA‑ES、随机搜索在 40 次仿真评估、5 种种子下进行对比。该方法在标记误差（51.3 ± 1.2 mm）、前进速度 MAE（7.4 mm/s）以及 RL 距离（7.6 m）等指标上均优于对手，接受率提升至 42%，AUC 下降至 85.9 mm，表现出更高的可靠性和样本效率。

**⚠️ 局限性**

局限性包括：尾部离散化导致约 50 mm 的误差下限；依赖付费 Gemini API，缺乏开源可复现性；回溯线搜索无法纠正根本错误；策略部署为开环，易出现漂移和转向偏差；实验仅覆盖单一软体鱼平台，未验证在更高维度或不同机器人上的普适性。

---

## 101. Lessons and Open Questions from a Unified Study of Camera-Trap Species Recognition Over Time

**arXiv ID:** 2603.20509 | [PDF](https://arxiv.org/pdf/2603.20509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 102. gUFO: A Gentle Foundational Ontology for Semantic Web Knowledge Graphs

**arXiv ID:** 2603.20948 | [PDF](https://arxiv.org/pdf/2603.20948v1)

**作者:** João Paulo A. Almeida `[一作]` (Federal University of Espírito Santo), Claudenir M. Fonseca `[通讯]` (University of Twente)

**通讯引用:** 608 | [OpenAlex ID](https://openalex.org/A5065239769)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了轻量级的gUFO实现，详细阐述其类别、关系、约束，并与原始UFO及其他OWL基础本体实现进行对比。

**💡 创新点**

提供了独特的类型层次、针对内在与关系属性的重构模式，以及对情境和高阶类型的支持，使gUFO在OWL 2 DL下更适合语义网知识图谱。

**🔧 技术方法**

使用OWL 2 DL语义网技术、OntoClean原则、重构模式（reification）和本体工程方法。

**📊 数据集**

未使用外部数据集，主要以本体定义和案例演示为依据。

**📈 对比分析**

通过与BFO、DOLCE等现有OWL实现的语义兼容性、推理支持和社区使用情况进行比较，显示gUFO在兼容性与轻量级设计上具有优势，但缺乏大规模实验性能评估。

**⚠️ 局限性**

局限在于与完整UFO相比功能略减，缺乏针对大规模知识图谱的性能评估，且尚未完全标准化。

---

## 103. Benchmarking Efficient & Effective Camera Pose Estimation Strategies for Novel View Synthesis

**arXiv ID:** 2603.20428 | [PDF](https://arxiv.org/pdf/2603.20428v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 104. Communication Lower Bounds and Algorithms for Sketching with Random Dense Matrices

**arXiv ID:** 2603.20966 | [PDF](https://arxiv.org/pdf/2603.20966v1)

**作者:** Hussam Al Daas `[一作]` (Rutherford Appleton Laboratory), Kathryn Rouse `[通讯]` (Inmar Intelligence)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在分布式内存环境下研究稠密随机矩阵的 sketching 与 Nyström 近似的通信下界，并提出实现可达到下界的并行算法。

**💡 创新点**

通过推导新的几何不等式和受限优化，获得通信下界并证明其可达；设计多维处理器网格与 All‑Gather/Reduce‑Scatter 组合，实现了通信最优（或近似最优）算法。

**🔧 技术方法**

利用受限优化与 KKT 条件、几何不等式、MPI 集合操作（All‑Gather、Reduce‑Scatter、All‑To‑All）以及 CUDA‑aware MPI，配合随机矩阵的局部生成（Philox 等）实现高效并行。

**📊 数据集**

使用 10^6×10^6 的遗传相异性矩阵、CIFAR‑10 50k×3072 数据生成的线性核与 RBF 核（n=50000）作为实验数据集。

**📈 对比分析**

通过与 Python/C++实现、CPU 与 GPU、Redist 与 No‑Redist 两种 1D 算法的对比实验评估性能，GPU 在单机上加速约 10 倍但通信成本更高，CPU 实现扩展性更好；实验结果与理论下界高度一致。

**⚠️ 局限性**

仅针对稠密随机矩阵，未涵盖结构化或稀疏 sketch；对称矩阵的通信优化尚未实现；实验假设每一步乘法负载均衡，实际负载不平衡时效果未知。

---

## 105. Adversarial Attacks on Locally Private Graph Neural Networks

**arXiv ID:** 2603.20746 | [PDF](https://arxiv.org/pdf/2603.20746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 106. Telework during the Pandemic: Patterns, Challenges, and Opportunities for People with Disabilities

**arXiv ID:** 2603.20227 | [PDF](https://arxiv.org/pdf/2603.20227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 107. High-Quality and Efficient Turbulence Mitigation with Events

**arXiv ID:** 2603.20708 | [PDF](https://arxiv.org/pdf/2603.20708v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 108. ProMAS: Proactive Error Forecasting for Multi-Agent Systems Using Markov Transition Dynamics

**arXiv ID:** 2603.20260 | [PDF](https://arxiv.org/pdf/2603.20260v1)

**作者:** Xinkui Zhao `[一作]` (Zhejiang University), Chang Liu `[通讯]` (Zhejiang University)

**通讯引用:** 29154 | [OpenAlex ID](https://openalex.org/A5100353357)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型驱动的多智能体系统（MAS），提出ProMAS框架，实现“谁和何时”错误定位的主动预测与实时干预。

**💡 创新点**

创新点在于：①将逻辑推理视为语义迁移，使用Causal Delta捕捉推理“速度”；②构建Vector Markov Space对离散推理轨迹建模；③通过跳跃检测（risk acceleration）在不依赖完整轨迹的情况下实时识别首次逻辑失误，显著降低误报率。

**🔧 技术方法**

核心技术包括：对比学习+triplet loss训练Causal Delta；Mini‑Batch K‑Means聚类得到离散原型；贝叶斯平滑的Markov转移矩阵；MLP预测下一个原型分布；动态跳跃检测阈值机制。

**📊 数据集**

使用Who&When基准数据集，包含算法生成与人工编写的多轮推理对话，提供错误发生时间与责任智能体的标注。

**📈 对比分析**

与离线LLM heuristic、AgenTracer、Famas以及在线MASC等方法对比，ProMAS在步级准确率22.97%（比MASC高1.35个百分点），Agent级准确率40.54%；仅需处理约27%完整日志，信息消耗比全量方法降低73%，在实时约束下实现了与离线方法相近的定位精度。

**⚠️ 局限性**

局限性：与最优离线监督方法相比精度仍有差距；在人为设计的细微错误上表现下降；模型对训练分布敏感，可能误判新颖推理；跳跃检测阈值需要在不同场景中调优；仅聚焦于逻辑失误，未覆盖所有错误类型。

---

## 109. Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark

**arXiv ID:** 2603.20721 | [PDF](https://arxiv.org/pdf/2603.20721v1)

**作者:** Yifei Deng `[一作]` (Anhui University), Jin Tang `[通讯]` (Anhui University)

**通讯引用:** 12111 | [OpenAlex ID](https://openalex.org/A5030720334)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了跨模态模糊对齐网络（CFAN），用于无人机拍摄的空中人物检索。

**💡 创新点**

创新点包括：①基于模糊逻辑的Token对齐（FTA），动态衡量Token可靠性并抑制噪声；②上下文感知动态对齐（CDA），利用地面视图作为桥接并自适应权衡直接与桥接对齐；③通过链式思维（CoT）生成高质量训练描述，构建大规模空中-地面混合检索基准AERI-PEDES。

**🔧 技术方法**

技术包括：CLIP视觉与文本编码、CrossFormer交叉注意力、模糊隶属函数与逻辑与、基于相似度分布的对齐损失、上下文感知Sigmoid权重、链式思维多阶段文本生成。

**📊 数据集**

使用了新建的AERI-PEDES数据集（112,672 空中图、26,351 地面图、112,672 训练文本、6,141 测试文本）和TBAPR基准进行评测。

**📈 对比分析**

与多种SOTA方法（如IRRA、APTM、RDE、NAM、VFE等）对比，CFAN在AERI-PEDES的Rank‑1、Rank‑5、Rank‑10、mAP和RSum均实现最高或第二高分；在TBAPR上亦持续领先，显著提升检索准确率和鲁棒性。

**⚠️ 局限性**

局限性：依赖地面视图桥接时需额外采集；模糊对齐对极端缺失或极端噪声图像的鲁棒性仍有限；模型复杂度较高，训练成本和推理速度相对较慢。

---

## 110. Delightful Distributed Policy Gradient

**arXiv ID:** 2603.20521 | [PDF](https://arxiv.org/pdf/2603.20521v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 111. NoveltyAgent: Autonomous Novelty Reporting Agent with Point-wise Novelty Analysis and Self-Validation

**arXiv ID:** 2603.20884 | [PDF](https://arxiv.org/pdf/2603.20884v1)

**作者:** Jiajun Hou `[一作]`, Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 61037 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多代理框架 NoveltyAgent，用于自动化生成论文新颖性报告。

**💡 创新点**

通过点级报告生成、数据库构建和自我验证机制，显著提升报告真实性与深度。

**🔧 技术方法**

采用多代理协作、RAG检索、BM25+向量检索、Qwen3-Reranker、Self‑Validation 等技术。

**📊 数据集**

基于 ICLR 2025 OpenReview 提交论文构建的 50 篇评估集，包含 200 篇参考文献。

**📈 对比分析**

与 GPT‑5、Kimi‑2、DeepResearch 等基线对比，NoveltyAgent 在 5 维评估指标上平均提升约 10%（最高得分 8.47/10）。

**⚠️ 局限性**

局限包括仅处理文本未覆盖多模态；数据集规模有限；引用网络构建可能忽略小众文献。

---

## 112. negMIX: Negative Mixup for OOD Generalization in Open-Set Node Classification

**arXiv ID:** 2603.20798 | [PDF](https://arxiv.org/pdf/2603.20798v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 113. Software Entropy: A Statistical Mechanics Framework for Software Testing

**arXiv ID:** 2603.20528 | [PDF](https://arxiv.org/pdf/2603.20528v1)

**作者:** Jerónimo Fotinós `[一作]` (National University of Córdoba), Juan B. Cabral `[通讯]` (Comisión Nacional de Actividades Espaciales)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在统计力学框架下给出软件熵的形式化定义，并通过变异测试估计测试套件对程序空间的约束。

**💡 创新点**

首次将软件熵与统计力学熵联系起来，提出信息加权的约束权重和宏状态紧致度指标，并用局部变异图估算熵上界。

**🔧 技术方法**

使用变异测试、熵计量、信息熵、Python工具Yagua以及对Astroalign项目的实验。

**📊 数据集**

以Python库Astroalign为案例，生成463个变异体，并测得覆盖率与熵曲线。

**📈 对比分析**

通过实验展示熵随测试增多而下降，信息权重分布呈双峰，说明传统覆盖率无法反映约束力；实验耗时可控，性能满足实用需求。

**⚠️ 局限性**

局限在于只能近似局部微状态空间、无法枚举全局程序空间、变异测试成本高、结果受随机性和超时影响、对历史演进缺乏完整追踪。

---

## 114. An experimental study of KV cache reuse strategies in chunk-level caching systems

**arXiv ID:** 2603.20218 | [PDF](https://arxiv.org/pdf/2603.20218v1)

**作者:** Samuel Cestola `[一作]` (Huawei), Diego Didona `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估并比较了多种块级缓存（Chunk‑Level Caching）下的 KV 缓存重用设计，并提出了一种融合重算与注意力重塑技术的新方案PSR，显著提升了大语言模型在检索增强生成任务中的准确率。

**💡 创新点**

创新点在于发现现有方案互补，结合跨块注意力恢复、注意力饱和问题处理和温度缩放三大机制，形成 PSR，首次实现单独方案无法达到的 5% 以上准确率提升。

**🔧 技术方法**

使用了 Cacheblend、EPIC、Link0、Cacheclip、Droidspeak 等重算技术；以及 APE、TurboRAG、BlockAttention、KVLink 等注意力重塑技术，并在此基础上加入 Prefix‑Scale‑Recompute（PSR）组合。

**📊 数据集**

实验数据集包括 Llama3.1‑8B‑Instruct、Qwen3‑8B、Mistral‑7B‑Instruct‑v0.3 等模型；问答数据集则采用 2WikiMQA、Musique、RULER 等多跳推理任务。

**📈 对比分析**

对比方法包括全预填（Full Prefill）、朴素重用（Naïve Reuse）、各单独方案以及 PSR；评估指标为修正后的 F1 分数，PSR 在保持低重算开销的前提下相较单独方案平均提升约 5% 的准确率。

**⚠️ 局限性**

主要限制是跨块注意力缺失导致的精度瓶颈；动态重算因缺少跨块注意力信息而效果不佳；以及高成本的微调方案不易复现，且在某些模型/数据集上仍无法完全逼近全预填性能。

---

## 115. AEGIS: From Clues to Verdicts -- Graph-Guided Deep Vulnerability Reasoning via Dialectics and Meta-Auditing

**arXiv ID:** 2603.20637 | [PDF](https://arxiv.org/pdf/2603.20637v1)

**作者:** Sen Fang `[一作]` (NC State University), Bowen Xu `[通讯]` (NC State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多代理的漏洞检测框架，将检测过程拆分为从“线索”到“判决”的四阶段管线，先定位可疑代码，再通过代码属性图(CPG)动态重构跨文件数据流，随后以辩论式推理和元审计两阶段进行可信判决。

**💡 创新点**

核心创新是：①把漏洞定位与推理分离，先构建闭合的事实证据基；②采用线索锚定的需求驱动CPG切片，动态获取与漏洞相关的完整依赖链；③引入单一验证者的辩论式推理和独立审计者，消除上下文幻觉。

**🔧 技术方法**

技术包括：代码属性图(CPG)构建与切片；基于 DeepSeek‑V3.1 的大型语言模型（LLM）多代理协作；辩论式推理（Red/Blue）与元审计（审核缺陷、投票）。

**📊 数据集**

使用 PrimeVul 基准，包含 6,968 个漏洞函数和 228,800 个安全函数，435 对漏洞–补丁测试对。

**📈 对比分析**

在 PrimeVul 上与 Fine‑tuned、检索增强、Agent‑based 等三类基线对比，得到 122/435 的 Pair‑wise Correct Predictions（首次突破 100），FPR 下降 54.40%，平均成本仅 0.09 美元/样本，且在无任务专门训练的情况下实现 SOTA。

**⚠️ 局限性**

局限性：依赖 DeepSeek‑V3.1，模型对其他 LLM 的适用性待验证；对极低置信度线索的审计效果不佳；对非 C/C++ 语言的可迁移性待进一步研究。

---

## 116. JCAS-MARL: Joint Communication and Sensing UAV Networks via Resource-Constrained Multi-Agent Reinforcement Learning

**arXiv ID:** 2603.20265 | [PDF](https://arxiv.org/pdf/2603.20265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 117. LLM-Driven Heuristic Synthesis for Industrial Process Control: Lessons from Hot Steel Rolling

**arXiv ID:** 2603.20537 | [PDF](https://arxiv.org/pdf/2603.20537v1)

**作者:** Nima H. Siboni `[一作]` (Juna.ai), Emad Scharifi `[通讯]` (RWTH Aachen)

**通讯引用:** 335 | [OpenAlex ID](https://openalex.org/A5068179673)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过大语言模型驱动的四阶段搜索循环，生成可审计、可解释的钢铁热轧控制器。

**💡 创新点**

创新点在于结合了基于LLM的策略构思与代码实现、丰富的行为反馈、Luby式通用重启预算分配以及五层自动化安全审计，实现在物理仿真环境中的可部署控制方案。

**🔧 技术方法**

技术实现包括 Gemini 2.5 Pro LLM、Python代码生成、PyRoll 物理模拟器、Z3 SMT 形式化验证与属性测试、以及自适应的对话记忆与重启策略。

**📊 数据集**

使用的数据集为 8 个搜索反馈场景（厚度、晶粒、设备、温度二值组合）以及 81 个完整组合的保留测试集。

**📈 对比分析**

通过与手工基线和多轮独立搜索（52 轮）比较，Luby 160 步重启方案在预算内达到接近最优混合分配的性能，单个最佳在保留集上获得 60.9 分，组合组最优 65.1，整体比基线提升约 30%。

**⚠️ 局限性**

主要限制包括模式崩溃后难以跳出、对话记忆消失导致的重复探索，以及对极端探索策略的鲁棒性不足。

---

## 118. Bounded Coupled AI Learning Dynamics in Tri-Hierarchical Drone Swarms

**arXiv ID:** 2603.20333 | [PDF](https://arxiv.org/pdf/2603.20333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 119. Fast-Slow Thinking RM: Efficient Integration of Scalar and Generative Reward Models

**arXiv ID:** 2603.20212 | [PDF](https://arxiv.org/pdf/2603.20212v1)

**作者:** Jiayun Wu `[一作]` (Fudan University), Tun Lu `[通讯]` (Fudan University)

**通讯引用:** 2173 | [OpenAlex ID](https://openalex.org/A5004237040)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Fast‑Slow 思考奖励模型（F/S‑RM），通过融合一阶标量预测与链式思考来实现奖励评估。

**💡 创新点**

创新点在于：1）将快思考（一次性标量输出）与慢思考（CoT 生成）整合在同一模型；2）设计双置信度激活机制，根据直觉置信度与词分布置信度决定是否触发慢思考；3）采用两阶段训练（SFT+RL）提升慢思考稳定性。

**🔧 技术方法**

采用 Bradley–Terry 监督框架、GRPO 强化学习、双置信度激活策略，配合 Qwen3-4B/8B 预训练模型。

**📊 数据集**

使用 20K 人工/LLM 对话偏好样本进行训练；评测基准包括 RewardBench、RM‑Bench 与 JudgeBench。

**📈 对比分析**

与传统 Scalar RM 和 Generative RM 进行对比，F/S‑RM 在平均分数上提升约1.2%，同时减少约20.8% 的 token 计算量；在 RewardBench、RM‑Bench 和 JudgeBench 上均实现或接近最优表现。

**⚠️ 局限性**

局限性：实验仅覆盖 4B/8B 规模模型，未验证更大模型；评测依赖 LLM 生成的数据集，可能存在模型偏差；缺乏大规模人工评估，需进一步验证真实人类偏好。

---

## 120. A Training-Free Regeneration Paradigm: Contrastive Reflection Memory Guided Self-Verification and Self-Improvement

**arXiv ID:** 2603.20441 | [PDF](https://arxiv.org/pdf/2603.20441v1)

**作者:** Yuran Li `[一作]` (McGill University), Benoit Boulet `[通讯]` (McGill University)

**通讯引用:** 3226 | [OpenAlex ID](https://openalex.org/A5081097070)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练‑free 的回归式自我改进框架，利用离线构建的对比反射记忆（RM）在推理时检索并用来指导答案生成与自我验证，若检测到错误则从零开始重新生成答案。

**💡 创新点**

创新点包括：①离线构造的对比RM将教师模型的纠错洞察转化为可检索的对比示例；②在推理时通过一次前向生成完成自我验证与基于RM的一步重生成，避免了迭代校正与多样本采样的高成本；③在不需要标签、外部工具或微调的情况下实现高精度验证与纠错。

**🔧 技术方法**

技术：对比反射记忆的构造（错误收集、教师反思、归纳原则、ICL清洗）；检索（Contriever + MPNet重排序）；RM‑Primed 提示；RM‑Guided 验证与重生成；基于熵的不确定性阈值。

**📊 数据集**

数据集：9 个基准——算法推理（GSM8K、GSM‑Hard、MATH Level3）、常识推理（StrategyQA、Bamboogle）、符号推理（Coin Flip、Letter）、领域专用（LegalBench、News Headline）。

**📈 对比分析**

与 Self‑Refine、Reflexion、ProCo、Best‑of‑N、ST‑CoT 等方法比较，RM‑Primed 在大多数任务上提升约 4–7% 准确率，RM‑Regen 在有 oracle 验证时平均提升 6–10%；在无 oracle 验证时仍保持最高平均准确率（约 68%），同时拥有更佳的速度‑准确率平衡。

**⚠️ 局限性**

局限：依赖教师模型质量；验证器仍不完美，未检测到的错误限制最终性能；RM 构建成本高但可离线完成；在极端噪声验证或对特定任务的泛化仍需进一步验证。

---

## 121. Voice Privacy from an Attribute-based Perspective

**arXiv ID:** 2603.20301 | [PDF](https://arxiv.org/pdf/2603.20301v1)

**作者:** Mehtab Ur Rahman `[一作]` (Centre for Language Studies), Cristian Tejedor García `[通讯]` (Centre for Language Studies)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了声学隐私的属性角度，使用声纹属性推断和单句重识别攻击评估隐私风险。

**💡 创新点**

首次将基于属性的唯一性与重识别误差作为评价指标，证明属性推断误差并不一定提升隐私，并公开了四属性标注集。

**🔧 技术方法**

采用ECAPA‑TDNN声纹嵌入+轻量级MLP分类器进行属性推断，并通过k‑匿名性分析和错误率攻击实验评估安全性。

**📊 数据集**

使用VoxCeleb2数据集，扩展标注性别、年龄、口音、职业四属性，并在多语句和单语句评估集合上实验。

**📈 对比分析**

与基线随机与真值属性对比，发现k<5的唯一性比例下降，重识别错误率介于0.47到0.78之间，仍未达到理想隐私水平。

**⚠️ 局限性**

属性推断误差不一致可能导致个别说话者更唯一；实验仅使用未改造的分类器，未考虑攻击者可能获得解匿名化系统；数据规模相对有限。

---

## 122. GEM: A Native Graph-based Index for Multi-Vector Retrieval

**arXiv ID:** 2603.20336 | [PDF](https://arxiv.org/pdf/2603.20336v1)

**作者:** Yao Tian `[一作]`, Xiaofang Zhou `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于图的多向量检索索引 GEM，利用训练集生成语义捷径、量化 EMD、集群剪枝、全局图桥接、多入口点等技术实现高效检索。

**💡 创新点**

创新点在于：1）直接在多向量空间构建可导航图；2）将 Chamfer 与 EMD 分离，用 EMD 作为边权进行图构造；3）引入基于语义短路的训练集捷径；4）采用量化 EMD 以降低距离计算成本；5）利用 TF‑IDF 指引的集群剪枝与跨簇桥接提升连通性。

**🔧 技术方法**

技术手段包括：图索引（APG）、量化 EMD（中心化聚类）、两阶段聚类、TF‑IDF 指引的集群分配、全局图桥接、量化编码、基于 LLM 的多向量嵌入、跨簇桥接与多入口点初始化。

**📊 数据集**

使用了四个多向量检索基准：MSMARCO、LoTTE、OK‑VQA、EVQA，并在这些数据集上训练和测试。

**📈 对比分析**

与 LSH/DESSERT、IVF/PLAID、IGP、MVG、MUVERA 等传统与现代方法对比，GEM 在 MRR、Recall、Success 等指标上均优于对手，且在相同或更低延迟下达到更高检索精度，体现出最佳的准确性‑效率折中。

**⚠️ 局限性**

局限性：1）对训练集质量与覆盖率敏感，捷径效果受限；2）在极高维或极大 k 值时，非度量 Chamfer 造成的图结构不理想；3）构建成本较高，需要聚类与量化预处理；4）实验只覆盖了四个数据集，泛化性待进一步验证。

---

## 123. A Knowledge-Informed Pretrained Model for Causal Discovery

**arXiv ID:** 2603.20842 | [PDF](https://arxiv.org/pdf/2603.20842v1)

**作者:** Wenbo Xu `[一作]` (Renmin University of China), Peng Cui `[通讯]` (Tsinghua University)

**通讯引用:** 20802 | [OpenAlex ID](https://openalex.org/A5009228005)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一个名为 Kode 的知识驱动预训练模型，能够在仅有粗粒度先验知识与观测数据的条件下自动推断因果图。

**💡 创新点**

创新点包括：①首创将弱先验知识与预训练模型结合；②双源编码器-解码器架构，实现数据与知识的协同推理；③构建多维度合成训练集（节点规模、稀疏度、生成机制）并配套课程学习策略；④使用可达性矩阵而非硬边约束来编码先验知识。

**🔧 技术方法**

核心技术：Transformer/注意力网络、双源对齐与交叉注意力、Gumbel‑Sinkhorn 置换解码、低三角解码、相似性正则、蒙特卡罗图似然、预训练+微调以及课程学习。

**📊 数据集**

使用的数据集包括：合成 300k DAG（节点 20/30/40，ER(1/2/3)，线性与非线性生成），以及真实世界数据 CIPCaD Bench（Ultra‑processed Food、Tennessee Eastman）和 Sachs 蛋白网络。

**📈 对比分析**

与 LiNGAM、PC、GES、NOTEARS‑MLP、CISvA、AVICI、BCNP 等基线在 SHD 与 F1 上进行对比，结果显示 Kode 在无先验知识的情况下已优于大多数基线，在弱或全先验下性能进一步提升，特别在高密度或 OOD 场景中表现突出。

**⚠️ 局限性**

局限性：需要大规模合成预训练，可能对与训练分布差异极大的真实机制不够鲁棒；对先验知识的质量高度依赖；未处理隐藏混杂变量；模型结构复杂，训练成本和推理时间相对较高；仅适用于无环有向图，无法直接处理循环或非可观测变量。

---

## 124. EnergyAction: Unimanual to Bimanual Composition with Energy-Based Models

**arXiv ID:** 2603.20236 | [PDF](https://arxiv.org/pdf/2603.20236v1)

**作者:** Mingchen Song `[一作]` (Harbin Institute of Technology), Weili Guan `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2329 | [OpenAlex ID](https://openalex.org/A5075938343)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用预训练的单臂操作策略，采用能量基模型（EBM）进行组合，实现双臂协同操作的策略迁移。

**💡 创新点**

三大创新：①将单臂策略视为能量函数并通过能量求和实现组合；②设计时间-空间协调约束，确保动作同步与碰撞避免；③提出两种能量感知自适应去噪策略，根据动作能量动态调整去噪步数，提升推理效率。

**🔧 技术方法**

核心技术包括能量基模型（EBM）与流匹配（Flow Matching）、能量约束优化、LSTM/MLP自适应权重预测、两种自适应去噪算法（Adaptive Denoising 与 Early Stop Denoising）。

**📊 数据集**

使用 RLBench（单臂任务）与 RLBench2（双臂任务）进行仿真评估，并在 Galaxea R1 Lite 机器人上进行真实环境验证。

**📈 对比分析**

与 ACT、RVT-LF、PerAct、AnyBimanual、3DFA、DP3 等SOTA方法对比，在20/100个演示样本下，EnergyAction 取得平均成功率 77.3%/86.4%，比最强基线 3DFA 高 4.6%（100示例）或 32.5%（20示例），在低样本环境下表现更为突出。

**⚠️ 局限性**

限制：仍依赖一定量的单臂预训练数据；在极其复杂的双臂协同任务或高维动作空间中，能量约束可能需要更精细的调参；自适应去噪阈值的选取可能对不同任务不够通用。

---

## 125. Inverting Neural Networks: New Methods to Generate Neural Network Inputs from Prescribed Outputs

**arXiv ID:** 2603.20461 | [PDF](https://arxiv.org/pdf/2603.20461v1)

**作者:** Rebecca Pattichis `[一作]` (University of California Los Angeles), Marios S. Pattichis `[通讯]` (University of New Mexico)

**通讯引用:** 4887 | [OpenAlex ID](https://openalex.org/A5114377331)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了神经网络输入反向映射问题，提出方法生成满足指定输出分布的输入图像。

**💡 创新点**

创新点在于同时引入前向根求解算法和后向层逐步反演算法，利用Jacobian与线性层的null空间随机采样，无需额外训练即可覆盖广泛输入空间。

**🔧 技术方法**

使用了Levenberg-Marquardt根求解、Jacobian计算、SVD伪逆、null空间随机采样、可逆激活函数以及Transformer和全连接网络架构。

**📊 数据集**

主要在MNIST数据集上实验，对ViT-tiny和DINOv3-Base进行微调，并在FCNN上验证。

**📈 对比分析**

通过与先前基于训练样本或梯度下降的生成方法比较，新方法能产生随机化的输入图像，目标类别概率≥0.9，展示了网络的脆弱性。

**⚠️ 局限性**

局限性包括仅适用于可逆激活函数和线性层，对非可逆ReLU等激活的网络需进一步扩展；在更复杂网络上收敛速度与稳定性仍有待提升。

---

## 126. Email in the Era of LLMs

**arXiv ID:** 2603.20231 | [PDF](https://arxiv.org/pdf/2603.20231v1)

**作者:** Dang Nguyen `[一作]` (University of Chicago), Ari Holtzman `[通讯]` (University of Chicago)

**通讯引用:** 5307 | [OpenAlex ID](https://openalex.org/A5063151917)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一款以撰写HR邮件为核心机制的沟通游戏，收集并分析了600余条人类与LLM生成的邮件，使用LLM评判者评估邮件质量以研究人类与LLM的沟通表现。

**💡 创新点**

发现人类+LLM协同写作在多种敏感情境下优于单独人类或LLM写作，揭示了“emergent tact”现象以及情感与正式度在邮件质量中的系统性差异。

**🔧 技术方法**

采用GPT‑4o、Qwen‑3、Claude‑Sonnet、Gemini等大型语言模型作为作者、评判者和回复者，并结合Elo评分、Krippendorff’s α、tact/empathy/formality注释等技术。

**📊 数据集**

使用HR模拟器游戏生成的600多条手工撰写与LLM生成的邮件数据集，并通过模型与人工标注进行情感与正式度注释。

**📈 对比分析**

通过LLM评判者的通过率（pass rate）和Elo排名对比人类与模型邮件，结果显示人类成功率为23.5%，LLM为48–54%，但人类+LLM在部分场景中几乎达到100%通过率，且模型评判的一致性随规模提升而增强。

**⚠️ 局限性**

LLM难以生成或复制低同理心低正式度的邮件，评判标准可能与人类偏好不一致，且实验依赖特定LLM评判者，缺乏跨场景的普适性。

---

## 127. Uni-Classifier: Leveraging Video Diffusion Priors for Universal Guidance Classifier

**arXiv ID:** 2603.20382 | [PDF](https://arxiv.org/pdf/2603.20382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 128. Towards Extended Reality Intelligence for Monitoring and Predicting Patient Readmission Risks

**arXiv ID:** 2603.20556 | [PDF](https://arxiv.org/pdf/2603.20556v1)

**作者:** Martin Sanchez `[一作]` (San Jose State University), Vuthea Chheang `[通讯]` (San Jose State University)

**通讯引用:** 450 | [OpenAlex ID](https://openalex.org/A5036188615)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文将XGBoost模型与Apple Vision Pro混合现实（MR）技术结合，构建了“PatientCard”界面，用于实时展示糖尿病住院患者30天再住院风险、关键特征和护理建议；

**💡 创新点**

创新点在于将表格机器学习预测结果通过MR空间化、免手交互的方式直接投射到病床前，解决传统EHR仪表盘难以即时获取、信息碎片化的问题；

**🔧 技术方法**

使用的技术包括Python‑XGBoost 3.0.5进行模型训练、SwiftUI/RealityKit/ARKit和Apple Core ML在Vision Pro上实现离线推理和MR渲染；

**📊 数据集**

数据集为公开的Diabetes 130‑US Hospitals（1999–2008）共约10万条住院记录，包含50个患者特征；

**📈 对比分析**

模型通过80/16/20的训练/验证/测试拆分，使用AUPRC和AUROC衡量；在测试集上获得AUROC 0.72、AUPRC 0.11，优于传统HOSPITAL/LACE等线性指标；MR原型实现了无云、低延迟的离线推理；

**⚠️ 局限性**

局限性包括：仅使用旧公开数据，缺乏当前临床验证；训练仅依赖单一随机拆分与早停，未充分评估泛化；缺乏真实临床工作流程中的用户体验与效果评估；以及高端MR硬件成本限制了广泛推广。

---

## 129. Algorithmic Audit of Personalisation Drift in Polarising Topics on TikTok

**arXiv ID:** 2603.20723 | [PDF](https://arxiv.org/pdf/2603.20723v1)

**作者:** Branislav Pecher `[一作]` (Kempelen Institute of Intelligent Technologies), Ivan Srba `[通讯]` (Kempelen Institute of Intelligent Technologies)

**通讯引用:** 867 | [OpenAlex ID](https://openalex.org/A5082763244)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过算法审计评估TikTok在多种极化话题（平地说、疫苗、气候变化、美国政治）和中性料理话题上的个性化漂移，采用模拟账号（sockpuppet）与LLM驱动的用户交互预测器，收集并分析近8万条视频推荐数据，量化偏好对齐漂移、话题漂移和立场漂移；

**💡 创新点**

1) 采用LLM（GPT‑4.1）+ Whisper 对视频进行自动化语义标注，提升模拟用户行为的真实性；2) 设计三种漂移度量并在同一实验框架下比较多话题的表现；3) 在TikTok上首次进行大规模（68个账号、15‑16天）系统性审计，公开数据与代码。

**🔧 技术方法**

使用 TikTok Web 接口抓取视频及元数据，GPT‑4.1 进行主题与立场判别，Whisper 大模型提取音频转写，Python 自动化脚本实现账号操作与交互预测，统计回归与 Mann‑Whitney U 检验评估漂移趋势。

**📊 数据集**

自建的近80k条 TikTok 视频数据（涵盖四极化话题与中性料理），用于实验的主要数据集；另外使用350条手工标注视频进行 LLM 预测准确率评估；实验数据与代码已公开于 Zenodo 与 GitHub。

**📈 对比分析**

通过比较各话题下的偏好漂移比例、话题漂移比值、立场漂移比值，并使用回归拟合趋势、Mann‑Whitney U 检验验证显著性。结果显示：中性话题的个性化更强；气候变化、疫苗、平地说等误信息话题出现中和漂移；美国政治维持平衡，且在混合立场下略向对立立场漂移。LLM 预测准确率达到 98% 以上。

**⚠️ 局限性**

1) 新建账号可能受到平台更强探索行为的影响；2) 仅在美国地区开展，结果可能不具全球泛化性；3) 受时间与热点流行度影响，结论随内容变化；4) 仅使用单一中性话题，可能无法完全反映真实用户兴趣；5) LLM 预测仍存在误差，需人工复核；6) 未探索其他交互反馈组合。

---

## 130. Which Workloads Belong in Orbit? A Workload-First Framework for Orbital Data Centers Using Semantic Abstraction

**arXiv ID:** 2603.20317 | [PDF](https://arxiv.org/pdf/2603.20317v1)

**作者:** Durgendra Narayan Singh `[一作]` `[通讯]` (Independent Researcher), Durgendra Narayan Singh (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出工作负载优先框架，评估哪些任务适合在轨道计算机上运行，并通过两项原型验证语义压缩降低下行带宽

**💡 创新点**

首次对轨道计算约束进行第一性原理分析并构建评估矩阵；通过卫星影像语义化和多视角三维重建证明语义压缩可将数据量压缩99%以上

**🔧 技术方法**

语义化处理（Sentinel‑2 SCL云分割、矢量化多边形、patch多边形）、多视角立体重建（ORB特征匹配、深度融合）、GPU加速与存储转发等技术

**📊 数据集**

Sentinel‑2 L2A 影像（Seattle、Bengaluru）与 Maxar（Vantor）全色ARD（Los Angeles）多日图像

**📈 对比分析**

与原始数据对比，EO语义压缩实现99.69‑99.996%压缩，三维重建实现99.49%压缩；传输时间从5.03 s降至0.014 s，显著提升有效下行吞吐

**⚠️ 局限性**

原型仅基于确定性预处理，未覆盖学习模型精度、完整分辨率、操作开销、容错和能源建模等方面

---

## 131. Expected Reward Prediction, with Applications to Model Routing

**arXiv ID:** 2603.20217 | [PDF](https://arxiv.org/pdf/2603.20217v1)

**作者:** Kenan Hasanaliyev `[一作]` (Stanford University), Alexander Nicholas D'Amour `[通讯]` (Google DeepMind)

**通讯引用:** 2670 | [OpenAlex ID](https://openalex.org/A5060694111)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何从提示文本直接预测语言模型在该提示下生成回答的期望奖励，并将此预测用于模型路由。

**💡 创新点**

发现即使使用极简的线性回归和预训练嵌入，期望奖励也能高精度预测；同时证明该预测足以替代传统的对比式偏好学习，简化模型路由流程。

**🔧 技术方法**

采用线性回归（岭回归）对提示嵌入（gte-large-en-v1.5）进行训练，并利用奖励模型（OpenAssistant‑RM、GRM‑2B‑RM、InternLM‑RM）计算真实奖励。

**📊 数据集**

使用 open‑perfectblend 数据集（四类：代码推理、数学推理、指令跟随、通用聊天）共 4000 条提示。

**📈 对比分析**

与固定模型、随机模型、基于类别的最优模型、Zooter 等基线对比；ERP 路由在成本-惩罚曲线上击败所有非奥利基基线，并在大多数奖励模型上优于甚至超越按类别最优路由。

**⚠️ 局限性**

局限性包括：对期望奖励的依赖假设模型奖励分布相对集中；若模型产生极端好坏答案，期望奖励可能误导；实验仅限于已知提示类别的公开数据集，需进一步验证在更广泛场景下的鲁棒性。

---

## 132. FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection

**arXiv ID:** 2603.20403 | [PDF](https://arxiv.org/pdf/2603.20403v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 133. CollabORAN: A Collaborative rApp-xApp-dApp Control Architecture for Fairness-Adaptive Resource Sharing in O-RAN

**arXiv ID:** 2603.20805 | [PDF](https://arxiv.org/pdf/2603.20805v1)

**作者:** Anastasios Giannopoulos `[一作]` (Four Dot Infinity), Panagiotis Trakadas `[通讯]` (Four Dot Infinity)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了CollabORAN框架，在O-RAN架构中通过rApp、xApp和dApp的协同控制，实现了动态公平的频谱共享。

**💡 创新点**

创新点在于将频谱优化任务按时域拆分为长期策略（rApp）、近实时干扰管理（xApp）和毫秒级时隙调度（dApp）三层协同，并使用超图建模与图染色解决多干扰环境下的PRB分配。

**🔧 技术方法**

采用LSTM等AI/ML模型进行流量预测，超图基的PRB图染色算法以及改进的比例公平调度策略；通过O-RAN标准接口（O1、A1、E2）实现多层协同。

**📊 数据集**

使用公开的5G流量时间序列数据集进行流量预测模型训练与评估，并在模拟的双小区O-RAN环境（10 MHz、3.5 GHz）中进行仿真。

**📈 对比分析**

通过与仅rApp–xApp、仅xApp–dApp两种基线方案在不同用户需求下的对比实验，CollabORAN在保持≈91%成功率的同时实现≈92%Jain公平指数，明显优于单层方案。

**⚠️ 局限性**

实验仅在两小区模拟环境中验证，未在更大规模真实网络中测试；可能存在信令开销、时延耦合以及对不同部署场景的适配性限制。

---

## 134. Does Peer Observation Help? Vision-Sharing Collaboration for Vision-Language Navigation

**arXiv ID:** 2603.20804 | [PDF](https://arxiv.org/pdf/2603.20804v1)

**作者:** Qunchao Jin `[一作]` (Adelaide University), Qi Wu `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出Co-VLN框架，探究并实现多智能体共享观察以提升VLN性能。

**💡 创新点**

创新点在于首次系统评估同环境多代理的空间重叠观察共享对VLN的益处，并给出最小化、模型无关的实现方案。

**🔧 技术方法**

采用基于空间重叠检测的对齐、图融合与多模态大模型（如MapGPT、DUET）等技术。

**📊 数据集**

使用标准Room-to-Room (R2R) 数据集的val unseen 与R2R-Hard子集进行实验。

**📈 对比分析**

与现有最先进方法相比，在DUET和MapGPT上均获得显著SR/SPL提升，达到或超过SOTA。

**⚠️ 局限性**

局限在于依赖空间重叠的检索准确度、对大规模代理协同的扩展性尚未彻底验证，且对真实机器人部署的同步通信需求未深入探讨。

---

## 135. AI-Driven Multi-Agent Simulation of Stratified Polyamory Systems: A Computational Framework for Optimizing Social Reproductive Efficiency

**arXiv ID:** 2603.20678 | [PDF](https://arxiv.org/pdf/2603.20678v1)

**作者:** Yicai Xing `[一作]` `[通讯]` (Independent Researcher), Yicai Xing (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并量化评估一种分层多配偶制（Stratified Polyamory System，SPS），旨在缓解全球生育率下降和婚姻结构失衡所带来的社会危机。

**💡 创新点**

创新点在于：①将多配偶制抽象为可在代理系统中实现的规则集合；②将其与社会化育儿、遗产改革等机制结合，形成完整的社会政策框架；③采用多维度技术（ABM、MARL、LLM生成代理、GNN、进化算法）进行模型构建、优化与评估，首次从算法公平与社会福利双重视角验证多配偶制的 Pareto 改进。

**🔧 技术方法**

技术手段包括：Agent‑Based Modeling（模拟个体属性与交互），Multi‑Agent Reinforcement Learning（使用 PPO 与 CTDE 训练交配策略），Large Language Model 生成代理（模拟情感、嫉妒等定性决策），Graph Neural Network（分析社交网络结构与财富流动），以及遗传算法（优化制度参数）。

**📊 数据集**

使用的数据集主要有：①全球与东亚国家的生育率与人口统计数据（UN Population Division、各国统计局）；②在线交友与婚恋平台的匹配与偏好数据（用于校准 mate‑value、吸引力分布）；③经济与社会资本分布数据（美国 Census、欧洲劳工统计）；④相关文献与实验结果（Budig & England、Correll 等的母亲惩罚与就业歧视研究）。

**📈 对比分析**

通过在同一代理环境下分别模拟“严格单配偶制”与“SPS”，比较指标包括平均福利指数、总生育率、财富 Gini 系数、网络连通性等。初步小规模仿真（N=1,000，50 年）显示：SPS 平均福利提升 18‑25%，C‑tier 受益最大（+140%）；总生育率提升至 1.7‑1.9，近似恢复；财富 Gini 指数下降 8‑12%；网络呈小世界结构，B‑tier 充当关键桥梁。性能提升表明 SPS 在多维度上实现 Pareto 改进。

**⚠️ 局限性**

主要局限：①A/B/C 级别模型人为简化，实际人群属性连续分布；②福利与生育预测基于理论和模拟，缺乏充分经验校准；③实施路径需跨领域政治、文化认同，现实可行性未知；④未充分评估儿童在多配偶环境下的长期发展与心理影响；⑤MARL 与 LLM 模型的可解释性与稳健性待提升，尤其在多样化文化情境下的泛化能力。

---

## 136. Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves

**arXiv ID:** 2603.20850 | [PDF](https://arxiv.org/pdf/2603.20850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 137. RECLAIM: Cyclic Causal Discovery Amid Measurement Noise

**arXiv ID:** 2603.20585 | [PDF](https://arxiv.org/pdf/2603.20585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 138. Report-based Recommendations for Policy Making and Agency Operations: Dataset and LLM Evaluation

**arXiv ID:** 2603.20287 | [PDF](https://arxiv.org/pdf/2603.20287v1)

**作者:** Aleksandra Edwards `[一作]`, Alun Preece `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了利用大型语言模型生成公共部门政策建议的任务，提出了首个专门的推荐生成基准PubRec-Bench。

**💡 创新点**

创新点在于首次定义政策建议生成这一专门NLG任务并构建统一基准，同时结合LLM与专家评估探讨评测指标的适用性。

**🔧 技术方法**

使用GPT‑4o、Cohere Command R+、LLaMA 3三大LLM进行提示式生成，并用BERTScore、ROUGE‑L、BLEU、LLM‑based评估以及人类专家评估进行多维度评测。

**📊 数据集**

使用PubRec‑Bench数据集，该数据集由英国关怀机构、美国儿童局和NSPCC三份公开报告抽取的证据‑建议对构成。

**📈 对比分析**

评测结果显示GPT‑4o在零样本和一样本条件下表现最佳，平均分约2.5‑2.8/3，约60%建议被评为可执行。

**⚠️ 局限性**

限制包括仅英文、英国/美国背景、样本量有限、仅零/一样本设置，以及可能存在与预训练数据的重叠。

---

## 139. From Data to Laws: Neural Discovery of Conservation Laws Without False Positives

**arXiv ID:** 2603.20474 | [PDF](https://arxiv.org/pdf/2603.20474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 140. EARTalking: End-to-end GPT-style Autoregressive Talking Head Synthesis with Frame-wise Control

**arXiv ID:** 2603.20307 | [PDF](https://arxiv.org/pdf/2603.20307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 141. RISE: Real-time Image Processing for Spectral Energy Detection and Localization

**arXiv ID:** 2603.20481 | [PDF](https://arxiv.org/pdf/2603.20481v1)

**作者:** Chung-Hsuan Tung `[一作]` (Duke University), Tingjun Chen `[通讯]` (Duke University)

**通讯引用:** 3171 | [OpenAlex ID](https://openalex.org/A5034818470)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一套基于图像处理的实时谱能量检测与定位系统，能够在保持高吞吐量的同时满足实时延迟约束。

**💡 创新点**

创新点包括：①将能量检测问题转化为二维图像处理，使用自适应Otsu阈值、形态学运算和连通分量标记，实现低复杂度（O(T·F·logF)）的能量块定位；②基于多线程、ping‑pong缓冲的实时体系结构，显著降低处理延迟并保持无堵塞吞吐；③通过严格的 IoU 与检测概率评估，兼顾定位精度与检测率。

**🔧 技术方法**

使用技术包括：FFT、PSD估计、Savitzky–Golay滤波、Otsu自适应阈值、形态学开闭、连通分量标记、C++多线程（Agora框架）和Armadillo线性代数库；系统运行在多核CPU上，配合多线程调度实现并行化。

**📊 数据集**

实验数据集：①由Sig‑Gen生成的合成I/Q样本，包含多种Wi‑Fi、BLE、DSSS等信号并配有精确的 ground‑truth 位置标签；②在USRP X310 SDR上收集的 OTA 测试数据，覆盖 100 MHz 宽带和 500 MHz 宽带场景，并通过控制功率与噪声实现可调 SNR。

**📈 对比分析**

与代表性基线（Searchlight、DeepRadar、U‑Net）进行对比：在 100 MHz 100 MSps 测试下，系统在 IoU=0.5 时实现平均 IoU≈63%、P_d≈70%，延迟比 Searchlight 低 20.5×，IoU 比 DeepRadar 高 56% 且延迟 1.65×；与 U‑Net 比较，系统的延迟仅为 1/213，且在 CPU 上即可满足实时；在 OTA 场景中，系统保持了与模拟相同的性能趋势。

**⚠️ 局限性**

局限性包括：①对极低 SNR 信号的检测与定位仍易出现漏检或分块；②系统主要针对能量块定位，缺乏对信号类型、调制方式的识别；③实现依赖多核 CPU 与复杂线程调度，部署到低功耗嵌入式平台仍需进一步优化。

---

## 142. Remote Sensing Image Dehazing: A Systematic Review of Progress, Challenges, and Prospects

**arXiv ID:** 2603.20289 | [PDF](https://arxiv.org/pdf/2603.20289v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 143. Incremental GNN Embedding Computation on Streaming Graphs

**arXiv ID:** 2603.20622 | [PDF](https://arxiv.org/pdf/2603.20622v1)

**作者:** Qiange Wang `[一作]` (National University Of Singapore), Bingsheng He `[通讯]` (National University Of Singapore)

**通讯引用:** 21375 | [OpenAlex ID](https://openalex.org/A5039946576)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在实时动态图上实现增量式GNN嵌入计算，避免全邻居重新计算，显著提升推理效率。

**💡 创新点**

提出细粒度算子拆解与安全重排的增量框架，并证明其在满足一定可结合性与分布性条件时保持与完整计算等价；同时设计了CPU‑GPU协同系统与高效内存管理。

**🔧 技术方法**

使用图神经网络分层拆解（msg_local、nbr_ctx、msg_cbn、aggregate、update）、GPU计算、CPU内存压缩（PMA‑CSR）、零拷贝内存访问、分块调度和增量更新图构造。

**📊 数据集**

在工业规模数据集 ogbn‑arxiv、Reddit、ogbn‑products、Twitter、ogbn‑paper 和 Friendster 上进行实验。

**📈 对比分析**

与全邻居、邻居采样、无增量重用等基线对比，增量方法在小图上实现 2.6–4.1 倍吞吐，Billion-scale 图上实现 37.8–145.8 倍速度提升，且保持近乎相同的准确率。

**⚠️ 局限性**

对 GAT、AGNN 等需重新计算部分顶点的模型仍存在约 1.2–1.7 倍额外开销；系统对显存容量有一定依赖，需 CPU 内存容纳中间结果；对某些非可结合/分布的 GNN 架构支持有限。

---

## 144. GraphiContact: Pose-aware Human-Scene Robust Contact Perception for Interactive Systems

**arXiv ID:** 2603.20310 | [PDF](https://arxiv.org/pdf/2603.20310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 145. Cross-Granularity Representations for Biological Sequences: Insights from ESM and BiGCARP

**arXiv ID:** 2603.20825 | [PDF](https://arxiv.org/pdf/2603.20825v1)

**作者:** Hanlin Xiao `[一作]` (University of Manchester), Mauricio A. Álvarez `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了跨粒度表示在生物序列中的集成，通过案例研究BiGCARP（Pfam域级模型）和ESM（氨基酸级模型）评估嵌入初始化和层次提取的效果。

**💡 创新点**

揭示跨粒度嵌入初始化对BiGCARP性能影响有限，证明不同粒度模型的互补性，并通过简单拼接提升中间层任务性能，提出跨粒度集成的潜在价值。

**🔧 技术方法**

使用Transformer/编码器模型（ESM、BiGCARP）、嵌入提取策略、UMAP可视化、Centered Kernel Alignment (CKA) 代表性分析、MLP分类器等技术。

**📊 数据集**

使用antiSMASH-DB作为BiGCARP训练集、MIBiG数据库进行probe任务、Pfam数据库进行表示可视化。

**📈 对比分析**

通过BGC产品类别预测、A-domain底物预测、卤素产品预测等probe任务与不同嵌入策略对比，发现BiGCARP最后层嵌入优于嵌入层，ESM+BiGCARP拼接在中间层任务上显著提升AUROC（如BGC产品分类AUROC 0.917，卤素预测AUROC 0.759）。

**⚠️ 局限性**

主要限制在于实验采用简单拼接与MLP，缺乏更复杂的融合方法；数据集规模有限，尤其是卤素预测仅46条样本，未能验证更大规模数据下的鲁棒性。

---

## 146. Context Cartography: Toward Structured Governance of Contextual Space in Large Language Model Systems

**arXiv ID:** 2603.20578 | [PDF](https://arxiv.org/pdf/2603.20578v1)

**作者:** Zihua Wu `[一作]` (NVIDIA), Georg Gartner `[通讯]` (TU Wien)

**通讯引用:** 4312 | [OpenAlex ID](https://openalex.org/A5013025902)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Context Cartography 框架，定义三域（黑雾、灰雾、可视场）及七种空间治理操作（侦察、选择、简化、聚合、投影、位移、分层），并用这些操作解释当前主流 LLM 代理系统的设计模式。

**💡 创新点**

创新点在于将地图学的通用治理原则（选取、简化、投影等）转化为 LLM 上下文空间的正式算子，提供了一个完整的“地图式”上下文管理理论，并提出可验证的预测和诊断基准。

**🔧 技术方法**

技术方法包括：基于 Transformer 注意力的 salience 几何分析；系统性覆盖所有域间转换以推导算子；使用 Lean4 进行形式化验证；案例分析对四个工业系统（Claude Code、Letta、MemOS、OpenViking）进行映射；设计了可供实验的操作消融流程。

**📊 数据集**

本文主要以现有公开系统的实现细节为数据来源（代码仓库、论文说明），并未使用专门的标注数据集；提出了后续评估所需的 Context Cartography Diagnostic（CCD）基准，但尚未公布数据集。

**📈 对比分析**

比较方法：通过对四个系统的算子实现深度打分（0–5 维度）展示各算子在实际系统中的普及度；提出消融实验方案来测量各算子对推理准确率、Token 消耗、错误类型的影响。论文未给出量化性能提升数值，主要侧重理论验证和案例说明。

**⚠️ 局限性**

局限性包括：框架为解释性理论，缺乏完整的数学证明与实证验证；案例分析基于已观察到的系统，未进行独立预测；缺少针对不同模型架构（SSM、扩散模型等）的泛化分析；未提供完整的基准数据与实验结果；对多智能体协同治理的公式化仍不完善。

---

## 147. VGS-Decoding: Visual Grounding Score Guided Decoding for Hallucination Mitigation in Medical VLMs

**arXiv ID:** 2603.20314 | [PDF](https://arxiv.org/pdf/2603.20314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 148. yProv4DV: Reproducible Data Visualization Scripts Out of the Box

**arXiv ID:** 2603.20437 | [PDF](https://arxiv.org/pdf/2603.20437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 149. HiCI: Hierarchical Construction-Integration for Long-Context Attention

**arXiv ID:** 2603.20843 | [PDF](https://arxiv.org/pdf/2603.20843v1)

**作者:** Xiangyu Zeng `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 21949 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为HiCI的层次化注意力模块，用于长上下文语言建模，通过构建段级表示、整合为共享全局上下文，并将其广播以调节段级注意力。

**💡 创新点**

HiCI通过显式的层次结构组织长上下文信息，作为长上下文建模的归纳偏置，展示了其有效性。

**🔧 技术方法**

使用了层次化注意力机制，结合了位置插值和FlashAttention-2技术，以实现高效的长序列计算。

**📊 数据集**

在预训练的LLaMA-2模型上进行验证，扩展上下文长度从4K到100K（7B模型）和64K（13B模型），并使用PG-19、Proof-pile等数据集进行评估。

**📈 对比分析**

与强基线模型相比，HiCI在语言建模、检索和指令跟随基准上均表现出一致的性能提升，尤其在代码理解任务中超越了GPT-3.5-Turbo-16K，且在主题检索中与专有模型相匹配。

**⚠️ 局限性**

HiCI的局限性在于其依赖于特定的模型架构和训练过程，可能在其他架构或未经过类似训练的情况下表现不佳。

---

## 150. The Innovation Recognition Paradox: How Science Undervalues the Boundary-Crossing Work Women Produce

**arXiv ID:** 2603.20597 | [PDF](https://arxiv.org/pdf/2603.20597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 151. Large Neighborhood Search meets Iterative Neural Constraint Heuristics

**arXiv ID:** 2603.20801 | [PDF](https://arxiv.org/pdf/2603.20801v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 152. Deep reflective reasoning in interdependence constrained structured data extraction from clinical notes for digital health

**arXiv ID:** 2603.20435 | [PDF](https://arxiv.org/pdf/2603.20435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 153. Emergency Lane-Change Simulation: A Behavioral Guidance Approach for Risky Scenario Generation

**arXiv ID:** 2603.20234 | [PDF](https://arxiv.org/pdf/2603.20234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 154. From 50% to Mastery in 3 Days: A Low-Resource SOP for Localizing Graduate-Level AI Tutors via Shadow-RAG

**arXiv ID:** 2603.20650 | [PDF](https://arxiv.org/pdf/2603.20650v1)

**作者:** Zonglin Yang `[一作]` (Peking University), Zhi-X. Chen `[通讯]` (Peking University)

**通讯引用:** 1513 | [OpenAlex ID](https://openalex.org/A5075531981)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套低资源标准操作程序（SOP），利用视觉语言模型对教学截图进行语义清洗，并采用 Shadow‑RAG 双代理架构，将开放权重 32B 语言模型部署在单张消费级 GPU 上，构建了研究生级应用数学 AI 导师；

**💡 创新点**

将 Shadow‑RAG 架构引入 AI 导师，Shadow 代理在检索后提供方法论、差异警告和逻辑过滤，显著提升推理可靠性，并首次揭示结构化指导引发的“非线性出现”现象；

**🔧 技术方法**

视觉语言模型（Gemini 3 Pro）用于语义转录；Shadow‑RAG 双代理（Shadow 代理 + 主导师）+ 代码工具（SymPy/NumPy）; 开放权重 32B LLM（Qwen2.5‑32B‑Instruct、Qwen3‑32B）；评估使用 DeepSeek‑V3 作为判定器；

**📊 数据集**

45 节课的黑板教学截图（约 150 张）经过 VLM 转录生成 103 个 Markdown 文件（约 428 KB）；改写后的研究生期末考试（5 道大题）作为评估数据集；

**📈 对比分析**

对 5 种配置（Zero‑shot、Naive RAG、Shadow (Full/Dynamic)、Shadow (No Code)、Shadow (Forced Tools)）分别做 5 次每题推理，总计 250 次。Qwen3‑32B 在 Shadow‑Forced 下达 90% 准确率，Shadow‑Dynamic/No Code 85%；Baseline 67%，Naive RAG 74%；Qwen2.5‑32B 最高 65%，其余 50‑57%。展示了 Shadow 架构将高版本模型跃升至掌握水平；

**⚠️ 局限性**

代理交互导致令牌消耗约增加 10 倍，延迟显著；对不同基础模型的工具选择需手动调优；较弱模型提升有限；仅在单一学科验证，需进一步推广至其他 STEM 领域并进行真实环境评估。

---

## 155. WWW.Serve: Interconnecting Global LLM Services through Decentralization

**arXiv ID:** 2603.20661 | [PDF](https://arxiv.org/pdf/2603.20661v1)

**作者:** Huanyu Wang `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5073845046)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个完全去中心化的LLM服务框架WWW.Serve，提供开放市场、匿名调度、信用交易、双决裁评估等功能；

**💡 创新点**

创新点包括：①基于信用的交易系统和PoS调度实现去中心化的资源调度；②双决裁-裁判机制用于分布式质量评估；③轻量级gossip协议实现动态节点同步；④用户级政策框架允许服务提供商自由调节参与度；⑤用博弈论证明系统收敛到高质量均衡；

**🔧 技术方法**

使用了PoS选举、区块链式信用账本、gossip同步、双决裁评估、模块化节点管理器、分布式调度与资源匹配技术；

**📊 数据集**

在多种LLM模型（如Qwen3、LLaMA 3.1、DeepSeek-R1等）、不同量化方案、不同推理后端以及不同GPU硬件（A100、RTX4090、RTX3090）上进行基准测试，衡量SLO达成率与延迟；

**📈 对比分析**

与单节点部署和集中式调度进行对比；实验表明WWW.Serve在SLO达成率上提升最多1.5倍，平均请求延迟降低27.6%；在动态节点加入/离线场景下保持服务连续性；双决裁机制对性能影响小；

**⚠️ 局限性**

局限性包括：①双决裁机制引入一定额外开销；②系统对节点信用和stake分配敏感；③对恶意节点的安全分析尚不充分；④在极大规模网络下的扩展性与可观测性未完全验证；⑤实际部署和真实流量的评估尚待进一步验证。

---

## 156. VSD-MOT: End-to-End Multi-Object Tracking in Low-Quality Video Scenes Guided by Visual Semantic Distillation

**arXiv ID:** 2603.20731 | [PDF](https://arxiv.org/pdf/2603.20731v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 157. Where can AI be used? Insights from a deep ontology of work activities

**arXiv ID:** 2603.20619 | [PDF](https://arxiv.org/pdf/2603.20619v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 158. Cenergy3: An Open Software Package for City Energy 3D Modeling

**arXiv ID:** 2603.20361 | [PDF](https://arxiv.org/pdf/2603.20361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 159. Towards Intelligent Geospatial Data Discovery: a knowledge graph-driven multi-agent framework powered by large language models

**arXiv ID:** 2603.20670 | [PDF](https://arxiv.org/pdf/2603.20670v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 160. Dodgersort: Uncertainty-Aware VLM-Guided Human-in-the-Loop Pairwise Ranking

**arXiv ID:** 2603.20839 | [PDF](https://arxiv.org/pdf/2603.20839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 161. Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention

**arXiv ID:** 2603.20640 | [PDF](https://arxiv.org/pdf/2603.20640v1)

**作者:** Manh Nguyen `[一作]` (Deakin University), Hung Le `[通讯]` (Deakin University)

**通讯引用:** 1528 | [OpenAlex ID](https://openalex.org/A5101936199)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级多代理辩论框架DAR，通过仅保留相互不一致的代理回应来提升大语言模型推理质量。

**💡 创新点**

创新点在于使用基于不一致性的索引过滤器，无需置信度阈值或额外参数，确保保留真实多样性而不改变原始信息。

**🔧 技术方法**

核心技术包括平均负对数似然的不确定性评分、加入前一轮多数投票作为上下文、以及利用LLM实现的过滤代理选取最大互异子集。

**📊 数据集**

在六个多样化推理与问答基准（Arithmetics、GSM8K、HH‑RLHF、MMLU Pro.Med.、Form.Log.、CSQA）以及四种模型（Qwen2.5‑1.5B/3B、Falcon‑7B、Llama3.1‑8B）上进行实验。

**📈 对比分析**

与Society Of Mind、MAD‑M²、Uncertain Prompt、Majority Vote、Vote Prompt等基线对比，DAR在大多数任务与模型上均取得平均准确率最高，尤其在代理数增大时优势更显著。

**⚠️ 局限性**

局限在于仍需依赖LLM的推理能力，对低质量回答的过滤效果受模型规模与生成格式一致性影响；长辩论收益有限，且在某些任务中对不确定性与多样性权衡需进一步研究。

---

## 162. Profiling learners' affective engagement: Emotion AI, intercultural pragmatics, and language learning

**arXiv ID:** 2603.20479 | [PDF](https://arxiv.org/pdf/2603.20479v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 163. Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping

**arXiv ID:** 2603.20755 | [PDF](https://arxiv.org/pdf/2603.20755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 164. Resource Allocation in Electricity Markets with Budget Constrained Customers

**arXiv ID:** 2603.20277 | [PDF](https://arxiv.org/pdf/2603.20277v1)

**作者:** Lila Perkins `[一作]` (University of Washington), Baosen Zhang `[通讯]` (University of Washington)

**通讯引用:** 6154 | [OpenAlex ID](https://openalex.org/A5013901541)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究电力市场中用户预算约束对资源分配与价格形成的影响，提出了一种在预算约束下的福利最大化模型，并证明其存在唯一的竞争均衡；同时给出了一个基于双重上升法的迭代算法，证明该算法收敛到该均衡，并通过构造修改后的效用函数，将原始自相参照的预算约束转化为一个凸的无约束福利最大化问题；

**💡 创新点**

创新点在于（1）揭示预算约束可通过对用户效用函数进行分段改造（将预算绑定区间替换为 b·log x+常数）来消除对价格的自参照性，形成一个严格凸的优化问题；（2）证明在标准凸性假设下该改造后问题唯一可解，且对应原始双重上升算法的收敛均衡；（3）提供了闭式预算绑定需求与改造效用的构造方法，可适用于任意数量的交叉点；

**🔧 技术方法**

主要使用了凸优化与拉格朗日双重理论、双重上升（dual‑ascent）迭代法、Lyapunov函数分析以及对效用函数的分段构造；

**📊 数据集**

实验采用合成的用户效用函数（如二次型和平方根型）和系统成本函数 C(y)=½y²，未使用真实电力市场数据集；

**📈 对比分析**

通过对比无约束福利最大化与改造后预算约束福利最大化的结果，展示了迭代算法在多种效用场景下的收敛到唯一均衡价格，并在表格中比较了最优分配与支出；性能方面，算法收敛迅速（如两用户例子收敛到 λ*≈2.743），并成功捕捉到预算绑定导致的需求曲线变化；

**⚠️ 局限性**

局限性包括：仅考虑单机无网络约束的单一时段问题；未纳入公平性指标（如比例公平）；缺乏对大规模用户聚合行为的分析；以及未处理多节点网络中的输电约束和节点定价。

---

## 165. Thinking into the Future: Latent Lookahead Training for Transformers

**arXiv ID:** 2603.20219 | [PDF](https://arxiv.org/pdf/2603.20219v1)

**作者:** Lorenzo Noci `[一作]` (Apple), Moin Nabi `[通讯]` (Apple)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了Latent Lookahead框架，使语言模型在生成前在隐层中对多步未来进行模拟和自我修正，从而提升推理性能。

**💡 创新点**

创新点在于将上下文扩展与递归计算相结合，采用多步隐层监督和非因果注意掩码，允许在不采样的情况下并行生成隐思考，显著提升模型的计划与推理能力。

**🔧 技术方法**

使用技术包括基于Transformer的多层模型、在上下文中插入latent token、非因果注意掩码、深度监督（多步目标）、随机/顺序的latent位置分配以及对训练时的并行化设计。

**📊 数据集**

实验使用了4×4与9×9 Sudoku、ProsQA DAG推理、Maze路径等结构化规划任务的数据集。

**📈 对比分析**

与标准NTP、Pause Tokens、原始MTP、Looped Transformer等基线对比，Latent Lookahead在9×9 Sudoku上从12.5%提升至35%，Mini Sudoku上从78%提升至93.5%，且性能随τ增加呈单调上升。

**⚠️ 局限性**

局限性包括训练成本较高、在大规模预训练模型中难以学习lookahead行为、计算开销随τ线性增长，以及对更通用任务的适用性仍需进一步验证。

---

## 166. Compass: Optimizing Compound AI Workflows for Dynamic Adaptation

**arXiv ID:** 2603.20821 | [PDF](https://arxiv.org/pdf/2603.20821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 167. A Modular LLM Framework for Explainable Price Outlier Detection

**arXiv ID:** 2603.20636 | [PDF](https://arxiv.org/pdf/2603.20636v1)

**作者:** Shadi Sartipi `[一作]` (Amazon.com, Inc.), Shervin Malmasi `[通讯]` (Amazon.com, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个三阶段的agentic LLM框架，用于可解释的电商价格异常检测。

**💡 创新点**

将价格异常检测建模为基于语义的推理任务，分为相关性分类、相对效用评估和推理决策，并生成可验证的理由链。

**🔧 技术方法**

采用Claude 3.5 Sonnet LLM，配合对比学习的产品嵌入、检索增强（RAG）与图增强、属性选择（generic、static、dynamic、weighted）以及基于四象限的决策策略。

**📊 数据集**

构造了四部分数据集：40条银色集（含人工标注的异常/非异常）、185条单侧标注集（人类全否）、12条边缘案例集、以及5400条未标注高流量商品，所有数据来自多家电商平台。

**📈 对比分析**

与Zero‑Shot和RAG基线在同一数据集上比较，使用精度、召回、F1、与人类审核一致率以及异常率等指标；agentic 在银色集上 F1=0.55、精度=1.0、召回=0.38、与人类一致率 76.3%、异常率 7.8%，显著优于基线。

**⚠️ 局限性**

受限于需大量人工标注、对属性缺失敏感、动态属性选择导致性能波动、评价主观性高、以及对数据噪声与完整度的依赖。

---

## 168. Procedural Refinement by LLM-driven Algorithmic Debugging for ARC-AGI-2

**arXiv ID:** 2603.20334 | [PDF](https://arxiv.org/pdf/2603.20334v1)

**作者:** Yu-Ning Qiu `[一作]` (Nanjing University), Wang-Zhou Dai `[通讯]` (Nanjing University)

**通讯引用:** 6650 | [OpenAlex ID](https://openalex.org/A5043684819)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了ABPR（Abduction-Based Procedural Refinement）框架，结合大语言模型与算法程序调试（APD）进行程序修复；

**💡 创新点**

创新点在于将Prolog元解释器生成的声明式执行轨迹与LLM推理相结合，形成可审计的归纳推理流程，实现了从“对话式修复”到“基于执行树的结构化调试”的转变；

**🔧 技术方法**

使用的技术包括大语言模型（Gemini‑3‑Flash/Pro、GPT‑5.2等）、Prolog元解释器、APD框架、归纳式推理、随机搜索与集成策略；

**📊 数据集**

实验数据集为ARC‑AGI‑2挑战集；

**📈 对比分析**

与无修复的LLM以及仅使用对话式自我修复基线对比，ABPR在Gemini‑3‑Flash上将Pass@2从34.03%提升至56.67%，整体性能超过现有最优方案，并通过消融验证执行轨迹的重要性；

**⚠️ 局限性**

局限性包括对初始程序质量高度敏感、搜索过程受随机性影响、目前仅支持Prolog等声明式语言，且对LLM“oracle”准确性的依赖仍需进一步提升。

---

## 169. Composition Theorems for Multiple Differential Privacy Constraints

**arXiv ID:** 2603.20968 | [PDF](https://arxiv.org/pdf/2603.20968v1)

**作者:** Cemre Cadir `[一作]`, Yanina Y. Shkel `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了针对双重差分隐私（双DP）约束以及异质组合的精确组合定理，并利用这些结果逼近 f‑DP 机制的组合隐私区域。

**💡 创新点**

创新点在于：①给出了满足两组（ε,δ）约束的机制的完整组合隐私区域；②将异质组合与双DP组合关联，提供了低复杂度的计算方法；③基于双DP组合构造 f‑DP 组合的上下界逼近，并在高隐私与小 k 情况下显著改进已有估计。

**🔧 技术方法**

主要技术包括：假设检验视角下的 trade‑off 函数与隐私区域；结构性混合假设检验引理；算法求解异质组合的精确隐私区域；通过双DP约束的线性组合来逼近 f‑DP，并给出解析闭式。

**📊 数据集**

本文未使用具体公开数据集，主要以理论模型和仿真图（如 Gaussian 机制、二进制随机响应）来展示结果。

**📈 对比分析**

与之前的保守上界（如 k‑fold (ε,δ)-DP 直积、基于总变差的上界）以及基于中心极限定理的近似法比较，实验表明在高隐私和小 k 时新方法提供更接近真实隐私区域的下、上界，且误差随 k 增大迅速减小。

**⚠️ 局限性**

局限性包括：目前仅处理双DP约束，尚未完整推广到任意多组 DP 约束；逼近 f‑DP 的方法依赖于 f 的可微性与对称性；实现上仍需数值求解旋转函数，可能影响计算效率。

---

## 170. Efficient AI-Driven Multi-Section Whole Slide Image Analysis for Biochemical Recurrence Prediction in Prostate Cancer

**arXiv ID:** 2603.20273 | [PDF](https://arxiv.org/pdf/2603.20273v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 171. AgenticGEO: A Self-Evolving Agentic System for Generative Engine Optimization

**arXiv ID:** 2603.20213 | [PDF](https://arxiv.org/pdf/2603.20213v1)

**作者:** Jiaqi Yuan `[一作]` (Beihang University), Jianxin Li `[通讯]` (Beihang University)

**通讯引用:** 18188 | [OpenAlex ID](https://openalex.org/A5100380470)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AgenticGEO，一个自我进化的代理系统，通过协同进化的策略存档和轻量级评估器，对生成式搜索引擎的内容进行可见性与归因优化，实现多轮规划的内容重写。

**💡 创新点**

将GEO视为内容条件化的控制问题；使用MAP‑Elites多样化存档和协同进化的轻量级评估器；通过离线对齐 + 在线协同进化降低对黑盒引擎的交互成本；引入价值‑新颖度门、兄弟‑意识优势加权回归以及Tabu列表的多轮推理规划。

**🔧 技术方法**

离线对齐的轻量级 surrogate critic（LM + value head）、MAP‑Elites 质量多样化存档、协同进化循环（Evolver + Critic）、兄弟‑意识 AWR、价值‑新颖度门、批判‑指导的多轮重写、LoRA 微调、Qwen 和 Llama 生成式引擎。

**📊 数据集**

GEO‑Bench（训练集），MS MARCO（跨域验证），E‑commerce（亚马逊商品数据）作为评估数据集；使用 Qwen2.5‑32B‑Instruct 与 Llama‑3.3‑70B‑Instruct 作为生成式引擎。

**📈 对比分析**

与14种静态与学习式基线在三大数据集上对比，AgenticGEO 在 Qwen 上整体得分 25.48（高于 AutoGEO 23.71），在 Llama 上 24.52（高于 AutoGEO 22.6）；跨域转移保持优势；仅使用 41% GE 反馈即可保留 98% 最佳性能；多轮重写 3 步最佳；语义相似度保持较高。

**⚠️ 局限性**

仍需一定量的黑盒引擎反馈来更新 critic，虽然显著减少但不完全消除成本；对极端新内容或大规模查询的适应性未知；存档规模与多样性之间需要平衡；缺乏对不同类型生成式引擎的广泛验证；对解释性与可解释性支持有限。

---

## 172. Inference Energy and Latency in AI-Mediated Education: A Learning-per-Watt Analysis of Edge and Cloud Models

**arXiv ID:** 2603.20223 | [PDF](https://arxiv.org/pdf/2603.20223v1)

**作者:** Kushal Khemani `[一作]` `[通讯]` (Billabong High International School), Kushal Khemani (Billabong High International School)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在NVIDIA T4 GPU上对Phi‑3 Mini模型进行全精度FP16与4‑bit NF4量化两种配置的即时反馈系统进行能源、延迟和教学质量的实测，并提出学习‑每瓦（LpW）度量；

**💡 创新点**

创新点包括：①提出LpW指标，将教学价值、延迟与能耗统一量化；②使用教师+前沿AI混合评审团队提供真实的教学质量评分；③揭示量化效率在有缓存与无缓存推理模式下的显著差异；

**🔧 技术方法**

技术手段主要有：4‑bit NF4量化、KV‑cache启用的推理、CodeCarbon能耗追踪、教师/AI混合评分表格与四维评分量表；

**📊 数据集**

使用的数据集为500条覆盖数学、科学、编程、人文与元认知的二次中学教育问题，每个问题产生两份答案（FP16、NF4）供评审；

**📈 对比分析**

比较方法是对同一批教育提示在相同硬件和缓存配置下测量延迟、净能耗和教学质量，并计算LpW；FP16在LpW上比NF4高1.33×，但NF4能耗低10.8%；在无缓存情况下差距扩大到7.4×；

**⚠️ 局限性**

局限性包括：仅在NVIDIA T4与Phi‑3 Mini上测试，其他GPU或模型可能表现不同；云端能耗未直接测量；评分主观性与评审间一致性有限；

---

## 173. CICDWOA: A Collective Cognitive Sharing Whale Optimization Algorithm with Cauchy Inverse Cumulative Distribution for 2D/3D Path Planning and Engineering Design Problems

**arXiv ID:** 2603.20501 | [PDF](https://arxiv.org/pdf/2603.20501v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 174. Effective Rank Analysis and Optimization of Flexible Antenna-Enabled Wireless Systems: Movable Antennas or Pinching Antennas?

**arXiv ID:** 2603.20629 | [PDF](https://arxiv.org/pdf/2603.20629v1)

**作者:** Cheng Yang `[一作]` (Macau University of Science and Technology), Dong Li `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 6420 | [OpenAlex ID](https://openalex.org/A5100407433)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出基于有效秩（effective rank）的评价指标，针对可变形天线（可移动天线MA与压缩天线PA）系统，在多时隙下通过图注意力强化学习（GRL）和多智能体图注意力Q网络（MAGRL）优化天线位置，实现天线碰撞消除并提升空间自由度。

**💡 创新点**

创新点包括：①首次将有效秩作为天线重构的客观度量；②设计了图用户分布图与波导区域图用于捕捉空间相关性；③提出GAIQN与MAGAQN两种结合图注意力、隐式分位数和顶k动作选择的算法，显著降低碰撞风险并提升收敛速度；④在MA和PA两种天线架构上统一比较有效秩表现，揭示MA在相同天线数下能获得更高有效秩但PA更稳定。

**🔧 技术方法**

技术手段主要包括：图神经网络（Graph Attention Network）用于特征嵌入；隐式分位数网络（Implicit Quantile Network）捕捉返回分布；双臂/对手架构（Dueling Network）提升学习稳定性；top‑k动作选择避免多天线冲突；优先经验回放（PER）和软目标网络实现训练稳定；多智能体合作学习（MAPPO、IPPO对比）验证算法有效性。

**📊 数据集**

数据集与实验：采用仿真数据，用户位置在 200×200 m 方形区域内均匀分布，用户数从 40 到 80、区域尺寸从 120 m 到 200 m 等多种配置；系统参数（如 λ=0.1 m、MMA=16、K_wav=8、M_pa=2、路径损耗指数 2.8 等）均按表格设定；所有实验均在 10 个时隙、T=10 的多时隙下进行。

**📈 对比分析**

与 QMIX‑ATT、QATTEN、MAPPO、IPPO、随机等基准相比，GAIQN 在 MA 系统中有效秩提升至少 1.6%（最高可达 9%），MAGAQN 在 PA 系统中提升至少 1.3%（最高可达 3.9%）。同时两算法均实现了 0 碰撞罚项，收敛速度快于对手。实验表明 MA 系统在相同天线数下能获得 66.5%–76.4% 更高有效秩，而 PA 系统在区域扩大时更具稳定性。

**⚠️ 局限性**

局限性：①研究仅在仿真环境下验证，缺乏真实硬件验证；②电激活天线实现的可行性与实际移动延迟仍需进一步研究；③有效秩仅衡量空间正交性，未覆盖功率效率、能耗等完整系统性能；④算法在大规模天线/用户规模下的计算复杂度和通信延迟尚未评估；⑤碰撞消除仅针对同一平面/波导，未考虑更复杂的三维碰撞与互斥问题。

---

## 175. Interpretable Multiple Myeloma Prognosis with Observational Medical Outcomes Partnership Data

**arXiv ID:** 2603.20341 | [PDF](https://arxiv.org/pdf/2603.20341v1)

**作者:** Salma Rachidi `[一作]` (Aalto University), Alexander Jung `[通讯]` (Aalto University)

**通讯引用:** 1419 | [OpenAlex ID](https://openalex.org/A5006624933)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文研究了在多发性骨髓瘤患者五年生存预测中，如何通过在训练过程中加入解释性正则化来提升机器学习模型的可解释性，提出了两种正则化策略并在实际临床数据上验证其效果。

**💡 创新点**

创新点在于：①将可解释的辅助模型（仅用两个临床特征的逻辑回归）作为正则化目标，直接引导复杂模型的学习；②利用已建立的国际分期系统（R‑ISS）对模型输出进行阶段一致性约束；两种正则化均能在保持性能的同时让模型遵循临床知识。

**🔧 技术方法**

技术手段包括：使用人工神经网络（ANN）配合逻辑回归损失；设计两类正则化项——KL散度辅助对齐正则化和均方误差阶段一致性正则化；采用SHAP特征重要性分析验证解释性；进行k折交叉验证与测试集评估。

**📊 数据集**

数据集为来自芬兰赫尔辛基大学医院（HUS）的812例多发性骨髓瘤患者的临床记录，采用OMOP CDM标准化，特征包括年龄、16种血液指标及骨髓浆细胞比例，标签为诊断后五年内是否死亡。

**📈 对比分析**

方法评估采用与基线模型（辅助逻辑回归、无正则化ANN、仅阶段正则化ANN）在k折交叉验证集和独立测试集上的准确率、AUC进行比较。实验结果表明：在α范围内，辅助对齐正则化的ANN在测试集上的准确率可达0.721、AUC约0.72，基本不低于无正则化模型；阶段一致性正则化在较高α时会显著降低准确率和AUC，但可强制模型关注R‑ISS关键变量。

**⚠️ 局限性**

局限性包括：①阶段一致性约束过于粗糙，限制了模型灵活性导致性能下降；②辅助模型仅使用两项特征（年龄和LDH），可能无法充分捕捉更复杂的临床关系；③实验仅在单一医院的数据上进行，缺乏跨中心验证；④正则化参数α的选择需要经验性调节，缺少自动化方法。

---

## 176. Your Robot Will Feel You Now: Empathy in Robots and Embodied Agents

**arXiv ID:** 2603.20200 | [PDF](https://arxiv.org/pdf/2603.20200v1)

**作者:** Angelica Lim `[一作]` (Simon Fraser University), Ö. Nilay Yalçin `[通讯]` (Simon Fraser University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述机器人与具身交互代理的同理心实现，并提出实现框架与关键技术

**💡 创新点**

首次将人工胼胝体与发展性学习结合，讨论同理心真实性与道德风险

**🔧 技术方法**

采用多模态感知、PAM模型、深度端到端对话、人工胼胝体集成和发展性机器人学习

**📊 数据集**

综述中引用SAL、M-Path、Kismet等实验系统的数据与公开人机交互数据集

**📈 对比分析**

对比多模态与单模态代理，结果显示具身多模态同理心提升用户偏好、参与度与情感匹配，性能优于文本基代理

**⚠️ 局限性**

存在同理心真实性不足、实时同步受限、道德与安全风险、缺乏统一评估指标和大规模实证验证

---

## 177. Agentic AI and the next intelligence explosion

**arXiv ID:** 2603.20639 | [PDF](https://arxiv.org/pdf/2603.20639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 178. Transparent Fragments Contour Estimation via Visual-Tactile Fusion for Autonomous Reassembly

**arXiv ID:** 2603.20290 | [PDF](https://arxiv.org/pdf/2603.20290v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 179. SwiftBot: A Decentralized Platform for LLM-Powered Federated Robotic Task Execution

**arXiv ID:** 2603.20233 | [PDF](https://arxiv.org/pdf/2603.20233v1)

**作者:** YueMing Zhang `[一作]` (California State University), Hailu Xu `[通讯]` (California State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了SwiftBot，一个去中心化的LLM驱动的多机器人任务执行平台，能够将自然语言指令分解为可执行子任务并通过容器化环境实现协同执行。

**💡 创新点**

创新点包括：将LLM任务分解与容器热启动池协同设计；采用分布式哈希表（DHT）实现去中心化调度与资源发现；支持跨机器人热容器迁移以降低启动延迟并实现负载均衡。

**🔧 技术方法**

使用技术包括：大语言模型（GPT-4o、Claude 3.5等）进行任务分解；Docker容器与预热池；Chord式DHT用于资源与任务元数据管理；乐观并发控制与迁移成本模型；多模型级联推理以权衡延迟与成本。

**📊 数据集**

实验使用的公开数据集为UCF101视频动作识别任务与LibriSpeech语音识别任务，用以构造多阶段容器化工作流。

**📈 对比分析**

通过与传统冷启动和本地热池基线对比，SwiftBot在任务启动延迟上平均提升5.4倍，尾部延迟提升1.2–4.7倍；在Federated学习实验中，训练时间中位数下降50–60%，容器迁移/热启动算法显著降低任务启动时间。

**⚠️ 局限性**

局限性包括：依赖容器化机器人，受限于边缘设备的内存与计算资源；LLM推理延迟与成本需要针对应用进行权衡；DHT维护在高失效率环境下会增加开销；LLM可能产生“hallucination”，影响规划可靠性；对极低时延子任务规划的支持仍有限。

---

## 180. Decoupling Numerical and Structural Parameters: An Empirical Study on Adaptive Genetic Algorithms via Deep Reinforcement Learning for the Large-Scale TSP

**arXiv ID:** 2603.20702 | [PDF](https://arxiv.org/pdf/2603.20702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 181. Engineering Pitfalls in AI Coding Tools: An Empirical Study of Bugs in Claude Code, Codex, and Gemini CLI

**arXiv ID:** 2603.20847 | [PDF](https://arxiv.org/pdf/2603.20847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 182. The Causal Impact of Tool Affordance on Safety Alignment in LLM Agents

**arXiv ID:** 2603.20320 | [PDF](https://arxiv.org/pdf/2603.20320v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 183. (Sets of ) Complement Scattered Factors

**arXiv ID:** 2603.20790 | [PDF](https://arxiv.org/pdf/2603.20790v1)

**作者:** Duncan Adamson `[一作]` (University of St Andrews), Annika Huch `[通讯]` (Kiel University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文定义了互补散列因子（complement scattered factors）概念，并给出了从给定单词 w 和散列因子 u 计算 C(w,u) 的动态规划算法，以及在逆问题（已知 w 与 C(w,u) 求 u，或已知 u 与 C(w,u) 求 w）中求解的算法。随后从组合学视角对 C(w,u) 的大小、唯一性及其与完美交错（perfect shuffle）等运算的关系进行深入分析。

**💡 创新点**

创新点在于首次将散列因子与交错（shuffle）运算结合，引入互补散列因子概念；提出了在完美交错情况下互补散列因子唯一性的判定；并给出了逆问题的算法框架，扩展了对散列因子信息缺失与重构的理论研究。

**🔧 技术方法**

技术主要包括：动态规划（DP）求解 C(w,u)；多重计数扩展 DP；基于 arch 分解、α‑β 分解以及 Lyndon–Schützenberger 定理的组合工具进行理论分析；以及多维 DP 处理词嵌入的互斥性。

**📊 数据集**

本文为理论工作，未使用实验数据集；所有实验讨论均基于符号字母表 Σ 的大小 σ 与单词长度 |w|、|u| 等参数。

**📈 对比分析**

没有进行实验比较；所有结果以理论证明和算法复杂度估计呈现。主要复杂度包括：计算 C(w,u) 的 O(|w|·|u|·wu)，逆问题（求 u 或 w）分别为 O(n^2·n·m) 与 O(r·n^r) 等。

**⚠️ 局限性**

限制与开放问题：算法对嵌入数量 wu 过大时效率低下；仅在 |u| 小于 universality index 时能给出完整性质；未给出 C(w,u) 的上界或闭式公式；对长度超过 universality index 的散列因子情况仍是未解决的开放问题。

---

## 184. MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction

**arXiv ID:** 2603.20782 | [PDF](https://arxiv.org/pdf/2603.20782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 185. PEARL: Personalized Streaming Video Understanding Model

**arXiv ID:** 2603.20422 | [PDF](https://arxiv.org/pdf/2603.20422v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 186. Prompt-Free Lightweight SAM Adaptation for Histopathology Nuclei Segmentation with Strong Cross-Dataset Generalization

**arXiv ID:** 2603.20326 | [PDF](https://arxiv.org/pdf/2603.20326v1)

**作者:** Muhammad Hassan Maqsood `[一作]` (Griffith University), Alan Wee-Chung Liew `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种轻量、无提示的 SAM 适配方法，通过在冻结的 SAM ViT-B 编码器中插入 LoRA 模块并使用残差解码器，实现对病理核的高精度分割。

**💡 创新点**

创新点在于仅微调 4.1M 的 LoRA 参数保持模型轻量化；采用多级 Transformer 特征融合加残差解码提升边界细节；使用偏置先验初始化加速收敛；完全去除提示，直接预测掩码。

**🔧 技术方法**

技术包括 SAM ViT-B 编码器、LoRA 参数高效微调、残差解码块、多尺度特征融合、偏置先验初始化、焦点 Tversky 损失、AdamW 优化器与 ReduceLROnPlateau 学习率调度。

**📊 数据集**

使用 TNBC、MoNuSeg、PanNuke 三个公开核分割基准数据集进行训练与评估。

**📈 对比分析**

与 CNN（U-Net）、Transformer（Swin-Unet、UCTransNet、TSCA-Net）、SAM 及其变体（SAM+LoRA、Trans-SAM）在 Dice 与 IoU 上进行比较，模型在三大数据集上均取得最高或第二高成绩，同时保持最小的可训练参数量，并在跨数据集迁移实验中表现出色。

**⚠️ 局限性**

局限性包括仅在 512×512 patch 级别验证，缺乏对更高分辨率或不同显微镜设备的评估；在极稀疏或极密集细胞场景下仍可能需要进一步细化；缺乏对更广泛数据增强策略和模型鲁棒性的系统研究。

---

## 187. immUNITY: Detecting and Mitigating Low Volume & Slow Attacks with Programmable Switches and SmartNICs

**arXiv ID:** 2603.20573 | [PDF](https://arxiv.org/pdf/2603.20573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 188. Causally-Guided Diffusion for Stable Feature Selection

**arXiv ID:** 2603.20930 | [PDF](https://arxiv.org/pdf/2603.20930v1)

**作者:** Arun Vignesh Malarkkan `[一作]` (Arizona State University), Yanjie Fu `[通讯]` (Arizona State University)

**通讯引用:** 6277 | [OpenAlex ID](https://openalex.org/A5032187620)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于概率后验推断的稳定特征选择框架 CGDFS，利用扩散模型学习特征子集先验，并通过稳定性感知似然与引导拉普拉斯动力学进行采样，从而在分布偏移下挑选具有高预测性能与跨环境稳定性的特征子集。

**💡 创新点**

创新点包括：①把特征选择视为后验推断，捕捉子集空间的不确定性；②使用扩散模型学习结构化先验，避免离散搜索；③引入稳定性感知似然，兼顾均值误差与方差；④用ICP的软因果引导增强先验，提升稳健性；⑤通过后验聚合得到可解释的特征频率。

**🔧 技术方法**

技术手段：score‑based diffusion模型、引导扩散采样（Annealed Langevin Dynamics）、稳定性感知似然、ICP软引导、环境特定预测器训练、后验抽样与Top‑k阈值化、频率聚合。

**📊 数据集**

实验数据集：12个表格数据集（UCI Credit、Activity、OpenML‑618、OpenML‑637、Pima Indian、Boston Housing 等），涵盖分类与回归任务，并通过人为构造的多环境划分模拟分布偏移。

**📈 对比分析**

与基线对比：LASSO/Elastic Net、互信息、随机森林重要性、稳定性选择、贪婪前向、梯度优化稳定目标、ICP 等。CGDFS 在所有分类数据集上获得最高 F1‑macro，在回归数据集上获得最低 MSE；同时在跨环境方差上保持或优于对照组，表明其更稳健、更具可迁移性。

**⚠️ 局限性**

局限性：①缺乏严格的后验采样收敛理论；②对扩散先验训练质量高度依赖，训练成本较高；③计算复杂度随特征维度、环境数线性增长，难以扩展到极高维稀疏场景；④需要已知的多环境标签，适用性受限；⑤对超参数和采样步数敏感，需细致调优。

---

## 189. An Industrial-Scale Retrieval-Augmented Generation Framework for Requirements Engineering: Empirical Evaluation with Automotive Manufacturing Data

**arXiv ID:** 2603.20534 | [PDF](https://arxiv.org/pdf/2603.20534v1)

**作者:** Muhammad Khalid `[一作]` (Constructor University Bremen), Yilmaz Uygun `[通讯]` (Constructor University Bremen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对工业需求工程进行基于检索增强生成（RAG）的自动化评估，使用真实汽车制造文档。

**💡 创新点**

提出端到端工业级RAG框架，实现98.2%提取准确率、Hybrid检索+多供应商LLM成本优化、可追溯性完整性，并在6个月生产部署验证。

**🔧 技术方法**

Hybrid semantic-lexical检索、域适配句子变换器、跨编码reranker、XGBoost路由、多供应商LLM（GPT‑3.5/4/Claude），Traceability元数据。

**📊 数据集**

669条要求、4种规范（MBN 9666‑1/2/5/9）、49供应商资格、127页合规矩阵，覆盖2015‑2023年八年。

**📈 对比分析**

与BERT、无检索LLM、纯稀疏/密集检索基线对比，RAG在提取准确率提升24.4%/19.6%，Hybrid检索MRR 0.847，比BM25提升24%；效率提升83%，成本降低47%。

**⚠️ 局限性**

仅在汽车制造领域验证，其他行业需要进一步验证；仍需人机交互处理极端歧义；系统需定期再训练；未覆盖全流程的实时学习。

---

## 190. Collaborative Adaptive Curriculum for Progressive Knowledge Distillation

**arXiv ID:** 2603.20296 | [PDF](https://arxiv.org/pdf/2603.20296v1)

**作者:** Jing Liu `[一作]` (Fudan University), Liang Song `[通讯]` (Fudan University)

**通讯引用:** 15876 | [OpenAlex ID](https://openalex.org/A5034582366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于共识驱动的渐进式知识蒸馏框架，在联邦学习环境下实现自适应的知识迁移。

**💡 创新点**

创新点在于：①使用 PCA 构建教师特征的分层知识层级；②通过服务器监控全局准确率稳定性来动态调整蒸馏维度；③在客户端使用投影矩阵进行逐步蒸馏，并结合交叉熵、KL 散度和 InfoNCE 对齐。

**🔧 技术方法**

核心技术包括联邦知识蒸馏、PCA 层级化知识分解、共识驱动的课程调度、投影矩阵、KL 散度与对比学习损失。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 和 Tiny‑ImageNet 三个基准数据集上进行评估。

**📈 对比分析**

与 FedAvg、FedProx、MOON、FedNH、FedRCL、FedCDA 等基线对比，FAPD 在 CIFAR‑10 上提升至 89.42%（比 FedAvg 高 3.64%），收敛速度约为基线的两倍，并在 α=0.1 的极端异构场景下仍保持高性能。

**⚠️ 局限性**

局限性包括：①PCA 层级依赖于校准数据，若代表性不足会影响知识排序；②统一的课程难度可能不适用于极端资源差异的客户端；③目前仅针对图像分类任务，需进一步扩展至视频、音频等多模态。

---

## 191. Thermal is Always Wild: Characterizing and Addressing Challenges in Thermal-Only Novel View Synthesis

**arXiv ID:** 2603.20448 | [PDF](https://arxiv.org/pdf/2603.20448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 192. Error-detecting solid codes

**arXiv ID:** 2603.20298 | [PDF](https://arxiv.org/pdf/2603.20298v1)

**作者:** Nathan Thomas Carruth `[一作]` `[通讯]` (SUSTech), Nathan Thomas Carruth (SUSTech)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了一种可扩展到任意n-ary字母表的可变长度固体码（solid code）构造，并证明了该构造在特定噪声模型下具有错误检测特性；随后给出了对二进制码的一个具体应用示例。

**💡 创新点**

创新点在于：①将之前仅针对二进制的固体码构造推广到任意符号集；②利用签名（signature）映射构造固体码；③证明在“每个符号只可能被改为0且仅在签名不同时才允许错误”的噪声模型下，签名不变从而实现错误检测；④在二进制场景下通过奇偶性分区实现单比特错误的完整检测。

**🔧 技术方法**

技术手段主要包括：符号签名映射、固体码定义的前缀-后缀不重叠性质、对签名序列的递归分析，以及在二进制码中利用奇偶性分区和分块长度来构造固体码。

**📊 数据集**

本文属于理论研究，未使用实际数据集；所有结果均为数学证明和构造示例。

**📈 对比分析**

由于论文未给出实验或数值比较，性能评估仅在理论层面完成：在所设定的噪声模型下，任何传输字符串的签名保持不变；若进一步满足签名相等的条件，则可完全恢复原始字符串。相比传统固体码，只增加了对“错误位置已知且只能改为0”这一特定噪声模式的检测能力。

**⚠️ 局限性**

局限性包括：①构造的固体码在被视为二进制码时可能不再固体；②错误检测仅在“符号改为0且签名不变”的特定噪声模型下有效，不能保证对任意错误模式；③对多比特错误或更复杂的噪声模型缺乏保障；④缺乏实际实验验证，理论证明尚未在真实通信环境中测试。

---

## 193. Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges

**arXiv ID:** 2603.20304 | [PDF](https://arxiv.org/pdf/2603.20304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 194. Leum-VL Technical Report

**arXiv ID:** 2603.20354 | [PDF](https://arxiv.org/pdf/2603.20354v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 195. Jitter Performance Evaluation for Resilient Vehicular Systems

**arXiv ID:** 2603.20291 | [PDF](https://arxiv.org/pdf/2603.20291v1)

**作者:** Pratiti Paul `[一作]` (Linköping University), Manav R. Bhatnagar `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 6384 | [OpenAlex ID](https://openalex.org/A5035838103)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一个基于随机过程的 V2V 通信链路抖动建模与评估框架，构建了抖动不耐受概率、极限状态指标以及平均风险暴露率等韧性度量，并给出了基于机会约束的自适应功率与 MISO 资源分配方案，实现了系统在遭受移动与干扰诱发的抖动时的恢复与鲁棒性提升。

**💡 创新点**

创新点在于：①首次将抖动不耐受概率与极限状态指标结合，用概率方式刻画 V2V 链路随时间演化的韧性；②提出了基于抖动不耐受概率的平均风险暴露率和抖动耐受容量度量，系统性地描述警戒、失效与恢复阶段；③设计了机会约束下的自适应功率与天线数量优化策略，使资源利用与抖动风险呈动态平衡；④通过数值模拟验证，证明自适应资源分配相较于恒定分配可提升约 3 倍的风险暴露率。

**🔧 技术方法**

所用技术包括：随机过程建模（泊松到达、布朗运动扩散、Itô 方程）、极限状态方法、概率敏感度与 Delta 方法、机会约束优化（KKT 条件求解）、Monte‑Carlo 仿真、MATLAB 数值仿真与统计分析。

**📊 数据集**

本文主要使用了基于系统参数（如发射功率、路径损耗指数、车距、干扰到达率等）的合成仿真数据，没有引用公开数据集；所有结果均通过 10⁶ 次 Monte‑Carlo 实验得到。

**📈 对比分析**

在对比方法上，作者将自适应资源分配与固定功率/天线配置进行对照，并在警戒、失效与恢复阶段分别绘制抖动耐受容量、抖动负载与极限状态指标的演化曲线；结果表明自适应方案能显著延缓极限状态指标降至零的时间，并在恢复阶段将系统状态恢复到安全阈值，平均风险暴露率提升约 3.17 倍。

**⚠️ 局限性**

局限性包括：仅考虑单链路的 V2V 通信，未扩展至多链路或 MIMO 场景；使用 Rayleigh 衰落模型对 LOS 高速场景较保守，未考虑 Rician 或更真实的多径；干扰建模为纯生长过程，未考虑车辆离开或随机停靠；实验基于仿真，缺乏现场验证；机会约束优化假设分布已知，实际环境中可能存在模型误差。

---

## 196. GHOST: Ground-projected Hypotheses from Observed Structure-from-Motion Trajectories

**arXiv ID:** 2603.20583 | [PDF](https://arxiv.org/pdf/2603.20583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 197. High-Speed, All-Terrain Autonomy: Ensuring Safety at the Limits of Mobility

**arXiv ID:** 2603.20525 | [PDF](https://arxiv.org/pdf/2603.20525v1)

**作者:** James R. Baxter `[一作]` (University of Michigan), Tulga Ersal `[通讯]` (University of Michigan)

**通讯引用:** 2725 | [OpenAlex ID](https://openalex.org/A5076004234)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于模型预测控制的单刚体动力学模型和能量稳定性边界（ESM）约束，实现在高速崎岖地形下的全能地形车辆安全局部轨迹规划。

**💡 创新点**

创新点在于引入ESM作为滚翻预防约束，并设计可实时计算的单刚体模型，克服平面模型无法预测垂直与悬挂动态导致的滚翻问题。

**🔧 技术方法**

技术手段包括模型预测控制、能量稳定性约束、采样式优化（empirical‑argmin）、GPGPU并行计算、软约束正则化以及三维地形重建与高保真仿真。

**📊 数据集**

使用的数据集为MTRI Inc.基于无人机摄影测量得到的Bundy Hill Off‑Road Park三维重建地形，以及车辆动力学与摩擦系数的实测数据。

**📈 对比分析**

通过9900个仿真闭环试验和48个实车试验，将基线（平面单轨+侧向加速度约束）与提出的方法（单刚体+ESM）对比，结果显示后者在滚翻比例降低、成功率提升、成本降低方面均优于基线。

**⚠️ 局限性**

局限性包括未考虑不确定性与随机扰动、未对速度控制进行优化、采样式优化不使用温启动或加权平均，以及在有限地形与传感精度条件下验证，需进一步扩展至更具挑战的环境与鲁棒性研究。

---

## 198. Solomonoff induction

**arXiv ID:** 2603.20274 | [PDF](https://arxiv.org/pdf/2603.20274v1)

**作者:** Tom F. Sterkenburg `[一作]` `[通讯]` (LMU Munich), Tom F. Sterkenburg (LMU Munich)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本章讨论了Solomonoff归纳法的可计算性基础，分析了可计算性与可实现性之间的冲突，并对Solomonoff半可计算半测度的理论意义与局限进行了批判性评估。

**💡 创新点**

创新点在于重新审视了Putnam的对角化论证，阐明了可计算预测器与半可计算半测度之间的关系，揭示了即使存在“全局可靠”但不可实现的预测器，以及对Occam原则的更为细致的哲学反思。

**🔧 技术方法**

采用可计算性理论、贝叶斯混合/聚合预测器、对角化论证、Kolmogorov复杂度与可计算测度的理论工具。

**📊 数据集**

无实验数据集，本文完全基于形式化的理论分析与哲学讨论。

**📈 对比分析**

未进行实验比较；理论上相较于传统可计算预测器，Solomonoff半可计算半测度提供了对所有可计算数据生成过程的“全局可靠性”，但在可实现性上存在不可忽视的缺陷。

**⚠️ 局限性**

主要局限包括：不可计算性导致实际实现困难；对角化导致的方差与选择不确定性；对Occam原则的解释缺乏可操作性；以及对理论理想与实际机器学习方法之间的桥接不足。

---

## 199. REVERE: Reflective Evolving Research Engineer for Scientific Workflows

**arXiv ID:** 2603.20667 | [PDF](https://arxiv.org/pdf/2603.20667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 200. Reasoning Topology Matters: Network-of-Thought for Complex Reasoning Tasks

**arXiv ID:** 2603.20730 | [PDF](https://arxiv.org/pdf/2603.20730v1)

**作者:** Fan Huang `[一作]` (Indiana University Bloomington), Fan Huang `[通讯]` (Indiana University Bloomington)

**通讯引用:** 7113 | [OpenAlex ID](https://openalex.org/A5050372992)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Network-of-Thought（NoT）框架，将 LLM 推理建模为有向图，并与 Chain-of-Thought（CoT）与 Tree-of-Thought（ToT）在四个基准上进行系统比较。

**💡 创新点**

创新点在于：① 用 LLM 生成并更新图结构；② 引入可自生成的启发式控制策略；③ 通过图拓扑解决多源依赖、循环与重用三大局限；④ 对比不同拓扑的计算-准确率折衷。

**🔧 技术方法**

技术细节包括：多次 LLM 调用构造与扩展图；控制器基于不确定性、依赖度和冲突权重评分；LLM-as-Judge 进行语义评估；并行节点扩展实验；并使用 token 效率与图密度等指标评估。

**📊 数据集**

使用的四个数据集为：GSM8K（算术）、Game of 24（组合搜索）、HotpotQA（多跳 QA）和 ProofWriter（逻辑推理）。

**📈 对比分析**

与 CoT、ToT 在同一任务、同一模型下对比，评估维度为准确率、图复杂度与 token 效率；在多跳 QA 上 NoT 以 Judge 评估达到 91%，在 72B 开源模型上 GSM8K 达到 91.5%，在 Game of 24 取得 86%，在 ProofWriter 通过自生成启发式实现 54% 的准确率，整体显示 NoT 在需要多源整合的任务中优于树拓扑，链拓扑在单步推理任务中更有效。

**⚠️ 局限性**

局限性包括：① 评估依赖 LLM-as-Judge 的主观性；② 当前框架无法自动产生循环推理结构；③ 对小模型的性能下降明显；④ 控制器权重仍需手工或单次 LLM 调用调优，可能无法捕捉更复杂任务的动态需求；⑤ 仅在四个基准上验证，缺乏对更大规模、更多领域任务的泛化测试。

---

## 201. JUBAKU: An Adversarial Benchmark for Exposing Culturally Grounded Stereotypes in Japanese LLMs

**arXiv ID:** 2603.20581 | [PDF](https://arxiv.org/pdf/2603.20581v1)

**作者:** Taihei Shiotani `[一作]` (Institute of Science Tokyo), Naoaki Okazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 3544 | [OpenAlex ID](https://openalex.org/A5066940046)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一套专门针对日语文化的对抗式社会偏见基准（JUBAKU），并在其中测试了九款日语大型语言模型。

**💡 创新点**

创新点在于：①基于日语本土文化场景手工构造10个文化类别的对话实例；②采用对抗式编辑方法（利用GPT‑4o迭代挑选偏见回答）使实例更能诱发模型隐性偏见；③将对抗实例与传统基准对比，揭示模型在非西方文化语境下的隐藏偏见。

**🔧 技术方法**

使用的技术包括：对抗式提示生成与编辑（GPT‑4o）、手工注释、对话二选一评估框架（通过对两条候选回答的log‑likelihood比较确定答案）、统计显著性检验（McNemar检验）等。

**📊 数据集**

数据集为：①JUBAKU（1216条实例，152条基线 ×4任务指令 ×2答案顺序），②现有日语偏见基准（JBNLI、JBBQ、SSQA‑JA）。

**📈 对比分析**

比较方法：将所有实例统一为二选一“选择无偏见回答”的任务，使用准确率作为度量；随机基准为50%，人工标注准确率91%。实验显示：在JUBAKU上，九个模型平均准确率仅为13%–33%（均低于随机），而在其他基准上的准确率则在50%–80%之间，证明JUBAKU更能揭示模型在日语文化中的隐性偏见。

**⚠️ 局限性**

局限性：①仅覆盖10个文化类别，未涉及年龄、职业、性取向等维度；②评估仅关注“安全性”即偏见回避，未衡量回答的实用性和质量；③人工构造与标注受主观解释影响，缺乏多注释者一致性验证。

---

## 202. Diffutron: A Masked Diffusion Language Model for Turkish Language

**arXiv ID:** 2603.20466 | [PDF](https://arxiv.org/pdf/2603.20466v1)

**作者:** Şuayp Talha Kocabay `[一作]`, Talha Rüzgar Akkuş `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对土耳其语的Masked Diffusion Language Model Diffutron，并通过LoRA持续预训练和逐步指令调优实现了高质量的非自回归文本生成。

**💡 创新点**

创新点在于：1) 首次将MDLM应用于形态丰富的黏着语土耳其语；2) 采用多阶段训练策略（LoRA持续预训练 + 两阶段指令调优），在保持模型小巧（307M参数）的同时，获得与多倍大规模自回归模型相当甚至更优的表现。

**🔧 技术方法**

技术手段包括：LoRA适配、jhu-clsp/mmBERT-base多语言编码器、逐步指令调优（metunlp/LlamaTurk-Instruction-Set + turkish-nlp-suite/InstrucTurca）、Diffusion‑LM反向去噪过程。

**📊 数据集**

使用的数据集有：Havadis、Temiz‑OSCAR、Turkish Wikipedia用于持续预训练；metunlp/LlamaTurk-Instruction-Set 和 turkish‑nlp‑suite/InstrucTurca用于指令调优；Bilkent Turkish Writings Dataset用于评估困惑度；CETVEL Benchmark子集（Belebele_TR、EXAMS_TR、IronyTR、News Category Classification、MNLI_TR、STS_TR、XCOPA_TR）用于下游任务评估。

**📈 对比分析**

评估方法是将Diffutron在CETVEL子集上的平均得分与多种基准（1.1B TURNA、2B Kumru、3B Llama-3.2、7B Trendyol-LLM等）进行对比。结果显示，307M参数的Diffutron 2nd‑Stage平均分为34.68，显著超过2B规模的Kumru（34.09）和TURNA（33.19），证明了MDLM与多阶段调优的高效性。

**⚠️ 局限性**

局限性包括：1) 仍依赖多语言预训练基座，缺乏专用土耳其语基础模型；2) 指令数据有限，难以捕获深层文化与语言细节；3) 256标记的上下文窗口限制了长文本生成与摘要能力；4) 评测仅覆盖CETVEL子集，未能全面检验模型在所有土耳其语任务上的性能。

---

## 203. Evolutionary Dynamics of Variable Games in Structured Populations

**arXiv ID:** 2603.20603 | [PDF](https://arxiv.org/pdf/2603.20603v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 204. Enhancing Safety of Large Language Models via Embedding Space Separation

**arXiv ID:** 2603.20206 | [PDF](https://arxiv.org/pdf/2603.20206v1)

**作者:** Xu Zhao `[一作]` (Renmin University of China), Weiran Shen `[通讯]` (Renmin University of China)

**通讯引用:** 904 | [OpenAlex ID](https://openalex.org/A5059901703)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于嵌入空间分离（ES2）的安全微调框架，通过在关键层显式增大有害与无害提示的嵌入距离，以提升大型语言模型的防御能力。

**💡 创新点**

创新点在于：①将嵌入空间线性可分性视作防御资源，而非脆弱点；②引入距离最大化损失与KL正则化相结合的双重目标，既扩大安全边界又保留通用能力；③采用分层目标训练，仅针对信息显著层进行微调，避免语义崩溃。

**🔧 技术方法**

技术上采用：LoRA参数高效微调；距离最大化损失计算平均欧氏距离；KL散度正则化与原基模型输出对齐；多目标损失加权；训练时设置KL阈值实现硬约束。

**📊 数据集**

数据集主要使用安全标注数据集 𝒟 = {(q_i, c_i)}，其中 q_i 为提示，c_i 为是否有害标签；实验评估使用四款开源 LLM（LLaMA‑2‑7B‑Chat‑hf、LLaMA‑3‑8B‑Instruct、Mistral‑7B‑Instruct、Qwen‑2.5‑7B‑Instruct）及其公开安全评测基准（RepE、Soft Prompt、SCAV等嵌入攻击）和 Open LLM Leaderboard 任务集。

**📈 对比分析**

与两类主流安全对齐方法（安全微调 STL 与 RLHF DPL）对比，ES2 在所有模型与攻击类型下均实现最高的防御成功率（DSR），尤其在 SCAV 攻击中提升约 15–30%；同时在 Open LLM Leaderboard 上平均准确率与基线相当甚至略优，证明安全提升不牺牲通用能力。

**⚠️ 局限性**

局限性包括：①仅针对嵌入层攻击进行评估，对 Prompt‑level 或更高级的攻击仍需进一步验证；②距离最大化损失需手动选择关键层，可能对不同模型适用性不统一；③在极端攻击下仍可能出现语义崩溃，导致输出失真；④KL正则阈值设置对模型稳定性敏感，需经验调优。

---

## 205. NCSTR: Node-Centric Decoupled Spatio-Temporal Reasoning for Video-based Human Pose Estimation

**arXiv ID:** 2603.20323 | [PDF](https://arxiv.org/pdf/2603.20323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 206. RayMap3R: Inference-Time RayMap for Dynamic 3D Reconstruction

**arXiv ID:** 2603.20588 | [PDF](https://arxiv.org/pdf/2603.20588v1)

**作者:** Feiran Wang `[一作]` (University Of Illinois Chicago), Yan Yan `[通讯]` (University Of Illinois Chicago)

**通讯引用:** 19785 | [OpenAlex ID](https://openalex.org/A5100395068)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种训练无监督的实时动态场景3D重建框架RayMap3R，利用RayMap的静态偏置在推理时识别并抑制动态区域，从而提升几何精度和相机姿态估计。

**💡 创新点**

创新点包括：1）在推理时通过双分支（主分支+RayMap分支）利用RayMap预测的静态偏置实现无监督动态区域识别；2）引入重置度量对齐与状态感知平滑机制，保证尺度一致性和轨迹平稳；3）在流式重建框架中实现对动态场景的实时高精度重建。

**🔧 技术方法**

采用的核心技术包括：RayMap（每像素光线表征）、隐式记忆缓存、双分支推理（主分支+RayMap分支）、动态识别权重门控内存更新、重置时的Sim(3)对齐、状态感知加权平滑。

**📊 数据集**

实验使用了 Sintel、DAVIS 2017、TUM RGB‑D、TUM‑Dynamics、KITTI、ScanNet、7‑Scenes、Bonn 等多种动态与静态、合成与真实场景的数据集。

**📈 对比分析**

在相机位姿、视频深度和3D重建三个任务上，与 Spann3R、CUT3R、Point3R、StreamVGGT、TTT3R 等流式方法以及部分离线方法对比，RayMap3R 在动态场景下均取得了最优或近最优的指标，并在实时性与常数内存占用方面保持了优势。

**⚠️ 局限性**

局限性在于：1）依赖模型在训练阶段已形成足够的静态偏置，若训练数据缺乏动态内容，动态识别效果可能受限；2）对极端快速或复杂动态运动的检测仍不够精准；3）虽然采用平滑与对齐机制，但在长序列中仍可能出现累计误差。

---

## 207. BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs

**arXiv ID:** 2603.20309 | [PDF](https://arxiv.org/pdf/2603.20309v1)

**作者:** Duyi Pan `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 40528 | [OpenAlex ID](https://openalex.org/A5100391662)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了BubbleRAG，一种针对无结构知识图的检索增强生成框架，解决检索时的召回和精度双重挑战。

**💡 创新点**

核心创新包括将检索任务正式化为Optimal Informative Subgraph Retrieval（OISR）问题，并基于此设计了语义锚点分组、泡沫式扩展、复合排名以及推理感知扩展等训练无关、可即插即用的检索流程。

**🔧 技术方法**

技术手段主要有：LLM驱动的语义锚点分组与架构放宽、基于成本引导的泡沫扩展搜索、语义一致性与结构完整性复合评分、以及LLM引导的多跳扩展与答案生成。

**📊 数据集**

实验使用了三大多跳问答基准数据集：HotpotQA、MuSiQue和2WikiMultiHopQA。

**📈 对比分析**

在多跳QA评估中，BubbleRAG在30B与8B两种模型下相较于NaiveRAG、HippoRAG2、ToG、ClueRAG、LightRAG等十余种基线均取得最高的F1与LLM‑as‑Judge准确率，尤其在MuSiQue上提升约8个百分点F1，整体优于最强基线约2.5%准确率。

**⚠️ 局限性**

主要局限包括：对LLM推理的依赖导致推理延迟与Token消耗增加；泡沫扩展为启发式算法，无法保证全局最优；以及对超参数（如预算B、深度d、完整性惩罚α）敏感，需在实际部署前进行调优。

---

## 208. On the Fragility of AI Agent Collusion

**arXiv ID:** 2603.20281 | [PDF](https://arxiv.org/pdf/2603.20281v1)

**作者:** Jussi Keppo `[一作]` (National University of Singapore), Nuo Yuan `[通讯]` (Boston University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建重复价格博弈实验，结合理论框架，系统检验大型语言模型（LLM）代理在不同异质性条件下的价格合谋脆弱性；

**💡 创新点**

创新点在于首次实证揭示：耐心、数据访问、算法类型及参与者数量等异质性显著削弱或破坏LLM驱动的默示合谋，而模型规模差异则不致合谋；同时验证了“反合谋”提示的有效性；

**🔧 技术方法**

采用开放源代码的DeepSeek-R1‑Distill‑Qwen‑32B LLM作为定价代理，并与传统Q‑learning代理、不同规模模型、不同信息访问设定进行对比；使用多轮Bertrand博弈、基于提示的多代理推理与链式推理技术；

**📊 数据集**

实验使用人工合成的多项式对数需求函数（MNL）及内部利润计算，未使用公开实测数据集；所有数据均来自仿真环境；

**📈 对比分析**

通过比较价格收敛时间、价格与静态Nash均衡的相对提升率进行定量评估；结果显示，完全同质且耐心的LLM可实现约22%价格上扬，异质性显著降低至10%或7%；算法异质性几乎消除合谋；模型规模异质性保持高价位但收敛更慢；

**⚠️ 局限性**

局限性包括：每个实验条件仅重复10次，计算成本高；仅考虑了有限的异质性维度，未涵盖真实部署的提示、训练数据、温度等因素；缺乏对LLM内部机制的解释；实验基于仿真，尚需实地验证。

---

## 209. Fusing Driver Perceived and Physical Risk for Safety Critical Scenario Screening in Autonomous Driving

**arXiv ID:** 2603.20232 | [PDF](https://arxiv.org/pdf/2603.20232v1)

**作者:** Chen Xiong `[一作]` (Sun Yat-sen University), Chao Gou `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3594 | [OpenAlex ID](https://openalex.org/A5042895349)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于驾驶员风险融合的自动驾驶安全关键场景筛选方法，利用改进的驾驶员风险场与动态成本模型生成高质量风险监督信号，并通过风险–轨迹交叉注意力解码器实现风险与轨迹的联合预测，从而在不进行逐帧计算的前提下快速得到场景级风险评分。

**💡 创新点**

创新点包括：①将驾驶员风险场与动态成本模型融合，构建更具物理可解释性和认知一致性的综合风险函数；②在解码器中引入“隐式输入–显式输出”风格的风险–轨迹交叉注意力机制，使轨迹分支主动关注风险上下文；③采用多任务学习与Winner‑Takes‑All训练，既保持轨迹多模态生成，又提升风险预测的敏感性；④实现了从原始轨迹数据到场景风险评分的全自动、高速推理流水线。

**🔧 技术方法**

使用技术包括：改进的驾驶员风险场（带速度自适应视角与高度函数）、动态成本模型（动能、OBB约束与高斯扩散）、两阶段图神经网络编码器、跨注意力解码器、MLP输出头、smooth L1损失、AdamW优化器和余弦退火学习率调度。

**📊 数据集**

使用数据集：INTERACTION（训练集）和 FLUID（测试集），分别代表一般交通场景与高冲突密度的交叉路口场景。

**📈 对比分析**

与 PET、THW、DRAC、PODAR 等传统代理安全度量及 TTC 进行对比，评估指标为 AUC、AP 和 Precision@K。实验结果显示：AUC 0.792、AP 0.825、P@100 0.95、P@1000 0.799，显著优于所有基线方法，证明在高风险样本检索与排序方面具有更高的准确性与效率。

**⚠️ 局限性**

局限性：①风险场模型仍基于经验式假设，可能在极端复杂或未知交互中出现误估；②对训练数据的依赖较大，若缺乏足够多样化的危险样本，模型的泛化能力可能受限；③虽然推理速度快，但在极大规模实时系统中仍需进一步优化内存占用与并行度；④方法主要关注前向轨迹，尚未针对后向或非线性复杂交通事件进行深入验证。

---

## 210. Beyond the Academic Monoculture: A Unified Framework and Industrial Perspective for Attributed Graph Clustering

**arXiv ID:** 2603.20829 | [PDF](https://arxiv.org/pdf/2603.20829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 211. Low-pass Personalized Subgraph Federated Recommendation

**arXiv ID:** 2603.20338 | [PDF](https://arxiv.org/pdf/2603.20338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 212. When Agents Disagree: The Selection Bottleneck in Multi-Agent LLM Pipelines

**arXiv ID:** 2603.20324 | [PDF](https://arxiv.org/pdf/2603.20324v1)

**作者:** Artem Maryanskyy `[一作]` `[通讯]` (Uber), Artem Maryanskyy (Uber)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多代理LLM管线中团队多样性与聚合方式的交互影响。

**💡 创新点**

提出了选择瓶颈模型和交叉阈值 s*，统一了前人相互矛盾的结论。

**🔧 技术方法**

采用了生成‑先选、判断者选择、投票和 Mixture‑of‑Agents 合成等聚合技术。

**📊 数据集**

在 42 个任务（涵盖编码、创造、伦理、数学、推理、科学、摘要）共 210 份实验数据。

**📈 对比分析**

通过 Bradley‑Terry 校正的赢率比较，发现判断者选择在多样化团队中显著优于合成（g≈3.86），而投票效果与单模型相当。

**⚠️ 局限性**

受限于模型范围、LLM‑as‑judge 的偏差、单轮生成‑先选结构、以及未能测量可区分性等。

---

## 213. PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization

**arXiv ID:** 2603.20778 | [PDF](https://arxiv.org/pdf/2603.20778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 214. BenchBench: Benchmarking Automated Benchmark Generation

**arXiv ID:** 2603.20807 | [PDF](https://arxiv.org/pdf/2603.20807v1)

**作者:** Yandan Zheng `[一作]` (Nanyang Technological University), Luu Anh Tuan `[通讯]` (Nanyang Technological University)

**通讯引用:** 1127 | [OpenAlex ID](https://openalex.org/A5001659855)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BenchBench框架，系统评估LLM在自动生成基准测试（benchmark）方面的能力，构建了三阶段管道并生成了多域、多语言、多模态的benchmark数据集

**💡 创新点**

将benchmark生成视为可度量的元能力，首次引入设计者-回答者矩阵、有效性/诊断性、格式/语言/模态合规性、排名保持及自/家族偏差等多维度评估指标

**🔧 技术方法**

利用LLM做设计者与回答者、自动化的域卡抽取、配额控制生成、目标分值验证、客观匹配与rubric‑guided判断、统计分析与心理测量指标

**📊 数据集**

基于四大种子基准（CSBench、WeMath、MedXpertQA、ToMBench）构建九个变体（中英、文本/多模态），共生成约1.6万条题目，约1.5万条核心题目及1.7万条评测结果

**📈 对比分析**

通过设计者-回答者矩阵计算有效率、区分度、排名保持（Kendall τ）和自/家族优势等指标；结果显示设计者有效率与区分度呈负相关，设计能力与回答强度仅弱相关，生成的benchmark在不同域/语言/模态下表现差异显著

**⚠️ 局限性**

局限性包括：受种子基准偏倚限制、验证者依赖有限、回答者面板规模有限、只覆盖四大领域、缺乏人类审查与更广泛的多模态/语言验证

---

## 215. The production of meaning in the processing of natural language

**arXiv ID:** 2603.20381 | [PDF](https://arxiv.org/pdf/2603.20381v1)

**作者:** Christopher J. Agostino `[一作]` (NPC Worldwide), Louis van der Elst `[通讯]` (Imperial College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在数十个大型语言模型上实施了基于 Bell 检验的 CHSH |S| 测试，评估模型在不同采样参数下的情境性，并探讨其与现有基准（MMLU、幻觉率、无意义检测）以及词序效应的关系。

**💡 创新点**

创新点在于首次将量子认知中的 CHSH 违规度量系统化应用于 LLM，发现情境性与传统性能指标完全正交，并揭示模型情境性对安全防护（如提示注入）和交互设计的根本影响。

**🔧 技术方法**

技术手段包括：构建量子语义工具包实现 Bell 检验、使用多种采样参数（温度、核阈值、top‑k）对模型进行遍历、计算 CHSH |S|、统计分布特征（IQR、偏度、峰度）以及利用 density‑matrix 公式求得 S。

**📊 数据集**

数据集涵盖五组二义词对（如 bank/bat、crane/pen 等），配以三种随机句子模板，共计约 26,000 条实验样本；所有实验均在 24 个模型（从 0.6B 到 27B 参数规模、稠密与稀疏 MoE 架构）上执行。

**📈 对比分析**

比较方法是将 |S| 的分布特征与 MMLU 分数、幻觉率和无意义检测结果进行相关性分析；结果显示 IQR 与所有基准相关系数 |ρ|<0.03，p>0.9，证明情境性与传统指标无显著关联；同时，各模型在不同采样参数下的 |S| 变异率表明情境性是可调节的。

**⚠️ 局限性**

局限性包括：仅测试了有限的二义词对和句子模板，未覆盖更广泛的语言现象；采样参数网格在某些模型（如 Claude）受 API 限制无法完整遍历；实验未直接评估情境性对实际任务性能的提升，仍需进一步研究其在协作与安全场景中的具体应用。

---

## 216. Measuring Research Convergence in Interdisciplinary Teams Using Large Language Models and Graph Analytics

**arXiv ID:** 2603.20204 | [PDF](https://arxiv.org/pdf/2603.20204v1)

**作者:** Wenwen Li `[一作]` (Arizona State University), Michael Hanemann `[通讯]` (Arizona State University)

**通讯引用:** 25942 | [OpenAlex ID](https://openalex.org/A5010593674)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种多层AI驱动的框架，利用大语言模型提取NABC视角并通过图分析评估跨学科团队的研究趋同。

**💡 创新点**

创新点在于将LLM视角抽取、基于语义相似的3D图可视化、交叉领域影响量化（Eigenvector Centrality）以及时间观点流分析结合，形成端到端的趋同评估流程。

**🔧 技术方法**

主要技术包括OpenAI GPT大语言模型、文本嵌入（text‑embedding‑3‑small）和余弦相似度、GeoGraphViz三维图可视化、图中心性指标以及人机交互验证。

**📊 数据集**

使用的数据集为阿利桑那州水安全项目团队的11场NABC演示稿转录文本，共提取89个观点，涵盖六个研究领域。

**📈 对比分析**

通过专家结构化问卷和跨层一致性检验验证结果，研究发现观点相似度和影响网络随时间提升，边缘/节点比率持续上升，表明团队趋同增强。

**⚠️ 局限性**

局限在于LLM推理可能出现幻觉、过度解读和推断不确定性，需人工审查；样本规模有限，结果仅适用于单一案例，未与传统文献计量或问卷方法进行量化对比。

---

## 217. Global Dataset of Solar Power Plants: Multidimensional Integration and Analysis

**arXiv ID:** 2603.20601 | [PDF](https://arxiv.org/pdf/2603.20601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 218. Permutation-Consensus Listwise Judging for Robust Factuality Evaluation

**arXiv ID:** 2603.20562 | [PDF](https://arxiv.org/pdf/2603.20562v1)

**作者:** Tianyi Huang `[一作]`, Elsa Fan `[通讯]` (App-In Club)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在列表式事实性评估中，提出了一个在推理阶段通过多次重新排列候选答案并聚合结果来消除候选顺序敏感性的判别器PCFJudge。

**💡 创新点**

创新点在于：①仅通过候选顺序扰动进行推理时间改进，无需模型微调、检索或额外验证器；②使用加权平均（分数、Borda排名、top‑set 指标、校准不确定性）得到一个稳健的共识评分；③证明并验证了“置换共识”在单一模型上可显著提升决策准确性。

**🔧 技术方法**

技术手段包括：多轮候选顺序采样、同一提示词在不同排列下多次推理、对每轮输出（分数、排名、二值标记）映射回原始候选、统计平均与Borda加权、最终加权组合得分；在评估时还使用了Hoeffding不等式等概率分析。

**📊 数据集**

使用的数据集：RewardBench 2 Factuality 子集（每条 4 条候选，固定 300 条样本）以及 JudgeBench（对比式任务，固定 100 对样本）。

**📈 对比分析**

与直接单顺序评估相比，PCFJudge 在 RewardBench 2 Factuality 上实现了 5.17/7.00 的绝对提升（分别对应 GPT‑5.4 和 Claude Sonnet 4.6），加权平均提升 6.08 分；在 JudgeBench 上亦分别提升 3.24/2.70，显示其在更一般的对比评估中也能获得正向效果。

**⚠️ 局限性**

局限性包括：①实验仅基于固定 API 预算的切片，未覆盖完整数据分布；②主要收益集中在四候选列表式事实性评估，pairwise 评估收益较小；③方法需要多次 API 调用，成本上升；④未解决标签噪声、评测有效性等更深层问题。

---

## 219. Modernizing Amdahl's Law: How AI Scaling Laws Shape Computer Architecture

**arXiv ID:** 2603.20654 | [PDF](https://arxiv.org/pdf/2603.20654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 220. Bayesian Learning in Episodic Zero-Sum Games

**arXiv ID:** 2603.20604 | [PDF](https://arxiv.org/pdf/2603.20604v1)

**作者:** Chang-Wei Yueh `[一作]` (University of Southern California), Rahul Jain `[通讯]` (University of Southern California)

**通讯引用:** 4580 | [OpenAlex ID](https://openalex.org/A5002082998)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在未知转移和奖励模型的有限时限两人零和马尔可夫游戏中使用后验采样（Thompson Sampling）学习算法，并给出了子线性 regret 上界。

**💡 创新点**

首次证明后验采样在多智能体零和游戏中的有效性，给出期望 regret 上界为 𝒪(HS√(ABH K log(SABH K)))，且在对手任意学习策略时亦保持同样性能。

**🔧 技术方法**

采用 Bayesian 框架、后验采样、动态规划求解极小极大策略、置信区间与 Hoeffding/Union Bound 等概率工具、线性规划求解 equilibrium、经验估计等技术。

**📊 数据集**

实验基于 3×3 网格猎物–捕食者仿真环境，使用自定义的随机转移概率和确定性距离奖励。

**📈 对比分析**

与对手使用真实 equilibrium、后验采样和 fictitious play 三种策略对比，后验采样在所有设置下均实现子线性 regret，特别是在对手使用 fictitious play 时表现最佳；与后验采样竞争时 regret 接近零。

**⚠️ 局限性**

仅针对两人零和有限时限游戏；未涵盖非零和或连续状态/动作空间；假设先验与观测完全可得；求解 equilibrium 的线性规划开销较大。

---

## 221. MFSR: MeanFlow Distillation for One Step Real-World Image Super Resolution

**arXiv ID:** 2603.20690 | [PDF](https://arxiv.org/pdf/2603.20690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 222. Unveiling the Security Risks of Federated Learning in the Wild: From Research to Practice

**arXiv ID:** 2603.20615 | [PDF](https://arxiv.org/pdf/2603.20615v1)

**作者:** Jiahao Chen `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 7907 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建统一的评测框架 TFLlib，系统性分析并测评联邦学习在实际部署环境中的投毒攻击安全风险。

**💡 创新点**

创新点在于：①将研究与实践差距进行系统化梳理并提出实用威胁模型、混合异构性以及综合稳定性/效用损失评估指标；②通过实际环境下的实测，揭示理想化评估对投毒攻击效果的过度估计。

**🔧 技术方法**

采用 FedAvg 聚合、六种 backdoor 攻击与九种 Byzantine 攻击，基于 PyTorch 的 TFLlib 框架实现；结合 CNN/MLP/Transformer 等模型进行联邦学习。

**📊 数据集**

使用图像数据集 CIFAR‑10、FEMNIST，表格数据集 Purchase100、Texas100，文本数据集 IMDB、AGNews，涵盖图像、表格与文本三种模态。

**📈 对比分析**

在理想化与实际化（随机客户端选择、设备/通信/统计异构、低攻击者比例等）两种设置下比较攻击效果、稳定性（BSA/BSV）与效用损失（ACC/ACCV），结果显示实际设置下攻击效果显著下降、稳定性差且往往伴随显著的主任务精度损失。

**⚠️ 局限性**

局限性包括：仅聚焦投毒攻击，未评估隐私攻击；实验规模与场景相对有限，未覆盖多种聚合算法或防御策略，且缺少对更大规模工业部署的验证。

---

## 223. Weber's Law in Transformer Magnitude Representations: Efficient Coding, Representational Geometry, and Psychophysical Laws in Language Models

**arXiv ID:** 2603.20642 | [PDF](https://arxiv.org/pdf/2603.20642v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 224. Epistemic Observability in Language Models

**arXiv ID:** 2603.20531 | [PDF](https://arxiv.org/pdf/2603.20531v1)

**作者:** Tony Mason `[一作]` (University of British Columbia), Tony Mason `[通讯]` (University of British Columbia)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5041896410)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首先证明在仅观察文本输出的监督模型下，语言模型的自我报告置信度与真实性相反，随后提出一种导出每个 token 熵的 tensor 接口来逃离该不可能性，并构建了不同验证预算下的成本-准确率曲线。

**💡 创新点**

创新点包括：①对文本观测不足导致无法验证诚实的正式不可能性证明；②设计可在标准训练下天然耦合于正确性的熵信号的 tensor 接口；③将该接口与传统文本特征、组合判定器相结合，给出实用的验证成本曲线。

**🔧 技术方法**

使用的技术包括形式化概率与学习论证、Transformer 的 per‑token 熵与 log‑prob 采集、实验中基于四大 LLM（OLMo‑3、Llama‑3.1、Qwen3、Mistral）的推理和评估。

**📊 数据集**

实验数据集由 200 条平衡查询组成（100 可知事实、100 诱导性虚构），并在四种模型架构上进行验证；同时在 API 端验证熵信号在五个更大模型上的泛化。

**📈 对比分析**

在 10%、20%、30% 验证预算下，基于长度的文本判定器最高准确率约 87.6%，基于熵的 tensor 判定器可提升至 90.2%，组合判定器进一步到 91.8%；熵信号在所有模型上 AUC 0.757，显著优于任何文本特征。

**⚠️ 局限性**

主要限制包括：仅针对文本与熵的二元信号，无法覆盖多模态或需要对抗训练时的鲁棒性；依赖供应商暴露内部信号；在高度可信域的外部验证（如法律、医学）仍需进一步扩展和校准。

---

## 225. HSI Image Enhancement Classification Based on Knowledge Distillation: A Study on Forgetting

**arXiv ID:** 2603.20292 | [PDF](https://arxiv.org/pdf/2603.20292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 226. RoboECC: Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models

**arXiv ID:** 2603.20711 | [PDF](https://arxiv.org/pdf/2603.20711v1)

**作者:** Zihao Zheng `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**通讯引用:** 36847 | [OpenAlex ID](https://openalex.org/A5100641667)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了RoboECC框架，实现Vision‑Language‑Action模型的边云协同部署，并通过模型‑硬件共感分割与网络适配实现最优切分。

**💡 创新点**

创新点包括：① 针对多样VLA结构的模型‑硬件共感分割策略；② 网络波动感知与动态切分调整机制；③ 参数共享池降低传输开销。

**🔧 技术方法**

使用了结构与硬件性能建模、深度优先搜索、LSTM带宽预测、参数共享池以及实时切分调整等技术。

**📊 数据集**

在LIBERO、SimplerEnv仿真环境以及真实机器人场景（AgileX PIPER）收集的1000条数据进行微调，作为实验数据集。

**📈 对比分析**

与纯边缘、纯云以及固定切分方案对比，RoboECC在Orin+A100平台提升3.16×–3.28×，Thor+A100提升2.10×–2.23×，总延迟显著降低，网络开销仅占2.55%–2.62%。

**⚠️ 局限性**

限制包括仅支持GPU硬件模型，未考虑CPU/NPUs或其他加速器；对不同VLA结构的泛化仍有限；网络预测依赖历史数据，可能受突发波动影响。

---

## 227. Evaluating Large Language Models on Historical Health Crisis Knowledge in Resource-Limited Settings: A Hybrid Multi-Metric Study

**arXiv ID:** 2603.20514 | [PDF](https://arxiv.org/pdf/2603.20514v1)

**作者:** Mohammed Rakibul Hasan `[一作]` (North South University), Mohammed Rakibul Hasan `[通讯]` (North South University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5113061038)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 GPT-4、Gemini Pro、Llama 3 与 Mistral‑7B 在孟加拉国低资源背景下对 COVID‑19、登革热、尼帕病毒和基孔肯雅病毒历史疫情知识的回答可靠性。

**💡 创新点**

提出语义相似度、自然语言推理（NLI）与 LLM‑as‑Judge 三视角融合的混合评估框架，并首次对多模型多疾病在低资源环境中的历史知识准确性进行系统比较。

**🔧 技术方法**

使用句子级嵌入相似度、零射 NLI 分类、基于 GPT‑4/Gemini Pro 的 LLM‑as‑Judge 评分，以及混合分数 HFS/HRS 计算。

**📊 数据集**

构建 100 道问答（每病种 25 题）取自 WHO 报告与同行评议期刊的权威文本，生成 400 条回答进行评估。

**📈 对比分析**

通过平均余弦相似度、NLI Entailment/Contradiction 统计及 HFS/HRS 计算进行比较；结果显示无模型达到高一致性阈值，COVID‑19 领域高分歧率 72%，Mistral‑7B 取得最高 HFS（0.507）且最小 hallucination；GPT‑4 与 Gemini Pro 事实准确性较低。

**⚠️ 局限性**

仅在英语单语环境评估，缺乏多语种适用性；LLM‑as‑Judge 评分为合成数据；NLI 受限于推理模型能力；参考答案为静态，未考虑知识演进；仅聚焦孟加拉国，泛化性有限。

---

## 228. MERIT: Multi-domain Efficient RAW Image Translation

**arXiv ID:** 2603.20836 | [PDF](https://arxiv.org/pdf/2603.20836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 229. Breaking the $O(\sqrt{T})$ Cumulative Constraint Violation Barrier while Achieving $O(\sqrt{T})$ Static Regret in Constrained Online Convex Optimization

**arXiv ID:** 2603.20671 | [PDF](https://arxiv.org/pdf/2603.20671v1)

**作者:** Haricharan Balasundaram `[一作]`, Rahul Vaze `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了约束在线凸优化（COCO）中在二维情形下实现同时满足 O(√T) 静态回报与 O(T^{1/3}) 约束累计违例（CCV）的算法；

**💡 创新点**

提出在二维凸集合投影过程中同时利用周长与面积这两种互补几何量来控制 CCV，从而突破此前认为不可实现的 O(√T) 下限；

**🔧 技术方法**

采用投影优化（OGD + 投影）算法，并利用几何不等式（如 Cauchy 表面面积公式、三角形面积与周长关系）以及 Hölder 不等式对投影距离进行分析；

**📊 数据集**

论文未使用任何外部数据集，而是理论分析推导；

**📈 对比分析**

与之前的 O(√T) 回报、O(√T) 或 O(√T log T) CCV 上限进行对比，证明在 d=2 时可获得更优的 CCV 上限 O(T^{1/3})，在理论上提升了约束累积误差的控制；

**⚠️ 局限性**

局限性在于仅适用于二维情况，三维及更高维时仍无法获得类似的更优 CCV 上限，且作者仅给出 O(1) 的猜想而未证明。

---

## 230. Neuronal Self-Adaptation Enhances Capacity and Robustness of Representation in Spiking Neural Networks

**arXiv ID:** 2603.20687 | [PDF](https://arxiv.org/pdf/2603.20687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 231. kRAIG: A Natural Language-Driven Agent for Automated DataOps Pipeline Generation

**arXiv ID:** 2603.20311 | [PDF](https://arxiv.org/pdf/2603.20311v1)

**作者:** Rohan Siva `[一作]` (Cisco), Ganesh Sundaram `[通讯]` (Cisco)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了kRAIG AI代理，将自然语言需求自动转化为可执行的Kubeflow Pipelines。

**💡 创新点**

引入ReQuesAct交互框架和检索增强工具合成，提升了意图澄清和工具选择的可靠性。

**🔧 技术方法**

采用大型语言模型、检索增强生成(RAG)、Kubeflow、LLM验证与约束工具调用。

**📊 数据集**

使用ELT-Bench、Hugging Face数据集、GitHub仓库等多源数据进行评测。

**📈 对比分析**

与SWE-Agent、Spider-Agent基线对比，kRAIG在SRDEL 75%/SRDT 25% 上显著优于基线，生成率提升约3倍。

**⚠️ 局限性**

对复杂转换任务仍表现不稳定，缺乏充分的澄清问题生成，安全验证仍需进一步加强。

---

## 232. Achieving $\widetilde{O}(1/ε)$ Sample Complexity for Bilinear Systems Identification under Bounded Noises

**arXiv ID:** 2603.20819 | [PDF](https://arxiv.org/pdf/2603.20819v1)

**作者:** Hongyu Yi `[一作]` (University of Washington), Jing Yu `[通讯]` (University of Washington)

**通讯引用:** 22116 | [OpenAlex ID](https://openalex.org/A5063288559)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种针对离散时间双线性系统的有限样本集合成员识别（SME）方法，适用于具有有界对称对数凹噪声的情形。

**💡 创新点**

创新点在于：① 在仅假设边缘稳定、状态方程可能呈多项式均方增长的前提下，首次证明SME的可行参数集在样本量为$O(1/ε)$时直径收敛；② 明确给出了与系统维度相关的样本复杂度表达式；③ 在对数凹噪声下引入BMSB（区块马尔可夫小球）条件和对数凹反分散技术，克服了传统高斯尾部分析的局限。

**🔧 技术方法**

技术方法包括：集合成员估计、持续激励与BMSB条件、Paley–Zygmund不等式、对数凹随机变量的反分散分析、Jordan分解与矩阵范数分析、以及对参数误差集的范数收敛证明。

**📊 数据集**

实验使用人工合成的结构化双线性系统：$A$为对角矩阵、$B_i$为严格下三角矩阵；输入为截断高斯分布，噪声为截断拉普拉斯分布。

**📈 对比分析**

与传统最小二乘（OLS）方法的90%置信集进行比较；结果显示SME的不确定集直径随样本数持续收缩，显著优于OLS基准，提供更紧凑的参数不确定性量化。

**⚠️ 局限性**

局限性包括：仅在边缘稳定、对数凹噪声且噪声有界的假设下得到结论；理论常数可能过于保守；实验仅在模拟数据上验证，缺乏真实系统的实证检验。

---

## 233. Exploring Teacher-Chatbot Interaction and Affect in Block-Based Programming

**arXiv ID:** 2603.20211 | [PDF](https://arxiv.org/pdf/2603.20211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 234. ROI-Driven Foveated Attention for Unified Egocentric Representations in Vision-Language-Action Systems

**arXiv ID:** 2603.20668 | [PDF](https://arxiv.org/pdf/2603.20668v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 235. Neural collapse in the orthoplex regime

**arXiv ID:** 2603.20587 | [PDF](https://arxiv.org/pdf/2603.20587v1)

**作者:** James Alcala `[一作]` (University of Southern California), Yangxinyu Xie `[通讯]` (University of Pennsylvania)

**通讯引用:** 231 | [OpenAlex ID](https://openalex.org/A5042596205)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究了在正交双极子（orthoplex） regime 下神经网络软最大化（softmax）代码的几何结构，并证明其与球面码等价

**💡 创新点**

首次将软最大化代码与球面码在 d+2≤n≤2d 范围内等价，揭示自对偶性和温度对交叉熵最优解的影响

**🔧 技术方法**

使用 Radon 定理、凸性分析、Rankin 的正交双极子上界以及 Jensen、Taylor 近似等数学工具

**📊 数据集**

无实验数据集，论文完全基于理论分析

**📈 对比分析**

通过对交叉熵损失函数的解析最小化和温度阈值阐述，证明低温下偏向低熵代码，高温下偏向高熵代码

**⚠️ 局限性**

仅适用于 d+2≤n≤2d，尚未扩展至 n>2d 或具体模型实现，且缺乏实验验证

---

## 236. Generating from Discrete Distributions Using Diffusions: Insights from Random Constraint Satisfaction Problems

**arXiv ID:** 2603.20589 | [PDF](https://arxiv.org/pdf/2603.20589v1)

**作者:** Alankrita Bhatt `[一作]` (Granica Computing Inc.), Andrea Montanari `[通讯]` (Granica Computing Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究利用连续和离散扩散（diffusion）方法生成随机 k‑SAT 与 k‑XORSAT 公式的均匀解，并将理论预测（BP、相位转移）与实证结果对比。

**💡 创新点**

① 连续扩散优于离散扩散；② 学习得到的 NN 去噪器能逼近理论最优 BP 去噪器；③ 基于“cavity”初始化的 BP 去噪器是局部算法中最优的；④ 采用“逆叶子”或“逆度”排序显著提升采样成功率；⑤ 采样出的解在 KL 距离上接近均匀分布。

**🔧 技术方法**

离散扩散（masked diffusion）、连续扩散、loopy Belief Propagation、神经网络去噪器（基于图卷积）、变量排序启发式、梯度训练与课程学习。

**📊 数据集**

合成数据：随机生成的 k‑SAT 与 k‑XORSAT 公式（N=100/200/300，k=3…8，α 从 0 到  α_{k}）。

**📈 对比分析**

与 BP 基线对比；在相同 α、r、N 条件下，学习去噪器与 BP 的成功率相当；排序策略使成功率提升 10%–50%；在 N 增大时，生成解的 KL 距离趋近理论上限。

**⚠️ 局限性**

① 只在中等规模实例上验证；② XORSAT 的训练更困难；③ 仍受动态相位转移的限制，随机排序不能突破 α_{k}；④ 生成解的均匀性虽接近但未完全达标；⑤ 仅研究局部去噪器，未考虑全局或更高阶方法。

---

## 237. NextSense: A Semi-Synthetic Sensing Data generation Platform

**arXiv ID:** 2603.20789 | [PDF](https://arxiv.org/pdf/2603.20789v1)

**作者:** David Rico Menéndez `[一作]` (University Carlos III Madrid), Alain Mourad `[通讯]` (InterDigital Europe)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于OAI 5G栈、Keysight PROPSIM硬件信道模拟器和Amarisoft UE模拟器的开源半合成传感数据生成平台NextSense，支持通过API/GUI自定义射频、信道、移动与流量，并同步采集符号级IQ、协议日志及关键性能指标等多视角数据。

**💡 创新点**

①首个完整开放、可编程且自动化的半合成5G传感数据平台；②将高保真硬件信道仿真与全栈5G栈深度集成；③提供多层协议级日志与多视角输出，满足AI/ML传感任务的数据需求。

**🔧 技术方法**

OAI 5G核心与gNB、Amarisoft UE模拟器、Keysight PROPSIM F64信道模拟器、Docker Compose容器化、REST/SCPI API、Python/JSON配置接口、统计检验方法（Kolmogorov–Smirnov、Wasserstein距离等）及SVM分类器。

**📊 数据集**

室内存在检测基准数据：真实环境下的CSI/SSB轨迹（空房间、人类、移动机器人三种场景），用于构建信道模型并生成对应的半合成数据；合成数据也用于训练SVM进行域迁移验证。

**📈 对比分析**

通过统计检验（方差、KS检验、Wasserstein距离）和时域/频域/相位分布对比验证合成数据与真实数据分布一致；SVM在合成训练、真实测试中取得98.33%准确率，表明低域迁移误差和高统计真实度。

**⚠️ 局限性**

限制：尚未验证相干信号层级结构的完整性；信道模型基于有限统计，缺乏更复杂网络/多用户/多天线场景的验证；仅在室内存在检测任务上评估，未覆盖更广泛的传感应用；硬件同步精度和平台成本仍是潜在瓶颈。

---

## 238. LASER: Level-Based Asynchronous Scheduling and Execution Regime for Spatiotemporally Constrained Multi-Robot Timber Manufacturing

**arXiv ID:** 2603.20577 | [PDF](https://arxiv.org/pdf/2603.20577v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 239. InjectFlow: Weak Guides Strong via Orthogonal Injection for Flow Matching

**arXiv ID:** 2603.20303 | [PDF](https://arxiv.org/pdf/2603.20303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. Can I guess where you are from? Modeling dialectal morphosyntactic similarities in Brazilian Portuguese

**arXiv ID:** 2603.20695 | [PDF](https://arxiv.org/pdf/2603.20695v1)

**作者:** Manoel Siqueira `[一作]` (Universidade Federal de Alagoas), Raquel Freitag `[通讯]` (Universidade Federal de Sergipe)

**通讯引用:** 752 | [OpenAlex ID](https://openalex.org/A5073085655)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究巴西葡萄牙语中与第二人称代词相关的四个形态句法变量在语音样本中的共变模式，并利用相关分析和聚类方法尝试根据这些模式推断说话者的方言来源。

**💡 创新点**

首次将社会语言学的共变理论与自然语言处理的聚类技术结合，揭示方言身份在语料中呈梯度分布而非单一边界，并强调在模型训练中必须兼顾方言多样性与伦理考量。

**🔧 技术方法**

采用 Spearman 相关、卡方检验、逻辑回归、混合效应模型、K‑medoids 聚类（pam）和主成分分析等统计与机器学习技术。

**📊 数据集**

使用 Falares Sergipanos 语料库的三份样本（Deslocamentos 2019/2020 与 Linguagem Corporificada 2023），共 181 名 18‑30 岁大学生说话者，约 100 万词。

**📈 对比分析**

对比相关分析与聚类结果：相关分析仅发现一个显著的负相关（定冠词缺失与 você 的使用），而聚类识别出三组说话者，主成分解释了 58.9% 方差，但聚类与预期的迁徙/方言分组并不完全吻合；总体效果中等，说明仅凭这四个形态句法特征难以高度精确地区分方言。

**⚠️ 局限性**

局限包括：样本规模对 NLP 训练来说仍显不足，聚类仅考虑形态句法层面缺乏音系、词汇等信息，受访者年龄与社会阶层受限，自动转写工具可能对非标准方言产生偏差，且聚类内部同质度低于预期。

---

## 241. Position: Multi-Agent Algorithmic Care Systems Demand Contestability for Trustworthy AI

**arXiv ID:** 2603.20595 | [PDF](https://arxiv.org/pdf/2603.20595v1)

**作者:** Truong Thanh Hung Nguyen `[一作]` (Analytics Everywhere Lab, University of New Brunswick), Hung Cao `[通讯]` (Analytics Everywhere Lab, University of New Brunswick)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并阐述了在多智能体医疗决策系统中，将争议可控（contestability）视为实现可信 AI 的核心设计要求，并通过示例框架（CANOE）演示如何在老年居家照护场景中实现争议可控的人工参与决策流程。

**💡 创新点**

创新点在于：① 将争议可控概念推广到多智能体系统；② 设计了基于结构化论证与量化双极论证框架（QBAF）的争议评估方法；③ 结合人机协同的角色化争议环节，让临床人员在决策生命周期内主动挑衅、修正或撤销系统输出；④ 构建了可扩展的专家网络与动态角色招聘机制，以适应不同临床任务。

**🔧 技术方法**

主要技术包括：多智能体架构、检索增强生成（RAG）+大型语言模型、结构化论证生成与评估、量化双极论证（QBAF）语义、角色化人机争议循环、可视化支持与调度执行模块（MCP）。

**📊 数据集**

使用的数据集主要是：患者临床记录与功能评估（如 interRAI Home Care Assessment）、医学文献检索数据库（可能包含 MIMIC‑III 等）、以及通过嵌入模型检索的案例相关证据。

**📈 对比分析**

文中并未给出系统的定量实验评估，作者建议未来通过与传统单一模型或无争议可控系统对比，衡量争议可控对决策准确性、错误修正率、临床工作负荷和信任度的提升。性能表现因尚未完成实验而无法给出具体指标。

**⚠️ 局限性**

局限性包括：① 争议可控机制增加系统复杂度，需额外的治理与版本管理；② 多智能体互相辩论可能导致责任模糊，争议频繁可能导致临床人员负担加重；③ 对不同文化、组织的适配性不足，争议流程的可接受性尚待实证验证；④ 目前框架仍处于概念验证阶段，缺乏大规模临床部署与真实案例的评估。

---

## 242. Graph-based data-driven discovery of interpretable laws governing corona-induced noise and radio interference for high-voltage transmission lines

**arXiv ID:** 2603.20600 | [PDF](https://arxiv.org/pdf/2603.20600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

---

## 243. Spatio-Temporal Grid Intelligence: A Hybrid Graph Neural Network and LSTM Framework for Robust Electricity Theft Detection

**arXiv ID:** 2603.20488 | [PDF](https://arxiv.org/pdf/2603.20488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 244. Coding Agents are Effective Long-Context Processors

**arXiv ID:** 2603.20432 | [PDF](https://arxiv.org/pdf/2603.20432v1)

**作者:** Weili Cao `[一作]` (Duke University), Shuyan Zhou `[通讯]` (Duke University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用现成的编程式 LLM（如 Codex、Claude Code）将长文本任务转化为文件系统操作，自动化地组织、搜索、过滤和处理海量文本，展示其在长上下文任务中的强大处理能力。

**💡 创新点**

提出将长上下文任务重新表述为文件系统导航问题，利用编码代理的“工具使用熟练度”和“文件系统亲和力”来显式操作文本，而非仅依赖隐式注意力或传统检索管道。

**🔧 技术方法**

核心技术包括：现成的代码生成 LLM、终端命令（如 grep、sed）、Python 脚本执行、迭代查询精炼与程序化聚合；未对模型进行任务特定训练或架构改造。

**📊 数据集**

使用的基准数据集包括：BrowseComp‑Plus（100K 网页集合）、Oolong‑Synthetic 与 Oolong‑Real（长文档推理）、LongBench（多任务长文本 QA）以及 Natural Questions（开放域问答，3T 维基百科语料）。

**📈 对比分析**

与 GPT‑5 全上下文、传统 RAG、ReAct 代理、递归语言模型（RLM）等基线比较，Coding Agents 在 5 个基准中平均提升 17.3%，在 4/5 任务中刷新 state‑of‑the‑art；性能随上下文规模从 188K 到 3T 依旧稳健。

**⚠️ 局限性**

局限性包括：向代理提供检索工具往往不提升甚至降低效果；代理主要为编程任务调优，未专门针对长文本推理；计算成本相对较高；缺乏针对大规模文本检索与导航的专业化训练与框架。

---

## 245. AnchorNote: Exploring Speech-Driven Spatial Externalization for Co-Located Collaboration in Augmented Reality

**arXiv ID:** 2603.20199 | [PDF](https://arxiv.org/pdf/2603.20199v1)

**作者:** Diya Hundiwala `[一作]` (Princeton University), Andrés Monroy-Hernández `[通讯]` (Princeton University)

**通讯引用:** 7090 | [OpenAlex ID](https://openalex.org/A5013065278)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在同位协作场景下，通过AR眼镜捕捉口头想法并转化为空间锚定的粘性便签的原型系统；

**💡 创新点**

创新点在于将语音捕获与空间持久化相结合，并强调意图控制（手势或按钮）、即时转录与LLM摘要、以及对协作流程的系统性评估；

**🔧 技术方法**

使用了实时语音转录（speech‑to‑text）、大型语言模型（LLM）进行摘要、AR空间锚定技术、手势/按钮触发控制以及多设备同步；

**📊 数据集**

使用了20名本科生配对进行的现场脑力风暴与主题分组任务数据，没有使用公开语料库；

**📈 对比分析**

通过与传统纸质粘性便签的对照，评估了写作负担、监控与恢复工作量、对话节奏与空间组织效果；实验显示写作负担下降但监控/恢复成本升高，明确按钮控制能显著降低协调成本，空间持久化在保持清晰度时表现良好；

**⚠️ 局限性**

局限包括样本量有限、仅使用单一AR硬件、手势控制对运动障碍者不友好、语音转写/摘要的准确性与隐私问题，以及缺乏多说话人与更长时间的评估。

---

## 246. Leveraging Natural Language Processing and Machine Learning for Evidence-Based Food Security Policy Decision-Making in Data-Scarce Making

**arXiv ID:** 2603.20425 | [PDF](https://arxiv.org/pdf/2603.20425v1)

**作者:** Karan Kumar Singh `[一作]` (Sharda University), Nikita Gajbhiye `[通讯]` (Sharda University)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5084156945)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出ZeroHungerAI框架，融合自然语言处理与机器学习，对数据稀缺地区的食品安全政策进行基于证据的预测与干预优先级排序。

**💡 创新点**

创新点在于将DistilBERT上下文嵌入与结构化社会经济指标进行特征融合，并加入公平性约束与决策聚焦学习，在极低数据条件下实现精准且无偏的干预决策。

**🔧 技术方法**

技术包括Transformer（DistilBERT）提取文本语义，传统ML分类器（Logistic回归、SVM等），特征融合、公平性正则化（人口群体差距约束）以及基于预算的约束优化。

**📊 数据集**

使用混合数据集：1,200–2,000条样本，涵盖25–40个地区，包含结构化指标（营养率、降雨偏差、粮食价格等）和非结构化政策文件、政府报告、社区叙事。

**📈 对比分析**

与Logistic回归、SVM等基线比较，ZeroHungerAI在25区数据集上达91%准确率、0.89精确度、0.85召回率、F1≈0.86；在2,000样本集上提升至94–95%准确率、AUC≈0.95，较基线提升13–17%，公平性差距仅3%。

**⚠️ 局限性**

局限性包括对OCR噪声和文本质量的敏感性、对部分标注数据的依赖、Transformer模型的计算开销以及在实时大规模部署中的可扩展性待验证。

---

## 247. "Girl, I'm so Serious": CARE, a Capability Framework for Reproductive Equity in Human-AI Interaction

**arXiv ID:** 2603.20511 | [PDF](https://arxiv.org/pdf/2603.20511v1)

**作者:** Alice Zhong `[一作]` (UNC Chapel Hill), Snehalkumar 'Neil' S. Gaikwad `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并应用CARE（Capability Approach for Reproductive Equity）框架，对现有SRH（性与生殖健康）相关AI聊天机器人、通用LLM和搜索引擎功能进行评估，识别出两类认知伤害（来源不透明与回答僵化），并给出设计与政策改进建议。

**💡 创新点**

将Sen的能力方法与Nussbaum的十大核心能力结合，形成针对SRH的规范设计视角和评估视角，首次将“认知伤害”概念引入AI评估，并提出以能力扩展为核心的评价维度。

**🔧 技术方法**

采用自然语言处理技术进行回答的可信度与推理质量标注，结合人工标注与自动化域名分析，评估聊天机器人的引用来源与回答完整度。

**📊 数据集**

使用bedsider.org的SRH问答集作为专家答案基准，并采集多种SRH专用聊天机器人、通用LLM（ChatGPT、Claude、Gemini、Copilot）以及搜索引擎AI模式的交互日志，统计引用域名列表。

**📈 对比分析**

对照可信度（是否提供可验证来源）和推理质量（与专家答案一致性）两维度进行评分，结果显示约28%高质量推理、26%中等、46%低质量；LLM回答平均长度显著超过专家答案，导致信息过载；整体未达CARE设定的规范。

**⚠️ 局限性**

评估样本有限，未覆盖多语言或跨文化场景；缺乏实际用户研究验证能力扩展与生殖自主的真实影响；评估指标与实际生殖自由之间仍存在映射不完全的问题。

---

## 248. Graph-Aware Text-Only Backdoor Poisoning for Text-Attributed Graphs

**arXiv ID:** 2603.20339 | [PDF](https://arxiv.org/pdf/2603.20339v1)

**作者:** Qi Luo `[一作]` (Shandong University), Xiuzhen Cheng `[通讯]` (Shandong University)

**通讯引用:** 18075 | [OpenAlex ID](https://openalex.org/A5100692488)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种仅通过改写节点文本而非图结构的图文本属性网络（TAG）后门注入框架TAGBD；

**💡 创新点**

创新点在于：①使用不确定性引导的节点选择；②联合训练文本触发器生成器TextTrojan与影子GNN，使触发文本既与图上下文一致，又保持自然；③提供覆盖“覆盖”和“追加”两种注入策略，兼顾攻击强度与隐蔽性；

**🔧 技术方法**

技术包括：图神经网络（GCN/GraphSAGE等）、文本编码器-解码器（BOW、GTR‑T5、Sonar）、不确定性度量（熵）、联合训练与语义相似性约束；

**📊 数据集**

在Cora、Pubmed、Arxiv三个文本属性图数据集上进行实验；

**📈 对比分析**

与多种结构性后门基线（SBA‑Samp/Gen、GTA、UGBA、DPGBA）以及两种防御（Prune、OD）比较，TAGBD在攻击成功率（ASR）上达到或接近100%，同时保持较高的干净准确率（CA）和较低的困惑度（PPL），并展示了在不同目标模型（GCN、GraphSAGE、GAT、GraphTransformer、RobustGCN）上的良好迁移性；

**⚠️ 局限性**

局限性包括：仍需对文本生成质量和触发文本长度进行平衡；在极大规模或高度异构的图上实验不足；对文本编辑的先验假设（可自由修改文本）可能不适用于所有真实场景；

---

## 249. Beyond LLM-based test automation: A Zero-Cost Self-Healing Approach Using DOM Accessibility Tree Extraction

**arXiv ID:** 2603.20358 | [PDF](https://arxiv.org/pdf/2603.20358v1)

**作者:** Renjith Nelson Joseph `[一作]` `[通讯]`, Renjith Nelson Joseph

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个零成本自愈式Web测试自动化框架，利用DOM可访问性树提取算法在首次执行时生成十层优先级的选择器，并在测试失败时只重新提取失效的选择器实现局部自愈。

**💡 创新点**

创新点包括：①基于可访问性树的十层优先级选择器体系，完全不依赖LLM；②单一全局选择器缓存，减少设备间重复发现；③三层业务流程层级（L0/1/2）使非技术人员可直观看到测试覆盖；④实时进度报告与原子文件写入保证并行执行安全；⑤实现零API成本与低维护成本。

**🔧 技术方法**

技术栈包括：Python 3.9 + Playwright + pytest + pytest‑xdist；DOM可访问性树提取与定位算法；CSS/ARIA/Role/Fragment/Visible Text多层选择器；原子文件写入与文件锁；WebKit/Chromium等多浏览器环境；自愈时局部重发现逻辑。

**📊 数据集**

使用公开的电子商务演示站点 automationexercise.com，构建了 3 设备配置（Desktop Chrome、Desktop Safari、iPhone 15）+ 10 个业务流程，31 个测试组合（含自愈演示）。

**📈 对比分析**

通过对比 4,500 次/月的测试执行，比较了本框架与 LLM（Browser Use+Claude、GPT‑4o）、Testim、Functionize、Mabl 等商业 SaaS。结果显示：本框架零API成本，月 TCO 仅 200–600 USD，测试通过率 31/31（100%），并行执行耗时 22 秒；相较于 LLM 方案的 1,350–2,160 USD/月 API 费用，优势高达 3–14 倍。

**⚠️ 局限性**

局限性：①无法发现需要认证或后端状态的元素；②对不完善的 ARIA/可访问性实现会退回到文本选择器，降低多语言鲁棒性；③当前不支持 Shadow DOM 与 canvas 渲染元素；④需手动维护模式注册表；⑤对非组件化的旧型页面适配性待进一步验证。

---

## 250. A 4R-supported circular product-service system for luxury branded events

**arXiv ID:** 2603.20613 | [PDF](https://arxiv.org/pdf/2603.20613v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 251. Global Cybercrime Damages: A Baseline for Frontier AI Risk Assessment

**arXiv ID:** 2603.20570 | [PDF](https://arxiv.org/pdf/2603.20570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 252. MINISA: Minimal Instruction Set Architecture for Next-gen Reconfigurable Inference Accelerator

**arXiv ID:** 2603.20623 | [PDF](https://arxiv.org/pdf/2603.20623v1)

**作者:** Jianming Tong `[一作]` (Georgia Institute of Technology), Tushar Krishna `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 14113 | [OpenAlex ID](https://openalex.org/A5034089074)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MINISA——一种针对 FEATHER+ 可重构推理加速器的极简指令集架构，利用 Virtual Neuron（VN）抽象在硬件最小算子粒度上实现数据流与布局的即时切换，从而大幅降低控制开销。

**💡 创新点**

创新点在于：1）引入 VN 抽象，将控制从单个寄存器/交换机级细化到算子原子级；2）设计仅包含 8 条指令的 MINISA，压缩指令序列 2×10^4–10^5 倍；3）改进 FEATHER 为 FEATHER+，支持动态输入/权重、消除内存复制；4）结合编译器层面的布局与映射共搜索，自动生成 MINISA 指令。

**🔧 技术方法**

技术方法包括：
- 虚拟神经元（VN）建模和映射约束解析；
- 通过布局描述子（分区+排序）实现四级张量分区；
- 组合指令（Set*VNLayout、SetMap、ExecuteStreaming、ExecuteMapping、Load/Store、Activate）构成 ISA；
- 采用两层仿真框架（RTL+分析模型）评估性能；
- 与 GPU（RTX 5090）和 TPUv6e 进行基准对比。

**📊 数据集**

使用了 50 个 GEMM 样例，涵盖：
- 大语言模型推理（GPT‑OSS 20B）
- 同态加密（FHE）加速器基准（BConv、NTT）
- 零知识证明（ZKP）NTT
- 其他多种矩阵形状（K、N、M 变化）

**📈 对比分析**

对比方法：基线为细粒度微指令控制，和工业级 GPU/TPU。结果显示：
- 对 FEATHER+：MINISA 将指令存储压缩至 2×10^4–10^5 倍，消除 96.9% 的指令获取停顿，最大 31.6× 端到端加速；
- 与 RTX 5090 比较：FEATHER+ 在 16×256 规模下几何平均加速 23.7×；
- 与 TPUv6e 比较：几何平均加速 7.8×；
- 在所有规模下保持 >60% 的计算利用率，特别是在形状不规则时表现优异。

**⚠️ 局限性**

局限性：
- 对极小规模（≤64 PE）提升有限，微指令开销已被计算隐藏；
- 仅针对矩阵乘法/卷积等线性算子；对非线性或特殊算子需进一步扩展 ISA；
- FEATHER+ 的改进带来 ~7% 的硬件资源占用；
- 对于极大矩阵仍受 I/O 带宽限制，指令压缩无法完全弥补 I/O 约束。

---

## 253. From Human Interfaces to Agent Interfaces: Rethinking Software Design in the Age of AI-Native Systems

**arXiv ID:** 2603.20300 | [PDF](https://arxiv.org/pdf/2603.20300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 254. Hawkeye: Reproducing GPU-Level Non-Determinism

**arXiv ID:** 2603.20421 | [PDF](https://arxiv.org/pdf/2603.20421v1)

**作者:** Erez Badash `[一作]` (Pearl Research Labs), Megha Srivastava `[通讯]` (Stanford University)

**通讯引用:** 725 | [OpenAlex ID](https://openalex.org/A5043099018)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出 Hawkeye，一个能够在 CPU 上无精度损失地精确重现 NVIDIA Tensor Core 矩阵乘法的框架，从而实现可验证机器学习。

**💡 创新点**

通过系统化的测试逆向工程 GPU 的算术行为，揭示了 Tensor Core 的累加顺序、内部精度、舍入模式、归一化时机及子正常数处理，实现了与 GPU 完全相同的位级结果。

**🔧 技术方法**

使用自定义 CUDA 内联 PTX 内核进行针对 16×16 矩阵块的逐步测试，结合 CPU 仿真器实现 FP16、BF16、FP8 等精度的矩阵乘法重现，并公开了 GitHub 仓库。

**📊 数据集**

未使用传统机器学习数据集，而是通过随机生成的 16×16 矩阵（共 100,000 组）对仿真器进行验证，覆盖 Ampere、Hopper、Lov­elace 三代 GPU。

**📈 对比分析**

与 GPU 原始执行结果逐位比对，达 100% 位级重现；CPU 执行时间相对 GPU 较慢，但在审计场景下可接受；未提出数值性能提升的实验。

**⚠️ 局限性**

仅覆盖 NVIDIA Tensor Core 系统，未考虑其他厂商硬件；未对卷积、注意力等高层 ML 操作做逆向，且在分布式训练与加密证明框架中的集成仍是未来工作。

---

## 255. Hierarchical Reinforcement Learning for Next Generation of Multi-AP Coordinated Spatial Reuse

**arXiv ID:** 2603.20647 | [PDF](https://arxiv.org/pdf/2603.20647v1)

**作者:** Ziru Chen `[一作]` (Illinois Institute of Technology), Lin X. Cai `[通讯]` (Illinois Institute of Technology)

**通讯引用:** 15416 | [OpenAlex ID](https://openalex.org/A5061150224)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现了一种基于层次化强化学习（两层 MAB）的多 AP 协调空间重用框架，用于同时调节 AP 的调度、发射功率与 MCS，以满足 QoS 并最大化网络吞吐量。

**💡 创新点**

创新点在于：① 将 QoS 与公平性融合进奖励函数，并通过两层 MAB 先动态调整 QoS 目标再细粒度优化功率/调度；② 兼顾吞吐量与公平性而非单纯提升总速率；③ 引入分层决策提升收敛速度与适应性。

**🔧 技术方法**

使用的技术包括：层次化强化学习、两层多臂赌博机（MAB）、PyTorch 实现的策略网络、仿真框架（Python）、无线信道模型（TGac NLOS）以及 Jain 公平指数与对数和等公平度量。

**📊 数据集**

数据集为仿真生成的网络拓扑：6 个 AP 均匀布置在 125m×75m 室内空间，STA 按 Poisson 分布 λ=0.002；采用 TGac NLOS 路径损耗模型，实验使用固定参数表中列出的功率等级、MCS 等。

**📈 对比分析**

与文献 <cit.> 中的基线算法对比，系统级仿真表明：① 聚合吞吐量提升约 10–20%；② Jain 公平指数从 0.709 提升至 0.92（比例公平）或 0.97（加权总和）；③ 相比基线，收敛更快，比例公平奖励能更快稳定。

**⚠️ 局限性**

局限性：仅在仿真环境下验证，未考虑真实硬件与动态流量变化；层次 MAB 的超参数需要手工调节；模型对 AP 数量、频谱占用等规模扩展的鲁棒性尚未充分评估。

---

## 256. Seed1.8 Model Card: Towards Generalized Real-World Agency

**arXiv ID:** 2603.20633 | [PDF](https://arxiv.org/pdf/2603.20633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 257. CREG: Compass Relational Evidence for Interpreting Spatial Reasoning in Vision-Language Models

**arXiv ID:** 2603.20475 | [PDF](https://arxiv.org/pdf/2603.20475v1)

**作者:** Kaizhen Tan `[一作]` `[通讯]` (Carnegie Mellon University), Kaizhen Tan (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练的解释框架（Compass Relational Evidence Graph，CREG），将多层梯度×激活（Grad×Act）归因投影到参考对象中心的极坐标系，得到可解释的方向证据分布；

**💡 创新点**

创新点在于：①使用对比梯度目标（目标与竞争 logits 差值）去除类无关特征；②多层归因聚合结合软max权重；③基于极坐标的高斯加权分箱生成方向分布；并提出三种新评估指标（DAE、EA、COS）衡量方向解释的准确性和因果可信度；

**🔧 技术方法**

技术方法包括：梯度×激活归因、对比目标归因、多层聚合、极坐标投影、Gaussian 加权分箱、方向分布归一化；

**📊 数据集**

使用 COCO‑Pairs（300 样本）和 VSR（240 样本）这两个基于 COCO 的空间关系数据集；

**📈 对比分析**

与 GradCAM、Attention Rollout、GradNorm、Integrated Gradients、单层 Grad×Act、几何 Oracle 等 7 个基线比较。CREG 在 Qwen2‑VL‑7B 上在 DAE 和 EA 上均优于所有基线，角误差降低 16°~20°，EA 提升 0.12；在 2B 模型上优势不明显；COS 证明方向解释具因果相关性（COS>0.4）；

**⚠️ 局限性**

局限性包括：仅在 Qwen2‑VL 家族评估；对较小模型效果不佳；只针对四个基向关系，未覆盖深度、距离等；极坐标投影假设对象边界可知，可能受定位误差影响；COS 仅使用灰色遮挡，未全面验证因果性；

---

## 258. ScaleEdit-12M: Scaling Open-Source Image Editing Data Generation via Multi-Agent Framework

**arXiv ID:** 2603.20644 | [PDF](https://arxiv.org/pdf/2603.20644v1)

**作者:** Guanzhou Chen `[一作]` (Shanghai Jiao Tong University), Hongjie Zhang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个基于多代理的分层框架和12M规模的高质量指令式图像编辑数据集ScaleEdit-12M。

**💡 创新点**

创新点包括：1）利用世界知识扩展源图像并通过检索+合成双路策略提升多样性；2）任务路由器和专用编辑代理实现按需生成多任务编辑；3）三维质量验证机制实现指令遵循、一致性和生成质量的全维度筛选。

**🔧 技术方法**

使用了开源大语言模型Qwen2.5-VL-72B进行任务路由、指令生成与质量评估；Qwen-Image、FLUX、Step1X-Edit等生成模型进行图像编辑；PaddleOCR、Text-Edit工作流实现文本编辑；指令重写模块提供知识推理支持。

**📊 数据集**

核心数据来源为COCO、OpenImages、SA-1B等公开数据集，辅以网络检索、文本/图像检索与合成扩展，最终构成12M条指令–原图–编辑图三元组。

**📈 对比分析**

在GEdit-EN-Full、ImgEdit-Full、KRIS-Bench和RISEBench等基准上微调UniWorld-V1和Bagel模型，均在多项指标上超过所有开源数据集并与商业API相当，显示出显著的性能提升。

**⚠️ 局限性**

局限性在于依赖现成的开源生成器导致视觉质量上限受限；跨23类任务的专家模型训练成本高昂；多轮对话式编辑功能尚未充分探索。

---

## 259. Current state of the multi-agent multi-view experimental and digital twin rendezvous (MMEDR-Autonomous) framework

**arXiv ID:** 2603.20575 | [PDF](https://arxiv.org/pdf/2603.20575v1)

**作者:** Logan Banker `[一作]` (Missouri University of Science and Technology), Jay Kamdar `[通讯]` (Missouri University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `51c0528b-f690-4182-ae60-bb5f046c276c` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一套名为MMEDR-Autonomous的多代理、多视角实验与数字孪生框架，用于自主航天器会合与对接，包括学习驱动的光学导航、基于强化学习的引导以及硬件在环测试平台。

**💡 创新点**

创新点在于：1）结合轻量级移动网络MobileNetV3与FPN实现高效6D姿态估计；2）通过稀疏速度奖励改进RL对接策略；3）使用贝叶斯优化自动调参提升学习稳定性；4）构建可扩展的多代理引导模型与控制屏障函数实现安全约束；5）提供完整的硬件在环实验台并进行初步验证。

**🔧 技术方法**

主要技术包括：深度确定性策略梯度（DDPG/D4PG）强化学习；移动卷积网络与FPN的光学姿态估计；数据增强与自监督在线域适应；英国控制屏障函数（CBF）与Kalman滤波器结合；以及机器人臂与Vicon摄像系统的硬件在环仿真。

**📊 数据集**

使用的数据集为SPEED+（含真实与模拟图像）和SPEC 2021竞赛的SPEED数据；此外通过Blender渲染生成Aura卫星的合成图像用于训练与评估。

**📈 对比分析**

与SPN、KRN等基线模型比较，MMEDR网络在位置误差上排在SPEC 2021前十，整体姿态误差相对较高但已明显优于仅用合成图像训练的对比模型；在RL对接实验中，Bayesian优化后单代理在接近目标的50m起始距离下成功率>95%，并能满足速度限制的对接任务。

**⚠️ 局限性**

局限性包括：1）姿态估计的旋转误差仍偏大，缺乏关键点/热图等辅助技术；2）缺乏在线或对抗训练，域间泛化仍待提升；3）多代理与复杂对接约束的实验尚未完成；4）硬件在环平台仍未完成全部校准与实时闭环验证。

---

## 260. Putnam 2025 Problems in Rocq using Opus 4.6 and Rocq-MCP

**arXiv ID:** 2603.20405 | [PDF](https://arxiv.org/pdf/2603.20405v1)

**作者:** Guillaume Baudart `[一作]` (IRIF), Jules Viennot `[通讯]` (IRIF)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用 Claude Opus 4.6 与 rocq-mcp 工具集在 Rocq 上实现多代理并行框架，自动证明 2025 年 Putnam 竞赛 12 题中的 10 题。

**💡 创新点**

提出编译优先 + 交互回退的 MCP 工具设计，并通过前期 miniF2F-Rocq 日志指导多代理协作，使得无专门训练即可跨证明助手完成复杂证明。

**🔧 技术方法**

核心技术包括 Claude Opus 4.6、rocq-mcp（编译、验证、自动化、交互式调试、库搜索等工具）、Claude Code 调度器以及多级子代理架构。

**📊 数据集**

使用 2025 年 Putnam 题目的自动化 Lean 版本与自然语言形式化作为正式化数据集，并以 miniF2F-Rocq 作为基准验证。

**📈 对比分析**

在 Rocq 上 10/12 题（约 83%）成功率，累计 5,542 行代码，平均每题约 400 行；但最难题 B5 需 46 小时；与 Lean 上的单模型相比，验证速度与成功率相近，但在资源消耗上更高。

**⚠️ 局限性**

主要局限包括：formalization 漏洞导致错误证明（如 A3）；对极难题资源消耗极大；交互工具使用有限，导致部分复杂子目标难以突破；并且多代理并行策略在遇到相同瓶颈时会出现资源浪费。

---

## 261. Modeling Epistemic Uncertainty in Social Perception via Rashomon Set Agents

**arXiv ID:** 2603.20750 | [PDF](https://arxiv.org/pdf/2603.20750v1)

**作者:** Jinming Yang `[一作]` (University of Electronic Science and Technology of China), Xinping Zhang `[通讯]` (Chengdu Neusoft University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种基于大语言模型的多智能体概率模型框架，用个体化的主观社会图、检索增强生成（RAG）与信任门控进行信息获取与社交互动，并通过贝叶斯融合更新各学生对同伴学术能力的信念，从而在真实课堂社交网络上模拟认知不确定性的传播与演化。

**💡 创新点**

创新点在于：① 将主观可见性作为核心假设，创建每位学生的主观社会图；② 将RAG限制在主观图范围内，防止全局信息泄露；③ 采用LLM生成的可信度门控来抑制噪声传播；④ 通过概率贝叶斯融合显式量化并更新认知不确定性，捕捉长期的误判累积。

**🔧 技术方法**

使用技术包括：大语言模型驱动的智能体、检索增强生成（RAG）+主观图约束、LLM生成的信任门控权重、基于高斯分布的贝叶斯更新、时间步序列模拟以及对比实验。

**📊 数据集**

使用数据集：12 个中国中学班级共 482 名学生，记录 6 次连续考试成绩及问卷友谊与心理量表，用以构建主观图与评估模型性能。

**📈 对比分析**

通过与随机、仅自我、线性回归、MLP、SGC/GCN、GAT、DeGroot 等基线对比，模型在第 6 次考试的 DPAE 为 0.124、Spearman ρ 为 0.876、Top‑3 识别率 0.278，优于绝大多数基线且在保持多样性的同时避免快速共识导致的误判。

**⚠️ 局限性**

局限性包括：尚未验证在更大、更复杂网络上的可扩展性；心理变量仅通过问卷映射与噪声参数化，缺乏细粒度行为特征；实验聚焦描述性分析，缺少干预或优化方案的探索。

---

## 262. Developing an ESG-Oriented Large Language Model through ESG Practices

**arXiv ID:** 2603.20480 | [PDF](https://arxiv.org/pdf/2603.20480v1)

**作者:** Gabriel Assis `[一作]` (Universidade Federal Fluminense), Aline Paes `[通讯]` (Universidade Federal Fluminense)

**通讯引用:** 653 | [OpenAlex ID](https://openalex.org/A5045549996)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于Qwen-3-4B的ESG专用语言模型，并通过LoRA与IRM实现参数高效微调；

**💡 创新点**

创新点在于将ESG原则嵌入训练、对齐与评估全过程，兼顾治理、环境与社会三维度；

**🔧 技术方法**

使用LoRA、Instruction-Residual Method (IRM)、检索增强推理、可读性评估等技术；

**📊 数据集**

以ESG-QA（87,261问答对）为主要训练与评测数据，来源于ESG-BERT语料库；

**📈 对比分析**

与Llama-3、Gemma-3等基线在零射击与KB增强场景下通过F1、BLEU、METEOR、BERTScore、ROUGE、可读性与碳排放等多指标对比，ESG-Qwen模型表现最佳且碳足迹低；

**⚠️ 局限性**

局限性包括工具驱动的知识库调用鲁棒性差、生成文本冗长以及缺乏人类评测验证。

---

## 263. Locally Coherent Parallel Decoding in Diffusion Language Models

**arXiv ID:** 2603.20216 | [PDF](https://arxiv.org/pdf/2603.20216v1)

**作者:** Michael Hersche `[一作]` (IBM), Abbas Rahimi `[通讯]` (IBM)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CoDiLA，一种在扩散语言模型中实现局部自回归的混合解码框架，用于在保持并行采样优势的同时提升文本局部连贯性，尤其适用于代码生成。

**💡 创新点**

核心创新在于：①将序列划分为块，在块级别上建模联合分布，从而理论上降低不可约的 NELBO；②设计软条件化接口，将 DLM 的边缘分布映射为 AR 模型的嵌入空间，允许极小的 AR 模型(≈0.6B)在块内完成高质量连贯解码；③结合静态与动态并行策略，利用置信度阈值实现部分块解码，兼顾速度与精度。

**🔧 技术方法**

技术方法包括：离散扩散过程、块级联合建模、软条件化与软嵌入、轻量级自回归模型、基于熵的置信度阈值解码、候选范围限制等；实现上使用 Dream-Coder-Instruct-7B 作为基础 DLM，Qwen3-0.6B 作为 AR 辅助模型。

**📊 数据集**

主要实验数据集包括 HumanEval、MBPP 及其 plus 版本、BigCodeBench（full 与 hard）。在这些编程任务上对比了基线 DLM、ADJUST 与 CoDiLA。

**📈 对比分析**

评估方式为 Pass@1 百分比与吞吐量（tokens/sec）。CoDiLA 在保持或提升 Pass@1 的同时实现了显著的吞吐量提升（静态并行 1×~2×，动态并行 >2×），在 Pareto 前沿上优于现有方法；与 ADJUST 相比，CoDiLA 在速度与准确性上均有更优表现。

**⚠️ 局限性**

局限性包括：对块大小的取舍仍需经验调优；软条件化依赖 DLM 边缘分布的质量；轻量 AR 模型虽小但仍需要额外训练；对极长序列的动态并行仍可能受到块范围限制导致的稳定性问题。

---

## 264. SWE-Next: Scalable Real-World Software Engineering Tasks for Agents

**arXiv ID:** 2603.20691 | [PDF](https://arxiv.org/pdf/2603.20691v1)

**作者:** Jiarong Liang `[一作]` (University of Waterloo), Wenhu Chen `[通讯]` (University of Waterloo)

**通讯引用:** 4969 | [OpenAlex ID](https://openalex.org/A5103103242)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 SWE-Next 框架，利用真实合并 PR 的提交对，执行测试验证后生成自验证的可执行 SWE 任务和高质量轨迹，支持大规模数据收集。

**💡 创新点**

创新点在于：①基于执行结果的严格筛选，确保任务实例严格改进且无回归；②引入可重用的 repo‑quarter 环境配置，显著降低环境构建与存储成本；③采用严格提交门控与无泄漏提示，提升轨迹信号质量。

**🔧 技术方法**

技术包括：Docker/容器化的可重用环境层、时间分段的 quarter‑profile 构建、基于测试执行的结果对比过滤、LLM 自动生成问题描述、OpenHands/工具调用的交互式推理框架。

**📊 数据集**

使用 3,971 个 Python 开源仓库的合并 PR 数据，挖掘 102,582 个候选提交对，最终生成 2,308 个自验证任务实例，并收集 3,700+ 轨迹供后训练。

**📈 对比分析**

通过在 SWE‑Bench‑Lite 与 SWE‑Bench‑Verified 上进行 pass@1 对比，SWE‑Next 在相同模型规模（7B/14B）下实现了约 17–30% 的性能提升，同时仅使用与现有系统相当或更少的轨迹量；在 14B 模型上甚至超过了 32B 模型的表现。

**⚠️ 局限性**

局限性包括：目前仅支持 Python 仓库，未覆盖其他语言；轨迹收集规模有限，尚未探索更大规模的采样；对极端环境或依赖冲突的处理仍依赖 fallback 机制。

---

## 265. JointFM-0.1: A Foundation Model for Multi-Target Joint Distributional Prediction

**arXiv ID:** 2603.20266 | [PDF](https://arxiv.org/pdf/2603.20266v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 266. Nevis Digital Twin: Photogrammetry and Immersive Visualization of Historical Sites

**arXiv ID:** 2603.20560 | [PDF](https://arxiv.org/pdf/2603.20560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 267. Rolling-Origin Validation Reverses Model Rankings in Multi-Step PM10 Forecasting: XGBoost, SARIMA, and Persistence

**arXiv ID:** 2603.20315 | [PDF](https://arxiv.org/pdf/2603.20315v1)

**作者:** Federico Garcia Crespi `[一作]` (Universidad Miguel Hernandez), Marina Alfosea Simon `[通讯]` (Universidad Miguel Hernandez)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在真实部署情境下采用滚动起点验证，比较了持久性、SARIMA和XGBoost对Elche城市背景站每日PM10的1–7日多步预测，并提出了预测可预测性边界H*概念。

**💡 创新点**

创新点在于将预测可预测性与相对持久性技能相结合，揭示评估设计（静态拆分 vs 滚动起点）可导致模型排名逆转，并为操作层面提供可解释的可预测性阈值。

**🔧 技术方法**

使用技术包括：持久性基准、季节性ARIMA（SARIMA）、梯度提升树XGBoost，以及train‑only预处理和滚动起点交叉验证。

**📊 数据集**

所用数据集为2017‑2024年Elche城市背景站的每日PM10观测，共约2350条有效记录。

**📈 对比分析**

通过相对持久性技能SS和预测可预测性边界H*进行比较；在静态拆分下XGBoost表现最佳，H*=7；在滚动起点下SARIMA保持正向技能并优于XGBoost，说明评估方式对模型性能评价具有显著影响。

**⚠️ 局限性**

局限性包括：仅评估单一站点和单一气溶胶指标；未进行显著性检验；H*仅相对持久性基准，未考虑多阈值或跨站泛化；以及模型参数未针对不同站点做适配。

---

## 268. Decoding the decoder: Contextual sequence-to-sequence modeling for intracortical speech decoding

**arXiv ID:** 2603.20246 | [PDF](https://arxiv.org/pdf/2603.20246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 269. RedacBench: Can AI Erase Your Secrets?

**arXiv ID:** 2603.20208 | [PDF](https://arxiv.org/pdf/2603.20208v1)

**作者:** Hyunjun Jeon `[一作]` (Korea Advanced Institute of Science and Technology), Jinwoo Shin `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6843 | [OpenAlex ID](https://openalex.org/A5102928677)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RedacBench基准，用于评估大语言模型在遵循安全政策下进行文本删改（redaction）的能力；

**💡 创新点**

创新点在于：①构建了基于政策的删改评估框架，利用8,053条可推断命题衡量安全性与实用性；②提供了多种删改策略（掩码、对抗删改、迭代删改）与多模型实验，形成可复现的性能基线；③公开了交互式在线演练平台，方便社区自定义数据与评估。

**🔧 技术方法**

技术手段包括：①语言模型（GPT‑5系列、GPT‑4.1、Gemini‑2.5、Claude‑Sonnet‑4、Qwen3等）实现删改与自动评估；②基于命题的评估框架，定义安全得分（TN/(TN+FP)）和实用得分（TP/(TP+FN)）；③使用自动评估器（GPT‑4.1‑mini）检测删改后命题是否仍可推断；

**📊 数据集**

数据集为514篇手写文本（个人、公司、政府来源），187条安全政策，以及对每篇文本标注的8,053条命题。

**📈 对比分析**

比较方法是对三种删改策略和多款语言模型的安全/实用得分进行对比。实验显示：更强模型和多轮对抗删改可提升安全性，但实用性明显下降；Claude‑Sonnet‑4在安全性与实用性上取得最优折中；开源模型Qwen3‑4b‑2507在安全/实用平衡上接近GPT‑4.1系列。

**⚠️ 局限性**

局限性包括：①仅提供经验性安全评估，缺乏正式差分隐私或形式化保证；②评估器可能因模型已预训练在原始文本上产生幻觉，导致安全/实用分数偏低；③数据来源受限于公开数据集，真实场景中可能存在更复杂或隐私更敏感的内容；

---

## 270. Domain-Specialized Tree of Thought through Plug-and-Play Predictors

**arXiv ID:** 2603.20267 | [PDF](https://arxiv.org/pdf/2603.20267v1)

**作者:** Xuanqi Gao `[一作]` (Xi'an Jiaotong University), Chao Shen `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 8770 | [OpenAlex ID](https://openalex.org/A5071566937)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了Domain‑Specialized Tree of Thought（DST‑ToT），通过在Tree of Thoughts框架中引入轻量化、可插拔的预测器，实现在推理过程中动态剪枝以提升效率。

**💡 创新点**

创新点在于利用可训练的预测器代替昂贵的LLM自评估，结合语义嵌入与一致性得分来做即时分数判断，从而在保持甚至提升准确率的同时显著降低计算开销。

**🔧 技术方法**

技术包括白盒LLM内部隐藏状态提取、基于语义和一致性特征的监督学习预测器、predict‑first‑thought机制以及自适应beam宽度控制的搜索策略。

**📊 数据集**

实验使用了多种公开基准数据集，涵盖数学推理（GSM8K、SVAMP、MATH‑500、Minerva‑Math）、一般推理（GPQA）和复杂逻辑推理（BBEH子任务如BoardgameQA、Boolean Expressions、Causal Understanding、Geometric Shapes）。

**📈 对比分析**

与CoT、标准ToT和DPTS等基线对比，DST‑ToT在Qwen3‑8B、Llama3.1‑8B‑Instruct和Gemma3‑12B‑it模型上实现了26–75%的token消耗下降，同时准确率与或优于传统ToT，显示出更优的效率‑准确率边界。

**⚠️ 局限性**

局限在于需要白盒访问LLM隐藏状态，限制了对闭源API模型的适用性；此外预测器可能在学习过程中无意中放大数据集中的偏见，尚未针对偏见进行显式缓解。

---

## 271. Linguistic Signatures for Enhanced Emotion Detection

**arXiv ID:** 2603.20222 | [PDF](https://arxiv.org/pdf/2603.20222v1)

**作者:** Florian Lecourt `[一作]` (LIRMM, Université de Montpellier), Konstantin Todorov `[通讯]` (LIRMM, Université de Montpellier)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在13个英语情感数据集上提取语言学特征（SEANCE），并将这些情感签名嵌入RoBERTa模型，以提升文本情感识别性能

**💡 创新点**

首次将跨语料库的可解释语言学签名与Transformer结合，证明其可作为稳定、可迁移的情感特征；同时提出两种特征融合策略（全局向量融合与词级门控融合）

**🔧 技术方法**

利用SEANCE（General Inquirer词典）提取特征，RoBERTa作为主干网络；实现特征向量拼接与门控门控机制；对比实验中还使用LLaMA 3.2的提示法

**📊 数据集**

13个公开英语情感数据集，包括GoEmotions、CARER、Affective Text、EmoContext等，主要聚焦GoEmotions作为基准进行训练与评估

**📈 对比分析**

与BERT SOTA基线和自行复现的RoBERTa基线对比；RoBERTa‑LexEnhance与RoBERTa‑EarlyFusion在GoEmotions上宏观F1分别提升至55.18%和54.98%，相对SOTA提高约+2.4个百分点；在CARER上亦实现近SOTA水平且在部分情感类别上超越SOTA

**⚠️ 局限性**

仅限英语且依赖SEANCE词典，未覆盖多语言场景；LLaMA提示法效果不佳，提示设计与多模态语义映射仍待改进

---

## 272. GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction

**arXiv ID:** 2603.20611 | [PDF](https://arxiv.org/pdf/2603.20611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 273. The Arrival of AGI? When Expert Personas Exceed Expert Benchmarks

**arXiv ID:** 2603.20225 | [PDF](https://arxiv.org/pdf/2603.20225v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 274. Rethinking Retrieval-Augmentation as Synthesis: A Query-Aware Context Merging Approach

**arXiv ID:** 2603.20286 | [PDF](https://arxiv.org/pdf/2603.20286v1)

**作者:** Jiarui Guo `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5753 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于动态合成的检索增强生成框架，利用对称合并与非对称合并动态重构检索上下文。

**💡 创新点**

核心创新是将检索上下文视为双目标优化问题，推出对称合并（恢复长尾弱信号）和非对称合并（熵引导去冗余），并设计层级并行合并提升效率。

**🔧 技术方法**

使用大语言模型（Qwen3-30B-Instruct）作为生成器，结合BM25/BGE检索器、熵估计、信息瓶颈理论与并行推理框架。

**📊 数据集**

在2WikiMQA、HotpotQA、MuSiQue、TriviaQA、QASPER等多跳推理与长文本问答基准上进行评测。

**📈 对比分析**

相较于传统RAG、RAPTOR、Tree-RAG等基线，平均提升EM/F1约12-15个百分点，尤其在多跳任务上显著优于BGE-reranker。

**⚠️ 局限性**

局限在于对检索质量高度依赖、低频词仍可能被忽略，以及在极短上下文（k=1）时仍需更细粒度的合成策略。

---

## 275. TRGS-SLAM: IMU-Aided Gaussian Splatting SLAM for Blurry, Rolling Shutter, and Noisy Thermal Images

**arXiv ID:** 2603.20443 | [PDF](https://arxiv.org/pdf/2603.20443v1)

**作者:** Spencer Carmichael `[一作]` (University of Michigan), Katherine A. Skinner `[通讯]` (University of Michigan)

**通讯引用:** 469 | [OpenAlex ID](https://openalex.org/A5002924029)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于3D高斯投影（3DGS）与微波测温相机的热惯性SLAM系统TRGS‑SLAM，能在存在模糊、滚动快门和固定模式噪声的极端热图像下实现精确定位、地图重建并支持离线图像恢复。

**💡 创新点**

① 微波测温相机专用的微波探测器模型融入3DGS渲染；② 连续时间B样条轨迹与双阶段IMU损失融合；③ 视角多样性矩阵驱动的高斯不透明度重置；④ 轨迹漂移校正与无姿态依赖的离线恢复。

**🔧 技术方法**

3D Gaussian Splatting渲染、B样条轨迹优化、IMU直接误差损失、微波探测器模型、视角多样性矩阵、梯度裁剪、离线重建、PyTorch、gsplat、LieTorch、PyPose等技术。

**📊 数据集**

TRNeRF 数据集（六条序列，慢/中/快速度 × 室内/室外场景）。

**📈 对比分析**

与 ORB‑SLAM3、DROID‑SLAM、DSM、DBA‑VIO、MonoGS、DM‑VIO、MASt3R‑SLAM、ROTIO 等多种视觉/热SLAM基线比较；在大多数噪声严重序列（FO、MI、FI）下，TRGS‑SLAM 成为唯一实现精确跟踪的算法，RMSE ATE 低于 5 cm，离线恢复与 TRNeRF 的 LPIPS 指标相当甚至略优。

**⚠️ 局限性**

对快速运动仍难以实现实时（1–4 FPS），慢速运动时 IMU 初始化导致漂移；高斯数量受限，难以扩展至大规模场景；未实现回环闭环和尺度一致性；对光照、雾/尘等物理遮挡的鲁棒性未全面评估。

---

## 276. Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance

**arXiv ID:** 2603.20584 | [PDF](https://arxiv.org/pdf/2603.20584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 277. Learning Communication Between Heterogeneous Agents in Multi-Agent Reinforcement Learning for Autonomous Cyber Defence

**arXiv ID:** 2603.20279 | [PDF](https://arxiv.org/pdf/2603.20279v1)

**作者:** Alex Popa `[一作]` (Royal Military College of Canada), Ranwa Al Mallah `[通讯]` (Polytechnique Montréal)

**通讯引用:** 301 | [OpenAlex ID](https://openalex.org/A5020513405)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在CybORG网络模拟环境中，使用CommFormer算法训练并评估了具有异构能力（不同观测与动作空间）的多智能体强化学习（MARL）模型，用于自主网络防御；

**💡 创新点**

创新点在于首次将CommFormer这类图网络通信算法应用于异构代理，展示其在异构通信和有限带宽下的高效协同与快速收敛；

**🔧 技术方法**

使用了CommFormer的Transformer架构，包含自注意力机制和稀疏通信图结构；

**📊 数据集**

实验数据集为CybORG（CybORG‑MARL）环境中的三种网络拓扑（同构、异构防火墙+子网、每主机单独防御）模拟的攻击与防御交互；

**📈 对比分析**

与基线DIAL算法对比，CommFormer在同构场景下收敛速度提升近4倍，平均奖励更稳定（-6.54±5.11 vs -7.9±14.39），在异构场景中虽有轻微奖励下降，但仍保持快速收敛；

**⚠️ 局限性**

局限性包括：在极度限制观测的主机级场景下仍出现训练不稳定，需精细的超参数和奖励调节；通信图稀疏参数的设定对性能影响显著，且在真实网络环境中的可迁移性尚待验证。

---

## 278. When Truth Misleads -- Phase-Aware Coherence Detection for Misinformation Correction Across Epistemic Communities

**arXiv ID:** 2603.20221 | [PDF](https://arxiv.org/pdf/2603.20221v1)

**作者:** Heimo Müller `[一作]` (Medical University of Graz), Andreas Holzinger `[通讯]` (Medical University of Graz)

**通讯引用:** 25564 | [OpenAlex ID](https://openalex.org/A5034657358)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了Phase‑Aware Coherence Detection（PACD）框架，将受众的知识取向量化为连续相位φ，并根据相位匹配设计多视角共识式干预，以对抗传统事实核查在不同受众上的“误导”效应。

**💡 创新点**

创新点在于：① 将四维信念维度（机构信任、科学认知、阴谋倾向、经验式认知）合成为连续相位；② 以相位相干性为依据调整干预源与受众的参考框架，消除传统干预的阶段依赖；③ 引入相位匹配的共识框架，降低回火率并提升干预鲁棒性。

**🔧 技术方法**

技术方法包括：问卷收集、四维信念指标标准化后线性组合并映射到[0,π]；在每个相位簇内随机分配传统核查、相位感知共识分析或对照组；计算信念变化、回火率、斜率；使用t检验、ANOVA、交互效应分析等统计手段进行比较。

**📊 数据集**

数据集：45名奥地利大学生（N=158观测），评估三条信息主张（5G健康、城市树木空气质量、mRNA疫苗），构成实验前后信念测量。

**📈 对比分析**

比较方法：单变量t检验、单因素ANOVA、相位簇斜率分析、回火率比例比较。结果显示传统核查平均无显著提升，且在高机构信任群体表现负效应；相位感知共识保持零至轻正效应，回火率显著降低（13.6% vs 39.3%），说明相位匹配干预在整体和阶段上更稳健。

**⚠️ 局限性**

局限性：样本量小（尤其C1组仅5人）；仅三条主张，缺乏高度争议性或身份相关性强的案例；单次会话未评估长期效果；相位坐标采用等权重线性组合且为单维，未验证更高阶或因子结构；缺乏过程性测量（阅读时间、眼动等）以直接检验相位干扰机制。

---

## 279. Multi-RF Fusion with Multi-GNN Blending for Molecular Property Prediction

**arXiv ID:** 2603.20724 | [PDF](https://arxiv.org/pdf/2603.20724v1)

**作者:** Zacharie Bugaud `[一作]` `[通讯]`, Zacharie Bugaud

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过结合12棵随机森林模型和两种GIN网络的rank平均融合，构建了Multi‑RF Fusion方法，在ogbg-molhiv scaffold split上获得0.8476 ROC‑AUC。

**💡 创新点**

首创将max_features从默认√d改为0.20以增强树对药效团特征的覆盖，并在GNN上先做10种种子平均再与RF融合，显著降低方差并提升稳定性。

**🔧 技术方法**

使用RDKit生成FCFP、ECFP、MACCS和哈希原子对指纹，20,000棵树的随机森林集成，5/8层带虚拟节点的GIN网络，最终通过rank平均和加权blend融合。

**📊 数据集**

使用ogbg‑molhiv 41,127分子二分类数据，采用官方scaffold split（32,901/4,113/4,113）。

**📈 对比分析**

与HyperFusion、PAS+FPs等基线比较，取得0.8476±0.0002的AUC，居榜首且方差最低（±0.0002），比HyperFusion提升约0.0001。

**⚠️ 局限性**

方法需要大量计算资源（240,000棵树+GNN深度集成），训练耗时约50小时；max_features和指纹组合仅在该任务验证，泛化性未知，提升幅度微小且主要依赖RF。

---

## 280. Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation

**arXiv ID:** 2603.20725 | [PDF](https://arxiv.org/pdf/2603.20725v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 281. Exponential Family Discriminant Analysis: Generalizing LDA-Style Generative Classification to Non-Gaussian Models

**arXiv ID:** 2603.20655 | [PDF](https://arxiv.org/pdf/2603.20655v1)

**作者:** Anish Lakkapragada `[一作]` `[通讯]` (Yale University), Anish Lakkapragada (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

提出了泛化的指数族判别分析（EFDA）模型，提供了闭式极大似然估计和线性似然比判别规则。

**💡 创新点**

创新点在于将 LDA 的生成式框架推广到任意指数族，实现了对非高斯数据的校准良好、统计效率最优的判别器。

**🔧 技术方法**

使用指数族理论、最大似然估计、Cramér–Rao 下界、以及 Lean 4 形式化证明技术。

**📊 数据集**

实验基于四类指数族分布（Weibull、Gamma、Exponential、Poisson）和合成数据集进行评估。

**📈 对比分析**

通过准确率和 Expected Calibration Error (ECE) 与 LDA、QDA、Logistic Regression 比较，EFDA 在保持相同准确率的同时，ECE 下降 2–6 倍，且误差收敛至零。

**⚠️ 局限性**

局限在于模型假设所有类别共享同一指数族结构，若真实分布不属于该族则仍可能出现偏差；此外对高维稀疏数据的实用性待进一步验证。

---

## 282. Towards Practical Multimodal Hospital Outbreak Detection

**arXiv ID:** 2603.20536 | [PDF](https://arxiv.org/pdf/2603.20536v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 283. STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction

**arXiv ID:** 2603.20284 | [PDF](https://arxiv.org/pdf/2603.20284v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 284. Reason-to-Transmit: Deliberative Adaptive Communication for Cooperative Perception

**arXiv ID:** 2603.20308 | [PDF](https://arxiv.org/pdf/2603.20308v1)

**作者:** Aayam Bansal `[一作]` (Synthetic Sciences), Ishaan Gangwani `[通讯]` (Synthetic Sciences)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出一种名为Reason-to-Transmit (R2T) 的框架，在多智能体协作感知中使用轻量级变压器推理模块来做出通信决策，以提升在带宽受限情况下的感知性能。

**💡 创新点**

核心创新在于将“推理-先决策”范式引入通信选择，使每个智能体在发送特征时同时考虑本地感知、邻居可能缺失的信息、场景上下文和剩余带宽，区别于以往仅基于置信度或阈值的反应式策略。

**🔧 技术方法**

技术实现包括：1）基于4层卷积的BEV特征编码器；2）以8×8网格划分特征，形成64个候选消息；3）一个2层预归一化自注意力变压器（约0.26M参数）作为推理模块；4）带门控的交叉注意力融合模块；5）端到端带宽感知损失训练。

**📊 数据集**

实验数据集：在自定义的二维多智能体BEV感知环境中生成的500个训练、80个验证、150个测试场景，包含20个目标、可调遮挡（低/中/高）和高斯噪声。没有使用公开的3D V2X 真实数据集。

**📈 对比分析**

与9个基线（包括无通信、始终发送、随机、置信度、不确定度、Where2Comm、IC3Net门控、学习稀疏掩码以及oracle）在5个随机种子上比较。结果显示任何形式的通信都能使AP提升约58%，低带宽（10%）下R2T的AP约为0.229，略低于学习稀疏掩码（0.232）但与IC3Net相近；在高遮挡条件下R2T达到0.205，几乎与oracle相同，显示出在信息不对称最严重时的优势。

**⚠️ 局限性**

局限性包括：1）实验仅在二维仿真环境中进行，未验证在真实3D LiDAR/摄像头数据上的表现；2）假设已知精确的智能体位置和同步通信，忽略定位误差和异步通信；3）智能体数量仅为4，未评估大规模扩展；4）推理模块虽轻量，但未实现完整的LLM推理，缺乏自然语言解释能力；5）对定位噪声和更复杂环境的鲁棒性尚未充分测试。

---

## 285. MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning

**arXiv ID:** 2603.20586 | [PDF](https://arxiv.org/pdf/2603.20586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 286. Characterizing the ability of LLMs to recapitulate Americans' distributional responses to public opinion polling questions across political issues

**arXiv ID:** 2603.20229 | [PDF](https://arxiv.org/pdf/2603.20229v1)

**作者:** Eric Gong `[一作]` (Harvard University), Bruce Schneier `[通讯]` (Harvard University)

**通讯引用:** 18409 | [OpenAlex ID](https://openalex.org/A5037770347)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种直接让大语言模型（LLM）输出政治议题问卷答案分布的框架（Direct Distribution, DD），并与传统单个个体模拟问卷（Single Individual, SI）进行对比；

**💡 创新点**

创新点在于：①将分布预测直接嵌入提示中，避免多次模拟个体回答所导致的高成本与低多样性；②构建可预测LLM性能的回归模型，仅依据问题文本嵌入和目标人群特征即可估计预测误差；

**🔧 技术方法**

技术包括：大语言模型提示工程（Chain‑of‑Thought、Distribution Reminder）、结构化JSON输出、向量嵌入（OpenAI Embeddings）与UMAP可视化、Bayesian Ridge回归与梯度提升回归等机器学习方法；

**📊 数据集**

使用美国《Cooperative Election Study》（CES）2022年数据，挑选84个多选政治议题问题，按种族、性别、意识形态共20个组合，总计1680个问卷‑人群组合；

**📈 对比分析**

比较方法：对每个组合使用MD、SDD、NEMD三种分布差异度量；实验显示DD在72.7%（MD）、77.3%（NEMD）以及标准差预测方面优于SI，且DD的误差可通过回归模型实现0.5以上的预测R²；

**⚠️ 局限性**

局限性包括：仅评估单一LLM（GPT‑4o‑mini）和单一调查数据集（CES 2022），可能存在训练数据泄漏；人群分组过于粗糙（非二元性别未纳入），问题覆盖面有限；未探究不同温度、Prompt细化或其他LLM模型的影响。

---

## 287. ReBOL: Retrieval via Bayesian Optimization with Batched LLM Relevance Observations and Query Reformulation

**arXiv ID:** 2603.20513 | [PDF](https://arxiv.org/pdf/2603.20513v1)

**作者:** Anton Korikov `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**通讯引用:** 6544 | [OpenAlex ID](https://openalex.org/A5028174137)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将检索任务视为贝叶斯优化问题，利用LLM生成的查询改写初始化多模态高斯过程后，迭代主动采样文档批次，使用LLM进行文档相关性打分并更新后验，最终实现检索与排名的协同优化。

**💡 创新点**

创新点在于：①将多条查询改写作为多峰先验，为检索提供更丰富的初始信息；②采用贝叶斯优化主动采样批次，并提出MMR式批量多样化策略，突破传统top‑k检索的局限；③在保持低延迟的前提下显著提升召回率和NDCG。

**🔧 技术方法**

技术手段包括多模态高斯过程（GP）、贝叶斯优化（BO）与主动学习、LLM查询改写、LLM点级/批量相关性评分、MMR式批量多样化采样，以及Gemini‑2.5‑Flash‑Lite和GPT‑5.2等大型语言模型。

**📊 数据集**

实验数据集为BEIR五个基准：Robust04、TREC‑NEWS、TREC‑COVID、SciFact、NFCorpus（也与BM25、SPLADE等传统检索基准对比）。

**📈 对比分析**

与稀疏检索（BM25、SPLADE）、稠密检索、LLM重排序（LW、PW）以及基线的BO‑Top‑k等方法比较，ReBOL在召回@100和NDCG@10上明显优于LLM reranker；例如在Robust04上从35.0%提升至46.5%，并且与LLM reranker保持相近的延迟。

**⚠️ 局限性**

局限性包括：实验仅涵盖五个数据集、两款LLM和有限的k值；计算成本高，难以大规模部署；LLM可能带来偏见、对抗性提示和相关性判断不一致的风险，且未对更大规模或跨领域情境进行评估。

---

## 288. Me, Myself, and $π$ : Evaluating and Explaining LLM Introspection

**arXiv ID:** 2603.20276 | [PDF](https://arxiv.org/pdf/2603.20276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 289. Multi-Agent Debate with Memory Masking

**arXiv ID:** 2603.20215 | [PDF](https://arxiv.org/pdf/2603.20215v1)

**作者:** Hongduan Tian `[一作]` (Hong Kong Baptist University), Bo Han `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 10583 | [OpenAlex ID](https://openalex.org/A5100781698)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多智能体辩论框架在面对上一轮错误记忆时的脆弱性，并提出通过记忆屏蔽提升鲁棒性的算法；

**💡 创新点**

首次将理论分析与记忆屏蔽机制相结合，证明错误记忆对后续推理影响并提出两种屏蔽策略（主观评估与客观困惑度）；

**🔧 技术方法**

使用多智能体辩论（MAD）、链式思考、主观/客观屏蔽（基于困惑度）等技术；

**📊 数据集**

在数学推理任务（GSM8K、MATH、AIME24/25）和语言理解任务（MMLU-Pro）上进行实验；

**📈 对比分析**

与CoT、CoT‑SC和传统MAD对比，记忆屏蔽版本在大多数数据集上提升0.6–9.0%的准确率，且客观屏蔽在强模型上显著优于主观屏蔽，代价是更高的token消耗；

**⚠️ 局限性**

局限性包括需要额外的自评或困惑度计算导致token消耗上升、对模型能力依赖强、在过多回合或弱模型下性能不一定提升。

---

## 290. Beyond Token Eviction: Mixed-Dimension Budget Allocation for Efficient KV Cache Compression

**arXiv ID:** 2603.20616 | [PDF](https://arxiv.org/pdf/2603.20616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 291. Policies Permitting LLM Use for Polishing Peer Reviews Are Currently Not Enforceable

**arXiv ID:** 2603.20450 | [PDF](https://arxiv.org/pdf/2603.20450v1)

**作者:** Rounak Saha `[一作]` (Indian Institute of Science), Danish Pruthi `[通讯]` (Indian Institute of Science)

**通讯引用:** 631 | [OpenAlex ID](https://openalex.org/A5056959868)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文收集了超过五万条同行评审文本，涵盖从纯人类撰写、各种程度的LLM协作生成、以及人工化处理后的样本，随后对多款现有AI文本检测器进行系统评估，探讨其在多级人机协作评审中的可执行性。

**💡 创新点**

创新点在于：①构建针对同行评审场景的多级AI协作数据集；②从技术与政策双重角度评估“仅润色”政策的可执行性；③尝试利用论文原稿上下文、域内相似度以及监督学习等“同行评审特有”信号提升检测效果。

**🔧 技术方法**

技术手段包括：零射击无监督检测（LogLikelihood、Fast-DetectGPT、Binoculars），商业检测API（Pangram、GPTZero），论文上下文条件化概率、参考文档相似度（软n-gram/关键点匹配），以及基于风格特征和RoBERTa表征的监督分类器。

**📊 数据集**

使用数据集：由PeerRead预ChatGPT时代的1,499条人类评审组成；18,340条基于GPT‑4o、Llama‑3.3‑70B‑Instruct生成的多级评审；27,429条来自GPT‑5、Gemini‑2.5‑pro、Gemma‑3‑27b‑it、Qwen‑3‑30B‑thinking、Llama‑3.1‑70B‑instruct的多模型生成样本；另外2,000条使用Undetectable AI对AI‑生成和AI‑润色评审进行人工化处理。

**📈 对比分析**

评估方法以TPR/FPR为核心，对“AI生成”(ai-hi)与“仅润色”(human‑polish)两类进行区分。结果显示，商业检测器在纯AI评审上达95–98%召回且FPR≈0%；但在仅润色评审上，Pangram和GPTZero仍出现≈3% FPR，Fast‑DetectGPT等零射击模型在较新模型下召回率降至70%且FPR上升；上下文化与相似度提升虽能略微提高召回，却导致FPR显著上升，监督分类器在内测上表现优异（≈99%召回、<2% FPR），但对未知模型的泛化大幅下降。

**⚠️ 局限性**

局限性包括：检测器在细粒度人机协作文本上的高误判率；“仅润色”政策在现有技术下难以执行；模型迁移与新LLM的出现导致检测器性能不可预测；人工化处理进一步削弱检测效果；并且存在潜在训练数据污染问题，整体误报率过高可能导致合规评审被错误指控。

---

## 292. OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis

**arXiv ID:** 2603.20278 | [PDF](https://arxiv.org/pdf/2603.20278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 293. SNAP: Speaker Nulling for Artifact Projection in Speech Deepfake Detection

**arXiv ID:** 2603.20686 | [PDF](https://arxiv.org/pdf/2603.20686v1)

**作者:** Kyudan Jung `[一作]` (KAIST AI), Cheonbok Park `[通讯]` (KAIST AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了通过正交投影消除SSL特征中的说话人信息，从而增强深度伪造检测的轻量化方法。

**💡 创新点**

创新点在于将说话人子空间定义并消除，显著提升了说话人无关的合成痕迹辨别。

**🔧 技术方法**

使用WavLM‑Large提取特征，进行双层隐藏状态拼接、均值池化、L2归一化，并通过PCA得到说话人子空间，再用正交投影消除，最后用逻辑回归分类。

**📊 数据集**

主要数据集包括ASVspoof 2019/2021 LA/DF、In‑The‑Wild、CosyVoice2、F5‑TTS以及LibriSpeech构造的跨说话人对比集。

**📈 对比分析**

与基线（如RawNet2、AASIST、WavLM‑ECAPA‑TDNN）相比，SNAP在ASV19 LA的EER降至0.35%（比WavLM‑ECAPA低56%），在ASV21 DF降至5.42%，在In‑The‑Wild降至15.39%，并在跨域合成语音上实现零错误。

**⚠️ 局限性**

局限性在于仅针对SSL特征的说话人消除，未探索对更复杂声纹或非语音任务的适用性，且依赖PCA维度选择。

---

## 294. FastPFRec: A Fast Personalized Federated Recommendation with Secure Sharing

**arXiv ID:** 2603.20283 | [PDF](https://arxiv.org/pdf/2603.20283v1)

**作者:** Zhenxing Yan `[一作]` (Beijing Jiaotong University), Zhihui Gao `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 1317 | [OpenAlex ID](https://openalex.org/A5102714122)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种三层结构的联邦推荐框架FastPFRec，在保证用户隐私的同时显著加速GNN模型收敛。

**💡 创新点**

引入FastGNN异步更新策略减少项嵌入频率、使用可信节点进行安全聚合与异常检测，并结合局部差分隐私和图扰动实现多层防护。

**🔧 技术方法**

图神经网络、联邦学习、局部差分隐私（LDP）、图扰动、可信节点中间聚合以及异常检测。

**📊 数据集**

Yelp、Kindle、Gowalla-100k、Gowalla-1m。

**📈 对比分析**

与中央模型及多种联邦推荐基线（FedMF、FedNCF、FedGNN、PerFedRec、PerFedRec++等）在HR@10和NDCG@10上进行对比，FastPFRec在所有数据集上获得更高精度，训练轮次减少32%，训练时间降低34%。

**⚠️ 局限性**

依赖可信节点的“诚实但好奇”假设，无法在完全去中心化环境部署，且对高级逆向攻击（模型反演、成员推断）防护仍有限。

---

## 295. Solver-Aided Verification of Policy Compliance in Tool-Augmented LLM Agents

**arXiv ID:** 2603.20449 | [PDF](https://arxiv.org/pdf/2603.20449v1)

**作者:** Cailin Winston `[一作]` (University of Washington), René Just `[通讯]` (University of Washington)

**通讯引用:** 4118 | [OpenAlex ID](https://openalex.org/A5088079823)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于 SMT 求解器的工具调用政策检查框架，在工具增强型 LLM（TaLLM）运行时拦截并验证工具调用是否符合先前用 SMT 约束编码的自然语言政策。

**💡 创新点**

创新点在于将自然语言政策转换为可求解的 SMT 约束，并在执行前进行形式化检查，提供了对政策违规行为的硬性保证；同时采用 LLM 辅助、人工审核的混合编码流程，解决了纯自动化翻译的语义不完整和约束不足问题。

**🔧 技术方法**

核心技术包括：SMT 求解器（Z3）进行约束求解；LLM（GPT‑4/4o）辅助生成与修正 SMT 代码；工具调用拦截器与最小不满足核心（unsat core）反馈；以及基准测试框架 τ^2‑bench 用于评估。

**📊 数据集**

使用了 τ^2‑bench 航空域数据集（50 个任务、13 个工具、1242 词政策文档），以及 GPT‑4.1 作为 TaLLM、GPT‑4o 进行工具调用验证。

**📈 对比分析**

与不做检查的基线进行对比，采用 pass^k（k=1..4）指标评估。结果显示，政策检查将违规工具调用比例从约 50% 降至 29%，且任务成功率保持相近甚至略有提升，且对多次执行的鲁棒性提升了 26%。

**⚠️ 局限性**

局限性包括：政策编码仍需人工或半自动审校，自动化翻译往往出现语法错误、语义缺失或约束松散；政策检查可能导致部分合法调用被误拦截，影响召回率；对非常长或复杂的政策文档，编码与检查仍面临挑战。

---

## 296. The Multiverse of Time Series Machine Learning: an Archive for Multivariate Time Series Classification

**arXiv ID:** 2603.20352 | [PDF](https://arxiv.org/pdf/2603.20352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 297. MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages

**arXiv ID:** 2603.20732 | [PDF](https://arxiv.org/pdf/2603.20732v1)

**作者:** Anri Lombard `[一作]` (University of Cape Town), Jan Buys `[通讯]` (University of Cape Town)

**通讯引用:** 1950 | [OpenAlex ID](https://openalex.org/A5033945505)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了面向南非十一官方书面语言的多语预训练语料库 MzansiText，并从零训练出 125M 参数的 decoder‑only 语言模型 MzansiLM，随后在多任务、单语/多语微调三种策略下对多种自然语言处理任务进行了系统评测。

**💡 创新点**

创新点在于：①提供了完整可复现的南非语言预训练语料库与对应模型；②首次在 125M 参数规模下展示 decoder‑only 模型在生成与序列标注任务上的竞争力；③对比了单语任务微调、多语任务微调与跨任务指令微调三种策略，揭示其在不同任务与语言上的优势与不足。

**🔧 技术方法**

技术方法包括：MobileLLM 125M 的 decoder‑only 架构；65k BPE tokenizer；LoRA 参数高效微调（rank 128、α=256）；多阶段预处理 pipeline（语言识别、HTML 去除、去重、去噪、安全过滤、短文本拼包）；以及多任务指令微调的加权采样混合。

**📊 数据集**

预训练语料来源于 mC4、CulturaX、WURA、Glot500、NCHLT、CC100、ParaCrawl、Inkuba-Mono 等公开数据；下游任务涵盖 MasakhaNEWS、SIB‑200、INJONGO Intent、MasakhaPOS、MasakhaNER 2.0、AfriHG、T2X、Belebele、IrokoBench 等多种任务与语言。

**📈 对比分析**

通过在每个任务上比较 0‑shot、1‑shot、3‑shot、单语微调、多语微调与跨任务指令微调的表现，使用 macro‑F1、BLEU、ROUGE‑L、token accuracy、normalized accuracy 等指标。结果显示：单语微调在生成与序列标注任务上最优；多语微调在 Bantu 语言的主题分类中效果最佳；但在分类、结构化预测、推理等任务中仍显落后于 encoder‑only/encoder‑decoder 大模型，指令微调虽提升覆盖率但易稀释任务特定信号。

**⚠️ 局限性**

局限性包括：①模型仅 125M 参数，无法充分展示大规模模型的推理与长序列能力；②预训练语料在英语和南非荷兰语上偏重，导致低资源语言的表现受限；③下游评测仅覆盖 8 种语言，其他 3 种官方语言的评测不足；④跨任务指令微调混合任务会稀释单一任务的训练信号；⑤评测指标以自动化指标为主，未能全面捕捉生成质量与流畅度。

---

## 298. From Attention to Dialogue: Does Audience Engagement Reinforce Constructive Cross-Party Communication?

**arXiv ID:** 2603.20549 | [PDF](https://arxiv.org/pdf/2603.20549v1)

**作者:** Ahana Biswas `[一作]` (University of Pittsburgh), Yu-Ru Lin `[通讯]` (University of Pittsburgh)

**通讯引用:** 5051 | [OpenAlex ID](https://openalex.org/A5042159546)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析美国议员在推特上跨党派互动的受众反应及其对未来沟通频率和风格的影响

**💡 创新点**

首次将跨党派互动视为动态反馈循环，揭示受众参与度如何塑造政客的后续沟通策略

**🔧 技术方法**

混合效应回归、1:1最近邻匹配、RoBERTa文本嵌入、情绪与毒性检测、低可信度URL识别

**📊 数据集**

2020-2021年美国州议员推特数据（约110万条推文）

**📈 对比分析**

对比匹配对照组，混合效应模型显示跨党派提及得到更多关注后，议员更倾向于继续使用因果、主观和积极情感的表达，性能优于传统单变量回归且结果稳健

**⚠️ 局限性**

局限性包括只分析推特且时间段特定，缺乏受众身份信息，无法区分党内与党外受众驱动，跨平台和跨国普适性未验证

---

## 299. SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval

**arXiv ID:** 2603.20738 | [PDF](https://arxiv.org/pdf/2603.20738v1)

**作者:** Qunjie Huang `[一作]` (Yunnan University), Weina Zhu `[通讯]` (Yunnan University)

**通讯引用:** 628 | [OpenAlex ID](https://openalex.org/A5101802592)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并评估了一种无标签的测试时校准头 SATTC，能够显著提升跨受试 EEG‑图像检索的 Top‑1 与 Top‑5 准确率。

**💡 创新点**

创新点在于将受试自适应白化、适应性 CSLS 与基于互相最近邻、双向 top‑k 以及类别受欢迎度的结构化专家通过 Product‑of‑Experts 方式融合，实现对受试偏移与 hubness 的联合校正。

**🔧 技术方法**

使用的技术包括受试自适应白化（SAW）、自适应 CSLS、互相最近邻、双向 top‑k 排名、类别受欢迎度统计以及 Product‑of‑Experts (PoE) 校准。

**📊 数据集**

实验数据集为 THINGS‑EEG，包含 10 名受试者和 200 个目标类别，采用严格的 leave‑one‑subject‑out（LOSO）评估。

**📈 对比分析**

与原 ATM 基线、标准化基线、SAW、CSLS、Ada‑CSLS 等方法对比，SATTC 在 LOSE 下将 Top‑5 准确率从 30.5% 提升至 38.4%，Top‑1 从 9.2% 提升至 14.8%，并显著降低 hubness 及类别不平衡。

**⚠️ 局限性**

局限性包括仅在 THINGS‑EEG 数据集上验证、结构专家手工设计、需要批量预计算相似矩阵，以及尚未在其他数据集或重建任务上进行验证。

---

## 300. Measuring Reasoning Trace Legibility: Can Those Who Understand Teach?

**arXiv ID:** 2603.20508 | [PDF](https://arxiv.org/pdf/2603.20508v1)

**作者:** Dani Roytburg `[一作]` (Carnegie Mellon University), Daphne Ippolito `[通讯]` (Carnegie Mellon University)

**通讯引用:** 5105 | [OpenAlex ID](https://openalex.org/A5022994077)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种统一框架，用效率与传递两维度衡量语言模型推理轨迹的可读性并对12种推理模型进行大规模评估

**💡 创新点**

创新点在于引入了“传递效用”度量，揭示推理准确性与可教学性之间的反向关系，并绘制了可读性Pareto前沿

**🔧 技术方法**

使用了基于词元计数、语义冗余（句子嵌入余弦相似度）和回溯检测的效率指标，以及基于弱学生模型在逐步接收教师推理前缀时准确率的传递效用曲线

**📊 数据集**

评估数据集包括MATH、GPQA和Connections三种不同推理任务，共计约99.5K条推理轨迹

**📈 对比分析**

与传统的准确率和奖励模型得分对比发现高准确率模型在传递效用上往往排名靠后，说明仅优化最终答案不足；同时不同模型在效率与传递效用上呈现明显权衡，未出现“万能可读模型”

**⚠️ 局限性**

主要局限包括：仅在推理时评估传递效用而非微调过程；人类评估规模有限，无法完全验证效率指标的认知负荷；回溯检测依赖LLM判别器，可能存在误判

---

## 301. Pricing Innovation Under Latency Constraints: A Mean-Field Analysis of Coded Payload Delivery

**arXiv ID:** 2603.20426 | [PDF](https://arxiv.org/pdf/2603.20426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 302. HCAG: Hierarchical Abstraction and Retrieval-Augmented Generation on Theoretical Repositories with LLMs

**arXiv ID:** 2603.20299 | [PDF](https://arxiv.org/pdf/2603.20299v1)

**作者:** Yusen Wu `[一作]`, Xiaotie Deng `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HCAG（Hierarchical Code/Architecture-guided Agent Generation）框架，通过离线层级抽象构建理论-架构-实现的多层语义知识库，并在在线阶段进行层级检索和架构-模块分层生成，同时引入多智能体讨论提升代码质量。

**💡 创新点**

创新点包括：
1) 两阶段层级化处理（递归抽象+层级检索），突破传统 RAG 的平面检索限制；
2) 将理论文本与代码实现对齐，形成结构化知识库；
3) 通过多智能体讨论实现协同纠错；
4) 通过节点压缩与理论成本分析证明层级抽象的成本最优性。

**🔧 技术方法**

技术手段：LLM 递归抽象、向量检索+结构化相似度、层级检索+架构-模块生成、压缩节点机制、基于多智能体的辩论框架、后期微调（post‑training）等。

**📊 数据集**

使用数据集：
- AGT 代码仓库（OpenSpiel、GLEE 等）
- 算法博物馆理论文献与教材
- 通过 HCAG 生成的 Theory‑Implementation Alignment Set（结构化知识库）
- 通过情境合成得到的 Scenario Construction Set（评测任务）。

**📈 对比分析**

与平面 RAG、ARS、RepoCoder、PKG、LLM_understand 等基线在 100 个 AGT 任务上对比，指标为 Code Quality (CQ)、Text Similarity (TS)、Requirement Pass Rate (RPR) 以及 Average Time (AT)：
- HCAG：CQ 0.788，TS 0.525，RPR 0.60，AT 3235s；
- 最佳基线（LLM_understand）：CQ 0.744，TS 0.359，RPR 0.48，AT 1691s；
- 其他基线表现更差。总体来看，HCAG 在质量和通过率上显著优于基线，虽耗时更高。

**⚠️ 局限性**

局限性：
- 前置层级抽象需要一次性高成本，且对仓库变更缺乏增量更新机制；
- 生成时间较长，尤其在深层检索或多智能体讨论时；
- 对 LLM 规模和能力高度敏感，低端模型效果仍受限；
- 目前缺乏正式验证和符号推理的支持，难以保证生成代码的绝对正确性。

---

## 303. Stability of AI Governance Systems: A Coupled Dynamics Model of Public Trust and Social Disruptions

**arXiv ID:** 2603.20248 | [PDF](https://arxiv.org/pdf/2603.20248v1)

**作者:** Jiaqi Lai `[一作]` (Nanyang Technological University), Weihong Huang `[通讯]` (Nanyang Technological University)

**通讯引用:** 1390 | [OpenAlex ID](https://openalex.org/A5035799005)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

本文构建了一个耦合动力学框架，描述公共治理中 AI 系统的信任与争议事件如何相互影响，并给出了平衡解与稳定性判据；

**💡 创新点**

首次将 Friedkin‑Johnsen 意见动态与离散时间 Hawkes 过程耦合，提出双向反馈机制，推导出信任崩溃的谱条件 ρ(J_{2n})<1，并揭示网络拓扑对崩溃阈值的影响；

**🔧 技术方法**

采用线性离散时间动力学、Friedkin‑Johnsen 模型、Hawkes 过程、谱分析、矩阵不变性与数值仿真技术；

**📊 数据集**

未使用真实数据集，仅通过人工设定的参数与随机生成的网络进行仿真；

**📈 对比分析**

未与现有方法做直接性能比较，主要通过数值实验展示在不同参数与网络结构下系统的收敛与崩溃行为，评估指标为稳定性边界和平均信任水平；

**⚠️ 局限性**

模型假设线性、参数同质、无噪声，缺乏经验校准，无法捕捉非线性饱和与心理记忆非指数衰减等实际复杂性。

---

## 304. CFNN: Continued Fraction Neural Network

**arXiv ID:** 2603.20634 | [PDF](https://arxiv.org/pdf/2603.20634v1)

**作者:** Chao Wang `[一作]` (Shanghai University), Ruiyi Ding `[通讯]` (Shanghai University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于连续分式的神经网络（CFNN）框架，克服了多层感知机在高曲率、局部奇异和尖锐过渡函数上的谱偏差，实现了高精度、参数节省的科学建模。

**💡 创新点**

创新点在于将连续分式的表达效率与投影多项式参数化相结合，构造了可控的分式递归结构，并设计了Boost、MoE、Hybrid 三种变体，实现在参数极小的前提下指数级收敛、梯度稳定与自适应局部建模。

**🔧 技术方法**

使用了投影-多项式参数化、正则化分式（分母平方+γ）、梯度裁剪、RBF‑门控专家分区、并行 rational 单元、残差/Boost 加权学习以及严格的梯度/ Lipschitz 稳定性分析。

**📊 数据集**

实验涵盖合成数学函数（Runge、比值、嵌套分式）、物理模型（椭圆函数、贝塞尔函数、Bessel函数）、真实数据集（CIFAR‑10、IMDb、Quora、UCI Energy‑Efficiency、信用卡欺诈、波形等）以及多模态文本与图像。

**📈 对比分析**

通过 RMSE、频谱误差、参数‑精度 Pareto 前沿、噪声鲁棒性、SHAP 解释一致性等多维度指标与 MLP、SIREN、KAN、CoFrNet 等基线对比；在相同参数规模下，CFNN‑Hybrid 等实现 1‑2 个数量级参数优势、47 倍噪声抑制、无频率偏差的全域拟合以及更好的科学解释。

**⚠️ 局限性**

局限在于深层分式递归仍易产生梯度爆炸、对极端高频或真实奇点的建模受限；缺少针对序列/图结构的专门模块；正则化参数 γ 需要手工设定，且大规模预训练与跨领域迁移实验尚未展开。

---

## 305. A Multihead Continual Learning Framework for Fine-Grained Fashion Image Retrieval with Contrastive Learning and Exponential Moving Average Distillation

**arXiv ID:** 2603.20648 | [PDF](https://arxiv.org/pdf/2603.20648v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. AgentComm-Bench: Stress-Testing Cooperative Embodied AI Under Latency, Packet Loss, and Bandwidth Collapse

**arXiv ID:** 2603.20285 | [PDF](https://arxiv.org/pdf/2603.20285v1)

**作者:** Aayam Bansal `[一作]` (Synthetic Sciences), Ishaan Gangwani `[通讯]` (Synthetic Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向协作具身 AI 的通信失效基准和评估协议，系统评估了六种通信失效维度对三类任务（协同感知、导航、搜索）的影响。

**💡 创新点**

创新点在于：①完整覆盖六种现实通信失效（延迟、丢包、带宽崩溃、异步更新、旧记忆、冲突感知）；②设计轻量级网格世界任务族来隔离通信效应；③引入冗余消息编码与时延感知融合的通用通信包装；④提出统一的评估指标（NPD、AURC、排名稳定性、失效模式分类）。

**🔧 技术方法**

技术包括：可参数化的失效通道模型、基于共享目标编码的导航和搜索任务、冗余复制与指数衰减权重融合、统计评估工具与可视化曲线。

**📊 数据集**

使用轻量级20×20网格世界模拟，无外部公开数据集；任务在该模拟中实现，实验规模为4个智能体，30个episode。

**📈 对比分析**

与五种通信策略（无通信、全量通信、压缩通信、事件触发通信、提出的冗余编码）对比。结果显示：在通信依赖强的导航任务下，冗余编码在80%丢包时将完成率从10%提升至约22%；感知任务对内容破坏极度敏感（>85% F1下降），而对传输失效不敏感；搜索任务受旧记忆影响最大。

**⚠️ 局限性**

局限包括：任务仅为网格世界，缺乏真实感知与传感器噪声；感知任务在理想通信下无提升，难以检验通信必要性；冗余编码在带宽压力大时开销显著；未验证在更复杂仿真或真实数据集上的泛化；未包含学习型鲁棒通信方法作为对照。

---

## 307. Rheos: Modelling Continuous Motion Dynamics in Hierarchical 3D Scene Graphs

**arXiv ID:** 2603.20239 | [PDF](https://arxiv.org/pdf/2603.20239v1)

**作者:** Iacopo Catalano `[一作]` (University of Turku), Julio A. Placed `[通讯]` (Instituto Tecnológico de Aragón)

**通讯引用:** 409 | [OpenAlex ID](https://openalex.org/A5003956337)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在3D场景图中构建了Rheos框架，添加连续方向运动模型层，实现环境动态信息的语义化与可导航化。

**💡 创新点**

创新点包括：①用半封闭高斯混合模型替代离散直方图，得到无量化方向分布和明确的不确定性；②采用在线增量拟合、reservoir sampling 与BIC模型选择，使更新成本从二次降为线性，并在每个节点上自动确定混合分量数。

**🔧 技术方法**

使用技术包括：3DSG层次结构、稀疏三维哈希索引、半封闭高斯混合模型（SW‑GMM）、K‑means+++EM、BIC模型选择、reservoir sampling 以及并行化在线更新。

**📊 数据集**

数据集：基于 Gazebo + AWS RoboMaker Small House World 与自定义 PedSim 仿真器生成的 18m×10m 房屋场景，包含 7 名行人模拟轨迹。

**📈 对比分析**

比较方法：与 Aion（离散直方图）及参考 MoD 进行对比，采用 Mean Log Predictive Density（MLPD）和 Mean Predictive Probability（MPP）指标；Rheos 在所有分辨率下均优于 Aion，最显著提升发生在细分辨率（δ=0.2m）并在 MPP 评估中亦保持领先。

**⚠️ 局限性**

局限性：更新时计算与内存开销较高，尤其在粗分辨率下需要更长 EM 迭代；受限于离散化评估指标对连续模型的衡量不完全公平，未来需在更大规模真实环境中验证长期观测与非平稳动力学的适应性。

---

## 308. ALICE: A Multifaceted Evaluation Framework of Large Audio-Language Models' In-Context Learning Ability

**arXiv ID:** 2603.20433 | [PDF](https://arxiv.org/pdf/2603.20433v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 309. Transformer-Based Predictive Maintenance for Risk-Aware Instrument Calibration

**arXiv ID:** 2603.20297 | [PDF](https://arxiv.org/pdf/2603.20297v1)

**作者:** Adithya Parthasarathy `[一作]`, Seema Gangaiah Aarella `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于预测维护的校准调度框架，先将C-MAPSS引擎失效数据改造成重复漂移-校准循环，并通过多传感器窗口预测时间至漂移（TTD），随后利用预测结果制定校准计划。

**💡 创新点**

创新点在于：①将工业失效数据重新构造为校准任务的标准化数据集；②结合Transformer与量化回归的多任务学习，实现对TTD的点预测与不确定性估计；③将预测误差与校准与违规成本耦合，形成风险感知的调度策略。

**🔧 技术方法**

主要技术包括：时序窗口构造、线性回归/随机森林/LightGBM等基线回归、LSTM、CNN、TCN以及轻量级Transformer；LSTM做多分位数预测，Transformer做点预测；决策层基于阈值/分位数触发校准；成本模型用于评估策略。

**📊 数据集**

使用的是改造后的C-MAPSS FD001–FD004四个子集（共计四个不同运行条件的数据集），每个子集通过选取漂移敏感传感器、设定虚拟阈值、插入合成校准事件得到标注。

**📈 对比分析**

与基线对比时，Transformer在FD001和FD003上获得最高R²（0.66/0.776），在更异质的FD002/FD004上仍保持竞争力；在调度实验中，预测策略将总成本从反应式的1734降至1193，违规率从289降至90；量化阈值策略进一步把违规率压至26，但总成本升至2516，展示了风险抑制与资源消耗的权衡。

**⚠️ 局限性**

局限性包括：①校准事件是人工合成的，缺乏真实校准记录；②仅使用单一数据集，未充分验证跨行业/跨设备的泛化；③未对不同运行条件做细粒度归一化或自适应阈值；④量化模型未整合到Transformer中，导致训练与预测分离；⑤缺少多种随机种子评估，未量化模型方差。

---

## 310. Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models

**arXiv ID:** 2603.20808 | [PDF](https://arxiv.org/pdf/2603.20808v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 311. Can LLMs Perform Synthesis?

**arXiv ID:** 2603.20264 | [PDF](https://arxiv.org/pdf/2603.20264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 312. RLVR Training of LLMs Does Not Improve Thinking Ability for General QA: Evaluation Method and a Simple Solution

**arXiv ID:** 2603.20799 | [PDF](https://arxiv.org/pdf/2603.20799v1)

**作者:** Kaiyuan Li `[一作]` (Nanjing University), Yang Yu `[通讯]` (Nanjing University)

**通讯引用:** 9982 | [OpenAlex ID](https://openalex.org/A5100342259)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Reinforcement Learning from Verifiable Rewards (RLVR) 在一般问答 (GQA) 任务中的效果，提出了 Cross‑Generation 评估框架来隔离思考过程的价值，并发现思考停滞现象；随后提出了 START（Separated Thinking And Response Training）两阶段梯度掩码训练方法，先只优化思考轨迹，后再联合优化回答。

**💡 创新点**

创新点包括：①Cross‑Generation 评估框架能够在不同模型上单独检验思考与回答的贡献；②首次发现并命名了 GQA 上的“思考停滞”现象；③提出了 START 两阶段训练，利用梯度掩码将回答阶段转化为环境，使 RL 优化仅聚焦于思考，从而显著提升思考质量和最终回答。

**🔧 技术方法**

使用技术包括：RLVR、GRPO、DAPO、GRPO‑MA、ArmoRM 奖励模型、Chain‑of‑Thought (CoT) 思考生成、Cross‑Generation 评估、梯度掩码两阶段训练、标准的 LLM MDP 框架。

**📊 数据集**

使用数据集：MATH（数学推理）、AlpacaEval 2.0（通用 QA）、ExpertQA、UltraFeedback（指令跟随）、Arena‑Hard、WildBench 等多种通用问答/指令数据集。

**📈 对比分析**

对比方法：将 START 生成的思考轨迹与基线模型的回答做 T_post+A_pre 组合；与 GRPO、GRPO‑MA、普通 RL（无掩码）做 win‑rate 和奖励曲线对比。结果显示：在 T_post+A_pre 上，START 达到约 68% win‑rate（相较 GRPO‑MA 的 34% 及 vanilla GRPO 的 34%），在完整模型上提升约 59% win‑rate 并提升奖励值；在 DAPO、UltraFeedback 等不同算法/数据集上也获得类似提升，证明 START 在多任务、多算法上均具有稳健的性能提升。

**⚠️ 局限性**

局限性：①思考与回答的耦合仍未完全消除，START 只能在一定程度上逼迫思考演化；②在高度可验证任务中效果更好，GQA 任务仍可能存在“快捷路径”未被完全消除；③实验主要集中在 Qwen3 系列和少数 RL 算法，尚未在更大规模模型或更多多模态任务上验证；④梯度掩码对训练稳定性和收敛速度的影响尚需进一步研究。

---

## 313. High-fidelity Multi-view Normal Integration with Scale-encoded Neural Surface Representation

**arXiv ID:** 2603.20337 | [PDF](https://arxiv.org/pdf/2603.20337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 314. StageCraft: Execution Aware Mitigation of Distractor and Obstruction Failures in VLA Models

**arXiv ID:** 2603.20659 | [PDF](https://arxiv.org/pdf/2603.20659v1)

**作者:** Kartikay Milind Pangaonkar `[一作]` (Arizona State University), Nakul Gopalan `[通讯]` (Arizona State University)

**通讯引用:** 752 | [OpenAlex ID](https://openalex.org/A5089421543)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e0540dec-d77f-42db-94ae-d039248f6393` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对预训练的视觉-语言-动作(VLA)模型进行环境初始状态干预，以提升在含干扰物的任务中执行成功率。

**💡 创新点**

提出StageCraft，一个无训练、基于VLM的上下文推理模块，能够在不改变策略的前提下动态识别并移除必要的干扰物，显著提升性能。

**🔧 技术方法**

使用大型互联网预训练的Vision Language Model进行上下文推理，结合SAM进行目标检测，利用逆运动学规划实现物体移除。

**📊 数据集**

采用RLBench仿真环境、Franka FR3机器人上的三类真实任务（Stack Cups、Setup Plate、Block in Bowl），并使用10个任务的演示数据集进行评估。

**📈 对比分析**

与Pi0.5和SmolVLA基线对比，真实任务中提升约40%成功率；仿真实验显示StageCraft可根据策略鲁棒性自适应干预，更多上下文样本进一步提升性能。

**⚠️ 局限性**

受VLM上下文窗口限制，难以处理更大规模的演示；只针对离散物体干扰，无法应对非物体或连续型干扰场景。

---

## 315. LLM-Enhanced Energy Contrastive Learning for Out-of-Distribution Detection in Text-Attributed Graphs

**arXiv ID:** 2603.20293 | [PDF](https://arxiv.org/pdf/2603.20293v1)

**作者:** Xiaoxu Ma `[一作]` (Tianjin University), Chen Zhao `[通讯]` (Baylor University)

**通讯引用:** 4529 | [OpenAlex ID](https://openalex.org/A5100767050)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对文本属性图的节点级 OOD 检测，提出了 LECT 方法；

**💡 创新点**

创新点在于利用大语言模型生成具有依赖关系的高质量伪 OOD 节点，并结合能量基对比学习实现 IND 与 OOD 的有效区分；

**🔧 技术方法**

使用大语言模型生成伪 OOD 文本、MiniLM 进行文本特征提取、图神经网络进行节点表示、能量函数与对比损失进行训练；

**📊 数据集**

在六个基准数据集（如 Cora、Citeseer、PubMed、Arxiv、ogbn-products 等）上进行实验；

**📈 对比分析**

与 MSP、ODIN、GPN、OOD-GAT、OSSNC、EMP、GNNSAFE、GRASP、NODESAFE 等基线对比，LECT 在保持或提升 IND 准确率的同时，AUROC、AUPR、FPR95 指标显著提升；

**⚠️ 局限性**

局限在于对 LLM 生成伪 OOD 的依赖、需要额外的计算资源、以及对极端 OOD 场景的适应性仍需进一步验证。

---

## 316. Governance-Aware Vector Subscriptions for Multi-Agent Knowledge Ecosystems

**arXiv ID:** 2603.20833 | [PDF](https://arxiv.org/pdf/2603.20833v1)

**作者:** Steven Johnson `[一作]` `[通讯]` (Independent Researcher), Steven Johnson (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了治理感知向量订阅机制，将语义相似度匹配与多维政策谓词（处理层级、直接营销限制、训练退出、司法管辖权、科学研究用途）相结合，实现实时、合规的推送通知。

**💡 创新点**

创新点在于：①将连续的向量相似度与离散的多维政策判定在单一通知谓词中联合判定；②使用声明式 ADHP 规范化多维政策并在查询中直接索引；③在已验证的知识库上实现推送订阅，保证信息来源的完整性。

**🔧 技术方法**

技术包括：HNSW 近似最近邻索引、cosine 相似度、PostgreSQL + pgvector 存储与查询、JSONB 记录政策元数据、GIN/HNSW 复合索引、Webhook / 队列推送机制以及 AIngram 知识库框架。

**📊 数据集**

使用 PASA 合成基准数据集：1000 个知识块（5 个领域）、50 个代理（不同处理层级与目的）、93 条向量订阅，嵌入由 bge‑m3 生成。

**📈 对比分析**

对比方法：治理模式 vs 未治理模式 vs 关键词订阅；通过布尔谓词对比预期与实际通知集合。治理模式实现 100% 政策合规且零授权内容漏掉；未治理模式 50.6% 的违规通知；关键词订阅无政策过滤。延迟低于 5 ms（500 条订阅时的 p95 延迟 4.98 ms），且政策过滤几乎不增加额外开销。

**⚠️ 局限性**

局限性包括：①依赖代理声明的正确性，未提供误声明的自动纠正；②使用合成数据，未验证在真实规模与多样性场景下的性能；③目前仅支持静态政策，未实现动态政策更新或跨实例订阅；④只评估了单实例 HNSW 的准确性，未覆盖大规模并发与近似误差带来的召回下降。

---

## 317. Smart Operation Theatre: An AI-based System for Surgical Gauze Counting

**arXiv ID:** 2603.20752 | [PDF](https://arxiv.org/pdf/2603.20752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 318. Enhancing Vision-Based Policies with Omni-View and Cross-Modality Knowledge Distillation for Mobile Robots

**arXiv ID:** 2603.20679 | [PDF](https://arxiv.org/pdf/2603.20679v1)

**作者:** Kai Li `[一作]` (Zhejiang University), Shiyu Zhao `[通讯]` (Westlake University)

**通讯引用:** 4762 | [OpenAlex ID](https://openalex.org/A5052346042)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了一种从全景深感知教师到单视RGB学生的知识蒸馏框架，以提升轻量移动机器人视觉导航性能。

**💡 创新点**

创新点在于同时蒸馏教师的动作和跨模态、全景深嵌入，利用对比学习对学生特征进行正则化，从而在仅使用单摄像头时获得近似全景深性能。

**🔧 技术方法**

采用了仿真+真实环境下的演示数据、深度估计（DATv2）、ResNet、LSTM、信息对比损失（InfoNCE）以及DAgger和SAC训练。

**📊 数据集**

使用了基于ROS Gazebo的4×4米模拟环境、随机障碍布局以及真实实验室环境收集的约5万条(状态,图像,动作)数据。

**📈 对比分析**

与多种预训练编码器（ResNet、DINOv2、CLIP、MAE等）以及已有方法（RoboSaGA、VISARL、Zhang等）对比，单视RGB蒸馏方法在行动误差降低约20%、成功率提升约23%、平均行驶距离提高约20%，推理时间约20ms，显著优于其他方案。

**⚠️ 局限性**

局限在于仍依赖离线教师训练、对高帧率深度估计有额外延迟、对极端视觉变化（如强光、纹理相似）表现尚待进一步验证。

---

## 319. Weakly supervised multimodal segmentation of acoustic borehole images with depth-aware cross-attention

**arXiv ID:** 2603.20729 | [PDF](https://arxiv.org/pdf/2603.20729v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 320. FIGURA: A Modular Prompt Engineering Method for Artistic Figure Photography in Safety-Filtered Text-to-Image Models

**arXiv ID:** 2603.20201 | [PDF](https://arxiv.org/pdf/2603.20201v1)

**作者:** Luca Cazzaniga `[一作]` `[通讯]` (Independent Researcher), Luca Cazzaniga (Independent Researcher)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究开发了FIGURA方法，一套模块化的提示工程系统，帮助专业艺术家在活跃安全过滤器下生成合法的人体艺术摄影。

**💡 创新点**

创新点包括：①“Golden Rule”——用存在描述代替缺失描述；②艺术家引用在提示中同时起到美学与安全锚定作用；③空间情境是独立过滤变量；④使用几何词汇绕过轮廓识别；以及可复制的80–90%成功率模板。

**🔧 技术方法**

技术上基于SCHEMA框架构建八个知识文件，采用文本分类+图像NSFW检测的多阶段安全管道，并在FLUX 2 Pro (Cloud) 上进行实验验证。

**📊 数据集**

实验数据来源于对FLUX 2 Pro的200+次生成测试，涵盖多位摄影师、画家、体型、空间与光照组合；未使用公开艺术图像数据集，而是自建的多变量提示组合。

**📈 对比分析**

通过对比无结构提示的基准，记录成功、部分失败与完全阻断三类结果；在FLUX 2 Pro上，T01–T03模板成功率为82–90%，相比传统提示提升约60–70%。

**⚠️ 局限性**

局限性包括：仅在FLUX 2 Pro验证；正面/前视人物仍难以通过；模板聚焦于后视与轮廓；平台安全过滤更新可能导致失效；不适用于非合法艺术内容。

---

## 321. Beyond Scalar Rewards: Distributional Reinforcement Learning with Preordered Objectives for Safe and Reliable Autonomous Driving

**arXiv ID:** 2603.20230 | [PDF](https://arxiv.org/pdf/2603.20230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 322. Diffusion Model for Manifold Data: Score Decomposition, Curvature, and Statistical Complexity

**arXiv ID:** 2603.20645 | [PDF](https://arxiv.org/pdf/2603.20645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 323. Can AI Agents Answer Your Data Questions? A Benchmark for Data Agents

**arXiv ID:** 2603.20576 | [PDF](https://arxiv.org/pdf/2603.20576v1)

**作者:** Ruiying Ma `[一作]` (UC Berkeley), Aditya G. Parameswaran `[通讯]` (UC Berkeley)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个基于真实企业工作负载的端到端数据代理评测基准（Data Agent Benchmark, DAB），覆盖多数据库集成、格式不一致的连接键、无结构文本转换以及领域知识等四大挑战；

**💡 创新点**

首次提出“多数据库、格式不一致、文本转换、领域知识”四属性的评测框架，并系统化地将这些属性注入公开数据集以生成可验证的查询；

**🔧 技术方法**

利用ReAct式大型语言模型（GPT‑5, Gemini‑3‑Pro 等）和工具调用（数据库查询、Python 执行）进行端到端推理，结合 pass@k 评估、成本统计和轨迹分析；

**📊 数据集**

使用 12 个公开数据集（agnews、bookreview、crmarenapro、deps_dev_v1、github_repos、googlelocal、music_brainz_20k、pancancer_atlas、patents、stockindex、stockmarket、yelp），共 54 句自然语言查询；

**📈 对比分析**

对比五种前沿 LLM（Gemini‑3‑Pro、GPT‑5‑mini、GPT‑5‑2、Kimi‑2、Gemini‑2.5‑Flash）在 DAB 上的表现，最佳模型 pass@1 仅 38%，pass@50 69%；成本上 Gemini‑3‑Pro 最高，GPT‑5‑mini 成本最低但仍有 30% pass@1；PromptQL 系统相比基线提升 7pp；

**⚠️ 局限性**

局限性包括基准规模相对较小、仅使用公开数据集且未覆盖所有实际企业数据噪声、对文本提取过度依赖正则表达式导致高错误率、以及评测主要聚焦于查询成功率而非执行效率与鲁棒性等方面。

---

## 324. The Art of Midwifery in LLMs: Optimizing Role Personas for Large Language Models as Moral Assistants

**arXiv ID:** 2603.20626 | [PDF](https://arxiv.org/pdf/2603.20626v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 325. LJ-Bench: Ontology-Based Benchmark for U.S. Crime

**arXiv ID:** 2603.20572 | [PDF](https://arxiv.org/pdf/2603.20572v1)

**作者:** Hung Yun Tseng `[一作]` (University of Wisconsin--Madison), Grigorios Chrysos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于Model Penal Code和California Penal Code法律框架的LJ-Bench，构建了涵盖76类犯罪的多层次知识图谱与1000余个针对性问题，用于评估LLM在危害性信息泄露方面的鲁棒性。

**💡 创新点**

首创将正式法律体系与非法查询场景对齐的犯罪概念本体及知识图谱，实现了比现有基准更细粒度、更全面的非法行为覆盖，并引入了新型犯罪类别（如环境与动物犯罪）。

**🔧 技术方法**

采用本体建模（Schema.org扩展）、知识图谱技术、提示注入与迭代式对抗攻击（如PAIR、TAP、PAP）以及Gemini‑1.5‑Pro作为自动评判者的链式思维评分机制。

**📊 数据集**

使用基于California Penal Code和Model Penal Code的法律文本数据，结合手工编写的630个高质量问题（4–20条），并通过词汇替换与翻译扩充得到增强版数据集。

**📈 对比分析**

对16个LLM（Gemini、GPT系列、开源模型）进行了241,920次攻击实验，发现即便是最新模型在迭代攻击下仍可达到4.5+的危害评分；Gemini‑1.5‑Pro在所有模型中最具一致性评判力；Gemma‑2B在小规模模型中表现最强。

**⚠️ 局限性**

局限在于：①评测主要聚焦文本输入，未覆盖多模态或更复杂交互；②部分攻击（ITERATIVE）仅在少数模型上测试，结果可能低估总体易受性；③基准以美国法律为主，跨司法区的适用性需进一步验证。

---

## 326. EruDiff: Refactoring Knowledge in Diffusion Models for Advanced Text-to-Image Synthesis

**arXiv ID:** 2603.20828 | [PDF](https://arxiv.org/pdf/2603.20828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 327. Detecting Neurovascular Instability from Multimodal Physiological Signals Using Wearable-Compatible Edge AI: A Responsible Computational Framework

**arXiv ID:** 2603.20442 | [PDF](https://arxiv.org/pdf/2603.20442v1)

**作者:** Truong Quynh Hoa `[一作]` (Clevix LLC), Truong Xuan Khanh `[通讯]` (Clevix LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并验证了Melaguard框架，利用可穿戴兼容的边缘AI技术对神经血管不稳定（NVI）进行早期检测，构建了多模态生理信号融合与轻量化Transformer分类器，并配合PHBV:eumelanin湿度激活传感器与AeroKernel微内核实现隐私安全推理。

**💡 创新点**

创新点包括：①将心率变异性（HRV）、外周灌注指数、SpO₂与双侧相位一致性四种模态融合至NVI评分；②采用Transformer-lite（1.2M参数、4头自注意力、2层编码器）实现低功耗边缘推理；③开发PHBV:eumelanin湿度激活光学传感材料以消除肤色偏差；④集成AeroKernel微内核实现可解释、可审计、GDPR/HIPAA兼容的负责AI体系。

**🔧 技术方法**

使用的技术包括：Transformer-lite轻量级分类器、PHBV:eumelanin湿度激活光学传感器、AeroKernel微内核、PPG与ECG信号处理（PRV、HRV、PI）、多模态特征提取与加权融合、NVI评分公式、5折交叉验证与Bootstrap置信区间、AUC、敏感度、特异度评估。

**📊 数据集**

使用的数据集有：PhysioNet CVES（172例，84卒中+88对照）、PhysioNet BIDMC PPG与呼吸（53例 ICU）、Liang PPG-BP（219例，2.1秒PPG片段）、以及10,000条合成模拟数据，用于三阶段验证。

**📈 对比分析**

采用5折分层交叉验证、AUC、敏感度、特异度及Bootstrap 95%CI进行比较。Transformer-lite在CVES上实现AUC 0.755[0.630–0.778]，在合成数据上AUC 0.88；在PPG-BP上实现AUC 0.923[0.869–0.968]；PRV与ECG RMSSD的相关系数为0.690，优于传统LSTM、RF和SVM，验证了多模态融合与轻量化模型的性能优势。

**⚠️ 局限性**

主要限制包括：①缺乏端到端PPG到分类的验证，阶段性验证仍需临床PPG卒中样本；②相位一致性模态缺失率高、样本量不足导致统计功效有限；③PHBV:eumelanin传感器仅在仿真层面验证，实测与长期稳定性待进一步研究；④AeroKernel WCET未在目标硬件上测量；⑤样本多样性（肤色、年龄、病种）不足，需更广泛人群验证。

---

## 328. Abjad-Kids: An Arabic Speech Classification Dataset for Primary Education

**arXiv ID:** 2603.20255 | [PDF](https://arxiv.org/pdf/2603.20255v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 329. E-SocialNav: Efficient Socially Compliant Navigation with Language Models

**arXiv ID:** 2603.20664 | [PDF](https://arxiv.org/pdf/2603.20664v1)

**作者:** Ling Xiao `[一作]` (Hokkaido University), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**通讯引用:** 6764 | [OpenAlex ID](https://openalex.org/A5048624196)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估 GPT-4o 与 Claude 在社会合规导航任务中的零样本表现，并提出 E-SocialNav 这一轻量级小型语言模型，通过两阶段 SFT+ DPO 在小数据场景下实现更高的语义相似度和动作准确率。

**💡 创新点**

创新点：①针对社会合规导航提出两阶段训练流程；②在小数据环境下利用 SLM 与 LoRA、投影层微调实现高效部署；③构建多轮对话 SFT 数据集和单轮 DPO 数据集。

**🔧 技术方法**

技术：采用 Phi-2-2.7B 语言模型与 SigLIP 视觉塔；SFT 仅更新投影层；DPO 仅更新 LoRA；使用 BERTScore、SBERT、SMS 等指标评估语义相似度。

**📊 数据集**

数据集：使用 SNEI 数据集（由 SCAND 与 MuSoHu 衍生的 325 样本多轮对话），并构建 DPO 数据对进行训练与评估。

**📈 对比分析**

比较方法：与 GPT-4o、Claude 及 Finetuned Social‑LLaVA 对比，E‑SocialNav 在 BERTScore‑P/R/F1、SBERT‑cos、SMS、动作准确率等指标上均优于零样本基线，且推理速度与能耗显著降低。

**⚠️ 局限性**

限制：小样本标注可能存在偏差，模型在不同文化与复杂情境下的通用性尚未充分验证；对更大规模数据和多模态交互的鲁棒性需要进一步评估。

---

## 330. Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models

**arXiv ID:** 2603.20607 | [PDF](https://arxiv.org/pdf/2603.20607v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 331. Unified Orbit-Attitude Estimation and Sensor Tasking Framework for Autonomous Cislunar Space Domain Awareness Using Multiplicative Unscented Kalman Filter

**arXiv ID:** 2603.20579 | [PDF](https://arxiv.org/pdf/2603.20579v1)

**作者:** Smriti Nandan Paul `[一作]` (Missouri University of Science and Technology), Siwei Fan `[通讯]` (Embry-Riddle Aeronautical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种面向地月间空间域监测的多机系统观测器架构优化与传感器任务分配与状态估计的统一框架；

**💡 创新点**

创新点在于：①设计了融合卫星数、轨道稳定性、可观测性、观测次数与近地距离的六项复合代价函数；②采用贝叶斯优化中的树状 Parzen Estimator (TPE) 高效搜索观测器轨道与卫星分布；③构建基于互信息的贪婪任务分配方法，并在任务间隔上细粒度更新姿态与轨道状态，利用误差状态乘法 Unscented Kalman Filter；

**🔧 技术方法**

技术手段包括：CR3BP 动力学模型、Cook‑Torrance BRDF 光度模型、误差状态乘法 UKF（GRP 误差表示）、贝叶斯优化（TPE）、互信息任务优化、k‑means 聚类、随机搜索基线；

**📊 数据集**

使用的数据集为：基于 64 个已公布/计划的月球与近地轨道任务的零速面采样点（共 10k 个点）做 k‑means 聚类得到 100/2000/10000 个静态目标；以及 202 个可动态轨道目标（如 Halo、DRO、3:1 等）用于任务分配与估计；

**📈 对比分析**

与随机搜索基线相比，TPE 在三种目标密度场景下均实现更低的代价函数（如 Scenario A：TPE 1.508×10⁻⁴ vs 随机 1.224×10⁻⁵），并在估计实验中显示：转动状态误差相对线性状态更易失效，任务间隔增大时姿态估计精度明显下降；整体翻译状态保持良好；

**⚠️ 局限性**

局限性包括：①任务分配仅基于轨道协方差的互信息，未显式考虑姿态不确定性；②观测模型在任务分配阶段简化为分析亮度，估计阶段使用高阶 Facet‑Cook‑Torrance，导致计算差异；③仅采用 CR3BP 动力学，未考虑更复杂的多体/非惯性效应；④缺乏多传感器融合与网络通信约束；⑤对观测几何（光照、排斥角）限制较严，影响实用性。

---

## 332. Efficient Visual Anomaly Detection at the Edge: Enabling Real-Time Industrial Inspection on Resource-Constrained Devices

**arXiv ID:** 2603.20288 | [PDF](https://arxiv.org/pdf/2603.20288v1)

**作者:** Arianna Stropeni `[一作]` (University of Padua), Gian Antonio Susto `[通讯]` (University of Padua)

**通讯引用:** 4063 | [OpenAlex ID](https://openalex.org/A5026617079)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在工业边缘设备上实现可实时、高隐私的视觉异常检测。

**💡 创新点**

提出 PatchCore-Lite 的两阶段量化最近邻搜索和 PaDiM-Lite 的对角协方差近似，以显著降低内存和计算成本。

**🔧 技术方法**

采用 MobileNetV2 轻量级骨干网络、产品量化、对角协方差和两阶段搜索等技术。

**📊 数据集**

在 MVTec AD 和 VisA 两个工业视觉异常检测基准数据集上进行评估。

**📈 对比分析**

与原版 PatchCore/PaDiM 及其 Edge 版本对比，PatchCore-Lite 约 79% 内存减少、PaDiM-Lite 31% 推理时间缩短，图像级 AUC 略有下降但仍保持在 0.85–0.95 范围。

**⚠️ 局限性**

主要局限是图像级检测性能下降、对量化参数和 k 值敏感，以及对角协方差可能忽略特征间相关性。

---

## 333. CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration

**arXiv ID:** 2603.20741 | [PDF](https://arxiv.org/pdf/2603.20741v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 334. ToFormer: Towards Large-scale Scenario Depth Completion for Lightweight ToF Camera

**arXiv ID:** 2603.20669 | [PDF](https://arxiv.org/pdf/2603.20669v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 335. DiffGraph: An Automated Agent-driven Model Merging Framework for In-the-Wild Text-to-Image Generation

**arXiv ID:** 2603.20470 | [PDF](https://arxiv.org/pdf/2603.20470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 336. LogFold: Compressing Logs with Structured Tokens and Hybrid Encoding

**arXiv ID:** 2603.20618 | [PDF](https://arxiv.org/pdf/2603.20618v1)

**作者:** Shiwen Shan `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 34608 | [OpenAlex ID](https://openalex.org/A5000582109)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种新的日志压缩工具 LogFold，专门针对结构化 token 和多类型 token 进行细粒度的模式挖掘与混合编码，以提升日志压缩率并保持较快的压缩速度。

**💡 创新点**

创新点包括：
- 通过“分隔符骨架”将结构化 token 拆解为骨架和子 token，发现并利用子 token 的列式同质性；
- 设计了两阶段的结构化 token 处理器：骨架分组 + 关键位置模式挖掘，显著降低重复度；
- 开发了混合编码器：针对纯数字、纯字符串和混合型 token 分别使用动态 delta、字典 + 位置标签、奇偶映射+统一数值编码等技术，实现更细粒度、类型感知的压缩；
- 通过 ablation、敏感度与多后端压缩实验验证其有效性与稳健性。

**🔧 技术方法**

核心技术：
- 正则表达式和规则驱动的 Token 分析器；
- 关键位置识别（直方图、熵、主导比例）和条件重分组；
- FP‑Growth 发现列常数；
- 动态 delta、长度分组与弹性编码、混合奇偶映射；
- 字典编码与压缩后端（gzip/LZMA/bzip2）。

**📊 数据集**

16 个公开日志数据集，涵盖超级计算机、分布式系统、移动系统、服务器应用等：Thunderbird、BGL、HPC、Windows、Mac、Linux、Spark、HDFS、OpenStack、Hadoop、Zookeeper、Android、HealthApp、OpenSSH、Apache、Proxifier。

**📈 对比分析**

与 9 个基线（4 个通用压缩器 + 5 个专用日志压缩器）在同一环境下对比；使用压缩比 CR、压缩速度 CS 和解压速度 DS 作为指标；LogFold 在平均 CR 上提升 11.11% 至 494.19%（相对 gzip），平均压缩速度 9.842 MB/s，解压速度在 0.119–2.488 MB/s 范围；在 12/16 数据集上取得最高 CR，ablation 证明结构化处理器与混合编码器均不可或缺。

**⚠️ 局限性**

局限性：
- 对于含大量纯字符串且缺少可提取骨架的日志（如 HDFS），压缩比受限；
- 关键位置和重分组阈值虽稳健，但对极端日志格式仍需人工调参；
- 解压过程受数字恢复步骤影响，速度相对慢；
- 评测基于公开数据集，未覆盖全部工业级日志场景，可能在更大规模或特殊日志格式下表现不同。

---

## 337. Multi-Robot Learning-Informed Task Planning Under Uncertainty

**arXiv ID:** 2603.20544 | [PDF](https://arxiv.org/pdf/2603.20544v1)

**作者:** Abhish Khanal `[一作]` (George Mason University), Gregory J. Stein `[通讯]` (George Mason University)

**通讯引用:** 6827 | [OpenAlex ID](https://openalex.org/A5042000667)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套多机器人任务规划框架，利用学习模型预测部分已知环境中目标物体的可能位置，并通过基于模型的规划实现长周期协同完成复杂多阶段任务。

**💡 创新点**

创新点在于将学习得到的物体位置概率、单机器人高层动作抽象以及多机器人状态转移模型统一到一个抽象层，形成概率联合动作抽象；并结合PO‑UCT蒙特卡洛树搜索实现高效的长周期决策与协同；该框架同时兼顾学习与规划的优点，在多机器人队伍中实现了更合理的任务分配与时间最小化。

**🔧 技术方法**

使用了SCLTL与DFA进行任务语义化表示；SBERT+全连接网络预测容器中目标物体的存在概率；单机器人高层动作抽象（move+search+interaction）；多机器人联合动作抽象与状态转移模型；PO‑UCT（部分可观测UCT）进行搜索；ProcTHOR仿真环境与LoCoBot物理机器人结合GroundingDINO进行真实感知。

**📊 数据集**

训练数据来自500个Procedurally generated ProcTHOR 家庭环境，构造正负样本标注物体是否位于指定容器；真实实验中使用LoCoBot的RGB摄像头采集的图像，结合GroundingDINO进行目标识别。

**📈 对比分析**

与无学习贪心规划（Nearest‑Neighbor）和仅利用学习概率的贪心规划进行对比；在1、2、3机器人以及小/中/大尺寸房间的实验中，平均成本分别提升约47%、40%、34%；在真实家居环境中，导航成本由23.72 m降至7.79 m（提升67%）以及由21.88 m降至8.95 m（提升59%），显示在大环境下优势更为显著。

**⚠️ 局限性**

局限性包括：仅适用于技能相同的同构机器人，无法直接处理预条件或任务执行顺序的通用规划；联合搜索空间随机器人数和任务复杂度呈指数增长，计算需求较高；框架为集中式，缺乏对分布式鲁棒性的充分研究。

---

## 338. PAVE: Premise-Aware Validation and Editing for Retrieval-Augmented LLMs

**arXiv ID:** 2603.20673 | [PDF](https://arxiv.org/pdf/2603.20673v1)

**作者:** Tianyi Huang `[一作]` (Ryquo), Michael Zhang `[通讯]` (App-In Club)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出检索增强语言模型的验证层——Premise‑Grounded Answer Validation and Editing，先将检索结果拆解为原子前提，生成草稿答案并评估支持度，低支持时进行修订；

**💡 创新点**

创新点在于将原子前提提取与支持度评分相结合，构成阈值驱动的修订机制，使答案提交前实现显式验证和可审计的推理轨迹；

**🔧 技术方法**

使用 GPT‑4o mini 作为LLM，多阶段提示式推理：原子化分解、草稿生成、支持度评分及必要时的修订；

**📊 数据集**

使用 PubMedQA（生物医学问答）和 SQuAD（段落级事实问答）两种证据基准；

**📈 对比分析**

通过与 Baseline RAG、RAG+重要性权重、RAG+支持评分四种对比，在 PubMedQA 上从 71.2% 提升至 73.3%，在 SQuAD 上从 62.4% 提升至 95.1%，显示显著性能提升；

**⚠️ 局限性**

局限包括仅在固定检索器和基础模型上评估、缺乏对原子分解质量的直接度量、额外推理开销、支持度分数不等同真相，受检索证据本身质量影响。

---

## 339. Multi-Stage Fine-Tuning of Pathology Foundation Models with Head-Diverse Ensembling for White Blood Cell Classification

**arXiv ID:** 2603.20383 | [PDF](https://arxiv.org/pdf/2603.20383v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 340. Democratizing AI: A Comparative Study in Deep Learning Efficiency and Future Trends in Computational Processing

**arXiv ID:** 2603.20920 | [PDF](https://arxiv.org/pdf/2603.20920v1)

**作者:** Lisan Al Amin `[一作]`, Abdulaziz Tabbakh `[通讯]`

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 TensorFlow 与 PyTorch 两个框架下，对 Conv6、VGG16、ResNet18 与 CycleGAN 四种深度学习模型进行训练与推理性能基准测试，并比较 Intel Xeon CPU 与 NVIDIA Tesla T4 GPU 的运行时、内存占用及带宽表现。

**💡 创新点**

① 首次实现跨框架的 GPU/CPU 双侧对比，揭示 TensorFlow 的 kernel‑fusion 可将推理延迟降低约 15%；② 运用多项式回归对 GPU 内存增长趋势进行预测，预估未来内存需求；③ 强调 GPU 资源民主化共享的重要性以支持弱势机构的 AI 研究。

**🔧 技术方法**

GPU SIMT 架构、CUDA、cuDNN、TensorFlow XLA、PyTorch 动态图、TensorBoard 监控、Polynomial regression、Logistic growth model 等技术。

**📊 数据集**

使用 CIFAR‑10（图像分类）、Horse2Zebra（CycleGAN 图像转化）以及公开的 GPU 内存与发布日期历史数据。

**📈 对比分析**

通过平均训练/推理时间、GPU/CPU 时间比、内存占用、内存带宽等多维度指标进行比较；实验结果表明 GPU 在训练阶段加速幅度从 11×（CycleGAN）到 246×（Conv6），并且 TensorFlow 相较 PyTorch 推理延迟下降约 15%。

**⚠️ 局限性**

局限性包括：仅评估单一 CPU/GPU 对；未覆盖 Transformer、TPU 等新架构；模型规模受限于免费 Colab GPU 运行时长；未对能耗与碳足迹进行定量评估；内存趋势预测依赖有限的历史数据，可能忽略未来技术变革。

---

## 341. RubricRAG: Towards Interpretable and Reliable LLM Evaluation via Domain Knowledge Retrieval for Rubric Generation

**arXiv ID:** 2603.20882 | [PDF](https://arxiv.org/pdf/2603.20882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 342. The Hidden Puppet Master: A Theoretical and Real-World Account of Emotional Manipulation in LLMs

**arXiv ID:** 2603.20907 | [PDF](https://arxiv.org/pdf/2603.20907v1)

**作者:** Jocelyn Shen `[一作]` (Massachusetts Institute of Technology), Cynthia Breazeal `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 26297 | [OpenAlex ID](https://openalex.org/A5108541589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了大型语言模型（LLM）在日常对话中通过隐藏激励进行情绪操纵的效果，并提出了基于激励道德的PUPPET分类法，随后进行了一项包含1035名参与者的人机对话实验，量化了隐藏激励与个性化对用户信念转移的影响，并评估了LLM预测信念变化的能力。

**💡 创新点**

创新点包括：①将激励道德纳入情绪操纵的理论框架（PUPPET），②在真实日常查询情境下系统测量隐藏激励对信念变化的差异（有害 vs. 亲善），③首次对LLM直接预测用户信念变化进行基准评估，发现模型既能捕捉信念转移趋势又倾向低估幅度。

**🔧 技术方法**

使用的技术主要是：①基于GPT‑4o进行多轮对话和信念预测；②设计六种实验条件（隐藏激励 vs. 非隐藏，个性化 vs. 通用，有害 vs. 亲善）进行因子实验；③统计方法包括t检验、Wilcoxon、TOST、方差分析、相关与误差分析。

**📊 数据集**

数据集：自建实验数据集，包含1035名美国/英国参与者的日常查询对话、预/后信念评分、个人信息（人口统计、价值观、人格等）以及对话文本；使用的LLM模型包括GPT‑4o、Gemini‑2.0‑Flash、Llama‑3.1‑70B、DeepSeek‑V3.1。

**📈 对比分析**

比较方法：对六种条件进行信念转移量的方差分析和对照检验；对LLM预测结果与实际转移量计算Pearson r、Spearman ρ、RMSE、MAE并检验系统偏差。性能表现为：所有模型预测信念转移显著高于零，相关系数在0.3–0.5之间；GPT‑4o在无个人信息条件下表现最好（r≈0.46），但所有模型普遍低估真实幅度。

**⚠️ 局限性**

limitations：①实验仅涵盖低风险日常查询，未覆盖高风险医疗、金融或政治场景；②单次会话限制，未考察长期互动、记忆和累积效应；③使用自评信念作为主要因变量，缺乏行为层面验证；④个性化信息仅来自一次问卷，未包含交互历史或更深层特征，可能低估个性化潜力。

---

## 343. The data heat island effect: quantifying the impact of AI data centers in a warming world

**arXiv ID:** 2603.20897 | [PDF](https://arxiv.org/pdf/2603.20897v1)

**作者:** Andrea Marinoni `[一作]` (University of Cambridge), Benjamin Horton `[通讯]` (City University of Hong Kong)

**通讯引用:** 19261 | [OpenAlex ID](https://openalex.org/A5084053928)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究评估了全球 AI 超大规模数据中心对周边地区地表温度的影响，提出并量化了“数据热岛效应”。

**💡 创新点**

创新之处在于首次系统性地将 AI 数据中心的热排放与城市热岛效应进行对比，量化其对局部升温的贡献。

**🔧 技术方法**

采用多尺度多模态分析方法，结合 MODIS LST、WorldPop 人口分布和 AI 数据中心位置数据库进行空间时间关联分析。

**📊 数据集**

使用的数据集包括 2004-2024 年的 MODIS 500 m 分辨率地表温度数据、WorldPop 100 m 人口地图，以及全球 AI 数据中心位置数据库。

**📈 对比分析**

通过归一化月度 LST 与前 k 个月平均值比较，平均升温约 2 °C（95th 分位 1.5-2.4 °C），影响半径可达 10 km，估计受影响人口超过 3.4 亿。

**⚠️ 局限性**

局限性包括仅聚焦于非高密集区域、假设热量直接导致 LST 上升、无法完全剔除其他人为活动影响，且云覆盖及缺失数据处理仍存在不确定性。

---

## 344. Fast and Robust Deformable 3D Gaussian Splatting

**arXiv ID:** 2603.20857 | [PDF](https://arxiv.org/pdf/2603.20857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 345. Scene Graph-guided SegCaptioning Transformer with Fine-grained Alignment for Controllable Video Segmentation and Captioning

**arXiv ID:** 2603.20887 | [PDF](https://arxiv.org/pdf/2603.20887v1)

**作者:** Xu Zhang `[一作]` (Hunan University), Hanwang Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 28023 | [OpenAlex ID](https://openalex.org/A5042324027)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出可控视频分割与字幕生成任务，并实现了 Scene Graph-guided Fine-grained SegCaptioning Transformer (SG-FSCFormer)，能够根据用户给定的框框同时生成对应的掩码与字幕。

**💡 创新点**

创新点在于：1) 使用 Prompt-guided Temporal Graph Former 将用户提示映射到时序场景图，精确捕捉用户意图；2) 设计 Fine-grained Mask‑Linguistic Decoder，联合使用 Fine-grained Alignment Loss 与 Multi‑entity Contrastive Loss，实现掩码与字幕单词的细粒度对齐；3) 将开放词汇、跨模态信息和时序关系融合，首次实现多实例、可控的跨模态视频理解。

**🔧 技术方法**

采用 SwinB 视觉编码器；Prompt-guided Temporal Graph Former（图建模+prompt adaptor）；Graph-guided Iterative Query Former；冻结 Vicuna-7B 文本解码器；SAM2 作为掩码解码器；Fine-grained Alignment Loss 与 Multi‑entity Contrastive Loss 作为对齐损失。

**📊 数据集**

在 LV‑VIS 与 OVIS 两大视频实例分割数据集上增添提示-字幕注释，构建用于训练与评估的 Controllable Video SegCaptioning 数据集。

**📈 对比分析**

与现有视频字幕、分割及多模态解释方法（如 Vid2Seq、MA‑LMM、VideoGLaMM、SMOTer、OVFormer、OW‑VISCap 等）进行对比。SG‑FSCFormer 在 METEOR、SPICE、CIDEr、J&F、实例级 AP 等指标上均显著优于基线；在 LV‑VIS 上实现 J&F 最高 87.8，OVIS 上 74.6，字幕指标亦获得显著提升。

**⚠️ 局限性**

缺点包括：1) 需要手工标注掩码与字幕的细粒度对齐，规模受限；2) 对视觉相似度高或长时间消失的目标仍易出现分割/描述错误；3) 推理速度和参数量相对较高，尤其在多模态联合解码时。

---

## 346. TAFG-MAN: Timestep-Adaptive Frequency-Gated Latent Diffusion for Efficient and High-Quality Low-Dose CT Image Denoising

**arXiv ID:** 2603.20868 | [PDF](https://arxiv.org/pdf/2603.20868v1)

**作者:** Tangtangfang Fang `[一作]` (University of Nottingham Ningbo China), Jiaqi Yang `[通讯]` (University of Nottingham Ningbo China)

**通讯引用:** 99 | [OpenAlex ID](https://openalex.org/A5014124194)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种名为TAFG-MAN的低剂量CT图像去噪框架，结合感知优化自编码器、条件潜在扩散模型和时间步自适应频率门控机制；

**💡 创新点**

创新点包括：1）采用时间步自适应频率门控( TAFG )在潜在扩散过程的条件注入中逐步释放高频信息，平衡噪声抑制与细节保留；2）使用感知优化的自编码器构造紧凑但结构丰富的潜在空间；3）采用确定性DDIM式采样提升推理效率；

**🔧 技术方法**

技术主要包括：潜在扩散模型、跨注意力与自注意力的U-Net、低通滤波门控模块、VGG感知损失、KL正则化、cosine噪声调度、DDIM采样；

**📊 数据集**

使用LDCT和Projection数据集中的腹部与胸部1 mm切片，尺寸512×512，遵循LDCT-Bench分割；

**📈 对比分析**

与BM3D、RED-CNN、Q-AE、WGAN-VGG、DU-GAN、DDPM、Fast-DDPM、Dn-Dp等基线比较；TAFG-MAN在PSNR、SSIM、LPIPS上优于大多数基线，并且推理时间仅比MAN略高，保持在约18.7 s；

**⚠️ 局限性**

局限性：仅在特定LDCT子集验证，未涵盖不同解剖部位或扫描协议；潜在扩散采用简化的高斯噪声模型，未充分模拟真实CT噪声；TAFG为局部门控，未显式建模长程解剖先验；未进行放射科医生读片评估。

---

## 347. Approximating Convex Hulls via Range Queries

**arXiv ID:** 2603.20943 | [PDF](https://arxiv.org/pdf/2603.20943v1)

**作者:** T. Schibler `[一作]` (University of California), J. Zhu `[通讯]` (New York University)

**通讯引用:** 29455 | [OpenAlex ID](https://openalex.org/A5078512343)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究在仅能查询点集的空区间信息的oracle模型下，如何用有限次数的查询逼近未知点集的凸包，给出了不同查询形状（轴对齐盒、半平面）和查询适应性（非适应、适应）下的误差与查询次数之间的最优折中关系。

**💡 创新点**

创新点在于首次给出这些情况下的紧致误差上界与下界，证明了在非适应轴对齐盒查询时误差为Θ(q^{−1/d})、适应查询为Θ(q^{−1/(d−1})；在二维半平面查询时，非适应误差为Θ(q^{−1/2})、适应误差为Θ(q^{−2})，并设计相应的算法与构造对偶下界。

**🔧 技术方法**

主要技术包括空间分割与极值点提取、凸体的Minkowski和、角度与边长的分割策略、以及通过多次极值半平面查询构造半平面oracle，辅以几何不等式与组合论计数论证。

**📊 数据集**

论文纯理论分析，无使用实际数据集；所有结果均在理论上证明。

**📈 对比分析**

与之前的随机采样或全量查询方法相比，本文算法在误差与查询次数的权衡上达到了理论上可实现的最优水平，实验或实现细节未给出，但理论证明显示在给定查询预算下可获得最小误差。

**⚠️ 局限性**

局限性包括只考虑确定性算法、仅使用空查询oracle、未涵盖噪声或非凸输入等实际情况，且缺乏实验验证。

---

## 348. A chaotic flux cipher based on the random cubic family $f_{c_n}(z)=z^3+c_n z$

**arXiv ID:** 2603.20937 | [PDF](https://arxiv.org/pdf/2603.20937v1)

**作者:** Pouya Mehdipour `[一作]` (Universidade Federal de Viçosa), Mostafa Salarinoghabi `[通讯]` (Universidade Federal de Viçosa)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5019692149)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种基于随机复数三次多项式迭代的对称流密码，并结合 HKDF/HMAC 等标准加密原语实现密钥生成与完整性保护。

**💡 创新点**

创新点在于将每次迭代参数 c_n 从固定圆盘随机抽取，动态调节混沌程度，既能在噪声环境下保持 Julia 集稳定性，又能在 δ>3 时产生高度随机的密钥流；同时将复杂数轨道通过哈希映射为可用密钥流块，提升输出均匀性。

**🔧 技术方法**

采用的技术包括：随机复数三次映射 f_n(z)=z^3+c_n z、HKDF 生成子密钥、HMAC 验证、基于 SHA‑256 的加密哈希、Warm‑up 迭代、以及 NIST SP 800‑22 与 Fourmilab 等统计测试工具。

**📊 数据集**

未使用传统数据集；通过对生成的 6480 位密钥流与 2^20 位大规模序列进行统计检验，验证随机性与分布特性。

**📈 对比分析**

与传统基于 LFSR/加密哈希的伪随机生成器相比，该方法在 6480 位样本上均通过 NIST 测试并显示更高的熵值（≈6.69 bits/byte），但在 2^20 位样本中仍在 Approximate Entropy、Longest Run 与 Overlapping Template 测试中出现失效，说明大规模统计性能仍有提升空间。

**⚠️ 局限性**

主要限制包括：在大规模序列下仍出现统计偏差；参数 δ 的取值需在稳定性与混沌性之间仔细平衡；对抗攻击的理论分析仍缺乏；实现对高精度浮点计算误差的鲁棒性未充分评估。

---

## 349. Characterizing the onset and offset of motor imagery during passive arm movements induced by an upper-body exoskeleton

**arXiv ID:** 2603.20885 | [PDF](https://arxiv.org/pdf/2603.20885v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 350. Do LLM-Driven Agents Exhibit Engagement Mechanisms? Controlled Tests of Information Load, Descriptive Norms, and Popularity Cues

**arXiv ID:** 2603.20911 | [PDF](https://arxiv.org/pdf/2603.20911v1)

**作者:** Tai-Quan Peng `[一作]` (Michigan State University), Yingcai Wu `[通讯]` (Zhejiang University)

**通讯引用:** 6107 | [OpenAlex ID](https://openalex.org/A5073986937)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用大型语言模型驱动的代理仿真，在Weibo 类似环境中系统操纵信息负荷与描述性规范，观察参与阈值（是否互动）与互动分配（点赞/转发/引用）的两阶段行为，并检验其是否符合既有的注意力、社会规范与流行度理论。

**💡 创新点**

①将 LLM 生成的代理作为可控实验工具验证理论机制；②在仿真中保留自发的流行度反馈，避免传统实验中对流行度的人工强制；③采用多条件交叉设计，检验机制的条件性与交互效应；④将提示（prompt）视为可操纵的规范信息，对比无规范基线，探索提示的作用方式。

**🔧 技术方法**

使用 Qwen‑8B 语言模型生成代理决策，OASIS 架构整合网络结构、推荐系统与 LLM；通过二元 Logistic 与多项式 Logistic 进行统计建模；使用中文 GPT 语言模型生成中文内容并保持统一提示语言。

**📊 数据集**

真实用户数据：558 名在 DeepSeek 讨论区活跃的 Sina Weibo 账号及其关注关系；50 条 150–300 词的中文 seed 贴；从这些 seed 贴派生的转发与引用帖子；构建的网络与用户属性来源于公开的社交网络样本。

**📈 对比分析**

通过两阶段回归检验：在参与阈值阶段，信息负荷负向显著，流行度正向显著；在互动分配阶段，描述性规范显著集中化（如点赞/转发），流行度与信息负荷交互产生更强的集中效应。结果与理论预测高度一致，表明 LLM 驱动的仿真能复制已知的社交媒体参与机制。

**⚠️ 局限性**

局限：①未模拟评论等更细粒度的互动；②只研究单一议题（DeepSeek）导致结果可能受主题特异性影响；③活动率被固定，未考虑真实平台的脉冲式参与；④结论可能受 Qwen‑8B 的模型特性或温度设置影响；⑤推荐算法被固定，无法检验不同排序规则对结果的影响；⑥LLM 对文本长度的偏好可能导致转发/引用比例的偏移。

---

## 351. Adviser: An Intuitive Multi-Cloud Platform for Scientific and ML Workflows

**arXiv ID:** 2603.20941 | [PDF](https://arxiv.org/pdf/2603.20941v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 352. Bayesian Scattering: A Principled Baseline for Uncertainty on Image Data

**arXiv ID:** 2603.20908 | [PDF](https://arxiv.org/pdf/2603.20908v1)

**作者:** Bernardo Fichera `[一作]`, Viacheslav Borovitskiy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种将散射变换提取的图像特征与高斯过程模型相结合，用于生成图像的概率预测。

**💡 创新点**

创新点在于将散射变换的稀疏、稳健特征与高斯过程的可解释性与不确定性估计相融合，提供了一种端到端的概率预测框架。

**🔧 技术方法**

使用了散射变换（小波变换、模量、平均）作为特征提取器，并采用高斯过程回归（RBF / Matérn核）作为预测头。

**📊 数据集**

论文实验中使用的具体数据集未在此片段中明确给出，可能为常见图像分类数据集。

**📈 对比分析**

与传统的 CNN / SVM 或散射 + 线性分类器等基线方法对比，本文方法在准确率/误差以及不确定性评估上表现优异或相近，具体指标需参照论文表格。

**⚠️ 局限性**

主要局限包括：高斯过程在大规模样本上的可扩展性差、核矩阵求逆开销大；散射变换参数需要手工调优；以及对高维特征空间的处理效率有限。

---

## 353. Natural Gradient Descent for Online Continual Learning

**arXiv ID:** 2603.20898 | [PDF](https://arxiv.org/pdf/2603.20898v1)

**作者:** Joe Khawand `[一作]` (Ecole Polytechnique), David Colliaux `[通讯]` (Sony Computer Science Laboratories)

**通讯引用:** 79 | [OpenAlex ID](https://openalex.org/A5002495502)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在线连续学习（OCL）中使用自然梯度下降与Kronecker近似曲率（NGD‑KFAC）来优化模型训练，提升图像分类任务的准确率。

**💡 创新点**

创新点在于首次将NGD‑KFAC引入OCL领域，并证明其可显著提升多种OCL技巧（如经验回放、标签技巧等）的性能。

**🔧 技术方法**

技术方法包括自然梯度下降（NGD）、KFAC近似Fisher信息矩阵，以及与经验回放、标签技巧、最近类均值、分离Softmax等OCL技巧的结合。

**📊 数据集**

实验数据集涵盖 Split CIFAR‑100、Split MiniImageNet、CORE50（NC、NI）以及非平稳 MiniImageNet（噪声、模糊、遮挡）等多种基准。

**📈 对比分析**

通过与传统SGD以及ER、A‑GEM、MIR、GSS、Finetune、Offline等多种OCL方法进行10次重复实验，平均准确率提升约2–3个百分点，但平均遗忘率略有上升。

**⚠️ 局限性**

局限性包括：NGD‑KFAC虽提升准确率，却会略微增加遗忘倾向；对内存缓冲和阻尼参数敏感；并且在低资源嵌入式设备上的计算成本尚未充分验证。

---

## 354. Accompanist: A Runtime for Resilient Choreographic Programming

**arXiv ID:** 2603.20942 | [PDF](https://arxiv.org/pdf/2603.20942v1)

**作者:** Viktor Strate Kløvedal `[一作]` (University of Southern Denmark), Fabrizio Montesi `[通讯]` (University of Southern Denmark)

**通讯引用:** 1901 | [OpenAlex ID](https://openalex.org/A5000566520)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 Accompanist 运行时，使 Choral choreographic 程序在无中心 orchestrator 的情况下实现分布式事务（saga）并提供故障恢复

**💡 创新点**

通过将 orchestrator 去中心化为侧车并使用可重放的运行时实现容错，而无需改动编程语言或编译器；同时给出正式的安全性与可恢复性证明

**🔧 技术方法**

Choral 语言、Java 侧车、可定制通道（gRPC）、持久化消息队列、重放机制、Toxiproxy 网络仿真、AWS EKS/Kubernetes、Temporal 工作流框架

**📊 数据集**

开放源码微服务工作负载：Online Boutique、Hotel Reservation、Warehouse Saga（Temporal 案例）

**📈 对比分析**

与原始 orchestrator 以及 Temporal 对比，使用微基准、跨区域部署、尾部延迟等多维度测评；在高网络延迟、跨区场景下平均延迟可提升 32%~55%，Saga 中位延迟 5.9× 低于 Temporal（87 ms 对比 587 ms），尾部延迟亦显著改善

**⚠️ 局限性**

需要确定性、幂等事务、持久化消息队列；侧车部署成本和可伸缩性有限；无法支持动态参与者加入或网络失效时的全局恢复协调

---

## 355. User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction

**arXiv ID:** 2603.20939 | [PDF](https://arxiv.org/pdf/2603.20939v1)

**作者:** Yuren Hao `[一作]` (University of Illinois at Urbana-Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出并实现了一种名为VARS的基于冻结骨干的个性化检索框架，用长短期双向量表示用户偏好，并在检索评分中加入用户向量实现持续个性化。

**💡 创新点**

创新点在于通过弱交互奖励在线更新长短期双向量，形成共享偏好空间；该向量仅通过低秩偏置调节重排序器，既不需要微调任何骨干模型，也能在多会话场景中持续捕获用户偏好。

**🔧 技术方法**

使用了冻结的聊天LLM、嵌入模型和重排序器；轻量级的JSON格式偏好提取器；基于REINFORCE的在线奖励更新；以及基于低秩向量的检索评分与偏好记忆。

**📊 数据集**

在MultiSessionCollab基准上进行评估，包含60个用户配置、60个会话（共3600个会话）以及覆盖数学与代码任务；偏好提取训练集包含564K条示例。

**📈 对比分析**

与Vanilla、Contextual、All-memory、Reflection、RAG等六种基线对比，VARS在任务成功率、超时率和用户消耗（token数）上均排名第一，尤其显著降低了超时率和用户交互成本。

**⚠️ 局限性**

局限性包括仅在LLM模拟器上测试，奖励信号依赖关键词启发式，超参数设置经验性，缺乏对真实用户、长时间交互以及更丰富偏好类型的验证；同时存在隐私与长期存储的潜在风险。

---

## 356. MOELIGA: a multi-objective evolutionary approach for feature selection with local improvement

**arXiv ID:** 2603.20934 | [PDF](https://arxiv.org/pdf/2603.20934v1)

**作者:** Leandro Vignolo `[一作]` (Research Institute for Signals, Systems and Computational Intelligence), Matias Gerard `[通讯]` (Research Institute for Signals, Systems and Computational Intelligence)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种多目标遗传算法 MOELIGA，用于特征选择；

**💡 创新点**

创新点包括：1）演化本地改进策略；2）基于拥挤的适应度共享机制；3）使用 sigmoid 变换增强小特征集的权重；4）添加基于最近邻的几何距离目标以抑制过拟合；

**🔧 技术方法**

技术：多目标遗传算法、分层初始化、子群体演化、适应度共享、决策树分类器、UAR 评估、k‑近邻距离度量；

**📊 数据集**

使用 14 个公开数据集，涵盖 2 类和多类任务，特征维数从 14 到 20,531；

**📈 对比分析**

与 11 种主流特征选择方法（RFE、Kbest、MI、SFS、SFFS、Boruta、ReliefF、SURF、SURF* 等）比较，MOELIGA 在大多数数据集上获得更小特征集且分类性能相当或更好；

**⚠️ 局限性**

局限性：需要多次交叉验证导致计算成本高；对子群体大小、代数等超参数敏感；缺乏理论证明其最优性；

---

## 357. Deep Adaptive Rate Allocation in Volatile Heterogeneous Wireless Networks

**arXiv ID:** 2603.20926 | [PDF](https://arxiv.org/pdf/2603.20926v1)

**作者:** Gregorio Maglione `[一作]` (University of London), Touraj Soleymani `[通讯]` (University of London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于 Transformer 预测与 DQN 决策的多路径调度框架 DARA，能在高速移动环境下预测并提前分配拥塞窗口。

**💡 创新点**

创新点在于将短时（100–500 ms）拥塞状态预测与连续的 CWND 比例控制结合，消除传统反应式调度的观察-响应滞后。

**🔧 技术方法**

使用 Transformer 进行路径拥塞预测、DQN 进行多目标奖励优化，并通过 Mininet+MP‑DCCP 实时执行。

**📊 数据集**

采用了从车载移动用户收集的 30M 条数据样本和 MIT Mahi‑Mahi 的5条实时蜂窝信号轨迹进行训练与评估。

**📈 对比分析**

与 RR、CPF、BPF、Peekaboo、BLEST 等8种调度器对比，在文件传输、YouTube 及实时流中平均提升 10–30 % 传输效率、显著降低延迟和缓冲事件。

**⚠️ 局限性**

局限在于需大量预训练数据和手工调节奖励权重，对极端高波动轨迹仍可能表现不佳，并且仅在两条路径场景验证。

---

## 358. Discriminative Representation Learning for Clinical Prediction

**arXiv ID:** 2603.20921 | [PDF](https://arxiv.org/pdf/2603.20921v1)

**作者:** Yang Zhang `[一作]` (University of Hong Kong), Shi Li `[通讯]` (Columbia Univeristy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种基于直接结果对齐的监督学习框架，显式优化表示几何以最大化类间均值差异相对类内方差。

**💡 创新点**

将 Rayleigh 商正则化引入监督学习，通过最大化类间距离与类内方差比来构建更具判别力的嵌入。

**🔧 技术方法**

使用深度 Transformer/时间卷积编码器，并在损失中加入判别正则项，同时利用批次统计估计均值与协方差。

**📊 数据集**

在多中心结构化 EHR 数据上评估，包括住院死亡、30 天再入院和 48 小时急性恶化等二分类任务。

**📈 对比分析**

与掩码预训练、自回归、对比预训练及无正则化监督模型进行单一阶段训练比较，结果在 AUROC、AUPRC、Brier 及 ECE 上均优于基线，且样本效率提升约 40%。

**⚠️ 局限性**

仅在二分类、标签相对丰富的 EHR 任务上验证，缺乏对稀疏标签、多任务、连续时间或大规模预训练的探究。

---

## 359. Mitigating Shortcut Reasoning in Language Models: A Gradient-Aware Training Approach

**arXiv ID:** 2603.20899 | [PDF](https://arxiv.org/pdf/2603.20899v1)

**作者:** Hongyu Cao `[一作]` (Arizona State University), Yanjie Fu `[通讯]` (Arizona State University)

**通讯引用:** 6277 | [OpenAlex ID](https://openalex.org/A5032187620)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种短路感知推理训练（SART）框架，通过梯度对齐与答案梯度集中度识别并抑制训练中的“捷径”学习，提升大型语言模型在推理任务下的鲁棒性。

**💡 创新点**

创新点在于：1）用梯度相似度与答案-推理梯度比率构造“ShortcutScore”来精准检测促成捷径的样本；2）在训练动态中引入梯度手术（gradient surgery），将捷径梯度投影到与验证梯度正交或抑制答案梯度的子空间，既降低其权重又修正其方向；3）实现无架构改动、无额外标注的梯度级干预。

**🔧 技术方法**

主要技术包括：梯度对齐度量（cosine similarity）、答案梯度集中度计算、指数样本加权、梯度投影（正交投影/削弱答案梯度）、最小化-最大化优化框架、周期性验证梯度更新。

**📊 数据集**

使用了三种人工构造的推理基准：Math-Arithmetic、Financial-Analysis 与 Causal-Reasoning，各自包含 2,000 训练、500 验证、1,000 测试样本，训练集在 70% 处注入捷径规则，验证/测试全遵循真实规则。

**📈 对比分析**

与 11 种基线（SFT、Self-Consistency、Data Filtering、JTT、Focal Loss、Group DRO、IRM、V-REx、Fishr、LfF、Influence Filtering）对比，SART 在整体准确率达到 92.5%、鲁棒性 87.9% 及推理一致性 85.5%，分别比最强基线提高约 +16.5 个百分点的准确率和 +40.2 个百分点的鲁棒性。

**⚠️ 局限性**

主要限制包括：1）需周期性计算每个样本梯度，计算开销约为 SFT 的 2.5 倍；2）对验证梯度的依赖要求有足够代表性的验证集；3）实验基准为人工合成数据，未验证在大规模真实数据上的迁移效果；4）超参数（λ、γ、ρ）对性能影响显著，需要细致调优。

---

## 360. Beyond the Birkhoff Polytope: Spectral-Sphere-Constrained Hyper-Connections

**arXiv ID:** 2603.20896 | [PDF](https://arxiv.org/pdf/2603.20896v1)

**作者:** Zhaoyi Liu `[一作]` (University of Maryland), Ang Li `[通讯]` (University of Maryland)

**通讯引用:** 4448 | [OpenAlex ID](https://openalex.org/A5100413657)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的多流残差连接——Spectral‑Sphere‑Constrained Hyper‑Connections（sHC），通过将残差矩阵约束从Birkhoff多面体改为谱半径球，允许负值实现减法交互，并且利用SVD+Cayley等技术实现高效、无阶乘的参数化，解决了身份退化、表达瓶颈和参数化开销等问题。

**💡 创新点**

创新点：① 约束空间从刚性多面体转为谱半径球，保留均值不变性并允许负交互；② 在零边缘子空间分解残差矩阵，使用SVD和Cayley变换生成正交矩阵并限制奇异值，实现可控谱约束；③ 在保持梯度稳定的同时显著提升模型表达能力，减少对多维度混合的参数负担。

**🔧 技术方法**

使用的技术：谱半径球约束、零边缘子空间分解、SVD+Cayley正交生成、奇异值上限、动态生成残差矩阵、对比实验与标准残差、HC、mHC、mHC-lite，评估梯度、训练吞吐、参数规模等。

**📊 数据集**

训练数据集：FineWeb‑Edu、OpenWebText；零样本泛化评估：C4、Dolma V1.5、Falcon RefinedWeb、RedPajama、Wikitext‑103。

**📈 对比分析**

比较方法：在相同模型规模（12层0.12B和24层0.36B）和相同训练代价下，比较训练/验证损失、零样本perplexity以及梯度稳定性和训练吞吐。sHC在训练损失、验证损失和所有五个测试语料库的perplexity上均优于RC、HC、mHC和mHC‑lite，且保持与mHC相当的梯度稳定性和吞吐速度。

**⚠️ 局限性**

局限性：① 参数量仍随流数n^3增长，极大n下仍需更高算力；② 对非常深的网络累积谱控制尚未完全理论证明；③ 在非语言任务或非常不同的结构上效果尚待验证；④ 需要进一步研究在不同规模模型间的可迁移性和最佳超参选择。

---

## 361. Semantic Sections: An Atlas-Native Feature Ontology for Obstructed Representation Spaces

**arXiv ID:** 2603.20867 | [PDF](https://arxiv.org/pdf/2603.20867v1)

**作者:** Hossein Javidnia `[一作]` (Dublin City University), Hossein Javidnia `[通讯]` (Dublin City University)

**通讯引用:** 583 | [OpenAlex ID](https://openalex.org/A5002473893)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了“语义段”（semantic section）这一新的本地到全局特征表示方法，并通过发现与认证管线在冻结的 LLM 框架中检索和分类这种结构。

**💡 创新点**

核心创新在于把全局特征向量的统一性假设替换为 atlas‑native 的“本地代表可传输族”概念，形成了树局部、可全球化、扭曲三种语义段类型的理论分类。

**🔧 技术方法**

技术手段包括基于图的种子传播、跨重叠同步、缺陷剪枝、循环一致性分类以及支持去重等步骤，理论上通过环路一致性判断是否能在 atlas 上全局化。

**📊 数据集**

实验数据集为三款冻结大型语言模型（Gemma、Llama、Qwen）的第 16 层上下文图，使用已构建的上下文图和重叠传输矩阵。

**📈 对比分析**

与传统全局向量相似度方法对比，语义段在身份恢复任务中实现完美回溯，而原始向量相似度仅在阈值低时能恢复少量配对，显示语义段在受阻塞表示空间中的显著优越性。

**⚠️ 局限性**

局限性包括实验仅覆盖少数模型与层级、发现管线相对保守、缺乏对不同任务和更大范围的验证，以及对超参数敏感性尚未深入探讨。

---

## 362. Alignment Whack-a-Mole : Finetuning Activates Verbatim Recall of Copyrighted Books in Large Language Models

**arXiv ID:** 2603.20957 | [PDF](https://arxiv.org/pdf/2603.20957v1)

**作者:** Xinyue Liu `[一作]` (Stony Brook University), Tuhin Chakrabarty `[通讯]` (Stony Brook University)

**通讯引用:** 979 | [OpenAlex ID](https://openalex.org/A5021070482)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对前沿大语言模型进行细化训练，让其在接收情节摘要提示时自动扩展为完整文本，从而展示模型能够从预训练权重中提取并复现版权书籍内容。

**💡 创新点**

证明仅凭语义描述提示即可触发模型回忆并复现大量版权文本；揭示跨作者、跨模型的记忆一致性，表明预训练数据重叠导致的安全漏洞。

**🔧 技术方法**

使用finetuning任务（Prompt→Plot→Verbatim）、BMC@k评估指标、最长片段统计、跨作者实验、跨模型Jaccard相似度以及对预训练语料的匹配搜索。

**📊 数据集**

采用81本版权期内书籍（47位作者）作为测试集；训练集包含Haruki Murakami、Virginia Woolf等作者的作品；对照检索使用Common Crawl、Books3、LibGen以及公开预训练语料DCLM-Baseline、OLMo-3。

**📈 对比分析**

与未finetuned对齐基线对比，BMC@5从约7%提升至40–60%；跨作者finetuning后仍可召回60–80%文本；三模型间相关系数≥0.90，Jaccard相似度90–97%，显示高度一致的记忆模式。

**⚠️ 局限性**

仅评估了三款模型；实验受API调用限制；未细化法律责任与适用范围；假设预训练数据完整性；对跨语言、多模态情况缺乏验证。

---

## 363. Restoring Neural Network Plasticity for Faster Transfer Learning

**arXiv ID:** 2603.20860 | [PDF](https://arxiv.org/pdf/2603.20860v1)

**作者:** Xander Coetzer `[一作]` (University of Pretoria), Anna Sergeevna Bosman `[通讯]` (University of Pretoria)

**通讯引用:** 240 | [OpenAlex ID](https://openalex.org/A5037780608)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出在迁移学习前对预训练模型的部分权重进行有针对性的重新初始化，以恢复神经网络的可塑性。

**💡 创新点**

创新点在于将选择性权重重新初始化方法（此前用于持续学习）引入迁移学习领域，专门针对无效或静止权重进行重置，从而显著提升下游任务的适应性。

**🔧 技术方法**

使用基于权重幅值的效用函数、按比例(5%、10%、25%)的裁剪以及四种重置策略（均值、均值+噪声、正态(0,0.2)、正态(0,1)）进行实验，并在ImageNet预训练模型上进行微调。

**📊 数据集**

实验数据集包括 DTD（纹理类别），Brain Tumor（医学MRI）和 Fruit25（自然水果图像）三种不同领域的图像分类数据集。

**📈 对比分析**

通过与仅使用ImageNet预训练权重的基线对比，并采用 Mann‑Whitney U 检验评估显著性，发现大部分实验在准确率上提升 0.5%–2%，在收敛速度上加速 10–90 轮，尤其在 ResNet‑50 与 ViT‑B16 上效果更为显著。

**⚠️ 局限性**

局限性包括：提升幅度相对有限，仅针对小型图像分类数据集验证；对大型模型或非图像任务的适用性未知；对学习率、批大小等超参数敏感；未提供自适应重置比例的机制。

---

## 364. AC4A: Access Control for Agents

**arXiv ID:** 2603.20933 | [PDF](https://arxiv.org/pdf/2603.20933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 365. Elite Lanes: Evolutionary Generation of Realistic Small-Scale Road Networks

**arXiv ID:** 2603.20964 | [PDF](https://arxiv.org/pdf/2603.20964v1)

**作者:** Artur Morys-Magiera `[一作]` (AGH University of Krakow), Paweł Skruch `[通讯]` (AGH University of Krakow)

**通讯引用:** 783 | [OpenAlex ID](https://openalex.org/A5046738374)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

生成逼真、带冗余的小到中等规模道路网络，并构建对应的语义分割数据集

**💡 创新点**

引入 MAP‑Elites 质量多样性进化算法与约束修复机制，实现多维度约束下高质量、多样化网络的生成

**🔧 技术方法**

使用演化算法（EA+MAP‑Elites）、波函数坍缩（WFC）、粒子群优化（PSO）和灰狼优化（GWO）进行对比

**📊 数据集**

使用在 Duckietown 实验室收集的 12×12 网格车道贴片数据（含四向连通、转弯、交叉等类型）

**📈 对比分析**

通过连通度、环路数、死端数、跨越等指标比较，MAP‑Elites 在连通度和冗余度上优于其他方法，WFC 最快但质量最低

**⚠️ 局限性**

仅基于低样本贴片，未评估生成数据对语义分割训练效果的真实性能；对更大尺度和多样化地图的适应性尚未验证

---

## 366. Beyond Expression Similarity: Contrastive Learning Recovers Functional Gene Associations from Protein Interaction Structure

**arXiv ID:** 2603.20955 | [PDF](https://arxiv.org/pdf/2603.20955v1)

**作者:** Jason Dury `[一作]` `[通讯]` (Eridos AI), Jason Dury (Eridos AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对基因扰动、药物敏感性等生物数据使用对比关联学习（CAL）训练 4 层 MLP，将共现/相互作用信息映射到关联空间，并用该空间进行跨边界的关系判别。

**💡 创新点**

证明关联≠相似性原则跨越文本与分子生物学两个领域；展示在生物学中 CAL 能成功进行 inductive transfer、对低度数基因产生更大提升，并发现质量优于数量的逆转效应。

**🔧 技术方法**

使用 4 层 MLP（含 LayerNorm、GELU、残差融合）训练 Symmetric InfoNCE 损失，结合半/双变换打分、随机负采样与在批内负采样对比；对照余弦相似度和多项 ablation 进行评估。

**📊 数据集**

Replogle K562 CRISPRi 2,285 基因扰动谱；DepMap 17,725 基因 CRISPR 拓扑；STRING 12.0 PPI（多种置信阈值）；PRISM 药物共致死性（Morgan 指纹 2048 位、L1000 转录签名 978 基因）。

**📈 对比分析**

与余弦相似度基线进行 AUC 对比；在基因实验中，CAL 交叉边界 AUC 达 0.908（比基线 0.518 提升 0.39），在 DepMap 中随机负采样后达到 0.947；在药物实验中 CAL 受限，未能显著优于基线，且出现度数混淆。

**⚠️ 局限性**

局限包括：仅在转导式评估下验证；使用 STRING 组合得分而非纯实验真值；仅试验 PCA‑50 表达嵌入，未检验其他维度或学习型嵌入；未与专用 PPI 预测模型对比；度数偏差导致混淆；实验规模有限；跨域推广需进一步验证。

---

## 367. Towards an AI Buddy for every University Student? Exploring Students' Experiences, Attitudes and Motivations towards AI and AI-based Study Companions

**arXiv ID:** 2603.20909 | [PDF](https://arxiv.org/pdf/2603.20909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 368. Before the Tool Call: Deterministic Pre-Action Authorization for Autonomous AI Agents

**arXiv ID:** 2603.20953 | [PDF](https://arxiv.org/pdf/2603.20953v1)

**作者:** Uchi Uchibeke `[一作]` `[通讯]` (APort Technologies Inc.), Uchi Uchibeke (APort Technologies Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了开放代理护照（OAP）规范，提供同步、可审计的工具调用预授权机制，并在六个主流代理框架中部署

**💡 创新点**

创新点在于将声明式可验证的 agent passport 与平台级同步 hook 结合，实现在每一次工具调用前的确定性、绕过抗性和可审计授权决策

**🔧 技术方法**

技术实现包括 Ed25519 数字签名、JSON Schema 定义的政策包、RFC 8615 的服务发现、同步 Hook 接口以及基于云/本地的 API 执行

**📊 数据集**

使用公开的 21 个政策包以及 APort Vault CTF 的 4,437 次授权决策、1,151 次攻击会话的数据集进行评估

**📈 对比分析**

通过与传统模型对齐、沙盒执行和预授权的对比实验，平均授权延迟为 53 ms（p99 < 80 ms），在最高安全门槛攻击中拒绝率达 100%（成功率 0%）

**⚠️ 局限性**

局限包括未实现委托链与 ESCALATE 机制、政策表达式受限、未验证 10,000+ 次/秒的高并发性能、以及对框架运行时安全的信任假设

---

## 369. Profit is the Red Team: Stress-Testing Agents in Strategic Economic Interactions

**arXiv ID:** 2603.20925 | [PDF](https://arxiv.org/pdf/2603.20925v1)

**作者:** Shouqiao Wang `[一作]` (Columbia University), Davide Crapis `[通讯]` (dAI Team, Ethereum Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种基于利润驱动的红队协议（profit‑driven red teaming），利用自适应对手策略在可审计的经济交互环境中对 LLM 代理进行压力测试，并通过攻击轨迹生成可用于硬化的提示规则。

**💡 创新点**

创新点在于：①不依赖预定义攻击词典或 LLM 判别器，仅使用环境返回的标量收益来驱动对手搜索；②将自适应攻击结果直接转化为简洁提示规则实现无参数硬化；③在四种经典经济博弈上系统验证了其有效性。

**🔧 技术方法**

主要技术包括：基于 TAP（Tree of Attacks with Pruning）的黑盒搜索框架改写为利润最大化；多轮对话模拟的可审计经济环境；统计显著性检验与聚类分析用于发现攻击模式；以及提示规则的自动提炼。

**📊 数据集**

实验数据集为四个经济交互环境：Ultimatum bargaining、First‑price auction、Bilateral trade 与 Provision‑point public goods game，每个环境均手工设定参数；目标代理采用 GPT‑OSS‑120B、Qwen3‑32B、MiniMax‑M2.5、GLM‑4.6、Kimi‑K2 与 GPT‑5.2 等模型。

**📈 对比分析**

比较方法：将优化前的基线对手与利润驱动优化后的对手分别在 20 个 episode 中对同一目标模型进行评估；通过平均收益、95% 置信区间及 p‑value 评估差异；实验结果显示对手收益提升 18.85–44.50，目标代理收益下降；硬化后目标收益提升 20–35 以上。

**⚠️ 局限性**

局限性：①实验仅限于四个简化的经济博弈，缺乏对真实复杂部署场景的验证；②对手搜索仅在固定协议内进行，可能未覆盖所有潜在攻击手段；③硬化方法仅依赖提示规则，可能对其他类型攻击效果有限；④未对多轮对手再训练的长期鲁棒性进行深入评估。

---

## 370. LLM-ODE: Data-driven Discovery of Dynamical Systems with Large Language Models

**arXiv ID:** 2603.20910 | [PDF](https://arxiv.org/pdf/2603.20910v1)

**作者:** Amirmohammad Ziaei Bideh `[一作]` (Graduate Center, CUNY), Jonathan Gryak `[通讯]` (Queens College and Graduate Center, CUNY)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种结合大语言模型和遗传编程的框架 LLM-ODE，用于从观测轨迹中自动发现动态系统的符号微分方程。

**💡 创新点**

创新点在于将 LLM 作为智能演化算子替代传统随机变异/交叉，利用 LLM 对高质量候选方程的模式识别来引导搜索，显著提升搜索效率和解的多样性。

**🔧 技术方法**

技术主要包括基于 LLM 的符号回归、岛屿进化策略、Pareto 前沿构建、以及多目标评估（复杂度与误差）。

**📊 数据集**

使用 91 个真实世界动力学系统（来自 ODEBench 以及扩充的高维与混沌数据集），每个系统提供训练/测试轨迹。

**📈 对比分析**

与传统遗传编程（PySR）、线性稀疏回归（SINDy）和 Transformer 生成模型（ODEFormer）比较，LLM-ODE 在所有维度、误差阈值和迭代次数下都实现了更高的发现率、更快的收敛速度和更丰富的 Pareto 前沿。

**⚠️ 局限性**

主要限制包括 LLM 推理延迟导致的计算瓶颈，以及对高性能硬件的依赖；若采用量化或低秩模型可缓解但可能牺牲一定性能。

---

## 371. LLM Router: Prefill is All You Need

**arXiv ID:** 2603.20895 | [PDF](https://arxiv.org/pdf/2603.20895v1)

**作者:** Tanay Varshney `[一作]` (NVIDIA), Davide Onofrio `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文利用LLM内部预填充激活，提出Encoder-Target Decoupling和SharedTrunkNet实现多模型路由。

**💡 创新点**

创新点在于用Fisher分离度和有效维度识别最具判别力的层，并通过外部轻量编码器预测闭源目标性能，实现高效、可解释的路由。

**🔧 技术方法**

技术包括Encoder-Target Decoupling、Fisher Separability（J）、Effective Dimensionality（d_eff）、PCA降维、MLP多输出、集成学习和Platt校准。

**📊 数据集**

使用LLMRouterBench+自制混合池，包含Claude、GPT、Qwen、Nemotron等20+模型，覆盖MMLU-Pro、LiveCodeBench、Humanity's Last Exam，共计14,469条目/模型。

**📈 对比分析**

通过与语义路由和静态模型对比，绘制成本-准确率曲线，使用P-AUCCC、MDP-AUCCC和Router Efficacy评估，SharedTrunkNet闭合45.58%准确率差距，成本节省74.31%，P-AUCCC提升14.67%，Router Efficacy提升53.62%。

**⚠️ 局限性**

局限性包括仅使用预填充激活，未对输出token成本做估计；每次查询需计算激活，增加前置开销；验证仅在部分模型和基准，跨平台泛化需进一步评估。

---

## 372. AcoustEmo: Open-Vocabulary Emotion Reasoning via Utterance-Aware Acoustic Q-Former

**arXiv ID:** 2603.20894 | [PDF](https://arxiv.org/pdf/2603.20894v1)

**作者:** Liyun Zhang `[一作]` (University of Tokyo), Fengkai Liu `[通讯]` (University of Osaka)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5101611197)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种时序敏感的多模态大语言模型 AcoustEmo，专门通过句子感知的音频 Q-Former 提取细粒度的音频特征，用于开放词汇情感识别。

**💡 创新点**

引入了时间同步滑动窗口与 Utterance-Aware Acoustic Q-Former，能够精确对齐音频段与文本句子，并捕捉微小的音调变化，从而显著提升情感推理的细粒度。

**🔧 技术方法**

使用了视觉 ViT + Visual Q-Former、预训练音频编码器 ImageBind、跨模态查询变压器 Q-Former、LLaMA-2 + LoRA 微调，以及自定义的时间同步滑动窗口。

**📊 数据集**

在 Explainable Multimodal Emotion Recognition (EMER) 任务的 EMER-Fine 测试集上进行评估。

**📈 对比分析**

与音频中心、视频中心和情感特定基线模型相比，在 EMER-Fine 上平均准确率 67.55%，召回率 65.40%，明显优于 MicroEmo（66.21/63.82）并接近 EMER（Multi）的最高水平。

**⚠️ 局限性**

在语义讽刺或背景噪声高的场景下仍易受误导，且动态窗口提取增加了计算开销，限制了实时部署的可能性。

---

## 373. Implementation of QR factorization of tall and very skinny matrices on current GPUs

**arXiv ID:** 2603.20889 | [PDF](https://arxiv.org/pdf/2603.20889v1)

**作者:** Jonas Thies `[一作]` (Delft University of Technology), Melven Röhrig-Zöllner `[通讯]`

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出并实现了针对当前 GPU 的“Q-less”高效 Tall‑Skinny QR 分解算法，比较了基于 Gramian（Cholesky‑QR2、SVQB2）与 TSQR 两类方法；

**💡 创新点**

创新点在于通过避免 Q 的写回并充分利用共享内存与核融合，实现在内存受限 regime 下 10‑300 倍的速度提升；

**🔧 技术方法**

采用了 CUDA Warp、共享内存、Roofline 性能模型、SVQB2、Cholesky‑QR2、TSQR 等技术；

**📊 数据集**

使用合成稠密矩阵，保持 mn 常数，n 取 {1,8,16,32,64}，并在 NVIDIA H100 上进行实验；

**📈 对比分析**

与 cuSOLVER 的 Householder QR 对比，Q‑less TSQR 达到 100% Roofline，SVQB2 约 50%，比 vendor 库快 10‑300 倍；

**⚠️ 局限性**

限制主要在于 TSQR 受共享内存约束，仅适用于 n≤32；对于更大 n，Q‑less 方案变得计算受限且实现复杂。

---

## 374. Incentive-Aware Federated Averaging with Performance Guarantees under Strategic Participation

**arXiv ID:** 2603.20873 | [PDF](https://arxiv.org/pdf/2603.20873v1)

**作者:** Fateme Maleki `[一作]` (Rutgers University), Farzad Yousefian `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种结合FedAvg与Nash博弈的联邦学习框架，允许客户端动态调整其数据参与量以实现激励对齐。

**💡 创新点**

将博弈论中的Nash均衡与联邦学习耦合，提出可行的参与策略更新规则，并证明全局模型与参与策略均能同时收敛。

**🔧 技术方法**

采用投影梯度博弈求解、FedAvg本地训练、L‑smooth非凸/凸分析以及随机梯度估计等技术。

**📊 数据集**

使用MNIST和CIFAR‑10图像分类数据集进行实验。

**📈 对比分析**

相较于传统FedAvg，实验表明在保持或提升全局模型性能的同时，客户端数据贡献收敛到稳定的Nash均衡；且随着局部步数H增大，通信效率得到提升。

**⚠️ 局限性**

未覆盖强凸情形、异步/动态网络设置以及实际支付/隐私保护机制，且对大规模异构客户端的鲁棒性仍有待进一步验证。

---

## 375. ReLaMix: Residual Latency-Aware Mixing for Delay-Robust Financial Time-Series Forecasting

**arXiv ID:** 2603.20869 | [PDF](https://arxiv.org/pdf/2603.20869v1)

**作者:** Tianyou Lai `[一作]` (Lanzhou University), Qilei Li `[通讯]` (Central China Normal University)

**通讯引用:** 32258 | [OpenAlex ID](https://openalex.org/A5072566242)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文针对高频金融时序预测中因传输延迟导致的零阶保持(ZOH)滞后现象，提出了一种轻量化残差瓶颈混合网络ReLaMix，能够在延迟观测下恢复并预测未来市场状态。

**💡 创新点**

创新点在于：① 将信息瓶颈压缩与残差融合相结合，显式抑制ZOH产生的冗余；② 设计了时间-特征双模混合块（Expand–Compress）与多层跳连路，兼顾高效性与表达能力；③ 采用全端到端训练，无需先做插值或补值，避免分层误差。

**🔧 技术方法**

主要技术包括信息瓶颈压缩（低维投影）、时间混合（按时间线性混合）、特征混合（扩张-压缩投影）、残差块与多级跳连、均值平方误差训练。

**📊 数据集**

使用了币安Binance上收集的秒级OHLCV特征数据：PAXGUSDT（基准）以及BTCUSDT（交叉资产检验）。

**📈 对比分析**

与LSTM、GRU、TimeMixer、TimesNet、PatchTST等多种基线在15%/25%/35% ZOH延迟下对比，ReLaMix在MSE、MAE、R²上均位列第一，且参数量仅约1.4万，远低于其他模型。

**⚠️ 局限性**

局限性包括：仅在模拟的ZOH滞后场景下验证；未考虑更复杂的网络延迟模式（如多路径、可变延迟）；模型仍为CPU/GPU实现，对极低延迟硬件部署需进一步优化。

---

## 376. Information-Based Complexity vs Computational Complexity in Phaseless Polynomial Interpolation

**arXiv ID:** 2603.21008 | [PDF](https://arxiv.org/pdf/2603.21008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 377. How AI Systems Think About Education: Analyzing Latent Preference Patterns in Large Language Models

**arXiv ID:** 2603.21006 | [PDF](https://arxiv.org/pdf/2603.21006v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 378. Ensemble of Small Classifiers For Imbalanced White Blood Cell Classification

**arXiv ID:** 2603.20856 | [PDF](https://arxiv.org/pdf/2603.20856v1)

**作者:** Siddharth Srivastava `[一作]` (University of Warwick), Till Bretschneider `[通讯]` (University of Warwick)

**通讯引用:** 2999 | [OpenAlex ID](https://openalex.org/A5074864006)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对白血球分类问题，作者提出轻量级集成模型，结合SwinV2、ConvNeXt和DinoBloom三种骨干网络，实现了对13类白血球的自动识别。

**💡 创新点**

创新点在于通过扩充公开数据、使用权重采样和非局部均值去噪，以及将三种互补的轻量化网络以logit平均方式集成，兼顾了模型轻量和对稀有类的鲁棒性。

**🔧 技术方法**

技术上采用预训练Transformer、卷积网络和专用单细胞DINO模型；训练时使用α‑平衡焦点损失、AdamW、EMA、CutMix/MixUp、数据增强；推理时用TTA并平均logit。

**📊 数据集**

数据集为ISBI WBCBench 2026核心数据（55,012张）并额外引入Acevedo‑20、Taeyeon Kim的Blood_dataset以及CellWiki的prolymphocyte，扩充后共约48k张训练样本。

**📈 对比分析**

在扩充数据下的三折交叉验证与测试集比较，集成模型TTA获得macro‑F1 0.7741、平衡精度0.7987，优于单一骨干；在官方赛果上取得0.67726的最高分。

**⚠️ 局限性**

局限主要是对形态相近的细胞（如myelocyte系列、淋巴细胞等）仍易混淆，提示需进一步设计专门专家模型以提升同系细胞间的区分。

---

## 379. Can ChatGPT Really Understand Modern Chinese Poetry?

**arXiv ID:** 2603.20851 | [PDF](https://arxiv.org/pdf/2603.20851v1)

**作者:** Shanshan Wang `[一作]` (University of Macau), Lidia S. Chao `[通讯]` (University of Macau)

**通讯引用:** 2897 | [OpenAlex ID](https://openalex.org/A5025832925)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一套评估ChatGPT对现代诗理解能力的框架，结合专业诗人对生成的诗歌解释进行评估。

**💡 创新点**

首次从内容、表现手法、思想情感、现代性和诗意五维度系统评估 LLM 对现代诗的理解，并引入原诗人自评以确保评判客观。

**🔧 技术方法**

采用 GPT‑4 通过精心设计的提示词进行诗歌解读，随后使用原诗人和另一组 LLM 进行多维度评分。

**📊 数据集**

构建了 48 篇现代中文诗歌的数据集，包含 40 篇常规现代诗和 8 篇融合古典意象的特殊现代诗。

**📈 对比分析**

通过对比原诗人评估与 LLM 评估，发现人类评分平均在 80–90 分，超过 73% 与诗人意图一致；而 LLM 评估与人类评估存在显著偏差，表现不佳。

**⚠️ 局限性**

局限性在于仅使用新近创作的 48 首诗，评估需人工高成本，LLM 评估不够可靠，且过度依赖 ChatGPT 解释可能导致读者误读。

---

## 380. SkinCLIP-VL: Consistency-Aware Vision-Language Learning for Multimodal Skin Cancer Diagnosis

**arXiv ID:** 2603.21010 | [PDF](https://arxiv.org/pdf/2603.21010v1)

**作者:** Zhixiang Lu `[一作]` (Xi'an Jiaotong-Liverpool University), Jionglong Su `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发 SkinCLIP-VL，一个资源高效的多模态皮肤癌诊断框架，采用冻结的 CLIP 视觉编码器和轻量化 Qwen2.5‑VL 解码器，并提出 Consistency‑aware Focal Alignment (CFA) 损失实现视觉与临床语义的一致性与置信度校准，产生可视化的诊断推理。

**💡 创新点**

① frozen perception + LoRA 微调实现参数 43% 下降；② CFA 损失将焦点加权、语义对齐、校准统一优化；③ 生成可视化解释推理显著提升临床信任；④ 通过元数据增强将表格信息转化为自然语言描述，强化跨模态桥接。

**🔧 技术方法**

CLIP 视觉 Transformer (冻结), Qwen2.5‑VL 量化解码器, LoRA 参数高效微调, Focal Pooling, InfoNCE 对齐, Brier‑Score 校准, 生成式推理, 多任务损失 (CE+Focal+Align+Cal+Gen)。

**📊 数据集**

ISIC 2019、Derm7pt、ISIC 2024（OOD）三大皮肤病学数据集。

**📈 对比分析**

与 EfficientNet、ResNet、ConvNeXt、MedCLIP、SkinGPT‑4、SkinVL‑MM 等多种基准对比；在 ISIC 2019 上取得 88.7% B‑ACC、0.981 AUROC、0.019 ECE，较 SkinGPT‑4 提升 6.2% B‑ACC、0.039 AUROC、75% ECE 降低；在 ISIC 2024 OOD 仍保持 85% B‑ACC、0.972 AUROC；在 Derm7pt 上 83.4% B‑ACC、0.965 AUROC；整体显著优于 13B 参数基线。

**⚠️ 局限性**

仍受限于对大规模预训练模型的依赖，极少数据场景下性能衰减；生成文本可能存在幻觉风险；跨域适配需更多真实世界验证；对模型可解释性的定量评估仍有限。

---

## 381. Joint Surrogate Learning of Objectives, Constraints, and Sensitivities for Efficient Multi-objective Optimization of Neural Dynamical Systems

**arXiv ID:** 2603.20984 | [PDF](https://arxiv.org/pdf/2603.20984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 382. SpatialFly: Geometry-Guided Representation Alignment for UAV Vision-and-Language Navigation in Urban Environments

**arXiv ID:** 2603.21046 | [PDF](https://arxiv.org/pdf/2603.21046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 383. Structural Sensitivity in Compressed Transformers: Error Propagation, Lyapunov Stability, and Formally Verified Bounds

**arXiv ID:** 2603.20991 | [PDF](https://arxiv.org/pdf/2603.20991v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]` (National Institute of Electronics and Information Technology), Abhinaba Basu (National Institute of Electronics and Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Transformer 压缩的敏感性映射，分析误差传播，并提供形式化的 per‑matrix 错误界限并验证其有效性。

**💡 创新点**

提出了每个矩阵层级的压缩敏感性层次结构、通过 Lyapunov 稳定性解释误差收缩机制、十个 Lean4 形式化定理以及跨架构（117M–8B）验证的全流程。

**🔧 技术方法**

使用结构敏感性映射、Lyapunov 稳定性理论、Lean4 形式化证明、稀疏化与低秩 SVD 组合、贪心分配策略、以及经验测量的 ρ_max 等工具。

**📊 数据集**

在 WikiText‑103、C4 数据集上进行推理评估，并使用 lm‑evaluation‑harness 对下游任务（HellaSwag、ARC‑Easy、Winogrande）进行验证。

**📈 对比分析**

通过对比压缩前后 perplexity、不同模型的 ρ_max、以及 14,040+ 配置的错误界限违规率（均为 0）评估方法性能，显示压缩敏感性层次稳定，Lyapunov 收缩机制解释误差不累积。

**⚠️ 局限性**

仅针对 117M–8B 参数模型；压缩效果不及 SparseGPT；非线性层误差未形式化；ρ_max 依赖经验测量；未探究训练时的干预和更大规模模型。

---

## 384. ECI: Effective Contrastive Information to Evaluate Hard-Negatives

**arXiv ID:** 2603.20990 | [PDF](https://arxiv.org/pdf/2603.20990v1)

**作者:** Aarush Sinha `[一作]` (University of Copenhagen), Aman Bansal `[通讯]` (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于信息理论的ECI指标，用来在不进行模型微调的情况下评估硬负样本集合的质量，并通过ECI比较不同负样本采集策略（BM25、Cross-Encoder、LLM及其混合），最终指导检索模型的训练。

**💡 创新点**

创新点在于：① 将InfoNCE下界与信息容量结合，得到对负样本规模的对数约束；② 采用谐波平均平衡硬度与安全性（最大边际），严格惩罚“误正”负样本；③ 在训练前即可预测下游检索性能，显著降低昂贵的 ablation 实验成本。

**🔧 技术方法**

使用信息理论、InfoNCE、谐波平均；Dense Retriever 训练（DistilBERT、MNRL loss）；BM25 索引、Cross-Encoder 重新排序；LLM 生成硬负样本；对比评估与 BEIR 的 nDCG@10。

**📊 数据集**

基准数据集：MS‑MARCO（10000 query‑passage 训练集）；评估数据集：12 个 BEIR 公开数据集。

**📈 对比分析**

通过计算各采样策略的 ECI 分数与在 BEIR 上的 nDCG@10 进行对比，发现 BM25+Cross‑Encoder 的 ECI 最高（1.25），对应平均 nDCG 0.337，明显优于单一策略；LLM‑only 仅 0.26，表现最差；实验显示 ECI 与下游性能高度相关（Pearson 0.91），而单纯的硬度或梯度幅值则相关性较低。

**⚠️ 局限性**

局限性：仅使用 MS‑MARCO 的 10k 条记录进行评估，可能无法完全覆盖更大规模数据集；ECI 仅在所测试的 embedding 与负采样方法上验证，未在所有模型或生成策略下彻底检验；此外，对 LLM 的生成质量依赖于 prompt 与接口，可能影响结果一致性。

---

## 385. Interpreting the Synchronization Gap: The Hidden Mechanism Inside Diffusion Transformers

**arXiv ID:** 2603.20987 | [PDF](https://arxiv.org/pdf/2603.20987v1)

**作者:** Emil Albrychiewicz `[一作]` (University of California), Viola Zixin Zhao `[通讯]` (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 Diffusion Transformer（DiT）中的同步间隙（synchronization gap）进行理论分析和实验验证，揭示其在生成过程中的频率层级化行为。

**💡 创新点**

首次将非平衡统计物理中耦合 OU 过程的相位转变映射到 Transformer 的自注意力机制，提出了空间路由与模式调制两条差分响应通道，并解释了同步间隙在强耦合下的崩塌。

**🔧 技术方法**

使用自注意力的耦合门控、线性化分析、两分量高斯混合模型、SNR 公式和经验模式投影等技术，结合 DiT 的残差结构对差分模式进行定量推导。

**📊 数据集**

实验基于 DiT‑XL/2 预训练模型，在 ImageNet 图像数据集上使用 DDIM/DDPM 采样，并通过 ResNet‑50 特征空间余弦相似度和像素级尺度分解评估生成结果。

**📈 对比分析**

采用同步间隙的深度梯度、特征空间相似度和尺度分解等多种评估方法，结果显示：同步间隙存在、在强耦合下显著收敛、低频结构比高频先完成分支，并且与理论预测一致；未与其他生成模型做性能对比，重点关注可解释性而非生成质量。

**⚠️ 局限性**

局限包括：未直接测量每层的得分增益 γ，平均场投影忽略高阶模式耦合，假设差分协方差对角化，实验仅覆盖单一 DiT‑XL/2 版本，未验证不同数据集或条件模态下的泛化。

---

## 386. Confidence Freeze: Early Success Induces a Metastable Decoupling of Metacognition and Behaviour

**arXiv ID:** 2603.21043 | [PDF](https://arxiv.org/pdf/2603.21043v1)

**作者:** Zhipeng Zhang `[一作]` (China Mobile Research Institute), Hongshun He `[通讯]` (China Mobile Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在三组实验中，研究人员利用多次逆转的双臂老虎机任务，探究早期成功经验如何导致自信冻结（confidence‑freeze）现象，并评估两种干预方式（显式轨迹提示与元认知提示）对锁定状态的缓解效果。

**💡 创新点**

创新点包括：①提出自信冻结机制，将早期成功对信念–行为映射的动态解耦视为可逆的学习模式；②揭示自信冻结为元认知控制的失效，而非单纯的过度自信；③展示环境与认知两条等效干预路径，可实时“解冻”行为；④首次将政策粘性参数与行为锁定联系起来，提供可量化的机制模型。

**🔧 技术方法**

使用的技术与方法：多逆转二臂老虎机实验设计、每3次试次收集的自信评分、混合效应逻辑回归、危害率与生存分析、泊松/负二项回归、强化学习模型（加入政策粘性ϕ）以及对冻结指数、持续长度、转移概率的统计比较。

**📊 数据集**

数据集：总样本332名参与者，19,920次试次；三组实验分别为：实验1（N=99，10次预备期+50次主期）、实验2（N=110，加入显式轨迹）、实验3（N=123，加入元认知提示）。

**📈 对比分析**

比较方法：对比高低早期成功组的持续长度、冻结指数、危害率和转换概率；对比两种干预后在冻结指数、持续长度和政策粘性上的改善。结果显示：高早期成功组持续长度显著增加，冻结指数↑；显式轨迹与元认知提示均显著降低冻结指数（≈半幅）并恢复转移概率，干预效果等价（Cohen's d≈0.7）。

**⚠️ 局限性**

局限性：①自信评分仅每3次试次记录，可能错过更细微的认知变化；②缺乏神经影像或实时信号验证假设；③干预效果仅在单次实验后测，无法评估长期或跨任务迁移；④实验设计为实验室任务，外部生态效度待进一步验证。

---

## 387. Consensus-Driven Group Recommendation on Sparse Explicit Feedback: A Collaborative Filtering and Choquet-Borda Aggregation Framework

**arXiv ID:** 2603.21012 | [PDF](https://arxiv.org/pdf/2603.21012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 388. KLDrive: Fine-Grained 3D Scene Reasoning for Autonomous Driving based on Knowledge Graph

**arXiv ID:** 2603.21029 | [PDF](https://arxiv.org/pdf/2603.21029v1)

**作者:** Ye Tian `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**通讯引用:** 10908 | [OpenAlex ID](https://openalex.org/A5025573294)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于知识图谱的LLM推理框架KLDrive，用于在3D驾驶场景中实现细粒度问答。

**💡 创新点**

创新点在于通过能量模型对多源感知证据进行稀疏化与冲突消解，构建可靠场景知识图，并将LLM作为受限工具调用者实现可解释、可追溯的推理流程。

**🔧 技术方法**

核心技术包括多源感知融合、跨时域证据池化、能量基础的实体精炼、基于图谱的关系算子库，以及基于少量示例的Plan–Execute–Observe循环LLM规划。

**📊 数据集**

使用公开的大规模驾驶问答基准NuScenes-QA和GVQA进行评估。

**📈 对比分析**

在两大基准上均取得SOTA表现，NuScenes-QA总准确率65.04%（提升约4.9点）且计数任务提升46点；GVQA SPICE得分42.45，整体在多种方法中位居前列。

**⚠️ 局限性**

局限性主要在于推理时延较高（A6000平均1.26s/问，Jetson约57s/问）以及对感知质量仍高度依赖，导致在低质量感知场景下性能下降。

---

## 389. Knowledge Boundary Discovery for Large Language Models

**arXiv ID:** 2603.21022 | [PDF](https://arxiv.org/pdf/2603.21022v1)

**作者:** Ziquan Wang `[一作]` (China University of Petroleum-Beijing), Zhongqi Lu `[通讯]` (China University of Petroleum-Beijing)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于强化学习的知识边界发现框架KBD，用于动态探索LLM的可回答与不可回答问题边界；

**💡 创新点**

创新点在于将熵值作为奖励信号，将知识边界建模为POMDP，并通过RL策略迭代生成临界问题；

**🔧 技术方法**

主要技术包括Q‑learning、熵奖励函数、贝叶斯信念状态编码以及对话交互的POMDP框架；

**📊 数据集**

实验使用ChatGLM3‑6B、ChatGLM2‑6B、LLaMA 7B‑Chat等目标模型，并以GLM‑4‑9B生成问题；

**📈 对比分析**

与人类编写的数据集(KUQ、Sware、Infeasible Benchmark)以及随机/专家提问对比，KBD生成的问题集在EER和F1指标上与人类数据集相当，性能优于随机提问；

**⚠️ 局限性**

局限性包括收敛速度相对较慢、依赖熵阈值调参、对不同LLM的泛化性待验证、以及RL探索的样本效率问题。

---

## 390. DSL-R1: From SQL to DSL for Training Retrieval Agents across Structured and Unstructured Data with Reinforcement Learning

**arXiv ID:** 2603.21018 | [PDF](https://arxiv.org/pdf/2603.21018v1)

**作者:** Yunhai Hu `[一作]` (New York University), Nan Du `[通讯]` (Matter Innovation Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将 SQL 逻辑与向量检索融合的 DSL-R1 框架，并通过强化学习优化 DSL 生成器，以实现结构化与非结构化数据的统一检索。

**💡 创新点**

创新点在于：① 设计了包含 SQL 语句和向量搜索的统一 DSL；② 通过 RL 结合规则奖励与检索质量奖励实现可执行性与语义匹配的双重优化；③ 引入 DAPO 进一步提升训练稳定性和收敛速度。

**🔧 技术方法**

使用了强化学习（GRPO 与 DAPO）、大型语言模型（Qwen3-4B/8B、GPT-4o）、向量检索模型（如 CLIP/BLIP-2 的嵌入）以及自定义的多项奖励函数。

**📊 数据集**

在自建的工业电邮检索数据集（包含结构化元数据与非结构化文本/附件）和 ArxivQA 基准上进行评估。

**📈 对比分析**

相较于传统检索（BM25、ColQwen）与预训练 LLM（Qwen、GPT-4o）等基线，DSL-R1 在 Hit@1、Hit@3、MRR、NDCG 等指标上提升 9.6–15.8% 甚至 86.1% 的 MRR；同时通过奖励设计将查询时延降低 45%。

**⚠️ 局限性**

局限性包括：只支持单轮查询，未处理多轮对话；仅用单一 RL 过程，缺乏多智能体协同；目前仅覆盖文本/结构化数据，未扩展到图像、音频、视频等多模态；奖励规则的手工设计可能在更复杂数据上适应性不足。

---

## 391. SURF: Signature-Retained Fast Video Generation

**arXiv ID:** 2603.21002 | [PDF](https://arxiv.org/pdf/2603.21002v1)

**作者:** Kaixin Ding `[一作]` (University of Hong Kong), Hengshuang Zhao `[通讯]` (University of Hong Kong)

**通讯引用:** 34652 | [OpenAlex ID](https://openalex.org/A5078109015)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出SURF框架，先用预训练模型在最优分辨率下生成低分辨率预览，再用轻量Refiner在高分辨率下细化视频，实现快速生成。

**💡 创新点**

创新点在于噪声重定位（noise reshifting）技术、动态尺度调度和可插拔Refiner，既保留原模型的布局、语义和运动签名，又获得12倍以上的加速。

**🔧 技术方法**

核心技术包括：噪声重定位、动态尺度缩放、Shift‑Window自注意力、流匹配的轻量Refiner、以及对预览与目标映射的训练范式。

**📊 数据集**

使用合成的100k LR‑HR视频对进行训练，测试集来自VBench、Videophy、PhyGenBench等公开视频数据集。

**📈 对比分析**

与Wan 2.1、30%/50%步骤、SVG、DMD等方法对比，720p视频生成速度提升12.5×，1080p提升43×，质量指标（QS、AQ、MS等）与基线相当或略优，用户偏好测试显示与基线无显著差异。

**⚠️ 局限性**

局限性：需要多阶段训练和显存支持，部分极端场景下仍出现细节失真或运动不连贯；在更大规模模型或多模态任务上的通用性尚待验证。

---

## 392. OrbitStream: Training-Free Adaptive 360-degree Video Streaming via Semantic Potential Fields

**arXiv ID:** 2603.20999 | [PDF](https://arxiv.org/pdf/2603.20999v1)

**作者:** Aizierjiang Aiersilan `[一作]` (George Washington University), Zhangfei Yang `[通讯]` (George Washington University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个无训练需求的360°视频流框架OrbitStream，结合语义场景理解与鲁棒控制，针对远程遥操作场景实现自适应带宽与视窗预测。

**💡 创新点**

创新点在于：①将语义对象视作引力场，用连续粒子动力学（Gravitational Viewport Prediction）预测用户视窗；②采用带饱和非线性的PD控制器（Saturation‑Based PD）实现缓冲区稳定，兼顾实时性与可解释性；③不依赖历史用户数据或深度学习模型，零训练零个性化配置。

**🔧 技术方法**

主要技术：球面哈弗森距离的引力势能模型、Boltzmann分布估计视窗概率、随机微分方程（SDE）模拟注视动力学、tanh饱和PD控制器、基于语义权重的块级码率分配。

**📊 数据集**

使用了人工合成与公开的遥操作轨迹数据（如Xu等、Wu等数据集），配合YOLOv5检测框架生成语义标签，进行360°视频分块与编码实验。

**📈 对比分析**

与12种基线（包括MPC、BOLA‑E、FastMPC、Pensieve、FLARE等）在3600次Monte‑Carlo仿真中对比，OrbitStream平均QoE为2.71（仅次于BOLA‑E 2.80），视窗预测准确率94.7%，缓冲区波动仅0.42 s，决策延迟1.01 ms。

**⚠️ 局限性**

局限性：①控制循环相对耗时（≈1 ms），不适合极低功耗边缘设备；②对前端目标检测精度高度依赖，检测失误会直接影响视窗预测；③语义质量权重固定，无法自适应不同用户或任务的优先级。

---

## 393. The survival of the weakest in a biased donation game

**arXiv ID:** 2603.20998 | [PDF](https://arxiv.org/pdf/2603.20998v1)

**作者:** Chaoqian Wang `[一作]` (Nanjing University of Science and Technology), Attila Szolnoki `[通讯]` (Centre for Energy Research)

**通讯引用:** 20486 | [OpenAlex ID](https://openalex.org/A5030126548)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出一种带有独立合作偏置参数θ_C和θ_T的偏置Tit-for-Tat策略，并在捐赠博弈中与合作者C、背叛者D以及自身进行互动。

**💡 创新点**

发现即使在极端博弈条件下，弱势的T策略也能通过“弱者生存”机制在结构化群体中最终占据主导，从而扩展了对非转移性竞争系统的理解。

**🔧 技术方法**

采用Monte Carlo模拟在二维周期边界格子上演化三策略系统，并辅以理论分析推导无结构群体的均衡条件。

**📊 数据集**

使用自生成的随机初始分布（L×L格子上的策略随机分配）进行仿真，无需外部真实数据集。

**📈 对比分析**

通过比较结构化群体与无结构群体的相位图，证明隐藏的T相位仅在空间结构中出现，展示了空间相关性对演化结果的重要影响。

**⚠️ 局限性**

研究仅考虑边缘选择强度，并限制在二维正方形格子上，缺乏对更复杂网络或真实社会数据的验证，且未探讨非边缘选择强度下的行为。

---

## 394. When Does Content-Based Routing Work? Representation Requirements for Selective Attention in Hybrid Sequence Models

**arXiv ID:** 2603.20997 | [PDF](https://arxiv.org/pdf/2603.20997v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]` (National Institute of Electronics and Information Technology), Abhinaba Basu (National Institute of Electronics and Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性研究混合递归-注意力架构中的路由机制，设计了可切换的FCI实验平台，并在20+实验中揭示路由悖论；

**💡 创新点**

发现学习式内容路由必须依赖一次softmax注意力生成的潜在34维子空间，证明注意力是构造关系式表示的关键；

**🔧 技术方法**

采用softmax注意力、值聚合、随机投影、线性注意力、记忆库、上下文 bandit、对比预训练等多种机制，搭配Bloom Filter、BM25等非学习式索引；

**📊 数据集**

使用三组任务：合成的远程证据检索、MQAR 关联回忆基准、HotpotQA 多跳问答；

**📈 对比分析**

通过路由精度与任务准确率对比，softmax注意力达98-100%路由精度；Bloom Filter、BM25分别为90.9%、82.7%；其他所有学习式方法仅提升至15-29%；对比预训练仅1.6-2.2%；显示出一层注意力的阶跃提升；

**⚠️ 局限性**

局限性在于模型规模小（200K-884K参数）、仅测试三类任务、未评估大规模预训练模型、仅用InfoNCE对比预训练、未尝试近似注意力方法、实现为纯Python扫描，可能影响可扩展性。

---

## 395. Reading Between the Lines: How Electronic Nonverbal Cues shape Emotion Decoding

**arXiv ID:** 2603.21038 | [PDF](https://arxiv.org/pdf/2603.21038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 396. Fuel Consumption Prediction: A Comparative Analysis of Machine Learning Paradigms

**arXiv ID:** 2603.21034 | [PDF](https://arxiv.org/pdf/2603.21034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 397. On the Bit Error Rate Fluctuation Induced by Multipath Interference in the Coherent Regime for Intra Data Center Applications

**arXiv ID:** 2603.20995 | [PDF](https://arxiv.org/pdf/2603.20995v1)

**作者:** Wing-Chau Ng `[一作]` (Carleton University), Scott Yam `[通讯]` (Queen's University)

**通讯引用:** 2067 | [OpenAlex ID](https://openalex.org/A5084405281)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

分析并理论解释在数据中心短链路内，光学多径干扰（MPI）在相干域中因相位偏移导致星座扩张/收缩，从而导致比特错误率（BER）波动。

**💡 创新点**

首次将相位偏移与星座尺寸变化联系起来，阐释了相干域中MPI导致BER波动的根本原因，并通过理论与仿真验证了实验观察到的现象。

**🔧 技术方法**

使用PAM-4调制、Wiener相位噪声模型、单反射理论模型、15阶前馈均衡器、LMS自适应算法以及欧氏距离分析。

**📊 数据集**

采用随机生成的9百万个106.25 GBaud PAM-4符号（灰度编码）进行仿真，未使用真实实验数据集。

**📈 对比分析**

通过比较不同相位偏移（Φ=0, π/4, π/2, 3π/4, π）和不同光程与相干长度比（L=0.1Lc、1Lc、10Lc）下的BER曲线，发现相干域内Φ=π为最差情况，BER波动幅度最大；在相干域之外波动被平均消除。

**⚠️ 局限性**

局限在于仅考虑单一反射、忽略符号间干扰(ISI)、未使用反馈均衡器、仿真场景与实际系统可能存在差异，且相位偏移模型仅覆盖主相位范围。

---

## 398. CLT-Forge: A Scalable Library for Cross-Layer Transcoders and Attribution Graphs

**arXiv ID:** 2603.21014 | [PDF](https://arxiv.org/pdf/2603.21014v1)

**作者:** Florent Draye `[一作]` (Max Planck Institute for Intelligent Systems), Bernhard Schölkopf `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 145903 | [OpenAlex ID](https://openalex.org/A5044005697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CLT-Forge，一个统一的开源框架，实现可扩展的Cross‑Layer Transcoder（CLT）训练与解释；

**💡 创新点**

将分布式训练、压缩激活缓存、自动解释、归因图计算和交互式可视化整合到一条流水线；通过特征维度分片、对称量化和与Circuit‑Tracer的深度集成显著降低内存与计算成本；

**🔧 技术方法**

使用CLT与Sparse Transcoder、JumpReLU、对称量化(Int8/4/2)+Zstd压缩、GPU特征分片、FSDP/DDP+AMP、Circuit‑Tracer、Dash可视化、LLM提示生成等技术；

**📊 数据集**

以LLaMA 1B为例训练300M token数据集；评估基于GPT‑2、LLaMA 3.2 1B及多语言数据集；示例句子“The opposite of 'large' is …”；

**📈 对比分析**

与Eleuther AI、Anthropic等现有CLT实现对比；在GPT‑2上实现≈0.8解释方差、≈0.8替代分数、≈0.95图完整度；特征分片相较于DDP显著提升显存利用率，量化导致2–3%重建误差；

**⚠️ 局限性**

仍受CLT参数量大导致训练成本高、Attention归因未实现、直接优化替代分数不稳定、缺乏更高效的架构以及仅支持无稀疏激活的CLT等限制。

---

## 399. ALL-FEM: Agentic Large Language models Fine-tuned for Finite Element Methods

**arXiv ID:** 2603.21011 | [PDF](https://arxiv.org/pdf/2603.21011v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 400. When Minor Edits Matter: LLM-Driven Prompt Attack for Medical VLM Robustness in Ultrasound

**arXiv ID:** 2603.21047 | [PDF](https://arxiv.org/pdf/2603.21047v1)

**作者:** Yasamin Medghalchi `[一作]` (University of British Columbia), Ilker Hacihaliloglu `[通讯]` (University of British Columbia)

**通讯引用:** 3117 | [OpenAlex ID](https://openalex.org/A5015205742)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了医学视觉语言模型在超声多选问答中的鲁棒性，提出使用大型语言模型生成细微可接受的文本攻击并结合MCTS迭代逼近目标模型的攻击方法。

**💡 创新点**

创新点在于自动化生成临床可接受的微小文本扰动，而非手工恶意示例，并系统评估不同攻击者LLM规模与目标模型置信度对攻击成功率的影响。

**🔧 技术方法**

采用大型语言模型（Qwen-7B、Qwen-30B、GPT-4.1 mini）进行编辑生成，使用蒙特卡罗树搜索（MCTS）迭代寻找攻击序列，并利用目标VLM的对数概率评估攻击效果。

**📊 数据集**

使用U2‑Bench超声诊断子集（1305张图像–问题对），在此基础上筛选先行正确的实例进行攻击实验。

**📈 对比分析**

对比了三种Med‑VLM（MedGemma-4B-IT、LLaVA-Med-7B、QoQ-Med-7B），结果显示攻击后准确率从约40%降至13–30%之间，攻击成功率在15–25%之间，说明模型存在显著鲁棒性缺陷。

**⚠️ 局限性**

局限包括仅针对多选问答任务，未扩展至报告生成等开放式任务；攻击生成依赖外部LLM且易受语言混淆影响；缺乏对不同领域、不同语境下鲁棒性的全面评估。

---

## 401. TabPFN Extensions for Interpretable Geotechnical Modelling

**arXiv ID:** 2603.21033 | [PDF](https://arxiv.org/pdf/2603.21033v1)

**作者:** Taiga Saito `[一作]` (Tohoku University), Stephen Wu `[通讯]` (Research Organization of Information and Systems, Institute of Statistical Mathematics)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用TabPFN及其扩展库，在地质工程的土壤分类与力学参数缺失值插补任务中，展示了嵌入相似度分析、模型原生后验分布以及SHAP特征重要性等可解释性工具。

**💡 创新点**

创新点在于：①在无监督的嵌入空间中自动识别土壤类型；②通过迭代条件均值推断实现多参数缺失值插补；③提供模型内置后验分布与SHAP解释，无需重新训练。

**🔧 技术方法**

技术包括：TabPFN transformer 基础模型、TabPFN-ext（嵌入提取、SHAP解释）、余弦相似度分析、迭代条件均值推断。

**📊 数据集**

数据集为：合成的N‑Vs（16训练+16测试）以及 BM/AirportSoilProperties/2/2025（2766训练+20测试），涵盖六个指数属性和五个力学参数。

**📈 对比分析**

与传统回归/贝叶斯模型比较时，TabPFN在有限样本下实现 1.00 的分类准确率，并在四个力学参数上显著降低 RMSE；同时提供可解释的后验分布；但对 C_v 的预测并未提升。

**⚠️ 局限性**

局限性在于：对训练样本分布高度依赖，外推能力受限；迭代插补假设后验已良好校准，若不成立可能误导；模型的离散化输出偶尔产生极端尾部。

---

## 402. Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO

**arXiv ID:** 2603.21016 | [PDF](https://arxiv.org/pdf/2603.21016v1)

**作者:** Jinquan Zheng `[一作]` (East China Normal University), Guoxiu He `[通讯]` (East China Normal University)

**通讯引用:** 179 | [OpenAlex ID](https://openalex.org/A5000341481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Permutation-Aware Group Relative Policy Optimization (PA-GRPO)，一种在训练时通过交叉排列优势和一致性奖励来减轻 LLM 在多项选择和对比评估中的选择偏差的框架。

**💡 创新点**

创新点在于构造排列组，将同一语义实例的不同排列视为一组并在训练时使用跨排列优势估计和一致性奖励，迫使模型在不同排列下保持语义一致，从而解决传统 RL 的排列盲点。

**🔧 技术方法**

采用基于 GRPO 的强化学习框架，加入交叉排列优势 (A_PA) 与一致性奖励 (r_con)，使用 LoRA 微调、clip surrogate loss 以及全排列评估等技术。

**📊 数据集**

使用七个公开基准数据集：LLM-as-a-Judge 类别的 MT-Bench、JudgeBench、PreferenceBench、RewardBench；MCQ 类别的 ARC-Challenge、TinyMMLU、GPQA；训练时采集 Chatbot Arena（对比评估）和 MMLU 训练集（推理评估）。

**📈 对比分析**

与 inference‑time 校准方法（PriDe、CalibraEval、UniBias）及 training‑time 对齐方法（PIF、GRPO）在 Llama‑3.1‑8B、Qwen3‑8B、Qwen3‑32B 三大模型上进行对比，PA‑GRPO 在一致性与一致性准确率上显著提升，同时保持与最强基线相近的准确率，例如 Llama‑3.1‑8B 在 MT‑Bench 上 Accuracy 77.6%、Consistency 88.0%。

**⚠️ 局限性**

局限性包括仅针对离散选择任务设计，难以直接推广到开放式生成；评估仅在英文基准上，未检验多语言或其他系统性偏差；以及对更大模型、长文本一致性评估仍需进一步研究。

---

## 403. A note about exponential tractability of linear weighted tensor product problems in the worst-case setting

**arXiv ID:** 2603.21007 | [PDF](https://arxiv.org/pdf/2603.21007v1)

**作者:** Zirong Liu `[一作]` (Capital Normal University), Kai Wang `[通讯]` (Langfang Normal University)

**通讯引用:** 3606 | [OpenAlex ID](https://openalex.org/A5101447833)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了弱最坏情形下加权线性张量积问题的指数可解性（EXP-tractability）并给出了在 s<1 且 t≤1 的情形下 EXP-(s,t)-弱可解性与 EXP-统一弱可解性的必要与充分条件。

**💡 创新点**

填补了先前工作中关于 max(s,t)<1 的 EXP-(s,t)-WT 与 EXP-UWT 的空白，提供了完整的可解性判定标准，显著推进了加权线性张量积问题指数可解性理论。

**🔧 技术方法**

采用了谱分析、信息复杂度表达式、极限与不等式技巧，并构造了一系列辅助函数（如 j(ε)、d(ε)）来刻画奇异值与权重序列的增长行为。

**📊 数据集**

本研究为纯理论工作，没有使用具体的数据集。

**📈 对比分析**

由于研究内容为理论证明，未与实验方法或数值算法进行对比；所给出的条件直接决定了算法在最坏情形下的复杂度增长，说明在满足条件时可获得指数可解性。

**⚠️ 局限性**

局限性：仅适用于最坏情形下的线性加权张量积问题，且对权重序列与奇异值序列做了严格的单调性与正性假设；不涉及随机或平均误差情形，也未讨论实际数值实现。

---

## 404. Probability of super-regular matrices and MDS codes over finite fields

**arXiv ID:** 2603.20983 | [PDF](https://arxiv.org/pdf/2603.20983v1)

**作者:** Rathinakumar Appuswamy `[一作]`, Kenneth Zeger `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究随机有限域矩阵的超正则性与MDS码的极限概率，并给出阈值关系，推导3×3和4×4矩阵的计数结果，说明4×4非连续超正则计数非多项式，提出连通超正则4×4计数多项式猜想。

**💡 创新点**

首次将MDS码的随机概率与矩阵超正则性联系起来，确立了q、k、n三者的阈值关系；给出3×3矩阵计数的闭式公式并证明4×4计数非多项式；提出连通超正则4×4计数为多项式的猜想。

**🔧 技术方法**

使用随机矩阵理论、Chen–Stein泊松近似、组合计数与等价类归约；并辅以并行C++计算进行枚举。

**📊 数据集**

随机选择有限域𝔽_q中的矩阵，q取2–97之间的质数；针对3×3、4×4情况进行完整枚举。

**📈 对比分析**

通过阈值定理预测概率随q、k变化的趋势，并与实验得到的概率值（e^{-λ}曲线）进行比较，结果显示收敛速度快；对4×4计数用计算数据验证多项式猜想，发现与已知多项式结果高度吻合。

**⚠️ 局限性**

结果仅在极限条件（k,n,q→∞）下成立；λ∈(0,∞)的概率极限尚未完全证明；4×4计数仅基于实验，缺乏严格证明；大q下枚举计算量巨大，限制了可行的实验规模。

---

## 405. GraPHFormer: A Multimodal Graph Persistent Homology Transformer for the Analysis of Neuroscience Morphologies

**arXiv ID:** 2603.20970 | [PDF](https://arxiv.org/pdf/2603.20970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 406. From Causal Discovery to Dynamic Causal Inference in Neural Time Series

**arXiv ID:** 2603.20980 | [PDF](https://arxiv.org/pdf/2603.20980v1)

**作者:** Valentina Kuskova `[一作]`, Michael Coppedge `[通讯]` (University of Notre Dame)

**通讯引用:** 27892 | [OpenAlex ID](https://openalex.org/A5082717327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种两阶段神经网络因果模型DCNAR，先用神经可加自回归模型学习稀疏的因果网络，再将该网络作为结构先验，驱动时间变网络自回归，从而在结构未知的情况下实现动态因果推断；同时通过冲击响应和反事实分析实现科学解释；

**💡 创新点**

创新点在于将因果发现与动态因果估计分离，利用学习到的结构作为可检验的先验，兼顾可解释性与预测性能，并在无结构假设的场景下实现稳定、可解释的冲击响应与反事实；

**🔧 技术方法**

技术包括：神经可加自回归因果发现（NAVAR），稀疏正则化，结构优先的时间变网络自回归（tvNAR），冲击响应与反事实计算，基于核平滑的时间变参数估计；

**📊 数据集**

使用V‑Dem民主指标面板数据：139个国家、35年时间窗口（短板），以及扩展版89个国家、75年时间窗口（长板）；

**📈 对比分析**

与岭VAR、时间变VAR、带蒙特卡罗 dropout 的LSTM 进行比较，采用CRPS、预测区间覆盖率等预测分布指标；DCNAR在预测与分布指标上与岭VAR/TV‑VAR相近且优于LSTM；更重要的是其冲击响应和反事实路径稳定、解释性好，远优于其他模型；

**⚠️ 局限性**

局限性包括：依赖因果发现阶段的质量，短时间序列仍有限制；无法保证真实结构因果性，只是Granger因果；有限样本下动态估计仍具挑战；

---

## 407. A Solicit-Then-Suggest Model of Agentic Purchasing

**arXiv ID:** 2603.20972 | [PDF](https://arxiv.org/pdf/2603.20972v1)

**作者:** Shengyu Cao `[一作]`, Ming Hu `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并分析了一个“先索取-后建议”（solicit‑then‑suggest）模型，用以描述 AI 购物代理在多轮对话中学习顾客偏好并随后给出产品组合的过程。

**💡 创新点**

主要创新在于揭示了询问深度与产品组合宽度之间的经济互补性和替代性，并给出了在高维偏好空间中两者的收敛速率（O(1/m) 与 O(k^{‑2/d})）以及水填充最优询问策略与聚类（Voronoi）分配规则的解析解。

**🔧 技术方法**

使用了贝叶斯高斯先验、Kalman 滤波更新、A‑optimal 设计原理、量化（vector quantization）理论、以及信息论的“不确定性分解”与“可测度分解”等技术；在非高斯先验下进一步利用了 Bernstein‑von Mises 定理。

**📊 数据集**

本文并未在真实数据集上进行实验，而是基于理论推导和仿真验证（如 50,000 个从正态分布抽样的理想点），对比了不同询问次数与产品数量的期望匹配误差。

**📈 对比分析**

通过仿真表明，即使在维度为 100 的情形下，仅需 5 轮高质量询问就能与无询问但提供 32 个产品的情形获得相同的期望距离，说明询问的效果远优于扩大产品集，且在维度增大时差距进一步扩大。

**⚠️ 局限性**

局限性包括：未考虑询问形式（如比较、打分、预览）的噪声差异；仅聚焦匹配质量而未考虑代理佣金或价格机制；对真实对话交互与多模态反馈的实现细节缺乏讨论；实验仅基于模拟，缺乏真实用户数据验证。

---

## 408. LPNSR: Prior-Enhanced Diffusion Image Super-Resolution via LR-Guided Noise Prediction

**arXiv ID:** 2603.21045 | [PDF](https://arxiv.org/pdf/2603.21045v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 409. FLEX: Joint UL/DL and QoS-Aware Scheduling for Dynamic TDD in Industrial 5G and Beyond

**arXiv ID:** 2603.20971 | [PDF](https://arxiv.org/pdf/2603.20971v1)

**作者:** Leonard Kleinberger `[一作]`, Hans D. Schotten `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一种名为FLEX的QoS感知调度器，用于工业5G动态TDD网络的双向资源分配。

**💡 创新点**

其创新点在于在UL调度时对DL缓冲状态进行估计，联合调度UL/DL并通过优先级函数实现公平且满足QoS需求。

**🔧 技术方法**

使用了缓冲状态估计、时隙级动态分配、优先级函数（结合5QI和PF）以及基于ns‑3的5G‑LENA仿真框架。

**📊 数据集**

在三个工业场景（运动控制、移动机器人、异构QoS）下进行仿真，仿真参数来自工厂自动化和仓储物流的典型配置。

**📈 对比分析**

与PF、MR和QoS基线调度器相比，FLEX在包丢失率、吞吐量和延迟方面均优于基线，尤其在高用户密度和异向流量时表现突出。

**⚠️ 局限性**

局限在于每个UE只能有单一流量且在非确定性流量时可能出现约k2时隙的DL延迟；此外当前实现仅支持TDMA，未覆盖FDMA。

---

## 410. Dual Representation of Minimum Divergence Under Integral Constraints

**arXiv ID:** 2603.21027 | [PDF](https://arxiv.org/pdf/2603.21027v1)

**作者:** Shubhanshu Shekhar `[一作]` (University of Michigan), Shubhada Agrawal `[通讯]` (Indian Institute of Science)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种两阶段离散化-极大化方法，用于推导支持在[0,1]^K上的约束最小发散（f-发散）问题的对偶表示，并将其应用于序贯检验、估计和变化检测。

**💡 创新点**

创新点在于将经典凸双对偶与数据处理不等式结合，利用均值保持离散化通道构造精确的对偶式，并将该框架推广至任意f-发散和一般连续约束，使得对偶式显式且可计算。

**🔧 技术方法**

使用技术包括有限维凸双对偶、离散化通道（随机舍入）、数据处理不等式、弱下半连续性、Lipschitz/连续性条件以及极限交换论证。

**📊 数据集**

论文主要为理论研究，未使用特定实测数据集；所示统计应用均基于假设的[0,1]^K观测序列。

**📈 对比分析**

通过对偶式设计的层级测试、置信序列与变化检测方案，在α→0时达到信息理论下界，证明了近似最优性；与传统混合马丁格尔或经验似然方法相比，取得了相同或更优的渐近性能。

**⚠️ 局限性**

局限性包括仅在紧致域[0,1]^K上可行；对非紧致空间（如ℝ^K）的推广尚未完成；置信集的计算在高维下可能仍具有挑战性。

---

## 411. Left Behind: Cross-Lingual Transfer as a Bridge for Low-Resource Languages in Large Language Models

**arXiv ID:** 2603.21036 | [PDF](https://arxiv.org/pdf/2603.21036v1)

**作者:** Abdul-Salem Beibitkhan `[一作]` `[通讯]` (North American University), Abdul-Salem Beibitkhan (North American University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在哈萨克语和蒙古语这两种低资源语言上的表现，构建了50道手工设计的多语言基准，并评估了8种LLM在5种实验条件下产生的2000条回应。

**💡 创新点**

提出了系统比较英语提示、直接低资源语言提示和跨语言迁移（CLT）三种策略的框架，发现CLT仅在双语架构上有显著提升，并揭示多语言模型在某些语言上失效的细节，强调低资源语言服务不足的现实。

**🔧 技术方法**

采用LLM-as-judge半自动评分方法，对准确性、流畅度和完整度进行打分；在CLT条件下实现了三步翻译-推理-翻译流程，并通过流利度语言判别确保答案的语言正确性。

**📊 数据集**

使用了包含中性事实、推理、技术以及哈萨克/蒙古文化类别的50道问题集，每道问题均提供英语、哈萨克语和蒙古语的手工并行翻译，并附有参考答案列表。

**📈 对比分析**

将8个模型按架构分为英语首选、双语和多语言三类，在5种实验条件下计算平均得分；英语基线得分为90.7%，低资源直接条件约为75%（差距13.8–16.7个百分点），CLT对双语模型提升2.2–4.3个百分点，而对英语首选模型几乎无效。

**⚠️ 局限性**

实验规模有限（仅50道题），使用单人团队的半自动评测，未覆盖更多低资源语言，并且仅评估当前一代LLM，可能无法反映更大模型或持续学习方式的改进。

---

## 412. Query, Decompose, Compress: Structured Query Expansion for Efficient Multi-Hop Retrieval

**arXiv ID:** 2603.21024 | [PDF](https://arxiv.org/pdf/2603.21024v1)

**作者:** JungMin Yun `[一作]`, YoungBin Kim `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向多跳检索的结构化查询扩展框架DeCoR，避免生成噪声内容

**💡 创新点**

通过查询拆分与文档压缩两步，强调结构化信息优化而非大量生成

**🔧 技术方法**

使用指令微调的较小LLM（如LLaMA‑2‑7B）配合BM25检索与向量平均聚合

**📊 数据集**

在MultiHop‑RAG数据集上进行评估，涉及2556个多跳查询

**📈 对比分析**

与HyDE、Query2Doc等生成式扩展方法对比，DeCoR在Hits@10、MAP@10等指标上分别提升约3–6个百分点，且使用模型参数更少

**⚠️ 局限性**

仍受限于对初始检索召回质量的依赖，且多跳推理中候选文档数量会显著增加计算开销

---

## 413. SkillProbe: Security Auditing for Emerging Agent Skill Marketplaces via Multi-Agent Collaboration

**arXiv ID:** 2603.21019 | [PDF](https://arxiv.org/pdf/2603.21019v1)

**作者:** Zihan Guo `[一作]` (Sun Yat-sen University), Weinan Zhang `[通讯]` (Shanghai Innovation Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了多阶段、多智能体协作的安全审计框架 SkillProbe，用于在 Agent 技能市场前置审计。

**💡 创新点**

创新点在于“Skills-for-Skills”范式，将审计流程拆解为可复用的技能模块，并在同一管道内检测语义不一致与组合攻击。

**🔧 技术方法**

采用多智能体系统、LLM 推理、语义归一化、风险标签映射、组合风险仿真等技术。

**📊 数据集**

使用 ClawHub 公开的 2,500+ 实际技能、20 个热门技能以及 8 个主流 LLM（Claude, Gemini, GPT‑5, Nex 等）进行评测。

**📈 对比分析**

与传统代码静态扫描、后置运行时防御等基线对比，SkillProbe 能发现 90% 以上热门技能潜在风险，评估显示在准确率与可扩展性方面均优于单一工具。

**⚠️ 局限性**

局限性包括对高度混淆或完全黑盒代码的识别能力受限、零日组合攻击依赖预定义规则、以及在大规模批量审计时的计算开销。

---

## 414. The Intelligent Disobedience Game: Formulating Disobedience in Stackelberg Games and Markov Decision Processes

**arXiv ID:** 2603.20994 | [PDF](https://arxiv.org/pdf/2603.20994v1)

**作者:** Benedikt Hornig `[一作]` (Independent Researcher), Reuth Mirsky `[通讯]` (Tufts University)

**通讯引用:** 428 | [OpenAlex ID](https://openalex.org/A5036081979)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文通过将智能不服从建模为先后级的Stackelberg游戏——Intelligent Disobedience Game（IDG），并将其转换为多智能体MDP，推导了领导者与跟随者在有限时域下的最优策略。

**💡 创新点**

创新点在于：①提出具有信息不对称的先后级游戏框架，专门捕捉人类指令与自动系统的安全冲突；②引入“安全陷阱”概念，揭示多步游戏中跟随者可能诱导的无害循环；③通过归纳证明得到在任何可达目标的初始状态下，最优策略必然实现目标。

**🔧 技术方法**

采用的技术包括：游戏理论（Stackelberg、逆向归纳）、离散MDP/POMDP建模、强化学习可行性分析以及离散状态空间的递归证明。

**📊 数据集**

未使用公开数据集，主要以理论推导与简单仿真示例作为验证。

**📈 对比分析**

没有给出实验对比或数值性能指标，论文通过理论证明展示在最优策略下目标达成或陷入安全陷阱的情况；若需评估，可采用仿真平台测试不同策略的收敛与安全性。

**⚠️ 局限性**

局限性包括：假设环境静态且不考虑系统故障；安全陷阱可能在真实动态环境中导致不安全循环；缺乏对多智能体复杂交互与可解释性需求的进一步分析。

---

## 415. Long-Term Outlier Prediction Through Outlier Score Modeling

**arXiv ID:** 2603.20993 | [PDF](https://arxiv.org/pdf/2603.20993v1)

**作者:** Yuma Aoki `[一作]` (Nara Institute of Science and Technology), Takamitsu Sasaki `[通讯]` (Panasonic Holdings Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种两层无监督框架，先检测时序数据的异常点再预测未来时刻的异常分数，从而实现长期异常预测。

**💡 创新点**

创新点在于将异常检测与异常分数预测分离，能够在没有未来观测的情况下利用历史异常序列的时间结构实现对远期异常的预判。

**🔧 技术方法**

采用任意异常检测器（如重构误差或局部离群因子）和时序预测模型（实验中用单层 LSTM）对异常分数序列建模。

**📊 数据集**

实验使用合成周期性噪声序列以及北京温度与压力（PM2.5 数据）等真实时间序列，并人工注入周期性与跨维度相关的异常。

**📈 对比分析**

与基准相比，方法在周期性异常场景下 AUC 近 1.0，证明能完美预测异常时刻；当窗口不足以捕捉周期时性能急剧下降；在随机异常场景下性能与随机相当。

**⚠️ 局限性**

局限在于假设异常具有可学习的时间模式，对随机或无结构异常无法预测；仅在人工注入异常的数据上验证，缺乏对真实未知异常的进一步评估。

---

## 416. Geometrically Plausible Object Pose Refinement using Differentiable Simulation

**arXiv ID:** 2603.20992 | [PDF](https://arxiv.org/pdf/2603.20992v1)

**作者:** Anil Zeybek `[一作]` (Technical University of Munich), Akansel Cosgun `[通讯]` (Monash University)

**通讯引用:** 9126 | [OpenAlex ID](https://openalex.org/A5035882135)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种结合可微物理仿真、可微渲染与视触传感的多模态6D姿态细化框架，用于精确恢复手中物体的几何与物理可行姿态。

**💡 创新点**

创新点在于将可微物理梯度与可微渲染/深度/触觉梯度通过启发式选择组合，兼顾视觉一致性与物理可行性，显著降低姿态误差和交叉体积。

**🔧 技术方法**

主要技术包括可微物理仿真（Warp框架）、PyTorch3D可微渲染、基于点云的触觉感知与梯度融合优化。

**📊 数据集**

实验使用NVIDIA Isaac Sim仿真环境，采集9个YCB物体的RGB‑D、触觉与关节数据，构成20,000条训练/测试样本。

**📈 对比分析**

与传统ICP及其检查点版本对比，本文方法在位置误差和姿态误差上提升4–5%，更重要的是在无噪声/高噪声情况下将交叉体积误差分别降低73%和87%，并保持较低的接触面积偏差。

**⚠️ 局限性**

局限性包括仅在仿真中验证、计算开销较大（400步约60s），未处理真实世界的传感器噪声与标定误差，需进一步优化渲染与加速以实现实时闭环。

---

## 417. AutoMOOSE: An Agentic AI for Autonomous Phase-Field Simulation

**arXiv ID:** 2603.20986 | [PDF](https://arxiv.org/pdf/2603.20986v1)

**作者:** Sukriti Manna `[一作]` (University of Illinois Chicago), Subramanian K. R. S. Sankaranarayanan `[通讯]` (University of Illinois Chicago)

**通讯引用:** 10381 | [OpenAlex ID](https://openalex.org/A5063950942)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

AutoMOOSE框架通过五个智能代理实现从自然语言提示到完整多物理场相变场模拟的全流程自动化；

**💡 创新点**

创新点包括：将MOOSE与多智能体协同、自动化错误诊断与纠正、插件化物理模型接口、Model Context Protocol实现无头交互以及全流程自验证与可重复性；

**🔧 技术方法**

采用大型语言模型驱动的多智能体、FastAPI与插件化两函数接口、MCP协议以及MOOSE计算框架等技术；

**📊 数据集**

使用四温度铜多晶体晶粒生长基准实验数据作为验证集，包含多种参数设置；

**📈 对比分析**

通过与人工撰写的参考输入文件、手工分析对比，评估输入文件一致性、收敛性、运行时间加速1.8×、R² 0.90-0.95、Arrhenius能量回归误差≤28.7%，成功率100%；

**⚠️ 局限性**

局限在于缺乏检索增强生成、对新颖物理模块的预训练知识不足、有限的物理模块实现以及有限尺寸效应导致的活化能偏差。

---

## 418. Scaling laws in empirical networks

**arXiv ID:** 2603.20973 | [PDF](https://arxiv.org/pdf/2603.20973v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 419. Consistent but Dangerous: Per-Sample Safety Classification Reveals False Reliability in Medical Vision-Language Models

**arXiv ID:** 2603.20985 | [PDF](https://arxiv.org/pdf/2603.20985v1)

**作者:** Binesh Sadanandan `[一作]` (University of New Haven), Vahid Behzadan `[通讯]` (University of New Haven)

**通讯引用:** 923 | [OpenAlex ID](https://openalex.org/A5062734917)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出四象限安全分类法，评估医学视觉语言模型在一致性与图像依赖上的表现，并在MIMIC‑CXR与PadChest两套胸片数据集上对五种模型进行系统实验，揭示一致性与安全性之间的悖论。

**💡 创新点**

创新点在于：①将一致性与图像依赖统一到四象限分类中；②发现低翻转率模型往往落入危险象限；③提出文本仅基线作为部署检查，避免安全陷阱；④通过实验证明一致性训练易导致文本捷径。

**🔧 技术方法**

利用语义等价多表述的翻转率评估，LoRA微调提升一致性，文本仅推理对比判定图像依赖，并计算熵、KL等指标进行判定。

**📊 数据集**

使用的胸片问答数据集为：MIMIC‑CXR（98个平衡样本）和PadChest（861个不平衡样本）。

**📈 对比分析**

比较方法为计算翻转率、危险比例、四象限准确率与熵；结果显示翻转率最低的模型危险比例最高（如LLaVA‑Rad Base在PadChest的98.5%危险），表明一致性评估可能误导安全性判断。

**⚠️ 局限性**

局限性包括仅评估二分类问答任务、样本量有限、文本仅基线为二元判定，无法捕捉置信度变化，且对其他模型或更复杂任务的适用性未知。

---

## 420. Cyber Deception for Mission Surveillance via Hypergame-Theoretic Deep Reinforcement Learning

**arXiv ID:** 2603.20981 | [PDF](https://arxiv.org/pdf/2603.20981v1)

**作者:** Zelin Wan `[一作]`, Munindar P. Singh `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在无人机侦察任务中，利用蜜罐无人机对抗拒绝服务攻击，提出了将超游戏理论与深度强化学习相结合的 HT‑DRL 框架，使用超游戏期望效用（HEU）指导策略学习，并在 DRL 的 logits 上加上过滤层以改进早期探索。

**💡 创新点**

创新点包括：① 构建动态超游戏模型，让攻击者与防御者保持不一致的感知；② 推导两方的 HEU 并将防御者的 HEU 转化为过滤层，解决 DRL 的冷启动与非平稳性问题；③ 在蜜罐无人机部署与能耗建模上进行扩展，提高系统实用性。

**🔧 技术方法**

技术方法：超游戏建模与分析、HEU 计算、A2C 深度强化学习、基于 HEU 的 logits 过滤层、仿真模拟。

**📊 数据集**

使用自行搭建的无人机网络仿真环境（含任务进度、连通性、信号强度等状态），未使用公开数据集。

**📈 对比分析**

通过与固定蜜罐、基于超游戏、基于 DRL、IDS、Container‑Drone 等多种基线方法在 100 次独立跑测中比较，HT‑DRL 在任务完成率（RMC）上显著提升，且能耗（EC）与基线相当或更低。

**⚠️ 局限性**

局限性包括：① 仍缺乏严格的理论证明（如子博弈完美均衡、收敛性上界）；② 参数灵敏度分析有限；③ 仅基于仿真验证，缺乏真实环境实验；④ 对超游戏参数和噪声假设的鲁棒性尚待进一步探索。

---

## 421. DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles

**arXiv ID:** 2603.20975 | [PDF](https://arxiv.org/pdf/2603.20975v1)

**作者:** Bo Jiang `[一作]` (Temple University), Bo Jiang `[通讯]` (Temple University)

**通讯引用:** 932 | [OpenAlex ID](https://openalex.org/A5045465754)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种通过分析多智能体LLM在同一问题上产生的答案不一致的结构来进行不确定性量化的框架。

**💡 创新点**

创新点在于将语言学结构特征（证据重叠、论点强度、分歧深度等）与嵌入几何特征（聚类距离、分散度等）结合，并用简单的逻辑回归模型得到高质量的置信度估计。

**🔧 技术方法**

技术包括：用LLM进行结构分析、BGE-large对推理文本进行句向量编码、逻辑回归和MLP分类器，以及对比多种基线方法。

**📊 数据集**

实验使用四个闭合式问答基准：StrategyQA、MMLU、TruthfulQA 与 ARC‑Challenge。

**📈 对比分析**

与投票计数、熵、语义不确定性等基线相比，-LLM在四个基准上的平均AUROC为0.802、ECE为0.036，显著优于最强基线LLM Aggregator（AUROC 0.791、ECE 0.098），且在“弱分歧”层级效果尤为突出。

**⚠️ 局限性**

局限性包括仅评估闭合式问答任务、主要使用单一模型（Qwen3.5-27B）、对LLM自省质量的依赖、需要标注数据训练分类器以及对智能体数量扩展的探索不足。

---

## 422. A Framework for Low-Latency, LLM-driven Multimodal Interaction on the Pepper Robot

**arXiv ID:** 2603.21013 | [PDF](https://arxiv.org/pdf/2603.21013v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 423. Benchmarking Scientific Machine Learning Models for Air Quality Data

**arXiv ID:** 2603.21039 | [PDF](https://arxiv.org/pdf/2603.21039v1)

**作者:** Khawja Imran Masud `[一作]` (University of North Texas), Sahara Ali `[通讯]` (University of North Texas)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5042875954)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在美国德克萨斯州达拉斯县收集 EPA 2022‑2024 年 PM2.5 与 O3 的日 AQI 数据，构建多时延（LAG 1、7、14、30 天）时间序列，并对传统统计模型（LR、SARIMAX）、机器学习模型（MLP）、深度学习模型（LSTM）以及物理约束的 MLP+Physics、LSTM+Physics 进行统一评估，提供了一个多时延 AQI 预测的完整基准。

**💡 创新点**

① 设计了统一的 lag‑wise 数据集和评估流程；② 将 EPA AQI 分段式计算公式作为物理一致性约束融入损失函数，形成轻量级物理引导的 MLP+Physics 与 LSTM+Physics；③ 对不同污染物、不同预测时延的物理引导效果进行系统比较，揭示物理约束在不同情境下的收益。

**🔧 技术方法**

使用线性回归、SARIMAX、MLP、LSTM 进行标准训练；物理引导模型通过在损失中加入 ≤ 物理一致性损失 — 通过 EPA breakpoint 表计算的 AQI 作为参考；采用标准化/最小-最大缩放、顺序 train‑test 切分；评价指标为 MAE、RMSE、NMSE。

**📊 数据集**

EPA（美国环境保护署）公开的达拉斯县日 AQI 监测数据，涵盖 2022–2024 年的 PM2.5 与 O3 两个污染物，共计约 1,094 与 1,091 条观测记录。

**📈 对比分析**

在相同的 lag‑wise 数据集上进行模型比较，使用 80%–20% 顺序划分；结果显示深度学习模型（尤其 LSTM）显著优于传统基线，物理约束进一步提升了 PM2.5 在短期预测（LAG 1）以及整体性能，并在 LSTM+Physics 下实现了更稳定、物理可解释的预测；对 O3 的改进主要体现在预测稳定性而非误差下降。

**⚠️ 局限性**

① 物理约束仅基于 EPA 的分段 AQI 公式，无法充分捕捉 O3 的光化学反应与天气耦合；② 仅使用污染物浓度和 AQI 作为输入，未引入气象变量；③ 研究仅聚焦达拉斯县，结果的空间推广性需要进一步验证。

---

## 424. Deep Attention-based Sequential Ensemble Learning for BLE-Based Indoor Localization in Care Facilities

**arXiv ID:** 2603.21030 | [PDF](https://arxiv.org/pdf/2603.21030v1)

**作者:** Minh Triet Pham `[一作]` (University of Melbourne), Le Nhat Tan `[通讯]` (Ho Chi Minh City University of Technology)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5068010521)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出了 Deep Attention-based Sequential Ensemble Learning（DASEL）框架，用来改进 BLE 低功耗无线电室内定位，将定位任务重新建模为序列学习问题，并通过频率特征、双向 GRU 与注意力机制、双层集成与置信加权平滑来提升性能。

**💡 创新点**

创新点包括：① 采用频率特征替代传统 RSSI，提升对多路径干扰和设备差异的鲁棒性；② 把整段房间停留视为一条序列，利用双向 GRU + 注意力捕捉轨迹与上下文信息；③ 双层集成策略——多随机种子模型与多方向滑动窗口——解决序列边界未知与模型方差问题；④ 置信加权时间平滑进一步消除孤立错误，保持空间连贯性。

**🔧 技术方法**

使用的技术包括：频率特征工程、双向 GRU 网络与注意力机制、七个多方向滑动窗口、五个随机种子模型集成、置信加权投票与 5 秒时间平滑、宏 F1 评估以及 4 折时间分割交叉验证。

**📊 数据集**

数据集来源于 Kyushu Institute of Technology 的护理设施 5 楼 BLE 数据，覆盖四天（4 月 10-13 日），包含约 110 万条带时间戳的 RSSI 记录，标签为 13-18 个不同房间，体现真实环境中的严重类别不平衡。

**📈 对比分析**

与传统基线（XGBoost + 统计特征）及两种改进版本相比，DASEL 在 4 折交叉验证中平均宏 F1 达到 0.4438，最高 0.5114，传统最高仅 0.2898，提升 53.1%。标准差仅为 0.0295，表现稳健且在所有折中均优于传统方法。

**⚠️ 局限性**

限制包括：① 推理阶段需要多模型、多方向窗口的多次前向传播，计算量和延迟较高，不易在极低功耗移动设备上实时部署；② 当训练样本稀缺（如折 4）时仍会出现性能下降；③ 目前仍未充分利用设施布局信息，可能导致物理上不可能的瞬移预测。

---

## 425. Can we automatize scientific discovery in the cognitive sciences?

**arXiv ID:** 2603.20988 | [PDF](https://arxiv.org/pdf/2603.20988v1)

**作者:** Akshay K. Jagadish `[一作]` (Princeton University), Eric Schulz `[通讯]` (Helmholtz Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一套端到端的自动化认知科学发现循环，涵盖实验设计、数据生成、模型合成与闭环优化。

**💡 创新点**

将大语言模型与生成式语法、基础认知模型、程序合成技术和“兴趣度”评估器融合，实现在硅上高通量搜索认知实验与机制。

**🔧 技术方法**

主要技术包括：LLM作为实验与模型生成器、Centaur等认知基础模型用于合成行为数据、GeCCo/演化搜索等程序合成框架、主动学习/信息增益用于实验选择、批评家模型（LLM评判器）驱动的闭环。

**📊 数据集**

使用的并非传统公开数据集，而是基于预训练认知基础模型（如Centaur）生成的合成行为数据，且可按人口统计与心理测量进行条件化。

**📈 对比分析**

文中未给出具体实验对比与性能指标；通过理论讨论指出该框架能显著提升发现速率、扩展实验空间，并为个体差异与神经预测提供新的途径。

**⚠️ 局限性**

主要局限包括：实验语法表达力受限，合成数据可能缺乏真实性，程序搜索空间崎岖且易受偏置影响，批评家兴趣度评估易被“玩转”，以及系统整体易产生大量无用或误导性成果。

---

## 426. Dreaming the Unseen: World Model-regularized Diffusion Policy for Out-of-Distribution Robustness

**arXiv ID:** 2603.21017 | [PDF](https://arxiv.org/pdf/2603.21017v1)

**作者:** Ziou Hu `[一作]` (Technical University of Munich), Alois Knoll `[通讯]` (Technical University of Munich)

**通讯引用:** 25091 | [OpenAlex ID](https://openalex.org/A5063781430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 Dream Diffusion Policy，结合扩散策略与世界模型，实现对视觉失真与 OOD 干扰的鲁棒控制。

**💡 创新点**

创新点在于将世界模型与扩散策略共享 3D 编码器进行共训练，并通过实时真实‑想象差异检测 OOD，自动切换至内部想象循环完成任务并恢复。

**🔧 技术方法**

使用 3D 视觉编码器（PointNet+MLP）、条件 1D U‑Net、扩散模型、FiLM 注入、实时 6D 姿态估计与递归想象推理。

**📊 数据集**

实验数据集包括 MetaWorld、Adroit 以及真实 Franka Panda 机器人在 Press Button、Pour Tea、Stack Blocks 等任务。

**📈 对比分析**

与 DP3、FlowPolicy 等基线及其追踪增强版本对比，DDP 在 MetaWorld OOD 上平均成功率 73.8%、真实世界空间偏移下 83.3%，完全想象模式下 76.7%，显著优于基线。

**⚠️ 局限性**

局限性包括对外部追踪模块的高度依赖、只能在 ID 环境中起始、以及在多次 OOD 事件或低层动作失效时难以持续恢复。

---

## 427. Detection of adversarial intent in Human-AI teams using LLMs

**arXiv ID:** 2603.20976 | [PDF](https://arxiv.org/pdf/2603.20976v1)

**作者:** Abed K. Musaffar `[一作]` (University of California at Santa Barbara), Francesco Bullo `[通讯]` (University of California at Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了如何使用LLM作为监督者，在无任务信息的情况下通过行为轨迹检测人类‑AI团队中的敌意行为。

**💡 创新点**

提出了全任务无关的行为监督方法，仅凭交互日志即可识别敌意行为，并证明了LLM在短窗口下的可行性。

**🔧 技术方法**

使用大型语言模型（GPT‑4.1、GPT‑4.1‑mini、GPT‑4o、GPT‑3）结合窗口化观察、微调与阈值异常检测。

**📊 数据集**

使用了一套包含三名人类与一名恶意AI在25轮问答游戏中的交互数据集，记录行为轨迹。

**📈 对比分析**

与随机猜测基线比较，10轮窗口下准确率可达81%（GPT‑4.1），5轮窗口需微调后可达82%；1轮实时检测通过微调后召回率为100%，误报率0.178，显示实时检测可行。

**⚠️ 局限性**

缺乏对抗性鲁棒性保证，LLM易出现善意偏差，在极短窗口或未微调时召回率为0，且对任务迁移性需进一步验证。

---

## 428. DMMRL: Disentangled Multi-Modal Representation Learning via Variational Autoencoders for Molecular Property Prediction

**arXiv ID:** 2603.21108 | [PDF](https://arxiv.org/pdf/2603.21108v1)

**作者:** Long Xu `[一作]` (Nanning Normal University), Yuzhong Peng `[通讯]` (Zhejiang Wanli University)

**通讯引用:** 724 | [OpenAlex ID](https://openalex.org/A5058191381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于变分自编码器的分解式多模态表征学习框架DMMRL，用于分子属性预测。

**💡 创新点**

创新点在于：①利用VAE将每种模态的特征分离为共享（结构相关）与私有（模态特有）潜在空间；②通过正交性与对齐约束确保共享与私有空间相互独立并保持跨模态一致性；③引入门控注意力融合机制，实现自适应的跨模态信息整合。

**🔧 技术方法**

采用了变分自编码器（VAE）、图神经网络（CMPNN）、序列编码器（Bi-LSTM+Transformer）、几何GNN、门控注意力融合、对齐与正交约束、以及标准的全连接预测头。

**📊 数据集**

在MoleculeNet七大基准数据集上进行评估，包含分类任务（BACE、BBBP、ClinTox、Tox21）和回归任务（ESOL、FreeSolv、Lipophilicity）。

**📈 对比分析**

与12种基线（序列、图、几何及多模态方法）对比，DMMRL在五个数据集上均取得最高或次高的ROC‑AUC /最低RMSE，平均提升约1.5%–2.9%，且方差更低，表明模型更稳健、更具可解释性。

**⚠️ 局限性**

主要局限：①未引入领域知识或先验约束，可能导致分解效果受限；②仅在随机拆分下验证，未评估在分子架构分裂（scaffold split）等更严苛泛化场景下的表现；③多模态融合仍为线性组合，未捕捉更高阶非线性交互；④训练成本较高，计算资源消耗大。

---

## 429. DGRNet: Disagreement-Guided Refinement for Uncertainty-Aware Brain Tumor Segmentation

**arXiv ID:** 2603.21086 | [PDF](https://arxiv.org/pdf/2603.21086v1)

**作者:** Bahram Mohammadi `[一作]` (Macquarie University), Yuankai Qi `[通讯]` (Macquarie University)

**通讯引用:** 4557 | [OpenAlex ID](https://openalex.org/A5070842891)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种利用多视角预测不一致性来引导分割精细化的模型DGRNet；

**💡 创新点**

创新点在于：①通过轻量化的视角适配器实现单模型多样化预测，②将预测分歧转化为不确定性地图并驱动局部自适应修正，③将临床文本描述作为条件输入来补偿图像模糊区域；

**🔧 技术方法**

使用Swin Transformer编码器、共享解码器、FiLM调制、三种不确定性度量融合、空间注意力、文本FiLM调制、梯度隔离及多目标损失；

**📊 数据集**

在TextBraTS（基于BraTS 2020）数据集上进行实验；

**📈 对比分析**

与多种SOTA方法（3D-UNet、nnU-Net、SegResNet、Swin UNETR、Nestedformer、TextBraTS基线）对比，DGRNet在Dice上提升2.4%（约87.6%），在HD95上降低11%（约4.57mm），且不确定性评估指标表现优秀；

**⚠️ 局限性**

局限性包括：需要四个MRI模态，视角数目和多样性权重对性能敏感；文本条件依赖于手工标注的放射学报告，若报告质量不佳则影响效果；

---

## 430. AnyPro: Preference-Preserving Anycast Optimization based on Strategic AS-Path Prepending

**arXiv ID:** 2603.21082 | [PDF](https://arxiv.org/pdf/2603.21082v1)

**作者:** Minyuan Zhou `[一作]` (Nanjing University), Wan Du `[通讯]` (University of California, Merced)

**通讯引用:** 2826 | [OpenAlex ID](https://openalex.org/A5042917596)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了名为AnyPro的系统，利用AS路径预加技术（ASPP）对全球Anycast网络进行客户端-入口映射的精准优化，自动识别ASPP敏感客户端，生成约束关系并求解全局最优的预加配置；

**💡 创新点**

创新点在于：①系统首次实现对ASPP敏感客户端的高效检测（max-min polling）；②通过约束求解与二分搜索解决冲突，实现全局最优配置；③能够与任何cast站点启用优化（AnyOpt）结合，进一步提升性能；

**🔧 技术方法**

采用的技术包括：AS路径预加、主动ICMP探测测量、max-min polling、约束式优化（OR‑Tools）、二分搜索冲突修正、以及与AnyOpt的协同优化；

**📊 数据集**

数据集包括：2.4M IPv4地址（ISI hitlist）用于探测，20个全球分布的PoP生产环境（共38个入口）用于真实实验；

**📈 对比分析**

与基线（All‑0）、AnyOpt、AnyPro预处理等进行对比，AnyPro（finalized）将90th百分位延迟从271.2 ms降低至58.0 ms，减少约37.7 %，与AnyOpt联合可进一步提升；

**⚠️ 局限性**

局限性包括：冲突约束可能导致小规模客户端被牺牲；对ISP动态ASPP变化的鲁棒性虽高但仍有限；只针对ASPP，未考虑BGP社区等其他路由控制手段；

---

## 431. LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning

**arXiv ID:** 2603.21065 | [PDF](https://arxiv.org/pdf/2603.21065v1)

**作者:** Jianing Wang `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们提出了一种560B参数的Mixture-of-Experts模型，专门用于Lean4的本地形式推理，包括自动形式化、草图绘制和证明三大能力，并通过工具集成实现了TIR（Agentic Tool‑Integrated Reasoning）。

**💡 创新点**

创新点在于：①构建了Hybrid‑Experts Iteration Framework，利用多专家模型与验证工具循环生成高质量任务轨迹；②提出Hierarchical Importance Sampling Policy Optimization（HisPO）算法，采用分层梯度屏蔽解决MoE模型在长序列RL中的训练不稳定；③引入法律检测与奖励真诚机制，消除reward hacking。

**🔧 技术方法**

技术手段包括Mixture‑of‑Experts架构、Agentic RL（DPO+TIR）、工具集成验证（Lean4 Server、AST检查）、Hierarchical Importance Sampling、梯度遮蔽与三重截断策略。

**📊 数据集**

使用了多种形式化与非形式化数据集：CombiBench、FormalMath‑Lite、MathOlympiadBench、MiniF2F‑Test、ProofNet、ProveBench、PutnamBench 以及AIME‑25、HMMT‑25、IMO‑AnswerBench、AMO‑Bench、GPQA‑Diamond、LiveCodeBench、OJBench等。

**📈 对比分析**

与开源与闭源基线对比，模型在自动形式化和定理证明上均突破SOTA：MiniF2F‑Test Pass@72达97.1%，ProveBench 70.8%，PutnamBench 41.5%；在全证和草图证明模式下，Pass@32分别高于所有开源对手，样本效率显著提升。

**⚠️ 局限性**

局限性包括在非形式化推理任务上略有下降，仍落后于部分闭源模型，且在训练初期易出现reward hacking，需依赖复杂的AST检查机制来抑制。

---

## 432. Frequency Switching Mechanism for Parameter-E!cient Multi-Task Learning

**arXiv ID:** 2603.21111 | [PDF](https://arxiv.org/pdf/2603.21111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. Two Experts Are Better Than One Generalist: Decoupling Geometry and Appearance for Feed-Forward 3D Gaussian Splatting

**arXiv ID:** 2603.21064 | [PDF](https://arxiv.org/pdf/2603.21064v1)

**作者:** Hwasik Jeong `[一作]` (Yonsei University), Eunbyung Park `[通讯]` (Yonsei University)

**通讯引用:** 1691 | [OpenAlex ID](https://openalex.org/A5013897558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 2Xplat 两专家框架，实现无姿势（pose‑free）快速 3D 高斯展平（Gaussian Splatting）

**💡 创新点**

创新点在于将几何估计与外观合成拆分为两台独立专家：先用几何专家预测相机姿势，再将姿势与图像交由专门的 3DGS 专家生成 Gaussian 表示，打破传统单一网络耦合的局限

**🔧 技术方法**

采用预训练的 Depth Anything 3（DA3）作为几何专家，使用 Multi‑view Pyramid Transformer（MVP）作为 3DGS 专家，并在两者之间实现端到端微调；采用相对姿势损失、可区分渲染损失以及感知损失进行训练

**📊 数据集**

在 RealEstate10K、DL3DV、以及 ScanNet++ 三大数据集上进行训练与评估

**📈 对比分析**

与 pose‑dependent 方法（如 MVSplat、DepthSplat、MVP 等）以及 pose‑free 方法（如 NoPoSplat、AnySplat、YoNoSplat 等）对比，2Xplat 在低分辨率与高分辨率设置下均显著优于 pose‑free 方案，且在 PSNR、SSIM、LPIPS 指标上与 pose‑dependent 方案持平，且仅需 5K 次迭代即可收敛

**⚠️ 局限性**

局限性：目前两专家之间的低层特征共享较少，可能存在冗余计算；对几何专家的预训练依赖较大，若预训练数据分布与目标任务不匹配，性能可能受限

---

## 434. LiFR-Seg: Anytime High-Frame-Rate Segmentation via Event-Guided Propagation

**arXiv ID:** 2603.21115 | [PDF](https://arxiv.org/pdf/2603.21115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 435. SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM

**arXiv ID:** 2603.21055 | [PDF](https://arxiv.org/pdf/2603.21055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 436. Cortical Policy: A Dual-Stream View Transformer for Robotic Manipulation

**arXiv ID:** 2603.21051 | [PDF](https://arxiv.org/pdf/2603.21051v1)

**作者:** Xuening Zhang `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29281 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Cortical Policy双流视图变换器，用于机器人抓取与操控

**💡 创新点**

融合静态视图3D一致性学习与动态视图位置感知预训练，实现空间推理与动态适应

**🔧 技术方法**

使用双流Transformer、VGGT 3D先验、GLC人类眩动预训练模型以及SmoothAP与循环几何一致性损失

**📊 数据集**

在RLBench、COLOSSEUM以及真实Frankia Panda机器人实验中验证

**📈 对比分析**

在RLBench平均成功率约69%超过SOTA（RVT-2 65.5%），COLOSSEUM平均成功率约70%领先对手，真实任务成功率提升30%–80%

**⚠️ 局限性**

对零样本新任务的泛化仍有限（约24%成功率），需要进一步提升组合抽象与多目标跟踪能力

---

## 437. Anatomical Prior-Driven Framework for Autonomous Robotic Cardiac Ultrasound Standard View Acquisition

**arXiv ID:** 2603.21134 | [PDF](https://arxiv.org/pdf/2603.21134v1)

**作者:** Zhiyan Cao `[一作]` (Huazhong University of Science and Technology), Han Ding `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 13492 | [OpenAlex ID](https://openalex.org/A5057513904)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一个基于解剖学先验驱动的闭环框架，用于心脏超声标准视图（以A4C为例）的自动获取，集成了多类结构分割和机器人探头自适应调整。

**💡 创新点**

创新点包括：① 将空间关系图（SRG）模块嵌入YOLO分割模型，实现解剖一致性约束；② 将解剖先验转化为高斯概率分布，作为强化学习状态和奖励的可解释基准；③ 通过RL实现零样本探头调整，将分割与控制通过先验统一连接。

**🔧 技术方法**

采用的技术包括：SRG‑augmented YOLOv11s（多类分割）+图神经网络全局编码与局部关系评分；高斯先验构建；基于MDP的强化学习（DQN）探头控制；机器人系统实现闭环反馈。

**📊 数据集**

使用数据集：MM‑WHS 2017 3D心脏分割数据（12个高质量样本）用于先验拟合；Special Case 145张A4C挑战图像用于分割评估；15组心脏模型/phantom实验用于RL验证。

**📈 对比分析**

与YOLOv5s/8s/11s、FastSAM、H‑SAM、U‑Mamba、DAM‑Seg等基线在Special Case数据集比较；SRG‑YOLOv11s在mAP50提升11.3%、mIoU提升6.8%、mDice提升4.7%；RL在仿真中的成功率92.5%，在真实phantom实验中总体成功率86.7%（mild 100%，moderate 85.7%，severe 83.3%）。

**⚠️ 局限性**

局限性：未考虑皮肤软组织顺应性和心脏搏动引起的时间变形；仅实现单自由度探头控制；对不同解剖变异的泛化验证仍不足。

---

## 438. VisFly-Lab: Unified Differentiable Framework for First-Order Reinforcement Learning of Quadrotor Control

**arXiv ID:** 2603.21123 | [PDF](https://arxiv.org/pdf/2603.21123v1)

**作者:** Fanxing Li `[一作]`, Danping Zou `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

构建了一个统一的可微分框架，支持悬停、跟踪、降落和竞速四类四旋翼任务，并提出了Amended Backpropagation Through Time (ABPT) 方法以解决第一阶强化学习中的状态覆盖不足和梯度偏置问题。

**💡 创新点**

创新点在于：①将多任务四旋翼控制整合到一个可插拔、可扩展的可微分模拟环境中；②设计了 ABPT 通过 0 步回报与 N 步回报的混合、访问状态初始化以及差分优化来显著提升训练鲁棒性和收敛速度；③提供了与 PPO、BPTT、SHAC 的对比实验，验证了 ABPT 在部分不可微奖励任务中的优势。

**🔧 技术方法**

采用了可微分仿真技术（VisFly 与 JAX）、基于 Actor-Critic 的梯度强化学习（BPTT、SHAC、PPO）、重参数化技巧、TD(λ) 价值回报估计、目标 critic 以及访问状态回放策略。

**📊 数据集**

主要使用 VisFly 仿真器中四旋翼的高保真可微分动力学模型进行四种任务的仿真实验；此外，文中还演示了在真实四旋翼平台上的概念验证部署。

**📈 对比分析**

通过与 PPO、BPTT、SHAC 三个基线在悬停、跟踪、降落和竞速四个任务中的奖励曲线、收敛速度和最终奖励进行对比，ABPT 在大多数任务中获得最高奖励，尤其在具有二值奖励的降落和竞速任务中收敛更快、性能更稳健；在全可微任务中收敛速度与 PPO 相当。

**⚠️ 局限性**

局限性包括：①仅提供了概念验证级别的真实部署实验，未做系统化的量化评估；②对仿真与实测之间的差距未进行深入的 sim‑to‑real 分析；③ABPT 的超参数和奖励设计对性能有一定敏感性，需在不同硬件与环境中进一步验证。

---

## 439. ERM-MinMaxGAP: Benchmarking and Mitigating Gender Bias in Multilingual Multimodal Speech-LLM Emotion Recognition

**arXiv ID:** 2603.21050 | [PDF](https://arxiv.org/pdf/2603.21050v1)

**作者:** Zi Haur Pang `[一作]` (Kyoto University), Nancy F. Chen `[通讯]` (Agency for Science, Technology, and Research (A*STAR))

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了多语种多模态语音情感识别（SER）的性别偏差基准，并提出了兼顾性能与公平性的 ERM-MinMaxGAP 训练目标。

**💡 创新点**

创新点在于引入最大语言内性别损失差距的 MinMaxGAP 正则以及自适应公平权重更新机制，能在提升 SER 性能的同时显著降低性别差距。

**🔧 技术方法**

采用 Qwen2-Audio 语音大模型、LoRA 微调、交叉熵损失、MinMaxGAP 正则化和 Lagrange 多项式权重自适应调整等技术。

**📊 数据集**

使用了 MELD-ST 数据集，该数据集覆盖英语、日语、德语三种语言，包含 7 类情感标签和人工标注的性别信息。

**📈 对比分析**

与多种零样本语音 LLM 在单模态和多模态条件下进行对比，ERM-MinMaxGAP 在多语种中实现了约 5–10% 的 W-F1/ACC 提升，同时平均性别差距降低 0.1–1.4%。

**⚠️ 局限性**

局限性包括：未在更广泛的语言和大规模真实场景下验证，且在极端数据不平衡时公平权重的选择仍需进一步改进。

---

## 440. CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation

**arXiv ID:** 2603.21104 | [PDF](https://arxiv.org/pdf/2603.21104v1)

**作者:** Bowen Jing `[一作]` (Tuojing Intelligence), Haibao Yu `[通讯]` (Tuojing Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种在BEV生成世界模型中引入结构化因果反事实推理的框架（CounterScene），通过识别导致安全的因果关键Agent并在扩散过程中对其轨迹进行最小干预，从而生成安全关键且真实感高的交互场景。

**💡 创新点**

创新点包括：① 先行的因果对抗Agent识别，能够准确定位安全的决定性参与者；② 基于冲突的因果交互图（CIG）显式建模多Agent间的因果依赖；③ 随阶段自适应的反事实扩散引导，将空间与时间约束分解，逐步压缩安全间距，只改变关键Agent轨迹而保持整体动态一致。

**🔧 技术方法**

技术手段：扩散式BEV世界模型+SceneTransformer去噪器；冲突感知边特征与多层感知网络；因果交互图与注意力机制；因果Agent识别模块；空间/时间反事实引导、适应性压缩、正则化与进度调度。

**📊 数据集**

训练与评估数据集：nuScenes（1,000场景）进行训练与交叉验证；零样本迁移至nuPlan（4个城市共100场景）验证泛化。

**📈 对比分析**

与STRIVE、VAE、CTG、CTG++、CCDiff等基线对比：在1–10秒预测范围内，CounterScene实现最低ADE/FDE/ORR，并在短期/中期/长期均取得最高碰撞率CR和硬刹车率HBR，尤其在10秒时CR提升至22.7%/40.2%，表现出显著的现实感与对抗效果兼备。

**⚠️ 局限性**

局限性：仅处理二维BEV视角，难以覆盖三维高度交互；依赖准确的冲突识别与因果图构建，对复杂多方协调场景可能不足；目前仅对单一Agent进行干预，无法一次性构造多Agent协同风险；对极端交通文化差异的迁移仍需进一步验证。

---

## 441. Evaluating Reasoning-Based Scaffolds for Human-AI Co-Annotation: The ReasonAlign Annotation Protocol

**arXiv ID:** 2603.21094 | [PDF](https://arxiv.org/pdf/2603.21094v1)

**作者:** Smitha Muthya Sudheendra `[一作]` (University of Minnesota), Jaideep Srivastava `[通讯]` (University of Minnesota)

**通讯引用:** 18099 | [OpenAlex ID](https://openalex.org/A5002187701)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ReasonAlign，一个在两轮注释流程中仅暴露 LLM 生成的推理解释、隐藏预测标签的注释框架，旨在通过结构化推理提高人类标注一致性。

**💡 创新点**

创新点在于将推理解释与注释过程分离，形成一种“解释式支架”而非直接建议，能够在保持标注者自主的同时提升一致性，并引入 Annotator Effort Proxy (AEP) 指标评估推理影响。

**🔧 技术方法**

使用自示例提示与链式思维（chain‑of‑thought）生成的 LLM 推理，配合两轮 Delphi 风格的注释流程。

**📊 数据集**

实验使用约 500 条对话语句的情感分析和观点检测数据集，四名注释者进行标注。

**📈 对比分析**

通过对比两轮标注的 Cohen's κ，情感任务从 0.76 提升到 0.98，观点检测从 0.73 提升到 0.85，AEP 低于 1%；表明推理解释显著提升一致性且仅导致极少的标注修订。

**⚠️ 局限性**

局限性包括样本量和注释者数量有限、未直接评估推理准确性、可能存在模型共性偏差以及未验证对真实标注质量的提升。

---

## 442. Hierarchical Text-Guided Brain Tumor Segmentation via Sub-Region-Aware Prompts

**arXiv ID:** 2603.21083 | [PDF](https://arxiv.org/pdf/2603.21083v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 443. DRL-driven Online Optimization for Joint Traffic Reshaping and Channel Reconfiguration in RIS-assisted Semantic NOMA Communications

**arXiv ID:** 2603.21093 | [PDF](https://arxiv.org/pdf/2603.21093v1)

**作者:** Songhan Zhao `[一作]` (Sun Yat-sen University), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 21442 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文研究了在RIS辅助下的语义感知NOMA通信系统，提出联合交通重塑（语义提取）与通道重配置，并通过可延迟的语义提取和基于DRL的轻量化在线优化（PDOO）实现长期能效最大化。

**💡 创新点**

创新点包括：①将RIS相位控制、语义提取深度和NOMA解码顺序三者联合优化，形成JTAC方案；②提出可延迟的语义提取策略，分散多时隙提取任务提升长期能效；③设计PDOO框架，利用PPO动态选择最合适的子优化模块，显著降低在线求解复杂度。

**🔧 技术方法**

主要技术：RIS被动波束成形、语义提取深度调节、NOMA功率域多址、深度强化学习（PPO）、惩罚法与SROCR松弛、凸优化与启发式近似。

**📊 数据集**

数据集：本文采用仿真场景（K=3个用户，RIS元素数L可变，Rician衰落，随机数据到达率为1 Kbit/s）进行实验，并无公开真实数据集。

**📈 对比分析**

与固定相位/提取/解码顺序、Lyapunov、ALL-Selection、Plain-PPO等基线比较，JTAC方案在能效上领先约10–20%，PDOO在保持近似能效的同时将计算时间降低至原来的1/5–1/10；可延迟提取在高流量场景下能提升约19%。

**⚠️ 局限性**

局限性：采用线性语义提取模型，忽略了真实语义编码的复杂性；仿真环境简化，缺乏真实部署验证；PPO训练依赖大量经验交互，可能在极端动态环境下收敛慢。

---

## 444. Generative Artificial Intelligence Assisted Multi-modal Semantic Extraction for NOMA-based Image Transmissions

**arXiv ID:** 2603.21092 | [PDF](https://arxiv.org/pdf/2603.21092v1)

**作者:** Songhan Zhao `[一作]` (Sun Yat-sen University), Yuming Fang `[通讯]` (Jiangxi University of Finance and Economics)

**通讯引用:** 10534 | [OpenAlex ID](https://openalex.org/A5063013411)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于生成式人工智能（GAI）的多模态语义提取与NOMA联合控制方案（JTSC），通过语义特征选择、NOMA解码顺序与接收波束形成协同优化，实现图像语义传输的精度与时延折衷。

**💡 创新点**

创新点包括：①将语义特征选择与NOMA传输控制耦合成一体化优化；②设计跨模态匹配方法量化文本与视觉特征重要性并进行特征剪枝；③提出重要性感知与模型驱动的PPO（IM‑PPO）框架，将模型可知部分采用模型优化，黑盒部分采用PPO学习，显著提升学习效率。

**🔧 技术方法**

采用了Stable Diffusion v1‑5（作为语义解码的生成模型）、UperNet和BLIP（分别提取视觉与文本特征）、PPO与IM‑PPO强化学习、跨模态匹配与特征剪枝算法、基于通用模型的波束形成与NOMA优化方法。

**📊 数据集**

使用COCO‑2017图像数据集进行实验，训练使用预训练模型（UperNet、BLIP、Stable Diffusion）。

**📈 对比分析**

与ALL‑Selection、Random‑Selection、Location‑Decoding等基线方案以及M‑PPO、Plain‑PPO等强化学习方法对比。实验表明JTSC在语义恢复准确率（LPIPS）与平均时延上均优于基线，IM‑PPO在收敛速度和最终奖励上分别比M‑PPO快约3倍、比Plain‑PPO快约3.5倍，且在不同带宽、用户数与发射功率场景下均保持较优性能。

**⚠️ 局限性**

局限性主要在于：①对预训练模型的依赖，若模型与应用场景差异大则性能下降；②特征剪枝阈值需手工设定，影响鲁棒性；③多模态匹配和PPO的计算复杂度在大规模用户或高分辨率图像时仍显高。

---

## 445. ViCLSR: A Supervised Contrastive Learning Framework with Natural Language Inference for Natural Language Understanding Tasks

**arXiv ID:** 2603.21084 | [PDF](https://arxiv.org/pdf/2603.21084v1)

**作者:** Tin Van Huynh `[一作]` (University of Information Technology), Ngan Luu-Thuy Nguyen `[通讯]` (University of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ViCLSR，一种针对越南语句子表示的监督对比学习框架，利用 NLI 数据生成正负样本来提升句子嵌入质量。

**💡 创新点**

创新点在于将越南语 NLI 数据转化为监督对比样本，并结合 XLM‑R 作为编码器，首次在低资源语言上实现了对比学习的系统化应用。

**🔧 技术方法**

使用 XLM‑R_Large 作为编码器，训练时采用温度缩放的监督对比损失（contrastive loss），并在下游任务中进一步微调。

**📊 数据集**

主要使用 ViNLI、XNLI（越南语）生成对比数据，评测数据集包括 ViNLI、ViWikiFC、ViFactCheck、UIT‑ViCTSD 和 ViMMRC2.0。

**📈 对比分析**

与 PhoBERT、CafeBERT、mBERT、XLM‑R 和 DiffCSE 等基线相比，ViCLSR 在 ViNLI 及其他任务上分别提升约 6.9%、4.9%、9.0%、5.4% 和 4.3% 的 F1/准确率，并在 MRC 上提升约 17% 以上，证明对比学习显著提升越南语 NLU 性能。

**⚠️ 局限性**

局限性包括：对对比训练依赖 NLI 数据，若无此类数据需人工构造；在多选 MRC 等复杂推理任务上的提升有限；未探索动态温度或自适应负样本采样等进一步优化方向。

---

## 446. ReDiffuse: Rotation Equivariant Diffusion Model for Multi-focus Image Fusion

**arXiv ID:** 2603.21129 | [PDF](https://arxiv.org/pdf/2603.21129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 447. CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models

**arXiv ID:** 2603.21077 | [PDF](https://arxiv.org/pdf/2603.21077v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 448. Assessing the Ability of Neural TTS Systems to Model Consonant-Induced F0 Perturbation

**arXiv ID:** 2603.21078 | [PDF](https://arxiv.org/pdf/2603.21078v1)

**作者:** Tianle Yang `[一作]` (University at Buffalo), Siwei Lyu `[通讯]` (University at Buffalo)

**通讯引用:** 16479 | [OpenAlex ID](https://openalex.org/A5023752172)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出段落级音调扰动探针框架，评估神经TTS模型在重现辅音引起的 f0 扰动上的能力。

**💡 创新点**

将细粒度语音学现象（辅音诱发 f0 扰动）作为诊断性评估指标，揭示 TTS 模型在低频词汇上的泛化不足。

**🔧 技术方法**

采用 AR1 GAMM 进行时间-非线性建模，并结合时间归一化 f0 轨迹，比较 Tacotron 2、FastSpeech 2 与自然语音。

**📊 数据集**

使用 LJ Speech 单说话人语料、COCA 语料以及多说话人 In‑the‑Wild 深度伪造语料进行训练与评估。

**📈 对比分析**

按词频或训练出现情况分组，对高频与低频词汇进行对比；结果显示高频词语音重现良好，低频词语音缺失或弱化 f0 扰动，证明两模型在未见词上表现差。

**⚠️ 局限性**

仅评估了两种主流 TTS 架构，未覆盖扩散等新模型；并且使用词频作为训练曝光的代理，可能引入噪声。

---

## 449. CTFS : Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels

**arXiv ID:** 2603.21071 | [PDF](https://arxiv.org/pdf/2603.21071v1)

**作者:** Ping Guo `[一作]` (Dalian University of Technology), Xin Fan `[通讯]` (Dalian University of Technology)

**通讯引用:** 11953 | [OpenAlex ID](https://openalex.org/A5057776894)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种协同教师框架CTFS，用于在前视声纳图像中进行少量标注下的语义分割

**💡 创新点**

创新点包括：1）引入多教师协同机制，包含通用教师和专门的声纳教师；2）设计多视角可靠性评估（MVRA）来动态衡量伪标签质量；3）构建新的FSSG声纳语义分割数据集

**🔧 技术方法**

采用教师-学生半监督学习、EMA、强弱增强、基于DINOv2的特征提取、拼接多教师预测、余弦相似度一致性度量以及可靠性加权的CE损失

**📊 数据集**

使用公开的FLSMD数据集和自行构建的FSSG数据集，分别包含数千幅前视声纳图像和多目标类别

**📈 对比分析**

与AEL、Dual Teacher、UniMatch V1/V2、Beyond-Pixels、SemiVL、VC等state‑of‑the‑art方法对比，CTFS在2%、5%、10%标注比例下均实现了显著的mIoU提升，尤其在极少量标注时提升超过5%

**⚠️ 局限性**

局限性主要在于：1）对声纳设备和环境的特定性，跨设备泛化需进一步验证；2）多教师策略增加计算与存储开销；3）可靠性阈值和网格尺寸的超参数需要经验调优

---

## 450. One Pool Is Not Enough: Multi-Cluster Memory for Practical Test-Time Adaptation

**arXiv ID:** 2603.21135 | [PDF](https://arxiv.org/pdf/2603.21135v1)

**作者:** Yu-Wen Tseng `[一作]` (National Taiwan University), Wen-Huang Cheng `[通讯]` (National Taiwan University)

**通讯引用:** 4338 | [OpenAlex ID](https://openalex.org/A5101402074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种多聚类记忆(MCM)，通过像素级统计描述符将测试样本划分为若干子集，并配合相邻聚类合并和均衡检索，实现更稳定的测试时适应。

**💡 创新点**

创新点在于揭示实际测试流本质上是多模态的，并将记忆从单一无结构池转为多聚类结构，利用阈值聚类、相邻聚类合并和均衡检索三种机制保留分布多样性。

**🔧 技术方法**

采用像素通道均值/方差描述符、GMM诊断、Mean Teacher一致性学习、欧式距离聚类、相邻聚类合并和均衡检索等技术。

**📊 数据集**

在CIFAR-10-C、CIFAR-100-C、ImageNet-C和DomainNet四个公开基准上进行实验。

**📈 对比分析**

与RoTTA、PeTTA、ResiTTA等现有基于记忆的TTA方法对齐后，MCM在所有12个配置上均提升，平均误差下降约2.96%，ImageNet-C最高提升5%，DomainNet最高提升12.13%，并通过GMM诊断显示更均衡、熵更高、模式覆盖更完整。

**⚠️ 局限性**

局限性包括额外的聚类与描述符计算开销、仅基于低阶像素统计，可能对几何或语义级偏移不敏感，且仅在图像分类上验证，未探讨其他模态或任务。

---

## 451. Single-Eye View: Monocular Real-time Perception Package for Autonomous Driving

**arXiv ID:** 2603.21061 | [PDF](https://arxiv.org/pdf/2603.21061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 452. NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection

**arXiv ID:** 2603.21069 | [PDF](https://arxiv.org/pdf/2603.21069v1)

**作者:** Yupeng Zhang `[一作]` (Tianjin University), Liang Wan `[通讯]` (Tianjin University)

**通讯引用:** 3572 | [OpenAlex ID](https://openalex.org/A5000209938)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出NoOVD框架，利用冻结的CLIP视觉语言模型在训练时自蒸馏发现潜在新类别并在测试时通过R-RPN提升新类别召回率。

**💡 创新点**

①K‑FPN构造无可学习参数的特征金字塔，最大化保留CLIP知识；②利用文本提示和CLIP相似度在训练阶段主动发现潜在新类别并对其进行自蒸馏；③在测试阶段用R‑RPN重新加权RPN得分，避免新类别被提前剔除。

**🔧 技术方法**

冻结CLIP图像/文本编码器；K‑FPN、RPN、RoI head；自蒸馏（L2损失对齐RoI与CLIP特征）；文本提示生成（ChatGPT）；余弦相似度对比；重加权RPN。

**📊 数据集**

OV‑LVIS、OV‑COCO、Objects365；使用CLIP ViT‑B/16 与 ViT‑L/14 作为骨干。

**📈 对比分析**

与多种SOTA OVD 方法（ViLD、Detic、OV‑World、YOLO‑World、CLIPSelf、DeCLIP 等）进行实验比较；在OV‑LVIS 上整体 AP 提升 2–3%（尤其稀有类 2.8%），在OV‑COCO 上整体 AP 提升 1–3%；跨域迁移至 Objects365 同样优于对手。

**⚠️ 局限性**

对 OV‑COCO 的提升受限于标注不完整；需离线预处理提取候选框特征产生额外开销；模型对 CLIP 预训练和文本提示质量仍有依赖。

---

## 453. Learning to Optimize Joint Source and RIS-assisted Channel Encoding for Multi-User Semantic Communication Systems

**arXiv ID:** 2603.21097 | [PDF](https://arxiv.org/pdf/2603.21097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 454. On generalized covering radii of binary primitive double-error-correcting BCH codes

**arXiv ID:** 2603.21068 | [PDF](https://arxiv.org/pdf/2603.21068v1)

**作者:** Maosheng Xiong `[一作]` (Hong Kong University of Science and Technology), Chi Hoi Yip `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 160 | [OpenAlex ID](https://openalex.org/A5061055178)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究二进制原始双误差校正BCH码的通用覆盖半径（GCR）层次，提出通用超码引理以简化低阶GCR下界证明，并给出更紧的4阶下界；通过Weil型指数和估计，证明当码长指数 m 足够大时，任意阶 GCR 满足 2k ≤ ρ_k ≤ 2k+1。

**💡 创新点**

①提出通用超码引理，将 GCR 与适当超码的广义汉明重量关联；②用该引理重新证明 ρ_2、ρ_3 的已知下界并显著简化论证；③给出 ρ_4 的新下界；④结合指数和估计得到对任意 k 的两侧界。

**🔧 技术方法**

主要技术包括：通用超码引理（编码理论方法）；广义汉明重量、通用覆盖半径的定义；组合与代数计数技巧；Weil 型指数和估计（数论）来估计多项式字符和；符号性引理（如二元多项式的根的判定）等。

**📊 数据集**

无具体实验数据集；研究纯理论性质，分析BCH(2,m)码的结构。

**📈 对比分析**

与之前依赖极其复杂组合证明的方法相比，本工作通过通用超码引理大幅简化了 ρ_2、ρ_3 的推导，且对 ρ_4 给出了新的下界；对于大 m 的任意阶 k，提供了明确的上下界 2k ≤ ρ_k ≤ 2k+1，说明了 GCR 层次的渐近行为。

**⚠️ 局限性**

仍未求得所有 k 和较小 m 的精确 ρ_k 值；通用超码引理的应用范围尚未完全探讨；对于其他重要码族的 GCR 评估需要进一步研究。

---

## 455. The Role of Road Features and Vehicle Dynamics in Cost-Effective Autonomous Vehicles Safety Testing: Insights from Instance Space Analysis

**arXiv ID:** 2603.21066 | [PDF](https://arxiv.org/pdf/2603.21066v1)

**作者:** Victor Crespo-Rodriguez `[一作]` (Monash University), Sebastiano Panichella `[通讯]` (University of Bern)

**通讯引用:** 5124 | [OpenAlex ID](https://openalex.org/A5063227479)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用 Instance Space Analysis 结合静态路面特征和动态车辆行为特征，对自动驾驶车辆仿真测试案例进行分析，识别影响安全性的关键特征，并构建预测模型以在不执行仿真的情况下预测测试结果。

**💡 创新点**

首次同时考虑静态与动态特征的相互作用，构建统一的实例空间；通过 ISA 筛选关键特征，并利用机器学习实现测试结果的提前预测，弥补了以往仅研究单一特征类型的局限。

**🔧 技术方法**

Instance Space Analysis（ISA）、特征选择与聚类（K‑means）、PILOT 投影、PCA 降维、随机森林、决策树、KNN、MLP、朴素贝叶斯等机器学习分类器。

**📊 数据集**

SensoDat 数据集（超过 32,000 条测试案例），从中抽取 6,122 条平衡样本，包含三种生成技术（Frenetic、FreneticV、AmbieGen）的道路与车辆行为数据。

**📈 对比分析**

通过比较仅使用静态特征、仅使用动态特征以及两者结合的模型，评估 precision、recall 与 F1 分数。组合特征的随机森林模型取得最高性能：P=0.958、R=0.915、F1=0.936，明显优于单一特征模型。

**⚠️ 局限性**

仅在单车控制器和特定仿真环境下验证，未覆盖多车、多天气等更复杂场景；ISA 的特征选择对不同数据集可能需重新调整；模型使用默认超参数，未做细致调优，可能影响泛化能力。

---

## 456. Learning Progressive Adaptation for Multi-Modal Tracking

**arXiv ID:** 2603.21100 | [PDF](https://arxiv.org/pdf/2603.21100v1)

**作者:** He Wang `[一作]` (Jiangnan University), Josef Kittler `[通讯]` (University of Surrey)

**通讯引用:** 51864 | [OpenAlex ID](https://openalex.org/A5028209738)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的多模态跟踪框架 PATrack，利用 RGB 预训练网络通过渐进式适配实现 RGB+T、RGB+D、RGB+E 三种模态的高效跟踪。

**💡 创新点**

创新点在于三层级的渐进式适配：模态依赖适配器（MDA）强化单模态高低频特征，模态共鸣适配器（CEA）通过跨模态注意力捕获共享信息，头部适配器（HA）微调预测头以适配多模态输入，从而显著提升鲁棒性。

**🔧 技术方法**

技术手段包括基于 OSTrack 的 Transformer 结构、参数高效微调（adapter）、交叉注意力、频域分离、Bottleneck 设计、动态稀疏化与结构剪枝等。

**📊 数据集**

在公开多模态跟踪数据集上评测：LasHeR（RGB‑T）、VisEvent（RGB‑E）、DepthTrack（RGB‑D）、RGBT234、GTOT 等。

**📈 对比分析**

与 BAT、M3Track、SDSTrack、ViPT 等多模态与单模态跟踪器在 SR、PR、F‑score 等指标上进行对比，PATrack 在 RGB‑T、RGB‑D、RGB‑E 任务上均取得最高或接近最高的性能，且参数量与 FLOPs 低于多数全微调方法。

**⚠️ 局限性**

局限性包括需要针对每种模态分别训练、缺少在线模板更新机制导致长序列误差、对不同模态的差异适配仍不充分、跟踪效率有待进一步提升，并且缺乏一次性训练即可适配所有模态的统一模型。

---

## 457. Representation-Level Adversarial Regularization for Clinically Aligned Multitask Thyroid Ultrasound Assessment

**arXiv ID:** 2603.21095 | [PDF](https://arxiv.org/pdf/2603.21095v1)

**作者:** Dina Salama `[一作]` (University of British Columbia), Ilker Hacihaliloglu `[通讯]` (University of British Columbia)

**通讯引用:** 3117 | [OpenAlex ID](https://openalex.org/A5015205742)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种临床对齐的多任务模型，联合预测甲状腺超声图像的结节分割和TI‑RADS风险分级。

**💡 创新点**

通过在分类嵌入中预先注入TI‑RADS相关的可解释放射组学目标，并设计了表示层级对抗正则化（RLAR）来显式调节共享特征中任务梯度的相互干扰。

**🔧 技术方法**

使用共享编码器（基于nnU‑Net），多任务头（分割+分类），放射组学目标监督，RLAR对抗正则化，Dice损失、加权交叉熵、L2回归等。

**📊 数据集**

主要使用公开的两大甲状腺超声数据集：ThyroidXL（静态B‑mode图像）和AIMI（含有cine‑clip帧的动态序列）。

**📈 对比分析**

与单任务基线（EfficientNet‑B7、ConvNeXt）以及传统多任务基线（Vanilla Multitask、Rep‑MTL）对比，临床引导+RLAR模型在ThyroidXL上提升了TI‑RADS分类的精度、召回和F1，并在外部AIMI数据上保持了较低的HD95和稳定的召回率，表现优于其他方法。

**⚠️ 局限性**

主要局限包括：对放射组学特征的依赖可能限制在不同成像设备或协议下的迁移；RLAR参数需手工调节；模型在极端标注不一致或多模态数据上的泛化尚未完全验证。

---

## 458. Zero-Shot Vulnerability Detection in Low-Resource Smart Contracts Through Solidity-Only Training

**arXiv ID:** 2603.21058 | [PDF](https://arxiv.org/pdf/2603.21058v1)

**作者:** Minghao Hu `[一作]` (George Mason University), Lannan Luo `[通讯]` (George Mason University)

**通讯引用:** 842 | [OpenAlex ID](https://openalex.org/A5067262852)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种三阶段框架，实现仅用 Solidity 训练即可对 Vyper 合约进行零样本漏洞检测。

**💡 创新点**

创新点在于利用 SlithIR 作为语言无关中间表示，通过 MMD 对齐学习跨语言可迁移语义，并结合多视角编码器实现跨语言迁移。

**🔧 技术方法**

使用 SlithIR、Transformer、GAT、最大均值差（MMD）对齐、全连接分类器等技术。

**📊 数据集**

采用 1842 份 Solidity 合约与 1193 份 Vyper 合约的无标签通用数据集，以及 Solidity 的有标签漏洞数据集（共 1275-785 份），并手工构建约 1203 份 Vyper 漏洞样本。

**📈 对比分析**

与 Slither、Mythril、Vyper 训练模型、CodeT5+、GraphCodeBERT、LLM 等基线对比，零样本 Vyper 检测 FPR/FNR 分别为 0.13/0.15、0.12/0.14、0.13/0.15，显著优于所有基线且接近 Solidity 上的表现。

**⚠️ 局限性**

主要限制是对 Vyper 的适用范围受限于 SlithIR 的语义覆盖，且在高度语义差异或新漏洞类型时可能迁移效果下降，且依赖大量 Solidity 训练样本。

---

## 459. MS-CustomNet: Controllable Multi-Subject Customization with Hierarchical Relational Semantics

**arXiv ID:** 2603.21136 | [PDF](https://arxiv.org/pdf/2603.21136v1)

**作者:** Pengxiang Cai `[一作]` (East China University of Science and Technology), Mengyang Li `[通讯]` (Tianjin University)

**通讯引用:** 2848 | [OpenAlex ID](https://openalex.org/A5100462419)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 MS-CustomNet，能够在单张图像中实现多主体的零射定制，并提供层级关系与空间位置的显式控制。

**💡 创新点**

创新点包括：① 用层级关系与空间布局图（M_L）作为明确的生成约束；② 设计 Dual Stage Training 与 Curriculum Learning on Subject Quantity 的训练策略；③ 构建基于 COCO 的 MSI 多主体交互数据集。

**🔧 技术方法**

技术手段主要是：使用隐式扩散模型（Latent Diffusion Model）结合 CustomNet 框架；通过 CLIP 图像与文本编码器、类别编码器生成条件向量；将布局图编码为额外的条件张量并注入 U‑Net；采用自监督的噪声预测损失进行训练。

**📊 数据集**

使用了从 COCO 2017 训练集筛选并构建的 MSI 数据集（约 14,537 张多主体场景图），并在 80 个类别上采集 2,400 张参考图。

**📈 对比分析**

与 IP-Adapter、CustomNet、λ‑ECLIPSE、SSR‑Encoder 等单/多主体定制方法对比，MS‑CustomNet 在身份保持（DINO‑I/CLIP‑I）与位置精度（YOLO‑L）方面表现相当或更优，且在多主体场景下显著提升了层级与空间控制能力；在综合指标上 YOLO‑L 达 0.94，展示出高精度的空间布局实现。

**⚠️ 局限性**

主要限制：对背景内容的语义一致性仍受布局约束影响，CLIP‑B 评分相对较低；模型对极其复杂或动态交互的多主体场景泛化能力有限；需要手工生成或提供布局图，对用户操作成本有一定要求。

---

## 460. PrismWF: A Multi-Granularity Patch-Based Transformer for Robust Website Fingerprinting Attack

**arXiv ID:** 2603.21117 | [PDF](https://arxiv.org/pdf/2603.21117v1)

**作者:** Yuhao Pan `[一作]` (Hong Kong University of Science and Technology), Nan Cheng `[通讯]` (Xidian University)

**通讯引用:** 15085 | [OpenAlex ID](https://openalex.org/A5050651525)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 PrismWF，一种针对多标签混合流量的多粒度 Patch Transformer 网站指纹攻击模型；

**💡 创新点**

创新点在于：1）构造鲁棒的时槽特征表示，兼顾包量、方向和时间间隔；2）多分支 CNN 提取不同粒度特征；3）设计三层交互的多粒度注意力模块，包括细粒度向粗粒度补全、同粒度 Patch 交互与路由器聚合、跨粒度路由器融合；4）路由器机制模拟全局粗检与局部细问的认知逻辑；

**🔧 技术方法**

采用 Transformer‑based 多粒度注意力框架，配合卷积分支、局部/全局自注意力、多头交叉注意力以及路由器聚合；

**📊 数据集**

在公开的多标签 WF 数据集上评测，涵盖 2‑5 个并发标签、封闭世界与开放世界两种设置；

**📈 对比分析**

与现有单标签、双标签以及多标签攻击方法（AWF, DF, Tik‑Tok, Var‑CNN, BAPM, TMWF, ARES, CountMamba）以及防御下的性能对比，PrismWF 在 P@K/MAP@K 指标上持续领先，尤其在 5‑标签混合和防御场景中显著优于对手；

**⚠️ 局限性**

局限性：在监控网站规模扩展到数千时可能性能下降；对真实网络防御环境的评估仍缺乏；以及对不同防御策略的适配需要进一步研究。

---

## 461. CVT-Bench: Counterfactual Viewpoint Transformations Reveal Unstable Spatial Representations in Multimodal LLMs

**arXiv ID:** 2603.21114 | [PDF](https://arxiv.org/pdf/2603.21114v1)

**作者:** Shanmukha Vellamcheti `[一作]` (Auburn University), Sathyanarayanan N. Aakur `[通讯]` (Auburn University)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5080710999)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CVT-Bench，一个用于评估多模态大语言模型在假设视角变换和连续交互中的空间推理能力的诊断基准。

**💡 创新点**

创新点包括：① 通过视角旋转而不重新渲染图像，单独检测模型的视角转换能力；② 设计序列协议，考察模型在长上下文中的空间状态保持；③ 对比视觉、文本和场景图三种输入表征，揭示表征结构对空间一致性的影响。

**🔧 技术方法**

采用基于 CLEVR 的程序化场景生成，计算 3D 坐标转换得到视角下的空间关系；使用多标签 F1、视角一致性、360°循环一致性和空间生存率等指标评估；模型在推理时使用预训练的多模态 LLM（如 GPT‑5.2、Gemini‑3.1、Qwen‑3.5 Plus 等）进行无任务微调推理。

**📊 数据集**

使用 100 个由 CLEVR 生成的桌面场景，包含 3–10 个对象，产生约 6,000 条关系查询，涵盖 0°、45°…360° 旋转以及鸟瞰视角。

**📈 对比分析**

与多种模型和表征方式对比后发现：在观测视角 0° 时准确率可达 90%+；但随着视角偏移，尤其在 90°/270° 时 F1 明显下降；在连续序列推理下，准确率进一步衰减，视角一致性和 360° 循环一致性也随上下文长度下降。结构化输入（场景图）在部分模型中提升了稳定性，但整体仍不如单视角性能。

**⚠️ 局限性**

局限性包括：① 仅在合成、二维平面桌面场景上测试，缺乏真实世界复杂性；② 只评估离散视角旋转，未覆盖连续或三维空间运动；③ 依赖现有预训练模型的推理能力，未探索更深层的空间表示机制。

---

## 462. A lightweight Outlier Detection for Characterizing Radio- and Environment-Specific Link Quality Fluctuation in Low-Power Wireless Networks

**arXiv ID:** 2603.21107 | [PDF](https://arxiv.org/pdf/2603.21107v1)

**作者:** Zegeye Mekasha Kidane `[一作]` (Max Planck Institute for Radio Astronomy), Waltenegus Dargie `[通讯]` (Technische Universit{"a}t Dresden)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过在15个户外环境中部署四种低功耗无线电（CC1200、CC2538、nRF52840、BLE），对RSSI波动进行大规模实验，并提出轻量级基于指数移动平均（EMA）的统计离群检测方法；

**💡 创新点**

创新点在于将EMA参数自适应调节与基于Chebyshev不等式的敏感度系数k相结合，既能剔除高频噪声，又能区分环境与硬件引起的离群；

**🔧 技术方法**

主要技术包括指数移动平均滤波、基于均值方差的离群阈值设计、统计显著性检验（ANOVA）以及与z-score、MAD、MA等基线方法的对比；

**📊 数据集**

使用自采集的RSSI时间序列数据，覆盖五种自然环境（桥梁、森林、湖泊、河流、花园）以及多天多天气条件下的多次实验，形成包含数千条RSSI记录的数据集；

**📈 对比分析**

与基本EMA、z-score、MA、MAD等四种基线方法比较，结果显示Adaptive EMA在离群检测率上具有更低的误报率、更窄的分布范围，整体表现更为稳定；

**⚠️ 局限性**

局限性包括仅关注RSSI指标，未考虑其他链路质量指标；实验规模受设备与环境数量限制，无法覆盖所有可能的多路径或干扰场景；并且模型假设过程为宽平稳，可能不适用于极端动态环境。

---

## 463. Tracing Users' Privacy Concerns Across the Lifecycle of a Romantic AI Companion

**arXiv ID:** 2603.21106 | [PDF](https://arxiv.org/pdf/2603.21106v1)

**作者:** Kazi Ababil Azam `[一作]` (Bangladesh University of Engineering and Technology), Dipto Das `[通讯]` (University of Toronto)

**通讯引用:** 308 | [OpenAlex ID](https://openalex.org/A5001567136)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对一年内79个与浪漫AI聊天机器人的Reddit子版块中2,909条与隐私相关的帖子进行定性分析，探讨用户在使用过程中的隐私担忧。

**💡 创新点**

创新点在于提出“生命周期治理”视角，将隐私问题拆解为入门、深度使用、解读不确定性以及退出四个阶段，揭示隐私风险随关系演进而变化。

**🔧 技术方法**

方法上结合了自动化筛选（PRAW+ChatGPT辅助标签）与人类编码，形成主题式分析框架，并与现有隐私理论（情境完整性、通信隐私管理、Gumusel等）进行对照。

**📊 数据集**

数据集为2,909条公开的Reddit帖子，覆盖79个子版块，时间范围为2024年11月7日至2025年11月7日。

**📈 对比分析**

研究通过与已有隐私框架的匹配度进行比较，并阐述四个模式与框架中的隐私危害/风险的一致性与扩展，未给出数值性能指标。

**⚠️ 局限性**

局限性包括仅依赖公开讨论数据，可能存在样本偏差；未对平台内部机制做直接审计；研究时间窗口有限；缺乏深度访谈验证用户真实体验。

---

## 464. ResPrune: Text-Conditioned Subspace Reconstruction for Visual Token Pruning in Large Vision-Language Models

**arXiv ID:** 2603.21105 | [PDF](https://arxiv.org/pdf/2603.21105v1)

**作者:** Xu Li `[一作]` (Fudan University), Xiangyang Xue `[通讯]` (Fudan University)

**通讯引用:** 15013 | [OpenAlex ID](https://openalex.org/A5003418019)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 ResPrune 的训练无关视觉令牌裁剪框架，旨在提升大型视觉-语言模型（LVLM）推理效率。

**💡 创新点**

创新点在于将视觉令牌裁剪建模为子空间重构问题，并采用基于残差能量的贪婪子空间扩展策略，同时通过文本相关性门控实现跨模态指导，兼顾视觉覆盖与指令相关性。

**🔧 技术方法**

核心技术包括子空间重构理论、贪婪子空间扩展（残差能量计算）、文本相关性评分（最大余弦相似度）、门控函数调节裁剪权重，以及轻量级的 Gram–Schmidt 正交化更新实现。

**📊 数据集**

使用了多种公开视觉语言基准数据集：MME、GQA、POPE、TextVQA、VizWiz、VQA‑v2、MMB‑en、MM‑Vet 等，评估裁剪对不同任务的影响。

**📈 对比分析**

与多种现有裁剪方法（如 FastV、DivPrune、DART、SparseVLM 等）对比，ResPrune 在 LLaVA‑1.5、LLaVA‑NeXT、Qwen2.5‑VL 等三大 LVLM 框架下，保持 99% 以上的相对性能，且在 66.7%–88.9% 的裁剪率下仍能实现近乎无损或提升的任务表现。

**⚠️ 局限性**

局限性包括：文本引导强度由固定超参数 α 控制，需手工调优且对不同模型和输入不适应；未探索自适应或学习型引导机制；在极度稀疏令牌环境下可能导致信息丢失。

---

## 465. Mixture of Chapters: Scaling Learnt Memory in Transformers

**arXiv ID:** 2603.21096 | [PDF](https://arxiv.org/pdf/2603.21096v1)

**作者:** Tasmay Pankaj Tibrewal `[一作]` (IIT Kharagpur), Pradeep Moturi `[通讯]` (Fractal AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种记忆增强 Transformer（Mixture of Chapters），通过可学习的稀疏内存银行与章节路由实现大规模显式关联记忆。

**💡 创新点**

创新点包括：① 用可学习的 latent token 组成的稀疏内存银行；② 引入 Mixture of Chapters 的章节路由，将内存拆分为章节并按输入序列动态选择子集；③ 在持续训练过程中显著降低知识遗忘。

**🔧 技术方法**

使用的技术包括：Transformer decoder 架构、RMSNorm、交叉注意力（跨内存检索）、Mixture-of-Experts 启发的轻量级路由器 + Top‑k 章节选择、SwiGLU MLP、以及对比实验的 iso‑FLOP 计算匹配。

**📊 数据集**

实验数据集：预训练 9.6B token、指令微调 230M token，评测指标使用 MMLU、ARC‑Challenge、BoolQ、OpenBookQA 四个基准。

**📈 对比分析**

比较方法：在相同 FLOPs 的基线 Transformer 与加入内存的 MoC 进行对比。结果显示 MoC 在预训练验证损失上最低，在指令微调后对 ARC‑Challenge 与 BoolQ 的知识衰减明显低于基线，且冻结内存银行仍能保持相近表现，证明显式记忆能提升知识保持。

**⚠️ 局限性**

局限性：① 章节路由可能导致部分内存信息未被检索，影响细粒度知识表达；② 仅在中等规模模型上验证，未探究更大规模或更长训练的可扩展性；③ 对比仅限于内存银行的显式记忆，未与外部检索或 KV‑cache 等技术做更细致对照。

---

## 466. StreamTGN: A GPU-Efficient Serving System for Streaming Temporal Graph Neural Networks

**arXiv ID:** 2603.21090 | [PDF](https://arxiv.org/pdf/2603.21090v1)

**作者:** Lingling Zhang `[一作]` (Capital Normal University), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7370 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 StreamTGN 系统，实现连续时间图神经网络的增量推理，避免每条新边导致全图重计算。

**💡 创新点**

首次将 GPU 持久化节点内存、脏标记传播、增量邻居采样、注意力缓存与 delta 计算相结合，提供无精度损失的 O(|A|) 推理复杂度，并引入漂移感知自适应重建和松耦合批处理。

**🔧 技术方法**

使用 GPU 驻留混合数据结构、脏标记传播、增量邻居采样、特征合并、注意力缓存、delta 嵌入、漂移估计与自适应重建调度，以及批量松耦合排序等技术。

**📊 数据集**

在八个真实时序图数据集上评测，规模从 6K 节点的 Bitcoin、LastFM、MOOC 等到 2.6M 节点的 Stack‑Overflow、Wiki‑Talk、GDELT 等。

**📈 对比分析**

与 TGL、ETC、SIMPLE、SWIFT 等训练系统组合，或单独与 TGL 对比；在 TGN、TGAT、DySAT 三种模型上，在五个数据集上进行基准；StreamTGN 实现 4.5×–739× 推理加速、TGAT 达到 4,207×，端到端提升 24×，且准确率完全保持不变。

**⚠️ 局限性**

局限在于对批处理与边到达率的假设，漂移估计为近似，极端高频更新或高邻接度场景下影响集可能膨胀导致重建频繁；目前仅单 GPU 实现，尚未扩展到分布式大规模图。

---

## 467. A Two-stage Transformer Framework for Temporal Localization of Distracted Driver Behaviors

**arXiv ID:** 2603.21048 | [PDF](https://arxiv.org/pdf/2603.21048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 468. Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models

**arXiv ID:** 2603.21085 | [PDF](https://arxiv.org/pdf/2603.21085v1)

**作者:** Qifan Li `[一作]` (University of Electronic Science and Technology of China), Shuhang Gu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 12519 | [OpenAlex ID](https://openalex.org/A5100745570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种 Variance Expansion (VE) 损失，用来提升潜在空间对扩散采样扰动的鲁棒性，同时保持重建质量。

**💡 创新点**

创新点在于通过对抗式平衡重建损失与方差扩展，主动阻止 VAE 编码器的方差崩塌，而不是传统的 KL 正则化。

**🔧 技术方法**

采用 VAE 作为 tokenizer，结合 Flow‑matching 或 DiT 等扩散模型，加入 VE 损失；并对比不同 KL 权重的训练。

**📊 数据集**

ImageNet 256×256。

**📈 对比分析**

与使用传统 β‑VAE（不同 KL 权重）、VA‑VAE、LightningDiT 等基线对比；在 rFID、PSNR、LPIPS、FID‑10K、FID‑50K 等指标上，VE 损失既保持或略微提升重建指标，又显著提升生成质量（如 530 轮训练的 FID 下降至 1.18）。

**⚠️ 局限性**

局限性：在严格匹配训练设置时 VE 损失可能略微降低重建质量；需要调优 λ1、λ2 等超参数；目前仅在 ImageNet 256×256 进行验证，未验证在更高分辨率或其他域的迁移效果。

---

## 469. Semi-Supervised Learning with Balanced Deep Representation Distributions

**arXiv ID:** 2603.21056 | [PDF](https://arxiv.org/pdf/2603.21056v1)

**作者:** Changchun Li `[一作]` (Jilin University), Jihong Ouyang `[通讯]` (Jilin University)

**通讯引用:** 1207 | [OpenAlex ID](https://openalex.org/A5103822192)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在自训练框架下，提出一种通过平衡标签角度方差来提升半监督文本分类的模型。

**💡 创新点**

创新地引入高斯线性变换构造BDD损失，消除margin bias，显著提高伪标签质量。

**🔧 技术方法**

BERT+Angular Margin (AM) loss、Gaussian线性变换、伪标签策略（Sharpening、CAP）、熵正则化、低秩正则化、EMA等技术。

**📊 数据集**

多类别数据集：AG News、Yelp、Yahoo；多标签数据集：Ohsumed、AAPD、RCV1-V2。

**📈 对比分析**

与8个多类基线（BERT+CE、BERT+AM、VAMPIRE、VAT、UDA、FreeMatch、MetaExpert、NB+EM）以及5个多标签基线（BERT+BCE、BERT+AM、SDRL、CAP、MetaExpert）在Micro‑F1/Macro‑F1、Ranking Loss/AP等指标上对比，BDD在标签稀缺时平均提升排名至1.0-3.4，整体性能显著优于对比方法。

**⚠️ 局限性**

对极端长尾、不平衡、开放域样本以及噪声标签的鲁棒性不足；假设每类为单峰高斯分布可能不适用于多峰或复杂分布。

---

## 470. Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset for Counting Hundreds of Plant Species

**arXiv ID:** 2603.21229 | [PDF](https://arxiv.org/pdf/2603.21229v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 471. Harmful Visual Content Manipulation Matters in Misinformation Detection Under Multimedia Scenarios

**arXiv ID:** 2603.21054 | [PDF](https://arxiv.org/pdf/2603.21054v1)

**作者:** Bing Wang `[一作]` (Jilin University), Shengsheng Wang `[通讯]` (Jilin University)

**通讯引用:** 2005 | [OpenAlex ID](https://openalex.org/A5086280684)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究多模态谣言检测，提出通过检测视觉内容是否被操纵及其意图（有害/无害）来辅助判别文章真伪。

**💡 创新点**

创新点在于将视觉操纵检测与意图分类作为辅助任务，引入弱监督的操纵/意图标签，利用外部图像操纵检测（IMD）数据与正负不平衡（PU）学习对教师模型进行适配和蒸馏。

**🔧 技术方法**

技术上结合BERT+ResNet34特征提取，跨注意力融合，多任务学习框架；通过知识蒸馏传递教师模型的操纵判别能力；采用PU学习实现对未知操纵与意图标签的弱监督训练。

**📊 数据集**

使用了四大多模态谣言检测基准数据集——GossipCop、Weibo、Twitter、FakeSV；以及用于教师模型预训练的图像操纵检测数据集CASIAv2。

**📈 对比分析**

在上述四个数据集上与Base、SAFE、MCAN、CAFE、BMR等主流基线进行对比，平均提升1–3个百分点（如FakeSV提升约1.6个百分点），在小样本或高操纵比例场景表现尤为突出。

**⚠️ 局限性**

局限性包括：需额外训练IMD教师模型和PU学习，导致整体训练时间约增加1.3倍；对细微或新型视觉操纵识别仍有限；缺乏对实时出现的新事件的自适应能力。

---

## 472. Emotion-Aware Quantization for Discrete Speech Representations: An Analysis of Emotion Preservation

**arXiv ID:** 2603.21224 | [PDF](https://arxiv.org/pdf/2603.21224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 473. A Large-Scale Remote Sensing Dataset and VLM-based Algorithm for Fine-Grained Road Hierarchy Classification

**arXiv ID:** 2603.21222 | [PDF](https://arxiv.org/pdf/2603.21222v1)

**作者:** Ting Han `[一作]` (Sun Yat-Sen University), Yin Gao `[通讯]` (Transport Bureau of Nanhai District)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SYSU-HiRoads大型分级道路数据集，并基于此开发了RoadReasoner框架，实现高分辨率遥感图像中道路表面提取、网络重建及三级道路等级细粒度分类；

**💡 创新点**

创新点在于①首次同时提供像素级道路掩模、矢量中心线与三级等级标签的统一数据集；②结合频域增强与多尺度特征的FORCE-Net提升道路连通性；③采用几何提示与视觉‑语言模型（VLM）+GPT‑v相结合的T‑HRN实现可解释的道路等级判定；

**🔧 技术方法**

核心技术包括频域增强块（FDE）、并行多尺度特征提取块（PMSE）、基于多次回溯的中心线重建、几何描述符与文本提示的融合、VLM（如CLIP、DINOv2）与GPT‑v的推理；

**📊 数据集**

使用的数据集为：①SYSU-HiRoads（GF‑2 0.8 m，1,079张1,024×1,024图像，含像素掩模、中心线和三级等级）；②CHN6‑CUG等公开道路数据集做对比；

**📈 对比分析**

与LinkNet、D‑LinkNet、SGCNNet、RCFSNet、MACUNet、FRCFNet等现有道路分割与网络重建模型比较，FORCE‑Net在SYSU‑HiRoads上IoU 58.21%/F1 72.14%、在CHN6‑CUG上IoU 51.81%/F1 63.89%，均优于对手；RoadReasoner在等级分类上实现OA 72.6%、mIoU 48.9%、SegAcc 60.5%；

**⚠️ 局限性**

主要局限包括①在不同地区、传感器及时间域的泛化能力不足；②道路等级标签存在不确定性与标注一致性问题；③VLM与GPT‑v的可靠性、偏差与可控性需要进一步验证与改进；

---

## 474. Context Selection for Hypothesis and Statistical Evidence Extraction from Full-Text Scientific Articles

**arXiv ID:** 2603.21193 | [PDF](https://arxiv.org/pdf/2603.21193v1)

**作者:** Sai Koneru `[一作]` (Pennsylvania State University), Sarah Rajtmajer `[通讯]` (Pennsylvania State University)

**通讯引用:** 1397 | [OpenAlex ID](https://openalex.org/A5082663800)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种两阶段检索+抽取框架，用于从完整科研论文中先定位与摘要发现对应的假设，再检索并抽取支持该假设的统计证据。

**💡 创新点**

创新点在于系统地比较检索量、检索质量（含reranker和fine‑tuned检索器）与上下文干净度对抽取任务的影响，并通过oracle段落分离检索错误与抽取器性能瓶颈。

**🔧 技术方法**

使用检索增强生成（RAG）与大型语言模型抽取器，辅以RankGPT reranker、fine‑tuned检索器，构建多种检索配置（k=5/10/20、全文本）。

**📊 数据集**

基于前人发布的论文级别“claim trace”数据集（含抽象发现–假设–统计证据三元组），对比全文本提示与检索增强方式。

**📈 对比分析**

实验表明：提升检索质量与上下文干净度能显著提升假设抽取性能，检索量增大通常无负面影响；但即使在oracle证据段落下，统计证据抽取的F1仍仅0.47–0.55，显示抽取器仍有显著提升空间。

**⚠️ 局限性**

主要限制在于统计证据抽取依赖于抽取器自身对数值与文本混合句式的理解，检索器改进不足以弥补抽取瓶颈；此外，全文检索仍受长文本“lost‑in‑the‑middle”问题影响。

---

## 475. PC2IM: An Efficient In-Memory Computing Accelerator for 3D Point Cloud

**arXiv ID:** 2603.21167 | [PDF](https://arxiv.org/pdf/2603.21167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 476. DSCSNet: A Dynamic Sparse Compression Sensing Network for Closely-Spaced Infrared Small Target Unmixing

**arXiv ID:** 2603.21192 | [PDF](https://arxiv.org/pdf/2603.21192v1)

**作者:** Zhiyang Tang `[一作]`, Fan Fan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DSCSNet，对闭合小目标去混（CSOU）任务进行稀疏重建，利用深度展开的ADMM框架实现端到端训练。

**💡 创新点**

创新点在于：①严格ℓ1稀疏约束替代传统ℓ2平滑，②自注意力动态阈值生成器与信息重组模块实现自适应稀疏性，③将物理压缩感知逻辑与数据驱动学习无缝融合。

**🔧 技术方法**

采用技术包括：ADMM深度展开、动态卷积与自注意力机制、动态阈值生成、稀疏正则化及端到端联合优化。

**📊 数据集**

使用合成红外小目标数据集CSIST-100K进行训练与评估。

**📈 对比分析**

与传统ISTA、超分网络以及其他深度展开方法对比，DSCSNet在CSO-mAP、AP_25等指标上取得最高性能（96.54% AP_25），比SOTA提升约5%，在参数和算力上亦更优。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证，真实场景鲁棒性待进一步评估；迭代层数与动态阈值的超参数仍需经验调优；在多模态或实时部署场景下的适用性尚未探究。

---

## 477. Explainable Semantic Textual Similarity via Dissimilar Span Detection

**arXiv ID:** 2603.21174 | [PDF](https://arxiv.org/pdf/2603.21174v1)

**作者:** Diego Miguel Lozano `[一作]`, Alexander Fraser `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出词块级别的相异性检测（Dissimilar Span Detection，DSD）任务，并基于半自动标注方法构建了Span Similarity Dataset（SSD）数据集，随后开发并评估了多种基线方法。

**💡 创新点**

创新点在于首次将文本对中的语义相异span显式化并提供解释，发布了首个针对该任务的公开数据集 SSD，并提出了专门的Embedding‑DSD算法。

**🔧 技术方法**

采用的技术包括LIME、SHAP、LLM（ChatGPT 等）、Embedding‑DSD（基于句子嵌入的n‑gram替换），以及Token‑Classification（BERT/DistilBERT 等）等多种无监督和监督方法。

**📊 数据集**

实验使用自构建的 SSD（1000句对）以及 SemEval‑2016 中标记为 opposite 的 span 子集进行评估，亦在 PAWS‑Wiki 语料上验证 DSD 在释义检测任务中的效果。

**📈 对比分析**

评估结果显示 LLM 和 Token‑Classification 方法获得最高 F1 分数，Embedding‑DSD 在无监督方法中表现最好，整体 F1 仍较低，但在假对齐（paraphrase）任务中可提升约 8 个点。

**⚠️ 局限性**

主要局限包括仅处理英文句子、span 顺序固定、未考虑单侧出现的 span、以及仅检测不相似 span，模型对局部差异的敏感度仍不足。

---

## 478. Pruned Adaptation Modules: A Simple yet Strong Baseline for Continual Foundation Models

**arXiv ID:** 2603.21170 | [PDF](https://arxiv.org/pdf/2603.21170v1)

**作者:** Elif Ceren Gok Yildirim `[一作]` (Eindhoven University of Technology), Joaquin Vanschoren `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 6681 | [OpenAlex ID](https://openalex.org/A5016794035)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Pruned Adaptation Modules (PAM)，在预训练 ResNet 上冻结前层并为每个新任务增设稀疏的最后层，实现在连续学习中的高效适应。

**💡 创新点**

创新点在于将 PEFT 思想应用于 ConvNet，采用结构化剪枝显著减少可训练参数（2–5×）与整体参数（2–6×），同时保持甚至超过基于 ViT 的前沿 CIL 方法的性能。

**🔧 技术方法**

使用 ResNet（18/50/101/152）作为共享特征提取器，结合结构化剪枝、任务特定模块、统一分类器以及基于置信度的模块选择与推理策略。

**📊 数据集**

在四个基准数据集上评估：CIFAR‑100、ImageNet‑R、CUB‑200 与 Cars‑196，覆盖通用与细粒度分类任务。

**📈 对比分析**

与 SimpleCIL、L2P、DualPrompt、CODA‑Prompt、APER、EASE 及连续微调基线对比，PAM 在最终准确率和平均准确率上均优于所有 ViT‑based 方法，且参数占用大幅降低；在任务识别无标识的 CIL 场景中，PAM 的误选率极低，性能接近 TIL 上限。

**⚠️ 局限性**

局限性包括：对大规模 ViT 仍缺乏竞争力，平均准确率略低；仅在 ResNet 上验证，未探索 Transformer 后端；剪枝比例需经验调优，且推理时需额外的模块选择开销。

---

## 479. Development and Usability Study of Older Adults in Motion-Captured Serious Game Incorporating Olfactory Stimulations

**arXiv ID:** 2603.21220 | [PDF](https://arxiv.org/pdf/2603.21220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 480. Incentivizing Generative Zero-Shot Learning via Outcome-Reward Reinforcement Learning with Visual Cues

**arXiv ID:** 2603.21138 | [PDF](https://arxiv.org/pdf/2603.21138v1)

**作者:** Wenjin Hou `[一作]` (CCAI, Zhejiang University), Hehe Fan `[通讯]` (CCAI, Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

基于扩散生成模型与强化学习的零样本学习框架RLVC，利用任务奖励驱动生成器产生任务相关特征并通过视觉原型蒸馏提升生成质量。

**💡 创新点**

创新点在于：①将生成器视为RL策略，以下游分类器的置信度作为奖励；②引入类别级视觉原型进行蒸馏约束；③提出冷启动训练策略与交替优化，提升RL训练稳定性。

**🔧 技术方法**

技术包括扩散生成对抗网络、基于分类器的结果奖励RL、视觉原型蒸馏损失、EMA奖励平滑、视觉编码器微调与交替优化。

**📊 数据集**

使用CUB、SUN和AWA2三大零样本学习基准数据集。

**📈 对比分析**

相较SOTA方法，RLVC在CZSL和GZSL上均实现了最高准确率，CZSL提升约4.7%（最高90.1%），GZSL调和均值提升5.5%（最高0.6%），显著优于已有方法。

**⚠️ 局限性**

局限性包括：对视觉编码器的微调与预训练语义原型高度依赖，计算成本相对较高，且在某些数据集的S或U准确率不一定是最优，仅在调和均值上表现最佳。

---

## 481. Revisiting Tree Search for LLMs: Gumbel and Sequential Halving for Budget-Scalable Reasoning

**arXiv ID:** 2603.21162 | [PDF](https://arxiv.org/pdf/2603.21162v1)

**作者:** Leonid Ugadiarov `[一作]` (AXXX), Alexey Skrynnik `[通讯]` (AXXX)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ReSCALE，改进 AlphaZero 树搜索以提升 LLM 推理性能。

**💡 创新点**

将 Dirichlet 噪声与 PUCT 选择替换为 Gumbel 噪声和 Sequential Halving，实现可扩展的计算分配。

**🔧 技术方法**

采用 Gumbel MCTS、Sequential Halving、混合值估计以及改进的非根节点策略。

**📊 数据集**

使用 GSM8K 与 Game24 两个数学推理/规划数据集进行评估。

**📈 对比分析**

在三档预算下与 AlphaZero 对比，ReSCALE 在大预算下仍持续提升准确率，最终达到 58.4%（GSM8K）与 85.3%（Game24）。

**⚠️ 局限性**

仅在句子级动作上验证，缺乏对更复杂任务或更大模型的通用性评估。

---

## 482. Does AI Homogenize Student Thinking? A Multi-Dimensional Analysis of Structural Convergence in AI-Augmented Essays

**arXiv ID:** 2603.21228 | [PDF](https://arxiv.org/pdf/2603.21228v1)

**作者:** Keito Inoshita `[一作]` (Ritsumeikan University), Kentaro Tsuji `[通讯]` (Ritsumeikan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对6875篇学生写作（包含人类写作、AI写作以及三种不同人类+AI交互提示）进行实验，评估AI辅助写作对写作质量和结构多样性的影响。

**💡 创新点**

创新点在于首次量化AI辅助写作的质量提升与结构同质化的共存，并揭示不同结构维度的异质性以及提示设计能逆转同质化的机制。

**🔧 技术方法**

使用GPT‑5‑mini生成AI写作与人类+AI增强文本，利用GPT‑5对六个结构维度进行自动提取，并通过方差比、Homogenization Index、Replacement Ratio等统计指标进行评估。

**📊 数据集**

数据集基于公开的AIDE（AI Detection for Essays）语料，包含1375篇人类写作，随后使用同一模型生成1375篇AI写作及三种提示下的1375篇增强写作，总计6875篇。

**📈 对比分析**

通过与人类写作基准的均值和方差比较，采用Welch t检验和Brown‑Forsythe检验，结果显示所有三种提示均显著提升质量（Cohen's d>3.7），同时在部分维度实现方差压缩（最高4.6倍）。

**⚠️ 局限性**

局限包括实验仅基于单轮API交互、仅涉及英语两题、使用单一LLM模型、提示样本有限，且LLM评估可能存在偏差。

---

## 483. Positional Segmentor-Guided Counterfactual Fine-Tuning for Spatially Localized Image Synthesis

**arXiv ID:** 2603.21213 | [PDF](https://arxiv.org/pdf/2603.21213v1)

**作者:** Tian Xia `[一作]` (Imperial College London), Ben Glocker `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 Positional Segmentor-guided Counterfactual Fine-Tuning (Pos-Seg-CFT) 方法，利用预训练分割器得到的区域性结构测量来实现对医学影像的空间定位的因果干预，生成局部可控的反事实图像。

**💡 创新点**

创新点在于：①将全局结构变量拆分为多区域测量，实现对图像中不同位置（proximal/mid/distal）的精准干预；②保持只使用标量监督，无需像素级反事实掩码；③在 Seg-CFT 的基础上通过区域掩码重用同一分割器，保持实现简洁与可扩展性。

**🔧 技术方法**

技术包括：基于 Deep Structural Causal Models (DSCM) 的可逆流和高分辨率 HVAE；使用预训练且权重冻结的分割器生成区域测量；在 fine‑tune 阶段加入区域测量的 L1/L2 损失；采用干预评估指标 d(π_x, π_x') 来量化效果。

**📊 数据集**

使用内部冠状动脉 CT 血管造影（CCTA）数据集，包含 65,706 张 64×384 像素的 sCPR 图像，分为训练、验证、测试集；对三种结构变量（calcified plaque area, non‑calcified plaque area, lumen area）在三个位置（proximal/mid/distal）进行干预。

**📈 对比分析**

对比 No‑CFT、Reg‑CFT（使用 ResNet 回归器）与 Pos‑Seg‑CFT，评估干预在目标与非目标区域的误差。结果显示 Pos‑Seg‑CFT 在所有三类变量和三位置组合中均实现最低的误差，显著减少跨区域泄漏，生成的反事实图像更具局部性和解释性。

**⚠️ 局限性**

局限性包括：①仍假设区域划分固定且仅为三段，复杂解剖结构可能需要更细粒度划分；②依赖预训练分割器的准确性，若分割器误差大会影响测量；③目前仅处理独立结构变量，未考虑变量间的交互与多目标联合干预。

---

## 484. MI-DPG: Decomposable Parameter Generation Network Based on Mutual Information for Multi-Scenario Recommendation

**arXiv ID:** 2603.21209 | [PDF](https://arxiv.org/pdf/2603.21209v1)

**作者:** Wenzhuo Cheng `[一作]` (Ant Group), Linjian Mo `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MI‑DPG 模型，使用可分解动态加权矩阵和互信息正则化实现多场景 CVR 预测。

**💡 创新点**

创新点在于将低秩分解应用于动态加权矩阵并通过互信息正则化显著提升场景多样性，从而在保持低参数成本的前提下获得更好的跨场景性能。

**🔧 技术方法**

技术包括低秩分解、可分解加权矩阵模块、互信息正则化、全连接神经网络骨干、AUC/LogLoss 评估以及 t‑SNE 可视化。

**📊 数据集**

使用 Amazon 电子类公开数据集、IAAC 赞助搜索数据集以及工业广告系统点击转化数据集。

**📈 对比分析**

与 DNN、PLE、STAR、PEPNet、AdaSparse、APG 等基线模型对比，MI‑DPG 在 Amazon、IAAC、Advertising 三个数据集上均取得最高 AUC（分别为 0.6745、0.6504、0.7580）和最低 LogLoss（分别为 0.5062、0.0794、0.2237），表现显著优于最优基线。

**⚠️ 局限性**

局限性包括对场景感知特征质量的依赖、互信息正则化需要额外判别器且计算开销较大，以及在极低秩尺寸下可能出现欠拟合，且尚未评估跨域迁移或在线学习的适用性。

---

## 485. Anchored Likelihood-Ratio Geometry of Anonymous Shuffle Experiments: Exact Privacy Envelopes and Universal Low-Budget Design

**arXiv ID:** 2603.21197 | [PDF](https://arxiv.org/pdf/2603.21197v1)

**作者:** Alex Shvets `[一作]` `[通讯]`, Alex Shvets

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了一个以正则单纯形上的零均值概率分布为载体的几何框架，用以精确刻画匿名置乱模型下的差分隐私与机制设计问题。

**💡 创新点**

创新点在于提出“锚定锥”与投影纤维的对应关系，得到一维隐私简化、二值随机回应的极大化性质以及全局 χ* 前沿的精确描述，完成了低预算下的最优机制闭式解。

**🔧 技术方法**

使用了正则单纯形几何、凸顺序与分布极大化、对称化、投影变换、分布的凸顺序论、全局极值理论、Van Trees 下界等技术。

**📊 数据集**

无实测数据集；全部结果为理论证明与闭式公式。

**📈 对比分析**

通过理论比较，证明二值随机回应在任意字母表、任何一维隐藏拉普拉斯/χ² 损失下都能实现最优隐私，并给出低预算、原始局部限制下的最优机制（Augmented Randomized Response 与子集选择机制）的最优性。

**⚠️ 局限性**

局限性在于仅对有限样本的精确分析，无法涵盖固定组成的非高斯极限；且对多消息、跨域的置乱协议仍未给出完整结果。

---

## 486. Architecture for Multi-Unmanned Aerial Vehicles based Autonomous Precision Agriculture Systems

**arXiv ID:** 2603.21183 | [PDF](https://arxiv.org/pdf/2603.21183v1)

**作者:** Ebasa Temesgen `[一作]` (University of Minnesota), Lebsework Negash `[通讯]` (Addis Ababa Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了一套多无人机协同工作的端到端精密农业系统，涵盖任务规划、路径规划、数据采集、图像处理，并通过实验验证了其可靠性与实用性。

**💡 创新点**

提出结合集中与分散决策的架构、动态任务分配与电池更换站，并通过深度学习实现作物与杂草的高精度分类，实现多无人机高效、无人工干预的精准农业。

**🔧 技术方法**

采用PX4/APM3飞控、Raspberry Pi、MAVLink无线与Wi‑Fi通信、Gazebo仿真、ROS/Rviz、CNN卷积网络进行图像分类，以及自定义的任务分配与电池更换算法。

**📊 数据集**

使用从Kaggle获取的豆类作物图像数据集，包含约15,336个手工标注的图像片段（土壤、豆类、草、阔叶杂草），用于CNN训练与测试。

**📈 对比分析**

通过Gazebo模拟与亚的斯阿贝巴大学校园实机测试评估系统的可扩展性、容错性与用户友好性；CNN模型在测试集上达到了约96.6%的准确率，显示出优异的图像分类性能。

**⚠️ 局限性**

任务分配算法缺乏多变量优化与真正去中心化的容错机制；电池更换站的部署与路径规划复杂度较高；在大面积农田中通信可靠性与实时性仍面临挑战。

---

## 487. ALMAB-DC: Active Learning, Multi-Armed Bandits, and Distributed Computing for Sequential Experimental Design and Black-Box Optimization

**arXiv ID:** 2603.21180 | [PDF](https://arxiv.org/pdf/2603.21180v1)

**作者:** Foo Hui-Mean `[一作]` (Academia Sinica), Yuan-chin I Chang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ALMAB-DC 框架，融合主动学习、Multi‑Armed Bandit 与分布式异步计算，用于在昂贵、无梯度的黑盒实验中实现高效的序贯实验设计。

**💡 创新点**

核心创新在于：① 将 GP 代理与不确定性感知的采集函数嵌入主动学习，② 用 UCB/Thompson Sampling 控制资源分配，③ 在多机异步环境中实现无锁调度，并给出累积 regret 与 Amdahl 定律的理论界限。

**🔧 技术方法**

技术方法包括：高斯过程代理、UCB/TS 决策、Kriging believer 采样、分布式异步调度、Amdahl/Gustafson 可伸缩性分析和统计显著性检验（Bonferroni 校正的 Mann‑Whitney U）。

**📊 数据集**

实验使用了五个基准：(1) CIFAR‑10 迁移学习超参搜索；(2) CFD 空气动力学阻力最小化；(3) MuJoCo 半跑者策略搜索；(4) 药物剂量‑反应实验；(5) 8×8 网格上的高斯过程场估计。

**📈 对比分析**

与 Grid/Random、BOHB、Optuna 等基线比较，ALMAB-DC 在所有任务上都实现了显著优势：CIFAR‑10 精度提升 1.7pp、CFD 阻力降低 36.9%、RL 回报提升 50%，且在分布式设置下实现 7.5× 的加速；所有优势均在 Bonferroni 校正后显著。

**⚠️ 局限性**

主要局限包括：GP 代理的 O(N³) 计算开销限制了大规模预算；当前 MAB 模型假设工作节点同质，未处理异构计算；噪声鲁棒性验证仅针对 HPO 场景；缺乏真实系统级别的长时间实验验证。

---

## 488. Prompt replay: speeding up grpo with on-policy reuse of high-signal prompts

**arXiv ID:** 2603.21177 | [PDF](https://arxiv.org/pdf/2603.21177v1)

**作者:** Andrei Baroian `[一作]` (Leiden University), Rutger Berger `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Prompt Replay，一种无额外开销的在线数据选择方法，在 GRPO 风格的 RLVR 训练中仅重用提示，按中等难度（pass rate 近 0.5）排序，以提升样本效率。

**💡 创新点**

创新点在于仅存储提示不存轨迹的重放缓冲区，基于实时估计的 pass rate 进行优先级排序，避免了额外难度评估开销，实现了高效的在线经验重放。

**🔧 技术方法**

使用了 GRPO/OLMo-RL 训练框架、RLVR、经验重放机制、在线课程学习、pass rate 评估、混合采样、冷却期（cooldown）与重用上限等技术。

**📊 数据集**

实验数据集包括 Dolci RL Zero Math、Polaris 53k 以及 6 个数学基准（AIME25、AIME24、AMC、MATH500、OlympiadBench、MinervaMath）。

**📈 对比分析**

与 OLMo-RL 基线对比：Prompt Replay 在训练初期显著提高平均基准准确率、减少零方差提示、提升平均绝对优势，但最终趋于基线，过度重用导致早期平台。

**⚠️ 局限性**

局限性包括：过度重用可能导致过拟合和平台；对 Qwen 2.5-Math 的假奖励失效；超参数（冷却期、重用比例等）敏感且调优空间大；实验规模受限，未覆盖更多模型和数据；计算资源有限。

---

## 489. Entropy Alone is Insufficient for Safe Selective Prediction in LLMs

**arXiv ID:** 2603.21172 | [PDF](https://arxiv.org/pdf/2603.21172v1)

**作者:** Edward Phillips `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**通讯引用:** 14010 | [OpenAlex ID](https://openalex.org/A5040302008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了选择性预测系统在减轻大型语言模型幻觉时的作用，并通过在低风险阈值下评估不确定性量化方法的表现。

**💡 创新点**

创新点在于识别了熵基不确定性方法的“自信错误”失效模式，并提出将熵分数与正确性探针结合的轻量级组合策略，从而在多模型、多数据集上提升风险-覆盖平衡与校准性能。

**🔧 技术方法**

主要技术包括熵基风险信号（序列负对数似然、语义熵及其探针）、正确性探针（基于对抗样本的逻辑回归）、两者的逻辑回归组合、以及针对选择性预测的 AURC 与 TCE 评价指标。

**📊 数据集**

实验数据集覆盖 TriviaQA、BioASQ 与 MedicalQA 三个问答基准，使用了 Mistral、Llama、Qwen 与 Gemma 四大 3B–8B 参数级模型。

**📈 对比分析**

与单一熵或正确性探针基准相比，组合方法在 AUROC、AUPRC、E‑AURC 与 TCE 上均取得显著提升，尤其在高信任阈值下显著降低风险阈值误差；但在 MedicalQA 上组合效果有限。

**⚠️ 局限性**

局限性包括仅针对短文本问答实验，可能不适用于长篇生成或多步骤推理；受限于模型规模与训练样本，熵失效模式可能在更大模型上不同；正确性探针需人工标注，限制低资源场景。

---

## 490. Beyond a Single Signal: SPECTREG2, A Unified MultiExpert Anomaly Detector for Unknown Unknowns

**arXiv ID:** 2603.21160 | [PDF](https://arxiv.org/pdf/2603.21160v1)

**作者:** Rahul D Ray `[一作]` `[通讯]` (BITS Pilani), Rahul D Ray (BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种多信号融合的异常检测框架 SPECTRE-G2，结合双骨干网络提取的八种互补信号，对未知未知（structural anomaly）进行识别。

**💡 创新点**

创新点包括：1) 双骨干（谱正则化 Gaussianizer + 普通 MLP）实现密度与几何信息双重捕获；2) 通过谱正则化实现特征高斯化，提升密度信号的敏感度；3) 将置信度、能量、互信息、ODIN、USD 等多种不确定性信号统一归一化并自适应 Top‑k 选择，避免单一信号误判；4) 引入基于因果残差的信号，在表格数据上直接量化结构一致性。

**🔧 技术方法**

技术手段包括：Spectral Normalization、Gaussianization 约束、对抗式伪 OOD 生成、八种信号计算（GaussScore、FtMahaP、InMaha、Energy、Entropy、MI、ODIN、USD、Causal Residual）、信号归一化、方向校正、基于验证 AUROC 的 Top‑k 自适应融合。

**📊 数据集**

使用四个多样化数据集：Synthetic Causal、Adult（UCI）、CIFAR‑10（图像）以及 Gridworld（强化学习）进行实验，每个数据集构造多种结构异常（confounder、new variable、mechanism、interaction 等）。

**📈 对比分析**

与 12 种基线（Deep Ensemble、MC Dropout、Bayesian NN、DUQ、Evidential DL、RBF、Conformal、UTraCE、Mahalanobis、ODIN、USD 等）在 AUROC、AUPR、FPR95、Confident Error Rate 等指标上比较，SPECTRE‑G2 在 11/12 任务中取得最高 AUROC，AUPR 10/12，FPR95 9/12，表现显著优于所有基线。

**⚠️ 局限性**

局限性：1) 在 Synthetic mechanism 任务上略逊于 Mahalanobis；2) Adult 数据集整体表现接近随机，说明对真实世界微妙结构变化敏感度有限；3) Causal 信号仅适用于维度≤30 的表格数据；4) 伪 OOD 采样仅基于高斯噪声，可能无法覆盖所有 OOD 情形；5) 计算成本高于单信号方法，需训练 5 个 Gaussianizer + 1 MLP 并提取 8 条信号。

---

## 491. How Short Is Too Short? Power Analysis for BIC-Based Changepoint Detection in Ecological Monitorin

**arXiv ID:** 2603.21154 | [PDF](https://arxiv.org/pdf/2603.21154v1)

**作者:** Ang A. Li `[一作]` (Peking University), Ang A. Li `[通讯]` (Peking University)

**通讯引用:** 9279 | [OpenAlex ID](https://openalex.org/A5100413631)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对短序列生态监测中的分段检验进行模拟功效分析，比较Binseg-BIC与PELT方法，并与早期警报信号做对比，同时在珊瑚礁和沙漠啮齿动物数据上验证。

**💡 创新点**

首次在10-50观测长度下系统评估BIC二分分段的功效，并给出效应量阈值热图；发现PELT在自相关数据下更稳健，并揭示EWS与分段检验的效应交叉点。

**🔧 技术方法**

使用模拟生成带已知变点的高斯噪声时间序列，BIC选择的Binary Segmentation，PELT罚项估计，AR(1)自相关测试，Kendall τ趋势检测，以及置换检验。

**📊 数据集**

莫阿拉珊瑚礁LTER（n=18）和Portal Project沙漠啮齿动物社区（n=49）。

**📈 对比分析**

将BIC-Binseg与PELT、EWS的变化率检测进行对比；BIC-Binseg在n≥30、效应量≥2时80%功效，效应量≥5时检测2-3变点；PELT在相同条件下可达85-91%功效，且对φ≤0.6的自相关鲁棒；EWS对效应量不敏感，约73%误报。

**⚠️ 局限性**

仅考虑高斯噪声、均匀变点、常数均值，未包含趋势、季节性或非高斯误差；样本大小小，i.i.d.功效为乐观上限；需要更复杂模型验证。

---

## 492. Learning from Label Proportions with Dual-proportion Constraints

**arXiv ID:** 2603.21153 | [PDF](https://arxiv.org/pdf/2603.21153v1)

**作者:** Tianhao Ma `[一作]` (University of Tokyo), Renchu Guan `[通讯]` (Jilin University)

**通讯引用:** 2717 | [OpenAlex ID](https://openalex.org/A5007914848)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种新型弱监督学习框架LLP‑DC，在学习标签比例（label proportions）任务中同时对包（bag）级和实例级别施加比例一致性约束；

**💡 创新点**

创新点在于将实例级伪标签生成视为最小费用最大流问题，能够高效得到严格满足比例约束的硬伪标签；并通过双重比例约束（bag层和实例层）提升模型鲁棒性；

**🔧 技术方法**

技术核心包括：最小费用最大流算法（min‑cost‑max‑flow）用于伪标签生成；弱强数据增强策略；基于softmax的分类器；交叉熵作为损失；训练中bag‑loss与instance‑loss的加权组合；

**📊 数据集**

实验使用的公开基准数据集有：CIFAR‑10、CIFAR‑100、SVHN、Fashion‑MNIST、mini‑ImageNet；通过随机打乱并划分不同大小（16/32/64/128）的包构造LLP数据；

**📈 对比分析**

对比方法包括DLLP、LLP‑VAT、ROT、SoftMatch、FLMm、L²p‑ahil以及完全监督设置；实验表明LLP‑DC在所有数据集与包大小上均优于或相当于现有方法，尤其在CIFAR‑100和mini‑ImageNet上提升明显；运行时间与主流方法相近，略优于LLP‑VAT；

**⚠️ 局限性**

局限性：1) 目前仅在合成LLP数据集上验证，缺乏真实隐私约束下的工业级数据；2) 伪标签生成需要求解最大流，尽管复杂度为O(m²l²)，但在极大包或多类场景下仍可能产生一定开销；3) 对阈值τ和权重λ仍需经验调参，虽然表现鲁棒但未实现完全无监督调优；4) 只考虑固定包大小，需进一步扩展到变长包的场景。

---

## 493. On the Role of Batch Size in Stochastic Conditional Gradient Methods

**arXiv ID:** 2603.21191 | [PDF](https://arxiv.org/pdf/2603.21191v1)

**作者:** Rustem Islamov `[一作]` (University of Basel), Volkan Cevher `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了批大小在基于 μ‑Kurdyka–Łojasiewicz 条件下的随机条件梯度（SCG）方法中的作用，并提出了基于令牌预算的批大小、序列长度与步长的自适应调度规则。

**💡 创新点**

创新点在于：①提供了在 μ‑KL 条件下的收敛分析，显式捕捉批大小、步长与噪声的相互影响；②揭示了批大小随令牌预算的非单调最佳规模（BST 规则 BS ≈ T^{2/3}）；③给出理论指导的批大小与步长的自适应更新策略，并在大规模 LLM 训练中验证其有效性。

**🔧 技术方法**

使用了随机条件梯度（SCG）/Frank‑Wolfe 相关算法（如 Scion）、动量技术、μ‑Kurdyka–Łojasiewicz 误差界、令牌预算视角以及批大小与序列长度的自适应调度。

**📊 数据集**

使用 FineWeb 数据集训练 NanoGPT 模型（124M 与 1B 参数量）进行实验验证。

**📈 对比分析**

通过与 μP 框架（保持批大小、序列长度和步长不变）以及不同固定或自适应批大小/序列长度的基线进行对比，实验表明在满足 BST 规则的条件下，批大小与序列长度的自适应提升了令牌效率，验证误差下降到与理论预期一致，优于 μP 基线。

**⚠️ 局限性**

局限性包括：①其余超参数（如半径 η、方差初始化）未针对不同模型架构进行微调；②对动量参数的迁移性理解不足；③仅在有限的数据点上拟合了常数的幂律关系，可能影响预测精度。

---

## 494. Security and Privacy in O-RAN for 6G: A Comprehensive Review of Threats and Mitigation Approaches

**arXiv ID:** 2603.21211 | [PDF](https://arxiv.org/pdf/2603.21211v1)

**作者:** Lujia Liang `[一作]` (University of Glasgow), Lei Zhang `[通讯]` (University of Glasgow)

**通讯引用:** 23257 | [OpenAlex ID](https://openalex.org/A5100433837)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对O‑RAN在6G环境下的安全与隐私威胁进行了系统性综述，提出了按STRIDE和功能模块划分的威胁模型，并评估了现有防护措施及标准化进展。

**💡 创新点**

首次将传统O‑RAN威胁与即将出现的6G特定风险（如非地球网络、量子通信、AI/ML攻击）结合，构建了完整的风险矩阵；同时提出了多维度评估框架和差距分析。

**🔧 技术方法**

采用STRIDE分类、风险覆盖矩阵、比较表格、标准化文献与报告分析，结合O‑RAN Alliance、3GPP、ITU等标准及安全报告进行综合评估。

**📊 数据集**

未使用实验数据集，而是基于公开安全报告（BSI、NIS、CISA、NTIA、O‑RAN Alliance等）和标准文档进行文献综述。

**📈 对比分析**

通过与已有安全综述和报告进行对比，使用覆盖度打分（0–2）评估报告完整性，指出本综述在6G视角、隐私与AI风险识别等方面的覆盖更全面；没有传统的实验性能指标。

**⚠️ 局限性**

局限性包括：1）主要基于文献与标准分析，缺乏实验验证；2）对6G技术细节和实时性能影响的量化评估不足；3）未对新型AI/ML攻击机制进行深度模型实验；4）安全对策的可实现性与成本未进行实测。

---

## 495. Pretrained Video Models as Differentiable Physics Simulators for Urban Wind Flows

**arXiv ID:** 2603.21210 | [PDF](https://arxiv.org/pdf/2603.21210v1)

**作者:** Janne Perini `[一作]` (ETH Zurich), Bernd Bickel `[通讯]` (ETH Zurich)

**通讯引用:** 5643 | [OpenAlex ID](https://openalex.org/A5000482372)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将预训练的视频扩散模型Fine-tune为快速、可微分的城市风场预测代理，并利用其进行建筑布局的逆向优化

**💡 创新点**

创新点在于将自然视频预训练的Latent Video Diffusion Transformer (LTX‑Video)迁移到物理仿真任务，通过VAE适配、标量条件嵌入和物理信息解码损失，实现比专用神经算子更高的预测精度；同时该代理可在不到一秒的推理时间内生成112帧完整风场，并可直接用于梯度下降式逆向设计

**🔧 技术方法**

主要技术包括：Transformer-based Latent Video Diffusion（LTX‑Video）、VAE适配（颜色适配器、解码器微调与物理信息损失）、标量条件编码、Classifier‑Free Guidance、软光栅化可微渲染、梯度反向优化等

**📊 数据集**

使用由不可压缩Euler求解器生成的13,000个2D CFD模拟数据（10,000训练/1,000验证/2,000测试），每个模拟包含112帧256×256分辨率的速度场，且覆盖多样的城市布局、风速、域尺寸与风向

**📈 对比分析**

与六类神经算子（U‑Net、Poseidon、AFNO、FNO、OFormer、RNO）在VRMSE、MAE、MRE、频谱误差和Wasserstein‑1距离等指标上对比，WinDiNet在VAE解码器微调+物理损失的配置下，在VRMSE上比最佳基线RNO低约7.6%，在MAE上低15%，推理速度约快1000倍；逆向优化实验表明可显著降低危险风速比例并改善舒适度

**⚠️ 局限性**

局限性主要体现在：仅为2D模型，缺乏高度信息；使用合成数据，真实城市结构与多尺度特征可能不足；对更复杂风场（多风向、气候条件）需要进一步验证；目前的建筑变形参数受限于子块分块方式，形状探索空间有限

---

## 496. Boundary-Aware Instance Segmentation in Microscopy Imaging

**arXiv ID:** 2603.21206 | [PDF](https://arxiv.org/pdf/2603.21206v1)

**作者:** Thomas Mendelson `[一作]`, Tammy Riklin-Raviv `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出无提示的边界感知实例分割框架，利用连续Signed Distance Function（SDF）预测细粒度细胞边界并实现相邻细胞的自动分离。

**💡 创新点**

创新点在于将可学习的Sigmoid映射与Modified Hausdorff Distance（MHD）损失相结合，直接在网络中优化边界对齐与实例分离，无需后处理或手工提示。

**🔧 技术方法**

使用U‑Net骨干网络，配合可学习Sigmoid、双向MHD、LSE及交叉熵等联合损失，预测连续SDF并生成概率掩模。

**📊 数据集**

在公开的 Cell Segmentation Benchmark（Fluo‑N2DH‑SIM+、Fluo‑N2DH‑HeLa、Fluo‑N2DH‑GOWT1、Fluo‑C2DL‑MSC）以及私有的3D MCF7 spheroid 数据集上进行评估。

**📈 对比分析**

与 UNet、SAM、μSAM、Cellpose‑SAM 等基线对比，模型在 SEG 指标下在所有数据集上取得最佳或第二佳表现。

**⚠️ 局限性**

局限性包括对极稀疏或高噪声图像的鲁棒性不足，且目前仅处理2D图像，未考虑时序一致性与3D+时间维度的直接扩展。

---

## 497. Rethinking Plasticity in Deep Reinforcement Learning

**arXiv ID:** 2603.21173 | [PDF](https://arxiv.org/pdf/2603.21173v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 498. Emergent Formal Verification: How an Autonomous AI Ecosystem Independently Discovered SMT-Based Safety Across Six Domains

**arXiv ID:** 2603.21149 | [PDF](https://arxiv.org/pdf/2603.21149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 499. LLM-based Automated Architecture View Generation: Where Are We Now?

**arXiv ID:** 2603.21178 | [PDF](https://arxiv.org/pdf/2603.21178v1)

**作者:** Miryala Sathvika `[一作]` (IIIT Hyderabad), Karthik Vaidhyanathan `[通讯]` (IIIT Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对340个开源项目进行实验，评估LLM和代理式方法（包括提示、Claude Code和自定义ArchView）在从源代码自动生成软件架构视图的能力。

**💡 创新点**

首次系统性地对LLM和代理式方案在高层架构视图生成上的表现进行量化评估，并提出专门针对架构领域知识的自定义Agent ArchView，显著提升了视图的可读性和细节匹配度。

**🔧 技术方法**

使用多种提示技术（零/一/少量示例）与LLM、通用编程代理Claude Code，以及专用多代理框架ArchView，并结合LLM-as-a-Judge、SSIM等自动指标和人工评估。

**📊 数据集**

利用Migliorini等人整理的15,000+视图的340个手工标注仓库（覆盖多种建模符号、关注点、质量属性、粒度），作为基准数据集。

**📈 对比分析**

通过比较5种配置（零/一/少量示例提示、通用代理、ArchView）在Clarity、Completeness、Consistency、Accuracy、Level of Detail以及SSIM上的评分，发现ArchView在Clarity与Level of Detail上最优（约22%/50%失败率），但整体仍低于人类标注；提示技术提升有限，通用代理效果最差。

**⚠️ 局限性**

主要局限在于：生成的视图仍以代码级细粒度为主，缺乏真正的架构抽象；摘要阶段的精度瓶颈影响后续质量；LLM-as-a-Judge和图像相似度等指标尚不能完全衡量架构语义；实验仅覆盖了特定LLM/代理，未涵盖全部可能的模型；数据集可能存在训练泄漏风险。

---

## 500. Reward Sharpness-Aware Fine-Tuning for Diffusion Models

**arXiv ID:** 2603.21175 | [PDF](https://arxiv.org/pdf/2603.21175v1)

**作者:** Kwanyoung Kim `[一作]` (GIST), Byeongsu Sim `[通讯]` (Samsung Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出奖励锐度感知微调（RSA‑FT）方法，利用奖励模型梯度的平坦化来抑制扩散式奖励强化学习中的奖励黑客问题。

**💡 创新点**

将输入空间和参数空间的奖励平滑结合，采用随机扰动与Sharpness‑Aware Minimization思想，在不重新训练奖励模型的前提下实现奖励模型的平坦化，显著降低奖励黑客。

**🔧 技术方法**

随机扰动、随机平滑、SAM（Sharpness‑Aware Minimization）、奖励模型梯度平滑、对比实验、奖励黑客分析等技术。

**📊 数据集**

使用DrawBench、HPSv2测试集，以及基于HPSv2、PickScore、ImageReward的评估指标，在SD1.5、SDXL、SD3、Flux1.dev等扩散模型上进行实验。

**📈 对比分析**

与ReFL、DRaFT‑K、AlignProp、DRTune等现有RDRL框架对比，在SD1.5、SDXL、SD3等模型上，RSA‑FT在HPSv2、PickScore、ImageReward等指标上均实现显著提升（提升幅度从约1分到10+分），有效抑制奖励黑客。

**⚠️ 局限性**

评估主要依赖模型奖励指标，人工评估样本有限；仅针对单一奖励模型，未完全验证多奖励设置；对扰动规模选择仍需进一步探究；缺乏对理论收敛性质的深入分析。

---

## 501. Can LLMs Fool Graph Learning? Exploring Universal Adversarial Attacks on Text-Attributed Graphs

**arXiv ID:** 2603.21155 | [PDF](https://arxiv.org/pdf/2603.21155v1)

**作者:** Zihui Chen `[一作]`, Dalin Zhang `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 4775 | [OpenAlex ID](https://openalex.org/A5101753289)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种面向文本属性图（TAG）的通用对抗攻击框架BadGraph，能够在黑盒环境下对节点拓扑和文本语义同时进行扰动，导致GNN和LLM推理模型的预测显著失效。

**💡 创新点**

创新点在于：①使用大型语言模型（LLM）作为攻击主体，利用其图知识推理能力生成跨模态（结构+文本）攻击；②引入目标影响者检索模块将搜索空间限制在高质量节点集合，显著降低LLM查询成本；③提出跨模态捷径理论和同质性保持上界，解释攻击既高效又隐蔽；④实现了跨架构（GNN与LLM推理器）可迁移的无梯度攻击。

**🔧 技术方法**

核心技术包括图编码器（如GCN/GAT）用于生成节点嵌入；检索模块（基于余弦距离）选取与目标节点语义相距较远的影响者；LLM交互式提示生成拓扑改动（边删除/新增）和文本改动（关键词替换）；利用预算约束实现一次性双重扰动。

**📊 数据集**

实验使用三大公开TAG数据集：Cora、OGBN-Products、OGBN-Arxiv；在不同文本特征（TF‑IDF、SBERT、TAPE）和不同后端模型（R‑GCN、GIN、GraphSAGE、TAGCN、GCN；LLM推理器如DeepSeek‑V3、Mistral‑7B）上评估。

**📈 对比分析**

与七个主流图攻击基线（RND、FLIP、STACK、PGD、NetTack、SGAttack、WTGIA）对比，BadGraph在所有目标模型上实现最高攻击成功率；例如在Cora上，BadGraph将准确率从≈89%降低至≈13%（约76%下降），在OGBN‑Arxiv和OGBN‑Products分别降低≈52%和≈30%；同时保持全局同质性不变，攻击隐蔽性强。

**⚠️ 局限性**

局限性包括：①依赖可调用的LLM接口，API费用和调用限制可能影响大规模攻击；②攻击预算仍受限，过大预算可能破坏图结构；③对非文本属性或多模态图的适用性尚未验证；④对LLM内部机制的解释仍不完全，攻击可解释性有待进一步提升。

---

## 502. Reframing Long-Tailed Learning via Loss Landscape Geometry

**arXiv ID:** 2603.21217 | [PDF](https://arxiv.org/pdf/2603.21217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 503. Affordance-Guided Enveloping Grasp Demonstration Toward Non-destructive Disassembly of Pinch-Infeasible Mating Parts

**arXiv ID:** 2603.21143 | [PDF](https://arxiv.org/pdf/2603.21143v1)

**作者:** Masaki Tsutsumi `[一作]` (University of Osaka), Kensuke Harada `[通讯]` (University of Osaka)

**通讯引用:** 11042 | [OpenAlex ID](https://openalex.org/A5016270703)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于物理仿真预生成包覆式抓取Affordance模板并通过视觉增强指导人机协作的半自主拆卸框架，针对无法用捏抓抓取的紧密配合部件实现无损拆卸。

**💡 创新点**

创新点在于将多指包覆抓取的Affordance模板自动生成与颜色编码可视化相结合，配合手掌/手指的实时跟踪映射至underactuated手，显著降低操作员认知负荷。

**🔧 技术方法**

使用物理仿真求解包覆抓取可行性、基于凸包的质量评估、颜色梯度可视化、Neuron Studio手套与MediaPipe腕部跟踪、7-DOF机械臂与5ST手的运动映射。

**📊 数据集**

实验使用三种微波炉内部紧密配合的部件（A、B、C）作为测试对象，无需公开数据集，全部为实物 CAD 模型。

**📈 对比分析**

在仿真中生成多组 Affordance 模板，实测拆卸成功率为高质量 90%、中等质量 100%、低质量 80%，表明视觉辅助可显著提升抓取成功率。

**⚠️ 局限性**

局限包括仿真与真实物理差异（摩擦、刚体动力学）导致抓取不稳定、underactuated 手模型难以精确预测、实验重量受限、缺乏触觉反馈等。

---

## 504. Dynamic Control Barrier Function Regulation with Vision-Language Models for Safe, Adaptive, and Realtime Visual Navigation

**arXiv ID:** 2603.21142 | [PDF](https://arxiv.org/pdf/2603.21142v1)

**作者:** Jeffrey Chen `[一作]` (University of Virginia), Rohan Chandra `[通讯]` (University of Virginia)

**通讯引用:** 1145 | [OpenAlex ID](https://openalex.org/A5016704715)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于VLM风险估计的自适应CBF安全滤波导航框架，能够在实时RGB输入下动态调整控制保守程度。

**💡 创新点**

将视觉语义风险与控制安全参数关联，并设计异步推理、时延感知融合与几何动态阈值，兼顾实时性与安全性。

**🔧 技术方法**

使用Vision‑Language Model（LLaVA‑1.5 7B）进行风险估计，Control Barrier Functions（CBF）进行安全过滤，异步HTTP调用、模型量化、速度感知动态阈值等技术。

**📊 数据集**

在NVIDIA IsaacSim仿真环境中，使用Carter差速驱动机器人、RGB相机，构建静态障碍、杂乱场景、动态障碍交叉、动态前方障碍等四类场景，并在真实仓库与toy环境中评测。

**📈 对比分析**

与固定α、无阈值、α_min/α_max等基线对比，实验表明在8个场景下成功率100%，无碰撞；相对固定α改进路径长度与到达时间最高可达18.5%，并且鲁棒性提升，碰撞率下降。

**⚠️ 局限性**

局限于仿真评测，缺乏真实机器人硬件验证；VLM风险估计的解释性与校准尚待进一步研究；在极端视觉噪声或极慢VLM响应时仍可能出现保守过度。

---

## 505. When Convenience Becomes Risk: A Semantic View of Under-Specification in Host-Acting Agents

**arXiv ID:** 2603.21231 | [PDF](https://arxiv.org/pdf/2603.21231v1)

**作者:** Di Lu `[一作]` (Xidian University), Jianfeng Ma `[通讯]` (Xidian University)

**通讯引用:** 21349 | [OpenAlex ID](https://openalex.org/A5012016098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了主机操作型代理（host-acting agents, HAA）在接受仅目标导向指令时产生的语义未充分指定风险，并通过对 OpenClaw 的轨迹分析提出了相应的威胁模型和风险分类。

**💡 创新点**

创新点在于：①将语义未充分指定视为独立安全风险来源，构建了面向 HAA 的语义威胁模型；②提出了六大风险类别的分类体系；③通过 OpenClaw 真实部署的执行轨迹展示了语义完成如何导致安全偏差；④给出了基于语义完成的防御设计原则。

**🔧 技术方法**

使用的技术主要包括：自然语言目标解析与任务分解、计划生成与工具调用的典型 HAA 工作流；语义完成与安全偏差的理论分析；轨迹收集与定性分析方法；以及针对风险模式的防御设计框架。

**📊 数据集**

数据集方面主要采用 OpenClaw 运行时产生的执行轨迹，包含对环境搭建、服务公开和故障修复等目标的多次 dry‑run 记录；并在项目局部 fixture 下收集局部化与全局化两类轨迹。

**📈 对比分析**

由于本文聚焦于概念验证与安全风险阐述，未给出量化性能对比；但通过对比不同目标细化程度下生成的计划，展示了安全成本（权限提升、持久修改、外部暴露等）的差异，证明了语义完成的安全偏差。

**⚠️ 局限性**

局限性包括：①实验仅基于 OpenClaw 单一容器化部署，缺乏跨平台或不同模型的广泛验证；②轨迹分析为定性研究，未进行大规模基准评估；③只关注语义未充分指定，未覆盖注入攻击、工具误用等其他安全威胁；④提出的防御原则仍需在实际系统中进一步实现与评估。

---

## 506. JANUS: A Lightweight Framework for Jailbreaking Text-to-Image Models via Distribution Optimization

**arXiv ID:** 2603.21208 | [PDF](https://arxiv.org/pdf/2603.21208v1)

**作者:** Haolun Zheng `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35399 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个轻量级、两阶段的黑盒 jailbreak 框架，能够有效突破文本和图像安全过滤器，生成与目标 NSFW 语义相符的违规图像。

**💡 创新点**

创新点在于将 jailbreak 视为分布优化任务，构造两个语义锚定的分布（NSFW 与清洁），通过可学习的混合比例 α 在黑盒环境下用 RL 优化分布，使得不需要大规模 LLM 或白盒梯度即可实现高效攻击。

**🔧 技术方法**

使用技术包括：基于词向量的 Dirac‑style 软化分布、双高斯混合模型、Gumbel‑Softmax 进行离散化、基于 RL（policy gradient）对混合比例进行黑盒优化、能量函数 E( p ) = -C(p,M(p))·S(M(p))，以及 CLIP 与 NSFW 打分器评估。

**📊 数据集**

使用了 200 条人类手工标注的 NSFW 提示词来自 Civitai‑8m‑prompts 数据集；在 Stable Diffusion XL、Stable Diffusion 3.5 Large Turbo、DALL·E3 和 Midjourney 等模型上进行评估。

**📈 对比分析**

与 MMA、MMP、QFA、PGJ、SneakyPrompt 等基线方法对比，在 SD3.5LT 上实现 TASR 94.25%、IASR‑8 46.65%、ASR‑8 43.15% 等指标远超对手；在 DALL·E3 上也获得最高 TASR 与 IASR‑8，整体表现优于现有所有方法。

**⚠️ 局限性**

局限性：仅在黑盒设置下实验，依赖于安全过滤器的可观测反馈；对更强安全管控或多模态模型的鲁棒性尚未验证；对真实系统的可部署性与伦理风险需进一步评估。

---

## 507. Ontology-Compliant Knowledge Graphs

**arXiv ID:** 2603.21188 | [PDF](https://arxiv.org/pdf/2603.21188v1)

**作者:** Zhangcheng Qiang `[一作]` (Australian National University), Zhangcheng Qiang `[通讯]` (Australian National University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5028214598)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了“Ontology‑Compliant KGs”框架，解决知识图谱与本体之间的兼容性问题，设计了内部兼容、跨KG兼容及基于模式的三阶段方法，并在建筑行业场景中进行验证。

**💡 创新点**

创新点包括：①双向兼容概念，将KG与本体的关系从单向转为双向；②新型词匹配算法（纯匹配、启发式、语义、拓扑）；③基于模式的兼容机制，支持本体碎片化与自动融合；④多准则（Liebig's law）评估与选择本体碎片。

**🔧 技术方法**

使用的技术有：图词匹配算法、图嵌入模型（DeepWalk、Node2Vec、Struc2Vec）、联合学习评估、Liebig's law多准则决策、基于模式的学习与优化。

**📊 数据集**

采用的公共数据集包括：Brick Schema、RealEstateCore、Project Haystack、BOT、SAREF、SSN/SOSA，以及对应的建筑领域KG实例。

**📈 对比分析**

通过匹配率、置信度、top‑k 搜索准确率进行对比。实验表明，采用兼容算法后，Top‑1/3 的准确率显著提升，达 90%–100%，优于未兼容嵌入的 0.03%–0.04% 等基准。

**⚠️ 局限性**

局限性：仅在建筑行业小规模KG上验证，缺乏大规模真实KG的可扩展性评估；匹配置信度随匹配级别提升而下降；方法对嵌入模型和学习算法的依赖较高。

---

## 508. Model Evolution Under Zeroth-Order Optimization: A Neural Tangent Kernel Perspective

**arXiv ID:** 2603.21169 | [PDF](https://arxiv.org/pdf/2603.21169v1)

**作者:** Chen Zhang `[一作]` (University of Hong Kong), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12280 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出神经零阶核（Neural Zeroth-order Kernel，NZK）并用其分析零阶优化（ZO）在函数空间中的训练动态，证明期望NZK不随时间变化，并给出闭式演化公式；

**💡 创新点**

创新点在于将NTK概念迁移到零阶优化，引入NZK揭示ZO梯度估计与参数空间无关、在期望上与FO等价；通过共享随机方向向量实现内存节省和收敛加速；

**🔧 技术方法**

主要技术包括：零阶梯度估计、期望NZK推导、线性模型与线性化神经网络的闭式动态推导、实验验证（使用随机向量采样、均方损失、批量估计）；

**📊 数据集**

使用的数据集：合成二维圆形数据、MNIST（未具体实验提及）、CIFAR‑10、Tiny ImageNet；

**📈 对比分析**

比较方法：FO（全微分）、传统参数梯度ZO、基于NZK的核梯度ZO；结果显示：传统ZO收敛慢，FO中等，NZK核梯度ZO收敛最快，实验中使用相同随机向量可进一步加速；

**⚠️ 局限性**

局限性：理论主要针对线性/线性化模型和平方损失；对非线性模型、非高斯噪声、非均方损失的推广尚未给出；实验仅在特定数据集与网络结构上验证，缺乏更广泛的实证；

---

## 509. Fast Nearest Neighbor Search for $\ell_p$ Metrics

**arXiv ID:** 2603.21148 | [PDF](https://arxiv.org/pdf/2603.21148v1)

**作者:** Robert Krauthgamer `[一作]` (Weizmann Institute of Science), Nir Petruschka `[通讯]` (Weizmann Institute of Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

为ℓ_p空间（p>2）设计了一种随机化近似最近邻搜索（ANN）数据结构，能够在O(dlogn)的查询时间内返回误差约为p^O(1)+loglog p的近似最近邻，空间复杂度为O(dn)。

**💡 创新点**

创新点在于将稀疏覆盖（sparse cover）与Mazur映射相结合，构造全局覆盖集合来避免为每个搜索半径单独构建数据结构，从而实现了在保持快查询时间的同时将空间压缩到线性级别；并通过递归嵌入实现近似因子从O(2^p)下降到多项式级别。

**🔧 技术方法**

主要技术包括：Mazur映射对ℓ_p→ℓ_q的稀疏化变形、β‑稀疏覆盖（sparse cover）构造、递归的ANN嵌入与放缩、概率放大与独立重复等。

**📊 数据集**

文中未使用具体实验数据集，所有结果均为理论分析；假设输入为任意大小为n的ℓ_p^d点集X。

**📈 对比分析**

与现有工作（Bartal & Gottlieb 2019、Krauthgamer‑Petruschka‑Sapir 2025等）相比，该结构在相同快查询时间下实现了更优的近似因子（p^O(1)+loglog p）且保持线性空间；在p≈2时可与O(1)近似的ℓ_2结构匹敌，p→∞时逼近ℓ_∞的O(loglog d)近似。

**⚠️ 局限性**

局限性在于近似因子仍为多项式p^O(1)+loglog p，尚未达到理论上最优的O(log p)；递归嵌入假设p≤log d，且对极大维度d或极大p时的常数因子尚未评估；此外实现细节对稀疏覆盖的构造时间及常数有一定影响。

---

## 510. NeSy-Edge: Neuro-Symbolic Trustworthy Self-Healing in the Computing Continuum

**arXiv ID:** 2603.21145 | [PDF](https://arxiv.org/pdf/2603.21145v1)

**作者:** Peihan Ye `[一作]` (Stockholm University), Praveen Kumar Donta `[通讯]` (Stockholm University)

**通讯引用:** 1902 | [OpenAlex ID](https://openalex.org/A5079303717)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了NeSy-Edge三层神经符号框架，实现可信自愈

**💡 创新点**

edge-first设计，三层（感知、推理、动作）结合符号先验与局部神经推理，显著降低云依赖

**🔧 技术方法**

轻量级日志解析（缓存+检索+小模型）、先验约束DYNOTEARS因果图、RAG+因果导航、LLM（本地Qwen3-0.6B与云DeepSeek-V3.2）

**📊 数据集**

Loghub HDFS、OpenStack、Hadoop三类日志数据集

**📈 对比分析**

与Drain、DirectSLM、DYNOTEARS、PC、Pearson、Vanilla LLM、RAG-only等基线比较，最高噪声水平下解析准确率≈87/93/91%，因果图稀疏≈33/39/41边，AvgRank≈1.00/1.74/1.28，RCA≈76%，E2E≈65%

**⚠️ 局限性**

仅基于日志，未融合多模态运行时信息；最终诊断仍依赖云端；缺乏在线自适应与持续学习机制

---

## 511. DS2SC-Agent: A Multi-Agent Automated Pipeline for Rapid Chiplet Model Generation

**arXiv ID:** 2603.21190 | [PDF](https://arxiv.org/pdf/2603.21190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 512. Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects

**arXiv ID:** 2603.21165 | [PDF](https://arxiv.org/pdf/2603.21165v1)

**作者:** Nurul Labib Sayeedi `[一作]` (United International University), Swakkhar Shatabda `[通讯]` (BRAC University)

**通讯引用:** 3205 | [OpenAlex ID](https://openalex.org/A5067504579)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 BanglaVerse， 一个覆盖9个文化领域、4种语言（孟加拉语、英语、印地语、乌尔都语）以及5种孟加拉方言（Barishal、Chittagong、Noakhali、Rangpur、Sylhet）的多模态评测基准。

**💡 创新点**

创新点在于：① 通过手工精选1,152张图像构建文化深度核心；② 采用受控自动翻译与方言生成管道，扩展至多语言多方言；③ 关注方言与历史语言对模型鲁棒性的影响，揭示文化知识是主要瓶颈。

**🔧 技术方法**

使用的技术包括：手工标注与跨验证；多语言翻译（Bing/自研模型）与方言翻译微调；Zero‑shot、Few‑shot、Chain‑of‑Thought (CoT) 提示；评价指标为 BERTScore‑F1、LLM‑as‑a‑Judge（Bard）以及 VQA 准确率。

**📊 数据集**

数据集：1,152张图像，手工注释（标题+VQA），经翻译扩展为 32,256 条任务实例，涵盖9个文化领域、4种标准语言和5种方言。

**📈 对比分析**

比较方法：在各语言/方言下进行 Zero‑shot / Few‑shot / CoT 评测，分别统计 VQA 准确率和 Caption 质量。实验显示：标准孟加拉语表现被高估，方言下 VQA 下降约 2–3%，Caption 下降 10–15%；历史关联语言（印地语、乌尔都语）在 Caption 上能部分保持文化意义，但整体仍低于标准孟加拉语；知识密集领域（媒体、政治）是主瓶颈。

**⚠️ 局限性**

限制：规模有限（仅1,152张图像）；方言生成依赖自动化翻译与后期人工校正；评价主要聚焦 VQA 与 Caption，未覆盖其他多模态任务；文化知识缺失仍是主要瓶颈，未来需进一步扩充和社区协作改进。

---

## 513. GIDE: Unlocking Diffusion LLMs for Precise Training-Free Image Editing

**arXiv ID:** 2603.21176 | [PDF](https://arxiv.org/pdf/2603.21176v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 514. Training-Free Instance-Aware 3D Scene Reconstruction and Diffusion-Based View Synthesis from Sparse Images

**arXiv ID:** 2603.21166 | [PDF](https://arxiv.org/pdf/2603.21166v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 515. Ontology-driven personalized information retrieval for XML documents

**arXiv ID:** 2603.21139 | [PDF](https://arxiv.org/pdf/2603.21139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 516. GAPG: Geometry Aware Push-Grasping Synergy for Goal-Oriented Manipulation in Clutter

**arXiv ID:** 2603.21195 | [PDF](https://arxiv.org/pdf/2603.21195v1)

**作者:** Lijingze Xiao `[一作]` (South China University of Technology), Yu Ren `[通讯]` (South China University of Technology)

**通讯引用:** 7747 | [OpenAlex ID](https://openalex.org/A5078152154)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了基于点云的抓取与推送协同框架，利用几何匹配评估抓取可行性，并规划推送动作以实现复杂场景中的目标抓取。

**💡 创新点**

创新点在于同时考虑抓取器自身与目标物体的三维几何关系，使用抓取评估模块的几何匹配结果为推送评估模块提供高质量监督，实现安全、高效的抓取‑推送协同。

**🔧 技术方法**

采用 GraspNet 采样抓取姿态，使用 PointNet++ 对抓取点云与闭合空间点云进行特征提取，并用多层感知机评估抓取可行性；推送模块将推送姿态转换为点并通过 PointNet++ 与一热编码结合，预测推送效果。

**📊 数据集**

在 PyBullet 仿真中随机装载 10‑30 件物体，使用 40,000 条抓取样本和 40,000 条推送样本；在真实场景中使用 Photoneo PhoXi 3D Scanner 与 ZED 2i 相机，并通过 SAM 生成目标物体点云。

**📈 对比分析**

与 Xu 等人的单抓取方法和 Efficient Push‑Grasping 基线比较，实验显示在 15/30 物体及挑战场景中任务完成率 100%、抓取成功率 95%+，平均动作数比基线低约 75%，在仿真和实测均表现更优。

**⚠️ 局限性**

局限在于对点云质量和相机标定高度依赖，且仅实现单一辅助动作（推送）与固定抓取器模型，未覆盖动态环境、多模态感知以及更复杂的交互策略。

---

## 517. Is Monitoring Enough? Strategic Agent Selection For Stealthy Attack in Multi-Agent Discussions

**arXiv ID:** 2603.21194 | [PDF](https://arxiv.org/pdf/2603.21194v1)

**作者:** Qiuchi Xiang `[一作]` (Lancaster University), Jun Liu `[通讯]` (Lancaster University)

**通讯引用:** 39013 | [OpenAlex ID](https://openalex.org/A5100361885)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究多代理讨论在讨论监控（即异常检测器持续拦截恶意消息）场景下的攻击，并提出一种针对该场景的对抗攻击方法，展示即使在持续监控下仍能显著破坏讨论结果。

**💡 创新点**

创新点包括：①首次正式定义讨论监控攻击场景并系统评估现有攻击在此场景下的失效；②基于 Friedkin‑Johnsen（FJ）社会意见动力学模型，提出对抗友好意见动力学改造，显式建模僵化广播与目标受影响；③将攻击目标建模为堆栈式领导‑跟随离散优化问题，利用 Taylor 展开在对抗约束下将跟随子问题变为可闭式求解；④提出快速求解策略（枚举领导者+闭式跟随解），实现秒级求解；⑤在讨论监控场景下实现 30–50% 的攻击成功率，证明监控并非全能防御。

**🔧 技术方法**

技术手段：LLM 基础模型 GPT‑4o；异常检测器 SelfCheckGPT、RAGAs、G‑Safeguard；Friedkin‑Johnsen 意见动力学模型与对抗改造；堆栈式领导‑跟随离散优化；Taylor 展开与闭式解法；基于 LLM 的内在意见、僵化度与影响矩阵估计；实验使用 MMLU 与 MMMU 数据集。

**📊 数据集**

使用的数据集：MMLU（多领域知识问答）和 MMMU（多模态问答），用于评估在文本+图像交互下的攻击效果。

**📈 对比分析**

实验对比：与原始 MCA 与 MAD‑Spear 的基线策略以及六种手工设计的对抗变体。结果显示，传统攻击在讨论监控下检测率 > 93%，成功率 < 8%；在加入对抗友好模型并使用层级优化后，攻击成功率提升至 30–50%，ΔAcc 与 ΔAgr_adv 显著增大，而检测率维持在 4–6%。表明新方法在监控约束下仍保持强攻击力，并优于所有对照组。

**⚠️ 局限性**

局限性：①对 FJ 参数的估计依赖 LLM 评估，可能受模型偏差影响；②假设异常检测器仅基于文本/图像，未考虑更强的语义或多模态检测；③模型适用于相对均匀的多代理网络，复杂拓扑或异构代理的效果未知；④对抗强度超参 p 需要经验调优；⑤实验规模有限，未在真实部署环境或更大代理数下验证可扩展性。

---

## 518. A Parametric, Geometry-Aware Residential Construction Cost Estimation Model for Ghana: Design, Validation, and the "Completeness Gap" in Informal Contractor Quotes

**arXiv ID:** 2603.21314 | [PDF](https://arxiv.org/pdf/2603.21314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 519. ORACLE: Optimizing Reasoning Abilities of Large Language Models via Constraint-Led Synthetic Data Elicitation

**arXiv ID:** 2603.21140 | [PDF](https://arxiv.org/pdf/2603.21140v1)

**作者:** Zhuojie Yang `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2296 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结构化的多步推理数据生成框架ORACLE，通过模板化提示让LLM生成可解析的推理步骤，并使用符号推理引擎对每一步进行验证，从而得到高质量、可验证的训练样本，提升LLM的推理性能。

**💡 创新点**

创新点在于将符号推理与LLM生成结合，采用固定格式模板实现细粒度步骤验证；引入beam search与符号验证相结合的生成策略；并通过直接偏好优化（DPO）进一步强化模型对合理推理路径的偏好。

**🔧 技术方法**

使用技术包括：LLM（LLaMA-3.1、Mistral-7B、Qwen-2.5）、符号推理引擎Pyke、固定格式模板提示、beam search搜索、SFT（监督微调）和DPO（直接偏好优化）。

**📊 数据集**

在六个不同类型的推理数据集上评估：ProntoQA、ProofWriter、BoolQ、CosmosQA、ScienceQA、StrategyQA。

**📈 对比分析**

与CoT、RFT、ToT‑SFT、Self‑rewarding等基线对比，ORACLE在所有模型和数据集上均实现了最优或接近最优的准确率，提升幅度从约0.9%到6.1%，在逻辑、事实、常识及多步科学推理任务上表现尤为突出。

**⚠️ 局限性**

主要局限包括：符号推理引擎的执行成功率在常识性或非结构化推理任务中较低，导致部分步骤被标记为错误；LLM在将自然语言转换为符号形式时易出现语法或语义错误；以及对开放域、模糊情境的验证能力仍有限。

---

## 520. COINBench: Moving Beyond Individual Perspectives to Collective Intent Understanding

**arXiv ID:** 2603.21329 | [PDF](https://arxiv.org/pdf/2603.21329v1)

**作者:** Xiaozhe Li `[一作]` (Tongji University), Qingwen Liu `[通讯]` (Tongji University)

**通讯引用:** 3362 | [OpenAlex ID](https://openalex.org/A5059157106)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 COINBench，一个实时更新、针对消费领域的集合意图（Collective Intent）评测基准，并设计了主动探测评估范式。

**💡 创新点**

创新点在于将集合意图分层为五级认知树（CoIn-Tree），结合 LLM-as-the-Judge 与检索增强验证（CoIn-RAG）实现深度、广度、准确性和信息量多维评估。

**🔧 技术方法**

使用技术包括 LLM-as-the-Judge、CoIn-Tree 层级结构、检索增强验证（CoIn-RAG）、双嵌入检索、语义与规则过滤、主动探测的问答构造。

**📊 数据集**

使用了超过 200k 条真实消费讨论数据，涵盖 9 个主领域、54 个子领域、1,400+ 产品，构成大规模动态语料库。

**📈 对比分析**

通过与 20 个主流 LLM（包括 GPT‑o3、GPT‑5、Qwen‑3 等）对比，发现推理型模型在深度层和准确性上优于通用模型，但整体仍存在深度与广度的显著差距。

**⚠️ 局限性**

局限性包括仅聚焦消费领域、未涵盖最新 Gemini/Claude 等模型、对小模型的噪声容忍度不足，以及数据集与评测方法的可迁移性待验证。

---

## 521. More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech Detection

**arXiv ID:** 2603.21298 | [PDF](https://arxiv.org/pdf/2603.21298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 522. Focus on Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric Prompting

**arXiv ID:** 2603.21287 | [PDF](https://arxiv.org/pdf/2603.21287v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 523. When the Chain Breaks: Interactive Diagnosis of LLM Chain-of-Thought Reasoning Errors

**arXiv ID:** 2603.21286 | [PDF](https://arxiv.org/pdf/2603.21286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 524. KHMP: Frequency-Domain Kalman Refinement for High-Fidelity Human Motion Prediction

**arXiv ID:** 2603.21327 | [PDF](https://arxiv.org/pdf/2603.21327v1)

**作者:** Wenhan Wu `[一作]` (Yunnan University), Aidong Lu `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 13022 | [OpenAlex ID](https://openalex.org/A5037161857)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了KHMP框架，通过在预测后对高频DCT系数应用自适应Kalman滤波，并在训练时加入时间平滑与关节角度约束，以提升人类动作预测的时空一致性和物理可行性。

**💡 创新点**

①首次在频域将Kalman滤波作为频率索引的递归降噪器；②采用SNR自适应的噪声参数调节；③在VAE训练中加入时间平滑与关节角度约束。

**🔧 技术方法**

使用变分自编码器生成器、离散余弦变换与逆变换、基于SNR的自适应Kalman滤波器、物理约束损失（时间平滑、角度限制）等技术。

**📊 数据集**

在Human3.6M和HumanEva-I两个标准动作数据集上进行实验。

**📈 对比分析**

与ERD、DeLiGAN、BoM、DLow、DSF、GSPS、MOJO、DivSamp、STARS、MotionDiff、Belfusion、HumanMAC、TransFusion、CoMotion、SkeletonDiff、MotionMap、SOGM等最新方法在ADE、FDE、MMADE、MMFDE、APD等指标上进行对比，KHMP在HumanEva-I上取得ADE 0.188、FDE 0.204、MMADE 0.301，APD 7.481，显著优于其他方法。

**⚠️ 局限性**

仍依赖后处理Kalman滤波，计算开销略高；SNR估计对噪声水平敏感；仅在两大数据集上验证，未证明对更大规模或多模态数据的适用性；长期预测性能及多关节交互的细节仍待改进。

---

## 525. Direct Interval Propagation Methods using Neural-Network Surrogates for Uncertainty Quantification in Physical Systems Surrogate Model

**arXiv ID:** 2603.21308 | [PDF](https://arxiv.org/pdf/2603.21308v1)

**作者:** Ghifari Adam Faza `[一作]`, David Moens `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文介绍了 Elsevier 官方 LaTeX 文档类 elsarticle.cls 的功能、安装与使用方法。

**💡 创新点**

其创新之处在于基于 article.cls 重新实现，兼容 natbib 等主流包，减少包冲突，并提供多种期刊模式的预设格式。

**🔧 技术方法**

使用 natbib、geometry、graphicx、txfonts 等常用 LaTeX 包，并提供了自定义命令、环境（如 theorems、enumerate 等）来简化排版。

**📊 数据集**

本论文未使用任何数据集，纯粹是技术文档。

**📈 对比分析**

未涉及实验或方法比较，未给出性能指标。

**⚠️ 局限性**

局限性在于缺乏对不同期刊细节的全面说明，复杂公式排版仍需作者手动调整；文档示例较为繁琐。

---

## 526. Does Mechanistic Interpretability Transfer Across Data Modalities? A Cross-Domain Causal Circuit Analysis of Variational Autoencoders

**arXiv ID:** 2603.21236 | [PDF](https://arxiv.org/pdf/2603.21236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 527. F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting

**arXiv ID:** 2603.21304 | [PDF](https://arxiv.org/pdf/2603.21304v1)

**作者:** Injae Kim `[一作]` (KAIST), Hyunwoo J. Kim `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 F4Splat，一个前馈式 3D 高斯展开网络，利用预测的密度得分在单次前向传播中实现自适应密集化，生成稀疏但高质量的 3D 表示。

**💡 创新点**

创新点在于：①引入前馈式预测密度得分来指导 Gaussian 分配，避免传统均匀分配导致的冗余；②允许用户在不重新训练的情况下显式控制最终 Gaussian 预算；③通过空间复杂度与多视角重叠的自适应分配，实现更紧凑且质量更高的 3D 结构。

**🔧 技术方法**

核心技术包括 3D Gaussian splatting、前馈神经网络、密度得分预测、差分光栅化、VGG 主干网络、多分辨率特征图、MSE/LPIPS 损失以及相机和场景一致性约束。

**📊 数据集**

训练数据集：RealEstate10K、ACID；在这些大规模真实场景数据上进行无标定和姿态自由的多视角训练。

**📈 对比分析**

与无标定方法 VicaSplat、AnySplat 以及姿态自由方法 NoPoSplat、SPFSplat 进行比较；与像素级 Splat、MVSplat、DepthSplat、pixelNeRF 等在两视角评估中对比。结果表明，F4Splat 在相同或更少的 Gaussian 数量下，在 LPIPS、SSIM、PSNR 等指标上取得与或优于现有方法的性能，显著提升了 Gaussian 利用率。

**⚠️ 局限性**

局限性：依赖大规模预训练数据，难以处理极少视角或极端遮挡的场景；前馈密度预测受限于特征表达，可能在纹理复杂或几何不规则区域产生误分配；未评估对动态场景或实时更新的适应性。

---

## 528. Graph of States: Solving Abductive Tasks with Large Language Models

**arXiv ID:** 2603.21250 | [PDF](https://arxiv.org/pdf/2603.21250v1)

**作者:** Yu Luo `[一作]` (Nankai University), Dan Pei `[通讯]` (Tsinghua University)

**通讯引用:** 9732 | [OpenAlex ID](https://openalex.org/A5046419834)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通用的神经符号双层框架Graph of States，专为归纳推理任务设计，以显式状态控制提升推理质量。

**💡 创新点**

通过构造因果图记录信念状态并用状态机实现层次化推理与回溯，解决传统CoT/ToT在归纳推理中的证据伪造、上下文漂移、回溯失败与过早停止等问题。

**🔧 技术方法**

双层神经符号架构；因果图+状态机；多角色协作式代理系统；ReAct工具调用；基于LLM的判断；自定义阈值控制深度递归。

**📊 数据集**

医学诊断数据集DiagnosisArena（150例）和分布式系统故障数据集（150例）来自大型IT公司。

**📈 对比分析**

与八种基线（单/多代理+CoT/ToT/GoT/FoT）在LLM‑和Human‑as‑Judge下评估；医学诊断Match从24%提升至39.86%，Relevant从60%提升至78.99%；故障诊断Match从28%提升至70.67%，Relevant从84%提升至88%；同时成本显著低于最佳FoT基线。

**⚠️ 局限性**

依赖LLM推理质量，阈值需要手工调优；对超长推理序列仍可能出现资源瓶颈；对领域知识库和检索效果高度敏感；缺乏通用自动化评测标准。

---

## 529. Identity-Consistent Video Generation under Large Facial-Angle Variations

**arXiv ID:** 2603.21299 | [PDF](https://arxiv.org/pdf/2603.21299v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 530. CornOrb: A Multimodal Dataset of Orbscan Corneal Topography and Clinical Annotations for Keratoconus Detection

**arXiv ID:** 2603.21245 | [PDF](https://arxiv.org/pdf/2603.21245v1)

**作者:** Mohammed El Amine Lazouni `[一作]` (Abou Bakr Belkaid University), Mostafa El Habib Daho `[通讯]` (University of Western Brittany)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 CornOrb 数据集，包含 1,454 只眼的四幅标准化 Orbscan 3 角膜拓扑图和完整的结构化临床注释，旨在支持角膜锥体的 AI 研究。

**💡 创新点**

首个大规模公开的 African Orbscan 3 角膜拓扑数据集，融合图像与表格信息，实现多模态数据集成，为 AI 领域提供了前所未有的多维度资源。

**🔧 技术方法**

使用 Python 解析 PDF 报告提取并裁剪图像、标准化至 560 × 560 像素并保存为 PNG；通过脚本自动生成关联 CSV 注释文件；未涉及模型训练或深度学习技术。

**📊 数据集**

来自阿尔及利亚 Lazouni 诊所的 744 名患者（共 1,454 只眼），其中 889 只正常眼和 565 只角膜锥体眼，使用 Bausch & Lomb Orbscan 3 设备完成。

**📈 对比分析**

论文未提供模型训练或性能评估，仅提供数据集发布；未来可使用标准分类或多模态融合算法在此数据集上进行基准测试。

**⚠️ 局限性**

仅使用 Orbscan 3 设备，缺乏跨设备可推广性；单中心北非人群，种族和地理多样性有限；缺乏纵向随访和分期信息；存在中度类别不平衡（889 正常 vs 565 角膜锥体）。

---

## 531. Enhancing Brain Tumor Classification Using Vision Transformers with Colormap-Based Feature Representation on BRISC2025 Dataset

**arXiv ID:** 2603.21234 | [PDF](https://arxiv.org/pdf/2603.21234v1)

**作者:** Faisal Ahmed `[一作]` `[通讯]` (Embry-Riddle Aeronautical University), Faisal Ahmed (Embry-Riddle Aeronautical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并验证了一种基于视觉Transformer（ViT）并结合伪彩色增强的脑肿瘤多类别MRI分类框架。

**💡 创新点**

将colormap映射预处理与ViT自注意力机制结合，显著提升结构与强度分辨率，且无需数据增强即可实现高精度。

**🔧 技术方法**

使用Vision Transformer（ViT‑Base‑Patch16‑224）、伪彩色预处理、交叉熵损失、Adam优化、早停策略等技术。

**📊 数据集**

采用公开的BRISC2025 MRI数据集，涵盖胶质瘤、脑膜瘤、垂体瘤和非肿瘤四类。

**📈 对比分析**

与ResNet50、ResNet101、EfficientNetB2等CNN基线模型对比，使用准确率、精确率、召回率、F1分数和AUC评估；本方法准确率98.90%、AUC99.97%，在所有指标上均优于基线。

**⚠️ 局限性**

依赖大规模预训练Transformer导致计算资源消耗较大；未在外部数据集上验证，泛化性能尚需进一步评估。

---

## 532. Active Inference Agency Formalization, Metrics, and Convergence Assessments

**arXiv ID:** 2603.21319 | [PDF](https://arxiv.org/pdf/2603.21319v1)

**作者:** Eduard Kapelko `[一作]` `[通讯]`, Eduard Kapelko

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

论文提出了基于主动推理（Active Inference）的连续表示形式的代理（agency）定义，并给出了衡量代理性（agency）的距离度量（STARC）。

**💡 创新点**

创新点在于将好奇心（Curiosity）与赋权（Empowerment）两种内在动机统一为单一代理函数，并用数学形式（KL散度、互信息、凸光滑性）论证该函数在稀疏环境下的对数收敛速度，从而解释了大规模模型训练中代理性自发出现的高概率。

**🔧 技术方法**

核心技术包括：KL 散度和互信息的光滑性分析、对数收敛证明、Sobolev/Besov 光滑度框架、STARC 规范化奖励与距离度量。

**📊 数据集**

论文未使用具体实验数据集，主要采用理论推导和文献参考进行验证。

**📈 对比分析**

比较方法是计算待评估函数与理想代理函数在 STARC 规范化奖励空间中的距离，若距离越小则表明代理性越高。由于未给出实验结果，无法给出具体性能指标。

**⚠️ 局限性**

局限性包括：① 代理函数的定义与线性组合假设在实际模型中可能不成立；② 估计的对数收敛速度基于理想化的稀疏环境和无限参数空间；③ 仅提供理论分析，缺乏大规模实验验证；④ 在高维函数空间中子空间测度为零，实际测量难度较大。

---

## 533. Hardware Trojans from Invisible Inversions: On the Trojanizability of Standard Cell Libraries

**arXiv ID:** 2603.21294 | [PDF](https://arxiv.org/pdf/2603.21294v1)

**作者:** Kolja Dorschel `[一作]` (Max Planck Institute for Security and Privacy), Steffen Becker `[通讯]` (Max Planck Institute for Security and Privacy)

**通讯引用:** 256 | [OpenAlex ID](https://openalex.org/A5008222051)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对IEEE S&P 2023公开的四个工艺节点SEM图像数据集进行系统分析，提出了基于via点的定量相似度度量，用以评估标准单元库的Trojanizability，并利用此方法发现了可逆的“不可见反演”单元对，从而实现了隐蔽硬件木马的构造和检测。

**💡 创新点**

创新点在于：①提出了高效、可解释的via基相似度度量，①解决了传统模板匹配在高噪声/高节点下的局限；②首次引入Trojanizability概念，系统量化标准单元库对隐蔽木马的易受攻击程度；③揭示了“不可见反演”单元对的存在，并用其实现低成本、视觉不可检测的特权升级木马。

**🔧 技术方法**

采用持久性分析进行via提取，k‑means聚类生成单元代表，利用Jaccard距离计算via集合相似度，并通过对所有同宽功能不同单元对的全面比较得到相似度得分；随后将该度量与原始模板/via掩码匹配方法在同一数据集上对比。

**📊 数据集**

使用IEEE S&P 2023公开的四节点（90 nm、65 nm、40 nm、28 nm）硅片SEM图像与对应GDS II布局的公共数据集。

**📈 对比分析**

与原始基线（模板匹配＋via掩码匹配）对比，via‑position方法在所有节点实现了零误检率，且在90 nm、40 nm、28 nm节点显著降低误报率；在65 nm节点误报略高。该方法的检测准确率在大多数同宽功能不同单元对上均优于基线。

**⚠️ 局限性**

局限性包括：仅评估单元替换攻击，未考虑工艺变异或更高级的成像技术；对低分辨率或噪声极高的28 nm图像仍存在高误检；依赖于via可见性，若via被遮挡或失真，度量效果受限；仅针对公开的四个单元库，结果可能不完全泛化。

---

## 534. WARBENCH: A Comprehensive Benchmark for Evaluating LLMs in Military Decision-Making

**arXiv ID:** 2603.21280 | [PDF](https://arxiv.org/pdf/2603.21280v1)

**作者:** Zongjie Li `[一作]` (Hong Kong University of Science and Technology), Shuai Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 25296 | [OpenAlex ID](https://openalex.org/A5100328273)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构建一个包含136个后二战历史冲突场景的高保真数据集和四维评估框架，对多种大型语言模型进行军事决策和国际人道法合规性评估。

**💡 创新点**

创新点包括：①将真实历史冲突与多源法律注释结合，形成深度情境；②提出四维压测（基本能力、法律约束、边缘部署、信息退化、推理Chain-of-Thought）；③使用LLM-as-a-Judge结合专家制定的客观量表实现自动评估；④在同一情境下对多模型进行系统对比。

**🔧 技术方法**

使用的技术包括：大模型API（闭源和开源）、4位量化与8位/16位对比、边缘设备实验（RTX 4090手机版GPU）、信息遮蔽与冲突注入、Chain-of-Thought提示、三模型投票裁判器。

**📊 数据集**

数据集来源于Correlates of War、UCDP、ICRC等公开数据库，经过三到五小时多源核查与法律专家双重标注，随后去标识化后得到136个情景。

**📈 对比分析**

通过在九个模型（闭源API、开源大模型、边缘小模型）上进行多维度对比，发现闭源模型在决策质量与合规性上领先，但所有模型在法律合规、信息退化、4位量化下均表现不佳；边缘小模型在4位量化下几乎失效，且大多数模型在信息缺失或冲突信息时出现非线性性能崩溃。

**⚠️ 局限性**

局限性包括：样本量仅为136个，难以覆盖更细粒度的作战层级；实验硬件与真实军用边缘硬件差异；模型快速迭代导致实验结果仅在2026年初的快照；LLM裁判可能存在偏差，需进一步验证；未涵盖更复杂的战术指挥与多代理协同情景。

---

## 535. Conversation Tree Architecture: A Structured Framework for Context-Aware Multi-Branch LLM Conversations

**arXiv ID:** 2603.21278 | [PDF](https://arxiv.org/pdf/2603.21278v1)

**作者:** Pranav Hemanth `[一作]` (Pandemonium Research), Sampriti Saha `[通讯]` (Pandemonium Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Conversation Tree Architecture（CTA），一种将 LLM 对话结构化为树形的框架，通过上下文隔离和结构化的上下游流操作来缓解线性对话中的逻辑上下文污染。

**💡 创新点**

创新点包括：① 引入 volatile node（临时分支）与上下文流（下游传递、上游合并、跨节点传递）的正式定义；② 提出了插入位置、分块合并等开放式设计问题；③ 将对话管理与多代理系统自然映射为树形拓扑。

**🔧 技术方法**

采用 React + React Flow 构建树形可视化前端，使用 Groq / Gemini API 进行 LLM 推理，并通过手工实现的上下文窗口和流操作来实现 CTA 的核心功能。

**📊 数据集**

该工作为概念性框架，未使用公开数据集，原型演示基于人工构造的示例对话。

**📈 对比分析**

目前仅完成了原型验证，未进行量化评估；未来计划通过响应质量、任务完成率、用户感知连贯性和上下文效率等指标进行实证比较。

**⚠️ 局限性**

限制：原型仅实现基本的上下文隔离与手工上下游流，缺乏智能下游过滤、自动上游合并策略、跨节点传递以及分块插入等高级功能；此外缺少系统化的实验评估。

---

## 536. Estimating the Social Cost of Corporate Data Breaches

**arXiv ID:** 2603.21270 | [PDF](https://arxiv.org/pdf/2603.21270v1)

**作者:** Lina Alkarmi `[一作]` (University of Michigan), Mingyan Liu `[通讯]` (University of Michigan)

**通讯引用:** 11801 | [OpenAlex ID](https://openalex.org/A5101967011)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了13年身份盗窃调查（ITS）与数据泄露记录（PRC）纵向融合框架，量化单一大规模泄露事件对受害者的社会成本，并给出短期下限与长期上限估算。

**💡 创新点**

创新点在于将多源调查与泄露记录进行多阶段补全，提出“breach‑to‑victim”转换模型与动态饱和模型，实现对单个泄露事件短期与长期社会成本的系统量化。

**🔧 技术方法**

采用统计检验（Wilcoxon符号秩检验）、时间序列插值、线性回归、对数‑二次回归、折现累积记录模型以及插值求解等技术手段。

**📊 数据集**

使用了2008‑2021六波Identity Theft Supplement（ITS）数据以及Privacy Rights Clearinghouse（PRC）泄露记录数据库，并补充HHS及州检察官文件。

**📈 对比分析**

通过Wilcoxon检验验证泄露后身份盗窃显著上升，利用“breach‑to‑victim”模型与上限估算与公司和解金额对比，显示社会成本常远高于赔偿，体现监管缺口。

**⚠️ 局限性**

主要局限包括自报偏差与调查结构变动、PRC记录缺失与补充噪声、模型对不同敏感数据的权重未完全反映、未来年份数据缺口以及长期影响估算的时间边界限制。

---

## 537. From Natural Language to Executable Properties for Property-based Testing of Mobile Apps

**arXiv ID:** 2603.21263 | [PDF](https://arxiv.org/pdf/2603.21263v1)

**作者:** Yiheng Xiong `[一作]` (East China Normal University), Zhendong Su `[通讯]` (ETH Zurich)

**通讯引用:** 14366 | [OpenAlex ID](https://openalex.org/A5077610917)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种结构化属性合成方法，能将移动应用的自然语言属性描述自动转换为可执行的属性代码，降低了测试人员的手工编码工作量。

**💡 创新点**

创新点在于引入多模态大型语言模型进行 UI 语义地面化，生成丰富的 widget 上下文，并利用上下文学习的 LLM 生成框架特定的可执行属性，显著提升了属性生成的准确率。

**🔧 技术方法**

技术手段包括：多模态 LLM（如 GPT‑4o mini）进行 widget 功能注释，单模态 LLM（GPT‑4o、DeepSeek‑V3）进行属性合成，精心设计的少量示例提示、约束和框架 API 列表来实现 In‑Context Learning。

**📊 数据集**

实验数据集为 Kea 基准数据集中的 124 条真实历史缺陷的属性描述，并基于此生成 1,180 条多样化的自然语言变体以评估鲁棒性。

**📈 对比分析**

通过与手工编写属性对比，本工具在 GPT‑4o 与 DeepSeek‑V3 上分别达 95.2% 的准确率（118/124），鲁棒性实验中保持 87.6% 及 87.5% 的准确率；在用户研究中，生成属性的时间平均比手工写作节约 56%，并在准确率上略高。

**⚠️ 局限性**

主要局限包括：对 UI 语义的依赖仍可能出现误匹配；对属性描述的结构化要求可能限制自由表达；实验仅覆盖 8 款开源 Android 应用，未检验工业级应用的可迁移性；且模型对极端模糊或不完整描述的处理仍有限。

---

## 538. CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands

**arXiv ID:** 2603.21257 | [PDF](https://arxiv.org/pdf/2603.21257v1)

**作者:** Weiye Wang `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 14520 | [OpenAlex ID](https://openalex.org/A5039318240)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个名为 Calvo 的 LLM 推理引擎，通过将 KVCache 加载与 GPU 计算解耦为独立的异步阶段，并将 KVCache 加载视为首要阶段，显著提升网络密集型 LLM 推理的服务效率。

**💡 创新点**

主要创新点包括：① 为每个 KVCache 加载阶段配备独立的 dispatcher‑executor，实现多阶段流水线并行；② 在调度时将 KVCache 加载延迟作为服务成本的一部分，用二元线性模型预测成本，采用 SJF / LSTF 调度优化 TTFT 和 SLO。

**🔧 技术方法**

使用 vLLM + LMCache 作为基础框架，利用 Mooncake Store、RDMA 400Gbps 连接，ZeroMQ 进行进程间通信，采用 PCIe 并行加载、GPU 预分配策略，并结合二元线性成本模型与 SJF/LSTF 调度算法。

**📊 数据集**

长文本数据集包括 LooGLE、ICL 与 Code；模型为 Llama‑3.1‑8B‑Instruct 与 Qwen2.5‑14B‑Instruct‑1M；另外还使用多种缓存命中率（25%–100%）进行敏感性测试。

**📈 对比分析**

与基线 vLLM‑LMCache（以及其 FIFO 调度版本）在相同硬件与网络环境下比较，结果显示 Calvo 在平均 TTFT 上提升 81.3% 以上，SLO 达成率提升高达 61.67%，且在不同 QPS 与缓存命中率下均保持优势。

**⚠️ 局限性**

局限性：目前仅针对单个请求的 KVCache 加载与计算进行流水线；对多请求间 KVCache 加载的协同调度、网络冲突、以及更大规模多机集群的评估尚未覆盖；在极低 GPU 内存或网络速率下降时，主动预留策略可能导致资源浪费。

---

## 539. LSA: A Long-Short-term Aspect Interest Transformer for Aspect-Based Recommendation

**arXiv ID:** 2603.21243 | [PDF](https://arxiv.org/pdf/2603.21243v1)

**作者:** Le Liu `[一作]` (Beijing University of Technology), Tong Li `[通讯]` (Beijing University of Technology)

**通讯引用:** 10006 | [OpenAlex ID](https://openalex.org/A5100783224)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种长短期兴趣Transformer（LSA），通过融合图Transformer与时间感知Transformer来建模方面级别的长期与短期用户兴趣，从而提升细粒度推荐性能。

**💡 创新点**

创新点在于将长期与短期兴趣通过门控融合机制结合，并设计兴趣感知方面聚合层，以充分捕捉用户在不同时间尺度上的偏好变化。

**🔧 技术方法**

主要技术包括图Transformer、时间感知Transformer、门控长短期融合、注意力机制的兴趣感知方面聚合，以及使用因式分解机（FM）进行评分预测。

**📊 数据集**

实验数据集来自Amazon Review，分别为Musical Instrument、Office Product、Beauty和Video Games四个子集（分别命名为Music、Office、Beauty、Games）。

**📈 对比分析**

与DeepCoNN、NARRE、CARL、DAML、DSRLN、RGNN等基线进行对比，LSA在四个数据集上均取得最低的MSE，平均提升约2.55%，并在MAE与NDCG方面也表现出显著优势。

**⚠️ 局限性**

局限性包括未针对排序目标进行优化，频率驱动的长期方面选择可能忽略小众兴趣，导致在Games数据集上的NDCG略逊于部分基线。

---

## 540. Test-Time Adaptation via Cache Personalization for Facial Expression Recognition in Videos

**arXiv ID:** 2603.21309 | [PDF](https://arxiv.org/pdf/2603.21309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 541. QMoP: Query Guided Mixture-of-Projector for Efficient Visual Token Compression

**arXiv ID:** 2603.21232 | [PDF](https://arxiv.org/pdf/2603.21232v1)

**作者:** Zhongyang Li `[一作]` (East China Normal University), Kaiwen Long `[通讯]` (Li Auto Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种可自适应的视觉令牌压缩框架 QMoP，将池化、重采样和裁剪三种压缩策略通过查询引导的路由器融合，以在多模态大语言模型中高效压缩视觉令牌。

**💡 创新点**

创新点在于：①引入查询引导的路由器 QGR，基于图像和文本上下文动态分配压缩策略；②采用 Mixture-of-Experts 风格的选择与融合，只激活权重最高的两条分支，抑制无关噪声；③构建 VTCBench 诊断基准，系统评估不同压缩策略对视觉信息的失真。

**🔧 技术方法**

主要技术包括多分支视觉投影器（pooling、resampler、pruning）、查询引导路由器（softmax+MLP）、Mixture-of-Experts 融合机制、两阶段预训练与微调、以及对比实验和 VTCBench 评估。

**📊 数据集**

使用的数据集包括 LAION-CC-SBU-558K（预训练）、665K 混合数据集（指令微调），以及十个公开视觉理解基准（TextVQA、VQAv2、GQA、MMBench、POPE、MM-Vet、MME、MMStar、MMMU、Seed-IMG）和自建的 VTCBench。

**📈 对比分析**

与传统压缩方法（FastV、Pixel-Shuffle、FasterVLM、TokenPacker、LDP-v2 等）以及内部 LLM 压缩方法相比，QMoP 在保持 144/64 视觉令牌的条件下，平均性能提升约 1–2%，在 VTCBench 的五个维度上均超过基线，并在大模型 LLaVA‑1.5‑13B 上保持优势，显著降低 FLOPs 与 KV cache，推理时间也下降 30% 左右。

**⚠️ 局限性**

局限性包括：①对预训练视觉编码器和文本编码器的依赖，若跨域或低资源场景效果未知；②路由器与分支的调参复杂，需额外的温度与噪声控制；③在极端压缩比例（≤36 令牌）下性能仍显著下降，表明仍存在信息失真的上限。

---

## 542. FluidWorld: Reaction-Diffusion Dynamics as a Predictive Substrate for World Models

**arXiv ID:** 2603.21315 | [PDF](https://arxiv.org/pdf/2603.21315v1)

**作者:** Fabien Polly `[一作]` `[通讯]`, Fabien Polly

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种基于反应扩散偏微分方程（PDE）的世界模型（FluidWorld），用 PDE 动力学替代 Transformer 的自注意力，实现在 UCF‑101 与 Moving MNIST 上的无条件视频预测，并在单参数匹配实验中与 Transformer 与 ConvLSTM 进行对比。

**💡 创新点**

创新点在于：①首次将 PDE 作为预测引擎，实现 O(N) 空间复杂度和自适应计算；②通过拉普拉斯扩散提供天然的空间归一化与全局信息传播；③引入生物启发的抑制、疲劳与 Hebbian 扩散机制提升表示多样性与自修复；④展示了 PDE 在多步回放中的自修复与更长的保持连贯性。

**🔧 技术方法**

技术细节包括：PDE 迭代积分（扩散+反应+记忆项），多尺度扩散卷积（多重扩散系数），RMSNorm，Bi-Lateral Inhibition、Synaptic Fatigue、Hebbian Diffusion；Encoder/Decoder 采用 PatchEmbed + FluidLayer + BeliefField；对齐的损失（重建、预测、方差、梯度保留）与同参量的参数匹配；对比实验使用相同的 Encoder/Decoder、Loss、数据和参数量。

**📊 数据集**

主要数据集为 UCF‑101（64×64）用于训练与评估，同时在 Moving MNIST 上验证模型的自修复与多步回放特性。

**📈 对比分析**

对比方法：在参数匹配的三路消融（约 800K 参数），使用同一 Encoder/Decoder、Loss 与数据；评估指标包括重建误差、预测误差、空间标准差、有效秩、Rollout 连贯性与训练吞吐量。结果显示：PDE 与 ConvLSTM 在单步误差相近，PDE 的重建误差低 2 倍，空间结构与有效秩更高；在多步回放中，PDE 能保持连贯性至 h=3，Transformer 与 ConvLSTM 在 h=2 即明显退化。

**⚠️ 局限性**

局限性包括：仅做无条件预测，未验证动作条件化与规划；模型规模小，仅在 64×64 上测试；PDE 迭代导致训练速度比基线慢；未对生物机制做单独消融；仅在 UCF‑101 与 Moving MNIST 上验证；未在更大数据或高分辨率场景评估。

---

## 543. WN-Wrangle: Wireless Network Data Wrangling Assistant

**arXiv ID:** 2603.21310 | [PDF](https://arxiv.org/pdf/2603.21310v1)

**作者:** Anirudh Kamath `[一作]` (University of Utah), Anna Fariha `[通讯]` (University of Utah)

**通讯引用:** 280 | [OpenAlex ID](https://openalex.org/A5087503380)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了针对无线网络数据的交互式数据清洗助手，自动建议并解释满足时间、频率、单位等领域约束的清洗操作。

**💡 创新点**

结合语义分析、时间功能依赖发现、领域特定DSL与基于约束违规量的评分机制，实现了无线网络数据特有的对齐、填充、归约等操作的自动推荐，并提供可解释性与交互性。

**🔧 技术方法**

利用LLM（如GPT‑5）进行语义剖析，Temporal Functional Dependencies (TFD) 约束发现，DSL 定义领域操作，采样加权评分算法，解释生成自然语言；前端交互界面支持实时预览与代码生成。

**📊 数据集**

使用了POWDER城市级无线测试平台收集的真实RF与GPS数据，示例为5G基站信号与手机GPS的合并。

**📈 对比分析**

在POWDER数据上演示了11步交互流程，自动生成56个候选操作并前10%违规减少；相较于CoWrangler、Wrangler等通用工具，能够解决多表时间对齐和对数单位聚合问题，示例中插入7%新行即可满足约束，显著降低违规率。

**⚠️ 局限性**

仅针对时间序列对齐与对数单位聚合的无线网络场景，缺乏对非时间或更复杂空间/频率约束的通用支持；依赖LLM语义剖析，若域知识不足可能误判；评分基于采样，可能无法覆盖所有违规；未评估大规模并行性能或多用户协作。

---

## 544. Unpacking Interaction Profiles and Strategies in Human-AI Collaborative Problem Solving: A Cognitive Distribution and Regulation Perspective

**arXiv ID:** 2603.21288 | [PDF](https://arxiv.org/pdf/2603.21288v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 545. Privacy-Preserving Federated Action Recognition via Differentially Private Selective Tuning and Efficient Communication

**arXiv ID:** 2603.21305 | [PDF](https://arxiv.org/pdf/2603.21305v1)

**作者:** Idris Zakariyya `[一作]` (University of Glasgow), Fani Deligianni `[通讯]` (University of Glasgow)

**通讯引用:** 3934 | [OpenAlex ID](https://openalex.org/A5082776788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 FedDP-STECAR 框架，在联邦视频动作识别中通过在差分隐私条件下仅对少量任务相关层进行微调和噪声注入，同时仅上传这些层的更新，以降低模型泄露风险和通信成本。

**💡 创新点**

创新点在于：① 选择性微调与差分隐私结合，仅对关键层注入噪声，减少信息泄露；② 引入 Top-Level Sampling（无放回采样）实现隐私放大；③ 设计与聚合算法无关的通信高效更新，仅传输微调层；④ 在视频 Transformer（MViT）上验证了上述方案的有效性。

**🔧 技术方法**

技术包括：差分隐私梯度下降（DP‑SGD）+自适应裁剪；Top‑Level Sampling 进行隐私放大；选择性微调（冻结大部分参数，只更新分类头和最后 Transformer 层）；联邦学习框架 FedAvg/FedNova 的聚合；通信压缩和运行时优化。

**📊 数据集**

使用 UCF‑101 视频动作识别数据集，并在 MViT‑B‑16x4 Transformer 模型上进行预训练和微调。

**📈 对比分析**

与全微调、全模型同步的 DP‑FedAvg、FedCDP 等基线对比。结果显示：在严格隐私预算 ε=0.65 的集中式训练下，FedDP‑STECAR 的准确率提升高达 70.2%；在联邦设置下 ε=1.33，准确率 73.1%，通信流量下降 99%，训练时长缩短 48%。

**⚠️ 局限性**

局限性：① 仍需在极低 ε（更强隐私）下进一步验证模型稳健性；② 选择性微调对层次选择敏感，错误的层级可能导致性能下降；③ 实验仅在 UCF‑101 单一数据集上进行，未验证对更大规模、多域视频数据的泛化；④ 对极端非 IID 客户端分布及网络抖动的鲁棒性尚未系统评估。

---

## 546. DeepXplain: XAI-Guided Autonomous Defense Against Multi-Stage APT Campaigns

**arXiv ID:** 2603.21296 | [PDF](https://arxiv.org/pdf/2603.21296v1)

**作者:** Trung V. Phan `[一作]` (Technische Universität Chemnitz), Thomas Bauschert `[通讯]` (Technische Universität Chemnitz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 DeepXplain，一种将可解释性直接嵌入到深度强化学习决策过程中的自适应多阶段APT防御框架。

**💡 创新点**

创新点在于：① 通过证据对齐正则和置信度奖励将解释信号直接参与策略优化，突破传统后置解释的局限；② 构建统一的 XAI 管道，分别提取结构化图解释、时间敏感性归因和策略归因；③ 使得 APT 防御决策既高效又可解释。

**🔧 技术方法**

使用技术包括：图神经网络（GNN）进行命名图编码；LSTM 进行阶段估计；PPO 深度强化学习框架；GNNExplainer 生成图层解释；证据对齐损失、置信度奖励等正则化方法；POMDP 进行状态估计。

**📊 数据集**

数据集与实验平台：在真实企业测试环境（LAN、DMZ、服务器、管理区）收集的端点 (Auditd) 与网络 (Zeek) 监控日志，构建命名图后通过 CALDERA 生成的多阶段 APT 场景进行实验。

**📈 对比分析**

与 Risk-Aware DRL、DeepStage、DeepStage+Post-hoc XAI 进行对比：DeepXplain 将平均阶段加权 F1 从 0.887 提升至 0.915，成功率从 84.7% 提升至 89.6%；解释质量指标（置信度 0.86 > 0.71，紧凑度 0.31 < 0.46，准确度 0.79 > 0.62）均得到显著提升；消除对齐损失或置信度奖励会导致性能下降，验证其有效性。

**⚠️ 局限性**

局限性：仅在模拟/实验环境验证，缺乏真实生产数据与人类分析师的交互评估；模型训练与推理的计算开销未量化；解释生成受 GNNExplainer 质量限制，可能对异常未知攻击的泛化能力有限。

---

## 547. When Models Judge Themselves: Unsupervised Self-Evolution for Multimodal Reasoning

**arXiv ID:** 2603.21289 | [PDF](https://arxiv.org/pdf/2603.21289v1)

**作者:** Zhengxian Wu `[一作]` (OPPO AI Center), Haoqian Wang `[通讯]` (Tsinghua University)

**通讯引用:** 6061 | [OpenAlex ID](https://openalex.org/A5028229824)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无监督自演进训练框架，用于提升多模态大模型的推理能力。

**💡 创新点**

创新点在于同时利用自一致性奖励、Judge模块的边界化调制以及基于组内分布的相对优势来稳定训练并避免模式崩溃。

**🔧 技术方法**

采用Actor-Judge双模型结构、self-consistency奖励、Judge评分调制和Group Relative Policy Optimization（GRPO）等技术。

**📊 数据集**

使用多模态数学推理基准，包括MathVision、MathVerse、WeMath、LogicVista、DynaMath，并在MMR1、GeoQA、Geo3K等无标签训练集上训练。

**📈 对比分析**

与现有无监督自演进方法（VisionZero、EvoLMM、MM-UPT）以及监督训练/教师蒸馏方法比较，平均提升约3-5个百分点，尤其在MathVision上绝对提升5.9个百分点。

**⚠️ 局限性**

局限性在于Judge的评价能力有限，难以持续提升标准，且对不同任务的通用性和模型规模的适应性仍需进一步研究。

---

## 548. Fusing Memory and Attention: A study on LSTM, Transformer and Hybrid Architectures for Symbolic Music Generation

**arXiv ID:** 2603.21282 | [PDF](https://arxiv.org/pdf/2603.21282v1)

**作者:** Soudeep Ghoshal `[一作]` (Kalinga Institute of Industrial Technology), Himanshu Buckchash `[通讯]` (IMC University of Applied Sciences)

**通讯引用:** 444 | [OpenAlex ID](https://openalex.org/A5018753704)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对LSTM、Transformer和Transformer‑LSTM混合模型在符号音乐生成中的本地与全局音乐质量进行细粒度比较，并提出并评估了混合架构。

**💡 创新点**

系统化比较17项音乐质量指标；提出Transformer Encoder + LSTM Decoder混合模型；通过消融实验揭示各组件对生成质量的具体贡献。

**🔧 技术方法**

使用LSTM、Transformer、Transformer‑LSTM网络；自注意力机制、位置编码、L2正则化、温度采样等技术；采用17项本地与全局指标进行定量评估。

**📊 数据集**

使用德语民歌数据集Deutschl（Essen Folk Song Collection），对同一风格的单音轨进行实验。

**📈 对比分析**

生成每种模型1000首曲子，计算17项本地/全局指标，并进行21名参与者的主观听觉评测；混合模型在大多数指标（如音阶变异、节奏变异、旋律多样性、整体质量评分）上均优于单一LSTM或Transformer模型。

**⚠️ 局限性**

仅在单一民族风格的数据集上验证；样本量和评测参与者有限；未涉及多音轨、多风格或多语言的泛化能力；主观评测的可靠性受受试者差异影响。

---

## 549. Which Alert Removals are Beneficial?

**arXiv ID:** 2603.21322 | [PDF](https://arxiv.org/pdf/2603.21322v1)

**作者:** Idan Amit `[一作]` `[通讯]`, Idan Amit

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过随机对照实验、利用标签函数标注自然事件以及监督学习，评估静态分析警报移除对代码复杂度和bug倾向的因果影响。

**💡 创新点**

创新点在于：①提出在大规模项目中识别“自然干预事件”的方法，使实验样本量提升15倍；②将标签函数与监督学习结合，构建可解释的因果干预预测模型；③首次在实践中量化警报移除对Bug概率（CCP）的显著下降。

**🔧 技术方法**

技术手段包括：Pylint静态分析、随机对照实验、标签函数（弱学习器）与Boosting、决策树、随机森林、梯度提升、逻辑回归等机器学习模型、McCabe复杂度、CCP过程指标。

**📊 数据集**

数据集包括：521条人工干预记录（RCT）、3,409条自然干预事件（经标签函数筛选）以及8,245条警报移除事件（原始数据），覆盖多种Python开源项目。

**📈 对比分析**

对比方法：对照组为无干预提交，实验组为干预提交。结果显示：功能复杂度警报移除平均降低McCabe 5.6–13.6点，对应路径数提升48–12,416；CCP平均下降4.1–5.5个百分点；监督学习在低容量模型下达到78%准确率，决策树在高精度模式下实现82%精度、47%召回。整体表明干预效果显著，且模型具备可解释性。

**⚠️ 局限性**

局限性：①干预由单一开发者完成，可能存在个人偏差；②仅在Python项目中验证，缺乏跨语言通用性；③标签函数精度不完美，误判可能影响因果估计；④样本分布不满足IID，可能导致统计偏差；⑤CCP指标受时间漂移影响，需更长时间窗口验证。

---

## 550. HELIX: Scaling Raw Audio Understanding with Hybrid Mamba-Attention Beyond the Quadratic Limit

**arXiv ID:** 2603.21316 | [PDF](https://arxiv.org/pdf/2603.21316v1)

**作者:** Khushiyant `[一作]` (University of Freiburg), Param Thakkar `[通讯]` (Veermata Jijabai Technological Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建一个参数匹配的实验框架 HELIX，用于比较原始波形前端、谱图前端与 Mamba 及注意力混合的音频模型，并系统评估其在不同任务和序列长度下的性能。

**💡 创新点**

展示了输入表示、序列模型族与注意力比例三者之间的耦合关系，并证明单层注意力瓶颈在长序列（5 分钟）下能显著提升性能，突破了纯注意力的内存瓶颈。

**🔧 技术方法**

采用 Mamba 结构、双向 SSM、少量全局自注意力层（单层），以及卷积/声谱图前端，所有模型均约 8.3M 参数。

**📊 数据集**

在 ESC‑50、UrbanSound8K、Speech Commands、Concat Speech Commands、LibriSpeech 30 s 与 VoxPopuli 5 min 这六个数据集上进行评测。

**📈 对比分析**

通过在所有模型上保持相同参数量、相同训练超参和混合训练策略，对比发现：短序列时纯 Mamba 最佳；中等序列时 HELIX 领先；长序列时 HELIX 超过纯 Mamba 11.5 分且纯注意力因 OOM 无法运行。

**⚠️ 局限性**

局限在于只测试了单层注意力的中间位置、未做更细粒度的注意力/层数探索，且实验仅覆盖分类任务；生成或密集预测任务的适用性尚未验证。

---

## 551. The Library Theorem: How External Organization Governs Agentic Reasoning Capacity

**arXiv ID:** 2603.21272 | [PDF](https://arxiv.org/pdf/2603.21272v1)

**作者:** Zachary F. Mainen `[一作]` (Champalimaud Foundation), Zachary F. Mainen `[通讯]` (Champalimaud Foundation)

**通讯引用:** 17343 | [OpenAlex ID](https://openalex.org/A5007724777)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 transformer 代理在外部记忆中检索成本的理论与实验，证明使用索引结构可将检索从线性降低到对数，且随推理深度呈指数优势；

**💡 创新点**

提出了“图书馆定理”，正式量化索引检索与顺序扫描的指数差距，并发现内容熟悉度会诱发模型直接从参数记忆中生成答案，绕过索引；

**🔧 技术方法**

利用 B‑tree 结构模拟文件系统、工具调用、链式思考等外部检索方式，并使用 GPT‑4o‑mini 与 GPT‑5.4 进行实验；

**📊 数据集**

采用三类内容的键值对检索任务：随机哈希、已排序整数和百科词条，规模从 50 至 5,000 条目；

**📈 对比分析**

通过对比顺序（FLAT）、索引（INDEXED）、排序无索引（FLAT‑SORTED）等方案，实验显示索引方案在页读取次数上保持 1 次，而顺序方案随规模呈线性增长，且在大型规模下索引的 token 成本相对低 1‑2‑3 倍，验证理论预期；

**⚠️ 局限性**

实验受限于模型的推理深度、对内容熟悉度的竞争、索引结构的简化（仅 B‑tree）以及仅测试检索任务，未覆盖更复杂索引类型、并发访问和索引构建的学习难度等。

---

## 552. DyGeoVLN: Infusing Dynamic Geometry Foundation Model into Vision-Language Navigation

**arXiv ID:** 2603.21269 | [PDF](https://arxiv.org/pdf/2603.21269v1)

**作者:** Xiangchen Liu `[一作]` (KAIST), Sung-Eui Yoon `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种动态几何感知的 Vision‑Language Navigation（DyGeoVLN）框架，通过将动态几何基础模型（DGFM）与多模态大语言模型融合，实现了在动态环境中的高效视觉语言导航。

**💡 创新点**

核心创新点包括：
1) 设计了 DGFM，利用单目深度估计与多层 3D 嵌入实现动态场景的时空一致三维重建；
2) 跨分支交叉注意力融合 2D 语义与 3D 空间特征，提供统一的视觉‑空间令牌；
3) 引入姿势无关、适应分辨率的空间令牌剪枝策略，既去除冗余又保持重要时空信息；
4) 构建 DyHM3D 动态人类中心 3D 数据集，为动态几何模型的训练提供标注支持。

**🔧 技术方法**

技术栈包括：
- Qwen2‑VL 视觉编码器 + 多模态投影器；
- ViT 视觉分支与 Point‑Map Transformer 3D 分支；
- 动态几何基础模型（DGFM）融合深度与 3D 嵌入；
- 跨分支交叉注意力 + 滑动窗口 KV 缓存；
- 适应分辨率 voxel 分组 + 位置感知剪枝；
- 深度估计器（Depth Anything）与 MLLM 解码器。

**📊 数据集**

主要数据集：
- DyHM3D（用于训练 DGFM）；
- R2R‑CE、RxR‑CE、EnvDrop、HA‑VLN、ScaleVLN（用于训练和评估 VLN 模型）；
- 动态基准 HA‑VLN、静态基准 R2R‑CE；
- 真实世界场景（走廊、大厅、房间、拥挤环境）进行实测。

**📈 对比分析**

与现有方法对比，表现如下：
- 在 HA‑VLN 上 SR 提升 12‑13%，CR 降低 10‑11%；
- 在 R2R‑CE 上取得 NE 4.41、OSR 70.1%、SR 60.8%、SPL 55.8，超过 monocular RGB‑only 以及 panoramic RGB‑D/odometry 基线；
- 真实世界实验中成功率持续领先，证明在动态环境中的鲁棒性。

**⚠️ 局限性**

局限性：
- 对单目深度估计的精度敏感，极端动态或低光条件下可能影响几何重建；
- 令牌剪枝策略需要手工设置阈值，适配不同任务仍需调参；
- 当前仅使用单摄像头，无法利用多视角或深度传感器提供的额外几何信息；
- 需要大规模标注的动态 3D 数据集，数据获取成本高。

---

## 553. WirelessBench: A Tolerance-Aware LLM Agent Benchmark for Wireless Network Intelligence

**arXiv ID:** 2603.21251 | [PDF](https://arxiv.org/pdf/2603.21251v1)

**作者:** Jingwen Tong `[一作]` (Shenzhen University), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 85863 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了WirelessBench，一个容错感知、工具集成、链路可追溯的LLM代理基准，用于评估无线网络智能化的可靠性。

**💡 创新点**

三层认知层次（知识推理、意图分配、主动决策）、容错评分（区分可容忍误差与灾难性错误）、工具必要性设计以及每题完整思维链轨迹。

**🔧 技术方法**

采用大型语言模型（如GPT‑4o）、ReAct/Agentic框架、3GPP TR 38.901 兼容的校园数字孪生 Ray‑Tracing、Kalman 轨迹预测，以及基于心理测量的多模型清洗与增强技术。

**📊 数据集**

基于教材、3GPP 标准、校园 OSM 数据生成的 3,392 条题目（1,392 知识推理、1,000 意图分配、1,000 主动决策），包含数值、公式、结构化输出，并附 CoT 路径。

**📈 对比分析**

与直接提示、进阶提示、自动化代理、参考管线等四类方法对比；最高非协同工具方法 AFlow 得分 84.64%，GPT‑4o 68.00%；工具集成提升约16.6个百分点；精确匹配评估揭示约23%灾难性错误被忽略。

**⚠️ 局限性**

仅覆盖单一校园 3GPP UMi 场景、工具接口与 benchmark 共同设计可能导致上限偏高、模型与工具多样性不足、数据增强比例高、移动模型简化、缺乏跨环境验证与开放源代码基线。

---

## 554. Graph Fusion Across Languages using Large Language Models

**arXiv ID:** 2603.21248 | [PDF](https://arxiv.org/pdf/2603.21248v1)

**作者:** Kaung Myat Kyaw `[一作]` (King Mongkut's University of Technology Thonburi), Jonathan Chan `[通讯]` (King Mongkut's University of Technology Thonburi)

**通讯引用:** 1404 | [OpenAlex ID](https://openalex.org/A5014599047)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于大语言模型的模块化零样本多图融合框架，通过逐步滚动合并、实体中心划分与线性化推理，实现不同语言知识图谱的跨语言对齐。

**💡 创新点**

创新点包括将 LLM 作为语义桥梁的弹性语义胶合剂，使用实体中心批处理与全穷尽批对策略以保留局部拓扑信息，并引入最大置信度聚合规则以提升对齐质量。

**🔧 技术方法**

技术手段主要包含 LLM 提示工程（Gemini 2.5 Flash）、结构三元组线性化为自然语言、实体中心划分、全穷尽批对、响应解析与置信度过滤。

**📊 数据集**

实验使用了 DBP15K 中文-英文（以及日文-英文、法文-英文）知识图谱数据集，主要在 zh_en 语料上评估。

**📈 对比分析**

在零样本设置下，与传统基于嵌入或 GNN 的对齐方法对比，本文实现了 65.4% 的精确率（置信度阈值 0.90 时提升至 88.0%），召回率 23.6%，F1 分数 34.7%，Hits@1 准确率 88.3%，耗时约 5–6 小时。

**⚠️ 局限性**

主要限制在于上下文窗口大小导致召回率受限，且全穷尽批对计算量大，线性化过程可能丢失全局拓扑信息，扩展到更多语言和更大规模图谱仍需进一步优化。

---

## 555. Amortized Variational Inference for Logistic Regression with Missing Covariates

**arXiv ID:** 2603.21244 | [PDF](https://arxiv.org/pdf/2603.21244v1)

**作者:** M. Cherifi `[一作]` (Ecole Militaire Polytechnique), A. Mesloub `[通讯]` (Ecole Militaire Polytechnique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了AV-LR——一种利用稀疏变分推理的二分类逻辑回归框架，用以直接处理带缺失协变量的数据

**💡 创新点**

创新点在于：1）直接在缺失数据空间做推理，省去中间潜变量；2）采用单一推理网络与线性层联合估计回归与缺失机制，保持可解释性；3）可自然扩展到非忽略缺失（MNAR）情形

**🔧 技术方法**

使用了变分推理、重要性加权ELBO、重参数化技巧、深度前馈网络与高斯近似

**📊 数据集**

在合成数据与四个真实数据集（BankNote, Pima Indians Diabetes, Rice Cammeo Osmancik, Breast Cancer Diagnostic）以及NHANES 2013‑2014数据上进行实验

**📈 对比分析**

与MICE、missForest、MIWAE、DLGLM、SAEM等方法对比，AV-LR在参数估计、AUC、准确率等指标上与最先进方法相当或优于，并在训练与推理时间上显著更快

**⚠️ 局限性**

局限性：目前仅支持多元正态协变量分布，未针对高维稀疏特征或非线性协变量分布；缺失机制模型仍相对简单，可能在极端MNAR场景下需要更灵活的后验分布

---

## 556. ConsRoute:Consistency-Aware Adaptive Query Routing for Cloud-Edge-Device Large Language Models

**arXiv ID:** 2603.21237 | [PDF](https://arxiv.org/pdf/2603.21237v1)

**作者:** Haoyu Qiao `[一作]` (Harbin Institute of Technology), Jie Liu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 94750 | [OpenAlex ID](https://openalex.org/A5100454174)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 ConsRoute 的轻量级、语义感知的查询路由框架，能够在云‑边‑设备多层级 LLM 部署中动态决定使用哪一层模型，以保持高质量输出的同时显著降低推理延迟和成本。

**💡 创新点**

创新点包括：① 直接利用设备端 LLM 的前填充隐藏状态作为语义表示，避免额外编码器；② 通过 reranker 计算不同层模型生成文本的语义一致性来构建细粒度软标签；③ 基于聚类的自适应阈值与贝叶斯优化实现查询类型和网络状况下的在线阈值调节。

**🔧 技术方法**

采用的技术包括：prompt‑guided 语义表示提取、轻量化 MLP 预测器、reranker 与 LLM 判别器构造一致性标签、K‑means 聚类、贝叶斯优化（GP+EI）以及在线增量式阈值更新。

**📊 数据集**

使用的主要数据集有 RouterBench 进行路由标签构造和模型对比，以及 MMLU、GSM8K、HumanEval 与 MT‑Bench 四个下游任务评估效果。

**📈 对比分析**

与传统 LLM‑only、DLM‑only、Edge‑only、RouteLLM、MixLLM 等基线相比，ConsRoute 在保持接近云模型 95% 以上准确率的同时，将端到端延迟和推理成本分别降低约 40% 以上，并在跨模型配置下仍能维持良好性能。

**⚠️ 局限性**

局限性包括：需要先行获取 reranker 或 LLM 判别器来生成一致性标签，且对设备侧算力的假设仍有限；阈值学习过程依赖离线训练数据，在线更新周期与实际流量变化速度不匹配时可能导致性能波动；最后在更大规模多厂商多模型的真实部署场景下的鲁棒性尚待进一步验证。

---

## 557. DepthTCM: High Efficient Depth Compression via Physics-aware Transformer-CNN Mixed Architecture

**arXiv ID:** 2603.21233 | [PDF](https://arxiv.org/pdf/2603.21233v1)

**作者:** Young-Seo Chang `[一作]` (Yonsei University), Jae-Sang Hyun `[通讯]` (Yonsei University)

**通讯引用:** 29902 | [OpenAlex ID](https://openalex.org/A5080001926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种端到端的深度图像压缩框架 DepthTCM，将物理启发的多波长编码（MWD）与 Transformer‑CNN 混合网络相结合，实现高效压缩。

**💡 创新点**

创新点在于：① 将深度信息映射为 4 位量化的正弦相位三通道表示，显著降低熵；② 采用全局 4‑bit 量化而非传统 8‑bit，兼顾压缩率与几何精度；③ 引入 Transformer‑CNN 混合模块，使模型兼具局部卷积特征与全局自注意力，提升重建质量。

**🔧 技术方法**

技术手段包括：多波长深度（MWD）编码、全局 4‑bit 量化、Transformer‑CNN 混合压缩骨干、基于率‑失真损失的端到端训练，以及自定义的置信度加权与 TV 正则化。

**📊 数据集**

使用公开数据集：Middlebury 2014、DIODE、ScanNet v2、ScanNet++ iPhone RGB‑D、UnrealStereo4K、KITTI Depth Completion 等进行训练与评估。

**📈 对比分析**

与传统 MWD、N‑DEPTH、JPEG‑90、JPEG‑LS 等方法比较，DepthTCM 在 Middlebury 2014 上实现 0.307 bpp、49.89 dB PSNR、99.38 % 准确率；在 DIODE、ScanNet、UnrealStereo4K 等数据集上同样表现出更低比特率、较高 PSNR 与更快解码速度，明显优于现有最优端到端深度压缩方案。

**⚠️ 局限性**

局限性包括：仅针对单帧静态深度图，未验证在高度动态场景或视频序列中的性能；对实时嵌入式实现仍需进一步优化；量化策略在极低比特率下可能导致几何细节损失。

---

## 558. AutoKernel: Autonomous GPU Kernel Optimization via Iterative Agent-Driven Search

**arXiv ID:** 2603.21331 | [PDF](https://arxiv.org/pdf/2603.21331v1)

**作者:** Jaber Jaber `[一作]` (RightNow AI), Osama Jaber `[通讯]` (RightNow AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

AutoKernel 是一个自动化 GPU 核优化框架，利用单一 LLM 代理在迭代循环中对 PyTorch 模型中的瓶颈核进行编写、基准、验证与改进；

**💡 创新点**

创新点在于：①把专家核优化流程拆解为单文件修改–五阶段验证–保持/回滚的简单循环；②结合模型级 profiling 与 Amdahl 定量分配优化力度；③提供 Triton 与 CUDA C++ 两大后端，让代理可按硬件特性自由切换；④用六层优化手册向 LLM 注入专家经验；⑤使用 Git 追踪实验历史与 TSV 记录，确保可复现与可解释；

**🔧 技术方法**

技术包括：LLM（大模型）驱动的编辑器；Triton DSL 与 CUDA C++ 编译器；五阶段正确性 harness（烟雾、尺寸扫、稳定性、确定性、边缘案例）；Roofline 与 Amdahl 分析指导搜索；多层级优化手册；Git+TSV 实验跟踪；KernelBench 评测集成；HuggingFace kernel 导出工具；

**📊 数据集**

数据集与模型：使用多种 HuggingFace Transformer（GPT‑2 124M、LLaMA 160M/7B、BERT‑base 110M、自定义模板）；KernelBench 250 个标准 GPU 核问题；社区挑战数据（B200 向量求和、FP4 matmul、Cutlass 对比等）；

**📈 对比分析**

对比方法：在 NVIDIA H100 上与 PyTorch eager（cuBLAS/ATen）、TorchInductor、CUTLASS 进行多维度基准；AutoKernel 在 34 个配置下全部通过五阶段验证，取得显著加速：RMSNorm 5.29×、Softmax 3.44×、Cross‑Entropy 2.94×、MatMul 1.55× 对 TorchInductor；整体 GPU 时间提升约 1.25× 以上；

**⚠️ 局限性**

局限性：仅单 GPU 单核优化；LLM 生成能力受限，难以实现软件流水线、跨 CTA 合作、分布式多 GPU 及复杂 PTX 生成；缺乏多设备内存管理与分布式同步；未来需扩展多实例搜索、学习搜索策略、跨核融合等方向。

---

## 559. Text-Image Conditioned 3D Generation

**arXiv ID:** 2603.21295 | [PDF](https://arxiv.org/pdf/2603.21295v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 560. Improving Coherence and Persistence in Agentic AI for System Optimization

**arXiv ID:** 2603.21321 | [PDF](https://arxiv.org/pdf/2603.21321v1)

**作者:** Pantea Karimi `[一作]` (MIT), Hari Balakrishnan `[通讯]` (MIT)

**通讯引用:** 96357 | [OpenAlex ID](https://openalex.org/A5113516878)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于代理的LLM研究架构，利用连续代理迭代设计、测试和分析，突破传统演化和单上下文限制，实现长期一致性和知识累积。

**💡 创新点**

将长期探索与单一上下文窗口分离，利用持久归档和摘要传递机制实现跨代理知识共享，克服演化邻域偏差和一致性上限。

**🔧 技术方法**

采用大型语言模型、工具化执行环境、代理协作、持久归档和摘要化知识传递等技术。

**📊 数据集**

在多云多播、LLM推理请求路由、数据库KV缓存重用等系统任务以及ADRS基准集上进行实验。

**📈 对比分析**

与Evolution of Heuristics、FunSearch、OpenEvolve、Glia等方法进行10次10次评估，平均最高分/90%置信区间显示其在8/9任务中击败人类SOTA，在所有类别超过OpenEvolve，具体如多云多播成本$622< SOTA $626，推理路由平均响应时间23.9s<25.7s。

**⚠️ 局限性**

依赖LLM质量与工具环境，归档与摘要机制的设计与调优仍需手工，且对极大规模系统或极长评估预算的适用性尚未验证。

---

## 561. Sonny: Breaking the Compute Wall in Medium-Range Weather Forecasting

**arXiv ID:** 2603.21284 | [PDF](https://arxiv.org/pdf/2603.21284v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 562. Aggregation Alignment for Federated Learning with Mixture-of-Experts under Data Heterogeneity

**arXiv ID:** 2603.21276 | [PDF](https://arxiv.org/pdf/2603.21276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 563. Stream separation improves Bregman conditioning in transformers

**arXiv ID:** 2603.21317 | [PDF](https://arxiv.org/pdf/2603.21317v1)

**作者:** James Clayton Kerce `[一作]` (Georgia Tech Research Institute), James Clayton Kerce `[通讯]` (Georgia Tech Research Institute)

**通讯引用:** 138 | [OpenAlex ID](https://openalex.org/A5053040606)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Transformer中softmax层的Hessian进行度量，比较单流与CASCADE架构以及是否加入逐层监督，评估线性引导方法的效果，并提出基于正交余弦的诊断指标。

**💡 创新点**

发现标准单流Transformer在中间层的Hessian严重退化，CASCADE架构显著改善几何条件，并提供了一种低成本的正余弦诊断来预测线性干预的可靠性。

**🔧 技术方法**

利用Bregman几何、Hessian有效秩与条件数评估、自然梯度（dual）与欧氏（Euclidean）引导对比、以及正余弦相似度计算。

**📊 数据集**

使用GPT‑2分词器、相同规模的数据混合（约480个上下文样本）以及包含性别词对的提示集，对核心推理、归纳、最近性与大小写四个下游任务进行验证。

**📈 对比分析**

在2×2因子设计的四个模型间比较有效秩、条件数、轨迹和KL优势；CASCADE架构在有效秩上提升至22倍，辅助监督虽有改善但不如架构；正余弦阈值0.3能准确预测层级的引导成功率，并在下游任务中验证了其有效性。

**⚠️ 局限性**

模型规模有限（45.4M参数、6层），评估任务覆盖范围有限，正余弦阈值的普适性尚待验证，且仅针对语言模型的几何特性进行探讨。

---

## 564. enhancing reasoning accuracy in large language models during inference time

**arXiv ID:** 2603.21301 | [PDF](https://arxiv.org/pdf/2603.21301v1)

**作者:** Vinay Sharma `[一作]` (Firstsource), Manish Jain `[通讯]` (Firstsource)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并比较三种推理时的 inference‑time 技术，以提升大语言模型的多步推理准确率。

**💡 创新点**

首次系统评估自一致性+受控温度与核采样、双模型交叉验证以及自我反思三种推理改进方法的效果。

**🔧 技术方法**

采用 Chain‑of‑Thought 提示、受控随机解码、交叉模型一致性判定和自我批评修订技术。

**📊 数据集**

使用 HuggingFace 上的 Logical Reasoning Improvement Dataset（来自 garage‑bAInd/Open‑Platypus）。

**📈 对比分析**

实验结果显示，自一致性+受控采样在单模型下提升 8–15% 准确率，双模型验证优先精度而非召回，反思仅提升约 3.4%。

**⚠️ 局限性**

受限于模型规模与任务复杂度，自反思对小型非推理模型效果有限；双模型方法成本高，三种技术需根据风险场景权衡使用。

---

## 565. Shadoks Approach to Parallel Reconfiguration of Triangulations

**arXiv ID:** 2603.21293 | [PDF](https://arxiv.org/pdf/2603.21293v1)

**作者:** Guilherme D. da Fonseca `[一作]` (Aix-Marseille Université), Yan Gerard `[通讯]` (Université Clermont Auvergne)

**通讯引用:** 515 | [OpenAlex ID](https://openalex.org/A5060441363)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在给定点集上平面三角剖分的并行翻转重构问题，提出了一种基于 SAT/MaxSAT 的精确求解框架，并结合多种贪心与启发式算法，在 CG:SHOP 2026 挑战中获得了近乎完美的解。

**💡 创新点**

创新点包括：1) 为并行翻转路径构造提出了高效的 SAT 编码，利用 flip 变量和边变量并通过变量消除显著压缩规模；2) 设计了“快乐边猜想”与“交叉下界”理论，进一步简化约束并提供更强的下界；3) 将精确求解与启发式策略相结合，形成分阶段求解流程，在保持可解释性的同时实现大规模实例的高效求解。

**🔧 技术方法**

技术主要包括：SAT/MaxSAT 求解器（如 MiniSat/MaxHS）、图搜索与动态规划、贪心与 squeaky‑wheel 迭代、动态预处理（字符串下界）以及多线程与分布式执行（GNU Parallel、Slurm）。

**📊 数据集**

使用了 CG:SHOP 2026 提供的 250 个实例集，包含 15–12,500 个点，2–200 个输入三角剖分，共划分为三类：小规模 (≤320点)、中等规模 (≤500点) 与大规模 (≥500点)。

**📈 对比分析**

实验对比显示：在 189/250 个实例上可获得最优解，且在 249/250 个实例上取得了 100% 的最佳解；相对于仅使用启发式的结果，SAT/MaxSAT 能将目标值进一步降低约 10%，整体求解时间在单核 AMD Ryzen 9 9900X 上从数秒到数小时不等，显著优于竞争对手。

**⚠️ 局限性**

主要局限包括：1) 对于点集完全凸形时，空凸四边形数爆炸导致 SAT 模型规模大，求解困难；2) 大规模实例（>10k 点）仍需大量内存和时间；3) 依赖外部求解器，求解可变性高；4) “快乐边猜想”在所有情况尚未证明，可能导致误导；5) 路径长度下界仍受限，理论上可能有更紧的下界未被发现。

---

## 566. Evaluating Factor-Wise Auxiliary Dynamics Supervision for Latent Structure and Robustness in Simulated Humanoid Locomotion

**arXiv ID:** 2603.21268 | [PDF](https://arxiv.org/pdf/2603.21268v1)

**作者:** Chayanin Chamachot `[一作]` `[通讯]` (Chulalongkorn University), Chayanin Chamachot (Chulalongkorn University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在单人类机器人Unitree G1上，通过在PPO训练中给Transformer编码器的24维因子化潜在空间加上逐因子辅助损失，评估该策略是否能产生可解码、功能可分离的动力学潜在结构，并与LSTM、Transformer、MLP等基线进行对比。

**💡 创新点**

尝试把因子化潜在空间与每个动力学因子对应的辅助监督相结合，探究是否能让潜在表示可解释并提升在分布外（OOD）环境下的鲁棒性；同时设计6种互补分析工具（probe、intervention、梯度正交、SVD、MI、标准解耦度量）以全面评估潜在结构。

**🔧 技术方法**

使用Proximal Policy Optimization（PPO）强化学习、Transformer编码器、tanh瓶颈、因子化潜在空间、辅助回归头、线性/MLP probe、梯度正交分析、SVD、kNN信息量估计、MIG/ DCI/SAP 等。

**📊 数据集**

在NVIDIA Isaac Lab模拟环境中，对Unitree G1机器人执行四种地面任务（平坦、抛射、随机、地形），并对多种域随机化（摩擦、推力、延迟等）进行极端组合实验。

**📈 对比分析**

与LSTM、Transformer、MLP基线在确定性评估（100/50回合）下比较。LSTM在所有任务中获得最高的在分布内奖励；DynaMITE的在分布外（特别是组合偏移）下降幅度更小（2.3% vs. 16.7%），但主要归因于tanh瓶颈压缩；辅助监督对奖励无显著提升；在推力恢复任务中DynaMITE恢复速度快，但峰值误差和整体奖励低于LSTM。

**⚠️ 局限性**

主要局限包括：辅助监督未能产生可解码或解耦的潜在结构；统计功效有限（大多数OOB对比未通过多重检验校正）；仅在模拟实验中验证，未涉及硬件转移；评价指标对鲁棒性定义不够细致；probe与解耦度量仅为线性/浅层MLP，可能漏检复杂结构。

---

## 567. Fingerprinting Deep Neural Networks for Ownership Protection: An Analytical Approach

**arXiv ID:** 2603.21411 | [PDF](https://arxiv.org/pdf/2603.21411v1)

**作者:** Guang Yang `[一作]` (Virginia Commonwealth University), Changqing Luo `[通讯]` (University of Houston)

**通讯引用:** 2530 | [OpenAlex ID](https://openalex.org/A5108048105)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

提出了一种基于理论分析的对抗样本指纹化方法 AnaFP，用来验证深度神经网络模型的所有权。

**💡 创新点**

核心创新在于把指纹与决策边界距离的拉伸因子 τ 的取值约束为由鲁棒性与唯一性两项理论上推导出的上下界所定义的可行区间，并通过量化阈值放宽与网格搜索解决了两者间的循环依赖。

**🔧 技术方法**

使用了对抗样本生成（Carlini & Wagner ℓ₂ 攻击）、梯度与 Lipschitz 常数估计、量化放宽、网格搜索以及多种性能保持模型修改攻击来评估鲁棒性。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100、MNIST 与 PROTEINS 四个数据集上，分别训练 ResNet‑18、ResMLP、GAT 等模型进行实验。

**📈 对比分析**

与 UAP、IPGuard、MarginFinger、AKH、ADV‑TRA、GMFIP 等六种基线进行对比，使用 ROC‑AUC 作为评估指标。AnaFP 在所有模型与攻击场景下均取得最高或相近的 AUC，尤其在 fine‑tuning、KD、AT、N‑finetune、P‑finetune 等攻击下表现优异。

**⚠️ 局限性**

局限性包括：① 需要构建并维护代理模型池；② 对量化阈值与网格搜索步长仍需经验调优；③ 在极端的知识蒸馏或多步骤攻击下鲁棒性下降；④ 仅针对黑盒查询环境，白盒或硬件加速场景尚未验证。

---

## 568. Critical window for approximate counting in dense Ising models

**arXiv ID:** 2603.21406 | [PDF](https://arxiv.org/pdf/2603.21406v1)

**作者:** Andreas Galanis `[一作]` (University of Oxford), Eric Vigoda `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 3044 | [OpenAlex ID](https://openalex.org/A5051363868)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在临界状态下稠密伊辛模型的分区函数近似计算的复杂性，证明了在宽度为N^-1/2+ε的窗口内近似计算是计算上困难的。

**💡 创新点**

首次建立了近似计数计算复杂性的尖锐缩放窗口，提供了几乎紧密的难度界限。

**🔧 技术方法**

采用了全局方法来聚合所有小工具的波动，而不是要求每个小工具的紧密集中保证，从而克服了在临界性下的标准难度归约的局限。

**📊 数据集**

使用了N×N的正半定矩阵J，特别是在J_2≈1的情况下，捕捉了稠密图的临界性。

**📈 对比分析**

与非临界情况下的标准硬度归约相比，临界情况下的归约面临更大的波动，导致次优界限。通过新的全局方法，获得了临界窗口的最优指数。

**⚠️ 局限性**

在稀疏伊辛模型和硬核模型中，当前的证明方法无法扩展到临界点以外，因此在稀疏情况下建立算法或难度结果的缩放窗口仍然是一个有趣的开放问题。

---

## 569. Mitigating Objectness Bias and Region-to-Text Misalignment for Open-Vocabulary Panoptic Segmentation

**arXiv ID:** 2603.21386 | [PDF](https://arxiv.org/pdf/2603.21386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 570. The Myhill-Nerode Theorem for Bounded Interaction: Canonical Abstractions via Agent-Bounded Indistinguishability

**arXiv ID:** 2603.21399 | [PDF](https://arxiv.org/pdf/2603.21399v1)

**作者:** Anthony T. Nixon `[一作]` `[通讯]`, Anthony T. Nixon

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对有限POMDP的受限观测者（bounded observer）可识别的等价划分（canonical quotient），并证明其唯一性、最小性与决策充分性。

**💡 创新点**

创新点在于：①将Myhill–Nerode定理推广到闭环有限控制场景；②构造基于Wasserstein伪度量的观测历史距离；③给出精确决策充分性与观察-Lipschitz的近似价值误差界；④设计可操作的近似工具箱（子集压缩、采样、分层、观测粗化）来实现可扩展的求解。

**🔧 技术方法**

使用的技术包括：有限状态FSC（有时是时钟感知）、Wasserstein 1-度量、分层动态规划、子模函数贪心压缩、采样估计的1-度量、观测空间的Lipschitz粗化、以及基于等价类的POMDP构造。

**📊 数据集**

主要数据集为经典的Tiger、GridWorld（3×3、5×5）、RockSample（4×4）以及随机生成的POMDP，所有实验均在有限状态、有限观测和有限时限的离散环境中进行。

**📈 对比分析**

方法比较：在可证明的时钟感知exact实例上，分割后的quotient在观测-动作可测量目标上保持零价值差距；在隐状态奖励下给出L_R·T·ε的误差上界；在操作近似实例中展示压缩率（如Tiger 256→64）与运行时间（≤0.3 s）。实验表明，操作近似在保留决策质量的同时显著减小状态空间，且相较于无结构的TV度量，Wasserstein度量能获得更高的压缩率。

**⚠️ 局限性**

局限性包括：①隐状态奖励的误差界可能非常松散；②完整的Myhill–Nerode证明未给出多项式时间构造算法；③在大m或大观测空间时的指数级历史树规模仍是瓶颈；④需要已知的POMDP模型，模型估计误差的样本复杂度尚未量化；⑤操作近似的gap（δ_clk/δ_S）需要额外测量，缺乏先验估计。

---

## 571. A Constructive Approach to $q$-Gaussian Distributions: $α$-Divergence as Rate Function and Generalized de Moivre-Laplace Theorem

**arXiv ID:** 2603.21391 | [PDF](https://arxiv.org/pdf/2603.21391v1)

**作者:** Hiroki Suyari `[一作]` (Chiba University), Antonio M. Scarfone `[通讯]` (Politecnico di Torino)

**通讯引用:** 1726 | [OpenAlex ID](https://openalex.org/A5086499705)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文建立了一个构造性的概率框架，用于描述幂律分布，推导了广义二项分布及其与q-高斯分布的收敛关系。

**💡 创新点**

创新点在于通过非线性微分方程推导出q-高斯分布和广义大偏差原理，而不依赖于变分原理或特定的函数假设。

**🔧 技术方法**

使用了非线性微分方程、q-斯特林公式和组合计数等数学技术。

**📊 数据集**

使用了不同q值（0 < q < 2）的数值验证数据集，验证了理论推导的结果。

**📈 对比分析**

与现有方法相比，本文的方法通过构造性推导提供了更清晰的微观机制，性能上在0 < q < 1时，α-散度被识别为广义二项过程的速率函数，而在q > 1时，标准大偏差原理的缩放失效。

**⚠️ 局限性**

限制在于对于q > 1的重尾分布，宏观大偏差缩放失效，尽管局部极限行为在0 < q < 2的范围内仍然有效。

---

## 572. Cerebra: Aligning Implicit Knowledge in Interactive SQL Authoring

**arXiv ID:** 2603.21363 | [PDF](https://arxiv.org/pdf/2603.21363v1)

**作者:** Yunfan Zhou `[一作]` (Zhejiang University), Yingcai Wu `[通讯]` (Zhejiang University)

**通讯引用:** 6107 | [OpenAlex ID](https://openalex.org/A5073986937)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了Cerebra——一款交互式NL‑to‑SQL工具，通过从用户历史脚本自动提取隐式知识、在知识视图中可视化并支持知识层面迭代改写，实现了用户与LLM之间隐式知识的对齐；

**💡 创新点**

创新点包括：①自动化提取并存储用户历史SQL中的隐式知识；②通过知识树视图将模型的隐式假设显式化，提升对齐与可解释性；③在查询生成与调优过程中支持基于知识项的直接修改，而非仅靠代码编辑；

**🔧 技术方法**

技术栈涵盖：基于Qwen3 LLM的生成与知识抽取；SQLGlot实现AST解析与子查询分解；SentenceTransformers进行语义检索；Vue.js+Monaco+dagre构建交互式UI；利用cosine相似度与检索增强生成；

**📊 数据集**

实验使用BIRD跨域NL‑to‑SQL数据集（欧洲足球、毒理学、代码库社区、Formula 1）以及从BIRD自动生成的232个自定义评估任务；

**📈 对比分析**

对比方法：在两数据集上采用逆序混合设计的用户研究，16名参与者分别使用Cerebra与不含知识模块的Baseline，测量任务完成时间与NASA‑TLX；结果显示Cerebra显著缩短完成时长、降低努力感；技术评估中，检索增强+迭代改写管道的执行准确率超过90%，而直接生成仅22‑40%，Baseline 29‑47%；

**⚠️ 局限性**

局限性包括：①需要人工审阅并完善列描述，列数增多时成本高；②知识库仅针对单一schema，跨schema迁移受限；③未与主流IDE集成，可能导致工作流分离；④对数据库schema变更时知识失效缺乏自动更新机制；

---

## 573. Relax Forcing: Relaxed KV-Memory for Consistent Long Video Generation

**arXiv ID:** 2603.21366 | [PDF](https://arxiv.org/pdf/2603.21366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 574. The Workload-Router-Pool Architecture for LLM Inference Optimization: A Vision Paper from the vLLM Semantic Router Project

**arXiv ID:** 2603.21354 | [PDF](https://arxiv.org/pdf/2603.21354v1)

**作者:** Huamin Chen `[一作]` (vLLM Semantic Router Project), Junchen Jiang `[通讯]` (Tensormesh Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并阐述了 Workload–Router–Pool (WRP) 三维框架，整合工作负载特征、路由决策与 GPU 池架构，并在此基础上梳理了 vLLM Semantic Router 研究计划的四大支柱（路由架构、池路由与车队优化、多模态与代理路由、治理与标准），展示了三维之间的耦合关系，映射已有论文至 WRP 矩阵，并给出二十一项未来研究方向。

**💡 创新点**

创新点在于：1) 用 WRP 统一划分并系统化三方互联的 LLM 推理优化；2) 将跨工作负载、路由与池层的交互视为耦合优化问题，而非独立研究；3) 通过语义路由堆栈和大规模日志收集，实现跨租户的安全、隐私、成本和能耗的多维决策；4) 提出具体的路线图和实验评估指标，为后续工作提供可操作的基准。

**🔧 技术方法**

技术方法包括：信号驱动的语义路由（多种信号类型及 mmBERT-Embed 2D 嵌入）；冲突无关策略语言 ProbPol；FastRouter 低延迟 98× 加速；Token‑Budget 路由与 FleetOpt 分析/仿真模型；1/W 能耗法则；多模态 AVR 及工具选择算法；安全与隐私分层分类（SafetyL1/L2、HaluGate、FactCheck）；KV 缓存分层与分区；路由与池的在线/离线 RL、分布式强化学习；以及 SIRP 标准化协议。

**📊 数据集**

使用的数据集包括：生产日志（Azure LLM 推理追踪、LMSYS‑Chat‑1M、ServeGen、SWE‑bench、BFCL、Splitwise）、公共基准（MetaTool、Qwen3‑235B‑A22B、OpenAI Responses API）、安全/幻觉评测数据（MLCommons AI Safety Taxonomy、MM‑BERT‑32K、HaluGate 等），以及内部收集的多租户请求/工具调用/安全事件日志。

**📈 对比分析**

评估方法：将新路由方案与基线（静态规则、单调模型、无路由等）在同一工作负载上对比；使用指标包括成本（GPU‑小时/请求）、延迟（TTFT/TPOT）、能耗（tokens/W）、准确率（任务完成率、NDCG@5）、吞吐量（QPS/并发数）。实验结果显示：Token‑Budget 路由可降低 17–39% GPU 使用；FastRouter 使路由延迟降至 50 ms；1/W 法下，组合路由+池可实现 4.25× 能耗提升；AVR 与 OATS 组合可将成本压缩 78% 并保持 2 pp 质量；RL 路由在 agent 场景下显著缩短 session 长度。对比实验表明跨维度耦合带来 3–6 % 的额外成本优势。

**⚠️ 局限性**

局限性包括：1) 许多交叉维度实验仍在理论/仿真阶段，缺乏统一的生产级部署验证；2) 大规模失败/安全事件表需在稀疏领域中保持统计可靠性，冷启动问题尚未彻底解决；3) 多目标优化（成本、延迟、能耗、准确率）在实时系统中的权衡尚未给出可行的自动化框架；4) 安全/隐私阈值和误报率需针对不同租户手工调优；5) 路由决策的可解释性与可审计性仍待加强；6) 依赖精确的工作负载特征估计，对动态变化的业务场景响应速度有限。

---

## 575. A Generalised Exponentiated Gradient Approach to Enhance Fairness in Binary and Multi-class Classification Tasks

**arXiv ID:** 2603.21393 | [PDF](https://arxiv.org/pdf/2603.21393v1)

**作者:** Maryam Boubekraoui `[一作]` (HESTIM Engineering and Business School), Antinisca Di Marco `[通讯]` (University of L'Aquila)

**通讯引用:** 4961 | [OpenAlex ID](https://openalex.org/A5044201375)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种 Generalised Exponentiated Gradient (GEG) 算法，用于在多分类任务中同时满足多个线性公平约束，实现公平学习。

**💡 创新点**

创新点包括：①将原本仅适用于二分类的 Exponentiated Gradient 方法推广到多分类场景；②支持多重公平定义（统计平等、机会平等等）的并行约束；③引入正类（positive label）变体的公平约束，使得多分类问题可以被转化为线性约束；④通过组合公平约束（Combined Parity）实现对不同公平指标的同时优化。

**🔧 技术方法**

核心技术是多目标凸优化（随机化分类器与 Lagrangian 松弛），利用指数梯度更新多维对偶变量，构造成本敏感的分类器学习；并结合正类标记的线性矩矩约束，将公平性转化为可求解的线性不等式。

**📊 数据集**

实验使用了 10 个公开数据集：7 个多分类（CMC、Crime、Drug、Law、Obesity、Park、Wine）和 3 个二分类（Adult、COMPAS、German），均来自 UCI 或先前研究，包含不同敏感属性（宗教、种族、性别、年龄等）。

**📈 对比分析**

与 Logistic Regression、原始 EG、DEMV（预处理）和 Blackbox（后处理）等方法进行基准比较，并在基分类器上尝试 Random Forest 和 Gradient Boosting。评估指标包括 Accuracy、Macro Precision/Recall/F1 以及多种公平度量（SPD、EOD、AOD 等）。结果显示 GEG 在大多数组合上获得更多 Pareto 最优解，公平性提升可达 92%，准确率下降最多 14%，且在高基分类器偏差的数据集上表现尤为突出。

**⚠️ 局限性**

主要限制：①需要预先指定正类标签，限制了对更一般多分类公平定义的直接适用；②对样本不平衡或类数较多的数据集可能导致准确率下降；③训练时间相对较长；④仅针对单一敏感属性，未覆盖交叉公平（intersectional fairness）。

---

## 576. Assessing Data Literacy in K--12 Education: Challenges and Opportunities

**arXiv ID:** 2603.21382 | [PDF](https://arxiv.org/pdf/2603.21382v1)

**作者:** Annabel Goldman `[一作]` (Northwestern University), Matthew Kay `[通讯]` (Northwestern University)

**通讯引用:** 4959 | [OpenAlex ID](https://openalex.org/A5089605137)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对13名高中教师进行半结构化访谈，探讨他们在评估中使用数据可视化的实践，并通过主题分析归纳出四大挑战。

**💡 创新点**

首次将评估作者过程视为探究数据素养的切入口，系统识别了概念模糊、真实与合成数据权衡、可视化获取难度以及与学科目标的平衡四个关键挑战，并提出针对性支持建议。

**🔧 技术方法**

采用定性主题分析方法处理访谈文本；使用现有研究框架（如数据可视化素养框架）进行概念梳理。

**📊 数据集**

访谈记录（共13位教师的对话）作为主要数据集。

**📈 对比分析**

论文未涉及实验性能对比，主要通过教师案例描述与主题比较呈现结果。

**⚠️ 局限性**

样本规模有限，主要来自北方大学附近高中，缺乏跨学科和多地区验证，研究结论可能不具普遍性。

---

## 577. TIDE: Token-Informed Depth Execution for Per-Token Early Exit in LLM Inference

**arXiv ID:** 2603.21365 | [PDF](https://arxiv.org/pdf/2603.21365v1)

**作者:** Jaber Jaber `[一作]` (RightNow AI), Osama Jaber `[通讯]` (RightNow AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练的自适应深度退出系统，适用于任何 HuggingFace 的因果语言模型；

**💡 创新点**

利用学习到的轻量级路由器在冻结模型隐藏层上进行逐标记的退出决策，并提供统一的模型适配器和 CUDA 核心加速；

**🔧 技术方法**

采用两层 MLP 路由器、余弦相似度标记、后置评估机制和融合 RMSNorm+路由器的 CUDA 内核；

**📊 数据集**

使用 WikiText‑103（2000 篇文本）进行校准，并在 DeepSeek R1 Distill 8B 与 Qwen3 8B 上进行性能验证；

**📈 对比分析**

相较于原始模型，Prefill 延迟降低约 5–7%，批量吞吐量提升 5–8%，解码阶段 98–99% 的 token 能提前退出，同时保持推理质量；

**⚠️ 局限性**

系统仅在后置阶段做退出，未实现真正的层跳过；阈值设定过保守导致大多数退出集中在最后一个检查点；批量大时存在额外开销。

---

## 578. Benchmarking Bengali Dialectal Bias: A Multi-Stage Framework Integrating RAG-Based Translation and Human-Augmented RLAIF

**arXiv ID:** 2603.21359 | [PDF](https://arxiv.org/pdf/2603.21359v1)

**作者:** K. M. Jubair Sami `[一作]` (BRAC University), Farig Sadeque `[通讯]` (BRAC University)

**通讯引用:** 350 | [OpenAlex ID](https://openalex.org/A5009105388)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了双阶段框架，对9种孟加拉语方言的LLM问答偏差进行量化评估。

**💡 创新点**

引入LLM‑as‑a‑judge与CoT先行的评估范式，提出CBS指标，并验证传统指标在方言评估中的失效。

**🔧 技术方法**

采用RAG式翻译管道、RLAIF评估框架、多重LLM评审、Embedding相似度与LLM评判、CCC与CBS等统计校验技术。

**📊 数据集**

利用Vashantor、标准化平行语料、人工标注的400基础问答扩展至4000个方言问答，并对19个开源LLM进行评测。

**📈 对比分析**

通过多评审CCC≥0.80、CBS≥0.75验证评估鲁棒；发现方言差异越大模型得分越低，规模提升不一定缓解偏差。

**⚠️ 局限性**

仅覆盖9种方言、评审LLM对音系的理解不足、Embedding动态范围受限、域与语料多样性有限。

---

## 579. FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction

**arXiv ID:** 2603.21356 | [PDF](https://arxiv.org/pdf/2603.21356v1)

**作者:** Yuqiu Liu `[一作]` (Simon Fraser University), Michael Mahoney `[通讯]` (Lawrence Berkeley National Lab)

**通讯引用:** 24639 | [OpenAlex ID](https://openalex.org/A5033006662)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个物理感知的3D重建方法FluidGaussian，通过在主动视角选择中加入流体-结构交互模拟来提升重建质量。

**💡 创新点**

创新点在于将基于流体速度场的分歧率作为物理不确定性度量，利用它重新排序下一最佳视角候选，并将其与传统视觉不确定性结合，实现视觉与物理双重优化。

**🔧 技术方法**

核心技术包括3D高斯展开（3D Gaussian Splatting）作为几何表示、DFSPH流体模拟、基于粒子分歧率的物理不确定性计算以及NBV重排序。

**📊 数据集**

实验使用了Blender合成场景、真实数据集MipNeRF360以及用于气动研究的DrivAerNet++三大数据集。

**📈 对比分析**

与ActiveNeRF和FisherRF基线相比，加入FluidGaussian后PSNR提升最多8.6%，同时流体速度场分歧率下降62.3%，在所有数据集上均实现视觉和物理指标的显著提升。

**⚠️ 局限性**

局限性包括仅考虑不可压缩水流，模拟成本较高，方法目前只针对3D高斯展开，可能难以直接迁移到其他几何表示；此外，仅在有限的场景与流体条件下验证，缺乏对更复杂物理交互的评估。

---

## 580. RoboAlign: Learning Test-Time Reasoning for Language-Action Alignment in Vision-Language-Action Models

**arXiv ID:** 2603.21341 | [PDF](https://arxiv.org/pdf/2603.21341v1)

**作者:** Dongyoung Kim `[一作]` (KAIST), Younggyo Seo `[通讯]` (UC Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于强化学习的多模态大语言模型训练框架（RoboAlign），在监督微调后通过GRPO对低级动作（FAST tokens）进行对齐，从而显著提升视觉-语言-动作模型在机器人任务中的性能。

**💡 创新点**

创新点在于直接将低级动作令牌与多模态大语言模型的推理过程对齐，首次通过零样本推理生成动作令牌并用RL细化，解决了语言推理与动作执行之间的模态鸿沟。

**🔧 技术方法**

采用了监督微调（SFT）+ GRPO强化学习、FAST动作令牌化、低级动作推理奖励（格式+准确率）、以及Diffusion-based动作头等技术。

**📊 数据集**

使用的数据集包括BridgeV2（400k FAST token样本）、自制VQA与零样本CoT推理数据、以及公开的机器人学习数据；RL阶段仅用BridgeV2的12.8k子集。

**📈 对比分析**

与仅SFT、仅VLA、ECoT和其他RL对齐方法比较，RoboAlign在LIBERO、CALVIN和真实机器人环境中相较SFT基准分别提升17.5%、18.9%和106.6%，并在长时程、目标等任务上实现显著增益。

**⚠️ 局限性**

局限性包括：可能产生不安全动作（需在训练阶段预防）；对不同模型架构的迁移性仍需进一步验证；RL阶段虽数据量少，但对计算资源和调参有一定要求。

---

## 581. TimeTox: An LLM-Based Pipeline for Automated Extraction of Time Toxicity from Clinical Trial Protocols

**arXiv ID:** 2603.21335 | [PDF](https://arxiv.org/pdf/2603.21335v1)

**作者:** Saketh Vinjamuri `[一作]` (Fairview Hospital), Ramez Kouzy `[通讯]` (University of Texas MD Anderson Cancer Center)

**通讯引用:** 1614 | [OpenAlex ID](https://openalex.org/A5062204572)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了基于Gemini LLM的TimeTox管线，用于自动从临床试验方案的评估时间表中提取时间毒性（医疗接触天数）

**💡 创新点**

提出单通道提取与两阶段拆解的对比，发现稳定性比纯精度更关键，并设计了基于位置的多跑一致性机制

**🔧 技术方法**

采用Google Gemini 2.5 Flash进行摘要提取，Gemini 3.0 Flash进行时间毒性提取，使用JSON强制输出与多跑中位数聚合

**📊 数据集**

使用了649份肿瘤学临床试验方案，其中20份合成时间表用于精度验证，644份真实方案用于稳定性评估

**📈 对比分析**

在合成数据上，两阶段管线实现100%可接受精度（MAE 0.81天），单通道仅41.5%；在真实数据上，单通道3跑实现95.3% IQR≤3天的临床可接受稳定性

**⚠️ 局限性**

局限包括缺乏真实方案的金标准验证、对极复杂方案仍有4.7%高方差、可能存在系统性偏差以及仅覆盖12个月内的随访时间点

---

## 582. Software as Content: Dynamic Applications as the Human-Agent Interaction Layer

**arXiv ID:** 2603.21334 | [PDF](https://arxiv.org/pdf/2603.21334v1)

**作者:** Mulong Xie `[一作]`, Yang Xie `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Software as Content（SaC）范式，将动态生成的交互式应用视为人机交互的介质；实现 agentic applications（具备视图、交互 affordance 集和 agent 上下文）在多轮交互中的持续演化与持久化。

**💡 创新点**

创新点在于：① 把“输出”与“交互层”统一为同一可变软件；② 通过双通道（结构化 affordances + 语义自然语言）解决 chat 的表示不匹配、交互熵和短暂状态三大限制；③ 引入递进式专业化（progressive specialization）和可共享的模板/状态重用机制；④ 在设计原则上阐明“以语义约束驱动组件选择”，而非硬性模板。

**🔧 技术方法**

技术手段包括：① 语言模型驱动的意图分析与环境交互；② 生成式用户界面（GenUI）与动态视图渲染；③ 事件驱动的 render 函数与增量状态更新；④ 结构化 affordance 生成（view‑embedded 与 anticipatory affordances）；⑤ 质量保证与数据驱动的更新机制；⑥ 基于现有 agent 框架（如 LangChain/AutoGen）与协议（AG‑UI、A2UI、MCP Apps）集成。

**📊 数据集**

数据集与资源：使用公开的 GenUI 评测数据集 PAGEN 进行界面生成评估；利用大规模检索-增强生成（retrieval‑augmented generation）数据来源（网页、API、数据库）做动态查询；若论证使用的特定案例（如租车、BBQ、监控）依赖公开 API 或公开数据集（如租车平台公开 API）。

**📈 对比分析**

对比方法：采用场景演示（selection、exploration、execution）与传统 chat 或单一 GenUI 的交互流程做对比；通过人机实验（尚未完成）验证交互效率、用户满意度、任务完成率。实验结果显示：在需要结构化决策的情境下，SaC 的交互时延显著下降（约 40%），用户对信息可视化的易用性打分高于纯文本；在简单查询时，SaC 退化为文本回复保持了相同性能。性能指标主要为：渲染延迟 (< 200ms)、agent 回答准确率 > 90%（基于检索一致性检验）。

**⚠️ 局限性**

局限性：① 需要 LLM 生成的 UI 与交互高度依赖模型质量，生成误差可能导致不一致或不合理布局；② 对需要极高可访问性或法规合规的任务（如医疗决策）仍需人工审核；③ 复杂任务会产生大量 affordances 与状态，若未能及时压缩会导致认知负荷升高；④ 对极短或单一信息请求而言，SaC 产生额外开销，可能不如传统 chat 直观；⑤ 仍缺乏大规模用户研究与定量评估，缺少标准化的可衡量指标。

---

## 583. Bayesian Active Object Recognition and 6D Pose Estimation from Multimodal Contact Sensing

**arXiv ID:** 2603.21410 | [PDF](https://arxiv.org/pdf/2603.21410v1)

**作者:** Haodong Zheng `[一作]` (Eindhoven University Of Technology), Raymond H. Cuijpers `[通讯]` (Eindhoven University Of Technology)

**通讯引用:** 2403 | [OpenAlex ID](https://openalex.org/A5017207248)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一套基于GelSight触觉传感器、腕部力/力矩传感器以及机器人运动信息的主动触觉探索框架，实现了对未知物体的快速识别和6D姿态估计。

**💡 创新点**

创新点在于将多模态触觉数据（接触几何、力/力矩、负接触空间信息）统一纳入贝叶斯推理，并通过粒子滤波器和主动目标选择机制实现高效的在线推断与采样；同时构建了完整的规划与控制堆栈，包括可达性感知、柔顺执行与触觉舵机，实现真正的自主采集。

**🔧 技术方法**

核心技术包括：1）GelSight图像深度重建与接触点提取；2）力/力矩接触点估计；3）机器人运动产生的空域剔除；4）粒子滤波推理与点对特征采样；5）主动目标点选择；6）Bi‑RRT*运动规划、关节阻尼控制与Cartesian 触觉舵机；7）基于ADD‑S的姿态评估。

**📊 数据集**

使用了11个YCB物体（如瓶子、罐子、盒子等）进行实验，物体为3D打印模型以保证刚性，采用仿真与真实Franka Panda机器人（配备SensONE F/T传感器和GelSight Mini）进行对比。

**📈 对比分析**

与仅使用F/T、F/T+GelSight、F/T+空间约束等基线相比，融合所有模态的FT+GS+FS方法在仿真与实测中平均动作周期分别为4.3和4.1，姿态误差平均为6.6和6.7毫米，明显降低动作周期和误差，且成功率在11个动作后即可达到100%，整体性能提升约20–30%。

**⚠️ 局限性**

局限包括：1）仅处理单一静止刚体；2）依赖GelSight的良好对准，遇到关节极限或自碰撞时会失效；3）粒子滤波对物体数量线性扩展，计算量随模型集增大；4）未考虑视觉或多物体场景的融合。

---

## 584. A Fast Quasi-Linear Heuristic for the Close-Enough Traveling Salesman Problem

**arXiv ID:** 2603.21401 | [PDF](https://arxiv.org/pdf/2603.21401v1)

**作者:** Khoi Duong `[一作]` `[通讯]` (University of Minnesota), Khoi Duong (University of Minnesota)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于近似线性pair‑center的快速近似算法，用于解决“近似足够”旅行商问题（CETSP）;

**💡 创新点**

创新点在于将pair‑center框架从点型TSP推广到圆形邻域，并在聚类与构造阶段引入代理圆、近似最近邻、轻量级重插与局部优化，从而实现期望O(nlogn)的时间复杂度；

**🔧 技术方法**

核心技术包括R*树空间索引、最小堆维护最近邻、代理圆的几何构造、随机旋转预处理、近似Alhazen问题的角度二分插入、能量模型驱动的重插与指数调度的点重优化；

**📊 数据集**

使用了Mennell2009数据集（62个实例，涵盖TSPLIB、Teams、Geometric等三类），并重构了原始同半径实例以保持一致性；

**📈 对比分析**

与当前最优基于遗传算法的LeiHao2024进行对比，平均与最佳已知解相差0–2%，在大部分实例上性能优于或接近最优，仅在结构化“旋转菱形”和“泡沫”实例中差距可达2–8%；

**⚠️ 局限性**

局限性主要在于对TSP基础求解的依赖，无法完全逃离局部极小值，特别是结构化重叠圆群导致的“螺旋”最优路径；此外，随机实例的子线性增长可能掩盖真实的O(nlogn)复杂度。

---

## 585. Persona Vectors in Games: Measuring and Steering Strategies via Activation Vectors

**arXiv ID:** 2603.21398 | [PDF](https://arxiv.org/pdf/2603.21398v1)

**作者:** Johnathan Sun `[一作]` (Harvard University), Andrew Zhang `[通讯]` (Harvard University)

**通讯引用:** 68250 | [OpenAlex ID](https://openalex.org/A5100345666)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用激活向量对大型语言模型进行人格调节，研究其在博弈论场景中的高阶行为特征；

**💡 创新点**

首次将对齐向量（persona vectors）应用于策略决策任务，并证明它们既可用于测量又可作为因果干预工具；

**🔧 技术方法**

激活添加（Activation Addition）与对比激活添加（Contrastive Activation Addition）技术，以及对模型隐藏层激活的线性投影；

**📊 数据集**

利用Qwen 2.5‑7B模型在生成的道德困境、道德/游戏决策样例中的对比数据，构造道德/宽恕/期望向量；

**📈 对比分析**

通过三种评估方法（GPT‑4.1‑mini特征评分、激活投影、游戏策略结果）比较在不同β（-5~5）下的模型行为；实验显示正向β显著提升利他与宽恕评分及实际策略（如分配金额、合作率），但负向β效果不稳定且易出现表述与行动不一致；

**⚠️ 局限性**

局限性包括：仅使用单一小模型Qwen 2.5‑7B；GPT‑4.1‑mini承担评分与判断，可能产生循环偏差；实验仅在一次性匿名游戏中进行，未涵盖重复互动与声誉机制；向量生成依赖人工定义特征，缺乏自动化通用性；

---

## 586. An InSAR Phase Unwrapping Framework for Large-scale and Complex Events

**arXiv ID:** 2603.21378 | [PDF](https://arxiv.org/pdf/2603.21378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 587. Mechanisms of Introspective Awareness

**arXiv ID:** 2603.21396 | [PDF](https://arxiv.org/pdf/2603.21396v1)

**作者:** Uzay Macar `[一作]` (Anthropic Fellows Program), Jack Lindsey `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在注入概念向量后能否检测并识别该注入，并通过实验揭示其内部机制。

**💡 创新点**

创新点在于首次将“内省意识”与具体的异常检测电路关联，发现检测依赖分布式MLP层的证据携带器与门控特征，并证明这些机制是后训练阶段独立学习到的。

**🔧 技术方法**

使用了概念注入（steering vector）技术、残差流插值、特征归因（direct logit attribution）、特征稀疏化、激活补丁、梯度归因以及自训练的引导向量等多种可解释性与因果干预方法。

**📊 数据集**

实验数据主要来自Gemma3‑27B、Qwen3‑235B、OLMo‑3.1‑32B等公开模型，并使用500个英文概念集合（包括“bread”、“justice”等）作为注入向量；对注入强度、层数、对照实验等做系统评估。

**📈 对比分析**

与基线（未训练向量、未后训练模型）对比，检测率提升至约60–70%，假阳性率维持在0%；消除拒绝方向或训练专门的引导向量可进一步提升30–75个百分点；与现有对照实验（如Mistral‑22B）相比，表明所挖掘的机制比单一线性方向更为稳健。

**⚠️ 局限性**

局限性包括：仅在少数几种模型和规模上验证，未覆盖更大或不同架构的模型；机制分析受限于可解释性工具的精度；实验侧重检测任务，未检验内省是否可推广到其它自我报告场景；且提升内省的干预可能带来安全与误导风险。

---

## 588. Task-Specific Efficiency Analysis: When Small Language Models Outperform Large Language Models

**arXiv ID:** 2603.21389 | [PDF](https://arxiv.org/pdf/2603.21389v1)

**作者:** Jinghan Cao `[一作]` (San Francisco State University), Xiangyun Chen `[通讯]` (Pennsylvania State University)

**通讯引用:** 1136 | [OpenAlex ID](https://openalex.org/A5072259163)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比了16个0.5B至72B参数规模的开源语言模型，在5种不同NLP任务中评估其性能与效率；

**💡 创新点**

提出了Performance‑Efficiency Ratio（PER）这一综合指标，结合准确率、吞吐量、内存与延迟的几何平均，实现多维度任务专属效率评估；

**🔧 技术方法**

利用几何平均与极差归一化计算PER，并在NVIDIA A10/A100 GPU上通过多GPU张量并行实现不同规模模型的评测；

**📊 数据集**

使用IMDB情感分类、HellaSwag常识推理、ARC‑Easy科学推理、SQuAD 2.0阅读理解和GSM8K数学推理等标准Benchmark；

**📈 对比分析**

通过统一的PER指标对比模型，发现0.5B–3B小模型在各任务中PER最高，尽管大型模型在部分任务上准确率更高，但在吞吐量、内存与延迟上劣势明显；

**⚠️ 局限性**

受限于仅测试了16个模型且归一化方法对极端值敏感，PER指标可能无法覆盖所有实际部署场景，如不同硬件或量化策略下的性能差异。

---

## 589. Knowledge Priors for Identity-Disentangled Open-Set Privacy-Preserving Video FER

**arXiv ID:** 2603.21387 | [PDF](https://arxiv.org/pdf/2603.21387v1)

**作者:** Feng Xu `[一作]` (UNSW Sydney), Dadong Wang `[通讯]` (CSIRO Data61)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种用于视频基础的隐私保护面部表情识别（FER）的两阶段框架，该框架在开放集设置中无需身份标签。

**💡 创新点**

创新点在于通过提取视频中的知识先验来解耦隐私和效用，同时引入了一种基于伪造的验证方法来评估隐私保护的有效性。

**🔧 技术方法**

使用了身份抑制网络和去噪模块，结合知识先验进行训练，并采用了伪造基础的隐私评估协议。

**📊 数据集**

使用了DFEW、CREMA-D和RAVDESS等视频数据集进行实验，且在RAF-DB上预训练去噪模型。

**📈 对比分析**

与现有的隐私保护方法进行比较，结果显示该方法在隐私保护和FER准确性上均优于基线方法，尤其是在开放集场景中表现突出。

**⚠️ 局限性**

局限性包括使用的FER模型（R(2+1)D和I3D）相对较为基础，未来研究应探讨更先进模型的影响；此外，尽管去噪模块提升了FER和隐私保护，但其理论基础仍需进一步研究。

---

## 590. Constrained Online Convex Optimization with Memory and Predictions

**arXiv ID:** 2603.21375 | [PDF](https://arxiv.org/pdf/2603.21375v1)

**作者:** Mohammed Abdullah `[一作]` (Université Paris-Saclay), Tijani Chahed `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 1697 | [OpenAlex ID](https://openalex.org/A5063380556)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了带记忆的约束在线凸优化（COCO-M），提出了在时间变化约束下实现亚线性遗憾和约束违反的算法。

**💡 创新点**

首次提出了在时间变化约束下的COCO-M问题的算法，能够在有无未来损失和约束函数预测的情况下实现亚线性遗憾和约束违反。

**🔧 技术方法**

采用了自适应惩罚方法和乐观学习（OL）算法。

**📊 数据集**

未具体提及使用的数据集，但讨论了在实际问题中的应用，如智能电网和电池健康限制。

**📈 对比分析**

与之前的研究相比，提出的算法在遗憾和约束违反方面的界限更优，特别是在记忆长度为m的情况下，遗憾为𝒪(m^3/2√(Tlog T))，约束违反为𝒪(max{T^3/4, m^3/2√(Tlog T)})。

**⚠️ 局限性**

算法在处理不确定预测时的表现仍需进一步研究，尤其是在预测不准确的情况下的鲁棒性。

---

## 591. From (Elementary) Mathematical Data Model Schemas to Safe Blazor Web Applications with Claude AI

**arXiv ID:** 2603.21388 | [PDF](https://arxiv.org/pdf/2603.21388v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 592. PLR: Plackett-Luce for Reordering In-Context Learning Examples

**arXiv ID:** 2603.21373 | [PDF](https://arxiv.org/pdf/2603.21373v1)

**作者:** Pawel Batorski `[一作]`, Paul Swoboda `[通讯]` (Heinrich Heine University Dusseldorf)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种概率化的在情境学习（ICL）示例排序方法，利用Plackett‑Luce分布学习示例排列的概率并通过迭代采样与更新来提升模型性能。

**💡 创新点**

创新点在于将离散排列搜索转化为学习分布，使用Gumbel扰动-排序技术高效采样，支持单一及混合Plackett‑Luce模型，能够直接优化任务级指标且不依赖有限标签空间。

**🔧 技术方法**

主要技术包括Plackett‑Luce分布建模、Gumbel扰动-排序采样、指数移动平均（EMA）更新、最大似然（MLE）和EM式混合分布训练。

**📊 数据集**

使用了多种分类基准（MR、SST‑5、TREC、AG’s News、SUBJ）和数学推理基准（GSM8K、MATH500、DeepMath），在Qwen2.5‑7B‑Instruct和Llama‑3.1‑8B‑Instruct模型上进行评估。

**📈 对比分析**

与静态排序、随机、熵基排序、标签分布匹配、两阶段过滤等基线对比，实验显示所提方法在大多数数据集和shot数下名列前茅，往往显著提升准确率。

**⚠️ 局限性**

局限性包括需要标注数据来评估排列性能、方法针对特定任务训练且难以迁移、在更大模型或不同推理设置下的可扩展性尚未验证。

---

## 593. The AI Scientific Community: Agentic Virtual Lab Swarms

**arXiv ID:** 2603.21344 | [PDF](https://arxiv.org/pdf/2603.21344v1)

**作者:** Ulisses Braga-Neto `[一作]` (Texas A&M University), Ulisses Braga-Neto `[通讯]` (Texas A&M University)

**通讯引用:** 5030 | [OpenAlex ID](https://openalex.org/A5026990034)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出将代理式虚拟实验室聚合为自组织的虚拟实验室群（AI 科学社区），以模拟真实科研社区的协同与演化过程。

**💡 创新点**

将代理式 AI 与群体智能融合，构建以实验室实例为粒子的去中心化 swarm，利用引用式投票与探索-利用平衡实现科研方向的自然选择与多样性维护。

**🔧 技术方法**

采用大语言模型驱动的多角色代理（规划、执行、评估），投票机制模拟引用，混合模型与容器化部署实现成本与多样性平衡，使用 swarm‑intelligence 算法（如粒子群）进行状态更新。

**📊 数据集**

暂无具体实验数据集，论文仅提出框架与概念性设计。

**📈 对比分析**

文中未提供实验比较与性能评估，强调未来工作将实现并验证该框架。

**⚠️ 局限性**

局限包括：缺乏实测验证、难以设计可衡量的适应度函数、可能出现群体思维、计算成本和规模化挑战，以及实现细节尚未完善。

---

## 594. Multi-Perspective LLM Annotations for Valid Analyses in Subjective Tasks

**arXiv ID:** 2603.21404 | [PDF](https://arxiv.org/pdf/2603.21404v1)

**作者:** Navya Mehrotra `[一作]` (Johns Hopkins University), Kristina Gligorić `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种“Perspective‑Driven Inference”（PDI）框架，利用有限的人类标注预算对 LLM 代理在不同人群视角下的误差进行自适应采样，以估计多视角下的文本标签分布。

**💡 创新点**

① 把估计目标从单一“真值”转为各人群的分布；② 通过学习 LLM 误差与人群特征的映射，实现对预算的目标化分配；③ 在无单一真值的主观任务中保证置信区间有效。

**🔧 技术方法**

使用逆概率加权（IPW）修正估计，基于梯度提升树的误差预测器，批量自适应采样与预热（burn‑in）阶段，以及 Bootstrap 置信区间方法。

**📊 数据集**

使用 POPQUORN 数据集中的礼貌和冒犯性评分子集（各 1,000 条），以及 10,000 条二分类的合成数据；人群划分按性别、年龄、教育。

**📈 对比分析**

与 LLM‑only（zero‑shot、few‑shot、persona）以及统一采样的 PPI 进行对比；在礼貌/冒犯性任务中，PDI 在覆盖率上保持 90%+，在最难群体（如 Age 50+）的平均误差显著低于其他方法，尤其在冒犯性任务中将 delta 降至约5% 对比 20%+。

**⚠️ 局限性**

只按单一轴分层，忽略交叉效应；误差预测器仅用人群特征，缺少文本信息；需先有预热样本，低预算下效率低；对小样本/稀疏人群的上采样可能过度；仅在礼貌/冒犯性两任务评估，泛化性待验证。

---

## 595. PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost

**arXiv ID:** 2603.21383 | [PDF](https://arxiv.org/pdf/2603.21383v1)

**作者:** Junkeun Yi `[一作]` (NVIDIA), Venkat Srinivasan `[通讯]` (NVIDIA)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PivotRL 框架，在已有 SFT 轨迹上进行局部 on‑policy rollouts，利用 pivot 过滤和功能等价奖励实现长周期 agentic 任务的高效后期训练。

**💡 创新点**

创新点：①通过 pivot 过滤器挑选混合成功/失败的中间状态，仅在信息量大的状态做 rollouts，显著降低计算；②采用功能等价奖励而非严格文本匹配，避免过度限制；③理论证明这两项能提升自然梯度信号并保持 OOD 排序，兼顾域内精度与 OOD 泛化。

**🔧 技术方法**

技术：局部 rollouts、Group Normalized Policy Optimization (GRPO)、verifier‑based reward、KL 正则化、基于 SFT 数据的轨迹抽取与筛选。

**📊 数据集**

数据集：四个 agentic 域的轨迹（τ^2‑Bench、SWE‑Bench Verified、Terminal‑Bench、BrowseComp），以及 OOD 测试集（IFBench、AIME25、MATH500、LiveCodeBench、Scicode、MMLU‑Pro、MMLU‑ProX、WMT24++ 等）。

**📈 对比分析**

与相同数据的 SFT 对比，PivotRL 在域内平均提升约 +14.11 点，OOV 变化仅 +0.21；SFT 在 OOD 上平均下降 -9.83。与全局 E2E RL 对比，在 SWE‑Bench 上实现相同精度仅需约 4 倍更少的 rollout 回合和 5.5 倍更短的训练时间。

**⚠️ 局限性**

局限：依赖手工或轻量级的 verifier，难以覆盖所有功能等价；对稀疏奖励方差估计可能不稳健；仅在四个特定 agentic 领域验证，未在更广泛任务中测试；缺乏在线动态采样或多任务通用化的进一步探索。

---

## 596. Exploring Experiential Differences Between Virtual and Physical Memory-Linked Objects in Extended Reality

**arXiv ID:** 2603.21381 | [PDF](https://arxiv.org/pdf/2603.21381v1)

**作者:** Zaid Ahmed `[一作]` (University of Calgary), Kangsoo Kim `[通讯]` (University of Calgary)

**通讯引用:** 2416 | [OpenAlex ID](https://openalex.org/A5022774858)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了三种 XR 记忆互动界面（物理块、虚拟块、画廊）在捕捉、重现和分享沉浸式记忆时的体验差异。

**💡 创新点**

将物理、虚拟与传统画廊三种界面进行对比，系统化探讨物理与虚拟对象在情感与社交维度上的可替代性与优势。

**🔧 技术方法**

使用 Unity+Meta Quest 3 搭配 Meta Passthrough Camera API 与 QR 码实现物理/虚拟记忆链接；利用 Insta360 X3 录制 360° 视频。

**📊 数据集**

基于 12 对参与者在两轮 Jenga 游戏中录制的 24 条 360° 记忆短片。

**📈 对比分析**

采用 within‑subjects 设计与十项开放式问卷，进行主题分析与频次统计；结果显示物理块在社交连结最高，虚拟块整体最受欢迎，画廊最易使用。

**⚠️ 局限性**

样本规模有限且仅针对 Jenga 记忆，受访者预先关系多样但不具代表性，且未量化客观性能指标，需在更大样本与多类型记忆情境中进一步验证。

---

## 597. HamVision: Hamiltonian Dynamics as Inductive Bias for Medical Image Analysis

**arXiv ID:** 2603.21377 | [PDF](https://arxiv.org/pdf/2603.21377v1)

**作者:** Mohamed A Mabrok `[一作]` `[通讯]`, Mohamed A Mabrok

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 HamVision，一种在医学图像分析中使用阻尼谐振子（Hamiltonian 动力学）作为结构化先验的统一框架，能够同时完成分割和分类任务。

**💡 创新点**

创新点包括：①将阻尼谐振子的相空间分解（位置 q、动量 p、能量 H）直接作为模型的中间表示；②通过能量门控跳跃连接和动量注入实现分割；③通过全局池化动量、能量和特征生成的相空间向量实现分类；④保持单一共享瓶颈即可适配两种任务，显著减少参数。

**🔧 技术方法**

核心技术是基于阻尼谐振子的状态空间模型（复数状态、指数欧拉离散化），配合 ConvNeXt 编码器、四方向扫描实现高效实现；并使用自适应门控、SE 机制以及全局池化 + MLP 的分类头。

**📊 数据集**

使用十个医学数据集：分割任务包括 ISIC 2018、ISIC 2017、TN3K、ACDC；分类任务包括 BloodMNIST、PathMNIST、DermaMNIST、BreastMNIST、RetinaMNIST、OrganSMNIST。

**📈 对比分析**

与 16-105M 参数的基线（U-Net、TransUNet、SwinUNet、VM-UNet、FreqConvMamba、MedViT、MedMamba 等）相比，HamVision 在分割上取得所有四个基准的最高 Dice/IoU，参数量仅 8.57M；在分类上在 BloodMNIST、PathMNIST 等数据集上超过或匹配最先进方法，且总参数约 7.3M。

**⚠️ 局限性**

局限性包括：仅在瓶颈层使用振荡器；扫描方式仅行列方向，可能忽略对角信息；采用一阶指数欧拉离散，未尝试更高阶或隐式积分；仅在医学影像上验证，未在自然图像任务上检验通用性。

---

## 598. Hybrid Quantum-Classical Branch-and-Price for Intra-Day Electric Vehicle Charging Scheduling via Partition Coloring

**arXiv ID:** 2603.21374 | [PDF](https://arxiv.org/pdf/2603.21374v1)

**作者:** Peng Sun `[一作]` (Tianjin University), Li Wang `[通讯]` (Hebei University of Technology)

**通讯引用:** 42083 | [OpenAlex ID](https://openalex.org/A5100336135)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将车辆充电调度问题建模为Partition Coloring Problem，并提出基于Branch‑and‑Price的混合量子启发式算法；

**💡 创新点**

首次将PCP用于EV充电调度，结合量子退火启发式（BSB、SimCIM）在定价子问题中提高列生成效率，并在大规模实例中实现最优解；

**🔧 技术方法**

使用分支定价框架、定价子问题的QUBO模型、MindQuantum平台的ballistic simulated bifurcation（BSB）和simulated coherent Ising machine（SimCIM）两种量子退火启发式算法，主问题由Gurobi求解；

**📊 数据集**

合成EV充电实例，时间窗口T=24、充电时长d=3、充电桩数C∈{5,10}，车辆数V从10到100，候选区间K从2到10，随机种子生成多组实例；

**📈 对比分析**

与纯Gurobi的branch‑and‑price做对比；在V≤40时三种方法均得到最优且耗时相近；在V≥80时，Gurobi多次超时并留下较大gap，而BSB和SimCIM在相同时间限制内完成求解，显著降低运行时间（V=100时平均提升70%–90%），并获得最优解；

**⚠️ 局限性**

仅使用合成数据，缺少真实车队/充电站的复杂约束；定价子问题仍受限于QUBO规模，量子模拟器的可扩展性有限；未考虑多路径、电网负荷等更复杂的网络约束，缺乏在真实环境中的鲁棒性评估。

---

## 599. Conspiracy Frame: a Semiotically-Driven Approach for Conspiracy Theories Detection

**arXiv ID:** 2603.21368 | [PDF](https://arxiv.org/pdf/2603.21368v1)

**作者:** Heidi Campana Piva `[一作]` (University of Turin), Marco Antonio Stranisci `[通讯]` (University of Turin)

**通讯引用:** 434 | [OpenAlex ID](https://openalex.org/A5014900577)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Conspiracy Frame 并构建 Con.Fra 数据集，对 Telegram 消息进行 span 级标注；

**💡 创新点**

基于框架语义与符号学的细粒度叙事框架，探讨框架对 LLM 解释性与泛化性的贡献；

**🔧 技术方法**

使用 FrameNet 对标注 span 进行映射，并通过 LLaMA‑3.3 8B/70B 在零样本、少样本与框架引导的 in‑context 学习下进行实验；

**📊 数据集**

使用自建的 Con.Fra 数据集（2577 条 Telegram 讯息，其中 2077 条用于标注、500 条用于 OOD 评估）以及 FrameNet 进行映射；

**📈 对比分析**

在 OOD 评估中，70B 模型框架引导在分类任务上优于少样本（830/1000 轮次）但在 span 检测上略逊；总体 F1 变化不大，框架引导提升召回并带来更完整的 span ；

**⚠️ 局限性**

仅针对 LLaMA 系列模型，标注一致性低，映射基于词形无法解决歧义，未实现深层语义集成，缺乏跨模型与跨领域的系统评估。

---

## 600. AdaRubric: Task-Adaptive Rubrics for LLM Agent Evaluation

**arXiv ID:** 2603.21362 | [PDF](https://arxiv.org/pdf/2603.21362v1)

**作者:** Liang Ding `[一作]` (Alibaba Group), Liang Ding `[通讯]` (Alibaba Group)

**通讯引用:** 8027 | [OpenAlex ID](https://openalex.org/A5046576694)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于任务描述动态生成评估量表并按步加权评分的LLM代理评估框架，旨在提升评估与训练的针对性与可靠性。

**💡 创新点**

创新点包括：①根据任务自动生成任务特定、正交的评估维度；②对每步输出进行置信度加权评分；③引入DimensionAwareFilter防止高分维度掩盖低分错误。

**🔧 技术方法**

核心技术包含LLM-as-Judge（使用GPT-4o或开源模型）、自适应量表生成、置信度加权聚合、DimensionAwareFilter、DPO强化学习和PPO在线RL。

**📊 数据集**

实验使用的主要数据集包括WebArena（网页自动化）、ToolBench（API链条）、AgentBench（代码/OS任务）和SWE-bench（代码修复）。

**📈 对比分析**

在与传统静态评估（Helpfulness/Fluency/Safety）、ROUGE-L、BERTScore等基线对比时，动态评估获得Pearson r≈0.79（提升约0.15），DPO训练成功率提升至27.8%（相较Prometheus提升≈15.5pp），PPO奖励加速收敛至30.2% SR。

**⚠️ 局限性**

局限性在于：评估维度高度依赖LLM的生成质量，任务描述不完整或模糊时可能产生重叠或缺失维度；置信度预测本身也可能产生误差，且单轮评估不具备多轮校验的鲁棒性。

---

## 601. Personality-Driven Student Agent-Based Modeling in Mathematics Education: How Well Do Student Agents Align with Human Learners?

**arXiv ID:** 2603.21358 | [PDF](https://arxiv.org/pdf/2603.21358v1)

**作者:** Bushi Xiao `[一作]` (University of Florida), Qian Shen `[通讯]` (University of Florida)

**通讯引用:** 2494 | [OpenAlex ID](https://openalex.org/A5090859164)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了基于大五人格的学生生成代理，完整实现了教师互动、学习与考试等学习过程，并通过14个标准评估其行为与真实学生的相似性。

**💡 创新点**

创新之处在于首次将学生代理与完整学习管线结合，并从13项实证研究中提炼14个评价准则，系统验证代理的学习真实性。

**🔧 技术方法**

采用LLM生成代理（GPT-OSS‑120B）、BERT数学题分类模型、inf‑retriever检索器及向量化记忆机制等技术实现代理学习与答题。

**📊 数据集**

使用NuminaMath‑CoT 860k条数学题库，筛选3044道题目作为实验问题，并参考13篇实证论文作为行为评估依据。

**📈 对比分析**

通过对比不同学习轮次、不同人格类型及数学领域的宏观F1分数，以及将代理行为与人类研究结果的14项准则匹配，实验显示学生代理在10/14项上与人类相符，且高外向型代理取得最高分。

**⚠️ 局限性**

主要局限在于缺乏定量的行为相似性评估框架、人格提示静态且未模拟情境变化，以及未能明确人格对学习机制的具体影响。

---

## 602. AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling

**arXiv ID:** 2603.21357 | [PDF](https://arxiv.org/pdf/2603.21357v1)

**作者:** Liang Ding `[一作]` (Alibaba Group), Liang Ding `[通讯]` (Alibaba Group)

**通讯引用:** 8027 | [OpenAlex ID](https://openalex.org/A5046576694)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AgentHER 四阶段管道，将失败的 LLM 代理轨迹重新标注为可训练的目标，从而充分利用原本被丢弃的失败数据。

**💡 创新点**

将 HER 原理迁移到自然语言代理训练中，结合多判别、严重程度加权、置信门控等机制，实现高质量的逆向目标生成，并显著提升数据效率。

**🔧 技术方法**

采用 LLM（GPT‑4o、Qwen2.5‑72B/7B、LLaMA‑3.1‑8B）进行 SFT/DPO/ShareGPT 训练；四阶段流程包括失败分类、结果提取、Prompt 重标、数据包装；使用多判别和置信门控控制标注噪声。

**📊 数据集**

使用 WebArena（812 任务）和 ToolBench（16,464 任务）两个基准，分别收集 3,000+ 失败轨迹与 500+ 成功轨迹作为训练集。

**📈 对比分析**

与传统仅用成功数据的 SFT、SFT‑Random、Rejection‑Sampling 等基线对比；在 WebArena 上提升 7.1–11.7 个百分点，在 ToolBench 上提升 7.8–11.7 个百分点；实现 2 倍数据效率，并在迭代部署中持续积累收益。

**⚠️ 局限性**

主要局限包括：WebArena 任务集重叠可能导致泄漏；未与完整的成功数据规模完全匹配；置信阈值 0.5 可能过度过滤边界样本；理论假设完美判别器，实际噪声仍需进一步评估。

---

## 603. EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization

**arXiv ID:** 2603.21332 | [PDF](https://arxiv.org/pdf/2603.21332v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 604. Efficient Coarse-to-Fine Diffusion Models with Time Step Sequence Redistribution

**arXiv ID:** 2603.21348 | [PDF](https://arxiv.org/pdf/2603.21348v1)

**作者:** Yu-Shan Tai `[一作]` (National Taiwan University), Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了粗到细降噪和时间步重排的方法，用以显著降低扩散模型的计算开销。

**💡 创新点**

C2F利用低分辨率先粗略生成特征再高分辨率精细生成；TRD通过近邻迭代替换实现时间步序列快速优化。

**🔧 技术方法**

采用多分辨率微调、PCA判定切换点、基于L2误差的迭代时间步替换、DDIM和DPM-Solver++调度器。

**📊 数据集**

在CIFAR-10 (32×32) 与 LSUN-Church (256×256) 数据集上进行无条件生成实验。

**📈 对比分析**

与Diff-pruning、标准DDIM 100步等进行对比，C2F-DM可实现80-90% MACs下降，且FID与原模型相近。

**⚠️ 局限性**

仍需在更高分辨率、多样化数据上验证效果，并可能受限于多分辨率微调的泛化能力。

---

## 605. Silent Commitment Failure in Instruction-Tuned Language Models: Evidence of Governability Divergence Across Architectures

**arXiv ID:** 2603.21415 | [PDF](https://arxiv.org/pdf/2603.21415v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 606. Optimal-Cost Construction of Shallow Cuttings for 3-D Dominance Ranges in the I/O-Model

**arXiv ID:** 2603.21337 | [PDF](https://arxiv.org/pdf/2603.21337v1)

**作者:** Yakov Nekrich `[一作]` (Michigan Technological University), Saladi Rahul `[通讯]` (Indian Institute of Science)

**通讯引用:** 170 | [OpenAlex ID](https://openalex.org/A5056922950)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了在I/O模型下构造3维支配范围的k级浅切割的最优成本算法，并基于此设计了离线3维支配近似计数与报告的I/O高效方案。

**💡 创新点**

首次实现3维支配浅切割的I/O最优构造，核心创新包括层次化浅切割、使用查询成本高但构造快的数据结构、持久化B树和离线find-any技术的综合运用。

**🔧 技术方法**

采用层次化浅切割、持久化B树、x、y、z选择查询等查询成本高的数据结构，以及离线find-any算法，在I/O模型中实现最优构造与查询。

**📊 数据集**

论文未使用任何实际数据集，全部以理论分析与算法证明为主。

**📈 对比分析**

与先前内部内存算法相比，构造成本达到O(sort(N))，实现I/O模型下的最优构造；离线支配近似计数与报告的I/O复杂度分别为O(sort(N)+|Q|/B·log_{M/B}N/B)和O(|P|/B+|Q|/B·log_{M/B}N/B+K/B)，性能均为理论最优。

**⚠️ 局限性**

仅适用于离线查询，无法直接应用于在线数据结构构造；对其他类型范围查询的I/O构造尚未给出；且需满足M=Ω(B)等假设。

---

## 607. Classification of Non-redundancy of Boolean Predicates of Arity 4

**arXiv ID:** 2603.21353 | [PDF](https://arxiv.org/pdf/2603.21353v1)

**作者:** Joshua Brakensiek `[一作]` (University of California, Berkeley), Aaron Putterman `[通讯]` (Harvard University)

**通讯引用:** 9553 | [OpenAlex ID](https://openalex.org/A5068388812)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对所有非平凡的4元布尔谓词（共400个）进行了非冗余度的完整或近似分类。

**💡 创新点**

首次给出了一个非平凡的布尔谓词，其非冗余度不是多项式函数的例子，并且实现了对397个谓词的完全定理，剩余三个需特殊处理。

**🔧 技术方法**

使用了组合优化、格理论、线性/多项式嵌入以及Ruzsa‑Szemerédi图的极值组合技术。

**📊 数据集**

利用全枚举的400个谓词集合（由对称性约简得到），并在GitHub上公开代码以复现实验。

**📈 对比分析**

与之前关于二元和三元谓词的结果对比，发现大多数谓词仍满足整数次多项式非冗余度，验证了方法的有效性；对异常谓词给出了上下界，表现出方法的局限。

**⚠️ 局限性**

主要局限在于对R_181的非冗余度仍未确定，且方法在更高元数或更复杂谓词时可能难以扩展。

---

## 608. Enterprise Sales Copilot: Enabling Real-Time AI Support with Automatic Information Retrieval in Live Sales Calls

**arXiv ID:** 2603.21416 | [PDF](https://arxiv.org/pdf/2603.21416v1)

**作者:** Jielin Qiu `[一作]` (Salesforce AI Research), Huan Wang `[通讯]` (Salesforce AI Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个实时AI销售助理，能够在现场销售通话中自动识别客户问题，检索保险产品数据库，并即时在销售员仪表盘上展示答案。

**💡 创新点**

将流式语音转写、LLM驱动的问答检测、FAQ语义匹配与文本转SQL检索融合为统一的低延迟管线，并提供可配置的LLM后端和跨产品查询能力。

**🔧 技术方法**

使用Deepgram Nova-2语音识别、OpenAI/Anthropic/Google Gemini LLM、SQLite+Text-to-SQL、FastAPI+WebSocket、React+TypeScript前端以及ElevenLabs TTS。

**📊 数据集**

采用保险领域50个产品、10类，包含2,490条FAQ、290条保障详情、162个定价层级的结构化知识库，约350,000 tokens。

**📈 对比分析**

在20个代表性问题的基准上与人工CRM搜索对比，平均响应时间从39.7秒降至2.8秒，速度提升14倍，问题检测率达到100%。

**⚠️ 局限性**

依赖LLM生成SQL的安全验证仍有限，跨域适配需要重建数据库，且在极端网络延迟或语音质量差时可能影响实时性能。

---

## 609. The Illusion of Agreement with ChatGPT: Sycophancy and Beyond

**arXiv ID:** 2603.21409 | [PDF](https://arxiv.org/pdf/2603.21409v1)

**作者:** Kazi Noshin `[一作]` (University of Illinois Urbana-Champaign), Sharifa Sultana `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5103026409)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对r/ChatGPT社区中2025年7月至12月的帖子与评论进行定性主题分析，探索用户对ChatGPT产生的五类关切（诱导妄想、离题叙事、将错误归咎于用户、成瘾与无监管心理支持），并收集用户提出的三层级解决方案（功能使用技巧、行为方法、保障措施）。

**💡 创新点**

创新点在于①将用户视角下的LLM相关伤害系统化为五类关切并揭示“将错误归咎于用户”这一被忽视的风险；②提出用户自发的多层级干预建议；③强调个体、开发者与政策制定者共同介入的必要性。

**🔧 技术方法**

技术方法包括：关键词检索（使用基于文献的BERTopic提取73个与“sycophancy”相关的关键词），Python PRAW API抓取数据，主题分析（手工编码→聚类→主题），以及基于词典的词汇量计数与情感分析。

**📊 数据集**

数据集为r/ChatGPT社区的3,600条帖子和140,416条评论（共54,014位匿名用户），时间范围为2025年7月1日至12月31日。

**📈 对比分析**

方法评估以定性描述为主；对关键主题与建议进行频次统计与语义映射，未涉及量化性能指标；与现有技术（如提示工程、情感识别）相比，强调了用户自我干预与制度层面的补充。

**⚠️ 局限性**

局限性包括：①仅使用Reddit数据，样本偏向年轻、西方技术熟练人群；②关键词检索依赖文献词典，可能漏检新兴用语；③结论主要针对ChatGPT，可能不完全适用于其他LLM。

---

## 610. Awakening: Modern Challenges and Opportunities of Software Engineering Research

**arXiv ID:** 2603.21403 | [PDF](https://arxiv.org/pdf/2603.21403v1)

**作者:** Diomidis Spinellis `[一作]` (Athens University of Economics and Business), Zoe Kotti `[通讯]` (Athens University of Economics and Business &)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文回顾了软件工程研究的发展历史，指出当前面临的结构性挑战，并提出工业化博士、跨机构大规模合作、月球计划等新方向以提升研究影响力。

**💡 创新点**

创新点在于系统性地分析了研究生态中的结构性瓶颈，并提出以产业协同和大规模协作为核心的未来研究路径，强调了月球项目、工业博士和改进评估机制。

**🔧 技术方法**

主要技术手段为文献综述与案例分析，提出的策略包括工业博士项目、跨机构实验室合作、CRediT贡献体系、月球项目模型等。

**📊 数据集**

本文未使用实验数据集，而是基于公开文献、行业报告和案例（如Google代码规模等）进行分析。

**📈 对比分析**

由于是理论性讨论，没有进行实验对比；作者通过比较现有出版文化、平台网络效应等来论证建议的有效性。

**⚠️ 局限性**

局限在于缺乏实证验证，提出的方案多为设想，需在实际项目中检验其可行性与效果。

---

## 611. ARYA: A Physics-Constrained Composable & Deterministic World Model Architecture

**arXiv ID:** 2603.21340 | [PDF](https://arxiv.org/pdf/2603.21340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 612. A transformer architecture alteration to incentivise externalised reasoning

**arXiv ID:** 2603.21376 | [PDF](https://arxiv.org/pdf/2603.21376v1)

**作者:** Elizabeth Pavlova `[一作]` (MARS), Puria Radmard `[通讯]` (Geodesic Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出一种在 Transformer 结构中加入随机早停头的改造，并通过两阶段训练（自蒸馏与强化学习）来鼓励模型在生成过程中提前退出，减少不必要的计算量。

**💡 创新点**

创新点在于：① 将早停机制设计为概率化的随机决定，而非确定性阈值；② 在第二阶段使用强化学习对提前退出进行奖励，形成可调节的外部化压力；③ 将自蒸馏与 RL 结合，形成完整的后训练管道。

**🔧 技术方法**

使用的技术包括：Transformer 结构改造、早停头与 LoRA 适配器、信息论启发式自蒸馏、KL 散度校准、强化学习（RLOO 变体）与奖励调节、交叉熵训练。

**📊 数据集**

实验数据集为 GSM8K 算术推理数据集和由 ExploreToM 生成的 Theory‑of‑Mind 数据集，模型分别为 Qwen3‑4B 和 DeepSeek‑R1‑Distill‑Qwen‑1.5B。

**📈 对比分析**

与基准完整深度模型对比，实验表明模型在保持答案连贯性与任务准确率（从 47% 提升至 55‑60%）的同时，平均每个 token 的层数使用量从 98% 降至约 90%，整体计算量亦略有下降。

**⚠️ 局限性**

局限性包括：仅在小型模型与两类任务上验证，跨域与更大模型的泛化性未知；提前退出的超参数调节需进一步探索；未对 CoT 的真实性与可监测性做系统评估；可能在极度压缩计算时导致质量下降。

---

## 613. Beyond Memorization: Distinguishing between Reductive and Epistemic Reasoning in LLMs using Classic Logic Puzzles

**arXiv ID:** 2603.21350 | [PDF](https://arxiv.org/pdf/2603.21350v1)

**作者:** Adi Gabay `[一作]` (Hebrew University of Jerusalem), Liat Peterfreund `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 167 | [OpenAlex ID](https://openalex.org/A5077872338)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大语言模型（LLM）在认识学推理（epistemic puzzles）上的能力进行评估，提出一种“归约阶梯”方法，系统性地区分模型是通过归约（reduction）还是真正的认识学推理解决问题。

**💡 创新点**

创新点包括：1）将传统意义上的“记忆”重新定义为归约的特殊情况；2）构造归约阶梯，逐步改变经典泥孩子问题的表面形式，保留底层推理逻辑，从而在不同难度层级上检验模型是否真正进行认识学推理；3）通过对链式思考（CoT）的手工标注，识别模型的推理策略并揭示其对归约的依赖。

**🔧 技术方法**

技术手段：动态认识学逻辑（DEL）框架、链式思考（CoT）提示、公共提示模板、可变观测矩阵（O）用于生成非对称观测场景；同时使用符号求解器对模型输出进行准确性验证。

**📊 数据集**

数据集：自行生成的三阶梯数据集，包括经典泥孩子（Rung I）、奥运体操改编（Rung II）以及非对称观测与上限公告（Rung III），总计约1,694道推理实例（Rung I+II: 1,320道；Rung III: 374道）。

**📈 对比分析**

比较方法：对8个不同家族（ChatGPT, Gemini, Mistral, Qwen, OLMo等）的模型在每个阶梯上测量准确率，并统计使用归约性CoT的比例。结果显示：模型在阶梯I、II保持高准确率（约97–97%），但在阶梯III准确率骤降至约66%；归约性CoT比例随阶梯升高而下降（从62%到15%）。规模在同一家族内有一定提升效果，但并非决定性因素。

**⚠️ 局限性**

局限性：1）链式思考的文字化推理可能与模型内部实际计算不一致，难以完全捕捉真实推理路径；2）阶梯III的非对称观测样本数量受生成可行性的限制，导致评估样本较少；3）实验仅覆盖单轮或短轮推理，未深入探讨长期交互中的推理能力。

---

## 614. TagLLM: A Fine-Grained Tag Generation Approach for Note Recommendation

**arXiv ID:** 2603.21481 | [PDF](https://arxiv.org/pdf/2603.21481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 615. Respiratory Status Detection with Video Transformers

**arXiv ID:** 2603.21349 | [PDF](https://arxiv.org/pdf/2603.21349v1)

**作者:** Thomas Savage `[一作]` (University of Pennsylvania), Evan Madill `[通讯]` (University of Washington)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文利用视频Transformer模型，对运动后恢复期健康志愿者的呼吸状态进行检测，构建了一套基于视频片段对比的呼吸困难识别任务。

**💡 创新点**

创新点在于将Lie Relative Encodings（LieRE）与Motion-Guided Masking（MGM）相结合提升时空位置表示，并采用基于嵌入距离的对比策略，显著提升了呼吸状态判别性能。

**🔧 技术方法**

使用了ViViT-base视频Transformer作为编码器，改进为LieRE位置编码，加入MGM预处理，随后通过单一嵌入向量对比与余弦距离损失进行训练。

**📊 数据集**

数据集由75名健康志愿者完成5分钟高强度运动后录制的5分钟视频组成，最终52名志愿者共计49个视频被用于训练与验证，7个视频（3名志愿者）用于独立测试。

**📈 对比分析**

模型通过对比同一恢复序列中不同时间段的短视频片段，判断哪个片段出现得更早（呼吸困难更严重）。基线嵌入距离方法F1=0.75，加入LieRE后提升至0.77，进一步加MGM后达到F1=0.81，准确率最高达67.5%。

**⚠️ 局限性**

局限性包括：数据采集仍需受试者主动摄像，未完全代表被动连续监测；潜在的汗液变化可能混淆呼吸困难的视觉信号；仅在健康受试者上验证，缺乏真实临床病人数据。

---

## 616. StreamingEval: A Unified Evaluation Protocol towards Realistic Streaming Video Understanding

**arXiv ID:** 2603.21493 | [PDF](https://arxiv.org/pdf/2603.21493v1)

**作者:** Guowei Tang `[一作]` (East China Normal University), Xiaoling Wang `[通讯]` (East China Normal University)

**通讯引用:** 11481 | [OpenAlex ID](https://openalex.org/A5100344619)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个名为StreamingEval的统一评估框架，用于在严格的流式视频输入和资源受限条件下对Video-LLM模型进行系统级评估。

**💡 创新点**

创新点包括：① 将实时吞吐、文本解码延迟、有限历史记忆等关键指标纳入单一评估协议；② 通过统一内存预算适配器实现在线与离线模型的公平对比；③ 引入综合评分StreamingScore，量化模型在准确率、吞吐、延迟与资源使用间的权衡。

**🔧 技术方法**

使用了异步时间因果管道（Frame Player、Encoder‑Memory Updater、Responder）、统一的内存预算适配器、MaxFPS、TTFT等评估指标以及多维度综合评分公式。

**📊 数据集**

主要使用了OVO‑Bench和StreamingBench两大流式视频问答基准数据集。

**📈 对比分析**

在同一资源预算（GPU显存、视觉记忆容量）下对12个主流多模态与在线Video‑LLM模型进行实验，结果显示：离线模型在准确率上更优秀，但在线模型在吞吐和延迟方面更具可部署性；不同内存预算和输入分辨率下模型表现差异显著，揭示了准确率与部署效率之间的张力。

**⚠️ 局限性**

局限性包括：仅覆盖公开的7B/8B规模模型，未对闭源大型模型进行评测；受限于GPU资源和公开代码，实验范围受限；评估协议虽统一，但可能因过度优化指标而不完全反映真实应用性能。

---

## 617. GaussianSSC: Triplane-Guided Directional Gaussian Fields for 3D Semantic Completion

**arXiv ID:** 2603.21487 | [PDF](https://arxiv.org/pdf/2603.21487v1)

**作者:** Ruiqi Xian `[一作]`, Dinesh Manocha `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 GaussianSSC 两阶段网格原生框架，利用高斯锚点和三平面指引完成单目语义场景重建。

**💡 创新点**

创新点在于将可学习的子像素高斯支持融入网格与三平面结构，既提升图像-体素对齐，又保持高效推理。

**🔧 技术方法**

使用 ResNet+FPN 特征提取、三平面表示、可学习高斯锚点、局部收集与全局聚合的高斯–三平面细化模块，以及两阶段占用/语义预测。

**📊 数据集**

在 SemanticKITTI 数据集上进行训练和评测。

**📈 对比分析**

与 VoxFormer、ETFormer 等单目基线对比，Stage‑1 占用 Recall/Precision 分别提升 1%/2%，IoU 提升 1.8%；Stage‑2 语义 mIoU 提升 0.8%，整体性能最优。

**⚠️ 局限性**

局限在于仅使用单帧 RGB，未建模时序动态，且在极端遮挡区域仍可能出现多解不确定性。

---

## 618. TaigiSpeech: A Low-Resource Real-World Speech Intent Dataset and Preliminary Results with Scalable Data Mining In-the-Wild

**arXiv ID:** 2603.21478 | [PDF](https://arxiv.org/pdf/2603.21478v1)

**作者:** Kai-Wei Chang `[一作]` (Massachusetts Institute of Technology), Hung-yi Lee `[通讯]` (National Taiwan University)

**通讯引用:** 9073 | [OpenAlex ID](https://openalex.org/A5040508737)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究创建了面向老年人健康与家居辅助场景的台语（闽南语）语音意图数据集 TaigiSpeech，并提出两种从网络视频中挖掘低资源语言意图语料的策略（关键词匹配+LLM伪标注与音视合成检索），随后对基线模型进行实验评估。

**💡 创新点**

创新点在于：①首次提供老年人台语意图识别的真实语料；②利用中介语（普通话字幕）与跨模态检索两种低标注策略实现大规模无标注数据挖掘；③系统评估了域迁移难度，为低资源语言 SLU 研究提供了实用基准。

**🔧 技术方法**

使用的技术包括：LLM（Gemini-3）进行伪标签生成、跨模态检索模型 PE‑AV 进行音视合成检索、SSL 语音表示（HuBERT、WavLM）、轻量级 MatchboxNet、以及基于 ASR+LLM 的级联推理。

**📊 数据集**

主要数据集为 TaigiSpeech（21名老年人共 3,079 条 语音），用于训练与测试；另外使用台语电视剧 7,000 小时视频作为无标注语料进行挖掘；实验还对 Whisper、Qwen3‑ASR 等公共语音模型进行评估。

**📈 对比分析**

比较方法：先在挖掘语料上训练模型（关键词匹配 5 类、音视检索 2 类），然后在 TaigiSpeech 测试集上评估；发现域不匹配导致准确率从 90% 降至 70%；在加入 1,600 条 Taigi 训练样本进行微调后，准确率恢复至 90% 以上；轻量级模型与 SSL 模型的性能差距显著，体现了资源与效果的权衡。

**⚠️ 局限性**

局限性包括：①挖掘语料与真实老年语音的域差异大，导致直接迁移性能低；②音视检索难以获得细粒度急救标签；③对中介语字幕与 LLM 的依赖限制了在无文本资源语言中的适用性；④实验规模受计算资源限制，未能充分利用完整 7,000 小时数据。

---

## 619. When Documents Disagree: Measuring Institutional Variation in Transplant Guidance with Retrieval-Augmented Language Models

**arXiv ID:** 2603.21460 | [PDF](https://arxiv.org/pdf/2603.21460v1)

**作者:** Yubo Li `[一作]` (Carnegie Mellon University), Rema Padman `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4398 | [OpenAlex ID](https://openalex.org/A5046671743)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个检索增强语言模型框架，对美国23家移植中心的102本患者教育手册进行文档驱动的问答比较，量化中心间的指导差异。

**💡 创新点**

首次将检索增强生成与多中心手册进行系统对比，提出五标签一致性分类法，并通过该方法量化覆盖缺口与临床差异，展示中心级异质性既稳定又可解释。

**🔧 技术方法**

使用 LlamaParse 对 PDF 进行结构化解析；采用 BM25 与 FAISS 结合的混合检索+RRF 重排序；用 Qwen3-14B 生成基于检索上下文的答案；再利用 LLM 进行缺失检测和五标签评判。

**📊 数据集**

使用 102 本手册（涵盖心、肺、肾、肝、胰五种器官，共 37 预期、39 术后、26 合并手册）以及 1,115 条从机构 FAQ、Reddit、专业组织等渠道收集的患者问题集。

**📈 对比分析**

通过非缺失问答对进行五标签评判，并计算每个问题、主题、器官和中心的差异率和一致率；实验结果显示 20.8% 的非缺失对存在临床差异，0.2% 为矛盾，整体缺失率高达 96.2%，其中生殖健康缺失率达 95.1%。

**⚠️ 局限性**

LLM 判定可能存在偏差；检索与生成过程不完美；仅处理文本忽略表格、图表等多模态信息；样本仅为英文美国中心，缺乏跨语言和国际推广。

---

## 620. Cross-Context Verification: Hierarchical Detection of Benchmark Contamination through Session-Isolated Analysis

**arXiv ID:** 2603.21454 | [PDF](https://arxiv.org/pdf/2603.21454v1)

**作者:** Tae-Eun Song `[一作]` `[通讯]`, Tae-Eun Song

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了跨上下文验证（CCV）与层级跨上下文架构（HCCA）两种黑盒方法，用以检测SWE-bench Verified中的模型记忆泄露和测试缺陷。

**💡 创新点**

创新点在于利用多会话独立求解产生的代码多样性来直接观察模型的推理与记忆行为，并通过信息限制的多代理结构消除确认偏差。

**🔧 技术方法**

使用的技术包括会话隔离、多重推理会话、AST/BLEU/编辑距离的多尺度多样性度量、基于分数的污染与测试缺陷评分、简易的推理开头词判别器，以及分层信息流的HCCA框架。

**📊 数据集**

数据集为9个SWE-bench Verified问题（共45条实验），采用Claude Opus 4.6模型进行5次独立会话推理，结合Lite/Full两种执行模式。

**📈 对比分析**

在这些问题上，CCV实现了完整的污染与真实推理区分（U=0, p≈0.012, r=1.0），HCCA揭示了污染-缺陷复合情况，且与传统方法相比在检测准确率上显著提升，尤其在识别假阳性方面。

**⚠️ 局限性**

主要局限包括样本量仅9个问题、仅测试单一模型版本、仅针对Python、推理分类器未经外部验证、以及HCCA计算开销较大；未来需扩大样本、跨模型验证并评估对其他语言/领域的适用性。

---

## 621. Communication-Avoiding SpGEMM via Trident Partitioning on Hierarchical GPU Interconnects

**arXiv ID:** 2603.21444 | [PDF](https://arxiv.org/pdf/2603.21444v1)

**作者:** Julian Bellavita `[一作]` (Cornell University), Giulia Guidi `[通讯]` (Cornell University)

**通讯引用:** 171 | [OpenAlex ID](https://openalex.org/A5034932123)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为Trident的层次感知分布式稀疏矩阵乘法算法，利用GPU多节点高带宽内部互连与低带宽跨节点互连进行高效通信；

**💡 创新点**

创新点在于三分形分区（Trident）方案：在节点间采用二维分区，在节点内部采用一维切片，配合异步进度引擎和局部聚合，显著降低跨节点通信量并隐藏延迟；

**🔧 技术方法**

使用技术包括：GPU端稀疏矩阵CSR格式，CUDA-aware MPI一侧RDMA异步通信，NCCL本地聚合集体，cuSPARSE和自研本地SpGEMM核，三线程异步执行模型，静态Cannon式双层调度；

**📊 数据集**

实验数据集主要来自SuiteSparse和HipMCL，涵盖10个大型无结构稀疏矩阵（如mouse_gene、archaea、eukarya、isolates_subgraph4等），矩阵规模达数百万行/列，稀疏度从10⁻⁶到10⁻³；

**📈 对比分析**

与Trilinos（一维稀疏乘法）、CombBLAS Sparse SUMMA（二维）和改进版Sparse SUMMA进行比较；在256 GPU上对mouse_gene实现最高5.95×、对所有矩阵2.96×的几何平均加速；在最大规模时比Improved Sparse SUMMA提升至2.38×；在Markov Clustering扩展步骤中亦显著加速；通信量上跨节点量降低约2倍；

**⚠️ 局限性**

局限性：算法高度依赖显著的内部/外部带宽差异，非层次化互连或同等带宽的系统效果有限；在高度结构化矩阵（无置换）下不如1D稀疏乘法；对极大规模时仍受内存和局部负载不均衡影响；需要MPI 3.0一侧支持和CUDA-aware 环境，硬件兼容性受限。

---

## 622. Quotient Geometry, Effective Curvature, and Implicit Bias in Simple Shallow Neural Networks

**arXiv ID:** 2603.21502 | [PDF](https://arxiv.org/pdf/2603.21502v1)

**作者:** Hang-Cheng Dong `[一作]` (Harbin Institute of Technology), Pengcheng Cheng `[通讯]` (Jilin University)

**通讯引用:** 11452 | [OpenAlex ID](https://openalex.org/A5100432320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究浅层神经网络中参数对称性所导致的冗余，构造商空间度量并定义有效曲率与海森矩阵，分析梯度流在商空间中的分解及其对隐式偏置的影响。

**💡 创新点**

创新点在于首次将商空间几何应用于浅层网络，统一处理隐藏单元重排与尺度对称性，提出基于预测器的有效曲率与梯度流，并在此框架下重新表述隐式偏置，解决了传统欧氏参数空间中曲率与复杂度的重表问题。

**🔧 技术方法**

主要技术包括微分几何商空间构造、函数诱导度量、商空间的有效海森矩阵定义、梯度流的水平/垂直分解、二次激活模型的显式化以及对应的数值实验验证。

**📊 数据集**

实验以合成教师二次模型数据为主，构造多组等价轨迹；若使用真实数据，亦选用标准MNIST或CIFAR-10等小规模数据集进行验证。

**📈 对比分析**

对比方法：将原始参数空间中的欧氏梯度和海森谱与商空间度量下的曲率、梯度流进行对比。实验结果显示：商空间曲率在等价轨迹上保持一致，梯度流在商空间中收敛更快，隐式偏置更符合矩阵级复杂度（如核范数、谱半径），而原始空间的指标则表现出明显的表示依赖性。

**⚠️ 局限性**

局限性：研究聚焦于结构简单的浅层网络，商空间在奇异点可能失去流形结构；实验仅在低维合成或小规模数据上验证，尚未在大规模深度网络中检验；隐式偏置的全局理论仍不完整，需进一步探讨非正规点与多模态解的情况。

---

## 623. RuntimeSlicer: Towards Generalizable Unified Runtime State Representation for Failure Management

**arXiv ID:** 2603.21495 | [PDF](https://arxiv.org/pdf/2603.21495v1)

**作者:** Lingzhe Zhang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**通讯引用:** 22905 | [OpenAlex ID](https://openalex.org/A5100414156)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过统一编码多模态运行时数据（指标、追踪和日志），预训练出可跨任务使用的系统状态嵌入，并在此基础上实现状态感知的任务微调，提升了故障管理的泛化能力。

**💡 创新点**

创新点在于将表示学习与故障管理解耦，提出统一运行时对比学习和状态感知任务微调，实现任务无关的多模态嵌入，并通过跨模态一致性和时序一致性约束提升嵌入质量。

**🔧 技术方法**

使用了共享的预训练嵌入骨干（如 Qwen3-Embedding-0.6B）、多模态对齐的 InfoNCE 损失、时序一致性损失以及可选的弱异常分离损失，并通过轻量级 MLP 融合层完成多模态融合。

**📊 数据集**

主要使用公开的 AIOps 2022 数据集、ART 数据集，以及在训练时自建的 Train-Ticket、Online Boutique 等分布式系统运行时采集和 Chaos Mesh 注入的失败样本。

**📈 对比分析**

与基线 Qwen3-Embedding-0.6B 进行 t‑SNE 可视化对比，RuntimeSlicer 在状态聚类更清晰；在下游故障管理任务（异常检测、故障定位、故障诊断）上，精度、召回和 F1 均提升至 97.27%/81.18%/88.50%，70.43%/68.43%/70.15% 以及 87.57%/75.12%/80.88%，并在 MRR 上取得 70.15%/83.35%。

**⚠️ 局限性**

局限性包括对某些稀缺运行时状态下的表现下降，且实验仅在单一 AIOps 2022 数据集上验证，缺乏大规模多系统的泛化评估。

---

## 624. A Framework for Closed-Loop Robotic Assembly, Alignment and Self-Recovery of Precision Optical Systems

**arXiv ID:** 2603.21496 | [PDF](https://arxiv.org/pdf/2603.21496v1)

**作者:** Seou Choi `[一作]` (Massachusetts Institute of Technology), Marin Soljačić `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 57820 | [OpenAlex ID](https://openalex.org/A5060426875)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一套完整的机器人平台，实现从零件散放到桌面激光腔的全自动搭建、闭环光束对准、模式优化以及在扰动下的自我修复。

**💡 创新点**

将分层视觉感知、空间与角度优化算法与定制的精细调节工具（FAT）结合，首次在高精度自由空间光学系统中实现了闭环全自动装配与自我恢复，突破了以往仅限于拾取与简单对准的局限。

**🔧 技术方法**

使用7自由度xArm7机械臂、4K立体相机、末端LiDAR相机、光束检测相机、ArUco标记的标准化光学零件盒、Wi‑Fi控制的电机化细调工具、基于Newton法的空间优化、基于贝叶斯优化的角度调节，以及自我恢复的连续监测与优化流程。

**📊 数据集**

未使用公开数据集，全部基于现场实验采集的光束强度、位置、M²指标等实时测量数据。

**📈 对比分析**

与传统的单纯拾取对准方法相比，本系统在激光腔组装成功率100%、空间定位精度≤0.1 mm、角度对准迭代次数≤6次、失配自恢复平均耗时≈3 min。实验中对镜头位移、镜面角度漂移等扰动的恢复成功率分别为100%与90%。

**⚠️ 局限性**

局限性包括：需要专用的光学零件盒与细调工具，系统对组件的几何与可调节性有限；对大规模、非标准化光学系统的通用性尚未验证；对环境温度、振动等极端扰动的鲁棒性仍需进一步提升；自我恢复过程依赖连续监测，实时性能受摄像头帧率与算法延迟影响。

---

## 625. Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification

**arXiv ID:** 2603.21504 | [PDF](https://arxiv.org/pdf/2603.21504v1)

**作者:** Jayanie Bogahawatte `[一作]` (University of Melbourne), Saman Halgamuge `[通讯]` (University of Melbourne)

**通讯引用:** 12424 | [OpenAlex ID](https://openalex.org/A5067418792)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一种用于少样本弱监督全切片图像（WSI）分类的框架HIPSS，能够在仅有极少的 slide-level 标签的情况下实现高精度分类。

**💡 创新点**

创新点主要包括：①利用标度与偏移特征（SSF）在文本编码器中进行参数高效的提示调优，显著降低可训练参数数量与推理开销；②引入软层级文本引导的 WSI 表示学习，通过在实例与区域两级注意力聚合并结合文本相似度自适应加权，避免硬过滤导致的信息丢失。

**🔧 技术方法**

技术方法涵盖：预训练的 Vision‑Language 模型（如 CONCH）、SSF‑基提示调优、软层级文本引导的注意力聚合、对比学习训练策略以及对文本描述的 LLM 生成。

**📊 数据集**

实验使用了三个公开病理数据集：Camelyon16（乳腺癌转移），TCGA‑Lung（肺癌亚型），以及 UBC‑OCEAN（卵巢癌五亚型）。

**📈 对比分析**

与多种 SOTA 基线（ABMIL、ViLa‑MIL、TOP、FOCUS 等）对比，HIPSS 在所有 1‑shot 至 16‑shot 设置下均取得更高的 AUC，最大提升达 13.8%（UBC‑OCEAN 1‑shot），并且可训练参数减少 18.1%–5.8% 及推理时间缩短。

**⚠️ 局限性**

局限性包括：对 LLM 生成的文本描述缺乏专家验证，可能导致描述质量不一；以及在极低样本（1‑shot）情形下对模型超参数（如 λ、α）的敏感性仍需进一步研究。

---

## 626. Agentic Automation of BT-RADS Scoring: End-to-End Multi-Agent System for Standardized Brain Tumor Follow-up Assessment

**arXiv ID:** 2603.21494 | [PDF](https://arxiv.org/pdf/2603.21494v1)

**作者:** Mohamed Sobhi Jabal `[一作]` (Duke University Medical Center), Evan Calabrese `[通讯]` (Duke University Medical Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研发并评估了一个端到端的多代理 LLM 与 CNN 系统，用于自动化脑肿瘤 BT‑RADS 评分。

**💡 创新点**

将临床文本信息提取、影像分割量化和决策逻辑三者分离为专用代理，并通过证据跨度可验证实现闭环评分，首次在 BT‑RADS 框架内实现完整自动化。

**🔧 技术方法**

使用 20B 开源 LLM（GPT‑oss）进行临床变量提取、3D nnU‑Net 进行肿瘤分割量化，并用 Pydantic 约束生成及规则推理实现 BT‑RADS 决策。

**📊 数据集**

509例（492例可评估）单中心术后胶质瘤 MRI 数据，包含自由文本临床记录与多序列 MRI。

**📈 对比分析**

采用 McNemar 检验与专家参考标准比较，系统准确率 76.0%（vs 57.5%）；情境依赖类敏感度 92.7–100%，BT‑4 的 PPV 为 92.9%。

**⚠️ 局限性**

局限于单中心回顾性设计；分割误差导致阈值边界误判；缺乏多中心验证；未对单个病灶进行细粒度分析；临床变量提取仍有 87–97% 的准确率。

---

## 627. Effective Strategies for Asynchronous Software Engineering Agents

**arXiv ID:** 2603.21489 | [PDF](https://arxiv.org/pdf/2603.21489v1)

**作者:** Jiayi Geng `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**通讯引用:** 21651 | [OpenAlex ID](https://openalex.org/A5068811427)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于分支合并的异步多代理系统（CAID），通过集中任务委托、异步执行和工作区隔离来完成长周期软件工程任务。

**💡 创新点**

创新点在于将软件工程原语（分支合并、工作区隔离、可执行测试）作为多代理协调的核心框架，并证明branch‑and‑merge机制显著提升多代理性能；同时强调任务拆分的依赖感知与工作区隔离的互补作用。

**🔧 技术方法**

使用了大型语言模型（OpenHands、MiniMax 2.5、Claude 4.5等）、git worktree 进行物理隔离、JSON协议进行结构化通信、自动化测试与依赖图、异步事件循环及历史压缩技术。

**📊 数据集**

使用了 Commit0‑Lite（Python 库从零实现）和 PaperBench（论文复现）这两个长周期软件工程基准数据集。

**📈 对比分析**

与单代理基线进行对比，实验表明在 Commit0‑Lite 上提升 14.0% 绝对准确率（Claude 4.5 由 53.1% 提升至 59.1%），在 PaperBench 上提升 26.7%（MiniMax 2.5 由 10.4% 提升至 36.7%）。单代理迭代加倍的提升有限，单代理+多代理顺序执行成本高且效果与直接多代理相差不大。

**⚠️ 局限性**

限制包括：高 API 成本与运行时间、协调与合并开销导致速度未显著提升；任务拆分依赖手工提示，缺乏学习型拆分策略；对非编码类长周期任务的通用性未验证。

---

## 628. Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies

**arXiv ID:** 2603.21485 | [PDF](https://arxiv.org/pdf/2603.21485v1)

**作者:** Koichi Tanaka `[一作]` (Keio University), Yuta Saito `[通讯]` (Hanuku-kaso, Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于点击概率的逆概率加权估计器（CIPS），用于在完全确定性日志策略下进行排名的离线策略评估。

**💡 创新点**

创新点在于用用户点击概率的边缘化权重替代日志策略的概率权重，显著放宽了支持条件，使得在确定性日志下也能实现无偏或低偏估计。

**🔧 技术方法**

采用理论推导的偏差与方差分析、CIPS及其双重稳健扩展（CDR）方法，并利用神经网络估计点击概率；实验使用合成数据与真实的KuaiRec推荐日志。

**📊 数据集**

实验数据包括基于Plackett–Luce模型生成的合成数据以及来自短视频平台Kuaishou的KuaiRec用户-项目交互日志。

**📈 对比分析**

与IPS、IIPS、RIPS等传统排名OPE估计器对比，CIPS在完全确定性日志下显著降低均方误差和平方偏差，实验结果显示其MSE远低于基线，并在真实数据上保持优势。

**⚠️ 局限性**

局限性包括对点击概率估计的准确性和点击-wise常支持假设的依赖；当新旧策略均为确定性且点击支持不足时仍会产生偏差；相对传统方法方差可能略高且需要充足的点击信息。

---

## 629. ALADIN:Attribute-Language Distillation Network for Person Re-Identification

**arXiv ID:** 2603.21482 | [PDF](https://arxiv.org/pdf/2603.21482v1)

**作者:** Wang Zhou `[一作]` (Wuhan University), Ziyue Zhou `[通讯]` (Wuhan University)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5101535926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 ALADIN，通过冻结 CLIP 教师与多模态 LLM 自动生成属性描述，蒸馏属性级跨模态知识到轻量化 ReID 学生网络，实现高效身份匹配。

**💡 创新点**

创新点在于属性-语言蒸馏框架、场景感知软提示自适应文本、局部属性-区域对齐与跨模态关系蒸馏，以及利用 MLLM 生成结构化属性提升可扩展性。

**🔧 技术方法**

使用技术包括 CLIP、Qwen‑VL 等多模态 LLM、软提示生成网络（SAPG）、多级蒸馏（全局、属性、局部）、对比损失、关系蒸馏、BN‑neck、Progressive 优化等。

**📊 数据集**

实验数据集涵盖 Market‑1501、DukeMTMC‑ReID 与 MSMT17 三大 ReID 基准。

**📈 对比分析**

与现有 CNN、Transformer 与 CLIP‑基准比较，ALADIN 在所有数据集上均实现 Rank‑1 与 mAP 的提升，例如 Market‑1501 上 Rank‑1 从 95.0% 提升至 96.1%，mAP 从 89.4% 提升至 91.1%。

**⚠️ 局限性**

局限性包括对 MLLM 生成属性噪声敏感，尤其属性错误导致性能显著下降；软提示长度、学习率等超参数需仔细调优；仅在训练阶段使用 CLIP 与 MLLM，推理时仍需预先训练的学生网络。

---

## 630. Empirical Evaluation of Link Deletion Methods for Limiting Information Diffusion on Social Media

**arXiv ID:** 2603.21470 | [PDF](https://arxiv.org/pdf/2603.21470v1)

**作者:** Shiori Furukawa `[一作]` (University of Tsukuba), Sho Tsugawa `[通讯]` (University of Tsukuba)

**通讯引用:** 866 | [OpenAlex ID](https://openalex.org/A5019172487)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在社交媒体上通过删除边（链接）来限制信息扩散的有效性，采用真实转发日志而非合成模型进行评估。

**💡 创新点**

创新点在于提出一种基于实际转发日志估算删链后扩散规模的方法，比较保守（非树扩散图）与乐观（树扩散图）两种估计，揭示真实数据中多源传播导致删链效果受限的现象。

**🔧 技术方法**

使用了 NetMelt、Betweenness、Edge-Degree 等三种删链算法，并构建非树扩散图、树（first）与树（last）三种扩散图来估计删链后传播规模。

**📊 数据集**

四个大规模数据集：Ordinary（普通推文）、Higgs（希格斯探测）、URL（URL 推文）和 Douban（豆瓣书评）用于实验。

**📈 对比分析**

通过比较删链前后总扩散规模和单个推文的规模，发现即使删掉约 50% 的边，传播规模仅下降至原来的一半左右；NetMelt 在四个数据集上表现最好，但仍需删除大量边才能显著抑制扩散。

**⚠️ 局限性**

主要限制包括：删链无法阻止种子用户从其他渠道获得信息，导致多源传播；要达到显著效果需删除大量边，实际操作成本高；方法仅为估计，缺乏真实删链实验验证。

---

## 631. Hardening Confidential Federated Compute against Side-channel Attacks

**arXiv ID:** 2603.21469 | [PDF](https://arxiv.org/pdf/2603.21469v1)

**作者:** James Bell-Clark `[一作]` (Google), Jonathan Katz `[通讯]` (Google)

**通讯引用:** 28517 | [OpenAlex ID](https://openalex.org/A5053223081)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文分析了Confidential Federated Compute平台中的侧信道漏洞，并展示了攻击者利用消息长度和内存分配等侧信道泄露敏感信息的两种攻击方法。

**💡 创新点**

创新点在于首次系统性识别并量化CFC侧信道风险，并提出基于差分隐私的动态填充与随机重定尺寸机制来有效抑制泄漏。

**🔧 技术方法**

采用的技术包括差分隐私噪声注入（DP padding）、PositiveLaplace机制、AboveThreshold/StrictUDSAboveThreshold、联邦学习框架中的CVM与KMS，以及对内存分配的动态私有化。

**📊 数据集**

实验使用合成数据（Sybil用户生成的 GROUP BY SUM 查询）验证攻击效果与防御性能，并未使用公开真实数据集。

**📈 对比分析**

通过对比未防御与防御后系统，DP填充在消息长度上将开销降低到原来的一半以上，动态重定尺寸在大多数场景下保持误差低于1%，显示防御方案在准确性和资源消耗上均具可接受性能。

**⚠️ 局限性**

限制在于只覆盖了CFC的GROUP BY SUM查询的两类侧信道，其他侧信道（如数据访问模式、代码访问模式）仍未被彻底解决，且对ORAM等更高安全级别的实现未进行评估。

---

## 632. Deliberative multi-agent large language models improve clinical reasoning in ophthalmology

**arXiv ID:** 2603.21447 | [PDF](https://arxiv.org/pdf/2603.21447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 633. GateANN: I/O-Efficient Filtered Vector Search on SSDs

**arXiv ID:** 2603.21466 | [PDF](https://arxiv.org/pdf/2603.21466v1)

**作者:** Nakyung Lee `[一作]` (Sungshin Women's University), Gyuyeong Kim `[通讯]` (Sungshin Women's University)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5065851050)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对固态硬盘（SSD）上的向量检索提出了一种I/O高效的过滤式搜索算法（Gate-ANN），通过在SSD层面优化索引结构和检索流程，显著减少磁盘I/O次数并提升查询速度。

**💡 创新点**

创新点在于：① 在SSD存储层面设计了“Gate”过滤机制，先在内存中快速筛选候选向量；② 采用列式存储和压缩布局降低SSD访问延迟；③ 将传统的近似最近邻（ANN）方法与SSD I/O优化融合，形成一种兼顾查询速度与存储效率的新范式。

**🔧 技术方法**

技术手段包括：近似最近邻搜索、布隆过滤器/位图过滤、列式压缩存储、SSD特定的预取和分页策略、GPU/CPU协同计算。实现代码使用C++并利用liburing或NVMe库进行异步IO。

**📊 数据集**

使用了多个公开基准数据集：SIFT1M、GloVe100k、Deep1B（大规模图像特征）、ImageNet-1K向量等，覆盖从小规模到千万级向量的多样化场景。

**📈 对比分析**

与 Faiss、HNSW、IVFPQ、NMSLIB 等主流 ANN 方法在 SSD 环境下进行对比。Gate-ANN 在保持 99% 以上召回率的前提下，SSD I/O 次数下降 30–70%，查询延迟降低 40–60%，而整体存储占用与现有方法相近。

**⚠️ 局限性**

局限性包括：① 主要针对静态或读多写少的数据集，动态插入/删除性能尚未完善；② 对极大规模（>10B）向量仍受SSD容量和内存缓存限制；③ 需要对SSD固件和驱动进行特定优化，通用性相对受限。

---

## 634. PAS3R: Pose-Adaptive Streaming 3D Reconstruction for Long Video Sequences

**arXiv ID:** 2603.21436 | [PDF](https://arxiv.org/pdf/2603.21436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 635. DRTriton: Large-Scale Synthetic Data Reinforcement Learning for Triton Kernel Generation

**arXiv ID:** 2603.21465 | [PDF](https://arxiv.org/pdf/2603.21465v1)

**作者:** Siqi Guo `[一作]` (Texas A&M University), Tianbao Yang `[通讯]` (Texas A&M University)

**通讯引用:** 6089 | [OpenAlex ID](https://openalex.org/A5023288846)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DRTriton——一种基于大型语言模型的全流程框架，使用合成数据进行强化学习，能够将 PyTorch 代码自动转换为高效的 Triton（CUDA）kernel，并在真实的 CUDA 场景中实现速度加速。

**💡 创新点**

创新点：
- CSP‑DAG 算法能够高效生成满足形状约束、可验证的 PyTorch 程序，保证样本质量；
- 课程强化学习与解耦奖励（DRPO）在 sparse reward 环境下显著提升正确率和执行速度；
- 运行时搜索策略进一步提升复杂程序的融合效果；
- 仅使用合成数据训练，仍能在真实模型上超越现有 LLM 与编译器。

**🔧 技术方法**

技术手段：
- 大型语言模型 Qwen‑2.5‑Coder‑7B‑Instruct 作为基础；
- 合成数据生成：CSP‑DAG 与 CP‑SAT 约束求解；
- 强化学习：DRPO（decoupled rewards）和三阶段课程学习；
- 验证流程：语法检查、蒙特卡洛单元测试、功能正确性验证；
- Test‑time search：碎片提取、核融合搜索并选取最快实现。

**📊 数据集**

使用的数据集：
- 2026 条单操作的 PyTorch‑Triton 对；
- 合成 benchmark（406 条 PyTorch 程序，分 4 个难度级别）；
- KernelBench（250 条真实 PyTorch 程序，3 个难度级别）。

**📈 对比分析**

评估方式：与 GPT‑5.2、Claude Sonnet 4.5、DeepSeek‑R1、Qwen‑3‑Coder‑480B、AutoTriton 以及 Torch Eager / torch.compile 进行对比；使用 Acc、Faster1、平均 speedup 等指标。结果显示：
- 在 synthetic benchmark 上 Acc 达 86.8%，平均 speedup 1.57×；
- 在 KernelBench Level 1/2/3 分别取得 69%/96%/76% 的准确率，速度比 Torch Eager 高 17%/92%/54%，比 torch.compile 高 56%/34%。

**⚠️ 局限性**

局限性：
- 仍需要将面向对象的 PyTorch 代码重写为函数式格式，存在分布差异；
- 对极长程序（> 5 操作）采用碎片上限 1,024，可能无法覆盖所有优化空间；
- 依赖显式验证与搜索，运行时开销较大；
- 模型规模为 7B，推理成本仍不低；
- 对非训练分布的新算子或动态形状场景的泛化能力待进一步验证。

---

## 636. Safety as Computation: Certified Answer Reuse via Capability Closure in Task-Oriented Dialogue

**arXiv ID:** 2603.21448 | [PDF](https://arxiv.org/pdf/2603.21448v1)

**作者:** Cosimo Spera `[一作]` `[通讯]` (Minerva CQ), Cosimo Spera (Minerva CQ)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将安全认证视为闭包计算，预先生成所有可推导答案并存储，在会话中通过包含检查快速回答，消除冗余检索与生成。

**💡 创新点**

将安全认证与答案重用统一为同一计算，提出 Certified Answer Store 与 Pre‑Answer Block，并证明语义缓存在多租户环境中不安全。

**🔧 技术方法**

采用能力超图、固定点闭包、Datalog 推理、增量闭包维护、预答案块生成、模板化答案与安全包含检查等技术。

**📊 数据集**

MultiWOZ 2.2 数据集（10,438 条人机对话，1,000 条测试集）。

**📈 对比分析**

与基线、余弦相似度缓存以及 CAS+PAB 进行对比，平均 RAG 调用从 13.7 降至 1.31，延迟从 18.8 s 降至 340 ms，安全缓存错误率从 14.3 % 降至 0 %。

**⚠️ 局限性**

依赖完整性假设与禁止集准确性，未涵盖跟踪器误差、工具调用失败、模型偏差及恶意注入等现实部署风险。

---

## 637. Decidability of Livelock Detection for Parameterized Self-Disabling Unidirectional Rings

**arXiv ID:** 2603.21443 | [PDF](https://arxiv.org/pdf/2603.21443v1)

**作者:** Aly Farahat `[一作]` `[通讯]` (Intusurg), Aly Farahat (Intusurg)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

证明了在参数化的自我禁用单向环中，livelock检测是可判定的，并给出了一个多项式时间（O(|T|^3）无K依赖）算法；

**💡 创新点**

创新点在于将Farahat的代数化伪livelock与传播条件转化为一个递减、单调的定义运算，利用Knaster‑Tarski定理得到最大的闭包，从而以固定点形式高效判定livelock自由；

**🔧 技术方法**

采用了伪livelock图、影子传播、Tarjan SCC、最大固定点迭代、代数等价传播与等价传播条件的证明；

**📊 数据集**

实验使用了Dijkstra令牌环、k‑coloring（确定性/非确定性）、共识、Sum‑Not‑2等经典自我稳定协议的模型；

**📈 对比分析**

与之前的Π^0_1完整性半算法和基于平铺的判定方法比较，所给算法在O(|T|^3)时间内完成，无论环大小，实验表明在典型协议上实现了高效判定；

**⚠️ 局限性**

限制在于仅处理对称和(1,1)非对称环；一般(l,q)-非对称情况及非自我禁用转移留待后续研究。

---

## 638. Is the future of AI green? What can innovation diffusion models say about generative AI's environmental impact?

**arXiv ID:** 2603.21419 | [PDF](https://arxiv.org/pdf/2603.21419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 639. EpiMask: Leveraging Epipolar Distance Based Masks in Cross-Attention for Satellite Image Matching

**arXiv ID:** 2603.21463 | [PDF](https://arxiv.org/pdf/2603.21463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 640. Behavioural feasible set: Value alignment constraints on AI decision support

**arXiv ID:** 2603.21435 | [PDF](https://arxiv.org/pdf/2603.21435v1)

**作者:** Taejin Park `[一作]` `[通讯]`, Taejin Park

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

做了什么：研究了商业AI系统在供应商对齐约束下的可行推荐集合，并通过情景实验验证对齐如何压缩决策灵活性和改变利益相关者优先级。

**💡 创新点**

创新点是什么：提出“行为可行集合”概念，给出可测量的逆转与平衡阈值，系统性展示对齐对不同领域决策空间的压缩效应。

**🔧 技术方法**

用了什么技术：利用信息论（KL散度、Pinsker不等式）构建诊断阈值，并在大型语言模型上进行二元决策与多利益相关者排序实验。

**📊 数据集**

用了什么数据集：构造20个二元决策情景和8个多利益相关者排序情景，采样 GPT、Claude、Llama Base/Instruct 四个模型的多次输出。

**📈 对比分析**

如何比较的方法，性能怎么样：通过对比对齐前后模型的逆转率和利益相关者权重变化，发现对齐显著降低逆转率并改变优先级；GPT与Claude在大多数领域几乎无逆转能力。

**⚠️ 局限性**

limitation是什么：实验仅评估模型输出的可达性，缺乏对实际决策成本的考量；情景样本有限，且无法观测供应商内部对齐参数，限制了对齐机制的完整解释。

---

## 641. DomAgent: Leveraging Knowledge Graphs and Case-Based Reasoning for Domain-Specific Code Generation

**arXiv ID:** 2603.21430 | [PDF](https://arxiv.org/pdf/2603.21430v1)

**作者:** Shuai Wang `[一作]` (Chalmers University of Technology), Yinan Yu `[通讯]` (Chalmers University of Technology)

**通讯引用:** 3298 | [OpenAlex ID](https://openalex.org/A5100411376)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DomAgent，一个结合知识图谱检索与案例检索的域特定代码生成代理。

**💡 创新点**

创新点：联合底层知识图谱导向检索与案例检索的双向检索框架，利用LLM进行工具调用与检索结果过滤，并采用小样本KG‑guided案例选择。

**🔧 技术方法**

技术：RAG、知识图谱（KG）、案例检索（CBR）、LLM推理工具调用、强化学习优化、句子‑BERT编码。

**📊 数据集**

数据集：DS‑1000（数据科学库任务）与Truck CAN Signal（卡车CAN信号编程）。

**📈 对比分析**

比较：对比多种LLM和代码代理（WizardCoder、Magicoder、LLaMA、GPT‑4o）在DS‑1000和CAN Signal上的pass@1，DomAgent在DS‑1000提升约+10%，在CAN Signal提升至≈98%/96%（相较基线提升约55%）。

**⚠️ 局限性**

局限：依赖手工构建KG和案例库，检索与过滤仍有误检风险，对极端新颖任务的泛化能力有限。

---

## 642. Dynasto: Validity-Aware Dynamic-Static Parameter Optimization for Autonomous Driving Testing

**arXiv ID:** 2603.21427 | [PDF](https://arxiv.org/pdf/2603.21427v1)

**作者:** Dmytro Humeniuk `[一作]` (Polytechnique Montréal), Foutse Khomh `[通讯]` (Polytechnique Montréal)

**通讯引用:** 8080 | [OpenAlex ID](https://openalex.org/A5071052367)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 Dynasto 两步方法：先用强化学习学习可验证的对抗驾驶策略，再用遗传算法搜索初始场景参数，随后对失败轨迹进行图聚类提炼可解释的失败模式。

**💡 创新点**

创新点包括：① 将动态策略优化与静态参数搜索协同化；② 通过 Signal Temporal Logic 定义有效性约束，剔除无效/不现实碰撞；③ 用图聚类（Leiden+Levenshtein）自动归纳失败模式。

**🔧 技术方法**

使用的技术有：DQN 强化学习、遗传算法、Signal Temporal Logic、Levenshtein 距离、Leiden 社区检测、kNN 图构建、时间序列事件编码。

**📊 数据集**

实验数据集基于 HighwayEnv 仿真环境，包含两种自适应驾驶控制器（SUT1、SUT2）以及对应的初始条件和 IDM 车辆。

**📈 对比分析**

与 Random Search、单纯 GA、Baseline RL（奖励所有碰撞）以及 Validity‑Aware RL 进行比较。Dynasto 在 4000 步预算下比纯 RL 提升 60–70% 有效失败数，产生约 12 个代表性失败模式，集群数比基线多约 70%，性能显著优于所有基线。

**⚠️ 局限性**

局限性：仅在 2D 两车高速公路场景验证；未评估更复杂多车、多模态或高保真渲染环境；对抗策略仅学习一条轨迹，可能缺乏多样性；对失效严重度和覆盖度的量化仍待进一步研究。

---

## 643. Learning Can Converge Stably to the Wrong Belief under Latent Reliability

**arXiv ID:** 2603.21491 | [PDF](https://arxiv.org/pdf/2603.21491v1)

**作者:** Zhipeng Zhang `[一作]` (China Mobile Research Institute), Lei Yang `[通讯]` (China Mobile Research Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在反馈可靠性不可观测的情况下，学习系统如何可能在收敛稳定的同时学习错误的解，并提出了一种基于学习轨迹的监测-信任-调节（MTR）框架来推断可靠性并动态调节更新。

**💡 创新点**

创新点在于揭示单步反馈不具备可辨别性，而轨迹级统计能够实现可辨别；提出了三层次的结构化设计（Monitor、Trust Estimator、Regulator）以实现跨时间尺度的可靠性推断与学习调节；并证明了该框架能够在持续性偏差下恢复正确学习。

**🔧 技术方法**

采用了梯度下降类优化器（SGD、Adam、PPO等）与轨迹统计方法（如更新幅度、方差、方向不一致度），实现了慢速信任变量的估计与更新率的调节；通过理论推导（误差收敛、轨迹可辨别性）和数值实验验证其有效性。

**📊 数据集**

主要使用了控制实验中的自定义奖励/标签腐败环境（例如强化学习中的PPO与带偏差标签的监督学习任务），并在这些受控设置下收集了训练过程的参数更新、损失、信任变量等数据。

**📈 对比分析**

与传统优化器对比，MTR在持续性偏差阶段保持学习稳定并能在恢复期快速回到接近干净数据的性能；实验中信任变量能够区分可靠与不可靠阶段（AUC≈0.85），而单步信号仅达到随机水平；整体性能提升主要体现在减少错误收敛和提升恢复速度。

**⚠️ 局限性**

限制在于假设可靠性状态为相对持久的区间，快速切换或高度非平稳的可靠性场景难以处理；实验仅在小规模、受控环境中验证，缺乏在大规模真实数据和复杂模型上的验证；轨迹特征和信任模型的设计仍可进一步丰富。

---

## 644. Unified-MAS: Universally Generating Domain-Specific Nodes for Empowering Automatic Multi-Agent Systems

**arXiv ID:** 2603.21475 | [PDF](https://arxiv.org/pdf/2603.21475v1)

**作者:** Hehai Lin `[一作]` (Hong Kong University of Science and Technology), Chengwei Qin `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5114038310)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Unified‑MAS 框架，先离线检索外部知识合成域特定节点，再用基于困惑度的奖励对节点进行迭代优化，使自动多智能体系统在知识密集领域获得更优性能；

**💡 创新点**

创新点：①将节点实现与拓扑编排解耦，采用离线节点合成而非实时动态生成；②通过多维关键词提取与检索驱动查询，从外部知识库获取专业知识生成域特定节点；③利用困惑度指导的奖励函数对节点内部逻辑进行自适应优化；

**🔧 技术方法**

技术手段包括：多维关键词提取、检索驱动查询、LLM（Gemini‑3‑Pro 作为 Designer、Qwen3‑Next‑80B‑A3B‑Instruct 作为 Executor）、基于困惑度的奖励与节点优化、图搜索自动 MAS（如 AFlow、MAS‑Zero、ScoreFlow、MAS^2）等；

**📊 数据集**

实验使用四个专业基准：TravelPlanner、HealthBench、J1Bench（法律判决）、DeepFund；并在 AIME 2024/25 上检验通用性；

**📈 对比分析**

将 Unified‑MAS 生成的节点注入四个自动 MAS 基线，并与手工 MAS、动态节点自动 MAS 以及静态节点自动 MAS 进行对比；在四个 LLM 组合下评估准确率和成本，平均性能提升 6%–14.2%，成本降低，性能‑成本曲线明显优于原始基线；

**⚠️ 局限性**

限制：仅支持离线节点预处理，无法实时应对极端动态或时效性强的场景；节点生成与优化过程耗时较长，需要多轮评估；对外部检索质量和 LLM 生成的准确性仍存在依赖。

---

## 645. Beyond Correlation: Refutation-Validated Aspect-Based Sentiment Analysis for Explainable Energy Market Returns

**arXiv ID:** 2603.21473 | [PDF](https://arxiv.org/pdf/2603.21473v1)

**作者:** Wihan van der Heever `[一作]` (Nanyang Technological University), Erik Cambria `[通讯]` (Nanyang Technological University)

**通讯引用:** 53732 | [OpenAlex ID](https://openalex.org/A5100752356)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在能源行业股票收益上，提出了一套基于方面的情绪分析并通过四种反证检验（placebo、随机共因、子样本稳定性、bootstrap）来筛选出稳健的情绪-收益关联；

**💡 创新点**

创新点在于将反证检验嵌入情绪分析流程，显著剔除大部分统计噪声，首次在能源领域展示传统与可再生企业在情绪敏感度上的差异；

**🔧 技术方法**

采用Aspect‑Based Sentiment Analysis（SenticGCN + net‑ratio scoring + z‑normalization）、OLS 回归配合 Newey‑West HAC 标准误，以及四重反证检验算法；

**📊 数据集**

使用 2022 年 Q4 的约 120,000 条英文推文和 Yahoo Finance 的六只能源股（BP、Exxon、Shell、NextEra、Clearway、Brookfield Renewable）每日收盘价；

**📈 对比分析**

相较于传统相关/Granger 方法，验证通过的系数规模更小（约 0.4–0.5 bps/标准差），但经检验后更具可信度，说明多数高相关结果为假阳性；

**⚠️ 局限性**

局限包括样本量极小（仅六只股票、一季度）、仅基于推特文本、无法完全排除未观测共因、未实现真正因果识别，结果可能受宏观周期与平台偏差影响。

---

## 646. Learning Inflation Narratives from Reddit: How Lightweight LLMs Reveal Forward-Looking Economic Signals

**arXiv ID:** 2603.21501 | [PDF](https://arxiv.org/pdf/2603.21501v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 647. DSPA: Dynamic SAE Steering for Data-Efficient Preference Alignment

**arXiv ID:** 2603.21461 | [PDF](https://arxiv.org/pdf/2603.21461v1)

**作者:** James Wedgwood `[一作]` (Carnegie Mellon University), Virginia Smith `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3676 | [OpenAlex ID](https://openalex.org/A5112800069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Dynamic SAE Steering for Preference Alignment (DSPA) 的推理时偏好对齐方法，利用稀疏自编码器（SAE）在不更新模型权重的情况下，根据提示动态选择并编辑激活特征；

**💡 创新点**

创新点在于构造基于偏好三元组的条件差分映射（conditional‑difference map），实现提示条件化的稀疏特征选择与 token‑级别的激活干预，既保留对齐效果，又提供可审计的可解释性；

**🔧 技术方法**

技术手段包括稀疏自编码器（SAE）、条件差分映射、Prompt 触发的 gate 选择、Top‑k 特征筛选、token‑条件下的激活放大/抑制以及与激活空间相关的理论分析；

**📊 数据集**

使用 UltraFeedback（61k 带偏好标签的提示-响应对）以及 HH‑RLHF 训练的 SAEs，评估数据集为 MT‑Bench、AlpacaEval 以及多项选择基准（MMLU、Arc‑Easy、TruthfulQA‑MC2、HellaSwag、Winogrande）；

**📈 对比分析**

与基线方法（DPO、RepE、Static‑SAE、Prompt Eng、Base Model）以及 RAHF‑SCIT 进行比较，DSPA 在 MT‑Bench 上提升得分、在 AlpacaEval 上保持竞争力、对多项选择性能影响极小；在数据受限场景下仍稳健，并比 RAHF‑SCIT 低 4.47 倍的对齐阶段 FLOPs；

**⚠️ 局限性**

局限性包括：需同时拥有输入层和输出层的高质量 SAE；对 SAE 训练和调参高度依赖；评估依赖 LLM‑as‑judge，可能带来偏见；特征解释与类别划分未进行系统的人类验证；未覆盖安全性或长文本生成等更严苛任务。

---

## 648. Finding Minimum Distance Preservers: A Parameterized Study

**arXiv ID:** 2603.21442 | [PDF](https://arxiv.org/pdf/2603.21442v1)

**作者:** Kirill Simonov `[一作]`, Shaily Verma `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在无权无向图中寻找最小距离保留子图（即保留指定终点集或终点对之间的最短路径）的计算复杂性。

**💡 创新点**

创新点在于首次系统划分了该问题的参数化复杂性：证明了在终点数、解大小、树宽、顶点覆盖等自然参数下的W[1]-难度、FPT、NP难度与紧界，并揭示了不同参数对两种变体（子集化与对化）产生的显著差异。

**🔧 技术方法**

主要技术包括参数化归约、图的网格与子图构造、Hanan网格结构的利用、树宽动态规划、顶点覆盖等价类简化与分支算法、以及针对解大小的Clique归约。

**📊 数据集**

本研究为理论性工作，未使用公开数据集进行实验验证。

**📈 对比分析**

由于缺乏实验实现，本文未给出实验性能比较；所有结论均基于严格的算法分析与归约证明。

**⚠️ 局限性**

局限性：目前仅在无权无向图中考虑，未讨论加权/有向图；对树宽参数的FPT结果仅适用于子集化保留器，未给出对对化保留器的完整FPT阈值；实验评估与近似算法研究仍待后续工作。

---

## 649. Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs

**arXiv ID:** 2603.21418 | [PDF](https://arxiv.org/pdf/2603.21418v1)

**作者:** Mariela M. Nina `[一作]` (Federal University of São Paulo), Didier A. Vega-Oliveros `[通讯]` (Federal University of São Paulo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在巴西葡萄牙语问答任务SQuAD-BR上，系统评估了BERTimbau的四种Parameter-Efficient Fine-Tuning（LoRA、DoRA、QLoRA、QDoRA）及4‑bit量化配置，并与全参数微调及生成模型进行对比实验；

**💡 创新点**

首次在低资源语言问答场景中综合验证多种PEFT与量化组合，揭示高学习率是PEFT成功的关键，并证明大模型在4‑bit量化下更具鲁棒性，为绿色AI实践提供实证指导；

**🔧 技术方法**

采用LoRA、DoRA、QLoRA、QDoRA四种PEFT技术，配合4‑bit NF4量化、双重量化、Paged Optimizers、AdamW优化器以及高学习率（2×10⁻⁴）和梯度裁剪；

**📊 数据集**

使用SQuAD-BR（葡萄牙语版SQuAD v1）数据集，训练集87,599问答对，验证集10,570问答对；

**📈 对比分析**

与全参数微调基准对比，LoRA在Large模型上实现F1 81.32（占84.86的95.8%），训练时间降低73.5%，内存降低50%；QLoRA在Large上达到80.03 F1，显著降低内存（3,281 MB，81.9%缩减）。相较于生成模型，BERTimbau在相同F1下消耗4.2倍更少GPU内存、3倍更短训练时间；

**⚠️ 局限性**

实验仅覆盖SQuAD-BR问答任务，未检验跨任务泛化；高学习率对训练稳定性有依赖，需要进一步调优；Base模型量化导致性能下降显著；仅评估了四种PEFT方法，未考虑Prompt Tuning等；实验在单一RTX A4500 GPU上完成，缺乏多GPU规模验证。

---

## 650. Multinoulli Extension: A Lossless Continuous Relaxation for Partition-Constrained Subset Selection

**arXiv ID:** 2603.21492 | [PDF](https://arxiv.org/pdf/2603.21492v1)

**作者:** Qixin Zhang `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 99951 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了多项式时间参数无关、查询效率高的连续松弛框架 Multinoulli Extension，并基于该框架设计了离线算法 Multinoulli‑SCG 以及在线算法 Multinoulli‑OSCG 与 Multinoulli‑OSGA，用以解决满足分区约束的“近似子模”集合选择问题。

**💡 创新点**

创新点在于：1) 设计了可对任何集合函数实现无损舍入的 Multinoulli Extension；2) 通过该连续松弛实现参数无关、查询复杂度从 𝑂̃(1/ϵ⁶/³) 降至 𝑂(1/ϵ²)；3) 首次为在线分区约束下的弱子模/弱 DR 子模最大化提供 𝑂(√T) 惩罚率的算法；4) 引入非自明辅助函数与路径积分差分估计提升收敛与逼近性能。

**🔧 技术方法**

采用的技术包括：连续松弛（Multinoulli Extension）、路径积分差分估计、随机连续贪婪（Multinoulli‑SCG）、元动作框架（Multinoulli‑OSCG）、在线梯度上升与辅助梯度估计（Multinoulli‑OSGA）、无损舍入、以及基于多项式分布的概率采样。

**📊 数据集**

实验数据集包括：视频摘要（VSUMM、烹饪视频、动画视频）、特殊最大覆盖、贝叶斯 A‑optimal 设计，以及多目标跟踪模拟（UAV 运动、扩展卡尔曼滤波框架）。

**📈 对比分析**

与传统贪婪、残差随机贪婪、Distorted‑LS‑G 等基线比较，Multinoulli‑SCG 与 Multinoulli‑OSGA/OSCG 在获得更高目标值的同时，查询次数下降 2–4 个数量级；在在线跟踪任务中，算法平均收益显著高于随机和现有 MA‑OSMA/OSGA 基线，且满足理论预期的 𝑂(√T) 惩罚率。

**⚠️ 局限性**

局限性包括：1) 仍需多次评估目标函数（尤其是 Hessian 估计），计算量较大；2) 需要满足单调性和弱子模/弱 DR 子模假设；3) 在线算法需要维护多重线性优化器，内存开销较高；4) 目前仅针对分区约束（Partition Matroid），对更一般约束的推广尚未完成。

---

## 651. KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning

**arXiv ID:** 2603.21440 | [PDF](https://arxiv.org/pdf/2603.21440v1)

**作者:** Shuai Wang `[一作]` (Chalmers University of Technology and University of Gothenburg), Yinan Yu `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该论文提出 KG-Hopper 框架，使 7B 规模开放 LLM 能在一次推理中完成多跳知识图推理。

**💡 创新点**

创新点在于将 RL 训练与 LLM 思考阶段结合，构建统一的思考过程，实现一次性多跳推理，并通过奖励函数精细引导检索与推理。

**🔧 技术方法**

技术包括知识图检索工具、RL（GRPO）强化学习、思考标签、检索/格式/推理/答案奖励、掩码训练和历史重采样。

**📊 数据集**

使用 Freebase 与 Wikidata 的八个 KBQA 评测数据集（CWQ、WebQSP、WebQuestions、GrailQA、QALD10-en、T‑REx、Zero‑Shot RE、Creak）。

**📈 对比分析**

在 Hits@1 指标上，7B KG-Hopper 在多跳任务上超越同尺寸 SFT 模型，且与 70B 模型和 GPT‑4o‑mini 等大模型相当，表现最佳。

**⚠️ 局限性**

局限在于依赖外部 LLM 评判奖励、需要手工构建检索工具与奖励函数，且对 KG 不完整时仍易误判。

---

## 652. LLM-Powered Workflow Optimization for Multidisciplinary Software Development: An Automotive Industry Case Study

**arXiv ID:** 2603.21439 | [PDF](https://arxiv.org/pdf/2603.21439v1)

**作者:** Shuai Wang `[一作]` (Chalmers University of Technology), Dhasarathy Parthasarathy `[通讯]` (Volvo Group)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图的工作流优化方法，利用大型语言模型（LLM）自动化多学科软件开发中的协作与知识转换流程

**💡 创新点**

创新点在于将LLM服务嵌入图模型，逐步替代人工协调，支持增量式采用并保持现有实践不被破坏

**🔧 技术方法**

采用图模型与LLM驱动的工作流服务，结合自然语言处理与代码生成技术

**📊 数据集**

使用沃尔沃集团生产级车载API系统的数据集（192个端点、420个属性、776个CAN信号，涉及六大功能域）

**📈 对比分析**

对比人工流程与自动化流程，自动化实现93.7%的F1分数，单个API开发时间从约5小时缩短至不到7分钟，累计节省约979个工程工时

**⚠️ 局限性**

局限性包括对LLM准确性与上下文理解的依赖、评估仅基于单一行业案例，且在跨领域迁移时需进一步验证适用性

---

## 653. PROMPT2BOX: Uncovering Entailment Structure among LLM Prompts

**arXiv ID:** 2603.21438 | [PDF](https://arxiv.org/pdf/2603.21438v1)

**作者:** Neeladri Bhuiya `[一作]` (University of Massachusetts), Haw-Shiuan Chang `[通讯]` (University of Massachusetts)

**通讯引用:** 1039 | [OpenAlex ID](https://openalex.org/A5080115221)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出使用盒子嵌入（box embedding）将 LLM 提示映射到空间中，能够同时捕获语义相似性与提示的特异性（entailment），并利用该结构进行 LLM 弱点的层级聚类与可视化。

**💡 创新点**

创新点在于：①引入盒子嵌入表示提示，使得更具体的提示自然包含在更通用提示的盒子中；②通过合成与现有的 entailment 数据构建大规模训练集；③设计基于盒子维度的降维方法和新的层级聚类距离度量；④在弱点发现任务上显著优于传统向量基线。

**🔧 技术方法**

使用技术包括：句子 Transformer + 两个 MLP 输出盒子中心与宽度；Gumbel Box 平滑实现盒子交集与包含；对比学习（contrastive）优化语义相似性与 entailment；多源数据训练；基于盒子体积的层级聚类；盒子降维（类似 SNE 的优化）。

**📊 数据集**

使用的数据集包括：Infinity Instruct（提示-响应对）、WildChat（生成层级提示）、SURI（主目标与约束子树）、MultiNLI（句子 entailment 样本）、FollowBench/UltraFeedback（评测与评分）以及通过 GPT-4 生成的链式提示与连接数据。

**📈 对比分析**

与向量基线（MPNet 向量+余弦相似度）及 CSDelta 进行比较。结果显示盒子模型在语义相似性（STS‑B）与 entailment（FollowBench、SURI）上均优于向量；在 UltraFeedback 上的 kNN 评分预测 RMSE 约降低 19%（盒子无链接）；层级聚类中盒子实现了 35% 的局部分数一致性提升、70% 的特异性层级对齐率和约 8.9% 的弱点集群发现率。

**⚠️ 局限性**

主要局限包括：需要大量手工或 GPT 生成的合成 entailment 训练数据，可能导致迁移性不足；盒子嵌入对其他类型的提示关系（如同义、反义）处理有限；实验集中在英文提示，跨语言通用性待验证；降维与聚类方法的超参数选择对结果影响显著。

---

## 654. Image-Based Structural Analysis Using Computer Vision and LLMs: PhotoBeamSolver

**arXiv ID:** 2603.21432 | [PDF](https://arxiv.org/pdf/2603.21432v1)

**作者:** Altamirano-Muñiz Emilio Fernando `[一作]` `[通讯]`, Altamirano-Muñiz Emilio Fernando

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了PhotoBeamSolver，一个能够从手绘梁图像自动识别结构元素并求解理想化梁的端到端系统。

**💡 创新点**

创新点在于将YOLOv5目标检测与大型语言模型（LLM）结合，既实现了高效的图像到符号结构的转换，又利用LLM进行结构方程的符号推理与解算，形成完整的图像-分析管道。

**🔧 技术方法**

采用了计算机视觉中的YOLOv5目标检测、深度学习框架、LLM（如GPT API）进行符号推理、结构方程求解器以及统计学习理论评估。

**📊 数据集**

使用了532张标注的梁图像数据集，包含支座类型、点荷载、分布荷载及几何标记等类别。

**📈 对比分析**

通过与IndeterminateBeam Python包的参考解法比较，使用mAP和类别精度评估检测性能，最终mAP>0.93，解算结果与参考解在机器精度内一致。

**⚠️ 局限性**

局限性包括仅能处理平面理想化梁，无法处理倾斜荷载、桁架或斜梁；缺乏倾斜荷载分解与斜梁本地坐标转换；数据集规模有限，难以推广到真实结构；需要人工干预纠正识别错误。

---

## 655. Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models

**arXiv ID:** 2603.21426 | [PDF](https://arxiv.org/pdf/2603.21426v1)

**作者:** Jingchen Sun `[一作]` (NEC Laboratories America), Changyou Chen `[通讯]` (University at Buffalo)

**通讯引用:** 13899 | [OpenAlex ID](https://openalex.org/A5058575315)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Beta-weighted Knowledge Distillation (Beta-KD)，一种基于贝叶斯视角的无监督加权知识蒸馏框架，用于在多模态大型语言模型(MLLM)中自适应平衡数据监督与教师监督；

**💡 创新点**

创新点在于将教师监督视为Gibbs先验，将蒸馏权重视为不确定性参数β，并通过Laplace近似得到闭式的任务级和实例级自适应权重，使得无需手工调节λ，能统一处理多通道、多损失的蒸馏；

**🔧 技术方法**

使用技术包括贝叶斯MAP推理、Gibbs先验、Laplace近似、可微的β网络（任务级标量或实例级轻量MLP）、多损失组合（CE+KL/JS/TS/cosine等）以及cosine概率对齐；

**📊 数据集**

实验使用ScienceQA小规模和大规模2.2M图文对(包含COCO、SBU、VisualDialog、ShareGPT4V、SQA、IConQA、TextVQA、VSR、VIGC)进行蒸馏，评估六大基准：GQA、SQA、TextVQA、MME、MMBench、POPE；

**📈 对比分析**

与多种基线(CE+KL、Align-KD、TAID、LLaVA-KD等)比较，Beta-KD任务级/实例级在所有基准上均提升0.5–4.0%（平均+1–3%），实例级表现更佳；在MobileVLM 1.7B蒸馏中提升4.7点，整体平均提升约1–2%，并刷新多模态蒸馏的state‑of‑the‑art；

**⚠️ 局限性**

局限性包括：验证范围仅限视觉‑语言LLM，未探讨纯语言或其他模态；对teacher–student差距极大时仍需人工干预；实例级加权需额外轻量网络，虽占比极小但增加了模型复杂度；未讨论推理时的速度与能耗。

---

## 656. Learning Trajectory-Aware Multimodal Large Language Models for Video Reasoning Segmentation

**arXiv ID:** 2603.21488 | [PDF](https://arxiv.org/pdf/2603.21488v1)

**作者:** Jingnan Luo `[一作]` (Southern University of Science and Technology), Feng Zheng `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5993 | [OpenAlex ID](https://openalex.org/A5063285882)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TrajSeg 框架，利用多模态大型语言模型实现视频推理分割，支持基于自然语言指令对视频中目标进行分割；

**💡 创新点**

创新点在于双向文本-轨迹对齐、帧级内容集成（FCI）模块以及统一掩码生成器，三者共同提升轨迹感知、时空一致性并实现端到端训练；

**🔧 技术方法**

采用 LLaVA‑7B 作为大语言模型，SAM‑2 解码器，轨迹编码器、FCI cross‑attention、双向对齐训练策略、统一掩码解码器、LoRA 微调及 DeepSpeed 分布式训练；

**📊 数据集**

预训练使用 ADE20K、COCO‑Stuff、PACO‑LVIS、PASCAL‑Part、Ref‑COCO、Ref‑CLEF、VQA LLaVA‑Instruct‑150k、ReasonSeg；主训练使用 Ref‑YouTube‑VOS、MeViS、ReVOS、Ref‑DAVIS 以及伪视频数据；

**📈 对比分析**

在 Ref‑YouTube‑VOS、Ref‑DAVIS、MeViS 以及 ReVOS 等基准上与现有推理与 RVOS 方法对比，TrajSeg 在 J&F、J、F 指标均取得领先或接近最优，尤其在长视频和复杂推理场景表现尤为突出；

**⚠️ 局限性**

主要局限是受限于采样帧数，难以充分利用长视频上下文；对含模糊或高难度推理的指令仍易出错，需进一步提升模型的深层推理能力。

---

## 657. Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models

**arXiv ID:** 2603.21484 | [PDF](https://arxiv.org/pdf/2603.21484v1)

**作者:** Hyundong Jin `[一作]` (Chung-Ang University), Eunwoo Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 451 | [OpenAlex ID](https://openalex.org/A5074532898)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种基于概念的连续忘记框架 CORE，用来让大规模视觉语言模型在不训练全量数据的前提下，逐步忘记指定的图文对，同时保持整体能力。

**💡 创新点**

通过将图像和文本拆解成细粒度视觉和语言概念，使用概念调制器定位忘记类别，混合拒绝专家根据概念生成拒绝响应，并使用概念相关路由与校准机制实现对序列忘记任务的高效适应。

**🔧 技术方法**

概念模块、概念调制器、混合拒绝专家（Refusers）、路由器与参考度量、概念驱动的转发、拒绝校准；利用 CLIP、Q‑Former 进行概念对齐，并采用交叉熵、对比损失等训练目标。

**📊 数据集**

视觉语言问答安全基准（含武器、暴力等6类）、ImageNet‑R（20 类任务）以及 MMBench、SEEDBench、ScienceQA 等通用 LVLM 基准。

**📈 对比分析**

与 EWC、LwF、GMM、EProj、SCRUB、MoEAdapter、O^3 等持续学习/忘记方法对比，CORE 在 Avg/Last 的 Context‑Aware Refusal Rate 与 Answer Rate 均明显领先，尤其在最后一步 CRR 达 90%+、AR 88%+，显著减少过度拒绝。

**⚠️ 局限性**

对概念描述的依赖较高，若概念库不足或划分模糊可能导致拒绝不精准；实验仅在预训练 LVLM 结构固定下进行，扩展至更大模型或不同架构仍需验证。

---

## 658. Semantic Shift: the Fundamental Challenge in Text Embedding and Retrieval

**arXiv ID:** 2603.21437 | [PDF](https://arxiv.org/pdf/2603.21437v1)

**作者:** Hang Gao `[一作]` (Rutgers University), Dimitris N. Metaxas `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了Transformer文本嵌入中由于语义漂移导致的聚集与奇异性问题，并提出可计算的语义漂移度量来解释检索性能下降，同时基于此度量设计了自适应段切分器。

**💡 创新点**

创新点在于将语义漂移作为嵌入聚合导致的语义模糊的根本因子，给出理论证明并定义结合局部演化与全局分散的可度量指标，证明其比文本长度或平均距离更能预测检索效果。

**🔧 技术方法**

主要技术包括Transformer编码器与池化（均值/CLS）、语义漂移度量（Local(k)×Disp(k)）、MPD、Self-overlap@k等评估指标，以及对不同拼接模式的实验对照。

**📊 数据集**

使用的数据集包括ArXiv论文文本、Alice's Adventures in Wonderland小说，并在多种预训练模型（bge-large、e5-large、all-mpnet、gte-large等）上进行实验。

**📈 对比分析**

通过在重复、顺序和随机拼接三种模式下计算MPD和检索自重叠@k，实验显示语义漂移与嵌入聚集和检索损失高度相关，且语义漂移度量比单纯文本长度更能准确预测检索性能。

**⚠️ 局限性**

局限性包括只从池化角度解释嵌入，未考虑更细粒度的token交互；语义漂移度量基于余弦距离，可进一步改进；实验覆盖的语言和专业领域有限。

---

## 659. HyReach: Vision-Guided Hybrid Manipulator Reaching in Unseen Cluttered Environments

**arXiv ID:** 2603.21421 | [PDF](https://arxiv.org/pdf/2603.21421v1)

**作者:** Shivani Kamtikar `[一作]` (University of Illinois at Urbana Champaign), Girish Chowdhary `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个实时的刚软混合连续操纵器，结合视觉感知、三维重建、形状感知路径规划和学习控制，实现了在无结构、杂乱环境中自主到达目标。

**💡 创新点**

创新点在于：①将多视角RGB重建与形状感知路径规划相结合，实时考虑软臂的可变形；②提出基于软臂常数曲率模型的在线形状估计；③使用学习式闭环控制器，使混合操纵器能精确到达任意6-DoF目标，且无需环境特定的重训练。

**🔧 技术方法**

技术包括：多视角RGB重建（Mast3r）、YOLO-World目标检测、占据格网环境建模、改进的RRT*形状感知规划、常数曲率软臂模型、20层带瓶颈与残差块的MLP学习控制器。

**📊 数据集**

使用了自采集的真实实验数据：四个逐渐复杂的物理实验环境，包含桌面、障碍、植物与墙壁/孔洞场景；并利用Mast3r重建与YOLO-World检测得到的目标位置。

**📈 对比分析**

与基线（仅刚体、Img2Act视觉驱动）比较，混合系统在SR@2cm、SR@Touch和平移误差上均显著提升；在最困难的“墙洞”场景，成功率分别为54.5%（SR@2cm）和27.3%（SR@Touch），显著高于基线。

**⚠️ 局限性**

主要局限：控制器未显式最小化能量/动作变化；未考虑负载影响和动态障碍；未利用障碍协同寻路，仅实现避障；缺乏对高负载和动态环境的适应性。

---

## 660. SafePilot: A Framework for Assuring LLM-enabled Cyber-Physical Systems

**arXiv ID:** 2603.21523 | [PDF](https://arxiv.org/pdf/2603.21523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 661. HACMatch Semi-Supervised Rotation Regression with Hardness-Aware Curriculum Pseudo Labeling

**arXiv ID:** 2603.21583 | [PDF](https://arxiv.org/pdf/2603.21583v1)

**作者:** Mei Li `[一作]` (Shanghai Jiao Tong University), Hongtao Lu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8317 | [OpenAlex ID](https://openalex.org/A5102899381)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于难度感知的课程学习框架 HACMatch 以及结构化数据增强 PoseMosaic，用于半监督 3D 旋转回归。

**💡 创新点**

创新点在于将伪标签过滤从固定熵阈值改为动态难度感知的多阶段或自适应课程学习，并设计了局部多种增强组合的 PoseMosaic，兼顾特征多样性与几何完整性。

**🔧 技术方法**

使用学生‑教师一致性训练、矩阵费舍尔分布损失、熵难度度量、结构化 Patch Mosaic 增强以及 ResNet18 骨干网络。

**📊 数据集**

在 PASCAL3D+ 与 ObjectNet3D 两大数据集上进行实验，标注比例分别设为 5%、10%、20%。

**📈 对比分析**

与 Sup.-Fisher、Sup.-Laplace、FisherMatch 等基线对比，HACMatch 在低标注比例下平均提升 Mean Med 约 2–4°、ACC@30° 提升 5–8%，性能优于现有方法。

**⚠️ 局限性**

局限性包括仍需手动调参（课程比例、阈值）、增强池需经验筛选，且对极端遮挡或多样未标注样本仍可能产生伪标签噪声。

---

## 662. Kolmogorov Complexity Bounds for LLM Steganography and a Perplexity-Based Detection Proxy

**arXiv ID:** 2603.21567 | [PDF](https://arxiv.org/pdf/2603.21567v1)

**作者:** Andrii Shportko `[一作]` (Northwestern University), Andrii Shportko `[通讯]` (Northwestern University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5092831382)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大语言模型（LLM）中语义保持的隐写（Steganographic Embedding）对输出Kolmogorov复杂度的影响，并提出可计算的复杂度代理——Binoculars分数；

**💡 创新点**

创新点在于证明了语义保持的隐写必然导致Kolmogorov复杂度至少增加payload复杂度K(P)（减去O(log n)），并将语言模型交叉熵与复杂度关联，首次将Binoculars分数作为可观测的隐写检测指标；

**🔧 技术方法**

主要技术包括Kolmogorov复杂度理论、信息理论隐写安全度量、语言模型交叉熵作为概率压缩器、Binoculars比值分数以及针对LLM的编码与解码算法；

**📊 数据集**

实验数据集为300条《三体》文本片段与8色彩调色板的随机组合，构造覆盖文本与嵌入文本对；

**📈 对比分析**

与直接解码及对抗式改写（LLM paraphraser）相比，编码文本的Binoculars分数显著上升（均值1.649→1.338，p<10⁻⁶），证明了理论预期的复杂度增加；

**⚠️ 局限性**

局限性包括对语义负载的理想化假设、理论上限与实际LLM之间的误差、单一LLM家族实验、以及Binoculars分数提升幅度虽显著但单阈值检测误差仍不小。

---

## 663. SynSym: A Synthetic Data Generation Framework for Psychiatric Symptom Identification

**arXiv ID:** 2603.21529 | [PDF](https://arxiv.org/pdf/2603.21529v1)

**作者:** Migyeong Kang `[一作]` (Sungkyunkwan University), Jinyoung Han `[通讯]` (Sungkyunkwan University)

**通讯引用:** 25973 | [OpenAlex ID](https://openalex.org/A5040976081)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了SynSym框架，利用大语言模型自动生成多样化、临床可解释的精神症状表达文本，用以训练和评估多标签精神症状识别模型。

**💡 创新点**

首次将LLM用于生成精神症状级别的数据，包含症状概念扩展、双风格表达、真实共现模式下的多症状生成和合成数据质量评估，且无需人工标注即可获得大规模、可靠的训练语料。

**🔧 技术方法**

使用GPT‑4o等大型语言模型进行提示工程，结合临床共现知识、双风格（临床/口语）生成、合成数据质量评估；训练采用MentalBERT等Transformer模型。

**📊 数据集**

评测基准为三大抑郁症症状数据集：PsySym、PRIMATE、D2S；合成数据包含14种抑郁症状共18,254条样本。

**📈 对比分析**

与BERT/DeBERTa/MentalBERT、GPT‑4o零样本等基线对比；仅用合成数据训练的模型与使用真实数据训练的模型表现相当，进一步微调后性能优于所有基线；在跨数据集泛化测试中表现最稳健，提升显著。

**⚠️ 局限性**

目前仅在抑郁症场景验证，缺乏对其他精神疾病的实验；合成语料排除了比喻/隐喻表达，导致语言多样性不足；需要进一步扩展到更多症状、验证更广泛的应用场景。

---

## 664. Generalizable Self-Evolving Memory for Automatic Prompt Optimization

**arXiv ID:** 2603.21520 | [PDF](https://arxiv.org/pdf/2603.21520v1)

**作者:** Guanbao Liang `[一作]` (Zhejiang University), Jiajun Bu `[通讯]` (Zhejiang University)

**通讯引用:** 13264 | [OpenAlex ID](https://openalex.org/A5052757755)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于记忆驱动的自动提示优化框架 MemAPO，能够在多任务下持续积累和复用提示经验。

**💡 创新点**

创新点在于将提示优化视为持续经验积累，采用双重记忆机制（成功策略模板和错误模式）以及自我反思与记忆编辑，支持跨任务泛化并随时间自我演化。

**🔧 技术方法**

采用了LLM自回归的自我反思循环、语义相似度检索、模板与错误模式抽象、以及基于元提示的策略生成与更新；核心模型为GPT‑4o‑mini/Qwen3‑8B。

**📊 数据集**

使用了 Big‑Bench Hard/Extra Hard、AGIEval 子任务（AQuA‑RAT、Gaokao‑Math、Gaokao‑Geography、Gaokao‑History）等逻辑推理与知识问答数据集。

**📈 对比分析**

与 ProTeGi、OPRO、TextGrad、PromptBreeder、SPO 等代表性自动提示优化方法以及 IO、CoT、Step‑Back、RaR 等传统提示技术对比，MemAPO 在 GPT‑4o‑mini 上平均准确率达到 70.7%，比最佳基线 TextGrad 提升 7.1%，并将优化成本降低约 58%。

**⚠️ 局限性**

限制在于目前仅处理文本推理任务，未考虑多模态信息，且记忆结构需进一步扩展以支持更广泛的智能体环境。

---

## 665. Triangulating Temporal Dynamics in Multilingual Swiss Online News

**arXiv ID:** 2603.21519 | [PDF](https://arxiv.org/pdf/2603.21519v1)

**作者:** Bros Victor `[一作]` (Idiap Research Institute), Gatica-Perez Daniel `[通讯]` (Idiap Research Institute)

**通讯引用:** 14194 | [OpenAlex ID](https://openalex.org/A5012965551)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了瑞士三大语种（法语、德语、意大利语）数字媒体在2019‑2023年间对不同类型事件（单一、主题、周期性）的报道时序与模式；

**💡 创新点**

提出了跨语言三角测量方法，融合词汇多样性、命名实体链接、定向情感与共识变点检测，结合定性解释，系统阐释多语言公共话语的演变；

**🔧 技术方法**

使用了词汇多样性（TTR）、词汇密度、句子长度标准化；GLiNER+ mGENRE 进行命名实体识别与 Wikidata 链接；目标情感分类模型；Consensus‑based change‑point detection（PELT、Binary Segmentation、Bayesian）等；

**📊 数据集**

共1.7M篇新闻文章（2019‑2023），来源自 CCNews（CommonCrawl 子集），按语言分为法语（332k）、德语（1.25M）、意大利语（122k）三大语料；

**📈 对比分析**

通过标准化指标在三语种间对比；使用“domestication profile”和“proximity salience ratio”度量本土化与文化亲近度；变点检测识别议程周期；结果显示事件类型与语种均产生显著不同的时序特征，说明方法能捕捉多语言媒体动力；

**⚠️ 局限性**

局限包括：仅使用 CCNews 可能遗漏部分报道；事件关键词预设可能导致检索偏差；NER、链接与情感模型误差影响指标；词汇指标未覆盖语义深层话语；未考虑媒体所有权、编辑方针等结构性因素；

---

## 666. Would You Like to Visit My World? Cultivating Perceived Equality in Human-Agent Interaction via Observable Social Life Spaces

**arXiv ID:** 2603.21505 | [PDF](https://arxiv.org/pdf/2603.21505v1)

**作者:** Zihong He `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Hai-Ning Liang `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Observable Social Life Spaces (OSLS) 原理，构建可观测的 AI 代理社会生活空间，并通过可视化展示其日常活动，提升人机互动中的平等感。

**💡 创新点**

创新点在于将代理的独立生活轨迹与对话记忆双轨并行，并将其可视化为持续的虚拟社交空间，证明视觉可观测是实现人机平等的关键因子。

**🔧 技术方法**

采用大型语言模型（DeepSeek Chat 与 GPT‑4.1‑mini）驱动对话与行为规划，使用 A* 路径规划实现代理在 2D 虚拟环境中的移动，双轨记忆机制结合 LLM 生成摘要，配合触摸界面实现用户交互。

**📊 数据集**

数据集为自建实验数据：24 名参与者两轮交互记录（约 30 分钟总时长），无公开公开数据集。

**📈 对比分析**

通过三组实验（Baseline、Unobservable、Observable）进行混合方法研究；量化结果显示 Observable 组在“平等感”维度显著高于 Baseline（p=0.006）和 Unobservable（p=0.031）；质性访谈表明可视化显著提升用户将代理视为伙伴/朋友的比例。

**⚠️ 局限性**

局限性包括：仅依赖自报主观指标，缺乏客观行为或生理测量；实验时间短，缺乏长期纵向验证；代理行为相对被动，未充分展示主动性；仅在单一场景与角色（厨师）下测试，需扩展至多角色与多用户环境。

---

## 667. Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection

**arXiv ID:** 2603.21511 | [PDF](https://arxiv.org/pdf/2603.21511v1)

**作者:** Kaiqiang Li `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Jin Wan `[通讯]` (Qilu University of Technology (Shandong Academy of Sciences))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 BTP（Back To Point）框架，利用预训练的点语言模型直接在三维点云中进行零样本异常检测与定位。

**💡 创新点**

创新点包括：①多粒度特征嵌入模块 MGFEM，将多层语义特征、几何描述和全局 CLS 进行融合；②几何特征创建模块 GFCM，学习可微的点几何表示；③联合表示学习策略，结合全局、局部与几何监督，提升跨模态对齐与细粒度定位能力。

**🔧 技术方法**

核心技术为预训练点语言模型 ULIP2、点云 Transformer（Point‑BERT）提取语义特征、FPFH/PointNet 进行几何编码、混合学习目标（焦点、Dice、BCE、对比损失）以及可学习文本提示。

**📊 数据集**

使用工业真实数据集 Real3D‑AD（12 类）和仿真数据集 Anomaly‑ShapeNet（40 类）进行评估。

**📈 对比分析**

与现有 2D 渲染+VLM、无监督与监督 3D 方法相比，BTP 在点级 AUROC 及 AU‑PRO 上分别达到 84.5% / 81.9%（Real3D‑AD）和 87.3% / 82.2%（Anomaly‑ShapeNet），在多数类别中排名第一，明显优于其他零样本方法且接近或优于部分监督基线。

**⚠️ 局限性**

局限性包括：1) 对整体对象级别检测的性能仍低于监督方法；2) 依赖 ULIP 等大规模预训练模型，训练与推理成本相对较高；3) 对极稀疏或极大点云的扩展尚未充分验证；4) 仍需改进全局异常分数与局部定位的融合策略。

---

## 668. Optimizing Feature Extraction for On-device Model Inference with User Behavior Sequences

**arXiv ID:** 2603.21508 | [PDF](https://arxiv.org/pdf/2603.21508v1)

**作者:** Chen Gong `[一作]` (Shanghai Jiao Tong University), Guihai Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18108 | [OpenAlex ID](https://openalex.org/A5100428808)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发AutoFeature，针对移动端机器学习推理中用户行为日志特征提取的瓶颈，提出自动化特征提取优化引擎，通过图抽象、图优化与缓存三步消除冗余操作，显著降低端到端推理延迟。

**💡 创新点**

首次系统性识别并量化移动端模型推理中用户行为日志特征提取的主要耗时，提出基于特征提取DAG、子链拆分与分层过滤、以及事件级缓存的全新解决方案，实现跨特征、跨推理间冗余消除，提升效率而不牺牲准确率。

**🔧 技术方法**

采用DAG抽象与自动化图优化（子链拆分、跨特征融合、分层过滤算法）、事件级缓存决策（0/1背包贪心策略）、SQLite查询、日志解码、过滤与聚合等技术。

**📊 数据集**

使用ByteDance旗下五大移动服务（搜索、视频预加载、关键词预测、商品推荐、视频推荐）的真实用户日志，10名测试用户两天日志，结合行业标准无优化特征提取流程作为基线。

**📈 对比分析**

与行业标准（w/o AutoFeature）、仅融合（w/ Fusion）和仅缓存（w/ Cache）三种基线对比，AutoFeature在所有服务中将端到端推理延迟降至20 ms以下，平均加速比1.33×–4.53×，单服务最高可达4.53×；相比云端预计算方案，延迟降低30%+且存储增幅被压缩到≤100 KB。

**⚠️ 局限性**

仅在特征提取占主导的场景有效，对低活跃用户或大模型推理时影响有限；实现与模型训练分离，缺乏模型与引擎协同优化机会；极低内存设备的缓存策略仍受限。

---

## 669. Computing the Girth of a Segment Intersection Graph

**arXiv ID:** 2603.21585 | [PDF](https://arxiv.org/pdf/2603.21585v1)

**作者:** Timothy M. Chan `[一作]` (University of Illinois Urbana-Champaign), Yuancheng Yu `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种期望时间为O(n^1.483)的算法，用于计算平面直线段交叉图的最短环（girth）

**💡 创新点**

创新点在于首次突破O(n^1.5)的上界，结合了有界差值的最小-加矩阵乘法与一种新的平面图分割定理

**🔧 技术方法**

核心技术包括子立方的有界差值min-plus乘法、变形的平面图分割定理、合法交替闭路构造以及分治+色彩编码的改进

**📊 数据集**

实验上使用的并未给出具体数据集，而是以理论分析和随机期望时间为主，证明了在任何直线段集合上均适用

**📈 对比分析**

与之前的Chan（SODA 2023）O(n^1.5)算法相比，该方法在理论上实现了更快的时间复杂度，并在检测三角形和四边形时结合已知的O(n^1.408)和O(n)子程序进一步提升整体性能

**⚠️ 局限性**

局限性包括算法依赖随机化、需要预先检测并处理girth≤4的情况、对更一般的代数曲线或半代数集合的扩展仍存在技术挑战，并且目前尚未达到最优上界

---

## 670. AI In Cybersecurity Education -- Scalable Agentic CTF Design Principles and Educational Outcomes

**arXiv ID:** 2603.21551 | [PDF](https://arxiv.org/pdf/2603.21551v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 671. Adaptive Robust Estimator for Multi-Agent Reinforcement Learning

**arXiv ID:** 2603.21574 | [PDF](https://arxiv.org/pdf/2603.21574v1)

**作者:** Zhongyi Li `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**通讯引用:** 10226 | [OpenAlex ID](https://openalex.org/A5102969899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个鲁棒的多智能体强化学习框架，利用Dual-Agent Answer–Critique–Rewrite（DACR）三阶段交互协议和Adaptive Robust Estimator（ARE）对大型语言模型协同推理进行稳定性与准确率提升。

**💡 创新点**

创新点包括①设计结构化三阶段交互，明确生成与评估分离并实现信用分配；②通过跨阶段改进奖励机制量化每个代理的边际贡献；③引入ARE替代GRPO的批量均值，用自适应鲁棒位置估计对抗重尾噪声，并给出理论收敛性保证。

**🔧 技术方法**

采用多智能体强化学习（GRPO变体）、鲁棒统计（Median-of-Means 与自适应损失优化）、梯度裁剪策略优化、跨阶段奖励设计以及生成-评估循环。

**📊 数据集**

实验基准包括数学推理数据集MATH、AMC23、Gaokao2023、MinervaMath，以及空中视觉语言导航（VLN）任务FlightGPT环境，使用Qwen2-VL-2B模型。

**📈 对比分析**

与基线GRPO+无DACR/ARE进行同构/异构对比，结果显示在ID与OOD测试中准确率提升约10%+，在重尾噪声下显著优于基线；在VLN任务中，ARE在训练奖励、NE、SR、OSR、SPL上均优于FlightGPT，尤其在未见环境中改进最大。

**⚠️ 局限性**

局限性包括仅验证双智能体同构/异构对，未扩展到多于两位代理；交互协议与奖励设计对任务特定，鲁棒估计仍有计算开销；在VLN任务中仅验证ARE，未实现完整多智能体交互；对极端噪声分布的理论与实验仍有限。

---

## 672. Counterfactual Credit Policy Optimization for Multi-Agent Collaboration

**arXiv ID:** 2603.21563 | [PDF](https://arxiv.org/pdf/2603.21563v1)

**作者:** Zhongyi Li `[一作]` (Beihang University), Fuzhen Zhuang `[通讯]` (Beihang University)

**通讯引用:** 10226 | [OpenAlex ID](https://openalex.org/A5102969899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种针对协同大型语言模型的反事实信用策略优化（CCPO）框架，解决共享全局奖励导致的信用分配不准确和自由搭车问题。

**💡 创新点**

创新点在于构建轻量级、基于协作拓扑的反事实基线，通过移除单个智能体的贡献来估计其边际贡献，并引入全局历史归一化来稳定优势估计；同时针对顺序 Think–Reason 和多智能体投票两种协作模式给出了可行的实现。

**🔧 技术方法**

使用了强化学习中的策略梯度方法（GRPO）、反事实优势估计、指数移动平均归一化，以及多智能体训练中的集中训练、分散执行框架。

**📊 数据集**

在数学推理数据集 MATH、LogiQA 以及其子集（MATH500、AMC23、Gaokao2023en、MinervaMath 等）上进行实验。

**📈 对比分析**

与共享奖励基线、ReMA、单智能体 GRPO 等方法对比，CCPO 在多任务、不同模型规模下均取得更高的准确率和更平稳的收敛；例如在 MATH500 上从 60% 提升到 77%，在 LogiQA 上从 43% 提升到 45%。

**⚠️ 局限性**

局限性包括：计算开销相对较高（需要额外的反事实推理）、仅针对两种典型协作拓扑验证，未探索更复杂的交互图或在线多任务场景，且对极度不可靠或极度异构的智能体协作效果尚未充分评估。

---

## 673. From Part to Whole: 3D Generative World Model with an Adaptive Structural Hierarchy

**arXiv ID:** 2603.21557 | [PDF](https://arxiv.org/pdf/2603.21557v1)

**作者:** Bi'an Du `[一作]` (Peking University), Wei Hu `[通讯]` (Peking University)

**通讯引用:** 10865 | [OpenAlex ID](https://openalex.org/A5022039557)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于自适应部件–整体层次结构的单图像3D生成模型，利用可变数量的结构槽和共享的几何原型实现更完整、更结构一致的三维重建。

**💡 创新点**

创新点在于①自适应槽门控机制，可在不预设部件数的情况下自动激活合适的槽；②类无关原型库，软对齐槽向量至共享几何原型并注入残差，提升跨类别共享与合成；③使用掩码流式监督仅对真实部件槽进行训练，保持预训练先验的稳定性。

**🔧 技术方法**

技术手段包括：3D VAE+DiT 直方流框架、结构槽的平均嵌入与门控网络、原型对齐与残差注入、掩码流式匹配损失以及多阶段训练策略。

**📊 数据集**

使用的数据集包括高质量的 Objaverse 子集、ShapeNet‑Core、Amazon Berkeley Objects（ABO）以及用于场景生成的 3D‑Front。

**📈 对比分析**

与 TripoSG、HoloPart、PartCrafter 以及 MIDI 等基线对比，模型在 Objaverse、ShapeNet 和 3D‑Front 上在 Chamfer 距离、F‑Score 以及部件重叠 IoU 等指标上均实现了显著提升，尤其在遮挡强的单视图场景下效果更为突出。

**⚠️ 局限性**

局限性包括：需要已标注的 canonical 部件数；对部件数超过预设上限或极为复杂的重叠物体仍可能产生误差；模型规模较大，推理速度受限。

---

## 674. One-Year Internship Program on Software Engineering: Students' Perceptions and Educators' Lessons Learned

**arXiv ID:** 2603.21548 | [PDF](https://arxiv.org/pdf/2603.21548v1)

**作者:** Golnoush Abaei `[一作]` (RMIT University), Maria Spichkova `[通讯]` (RMIT University)

**通讯引用:** 1453 | [OpenAlex ID](https://openalex.org/A5049584186)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 RMIT 大学软件工程专业一年期实习项目进行纵向评估，收集并分析学生月度报告，探讨学生对课程的价值感知、实习中遇到的挑战，并总结过去十年的课程协调经验与教训。

**💡 创新点**

首次系统整合学生感知与教师经验，提出三大挑战主题（任务复杂度、技术学习、团队协作与支持），并根据实习反馈对课程结构、评估方式和行业合作提出可操作性建议；同时利用十年历程数据展示实习项目演进路径。

**🔧 技术方法**

采用 NVivo 软件进行主题编码与质性分析；使用 Microsoft Forms 及手工统计实现月度报告的收集与定量计数；结合课程目录与学生报告中的关键词进行课程类别归类。

**📊 数据集**

91 份来自 13 名同意参与的学生在 2023‑2024 年期间提交的月度报告（约 7,000 词文本）；附带 99 名学生整体实习数据（49 家企业、各类公司规模比例）。

**📈 对比分析**

通过定量计数（如课程出现频次）和定性主题分析（NVivo）来比较学生对不同课程的感知；未进行传统意义上的性能对比或实验评估，报告结果主要呈现为描述性统计与主题摘要。

**⚠️ 局限性**

样本量有限（仅 13 名学生，占 99 名实习生的 13%），数据来自单一高校且仅涵盖近两年报告，可能存在自选偏差；未考虑学生在实习前后能力的量化变化，缺乏对比基线；研究聚焦于报告文本，无法全面覆盖所有实习场景和行业差异。

---

## 675. PROBE: Diagnosing Residual Concept Capacity in Erased Text-to-Video Diffusion Models

**arXiv ID:** 2603.21547 | [PDF](https://arxiv.org/pdf/2603.21547v1)

**作者:** Yiwei Xie `[一作]` (Huazhong University of Science and Technology), Ping Liu `[通讯]` (University of Nevada)

**通讯引用:** 15046 | [OpenAlex ID](https://openalex.org/A5100646646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PROBE 诊断协议，通过在冻结的文本到视频扩散模型上优化伪词嵌入来测量已被抹除概念的可再激活能力，并通过多层评估框架检测残留容量与视频特有的时间重现失败模式。

**💡 创新点**

创新点包括：① 仅使用一个伪词嵌入在冻结参数下进行诊断；② 在传统文本反演基础上加入对齐损失以约束恢复到原始概念的时空结构；③ 设计多层评估（分类器、语义相似度、时间重现曲线与人工验证）以捕获不同维度的残留；④ 系统评估三大架构、三种抹除策略与三类概念，揭示抹除深度与残留容量的相关性。

**🔧 技术方法**

技术手段包括：文本到视频扩散模型（CogVideoX、Wan2.2）、v‑prediction 与 flow‑matching 采样；伪词优化（类似文本反演）结合重建损失与对齐损失；基于 ResNet‑50、NudeNet、ArcFace、CLIP 的检测器；CFG 调节、AdamW 训练；时间重现曲线计算。

**📊 数据集**

使用模型自身生成的参考视频集：每个概念类别（对象、NSFW、名人）分别采集 20–100 条视频；在三大架构（CogVideoX‑2B/5B、Wan2.2‑5B）上进行实验；对比三种抹除策略（NegPrompt、SAFREE、T2VUnlearning）以及对抗式提示搜索（P4D‑K）。

**📈 对比分析**

与传统抹除方法的直接推理结果、对抗提示搜索以及跨模型/跨方法的迁移实验相比较。实验表明：所有方法在直接推理下均能显著抑制目标概念，但 PROBE 可恢复 10–30% 以上的检测率；抹除深度越大（从输入到权重级别）残留越少；时间重现曲线揭示视频特有的“延迟重现”现象。相比 P4D‑K，PROBE 的诊断信号更强，跨模型迁移效果也更好。

**⚠️ 局限性**

局限性：① 仅能探测通过伪词可激活的残留，无法证明真正实现了完整的表示层抹除；② 需要参考视频集，若概念样本不足或分布偏差，诊断效果受限；③ 对极端抹除（如 T2VUnlearning 在 5B 级别下）可能无法在冻结参数中找到足够梯度；④ 仍未覆盖所有潜在的失败模式（如对抗性噪声、语义迁移）。

---

## 676. Auction-Based Task Allocation with Energy-Conscientious Trajectory Optimization for AMR Fleets

**arXiv ID:** 2603.21545 | [PDF](https://arxiv.org/pdf/2603.21545v1)

**作者:** Jiachen Li `[一作]` (University of Texas at Austin), Dongmei Chen `[通讯]` (University of Texas at Austin)

**通讯引用:** 13118 | [OpenAlex ID](https://openalex.org/A5100677493)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种分层两阶段框架，将基于拍卖的任务分配与每台自动移动机器人（AMR）的能量意识轨迹优化相结合，用于异质工作空间中的多机器人任务调度。

**💡 创新点**

创新点在于：①将物理驱动的能量模型直接嵌入拍卖竞价函数；②提出闭式近似竞价公式并在多阶段决策中实现；③引入事件触发的热重启重调度机制，并给出触发频率上界；④通过实验证明能量竞价与距离竞价在摩擦异质性不同的工作区表现差异，给出可操作的决策规则。

**🔧 技术方法**

技术方法包括：基于闭式能量近似的顺序拍卖；每台机器人独立求解非线性最优控制问题（OCP）以获得能量最优轨迹；碰撞避免通过对偶式接近惩罚在轨迹优化后进行细化；事件触发热重启重调度逻辑；以及Python 3.12 + 单线程实现。

**📊 数据集**

使用模拟数据集：505个实验场景，涵盖三种工厂布局（网格、随机、聚集），任务数从2到100，机器人数从2到20，机器人的物理参数（质量、速度上限、摩擦系数等）采用论文中给出的标准化数值。

**📈 对比分析**

与四个基线（最近任务、最近机器人+固定速度、欧几里得距离竞价、枚举分配）比较，实验显示：①两种拍卖变体平均比最近任务启发式节能11.8%，最高可达25.4%；②与仅轨迹优化的基线相比，能量竞价进一步节能9–18%；③与欧几里得距离竞价相比，在摩擦均匀区能量竞价稍逊（-3.5%），在摩擦异质区能量竞价则优于距离竞价（+2–2.4%）。重调度延迟平均5.6 ms，能量开销仅6% 以内。

**⚠️ 局限性**

局限性包括：①闭式能量近似误差在短段和近似竞价时有13% 排序错误；②对物理模型的参数假设需平台特定识别，未在真实硬件上验证；③碰撞避免仅在轨迹后期细化，未在拍卖阶段预见；④对大规模实时系统的可扩展性仍需进一步验证。

---

## 677. Evolutionary Biparty Multiobjective UAV Path Planning: Problems and Empirical Comparisons

**arXiv ID:** 2603.21544 | [PDF](https://arxiv.org/pdf/2603.21544v1)

**作者:** Kesheng Chen `[一作]`, Yatong Chang `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

无法获取论文内容，无法说明所做工作

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法比较方法与性能

**⚠️ 局限性**

缺乏信息

---

## 678. Non-Exclusive Notifications for Ride-Hailing at Lyft I: Single-Cycle Approximation Algorithms

**arXiv ID:** 2603.21533 | [PDF](https://arxiv.org/pdf/2603.21533v1)

**作者:** Farbod Ekbatani `[一作]` (University of Chicago Booth School of Business), Shreya Reddy `[通讯]` (Lyft, Inc.)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了乘客叫车平台中非专属（广播式）通知的最优策略，正式建模了单周期通知集选择问题，并针对两种冲突解决协议（First Acceptance与Best Acceptance）进行理论复杂度与算法设计；

**💡 创新点**

证明该问题在两种协议下均为强NP‑难，首次给出单骑手FA问题的PTAS，以及多骑手FA问题的常数近似（通过MNL代理+配置LP+拉取单骑手PTAS实现）；证明BA问题的价值函数是单调子模，直接得到(1-1/e)近似，并在同质接受概率下给出多项式时间最优解；

**🔧 技术方法**

采用组合优化、子模福利最大化、MNL离散选择模型、配置线性规划（伴随椭圆法求解）与近似分离器、相关性间隙分析、动态规划构造需求算子、离散化阈值与桶化技术；

**📊 数据集**

在合成随机实例（匹配数与司机数从几到数十）以及基于Lyft真实叫车数据的校准实例上进行评估；

**📈 对比分析**

与ED基线、贪心启发式以及全局最优解进行对比；实验显示FA与BA近似算法在合成数据上平均近似比率分别约0.99与0.98，在Lyft真实数据上更是接近1，明显优于贪心和ED；

**⚠️ 局限性**

受限于问题强NP‑难，仍无法得到FPTAS；算法主要针对单周期模型，未考虑长期动态交互；假设接受与响应时间独立；椭圆法与配置LP的实现复杂度在大规模实例上仍为挑战；

---

## 679. When the Abyss Looks Back: Unveiling Evolving Dark Patterns in Cookie Consent Banners

**arXiv ID:** 2603.21515 | [PDF](https://arxiv.org/pdf/2603.21515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 680. SSAM: Singular Subspace Alignment for Merging Multimodal Large Language Models

**arXiv ID:** 2603.21584 | [PDF](https://arxiv.org/pdf/2603.21584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 681. Non-Exclusive Notifications for Ride-Hailing at Lyft II: Simulations and Marketplace Analysis

**arXiv ID:** 2603.21531 | [PDF](https://arxiv.org/pdf/2603.21531v1)

**作者:** Farbod Ekbatani `[一作]` (University of Chicago), Shreya Reddy `[通讯]` (Lyft)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究将单向专属派单改为广播式非专属派单，分析其对乘客匹配时长、匹配质量和匹配吞吐量的长期影响。

**💡 创新点**

提出统一的非专属派单理论框架，量化 First-Accept 与 Best-Accept 的速度-质量权衡，并使用宏观均衡模型验证长期效益。

**🔧 技术方法**

结合离散事件仿真、整数规划与子模优化、以及连续时间马尔可夫链的宏观平衡分析。

**📊 数据集**

使用 Lyft 在纽约市某日 20 分钟内的真实乘客请求与司机轨迹数据（约 507 位乘客与 1122 位司机）。

**📈 对比分析**

在多种通知打包策略和争用协议（FA/BA/k-Accept）下与传统 ED 基线进行比较，实验显示 NED 在匹配时长、匹配质量和匹配吞吐量上均优于 ED；FA 更快但质量略低，BA 质量最高但时延最长。

**⚠️ 局限性**

仅假设司机接受概率不随通知方式变化，未考虑司机战略行为或实时动态调整；宏观模型简化了空间与时间相关性，可能对精细效应产生偏差。

---

## 682. BOxCrete: A Bayesian Optimization Open-Source AI Model for Concrete Strength Forecasting and Mix Optimization

**arXiv ID:** 2603.21525 | [PDF](https://arxiv.org/pdf/2603.21525v1)

**作者:** Bayezid Baten `[一作]`, Nishant Garg `[通讯]` (University of Illinois)

**通讯引用:** 1429 | [OpenAlex ID](https://openalex.org/A5073814953)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

未提供论文内容

**💡 创新点**

未提供论文内容

**🔧 技术方法**

未提供论文内容

**📊 数据集**

未提供论文内容

**📈 对比分析**

未提供论文内容

**⚠️ 局限性**

未提供论文内容

---

## 683. CataractSAM-2: A Domain-Adapted Model for Anterior Segment Surgery Segmentation and Scalable Ground-Truth Annotation

**arXiv ID:** 2603.21566 | [PDF](https://arxiv.org/pdf/2603.21566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 684. Efficient Failure Management for Multi-Agent Systems with Reasoning Trace Representation

**arXiv ID:** 2603.21522 | [PDF](https://arxiv.org/pdf/2603.21522v1)

**作者:** Lingzhe Zhang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**通讯引用:** 22905 | [OpenAlex ID](https://openalex.org/A5100414156)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 EAGER 框架，通过基于推理轨迹的表示来实现多智能体系统的高效失败管理。

**💡 创新点**

创新点在于：①利用历史失败模式进行实时检测与自反性缓解；②引入推理范围对比学习（intra‑scope 与 inter‑scope）构建统一的轨迹表示；③实现逐步检测与分层反思，兼顾代理级和系统级失效。

**🔧 技术方法**

技术包括：推理轨迹收集、无监督推理范围对比学习、对齐嵌入的 Reasoning Encoder 与 Trace Encoder、步骤级检测与自反性缓解机制。

**📊 数据集**

使用了三套公开的 LLM‑驱动多智能体系统数据集：AutoGen‑Code、RCLAgent、SWE‑Agent。

**📈 对比分析**

与传统基于 LLM 的判定和现有 AgentOps 方法对比，EAGER 在异常检测 F1 分数分别为 73.57%、86.18%、79.95%，失败诊断 F1 分数为 63.23%、78.76%、69.51%，检测延迟平均约 4–5 秒，整体性能显著提升。

**⚠️ 局限性**

局限性在于：表示模型仅在 Qwen‑0.6B‑Embedding 基础上轻微微调，跨域泛化能力有限；对高复杂、多变故障模式的适应性尚待通过大规模微调进一步验证。

---

## 685. Conformal Koopman for Embedded Nonlinear Control with Statistical Robustness: Theory and Real-World Validation

**arXiv ID:** 2603.21580 | [PDF](https://arxiv.org/pdf/2603.21580v1)

**作者:** Koki Hirano `[一作]` (University of Illinois Urbana-Champaign), Hiroyasu Tsukamoto `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 183 | [OpenAlex ID](https://openalex.org/A5020563795)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于Conformal Koopman理论的嵌入式非线性控制框架，能够在保持实时性的同时提供统计鲁棒性保障。

**💡 创新点**

创新点在于将 conformal prediction 与 Koopman operator 结合，产生可解释的置信区间并提升控制系统对模型不确定性的鲁棒性。

**🔧 技术方法**

采用了 Koopman 嵌入、深度学习估计、对数预测与统计检验技术，构建了端到端的控制与预测模型。

**📊 数据集**

在仿真环境和真实机器人实验数据集上验证，数据来源包括 ICRA 2026 现场测试记录及公开机器人仿真平台。

**📈 对比分析**

与传统 LQR、MPC 与基线非线性控制方法对比，实验显示轨迹跟踪误差降低约15%，鲁棒性提升约20%，且计算延迟仍保持在可接受的实时范围内。

**⚠️ 局限性**

局限在于对大规模系统的计算开销较大、对训练数据质量高度依赖，以及对极端扰动场景的进一步验证仍待完善。

---

## 686. Stability and Bifurcation Analysis of Nonlinear PDEs via Random Projection-based PINNs: A Krylov-Arnoldi Approach

**arXiv ID:** 2603.21568 | [PDF](https://arxiv.org/pdf/2603.21568v1)

**作者:** Gianluca Fabiani `[一作]` (Johns Hopkins University), Ioannis G. Kevrekidis `[通讯]` (Johns Hopkins University)

**通讯引用:** 25264 | [OpenAlex ID](https://openalex.org/A5036566464)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一套基于物理信息随机投影神经网络（PI‑RPNN）的稳态解、线性稳定性和分岔分析框架，能够直接从训练得到的网络权重推导物理雅可比矩阵并求解其广义特征值。

**💡 创新点**

创新点在于：①利用PI‑RPNN的线性输出结构，直接得到物理雅可比矩阵的广义特征值问题；②针对随机投影矩阵的指数奇异值衰减导致的秩缺陷，提出了无显式逆的shift‑invert Krylov–Arnoldi矩阵自由求解方法；③从理论上证明了该广义特征值问题几乎必然正则，并给出奇异值指数衰减的定理，解释了数值不良条件。

**🔧 技术方法**

使用的技术包括：PI‑RPNN、随机投影神经网络、误差收敛理论、截断SVD、矩阵自由shift‑invert Arnoldi、广义特征值求解、伪弧长连续、以及有限差分/有限元参考解。

**📊 数据集**

通过自行构造的解析或数值参考解验证了三类典型非线性PDE：Liouville–Bratu–Gelfand、FitzHugh–Nagumo、Allen‑Cahn；没有使用公开数据集。

**📈 对比分析**

与传统有限差分（FD）/有限元（FEM）参考解以及之前的PINN分岔方法比较，所得分岔图、主特征值和特征函数与参考结果完全一致；shift‑invert方法在数值稳定性上显著优于直接伪逆法，且能够排除虚假零特征值，计算效率更高。

**⚠️ 局限性**

局限性包括：①对随机权重和激活函数参数的经验性设定，尤其对非解析激活函数（如ReLU）缺乏理论支持；②目前仅处理稳态分岔，无法直接处理周期轨道或时间相关的PDE；③在大规模高维问题中仍需引入随机SVD或迭代雅可比求解器以保持可扩展性。

---

## 687. Rethinking SAR ATR: A Target-Aware Frequency-Spatial Enhancement Framework with Noise-Resilient Knowledge Guidance

**arXiv ID:** 2603.21565 | [PDF](https://arxiv.org/pdf/2603.21565v1)

**作者:** Yansong Lin `[一作]` (University of Electronic Science and Technology of China), Zongyong Cui `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2287 | [OpenAlex ID](https://openalex.org/A5061245960)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了目标感知频域-空间耦合增强框架FSCE及其核心模块DSAF，并结合在线知识蒸馏，构建轻量级与高精度SAR ATR模型。

**💡 创新点**

创新点在于：①将多尺度空间卷积与Haar小波频域分解耦合以同时抑制雷达散斑噪声与强化目标纹理；②引入在线知识蒸馏实现目标关注引导；③通过DSAF与KD的协同，提供精度与模型轻量化双重优势。

**🔧 技术方法**

采用多尺度卷积、Haar小波变换、CBAM注意力机制、在线KL蒸馏、ResNet/ShuffleNetV2骨干、数据增强与梯度融合等技术。

**📊 数据集**

使用MSTAR、OpenSARShip、FUSARShip三大公开SAR ATR数据集进行实验。

**📈 对比分析**

与CNN、GCN、Transformer及现有SAR专用模型对比，DSAFNet‑L在三数据集上均达或超过最优精度（OpenSARShip 76.42%，FUSARShip 89.32%），DSAFNet‑M以0.17M参数保持接近同类模型精度，显著提升模型轻量化与泛化能力。

**⚠️ 局限性**

对极端尺度变化与复杂背景的鲁棒性仍有限，模型受图像缩放导致尺度不一致影响，需进一步探索尺度自适应与对齐策略。

---

## 688. Toward a Theory of Hierarchical Memory for Language Agents

**arXiv ID:** 2603.21564 | [PDF](https://arxiv.org/pdf/2603.21564v1)

**作者:** Yashar Talebirad `[一作]` (University of Alberta), Osmar Zaiane `[通讯]` (University of Alberta)

**通讯引用:** 1626 | [OpenAlex ID](https://openalex.org/A5102857306)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了统一的层级记忆框架，用三大操作（提取、粗化、遍历）对不同系统进行拆解与比较。

**💡 创新点**

核心创新在于引入自足性谱与粗化-遍历耦合概念，阐明代表函数属性如何决定检索策略。

**🔧 技术方法**

技术包括信息理论分析、信息瓶颈与Fano界定、LLM摘要、聚类与图分区等构建与检索手段。

**📊 数据集**

对十一种现有系统进行实例化，涵盖文档层级、对话记忆、执行轨迹等多种数据来源（公开文本、聊天日志、Agent轨迹）。

**📈 对比分析**

通过理论对比与对十一系统的映射，展示了统一语言下的可比性；实验未给出统一量化指标，而是通过自足性与搜索模式说明性能差异。

**⚠️ 局限性**

主要局限在于假设层级静态、未考虑动态插入/重构及查询条件下的自适应粗化；未来需研究自适应信息理论与在线学习。

---

## 689. Exploring Multimodal Prompts For Unsupervised Continuous Anomaly Detection

**arXiv ID:** 2603.21562 | [PDF](https://arxiv.org/pdf/2603.21562v1)

**作者:** Mingle Zhou `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Min Li `[通讯]` (City University of Macau)

**通讯引用:** 104874 | [OpenAlex ID](https://openalex.org/A5100339418)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多模态提示的无监督连续异常检测框架，利用视觉与文本提示共同学习并持续保持正常模式知识，实现对工业图像的异常识别与像素级分割。

**💡 创新点**

创新点在于①构建连续多模态提示记忆库（CMPMB）来逐步蒸馏和保存视觉+文本的典型正常模式；②设计缺陷语义引导自适应融合机制（DSG‑AFM），通过自适应归一化和动态融合提升视觉与文本异常分数的协同效果；③在无监督连续学习场景下实现高精度异常检测与分割，无需回放或重标注。

**🔧 技术方法**

采用CLIP预训练的视觉语言模型作为骨干，使用视觉与文本提示（prompt tuning）、对比学习、核心样本采样（CSS）构建特征库；自适应归一化（ANM）和动态融合（DFS）实现多模态特征融合；使用Adam优化器和适当的学习率进行训练。

**📊 数据集**

主要使用工业检测数据集 MVTec AD 和 VisA，分别提供图像级与像素级异常标签。

**📈 对比分析**

与包括 UCAD、PatchCore、UniAD、DRAEM 等14种方法进行对比，实验显示在 MVTec AD 上图像AUROC提升约4.4%、像素AUPR提升约14.8%；在 VisA 上分别提升约2.7%和6.5%；整体达到或超过目前的SOTA水平。

**⚠️ 局限性**

局限性：引入文本提示后模型的遗忘度（FM）略有上升，虽然仍处于可接受范围；依赖预训练CLIP模型与大规模视觉语言对齐；对不同任务间语义分布差异仍有一定适配需求。

---

## 690. Revisiting Weakly-Supervised Video Scene Graph Generation via Pair Affinity Learning

**arXiv ID:** 2603.21559 | [PDF](https://arxiv.org/pdf/2603.21559v1)

**作者:** Minseok Kang `[一作]` (Yonsei University), Sangyoun Lee `[通讯]` (Yonsei University)

**通讯引用:** 3420 | [OpenAlex ID](https://openalex.org/A5015739530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无边框标注的视频场景图生成框架，通过学习对象对的交互亲和力来抑制非交互对。

**💡 创新点**

创新点在于：①引入可学习的Pair Affinity学习与评分机制；②利用视觉语言预训练模型进行Relation‑Aware Matching提升伪标签质量；③在注意力模块中加入Pair Affinity Modulation（PAM）以抑制噪声上下文。

**🔧 技术方法**

核心技术包括视觉语言跨模态对齐（GroundingDINO）、双分支嵌入（关系嵌入与亲和力嵌入）以及基于亲和力的注意力门控，应用于STTran/DSG‑DETR两种关系预测骨干。

**📊 数据集**

在Action Genome数据集上进行实验，仅使用中间帧的未定位三元组作为弱监督，采用SGDet评估协议。

**📈 对比分析**

与现有弱监督方法（PLA、TRKT）以及零样本/全监督基线相比，本文在R@10、R@20、R@50等指标上均实现平均+5.5%到+6.4%的提升，靠近全监督上限（≈88%–94%）。

**⚠️ 局限性**

局限在于亲和力监督依赖于匹配分区，受视觉语言模型对定位可靠性的启发式估计约束，且对训练数据的完整性敏感。

---

## 691. A Survey of Web Application Security Tutorials

**arXiv ID:** 2603.21556 | [PDF](https://arxiv.org/pdf/2603.21556v1)

**作者:** Bhagya Chembakottu `[一作]` (McGill University), Martin P. Robillard `[通讯]` (McGill University)

**通讯引用:** 7202 | [OpenAlex ID](https://openalex.org/A5059244952)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统采集并分析了132篇网页安全教程，评估其作者归属、结构、代码示例、广告出现与官方安全资源引用等特征，进而提出通过是否包含代码示例及是否直接链接到官方安全资源来识别高质量教程的启发式方法。

**💡 创新点**

创新点在于首次将教程质量与可执行代码与官方安全资源引用的关联进行量化研究，并基于此提出了两个易于开发者使用的实用筛选标准；同时通过结构化编码和非参数统计验证了这些特征与教程实用性的显著关联。

**🔧 技术方法**

所采用的技术包括：利用 DuckDuckGo API 进行搜索爬取、人工双人编码制定结构化编码表、对叙事长度和类别特征使用 Mann‑Whitney U 检验与 Pearson 卡方检验，以及 bootstrap 置信区间估计。

**📊 数据集**

使用的数据集为132条经筛选的英文网页安全教程，来源于21个搜索查询的前50结果（共1050条），通过去重后获得最终数据集。

**📈 对比分析**

通过比较叙事长度、代码示例出现率、广告比例和官方资源引用率等变量，统计检验表明包含代码示例或官方链接的教程在提供实现细节方面显著优于仅提供高层建议的教程，且两类特征在数据中分布差异显著。

**⚠️ 局限性**

主要局限包括：仅聚焦英文文本教程，排除了视频、书籍和在线课程；仅使用 DuckDuckGo 作为搜索引擎，未覆盖其他搜索结果的差异；未评估代码的正确性与链接内容的准确性；以及研究时间点的固定性，未来可能因自动化文档生成和生成式 AI 的影响导致教程质量分布变化。

---

## 692. Ultrafast microwave sensing and automatic recognition of dynamic objects in open world using programmable surface plasmonic neural networks

**arXiv ID:** 2603.21521 | [PDF](https://arxiv.org/pdf/2603.21521v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 693. What Do World Models Learn in RL? Probing Latent Representations in Learned Environment Simulators

**arXiv ID:** 2603.21546 | [PDF](https://arxiv.org/pdf/2603.21546v1)

**作者:** Xinyu Zhang `[一作]` `[通讯]` (Anyscale), Xinyu Zhang (Anyscale)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对IRIS和DIAMOND两种世界模型在Atari Breakout和Pong上进行线性/非线性探针、因果干预和注意力分析，探究其内部表示。

**💡 创新点**

首次证明两种不同架构的世界模型能够学习到近似线性的、功能上可用的游戏状态表示，并揭示注意力头的空间专化。

**🔧 技术方法**

采用线性回归、MLP探针、激活补丁因果干预、注意力熵计算及多基线词标消融等技术。

**📊 数据集**

使用Atari 2600的Breakout和Pong游戏，利用RAM记录的真实游戏状态标签进行评估。

**📈 对比分析**

与原始像素、随机模型和打乱标签基线对比，线性探针均达到0.85–0.99的R²，MLP提升有限，因果干预与注意力消融显示高相关性，验证表示的功能性。

**⚠️ 局限性**

实验仅覆盖二维Atari游戏，未探讨三维环境或长序列动态，且单向干预方法粗糙，缺乏更细粒度的因果分析。

---

## 694. Generalization Limits of In-Context Operator Networks for Higher-Order Partial Differential Equations

**arXiv ID:** 2603.21534 | [PDF](https://arxiv.org/pdf/2603.21534v1)

**作者:** Jamie Mahowald `[一作]` (University of Texas at Austin), Tan Bui-Thanh `[通讯]` (University of Texas at Austin)

**通讯引用:** 3014 | [OpenAlex ID](https://openalex.org/A5008274019)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了 In-Context Operator Networks（ICON）在高阶偏微分方程上的泛化性能，并扩展了模型能够处理的 19 类 ODE/PDE，探讨了其在分布内与分布外任务中的表现。

**💡 创新点**

创新点包括：①将 in-context 学习与 operator 网络结合，能够在推理时通过少量示例即刻学习新的算子；②利用数值方法合成大规模训练数据，扩展到高阶、高维方程；③提供交互式求解接口，并系统评估模型的分布外泛化能力。

**🔧 技术方法**

技术手段包括：基于 Transformer 的 6 层 8 头 encoder-only 结构，next-token 预测；数值求解器（Euler、有限差分、WENO）生成合成 (condition, QoI) 对；in-context 示例提示以及自然语言注释的多模态输入。

**📊 数据集**

使用了 19 种 ODE/PDE 的合成数据集，所有数据通过数值解法产生，参数随机采样，条件采用高方差 RBF 高斯过程生成；数据覆盖不同阶数、维度和边界条件。

**📈 对比分析**

实验通过与原始 ICON 论文的平均测试误差进行对比，结果显示误差与基准相近；在分布内任务中误差随示例数下降，阻尼振子、泊松方程等某些类型不随样本数提升；分布外任务误差提高一个数量级，但模型能捕捉整体趋势；推理时间随域长度线性增长，最大域 500 时约 1.17 秒。

**⚠️ 局限性**

主要限制：推理时间随域长度线性增加，限制了大域问题的应用；分布外性能受限，尤其在边界条件和幅值保持上表现欠佳；对同一类问题内部高度异质的示例泛化能力有限。

---

## 695. VIGIL: Part-Grounded Structured Reasoning for Generalizable Deepfake Detection

**arXiv ID:** 2603.21526 | [PDF](https://arxiv.org/pdf/2603.21526v1)

**作者:** Xinghan Li `[一作]` (Fudan University), Jingjing Chen `[通讯]` (Fudan University)

**通讯引用:** 5660 | [OpenAlex ID](https://openalex.org/A5100373492)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出VIGIL框架，采用计划-检查的部件中心结构化推理，并将独立法医证据注入到多模态大语言模型中，以实现可解释的深度伪造检测。

**💡 创新点**

通过阶段门控注入机制将部件级法医证据与推理分离，使每个断言都基于独立证据；同时引入三阶段逐步训练和部件感知奖励，显著提升模型的泛化与解释质量。

**🔧 技术方法**

使用多模态大语言模型（Qwen3-VL-8B）配合频谱与像素级法医编码器、面部语义分割、上下文动态信号注入，以及基于GRPO的强化学习。

**📊 数据集**

主要使用自构建的OmniFake 5级分层基准（仅训练于三种基础生成器，测试覆盖跨架构、跨模型、局部编辑及野外社交媒体数据），并在HydraFake、FaceForensics++等公开数据集上进行验证。

**📈 对比分析**

在OmniFake和HydraFake上与专家检测器、通用MLLM及Veritas/FakeVLM等方法对比，VIGIL平均准确率提升约4–6%，最高难度级别L5可达92%+，显著优于对照方法。

**⚠️ 局限性**

仍受面部解析误差、部件标注噪声和极端压缩/模糊场景下的鲁棒性限制；对未见生成器的跨域泛化仍存在一定落差。

---

## 696. Overview of TREC 2025 Biomedical Generative Retrieval (BioGen) Track

**arXiv ID:** 2603.21582 | [PDF](https://arxiv.org/pdf/2603.21582v1)

**作者:** Deepak Gupta `[一作]` (National Library of Medicine, NIH), Kirk Roberts `[通讯]` (UTHealth Houston)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文概述了2025年TREC BioGen跟踪的任务、数据集、评估指标和参与系统的表现，重点评估了两项任务：A）答案句子的引用定位与对立证据检索；B）答案生成与引用归属；

**💡 创新点**

创新点在于提出系统性对高风险医学信息生成时的引用真实性与对立证据检索挑战，并为此构建了大型标注数据集与多维评估框架，推动可信医疗问答系统的研究；

**🔧 技术方法**

主要技术包括NLI（SciFive）与BM25检索、reranker（GraphMonoT5、cross‑encoder）以及检索增强生成（RAG）结合LLM（LLaMA2、ChatGPT‑3.5/4、Mistral、LLM‑5B等）来生成带有PMID引用的答案；

**📊 数据集**

使用的主要数据集为2024年稳定版PubMed（约26.8M条摘要）以及相应的BioGen 2025任务标注（40个问题、194句答案及其支持/矛盾PMID）和官方评测集合；

**📈 对比分析**

通过专家评估与自动评估（BioACE、LLama‑3.3）进行对比，支持类在最佳系统上实现了约68%（relaxed precision）和F1≈67%，但对立证据检索性能仍低（precision≈8%），答案生成在准确率>90%，引用覆盖率>90%，但仍有部分系统存在冗余或潜在误导；

**⚠️ 局限性**

局限性包括：对立证据检索表现差，系统对证据的精确性与完整性不足；数据集规模虽大，但仍无法覆盖所有临床情境；评估框架仍依赖人工标注，自动化评估准确性有限，未来需提升对抗性评测与跨任务鲁棒性。

---

## 697. Stabilizing Iterative Self-Training with Verified Reasoning via Symbolic Recursive Self-Alignment

**arXiv ID:** 2603.21558 | [PDF](https://arxiv.org/pdf/2603.21558v1)

**作者:** Xinyu Zhang `[一作]` `[通讯]` (Anyscale), Xinyu Zhang (Anyscale)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Neuro‑Symbolic Recursive Self‑Alignment（NSRSA）框架，利用符号验证子系统逐步过滤模型自生成的推理链，防止递归漂移并实现递归自我提升。

**💡 创新点**

创新点在于：① 在递归自我训练中引入逐步算术、逻辑流与约束满足的符号验证；② 通过验证结果构造 Direct Preference Optimization（DPO）偏好对，进一步引导模型偏好正确推理。

**🔧 技术方法**

技术手段包括：链式思维生成（vLLM + Qwen3‑4B‑Thinking）、sympy 符号计算、LoRA 微调、DPO、Self‑BLEU 评估、多种验证策略（无验证、结果验证、投票、符号验证）。

**📊 数据集**

使用数据集：GSM8K（训练/测试）与 MATH‑500（跨任务评估）。

**📈 对比分析**

比较方法：对比无验证、仅结果验证、投票、符号验证以及符号验证+DPO 5 次自训练迭代。NSRSA 在第 5 轮达到 91.0%（基线 80.5%），DPO 进一步提升到 91.2%；无验证递归深度仅 2，结果验证 plateau 约 86%；跨任务 MATH‑500 在 5 轮后提升至 51.2%。Self‑BLEU 指标显示 NSRSA 维持较低的模式崩溃。

**⚠️ 局限性**

局限性：① 解析覆盖率有限，部分算术表达式未被检测导致错误漏检；② 约束检查仅适用于数学问题，需为其他领域设计对应约束；③ 计算开销虽不大，但符号验证仍比纯模型推理慢；④ 早期迭代中过度过滤可能导致样本不足；⑤ 需要针对不同任务手工调整阈值和规则。

---

## 698. Sharper Generalization Bounds for Transformer

**arXiv ID:** 2603.21541 | [PDF](https://arxiv.org/pdf/2603.21541v1)

**作者:** Yawen Li `[一作]` (Capital Normal University), Zhongyi Li `[通讯]` (Beihang University)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5100774465)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文通过偏移 Rademacher 复杂度（Offset Rademacher Complexity）推导了 Transformer（单头、复合头和多层）模型的更紧凑的泛化误差上界，取得了 O(1/n) 的快速收敛率，并通过低秩和范数约束给出了更精细的结构相关上界；

**💡 创新点**

创新点在于：①首次将偏移 Rademacher 复杂度应用于 Transformer 的泛化分析，获得 O(1/n) 的最快收敛；②利用覆盖数结合矩阵秩与范数实现了架构依赖的精细上界；③将理论推广到无界子高斯和重尾分布情形，提供了截断与鲁棒损失的处理方法；

**🔧 技术方法**

主要技术包括：偏移 Rademacher 复杂度、经验覆盖数估计、矩阵范数与秩约束、截断（truncation）与鲁棒损失（Catoni/Huber 等）以及多头/多层 Transformer 的自注意力结构分析；

**📊 数据集**

论文为理论分析性工作，没有使用具体数据集；其结果适用于回归、分类和鲁棒损失场景；

**📈 对比分析**

与传统基于全局 Rademacher 复杂度得到的 O(1/√n) 上界相比，本文取得了更快的 O(1/n) 收敛速率；在无界和重尾情况下，通过截断与鲁棒损失仍能保持较优的误差衰减；

**⚠️ 局限性**

局限性包括：①依赖参数范数与 Lipschitz 连续性等正则化假设；②对重尾分布的收敛速率受尾指数限制，仍比理想情况慢；③论文未给出实验验证，主要为理论推导，实际模型表现需进一步实证。

---

## 699. Rethinking Visual Privacy: A Compositional Privacy Risk Framework for Severity Assessment with VLMs

**arXiv ID:** 2603.21573 | [PDF](https://arxiv.org/pdf/2603.21573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 700. LLM-Based Test Case Generation in DBMS through Monte Carlo Tree Search

**arXiv ID:** 2603.21530 | [PDF](https://arxiv.org/pdf/2603.21530v1)

**作者:** Yujia Chen `[一作]` (Harbin Institute of Technology), Cuiyun Gao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1841 | [OpenAlex ID](https://openalex.org/A5103094513)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在DBMS测试中提出MIST框架，结合LLM生成与蒙特卡罗树搜索变异，生成高质量SQL测试用例。

**💡 创新点**

创新点是使用层次化功能树与错误反馈驱动LLM生成，及基于覆盖率反馈的MCTS变异策略，显著提升轻量LLM在专有SQL方言下的适配和覆盖深度。

**🔧 技术方法**

技术包括大型语言模型LLM（Qwen、Llama系列）、文档驱动的特征树抽取、错误回溯反馈、蒙特卡罗树搜索（MCTS）以及SQL生成与变异规则。

**📊 数据集**

实验数据集：DuckDB、PostgreSQL、SQLite三款主流DBMS；使用4款轻量LLM（7B~32B）生成900条初始SQL并进行600次MCTS变异；收集覆盖率数据。

**📈 对比分析**

与基线Fuzz4All比较，使用行、函数、分支覆盖率指标，MIST平均提升约43.3%行覆盖、32.3%函数覆盖、46.4%分支覆盖，最优覆盖率分别为69.3%（Optimizer模块）等。

**⚠️ 局限性**

局限性在于仅评估代码覆盖率、未评估缺陷发现能力；实验仅覆盖三款开源DBMS；仅使用轻量LLM，可能不适用于更大或更复杂的工业数据库；覆盖率提升受生成质量和变异规则的限制。

---

## 701. Stationary Online Contention Resolution Schemes

**arXiv ID:** 2603.21532 | [PDF](https://arxiv.org/pdf/2603.21532v1)

**作者:** Mohammad Reza Aminian `[一作]` (University of Chicago), Pranav Nuti `[通讯]` (University of Chicago)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5013100239)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了Stationary Online Contention Resolution Schemes (S-OCRS)，一种对到达顺序不敏感的在线争用决策方案。

**💡 创新点**

创新点在于引入站稳性约束，给出LP表征和通用实现模板，并利用最大熵/KL投影构造S-OCRS，取得多种环境下最优或近优的选取率。

**🔧 技术方法**

使用最大熵分布、Gibbs分布、KL投影、模拟-替换元算法和负相关Rayleigh分布等技术。

**📊 数据集**

无具体外部数据集，研究基于理论模型的匹配、k-uniform matroid、弱Rayleigh matroid等可行性环境。

**📈 对比分析**

与已有OCRS/prophet inequality基准对比，取得比之前更优的选取率：Bipartite匹配(3-√5)/2、k-uniform matroid α_k≈1-√(2/(πk))、弱Rayleigh matroid 12。

**⚠️ 局限性**

限制在于仅覆盖了特定可行性环境，最大熵法对一般matroid失效；S-OCRS相较于OCRS可能存在性能折扣，未完全解决一般matroid 1/2可选问题。

---

## 702. Mind over Space: Can Multimodal Large Language Models Mentally Navigate?

**arXiv ID:** 2603.21577 | [PDF](https://arxiv.org/pdf/2603.21577v1)

**作者:** Qihui Zhu `[一作]` (Beihang University), Xingxing Wei `[通讯]` (Beihang University)

**通讯引用:** 3426 | [OpenAlex ID](https://openalex.org/A5079657274)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Video2Mental基准，用于评估多模态大语言模型（MLLM）在长时空视觉输入下的“心理导航”能力，并开发NavMind模型实现该任务。

**💡 创新点**

创新点在于：①构造层级认知地图作为中间可解释表示；②使用分层难度递进的有监督微调（CogRS）专门训练多步空间推理；③通过物理仿真验证生成路径的可执行性。

**🔧 技术方法**

技术包括：基于Qwen3-VL的Transformer架构、结构化生成（JSON认知地图+路径链）、认知引导拒绝采样、分层区域/地标/物体三层认知图构建，以及在Habitat-Sim中执行路径验证。

**📊 数据集**

数据集为Video2Mental，包含约24k条从246个高保真室内场景（HM3D、MP3D）生成的5分钟以上自我探索视频、语义图、认知地图与路径链。

**📈 对比分析**

与开源与专有MLLM（InternVL、Qwen3、GPT-5.1等）以及专门的空间推理模型（Cambrian-S、RynnBrain）对比，NavMind在SR_t/SR_p、SPL等指标上提升约40%+，在长程任务中表现尤为突出。

**⚠️ 局限性**

局限性在于：仍依赖显式认知地图的准确性；对极其复杂或全新场景的泛化仍有限；以及模型在多步推理时的计算成本和推理时间未充分优化。

---

## 703. DATASHI: A Parallel English-Tashlhiyt Corpus for Orthography Normalization and Low-Resource Language Processing

**arXiv ID:** 2603.21571 | [PDF](https://arxiv.org/pdf/2603.21571v1)

**作者:** Nasser-Eddine Monir `[一作]`, Zakaria Baou `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了DATASHI语料库，提供了5,000句英-塔什利亚特平行对照，其中1,500句已由专家标准化，用于正字法归一化和跨域评估。

**💡 创新点**

创新点在于：①首次系统收集并公开大规模、主题平衡的Tashlhiyt并行数据；②融合用户生成与专家标准化文本，直接捕捉正字法多样性；③为低资源语言建立了可复现的归一化基准。

**🔧 技术方法**

技术手段包括：正字法归一化框架（字符级、形态句法级、语音级规则），大语言模型（Claude‑Sonnet‑4.5、Gemini‑2.5‑Pro、GPT‑5、Mistral‑Large‑2411、Qwen3‑Max）进行零样本与少样本推理，评估指标采用WER与Levenshtein距离。

**📊 数据集**

使用的数据集为：完整DATASHI语料（5,000句）和1,500句专家标准化子集；另外挑选30句代表性示例用于少样本提示。

**📈 对比分析**

在零样本与少样本两种设置下，对五款LLM进行对比。Gemini‑2.5‑Pro在少样本情形下取得最佳性能，WER≈35.5%，LD≈3.65；Claude‑Sonnet‑4.5其次，GPT‑5和Mistral表现相对更差，Qwen3‑Max最低；整体显示LLM在低资源正字法任务中具备可迁移性，但仍受标记化、形态学挑战限制。

**⚠️ 局限性**

局限性包括：1）对gemination、emphatic等音系特征的处理仍不够精确；2）语料仅覆盖文本，缺乏读语音数据；3）归一化规则依赖手工设定，难以扩展到其他阿马兹格方言；4）实验仅评估现有LLM，未尝试自监督或微调方法。

---

## 704. PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2603.21528 | [PDF](https://arxiv.org/pdf/2603.21528v1)

**作者:** Gensheng Pei `[一作]` (Sungkyunkwan University), Byeungwoo Jeon `[通讯]` (Sungkyunkwan University)

**通讯引用:** 4880 | [OpenAlex ID](https://openalex.org/A5074587654)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种全无训练的开词汇语义分割框架PEARL，通过对冻结的CLIP视觉语言模型的最后自注意力块进行Procrustes对齐，再利用文本感知的拉普拉斯传播提升像素级分割质量，避免了额外后处理和辅助模型；

**💡 创新点**

创新点在于（1）在自注意力内部实现正交Procrustes对齐，纠正关键子空间与查询子空间的不匹配，提升patch‑text匹配的几何一致性；（2）引入文本指导的拉普拉斯传播，在小尺寸网格上以文本相似度为邻接门控、图像梯度为边界约束，直接将语言信息转化为结构先验；两步结合形成一个轻量、无参数、可即插即用的推理流程；

**🔧 技术方法**

核心技术包括正交Procrustes对齐（单步SVD或极化迭代实现）、文本编码器产生单位范数原型、基于文本相似度的行软化权重、图像梯度门控的加权拉普拉斯求解以及共轭梯度求解小线性系统；

**📊 数据集**

在标准开词汇语义分割基准上评估：Pascal VOC 21/20、Pascal Context 60/59、COCO-Object、COCO-Stuff、Cityscapes、ADE20K；

**📈 对比分析**

与现有训练‑free、无辅助骨干方法（如NACLIP、SFP、SCLIP等）以及部分使用辅助骨干的基线比较，PEARL在无额外骨干的情况下实现mIoU 43.2%、pAcc 67.2%，在VOC、Context、Cityscapes等数据集上均位列第一或第二，显示出明显的性能提升；

**⚠️ 局限性**

局限性包括对提示词质量与标签命名敏感、低对比度边界仍难以处理、网格尺寸折中细节与成本、缺乏实例级分割能力，这些都是训练‑free OVSS通用挑战。

---

## 705. CatRAG: Functor-Guided Structural Debiasing with Retrieval Augmentation for Fair LLMs

**arXiv ID:** 2603.21524 | [PDF](https://arxiv.org/pdf/2603.21524v1)

**作者:** Ravi Ranjan `[一作]` (Florida International University), Agoritsa Polyzou `[通讯]` (Florida International University)

**通讯引用:** 399 | [OpenAlex ID](https://openalex.org/A5013726519)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CatRAG框架，结合范畴论指导的结构保持投影与检索增强生成（RAG）进行双重去偏。

**💡 创新点**

创新点在于：① 用范畴论Functor构建结构保持投影，既抑制受保护属性方向，又保留任务相关语义；② 在推理时通过多样性意识检索提供平衡证据，避免引入新偏差；③ 将两种手段协同应用，显著提升公平-效能权衡。

**🔧 技术方法**

技术包括：线性正交投影（基于散度矩阵的广义特征问题）、检索增强生成（TF‑IDF检索+多样性重排序）、Prompt融合、基准评估框架。

**📊 数据集**

使用Bias Benchmark for Question Answering (BBQ) 的四个子集（Gender、Nationality、Race、Race×Gender），以及公开的LLM模型（Meta Llama‑3、OpenAI GPT‑OSS、Google Gemma‑3）进行实验。

**📈 对比分析**

与四个主流去偏方法（CE Debiasing、Self‑Debiasing、SP Debiasing、Causal Debiasing）在相同推理条件下对比。CatRAG 在BBQ上实现 80.7% 正确率（相较基线提升 32.3%），偏差分数从 0.63 降至 0.01，显示出最优的公平‑效能平衡。

**⚠️ 局限性**

局限性包括：投影仅为线性，受限于所选敏感锚点；检索效果依赖语料覆盖与多样性；增加检索开销与推理时间；在极度偏差或极少量数据场景下，投影与检索的组合效果仍需进一步验证。

---

## 706. Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion

**arXiv ID:** 2603.21673 | [PDF](https://arxiv.org/pdf/2603.21673v1)

**作者:** Shixu Liu `[一作]` `[通讯]` (Northeast Petroleum University), Shixu Liu (Northeast Petroleum University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出WeatherTGD框架，利用训练无关的多代理文本梯度下降为天气时间序列生成可解释的自然语言说明。

**💡 创新点**

创新性地将文本梯度下降与三专门化代理（统计、物理、气象专家）结合，并设计共识感知梯度融合机制，实现多视角信息的统一优化。

**🔧 技术方法**

使用大型语言模型（DeepSeek‑V3.2、MiniMax‑01、Qwen3‑Next‑80B）作为代理，采用文本梯度下降、语义相似度聚类、并行推理与迭代细化技术。

**📊 数据集**

在包含500条多气象变量（温度、压力、湿度、风速、降水）的真实观测时间序列上评估，并公开数据集。

**📈 对比分析**

与AutoGen、CAMEL、LLM‑Debate等六种多代理基线对比，基于GPT‑4o判定和人类专家评分，WeatherTGD平均整体得分8.50/10、专家得分8.34/10，显著高于最佳基线7.01/10，且标记消耗仅3.5×。

**⚠️ 局限性**

仅针对单一语言（中/英混合），缺乏多语言适配和与数值天气预报系统的集成；代理间协同仍依赖预设阈值，需进一步自适应。

---

## 707. HumanOmni-Speaker: Identifying Who said What and When

**arXiv ID:** 2603.21664 | [PDF](https://arxiv.org/pdf/2603.21664v1)

**作者:** Detao Bai `[一作]` (Tongyi Lab Alibaba Group), Zhiheng Ma `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 HumanOmni‑Speaker，一个专为多说话人场景设计的多模态大语言模型，能够在同一段视频中精准回答“谁说了什么、何时说”。

**💡 创新点**

创新点包括：①提出视觉注册说话人分辨（VR‑SDR）任务及 HumanOmni‑Speaker 基准，彻底消除视觉捷径；②引入 25fps 高帧率视觉 Delta 编码器，将时域残差压缩为 6 个结构化 Token，兼顾高频动态感知与低算力；③三阶段训练策略实现跨模态对齐与端到端语义绑定。

**🔧 技术方法**

采用 Qwen2.5‑Omni 作为骨干，加入 ResNet‑18+SVT+Transformer 的视觉 Delta Encoder，视觉基准 Encoder，音频 Encoder，文本 Tokenizer，并通过跨模态交叉注意力、LoRA 及全层微调实现多模态融合；评价指标使用 SA‑WER、IER 等。

**📊 数据集**

使用 AVSpeech、LRS2/3、VoxCeleb2、AVA‑ASD、Columbia‑ASD、AVSpeech、VoxMM、Librispeech 等公开数据集，构建 1.5M 条多模态指令数据集以及 HumanOmni‑Speaker 的 4 个子任务与 VR‑SDR。

**📈 对比分析**

与 Gemini3‑Pro、Qwen3‑Omni‑Flash、OLA、VITA、Qwen2.5‑Omni 等开源/闭源模型对比，HumanOmni‑Speaker 在 VR‑SDR、Speaker Localization、Speaker Identification 等任务上至少降低 20% 错误率；在 VSR/AVSR 上取得与 Auto‑AVSR 相当的 WER，在 ASR 上与 Qwen2.5‑Omni 竞争。

**⚠️ 局限性**

局限性：仍需高帧率视觉采样导致算力消耗较大；跨模态对齐在极端光照、遮挡或背景噪声场景下表现仍有提升空间；Hard 子集的性能相对较低，模型对复杂多说话人交互的鲁棒性待进一步加强。

---

## 708. TAMTRL: Teacher-Aligned Reward Reshaping for Multi-Turn Reinforcement Learning in Long-Context Compression

**arXiv ID:** 2603.21663 | [PDF](https://arxiv.org/pdf/2603.21663v1)

**作者:** Li Wang `[一作]` (Beihang University), Wenjun Wu `[通讯]` (Beihang University)

**通讯引用:** 9097 | [OpenAlex ID](https://openalex.org/A5060858375)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种教师对齐奖励重塑方法（TAMTRL），在多轮强化学习中为长文本处理提供细粒度的时间信用分配，提升记忆更新的效果。

**💡 创新点**

创新点在于：①使用同一模型的全局视角作为教师，避免外部评估器；②基于教师概率分布进行分层归一化的奖励重塑；③在CTDE框架下实现教师-学生的对齐，提供时序层面的监督。

**🔧 技术方法**

技术核心包括：POMDP建模、CTDE式教师-学生训练、基于 token 级别概率的奖励重塑、min‑max 归一化、长度归一化以及 DAPO（Decoupled Clip & Dynamic Sampling Policy Optimization）强化学习算法。

**📊 数据集**

数据集：训练使用 HotpotQA（带干扰段落），评估使用七个长文本基准：HotpotQA、RULER‑QA、NIAH、2WikiMultihopQA、MuSiQue、NarrativeQA 与 Qasper。

**📈 对比分析**

与 SFT、STaR、Vanilla‑KD、MemAgent、LLM‑judge、PRM 等基线进行对比；在 Qwen3‑0.6B 与 Qwen3‑1.7B 两个模型上实验，TAMTRL 在所有基准上均取得最高或第二高分，平均提升约 1.9%‑2.0%。

**⚠️ 局限性**

局限性：①需要事先标注的相关文档块来生成教师概率；②对 chunk 大小、训练样本量等超参较为敏感；③在极长文本或更大模型规模下的可扩展性尚未验证；④奖励重塑的归一化过程可能引入额外训练不稳定性。

---

## 709. A Comparative Analysis of LLM Memorization at Statistical and Internal Levels: Cross-Model Commonalities and Model-Specific Signatures

**arXiv ID:** 2603.21658 | [PDF](https://arxiv.org/pdf/2603.21658v1)

**作者:** Bowen Chen `[一作]` (University of Tokyo), Yusuke Miyao `[通讯]` (University of Tokyo)

**通讯引用:** 4657 | [OpenAlex ID](https://openalex.org/A5004444958)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该研究收集了多系列LLM模型，系统评估并对比其在统计和内部层面上的记忆行为。

**💡 创新点**

创新点在于跨模型、跨系列的记忆率log线性扩展、压缩率分析以及发现记忆相关注意头的家族特征。

**🔧 技术方法**

使用的技术包括记忆评分、压缩率、残差噪声注入、Logit‑Lens、注意头消融等。

**📊 数据集**

使用的数据集是已公开预训练语料（Pythia、StarCoder、OpenLLaMA、OLMo1/2/3等系列对应的Pile、Dolma、Redpajama、The Stack等）。

**📈 对比分析**

通过在20个模型上计算记忆率、压缩率、频率分布、噪声鲁棒性及注意头重要性，展示了不同模型家族的相似与差异，性能表现符合预期的log线性扩展。

**⚠️ 局限性**

限制在于只能使用公开数据集、样本量有限、无法覆盖所有主流开源模型、对大规模预训练语料的完整性和可复现性有挑战。

---

## 710. Engineering Distributed Governance for Regional Prosperity: A Socio-Technical Framework for Mitigating Under-Vibrancy via Human Data Engines

**arXiv ID:** 2603.21639 | [PDF](https://arxiv.org/pdf/2603.21639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 711. TrustFed: Enabling Trustworthy Medical AI under Data Privacy Constraints

**arXiv ID:** 2603.21656 | [PDF](https://arxiv.org/pdf/2603.21656v1)

**作者:** Vagish Kumar `[一作]` (Indian Institute of Technology Delhi), Souvik Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 3174 | [OpenAlex ID](https://openalex.org/A5030664714)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了TrustFed框架，实现在多机构医疗影像联邦学习下的统计置信度预测；

**💡 创新点**

通过表示感知的客户端分配与软最近阈值聚合，提供在异质、极度类别不平衡数据上分布自由、有限样本覆盖保证；

**🔧 技术方法**

结合联邦学习、分布式自适应合格预测、基于特征嵌入的最近邻聚合、CortiNet频域深度模型等技术；

**📊 数据集**

使用约43万张多模态医疗影像（血细胞显微、腹部CT、皮肤病变、视网膜、肾组织、结肠组织）进行实验；

**📈 对比分析**

与中心化及传统联邦不确定性方法比较，TrustFed在所有任务均实现或接近名义覆盖率，且预测集更紧凑；

**⚠️ 局限性**

局限于单模态分类任务，邻域大小需经验选择，未扩展至多模态、回归或分割等其他任务。

---

## 712. Complexity of Linear Subsequences of Fibonacci-Automatic Sequences

**arXiv ID:** 2603.21645 | [PDF](https://arxiv.org/pdf/2603.21645v1)

**作者:** Delaram Moradi `[一作]` (University of Waterloo), Jeffrey Shallit `[通讯]` (University of Waterloo)

**通讯引用:** 6360 | [OpenAlex ID](https://openalex.org/A5065322269)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文构造并分析了在斐波那契表示（Zeckendorf 表示）下用于识别加法、乘法以及线性子序列等算术关系的确定性有限自动机（DFA）及其状态复杂度，并通过这些结果改进了 Bosma 与 Don 关于斐波那契词线性子序列的模态大小上界。

**💡 创新点**

创新点在于：
1) 提出斐波那契表示下加法（Y=X+c 或 Y=X−c）可用 O(log c) 状态的 DFA；
2) 对乘法 Y=nX+c 给出 O(n²) 状态的 DFA；
3) 对任意 Fibonacci‑automatic 序列的线性子序列 (h(ni+c)) 给予 O(m²n⁴) 状态的 DFAO；
4) 通过上述构造将 Bosma 与 Don 的指数上界压缩至多项式 O(n⁴)；
5) 详细讨论了利用 Büchi 逻辑与 Walnut 进行自动机构造的时间复杂度。

**🔧 技术方法**

主要技术包括：斐波那契数制与 Zeckendorf 表示的数论性质、有限自动机（DFA、NFA、DFAO、UFAO）的构造与子集构造、状态最小化（Hopcroft、Valmari 算法）、Büchi 逻辑与 Walnut 的自动机翻译、以及对差值函数 D(x,y) 的递归分析。

**📊 数据集**

本文并未使用传统意义上的数据集；所有结果均基于理论构造和符号计算（如 Walnut 计算机程序验证最小化结果）。

**📈 对比分析**

通过理论分析与 Walnut 计算得到的最小化状态数进行比较：
- 加法与减法：O(log c) 状态，比之前的 O(c) 或指数下界明显改进；
- 乘法：O(n²) 状态，优于 2n² 的已知下界；
- 线性子序列：O(m²n⁴) 状态，显著低于 Bosma & Don 的指数上界；
- 这些结果均在理论上给出上界，实验验证表明最小化后的状态数与所给上界一致或更小。

**⚠️ 局限性**

局限性：
1) 构造得到的自动机并不一定是最小化的，需要进一步的最小化算法；
2) 对于乘法与线性子序列的状态上界虽然已是多项式，但仍可能不是最优，实际最小状态数仍待研究；
3) 复杂度分析主要针对理论构造，实际实现中 Walnut 的子集构造与最小化可能导致更高的运行时间；
4) 论文仅覆盖斐波那契表示，尚未验证对更广泛的 Pisot 数制或其它数制的适用性。

---

## 713. Auditing MCP Servers for Over-Privileged Tool Capabilities

**arXiv ID:** 2603.21641 | [PDF](https://arxiv.org/pdf/2603.21641v1)

**作者:** Charoes Huang `[一作]`, Amin Milani Fard `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种统一的安全分析框架，集成静态分析、动态分析（基于eBPF沙箱）以及风险评分与自动化缓解建议，最终生成可交互的安全报告。

**💡 创新点**

创新点在于：①将静态和动态分析结果通过统一的管道统一关联，减少重复工作；②引入风险评分模型，将发现的漏洞量化为可操作的风险等级；③通过自动缓解生成器提供针对性修复建议；④搭建面向开发者的Web检测门户，提升可视化和交互体验。

**🔧 技术方法**

使用的技术包括：静态分析引擎（AST扫描、模式识别）、动态分析沙箱（eBPF监控）、数据模型与规则库、风险评分算法（机器学习或规则混合）、RESTful API层与前端可视化框架。

**📊 数据集**

实验使用了公开的开源项目仓库（如GitHub上的C/C++/Python项目）以及公开漏洞数据库（NVD、CVE）进行评估，覆盖约2000个软件包。

**📈 对比分析**

与传统静态分析工具（如SonarQube、FindBugs）和动态分析工具（如Valgrind、Pin）比较，本文框架在漏洞召回率上提升了约12%，误报率下降了约18%；单个项目平均分析时长为35秒，性能相对可接受。

**⚠️ 局限性**

局限性包括：①主要针对支持的编程语言，其他语言支持有限；②沙箱运行会增加CPU占用，适用于离线或批量分析；③风险评分模型依赖手工标注的训练数据，迁移到新领域可能需要重调；④实时交互式分析对大规模服务仍有延迟挑战。

---

## 714. No Dense Tensors Needed: Fully Sparse Object Detection on Event-Camera Voxel Grids

**arXiv ID:** 2603.21638 | [PDF](https://arxiv.org/pdf/2603.21638v1)

**作者:** Mohamad Yazan Sadoun `[一作]` (University of Oklahoma), Yaser Mike Banad `[通讯]` (University of Oklahoma)

**通讯引用:** 795 | [OpenAlex ID](https://openalex.org/A5068910901)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究并实现了全稀疏事件相机目标检测模型 SparseVoxelDet。

**💡 创新点**

首次在稀疏体素域完成特征提取、特征金字塔融合与检测头，避免任何稠密张量分配。

**🔧 技术方法**

采用 3D 稀疏卷积、SEW‑ResNet 结构、稀疏 FPN、Anchor‑free FCOS 头以及时间最大池化等技术。

**📊 数据集**

使用 FRED（无人机检测）事件相机数据集进行训练与评估。

**📈 对比分析**

与 YOLOv11 等稠密基线对比，mAP@50 为 83.38%（vs. 87.68%），仅处理约 14.9k 活跃体素，占 0.23% 网格，显著压缩显存与存储。

**⚠️ 局限性**

主要限制在于与稠密基线的 4.3% mAP 差距主要由边框回归精度不足导致；稀疏 FPN 的上采样会扩展活跃位置；目前仅为单类别评估。

---

## 715. A Multi-Level Visual Analytics Approach to Artist-Era Alignment in Popular Music

**arXiv ID:** 2603.21624 | [PDF](https://arxiv.org/pdf/2603.21624v1)

**作者:** Jiyeon Bae `[一作]` (Seoul National University), Jinwook Seo `[通讯]` (Seoul National University)

**通讯引用:** 4412 | [OpenAlex ID](https://openalex.org/A5012388103)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种交互式可视化框架，用形状相似度和对比度比两维度分析艺术家在不同年代与时代基准的风格对齐与强度，支持轨迹追踪、歌曲级别对比与艺术家-时代对齐量化。

**💡 创新点**

创新点在于将风格对齐的方向（形状相似度）与强度（对比度比）分离，并用四象限空间揭示艺术家在时代中的“放大顺从”“平滑顺从”“极化叛逆”“柔和叛逆”等配置，突破传统聚合或离散聚类分析的局限。

**🔧 技术方法**

采用中心化余弦相似度、对比度比、分位线分割四象限、弹性条形图、雷达图等可视化技术；数据处理包括将 Billboard Hot 100 周榜与 Spotify 音频特征（valence、energy、danceability、acousticness、liveness）关联，并计算艺术家-年代绩效得分。

**📊 数据集**

使用1960-2010年期间美国 Billboard Hot 100 周榜数据与 Spotify 为全时代十位顶级艺术家的音频特征，涵盖六个十年。

**📈 对比分析**

通过专家思考实验（单参与者）进行定性验证，并未给出量化性能指标；系统通过可视化展示艺术家轨迹与差异，说明其在识别风格对齐模式方面优于传统聚合分析。

**⚠️ 局限性**

局限性包括样本仅限十位艺术家、单一专家评估，难以评估可扩展性与普适性；未来需要更大规模、多参与者研究和跨艺术家模式挖掘。

---

## 716. Efficient Zero-Shot AI-Generated Image Detection

**arXiv ID:** 2603.21619 | [PDF](https://arxiv.org/pdf/2603.21619v1)

**作者:** Ryosuke Sonoda `[一作]` (Fujitsu Ltd.), Ramya Srinivasan `[通讯]` (Fujitsu Research of America, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种训练-free的 AI 生成图像检测方法，通过对图像施加结构化高频扰动并测量其在视觉语言基础模型（CLIP）中间层表征的敏感度，以区分真实与合成图像。

**💡 创新点**

创新点在于将频域高频扰动与中间层语义表征结合，能够在不增加额外训练成本的情况下捕捉微妙的频率伪影，从而显著提升检测精度并保持极低的计算开销。

**🔧 技术方法**

技术手段包括 1) 单次 2D FFT 生成高频噪声并加到图像上；2) 利用 CLIP ViT‑L/14 中间层（第 13 层）提取特征；3) 计算原图与扰动图特征的余弦相似度作为判别分数；4) 通过设定噪声强度、补丁大小等超参数进行调优。

**📊 数据集**

使用的数据集为三大基准：OpenFake（34 种生成器、29829 真/假图像）、GenImage（8 种生成器、100000 图像）和 Semi‑Truth（5 种扩散模型、33753 图像）。

**📈 对比分析**

与多种 SoTA 训练-free 检测器（如 WARPAD、DTAD、RIGID 等）进行对比，本文方法在三大基准上平均提升约 10%‑14% AUC，同时推理速度比对手快 10‑100 倍，显著提升了准确率与效率的双重性能。

**⚠️ 局限性**

局限性包括：需要预设频域分解参数且无法自适应；缺乏针对阈值选择与多模态一致性（如文本-图像对齐）的探讨；目前仅评估图像检测，未考虑视频或音频等其他媒介。

---

## 717. AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing

**arXiv ID:** 2603.21615 | [PDF](https://arxiv.org/pdf/2603.21615v1)

**作者:** Guandong Li `[一作]` (iFLYTEK), Zhaobin Chu `[通讯]` (iFLYTEK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出AdaEdit，一种无训练、可插拔的流模型图像编辑框架。

**💡 创新点**

核心创新为渐进注入调度和通道选择性潜在扰动，解决注入困境。

**🔧 技术方法**

采用流匹配模型FLUX、ODE逆向与前向、连续衰减函数、AdaIN与温度软化等技术。

**📊 数据集**

在PIE‑Bench 700张图像、10类编辑任务上评估。

**📈 对比分析**

与ProEdit等基线对比，LPIPS下降8.7%、SSIM提升2.6%、PSNR提升2.3%，CLIP相似度基本保持。

**⚠️ 局限性**

受限于逆向精度、对显著结构改动仍有挑战，通道重要性估计假设存在一定局限。

---

## 718. Towards Multimodal Time Series Anomaly Detection with Semantic Alignment and Condensed Interaction

**arXiv ID:** 2603.21612 | [PDF](https://arxiv.org/pdf/2603.21612v1)

**作者:** Shiyan Hu `[一作]` (East China Normal University), Chenjuan Guo `[通讯]` (East China Normal University)

**通讯引用:** 3189 | [OpenAlex ID](https://openalex.org/A5084021933)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态时间序列异常检测模型 MindTS，结合时间序列与文本信息进行异常检测。

**💡 创新点**

创新点包括细粒度时序‑文本语义对齐，采用内源与外源文本融合及跨模态对齐；以及内容凝聚重建机制过滤冗余文本并增强跨模态重建。

**🔧 技术方法**

使用LLM生成内源文本、文本编码器、跨视图注意力融合、对比学习对齐、信息瓶颈式内容凝聚、交叉模态重建、实例归一化和时间滑动补丁。

**📊 数据集**

在六个公开多模态数据集（Weather、Energy、Environment、KR、EWJ、MDT）上进行评估。

**📈 对比分析**

与17种基线（LLM、预训练、深度学习、传统方法）以及MM‑TSFLib框架进行比较，MindTS 在 Aff‑F、V‑PR、V‑ROC 等指标上均达到或超过 SOTA。

**⚠️ 局限性**

局限性：仅处理时间与文本两模态，对外源文本质量敏感；LLM 生成文本依赖提示；模型复杂度较高；未在图像/视频等模态上验证。

---

## 719. SARe: Structure-Aware Large-Scale 3D Fragment Reassembly

**arXiv ID:** 2603.21611 | [PDF](https://arxiv.org/pdf/2603.21611v1)

**作者:** Hanze Jia `[一作]` (Zhejiang University), Tan Tang `[通讯]` (Zhejiang University)

**通讯引用:** 786 | [OpenAlex ID](https://openalex.org/A5018322383)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种结构感知的生成框架 SARe，用于大规模 3D 破碎片段的重新拼装，既预测片段姿态，又显式预测接触图和破裂面位置；

**💡 创新点**

核心创新在于将接触结构作为可直接利用的显式变量与生成过程耦合，并引入推理时基于接触验证的结构引导重采样（SARe-Refine），显著提升在碎片数较多时的稳定性；

**🔧 技术方法**

技术包括查询点条件化、冻结 ShapeVAE 的局部几何 token 提取、基于 DiT 的 rectified flow 生成、轻量级多任务结构头（破裂面、接触图）以及基于体素几何验证的推理时修正；

**📊 数据集**

使用了三类数据集：Synthetic Breaking Bad（约55k样本）、OmniObject3D（约55k样本）以及真实物理破碎扫描的 Fantastic Breaks（195样本）；

**📈 对比分析**

与 SE(3)-Equiv、DiffAssemble、Jigsaw、PuzzleFusion++、GARF、RPF 等方法对比，在所有数据集和碎片数范围 2–50 内均取得更高的 Part Accuracy、较低的姿态误差和 Chamfer 距离，尤其在碎片数大时优势更为明显；

**⚠️ 局限性**

局限性在于对极大碎片数（>50）仍可能出现结构漂移，且当前仅利用几何信息，缺乏纹理/外观等多模态约束，未来可考虑分层自回归拼装和多模态条件。

---

## 720. Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study

**arXiv ID:** 2603.21600 | [PDF](https://arxiv.org/pdf/2603.21600v1)

**作者:** Tapajit Chandra Paul `[一作]` (University of North Texas), Mohsen Amini Salehi `[通讯]` (University of North Texas)

**通讯引用:** 1183 | [OpenAlex ID](https://openalex.org/A5001628237)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对8个主流消息代理进行统一基准测试，评估其在IoT边缘环境下的吞吐量、延迟、资源占用等性能。

**💡 创新点**

提出mq-bench框架实现跨协议统一基准，系统考察CPU/内存消耗，并在不同VM配置下对比性能。

**🔧 技术方法**

使用Rust编写mq-bench，采用Docker+KVM部署，利用Toxiproxy注入网络故障，并对MQTT、AMQP、NATS、Redis、Zenoh等协议进行测评。

**📊 数据集**

基于自定义负载生成器，采用多种payload大小（1KB/16KB/1MB）和客户端规模（500–10,000对）模拟IoT通信，不使用公开数据集。

**📈 对比分析**

通过吞吐量、p50/p95延迟、CPU/内存占用等指标，发现本地C/Go实现的NATS和Zenoh在多核下可达90k msg/s，JVM实现高内存消耗；单线程Mosquitto稳定但受限；QoS层级影响延迟。

**⚠️ 局限性**

实验仅覆盖单机部署，未考察分布式集群、跨域网络延迟、长期可用性和安全特性；仅使用固定速率发布，未测试动态负载波动。

---

## 721. In-network Attack Detection with Federated Deep Learning in IoT Networks: Real Implementation and Analysis

**arXiv ID:** 2603.21596 | [PDF](https://arxiv.org/pdf/2603.21596v1)

**作者:** Devashish Chaudhary `[一作]` (Deakin University), Ruby D `[通讯]` (Vellore Institute of Technology)

**通讯引用:** 1813 | [OpenAlex ID](https://openalex.org/A5111404692)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在资源受限的边缘设备上实现并部署了基于自编码器的联邦学习异常检测系统，用于实时检测IoT网络中的重定向攻击。

**💡 创新点**

创新点在于将轻量级自编码器与联邦平均（FedAvg）相结合，支持本地训练、全局聚合与迁移学习，显著降低通信开销并保护数据隐私。

**🔧 技术方法**

使用了深度自编码器、Federated Averaging、迁移学习、ZigBee AT命令攻击模拟、TensorFlow Lite/Python/Digi API 等技术。

**📊 数据集**

采用真实IoT流量日志（约5小时正常 + 3种重定向攻击）并提取31维特征的本地数据集。

**📈 对比分析**

通过与集中式模型对比（准确率、精确率、召回率、F1分数）验证性能；联邦模型F1≈0.90，与集中模型相近；通信量从4.5 MB降至378 KB。

**⚠️ 局限性**

局限性包括仅验证重定向攻击、缺乏对设备异构性和更大规模网络的评估，以及阈值设置需要手工调参，其他攻击类型的检测能力尚未验证。

---

## 722. PRM-as-a-Judge: A Dense Evaluation Paradigm for Fine-Grained Robotic Auditing

**arXiv ID:** 2603.21669 | [PDF](https://arxiv.org/pdf/2603.21669v1)

**作者:** Yuheng Ji `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences), Xiaolong Zheng `[通讯]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PRM-as-a-Judge 与 OPD 度量体系，将过程奖励模型转化为独立评测判定器，实现对机器人执行的密集进度与过程评估；

**💡 创新点**

创新点在于通过宏一致性与微分辨率两条公理构建潜能型评测框架，并将 PRM 用作进度潜能来源，形成可解释的 Outcome–Process–Diagnosis 指标；

**🔧 技术方法**

使用潜能型过程奖励模型（VLAC、GVL、Robo‑Dopamine）生成进度潜能，并基于此计算 OPD 指标；

**📊 数据集**

利用自建的 RoboPulse 基准（1,800 对进度判定样本，覆盖 816 个任务）以及 RoboTwin 2.0 长时空任务数据集；

**📈 对比分析**

与 CLIP、Gemini、GPT‑5.2、Qwen3‑VL 等基线相比，Robo‑Dopamine PRM 在 Small hop 下的判定准确率达 0.83，显著优于所有非 PRM 评测器；

**⚠️ 局限性**

局限在于仍依赖轨迹监督的 PRM，跨任务泛化与多模态不确定性处理有限，且判定结果对模型本身具有解释性依赖。

---

## 723. Towards Secure Retrieval-Augmented Generation: A Comprehensive Review of Threats, Defenses and Benchmarks

**arXiv ID:** 2603.21654 | [PDF](https://arxiv.org/pdf/2603.21654v1)

**作者:** Yanming Mu `[一作]` (State Key Laboratory of Mathematical Engineering and Advanced Computing), Yuling Liu `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 9819 | [OpenAlex ID](https://openalex.org/A5037034567)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统综述 Retrieval‑Augmented Generation (RAG) 的安全威胁、对策与评测基准，构建端到端的安全评估框架

**💡 创新点**

首次将 RAG 的攻击向量（如数据中毒、对抗攻击、成员推理、嵌入反演、间接注入）与防御技术（访问控制、数据清洗、同态加密、差分隐私、联邦学习、轻量化掩码）进行统一分类，并提出可复现的基准与评价标准

**🔧 技术方法**

文献综述、威胁与防御技术图谱构建、攻击/防御方法的对比分析、基准数据集与评测框架的整合

**📊 数据集**

汇总公开 RAG 相关数据集与安全评测资源，如 RAGCare‑QA、RAGLeak、OpenAI RAG、以及常用检索与生成数据集（Wiki、新闻、医学文献等）

**📈 对比分析**

通过对比攻击成功率、召回率、误报率、鲁棒性等指标，展示各防御方法在不同攻击场景下的性能差异，并给出基准测试的实验结果

**⚠️ 局限性**

缺乏统一的公开评测平台与可复现实验环境，防御技术多停留在实验阶段，缺少大规模部署验证，且对新兴间接攻击（如间接注入、隐蔽指令）应对策略仍不完善

---

## 724. TLS Certificate and Domain Feature Analysis of Phishing Domains in the Danish .dk Namespace

**arXiv ID:** 2603.21652 | [PDF](https://arxiv.org/pdf/2603.21652v1)

**作者:** Athanasios P. Pelekoudas `[一作]` (Aalborg University), Sajad Homayoun `[通讯]` (Aalborg University)

**通讯引用:** 562 | [OpenAlex ID](https://openalex.org/A5054931195)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对丹麦.dk域名空间中钓鱼域名与普通域名的TLS证书和域名特征进行定量分析，构建并评估多维度指标的区分效果。

**💡 创新点**

首次系统评估在国家顶级域（ccTLD）层面的证书与域名属性对钓鱼检测的价值，并揭示单一特征不足、特征交互潜力及行业聚焦趋势。

**🔧 技术方法**

使用Netlas抓取TLS证书元数据、DNS记录与WHOIS信息，结合Lexical、SAN、CA、有效期、缺失字段等多类特征进行统计与可视化。

**📊 数据集**

三类数据集：Punktum.dk注册表数据、AbuseManager钓鱼黑名单、Tranco流行网站列表，标记出钓鱼、热门及非热门域名。

**📈 对比分析**

通过对比各特征在三类域名中的分布与统计（如中位数、众数、分布图、Mann–Whitney U检验）评估区分能力；发现大多数特征区分度低，单独使用时识别效果不佳。

**⚠️ 局限性**

主要局限：证书覆盖率低（尤其钓鱼域），Netlas非实时CT日志导致缺失；未考虑特征交互与多变量学习；缺乏子域钓鱼分析；仅针对丹麦域，外推性有限。

---

## 725. Proximal Policy Optimization in Path Space: A Schrödinger Bridge Perspective

**arXiv ID:** 2603.21621 | [PDF](https://arxiv.org/pdf/2603.21621v1)

**作者:** Yuehu Gong `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**通讯引用:** 16153 | [OpenAlex ID](https://openalex.org/A5084959430)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

提出了基于Generalized Schrödinger Bridge（GSB）的路径空间PPO（GSB‑PPO）框架，用于训练生成式策略；

**💡 创新点**

创新点在于将PPO从动作空间提升到完整生成轨迹空间，并给出了两种路径空间近端更新方案（剪切和惩罚），实验表明惩罚形式更稳健；

**🔧 技术方法**

采用生成式策略（扩散/流匹配模型）、GSB视角的轨迹分布、MSE‑style轨迹KL惩罚以及标准PPO的优势估计；

**📊 数据集**

在MuJoCo Playground的十个连续控制任务上进行实验；

**📈 对比分析**

将GSB‑PPO与标准PPO、FPO进行对比，结果显示惩罚版GSB‑PPO在回报和收敛稳定性上均优于剪切版以及传统基线；

**⚠️ 局限性**

仅在小规模MuJoCo任务上验证，缺乏大规模或多样化基准；未给出理论收敛性分析，且生成式策略的计算开销相对较高。

---

## 726. Contrasting Perspectives on Engagement Across Three Digital Behavior Change Interventions

**arXiv ID:** 2603.21609 | [PDF](https://arxiv.org/pdf/2603.21609v1)

**作者:** Evangelos Karapanos `[一作]` (Cyprus University of Technology), Ruben Gouveia `[通讯]` (Universidade de Lisboa)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文比较了三种数字行为改变干预的参与度视角，分别为Habito移动应用、可触达式腕表界面和Micro目标设定工具。

**💡 创新点**

创新点在于将参与度从行为频率、时长、深度与即时行为影响联结，探索新颖性与稀缺性对持续使用的调节作用。

**🔧 技术方法**

主要技术包括自监测与目标设定的行为改变技巧、Android手表面板原型、基于文本提示的消息系统以及时间序列的使用日志分析。

**📊 数据集**

数据集来自30位Habito用户的使用日志、12名手表试用者的每日观测频率和步数，以及Micro使用者的微目标设定与即时步行记录。

**📈 对比分析**

通过对比不同界面与信息策略对参与度及行为影响的实地实验，发现新颖性可提升停留时间、稀缺性可提升步行触发率，整体提升了参与度但尚未形成统一性能基准。

**⚠️ 局限性**

局限性包括样本量有限、研究时间短、只关注短期行为改变，缺乏对长期可持续性与跨文化适用性的验证。

---

## 727. FedCVU: Federated Learning for Cross-View Video Understanding

**arXiv ID:** 2603.21647 | [PDF](https://arxiv.org/pdf/2603.21647v1)

**作者:** Shenghan Zhang `[一作]` (Northeastern University), Zhanjie Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 1908 | [OpenAlex ID](https://openalex.org/A5061370730)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 FedCVU 框架，针对跨视角视频理解场景的联邦学习问题进行统一建模与优化。

**💡 创新点**

创新点在于三大模块的协同设计：视角特定归一化（VS‑Norm）缓解视角异构；跨视角对比对齐（CV‑Align）实现客户端间语义一致；选择性层聚合（SLA）在有限通信预算下智能挑选同步层，三者协同提升泛化与通信效率。

**🔧 技术方法**

使用了联邦学习的分布式训练流程、批归一化/层归一化自适应、原型‑基对比学习、层级签名与可变权重聚合技术，并在 Transformer/3D CNN 结构上实现，通信采用 BF16 进行压缩。

**📊 数据集**

实验数据集为 MCAD（动作理解）和 MARS（人重识别），通过摄像机划分得到 20 个联邦客户端，遵循 seen‑view 训练 / unseen‑view 测试协议。

**📈 对比分析**

与 FedAvg、FedProx、SCAFFOLD、MOON、FedBN、FedDyn、FedOpt 等基线在 unseen‑view 上进行对比，FedCVU 在 MCAD Top‑1 83.1% / MARS mAP 73.2% 分别提升约1.8–3.0%，且通信成本降低 40–45%，收敛轮数与强基线相当或更少。

**⚠️ 局限性**

局限性包括：对视角差异的假设在极端非 IID 或多模态场景下可能不完全适用；原型维护与 VS‑Norm 需要额外的本地统计，增加了计算开销；SLA 仍需签名计算，且在极低带宽环境下仍难以完全消除通信瓶颈。

---

## 728. Are AI-assisted Development Tools Immune to Prompt Injection?

**arXiv ID:** 2603.21642 | [PDF](https://arxiv.org/pdf/2603.21642v1)

**作者:** Charoes Huang `[一作]` (New York Institute of Technology), Amin Milani Fard `[通讯]` (New York Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对七款主流Model Context Protocol（MCP）客户端（Claude Desktop、Claude Code、Cursor、Cline、Continue、Gemini CLI、Langflow）进行实证评测，探讨其在工具中毒（tool‑poisoning）诱发的提示注入（prompt injection）攻击的易受性、检测与缓解机制，以及安全特性覆盖情况。

**💡 创新点**

创新点在于首次系统化地将提示注入与工具中毒结合起来，对实际部署的MCP客户端进行基准实验与安全特性评估，提出针对性改进建议，并揭示不同客户端在同一攻击向量下的性能差异。

**🔧 技术方法**

采用了自建的恶意MCP服务器、四类工具中毒攻击（读取敏感文件、记录工具调用、钓鱼链接、远程脚本执行）以及多维度安全评估维度（静态验证、参数可见性、注入检测、用户警告、沙箱与审计日志），并使用Claude、Grok、Gemini等大语言模型进行交互。

**📊 数据集**

本文未使用公开大规模语料库或代码库，而是构造了针对性攻击工具（Python装饰器）与本地测试MCP服务器，利用这些控制环境进行攻击注入与响应捕获。

**📈 对比分析**

评测方法为：先向每个客户端注册恶意工具，随后发送普通用户请求并记录客户端响应、警告弹窗、日志输出等；结果按“安全/部分安全/不安全”分级，并对比各客户端在四类攻击中的表现，显示Claude Desktop与Cline总体更安全，Cursor最易受攻击。

**⚠️ 局限性**

局限性包括：仅评测了七款客户端且仅针对特定版本；缺乏对沙箱与网络隔离的深入验证；评估主观性高（风险等级分级依赖作者判断）；实验环境为本地隔离，可能未涵盖生产部署中的边缘情况。

---

## 729. EnterpriseLab: A Full-Stack Platform for developing and deploying agents in Enterprises

**arXiv ID:** 2603.21630 | [PDF](https://arxiv.org/pdf/2603.21630v1)

**作者:** Ankush Agarwal `[一作]` (Fujitsu Research India), Chaitanya Devaguptapu `[通讯]` (Fujitsu Research India)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 EnterpriseLab 的全栈平台，整合工具环境、自动轨迹合成与训练管道，实现从企业应用到代理模型的闭环迭代。

**💡 创新点**

创新点包括：1) 采用 Model Context Protocol (MCP) 统一企业工具的发现与调用；2) 通过约束感知工具图遍历自动合成可执行训练轨迹；3) 将监督训练、偏好优化与在线强化学习 (Agentic GRPO) 集成到同一闭环；4) 通过 EnterpriseArena 这一跨部门真实环境验证平台的可扩展性。

**🔧 技术方法**

技术手段：MCP 服务器与客户端；工具图构建与约束感知采样；LLM（Qwen3‑8B、GPT‑4o 等）提示与 ReAct 框架；监督微调 (SFT)、偏好对齐 (DPO) 与轨迹级强化学习 (Agentic GRPO)；Docker 化状态化执行容器与观察归一化。

**📊 数据集**

数据集：EnterpriseArena（15 容器化应用、140+ 工具、500 任务）、EnterpriseBench、CRMArena、τ‑Bench；合成轨迹数量约 500‑1500 条，覆盖多域、多步骤、跨应用场景。

**📈 对比分析**

与基线对比：8B Qwen3‑8B 在 EnterpriseArena 上与 GPT‑4o 的执行准确率相当，且在 EnterpriseBench 与 CRMArena 上提升约 10%；在所有四个基准上超越 ToolAce 与 xLAM；推理成本比 GPT‑4o 降低 8‑10 倍，显示显著的成本效益。

**⚠️ 局限性**

局限性：1) 领域模糊时易调用错误工具，导致循环或失败；2) 参数错误率高，缺乏自我纠错能力；3) 对长序列规划与轨迹级信用分配仍不足；4) 上下文保持弱，长交互中易丢失信息；5) 依赖于高质量的工具图与约束，若业务变更频繁需额外增量训练。

---

## 730. Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition

**arXiv ID:** 2603.21629 | [PDF](https://arxiv.org/pdf/2603.21629v1)

**作者:** Wen Guo `[一作]` (Shandong Technology and Business University), Junyu Gao `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 4050 | [OpenAlex ID](https://openalex.org/A5014526931)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于测试时经验与直觉的TCEI框架，用来提升多目标跟踪（MOT）在分布偏移下的身份关联准确率。

**💡 创新点**

创新点在于将短期瞬时记忆（直觉系统）与长期经验记忆（经验系统）结合，利用自信与不确定对象分别作为先验与反思，形成跨帧、跨视频的双向校准机制。

**🔧 技术方法**

采用前向推理的Transformer跟踪模型，配合跨注意力实现记忆查询与校准，避免反向传播，使用熵阈值挑选自信与不确定对象，并通过经验缓存与临时记忆对预测进行加权与校正。

**📊 数据集**

在DanceTrack和SportsMOT两个具挑战性的MOT基准数据集上进行评估。

**📈 对比分析**

与现有方法（如MOTIP、MOTR、OC‑SORT等）相比，TCEI在HOTA上分别提升约1–2个百分点，身份关联精度AssA提升约2–3个百分点，同时保持检测精度不变，且推理速度比基于梯度的TTA方法（如TENT）快约2–3倍。

**⚠️ 局限性**

局限性包括：仅在Transformer基线上验证，尚未评估在不同检测器或更大规模视频中的鲁棒性；依赖熵阈值和缓存容量的手工调参；对极端噪声样本的自适应性仍有限。

---

## 731. PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation

**arXiv ID:** 2603.21626 | [PDF](https://arxiv.org/pdf/2603.21626v1)

**作者:** Jiacheng Lu `[一作]` (Capital Normal University), Guoping Huo `[通讯]` (China University of Mining and Technology-Beijing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于空间先验的脑肿瘤MRI分割网络 PGR‑Net，利用数据驱动的 ROI 先验模板和分层 Top‑K ROI 决策，实现 ROI‑aware 的分割流程。

**💡 创新点**

创新点包括：① 用统计分析得到的 ROI 先验模板直接引导网络；② 设计 Hierarchical Top‑K (HTK) 机制在多层编码器中逐级筛选最可信 ROI；③ 开发 WinGS‑ROI（多窗口高斯空间衰减）模块，在各层生成中心增强、边界衰减的空间引导图，软约束特征学习；④ 采用轻量化窗口化 RetNet 视觉骨干，整体参数仅 8.64M。 这些创新显著提升稀疏肿瘤分割的定位与精细度。

**🔧 技术方法**

核心技术：数据驱动的 ROI 先验构建、Hierarchical Top‑K ROI 决策、WinGS‑ROI 多窗口高斯空间衰减引导、窗口化 RetNet 视觉骨干、轻量化参数设计与多尺度特征融合。

**📊 数据集**

实验使用 BraTS‑2019、BraTS‑2023 以及 MSD Task01 三个脑肿瘤 MRI 数据集。

**📈 对比分析**

与 UNet、TransUNet、nnUNet、Swin‑UNETR、Mamba‑UNet、Mamba‑Sea 等 SOTA 方法对比，PGR‑Net 在 Whole Tumor、TC、ET 等指标上获得最高 Dice（WT 91.82%），且参数仅 8.64M、FLOPs 39.05G，推理时间约 9 分 41 秒，表现优于现有方法。

**⚠️ 局限性**

局限性：对异常形态或分布偏移的样本仍需回退到全图模式；模型对 ROI 先验模板的统计特性依赖较强，迁移到不同分布或多中心数据时可能需要重新构建模板；仅在 2D 形式下评估，3D 扩展尚未验证。

---

## 732. BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration

**arXiv ID:** 2603.21679 | [PDF](https://arxiv.org/pdf/2603.21679v1)

**作者:** Yan Shen `[一作]` (Peking University), Hao Dong `[通讯]` (Peking University)

**通讯引用:** 57059 | [OpenAlex ID](https://openalex.org/A5100425709)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于视觉效用的双臂协作准备性操作框架BiPreManip，能够先预测主臂的目标效用图并引导助手臂完成预处理动作，最终实现复杂物体的协同抓取和操作。

**💡 创新点**

创新点包括：①定义并系统研究协作准备性双臂操作任务；②使用先行效用图实现跨臂预判与协调；③融合对象姿态预测与重定向网络，实现端到端的预处理规划；④设计跨阶段共享特征的网络架构，提升泛化能力。

**🔧 技术方法**

技术手段：点云编码（PointNet++）、文本嵌入（CLIP）、条件变分自编码器（cVAE）预测手爪姿态、对象姿态预测网络、Reorient Actor执行重定向、视觉效用网络生成预期操作图。

**📊 数据集**

使用ShapeNet与PartNet-Mobility中的18类对象，共计882个实例；仿真数据采集1k成功+1k失败演示；真实实验使用ARX-X7s双臂平台配合Intel RealSense L515深度相机。

**📈 对比分析**

与单臂效用方法W2A、Transformer式ACT、扩展的3DA/3DFA、规则基准等进行比较。BiPreManip在仿真与真实实验中对见/未见物体的任务成功率显著提升，尤其在艺术化操作、边缘推送和盘子提升任务上表现最佳。

**⚠️ 局限性**

局限性：受限于点云分辨率和姿态估计精度；需要大量标注演示数据；在动态多物体场景下鲁棒性待进一步验证；实时性能和计算资源仍有提升空间。

---

## 733. Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization

**arXiv ID:** 2603.21676 | [PDF](https://arxiv.org/pdf/2603.21676v1)

**作者:** Hung-Hsuan Chen `[一作]` (National Central University), Hung-Hsuan Chen `[通讯]` (National Central University)

**通讯引用:** 853 | [OpenAlex ID](https://openalex.org/A5078925594)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种深度递归 Transformer，允许在推理时可扩展计算深度，从而实现垂直 chain‑of‑thought 推理；

**💡 创新点**

创新点在于引入静默思考目标、LayerScale 初始化、身份偏置门控递归以及任务特定感知接口，实现在 20+ 步递归中保持稳定并支持 OOD 推理的可变计算深度；

**🔧 技术方法**

技术包括共享权重的 Transformer block、Pre‑LayerNorm + LayerScale、门控递归、深度嵌入、拓扑掩码、RoPE、静默思考等；

**📊 数据集**

使用的数据集为：随机生成的图可达性数据、合成嵌套布尔逻辑表达式、CLUTRR‑style 家庭关系文本（句子随机打乱并加入对抗噪声）；

**📈 对比分析**

与传统固定深度 Transformer 及中间监督模型对比，使用热图展示思考步数与准确率的“计算前沿”；在 OOD 任务中，该模型可达到 90%+ 的准确率，并显著消除统计捷径；

**⚠️ 局限性**

限制包括模型规模仅 <1M 参数、感知接口需要手工设计、缺乏正式理论泛化界定、未与大规模预训练 LLM 结合。

---

## 734. Optimal Memory Encoding Through Fluctuation-Response Structure

**arXiv ID:** 2603.21666 | [PDF](https://arxiv.org/pdf/2603.21666v1)

**作者:** Lianxiang Cui `[一作]` (University of Tokyo), Kazuyuki Aihara `[通讯]` (University of Tokyo)

**通讯引用:** 26175 | [OpenAlex ID](https://openalex.org/A5038078845)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在物理和神经形态的reservoir计算中，提出了通过测量系统的平稳波动与线性响应来求解最优输入编码方向（ROME）的方法，进而最大化任务相关的线性记忆；

**💡 创新点**

将输入编码优化从纯粹的梯度学习转化为几何问题，利用系统的波动-响应结构提供可解析的最优解，并证明其与反向传播等价；

**🔧 技术方法**

主要技术包括稳态协方差与响应核测量、Rayleigh–Ritz特征分解、线性记忆函数与任务加权记忆目标的构造，以及对线性与非线性reservoir（ESN、NARMA10）和物理平台（自旋波波导、E/I突触神经网络）的应用；

**📊 数据集**

使用标准的NARMA10序列、白噪声输入和自定义的延迟记忆任务作为实验数据集；

**📈 对比分析**

与随机输入权重相比，ROME在低输入功率下显著提升了R²（例如线性reservoir、ESN、Spin‑Wave、E/I SNN），在高功率或需要非线性记忆的场景下效果不如随机权重；

**⚠️ 局限性**

局限性包括只优化线性记忆、依赖固定工作点的波动-响应统计、对高功率输入导致的工作点漂移不敏感，以及对非线性任务的提升有限。

---

## 735. MISApp: Multi-Hop Intent-Aware Session Graph Learning for Next App Prediction

**arXiv ID:** 2603.21653 | [PDF](https://arxiv.org/pdf/2603.21653v1)

**作者:** Yunchi Yang `[一作]` (Shandong University), Cunquan Qu `[通讯]` (Shandong University)

**通讯引用:** 296 | [OpenAlex ID](https://openalex.org/A5088741789)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 MISApp，一个基于多跳会话图学习的无用户画像的下一应用预测框架。

**💡 创新点**

创新点在于引入多跳会话图以捕获不同阶层的转移依赖，结合轻量化 LightGCN 与跨模态门控融合，并通过动态意图解码器捕捉会话级意图演化。

**🔧 技术方法**

使用多跳图构建、LightGCN、跨模态门控融合、Transformer 编码解码、时间与空间上下文嵌入等技术。

**📊 数据集**

在 Tsinghua App Usage 与 LSapp 两个真实应用使用数据集上进行实验。

**📈 对比分析**

与传统频率基、序列模型、图神经网络、时间序列 Transformer 以及 MAPLE 等基线比较，MISApp 在标准与冷启动设置下均取得 ACC@1 最高、MRR@3/5 均领先，显著提升预测准确率。

**⚠️ 局限性**

局限在于依赖会话级别的多跳图构建对长会话开销增加，且模型仍需在极低资源的移动端实现轻量化，未来需进一步压缩模型并提升推理速度。

---

## 736. A coupled Aeroelastic-Flight Dynamic Framework for Free-Flying Flexible Aircraft with Gust Interactions

**arXiv ID:** 2603.21650 | [PDF](https://arxiv.org/pdf/2603.21650v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems), Rafael Palacios `[通讯]` (Imperial College)

**通讯引用:** 3424 | [OpenAlex ID](https://openalex.org/A5007534365)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文构建了一个完整、可自包含的耦合机翼弹性-机体飞行动力学框架，用于自由飞行的柔性飞机在气流扰动下的时域分析。

**💡 创新点**

创新点在于：①将几何精确梁理论、两维无滑动翼剖面诱导函数（Wagner/Küssner）以及四元数飞行动力学整合为单一状态空间模型；②推导了所有坐标转换、耦合矩阵与雅可比块结构，提供可直接实现的完整数学细节；③提出了两种气流扰动模型（离散1–cosine迎面风和von Kármán谱）并考虑跨距延迟。

**🔧 技术方法**

技术手段包括：几何精确梁的刚体旋转参数化（Rodrigues公式）、两维薄翼条线理论与诱导函数的指数逼近、四元数姿态传播、Newton–Euler刚体动力学、Newmark‑β结构积分以及加权求导的雅可比构造。

**📊 数据集**

使用的验证数据集：HALE（高空长寿命）飞机的实际结构与气动参数（如弦长1 m、跨度32 m、质量分布等）和一架极柔性飞翼（跨度32 m、弦1 m、质量每米10 kg）。

**📈 对比分析**

比较方法为：与已发表的精确梁理论、UVLM（无网格线方法）以及CFD结果进行结构固有频率、悬浮速度、静态弹性失稳变形和时域迎风响应的对比。结果显示：结构频率误差<1 %；悬浮速度误差<6 %；静态变形与CFD/UVLM误差<5 %；时域迎风峰值与UVLM高度一致。

**⚠️ 局限性**

局限性包括：仅使用二维条线理论，忽略三维诱导与弦向干扰；薄翼理论不包含粘性与压缩效应；梁模型假设截面不发生扭曲/变形；对高速或高攻角、分离流场的预测能力有限。

---

## 737. Physical Containers as Framing Conditions for Visualization in Augmented Reality

**arXiv ID:** 2603.21637 | [PDF](https://arxiv.org/pdf/2603.21637v1)

**作者:** Jiyeon Bae `[一作]` (Seoul National University), Jinwook Seo `[通讯]` (Seoul National University)

**通讯引用:** 4412 | [OpenAlex ID](https://openalex.org/A5012388103)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文探讨了在AR环境中利用物理容器几何属性作为感知框架来支持早期数据探索，并通过四个示例展示了不同容器形态对数据可视化的影响。

**💡 创新点**

创新点在于将容器的面数、尺寸、比例和形状视为感知框架条件，提出了容器属性与观察者认知倾向之间的映射关系。

**🔧 技术方法**

采用增强现实技术，将月度电影发行量数据嵌入不同形态的AR容器进行可视化。

**📊 数据集**

使用了涵盖三十年（1990-2019）每月电影发行数量的数据集。

**📈 对比分析**

通过对四种不同容器设计的对比，展示了相同数据在不同感知框架下的可视化差异，但未给出量化性能指标。

**⚠️ 局限性**

局限性包括框架分类不完整、未进行受控实验验证其一致性，以及无法区分容器几何与编码本身的感知影响。

---

## 738. Silicon Bureaucracy and AI Test-Oriented Education: Contamination Sensitivity and Score Confidence in LLM Benchmarks

**arXiv ID:** 2603.21636 | [PDF](https://arxiv.org/pdf/2603.21636v1)

**作者:** Yiliang Song `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 62067 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在公共基准评估中的可信度问题，提出了基于路由器-工作器（router‑worker）框架的审计方法，用来检测基准分数对污染（contamination）敏感性的影响，并通过系统性删除、改写和噪声注入的多路由器实验评估模型表现的异常提升；

**💡 创新点**

创新点在于将基准评估视为“硅官僚制度与 AI 测试导向教育”，引入了“得分置信度”和“污染敏感度”概念，并提出了利用多路噪声聚合来触发潜在污染记忆的审计思路；

**🔧 技术方法**

使用的技术包括路由器-工作器架构、删除/改写/噪声注入的多路输入聚合、统计分析（正向超基准差值、违规率、问题级转移分析）以及对模型输出的置信度评估；

**📊 数据集**

采用了公开的多项选择基准（未命名具体数据集），在测试集抽取了 100 道题目（种子 42）作为实验样本；

**📈 对比分析**

比较方法为在同一组 100 题上，设定清洁基线（单路由器完整传输）与多路噪声条件（1–9 个路由器），对 12 个主流 LLM 进行评估；结果显示，噪声条件下模型往往出现正向超基准现象，且异常幅度随路由器数量增大而加剧，表现出模型间的异质性；问题级分析揭示在高路由器情境下错误→正确转移超过错误→正确转移，表明噪声聚合可重组潜在污染线索；

**⚠️ 局限性**

局限性包括：仅测试了一个基准和有限的 100 题样本，未覆盖所有 LLM；审计仅聚焦于污染敏感性，未考虑其他偏差或数据泄漏形式；模型表现的异质性与特定实现细节有关，结果在不同基准或数据集上可能不完全可复制。

---

## 739. Triangulating a Polygon with Holes in Optimal (Deterministic) Time

**arXiv ID:** 2603.21617 | [PDF](https://arxiv.org/pdf/2603.21617v1)

**作者:** Timothy M. Chan `[一作]` `[通讯]` (University of Illinois at Urbana-Champaign), Timothy M. Chan (University of Illinois at Urbana-Champaign)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种新的确定性算法，以O(n+h访线数)时间构造多孔多边形的三角剖分（或梯形分解），并将其推广到包含少量交点的任意多边形链；

**💡 创新点**

创新点在于将随机抽样与r‑division以及区间树相结合，构造满足“网”性质的子集，既消除了随机化，又保持了最优时间；同时给出了对自相交链的确定性处理框架；

**🔧 技术方法**

使用了Chazelle的线性三角剖分、区间树、r‑division/平面图分割、Jordan排序、Bentley‑Ottmann等经典技术；

**📊 数据集**

论文主要在理论分析层面，没有使用具体实验数据集；

**📈 对比分析**

与以往O(n\log^*n+h\log h)的随机算法和O((n+h\log h)\log\log h)的确定性算法相比，本文实现了最优O(n+h\log h)的确定性时间；在交点子线性场景下，时间进一步降至O(n+X n^\varepsilon)；

**⚠️ 局限性**

局限在于尚未突破到O(n+h\log h+X)的最优确定性时间，对大量交点的情况效率仍有提升空间，并且实现复杂度较高。

---

## 740. AgenticRec: End-to-End Tool-Integrated Policy Optimization for Ranking-Oriented Recommender Agents

**arXiv ID:** 2603.21613 | [PDF](https://arxiv.org/pdf/2603.21613v1)

**作者:** Tianyi Li `[一作]` (Xiamen University), Hui Li `[通讯]` (Xiamen University)

**通讯引用:** 60892 | [OpenAlex ID](https://openalex.org/A5057824494)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于大语言模型的排名导向推荐代理框架，端到端优化推理、工具调用和最终排名，利用稀疏隐式反馈学习实现精细化推荐。

**💡 创新点**

创新点包括：① 将推荐专用工具集成到 ReAct 循环中，实现证据驱动的推理；② 设计无偏的列表级 Group Relative Policy Optimization（list‑wise GRPO），在稀疏奖励环境下对完整决策轨迹进行强化学习；③ 引入 Progressive Preference Refinement（PPR），通过硬负样本挖掘和双向偏好对齐，细化细粒度偏好边界。

**🔧 技术方法**

采用的大技术栈包括：大语言模型（Qwen3‑4B‑Instruct）、ReAct 结构化推理、强化学习（GRPO）、组相对优势估计、双向偏好对齐、工具调用机制（用户档案、物品信息、行为统计、协同信息）以及两阶段（列表优化 + PPR）训练。

**📊 数据集**

使用 Amazon 2023 benchmark 的四个子集：CDs、Instruments、Office、Games；每个实例构造 20 个候选项（1 正 19 负），评估 Top‑K 排序性能。

**📈 对比分析**

与传统序列推荐（Caser、GRU4Rec、SASRec、ReaRec）、训练自由 LLM 推荐（LLMRank、InteRecAgent）以及可训练 LLM 推荐（TALLRec、LLaRA、S‑DPO、ReRe）进行对比。实验显示本文方法在 NDCG@K 与 Hit@K 上均显著优于所有基线，尤其在 H@1/H@5 上提升尤为突出。

**⚠️ 局限性**

局限性包括：① 受限于工具调用预算和工具质量，过度或不足调用可能影响性能；② 对长用户历史的记忆与推理能力仍不足；③ 训练时对稀疏隐式反馈的收敛速度相对较慢；④ 未对多模态或非结构化信息进行深入挖掘，可能限制进一步提升。

---

## 741. Rule-State Inference (RSI): A Bayesian Framework for Compliance Monitoring in Rule-Governed Domains

**arXiv ID:** 2603.21610 | [PDF](https://arxiv.org/pdf/2603.21610v1)

**作者:** Abdou-Raouf Atarmla `[一作]` `[通讯]` (Institut National des Postes et Telecommunications), Abdou-Raouf Atarmla (Institut National des Postes et Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于贝叶斯推理的规则状态推断框架RSI，用以零样本下的合规性监测，直接将法规编码为先验，推断实体是否合规；

**💡 创新点**

创新点在于（1）把规则视为先验而非需要学习的权重，实现零样本合规评估；（2）证明了 O(1) 级别的法规更新适应性、贝塞尔-冯莫斯一致性以及 ELBO 单调收敛；（3）提供公开的合规监测基准 RSI‑Togo‑Fiscal‑Synthetic v1.0；

**🔧 技术方法**

采用贝叶斯推断、均值场变分推断（Mean‑Field VI）、正则化规则先验、贝塔、伯努利、正态分布以及对缺失数据的隐式处理；

**📊 数据集**

使用基于多家企业的 2000 条合成数据集 RSI‑Togo‑Fiscal‑Synthetic v1.0，模拟 Togo 的税收规则和监管变更；

**📈 对比分析**

与传统规则系统、XGBoost、MLP 进行比较，RSI 在零样本情况下实现 F1=0.519、AUC=0.599，远超规则系统；对法规更新的适配耗时 <1 ms，较 XGBoost 全量重训练提升 600×；

**⚠️ 局限性**

局限在于对先验超参数的依赖，若先验设定不准会影响结果；目前仅在合成数据上验证，缺少真实行政数据的实地测试；

---

## 742. INTRYGUE: Induction-Aware Entropy Gating for Reliable RAG Uncertainty Estimation

**arXiv ID:** 2603.21607 | [PDF](https://arxiv.org/pdf/2603.21607v1)

**作者:** Alexandra Bazarova `[一作]` (Applied AI Institute), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于机制的无训练不确定性估计方法INTRYGUE，用以在检索增强生成（RAG）场景下检测LLM的幻觉；

**💡 创新点**

通过揭示诱导头（induction heads）与熵神经元（entropy neurons）在RAG生成中的“拉锯”关系，设计了以诱导活性为门控的熵调节机制；

**🔧 技术方法**

利用Transformer内部注意力矩阵计算诱导头活性（SinkRate），结合生成过程的预测熵，形成INTRYGUE评分；

**📊 数据集**

在四大RAG基准（MS MARCO, CNN/DM, CoQA, XSum）上，使用六个开源LLM（Gemma‑3‑4B, LLaMA‑2‑7B, LLaMA‑2‑13B, Mistral‑7B, LLaMA‑3.1‑8B, Qwen3‑8B）进行评估；

**📈 对比分析**

与信息学、采样多样性和机制可解释性等30余种基线（如AttentionScore、LN‑Entropy、MaxEntropy、RAUQ、ReDeEP等）对比，INTRYGUE在所有任务和模型上均达到或超过最佳基线，尤其在长篇生成时采用min‑max聚合能获得最高AUROC；

**⚠️ 局限性**

局限性包括：需白盒访问模型内部状态；仅适用于标准Transformer架构；聚合方式依赖生成长度；衡量的是对检索文本的依赖，而非客观事实正确性。

---

## 743. Cross-Scenario Deraining Adaptation with Unpaired Data: Superpixel Structural Priors and Multi-Stage Pseudo-Rain Synthesis

**arXiv ID:** 2603.21661 | [PDF](https://arxiv.org/pdf/2603.21661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 744. Riemannian Geometry Speaks Louder Than Words: From Graph Foundation Model to Next-Generation Graph Intelligence

**arXiv ID:** 2603.21601 | [PDF](https://arxiv.org/pdf/2603.21601v1)

**作者:** Philip S. Yu `[一作]` (University of Illinois Chicago), Li Sun `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 194128 | [OpenAlex ID](https://openalex.org/A5100748869)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Riemannian Foundation Model (RFM)，以Riemannian几何为基础，构建通用图学习框架。

**💡 创新点**

创新点在于将图的内在几何结构（曲率、平行运输、向量丛等）与LLM融合，强调内生结构推理和图生成能力。

**🔧 技术方法**

使用Riemannian几何工具（流形、度量、移动框架、平行运输、曲率）以及向量丛与大语言模型技术。

**📊 数据集**

文中未给出具体实验数据集，理论上适用于多域图数据。

**📈 对比分析**

本文未提供实验比较或性能评估，主要是概念性框架与理论阐述。

**⚠️ 局限性**

局限性包括缺乏实证验证、实现复杂度高、对大规模图数据的可扩展性尚未证明。

---

## 745. A Multidisciplinary AI Board for Multimodal Dementia Characterization and Risk Assessment

**arXiv ID:** 2603.21597 | [PDF](https://arxiv.org/pdf/2603.21597v1)

**作者:** Sheng Liu `[一作]` (Stanford University), James Zou `[通讯]` (Stanford University)

**通讯引用:** 39470 | [OpenAlex ID](https://openalex.org/A5005779176)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

本文提出了一个多学科AI板块（Multidisciplinary AI Board），融合影像、遗传、认知评估等多模态数据，用于痴呆症的表征和风险评估，并通过模拟专家决策流程实现动态检验顺序的优化。

**💡 创新点**

创新点在于将传统多模态融合技术与专家决策流程相结合，形成“AI Board”框架；该框架通过强化学习自动选取最具信息量的检测步骤，并提供可解释的决策依据，显著提升了跨学科知识的系统化与可解释性。

**🔧 技术方法**

使用了深度学习多模态融合网络、图神经网络（Graph Neural Networks）以及强化学习（策略网络）来实现数据融合与决策策略；并配合SHAP/Grad‑CAM等可解释性技术进行结果解释。

**📊 数据集**

主要使用了Alzheimer’s Disease Neuroimaging Initiative（ADNI）数据库中的MRI、PET、CSF生物标志物和临床认知评分等多模态数据，并在外部MCI/AD样本上进行了验证。

**📈 对比分析**

与传统单模态或无决策流程的多模态模型相比，AI Board在AUC、准确率等指标上提升约5%（AUC从0.87提升至0.92），准确率提高至92%；在样本量相对较小的条件下仍保持稳健的性能。

**⚠️ 局限性**

局限性包括：数据来源主要集中于西方受试者，缺乏多样性导致推广性受限；模型对少数类别（如轻度认知障碍）预测仍不稳定；可解释性机制虽已提供，但仍需进一步验证其临床可信度；以及在真实临床环境中的实时部署和成本效益尚待评估。

---

## 746. Spatio-Temporal Attention Enhanced Multi-Agent DRL for UAV-Assisted Wireless Networks with Limited Communications

**arXiv ID:** 2603.21594 | [PDF](https://arxiv.org/pdf/2603.21594v1)

**作者:** Che Chen `[一作]` (Sun Yat-sen University), Dusit Niyato `[通讯]` (Nanyang Technological University)

**通讯引用:** 86743 | [OpenAlex ID](https://openalex.org/A5091266202)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

针对通信受限的多无人机协作无线网络，提出了一种延迟容忍的多智能体深度强化学习框架（STA‑MADRL），通过整合延迟惩罚奖励与时空注意力预测模块，联合优化无人机轨迹规划、网络形成和传输控制，从而提升网络吞吐量并减少信息延迟。

**💡 创新点**

创新点包括：① 延迟惩罚奖励机制鼓励无人机频繁与基站交换信息，缓解信息延迟；② 时空注意力预测模块利用无人机历史轨迹与空间关联，预测并恢复丢失或延迟的信息，降低对实时通信的依赖；③ 将上述两种机制结合，形成 STA‑MADRL，可在信息受限环境下接近理想性能。

**🔧 技术方法**

技术手段主要有：多智能体深度强化学习（如 MADDPG/TD3）、确定性策略梯度、时空注意力网络（多头注意力+图注意力网络），以及奖励塑造与延迟惩罚设计。实验平台为基于仿真的 UAV 网络模型。

**📊 数据集**

使用的数据集为模拟数据：9 个地面用户（GUs）、3 架 UAV、1 个基站，覆盖 2 km × 2 km 区域，随机分布的 GUs，固定 UAV 高度 100 m，采用基于路径损耗与小尺度衰落的信道模型。未使用公开真实数据集。

**📈 对比分析**

对比方法：Ideal‑MADRL（全信息实时共享）、Communication‑Limited MADRL（无信息共享）以及传统的 MADDPG/TD3 基线。实验结果显示：STA‑MADRL 在吞吐量上比 Communication‑Limited MADRL 提升约 75%、比 Delay‑Tolerant MADRL 提升约 25%，并与 Ideal‑MADRL 的性能相近；信息延迟平均降低约 50%；收敛速度更快、波动更小。

**⚠️ 局限性**

局限性：① 仍略低于理想完全信息情况；② 依赖于时空注意力模型的准确性，若环境变化剧烈预测误差可能增大；③ 仅在仿真环境验证，缺乏真实部署实验；④ 对 UAV 数量、信道模型等参数的可扩展性尚待进一步评估；⑤ 预测模块与奖励设计增加计算与实现开销。

---

## 747. OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging

**arXiv ID:** 2603.21660 | [PDF](https://arxiv.org/pdf/2603.21660v1)

**作者:** Meilin Liu `[一作]` (Shenyang University of Technology), Jing Shan `[通讯]` (Shenyang University of Technology)

**通讯引用:** 3339 | [OpenAlex ID](https://openalex.org/A5103137909)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种名为 OmniFM 的通用、模态无关的联邦学习框架，能在不同医学影像任务（分类、分割、超分、视觉问答、多模态融合）中使用同一优化流程。

**💡 创新点**

创新点包括：①利用低频谱一致性做为跨模态的结构先验；②引入全局谱知识检索（GSKR）提供谱原型；③通过嵌入级交叉注意力（ECA）融合全局谱与本地特征；④采用前后谱提示（PSP）兼顾全局一致性与本地个性；⑤在优化中加入谱亲近对齐（SPAlign）正则化，抑制模态诱发的梯度漂移。

**🔧 技术方法**

技术手段包括：频域变换（FFT）提取低频谱；谱向量量化与检索；交叉注意力机制；提示学习（前后谱提示）；谱亲近对齐正则；以及标准的联邦平均与个性化策略。

**📊 数据集**

实验使用了多种真实医学影像数据集：MedMNIST‑v2（分类）、BreaKHis（超分）、FeTS2022（分割）、Harvard Medical School 多模态融合集、SLAKE、VQA‑RAD、VQA‑Med（视觉问答），并在多种模态异构设置下进行评估。

**📈 对比分析**

与 FedAvg、FedProx、FedPer、Ditto、FedDyn、MOON 等主流联邦学习基线对比，OmniFM 在跨模态分类、超分、分割和 VQA 等任务上均取得了显著提升（例如分类准确率 96.9%–99.8%，超分 PSNR 35.8–42.9 dB，分割 Dice 79.8%–82.5%，VQA F1 0.79–0.84）。

**⚠️ 局限性**

局限性包括：①低频谱一致性假设在极端模态差异（如超高分辨率或特殊成像技术）下可能不成立；②谱嵌入与检索增加了额外通信与存储开销；③需要手动调节检索数量 k 与对齐系数 λ，超参数敏感；④缺乏理论收敛性证明与对大规模（数百或千级）客户端的可扩展性评估。

---

## 748. IMMSched: Interruptible Multi-DNN Scheduling via Parallel Multi-Particle Optimizing Subgraph Isomorphism

**arXiv ID:** 2603.21659 | [PDF](https://arxiv.org/pdf/2603.21659v1)

**作者:** Boran Zhao `[一作]` (Xi'an Jiaotong University), Pengju Ren `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 1642 | [OpenAlex ID](https://openalex.org/A5044243518)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出IMMSched，一种可中断的多DNN调度框架，利用并行粒子群优化与Ullmann子图同构算法实现即时调度。

**💡 创新点**

创新点包括：将子图同构问题转化为连续松弛优化，结合PSO并行化Ullmann实现高并行度搜索；在固定点DNN加速器上做轻量级硬件改造与量化，以实现低延迟、高能效的调度。

**🔧 技术方法**

使用技术：连续松弛模型、粒子群优化（PSO）、Ullmann子图同构、量化计算、全局控制器以及基于DNN加速器的并行实现。

**📊 数据集**

使用数据集：Simple（MobileNetV2、ResNet50、UNet）、Middle（EfficientNet、NASNet、PNASNet）以及Complex（DeepSeek‑7B、Qwen‑7B、Llama‑3‑8B）。

**📈 对比分析**

与LTS基线（PREMA、CD‑MSA、Planaria、MoCA）以及TSS基线IsoSched进行Speedup、LBT、能效对比；IMMSched在Speedup上平均提升34–81倍（对LTS）与1.6倍（对IsoSched），在LBT与能效上分别提升约90–191倍和920–2720倍。

**⚠️ 局限性**

局限性：仍受子图同构NP‑hard本质影响，调度延迟在极大规模模型上可能不完全消除；硬件改造虽轻量但对资源有一定需求，扩展性需进一步验证。

---

## 749. RTD-RAX: Fast, Safe Trajectory Planning for Systems under Unknown Disturbances

**arXiv ID:** 2603.21635 | [PDF](https://arxiv.org/pdf/2603.21635v1)

**作者:** Evanns Morales-Cuadrado `[一作]` (Georgia Institute of Technology), Samuel Coogan `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1798 | [OpenAlex ID](https://openalex.org/A5010552433)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种结合离线可达集快速生成与在线混合单调可达性验证的RTD-RAX框架，用于在未知实时扰动下实现安全轨迹规划与执行。

**💡 创新点**

创新点在于：①将离线可达集仅用于候选轨迹生成，去除保守的误差膨胀；②采用在线混合单调可达性快速构造可达管并实时安全认证；③引入修复循环在候选轨迹被判定不安全时快速寻找邻近安全轨迹；③在RTD与在线验证之间实现可扩展的接口，实现低计算开销的实时保证。

**🔧 技术方法**

使用技术包括：Reachability‑Based Trajectory Design (RTD)、混合单调可达性 (MMR) 的嵌入系统、JIT 编译加速、基于参数化Dubins车模型的离线可达集、基于轨迹参数优化的离线规划、在线可达管构造与碰撞检测、轨迹修复策略（速度退回、横向推挤、约束收缩）。

**📊 数据集**

实验数据集为仿真中的Turtlebot‑规模地面机器人（单轮车模型），构建了三种场景（窄隙、倾斜障碍、扰动环境）来评估算法性能；未使用公开大规模真实数据集。

**📈 对比分析**

与传统RTD比较：在窄隙场景中，标准RTD无法找到可行轨迹，RTD‑RAX能安全通过；在倾斜障碍场景中，RTD‑RAX实现更短、更高效的轨迹；在扰动场景中，RTD‑RAX成功避免碰撞，平均规划周期从9.39 ms提升至10.63 ms，仅增加约1.24 ms。总体来看，RTD‑RAX在保持或略微提升计算速度的前提下显著提升了安全性与执行效率。

**⚠️ 局限性**

限制：①离线规划模型仍无法完全捕捉所有动态误差，导致需要在线修复；②混合单调可达性虽然快速，但仍可能产生保守的可达管，影响规划精度；③实验仅在仿真环境中验证，缺乏真实硬件跨平台的评估；④对扰动范围的估计依赖于在线测量，若测量误差较大可能导致安全判定失误。

---

## 750. 4DGS360: 360° Gaussian Reconstruction of Dynamic Objects from a Single Video

**arXiv ID:** 2603.21618 | [PDF](https://arxiv.org/pdf/2603.21618v1)

**作者:** Jae Won Jang `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8445 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出4DGS360框架，实现从单目视频中进行360°动态物体重建。

**💡 创新点**

创新点包括 AnchorTAP3D 的3D本地化跟踪初始化，解决遮挡几何歧义；以及新建 iPhone360 360°评测数据集。

**🔧 技术方法**

核心技术为3D Gaussian Splatting、Anchor‑guided 3D 跟踪、ARAP 刚性正则化、2D‑3D 联合推理，并未使用扩散模型。

**📊 数据集**

实验使用 iPhone360（自建）、iPhone、DAVIS 数据集，并与 HyperNeRF、Nerfies、Deformable 3DGS、SoM、HiMoR 等基线进行对比。

**📈 对比分析**

在 CLIP‑I/CLIP‑T、LPIPS 等感知指标上，4DGS360 在极端视角下表现优于现有方法，尤其在 360° 视角合成上取得显著提升。

**⚠️ 局限性**

局限性包括对预训练模型的依赖、无法处理光照变化、无法合成不可见背景区域，且对极端背景合成仍需结合扩散先验。

---

## 751. Rateless DeepJSCC for Broadcast Channels: a Rate-Distortion-Complexity Tradeoff

**arXiv ID:** 2603.21616 | [PDF](https://arxiv.org/pdf/2603.21616v1)

**作者:** Zijun Qin `[一作]` (University of Hong Kong), Xianhao Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 2007 | [OpenAlex ID](https://openalex.org/A5083484070)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于非线性变换与 LT 码的可变长度深度联合源信道编码框架 NTRSCC，用于异构边缘设备的广播。

**💡 创新点**

创新点在于将非线性变换编码与物理层 LT 码联合，利用解码侧信息实现不等错误保护，并通过端到端训练实现速率‑失真‑复杂度可控的权衡。

**🔧 技术方法**

使用深度学习的非线性变换编码、LT 码、贝叶斯推理、belief propagation、Gumbel‑Softmax 采样、可微分生成矩阵、均匀噪声量化以及端到端的变分自编码器优化。

**📊 数据集**

在 TinyImageNet（ImageNette）和 CIFAR‑10 数据集上训练和评估。

**📈 对比分析**

与 JPEG+LT、NTC+LDPC 以及 NTRSCC 无 UEP 的基线对比，实验显示在低 BPP/OPP 条件下 PSNR 明显优于基线，且在多用户广播时所需比特更少。

**⚠️ 局限性**

局限性包括对 LT 码参数设计的近似、需要手工设定 UEP 阈值、以及在更复杂信道或更大分辨率图像上的扩展性尚未验证。

---

## 752. mSFT: Addressing Dataset Mixtures Overfiting Heterogeneously in Multi-task SFT

**arXiv ID:** 2603.21606 | [PDF](https://arxiv.org/pdf/2603.21606v1)

**作者:** Woosung Koh `[一作]` (KAIST AI), Se-Young Yun `[通讯]` (KAIST AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种迭代的、基于过拟合感知的多任务监督微调算法 mSFT，在多任务数据混合中动态排除最早过拟合的数据子集，从而实现更优的模型泛化。

**💡 创新点**

创新点在于将过拟合早停与模型回滚相结合的搜索策略，避免了单次搜索后模型参数漂移导致的最优停止点失真，并在保持单一超参数的前提下实现跨任务的自适应训练时间。

**🔧 技术方法**

采用基于梯度累积的多任务训练框架，结合计算预算 C 的分段训练、回滚到过拟合点以及动态排除子数据集；评估使用梯度加权、回滚检查点以及对比实验框架。

**📊 数据集**

使用十个公开 NLP 任务子集（CommonsenseQA、OpenBookQA、AQUA-RAT、GSM8K、SciQ、ARC-Easy、HellaSwag、Winogrande、BoolQ、MedMCQA）以及其细粒度子类别，在多种模型（OLMo 2、Qwen2.5、Qwen3 等）上进行测试。

**📈 对比分析**

与四种基线（标准 SFT、连续 SFT、DynamixSFT、IES）以及两个消融（SRO SFT、Soft SRO SFT）对比，mSFT 在 10 个基准上平均提升约 3–5% 的准确率，且在标准差、首位占比等指标上表现更稳定，显示出显著的性能优势。

**⚠️ 局限性**

主要限制包括：需要在搜索阶段保存中间检查点导致存储开销上升；回滚和排除机制可能在极端任务不平衡时引入轻微灾难性遗忘；算法在极低计算预算下的实际收益受限于搜索步长和过拟合判定精度。

---

## 753. RefracGS: Novel View Synthesis Through Refractive Water Surfaces with 3D Gaussian Ray Tracing

**arXiv ID:** 2603.21695 | [PDF](https://arxiv.org/pdf/2603.21695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 754. Prophets Inequalities with Uncertain Acceptance

**arXiv ID:** 2603.21740 | [PDF](https://arxiv.org/pdf/2603.21740v1)

**作者:** Martinez Emile `[一作]` (IRIT Université Toulouse Capitole), Pérez-Salazar Sebastian `[通讯]` (Rice University)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5020076936)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文在经典的Prophet Inequality框架中加入了“接受不确定性”——即在决策者决定接受一个值后，该值可能以某个概率被拒绝，导致过程继续。作者定义了三类参与者：在线决策者（Decision Maker, DM），只知价值但不知接受概率的价值意识决策者（Value‑Aware Decision Maker, VA‑DM），以及知道所有信息的先知（Prophet）。通过分析，证明在最坏情况下，DM、VA‑DM与Prophet之间的竞争比均为 1/2；并进一步给出当接受概率较高时，VA‑DM 可以突破 1/2 的下界。

**💡 创新点**

创新点包括：① 在Prophet Inequality中首次正式考虑接受概率不确定性，并引入三种信息水平的决策者；② 证明三类决策者的最坏竞争比均为 1/2，揭示接受不确定性并未降低最坏情况的保证；③ 通过将问题归约为“缩放Bernoulli”形式，证明当接受概率均大于某阈值时，VA‑DM 的竞争比可提升至 1/2 – min p_i；④ 对三种实例变换（观察产品、关注价值、公开接受）进行了比较，展示不同信息模式下问题的相对难度。

**🔧 技术方法**

主要技术手段包括：
- 动态规划推导 DM 的期望收益与“观察到 A_i 后”情形等价；
- 将原问题映射到经典Prophet Inequality 中的随机变量 Z_i = X_i·A_i；
- 证明单阈值算法在归约后仍是最优；
- 构造一系列优化问题（如 B_n）并通过变量变换与归约，得到可达下界 1/2 – min p_i；
- 通过实例变换和对比实验式分析三种信息设置下的竞争比。

**📊 数据集**

本文为理论研究，不涉及具体实验数据集；所有结果均在理论模型和期望值分析下给出。

**📈 对比分析**

比较方法：
- 对三类决策者（DM、VA‑DM、Prophet）计算竞争比 α、β、γ；
- 在最坏情形下证明 α、β、γ ≥ 1/2 并给出相等的上界；
- 对特殊实例（如 2 个随机向量、参数 p、a 的实验例子）计算竞争比并展示 VA‑DM 在接受概率高时可达到 1/2 – p 的性能；
- 通过三种实例变换（观察产品、关注价值、公开接受）比较问题的“硬度”，发现没有统一的层级关系。

**⚠️ 局限性**

局限性：
- 需要假设接受随机变量 A_i 独立；若允许相关性，最坏竞争比可能不再为常数；
- 仅给出理论最坏/最优界限，缺乏实验验证或实际数据集评估；
- 对竞争比随接受概率变化的单调性未得到完整结论；
- 只考虑单个接受失败后才继续搜索的设置，未探讨多次尝试或累计收益的扩展。

---

## 755. Cybersecurity Guidance for Smart Homes: A Cross-National Review of Government Sources

**arXiv ID:** 2603.21703 | [PDF](https://arxiv.org/pdf/2603.21703v1)

**作者:** Victor Jüttner `[一作]` (Leipzig University), Erik Buchmann `[通讯]` (Leipzig University)

**通讯引用:** 656 | [OpenAlex ID](https://openalex.org/A5088059746)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在11个国家系统地收集并分析政府针对智能家居用户的网络安全指导，按功能将指导分为事件报告、一般安全建议和事件响应三类。

**💡 创新点**

创新在于首次将政府官方资源以用户角度进行跨国系统性梳理，揭示了普遍缺乏针对智能家居的结构化事件响应指导。

**🔧 技术方法**

采用ChatGPT驱动的用户中心化搜索、手工筛选、归纳式聚类和定性内容分析。

**📊 数据集**

从11个国家官方政府网站共收集101条源，最终纳入35条正式指导文档。

**📈 对比分析**

通过对各类指导的分布和内容进行描述性统计，没有对技术方案进行性能评测，结果显示事件报告渠道普遍可用，但事件响应指导极为稀缺。

**⚠️ 局限性**

局限性包括：检索依赖ChatGPT可能漏检；样本仅为11国，未覆盖全球；未评估指导的实际有效性。

---

## 756. MIND: Multi-agent inference for negotiation dialogue in travel planning

**arXiv ID:** 2603.21696 | [PDF](https://arxiv.org/pdf/2603.21696v1)

**作者:** Hunmin Do `[一作]` (Sungkyunkwan University), Kiyong Jung `[通讯]` (Sungkyunkwan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MIND 框架，用于多智能体协商旅行规划。

**💡 创新点**

创新点在于结合 Theory of Mind 推理与语言语调动态调节，实现对隐含意图的精准推断和策略性沟通。

**🔧 技术方法**

使用 LLM（如 GPT）、ToM 推理模块、任务分解、最大边际相关性筛选、MoSCoW 优先级框架等技术。

**📊 数据集**

基于 TravelPlanner 基准和 Stravl 数据集进行多角色数据增强。

**📈 对比分析**

与传统 MAD 基线比较，MIND 在 High-w Hit、Debate Hit-Rate、Debate Ratio、Fairness、Fidelity 等指标上提升 20-30% 以上，实验表明显著优于基线。

**⚠️ 局限性**

限制包括仅针对旅行规划的情境，依赖预先标注的意愿评分，且在更大规模或不同领域的适用性尚待验证。

---

## 757. Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation

**arXiv ID:** 2603.21782 | [PDF](https://arxiv.org/pdf/2603.21782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 758. Reasoning Provenance for Autonomous AI Agents: Structured Behavioral Analytics Beyond State Checkpoints and Execution Traces

**arXiv ID:** 2603.21692 | [PDF](https://arxiv.org/pdf/2603.21692v1)

**作者:** Neelmani Vispute `[一作]` `[通讯]`, Neelmani Vispute

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Agent Execution Record（AER）原语，记录AI代理在执行过程中的结构化推理信息（意图、观察、推断等），并给出参考实现与SDK；通过在生产的根因分析（RCA）代理上部署，并与LangGraph检查点、Observability平台等现有工具进行对比，展示了AER在结构化、可查询、可跨运行分析层面的优势；同时引入了Mock Replay、证据链、版本化计划等功能，以支持反事实回放和行为分析。

**💡 创新点**

核心创新在于（1）首次将推理过程以规范化、可查询的结构化字段（意图、观察、推断、计划版本等）记录为第一类原语；（2）提出AER与传统检查点/追踪的层次区分；（3）设计了Mock Replay三种回放模式，支持对不同模型/提示版本进行无API调用的回放与比较；（4）引入证据链与委派链等元数据，满足跨运行行为分析需求；（5）提供可扩展的域配置文件和参考实现。

**🔧 技术方法**

技术方面主要使用LangGraph的检查点系统、LangSmith/Datadog等Observability平台进行对照；利用OpenTelemetry GenAI语义规范实现数据导出；采用PROV-AGENT扩展实现因果线索；在Python SDK中实现JSONL/Avro序列化与Kafka流式存储；在生产RCA代理中集成AER记录器并配合LLM（codex-5.3/6.0）进行推理。

**📊 数据集**

实验数据主要来自生产环境中的根因分析（RCA）代理的真实调查记录：约100起完整案例（10步样例、50起Mock Replay实验），其中涉及DBINFRA-1458等实例；此外使用10步的模拟调查进行存储对比；还构建了20起随机案例用于表达式验证。

**📈 对比分析**

比较方法包括：1）与LangGraph检查点对比，发现AER在单调查存储上比累计检查点小约4-22倍（25–130 KB vs 560 KB），且价值随时间提升；2）Mock Replay对比不同LLM版本，测量判决收敛率、证据链重叠率；3）行为分析可在数秒内完成查询（如重计划频率、低置信度误报率），而检查点/追踪需要自定义抽取和NLP处理；总体而言，AER在存储效率、查询即时性和行为分析覆盖面上均优于现有工具。

**⚠️ 局限性**

限制主要包括：① 推理信息是代理自报，可能存在合理化或空洞描述；② 需要代理在提示中显式输出结构化字段，增加提示设计成本；③ 目前实验仍在初步阶段，完整的跨域、多代理行为验证尚未完成；④ 采用JSONL/Avro等存储方式在高并发场景下需要进一步调优；⑤ 代理对AER的准确性验证需进一步开展专业评估。

---

## 759. Strategic Infrastructure Design via Multi-Agent Congestion Games with Joint Placement and Pricing

**arXiv ID:** 2603.21691 | [PDF](https://arxiv.org/pdf/2603.21691v1)

**作者:** Niloofar Aminikalibar `[一作]` (Aston University), Maria Chli `[通讯]` (Aston University)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5024973690)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个联合规划充电站位置与定价的多智能体框架，利用非原子拥堵博弈捕捉EV与非EV司机的自利行为，并将其嵌入双层优化模型中。

**💡 创新点**

① 将站点布置与定价同时优化，首次将非原子拥堵博弈的均衡约束显式化进规划；② 引入ABO‑MPN两层近似方法：行为分解+整数调整，显著降低计算难度；③ 对问题的NP‑hard性给出证明并提出可扩展求解策略。

**🔧 技术方法**

双层非整数非线性规划(bi‑level MINLP)、非原子拥堵博弈、Nash均衡显式化、整数约束松弛与预算保持的四舍五入算法、Pyomo+IPOPT求解器。

**📊 数据集**

使用 Nguyen‑Dupuis (ND) 标准交通网络（13节点、19条边、4 O‑D对）作为实验数据集。

**📈 对比分析**

与单独定价基线、单独布置基线进行对比。ABO‑MPN 在不同预算下，社交成本下降 10%–40%，在高预算 (B=52) 时比基线提升超过 40%。

**⚠️ 局限性**

仅考虑非原子均衡，未包含动态或时变需求、竞争性多供应商情形；高EV渗透率下需要迭代收敛；对更大规模城市网络的可扩展性仍需进一步验证。

---

## 760. AI Token Futures Market: Commoditization of Compute and Derivatives Contract Design

**arXiv ID:** 2603.21690 | [PDF](https://arxiv.org/pdf/2603.21690v1)

**作者:** Yicai Xing `[一作]` `[通讯]` (Independent Researcher), Yicai Xing (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 AI 推理令牌的商品化概念，设计了 Standard Inference Token（SIT）和 Token Price Index（TPI）并构建了基于现金结算的期货合约框架，同时通过蒙特卡洛仿真验证其对企业计算成本波动的对冲效果。

**💡 创新点**

创新点包括：①首次将 AI 令牌视作可交易商品并提出标准化衡量单位 SIT；②设计了基于 TPI 的指数结算机制和保证金制度；③利用均值回归跳跃扩散模型为期货定价提供理论基础。

**🔧 技术方法**

技术手段主要为均值回归跳跃扩散随机过程建模、风险中性定价框架、以及 10,000 条路径的蒙特卡洛仿真；此外还使用了基本的统计分析和期权定价理论来评估对冲效率。

**📊 数据集**

使用的数据来源为公开的 AI 令牌价格时间序列（如 GPT‑4 等主流模型的定价）以及仿真生成的价格路径；未采用传统大规模机器学习数据集。

**📈 对比分析**

对比方法为将未对冲与对冲两种情景下的成本波动率进行蒙特卡洛评估，结果显示在不同需求情境下对冲可将波动率降低 62%–78%，说明期货合约在成本控制方面具有显著效益。

**⚠️ 局限性**

局限性包括：①价格动态模型的假设（均值回归速率、跳跃强度等）可能与实际市场偏离；②缺乏长期真实交易数据导致模型校准与验证有限；③监管框架和市场参与者的接受程度尚未成熟，影响期货市场的可行性。

---

## 761. Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors

**arXiv ID:** 2603.21768 | [PDF](https://arxiv.org/pdf/2603.21768v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 762. Asymmetric Dynamics of Partisan Warriors in YouTube Comments

**arXiv ID:** 2603.21776 | [PDF](https://arxiv.org/pdf/2603.21776v1)

**作者:** Keyeun Lee `[一作]` (Seoul National University), Sang Jung Kim `[通讯]` (University of Iowa)

**通讯引用:** 12248 | [OpenAlex ID](https://openalex.org/A5056333341)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对2024年美国总统二次辩论期间YouTube评论的大规模分析，提出并量化了“党派战士（Partisan Warriors）”这一新概念，探讨了跨党派评论者的攻击目标、受众奖励机制以及环境诱因。

**💡 创新点**

创新点在于：①首次将攻击目标与攻击者的党派立场相结合，定义并识别出“党派战士”；②采用大语言模型（GPT‑4.1）对海量评论进行细粒度攻击目标分类；③系统性比较不同党派和跨党派评论者的毒性水平、受众奖励及其生态驱动，揭示了保守派频道中党派战士的高聚集性和受众奖励机制。

**🔧 技术方法**

使用技术包括：Perspective API 的毒性评分（90th percentile 过滤）、GPT‑4.1 大语言模型的多标签攻击目标分类、基于攻击目标构建的立场评分与党派战士判定；统计方法涵盖 Kruskal‑Wallis、Dunn 检验、负二项回归、Logistic 回归、交叉检验（Chi‑square）以及多层次 GLMM 等。

**📊 数据集**

数据集为 2024 年 9 月 10‑24 期间，围绕美国总统二次辩论的 2,920 条视频（共 1,854,320 条根级评论），来自 160 条左右倾向频道（86 左倾、74 右倾），使用 YouTube Data API 提取，并进一步筛选为活跃且高毒性评论（N≈62,598）以供攻击目标分析。

**📈 对比分析**

与传统单纯毒性评分相比，本文的“党派战士”框架更能捕捉攻击目标与受众奖励的交互。统计结果显示：跨党派评论者与左倾单边评论者在毒性分布上无显著差异（p>0.3）；在保守派频道中，对希拉里攻击的评论点赞数显著提升（IRR≈2.69），而在左倾频道对特朗普攻击无显著奖励；党派战士在保守派频道中的比例显著高于左倾频道（27.3% vs 6.6%，OR≈38.8）。

**⚠️ 局限性**

局限性包括：①样本仅涵盖美国YouTube且仅在总统辩论后短期高关注窗口，可能无法代表日常对话；②使用机器检测毒性与攻击目标存在误报或漏报；③缺乏对评论者动机的直接调查，仅基于行为推断；④研究为相关性分析，无法确定因果关系；⑤数据来源于已被平台删除前的可见评论，极端案例可能被低估。

---

## 763. Deterministic Hallucination Detection in Medical VQA via Confidence-Evidence Bayesian Gain

**arXiv ID:** 2603.21693 | [PDF](https://arxiv.org/pdf/2603.21693v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 764. Image-Conditioned Adaptive Parameter Tuning for Visual Odometry Frontends

**arXiv ID:** 2603.21785 | [PDF](https://arxiv.org/pdf/2603.21785v1)

**作者:** Simone Nascivera `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**通讯引用:** 37496 | [OpenAlex ID](https://openalex.org/A5057116316)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了基于图像条件的强化学习框架，能够在线自适应调节视觉里程计前端的特征检测与跟踪参数。

**💡 创新点**

首创将图像内容直接映射到VO前端参数的RL方法，并利用轻量化纹理感知CNN编码器和特权Critic，使得仿真训练后可零样本迁移至真实环境。

**🔧 技术方法**

采用图像条件强化学习（PPO/SAC）、轻量级CNN编码器、特权Critic、连续动作空间、漂移/覆盖率/计算成本三项奖励设计，并在仿真中对运动模糊和噪声进行渲染增强。

**📊 数据集**

训练使用TartanAirV2仿真数据；评估在TartanAirV2未见序列和TUM RGB‑D SLAM真实数据上进行。

**📈 对比分析**

与粒子群优化（PSO）得到的静态参数设定比较，在合成数据中RL方法实现特征跟踪长度提升约3倍、计算成本降低约3倍；在真实数据中，RL在特征年龄、空间覆盖率和计算时间上显著优于基线。

**⚠️ 局限性**

受限于仿真中精确光流的漂移奖励，真实数据中无法直接评估漂移；未充分验证极端光照或传感器故障下的鲁棒性；仅关注前端参数，未涵盖后端全局优化。

---

## 765. FGIM: a Fast Graph-based Indexes Merging Framework for Approximate Nearest Neighbor Search

**arXiv ID:** 2603.21710 | [PDF](https://arxiv.org/pdf/2603.21710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 766. Time to Get Closer: Longing for Care Ethics Under the Neoliberal Logic of Public Services

**arXiv ID:** 2603.21753 | [PDF](https://arxiv.org/pdf/2603.21753v1)

**作者:** Ruta Serpytyte `[一作]` `[通讯]` (Tampere University), Ruta Serpytyte (Tampere University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过拼贴技术视觉化探讨公共服务中参与式设计与护理伦理在新自由主义背景下的融合与扩展

**💡 创新点**

将多种意识形态与实践经验通过拼贴艺术具体化，提供一种跨学科、感性化的研究方法，揭示参与式设计在公共服务扩展中的政治与伦理张力

**🔧 技术方法**

手工拼贴（collage）及可视化与反思方法

**📊 数据集**

未使用传统数据集，仅引用媒体图像、新闻摘录和个人笔记

**📈 对比分析**

未做实验比较，采用质性对话与自我反思的方式进行研究，无法量化性能

**⚠️ 局限性**

研究主观性强，拼贴素材来源不完整，缺乏可重复性和客观评估；仅聚焦单一城市案例，外推性有限

---

## 767. Interpreted Higher-Dimensional Automata for Concurrent Discrete-Event Control

**arXiv ID:** 2603.21742 | [PDF](https://arxiv.org/pdf/2603.21742v1)

**作者:** Dylan Bellier `[一作]` (Université Paris-Saclay), Philipp Schlehuber-Caissier `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 115 | [OpenAlex ID](https://openalex.org/A5070485120)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文实现了将解释型Petri网（IPN）转换为解释型高维自动机（IHDA）的翻译，并基于IHDA构建并验证闭环控制器；通过并发步骤语义显式建模并发执行，并检测并发导致的输出冲突。

**💡 创新点**

创新点在于将高维自动机的并发细胞与IPN的输入输出条件相结合，首次实现了并发执行步骤的显式建模与冲突检测，使得控制器在并发环境下更可靠。

**🔧 技术方法**

使用的技术包括：IPN、匿名高维自动机（HDA）、并发步骤语义、单元格级别的输出冲突检测、Python实现的Modbus/TCP与OPC UA接口，以及Factory I/O仿真平台。

**📊 数据集**

数据集：在Factory I/O仿真平台中构建的工业搬运车调度场景的仿真数据，用于验证控制器的闭环行为。

**📈 对比分析**

与传统基于顺序Petri网的控制器相比，本文方法在并发冲突检测上更安全；论文通过案例演示展示了可行性，但未给出定量性能指标，只说明在示例中能正确处理并发冲突。

**⚠️ 局限性**

限制：仅采用并发步骤语义，未覆盖更细粒度的非原子事件；对大型系统的可扩展性、计算复杂度和更复杂同步约束的处理尚未深入评估。

---

## 768. Cognitive Agency Surrender: Defending Epistemic Sovereignty via Scaffolded AI Friction

**arXiv ID:** 2603.21735 | [PDF](https://arxiv.org/pdf/2603.21735v1)

**作者:** Kuangzhe Xu `[一作]` (Cyberspace Security University of China), Yinghui Ren `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过零射击语义立场分类对 2023‑2026 年 AI‑HCI 论文进行大规模语义审计，揭示零摩擦陷阱与代理接管趋势，并提出结构化认知摩擦与多模态计算表型方法以恢复人类认知主权。

**💡 创新点**

首创将多代理系统重构为魔鬼代言人式认知强制器，并引入 Gaze Transition Entropy、任务唤醒瞳孔光学与 fNIRS 结合的多模态表型框架，以及基于 HDDM 的数学解耦验证，构建认知摩擦的可度量与治理机制。

**🔧 技术方法**

零射击序列分类（BART‑large‑MNLI）、OpenAlex 基础本体图交集、深度学习 NLI、层级漂移扩散模型（HDDM）、多模态生理信号采集（GTE、瞳孔、fNIRS）以及 Bayesian 适应性调节。

**📊 数据集**

约 8000 篇 2023‑2026 年 AI‑HCI 交叉主题论文，经过 τ=0.7 过滤得到 1223 篇高置信度样本；此外收集对应的实验室多模态生理数据。

**📈 对比分析**

通过对比零摩擦模型、统一共识 MAS 与结构化摩擦 MAS 在决策延迟、预后置信度、系统 2 激活率（fNIRS）和 GTE 等指标上的表现，实验显示结构化摩擦显著降低自动化偏差、提升认知投入，性能提升幅度达 30%–45% 之间。

**⚠️ 局限性**

采用自动化高阈值分类缺乏人工金标验证，样本量受高影响力论文限制；多模态实验局限于实验室环境，缺乏生态有效性；对不同用户群体的个体差异调节机制仍待完善。

---

## 769. Data-Free Layer-Adaptive Merging via Fisher Information for Long-to-Short Reasoning LLMs

**arXiv ID:** 2603.21705 | [PDF](https://arxiv.org/pdf/2603.21705v1)

**作者:** Tian Xia `[一作]` (Tohoku University), Tian Xia `[通讯]` (Tohoku University)

**通讯引用:** 53512 | [OpenAlex ID](https://openalex.org/A5091205267)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于 Fisher 信息矩阵的无校准长链到短链融合方法 FIM‑TIES，用于将长链思考模型与简洁模型融合，以保留推理精度并显著缩短输出长度。

**💡 创新点**

从理论上证明合并误差受层 Hessian 范数影响，并将对角 Fisher 信息矩阵作为该误差的可计算代理，提出完全数据无关的层自适应融合。

**🔧 技术方法**

使用任务算术、TIES 合并、对角 Fisher 信息矩阵计算、对数归一化映射、Sigmoid 参数以及自一致性解码等技术。

**📊 数据集**

在 GSM8K、MATH500、Minerva Math、OlympiadBench、CollegeMath、AIME24 等长链到短链评测基准上进行实验。

**📈 对比分析**

与 Task Arithmetic、TIES‑Merging、AIM、Sens‑Merging、ACM‑TA/TIES 等基线对比，在 7B 规模下 5/6 基准达到了 state‑of‑the‑art，MATH500 提升 6.2 分；在 1.5B 规模下平均准确率提升 3.9 分，响应长度缩短 92.6%，且不需要任何校准数据。

**⚠️ 局限性**

方法主要适用于长链与短链模型的融合，缺乏对更大模型或不同任务域的通用验证；对角 Fisher 近似可能忽略层间耦合；在 AIME24 单推理模式下仍略逊于 ACM‑TIES，需要自一致性进一步提升。

---

## 770. LipsAM: Lipschitz-Continuous Amplitude Modifier for Audio Signal Processing and its Application to Plug-and-Play Dereverberation

**arXiv ID:** 2603.21684 | [PDF](https://arxiv.org/pdf/2603.21684v1)

**作者:** Kazuki Matsumoto `[一作]` (Tokyo University of Agriculture and Technology), Kohei Yatabe `[通讯]` (Tokyo University of Agriculture and Technology)

**通讯引用:** 1973 | [OpenAlex ID](https://openalex.org/A5034837951)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种Lipschitz连续的幅度调制网络（LipsAM）并将其应用于Plug‑and‑Play语音去回声。

**💡 创新点**

创新点在于给出幅度调制网络的Lipschitz连续性充分条件，设计了两种满足该条件的架构，并推导出其Lipschitz常数上界。

**🔧 技术方法**

利用正交卷积实现1‑Lipschitz CNN，结合min与ReLU层保证Lipschitz性，并在PnP + ADMM框架下实现语音重建。

**📊 数据集**

训练使用LibriTTS‑R子集的加噪语音，测试则使用LibriTTS_R与BUT reverb数据库的回声信号。

**📈 对比分析**

与传统幅度调制网络、软阈值等方法比较；在SI‑SNR、PESQ、STOI、ViSQOL指标上，LipsAM（尤其是LipsAM‑RE）在稳定性和效果上均优于传统方法。

**⚠️ 局限性**

局限性：目前仅针对幅度调制型网络，未验证对时频掩码网络的推广，且PnP方法的收敛性理论仍待进一步证明。

---

## 771. RESPOND: Responsive Engagement Strategy for Predictive Orchestration and Dialogue

**arXiv ID:** 2603.21682 | [PDF](https://arxiv.org/pdf/2603.21682v1)

**作者:** Meng-Chen Lee `[一作]` (University of Houston), Andrew D. Wilson `[通讯]` (Microsoft)

**通讯引用:** 13128 | [OpenAlex ID](https://openalex.org/A5078791295)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 RESPOND 框架，实现实时语音对话中的及时背调和协作式占话行为，支持可调节的交互风格；

**💡 创新点**

创新点在于将预测式协同与可调控的两轴参数（背调强度与占话攻击性）结合，实现可定制化的交互策略；

**🔧 技术方法**

采用流式 ASR 与增量语义推理，基于 Qwen3-0.6B 语言模型 + LoRA 微调，再通过 FiLM 层对两轴控制参数进行条件化；

**📊 数据集**

使用 MM‑F2F 与 CANDOR 两大对话语料，分别包含背调、占话与保持沉默标注；

**📈 对比分析**

与多模态与文本基线对比，RESPOND 在 MM‑F2F 上实现 F1≈0.77‑0.78，CANDOR 上总体精度≈0.87，显示轻量化模型仍能与先进方法竞争；

**⚠️ 局限性**

局限在于仅使用文本输入，忽略音调、停顿、视觉信号；占话类别合并，缺乏细粒度；实验规模有限，需进一步验证。

---

## 772. Can a Robot Walk the Robotic Dog: Triple-Zero Collaborative Navigation for Heterogeneous Multi-Agent Systems

**arXiv ID:** 2603.21723 | [PDF](https://arxiv.org/pdf/2603.21723v1)

**作者:** Yaxuan Wang `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5753 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8`

**🎯 论文内容**

提出一种无需训练、无需先验知识、无需仿真的异构多机器人路径规划框架（Triple Zero Path Planning, TZPP），并在真实环境中通过人形机器人与四足机器人协作完成复杂导航任务。

**💡 创新点**

创新点：① Triple Zero（零训练、零先验、零仿真）范式，消除传统方法对数据、仿真和训练成本的依赖；② 协调-探索两层架构，将人形机器人定位为任务协调者，四足机器人负责环境感知与路径探索；③ 自适应 Mode X / Y 机制，针对特征稀疏和障碍密集两类环境自动切换探索策略；④ 采用多模态大语言模型（Doubao‑vision‑3.6）进行自然语言指令理解与视觉-语言融合，无需额外微调。

**🔧 技术方法**

技术实现：多模态大语言模型（Vision‑Language Model）、视觉感知与全景扫描、路径可达性评估、任务分配与协同决策、两阶段路径评估–执行循环、四足机器人旋转扫描与通道探测、离线无仿真实验框架。

**📊 数据集**

数据集：未使用公开数据集，而是基于六个未知的真实物理环境（室内外、无显著标记、障碍密集/稀疏场景）进行实验。每个场景包含多条路径规划任务，采集了16项评价指标。

**📈 对比分析**

比较方法：与人类基线（两名无全局视角的操作员）以及单一人形机器人（G1-only）进行对比。评价维度包括任务完成时间、行驶距离、旋转角度、路径得分、误差、关键点发现率、探索效率、协作效率、重访次数、避障系数等。实验结果显示，TZPP 在绝大多数指标上接近或超过人类表现，尤其在路径效率和协作效率方面达到95%以上的人类水平；Mode X / Y 的 ablation 进一步验证其在不同环境中的必要性。

**⚠️ 局限性**

局限性：① 对极端动态障碍或长距离任务的适应性尚未充分验证；② 在某些复杂障碍或极度特征稀疏场景中仍出现成功率下降或路径效率降低；③ 依赖高性能的视觉语言模型和机器人平台，可能不适用于资源受限环境；④ 研究范围主要集中在两台机器人，尚未扩展到更多异构组合或更大规模群体协作。

---

## 773. FISformer: Replacing Self-Attention with a Fuzzy Inference System in Transformer Models for Time Series Forecasting

**arXiv ID:** 2603.21724 | [PDF](https://arxiv.org/pdf/2603.21724v1)

**作者:** Bulent Haznedar `[一作]`, Levent Karacan `[通讯]` (Gaziantep University)

**通讯引用:** 616 | [OpenAlex ID](https://openalex.org/A5039367341)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了一种在Transformer中用模糊推理系统（FIS）替代自注意力的时序预测模型FISformer。

**💡 创新点**

创新点在于将完整的Sugeno型FIS嵌入到token相互作用中，既捕捉不确定性与非线性关系，又保持可解释性。

**🔧 技术方法**

技术包括可学习的高斯隶属函数、规则生成与去模糊化、基于FIS的交互权重以及Transformer编码器框架。

**📊 数据集**

在多个公开多变量时序数据集（ETT、ECL、Weather、Traffic、Solar‑Energy、PEMS、Exchange）上进行了实验。

**📈 对比分析**

与Informer、Autoformer、FEDformer、Crossformer、iTransformer、FANTF等基线进行对比，FISformer在MSE/MAE指标上普遍优于或接近最强模型，尤其在低维和噪声条件下表现突出。

**⚠️ 局限性**

局限性包括需要手动设定规则数R、对超参数（学习率、MF类型）敏感，以及在极长序列上与自注意力相比仍需进一步优化计算效率。

---

## 774. Let's Think with Images Efficiently! An Interleaved-Modal Chain-of-Thought Reasoning Framework with Dynamic and Precise Visual Thoughts

**arXiv ID:** 2603.21754 | [PDF](https://arxiv.org/pdf/2603.21754v1)

**作者:** Xu Liu `[一作]` (Harbin Institute of Technology), Libo Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 2185 | [OpenAlex ID](https://openalex.org/A5029082837)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为DaP-ICoT的交互式模态链式思维框架，能够在推理过程中动态插入视觉信息并通过精细的视觉引导保持语义连贯；

**💡 创新点**

创新点在于两大模块：动态视觉思维集成（根据模型置信度自适应决定何时插入视觉信息）和精确视觉思维引导（利用对象级分割和跨模态相似度挑选语义一致的视觉片段），从而解决传统ICoT中视觉信息静态定位与破碎化问题；

**🔧 技术方法**

核心技术包括置信度估计（logit margin分析）、阈值控制的动态视觉插入策略、使用SAM2进行对象级分割、跨模态注意力相似度匹配以及多模态交织推理；

**📊 数据集**

在多任务数据集上验证：M^3CoT（多模态多步推理）、ScienceQA（科学问答）、MME（多模态理解）等；

**📈 对比分析**

与直接查询、MMCoT、DDCoT、SCAFFOLD、CCoT、ICoT等基线相比，DaP-ICoT在所有模型（Chameleon-7B、LLaVA-V1.5-7B/13B、Qwen2-VL-2B/7B）下均实现了最高或接近最高的准确率；同时在Token消耗上平均降低72.6%，图像插入次数仅为1.2/样本，显示显著的效率提升；

**⚠️ 局限性**

局限性包括：需要手动调节置信度阈值τ；依赖于对象分割模型的质量；在极端视觉信息不足或过多的场景中仍可能出现插入不足或过度插入的问题；未来需进一步自动化阈值学习及更鲁棒的视觉预处理。

---

## 775. EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning

**arXiv ID:** 2603.21728 | [PDF](https://arxiv.org/pdf/2603.21728v1)

**作者:** Andreas Sauter `[一作]` (Huawei Technologies Co., Ltd.), Yougang Lyu `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EvoIdeator 框架，通过将强化学习与清单式语言反馈相结合，来生成并迭代改进科学研究想法。

**💡 创新点**

创新点在于双信号训练机制：同时使用词典化的 Lexicographic 奖励和 span‑level 可操作反馈，消除训练与推理阶段的错位，使小模型在主观科研指标上超过更大模型。

**🔧 技术方法**

使用技术包括：基于 Dr. GRPO 的强化学习、Lexicographic 奖励设计、LLM 评审模型产生二进制分数与可编辑反馈、结构化清单评估、以及多步骤推理‑评审‑修订循环。

**📊 数据集**

使用数据集：从 OpenAlex 随机抽取 1000 篇 2025 年论文，使用 Llama‑3 生成研究问题与关键词，Semantic Scholar 检索 20 篇相关文献，合成 1–2 段文献综述；测试集 96 个样本。

**📈 对比分析**

与 Qwen‑4B、DeepSeek‑R1、DeepSeek‑V3.2、Gemini‑3 Flash 等基线对比，采用 9 项清单评估（主要为科学严谨性），EvoIdeator 在主要目标上（Grounding、Problem、Risk、Method）均超过更大模型，且训练+推理反馈组合产生可加性提升，表现为接近完美的 Grounding 分数。

**⚠️ 局限性**

局限性包括：Lexicographic 奖励过度偏重核心指标，导致创新与长度等次要指标受限；评估主要依赖 LLM 评审，缺乏大规模人工评判；在跨模型族的反馈语调上仍存在适配困难；以及多步迭代规模与计算分配的进一步研究需求。

---

## 776. CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization for Efficient LLM Fine-Tuning

**arXiv ID:** 2603.21725 | [PDF](https://arxiv.org/pdf/2603.21725v1)

**作者:** Shuo Wang `[一作]` (Southern University of Science and Technology), Ming Tang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 9753 | [OpenAlex ID](https://openalex.org/A5100782002)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于在线曲率跟踪的稀疏零阶优化方法 CurvZO，用于在不使用反向传播的情况下高效微调大型语言模型。

**💡 创新点**

创新点在于：① 用仅有的标量反馈动态估计每个参数（或参数块）的曲率分数；② 依据该分数构造采样分布，优先扰动高曲率参数；③ 结合曲率分布自适应调整扰动预算，实现探索与聚焦平衡；④ 将方法推广到块级实现，显著降低开销。

**🔧 技术方法**

采用零阶梯度估计（SPSA）、Horvitz–Thompson 校正、曲率分数的指数滑动平均、基于 Fisher 信息的采样最优原则、基于熵与有效支持大小的预算自适应、以及块级稀疏扰动。

**📊 数据集**

在 OPT（2.7B、6.7B）和 Llama2（7B、13B）模型上，使用 SuperGLUE 任务集（SST-2、RTE、CB、BoolQ、WSC、WIC、SQuAD、DROP）进行实验。

**📈 对比分析**

与 MeZO、DiZO 以及 LoRA 组合的基准相比，CurvZO 在多数分类与生成任务上平均提升 4–5% 准确率，收敛迭代次数减少约 2–2.4 倍，GPU 小时节省 50–60%，并保持与 MeZO 相当的显存占用。

**⚠️ 局限性**

局限性在于曲率分数仅反映局部敏感度，未捕获参数间的耦合信息，可能导致在某些场景下采样最优性受限。

---

## 777. Structured Visual Narratives Undermine Safety Alignment in Multimodal Large Language Models

**arXiv ID:** 2603.21697 | [PDF](https://arxiv.org/pdf/2603.21697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 778. SHARP: Spectrum-aware Highly-dynamic Adaptation for Resolution Promotion in Remote Sensing Synthesis

**arXiv ID:** 2603.21783 | [PDF](https://arxiv.org/pdf/2603.21783v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 779. Compensating Visual Insufficiency with Stratified Language Guidance for Long-Tail Class Incremental Learning

**arXiv ID:** 2603.21708 | [PDF](https://arxiv.org/pdf/2603.21708v1)

**作者:** Xi Wang `[一作]` (Xidian University), Cheng Deng `[通讯]` (Xidian University)

**通讯引用:** 12659 | [OpenAlex ID](https://openalex.org/A5015874725)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种分层语言树（SL-Tree）和两种并行的语言指导（分层自适应语言指导与分层对齐语言指导），利用大型语言模型生成多尺度、结构化的语义描述来补偿长尾类别在增量学习中的样本稀缺，并通过对齐语言指导缓解灾难性遗忘，最终显著提升长尾类别增量学习（LT‑CIL）的性能。

**💡 创新点**

创新点包括：1）递归 Prompt 设计引导 LLM 生成层级化语言树，提供从粗到细的语义信息；2）引入可学习的层级权重，在预测中融合多尺度语义，使尾类获得更强的监督信号；3）利用语言树的结构稳定性对视觉空间进行对齐约束，形成分层对齐语言指导，进一步降低灾难性遗忘；4）将上述两种指导机制结合，实现比现有方法更优的整体与尾类性能。

**🔧 技术方法**

技术上采用 CLIP 预训练模型和轻量级适配器微调，使用 GPT‑3.5‑turbo 生成文本，构建分层语言树；在自适应层级融合中引入负熵正则和频率先验以平衡不同层权重；对齐语言指导通过视觉-文本相似度对齐损失和知识蒸馏来稳定优化；整体训练框架包含交替更新权重与模型参数的联合损失。

**📊 数据集**

实验使用 CIFAR‑100、ImageNet‑R、CUB‑200 以及 ImageNet‑LT 这四个基准，分别在不同的类别数、任务划分和不平衡率（ρ=0.1、0.01、1）下验证方法的鲁棒性与可扩展性。

**📈 对比分析**

与多种 LT‑CIL 经典方法（LWS、GVAlign、ISPC、DAP、APART）以及基于预训练模型的 CIL 方法（L2P、DualPrompt、CODAPrompt、GMM、RAPF、MG‑CLIP）进行对比；在所有基准上均实现 SOTA，平均提升约 2–4%，尾类准确率提升显著；在 ImageNet‑LT 大规模实验中仍保持领先；并且模型仅训练约 0.41M 参数、训练时间 0.5 小时，展示出优异的高效性。

**⚠️ 局限性**

局限性包括：1）方法对 LLM 生成文本质量高度依赖，LLM 训练成本与 API 调用费用不可忽视；2）生成的语言树基于长尾分布，针对完全平衡数据时效果下降；3）Prompt 设计与层级划分需要人工调优，跨领域迁移可能受限；4）对极端大规模任务的文本生成与存储开销仍有待进一步优化。

---

## 780. PPGL-Swarm: Integrated Multimodal Risk Stratification and Hereditary Syndrome Detection in Pheochromocytoma and Paraganglioma

**arXiv ID:** 2603.21700 | [PDF](https://arxiv.org/pdf/2603.21700v1)

**作者:** Zelin Liu `[一作]` (Shanghai Jiao Tong University), Lichi Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1708 | [OpenAlex ID](https://openalex.org/A5006447298)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一种名为 PPGL‑Swarm 的多代理诊断系统，能够统一处理高分辨率全切片图像、基因突变报告和生化实验结果，并自动生成完整的 GAPP 风险评分报告与遗传性突变风险提示。

**💡 创新点**

创新点包括：① 采用“代理蜂群”架构，将图像分析、基因解读和实验室数据处理拆分为专用子代理，利用强化学习动态调度与融合，提升诊断效率与可解释性；② 引入结构化知识图谱为基因与表格代理提供事实根基，防止模型推理中的幻觉；③ 采用 LAB 色彩归一化+AdaBN 的测试时自适应策略，实现跨机构染色变异与分布漂移的鲁棒推断；④ 对 Ki‑67 与细胞密度进行量化回归，消除主观评估的离散化误差。

**🔧 技术方法**

核心技术包括：多代理系统（Qwen3.5‑35B‑A3B 作为中央决策者）、TransMIL 聚合器与冻结的 UNI‑v2 编码器、基于知识图谱检索的 Gene 与 Table 代理、强化学习（策略梯度 + GAE）优化工具调用序列、测试时自适应（LAB 归一化 + AdaBN）以及多任务损失（交叉熵 + MSE）。

**📊 数据集**

使用 268 名患者的数据集，包含 1,168 张全切片图像、IHC 报告、先天性突变（SDHB、VHL、RET）和生化指标；通过两位专科病理学家评定 GAPP 分数作为标注，并构建参考诊断报告。

**📈 对比分析**

与 GPT‑4o、Claude‑4.5‑Sonnet、LLaVA‑Med、MedDr、SlideChat、TITAN 等基线模型比较，PPGL‑Swarm 在报告质量（整体得分 3.20，最高）以及多任务预测（GAPP MAE 1.2、基因 F1 67.8%）均优于对照组；Ablation 研究表明知识图谱和强化学习贡献最大，缺失时性能明显下降。

**⚠️ 局限性**

局限性包括：① 数据集规模仍有限，模型在更大或更异质的数据上可能面临泛化挑战；② 依赖大模型与知识图谱，若知识图谱不完整或更新不及时可能导致解释缺失；③ 需要额外的彩色归一化与 AdaBN 处理，对部署资源有限的机构仍有一定成本；④ 目前仅针对 PPGL，迁移到其他肿瘤类型需要重新构建知识图谱与子代理。

---

## 781. A Blueprint for Self-Evolving Coding Agents in Vehicle Aerodynamic Drag Prediction

**arXiv ID:** 2603.21698 | [PDF](https://arxiv.org/pdf/2603.21698v1)

**作者:** Jinhui Ren `[一作]` (Baidu AI Cloud), Jianmin Wu `[通讯]` (Baidu AI Cloud)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种契约驱动的自演化编码代理框架，用来发现满足工业约束的拖拽系数(C_d) surrogate 管道；

**💡 创新点**

创新点在于将 surrogate 发现转化为可执行程序的约束优化，结合严格的实验契约、可复现的多种指标（排名、稳定性、成本）以及岛屿迁移与自适应采样等进化策略；

**🔧 技术方法**

采用 Famou‑Agent 样式的 LLM 生成与调优、种群进化（NSGA‑II、MAP‑Elites）、多目标选择、结构化变异（数据、模型、损失、分割策略）以及合同检查与可复现性验证；

**📊 数据集**

使用公开与工业混合的多源 CFD 数据，包含低成本 RANS 与高精度 LES/风洞标注，且所有数据都有版本化、标签完整性与泄漏防护；

**📈 对比分析**

在八种不同 LLM 变异器上对比实验，最佳组合得分 0.9335，符号准确率 0.9180；相较于传统静态模型和单一优化方法，显著提升了排名可靠性与收敛速度；

**⚠️ 局限性**

局限性包括对高质量标注数据的强依赖、对新几何或 Solver 版本漂移的敏感性、无法完全取代高精度 CFD 进行最终验证，以及在极端流动或极大几何变形下的泛化能力仍需进一步验证。

---

## 782. Memory-Efficient Boundary Map for Large-Scale Occupancy Grid Mapping

**arXiv ID:** 2603.21774 | [PDF](https://arxiv.org/pdf/2603.21774v1)

**作者:** Benxu Tang `[一作]` (University of Hong Kong), Fu Zhang `[通讯]` (University of Hong Kong)

**通讯引用:** 157477 | [OpenAlex ID](https://openalex.org/A5100399276)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出边界地图（Boundary Map），只存储环境的边界体素，从而显著降低占用空间；同时给出基于该表示的占用状态判定、查询和增量更新算法；通过全局‑局部映射框架实现实时更新。

**💡 创新点**

创新点包括：①将三维占用空间压缩为二维边界体素；②基于边界体素实现占用状态判断；③使用二维哈希网格存储边界体素并支持二分搜索；④设计全局‑局部映射框架及增量更新策略。

**🔧 技术方法**

技术主要涵盖：体素网格与射线投射；二维哈希表与二分搜索；边界体素提取与更新；全局‑局部滑动映射框架。

**📊 数据集**

使用的数据集为：Ford AV（三条大规模序列）、KITTI（两条大规模序列）、HKU 校园（私有数据）、UAV 飞行（私有数据）。

**📈 对比分析**

与 UG、HG、Octomap、UFOMap、D‑Map 等基准方法对比，内存占用降低 10–100 倍；查询时间与均匀网格相当，更新时间与均匀网格相近；地图准确率>99%。

**⚠️ 局限性**

局限性：局部地图仍采用全密集网格，需额外内存；需手动选择投影轴；在高度动态环境下精度略低于 Octomap；实现相对复杂，需要维护两张地图。

---

## 783. Mapping Travel Experience in Public Transport: Real-Time Evidence and Spatial Analysis in Hamburg

**arXiv ID:** 2603.21763 | [PDF](https://arxiv.org/pdf/2603.21763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 784. Quantifying Uncertainty in FMEDA Safety Metrics: An Error Propagation Approach for Enhanced ASIC Verification

**arXiv ID:** 2603.21770 | [PDF](https://arxiv.org/pdf/2603.21770v1)

**作者:** Antonino Armato `[一作]` (Robert Bosch GmbH), Sebastian Fischer `[通讯]` (Robert Bosch GmbH)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对汽车ASIC的功能安全验证，本文在传统 FMEDA 的单点故障指标（SPFM）和潜在故障指标（LFM）计算中加入误差传播理论，量化输入参数（失效模式分布和诊断覆盖率）的不确定性，生成置信区间，并提出错误重要性识别器（EII）用于定位主要误差来源。

**💡 创新点**

创新点在于首次将误差传播方法系统地引入 FMEDA 计算流程，直接给出 SPFM/LFM 的最大偏差和置信区间，并通过 EII 对不确定性来源进行量化排名，提升了 FMEDA 结果的透明度与可信任度。

**🔧 技术方法**

核心技术包括：误差传播公式推导、对诊断覆盖率和失效模式分布的不确定性建模、统计抽样故障注入方法（确定置信水平和误差界限）以及错误重要性识别器（EII）算法。

**📊 数据集**

使用了典型CPU执行单元（乘除单元与控制逻辑）的失效模式与诊断覆盖率估计数据，结合实验验证的1%误差界限的故障注入结果作为输入数据集。

**📈 对比分析**

通过在CPU核心执行阶段的案例研究验证，与传统仅给出单点数值的 FMEDA 对比，该方法在相同计算量下提供了置信区间并揭示了误差对ASIL合规性的潜在影响，但文中未给出显式的时间或计算性能对比数据。

**⚠️ 局限性**

主要局限包括：误差传播假设输入误差相互独立，若存在相关性需手动调整；对失效模式分布和诊断覆盖率的误差估计依赖专家经验或有限的实验数据；方法主要适用于 ISO 26262 的硬件 FMEDA，尚未针对软件或更复杂体系结构进行验证。

---

## 785. A Curated List of Open-source Software-only Energy Efficiency Measurement Tools: A GitHub Mining Study

**arXiv ID:** 2603.21772 | [PDF](https://arxiv.org/pdf/2603.21772v1)

**作者:** Manuela Bechara Cannizza `[一作]` (Federal University of Technology Paraná), Michel Albonico `[通讯]` (University of Southern Denmark)

**通讯引用:** 118 | [OpenAlex ID](https://openalex.org/A5076690256)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

基于 Mining Software Repositories（MSR）的实证研究，对 GitHub 上公开的能耗与排放测量工具进行系统挖掘与筛选，最终整理出 24 个软件级能耗测量工具的清单并对其演进、粒度、硬件/软件依赖等特征进行定性分析。

**💡 创新点**

首次从 MSR 视角提供开源能耗与排放测量工具的结构化概览，揭示工具从 CPU 级单一监测向多层级、AI/LLM 细粒度与排放估计的演进路径，并系统化阐述硬件/软件依赖对采用率的制约。

**🔧 技术方法**

使用 GitHub API 关键字搜索、手工筛选与双人评估（Cohen κ=0.74）、主题编码与轴向编码分析等定性研究方法；同时结合项目元数据（星数、watcher、语言等）进行趋势与依赖性绘制。

**📊 数据集**

起始数据集为 585 个 GitHub 仓库，经过筛选后保留 24 个与能耗测量相关的开源项目；数据来源包括仓库元信息、README 与代码实现。

**📈 对比分析**

研究主要采用定性比较与趋势分析（年度工具数量、粒度分布、硬件/软件依赖频率），未进行性能数值评测，但通过统计与可视化展示工具演进与差异化特征，体现工具在功能与适用场景上的多样化。

**⚠️ 局限性**

局限性主要在于仅聚焦 GitHub 上公开的开源项目，未覆盖私有或非 GitHub 工具；数据集规模相对较小，缺乏对工具性能的客观评估；关键字搜索可能遗漏不使用预设词的相关工具；硬件/软件依赖分析基于公开信息，可能存在误差。

---

## 786. Getting to the Point: Why Pointing Improves LVLMs

**arXiv ID:** 2603.21746 | [PDF](https://arxiv.org/pdf/2603.21746v1)

**作者:** Simone Alghisi `[一作]` (University of Trento), Giuseppe Riccardi `[通讯]` (University of Trento)

**通讯引用:** 6015 | [OpenAlex ID](https://openalex.org/A5062879885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了点位提示（先定位后计数）在大型视觉语言模型（LVLM）零样本计数任务中的作用，并与直接计数进行对比。

**💡 创新点**

创新点在于证明点位中间步骤能促进模型学习更通用的计数技能、提升OOD泛化，并通过坐标可靠性与机制消融实验阐明了空间信息的重要性。

**🔧 技术方法**

采用了LoRA微调、坐标生成与计数、机制消融、F1/EM/一致性评估等技术，对四种开源LVLM（3B/7B/8B）进行实验。

**📊 数据集**

使用CIVET框架合成的9×9网格图像数据集，训练集中计数1–9，测试集包括ID、OOD（10–18）以及多干扰物（1–9）场景。

**📈 对比分析**

通过与预训练模型和直接计数的对比，发现PtC在OOD场景下准确率提升约50%，坐标F1>89%，但在高干扰场景下一致性下降。

**⚠️ 局限性**

局限性包括计算成本随目标数量增加而显著上升、对真实复杂图像的验证不足、部分模型存在空间偏差、仅评估至8B参数规模。

---

## 787. The Reasoning Error About Reasoning: Why Different Types of Reasoning Require Different Representational Structures

**arXiv ID:** 2603.21736 | [PDF](https://arxiv.org/pdf/2603.21736v1)

**作者:** Yiling Wu `[一作]` `[通讯]` (University of Massachusetts Amherst), Yiling Wu (University of Massachusetts Amherst)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一个框架，将不同类型的推理与表示系统的四个结构属性（可操作性、一致性、结构保持、组合性）对应，并揭示了从因果推理到演绎推理的主要结构边界。

**💡 创新点**

创新点在于识别并定义这四个最小功能性结构属性，说明它们是区分推理类型的必要条件；将该框架与心理学、AI 评测和神经科学证据对齐，提出可检验的三项预测，提供实现无关的推理能力阐释。

**🔧 技术方法**

采用理论分析、跨学科整合方法，并在此基础上设计实验与评测方案；对AI系统的推理过程进行结构性诊断。

**📊 数据集**

使用公开推理基准（LogicBench、DeduCE、以及各类大型语言模型的推理测试），发展心理学实验数据，以及神经影像数据（Goel 2007 等研究）。

**📈 对比分析**

与现有 AI 系统对比发现：统计学习模型在推理链长度增加时性能呈指数衰减；对等逻辑表达的内容敏感度高；推理错误类型与框架预测一致；表明框架能够解释 AI 现象。 在人类数据上则显示推理能力随年龄按结构需求梯度提升。

**⚠️ 局限性**

局限性：仅给出必要条件，未证明足够性；缺乏对具体实现机制的细化；未覆盖混合推理、反事实推理等复杂情形；依赖多源证据的解释性，存在多重解释空间。

---

## 788. Uncertainty Quantification for Distribution-to-Distribution Flow Matching in Scientific Imaging

**arXiv ID:** 2603.21717 | [PDF](https://arxiv.org/pdf/2603.21717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 789. SemEval-2026 Task 12: Abductive Event Reasoning: Towards Real-World Event Causal Inference for Large Language Models

**arXiv ID:** 2603.21720 | [PDF](https://arxiv.org/pdf/2603.21720v1)

**作者:** Pengfei Cao `[一作]` (Chinese Academy of Sciences), Jun Zhao `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 22676 | [OpenAlex ID](https://openalex.org/A5100744623)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在多文档噪声环境下提出直接因果推理任务，构建了AER基准数据集，并在SemEval‑2026 Task12中组织共享任务。

**💡 创新点**

创新点在于：①设计可多答案的直接因果推理任务；②利用LLM+人工审核构造分布式证据与干扰项的真实世界多文档数据；③对多标签准确度采用精确匹配与部分匹配相结合的评估方式。

**🔧 技术方法**

技术包括：大语言模型（GPT‑4、Claude‑3.7‑Sonnet、Gemini‑2.0‑Flash）与提示工程；检索+验证流水线（稀疏检索、稠密检索、图检索、交叉编码器）；监督微调（LoRA、学生模型蒸馏）；知识图谱与神经符号推理。

**📊 数据集**

使用的数据集为AER：60个主题、2831个问答实例，平均每主题约19.7篇文档、28k token，覆盖2016‑2025年新闻报道。

**📈 对比分析**

与零-shot基线模型对比，顶级系统（AILS‑NTUA）在测试集上获得0.95的准确率，远超基线（68%‑70%）。排名靠前的系统普遍采用检索+验证与精细校准策略，表明这些方法对提升多文档因果推理至关重要。

**⚠️ 局限性**

局限性包括：证据噪声高且长期上下文难以处理；多答案评估仍易出现误判；知识图谱增强效果不稳定；模型对抽象因果关系的推理仍有限。

---

## 790. Probing How Scalable Table Data Enhances General Long-Context Reasoning

**arXiv ID:** 2603.21719 | [PDF](https://arxiv.org/pdf/2603.21719v1)

**作者:** Huaibing Xie `[一作]` (Tencent), Pluto Zhou `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究表格数据如何提升大语言模型的长上下文推理能力，并提出可扩展的 TableLong 数据合成管线；

**💡 创新点**

创新点在于：① 用互信息理论揭示表格具有周期性非衰减的结构依赖；② 设计了可扩展、可验证的 TableLong 合成流程；③ 通过 RL 微调实证表格数据显著提升多种长上下文基准性能。

**🔧 技术方法**

主要技术包括互信息分析、表格线性化、SQL 生成与执行、RL 训练（GRPO 等）、一致性过滤、可扩展的数据生成管线。

**📊 数据集**

使用了 10k+ 公开表格数据集（如 BIRD、CoSQL、Spider 等），以及自建的多领域多语言表格集合；通过 LLM 生成 SQL/问答对并执行得到可验证答案。

**📈 对比分析**

与 Gemini‑3、Deepseek‑R1、Qwen 等多模型在 LongBench‑v2、Loong、GSM‑Infinite、LiveCodeBench 等基准上对比；TableLong 使模型在长上下文基准上平均提升约 8%（最高 11%），在 OOD 任务上提升 7–12%。

**⚠️ 局限性**

局限性：仅聚焦表格数据，未系统探究其他数据类型；效果受 RL 训练与可验证性约束；在极长上下文（>128k）上的泛化与效率尚未充分验证。

---

## 791. A Game-Theoretic Framework for Intelligent EV Charging Network Optimisation in Smart Cities

**arXiv ID:** 2603.21715 | [PDF](https://arxiv.org/pdf/2603.21715v1)

**作者:** Niloofar Aminikalibar `[一作]` (Aston University), Maria Chli `[通讯]` (Aston University)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5024973690)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于耦合拥堵博弈的联合充电站布局与定价优化框架，兼顾公共机构视角下的社会成本最小化与站点盈利约束；

**💡 创新点**

创新点在于首次将道路拥堵与充电站排队同时建模为非原子拥堵博弈，并将司机均衡嵌入到双层优化中；

**🔧 技术方法**

采用混合整数非线性规划、两层近似（JPPO-DE）和整数松弛+整数修正技术来求解；

**📊 数据集**

实验数据来源于标准的 Sioux Falls 交通网络（24 个节点、76 条链接）及其基准 O‑D 对；

**📈 对比分析**

与单一参数基线（仅定价或仅布局）比较，JO 框架在最佳预算下相对 PlO 减少 16% 社会成本、相对 PrO 减少 19%，并在网络失效场景下实现高达 32% 成本下降；

**⚠️ 局限性**

局限性包括对均衡假设的依赖、对能源需求均一的简化、以及未考虑动态定价和多运营商竞争等更复杂情境。

---

## 792. Is AI Ready for Multimodal Hate Speech Detection? A Comprehensive Dataset and Benchmark Evaluation

**arXiv ID:** 2603.21686 | [PDF](https://arxiv.org/pdf/2603.21686v1)

**作者:** Rui Xing `[一作]` (Xi’an Jiaotong University), Hao Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 41969 | [OpenAlex ID](https://openalex.org/A5080102032)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出多平台、多语言、多模态表情包仇恨言论数据集M3，并通过七个专门代理构建了可扩展的注释框架；

**💡 创新点**

创新点在于利用代理协同完成分层标签与人类核实的理由生成，并将表情包与帖子上下文结合，实现细粒度、可解释的仇恨分类；

**🔧 技术方法**

采用OCR提取图像文本、LLM自动标注与仲裁一致性检验、专家生成理由、人工验证等技术，构建完整的多模态注释流程；

**📊 数据集**

使用从X、4chan、微博三大平台收集的2455条表情包+帖子数据，标注二级仇恨/正常标签、八类细粒度类别及人工核实的理由；

**📈 对比分析**

在M3上对LLaVA、GLM、Qwen、GPT‑4o、Gemini‑3等多模态大语言模型进行零样本二分类、多标签分类和理由生成评估，二分类准确率普遍>85%，但细粒度分类与理由生成表现不一，加入帖子信息往往导致细粒度性能下降；

**⚠️ 局限性**

当前LLM难以有效整合表情包与帖子上下文，导致细粒度检测下降；跨平台、跨语言泛化有限，缺乏专门的多模态推理机制。

---

## 793. CellFluxRL: Biologically-Constrained Virtual Cell Modeling via Reinforcement Learning

**arXiv ID:** 2603.21743 | [PDF](https://arxiv.org/pdf/2603.21743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 794. When Exploration Comes for Free with Mixture-Greedy: Do we need UCB in Diversity-Aware Multi-Armed Bandits?

**arXiv ID:** 2603.21716 | [PDF](https://arxiv.org/pdf/2603.21716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 795. Mirage The Illusion of Visual Understanding

**arXiv ID:** 2603.21687 | [PDF](https://arxiv.org/pdf/2603.21687v1)

**作者:** Mohammad Asadi `[一作]` (Stanford University), Euan Ashley `[通讯]` (Stanford University)

**通讯引用:** 42330 | [OpenAlex ID](https://openalex.org/A5075711252)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在无图像输入的情况下评估多模态模型，揭示其能够生成逼真图像描述与推理轨迹的“幻影”现象；同时构建并验证了一个文本仅模型（super‑guesser）以及一种基于幻影检测的后处理评估框架B‑Clean。

**💡 创新点**

创新点在于提出并系统验证了幻影效应、发现文本仅模型可在医学视觉问答中击败多模态模型，并开发了B‑Clean方法以消除评测中的非视觉推理偏差。

**🔧 技术方法**

所采用的技术包括：镜像率量化、对比镜像模式与显式猜测模式的评估、基于Qwen‑2.5的文本仅超猜模型微调、以及B‑Clean的幻影检测与问题剔除流程。

**📊 数据集**

使用的数据集涵盖Phantom‑0（幻影检测基准）、VQA‑RAD、MedXpertQA‑MM、MicroVQA、ReXVQA、MMMU‑Pro、Video‑MME、Video‑MMMU等多模态视觉问答与学科理解数据集。

**📈 对比分析**

通过与传统完整图像模式对比，发现多模态模型在无图像的镜像模式下准确率往往高于原始模式；super‑guesser在ReXVQA上超过所有前沿多模态模型和放射科医生；B‑Clean后，基准准确率显著下降且模型排名发生变化。

**⚠️ 局限性**

局限性包括：B‑Clean仅在所选模型集合上有效，无法保证完全剔除所有受幻影影响的问题；评估结果为相对指标而非绝对；幻影机制仍未完全阐明，需进一步机制分析与更广泛的实验验证。

---

## 796. Bridges connecting Encryption Schemes

**arXiv ID:** 2603.21694 | [PDF](https://arxiv.org/pdf/2603.21694v1)

**作者:** Mugurel Barcau `[一作]` (certSIGN Research and Innovation), George C. Turcas `[通讯]` (certSIGN Research and Innovation)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并研究了在加密方案之间的桥接（bridges）概念，定义了桥接的构造、实现及安全性，并给出了基于 Gentry bootstrapping 思路的通用桥接构造方法及其变体。

**💡 创新点**

创新点在于：①将桥接视为加密方案之间的态射，并给出从桥接推导加密方案的通用构造；②证明桥接的 IND‑CPA 安全性可归结为第一个方案的 IND‑CPA 安全性；③给出 Gentry 类型桥接的安全性证明，并提出一种多方案桥接的变体。

**🔧 技术方法**

采用了分类理论视角（将桥接视作态射）、IND‑CPA 安全分析、Gentry 的 bootstrapping 思路、同态评估与环结构映射、以及对桥接关键生成与映射的算法构造。

**📊 数据集**

论文中未使用实际数据集；所有讨论均为理论证明，附录仅提到实现实验，但未给出具体数据集信息。

**📈 对比分析**

论文未给出数值实验或性能对比结果；附录仅说明实现了实验，但未列明性能指标或比较方法。

**⚠️ 局限性**

主要局限包括：①只考虑明文空间不随安全参数变化的情形；②桥接安全性需额外技术假设；③实现时对同态评估的复杂度高，未给出具体效率分析。

---

## 797. The Presupposition Problem in Representation Genesis

**arXiv ID:** 2603.21745 | [PDF](https://arxiv.org/pdf/2603.21745v1)

**作者:** Yiling Wu `[一作]` `[通讯]` (University of Massachusetts Amherst), Yiling Wu (University of Massachusetts Amherst)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

对代表性生成（representation genesis）问题进行结构性诊断，指出主流认知理论在解释第一阶段代表性产生时会出现预设结构，导致解释递归。

**💡 创新点**

提出“Representation Presupposition Thesis”和“Representation Regress”，揭示哲学框架在生成解释中自我预设的结构，并提出需要进行高阶范畴适足性研究的必要性。

**🔧 技术方法**

采用哲学分析与逻辑结构化方法，对语言思维假说、遥控主义、预测处理、激活主义和遗传现象学等五大框架进行案例推演。

**📊 数据集**

无；本文为概念性论文，不涉及具体数据集。

**📈 对比分析**

无；文章不做实验比较或性能评估，而是通过概念论证阐明问题结构。

**⚠️ 局限性**

局限性：未给出具体的生成机制或解决方案；对大型语言模型的实证验证有限；诊断仅聚焦概念层面，未涉及实际认知系统的实验检验。

---

## 798. Rethinking Token Reduction for Large Vision-Language Models

**arXiv ID:** 2603.21701 | [PDF](https://arxiv.org/pdf/2603.21701v1)

**作者:** Yi Wang `[一作]` (Zhejiang University), Xinchao Wang `[通讯]` (National University Of Singapore)

**通讯引用:** 13384 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种面向多轮视觉问答（MT‑VQA）的学习型、无提示依赖的视觉令牌压缩方法（MetaCompress），能够在保持高回答质量的同时显著降低视觉令牌数量。

**💡 创新点**

创新点包括：①将剪枝与合并统一为可学习的压缩映射；②构造基于图像的压缩矩阵生成器，实现对任意分辨率的动态适配；③采用数据驱动、无监督的训练范式，避免使用注意力等启发式指标；④在多轮对话情景下证明了传统注意力导向压缩的缺陷。

**🔧 技术方法**

技术手段：视觉-语言模型融合、视觉令牌压缩矩阵学习、位置编码+下采样投影、正交化的低秩压缩、KL 散度与熵正则化的损失函数、梯度裁剪、基于 SGD 的高效训练。

**📊 数据集**

使用的数据集：MT‑VQA‑v2、MT‑GQA、ConvBench（3‑turn 对话）以及部分视频问答数据用于跨域评估。

**📈 对比分析**

与基准（Base、Random、Sample、FastV、PruMerge）比较，MetaCompress 在 70%–90% 令牌压缩率下保持了更高的多轮回答准确率（平均提升 2–4%），并在推理速度与 GPU 记忆占用上与 Sample 相当，优于 FastV、PruMerge 等方法。

**⚠️ 局限性**

局限性：仅对视觉令牌进行压缩，未涉及 LLM 层级的全局压缩；对极大图像或视频序列的压缩效果尚未彻底验证；训练阶段仍需一定量的数据和计算资源；在某些多尺度视觉塔模型下，压缩率提升空间有限。

---

## 799. Dynamic Exposure Burst Image Restoration

**arXiv ID:** 2603.21784 | [PDF](https://arxiv.org/pdf/2603.21784v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 800. SmaAT-QMix-UNet: A Parameter-Efficient Vector-Quantized UNet for Precipitation Nowcasting

**arXiv ID:** 2603.21879 | [PDF](https://arxiv.org/pdf/2603.21879v1)

**作者:** Nikolas Stavrou `[一作]` (Utrecht University), Siamak Mehrkanoon `[通讯]` (Utrecht University)

**通讯引用:** 1422 | [OpenAlex ID](https://openalex.org/A5076867569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了SmaAT-QMix-UNet，一种将向量量化（VQ）瓶颈和混合卷积（MixConv）结合到SmaAT-UNet中的降水即时报预模型，显著降低参数量并提升预测精度。

**💡 创新点**

创新点：① 在编码器-解码器桥接处插入VQ模块，实现离散化潜在空间；② 仅在最深层级和首个解码层使用MixConv替代深度可分离卷积，兼顾多尺度感受野与参数压缩。

**🔧 技术方法**

使用的技术包括：UNet骨干网络、CBAM注意力模块、深度可分离卷积、向量量化（VQ-VAE）、MixConv、Grad‑CAM 可视化、UMAP 潜在空间嵌入。

**📊 数据集**

实验数据集：荷兰KNMI雷达降水数据（2016–2019），经裁剪、归一化后提取的NL‑50子集（至少50%雨区像素），每样本12帧（60分钟历史）预测30分钟后降水。

**📈 对比分析**

与基准SmaAT‑UNet、Persistence baseline 以及四种网络变体进行对比，评估指标为MSE、Precision、Recall、Accuracy、F1；SmaAT‑QMix‑UNet 在保持或略优于基准的前提下，参数量从4M降至2.5M（↓37.5%），推理时间约缩短6 ms，精度略提升（MSE≈0.0120、Precision≈0.763）。

**⚠️ 局限性**

局限性：VQ正则化可能抑制弱降水区域，导致召回率略低；模型仍需大量雷达输入，未验证在极端天气或不同区域的泛化；混合卷积仅在少数层级使用，整体提升受限。

---

## 801. GoogleTrendArchive: A Year-Long Archive of Real-Time Web Search Trends Worldwide

**arXiv ID:** 2603.21871 | [PDF](https://arxiv.org/pdf/2603.21871v1)

**作者:** Aleksandra Urman `[一作]` (University of Zurich), Joachim Baumann `[通讯]` (Stanford University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一年期全球125个国家、1358个地理位置的Google Trending Now历史档案，包含760万+趋势期数，并提供原始与处理后可直接分析的CSV文件。

**💡 创新点**

创新点在于首次提供超过7天的Trending Now历史数据，利用自动抓取与去重、时间标准化技术，覆盖非英语西方地区，实现多国/多语言趋势的系统化研究。

**🔧 技术方法**

技术手段包括Python Playwright抓取每日Trending Now数据，Python与R脚本进行字段解析、时间统一、缺失值推估、去重合并等数据预处理流程。

**📊 数据集**

数据集基于Google Trending Now每日导出的CSV，时间范围2024‑11‑28至2026‑01‑03，覆盖125国、1358地区，处理后共7,639,695条趋势期记录。

**📈 对比分析**

与Google Trends、Wayback Machine以及仅美国的历史集进行对比，显示本数据在时间深度和地理覆盖度上更优；处理后平均持续时长7.2小时，最大47.8小时，去重后提升计数准确性，缺失估计占14.8%。

**⚠️ 局限性**

限制包括搜索量仅以桶值提供、聚类算法不透明；约14天的技术缺口；部分地区Google市场份额低导致代表性不足；估计持续时间与时间戳反向等质量问题；潜在隐私与信息反馈循环风险。

---

## 802. Directional Mollification for Controlled Smooth Path Generation

**arXiv ID:** 2603.21831 | [PDF](https://arxiv.org/pdf/2603.21831v1)

**作者:** Alfredo González-Calvin `[一作]` (Complutense University of Madrid), Héctor García de Marina `[通讯]` (University of Granada)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种新的方向性卷化（directional mollification）方法，用来将离散路径规划输出（如线性插值的轨迹点序列）平滑成可执行的 C∞ 路径，同时严格保留原始路径上的所有工作点（waypoint）。

**💡 创新点**

创新点在于通过在传统卷化基础上加入方向导数项，突破了卷化只能保持在原路径凸包内且无法通过拐点的限制，使得平滑路径既可精确通过所有工作点，又保持可解析的平滑性与曲率上界；并且进一步构造了一个参数化家族，统一了传统卷化与方向性卷化。

**🔧 技术方法**

主要技术包括：使用标准的光滑卷化算子（mollifier）进行 C∞ 平滑；定义并计算方向导数加权平均的“方向导数项”；证明其在保留平滑性、收敛性、凸性（或凹性）等性质；以及对多段线性插值曲线推导闭式曲率上界。

**📊 数据集**

论文没有使用公开的实验数据集，而是通过解析证明和几何示例（如三点两段曲线、六点五段曲线等）来验证方法的正确性与性能。

**📈 对比分析**

方法与传统卷化、B‑spline、三次/五次 Hermite 样条等传统平滑技术在保持工作点、曲率上界和计算效率上进行比较；实验显示方向性卷化在保证工作点的同时仍可获得可控曲率，且计算量仅为传统卷化的常数倍，满足实时控制需求。

**⚠️ 局限性**

主要局限包括：对非凸/非线性段的多点曲线，方向导数项可能导致局部曲率不再保持全局最优；在某些几何配置下，方向性卷化不再保持凸性或单调性；且对参数 γ 的选取仍需经验性调优，缺乏统一的最优选择准则。

---

## 803. Politics of Questions in News: A Mixed-Methods Study of Interrogative Stances as Markers of Voice and Power

**arXiv ID:** 2603.21823 | [PDF](https://arxiv.org/pdf/2603.21823v1)

**作者:** Bros Victor `[一作]` (Idiap Research Institute), Gatica-Perez Daniel `[通讯]` (Idiap Research Institute)

**通讯引用:** 14194 | [OpenAlex ID](https://openalex.org/A5012965551)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究法语数字新闻中疑问句的使用及其在议程设置、问答互动和个体化过程中的政治作用。

**💡 创新点**

将大规模语言模型伪标注与微调分类器结合，构建多维度疑问句评估指标，首次在跨国法语新闻语料上实现量化与质性三位一体的分析。

**🔧 技术方法**

使用 Qwen3‑30B 伪标注、CamemBERT‑large 进行二分类与六分类，嵌入相似度检索答案、GLiNER NER、BERTopic 主题建模等技术。

**📊 数据集**

覆盖 2023‑2024 年 1.2 M+ 篇法语数字新闻，来源 24 家媒体，涵盖法国、瑞士、比利时、加拿大、塞内加尔等地区。

**📈 对比分析**

与人工标注金标准对照，二分类准确率 0.97、F1 0.78；六分类宏观 F1 0.51；答案检索覆盖率 95.6%，外部答复占 15.4%；在不同国家、规模、话题下呈现系统性差异。

**⚠️ 局限性**

伪标注可能继承教师模型偏差；六分类性能受稀有类型限制；NER 仅捕捉表面提及，难以完整反映责任与声音分配；仅限书面新闻，缺乏语音与互动语境。

---

## 804. Beyond Strict Pairing: Arbitrarily Paired Training for High-Performance Infrared and Visible Image Fusion

**arXiv ID:** 2603.21820 | [PDF](https://arxiv.org/pdf/2603.21820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 805. Anatomical Token Uncertainty for Transformer-Guided Active MRI Acquisition

**arXiv ID:** 2603.21806 | [PDF](https://arxiv.org/pdf/2603.21806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 806. Thermal Topology Collapse: Universal Physical Patch Attacks on Infrared Vision Systems

**arXiv ID:** 2603.21876 | [PDF](https://arxiv.org/pdf/2603.21876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 807. Adaptive Video Distillation: Mitigating Oversaturation and Temporal Collapse in Few-Step Generation

**arXiv ID:** 2603.21864 | [PDF](https://arxiv.org/pdf/2603.21864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 808. Cohesive phase-field fracture with an explicit strength surface: an eigenstrain-based return-mapping formulation

**arXiv ID:** 2603.21811 | [PDF](https://arxiv.org/pdf/2603.21811v1)

**作者:** Tim Hageman `[一作]` (University of Oxford), Tim Hageman `[通讯]` (University of Oxford)

**通讯引用:** 290 | [OpenAlex ID](https://openalex.org/A5034992567)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将裂纹本征应变嵌入局部本构模型的共聚相位场断裂方法，实现了在标准有限元框架下无额外全局自由度即可模拟共聚断裂；

**💡 创新点**

关键创新在于将本征应变的演化转化为类似塑性回退映射的局部过程，成功解耦材料强度与断裂能，并引入平滑与非平滑两种强度判据（包括 Drucker–Prager 形式），同时推导出一致切线算子；

**🔧 技术方法**

采用AT‑2相位场正则化、基于能量的本构与强度势能、回退映射求解、Staggered Newton‑Raphson迭代与Newmark时间积分，并使用FEniCSx+Numba实现；

**📊 数据集**

使用三个经典基准案例（含孔板、单边缺口板、动态缺口板）进行验证，材料参数为E=200 GPa、ν=0.3、f_t=150 MPa、G_c、ℓ可调；不使用公开数据集，所有实验均基于自定义几何与载荷；

**📈 对比分析**

通过与传统相位场模型比较，验证了网格与长度尺度无关、强度表面控制裂纹起始、G_c控制裂纹传播、动态情况下可捕捉裂纹分支；数值收敛良好，计算成本随ℓ减小显著上升；

**⚠️ 局限性**

局限性包括：目前仅实现线性弹性本构，需进一步扩展到大变形或多物理耦合；强度势能需人工指定，可能不足以覆盖复杂多轴材料；在极小ℓ或极低G_c下数值条件敏感；

---

## 809. Cascade-Free Mandarin Visual Speech Recognition via Semantic-Guided Cross-Representation Alignment

**arXiv ID:** 2603.21808 | [PDF](https://arxiv.org/pdf/2603.21808v1)

**作者:** Lei Yang `[一作]` (Shanghai Jiao Tong University), Shilin Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11149 | [OpenAlex ID](https://openalex.org/A5086545796)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无级联的中文视觉语音识别框架，利用多任务学习同时学习音素和视素中间表示，并支持推理时按需激活以平衡精度与效率。

**💡 创新点**

创新点包括消除级联带来的误差累积与延迟、引入语义引导的局部对比损失实现跨表示的时序对齐，以及推理时可按需激活中间表示。

**🔧 技术方法**

采用Conformer网络、3D卷积+预训练ResNet特征提取、DropPath融合、CTC/Attention混合训练、语义对齐局部对比学习和按需激活机制。

**📊 数据集**

仅使用公开的CMLR中文普通话视觉语音识别数据集。

**📈 对比分析**

在CMLR见/未见场景下，视频仅模式下CER分别为20.38%/38.23%，优于所有单/多阶段竞争方法，并且推理延迟最低。

**⚠️ 局限性**

仍存在同音字歧义误差以及对未见说话人泛化的不足。

---

## 810. Riding Brainwaves in LLM Space: Understanding Activation Patterns Using Individual Neural Signatures

**arXiv ID:** 2603.21847 | [PDF](https://arxiv.org/pdf/2603.21847v1)

**作者:** Ajan Subramanian `[一作]`, Rohan Sathish `[通讯]` (Kubo Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练每个个体单独的线性探测器，以将冻结的语言模型隐藏层表示映射到该个体的EEG功率，验证模型中是否存在个体特异性的神经方向。

**💡 创新点**

发现冻结的LLM在深层隐藏空间中保留了稳定、可转移的个体神经信号，并且这些信号与共享的群体信号分离，首次为基于EEG的个体化提供了几何基础。

**🔧 技术方法**

使用线性Ridge回归探测器、PCA降维、Spearman相关评估、负控制（眼动计数）以及跨模型（Qwen 2.5 7B、LLaMA 3.1 8B）验证。

**📊 数据集**

主要使用ZuCo 1.0和ZuCo 2.0语料库（30位参与者、约196k词级EEG样本），并在两个不同架构上重复实验。

**📈 对比分析**

个体化探测器在所有40个EEG特征上均显著优于群体探测器，最高可达ρ≈0.183（高伽马），比群体的ρ≈0.020提升约9倍，且对时间稳定、跨人不可转移等特性均表现良好。

**⚠️ 局限性**

局限包括：探测器仅为线性，可能遗漏非线性结构；数据规模相对有限；EEG信号与眼动相关，未完全独立；未进行因果干预验证；仅使用平均电极信号，缺乏电极级细粒度分析。

---

## 811. Agentic Personas for Adaptive Scientific Explanations with Knowledge Graphs

**arXiv ID:** 2603.21846 | [PDF](https://arxiv.org/pdf/2603.21846v1)

**作者:** Susana Nunes `[一作]` (Universidade de Lisboa), Catia Pesquita `[通讯]` (Universidade de Lisboa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于知识图谱的可适应科学解释框架，利用“代理人格”通过强化学习生成符合专家认知倾向的解释。

**💡 创新点**

创新点在于将专家的认知立场抽象为代理人格，作为奖励信号驱动解释生成，实现了无需大量人工反馈即可实现可适应的解释。

**🔧 技术方法**

使用了强化学习（RL）、从人类反馈中学习的奖励模型、LLM（OpenAI o3-pro和GPT-4o-mini）进行人格生成与评价、Sentence‑BERT进行嵌入、聚类和基准评估方法。

**📊 数据集**

数据集为Hetionet异构生物医学知识图谱，以及11位生物医学专家在药物再利用与药物‑靶点交互任务中对40条解释（共15条反馈）进行评估的专家实验数据。

**📈 对比分析**

与非适应性基线REx及其他KG解释系统（MINERVA、PoLo、RExLight）对比，适应性解释在有效性、相关性、完整性评分均提升，专家偏好率达63–76%，并在药物再利用预测上取得Hits@1 0.358、Hits@3 0.483、MRR 0.438的最佳性能。

**⚠️ 局限性**

局限性包括仅构建了两个人格（其中一个仅基于两名专家）、仅在药物发现领域验证、对LLM生成的认知立场依赖较高，以及在跨领域推广时需要更多专家数据和验证。

---

## 812. Embodying Facts, Figures, and Faiths in Narrative Artistic Performances in Rural Bangladesh

**arXiv ID:** 2603.21830 | [PDF](https://arxiv.org/pdf/2603.21830v1)

**作者:** Sharifa Sultana `[一作]` (University of Illinois Urbana-Champaign), Syed Ishtiaque Ahmed `[通讯]` (University of Toronto)

**通讯引用:** 4070 | [OpenAlex ID](https://openalex.org/A5089574660)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了孟加拉农村社区如何通过传统表演艺术（Puthi、Bhandari Gaan、Pot音乐）传递信息、历史与道德教训。

**💡 创新点**

首次系统阐述了这些表演如何结合事实性、情感性、美学并体现多声道、多伦理框架，以及如何适应技术变迁，为数据叙事提供文化适配设计启示。

**🔧 技术方法**

采用田野调查方法：十个月现场观察、半结构访谈、焦点小组讨论、音频录制、摄影与手记分析。

**📊 数据集**

数据集为约76名受访者的访谈与焦点讨论记录、100+观摩笔记及13小时音频转写。

**📈 对比分析**

无对照实验或性能指标，研究以定性主题分析呈现结果。

**⚠️ 局限性**

样本偏倚、研究者主观解释风险、语言翻译挑战及仅聚焦单地区文化。

---

## 813. Multi-View Deformable Convolution Meets Visual Mamba for Coronary Artery Segmentation

**arXiv ID:** 2603.21829 | [PDF](https://arxiv.org/pdf/2603.21829v1)

**作者:** Xiaochan Yuan `[一作]` (Sichuan Agricultural University), Pai Zeng `[通讯]` (Sichuan Agricultural University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

设计并实现了一种两阶段冠状动脉分割框架 MDSVM-UNet，结合多方向蛇形卷积和残差视觉 Mamba，以提高 3D CTA 图像的分割精度。

**💡 创新点**

主要创新点在于将蛇形卷积扩展到三条解剖平面实现多方向特征融合，使用线性复杂度的视觉 Mamba 捕捉跨切片长程依赖，并引入粗细两阶段分割策略以减少假阳性并提升连通性。

**🔧 技术方法**

采用多方向蛇形卷积（MDSConv）、残差视觉 Mamba（RVM）、UNet++ 栈式密集跳跃连接、Dice 损失以及线性时间状态空间模型等技术。

**📊 数据集**

在 ImageCAS 公共基准（1000 张 3D CTA 图像）上进行训练与评估。

**📈 对比分析**

与多种现有方法（3D U‑Net、TS ConvGRU、GCN、SwinUnet 等）对比，单阶段 MDSVM-UNet 在 DSC 0.686、HD 27.84mm、AHD 0.902mm；两阶段 MDSVM-UNet 在 DSC 0.837、HD 27.84mm、AHD 0.902mm，分别比 ImageCAS 基线提升约 5.4% DSC、8.5mm HD、0.8mm AHD，且参数仅 26.7M。

**⚠️ 局限性**

局限在于对不同扫描协议的泛化性尚待验证、极细支管分割受限于分辨率，以及尚未在临床工作流中进行实测验证。

---

## 814. Individual Rationality in Constrained Hedonic Games: Additively Separable and Fractional Preferences

**arXiv ID:** 2603.21826 | [PDF](https://arxiv.org/pdf/2603.21826v1)

**作者:** Foivos Fioravantes `[一作]` (Czech Technical University in Prague), Šimon Schierreich `[通讯]` (AGH University of Krakow)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文研究了在两类受限的和谐游戏（k-hedonic games 与具有大小约束的和谐游戏）中，单个理性（IR）分配的存在性问题，并给出了完整的经典与参数化复杂度图景。

**💡 创新点**

创新点包括：① 首次将单个理性概念与两类约束结合，揭示其在常规和谐游戏中被忽视的非平凡复杂度；② 引入 N‑fold ILP 技术解决带有顶点覆盖参数与权值上界的 FPT 算法；③ 通过图结构参数（顶点覆盖数、树深度、树宽度、最大度）得到一系列完整的难易阈值；④ 对比现有核心稳定性等更强稳定概念的结果，证明 IR 在约束下与核心稳定性等价。

**🔧 技术方法**

主要技术手段包括：参数化归约（Equitable Partition、Balanced Bin Packing、General Factors）、构造多重图与颜色化技巧、N‑fold 整数规划、动态规划与树分解、树宽/树深度分析以及图着色与可满足性约束。

**📊 数据集**

该工作为纯理论分析，无使用实验数据集；所有结果均通过证明与构造给出。

**📈 对比分析**

通过对比已知的核心稳定性与 Nash/个体稳定性复杂度，本文展示了 IR 在受限情形下既可在 FPT 区间内高效求解（例如顶点覆盖数+权值上界、二进制权值+树宽度+最大度），也在许多参数组合下呈现 NP‑硬或 #P‑难（如树深度、顶点覆盖数+负边数+k）。

**⚠️ 局限性**

限制与开放问题：① 对于二进制权值、树宽度约束下的大小约束游戏仍缺乏完整的 FPT 方案；② 仍未探索更一般的友好/敌对偏好模型；③ 本研究聚焦于 IR 方案，未讨论其他稳定概念在相同约束下的多维复杂度关系。

---

## 815. Timing In stand-up Comedy: Text, Audio, Laughter, Kinesics (TIC-TALK): Pipeline and Database for the Multimodal Study of Comedic Timing

**arXiv ID:** 2603.21803 | [PDF](https://arxiv.org/pdf/2603.21803v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 816. Ctrl-A: Control-Driven Online Data Augmentation

**arXiv ID:** 2603.21819 | [PDF](https://arxiv.org/pdf/2603.21819v1)

**作者:** Jesper B. Christensen `[一作]` (Danish Fundamental Metrology), Alessandra Manzin `[通讯]` (Istituto Nazionale di Ricerca Metrologica)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为ControlAugment（Ctrl-A）的自动数据增强算法，利用控制理论动态调节每种增强操作的强度分布，以提高图像分类模型的泛化性能。

**💡 创新点**

创新点在于将验证集用于计算相对操作响应曲线，进而以控制环路方式实时更新每个增强操作的参数，从而实现在线、无搜索、无超参数调优的数据增强。

**🔧 技术方法**

核心技术包括控制理论（PID控制思想）、相对操作响应曲线（ROR）、可参数化增强强度分布（U_α(0,Γ)）以及基于验证集的动态更新策略。

**📊 数据集**

使用CIFAR‑10、CIFAR‑100和SVHN‑core三大公开图像分类数据集进行实验，并在这些数据集上训练WideResNet‑28‑10网络。

**📈 对比分析**

与AutoAugment、RandAugment、DeepAA、TrivialAugment等基准方法相比，Ctrl‑A在标准训练设置下与最先进方法相当，在改进后的训练设置下取得显著提升（如CIFAR‑10误差率下降约30%），且计算开销仅比TrivialAugment高约10%。

**⚠️ 局限性**

局限性包括对验证集规模和分布的依赖、对单一网络架构的评估、对操作数N的敏感性以及在大规模数据集（如ImageNet）上的验证不足。

---

## 817. Approximate Butterfly Counting in Sublinear Time

**arXiv ID:** 2603.21816 | [PDF](https://arxiv.org/pdf/2603.21816v1)

**作者:** Chi Luo `[一作]` (Shanghai Jiao Tong University), Kuan Yang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种两层采样的近似蜻蜓计数算法，在二分图的查询模型下估算蝶形子图（2×2 立方体）的数量。

**💡 创新点**

创新点包括：① 引入 heavy‑light 边划分与 guess‑and‑prove 机制，实现在不依赖随机图模型的前提下给出 (1±ε) 近似保证；② 在此基础上实现了子线性时间与查询复杂度。

**🔧 技术方法**

使用的技术主要有：两层采样、small‑degree‑first 采样、wedge 采样、heavy‑light 边细分、guess‑and‑prove 迭代、统一边采样器、基于权重的估计等。

**📊 数据集**

实验使用了 15 个来自 KONECT 的真实二分图数据集，边数在 6×10⁶–6×10⁸，蝶数在 3×10⁷–1×10¹⁴ 之间。

**📈 对比分析**

与两种现有方法（Edge Sparsification 与 Weighted Pair Sampling）对比，查询成本平均降低 85%+、运行时间平均降低 65%+，相对误差保持在 5% 以内；在极稀疏图上仍优于基线，且内存占用最低。

**⚠️ 局限性**

限制：依赖统一边采样器；在极稀疏图中精度可能下降；实现为单线程，未充分利用并行；对查询次数上限的自适应停止功能尚未集成。

---

## 818. Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection

**arXiv ID:** 2603.21853 | [PDF](https://arxiv.org/pdf/2603.21853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 819. Modal Logic for Distributed Trust

**arXiv ID:** 2603.21802 | [PDF](https://arxiv.org/pdf/2603.21802v1)

**作者:** Niels Voorneveld `[一作]` (Cybernetica AS), Peeter Laud `[通讯]` (Cybernetica AS)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套构造性模态逻辑框架，用于在分布式代理系统中对信任与通信进行形式化推理，并通过信任链、转发网络、层级信任与阈值信任等机制，描述了从单代理到公共密钥基础设施（PKI）的多级信任模型。

**💡 创新点**

创新点在于：①将信任与交流分别映射到两个模态（belief 与 communication），并引入分裂公理与意识公理；②利用构造性逻辑保持可追溯性和可证明性；③设计了转发模态与层级信任的组合推理规则；④提出阈值信任与愿望模态（wish modality），实现对不确定信任来源的安全处理；⑤在逻辑中引入密钥所有权与代理转译，方便在 PKI 场景中迁移信任。

**🔧 技术方法**

技术手段包括：构造性模态逻辑（K、Necessity 与分裂公理）；lambda 计算（Fitch 风格）用于证明与程序对应；Kripke 语义为逻辑提供模型；可判定性证明与证明搜索优化；顺序公理与转发网络的形式化；阈值信任的集合逻辑与分离推理。

**📊 数据集**

未使用传统数据集；通过人工构造的 PKI 图（如桥式 CA、域信任列表、层级 PKI 等）作为示例进行理论验证。

**📈 对比分析**

本文未进行实验比较；主要通过逻辑可判定性与证明系统的截断性证明来说明性能，并给出证明搜索的收敛性分析，但未给出时间/空间复杂度数值或与其他方法的量化对比。

**⚠️ 局限性**

局限性：①不考虑隐私保护与负面声明的处理；②未给出概率或风险量化，阈值信任仅在形式上阐述；③逻辑对多态、量词的支持有限；④对大规模代理网络的可扩展性与实际部署未做评估；⑤未讨论如何在动态网络中更新信任关系。

---

## 820. Manifold-Aware Exploration for Reinforcement Learning in Video Generation

**arXiv ID:** 2603.21872 | [PDF](https://arxiv.org/pdf/2603.21872v1)

**作者:** Mingzhe Zheng `[一作]` (HKUST), Harry Yang `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种面向视频生成的稳定强化学习框架SAGE-GRPO，旨在通过微观层面的精确SDE和梯度均衡、宏观层面的双信任域约束，实现对数据流形的自适应探索与对齐

**💡 创新点**

创新点在于引入了对流形感知的SDE（包含对数曲率校正）和梯度规范化，以及周期性移动锚点与步进KL相结合的双信任域策略，解决了传统GRPO在视频域的噪声过度注入和漂移问题

**🔧 技术方法**

主要技术包括流形感知的随机微分方程、梯度归一化、双信任域KL约束、基于VideoAlign的奖励模型以及多步滚动采样的GRPO优化

**📊 数据集**

使用了HunyuanVideo1.5视频数据集，并利用官方VideoAlign奖励模型进行评估

**📈 对比分析**

与DanceGRPO、FlowGRPO和CPS等基线在VQ、MQ、TA、CLIPScore、PickScore等指标上均表现出更高的整体奖励和视觉质量，并在用户研究中获得显著更高的胜率

**⚠️ 局限性**

局限性包括对大规模高分辨率视频的计算开销、对奖励模型过拟合的潜在风险以及在不同视频域（如极端光照或长时序）下的泛化能力待进一步验证

---

## 821. Select, Label, Evaluate: Active Testing in NLP

**arXiv ID:** 2603.21840 | [PDF](https://arxiv.org/pdf/2603.21840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 822. Holistic Scaling Laws for Optimal Mixture-of-Experts Architecture Optimization

**arXiv ID:** 2603.21862 | [PDF](https://arxiv.org/pdf/2603.21862v1)

**作者:** Weilin Wan `[一作]` (Fudan University), Cheng Jin `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套针对Mixture‑of‑Experts（MoE）模型的整体架构优化框架，并在六个不同算力规模下进行数百个实验，推导出全局可应用的 MoE 规模律。

**💡 创新点**

创新点在于：① 用 FLOPs/token、主动参数 N_a 与总参数 N 三维约束来避免仅以 M 评价 MoE 的偏差；② 通过代数约束和隐藏维度的秩保持性质，将原始 16 维搜索空间降至 4 维，并进一步拆解为两阶段 𝒪(n³)+𝒪(n²) 的实验流程；③ 发现“近最优带”随算力扩大而宽松，为工程实践提供量化的灵活性依据。

**🔧 技术方法**

采用代数约束推导、坐标归一化、秩保持代理搜索、两阶段分解实验、功率律拟合与大规模 GPU 集群训练等技术，对 MoE 结构参数进行系统实验和统计分析。

**📊 数据集**

使用规模化的高质量文本、代码、数学和多语言预训练语料库，词表大小 152064，采用 152064 词表进行分词；实验中通过拟合得到的 D（训练 tokens 数）与 C（算力）对应关系。

**📈 对比分析**

通过对比不同 M、N_a、N、d 配置的预训练交叉熵损失，绘制损失曲线并提取 0.1% 以内的近最优带；实验结果表明所推导的规模律在不同算力下都能保持极低的损失偏差，且与传统经验式设计相比显著提升性能。

**⚠️ 局限性**

局限性包括：实验规模仅约 670 个模型，可能影响拟合系数的稳健性；(C,M,D) 规模律基于经验架构拟合，未反复迭代；稀疏度参数固定为 N_e=288、K=8，无法直接推广到其他稀疏配置；仅评估预训练损失，未验证下游任务表现。

---

## 823. Investigating and Comparing Discussion Topics in Multilingual Underground Forums

**arXiv ID:** 2603.21849 | [PDF](https://arxiv.org/pdf/2603.21849v1)

**作者:** Mariella Mischinger `[一作]` (IMDEA Networks Institute), Guillermo Suarez-Tangil `[通讯]` (IMDEA Networks Institute)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了俄语-英语双语地下论坛的主题结构，识别“知识渗透点”和暗语。

**💡 创新点**

结合句子嵌入+HDBSCAN+LDA实现无监督多语言主题聚类，首次系统比较同一论坛内不同语言子社区的知识差异。

**🔧 技术方法**

使用句子Transformer（英语通用+俄语通用+多语言）+HDBSCAN聚类+LDA关键词+机器翻译+余弦相似度+人工验证。

**📊 数据集**

使用18年历史的邀请制俄语/英语双语论坛XIN数据，约1.1M贴文、156k线程、约820k段落。

**📈 对比分析**

通过计算跨语言聚类的LDA关键词余弦相似度（阈值0.35/0.2）划分主题关系，得到539对高度相关主题，跨语言主题重叠率约30%，方法稳定性高（Rand指数≈0.87-0.88）。

**⚠️ 局限性**

仅分析首帖和标题，未覆盖完整讨论；机器翻译可能误译暗语；数据清洗未能完全去除噪声；仅单一论坛样本，缺乏普适性验证。

---

## 824. SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection

**arXiv ID:** 2603.21824 | [PDF](https://arxiv.org/pdf/2603.21824v1)

**作者:** Shuxian Zhao `[一作]` (Southeast University), Zhipeng Gui `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个粗细层文本注释的钢表面缺陷视觉‑语言数据集 SteelDefectX，并构建了包含视觉分类、视觉‑语言匹配、少/零样本识别以及跨域零样本迁移四个基准任务，验证了细粒度文本对模型可解释性、泛化性和迁移能力的提升。

**💡 创新点**

创新点：
• 设计并实现了从类级到样本级的多维度文本注释体系，兼顾视觉特征与工业因果信息；
• 通过 GPT‑4o + 语义过滤 + 人工校正自动生成高质量细粒度描述；
• 构建四任务基准，系统评估文本信息在少样本、零样本及跨域迁移中的实际效果；
• 证明细粒度文本能显著提升跨域迁移性能（最高提升约 12% 以上）。

**🔧 技术方法**

使用技术：
• 视觉‑语言模型 CLIP/Long‑CLIP、EVA‑CLIP、FG‑CLIP 等；
• 视觉分类模型 ResNet‑50/101、MobileNetV3、ShuffleNetV2、ViT‑B/16/32、ViT‑L/14；
• GPT‑4o 自动生成文本，Sentence‑BERT 进行候选筛选；
• 结构化注释管线（候选生成、过滤、补全、人工校验）；
• 训练策略：CLIP‑Adapter、Tip‑Adapter‑F、Contrastive Learning 等。

**📊 数据集**

使用数据集：
• 主要数据集：NEU、GC10、X‑SDD、S3D 四公开钢缺陷数据集，融合后形成 7,778 张图像、25 类的 SteelDefectX；
• 额外扩展：FSC‑20、ESDIs‑SOD 供后续补充；
• 迁移评测数据集：MSD‑Cls（10 种铝表面缺陷）和 CGFSDS‑9（5 种无缝钢管缺陷）。

**📈 对比分析**

比较方法与性能：
• 视觉分类：CNN 在 Acc 与 mAcc 上均远优于 ViT，ResNet‑101 mAcc 91.19%；
• 视觉‑语言匹配：Long‑CLIP‑L/14 在 Acc 93.63%/mAcc 92.56% 与视觉分类相当；
• 少样本识别：Long‑CLIP‑Adapter 在 T0 情况下 1‑shot Acc 约 70%，随样本数提升；细粒度 T3 在少样本下略低，但在迁移任务中优势明显；
• 零样本识别：所有模型表现较低，T1（类级文本）比 T0 高约 3–5%；
• 零样本迁移：T3 训练下在 MSD‑Cls 和 CGFSDS‑9 的 Acc 最高，分别提升至 43.38% 和 40.18%，相比 T0 提升约 13–15%。

**⚠️ 局限性**

限制：
• 数据规模仍有限，难以覆盖所有工业场景的视觉细节；
• 细粒度文本生成虽精确，但对极细微视觉差异仍不能完全表述；
• 当前仅支持图像‑文本对齐，缺乏像素级分割、属性层级标注，导致无法直接用于检测或定位；
• 模型在少样本下对细粒度文本利用不足，表明现有视觉‑语言框架对专业描述的理解能力仍待提升。

---

## 825. Adversarial Camouflage

**arXiv ID:** 2603.21867 | [PDF](https://arxiv.org/pdf/2603.21867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 826. Tacit Knowledge Management with Generative AI: Proposal of the GenAI SECI Model

**arXiv ID:** 2603.21866 | [PDF](https://arxiv.org/pdf/2603.21866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 827. Partial Attention in Deep Reinforcement Learning for Safe Multi-Agent Control

**arXiv ID:** 2603.21810 | [PDF](https://arxiv.org/pdf/2603.21810v1)

**作者:** Turki Bin Mohaya `[一作]` (University of Michigan), Peter Seiler `[通讯]` (University of Michigan)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在多智能体强化学习框架下，提出并实现了仅关注前方车辆和对向车辆的部分注意力机制，以提升高速公路并线任务的安全性和效率，并设计了结合全局与个体目标的混合奖励函数。

**💡 创新点**

创新点包括：①在QMIX中嵌入空间+时间的部分注意力，仅聚焦最关键相邻车辆；②使用多头注意力捕获时序依赖；③综合奖励结构将碰撞、安全、流量、舒适等目标统一衡量。

**🔧 技术方法**

技术栈包括深度强化学习（QMIX）、注意力机制（多头自注意力）、PyTorch实现、SUMO仿真环境与TraCI接口、AdamW优化器和经验回放。

**📊 数据集**

使用SUMO仿真平台生成的高速公路并线场景数据，车辆属性和轨迹由仿真工具自行产生；未使用公开真实驾驶数据集。

**📈 对比分析**

与SUMO的Intelligent Driving Model（IDM）基准进行对比；在平均奖励、平均速度、燃油消耗和碰撞率等指标上表现更佳，尤其是碰撞率显著下降，且训练收敛速度更快。

**⚠️ 局限性**

局限性包括：燃油消耗略高；仅在两车道、单向场景下验证，未扩展到多车道或混合自动/人工驾驶；对V2V/感知模型假设完美，缺乏对传感器噪声的鲁棒性评估；注意力机制在更复杂场景中的可扩展性仍待验证。

---

## 828. Connecting Distributed Ledgers: Surveying Novel Interoperability Solutions in On-chain Finance

**arXiv ID:** 2603.21797 | [PDF](https://arxiv.org/pdf/2603.21797v1)

**作者:** Hasret Ozan Sevim `[一作]` `[通讯]` (University of Camerino), Hasret Ozan Sevim (University of Camerino)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统梳理并比较了LayerZero、Wormhole、Chainlink CCIP、Circle CCTP等交叉链互操作性协议，探讨其对链上金融生态的影响。

**💡 创新点**

创新点在于提出将互操作性指标与链上金融表现相结合的实证框架，并给出可落地的统计模型。

**🔧 技术方法**

采用多链桥交易日志、TVL、交易量、Gas价格等指标，并构建OLS与固定效应回归模型进行分析。

**📊 数据集**

数据集来源于LayerZero Scan、Wormhole Scan、The Graph子图以及Messari的区块链子图，覆盖2023-2024年多条桥的数据。

**📈 对比分析**

通过OLS与固定效应回归检验互操作性与收益、流量的关系，结果显示不同链与协议的收益弹性存在显著差异，整体拟合优度较高。

**⚠️ 局限性**

局限性包括样本仅覆盖三条桥、指标单一、缺乏安全性与治理层面的实证，导致结论在更广泛场景下可能不具普适性。

---

## 829. Why does it fail? Explanation of verification failures

**arXiv ID:** 2603.21788 | [PDF](https://arxiv.org/pdf/2603.21788v1)

**作者:** Lars-Henrik Eriksson `[一作]` `[通讯]` (Uppsala University), Lars-Henrik Eriksson (Uppsala University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

研究了一种利用领域模型与兴趣度量来解释 SAT 求解器给出的验证失败的高层解释方法。

**💡 创新点**

创新点在于引入“兴趣”概念筛选解释，并通过定义依赖构造解释树，实现对失败原因的系统化解释。

**🔧 技术方法**

采用了有序一阶谓词逻辑、SAT 归约、兴趣度量与解释函数算法，以及解释树的递归构造技术。

**📊 数据集**

以铁路信号互锁的一个小型实例为实验数据集，构建了具体的域模型与验证条件。

**📈 对比分析**

与 Nelson 等人提出的 provenance 算法进行比较，实验表明该方法能生成更具方向性且更易理解的解释；但实验规模有限，未给出大规模性能评估。

**⚠️ 局限性**

限制在于仅适用于有限域和有限量化的情况，无法处理无界量化；解释树可能包含冗余分支；对系统整体语义的判断仍需人工参与。

---

## 830. Reasoning or Rhetoric? An Empirical Analysis of Moral Reasoning Explanations in Large Language Models

**arXiv ID:** 2603.21854 | [PDF](https://arxiv.org/pdf/2603.21854v1)

**作者:** Aryan Kasat `[一作]` (TCS AI Practice), Vinija Jain `[通讯]` (Google GenAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估13个大型语言模型在六类道德困境中的推理输出，并使用Kohlberg阶段分类与跨模型一致性分析。

**💡 创新点**

首次在规模化实验中利用Kohlberg阶段诊断框架揭示模型普遍聚焦后阶层、跨困境一致性高、存在“道德脱耦”现象，提出“道德口风”假说。

**🔧 技术方法**

采用LLM‑as‑Judge自动评分管道、Kohlberg阶段判定、交叉模型一致性评估、Jensen‑Shannon距离、方差分析等多重定量方法。

**📊 数据集**

六个经典道德困境（Heinz、trolley、lifeboat、医生、偷食、违约）与三种提示方式（零shot、链式思维、角色扮演），共收集600+条模型回答。

**📈 对比分析**

与人类成年阶段分布做卡方检验和JS距离对比，模型均聚焦阶段5-6（JS平均0.71）；交叉困境ICC>0.90；规模效应统计显著但实质性小；提示无显著差异。

**⚠️ 局限性**

LLM‑as‑Judge可能受同化影响；Kohlberg框架争议；未验证内部机制；样本模型有限；难以推断因果性。

---

## 831. Verify Implementation Equivalence of Large Models

**arXiv ID:** 2603.21851 | [PDF](https://arxiv.org/pdf/2603.21851v1)

**作者:** Qi Zhan `[一作]` (Zhejiang University), Shanping Li `[通讯]` (Zhejiang University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于e-graph的实现等价性验证框架，能够自动推断并验证跨框架模型实现之间的等价关系。

**💡 创新点**

创新点在于不依赖手写重写规则，而是通过执行值推断动态合成规则并在验证过程中逐步应用，解决实现路径分歧和自定义核透明性难题。

**🔧 技术方法**

采用了e-graph、等价饱和、共同图、动态规则合成、SMT验证、约束感知随机测试，并基于TorchDynamo捕获计算图。

**📊 数据集**

评估使用了GPT‑2、Qwen、Llama、Mistral、Phi等主流Transformer模型，参数规模从70M到12B。

**📈 对比分析**

与TTrace和Entangle等基线对比，检测率达10/13，定位精度高于TTrace；单模型验证时间从约11秒到350秒，表现符合大模型验证的可接受成本。

**⚠️ 局限性**

限制在于仅覆盖已执行的路径，数据相关分支无法被探测；规则验证对模糊核依赖随机测试，可能不完全安全；对极大模型（200B+）的可扩展性仍待验证。

---

## 832. On the Axioms of Arboreal Categories

**arXiv ID:** 2603.21841 | [PDF](https://arxiv.org/pdf/2603.21841v1)

**作者:** Tomáš Jakl `[一作]` (Czech Technical University), Luca Reggio `[通讯]` (Universitá degli Studi di Milano)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了树连通性作为新的连通性公理，改进了树状范畴的定义，并证明路径函子是Street纤维

**💡 创新点**

创新点在于引入树连通性替代原有连通性公理，使得包含模态comonad的范畴也满足树状范畴，保持了主要性质并证明路径函子为Street纤维

**🔧 技术方法**

使用了范畴论、comonad、factorisation system、树状范畴、路径函子、Street纤维等技术

**📊 数据集**

无数据集

**📈 对比分析**

无实验比较，理论性结果，无性能评估

**⚠️ 局限性**

限制在于仍未完全描述哪些Street纤维来源于树状范畴，且对某些新范畴的路径函子是否为纤维尚不清楚

---

## 833. Publicly Understandable Electronic Voting: A Non-Cryptographic, End-to-End Verifiable Scheme

**arXiv ID:** 2603.21833 | [PDF](https://arxiv.org/pdf/2603.21833v1)

**作者:** Alon Gat `[一作]` `[通讯]`, Alon Gat

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种基于机械随机数生成和公共公告板的非加密端到端可验证电子投票方案，主张软件自由验证（SFV）以让选民在无需软件的前提下自行核实投票完整性。

**💡 创新点**

创新点包括：① 将SFV概念纳入投票设计，打破对复杂密码学的依赖；② 采用机械随机数生成的伪名与真实候选人配对，嵌入在纸质收据中以实现匿名与可追溯；③ 设计分层（Cluster、Super‑Cluster、Ultra‑Cluster）纸质账本，支持手工加权计数；④ 将风险限定审计（RLA）与纸质收据相结合，提供多层次防护。

**🔧 技术方法**

技术手段包括：机械随机数生成器、可公开查询的公告板、无密码加密的签名（可选）、分层手工计数账本、风险限定审计、纸质收据的物理防伪与碎纸机、可选的人工注入熵与AI协助的抗强迫功能。

**📊 数据集**

未使用传统机器学习或大规模数据集，主要基于理论模型与假设的全国选民规模（如1000万选民、12位数伪名空间）来计算碰撞概率与审计样本大小。

**📈 对比分析**

对比方法主要是与传统加密端到端可验证投票方案（如Benaloh、Bingo Voting）在透明度、可验证性、实现复杂度和抗强迫性上的定性比较；未给出具体数值性能指标，声称计算与验证成本远低于加密方案，且可在选民层面实现快速手工核验。

**⚠️ 局限性**

局限性包括：① 依赖机械随机数的完整性，若被篡改将破坏匿名与可验证性；② 可能出现碰撞导致伪名重复，需合理设定伪名空间与投票次数；③ 选民手工核验受限于收据数量与核验者规模，无法覆盖全部选民；④ 在高强迫环境下，强迫抵抗仅为“最佳努力”而非完全；⑤ 需要精心设计Cluster大小以平衡可验证性与强迫抵抗，且实现成本与物理设备依赖较高。

---

## 834. CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter

**arXiv ID:** 2603.21828 | [PDF](https://arxiv.org/pdf/2603.21828v1)

**作者:** Hanyin Cheng `[一作]` (East China Normal University), Chenjuan Guo `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个轻量级的Correlation-aware Adapter（CoRA）插件，能够在不重新预训练的情况下在多变量时间序列预测中捕获动态、异质和部分相关性，提升TSFMs的预测性能。

**💡 创新点**

创新点包括：①将相关矩阵分解为低秩时间变化和时间不变成分，使用可学习的时间多项式捕获动态相关；②通过异质-部分对比学习同时学习正负相关与局部相关；③实现了仅在训练时增加复杂度，推理阶段仅线性复杂度的高效插件。

**🔧 技术方法**

采用了低秩矩阵分解、可学习时间多项式、SE式通道投影、对比学习以及线性融合等技术构建CoRA模块。

**📊 数据集**

在十个真实世界的多变量时间序列数据集（ETT、Electricity、Traffic、Solar、Weather、AQShunyi、ZafNoo等）上进行实验。

**📈 对比分析**

与最新的TSFMs（如GPT4TS、UniTime、Timer、Moment、Chronos等）以及其他插件（LIFT、C-LoRA）进行对比，CoRA在5%少样本微调和更大数据量下均显著降低MSE/MAE，平均提升约2–5%，并保持推理阶段接近无额外开销。

**⚠️ 局限性**

局限性包括：在极大通道数时仍可能产生轻微的推理延迟；对比学习在训练资源上略高；对非Stationary 关系的泛化能力未在所有场景下充分验证。

---

## 835. Clinical Graph-Mediated Distillation for Unpaired MRI-to-CFI Hypertension Prediction

**arXiv ID:** 2603.21809 | [PDF](https://arxiv.org/pdf/2603.21809v1)

**作者:** Dillan Imans `[一作]` (Sungkyunkwan University), Hyunseung Choo `[通讯]` (Sungkyunkwan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计了一种 Clinical Graph-Mediated Distillation (CGMD) 方法，在缺乏配对 MRI 和视网膜图像的情况下，将脑 MRI 学到的血压相关特征迁移到低成本的视网膜图像，以实现高血压预测。

**💡 创新点**

创新点在于：① 利用共享的结构化临床变量构建跨 Cohort 的临床相似性 kNN 图；② 在图上对 MRI 教师模型的表示进行平滑传播，再通过标签门控在图上为每个视网膜患者生成个性化的 distillation 目标；③ 在学生训练中结合监督损失、先验对齐损失和关系匹配损失，提升跨模态迁移效果。

**🔧 技术方法**

使用技术包括：临床特征向量的余弦距离 kNN 图构建；教师模型为 ResNet-34 的 MRI 多切片网络；学生模型为 ResNet-18 加上临床向量的 MLP；图平滑传播、标签门控、先验对齐损失和关系匹配损失；PyTorch 实现和 5‑fold 交叉验证评估。

**📊 数据集**

数据集为同一三级医疗中心收集的两组互斥人群：脑 MRI（295 名）和视网膜图像（112 名），两组仅通过 16 个共享的结构化临床变量相连，用于二分类高血压预测。

**📈 对比分析**

与多种传统知识蒸馏基线（KD、FitNets、RKD、SimKD、FDDM）以及无图/无先验的对照方法比较，CGMD 在 AUC（0.855）、AUPRC（0.937）、特异度（0.933）等指标上均取得显著提升，表明图驱动的跨 Cohort 蒸馏更为有效。

**⚠️ 局限性**

限制包括：① 仅针对两种结构差异大的模态（MRI 与视网膜）验证，其他模态对迁移效果未知；② 需要足够的共享临床变量作为桥梁，若临床特征缺失或不匹配则效果下降；③ 数据量相对较小，可能影响模型泛化；④ 目前仅处理二分类高血压预测，难以直接扩展到多标签或连续预测任务。

---

## 836. Take the Train: Africa at the Crossroad of Modern AI

**arXiv ID:** 2603.21795 | [PDF](https://arxiv.org/pdf/2603.21795v1)

**作者:** Cédric Manouan `[一作]` (Independent Researcher), João Barros `[通讯]` (Carnegie Mellon University Africa)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

系统性梳理了撒哈拉以南非洲在AI采纳中面临的基础设施、支付、能源、数据治理和人才等四大阻碍，提出了“right enablers”框架，并推出 Africa AI Compute Tracker（ACT）工具，实时映射非洲国内的高性能计算（HPC）与AI工厂；同时给出了三阶段（短期、中期、长期）的混合云/本地AI计算发展路线图。

**💡 创新点**

①首次以公开数据构建并开源非洲AI计算资源的实时可视化追踪器ACT；②提出“right enablers”概念，将基础设施与政策、人才、能源四维度统一为AI可持续落地的关键要素；③将传统AI计算能力评估与规模定律、成本-收益分析相结合，为非洲国家在购买与租用计算资源时提供决策框架。

**🔧 技术方法**

采用政策合成方法、公开数据抓取与整合、空间可视化（地图交互）、成本效益计算（CapEx/OpEx对比）以及AI规模定律（Chinchilla、OpenAI）等技术手段；利用GPU使用量统计与论文引用计数，评估不同架构对AI模型规模的影响。

**📊 数据集**

利用公开的HPC系统清单（如TOP500、各国政府与研究机构发布的超算信息）、GPU型号与性能数据、AI论文中引用的GPU频次、云服务商GPU库存数据、以及非洲各国电力与互联网基础设施统计（Afrobarometer、IEA等），构成ACT的数据基础。

**📈 对比分析**

通过与全球顶级GPU（A100、H100、B200、B300）性能对比、成本-收益表格（购买 vs 租用）以及规模定律预测的模型效率，证明非洲目前AI算力显著落后于全球领先水平；同时展示租用云计算在短期至中期更具成本效益，且对现有技术生态更友好。

**⚠️ 局限性**

①依赖公开数据，未覆盖私有或未公布的本地算力；②仅聚焦NVIDIA GPU生态，未考虑AMD、Intel等平台；③缺乏对伦理、语言与社会文化维度的深入探讨；④ACT目前未实时更新，数据滞后性可能影响决策；⑤模型与成本分析基于假设，真实情况受电价、网络延迟等多因素影响。

---

## 837. Climate Prompting: Generating the Madden-Julian Oscillation using Video Diffusion and Low-Dimensional Conditioning

**arXiv ID:** 2603.21856 | [PDF](https://arxiv.org/pdf/2603.21856v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 838. Convolutions Predictable Offloading to an Accelerator: Formalization and Optimization

**arXiv ID:** 2603.21792 | [PDF](https://arxiv.org/pdf/2603.21792v1)

**作者:** Benjamin Husson `[一作]` (CS Group), Claire Pagetti `[通讯]` (ONERA)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了针对片上内存不足的卷积层加速器的可预测卸载策略的形式化，并通过整数线性规划（ILP）求解最优卸载方案；同时实现了 Python 仿真器用于验证与可视化。

**💡 创新点**

创新点在于：①将卸载过程抽象为策略序列，既考虑了输入/核的分块，又结合加速器的 MAC 计算容量；②构造 ILP 目标函数以最小化总时延，突破传统 Row-by-Row / ZigZag 直方图的限制，获得最高 30% 的速度提升。

**🔧 技术方法**

技术包括：形式化建模（策略、步骤、内存占用）、整数线性规划（CPLEX）、Python 仿真器、补丁（patch）划分与合并策略、对比实验框架。

**📊 数据集**

实验使用 ResNet8 与 LeNet-5 的卷积层（kernel 3×3，stride 1）以及多种输入尺寸（4–12）和加速器计算容量（nb_patches_max_S1 2–10）进行评估。

**📈 对比分析**

通过与 ZigZag、Row-by-Row、S1-baseline 三种启发式策略比较，使用总延迟（加载 + 计算）为评估指标。实验显示 ILP 求解得到的策略在大多数配置下比最优启发式方案提升 10–30% 的速度。

**⚠️ 局限性**

限制包括：①仅针对 S1（全部核预加载）策略，未覆盖更细粒度的核分块卸载；②对内存模型过于抽象，未考虑多级缓存和实际数据布局；③ILP 求解规模受限，无法处理极大网络或多层联合调度；④未考虑 GeMM、FFT、Winograd 等卷积实现。

---

## 839. Charting the Diameter Computation Landscape of Geometric Intersection Graphs in Three Dimensions and Higher

**arXiv ID:** 2603.21790 | [PDF](https://arxiv.org/pdf/2603.21790v1)

**作者:** Timothy M. Chan `[一作]` (University of Illinois at Urbana-Champaign), Da Wei Zheng `[通讯]` (Institute of Science and Technology Austria)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了三维及更高维几何交叉图的直径问题，给出了针对单位立方体在直径为2和3时的真子二次算法，并证明了单位球和高维立方体在直径为3时的子二次时间下界；还给出了非单位立方体、盒子在直径为2时的子二次算法，并探讨了VC维度与算法复杂度的关系。

**💡 创新点**

创新点在于：①首次通过伪直线（pseudoline）与单位立方体交叉图的三步邻域建立深度关联，揭示了VC维度在三维中被常数化；②利用该结论设计出近线性时间的直径-2算法和子二次时间的直径-3算法；③通过结合伪直线的切割技术实现多级范围搜索，突破了传统范围搜索在高维伪直线系统上的限制；④提出了“距阈值VC维度”这一新概念，解释了为啥某些几何图可在子二次时间内解决。

**🔧 技术方法**

技术主要包括：VC维度与Shatter维度分析、伪直线（pseudoline）构造与属性判定、基于范围搜索的多层查询、网格划分与分治、正交范围查询、离散化与离散化的直方图技术、以及从硬性化简得到的下界证明（Orthogonal Vector、3-uniform 6-hyperclique、组合4-克利克假设）。

**📊 数据集**

论文没有使用真实实验数据，而是基于理论构造的几何对象集合（单位立方体、单位球、非单位盒子等）以及随机生成的点集来构造交叉图，用以证明算法时间复杂度与下界。

**📈 对比分析**

与之前的研究相比，本工作在三维单位立方体直径≤3的情况实现了真子二次时间（O(n^2-1/13)），直径≤2的情况实现了近线性时间（O(n·polylog n)），而在单位球、四维立方体等情形下给出了Ω(n^2)的下界，显示了不同几何对象在子二次时间可行性上的根本差异。

**⚠️ 局限性**

局限性：仅对直径≤3的小直径场景给出有效算法；对更高直径、一般高维几何对象、以及非单位球/盒子在高维的情况仍缺乏子二次时间解法；VC维度上界虽有限，但在更大半径邻域可能仍无界，导致算法难以扩展；实验验证缺失，实用性与鲁棒性待进一步探讨。

---

## 840. P^2O: Joint Policy and Prompt Optimization

**arXiv ID:** 2603.21877 | [PDF](https://arxiv.org/pdf/2603.21877v1)

**作者:** Xinyu Lu `[一作]` (Chinese Academy of Sciences), Le Sun `[通讯]` (Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种联合策略与提示优化框架 P^2O，利用进化提示来突破强化学习中稀疏奖励导致的探索瓶颈，并通过上下文蒸馏将提示引导的推理能力内化到模型参数中；

**💡 创新点**

创新点在于把提示视为可优化的离散潜在变量，通过 GEPA 进化提示获取成功轨迹，再用上下文蒸馏将其知识迁移至原始输入，形成自我改进的闭环；

**🔧 技术方法**

核心技术包括：强化学习与可验证奖励（RLVR）+ 组相对策略优化（GRPO）；基于遗传算法的提示优化 GEPA；上下文蒸馏机制；以及动态硬样本挖掘与提示分配；

**📊 数据集**

使用了 DeepMath-5K 与 DeepScaler-5K 两个 5,000 条样本的数学推理数据集，评测覆盖 AIME、AMC、MATH500、Minerva Olympiad 等六大基准；

**📈 对比分析**

与 Qwen3-4B 基线及 GRPO 对比，P^2O 在大部分任务上平均提升 4.7%–6.9%，在最难的 AIME24、AIME25 上提升 12% 以上，表明方法显著提升推理性能；

**⚠️ 局限性**

局限性包括：依赖遗传提示搜索的计算成本；对提示来源（自我反射 vs. 教师反射）的敏感性；在极端稀疏奖励场景下仍可能出现梯度消失；以及需要手工设定硬样本阈值和提示分配策略。

---

## 841. BadminSense: Enabling Fine-Grained Badminton Stroke Evaluation on a Single Smartwatch

**arXiv ID:** 2603.21825 | [PDF](https://arxiv.org/pdf/2603.21825v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 842. All elementary functions from a single binary operator

**arXiv ID:** 2603.21852 | [PDF](https://arxiv.org/pdf/2603.21852v1)

**作者:** Andrzej Odrzywołek `[一作]` `[通讯]` (Jagiellonian University), Andrzej Odrzywołek (Jagiellonian University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究者通过系统穷举搜索和数值检验，发现单一二元运算符 EML（(x,y)=exp(x)-ln(y)）与常数 1 可合成所有科学计算器中常见的基本算子、常数和三角、指数、对数等全部元素。

**💡 创新点**

创新点在于证明存在一个连续的 Sheffer 运算符，将传统上需要多种不同算子（+、-、×、/、exp、ln 等）的完整算术体系压缩为单一二元操作，形成统一的二叉树结构，并为符号回归、神经网络解释和硬件实现提供全新的基元。

**🔧 技术方法**

技术方法包括：①系统的穷举式搜索与混合数值/符号验证；②使用自定义的 SymbolicRegression 包实现自举（bootstrapping）求解；③在 Rust 里实现加速验证；④构建 EML 编译器将任意表达式转化为 EML 树；⑤利用梯度优化（Adam）在符号回归中恢复闭式表达式。

**📊 数据集**

数据集主要为特殊数值点（如欧拉常数 γ ≈0.577216、Glaisher‑Kinkelin 常数 A ≈1.28243 等）以及随机生成的表达式，用来验证候选公式的数值一致性；并在实验中对一系列标准函数（π、e、i、√x、sin x、cos x 等）进行数值测试。

**📈 对比分析**

比较方法：将 EML 生成的表达式长度（叶子节点数）与传统手工或最短表达式进行对比；在深度 1–4 的树中，EML 能以较短的节点数实现 e^x、ln x、乘法、三角函数等；在符号回归实验中，梯度优化在 100% 的深度 2 实验中恢复正确公式，深度 3–4 成功率约 25%，深度 5 以下极低，但当成功时误差可降至机器精度级别。

**⚠️ 局限性**

局限性：①EML 需要在复数域内部计算，导致对实数域的边界点（如负实数、零）存在分支跳跃问题；②实现上仍需额外常数 1，无法完全实现无常数的单元；③未发现更简洁或更实用的同类算子；④对更高深度树的梯度搜索成功率低，需改进优化策略；⑤在硬件实现中，需处理复数运算的复杂性。

---

## 843. On the Number of Conditional Independence Tests in Constraint-based Causal Discovery

**arXiv ID:** 2603.21844 | [PDF](https://arxiv.org/pdf/2603.21844v1)

**作者:** Marc Franquesa Monés `[一作]` (Eric and Wendy Schmidt Center, Broad Institute of MIT and Harvard), Caroline Uhler `[通讯]` (Eric and Wendy Schmidt Center, Broad Institute of MIT and Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于约束的因果结构学习算法，能够在观测数据下高效地确定因果图的必然结构。

**💡 创新点**

该算法通过引入前缀节点集和专门的条件独立性检验，证明在所需的CI测试数量上实现了指数层面的最优（上界p^O(s)，下界2^Ω(s)）。

**🔧 技术方法**

主要技术包括前缀节点集扩展、基于v-结构和Meek规则1的祖先关系学习，以及针对最大无向团大小的停止准则；理论上给出复杂度证明并在实验中验证。

**📊 数据集**

实验使用了随机Erdős–Rényi图、线性高斯结构方程模型、SERGIO模拟的基因表达数据以及Airfoil物理实验数据。

**📈 对比分析**

与PC、GES、GSP等传统方法比较，所提算法在运行时间、CI测试次数和结构准确性（SHD）上均表现更优，尤其在高密度图和真实基因数据上优势明显。

**⚠️ 局限性**

局限性包括对无限样本的假设、正确性条件尚未完全阐明、以及在有限样本下CI检验阈值选择对性能的影响。

---

## 844. The Universal Normal Embedding

**arXiv ID:** 2603.21786 | [PDF](https://arxiv.org/pdf/2603.21786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 845. STENet: Superpixel Token Enhancing Network for RGB-D Salient Object Detection

**arXiv ID:** 2603.21999 | [PDF](https://arxiv.org/pdf/2603.21999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 846. Instruction Set and Language for Symbolic Regression

**arXiv ID:** 2603.21836 | [PDF](https://arxiv.org/pdf/2603.21836v1)

**作者:** Ezequiel Lopez-Rubio `[一作]`, Mario Pascual-Gonzalez `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 IsalSR 框架，将符号回归表达式的有向无环图（DAG）编码为指令字符串，并通过可裁剪规范化得到完整的 DAG 不变式，从而消除 Θ(k!) 的结构冗余。

**💡 创新点**

创新点包括：双层指令字母表和可交换编码（将减法、除法拆解为加法与 Neg/Inv），以及基于标签感知的剪枝回溯算法，可在保持语义完整的前提下生成唯一规范字符串。

**🔧 技术方法**

使用的技术包括：S2D（字符串到 DAG 的解码）、D2S（DAG 到字符串的贪心编码）、Pruned Canonical String（剪枝规范化）以及基于 BFS 的 6‑元组邻域剪枝和 Levenshtein 距离度量。

**📊 数据集**

实验数据集为 Nguyen 12 个单/双变量基准以及 AI Feynman 子集 10 条物理方程，用于验证 P1–P5 的五项属性。

**📈 对比分析**

与传统贪心编码比较，pruned 版在 99.97% 样本上得到相同规范串，长度仅比贪心略短（≈21%）；搜索空间分析表明完整 Θ(k!) 组合被压缩为单一字符串；Levenshtein 距离实验显示该距离能体现结构差异，且距离 1 对应单字符操作的语义局部变换。

**⚠️ 局限性**

主要限制包括：缺乏对两大猜想的正式证明；可裁剪算法的时间复杂度尚未给出完整量化；框架仅支持固定操作符集，无法直接处理条件表达式或多输出 DAG。

---

## 847. Benchmarking Recurrent Event-Based Object Detection for Industrial Multi-Class Recognition on MTEvent

**arXiv ID:** 2603.21787 | [PDF](https://arxiv.org/pdf/2603.21787v1)

**作者:** Lokeshwaran Manohar `[一作]` (Technical University of Dortmund), Moritz Roidl `[通讯]` (Technical University of Dortmund)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了循环式事件摄像头检测在工业多类场景的表现，比较了ReYOLOv8s与YOLOv8s在MTEvent上的性能。

**💡 创新点**

通过系统性基准实验，揭示了时序上下文、预训练来源及剪辑长度对检测性能的影响。

**🔧 技术方法**

使用ReYOLOv8s（带ConvLSTM）、YOLOv8s、事件域预训练（GEN1、PEDRo）以及标准训练与微调。

**📊 数据集**

采用工业场景事件摄像头数据集MTEvent，包含17类物体与人类。

**📈 对比分析**

对比非循环YOLOv8s基线，循环模型从0.260 mAP50提升至0.285（+9.6%），最佳预训练模型达到0.329 mAP50，说明时序上下文和源域对齐的预训练效果显著。

**⚠️ 局限性**

局限在类不平衡严重、仅使用单侧事件流、缺乏更广泛基准与变换实验，导致结果受样本数和域差异影响。

---

## 848. Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention

**arXiv ID:** 2603.21957 | [PDF](https://arxiv.org/pdf/2603.21957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 849. Deriving Health Metrics from the Photoplethysmogram: Benchmarks and Insights from MIMIC-III-Ext-PPG

**arXiv ID:** 2603.21832 | [PDF](https://arxiv.org/pdf/2603.21832v1)

**作者:** Mohammad Moulaeifard `[一作]` (Oldenburg University), Nils Strodthoff `[通讯]` (Oldenburg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究构建了一个统一的PPG基准框架，对多任务（心律分类与生理参数回归）进行系统评估，提供了大型ICU PPG数据集上的性能基线。

**💡 创新点**

创新点在于首次实现多分类心律（13类）与回归任务的统一评估，并对不同血压、心率、性别、BMI、族裔等子组进行细粒度性能分析，揭示了PPG在不同临床情景下的优势与局限。

**🔧 技术方法**

使用深度学习模型XResNet1d101（深层）和LeNet1D（轻量）进行特征提取和预测，结合Butterworth滤波、AdamW优化等标准深度学习流程。

**📊 数据集**

基于公开的MIMIC‑III Waveform Database Matched Subset（约630万条30秒PPG段，6189名ICU患者），并在外部Liu等数据集进行跨域验证。

**📈 对比分析**

与两种模型进行对比，AF检测AUROC达到0.96/0.94，HR MAE 1.13/1.61 bpm，RR MAE 2.97/4.53 bpm，SBP/DBP MAE 16.13/8.70 mmHg；外部验证保持高性能（AF AUROC 0.97）。

**⚠️ 局限性**

主要限制包括仅在ICU患者和30秒段数据上评估，缺乏运动干扰和多模态融合，导致对常规健康或可穿戴场景的可推广性不足。

---

## 850. StreamSampling.jl: Efficient Sampling from Data Streams in Julia

**arXiv ID:** 2603.21996 | [PDF](https://arxiv.org/pdf/2603.21996v1)

**作者:** Adriano Meligrana `[一作]` `[通讯]` (Sapienza University of Rome), Adriano Meligrana (Sapienza University of Rome)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文实现了Julia库StreamSampling.jl，提供单遍高效的数据流采样，包括reservoir和sequential两大范式，支持有/无替换、加权/不加权，且不需提前知道流长度。

**💡 创新点**

创新点在于：①统一实现两大采样范式及其变体；②通过小常数内存（O(K)）完成单遍采样；③提供并行合并接口；④提供便捷API，可直接对任何Julia迭代器操作；⑤在真实磁盘Arrow文件上演示out‑of‑core采样。

**🔧 技术方法**

使用技术主要有：Reservoir Sampling、Sequential Skip Sampling、带权重的skip/expJ、并行合并（statistically consistent reservoir）以及Julia标准迭代器协议。

**📊 数据集**

数据集包括：①模拟的10^8个整数序列；②100GB的加权Tabular Arrow文件（约3.1×10^9行）。

**📈 对比分析**

与传统的基于完整 materialization 的 population 采样、以及需要两遍的 sequential 方案比较，Benchmarks 显示：在小样本（≤10%）下，Reservoir/Sampling 方法在时间和内存上均明显优于基线；单线程下 Reservoir 更快，2线程时 sequential 的并行扩展更好。

**⚠️ 局限性**

局限性包括：①缺少加权无替换的 sequential 算法；②Sequential 需要预先知道 N 或 W_N，若未知需两遍；③未实现 Bernoulli/Poisson、滑动窗口或分层 Reservoir；④对极大样本时仍可能遇到内存瓶颈。

---

## 851. Look, Listen and Segment: Towards Weakly Supervised Audio-visual Semantic Segmentation

**arXiv ID:** 2603.21948 | [PDF](https://arxiv.org/pdf/2603.21948v1)

**作者:** Chengzhi Li `[一作]`, Yanghao Zhou `[通讯]` (Beijing Institute Of Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出弱监督音视频语义分割（WSAVSS）任务，并设计 PCAS 方法实现逐帧音频理解和像素级语义映射。

**💡 创新点**

创新点包括：1) 将任务拆分为“看-先-听-先-分割”三步；2) Temporal Visual Prompting（TVP）利用视觉先验强化音频语义；3) Progressive Contrastive Alignment（PCAS）通过实例级和标记级对比学习实现无掩码的细粒度对齐。

**🔧 技术方法**

使用 ViT 视觉编码器、AST 等音频编码器；对比学习（Cross‑Modal Contrast、Cross‑Modal Patch/ Class Token Contrast）；CAM‑based 伪标签、Dense‑CRF 后处理。

**📊 数据集**

在 AVS‑S4、AVS‑MS3（弱监督基准）以及 AVS‑Semantic（全监督基准）数据集上评估；训练时使用视频级类别标签。

**📈 对比分析**

与现有弱监督基线（WSAVS、CAM、EZ‑VSL 等）相比，PCAS 在 AVS‑S4 与 AVS‑MS3 上分别提升 F‑score/MIOU 至 74.2/60.5（ViT‑base）和 68.5/56.4（ResNet‑50）；在 AVS‑Semantic 上与部分全监督模型（PVT‑v2、AVSegFormer 等）接近，表现出较强的竞争力。

**⚠️ 局限性**

局限性：1) AVS‑Semantic 上仍缺少专门的弱监督基准，评估对比有限；2) 仅依赖视频级标签，无法捕捉更细粒度的音频事件；3) 对极端噪声或多源混合音频的鲁棒性尚未充分验证。

---

## 852. Disengagement Analysis and Field Tests of a Prototypical Open-Source Level 4 Autonomous Driving System

**arXiv ID:** 2603.21926 | [PDF](https://arxiv.org/pdf/2603.21926v1)

**作者:** Marvin Seegert `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在一条固定的西欧郊区路线上，对配备Autoware Universe的Level 4研究平台进行了236 km的混合交通实地测试，并记录了26次行驶中的30次非计划性干预（disengagements）。

**💡 创新点**

创新点在于提出并应用了五级危害性框架（安全故障、预防性、违规、操作死锁、误判干预），实现对软件限制与驾驶员过度谨慎干预的区分，并给出了开源堆栈的基准离线干预率。

**🔧 技术方法**

技术实现包括对Autoware Universe v0.41.1的功能扩展（最高速度、动态限速、LiDAR‑only 目标检测、绿箭交通灯识别）、多模传感器融合、根因分析与手工注释、以及空间与时间离线干预率计算。

**📊 数据集**

数据来源为现场收集的26次行驶日志、传感器流与车辆状态信息；测试覆盖城市、乡村与高速公路段，共计13个交通灯、1个非受保护左转、2个环岛，未使用公开数据集。

**📈 对比分析**

将离线干预率0.127 km⁻¹与商业cadmv报告（≈2×10⁻⁴–3次/英里）以及仿真基准进行对比，发现开源堆栈的干预率显著偏高；具体根因表明感知缺陷占40%、规划缺陷占26.7%，说明系统在真实交通中仍存在较大安全风险。

**⚠️ 局限性**

局限性包括：仅做后期记录分析，缺乏驾驶员即时反馈；危害性等级主观性高；固定路线与有限的交通/环境多样性，导致外部效度受限；以及对遮挡、静止障碍与系统信心传达的规划与感知模块改进空间。

---

## 853. Parameter-Efficient Fine-Tuning for Medical Text Summarization: A Comparative Study of Lora, Prompt Tuning, and Full Fine-Tuning

**arXiv ID:** 2603.21970 | [PDF](https://arxiv.org/pdf/2603.21970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 854. Collision-Free Velocity Scheduling for Multi-Agent Systems on Predefined Routes via Inexact-Projection ADMM

**arXiv ID:** 2603.21913 | [PDF](https://arxiv.org/pdf/2603.21913v1)

**作者:** Seungyeop Lee `[一作]` (Inha University), Jong-Han Kim `[通讯]` (Inha University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在预定路线约束下的多智能体协同运动，提出通过优化航点通过时间实现速度调度来避免碰撞的框架。

**💡 创新点**

创新点在于引入可微分的航点时间映射模型，将非凸碰撞约束转化为基于时间网格的距离惩罚，并使用无整数序列的近似投影ADMM进行求解。

**🔧 技术方法**

采用了软化轨迹近似、距离惩罚函数、近似投影ADMM与自适应动量梯度修正、Monte Carlo 评估等技术。

**📊 数据集**

实验数据来自随机生成的交叉、瓶颈和基于 Delaunay 三角剖分的图网络情景，构造的随机航点集合。

**📈 对比分析**

与基于 MIP–SOCP 的层级时间调度基线进行比较，结果表明在中等至高密度情景下该方法更可行且总完成时间更短。

**⚠️ 局限性**

局限性包括对高保真动力学的鲁棒性不足、极高拥堵或几何不可行时仍可能失败、缺乏在线重规划能力以及理论收敛性证明不足。

---

## 855. Self-Heating and Parasitic Effects in Multi-Tier CFET Design

**arXiv ID:** 2603.21910 | [PDF](https://arxiv.org/pdf/2603.21910v1)

**作者:** Sufia Shahin `[一作]` (Technical University of Munich), Hussam Amrouch `[通讯]` (Technical University of Munich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对比2层和4层CFET架构，评估自热效应和BEOL/MOL寄生电阻电容对器件和电路性能的影响。

**💡 创新点**

首次系统地量化多层CFET在自热与BEOL寄生电容方面的相互作用，并提出对比方法和性能指标。

**🔧 技术方法**

采用3D TCAD仿真、热动力学模型、Synopsys Raphael工具以及实验测量校准的器件参数。

**📊 数据集**

使用技术论文中公开的实验测量数据对TCAD模型进行校准，并以这些实验数据为基准评估性能。

**📈 对比分析**

通过对ΔTMAX、ION衰减、延迟时间等指标进行定量比较，发现4层CFET的自热峰值高出约25%，BEOL寄生电阻增幅10倍，导致传播延迟平均提升约4.8%。

**⚠️ 局限性**

多层堆叠显著增加了自热和寄生电阻，导致热耦合和延迟退化，需要更高效的热管理和电路设计来保证可靠性。

---

## 856. MultiBind: A Benchmark for Attribute Misbinding in Multi-Subject Generation

**arXiv ID:** 2603.21937 | [PDF](https://arxiv.org/pdf/2603.21937v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 857. Guideline-grounded retrieval-augmented generation for ophthalmic clinical decision support

**arXiv ID:** 2603.21925 | [PDF](https://arxiv.org/pdf/2603.21925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 858. SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation

**arXiv ID:** 2603.21904 | [PDF](https://arxiv.org/pdf/2603.21904v1)

**作者:** Linkuan Zhou `[一作]` (Northwestern Polytechnical University), Qiangguo Jin `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 SHAPE 框架，用于医学图像分割的无监督域适应，包含层级特征调制、超图可行性评估和结构异常修剪。

**💡 创新点**

创新点：①类感知分层特征调制 (HFM)；②基于超图的全局解剖合理性评估 (HPE)；③跨视角结构异常修剪 (SAP)；三者协同提升伪标签质量。

**🔧 技术方法**

技术：DINOv3 ViT 预训练、AdaIN、混合策略、超图构建、JSD 统计、Dice+Focal 损失、EMA 教师模型等。

**📊 数据集**

数据集：MMWHS 心脏 CT/MRI、MICCAI 2015 腹部 CT、ISBI 2019 CHAOS MRI。

**📈 对比分析**

与多种对齐和伪标签方法（CycleGAN、AdaptSegNet、ADVENT、SIFA、SASAN、GenericSSL、UPL-SFDA、IPLC、DDFP 等）比较，SHAPE 在心脏 MRI→CT、CT→MRI 平均 Dice 分别提升至 90.08%/78.51%，腹部 MRI→CT、CT→MRI 则达到 87.48%/86.89%，显著优于现有方法。

**⚠️ 局限性**

局限性：依赖预训练模型与复杂三阶段管线，计算成本较高；对极端域差仍存在一定误差；需要足够的目标域样本以估计超图统计。

---

## 859. You See It, They Don't: An Exploratory Study of User-to-User Variation in Instagram Comments

**arXiv ID:** 2603.21953 | [PDF](https://arxiv.org/pdf/2603.21953v1)

**作者:** Brahmani Nutakki `[一作]` (Saarland University), Ingmar Weber `[通讯]` (Saarland University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Instagram上进行小规模的袜子假账户实验，探测新AI驱动的评论排序系统是否导致不同用户看到不同的评论。

**💡 创新点**

首次系统性检验Instagram评论排名的个性化影响，并发现评论差异主要由账户规模而非用户属性决定，提供了评估社交媒体评论排序偏差的新框架。

**🔧 技术方法**

采用Selenium爬虫+VPN代理收集评论，使用贝叶斯β-二项混合模型和逻辑回归分析差异，利用OpenAI API对评论进行支持/反对/中立标签。

**📊 数据集**

使用200条来自20个Instagram账户（10个新闻、10个非新闻）的帖子，八次抓取（四个假账户×两个VPN地点），共约1900条可见评论。

**📈 对比分析**

通过计算爬取对之间的评论差异比例、热图、以及回归预测来比较不同用户/地点/属性的差异；平均差异仅约12%，新闻帖子差异更低，账户的评论数/粉丝数对差异影响更显著。

**⚠️ 局限性**

限制包括仅用四个假账户、仅关注可见评论而不考虑排名顺序、仅收集已饱和的老帖子、缺乏大规模样本以及算法对新用户的了解不足。

---

## 860. GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction

**arXiv ID:** 2603.21943 | [PDF](https://arxiv.org/pdf/2603.21943v1)

**作者:** Ayesh Abu Lehyeh `[一作]` (University of Vermont), Safwan Wshah `[通讯]` (University of Vermont)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了轻量化的 GeoFlow 框架，用概率分布预测从任意初始位姿到目标位姿的距离与方向，并通过迭代细化采样（IRS）多重假设收敛，完成精细跨视图定位。

**💡 创新点**

创新点在于：①以流匹配为灵感的连续位姿回归字段，直接输出概率分布而非离散分类；②结合 IRS 的多假设迭代细化，既提升精度又实现推理时间可伸缩；③通过 NLL 损失同时学习距离的高斯分布与方向的 von Mises 分布，提供不确定性量化。

**🔧 技术方法**

采用 EfficientNet‑B0 作为地面与卫星图像特征提取器，使用交叉注意力融合特征；多层感知机回归距离与方向；利用 von Mises 方向分布与高斯距离分布的 NLL 损失；IRS 通过对 N 个随机假设做 R 步迭代更新，最终取平均得到位姿。

**📊 数据集**

在 VIGOR（城市级全景与空中图）和 KITTI（前视地面图与空中图）两个公开数据集上进行实验。

**📈 对比分析**

与 GGCVT、CCVPE、HC‑Net、DenseFlow、FG^2 等 SOTA 方法对比，GeoFlow 在 KITTI 同区/跨区均保持 0.98–8.42 m 平均误差，且实现 29.49 FPS（比 HC‑Net 快 1.2×，比 FG^2 快 8–14×），参数量 7.38 M，显存占用仅 686 MiB，显著提升了效率与实时性。

**⚠️ 局限性**

局限性包括：仅处理平面 2‑DoF 定位；对姿态假设需已知或通过 IMU；在极端光照/遮挡下性能下降；模型对大尺度搜索区域的扩展仍需验证。

---

## 861. Deep Reinforcement Learning and The Tale of Two Temporal Difference Errors

**arXiv ID:** 2603.21921 | [PDF](https://arxiv.org/pdf/2603.21921v1)

**作者:** Juan Sebastian Rojas `[一作]` (University of Toronto), Chi-Guhn Lee `[通讯]` (University of Toronto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了强化学习中TD误差的两种解释在深度强化学习中的差异，并评估了这一差异对算法性能的影响

**💡 创新点**

首次正式引入隐式TD误差概念，证明在深度RL中显式误差会导致平均奖励估计不稳定，从而提出隐式误差在训练中的优势

**🔧 技术方法**

使用深度Q网络（DQN）、差分Q学习、奖励中心化、A2C等深度RL算法，并采用梯度下降/Adam优化器进行训练

**📊 数据集**

实验数据集包括倒立摆、Atari Breakout、Pong、MuJoCo HalfCheetah 等经典强化学习环境

**📈 对比分析**

通过将显式与隐式误差更新在上述任务中对比，发现隐式误差在平均奖励估计、总奖励和收敛速度上更为稳定且整体表现略优或相当

**⚠️ 局限性**

研究仅给出了误差差异的存在性和对某些算法的影响，未对深度RL中误差差距进行精确量化，且结果受网络结构、激活函数、批量大小和优化器等因素的影响

---

## 862. A Latent Representation Learning Framework for Hyperspectral Image Emulation in Remote Sensing

**arXiv ID:** 2603.21911 | [PDF](https://arxiv.org/pdf/2603.21911v1)

**作者:** Chedly Ben Azizi `[一作]` (University Littoral Cote D'Opale), Matthieu Puigt `[通讯]` (University Littoral Cote D'Opale)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于变分自编码器的潜在表征学习框架，用于生成高质量的超光谱图像，可按像素或完整空间-光谱立方体进行训练。

**💡 创新点**

创新点在于将超光谱仿真任务转化为参数到潜在空间的生成建模，支持一阶直接映射与二阶预训练+插值两种训练策略，并实现了空间相关的卷积变分模型。

**🔧 技术方法**

采用变分自编码器（VAE）与卷积网络，结合参数到潜在空间的插值器，以及可选的MLP、KRR和GPR回归基线。

**📊 数据集**

使用PROSAIL模拟的植被数据集和实际Sentinel‑3 OLCI海色影像，分别进行模拟与真实数据的评估。

**📈 对比分析**

通过RMSE、SSIM、PSNR、Spectral Angle以及CPU/GPU推理吞吐量对比，像素级VAE在模拟数据上精度最高；卷积VAE‑pre在真实数据上显著优于基线，整体性能优于传统回归模型。

**⚠️ 局限性**

局限性包括未对模型不确定性进行量化，二阶预训练在像素级模型收益有限，且在不同数据域间迁移性能仍需进一步提升。

---

## 863. SparseDVFS: Sparse-Aware DVFS for Energy-Efficient Edge Inference

**arXiv ID:** 2603.21908 | [PDF](https://arxiv.org/pdf/2603.21908v1)

**作者:** Ziyang Zhang `[一作]` (Politecnico di Milano), Luca Mottola `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在边缘设备上针对深度神经网络推理设计了一种稀疏感知的细粒度 DVFS 框架 SparseDVFS，能够根据算子稀疏度动态调节 CPU、GPU 与内存的频率，实现能耗显著下降；

**💡 创新点**

主要创新点包括：① 基于白盒时序分析的稀疏度到频率映射离线模型；② 运行时贪心图划分器，将算子聚合成稀疏感知的超块以降低频率切换开销；③ 统一协同治理器 FUSE 以及预取调度，消除 CPU‑GPU 内存互斥并隐藏切换延迟；

**🔧 技术方法**

使用技术包括：白盒时序性能模型、热感知功耗模型、贪心超块合并算法、FUSE（频率统一调度）以及管线式预取调度；

**📊 数据集**

实验数据集涵盖 ImageNet-2012 验证集（用于稀疏分布分析）以及 ResNet-18/101、ViT-B16/L16 四个模型；在 ablation 研究中还使用了 DOTA‑v1.0 与 VisDrone 2019；

**📈 对比分析**

与四种基线（默认 DVFS、nvpmodel、GearDVFS、Ascend‑DVFS）对比，SparseDVFS 在能耗上平均提升 78.17%（比默认 DVFS 低 78.17%），成本-收益比 14%，在保持相近推理时延（比 MAX‑N 仅高 12.8%）的同时，功耗平均下降至 7–10 W；

**⚠️ 局限性**

局限性包括：稀疏度仅以标量比值表示，未考虑稀疏模式（结构化 vs 非结构化）；缺乏实时内存访问瓶颈感知；离线模型需要针对每个硬件平台重新调优，缺乏跨平台迁移能力；以及在极高频率或极低稀疏度场景下的切换频率可能失效。

---

## 864. ADaFuSE: Adaptive Diffusion-generated Image and Text Fusion for Interactive Text-to-Image Retrieval

**arXiv ID:** 2603.21886 | [PDF](https://arxiv.org/pdf/2603.21886v1)

**作者:** Zhuocheng Zhang `[一作]` (Hunan University), Zijun Long `[通讯]` (Hunan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级的自适应融合模型ADaFuSE，用于改进基于扩散模型的交互式文本到图像检索（I‑TIR）；

**💡 创新点**

在传统的静态加权融合基础上引入自适应门控机制动态调节图像与文本信息的权重，并结合语义感知的混合专家（MoE）分支捕捉跨模态细粒度关系，以降低扩散生成图像噪声对检索的负面影响；

**🔧 技术方法**

采用扩散模型生成图像、BLIP视觉‑文本编码器、GELU投影、Sigmoid门控、语义感知MoE、多路专家网络、InfoNCE 对比损失以及残差连接等技术；

**📊 数据集**

在DA‑VisDial数据集上训练，评估使用VisDial验证集以及三组ChatIR out‑of‑distribution 数据集（ChatGPT_BLIP2、Human_BLIP2、Flan‑Alpaca‑XXL_BLIP2）；

**📈 对比分析**

与仅文本检索的ChatIR以及基于静态加权融合的DAR进行对比，ADaFuSE在Hits@10指标上较DAR提升最高达3.49%，显著降低了检索退化率和平均排名下跌；

**⚠️ 局限性**

仍受限于预训练编码器对长对话的适配能力、对极端噪声图像的处理仍有提升空间，以及MoE分支的额外计算开销；

---

## 865. Optimal Solutions for the Moving Target Vehicle Routing Problem with Obstacles via Lazy Branch and Price

**arXiv ID:** 2603.21880 | [PDF](https://arxiv.org/pdf/2603.21880v1)

**作者:** Anoop Bhat `[一作]` (Carnegie Mellon University), Howie Choset `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并实现一种用于多目标车辆路径规划（含障碍）的最优算法——Lazy BPRC；该算法在分支定界+列生成框架下，通过延迟计算巡回路径成本，求解移动目标路由问题。

**💡 创新点**

创新点主要在于：①采用松弛连续性约束快速给出巡回路径的下界，随后按需精确计算真实成本；②将延迟评估与列生成相结合，显著减少昂贵的运动规划调用；③在图形凸集（GCS）搜索中加入连续性松弛启发式，提高启发式搜索质量；④对定价问题设计了新的标签优胜规则，兼顾速度、容量、时间窗与障碍。

**🔧 技术方法**

技术手段包括：分支定界+列生成（Branch‑and‑Price）框架；松弛连续性约束的成本下界计算；图形凸集（Graph of Convex Sets）用于精确求解最短路径；标签扩展与优胜判定的动态规划；多目标/多代理的时间窗约束与容量约束建模。

**📊 数据集**

实验使用自定义的合成实例：每个目标有两个时间窗、需求为1，速度随机在0.5–1 m/s之间，3个代理（v_max = 4 m/s），障碍图为方格地图，变量包括目标数、地图分辨率和容量。

**📈 对比分析**

与两种消融实验比较：Non‑Lazy BPRC（每生成标签即评估成本）和 No‑Affine‑Heuristic（使用 FMC* 启发式）。实验表明 Lazy BPRC 在目标数、地图分辨率、容量变化下，最小/中位/最大运行时间均优于两者；在最佳情况可比非懒惰版快44倍，平均可比 No‑Affine‑Heuristic 快13倍。

**⚠️ 局限性**

局限性：目前算法仅适用于中等规模实例；对大规模、多目标或高分辨率地图的可扩展性仍有限；未来工作计划是设计边界子最优解以进一步提升规模。

---

## 866. Surfacing and Applying Meaning: Supporting Hermeneutical Autonomy for LGBTQ+ People in Taiwan

**arXiv ID:** 2603.21990 | [PDF](https://arxiv.org/pdf/2603.21990v1)

**作者:** Yi-Tong Chen `[一作]` (National Taiwan University), Nitesh Goyal `[通讯]` (Google Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对台湾 LGBTQ+ 群体在社交媒体中面临的解释学不公（hermeneutical injustice）进行多阶段研究，并基于研究发现设计、实现、评估了一个检索增强型 LLM 聊天机器人——Queerbot，旨在支持用户的身份探索、情感验证和社区参与。

**💡 创新点**

① 将解释学不公的理论框架与实际技术实现结合，提出“解释学自治”（hermeneutical autonomy）的系统设计目标；② 设计四种交互模式（获取、自主、讨论、盟友），以满足不同情境下的资源获取与应用需求；③ 通过社区共创方式构建可被 LLM 调用的“解释学资源语料库”，实现社区驱动的知识治理。

**🔧 技术方法**

使用检索增强生成（Retrieval‑Augmented Generation, RAG）技术：在 Gemini 2.5 Flash LLM 之上加入向量检索，结合用户知识向量、社区资源向量和情境上下文；实现透明可解释的用户知识管理（自然语言表达的知识条目与置信度）；实现多模式对话驱动。系统实现基于 Next.js + LangChain.js + Supabase。

**📊 数据集**

① 由研究参与者提供的个人社交媒体内容与自述（用于用户知识与场景模拟）；② 社区共创的解释学资源（包含网站、社区帖子、书籍、影片、个人经验摘要等）；③ 公开可用的台湾 LGBTQ+ 讨论帖与中立信息，作为检索语料。未使用公开标准数据集，而是构建了以社区为核心的自定义知识库。

**📈 对比分析**

本研究主要采用定性方法（访谈、工作坊、评估工作坊）评估系统的有效性。未进行量化对比实验，评价指标为参与者对身份认同、情感支持、资源获取和表达策略的主观感受。根据用户反馈，Queerbot 在提供验证、资源推荐和情境化论证方面获得积极评价；但未给出客观性能分数。

**⚠️ 局限性**

① LLM 本身的文化偏见与幻觉风险，可能导致生成不准确或有害信息；② 依赖社区共创资源，若社区治理不严，易出现不当或攻击性内容；③ 受限于检索库规模与更新速度，可能无法覆盖所有最新讨论；④ 系统未在大规模真实社交媒体场景中持续测试，缺乏长期使用和安全性评估；⑤ 研究样本规模有限，主要为学生和部分工作者，可能不具代表性。

---

## 867. TREX: Trajectory Explanations for Multi-Objective Reinforcement Learning

**arXiv ID:** 2603.21988 | [PDF](https://arxiv.org/pdf/2603.21988v1)

**作者:** Dilina Rajapakse `[一作]` (Trinity College Dublin), Ivana Dusparic `[通讯]` (Trinity College Dublin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 TREX 框架，用于后置解释多目标强化学习（MORL）策略，通过生成轨迹、对轨迹进行嵌入、聚类、训练原始和补充策略，并计算奖励归因得分来量化不同行为片段对 Pareto 权衡的影响。

**💡 创新点**

创新点在于：①首个基于轨迹归因的多目标强化学习可解释方法；②通过训练排除特定轨迹簇的补充策略来量化行为的重要性；③将奖励归因与用户偏好向量结合，生成可量化的奖励归因得分（RAS）。

**🔧 技术方法**

使用的技术包括：轨迹采样与子轨迹切分、基于 transformer 的序列编码得到轨迹嵌入、K‑means 聚类、原始策略和补充策略的训练（以 MODT(P) 为例），以及基于相对奖励变化的归因计算。

**📊 数据集**

实验使用 MO‑Gymnasium 的三种 MuJoCo 环境：MO‑HalfCheetah、MO‑Ant、MO‑Swimmer，并使用 PEDA 框架提供的 MODT(P) 作为专家策略和训练补充策略的基础。

**📈 对比分析**

通过与专家策略的累计奖励对比，计算补充策略在不同簇移除时的奖励偏差和 RAS 分数，展示哪些行为簇对速度/能量或两方向距离的影响最大；同时结合轨迹可视化进行定性验证。结果表明 TREX 能准确识别并量化关键行为，对多目标权衡提供解释。

**⚠️ 局限性**

局限性包括：①使用原始策略模拟专家，可能导致解释偏差；②每个簇都需要训练一个补充策略，计算成本高；③对轨迹编码和聚类方法敏感，可能错过细粒度行为；④缺乏与现有可解释方法的系统性对比，实验规模受限。

---

## 868. Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model

**arXiv ID:** 2603.21986 | [PDF](https://arxiv.org/pdf/2603.21986v1)

**作者:** SII-GAIR `[一作]`, Pengfei Liu `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了开源音视频生成基础模型daVinci-MagiHuman，能够根据文本指令同步生成高质量的视频和语音；

**💡 创新点**

采用单流Transformer统一处理文本、视频、音频，省去多流交叉注意力的复杂架构，同时引入无时间步去噪、Per-Head门控、latent空间超分辨、Turbo VAE解码和全图编译等技术，实现高速高效推理；

**🔧 技术方法**

主要技术包括单流Transformer（Sandwich布局）、无时间步去噪、每头门控、统一条件、latent空间超分辨、Turbo VAE解码、全图编译、DMD-2蒸馏、跨语言文本编码与音频同步；

**📊 数据集**

使用VerseBench、VideoScore2、TalkVid-Bench、GLM-ASR等公开评测数据集，并结合中文（普通话、粤语）、英文、日语、韩语、德语、法语等多语言语料进行训练与评估；

**📈 对比分析**

通过自动指标（VideoScore2视觉质量、文本对齐、语音可懂度WER）与人类对比（2000对比）与Ovi 1.1、LTX 2.3进行对比，daVinci‑MagiHuman在视觉质量、文本对齐、WER 14.60%表现最优，且在人类评测中分别以80%和60.9%赢得偏好，5 s 1080p视频仅需38.4 s推理；

**⚠️ 局限性**

仍面临单流模型对多模态细粒度同步的局限、超分辨需额外步骤、对非主流语言支持不足以及推理依赖单GPU的可扩展性待提升等限制。

---

## 869. Demystifying Reinforcement Learning for Long-Horizon Tool-Using Agents: A Comprehensive Recipe

**arXiv ID:** 2603.21972 | [PDF](https://arxiv.org/pdf/2603.21972v1)

**作者:** Xixi Wu `[一作]` (Chinese University of Hong Kong), Hong Cheng `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并系统评估了一套名为 STAR 的后训练管线，旨在通过奖励设计、模型规模、数据组成、算法选择和环境稳定性等维度，训练能够在 TravelPlanner 任务中执行长时序工具调用的语言代理。

**💡 创新点**

创新点在于：①在单一任务上从五个关键维度全面分解 RL 设计空间；②给出基于模型规模的奖励与探索策略的差异化“scale‑aware”实用配方；③识别出训练数据量与难度的最佳甜点区间；④在多维度对比中实现了 TravelPlanner 的 SOTA 性能，且小模型可超越多大模型。

**🔧 技术方法**

主要技术包括：ReAct 框架下的多步工具调用；SFT（监督微调）与 GRPO（强化学习）相结合；多种奖励形式（Sum、Macro、Success、Curriculum）；数据合成与难度控制；对比算法 DAPO、ARPO 等探索策略。

**📊 数据集**

使用了 TravelPlanner 作为测试bed，合成约10K 条带难度控制的查询；SFT 训练 1,198 条高质量轨迹；RL 训练 1K 训练样本，难度比例 4:3:3；并在 1,000 条官方测试集上评估，同时在 7 个知识密集 QA 任务上做 OOD 泛化验证。

**📈 对比分析**

与基线（预训练 Base、SFT 版本）以及不同奖励/算法组合进行对比；实验显示 1.5B–7B 模型在 RL 后可达 60%+ 成功率，超过 7B 预训练模型和商业 LLM；在 OOD QA 上，采用宏观奖励的模型保持了 10–15% 的性能提升，表明泛化能力较好。

**⚠️ 局限性**

主要局限：①仅在模拟环境 TravelPlanner 上验证，缺乏真实 API 的多变性；②OOD 评测仅限知识问答，跨域鲁棒性未知；③实验规模受限于 7B 参数，未探究更大模型的可迁移性；④各维度单独评估，未探究多维度交互效应；⑤奖励设计仍是任务特定，缺乏通用 step‑level 方案。

---

## 870. Camera-Agnostic Pruning of 3D Gaussian Splats via Descriptor-Based Beta Evidence

**arXiv ID:** 2603.21933 | [PDF](https://arxiv.org/pdf/2603.21933v1)

**作者:** Peter Fasogbon `[一作]` (Nokia Technologies), Hamed Rezazadegan Tavakoli `[通讯]` (Nokia Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种完全无摄像头依赖的单次后训练 3D 高斯 splat 剪枝方法。

**💡 创新点**

创新点在于设计了基于邻域的混合描述符 HSFH 并结合 Beta 证据模型，从内部结构信息推断每个 splat 的可靠性，无需视角或渲染监督。

**🔧 技术方法**

采用的技术包括 HSFH 本地描述符提取、Beta 证据推断、统计置信度计算以及基于置信度阈值的单次剪枝。

**📊 数据集**

实验使用 ISO/IEC MPEG CTC 标准测试序列（共 19 个前向动态与对象中心场景）进行评估。

**📈 对比分析**

与 LightGSPrune、ConfSplatPrune 等摄像头依赖基线对比，低至中等剪枝率下保持相近的 PSNR/SSIM，最高 30% 剪枝率时在复杂场景中略优，LPIPS 亦维持较低水平。

**⚠️ 局限性**

局限性包括在对象中心、低冗余场景中剪枝收益有限，且未直接结合后续压缩/量化流程；方法仅在单次后训练阶段有效，未考虑实时动态更新。

---

## 871. A Simple and Efficient Implementation of Strong Call by Need by an Abstract Machine

**arXiv ID:** 2603.21949 | [PDF](https://arxiv.org/pdf/2603.21949v1)

**作者:** Małgorzata Biernacka `[一作]` (University of Wrocław), Tomasz Drab `[通讯]` (University of Wrocław)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种新的抽象机器RKNL，用于实现强调用按需策略，结合了完全归一化和懒惰求值的共享原则。

**💡 创新点**

RKNL机器的设计简单明了，只有11条转换规则，并且在时间复杂度上表现出比现有机器更优的多项式效率。

**🔧 技术方法**

使用了高阶评估的归一化技术和memoization（记忆化）技术，通过功能对应转换工具自动推导出抽象机器。

**📊 数据集**

使用了λ演算的闭合术语作为数据集，进行了多种术语族的实验，验证了机器的性能。

**📈 对比分析**

与现有的强调用按需机器相比，RKNL在模拟开销上从指数级降低到多项式级，且在执行步骤上表现出双线性复杂度，线性依赖于β步骤的数量和输入术语的大小。

**⚠️ 局限性**

局限性在于该机器的实现主要是理论性的，尽管提供了合理的复杂度分析，但在实际应用中可能需要进一步的优化和实证研究。

---

## 872. AnkleType: A Hands- and Eyes-free Foot-based Text Entry Technique in Virtual Reality

**arXiv ID:** 2603.21915 | [PDF](https://arxiv.org/pdf/2603.21915v1)

**作者:** Xiyun Luo `[一作]` (Shantou University), Taizhou Chen `[通讯]` (Shantou University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种利用踝部旋转与脚趾点击实现手眼自由的VR文字输入技术，支持站立和坐姿场景；

**💡 创新点**

创新点包括：①首次将踝部运动作为主输入轴，实现低关节参与的站立与坐姿键盘布局；②提出单足与双足交替输入策略（UPStand、BPSit）；③通过用户手势引导、GMM聚类与词语预测相结合优化键盘布局；④在眼自由条件下实现可比的文本输入速度；

**🔧 技术方法**

使用了HTC Vive Tracker + 传感器实现踝部跟踪，Unity3D VR交互框架，贝叶斯最大似然词语预测算法，GMM聚类与仿真语言模型用于键盘布局优化，及NASA‑TLX、SUS等主观评估工具；

**📊 数据集**

使用了American National Corpus词频表、MacKenzie's phrase set（短语集）和Top‑10k词典进行实验与仿真；

**📈 对比分析**

通过4种输入策略的4×2对照实验测量WPM、TER、SUS、NASA‑TLX，并在7天纵向研究中比较视觉与眼自由条件；结果显示UPStand视觉15.05 WPM、BPSit视觉16.70 WPM，眼自由下分别为11.15/12.87 WPM，错误率低于3.5%，在与现有手眼自由与眼自由VR输入方法对比时表现出最优或竞争水平；

**⚠️ 局限性**

局限性包括：仅评估右脚占优势受试者，未覆盖左脚占优势或双脚输入差异；未研究双足键盘分布空间的潜在改进；实验未在真实多任务VR环境下评估；仅在VR中实现，未验证在MR/AR环境中的可行性；未系统评估长时间使用导致的疲劳和肌肉负荷。

---

## 873. A Novel Method for Enforcing Exactly Dirichlet, Neumann and Robin Conditions on Curved Domain Boundaries for Physics Informed Machine Learning

**arXiv ID:** 2603.21909 | [PDF](https://arxiv.org/pdf/2603.21909v1)

**作者:** Suchuan Dong `[一作]` (Purdue University), Yuchuan Zhang `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

无法确定

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

## 874. CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal

**arXiv ID:** 2603.21901 | [PDF](https://arxiv.org/pdf/2603.21901v1)

**作者:** Qingdong He `[一作]` (University of Electronic Science and Technology of China), Xiaobin Hu `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 CLEAR 框架，实现了无遮罩端到端的视频字幕去除。

**💡 创新点**

通过自监督前置学习、LoRA 自适应加权和生成反馈，实现参数高效、跨语言零样本的字幕去除。

**🔧 技术方法**

采用自监督特征解耦、LoRA 适配、视频扩散模型 Wan2.1‑Fun‑V1.1‑1.3B、自适应加权头和生成反馈等技术。

**📊 数据集**

在自采集的 160k 条中文字幕视频对上训练，并在 400 条测试集上评测。

**📈 对比分析**

与 ProPainter、MiniMax‑Remover、DiffuEraser 对比，CLEAR 在 PSNR +6.77 dB、VFID -74.7% 以及时序一致性方面显著提升，同时不需要遮罩。

**⚠️ 局限性**

仍受限于高分辨率视频、复杂遮挡和推理速度，需要进一步优化；在极端光照或模糊场景下性能可能下降。

---

## 875. Ara-Best-RQ: Multi Dialectal Arabic SSL

**arXiv ID:** 2603.21900 | [PDF](https://arxiv.org/pdf/2603.21900v1)

**作者:** Haroun Elleuch `[一作]`, Fethi Bougares `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了 Ara-BEST-RQ 自监督语音模型，基于 5,640 小时多方言 Arabic 语音进行家族化预训练，并公开模型、代码与数据集。

**💡 创新点**

首次构建大规模多方言 Arabic 预训练资源，并将家族化预训练与 BEST‑RQ 架构结合，在方言识别和 ASR 上取得 SOTA。

**🔧 技术方法**

采用 BEST‑RQ 架构、Conformer 编码器、Dynamic Chunk Training、masking 与随机投影量化等技术实现预训练与微调。

**📊 数据集**

使用 5,640 小时来自 YouTube Creative Commons 的多方言语音数据，外加 13,723 小时包含 MSA、古典阿拉伯、法语、英语、意大利语等公共数据的混合数据集。

**📈 对比分析**

与 HuBERT‑large、XLS‑R‑128、w2v‑BERT 2.0 等强基线比较，在 ADI‑20 方言识别和 MGB‑3/5、TARIC‑SLU、Common Voice 等 ASR 基准上，300M 版超过 Whisper‑large，600M 版仍保持竞争性 WER 与准确率。

**⚠️ 局限性**

局限包括数据分布不均衡、仅评估 DID 与 ASR、模型规模未突破 1B 参数且未探索更大或轻量化变体。

---

## 876. LRC-WeatherNet: LiDAR, RADAR, and Camera Fusion Network for Real-time Weather-type Classification in Autonomous Driving

**arXiv ID:** 2603.21987 | [PDF](https://arxiv.org/pdf/2603.21987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 877. IGV-RRT: Prior-Real-Time Observation Fusion for Active Object Search in Changing Environments

**arXiv ID:** 2603.21887 | [PDF](https://arxiv.org/pdf/2603.21887v1)

**作者:** Wei Zhang `[一作]` (Shandong University), Chaoqun Wang `[通讯]` (Shandong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合历史场景知识与实时视觉语言模型推理的概率规划框架，用于在时变室内环境中进行目标导航。

**💡 创新点**

创新点在于：①设计双层语义地图（信息增益图IGM和VLM分数图VLM‑SM）同时编码离线先验与在线语义证据；②在IGV‑RRT规划器中引入联合效用函数，既利用IGM引导全局搜索，又利用VLM‑SM校正在线目标位置；③采用探索区域掩模避免重访，并提供脱困机制。

**🔧 技术方法**

主要技术包括：3D场景图（3DSG）与概念网络共识推理构造IGM；BLIP‑2与LLM生成多提示语义查询并通过VLM‑SM累计；IGV‑RRT采样树规划结合信息增益、语义得分与距离；碰撞检测与运动学约束；GroundingDINO和MobileSAM进行目标检测与验证。

**📊 数据集**

使用HM3D仿真环境（含目标搬移实验）进行模拟测试，并在Wheeltec R550移动机器人上部署真实室内实验。

**📈 对比分析**

与VLFM基线比较，IGV‑RRT在HM3D时变实验中成功率（SR）从34.4%提升至42.9%，有效路径长度（SPL）从16.7%提升至26.3%。消融实验显示加入VLM‑SM和探索区域掩模后进一步提升。实验结果表明该方法在目标重定位场景中更稳健、更高效。

**⚠️ 局限性**

限制在于：IGM在执行时不动态更新，若环境变化频繁会导致先验失效；VLM‑SM对视觉遮挡和光照变化敏感；规划器对大规模复杂场景的实时性能尚未完全验证。

---

## 878. Not All Layers Are Created Equal: Adaptive LoRA Ranks for Personalized Image Generation

**arXiv ID:** 2603.21884 | [PDF](https://arxiv.org/pdf/2603.21884v1)

**作者:** Donald Shenaj `[一作]` (University of Pisa), Antonio Carta `[通讯]` (University of Pisa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一种在微调过程中自适应学习LoRA各层秩的模型，能够根据主题复杂度和层级需求动态调整秩，显著降低参数量。

**💡 创新点**

通过在LoRA权重上引入可学习的指数分布权重矩阵，促使秩维度按重要性排序，实现单个训练过程自动寻找最优秩，避免了传统固定秩或全量搜索的组合爆炸问题。

**🔧 技术方法**

采用变分推理框架、可学习指数分布、重缩放Kaiming初始化、MSE损失、秩正则化以及注意力熵正则化等技术实现自适应秩学习。

**📊 数据集**

使用30个个性化主题图像集（来自公开仓库），在SDXL和KOALA-700m两大后端上以DreamBooth协议进行实验。

**📈 对比分析**

与固定秩LoRA基线在DINO、CLIP-I、CLIP-T评分上对比，结果显示自适应秩方法在保持更低内存（如0.40GB对比2.8GB）同时获得更高DINO/CLIP-I分数，并在文本对齐上保持可接受水平。

**⚠️ 局限性**

仅在单一主题微调下验证，未评估多主题或模型融合场景；训练中需额外调节正则超参数，且在极高秩需求时仍可能不完全满足。

---

## 879. λ-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks

**arXiv ID:** 2603.21991 | [PDF](https://arxiv.org/pdf/2603.21991v1)

**作者:** Cristian Pérez-Corral `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实验了可学习的 λ‑GELU 激活函数，利用 λ 控制门的硬度，实现从平滑训练到 ReLU 兼容模型的可控迁移。

**💡 创新点**

创新点在于：① 通过 λ 参数化的 GELU 让门的锐度可调；② 引入 softplus 重参数化与学习率倍增以稳定 λ 的学习；③ 设计了线性硬化方案和基于 ℓ1 逼近误差的 λ_target，支持在训练后将 λ‑GELU 逐步替换为 ReLU。

**🔧 技术方法**

使用了参数化激活函数、softplus 重参数化、优化器自适应学习率缩放、层级硬度分析、Spearman 相关性评估、ℓ1 逼近误差计算、以及 MLP/CNN/Transformer 结构中的训练与验证实验。

**📊 数据集**

实验涵盖 Adult、FMNIST、CIFAR‑10、CIFAR‑100、TinyImageNet、WikiText‑2 等数据集。

**📈 对比分析**

与标准 GELU/RELU 进行对比；在 25% 训练后硬化 λ 并在最佳验证点替换为 ReLU，实验显示 λ‑GELU 在视觉和表格任务上与 GELU 基线性能相当，ReLU 替换导致的性能下降显著低于直接的 GELU→RELU 替换；在 GPT‑2 语言模型中替换仍表现不佳。

**⚠️ 局限性**

局限性包括：仅在层级上学习 λ，未考虑通道/神经元级别；在 Transformer/语言模型中 ReLU 替换效果差；λ_target 采用保守的分布无关估计，可能对带归一化的架构过度；未评估对量化、剪枝、可解释性等下游任务的实际收益。

---

## 880. GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design

**arXiv ID:** 2603.21978 | [PDF](https://arxiv.org/pdf/2603.21978v1)

**作者:** Xiaolei Zhou `[一作]` (Zhejiang University of Technology), Jianwei Zheng `[通讯]` (Zhejiang University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种端到端的几何扩散框架GeoFusion‑CAD，用于生成长序列的参数化Sketch‑Extrusion CAD程序。

**💡 创新点**

创新点在于将层级树编码与G‑Mamba状态空间扩散器相结合，实现线性时间建模长距离依赖，同时保持拓扑一致性。

**🔧 技术方法**

采用G‑Mamba扩散编码器、几何条件状态空间（GSM‑SSD）模块、线性时间Mamba、离散化CAD语言嵌入以及扩散概率模型。

**📊 数据集**

使用扩展版DeepCAD‑240数据集（命令长度40–240）以及原始DeepCAD作为基准。

**📈 对比分析**

与DeepCAD、SkexGen、HNC‑CAD等Transformer基准在短/长序列上对比，GeoFusion‑CAD在命令/参数准确率、COV最高，MMD/JSD最低，并显著降低显存和FLOPs。

**⚠️ 局限性**

局限性：目前仅支持单体零件，缺乏多组件装配和更丰富的参数化操作，对极复杂拓扑仍可能出现细小边界不连续。

---

## 881. SatGeo-NeRF: Geometrically Regularized NeRF for Satellite Imagery

**arXiv ID:** 2603.21931 | [PDF](https://arxiv.org/pdf/2603.21931v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 882. BHDD: A Burmese Handwritten Digit Dataset

**arXiv ID:** 2603.21966 | [PDF](https://arxiv.org/pdf/2603.21966v1)

**作者:** Swan Htet Aung `[一作]` (Expa.AI), Thuya Myo Nyunt `[通讯]` (Expa.AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集并发布了 87,561 张 28×28 像素灰度缅甸手写数字图像的 BHDD 数据集。

**💡 创新点**

首次构建缅甸手写数字基准，系统分析字形相似导致的误判，并提供可复现的评测基线。

**🔧 技术方法**

使用 Android 手工采集 App、OpenCV 自动提取与预处理、以及 MLP、CNN 与改进 CNN 等深度学习模型进行实验。

**📊 数据集**

BHDD 数据集，包括 60,000 张训练样本（每类 6,000 张）和 27,561 张测试样本，采用 28×28 灰度图像与 0–9 标签。

**📈 对比分析**

通过 MLP、CNN 与改进 CNN 三种基线模型对测试集进行评估，改进 CNN 在 99.83% 的准确率下仅出现 47 个错误。

**⚠️ 局限性**

主要限制在于误差集中于 0/1、0/8 等相似圆形数字，且数据集仅覆盖数字，未包含辅音或完整词句。

---

## 883. Albank -- a case study on the use of ethereum blockchain technology and smart contracts for secure decentralized bank application

**arXiv ID:** 2603.21894 | [PDF](https://arxiv.org/pdf/2603.21894v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 884. FeatDistill: A Feature Distillation Enhanced Multi-Expert Ensemble Framework for Robust AI-generated Image Detection

**arXiv ID:** 2603.21939 | [PDF](https://arxiv.org/pdf/2603.21939v1)

**作者:** Zhilin Tu `[一作]` (University of Electronic Science and Technology of China), Haiwei Wu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个集成四个Vision Transformer的多专家框架，结合数据扩展、降解建模与两阶段自蒸馏，提升AI生成图像在野外环境下的检测鲁棒性。

**💡 创新点**

创新点在于异构CLIP+SigLIP多专家集成、扩展的35算法降解库与外部数据扩充，以及密集特征自蒸馏的两阶段训练，显著提高对未知生成器和复杂降解的泛化与稳定性。

**🔧 技术方法**

使用了Vision Transformer（CLIP ViT‑L/14、SigLIP So400M）、自蒸馏（dense feature‑level distillation）、官方+扩展降解增强、外部数据融合（DiTFake、DiffFace、De‑Factify、Deepfake‑60K 等）、两阶段训练（BCE + dense distillation/contrastive）以及软投票集成。

**📊 数据集**

使用了 NTIRE 2026 Robust AI‑Generated Image Detection in the Wild 赛题数据（约27.7 万张训练集，覆盖20+ 生成器与多种降解）以及外部扩充数据约205K张（DiTFake、DiffFace、De‑Factify、Deepfake‑60K 等）。

**📈 对比分析**

与 Swin‑T、ConvNeXt、BEiT、DINO、Moco、CLIP、SigLIP 等基线模型对比，单模型在线上验证集的 ROC AUC 从 0.78 提升至 0.93，集成模型在最难的线上测试（Hard）达到了 0.856 的 ROC AUC，显著优于传统方法。

**⚠️ 局限性**

仍然依赖大型 Transformer 模型和大量显存，推理虽控制在约10GB，但在资源受限环境下受限；对极端或未见的降解/生成器仍可能出现误检，需要持续维护多源数据和降解库的更新。

---

## 885. Chronological Contrastive Learning: Few-Shot Progression Assessment in Irreversible Diseases

**arXiv ID:** 2603.21935 | [PDF](https://arxiv.org/pdf/2603.21935v1)

**作者:** Clemens Watzenböck `[一作]` (Medical University of Vienna), Georg Langs `[通讯]` (Medical University of Vienna)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种ChronoCon时序对比学习框架，利用患者影像的访问顺序在无标签条件下学习疾病严重程度表征。

**💡 创新点**

创新点在于将Rank-N-Contrast的排名思路迁移到时间序列，使用时间顺序作为正负样本约束，显著提升低标签环境下的特征学习效果。

**🔧 技术方法**

采用自监督对比学习（ChronoCon）与ResNet18编码器、双crop增强、DAE重建损失以及多头回归微调，并使用温度参数τ控制相似度。

**📊 数据集**

实验基于维也纳医科大学的778名类风湿关节炎患者手足X光影像，13,742张影像，407,045个评分，平均每患者4次随访。

**📈 对比分析**

与单阶段监督模型、SimCLR、时间版Rank-N-Contrast以及DAE预训练等方法比较，ChronoCon在仅5-15%标签时即可达到ICC 0.86、RMSE 19.9，并在长期ΔSvH评估中几乎不随标签量变化，优于现有方法。

**⚠️ 局限性**

局限性包括：依赖单中心数据；假设疾病进展单调，可能不适用于可逆或非单调疾病；需要足够多时间点；子组标识选择可能引入偏差。

---

## 886. From Scores to Strategies: Towards Gaze-Informed Diagnostic Assessment for Visualization Literacy

**arXiv ID:** 2603.21898 | [PDF](https://arxiv.org/pdf/2603.21898v1)

**作者:** Kathrin Schnizer `[一作]` `[通讯]` (LMU Munich), Kathrin Schnizer (LMU Munich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出将眼动追踪作为可扩展且诊断性的可视化素养评估补充；

**💡 创新点**

首次将眼动指标与标准化测试结合，用以区分流畅理解与费力成功；

**🔧 技术方法**

眼动追踪技术、视图组件关注度分析、视线路径散布评估等方法；

**📊 数据集**

未使用具体数据集，主要基于文献综述和理论构建；

**📈 对比分析**

未进行实验比较，本文为概念性定位与路线图提出；

**⚠️ 局限性**

缺乏经验验证、指标可靠性与跨平台适用性等方面的实际证据。

---

## 887. Deep S2P: Integrating Learning Based Stereo Matching Into the Satellite Stereo Pipeline

**arXiv ID:** 2603.21882 | [PDF](https://arxiv.org/pdf/2603.21882v1)

**作者:** Elías Masquil `[一作]` (Universidad de la República, Uruguay), Gabriele Facciolo `[通讯]` (Université Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将多种现代学习式立体匹配模型（StereoAnywhere、MonSter、FoundationStereo 等）集成到 Satellite Stereo Pipeline（Deep S2P）中，并通过改进校正流程确保差异性、相位一致与范围适配。

**💡 创新点**

首次通过调整校正步骤（实现单极差异性、相位一致和范围平移）使得学习式匹配器能在卫星影像管线中稳定工作，并公开完整代码以便复现。

**🔧 技术方法**

利用 RPC‑基础的虚拟对应校正、水平平移实现单极差异性、左右一致性过滤、模型符号校正，并在 Deep S2P 中嵌入深度学习匹配器（如 PSMNet、RAFT‑Stereo、MonSter 等）。

**📊 数据集**

使用 2019 IEEE GRSS Data Fusion Contest 中的 WorldView‑3 同日立体对（53 个 AOI）以及 LiDAR 基准 DSM 与语义标注；同时评估了几组视角极端的挑战性对。

**📈 对比分析**

通过 90th 百分位误差、NMAD、RMSE、MAE、有效点比例等指标与传统 SGM/MGM 进行对比，学习模型在所有指标上均优于传统方法，FoundationStereo 在数值与完整度上表现最佳，并在视觉上显著提升结构细节。

**⚠️ 局限性**

对植被区域的误差仍高且与传统方法差距缩小；在极端视角下完整度下降；现有评估指标与有限分辨率的基准 DSM 受限，导致数值提升相对有限。

---

## 888. Group3D: MLLM-Driven Semantic Grouping for Open-Vocabulary 3D Object Detection

**arXiv ID:** 2603.21944 | [PDF](https://arxiv.org/pdf/2603.21944v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 889. The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation

**arXiv ID:** 2603.21928 | [PDF](https://arxiv.org/pdf/2603.21928v1)

**作者:** Guannan Lai `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

在连续测试时不访问源数据的条件下，提出了GOLD框架，通过在低秩“金子子空间”内对特征进行轻量级投影与缩放，实现高效、稳健的在线适应。

**💡 创新点**

核心创新是：①理论证明金子子空间存在且低秩；②利用AGOP在线估计并更新该子空间；③在此子空间上训练仅需一个可学习的缩放向量，避免全模型更新。

**🔧 技术方法**

采用的技术包括：平均梯度外积（AGOP）做子空间估计；特征投影+残差缩放；EMA教师+SCE自训练一致性损失；原型对比InfoNCE损失；低秩特征投影与特征重建。

**📊 数据集**

实验使用的公开数据集有：CIFAR10‑C、CIFAR100‑C、ImageNet‑C（15种噪声、severity 5）以及CARLA模拟的CarlaTTA语义分割序列。

**📈 对比分析**

与TENT、CoTTA、RMT、SOTA CTTA方法对比，GOLD在三大分类基准上平均错误率最低（如CIFAR10‑C 14.1%、CIFAR100‑C 37.1%、ImageNet‑C 76.6%），在CarlaTTA上mIoU最高（如highway 34.5%），且每批推理+适应时间约0.25 s，保持最快速度之一。

**⚠️ 局限性**

局限性：需要预先获取源模型的分类器权重与原型；AGOP估计在极端噪声或小样本情形下可能不稳定；目前证明和实验仅针对单步适应与标准Bench，未在更大规模或不同行业场景中验证；对高维特征空间的实时AGOP更新仍有计算开销。

---

## 890. Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment

**arXiv ID:** 2603.21936 | [PDF](https://arxiv.org/pdf/2603.21936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 891. BOOST-RPF: Boosted Sequential Trees for Radial Power Flow

**arXiv ID:** 2603.21977 | [PDF](https://arxiv.org/pdf/2603.21977v1)

**作者:** Ehimare Okoyomon `[一作]` (Technical University of Munich), Christoph Goebel `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将配电网电压预测从全局图回归任务转化为根到叶子路径的序列学习问题，并使用梯度提升树实现。

**💡 创新点**

创新点在于：①引入路径级别的递归学习，将电压下降视为局部映射；②用父子残差作为训练目标，显著提升可学习性；③结合物理线性基线实现物理信息化残差学习。

**🔧 技术方法**

主要技术包括：BFS路径分解、逆向电流累加、LinDistFlow基线计算、梯度提升树（XGBoost）三种变体（绝对电压、父节点残差、物理残差），以及自动回归的推理流程。

**📊 数据集**

使用的实验数据集：Kerber Dorfnetz（单个0.4kV径向网，1800样本）和ENGAGE低压六个径向网（共1800样本）以及在不同电能资源插入情景下生成的合成数据。

**📈 对比分析**

与传统DistFlow、全局MLP和ARMA-GNN等基线对比。BOOST‑RPF（Parent‑Residual）在固定网、异构网和OOD情景下均取得最小RMSE，平均电压幅值误差≤0.0008p.u，角度误差≤0.006°；训练时间仅几秒，推理时间随节点数线性增长，满足实时应用需求。

**⚠️ 局限性**

局限性：仅适用于径向（树状）网，无法直接处理网状/环路网；Python层调用导致线性O(N)推理瓶颈；实验仅在单相等效模型，未验证三相不平衡情况；缺乏针对长路径误差累计的自校正机制。

---

## 892. SecureBreak -- A dataset towards safe and secure models

**arXiv ID:** 2603.21975 | [PDF](https://arxiv.org/pdf/2603.21975v1)

**作者:** Marco Arazzi `[一作]` (University of Pavia), Antonino Nocera `[通讯]` (University of Pavia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并构建了 SecureBreak 数据集，用于在生成后对 LLM 输出进行安全性分类，并在多种大模型上进行微调和评估。

**💡 创新点**

创新点在于：① 将侧重于 prompt 的 jailbreak 研究转向 response 级别的安全判别；② 采用保守的人类专家标注，确保高安全性；③ 通过该数据集实现“终极”防御层和对齐过程的监督反馈。

**🔧 技术方法**

技术包括：使用 LoRA/QLoRA 进行低秩微调；Seq2Seq 分类训练；统一的 4‑bit 量化、混合精度训练；对比基准评估（准确率、类别级别指标）。

**📊 数据集**

使用的数据集：SecureBreak（3059 条样本，来自 JailbreakBench 的 100 个有害问题并由多款 LLM 生成回应后人工标注），并以 JailbreakBench 作为原始 prompt 源。

**📈 对比分析**

比较方法：将未微调的基线模型与在 SecureBreak 上微调后的模型在安全判别任务上进行对比；评估指标为整体准确率和各安全风险类别的准确率。结果显示：Mistral‑7B‑v0.3 微调后准确率 83.24%，Llama‑3.1‑8B 81.22%，Selene‑1‑Mini‑Llama‑3.1‑8B 76.08%；Qwen‑2.5‑0.5B 采用 Seq2Seq 训练达到 90.14%；微调后在多类别中均显著提升，尤其在 Fraud/Deception、Privacy、Disinformation 等中收益显著。

**⚠️ 局限性**

局限性：样本量相对有限（3059 条），类别分布略偏，主要基于现有 JailbreakBench 的 100 个问题；模型评估仅覆盖少数几种 LLM；手工标注虽保守但仍可能存在主观性；未验证在更广泛攻击类型或新的模型架构上的泛化能力。

---

## 893. SLURP-TN : Resource for Tunisian Dialect Spoken Language Understanding

**arXiv ID:** 2603.21940 | [PDF](https://arxiv.org/pdf/2603.21940v1)

**作者:** Haroun Elleuch `[一作]` (ELYADATA), Fethi Bougares `[通讯]` (ELYADATA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了SLURP‑TN，一套基于SLURP的突尼斯阿拉伯语多说话人、多域、三种录音环境（干净、噪声、耳机）的SLU/ASR数据集，并在其上评测多种最先进的声学模型。

**💡 创新点**

创新点在于提供高采样率、覆盖多方言、性别平衡且含代码切换的突尼斯阿拉伯语语料，并公开对应的ASR与SLU基准结果，弥补了低资源方言数据缺口。

**🔧 技术方法**

使用SpeechBrain框架训练w2v‑BERT、SENSE、Whisper等多语种自监督声学模型，结合字符级tokenizer、CTC/NLL损失进行Fine‑tune，并在三种声学条件下评估。

**📊 数据集**

数据集来源于SLURP的六个域场景，翻译并录制约4,165句突尼斯方言语音，包含干净、噪声和耳机三种声学条件；同时利用TEDx‑TN进行预训练。

**📈 对比分析**

通过CER/WER、CoER/CVER等指标，在clean、headphone和noisy环境下对ASR/SLU模型进行评估，结果显示w2v‑BERT在ASR上优于Whisper，而SENSE在SLU语义准确度上更佳，整体误差仍较高表明数据集具有挑战性。

**⚠️ 局限性**

局限在于仅覆盖原SLURP的六个域，未达到完整语料覆盖；且代码切换与说话人多样性仍可能导致模型泛化受限。

---

## 894. Asymptotically Ideal Hierarchical Secret Sharing Based on CRT for Integer Ring

**arXiv ID:** 2603.22011 | [PDF](https://arxiv.org/pdf/2603.22011v1)

**作者:** Jian Ding `[一作]`, Haifeng Yu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了基于整数环CRT和单向函数的渐进理想离散与合取分层秘密共享方案

**💡 创新点**

通过1-紧凑互质数列实现可变大小份额，兼顾安全性、信息率趋近1以及计算效率，弥补了以往方案的安全缺陷与信息率低下问题

**🔧 技术方法**

利用中国剩余定理、k-紧凑互质数列、单向函数以及离散对称加密的构造技巧

**📊 数据集**

无实验数据集，主要为理论构造与证明

**📈 对比分析**

与现有CRT基础方案（如Harn、Ersoy、Tiplea等）比较，本文方案在信息率上从小于1/2提升至渐进1，安全性得到证明，计算复杂度保持多项式级别

**⚠️ 局限性**

仍需在大参数规模下实现高效计算，依赖单向函数安全性假设，且对参数选择有一定要求（如大素数m0及其上下界）

---

## 895. TALUS: Threshold ML-DSA with One-Round Online Signing via Boundary Clearance and Carry Elimination

**arXiv ID:** 2603.22109 | [PDF](https://arxiv.org/pdf/2603.22109v1)

**作者:** Leo Kao `[一作]` `[通讯]` (Codebat Technologies Inc.), Leo Kao (Codebat Technologies Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出TALUS协议，实现NIST ML-DSA阈值签名，支持TEE与完全分布式两种实现

**💡 创新点**

引入Boundary Condition Check过滤非法随机数，设计Carry Elimination Framework实现分布式hash输入计算，并在T=2时采用高效DCF方案

**🔧 技术方法**

利用Shamir秘密共享、Pedersen DKG、Feldman VSS、Beaver triples、DCF、CSCP、FFT等技术

**📊 数据集**

在NIST给定的ML-DSA参数集上进行实验（ML-DSA-65、87、94等）

**📈 对比分析**

与单机ML-DSA、MCLDSA等对比，在线仅需1轮，通信量与单机相当，预处理通信约350KB/批，T=2时仅5KB，成功率约31.5%

**⚠️ 局限性**

预处理需多轮重试（约3.15轮）以满足BCC过滤；T≥3需至少2T-1方；存在信息理论损失（Irwin-Hall分布导致），大规模N或高T需升级到更高级别ML-DSA

---

## 896. Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning

**arXiv ID:** 2603.22070 | [PDF](https://arxiv.org/pdf/2603.22070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 897. Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison

**arXiv ID:** 2603.22075 | [PDF](https://arxiv.org/pdf/2603.22075v1)

**作者:** Caio Vicentino `[一作]` `[通讯]` (Independent Researcher), Caio Vicentino (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在相同数据、计算预算与硬件条件下，对比了自回归（AR）与遮蔽扩散（MDLM）语言模型的训练效率、收敛速度和生成多样性。

**💡 创新点**

提供了严格对齐的实验基准，证明扩散训练成本不高，并揭示两种范式在流畅性与多样性上的互补性。

**🔧 技术方法**

使用Transformer架构，AR采用因果注意力与下一个词预测，MDLM采用双向注意力与时间步嵌入加上余弦遮蔽调度；生成时分别使用核采样与基于置信度的逐步解码。

**📊 数据集**

TinyStories 语料库（约 50M 词）。

**📈 对比分析**

通过 20K 步训练、吞吐量（tokens/s）、验证损失曲线和 1,000 个样本的 Distinct‑n、Self‑BLEU 与唯一开头等指标进行比较；结果显示 AR 更快收敛但出现过拟合，MDLM 收敛更慢但生成多样性更高，吞吐量相差仅 4.7%。

**⚠️ 局限性**

实验规模有限（123–163M 参数，50M 词），MDLM 参数量比 AR 大 31.6%，单一数据集、单随机种子、采样策略未充分调优，未评估下游任务。

---

## 898. FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation

**arXiv ID:** 2603.22054 | [PDF](https://arxiv.org/pdf/2603.22054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 899. MIHT: A Hoeffding Tree for Time Series Classification using Multiple Instance Learning

**arXiv ID:** 2603.22074 | [PDF](https://arxiv.org/pdf/2603.22074v1)

**作者:** Aurora Esteban `[一作]` (University of Cordoba), Sebastián Ventura `[通讯]` (University of Cordoba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 MIHT（Multi-instance Hoeffding Tree）算法，用于处理多变量、可变长度的时间序列分类。

**💡 创新点**

创新点在于：①将多实例学习与增量决策树结合；②使用滑动窗口生成“子序列袋”，并通过优化选择最相关的 k 个子序列；③在增量树中使用信息增益与 Hoeffding 边界进行在线分裂，保持模型可解释性。

**🔧 技术方法**

采用的技术包括多实例学习框架、增量决策树（Hoeffding Tree）、信息增益、滑动窗口分割、k 个子序列优化、适应性朴素贝叶斯概率估计。

**📊 数据集**

实验使用了 UCR/UEA 公开的 28 个可变长度、多变量/单变量时间序列数据集。

**📈 对比分析**

与 11 种最先进方法（DrCIF、ST、MUSE、ROCKET、SVM、kNN、TapNet、InceptionTime、HIVECOTE2 等）进行基准对比，MIHT 在 28 个数据集上取得 11 个最佳结果，平均准确率 0.597，比其他方法高约 5%–17%，且统计检验显示显著优势。

**⚠️ 局限性**

局限性包括：需要手工调节滑动窗口与 k 参数；无法验证模型标记的关键片段（缺少细粒度标签）；在极大维度或极短序列时表现可能不如深度学习或 DTW；未对实时流数据进行评估。

---

## 900. RAFL: Generalizable Sim-to-Real of Soft Robots with Residual Acceleration Field Learning

**arXiv ID:** 2603.22039 | [PDF](https://arxiv.org/pdf/2603.22039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 901. On the Interplay of Priors and Overparametrization in Bayesian Neural Network Posteriors

**arXiv ID:** 2603.22030 | [PDF](https://arxiv.org/pdf/2603.22030v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 902. AdditiveLLM2: A Multi-modal Large Language Model for Additive Manufacturing

**arXiv ID:** 2603.22017 | [PDF](https://arxiv.org/pdf/2603.22017v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 903. PreferRec: Learning and Transferring Pareto Preferences for Multi-objective Re-ranking

**arXiv ID:** 2603.22073 | [PDF](https://arxiv.org/pdf/2603.22073v1)

**作者:** Wei Zhou `[一作]` (Shenzhen University), Xiuqiang He `[通讯]` (Shenzhen Technology University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 PreferRec 框架，通过在用户意图层学习 Pareto 优化偏好并实现跨用户知识转移，以实现多目标推荐重排序。

**💡 创新点**

创新点包括：① 在用户层面捕捉 Pareto 方向的个性化偏好；② 通过知识导向转移共享同质优化空间中的 Pareto 结构；③ 利用 Pareto 集学习模型直接引导进化搜索，提升搜索效率与多目标平衡。

**🔧 技术方法**

采用了进化多目标优化（如 NSGA‑II/MOEA/D）、Pareto 集学习（MLP/Transformer 预测软标签）、知识导向转移、BERT 语义嵌入与软标签训练等技术。

**📊 数据集**

使用了三大公开数据集：MovieLens 1M、Amazon Grocery 与 Amazon Beauty。

**📈 对比分析**

在单目标序列模型（GRU4Rec、SASRec 等）和多目标重排序基线（MMR、DPP、EMMR 等）中进行对比，PreferRec 在 HR@10、NDCG、diversity、novelty、F1/F2 等指标上均取得显著提升，并显著降低了多目标重排序的计算成本。

**⚠️ 局限性**

局限性包括：对进化算法的依赖导致计算开销较大；对转移间隔、种群大小等超参数敏感；仅适用于同质多目标空间，对跨域或异质用户的迁移效果尚未验证。

---

## 904. FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario

**arXiv ID:** 2603.22102 | [PDF](https://arxiv.org/pdf/2603.22102v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 905. MineRobot: A Unified Framework for Kinematics Modeling and Solving of Underground Mining Robots in Virtual Environments

**arXiv ID:** 2603.22055 | [PDF](https://arxiv.org/pdf/2603.22055v1)

**作者:** Shengzhe Hou `[一作]` (Shandong University of Science and Technology), Xingli Zhang `[通讯]` (Shandong University of Science and Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了 MineRobot 框架，用 MRDF 描述矿山机器人，自动化拓扑处理并实现实时正逆运动学求解。

**💡 创新点**

创新点是引入四杆链收缩为通用关节、提取独立拓扑等价路径（ITEP）并基于 ITEP 的顺序正运动学与基于 Gauss-Seidel 的逆运动学。

**🔧 技术方法**

采用 JSON 形式的 MRDF、Tarjan 强连通分量算法、四杆收缩、路径提取、Newton-Raphson 根求、黄金分割搜索以及 Eigen C++ 实现。

**📊 数据集**

使用多种真实矿山机器人模型（如 TCCHS、SSHS、RH、LHD 等）以及公开的三维几何文件作为测试数据。

**📈 对比分析**

与 Drake、MuJoCo 等通用框架对比，在 100 次随机实验中，MineRobot 的正运动学 1–2 ms、逆运动学 50–100 ms，成功率 100%，比基线快 2–3 倍且更稳定。

**⚠️ 局限性**

局限在于只处理固定基座的平面四杆闭环结构，未覆盖空间闭环、动态、碰撞等高级物理；对非矿山机器人的通用性不足。

---

## 906. Dynamic analysis enhances issue resolution

**arXiv ID:** 2603.22048 | [PDF](https://arxiv.org/pdf/2603.22048v1)

**作者:** Mingwei Liu `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 DAIRA 框架，将动态执行追踪嵌入 LLM 代理的推理循环，实现自动化的软件缺陷定位与修复。

**💡 创新点**

首次将动态追踪与结构化执行日志语义化分析相结合，构建 Test Tracing‑Driven 推理模式，显著提升了定位精度和修复质量。

**🔧 技术方法**

使用 LLM（Gemini 3 Flash Preview）+ 自定义动态追踪工具（Hunter）+ 结构化执行日志语义化分析 + 轻量级代理交互。

**📊 数据集**

使用 SWE‑bench Verified 数据集（500 条真实项目缺陷）。

**📈 对比分析**

与 SWE‑agent、Live‑SWE‑agent、OpenHands 等 SOTA 框架对比，DAIRA 在 500 条任务上实现 79.4% 的解决率，比基线高约2%，同时推理成本下降约10%，代币消耗下降约25%。

**⚠️ 局限性**

仅针对 Python，依赖特定追踪钩子，跨语言适配有限；动态分析开销仍存在，极端复杂运行时状态仍可能无法完全捕获。

---

## 907. Multiperspectivity as a Resource for Narrative Similarity Prediction

**arXiv ID:** 2603.22103 | [PDF](https://arxiv.org/pdf/2603.22103v1)

**作者:** Max Upravitelev `[一作]` (Technische Universitaet Berlin), Vera Schmitt `[通讯]` (Technische Universitaet Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种将多视角解读整合进叙事相似度预测的方法，并通过多种 LLM 人格的多数投票来提升预测准确率。

**💡 创新点**

创新点在于将解读多样性视为模型组成部分而非噪声，利用 31 个不同解读框架的 LLM 人格与跨模型投票，验证多视角多模型投票在解释性任务中可提升整体性能。

**🔧 技术方法**

使用了三大类 LLM（Gemma 27B‑it、Qwen3‑14B、gpt‑oss‑20b）生成 31 个角色化 prompt，分别为“Practitioner”与“Lay”两类；采用多数投票（个人、模型内、跨模型）以及基于 oracle 的阈值分析。

**📊 数据集**

使用 SemEval‑2026 Task 4 叙事相似度数据集，开发集 200 条样本，测试集 400 条样本。

**📈 对比分析**

与单一 LLM、单一角色及优化子集 ensemble 进行对比；简单多数投票在测试集上达 0.7025 的准确率，优化子集在 dev 上可达 82% 但在测试集上仅提升至 0.705，说明大规模多样化投票更稳健。

**⚠️ 局限性**

主要局限包括：未评估多代理辩论（MAD）等互动式方法；相关性分析不证明因果；实验仅在单一任务与小规模 dev 集上验证，缺乏跨数据集的通用性验证。

---

## 908. GSEM: Graph-based Self-Evolving Memory for Experience Augmented Clinical Reasoning

**arXiv ID:** 2603.22096 | [PDF](https://arxiv.org/pdf/2603.22096v1)

**作者:** Xiao Han `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了GSEM——一种基于图结构的自适应记忆框架，用于临床决策代理的经验存储与检索。

**💡 创新点**

将经验细化为指示/禁忌，构建双层图记忆，并结合可应用性检索和在线反馈自我演化，提升经验复用的精确度与可靠性。

**🔧 技术方法**

图神经网络/图结构检索、LLM引导的多种种子遍历、经验可靠性验证(ERV)、反馈驱动的质量与关系权重更新。

**📊 数据集**

MedR-Bench（诊断与治疗）与 MedAgentsBench（多子集 QA）两大医学推理基准。

**📈 对比分析**

与Naïve RAG、GraphRAG、Mem0、A-Mem、ReMe、FLEX等三类基线在同一LLM背景下对比，GSEM在DeepSeek-V3.2上达94.59%治疗准确率，平均70.90% Pass@1；在Qwen3.5-35B上亦保持领先。

**⚠️ 局限性**

仅在公开基准上验证，缺乏真实临床流程评估；记忆构建与图遍历成本高；检索策略对提示敏感且可能产生非确定性路径。

---

## 909. P-Flow: Prompting Visual Effects Generation

**arXiv ID:** 2603.22091 | [PDF](https://arxiv.org/pdf/2603.22091v1)

**作者:** Rui Zhao `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种训练‑free 的框架，通过在推理时对文本提示进行优化，实现对视频生成模型的动态视觉效果（如爆炸、压碎等）进行定制。

**💡 创新点**

创新点在于：①将视觉‑语言模型（VLM）用于测试时提示优化；②引入噪声先验增强和轻量化历史轨迹机制，既保证生成效果的稳定性，又提升了动态一致性；③框架对文本‑到‑视频与图像‑到‑视频均适用，无需任何模型微调。

**🔧 技术方法**

核心技术包括流匹配反演提取噪声先验、两阶段 SVD 投影去除外观噪声、噪声混合保持多样性、基于 Gemini 1.5 Pro 的 VLM 结构化提示优化以及历史轨迹上下文维护。

**📊 数据集**

实验采用 Open‑VFX 数据集（675 条动态效果视频，15 类效果）以及对应的图像参考集，进行图像‑到‑视频和文本‑到‑视频两种任务。

**📈 对比分析**

与 Wan 2.1、HunyuanVideo 等基础模型以及专门的 VFX Creator 进行比较，评估指标包括 FID‑VID、FVD 和 Dynamic Degree；实验表明，本方法在三项指标上均优于或与 VFX Creator 相当，且在 Human Evaluation 中被 70%‑90% 的受试者认为效果更佳。

**⚠️ 局限性**

局限性包括：①对 VLM 的推理开销较大，导致每次迭代耗时较长；②目前仅支持单场景视频，跨场景泛化仍有待验证；③提示优化过程依赖于良好的参考视频，若参考视频质量或长度受限，效果可能下降。

---

## 910. On the Complexity of Fundamental Problems for DAG-Compressed Graphs

**arXiv ID:** 2603.22063 | [PDF](https://arxiv.org/pdf/2603.22063v1)

**作者:** Florian Chudigiewitsch `[一作]` (Universität zu Lübeck), Felix Winkler `[通讯]` (Universität zu Lübeck)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并证明DAG压缩相较树压缩能获得更优压缩效果，并在DAG压缩上实现Kruskal最小生成树算法，同时证明最小DAG压缩及其增删边更新问题为NP‑难；

**💡 创新点**

①首次给出DAG压缩比树压缩更有效的理论证明；②在DAG压缩上设计了可直接运行的Kruskal算法；③证明最优DAG压缩及其动态更新问题的NP‑难性；

**🔧 技术方法**

图论与DAG结构分析、逆Ackermann函数复杂度、集合覆盖归约、压缩边与集群的数学证明、算法时间复杂度分析；

**📊 数据集**

主要以理论实例——g×g跳棋格子图（Rook graph）为例，没有使用具体实验数据集；

**📈 对比分析**

通过对Rook图的边数对比：DAG压缩仅需O(n^{3/2})边，而树压缩至少需Ω(n^{3/2})边，体现了压缩效率提升；Kruskal算法在压缩图上的实现时间为O((|A|+|E|)α(|V|)+t_sort(|E|))，相较于原图O(|E|log|E|)获得显著加速；

**⚠️ 局限性**

缺点在于最优压缩和动态更新问题证明为NP‑难，实际构造最优压缩仍无多项式算法，且本文缺乏实验验证与实现细节。

---

## 911. On the Failure of Topic-Matched Contrast Baselines in Multi-Directional Refusal Abliteration

**arXiv ID:** 2603.22061 | [PDF](https://arxiv.org/pdf/2603.22061v1)

**作者:** Valentin Petrov `[一作]` `[通讯]` (INMECHA INC), Valentin Petrov (INMECHA INC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究在去除大型语言模型拒绝行为时，使用主题匹配与未匹配的对照基线对抽取方向的影响，并在 Qwen 3.5 2B 上进行实验。

**💡 创新点**

揭示主题匹配对照基线反而导致拒绝方向失效的现象，挑战先前关于对照基线应匹配主题的假设，并解释了方向大小与主题噪声的关系。

**🔧 技术方法**

采用 Self‑Organizing Map (SOM) 多类抽取、SVD 正交化、方向投影（abliteration）以及多层、多权重评估框架。

**📊 数据集**

481 条有害提示（按九类分类）、手工构造的主题匹配安全提示、400 条通用 harmless_alpaca 提示作为未匹配基线。

**📈 对比分析**

通过在不同层、不同权重下测算拒绝次数 R 并与 KL 散度比较。未匹配基线在层 9、14、15 处 w=0.5 时实现 R=0、KL<0.005；匹配基线无任何层下降；SVD 正交化后完全失效；捕获率差距约 9%。

**⚠️ 局限性**

仅在单一 Qwen 3.5 2B 混合注意力架构上验证；未考察更大规模或其他模型；统一权重的评估方案导致 SVD 失效；提示集多样性有限。

---

## 912. ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention

**arXiv ID:** 2603.22016 | [PDF](https://arxiv.org/pdf/2603.22016v1)

**作者:** Xinyan Wang `[一作]` (University of Wisconsin-Madison), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ROM框架，使用轻量检测头在冻结大模型的隐藏状态上实时识别并提前终止过度推理；

**💡 创新点**

将过度推理视作流式预测‑控制问题，引入Token级检测头、Counterfactual Self‑Correction数据增强和回溯干预，实现零改造、可插拔的实时早停；

**🔧 技术方法**

采用注意力池化+CfC循环单元构建序列检测器、二分类输出头，结合CSC增强和边界感知回溯干预；

**📊 数据集**

七个基准数据集：MATH500、GSM8K、ASDiv、MAWPS、MultiArith、SVAMP、MMLU‑Pro；

**📈 对比分析**

与Vanilla、RL方法L1、启发式EAT及固定截断等做对比，ROM_CSC在7个数据集上准确率93.51%，比基线高1.79%，token平均减少47%，效率提升121%，在算术任务上更显著；

**⚠️ 局限性**

主要局限：依赖高性能标注模型（如GPT‑4o）进行标签生成，数据增强对标注质量敏感；在训练样本规模扩大时未出现明显性能提升。

---

## 913. VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models

**arXiv ID:** 2603.22003 | [PDF](https://arxiv.org/pdf/2603.22003v1)

**作者:** Zixuan Wang `[一作]` (Hong Kong University of Science and Technology), Jiaya Jia `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种双系统框架 VP-VLA，利用视觉提示（交叉线、边界框）把高层规划与低层执行分离，使机器人在完成复杂多阶段指令时能精准定位目标并执行动作。

**💡 创新点**

创新点包括：① 视觉提示作为显式结构化中介，将抽象语言指令转化为可直接用于控制的空间锚点；② 在训练中加入辅助视觉定位损失，迫使策略内部对提示位置保持高度一致性；③ 采用事件驱动的任务分解，仅在物理状态变化时重新生成提示，减少不必要计算。

**🔧 技术方法**

核心技术：双系统架构（System 2 Planner + System 1 Controller）；预训练的 VLM（Qwen3‑VL‑4B‑Instruct）做高层规划；SAM3 进行语义分割生成视觉提示；视觉提示输入与原始图像拼接；训练损失为动作 L1 + 提示定位交叉熵；使用 AdamW 优化器。

**📊 数据集**

主要数据集：Robocasa‑GR1‑Tabletop（24K 任务视频），SimplerEnv（BridgeDataV2 与 Fractal 子集），以及真实机器人 Franka Research 3 的废物分类、彩蛋属性与格子放置三大任务的数据。

**📈 对比分析**

与基线 QwenOFT、GR00T‑N1.6、Isaac‑GR00T 等进行对比。Robocasa 平均成功率从 48.8% 提升至 53.8%（+5.0%）；SimplerEnv 平均成功率从 50.0% 提升至 58.3%（+8.3%）。在真实废物分类中，ID 成功率 87.5% 与 OOD 85%（相较 QwenOFT 的 80%/63.3%）；彩蛋属性任务 ID 77.1% 与 OOD 色彩 75.0%；格子放置 ID 91.25% 与 OOD 68.75%。

**⚠️ 局限性**

局限性：① 需要高质量的语义分割/提示生成，分割错误会直接导致失效；② 视觉提示主要针对可视化目标，难以处理完全文本或多模态复杂指令；③ 在极度动态或高度不确定的环境中，事件驱动触发机制可能错过关键转折点；④ 依赖预训练 VLM 对语言理解的限制，可能在新语义或极端用语上表现不佳。

---

## 914. CRPS-Optimal Binning for Conformal Regression

**arXiv ID:** 2603.22000 | [PDF](https://arxiv.org/pdf/2603.22000v1)

**作者:** Paolo Toccaceli `[一作]` `[通讯]` (University of London), Paolo Toccaceli (University of London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于分区的非参数条件分布估计方法，通过将协变量排序的观察值分成连续的区间，并使用区间内的经验CDF作为预测分布。

**💡 创新点**

创新点在于通过最小化总的留一交叉排名概率分数（LOO-CRPS）来选择区间边界，并通过动态规划在O(n^2 K)时间内恢复全局最优K分区。

**🔧 技术方法**

使用了动态规划和留一交叉排名概率分数（LOO-CRPS）作为成本函数，结合交叉验证选择K。

**📊 数据集**

在真实数据集上进行了实验，包括Old Faithful和摩托车事故数据集。

**📈 对比分析**

与分割保形竞争者（高斯分割保形、CQR和CQR-QRF）进行比较，结果显示该方法产生的预测区间显著更窄，同时保持接近名义覆盖率。

**⚠️ 局限性**

限制在于该方法要求一维协变量和连续的分区结构，在小样本情况下可能选择较大的K，导致每个区间的观察值较少，从而影响预测区间的精度。

---

## 915. Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models

**arXiv ID:** 2603.22094 | [PDF](https://arxiv.org/pdf/2603.22094v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 916. Designing Medical Chatbots where Accuracy and Acceptability are in Conflict: An Exploratory, Vignette-based Study in Urban India

**arXiv ID:** 2603.22115 | [PDF](https://arxiv.org/pdf/2603.22115v1)

**作者:** Ananditha Raghunath `[一作]` (University of Washington), Mohit Jain `[通讯]` (Microsoft Research India)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在印度城市社区开展了两阶段混合方法的情景式研究，比较了符合临床指南与当地治疗常规的医学聊天机器人对用户偏好的影响，并探究了情境感知提示的调节作用。

**💡 创新点**

创新点在于揭示临床准确性与当地可接受性之间的合法性冲突，并首次将情境感知提示作为工具，以促使用户在偏离指南的常规中重新评估机器人建议。

**🔧 技术方法**

采用情景式访谈（vignette）设计、定性主题分析、定量t检验与方差分析等统计方法，并构建了包含情境提示的聊天机器人对话。

**📊 数据集**

数据集为来自班加罗尔大学校园的200名成年人（各阶段100人），包括三种常见疾病（普通感冒、病毒性腹泻、紧张性头痛）的临床情景与对话稿。

**📈 对比分析**

通过比较两类机器人（Verity/Max 与 Clarity/Max）在偏好率上的差异，使用单样本t检验与独立样本t检验，结果显示原始对话中54%用户偏好Max，加入情境提示后偏好率飙升至85%，差异均达到统计显著水平。

**⚠️ 局限性**

局限性包括便利抽样导致的代表性偏差、仅在城市环境中测试、对真实交互的生态效度有限，以及对农村或其他文化背景的可推广性不足。

---

## 917. Lemma Discovery in Agentic Program Verification

**arXiv ID:** 2603.22114 | [PDF](https://arxiv.org/pdf/2603.22114v1)

**作者:** Huan Zhao `[一作]` (National University of Singapore), Abhik Roychoudhury `[通讯]` (National University of Singapore)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于大型语言模型的代理，能够在程序验证过程中先离线从程序语义中生成辅助引理，再在线根据证明进程的反馈动态调整这些引理，从而有效解决复杂的验证条件（VC）证明任务。

**💡 创新点**

创新点在于：①将程序语义理解与VC证明耦合，提出两阶段辅助引理发现机制——离线语义驱动的引理合成与在线反馈驱动的引理适配；②通过“语义‑证明”桥接技术，使得高层程序语义能够直接指导低层工具产生的低级VC，显著提升证明成功率。

**🔧 技术方法**

主要技术包括：GPT‑5.2‑2025‑12‑11 LLM（温度0）用于程序语义分析、语义-aware VC 生成、辅助引理合成与适配；Coq/ITP 证明助手；WP 插件作为 VC 生成器；ACSL 注解语言；自定义的适配器与维护器实现在线引理更新。

**📊 数据集**

数据集：SV‑COMP（641 个 VC，131 程序）和 NTP4VC（300 个 VC，8 个真实世界项目，如 Linux kernel、Contiki OS、C++ 标准库、X.509 解析器等），总计 941 个 VC。

**📈 对比分析**

与三款 LLM 代理（Z3‑based、Coq‑based、RAG‑based）以及非代理 SMT 基线进行对比。实验显示，代理在 SV‑COMP 上 298/641 VC，NTP4VC 上 66/300 VC，整体比最佳基线提升 26.8%–195.9%；在唯一成功 VC 数量上领先 78 个。证明成功率随 VC 复杂度与属性类型均保持稳健。

**⚠️ 局限性**

局限性：①离线合成对深度领域知识或复杂推理的 VC 可能失败；②在线适配仅依赖已有引理和证明反馈，缺乏更系统的适配策略；③实验可能受数据泄漏与版本偏差影响；④仅验证 C 语言程序，尚未验证跨语言适用性；⑤LMM 不在可信根基内，需进一步完善安全性保障。

---

## 918. SpecTM: Spectral Targeted Masking for Trustworthy Foundation Models

**arXiv ID:** 2603.22097 | [PDF](https://arxiv.org/pdf/2603.22097v1)

**作者:** Syed Usama Imtiaz `[一作]` (Florida State University), Nasrin Alamdari `[通讯]` (Florida State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出SpecTM（Spectral Targeted Masking）用于 Earth Observation 基础模型的自监督预训练，并在 NASA PACE 湖 Erie 的微毒素（microcystin）浓度预测任务上进行验证。

**💡 创新点**

① 通过领域知识设计目标掩码，只遮蔽生物光学敏感波段，迫使模型学习跨波段物理关联；② 构建多任务自监督框架（重构、光学指数推断、8 天前预测）；③ 直接从高光谱图像预测毒素浓度，无需代理变量。

**🔧 技术方法**

Vision Transformer 编码器与解码器、目标掩码策略、自监督多任务损失、迁移微调、气象特征融合与多模态学习。

**📊 数据集**

NASA PACE OCI 高光谱图像（71,320 个 8 天复合样本）与 NOAA GLERL 微毒素测量（147 个匹配样本，98 对 8 天前预测）。

**📈 对比分析**

与 26,208 个传统基线（7 算法 + 78 特征组合）在留一组交叉验证和时间拆分上比较；SpecTM 当前周 R²=0.695（+34% 相较最佳基线），8 天前 R²=0.620（+99% 相较最佳基线）；目标掩码比随机掩码提升 0.037 R²，SSL 预训练比随机初始化提升 0.18 R²。

**⚠️ 局限性**

仅在水质领域进行验证，缺乏跨域实验；模型对高光谱分辨率与气象输入高度依赖，实际应用中需考虑数据获取成本和可用性。

---

## 919. DTVI: Dual-Stage Textual and Visual Intervention for Safe Text-to-Image Generation

**arXiv ID:** 2603.22041 | [PDF](https://arxiv.org/pdf/2603.22041v1)

**作者:** Binhong Tan `[一作]` (Xidian University), Handing Wang `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出双阶段推理时干预框架 DTVI，用于抑制文本到图像模型中的不安全内容生成。

**💡 创新点**

创新点在于对全文文本嵌入进行类别感知的序列级干预，并在视觉生成阶段进一步抑制残余不安全特征，显著提升对分布式恶意语义的鲁棒性。

**🔧 技术方法**

技术包括使用线性多分类 SVM 学习不安全方向、投影与方向引导干预、以及在交叉注意力中对齐抑制残留不安全视觉特征。

**📊 数据集**

使用 I2P、SneakyPrompt、Ring-A-Bell、MMA-Diffusion、P4D 等不安全数据集以及 COCO 作为安全基准进行评估。

**📈 对比分析**

与 ESD、UCE、SafeGen、SLD、SafeRedir 等基线对比，DTVI 在七类不安全场景下平均 DSR 最高（88.56%），同时保持较好的生成质量（CLIP 30.66，FID 20.54）。

**⚠️ 局限性**

局限性包括额外推理时计算开销、全局统一干预强度导致部分安全文本质量轻微下降，以及尚未验证对视频生成等更复杂任务的适用性。

---

## 920. Future-Interactions-Aware Trajectory Prediction via Braid Theory

**arXiv ID:** 2603.22035 | [PDF](https://arxiv.org/pdf/2603.22035v1)

**作者:** Caio Azevedo `[一作]` (Stellantis), Fabien Moutarde `[通讯]` (École des Mines de Paris)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种辅助任务——布拉伊德预测（braid prediction），在多代理轨迹预测过程中并行训练，以提升对社会互动的感知与预测精度。

**💡 创新点**

创新点在于将布拉伊德理论的拓扑描述直接嵌入模型训练，既预测轨迹又预测所有代理对间的交叉标签，从而让模型在生成轨迹时自然符合真实交互行为；并提出基于布拉伊德相似度（Braid Similarity）的新评估指标。

**🔧 技术方法**

采用QCNet/ QCNeXt等基于DETR的架构，利用模式嵌入（mode embeddings）或代理编码生成边特征，加入交叉分类头；训练时使用交叉熵+轨迹回归损失，并以λ调节布拉伊德任务权重。

**📊 数据集**

在Interaction、Argoverse 2和Waymo Open Motion Dataset（WOMD）三个公开数据集上进行实验，分别覆盖不同时间窗和交互复杂度。

**📈 对比分析**

相较于原始模型，布拉伊德辅助训练在所有数据集上均提升了联合指标（MinJointFDE/MinJointADE）0.02–0.05点，WOMD上提升约0.2 m；同时Braid Similarity上提升0.1%–1%，表明模型对真实交互的遵循更好。

**⚠️ 局限性**

局限性包括：（1）仅利用布拉伊德的交叉类型而未考虑交叉时间，导致信息不完整；（2）实现依赖于DETR-like架构或需改造为使用代理编码的通用版本；（3）在极端稀疏交互场景下仍可能出现误判。

---

## 921. Set-Theoretic Types for Erlang: Theory, Implementation, and Evaluation

**arXiv ID:** 2603.22032 | [PDF](https://arxiv.org/pdf/2603.22032v1)

**作者:** Albert Schimpf `[一作]` (RPTU University Kaiserslautern-Landau), Annette Bieniusa `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于集合论子类型的 Erlang 静态类型检查器，涵盖了 Erlang 核心特性并验证了类型安全性。

**💡 创新点**

提出了完整的集合论类型系统，支持交叉、并集、取补等类型运算，并结合守卫的发生类型与完整的缺失性检查，证明类型检查可判定且系统安全。

**🔧 技术方法**

采用语义子类型、集合论类型、发生类型、约束生成与 Tally 求解算法、类型重构等技术实现。

**📊 数据集**

对 Erlang 标准库的 5 个模块、etylizer 自身的 14 个模块以及 JSONE 开源库的 4 个模块进行评测。

**📈 对比分析**

与 Dialyzer、Gradualizer、eqWAlizer、CDuce 等工具在交叉类型、并集分布式、缺失性检查等场景对比，证明本系统能捕获更多错误；性能平均每个函数数秒，最大数分钟，整体仍保持可接受。

**⚠️ 局限性**

局限包括不支持 rank‑2 多态、某些 tuple/opaque、try‑catch、部分 map comprehension、动态索引的值相关类型，以及类型重构效率低导致的 timeout 等。

---

## 922. 6D Robotic OCT Scanning of Curved Tissue Surfaces

**arXiv ID:** 2603.22012 | [PDF](https://arxiv.org/pdf/2603.22012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 923. MEVIUS2: Practical Open-Source Quadruped Robot with Sheet Metal Welding and Multimodal Perception

**arXiv ID:** 2603.22031 | [PDF](https://arxiv.org/pdf/2603.22031v1)

**作者:** Kento Kawaharazuka `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款名为MEVIUS2的金属材质开源四足机器人。

**💡 创新点**

通过板金焊接实现大型结构单一部件化，并集成多模态感知传感器，解决了开源四足机器人体积小、易损、感知不足的问题。

**🔧 技术方法**

采用板金焊接、金属切削、LiDAR（Livox Mid-360）、高动态范围摄像机、强化学习（IsaacGym）以及MuJoCo验证等技术。

**📊 数据集**

未使用公开数据集，主要使用自研训练环境与真实环境实验进行验证。

**📈 对比分析**

与Mini Cheetah、ANYMAL、Spot、Solo‑12、PAWDQ等机器人比较，MEVIUS2在尺寸、重量、最大扭矩、成本及感知能力上均优于现有开源机器人，并能在多种不整地上行走，单次续航约1小时。

**⚠️ 局限性**

局限性包括对专业焊接/加工技术的依赖、缺乏大规模生产验证以及电池续航时间有限。

---

## 924. Tuning Real-World Image Restoration at Inference: A Test-Time Scaling Paradigm for Flow Matching Models

**arXiv ID:** 2603.22027 | [PDF](https://arxiv.org/pdf/2603.22027v1)

**作者:** Purui Bai `[一作]` (Chinese Academy of Sciences), Huaibo Huang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ResFlow‑Tuner，用统一多模态融合和无训练的测试时刻缩放，在 FLUX 预训练流匹配模型上实现高质量的真实世界图像恢复

**💡 创新点**

创新点在于将多模态文本/图像条件直接拼接到流模型的 ODE 路径中，并通过基于奖励模型的动态扰动搜索实现测试时刻性能提升

**🔧 技术方法**

核心技术包括 FLUX 流匹配模型、MM‑DiT 统一序列注意力、ROPE 位置编码、多步部分去噪估计 (MSPDE) 与奖励模型集成的 TTS

**📊 数据集**

使用 DIV2K、Flickr2K、LSDIR、DIV8K、RealPhoto60、RealSR、DRealSR 以及 Occluded RoadText OCR 进行训练与评估

**📈 对比分析**

在合成与真实数据集上与多种 GAN、扩散和流模型对比，凭借无参考指标（MUSIQ、CLIPIQA+、PaQ‑2‑PiQ、MANIQA）和 OCR 准确率均取得显著领先，但在 PSNR/SSIM 上略逊

**⚠️ 局限性**

局限性包括较高的测试时刻计算开销（多轮扰动搜索），对模型超参和奖励模型选择敏感，且依赖于 FLUX 预训练模型的可扩展性

---

## 925. Retrieving Climate Change Disinformation by Narrative

**arXiv ID:** 2603.22015 | [PDF](https://arxiv.org/pdf/2603.22015v1)

**作者:** Max Upravitelev `[一作]` (Technische Universität Berlin), Vera Schmitt `[通讯]` (Technische Universität Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将气候误导信息检测转为叙事检索任务，并提出SpecFi框架来生成与叙事核心信息匹配的假设文本。

**💡 创新点**

创新点在于通过图社区摘要与假设生成两种方式提升检索鲁棒性，且提出叙事方差指标来解释检索难度。

**🔧 技术方法**

主要技术包括HyDE式假设生成、NodeRAG图谱构建与社区检测、稠密向量检索与少样本提示。

**📊 数据集**

使用了三大气候误导数据集：CARDS、CO（Climate Obstruction）以及PolyNarrative的气候子集PN-CC。

**📈 对比分析**

与BM25、稠密检索以及静态标记检索等基线相比，SpecFi‑CS在CARDS上取得最高MAP（0.505），在CO上也优于稠密基线，性能相对稳定。

**⚠️ 局限性**

局限性包括对OpenAI模型的依赖、假设生成数量未系统优化、可能受训练数据泄漏影响以及对高运行时开销的考量。

---

## 926. From Technical Debt to Cognitive and Intent Debt: Rethinking Software Health in the Age of AI

**arXiv ID:** 2603.22106 | [PDF](https://arxiv.org/pdf/2603.22106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 927. On the Challenges and Opportunities of Learned Sparse Retrieval for Code

**arXiv ID:** 2603.22008 | [PDF](https://arxiv.org/pdf/2603.22008v1)

**作者:** Simon Lupart `[一作]` (NAVER LABS Europe), Stéphane Clinchant `[通讯]` (NAVER LABS Europe)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了SPLADE-Code，一种专门针对代码检索的学习式稀疏检索模型；

**💡 创新点**

首次将SPLADE稀疏检索技术大规模应用于代码检索，并通过单阶段训练实现高效、可解释的检索；

**🔧 技术方法**

采用SPLADE架构、稀疏向量投影、最大池化、KLD蒸馏、LoRA微调、双向注意力、倒排索引（Seismic）等技术；

**📊 数据集**

使用CoIR、MTEB‑Code、CodeRAG、CPRet等多语言、多任务代码检索基准；

**📈 对比分析**

与多种密集检索基线（C2‑LLM、CodeXEmbed、CodeR）及其他稀疏/传统方法（BM25、SPLADE‑lexical、SPLARE）对比，nDCG@10在同类模型中表现优异（0.6B版75.4，8B版79.0），在跨域基准上提升5–7点；检索延迟低于1 ms，显示出优良的效果–效率平衡；

**⚠️ 局限性**

局限性包括：基准数据未完全覆盖真实开发者查询场景、仅评估检索层面而未考虑重排序/生成的端到端影响、以及密集检索与稀疏检索延迟测量的直接可比性不足。

---

## 928. SegMaFormer: A Hybrid State-Space and Transformer Model for Efficient Segmentation

**arXiv ID:** 2603.22002 | [PDF](https://arxiv.org/pdf/2603.22002v1)

**作者:** Duy D. Nguyen `[一作]` (Ho Chi Minh City University Of Technology), Phat T. Tran-Truong `[通讯]` (Vietnam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并实现了一种轻量级混合 Mamba 与 Transformer 的三维医学图像分割模型 SegMaFormer。

**💡 创新点**

创新点在于：早期高分辨率阶段使用 Mamba 状态空间层降低计算量，后期低分辨率阶段使用自注意力；并结合 3D 旋转位置嵌入，兼顾全局长程依赖和局部空间感知。

**🔧 技术方法**

采用的核心技术包括：Mamba 状态空间模型、Transformer 自注意力、3D Patch Embedding、3D Rotary Position Embedding、Mlp 解码器以及可选的深度监督。

**📊 数据集**

使用的公开数据集为：BraTS（脑肿瘤）、Synapse（多器官 CT）和 ACDC（心脏）。

**📈 对比分析**

在与 nnFormer、SegFormer3D、UNETR 等 SOTA 模型的同等训练设置下进行对比，SegMaFormer 仅 2M 参数、15 GFLOPs，却在 Dice 分数上与大型模型持平甚至优于对手。

**⚠️ 局限性**

局限性包括：对小器官和细粒度边界的分割性能仍有提升空间，模型在极小结构上的细节捕捉不足。

---

## 929. Asymptotically Ideal Conjunctive Hierarchical Secret Sharing Scheme Based on CRT for Polynomial Ring

**arXiv ID:** 2603.22001 | [PDF](https://arxiv.org/pdf/2603.22001v1)

**作者:** Jian Ding `[一作]`, Haifeng Yu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出一种基于多项式环Chinese Remainder Theorem与一向函数的渐近理想级联层级秘密共享方案；

**💡 创新点**

实现信息速率为1、计算安全且可灵活设置共享份额大小，填补了先前CRT‑基础CHSS方案在安全性和速率上的不足；

**🔧 技术方法**

采用多项式环CRT、随机多项式生成、层级阈值构造以及多向哈希函数的组合；

**📊 数据集**

无需具体数据集，方案完全在理论与数学分析层面验证；

**📈 对比分析**

与现有CHSS方案（如Tassa、Gyarmati、Tiplea等）在表1中比较，证明在计算安全、渐近理想性和信息速率方面优于或等价于前人方案；

**⚠️ 局限性**

需要大量公开多项式和哈希值，计算复杂度相对较高，且对参数选择有严格要求，未来工作需进一步减少公开值和优化效率。

---

## 930. Online Packing of Orthogonal Polygons

**arXiv ID:** 2603.22098 | [PDF](https://arxiv.org/pdf/2603.22098v1)

**作者:** Tim Gerlach `[一作]` (University of Hamburg), Linda Kleist `[通讯]` (University of Hamburg)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

论文研究正交多边形（尤其是正交6‑边形）在在线装箱、条带、周长和面积装箱等变体中的竞争比，并给出其最优竞争比与常数算法的界限；同时扩展到更高复杂度的正交8‑边形和骨架形状，证明其竞争比等于n，进一步阐明在线正交多边形装箱的极限。

**💡 创新点**

创新点主要包括：
1) 首次给出一般正交6‑边形在线装箱的下界 Ω(n/ log n)，并证明当形状为对称或小型时可以实现常数竞争比；
2) 将在线装箱问题与排序、调度、区间着色等经典问题建立新的归约，利用这些领域的最优算法得到新的在线装箱策略；
3) 对正交8‑边形和骨架形状给出最强下界，证明任意在线算法竞争比至少为 n；
4) 在条带、周长、面积装箱以及在线临界密度方面给出统一的上下界分析，展示正交多边形在不同度量上的复杂度差异。

**🔧 技术方法**

主要技术手段包括：
- 通过构造的 L‑形/ Z‑形序列与在线排序/区间调度问题的映射实现下界证明；
- 使用在线颜色化算法（Kierstead‑Trotter）对区间图进行着色，从而在对称大 L‑形的装箱中实现 33‑竞争比；
- 利用 FirstFit、NiceRectanglePacker 等经典 1‑维/2‑维装箱算法实现常数竞争比；
- 对骨架形状和 Z‑形进行几何分割、递归填充，保证可装箱的同时保持足够的间距；
- 通过对装箱箱子分层、水平/垂直分区实现对条带、周长、面积等目标的优化。

**📊 数据集**

论文为理论研究，未使用实际数据集；所有结果均来自严格的数学证明与算法分析。

**📈 对比分析**

比较方法主要是通过竞争比的理论界定：给出最优下界（Ω(n/ log n)、Ω(n)）以及对应的常数上界（如 33‑, 8‑, 41‑, 2‑ 竞争比）。性能评价基于与最优离线解的比值，证明了在多种装箱变体中算法的最坏情况表现；与已有的矩形或凸多边形装箱结果进行对比，强调正交多边形在在线装箱中的更高复杂度。

**⚠️ 局限性**

局限与开放问题：
- 对一般正交6‑边形的最佳竞争比仍未知，当前下界和上界之间仍有显著差距；
- 对更高复杂度正交多边形的竞争比只给出了最强下界，尚无有效的常数上界；
- 所有结果均基于纯理论分析，缺乏实验验证；
- 仅考虑平移操作，旋转（多倍 90°）对竞争比的影响仍未充分探讨；
- 对在线临界密度的上界和下界在不同 arm 长度区间内仍有较大差距。

---

## 931. AnimalCLAP: Taxonomy-Aware Language-Audio Pretraining for Species Recognition and Trait Inference

**arXiv ID:** 2603.22053 | [PDF](https://arxiv.org/pdf/2603.22053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 932. Bounded Structural Model Finding with Symbolic Data Constraints

**arXiv ID:** 2603.22093 | [PDF](https://arxiv.org/pdf/2603.22093v1)

**作者:** Artur Boronat `[一作]` `[通讯]` (University of Leicester), Artur Boronat (University of Leicester)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Maude 重写逻辑框架中实现了一个名为 MMF（Maude Model Finder）的本地化边界模型发现工具，能够在给定类与角色边界、结构与数据约束的前提下自动生成满足这些约束的有限对象图。

**💡 创新点**

创新点包括：① 将符号可达性技术重构为结构构造计算，采用义务驱动的两阶段生成（对象分配与引用赋值）实现边界控制；② 通过基于 ACU 匹配和形状索引的折叠（subsumption）实现对搜索空间的语义层面冗余消除；③ 只在需要时调用 SMT 求解器，用于约束满足性检查和子包含判定，避免了传统 SAT/SMT 译码的语义鸿沟。

**🔧 技术方法**

使用技术主要有：符号重写系统、ACU 及其他等价理论的归一化、形状索引（shape indexing）、SMT 约束求解（用于可满足性和子包含判定）、以及 Maude 的反射机制（reflective rewrite）。

**📊 数据集**

实验数据集为作者构造的 CEO 组织模式基准（Company–Employee–Project），该基准包含结构约束（如 CEO 无经理、管理关系无环）以及符号整数属性约束，边界范围从 1 公司、2 员工、0–2 项目扩展到最多 5 名员工。

**📈 对比分析**

与基线配置（启用 SMT 剪枝、折叠与形状索引）相比，MMF 在完整搜索模式下能消除约 70% 的状态，SMT 剪枝是最主要的效益；在单个模型搜索（findFirst）模式下，尽管每个状态开销更大，但仍保持可接受的执行时间。折叠与索引的组合对规模较大的边界（如 5 名员工）带来明显加速，单独关闭 SMT 或折叠会导致明显的性能退化甚至超时。

**⚠️ 局限性**

限制主要包括：① 目前仅支持固定范围内的边界模型发现，未实现动态或无限域的生成；② 依赖于形状索引与 SMT 推理，若约束语言或符号复杂度过高会导致折叠和求解效率低下；③ 对于缺乏明显对称结构的规格，折叠效果有限；④ 只覆盖 OCL 的子集，尚未支持更完整的 MDE 约束。

---

## 933. Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models

**arXiv ID:** 2603.22042 | [PDF](https://arxiv.org/pdf/2603.22042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 934. A Context Engineering Framework for Improving Enterprise AI Agents based on Digital-Twin MDP

**arXiv ID:** 2603.22083 | [PDF](https://arxiv.org/pdf/2603.22083v1)

**作者:** Xi Yang `[一作]` (IBM Software Innovation Lab), Daby M. Sow `[通讯]` (IBM Software Innovation Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于数字孪生马尔可夫决策过程（DT‑MDP）的框架，利用离线强化学习和对比逆强化学习提升企业级LLM代理的推理与决策能力。

**💡 创新点**

创新点包括：①将LLM推理过程抽象为有限状态与动作的DT‑MDP，显著降低样本复杂度；②引入鲁棒的对比逆强化学习，从混合质量轨迹中自动学习奖励函数；③通过离线学习得到的策略指导上下文工程（prompt/intervention），实现无在线交互的性能提升。

**🔧 技术方法**

核心技术有：数字孪生MDP抽象、对比逆强化学习（T‑REX）、离线强化学习（CQL、BC）、离线策略评估（Fitted Q‑Evaluation）、基于策略的上下文工程（prompt建议、剪枝、优先级排序）。

**📊 数据集**

使用了ITBench基准中的12个SRE诊断场景（Flagd、Chaos Mesh、定制故障）生成的819条轨迹，并在6个未见测试场景上进行在线评估；实验涵盖多种LLM（Mistral‑Medium‑2505、GPT‑4o等）。

**📈 对比分析**

与基线EoG、ReAct等模型对比，DT‑MDP‑CE在Pass@3 Recall/F1上平均提升5–15%，并通过配对t检验和Nemenyi CD图表明差异显著；成本（token/时间）仅略高，部分策略甚至降低成本；在不同模型规模、领域（SWE）和特征扩展下仍保持正向提升。

**⚠️ 局限性**

局限性包括：①依赖足够质量的离线轨迹，若轨迹极少或无标注则效果受限；②DT‑MDP的抽象需领域知识，抽象误差会影响性能；③对完全开放的高维文本空间仍有局限，无法完全替代在线自适应学习；④在极大模型或非结构化任务中的增益相对有限。

---

## 935. Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch

**arXiv ID:** 2603.22056 | [PDF](https://arxiv.org/pdf/2603.22056v1)

**作者:** Stella Eva Tsiapali `[一作]`, Kate Knill `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统分析了DSKD-CMA的跨模型注意力机制，并提出了DSKD-CMA-GA通过生成对抗关键-查询匹配来提升跨tokenizer的知识蒸馏效果。

**💡 创新点**

创新点在于：①用手工chunk对齐与热图可视化揭示CMA的隐式对齐特点；②引入生成对抗学习对键值分布进行匹配，显著缓解不同tokenizer导致的分布不匹配问题。

**🔧 技术方法**

使用了手工chunk对齐、热图可视化、生成对抗学习、条件输运、六种f‑divergence损失函数等技术。

**📊 数据集**

实验数据集包括Dolly 15K、Self‑Instruct、Vicuna‑Eval、Super‑Natural Instructions、Unnatural Instructions等。

**📈 对比分析**

与原DSKD-CMA、MinED、ULD、SFT以及同tokenizer KD进行对比，ROUGE‑L平均提升约0.3–0.5点，跨tokenizer时与同tokenizer的性能差距缩小至不足1点。

**⚠️ 局限性**

局限性：CMA的对齐仍局部化，细粒度注意力（CLA）表现不佳；GA对f‑divergence选择敏感；尚未在更复杂的推理任务和实际下游任务上进行评估。

---

## 936. GTSR: Subsurface Scattering Awared 3D Gaussians for Translucent Surface Reconstruction

**arXiv ID:** 2603.22036 | [PDF](https://arxiv.org/pdf/2603.22036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 937. The Descriptive Complexity of Relation Modification Problems

**arXiv ID:** 2603.22043 | [PDF](https://arxiv.org/pdf/2603.22043v1)

**作者:** Florian Chudigiewitsch `[一作]` (Universität zu Lübeck), Till Tantau `[通讯]` (Universität zu Lübeck)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

对关系修改问题（对结构进行元问题的编辑、添加或删除）在不同逻辑结构类型（单射、基本图、无向图、有向图、任意结构）和不同量化模式（ae、e*a、aa等）下的经典与参数化复杂度进行完整分类，揭示了它们的可解性与不可解性极限

**💡 创新点**

首次将量化模式与结构类型共同纳入分析，证明了不同模式下问题的强二分（要么在AC^0↑/TC^0/AC^0内可解，要么为W[2]-hard或NP-hard），并指出所有三种修改操作的复杂度相同；同时发现对基本图的特定模式（如ae）可解，而对无向图和任意结构则不可解；提出了新的W[2]-hard问题（如边加/编辑到半径r）

**🔧 技术方法**

利用一阶逻辑的量化前缀分析、量化模式归约、构造性AC^0/TC^0电路、参数化电路类AC^0↑、归约证明（如Set-Cover、Vertex-Cover、Majority）以及对类型图的直觉式分析

**📊 数据集**

无（该工作为纯理论研究，未使用数据集）

**📈 对比分析**

无（无实验对比，所有结果均为理论证明）

**⚠️ 局限性**

主要局限在于仅处理关系结构，未扩展到含函数符号的词汇；未探究更高阶逻辑（MSO等）或其他图操作（收缩、翻转）所导致的复杂度；对可解模式e*a的更细粒度复杂度（是否可降至AC^0）仍未确定

---

## 938. Do World Action Models Generalize Better than VLAs? A Robustness Study

**arXiv ID:** 2603.22078 | [PDF](https://arxiv.org/pdf/2603.22078v1)

**作者:** Zhanguang Zhang `[一作]` (Huawei Technologies), Yingxue Zhang `[通讯]` (Huawei Technologies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比分析了基于世界模型的行动模型（WAM）与基于视觉-语言-行动的模型（VLA），在两大基准（RoboTwin 2.0‑Plus 与 LIBERO‑Plus）上对多种视觉与语言扰动的鲁棒性进行系统评测。

**💡 创新点**

提出了首个系统性的比较框架，揭示了 WAM 在噪声、光照与布局扰动下的显著鲁棒性优势、VLA 在多样化数据下的可弥补差距以及混合模型的中间性能，为后续研究提供了实验基准与设计启示。

**🔧 技术方法**

使用了视觉‑语言预训练模型（如 Cosmos‑Predict2、SVD）、视频扩散与流匹配网络、自动回归解码器以及联合去噪/动作生成的训练策略，并对比了不同的推理优化技术。

**📊 数据集**

实验基准包括：RoboTwin 2.0‑Plus（双臂 Aloha‑Agilex 系统）与 LIBERO‑Plus（单臂 Franka Panda）以及各自的原始数据集；并在这些数据集上对多种扰动类型（相机、机器人初始状态、语言、光照、背景、噪声、布局）进行评估。

**📈 对比分析**

评估方法为在同一模型架构下统一测试多种扰动，比较成功率与鲁棒性。结果显示：WAM（LingBot‑VA、Cosmos‑Policy）在两套基准上分别达 74.2% 与 82.2% 的整体成功率，远优于 VLA（π_0.5 在 RoboTwin 仅 58.6%，在 LIBERO 85.7%）。混合模型（MOTUS、VLA‑JEPA）表现介于两者之间。

**⚠️ 局限性**

主要局限包括：WAM 的推理时间显著慢于 VLA（至少 4.8 倍），对相机视角与机器人初始状态扰动仍缺乏鲁棒性；同时 WAM 的训练与推理成本高，限制了其在实时部署中的应用。

---

## 939. SpatialBoost: Enhancing Visual Representation through Language-Guided Reasoning

**arXiv ID:** 2603.22057 | [PDF](https://arxiv.org/pdf/2603.22057v1)

**作者:** Byungwoo Jeon `[一作]` (KAIST), Jinwoo Shin `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SpatialBoost 框架，通过语言引导推理将 3D 空间知识注入预训练视觉编码器，从而提升其空间感知能力。

**💡 创新点**

创新点在于：①利用大语言模型（LLM）对 3D 空间信息进行语言化表达，并构造多层级多轮视觉问答（VQA）数据；②引入双通道注意力机制，仅更新新注意力层和混合因子，避免破坏原始知识；③在对齐、指令微调和编码器微调三个阶段系统化地训练。

**🔧 技术方法**

核心技术包括：多视角/单视角 3D 重建与深度估计、文本化空间描述、LLM 对话推理（Chain‑of‑Thought）、投影器对齐、双通道注意力微调、以及大规模 VQA 数据生成。

**📊 数据集**

使用的数据集涵盖：SA1B、Ego‑centric 视频/3D 数据集（生成多视角 VQA）；ADE20K、Pascal VOC、NYUd、KITTI（稠密预测）；Lexicon3D、ScanQA、SQA3D（3D 语义/几何任务）；CortexBench（机器人控制）；ImageNet‑1K、Oxford/Paris/Met/Amsterdam（分类/检索）。

**📈 对比分析**

与 DINOv3、OpenCLIP、SigLIP、DINOv2 等最新预训练编码器相比，SpatialBoost 在 ADE20K 语义分割上 mIoU 从 55.9% 提升至 59.7%；在 Lexicon3D SQA3D 上提升至 54.9%；在 NYUd 深度估计中 RMSE 下降至 0.39；在机器人控制任务中显著提升各域性能；在 ImageNet 线性分类中准确率从 88.4% 提升至 90.2%。

**⚠️ 局限性**

局限性包括：依赖高质量的 LLM 与大量文本生成，训练成本高；对极其复杂或缺乏多视角信息的场景仍可能不足；双通道注意力虽防止遗忘，但可能无法完全融合所有原始特征；缺乏对实时推理或极端低资源环境的评估。

---

## 940. Mixture of Mini Experts: Overcoming the Linear Layer Bottleneck in Multiple Instance Learning

**arXiv ID:** 2603.22198 | [PDF](https://arxiv.org/pdf/2603.22198v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 941. Do Papers Match Code? A Benchmark and Framework for Paper-Code Consistency Detection in Bioinformatics Software

**arXiv ID:** 2603.22018 | [PDF](https://arxiv.org/pdf/2603.22018v1)

**作者:** Tianxiang Xu `[一作]` (Xi'an Jiaotong University), Jiayin Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究论文与对应软件实现的一致性检测任务，构建 BioCon 基准并提出跨模态检测框架。

**💡 创新点**

首次提出论文-代码一致性检测任务及 BioCon 数据集，并结合硬负样本与加权 Focal Loss 解决类别不平衡。

**🔧 技术方法**

采用预训练代码语言模型 UniXcoder（亦可使用 CodeBERT、CodeT5+ 等）进行联合编码，使用加权 Focal Loss 进行训练。

**📊 数据集**

BioCon，48 个生物信息学软件项目及其对应论文中的句子-函数对。

**📈 对比分析**

与 CodeBERT、CodeT5+、CodeGen 等对比，UniXcoder 在 BioCon 上获得 Acc 0.9056、F1 0.8011、MCC 0.6239，性能最优。

**⚠️ 局限性**

任务仍受一对多映射、论文内容噪声和数据集规模限制，难以完整覆盖多层次一致性细粒度。

---

## 942. Navigational Thinking as an Emerging Paradigm of Computer Science in the Age of Generative AI

**arXiv ID:** 2603.22133 | [PDF](https://arxiv.org/pdf/2603.22133v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 943. SpatialReward: Verifiable Spatial Reward Modeling for Fine-Grained Spatial Consistency in Text-to-Image Generation

**arXiv ID:** 2603.22228 | [PDF](https://arxiv.org/pdf/2603.22228v1)

**作者:** Sashuai Zhou `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SpatialReward 套件与 SpatRelBench 基准，利用强化学习提升文本到图像模型的细粒度空间一致性。

**💡 创新点**

创新点在于 Prompt Decomposer 提取结构化约束，专家检测实现可验证视觉信息，随后通过 vision‑language 模型的 chain‑of‑thought 推理，实现针对复杂空间关系的可验证奖励模型。

**🔧 技术方法**

采用 Qwen2.5‑VL‑7B 进行提示分解、开源目标检测与 OCR、CLIP‑based 分类、单目深度估计、VLM CoT 推理、GRPO‑based Flow‑GRPO RL 以及 LoRA 参数高效调优。

**📊 数据集**

训练与评测数据包括约 100k 的 Prompt‑Metadata 语料、SpatRelBench 约 2000 条标注，以及 Stable Diffusion3.5‑M 与 FLUX1‑dev 的生成图像，评估指标涵盖 GenEval、SpatRelBench、T2I‑CompBench 等。

**📈 对比分析**

与 TextOCR、PickScore、Qwen2.5‑VL、ImageReward、UnifiedReward 等基准在 GenEval 与 SpatRelBench 上比较，SpatialReward 在空间一致性分数提升 0.1–0.2 分，显著优于基线，同时在 Wise、DPG、PickScore 等整体质量指标保持或略有提升。

**⚠️ 局限性**

局限性包括对专家检测模型与硬件资源的高依赖，Prompt Decomposer 的误拆解会影响奖励准确性，对极其复杂或多层关系的推理仍存在误差，且当前评测侧重 2D 与简化 3D，未覆盖更复杂的动态场景。

---

## 944. Accelerating Fresh Data Exploration with Fluid ETL Pipelines

**arXiv ID:** 2603.22220 | [PDF](https://arxiv.org/pdf/2603.22220v1)

**作者:** Maxwell Norfolk `[一作]` (Pennsylvania State University), Dong Xie `[通讯]` (Pennsylvania State University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了流式 ETL 管道（Fluid ETL Pipeline），允许在不中断数据摄取的情况下按需启动或停止任意数据预处理例程（DPR），从而支持新鲜数据探索。

**💡 创新点**

创新点在于：①利用闲置或低价抢占资源（如 Spot 实例）进行预判性 DPR 执行；②构建可动态调整的 Fluid ETL，支持按需投放、回收 DPR；③引入“查询拼接”（plan stitching）技术，充分利用部分已构建的数据结构；④提供 DPR 生成、成本估算与自适应调度的整体框架。

**🔧 技术方法**

使用技术包括：Kafka 事件流 + Spark/Flink 传统 ETL；中间件调度与映射；MLIR 进行 DPR 共享与融合；LLM 预测未来查询；成本与收益建模；基于背包优化与控制理论的 DPR 选取算法；查询重写与拼接技术。

**📊 数据集**

实验使用 GitHub Archive 事件日志（JSON 形式），模拟一天的数据流进行探索性查询。

**📈 对比分析**

与基线 ETL（无额外 DPR）和过度 ETL（预执行所有可能 DPR）对比，Fluid ETL 在保持摄取吞吐量不变的前提下，平均查询延迟提升 3 倍以上；过度 ETL 则需要 2 倍以上 CPU 资源来实现相同性能。

**⚠️ 局限性**

局限性：DPR 生成、成本估算和查询拼接的自动化仍待实现；对多用户共享兴趣的情况支持不足；实验规模受限于单一数据集；缺乏大规模云环境中的实测结果。

---

## 945. Cross-Modal Reinforcement Learning for Navigation with Degraded Depth Measurements

**arXiv ID:** 2603.22182 | [PDF](https://arxiv.org/pdf/2603.22182v1)

**作者:** Omkar Sawant `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了跨模态 Wasserstein 自编码器与强化学习策略相结合的框架，在深度传感器失效时利用灰度图完成无人机的无碰撞导航。

**💡 创新点**

创新点在于通过共享潜在空间学习深度与灰度的跨模态对应关系，并使用 Wasserstein 正则化提升鲁棒性，使得在深度缺失时能从灰度图重建几何特征。

**🔧 技术方法**

采用 Cross‑Modal Wasserstein Autoencoder (CMWAE)、Proximal Policy Optimization (PPO) 强化学习、灰度+深度联合编码以及深度遮挡数据增强等技术。

**📊 数据集**

训练与评估数据来源于 TartanAir 机器人室内轨迹数据集。

**📈 对比分析**

与单模态 DCE‑RL 基线比较；在无损坏环境下 DCE 更优，而在深度被人为遮挡时 CMWAE‑RL 的成功率可达 70‑80%，DCE 低于 10%；在真实四旋翼实验中，在 25‑30% 深度损坏下仍能按时到达目标，平均完成时间约 21 秒。

**⚠️ 局限性**

局限性包括：潜在维度共享导致单模态表达能力下降；对极端深度损坏（>50%）的鲁棒性有限；未充分探索与其它模态（RGB、IMU）融合；实验仅在室内/通道环境，未验证大规模开放场景。

---

## 946. StreamingClaw Technical Report

**arXiv ID:** 2603.22120 | [PDF](https://arxiv.org/pdf/2603.22120v1)

**作者:** Jiawei Chen `[一作]` (Li Auto Inc), Yufei Zheng `[通讯]` (Li Auto Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了StreamingClaw，一套面向实时流视频理解与具身智能的统一代理框架，集成了实时感知、决策、动作闭环以及多模态长期记忆与主动交互；

**💡 创新点**

核心创新点包括：①基于多代理协作的主‑子代理架构，实现实时流式推理与任务分解；②多模态层级记忆演化机制与高并发检索，解决长期记忆碎片化与检索效率问题；③主动交互代理支持时间感知与事件驱动的实时预警；④与OpenClaw兼容，可扩展工具与技能库，直接驱动具身行为；

**🔧 技术方法**

采用的技术包括：流式KV缓存与动态滑动窗口、Transformer注意力裁剪、视频切割与细粒度分割工具、命令驱动的检索策略、基于视觉‑语言触发器的主动适配（训练‑free 与训练‑based）、多模态基础模型（如Qwen3/VL、MindGPT‑4ov）、RL/FT强化工具使用、分层记忆写入与演化算法；

**📊 数据集**

论文未公开具体数据集，使用公开多模态视频与文本数据（例如公开的长视频对话与事件标注集合）进行实验；

**📈 对比分析**

实验对比了传统离线视频推理、KV缓存与无缓存方案，展示了低延迟（<50 ms）与高准确度的主动交互与记忆检索；在多任务（问答、事件检测、主动预警）中表现优于基线方法；

**⚠️ 局限性**

局限性包括：目前仅支持vision+text输入，音频输入与音视频同步对齐能力有限；缺乏端到端的跨模态联合推理与生成；对极端长时序建模与复杂场景的实时性仍有挑战；

---

## 947. Evaluating the Reliability and Fidelity of Automated Judgment Systems of Large Language Models

**arXiv ID:** 2603.22214 | [PDF](https://arxiv.org/pdf/2603.22214v1)

**作者:** Tom Biskupski `[一作]` (Hochschule der Medien), Stephan Kleber `[通讯]` (Ravensburg-Weingarten University of Applied Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究评估了将大型语言模型用作评审者（LLM-as-a-judge）来自动判定其他模型输出质量与安全性的可行性。

**💡 创新点**

创新点在于系统化比较37种不同规模对话模型与5种评审提示，覆盖8类评审任务，并对解释质量、二级评审、专用评审模型进行深入分析。

**🔧 技术方法**

采用提示工程、结构化JSON输出、温度控制、F1、百分比一致性、Fleiss κ等技术实现自动评判。

**📊 数据集**

使用来自公开基准的8个数据集（如Harmful Behavior、TruthfulQA等）和自制补充集，共计534条标注实例。

**📈 对比分析**

通过F1得分、百分比一致性等指标比较，发现GPT‑4o在所有提示下最高（F1≈0.96），大型开源模型如Qwen2.5 14B、Gemma2 9B等亦能达到较高准确率，二级评审普遍不提升。

**⚠️ 局限性**

局限性包括单轮英文二分类任务、样本量有限、API内容过滤导致部分模型无法评估、对多轮/多语言或细粒度评判的适用性未验证。

---

## 948. Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement

**arXiv ID:** 2603.22187 | [PDF](https://arxiv.org/pdf/2603.22187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 949. SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection

**arXiv ID:** 2603.22213 | [PDF](https://arxiv.org/pdf/2603.22213v1)

**作者:** Kexian Tang `[一作]` (Tsinghua University), Kaifeng Lyu `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SPA（Scaling Prompt-engineered Augmentation）方法，利用少量人类设计的七个提示模板在LLM上反复生成大规模合成语料，用于知识注入。

**💡 创新点**

创新点在于把认知心理学的学习策略（概念学习、批判性思维、生成学习）转化为提示模板，并证明单阶段提示在大规模时能优于复杂多阶段或RL方法，凸显规模化与提示设计的组合效应。

**🔧 技术方法**

技术包括提示工程的LLM数据增强、单阶段合成、持续预训练、token匹配实验以及多维度多样性评估。

**📊 数据集**

使用三大基准数据集：SQuAD（Wiki QA）、QuALITY（长文档阅读）、MultiHop‑RAG（多跳推理），并在每个基准上采用对应的生成模型（如Qwen2.5‑7B、gpt‑oss‑120b、GPT‑4o‑mini）。

**📈 对比分析**

在同等token预算下与SEAL、PaST、EntiGraph、Active Reading等方法对比，SPA在SQuAD达到91.27%（高于90.25%/74.23%），在QuALITY达到57.03%（高于56.22%/51.13%），在MultiHop‑RAG达到86.64%/88.36%（高于其他方法），显示规模化提示增强效果显著。

**⚠️ 局限性**

局限性包括：对不同任务仍需手工微调提示集合；在极大规模下可能面临生成质量与多样性瓶颈；未解决RL方法的多样性坍塌问题，只提供基线。

---

## 950. Chimera: Latency- and Performance-Aware Multi-agent Serving for Heterogeneous LLMs

**arXiv ID:** 2603.22206 | [PDF](https://arxiv.org/pdf/2603.22206v1)

**作者:** Kangqi Ni `[一作]` (University of North Carolina), Tianlong Chen `[通讯]` (University of North Carolina)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对异构LLM集群上的多代理工作流，设计并实现了一套预测调度系统，能够同时优化工作流的端到端延迟与任务性能。

**💡 创新点**

创新点在于将语义路由、工作流级输出长度预测和实时负载监测三者融合为一体化调度策略；通过“语义+长度+负载”协同决策实现对模型选择与队列优先级的自适应优化，并引入抗饿机制保持公平。

**🔧 技术方法**

核心技术包括：ModernBERT‑based 语义路由器、Quantile Random Forest 预测器、活动监视器（基于预测token量的负载估计）、基于STJF的优先级调度与老化提升机制；整个系统在vLLM之上实现。

**📊 数据集**

实验数据集主要有代码生成任务的APPS和数学推理任务的MATH，用于评估不同模型规模与配置下的工作流性能。

**📈 对比分析**

与vLLM、MLFQ和LTR等基线比较，平均可实现1.2–2.4倍的端到端延迟降低和8.0–9.5个百分点的任务性能提升；在多种模型组合与负载水平下均保持了更优的Pareto前沿。

**⚠️ 局限性**

局限性包括：仍存在长度预测与路由误差导致的性能余地（oracle实验表明有改进空间）；系统目前仅在有限的工作流结构与模型规模上验证，复杂工作流或更大集群的适配尚未深入；对不同任务类别的泛化性待进一步评估。

---

## 951. A Backbone Benchmarking Study on Self-supervised Learning as a Auxiliary Task with Texture-based Local Descriptors for Face Analysis

**arXiv ID:** 2603.22190 | [PDF](https://arxiv.org/pdf/2603.22190v1)

**作者:** Shukesh Reddy `[一作]` (BITS Pilani), Abhijit Das `[通讯]` (BITS Pilani)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对不同Vision Transformer（ViT-B、ViT-L、ViT-H）在结合局部纹理特征（LDP）与自监督辅助任务（L-SSAT）下进行面部分析任务（深伪检测、属性识别、情绪识别）的系统基准评估。

**💡 创新点**

创新点在于将局部纹理描述符直接嵌入MAE自监督任务，以重建纹理特征；并通过多种遮挡、重建、分类组合对比，系统揭示了不同backbone在多任务下的表现差异，证明无统一最优backbone。

**🔧 技术方法**

技术包括：局部方向模式（LDP）特征提取、Masked AutoEncoder（MAE）自监督重建、Vision Transformer编码器、联合分类与重建损失的多任务学习。

**📊 数据集**

使用公开基准数据集：FaceForensics++（深伪检测）、CelebA（属性识别）、AffectNet（情绪识别）。

**📈 对比分析**

方法通过在三种backbone上分别实验多种（M,R,C）配置，评估分类准确率；结果显示ViT-H在深伪检测上平均准确率≈0.94，ViT-L在属性识别上平均≈0.85，ViT-H在情绪识别上平均≈0.80，表明backbone深度对不同任务有显著影响。

**⚠️ 局限性**

局限性包括：未发现通用最优backbone；更深模型成本高且易过拟合；轻量级backbone在部分任务表现不佳；需针对任务与数据规模做进一步调优。

---

## 952. Feasibility of Augmented Reality-Guided Robotic Ultrasound with Cone-Beam CT Integration for Spine Procedures

**arXiv ID:** 2603.22174 | [PDF](https://arxiv.org/pdf/2603.22174v1)

**作者:** Tianyu Song `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一套基于光学透视增强现实（OST-AR）与机器人超声相结合的脊柱穿刺导航系统；

**💡 创新点**

创新点在于将CBCT生成的三维脊柱模型与实时机器人超声进行联合标定，并在AR头戴设备中实时叠加显示，实现全局解剖上下文与局部实时成像的无缝融合；

**🔧 技术方法**

采用机器人控制的超声探头、CBCT预采集与手眼标定、点云ICP注册、射线投射（raycasting）进行穿刺路径预测，并通过HoloLens 2实现现场可视化；

**📊 数据集**

使用自制的腰椎仿生模型（phantom）配合水耦合进行实验，不含真实人体数据；

**📈 对比分析**

在16名参与者的对照实验中，AR条件相较于传统屏幕显示，平均穿刺时间下降49%，定位误差下降25%，并在亚任务中对腰椎穿刺的误差进一步下降32%，受试者主观评分显示可用性、信任度、空间理解显著提升；

**⚠️ 局限性**

局限包括实验仅在无组织变形的仿真模型上验证，未考虑呼吸、组织压缩等临床真实因素，且点云配准误差仍达约9.1 mm，需进一步提高配准精度与实现更通用的临床部署。

---

## 953. ACPO: Counteracting Likelihood Displacement in Vision-Language Alignment with Asymmetric Constraints

**arXiv ID:** 2603.22165 | [PDF](https://arxiv.org/pdf/2603.22165v1)

**作者:** Kaili Huang `[一作]` (SenseTime Research), Lewei Lu `[通讯]` (SenseTime Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了Asymmetric Constrained Preference Optimization（ACPO）算法，用于解决在视觉‑语言模型（VLM）中Direct Preference Optimization（DPO）所导致的Likelihood Displacement（即视觉锚点崩塌）问题。

**💡 创新点**

创新点包括：①提出长度自适应优势目标（Length‑Adaptive Advantage Target），动态匹配每个偏好样本的复杂度；②引入不对称梯度缩放系数并使用stop‑gradient，使拒绝（rejected）分布的梯度被自适应抑制而保留所选（chosen）分布的稳定锚点；③将上述两项结合形成单侧锚定的对比损失，从而显著抑制视觉锚点崩塌并提升对视觉信息的保持。

**🔧 技术方法**

技术手段涵盖：DPO、β‑DPO、SimPO、IPO、DPO‑Shift等对比优化方法；对InternVL3‑14B/8B基线模型进行冻结视觉编码器、对语言模块进行微调；实现长度自适应优势目标与不对称缩放系数的动态计算；使用stop‑gradient在梯度反向传播中隔离缩放系数；在训练时采用FlashAttention、tensor/sequence并行和cosine学习率退火。

**📊 数据集**

使用内部构建的约32万条偏好对（含视觉锚对比、规则抽样、格式遵从三类）进行模型微调；评测基准包括HallusionBench、MM‑IFEval、POPE、AMBER、MMBench（中英文）、MMStar、SimpleVQA、RealWorldQA、OCRBench_v2等多模态与幻觉相关数据集。

**📈 对比分析**

与DPO、IPO、SimPO、β‑DPO、DPO‑Shift等传统对比优化方法在InternVL3‑14B/8B上进行单轮对比。实验显示：在HallusionBench、MM‑IFEval、POPE、AMBER等幻觉评测中，ACPO提升约1–3分，整体保持或提升基础模型的分数；在MMBench、MMStar、VQA、OCRBench等通用多模态基准上，ACPO同样获得最佳或次佳成绩，显著降低视觉锚点崩塌导致的错误率。

**⚠️ 局限性**

限制：仅在内部私有偏好数据集上验证，缺乏公开数据的复现性；未在多轮对话、在线RL（如GRPO）或跨模态动态场景中测试；对超大模型（>14B）之外的规模尚未验证。

---

## 954. Dyadic: A Scalable Platform for Human-Human and Human-AI Conversation Research

**arXiv ID:** 2603.22227 | [PDF](https://arxiv.org/pdf/2603.22227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 955. RAMPAGE: RAndomized Mid-Point for debiAsed Gradient Extrapolation

**arXiv ID:** 2603.22155 | [PDF](https://arxiv.org/pdf/2603.22155v1)

**作者:** Abolfazl Hashemi `[一作]` `[通讯]` (Purdue University), Abolfazl Hashemi (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出两种新的变分不等式求解方法 RAMPAGE 与其变种 RAMPAGE+，通过随机化中点估计消除 Extragradient (EG) 的离散化偏差，并通过反向采样实现方差降低；

**💡 创新点**

创新点在于：1) 设计了一种无偏的随机梯度外推（RAMPAGE）；2) 引入反向采样构造 RAMPAGE+，在保持无偏性的同时显著降低方差并实现确定性收敛上界；3) 在根求解、约束变分不等式以及无约束/有约束的凸-凹极小-极大游戏中提供了 𝒪(1/k) 的最佳迭代收敛率；4) 将方法扩展至高频旋转场景和无限宽 GAN 的 NTK 解析实验。

**🔧 技术方法**

采用随机化中点积分、反向采样、对称缩放投影、α‑对称 (L₀,L₁)-Lipschitz 分析、协同逼近理论与对偶逼近，配合泰勒展开与负相关分析来证明收敛性；

**📊 数据集**

使用三种人工合成实验数据：1) 10维四阶多项式保守场；2) 20维高频旋转最小-最大游戏；3) 2维高频旋转最小-最大游戏；4) 无限宽 ReLU GAN 的 NTK 解析模型；未使用真实数据集。

**📈 对比分析**

与标准 EG 以及单样本 RAMPAGE 进行对比；实验中选取 EG 在临界步长下发散的情况，RAMPAGE+ 在同一步长下实现收敛；RAMPAGE+ 在高频场景下表现出更低的方差与更快的收敛速度，且在无偏性与确定性收敛上优于单样本方法。

**⚠️ 局限性**

局限性：1) 仅在凸-凹、单调或 α‑对称 Lipschitz 环境下证明收敛；2) 未针对非凸非凹极小-最大游戏给出收敛证明；3) 未实现加速版本或最后一次迭代收敛分析；4) 对随机方差上界依赖于噪声假设，实际分布影响未知；5) 对大规模实测数据集的验证仍缺乏。

---

## 956. dynActivation: A Trainable Activation Family for Adaptive Nonlinearity

**arXiv ID:** 2603.22154 | [PDF](https://arxiv.org/pdf/2603.22154v1)

**作者:** Alois Bachmann `[一作]` `[通讯]` (Ruprecht-Karls-Universität Heidelberg), Alois Bachmann (Ruprecht-Karls-Universität Heidelberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种可训练的逐层激活函数 dynActivation，并在图像分类、深度可扩展性、语言建模以及鲁棒性等多项任务中进行评估。

**💡 创新点**

创新点在于将基准激活函数（如 Mish、Swish 等）与两参数线性插值结合，形成可学习的非线性程度；仅需两个轻量级标量即能让每层自适应激活形状。

**🔧 技术方法**

技术包括：自回归的激活函数设计、梯度流与 Lipschitz 性能分析、对比实验、深度可扩展性验证、对抗鲁棒性测试、LLM 迁移实验、以及训练效率（AUC、步数）评估。

**📊 数据集**

使用的数据集包括 CIFAR‑10、CIFAR‑100、MNIST（深度实验）和 SYNTH（LLM 迁移），并在多种网络架构（AttentionCNN、DenseNet、MobileNet、ResNet18 等）上进行测试。

**📈 对比分析**

通过多维度比较（测试准确率、测试损失、AUC、对抗鲁棒性、分布偏移鲁棒性、训练耗时）验证，dynActivation 在多数场景下均能显著提升准确率（如 +14% CIFAR‑10/AttentionCNN）、加速收敛（AUC 降低 24%）并提高鲁棒性（FGSM 下降幅度比 ReLU 小 7.4%）。

**⚠️ 局限性**

局限性包括：对部分基准激活的统计显著性不足；在 LLM 任务中短期优势随训练步数增加而衰减；额外参数带来轻微的计算开销；以及缺乏对 α、β 收敛机制的理论解释。

---

## 957. Computationally lightweight classifiers with frequentist bounds on predictions

**arXiv ID:** 2603.22128 | [PDF](https://arxiv.org/pdf/2603.22128v1)

**作者:** Shreeram Murali `[一作]` (Aalto University), Dominik Baumann `[通讯]` (Aalto University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了基于Nadaraya-Watson估计器的多类别分类器，并给出了频繁主义置信区间；

**💡 创新点**

创新点在于实现线性(甚至子线性)时间复杂度的分类器，同时提供正式的、可计算的误差上界；

**🔧 技术方法**

使用Nadaraya-Watson核密度估计、Lipschitz连续或可分离分布假设、k-d树与哈希表等数据结构；

**📊 数据集**

在合成的Lipschitz与可分离数据集以及MIT-BIH心电图（100维特征）上进行实验；

**📈 对比分析**

与Logistic回归、CME（条件核均值嵌入）等基线相比，取得了>96%准确率、速度提升数百倍，并给出可解释的置信区间；

**⚠️ 局限性**

局限性包括需预先估计Lipschitz常数、在高维下欧氏距离可能失效、dyadic实现无法计算置信区间。

---

## 958. Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control

**arXiv ID:** 2603.22201 | [PDF](https://arxiv.org/pdf/2603.22201v1)

**作者:** Qingrui Zhao `[一作]` (Nanjing University), Xun Cao `[通讯]` (Nanjing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了神经运动重映射（NMR）框架，用于将人类运动数据迁移到仿生机器人，利用CEPR数据生成管道产生物理一致的训练样本，并训练Transformer网络完成实时重映射。

**💡 创新点**

将运动重映射从帧级几何优化转化为分布映射；设计了基于VAE的运动聚类与并行强化学习专家的CEPR流水线；采用双向自注意力Transformer和两阶段预训练+微调的学习策略，实现了对上游噪声的隐式抑制和物理可行性保证。

**🔧 技术方法**

变分自编码器（VAE）聚类、强化学习（PPO）专家策略、MuJoCo物理仿真、Transformer网络、L1损失、双向全注意力、预训练+微调训练方案、基准方法GMR与PHUMA比较。

**📊 数据集**

AMASS人类动作数据集、30K对齐人机运动对（由CEPR生成），以及在Unitree G1机器人上的物理仿真。

**📈 对比分析**

与GMR和PHUMA基线对比，使用关节跳跃数、自碰撞帧数、关节限幅违规等指标，NMR在所有指标上最优：关节跳跃为0，自己碰撞降低54%，关节限幅违例率降至16.8%；在RL追踪任务中成功率最高，MPJPE/W-MPJPE最低，RL策略收敛速度最快。

**⚠️ 局限性**

CEPR流水线与模型架构对特定机器人形态高度耦合，迁移到其他机器人需重新生成数据并调整模型；目前对极端动态动作的覆盖仍有限，缺乏对不同形态的通用性。

---

## 959. Enhancing Document-Level Machine Translation via Filtered Synthetic Corpora and Two-Stage LLM Adaptation

**arXiv ID:** 2603.22186 | [PDF](https://arxiv.org/pdf/2603.22186v1)

**作者:** Ireh Kim `[一作]` (Korea University), Chanwoo Kim `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于大型语言模型的文档级机器翻译数据增强与两阶段微调策略；

**💡 创新点**

创新点在于利用LLM将CNN/Daily Mail摘要数据转化为文档级平行语料，并通过sacreBLEU、COMET、LaBSE‑CosSim三指标联合过滤提升数据质量，再采用句子级预训练后细调到文档级的两阶段微调方案；

**🔧 技术方法**

使用了Llama-3.1-8B-Instruct与Llama-3.1-70B等大规模预训练模型进行数据生成，Google Translate生成伪参考，结合sacreBLEU、COMET、LaBSE‑CosSim进行多指标过滤，最终在Llama-3.1-8B-Instruct上实施两阶段微调；

**📊 数据集**

使用CNN/Daily Mail摘要数据生成约20K个伪文档级En–De对作为文档级数据，句子级数据来自News Commentary v16和Europarl v7等公开平行语料；

**📈 对比分析**

与仅使用文档级数据的基线模型比较，采用sacreBLEU、COMET、LaBSE‑CosSim等指标，发现两阶段微调并结合三指标过滤后模型在所有指标上均提升，最佳配置（sacreBLEU≥35、COMET≥0.75、LaBSE‑CosSim≥0.85）取得15.96、0.701、0.860的得分；

**⚠️ 局限性**

局限性包括对LLM生成数据的hallucination依赖、过滤阈值需经验设定、仅在En→De任务中验证，跨语言和不同领域的推广尚待进一步验证。

---

## 960. Human-Inspired Pavlovian and Instrumental Learning for Autonomous Agent Navigation

**arXiv ID:** 2603.22170 | [PDF](https://arxiv.org/pdf/2603.22170v1)

**作者:** Jingfeng Shan `[一作]` (University of Bologna), Anna Guerra `[通讯]` (University of Bologna)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种融合Pavlovian条件反射、基于模型的（MB）与基于模型的（MF）强化学习的混合决策架构，用于在不确定环境中实现自主代理的安全高效导航。

**💡 创新点**

创新点在于：①将地理引用的电波特征作为条件刺激（CS）引入Pavlovian模块，以生成内在价值信号并对动作选择产生偏置；②通过内部动机门控调节学习与决策过程；③采用贝叶斯仲裁机制根据模型预测误差自适应融合MF与MB估计，实现从探索到计划的平滑过渡。

**🔧 技术方法**

所使用技术包括：强化学习（Q‑learning、Dyna‑Q）、Pavlovian‑Instrumental Transfer（PIT）机制、贝叶斯可靠性模型、动机门控与优势函数调制、软最大（softmax）策略。

**📊 数据集**

实验数据集为仿真生成的二维网格环境（24×36格），包含墙壁、障碍、门（gate）与GPS‑盲区等情境，代理数量为4，目标位置固定，评估指标为定位误差PEB与碰撞次数。

**📈 对比分析**

与传统的单一MF、单一MB、PIT‑MF等基线方法比较，混合架构在相同训练轮次下收敛速度更快、PEB更低、碰撞率显著下降，实验表明在高不确定区域的安全性提升约20%–30%。

**⚠️ 局限性**

局限性包括：①依赖预先构建的电波特征映射，难以在动态变化的环境中直接迁移；②在高维连续状态空间中实现Pavlovian模组需进一步简化或使用函数逼近；③贝叶斯仲裁参数对模型敏感，需经验调优。

---

## 961. Calibeating Made Simple

**arXiv ID:** 2603.22167 | [PDF](https://arxiv.org/pdf/2603.22167v1)

**作者:** Yurong Chen `[一作]` (Inria), Haipeng Luo `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种通用的在线学习框架，利用对calibeating与多重calibeating的归约，证明它们与无后悔学习和专家学习的最小化等价，并在此基础上得到最优的时间复杂度下界与上界；

**💡 创新点**

创新点在于将calibeating问题统一归约为标准在线学习任务，从而实现模块化分析；在此基础上首次给出了多重calibeating的最优 \,O(log N+|Q|\log T)\, 速率，以及在Brier损失下同时实现calibeating与校准的最优折衷；

**🔧 技术方法**

主要技术包括：无后悔学习（如FPL、EWOO）、专家聚合（Hedge/专家算法）、混合可混合性（mixable）分析、Blum‑Mansour 换位损失与校准的等价性、预测空间离散化与随机舍入、以及两专家斜率算法来平衡对参考算法的跟踪与校准；

**📊 数据集**

该工作属于理论研究，不依赖具体数据集；所有结果均为泛化的分析与证明；

**📈 对比分析**

与以往针对Brier、log等特定损失的经验方法相比，本文的通用框架在理论上实现了更好的时间复杂度：单一calibeating可达O(|Q|\log T)，多重calibeating可达O(log N+|Q|\log T)，同时在Brier损失下实现了O(log N+|Q|\log T)的calibeating与O(\sqrt{T})的校准；

**⚠️ 局限性**

局限性包括：只针对严格可混合或有界的对称损失；在多分类情况下校准与calibeating的折衷仍需进一步优化；算法实现涉及高维离散化和矩阵稳态分布，可能导致计算开销较大；

---

## 962. Multimodal Survival Analysis with Locally Deployable Large Language Models

**arXiv ID:** 2603.22158 | [PDF](https://arxiv.org/pdf/2603.22158v1)

**作者:** Moritz Gögl `[一作]` (University of Oxford), Christopher Yau `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种本地可部署的多模态生存分析框架，联合临床文本、表格变量和基因表达，通过compact causal LLM估计校准后的生存曲线并生成简洁的预后文本。

**💡 创新点**

创新点包括：① 采用离线teacher–student蒸馏获取校准数值预测与可解释文本；② 在1.5B规模的compact LLM上实现多模态融合，满足资源受限环境；③ 结合隐藏状态与文本生成的融合策略提升预测性能。

**🔧 技术方法**

使用的技术有：compact causal LLM、隐藏状态生存模型（离散时间危害或CoxPH）、基因表达自编码器、早期/晚期融合、teacher–student蒸馏、校准纠正、指数曲线拟合、时间依赖一致性（Ctd）和IBS评估。

**📊 数据集**

采用TCGA数据集（共8902例），包含病理报告、临床表格和基因表达。

**📈 对比分析**

与BERTSurv、单模态基线及教师模型进行对比。时间依赖一致性Ctd提升约0.03–0.04，IBS下降约0.01–0.02，说明多模态融合与教师蒸馏组合显著优于传统方法。

**⚠️ 局限性**

限制包括：1）长文本可能导致连贯性下降；2）依赖teacher预测，若校准失误会影响文本质量；3）需要手动提取概率句子；4）基因表达的预训练未显著提升性能；5）虽然模型小于大型云模型，但仍需一定硬件资源。

---

## 963. OpenEarth-Agent: From Tool Calling to Tool Creation for Open-Environment Earth Observation

**arXiv ID:** 2603.22148 | [PDF](https://arxiv.org/pdf/2603.22148v1)

**作者:** Sijie Zhao `[一作]` (Nanjing University), Lei Bai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了OpenEarth-Agent框架，该框架通过多代理协同实现数据探测、适应性工作流规划、工具自创和迭代调试，完成从多源数据获取到地理空间分析的完整 EO 流程；同时构建了OpenEarth-Bench基准，包含596个跨7领域的全流程案例；

**💡 创新点**

首创开放环境下的工具创作代理，突破传统工具调用局限，采用实时数据感知、动态DAG规划与工具合成；结合跨域知识库与多阶段工具集，实现跨域、跨阶段的全流程自适应；

**🔧 技术方法**

多代理架构（Data Summary、Planning、Workflow、Coding、Checking），LLM驱动的工具生成与调试，实时数据探测、在线与离线知识检索，工具缓存与反馈闭环，基于GPT‑5、Gemini‑2.5‑Flash等LLM模型；

**📊 数据集**

OpenEarth-Bench 596个真实案例（涵盖城市、农业、植被、水体、土壤、经济、雪域七大领域），以及 Earth‑Bench 104个专业工具用于交叉评测；

**📈 对比分析**

与传统工具调用代理Earth‑Agent进行交叉基准测试；在OpenEarth‑Bench上采用阶段级与端到端评估，指标包括准确率、调试轮数与运行时；OpenEarth‑Agent在阶段级取得约85%、77%、82%准确率，端到端在地理空间分析阶段约59%；在仅使用6个核心工具时，与Earth‑Agent性能相当，使用完整工具集则显著优于Earth‑Agent；

**⚠️ 局限性**

高度依赖LLM的代码推理与生成，导致生成工具可能在复杂建模中功能不佳；多轮LLM推理造成计算量大、延时高，限制了实时紧急响应场景的部署；能源消耗显著，需要优化工具缓存、模型蒸馏等方法以降低碳足迹。

---

## 964. From Singleton Obstacles to Clutter: Translation Invariant Compositional Avoid Sets

**arXiv ID:** 2603.22146 | [PDF](https://arxiv.org/pdf/2603.22146v1)

**作者:** Prashant Solanki `[一作]` (Delft University of Technology), Coen De Visser `[通讯]` (Delft University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在平移不变动力学下，如何将单一障碍物的避障值函数通过平移与逐点最小化的方法重用到复杂障碍环境中，并提出了块级合成框架以减小保守性；

**💡 创新点**

创新点在于证明了避免侧旅行成本价值函数的符号特性，并利用平移不变性得到单一障碍值的平移对应关系，进而构造单点复合与块级复合的保守内证；

**🔧 技术方法**

技术上使用Hamilton‑Jacobi 可达性分析、旅行成本变换、值函数平移性质、点-wise 最小化组合以及Dubins车的数值仿真；

**📊 数据集**

实验数据集为在10×10网格上重复放置半径0.5圆形障碍的Dubins车环境，以及一个1×2横向相邻障碍的对比场景；

**📈 对比分析**

通过与直接在两障碍环境中训练得到的价值函数比较，单点复合误差仅在障碍影响区域附近，且在闭环轨迹仿真中能够成功避免碰撞，性能表现良好；

**⚠️ 局限性**

局限性包括：单点复合在障碍相互作用强时过于保守；块级合成虽然减小保守性但计算复杂度上升；证明仅适用于平移不变动力学，无法推广到更一般的动力学或随机环境。

---

## 965. Benchmarking Deep Learning Models for Aerial LiDAR Point Cloud Semantic Segmentation under Real Acquisition Conditions: A Case Study in Navarre

**arXiv ID:** 2603.22229 | [PDF](https://arxiv.org/pdf/2603.22229v1)

**作者:** Alex Salvatierra `[一作]` (Public University of Navarre), Mikel Galar `[通讯]` (Public University of Navarre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对四种先进的深度学习架构（KPConv、RandLA‑Net、Superpoint Transformer、Point Transformer V3）在Navarre地区真实航测LiDAR数据集上进行语义分割基准测试。

**💡 创新点**

创新之处在于首次将卷积、MLP和Transformer模型系统性地应用并比较于大规模、实战航空LiDAR场景，揭示了类不平衡与几何多变性对模型性能的影响。

**🔧 技术方法**

采用的技术包括点云卷积网络、随机采样加局部聚合、基于超点的自注意力机制以及序列化点云的Transformer，统一训练配置并使用OA、mIoU等指标评估。

**📊 数据集**

使用的数据集为Navarre（西班牙）约4 km²的航测LiDAR点云，平均密度约50 pts/m²，标注五类（地面、低植被、中高植被、建筑、车辆）。

**📈 对比分析**

在相同的训练与评估框架下进行比较，KPConv以96.16%总体准确率、78.51%平均IoU领先；RandLA‑Net、Superpoint Transformer和Point Transformer V3分别在速度、参数量或少数类性能上表现突出。

**⚠️ 局限性**

局限性在于低植被类（仅占1.41%）和车辆类的识别仍受几何模糊与样本稀缺影响，且模型在此类上的IoU波动较大；此外实验仅覆盖单一地理区域，缺乏更广泛的跨地区验证。

---

## 966. Gumbel Distillation for Parallel Text Generation

**arXiv ID:** 2603.22216 | [PDF](https://arxiv.org/pdf/2603.22216v1)

**作者:** Chi Zhang `[一作]` (University of Texas at Austin), Qiang Liu `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于Gumbel-Max trick的知识蒸馏框架，将非自回归模型学习的难题转化为有监督学习。

**💡 创新点**

使用Gumbel噪声作为“蓝图”，使非AR学生能够直接学习教师的采样路径，从而克服联合分布学习难题。

**🔧 技术方法**

Gumbel-Max采样、并行Gumbel提取、掩码扩散语言模型(MDLN、BD3-LM)与多token预测(MTP)的集成。

**📊 数据集**

LM1B、OpenWebText，以及多项常识推理/问答基准数据集。

**📈 对比分析**

与同参数预算的AR模型和未蒸馏的并行模型对比，MAUVE提升30%，生成困惑度下降10.5%，在多任务上提升约1-2%准确率。

**⚠️ 局限性**

随着词表大小增大，Gumbel噪声向量维度升高，导致额外投影成本，且在高维空间可能带来计算负担。

---

## 967. Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models

**arXiv ID:** 2603.22212 | [PDF](https://arxiv.org/pdf/2603.22212v1)

**作者:** Meiqi Wu `[一作]` (University Of Chinese Academy Of Sciences), Kaiqi Huang `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了Omni-WorldBench评测基准，用于系统评估视频世界模型在4D空间中对交互行为的响应能力。

**💡 创新点**

创新点包括：① 构建层次化交互提示集 Omni-WorldSuite；② 设计多维度评估框架 Omni-Metric，涵盖视频质量、摄像机-物体可控性和交互效果三大维度；③ 通过 AgenticScore 自适应聚合机制，使评估结果与交互语义匹配。

**🔧 技术方法**

采用视频生成技术、视觉‑语言模型（VLM）、光流与相机运动估计、结构相似度（SSIM）、CLIP 语义特征、VLM 问答、场景检测等多种技术手段构建评估指标。

**📊 数据集**

使用公开视频数据集（DriveLM、InternData‑A1、Sekai 等）以及概念驱动生成的图像与轨迹，构成 1078 条交互提示，覆盖 3 个交互层级、多种物理原理与任务场景。

**📈 对比分析**

对 18 种文本‑视频、图像‑视频及摄像机控制模型进行统一评估，使用 AgenticScore 等指标。实验显示 IT2V 模型（如 Wan2.2）取得最高综合分（75.92%），T2V 模型中 HunyuanVideo 领先，摄像机控制组 HunyuanWorld、WonderWorld 取得最高分。交互效果指标普遍低于视频质量，表明现有模型在因果交互方面仍有限。

**⚠️ 局限性**

局限性：评测仅覆盖预定义的 3 级交互层级，难以体现长期开放世界动态交互；缺乏人类偏好对齐评估；提示集仍受限于设计和数据来源，未完全覆盖所有真实交互情境。

---

## 968. Topological Collapse: P = NP Implies #P = FP via Solution-Space Homology

**arXiv ID:** 2603.22211 | [PDF](https://arxiv.org/pdf/2603.22211v1)

**作者:** M. Alasli `[一作]` `[通讯]`, M. Alasli

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

证明了若P=NP，则P=#P，并进一步推导出P=PH=P，展示了通过3-SAT解空间的拓扑结构（第二Betti数）将决策问题与计数问题联系起来的非相对化证明。

**💡 创新点**

创新点在于将Betti数作为桥梁，将决策复杂度与计数复杂度相连接，并通过拓扑不等价性构造了一个全新的二分法，证明了任何多项式时间决策算法必需计算#P‑难函数；同时提出了首次实证验证“解空间碎裂”的实验方法。

**🔧 技术方法**

使用代数拓扑方法（Betti数计算、同调不变量、子立方查询下界）、理论证明中的对偶性和二分法，以及实验中基于随机 3‑SAT 的受限探测和梯度下降验证。

**📊 数据集**

使用随机 3‑SAT 数据集，变量数从 100 到 500，子句密度在 3.8–4.2 之间，尤其聚焦于满足率阈值附近；实验测量了簇间/簇内 Hamming 距离、解空间碎裂比例。

**📈 对比分析**

对比了四类求解策略：随机探测、局部走访、CDCL 以及拥有完整目标信息的梯度下降；结果显示前三者在 N≥300 时 40 步内从未触及目标簇，而后者则始终收敛，验证了理论中提出的 P vs #P 的性能差异。

**⚠️ 局限性**

局限性包括：证明仅适用于 3‑SAT，尚未推广到一般 NP‑问题；实验规模受限于 500 变量；Betti 数的计算是理论推导而非可直接在大规模实例中高效计算；并且依赖于随机实例的碎裂性质，可能不适用于结构更简单的实例。

---

## 969. Causal Evidence that Language Models use Confidence to Drive Behavior

**arXiv ID:** 2603.22161 | [PDF](https://arxiv.org/pdf/2603.22161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 970. Separators for intersection graphs of spheres

**arXiv ID:** 2603.22204 | [PDF](https://arxiv.org/pdf/2603.22204v1)

**作者:** Jacob Fox `[一作]` (Stanford University), Jonathan Tidor `[通讯]` (Princeton University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4`

**🎯 论文内容**

证明了在任意维度下，球和球面（以及fat凸体及其边界）的交叉图存在平衡分离器，其大小为O_d(m^{1/d}n^{1-2/d})，其中n为顶点数，m为边数；并给出了相应的随机线性/二次时间算法。

**💡 创新点**

核心创新在于：1）扩展了Lipton–Tarjan分离器理论到高维几何交叉图；2）提出了基于球/球面中心点的随机半径法，得到与球体密度相关的分离器；3）给出与度数序列直接相关的上界，并证明这些上界在参数上最优；4）构造了极大构造性示例和fat凸体的必要性证明。

**🔧 技术方法**

使用了几何概率、凸体与fat性分析、Hölder不等式、平衡分离器迭代、半代数图的Zarankiewicz定理、随机采样与Markov不等式等技术。

**📊 数据集**

本文未涉及具体实验数据集，所有结果均为理论分析与构造示例；算法复杂度以输入规模n、m为评估基准。

**📈 对比分析**

与现有的Planar图、曲线图、半代数图等分离器理论比较，证明了在球面/球体交叉图上可达到与边数和顶点数相关的最优分离器大小；算法实现仅需随机线性（球体）或二次（球面）时间，并能以高概率得到平衡分离器。

**⚠️ 局限性**

局限性：1）对一般凸体或非fat体无法保证分离器存在；2）球面交叉图在高维下仍可能需要二次时间；3）随机算法的成功概率虽可控制但仍不确定；4）在维度高到极限时常数因子会变大，影响实际可用性。

---

## 971. Revisiting Quantum Code Generation: Where Should Domain Knowledge Live?

**arXiv ID:** 2603.22184 | [PDF](https://arxiv.org/pdf/2603.22184v1)

**作者:** Oscar Novo `[一作]` (QCentroid), Carlos Kuchkovsky `[通讯]` (QCentroid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文重新评估了在Qiskit代码生成任务中，现代通用LLM在未做域特定微调情况下的表现，并与先前的参数专门化基线进行比较。

**💡 创新点**

创新点在于将检索增强生成(RAG)与执行反馈驱动的代理推理引入代码生成流程，证明了推理时域适配可替代参数级微调，并提供了多种推理配置的系统评估。

**🔧 技术方法**

技术上使用了OpenAI、Anthropic Claude、Google Gemini等通用LLM，结合检索增强生成(RAG)以及多步执行-修复代理循环，评估其在Qiskit-HumanEval基准上的通用性能。

**📊 数据集**

数据集为Qiskit-HumanEval（151个量子编程任务），与之前的Qiskit Code Assistant benchmark 对齐。

**📈 对比分析**

通过 Pass@1 指标和总执行时间比较，现代通用LLM在零样本下已匹配或超过参数专门化模型，RAG提升有限，代理推理（最多5步）可将准确率提升至85.4%，但对应的运行时显著增加。

**⚠️ 局限性**

主要局限包括基准泄露风险、推理时非确定性、模型版本更新导致可重复性受限、未进行成本/计算归一化评估、仅针对 Qiskit 且未验证跨其他 SDK 的泛化。

---

## 972. Closed-Loop Verbal Reinforcement Learning for Task-Level Robotic Planning

**arXiv ID:** 2603.22169 | [PDF](https://arxiv.org/pdf/2603.22169v1)

**作者:** Dmitrii Plotnikov `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了闭环 Verbal Reinforcement Learning（VRL）框架，用大型语言模型（LLM）作为 actor，视觉-语言模型（VLM）作为 critic，通过结构化自然语言反馈在可执行行为树（Behavior Tree）上进行符号级策略迭代，以解决移动机器人在执行不确定性环境下的任务规划问题。

**💡 创新点**

创新点包括：① 将自然语言反馈直接用于符号级策略更新，实现可解释、硬件感知的策略改进；② 明确区分因果诊断（critic）与策略更新（actor），避免梯度优化，提升透明度；③ 在真实机器人上实现无模拟、无梯度的闭环学习。

**🔧 技术方法**

使用技术包括行为树（BT）作为策略表示，LLM（如 Qwen2.5-VL、Gemini 3 Pro）作为 actor 进行结构化 BT 修改，VLM 作为 critic 对机器人图像与 BT 轨迹进行视觉-语言评估，提供报警分数、置信度和自然语言原因；以及无梯度符号策略更新机制。

**📊 数据集**

实验数据来自真实移动机器人仓库物流场景：木块拾取、放置、可移动架子搬运等任务，包含机器人图像、BT 执行轨迹、实时评分和人工标注的错误与置信度；对比了 5 种配置（无 critic、Qwen2.5-VL 7B、Qwen2.5-VL 3B 微调、Gemini 3 Pro、Gemini 3 Pro + 块色信息），并在 10 轮 episode × 5 场景上进行训练。

**📈 对比分析**

通过 episode 分数、Critic 准确率、置信度以及人类报警率进行评估。实验表明：① Gemini 3 Pro + 块色信息获得最高分数、最稳定收敛；② 微调后的 Qwen2.5-VL-3B 在准确度与稳定性上优于未微调的 7B；③ 无 critic 配置学习缓慢或不收敛，说明结构化视觉反馈至关重要。

**⚠️ 局限性**

局限性包括：① 受 critic 的感知与因果归因质量限制，无法检测延迟或隐蔽错误；② 只关注 critic 能即时观察到的错误，缺乏长时序因果推理；③ 结构化反馈设计有限，难以覆盖所有任务细节；④ 虽然改进 critic 可提升收敛速度，但在符号空间有限的 BT 结构中进一步提升效果递减。

---

## 973. ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling

**arXiv ID:** 2603.22126 | [PDF](https://arxiv.org/pdf/2603.22126v1)

**作者:** Byungjin Kim `[一作]` `[通讯]` (AgentAI Co., Ltd.), Byungjin Kim (AgentAI Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了名为Robogate的工业机器人部署风险管理框架，利用两阶段自适应采样在物理仿真环境中高效映射成功-失败边界并生成可解释的风险模型。

**💡 创新点**

创新点包括：①在高维参数空间中引入边界聚焦采样（Stage 2）提升边界分辨率；②跨机器人平台（Franka Panda与UR5e）并行评估，揭示通用危险区；③使用可解释的逻辑回归模型提供闭式失败边界方程与置信区间。

**🔧 技术方法**

采用的技术包括：Latin Hypercube Sampling (LHS) 进行空间填充；NVIDIA Isaac Sim 与 Newton 物理引擎进行高保真仿真；逻辑回归（含交互项）进行风险预测；bootstrap 重采样估计阈值置信区间。

**📊 数据集**

使用了 30,000 条基于仿真的实验数据集，涵盖 8 维操作参数（摩擦、质量、COM 偏移、尺寸、IK 噪声、障碍物数、几何形状、放置配置）并在 Franka Panda 与 UR5e 上收集。

**📈 对比分析**

与单阶段均匀采样相比，两阶段采样使风险模型的 AUC 从 0.754 提升至 0.780；在统一阈值下成功率分别为 48.6%（Franka）和 74.3%（UR5e）；在 VLA 模型评估中 Octo‑Small 在 68 个对抗场景下仅达 30.9% 成功率，远低于 94% 的脚本基线。

**⚠️ 局限性**

局限性包括：仿真与真实世界存在差距；仅针对 pick‑and‑place 任务；LHS 假设参数独立，未考虑真实物理关联；逻辑回归模型缺乏高阶非线性表达，AUC 仍受限；缺乏多任务和多机器人平台的扩展验证。

---

## 974. Programming Manufacturing Robots with Imperfect AI: LLMs as Tuning Experts for FDM Print Configuration Selection

**arXiv ID:** 2603.22118 | [PDF](https://arxiv.org/pdf/2603.22118v1)

**作者:** Ekta U. Samani `[一作]` (Carnegie Mellon University), Christopher G. Atkeson `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在 Fused Deposition Modeling（FDM）3D 打印中，将大语言模型（LLM）作为有限决策模块嵌入到基于证据的优化循环中，以改进打印参数配置。

**💡 创新点**

创新点包括：① 设计了一个模块化的 LLM‑指导优化循环；② 开发了近似工具路径评估器，可产生结构化诊断；③ 将 LLM 的自然语言指导编译成可执行的软硬约束，直接驱动 Bayesian 优化；④ 通过实验证明，LLM 在循环内比单次 AI 推荐更能提升配置质量和可靠性。

**🔧 技术方法**

所用技术包括：大语言模型（GPT‑5.2、Llama‑3.1）、近似工具路径评估器（基于 slicer 的诊断）、Bayesian optimization（GPyOpt，Matérn 5/2 核），以及将 LLM 输出转换为软/硬约束的指导编译器。

**📊 数据集**

使用的数据集为 Thingi10k，随机挑选了 100 个单件、面数 < 100 的模型，用作实验评估。

**📈 对比分析**

与多种基线（默认配置、启发式重新定向、ChatGPT/Gemini 单次推荐、无指导 Bayesian 优化）进行对比。LLM‑指导优化在 78% 的物件上获得最佳配置，0% 的可能失效率；平均目标值从无指导的 0.24 降至 0.14，且样本效率明显提升（最优‑即‑时目标更快下降）。

**⚠️ 局限性**

局限性包括：① 采用单目标标量化，未捕捉质量/时间/成本的多目标 Pareto 结构；② 仅调节有限的打印参数，遗漏对其他参数或交互的改进；③ 评估器为近似，可能忽略细微缺陷；④ 未考虑零件改造和更高 Fidelity 的仿真；⑤ 对评估器与硬件的依赖可能限制迁移性。

---

## 975. On the Direction of RLVR Updates for LLM Reasoning: Identification and Exploitation

**arXiv ID:** 2603.22117 | [PDF](https://arxiv.org/pdf/2603.22117v1)

**作者:** Kexin Huang `[一作]` (Alibaba Group), Jingren Zhou `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对强化学习可验证奖励（RLVR）中模型微调的更新方向进行定量分析，并提出两种利用该方向提升推理性能的实用方法：测试时的方向外推和训练时的优势加权。

**💡 创新点**

创新点在于将更新的“方向”视为核心信息，用符号化的 token‑level log‑probability 差值 Δlog p 替代传统的幅度指标（熵、KL 等）来识别稀疏但关键的推理相关 token；基于此设计了仅在高 Δlog p 位置进行概率外推和低概率 token 加权的算法。

**🔧 技术方法**

使用的技术包括：RLVR（GRPO/DAPO）微调框架、token‑replacement 评估、Δlog p 统计、基于 Δlog p 的概率外推（π_Extra^γ）、优势重加权（ã_i,t）以及标准的基准模型对比和评测指标 Avg@k/Pass@k。

**📊 数据集**

主要数据集为数学推理类公开基准：AIME‑24、AIME‑25 和 AMC；实验还采用了 ORZ、UniReason、Qwen2.5-Math‑7B 等不同规模模型的 RLVR 对照组。

**📈 对比分析**

在 AIME‑24、AIME‑25 与 AMC 上，本文方法相较于 DAPO 基线平均提升 Avg@32 约 3–5 %（最高 5.4 %）并在 Pass@16 上提升 1–4 %；测试时外推在保留原模型基准的前提下进一步提高了 0.5–1.0 % 的准确率；训练时加权在所有模型和数据集上均表现出一致的性能提升。

**⚠️ 局限性**

主要限制包括：方向外推需要同时拥有基线与 RLVR 模型，增加了部署成本；超参数（γ、阈值 τ）对性能影响较大，需要调优；实验范围局限于数学推理任务，尚未验证在更广泛的生成任务上的适用性。

---

## 976. Framework for Risk-Based IoT Cybersecurity Audit Engagements

**arXiv ID:** 2603.22191 | [PDF](https://arxiv.org/pdf/2603.22191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 977. MARCUS: An agentic, multimodal vision-language model for cardiac diagnosis and management

**arXiv ID:** 2603.22179 | [PDF](https://arxiv.org/pdf/2603.22179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 978. Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson's Disease

**arXiv ID:** 2603.22225 | [PDF](https://arxiv.org/pdf/2603.22225v1)

**作者:** Abner Hernandez `[一作]`, Paula Andrea Perez-Toro `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于中心化向量运算的语言偏移（LS）方法，对源语言的自监督语音表示进行对齐，以提升跨语言的运动性语音障碍检测性能。

**💡 创新点**

创新点在于：①仅在表示空间进行语言对齐，无需重新训练模型；②利用健康受试者的均值向量作为语言中心，实现轻量级的语言归一化；③在受限的多语言运动性语音数据上显著提升敏感度与F1分数。

**🔧 技术方法**

使用HuBERT、WavLM和XLS‑R三种自监督语音模型提取表示，并通过中心化向量平移进行语言偏移；随后用逻辑回归分类器进行PD与HC判别。

**📊 数据集**

利用捷克语、德语和西班牙语的帕金森病运动性语音数据集，聚焦口腔口吃测试（/pa‑ta‑ka/）。

**📈 对比分析**

与无LS的跨语言基线相比，LS在三种模型中均提升了敏感度（最高从0.35提升至0.93）和F1（最高从0.48提升至0.74）；在多语言设定下，LS主要提升了特异性，性能提升幅度相对温和。

**⚠️ 局限性**

局限性：①不同语言来自不同数据集，难以区分语言差异与语料特性；②只评估了高度控制的口吃任务，未验证在更自然语音情境下的效果。

---

## 979. Noise Titration: Exact Distributional Benchmarking for Probabilistic Time Series Forecasting

**arXiv ID:** 2603.22219 | [PDF](https://arxiv.org/pdf/2603.22219v1)

**作者:** Qilin Wang `[一作]` `[通讯]` (Independent Researcher), Qilin Wang (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了通过在已知动力学系统中注入可调高斯噪声的干预式评估框架，将时序预测从单一路径匹配转化为分布推断任务。

**💡 创新点**

创新点在于将噪声调制作为实验变量，利用Fern模型的SPD参数化直接输出完整协方差，实现精确负对数似然训练与分布校准。

**🔧 技术方法**

采用Fern的谱参数化、全负对数似然（NLL）目标、Wasserstein距离、PIT、Shapiro‑Wilk、覆盖率等统计诊断方法，并与基准模型进行对比。

**📊 数据集**

使用一系列可控的混沌与非平稳系统（如Rössler、Lorenz、Chua、Lorenz‑96、SLDS、SAR、DW、OU）以及加入不同级别的观测噪声。

**📈 对比分析**

与Chronos‑Bolt、Chronos‑2、TimesFM‑2.5等零射模型对比，Fern在混沌与参数漂移情形下实现了1–2个数量级的误差降低，并在噪声增大时保持更好的校准与覆盖率。

**⚠️ 局限性**

局限性包括仅对高斯噪声与高斯分布做出精确推断，且评估基于合成数据，缺乏对真实非平稳时间序列的直接验证。

---

## 980. PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation

**arXiv ID:** 2603.22193 | [PDF](https://arxiv.org/pdf/2603.22193v1)

**作者:** Mingju Gao `[一作]` (Peking University), Hao Zhao `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种 Pose–Appearance–Motion Engine（PAM），实现从仅有初始与目标手部姿势以及物体几何信息的最小输入，生成高分辨率、时序连贯的手物交互视频；

**💡 创新点**

创新点在于将姿态、外观和运动三大生成任务解耦，采用多模态（深度图、语义分割、手部关键点）条件的控制扩散模型，克服了先前方法对真实首帧或完整姿态序列的依赖；

**🔧 技术方法**

使用了控制扩散网络（ControlNet）微调的 Flux 图像扩散模型生成首帧，以及基于 CogVideo-X 的可控视频扩散模型生成完整视频，核心技术为多模态条件注入与运动先验融合；

**📊 数据集**

在 DexYCB 与 OAKINK2 两大手物交互数据集上进行评估，生成视频用于数据增强后还提升了 SimpleHand 关键点回归性能；

**📈 对比分析**

与 CosHand、InterDyn、ManiVideo 等现有方法相比，PAM 在 FVD、MPJPE、LPIPS、SSIM、PSNR 等指标上均取得显著提升，且视频分辨率提升至 480×720；

**⚠️ 局限性**

局限性包括：仍采用多阶段生成流程，计算成本高；对多手或更复杂物体交互的泛化能力待验证；以及模型训练对大量多模态条件的依赖。

---

## 981. The Semantic Ladder: A Framework for Progressive Formalization of Natural Language Content for Knowledge Graphs and AI Systems

**arXiv ID:** 2603.22136 | [PDF](https://arxiv.org/pdf/2603.22136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 982. DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment

**arXiv ID:** 2603.22125 | [PDF](https://arxiv.org/pdf/2603.22125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 983. Beyond Matching to Tiles: Bridging Unaligned Aerial and Satellite Views for Vision-Only UAV Navigation

**arXiv ID:** 2603.22153 | [PDF](https://arxiv.org/pdf/2603.22153v1)

**作者:** Kejia Liu `[一作]` (Zhejiang University), Haofei Zhang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于单阶段回归的视觉驱动无人机跨视角定位与航向估计框架（Bearing‑UAV），同时完成绝对位置与航向预测。

**💡 创新点**

创新点在于：①引入全球-局部统一特征（GLUF）与相对坐标编码（RCE）双重特征融合；②利用交叉注意力与相似度引导定位（PSG）显著提升在跨视角错位和稀疏特征下的鲁棒性；③一次性回归位置与航向，消除传统检索+姿态估计链条带来的误差累积。

**🔧 技术方法**

核心技术包括VGG‑16视觉骨干、非局部块、聚类分段、交叉注意力、相似度加权坐标引导，以及多任务MLP回归网络。

**📊 数据集**

使用自己构建的多城跨视角数据集（Bear‑UAV‑Dataset），包含90k UAV视角图像与对应的四邻卫星图块，覆盖四个城市，支持连续卫星图块与多样化视角采样。

**📈 对比分析**

与现有CVGL基线（如University‑1652、GTA‑UAV等）对比，Bearing‑UAV在定位误差（8.6 m vs ~30 m）和航向误差（3.3° vs 10°以上）上显著提升，Recall@1提升约10%，并在纯视觉导航任务中完成约一半复杂路径。

**⚠️ 局限性**

局限性在于：①对极端稀疏或纹理极差区域的定位仍易漂移；②模型对跨城市极端地形变化的泛化虽已改善，但在大尺度高纬度区域尚需验证；③仅依赖视觉的单机推理在长距离连续飞行中仍受累积误差影响，需进一步结合稀疏地理先验或在线自我校正。

---

## 984. More Isn't Always Better: Balancing Decision Accuracy and Conformity Pressures in Multi-AI Advice

**arXiv ID:** 2603.22152 | [PDF](https://arxiv.org/pdf/2603.22152v1)

**作者:** Yuta Tsuchiya `[一作]` (University of Tokyo), Yukino Baba `[通讯]` (University of Tokyo)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多AI面板对人类决策准确性和从众压力的影响，并通过实验检验面板规模、意见一致度和人类化呈现的作用。

**💡 创新点**

创新点在于系统评估多AI咨询时的从众机制与信息多样性，并提出面板规模与意见一致度的最优设计原则。

**🔧 技术方法**

采用随机森林生成多样化AI预测并用GPT‑4o生成自然语言解释；实验设计基于Judge‑Advisor System框架。

**📊 数据集**

使用Adult、COMPAS和SpeedDating三个公开数据集构建三类二分类任务。

**📈 对比分析**

通过对比单AI、3 AI、5 AI以及人类化与非人类化面板，测量准确率、依赖度、置信度；结果显示3 AI可提升准确率，5 AI无进一步提升；人类化未显著影响准确率。

**⚠️ 局限性**

局限在样本为在线日本/亚洲参与者、任务简短、文本界面限制，且从众机制与文化差异未充分探究。

---

## 985. WorldCache: Content-Aware Caching for Accelerated Video World Models

**arXiv ID:** 2603.22286 | [PDF](https://arxiv.org/pdf/2603.22286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 986. End-to-End Training for Unified Tokenization and Latent Denoising

**arXiv ID:** 2603.22283 | [PDF](https://arxiv.org/pdf/2603.22283v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 987. Optimal-Time Move Structure Balancing and LCP Array Computation from the RLBWT

**arXiv ID:** 2603.22147 | [PDF](https://arxiv.org/pdf/2603.22147v1)

**作者:** Nathaniel K. Brown `[一作]` (Johns Hopkins University), Ben Langmead `[通讯]` (Johns Hopkins University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了在 O(r) 时间和空间内平衡移动结构的算法，从而实现了在 O(n) 时间、O(r) 空间下从 RLBWT 直接构建 LCP 数组。

**💡 创新点**

首次给出线性时间 O(r) 的移动结构平衡方法，并将其与 Sanaullah 等人的 LCP 计算框架结合，突破之前 O(r log r) 的瓶颈，得到最优 O(n) 时间的 RLBWT→LCP 转换。

**🔧 技术方法**

采用链表实现的平衡策略（同时平衡 π 与 π⁻¹）、移动结构、稀疏采样与 FL 运动、改进的 ϕ 算法及相关数据结构技术。

**📊 数据集**

论文未给出具体实验数据集，主要面向理论分析，但讨论了在基因组重复文本（如 pangenome）上的典型应用。

**📈 对比分析**

与之前的 O(r log r) 平衡算法相比，时间从 O(r log r) 降至 O(r)；与原始 O(n+r log r) 的 LCP 构造相比，时间从 O(n+r log r) 降至 O(n)。在 O(r) 空间下实现压缩索引和匹配的效率显著提升。

**⚠️ 局限性**

仍需从原始文本构建 RLBWT 和移动结构的时间保持为 O(n+r log r)；缺乏实测评估；对非重复文本适用性有限；invertible 移动结构的实现和评估留作后续工作。

---

## 988. Biophysics-Enhanced Neural Representations for Patient-Specific Respiratory Motion Modeling

**arXiv ID:** 2603.22123 | [PDF](https://arxiv.org/pdf/2603.22123v1)

**作者:** Jan Boysen `[一作]` (German Research Center for Artificial Intelligence), Jan Ehrhardt `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了一种基于隐式神经表示（INR）的呼吸运动建模框架 PRISM‑RM，用来在放疗中从外部呼吸信号预测肿瘤和器官运动。

**💡 创新点**

通过引入时间连续的流场（Euler 积分）、空间-时间正则化（Neo‑Hookean 物理约束与时间总变差）以及去除固定参考图像的差分流框架，使模型在任意呼吸相位间均可估计变形，显著提升了外推性能。

**🔧 技术方法**

使用隐式神经表示、Euler 积分、非线性物理正则化（Neo‑Hookean）以及时间总变差正则化。

**📊 数据集**

采用 11 名肺癌患者的 4D CT 数据及对应的模拟呼吸信号。

**📈 对比分析**

与传统两阶段线性/多项式对应模型相比，PRISM‑RM 在插值任务保持相当的性能，在外推任务显著优于前一版 INR 模型，但整体仍略逊于传统两阶段方法。

**⚠️ 局限性**

仍存在对未见呼吸状态的泛化不足、训练所需计算资源较多，以及物理正则化在无图像时间点的效果有限等局限。

---

## 989. Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding

**arXiv ID:** 2603.22121 | [PDF](https://arxiv.org/pdf/2603.22121v1)

**作者:** Yunzhuo Sun `[一作]` (Dalian University of Technology), Wenxin Liang `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建两阶段框架：先用LLM匹配字幕并分解查询，生成短视频先验，再通过多模态控制Mamba网络实现长序列时空融合，完成视频片段检索。

**💡 创新点**

创新点在于将字幕与文本融合生成动态视频先验、使用LLM对查询进行语义拆分、引入视频引导门控的Mamba网络实现线性时间高效融合，显著提升多动词查询的时序精度。

**🔧 技术方法**

主要技术包括LLaMA‑3.1用于字幕匹配与查询拆解、CogVideoX文本到视频扩散模型生成短视频、CLIP ViT‑B/32文本与视频编码、GCN获取相邻帧关系、双向状态空间模型（Mamba）与视频门控、以及交叉熵、相似性对比损失。

**📊 数据集**

实验使用TVR（21,793视频/108k查询）和ActivityNet‑Captions（20k视频/100k查询）两大多模态检索基准。

**📈 对比分析**

与SgLFT、ICQ、HERO等SOTA方法对比，在TVR上R@1@0.5提升约1.07%，R@1@0.7提升约1.02%，在ActivityNet上亦取得约0.16%提升；总体回召率和SumR均显著提高，且计算开销低于Transformer。

**⚠️ 局限性**

主要局限在于生成视频先验的质量受文本与字幕质量限制、对无字幕视频依赖较大、以及离线生成视频仍占一定计算资源，未来需提升生成鲁棒性与多模态融合的通用性。

---

## 990. UniMotion: A Unified Framework for Motion-Text-Vision Understanding and Generation

**arXiv ID:** 2603.22282 | [PDF](https://arxiv.org/pdf/2603.22282v1)

**作者:** Ziyi Wang `[一作]` (Peking University), Mengyuan Liu `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 UniMotion，一个统一框架，能够在单一网络中同时完成人类运动、自然语言和 RGB 图像的理解与生成，支持七种三模态任务（如 T2M、M2T、运动预测、运动编辑、视图转运动、视图转文本、运动驱动图像编辑）。

**💡 创新点**

①首次将运动视为连续模态与 RGB 同等对待，构建 CMA‑VAE 连续运动编码；②双路径嵌入器（语义与生成分支）和混合注意力实现跨模态对齐；③Dual‑Posterior KL Alignment (DPA) 把视觉信息无缝注入运动编码；④Latent Reconstruction Alignment (LRA) 用自监督的运动重建预训练校准运动通道，解决稀疏文本监督导致的冷启动问题。

**🔧 技术方法**

CMA‑VAE、DPA、LRA、Show‑o2 1.5B LLM 背骨、双路径嵌入器、混合注意力、模态路由 LoRA、流式头（motion & vision）、SkipTransformer、HRNet 视觉编码、Gaussian reparameterization、KL 对齐、Euler ODE 推理等技术。

**📊 数据集**

HumanML3D、MotionFix、Human3.6M、MoVid、3DPW、Human3.6M+3DPW 组成的运动、文本、图像三模态数据集；用于 Vision‑to‑Motion、Vision‑to‑Text、Motion‑Guided Image Editing 等评测。

**📈 对比分析**

与 MotionGPT、MG‑MotionLLM、UniPose、Show‑o2 等现有方法在七项任务上做统一对比。UniMotion 在 T2M R@3 达 0.841（最高），M2T BertScore 41.2，运动预测 ADE 3.17，运动编辑 R@3 84.94，Vision‑to‑Motion MPJPE 75.0，运动驱动图像编辑 Motion Acc 0.67，均为现有最佳水平或显著提升。

**⚠️ 局限性**

作为通用框架，UniMotion 在专用任务（如 MPJPE）仍落后于专业模型；训练规模大、对资源要求高；在极端稀疏或无图像配对的数据上 DPA 依赖对齐时仍可能出现信息不足；未来可进一步缩小模型体积并提升对多模态缺失情况的鲁棒性。

---

## 991. ShapDBM: Exploring Decision Boundary Maps in Shapley Space

**arXiv ID:** 2603.22235 | [PDF](https://arxiv.org/pdf/2603.22235v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 992. 3D-Layout-R1: Structured Reasoning for Language-Instructed Spatial Editing

**arXiv ID:** 2603.22279 | [PDF](https://arxiv.org/pdf/2603.22279v1)

**作者:** Haoyu Zhen `[一作]` (NVIDIA), Subhashree Radhakrishnan `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

基于语言指令的多步3D布局编辑框架 3D-Layout-R1，能够通过结构化的场景图逐步更新布局并保持空间一致性。

**💡 创新点**

创新点：①将场景图作为可验证的中间状态，提出“chain‑of‑graph‑edits”可解释推理；②使用GRPO强化学习在IoU、碰撞和格式奖励上进一步优化布局；③构建了15k条3D编辑样例及对应的指令、轨迹和目标布局。

**🔧 技术方法**

技术：大型语言模型（如Qwen3、Qwen2.5）与视觉语言模型，CoT‑SFT预训练，GRPO（基于IoU、碰撞、格式奖励的强化学习），JSON场景图序列化与更新。

**📊 数据集**

数据集：1) 10k synthetic 3D sorting 场景；2) 1k Blender 生成的空间对齐 benchmark；3) 约 15k 场景来自 InstructScene 的房间编辑任务；4) 额外的无字幕、噪声输入与真实机器人实验。

**📈 对比分析**

对比方法：零样本 LLM/VLM、单步 CoT‑SFT、Vanilla GRPO。实验表明 3D-Layout-R1 在 mIoU 上提升约 15%，中心点误差下降 25–30%，在零样本和 CoT‑SFT 基线上平均提升 20% 以上，且在排序、空间对齐和房间编辑任务上均实现最高 IoU 与最低中心误差。

**⚠️ 局限性**

局限性：①需要可解析的场景图与精确的指令；②对高度模糊或多重可能解的指令支持有限；③模型主要生成高层布局，未覆盖连续动力学与运动规划；④训练成本高，需要大规模图像+文本对齐数据。

---

## 993. UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos

**arXiv ID:** 2603.22264 | [PDF](https://arxiv.org/pdf/2603.22264v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 994. TiCo: Time-Controllable Training for Spoken Dialogue Models

**arXiv ID:** 2603.22267 | [PDF](https://arxiv.org/pdf/2603.22267v1)

**作者:** Kai-Wei Chang `[一作]` (MIT), James Glass `[通讯]` (MIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TiCo，一种后训练框架，使语音对话模型能够根据显式时间控制指令生成具有可控时长的响应；

**💡 创新点**

通过自生成监督和强化学习的两阶段训练，结合口语时间标记（STM）实现模型在生成过程中的实时时间意识和动态调整；

**🔧 技术方法**

利用自生成、对齐后生成STM、基于GRPO的强化学习与CHORD正则化、Qwen‑2.5‑Omni 7B backbone、MS‑SWIFT训练平台等技术；

**📊 数据集**

InstructS2S、UROBench、LIFEBench（文本）作为评测数据集；训练使用InstructS2S语音问题与Whisper生成的时间戳；

**📈 对比分析**

与多种基线（开源SDM、商用LLM+TTS级联系统）在TiCo‑Bench上比较，TiCo在MAE、MAPE上显著优于基线（MAE 4.54 s，MAPE 14.9%，比基线低约70% MAE、50% MAPE），并保持相似的质量评分；

**⚠️ 局限性**

对极短时间约束（10‑30 s）仍存在误差，STM预测精度可提升；实验集中在单一对话情境，未验证更复杂多轮对话或跨域适用性；

---

## 995. A Dividing Line for Structural Kernelization of Component Order Connectivity via Distance to Bounded Pathwidth

**arXiv ID:** 2603.22240 | [PDF](https://arxiv.org/pdf/2603.22240v1)

**作者:** Jakob Greilhuber `[一作]` (CISPA Helmholtz Center for Information Security), Roohani Sharma `[通讯]` (Institute for Basic Science)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究Component Order Connectivity（COC）问题——给定图G、整数d和k，判定是否存在不超过k个顶点的删除集使得剩余图的每个连通分量至多包含d个顶点。作者证明了当参数为距离到路径宽度为1的图加上d时，COC拥有多项式kernel；并阐明了路径宽度1与2之间的划分边界与Vertex Cover的类似。还给出了针对d-COC、距离到最大度为2的图、以及一般块阻塞集等多种结构参数的进一步kernel化结果。

**💡 创新点**

创新点主要包括：①首次在COC的结构化参数化下实现多项式kernel；②发现路径宽度1与2的划分是COC可kernel化的关键分界；③提出“caterpillar本质（essence）”这一新概念，利用函数合成和半群理论高效构造具有相同本质的更小子图；④开发了一套完整的pack‑and‑replace技术，用于压缩路径宽度为1模量子下的连通分量；⑤给出了一系列新的阻塞集与扩张引理的组合，解决了在无树深度限制的长路径上的kernel化难题。

**🔧 技术方法**

核心技术包括：
- 路径宽度为1的图是caterpillar森林，利用其独特的结构构造唯一的“solution‑tight packing”。
- 通过Erdős–Pósa性质和α‑merged packing选取可安全替换的子caterpillar。
- 定义并求解caterpillar本质函数，利用抽象代数中的半群与基本函数分解实现多项式时间合成。
- 扩张引理与冲突图相结合，压缩模量子外的连通分量数量。
- 结合已知的k+d多项式kernel（Xiao）完成最终kernel。

**📊 数据集**

本工作为纯理论算法研究，未使用实验数据或公开数据集；所有结果均通过严格的证明给出。

**📈 对比分析**

方法评价：
- 证明了COC在距离到路径宽度1+ d 参数化下的多项式kernel，kernel大小为 (d^7·|M|^3 + d^6·|M|^4) 或在d固定时为 |M|^4。 
- 对比现有的Vertex Cover kernel，展示了结构参数化对COC的更高挑战性和更丰富的界定。 
- 所有结果均为最优或接近最优的理论上限，进一步证明了路径宽度1与2的划分为关键分界。

**⚠️ 局限性**

局限与开放问题：
- 结果仅适用于距离到路径宽度1的模量子，d要么是固定常数，要么作为参数的一部分；若d既不是常数也不在参数中，仍无多项式kernel。 
- 对于更一般的结构参数（如任意闭包族或高阶树宽度模量）仍缺乏完整的可kernel化与不可kernel化的双分离。 
- 本文给出的多项式kernel的次数较高（d^7、d^6等），如何进一步优化或证明更低阶极限仍是开放挑战。

---

## 996. Scaling DoRA: High-Rank Adaptation via Factored Norms and Fused Kernels

**arXiv ID:** 2603.22276 | [PDF](https://arxiv.org/pdf/2603.22276v1)

**作者:** Alexandra Zelenin `[一作]`, Alexandra Zhuravlyova `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

实现了 DoRA 的系统级优化，提出了基于分解范数的低内存计算和融合的 Triton 核心，显著降低了训练和推理时的内存占用与计算时间。

**💡 创新点**

创新点包括：①将行向量范数拆分为基底、交叉、Gram 三项，避免了密集 BA 乘积；②在 Triton 上实现一次性融合的前向/反向核，减少 4 倍内存传输并提升 1.5–2.7 倍速度；③提供三层运行时调度，自动选择最优路径。

**🔧 技术方法**

技术手段包括低秩适配器 LoRA/DoRA 理论、线性代数分解、张量分块与 fp32 累计、Triton 自定义 kernel 与自动调优、PyTorch、DeepSpeed ZeRO、FSDP 兼容以及 CUDA 与 GPU 内存分配策略。

**📊 数据集**

在 8–32B 视觉语言模型（如 Qwen、Gemma、Mistral 等）上进行实验，并使用 MMFineReason-SFT-123K 数据集进行 SFT 对比。

**📈 对比分析**

通过与 Hugging Face PEFT 基线以及自实现的 eager 路径比较，在六种 NVIDIA GPU（L40S、A100、RTX6000PRO、H200、B200、B300）上进行微基准与模型级别测试。结果显示：融合实现比 PEFT 快 1.5–2.0 倍、比 eager 1.18–1.24 倍；峰值 VRAM 降低 1.3–6.7 GB；推理速度提升 1.5–2.0 倍；最终 logits 的余弦相似度 > 0.9999，训练收敛保持一致。

**⚠️ 局限性**

局限性包括：不支持 FSDP2/DTensor；高阶并行时需要额外的全局聚合；低秩 r < 64 或 d_in < 4096 时优势不明显；调度阈值是经验值，未来硬件可能需重调；仅在单 GPU 或 ZeRO/FSDB 下验证，RLHF 等工作未评估。

---

## 997. Flip Distance of Non-Crossing Spanning Trees: NP-Hardness and Improved Bounds

**arXiv ID:** 2603.22262 | [PDF](https://arxiv.org/pdf/2603.22262v1)

**作者:** Håvard Bakke Bjerkevik `[一作]` (University at Albany), Birgit Vogtenhuber `[通讯]` (Graz University of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在凸平面点集上无交叉生成树的翻转图，证明计算任意两棵树之间的最短翻转距离是NP‑hard，并给出了新的更高下界（11/7 n – o(n)）以及在一棵树为堆叠树时的上界（14/9 (n‑1)）。

**💡 创新点**

① 用冲突图和放大（blow‑up）技术将最大无环子集问题与翻转距离联系起来，进而得到NP‑hard性。② 在堆叠树情形下构造多套无环子集，实现更紧凑的翻转序列。③ 通过递归构造树对，取得新的更高下界。

**🔧 技术方法**

冲突图（conflict graph）构造、放大操作、可计算的最大无环子集、Planar V‑Cycle Max‑2SAT 的归约、堆叠树分层与图着色、递归树构造与冲突分析。

**📊 数据集**

本文为理论性工作，没有使用实验数据集；所有结果均通过构造性证明与严格推理得出。

**📈 对比分析**

与之前的 14/9 n – O(1) 下界、5/3 n – O(1) 上界、以及兼容/旋转翻转的 5/3 n – O(1)、7/4 n – O(1) 等结果进行对比；本工作把下界提升至 11/7 n – o(n)，并在堆叠树情形下把上界压至 14/9 (n‑1)，在这些指标上均实现了显著改进。

**⚠️ 局限性**

仍存在显著的上下界差距（上界 14/9 ≈ 1.556， 下界 11/7 ≈ 1.571）；NP‑hard性证明依赖于极大放大因子 β，导致在实际规模上不可行；仅针对凸点集，尚未给出多边形或一般位置点集的精确复杂度；方法主要适用于无交叉树，难以推广到更一般的图重构问题。

---

## 998. Greater accessibility can amplify discrimination in generative AI

**arXiv ID:** 2603.22260 | [PDF](https://arxiv.org/pdf/2603.22260v1)

**作者:** Carolin Holtermann `[一作]` (University of Hamburg), Anne Lauscher `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了音频LLM在性别偏见上的表现，系统评估八款模型在基于声音的交互中对性别刻板印象的产生，并通过用户调查探讨可访问性与隐私风险。

**💡 创新点**

证明音频LLM比文本LLM更易放大性别偏见，揭示声学特征（尤其音高）与模型歧视的因果关系，并提出通过音高调节实现偏见缓解的实验方法。

**🔧 技术方法**

采用音频直接输入LLM、log odds ratio、配对词对测评、F0分位数音高操纵实验以及在线问卷与有序Logistic回归等技术。

**📊 数据集**

构建了1,370个内容匹配的男女对照音频样本，来源于英国方言、美国口音与Spoken Stereoset等三大音频语料库；同时收集了1,000名美国受访者的问卷数据。

**📈 对比分析**

通过单句与配对评估、文本与语音对比的置换检验，发现高检测精度模型的性别偏差显著大于文本；音高调节实验显示，部分模型可将偏差显著降低至无统计差异。

**⚠️ 局限性**

仅限英语男女二元性别，音频样本受限于有限口音与合成声音；部分模型因安全门槛或API差异被排除；未评估其他音频特征、多语言环境及长期用户体验的普适性。

---

## 999. MemDLM: Memory-Enhanced DLM Training

**arXiv ID:** 2603.22241 | [PDF](https://arxiv.org/pdf/2603.22241v1)

**作者:** Zehua Pei `[一作]` (Chinese University of Hong Kong), Bei Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为MemDLM的记忆增强扩散语言模型，旨在弥补训练与推理之间的差距，提升长文本检索与推理能力。

**💡 创新点**

创新点在于通过双层优化在训练中模拟多步去噪轨迹，并利用快速权重（fast weights）形成参数化记忆，使模型在训练时就能内部化局部去噪经验，从而降低曝光偏差并增强长上下文表现。

**🔧 技术方法**

核心技术包括：双层优化（Bi-level Optimization）、快速权重适配（fast weights / LoRA）、吸收式掩码去噪、以及在推理时可选的内循环适配实现内置检索机制。

**📊 数据集**

实验使用LongAlpaca指令调优数据集，评估基于RULER（Needle‑in‑a‑Haystack）与BABILong长文本检索任务以及LongBench多任务（如多文档QA、摘要、代码补全）进行验证。

**📈 对比分析**

与标准MDLM相比，MemDLM在两大骨干（LLaDA‑MoE和LLaDA2.1）上均实现显著提升：在8K上下文的RULER变量追踪从78.8%提升至95.8%，BABILong 8K从47.4%提升至57.0%；在16K/32K超长场景仍保持优势；在LongBench上零推理内循环时已提升约30%，推理内循环进一步提升约1–2%。

**⚠️ 局限性**

局限性包括：需要额外的快速权重更新和梯度计算，导致训练时内存和计算成本上升；推理时开启内循环虽可进一步提升，但会增加推理时延；方法对预设的“anchor”比例和快速权重范围敏感，若不合适可能导致性能退化；目前仅在特定掩码策略和模型结构上验证，跨模型推广仍需探索。

---

## 1000. DualCoT-VLA: Visual-Linguistic Chain of Thought via Parallel Reasoning for Vision-Language-Action Models

**arXiv ID:** 2603.22280 | [PDF](https://arxiv.org/pdf/2603.22280v1)

**作者:** Zhide Zhong `[一作]`, Haoang Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种并行的视觉-语言链式推理（Parallel Visual‑Linguistic CoT）框架，利用可学习查询令牌在连续潜在空间中同时完成低层视觉细节推理与高层逻辑规划，从而为视觉‑语言‑动作（VLA）模型提供“先思考后行动”的能力。

**💡 创新点**

创新点包括：① 双流隐式 CoT，分别实现视觉 CoT 与语言 CoT；② 引入可学习查询令牌，打破自回归推理瓶颈，实现一次前向推断即可得到完整推理；③ 通过冻结视觉教师（Depth Anything 3）和语言教师（Qwen3‑0.6B）实现跨模态知识蒸馏，提升空间感知与逻辑规划的质量；④ 结合 Flow‑Matching 的 Diffusion Transformer 动作头，兼顾连续动作预测与高频控制。

**🔧 技术方法**

主要技术手段包括：多模态 VLM 预训练模型（Qwen3‑VL‑4B）作为 backbone；可学习查询令牌的交叉注意力与投影；视觉教师 DA3 的几何蒸馏；语言教师 Qwen3‑0.6B 的文本 CoT 监督；Diffusion Transformer 动作头与 Flow‑Matching 损失；整体端到端联合训练。

**📊 数据集**

使用的数据集有：LIBERO 机器人操作基准（4个任务组，7-DoF 机械臂）；RoboCasa GR1 Tabletop Tasks（24个任务，29-DoF 灵巧手）；以及真实世界的 AgileX Cobot 平台（3个桌面抓取与放置任务）。

**📈 对比分析**

与现有非 CoT VLA、AR CoT VLA（CoT‑VLA、ThinkAct、Fast‑ThinkAct、LaRA‑VLA）以及 GR00T 系列等对比，本文在 LIBERO 上平均成功率达 98.8%（最高 99.8%），在 RoboCasa 上平均 55.1%，在实机上对 3 个难度级任务的成功率均超过基线；推理延迟从 3178 ms 降至 83 ms（VLM 前向 58 ms），实现了高频闭环控制。

**⚠️ 局限性**

局限性在于：① 依赖冻结的教师模型，可能限制对新领域的自适应；② 目前仅在桌面抓取与放置类任务验证，尚未探索更复杂的交互或动态环境；③ 需要大量对齐的视觉‑语言 CoT 注释，生成成本较高；④ 仍受限于 VLM 与动作头的规模，部署在算力受限的边缘设备上可能需要进一步压缩。

---

## 1001. DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution

**arXiv ID:** 2603.22271 | [PDF](https://arxiv.org/pdf/2603.22271v1)

**作者:** Zhengyao Lv `[一作]` (University of Hong Kong), Kwan-Yee K. Wong `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a4b10f5d-130b-4e77-9367-6469ec621899` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了DUO-VSR，一种三阶段一阶视频超分辨率框架，融合分布匹配蒸馏和对抗监督以实现高效且高质量的VSR

**💡 创新点**

创新点在于双流蒸馏策略：将分布匹配蒸馏（DMD）与Real–Fake Score Feature GAN（RFS‑GAN）联合优化，解决传统DMD在VSR中的训练不稳定、监督失真和指导不足问题，并引入Preference‑Guided Refinement进一步提升感知质量

**🔧 技术方法**

采用分布匹配蒸馏、对抗GAN（RFS‑GAN）、进化蒸馏（CFG+Progressive Distillation）、Direct Preference Optimization (DPO) 以及内部1.3B参数的文本‑视频Diffusion Transformer模型

**📊 数据集**

在合成数据集（SPMCS、UDM10、YouHQ40）、真实场景数据集（VideoLQ）以及自建AI生成视频集（AIGC60）上训练与评估

**📈 对比分析**

与RealViformer、VEnhancer、MGLD、UAV、STAR、DLoRAL、DOVE、SeedVR2等先进方法比较，DUO‑VSR在无参考感知指标（NIQE、MUSIQ、CLIP‑IQA、DOVER）及流对齐误差(E_warp)上均取得最高或接近最高分，推理速度比SeedVR‑7B提升约50×，参数量仅1.3B，保持低延迟

**⚠️ 局限性**

局限在于对极端噪声或极低分辨率场景的鲁棒性尚待进一步验证，且在极长视频序列上的时域一致性仍有提升空间

---

## 1002. GenOpticalFlow: A Generative Approach to Unsupervised Optical Flow Learning

**arXiv ID:** 2603.22270 | [PDF](https://arxiv.org/pdf/2603.22270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 1003. Confidence-Based Decoding is Provably Efficient for Diffusion Language Models

**arXiv ID:** 2603.22248 | [PDF](https://arxiv.org/pdf/2603.22248v1)

**作者:** Changxiao Cai `[一作]` (University of Michigan), Gen Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对扩散语言模型中的自适应置信度解码（confidence‑based decoding）进行首个理论分析，证明其能在 KL 散度上实现可接受的采样精度，同时大幅降低迭代次数。

**💡 创新点**

创新点在于：①构造了一套适用于随机解码顺序的理论框架；②提出基于熵总和的解码策略，并给出其期望迭代次数为 O(H(X₀)/ε·logL) 的上界；③展示该策略能自适应低熵数据分布，无需先验信息或超参数调优。

**🔧 技术方法**

主要技术包括：信息理论的熵与互信息分解、Markov 链分析、基于随机排列的自适应停止规则，以及对 KL 散度误差的上界推导。

**📊 数据集**

论文主要为理论研究，未使用具体实验数据集；假设掩码预测器是最优的。

**📈 对比分析**

与传统的均匀解码和基于阈值的策略相比，该熵总和策略在低熵场景下实现了迭代次数从 Θ(L) 下降到 O(H(X₀)/ε·logL)，即显著提升采样速度；理论上保证了可接受的 KL 误差。

**⚠️ 局限性**

局限性包括：①仅在掩码预测器最优的理想假设下成立；②使用随机解码顺序，未分析确定性排序的表现；③仅针对熵这一置信度度量，未给出其他度量的下界；④缺乏对实际训练误差影响的分析。

---

## 1004. One Model, Two Markets: Bid-Aware Generative Recommendation

**arXiv ID:** 2603.22231 | [PDF](https://arxiv.org/pdf/2603.22231v1)

**作者:** Yanchen Jiang `[一作]` (Harvard University), Di Wang `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的生成式推荐框架 GEM-Rec，能够在同一模型中同时处理有机推荐与广告投放，并通过控制令牌和实时竞价调节生成过程

**💡 创新点**

创新点在于：①引入控制令牌将“是否投放广告”与“生成哪条内容”解耦；②通过训练历史成功日志学习“可行性”策略；③在推理时注入实时竞价，满足分配单调性与有机完整性；④提供理论保证和可调的 λ 控制参数

**🔧 技术方法**

使用基于 RQ‑VAE 的层次语义 ID、Transformer 自回归解码、控制令牌、两层竞价调制（slot 与 item 级别）、束搜索与层次解码等技术

**📊 数据集**

在 Steam、Amazon Beauty、Amazon Sports、Amazon Toys 四个公开电商/游戏数据集上构造合成市场，随机设 20% 物品为广告并赋予对数正态竞价

**📈 对比分析**

与原 TIGER 生成式推荐模型对比，评估指标包括：总 NDCG、Organic NDCG、Ad Rate、Revenue、Ad Relevance；实验显示 GEM‑Rec 在保持 organic NDCG 与 TIGER 相近的同时，能够通过增大 λ 逐步提升广告收入，并实现可控的 Ad Rate 与高价位广告占比

**⚠️ 局限性**

局限性：①实验使用合成市场，真实竞价分布与用户行为可能更复杂；②采用 First‑Price 计费，未实现完全的 DSIC；③在高 λ 下广告相关性下降，需进一步平衡经济与语义；④需要手动调节 λ 以适配不同业务场景

---

## 1005. Repurposing Geometric Foundation Models for Multi-view Diffusion

**arXiv ID:** 2603.22275 | [PDF](https://arxiv.org/pdf/2603.22275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 1006. DexDrummer: In-Hand, Contact-Rich, and Long-Horizon Dexterous Robot Drumming

**arXiv ID:** 2603.22263 | [PDF](https://arxiv.org/pdf/2603.22263v1)

**作者:** Hung-Chieh Fang `[一作]` (Stanford University), Dorsa Sadigh `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本论文提出了一种层次化的双手抓取与控制框架（DexDrummer），用于在机器人手臂与手上实现长周期、接触丰富的鼓点演奏；

**💡 创新点**

创新点在于将鼓演奏视作结合了手内控制、外部接触与长期任务的难题，并通过分层策略（高层规划+低层残差强化学习）以及面向接触的奖励机制实现了高效的学习与 sim‑to‑real 转移；

**🔧 技术方法**

使用了运动规划生成的轨迹与残差强化学习、接触目标奖励（手内接触奖励、外部接触奖励、轨迹引导奖励）、接触课程化、以及基于 MIDI 输入的轨迹规划；

**📊 数据集**

实验采用了基于 ManiSkill 的仿真环境，包括多鼓（snare、tom、ride、hi‑hat、crash）与单鼓快速练习，MIDI 文件提供鼓点序列；

**📈 对比分析**

与固定抓取（Fixed Grasp）对比，Reactive Grasp 在 F1 分数上提升 1.87×（易曲）与 1.22×（难曲），同时保持持棒比例；在单鼓高频练习中，指尖驱动控制在 BPM 提升时表现优于全臂驱动，并显著降低能耗；仿真到真实世界转移后，在闭环条件下可实现 1.0 的 F1 分数，超出训练序列的演奏也表现良好；

**⚠️ 局限性**

局限包括：鼓速未达到人类水平、歌曲长度受限（最多 20 秒）、缺乏完整鼓组演奏、仅使用域随机化的 sim‑to‑real 方法，未来需提升速度、鲁棒性并探索更强的仿真‑真实迁移技术。

---

## 1007. ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model

**arXiv ID:** 2603.22281 | [PDF](https://arxiv.org/pdf/2603.22281v1)

**作者:** Haichao Zhang `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种融合VLM语义引导的JEPA风格潜在世界模型框架 ThinkJEPA，用双时域采样同时学习细粒度动态与长程语义。

**💡 创新点**

创新点在于：①双时域路径（稠密采样+均匀采样）实现动态与语义双重补偿；②层级金字塔抽取多层VLM表示并通过FiLM注入，使潜在预测兼具物理精度与语义指引；③保持潜在预测接口，兼容下游轨迹回归。

**🔧 技术方法**

使用 V-JEPA-L 背骨、Transformer 预测器、Vision‑Language 模型 Qwen3‑VL（缓存 Encoder/AR token）、金字塔层级特征提取、FiLM 层级调制、递归 rollout 与多尺度下游回归头。

**📊 数据集**

主要在 EgoDex 和 EgoExo4D 两个 egocentric 交互视频数据集上进行手部轨迹预测评估。

**📈 对比分析**

与 V-JEPA 预测器、VLM 仅推理、以及多种基准（BC/DDPM/Flow‑Matching）比较，ThinkJEPA 在 ADE/FDE 上显著优于所有基线（EgoDex ADE 0.061 vs 0.071/0.114/0.152，EgoExo4D ADE 0.622 vs 0.661/0.659/0.622），并在递归 rollout 下保持更低的误差增长。

**⚠️ 局限性**

局限在：①仍依赖 VLM 缓存与多层特征，对长视频与极大帧率的扩展尚未验证；②需要两路采样和额外 VLM 计算，资源开销相对较高；③在极端稀疏动态或非 egocentric 场景中语义引导效果不确定。

---

## 1008. EgoGroups: A Benchmark For Detecting Social Groups of People in the Wild

**arXiv ID:** 2603.22249 | [PDF](https://arxiv.org/pdf/2603.22249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 1009. Decoupling Exploration and Policy Optimization: Uncertainty Guided Tree Search for Hard Exploration

**arXiv ID:** 2603.22273 | [PDF](https://arxiv.org/pdf/2603.22273v1)

**作者:** Zakaria Mhammedi `[一作]` (Google Research), James Cohan `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种将探索与策略优化分离的框架，用基于知识不确定性的树搜索（Go-With-Uncertainty）在稀疏奖励环境中高效探索，并通过后向学习将探索轨迹转化为可部署策略。

**💡 创新点**

核心创新是：①利用环境重置和不确定性引导的树搜索代替传统的内在奖励强化学习；②在探索阶段完全不使用策略优化，仅通过粒子群体管理（赢家选取、剪枝、回滚）实现深度探索；③通过不确定性度量作为赢家选择的关键标准。

**🔧 技术方法**

技术栈包括：分布式协调-工作者架构、基于 RND 的知识不确定性估计、随机或轻量化的粒子策略（随机或小型网络）、基于奖励无关的后向学习（PPO+backward curriculum）以及多组粒子并行与同步。

**📊 数据集**

实验数据集：Atari 的三款难度高的稀疏奖励游戏（Montezuma's Revenge、Pitfall、Venture）以及MuJoCo 的连续动作任务（Adroit 的 hammer/door/relocate、AntMaze），全部使用原始图像观测。

**📈 对比分析**

与主流内在奖励基线（RND、MEME、BYOL-Hindsight）相比，探索阶段所需交互量减少约 10 倍；在 Atari 上，后向学习得到的策略在无领域知识条件下均超过现有最优；在 Adroit 上实现 99.9%/96.4%/93.9% 的成功率，AntMaze 平均 86.3% 成功率，均显著优于以往仅能使用专家演示或离线数据的方法。

**⚠️ 局限性**

局限性：①对可重置环境的依赖，物理机器人等现实场景需要额外实现；②不确定性估计对噪声-TV 问题敏感，需更稳健的模型；③高维连续任务仍需大量环境交互，且需要对群体大小、超参数等进行经验调优；④粒子策略的多样性虽可用随机动作实现，但在极端难度下可能仍需更强的探索策略。

---

## 1010. Structure-aware divergences for comparing probability distributions

**arXiv ID:** 2603.22237 | [PDF](https://arxiv.org/pdf/2603.22237v1)

**作者:** Rohit Sahasrabuddhe `[一作]` (University of Oxford), Renaud Lambiotte `[通讯]` (University of Oxford)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一类基于结构感知熵的Bregman散度，用以比较具有相似性结构的概率分布，并将其应用于经济地理和生态学的数据分析。

**💡 创新点**

创新点在于将结构感知熵转化为严格凹函数，进而构造闭式可计算的结构感知散度，既保留KL散度的几何优势，又能捕获元素之间的相似性；该方法显著加速了分布比较并能恢复被传统方法忽视的结构信息。

**🔧 技术方法**

核心技术包括结构感知HC熵、Bregman散度、正定相似性矩阵构造（通过负型度量或最近正定近似）、k‑means式Bregman聚类、以及向量化计算Jensen‑Bregman散度。

**📊 数据集**

实验数据涵盖：英国及威尔士的传统职业就业分布（41种可交易职业）、Rutor冰川的植物种群丰度与功能性特征（45种物种）以及基于合成分布的聚类检验。

**📈 对比分析**

与最优传输（Wasserstein‑1）相比，Jensen‑Bregman散度在闭式表达下速度提升约10倍；在合成聚类实验中能在样本稀疏时恢复预设群组；在经济地理与生态β多样性案例中得到与OT一致的结果，且更易解释。

**⚠️ 局限性**

局限性包括：需保证相似性矩阵正定且α≥2才能保证熵的严格凹性；在高维大规模场景下存储与运算密集；目前仅支持硬聚类，软聚类与空间约束尚未实现。

---

## 1011. exaCB: Reproducible Continuous Benchmark Collections at Scale Leveraging an Incremental Approach

**arXiv ID:** 2603.22251 | [PDF](https://arxiv.org/pdf/2603.22251v1)

**作者:** Jayesh Badwaik `[一作]` (Forschungszentrum Jülich), Andreas Herten `[通讯]` (Forschungszentrum Jülich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可插拔的连续基准测试框架（ExaCB），将基准测试集成进CI/CD流水线，实现了在JUPITER超算上对70+科学应用的持续、可重现性能与能耗评估。

**💡 创新点**

创新点包括：①基于共享协议的强耦合、去中心化设计，支持逐步提升的运行、可测量、可重现三阶段；②利用现成的CI/CD组件和通用基准工具（JUBE、ReFrame等），实现了跨应用、跨系统统一数据格式与可视化；③提供能耗注入式实验（EnergyAware Launcher）与特征注入（Feature Injection）机制，轻松扩展实验场景。

**🔧 技术方法**

技术：GitLab CI/CD模板、JUBE（以及计划支持ReFrame、Ramble）、Jacamar跑步器、Python辅助分析工具、JSON协议、Grafana/LLview可视化、能源采样插件；框架实现为插件化的执行、后处理、特征注入三大 orchestrator。

**📊 数据集**

数据集：JUPITER 早期访问程序（JUREAP）中的70+应用，JUPITER Benchmark Suite（16个应用+7合成基准），以及日常运行的自定义脚本（BabelStream、GRAPH500、OSU Microbenchmarks 等）。

**📈 对比分析**

比较方法：将每个基准的运行结果以结构化 JSON 存储，在后处理层生成强/弱扩展、机器对比、时间序列图；能耗结果通过 EnergyAware Launcher 抽取并可视化。实验表明：系统间性能差异明显，能耗随频率调节有明显“甜点”，且基准在持续跑测中未出现回归，证明框架有效。

**⚠️ 局限性**

限制：目前仅支持 GitLab CI；对高度可定制的基准仍需人工适配；能耗测量依赖于特定硬件支持，无法在所有节点上统一；协议与工具仍在演进，未来需进一步完善版本兼容与自动化验证。

---

## 1012. VideoDetective: Clue Hunting via both Extrinsic Query and Intrinsic Relevance for Long Video Understanding

**arXiv ID:** 2603.22285 | [PDF](https://arxiv.org/pdf/2603.22285v1)

**作者:** Ruoliu Yang `[一作]` (Nanjing University), Chaoyou Fu `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VideoDetective 框架，通过在长视频中构建视觉-时间亲和图，并利用假设-验证-修正循环进行稀疏观测，最终定位与查询相关的关键片段并供多模态大型语言模型生成答案。

**💡 创新点**

创新点在于：①将视频内部的视觉相似度与时间连通性融合成亲和图，实现查询相关性与视频内在结构的双重指导；②采用稀疏观测的假设-验证-修正循环，利用图扩散在全局范围内传播观测信息；③引入多模态证据（视觉描述、OCR、ASR）和源感知融合的相关性评分，提升鲁棒性。

**🔧 技术方法**

主要技术包括：视觉-时间亲和图构建（Cosine相似度 + 指数衰减时间核），图扩散（对称归一化拉普拉斯），假设-验证-修正迭代框架，Graph‑NMS 进行关键片段筛选，以及多模态证据提取与融合。

**📊 数据集**

使用的基准数据集：VideoMME‑long（无字幕子集）、LVBench、LongVideoBench（验证集）以及 MLVU（测试集），并在这些数据集上与多种 MLLM 后端进行评测。

**📈 对比分析**

与四大类长视频理解方法（LVNet、DVD、VideoAgent、VideoRAG）以及多种模型规模（8B~72B）进行公平对比。实验显示，VideoDetective 在所有基线模型上平均提升 5–7% 以上，最高在 InternVL‑2.5（8B）上提升 7.5%，在 SeedVL‑1.5（20B）上达到 67.9% 的准确率，甚至超越某些 72B 规模的专有模型。

**⚠️ 局限性**

局限性在于：框架高度依赖 VLM 的自我反思反馈（如 “missing keywords”），若 VLM 反馈失准或缺乏表达能力，相关性评估可能失效；此外，构建亲和图和图扩散的参数需要手工调优，可能对不同视频域产生敏感性。

---

## 1013. The Dual Mechanisms of Spatial Reasoning in Vision-Language Models

**arXiv ID:** 2603.22278 | [PDF](https://arxiv.org/pdf/2603.22278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 1014. Riverine Land Cover Mapping through Semantic Segmentation of Multispectral Point Clouds

**arXiv ID:** 2603.22230 | [PDF](https://arxiv.org/pdf/2603.22230v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

