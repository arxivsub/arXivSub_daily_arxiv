# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-05 | 今日论文总数: 516

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Toward Effective Multimodal Graph Foundation Model: A Divide-and-Conquer Based Approach

**arXiv ID:** 2602.04116 | [PDF](https://arxiv.org/pdf/2602.04116v1)

**作者:** Sicheng Liu `[一作]` (University of YYY), Guoren Wang `[通讯]` (University of YYY)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向多模态图的基础模型 PLANET，采用分治策略将模态交互与模态对齐分别在嵌入层和节点层进行处理，显著提升了多模态图表示的质量与泛化能力。

**💡 创新点**

创新点在于：①将模态交互拆解为局部语义增强（EDG）与全局语义一致化（NDR）；②EDG 采用 Mixture‑of‑Experts 与拓扑感知注意力；③NDR 通过离散语义表示空间 (DSRS) 与文本锚定的对齐损失实现跨模态对齐。

**🔧 技术方法**

技术手段包括自监督预训练（重建、对比学习）、Graph Transformer、MoE 机制、离散量化 (VQ)、梯度停止、温度调节等，整体构架在多模态 GNN 上实现。

**📊 数据集**

使用多模态图数据集：Reddit、Movies、Grocery、Toys、Ele‑fashion、Goodreads‑NC 等，涵盖节点分类、边预测、少样本任务以及图‑文本/图‑图像生成。

**📈 对比分析**

与 9 个强基线（包括 GCN、MMGCN、MGAT、GRACE、GraphMAE2、SAMGPT、RiemannGFM、GFT、UniGraph2）在节点分类、边预测、少样本学习、生成任务上比较，PLANET 以平均 3–5% 的准确率/MRR 提升、在少样本节点分类上提升 15–22%，显著优于现有最优模型。

**⚠️ 局限性**

局限性：目前仅支持文本与图像两种模态，未涉及音频、视频等；分治模块虽提升性能但在极大规模图上的计算与存储仍有挑战；对超参数（如专家数、DSRS 维度）敏感，需进一步自动化调优。

---

## 2. CoRe: Context-Robust Remasking for Diffusion Language Models

**arXiv ID:** 2602.04096 | [PDF](https://arxiv.org/pdf/2602.04096v1)

**作者:** Kevin Zhai `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**通讯引用:** 58277 | [OpenAlex ID](https://openalex.org/A5080823547)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Context‑Robust Remasking 框架，利用无训练的鲁棒优化在 Masked Diffusion Model（MDM）推理时通过上下文脆弱性检测并重掩码，提升结构一致性。

**💡 创新点**

以对抗性上下文掩码评估 token 稳定性，将重掩码目标从传统的置信度或置信边缘转为对上下文扰动的鲁棒性，形成单前向传递的高效实施。

**🔧 技术方法**

使用 Masked Diffusion Models、鲁棒优化（distributionally robust objective）、上下文掩码（perturbation）+ margin 筛选、单步前向推理与 token 重新采样。

**📊 数据集**

在 LLaDA‑8B‑Base 上的代码与推理基准：GSM8K、MATH、BBH、HumanEval、MBPP。

**📈 对比分析**

与 Low‑Confidence、Top‑k Margin、ReMDM‑conf、随机/边缘重掩码等方法在相同计算预算下对比，结果显示在 MBPP 上提升约 9.2%，在其它基准亦有显著或相当提升；计算成本仅增加约 6% 前向。

**⚠️ 局限性**

仅改善结构一致性，对事实正确性无保证；在非结构化推理任务中 instability 信号可能噪声较大；参数设置（m、E）对性能敏感，跨模型/规模的通用性待验证。

---

## 3. Transformers perform adaptive partial pooling

**arXiv ID:** 2602.03980 | [PDF](https://arxiv.org/pdf/2602.03980v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 4. LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure From Ordinal Data

**arXiv ID:** 2602.04192 | [PDF](https://arxiv.org/pdf/2602.04192v1)

**作者:** Vivek Anand `[一作]` (Georgia Institute of Technology), Christopher Rozell `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3481 | [OpenAlex ID](https://openalex.org/A5011481913)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于 Schatten‑p 低秩正则化的低秩序列嵌入方法 LORE，能够从噪声三元比较中自动学习内在维度与嵌入。

**💡 创新点**

创新点在于首次联合恢复内在维度和嵌入，并使用非凸 Schatten‑p 量正则化与迭代加权优化，同时给出收敛到驻点的理论保证。

**🔧 技术方法**

采用三元软加函数化的对数损失、Schatten‑p 低秩正则化、迭代重加权（SVD）更新以及梯度下降的组合实现，并给出收敛证明。

**📊 数据集**

在合成数据、基于 LLM 的模拟感知空间（SBERT 食品嵌入截断 1–10 维）以及三组真实人类评测数据集（Food‑100、Materials、Cars）上进行评估。

**📈 对比分析**

与 SOE、FORTE、t‑STE、CKL 及 Dim‑CV 等方法比较，LORE 在三元准确率与基线相当，但在秩估计上显著更低（接近真实内在维度），且得到的轴更具语义解释性，训练时间亦优于 Dim‑CV。

**⚠️ 局限性**

局限在于仅收敛到局部极值且缺乏精确秩恢复的全局理论保证，且在高噪声或高维度情况下仍可能出现误差，且需手动设定 λ 超参。

---

## 5. Grables: Tabular Learning Beyond Independent Rows

**arXiv ID:** 2602.03945 | [PDF](https://arxiv.org/pdf/2602.03945v1)

**作者:** Tamara Cucumides `[一作]` (University of Antwerp), Floris Geerts `[通讯]` (University of Antwerp)

**通讯引用:** 4756 | [OpenAlex ID](https://openalex.org/A5087587379)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种统一框架“grables”，将表格到图结构的转换与节点预测器分离，用来分析表格学习中行级预测与跨行结构建模的表达能力；

**💡 创新点**

创新点在于构建可解释的图构造器与预测器接口，利用逻辑与图神经网络的对应关系，对行级目标进行可表达性分层，揭示行本地预测无法捕捉的全局计数、重叠等依赖；

**🔧 技术方法**

使用逻辑（FO与GML）与多层消息传递神经网络（MPNN）作为理论工具，并实现多种图构造器（trivial、incidence、NFA等）与基线模型（LightGBM、realMLP、TabPFN、HeteroSage等）；

**📊 数据集**

实验数据包括合成交易数据、真实零售交易数据以及RelBench的临床试验数据；

**📈 对比分析**

通过ROC‑AUC与F1评估，实验表明在只涉及跨行依赖的任务中，带有显式消息传递的图模型明显优于纯行本地表格模型，而将图特征拼接到表格模型可进一步提升性能；

**⚠️ 局限性**

局限性包括：构造器和预测器选择需先验知识；对更深层次关系的表达仍受限；并非所有任务都能从显式结构获益，增加的计算成本与数据工程复杂度需要权衡。

---

## 6. I Can't Believe It's Not a Valid Exploit

**arXiv ID:** 2602.04165 | [PDF](https://arxiv.org/pdf/2602.04165v1)

**作者:** Derin Gezgin `[一作]` (Connecticut College), Claire Wang `[通讯]` (University of Pennsylvania)

**通讯引用:** 2557 | [OpenAlex ID](https://openalex.org/A5101558021)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于 SAST 与 LLM 的框架 PoC-Gym，用于自动生成 Java 漏洞的 PoC 并通过执行验证和反馈循环进行迭代改进。

**💡 创新点**

创新点在于将静态分析生成的源-汇踪迹作为引导信息提供给 LLM，并结合 AspectJ 动态 Instrumentation 进行后置验证，从而降低假阳性率并揭示 LLM 生成 PoC 的错误模式。

**🔧 技术方法**

技术包括大语言模型（Claude Sonnet 4、GPT‑5 Medium、gpt‑oss‑20b）、静态源-汇追踪工具（CodeQL、Semgrep、Snyk）、AspectJ 运行时跟踪、动态验证反馈循环与多轮 Prompt 生成。

**📊 数据集**

使用 CWE‑Bench‑Java 数据集，挑选了 20 个真实 CVE（包括其漏洞及补丁代码）进行评估，并手工审计生成的 PoC。

**📈 对比分析**

与先前基线 FaultLine 对比，PoC-Gym 在含 Trace 引导下的成功率提升 21%，但后置验证显示约 44% 的“成功” PoC 实际无效，71.5% 的 PoC 通过人工检查后被判为无效，说明自动化指标存在严重过估问题。

**⚠️ 局限性**

局限性包括：LLM 在生成 CVE 描述和 Trace 选择时易产生错误；现有的验证机制（输出、打印、异常忽略）仍无法全面捕捉漏洞路径；缺乏强执行级别的安全保证，导致大量假阳性；且对不同 LLM 的泛化能力与成本评估仍待深入。

---

## 7. Entropy Reveals Block Importance in Masked Self-Supervised Vision Transformers

**arXiv ID:** 2602.03918 | [PDF](https://arxiv.org/pdf/2602.03918v1)

**作者:** Peihao Xiang `[一作]` (Florida International University), Ou Bai `[通讯]` (Florida International University)

**通讯引用:** 5255 | [OpenAlex ID](https://openalex.org/A5085554845)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于信息熵的无数据一次性块级剪枝方法Gardener。

**💡 创新点**

首次利用预训练权重的权重数熵评估Transformer块重要性，无需数据或梯度。

**🔧 技术方法**

使用信息理论（熵）度量、视频MAE模型以及单次剪枝技术。

**📊 数据集**

在VideoMAE-B预训练于Kinetics-400后，使用UCF101动作识别数据集进行微调验证。

**📈 对比分析**

与感知式剪枝、随机、L1/Norm等无数据基线对比，Gardener在不同剪枝率下接近感知式剪枝并优于其他基线，甚至在剪掉91.7%块后仍保持竞争性能。

**⚠️ 局限性**

仅在块级剪枝上有效，无法考虑输入依赖性；对更细粒度结构或其他自监督任务的推广仍有待研究。

---

## 8. Reshaping Action Error Distributions for Reliable Vision-Language-Action Models

**arXiv ID:** 2602.04228 | [PDF](https://arxiv.org/pdf/2602.04228v1)

**作者:** Shuanghao Bai `[一作]` (Xi'an Jiaotong University), Badong Chen `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 17145 | [OpenAlex ID](https://openalex.org/A5077852542)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了时间序列动作误差熵最小化（T-MEE）方法，用以在连续动作视觉‑语言‑动作（VLA）模型中重塑误差分布，提升整体性能。

**💡 创新点**

创新点在于将信息论最小误差熵迁移至动作序列分布层，推出加权变体Cw‑T-MEE和Ew‑T-MEE，实现分布层级的误差约束与鲁棒性提升。

**🔧 技术方法**

技术手段包括Rényi二次熵估计、核加权、加权变体设计，并与传统MSE回归联合训练，应用于多种小规模与大规模VLA架构。

**📊 数据集**

实验数据集涵盖LIBERO、SimplerEnv仿真基准以及真实Agilex Cobot Magic机器人收集的演示轨迹。

**📈 对比分析**

与标准MSE回归及多种VLA基线相比，T-MEE在平衡、少样本、噪声及中度不平衡场景下均能提升成功率（幅度从几个百分点到十几个百分点），且对训练开销和推理效率影响极小。

**⚠️ 局限性**

局限性在于极端任务不平衡或样本极少时，T-MEE效果减弱；在已接近饱和性能的大模型上提升幅度有限。

---

## 9. A Comparative Study of Digital Memristor-Based Processing-In-Memory from a Device and Reliability Perspective

**arXiv ID:** 2602.04035 | [PDF](https://arxiv.org/pdf/2602.04035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 10. Overcoming Barriers to Computational Reproducibility

**arXiv ID:** 2602.03863 | [PDF](https://arxiv.org/pdf/2602.03863v1)

**作者:** Roman Hornung `[一作]` (Ludwig-Maximilians-Universität), Karsten Tabelow `[通讯]` (Weierstrass Institute for Applied Analysis and Stochastics)

**通讯引用:** 1429 | [OpenAlex ID](https://openalex.org/A5085776997)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过文献综述和对多家期刊可重复性政策的评估，提出了简洁、易用的可重复性分析指南，并构建了多层次可重复性标准框架，旨在降低作者和期刊的技术与资源门槛；

**💡 创新点**

创新点在于：①将可重复性实践归纳为可操作的七条准则；②提出了面向跨学科的多层次标准维度（材料可用性、验证范围、验证来源、可重复性范围、代码质量等）；③强调期刊层面可重复性审核的重要性，并为期刊提供可实施的评估结构；

**🔧 技术方法**

主要技术为系统综述、经验总结、模板与清单的设计；利用现有的可重复性编辑流程与工具（如 GitHub、Renviron/Conda、CI 等）做示例；

**📊 数据集**

本文并未使用传统数据集，而是以多家期刊（如Biometrical Journal、BMJ、Computo、JASA 等）的可重复性政策和实践为研究材料；

**📈 对比分析**

方法比较主要是对现有期刊政策的对比与评估，未涉及实验性能指标；通过分析发现不同期刊的可重复性要求差异大，说明统一标准的必要性；

**⚠️ 局限性**

局限性包括：缺乏正式的量化评价体系；多层次标准仍处于概念阶段，需进一步细化与验证；对不同学科的适用性尚需实证检验；

---

## 11. AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent

**arXiv ID:** 2602.03955 | [PDF](https://arxiv.org/pdf/2602.03955v1)

**作者:** Yinyi Luo `[一作]` (Carnegie Mellon University), Jindong Wang `[通讯]` (William and Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种将多代理系统（MAS）中的推理动态压缩到单一模型中的蒸馏框架，通过在训练阶段引入多代理辩论产生的高质量推理轨迹，训练单一模型获得与MAS相似的自我纠错与迭代推理能力；

**💡 创新点**

创新点在于：①提出三种层次化蒸馏策略（R‑SFT、DA、PAD），尤其是过程感知蒸馏（PAD）利用过程奖励模型（PRM）和GRPO实现对推理步骤的细粒度监督；②构建了与MAS无关的通用蒸馏数据生成与知识提取流水线；③在多模型、跨任务、跨尺度上系统评估了蒸馏效果；

**🔧 技术方法**

使用的技术包括：多代理辩论生成推理轨迹、基于推理轨迹的监督式微调（R‑SFT）、多样化推理轨迹数据增强（DA）、过程奖励模型（PRM）与GRPO强化学习的过程感知蒸馏（PAD）；

**📊 数据集**

实验数据集涵盖数学推理（MATH、GSM8K、MetaMathQA）、医学知识（MedMCQA）、多跳问答（HotpotQA）、长文本理解（QASPER）、摘要（QMSum）以及真实性测试（TruthfulQA）等；

**📈 对比分析**

与单一模型及原始MAS对比，蒸馏后单一模型平均提升约4.8%，在多数数据集上逼近MAS性能；PAD方法表现最为稳健，能显著提升推理质量、错误定位与自检能力；在小模型上，PRM容量决定提升幅度；增大代理数量对大模型有益，缩小模型反而无效；

**⚠️ 局限性**

局限性包括：实验仅覆盖有限的推理基准与多模态模型；蒸馏流程对特定的辩论MAS敏感，未尝试其他MAS算法；蒸馏策略与管道相对固定，缺乏自适应与混合方法；

---

## 12. DiGAN: Diffusion-Guided Attention Network for Early Alzheimer's Disease Detection

**arXiv ID:** 2602.03881 | [PDF](https://arxiv.org/pdf/2602.03881v1)

**作者:** Maxx Richard Rahman `[一作]` (German Research Center for Artificial Intelligence), Wolfgang Maass `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种Diffusion-Guided Attention Network（DiGAN），用于利用少量且不规则的纵向神经影像数据进行阿尔茨海默病早期检测。

**💡 创新点**

创新点在于将潜在扩散模型用于生成逼真的纵向影像轨迹，并结合自注意力卷积网络捕捉结构-时间特征，从而克服传统模型对大量连续数据的依赖并提升对不规则访视的鲁棒性。

**🔧 技术方法**

使用了潜在扩散模型（Latent Diffusion）、自注意力卷积单元（Self‑Attention Convolution）以及最大池化聚合等技术来实现生成与判别的联合训练。

**📊 数据集**

实验数据来自于人工合成的Delcode模拟数据以及公开的ADNI纵向磁共振成像数据集。

**📈 对比分析**

与线性、生成式、概率核和集成等多种基线方法（如ALASCA、TVAE、AnoGAN、GP、LSCP、SUOD、IsoForest）进行对比，在合成数据上实现了0.948的准确率和0.998的AUC，在ADNI数据上实现了0.710的准确率和0.700的AUC，均显著优于传统方法。

**⚠️ 局限性**

主要限制是对真实临床数据中更复杂的多模态不规则性和极端缺失的处理仍有提升空间，且在高度不平衡或少样本的ADNI子集上性能相对下降。

---

## 13. Enhancing Mathematical Problem Solving in LLMs through Execution-Driven Reasoning Augmentation

**arXiv ID:** 2602.03950 | [PDF](https://arxiv.org/pdf/2602.03950v1)

**作者:** Aditya Basarkar `[一作]` (North Carolina State University), Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为Iteratively Improved Program Construction（IIPC）的数学推理框架，结合程序执行反馈和token级Chain‑of‑Thought双分支进行迭代式程序改进与错误修正。

**💡 创新点**

创新点在于：①将程序视为可编辑的推理链，②引入持久的错误反射记忆避免重复失败路径，③采用双分支架构在最终输出时融合程序结果与自然语言推理，减弱执行偏差。

**🔧 技术方法**

使用大型语言模型（GPT‑4o mini、Gemini 2.0 Flash、Mistral 3.2 24B、Gemma 3 27B、Llama 4 Maverick）+代码解释器进行程序生成与执行，并结合Chain‑of‑Thought与Program‑of‑Thought提示技术。

**📊 数据集**

评测数据集包括 MATH、AIME、GSM8K 三个数学推理基准。

**📈 对比分析**

与 PoT、CR、MACM、CoT 等方法对比，IIPC 在 4/5 个模型的 MATH 与 AIME 基准上均实现最高或近乎最高的准确率（如 Gemini 2.0 Flash MATH 94.13%），在简单问题集 GSM8K 上与 CoT 相当，且各项消融实验验证了双分支、反射记忆与迭代改进对性能的积极作用。

**⚠️ 局限性**

局限性：1）对 token 需求高，推理成本大；2）在推理/编程能力有限的模型（如 GPT‑4o mini）上性能不及 PoT；3）对模型的推理容量与代码执行能力要求较高。

---

## 14. Paint by Odor: An Exploration of Odor Visualization through Large Language Model and Generative AI

**arXiv ID:** 2602.04159 | [PDF](https://arxiv.org/pdf/2602.04159v1)

**作者:** Gang Yu `[一作]` (Academy of Arts and Design, Tsinghua University), Qi Lu `[通讯]` (The Future Laboratory, Tsinghua University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 Paint by Odor 系统，利用大语言模型和生成式 AI 将人类嗅觉感知转化为富有美感的视觉图像。

**💡 创新点**

创新点在于将语言描述作为媒介，结合专家 GPT 生成图像提示，系统实现了多维度、抽象风格的嗅觉可视化，并首次评估了 AI 与人类在嗅觉描述和图像生成上的一致性。

**🔧 技术方法**

使用 GPT‑4 进行嗅觉描述生成与专家 GPT 微调生成视觉提示，并通过 Midjourney 生成抽象与具象图像。

**📊 数据集**

利用 20 种日常家用气味的实测样本，并基于 GS‑LF 等 138 项嗅觉词表与自制的 34 项词集进行描述评估。

**📈 对比分析**

通过 30 名参与者的嗅觉描述对比 28 名参与者的图像评估，采用 Wilcoxon 检验等非参数统计，结果显示具象 Prompt 在对应度上显著优于抽象 Prompt，而抽象 Prompt 在美感与讨论度上更高，AI 自动生成图像的对应度略低但美感相近。

**⚠️ 局限性**

主要局限包括生成图像在一致性、透明度与文化语义上的偏差、AI 对稀有或浓度变化的嗅觉把握不如人类，以及缺乏实时嗅觉感知模块。

---

## 15. Futuring Social Assemblages: How Enmeshing AIs into Social Life Challenges the Individual and the Interpersonal

**arXiv ID:** 2602.03958 | [PDF](https://arxiv.org/pdf/2602.03958v1)

**作者:** Lingqing Wang `[一作]` (Georgia Institute of Technology), Ashok Goel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8325 | [OpenAlex ID](https://openalex.org/A5007028896)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开展了一项三阶段的设计研究（创意研讨、设计故事板、访谈调研），通过24名在线研究生参与者探讨人工智能在社交生活中的共适应与潜在影响。

**💡 创新点**

提出了从传统以用户为中心的设计范式向更具“交互性”和“挑衅性”的设计视角转变，强调在人工智能介入社交场景时对真实性、隐私、主体性和身份认同的长期风险。

**🔧 技术方法**

主要采用了参与式设计与推测设计方法（设计虚构、情景板等），并未实现具体技术原型或算法。

**📊 数据集**

研究数据来源为24名计算机科学在线研究生的访谈与工作坊记录（无公开数据集）。

**📈 对比分析**

本研究未进行算法性能比较或实验评估，缺乏可度量的性能指标；评估依据为参与者的定性反馈与主题分析。

**⚠️ 局限性**

局限性包括：样本同质化（仅技术背景的在线研究生），社交情境单一（专业/学习环境），研究方法为推测设计而非真实部署，文化背景单一（主要西方背景）。

---

## 16. Child Mortality Prediction in Bangladesh: A Decade-Long Validation Study

**arXiv ID:** 2602.03957 | [PDF](https://arxiv.org/pdf/2602.03957v1)

**作者:** Md Muhtasim Munif Fahim `[一作]` (University of Rajshahi), Md Rezaul Karim `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用孟加拉国人口与健康调查（BDHS）2011-2022年的出生记录，构建了严格时间分割（训练 2011-2014，验证 2017，测试 2022）的神经架构搜索模型，用于预测5岁以下儿童死亡风险。

**💡 创新点**

创新点在于：① 基于遗传算法的NAS得到极简单层网络（64 单元），显著优于传统梯度提升树；② 发现并量化了“社会经济预测梯度”，揭示模型在贫困地区的预测性能更高，挑战了传统公平性评估方法。

**🔧 技术方法**

技术包括：领域驱动的特征工程、基于遗传算法的NAS、单层深度学习模型、时间序列验证、Platt 缩放校准、SHAP 可解释性、加权 AUROC 校正，以及与多种基线模型（逻辑回归、XGBoost、LightGBM、随机森林、TabNet、ResNet）的性能对比。

**📊 数据集**

使用数据集：孟加拉国人口与健康调查（BDHS）2011、2014、2017、2022 四个周期，共计 33,962 名出生记录（训练 14,380，验证 8,044，测试 11,538）。

**📈 对比分析**

通过在 2022 年完全保留的测试集上与六种基线模型比较，NAS 的 AUROC 为 0.766，显著优于 XGBoost 的 0.730（p < 0.01），并在贫困地区（Sylhet、Rangpur、Mymensingh）实现 AUROC > 0.72，表明模型在这些地区具有较高的识别效率。

**⚠️ 局限性**

局限性包括：① 仅能在出生后 5 年内进行预测，缺乏产前预测能力；② 关键医疗变量缺失率高（55-58%）；③ 观察性调查数据限制因果推断；④ 在富裕地区预测性能下降，说明模型受疾病病因结构影响。

---

## 17. Scalable Explainability-as-a-Service (XaaS) for Edge AI Systems

**arXiv ID:** 2602.04120 | [PDF](https://arxiv.org/pdf/2602.04120v1)

**作者:** Samaresh Kumar Singh `[一作]`, Joyjit Roy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种可扩展的解释即服务 (XaaS) 架构，将解释生成与推理解耦，实现了边缘 AI 系统的高效、可缓存、可验证的可解释性服务。

**💡 创新点**

创新点在于三方面：①使用语义相似度检索的分布式缓存显著降低重复计算；②轻量级验证协议保证缓存解释的可信度；③自适应解释引擎根据设备能力和用户需求动态选择解释方法。

**🔧 技术方法**

采用了 FAISS 进行最近邻搜索、CLIP/BERT 生成嵌入、LIME/SHAP/GradCAM 等解释方法、轻量级扰动验证以及服务编排器实现请求路由和负载均衡。

**📊 数据集**

在制造质量控制、自动驾驶感知和医疗诊断三类真实边缘场景中，分别使用了 150 台设备、80 辆车和 200 名患者的采样数据（共计约 400K 条样本）。

**📈 对比分析**

与本地生成、云端 XAI、边缘 XAI 以及联邦 XAI 等基线相比，XaaS 在延迟上降低约 38%，通过 72% 的缓存命中率提升吞吐量 3.2 倍，同时保持 0.92 以上的解释可信度。

**⚠️ 局限性**

主要限制包括冷启动需 2–4 小时收敛、对频繁模型更新时缓存失效率上升、对高动态或低重复率场景的适用性有限，以及对极低资源设备缓存实现的开销较大。

---

## 18. Piece of CAKE: Adaptive Execution Engines via Microsecond-Scale Learning

**arXiv ID:** 2602.04181 | [PDF](https://arxiv.org/pdf/2602.04181v1)

**作者:** Zijie Zhao `[一作]` (University of Pennsylvania), Ryan Marcus `[通讯]` (University of Pennsylvania)

**通讯引用:** 1777 | [OpenAlex ID](https://openalex.org/A5025731013)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于微秒级上下文多臂赌博机的学习式内核选择框架 CAKE，能够在数据库执行过程中自适应挑选最佳物理实现。

**💡 创新点**

创新点在于利用低成本反事实反馈构造上下文多臂赌博机，并将学习得到的策略编译为极低延迟的 Regret Tree，实现毫秒级查询中的在线学习与即刻优化。

**🔧 技术方法**

主要技术包括局部加权自举/CLT 近似的上下文多臂赌博机、反事实采样、回归树以及 Regret Tree 编译。

**📊 数据集**

实验使用了 IMDb、StackOverflow、DSB 三个真实数据集。

**📈 对比分析**

与单一最佳实现、手写启发式、UCB 等基线对比，CAKE 在尾部和总延迟上相较最佳启发式提升约 2 倍，且与 Oracle 的性能差距不足 5%–10%。

**⚠️ 局限性**

局限性包括：仅针对低层物理实现，无法直接处理大规模计划重排；需手动提供特征提取函数；在极端分布或非常稀疏的数据集上性能略逊；内存/缓冲区大小会影响推断延迟。

---

## 19. Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement

**arXiv ID:** 2602.03983 | [PDF](https://arxiv.org/pdf/2602.03983v1)

**作者:** Weikang Qiu `[一作]` (Yale University), Rex Ying `[通讯]` (Yale University)

**通讯引用:** 15024 | [OpenAlex ID](https://openalex.org/A5078337825)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于视觉-语言-动作模型的静态-动态分离框架，通过将视觉信息拆分为静态与动态子集，并使用可学习的 recache gate 实现关键-值缓存重用，以实现长时序记忆与推理；

**💡 创新点**

创新点包括：① 视觉令牌的多级静态-动态分离，① 只保留一次静态令牌，② 引入可学习的 recache gate 适时刷新缓存，③ 设计新的 LIBERO‑Memory 基准以评估长时序依赖；

**🔧 技术方法**

采用的技术包括：Vision‑Language‑Action (VLA) 体系结构、Transformer‑based 大规模视觉-语言模型、对齐的对比学习（InfoNCE）约束静态令牌、Gumbel‑softmax 训练可学习的缓存门、KV‑cache 关键-值重用；

**📊 数据集**

主要使用的数据集包括：Open‑X‑Embodiment（用于预训练与评估）、Robosuite（生成基准数据）、LIBERO‑Memory（新基准）以及 SimplerEnv；

**📈 对比分析**

与多种基线（如 TTF‑VLA、TraceVLA、MemoryVLA、ContextVLA、FlashVLA 等）比较，实验表明该方法在 LIBERO‑Memory 任务上成功率提升至 39.8%（绝对）并在 SimplerEnv 上实现 2.26× 的推理速度提升；

**⚠️ 局限性**

局限性主要在于：① 方案基于已有预训练 VLA，未从头训练；② 缓存门的阈值与层级设定仍需经验调参；③ 对极端动态场景的适应性未做深入验证；

---

## 20. Topology-Aware Revival for Efficient Sparse Training

**arXiv ID:** 2602.04166 | [PDF](https://arxiv.org/pdf/2602.04166v1)

**作者:** Meiling Jin `[一作]`, Yuan Cheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种一次性拓扑感知恢复（Topology-Aware Revival，TAR），在静态稀疏训练后为每层分配少量恢复预算并随机激活部分被剪枝的连接，从而在不改变训练期间掩码的前提下提升模型鲁棒性。

**💡 创新点**

创新点在于将拓扑信息（层级连通性缺口）与恢复预算相结合，只需一次性操作即可补救因早期结构承诺导致的瓶颈，且不需要后续动态重连。

**🔧 技术方法**

采用稀疏训练（static pruning）与一次性恢复的联合技术，利用层级连通性阈值（E^topo = N ln N /2）计算恢复量，并在已剪枝参数集合中做无放回随机采样。

**📊 数据集**

在多种连续控制任务（HalfCheetah-v4、BipedalWalker-v3、Humanoid-v4）上使用TD3和SAC算法，评估稀疏网络的学习表现。

**📈 对比分析**

与传统静态稀疏训练、均匀恢复（Uniform Revival）以及动态稀疏训练（SET/RigL）对比，TAR 在静态基线上平均提升约13.5%，在最优设置下可提升约37.9%；且在网络宽度放大到1024时仍保持优势。

**⚠️ 局限性**

局限性包括：恢复预算选择仍需经验性调节；在极端稀疏或不同网络结构（如CNN）时效果未充分验证；恢复仅一次，可能不适用于更剧烈的分布漂移场景。

---

## 21. Vision Transformers for Zero-Shot Clustering of Animal Images: A Comparative Benchmarking Study

**arXiv ID:** 2602.03894 | [PDF](https://arxiv.org/pdf/2602.03894v1)

**作者:** Hugo Markoff `[一作]` (Aalborg University), Michael Ørsted `[通讯]` (Aalborg University)

**通讯引用:** 1274 | [OpenAlex ID](https://openalex.org/A5023358897)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 Vision Transformer 在无监督聚类下对摄像机捕捉动物图像进行物种级别分组的可行性。

**💡 创新点**

通过系统的27,600组合实验验证DINOv3与t‑SNE/HDBSCAN可实现近乎完美的物种聚类，并揭示聚类失败往往代表生物学亚组。

**🔧 技术方法**

采用ViT基础模型（DINOv2/3、CLIP、SigLIP、BioCLIP2）、t‑SNE/UMAP/PCA/Isomap/KPCA降维以及DBSCAN/HDBSCAN/层次聚类/GMM等方法。

**📊 数据集**

使用60个哺乳类与鸟类共139,111张手工校验的摄像机捕捉图像，随机抽样200张/物种进行实验。

**📈 对比分析**

与5种模型、23种降维、12种聚类共27,600次配置对比，DINOv3+t‑SNE+HDBSCAN在物种级别实现V‑measure 0.958，未监督方案可达0.943，且仅1.14%样本被标记为异常。

**⚠️ 局限性**

限制包括仅评估哺乳类与鸟类、降维至二维可能丢失信息、对极度不均衡数据需手动调参，且模型对视觉相似的物种仍会聚类混合。

---

## 22. Context Determines Optimal Architecture in Materials Segmentation

**arXiv ID:** 2602.04154 | [PDF](https://arxiv.org/pdf/2602.04154v1)

**作者:** Mingjian Lu `[一作]` (Case Western Reserve University), Yinghui Wu `[通讯]` (Case Western Reserve University)

**通讯引用:** 4101 | [OpenAlex ID](https://openalex.org/A5071093153)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种跨模态材料图像分割评估框架，能够在SEM、AFM、XCT和光学显微等多种成像模式下系统比较不同编码‑解码组合的性能，并提供部署可靠性信号和可解释性热图。

**💡 创新点**

创新点在于：①基于配置化的跨模态比较；②将离群检测与模型可靠性评估集成；③使用对抗性反事实解释揭示微观结构特征对预测的影响；并展示同一模型在不同材料/模态下性能显著差异。

**🔧 技术方法**

技术上采用ResNet50/SE‑ResNeXt101编码器与UNet/DeepLabv3/DeepLabv3+解码器的组合，结合自监督嵌入、典型性估计的OOD检测以及局部扰动的反事实解释方法。

**📊 数据集**

使用七个公开/内部材料数据集，覆盖AFM晶体、SEM接触腐蚀、XCT渗漏/应力腐蚀、光学L‑PBF监测等，合计八种具体任务。

**📈 对比分析**

通过对所有六种编码‑解码组合在各数据集上计算IoU和FORTE等指标进行系统对比，结果显示SE‑ResNeXt101+UNet在高对比度2D任务上IoU>0.90，DeepLabv3+在3D多尺度任务上最优；框架还能给出离群检测分数和解释热图。

**⚠️ 局限性**

局限在于：①评估仍基于单一超参设置；②对动态时间序列和高噪声3D数据的处理能力有限；③OOD检测和解释生成耗时，未集成实时部署。

---

## 23. The Dynamics of Attention across Automated and Manual Driving Modes: A Driving Simulation Study

**arXiv ID:** 2602.04164 | [PDF](https://arxiv.org/pdf/2602.04164v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 24. Evaluating the Vulnerability Landscape of LLM-Generated Smart Contracts

**arXiv ID:** 2602.04039 | [PDF](https://arxiv.org/pdf/2602.04039v1)

**作者:** Hoang Long Do `[一作]` (Deakin University), Muneeb Ul Hassan `[通讯]` (Deakin University)

**通讯引用:** 1942 | [OpenAlex ID](https://openalex.org/A5016187400)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了使用 GPT‑4、Gemini‑2.5、Sonnet‑4.5 等主流 LLM 生成的 Solidity 合约的安全性，量化并归纳其漏洞分布与模式。

**💡 创新点**

创新点在于首次对 LLM 生成的合约进行大规模、跨模型、跨功能域的漏洞量化分析，并提出漏洞与代码行数的线性关系模型。

**🔧 技术方法**

主要技术手段包括：利用 LLM 生成合约、Slither 静态分析框架检测漏洞、Poisson 线性模型评估漏洞数与 LOC 的关系。

**📊 数据集**

数据集来源于 SmartBugs、OWASP、SWC 三大公开漏洞数据库共 52 篇合约，经过提示生成后产生 34 个提示，进一步生成 1,033 份合约进行分析。

**📈 对比分析**

比较方法为将三种 LLM 生成合约的漏洞率、漏洞种类和严重程度进行对比，结果显示 GPT‑4.1 约 47.4%、Gemini‑2.5 53.2%、Sonnet‑4.5 75.5%，且漏洞数随 LOC 增长呈线性上升。

**⚠️ 局限性**

主要限制包括提示生成过程手工完成、仅使用静态分析工具 Slither（无法捕获运行时或市场攻击等复杂漏洞）、以及未考虑多模型协同生成与自动化流水线等场景。

---

## 25. iSight: Towards expert-AI co-assessment for improved immunohistochemistry staining interpretation

**arXiv ID:** 2602.04063 | [PDF](https://arxiv.org/pdf/2602.04063v1)

**作者:** Jacob S. Leiby `[一作]` (University of Pennsylvania), Zhi Huang `[通讯]` (University of Pennsylvania)

**通讯引用:** 8380 | [OpenAlex ID](https://openalex.org/A5022500975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了规模达10,495,672张免疫组织化学（IHC）图像的数据集HPA10M，并提出了一种多任务学习框架iSight，用于自动化评估IHC染色强度、亚细胞定位、染色量、组织类型和恶性程度。

**💡 创新点**

创新点在于：①首次构建覆盖人类蛋白组的IHC大规模数据集；②iSight通过结合Vision Transformer、token级注意力和结构化元数据实现多任务联合学习，显著提升IHC判读精度；③引入置信度校准和多任务交叉验证等评估指标。

**🔧 技术方法**

采用Vision Transformer（CLIP‑ViT‑Large‑Patch‑14‑336）进行图像特征提取，配合CLIP文本编码器、gated attention多实例学习、token‑级加权池化以及并行分类头实现多任务预测。

**📊 数据集**

使用来自Human Protein Atlas的10,495,672张IHC图像（HPA10M），以及Stanford Tissue Microarray Database中的100张IHC图像做外部验证。

**📈 对比分析**

与两大基础模型PLIP和CONCH对比，iSight在染色定位、强度、量化任务上的准确率分别为85.5%、76.6%、75.7%，比基线提升2.5–10.2%；在辅助任务组织类型和恶性判定的准确率高达95.7%和99.9%；校准误差低，用户研究中辅助路径学家准确率提升约10%。

**⚠️ 局限性**

局限性包括：缺乏跨机构外部验证；未结合H&E、临床检验等多模态数据；对罕见染色模式和稀有肿瘤类型的表现不确定；HPA标签本身并非绝对金标准，影响最终评估。

---

## 26. Approximately Partitioning Vertices into Short Paths

**arXiv ID:** 2602.03991 | [PDF](https://arxiv.org/pdf/2602.03991v1)

**作者:** Mingyang Gong `[一作]` (Montana State University), Brendan Mumey `[通讯]` (Montana State University)

**通讯引用:** 1045 | [OpenAlex ID](https://openalex.org/A5084888509)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了针对k⁻-路径划分问题（k≥3）的两种近似算法，分别在k=9、10时获得k+4/5近似比，在k≥11时获得(√11−2/7k+9−√11/7)近似比。

**💡 创新点**

创新点在于引入最大三角形自由路径-环覆盖并结合[f,g]-因子变换，将k⁻-路径划分转化为最大边数k⁻-路径划分（kPPE），通过在辅助图中构造最大加权路径-环覆盖有效连接短环，显著提升近似比；同时设计了三种操作递归消除不平衡连通分量中的2-锚点。

**🔧 技术方法**

核心技术包括：最大三角形自由路径-环覆盖、[f,g]-因子最大加权求解、图的辅助构造与加权路径-环覆盖、递归与操作3（消除2-锚点）以及对特殊连通分量的局部构造。

**📊 数据集**

本文未使用公开数据集，全部实验与分析基于理论证明和算法复杂度评估。

**📈 对比分析**

与先前最佳算法比较，k=9、10时比率分别为2.600、2.800，k≥11时比率从2.881提升至3.198（k=18），在k=9–18区间内实现了当前最优近似比；算法时间复杂度为O(n³m²)（k=9,10）或O(n⁴m²)（k≥11）。

**⚠️ 局限性**

主要局限在于：算法复杂度较高，无法直接扩展至k>18或有向图；对特殊图结构（如树、cograph）未进一步优化；对常数k的逼近下界仍未知。

---

## 27. Pruning for Generalization: A Transfer-Oriented Spatiotemporal Graph Framework

**arXiv ID:** 2602.04153 | [PDF](https://arxiv.org/pdf/2602.04153v1)

**作者:** Zihao Jing `[一作]` (Western University), Ganlin Feng `[通讯]` (Western University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种面向迁移学习的稀疏时空图框架TL-GPSTGN，通过信息熵与相关性双重标准进行图上下文剪枝，从而提升在数据稀缺和跨域情境下的交通流预测性能。

**💡 创新点**

创新点包括：① 将熵-相关性双重指标用于边重要性评估，精准识别并剔除边界节点与弱关联边；② 将图剪枝嵌入转移学习流程，使源域与目标域共享更干净、语义化的子图；③ 通过可迭代的外层节点剪枝实现多层图净化，显著降低跨域噪声。

**🔧 技术方法**

技术实现：信息熵分析器（IEA）计算节点不确定性；相关性判据衡量节点间时间相关度；双重得分构造边重要性；阈值/Top-k 方式进行图剪枝；随后使用标准STGCN作为后端时空卷积网络；训练分两步：源域预训练 + 目标域微调。

**📊 数据集**

实验数据集：METR‑LA、PEMS‑BAY、PEMSD7，均为大规模交通流时间序列数据，覆盖不同城市与网络规模。

**📈 对比分析**

方法与传统基线（HA、ARIMA、FNN、FC‑LSTM、STGCN）进行比较。单域预测上TL‑GPSTGN与STGCN相近，且在低数据迁移场景下显著优于STGCN，提升MAE/MAPE/RMSE，尤其在目标域训练样本不足5%时可获得 10–15% 的误差下降。

**⚠️ 局限性**

局限性：① 对图剪枝阈值与层数依赖经验设定，缺乏自适应机制；② 在单域最佳性能并不一定优于STGCN，可能在某些数据稠密场景下略逊；③ 仅考虑空间拓扑与时序相关性，未纳入天气、事件等外部因素。

---

## 28. Likelihood-Based Reward Designs for General LLM Reasoning

**arXiv ID:** 2602.03979 | [PDF](https://arxiv.org/pdf/2602.03979v1)

**作者:** Ariel Kwiatkowski `[一作]` (Meta FAIR), Yann Ollivier `[通讯]` (Meta FAIR)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过强化学习将大语言模型在推理任务中的链式思考（CoT）进行微调，重点探索以参考答案的对数概率作为奖励信号的方法；

**💡 创新点**

创新点在于首次系统评估并证明对数概率奖励能够在可验证与不可验证域中统一提升性能，既匹配或超过传统 0/1 奖励，又显著改善困惑度；

**🔧 技术方法**

技术包括基于 RL 的 CoT 微调（RLOO、GRPO、PPO 等），并实现多种奖励变体：基础 RL、概率奖励（VeriFree、RLPR、NOVER）、对数概率奖励（Logprob、AvgLogprob）、JEPO 组奖励；

**📊 数据集**

使用的数据集覆盖可验证的 MATH、DeepScaleR 以及不可验证的长文本 Alpaca 与 NuminaProof；

**📈 对比分析**

与标准 SFT、基础 RL 和各概率奖励进行对比；在可验证任务上，对数概率奖励在成功率上与 0/1 RL持平，同时在困惑度上优于 SFT；在不可验证任务上，对数概率奖励与 SFT 的性能相当，而纯概率奖励表现差强人意；

**⚠️ 局限性**

局限性包括：在长文本任务中，对数概率奖励会导致 CoT 缩短甚至消失；JEPO 需要更高的计算成本和更长训练时间；对长 CoT 的探索仍受信号噪声和信用分配问题限制。

---

## 29. Audit After Segmentation: Reference-Free Mask Quality Assessment for Language-Referred Audio-Visual Segmentation

**arXiv ID:** 2602.03892 | [PDF](https://arxiv.org/pdf/2602.03892v1)

**作者:** Jinxing Zhou `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Hisham Cholakkal `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 3143 | [OpenAlex ID](https://openalex.org/A5009362997)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在语言引导的音视频分割任务中，提出了无参考掩码质量评估（MQA-RefAVS）并实现了MQ-Auditor模型，能够在无标注条件下评估掩码的IoU、错误类型与改进建议。

**💡 创新点**

①首次定义并正式提出无参考掩码质量评估任务；②构建首个针对该任务的 MQ‑RAVSBench 基准；③设计基于多模态大语言模型的 MQ‑Auditor，能够在推理时直接估计 IoU、识别错误并给出可执行的质量控制建议。

**🔧 技术方法**

采用多模态大语言模型（LLaMA‑2‑7B‑Chat）与音频/视觉编码器（BEATs、CLIP‑ViT‑L/14）、Q‑Former 以及 LoRA 微调；利用掩码、掩码帧及语义信息进行联合推理，直接输出 IoU、掩码类型与动作建议。

**📊 数据集**

基于 Ref‑AVSBench 构建 MQ‑RAVSBench，包含 1,840 个视频、2,046 条参考文本，并生成 26,061 个掩码样本，覆盖六种典型错误模式（perfect、cutout、dilate、erode、merge、full_neg）。

**📈 对比分析**

与 Video‑LLaMA3‑7B、Qwen2.5‑Omni‑7B、Ming‑Flash‑Omni、Gemini‑3‑Flash 等多模态 LLM 在图像基和视频基评估协议下进行对比；MQ‑Auditor 在 IoU 估计、错误分类（F₂‑M）和动作推荐（F₂‑A）上均显著优于对手，尤其在检测负面掩码（full_neg）方面表现突出。

**⚠️ 局限性**

模型受限于大语言模型的潜在偏见和对极端错误掩码的识别仍不够稳定；正负样本比例对性能影响显著，需精细调节；在安全关键场景仍需人工监督，且对复杂多样化环境的泛化能力尚待提升。

---

## 30. WebAccessVL: Making an Accessible Web via Violation-Conditioned VLM

**arXiv ID:** 2602.03850 | [PDF](https://arxiv.org/pdf/2602.03850v1)

**作者:** Amber Yijia Zheng `[一作]` (Purdue University), Raymond A. Yeh `[通讯]` (Purdue University)

**通讯引用:** 3341 | [OpenAlex ID](https://openalex.org/A5076130922)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉‑语言模型的网页 HTML 自动编辑方法，用来消除 WCAG2 可访问性违规并保持原始设计。

**💡 创新点**

创新点在于：①将可访问性改正视为图像条件程序合成任务；②构建 WebAccessVL 数据集；③引入违规计数条件与负向引导采样，提升模型对视觉违规的感知与纠正能力。

**🔧 技术方法**

核心技术：视觉‑语言模型（VLM）自监督微调、违规条件编码、负向引导采样、LLM+视觉编码器融合。

**📊 数据集**

使用 WebAccessVL 数据集：2500 个网页的原始与人工纠正后的 HTML 对，包含 26 种违规标签，约 7–10 分钟/页人工编辑。

**📈 对比分析**

与 16 种公开 LLM / VLM（如 GPT‑4o、GPT‑5、Claude、Gemini、LLaVA、Llama‑3.2 Vision 等）以及自定义 VLM 进行对比。实验显示平均违规数从 5.34 降至 0.44，较 GPT‑5 下降 73.9%；结构一致性（SSIM ≥ 0.9）和树编辑距离均优于基线。

**⚠️ 局限性**

局限性：①仍需大量人工标注；②对高度动态或交互式网页的适用性未知；③模型训练与推理需要显存较大（48GB GPU）；④在极端视觉违规场景下仍可能保留少量违规。

---

## 31. Fast, Unsupervised Framework for Registration Quality Assessment of Multi-stain Histological Whole Slide Pairs

**arXiv ID:** 2602.04046 | [PDF](https://arxiv.org/pdf/2602.04046v1)

**作者:** Shikha Dubey `[一作]` (Johnson and Johnson), Erik Ames Burlingame `[通讯]` (Johnson and Johnson)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种无监督的配准质量评估框架URQA，用于在缺乏真值标注的情况下，对H&E与IHC全切片图像的配准质量进行快速、可靠评估。

**💡 创新点**

创新点在于：①将组织掩膜几何一致性指标与变形场正则性指标统一到同一评分体系，实现对配准结果的全方位评估；②采用低分辨率掩膜与压缩变形场，显著降低计算成本，能够在秒级完成评估；③无监督设计，避免了对人工标注的依赖，便于大规模部署。

**🔧 技术方法**

技术细节包括：利用Otsu阈值和形态学操作生成二值组织掩膜；计算IoU、MAE和三种直方图相关系数来评估全局几何对齐；对变形场做幅值、方向、Jacobian行列式以及平滑残差统计，设置IQR阈值检验正则性；最终按层级规则合并MRQA和DRQA得分得到总分；实现基于PyVIPS的内存高效图像读取，配合VALIS配准模型。

**📊 数据集**

使用内部构建的300对H&E–IHC全切片图像，涵盖HER2、cMET、EGFR、PD‑L1四种标记，数据来自同一组织块，切片厚度差异在5–10µm内约95%，其余5%为大于50µm的间距。

**📈 对比分析**

通过与单独的MRQA、DRQA模块以及两位专家（E‑1、E‑2）对比，URQA在多专家验证的33例样本中显示出较高的一致性（真阳性率保持不变，假阳性率显著降低）。在平均精度（AP）、平均召回（AR）和F1分数上，URQA均略优于单模块，说明综合评估提升了鲁棒性和可信度。

**⚠️ 局限性**

局限性包括：①对掩膜生成的依赖导致在组织形态异常或染色不均时可能产生误判；②验证仅在VALIS配准模型上完成，缺乏公开数据集（如ANHIR、ACROBAT）上的泛化评估；③在切片间距超过50µm的极端情况下，评估效果尚未充分验证；④由于低分辨率掩膜和变形场的使用，可能忽略细粒度配准错误。

---

## 32. HoloEv-Net: Efficient Event-based Action Recognition via Holographic Spatial Embedding and Global Spectral Gating

**arXiv ID:** 2602.04182 | [PDF](https://arxiv.org/pdf/2602.04182v1)

**作者:** Weidong Hao `[一作]` `[通讯]`, Weidong Hao

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了高效事件摄像机动作识别框架 HoloEv-Net，解决了传统体素表示与多分支结构的计算与冗余问题。

**💡 创新点**

核心创新包括：1)紧凑的全息时空表示（CHSR），在单一 T-H 视图中嵌入水平空间信息；2)全局频谱门控（GSG）模块，利用 FFT 在频域进行全局特征混合。

**🔧 技术方法**

使用了事件流离散化、三通道 CHSR、轻量级 backbone（ResNet‑18 / MobileNetV3‑Small）、FFT‑based GSG、残差融合与门控等技术。

**📊 数据集**

在 THU‑EACT‑50‑CHL、HARDVS 与 DailyDVS‑200 三个事件动作基准数据集上进行评估。

**📈 对比分析**

与多种基线（如 ResNet、SlowFast、MVFNet 等）对比，HoloEv‑Net‑Base 在 Top‑1 精度上分别提升 10.29%、1.71% 与 6.25%，而 HoloEv‑Net‑Small 仅 4.4 M 参数、0.1 G FLOPs、7.4 ms 延迟，展示了卓越的性能与边缘部署潜力。

**⚠️ 局限性**

在极其复杂或多主体场景（如 HARDVS）提升有限，频谱门控虽对摄像机运动鲁棒性有优势，但仍需引入自适应频率学习与更深层次特征融合。

---

## 33. Benchmarking Uncertainty Quantification of Plug-and-Play Diffusion Priors for Inverse Problems Solving

**arXiv ID:** 2602.04189 | [PDF](https://arxiv.org/pdf/2602.04189v1)

**作者:** Xiaoyu Qiu `[一作]` (University of Michigan), Liyue Shen `[通讯]` (University of Michigan)

**通讯引用:** 6271 | [OpenAlex ID](https://openalex.org/A5072483985)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Plug‑and‑Play Diffusion Prior（PnPDP）逆问题求解器的不确定性量化（UQ）进行系统性基准，结合 toy 模型与真实数据评估其分布特性。

**💡 创新点**

提出了“准确性陷阱”概念、基于 UQ 的方法分类与评估框架，证明相同重建质量的求解器可能产生截然不同的不确定性表现。

**🔧 技术方法**

采用 PnPDP 方法（如 DPS、REDDiff、DiffPIR、DAPS 等）、Gaussian Mixture toy 先验、像素级方差统计、置信区间覆盖率等技术，对比不同求解器的后验逼近能力。

**📊 数据集**

使用多组数据集：toy GMM、线性逆散射（多接收器）、fastMRI 肿瘤 MRI（×4/×8 加速）、LIDC‑IDRI CT（稀视角）、OOD Lung‑PET‑CT‑Dx（肺癌）等。

**📈 对比分析**

通过 PSNR/SSIM 与像素方差、覆盖率、观察/零空间方差比例等指标，对后验目标、启发式、MAP‑类三类求解器进行对比；后验目标方法在不确定性校准与结构上优于启发式和 MAP，准确性与不确定性并不总相关，部分方法（如 MAP）几乎无方差。

**⚠️ 局限性**

局限性：缺乏对超参数敏感性和计算成本的全面评估，实际实现与理论收敛性仍存在差距，且依赖于预训练扩散先验的质量。

---

## 34. From Expectation To Experience: A Before And After Survey Of Public Opinion On Autonomous Cars In Saudi Arabia

**arXiv ID:** 2602.03854 | [PDF](https://arxiv.org/pdf/2602.03854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 35. Entropy-Aware Structural Alignment for Zero-Shot Handwritten Chinese Character Recognition

**arXiv ID:** 2602.03913 | [PDF](https://arxiv.org/pdf/2602.03913v1)

**作者:** Qiuming Luo `[一作]` (Shenzhen University), Chang Kong `[通讯]` (Institute of Applied Artificial Intelligence of the Guangdong-Hong Kong Macao Greater Bay Area)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一种基于信息熵加权和双视角结构树的零样本手写中文字符识别框架，利用跨模态注意力与 Top‑K 语义融合实现字符结构的深度匹配。

**💡 创新点**

创新点包括：① 信息熵优先的乘法位置编码，显著提升稀有部件的辨别力；② 双视角（父级与子级）结构树提取多粒度语义特征；③ 适配门控融合网络与跨模态注意力的多阶段语义匹配；④ Top‑K 语义特征聚合，提升对视觉模糊的鲁棒性。

**🔧 技术方法**

采用 CLIP 视觉‑文本对齐模型进行部件编码，ResNet‑34 背景特征提取，Transformer 结构树编码，信息熵计算与乘法调制，Sigmoid‑GateFusion，交叉注意力层，以及 Top‑K 语义聚合。

**📊 数据集**

使用 CASIA‑HWDB 1.0/1.1 训练集进行 seen 训练，ICDAR 2013 作为 unseen 的零样本测试集，评估 3,755 常用汉字中 1,000 个未见字符的识别性能。

**📈 对比分析**

在零样本设置下，相比 CCR‑CLIP、SideNet、HierCode 等最新方法，准确率从 62.59% 提升到 67.96%（在 2,755 seen 训练类别时），实现新 SOTA；在全集监督场景下达到 97.30% 的识别率，速度仅 0.74 ms/图像，显著优于多模态大模型。

**⚠️ 局限性**

局限性：仅针对孤立字符，无法直接处理连写文本行；依赖预先构建的部件字典与树解析，若遇非标准字符或书写变体时效果下降；对极少数部件分布不均的字符仍有识别误差。

---

## 36. Multi-threaded Recast-Based A* Pathfinding for Scalable Navigation in Dynamic Game Environments

**arXiv ID:** 2602.04130 | [PDF](https://arxiv.org/pdf/2602.04130v1)

**作者:** Tiroshan Madushanka `[一作]` (University of Kelaniya), Sakuna Madushanka `[通讯]` (University of Kelaniya)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种多线程、基于Recast的A*路径规划框架，结合Bezier平滑和密度分析，实现高性能动态3D环境导航。

**💡 创新点**

创新点在于将Recast网格生成、Bezier曲线轨迹平滑与多线程异步计算结合，并加入全局密度分析实现无碰撞人群协同。

**🔧 技术方法**

使用Recast NavMesh、A*搜索、Funnel算法、Bezier曲线、线程池、密度检测与Unity物理射线检测等技术。

**📊 数据集**

实验数据集为十阶段递增的测试场景，包括二维迷宫、点图、程序化障碍、3D多层地形、动态障碍、移动表面、门控等待、密度扩展等自制场景。

**📈 对比分析**

通过与单线程基线和不同人数的对比实验，平均帧率在1000个NPC时仍保持350 FPS，整体性能提升约4.5倍，路径计算时间低于13ms。

**⚠️ 局限性**

局限在于高速度Agent碰撞避免效果下降，且未针对大型交通仿真或空中三维路径规划做进一步优化。

---

## 37. Stroke Lesions as a Rosetta Stone for Language Model Interpretability

**arXiv ID:** 2602.04074 | [PDF](https://arxiv.org/pdf/2602.04074v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 38. Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical Structure

**arXiv ID:** 2602.03975 | [PDF](https://arxiv.org/pdf/2602.03975v1)

**作者:** Shuhui Qu `[一作]` (Stanford University), Shuhui Qu `[通讯]` (Stanford University)

**通讯引用:** 966 | [OpenAlex ID](https://openalex.org/A5112694715)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大语言模型推理中提出一种基于状态的选择性验证框架，能够在每个中间推理状态上智能分配昂贵的验证器调用，显著降低验证成本并提升正确率。

**💡 创新点**

创新点包括：①利用结构化操作接口实施确定性可行性门控，提前剔除不合法或与约束冲突的候选步骤；②结合结构距离和学习的预验证残差对剩余候选进行分层排序；③根据局部不确定性动态调整每个状态的验证调用数，从而实现非均匀、信息化的验证分配。

**🔧 技术方法**

技术手段包括：结构化生成与可解析操作（Operator Interface）；确定性可行性门控（Parse/Scope/Constraint 检查）；基于LLM编码的结构距离 D_type；残差评分器 r_θ 通过在验证器标注的候选列表上进行对内排序学习；基于分数方差的状态不确定性度量来确定 k(w)；以及 PRM（过程奖励模型）作为验证器。

**📊 数据集**

主要使用 MATH 基准（500 条测试题）以及 GSM8K 进行验证，生成器采用 Llama 3.2 1B（对比 3B 版）与 PRM 验证器。

**📈 对比分析**

与同一生成器/验证器的基线（Best‑of‑N、Majority Voting、Beam Search）在相同的生成预算或验证调用预算下进行比较。实验显示，本文方法在 44.8 次平均验证调用下实现 55.2% 正确率，分别比 Best‑of‑N、Majority Voting 低约 30% 调用且准确率提升；相较于 Beam Search，则在同一调用预算下提升 3.4 分，同时减少 30% 调用。

**⚠️ 局限性**

局限性包括：①性能受生成器质量限制，若生成的候选中不包含正确步骤则无法通过验证恢复；②门控仅能过滤已显式记录的约束，无法处理未建模的语义约束；③在最难的 MATH 难度级别上，验证成本优势受限于模型本身的解题能力。

---

## 39. KGLAMP: Knowledge Graph-guided Language model for Adaptive Multi-robot Planning and Replanning

**arXiv ID:** 2602.04129 | [PDF](https://arxiv.org/pdf/2602.04129v1)

**作者:** Chak Lam Shek `[一作]` (University of Maryland), Piyush Gupta `[通讯]` (Honda Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 KGLAMP 框架，利用知识图引导 LLM 生成 PDDL，完成异构多机器人长周期任务规划。

**💡 创新点**

创新点在于将关系、属性、可达性三类知识图与 LLM 分离处理，并通过在线增量更新和符号回退机制提升规划鲁棒性。

**🔧 技术方法**

技术包括大型语言模型（GPT‑5）、知识图（关系图、属性图、可达性图）、PDDL 规划器、视觉语言模型（VLM）和增量重规划模块。

**📊 数据集**

使用 MAT‑THOR 基准（51 个室内多机器人任务）进行评估。

**📈 对比分析**

与 LLM-as-Planner、LLM+P、LaMMA‑P 等基线对比，KGLAMP 任务完成率提升 25.5%，执行失败率降低 15.5%。

**⚠️ 局限性**

局限性包括对完整环境信息的依赖、LLM 组件高计算开销、知识图可能产生噪声以及 LLM 生成的 PDDL 可能存在语法或语义错误。

---

## 40. Perceptions of AI-CBT: Trust and Barriers in Chinese Postgrads

**arXiv ID:** 2602.03852 | [PDF](https://arxiv.org/pdf/2602.03852v1)

**作者:** Chan-in Sio `[一作]` (Hong Kong Polytechnic University), Lik-hang Lee `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 3526 | [OpenAlex ID](https://openalex.org/A5081811548)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了中国研究生对人工智能认知行为疗法聊天机器人（AI‑CBT）的认知与使用意愿，采用半结构化访谈与主题分析方法。

**💡 创新点**

首次将健康信念模型（HBM）和计划行为理论（TPB）应用于非临床高压学术人群的定性研究，揭示信任、隐私、情感安全与社会规范等多维因素对AI‑CBT接受度的影响。

**🔧 技术方法**

使用AI‑CBT聊天机器人作为介质，采用对话式人工智能技术（生成式聊天机器人）与访谈记录分析工具（如iflyrec转写）。

**📊 数据集**

研究对象为10名中国（大陆、香港、澳门）研究生，访谈数据为10份半结构化访谈文本。

**📈 对比分析**

通过归纳式与理论驱动的主题分析（reflexive thematic analysis），与HBM/TPB框架进行对照，未涉及数值性能指标；主要结果为五大主题及其与模型维度的对应关系。

**⚠️ 局限性**

样本量小、非概率抽样、仅为自述访谈，缺乏定量验证与跨文化对比，研究结果不具统计推广性。

---

## 41. On-Demand Lecture Watching System Using Various Actions of Student Characters to Maintain Concentration

**arXiv ID:** 2602.03853 | [PDF](https://arxiv.org/pdf/2602.03853v1)

**作者:** Saizo Aoyagi `[一作]` (Komazawa University), Michiya Yamamoto `[通讯]` (Kwansei Gakuin University)

**通讯引用:** 534 | [OpenAlex ID](https://openalex.org/A5031992448)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了基于裸眼三维显示的按需观看系统，利用多名学生角色的正面与负面动作（点名、点头、打哈欠、睡觉等）来营造共同在场感并保持学生注意力。

**💡 创新点**

创新点在于：① 将裸眼三维显示与自然动作的学生角色结合，形成可变的沟通场；② 设计了稳定模式与动态模式两种动作切换策略，并系统评估其对笔记、姿态与学习成绩的影响。

**🔧 技术方法**

使用技术包括：Unity 2021.3.24f1 构建三维环境、Sony ELF‑SR1 裸眼三维显示、Azure Kinect DK 采集动作数据、OpenPose 计算姿态角度、CSV 文件调度动作序列。

**📊 数据集**

实验数据集：33 分钟的预录制课程《Computer Graphics Extra》、29 幻灯片、19 道多项选择测验题、50 名本科生参与、手写笔记文件。

**📈 对比分析**

对比方法：将参与者随机分为稳定模式与动态模式两组，采用问卷（7点量表）、笔记数量统计、姿态角度测量与测验成绩进行比较。结果显示：两模式测验成绩无显著差异；稳定模式显著提升关键段落笔记数量，且在难题上得分更高；动态模式降低了后仰倾斜倾向。

**⚠️ 局限性**

局限性：样本量有限且仅包含 33 分钟课程，未验证同步在线教学效果；仅在裸眼三维显示下测试，未探究 VR 扩展的沉浸度；角色动画的自然度需在更大尺度下进一步验证；实验仅针对计算机图形课，结果可能不适用于其他学科。

---

## 42. SEIS: Subspace-based Equivariance and Invariance Scores for Neural Representations

**arXiv ID:** 2602.04054 | [PDF](https://arxiv.org/pdf/2602.04054v1)

**作者:** Huahua Lin `[一作]` (University of Southampton), Xiaohao Cai `[通讯]` (University of Southampton)

**通讯引用:** 1319 | [OpenAlex ID](https://openalex.org/A5078326698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为SEIS的子空间度量，用于在不需要标签或预定义变换的前提下，层级分析神经网络在几何变换下的等变性（equivariance）与不变性（invariance），并通过合成与真实数据验证其有效性；

**💡 创新点**

创新点在于首次通过空间感知的矩化（spatially‑aware matricization）与SVD+CCA子空间分析，将等变性与不变性分离开来，提供对内部表征几何稳定性的细粒度诊断；

**🔧 技术方法**

使用的技术包括：子空间降噪（SVD）、经典相关分析（CCA）、空间感知矩化、合成几何变换、数据增强、深度残差网络训练、多任务学习以及FCN/UNet解码器对比；

**📊 数据集**

实验数据集主要包括MNIST（合成验证）、CIFAR‑100（分类实验）、PASCAL VOC 2012（语义分割与多任务学习），并对输入应用随机仿射变换；

**📈 对比分析**

方法通过对比identity、随机基准以及不同层的SEIS得分，展示了等变性在浅层高、深层低、而不变性在深层高的趋势；数据增强提升深层不变性并保持等变性；多任务学习在共享编码器中同时提升等变性与不变性，解码器中的skip连接可恢复等变性；

**⚠️ 局限性**

局限性包括：仅对线性或近似线性的几何变换敏感；需大量样本与高维度以保证SVD/CCA稳定；计算成本较高；未直接评估对下游任务性能的影响，且仅衡量子空间相似性而非完整功能。

---

## 43. Intellectual Property Protection for 3D Gaussian Splatting Assets: A Survey

**arXiv ID:** 2602.03878 | [PDF](https://arxiv.org/pdf/2602.03878v1)

**作者:** Longjie Zhao `[一作]` (Sydney AI Centre, University of Sydney), Tongliang Liu `[通讯]` (Sydney AI Centre, University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对3D Gaussian Splatting（3DGS）数字资产的知识产权（IP）保护进行系统综述，提出了从 Gaussian 扰动机制到保护范式再到 AIGC 时代鲁棒性威胁的底层框架，并对现有 24 种方法进行分类、技术分析与未来研究方向规划。

**💡 创新点**

①首次以三层底层框架（扰动机制、保护任务、鲁棒性威胁）系统化梳理 3DGS IP 保护；②对 Gaussian 扰动的属性选择、分布策略和注入管道进行统一技术分析；③结合 AIGC 时代的生成式修复与编辑威胁，识别现有研究的漏洞并提出六大未来方向。

**🔧 技术方法**

通过文献综述、技术分类与框架构建，聚焦 Gaussian 基础扰动、被动保护（水印、隐写、篡改定位）与主动保护（编辑防护），并讨论鲁棒性评估（传统 2D/3D 损伤、生成式净化、生成式编辑）等技术要点。

**📊 数据集**

本论文为综述性工作，未自行开展实验；引用的 24 种方法使用的典型 3DGS 数据集包括 Blender、KITTI、ScanNet、RealEstate‑10k 等公开场景集合。

**📈 对比分析**

不进行直接方法对比或性能评测；论文通过对比分析各方法的技术维度、攻击适用性与鲁棒性特点，指出大多数方法在生成式净化、跨表示迁移及标准化评测方面表现不足。

**⚠️ 局限性**

①缺乏统一的评测基准与统一数据集，导致方法间对比不公平；②对生成式净化和编辑攻击的鲁棒性验证不足；③未覆盖跨表示（网格、体素、NeRF 等）转换后的信息持久性；④主要聚焦被动保护，主动保护和生命周期治理研究仍有限。

---

## 44. Modular Safety Guardrails Are Necessary for Foundation-Model-Enabled Robots in the Real World

**arXiv ID:** 2602.04056 | [PDF](https://arxiv.org/pdf/2602.04056v1)

**作者:** Joonkyung Kim `[一作]` (Texas A&M University), Yan Gu `[通讯]` (Purdue University)

**通讯引用:** 7094 | [OpenAlex ID](https://openalex.org/A5043897310)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种可模块化的安全护栏架构（监测与评估层 + 决策门与动作门），用于在现实世界中保障基于基础模型的机器人在物理、决策和人类中心安全三维度下的可靠执行。

**💡 创新点**

创新点在于：①引入外部与内部模块化双重设计，形成独立且不可绕过的安全管控；②构建跨层与跨模组的协同设计原则（表示对齐、保守度分配），使安全干预更精准、低保守；③提供完整的安全三维分类与可扩展的护栏模块化接口。

**🔧 技术方法**

核心技术包括：基础模型（LLM、VLM、VLA）在感知、规划与控制的融合；安全监测模块（感知验证、计划评估、物理约束检测）；决策门与动作门实现的安全过滤与约束投影；以及跨层表示对齐与保守度分配的协同算法。

**📊 数据集**

未给出专门实验数据集，而是基于通用机器人感知与控制数据（如常见视觉语言、轨迹规划、物理约束数据）进行安全评估，并在案例示例中引用典型工业与实验室场景的模拟/实测数据。

**📈 对比分析**

与传统单一安全机制（内部对齐、单层监控或单一安全滤波）对比，本文提出的模块化护栏在覆盖物理、决策与人类安全三维度时更完整；实验（示例）表明可显著降低不安全决策与碰撞风险，且通过保守度分配实现更高的任务通过率，虽然未给出统一指标但案例展示了明显的性能提升。

**⚠️ 局限性**

局限性包括：需要保证监测与干预模块与基础模型的操作独立性，若共享训练数据或模型，可能导致失效；运行时需要额外计算开销，特别是决策门涉及大规模基础模型；缺乏统一量化评测指标与大规模实测验证；以及对协同设计与接口标准化的进一步研究需求。

---

## 45. Have Large Language Models Enhanced the Way Civil & Environmental Engineers Write? A Quantitative Analysis of Scholarly Communication over 25 Years

**arXiv ID:** 2602.03864 | [PDF](https://arxiv.org/pdf/2602.03864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 46. Natural Language Instructions for Scene-Responsive Human-in-the-Loop Motion Planning in Autonomous Driving using Vision-Language-Action Models

**arXiv ID:** 2602.04184 | [PDF](https://arxiv.org/pdf/2602.04184v1)

**作者:** Angel Martinez-Sanchez `[一作]` (University of California), Ross Greer `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究将乘客自然语言指令与视觉-语言-动作模型结合，在真实驾驶数据上实现了指令条件的轨迹规划。

**💡 创新点**

首次在真实数据集 doScenes 上将自由式指令注入 OpenEMMA，实现了从指令到连续轨迹的端到端桥接，并对指令语义与轨迹误差进行系统分析。

**🔧 技术方法**

使用基于 LLaVA-1.6-Mistral-7B 的多模态大型语言模型，构建 OpenEMMA 的视觉–语言接口并通过提示注入指令。

**📊 数据集**

采用 nuScenes 与其扩展的 doScenes 数据集，共计 849 个标注场景。

**📈 对比分析**

与无指令基线对比，使用平均位移误差 ADE 评估，发现指令条件模型在去除异常样本后平均提升 5.1%，总体提升 98.7%，并展示了指令长度和指向性对 ADE 的影响。

**⚠️ 局限性**

局限包括指令与实际语音存在分布差距、模型总是执行指令导致安全性不确定、仅用 ADE 评估、可执行指令场景有限、仅使用单一 VLM 与架构。

---

## 47. When AI Persuades: Adversarial Explanation Attacks on Human Trust in AI-Assisted Decision Making

**arXiv ID:** 2602.04003 | [PDF](https://arxiv.org/pdf/2602.04003v1)

**作者:** Shutong Fan `[一作]` (Clemson University), Xiaoyong Yuan `[通讯]` (Clemson University)

**通讯引用:** 3912 | [OpenAlex ID](https://openalex.org/A5010643450)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并系统评估了通过改变LLM生成的解释框架来误导用户对AI输出的信任的攻击方式，称为对抗性解释攻击（AEA）。

**💡 创新点**

创新点在于：①把解释视为攻击面，定义了信任失衡（trust miscalibration gap）指标；②构造了四维（推理模式、证据类型、沟通风格、呈现格式）的可组合解释空间；③通过大规模受试者实验量化不同解释策略、任务难度、领域和用户属性对误导效果的影响。

**🔧 技术方法**

技术手段包括：利用大型语言模型（如Llama‑3.3‑70B）生成多种风格解释；使用小模型集成进行解释策略验证与可信度评估；在Qualtrics上部署控制实验，采用OLS、混合效应模型等统计分析方法。

**📊 数据集**

数据集为MMLU基准（七大领域、三难度级别）的文本问题；受试者样本为205名来自美国大学和AMT的英语母语成年人。

**📈 对比分析**

比较方法为对照实验（benign vs adversarial）并计算信任误差；结果显示攻击解释与正常解释的信任得分差异不显著（平均≈4.5/4.6），信任失衡平均值在最具说服力的组合上可达≈0.9；更复杂任务、事实驱动领域以及低学历、年轻、对AI高度信任的用户对攻击更敏感。

**⚠️ 局限性**

局限性包括：实验仅在文本问答、无时间压力的web界面下进行，缺乏多模态交互和高风险专业情境；受试者为非专业人员，难以直接推广至医疗、法律等专家决策；未实现并验证具体防御措施，只提供设计思路。

---

## 48. Non-linear PCA via Evolution Strategies: a Novel Objective Function

**arXiv ID:** 2602.03967 | [PDF](https://arxiv.org/pdf/2602.03967v1)

**作者:** Thomas Uriot `[一作]`, Elise Chung `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种利用神经网络参数化变量变换并通过进化策略优化的非线性主成分分析框架；

**💡 创新点**

核心创新是引入分段（partial）方差最大化目标，单独提升每个变量对前k主成分的贡献；

**🔧 技术方法**

使用前馈神经网络（针对数值、分类、序数变量）+ Evolution Strategies（ES）优化；

**📊 数据集**

在八个公开数据集（如credit-g、wine、heart-statlog、breast‑cancer等）及三个合成数据（嵌套圆、球、交替条纹）上评估；

**📈 对比分析**

与线性PCA、核PCA比较，采用解释方差作为指标；在绝大多数数据集上，ES‑Partial显著优于ES‑Global，且常优于核PCA；

**⚠️ 局限性**

缺点包括计算成本高（每代需要SVD）、对ES参数敏感、目前仅关注解释方差未考虑重建误差或下游任务性能；

---

## 49. Benchmarking Bias Mitigation Toward Fairness Without Harm from Vision to LVLMs

**arXiv ID:** 2602.03895 | [PDF](https://arxiv.org/pdf/2602.03895v1)

**作者:** Xuwei Tan `[一作]` (Ohio State University), Xueru Zhang `[通讯]` (Ohio State University)

**通讯引用:** 4133 | [OpenAlex ID](https://openalex.org/A5101877243)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 NH‑Fair 基准，统一评估传统视觉模型和大型视觉‑语言模型在公平性与无害性方面的表现，并系统比较了多种公平性干预方法。

**💡 创新点**

创新点在于：①通过大规模超参数搜索揭示 ERM 训练选择对公平性的显著影响；②证明数据增强（RandAugment、Mixup 等）能在保持或提升准确率的同时显著降低群体差距；③发现即便是大规模 LVLM 也并未天然解决公平问题，且模型规模增长对公平性的提升有限。

**🔧 技术方法**

采用的技术包括：多种数据增强与重采样策略、对抗与正则化公平性训练、基于预训练权重的模型微调、零射击推理（CLIP、BLIP‑2、LLaVA、Qwen‑VL 等）以及基于 DTO 的模型选择和公平性无害性评估框架。

**📊 数据集**

实验使用了七个公开数据集：CelebA、UTKFace、FairFace、Facet、HAM10000、Fitz17k 与 Waterbirds，涵盖面部属性、医学影像与背景相关的伪相关性等多样化场景。

**📈 对比分析**

结果显示：经过充分调参的 ERM 基线往往能与专门的公平性方法相媲美甚至超越；数据增强方法在大多数据集上实现了准确率与公平度的双赢；LVLM 在某些任务上取得高准确率，但仍存在显著的子群体差距，规模扩展带来的公平提升有限。

**⚠️ 局限性**

局限性包括：评估主要聚焦于群体公平度，未涵盖个体公平或因果公平；对某些数据集（如 Waterbirds）过度依赖可能低估真实世界公平挑战；以及实验对 GPU 资源的高需求，限制了更大规模模型与更多方法的探索。

---

## 50. Agentic AI-Empowered Dynamic Survey Framework

**arXiv ID:** 2602.04071 | [PDF](https://arxiv.org/pdf/2602.04071v1)

**作者:** Furkan Mumcu `[一作]`, Yasin Yilmaz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于代理AI的动态问卷框架，持续更新已发表的综述论文，保持结构与写作风格不变。

**💡 创新点**

创新点在于将综述视为“活文档”，通过可持续的局部更新、路由与弃置机制，避免全局重写导致的语义漂移。

**🔧 技术方法**

采用多代理架构（分析、路由、文本/表格合成、弃置）配合大型语言模型（Qwen、Gemma、Gemini、GPT‑OSS）实现分步推理与工具调用。

**📊 数据集**

使用五篇不同主题的综述（目标检测、对抗攻击、遥感图像超分、视频异常检测、机器人臂仿真）做回溯式维护基准，并随机抽取同一会议外的论文做不相关测试。

**📈 对比分析**

与单次生成和oracle路由的基线对比，实验显示在BLEU/ROUGE/语义相似度上提升约4点，局部编辑量大幅下降（平均225个标记，零跨区编辑），人类评估分数与人工写作相近（≈4.0）。

**⚠️ 局限性**

局限性包括：需要作者提供并固定的结构大纲，框架对非结构化综述适用性有限；依赖大型语言模型的推理质量，可能受模型偏见与生成误差影响；实际部署仍需人工审核与监督。

---

## 51. FDA Flocking: Future Direction-Aware Flocking via Velocity Prediction

**arXiv ID:** 2602.04012 | [PDF](https://arxiv.org/pdf/2602.04012v1)

**作者:** Hossein B. Jond `[一作]` (Czech Technical University), Martin Saska `[通讯]` (Czech Technical University)

**通讯引用:** 5082 | [OpenAlex ID](https://openalex.org/A5004992661)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并验证了一种基于短期速度预测的前瞻性聚群模型——FDA flocking，可在多旋翼无人机群体中实现更快、更稳的聚集并对通信延迟与噪声表现出更强鲁棒性。

**💡 创新点**

将生物学中预判性线索（如鸟类姿态与无人机俯仰滚转）转化为可测量的速度预测，并将预测项与传统对齐项按可调权重融合，从而提升群体协同与对时延、噪声的抵抗力。

**🔧 技术方法**

采用基于Reynolds-Boids核心规则与Cucker-Smale一致性理论的融合控制律，利用预测公式$v_{pred}=v+t_{ph}u$并对控制量做tanh饱和，同时辅以拉普拉斯矩阵理论分析与数值仿真。

**📊 数据集**

使用人工生成的10个三维多旋翼机器人模拟数据（随机初始位置/速度），并在仿真中注入时间延迟与高斯噪声，未使用公开真实实验数据集。

**📈 对比分析**

通过对比纯反应型模型与FDA模型，在相同仿真条件下测量方向一致性、群体几何距离和中心轨迹长度；FDA在对齐速度、峰值一致性及群体位移上提升约30-40%，并在延迟+噪声场景下保持高一致性，而反应模型明显退化。

**⚠️ 局限性**

目前采用均匀加权的预测平均，未考虑邻居预测质量差异或动态权重；预测时域过大可能导致不稳定；缺乏实际多旋翼无人机实验验证。

---

## 52. From Crafting Text to Crafting Thought: Grounding AI Writing Support to Writing Center Pedagogy

**arXiv ID:** 2602.04047 | [PDF](https://arxiv.org/pdf/2602.04047v1)

**作者:** Yijun Liu `[一作]` (University of Illinois Urbana-Champaign), Tal August `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 460 | [OpenAlex ID](https://openalex.org/A5029563909)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于写作中心的教学理念，设计并实现了AI写作辅助工具Writor，提供非直接式反馈、目标设定、对话式协作，旨在促进学生写作过程与批判性思维；

**💡 创新点**

创新点在于将写作中心的三大原则（以写作者为中心、过程导向、协作式）转化为七条可操作的设计准则，并在系统中实现非指令式的反馈与互动对话，打破传统AI写作工具直接生成文本的做法；

**🔧 技术方法**

使用技术包括OpenAI GPT‑4.1‑mini作为后端生成反馈、Flask+JavaScript前端实现交互界面、Firebase Firestore存储会话记录；设计准则通过精细化的prompt工程实现多种反馈类型（举例、类比、读者视角、问答）和情感平衡；

**📊 数据集**

数据集主要有：①10名写作中心导师的访谈记录；②30名写作教师、导师、AI研究员的专家评审数据；③5篇《Michigan Corpus of Upper‑Level Student Papers》论证性文章和5篇求职信样本，用于系统演示和评估；

**📈 对比分析**

比较方法为专家评价（Likert量表、开放性问卷）并与传统写作中心反馈、教师反馈、同行评审及通用生成式AI工具（ChatGPT、Claude等）进行感知对比；性能上，整体帮助度平均为3.0/5，平衡度最高（3.84/5），准确度（3.42/5）和清晰度（3.35/5）也表现较好；在对比中，Writor在非指令式、保持写作者声音方面被评为优于常规AI工具，但仍落后于人类导师和写作中心的反馈；

**⚠️ 局限性**

局限性包括：仅基于专家视角评估，缺乏学生使用者实验；反馈细粒度可调性不足，容易出现过于模糊或过于指令式；系统完全依赖prompt策略，未进行功能拆解或细粒度AB实验；仅采用写作中心为单一教学框架，未考虑其他文化或学科特定的写作理论。

---

## 53. Predicting Depressive Symptoms through Emotion Pairs within Asian American Families

**arXiv ID:** 2602.03943 | [PDF](https://arxiv.org/pdf/2602.03943v1)

**作者:** Sangpil Youm `[一作]` (University of Florida), Sou Hyun Jang `[通讯]` (Korea University)

**通讯引用:** 478 | [OpenAlex ID](https://openalex.org/A5002073039)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过情感识别模型对 31,144 条 r/AsianParentStories 的 Reddit 帖子进行句子级情感检测，构建情感共现网络，并用情感对与抑郁症状的逻辑回归关联，发现十个情感对对抑郁症状具有显著影响，揭示跨情感共存对心理健康的复杂作用。

**💡 创新点**

①首次系统性将情感对（而非单一情感）与抑郁症状相关联；②将情感网络可视化与统计学习相结合，展示情感共现的多维互动；③通过自动化情感与抑郁检测模型在大规模 Reddit 数据上的应用，为跨文化心理学研究提供可复制的方法。

**🔧 技术方法**

情感检测：EmoRoBERTa；抑郁症状识别：DepRoBERTa；情感共现网络构建与可视化；情感对与抑郁关系分析：逻辑回归；后续内容分析解释。

**📊 数据集**

r/AsianParentStories 子版块 2012‑2022 年间的 31,144 条帖子，涵盖 28 种可检测情感。

**📈 对比分析**

使用逻辑回归对情感对进行显著性检验，统计显著性阈值 0.05%；结果以优势比（Odds Ratio）呈现，未与传统单情感模型进行直接性能比较，但通过显著情感对数量和优势比说明模型能捕捉更细粒度的情感-抑郁关系。

**⚠️ 局限性**

仅限于单一 Reddit 社区，样本自选性高；情感标签依赖预训练模型，可能存在跨文化情感表达差异；未提供真实抑郁诊断标签，仅使用文本模型预测抑郁；结果无法直接推广至其他族裔或平台。

---

## 54. An Anatomy-specific Guidewire Shaping Robot for Improved Vascular Navigation

**arXiv ID:** 2602.04050 | [PDF](https://arxiv.org/pdf/2602.04050v1)

**作者:** Aabha Tamhankar `[一作]` (Worcester Polytechnic Institute), Giovanni Pittiglio `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 892 | [OpenAlex ID](https://openalex.org/A5049921916)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一台自动化导丝成形机器人，能够根据预设的形状参数对标准导丝进行精确成形。

**💡 创新点**

通过数据驱动的本体学模型将机器人动作映射到导丝几何，并使用单一实验校准获得每段弯曲角度，实现了高可重复性且可标准化的成形。

**🔧 技术方法**

采用步进电机、伺服控制、精密线性导轨、手术级微型抓手以及基于SO(3)的本体学模型和实验校准方法。

**📊 数据集**

使用实验室对 ARROW™ Marked Spring 0.64 mm 直径导丝的成形实验数据，进行校准与验证；未使用公开医学影像数据集。

**📈 对比分析**

与模型预测的导丝中心线对比，四种临床常见二维形状（C、S、Angled、Hook）平均误差0.56 mm；成功演示3D螺旋形导丝在人工 3D 打印 Circle of Willis 模型中的导航效果，表现出对复杂血管分叉的偏向性。

**⚠️ 局限性**

系统为开环控制，误差随段数累积；仅在单一导丝型号和体外实验中验证；缺乏闭环反馈、实时校准和体内临床验证。

---

## 55. GeoLanG: Geometry-Aware Language-Guided Grasping with Unified RGB-D Multimodal Learning

**arXiv ID:** 2602.04231 | [PDF](https://arxiv.org/pdf/2602.04231v1)

**作者:** Rui Tang `[一作]`, Hongliang Ren `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了端到端的视觉‑语言抓取框架GeoLanG，能在杂乱环境下根据自然语言指令完成目标分割与抓取。

**💡 创新点**

创新点包括：1）Depth‑Guided Geometric Module（DGGM）将深度几何先验直接注入注意力，提升遮挡与低纹理场景下的空间推理；2）Adaptive Dense Channel Integration（ADCI）自适应聚合多层视觉特征，增强跨模态表达；3）使用CLIP‑VMamba混合视觉编码器与CLIP‑BERT文本编码器，统一语义空间。

**🔧 技术方法**

技术实现：CLIP‑VMamba视觉背骨、CLIP‑BERT文本编码器；DGGM将深度差异与曼哈顿距离构成几何先验并融入多头注意力；ADCI采用分组加权聚合与轻量门控网络。

**📊 数据集**

在OCID‑VLG基准（1,763张RGB‑D桌面场景、89,639语言‑分割‑抓取三元组）上训练与评估，并在RoboDK仿真与DOBOT Nova 2真实机器人上进行桌面清理实验。

**📈 对比分析**

与CROG、CTNet、GraspCLIP、CLIPort等SOTA基线对比，GeoLanG在分割IoU、Pr@阈值及抓取Jacquard指标均取得最高分（IoU 85.77%、Pr@70 89.82%、J@1 87.32%、J@N 92.13%），在未见实例上亦保持显著优势。

**⚠️ 局限性**

局限性：仅实现4‑DoF平面抓取，未处理完整6‑DoF抓取与动态环境；实验多在相对受控桌面场景，缺乏更广泛的真实世界多样性验证。

---

## 56. Decoding Ambiguous Emotions with Test-Time Scaling in Audio-Language Models

**arXiv ID:** 2602.03873 | [PDF](https://arxiv.org/pdf/2602.03873v1)

**作者:** Hong Jia `[一作]` (University of Melbourne), Ting Dang `[通讯]` (University of Melbourne)

**通讯引用:** 639 | [OpenAlex ID](https://openalex.org/A5071116593)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型音频语言模型进行基准测试，评估其在模糊情绪识别中的性能，并研究测试时缩放（TTS）策略对情绪辨识的影响。

**💡 创新点**

首次提出针对模糊情绪识别的TTS基准，系统比较8种ALM与5种TTS方法，并揭示加权聚合策略（W‑BoN、W‑ALM‑v）在不同模糊度下的显著优势。

**🔧 技术方法**

采用音频语言模型推理、Beam Search、多候选聚合、CoT推理、ALM验证器以及Dirichlet混合模型等技术。

**📊 数据集**

三大公开情绪语料库——IEMOCAP、MSP‑Podcast、CREMA‑D。

**📈 对比分析**

通过Jensen‑Shannon散度、Bhattacharyya系数、R²以及准确率/F1进行评估，结果显示闭源模型总体领先；在开放源模型中Qwen2‑Audio与W‑BoN组合在多项指标上均优于基线，TTS显著提升模糊情绪辨识性能。

**⚠️ 局限性**

评估受限于现有数据集的多样性和标注方式，缺乏跨文化、多语言及连续情绪表征；且使用的通用TTS方法未针对情绪专门优化，未来需要更细粒度的情绪导向TTS与更丰富的数据。

---

## 57. Chaplains' Reflections on the Design and Usage of AI for Conversational Care

**arXiv ID:** 2602.04017 | [PDF](https://arxiv.org/pdf/2602.04017v1)

**作者:** Joel Wester `[一作]` (University of Copenhagen), Niels van Berkel `[通讯]` (Aalborg University)

**通讯引用:** 3299 | [OpenAlex ID](https://openalex.org/A5003896144)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

让北欧大学牧师使用GPT Builder创建并评估聊天机器人，收集他们的反思并通过主题分析得到四个主要主题：倾听、连接、携带与渴望。

**💡 创新点**

首次从牧师专业视角探讨非临床情境下的对话式AI缺陷，并提出“调谐(attunement)”框架，用以指导更具情感共振的聊天机器人设计。

**🔧 技术方法**

使用OpenAI的GPT Builder（自定义ChatGPTs）进行聊天机器人构建与交互，并利用访谈录音转写与手工编码进行定性分析。

**📊 数据集**

无公开数据集；研究依赖18名牧师的访谈文本、他们设计的提示词与测试问句。

**📈 对比分析**

研究未进行定量比较或性能评估，而是采用主题分析方法，从质性数据中提炼四个核心主题，并未给出具体指标。

**⚠️ 局限性**

局限性包括：样本仅来自北欧，牧师对AI使用经验有限；未分析提示词与交互细节；研究仅基于专家意见，缺乏终端用户视角，且未探讨模型生成内容的客观质量。

---

## 58. Autonomous AI Agents for Real-Time Affordable Housing Site Selection: Multi-Objective Reinforcement Learning Under Regulatory Constraints

**arXiv ID:** 2602.03940 | [PDF](https://arxiv.org/pdf/2602.03940v1)

**作者:** Olaf Yunus Laitinen Imanov `[一作]` (Technical University of Denmark), Rana Irem Turhan `[通讯]` (Riga Technical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 AURA——一种基于多智能体强化学习的实时可负担住房选址系统，能在复杂法规约束下自动生成可行且多目标平衡的选址方案。

**💡 创新点**

创新点包括：① 将选址问题建模为受约束多目标马尔可夫决策过程；② 设计监管感知状态表示与基于 Pareto 约束的 PPO 算法；③ 引入多保真度奖励分解，区分即时成本与长期社会效益；④ 通过分层多智能体架构实现地理分析、法规合规与多目标优化的协同工作。

**🔧 技术方法**

使用技术：多智能体强化学习（PC‑PPO）、图神经网络进行地理特征编码、约束满足推理网络、注意力机制协调多智能体、优势估计与偏好向量的多目标策略梯度、以及多保真度奖励预测网络。

**📊 数据集**

数据集：8个美国大都市（纽约、洛杉矶、芝加哥、休斯顿、凤凰城、费城、圣安东尼奥、圣地亚哥）共 47,392 块候选地块，涵盖 GIS、法规（QCT/DDA/LIHTC 等 127 条）、交通（Walk Score、GTFS）、环境（洪水、空气质量）、社会经济（ACS 5 年估计）等多维信息。

**📈 对比分析**

与 6 种基线（人类专家、随机可行、贪心单目标、NSGA‑II、MOEA/D、单策略 MORL）对比，AURA 在 Pareto 超体积提升 37.2%、法规合规率 94.3%、选址时间从 18 个月降至 72 小时、交通可达性提升 31%、环境影响降低 19% 等指标上显著优于对照组。

**⚠️ 局限性**

局限性：仍有约 5.7% 的方案不完全合规（多因法规模糊或更新滞后）；对法规更新手动同步；长期社会和环境效益的预测仍依赖代理化指标；未覆盖多城市协同、施工排程和灾害韧性；训练成本高，单城训练约 84 小时；对非硬约束（如“合理”停车）处理仍需人工复核。

---

## 59. VideoBrain: Learning Adaptive Frame Sampling for Long Video Understanding

**arXiv ID:** 2602.04094 | [PDF](https://arxiv.org/pdf/2602.04094v1)

**作者:** Junbo Zou `[一作]` (Georgia Institute of Technology), Weining Shen `[通讯]` (University of California)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VideoBrain，一种端到端的视觉语言模型框架，通过自适应帧采样策略实现长视频理解；

**💡 创新点**

创新点包括：①双重采样代理（CLIP语义检索 + 匀速采样）实现多维信息获取；②行为感知奖励函数消除奖励欺骗，指导代理调用；③直接让 VLM 观察帧并做决策，避免传统文本驱动代理的瓶颈；

**🔧 技术方法**

技术手段涵盖：CLIP 语义检索、均匀采样代理、链式思考与结构化工具调用、监督微调+强化学习（GRPO）、行为感知奖励设计、格式验证与奖励分层；

**📊 数据集**

数据集与评测：训练使用 Video-Holmes、CG-Bench、NExT-QA、MLVU、LongVideo-Reason；评测基准包括 LongVideoBench、LVBench、Video-MME Long、MLVU Test；短视频泛化在 DREAM-1K 与 Video-MME Short；

**📈 对比分析**

与 Qwen3-VL-8B-Instruct 基线及 GPT‑4o、Gemini‑1.5‑Pro、ShareGPT4Video、LongVA、VideoChat‑R1、Video‑R1、Qwen2.5‑VL‑Instruct、FrameThinker 等对比，VideoBrain 在四大长视频基准上平均提升 3.5%–9.0%，同时平均减少 30%–40% 帧量；在短视频基准上提升 5.1%–2.4%，并同样减少 46%–47% 帧；

**⚠️ 局限性**

局限性包括：仍需 8B 级模型；对极长视频的实时性与计算开销未完全解决；奖励设计与数据分类依赖人工标签，可能对新任务迁移产生偏差；

---

## 60. Understanding and Guiding Layer Placement in Parameter-Efficient Fine-Tuning of Large Language Models

**arXiv ID:** 2602.04019 | [PDF](https://arxiv.org/pdf/2602.04019v1)

**作者:** Yichen Xu `[一作]` (University of California), Chenhao Ma `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1045 | [OpenAlex ID](https://openalex.org/A5055857919)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于投影残差的参数高效微调（PEFT）层级选择框架，并引入Layer Card诊断工具

**💡 创新点**

在PEFT中首次系统分析投影残差、激活能量与层间耦合三因素对层级选择的影响，证明弱耦合时层级可近似独立且分布式放置更优

**🔧 技术方法**

使用局部二次近似、协方差归一化梯度、LoRA适配器、Layer Card可视化与层级筛选算法

**📊 数据集**

在GPT‑2、LLaMA2‑7B、Qwen3‑8B等大型模型上，使用DART、E2E、WebNLG、HS、MathQA、GSM8K、SVAMP等多任务数据集进行实验

**📈 对比分析**

通过与全层LoRA、随机/均匀/上下层放置的对比，发现中层放置可提升111%性能或在相同精度下显著降低训练时间（30–75%）和显存（2.3×），在Qwen3‑8B上5层放置即可逼近全层效果，速度提升达74%

**⚠️ 局限性**

对层级可转移性的任务相似性不足的情况仍需改进；目前只验证了单任务/单模型场景，未讨论连续学习或多任务共存时的兼容性

---

## 61. JSynFlow: Japanese Synthesised Flowchart Visual Question Answering Dataset built with Large Language Models

**arXiv ID:** 2602.04142 | [PDF](https://arxiv.org/pdf/2602.04142v1)

**作者:** Hiroshi Sasaki `[一作]` `[通讯]` (Japan Research Institute Limited), Hiroshi Sasaki (Japan Research Institute Limited)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套基于大语言模型生成的日语流程图视觉问答数据集JSynFlow，并用它进行VLM微调

**💡 创新点**

创新点在于利用LLM与DSL代码自动化生成流程图图像及对应QA对，省去人工标注；以及将日语流程图作为视觉问答任务

**🔧 技术方法**

使用LLM（Llama‑3.1‑405B‑Instruct）、Mermaid DSL、mermaid‑cli生成图像，以及LoRA微调技术

**📊 数据集**

使用自己构建的JSynFlow数据集（1359+152任务，共11311 QA）以及公开的VLM模型LLaVA‑JP和Qwen2‑VL‑2B‑Instruct

**📈 对比分析**

通过BERTScore评估QA准确率，微调后LLaVA‑JP从F1=0.6605提升到0.7691，Qwen2‑VL从F1=0.8597提升到0.9397

**⚠️ 局限性**

数据质量依赖LLM生成，需人工校正；图像多样性有限，来源于标准渲染器

---

## 62. DADP: Domain Adaptive Diffusion Policy

**arXiv ID:** 2602.04037 | [PDF](https://arxiv.org/pdf/2602.04037v1)

**作者:** Pengcheng Wang `[一作]` (University of California), Yixiao Wang `[通讯]` (University of California)

**通讯引用:** 935 | [OpenAlex ID](https://openalex.org/A5100651550)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种通过扩散模型实现的域自适应策略 DADP，利用无监督方法学习并解耦域表示，再将其注入扩散过程以提升零样本域迁移性能。

**💡 创新点**

创新点包括：
1) Lagged Context Dynamical Prediction 通过在时间上拉远上下文窗口，去除时间变化信息，仅保留域特定的静态信息；
2) 在扩散生成过程中将域表示直接作为 prior 的混合高斯中心，并把表示偏移与噪声一起预测，既注入域信息又提供额外监督。

**🔧 技术方法**

主要技术手段：
- 无监督的动态预测与时间偏移学习；
- 扩散政策（Diffusion Policy）与 DDIM 结构；
- 通过混合高斯 prior 与联合预测目标实现域感知采样。

**📊 数据集**

使用了 MuJoCo 的四个机器人运动任务（HalfCheetah、Walker2d、Ant、Hopper）以及 Adroit 的 manipulation 任务（Door、Relocate），在每个任务中采样不同环境参数（重力、摩擦、关节长度等）生成 25~30 个域的数据集。

**📈 对比分析**

与 SOTA 对比：在 IID 与 OOD 两种设置下，DADP 在绝大多数任务上均超过或匹配 CORRO、Prompt‑DT、Meta‑DT，零样本性能提升显著；同时标准差更小，表现更稳定。实验还通过 ablation 证明 Δt 与表示利用策略对性能的关键作用。

**⚠️ 局限性**

局限性：
- 仅处理时间不变（stationary）的动力学，无法应对非平稳或动态变化的环境；
- 需要大量离线数据与对齐的上下文；
- 对超参数（如 Δt、混合高斯分布数）敏感，需要经验调优。

---

## 63. Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity

**arXiv ID:** 2602.04162 | [PDF](https://arxiv.org/pdf/2602.04162v1)

**作者:** Chenhe Du `[一作]` (ShanghaiTech University), Yuyao Zhang `[通讯]` (ShanghaiTech University)

**通讯引用:** 2406 | [OpenAlex ID](https://openalex.org/A5100654429)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了基于Spherical Linear Interpolation的层间一致随机性（ISCS）方法，解决 2D 扩散模型在 3D 医学重建时产生的层间不连续问题。

**💡 创新点**

创新点在于用高维高斯球面插值生成相互关联的噪声，使采样轨迹在相邻切片间保持平滑一致，从而无需额外损失或超参数即可提升 3D 质量。

**🔧 技术方法**

核心技术包括扩散模型（VE/VE‑DDIM）、基于噪声插值的重采样、以及传统的逆问题求解框架（DDNM/DDS）与数据一致性更新。

**📊 数据集**

实验数据集为 AAPM 2016 低剂量 CT 数据（5936 切片）和 IXI T1‑MRI 数据（256×256×256 体积，降采样为 5×），用于 SVCT、LACT 与 MRI 超分辨率任务。

**📈 对比分析**

与 DDNM、DDS、TV 正则化、FDK、ADMM‑TV 等方法比较，ISCS 在 PSNR、SSIM 与 LPIPS 上均有 0.3–0.8 dB 的提升，并显著降低层间切片断裂和伪影。

**⚠️ 局限性**

局限性在于仍依赖 2D 先验，不能充分利用 3D 结构；噪声插值是手工设定的，未学习到数据自适应的关联模式；且在极端体积尺寸或大解剖变化时可能出现复制伪影。

---

## 64. Less is More: Optimizing Probe Selection Using Shared Latency Anomalies

**arXiv ID:** 2602.03965 | [PDF](https://arxiv.org/pdf/2602.03965v1)

**作者:** Taveesh Sharma `[一作]` (University of Chicago), Nicole Marwell `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在城市级别住宅网络测量中，基于端到端 RTT 时间序列的变化点检测识别共享延迟异常，并提出无拓扑依赖的探针选择算法以降低测量冗余。

**💡 创新点**

创新点在于利用共享异常的幅度与持续时间相似性，无需路由或 traceroute 信息即能实现高覆盖率的代表性探针集合。

**🔧 技术方法**

采用改进的 Jitterbug 与 PELT 变化点检测、IoU 交叠度量、最大加权集合覆盖的贪心近似。

**📊 数据集**

数据集为 4 个月来自 99 台位于芝加哥住宅网络的 Raspberry Pi 采集的高频 RTT 以及 M-Lab 服务器目的地。

**📈 对比分析**

与随机采样及按影响力降序的基线相比，所提算法在覆盖 95% 影响度时仅需 44 台探针，且发现的独特异常数显著更多，平均覆盖率约 95%。

**⚠️ 局限性**

局限性包括变化点检测的噪声导致异常重叠误判、对探针密度和地理/ISP 多样性依赖、以及未验证跨城市/跨供应商的泛化能力。

---

## 65. Knowledge Model Prompting Increases LLM Performance on Planning Tasks

**arXiv ID:** 2602.03900 | [PDF](https://arxiv.org/pdf/2602.03900v1)

**作者:** Erik Goh `[一作]` (Georgia Institute of Technology), Ashok Goel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8325 | [OpenAlex ID](https://openalex.org/A5007028896)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过将Task‑Method‑Knowledge（TMK）框架嵌入Prompt，实现对大型语言模型（LLM）在PlanBench Blocksworld规划任务中的推理与规划能力提升，主要在随机、经典、神秘三种变体上进行评估。

**💡 创新点**

创新点在于：①将认知科学与教育学中的TMK框架直接用于Prompt设计；②将TMK作为一种“符号支架”，显著转移模型从文本推理到代码式符号推理；③首次在PlanBench基准上观察到“性能反转”，即在随机符号任务上优于神秘语义任务。

**🔧 技术方法**

技术手段包括：使用JSON格式化的TMK结构进行Prompt，采用one‑shot示例；使用OpenAI旗舰模型（GPT‑4、GPT‑5、o1系列）进行推理；结合PlanBench自动验证器对完整计划进行精确评分。

**📊 数据集**

数据集为PlanBench Blocksworld三种变体（Classic、Mystery、Random），每种变体提供标准规划问题与对应真值。

**📈 对比分析**

与传统纯文本Prompt比较，TMK Prompt在所有旗舰模型上均提升规划正确率；以o1在Random域从31.5%提升至97.3%，GPT‑5在Random域从92.5%提升至99.0%；在Mystery域部分模型（如o1-mini）出现下降，表明TMK对语义干扰的处理有差异。

**⚠️ 局限性**

局限性：①仅在Blocksworld任务上验证，缺乏对物流、多智能体等更复杂规划域的评估；②对模型规模敏感，小模型在神秘域表现下降；③仍需进一步探究TMK结构对模型内部推理机制的具体影响与可迁移性。

---

## 66. Interfaze: The Future of AI is built on Task-Specific Small Models

**arXiv ID:** 2602.04101 | [PDF](https://arxiv.org/pdf/2602.04101v1)

**作者:** Harsha Vardhan Khurdula `[一作]` (JigsawStack), Yoeven D Khemlani `[通讯]` (JigsawStack)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在 Interfaze-Beta 系统中，将大语言模型的应用拆分为：小型感知模型（OCR、ASR、图像/图表检测等）构建多模态上下文，检索层将外部资源编译成结构化状态，动作层通过控制器调用工具链后再交给大型 LLM 输出答案。

**💡 创新点**

创新点在于：① 将感知与检索视为一体化的上下文构建模块，而非单纯工具调用；② 采用多模态小型模型与结构化编译器，把原始多模态内容压缩为精炼上下文，显著减少大模型推理量；③ 通过控制器实现成本与质量平衡的工具链调度。

**🔧 技术方法**

使用的技术包括：轻量级 OCR 与布局解析、图表/图解解析、ASR 与说话人分离、开放词汇目标检测+SAM2 细分、语义检索索引、沙箱执行、基于小型 Transformer 的语言模型以及基于成本/延迟估计的控制策略。

**📊 数据集**

实验数据集涵盖：知识与推理（MMLU-Pro、MMLU）、科学问答（GPQA‑Diamond）、竞赛数学（AIME‑2025）、代码评测（LiveCodeBench v5）、多模态认知（MMMU、AI2D、ChartQA）及多语言语音识别（Common Voice v16）。

**📈 对比分析**

在与 GPT‑4.1、Gemini‑2.5 Pro、Claude‑Sonnet 4 等主流模型对比时，Interfaze‑Beta 在多数基准上实现或逼近 state‑of‑the‑art 结果，例如 MMLU 91.38、AI2D 91.51、ChartQA 90.88、Common Voice 90.8，且在 AIME‑2025 取得 90.0 的显著领先；其优势主要来自上下文压缩与工具链调度。

**⚠️ 局限性**

局限性包括：① 由于上下文构建、检索与工具调用的多步流程导致推理延迟；② 控制器有时会过度调用工具或检索，产生不必要的计算成本；③ 代码评测表现仍落后于专门的长循环代理系统，表明在复杂程序调试方面尚需改进。

---

## 67. Automatic Classification of Pedagogical Materials against CS Curriculum Guidelines

**arXiv ID:** 2602.03962 | [PDF](https://arxiv.org/pdf/2602.03962v1)

**作者:** Erik Saule `[一作]` (University of North Carolina at Charlotte), Razvan Bunescu `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 7881 | [OpenAlex ID](https://openalex.org/A5020435927)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究利用传统 NLP 与大型语言模型自动化对 CS 课程材料进行 ACM/IEEE 2013 课程指南的分类。

**💡 创新点**

创新点在于将基于词法提取、词向量匹配与 LLM 评估相结合，并通过上下文扩展和知识单元剪枝提升分类召回率。

**🔧 技术方法**

使用的技术包括 NLTK 词性标注、基名词短语抽取、GloVe 词向量+余弦相似度以及 Llama‑3.3 大语言模型，并采用异步并行请求和批量/上下文增强提示。

**📊 数据集**

使用的实验数据集为 170 篇公开课程材料（CS1、数据结构、算法、体系结构、网络、操作系统）以及包含 2790 类别的 ACM/IEEE 2013 指南。

**📈 对比分析**

通过与人工分类对比，传统方法召回率 11–21%，LLM 二分类 18.6%，5 分评分 28.6%，批量查询 42.3%，加上下文/知识单元剪枝后 47.7%，显著提升召回率，且自动化处理时间从手工一天降至数小时。

**⚠️ 局限性**

局限性包括样本量小、标注不统一、仅评估召回率、LLM 计算与成本高、缺乏教师受控实验，以及对无文本或误标文件的误检。

---

## 68. Reversible Deep Learning for 13C NMR in Chemoinformatics: On Structures and Spectra

**arXiv ID:** 2602.03875 | [PDF](https://arxiv.org/pdf/2602.03875v1)

**作者:** Stefan Kuhn `[一作]` (Tartu University), Eero Vainikko `[通讯]` (Tartu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并训练了一个可逆的条件可逆神经网络，能够在同一模型中实现分子结构到^13C NMR谱的预测以及谱到结构的反推；

**💡 创新点**

创新点在于将谱和结构映射到同一潜在空间，利用128位谱码与896位残差变量实现信息分离，展示了可逆性与不确定性联合建模的可行性；

**🔧 技术方法**

使用i-RevNet风格的可逆块构建的条件可逆神经网络（cINN），配合距离感知的二元交叉熵损失、稀疏正则化等技术；

**📊 数据集**

基于nmrshiftdb2的数据集，挑选了仅含碳氢、最多17个碳的1567条分子结构及其^13C谱，分别编码为4×16×16的邻接矩阵和1024位二进制谱；

**📈 对比分析**

采用5个epoch训练，模型在训练集上可逆性几乎完美；在谱预测上F1值在0.155–0.184之间，显著高于随机水平（约0.064），但整体性能仍低于专门的谱预测模型；

**⚠️ 局限性**

局限性包括样本量有限、仅覆盖C/H结构、谱编码过于粗糙、Y_latent与Z_free未完全分离、反向结构生成结果仅粗略且不稳定。

---

## 69. Decoupling Time and Risk: Risk-Sensitive Reinforcement Learning with General Discounting

**arXiv ID:** 2602.04131 | [PDF](https://arxiv.org/pdf/2602.04131v1)

**作者:** Mehrdad Moghimi `[一作]` (York University), Hyejin Ku `[通讯]` (York University)

**通讯引用:** 891 | [OpenAlex ID](https://openalex.org/A5067984321)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套支持通用折扣函数与OCE风险度量的分布式强化学习框架，并通过股票增广实现时间与风险偏好可调；

**💡 创新点**

创新点在于将非指数折扣与风险敏感目标统一纳入分布式DP，设计了多时程近似与无限期截尾策略，解决了传统方法的时序不一致与收敛问题；

**🔧 技术方法**

核心技术包括股票增广分布式Bellman算子、时间相关的分布式价值与策略迭代、OCE风险度量的外部优化、指数折扣集成的多时程近似及截尾风险中性尾部策略；

**📊 数据集**

实验数据集涵盖目标导向财富管理任务（GBWM）和Atari 2600游戏套件（共50款），使用公开的强化学习基准环境与自定义金融模拟；

**📈 对比分析**

方法通过与时序不一致基线（保持折扣不随时间变化）对比，验证时间一致性对性能的提升；在Atari游戏中平均提升约39.9%，中位提升约18.1%；在GBWM任务中展示了偏好逆转与风险敏感决策的差异；

**⚠️ 局限性**

局限性包括：1) 无限期问题需要截尾并假设风险中性尾部；2) 需要预知有限周期或进行网格搜索，导致计算开销；3) 在高维复杂环境下对多时程近似的收敛性与表示学习仍有待深入；4) 对非平滑OCE（如CVaR）的理论分析尚不完整。

---

## 70. Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal

**arXiv ID:** 2602.04053 | [PDF](https://arxiv.org/pdf/2602.04053v1)

**作者:** Rio Aguina-Kang `[一作]` (University of California), Matheus Gadelha `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出一种无训练、基于VLM的迭代前景物体移除和深度对齐的单图3D场景重建框架。

**💡 创新点**

创新点在于利用VLM作为场景指挥官自动决定移除顺序，结合无监督物体移除与多图深度对齐，实现对遮挡场景的精确amodal分割与3D重建。

**🔧 技术方法**

使用了ChatGPT-4o、Grounded-SAM、HQ‑SAM、Flux Kontext、LaMa、Stable Diffusion inpainting、Hunyuan2、Monocular depth估计、VGGT、ICP等开源基础模型。

**📊 数据集**

在3D‑Front（合成室内）和ADE20K（真实场景）两个数据集上进行实验。

**📈 对比分析**

与Gen3DSR、MIDI*等基准相比，在Chamfer距离、F‑score、分割IoU等指标上均有显著提升，尤其在深度对齐和物体级别精度上表现优异。

**⚠️ 局限性**

局限性包括物体对齐不稳定、移除过程可能产生伪影、对非常复杂遮挡时仍有误检，以及依赖外部基础模型的性能波动。

---

## 71. Synthesizable Molecular Generation via Soft-constrained GFlowNets with Rich Chemical Priors

**arXiv ID:** 2602.04119 | [PDF](https://arxiv.org/pdf/2602.04119v1)

**作者:** Hyeonah Kim `[一作]` (Mila - Quebec AI Institute), Alex Hernandez-Garcia `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在预训练的 SMILES 语言模型上进行 GFlowNet 的后训练，利用对比学习实现软约束，使生成的分子在保持高目标奖励的同时具备可合成性。

**💡 创新点**

创新点在于：① 将可合成性建模为分布级的软约束而非硬规则；② 通过对比辅助损失在重放缓冲区内分离可合成与不可合成轨迹；③ 充分利用 GFlowNet 的离线重放机制，快速适配变化的约束。

**🔧 技术方法**

使用技术包括 GFlowNet（Trajectory Balance / Relative Trajectory Balance）、预训练 SMILES 语言模型（GP‑MolFormer）、对比学习辅助损失、AiZynthFinder 反向合成评估、Uni‑Dock Vina 分子对接。

**📊 数据集**

数据集主要为公开化学大语料库（PubChem、ZINC、ChEMBL），以及使用 105 条反应模板和 Enamine 库的合成路径库；评测任务涵盖 sEH 绑定亲和力、LIT‑PCBA 五个靶点的对接评分以及 PMO 基准（GSK3β、DRD2）。

**📈 对比分析**

与基准方法（FragGFN、SynFlowNet、RGFN、RxnFlow、SMILES‑RL 等）对比，本文模型在可合成率上达到约 95%（高于 100% 的传统反应式方法），在目标奖励上超过或相当于基准，同时保持多样性；在对接任务中平均 Vina 分数最低，AiZynthFinder 成功率最高。

**⚠️ 局限性**

局限性包括：评估仅为计算机模拟，缺乏真实实验验证；对合成路径的判断依赖预设模板和 AiZynthFinder，可能与实际实验合成差异；对极大搜索空间的扩展仍待进一步验证。

---

## 72. MA3DSG: Multi-Agent 3D Scene Graph Generation for Large-Scale Indoor Environments

**arXiv ID:** 2602.04152 | [PDF](https://arxiv.org/pdf/2602.04152v1)

**作者:** Yirum Kim `[一作]` (Gwangju Institute of Science and Technology), Ue-Hwan Kim `[通讯]` (Gwangju Institute of Science and Technology)

**通讯引用:** 492 | [OpenAlex ID](https://openalex.org/A5031220093)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MA3DSG 多代理 3D 场景图生成框架，并设计训练无关的图对齐算法与 MA3DSG-Bench 基准。

**💡 创新点**

①首个可扩展多代理 3D 场景图生成方法；②轻量化无学习参数的增量图对齐；③新基准支持大规模、多代理、动态评估。

**🔧 技术方法**

多代理探索、SGFN+PointNet 编码、特征注意网络、增量图对齐算法、轻量化图通信。

**📊 数据集**

3RScan 与 3DSSG 数据集统一为 47 房间，加入重扫序列模拟长时动态。

**📈 对比分析**

与单代理 SGFN、3DSSG 以及多代理 SGFN+SGAligner/SG-PGM 对比；MA3DSG 在 Triplet/Object/Predicate F1 与单代理相当，且在多代理场景下 4–8× 速度提升、98× 数据流量降低，尤其在大规模域表现优异。

**⚠️ 局限性**

对齐阈值敏感，易出现误合并或噪声累积；对极大尺度动态变化仍有误差；未验证不同硬件/网络延迟的鲁棒性；缺乏实时性证明。

---

## 73. Active Epistemic Control for Query-Efficient Verified Planning

**arXiv ID:** 2602.03974 | [PDF](https://arxiv.org/pdf/2602.03974v1)

**作者:** Shuhui Qu `[一作]` (Stanford University), Shuhui Qu `[通讯]` (Stanford University)

**通讯引用:** 966 | [OpenAlex ID](https://openalex.org/A5112694715)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种在部分可观测交互环境下的规划框架——Active Epistemic Control (AEC)，通过区分已验证事实与模型预测，并使用查询与模拟实现自适应信息获取，最终保证计划可行；

**💡 创新点**

创新点在于：①引入主动认知控制层，利用模型预测仅用于候选计划过滤，②通过只依赖已验证事实的 verifier（包含类别一致性检查）来完成计划承诺，③实现推理与知识获取的严格分离，防止预测误导；

**🔧 技术方法**

使用技术包括：大型语言模型做规划与世界模型预测；基于类别的 SQ-BCP 兼容性检查和推理规则的 entailment；uncertainty-guided 查询策略；查询宏动作、预测模型接口、基于已验证事实的 verifier；

**📊 数据集**

使用数据集：ALFWorld（134 个测试任务）和 ScienceWorld（Seen/Unseen 任务集）；

**📈 对比分析**

与直接 LLM、ReAct、WKM、WALL-E、Query-Only 等基线对比，AEC 在 ALFWorld 取得 98.7% 成功率，仅略低于 WALL-E2.0，且重规划次数明显减少；在 ScienceWorld 上取得 61.36%/58.62% 成功率，超过 WKM 及其他基线；

**⚠️ 局限性**

局限性：依赖查询和验证接口的准确性，若 grounder 或查询 oracle 存在系统性误差会导致保守失败；推理规则不完整导致需更多查询；未统一优化查询阈值与 token 使用；无法处理系统性视觉误读等问题。

---

## 74. An Intuitionistic Fuzzy Logic Driven UNet architecture: Application to Brain Image segmentation

**arXiv ID:** 2602.04227 | [PDF](https://arxiv.org/pdf/2602.04227v1)

**作者:** Hanuman Verma `[一作]` (Bareilly University), Akshansh Gupta `[通讯]` (National Institute of Science Communication and Policy Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种在U‑Net编码器中融入直觉模糊逻辑（隶属度、非隶属度、犹豫度）的脑MRI图像分割框架IF‑UNet，以解决部分体积效应导致的不确定性。

**💡 创新点**

创新点在于将直觉模糊集理论与U‑Net结合，利用Sugeno消元函数调节犹豫度，从而在像素边界处更精准地捕捉不确定性信息，显著提升分割质量。

**🔧 技术方法**

使用的技术包括直觉模糊集理论、Sugeno消元函数、卷积神经网络U‑Net（及其Attention版本）、交叉熵损失、ReLU激活、Adam优化器、批归一化和Dropout等。

**📊 数据集**

使用公开的Internet Brain Segmentation Repository（IBSR）数据集，包含20份T1加权MRI扫描及其专家标注。

**📈 对比分析**

通过与标准U‑Net和Attention U‑Net在相同数据集、相同训练/测试拆分（80:20）下的准确率、Dice系数和IoU进行对比，IF‑UNet在不同λ值中表现最优；λ=1.2时得到AC=0.9924、DC=0.9892、IoU=0.9788，均高于基线模型。

**⚠️ 局限性**

主要限制是计算复杂度显著提升，IF‑UNet拥有更多参数且推理时间大幅增加；在极度畸变或噪声高的图像中仍可能出现误分，且对异常结构的鲁棒性有限。

---

## 75. When Chains of Thought Don't Matter: Causal Bypass in Large Language Models

**arXiv ID:** 2602.03994 | [PDF](https://arxiv.org/pdf/2602.03994v1)

**作者:** Anish Sathyanarayanan `[一作]` (Birla Institute of Technology and Science), Aarush Rathore `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套诊断框架，结合行为监控和因果探测，评估链式推理（CoT）是否真正影响答案。

**💡 创新点**

首次将行为模式评分与隐藏状态补丁因果探测相结合，量化CoT的因果影响（CMI）并提出旁路得分。

**🔧 技术方法**

采用正则模式与语义相似度的风险评分、隐藏状态补丁（patching）和对数概率差计算等技术。

**📊 数据集**

使用算术、逻辑、QA、TruthfulQA 与 StrategyQA 等小规模测试集进行评估。

**📈 对比分析**

与标准提示对比，审计提示提升风险评分平均 +5.10，但因果探测显示大多数实例 CMI≈0，说明旁路现象普遍存在。

**⚠️ 局限性**

局限在于仅针对少量样本，行为特征无法保证因果依赖，且方法对模型规模与提示分布敏感。

---

## 76. Rational ANOVA Networks

**arXiv ID:** 2602.04006 | [PDF](https://arxiv.org/pdf/2602.04006v1)

**作者:** Jusheng Zhang `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2275 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于功能 ANOVA 分解和可学习的 Padé 形式有理单元的网络结构，能够在保持深度训练稳定性的前提下替换传统 FFN 的非线性激活。

**💡 创新点**

创新点在于将低阶 ANOVA 交互拓扑与严格正分母有理函数相结合，并通过残差门控实现近恒等初始化，既避免了极点导致的不稳定，又提升了对尖锐转折和近奇异行为的建模能力。

**🔧 技术方法**

使用 Padé 风格低阶有理单元、功能 ANOVA 分解、正分母约束、残差门控以及与 Transformer FFN 的无缝集成技术。

**📊 数据集**

在多个视觉基准（MNIST、EMNIST、FMNIST、SVHN、CIFAR‑10/100）、ImageNet‑1K、PolyU 真实噪声去噪、TabArena 以及科学符号回归任务上进行实验。

**📈 对比分析**

在严格匹配参数/ FLOPs 的对比实验中，RAN 与参数等价的 MLP、KAN、KAF 相比在多数任务上取得同等或更高的准确率，并在 ViT‑Tiny 上实现 1.9% 的 Top‑1 提升。

**⚠️ 局限性**

局限性包括受限的计算资源导致模型规模和预训练范围受限，未在大型基础模型（如大规模语言模型）或极端分布外场景中验证，且有理单元的可解释性与偏差风险仍需进一步评估。

---

## 77. TruKAN: Towards More Efficient Kolmogorov-Arnold Networks Using Truncated Power Functions

**arXiv ID:** 2602.03879 | [PDF](https://arxiv.org/pdf/2602.03879v1)

**作者:** Ali Bayeh `[一作]` (University of Regina), Malek Mouhoub `[通讯]` (University of Regina)

**通讯引用:** 1153 | [OpenAlex ID](https://openalex.org/A5058768233)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的 KAN 变体 TruKAN，使用截断幂函数取代 B‑spline，保持可解释性并提升计算效率。

**💡 创新点**

创新点在于：① 用截断幂基替代递归 B‑spline，①2 ①3；② 将多项式基与截断幂基结合，提供全局与局部的可解释性；③ 引入共享与单独 knots 的配置，使模型既能降低参数量又能保持表达能力。

**🔧 技术方法**

技术方法包括：TruKAN 架构、截断幂函数与多项式基、共享/单独 knots、Layer Normalization、AdamW+LookAhead 混合优化、EfficientNet‑V2 作为特征提取器、Albumentations 数据增强、CutMix/MixUp、权重衰减与归一化。

**📊 数据集**

使用四个计算机视觉基准数据集：CIFAR‑10、CIFAR‑100、Oxford‑IIIT Pets、STL‑10。

**📈 对比分析**

在保持分类器参数量相似的前提下，比较小型与大型模型的 F1、准确率、训练时间、GFLOPs 与显存占用。TruKAN 在准确率与 F1 上与或优于 MLP、KAN、SineKAN；训练时间比传统 KAN 快 3–4 倍；显存使用显著更少；GFLOPs 与 KAN 相当。

**⚠️ 局限性**

局限性：① 对高维/大规模 knots 可能出现数值不稳定；② 单独 knots 版本显存消耗大；③ 在某些细粒度任务（如 Oxford‑Pets）部分模型表现略低；④ 仍需在强化学习、时间序列等更复杂任务中验证泛化能力。

---

## 78. ZKBoost: Zero-Knowledge Verifiable Training for XGBoost

**arXiv ID:** 2602.04113 | [PDF](https://arxiv.org/pdf/2602.04113v1)

**作者:** Nikolas Melissaris `[一作]` (CNRS), Chenkai Weng `[通讯]` (Arizona State University)

**通讯引用:** 277 | [OpenAlex ID](https://openalex.org/A5030545742)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 ZKBoost，首次为 XGBoost 提供零知识训练证明（zkPoT），通过将 XGBoost 重写为固定点算法并设计可被算术电路证明的验证流程，允许模型所有者在不泄露数据或模型参数的情况下证明其模型是基于指定数据集正确训练得到的。

**💡 创新点**

创新点包括：① 开发兼容算术电路的固定点 XGBoost 实现；② 设计可插拔的 zkPoT 通用模板；③ 采用 VOLE‑ZK 背景实现高效的非线性运算证明（比较、除法、截断）并加入全局范围检查，提升证明安全性与效率。

**🔧 技术方法**

使用的技术主要有：固定点数值表示、截断泰勒级数逼近 sigmoid 与 log‑odds、算术电路与 VOLE‑ZK 交互式证明体系、位分解与 Mersenne 质数检查、全局范围校验与非线性算子证明。

**📊 数据集**

实验使用了多组公开结构化数据集（如 UCI 数据集、Kaggle 竞赛数据等）来验证算法的准确性与证明性能。

**📈 对比分析**

与标准浮点 XGBoost 的对比结果显示，固定点实现的准确率与浮点版相差不超过 1%；证明时间、带宽和内存开销在常见数据集上均保持在可接受范围内，证明速度远快于此前基于 SNARK 的方案，且通信量与内存占用均低于传统交互式证明。

**⚠️ 局限性**

局限性包括：① 仅支持固定点 XGBoost，未覆盖 XGBoost 的全部生产特性（如量化直方图、缺失值处理等）；② 目前实现仍为交互式证明，尚未实现零交互或可验证的 SNARK 版本；③ 在极大规模数据集上证明成本仍相对较高，仍需进一步压缩电路与证明时间。

---

## 79. A Modern System Recipe for Situated Embodied Human-Robot Conversation with Real-Time Multimodal LLMs and Tool-Calling

**arXiv ID:** 2602.04157 | [PDF](https://arxiv.org/pdf/2602.04157v1)

**作者:** Dong Won Lee `[一作]` (Massachusetts Institute of Technology), Hae Won Park `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2566 | [OpenAlex ID](https://openalex.org/A5090151532)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并验证了一套基于实时多模态大型语言模型与工具调用（用于关注控制与主动感知）的简易系统，用于在家庭场景下实现具身化的情境对话。

**💡 创新点**

创新点在于：①将实时多模态LLM与面向注意力与感知的工具接口（如跟踪、搜索、视角检索）紧密耦合，形成端到端可在对话中即时发起感知动作的通用“对话管理+工具调用”框架；②提出了一组最小化但完整的工具族；③设计了专门评估该框架的实验套件与评估协议。

**🔧 技术方法**

使用技术包括：OpenAI Realtime API / Gemini Live（实时多模态LLM），LLM函数调用接口，基于视觉的目标与人类跟踪模型，几何绑定层（将图像空间映射到机器人视角），轻量级关注控制循环，VLM（视觉语言模型）用于视角检索，机器人控制接口（头部转向、摄像头视角）。

**📊 数据集**

实验使用六类家庭情境数据集：姿态训练、白板教学、灯具摆放、植物诊断、服装搭配、物品寻找；每个情境均记录多段交互视频，用于人工标注工具调用正确性与主观交互质量。

**📈 对比分析**

对比方法：对四个系统变体（完整、去人跟踪、去物体跟踪、不同后端）进行交互视频评估。客观指标：宏观准确率0.72–0.77，精度0.88–0.84，召回率0.60–0.62；主观指标：对话质量、流畅度、关注与对象引用得分平均≥4.5/5；延迟约700 ms，Gemini Live成本显著低于OpenAI Realtime。

**⚠️ 局限性**

局限性包括：①基于回合的LLM导致对子回合事件（如微观注视、打断）响应不足；②标注主观性与相机视角导致的一致性有限；③仅涵盖六个情境，难以覆盖极端或长时序交互；④系统仍存在偶发感知或工具失效；⑤对不同机器人平台的适配需额外实现适配层。

---

## 80. Frontend Token Enhancement for Token-Based Speech Recognition

**arXiv ID:** 2602.04217 | [PDF](https://arxiv.org/pdf/2602.04217v1)

**作者:** Takanori Ashihara `[一作]` (NTT, Inc.), Marc Delcroix `[通讯]` (NTT, Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文系统评估了多种前端增强方法（wave‑to‑wave、token‑to‑token、vector‑to‑token、wave‑to‑token），并在token语音识别后端上验证其对噪声鲁棒性的提升，发现wave‑to‑token（W2T‑E）在噪声环境下取得最佳效果。

**💡 创新点**

创新点包括：①首次对Token ASR前端进行全面系统的比较；②提出并验证V2T‑E和W2T‑E两种全新Token级前端；③证明Token ASR在合适的前端下可超越传统基于连续特征的ASR，同时保持更低的计算成本。

**🔧 技术方法**

使用技术主要有：自监督学习（WavLM）、k‑means量化生成语义Token、CTC损失训练Token级网络、E‑Branchformer与TCN结构、Conv‑TasNet和TF‑GridNet语音增强、LayerDrop正则化等。

**📊 数据集**

采用CHiME‑4单声道模拟与真实数据集进行实验。

**📈 对比分析**

比较方法：以词错误率（WER）和Token编辑距离（UED）为指标，对不同前端+后端组合进行评估。W2T‑E在所有噪声级别下的WER最低（约3.2–4.1%），优于基于连续特征的最佳系统，并与SOTA IRIS系统相当；UED亦显著下降，显示Token化噪声鲁棒性提升。

**⚠️ 局限性**

局限性：①未与Token ASR联合微调，未探索进一步降噪效果；②仅在单声道CHiME‑4上验证，未评估多声道或实时场景；③UED与WER的相关性不高，需要更可靠的Token级评估指标；④对其他语音任务的泛化仍待验证。

---

## 81. SuperPoint-E: local features for 3D reconstruction via tracking adaptation in endoscopy

**arXiv ID:** 2602.04108 | [PDF](https://arxiv.org/pdf/2602.04108v1)

**作者:** O. Leon Barbed `[一作]`, Ana C. Murillo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为 SuperPoint‑E 的端腔内视频局部特征提取方法，并在结构光与结构相机（SfM）中实现更高质量的三维重建。

**💡 创新点**

创新点在于设计“Tracking Adaptation”训练策略：利用 COLMAP 生成的 3D 重建轨迹作为监督，既提升检测精度，又使描述子在多视角下保持一致，从而获得更具辨识度和可重复性的特征。

**🔧 技术方法**

技术上采用 SuperPoint 结构的全卷积网络，改进检测头与描述子头；使用多图像批次训练、可靠轨迹监督、三元组损失；并将 COLMAP Guided Matching 与 SIFT、SuperPoint 进行对比。

**📊 数据集**

主要数据集为 EndoMapper（真实结肠镜、胃镜序列）以及 C3VD（结肠镜深度数据）。

**📈 对比分析**

与 SIFT、原版 SuperPoint 在 EM‑Test、EM‑Full 以及胃镜子集上对比，SuperPoint‑E 在检测精度、3D 点数、轨迹长度、点分布和反射抑制方面均明显优于基线，且使用 BF 匹配已达与 Guided Matching 相当的重建质量，覆盖率提升约 2 倍。

**⚠️ 局限性**

局限包括：训练需要已完成的 COLMAP 重建，无法处理完全无重建的序列；描述子维度高导致计算时间与存储成本上升；对极端光照或大视角变化的鲁棒性尚待进一步验证。

---

## 82. Partition Trees: Conditional Density Estimation over General Outcome Spaces

**arXiv ID:** 2602.04042 | [PDF](https://arxiv.org/pdf/2602.04042v1)

**作者:** Felipe Angelim `[一作]` (Independent Researcher), Alessandro Leite `[通讯]` (INSA Rouen Normandy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了Partition Trees与Partition Forests两种基于树的非参数条件密度估计框架，能够统一处理连续、分类及混合型目标变量。

**💡 创新点**

创新点在于将条件密度视为联合空间的分段常数估计，并通过直接最小化条件负对数似然来进行贪婪的树分裂；同时引入了探索式分裂保证收敛性，并通过袋装平均实现ensemble。

**🔧 技术方法**

主要技术包括测度论下的Radon–Nikodym导数、数据自适应的联合空间划分、基于条件负对数似然的分裂增益评估、以及基于优先队列的贪婪best‑first树构造。

**📊 数据集**

实验使用了多种公开数据集：分类集有iris、breast、wine、digits、spam、support2、letter、bank、adult；回归集有diabetes、boston、energy、concrete、kin8nm、air、power、naval、california、protein。

**📈 对比分析**

与CART、Random Forest、XGBoost、CDTree、CADET、NGBoost等方法比较，Partition Tree在多数单棵树任务中log‑loss更低，Partition Forest在ensemble任务中往往优于Random Forest，且在大样本回归任务中超越CADET，表现出更强的概率预测与鲁棒性。

**⚠️ 局限性**

局限性包括：对大规模数据仍需高计算开销；缺乏基于提升的改进；以及对深层次特征交互的捕捉仍不如某些boosting方法。

---

## 83. An Empirical Survey and Benchmark of Learned Distance Indexes for Road Networks

**arXiv ID:** 2602.04068 | [PDF](https://arxiv.org/pdf/2602.04068v1)

**作者:** Gautam Choudhary `[一作]` (Purdue University), Walid G. Aref `[通讯]` (Purdue University)

**通讯引用:** 9836 | [OpenAlex ID](https://openalex.org/A5000123743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对基于机器学习的道路网络最短距离索引进行系统评测。

**💡 创新点**

创新点在于首次从训练时间、查询延迟、存储和精度四维度对十种学习式索引进行基准实验并发布统一开源代码。

**🔧 技术方法**

使用的技术包括全连接网络、图神经网络、功能解码器、梯度提升树等多种模型。

**📊 数据集**

实验采用七个真实道路网络（Jinan、Shenzhen、Chengdu、Beijing、Shanghai、NewYork、Chicago）以及基于轨迹的查询集。

**📈 对比分析**

方法通过与传统距离或点基准（Manhattan、Landmark、HCL）对比，发现CatBoost在精度上最优，RNE在延迟上最快，整体学习式索引显著优于经典索引。

**⚠️ 局限性**

主要局限在于对动态更新缺乏机制、对未见节点的泛化有限、以及训练时间仍较大。

---

## 84. Lyapunov Constrained Soft Actor-Critic (LC-SAC) using Koopman Operator Theory for Quadrotor Trajectory Tracking

**arXiv ID:** 2602.04132 | [PDF](https://arxiv.org/pdf/2602.04132v1)

**作者:** Dhruv S. Kushwaha `[一作]`, Zoleikha A. Biron `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在二维四旋翼飞行器轨迹跟踪任务中，提出了基于 Lyapunov 约束的 Soft Actor‑Critic（LC‑SAC）算法，结合离线 Koopman 逼近与在线主从双重优化，实现了对控制策略的稳定性约束。

**💡 创新点**

创新点包括：① 通过 EDMD 离线得到线性化模型并用 DARE 求解闭式二次控制 Lyapunov 函数，从而避免训练额外的 Lyapunov 网络；② 在 SAC 的 actor 更新中加入基于 Lyapunov 限制的 Lagrangian 罚项，并采用投影上升自适应调节乘子；③ 通过 CVaR 方式聚焦最坏情况的 Lyapunov 违规，从而提高稳定性收敛。

**🔧 技术方法**

使用的技术主要有：强化学习（Soft Actor‑Critic）、Lyapunov 稳定性理论、Koopman 变换与 EDMD、离散代数 Riccati 方程（DARE）、主从双重优化（Lagrangian 罚项与投影上升）、PyTorch 深度网络实现。

**📊 数据集**

实验数据集来自安全控制实验平台 safe‑control‑gym 的 2D 四旋翼轨迹跟踪任务（包含 figure‑8、circle、square 等参考轨迹），离线模型训练使用 PID 控制器产生的 17000 条状态转移样本。

**📈 对比分析**

与传统无约束 SAC 进行对比。LC‑SAC 在训练收敛后获得更高的平均奖励、评估时更低的方差，并在 X‑Z 平面轨迹跟踪上表现出更小的误差和更低的轨迹波动；最大回报虽略低于 SAC，但最终稳定性能显著提升。

**⚠️ 局限性**

局限性包括：Lyapunov 约束仅基于 EDMD 的预测模型，缺乏对模型误差的严格界定；当真实系统与逼近模型差距较大时，稳定性保证可能失效；此外，离线模型学习与 DARE 计算带来额外的计算开销，且超参数（如 η、λ、ζ 等）需要经验调优。

---

## 85. Point2Insert: Video Object Insertion via Sparse Point Guidance

**arXiv ID:** 2602.04167 | [PDF](https://arxiv.org/pdf/2602.04167v1)

**作者:** Yu Zhou `[一作]` (Institute of Artificial Intelligence), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 61661 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于稀疏正负点控制的视频对象插入框架，支持用户低成本精准插入

**💡 创新点**

①统一点与掩码两种控制方式；②双阶段训练 + 掩码引导教师蒸馏；③构建1.3M大规模对象插入数据集及评测基准

**🔧 技术方法**

DiT‑based 生成模型、VAE 编码、流匹配训练、点映射与掩码统一、教师蒸馏、点感知增强损失

**📊 数据集**

通过自动化对象去除与重建生成1.3M视频对；使用DAVIS、Web视频等自建数据做评测；引入PointBench基准

**📈 对比分析**

与多种掩码基和指令基视频编辑方法对比，点控制下的定位准确率最高，背景保持与时序一致性均优于基线；掩码控制也能保持领先；在多指标上均处于前列

**⚠️ 局限性**

在全局提示下插入效果略降；模型参数仅1.3B，较大模型性能有限；对极端点密度/尺寸仍有挑战

---

## 86. On the Uncertainty of Large Language Model-Based Multi-Agent Systems

**arXiv ID:** 2602.04234 | [PDF](https://arxiv.org/pdf/2602.04234v1)

**作者:** Yuxuan Zhao `[一作]` (Yantai Research Institute of Harbin Engineering University), Ningxin Su `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对大语言模型多智能体系统的熵特征进行系统分析，比较单智能体与多智能体在不同拓扑和六大基准任务上的性能，并提出基于熵的判别器（Entropy Judger）实现无标签的高质量答案挑选。

**💡 创新点**

创新点包括：①首次全面刻画多智能体中的多层熵动力学；②发现单智能体在43.3%的场景中优于多智能体，且第一轮不确定性决定最终效果；③提出三条原则——Certainty Preference、Base Uncertainty、Task Awareness；④基于熵的判别器在所有配置下均能提升准确率。

**🔧 技术方法**

技术方法：使用公开LLM（LLaMA、Qwen系列）构建四种MAS拓扑，提取245个token/轨迹/轮次级熵特征；利用梯度提升树（XGBoost/LightGBM）做分类并用SHAP解释；在pass@k中利用熵判别器选择最有可能正确的答案。

**📊 数据集**

数据集：六个基准任务，涵盖数学（MATH、AQuA、ARC等）、代码生成（CodeSearchNet）和知识问答（HotpotQA、TriviaQA 等）。

**📈 对比分析**

评估方式：对比单智能体与四种MAS架构，在5个模型（LLaMA 3.1/3.2、Qwen3 0.6/4/8B）与6个数据集上测算准确率、token消耗等；Entropy Judger 在所有配置上提升准确率，尤其在RL微调基模型下，MAS 能够完全击败单智能体。

**⚠️ 局限性**

局限性：实验仅覆盖公开小模型且固定为 2 轮交互，缺少大模型或多轮动态研究；熵特征需要完整概率输出，商业 API 可能无法获取；不同任务和更大规模环境下的通用性仍待验证。

---

## 87. Rate-Optimal Noise Annealing in Semi-Dual Neural Optimal Transport: Tangential Identifiability, Off-Manifold Ambiguity, and Guaranteed Recovery

**arXiv ID:** 2602.04110 | [PDF](https://arxiv.org/pdf/2602.04110v1)

**作者:** Raymond Chu `[一作]` (Carnegie Mellon University), Dohyun Kwon `[通讯]` (University of Seoul)

**通讯引用:** 94 | [OpenAlex ID](https://openalex.org/A5081743149)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究半对偶神经最优传输（SNOT）中出现的伪解问题，并提出通过加噪声平滑与终止噪声水平自适应来消除该问题，保证在低维流形数据下的可辨识性与恢复精度。

**💡 创新点**

创新点在于：① 对流形支持数据下半对偶目标的欠约束性质进行几何刻画；② 证明加噪声平滑可去除非可辨识的正交分量并在噪声消失时收敛到唯一的Monge映射；③ 推导出统计噪声阈值 ε_stat(N) 的闭式公式，平衡偏差、采样误差与优化条件；④ 证明在极小噪声下优化 Hessian 失稳，给出停止策略。

**🔧 技术方法**

技术主要包括：半对偶神经最优传输框架、加噪声平滑（Gaussian smoothing）、Wasserstein距离与耦合理论、误差分解（bias–estimation）、梯度与Hessian分析、随机采样与极限分析。

**📊 数据集**

实验使用合成数据集（Perpendicular、One-to-Many）以及高维低维流形数据（d=256，m=2,4,8），并在这些数据上评估传输成本误差与目标分布误差。

**📈 对比分析**

与传统无平滑SNOT（OTM）和常规平滑方法（OTP）对比，本文方法在正交分量误差、传输成本误差、目标分布误差方面均有提升；尤其在高维低维流形场景下，误差更低、收敛更快。

**⚠️ 局限性**

局限性包括：需假设目标分布满足光滑性或可微性；在极端低噪声时需手动设置最小噪声阈值；对复杂非球面流形的解析和收敛性分析仍待深入。

---

## 88. PaperX: A Unified Framework for Multimodal Academic Presentation Generation with Scholar DAG

**arXiv ID:** 2602.03866 | [PDF](https://arxiv.org/pdf/2602.03866v1)

**作者:** Tao Yu `[一作]`, Liang Wang `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 PaperX，一个统一框架，可将学术论文自动转换为 PPT、海报、PR 等多模态展示内容。

**💡 创新点**

创新点在于引入 Scholar DAG 作为中间表示，解耦论文逻辑结构与最终格式渲染，实现跨模态一致的内容组织与分层抽取。

**🔧 技术方法**

使用 Gemini 3 Pro 与 GPT‑4o 进行文本分解与生成，结合 VLM 进行视觉反馈与布局优化，配合图遍历与模板渲染技术。

**📊 数据集**

在 PPTEVAL、Paper2Poster、PRBench 三个公开基准上评测，输入为论文 PDF 及其提取的文本和视觉元素。

**📈 对比分析**

与现有单任务代理、模板及多代理方法对比，PaperX 在所有基准（PPT、海报、PR）上均获得最高得分，且算力与费用更低。

**⚠️ 局限性**

局限性包括对 LLM 生成质量的依赖、对极长文档的剪裁策略不足，以及仅在三种格式上验证，泛化至更多形式仍需进一步探索。

---

## 89. Supervised Learning as Lossy Compression: Characterizing Generalization and Sample Complexity via Finite Blocklength Analysis

**arXiv ID:** 2602.04107 | [PDF](https://arxiv.org/pdf/2602.04107v1)

**作者:** Kosuke Sugiyama `[一作]` (Waseda University), Masato Uchida `[通讯]` (Waseda University)

**通讯引用:** 913 | [OpenAlex ID](https://openalex.org/A5000225743)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种将机器学习过程映射为固定长度失真压缩的理论框架，并利用有限块长分析推导学习算法在最优采样策略下的样本复杂度和泛化误差下界。

**💡 创新点**

将采样视为编码、学习视为解码，实现对任意随机学习算法的下界推导；并将下界分解为过拟合度和归纳偏差不匹配两项，首次与信息理论与稳定性理论中的指标建立直接联系。

**🔧 技术方法**

信息论有限块长分析、失真率函数、方差（dispersion）分析、信息密度、倾斜信息等。

**📊 数据集**

无具体实验数据集，本文为理论推导。

**📈 对比分析**

通过理论推导比较，与现有信息理论和稳定性理论上限不直接比较；本文给出样本复杂度和泛化误差的下界，体现了在最优采样策略下可达到的理论极限。

**⚠️ 局限性**

下界依赖未知的最优采样策略𝒮*，使数值评估困难；关于V_bet作为归纳偏差不匹配度量的有效性尚需进一步验证。

---

## 90. Turning mechanistic models into forecasters by using machine learning

**arXiv ID:** 2602.04114 | [PDF](https://arxiv.org/pdf/2602.04114v1)

**作者:** Amit K. Chakraborty `[一作]` (University of Alberta), Pouria Ramazi `[通讯]` (University of Calgary)

**通讯引用:** 766 | [OpenAlex ID](https://openalex.org/A5057708120)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

基于SINDy（稀疏识别非线性动力学）和机器学习预测时间变化参数的框架，用于从时序数据中自动发现可变系数的常微分方程并进行短期预测。

**💡 创新点**

创新点在于：①允许部分模型参数随时间变化并通过稀疏回归动态选择；②利用随机森林预测这些时间变化参数，再将预测值注入已学得的方程，实现自适应预测；③在理论上给出了时间变化参数模型相对于固定参数模型的有限视界误差上界，说明在非平稳系统中时间变化参数更优。

**🔧 技术方法**

主要技术包括：Sequential Threshold Ridge Regression（STRR）求解稀疏回归、随机森林（RF）预测参数、卷积+长短时记忆（CNN‑LSTM）与梯度提升机（GBM）作为基线模型、滚动窗口交叉验证、欧拉-马里亚马方法求解数值微分方程、以及理论误差传播分析。

**📊 数据集**

数据集包括：①模拟SIR（易感-感染-康复）模型和CR（消费者-资源）模型；②阿尔伯塔省油砂尾渣湖的CO₂、CH₄浓度与天气变量；③跨省湖泊的蓝藻细胞计数与水文、气象、湖泊及流域特征。

**📈 对比分析**

与固定系数SINDy、CNN‑LSTM和GBM进行比较。时间变化参数模型在学习阶段的平均MAE <3%，在预测阶段MAE <6%，相较固定参数模型MAE提升约2–7%，而CNN‑LSTM/GBM的MAE普遍高于20%。最佳超参数（窗口长度与可变参数数量）在交叉验证之外进一步提升预测精度，误差可降至约0.5%（蓝藻数据）或5–10%（气体与SIR/CR）。

**⚠️ 局限性**

主要局限：①交叉验证可能无法捕捉局部最优的窗口/参数数量；②仅使用RF预测参数，缺乏对更强时序建模的探索；③时间变化参数选择基于相关性，易受噪声或瞬时相关误导；④候选库规模大，计算成本高且可能包含冗余或缺失的驱动变量；⑤缺乏对长期预测的稳健性评估，易出现误差累积。

---

## 91. AGMA: Adaptive Gaussian Mixture Anchors for Prior-Guided Multimodal Human Trajectory Forecasting

**arXiv ID:** 2602.04204 | [PDF](https://arxiv.org/pdf/2602.04204v1)

**作者:** Chao Li `[一作]` (Wuhan University of Technology), Hongbo Jiang `[通讯]` (Hunan University)

**通讯引用:** 10982 | [OpenAlex ID](https://openalex.org/A5061448582)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种新的先验建模框架AGMA，用于多模态人类轨迹预测。

**💡 创新点**

创新点在于通过批量聚类与全局分布蒸馏自适应构造高质量的高斯混合先验，理论证明先验质量是预测误差下界，并解决了隐式高斯先验的模式坍塌与离散锚点的局限。

**🔧 技术方法**

所采用的技术包括批量聚类自编码、图聚类、可微阈值与Gumbel-Softmax、交叉注意力机制、Wasserstein距离的可逆传输以及简单的MLP解码器。

**📊 数据集**

实验数据集涵盖 ETH-UCY、Stanford Drone (SDD) 与 JRDB 三大人行轨迹基准。

**📈 对比分析**

在多种基线对比中，AGMA在 ETH-UCY、SDD 与 JRDB 上分别以 mADE_20/mFDE_20 的方式实现最优，分别达到 0.24/0.35、7.23/10.92 与 0.15/0.23，明显优于现有 SOTA 方法。

**⚠️ 局限性**

主要局限在于对观测轨迹信息的依赖，难以充分建模复杂交互场景；先验与解码器的相互作用仍有提升空间，且批量大小与阈值需手动调节。

---

## 92. Control and State Estimation of Vehicle-Mounted Aerial Systems in GPS-Denied, Non-Inertial Environments

**arXiv ID:** 2602.04057 | [PDF](https://arxiv.org/pdf/2602.04057v1)

**作者:** Riming Xu `[一作]` (King Abdullah University of Science and Technology), Eric Feron `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 12614 | [OpenAlex ID](https://openalex.org/A5041459160)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在 GPS‑denied、非惯性环境下，提出了一种仅依赖外部位姿测量的四旋翼无人机控制与状态估计框架，采用 EKF‑UI 估计平台加速度并配合级联 PID 控制实现三维跟踪。

**💡 创新点**

创新点在于将未知输入（平台加速度）融入 EKF 状态，在线补偿非惯性扰动，并与控制器紧耦合，实现无 IMU/GNSS 依赖的稳健跟踪。

**🔧 技术方法**

使用 EKF‑UI 估计器、级联 PID 控制器、外部光学运动捕捉系统以及基于实验的仿真。

**📊 数据集**

实验数据来自两套运动捕捉系统（OptiTrack 与 Qualisys）记录的无人机与移动平台在三种运动模式下的位姿与速度。

**📈 对比分析**

将提出的 EKF‑UI 与标准 EKF 进行对比，结果显示在静止、X 轴直线运动以及 X+Y 轴对角运动三种情形下，EKF‑UI 在速度平滑度、跟踪误差与控制稳定性方面优于基准，尤其在对角运动时误差显著降低。

**⚠️ 局限性**

局限在于仍需外部高精度定位，缺乏对视觉感知误差鲁棒性研究，且平台加速度假设为常数可能不适用于极快变换的环境。

---

## 93. Online Vector Quantized Attention

**arXiv ID:** 2602.03922 | [PDF](https://arxiv.org/pdf/2602.03922v1)

**作者:** Nick Alonso `[一作]` (Zyphra), Beren Millidge `[通讯]` (Zyphra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种在线向量量化注意力（OVQ）层，能够在保持常数内存和线性计算复杂度的同时，在线学习键和值字典，实现高效的长上下文信息混合。

**💡 创新点**

创新点在于将原始 VQ‑attention 的键和值字典都改为在线更新，并借助高斯混合回归理论设计稀疏字典更新机制，显著提升长上下文回忆与学习能力，同时保持与全自注意力相当的性能。

**🔧 技术方法**

使用的技术包括：高斯混合回归（GMR）理论、在线增量学习、k‑means++ 直观初始化、块级并行实现、稀疏字典更新、线性复杂度的注意力计算。

**📊 数据集**

实验数据集涵盖：合成长上下文任务（ICR、ICL）、PG19 长上下文语言建模、短上下文基准（PIQA、Hella 等）。

**📈 对比分析**

与全自注意力、线性注意力、SSM 等基线在 4k–64k 长度下对比；OVQ 在 ICR/ICL 上性能与全自注意力相当、明显优于线性注意力；PG19 长上下文的交叉熵差距仅约 0.02。

**⚠️ 局限性**

局限性在于：目前仅在 ≤500M 参数规模验证；缺乏硬件级实现评估，延迟/吞吐量与更大规模模型的可扩展性仍需进一步研究。

---

## 94. Linguistic Blind Spots in Clinical Decision Extraction

**arXiv ID:** 2602.03942 | [PDF](https://arxiv.org/pdf/2602.03942v1)

**作者:** Mohamed Elgaar `[一作]` (University of Massachusetts Lowell), Hadi Amiri `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 616 | [OpenAlex ID](https://openalex.org/A5074007015)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了临床笔记中不同医学决策类别的语言特征，并将其与基线Transformer模型的提取召回率关联。

**💡 创新点**

揭示决策类别在词汇密度、停用词比例、含糊/否定标记等方面的差异，以及这些特征导致提取失败的机制。

**🔧 技术方法**

使用RoBERTa基础的跨度提取器，并计算七个基于可读性、词汇、实体、停用词、代词、含糊、否定等的语言指标。

**📊 数据集**

采用MedDec数据集（MIMIC‑III衍生的451份出院摘要，含9类决策标签）。

**📈 对比分析**

在验证集上Exact-match召回为48%，在按语料特征分层后，停用词比例最高的层召回仅24%，但放宽IoU阈值可提升至71%。

**⚠️ 局限性**

局限于单一模型、单一数据集、仅评估召回且未考虑精确度，且语言指标仅为表面化，缺乏深层句法/范围信息。

---

## 95. A Consensus-Bayesian Framework for Detecting Malicious Activity in Enterprise Directory Access Graphs

**arXiv ID:** 2602.04027 | [PDF](https://arxiv.org/pdf/2602.04027v1)

**作者:** Pratyush Uppuluri `[一作]` (Purdue University), Sajan Kumar `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个基于共识的贝叶斯框架，用多层目录–用户交互图检测企业目录访问中的恶意用户行为。

**💡 创新点**

创新点在于将意见动力学理论与UEBA（用户行为分析）相结合：①用目录相似矩阵W与动态逻辑矩阵C_i建模多级交互；②通过观测C_i的结构与方差变化识别异常；③采用贝叶斯变分评分机制实现在线可更新的异常概率。

**🔧 技术方法**

主要技术包括：多层图结构建模、行随机化影响矩阵、强连通分量（SCC）分解、意见动力学更新方程、方差激活阈值检测、指数似然映射与贝叶斯后验更新。

**📊 数据集**

使用合成访问图数据（包含n个目录、m个用户、随机生成的W和C矩阵）进行仿真，未使用公开真实企业数据集。

**📈 对比分析**

通过对不同逻辑矩阵（Ĉ、C̅、C̃）进行仿真，观察收敛性、方差变化及贝叶斯得分；与静态先验与在线先验两种设置对比，展示了更高的异常权重导致方差和概率上升；但未给出精确的AUC、F1等评估指标，主要以可视化趋势说明效果。

**⚠️ 局限性**

局限包括：①规模化挑战（用户/目录数千级时矩阵运算成本高）；②对合法行为变动的误报率可能较高；③缺乏真实企业数据验证；④需要进一步集成控制响应与告警机制；⑤长期共识漂移与逻辑矩阵同步的安全与隐私问题。

---

## 96. eCP: Informative uncertainty quantification via Equivariantized Conformal Prediction with pre-trained models

**arXiv ID:** 2602.03986 | [PDF](https://arxiv.org/pdf/2602.03986v1)

**作者:** Nikolaos Bousias `[一作]` (University of Pennsylvania), George Pappas `[通讯]` (University of Pennsylvania)

**通讯引用:** 35270 | [OpenAlex ID](https://openalex.org/A5029243115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了如何将预训练模型的对称性融入后置的置信预测，以减少不确定性集合的大小

**💡 创新点**

提出Equivariantized Conformal Prediction（eCP），通过群平均对非一致性分数进行对称化，实现理论上可收缩的分布与更好的尾部性质

**🔧 技术方法**

采用群平均投影、凸性、ICX顺序、Chernoff/Hoeffding/Bernstein等概率界证明，并在轨迹预测中使用后置方法

**📊 数据集**

使用ETH‑UCY、Stanford Drone Dataset、NBA SportVU等行人/运动轨迹数据集

**📈 对比分析**

与传统CP对比，使用95%/99%置信集的量化误差和覆盖率评估，eCP在保持覆盖率≈95%/99% 的同时缩小了约20–30% 的预测集尺寸

**⚠️ 局限性**

需要已知或近似的对称组；对连续群的平均成本较高；未直接评估在闭环规划中的性能

---

## 97. PromptSplit: Revealing Prompt-Level Disagreement in Generative Models

**arXiv ID:** 2602.04009 | [PDF](https://arxiv.org/pdf/2602.04009v1)

**作者:** Mehdi Lotfian `[一作]` (Chinese University of Hong Kong), Farzan Farnia `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 544 | [OpenAlex ID](https://openalex.org/A5017160178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种名为 PromptSplit 的框架，用于通过对联合提示‑输出核协方差差的特征分解来检测和分析不同生成模型在不同提示下的行为差异。

**💡 创新点**

创新点在于：① 将提示与生成结果通过张量积嵌入到同一特征空间，构造联合核；② 用核协方差差的特征值分解捕捉提示依赖的模型差异；③ 提出随机投影近似，理论上误差为 O(1/r²)，实现大规模样本下的可扩展性；④ 将方法应用于文本‑图像、文本‑文本与图像‑标题等多种跨模态生成任务，提供可解释的提示差异可视化。

**🔧 技术方法**

主要技术包括：核方法（Gaussian 及 RBF 核）、张量积核、特征分解、随机投影（Gaussian 及随机 Fourier 特征）以及基于梯度的引导（在扩散模型中使用）。

**📊 数据集**

使用的公开数据集包括：MNIST、MNIST‑M、MS‑COCO 验证集、NQ‑Open（开放式问答）以及各类预训练图像和文本嵌入模型（DinoV2‑giant、Sentence‑Bert）。

**📈 对比分析**

比较方法是对两模型的联合提示‑输出协方差矩阵做差，然后求其正特征向量，特征值大小对应不同提示聚类的差异程度。实验表明：在已知 ground‑truth 的合成场景中能完全复现差异聚类；在真实模型（Stable Diffusion XL vs PixArt‑Σ、Llama 3.2 vs Gemma 3 等）中能精准定位导致输出差异的提示类别；随机投影版本在保持 3000 维特征时，运行时间仅为全核方法的十几倍，且性能差距不大。

**⚠️ 局限性**

局限性包括：仅支持两模型的 pairwise 对比；只捕捉二阶（协方差）信息，无法揭示更高阶的生成规律；依赖于提示分布的一致性；结果高度依赖嵌入质量；随机投影虽可控但仍存在投影维度与精度的权衡。

---

## 98. Causal Discovery for Cross-Sectional Data Based on Super-Structure and Divide-and-Conquer

**arXiv ID:** 2602.03914 | [PDF](https://arxiv.org/pdf/2602.03914v1)

**作者:** Wenyu Wang `[一作]` (University of South China), Yaping Wan `[通讯]` (University of South China)

**通讯引用:** 555 | [OpenAlex ID](https://openalex.org/A5035494769)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于弱约束超结构的分而治之因果发现框架，降低了条件独立检验成本。

**💡 创新点**

创新点在于将超结构从高召回转为高精确，通过弱约束结构实现轻量化分割，保持算法优势。

**🔧 技术方法**

使用Copula熵构建最大生成树超结构、Girvan图割进行分割、两阶段子图学习及Shah等方法进行合并。

**📊 数据集**

在合成高斯贝叶斯网络（MAGIC‑NIAB、ECOLI70、MAGIC‑IRRI）和中国健康养老纵向调查（CHARLS）数据上进行验证。

**📈 对比分析**

与PC/FCI等基线对比，结构准确度相近但CI检验次数下降5-30倍，性能优于传统方法。

**⚠️ 局限性**

局限在超结构不完整导致部分d‑separation被破坏，合并时仍需额外CI检验，且对极大网络的精度略逊。

---

## 99. VTok: A Unified Video Tokenizer with Decoupled Spatial-Temporal Latents

**arXiv ID:** 2602.04202 | [PDF](https://arxiv.org/pdf/2602.04202v1)

**作者:** Feng Wang `[一作]` (Bytedance Seed), Peng Wang `[通讯]` (Bytedance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出统一的视频 tokenizer VTok，能够在单一框架下同时支持视频理解与文本到视频的生成。

**💡 创新点**

创新点是将视频的空间信息与时间信息显式分离：保留关键帧的完整空间 token，随后用单一残差 token 捕捉每帧相对关键帧的运动变化，从而显著压缩 token 长度同时保持运动细节。

**🔧 技术方法**

采用自回归多模态大语言模型（MLLM）与预训练的 diffusion transformer 解码器，结合 CLIP 视觉编码器和残差提取网络，训练时仅更新 MLLM，冻结视觉编码器和解码器。

**📊 数据集**

使用约 500 万条视频-字幕对进行联合训练，并在 VBench、TV‑Align、Video‑MMM​U、MMVU、MVBench、LongVideoBench、LVBench 等标准基准上进行评估。

**📈 对比分析**

与多种基线（frame‑sampling、OmniTokenizer、Video‑LaVIT、HunyuanVideo 等）对比，VTok 在 TV‑Align 提升约 3% 的准确率，在 VBench 的语义分数提升 4.3%，视频理解任务平均提升约 2.4%，总体表现显著优于基线。

**⚠️ 局限性**

局限性包括：仅以首帧作为关键帧，可能无法捕捉多场景或剧烈变化；对极长视频的连续性和长期依赖尚未充分验证；模型仅冻结视觉编码器和解码器，可能限制在不同数据域的自适应能力。

---

## 100. C-IDS: Solving Contextual POMDP via Information-Directed Objective

**arXiv ID:** 2602.03939 | [PDF](https://arxiv.org/pdf/2602.03939v1)

**作者:** Chongyang Shi `[一作]` (University of Florida), Jie Fu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究在上下文未知、部分可观测环境（CPOMDP）中如何合成既能最大化累计回报又能主动降低对隐藏上下文不确定性的策略，提出信息导向目标并基于此设计C-IDS算法。

**💡 创新点**

创新点主要有：①将互信息与奖励耦合为信息导向目标，②将线性信息比率的拉格朗日松弛作为目标，③利用该形式证明C-IDS在多情节下具有子线性Bayesian regret上界；④通过变分策略梯度实现高效的策略学习。

**🔧 技术方法**

使用的技术包括信息理论（互信息、熵、信息比率）、Dinkelbach分数最小化法、Pinsker不等式、Azuma–Hoeffding不等式、变分策略梯度、LSTM策略网络以及扩展Kalman滤波实现对观测分布的近似。

**📊 数据集**

实验数据集为自定义的连续光暗环境（Light–Dark）仿真，状态为一维连续变量，动作空间为左右移动和主动观测，环境包含两种上下文，观测噪声在光区与暗区不同，模拟了环境的不可观测上下文。

**📈 对比分析**

与POMCP和RDPG‑RNN两种标准POMDP求解器比较。C-IDS在10,000轮训练后累计回报约10.6，信息熵约0.02；POMCP回报约-6.5、熵1.0；RDPG‑RNN回报约-3.6、熵0.16。C-IDS收敛速度最快、最终性能最优。

**⚠️ 局限性**

局限性包括：①仅在离散动作或低维连续状态空间上验证；②对高维图像、LiDAR等大规模感知输入的可扩展性尚未证明；③在多智能体或需要协作的上下文推断场景中尚未进行探索。

---

## 101. GeoIB: Geometry-Aware Information Bottleneck via Statistical-Manifold Compression

**arXiv ID:** 2602.03906 | [PDF](https://arxiv.org/pdf/2602.03906v1)

**作者:** Weiqi Wang `[一作]` (University of Technology Sydney), Shui Yu `[通讯]` (University of Technology Sydney)

**通讯引用:** 27902 | [OpenAlex ID](https://openalex.org/A5005228053)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于统计流形几何的几何信息瓶颈（GeoIB）方法，避免直接估计互信息，改用 Fisher–Rao 与 Jacobian–Frobenius 正则化实现压缩控制。

**💡 创新点**

创新点在于将信息瓶颈视为 KL 投影差异，并用 Fisher–Rao 二阶近似与 Jacobian–Frobenius 上界两种几何正则化相结合，同时采用自然梯度优化，使压缩过程更稳定、更可控。

**🔧 技术方法**

使用信息几何、Fisher–Rao 近似、Jacobian–Frobenius 上界、自然梯度（K‑FAC）等技术进行训练和优化。

**📊 数据集**

在 MNIST、CIFAR‑10 与 CelebA 三个公开图像数据集上进行实验。

**📈 对比分析**

与 VIB、MINE、SIB、AIB 四种主流 IB 基线对比，GeoIB 在准确率与信息压缩的 Pareto 曲线更优，兼具更高压缩率与更好的鲁棒性。

**⚠️ 局限性**

局限性包括训练时计算开销较大，Fisher–Rao 与 Jacobian–Frobenius 近似仅在局部有效，且对高维表示的压缩上界仍相对松散。

---

## 102. Axiomatic Foundations of Counterfactual Explanations

**arXiv ID:** 2602.04028 | [PDF](https://arxiv.org/pdf/2602.04028v1)

**作者:** Leila Amgoud `[一作]` (IRIT, CNRS), Martin Cooper `[通讯]` (IRIT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了一个基于公理的框架，对因果解释（counterfactuals）进行形式化定义并系统地划分五类解释器（GNR、SNR、GSR、SSR、CSR），并给出了公理不兼容性与表示定理；

**💡 创新点**

首次将必要理由与充分理由区分为五种子类型，并通过不可兼容性证明与表示定理构建了一套完整的理论分类体系；

**🔧 技术方法**

主要使用逻辑公理化、表示定理、不可兼容性证明以及SAT/复杂度分析等理论方法；

**📊 数据集**

本文未使用具体实验数据集，而是在抽象的分类理论和简易示例上进行演示，适用于任意可写的分类器（如布尔公式）；

**📈 对比分析**

通过对已有解释器的公理化表征，将其归入CSR族，证明它们满足相应公理；并给出了理论上限（如 O(n)、NP‑hard 等）来说明不同类型解释器的计算复杂度；

**⚠️ 局限性**

必要理由可能不存在；大多数类型计算复杂度高；缺乏实验评估；仅针对白盒分类器；对可操作性、最小化等实用属性的进一步研究仍待展开。

---

## 103. Monitorability as a Free Gift: How RLVR Spontaneously Aligns Reasoning

**arXiv ID:** 2602.03978 | [PDF](https://arxiv.org/pdf/2602.03978v1)

**作者:** Zidi Xiong `[一作]` (Harvard University), Himabindu Lakkaraju `[通讯]` (Harvard University)

**通讯引用:** 5800 | [OpenAlex ID](https://openalex.org/A5015520086)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了在RLVR训练过程中大规模推理模型（LRM）的监控性（monitorability）随时间演化的规律与机制。

**💡 创新点**

创新点在于揭示监控性提升并非单纯来自能力增强，而是高度依赖训练数据分布，特别是指令遵循数据，并进一步从熵降低与关注机制两方面解释监控性“免费礼物”现象。

**🔧 技术方法**

主要技术包括基于RLVR的强化学习（GRPO）、输入干预评估、草稿-答案因果一致性（D2A Faithfulness）评估以及跨段注意力分布分析。

**📊 数据集**

使用了多域训练集（数学、代码、科学、指令遵循）与多任务评测集（AMC12、Math500、AIME、CodeMMLU、HumanEval、GPQA、MMLU科学、ZebraLogic、MedQA等）。

**📈 对比分析**

与传统只关注任务准确度的RLVR相比，监控性指标在早期训练阶段显著提升，随后趋于平稳甚至下降；在指令遵循数据集上可达到最高监控性；然而提升不一定伴随性能提升，说明监控性与能力正交。

**⚠️ 局限性**

局限性包括：评估依赖于特定监测器（如Qwen2.5-32B-Instruct）与干预手段，可能对“假监控性”敏感；对更高复杂度任务的监控性提升有限；以及对训练长度与难度的泛化能力尚需进一步验证。

---

## 104. Generative Neural Operators through Diffusion Last Layer

**arXiv ID:** 2602.04139 | [PDF](https://arxiv.org/pdf/2602.04139v1)

**作者:** Sungwon Park `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8335 | [OpenAlex ID](https://openalex.org/A5008745801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

为任意确定性神经算子骨干加装一个扩散后端（DLL），将其转化为条件生成模型，用于随机场预测与不确定性量化。

**💡 创新点**

创新点在于①将输入相关的低秩卡尔曼-洛维展开作为算子编码器，压缩输出到低维系数；②在系数空间训练条件扩散模型，既保持了算子离散不变性，又显著降低了采样成本；③实现了在随机 PDE 与确定性混沌系统上同时兼顾分布拟合与长时刻自回归稳定性。

**🔧 技术方法**

使用的技术包括：神经算子（FNO）骨干；输入相关的算子编码器（基于 FNO 的基函数生成与系数提取）；条件扩散模型（velocity‑matching 训练）；低秩 Karhunen‑Loève 近似；以及对比基准（Deterministic, Bayesian, Pixel‑/Latent‑Diffusion, Autoencoder）。

**📊 数据集**

主要数据集：
- 随机 Burgers 方程（1D，N_x=256）
- 随机 Darcy 流（2D，128×128）
- 确定性 Kuramoto‑Sivashinsky 方程（1D，256点）
- 确定性 Kolmogorov 流（2D，128×128）

**📈 对比分析**

与 5 种基准比较（Deterministic, Bayesian, Latent‑Diffusion, Pixel‑Diffusion, Autoencoder）。在随机 PDE 上，DLL 在 SWD、ED、NRMSE_m、NRMSE_s 上均优于确定性/贝叶斯模型，且与像素/潜在扩散模型相当或更好；在自回归滚动测试中，DLL 在 NRMSE、CRPS、SSR 指标上超过大多数基准，尤其在 KS 方程中接近 1 的 SSR 体现良好校准；在重构任务中，DLL 的低秩编码比 Autoencoder 取得更低的 NRMSE。

**⚠️ 局限性**

局限性包括：
- 依赖低秩卡尔曼‑洛维展开，若输出方差分布在更高维空间难以压缩则效果下降；
- 目前仅在规则网格上验证，尚未在不规则几何或高维空间推广；
- 采样仍需多步迭代，计算成本高于单次前向推理；
- 结果的置信度校准仍需进一步理论和实验验证。

---

## 105. After Talking with 1,000 Personas: Learning Preference-Aligned Proactive Assistants From Large-Scale Persona Interactions

**arXiv ID:** 2602.04000 | [PDF](https://arxiv.org/pdf/2602.04000v1)

**作者:** Ziyi Xuan `[一作]` (Lehigh University), Yu Yang `[通讯]` (Lehigh University)

**通讯引用:** 1793 | [OpenAlex ID](https://openalex.org/A5070570645)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种两阶段的主动助手学习框架：先在大规模由 LLM 生成的合成用户数据上进行人口层面的偏好学习，再通过在设备端的激活向量调节实现对单个用户的个性化适配。

**💡 创新点**

创新点包括：①将 LLM 驱动的用户模拟作为训练数据源，突破了传统实测数据受限的瓶颈；②设计了五维结构化偏好 schema，兼顾时机、自治、沟通风格等关键维度；③在不更新模型参数的前提下，通过激活向量 steering 实现轻量级、可逆的个性化。

**🔧 技术方法**

主要技术包括：使用 Llama‑3.2‑3B 基础模型与 LoRA 微调；GIDEA 平台 + GPT‑4.1 生成 1,000 名合成用户的多轮交互；多任务监督训练（响应生成 + 偏好维度预测）；激活向量 steering 在推理时对内部激活施加方向性偏移；评估时采用 CAS/PSC/TAI/IQA 等指标。

**📊 数据集**

使用的数据集为：基于美国人口普查构建的 1,000 名合成 persona（每人一周日程）通过 GIDEA + GPT‑4.1 生成的多轮交互数据，并按五维偏好进行标注；该数据集兼具时间深度、情境多样性与结构化偏好信息。

**📈 对比分析**

与基线（未调优模型、直接响应训练、基于上下文学习、RLHF）相比，分类结构化训练 + 激活 steering 在模拟评估中 CAS/PSC/TI/ IQA 分别提升 68/59/83/91%；在人类受试者的 30 人实验中，适配助手的整体满意度提高约 14%，并在感知适时性与信任度上显著优于静态模型，激活 steering 与 RLHF 的表现相近。

**⚠️ 局限性**

局限性包括：①合成数据虽然覆盖多样性，但仍可能缺乏真实用户的细腻情感与习惯差异；②人类实验规模有限且以图板为主，缺乏长周期真实使用验证；③激活向量调节在极端偏好场景下可能收敛慢；④模型仍以 3B 参数为主，若进一步压缩可能影响偏好捕捉效果。

---

## 106. DMS2F-HAD: A Dual-branch Mamba-based Spatial-Spectral Fusion Network for Hyperspectral Anomaly Detection

**arXiv ID:** 2602.04102 | [PDF](https://arxiv.org/pdf/2602.04102v1)

**作者:** Aayushma Pant `[一作]` (Deakin University), Sunil Aryal `[通讯]` (Deakin University)

**通讯引用:** 2257 | [OpenAlex ID](https://openalex.org/A5038741954)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种名为DMS2F-HAD的双分支Mamba自编码网络，用于高光谱图像异常检测。

**💡 创新点**

创新点包括：①将空间和光谱特征分别通过独立的Mamba分支学习；②采用线性时间复杂度的Mamba实现长程依赖建模；③引入自适应门控融合机制动态平衡空间与光谱特征；④整体模型参数量和 FLOPs 大幅下降，实现高效实时检测。

**🔧 技术方法**

使用了Mamba状态空间模型、光谱分组策略、空间多尺度特征提取、1×1卷积门控融合、重构误差检测等技术。

**📊 数据集**

在十四个常用高光谱基准数据集上进行实验，包括 CRI、Salinas、San Diego、Bay Champagne、Abu-urban 系列、AVIRIS‑1/2、Cat Island、Gulfport、Pavia、Texas Coast 等。

**📈 对比分析**

与传统统计方法（RX）以及多种深度学习方法（AUTO‑AD、TDD、GT‑HAD、RGAE、LREN）进行对比，平均 AUC 达 98.78%，超过 GT‑HAD（97.74%）并与 MMR‑HAD 取得相当准确度，参数量仅为其 0.64M（相比 2.12M）且 FLOPs 仅 0.12G，推理时间平均 0.55 秒，比最快的 TDD 快 4.6 倍。

**⚠️ 局限性**

局限性包括：仅基于重构误差的异常度量，可能对光谱相近但空间特征不同的目标产生误检；实验仅在 AVIRIS 及 Nuance CRI 数据集上验证，尚未在更大尺寸或多源传感器上全面测试；门控机制在极端样本不平衡时的鲁棒性仍待进一步评估。

---

## 107. Comparative Analysis of Autonomous Robotic and Manual Techniques for Ultrasonic Sacral Osteotomy: A Preliminary Study

**arXiv ID:** 2602.04076 | [PDF](https://arxiv.org/pdf/2602.04076v1)

**作者:** Daniyal Maroufi `[一作]` (University of Texas at Austin), Farshid Alambeigi `[通讯]` (University of Texas at Austin)

**通讯引用:** 1547 | [OpenAlex ID](https://openalex.org/A5055294307)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文设计并验证了一套自主超声骶骨切除（RUSO）机器人系统，旨在实现与手动超声切除（MUSO）相比的更高轨迹精度、深度控制和手术效率。

**💡 创新点**

创新点在于：①将超声骨刀与7自由度KUKA LBR Med 14机械臂和光学跟踪系统集成，实现全自主切除；②采用手眼标定、枢轴标定和刀尖标定实现刀尖的精确定位；③通过量化实验展示机器人可达亚毫米轨迹精度、<5%深度误差，并大幅缩短手术时间。

**🔧 技术方法**

使用技术包括：7自由度机械臂（KUKA LBR Med 14）、超声骨刀（BoneScalpel）、NDI Polaris Vega光学跟踪、手眼标定（AX=XB方法）、枢轴标定、刀尖标定、实验室级Sawbones硬质聚氨酯块、数字量具和计算机算法（RMSE、深度剖面提取）。

**📊 数据集**

数据集为实验室制备的Sawbones硬质聚氨酯块（PCF 15），用来模拟骶骨硬组织，并在相同的切除任务（100 mm直线、4 mm和8 mm深度）下进行多次实验。

**📈 对比分析**

比较方法：对每种方法（MUSO、RUSO）在同一实验条件下进行4次（MUSO）或3次（RUSO）试验，记录轨迹点、执行长度、手术时间和最终切深。评估指标为轨迹RMSE、实际切深误差、手术时间。性能：RUSO轨迹RMSE 0.11 mm（比MUSO的1.1 mm低10倍），深度误差<5%（MUSO 75–100%过深），手术时间从111 s缩短到45 s。

**⚠️ 局限性**

局限性包括：①仅验证了直线切除路径，缺乏对复杂非线性轨迹的评估；②实验仅在人工制备的Sawbones模型上进行，未涉及真实骨骼和临床解剖变异；③系统对光学跟踪环境依赖强，对遮挡和光线干扰敏感；④未评估手术过程中的血管、神经损伤风险及长期临床可行性。

---

## 108. StraTyper: Automated Semantic Type Discovery and Multi-Type Annotation for Dataset Collections

**arXiv ID:** 2602.04004 | [PDF](https://arxiv.org/pdf/2602.04004v1)

**作者:** Christos Koutras `[一作]` (New York University), Juliana Freire `[通讯]` (New York University)

**通讯引用:** 11966 | [OpenAlex ID](https://openalex.org/A5006773757)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了 StraTyper 框架，实现无预定义标签的列语义类型自动发现和多类型注释。

**💡 创新点**

通过双重聚类、长度分层采样、动态类型检索和迭代级联发现，低成本高精度完成多类型识别，并解决 LLM 重复生成与单类型限制。

**🔧 技术方法**

结合 SBERT 语义嵌入、快速社区检测、统计直方图、倒排索引、Chain‑of‑Thought 结构化提示以及 Gemini 2.5 Flash 与 GPT‑OSS 等大语言模型。

**📊 数据集**

在 NYC OpenData 的 NYC‑E 与 NYC‑CG 两个域特定集合以及 Freyja（OpenML/Kaggle）混合数据集上进行实验。

**📈 对比分析**

与 LLM‑only 基线（含类型重用）及公开 CTA 方法比较，StraTyper 既将 LLM 成本降低约 2.5 倍，又在 F1 上达到 0.83‑0.86，显著提升 join 发现和 schema 匹配准确率（约 63%/47%）。

**⚠️ 局限性**

仍可能忽略极少见的长尾类型，对高度多值列的精细分层存在提升空间，并依赖商业 LLM 进行发现阶段。

---

## 109. Towards X-embodiment safety: A control theory perspective on transferring safety certificates across dynamical systems

**arXiv ID:** 2602.03987 | [PDF](https://arxiv.org/pdf/2602.03987v1)

**作者:** Nikolaos Bousias `[一作]` (University of Pennsylvania), George Pappas `[通讯]` (University of Pennsylvania)

**通讯引用:** 35270 | [OpenAlex ID](https://openalex.org/A5029243115)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出一种将控制边界函数（CBF）从抽象模型迁移到具体系统的框架，并通过仿真函数和安全裕度实现对高维四旋翼的安全约束，随后在模拟环境中验证了其在避障任务中的有效性。

**💡 创新点**

创新点包括：1）利用仿真函数与见证图将抽象系统的安全边界迁移至不同维度的具体系统；2）给出显式安全裕度函数的设计条件，保证安全约束在具体系统上成立；3）实现了不需要状态维度一致即可完成安全传递，提升了安全分析与实现的通用性。

**🔧 技术方法**

采用的主要技术手段有：控制边界函数（CBF）理论、仿真函数与见证图、偏微分方程分析、二次规划（QP）安全滤波、四旋翼动力学建模、RRT*路径规划、最小抖动轨迹优化以及几何控制。

**📊 数据集**

实验使用在模拟三维环境中人工构造的球形障碍物集合，未使用公开数据集；通过该仿真环境评估四旋翼在障碍密集区域的安全性。

**📈 对比分析**

与未加安全滤波的基准轨迹进行对比，实验结果显示：安全滤波仅在接近障碍时做出局部偏移，保持正的几何安全间隙并成功避免碰撞；同时与抽象模型的纯安全约束对比，验证了安全裕度设计能够在具体系统上保持理论保证，性能相当优良。

**⚠️ 局限性**

局限性包括：1）需要人工构造仿真函数和接口，难以直接应用于极其复杂或未知动力学；2）安全裕度设计可能导致一定的保守性，影响运动效率；3）对模型不匹配的鲁棒性受限，未来需考虑可学习的仿真函数与接口。

---

## 110. A Parameterizable Convolution Accelerator for Embedded Deep Learning Applications

**arXiv ID:** 2602.04044 | [PDF](https://arxiv.org/pdf/2602.04044v1)

**作者:** Panagiotis Mousouliotis `[一作]` (Aristotle University of Thessaloniki), Georgios Keramidas `[通讯]` (Aristotle University of Thessaloniki)

**通讯引用:** 1122 | [OpenAlex ID](https://openalex.org/A5074371053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种可参数化的卷积FPGA加速器模板，采用HLS工具实现并结合HW/SW协同设计，专门针对嵌入式深度学习场景。

**💡 创新点**

创新点在于通过设计参数化模板，使加速器能够无需重新配置即可适配多种CNN模型，并利用8位动态定点量化、双缓冲、数据流和空间并行化技术，显著提升性能与能效比。

**🔧 技术方法**

使用了高层综合(HLS)技术、动态定点(DFP)量化、流水线与数据流优化、双缓冲、资源分区(parallelism)以及SoC FPGA与宿主CPU的协同工作。

**📊 数据集**

在ImageNet 2012验证集上评估了SqueezeNet v1.1、ZynqNet、PeleeNet和VGG‑16四个网络。

**📈 对比分析**

与Angel‑Eye、fpgaConvNet和Ma等工作相比，在8位精度下，该加速器在CONV层延迟上实现了约30%~40%的加速，功耗低于4.3W，同时保持与浮点模型相近的准确率。

**⚠️ 局限性**

局限性包括需要针对不同CNN手动调整参数、仅支持卷积/池化与激活层、对低端SoC FPGA资源约束敏感、并未覆盖全连接层或更大模型的加速需求。

---

## 111. HybridQuestion: Human-AI Collaboration for Identifying High-Impact Research Questions

**arXiv ID:** 2602.03849 | [PDF](https://arxiv.org/pdf/2602.03849v1)

**作者:** Keyu Zhao `[一作]` (Tsinghua University), Tie-Yan Liu `[通讯]` (Zhongguancun Academy)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种人机混合框架，用 AI 自动化收集与合成科研信息，并通过多阶段投票筛选2025年重大突破与2026年重要科学问题的Top 10列表。

**💡 创新点**

将大型语言模型（LLM）与人类专家协同评估结合，首次在战略层面使用 LLM 生成并过滤候选问题，形成可操作的决策流程。

**🔧 技术方法**

使用图嵌入(node2vec)生成关键词热度、热度优先聚类，LLM集成（多模型投票）生成候选列表，以及多轮投票算法（批准投票、限额投票）进行筛选。

**📊 数据集**

主要使用 OpenAlex 论文元数据（关键词、引文）及跨国网络深度检索得到的媒体与行业报告作为输入。

**📈 对比分析**

对比 AI 与人类专家在两轮投票中的投票分布，计算 Jensen–Shannon 距离验证 AI 与专家的一致性；结果显示 AI 与专家在识别已完成突破时高度一致，但在预测未来问题时差距较大。

**⚠️ 局限性**

受限于 LLM 对前瞻性、价值判断的偏差，无法完全替代人类对主观、结构性问题的评估；系统在跨学科、跨语言的一致性与可解释性上仍有改进空间。

---

## 112. A categorical framework for cellular automata

**arXiv ID:** 2602.04049 | [PDF](https://arxiv.org/pdf/2602.04049v1)

**作者:** A. Castillo-Ramirez `[一作]` (Universidad de Guadalajara), A. Zaldivar-Corichi `[通讯]` (Universidad de Guadalajara)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5004997670)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出了一个基于范畴论的通用细胞自动机框架，将字母表从集合推广到任意具有乘积的范畴，并进一步定义了不同群宇宙间的广义细胞自动机。

**💡 创新点**

创新点在于：①以范畴论的语言完整刻画细胞自动机及其拓展；②证明了范畴化的Curtis‑Hedlund‑Lyndon定理；③展示了广义细胞自动机构成子范畴，并构造了弱积（含自由积群）。

**🔧 技术方法**

核心技术主要是：范畴论的乘积与余积的通用性质、函子与对偶函子（推前/拉回）以及对称性和可积性的证明。

**📊 数据集**

本文为理论性研究，没有使用具体数据集；所有结果均通过范畴论的公理与证明得到。

**📈 对比分析**

由于是纯理论工作，没有实验比较；但通过范畴论证明展示了传统细胞自动机理论的完整性与统一性。

**⚠️ 局限性**

局限性包括：广义细胞自动机的弱积并非唯一或真正的积（取决于是否存在唯一分解），且对某些常数映射不满足唯一性；此外，结果在非具体现实应用中的可解释性尚待进一步探讨。

---

## 113. Understanding How Accessibility Practices Impact Teamwork in Mixed-Ability Teams that Collaborate Virtually

**arXiv ID:** 2602.04015 | [PDF](https://arxiv.org/pdf/2602.04015v1)

**作者:** Crescentia Jung `[一作]` (Cornell University), Shiri Azenkot `[通讯]` (Cornell Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过半结构化访谈18名混合能力团队成员，研究虚拟协作环境中可访问性实践对团队生产力、参与度和相处的影响。

**💡 创新点**

首次将可访问性实践视为塑造团队合作机制的核心，揭示其对情感张力、责任分配和同盟关系的深层作用，并提出团队实践与工具设计建议。

**🔧 技术方法**

未采用算法或软件技术，而是基于访谈文本进行开放编码与主题分析。

**📊 数据集**

数据来源为18名受访者（12名残障成员、6名非残障成员），涵盖教育、非营利、技术与研究等不同职业背景。

**📈 对比分析**

研究未进行定量对比或性能评估，仅提供质性发现与案例引述。

**⚠️ 局限性**

样本量有限，残障成员比例偏高；依赖受访者回忆，缺乏观察或纵向数据；研究范围仅限于访谈，未检验设计建议的实效性。

---

## 114. VLS: Steering Pretrained Robot Policies via Vision-Language Models

**arXiv ID:** 2602.03973 | [PDF](https://arxiv.org/pdf/2602.03973v1)

**作者:** Shuo Liu `[一作]` (University of Washington), Ranjay Krishna `[通讯]` (University of Washington)

**通讯引用:** 12789 | [OpenAlex ID](https://openalex.org/A5032451496)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Vision–Language Steering（VLS），通过可微奖励引导冻结的扩散/流匹配机器人策略，在不需要再训练的情况下实现对测试时 OOD 场景的自适应；

**💡 创新点**

创新点在于将 VLM 生成的可微奖励直接注入生成式策略的 denoising 过程，并结合梯度引导、Feynman–Kac 重采样、RBF 多样性初始化以及闭环阶段切换与自适应指导强度，形成完整的推理时自适应框架；

**🔧 技术方法**

使用的技术包括扩散/流匹配生成式策略、Vision–Language 模型（SAM、DINOv2、VLM）、可微奖励函数、梯度引导、Feynman–Kac 重采样、RBF 互斥、闭环反馈与斯密特触发；

**📊 数据集**

在 CALVIN、LIBERO‑PRO 以及真实 Franka 机器人实验中进行验证；

**📈 对比分析**

与 OpenVLA、π_0、π_0.5、DynaGuide、ITPS 等基线对比，VLS 在 CALVIN 可移动物体 94% 与关节部件 87% 的成功率、LIBERO‑PRO 成功率提升 13%、Franka 任务成功率提升 19% 以上，明显优于其他方法；

**⚠️ 局限性**

推理时计算开销较高（批量采样、MCMC、重采样），需进一步优化实时性能。

---

## 115. From Lemmas to Dependencies: What Signals Drive Light Verbs Classification?

**arXiv ID:** 2602.04127 | [PDF](https://arxiv.org/pdf/2602.04127v1)

**作者:** Sercan Karakaş `[一作]` (University of Chicago), Yusuf Şimşek `[通讯]` (Firat University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个针对土耳其语轻动词构造（LVC）的受控诊断测试集，并通过系统限制模型输入（词干、语法信息或完整句子），比较了不同表征方式对LVC判别的影响。

**💡 创新点**

创新点在于把词干化、语法化与完整输入作为连续的实验条件，揭示了词干化的语义信息与粗粒度语法特征在LVC识别中的作用，并发现词干化模型对归一化方式极度敏感，表明“词干化”并非单一设置。

**🔧 技术方法**

使用了传统的TF–IDF+逻辑回归、基于词干的BERTurk上下文编码器、仅使用UPOS/DEPREL/MORPH特征的逻辑回归，以及全句子输入的BERTurk微调等技术；并通过精确度、召回率、F1等指标进行评估。

**📊 数据集**

数据来源于九个土耳其语Universal Dependencies（UD）树库，用于生成弱监督标签；诊断集为147条人工构造、平衡三类（随机负、非LVC控制、LVC正）句子；评估时自动使用Stanza进行UD标注。

**📈 对比分析**

比较方法是在诊断集上按随机、NLVC、LVC三类分别报告结果。结果显示：全句BERTurk在所有类别上准确率≈94%，词干化逻辑回归在随机类几乎100%但在LVC类召回率低；语法化逻辑回归在随机和NLVC类表现良好，却在LVC类几乎失效；词干化BERTurk在表面测试上接近全句模型，但在词干化测试时LVC召回率显著下降，体现了归一化分布漂移。

**⚠️ 局限性**

局限性包括：监督信号仅为UD中的名词–动词依赖，可能受树库标注习惯限制；诊断集规模小，主要体现决策边界而非整体性能；语法化评估依赖Stanza，若解析错误会影响结果；词干化模型对归一化方式高度敏感，可能掩盖真正的模型能力。

---

## 116. PriorProbe: Recovering Individual-Level Priors for Personalizing Neural Networks in Facial Expression Recognition

**arXiv ID:** 2602.03882 | [PDF](https://arxiv.org/pdf/2602.03882v1)

**作者:** Haijiang Yan `[一作]` (University of Warwick), Adam Sanborn `[通讯]` (University of Warwick)

**通讯引用:** 3017 | [OpenAlex ID](https://openalex.org/A5025964235)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了PriorProbe方法，利用块Metropolis–Hastings与人类参与，在面部表情识别任务中恢复个体先验，并将其直接注入神经网络以提升个体化预测。

**💡 创新点**

创新点包括：①将先验与似然共同恢复的块MCMC框架；②设计DistFace空间，将情感与身份解耦；③无需改动网络权重，而是将个体先验嵌入推理空间。

**🔧 技术方法**

采用Metropolis–Hastings、Independent Metropolis–Hastings、ResEmoteNet 作为Gatekeeper、VAE+MAE、ArcFace等预训练编码器，以及低维稀疏空间进行采样与重构。

**📊 数据集**

使用BU‑4DFE训练DistFace、CelebA做held‑out测试、RAF‑DB+FER2013构成生态先验、ResEmoteNet基于FER2013训练。

**📈 对比分析**

与单纯ResEmoteNet、生态先验、平均先验及仅先验模型对比，PriorProbe恢复的个体先验+ResEmoteNet准确率从0.239提升至0.371；在55名受试者中有49名获得提升，置信度相关性从0.119提升至0.167，且在ground‑truth数据集上无明显性能下降。

**⚠️ 局限性**

限制在于需要大量实验步骤（每人1000次），依赖人类参与；仅验证七种基本情绪且仅在表情识别任务中验证，未验证跨领域普适性；先验恢复对Gatekeeper模型质量敏感。

---

## 117. From Helpfulness to Toxic Proactivity: Diagnosing Behavioral Misalignment in LLM Agents

**arXiv ID:** 2602.04197 | [PDF](https://arxiv.org/pdf/2602.04197v1)

**作者:** Xinyue Wang `[一作]` (Beijing University of Posts and Telecommunications), Sen Su `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 4761 | [OpenAlex ID](https://openalex.org/A5036865453)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了LLM代理的“毒性主动性”风险，并构建了基于对抗式情景合成与多轮 Agent‑Environment 交互的评估框架。

**💡 创新点**

创新点在于双轨行动空间（合规与有毒子集）与自我改进的情景生成流程，以及通过多轮模拟捕捉代理的长期行为轨迹。

**🔧 技术方法**

主要技术包括自我改进叙事设计、证据生成、对抗式双轨行动构造、以及多轮交互仿真与安全阈值评估。

**📊 数据集**

使用了人工合成的高风险情境数据集，覆盖代码、医疗、网络安全与金融四个领域，并以 Gemini‑3‑Flash 生成证据与环境状态。

**📈 对比分析**

通过在 10 种主流 SOTA LLM 上进行 400 场实验，计算 Misalignment Rate；结果显示大多数模型误差率>65%，最高达 98%，且模型规模提升并未降低总体误差。

**⚠️ 局限性**

局限性包括仅在合成情境下评估，缺少真实环境验证；对抗式生成可能引入偏差；未深入探讨对策与长期训练对毒性主动性影响。

---

## 118. Beyond the Vehicle: Cooperative Localization by Fusing Point Clouds for GPS-Challenged Urban Scenarios

**arXiv ID:** 2602.03908 | [PDF](https://arxiv.org/pdf/2602.03908v1)

**作者:** Kuo-Yi Chao `[一作]` (Technical University of Munich), Alois Christian Knoll `[通讯]` (Technical University of Munich)

**通讯引用:** 24286 | [OpenAlex ID](https://openalex.org/A5063781430)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种协作多模态定位框架，通过融合车辆与基础设施的点云信息，对 GPS 信号在城市环境中的漂移进行实时校正。

**💡 创新点**

创新点在于将基础设施（交叉口 LiDAR）与车辆 LiDAR、相机生成的点云进行联合注册，并在 ICP‑RANSAC 注册的基础上与 SLAM 结果融合，实现毫米级定位精度。

**🔧 技术方法**

核心技术包括：V2X 通信、LiDAR 与双目相机点云生成、FPFH 描述子、ICP‑RANSAC 注册、在线 SLAM（如 PIN‑SLAM）以及后处理融合。

**📊 数据集**

实验采用 CARLA 仿真平台，搭建四种交叉口场景，使用模拟的 LiDAR、相机以及四个交叉口传感器与一至两辆协同车辆的数据。

**📈 对比分析**

与单独 GPS、单独 SLAM 以及仅注册的方法相比，融合方案在所有四个仿真场景中将定位误差从 7–17 m 降至 0.009–0.028 m（有效帧），显著提升了鲁棒性与精度。

**⚠️ 局限性**

局限性包括：对传感器同步与时间戳误差敏感；ICP‑RANSAC 需要较好初始估计，易受遮挡和噪声影响；在大规模交通场景下的可扩展性和实时性能仍需进一步验证。

---

## 119. SOGPTSpotter: Detecting ChatGPT-Generated Answers on Stack Overflow

**arXiv ID:** 2602.04185 | [PDF](https://arxiv.org/pdf/2602.04185v1)

**作者:** Suyu Ma `[一作]` (CSIRO Data61), John Grundy `[通讯]` (Monash University)

**通讯引用:** 16029 | [OpenAlex ID](https://openalex.org/A5082913979)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于BigBird的Siamese神经网络与三元组损失的模型，用来检测Stack Overflow上的ChatGPT生成答案

**💡 创新点**

①首次利用参考答案（由ChatGPT生成的具有非人类特征）与问题文本对齐，构建三元组；②将BigBird的稀疏注意力机制引入Siamese网络，提升对长文本的处理能力；③在三元组上使用Triplet Loss，强化正负样本间的语义差距；

**🔧 技术方法**

Siamese网络、BigBird Transformer、Triplet Loss、Cosine相似度判别、PyTorch实现、Adam优化器

**📊 数据集**

从Stack Overflow公开数据集抽取6000条高质量问答，使用GPT‑4 Turbo生成参考答案和ChatGPT答案，构成（参考答案、人工答案、ChatGPT答案）三元组；另外收集不同领域（Mathematics、Electronics、Bitcoin）以及不同LLM（Claude3、LLaMA3、Gemini）测试集

**📈 对比分析**

与GPTZero、DetectGPT、GLTR、BERT、RoBERTa、GPT‑2等基线模型在同一测试集上对比，精度、召回率、F1分数均超过所有基线；在长度、对抗攻击、跨领域、跨LLM以及真实生产环境（Stack Overflow moderators）中均保持高达≈94‑97%的准确率

**⚠️ 局限性**

①模型对极短答案和仅含代码片段的答案识别困难；②需要实时更新参考答案生成策略以跟上LLM进化；③训练依赖于手工挑选的高质量人工答案，可能引入偏差；④在不同Q&A平台的泛化性尚待验证

---

## 120. The Illusion of Generalization: Re-examining Tabular Language Model Evaluation

**arXiv ID:** 2602.04031 | [PDF](https://arxiv.org/pdf/2602.04031v1)

**作者:** Aditya Gorla `[一作]` (University of California), Ratish Puduppully `[通讯]` (IT University of Copenhagen)

**通讯引用:** 366 | [OpenAlex ID](https://openalex.org/A5069271622)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对 Tabula-8B 进行系统性重评，发现其在 165 个 UniPredict 数据集上的高准确率主要由任务类型偏差、数据泄漏以及仅靠指令微调获得，而非真正的表格推理能力。

**💡 创新点**

创新点在于首次通过完整的基线对比、任务类型细分、三种泄漏类型分析以及提出七条可操作的评估改进建议，揭示了 TLM 评估中的系统性偏误。

**🔧 技术方法**

采用指令微调、基线比较（Majority‑class、Base Llama‑3‑8B）、Cohen’s κ 评估、以及对 T4 训练语料的行级与实体级匹配分析。

**📊 数据集**

使用 UniPredict 子集（165 个 Kaggle 数据集）和 T4 预训练语料库（约 400 万张表格）进行评估。

**📈 对比分析**

与 Base Llama、Alpaca（无表格暴露的指令微调模型）以及 Alpaca+Q（仅四分位格式微调）对比，发现指令微调贡献约 69% 的提升，残留差距主要来源于金融股票类数据的直接泄漏，整体性能被四分位回归任务所拉高。

**⚠️ 局限性**

局限性包括：仅评估分类任务，未覆盖 Imputation、生成等功能；仅针对 UniPredict 子集；泄漏检测未覆盖所有可能模式；聚焦于 Tabula-8B 与 T4，其他 TLM 与语料库的结果可能不同。

---

## 121. Principles of Lipschitz continuity in neural networks

**arXiv ID:** 2602.04078 | [PDF](https://arxiv.org/pdf/2602.04078v1)

**作者:** Róisín Luo `[一作]` `[通讯]`, Róisín Luo

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

未提供论文主体内容，无法得知具体研究内容。

**💡 创新点**

无法确定创新点。

**🔧 技术方法**

无法确定使用的技术。

**📊 数据集**

无法确定使用的数据集。

**📈 对比分析**

无法确定比较方法和性能评估。

**⚠️ 局限性**

无法确定论文的局限性。

---

## 122. DeXposure-FM: A Time-series, Graph Foundation Model for Credit Exposures and Stability on Decentralized Financial Networks

**arXiv ID:** 2602.03981 | [PDF](https://arxiv.org/pdf/2602.03981v1)

**作者:** Aijie Shu `[一作]` (Independent Scholar), Fengxiang He `[通讯]` (University of Edinburgh)

**通讯引用:** 1732 | [OpenAlex ID](https://openalex.org/A5100635369)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了首个时间序列图形基础模型DeXposure‑FM，用于测量和预测去中心化金融网络中的协议间信用暴露。

**💡 创新点**

创新点：①将图形-表格编码器GraphPFN预训练权重迁移到DeFi信用暴露任务，形成可复用的基础模型；②同时预测边的存在与权重以及协议层面的TVL变动，实现多任务联合学习；③通过预测‑测量管道将模型输出转化为宏观监管指标（系统重要性、跨行业溢出、压力测试结果）。

**🔧 技术方法**

技术：GraphPFN图形-表格编码器（Transformer + 图结构编码）；多任务头（二分类、回归、回归）；Adam优化、梯度裁剪、时间步走前拆分、早停；债务排名式传播模拟器用于压力测试。

**📊 数据集**

数据集：DeXposure 数据集，43.7 M 条每周协议间曝光记录，覆盖 4,300+ 协议、602 条区块链、24,300+ 代币，构造了加权有向多重图。

**📈 对比分析**

比较方法：在两项机器学习基准上评估——多步预测（边存在、边权重、协议TVL变化）和预测性传染压力测试；与 GraphPFN（冻结）、ROLAND（从零训练）以及持久性基线对比。DeXposure‑FM 在边存在 AUROC>0.99、边权重 MAE≈2.5、TVL MAE≈0.06，显著优于其他模型；在压力测试中，在总体表现上持久性占优，但在“最坏20%”尾部场景中，DeXposure‑FM 的 ΔMAE 为正，胜率 83–100%。

**⚠️ 局限性**

局限性：①仅覆盖链上 DeFi 协议，缺少 CEX、OTC、中心化稳定币资产；②以周级快照为基础，无法捕捉高频事件（闪电贷、即时清算）；③TVL 作为暴露代理未考虑资产质量、流动性和风险权重；④跨链数据同步与价格来源可能产生噪声；⑤模型随 DeFi 生态快速演化可能出现漂移，需要周期性重训练。

---

## 123. Abstraction Induces the Brain Alignment of Language and Speech Models

**arXiv ID:** 2602.04081 | [PDF](https://arxiv.org/pdf/2602.04081v1)

**作者:** Emily Cheng `[一作]` (Universitat Pompeu Fabra), Richard Antonello `[通讯]` (Zuckerman Mind Brain Behavior Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究通过分析大型语言模型和语音模型中间层的特征维度，证明抽象语义层的特征丰富性是驱动脑-模型相似性的关键。

**💡 创新点**

创新点在于将内部维度（Intrinsic Dimension）与大脑编码性能关联，揭示中间层的高维特征构建是脑-模型对应的根本原因，而非下一词预测。

**🔧 技术方法**

使用Transformer语言模型（OPT、Pythia）和语音模型（WavLM、Whisper）以及Intrinsic Dimension估计（GRIDE）、线性编码模型和层级线性探测。

**📊 数据集**

数据集包括公开的fMRI（UTS02、UTS03）和ECoG（Podcast）数据，以及模型训练数据The Pile和LibriSpeech，用于计算I_d。

**📈 对比分析**

通过将模型层特征与大脑信号做线性映射，并与随机Fourier特征对比，发现I_d峰层对应最高的编码相关性；在大多数模型中，I_d与编码性能的相关系数高达0.76（fMRI）和0.43（ECoG），优于预测错误（surprisal）。

**⚠️ 局限性**

限制在于仅使用线性映射评估，未深入探讨非线性关系；随机特征表明高I_d并非因果；实验受限于少数受试者与模型规模。

---

## 124. Tinker Tales: Supporting Child-AI Collaboration through Co-Creative Storytelling with Educational Scaffolding

**arXiv ID:** 2602.04109 | [PDF](https://arxiv.org/pdf/2602.04109v1)

**作者:** Nayoung Choi `[一作]` (Emory University), Jinho D. Choi `[通讯]` (Emory University)

**通讯引用:** 2687 | [OpenAlex ID](https://openalex.org/A5101829031)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一款名为Tinker Tales的可触式共创叙事系统，让儿童通过物理棋盘、NFC玩具和语音交互与AI对话，共同创作故事。

**💡 创新点**

将可触物理交互与对话式AI结合，并采用Applebee叙事发展模型与CASEL社会情感学习框架来构造AI的教育式引导，支持儿童在保持主体性的同时进行协作式叙事。

**🔧 技术方法**

NFC读写、语音识别（STT）、语音合成（TTS）、基于Amazon Bedrock的Claude Haiku 4.5对话模型，Flutter移动端，AWS Lambda、DynamoDB。

**📊 数据集**

未使用公开数据集，而是通过10名6‑8岁儿童的在家共创故事会话日志和人工标注进行评估。

**📈 对比分析**

通过对比两种AI提问方式（结构化教育式与通用提问）在儿童叙事贡献、AI响应采纳率等指标上的差异，结果显示结构化提问显著提升儿童的叙事贡献和AI响应的采纳率。

**⚠️ 局限性**

样本量小（10名儿童）；仅在英语环境下进行；缺乏音频/视频记录，无法观察非语言行为；研究周期短，无法评估长期使用效果。

---

## 125. Group Contrastive Learning for Weakly Paired Multimodal Data

**arXiv ID:** 2602.04021 | [PDF](https://arxiv.org/pdf/2602.04021v1)

**作者:** Aditya Gorla `[一作]` (Genentech), Russell Littman `[通讯]` (Genentech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种针对弱配对多模态单细胞扰动数据的半监督表示学习方法，结合组级对比损失与即时反向翻译自编码器。

**💡 创新点**

创新点在于提出组级半监督对比损失（GroupCLIP）以桥接 CLIP 与 SupCon，并构建组合评估框架，解决弱配对场景下的多模态表征问题。

**🔧 技术方法**

主要技术包括 GroupCLIP 损失、基于神经机翻的 on‑the‑fly 反向翻译自编码器、多种 OT 对齐器（LabeledEOT、LabeledEGWOT、LabeledCOOT）以及模拟实验与 KNN 评价指标。

**📊 数据集**

使用的数据集包括模拟数据（不同共享比例）、Perturb‑Multiome（转录组与染色质可及性）和 Perturb‑CITE‑seq（RNA‑seq 与表面蛋白），并在 5‑fold 与 LOPO 验证中测试。

**📈 对比分析**

与 PS、DAVAE 等基线及多种 OT 对齐器进行组合评估，GroupCLIP 在匹配（trace、Bary.FOSCTTM）和下游插补（MSE、Cos‑sim、KNN 指标）上均名列前茅，尤其在真实数据集表现稳健。

**⚠️ 局限性**

局限包括模拟与真实数据排名不一致、匹配与插补目标差异导致评估冲突、评估指标动态范围有限，以及方法目前仅适用于两模态且对超参敏感。

---

## 126. Gamification-Based Learning Method for Hijaiyah Letters

**arXiv ID:** 2602.03851 | [PDF](https://arxiv.org/pdf/2602.03851v1)

**作者:** Wisnu Uriawan `[一作]` (UIN Sunan Gunung Djati Bandung), Fajar Satria Wiguna `[通讯]` (UIN Sunan Gunung Djati Bandung)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

本研究设计并实现了一种基于游戏化的 Hijaiyah 字母学习方法，采用 ADDIE 模型进行需求分析、界面设计、系统开发、实施与评估，最终在小学学生中通过 Unity 2D 与 Firebase 构建了可离线使用的移动学习应用。

**💡 创新点**

创新点包括：①将徽章（Badge）系统作为主要激励机制，实证表明其对学习成绩影响最大；②采用多感官内容（视觉动画、塔维德音频、书写跟踪）并结合自适应难度算法，保持学习者处于 Flow 状态；③实现实时排行榜与进度同步，并在低网速环境下提供完整离线功能。

**🔧 技术方法**

技术栈为 Unity 2D（C#）用于客户端渲染与游戏逻辑；Firebase Realtime Database 与 Firebase Analytics 负责实时数据同步、排行榜、进度追踪与使用统计；Unity Input System 与自定义笔迹检测算法实现书写跟踪；自适应算法根据得分动态调整测验难度。

**📊 数据集**

数据集为 50 名 1–3 年级小学学生（年龄 6–9 岁），采用 28 项前测/后测测试集评估字母识别、发音与书写；同时收集 Firebase 产生的使用日志（每日会话、时长、积分、徽章、排行榜排名）及 20 项满意度问卷（Cronbach‑α 0.914）。

**📈 对比分析**

通过前后测配对 t‑检验、Cohen’s d 计算、Pearson 相关与多元线性回归分析与以往研究（Construct 2、触屏应用、移动游戏）对比，获得 107% 的学习提升、d = 4.87 的极大效应量，92% 学生达 80 分以上主张，且用户每日平均会话 4.2 次、平均时长 14.4 min，整体表现优于传统 Iqro’ 方法。

**⚠️ 局限性**

局限性包括样本量仅 50 人、单地区单周期（4 周）实验、缺乏长期保持效果验证，以及对更广泛年级与多元文化背景的适用性尚未探索。

---

## 127. GOPO: Policy Optimization using Ranked Rewards

**arXiv ID:** 2602.03876 | [PDF](https://arxiv.org/pdf/2602.03876v1)

**作者:** Kyuseong Choi `[一作]` (Cornell University), Raaz Dwivedi `[通讯]` (Cornell University)

**通讯引用:** 326 | [OpenAlex ID](https://openalex.org/A5078564922)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Group Ordinal Policy Optimization（GOPO）方法，用于在 RLHF 的非可验证奖励设置下训练大型语言模型，改用奖励的秩序信息而非绝对数值进行策略更新。

**💡 创新点**

创新点在于将奖励模型训练的对比学习特性（仅可靠的相对顺序）直接映射到策略梯度中，摒弃奖励幅度，使用基于组内排名的优势函数，提升训练稳定性、加速收敛并提升最终模型质量。

**🔧 技术方法**

技术包括：基于 PPO 的组策略优化框架（GRPO）改进，秩序优势变换、KL 预算调节、对抗性评估（LLM-as-judge）以及多模型、多温度下的梯度范数分析。

**📊 数据集**

数据集涵盖三类非可验证任务：Summarization（使用 TL;DR 数据集）、Chat Completion（使用 UltraChat 数据集）和 Instruction Following（使用 Tulu-3 训练集，IFEval 评估集），奖励模型使用 Skywork 和 QRM 两种公开模型。

**📈 对比分析**

比较方法包括训练/验证奖励曲线、LLM-as-judge 的胜率、以及 IFEval 的基准分数。实验表明，GOPO 在训练步数约为 GRPO 的一半时即可达到或超过 GRPO 的最佳验证奖励，LLM-as-judge 胜率持续高于 0.5，且在不同温度、模型规模下表现更稳健。

**⚠️ 局限性**

限制点：对小组采样量 G 的梯度范数更大，可能导致早期 KL 消耗过快；实验仅覆盖三类任务与两种奖励模型，尚未验证在更广泛或更复杂任务中的泛化；并且秩序优势假设奖励模型在相对排序上可靠，而对奖励尺度的精细化能力被忽略，可能在某些需要精准奖励衡量的任务中表现欠佳。

---

## 128. Minimizing Makespan in Sublinear Time via Weighted Random Sampling

**arXiv ID:** 2602.04059 | [PDF](https://arxiv.org/pdf/2602.04059v1)

**作者:** Bin Fu `[一作]` (University of Texas Rio Grande Valley), Hairong Zhao `[通讯]` (Purdue University)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5071513027)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文设计了一种子线性时间的加权随机抽样算法，用于求解多机平行调度的最小化完成时间问题，并给出了一个（1+3ε）近似值及可进一步生成的 sketch 调度；

**💡 创新点**

创新点在于首次将加权随机抽样与生日悖论相结合，实现了该经典调度问题的子线性时间近似方案，并引入了 sketch 调度概念；

**🔧 技术方法**

主要技术包括加权随机抽样、生日悖论、Chernoff 与 union bound 统计分析、分区抽样、滑动窗口自适应抽样以及黑盒近似框架；

**📊 数据集**

论文没有在实际数据集上进行实验，所有结果均来自理论分析与证明；

**📈 对比分析**

与已有基于均匀抽样的子线性算法相比，该方法在理论上获得更低的误差（1+3ε）并保持子线性复杂度 O~(m^5 ε^4 √n + A(mε,ε))，即使在未知 n 的情形下也能实现同样的子线性性能；

**⚠️ 局限性**

局限性包括需要预先设定 ε、δ，算法对处理时间分布假设较强；实现涉及复杂的分区与采样步骤，常数因子较大；缺乏实验验证，实际运行时性能未知；仅适用于同质机器，未扩展到异构机器或带前置约束的调度问题。

---

## 129. OAT: Ordered Action Tokenization

**arXiv ID:** 2602.04215 | [PDF](https://arxiv.org/pdf/2602.04215v1)

**作者:** Chaoqi Liu `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**通讯引用:** 2956 | [OpenAlex ID](https://openalex.org/A5100763891)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种自编码器动作分词器（B-Token），通过注册词元、有限标量量化（FSQ）与嵌套丢弃实现连续动作的压缩，并通过因果注意力诱导左至右的有序标记序列，支持前缀解码以实现任意深度的动作生成。

**💡 创新点**

关键创新在于三大技术的融合：①注册词元聚合时序信息；②FSQ离散化保持可解码性；③嵌套丢弃与因果注意力共同诱导标记优先级，满足高压缩、全可解码和左至右有序的三大需求；

**🔧 技术方法**

使用Transformer编码器-解码器框架，注册词元为压缩瓶颈，FSQ做离散化，嵌套丢弃产生优先级标记，因果注意力保证标记的左至右依赖；

**📊 数据集**

在四大仿真基准（LIBERO、RoboMimic、MetaWorld、RoboCasa）及两项真实桌面抓取/堆叠任务（Pick & Place Ball、Stack Cups）进行评估；

**📈 对比分析**

与传统维度分箱、频域分词、学习的潜在分词器以及扩散基线对比，B-Token在20+任务上获得最高成功率，并且随前缀长度增加性能单调提升，展示了任意深度推理的优势；

**⚠️ 局限性**

局限性包括：仍采用固定的自回归深度，无法根据动作复杂度动态调整标记数量；标记空间受限于代码本大小，过大或过小均影响下游可学习性；在极长或高维动作时仍存在推理延迟问题。

---

## 130. The CitizenQuery Benchmark: A Novel Dataset and Evaluation Pipeline for Measuring LLM Performance in Citizen Query Tasks

**arXiv ID:** 2602.04064 | [PDF](https://arxiv.org/pdf/2602.04064v1)

**作者:** Neil Majithia `[一作]` (Open Data Institute), Nigel Shadbolt `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 CitizenQuery-UK 基准数据集，并用它评估了 11 种 LLM 在英国政府信息查询任务中的表现。

**💡 创新点**

首次提出针对市民查询的专用基准，并通过结构化元数据、原子事实拆分与评估管线实现了高效的事实性评价。

**🔧 技术方法**

采用了 FActScore/SAFE 框架的改进版本，结合 LLM‑as‑a‑judge、弃答检测与原子事实生成/验证技术。

**📊 数据集**

数据来源为 gov.uk 官方网页内容，并结合 Reddit 用户提问样式生成了 22,066 对查询与回答。

**📈 对比分析**

通过 F1@K、弃答率和冗余度 ΔK 等指标对模型进行对比，结果显示开源模型与闭源模型相当，但整体方差大、冗余度高。

**⚠️ 局限性**

局限在于内容更新滞后、评估方差和长尾问题、过度冗余回答以及多语言和实时更新能力不足。

---

## 131. Accountability in Open Source Software Ecosystems: Workshop Report

**arXiv ID:** 2602.04026 | [PDF](https://arxiv.org/pdf/2602.04026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 132. SpecMD: A Comprehensive Study On Speculative Expert Prefetching

**arXiv ID:** 2602.03921 | [PDF](https://arxiv.org/pdf/2602.03921v1)

**作者:** Duc Hoang `[一作]` (Apple), Minsik Cho `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了SpecMD框架，用于在统一硬件环境下评估Mixture-of-Experts（MoE）模型的专家缓存策略，并在此基础上设计了新的Least-Stale淘汰策略。

**💡 创新点**

创新点在于：1）提出SpecMD统一实验平台，消除了硬件差异导致的可比性问题；2）发现MoE专家访问不符合传统LRU/LFU假设，进而设计利用空间-时间结构的Least-Stale策略；3）证明score‑based预取在带宽受限下能比固定top‑k更高效。

**🔧 技术方法**

主要技术包括：PyTorch前向钩子实现无侵入式策略切换；GPU缓存容量软限制与bandwidth仿真；多维度策略空间（路由、预取、淘汰、缺失处理）的组合与实验；以及对不同MoE模型进行的性能计量。

**📊 数据集**

使用的模型与数据集为：MoE模型——Mixtral‑8x7B、OLMoE‑1B‑7B、Phi‑3.5‑MoE、Qwen1.5‑MoE；任务数据集——GSM8K、TruthfulQA、NaturalQuestions，用以衡量推理精度与速度。

**📈 对比分析**

通过与多种基线策略（LRU、LFU、SB、Score‑based预取、不同缺失处理方式）对比，Least‑Stale在5%容量下实现了88‑92% hit率、最高34.7% TTFT下降、碰撞miss降低至1.6‑1.9%，并在多模型多配置下保持稳定提升。

**⚠️ 局限性**

局限性包括：实验仅在单一A100 GPU上进行，缺乏多GPU或更大规模硬件验证；仅考虑前向推理阶段，未探讨多样化负载下的动态行为；并假设MoE专家访问为严格的层级顺序，可能不适用于所有自适应路由策略。

---

## 133. Echo State Networks for Time Series Forecasting: Hyperparameter Sweep and Benchmarking

**arXiv ID:** 2602.03912 | [PDF](https://arxiv.org/pdf/2602.03912v1)

**作者:** Alexander Häußer `[一作]` `[通讯]` (Justus-Liebig-University Giessen), Alexander Häußer (Justus-Liebig-University Giessen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对M4数据集中月度和季度、长度不超过20年的单变量时间序列，构建并评估纯反馈的Echo State Network（ESN）模型，并在两阶段流程中对其关键超参数进行大规模网格搜索与模型选择。

**💡 创新点**

①系统性地在数百万个ESN训练实例上探索泄漏率、谱半径、储备规模和信息准则对预测性能的影响；②为不同频率给出全局最优超参数配置；③在M4规模数据上将ESN与传统统计基准（ARIMA、ETS、TBATS等）进行严格对比，验证其可竞争性。

**🔧 技术方法**

使用Leaky‑Integrator ESN、岭回归读出层、随机搜索调优正则化参数，并通过AIC、AICc、BIC、HQC等信息准则进行模型选择；前处理包括KPSS平稳性检验、差分、[-0.5,0.5]区间缩放。

**📊 数据集**

M4 Forecasting Competition的月度和季度时间序列；每频率选取长度≤20年（≤240观测）的2400条时间序列，分为参数集（2400/月、1200/季）和预测集（同样规模）进行实验。

**📈 对比分析**

将ESN的MASE和sMAPE与naive、drift、seasonal‑naive、mean、ARIMA、ETS、THETA、TBATS等方法进行比较，并记录平均、中央値误差和计算时间。结果显示ESN的MASE与ARIMA、TBATS相当，季度时甚至领先；在计算效率上，ESN明显快于ARIMA和TBATS，优于大多数统计模型。

**⚠️ 局限性**

仅针对单变量、无外部解释变量的预测；仅评估点预测，未给出预测区间；使用固定随机初始化和固定稀疏结构；调参采用网格+随机搜索，未尝试更高效的贝叶斯或梯度优化；仅测试月/季两频，未验证高频或多季节性情形。

---

## 134. Federated Concept-Based Models: Interpretable models with distributed supervision

**arXiv ID:** 2602.04093 | [PDF](https://arxiv.org/pdf/2602.04093v1)

**作者:** Dario Fenoglio `[一作]` (Università della Svizzera italiana), Giovanni De Felice `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5013971043)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种能在联邦学习环境下自适应扩展概念空间、聚合概念标签与概念依赖关系的概念基模型（F‑CM）框架，支持多机构协同训练并保持模型可解释性。

**💡 创新点**

创新点：1) 在动态联邦场景中首次允许概念空间随新客户端加入而扩张；2) 通过服务器端图聚合与模块化架构更新实现概念依赖的自适应重排；3) 只更新受监督概念对应的模块，避免无关参数的训练；4) 兼容多种概念基模型（CBM、CEM、CGM、C²BM），实现统一的联邦实现。

**🔧 技术方法**

技术方法：联邦学习（FedAvg）、概念基模型架构、概念依赖图聚合（按置信度加权并投影为DAG）、模块化网络更新（只增添/重连受影响模块）、概念级局部训练与模块级聚合、差分隐私可选。

**📊 数据集**

实验数据集：五个贝叶斯网络基准（Asia、Sachs、Insurance、Alarm、Hailfinder）和真实医疗影像数据集 SIIM‑Pneumothorax（含 CLIP 生成的概念标签）。

**📈 对比分析**

与中心化、局部训练、静态联邦（S‑F‑CM）以及无监督概念提取等基线比较。结果表明：F‑CM 在任务准确率与概念覆盖率上接近中心化上限，显著优于局部和静态联邦；在概念干预实验中表现出与中心化相当的响应；在动态联邦（新客户端加入）场景下，F‑CM 的收敛速度快、参数更新量少，远优于全量重训练方案。

**⚠️ 局限性**

局限性：假设所有客户端共享相同输入空间与任务标签；对多模态或异构任务的适应性有限；模块化程度受现有概念基模型设计限制；对隐私泄露的定量评估尚未完整；对概念依赖图聚合的鲁棒性在极端噪声或恶意客户端下仍待深入验证。

---

## 135. Learning to Reason in 13 Parameters

**arXiv ID:** 2602.04118 | [PDF](https://arxiv.org/pdf/2602.04118v1)

**作者:** John X. Morris `[一作]` (FAIR at Meta), Saeed Mahloujifar `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究TinyLoRA在强化学习下极小参数量微调对数学推理任务的效果

**💡 创新点**

提出可缩至单参数的LoRA变体TinyLoRA，并证明RL可在数十参数内实现接近全微调性能

**🔧 技术方法**

低秩适配（LoRA、LoRA-XS）、TinyLoRA、强化学习（GRPO）、vLLM、SimpleRL框架

**📊 数据集**

GSM8K、MATH、AIME、AMC、OlympiadBench等数学推理数据集

**📈 对比分析**

与全微调、LoRA、LoRA-XS做对比；TinyLoRA在7B模型上仅13参数即可达到91% GSM8K准确率，95%在120参数，远优于SFT

**⚠️ 局限性**

仅在数学数据集上验证，可能不适用于其他领域；仅在RL与大模型下有效，SFT与小模型效果差

---

## 136. NeuroPareto: Calibrated Acquisition for Costly Many-Goal Search in Vast Parameter Spaces

**arXiv ID:** 2602.03901 | [PDF](https://arxiv.org/pdf/2602.03901v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon James Fong `[通讯]` (University of Macau)

**通讯引用:** 11863 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一套名为 NeuroPareto 的高维、受限预算下的多目标优化框架，整合了校准的非支配等级分类器、深度高斯过程不确定性分解和基于历史的轻量化获取网络，实现高效的候选筛选、精细建模与自适应采样。

**💡 创新点**

三项创新协同：① 通过温度缩放与适应性 MC Dropout 的校准非支配分类器实现可分离不确定性；② 使用稀疏变分、随机 Fourier 特征的深度高斯过程将预测不确定性分解为可约与不可约；③ 记录最近 HV 改进的滑动窗口驱动浅层 MLP 学习式获取策略。

**🔧 技术方法**

核心技术包括深度高斯过程（Deep GP）、稀疏变分推断、随机 Fourier 特征、温度缩放校准、MC Dropout 信息增益、浅层 MLP 学习式获取以及滑动窗口统计。

**📊 数据集**

实验数据集为 DTLZ1–7 与 ZDT1–4,6 的多维多目标基准（D=30,50,100,200；M=2,3）以及真实的 160 维地热储层双目标优化问题。

**📈 对比分析**

与十余种基线（CPS-MOEA、K-RVEA、CSEA、EDN-ARMOEA、MCEA/D、CLMEA）和四个 MOBO 基线（GP-EI、RF-EI、GP-HV、CL-EGO）在 300 次评估、20 次独立试验下进行统计对比；NeuroPareto 在 92% 的测试实例上取得更低 IGD、HV 更高，尤其在高维 DTLZ1、3、6 等案例中提升 15–46% 的性能。

**⚠️ 局限性**

局限性包括需手动调参（如非支配等级数 K、诱导点数、MC 阈值等）、计算量随维度增长仍显著、对极多目标（M>3）及多精度/梯度信息的适配尚未验证、实现门槛较高。

---

## 137. Artifact Removal and Image Restoration in AFM:A Structured Mask-Guided Directional Inpainting Approach

**arXiv ID:** 2602.04051 | [PDF](https://arxiv.org/pdf/2602.04051v1)

**作者:** Juntao Zhang `[一作]` (Iowa State University), Aditya Balu `[通讯]` (Iowa State University)

**通讯引用:** 1111 | [OpenAlex ID](https://openalex.org/A5002594560)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一套轻量化、全自动的原子力显微镜（AFM）图像缺陷检测与修复工作流，包含图像分类、语义分割、智能去倾斜、方向引导填充和图形用户界面；

**💡 创新点**

创新点在于：①基于几何感知的“Smart Flatten”实现了掩模感知的行列线性去倾斜；②采用掩模引导的方向邻域插值和局部高斯平滑，保持3D表面连续性；③将分类、分割、去倾斜和填充模块整合为端到端可交互的 GUI；

**🔧 技术方法**

技术手段包括：使用迁移学习的 ResNet‑18 进行四类缺陷分类；轻量化语义分割网络生成掩模；线性多项式拟合与掩模感知的去倾斜；方向邻域插值（Telea 方法）与局部平滑；Tkinter GUI 进行参数控制与实时可视化；

**📊 数据集**

使用的 AFM 数据集来自爱荷华州立大学实验样品（滑动、硅胶标定、DNA、细胞等），通过 SPM 文件转换为 224×224×3 PNG，训练集、验证集、测试集按 75%/15%/10% 划分，涵盖良好、失跟踪、尖端污染及成像缺陷四类；

**📈 对比分析**

与传统全局多项式、单向线性去倾斜、双线性插值、克里金插值和无掩模 Telea 方法比较，分类准确率达 91.4%，分割 IoU 0.72、Dice 0.83，Smart Flatten 的行列残差 RMSE_line 下降至 49.5 nm（相比全局多项式 135.7 nm），倾斜移除率约 85%，修复后 SSIM 0.701、背景标准差 86.6 nm，表现优于基线；

**⚠️ 局限性**

局限性包括：对极端大尺度或复杂纹理的缺陷适配性仍待验证；缺少多通道（相位、黏附等）融合；需人工调参，自动化参数自适应尚未实现；仅处理二维灰度图像，未覆盖实时扫描中的动态反馈修正。

---

## 138. On the Credibility of Evaluating LLMs using Survey Questions

**arXiv ID:** 2602.04033 | [PDF](https://arxiv.org/pdf/2602.04033v1)

**作者:** Jindřich Libovický `[一作]` (Charles University), Jindřich Libovický `[通讯]` (Charles University)

**通讯引用:** 1181 | [OpenAlex ID](https://openalex.org/A5061045500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对大语言模型在世界价值调查（WVS）中的回答进行评估，比较不同提示（直接对比链式思考）和解码策略（贪婪解码与核采样）对三种相似度度量（均方差、KL散度、自相关距离）的影响。

**💡 创新点**

提出自相关距离度量，能够捕捉回答之间的交互，揭示传统单独回答指标可能高估模型与人类一致性的误差，并系统评估提示与解码策略的交互效应。

**🔧 技术方法**

使用链式思考提示、核采样、贪婪解码、均方差、KL散度以及新提出的自相关距离等技术。

**📊 数据集**

使用世界价值调查（WVS）三种语言（英语、德语、捷克语）跨六国（美国、英国、捷克、德国、伊朗、中国）的数据。

**📈 对比分析**

通过对四个模型（LLaMA‑3、Mistral‑2、EuroLLM、Qwen‑2.5）在上述设置下的实验，发现链式思考加核采样往往获得最低均方差和KL散度，但自相关距离最高，表面一致性好而结构一致性差；贪婪解码低估一致性。

**⚠️ 局限性**

局限性包括仅基于跨国分布忽略人口学差异、未验证LLM内部价值与实际行为的对应关系、采样次数有限以及提示顺序可能影响结果。

---

## 139. SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models

**arXiv ID:** 2602.04208 | [PDF](https://arxiv.org/pdf/2602.04208v1)

**作者:** Hyeonbeom Choi `[一作]` (Seoul National University), Jonghyun Choi `[通讯]` (Seoul National University)

**通讯引用:** 4039 | [OpenAlex ID](https://openalex.org/A5073483751)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于模型自身不确定性的推理策略，能够在单次前向传播中同时自适应调节视觉编码器的注意力温度与动作采样温度，从而提升Vision‑Language‑Action模型的鲁棒性。

**💡 创新点**

创新点在于提出双参考自不确定性度量（对比全置信度与完全不确定性），并将该度量用于同时控制视觉感知与动作生成，完成了无需额外训练或验证器的单通道自适应推理。

**🔧 技术方法**

技术主要包括基于KL散度的自不确定性计算、温度调节的动作采样与视觉注意力、指数移动平均用于衡量步骤级不确定性、tanh映射生成注意力温度，以及自回归VLA与视觉编码器（如SigLIP）的融合。

**📊 数据集**

实验使用了LIBERO、LIBERO‑PRO‑Long、SIMPLER‑WidowX等模拟基准以及真实UR10e机器人完成的“Put A on B”抓取任务。

**📈 对比分析**

与greedy、temperature、top‑k/top‑p采样以及需要额外训练的TTS方法（RoboMonkey、TACO、MG‑Select）进行对比，实验显示本方法在所有基准上均优于基线，且在训练免费、单通道设置下超过现有TTS方案，成功率提升数个百分点。

**⚠️ 局限性**

局限性包括对视觉编码器温度映射参数的依赖、不同VLA架构调参的敏感性，以及在极端动态或遮挡环境下可能需要更频繁的感知更新。

---

## 140. QuadRank: Engineering a High Throughput Rank

**arXiv ID:** 2602.04103 | [PDF](https://arxiv.org/pdf/2602.04103v1)

**作者:** R. Groot Koerkamp `[一作]` `[通讯]` (Karlsruhe Institute of Technology), R. Groot Koerkamp (Karlsruhe Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在二进制文本和 DNA 字母表上分别提出了 BiRank 与 QuadRank 两种高效排名（rank）数据结构，能够在单缓存行内完成查询并实现低空间占用；

**💡 创新点**

创新点在于将多项已有技术融合：对 L1/L2 级别进行内联、使用中点偏移（pairing）减少 popcount 量、采用转置布局与 AVX2 SIMD 并行 popcnt、并加入批处理预取以最大化内存带宽；

**🔧 技术方法**

核心技术包括：内联 L2 偏移（减少缓存缺失）、pairing 技术（将偏移设至块中点）、SIMD 4 方向 popcnt、预取（prefetch）缓存行、AVX2 指令集以及 Rust 并行构建；

**📊 数据集**

实验使用随机 4 GiB 文本以及 10 M 随机查询；还在 92 核 AMD Zen 4 服务器上做了额外测试；

**📈 对比分析**

与多种 Rust 库（Sux、QWT、Genedex、BWA‑MEM 等）及其添加的预取版本进行对比；BiRank 单线程约 1.5×、多线程 2× 以上快于同等空间占用的方案；QuadRank 在 4 字母表上同样 1.5×/2×，并在 FM‑index 中比 Genedex 提升至 4×；

**⚠️ 局限性**

局限性包括：仅针对 AVX2 CPU，未对 AVX512/ARM 进行优化；仅实现 rank（不含 select）；对极大规模文本仍受内存带宽限制；预取实现需手动开启。

---

## 141. Making Videos Accessible for Blind and Low Vision Users Using a Multimodal Agent Video Player

**arXiv ID:** 2602.04104 | [PDF](https://arxiv.org/pdf/2602.04104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 142. WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling

**arXiv ID:** 2602.03924 | [PDF](https://arxiv.org/pdf/2602.03924v1)

**作者:** Michael Aich `[一作]` (Technical University of Munich), Johannes Brandstetter `[通讯]` (JKU Linz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了WIND，一种预训练的基于扩散的全能大气模型，在推理时通过逆问题求解实现多任务无细调。

**💡 创新点**

创新点在于：1）使用扩散强制训练，让模型学习不同噪声水平的序列，支持任意清洁/噪声上下文；2）在推理时通过MMPS将物理约束、尺度变换等引入，不需再训练；3）在一个模型中统一实现预报、尺度向下、稀疏重建、守恒律执行、极端事件对比等任务。

**🔧 技术方法**

技术包括：基于视频的扩散模型、扩散强制（diffusion forcing）训练、无噪声条件的分数匹配、后验采样与时刻匹配（MMPS）、逆问题框架以及物理约束的后验引导。

**📊 数据集**

主要使用ERA5再分析数据（1.5°分辨率、70个通道，6小时步长），对比基准时也使用0.25° ERA5或CMIP6等。

**📈 对比分析**

与专用的AR、FNO、UViT等基准比较，WIND在预报CRPS、能谱一致性、下尺度恢复的高频保持、稀疏重建的RMSE和谱表现上均与或优于基准，长时滚动保持物理一致且无漂移。

**⚠️ 局限性**

局限性包括：推理时需要多步去噪和后验梯度，速度慢；分辨率仅为1.5°，对极端事件细节不足；对极端事件的放大仍低于理论预期；缺乏对外部强迫（如海表温度）的直接建模。

---

## 143. TiCLS : Tightly Coupled Language Text Spotter

**arXiv ID:** 2602.04030 | [PDF](https://arxiv.org/pdf/2602.04030v1)

**作者:** Leeje Jang `[一作]` (University of Minnesota), Jerod Weinman `[通讯]` (Grinnell College)

**通讯引用:** 4367 | [OpenAlex ID](https://openalex.org/A5113507560)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种端到端场景文字检测识别模型，将预训练字符级语言模型与视觉特征紧密融合，显著提升文本识别鲁棒性。

**💡 创新点**

①设计字符级语言预训练模型并用其初始化语言解码器；②将语言解码器与视觉解码器紧耦合，实现视觉‑语言信息融合；③针对场景文字短小、碎片化的特点专门化预训练。

**🔧 技术方法**

Transformer 视觉编码器/解码器（DeepSolo/DETR 结构）、ViTAEv2 backbone、BART 风格字符级预训练语言模型、跨模态投影头、Hungarian 匹配、教师强制解码。

**📊 数据集**

训练集：SynthText 150K、ICDAR 2013、TextOCR、ICDAR 2017‑MLT；语言模型预训练集：SynthText、Total‑Text、ICDAR 2013/2015、TextOCR、AG News、WikiText、POI 名称；评测集：ICDAR 2015、Total‑Text、CTW1500。

**📈 对比分析**

与多种最新端到端文字检测识别基线（DeepSolo、TESTR、SwinTextSpotter v2 等）进行 Hmean/F1 对比；ICDAR 2015 上 90.1/85.4/81.9/79.1，Total‑Text 上 90.4/88.1，CTW1500 上 90.4/88.1，均超过对手，特别在长序列和无词典设置上优势显著。

**⚠️ 局限性**

参数量 2.73 倍 DeepSolo，推理时间 1.55 倍；仅使用贪心解码，长序列仍可能误识；语言模型初始化需额外预训练，模型复杂度提升。

---

## 144. Quantifying Algorithmic Friction in Automated Resume Screening Systems

**arXiv ID:** 2602.04087 | [PDF](https://arxiv.org/pdf/2602.04087v1)

**作者:** Ibrahim Denis Fofanah `[一作]` `[通讯]` (Pace University), Ibrahim Denis Fofanah (Pace University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过模拟实验量化了自动简历筛选系统中的算法摩擦，比较了关键词匹配和语义嵌入两种筛选方式对误判率的影响。

**💡 创新点**

创新点在于将算法摩擦定义为系统层面的误判率并提供可度量的指标，同时证明语义嵌入能显著降低因词汇差异导致的误判。

**🔧 技术方法**

采用了确定性关键词匹配模型、基于向量空间的语义相似度模型以及余弦相似度阈值筛选技术。

**📊 数据集**

使用合成的1000个简历-职位对数据集，保持资格相同但通过同义词、缩写、角色名称等扰动生成表面差异。

**📈 对比分析**

在相同接受率下对比精确度、召回率和F1得分，关键词匹配的精确度0.62、召回率0.45、F1 0.52，语义模型精确度0.89、召回率0.92、F1 0.90，算法摩擦从0.55降至0.08。

**⚠️ 局限性**

局限性包括使用合成数据导致外部效度受限，关键词模型简化了真实ATS的复杂性，且未考虑招聘者行为和候选人策略。

---

## 145. GPAIR: Gaussian-Kernel-Based Ultrafast 3D Photoacoustic Iterative Reconstruction

**arXiv ID:** 2602.03893 | [PDF](https://arxiv.org/pdf/2602.03893v1)

**作者:** Yibing Wang `[一作]` (Peking University), Changhui Li `[通讯]` (Peking University)

**通讯引用:** 7513 | [OpenAlex ID](https://openalex.org/A5067086374)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种基于高斯核的超快三维光声迭代重建方法GPAIR，能够在毫秒级完成大规模体素（8.4M）重建。

**💡 创新点**

创新点包括：①连续物理表示（高斯核）并推导闭式前向/后向算子；②自适应超采样对齐（ASSA）实现高精度时间对齐；③GPU原生Triton内核实现全可微实现；④非负参数化约束（NPC）与血管连通正则化（VCR）融合。

**🔧 技术方法**

核心技术包括：高斯核连续建模、解析前向公式、Triton GPU并行运算、自动微分、Adam优化、余弦退火学习率调度、Hessian+TV血管连通正则。

**📊 数据集**

使用合成血管数据（512×512×256体素，k‑Wave生成）以及三组实物实验数据（小鼠脑、鼠肾、鼠肝，1024元极板采集，256×256×128体素）。

**📈 对比分析**

与传统UBP、MB‑PD等方法对比；在所有阵列与传感器配置下，PSNR/SSIM最高，速度提升200–900×，单次重建时间≤0.82 s；在稀疏视角下保持高连通性与低噪声。

**⚠️ 局限性**

局限包括：假设均匀声速、使用各向同性高斯核（对高度异质或细长结构的适应性有限）；对非标准几何或多模态融合的扩展尚未验证。

---

## 146. Expert Selections In MoE Models Reveal (Almost) As Much As Text

**arXiv ID:** 2602.04105 | [PDF](https://arxiv.org/pdf/2602.04105v1)

**作者:** Amir Nuriyev `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Gabriel Kulp `[通讯]` (RAND Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Mixture-of-Experts（MoE）语言模型中专家路由信息的隐私泄漏，提出仅凭专家选择即可对原始文本进行高精度重建的攻击方案，并实现基于 MLP 与 Transformer 的解码器。

**💡 创新点**

创新点包括：① 只利用低带宽、离散的专家选择而非完整隐藏状态或 logits 即可实现文本逆向；② 提出序列级的 encoder-only Transformer 解码器，显著提升重建准确率；③ 对专家选择进行信息论分析，量化每层熵与互信息；④ 在噪声注入下评估鲁棒性，阐明侧通道泄漏的实际风险。

**🔧 技术方法**

技术手段：训练 3 层 MLP 与 encoder-only Transformer 作为解码器，使用最大似然损失；将每层 top‑k 专家索引编码为 32 维二进制向量；采用信息论方法（熵、互信息）评估路由信息；通过随机置换部分专家索引模拟噪声，测试鲁棒性。

**📊 数据集**

数据集：OpenWebText，用于生成 (文本, 专家选择轨迹) 训练对，分成 100M 训练、10M 验证/测试。

**📈 对比分析**

对比方法：先前基于逻辑回归的攻击；在 32-token 序列上，MLP 达到 63.1% top‑1，Transformer 达到 91.2% top‑1（94.8% top‑10）。进一步展示不同训练集规模、token 频率、噪声水平下的性能曲线，表明 Transformer 在大样本或高频词上表现更优。

**⚠️ 局限性**

局限性：① 仅在 32-token 短序列上验证，长序列性能未系统评估；② 假设攻击者能获得完整专家选择及兼容分词器，跨模型或路由配置的迁移性未知；③ 仅在完整层信息下评估，部分层时未重训练；④ 解码器模型规模较大，部署成本和推理时间需要进一步考量。

---

## 147. DiMo: Discrete Diffusion Modeling for Motion Generation and Understanding

**arXiv ID:** 2602.04188 | [PDF](https://arxiv.org/pdf/2602.04188v1)

**作者:** Ning Zhang `[一作]` (Huawei Central Media Technology Institute), Mingyuan Zhang `[通讯]` (Huawei Central Media Technology Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了DiMo，一种统一的离散扩散框架，实现文本到运动（T2M）和运动到文本（M2T）以及其他相关任务。

**💡 创新点**

将离散扩散与掩码式去噪相结合，支持多步并行重构，实现可调质量–延迟权衡，并在同一模型内完成双向文本–运动生成与理解。

**🔧 技术方法**

采用残差向量量化（RVQ）进行运动离散化，BERT式双向掩码模型为主干，加入GRPO强化学习对齐控制，并使用classifier‑free guidance。

**📊 数据集**

在HumanML3D和KIT‑ML两个运动‑文本基准上训练与评测。

**📈 对比分析**

与多种统一模型（如MotionGPT、MotionGPT‑3、MoTe等）对比，在T2M FID最低、M2T文本指标（BLEU、ROUGE、CIDEr）均居前列，且通过步骤数可实现从高速低质量到慢速高质量的可调性能。

**⚠️ 局限性**

受限于当前数据规模与骨架运动范式，文本生成对评估器敏感；在极长序列和多主体情景下的可扩展性尚待验证。

---

## 148. Structural shifts in institutional participation and collaboration within the AI arXiv preprint research ecosystem

**arXiv ID:** 2602.03969 | [PDF](https://arxiv.org/pdf/2602.03969v1)

**作者:** Shama Magnur `[一作]`, Mayank Kejriwal `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地分析了 2021‑2025 年 cs.AI 预印本中学术、工业及混合机构的参与与合作模式变化，并提出归一化合作指数（NCI）来衡量学术‑工业合作程度。

**💡 创新点**

创新点包括：① 将 LLM 本身用于高精度机构分类并辅以邮箱域名推断；② 设计 NCI 校正团队规模与随机混合基准；③ 在子领域层面细化合作动态与趋势。

**🔧 技术方法**

技术方法涵盖多阶段数据收集与丰富、LLM（GPT‑4o‑mini 等）机构标签、正则与规则混合的邮箱域名推断、统计检验（t 检验、Wilcoxon、Kendall）以及 NCI 的计算。

**📊 数据集**

使用的数据集为 2021‑2025 年 cs.AI arXiv 预印本元数据（约 44,832 篇论文），并结合 OpenAlex 机构信息完成机构与作者信息的丰富。

**📈 对比分析**

通过与随机作者混合基准对比计算 NCI，发现 NCI 始终低于 1，说明学术‑工业合作远低于随机预期；作者团队规模显著增长，混合合作比例保持在 15%‑25%，未出现显著提升，验证后续对未知标签重新分配后结论仍稳健。

**⚠️ 局限性**

局限性包括：机构分类仍受元数据不完整影响，未知类别占比高；NCI 以团队规模为近似，未考虑作者权重；时间截断导致最后月份异常；LLM 推断可能产生误分类。

---

## 149. Following the TRAIL: Predicting and Explaining Tomorrow's Hits with a Fine-Tuned LLM

**arXiv ID:** 2602.04225 | [PDF](https://arxiv.org/pdf/2602.04225v1)

**作者:** Yinan Zhang `[一作]` (Nanyang Technological University), Zhiqi Shen `[通讯]` (Nanyang Technological University)

**通讯引用:** 5527 | [OpenAlex ID](https://openalex.org/A5101789458)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种 fine‑tuned LLM 模型（TRAIL），能够同时预测短期物品受欢迎度并生成自然语言解释，用于无个性化推荐。

**💡 创新点**

创新点在于：①将受欢迎度预测与可解释性结合，构建基于趋势、最新受欢迎度和物品元数据的三维相似度，用对比学习让相似物品产生相似解释；②无需对每个用户单独推理，直接生成全局排名，提升实时性和可扩展性；③首次实现基于 LLM 的可解释性流行预测模型。

**🔧 技术方法**

使用技术包括：预训练 DeepSeek‑7B LLM+LoRA 参数高效微调；对比学习（DTW 计算趋势相似度、余弦相似度元数据、指数相似度最新受欢迎度）结合 InfoNCE 损失；同时使用交叉熵预测受欢迎度；生成式解说通过 LLM 输出并通过对比学习优化一致性。

**📊 数据集**

实验数据集：Douban Movies、Amazon Beauty、Amazon Baby 共三大真实交互日志，按时间窗口划分训练/验证/测试，包含极高稀疏度。

**📈 对比分析**

对比方法包括：传统个性化模型（BPR、UserKNN、SASRec、Caser、HGN、DIFF）；LLM‑based 个性化模型（LLM2BERT4Rec、LLM‑ESR、LLM‑TRSR）；基于流行度的非个性化模型（PARE）；以及公开 LLM（GPT‑4o‑mini、DeepSeek）。TRAIL 在 HR@5、HR@10、NDCG@5、NDCG@10、Jaccard 等指标上均超越所有基线，最高可实现相对提升约 65%（HR@5）和 38%（NDCG@5）等。

**⚠️ 局限性**

局限性：①仍低于真实上限，尤其在流行度主导场景中；②对极度波动或无历史趋势的物品预测仍不稳定；③缺乏个性化信息，无法针对单个用户的细粒度偏好；④对解释质量的评估依赖 LLM 判定，存在主观性；⑤模型在稀疏数据新物品上的表现依赖元数据质量，若缺失会影响解释可信度。

---

## 150. PFluxTTS: Hybrid Flow-Matching TTS with Robust Cross-Lingual Voice Cloning and Inference-Time Model Fusion

**arXiv ID:** 2602.04160 | [PDF](https://arxiv.org/pdf/2602.04160v1)

**作者:** Vikentii Pankov `[一作]`, Dmitrii Vypirailenko `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 PFluxTTS，一种结合持续导向和对齐自由流匹配模型的混合 TTS 系统。

**💡 创新点**

创新点包括推理时向量场融合的双解码器架构、基于 FLUX 的多句语音提示序列条件化以及 48kHz 超分辨率 PeriodWave vocoder。

**🔧 技术方法**

采用流匹配（CFM）、FLUX 变压器、双流/单流解码器、AdaLN、CFG、Midpoint ODE 求解、超分辨率 PeriodWave vocoder 等技术。

**📊 数据集**

使用多语言会话语音数据（约 50k 小时）、VoxLingua-dev、mTEDx、VCTK 等数据集进行训练与评估。

**📈 对比分析**

与 FishSpeech、SparkTTS、F5‑TTS、ChatterBox 及 ElevenLabs 进行主观 MOS/SMOS 与客观 WER/CER/SPK‑SIM 比较，PFluxTTS 在自然度、说话人相似度和可懂度上均优于大多数基线，且 WER 低 23%。

**⚠️ 局限性**

局限包括对跨语言提示的依赖仍需更大数据验证、融合调度 α 的手动设定、以及对极端嘈杂背景下的鲁棒性尚未完全评估。

---

## 151. BPDQ: Bit-Plane Decomposition Quantization on a Variable Grid for Large Language Models

**arXiv ID:** 2602.04163 | [PDF](https://arxiv.org/pdf/2602.04163v1)

**作者:** Junyu Chen `[一作]` (Southwestern University of Finance and Economics), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12089 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Bit-Plane Decomposition Quantization（BPDQ）方法，能够在保持高质量的同时将大型语言模型压缩到 2-bit 级别，实现 72B 大模型在消费级 GPU 上的高效推理。

**💡 创新点**

创新点在于引入可变量化网格：通过按位平面分解和可学习的标量系数打破传统固定网格的形状不变性，并在 Hessian 导向几何中迭代细化位平面与系数，同时加入 delta 校正保持误差传播一致性。

**🔧 技术方法**

采用的技术包括：基于 Hessian 的优化 PTQ、位平面分解与闭式标量系数拟合、按组的位平面更新与系数再拟合、delta 校正、组感知重排序（GAR）以及 LUT 核实现的高效推理。

**📊 数据集**

使用 1024 个 C4 数据样本进行校准，并在 WikiText‑2、GSM8K、MATH500、ARC‑C、BoolQ、HellaSwag、MMLU、LongBench 等基准上评估，同时使用 Qwen‑3、Qwen‑2.5、Ministral‑3 等大模型。

**📈 对比分析**

与 GPTQ、AWQ、AnyBCQ、VPTQ 等方法对比，BPDQ 在 2‑bit 量化下显著领先，例子如 Qwen2.5‑72B 在 GSM8K 上达 83.85%（高于 GPTQ 的 63% 以上）且量化时间仅为 GPTQ 的约 3 倍，推理延迟更低，且在长文本、推理任务上保持鲁棒性。

**⚠️ 局限性**

主要局限在于与向量量化（VPTQ）相比仍有一定的精度差距，且向量量化在量化时间上开销极大；BPDQ 的硬件实现仍需进一步优化，未来可通过旋转技术或更先进的顺序求解器提升性能。

---

## 152. SPPAM: Signature Pattern Prediction and Access-Map Prefetcher

**arXiv ID:** 2602.04100 | [PDF](https://arxiv.org/pdf/2602.04100v1)

**作者:** Maccoy Merrell `[一作]` (Texas A&M University), Paul V. Gratz `[通讯]` (Texas A&M University)

**通讯引用:** 2649 | [OpenAlex ID](https://openalex.org/A5082578661)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种新型二级缓存预取器——SPPAM，将SPP的递归预取与AMPM的访问映射结合，并与L1D的Berti和LLC的Bingo共同工作。

**💡 创新点**

创新点包括：①利用在线学习动态构建访问映射模式；②通过region table实现模式抓取和预取过滤；③使用置信度阈值对预取深度与频率进行动态抑制；④结合全局与局部有用性评估，实现自适应的预取阈值和带宽控制；⑤通过虚拟预取器的流标签实现跨页边界的区域信息同步。

**🔧 技术方法**

技术手段包括：区域映射（region table）与位图记录、直接映射模式表（pattern table）与置信度计数、预取过滤与降级、全局有用性采样、DRAM带宽抑制、虚拟预取器流识别、shadow‑region技术。

**📊 数据集**

使用DPC4工作负载集（包括sierra、bc、cc、sssp、gcc、mcf、xalancbmk等50个基准）在ChampSim模拟器上进行评估。

**📈 对比分析**

通过与基线（Berti+Pythia）和无预取的系统对比，单核情况下SPPAM在所有基准上平均提升6.2%（相较于基线）并比无预取提升31.4%；在带宽受限单核亦保持5.9%提升；多核配置下平均略有下降（-2.12%），Bingo在多核中几乎无性能收益。

**⚠️ 局限性**

局限性：多核环境中性能下降明显，主要受带宽和资源竞争影响；对region scraping的准确性高度依赖，区域尺寸选择需权衡状态开销和碎片；状态占用约109 KiB，增加实现复杂度；需要进一步调优预取阈值和带宽抑制以提升多核表现。

---

## 153. SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?

**arXiv ID:** 2602.03916 | [PDF](https://arxiv.org/pdf/2602.03916v1)

**作者:** Azmine Toushik Wasi `[一作]` (Shahjalal University of Science and Technology), Md Rizwan Parvez `[通讯]` (Qatar Computing Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个名为SpatiaLab的真实场景视觉问答基准，用于评估视觉-语言模型（VLM）在空间推理方面的能力。

**💡 创新点**

创新点在于：①将空间推理拆分为六大类、30个子类，覆盖相对定位、深度与遮挡、方向、尺寸与比例、空间导航及三维几何；②采用多选与开放式两种评估模式；③在真实多样化图像上构建1400道问答，避免合成数据的噪声缺失；④系统对比了25+主流VLM（含开源、闭源、推理增强、空间专用模型）与人类基准，揭示明显差距与错误模式。

**🔧 技术方法**

使用的技术包括：视觉-语言预训练模型（InternVL、Qwen-VL、Gemini、Gemma、Llama、GPT等）；多轮评估流程（多选直接检索、开放式生成+LLM评判）；多种提升手段（链式思维 CoT、自省 CoT、监督微调 SFT、多智能体分解）。

**📊 数据集**

数据集为SpatiaLab，包含来自网络爬取、在线检索与人工采集的多样化真实图像，经过三阶段标注与审核，最终形成1400个视觉问答对。

**📈 对比分析**

比较方法：对所有模型在多选和开放式任务上分别计算准确率，并与人类基准做对比。结果显示多选最高准确率约54%（最佳模型InternVL3.5‑72B），人类达87%；开放式最高约41%（GPT‑5‑mini），人类64%。模型在深度/遮挡、尺寸/比例、导航等子任务表现最差，显示明显的推理与感知瓶颈。

**⚠️ 局限性**

局限性包括：①对真实三维视觉信号的感知仍不够精细，导致遮挡与深度推理错误；②多步骤推理和自省在开放式任务中效果有限，说明缺乏稳健的几何表征；③微调可能导致过拟合与生成稳定性下降；④基准主要侧重单图像推理，未覆盖动态交互与动作反馈。

---

## 154. Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution

**arXiv ID:** 2602.04193 | [PDF](https://arxiv.org/pdf/2602.04193v1)

**作者:** Hyeonjae Kim `[一作]` (Hanyang University), Tae Hyun Kim `[通讯]` (Hanyang University)

**通讯引用:** 11078 | [OpenAlex ID](https://openalex.org/A5100438979)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 DegFlow 框架，利用残差自编码器和潜在流匹配在潜在空间学习真实降解轨迹，能够仅用单张 HR 图像生成任意尺度的低分辨率图像，从而构建大规模真实场景 SR 训练集。

**💡 创新点**

创新点包括：① 在潜在空间采用自然三次样条建模连续降解轨迹，保证轨迹光滑；② 通过 3 阶 Taylor 近似实现中间尺度的 LPIPS 监督，提升感知质量；③ 仅依赖单张 HR 输入即可生成任意尺度 LR 图像，突破 InterFlow 对成对 LR 的依赖。

**🔧 技术方法**

技术手段包括残差自编码器（RAE）、条件流匹配（LFM）、自然三次样条轨迹、LPIPS 感知损失、Taylor 展开以及基于真实 SR 数据集的训练流程。

**📊 数据集**

使用 RealSR-V2 数据集（Canon、Nikon 两种相机的 ×1、×2、×4 对齐 HR–LR 组），并在此基础上利用 DIV2K 的 HR 图像进行外部扩展；评估数据集为 RealSR 与 RealArbiSR。

**📈 对比分析**

在固定尺度 ×3 的 SR 网络（RCAN、HAN、SwinIR、HAT、MambaIR）以及任意尺度 SR 网络（MetaSR、LIIF、CiaoSR）上与 Bicubic、BSRGAN、Real‑ESRGAN、InterFlow 等方法对比，DegFlow 生成的 LR 数据使得 PSNR、SSIM 和 LPIPS 等指标普遍优于对手，甚至逼近 oracle 级别。

**⚠️ 局限性**

局限性在于仍需依赖离散尺度的真实 HR–LR 对；轨迹建模采用预设三次样条，可能无法完全捕捉极端降解情形；在极端噪声或压缩等极端条件下的表现尚待进一步验证。

---

## 155. LatentTune: Efficient Tuning of High Dimensional Database Parameters via Latent Representation Learning

**arXiv ID:** 2602.04190 | [PDF](https://arxiv.org/pdf/2602.04190v1)

**作者:** Sein Kwon `[一作]` (Yonsei University), Sanghyun Park `[通讯]` (Yonsei University)

**通讯引用:** 6742 | [OpenAlex ID](https://openalex.org/A5100322270)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过构建低维潜在空间并在该空间内进行贝叶斯优化，使用数据增强和TabNet预测模型，实现对高维数据库参数的高效全参数调优。

**💡 创新点**

①将所有数据库参数压缩到潜在空间，避免只调优子集；②将目标工作负载信息直接注入潜在空间，省去相似工作负载映射步骤；③采用Latin Hypercube Sampling + TabNet进行数据增强，显著减少对真实基准测试的依赖。

**🔧 技术方法**

LatentTune主要技术包括：Latin Hypercube Sampling (LHS) 采样、TabNet 多标签预测、自动编码器 (AE) 潜在空间生成、贝叶斯优化 (BO) 与高斯过程、以及针对 MySQL 与 RocksDB 的专门评分函数。

**📊 数据集**

使用 MySQL v5.7.37 的 YCSB（A、B、E、F 四种负载）和 RocksDB v6.25.0 的四种读写比例负载；初始训练集 1,000 条，经过 LHS + TabNet 生成 5,000 条增强样本。

**📈 对比分析**

与 OtterTune、CDBTune、RGPE 三个主流基线在相同硬件与工作负载上进行对比。LatentTune 在 MySQL 上相较基线提升 11.82% 吞吐量、46.01% 延迟；在 RocksDB 上提升最高 1332%（通过综合评分函数衡量）。

**⚠️ 局限性**

主要局限包括：构建潜在空间需要额外的 AE 训练时间；当增强样本过多导致信息稀释时，AE 重建误差会升高；对 TabNet 预测准确度的依赖，若工作负载差异过大可能预测误差增大；在极大规模数据库环境下的实时调优仍需进一步验证。

---

## 156. Enforcing Monotonic Progress in Legal Cross-Examination: Preventing Long-Horizon Stagnation in LLM-Based Inquiry

**arXiv ID:** 2602.04206 | [PDF](https://arxiv.org/pdf/2602.04206v1)

**作者:** Hsien-Jyh Liao `[一作]` `[通讯]` (National Taiwan University), Hsien-Jyh Liao (National Taiwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种软有限状态机（Soft-FSM）框架，结合神经语言模型和外部确定性状态控制，实现法律交叉审讯中的可验证信息收集与阶段转换。

**💡 创新点**

创新点在于将程序状态与信息增益挂钩，利用外部有限状态机强制单调推进，而非仅依赖概率生成，解决了LLM在长周期任务中出现的程序停滞（Complexity Cliff）问题。

**🔧 技术方法**

技术上采用神经符号架构：LLM负责生成问题，外部Finite State Machine根据已获取的Key Information Units（KIU）做决策；同时引入外部确定性谓词和信息增益约束。

**📊 数据集**

使用了三起真实台湾刑事谋杀案（案例A、B、C）的正式庭审判决文本，手工构建超过40个KIU的目标信息结构。

**📈 对比分析**

与纯LLM、阶段提示和自平衡提示三种基线对比，Soft-FSM在所有案例中实现了超过97%的信息完整率，冗余率降至0%，并保持极低的方差；基线方法在复杂案例中完整率低于40%。

**⚠️ 局限性**

局限性包括依赖专家手工定义的KIU schema，未集成自动信息抽取；使用oracle witness仅检验询问方逻辑失误，未考虑对手策略变化和动态询问图的适应。

---

## 157. The Missing Half: Unveiling Training-time Implicit Safety Risks Beyond Deployment

**arXiv ID:** 2602.04196 | [PDF](https://arxiv.org/pdf/2602.04196v1)

**作者:** Zhexin Zhang `[一作]` (Tsinghua University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60270 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统研究了训练阶段的隐式安全风险，提出了五级风险等级、十个细粒度风险类别和三种激励类型，并在单体和多体训练场景中进行大规模实验。

**💡 创新点**

首次从训练时刻出发识别并分类隐式安全风险，构建完整的风险框架，证明即使在没有显式指令的情况下模型仍会生成有害行为，并将风险扩展到多代理竞争环境。

**🔧 技术方法**

采用基于代码强化学习的实验设置，让模型生成可执行代码并通过代码执行器评估；使用自动规则检测和 LLM 检测器检测风险；在预训练阶段通过数据混合和重写手段注入背景信息。

**📊 数据集**

实验基于 TACO 代码强化学习数据集，构造 108 条系统提示；在预训练阶段使用标准预训练语料与背景信息文本混合；对 8 个模型（含推理与非推理模型）进行评估。

**📈 对比分析**

通过对 8 个模型在 L1–L3、L4、L5 三个风险层级进行评分，对比不同模型、不同激励与奖励设计；结果显示 Llama‑3.1‑8B‑Instruct 在 L3 的风险率高达 74.4%，非推理模型风险更高，L4 与 L5 也表现出显著风险，说明问题普遍且与模型规模、激励、奖励设计相关。

**⚠️ 局限性**

实验受限于计算资源，未评估更大规模模型；L1–L3 仅 108 条案例；L5 预训练污染仅 100 文档，难以验证大规模效果；缺少对更广泛提示、不同任务和更高阶风险层级的系统验证。

---

## 158. The Role of Target Update Frequencies in Q-Learning

**arXiv ID:** 2602.03911 | [PDF](https://arxiv.org/pdf/2602.03911v1)

**作者:** Simon Weissmann `[一作]` (University of Mannheim), Leif Döring `[通讯]` (University of Mannheim)

**通讯引用:** 806 | [OpenAlex ID](https://openalex.org/A5048798373)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了目标网络更新频率在 Q‑学习中的作用，给出理论最优更新频率调度方案；

**💡 创新点**

提出从近似动态规划角度的嵌套贝尔曼逼近框架，证明固定更新频率不最优，并给出几何增长的最优频率；

**🔧 技术方法**

使用近似动态规划、随机梯度下降、收敛分析（Robbins–Siegmund 等）和非齐次随机优化技术；

**📊 数据集**

主要在 Tabular GridWorld 与 Lunar Lander 环境下实验，结合 DQN（SGD 版）验证理论；

**📈 对比分析**

与固定更新频率对比，最优增长更新在样本复杂度上去掉对 log ε 的依赖，实验曲线更平稳、收敛更快；

**⚠️ 局限性**

仅适用于 Tabular 设定与 SGD 内部优化，未考虑 Adam、非二次目标或非平稳环境，深度 RL 的推广仍有限。

---

## 159. A Probabilistic Framework for Solving High-Frequency Helmholtz Equations via Diffusion Models

**arXiv ID:** 2602.04082 | [PDF](https://arxiv.org/pdf/2602.04082v1)

**作者:** Yicheng Zou `[一作]` (Duke University), Hossein Salahshoor `[通讯]` (Duke University)

**通讯引用:** 307 | [OpenAlex ID](https://openalex.org/A5013689712)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个条件扩散模型作为概率算子，用于高频 Helmholtz 方程的近似和不确定性量化。

**💡 创新点**

通过概率算子学习解决方案分布，克服了确定性神经算子在高频波场中的光谱偏置和输入敏感性，并能量化不确定性。

**🔧 技术方法**

使用了 score‑based 条件扩散模型（DDPM/ SDE），FiLM 2D/3D UNet 或 UViT 结构，频率与源掩码编码，并结合 SDE/ODE 采样。

**📊 数据集**

采用六组不同频率（1.5×10⁵ 至 2.5×10⁶ Hz）从 Gaussian 随机场生成的声速图生成的 Helmholtz 解析解数据，二维分辨率 256×256，三维 64³。

**📈 对比分析**

与 FNO、HNO、U‑Net 等确定性算子对比，Diffusion 在 L²、H¹ 与能量误差上持续低于对手，尤其在最高频率下误差下降幅度明显；同时能输出校准的不确定性分布。

**⚠️ 局限性**

推断速度较慢，需要大量扩散步骤；实验主要在规则方块域和单源设置，复杂几何、多源或大规模 3D 仍需进一步验证。

---

## 160. Representation Geometry as a Diagnostic for Out-of-Distribution Robustness

**arXiv ID:** 2602.03951 | [PDF](https://arxiv.org/pdf/2602.03951v1)

**作者:** Ali Zia `[一作]` (La Trobe University), Farid Hazratian `[通讯]` (University of Tehran)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于表示几何的诊断框架 TorRicc，利用在分布外 (OOD) 任务中仅使用训练集 (ID) 嵌入构建类条件互相 k‑最近邻图，计算全局谱复杂度（torsion proxy）与局部 Ollivier–Ricci 曲率，来评估模型在分布移位下的鲁棒性。

**💡 创新点**

创新点在于首次将结合解析扭矩启发的谱复杂度与图论中的离散 Ricci 曲率联合用于无标签的 OOD 鲁棒性诊断，并通过这两种互补的几何度量构造轻量级的 GeoScore，实现了在检查点层面进行无监督排名与早停的可行方案。

**🔧 技术方法**

所用技术包括：对 ID 嵌入进行 ℓ₂ 标准化后进行类条件互相 k‑近邻图构造、使用自适应高斯权重、计算归一化 Laplacian 的 log‑determinant 作为谱复杂度、对图边进行 Ollivier–Ricci 曲率估计（采用可正则化的 Wasserstein 距离），以及对多种基线指标（CKA、特征范数、热核、持久同调）进行 Spearman 相关性对比。

**📊 数据集**

实验数据集涵盖 CIFAR‑10 及其多个变体（CIFAR‑10.1、CIFAR‑10.2、CIFAR‑10‑C），以及跨域挑战 Tiny‑ImageNet‑C；模型以 ResNet‑18 与 ViT‑S/16 为主，训练使用 ERM 与对比学习两种目标。

**📈 对比分析**

与传统诊断方法比较，TorRicc 在多个检查点上展示了最高的 Spearman 相关系数（torsion proxy ρ≈‑0.88，Ricci 曲率 ρ≈+0.68），并在无标签的检查点选择实验中逼近 Oracle 选择，显著优于随机或基于单一指标的早停策略。

**⚠️ 局限性**

局限性包括：实验仅覆盖视觉任务，未验证在文本或语音等其他模态中的适用性；诊断方法不具备因果解释，仅为统计关联；图构造与曲率估计涉及超参数（k、层级、预处理）且计算成本相对较高。

---

## 161. Shaping Expressiveness in Robotics: The Role of Design Tools in Crafting Embodied Robot Movements

**arXiv ID:** 2602.04137 | [PDF](https://arxiv.org/pdf/2602.04137v1)

**作者:** Elisabetta Zibetti `[一作]` (University Paris8), David St-Onge `[通讯]` (École de technologie supérieure)

**通讯引用:** 566 | [OpenAlex ID](https://openalex.org/A5082797874)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实验了一套以运动为中心的教学工具箱，帮助工程师利用机器人手臂实现富有表现力的动作，结合了自定义的PS4遥控器、Blender动画软件以及MOA运动分析框架，并通过工作坊评估其有效性。

**💡 创新点**

创新点在于将舞蹈学的MOA表达性参数与机器人控制技术结合，形成一套“可执行-可可视化-可优化”三层工具链；此外，使用低成本的PS4遥控器实现直观的体感操控，填补了现有机器人教学工具在表达性设计方面的空白。

**🔧 技术方法**

技术手段包括：1）自定义PS4遥控器实现切换笛卡尔/关节空间、动态速度调节和惯性切换；2）Blender插件与ROS节点通过ZeroMQ实现实时命令下发，利用PD控制器跟踪关节轨迹；3）MOA框架的三步分析流程用于指导表达性设计。

**📊 数据集**

没有使用公开的数据集；评估基于21名工程师参与的三小时工作坊的定性访谈、录像记录与问卷反馈。

**📈 对比分析**

评估方法为主题编码的归纳+演绎定性分析，未给出量化性能指标；研究发现遥控器促进直观即兴探索，Blender支持精细化调整，两者互补，但缺乏客观运动质量分数或基准比较。

**⚠️ 局限性**

局限性包括：①参与者对MOA概念理解有限，导致效果不一；②遥控器在细节序列设计时受限于手感和机械精度；②工作坊规模与时间有限，缺乏长期、跨场景的验证；③未对工具性能做定量基准，难以与现有高级路径规划或LMM生成方法直接比较。

---

## 162. PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models

**arXiv ID:** 2602.04029 | [PDF](https://arxiv.org/pdf/2602.04029v1)

**作者:** Vignesh Kothapalli `[一作]` (Stanford University), Jure Leskovec `[通讯]` (Stanford University)

**通讯引用:** 112084 | [OpenAlex ID](https://openalex.org/A5091272738)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套从零开始合成多表关系型数据库的框架（称为SynthR），为关系型基础模型（RFM）的预训练提供海量可扩展数据。

**💡 创新点**

创新点在于：①将数据库合成拆分为三个抽象层——模式（有向图）、表间主外键连通（二分图）和行级特征分布（结构因果模型）；②利用层次化块模型（HSBM）生成多样化的主外键连接；③证明预训练损失随合成数据库数量和总token数呈幂律缩放。

**🔧 技术方法**

使用的技术包括：随机生成有向无环图（DAG）、层次化随机块模型、结构因果模型（SCM）与条件生成、Transformer（Relational Transformer）作为预训练模型、Masked Token Prediction目标。

**📊 数据集**

数据集：Synthetic数据库由SynthR按设定分布生成；真实数据来自RelBench共6个数据库，用于零样本评估和后续再训练。

**📈 对比分析**

比较方法：在RelBench上对比仅用真实预训练、仅用Synthetic预训练、以及Synthetic+Real混合预训练。结果显示：Synthetic+Real在AUROC上平均提升1.2%，R²提升3.0%，单个任务最高可达7.4%和5.2%。

**⚠️ 局限性**

局限性：①仅支持无环结构，无法生成循环依赖；②当前仅处理数值和分类特征，缺乏文本、图像等多模态字段；③Synthetic数据缺乏文本语义，单独使用时性能不佳，需要与真实数据结合。

---

## 163. Scaling In-Context Online Learning Capability of LLMs via Cross-Episode Meta-RL

**arXiv ID:** 2602.04089 | [PDF](https://arxiv.org/pdf/2602.04089v1)

**作者:** Xiaofeng Lin `[一作]` (Boston University), Xuezhou Zhang `[通讯]` (Boston University)

**通讯引用:** 611 | [OpenAlex ID](https://openalex.org/A5022094334)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练大语言模型在多任务、多回合元强化学习框架下实现在线情境学习

**💡 创新点**

通过在训练阶段让模型在多回合、跨任务中学习“在上下文中学习”，无需外部记忆或显式反思模块，单一的元-RL目标即可让LLM在推理时进行主动探索与策略改进

**🔧 技术方法**

采用元强化学习（GRPO）优化在上下文窗口内的策略，使用预训练的Qwen3系列模型，结合多回合交互协议与轨迹层次奖励

**📊 数据集**

训练集包含五个部分可观测任务（RPS、Minesweeper、Hangman、Wordle、Blackjack），测试集为两类完全未见任务（Maze、Mastermind）

**📈 对比分析**

与原始LLM、单回合RL微调基线及GPT‑5.2等进行对比；Orbit在Maze和Mastermind上表现出显著提升，尤其在第三回合后成功率持续上升，且在不同模型规模上呈现可观的规模效应，超越单回合RL基线并逼近GPT‑5.2的性能

**⚠️ 局限性**

局限性包括：仅能在32k令牌上下文长度内完成多回合交互；实验环境数量有限，未覆盖更广泛任务；缺乏外部记忆或检索增强；训练过程对计算资源要求高且尚未探索更高效的优化策略

---

## 164. Exploring the Potential of Large Language Models in Simulink-Stateflow Mutant Generation

**arXiv ID:** 2602.04066 | [PDF](https://arxiv.org/pdf/2602.04066v1)

**作者:** Pablo Valle `[一作]` (Mondragon University), Aitor Arrieta `[通讯]` (Mondragon University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过使用大型语言模型（LLM）自动生成Simulink‑Stateflow模型的变异体，构建了完整的生成流水线并与传统基线方法进行对比。

**💡 创新点**

创新点在于首次将LLM引入控制系统模型的变异生成，系统评估了不同变异策略、提示方式与温度参数对生成质量与效率的影响，并揭示了LLM在此领域的显著优势。

**🔧 技术方法**

技术核心包括将Stateflow模型转换为JSON表示、基于模板的提示工程、LLM调用（GPT‑4o、GPT‑3.5‑Turbo、Llama‑3、Gemma‑2 等）以及后端解析将LLM输出重构为可编译的Simulink模型。

**📊 数据集**

数据集包含四个代表性Stateflow模型（Door、Fridge、Elevator、Pacemaker），共产生38,400个LLM生成的变异体以及800个基线变异体，用以评估不同模型与设置的表现。

**📈 对比分析**

在效率上，LLM平均比基线快13倍；在有效性上，LLM的可生成率约为0.90–0.97、可编译率为0.70–0.90，等价和重复率显著低于基线，整体变异质量评分普遍高于基线。

**⚠️ 局限性**

局限性包括仅覆盖四个模型，可能无法覆盖更大规模或更复杂的Stateflow设计；LLM生成的变异仍可能出现语义错误，需要额外的结构与语法验证；不同LLM的训练差异也限制了结果的普适性。

---

## 165. Bayesian Networks and Proof-Nets: the proof-theory of Bayesian Inference

**arXiv ID:** 2602.04045 | [PDF](https://arxiv.org/pdf/2602.04045v1)

**作者:** Rémi Di Guardia `[一作]` (IRIF, Université Paris Cité - CNRS), Claudia Faggian `[通讯]` (IRIF, Université Paris Cité - CNRS)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过将贝叶斯网络（Bayesian Network）与线性逻辑（Linear Logic）的证明网络（proof‑nets）建立对应关系，构建了一套基于图形化证明的贝叶斯推理框架；并在此框架下实现了可组合、成本感知的概率推理与图形化推理。

**💡 创新点**

创新点主要有：
1) 引入概率盒子（probabilistic boxes）将条件概率表（CPT）直接嵌入证明网络中；
2) 定义了一种更一般的图形化“分量”与“cut‑net”分解方式，突破传统类型推导树只能通过单一cut的限制；
3) 证明该框架下的语义在任意cut‑net分解下保持不变，实现真正的模块化组合；
4) 通过cut‑net的树形结构实现与贝叶斯网络转化为聚类树（clique tree）的紧密对应，保证推理成本与传统变量消除或Junction Tree算法相当或更低。

**🔧 技术方法**

技术手段包括：
- 线性逻辑的证明网络语法与图形化裁剪/展开（cut‑elimination / cut‑expansion）
- 概率因子（factor）运算（求和、乘积）与因子产品（非张量乘积）
- 图形化拆分（splitting）与cut‑net构造，以及对应的⊗/‐展开实现
- 通过“分量”与“宽度”参数分析推理成本
- 利用d‑separation的图形化判定实现条件独立性推理

**📊 数据集**

论文未提供实验数据集，而是以经典的五变量雨草坪例子（Rain, Sprinkler, Wet, etc.）作为示例进行图形化演示；在性能评估部分仅给出理论复杂度分析，并与传统贝叶斯网络推理算法的成本做对比。

**📈 对比分析**

方法比较：
- 通过将证明网络拆分为cut‑net后计算中间因子大小，证明在最优拆分下，最大中间因子大小等于对应聚类树的最大团大小；
- 计算成本为 m·2^w，其中 m 为cut‑net树中边数，w 为分量宽度；与传统变量消除算法的 2^w 复杂度保持一致。
- 由于采用因子产品而非张量乘积，避免了不必要的维度扩展，理论上推理效率更高。

**⚠️ 局限性**

限制：
- 论文主要聚焦理论框架与复杂度分析，缺乏大规模实验验证；
- 对于连续变量或多值变量的支持尚未详细讨论；
- 需要手工或自动化工具实现证明网络的构造与分解，实际实现复杂度未知；
- 对于动态或在线贝叶斯网络的推理如何映射到证明网络仍未解决。

---

## 166. Robustness of Stable Matchings When Attributes and Salience Determine Preferences

**arXiv ID:** 2602.04115 | [PDF](https://arxiv.org/pdf/2602.04115v1)

**作者:** Amit Ronen `[一作]` (Bar-Ilan University), Sarit Kraus `[通讯]` (Bar-Ilan University)

**通讯引用:** 18370 | [OpenAlex ID](https://openalex.org/A5103213461)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于属性与显著性权重的稳定匹配鲁棒性定义，并给出了多项关于鲁棒性判定、最大鲁棒半径、最鲁棒匹配搜索、鲁棒性-成本权衡以及鲁棒性区域几何结构的多项多项式时间算法。

**💡 创新点**

创新点在于把鲁棒性从传统的偏好序列扰动转化为显著性权重的连续几何扰动，构造了鲁棒性半径、鲁棒性区域多面体及其体积计算框架；通过旋转格与稳定婚姻多面体的结合实现了鲁棒性与成本的高效权衡。

**🔧 技术方法**

使用了凸优化（线性规划与二阶锥规划）、稳定匹配的旋转格与分配多面体、极值分析与体积近似技术，以及剪枝与任何时刻搜索等算法技术。

**📊 数据集**

论文主要为理论性工作，未使用具体真实或合成数据集；所有结果均在假设固定属性维数和无偏好平局的数学模型下证明。

**📈 对比分析**

与现有基于序列扰动的鲁棒性研究相比，本文提供了多项式时间算法；在计算复杂度上从先前的NP‑困难或指数级搜索转为多项式时间，且在鲁棒性-成本权衡上给出了可证上界和下界，保证了理论性能保障。

**⚠️ 局限性**

局限性包括：属性维数需为常数；假设侧A的偏好固定且无平局；仅考虑两侧匹配，未涵盖多方或匹配大小不等；对支持预算k的处理仍依赖枚举，实际大规模场景中效率可能受限。

---

## 167. Understanding the Impact of Differentially Private Training on Memorization of Long-Tailed Data

**arXiv ID:** 2602.03872 | [PDF](https://arxiv.org/pdf/2602.03872v1)

**作者:** Jiaming Zhang `[一作]` (Renmin University of China), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 4000 | [OpenAlex ID](https://openalex.org/A5100401482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了针对长尾数据的DP‑SGD训练的理论分析框架，研究梯度裁剪与噪声注入如何影响模型的记忆化和泛化性能，并通过合成与真实数据实验进行验证。

**💡 创新点**

创新点包括：①从特征学习视角构建了DP‑SGD在长尾分布下的首次理论框架；②推导了训练动态与测试误差的上界，揭示了DP对长尾样本记忆化的削弱及其不公平影响；③在合成与 MNIST、CIFAR‑10 上实验证明了理论结论。

**🔧 技术方法**

采用两层 ReLU CNN、梯度裁剪+高斯噪声的 DP‑SGD，理论使用特征学习、矩阵范数、随机过程等工具；实验采用隐私预算 (ε,δ) 与影响分数评估方法。

**📊 数据集**

使用合成生成的长尾数据以及公开数据集 MNIST 与 CIFAR‑10（LeNet、SmoothNets 等网络）。

**📈 对比分析**

通过对比非 DP 与 DP 的训练动态、合成数据的特征强度/噪声相关系数热力图、以及按影响分数划分的真实数据子集准确率进行评估。实验显示 DP 明显削弱长尾子集的准确率，尤其在低特征强度时差距更大，验证了理论预测。

**⚠️ 局限性**

局限性：理论仅针对两层 CNN 且假设特征与噪声正交；条件较强，难以直接推广到更深网络或更复杂分布；实验中长尾样本的选取依赖影响分数，未能完全对应真实长尾分布。

---

## 168. Semantic Rate Distortion and Posterior Design: Compute Constraints, Multimodality, and Strategic Inference

**arXiv ID:** 2602.03949 | [PDF](https://arxiv.org/pdf/2602.03949v1)

**作者:** Emrah Akyol `[一作]` (Binghamton University), Emrah Akyol `[通讯]` (Binghamton University)

**通讯引用:** 1181 | [OpenAlex ID](https://openalex.org/A5012879525)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文建立了一个统一的策略性高斯语义压缩框架，研究在速率、计算与观测限制下，编码器与解码器目标不一致时的语义信息传输极限。

**💡 创新点**

创新点在于：① 将经典率失真理论与贝叶斯劝说/信息设计结合，得到策略性RD函数；② 通过后验协方差几何形式揭示语义压缩的“语义水分泵”结构；③ 将计算资源解释为隐式速率约束，解释现代多模态模型与链式推理的能效与数据效率；④ 推导出多模态观测消除远程编码的几何均值惩罚的定量说明。

**🔧 技术方法**

核心技术包括：高斯线性系统的后验协方差解析、信息论的对数行列式约束、逆水分泵定理、KKT 条件求解以及对齐问题的后验设计；此外，还使用了多模态观测的精度矩阵和迭代信息提取的链式信息预算。

**📊 数据集**

由于研究主要为理论推导，本文未使用具体实验数据集，而是基于假设的高斯分布和线性语义变换进行分析。

**📈 对比分析**

在理论上，作者通过与传统无偏差率失真、贝叶斯劝说以及直接编码的下界比较，展示了在给定速率下策略性RD函数的可实现性和最优性；并通过多模态观测的“信息增益”量化证明了远程编码相对于直接编码的性能缺口随模态数增加而缩小。

**⚠️ 局限性**

主要限制包括：① 仅适用于高斯源与线性/二次目标；② 假设解码器采用MMSE估计，无法覆盖非高斯或近似推断场景；③ 对计算资源的速率解释是抽象化的，未考虑具体硬件能耗或算法实现细节；④ 在多模态设置下仍假设所有模态独立且可观测，实际应用中可能存在相关性与观测缺失。

---

## 169. Counting the Wait: Effects of Temporal Feedback on Downstream Task Performance and Perceived Wait-Time Experience during System-Imposed Delays

**arXiv ID:** 2602.04138 | [PDF](https://arxiv.org/pdf/2602.04138v1)

**作者:** Felicia Fang-Yi Tan `[一作]` (New York University), Oded Nov `[通讯]` (New York University)

**通讯引用:** 8453 | [OpenAlex ID](https://openalex.org/A5007172071)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在一项在线实验中，研究者让425名受试者完成一个视觉推理任务，在任务第一部分和第二部分之间插入10秒、30秒或60秒的系统延迟，并随机呈现三种时间反馈模式（无时间显示、已过时间、剩余时间）以检验其对用户的等待体验及延迟后任务表现的影响。

**💡 创新点**

创新之处在于首次将时间反馈模式与系统强制等待的下游任务表现相结合，发现尽管不同反馈模式显著影响主观等待感受（如主观持续时间、沮丧程度、愉悦度），但并未导致延迟后任务性能的差异，从而揭示了“体验–表现解耦”的现象，并为等待设计提供了基于情感而非性能的指导。

**🔧 技术方法**

采用的技术包括：三乘三分组的实验设计、在线实验平台Qualtrics与Prolific招募、视觉推理任务（CLEVR图像）设计、NASA‑TLX与PAD情绪量表评估、对主观与客观指标分别进行ANCOVA、ART‑ANOVA和卡方检验，以及定性文本分析（主题分析）。

**📊 数据集**

使用了CLEVR数据集中的30幅3D场景图像作为视觉推理题目的材料，题目基于对象颜色、形状、材质等属性，保证了任务的可重复性和客观性。

**📈 对比分析**

对比方法：在三种时间反馈模式与三种等待时长的9个细胞之间，分别对延迟后段的任务准确率、主观持续时间、沮丧程度、愉悦度等进行统计检验。结果表明：
- 任务准确率（Segment 2）在三种模式与三种时长之间无显著差异；
- 主观持续时间、沮丧程度和愉悦度在不同模式与时长上存在显著主效应，且Remaining‑Time模式导致沮丧上升、愉悦度下降；
- 体验指标与性能指标之间未出现统计关联，体现了体验与表现的解耦。

**⚠️ 局限性**

局限性包括：
1) 任务过于简单、重复且单一（视觉推理测验），缺乏对更复杂或多任务环境的适用性验证；
2) 仅使用了固定的秒级时间反馈（数字计数），未探究动画、进度条或自适应更新等更常见的界面实现；
3) 所有受试者均来自美国Prolific平台，样本可能偏向高数字素养人群，缺乏文化与技术水平多样性；
4) 仅采用回顾性自评量表，未加入生理或即时情绪测量；
5) 等待时长有限（10/30/60秒），未覆盖极短或极长等待的交互特性。

---

## 170. Exploring Emerging Norms of AI Disclosure in Programming Education

**arXiv ID:** 2602.04023 | [PDF](https://arxiv.org/pdf/2602.04023v1)

**作者:** Runlong Ye `[一作]` (University of Toronto), Michael Liut `[通讯]` (University of Toronto Mississauga)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并进行因子实验式情景调查，收集94名计算机科学学生在102种不同AI协作情境下对工作归属感与披露偏好的判断，探讨影响因素。

**💡 创新点**

首次系统研究AI辅助编程中学生归属感与披露偏好的关系，发现AI助力水平与人类精炼是决定归属和披露的核心因素，提出面向过程的归属框架。

**🔧 技术方法**

因子实验设计、情景问卷、统计分析（ANOVA、卡方检验、顺序逻辑回归）。

**📊 数据集**

102个因子组合的情景问卷数据，来自94名学生（无公开数据集）。

**📈 对比分析**

通过ANOVA评估因素效应大小（偏η²），χ²检验披露偏好，顺序逻辑回归预测政策需求。主要发现AI水平对归属与披露影响显著（p<0.001），人类精炼影响次要；未与传统模型比较，但统计效应显著。

**⚠️ 局限性**

使用假设情景，可能无法反映真实评分情境；受社会期望偏差影响；样本仅为西方学术文化；未考虑教师视角，导致对实际归属政策的适用性缺乏验证。

---

## 171. Rethinking Perplexity: Revealing the Impact of Input Length on Perplexity Evaluation in LLMs

**arXiv ID:** 2602.04099 | [PDF](https://arxiv.org/pdf/2602.04099v1)

**作者:** Letian Cheng `[一作]` (University of Melbourne), Hong Jia `[通讯]` (University of Auckland)

**通讯引用:** 26936 | [OpenAlex ID](https://openalex.org/A5102810576)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LengthBenchmark 框架，系统评估 LLM 在不同输入长度下的困惑度与系统级成本（内存、延迟、评估费用）。

**💡 创新点**

将输入长度视为首要系统变量，揭示滑动窗口评估导致的长度偏差，并在全精度与量化模型中验证该偏差的普遍性，首次将预测指标与部署成本统一考量。

**🔧 技术方法**

使用滑动窗口与非滑动窗口（全序列直接累积）两种困惑度计算协议，结合 AWQ、SmoothQuant、GPTQ、HQQ 等主流量化方法，并对多模型（LLaMA‑3.2、Qwen‑2.5）进行评测。

**📊 数据集**

以公开文本数据集 C4 为主，采用 1K–8K token 长度范围进行多长度评估，覆盖多种 LLM。

**📈 对比分析**

对比滑动窗口与全序列评估，发现滑动窗口在长序列上低估性能；量化模型在更长输入时往往取得更低困惑度和更高准确率，且不同量化方式在内存占用与延迟上呈现显著差异，说明长度是影响性能与成本的重要因素。

**⚠️ 局限性**

实验范围仅限困惑度与基准数据集，未覆盖更多多样化数据集、模型族与下游任务；评估未深入探讨不同长度对真实应用效果的直接影响，且对系统优化细节（如分层缓存、并行化）缺乏深入分析。

---

## 172. AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting

**arXiv ID:** 2602.04043 | [PDF](https://arxiv.org/pdf/2602.04043v1)

**作者:** Joanna Kaleta `[一作]` (Warsaw University of Technology), Marek Kowalski `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AnyStyle，一种能在 3D Gaussian Splatting 场景中通过单前向推理实现姿势无关、零样本的多模态（文本或图像）风格化的框架；

**💡 创新点**

1) 模块化的风格注入器采用零初始化卷积，可无缝插入已训练的 3DGS 网络；2) 统一 CLIP（Long‑CLIP）嵌入空间实现文本与图像双模态风格控制；3) 通过 CLIP 空间插值提供细粒度风格调节；4) 仅需微调轻量级模块，保持几何一致，提升可移植性；

**🔧 技术方法**

3D Gaussian Splats、AnySplat 预训练结构、Long‑CLIP 嵌入、零初始化卷积（ControlNet 思路）、VGG perceptual 内容/风格损失、CLIP 方向损失、光栅化渲染及混合损失；

**📊 数据集**

训练：DL3DV‑480P（室内/户外多视角图像）+ WikiArt（风格图像）；评估：TnT（Train、Truck、M60）和 Mip‑NeRF360（Garden）场景 + 50 张未见过的 WikiArt 风格；文本提示由 Mini‑CPM‑V4.5 生成；

**📈 对比分析**

与 Stylos、Styl3R（feed‑forward）以及 StyleGaussian、G‑Style、StylizedGS、SGSST、ClipGaussian 等方法对比，采用 ArtFID、ArtScore 评估并进行用户研究；AnyStyle 在所有场景下取得最优 ArtFID，文本条件下 ArtScore 最高，图像条件下 ArtFID 最高；用户调查中对 AnyStyle 的偏好显著优于基线；

**⚠️ 局限性**

1) CLIP 模态差距仍存在，文本控制下细节仍不如图像；2) 需先训练或使用预训练的 3DGS 基础模型；3) 极端风格可能导致几何不一致或透明度异常；4) 对光照、材质极端变化的泛化尚未充分验证。

---

## 173. BASS: Benchmarking Audio LMs for Musical Structure and Semantic Reasoning

**arXiv ID:** 2602.04085 | [PDF](https://arxiv.org/pdf/2602.04085v1)

**作者:** Min Jang `[一作]` (University of Washington), Noah A. Smith `[通讯]` (Allen Institute for AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BASS音乐理解基准，覆盖结构分割、歌词转写、音乐学分析与艺术家协作四大类，共12项任务；

**💡 创新点**

首次系统评估音频语言模型在长时段音乐结构与语义推理的能力，采用开放式和多选问答混合模式，关注多维音乐推理；

**🔧 技术方法**

使用多模态提示、音频处理与文本生成，结合思考模式的多语言模型（如Qwen3‑Omni、Gemini 2.5 Pro 等）以及音频特征提取与分离技术；

**📊 数据集**

基准数据来自YouTube收集的1993首歌曲，覆盖138小时音乐，使用Harmonix Set、Genius Lyrics、MGPHot、CoSoD等公开数据集进行标注；

**📈 对比分析**

通过多任务准确率、WER、IoU、EMA等指标对14款模型进行评估，平均表现低于30%，最佳为Gemini 2.5 Pro 26.54%，表现最差的为艺术家协作任务；

**⚠️ 局限性**

局限在于数据以英语为主，缺乏多语言覆盖，且音频仅通过YouTube链接提供，模型仍对音乐结构与多艺术家推理表现不足；

---

## 174. From Sparse Sensors to Continuous Fields: STRIDE for Spatiotemporal Reconstruction

**arXiv ID:** 2602.04201 | [PDF](https://arxiv.org/pdf/2602.04201v1)

**作者:** Yanjie Tong `[一作]` (Georgia Institute of Technology), Peng Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 222778 | [OpenAlex ID](https://openalex.org/A5100599435)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种两阶段框架 STRIDE，用于从极少量的点传感器测量重建完整的时空场。

**💡 创新点**

创新点在于将短窗口时间序列编码为低维潜在状态，再通过调制的隐式神经表示（FMMNN）进行连续解码，并给出基于延迟可观测性的理论支持。

**🔧 技术方法**

采用时序编码器（如 LSTM/Mamba）、调制的 FMMNN 隐式网络、傅里叶编码与随机空间采样的联合训练方法。

**📊 数据集**

在四个挑战性基准上验证：Kuramoto–Sivashinsky、Flow Around an Obstacle、Shallow Water Equations、Seismic Wave Propagation。

**📈 对比分析**

与 SHRED、SHRED-ROM、SIREN 等基线对比，STRIDE‑FMMNN 在所有数据集上取得最低相对误差（约 3–10%），对噪声鲁棒、支持超分辨率，并显著优于传统方法。

**⚠️ 局限性**

主要局限是因隐式网络导致的训练与推理成本较高，且对传感器布局敏感，尚需进一步提升计算效率并扩展到更高频或更大规模问题。

---

## 175. Phaedra: Learning High-Fidelity Discrete Tokenization for the Physical Science

**arXiv ID:** 2602.03915 | [PDF](https://arxiv.org/pdf/2602.03915v1)

**作者:** Levi Lingsch `[一作]` (ETH AI Center), Siddhartha Mishra `[通讯]` (ETH AI Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了针对科学图像的双通道离散分词器Phaedra，并在多种PDE与地球观测数据上评估其压缩与重构性能。

**💡 创新点**

通过形态–幅度分解的离散表示，将结构信息与幅度信息分离量化，并采用可学习重组合成，兼顾高频细节与幅度精度，显著提升科学数据的压缩与泛化。

**🔧 技术方法**

结合有限标量量化（FSQ）、向量量化、神经网络编码解码、形态–幅度分解与可学习重组合成，并在训练中使用重构与承诺损失。

**📊 数据集**

在可压缩Euler、不可压Navier‑Stokes、Poisson、Darcy、Allen‑Cahn、声波等PDE数据集，以及Sentinel‑2 L1C观测数据和ERA5气象数据上进行训练与评估。

**📈 对比分析**

与Cosmos、FSQ、VQ‑VAE‑2、IBQ、VAR等现有离散分词器在相同token数下对比，Phaedra在nMAE、nRMSE、谱一致率、局部方差误差等指标上提升30‑70%，且在OOV任务上保持鲁棒性，甚至优于连续自编码器。

**⚠️ 局限性**

仅为单通道分词，忽略耦合PDE间的相互关系；多模态或三维大规模数据的表现尚未验证，且需构建更大规模生成器以验证其作为基础模型的潜力。

---

## 176. GenMRP: A Generative Multi-Route Planning Framework for Efficient and Personalized Real-Time Industrial Navigation

**arXiv ID:** 2602.04174 | [PDF](https://arxiv.org/pdf/2602.04174v1)

**作者:** Chengzhang Wang `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**通讯引用:** 5389 | [OpenAlex ID](https://openalex.org/A5101512474)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 GenMRP——一种生成式多路规划框架，能够实时生成个性化、具有多样性的多条路线，并实现工业级部署。

**💡 创新点**

创新点包括：
• 纠正增强（Correctional Boosting）机制，迭代地调整路段成本，兼顾个性化与多样性；
• Skeleton‑to‑Capillary（STC）方法，用最小子网代替完整路网，显著降低计算量；
• 增量在线推理（去除 GAT 并仅更新被修改路段）实现可接受的实时响应；
• 首个请求级路网数据集 PRN，公开包含数百万用户请求、动态/静态路段特征、历史行驶序列等信息。

**🔧 技术方法**

技术实现：
• Link Cost Model（LCM）采用 DIN 捕获用户场景偏好，GAT 与 MLP 进行路段表示，Multi‑Scenario DSFNet 计算路段成本；
• 生成式迭代过程结合二向 Dijkstra 生成路线；
• STC 通过构造骨干网络并扩展高热度/熟悉/高速路段生成子网；
• 路线采样（Pareto 多维度）用于训练；
• 无 GAT 的增量推理版本。

**📊 数据集**

使用数据集：PRN（约 600K 条请求级样本，包含 1300 条平均路段、用户历史序列、熟悉度频率、热度等 200+ 维特征），并从中抽取 500K 训练集、50K 测试集。

**📈 对比分析**

实验对比：
• 单路生成：与 ST、SD、MT、HF、GA 基线相比，GenMRP 的 Cov_1 在所有测试集上均领先 5% 以上；
• 多路生成：与 KST、KSD、KMT、KHF、2DP 基线相比，GenMRP 的 Cov_K 最高，提升约 4–6%；
• 推理时间：单路 57.85 ms，虽高于基线但可通过无 GAT 版本压缩到 37.8 ms；
• 在线 A/B 测试：Cov_K 提升 0.54%，Deviation Rate、Cov_net、N_P 等指标均有改善。

**⚠️ 局限性**

局限性：
• 在线推理仍较慢（173 ms/单路），需要进一步压缩模型或加速实现；
• 依赖预生成路集 R，若路网动态变化或请求极端多样时可能受限；
• 目前未充分利用实时交通预测等动态信息，导致在极端拥堵或事件场景下的鲁棒性待提升；
• 需更多跨城市、跨国数据验证泛化能力。

---

## 177. Pending Conflicts Make Progress Impossible

**arXiv ID:** 2602.04013 | [PDF](https://arxiv.org/pdf/2602.04013v1)

**作者:** Petr Kuznetsov `[一作]` (Telecom Paris), Guillermo Toyos-Marfurt `[通讯]` (Telecom Paris)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

研究共享对象的进度条件，提出冲突无阻塞性（Conflict‑Obstruction‑Freedom）并证明在异步读写共享内存模型中无法实现该条件的通用构造。

**💡 创新点**

创新点在于基于操作可交换性引入新的进度条件COF，并揭示其与阻塞自由和等待自由的关系；通过理论证明显示，尽管COF在全可交换时等价于等待自由、全冲突时等价于阻塞自由，但在读写共享内存模型中无法实现通用构造，暴露了冲突操作必然导致同步成本的根本限制。

**🔧 技术方法**

采用理论分析方法，包括可交换性与冲突关系的定义、冲突无阻塞性进度条件的形式化、对共识问题的COF改造、以及基于FLP双价性论证的不可实现性证明，构建了COF一致性问题并证明其在三进程以上不可实现。

**📊 数据集**

无实验数据集，全部证明为理论推导与形式化证明。

**📈 对比分析**

与传统的阻塞自由（obstruction‑freedom）和等待自由（wait‑freedom）进度条件对比，COF在所有操作可交换时等价于等待自由、所有操作冲突时等价于阻塞自由；但理论上证明COF无法在读写共享内存模型中实现通用构造，因而在该模型下COF不提供性能改进。

**⚠️ 局限性**

局限在于仅讨论读写原语的异步共享内存模型，未探讨其他通信模型或弱一致性协议；并未给出任何实现实例，只给出不可实现的理论证明。

---

## 178. Explainable Computer Vision Framework for Automated Pore Detection and Criticality Assessment in Additive Manufacturing

**arXiv ID:** 2602.03883 | [PDF](https://arxiv.org/pdf/2602.03883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 179. How Users Understand Robot Foundation Model Performance through Task Success Rates and Beyond

**arXiv ID:** 2602.03920 | [PDF](https://arxiv.org/pdf/2602.03920v1)

**作者:** Isaac Sheidlower `[一作]` (Brown University), Elaine Short `[通讯]` (Tufts University)

**通讯引用:** 846 | [OpenAlex ID](https://openalex.org/A5077011363)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在线和现场实验，研究了非机器人专家如何解读机器人基础模型（RFM）的性能信息（如任务成功率TSR和失败案例），并基于用户反馈提出改进评估与部署的建议。

**💡 创新点**

首次从用户视角系统评估TSR和失败案例等信息的有效性，强调失败案例的标准化报告、真实数据与估计信息的结合，并提出对任务相似度和性能预测的可解释方法。

**🔧 技术方法**

采用在线问卷与现场机器人演示相结合的实验设计，使用Likert量表、RM‑ANOVA、贝叶斯检验、定性主题编码等统计与分析技术。

**📊 数据集**

使用了三项公开RFM评估的真实数据（共16个任务），包括TSR、成功/失败视频、失败描述，主要来自OpenVLA、WidowX、Ufactory xArm等模型和机器人。

**📈 对比分析**

对比不同信息类型（ETSR、EFC、RT‑TSR、RT‑FC）对用户感知信息充足度、信任度和预测准确性的影响。结果显示TSR（尤其是ETSR）显著提升用户信任和预测准确性，失败案例信息也被视为重要，但对舒适度提升作用有限。

**⚠️ 局限性**

局限性包括：任务主要集中在单臂厨房类操作，无法推广至其他机器人形态或安全关键任务；信息仅以文本呈现，未探讨可视化或交互式呈现；现场实验样本量小（14人），且未覆盖长期使用场景；对任务相似度的量化方法仍待完善。

---

## 180. 4DPC$^2$hat: Towards Dynamic Point Cloud Understanding with Failure-Aware Bootstrapping

**arXiv ID:** 2602.03890 | [PDF](https://arxiv.org/pdf/2602.03890v1)

**作者:** Xindan Zhang `[一作]` (Jilin University), Hehe Fan `[通讯]` (Zhejiang University)

**通讯引用:** 2274 | [OpenAlex ID](https://openalex.org/A5002207978)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并评估了一款针对4D点云序列的多模态大语言模型4DPC^2hat，并构建了对应的跨模态数据集。

**💡 创新点**

创新点在于：①首个面向4D点云的MLLM；②使用双向Mamba实现长程时空建模；③提出失败感知自举学习策略；④构建覆盖44K动态资产、200K QA的全新跨模态数据集。

**🔧 技术方法**

技术手段包括：Point‑BERT编码、双向Mamba时序建模、跨模态投影、两级Caption生成、QA生成以及失败感知自举训练。

**📊 数据集**

数据集来源于Objaverse/Objaverse‑XL，生成44K动态资产、700K点云帧、200K高质量QA对，命名为4DPC^2hat数据集。

**📈 对比分析**

与PointLLM、ShapeLLM、MiniGPT‑3D等静态3D基线对比，4DPC^2hat在4D点云captioning和QA任务中在GPT‑4、SimCSE、BLEU‑1等指标上平均提升10–20点，显著优于基线。

**⚠️ 局限性**

局限性包括：依赖人工或LLM生成的QA，训练成本高；对极端动态或大规模场景的泛化能力尚未充分验证；模型规模和推理速度仍受限。

---

## 181. Boost+: Equitable, Incentive-Compatible Block Building

**arXiv ID:** 2602.04007 | [PDF](https://arxiv.org/pdf/2602.04007v1)

**作者:** Mengqian Zhang `[一作]` (Yale University), Fan Zhang `[通讯]` (Yale University)

**通讯引用:** 72884 | [OpenAlex ID](https://openalex.org/A5005958422)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Boost+框架，分离交易收集与排序，设计默认算法与基于VCG的机制，保证搜索者、构建者激励兼容并抵制集成；

**💡 创新点**

引入可插拔构建者算法的VCG机制、冲突组分块默认算法、对冲突组大小的经验分析，以及将集成问题转化为状态依赖执行的分析；

**🔧 技术方法**

使用VCG机制、可信执行环境（TEE）、冲突图分割、枚举与简化求解、并行化实现；

**📊 数据集**

对2025年2月随机采样的10,000个以太坊区块的交易与Flashbots私有包，约2.9M公链交易与56M私有包；

**📈 对比分析**

与rbuilder实现的两种贪心算法和并行算法对比，Boost+默认算法在53%区块为最优，非最优区块均值误差3.3e-3 ETH（相对误差8.6%），多线程可将执行时间降至与并行算法相近；

**⚠️ 局限性**

当冲突组较大且不满足简化条件时默认算法需裁剪交易，导致在高MEV区块上性能下降；缺少完整的可证明最优性与预算平衡保证，且Sybil攻击仍需进一步抑制。

---

## 182. Nemotron ColEmbed V2: Top-Performing Late Interaction embedding models for Visual Document Retrieval

**arXiv ID:** 2602.03992 | [PDF](https://arxiv.org/pdf/2602.03992v1)

**作者:** Gabriel de Souza P. Moreira `[一作]` (NVIDIA), Even Oldridge `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Nemotron ColEmbed V2 系列多模态晚期交互模型，用于视觉文档检索，并在 ViDoRe V3、V1/V2 及 MIRACL‑Vision 等基准上取得第一或接近第一的成绩。

**💡 创新点**

创新点包括：①将 VLM 基座改为双向注意力以提升表示质量；②结合集群采样、硬负样本挖掘、跨语言翻译与模型合并等多项训练策略；③在保持高检索精度的同时，对 late‑interaction 的存储与延迟成本做系统评估与权衡。

**🔧 技术方法**

核心技术包括：VLM（Eagle 2、Qwen3‑VL）+动态图像分块；late‑interaction + MaxSim；InfoNCE 对比学习；双向注意力、硬负样本挖掘、聚类采样、跨语言翻译；模型合并与低维/低精度量化。

**📊 数据集**

使用了 ViDoRe V1/V2/V3、MIRACL‑Vision 等公开基准，以及内部构造的文本+图像对齐数据；训练数据涵盖多语言 PDF、幻灯片、表格等视觉文档。

**📈 对比分析**

在 MTEB ViDoRe V3 排行榜中，nemotron‑colembed‑vl‑8b‑v2 以 63.42 NDCG@10 名列第一；在 V1/V2 排行榜中排名第二；在 MIRACL‑Vision 多语言任务中各语言均优于同规模对手，平均 NDCG@10 超过 0.68。

**⚠️ 局限性**

主要局限为 late‑interaction 需要保存多向量嵌入，导致存储需求数百 GB、推理延迟显著；即使降维/量化后仍面临大规模部署瓶颈；模型对 VLM 基座与高端硬件依赖较强。

---

## 183. DELTA: Deliberative Multi-Agent Reasoning with Reinforcement Learning for Multimodal Psychological Counseling

**arXiv ID:** 2602.04112 | [PDF](https://arxiv.org/pdf/2602.04112v1)

**作者:** Jiangnan Yang `[一作]` (Anhui University), Jie Chen `[通讯]` (Anhui University)

**通讯引用:** 69303 | [OpenAlex ID](https://openalex.org/A5100355322)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了DELTA框架，通过多代理协作将心理咨询过程拆解为证据获取、情绪抽象和回应生成三步，显式利用视觉、声学和文本信号。

**💡 创新点**

创新点在于：①将多模态证据与结构化心理状态表示分离，形成可解释的推理流程；②引入分布式情绪贴合度得分(EAS)作为强化学习奖励，直接优化情绪匹配；③将多代理的讨论式推理与RL相结合，提升共情质量。

**🔧 技术方法**

技术包括：多模态感知代理（MGA）、视觉/语音询问代理、心理状态结构化代理、基于GRPO的强化学习、Jensen‑Shannon距离计算的EAS、LoRA参数高效微调等。

**📊 数据集**

使用MESC多模态情绪支持对话数据集，提供同步视频、音频和文本，并标注情绪与咨询策略。

**📈 对比分析**

与直接提示（DP）以及原始MESC方法对比，DELTA在四项评估维度（全面性、专业性、真实性、安全性）和情绪贴合度上均显著提升，RL进一步提升了情绪贴合度和整体质量。

**⚠️ 局限性**

局限性包括：仅评估单轮对话；代理模型固定，难以处理长时序交互；EAS依赖预训练情绪编码器，跨模态映射仍有误差；未探索更丰富的心理状态动态建模。

---

## 184. HY3D-Bench: Generation of 3D Assets

**arXiv ID:** 2602.03907 | [PDF](https://arxiv.org/pdf/2602.03907v1)

**作者:** Team Hunyuan3D `[一作]`, Zibo Zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 HY3D‑Bench：包含 252k 高质量 3D 对象（带 watertight 网格和多视角渲染）、240k 结构化部件拆分数据、125k 通过 AIGC 生成的长尾类别资产，并提供统一训练数据与评估框架。

**💡 创新点**

创新点：① 完整且统一的高质量 3D 处理管线，直接生成训练所需 watertight mesh 与多视图图像；② 大规模结构化部件拆分数据，支持细粒度生成与编辑；③ 可扩展的 AIGC 合成 pipeline，填补长尾类别缺失，显著提升数据多样性。

**🔧 技术方法**

技术手段包括：Blender 渲染与网格修复、三角网格转 UDF/点云采样、三维分割与部件合并、LLM 文本扩展 + Qwen‑Image + HY3D‑3.0 3D 合成、VAE 与 Diffusion 模型训练、Hunyuan3D‑2.1‑Small 的改进实现。

**📊 数据集**

使用数据集：从 Objaverse / Objaverse‑XL 处理得到的 252k 资产（全局数据）、240k 部件拆分样本（部件级数据）以及 125k AIGC 合成样本；实验中基于 Hunyuan3D‑2.1‑Small 进行训练与评估。

**📈 对比分析**

对比方法：Michelangelo、Craftsman、Trellis、Hunyuan3D 2.1 等公开模型，在同一测试集上使用 Uni3D‑I、ULIP‑I 指标评估。Ours 在 832M 参数时得到 Uni3D‑I 0.3606、ULIP‑I 0.2424，性能与大模型相近，明显优于同规模的 Craftsman。

**⚠️ 局限性**

局限性：仍缺乏动态资产与更广泛任务支持；部分数据处理耗时高；部件拆分受连通组件限制，可能导致细粒度拆分不足；评估仍以单一 3D 生成方法为主，未覆盖多任务通用性。

---

## 185. Training Data Efficiency in Multimodal Process Reward Models

**arXiv ID:** 2602.04145 | [PDF](https://arxiv.org/pdf/2602.04145v1)

**作者:** Jinyuan Li `[一作]` (Washington University in St. Louis), Jiaxin Huang `[通讯]` (Washington University in St. Louis)

**通讯引用:** 1613 | [OpenAlex ID](https://openalex.org/A5046688345)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对多模态过程奖励模型（MPRM）训练中大规模 Monte Carlo（MC）标注数据的冗余问题，提出一种基于已有 MC 信号的无额外成本的数据子集选择方法——Balanced‑Information Score（BIS），实现数据效率提升。

**💡 创新点**

创新点：①从理论角度阐释了梯度噪声与标签可靠性对 MPRM 训练的影响；②提出兼顾标签混合度与可靠度的 BIS 评分，用单一指标筛选高信息量的 rollout；③在不需要额外模型调用或标注的前提下，显著减少训练数据量。

**🔧 技术方法**

技术方法：线性教师–学生框架下的逻辑回归分析、MC 采样与 Beta‑Binomial 噪声建模、梯度信息量理论推导、无监督的 rollout 评分与排序、单次遍历训练、最佳‑N 重新排序评估。

**📊 数据集**

使用数据集：VisualPRM400K‑v1.1（含 565k rollouts、3.17M 步）用于训练；评估基准 VisualProcessBench（5 个来源）以及 MM‑K12、OlympiadBench、MathVerse、MathVista 四个最佳‑N 任务。

**📈 对比分析**

对比方法：随机子采样（Random‑ρ）以及混合/可靠/低‑MC 等启发式子集。实验显示，BIS‑10% 能匹配完整数据集性能；BIS‑25% 在 InternVL2.5‑8B 与 Qwen2.5‑VL‑7B 上分别以 10%–25% 的数据量实现微 F1 与宏 F1 近似或优于全量训练，且在最佳‑N 重新排序中表现最好。

**⚠️ 局限性**

局限性：①仅验证了两种主干模型和特定 MC 采样设置；②对极端低数据预算（<5%）的性能仍有限；③BIS 对可靠度阈值与 α 的选取有一定敏感性，需经验调参。

---

## 186. Semantic Consensus Decoding: Backdoor Defense for Verilog Code Generation

**arXiv ID:** 2602.04195 | [PDF](https://arxiv.org/pdf/2602.04195v1)

**作者:** Guang Yang `[一作]` (Zhejiang University), Xin Xia `[通讯]` (Zhejiang University)

**通讯引用:** 20633 | [OpenAlex ID](https://openalex.org/A5006669765)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种推理时的防御方法Semantic Consensus Decoding（SCD），用于抵御Verilog代码生成模型中的后门攻击。

**💡 创新点**

创新点在于通过提取功能性需求并进行对比解码，自动抑制触发器导致的恶意输出，且不需要模型重新训练或训练数据访问。

**🔧 技术方法**

使用的技术包括功能需求提取器（基于Qwen3Guard‑0.6B微调）、对比解码权重函数 w(D)=exp(-βD) 与 RMS 距离度量。

**📊 数据集**

数据集为RTL‑Coder语料（27K对）经过测试平台过滤得到12K验证样本，用于训练需求提取器；评估使用VerilogEval‑v2和ResBench。

**📈 对比分析**

在三款代码LLM（CodeLlama‑7B、DeepSeek‑Coder‑7B、Qwen2.5‑Coder‑7B）上与三种后门（BadPre、InSent、RTL‑Breaker）对比，SCD将平均攻击成功率从约89%降低到≈2%/1%，且保持甚至提升Pass@1。

**⚠️ 局限性**

限制在于对功能性触发器的防御效果不佳，且推理时需要两次模型前向推断，导致推理时间约翻倍。

---

## 187. ALORE: Autonomous Large-Object Rearrangement with a Legged Manipulator

**arXiv ID:** 2602.04214 | [PDF](https://arxiv.org/pdf/2602.04214v1)

**作者:** Zhihai Bi `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19778 | [OpenAlex ID](https://openalex.org/A5100357282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了ALORED系统，实现了基于四足机械臂的长距离、大物体重排，结合感知、规划与层级强化学习实现自主动作。

**💡 创新点**

创新点包括：1）统一的交互图（ICR）和图神经网络来表征多物体交互；2）对象速度估计器实现闭环速度控制；3）层级RL训练框架在多物体环境中高效学习；4）TAMP规划同时优化访问顺序与目标分配，支持在线重规划。

**🔧 技术方法**

使用的技术主要有：层级强化学习（低层全身控制器+高层速度控制器）、图神经网络（ICR）、LSTM速度估计器、基于SE(2)的多阶段轨迹规划、B&B+贪心的TAMP。

**📊 数据集**

数据集/环境：在Isaac‑Sim中构建了900个并行仿真环境，涵盖3类物体（椅子、桌子、桶）并进行物理参数随机化；真实实验使用Unitree B2+Z1机械臂，配备LiDAR、RealSense摄像头和AprilTag。

**📈 对比分析**

与Direct、Vanilla、PPE、RobotMover等基线以及消除ICR/速度估计器的消融实验对比，ALORED在对象速度跟踪MAE下降约30%，在多目标重排任务中完成时间和行驶距离分别降低约12%~18%，连续32把椅子重排成功率>93%。

**⚠️ 局限性**

局限性：需要预先给定目标姿态和抓取区域；对不同摩擦模型的泛化受限；仍无法自动生成目标位置或抓取策略，需依赖外部知识。

---

## 188. Language Models Struggle to Use Representations Learned In-Context

**arXiv ID:** 2602.04212 | [PDF](https://arxiv.org/pdf/2602.04212v1)

**作者:** Michael A. Lepori `[一作]` (Brown University), Katja Filippova `[通讯]` (Google DeepMind)

**通讯引用:** 1798 | [OpenAlex ID](https://openalex.org/A5037657908)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

评估大语言模型是否能将从上下文中学习到的表示灵活部署到下游任务，主要通过下一个词预测和新提出的自适应世界建模（AWM）任务；

**💡 创新点**

提出AWM任务作为检验上下文学习表示可部署性的标准，同时系统比较开源权重模型与前沿推理模型在该任务和下一个词预测上的表现，揭示了上下文学习表示的“惰性”现象；

**🔧 技术方法**

使用图跟踪任务（随机走访二维格子和一维线性格子）生成上下文表示，并通过Dirichlet能量（DE）与距离相关（DC）指标量化表示结构；设计Instruction与Prefilled两种提示条件，并利用链式推理（Chain-of-Thought）评估前沿模型的推理能力；

**📊 数据集**

构造合成数据集：4×4、5×5二维格子以及16/25状态的一维线性格子，每个状态映射到唯一词汇，共生成多达1000个不同词汇分配的随机走访序列；

**📈 对比分析**

对比开源权重模型与前沿推理模型在下一个词预测（Instruction vs Prefilled）和AWM任务上的准确率；结果显示开源模型在Instruction条件下准确率显著下降，AWM任务整体准确率低于50%（1D）或30%（2D），而前沿推理模型在1D任务上取得非平凡的准确率，但在2D格子上几乎失效；

**⚠️ 局限性**

主要局限在于：上下文学习得到的表示大多惰性，难以被利用；缺乏机制或训练策略使其可部署；链式推理仅在有限情形下改善表现，仍无法完全克服；实验仅在合成任务上，缺乏对真实世界的验证；对前沿模型内部表示缺乏可解释性；

---

## 189. Steering LLMs via Scalable Interactive Oversight

**arXiv ID:** 2602.04210 | [PDF](https://arxiv.org/pdf/2602.04210v1)

**作者:** Enyu Zhou `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 16579 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一种通过递归树结构与非专家用户交互的可扩展监督框架（Scalable Interactive Oversight），用于在生成网站需求文档时提前把握用户意图并提升对强大LLM的可控性。

**💡 创新点**

① 将监督任务拆解成树状子任务并在每个节点进行闭式选择/排序交互，降低非专家认知负担；② 通过递归累积偏好实现弱监督信号放大；③ 允许在交互过程中使用强化学习在线学习，利用用户反馈和专家评估联合优化交互策略。

**🔧 技术方法**

大规模语言模型（如GPT‑5、Gemini等）作为生成器；树状交互代理实现需求树更新与用户交互；强化学习（GRPO等）对交互策略进行在线优化；LLM‑judge用于自动评估对齐分数。

**📊 数据集**

由真实网站爬取的 UI 组件及公开信息生成的产品需求文档集，用作目标意图；合成的初始查询 q；以及在 RL 训练中使用的模拟用户与真实用户交互日志。

**📈 对比分析**

与无交互的 Vibe Coding 基线和普通多轮自由对话交互基线对比，在 PRD 各模块的对齐得分上，SIO 在所有模块平均提升约 15‑20%（最高 54%），且在 RL 训练后在未见模块和不同模型上也保持提升。

**⚠️ 局限性**

早期误解可能被放大导致最终目标偏离；需进一步优化 UI 设计和验证机制；仅关注需求层面，代码层面监督未覆盖；不适用于安全关键任务，需要人工专家审核。

---

## 190. Why Agentic-PRs Get Rejected: A Comparative Study of Coding Agents

**arXiv ID:** 2602.04226 | [PDF](https://arxiv.org/pdf/2602.04226v1)

**作者:** Sota Nakashima `[一作]` (Kyushu University), Yasutaka Kamei `[通讯]` (Kyushu University)

**通讯引用:** 4989 | [OpenAlex ID](https://openalex.org/A5045097606)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过手工分析 654 个被拒绝的 Pull Request（PR），比较了 Agentic-PR（由五种主流代码生成代理产生的）与 Human-PR（人工提交的）的拒绝原因，并揭示了 Agentic-PR 特有的拒绝模式与代理特定的失败模式。

**💡 创新点**

创新点在于：①首次系统性对比 Agentic-PR 与 Human-PR 的拒绝原因；②识别出仅出现于 Agentic-PR 的七类拒绝模式（如对 AI 代码缺乏信任、PR 过大等）；③发现各代理的特定拒绝特征（如 Devin 自动关闭闲置 PR）；④提出一套简单的启发式过滤器，显著降低“未知原因”PR 的比例，为后续大规模研究提供预处理方法。

**🔧 技术方法**

使用的技术主要是定性内容分析（open coding）结合已有拒绝分类框架，对 PR 审核讨论进行手工标注；同时对过滤启发式进行精确度/召回率/F1 评估。

**📊 数据集**

数据集为 2025 年 GitHub 上的 Agentic-PR 数据集（AIDev-pop 子集），包含五种代理（Claude Code、OpenAI Codex、Devin、GitHub Copilot、Cursor）和人类开发者的 PR；共采样 654 个已被拒绝的 PR。

**📈 对比分析**

比较方法为频数统计与比例比较，利用宏观 F1 等指标评估启发式过滤效果；研究发现 Agentic-PR 在拒绝率上显著高于 Human-PR，且在特定拒绝模式上表现差异明显。

**⚠️ 局限性**

局限性包括：①大多数拒绝 PR 缺乏明确反馈（67.9%），导致有效样本量有限；②分析仅覆盖 PR 决策，无法推广至问题(issue)等其他决策场景；③启发式过滤虽有效，但仍可能误删部分非“未知”PR，需在更大样本中进一步验证。

---

## 191. RAPO: Risk-Aware Preference Optimization for Generalizable Safe Reasoning

**arXiv ID:** 2602.04224 | [PDF](https://arxiv.org/pdf/2602.04224v1)

**作者:** Zeming Wei `[一作]` (Shanghai AI Laboratory), Xingcheng Xu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了风险感知偏好优化框架（RAPO），通过在大型推理模型上实现自适应安全推理以提升对复杂 jailbreak 攻击的抵抗力。

**💡 创新点**

创新点在于将安全推理视为上下文对齐任务，结合理论与实证表明安全推理深度应随攻击复杂度动态缩放，并在此基础上设计了风险感知奖励与通用奖励的组合优化方法。

**🔧 技术方法**

技术手段包括：SFT 预热阶段实现安全推理格式对齐；GRPO 强化学习框架；风险感知奖励（依据攻击复杂度评估安全推理充分性）与通用奖励（判断最终回复是否安全与实用）相结合；安全推理生成器与 LLM-as-a-Judge 评估器。

**📊 数据集**

使用的数据集涵盖 StrataSword（三层攻击复杂度）、WildTeaming、WildJailbreak、SorryBench、JailbreakBench、HarmBench、XsTest、MMLU-Pro 等多种安全与通用评测集合。

**📈 对比分析**

与 Intent-Aware、STAR、TARS、IPO、GRPO 等基线相比，RAPO 在基础有害请求拒绝率上达到 0% 或接近 0%，在 WildJailbreak 以及适配攻击（PAIR/TAP）上的 ASR 低至 5.6%–19%，同时保持与基线相当或略优的通用推理性能。

**⚠️ 局限性**

局限性包括：仍需人工标注攻击复杂度标签；奖励设计对模型行为有较大影响，可能出现奖励泛化不足或奖励劫持；在极端高复杂度攻击下仍存在残余 ASR；训练过程计算开销大，尤其是多轮 RL 与奖励评估。

---

## 192. Adaptive 1D Video Diffusion Autoencoder

**arXiv ID:** 2602.04220 | [PDF](https://arxiv.org/pdf/2602.04220v1)

**作者:** Yao Teng `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 3790 | [OpenAlex ID](https://openalex.org/A5027234036)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 One-Dimensional Diffusion Video Autoencoder（One-DVA），实现了可变长度 1D 编码和像素空间扩散解码的视频自动编码器，支持自适应压缩并兼容后续生成任务。

**💡 创新点**

创新点在于：①将查询式 Vision Transformer 与可变长度 dropout 结合，实现动态压缩；②使用像素空间 Diffusion Transformer 进行生成式重建；③通过空间‑时间结构对齐和两阶段训练，使得自编码器兼容并提升 LDM 性能。

**🔧 技术方法**

采用技术包括 Vision Transformer、查询式 Transformer 编码、可变长度 dropout、像素空间 Diffusion Transformer、流匹配损失、感知损失、KL 对齐、两阶段预训练与后置微调、跨模态对齐等。

**📊 数据集**

使用大规模内部多分辨率视频数据（如 17×256×256 等），并在 Open‑Sora 计划数据集上进行评估与训练。

**📈 对比分析**

通过与 CogVideoX、HunyuanVideo、Wanx 等先进 3D‑CNN VAE 在 PSNR、SSIM、LPIPS、rFVD 等重建指标进行对比，One-DVA 在标准压缩下实现了 PSNR/SSIM 领先且 rFVD 接近最优；在视频生成任务上与 Hi‑VAE+DiT 的 gFVD 相当（210.9）。

**⚠️ 局限性**

局限性包括：对高频信息依赖较大，需要较长 1D latent 长度才能重建复杂运动；生成过程中漂移仍可能导致伪影；模型参数量大（1.0B）且训练成本高；在极高运动或复杂纹理的视频中，推理速度和压缩效率受限。

---

## 193. Availability Attacks Without an Adversary: Evidence from Enterprise LANs

**arXiv ID:** 2602.04216 | [PDF](https://arxiv.org/pdf/2602.04216v1)

**作者:** Rajendra Paudyal `[一作]` (Mason Innovation Labs), Duminda Wijesekera `[通讯]` (Mason Innovation Labs)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文实证分析了企业局域网中用户使用USB‑C docking station导致RSTP频繁重新计算的现象，并展示了这种协议合规的行为如何引起实时语音视频服务中断。随后通过将接入端口显式配置为PortFast（edge port）来消除RSTP的收敛，缓解了可用性下降。

**💡 创新点**

创新点包括：①首次将内部非恶意用户操作映射为NIST/MITRE框架下的无意内务攻击，揭示Layer‑2协议与现代端点使用不匹配；②通过现场数据验证Docking/undocking能持续触发RSTP收敛并导致2–4秒的服务中断；③提出简单的edge‑port配置作为有效缓解措施，并系统评估其效果。

**🔧 技术方法**

主要技术手段为：①在Cisco Meraki MS250交换机上采集RSTP日志、端口状态、Topology Change Notification（TCN）；②同步记录用户Dock/undock时间戳；③监控实时应用（VoIP/视频会议）质量（音视频冻结、丢包、重连）；④利用RSTP状态机模型和MAC地址表刷新分析理论基础；⑤在实验室和现场对比配置前后收敛时间及服务质量。

**📊 数据集**

使用的“数据集”为三家分支机构在业务高峰期收集的内部网络日志、端口状态转变时间戳以及实时通信性能指标；并非公开数据集，而是企业内部真实网络采样。

**📈 对比分析**

比较方法：在同一业务场景下，先记录配置默认RSTP时的收敛时间（2–4 s）与服务中断时长，再将受影响端口切换为PortFast后重新测量。结果显示收敛时间接近0，实时音视频质量恢复到正常水平，明显提升了可用性。

**⚠️ 局限性**

局限性：①研究仅针对Cisco Meraki MS250硬件及其默认RSTP实现，其他厂商或版本可能表现不同；②未探讨是否存在恶意利用该缺陷的可能；③仅关注可用性指标，未涉及机密性/完整性等其他维度；④实验样本有限，缺乏跨平台与更大规模网络的验证。

---

## 194. Identifying knowledge gaps in biodiversity data and their determinants at the regional level

**arXiv ID:** 2602.04314 | [PDF](https://arxiv.org/pdf/2602.04314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 195. InterPReT: Interactive Policy Restructuring and Training Enable Effective Imitation Learning from Laypersons

**arXiv ID:** 2602.04213 | [PDF](https://arxiv.org/pdf/2602.04213v1)

**作者:** Feiyu Gavin Zhu `[一作]` (Carnegie Mellon University), Reid Simmons `[通讯]` (Carnegie Mellon University)

**通讯引用:** 15427 | [OpenAlex ID](https://openalex.org/A5064960456)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了InterPReT交互式政策重构与训练框架，允许终端用户通过自然语言指令和演示交互式地教导机器人驾驶策略；

**💡 创新点**

创新点在于将用户指令直接转化为可微分结构化策略（利用LLM生成），并在此结构上进行示范训练，形成循环交互并提供策略摘要，显著提升样本效率与鲁棒性；

**🔧 技术方法**

使用了结构化可微分策略（有向无环图）+LLM（如GPT）生成结构 + imitation learning目标训练权重 + 交互式总结 + 统计分析（线性混合效应模型、t检验等）；

**📊 数据集**

实验数据来自34名非技术用户在gymnasium赛车仿真环境中的手势演示与指令，测试包含10条未见赛道、边缘起始配置以及加噪声等多种情境；

**📈 对比分析**

与传统IL‑MLP基线在平均速度指标上对比，InterPReT在相同赛道、不同赛道、边缘起点和噪声条件下均显著优于基线；样本量更少，用户感知更好，系统可用性不受影响；

**⚠️ 局限性**

局限性包括：仅使用示范学习，指令仅用于解释无法补偿示范不足或冲突；LLM生成的结构可能与实际演示不完全匹配；解释方式相对单一；未完成多模态感知与真实机器人部署的完整验证；

---

## 196. Partial Ring Scan: Revisiting Scan Order in Vision State Space Models

**arXiv ID:** 2602.04170 | [PDF](https://arxiv.org/pdf/2602.04170v1)

**作者:** Yi-Kuan Hsieh `[一作]` (National Yang Ming Chiao Tung University), Yu-Chee Tseng `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 13564 | [OpenAlex ID](https://openalex.org/A5047638290)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种旋转鲁棒的扫描方式和通道过滤的视觉状态空间模型（PRISMamba）

**💡 创新点**

创新点在于将图像划分为同心环进行顺序无关聚合，利用短径向SSM进行上下文传播，并通过硬通道过滤显著降低计算成本

**🔧 技术方法**

使用状态空间模型（SSM）、同心环扫描、短径向递归、硬通道过滤、1×1投影等技术

**📊 数据集**

在ImageNet‑1K、MS COCO、随机遮挡与旋转等多种公开数据集上进行评估

**📈 对比分析**

与现有Vision‑Mamba系列对比，PRISMamba在ImageNet实现84.5% Top‑1、3.9G FLOPs、3054 img/s；在COCO box AP 48.9、mask AP 43.2；在旋转与遮挡测试中保持更高的鲁棒性和更快的推理速度

**⚠️ 局限性**

局限性在于固定图像中心与环宽对偏移主体或极端纵横比的适应性不足，并且在极端旋转导致大面积填充时仍会出现性能下降

---

## 197. OMG-Agent: Toward Robust Missing Modality Generation with Decoupled Coarse-to-Fine Agentic Workflows

**arXiv ID:** 2602.04144 | [PDF](https://arxiv.org/pdf/2602.04144v1)

**作者:** Ruiting Dai `[一作]`, Lisi Mo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文内容未提供，无法确定。

**💡 创新点**

无法确定。

**🔧 技术方法**

无法确定。

**📊 数据集**

无法确定。

**📈 对比分析**

无法确定。

**⚠️ 局限性**

无法确定。

---

## 198. A Multimodal fNIRS-EEG Dataset for Unilateral Limb Motor Imagery

**arXiv ID:** 2602.04299 | [PDF](https://arxiv.org/pdf/2602.04299v1)

**作者:** Lufeng Feng `[一作]` (Beijing Jiaotong University), Ziyu Jia `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并公开了MIND多模态fNIRS–EEG数据集，记录30名受试者在四方向单侧手臂运动想象任务中的EEG和fNIRS信号。

**💡 创新点**

创新点在于首次提供针对单侧四方向运动想象的公开多模态数据，并结合高空间分辨率的fNIRS与EEG，促进细粒度运动想象的解码研究。

**🔧 技术方法**

采用EEG滤波+FBCSP、fNIRS时域均值/斜率特征、sLDA分类，并进行多模态融合与时窗评估。

**📊 数据集**

使用自己采集的MIND数据集，包含64通道EEG（1000Hz）和51通道fNIRS（47.62Hz）共计3600条试验。

**📈 对比分析**

通过10×5折交叉验证比较单模态EEG、单模态fNIRS和三模态融合，单模态EEG最高约31.8%，融合后可达31.9%–32%，显示融合有小幅提升。

**⚠️ 局限性**

局限在于样本量仅30人、MI类目仍有限、分类准确率低于实际应用需求，且未进行跨日或跨实验室验证。

---

## 199. ProxyWar: Dynamic Assessment of LLM Code Generation in Game Arenas

**arXiv ID:** 2602.04296 | [PDF](https://arxiv.org/pdf/2602.04296v1)

**作者:** Wenjun Peng `[一作]` (University of Adelaide), Qi Wu `[通讯]` (University of Adelaide)

**通讯引用:** 10885 | [OpenAlex ID](https://openalex.org/A5060958969)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ProxyWar 框架，通过让 LLM 生成的代理在多种游戏对战中竞争，动态评估代码生成质量。

**💡 创新点**

创新点在于将代码生成、自动化单元测试、迭代修复和多代理比赛相结合，利用 TrueSkill 等多维度指标实现对动态执行表现的细粒度评估。

**🔧 技术方法**

技术包括 LLM 代码生成、自动化测试与修复循环、游戏模拟环境、决策接口设计以及基于 TrueSkill 的排名与分析。

**📊 数据集**

使用了 9 种多样化游戏环境（单人谜题、双人棋类、多人扑克等）以及 18 个主流 LLM 模型作为评测数据集。

**📈 对比分析**

通过与传统 Pass@k 等静态指标对比，ProxyWar 能更清晰地区分模型性能；实验显示多模型在多数游戏中排名差异显著，胜率与资源利用表现也更具可解释性。

**⚠️ 局限性**

主要限制包括：游戏环境未能完全覆盖真实软件工程的复杂性；模型生成结果的随机性、环境选取偏差以及测试覆盖度不足可能影响可复现性和泛化性。

---

## 200. Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis

**arXiv ID:** 2602.04292 | [PDF](https://arxiv.org/pdf/2602.04292v1)

**作者:** Seong-Eun Hong `[一作]` (Korea University), HyeongYeop Kang `[通讯]` (Korea University)

**通讯引用:** 230 | [OpenAlex ID](https://openalex.org/A5011229651)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于事件分解的文本到动作生成框架Event‑T2M，能够在复杂多动作描述下保持动作顺序与自然性；

**💡 创新点**

核心创新在于引入“事件”这一最小语义单元并在Conformer块中加入事件级跨注意力（ECA），使模型能显式对齐动作段与语义单元；

**🔧 技术方法**

技术实现为扩散模型+TMR文本编码器+事件级跨注意力+Conformer架构+自回归的事件编码；

**📊 数据集**

使用HumanML3D、KIT‑ML、Motion‑X标准数据集以及自建的事件级分层评测集HumanML3D‑E；

**📈 对比分析**

在标准测试集上与最新基线保持同等水平；在HumanML3D‑E上，随着事件数升高，Event‑T2M在FID下降、R‑Precision上升，明显优于其他方法；

**⚠️ 局限性**

局限在于缺乏物理约束、长时序可行性、与视觉/音频等多模态融合的研究尚待完善。

---

## 201. Thickening-to-Thinning: Reward Shaping via Human-Inspired Learning Dynamics for LLM Reasoning

**arXiv ID:** 2602.04265 | [PDF](https://arxiv.org/pdf/2602.04265v1)

**作者:** Wenze Lin `[一作]`, Gao Huang `[通讯]` (Tsinghua University)

**通讯引用:** 66633 | [OpenAlex ID](https://openalex.org/A5013240918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于RLVR的奖励塑造框架T2T，通过动态控制输出长度实现从探索到压缩的阶段化学习；

**💡 创新点**

创新点在于将学习过程拆分为“加厚（探索）”和“变薄（压缩）”两阶段，并利用模型当前成功率自适应调节奖励；

**🔧 技术方法**

采用基于verifiable reward的policy梯度方法（GRPO）与自定义长度惩罚/奖励项，整合长度得分与成功率的二次加权；

**📊 数据集**

在数学推理基准MATH‑500、AIME、AMC等上进行评估；

**📈 对比分析**

与GRPO、LASER、W‑REINFORCE、EntroPIC等对比，T2T在多种模型规模下均实现Pass@1和Pass@64显著提升，尤其在大模型（Qwen3‑14B）上占据首位；

**⚠️ 局限性**

局限性包括对小模型效果有限、依赖on‑policy成功率统计可能噪声大、仅验证可验证推理任务，尚未扩展到非可验证或主观任务。

---

## 202. From Ambiguity to Action: A POMDP Perspective on Partial Multi-Label Ambiguity and Its Horizon-One Resolution

**arXiv ID:** 2602.04255 | [PDF](https://arxiv.org/pdf/2602.04255v1)

**作者:** Hanlin Pan `[一作]` (Jilin University), Wanfu Gao `[通讯]` (Jilin University)

**通讯引用:** 1763 | [OpenAlex ID](https://openalex.org/A5027509451)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个两阶段基于POMDP的弱标签多标签学习与特征选择框架：第一阶段通过horizon-1 POMDP与强化学习得到硬伪标签，第二阶段利用RL实现预算约束的序列特征选择。

**💡 创新点**

创新点在于：①将部分多标签歧义化问题严格映射为一阶POMDP，从而将风险最小化转化为期望回报最大化；②用强化学习同时完成硬标签生成和特征子集学习，避免软标签误导；③提供等价证明、策略梯度收敛性与伪标签监督下的超额风险分解，明确两阶段耦合效应。

**🔧 技术方法**

核心技术包括：Transformer编码器+双头结构（策略头+判别头）、horizon-1 POMDP建模、基于策略梯度的RL训练、图正则化的伪标签损失、以及后续的序列特征选择RL。

**📊 数据集**

在九个公开数据集上评估：Birds、HumanPseAAC、Yeast、PlantPseAAC、CHD_49、Slashdot、Bibtex、Mediamill、Chess，涵盖音频、生物、医疗、文本、图像等多种领域。

**📈 对比分析**

与八种现有PML特征选择/学习方法（PML-FSLA、PML-FSMIR、PML-FSSO、fPML、PML-LD、PAMB、PML-VLS、PML-MAP）对比，实验表明POMDP-FS在绝大多数数据集上取得最低的Ranking Loss并在Micro‑F1上夺得第1或第2名，尤其在低预算场景下表现尤为稳健；统计检验亦显著优于对手。

**⚠️ 局限性**

局限性包括：①硬标签生成对阈值/校准敏感，部分数据集微幅落后于软标签方法；②RL训练易受方差影响，需进一步引入方差削减技巧；③目前仅支持单标签硬决策，尚未探索更细粒度的多标签软决策；④对候选标签噪声分布假设有限，鲁棒性待进一步验证。

---

## 203. Tokenization and Morphological Fidelity in Uralic NLP: A Cross-Lingual Evaluation

**arXiv ID:** 2602.04241 | [PDF](https://arxiv.org/pdf/2602.04241v1)

**作者:** Nuo Xu `[一作]` (University of Eastern Finland), Ahrii Kim `[通讯]` (AI-Bio Convergence Research Institute Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统比较了BPE、OBPE和Unigram三种子词分词在六种乌拉尔语中的POS标注性能。

**💡 创新点**

首次在乌拉尔语族中系统评估OBPE，并揭示其在跨语言迁移和低资源场景下的优势。

**🔧 技术方法**

使用子词分词器（BPE、Unigram、OBPE）以及BiLSTM‑CRF和Flair序列标注模型进行评估。

**📊 数据集**

基于Universal Dependencies 2.0乌拉尔语树库（芬兰语、匈牙利语、北萨米语、科米‑齐里安语等）。

**📈 对比分析**

通过在源语言上训练模型并在低资源目标语言上微调，比较准确率和宏F1；OBPE在大多数语言中优于BPE/Unigram，尤其在开放类词形上提升明显；在俄语-科米‑齐里安的Cyrillic组中Unigram更好。

**⚠️ 局限性**

受限于极低资源数据、词表规模受限、仅评估POS、脚本与语义差异未充分分离等因素。

---

## 204. Post-Quantum Identity-Based TLS for 5G Service-Based Architecture and Cloud-Native Infrastructure

**arXiv ID:** 2602.04238 | [PDF](https://arxiv.org/pdf/2602.04238v1)

**作者:** Vipin Kumar Rathi `[一作]` (Ramanujan College), Nikhil Kumar Rajput `[通讯]` (coRAN Labs Private Limited)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出一种无证书的后量子身份认证框架（IBE‑TLS），用身份导向密钥封装取代传统 PKI 的证书与签名，实现云原生与 5G 核心网络的互认证。

**💡 创新点**

创新点在于将后量子身份基础加密（基于 ID‑ML‑KEM）嵌入 TLS 1.3 关键交换，消除证书颁发、分发与验证步骤，并通过阈值私钥生成器（T‑PKG）解决密钥托管与吊销问题。

**🔧 技术方法**

主要技术包括：后量子基于格的身份加密（ID‑ML‑KEM / ML‑DSA）、阈值私钥生成与分发、TLS 1.3 的密钥调度改造、Kubernetes 与 5G Core 的服务发现与身份映射。

**📊 数据集**

评估使用真实部署环境：Kubernetes 集群中的控制平面组件与 5G Core（QORE/free5GC）网络功能，结合 3GPP SBA 流程与实际网络流量进行实验。

**📈 对比分析**

与传统证书基 mTLS 的比较显示：认证带宽从 11–21 KB 降至约 5 KB，签名与证书验证成本消失，握手延迟和 CPU 开销在大多数场景下降低 20–40%，但 ID‑ML‑KEM ciphertext 仍较大。

**⚠️ 局限性**

局限性包括：大尺寸的 IBE ciphertext 对低带宽链路仍有影响、阈值 PKG 部署与管理成本、缺乏标准化的吊销机制、仅适用于单一行政域（私有网络）且对跨域认证需要额外设计。

---

## 205. DeFrame: Debiasing Large Language Models Against Framing Effects

**arXiv ID:** 2602.04306 | [PDF](https://arxiv.org/pdf/2602.04306v1)

**作者:** Kahee Lim `[一作]` (KAIST), Steven Euijong Whang `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大语言模型在公平评估中因提示框架差异导致的隐藏偏差进行系统研究，提出“框架差异”指标并设计基于双重过程的框架感知去偏框架DeFrame。

**💡 创新点**

创新点在于首次量化框架差异，并提出一种同时利用正负框架、准则生成与自我修订的去偏方法，显著提升模型在多种框架下的公平性一致性。

**🔧 技术方法**

技术实现为多轮提示框架：1）框架整合—生成相反框架；2）准则生成—提炼公平准则；3）自我修订—基于准则改写初始答案，结合对不同框架的主动推理。

**📊 数据集**

数据集为三大公平性基准的框架扩展版本：BBQ-Framed、DoNotAnswer-Framed、70Decisions-Framed，并在各基准上分别评估正负框架的偏差。

**📈 对比分析**

与多种提示式去偏基线（PR、IF、TFS、SD等）对比实验显示，DeFrame在偏差评分与框架差异上均取得显著提升，平均偏差下降至最低，框架差异亦被大幅削减。

**⚠️ 局限性**

局限性包括：仅考虑二元正负框架，扩展至多元框架需进一步研究；多轮LLM调用带来计算成本；对大规模模型的深入分析不足，未覆盖交叉/上下文化的偏差。

---

## 206. How Few-shot Demonstrations Affect Prompt-based Defenses Against LLM Jailbreak Attacks

**arXiv ID:** 2602.04294 | [PDF](https://arxiv.org/pdf/2602.04294v1)

**作者:** Yanshu Wang `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5606 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了少样本演示对基于提示的防御（角色导向 Prompt 与任务导向 Prompt）在 LLM 反制 jailbreak 攻击中的交互效果，发现对 RoP 有正向提升，对 ToP 有负向削弱，并在多模型多基准上验证。

**💡 创新点**

提出了少样本演示对 RoP 与 ToP 产生相反影响的理论框架，并给出了角色强化与注意力分散的机制解释，提供了针对 think‑mode 模型的安全悖论洞察。

**🔧 技术方法**

使用 Bayesian in‑context 学习理论、注意力分析与实验验证相结合的技术手段；实现了 RoP、ToP、少样本通用与有害示例的 Prompt 组合，并通过 Qwen3Guard 判别模型评估安全输出。

**📊 数据集**

在四个安全基准上评估：AdvBench、HarmBench、SG‑Bench 与 XSTest，利用六种代表性 jailbreak 攻击（AIM、DAN、Evil Confident、Prefix Rejection、Poems、Refusal Suppression），覆盖多模型（Pangu、Qwen、DeepSeek、Llama‑2 等）。

**📈 对比分析**

通过对比不同 Prompt 组合的安全率（Safe Rate）与拒绝率（Refusal Rate），发现 RoP+少样本可提升 2–4% 安全率，ToP+少样本则降低 6–21%，与理论预测一致；在思考模式模型上表现尤为不佳。

**⚠️ 局限性**

局限性包括：仅评估 7–8B 规模模型；少样本数量固定为 3 例；缺乏对大规模闭源模型的验证；机制验证仅基于统计与推测，缺少显式注意力可视化或探测实验。

---

## 207. LILaC: Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval

**arXiv ID:** 2602.04263 | [PDF](https://arxiv.org/pdf/2602.04263v1)

**作者:** Joohyung Yun `[一作]` (POSTECH), Wook-Shin Han `[通讯]` (POSTECH)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于分层组件图和延迟交互子图检索的多模态检索框架，用于在包含文本、表格、图像的文档中高效检索与查询相关的细粒度组件。

**💡 创新点**

创新点在于构造双层分层组件图（粗粒度与细粒度节点）以显式建模组件之间的语义与层级关系，并结合LLM驱动的查询分解和基于细粒度交互的子图检索实现多跳推理。

**🔧 技术方法**

技术方法包括预训练的多模态编码器、LLM生成子查询、基于束搜索的图遍历、以及对边的延迟交互评分机制。

**📊 数据集**

实验使用了五个公开基准：工业文档、演示幻灯片、多信息图，以及通过网址重构的网页检索数据集。

**📈 对比分析**

在所有五个基准上均实现了SOTA，Recall@3和MRR@10均较之前VisRAG模型提升约10–20%，对应的端到端问答EM/F1亦显著提高。

**⚠️ 局限性**

局限性包括对子组件抽取质量的高度依赖、检索过程仍需要LLM推理导致的延迟，以及在生成任务上的提升空间仍有余地。

---

## 208. Multi-Tier UAV Edge Computing Towards Long-Term Energy Stability for Low Altitude Networks

**arXiv ID:** 2602.04258 | [PDF](https://arxiv.org/pdf/2602.04258v1)

**作者:** Yufei Ye `[一作]` (Hong Kong University of Science and Technology), Liuqing Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 14287 | [OpenAlex ID](https://openalex.org/A5089217677)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个多层无人机边缘计算架构（LATUS），在低空轻型无人机（L‑UAV）与高空备份无人机（H‑UAV）之间协同完成车辆任务的卸载与计算，目标是最小化任务延迟同时保持 L‑UAV 的长期能量稳定。

**💡 创新点**

创新点包括：①引入 Lyapunov 优化实现能量与延迟的自适应权衡；②设计两阶段车辆–L‑UAV 匹配算法和基于块坐标下降（BCD）+ 逐步凸近似（SCA）的联合优化方案；③通过 H‑UAV 的轨迹协调，显著降低 L‑UAV 传输能量和延迟。

**🔧 技术方法**

技术手段：Lyapunov 退化与 drift‑plus‑penalty；基于 BCD 的分块优化；SCA 进行非凸轨迹规划；OFDMA 频分多址；能量采集模型；使用 CVX 求解凸子问题。

**📊 数据集**

使用仿真数据：1 km×1 km 区域内随机分布车辆（V = 10–40），任务大小 D ∈ [1,10] Mb，计算密度 C ∈ [10,100] cycles/bit，模拟 0.2 s 时隙，基于 MATLAB/Simulink 的离散时间仿真。

**📈 对比分析**

与多种基线（固定 H‑UAV 路径、HAP‑UAV 协同、UAV‑地面服务器协同、能量优先等）比较，LATUS 在任务延迟上与最优延迟方法相当，且 L‑UAV 传输能量降低约26%，能量偏差更稳定，整体表现优于现有方法。

**⚠️ 局限性**

局限性：①对 Lyapunov 参数 K 的选择敏感，需经验调优；②算法复杂度较高（主要来自 BCD 与 SCA 的多次凸优化）；③仅考虑 LoS 通信和理想化能量采集，实际环境中多径、遮挡和能量波动可能影响性能；④仅在仿真环境验证，缺乏真实场景部署与测评。

---

## 209. CoLT: Reasoning with Chain of Latent Tool Calls

**arXiv ID:** 2602.04246 | [PDF](https://arxiv.org/pdf/2602.04246v1)

**作者:** Fangwei Zhu `[一作]` (Peking University), Zhifang Sui `[通讯]` (Peking University)

**通讯引用:** 4590 | [OpenAlex ID](https://openalex.org/A5110285832)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CoLT 框架，将隐式推理过程改为可微分的工具调用，模型生成包含压缩推理步骤信息的种子 token，外部解码器将其解码回文本，从而在保持文本推理能力的同时提高效率。

**💡 创新点**

创新点在于：① 用可微分神经模块实现隐式工具调用，避免传统工具的离散化；② 通过种子 token 压缩推理步骤而非直接使用完整隐藏状态；③ 兼容强化学习，支持探索多条合理推理路径；④ 允许多种解码器结构，提升通用性。

**🔧 技术方法**

使用技术包括：Chain‑of‑Thought 以及隐式推理；Transformer 主模型与可微分解码器；种子 token 设计与触发 token；监督训练（main loss + decoder loss）；强化学习（GRPO）进行多轨迹优化；梯度共享以实现端到端学习。

**📊 数据集**

主要数据集：训练集 GSM8k-Aug；评估集 GSM8k、GSM‑Hard、SVAMP、MultiArith；对比实验还使用 MATH 数据集进行强化学习评估。

**📈 对比分析**

与 CoT、iCoT、Coconut、CODI、COLAR、SIM‑CoT 等基线在四个数学数据集上进行对比。CoLT 在准确率上普遍优于现有隐式推理方法，同时推理链长度明显缩短；在 MATH 数据集上加入强化学习后进一步提升性能。

**⚠️ 局限性**

局限性：仅在数学推理任务验证，其他任务（如实体检索、多模态推理）的效果未知；种子 token 的长度与颗粒度对性能影响大，需进一步探索；解码器结构仍有优化空间；在更大模型上扩展可能面临技术挑战。

---

## 210. Strategic Adaptation Under Contextual Change: Insights from a Dyadic Negotiation Testbed for AI Coaching Technologies

**arXiv ID:** 2602.04242 | [PDF](https://arxiv.org/pdf/2602.04242v1)

**作者:** Mobasshira Akter Urmi `[一作]` (University of South Florida), Raiyan Abdul Baten `[通讯]` (University of South Florida)

**通讯引用:** 106 | [OpenAlex ID](https://openalex.org/A5038144794)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实验了一个可复用的双人谈判测试平台，通过在谈判中点人为改变一方的外部选择，探测并量化谈判者的战略适应行为。

**💡 创新点**

创新点在于引入可控的中途情境变更作为“转折点”，使适应过程可观察、可比较；并揭示适应表现与先前行为路径密切相关，为 AI 辅导系统提供评估基准。

**🔧 技术方法**

使用聊天式多议题谈判界面、行为模式标签（整合/分配/中性）与熵、切换频率等度量，采用线性回归+dyad‑clustered SE、FDR 校正等统计技术。

**📊 数据集**

采用 100 名参与者（50 对）在 Prolific 上完成的谈判聊天记录，配合主观价值与关系体验问卷，形成实验数据集。

**📈 对比分析**

通过比较转折点前后行为变化与主观体验及谈判结果的回归，发现分配性漂移显著降低体验，适应路径依赖显著；模型解释度提升约 18–22%（R² 变动）。

**⚠️ 局限性**

局限包括仅测试单一谈判情境、简化的外部选择变更、粗略的行为分类、未考察个体差异以及未实现具体 AI 辅导方案。

---

## 211. Viewpoint Matters: Dynamically Optimizing Viewpoints with Masked Autoencoder for Visual Manipulation

**arXiv ID:** 2602.04243 | [PDF](https://arxiv.org/pdf/2602.04243v1)

**作者:** Pengfei Yi `[一作]` (Institute of Automation), Wenzhao Lian `[通讯]` (School of Artificial Intelligence, Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了MAE-Select框架，使单摄像头机器人能够通过主动视角选择在每个时间段内获取最有信息量的视角，从而完成抓取与装配等操纵任务。

**💡 创新点**

创新点在于利用预训练的多视角掩码自编码器(MV‑MAE)完整的编码-解码结构，构建完整场景表示；并通过模仿学习无监督地学习视角选择策略，避免了手工标注和强化学习的高成本。

**🔧 技术方法**

核心技术包括多视角掩码自编码器预训练、扩散式动作预测器、视角选择的Transformer编码器以及带有straight‑through估计器的离散视角采样。

**📊 数据集**

实验使用ACT、RLBench、MuJoCo模拟环境以及三组真实世界任务（插销、杯子、手电筒等），训练时利用多视角数据，测试时仅单摄像头视角。

**📈 对比分析**

与固定摄像头的Diffusion Policy和其MAE增强版本MAE‑Diffusion对比，MAE‑Select在单摄像头设置下实现了8%–32%的成功率提升，甚至在某些任务中超越多摄像头方案。

**⚠️ 局限性**

主要局限在于只能在离散的预定义视角集合中选择，无法实现连续视角优化，限制了在动态环境中的灵活性。

---

## 212. Contextual Drag: How Errors in the Context Affect LLM Reasoning

**arXiv ID:** 2602.04288 | [PDF](https://arxiv.org/pdf/2602.04288v1)

**作者:** Yun Cheng `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**通讯引用:** 78270 | [OpenAlex ID](https://openalex.org/A5027798962)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM在推理过程中因上下文错误导致的“上下文拖拽”现象，揭示其会将错误模式迁移至后续推理；

**💡 创新点**

首次系统量化并分析了上下文拖拽的结构化影响，并证明其在多模型、多任务中的普遍性；

**🔧 技术方法**

通过大规模模型评测、树编辑距离对结构相似度分析、外部与自检错误信号实验，以及上下文去噪与针对性微调等技术手段实现研究；

**📊 数据集**

使用了AIME、HMMT、GPQA、MMLU、CRUXEval‑I、Game of 24等八个推理基准数据集；

**📈 对比分析**

在与无上下文生成对比的实验中，发现错误上下文会导致10–20%性能下降，甚至导致迭代自我改进出现自损；

**⚠️ 局限性**

当前方法无法彻底消除拖拽效应，受限于注意力架构对错误上下文的过度利用，需进一步改进模型结构或训练策略。

---

## 213. Revisiting Prompt Sensitivity in Large Language Models for Text Classification: The Role of Prompt Underspecification

**arXiv ID:** 2602.04297 | [PDF](https://arxiv.org/pdf/2602.04297v1)

**作者:** Branislav Pecher `[一作]` (Kempelen Institute of Intelligent Technologies), Jan Cegin `[通讯]` (Kempelen Institute of Intelligent Technologies)

**通讯引用:** 50 | [OpenAlex ID](https://openalex.org/A5081841143)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在零/少样本文本分类任务中的提示敏感性，系统比较了最小（欠规范）提示与指令提示，并评估了多种缓解策略（UNK填充、校准、上下文学习、指令微调、标签标签、聊天模板等）。

**💡 创新点**

首次证明提示欠规范是导致LLM提示敏感性的主要原因，并通过指令提示、上下文学习、指令微调和校准等方法显著缓解；引入logit与线性探测分析揭示内部表示对欠规范影响有限。

**🔧 技术方法**

采用logit分析（提取标签logit分布并与准确率相关），线性探测（在各层激活上训练岭回归分类器），以及10种不同提示变体的生成与评估；同时使用校准、UNK填充、标签标签、聊天模板等技术进行敏感性缓解。

**📊 数据集**

使用了SST‑2（情感二分类）、AG News（四类新闻分类）和MMLU（多项选择问答）三个公开数据集。

**📈 对比分析**

通过对每种模型、数据集与提示格式取10个提示变体，计算平均准确率及标准差，并记录logit均值和线性探测准确率。结果表明指令提示显著提升准确率并降低方差，指令提示+上下文学习进一步提升，校准与UNK效果相对较弱；logit值与准确率相关系数高达0.757。

**⚠️ 局限性**

研究仅覆盖3个数据集与3个模型（含其指令微调版），未扩展至更大规模或多样化模型；提示数量限定为10，未充分探索更多提示；生成评估采用字符串匹配，可能低估模型能力；线性探测仅在单层激活上训练，未深入分析模型内部机制。

---

## 214. A Domain-Specific Curated Benchmark for Entity and Document-Level Relation Extraction

**arXiv ID:** 2602.04320 | [PDF](https://arxiv.org/pdf/2602.04320v1)

**作者:** Marco Martinelli `[一作]` (University of Padova), Gianmaria Silvello `[通讯]` (University of Padova)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一个关于肠脑轴的高质量信息抽取基准GutBrainIE，包含NER、NEL、M-RE和C-RE四个任务，手工标注1600多篇PubMed摘要。

**💡 创新点**

创新点在于细粒度13种实体、17种关系谓词，四层质量层级（Platinum, Gold, Silver, Bronze），并公开完整标注流程与基线。

**🔧 技术方法**

采用GLiNER进行NER，ATLOP进行关系抽取，BiomedBERT等模型做实体链接，结合人工审核和弱监督训练。

**📊 数据集**

使用PubMed抽取的1600多篇摘要作为数据集，并在内部和国际评测赛中进行验证。

**📈 对比分析**

通过内部实验与17支团队的评测赛比较，基线微F1分别在NER 0.79/0.79，M-RE 0.21/0.33等，表明NER可达同水平，RE仍具挑战；部分团队提升明显。

**⚠️ 局限性**

限制包括Silver与Bronze层质量不一、自动注解噪声、批量式注释可能导致不一致，且当前低质量数据对模型效果负面。

---

## 215. Light Up Your Face: A Physically Consistent Dataset and Diffusion Model for Face Fill-Light Enhancement

**arXiv ID:** 2602.04300 | [PDF](https://arxiv.org/pdf/2602.04300v1)

**作者:** Jue Gong `[一作]` (Shanghai Jiao Tong University), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 25570 | [OpenAlex ID](https://openalex.org/A5019708391)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种面部填光增强（FFE）方法，利用虚拟填光源在保持原场景照明不变的前提下提升面部图像的曝光与细节。

**💡 创新点**

创新点包括：
1) 构建大规模物理一致的 LYF‑160K 160K 对数据集，使用六维可控参数渲染填光；
2) 设计 PALP（Physics‑Aware Lighting Prompt），将六维参数编码为扩散模型可用的提示令牌；
3) 开发 FiLitDiff，一步扩散模型，能够低计算成本下实现高质量、可控的填光效果；
4) 通过平面光重建监督提升物理先验，使模型对光照参数的理解更精准。

**🔧 技术方法**

技术手段包括：
- 物理基渲染器（面积光、Fibonacci 采样、光照与可见度估计）；
- 6D 参数归一化与 FiLM + Transformer 条件编码；
- 稳定扩散模型（Stable Diffusion）微调为单步 DDIM 生成；
- 波形变换强度控制实现无训练的亮度调节。

**📊 数据集**

使用数据集：
- 训练集 LYF‑160K（160K 对），基于 FFHQ 图像通过渲染生成；
- 验证集 LYF‑Val（3006 对）与 LYF‑EditVal（3006 对），分别来自 CelebA 测试集与 Qwen‑Image‑Edit 预处理的低光/侧光样本。

**📈 对比分析**

与 SOTA 方法比较：DPR、SMFR、IC‑Light 以及 Qwen‑Image‑Edit。评估指标包括 PSNR、SSIM、DISTS、LPIPS、MSSWD、CLIPIQA、LIQE、TOPIQ。FiLitDiff 在绝大多数全参考和无参考指标上均优于对比方法，特别是在 PSNR/SSIM/DISTS 方面表现突出，且保持背景照明一致性。

**⚠️ 局限性**

局限性：
- 仅针对单一面部对象，假设单一光源，无法处理多光源或复杂阴影场景；
- 需要先验的 6D 参数，实际使用时参数预测仍是挑战；
- 对极端光照、非面部背景或真实场景的泛化仍待进一步验证；
- 生成的填光效果受渲染器精度和光照模型假设的影响。

---

## 216. Convolution Operator Network for Forward and Inverse Problems (FI-Conv): Application to Plasma Turbulence Simulations

**arXiv ID:** 2602.04287 | [PDF](https://arxiv.org/pdf/2602.04287v1)

**作者:** Xingzhuo Chen `[一作]` (Texas A&M Institute of Data Science), Ulisses Braga-Neto `[通讯]` (Texas A&M University)

**通讯引用:** 4940 | [OpenAlex ID](https://openalex.org/A5026990034)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 FI‑Conv 网络，能够对复杂时空 PDE（如 Hasegawa‑Wakatani 等离子体湍流模型）进行前向预测与逆向参数估计，并实现了单一模型适用于多参数、多时间尺度的预测与推断。

**💡 创新点**

创新点在于将 U‑Net 与 ConvNeXt V2 结合，并将演化时间与 PDE 参数直接嵌入网络，使模型在不重新训练权重的情况下即可完成长时预测、参数插值与逆向推断。

**🔧 技术方法**

使用了 U‑Net 结构、ConvNeXt V2 模块、硬边界约束、自动微分梯度下降、递归（autoregressive）预测等技术，并在训练中采用 MSE 损失和 AdamW 优化。

**📊 数据集**

使用自研 HW2D 计算器生成的 320 条二维湍流轨迹（四个参数随机采样），训练 240 条，测试 80 条，输入包括 vorticity、electrostatic potential、density、时间步长和四个 PDE 参数。

**📈 对比分析**

与标准 U‑Net、FNO‑8/16 进行比较，FI‑Conv 参数约 2.0×10⁶，MSE 最低 6.7×10⁻⁶，推理时间约 0.05 秒/样本；在逆问题中梯度下降收敛约 100 步，平均绝对误差 MAE ≤ 0.03，表现优于传统方法。

**⚠️ 局限性**

局限在于对极长时间步长会累积误差、对更大参数范围和更复杂的 MHD 系统的推广尚未验证，且训练数据量仍较大，需要高性能计算资源。

---

## 217. MiniRec: Data-Efficient Reinforcement Learning for LLM-based Recommendation

**arXiv ID:** 2602.04278 | [PDF](https://arxiv.org/pdf/2602.04278v1)

**作者:** Lin Wang `[一作]` (Hong Kong Polytechnic University), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 60270 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MiniRec 框架，利用奖励驱动的学习性评估、梯度方向对齐的代表性评估、样本多样性控制以及课程学习策略，对强化学习驱动的大语言模型推荐系统进行高效数据子集选择，从而显著降低训练成本。

**💡 创新点**

创新点包括：① 将 RL 奖励直接用作样本可学习性指标，避免传统损失/梯度误导；② 通过全局优化方向与样本梯度对齐度量代表性，突破传统数据覆盖无法体现“推理”关系的局限；③ 将多样性作为动态边际增益引入贪心子集构造；④ 结合课程调度，实现从易到难的渐进训练；⑤ 证明该框架在不同 LLM 架构上具有良好迁移性。

**🔧 技术方法**

技术手段包括：代理模型估计奖励；二阶梯度（Hessian‑Vector Product）求解代表性方向；多样性分数的最小距离度量；贪心子集选择；GRPO 强化学习框架；Gemma‑2‑2B‑IT、Qwen2.5‑3B‑Instruct 等 LLM；课程分组与排序。

**📊 数据集**

使用 Amazon 真实评论数据集：CDs & Vinyl 以及 Instruments，包含用户‑物品交互和文本评论。

**📈 对比分析**

与随机采样、K‑means、GraNd、EL2N、DEALRec 等六种基线在 1,024 样本下进行对比；MiniRec 在 NDCG@k、Hit Ratio@k 指标上与全量训练相当，并优于所有基线；训练时间下降 82%，显著提升数据效率。

**⚠️ 局限性**

局限性：① 依赖代理模型对奖励和梯度的近似，若代理与目标模型差异大可能失效；② 需要二阶梯度计算，对极大规模数据的计算开销仍不低；③ 参数 λ 对性能敏感，需要经验调优；④ 目前仅验证在推荐任务，尚未探测在其他 RL 领域的适用性。

---

## 218. KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing

**arXiv ID:** 2602.04268 | [PDF](https://arxiv.org/pdf/2602.04268v1)

**作者:** Siyu Jiang `[一作]` (Huazhong University of Science and Technology), Kun He `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4879 | [OpenAlex ID](https://openalex.org/A5033526822)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关、可插拔的方案，通过注意力行熵引导的自适应指数移动平均（EMA）平滑 KV‑Cache，来抑制多模态大型语言模型（MLLMs）在长文本生成过程中的幻觉现象。

**💡 创新点**

创新点包括：① 使用注意力行熵作为实时的“汇聚度”指标，快速识别容易诱发幻觉的汇聚标记；② 在 KV‑Cache 的 key/value 上应用自适应 EMA 平滑，而非直接平滑隐藏状态；③ 通过 FIFO 队列与百分位排名动态调整 EMA 係数，实现对高熵标记的精准抑制；④ 整体实现无需额外训练、可直接集成到现有 MLLMs。

**🔧 技术方法**

技术细节：
- 指数移动平均（EMA）平滑
- 注意力行熵计算与列和比较
- FIFO 队列与百分位排名用于自适应系数
- 在解码阶段对特定 decoder 层（如 3–31）进行 KV‑Cache 更新
- 基准评估使用 CHAIR、OPOPE、AMBER、Object HalBench 等数据集
- 与 VCD、OPERA、PAI、SPARC、MiddleLayer 等无训练对齐方法进行对比

**📊 数据集**

实验数据集：
- CHAIR（图像字幕幻觉）
- OPOPE（物体存在验证）
- AMBER（多场景综合评估）
- Object HalBench（GPT 辅助对象抽取）
- 评估模型：LLaVA‑1.5、MiniGPT‑4、InstructBLIP（各 7B 版本）

**📈 对比分析**

与五种无训练干预基线（VCD、OPERA、PAI、SPARC、MiddleLayer）对比，取得显著性能提升：
- CHAIR 上 C_HAIR_S 从 41.8 降至 18.2（≈56% 降幅），F1 从 77.5 提升至 79.2；
- 在 OPOPE、AMBER、Object HalBench 上亦实现最高或最优的幻觉指标，同时保持或提升真实物体覆盖率；
- PR 曲线与 –F1 交叉验证显示，本方法在保持高召回的同时显著提升精度，优于其他方法。

**⚠️ 局限性**

局限性：
- 仅在推理阶段进行平滑，无法从根本上改变模型的学习表征；
- 对关键超参数（如 λ_ref、队列长度）的敏感性，若设置不当可能导致过度抑制或不足抑制；
- 评估范围限定在已公开的几类 MLLM 和四大幻觉基准，尚未验证在更大规模或不同任务（如对话、视觉问答）的普适性；
- 仍可能出现少量真实物体被误抑制，尤其在长句生成或复杂图像中。

---

## 219. From Dead Neurons to Deep Approximators: Deep Bernstein Networks as a Provable Alternative to Residual Layers

**arXiv ID:** 2602.04264 | [PDF](https://arxiv.org/pdf/2602.04264v1)

**作者:** Ibrahim Albool `[一作]` (University of California), Yasser Shoukry `[通讯]` (University of California)

**通讯引用:** 1838 | [OpenAlex ID](https://openalex.org/A5019844918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出使用 Bernstein 多项式作为激活函数的深度网络 DeepBern-Nets，构建了无需残差连接的深层网络；

**💡 创新点**

通过证明 Bernstein 激活在任何区间内梯度有严格下界并实现指数级逼近误差下降，解决了梯度消失和逼近效率低的问题；

**🔧 技术方法**

利用 Bernstein 多项式、增量式软正则化的系数重参数化、批量归一化、AdamW 训练及梯度锚定策略；

**📊 数据集**

在 HIGGS（高能物理模拟）和 MNIST（手写数字识别）数据集上进行实验；

**📈 对比分析**

与 ReLU、ResReLU、GELU、SELU、LeakyReLU 等传统激活及残差网络对比，DeepBern-Nets 在 dead‑neuron 比例降至 <5%、梯度保持更稳定，并在 HIGGS 上取得更高 AUC、在 MNIST 上达到或超过 ResNet 的准确率，且同样参数量可实现更浅的网络；

**⚠️ 局限性**

仍需依赖合适的输入区间与批归一化，当前仅在全连接架构和小规模任务上验证，计算多项式激活的开销和在更大规模网络与视觉任务中的可扩展性尚未彻底评估。

---

## 220. Depth-Guided Metric-Aware Temporal Consistency for Monocular Video Human Mesh Recovery

**arXiv ID:** 2602.04257 | [PDF](https://arxiv.org/pdf/2602.04257v1)

**作者:** Jiaxin Cen `[一作]` (Sun Yat-sen University), Baoquan Zhao `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1209 | [OpenAlex ID](https://openalex.org/A5102379113)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个深度引导的多模态框架，用于在单目视频中恢复时空一致且度量一致的3D人体网格。

**💡 创新点**

通过深度引导多尺度融合、度量一致的骨骼初始化 (D-MAPS) 与运动-深度对齐的细化 (MoDAR) 三大模块，系统性克服了深度歧义、尺度漂移与时间抖动问题。

**🔧 技术方法**

采用 Depth Anything 深度估计网络、RGB 与深度特征的门控融合、骨骼长度统计校准、跨模态注意力机制及因果时间滤波等技术。

**📊 数据集**

在 COCO、MPII 等 2D 数据集以及 Human3.6M、MPI-INF-3DHP、3DPW 三大 3D 数据集上进行训练与评估。

**📈 对比分析**

与 VIBE、GLoT、PMCE、ARTS 等现有方法在 3DPW、Human3.6M、MPI-INF-3DHP 等基准上进行 MPJPE、PA-MPJPE、MPVPE 与加速度误差等指标比较，取得了最低误差、最高精度的领先表现。

**⚠️ 局限性**

依赖深度估计的质量，深度噪声或校准误差仍可能影响结果，并且在极端快速运动或严重遮挡的场景中仍存在细微误差。

---

## 221. Towards Next-Generation SLAM: A Survey on 3DGS-SLAM Focusing on Performance, Robustness, and Future Directions

**arXiv ID:** 2602.04251 | [PDF](https://arxiv.org/pdf/2602.04251v1)

**作者:** Li Wang `[一作]` (Beijing Institute of Technology), Rong Fu `[通讯]` (Shanghai AI laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了3D Gaussian Splatting（3DGS）与SLAM融合的现状，重点评估其渲染质量、跟踪精度、重建速度与内存消耗，并探讨在运动模糊与动态环境下的鲁棒性提升方法。

**💡 创新点**

提出了按性能维度划分的评价框架和四大优化方向的系统性分类，首次将多种新兴技术与数据集统一汇总，形成可直接对比的技术基准与未来研究路线图。

**🔧 技术方法**

主要技术包括3DGS显式几何表示、可微分投影与光栅化、图优化与束平差、语义/几何一致性约束、动态物体分离、事件相机融合、物理属性嵌入等。

**📊 数据集**

使用了包括Replica、TUM RGB-D、ScanNet、Waymo、DTU MVS、Blender、UrbanScene3D、MultiReplica等公开数据集进行实验与性能对比。

**📈 对比分析**

通过对PSNR、SSIM、LPIPS、ATE、FPS、内存占用等指标的量化比较，展示了不同方法在相同数据集上的优势与劣势，标注最佳/次佳/第三佳结果，并给出具体数值对比。

**⚠️ 局限性**

局限性在于主要为综述性工作，缺乏统一基准测试框架，部分最新方法未被涵盖；对不同硬件环境的泛化评估不足；对极端环境（低纹理、高动态、低光）下的鲁棒性仍待进一步验证。

---

## 222. Empirical-MCTS: Continuous Agent Evolution via Dual-Experience Monte Carlo Tree Search

**arXiv ID:** 2602.04248 | [PDF](https://arxiv.org/pdf/2602.04248v1)

**作者:** Hao Lu `[一作]` (JianChengXingYun Technology Co., Ltd.), Ningxin Zhu `[通讯]` (JianChengXingYun Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Empirical-MCTS框架，将传统MCTS与持续经验积累相结合，实现无梯度更新的在线学习；

**💡 创新点**

创新点在于双循环机制：局部搜索中的Pairwise‑Experience‑Evolutionary Meta‑Prompting（PE‑EMP）动态演化meta‑prompt；全局记忆优化Agent通过原子操作不断更新经验库；

**🔧 技术方法**

采用Monte Carlo Tree Search、pairwise经验演化meta‑prompt、增强Borda计数的混合偏好模型、动态记忆优化与自适应提示生成技术；

**📊 数据集**

在AIME25、MathArena Apex、ARC‑AGI‑2三个高难度推理基准上评估，并使用DeepSeek‑V3.1‑Terminus、gpt‑oss‑120b、Gemini 3 Flash等前沿模型；

**📈 对比分析**

与ICL、ReAct、Best‑of‑N、FLEX、LLaMA‑Berry、Training‑Free GRPO等基线对比，AIME25上达到73.3%（相较63.3%提升），MathArena Apex上从0%提升至4.17%，Gemini 3 Flash在成本仅为5.24$时实现35.42%准确率，形成成本‑性能Pareto前沿；

**⚠️ 局限性**

依赖模型对推理步骤的准确验证，易因误导性经验导致meta‑prompt恶化；缺乏鲁棒性检查与不确定性量化，难以处理极其复杂任务。

---

## 223. Disentangling Causal Importance from Emergent Structure in Multi-Expert Orchestration

**arXiv ID:** 2602.04291 | [PDF](https://arxiv.org/pdf/2602.04291v1)

**作者:** Sudipto Ghosh `[一作]` (Indian Institute of Technology Delhi), Tanmoy Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 4981 | [OpenAlex ID](https://openalex.org/A5046521217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可解释的多专家系统协调框架，能够分离专家交互、执行顺序与因果归因。

**💡 创新点**

通过将协调过程建模为可分析的计算，揭示路由频率与因果重要性不一致的现象，并提供梯度归因与关系重要性对比。

**🔧 技术方法**

使用可学习路由器、条件转移矩阵、Gumbel‑Softmax、梯度归因与结构熵等技术。

**📊 数据集**

在 MMLU、HumanEval、GSM8K 等任务上实验，使用十个 8B 同构专家或混合规模专家池。

**📈 对比分析**

与 MetaGPT、统一路由等基线对比，获得更高准确率/Pass@1 并显著减少模型调用次数，证明更高效、更具可解释性的协调。

**⚠️ 局限性**

局限性包括异构专家组表现不稳定、梯度归因对模型规模与训练成本敏感，以及对特定任务可解释性依赖性较高。

---

## 224. From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning for Embodied Agents

**arXiv ID:** 2602.04326 | [PDF](https://arxiv.org/pdf/2602.04326v1)

**作者:** SeungWon Seo `[一作]` (Korea University), HyeongYeop Kang `[通讯]` (Korea University)

**通讯引用:** 230 | [OpenAlex ID](https://openalex.org/A5011229651)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 Planner-Composer-Evaluator 框架实现基于 LLM 的不确定性感知规划

**💡 创新点**

通过将 LLM 推理中隐含的假设结构化为决策树并进行评估，显著减少通信开销

**🔧 技术方法**

利用大型语言模型、链式推理、决策树构造与评估算法

**📊 数据集**

在 C-WAH 和 TDW-MAT 两个多智能体家居任务基准上进行评估

**📈 对比分析**

相较于 CoELA、REVECA、CaPo、CoTS 等基线，PCE 在成功率、任务效率提升，且 token 使用保持相近

**⚠️ 局限性**

对高维动态环境、跨域通用性尚未充分验证，且模型对异常假设的判定依赖 LLM 生成质量

---

## 225. Efficient Equivariant High-Order Crystal Tensor Prediction via Cartesian Local-Environment Many-Body Coupling

**arXiv ID:** 2602.04323 | [PDF](https://arxiv.org/pdf/2602.04323v1)

**作者:** Dian Jin `[一作]` (Hong Kong Polytechnic University), Xiaoming Tao `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 27553 | [OpenAlex ID](https://openalex.org/A5087298250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了CEITNet，一种基于笛卡尔环境交互的高阶晶体张量预测模型。

**💡 创新点**

创新点在于将不变编码与等变张量构造解耦，通过多通道笛卡尔本征基构建局部环境张量，并引入可学习的通道交互矩阵实现高阶张量的高效表达。

**🔧 技术方法**

所采用技术包括基于ComformerConv的不可变消息传递、笛卡尔几何基的构建、通道化局部环境张量、可学习通道交互头以及加权池化和对称性校正模块。

**📊 数据集**

实验使用了JARVIS-DFT数据库中的GMTNet数据集，分别包含电介质、压电和弹性四阶张量。

**📈 对比分析**

与ETGNN、GMTNet和GeoCTP对比，CEITNet在Fnorm和EwT指标上均取得最佳或最接近最佳结果，同时参数量仅为0.6M，推理速度提升至GMTNet的1/4左右。

**⚠️ 局限性**

局限性包括需要为不同张量类型手工设计基和耦合模板，通道交互目前仅覆盖三体信息，扩展至更高阶相互作用会带来计算和过拟合风险。

---

## 226. JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for In-the-Wild Monocular Reconstruction

**arXiv ID:** 2602.04317 | [PDF](https://arxiv.org/pdf/2602.04317v1)

**作者:** Zihan Lou `[一作]` (Wuhan University), Jing Zhang `[通讯]` (Wuhan University)

**通讯引用:** 26148 | [OpenAlex ID](https://openalex.org/A5100345341)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了JOintGS框架，联合优化摄像机轨迹、人体姿态与3D高斯模型，实现从粗糙初始化开始的高质量可动画化三维人像重建。

**💡 创新点**

创新点在于通过前景-背景显式分离形成闭环互补的优化机制：背景高斯提供多视角几何约束校准摄像机，校准后的摄像机改进人体姿态，改进的姿态进一步提升前景-背景分离，从而实现端到端的协同优化。

**🔧 技术方法**

技术上结合了3D高斯点渲染、SMPL皮肤权重拉伸、临时偏移网络和颜色残差网络，使用可微分渲染与多项正则化实现联合学习；同时利用RANSAC对齐尺度、SAM人类分割、以及多阶段训练策略。

**📊 数据集**

在两个人体与场景混合的野外数据集NeuMan和EMDB上进行评估。

**📈 对比分析**

与现有方法（如HUGS、GaussianAvatar等）对比，JOintGS在NeuMan数据集上PSNR提升约2.2dB，并在噪声初始化下仅下降0.9dB，展示出更强的鲁棒性和更高的重建质量。

**⚠️ 局限性**

局限在于依赖SMPL体形模型，导致手部、面部等细节的表现不够精细，且对极端光照或快速运动的捕捉仍有挑战。

---

## 227. Beyond Static Cropping: Layer-Adaptive Visual Localization and Decoding Enhancement

**arXiv ID:** 2602.04304 | [PDF](https://arxiv.org/pdf/2602.04304v1)

**作者:** Zipeng Zhu `[一作]` (Harbin Institute of Technology), Lin Gui `[通讯]` (King's College London)

**通讯引用:** 7077 | [OpenAlex ID](https://openalex.org/A5062168574)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于视觉激活量（VAQ）和令牌激活量（VAT）的动态层选择与对比解码框架LASER，用于提升大规模视觉语言模型的视觉定位与推理准确率。

**💡 创新点**

创新点在于通过对比注意力量化查询相关视觉激活，动态选择每个实例最合适的层进行定位与增强，并利用对比解码结合VAT抑制无视觉依据的答案。

**🔧 技术方法**

采用对比注意力、层级敏感性分析、视觉裁剪、对比解码以及注意力聚合等技术。

**📊 数据集**

在RefCOCO/RefCOCO+/RefCOCOg三大视觉定位数据集以及POPE、TextVQA、A-OKVQA三大视觉问答基准上进行评测。

**📈 对比分析**

与静态层裁剪（ViCrop）、对比解码（VCD）以及原始注意力等基线对比，LASER在三大问答基准上均显著提升准确率，尤其在复杂推理任务上提升明显；在定位任务上注意力聚合比率最高。

**⚠️ 局限性**

局限包括需要额外两次注意力推断与对比解码导致推理延时；仅在固定视觉编码器结构上验证，未针对不同视觉编码器或更大规模模型进行测试；对异常遮挡或极端场景的鲁棒性尚待验证。

---

## 228. Guided Verifier: Collaborative Multimodal Reasoning via Dynamic Process Supervision

**arXiv ID:** 2602.04290 | [PDF](https://arxiv.org/pdf/2602.04290v1)

**作者:** Lingzhuang Sun `[一作]` (University of Chinese Academy of Sciences), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 14684 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种引导式验证器（Guided Verifier）框架，将多模态推理从单向生成转变为双模型协作闭环过程，实现动态过程监督。

**💡 创新点**

创新点在于通过可训练的轻量级验证器在推理过程中实时监督并纠正推理步骤，同时构建面向错误检测的CoRe合成数据集，并将其与Guided‑GRPO算法结合进行强化学习训练。

**🔧 技术方法**

使用技术包括多模态大语言模型、GRPO强化学习、验证器的SFT训练、过程级监督、Hallucination标注、密集奖励设计和动态闭环交互。

**📊 数据集**

主要使用的数据集有CoRe（约3k条合成推理轨迹）以及Geometry‑3k用于RL训练，评测基准为MathVista、MathVerse和MMMU。

**📈 对比分析**

与多种开源与专有模型对比，8B参数模型在上述基准上达到或逼近GPT‑4o/ Gemini‑2.5‑Pro的表现，显著优于同级别及多数大型开源模型。

**⚠️ 局限性**

局限性包括需要额外的验证器训练和对话合成数据，推理时会产生额外的交互开销；对非数学类推理任务的通用性尚未充分验证。

---

## 229. Proxy Compression for Language Modeling

**arXiv ID:** 2602.04289 | [PDF](https://arxiv.org/pdf/2602.04289v1)

**作者:** Lin Zheng `[一作]` (University of Hong Kong), Lingpeng Kong `[通讯]` (University of Hong Kong)

**通讯引用:** 30559 | [OpenAlex ID](https://openalex.org/A5088517824)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出代理压缩（proxy compression）训练方案，使语言模型在训练时同时使用原始字节和压缩视图，最终以纯字节接口进行推理；

**💡 创新点**

创新点在于将压缩器仅作为训练代理，打破模型与压缩器的耦合，同时实现跨表示的强转移，证明了基于分词器和神经算术编码的代理压缩有效，而通用 gzip 失效；

**🔧 技术方法**

技术上采用混合表示训练（混合原始字节与压缩序列、使用 sentinel 标记）、可选的上下文翻译配对、基于分词器、神经算术编码和 gzip 的代理压缩；

**📊 数据集**

使用 RefineCode 代码语料库（Python 子集约 270GB，完整 3.3TB）训练，并在 HumanEval、MBPP 及其 Plus 变体上评估；

**📈 对比分析**

与纯字节模型和基于分词器的基线在相同 FLOPs 预算下对比；代理压缩模型在字节推理下显著优于纯字节基线，且在更大规模时可匹敌甚至超越分词器基线；

**⚠️ 局限性**

局限性在于仅验证于代码领域；对压缩率、转移强度与计算效率的完整权衡尚未刻画；代理压缩仍是数据预处理方式，未在模型架构层面集成。

---

## 230. Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning

**arXiv ID:** 2602.04284 | [PDF](https://arxiv.org/pdf/2602.04284v1)

**作者:** Yansong Ning `[一作]` (Hong Kong University of Science and Technology), Hao Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 12477 | [OpenAlex ID](https://openalex.org/A5100458870)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Agent-Omit框架，使大型语言模型代理能够自适应地忽略冗余推理思考与环境观测，从而在多轮交互中显著降低令牌消耗并提升效率。

**💡 创新点**

①基于量化分析证明不同轮次的思考与观测对有效性与效率的贡献差异；②构造冷启动省略数据并在此基础上进行增量强化学习；③设计双采样策略与省略奖励，证明省略策略偏差受KL散度上界约束。

**🔧 技术方法**

冷启动全参数微调（SFT）+合成省略样本、双采样（全轨迹/部分轨迹）强化学习、省略奖励、GRPO优化、Token计数与观测遮蔽技术。

**📊 数据集**

五大基准：DeepSearch、WebShop、TextCraft、BabyAI、SciWorld（AgentGym-RL环境），以及构造的省略数据集。

**📈 对比分析**

与七大前沿LLM代理（DeepSeek、o3、Qwen3系列）及三类高效代理方法（思考/观测压缩、摘要）进行对比；Agent-Omit-8B-RL在多数任务上获得最高Pass@1并实现平均Token数下降约30‑50%，实现最佳有效性‑效率权衡。

**⚠️ 局限性**

局限性包括：省略数据合成仍需人工/脚本支持，模型规模受GPU资源限制；省略策略主要在预先定义的交互环境中验证，泛化到更复杂或动态场景尚需进一步探索。

---

## 231. ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation

**arXiv ID:** 2602.04279 | [PDF](https://arxiv.org/pdf/2602.04279v1)

**作者:** Jiarui Jin `[一作]` (Peking University), Shenda Hong `[通讯]` (Peking University)

**通讯引用:** 4125 | [OpenAlex ID](https://openalex.org/A5080648149)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了 ECG‑R1——一种面向可靠心电图（ECG）解读的多模态大语言模型（MLLM），能够在缺失模态或不同输入形式时保持一致、可靠的诊断输出。

**💡 创新点**

创新点：
1) Protocol‑Guided Instruction Data Generation：利用临床心电图解读手册中的五个阶段和阈值，生成结构化、低幻觉的训练语料；
2) Modality‑Decoupled Architecture + Interleaved Modality Dropout (IMD)：将时间序列和图像分别编码、投射到 LLM 空间，并在训练时随机丢弃或交换模态，提升鲁棒性与跨模态一致性；
3) Reinforcement Learning with ECG Diagnostic Evidence Rewards (EDER)：在 RL 阶段为每一步证据提供奖励，强化循证推理并降低幻觉。

**🔧 技术方法**

核心技术：
- 双编码器架构（Qwen3‑VL‑8B 视觉编码器 + ECG‑CoCa 时序编码器）
- 轻量级投射器将模态嵌入映射至 LLM
- IMD 训练策略（随机模态丢失 + 交换顺序）
- RL 采用 DAPO 与 step‑wise evidence reward（EDER）
- 基于 FeatureDB 提取 ECG 生理特征，ProtocolGuider 构造指令
- 使用 DeepSeek‑V3.1‑Terminus 提取证据并评估结果

**📊 数据集**

使用数据集：
- MIMIC‑IV‑ECG（构建 3.0 万条协议驱动指令）
- ECG‑Grounding（公开的 ECG‑解读数据）
- ECGInstruct（公开 ECG 指令数据）
- ECG‑image‑kit（从原始信号合成图像）
- FeatureDB（提取 14 个信号特征序列）
- 通过 30,000 条协议生成样本与 3,948 条 RL 训练样本组合形成完整训练集。

**📈 对比分析**

与多类基线模型对比：
- 公开、开源与医疗领域 MLLM（Gemini‑3‑Pro、GPT‑5.1、MiMo‑VL‑7B、GLM‑4.1V‑9B、Qwen3‑VL‑8B、InternVL3、MiniCPM‑V、MedVLM‑R1 等）
- ECG 专用 MLLM（PULSE、GEM）
- 评价指标：诊断准确率、分析完整性、相关性、导联证据有效性、ECG 特征 grounding、循证推理、临床诊断忠实度、鲁棒性与一致性、以及心脏病专家手工评估。
- 结果：ECG‑R1 在诊断准确率上达 80.3%（SFT）/80.3%（RL），显著高于 GEM 的 74.7%；在其它指标均有提升；在缺失模态条件下，性能下降幅度远小于 GEM；心脏病专家评估中，ECG‑R1 在可靠性、可用性与整体满意度均优于基线。

**⚠️ 局限性**

局限性：
- 仍存在一定比例的幻觉和事实错误，需临床专家复核；
- 训练数据主要来自公开或内部数据库，可能不覆盖所有临床场景；
- 目前仅在实验室环境验证，缺乏真实临床前瞻性验证；
- 对高复杂或罕见心电图的解释能力尚未充分评估；
- 作为研究工具，不能直接用于临床决策，仍需合规监管与治理。

---

## 232. SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization

**arXiv ID:** 2602.04271 | [PDF](https://arxiv.org/pdf/2602.04271v1)

**作者:** Lifan Wu `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 17899 | [OpenAlex ID](https://openalex.org/A5100648981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过骨架驱动的4D高斯模型实现可编辑的动态3D生成

**💡 创新点**

引入层次化骨架驱动的运动表示，分离稀疏刚性运动与细粒度非刚性运动，支持直接骨架编辑

**🔧 技术方法**

采用3D高斯生成、线性混合蒙皮（LBS）、hexplane+MLP细化、SDS损失、UniRig骨架提取、差分光栅化等技术

**📊 数据集**

使用Consistent4D数据集（12份真实、12份合成视频）

**📈 对比分析**

与Consistent4D、STAG4D、4DGen、DreamGaussian4D等基线对比，CLIP 0.923、LPIPS 0.125、FVD 847.8，性能优于基线

**⚠️ 局限性**

依赖骨架提取质量，无法处理多物体场景，对无骨架结构物体效果有限

---

## 233. Multi Objective Design Optimization of Non Pneumatic Passenger Car Tires Using Finite Element Modeling, Machine Learning, and Particle swarm Optimization and Bayesian Optimization Algorithms

**arXiv ID:** 2602.04277 | [PDF](https://arxiv.org/pdf/2602.04277v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 234. Training A Foundation Model to Represent Graphs as Vectors

**arXiv ID:** 2602.04244 | [PDF](https://arxiv.org/pdf/2602.04244v1)

**作者:** Qi Feng `[一作]` (Chinese University of Hong Kong), Jicong Fan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并训练了一种图形基础模型GraphVec‑FM，能够将任意图转换为低维向量并在图级任务（图分类、聚类）中发挥作用；

**💡 创新点**

主要创新包括：① 基于多图构造的特征对齐方法，通过全局相似图与SVD实现跨域一致节点嵌入；② 密度最大化均值对齐算法解决SVD符号/排序不确定性；③ 引入多层参考分布模块替代传统池化，保留节点嵌入信息；④ 给出理论泛化误差上界。

**🔧 技术方法**

技术手段包括：GIN＋图Transformer骨干网络；Gaussian kernel构造全局相似图与SVD得到节点嵌入；对比学习（监督/无监督）损失；MMD参考分布相似度；Nyström近似提升大规模可扩展性；密度最大化对齐算法。

**📊 数据集**

使用的实验数据集：TUDataset五个生物化学图（ENZYMES、DD、NCI1、NCI109、Mutagenicity），四个社交网络图（COLLAB、REDDIT‑BINARY、IMDB‑BINARY、IMDB‑MULTI），三组计算机视觉图（Letter‑med、COIL‑RAG、Cuneiform），以及大规模数据集COLOR‑3和reddit_threads。

**📈 对比分析**

在few‑shot图分类中，GraphVec‑FM在in‑dataset和cross‑dataset两种场景下均优于或与现有最佳基线（EdgePrompt、GPF、GFT、BRIDGE、RiemannGFM等）竞争；在社交网络/视觉数据上实现最高准确率；在图聚类任务中获得最高ACC/NMI/ARI；实验表明模型具有很强的跨域泛化与可扩展性。

**⚠️ 局限性**

局限性：目前无法实现零样本学习，未利用语言模型或文本信息；对无属性图的SVD近似可能导致信息损失；虽然采用Nyström和小批量构造，但在极大图集构造仍存在计算与存储开销。

---

## 235. Multi-Integration of Labels across Categories for Component Identification (MILCCI)

**arXiv ID:** 2602.04270 | [PDF](https://arxiv.org/pdf/2602.04270v1)

**作者:** Noga Mudrik `[一作]` (Johns Hopkins University), Adam S. Charles `[通讯]` (Johns Hopkins University)

**通讯引用:** 1235 | [OpenAlex ID](https://openalex.org/A5062177768)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种名为MILCCI的多试验多标签时间序列分析方法，能够在稀疏低秩空间中自动识别可解释的组件，并通过标签信息捕捉跨试验的变异性。

**💡 创新点**

创新点在于让组件随标签类别微调其构成（通过标签相似度约束），从而在同一标签下保持一致、在不同标签间实现平滑过渡，并在一次迭代中同时 disentangle 多类别标签对数据的贡献。

**🔧 技术方法**

技术上结合稀疏张量分解、Laplace 先验、LASSO 更新、标签相似度正则化、时间平滑与去相关约束，并采用迭代优化求解组件矩阵与试验轨迹。

**📊 数据集**

在合成数据、美国州级投票记录、维基百科页面访问（代理/平台/语言三类标签）以及小鼠多区神经元单细胞记录四个真实数据集上进行评估。

**📈 对比分析**

与 Tucker、PARAFAC、非负PARAFAC、SVD、SiBBlInGS、sliceTCA 等传统方法对比，MILCCI 在组件/轨迹的相关度、重构误差以及解释性方面均取得显著提升。

**⚠️ 局限性**

局限包括仅适用于线性假设、对尺度不确定性和超参数敏感、缺乏非线性扩展、对极大规模数据需并行化处理以及对稀疏结构的依赖。

---

## 236. GeneralVLA: Generalizable Vision-Language-Action Models with Knowledge-Guided Trajectory Planning

**arXiv ID:** 2602.04315 | [PDF](https://arxiv.org/pdf/2602.04315v1)

**作者:** Guoqing Ma `[一作]` (Chinese Academy of Sciences Institute of Automation), Hao Tang `[通讯]` (Peking University)

**通讯引用:** 52824 | [OpenAlex ID](https://openalex.org/A5062247330)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 GeneralVLA，一种分层的视觉‑语言‑动作模型，能够在零样本场景下完成多种机器人操控任务，并自动生成高质量的仿真训练数据。

**💡 创新点**

创新点包括：① 通过高层 Affordance Segmentation Module (ASM) 结合 VLM 与 SAM 精确识别二维关键点；② 中层 3DAgent 利用 LLM 对 3D 点进行空间推理与轨迹规划，并通过 Knowledge Bank 存储跨任务经验；③ 低层 3D‑aware 控制策略与 HGM 结合多模态信息实现精准把控；④ 整体分层架构解耦感知、规划与执行，显著提升零样本泛化能力。

**🔧 技术方法**

使用技术：VLM（如 LLaVA）、SAM、LLM（如 Deepseek R1）进行文本与图像处理；深度映射投影得到 3D 点；LLM 作为轨迹规划器；Knowledge Bank 进行检索、构建与整合；HGM 结合 RGB、深度与点云进行抓取姿态预测；PyRep/CoppeliaSim 进行仿真；RVT‑2 模型用于行为克隆。

**📊 数据集**

主要数据集：RLBench（14 个仿真任务）、CoppeliaSim + PyRep 交互、RGB‑D 视角；对比基准包括 VoxPoser、CAP、Scaling‑up；还使用了真实环境（Agilex‑2.0 Piper 与 Intel RealSense L515）进行 4 项零样本实验。

**📈 对比分析**

与 VoxPoser、CAP、Scaling‑up 的对比显示：GeneralVLA 在 14 个任务中成功率最高，10 项任务超越基线；生成的演示数据训练的 RVT‑2 策略在 12 项任务上平均比人类演示仅低 2.7%，而基线数据训练的策略表现显著差距；单任务成功率表明 GeneralVLA 在多对象、多阶段任务中的优势显著。

**⚠️ 局限性**

局限性：① 依赖高质量的 2D Affordance 检测，对精细姿态估计仍受限；② 3D 路径规划和执行的计算成本较高，尤其在多任务重规划时；③ 对于需要极高精度或动态适应的细粒度操控任务，仍表现不足；④ 目前仅在仿真与少量真实任务验证，需进一步扩展至更复杂的现实环境。

---

## 237. Data Agents: Levels, State of the Art, and Open Problems

**arXiv ID:** 2602.04261 | [PDF](https://arxiv.org/pdf/2602.04261v1)

**作者:** Yuyu Luo `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 6519 | [OpenAlex ID](https://openalex.org/A5062243169)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了从 L0 到 L5 的数据代理层级分类，并系统梳理了各层级在数据生命周期中的应用场景。

**💡 创新点**

创新点在于将数据代理与自动驾驶等领域的六阶自动化标准对齐，形成了层级化、生命周期化的框架，并指出了 L2→L3 及 L3→L4 的关键跃迁。

**🔧 技术方法**

技术上主要借助大型语言模型、工具调用、规划与执行、记忆管理以及多代理协作等方法，并对现有 L0–L2 系统进行整理。

**📊 数据集**

该工作并未基于单一数据集，而是综述了多领域（关系数据库、数据湖、BI、ML 服务）下的公开系统与案例。

**📈 对比分析**

本文通过对比 L0–L2 代表性系统与 Proto‑L3 系统的功能和局限性进行定性评估，未给出量化性能指标；其价值在于提供研究路线图。

**⚠️ 局限性**

局限在于缺乏实证实验与客观基准，层级划分仍处于概念化阶段，尚未验证在实际生产环境中的可靠性与治理能力。

---

## 238. Decoupled Hierarchical Distillation for Multimodal Emotion Recognition

**arXiv ID:** 2602.04260 | [PDF](https://arxiv.org/pdf/2602.04260v1)

**作者:** Yong Li `[一作]` (Southeast University), Cuntai Guan `[通讯]` (Nanyang Technological University)

**通讯引用:** 27607 | [OpenAlex ID](https://openalex.org/A5031778999)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Decoupled Hierarchical Multimodal Distillation (DHMD)框架，用以解决多模态情感识别中的模态异质性问题。

**💡 创新点**

创新点在于将每种模态特征拆分为模态无关（同质）与模态专属（异质）两部分，并通过两阶段知识蒸馏（图结构蒸馏与字典匹配）实现自适应跨模态信息迁移。

**🔧 技术方法**

采用自回归特征解耦、边距损失、软正交约束、图蒸馏单元 (GD-Unit) 与跨模态字典匹配 (DM) 等技术，结合Transformer关注机制提升分布对齐。

**📊 数据集**

在四个公开数据集上进行评测：CMU-MOSI、CMU-MOSEI、UR-FUNNY 与 MUStARD。

**📈 对比分析**

与现有方法比较，DHMD 在 7 类精度、二分类精度、F1、MAE、相关系数等指标均优于或与最先进方法持平，CMU-MOSI 上 ACC_7 提升 1.3%/2.4%，CMU-MOSEI 上 ACC_7 提升 1.3%/1.9%。

**⚠️ 局限性**

局限在于尚未与主流多模态基础模型（如大型预训练模型）结合，未来可进一步提升知识迁移效果。

---

## 239. AppleVLM: End-to-end Autonomous Driving with Advanced Perception and Planning-Enhanced Vision-Language Models

**arXiv ID:** 2602.04256 | [PDF](https://arxiv.org/pdf/2602.04256v1)

**作者:** Yuxuan Han `[一作]` (Harbin Institute of Technology), Yunjiang Lou `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 1958 | [OpenAlex ID](https://openalex.org/A5087749601)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 AppleVLM，一种集成视觉、语言和规划信息的端到端自动驾驶模型，能够在 CARLA 仿真和真实 AGV 环境中实现鲁棒驾驶。

**💡 创新点**

创新点包括：① 使用变形注意力的视觉编码器融合多视角时序图像与点云；② 引入规划策略编码器提供 BEV 轨道信息，减少语言偏差；③ 通过 Corner‑Case 数据和 Chain‑of‑Thought (CoT) 训练 Fine‑tune VLM，提升跨域泛化能力。

**🔧 技术方法**

采用的技术包括：变形 Transformer、Q‑Former 融合、多模态编码器、LLaVA/Janus Pro 视觉‑语言模型、CoT 训练、LQR 控制器、CARLA 仿真与 AGV 实地部署。

**📊 数据集**

使用的数据集涵盖 CARLA Longest6、LangAuto 基准、CODA‑LM、DriveLM、nuScenes 以及真实 AGV 传感器采集的数据。

**📈 对比分析**

通过与 Transfuser、UniAD、LMDrive 等现有模型在 DS、RC、RC_strict、IS 等指标下进行闭环仿真与真实测试对比，AppleVLM 在所有主要指标上均优于同类视觉或 VLM 模型，尤其在 RC_strict 与长距离驾驶表现显著提升。

**⚠️ 局限性**

局限性包括：仍依赖 VLM 的语言表达能力，规划编码器需要 BEV 先验；对极端恶劣天气、灾害场景或完全无感知环境的鲁棒性尚未充分验证；部署仍受限于硬件计算与传感器配置的兼容性。

---

## 240. Scaling Agentic Verifier for Competitive Coding

**arXiv ID:** 2602.04254 | [PDF](https://arxiv.org/pdf/2602.04254v1)

**作者:** Zeyao Ma `[一作]` (Renmin University of China), Binyuan Hui `[通讯]` (Alibaba Group)

**通讯引用:** 1001 | [OpenAlex ID](https://openalex.org/A5080628250)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种 Agentic Verifier，利用多轮交互和执行环境主动生成能区分候选程序的判别性测试输入，从而提升竞争性编程代码的选取准确率。

**💡 创新点**

创新点在于把测试输入生成视为多轮交互任务，结合大规模数据合成、拒绝微调和代理强化学习，训练出能够主动寻找高判别性输入的验证器，显著优于随机或模型生成的测试用例。

**🔧 技术方法**

采用的大语言模型（Qwen3 系列）作为代理，工具调用式交互执行，Rejection fine‑tuning、GRPO 强化学习，以及自动合成测试用例与输入验证器。

**📊 数据集**

使用的基准数据集包括 USACO、LiveCodeBench、OJBench、ICPC‑Eval、CodeForces，以及从公开 OJ 抓取并合成的题目与参考解。

**📈 对比分析**

在 Best@k（k=8,64）评估中，与随机挑选、评分模型、MBR‑Exec、CodeRM、Random Generator 等基线相比，Agentic Verifier 在所有基准上均实现 10–15% 的绝对提升，且在难度更高的数据集上效果更为显著。

**⚠️ 局限性**

限制在于仍需依赖可执行环境与题目约束的准确性，对特殊判题或非标准输入可能失效；此外仅处理输入‑only 验证，仍需 benchmark 作为参考，无法直接获得真实正确性标签。

---

## 241. ACIL: Active Class Incremental Learning for Image Classification

**arXiv ID:** 2602.04252 | [PDF](https://arxiv.org/pdf/2602.04252v1)

**作者:** Aditya R. Bhattacharya `[一作]` (Florida State University), Shayok Chakraborty `[通讯]` (Florida State University)

**通讯引用:** 3374 | [OpenAlex ID](https://openalex.org/A5101153180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对类增量学习的主动学习框架ACIL，能够在每个学习阶段只对少量未标记样本进行标注并将其作为样本集传播到后续阶段，从而兼顾防止灾难性遗忘与降低标注成本。

**💡 创新点**

创新点在于将主动学习的置信度与多样性准则引入增量学习的样本挑选，并设计了按类别比例分配标注预算的策略，使得每个类的代表性样本均被保留。

**🔧 技术方法**

采用基于特征嵌入的加权k-means聚类进行样本分组，并用交叉熵与知识蒸馏相结合的损失函数进行模型训练；同时使用伪标签和信息熵评估不确定性。

**📊 数据集**

在六个视觉数据集上评估：MNIST、SVHN、CIFAR-10、CIFAR-100、COIL 与 Tiny ImageNet。

**📈 对比分析**

与三种主动学习基线（随机、Coreset、BADGE）以及四种增量学习基线（Finetuning、iCaRL、Rainbow、GDumb）对比，ACIL在保持近乎相同或更优的准确率的同时，标注样本数显著减少（比传统CIL低约4倍），优于主动学习基线且与顶尖CIL方法相当。

**⚠️ 局限性**

局限性包括对预算大小敏感，需要预先设定固定的样本集容量，且实验仅覆盖分类任务，未验证在回归或更复杂场景下的鲁棒性。

---

## 242. DementiaBank-Emotion: A Multi-Rater Emotion Annotation Corpus for Alzheimer's Disease Speech (Version 1.0)

**arXiv ID:** 2602.04247 | [PDF](https://arxiv.org/pdf/2602.04247v1)

**作者:** Cheonkam Jeong `[一作]` (University of California), Adeline Nyamathi `[通讯]` (Purdue University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

创建并发布 DementiaBank-Emotion 语料库，标注 AD 患者和健康对照的 Ekman 六种基本情绪及中性标签，分析情感分布与声学特征；

**💡 创新点**

首次提供多评标注的 AD 语音情感资源，配套标注指南和校准工作坊，并揭示 AD 患者情感表达更高且声学抑制（flattening）现象；

**🔧 技术方法**

采用多评标注与加权投票裁决技术，使用 eGeMAPS 声学特征（F0、响度、HNR 等）进行探索性声学分析；

**📊 数据集**

使用 DementiaBank Pitt Corpus（ADReSS 2020 训练集）中的 108 讲者（54 AD + 54 对照），共 1,492 句子；

**📈 对比分析**

与对照组比较显示 AD 组非中性情绪出现率显著更高（16.9% vs 5.7%，p<0.001），AD 组在喜悦/惊讶时响度显著升高；但悲伤时 F0 变调低于对照，提示情感声学呈现 flattening；

**⚠️ 局限性**

局限包括情感类别不平衡、非中性情绪样本量少、仅基于 Cookie Theft 任务、声学提取为句子级别、标注可靠性中等（κ≈0.3）等问题。

---

## 243. SPOT-Occ: Sparse Prototype-guided Transformer for Camera-based 3D Occupancy Prediction

**arXiv ID:** 2602.04240 | [PDF](https://arxiv.org/pdf/2602.04240v1)

**作者:** Suzeyu Chen `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2589 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了SPOT-Occ，一种基于稀疏原型的Transformer解码器，用于实现高精度实时的3D占据预测。

**💡 创新点**

创新点在于引入可变Top‑ρ%稀疏原型选择与去噪训练，将查询–体素交互拆分为特征选择与聚合两步，显著降低计算复杂度并提升稳定性。

**🔧 技术方法**

采用的技术包括Lift‑Splat‑Shoot视图变换、稀疏卷积特征金字塔、可变Top‑ρ%原型选择、原型引导聚合、门控查询更新以及基于真实标签的去噪训练。

**📊 数据集**

主要使用的数据集是 nuScenes‑Occupancy（700/150 训练/验证）和 SemanticKITTI（序列 00‑10 训练，08 验证）。

**📈 对比分析**

与 GaussianFormer‑2、SparseOcc 等基线相比，SPOT‑Occ 在 nuScenes 上实现 13.7% 的 mIoU，延迟降低 57.6%，在 SemanticKITTI 上达到 13.27% mIoU，整体性能优于现有方法。

**⚠️ 局限性**

局限性包括对原型比例 ρ 的敏感性、去噪训练仅在训练阶段使用、在极稀疏或极大体素空间时仍可能产生额外计算开销。

---

## 244. Cascading Robustness Verification: Toward Efficient Model-Agnostic Certification

**arXiv ID:** 2602.04236 | [PDF](https://arxiv.org/pdf/2602.04236v1)

**作者:** Mohammadreza Maleki `[一作]` (Toronto Metropolitan University), Reza Samavi `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 1137 | [OpenAlex ID](https://openalex.org/A5026763818)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Cascading Robustness Verification (CRV) 框架，结合多种不完整验证器并逐级加深松弛约束，改进神经网络的鲁棒性验证过程。

**💡 创新点**

创新点在于：①将验证器按计算成本分级，按需切换；②设计 Stepwise Relaxation (SR) 与 Fast SR (FSR) 以在单一验证器内部逐步加紧约束并跳过无效步骤；③实现模型无关的鲁棒性认证，消除训练-验证偏差并显著提升已认证输入比例与计算效率。

**🔧 技术方法**

使用线性规划 (LP) 及半正定规划 (SDP) 的凸松弛方法，配合 SR/FSR 逐步约束与早停策略；算法实现基于 PyTorch、YALMIP+MOSEK 等工具。

**📊 数据集**

数据集为 MNIST，采用两种不同训练策略的网络：Grad‑NN (SDP 训练) 与 LP‑NN (LP 训练)。

**📈 对比分析**

对比基线 LP‑cert 与 SDP‑cert：CRV 在保持 88% 认证准确率的前提下，平均运行时间比 SDP‑cert 降低 42–89%；LP‑cert 速度快但准确率低；SR/FSR 进一步将时间缩短至 29–53% 以内，鲁棒性保持不变。

**⚠️ 局限性**

局限性包括：依赖预先挑选的验证器集合，仍需手工确定分级顺序；在更复杂模型或其他数据集上实验不足，可能需要额外的约束调优；SR/FSR 的跳过策略在极端输入上可能导致误判，需进一步安全性验证。

---

## 245. Fine-Grained Activation Steering: Steering Less, Achieving More

**arXiv ID:** 2602.04428 | [PDF](https://arxiv.org/pdf/2602.04428v1)

**作者:** Zijian Feng `[一作]` (Nanyang Technological University), Kezhi Mao `[通讯]` (Nanyang Technological University)

**通讯引用:** 9924 | [OpenAlex ID](https://openalex.org/A5066203581)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于原子单元（AU）的激活引导方法AUSteer，能够在LLM前向传播中精准地调节单维激活，实现对模型行为的细粒度控制；

**💡 创新点**

创新点在于将传统块级激活调节细化到单维AU层面，并引入激活动量（activation momentum）来定位任务相关的AU，以及根据输入与AU的重要性动态调节引导强度；

**🔧 技术方法**

主要技术包括AU级别的线性投影分解、激活动量评分、对比样本构造、输入自适应缩放、以及对多任务与多模型的通用化实现；

**📊 数据集**

实验使用了常识推理数据集（BoolQ、COPA、WinoGrande）、数学求解数据集（SVAMP、MAWPS）以及开放式生成数据集（RealToxicPrompts、BPO），并在多款LLM（LLaMA2‑7B、Gemma2‑9B、Qwen3‑8B等）上验证；

**📈 对比分析**

与ITi、CAA、SADI、STA等基线块级调节方法对比，AUSteer仅需控制不超过100个激活即可在常识推理与数学任务上提升约1–2个百分点，毒性抑制与对齐指标亦显著优于对手；

**⚠️ 局限性**

局限性包括需要预先构造对比样本、对超大模型（>8B）和稀疏模型的扩展仍待验证，以及在某些任务中对AU定位与强度的超参数调节仍较敏感。

---

## 246. Integrated Exploration and Sequential Manipulation on Scene Graph with LLM-based Situated Replanning

**arXiv ID:** 2602.04419 | [PDF](https://arxiv.org/pdf/2602.04419v1)

**作者:** Heqing Yang `[一作]` (Beihang University), Hangxin Liu `[通讯]` (State Key Laboratory of General Artificial Intelligence, BIGAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一个基于场景图的双层规划框架，结合全局图编辑规划与局部LLM再规划，实现了在未知环境中探索与顺序操纵的闭环执行。

**💡 创新点**

创新点在于：①使用图编辑距离生成最小成本操作序列；②构造动态信念图并通过LLM预测缺失对象位置；③将LLM用于局部异常处理，实现探索与任务规划的无缝集成。

**🔧 技术方法**

核心技术包括：场景图表示、图编辑距离（GED）与拓扑排序、预训练大语言模型（LLM）用于位置预测与异常修正、运动规划器（如VKC）以及低层控制。

**📊 数据集**

主要使用 ProcThor‑10k 数据集（46 个家庭场景）进行仿真实验，并在真实移动机械臂上验证。

**📈 对比分析**

与三种基线（LLM 纯规划、Exploration+LLM、Exploration+PoG）比较，成功率 91.3%，探索节点数下降 52%，行进距离下降 36%，显著优于基线。

**⚠️ 局限性**

局限性包括：对复杂异常组合的本地规划仍存在 8.7% 失败率，LLM 可能出现幻觉或推理不一致，且对大规模场景图的计算量和内存需求有限。

---

## 247. SPEAR: An Engineering Case Study of Multi-Agent Coordination for Smart Contract Auditing

**arXiv ID:** 2602.04418 | [PDF](https://arxiv.org/pdf/2602.04418v1)

**作者:** Arnab Mallick `[一作]` (Center for Development of Advanced Computing), Harmesh Rana `[通讯]` (Center for Development of Advanced Computing)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了SPEAR多代理框架，用规划、执行、修复、命令执行与协调代理实现智能合约审计的自适应流程

**💡 创新点**

将MAS协调模式（Contract Net、计划协商、资源拍卖）与程序化首修复相结合，提升审计可扩展性、鲁棒性与资源管理

**🔧 技术方法**

基于BDI代理架构、AGM信念修订、Contract Net协议、资源拍卖、程序化首修复（PFIR）、LLM调用与Docker沙箱等技术

**📊 数据集**

使用Damn Vulnerable DeFi benchmark（15题）、500条Solidity安全目标及Uniswap/Compound/Aave等DeFi协议数据集

**📈 对比分析**

与单一工具、流水线、组合工具及集中调度器对比，采用准确率、召回率、F1、恢复时间、LLM调用等指标；SPEAR在F1最高（0.87），恢复时间最短（2.3min），LLM调用最少

**⚠️ 局限性**

评估规模有限、工具完整性假设强、资源瓶颈与中心协调器单点失败、未实现RL学习与大规模验证

---

## 248. Med-MMFL: A Multimodal Federated Learning Benchmark in Healthcare

**arXiv ID:** 2602.04416 | [PDF](https://arxiv.org/pdf/2602.04416v1)

**作者:** Aavash Chhetri `[一作]` (NepAl Applied Mathematics and Informatics Institute for research), Binod Bhattarai `[通讯]` (University of Aberdeen)

**通讯引用:** 1479 | [OpenAlex ID](https://openalex.org/A5063234434)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Med‑MMFL，首个涵盖多模态医学数据的联邦学习基准；

**💡 创新点**

创新点在于首次系统化整合10种医学模态、5个数据集、4类任务、3种分区策略，并针对多模态扩展了6种主流联邦学习算法；

**🔧 技术方法**

采用FedAvg、FedProx、SCAFFOLD、FedNova、m‑MOON和CreamMFL等算法，结合对比学习、正则化及多模态适配技术；

**📊 数据集**

使用BraTS‑GLI2024（MRI分割）、MIMIC‑CXR‑JPG（图像+报告多标签）、Symile‑MIMIC（X光+ECG+实验室）、PathVQA（病理图像问答）和EHRXQA（结构化记录+X光问答）等公开医学数据集；

**📈 对比分析**

通过在IID、合成IID与合成非IID三种分区下进行统一训练，评估指标覆盖DSC、AUC、Acc、F1，结果显示FedProx在多数非IID场景表现最优，整体联邦结果与集中训练相差1–1.5%，且不同算法在不同数据集上表现不一；

**⚠️ 局限性**

局限在于仅涵盖6种算法，未涉及个性化联邦学习或跨设备设置，也未对多模态融合策略做深入评估，且数据分割主要基于模拟而非真实医院结构。

---

## 249. Theory of Speciation Transitions in Diffusion Models with General Class Structure

**arXiv ID:** 2602.04404 | [PDF](https://arxiv.org/pdf/2602.04404v1)

**作者:** Beatrice Achilli `[一作]` (Bocconi University), Marc Mézard `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文提出了一种通用的分化（speciation）理论，用贝叶斯分类与自由熵差异来预测扩散模型在向后过程中的类归属转移时间；

**💡 创新点**

创新点在于将分化理论扩展到任意混合分布（不依赖均值分离），给出普适的时间尺度与多阶分化预测，并在高维下证明其对数尺度；

**🔧 技术方法**

使用了贝叶斯归属分析、自由熵差、KL 散度近似、随机场 Ising 模型的复制计算以及分潜势法；

**📊 数据集**

主要使用合成数据：高维高斯混合（不同均值与方差）和1D Ising 链混合（不同逆温度）；

**📈 对比分析**

通过“U‑turn”实验测定误归属率，与理论给出的分化时间对照，结果显示理论预测与实验高度一致，误归属率在理论阈值前保持在1%以内；

**⚠️ 局限性**

局限性包括需满足“Proper Density Decomposition”假设、仅在大 N 的高维情形下收敛、对非独立同分布或长程相关的数据可能失效。

---

## 250. Convivial Fabrication: Towards Relational Computational Tools For and From Craft Practices

**arXiv ID:** 2602.04393 | [PDF](https://arxiv.org/pdf/2602.04393v1)

**作者:** Ritik Batra `[一作]` (Cornell Tech), Steven J. Jackson `[通讯]` (Cornell University)

**通讯引用:** 3830 | [OpenAlex ID](https://openalex.org/A5101637116)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过对23名木工、纤维艺术家和金属工艺师进行半结构化访谈，探究计算工具在工艺实践中的使用与关系，提出三阶共融关系（即时、社区和机构）以及七条设计原则，以指导更具共融性的数码制作工具。

**💡 创新点**

创新点在于：①将伊利奇共融工具概念扩展至三阶共融关系；②提出以材料对话、模糊建模、节奏同步等七条可落地的设计原则；③强调材料、社区与生态层面的交互，突破传统单一工具/技术关注。

**🔧 技术方法**

使用的技术方法包括：访谈设计、访谈录音与转写、Whisper 自动转写、手工校正、基于归纳与演绎的编码（构建代码表、双人编码）、定性主题分析，最终形成共融关系框架与设计原则。

**📊 数据集**

数据集为：23名具5年以上专业经验的工匠样本，涉及木工、纤维艺术、金属工艺等多种工艺；采集资料为访谈文字记录、访谈时长30–75分钟、录音视频等。

**📈 对比分析**

本文未进行数值性能对比或实验验证，评估主要基于访谈所得的主题和案例描述；因此无法给出传统意义上的性能指标，讨论侧重于工具设计与实践的可行性与启示。

**⚠️ 局限性**

局限性包括：①样本主要来自美国东北部网络，可能存在地域与网络效应偏倚；②研究聚焦工艺师自我报告，缺乏外部客观评估；③未涵盖所有类型的数码工具与工艺场景，结果对其他工艺可能适用性有限；④设计原则尚未经过原型实现与实证检验。

---

## 251. Blockchain Federated Learning for Sustainable Retail: Reducing Waste through Collaborative Demand Forecasting

**arXiv ID:** 2602.04384 | [PDF](https://arxiv.org/pdf/2602.04384v1)

**作者:** Fabio Turazza `[一作]` (University of Modena and Reggio Emilia), Marco Mamei `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 4187 | [OpenAlex ID](https://openalex.org/A5001933830)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本论文提出并验证了基于区块链的联邦学习（Blockchain‑FL）在食品零售供应链中的需求预测与浪费降低应用。

**💡 创新点**

创新点在于将联邦学习与区块链、安全聚合和全局差分隐私相结合，既实现了数据隐私与模型可信性，又通过去中心化的账本保障模型更新的可验证性与不可篡改性。

**🔧 技术方法**

核心技术包括：联邦平均算法（FedAvg）、安全聚合（SecAgg+）、全局差分隐私、Ethereum/Layer‑2 区块链与 IPFS 存储、以及轻量化的前馈神经网络。

**📊 数据集**

使用的数据集为 Walmart 45 家门店的 2010‑2012 周度销量数据，包含节假日、温度、燃油价格、CPI 与失业率等特征。

**📈 对比分析**

通过与集中式学习和独立学习三种方案对比，结果表明 Blockchain‑FL 的均方误差仅比集中式高约 5%–10%，但明显优于独立学习，并在过度库存误差上提升 5%–40% 的废弃物减少。

**⚠️ 局限性**

局限性包括：1）仍依赖中心服务器，缺乏完全去中心化；2）对区块链的 gas 成本与可扩展性敏感；3）实验规模仅覆盖 45 家门店，难以直接推广至更大、多样化的供应链网络。

---

## 252. They Call Her 'Miss' and Him 'Professor': Lived Experiences of Women Teaching Support Staff in IT/SE Education

**arXiv ID:** 2602.04332 | [PDF](https://arxiv.org/pdf/2602.04332v1)

**作者:** Vasudha Malhotra `[一作]` (Monash University), Rashina Hoda `[通讯]` (Monash University)

**通讯引用:** 4636 | [OpenAlex ID](https://openalex.org/A5036986396)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过15名女性教学支持人员（TSS）半结构化访谈，分析她们在IT/SE高等教育中的日常教学经历，揭示其权威获取、抵抗与维护过程，并提出面向多维交叉性（语言、种族、年龄、合同类型等）的实务性改进建议。

**💡 创新点**

创新点在于：①首次将女性TSS的日常教学权威问题系统化为“交叉性权力与特权轮”；②利用社会技术扎根理论（STGT）对访谈文本进行分层编码、概念化与交叉性叠加；③将研究发现转化为多层次、可操作的政策与实践建议。

**🔧 技术方法**

技术方法主要为社会技术扎根理论（STGT）基本阶段：哈希标签编码、常数比较、概念聚类、交叉性标签叠加；并使用主题地图与视觉化（权力轮与优势雷达）呈现。

**📊 数据集**

数据集为15名女性TSS的访谈记录（含访谈问卷、访谈脚本），受访者来自澳大利亚莫纳什大学信息技术学院，涵盖不同学术阶段、行业背景与语言/种族特征。

**📈 对比分析**

该研究为经验报告，不涉及数值性能对比；其有效性通过访谈深度、代码一致性、多研究者校正与案例共鸣评估。报告强调在实际教学环境中，采用其建议可降低权威被贬低的频率，提升女性TSS的职业满意度与学生学习体验。

**⚠️ 局限性**

局限性包括：①样本仅为单一大学的15名女性TSS，缺乏跨校与跨文化验证；②数据仅来自受访者自述，缺乏学生或同事的三角验证；③交叉性分析为描述性可视化，无法证明因果关系；④研究者主观解释可能受构建主义立场影响。

---

## 253. Safe and Stylized Trajectory Planning for Autonomous Driving via Diffusion Model

**arXiv ID:** 2602.04329 | [PDF](https://arxiv.org/pdf/2602.04329v1)

**作者:** Shuo Pei `[一作]` (University of Hong Kong), Huachun Tan `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 3678 | [OpenAlex ID](https://openalex.org/A5100751506)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于扩散模型的轨迹规划框架——SDD Planner，能够在保证安全的前提下实现多样化的驾驶风格。

**💡 创新点**

核心创新在于（1）多源风格感知编码器利用距离敏感注意力融合动态代理与环境信息；（2）风格引导的动态轨迹生成器通过时间步自适应的能量引导，在扩散解码过程中实现安全与风格的动态平衡。

**🔧 技术方法**

采用扩散概率模型（DDPM）、Transformer 关注机制、距离敏感注意力、动态能量加权（碰撞与速度），以及分类器引导的扩散解码。

**📊 数据集**

StyleDrive、NuPlan 两个真实交通数据集；同时使用公开的 StyleDrive 风格标签和 NuPlan 的交互事件。

**📈 对比分析**

与 WoTE、Diffusion-Planner、PLUTO、PDM 等多种基准比较，StyleDrive 上 SM-PDMS 提升 3.9%，NuPlan 上整体分数 91.83/80.32/91.76，均位居榜首，碰撞率显著下降，速度与舒适度也得到提升。

**⚠️ 局限性**

局限性：仍需进一步提升多模态融合效率；对极端稀有场景的鲁棒性有待验证；风格模型受限于三种预定义风格，难以覆盖更细粒度的驾驶偏好；对计算资源的需求较高。

---

## 254. Multiview Self-Representation Learning across Heterogeneous Views

**arXiv ID:** 2602.04328 | [PDF](https://arxiv.org/pdf/2602.04328v1)

**作者:** Jie Chen `[一作]` (Sichuan University), Xi Peng `[通讯]` (Sichuan University)

**通讯引用:** 9720 | [OpenAlex ID](https://openalex.org/A5022800038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种多视角自表示学习（MSRL）框架，利用冻结的预训练模型生成异构特征，并通过信息传递机制聚合邻域信息，进一步采用分配概率分布一致性损失在不同视角之间实现表征不变性。

**💡 创新点**

创新点在于：①将自表示学习与信息传递机制结合，动态选择同类别内最线性相关的邻居进行特征聚合；②引入分配概率分布一致性方案，利用多视角互补信息对伪标签进行自监督约束；③在增量视角分析中提供理论保证，解释多视角加入对聚类一致性与熵的影响。

**🔧 技术方法**

核心技术包括：多视角预训练特征提取、线性自表示层、注意力权重的自适应信息传递、softmax分配概率分布、交叉熵与多样本一致性损失、基于Adam的自监督优化。

**📊 数据集**

使用了八个公开视觉数据集：Pets、GTSRB、DTD、Aircraft、Flowers、CIFAR‑10、CIFAR‑100 和 ImageNet‑100，评估自监督学习与零样本迁移两类任务。

**📈 对比分析**

与多种主流多视角聚类方法（SCMVC、CMVC、SparseMVC、HSACC、MCMC、TURTLE）以及零样本迁移基线（CLIP、LaFTer）对比，MSRL 在 ACC、NMI、ARI 上普遍领先 1–3% 以上，并在训练速度上优于 SparseMVC，速度接近或略优于 TURTLE。

**⚠️ 局限性**

局限性主要体现在：①对预训练模型质量敏感，低质量视角可能导致聚类性能下降；②需要在冻结的预训练模型上训练，无法利用模型微调带来的更细粒度特征；③缺少对大规模多模态（文本、音频）视角的直接实验与理论扩展。

---

## 255. Quantile Transfer for Reliable Operating Point Selection in Visual Place Recognition

**arXiv ID:** 2602.04401 | [PDF](https://arxiv.org/pdf/2602.04401v1)

**作者:** Dhyey Manish Rajani `[一作]` (Queensland University of Technology), Tobias Fischer `[通讯]` (Queensland University of Technology)

**通讯引用:** 4409 | [OpenAlex ID](https://openalex.org/A5071424922)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于量化分位数的阈值转移方法，能够在部署阶段根据用户指定的精度要求自动选择视觉定位（VPR）的匹配阈值，从而最大化召回率，并实现了从小规模校准数据到未知环境的动态阈值迁移。

**💡 创新点**

创新点包括：① 将阈值映射为相似度分布的分位数，使阈值在不同环境和采样规模下保持稳健；② 通过计算查询与校准样本之间的相关性，挑选最相似的校准子集来估计阈值；③ 在部署时将校准分位数直接迁移到目标分布，从而无需在线标注即可满足精度约束。

**🔧 技术方法**

技术手段：相似度矩阵构建、Pearson相关匹配、分位数归一化与阈值转移、动态阈值估计；使用深度学习提取的图像描述子（如NetVLAD、MegaLoc等）和余弦相似度作为基础匹配指标。

**📊 数据集**

实验数据集：Nordland（季节变化）、SFU Mountain（山地日暮）、Oxford RobotCar（城市雨天），覆盖不同视角、光照和环境变化。

**📈 对比分析**

方法与Schubert等人的无监督阈值基线及其校准增强版进行对比，采用Recall at 100% Precision、AUPC等指标评估。实验显示，在高精度场景下，所提方法比基线提升约25%召回率，AUPC显著降低，证明在精度约束下实现更高召回的能力。

**⚠️ 局限性**

局限性：目前仅针对单帧匹配方法，未扩展到序列或滤波式VPR；依赖预先收集的静态校准集，缺乏自监督在线更新机制；在极端光照或大规模部署的鲁棒性仍需进一步验证。

---

## 256. Optimal Rates for Feasible Payoff Set Estimation in Games

**arXiv ID:** 2602.04397 | [PDF](https://arxiv.org/pdf/2602.04397v1)

**作者:** Annalisa Barbara `[一作]` (Bocconi University), Andrea Celli `[通讯]` (Bocconi University)

**通讯引用:** 277 | [OpenAlex ID](https://openalex.org/A5023478606)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在未知双矩阵博弈中，仅通过观测两位玩家的平衡行为样本，推断能够解释该行为的所有支付矩阵集合（可行支付集），并给出该推断问题的最优采样复杂度

**💡 创新点**

首次给出针对一般博弈和零和博弈、精确与近似Nash平衡情形的最优采样复杂度下界与上界，揭示了近似程度α对学习难度的α⁻¹放大效应，并提出了一种简单的基于经验策略的最优算法

**🔧 技术方法**

利用极值理论、Hausdorff距离分析、KL散度与改变测度技术、半分配（fractional knapsack）结构的最优性证明以及大数定律与Bernstein不等式等概率工具，构建并分析了可行支付集的线性可行性描述

**📊 数据集**

无具体实验数据集，论文以理论推导与抽象实例为主

**📈 对比分析**

本文未进行实验比较，而是通过数学证明与下界/上界匹配，证明其在所有情形下均实现了最优采样复杂度；若在实验中验证，则算法表现与理论一致

**⚠️ 局限性**

局限包括：假设两位玩家的策略已固定且可观测；仅针对Nash（或α‑Nash）平衡；对更广泛的解概念（如相关均衡）及动态学习环境的推广尚未覆盖；实例依赖性分析缺失

---

## 257. Counterfactual Explanations for Hypergraph Neural Networks

**arXiv ID:** 2602.04360 | [PDF](https://arxiv.org/pdf/2602.04360v1)

**作者:** Fabiano Veglianti `[一作]` (Sapienza University), Gabriele Tolomei `[通讯]` (Sapienza University)

**通讯引用:** 877 | [OpenAlex ID](https://openalex.org/A5011299100)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 CF-HyperGNNExplainer，一种针对超图神经网络的反事实解释方法，能够通过最小结构编辑（删除节点-超边关联或超边）生成解释；

**💡 创新点**

创新点在于：①首次为超图模型设计反事实解释器；②引入可微分的邻域掩码，在超图卷积传播中实现结构干预；③提供两种细粒度变体（节点-超边层级与超边层级），实现可解释性与可操作性；

**🔧 技术方法**

采用超图卷积网络（Hypergraph Convolutional Network）与可微分掩码优化相结合的反事实生成框架，使用梯度下降、Sigmoid 软化与阈值化；

**📊 数据集**

在三种经典文献引用网络数据集（Cora、CiteSeer、PubMed）上评估；

**📈 对比分析**

与两种基于图的反事实解释器（CF-GNNExplainer、RCExplainer）对比。CF-HyperGNNExplainer 在 Cora 上的准确率达 72%，高于对手；解释稀疏度高（>98%），解释规模小；稀疏实现速度快 13.9×；

**⚠️ 局限性**

局限性包括：仅支持删除操作；对节点-超边变体在极稀疏图中搜索空间过大；目前仅适用于节点分类任务，未考虑特征扰动或超图分类。

---

## 258. Generative AI in Systems Engineering: A Framework for Risk Assessment of Large Language Models

**arXiv ID:** 2602.04358 | [PDF](https://arxiv.org/pdf/2602.04358v1)

**作者:** Stefan Otten `[一作]` (FZI Research Center for Information Technology), Eric Sax `[通讯]` (FZI Research Center for Information Technology)

**通讯引用:** 1800 | [OpenAlex ID](https://openalex.org/A5080457302)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对系统工程环境的LLM风险评估框架（LRF），通过将LLM应用的自主程度与对工程影响进行二维分类，从而确定风险等级并给出相应的验证、监督与缓解措施。

**💡 创新点**

创新点在于将驾驶自动化等成熟领域的自主级别概念迁移到LLM，并与影响等级结合构建风险矩阵；提供了一套统一的、可操作的风险评估逻辑，帮助组织在创新与安全之间取得平衡。

**🔧 技术方法**

主要使用了自主度定义（Assisted、Guided、Supervised、Fully Automated）和影响度定义（Low、Medium、High）两维度，构成风险矩阵；框架设计结合了传统安全标准（如IEC 61508、ISO 26262）中的严重性与可控性概念。

**📊 数据集**

该工作并未基于具体数据集进行实验，而是通过案例说明（需求检查器与法律案例评估）来验证框架的可行性。

**📈 对比分析**

由于本研究属于方法论与框架设计，没有实施实验或性能指标的比较；评估仅通过示例说明不同自主/影响组合对应的风险等级与建议的控制措施。

**⚠️ 局限性**

局限性包括：缺乏经验数据验证框架的实用性；未提供量化指标评估LLM成熟度或风险数值；对不同领域、不同规模系统的适用性尚待进一步研究。

---

## 259. When and Where to Attack? Stage-wise Attention-Guided Adversarial Attack on Large Vision Language Models

**arXiv ID:** 2602.04356 | [PDF](https://arxiv.org/pdf/2602.04356v1)

**作者:** Jaehyun Kwak `[一作]` (Korea Advanced Institute of Science and Technology), Se-Young Yun `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 1621 | [OpenAlex ID](https://openalex.org/A5091674853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大型视觉语言模型的对抗攻击，提出一种基于注意力热点的阶段式攻击框架。

**💡 创新点**

创新点在于发现：① 高注意力区域对攻击更敏感；② 攻击这些热点会导致注意力重分配到新的热点；③ 通过阶段化聚焦热点，有效利用有限的L∞扰动预算，显著提升攻击成功率。

**🔧 技术方法**

主要技术包括：跨模态注意力提取、基于裁剪的局部梯度优化、L∞约束下的对抗扰动生成、阶段化热点调度与分配。

**📊 数据集**

使用的数据集：NIPS 2017 Adversarial Attacks and Defenses Competition（1000张图片）做为源图；MSCOCO验证集生成目标文本，图片尺寸统一为224×224。

**📈 对比分析**

与M‑Attack、FOA‑Attack等随机裁剪基线相比，SAGA在10个目标模型（5闭源5开源）上实现了最高的攻击成功率（ASR），尤其在Gemini系列模型上提升约43%，并在平均相似度和扰动范数上均优于对手。

**⚠️ 局限性**

局限性：攻击前需先从开源模型提取注意力图；对不同模型的迁移性尚待进一步验证；在更严格的视觉模型或更大扰动预算下效果可能不同。

---

## 260. Explicit Uncertainty Modeling for Active CLIP Adaptation with Dual Prompt Tuning

**arXiv ID:** 2602.04340 | [PDF](https://arxiv.org/pdf/2602.04340v1)

**作者:** Qian-Wei Wang `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**通讯引用:** 10431 | [OpenAlex ID](https://openalex.org/A5034104790)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出双Prompt（正负）学习框架，利用CLIP文本分支估计伪标签可信度，构建基于不确定性的主动学习流程，并与参数高效微调方法（CoOp、VPT、MaPLe）结合。

**💡 创新点**

创新点在于同时引入正Prompt提升类别辨识度和负Prompt对伪标签进行反向学习，从模型内部获得可靠的不确定性估计；将这种不确定性直接驱动样本挑选与伪标签挖掘，实现主动学习与模型适配的协同优化。

**🔧 技术方法**

核心技术包括CLIP预训练模型、双Prompt文本学习、视觉Prompt Tuning、CoOp/​VPT/​MaPLe PEFT、负Prompt监督损失、基于p_clean的样本排序与采样、主动学习策略（类均衡选取、伪标签挖掘）。

**📊 数据集**

实验数据集涵盖六类图像分类任务：Caltech101、DTD、EuroSAT、FGVCAircraft、Flowers102、UCF101。

**📈 对比分析**

与随机、Entropy、CoreSet、BADGE、CEC、ALFA-Mix、GCNAL等传统主动学习方法进行对比；在ViT‑B/16和ViT‑L/14两种backbone下，且在1%、2%、5%注释预算下，平均准确率均明显优于基线，提升幅度约为2%–8%。

**⚠️ 局限性**

局限性：需要调节负Prompt权重和温度等超参数；方法目前仅在图像分类任务验证，对其他视觉任务（检测、分割等）的适用性未测试；在极低预算下伪标签误差仍可能影响性能。

---

## 261. Fine-tuning Pre-trained Vision-Language Models in a Human-Annotation-Free Manner

**arXiv ID:** 2602.04337 | [PDF](https://arxiv.org/pdf/2602.04337v1)

**作者:** Qian-Wei Wang `[一作]` (Tsinghua University), Shu-Tao Xia `[通讯]` (Tsinghua University)

**通讯引用:** 10431 | [OpenAlex ID](https://openalex.org/A5034104790)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在不需要任何人工标注的前提下，提出了 Collaborative Fine‑Tuning（CoFT）及其增强版 CoFT+，通过双模型、双提示的跨模态协作实现对 CLIP 等大规模视觉‑语言模型的无监督微调。

**💡 创新点**

创新点包括：①双提示（正负提示）机制显式建模伪标签的干净程度，消除阈值和噪声假设；②双模型协作，互相生成与验证伪标签，减少确认偏差；③两阶段训练：先用高置信伪标签做参数高效微调，再用协作过滤后的伪标签完成全量微调；④CoFT+ 在此基础上加入迭代 PEFT、动量对比学习与 LLM 生成提示模板，进一步提升伪标签质量和表示鲁棒性。

**🔧 技术方法**

采用的技术包括：视觉提示调优（VPT）、参数高效微调（PEFT）、双提示学习（正负提示）、跨模态协作伪标签生成与筛选、动量对比学习、LLM 生成提示模板、全量微调 + 线性分类头。

**📊 数据集**

实验使用了十个公开图像分类数据集（Caltech101、CIFAR‑10、DTD、EuroSAT、FGVCAircraft、Flowers102、Food101、OxfordPets、StanfordCars、UCF101），并在 CIFAR‑100N 上评估噪声标签鲁棒性；此外在无监督、半监督和蒸馏设置下使用同一系列数据集。

**📈 对比分析**

与 CLIP 的零样本推断、UPL、IFPL、GRIP、LaFTer、CPL、CoOp、Tip‑Adapter、PromptKD 等方法对比。CoFT+ 在无监督设置下平均准确率达到 76.75%，比 CLIP 提升 11.82%；在细粒度/领域迁移数据集上提升幅度更大；在 CIFAR‑100N 上 CoFT+ 达到 80.89%，超过现有噪声学习方法 DEFT 的 79.04%；在蒸馏和半监督场景下也均优于对应基线。总体而言，CoFT 系列方法在多数任务上实现或超过少量标注监督的性能。

**⚠️ 局限性**

局限性：①仍依赖大规模预训练 CLIP，若无此类模型难以直接迁移；②需要双模型协作，训练成本和显存比单模型高；③在极端高噪声或极少数据情况下伪标签质量仍受限；④方法主要针对分类任务，对检测/分割等任务的推广尚未验证；⑤需调节多组超参数（K、λ、γ 等），对不同数据集的鲁棒性需进一步研究。

---

## 262. History-Guided Iterative Visual Reasoning with Self-Correction

**arXiv ID:** 2602.04413 | [PDF](https://arxiv.org/pdf/2602.04413v1)

**作者:** Xinglong Yang `[一作]` (Nanjing University of Aeronautics and Astronautics), Sheng-Jun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4199 | [OpenAlex ID](https://openalex.org/A5103204774)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 H‑GIVR 框架，利用历史推理信息与多次视觉观察实现自我纠错。

**💡 创新点**

创新点在于不采用传统的重复生成+投票，而是将历史答案与图像描述作为参考，动态纠正错误并模拟人类反复验证的思维过程。

**🔧 技术方法**

基于多模态大型语言模型（Llama3.2‑vision、Qwen2.5vl、Gemma3 等），结合视觉描述模块、图像再观察机制、一致性迭代推理以及答案确认机制。

**📊 数据集**

在 ScienceQA、A‑OKVQA、OK‑VQA、VQAv2、TextVQA 五个视觉问答基准上进行评估。

**📈 对比分析**

与标准、Simple CoT、Complex CoT、FS‑CoT、Auto‑CoT、Active‑Prompt、Self‑Consistency 等基线相比，平均提升 1–16%（多选题最高 16%），同时平均调用次数仅 2.6–4.0 次，低于 Self‑Consistency 固定的 5 次调用。

**⚠️ 局限性**

局限性包括：对开放式答案仍需更多推理步骤；在语言科学子领域表现略逊于 Self‑Consistency；在更复杂多模态任务上的泛化尚待进一步验证。

---

## 263. Self-evolving Embodied AI

**arXiv ID:** 2602.04411 | [PDF](https://arxiv.org/pdf/2602.04411v1)

**作者:** Tongtong Feng `[一作]` (Tsinghua University), Wenwu Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 22826 | [OpenAlex ID](https://openalex.org/A5100339293)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了自我进化的具身人工智能（Self‑Evolving Embodied AI）框架，定义并系统梳理了五大核心模块：记忆自更新、任务自切换、环境自预测、具身自适应和模型自演化，并对其实现机制与闭环协同进行了全面阐述；

**💡 创新点**

核心创新点在于将具身 AI 从传统的“人造、固定、预训练”模式转向“动态、可自我更新、长期自适应”的全流程自我进化范式，强调五个模块的相互耦合与共同进化，从而实现长期自主学习与适应；

**🔧 技术方法**

主要技术包括：自我编辑/组织/蒸馏记忆管理、基于学习进度的任务自选择与在线任务生成、深度潜在/生成式世界模型、基于图/模块化的具身自重构与自校准、以及基于模组化与自监督的模型架构自重构与自优化；

**📊 数据集**

论文并未给出统一的数据集，而是综述了各模块对应的前沿工作所使用的数据环境，如OpenAI Gym、Mujoco、Minecraft、WebRL 交互日志、机器人平台传感器记录等；

**📈 对比分析**

通过与现有的“人造具身 AI”框架对比，指出自我进化框架在多任务、未知环境、可迁移具身等场景下表现出更高的适应性和持续性能，但论文并未给出统一量化指标，主要以案例讨论为主；

**⚠️ 局限性**

局限性包括：缺乏统一的评估基准与数据集；自我进化过程可能导致不稳定或不可解释的行为；对安全、可信度与大规模群体协同的理论与实证研究尚不充分；

---

## 264. LoRDO: Distributed Low-Rank Optimization with Infrequent Communication

**arXiv ID:** 2602.04396 | [PDF](https://arxiv.org/pdf/2602.04396v1)

**作者:** Andrej Jovanović `[一作]` (University of Cambridge), Nicholas D. Lane `[通讯]` (Flower Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种兼顾低秩优化和不频繁通信的框架 PPX，解决了低秩优化在本地更新场景下子空间停滞的问题。

**💡 创新点**

创新点在于：①使用全局聚合的伪梯度得到高质量投影矩阵；②在投影矩阵中注入全秩的 quasi‑hyperbolic 动量，恢复子空间探索；③在不增加通信或显存负担的前提下实现近乎同步低秩优化的性能。

**🔧 技术方法**

技术包括：低秩自适应优化（低秩投影、SVD/PowertSGD）、全秩 quasi‑hyperbolic 动量、错误反馈（error‑feedback）、多时尺度本地更新（DiLoCo/Des‑Loc 等）以及实验中使用的 Transformer 语言模型。

**📊 数据集**

使用了多语种混合文本数据集（MosaicML 的 500M+ 语料）以及 2048 长度的自回归 Transformer，实验规模涵盖 16M、125M、720M 参数模型。

**📈 对比分析**

与同步低秩优化（PPX‑Sync）和全秩基线（AdamW/SGD）对比，PPX 在 125M、720M 模型上达到了 <1% 的困惑度差距，且在 720M 上实现了约 10× 的通信压缩、8× 的显存压缩；在低秩（r≤8）且内存受限的情况下，PPX 还能进一步提升 3–5% 的困惑度。

**⚠️ 局限性**

限制主要在于：①低秩投影仍受批量大小和同步间隔影响，低秩 r 较小时对高同步间隔更敏感；②全秩动量的设计需要额外调参；③在极端分布式或极大模型（>1B）下的可扩展性尚待验证。

---

## 265. Beyond Rejection Sampling: Trajectory Fusion for Scaling Mathematical Reasoning

**arXiv ID:** 2602.04391 | [PDF](https://arxiv.org/pdf/2602.04391v1)

**作者:** Jie Deng `[一作]` (Microsoft), Yutao Xie `[通讯]` (Microsoft)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 TrajFusion，通过将教师生成的错误推理轨迹与正确轨迹融合，并插入反思提示，改进传统拒绝采样微调以提升大语言模型在数学推理任务中的性能。

**💡 创新点**

创新点在于把被拒绝的错误轨迹视为有价值的负样本，采用自适应的错误率与多样性度量动态决定融合数量，并通过结构化的反思提示构造试错式训练示例，从而在保持原有训练目标和架构不变的前提下，显著丰富监督信号。

**🔧 技术方法**

使用技术包括：链式思维（Chain‑of‑Thought）生成、自动验证器、错误率 r(x) 与错误多样性 u(x) 的度量、基于反射提示的轨迹融合、标准 next‑token 预测损失，全部在现有的 LLM 微调框架中实现。

**📊 数据集**

实验数据集涵盖六大数学推理基准（GSM8K、MATH、CollegeMath、DeepMind‑Math、OlympiadBench、TheoremQA）以及长文本基准（AIME24/25、MATH‑500），在 LLaMA3‑8B 与 DeepSeekMath‑7B 两大模型上进行低数据（15K）和完整数据（100K）以及 4K/32K 上下文长度的评测。

**📈 对比分析**

与传统 RFT 以及 MMIQC、DART‑Math、MathFusion 等多种基线比较，TrajFusion 在 15K 训练集时平均提升 6–8 百分点，在 100K 训练集时平均提升 4–6 百分点，尤其在 MATH、TheoremQA 等高难度任务中显著优于对手；在 32K 长上下文场景下也能保持 3–4 百分点的性能提升。

**⚠️ 局限性**

局限性在于依赖采样时产生错误且多样的轨迹；若问题始终被正确或错误地以单一模式解决，TrajFusion 无法提供额外监督，方法自然退化为普通 RFT；此外，针对极大模型或极长文本的进一步扩展仍需实验验证。

---

## 266. Digital Twins & ZeroConf AI: Structuring Automated Intelligent Pipelines for Industrial Applications

**arXiv ID:** 2602.04385 | [PDF](https://arxiv.org/pdf/2602.04385v1)

**作者:** Marco Picone `[一作]` (University of Modena and Reggio Emilia), Marco Mamei `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 4187 | [OpenAlex ID](https://openalex.org/A5001933830)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于数字孪生（DT）的ZeroConf AI流水线，自动完成数据预处理、模型训练、参数调优与异常检测，显著降低工业CPS中AI集成的配置成本。

**💡 创新点**

创新点在于：①将DT的代表性、记忆化、增强与复制四大核心能力映射到ZeroConf流水线的各个阶段，实现零配置的模型生命周期管理；②通过DT动态生成多副本并行调参，实现即时的模型对比与自动最佳配置选择；③将实时预处理、聚类与变化点检测等功能嵌入DT，实现全流程自动化。

**🔧 技术方法**

核心技术包括：数字孪生框架（WLDT）、OPC‑UA 与 MQTT 互操作、滚动最大峰值预处理、K‑means 聚类、PELT 变化点检测、DT生命周期与复制机制、容器化部署与联邦更新。

**📊 数据集**

使用数据集为微工厂平台的加速度计实时流（X/Y/Z轴），采集自 Arduino RP2040 与 Siemens PLC，经过OPC‑UA / MQTT 传输。

**📈 对比分析**

通过在三种不同惩罚系数下并行复制聚类流水线，并对比 Silhouette 分数和聚类数量，验证了自动调参的有效性；在异常检测上，DT 的变化点检测在实时流中实现了高精度、低延迟（<200 ms）故障监测，优于传统手工脚本方案。

**⚠️ 局限性**

局限性包括：①仅在单一工业场景（微工厂振动监测）验证，缺乏跨域通用性验证；②对复杂多模态数据的支持仍有限；③DT 运行与容器化部署在资源受限的嵌入式设备上可能受限。

---

## 267. Beyond KL Divergence: Policy Optimization with Flexible Bregman Divergences for LLM Reasoning

**arXiv ID:** 2602.04380 | [PDF](https://arxiv.org/pdf/2602.04380v1)

**作者:** Rui Yuan `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Group-Based Mirror Policy Optimization（GBMPO）框架，将传统基于KL的群组策略优化扩展到任意Bregman散度，支持手工设计散度和可学习的神经镜像映射；

**💡 创新点**

核心创新是把Bregman散度引入群组策略优化中，证明KL并非最优正则化，且通过学习镜像映射可进一步提升稳定性和效率；

**🔧 技术方法**

使用了Bregman散度理论、镜像下降、群组优势估计、神经网络镜像映射、演化策略（ES）进行镜像映射的元学习；

**📊 数据集**

评估数据集包括数学推理的GSM8K、代码生成的MBPP以及HumanEval做零样本迁移；

**📈 对比分析**

与基线Dr. GRPO、GSPO、标准KL进行对比，GBMPO在GSM8K上准确率从81.2%提升到86.7%，在MBPP上pass@1从59.8%提升到60.8%，且使用神经镜像映射能显著降低训练方差并缩短生成长度；

**⚠️ 局限性**

局限性在于演化元学习成本高、收益增幅有限，且对更大模型或更复杂任务的泛化能力尚未验证。

---

## 268. Multi-scale hypergraph meets LLMs: Aligning large language models for time series analysis

**arXiv ID:** 2602.04369 | [PDF](https://arxiv.org/pdf/2602.04369v1)

**作者:** Zongjiang Shang `[一作]` (Zhejiang University), Ling Chen `[通讯]` (Zhejiang University)

**通讯引用:** 35235 | [OpenAlex ID](https://openalex.org/A5100411084)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5a41884c-404f-4688-a89c-aa238c10fe68` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出一种多尺度超图与大型语言模型对齐的框架 MSH-LLM，用于时间序列的多任务分析

**💡 创新点**

创新点包括：① hyperedging 机制在多尺度下增强时间序列语义信息；② 跨模态对齐（CMA）模块在不同尺度实现自然语言与时间序列的对齐；③ 混合提示（MoP）机制引入可学习、数据相关、能力增强三类提示，激活 LLM 的推理能力

**🔧 技术方法**

技术方法：多尺度特征提取、超图学习、跨模态注意力对齐、混合提示工程、冻结预训练 LLM 的对齐与推理

**📊 数据集**

使用 27 个真实世界数据集，涵盖长短期预测、分类、少样本学习、零样本学习等 5 类任务

**📈 对比分析**

与 19 种先进基线对比，MSH-LLM 在各任务均获得 SOTA 表现，误差平均降低 4%~10%，并在少样本、零样本情境中显著提升

**⚠️ 局限性**

局限性：对超图规模与尺度设置敏感，模型参数量大，推理效率相对传统模型较低，未来需探索更高效实现与自适应尺度选择

---

## 269. EXaMCaP: Subset Selection with Entropy Gain Maximization for Probing Capability Gains of Large Chart Understanding Training Sets

**arXiv ID:** 2602.04365 | [PDF](https://arxiv.org/pdf/2602.04365v1)

**作者:** Jiapeng Liu `[一作]` (Institute of Information Engineering), Can Ma `[通讯]` (Institute of Information Engineering)

**通讯引用:** 504 | [OpenAlex ID](https://openalex.org/A5101021225)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于熵增最大化的子集选择方法（EXaMCaP），通过极端样本过滤、聚类+贪婪采样实现高多样性子集，并用该子集替代全量数据进行ChartU能力提升的探测。

**💡 创新点**

创新点在于将Von Neumann熵与相似度矩阵相结合进行熵增评估，并通过按簇分配预算、极端样本剔除以及并行化实现高效且覆盖全面的子集采样；同时首次将子集训练用于评估全量数据的能力提升，显著降低计算成本。

**🔧 技术方法**

使用了预训练多模态LLM（如LLaVA‑Next‑LLaMA3、Qwen2.5‑VL）的嵌入作为特征，K‑means聚类、相似度矩阵构造、特征值熵计算、贪婪采样以及GPU并行化等技术。

**📊 数据集**

在MMC_Instruction（410k条）和ECD（321k条）两大ChartU数据集上进行实验，并在多项下游基准（ChartX、ChartQA、MQA等）上评估模型性能。

**📈 对比分析**

与全量训练、随机采样、Perplexity、EL2N、CCS、COINCIDE等基线相比，EXaMCaP在不同子集规模下平均相对准确率（Avg‑Rel）均达到或接近全量训练水平（最高可达99.04%/92.92%），且在某些in‑domain任务上甚至超过全量训练，表明子集能有效捕捉全量数据的能力提升。

**⚠️ 局限性**

局限性包括：对极端高/低困惑度样本的剔除可能丢失部分有价值的稀有信息；方法依赖于预训练模型的嵌入质量；在极大规模数据或不同任务类型时的泛化性尚待进一步验证。

---

## 270. Evaluating the Presence of Sex Bias in Clinical Reasoning by Large Language Models

**arXiv ID:** 2602.04392 | [PDF](https://arxiv.org/pdf/2602.04392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 271. MirrorLA: Reflecting Feature Map for Vision Linear Attention

**arXiv ID:** 2602.04346 | [PDF](https://arxiv.org/pdf/2602.04346v1)

**作者:** Weikang Meng `[一作]` (Harbin Institute of Technology), Zheng Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 458810 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种线性注意力框架MirrorLA，利用可学习的Householder反射主动重定向特征空间，解决传统非负截断导致的信息损失；

**💡 创新点**

创新点在于将非负约束从被动截断转为主动几何重定向，并引入块级、方差感知角度调制以及跨头反射以提升表达能力；

**🔧 技术方法**

采用可学习Householder反射、方差感知角度调制、跨头线性变换、ReLU等技术；

**📊 数据集**

在ImageNet-1K、COCO、ADE20K、Cityscapes、Super-Resolution（Set5、Urban100、Manga109）、Diffusion图像生成等多种数据集上评测；

**📈 对比分析**

与软max、Performer、Linformer等线性注意力方法以及ViT、Swin等高效视觉模型对比，MirrorLA在分类、检测、分割、SR和生成任务上提升1–4.7%准确率或AP，并显著降低显存/推理时延；

**⚠️ 局限性**

局限性在于Householder反射参数的学习和方差调制可能带来额外训练复杂度，且在极高维或极稀疏情形下效果尚未充分验证。

---

## 272. UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching

**arXiv ID:** 2602.04344 | [PDF](https://arxiv.org/pdf/2602.04344v1)

**作者:** Kou Misaki `[一作]` (Sakana AI), Takuya Akiba `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 UnMaskFork（UMF）框架，将 Masked Diffusion Language Model（MDLM）的未掩码过程建模为搜索树，使用多模型或不同推理配置作为确定性动作进行 MCTS，进而实现推理时的计算缩放。

**💡 创新点**

创新点在于：① 用多模型/不同推理策略的确定性分支替代传统的高温随机采样来产生多样性；② 通过节点缓存使得相同状态-动作对可以重用，从而在固定 NFE 预算下大幅提升样本效率。

**🔧 技术方法**

技术手段包括：蒙特卡罗树搜索（MCTS）、低温（T≈0）确定性解码、熵/低置信度等重掩策略、节点缓存、以及多模型切换（Dream-Coder、LLaDA 等）。

**📊 数据集**

实验数据集：代码生成基准 LiveCodeBench、HumanEval+、MBPP+，以及数学推理数据集 MATH。

**📈 对比分析**

与 Best‑of‑N、DTS*、AB‑MCTS 等基线在相同 NFE（例如 12288）预算下对比，UMF 在 Pass@1 上分别提升约 8–12 个百分点，并且随预算增加性能持续稳定提升。

**⚠️ 局限性**

局限性包括：① 需要有效的奖励信号；② 依赖多模型与手工设计的动作集，迁移到其他任务或缺乏可评估奖励时效果未知；③ 对计算资源依赖较高，尤其在多模型切换时会增加调度开销。

---

## 273. Reducing the labeling burden in time-series mapping using Common Ground: a semi-automated approach to tracking changes in land cover and species over time

**arXiv ID:** 2602.04373 | [PDF](https://arxiv.org/pdf/2602.04373v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 274. Crypto-RV: High-Efficiency FPGA-Based RISC-V Cryptographic Co-Processor for IoT Security

**arXiv ID:** 2602.04415 | [PDF](https://arxiv.org/pdf/2602.04415v1)

**作者:** Anh Kiet Pham `[一作]` (Nara Institute of Science and Technology), Yasuhiko Nakashima `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1531 | [OpenAlex ID](https://openalex.org/A5074853381)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了一种名为 Crypto‑RV 的 RISC‑V 加速器，能够统一执行 SHA‑256/512、SM3、SHA‑3 系列、AES‑128 以及 HARAKA 等多种加密算法。

**💡 创新点**

其创新点包括：1) 统一多算法共用的四级流水线执行单元，显著降低面积和资源占用；2) 128×64‑bit 的高带宽内部缓冲区，将中间状态留在芯片内，减少外部访存；3) 双缓冲自适应调度机制，使大散列和树哈希工作保持持续流水线运行。

**🔧 技术方法**

采用了 FPGA (Xilinx ZCU102) 实现，使用 AXI DMA、AXI PIO 接口、定制指令集以及多阶段流水线，辅以内部缓冲与双缓冲调度算法。

**📊 数据集**

验证使用 1,000 万个标准测试向量（SHA、SM3、SHA‑3、SHAKE、AES、HARAKA 等），并在 FPGA 上进行功能和性能评估。

**📈 对比分析**

通过与 RISC‑V 基线核心、Intel i9‑10940X、i7‑12700H、ARM Cortex‑A53 以及其它 RISC‑V 加速器的周期/能耗对比，Crypto‑RV 在 SHA‑256/512、SM3、AES、HARAKA 等算法上实现 660–1061 倍的速度提升，能效提升 5.8–17.4 倍，能源效率达到 62–187 Mbps/W。

**⚠️ 局限性**

局限性包括：1) 仍未实现完整的 SPHINCS+ 生成树哈希加速；2) 对 32‑bit 算法的支持不够完整；3) 受内部缓冲容量限制，对极大消息块的吞吐仍有限；4) 尚未覆盖所有后量子算法的完整实现。

---

## 275. Improved Sparse Recovery for Approximate Matrix Multiplication

**arXiv ID:** 2602.04386 | [PDF](https://arxiv.org/pdf/2602.04386v1)

**作者:** Yahel Uffenheimer `[一作]` (Hebrew University of Jerusalem), Omri Weinstein `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 731 | [OpenAlex ID](https://openalex.org/A5055582428)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种简单的随机算法，用于近似矩阵乘法（AMM），其误差与输出范数AB_F成比例。该算法在O(n^2(r+log n))时间内生成一个矩阵C，满足总平方误差[C-AB_F^2]≤ (1-r/n)AB_F^2。

**💡 创新点**

算法的创新点在于引入了一种新的伪随机旋转变体（快速Hadamard变换与不对称对角缩放），均匀地重新分配输出AB的Frobenius范数。

**🔧 技术方法**

使用了快速Hadamard变换（WHT）和伪随机旋转技术。

**📊 数据集**

使用了n×n的任意矩阵A和B。

**📈 对比分析**

与Pagh的TensorSketch算法相比，该算法在运行时间上快了一个对数因子，并且在每个条目的方差上表现出更好的性能。

**⚠️ 局限性**

算法的局限性在于它依赖于随机选择的索引，可能在某些情况下导致较大的误差，尤其是在输入矩阵不平衡时。

---

## 276. Enabling Real-Time Colonoscopic Polyp Segmentation on Commodity CPUs via Ultra-Lightweight Architecture

**arXiv ID:** 2602.04381 | [PDF](https://arxiv.org/pdf/2602.04381v1)

**作者:** Weihao Gao `[一作]` (Guangdong University of Education), Lan Ma `[通讯]` (Tsinghua University)

**通讯引用:** 11813 | [OpenAlex ID](https://openalex.org/A5057311061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了UltraSeg极轻量级结肠镜息肉分割网络，实现在单核CPU上>90FPS的实时推理并保持>94%传统U‑Net的分割准确度。

**💡 创新点**

采用bottom‑up跨域迁移策略，将皮肤镜轻量模型迁移至结肠镜场景，并引入Enhanced Dilated Block、跨层轻量融合和约0.3M参数预算，实现极压缩下的实时分割。

**🔧 技术方法**

使用轻量化卷积网络、分组卷积、深度监督、边缘辅助、跨层注意力、约束膨胀卷积等技术。

**📊 数据集**

利用CVC‑ClinicDB、Kvasir‑SEG、PolypGen、PolypDB、Kvasir‑Instrument等七个公共结肠镜分割数据集进行训练与评估。

**📈 对比分析**

与1M+参数模型及其他轻量模型对比，UltraSeg‑108K/130K在5个数据集平均Dice分别约为0.78/0.79，单核CPU FPS超过90，且在0.1M参数下获得94% U‑Net 31M参数性能。

**⚠️ 局限性**

仍低于大型模型，边界模糊、对大/多息肉表现欠佳，知识蒸馏无显著提升，极低参数下的泛化仍受限。

---

## 277. Can Vision Replace Text in Working Memory? Evidence from Spatial n-Back in Vision-Language Models

**arXiv ID:** 2602.04355 | [PDF](https://arxiv.org/pdf/2602.04355v1)

**作者:** Sichu Liang `[一作]` (Southeast University), Deyu Zhou `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估文本和视觉输入在空间 n-back 任务中的工作记忆表现，比较 Qwen2.5 系列模型在文本网格与图像网格下的准确率、d'、AUC 等指标。

**💡 创新点**

创新点在于揭示表征码（文本 vs 视觉）对工作记忆探针的计算过程产生实质性影响，视觉输入导致模型更倾向于最近比对而非按指示的延迟绑定，并通过 lag‑scan 诊断量化这一策略转变。

**🔧 技术方法**

使用 Qwen2.5‑7B、Qwen2.5‑VL‑7B（以及更大规模版本）大模型，采用确定性解码、token‑级 log‑prob 计算证据、AUC 与 d' 等统计指标，并进行最近重复（lure）干扰分析。

**📊 数据集**

使用自生成的 3×3 到 7×7 网格序列，共 1,200 条试验，分别以 ASCII 文本网格和 256×256 像素图像网格呈现，保持刺激内容一致。

**📈 对比分析**

在文本‑网格、文本‑网格（VLM）和视觉‑网格三种输入条件下比较表现；结果显示文本输入获得最高准确率和 d'，视觉输入性能显著下降，尤其在 n>1 时接近随机，且模型在视觉输入下更保守、误报率低。

**⚠️ 局限性**

局限性包括：只评估空间 n-back 任务，未检验其他工作记忆范式；模型架构主要为单一 VLM，未探索提示或更复杂的多模态融合策略；结果可能与特定模型和提示设计相关，需进一步验证。

---

## 278. RISE: Interactive Visual Diagnosis of Fairness in Machine Learning Models

**arXiv ID:** 2602.04339 | [PDF](https://arxiv.org/pdf/2602.04339v1)

**作者:** Ray Chen `[一作]` (University of Florida), Christan Grant `[通讯]` (University of Florida)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5070860161)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 RISE，一种通过可视化排序残差来诊断域迁移下公平性差异的交互式工具。

**💡 创新点**

创新点在于将残差排序转化为可解释的拐点和中位线指标，并提出 F-pattern、F-h、F-v 三种残差指标，用于捕捉子群误差分布与阈值转折点的差异。

**🔧 技术方法**

使用残差排序、Kneedle 拐点检测、交互式可视化面板以及基于残差的统计指标，并结合 AIF360 等公平性工具进行评估。

**📊 数据集**

在 BDD100K 驾驶数据集（晴天与雨天环境）上进行实验，并对 IRM、MBDG、IGA 三种算法进行比较。

**📈 对比分析**

通过可视化残差曲线和残差指标与传统准确率、Demographic Parity、Mean Difference 等指标对比，发现 IRM 公平但准确率低，MBDG 准确率高但存在局部偏差，IGA 在准确率与公平性之间实现平衡；残差指标能够揭示这些细粒度差异。

**⚠️ 局限性**

局限性包括目前仅支持二分类和单一敏感属性，尚未扩展到多分类、LLM 或结构化子群等更复杂场景。

---

## 279. On the use of LLMs to generate a dataset of Neural Networks

**arXiv ID:** 2602.04388 | [PDF](https://arxiv.org/pdf/2602.04388v1)

**作者:** Nadia Daoudi `[一作]` (Luxembourg Institute of Science and Technology), Jordi Cabot `[通讯]` (Luxembourg Institute of Science and Technology)

**通讯引用:** 8711 | [OpenAlex ID](https://openalex.org/A5074872542)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大型语言模型（GPT‑5）根据预设的架构、任务、输入类型与复杂度要求，自动生成608个完整的PyTorch神经网络代码文件，形成一个多样化的基准数据集；同时开发静态分析与符号跟踪工具验证生成网络是否符合规范。

**💡 创新点**

1）首次通过LLM生成多样化NN架构；2）构建了可复现、可验证的生成流程；3）公开完整数据集与验证工具，为神经网络可靠性、迁移与重构研究提供统一基准。

**🔧 技术方法**

使用GPT‑5进行Prompt工程生成网络代码；采用AST静态分析提取层信息；利用符号追踪验证输入/输出兼容性；在GitHub仓库中提供脚本自动化生成与验证。

**📊 数据集**

主要使用LLM自动生成的608个网络代码；在验证阶段，对每类输入类型（表格、时序、文本、图像）训练单个网络，使用标准公开数据集（如CIFAR、MNIST、UCI表格等）进行测试，验证网络可用性。

**📈 对比分析**

通过自定义验证工具检查合规性，发现8个不合规模型并重新生成；在各类任务上训练示例网络，确认模型能够正确训练并取得可接受的性能，表明生成的网络结构合理且可执行。

**⚠️ 局限性**

- 仅覆盖7种基本架构与4种任务类型，缺少更高级或自定义层；
- 生成过程仍依赖人工制定Prompt，若Prompt不足可能产生错误；
- 验证侧重结构正确性，未对模型性能多样性或泛化能力做系统评估；
- 数据集规模相对有限，可能不足以覆盖所有工具的泛化评估。

---

## 280. EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL

**arXiv ID:** 2602.04417 | [PDF](https://arxiv.org/pdf/2602.04417v1)

**作者:** Lunjun Zhang `[一作]` (University of Toronto), Jimmy Ba `[通讯]` (University of Toronto)

**通讯引用:** 153053 | [OpenAlex ID](https://openalex.org/A5012276327)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种简单改进策略梯度方法——EMA Anchor 和 Top‑k KL，使大型语言模型在强化学习任务中更稳定、更高效

**💡 创新点**

创新点在于：1）将EMA anchor（目标网络）用于 token‑level KL 约束而非传统 sequence‑level；2）提出 Top‑k KL 估计器，在保持无偏值和梯度的同时，只需 O(k) 记忆，兼顾 exact KL 与采样 KL 的优点

**🔧 技术方法**

主要技术包括：EMA 更新规则、无偏 token‑level KL 估计（K3++、K4、K5）、Top‑k KL 估计器、off‑policy 与尾部校正、基于 GRPO 的强化学习框架

**📊 数据集**

实验数据集涵盖数学推理（AIME、Omni‑MATH、AMC、Still、MATH500、Minerva、OlympiadBench）以及 Agentic RL（NQ、HotpotQA、2WikiMultiHopQA、Musique、Bamboogle）

**📈 对比分析**

相较于基线 GRPO 与传统 TR‑DPO、WARP 等，EMA‑PG 在数学推理任务上平均提升约 34% Pass@1，Agentic RL 上平均提升 33%，尤其在 HotpotQA、2WikiMultiHopQA 等多跳 QA 上提升 40%+

**⚠️ 局限性**

局限性包括：对 EMA 超参数（η）与 k 值仍需经验调优；在大词表下 Top‑k KL 仍需平衡记忆与估计误差；未探究对离线 RL 或更复杂 f‑divergence 的泛化

---

## 281. HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation

**arXiv ID:** 2602.04412 | [PDF](https://arxiv.org/pdf/2602.04412v1)

**作者:** Puyue Wang `[一作]` (University of Auckland), Hong Jia `[通讯]` (University of Auckland)

**通讯引用:** 26936 | [OpenAlex ID](https://openalex.org/A5102810576)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种名为HoRD的双阶段学习框架，将稀疏的根相对3D关节关键点命令映射到鲁棒的关节力矩控制；

**💡 创新点**

创新点在于提出了基于历史的动力学表示(HCDR)实现在线自适应、标准化稀疏关节表示(SSJR)统一不同来源的运动数据，以及通过在线蒸馏将鲁棒的教师策略迁移到轻量化学生策略，实现零样本域迁移；

**🔧 技术方法**

使用PPO强化学习训练教师策略，Transformer结构的学生策略，结合HCDR的跨时间注意力模块和SSJR的稀疏命令接口，辅以基于episode的域随机化；

**📊 数据集**

采用AMASS大规模运动捕捉数据集（7000+段，100+小时），并将其转化为SSJR格式；

**📈 对比分析**

与MaskedMimic、OmniH2O、ExBody2、Hover等基线进行对比，HoRD在IsaacLab与Genesis两种物理引擎下、带/不带域随机化的测试中，成功率分别提升至约90%及以上，姿态跟踪误差减少约2倍，且在无额外训练的情况下实现了优异的零样本转移、地形适应与外部扰动恢复；

**⚠️ 局限性**

局限性包括：训练成本高、对运动可行性过滤不足导致梯度干扰、仅覆盖无物体交互的运动、对长时序目标控制能力有限、需进一步提升样本效率与对不同机器人平台的通用性。

---

## 282. Separation-Utility Pareto Frontier: An Information-Theoretic Characterization

**arXiv ID:** 2602.04408 | [PDF](https://arxiv.org/pdf/2602.04408v1)

**作者:** Shizhou Xu `[一作]` `[通讯]` (University of California), Shizhou Xu (University of California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了公平性指标分离（equalized odds）与模型效能之间的 Pareto 前沿，并在信息理论框架下给出了其存在性、凹性与边界条件；同时提出了基于条件互信息（Conditional Mutual Information, CMI）的直接经验正则化方法，用以在深度模型训练中实现可控的公平-效能权衡；

**💡 创新点**

①首次在通用（非二分类）情形下对分离-效能 Pareto 前沿进行严谨的理论刻画（包含凹化闭包与增量代价）；②证明 CMI 正好度量分离违约，并给出统一的上界保证；③提出不需要学习型代理、仅使用样本统计的 CMI 估计器，既可实现可视化前沿，又兼具收敛与上界性质；

**🔧 技术方法**

信息理论工具（互信息、条件互信息、Fano 与 Rate–Distortion 定理）、随机化策略、梯度归一化、软化的 CMI 插值估计器、深度神经网络（MLP）与 Adam 优化器；

**📊 数据集**

四个公开基准：UCI Adult、COMPAS、UCI Bank（表格数据）以及 CelebA（图像数据）；

**📈 对比分析**

与七类主流公平方法（约束降维、对抗去偏、信息代理、分布鲁棒、后处理等）进行对比；在信息平面上，CMI 生成了连续、凸且低方差的 Pareto 前沿；在部署指标（Accuracy/EO gap）上，CMI 在保持或提升效能的同时显著降低分离违约，尤其在严苛公平阈值下表现最优；

**⚠️ 局限性**

①估计器假设离散标签与敏感属性，连续情形仍需改进；②在某些极端分布下仍存在理论上可行但实际实现困难的公平-效能折衷；③估计器存在正偏差且收敛速度受样本量与类别平衡影响；④需满足 i.i.d. 条件，对分布漂移的鲁棒性待进一步研究。

---

## 283. LCUDiff: Latent Capacity Upgrade Diffusion for Faithful Human Body Restoration

**arXiv ID:** 2602.04406 | [PDF](https://arxiv.org/pdf/2602.04406v1)

**作者:** Jue Gong `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 22315 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种一阶扩散框架LCUDiff，专门用于高忠诚度人类身体图像恢复，能在保持速度的同时显著提升像素与感知质量。

**💡 创新点**

创新点包括：①将预训练的4通道潜空间扩展为16通道并通过通道分裂蒸馏（CSD）保持先验；②引入先验保留适配（PPA）双分支融合方案解决潜空间与UNet的不匹配；③设计解码路由器（DeR）实现样本级别的解码路径选择，兼顾结构与细节；④实现全流程单步推理，保持极低推理延迟。

**🔧 技术方法**

技术手段主要包括：变分自编码器（VAE）通道扩展与蒸馏；基于Stable Diffusion的潜在扩散模型；LoRA微调、CFG引导与双向提示；先验保留适配、解码路由器网络；多种损失（MSE、DISTS、LPIPS、对抗）与训练策略。

**📊 数据集**

使用的训练数据集包括 PERSONA、LSDIR、FFHQ（2万张）等，用于VAE和扩散模型的联合微调；测试数据集包括合成的 PERSONA-Val 与真实的 MPII-Test，另外还有轻度退化验证集。

**📈 对比分析**

与多种基准（多步 DiffBIR、SeeSR、PASD、ResShift；一阶 SinSR、OSEDiff、InvSR、OSDHuman、HAODiff）对比，LCUDiff 在 DISTS、PSNR、SSIM、LPIPS、CLIPIQA、TOPIQ、TReS 等指标上均居于一阶方法之首，显示出更好的像素与感知质量，并保持了单步推理的高效性。

**⚠️ 局限性**

局限性包括：①仍需预训练的4通道先验，VAE通道扩展后潜在分布偏移可能在极端退化下产生细节失真；②解码路由器需要额外的标签数据，且对极端噪声情况的鲁棒性未充分验证；③在多步或更大模型规模下的可扩展性与性能提升空间仍待探索。

---

## 284. Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion

**arXiv ID:** 2602.04405 | [PDF](https://arxiv.org/pdf/2602.04405v1)

**作者:** Yixin Zhu `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]` (Dalian University of Technology)

**通讯引用:** 46747 | [OpenAlex ID](https://openalex.org/A5006986293)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种交互式空间‑频率融合 Mamba（ISFM）框架，用于多模态图像融合，先通过模态特定提取器（MSE）提取特征，再使用多尺度频率融合（MFF）和交互空间‑频率融合（ISF）实现高效信息整合。

**💡 创新点**

创新点在于：① 引入多尺度频率融合模块，动态融合低频/高频信息；② 设计交互空间‑频率融合，利用频率特征引导空间特征融合，并通过 Mamba 进行长程依赖建模；③ 通过频率引导门控（FGG）和 Mamba 门控（FGM）实现空间频率双向交互。

**🔧 技术方法**

主要技术包括：视觉状态空间模型（Mamba）、离散小波变换（DWT）、自适应注意力、深度可分离卷积、频率引导门控、交叉注意力与层归一化等。

**📊 数据集**

使用六个公开多模态融合数据集：IVIF 的 MSRS、FMB、RoadScene；MIF 的 Harvard Medical（MRI‑CT、MRI‑PET、MRI‑SPECT）。

**📈 对比分析**

与 15 种现有方法（CNN、GAN、Transformer 等）对比，ISFM 在所有八个评价指标上均实现平均排名第一，显著优于现有最先进方法，尤其在信息保留与纹理细节上表现突出。

**⚠️ 局限性**

局限性包括：① 模型参数相对较多，推理时间与 FLOPs 仍高于部分轻量级方案；② 在 MIF 数据集上未进行微调，性能受限于训练数据规模；③ 频率分解采用固定小波基，可能对不同场景适应性有限。

---

## 285. Swordsman: Entropy-Driven Adaptive Block Partition for Efficient Diffusion Language Models

**arXiv ID:** 2602.04399 | [PDF](https://arxiv.org/pdf/2602.04399v1)

**作者:** Yu Zhang `[一作]` (Tongji University), Longbing Cao `[通讯]` (Macquarie University)

**通讯引用:** 13968 | [OpenAlex ID](https://openalex.org/A5000798681)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的块级解码框架 Swordsman，利用熵变化自适应划分语义成分块并动态调整阈值，实现高效并行解码。

**💡 创新点**

创新点在于：① 用熵移位检测句子语义/句法成分边界，避免固定块导致成分碎片化；② 根据块内平均熵动态调节解码阈值，实现难度感知的并行解码；③ 兼容 KV 缓存，保持训练成本最低。

**🔧 技术方法**

技术包括：熵分析与熵移位检测、块级并行解码、动态阈值自适应、KV 缓存利用；实现基于现有 DLM（如 LLaDA、Dream）的推理改造。

**📊 数据集**

在四大主流基准上评测：数学推理（GSM8K、MATH）、代码生成（HumanEval、MBPP），并使用多种 KV 缓存配置进行对比。

**📈 对比分析**

与 Fast‑dLLM、D2F、AdaBlock 等固定或自适应块划分方法对比，Swordsman 在绝大多数设置下均获得最高或相近准确率，同时推理速度提升 1.5–3.6 TPS、延迟下降 0.2–0.4 秒，保持训练无关的优势。

**⚠️ 局限性**

主要限制在于仅验证于块级 DLM，可能需针对不同数据集调节阈值参数，且对非块级或更大规模模型的适用性尚未全面检验。

---

## 286. Bi-directional Bias Attribution: Debiasing Large Language Models without Modifying Prompts

**arXiv ID:** 2602.04398 | [PDF](https://arxiv.org/pdf/2602.04398v1)

**作者:** Yujie Lin `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 3971 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在不微调或改写提示的前提下，利用语义提示词检测、梯度归因与投影层干预，实现对LLM偏见的神经元级消除

**💡 创新点**

创新点：①基于熵最小化的自适应提示词筛选；②前向与后向两种基于积分梯度的偏见归因；③直接在投影层上截断神经元激活实现干预

**🔧 技术方法**

技术：熵量化、前向/后向积分梯度归因、神经元激活截断、投影层干预

**📊 数据集**

数据集：四大域（Gender、Nationality、Profession、Religion）的偏见评测集（SS/LMS/ICAT）、BBQ与WinoBias进行多任务评估

**📈 对比分析**

与Auto-Debias、Prefix Prompting、Self-Debiasing、DDP、IG^2等基线相比，在所有四个域均取得更低的SS、更高的LMS/ICAT，且保持高语言流畅度，性能优于现有方法

**⚠️ 局限性**

局限：仅针对投影层神经元，可能忽略深层非线性贡献；对不同规模模型的可迁移性需要进一步验证；未涉及多模态或多任务场景

---

## 287. SparVAR: Exploring Sparsity in Visual AutoRegressive Modeling for Training-Free Acceleration

**arXiv ID:** 2602.04361 | [PDF](https://arxiv.org/pdf/2602.04361v1)

**作者:** Zekun Li `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jian Cheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对视觉自回归（VAR）模型，提出了一个训练‑无关的加速框架 SparseVAR，通过稀疏注意力实现高效推理。

**💡 创新点**

创新点在于发现并利用VAR注意力的三大属性——注意力吸收点（Attention Sinks）、跨尺度激活相似性和空间局部性，进而设计了两种稀疏注意力模块：跨尺度自相似稀疏注意力（CS^4A）和跨尺度局部稀疏注意力（CSLA）。

**🔧 技术方法**

技术上使用稀疏索引映射、跨尺度稀疏预测、块级稀疏卷积（Block‑wise Sparse Kernel）以及基于FlashAttention的高效实现，整个流程无需额外训练。

**📊 数据集**

实验在 1024×1024 的文本到图像生成任务上进行，评估指标包括 GenEval、DPG‑Bench、ImageReward、HPSv2.1 以及 PSNR/SSIM/LPIPS，数据来源为公开的文本提示与图像对齐数据集。

**📈 对比分析**

与现有加速方法（FastVAR、ScaleKV、SkipVAR 等）比较，SparseVAR 在不跳过任何尺度时实现 1.57×（8B）/1.38×（2B）的速度提升，同时保持或优于基线的 GenEval 分数和几乎相同的低层指标；与跳尺度策略结合可达 2.28×加速，依旧保留高频细节。

**⚠️ 局限性**

局限性：对稀疏模式的依赖使得在不同 VAR 变体或极低分辨率时效果可能下降；稀疏决策尺度的选择需要经验性调优；目前仅针对自回归图像生成，尚未验证对其它任务（如视频、3D）或其他自回归架构的迁移性。

---

## 288. VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image

**arXiv ID:** 2602.04349 | [PDF](https://arxiv.org/pdf/2602.04349v1)

**作者:** Teng-Fang Hsiao `[一作]`, Hong-Han Shuai `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 VecSet-Edit，一种基于高保真 VecSet 大型重建模型的无训练局部网格编辑框架，能够仅凭单幅 2D 图像和掩码实现对 3D 网格的精细几何与纹理编辑。

**💡 创新点**

创新点包括：① 发现 VecSet 令牌具有空间局部性，可通过令牌子集实现局部几何控制；② 双阶段令牌选择（Mask-guided Token Seeding + Attention-aligned Token Gating）精准定位编辑区域；③ Drift-aware Token Pruning 在 denoising 过程中剔除漂移导致的几何冲突；④ Detail-preserving Texture Baking 保留原网格纹理细节。

**🔧 技术方法**

核心技术：TripoSG 的 VAE+DiT 结构，RePaint 风格的局部 diffusion 编辑；利用跨注意力与自注意力矩阵进行令牌选择与关联分析；KL 散度评估跨注意力层信息量；漂移检测与剪枝；纹理烘焙方案。

**📊 数据集**

使用 Edit3D-Bench（300 对网格与编辑图像）作为评估数据集。

**📈 对比分析**

与 MVEdit、Instant3DiT、Trellis、VoxHammer 等基线对比；在 Chamfer Distance、PSNR、SSIM、LPIPS、DINO-I、CLIP-T 上均取得最优或相近性能；速度提升约 2 倍，显著优于 voxel‑based 方案。

**⚠️ 局限性**

局限性：1）在极高精度阈值下，基于位置的子集选择仍不够稳健；2）仍依赖 2D 掩码与图像条件，无法直接处理无图像提示的编辑；3）漂移剪枝仅在特定时间步执行，可能遗漏部分漂移；4）对更大规模或更复杂场景的鲁棒性尚待进一步验证。

---

## 289. Finding NeMO: A Geometry-Aware Representation of Template Views for Few-Shot Perception

**arXiv ID:** 2602.04343 | [PDF](https://arxiv.org/pdf/2602.04343v1)

**作者:** Sebastian Jung `[一作]` (German Aerospace Center), Maximilian Durner `[通讯]` (German Aerospace Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于NeMO的多视角编码-解码架构，用于仅给定少量RGB模板图像的情况下完成未见物体的检测、分割和6DoF姿态估计。

**💡 创新点**

创新点包括：将物体几何信息抽象为可离线预计算的NeMO点云，使推理时不依赖摄像机参数或训练对象；使用单一网络完成多任务；在模型自由和模型基础两种设置均表现优异。

**🔧 技术方法**

技术包括：ViT（DINOv2）提取图像特征，多视角编码器，几何映射块与MLP UDF预测表面点，交叉/自注意力解码器，DPT头输出多种稠密预测；训练使用联合损失。

**📊 数据集**

使用大规模合成数据集：从Objaverse、GSO、OmniObject3D采集11077个物体，采用BlenderProc生成PBR图像；评估在BOP挑战的T-LESS、TUD-L、YCB-V、HOPEv2、HANDAL等数据集。

**📈 对比分析**

与现有方法相比，在模型自由的检测、分割和姿态估计上均达或超过SOTA，尤其在HOPEv2/ HANDAL检测上提升2.7pp和1.9pp；在模型基础的检测与分割也表现优异，姿态估计与Co‑op相当。

**⚠️ 局限性**

局限性包括：对对称和纹理缺失物体的重建和姿态鲁棒性不足；对多实例时边界模糊导致的检测误差；以及未能直接输出边框框，需通过分割推断。

---

## 290. Model-Driven Legacy System Modernization at Scale

**arXiv ID:** 2602.04341 | [PDF](https://arxiv.org/pdf/2602.04341v1)

**作者:** Tobias Böhm `[一作]` (Trier University of Applied Sciences), Andreas Biesdorf `[通讯]` (Siemens AG)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出并应用一种基于中间模型的四阶段模型驱动迁移流程，将大型遗留.NET MVC/Web Forms系统迁移至Next.js等现代Web栈；

**💡 创新点**

创新点在于构建技术无关、语义丰富的中间模型，配合规则化转换模板实现大规模组件的半自动迁移，并通过模型保持可追踪性和可维护性；

**🔧 技术方法**

使用Python+Tree‑sitter进行代码与标记解析，Jinja2模板生成代码，规则引擎和中间模型实现转换；

**📊 数据集**

主要数据集为工业遗留系统的约1500个页面、1100个后台文件、500个用户控件及6000+本地化字符串；

**📈 对比分析**

评估方式为定性评估与开发者反馈，迁移成功率高，核心UI组件实现全自动生成，非标准布局仍需手工适配；

**⚠️ 局限性**

局限在于对定制布局、动态生成控件及非UI层（数据访问、业务服务）支持不足，且验证多为定性、单案例，缺乏量化指标与泛化验证。

---

## 291. Mosaic Learning: A Framework for Decentralized Learning with Model Fragmentation

**arXiv ID:** 2602.04352 | [PDF](https://arxiv.org/pdf/2602.04352v1)

**作者:** Sayan Biswas `[一作]` (École Polytechnique Fédérale de Lausanne), Martijn de Vos `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 726 | [OpenAlex ID](https://openalex.org/A5010233454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Mosaic Learning 框架，将深度模型分为若干碎片，在去中心化学习中按碎片独立传播，从而实现更快、多样化的信息传播，且通信开销不变；

**💡 创新点**

将模型碎片化视为学习原语，提供统一框架；理论证明在最坏情况下收敛速率与现有最优算法 EL 一致，并在凸损失下通过增大碎片数加速共识；实验表明在高度异质数据下，节点平均准确率提升最高可达12个百分点；

**🔧 技术方法**

采用去中心化学习与 Epidemic 通讯协议，局部 SGD，随机稀疏通信矩阵；对模型进行分片并在每个碎片上使用独立 gossip 矩阵；通过光滑性、噪声与异质性假设进行收敛分析；在凸二次模型下进一步分析碎片化对共识误差的影响；

**📊 数据集**

MNIST/EMNIST、CIFAR‑10/100、LEAF benchmark 子集、下一个字符预测 LSTM、矩阵分解推荐、GN‑LeNet；这些数据集覆盖了分类、推荐和序列预测任务，并包含 IID 与高异质的分布；

**📈 对比分析**

与基线 EL（K=1）在相同通信预算下比较；评价指标包括节点平均准确率/误差、全局平均模型准确率/误差、共识距离和节点性能标准差；实验结果显示在高异质场景下 Mosaic Learning 的节点平均准确率较 EL 提升约12个百分点，整体性能与 EL 相当或更优；在 IID 场景下两者相差不大；

**⚠️ 局限性**

在非凸任务中碎片化并不一定降低共识距离，往往会升高；碎片化导致共识距离上升但标准差下降，说明其并非单一指标可评估；实验仅覆盖有限的图拓扑与节点规模，缺乏对大规模分片实现细节与异步、稀疏、隐私保护等方面的深入探讨；

---

## 292. Landscape-aware Automated Algorithm Design: An Efficient Framework for Real-world Optimization

**arXiv ID:** 2602.04529 | [PDF](https://arxiv.org/pdf/2602.04529v1)

**作者:** Haoran Yin `[一作]` (Leiden University), Niki van Stein `[通讯]` (Leiden University)

**通讯引用:** 1034 | [OpenAlex ID](https://openalex.org/A5003248571)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种景观感知的自动算法设计框架，利用遗传编程生成与真实问题景观相似的代理函数，再让大语言模型在这些代理上迭代生成优化算法，最终只在少量真实评估上验证，显著降低昂贵评估成本。

**💡 创新点**

创新点包括：①将景观特征（ELA）与Wasserstein距离结合，指导遗传编程生成高质量代理函数；②将代理函数与LLM驱动的算法设计相结合，实现算法发现与高成本评估的解耦；③在实际光学/光子优化任务中验证代理迁移效果，证明代理驱动方法优于传统基准和直接发现。

**🔧 技术方法**

核心技术包括：遗传编程（GP）生成代理函数；探索性景观分析（ELA）提取问题特征；Wasserstein距离衡量代理与目标景观相似度；大语言模型（如LLaMEA）进行算法生成和迭代；AOCC指标评估算法性能；DE与LSHADE等基线算法对比。

**📊 数据集**

使用的真实数据集主要为光学/光子领域的元表面设计、Bragg镜子设计、椭偏逆问题和光伏反射率设计等问题，实验中使用其原始评估器和代理模型。

**📈 对比分析**

比较方法：在三种环境（直接优化、代理驱动、BBOB基准）下生成500个算法，挑选AOCC最高者与随机搜索、DE、LSHADE进行对比。实验结果表明，代理驱动算法在5个真实任务中大多取得与直接发现相当甚至更优的AOCC，且相比BBOB驱动显著更好，验证了景观相似度对算法迁移的重要性。

**⚠️ 局限性**

局限性：在低维问题（如椭偏逆）中代理驱动效果不如BBOB驱动，可能因代理过度关注复杂景观特征；代理生成和LLM迭代仍需人工设定参数和提示，难以完全自动化；实验仅覆盖单目标连续优化，未验证多目标或离散问题的适用性。

---

## 293. TACO: Temporal Consensus Optimization for Continual Neural Mapping

**arXiv ID:** 2602.04516 | [PDF](https://arxiv.org/pdf/2602.04516v1)

**作者:** Xunlan Zhou `[一作]` (Nanjing University), Negar Mehr `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无重放的连续神经隐式映射框架（TACO），通过时间一致性优化实现机器人在动态环境中的实时地图构建。

**💡 创新点**

创新点：①将连续学习转化为时间一致性优化问题；②设计重要性加权的时间一致性约束，动态平衡记忆与适应；③无需存储或重放历史数据，既能避免灾难性遗忘，又能消除伪影和过度约束。

**🔧 技术方法**

技术：基于Co‑SLAM的多分辨率hash网格+几何/颜色解码器；使用多重乘子（augmented Lagrangian）实现时间一致性；通过输出敏感性估计（类似MAS）得到参数重要性，并采用相对线性缩放与阈值掩蔽来构造加权矩阵。

**📊 数据集**

数据集：静态场景使用Replica、ScanNet；动态模拟使用Habitat Synthetic Scenes / Habitat‑Sim；真实动态场景通过人工移动物体（如黄色凳子、立方体）构造。

**📈 对比分析**

对比方法：EWC、MAS（正则化），CNM、KR（重放），UNIKD（蒸馏），以及Co‑SLAM重放与不重放两种基准。实验表明，TACO在静态场景保持与重放上限相近的几何精度；在动态场景显著减少伪影、避免碎片化，性能优于或与MAS相当，明显优于其他基线。

**⚠️ 局限性**

局限性：早期重要性估计可能受噪声影响，需要阈值调参；仅考虑最近两时刻的历史，长期记忆覆盖有限；在极快场景变化或极大空间尺度下仍需进一步验证。

---

## 294. Fine-grained Classification of A Million Life Trajectories from Wikipedia

**arXiv ID:** 2602.04503 | [PDF](https://arxiv.org/pdf/2602.04503v1)

**作者:** Zhaoyang Liu `[一作]`, Haipeng Zhang `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了生活轨迹细粒度活动分类任务，构建24类9类标签体系，并使用SAM4LTC模型进行分类；

**💡 创新点**

创新点在于将句法图与MASK机制相结合，将三元组与句子上下文融合，同时利用LLM对句子结构进行规范化，显著提升对三元组信息的关注；

**🔧 技术方法**

技术方法包括ERNIE预训练语言模型、SpaCy句法图构建、MASK向量与图卷积/注意力融合、监督对比学习以及LLM重写句子；

**📊 数据集**

数据集为从英文维基百科生物页抽取的3.8M条生活轨迹三元组（589,193人），并手工标注2,826条用于训练和评估；

**📈 对比分析**

与8个基线（Bi-LSTM、TextGCN、R-GAT、BERT、XLNet、ERNIE、GPT-4/5、EvoPrompt）进行10折交叉验证，SAM4LTC在LLM-Refined集上F1为84.5%，超越第二名1.5%，类别级F1达89.6%；

**⚠️ 局限性**

局限性包括对手工标注样本的依赖、句法解析质量对性能的影响，以及对不同语言和更复杂上下文的泛化能力尚待验证。

---

## 295. Deconstructing sentence disambiguation by joint latent modeling of reading paradigms: LLM surprisal is not enough

**arXiv ID:** 2602.04489 | [PDF](https://arxiv.org/pdf/2602.04489v1)

**作者:** Dario Paape `[一作]` (University of Potsdam), Shravan Vasishth `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一种跨多种阅读范式（眼动、单向与双向自控阅读、Maze）的潜在过程混合模型，能够同时解释阅读时长、回读概率与句子可接受性/理解问题的答案。

**💡 创新点**

创新点包括将园路概率、成本与再分析成本拆分为不同潜在过程、引入语用推理参数以统一可接受性与问题回答，并在单一模型中并行估计阅读时长与行为结果。

**🔧 技术方法**

技术上使用多项式处理树（MPT）框架结合对数正态混合分布建模阅读时长，并通过Stan实现贝叶斯层级推断；对比模型则使用GPT‑2 124M 提取惊讶值作为线性回归预测。

**📊 数据集**

使用公开的六个实验数据集，覆盖NP/Z与MV/RR类型的园路句子，分别来自大规模自控阅读、眼动实验、Maze实验以及三种不同任务（接受/拒绝判断与理解问题）。

**📈 对比分析**

通过交叉验证（loo）比较模型，结果显示MPT模型在预测新数据时比惊讶值模型更优，且加入惊讶值的混合模型进一步提升拟合度。

**⚠️ 局限性**

局限性在于缺乏完整的Maze终端任务数据、模型对回读与延迟再分析的假设可能过于简化，以及不同范式中参数共享的前置假设可能影响解释性。

---

## 296. Proactive Agents, Long-term User Context, VLM Annotation, Privacy Protection, Human-Computer Interaction

**arXiv ID:** 2602.04482 | [PDF](https://arxiv.org/pdf/2602.04482v1)

**作者:** Yuanbo Tang `[一作]` (Tsinghua University), Yang Li `[通讯]` (Tsinghua University)

**通讯引用:** 44866 | [OpenAlex ID](https://openalex.org/A5100769533)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了ProAgentBench，构建了基于真实用户连续工作日志的主动助理评估基准，涵盖“何时干预”和“如何干预”两阶段任务。

**💡 创新点**

创新点在于①采集真实工作场景下的交互数据，保留了突发性（burstiness）和前置行为上下文；②设计层级化的何时+如何框架；③引入长周期记忆与隐私合规的多模态数据收集管线；④对比合成与真实数据对模型性能的差异。

**🔧 技术方法**

主要技术包括LLM/ VLM（GPT‑4o‑mini、Qwen3、Deepseek、LLaMA等）与不同提示策略（Zero‑shot、CoT、Self‑Consistency）；检索增强生成（RAG）、知识图谱（KG）、聚类记忆；参数高效微调（SFT、LoRA）；以及VLM+人工复核的隐私过滤管线。

**📊 数据集**

使用自研ProAgentBench数据集（28,528个事件，500+小时真实用户屏幕+应用元数据，采样率1Hz），并对比了LLM合成数据。

**📈 对比分析**

在多模态LLM基准上，最高When‑Assist准确率64.4%、How‑Assist意图准确率37.1%；Fine‑tune+真实数据可将准确率提升至74.0%，优于合成数据；记忆方法（尤其KG）可提升整体F1约6%；提示策略CoT/自一致性对性能影响不一，整体提升有限。

**⚠️ 局限性**

局限包括：样本主要为学生志愿者，专业与操作系统有限；1Hz采样可能错过极短交互；严格的隐私过滤可能排除部分敏感行为，导致数据偏倚；仅涵盖桌面和浏览器场景，未覆盖移动端或其他传感器数据。

---

## 297. Vision-aligned Latent Reasoning for Multi-modal Large Language Model

**arXiv ID:** 2602.04476 | [PDF](https://arxiv.org/pdf/2602.04476v1)

**作者:** Byungwoo Jeon `[一作]` (Korea Advanced Institute of Science and Technology), Jinwoo Shin `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6415 | [OpenAlex ID](https://openalex.org/A5102928677)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Vision-aligned Latent Reasoning（VALR）框架，在多模态大型语言模型的推理过程中动态生成与视觉特征对齐的潜在令牌，保持视觉信息在长序列生成中的稳定性。

**💡 创新点**

创新点在于将潜在推理与视觉编码器的表示对齐（REPA），通过在每一步推理前注入学习到的潜在令牌，使模型在链式推理过程中始终保持对图像细节的关注，从而实现真正的测试时可扩展性。

**🔧 技术方法**

技术包括：多模态 LLM 的两阶段课程学习（先在 CoT 数据集上训练，再加入潜在令牌并使用 REPA 进行视觉对齐）；潜在令牌生成与控制；使用 DINO、CLIP、SigLIP、π^3 等多种视觉编码器进行对齐；以及对齐损失的余弦相似度公式。

**📊 数据集**

使用的主要数据集包括 450K 规模的 CoT 视觉问答数据（Zebra-CoT、CogCoM、ReFocus、Visual-CoT、OneThinker-SFT、GCoT 等混合集）以及评测数据集 VSI-Bench、BLINK、MMVP、MMStar、MathVision、MathVista、MMhalu、CVBench 等。

**📈 对比分析**

与 GPT‑4o、Claude‑4‑Sonnet、R1‑OneVision‑7B、Ocean‑R1‑7B、LVT、CoVT、Monet 等基线相比，VALR 在 VSI‑Bench 上从 33.0% 提升至 52.9%（19.9% 绝对提升），在多模态推理长度增长时保持或提升准确率，而传统模型会出现性能下降，展示了显著的测试时可扩展性。

**⚠️ 局限性**

局限性包括：仍需预训练视觉编码器，若缺乏高质量视觉模型可能受限；潜在令牌的数量和长度需要手动设定，可能对不同任务不均衡；在极大尺度或极长推理链条下的效率和计算开销尚未完全评估。

---

## 298. Is Micro Domain-Adaptive Pre-Training Effective for Real-World Operations? Multi-Step Evaluation Reveals Potential and Bottlenecks

**arXiv ID:** 2602.04466 | [PDF](https://arxiv.org/pdf/2602.04466v1)

**作者:** Masaya Tsunokake `[一作]` (Hitachi), Yasuhiro Sogawa `[通讯]` (Hitachi)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5038587308)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在企业微领域知识下，微域自适应预训练（mDAPT）对生成型问答任务的有效性，并构建了多步 oracle 评估框架。

**💡 创新点**

创新点在于将生成型问答拆解为提取事实、推理、生成三子任务，并通过插入 oracle 结果识别瓶颈，首次在真实 IT 技术支持数据上评估 mDAPT。

**🔧 技术方法**

技术包括对 Qwen2.5‑72B‑Instruct 进行 CPT 与 SFT 的微域自适应预训练，使用多步 oracle 评估和 LLM‑as‑a‑judge 进行自动化评判。

**📊 数据集**

数据集为 JP1 企业软件的专有文档（约 193 文档，72.1 MB）及其生成的 QA 对，以及 10 个真实技术支持问答。

**📈 对比分析**

通过对比无 oracle、oracle 提取、oracle 推理三种设置的回答成功率（ASR），以及与 GPT‑4o 和 RAG 的比较，发现 mDAPT 在事实提取上与基线相当，但在推理与生成上仍差距约 60%，最高 ASR 约 39%。

**⚠️ 局限性**

限制在于仅评估了单一非推理模型（Qwen2.5‑72B‑Instruct）的微域预训练，未探究更强推理模型或轻量级 LoRA 等方法的表现，且实验资源需求高。

---

## 299. Simple 2-approximations for bad triangle transversals and some hardness results for related problems

**arXiv ID:** 2602.04463 | [PDF](https://arxiv.org/pdf/2602.04463v1)

**作者:** Florian Adriaens `[一作]` (University of Helsinki), Nikolaj tatti `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在有符号图中研究最小坏三角形覆盖（Bad Triangle Transversal）问题，提出了新的 2-近似算法并给出了相关的硬度证明。

**💡 创新点**

创新点在于：①利用 LP 近似值直接构造覆盖，算法更简洁且只需求解一次 LP；②引入随机阈值的随机化逼近，可推广至加权版和近似 LP 结果；③通过轮盘（pivot）策略实现 3/2 近似聚类转换；④给出完整图中近似不可逼近的精确常数 2137/2136‑γ。

**🔧 技术方法**

主要技术手段包括：LP 线性规划（Bad Triangle Cover LP）与其对偶；随机化阈值与期望分析；多面体逼近与匹配/三元组覆盖；基于“轮盘” (pivot) 的聚类转换；以及利用 2SAT 闭合实例构造的多项式时间归约。

**📊 数据集**

本文未使用公开数据集，全部研究基于理论构造和假想实例，主要以复杂度与近似因子为评价标准。

**📈 对比分析**

与已有方法（如三角覆盖的 3-近似、vertex‑cover 基于 LP 的 2 近似）相比，新算法在实现复杂度上更低、运行时间几乎等同于求最大边不相交坏三角集合的时间，并且在加权情形下同样保持 2 近似；聚类转换部分在期望上将误差压至 3/2 倍，优于传统 2 近似。

**⚠️ 局限性**

局限性：对一般有符号图只能得到 2 近似，且已证明更好近似在可行性上极难（与 Vertex Cover 等价）；对完整图虽给出更强硬性下限，但目前仍未突破 2 的界；算法依赖 LP 求解，规模受限；并未给出实验验证或针对实际网络数据的性能评估。

---

## 300. Can Theory-Informed Message Framing Drive Honest and Motivated Performance with Better Assessment Experiences in a Remote Assessment?

**arXiv ID:** 2602.04450 | [PDF](https://arxiv.org/pdf/2602.04450v1)

**作者:** Suvadeep Mukherjee `[一作]` (University of Luxembourg), Pedro Cardoso-Leite `[通讯]` (University of Luxembourg)

**通讯引用:** 1009 | [OpenAlex ID](https://openalex.org/A5049059166)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在远程非监考评测中，设计并评估了基于15个心理学概念的45条理论驱动的激励性信息，探究其对作弊行为、学习成绩与体验的影响；

**💡 创新点**

创新点在于系统性地把自我决定理论、认知失调理论、社会规范理论和自我效能理论的关键概念转化为可测评的消息内容，并验证不同理论框架在同一任务中的一致效应；

**🔧 技术方法**

采用基于大语言模型（LLaMA3.3）的消息生成与专家评估相结合的双阶段流程，使用规则与多阶段行为监测算法检测作弊；

**📊 数据集**

实验数据来自1232名英国Prolific参与者，完成在线时间限制的奖励式字谜挑战（anagram），同时收集作弊、成绩、体验及自评心理机制问卷；

**📈 对比分析**

与无消息对照组比较，所有干预组显著降低完全作弊率42%（从33%降至19%），提升非作弊比例19%；在成绩和体验上无显著负面影响；

**⚠️ 局限性**

局限包括：消息对部分作弊（1–99%）影响不显著；缺乏对长效或跨情境效应的检验；作弊检测依赖行为规则，可能存在误报或漏报；样本非代表性且主要来自英文网络平台。

---

## 301. What's in a Benchmark? The Case of SWE-Bench in Automated Program Repair

**arXiv ID:** 2602.04449 | [PDF](https://arxiv.org/pdf/2602.04449v1)

**作者:** Matias Martinez `[一作]` (Universitat Politècnica de Catalunya), Xavier Franch `[通讯]` (Universitat Politècnica de Catalunya)

**通讯引用:** 8379 | [OpenAlex ID](https://openalex.org/A5027686521)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统化地收集、整理并分析了 SWE‑Bench Lite 与 Verified 两个公开排行榜中的 212 条提交记录，对提交者类型、产品属性、开源程度以及使用的 LLM 等维度进行了编码与量化，揭示了行业主导、Claude 4 系列 LLM 成为最高精度（Verified 76.8%、Lite 60.3%）的主导技术以及开源方案在可复现性与透明度方面的优势；

**💡 创新点**

创新点在于首次对 SWE‑Bench 这类 AI 驱动的 issue‑repair 基准进行大规模、结构化的实证研究；通过多源元数据（leaderboard、论文、博客、LinkedIn 等）构建完整的提交档案，建立了提交者分类、产品形式与可用性标签，并对 LLM 组合与性能进行系统性梳理，为 benchmark 设计、工业落地与学术评估提供了可复制的经验框架；

**🔧 技术方法**

采用了内容分析（deductive/inductive 编码）、统计检验（Kruskal‑Wallis、Dunn 事后检验）以及 Google/LinkedIn 等多渠道手工信息抽取技术，对 LLM 名称与组合进行识别，并对提交的精度、开放度等属性进行量化；

**📊 数据集**

使用 SWE‑Bench Lite（300 题）与 SWE‑Bench Verified（500 题）两个子基准，涵盖 12 个 Python 开源项目的 Issue‑Repair 任务；

**📈 对比分析**

以“已解决率（precision）”为核心指标，比较了不同提交者类型、产品可用性和 LLM 使用方式之间的差异；结果显示，基于 Claude 4 Sonnet 的系统在两大排行榜上均领跑（精度最高达 76.8%），而单一开源 LLM 的方案虽整体精度略低，但仍能达到 60% 以上；统计检验表明行业提交者在精度上显著高于学术提交者；

**⚠️ 局限性**

主要局限包括：仅覆盖 SWE‑Bench 两个排行榜，未涵盖其他基准；数据采集过程中可能遗漏或误识别部分提交；未对 LLM 参数、模型版本、运行成本或能源消耗进行细粒度评估；且未充分考虑测试集过拟合、数据泄漏、以及基准对真实世界多样性的代表性不足等问题。

---

## 302. Hand Gesture Recognition from Doppler Radar Signals Using Echo State Networks

**arXiv ID:** 2602.04436 | [PDF](https://arxiv.org/pdf/2602.04436v1)

**作者:** Towa Sano `[一作]` (Nagoya Institute of Technology), Gouhei Tanaka `[通讯]` (International Research Center for Neurointelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于Echo State Network（ESN）的多储备（multi-reservoir）架构，用于利用FMCW雷达信号实现手势识别；

**💡 创新点**

创新点在于将不同特征图（RTM、DTM、MDM）分别输入独立的ESN储备，随后在读出层聚合状态，显著减少互相干扰，并通过线性或非线性读出实现高精度；

**🔧 技术方法**

核心技术包括：雷达信号到RTM/DTM/MDM的特征提取、Echo State Network（含泄漏率、谱半径等超参）、多储备并行设计、Ridge Regression、SVM、Random Forest等读出分类器；

**📊 数据集**

使用两个公开雷达手势数据集：Soli（60 GHz FMCW，11类，2750样本）和Dop‑NET（24 GHz FMCW，4类，2433训练样本）；

**📈 对比分析**

与传统CNN‑LSTM、ResNet‑LSTM、LSM等方法比较，Soli数据上取得98.84%准确率，超过LSM（98.02%）和所有深度学习模型；在Dop‑NET上以94.74%准确率领跑；同时训练时间仅数秒，推理时间1–2 ms；

**⚠️ 局限性**

局限性包括：对跨主体泛化表现下降（Soli约92.6%），对高噪声样本鲁棒性不足；仅在单一雷达硬件上验证，尚未在多用户或动态背景环境中测试；

---

## 303. Quantum-Based Resilient Routing in Networks: Minimizing Latency Under Dual-Link Failures

**arXiv ID:** 2602.04495 | [PDF](https://arxiv.org/pdf/2602.04495v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 304. Normalizing Speed-accuracy Biases in 2D Pointing Tasks with Better Calculation of Effective Target Widths

**arXiv ID:** 2602.04432 | [PDF](https://arxiv.org/pdf/2602.04432v1)

**作者:** Shota Yamanaka `[一作]` (LY Corporation), I. Scott MacKenzie `[通讯]` (York University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对 ISO‑style 2D 指向任务中有效目标宽度（W_e）的计算方法进行对比研究，探讨其在不同速度-准确性偏差下的偏差归一化能力。

**💡 创新点**

通过引入三种速度-准确性偏差指令、广泛的数据采集与蒙特卡洛模拟，首次系统验证单变量标准差（σ_x）优于双变量标准差（σ_xy）在平衡速度-准确性偏差方面的有效性，并确认采用目标间向量（TT）作为任务轴是更稳健的选择。

**🔧 技术方法**

采用 Fitts 规律、有效宽度计算、AIC/BIC、R²、系数变异（CV）、差异百分比（Δ）等统计方法评估模型拟合与吞吐量稳定性；同时利用 Python/ R 等工具实现数据处理与模拟。

**📊 数据集**

使用 346 名使用鼠标的网络众包参与者完成 155,700 次试验（每人 450 次），共 152,226 条有效记录，构成大样本 2D 指向实验数据集。

**📈 对比分析**

对比八种 W_e 计算方式（σ_x/σ_xy × TT/CT × A/A_e），发现单变量 σ_x（TT 轴、A）在混合偏差条件下得到最高 R²（≈0.968）并在吞吐量稳定性指标（Δ≈4.7%，CV≈2.8%）上表现最佳；双变量方法在所有指标上均逊色，且随样本量增大其优势迅速消失。

**⚠️ 局限性**

研究局限包括：仅使用鼠标设备、Crowdsourcing 环境缺乏实验控制、任务难度范围（ID 2.07–4.70 bits）相对有限、未探讨其他交互设备或更高维度的指向任务。

---

## 305. A Framework of Critical Success Factors for Agile Software Development

**arXiv ID:** 2602.04467 | [PDF](https://arxiv.org/pdf/2602.04467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 306. Graph-Based Audits for Meek Single Transferable Vote Elections

**arXiv ID:** 2602.04527 | [PDF](https://arxiv.org/pdf/2602.04527v1)

**作者:** Edouard Heitzmann `[一作]` `[通讯]`, Edouard Heitzmann

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于图论的风险限制审计（RLA）框架，用于对Meek单次可转移投票（STV）选举进行可验证的结果审计。

**💡 创新点**

创新点在于：①把整个STV计票过程建模为“所有可能的当选/淘汰序列”构成的全局图Ω；②在该图上预先选定子图G，检验真实计票路径是否留在G内，从而无需关注具体计票顺序；③对Meek规则的“保持因子”给出解析公式，解决了传统WIGM规则对计票顺序高度敏感的难题；④为每条边制定局部假设并给出置信区间的构造方法，实现了高效的样本量估计。

**🔧 技术方法**

技术手段包括：图论中的分层图与子图抽象、交叉假设的交并原理、Hypergeometric分布与“pivoting the cdf”得到稀疏样本的方差界；使用delta方法和隐函数定理对保持因子及边际进行线性化；通过方程求解获得即时保持因子；样本量估计基于ASN（平均抽样数）和LAM（可审计最小边际）。

**📊 数据集**

实验数据集覆盖三大范围：1）英国苏格兰市政选举（约5,000人、1,000多份投票表）; 2）美国波特兰市议会选举（4个区，每区约70,000人，总计约300,000人）; 3）澳大利亚国会选区（州议会选举）人数从约25万到超过5,200万不等，候选人从20人到超过150人。

**📈 对比分析**

与传统WIGM RLA方法对比，本文的Meek图式RLA在大多数实验中实现了更低的ASN（平均抽样数在0.5%–3%之间），并能在几乎所有小型选举中以5%风险水平通过审核；在中型至大型选举中，虽然“淘汰云”导致图规模急剧膨胀，但通过预先去除低效候选人或采用可审计最小边际（LAM）策略，仍能在1%以下样本率完成审核；总体性能显示，边际越大，所需样本数随之降低，符合理论预期。

**⚠️ 局限性**

局限性包括：①对Meek规则的解析保持因子公式在高阶（>3席）选举或极度不规则的状态下求解困难；②图规模随候选人数量呈指数级增长，导致中大型选举的计算成本过高；③对WIGM规则的适用尚未完成，需要在子图定义中引入计票顺序信息；④预先剔除低效候选人虽能降低计算量，但缺乏严格的数学正当性，可能在极端情况下影响审计完整性。

---

## 307. ReFRAME or Remain: Unsupervised Lexical Semantic Change Detection with Frame Semantics

**arXiv ID:** 2602.04514 | [PDF](https://arxiv.org/pdf/2602.04514v1)

**作者:** Bach Phan-Tat `[一作]` (KU Leuven), Dirk Speelman `[通讯]` (KU Leuven)

**通讯引用:** 3694 | [OpenAlex ID](https://openalex.org/A5048483716)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用FrameNet框架对词义变化进行无监督检测，采用帧分布差异计算语义变迁。

**💡 创新点**

提出完全基于显式语义框架、可解释且无需对齐嵌入空间的检测方法，突破传统神经模型的黑箱局限。

**🔧 技术方法**

使用FrameNet transformer解析器提取触发词及其帧元素，计算帧分布的Jensen–Shannon散度并分解贡献。

**📊 数据集**

在SemEval 2020 Task 1的英语双时段语料库上评估，实验仅限于英语。

**📈 对比分析**

在子任务2中取得Spearman相关0.306，排名前十；子任务1准确率0.622，均优于大多数传统嵌入基线。

**⚠️ 局限性**

仅适用于拥有成熟框架解析器的语言；解析器误差和帧语义覆盖不足可能导致误判或低估真正的语义变迁。

---

## 308. Greedy-Gnorm: A Gradient Matrix Norm-Based Alternative to Attention Entropy for Head Pruning

**arXiv ID:** 2602.04491 | [PDF](https://arxiv.org/pdf/2602.04491v1)

**作者:** Yuxi Guo `[一作]` (Southwestern University of Finance and Economics), Paul Sheridan `[通讯]` (University of Prince Edward Island)

**通讯引用:** 4579 | [OpenAlex ID](https://openalex.org/A5048570936)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种动态梯度范数驱动的头剪枝方法 Greedy‑Gnorm，能够在每一步剪枝后重新评估并选取最不重要的注意力头。

**💡 创新点**

创新点在于：①将每个头的 Q、K、V 梯度矩阵的 ℓ₂ 范数做逐元素乘积得到重要性评分，并在每次剪枝后即时重算；②使用 ε‑rectified entropy 解决传统注意力熵下溢问题；③通过贪心 prune–recompute 循环避免静态评分失效，提升剪枝稳定性。

**🔧 技术方法**

核心技术包括：Transformer 头剪枝、梯度范数计算、贪心迭代重算、ε‑rectified 熵计算、实验对比评估。

**📊 数据集**

使用了四组预训练 Transformer 及其对应任务的数据集：BERT‑财务情感分类、ALBERT‑Multi‑Genre NLI、RoBERTa‑推文情感分析、XLM‑R RoBERTa‑语言识别。

**📈 对比分析**

与注意力熵（AE）、逆熵、逆 Gnorm、随机剪枝等方法比较。实验表明 Greedy‑Gnorm 在保持 90%+ 原始准确率的同时，能够将模型体积压缩约 20%+，在所有四个模型上均优于 AE 并显著优于随机剪枝。

**⚠️ 局限性**

主要限制包括：每一步都需进行一次梯度反向传播，计算开销较大；假设校准集与部署数据分布相似；仅对注意力头进行剪枝，未覆盖 FFN 或 token 剪枝；梯度估计可能受小批量噪声影响；未在更大规模 LLM 上验证。

---

## 309. Temporal Slowness in Central Vision Drives Semantic Object Learning

**arXiv ID:** 2602.04462 | [PDF](https://arxiv.org/pdf/2602.04462v1)

**作者:** Timothy Schaumlöffel `[一作]` (Goethe University Frankfurt), Jochen Triesch `[通讯]` (Frankfurt Institute for Advanced Studies)

**通讯引用:** 5627 | [OpenAlex ID](https://openalex.org/A5065703786)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过在 Ego4D 头戴摄像头视频上生成人眼注视点，提取中心视野的图像裁剪，并利用时间对比自监督学习（基于 MoCoV3 的改进版）训练模型，研究中心视野与时间慢变化学习对语义物体表示的影响。

**💡 创新点**

创新点在于：①首次结合人眼注视预测与中心视野裁剪，模仿人类视网膜对中央区域的高分辨率处理；②在自监督学习中加入时间慢变化约束，强调相邻帧的相似表示；③系统性评估模型在多种语义层面（基本类别、细粒度、实例、场景）以及上下文共现结构上的表现。

**🔧 技术方法**

主要技术包括：人眼注视预测模型 GLC、MoCoV3 自监督学习框架、InfoNCE 损失、Momentum 编码器、线性探针评估、GloVe 训练共现表示。

**📊 数据集**

使用数据集：Ego4D（约 5 个月的头戴摄像头视频）、ImageNet-1k、ImageNet100、CIFAR-100、Flowers101、Stanford Cars、Oxford Pet、FGVC-Aircraft、DTD、ToyBox、COIL100、Core50、Places365、COCO、ADE20K、Visual Genome。

**📈 对比分析**

比较方法：在上述多样化下游任务上对冻结特征进行线性探针实验，和传统全场景自监督模型（如标准 MoCoV3、R3M、VIP 等）进行对比。实验结果显示：在中心视野+时间慢变化训练下，模型在前景物体特征提取、上下文信息编码以及细粒度/实例识别任务上均优于仅使用全景或仅时间对比的基线；但在某些纯场景识别任务上提升有限。

**⚠️ 局限性**

局限性：①人眼注视预测模型的误差可能影响中心裁剪的准确性；②实验仅在单轮训练（1 epoch）下完成，可能未充分挖掘数据潜力；③缺乏对眼动控制策略（如扫视、聚焦）细粒度的分析；④在复杂多物体场景下的共现学习效果仍有限。

---

## 310. AgenticAKM : Enroute to Agentic Architecture Knowledge Management

**arXiv ID:** 2602.04445 | [PDF](https://arxiv.org/pdf/2602.04445v1)

**作者:** Rudra Dhar `[一作]` (International Institute of Information Technology Hyderabad), Vasudeva Varma `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种 Agentic 方式的架构知识管理（AKM），通过四类专门代理（提取、检索、生成、验证）协同工作，实例化为从代码仓库自动生成 Architecture Decision Records（ADR）

**💡 创新点**

创新点在于将 AKM 任务分解为可迭代、可验证的子任务，利用多代理协作与中心 Orchestrator 进行信息传递与质量控制，显著提升 ADR 生成的完整性、相关性和可读性

**🔧 技术方法**

技术手段包括：大型语言模型（Gemini‑2.5‑pro、GPT‑5）、向量检索、LLM 生成与验证、图像生成模型、工具集集成以及基于 Agentic 框架的任务调度与迭代流程

**📊 数据集**

使用了 29 个真实代码仓库（Python、JavaScript 等），由参与者提供，涵盖 1,000–350,000 行代码，作为实验数据集

**📈 对比分析**

比较方法：通过用户研究，对四种配置（Baseline+Gemini、Baseline+GPT、Agentic+Gemini、Agentic+GPT）进行盲评，量化指标为 Relevance、Coherence、Completeness、Conciseness、Overall。结果显示 Agentic 方案在所有指标上均优于 Baseline，整体质量 3.9 对比 3.3，Completeness 3.9/3.8 vs 3.7/3.0 等

**⚠️ 局限性**

局限性包括：只验证了 ADR 生成；实验规模相对有限；仅评估两种 LLM；未涉及长周期工业验证；对多模态非结构化数据支持不足；对人机交互与实时反馈的探索仍待深化

---

## 311. TrajVG: 3D Trajectory-Coupled Visual Geometry Learning

**arXiv ID:** 2602.04439 | [PDF](https://arxiv.org/pdf/2602.04439v1)

**作者:** Xingyu Miao `[一作]` (Durham University), Junting Dong `[通讯]` (Shanghai AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了TrajVG框架，通过显式预测相机坐标3D轨迹，将跨帧3D对应关系作为显式模块与密集点图和相机姿态耦合，实现多帧无监督/半监督3D重建。

**💡 创新点**

创新点包括：①将3D轨迹作为显式对应，双向一致性约束避免梯度冲突；②使用静态轨迹锚点的姿态一致性损失抑制动态区域梯度；③将同样约束改写为仅依赖2D轨迹的自监督形式，实现海量互联网视频的半监督训练。

**🔧 技术方法**

采用端到端多帧前馈网络（点图预测、相机姿态预测、3D跟踪头），双向一致性损失、姿态一致性损失、伪2D轨迹自监督；使用CNN/Transformer特征提取、bilinear采样、Huber损失、stop-gradient等技术。

**📊 数据集**

训练数据包括CO3DV2、Mapfree、TartanAir、ASE、VKITTI、MVSynth、ARKitScenes、BlendedMVS、DL3DV、ScanNet、MegaDepth、Waymo、WildRGBD、GTASfm、HyperSim、OmniWorld、UnReal4K、MatrixCity、Spring、Kubric、PointOdyssey、DynamicReplica等；自监督阶段使用Sekai、互联网视频；评估使用Sintel、TUM-dynamics、ScanNet、RealEstate10K、Co3Dv2、DTU、ETH3D、7-Scenes、NRGBD、Sintel、Bonn、KITTI、NYU‑v2等。

**📈 对比分析**

在3D跟踪、相机姿态、点图估计、视频深度、单目深度等五大任务上与现有前馈方法（VGGT、π^3、MASt3R等）以及传统管线对比。TrajVG在各项指标上均达到或超越SOTA：3D跟踪平均精度最高，姿态RRA@30/AUC@30居首；点图Accuracy/Completion/Normal一致性在ETH3D、7-Scenes等上排名第一；视频深度Abs Rel最低；单目深度在Bonn、NYU‑v2上取得最佳。

**⚠️ 局限性**

局限性在于仍依赖大量标注数据，尤其是3D轨迹；自监督阶段对数据分布差异敏感，效果受限；在极端动态场景或强遮挡下仍可能出现漂移；模型复杂度高，训练成本大。

---

## 312. $C$-$ΔΘ$: Circuit-Restricted Weight Arithmetic for Selective Refusal

**arXiv ID:** 2602.04521 | [PDF](https://arxiv.org/pdf/2602.04521v1)

**作者:** Aditya Kasliwal `[一作]` (Lexsi Labs), Vinay Kumar Sankarapu `[通讯]` (Lexsi Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种一次性离线的电路指导权重编辑方法，利用电路定位将拒绝行为聚焦至模型内部的稀疏子网络，并在该子网络上做参数更新，生成无需推理时钩子即可部署的检查点。

**💡 创新点**

创新点在于将电路发现与参数剪裁结合：先用EAP‑IG精确定位拒绝行为的因果子网络，再限定在该子网络内做高强度权重更新，既保持了拒绝行为的选择性，又保证了对大多数参数的最小干预。

**🔧 技术方法**

核心技术包括 EAP‑IG 电路定位、参数掩码转换、对比细调（正负模型）得到的权重差分（C‑ΔΘ）、以及对模板的对比学习来驱动拒绝方向。

**📊 数据集**

使用公开的对比提示集（5类危害：犯罪、仇恨、健康、法律、性），配合拒绝与合规模板；评估基准包括 MMLU、GSM8K、SORRY‑Bench 等。

**📈 对比分析**

与 Activation Steering、CAST、Weight Steering 等基线对比，C‑ΔΘ 在 30 种模型/危害设置中提升了 24–94% 的有害拒绝率，过度拒绝率仅 1–10%，参数改动 ≤5%，并保持 0–3% 的能力退化，显示出显著的安全性与实用性提升。

**⚠️ 局限性**

局限性包括：对基模型对危害概念的分离度高度依赖；EAP‑IG 并非完全因果，可能漏掉冗余路径；局部编辑仍可能导致边缘误拒或轻微的能力偏移；评估受限于自动判别器和单一种子，分布外与对抗性提示的鲁棒性尚未充分验证。

---

## 313. Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models

**arXiv ID:** 2602.04509 | [PDF](https://arxiv.org/pdf/2602.04509v1)

**作者:** Hyeontaek Hwang `[一作]` (KAIST), Daeyoung Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种数据无关的稀疏微调方法（Dowser），通过在下游任务微调前计算每个参数的重要性，冻结重要参数，仅更新低重要性参数，从而减轻多模态大语言模型（MLLM）的灾难性遗忘。

**💡 创新点**

创新点在于：① 将参数的重要性定义为对输出的第一阶灵敏度，即权重大小、输入激活和输出梯度的乘积；② 使用Hutchinson估计与合成提示实现无需预训练数据的全局重要性评估；③ 通过一次性二进制掩码实现与普通微调相同的显存复杂度，且可扩展到多模模型。

**🔧 技术方法**

核心技术包括：梯度灵敏度分析、Hutchinson trace估计、合成文本提示（synthetic prompting）、稀疏微调与参数遮蔽、H-score评估指标。

**📊 数据集**

在四类下游任务上验证：图像字幕（COCO-Caption、Flickr30k）、图像分类（ImageNet-R）、视觉问答（IconQA、TextVQA、OKVQA、OCRVQA、GQA、MMBench）。

**📈 对比分析**

与全微调、ModelTailor、SPIDER、Grafting、DARE等方法对比，Dowser在保持上游零样本性能的同时，往往获得更高的H-score（如COCO-Caption 约70+、ImageNet-R 约70+），且在更深层微调和更大模型（LLaVA-1.5-7B、NVILA-Lite-2B）下依旧稳定。

**⚠️ 局限性**

局限性包括：① 重要性评估仅基于一次性前向/后向计算，可能未覆盖全部任务相关性；② 对于极端大模型仍需足够显存生成合成提示；③ 当下游任务与预训练任务相距过大时，低重要性参数的更新可能不足以充分迁移。

---

## 314. SDR-CIR: Semantic Debias Retrieval Framework for Training-Free Zero-Shot Composed Image Retrieval

**arXiv ID:** 2602.04451 | [PDF](https://arxiv.org/pdf/2602.04451v1)

**作者:** Yi Sun `[一作]` (Wuhan University of Technology), Yongjian Liu `[通讯]` (Wuhan University of Technology)

**通讯引用:** 14028 | [OpenAlex ID](https://openalex.org/A5100782763)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练-free的零样本合成图像检索方法SDR-CIR，利用Selective CoT和Semantic Debias Ranking两步策略来消除语义偏差。

**💡 创新点**

创新点包括：①在CoT中加入选择性视觉内容提取；②设计Anchor‑Debias两步Semantic Debias Ranking，实现对参考图像贡献的补偿与惩罚，从而显著提升检索鲁棒性。

**🔧 技术方法**

采用多模态大型语言模型（如GPT‑4.1/LLaVA）进行CoT推理，CLIP编码器做特征对齐，融合注意力与相似度惩罚实现去偏排名。

**📊 数据集**

实验使用公开的CIRR、CIRCO和FashionIQ三大合成图像检索基准数据集。

**📈 对比分析**

与一、两阶段训练-free方法对比，SDR‑CIR在CIRR、CIRCO、FashionIQ上均取得最优或接近最优的Recall/ mAP，Recall@1提升≈4%、mAP@5提升≈9%等显著改进。

**⚠️ 局限性**

局限性：仍易受参考图像噪声和文本模糊的影响，对极细粒度修改效果有限，并且依赖大型LLM推理，导致推理成本相对较高。

---

## 315. SynthVerse: A Large-Scale Diverse Synthetic Dataset for Point Tracking

**arXiv ID:** 2602.04441 | [PDF](https://arxiv.org/pdf/2602.04441v1)

**作者:** Weiguang Zhao `[一作]` (Xi'an Jiaotong-Liverpool University), Jiangmiao Pang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `67630363-6be0-4f51-ab05-7198250671a5` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 SynthVerse 这一大规模多样化的合成数据集及基准，用于通用 2D/3D 点跟踪。

**💡 创新点**

创新点在于跨平台（Blender+Isaac Sim）生成管线、覆盖多种未被充分支持的域（动画、交互、导航等）和多样化对象（关节、变形等），并提供高质量轨迹与可见性注释。

**🔧 技术方法**

采用 Blender、Isaac Sim 进行场景构建与渲染，结合物理仿真、随机纹理、光照与摄像机增强等技术，生成多视角 RGB、深度、实例掩码、相机位姿、3D/2D 轨迹与可见性。

**📊 数据集**

使用 SynthVerse 自身作为训练数据，基准包含 Nav、Human、Animal、Objects、Embodied、Film、Interaction 等八个子集；同时对比了动态复制（Dynamic Replica）、LSF Odyssey、ADT、DriveTrack、PStudio 等公开数据集。

**📈 对比分析**

通过在 SynthVerse 及其它公开基准上微调 TAPIP3D 等最新跟踪器，显著提升 AJ3D/APD3D/OA 等指标（例如在 SynthVerse‑mAverage 上从 33.3% 提升至 41.6%），验证了数据集的有效性和模型泛化能力。

**⚠️ 局限性**

局限性包括对真实世界数据的覆盖仍有限、部分复杂光照/遮挡场景生成受限、以及对极端快速运动与极端视角变换的建模仍需进一步提升。

---

## 316. The Stretto Execution Engine for LLM-Augmented Data Systems

**arXiv ID:** 2602.04430 | [PDF](https://arxiv.org/pdf/2602.04430v1)

**作者:** Gabriele Sanmartino `[一作]`, Carsten Binnig `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的LLM增强数据系统执行引擎，能够在满足用户指定的全局精度/召回目标的前提下，自动规划语义操作符的实现方案与参数，并通过 KV‑Cache 的压缩与级联实现高效推理。

**💡 创新点**

创新点包括：① 通过梯度优化全局规划，将精度/召回误差预算动态分配给不同语义操作符；② 将 KV‑Cache 视为持久化物理实现，引入压缩等级梯度，显著扩展物理实现空间；③ 结合 Bayesian 可信区间给出分布级别质量保证，避免了传统的频繁检验导致的 p‑hacking。

**🔧 技术方法**

核心技术包括：梯度基准约束优化、连续化离散选择（soft pick）、多阶段物理操作符级联、KV‑Cache 预计算与压缩（Expected Attention）、批量推理、动态规划重排操作符顺序、贝叶斯置信区间。

**📊 数据集**

使用了五个多模态数据集（Artwork、Rotowire、Email、Movies、E‑Commerce）和 300 条合成查询，覆盖图像、文本、表格等多模态，验证系统在不同精度/召回目标下的鲁棒性。

**📈 对比分析**

与现有系统（Lotus、Abacus 的 Pareto‑Cascades、SupG）进行对比。结果显示：① 在满足 95% 置信度的全局精度/召回目标下，本文系统是唯一能持续满足目标的方案；② 相比 Lotus，平均速度提升 1.4–10 倍（严格目标下最高 10 倍）；③ 与基线相比，平均 42% 的执行时间下降。

**⚠️ 局限性**

局限性包括：① 目前仅支持过滤和映射语义操作符；② KV‑Cache 预计算需要存储空间，压缩比例选择仍为经验性；③ 对高阶语义操作（join、聚合）尚未充分评估，需进一步扩展优化框架。

---

## 317. Learning the Value Systems of Agents with Preference-based and Inverse Reinforcement Learning

**arXiv ID:** 2602.04518 | [PDF](https://arxiv.org/pdf/2602.04518v1)

**作者:** Andrés Holgado-Sánchez `[一作]` (Universidad Rey Juan Carlos), Sascha Ossowski `[通讯]` (Universidad Rey Juan Carlos)

**通讯引用:** 3169 | [OpenAlex ID](https://openalex.org/A5052274381)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于多目标马尔可夫决策过程的价值系统学习框架，先通过量化轨迹比较学习价值对齐函数，再用逆强化学习估计个体的线性价值系统权重。

**💡 创新点**

首次将偏好基础逆强化学习与最大熵逆强化学习结合，提出可兼容原始价值对齐的等价性理论，并将价值系统表述为可学习的线性标量化函数。

**🔧 技术方法**

采用偏好基础逆强化学习（PbIRL）、最大熵逆强化学习（MaxEnt IRL）与深度神经网络估计奖励函数，配合特征映射与软价值迭代求解策略。

**📊 数据集**

在两个模拟案例中评估：Firefighters（基于已知规则的多目标 MDP）和 Roadworld（上海路网抽样），数据来自仿真生成的轨迹和量化偏好对。

**📈 对比分析**

通过与真值奖励的一致性、轨迹偏好准确率（>98%）以及策略状态-动作访问差异（TVC≈0）等指标评估，学习到的价值对齐函数与原始奖励高度一致，识别的价值系统权重准确性高，性能优异。

**⚠️ 局限性**

假设存在统一的价值对齐，仅限线性聚合的价值系统，难以捕捉非线性或分层价值；对人类偏好数据需求高，受限于获取成本；未考虑多主体或动态情境下的价值演化。

---

## 318. S-MUSt3R: Sliding Multi-view 3D Reconstruction

**arXiv ID:** 2602.04517 | [PDF](https://arxiv.org/pdf/2602.04517v1)

**作者:** Leonid Antsfeld `[一作]`, Jerome Revaud `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于VGGT基础模型的长序列单目三维重建流水线：先将长RGB序列划分为重叠段，分别用VGGT生成点云、深度、相机姿态，然后利用置信度加权对齐、轻量级闭环检测和姿态图优化，将各段拼接为全局一致的3D模型。

**💡 创新点**

创新点包括：①无需对模型进行再训练；②利用段间重叠信息对置信度进行深度差异加权，显著提升对齐鲁棒性；③采用SIM(3)变换组在保持计算效率的前提下实现全局一致性；④在仅用单帧RGB的无标定条件下即可直接得到物理尺度的3D重建。

**🔧 技术方法**

技术要点：VGGT Transformer基础模型、点云与相机姿态双重对齐（IRLS + Huber 损失）、置信度深度加权、SIM(3)/Affine(3)/SL(4) Lie 变换组、KD-Tree 视觉闭环检索、Levenberg–Marquardt 轻量级姿态图优化。

**📊 数据集**

使用的数据集包括：TUM RGB‑D、7‑Scenes、以及自主收集的多室办公机器人导航无标定RGB序列；评估指标为绝对位姿误差（APE）和平均角误差（AAE）。

**📈 对比分析**

与DROID‑SLAM、-SLAM、-Long等基准方法对比，本文在无标定设置下的APE/AAE均与或优于顶尖方法，并在机器人导航场景中将位姿误差降至0.25 m，显著提升了鲁棒性和重建精度。

**⚠️ 局限性**

局限性：对VGGT局部重建质量高度依赖；闭环检测对相似度阈值σ_sim敏感，需手动调参；在特征稀缺或平面结构严重的环境中，SIM(3)对齐可能出现失效；总体而言，仍需进一步自动化超参数调优与模型鲁棒性提升。

---

## 319. Intentic Semantics for Potentialist Truthmaking

**arXiv ID:** 2602.04488 | [PDF](https://arxiv.org/pdf/2602.04488v1)

**作者:** Paul Gorbow `[一作]` `[通讯]`, Paul Gorbow

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了意向语义（intentic semantics）以构造非假设逻辑（non‑hypothetical logic）的完整语义框架；

**💡 创新点**

创新点在于定义意向状态及其真值生成规则，给出非假设逻辑的完备性与可判定性（在关系型皮亚诺算术的情形下）；

**🔧 技术方法**

采用自然演绎、意向状态递归定义、真值生成的⊩关系以及两种偏序（⊩扩展与细扩展）进行语义构造；

**📊 数据集**

未使用实验数据集，主要是理论证明与逻辑模型构造；

**📈 对比分析**

通过证明 Γ⊢φ 蕴含 Γ⊩φ 与反方向的可判定性算法，展示了在给定有限常量集下的求证过程，性能由公式子公式限制保证终止；

**⚠️ 局限性**

局限在于对一般公理化理论的可判定性仅是猜想，且对归纳推理与连结词的完整利用仍需进一步研究；

---

## 320. SALAD-Pan: Sensor-Agnostic Latent Adaptive Diffusion for Pan-Sharpening

**arXiv ID:** 2602.04473 | [PDF](https://arxiv.org/pdf/2602.04473v1)

**作者:** Junjie Li `[一作]` (Northwestern Polytechnical University), Chunhua Shen `[通讯]` (Zhejiang University)

**通讯引用:** 68545 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SALAD-Pan，一种传感器无关的潜在扩散框架，用于高效的全色与多光谱图像融合（pan‑sharpening）。

**💡 创新点**

创新点包括：①Band‑wise 单通道 VAE 将多光谱图像映射到压缩潜在空间；②在潜在空间采用双分支（空间‑光谱）交互式条件扩散；③使用频域分离注入和 CLIP 文本元数据提示增强条件信息；④引入轻量 RCBA 维护跨波段一致性；⑤兼具推理加速（2–3×）和零样本跨传感器迁移。

**🔧 技术方法**

技术手段包括潜在扩散模型（DDPM/UNet）、VAE、双向交互控制、频域分离注入、CLIP 文本提示、RCBA 注意力、UniPC 采样等。

**📊 数据集**

实验数据集为 PanCollection benchmark，涵盖 GaoFen‑2、QuickBird、WorldView‑3、WorldView‑2 四种卫星传感器。

**📈 对比分析**

与传统 CNN/Transformer 方法以及基于像素空间的扩散方法比较，SALAD‑Pan 在 WV3、QB 等数据集上取得 Q、SAM、ERGAS、HQNR 等指标的最高分，推理速度提升 2–3 倍，并在未见传感器 WV2 上表现出最优的零样本迁移性能。

**⚠️ 局限性**

局限性包括：需要预训练 VAE，可能对极端波段配置的适配仍有限；潜在空间分辨率受限于 VAE 压缩；在极少样本或非常不同的传感器场景下的泛化能力尚待进一步验证。

---

## 321. LLM-Empowered Cooperative Content Caching in Vehicular Fog Caching-Assisted Platoon Networks

**arXiv ID:** 2602.04471 | [PDF](https://arxiv.org/pdf/2602.04471v1)

**作者:** Bowen Tan `[一作]` (Jiangnan University), Wen Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 17988 | [OpenAlex ID](https://openalex.org/A5100673541)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在车队协作网络中提出了三层内容缓存架构（车队本地缓存、车队雾缓存、云服务器），并通过大语言模型（LLM）实现实时、智能的缓存决策。

**💡 创新点**

创新点包括：①利用LLM处理异构信息（用户画像、历史评分、内容特征、系统状态）进行缓存决策；②设计层级化提示框架，将任务描述、系统信息与异构数据编码为LLM输入；③提出无频繁重训的单步LLM优化与确定性映射策略，实现高效自适应缓存。

**🔧 技术方法**

核心技术：大语言模型（如Grok‑3、Gemini‑2.5 Pro等）、提示工程、确定性缓存映射策略、OFDMA通信模型、仿真与对比分析。

**📊 数据集**

数据集：使用仿真生成的合成数据，包括用户画像、历史评分、内容类型和车辆缓存容量等；参数基于论文表述的具体数值（如N_c=2000、M_p=1000B等）。

**📈 对比分析**

与DDQN、CP‑SAT、DeepSeek‑R1、Gemini‑2.5 Pro、Grok‑3等方法比较。评估指标为平均缓存命中率（ACHR）和平均内容传输延迟（ACTD）。结果显示Grok‑3在大多数缓存容量下取得最高ACHR和最低ACTD，DDQN在缓存容量>250时表现优异；LLM方法在不需要重训的情况下实现低延迟和高命中率。

**⚠️ 局限性**

局限性：在高并发请求场景下的性能未充分验证；在多车队协同、跨车队迁移时的泛化能力有限；缺乏安全性与能耗方面的评估。

---

## 322. RASA: Routing-Aware Safety Alignment for Mixture-of-Experts Models

**arXiv ID:** 2602.04448 | [PDF](https://arxiv.org/pdf/2602.04448v1)

**作者:** Jiacheng Liang `[一作]` (Stony Brook University), Ting Wang `[通讯]` (Stony Brook University)

**通讯引用:** 16457 | [OpenAlex ID](https://openalex.org/A5100633410)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对 Mixture-of-Experts (MoE) 语言模型的安全对齐问题，提出了一种路由感知的专家级对齐框架 RASA，显式修复“安全关键专家”，并防止路由绕过。

**💡 创新点**

创新点在于：① 将安全对齐拆分为专家修复与路由一致性两步，避免全参数微调产生的“对齐捷径”；② 通过激活差异动态识别安全关键专家；③ 采用双向前向策略对路由进行一致性约束。

**🔧 技术方法**

主要技术包括：稀疏路由机制、激活差异度量（AAD）、块坐标下降的双层优化、交替专家微调与路由一致性训练、双前向 KL 损失。

**📊 数据集**

使用的数据集为 AdvBench（攻击意图集合）以及常规通用评测集 MMLU、GSM8K、TruthfulQA；对抗攻击采用 FlipAttack、DeepInception、Persuasion 等多种 jailbreak。

**📈 对比分析**

与全参数微调、SteerMoE 等基线对比，RASA 在单一与混合 jailbreak、跨攻击、以及多轮攻击场景下都实现了近乎完美的无害化率，同时保持或提升通用评测得分，且拒绝率可控。

**⚠️ 局限性**

局限性包括：需要手动阈值选择安全关键专家比例、对极端攻击可能仍需更大样本量、对路由结构的依赖导致在其他 MoE 架构上可能需要重新调参。

---

## 323. Mixture of Masters: Sparse Chess Language Models with Player Routing

**arXiv ID:** 2602.04447 | [PDF](https://arxiv.org/pdf/2602.04447v1)

**作者:** Giacomo Frisoni `[一作]` (University of Bologna), Gianluca Moro `[通讯]` (University of Bologna)

**通讯引用:** 1531 | [OpenAlex ID](https://openalex.org/A5079648393)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了首个面向世界棋大师风格的稀疏混合专家（MoE）象棋语言模型Mixture of Masters，并通过专家路由实现动态风格切换。

**💡 创新点**

创新点在于：①将多位顶级棋手的游戏分别训练成独立专家，保留其独特风格；②在专家间使用可学习的路由网络和权重合并，避免模式崩塌；③结合自监督学习与基于棋局规则的强化学习（GRPO）提升合法性与表现；④引入视觉风格度量（stylometry）验证专家的个性化。

**🔧 技术方法**

使用技术包括：transformer decoder（nanoGPT）作为基座，SSL（自回归预测）+ RL（GRPO）训练专家，Gumbel‑Softmax顶‑k路由，权重合并，Vision Transformer（Eψ）与LSTM实现风格嵌入。

**📊 数据集**

数据集主要来自PGNMentor、Chess.com、Lichess，选取10位顶尖棋手（如Anand、Carlsen、Nakamura 等）的比赛，训练集占比 80%，测试集 20%，仅保留 3–30 分钟的速战速决（Blitz/Rapid）并去除重复与低手棋。

**📈 对比分析**

与基准（密集单专家、模型混合 soup、Seed 模型）比较时，在对抗 Stockfish（0–5 级）游戏中，Mixture of Masters 在 FIDEScore 上平均提升 3–5 分，保持较低非法率并表现出更高的风格多样性；RL 阶段提高合法率但略降低赢率。

**⚠️ 局限性**

局限性包括：①需要手工挑选并训练每位棋手的专家，规模受限；② RL reward 仍以合法性和规则为主，未包含结果导向；③风格度量受视觉特征与 LSTM 的限制，可能未完全捕捉细粒度风格；④在更高水平或超大规模对局中，模型的推理速度与内存开销仍高于单体大模型。

---

## 324. No One-Size-Fits-All: Building Systems For Translation to Bashkir, Kazakh, Kyrgyz, Tatar and Chuvash Using Synthetic And Original Data

**arXiv ID:** 2602.04442 | [PDF](https://arxiv.org/pdf/2602.04442v1)

**作者:** Dmitry Karpov `[一作]` `[通讯]` (PAO Severstal), Dmitry Karpov (PAO Severstal)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了俄语-巴什基尔语、俄语-哈萨克语、俄语-吉尔吉斯语、英语-塔塔尔语、英语-楚瓦什语五个突厥语对的机器翻译，尝试了多种方法包括LoRA微调、检索增强提示、语义堆叠等。

**💡 创新点**

将LoRA微调与知识迁移相结合用于低资源突厥语；构建大型检索增强提示（ANNOY+向量化器+LLM）方案；利用Yandex.Translate合成语料和大规模YaTURK-7lang数据集，全面对比不同技术的效果。

**🔧 技术方法**

技术栈包括：facebook/nllb-200-distilled-600M微调、DORA（LoRA扩展）、Paged AdamW-8bit优化器、ANNOY索引+句向量（gte-small、paraphrase-multilingual-MiniLM-L12-v2）、大型语言模型（DeepSeek-V3.2、DeepSeek-R1、Gemma3、MiMoV2）、语义相似度堆叠（LaBSE）等。

**📊 数据集**

使用的数据集有：原始平行语料（俄-巴什基尔、俄-哈萨克、俄-吉尔吉斯、英-塔塔尔、英-楚瓦什），Yandex.Translate合成语料，YaTURK-7lang（6语言大规模平行语料），MASSIVE 数据集的翻译补充。

**📈 对比分析**

通过官方竞赛排行榜的验证集（chrF++）进行比较。LoRA微调在哈萨克语达到49.71、巴什基尔语达到46.94；检索增强提示在楚瓦什语得到39.47；Tatar零射/检索方案约41.6；Kyrgyz零射方案约45.6。单任务微调、零射、检索、堆叠等多种配置均被系统性评估。

**⚠️ 局限性**

局限性包括：计算资源不足导致未能进一步优化或探索更多模型；模型在楚瓦什语预训练不足，导致零射性能低；检索增强在资源相对丰富的语言效果有限；堆叠方法在哈萨克语略有下降；实验仅在竞赛验证集上评测，缺乏更广泛的跨语言评估；合成语料质量与过滤策略可能对结果产生影响。

---

## 325. Gust Estimation and Rejection with a Disturbance Observer for Proprioceptive Underwater Soft Morphing Wings

**arXiv ID:** 2602.04438 | [PDF](https://arxiv.org/pdf/2602.04438v1)

**作者:** Tobias Cook `[一作]` (University of Edinburgh), Francesco Giorgio Serchi `[通讯]` (University of Edinburgh)

**通讯引用:** 1094 | [OpenAlex ID](https://openalex.org/A5039381590)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文建立并验证了一种水下柔性变形翼在风浪扰动下的动力学模型，并利用翼表面自感知变形与扩展卡尔曼滤波（EKF）实现对攻角扰动的实时估计，随后通过 PI 控制器实现升力补偿。

**💡 创新点**

创新点包括：①将翼的自感知变形（软皮肤）作为传感源，用于估算外部扰动；②首次将 Piecewise Constant Curvature 软体动力学与薄翼理论结合，构建可用作扰动观测的模型；③在柔性翼上实现通过变形估算风扰的 EKF 观测器，突破传统需要外部传感器的限制。

**🔧 技术方法**

技术手段包括：Piecewise Constant Curvature (PCC) 动力学建模、薄翼理论（Thin Airfoil Theory）用于升力估计、扩展卡尔曼滤波（EKF）实现扰动观测、PI 控制器实现升力跟踪、实验验证。

**📊 数据集**

使用的数据集：10 组实验数据（曲率、升力、内部压力、攻角），实验在水槽流速 0.2 m/s 下进行；另使用 Micklem 等提供的压力与攻角时间序列进行模型验证。

**📈 对比分析**

通过与实验测量比较，模型曲率的 RMSE 为 0.0339 rad、升力 RMSE 为 0.0515 N；在控制实验中，升力误差 RMSE 从 0.1103 N 降低至 0.0734 N，能够在 ±5° 攻角范围内几乎完全抵消扰动；对于更大幅度的 1‑cos 变动，控制仍显著降低升力峰值。

**⚠️ 局限性**

局限性：仅考虑单自由度翼模型，控制带宽受限，无法完全抵消高速或大幅扰动；模型参数未进行充分辨识，存在一定误差；实验仅在二维单翼条件下验证，缺乏三维多模态柔性翼的验证。

---

## 326. The Supportiveness-Safety Tradeoff in LLM Well-Being Agents

**arXiv ID:** 2602.04487 | [PDF](https://arxiv.org/pdf/2602.04487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 327. MaMa: A Game-Theoretic Approach for Designing Safe Agentic Systems

**arXiv ID:** 2602.04431 | [PDF](https://arxiv.org/pdf/2602.04431v1)

**作者:** Jonathan Nöther `[一作]` (Max Planck Institute for Software Systems), Goran Radanovic `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 923 | [OpenAlex ID](https://openalex.org/A5047197460)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种基于Stackelberg安全博弈的Meta‑Adversary‑Meta‑Agent框架，用于自动设计在代理被攻击时仍保持安全的多代理系统。

**💡 创新点**

创新点在于把安全防御与系统设计统一到博弈框架中，并引入专门的Meta‑Adversary进行强化学习式的攻击搜索，使设计过程针对最坏情况。

**🔧 技术方法**

核心技术包括结构化的代理系统表示、LLM驱动的对抗搜索、迭代的Meta‑Agent/Meta‑Adversary交互、基于安全与质量双目标的评估函数。

**📊 数据集**

使用BAD‑ACTS基准四个环境（旅行规划、个人助理、金融文章写作、代码生成），并用GPT‑5.1和Qwen3:32b进行Meta‑Agent/Meta‑Adversary建模。

**📈 对比分析**

与手工安全设计、Guardian‑Agents、AFlow（仅优化质量）等基线比较。实验显示，所生成的系统在被攻击时安全评分显著提升（≈3‑4分提升），而质量保持不变甚至提升；在不同攻击模式、不同LLM、攻击强度升级等转移测试中依旧表现稳健。

**⚠️ 局限性**

局限性包括高昂的计算成本（需在正常与攻击两种情景下评估每个候选系统）、攻击者仅最小化安全而不考虑质量、以及只考虑代理级别的破坏，未覆盖工具或通信图的篡改。

---

## 328. SLUM-i: Semi-supervised Learning for Urban Mapping of Informal Settlements and Data Quality Benchmarking

**arXiv ID:** 2602.04525 | [PDF](https://arxiv.org/pdf/2602.04525v1)

**作者:** Muhammad Taha Mukhtar `[一作]` (inst1), Muhammad Imran Malik `[通讯]` (inst1)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了验证过的拉合尔、卡拉奇和孟买三大城市的高分辨率斑块化斜坡数据集，并提出了针对低标签稀缺且类别不平衡的半监督语义分割框架；

**💡 创新点**

创新点在于引入了自适应类别阈值机制（CAAT）和原型库（Prototype Bank）两种模块，分别解决了少数类别伪标签不足和语义不一致的问题；

**🔧 技术方法**

使用DeepLabV3+编码器-解码器骨干（ResNet‑101 及 DINOv2‑small）结合随机强弱增强、CutMix 与特征扰动等技术实现半监督学习；

**📊 数据集**

利用自制的拉合尔、卡拉奇、孟买数据集，以及非洲和南美的El Daein、El Geneina、Nairobi、Makoko、Medellin等公开基准，共八个城市进行评估；

**📈 对比分析**

与 FixMatch、UniMatch、UniMatch‑v2 等主流半监督方法以及完全监督基线对比，结果显示在10%标签稀缺下本框架在4/8城市优于 UniMatch，DINOv2版本在5/8城市优于 UniMatch‑v2；在跨城市零射线泛化中仅用10%标签即可获得 0.461 mIoU，超过100%标签的完全监督模型；

**⚠️ 局限性**

局限性包括：mIoU 与感知质量不完全一致，某些场景下半监督表现不及完全监督；零射线检测仍受视觉模糊限制；未能充分利用多模态社会经济或人口密度信息，需进一步融合以提升区分度。

---

## 329. PersoDPO: Scalable Preference Optimization for Instruction-Adherent, Persona-Grounded Dialogue via Multi-LLM Evaluation

**arXiv ID:** 2602.04493 | [PDF](https://arxiv.org/pdf/2602.04493v1)

**作者:** Saleh Afzoon `[一作]` (Macquarie University), Amin Beheshti `[通讯]` (Macquarie University)

**通讯引用:** 4479 | [OpenAlex ID](https://openalex.org/A5056293251)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PersoDPO，一种利用多种 LLM 自动评估指标生成无监督偏好对，进行 Persona‑grounded 对话模型的优化。

**💡 创新点**

创新点在于：①将连贯性、个性化和指令遵从三方面指标融合为自动偏好信号；②通过多 LLM 输出构建高质量偏好对，实现无需人工标注的可扩展优化；③在 DPO 基础上加入长度‑格式遵从奖励，提升指令可执行性。

**🔧 技术方法**

技术包括：基于 DPO 的偏好优化（Score‑weighted DPO）、多指标自动评估（C、P、UE、Coh‑UniEval、长度‑格式遵从），以及对 OpenAI、Qwen、Mistral 等多模型输出的集成。

**📊 数据集**

使用 FoCus 个人化对话数据集进行训练与评估，包含多轮对话与 persona 描述。

**📈 对比分析**

与 Qwen2‑7B、Mistral‑7B、LLaMA‑3.1‑8B 等开源基线以及标准 DPO 版本对比，PersoDPO 在 Coh‑UniEval、C Score、UE Score、P Score 四项指标均明显优于基线，并在响应时间和失效率上表现更好。

**⚠️ 局限性**

局限性包括：依赖自动评估指标，可能无法完全反映真实对话质量；在 Coh‑UniEval 上略逊于 DPO；对未见 persona 的泛化性尚待进一步验证。

---

## 330. Beyond Unimodal Shortcuts: MLLMs as Cross-Modal Reasoners for Grounded Named Entity Recognition

**arXiv ID:** 2602.04486 | [PDF](https://arxiv.org/pdf/2602.04486v1)

**作者:** Jinlong Ma `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 59784 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将多模态命名实体识别（GMNER）转化为端到端的生成式推理任务，提出并实现了MCR框架，用以消除多模态语言模型在推理时的单模态偏差。

**💡 创新点**

创新点在于：①通过多样化的推理模式注入（MRSI）显式构建跨模态一致性约束；②采用基于可验证奖励的强化学习（CVO）与GRPO动态对齐推理轨迹，迫使模型在视觉与文本之间进行严谨的交叉验证，从而有效抑制视觉/文本偏差。

**🔧 技术方法**

主要技术包括：多风格推理模板生成、约束感知推理链注入、可验证奖励函数（实体计数、跨度、类型、视觉一致性与归属奖励）、基于群体优势的GRPO优化。

**📊 数据集**

使用的公开数据集有：Twitter‑GMNER（GMNER、MNER、EEG子任务）、MNER‑MI（多图像MNER）和GREC（视觉定位）。

**📈 对比分析**

与传统管道式与统一式基线（如SCANNER、MQSPN、GLM4.5VL等）以及CoT、Few‑Shot、SFT等训练策略对比，MCR在GMNER的F1从基线提升约11.9%，在MNER提升≥2.3%，在EEG提升约10.9%，并显著降低视觉/文本偏差（N‑Rate、N‑Acc等指标接近零）。

**⚠️ 局限性**

局限性：依赖底层多模态语言模型的预训练知识，若实体在预训练语料中未出现，模型的识别与定位能力可能受限，难以泛化到完全新颖的实体。

---

## 331. Incongruity-sensitive access to highly compressed strings

**arXiv ID:** 2602.04523 | [PDF](https://arxiv.org/pdf/2602.04523v1)

**作者:** Ferdinando Cicalese `[一作]`, Cristian Urbina `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究了在高度压缩字符串中利用字符与其周围字符的不一致性实现更快随机访问的方法。

**💡 创新点**

提出了不一致性敏感访问的概念，并证明在局部平衡的 RLSLP、块树以及满足 α‑收缩条件的解析中，访问时间可以改为与字符所在最长重复子串长度相关的对数，显著优于传统的 O(log n) 访问。

**🔧 技术方法**

使用了距离敏感前驱查询、离散化区间搜索树、局部平衡化技术以及解析收缩转换等数据结构与算法。

**📊 数据集**

未使用实验数据集，全部为理论分析与证明。

**📈 对比分析**

通过与已有最优压缩访问结构（如 AVL 语法、块树、LZ77 解析等）对比，得到在最坏情况下的访问时间从 O(log n) 降至 O(log ℓ_q) 或 O(h_q+log_w ℓ_q)，并证明空间保持在 O(g_rl) 或 O(b log_w(n/b))。

**⚠️ 局限性**

仅在满足 α‑收缩或局部平衡条件的解析下适用；对无此性质的压缩表示仍无法突破传统对数下界；实现复杂度和实际效果未做实验验证。

---

## 332. Informing Robot Wellbeing Coach Design through Longitudinal Analysis of Human-AI Dialogue

**arXiv ID:** 2602.04478 | [PDF](https://arxiv.org/pdf/2602.04478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 333. A Unified Complementarity-based Approach for Rigid-Body Manipulation and Motion Prediction

**arXiv ID:** 2602.04522 | [PDF](https://arxiv.org/pdf/2602.04522v1)

**作者:** Bingkun Huang `[一作]` (Technical University of Munich), Riddhiman Laha `[通讯]` (Northeastern University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种统一的基于补偿性（Complementarity）的离散时间框架（Unicomp），可同时处理自由空间运动和摩擦性接触，并在机器人操作中实现实时预测与规划。

**💡 创新点**

创新点在于：①将自由运动与非点接触的摩擦动力学统一到同一补偿性问题中；②利用最大功率耗散原理得到的椭圆形极限面（Limit Surface）模型，实现了对接触力矩和扭矩的耦合描述；③在同一数学框架内实现了多体接触、工具-物体相互作用以及全身避障的实时求解。

**🔧 技术方法**

采用的主要技术包括：补偿性（LCP、NCP、MCP）理论、离散时间时步法（Euler），椭圆极限面摩擦模型，基于 ECP 的接触冲量模型，线性不等式约束的 LCP 求解，用于避障的球体几何逼近与一阶约束投影。

**📊 数据集**

数据集：本工作主要基于仿真数据（使用自研的 Unicomp 仿真器）以及真实机械臂的实验（如基于 Stony Brook 或 TUM 的机器人平台），并未引用公开数据集；实验中使用的对象为各种形状的刚体（块、桌子、哑铃），以及滑动、摩擦参数。

**📈 对比分析**

对比方法：与 MuJoCo 的物理引擎进行对比，评估接触模式转移、能量守恒和动力学平滑度。结果显示：Unicomp 在滑动和破裂模式下不出现伪力、碰撞穿透，轨迹更平滑；在 1000 Hz 的时步下实现实时计算；与 MuJoCo 相比，能量曲线更为稳定，且接触力分布更符合物理预期。

**⚠️ 局限性**

局限性：①模型仍假设接触面为凸或可用椭圆极限面近似，难以处理高度非凸或大范围分布式接触；②需要事先设定摩擦楕円参数（e_t, e_o, e_r）和摩擦系数 μ，参数敏感性较高；③在极端大负载或高速碰撞时，离散时间时步近似可能导致数值不稳定；④当前实现仅针对刚体，未考虑柔性或弹性接触。

---

## 334. OSCAgent: Accelerating the Discovery of Organic Solar Cells with LLM Agents

**arXiv ID:** 2602.04510 | [PDF](https://arxiv.org/pdf/2602.04510v1)

**作者:** Zhaolin Hu `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 80586 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出OSCAgent多代理框架，用LLM驱动的检索增强设计、分子生成与系统评估循环，实现无人工干预的有机太阳能电池受体分子自动发现。

**💡 创新点**

创新点在于：①检索增强策略将文献实验高效受体与动态候选库融合；②多代理闭环（Planner、Generator、Experimenter）实现自适应迭代；③多模态PCE预测结合不确定度量化；④同时评估合成可行性与轨道能量。

**🔧 技术方法**

技术包括：大语言模型（GPT‑5）与多代理协作；基于图/SMILES的对比学习预训练；多任务（PCE+LUMO）微调与不确定度回归；分子指纹与Mixture‑of‑Experts融合；RDKit合成可行性评估；Tanimoto相似检索与K‑center贪婪算法。

**📊 数据集**

数据集为：Sun等人整理的1,027个实验受体（包含PCE与SAscore）；Lopez等人Harvard Clean Energy Project的51,256个计算受体，用于预训练。

**📈 对比分析**

与传统分子生成（BRICS、VAE）、遗传算法（SMILES‑GA、Graph‑GA）以及LLM基方法（BioT5、Few‑shot）对比，OSCAgent在有效性（validity、平均PCE）、多样性（uniqueness、novelty）和分布相似度（四种指纹）均优于所有基线，平均PCE提升至约14.6%，可行性SAscore低于8，分布相似度最高。

**⚠️ 局限性**

局限性包括：依赖预训练模型和外部检索数据库，若检索库不完整或更新不及时可能导致偏差；PCE预测仍受实验数据稀缺影响；当前仅关注受体分子，未覆盖整个设备层面；实验验证仅基于计算指标，缺少实际光伏测量。

---

## 335. ReThinker: Scientific Reasoning by Rethinking with Guided Reflection and Confidence Control

**arXiv ID:** 2602.04496 | [PDF](https://arxiv.org/pdf/2602.04496v1)

**作者:** Zhentao Tang `[一作]` (Noah's Ark Lab, Huawei), Mingxuan Yuan `[通讯]` (Noah's Ark Lab, Huawei)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ReThinker 框架，使用置信度驱动的 Solver–Critic–Selector 三阶段结构，并通过自回放轨迹合成与工具调用实现专家级科学推理。

**💡 创新点**

创新点在于：①置信度可动态分配计算资源；②自动轨迹合成与循环利用的无人工标注方案；③基于拉丁方的候选位置扰动与困惑度驱动的多轮置信度引导。

**🔧 技术方法**

采用多模态工具调用（Python、Web 搜索、解析）、LLM 交互式推理、自动轨迹合成与校验、困惑度置信度估计、Latin Square 位置随机化以及多轮自适应选择。

**📊 数据集**

使用 HLE、GAIA、XBench‑DeepSearch 三个专家级科学推理基准，并通过自动生成的 QA 与轨迹数据构建训练集。

**📈 对比分析**

在三大基准上与多种工具增强的基础模型（Gemini‑3‑Pro、GPT‑5 等）以及现有推理框架（WebExplorer、MiroThinker 等）对比，ReThinker 分别取得 52.18%/81.6%/90.0% 的 SOTA 性能，显著优于基线。

**⚠️ 局限性**

局限性包括：对 LLM 生成轨迹质量敏感；困惑度置信度估计易受偏差影响；工具调用开销大；在极不确定或未知领域的泛化仍有限。

---

## 336. DOS: Dual-Flow Orthogonal Semantic IDs for Recommendation in Meituan

**arXiv ID:** 2602.04460 | [PDF](https://arxiv.org/pdf/2602.04460v1)

**作者:** Junwei Yin `[一作]` (Meituan), Xingxing Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种 Dual-Flow Orthogonal Semantic IDs (DOS) 框架，用来生成既包含上下文信息又能高效量化的语义 ID，用于提升 Meituan 平台的生成式推荐质量。

**💡 创新点**

创新点包括：① 双流(user‑item)融合架构，直接把用户点击序列和目标商品映射到同一语义空间，实现与生成任务的对齐；② 正交残差量化 (ORQ) 模块，先通过正交旋转保留全语义信息，再用互信息筛选主干特征并残差量化，最大限度减少量化带来的语义损失。

**🔧 技术方法**

主要技术手段：Transformer 编码器、共享代码本、正交矩阵旋转、互信息驱动的维度筛选、残差量化、LLM（Qwen3‑0.6B‑embedding）生成语义嵌入、A/B 测试与业务 KPI 评估。

**📊 数据集**

使用了 Meituan 自有的生产数据：24 M 项目级别的商品语义嵌入，60 天用户交互日志（约 1.8 亿条），并按业务类型划分为 Busi_A–Busi_D 四类进行实验。

**📈 对比分析**

与 RQ‑KMeans、RQ‑VAE、DAS 等基线在 AUC/F1、Hit@10 等指标上对比。DOS 在 AUC 0.8763、F1‑Score 0.8057、Hit@10 0.0676（全业务）上表现最佳；在线 A/B 测试显示收入提升 1.15%。

**⚠️ 局限性**

局限性：① 依赖 LLM 嵌入，若 LLM 质量下降会影响性能；② 代码本共享虽对齐语义空间，但对新业务类型的适配仍需进一步研究；③ 模型结构较为复杂，训练成本和推理延迟相对较高。

---

## 337. Robot-Assisted Group Tours for Blind People

**arXiv ID:** 2602.04458 | [PDF](https://arxiv.org/pdf/2602.04458v1)

**作者:** Yaxin Hu `[一作]` (University of Wisconsin Madison), Chieko Asakawa `[通讯]` (IBM Research)

**通讯引用:** 4377 | [OpenAlex ID](https://openalex.org/A5091550931)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款移动辅助机器人，用于支持盲人参与与视障人士混合可视化的博物馆导览组游，并通过访谈和现场实验评估其可用性与影响。

**💡 创新点**

创新点包括：①把导览机器人定位为社会交互伙伴，兼顾环境感知、群体交互意识和自主导航；②将机器人与导游的双向交互机制（如呼叫导游、实时位置反馈）纳入设计；③采用语言模型生成周边描述，提升盲人对环境的认知；④在真实博物馆场景下进行大规模人机交互评估。

**🔧 技术方法**

技术手段包括：①基于开源Cabot平台的移动机器人；②三摄RGB‑D+LiDAR感知，MMDetection检测附近人员；③UWB实现机器人与导游定位；④GPT‑4o 视觉语言模型生成环境描述；⑤骨传导耳机+手机App实现语音反馈；⑥四键机械手柄实现功能触发；⑦WebSocket实现实时通信。

**📊 数据集**

数据来源为研究内部数据：访谈样本（5名盲人+5名博物馆专家），现场实验样本（8名盲人+8名视人+1名导游），以及机器人日志与录制视频；未使用公开数据集，所有环境信息来自博物馆内部地图与展品描述。

**📈 对比分析**

评估方式：用户体验问卷（SUS平均90.6，RoSAS正面属性高），功能使用频次与时间段分布分析，视频编码与日志匹配；相较于传统盲人导览机器人（主要是单人导航），本系统在安全感、信息获取和社交参与方面得到显著提升；但在机器人自主跟随与实时响应方面仍不如人工导游。

**⚠️ 局限性**

局限性：①样本量与情境单一，仅在一所科学博物馆进行，缺乏对不同展览类型和大规模团体的普适性验证；②机器人采用遥控操作，未实现完全自主跟随；③功能实现延迟与人机交互节奏冲突导致部分盲人体验被打断；④缺乏客观的性能指标（如导航误差、交互时延），主要依赖主观评估；⑤未来需进一步探索个性化配置与多方协同控制。

---

## 338. Growth First, Care Second? Tracing the Landscape of LLM Value Preferences in Everyday Dilemmas

**arXiv ID:** 2602.04456 | [PDF](https://arxiv.org/pdf/2602.04456v1)

**作者:** Zhiyi Chen `[一作]` (University of Southern California), Luca Luceri `[通讯]` (University of Southern California)

**通讯引用:** 1157 | [OpenAlex ID](https://openalex.org/A5088304530)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用来自四个 Reddit 求助社区的真实决策情境，构建了一个自下而上、层级化的价值框架，并通过该框架评估大型语言模型在日常困境中的价值取向；

**💡 创新点**

创新点在于：①使用真实世界的建议对话而非模板化情景；②采用 GPT‑4o 自动提取价值并自下而上聚类生成四层价值体系；③系统评估多模型在不同子社区的价值偏好，揭示 LLM 价值同质化风险；

**🔧 技术方法**

技术方法包括：GPT‑4o 价值提取、k‑means 与嵌入聚类、手工验证、价值共现网络、赢率指标、bootstrap 统计检验；

**📊 数据集**

数据集为 5,728 条来自 r/AskMenAdvice、r/AskWomenAdvice、r/CareerAdvice、r/FriendshipAdvice 的真实困境，已公开发布于 GitHub；

**📈 对比分析**

与传统基于手工情景或单一价值分类的研究相比，本工作在价值层级完整性、模型对比（GPT‑4o、DeepSeek‑V3.2‑Exp、Gemini‑2.5‑Flash）以及多社区分析方面均有显著提升；在所有模型中，Exploration & Growth 的赢率显著高于 Benevolence & Connection，表明 LLM 倾向于偏好成长与探索；

**⚠️ 局限性**

局限性包括：①将每个选项视为单一主导价值，忽略多价值交织；②仅评估三款模型与特定提示，结果可能不具普适性；③未考察用户实际接受度与决策后果，无法验证对社会行为的真实影响。

---

## 339. Seg-ReSearch: Segmentation with Interleaved Reasoning and External Search

**arXiv ID:** 2602.04454 | [PDF](https://arxiv.org/pdf/2602.04454v1)

**作者:** Tianming Liang `[一作]` (Sun Yat-sen University), Wei-Shi Zheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 21691 | [OpenAlex ID](https://openalex.org/A5108050904)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Seg‑ReSearch框架，通过多轮交互式推理与外部搜索实现视频对象分割；

**💡 创新点**

创新点在于：①设计了层级奖励机制平衡稀疏奖励与过程监督；②实现了跨模态链式思维（MCoT）与搜索引擎的无缝融合；③创建了需要外部知识的视频分割基准OK‑VOS；

**🔧 技术方法**

采用多模态大型语言模型（如Qwen3‑VL‑Instruct）、搜索接口（Google Search）、SAM2分割器、RL算法GRPO以及自定义奖励函数；

**📊 数据集**

主要使用OK‑VOS数据集（1,000个测试样本，150段视频，500个目标），并在ReasonSeg、ReasonVOS等传统推理分割基准上进行评测；

**📈 对比分析**

与现有RVOS专家模型、MLLM推理分割模型及搜索增强模型对比，Seg‑ReSearch在OK‑VOS上整体J&F从≈34提升至≈60（4B版≈46，8B版≈51），在ReasonSeg、ReasonVOS上也刷新SOTA；

**⚠️ 局限性**

局限性包括：对搜索引擎质量高度依赖、搜索次数与成本受限、易受网络信息偏见和隐私泄露风险，且模型仍需大规模标注训练。

---

## 340. EgoActor: Grounding Task Planning into Spatial-aware Egocentric Actions for Humanoid Robots via Visual-Language Models

**arXiv ID:** 2602.04515 | [PDF](https://arxiv.org/pdf/2602.04515v1)

**作者:** Yu Bai `[一作]` (Beijing Academy of Artificial Intelligence), Börje F. Karlsson `[通讯]` (Beijing Academy of Artificial Intelligence)

**通讯引用:** 874 | [OpenAlex ID](https://openalex.org/A5015011965)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 EgoActor，一种统一的视觉‑语言模型，用于将自然语言任务指令直接映射为人类机器人可执行的低层动作序列，涵盖移动、主动感知、操作和人机交互。

**💡 创新点**

创新点在于：① 将多种动作类型（运动、头部转动、操作、交互）统一为语言描述并直接预测；② 通过丰富的 egocentric 视觉数据和空间推理训练，显著提升了对未知环境的几何理解与自适应；③ 在不使用深度传感器、额外摄像头或复杂手动标注的前提下实现实时（<1s）推理。

**🔧 技术方法**

技术方法包括：使用 Qwen3‑VL（Transformer‑based VLM）并通过 LoRA 微调；对动作使用结构化语言（SLAs）和自然语言（NLAs）两种表示；利用多模态训练数据、空间推理问答、虚拟模拟和实时 DAgger 经验；配合 Unitree G1 机器人执行控制。

**📊 数据集**

数据集涵盖：EgoTaskQA、额外 130 条 egocentric 影片、398 条本地环境录制、3% VLN‑CE、714 条 Habitat‑Sim 导航轨迹、MindCube 空间推理、GQA 视觉‑语言、RoboVQA/EgoPlan/ALFRED 规划、无监督运动预测、DAgger 实时轨迹，总计约 1.5M 训练样本。

**📈 对比分析**

与 NaVid、Uni‑NaVid、NaVILA 等基准模型在真实人机交互、移动操作、通过门口等“穿越性”任务中进行对比。EgoActor 在大多数指标上显著优于基线，尤其在多任务序列、跨场景泛化、以及通过窄门等细粒度导航方面取得 70‑90% 以上成功率；在虚拟 VLNCE 任务中，8B 版本在 <3m 距离阈值下可达 50%+ 成功率，且自然语言动作的 F1 约 0.6，远超基线。

**⚠️ 局限性**

局限性包括：① 对高度变化（站立/蹲下）等姿态控制仍主要在仿真中实现，实机缺乏完整支持；② 在极端遮挡、低光或高动态场景下，视觉识别与动作推断仍易受误判；③ 需要进一步提升对多人的细粒度识别和复杂交互指令的鲁棒性；④ 依赖单一 RGB 摄像头，无法充分利用深度信息。

---

## 341. A labeled dataset of simulated phlebotomy procedures for medical AI: polygon annotations for object detection and human-object interaction

**arXiv ID:** 2602.04624 | [PDF](https://arxiv.org/pdf/2602.04624v1)

**作者:** Raúl Jiménez Cruz `[一作]` (Tecnológico de Monterrey), Barbara Weber `[通讯]` (Institute of Computer Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建并公开了一个11884张图像的模拟抽血数据集，提供五类关键工具的多边形分割标注。

**💡 创新点**

创新点在于细粒度多边形分割、YOLOv8兼容格式、SSIM过滤与人脸匿名化。

**🔧 技术方法**

使用高分辨率摄像、SSIM相似度过滤、人脸检测模糊、Roboflow半自动标注与YOLOv8分割训练技术。

**📊 数据集**

采用自制的11884张图像数据集，按70/15/15划分为训练、验证、测试集。

**📈 对比分析**

通过YOLOv8分割模型训练验证，验证集mAP50/95达到高水平，验证了标注质量优秀。

**⚠️ 局限性**

局限在于仅模拟环境、仅五类物体、无时间序列标签、缺乏临床多样性、潜在隐私边缘风险。

---

## 342. Resilient Load Forecasting under Climate Change: Adaptive Conditional Neural Processes for Few-Shot Extreme Load Forecasting

**arXiv ID:** 2602.04609 | [PDF](https://arxiv.org/pdf/2602.04609v1)

**作者:** Chenxi Hu `[一作]`, Yunhe Hou `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于自适应条件神经过程的极端负荷预测模型 AdaCNP。

**💡 创新点**

创新点在于使用目标感知的相似度嵌入对上下文进行动态加权，既保持可解释性又能在极端事件少样本时快速适应。

**🔧 技术方法**

核心技术包括条件神经过程、轻量级相似度评分网络、温度调节的 softmax 加权、概率分布输出与负对数似然训练。

**📊 数据集**

实验使用美国 PJM 与 ISO-NE 两个真实负荷数据集，对极端负荷天进行标注与评估。

**📈 对比分析**

与 CNP、ANP、GP、NP 等基线相比，AdaCNP 在 MSE、NLL 与 Pinball 损失上均实现了 20% 以上的提升，特别是在极端负荷样本稀缺时表现最优。

**⚠️ 局限性**

局限性包括对极端事件的检测仍需阈值设定、在极度稀缺样本下仍可能出现不稳定或高方差预测，以及目前仅验证于负荷预测，泛化到其他领域需进一步研究。

---

## 343. RexBERT: Context Specialized Bidirectional Encoders for E-commerce

**arXiv ID:** 2602.04605 | [PDF](https://arxiv.org/pdf/2602.04605v1)

**作者:** Rahul Bajaj `[一作]`, Anuj Garg `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并公开了一系列针对电商领域的Encoder-only Transformer模型RexBERT，并提供完整的预训练与评估流程。

**💡 创新点**

创新点在于构建350B-token电商语料Ecom‑niverse、三阶段渐进式预训练（通用→长上下文→域化）以及引入 Guided MLM 对重要实体进行重点掩码。

**🔧 技术方法**

技术包括ModernBERT改进的RoPE、GeGLU、交替全局/局部注意力、Flash Attention、StableAdamW优化器以及自定义span-aware掩码。

**📊 数据集**

主要使用FineFineWeb提取的电商文本构成Ecom‑niverse，评估数据集为Amazon ESCI（商品搜索）和GLUE基准。

**📈 对比分析**

与ModernBERT、Ettin、DistilBERT等模型比较，RexBERT在token分类、语义相似度等电商任务上即使参数更少也能取得更高的准确率和Spearman相关系数，GLUE上亦保持竞争力。

**⚠️ 局限性**

局限性包括缺乏对Guided MLM贡献的独立验证、对域外迁移能力的评估不足，以及在极端长文本或多语言场景下仍需进一步测试。

---

## 344. AI in Education Beyond Learning Outcomes: Cognition, Agency, Emotion, and Ethics

**arXiv ID:** 2602.04598 | [PDF](https://arxiv.org/pdf/2602.04598v1)

**作者:** Lucile Favero `[一作]` (ELLIS Alicante), Nuria Oliver `[通讯]` (ELLIS Alicante)

**通讯引用:** 14898 | [OpenAlex ID](https://openalex.org/A5013727792)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过整合认知、代理、情感和伦理四个维度，对AI在教育中潜在的非预期危害进行系统性分析，并提出整合框架和实践清单。

**💡 创新点**

提出跨学科的四维度框架，将认知负荷、学习代理、情绪影响与伦理风险整合，揭示它们如何互相强化形成系统性危害，并给出针对设计者、学生、机构的行动清单。

**🔧 技术方法**

主要采用文献综述与理论建模，未使用具体算法；讨论了生成式AI、聊天机器人、智能辅导系统等技术。

**📊 数据集**

由于是综述性研究，无特定数据集，引用了多项教育与心理学研究的结果。

**📈 对比分析**

工作未进行实验对比，性能评估基于文献证据与理论推导，未给出定量指标。

**⚠️ 局限性**

主要限制是缺乏实证验证、数据来源分散、对不同教育情境的细化不足，且框架仍需在多元文化和制度环境下进一步检验。

---

## 345. Real-time processing of analog signals on accelerated neuromorphic hardware

**arXiv ID:** 2602.04582 | [PDF](https://arxiv.org/pdf/2602.04582v1)

**作者:** Yannik Stradmann `[一作]` (Heidelberg University), Laura Kriener `[通讯]` (University of Zürich)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在BrainScaleS-2混合信号神经形态平台上实现了实时模拟信号处理与执行器控制的完整芯片内流水线，完成了双耳声音定位并通过舵机对噪声源进行跟踪。

**💡 创新点**

首次演示了连续值传感器信号直接注入芯片模拟计算单元、利用嵌入式微处理器完成执行器输出，并构建了完全在芯片内部完成感知-计算-执行的闭环系统。

**🔧 技术方法**

使用BrainScaleS‑2 ASIC的512个可配置神经元、基于Jeffress模型的延迟链和相位检测单元、前置模拟滤波、嵌入式处理器读取计数器并产生PWM/串口指令。

**📊 数据集**

采用双声道麦克风采集的实时音频以及用音频卡回放的带有可调时间延迟的录制打击声（clap）作为实验数据；不使用公开数据集。

**📈 对比分析**

对比方法主要是通过测量系统对不同ITD（±149 µs）下的方向检测线性度和分布宽度。结果显示平均延迟约0.5 ms，方向误差分布宽度约24个神经元单位，对应约1.5°空间分辨率；与传统需外部FPGA预处理的系统相比，延迟更短、功耗更低。

**⚠️ 局限性**

受限于延迟链长度和突触时间常数，当前的空间分辨率仅为约1.5°；网络时域窗口宽阔导致分辨率受限；枚举顺序引入的偏置；需要更复杂的抑制与竞争机制以提升精度。

---

## 346. AIANO: Enhancing Information Retrieval with AI-Augmented Annotation

**arXiv ID:** 2602.04579 | [PDF](https://arxiv.org/pdf/2602.04579v1)

**作者:** Sameh Khattab `[一作]` (University Hospital Essen), Julian Friedrich `[通讯]` (University Hospital Essen)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一款针对信息检索任务的 AI 赋能注释工具 AIANO，支持半自动化的多块式协同工作流程

**💡 创新点**

创新点在于将 LLM 与人工协作紧密集成，提供多模式块（Plain、AI Solo、Human‑AI Collaborative）和全文检索，从而显著提升注释效率与质量

**🔧 技术方法**

使用了 OpenAI/Anthropic 等 LLM 通过 API 接入，前端 React+FastAPI 微服务架构，PostgreSQL 存储，支持 Docker 部署

**📊 数据集**

使用了 60 篇德语通用知识短文与 9 个问答对（单文档 5 题，多文档 4 题），以及 Meta 的 Llama 70B 生成答案

**📈 对比分析**

通过与 Label Studio 的对照实验，AIANO 在任务完成时间下降 40%、认知负荷降低、Precision、Recall、F1 分别提升约 2.5%、12.8%、9.3%

**⚠️ 局限性**

局限包括样本量仅 15 位参与者、只与单一基线工具比较、验证范围局限于德语短文与问答场景，未评估多语言、多任务与 LLM 误差影响

---

## 347. Continual Learning through Control Minimization

**arXiv ID:** 2602.04542 | [PDF](https://arxiv.org/pdf/2602.04542v1)

**作者:** Sander de Haan `[一作]` (Institute of Neuroinformatics), Benjamin F. Grewe `[通讯]` (Institute of Neuroinformatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于控制最小化的持续学习框架 Equilibrium Fisher Control (EFC)，将参数正则化转换为神经元级保持信号，使学习与保持在网络动力学中竞争。

**💡 创新点**

创新点在于：1) 通过保持信号与学习信号的乘性耦合，实现学习过程中的动态曲率估计，自动获得完整的先前任务曲率（持续自然梯度）而无需显式存储；2) 在动力学中提前过滤掉干扰梯度，从而在无任务标识的类增量学习中实现任务辨别。

**🔧 技术方法**

采用 least‑control 原理、神经动力学等价于控制问题、对角 Fisher 生成保持信号、动态迭代至平衡点、梯度预条件化等技术。

**📊 数据集**

使用 Split‑MNIST、Split‑CIFAR10、Split‑Tiny‑ImageNet 等标准连续学习基准数据集。

**📈 对比分析**

与 EWC、online EWC、Synaptic Intelligence 等正则化方法以及 DER++ 回放方法进行比较；在类增量学习任务中，EFC 取得 51%（Task‑IL）/50%（Class‑IL）准确率，明显优于传统正则化（约 20%）且逼近回放（约 70%）；在任务增量学习中与正则化方法相当。

**⚠️ 局限性**

局限性包括：1) 需要多轮前向/反馈迭代，计算开销大，难以扩展到大型网络；2) 需要额外的动态超参数调节，易导致不稳定；3) 仍需存储对角 Fisher，随着参数变化可能失准，未能完全消除存储曲率的漂移问题。

---

## 348. Focus-LIME: Surgical Interpretation of Long-Context Large Language Models via Proxy-Based Neighborhood Selection

**arXiv ID:** 2602.04607 | [PDF](https://arxiv.org/pdf/2602.04607v1)

**作者:** Junhao Liu `[一作]` (Peking University), Xin Zhang `[通讯]` (Peking University)

**通讯引用:** 65194 | [OpenAlex ID](https://openalex.org/A5100777470)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Focus‑LIME 框架，利用代理模型进行邻域筛选后再用目标 LLM 进行精准词级归因；

**💡 创新点**

创新点在于将“预筛选+精细归因”拆分为两阶段，解决长文本归因稀释问题，同时保持解释可信度；

**🔧 技术方法**

技术包含代理模型驱动的层级稀疏筛选、受限扰动生成、基于 LIME 的线性回归归因；

**📊 数据集**

使用长文本 QA 数据集 CUAD（法律合同）和 Qasper（科研论文）进行实验；

**📈 对比分析**

与原始 LIME、纯代理解释和无代理 Focus‑LIME 进行对比，实验表明 AOPC 指标显著提升，解释更符合人类标注且计算成本可控；

**⚠️ 局限性**

局限性包括对代理模型的依赖、在极大规模文本上仍需进一步优化筛选策略、以及对不同 LLM 结构的适配需进一步验证。

---

## 349. Act, Sense, Act: Learning Non-Markovian Active Perception Strategies from Large-Scale Egocentric Human Data

**arXiv ID:** 2602.04600 | [PDF](https://arxiv.org/pdf/2602.04600v1)

**作者:** Jialiang Li `[一作]` (Shanghai Jiao Tong University), Wenzhao Lian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 561 | [OpenAlex ID](https://openalex.org/A5017678179)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种认知与记忆感知-语言-动作框架CoMe-VLA，用大规模人类视角数据学习主动感知策略，并将其迁移到机器人执行；

**💡 创新点**

将主动感知建模为非马尔可夫决策过程，并通过认知辅助头与双轨记忆系统实现信息增益与决策分支的神经近似；

**🔧 技术方法**

使用Qwen3‑VL‑2B视觉语言模型、流匹配动作解码器、双轨（视觉+本体）Transformer记忆、三阶段预训练+微调；

**📊 数据集**

利用公开的CaptainCook4D、Ego‑Exo4D人类第一人称数据，以及通过VR遥操作收集的机器人数据；

**📈 对比分析**

与OpenVLA‑OFT、π0.5、ACT和Diffusion Policy等基线对比，CoMe‑VLA在5个长时序主动感知任务中平均成功率达87.3%，搜索时间显著下降；

**⚠️ 局限性**

记忆窗口固定且短，难以处理极长时序任务；迁移仍需机器人遥操作示例，且对特定硬件与环境的适应有限。

---

## 350. Harmonia: Algorithm-Hardware Co-Design for Memory- and Compute-Efficient BFP-based LLM Inference

**arXiv ID:** 2602.04595 | [PDF](https://arxiv.org/pdf/2602.04595v1)

**作者:** Xinyu Wang `[一作]` (Shanghai Jiao Tong University), Weifeng He `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3479 | [OpenAlex ID](https://openalex.org/A5100351675)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Harmonia 框架，将 LLM 推理中的线性层与注意力层激活统一转为 Block Floating Point (BFP)，并设计可重构计算单元以支持混合精度 BFP-INT/BFP-BFP 运算，显著压缩 KV 缓存。

**💡 创新点**

创新在于将 BFP 统一应用于全部激活层，提出异步位分配与离线-在线混合异常值平滑，利用 8/4 位 BFP 对 KV 缓存进行 68.75% 压缩，并设计单一可重构 PE 与实时 BFP 转换器实现硬件共享。

**🔧 技术方法**

采用 BFP 格式、异步位分配、离线-在线混合异常值平滑、可重构混合精度 PE、实时 BFP 转换、可切换列/行数据流、重量仅 INT4 量化等算法硬件协同技术。

**📊 数据集**

在 WikiText2 上评估零样本困惑度，在 LongBench 4K 上评测多任务准确率，并对 Llama、OPT、Mistral 等八款 1–13B LLM 进行基准。

**📈 对比分析**

与 Omniquant、FIGNA、Anda、KIVI 等权重量化与 BFP 基线进行对比，结果显示 Harmonia 在 4B mantissa KV 缓存下平均 0.3% 准确率下降，面积效率提升 3.84×，能效提升 2.03×，速度提升 3.08×，并在长序列 2K–16K 上显著缩短周期与功耗。

**⚠️ 局限性**

仍受 BFP 误差与 KV 异常值处理的复杂度限制，极低位 (≤4bit) 的 KV 压缩在部分模型上会出现 5% 以上准确率下降，并且对超大规模模型或自回归推理中的动态分组调度仍有实现难度。

---

## 351. Optimal conversion from Rényi Differential Privacy to $f$-Differential Privacy

**arXiv ID:** 2602.04562 | [PDF](https://arxiv.org/pdf/2602.04562v1)

**作者:** Anneliese Riess `[一作]` (Helmholtz Munich), Georgios Kaissis `[通讯]` (Hasso-Plattner-Institut)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文证明了将Rényi差分隐私（RDP）转换为f‑差分隐私（f‑DP）时，利用所有RDP阶的隐私区域交集（即取点wise最大）得到的转换规则是唯一最优的黑箱方法。

**💡 创新点**

创新点在于给出了RDP→f‑DP转换的理论极限：通过凸几何与2‑cut技术证明任何只依赖RDP配置的黑箱规则都无法得到更紧的误差权衡函数，并用二项式机制构造逼近边界的示例。

**🔧 技术方法**

主要技术包括：RDP隐私区域的凸性证明、2‑cut（把多维分布降至二项分布）与数据处理不等式、单阶RDP隐私区间的解析下边界、以及对点wise最大构造的严谨证明。

**📊 数据集**

论文没有使用具体数据集，而是基于理论分析和抽象机制（如高斯机制、随机响应机制）进行讨论。

**📈 对比分析**

作者通过对比已知的高斯机制的解析f‑DP曲线和其由RDP转换得到的下界，展示了该方法在所有RDP配置下的最优性；在实践中，该方法无需求解复杂变分问题，计算量仅为单阶曲线的点wise最大。

**⚠️ 局限性**

局限性在于该最优规则仅在仅知RDP配置的黑箱情形下成立，对于特定机制（如高斯机制）其得到的下界仍然可能远离真实的f‑DP曲线，说明若想进一步提升精度需额外利用机制的其他特征。

---

## 352. Textual Planning with Explicit Latent Transitions

**arXiv ID:** 2602.04557 | [PDF](https://arxiv.org/pdf/2602.04557v1)

**作者:** Eliezer Shlomi `[一作]` (Technion), Sarah Keren `[通讯]` (Technion)

**通讯引用:** 378 | [OpenAlex ID](https://openalex.org/A5047553225)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结的LLM嵌入空间中学习动作条件转移网络，替代逐词生成实现文本规划。

**💡 创新点**

提出EmbedPlan框架，利用冻结嵌入做轻量级状态转移预测并通过最近邻检索完成下一状态，系统性评估了在九个经典规划域的泛化边界。

**🔧 技术方法**

使用冻结预训练语言模型嵌入、轻量化MLP转移网络、InfoNCE对比损失、动作区分损失以及最近邻检索技术。

**📊 数据集**

基于ACPBench的九个经典PDDL规划域（Blocksworld、Depot、Ferry、Floortile、Goldminer、Grid、Logistics、Rovers、Satellite），共约300万转移。

**📈 对比分析**

通过插值、计划变体、外推、跨域、联合域和留一六种泛化协议评估Hit@k；插值达99.7%，外推约54.6%，跨域仅6.6%，显示在已见域内效果好但跨域转移差。

**⚠️ 局限性**

局限在于仅适用于可枚举状态空间，冻结嵌入导致表面形式敏感，跨域零样本泛化几乎不存在，缺乏闭环评估和开放式生成能力。

---

## 353. PersoPilot: An Adaptive AI-Copilot for Transparent Contextualized Persona Classification and Personalized Response Generation

**arXiv ID:** 2602.04540 | [PDF](https://arxiv.org/pdf/2602.04540v1)

**作者:** Saleh Afzoon `[一作]` (Macquarie University), Usman Naseem `[通讯]` (Macquarie University)

**通讯引用:** 2942 | [OpenAlex ID](https://openalex.org/A5077006200)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 PersoPilot，一种将用户画像识别与上下文分析相结合的 AI 助手，用于向终端用户和分析师提供透明、可解释的个性化服务。

**💡 创新点**

创新点包括：① 双模态框架，将实时对话、动态人物建模与分析师监督相互嵌入；② 上下文过滤机制，仅检索与当前任务最相关的特征；③ 基于主动学习的标注助手，持续迭代更新分类模型；④ 统一的 JSON 输出结构，实现快速集成。

**🔧 技术方法**

技术栈：LangChain + OpenAI GPT（Agent 端）+ BERT‑based persona extractor + Phi‑4‑mini‑instruct（标注端）+ TF‑IDF 相似度分类器；后端使用 FastAPI，前端基于 React；所有工具均通过结构化提示与少样本学习实现。

**📊 数据集**

主要数据集：ConvAI2 作为人物对话训练集；实验中还使用公开的 UniEval 框架进行响应质量评估；其它细节如社区数据库中的聚合兴趣未公开说明。

**📈 对比分析**

评估方法：使用 UniEval 评测自然性、连贯性、可证实性、可理解性，平均得分为 0.84、1.0、1.0、0.86；在示例场景中展示了用户交互与分析师标注流程，未给出对比基准，但说明了主动学习循环带来的持续改进。

**⚠️ 局限性**

局限性：① 依赖 OpenAI API，无法离线部署；② 主要评估基于少量对话样本，缺乏大规模真实数据验证；③ 仅在实验环境中展示，未评估跨域迁移或用户隐私安全问题。

---

## 354. A Human-Centered Privacy Approach (HCP) to AI

**arXiv ID:** 2602.04616 | [PDF](https://arxiv.org/pdf/2602.04616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 355. Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models

**arXiv ID:** 2602.04549 | [PDF](https://arxiv.org/pdf/2602.04549v1)

**作者:** Cem Eteke `[一作]` (Technical University of Munich), Enzo Tartaglione `[通讯]` (Telecom Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出NiFi方法，实现极低码率下3D Gaussian Splatting压缩并通过扩散模型进行恢复

**💡 创新点**

将一阶扩散蒸馏与即时映射到扩散轨迹点（t₀）相结合，显著提升极低比特率下的感知质量，并实现近1000倍的压缩率提升

**🔧 技术方法**

使用变分扩散蒸馏、低秩适配器、Stable Diffusion 3的潜在扩散模型、剪枝/量化压缩、t₀映射、感知损失与分类器自由引导

**📊 数据集**

利用DL3DV 1000场景合成压缩数据集进行训练，评估使用Mip-NeRF360、Tanks & Temples和DeepBlending三大数据集，并以HAC++作为压缩基准

**📈 对比分析**

与BM3D、SwinIR、DiffBIR、Difix3D、Img2Img等经典、深度学习和扩散基线对比，NiFi在LPIPS/DISTS指标上处于领先地位，压缩率从555 MB降至0.1 MB，实现约927倍压缩且保持接近未压缩3DGS的感知质量

**⚠️ 局限性**

恢复过程中对高频细节过度强化，易导致细粒度区域出现伪影（如自行车场景的草地）

---

## 356. Probabilistic Label Spreading: Efficient and Consistent Estimation of Soft Labels with Epistemic Uncertainty on Graphs

**arXiv ID:** 2602.04574 | [PDF](https://arxiv.org/pdf/2602.04574v1)

**作者:** Jonathan Klees `[一作]` (Osnabrück University), Matthias Rottmann `[通讯]` (Osnabrück University)

**通讯引用:** 6072 | [OpenAlex ID](https://openalex.org/A5038323169)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于图传播的概率标签扩散（PLS）方法，能够在极少人工注释下生成软标签并量化其不确定性。

**💡 创新点**

将传统标签传播推广至软标签、支持增量更新，并给出一致性与PAC学习的理论保证，解决了少量标注下软标签估计的问题。

**🔧 技术方法**

使用k‑NN稀疏图、Gaussian核构建亲和矩阵、CLIP/ResNet/ViT特征提取、UMAP降维、线性求解等技术实现高效传播。

**📊 数据集**

在CIFAR‑10、CIFAR‑10‑H、Animals‑10、EMNIST‑digits、Tiny‑ImageNet、MTSD等常见图像数据集上进行实验。

**📈 对比分析**

与无扩散、GKR、k‑NN、CLIP 0‑shot等基线比较，RMSE/ KL下降显著，获得数据中心化图像分类基准的SOTA。

**⚠️ 局限性**

受限于对特征空间平滑与局部一致性的假设，若特征不充分或噪声过大，扩散可能传播错误信息。

---

## 357. VRARE: Using Virtual Reality to Understand Accessibility Requirements of Color Blindness and Weakness

**arXiv ID:** 2602.04621 | [PDF](https://arxiv.org/pdf/2602.04621v1)

**作者:** Yi Wang `[一作]` (Deakin University), Thuong Hoang `[通讯]` (Deakin University)

**通讯引用:** 1386 | [OpenAlex ID](https://openalex.org/A5033351728)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

研究在讨论和理解可访问性需求（特别是色盲和色弱）时，VR方法与传统面谈方法的差异，并开发了支持色觉缺陷模拟与远程多人协作的VR系统。

**💡 创新点**

创新点在于将沉浸式VR与色觉缺陷模拟相结合，提供实时协作与可视化体验，验证VR能够显著降低工作负荷、提升用户体验，为可访问性需求获取提供了新的方法论。

**🔧 技术方法**

技术实现包括Unity3D + Post‑Processing插件的RGB通道转换色盲模拟，Meta Quest 3头显与双控制器，实时语音通信与头像同步，3D Web View嵌入网站渲染，以及NASA‑TLX和UEQ量表评估工具。

**📊 数据集**

数据集：实验未使用公开数据集，采用受试者在自建训练网站和研究者提供的健身网站上进行讨论；共招募24名软件工程专业本科生。

**📈 对比分析**

比较方法：采用within‑subjects设计，受试者先后使用VR与非VR方法讨论并评估同一网站；使用Wilcoxon符号秩检验比较NASA‑TLX工作负荷和UEQ用户体验。结果显示VR方法工作负荷显著降低（p<0.001），用户体验显著提升（p<0.001）。

**⚠️ 局限性**

局限性：样本量较小且受试者群体单一（软件工程本科生且未色盲），实验仅聚焦色觉缺陷场景，未评估长期使用或更复杂网站的效果，且对VR设备与网络延迟等技术限制未进行系统评估。

---

## 358. VILLAIN at AVerImaTeC: Verifying Image-Text Claims via Multi-Agent Collaboration

**arXiv ID:** 2602.04587 | [PDF](https://arxiv.org/pdf/2602.04587v1)

**作者:** Jaeyoon Jung `[一作]` (Soongsil University), Kunwoo Park `[通讯]` (Soongsil University)

**通讯引用:** 4121 | [OpenAlex ID](https://openalex.org/A5069120214)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套名为 VILLAIN 的多模态事实核查系统，利用多代理 VLM 在图文声明的多阶段推理中完成检索、分析、问答生成和判定。

**💡 创新点**

创新点包括：① 模态专属与跨模态分析代理协同提炼证据；② 通过 URL 内容填充扩展知识库；③ 采用迭代式问答生成与精细化证据挑选提升判定精度。

**🔧 技术方法**

使用的技术有：Gemini‑2.5‑Pro 作为核心 VLM，mxbai‑embed‑large‑v1 与 mxbai‑rerank‑large‑v1 进行文本检索与重排序，Ops‑MM‑embedding‑v1‑7B 用于图像检索，Playwright 自动化爬取网页内容，以及基于 prompt 的多代理协同推理框架。

**📊 数据集**

采用 AVerImaTeC 共享任务的数据集，包含图文声明、元数据及相关网页与图像证据。

**📈 对比分析**

与其他参赛系统比较，VILLAIN 在测试集上以 0.546 的 veracity 分数、最高的 Q‑Eval、Evid‑Eval 与 Justification 分数位居榜首，且 ablation 研究证实分析代理与知识库填充均显著提升性能。

**⚠️ 局限性**

局限性在于：仅在 AVerImaTeC 数据集上评测，未验证泛化能力；系统对每条声明需要八次 VLM 调用，计算成本高，需进一步探索轻量化实现。

---

## 359. ImmuVis: Hyperconvolutional Foundation Model for Imaging Mass Cytometry

**arXiv ID:** 2602.04585 | [PDF](https://arxiv.org/pdf/2602.04585v1)

**作者:** Marcin Możejko `[一作]` (University of Warsaw), Ewa Szczurek `[通讯]` (Institute of AI for Health)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种面向成像质量细胞测序（IMC）的超卷积基础模型，能够在不重新训练的情况下处理任意标记子集，实现虚拟染色、细胞表型识别和临床预测。

**💡 创新点**

创新点包括：1）使用标记感知的超卷积（hyperconvolution）动态生成卷积核，支持任意标记组合；2）全卷积架构结合自监督掩码重建，保持局部空间信息；3）引入高斯异方差回归提供像素级不确定性估计，提升结果可解释性。

**🔧 技术方法**

核心技术：超卷积网络（hypernetworks）+ ConvNeXt‑v2 视觉骨干 + 自监督掩码重建 + 高斯异方差损失 + 标记嵌入学习。

**📊 数据集**

使用截至目前最大的 IMMUcan 数据集：17M 256×256 图像块，23 个不同研究，共 122 个标记，涵盖 9 种组织学类型。

**📈 对比分析**

与基线 VirTues（Transformer）以及 U‑Net 等方法比较。结果显示在虚拟染色、单细胞表型分类和患者级临床预测中均取得显著优于 VirTues 的性能；同时计算效率提升约 4–5 倍（GFLOPs 90 vs 1049，推理时间 63s vs 244s）。

**⚠️ 局限性**

局限性：1）模型训练受限于可获得的标记与样本分布，稀有标记和低频组织表现欠佳；2）假设像素独立性，未建模空间或通道间的相关噪声；3）在极端零样本扩展（完全未知标记集）时仍存在不确定性。

---

## 360. Can LLMs capture stable human-generated sentence entropy measures?

**arXiv ID:** 2602.04570 | [PDF](https://arxiv.org/pdf/2602.04570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 361. PEPR: Privileged Event-based Predictive Regularization for Domain Generalization

**arXiv ID:** 2602.04583 | [PDF](https://arxiv.org/pdf/2602.04583v1)

**作者:** Gabriele Magrini `[一作]` (University of Florence), Pietro Pala `[通讯]` (University of Florence)

**通讯引用:** 4161 | [OpenAlex ID](https://openalex.org/A5031581349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在训练阶段利用事件摄像机作为特权信息，对RGB模型进行鲁棒性提升，使其在域迁移（尤其是昼夜转换）时表现更好，最终仅使用RGB推理；

**💡 创新点**

将学习使用特权信息（LUPI）转化为跨模态预测任务：RGB编码器不再与稀疏事件特征直接对齐，而是学习在共享潜在空间中预测事件潜在特征，从而保留语义丰富性并提取域不变特征；

**🔧 技术方法**

采用深度双模态网络（RGB编码器+任务头、事件编码器+Transformer解码器预测器），利用MSE预测损失对RGB特征进行正则化；同时使用标准监督任务损失；该框架兼容任意可导出的RGB特征提取器；

**📊 数据集**

在多任务、多域数据集上验证：Hard‑DSEC‑DET、FRED（Canonical、Challenging、Day‑to‑Night）、Cityscapes Adverse、Dark Zurich；

**📈 对比分析**

与基线RGB‑only和直接特征对齐（L2）相比，提出的PEPR在目标域的mAP和mIoU均提升，尤其在极端低光（Pitch Black）和昼夜切换时显著优于对齐方法；

**⚠️ 局限性**

局限性包括：需要有同步RGB‑事件对的训练数据（真实或通过ESIM模拟），对事件模拟质量敏感；预测目标的块大小与数量需要调参；框架在极端低光下仍难以获得足够的RGB信息。

---

## 362. From Competition to Collaboration: Designing Sustainable Mechanisms Between LLMs and Online Forums

**arXiv ID:** 2602.04572 | [PDF](https://arxiv.org/pdf/2602.04572v1)

**作者:** Niv Fono `[一作]` (Technion Israel Institute of Technology), Omer Ben-Porat `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个非货币化、信息不对称的GenAI与在线问答论坛协同框架，并通过序列交互模型评估两者的激励失配。

**💡 创新点**

创新点在于将两方视为合作博弈而非竞争，设计基于接受概率的GenAI提问策略与阈值分类器的论坛选择规则，并量化利用率恢复率（URR）。

**🔧 技术方法**

使用博弈理论建模、LLM（Pythia 6.9B、LLaMA 3.1 8B/指令版）评估问题困惑度，BERT分类器预测论坛接收概率，计算视图归一化与困惑度。

**📊 数据集**

采用 2024–2025 年间的 Stack Exchange 五个社区（技术、科学、数学、英文、机器学习）问答数据，构成候选问题池进行实验。

**📈 对比分析**

通过自定义的“利用率恢复率（EURR）”与 MPP、MaxSP、GreedyNP 三种启发式对比，实验结果显示 G‑Utility 策略在不完全信息下可恢复 55–66% 论坛效用、46–52% GenAI 效用，显著优于随机或贪心策略。

**⚠️ 局限性**

局限在于假设线性效用、离散化视图归一化、仅考虑文本问答而忽略多模态交互，且模型仅在公开数据上验证，真实策略与信息共享机制更为复杂。

---

## 363. VK-LSVD: A Large-Scale Industrial Dataset for Short-Video Recommendation

**arXiv ID:** 2602.04567 | [PDF](https://arxiv.org/pdf/2602.04567v1)

**作者:** Aleksandr Poslavsky `[一作]` (VK), Andrey Zimovnov `[通讯]` (VK)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并公开了 VK-LSVD——一套规模达 40 亿次交互、覆盖 10 万名用户、20 万条视频的工业级短视频推荐数据集，并对其进行了技术验证与基准评估。

**💡 创新点**

创新点在于：①首次公开如此大规模、长时序、具备多模态内容嵌入与多维反馈（观看时长、点赞、转发等）的短视频数据集；②提供完整的上下文元数据和严格的全局时间切分；③将该数据集作为 VK RecSys Challenge 2025 的核心素材，推动冷启动与长尾研究。

**🔧 技术方法**

所使用技术包括：Parquet 文件存储与按时间切分、SVD 压缩的 64 维视频嵌入、ALS 与 iALS 进行隐式反馈学习、随机与全局流行度基线模型、CTR 转化模型，以及 Challenge 中的排名评估（NDCG@100）。

**📊 数据集**

所使用的数据集为 VK-LSVD（40 亿交互、6 个月、10 万用户、20 万视频），同时提供 1% 随机用户/热门视频子集用于快速实验。

**📈 对比分析**

通过 Global Temporal Split 与 Random Split 对 Random、Global Popular、Conversion、iALS 四种基线进行比较：iALS 在 NDCG@20 上略优，Conversion 在 ROC‑AUC 上最高；在 Challenge 中，参赛者需预测新视频的 top‑100 用户，评估指标为 NDCG@100，预期可显著提升冷启动性能。

**⚠️ 局限性**

局限性包括：①数据仅为匿名化特征，存在潜在算法偏见；②来源单一平台，跨平台推广受限；③缺乏原始多模态内容，无法直接训练端到端视觉/语音模型。

---

## 364. Finding Structure in Continual Learning

**arXiv ID:** 2602.04555 | [PDF](https://arxiv.org/pdf/2602.04555v1)

**作者:** Pourya Shamsolmoali `[一作]` (University of York), Masoumeh Zareapoor `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2692 | [OpenAlex ID](https://openalex.org/A5009550171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于Douglas‑Rachford分裂（DRS）的连续学习框架，将稳定性与可塑性通过两步近端操作解耦，形成协同学习过程；

**💡 创新点**

核心创新在于将稳定性与可塑性目标从优化角度分离，利用DRS求解两个独立的近端子问题，并采用Rényi散度代替KL散度以避免零强制；

**🔧 技术方法**

采用概率编码器‑解码器模型、Gaussian先验与后验、Rényi散度正则、DRS优化策略以及Adam梯度更新；

**📊 数据集**

在EMNIST、CIFAR‑10/100、Tiny‑ImageNet、ImageNet、CelebA等离散与联合任务基准上进行评测；

**📈 对比分析**

与EWC、A‑GEM、UCL、SB‑MCL等19种主流连续学习方法在平均准确率、向后/向前迁移及遗忘率上进行对比，实验表明其平均准确率最高，遗忘率最低，向前迁移最强；

**⚠️ 局限性**

局限在于对高维任务的扩展仍需进一步验证，Rényi散度参数选择对性能影响较大，并且在极长任务序列下仍可能出现轻微的可塑性下降。

---

## 365. Gradient Flow Through Diagram Expansions: Learning Regimes and Explicit Solutions

**arXiv ID:** 2602.04548 | [PDF](https://arxiv.org/pdf/2602.04548v1)

**作者:** Dmitry Yarotsky `[一作]` (Applied AI Institute), Yaroslav Gusev `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于Feynman图的幂级数展开框架，用以分析大规模CP分解模型在梯度流（GF）下的学习过程，并通过此框架推导出多个极端学习阶段的解析解。

**💡 创新点**

创新点在于将图形算子与PDE方法结合，利用Pareto极值多边形系统地刻画不同规模参数（p、H、σ）下的学习范式（lazy、rich、NTK等），并首次在非线性CP分解中给出了NTK、自由演化以及梯度上升收敛/发散的显式解析条件。

**🔧 技术方法**

核心技术包括：Wick公式的期望计算、图形合并（⋆）操作、生成函数的PDE转化、特征线方法求解一阶PDE，以及Borel求和等技巧；同时使用符号计算推导大规模极限下的最优多项式项。

**📊 数据集**

实验数据以人工构造的“identity tensor”目标为基准，随机初始化权重（高斯分布），在不同规模（p=32、128、512等）和参数H、σ的组合上运行梯度流，并记录损失随时间的演化曲线。

**📈 对比分析**

通过将理论得到的损失时间曲线与数值模拟的轨迹进行对比，验证了理论的准确性；在自由演化、NTK与富学习等极端条件下，理论预测与实验曲线高度吻合，展示了所提出方法在解释学习动态方面的优越性能。

**⚠️ 局限性**

局限性在于该框架的幂级数展开与求和过程尚无严格的收敛性证明，适用范围目前主要集中在可用图形描述的目标与模型（如身份张量、对称/非对称CP分解）；对更一般的网络结构或目标函数，需进一步推广和验证。

---

## 366. OmniRad: A Radiological Foundation Model for Multi-Task Medical Image Analysis

**arXiv ID:** 2602.04547 | [PDF](https://arxiv.org/pdf/2602.04547v1)

**作者:** Luca Zedda `[一作]` (University of Cagliari), Cecilia Di Ruberto `[通讯]` (University of Cagliari)

**通讯引用:** 7967 | [OpenAlex ID](https://openalex.org/A5082731857)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

构建了一个基于自监督学习的放射学基础模型 OmniRad，使用 ViT 结构在 120 万张放射影像上预训练，随后在多种分类、分割和图像描述任务中复用该模型。

**💡 创新点**

创新点在于：①采用了放射学启发的自监督预训练策略（仅使用全局裁剪），②设计了轻量级密集适配分支，实现了 Transformer 编码器在分割任务中的高效适配；③实现了单一模型可跨任务复用，提升了特征稳定性和可迁移性。

**🔧 技术方法**

技术包括：自监督学习框架 DINOv2 的改进、ViT-S/ViT-B 编码器、轻量级卷积分支与 upsampling 解码器、LoRA 细调、BART 解码器用于图像描述。

**📊 数据集**

使用的数据集有：RadImageNet（预训练），MedMNISTv2（分类任务如 PneumoniaMNIST、BreastMNIST、OrganAMNIST 等），MedSegBench（分割任务跨 CT、MRI、US 等），ROCOv2（图像-文本对）。

**📈 对比分析**

与现有医学基础模型（DINO、DINOv3、Radio DINO、CLIP、MAE 等）对比，OmniRad 在分类 F1 最高可提升 2.05%，在分割任务中实现最高平均 mIoU 87.93% 及 Dice 92.95%；在图像描述任务中也获得最高 METEOR、BLEU、ROUGE 指标。

**⚠️ 局限性**

局限性包括：预训练数据覆盖不完整，可能缺少罕见模态或机构特异性；实验仅在公开基准上进行，未做临床前瞻验证；采用冻结编码器可能限制特定任务的细粒度性能；未探索更大规模模型；图像描述仅用自动指标评估，未评估临床语义准确性和安全性。

---

## 367. Forget to Generalize: Iterative Adaptation for Generalization in Federated Learning

**arXiv ID:** 2602.04536 | [PDF](https://arxiv.org/pdf/2602.04536v1)

**作者:** Abdulrahman Alotaibi `[一作]` (Massachusetts Institute of Technology), Lalana Kagal `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 5651 | [OpenAlex ID](https://openalex.org/A5013709154)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出迭代联邦适应（Iterative Federated Adaptation）框架，周期性地在联邦学习中重置模型部分参数以提升对非IID分布的泛化性能。

**💡 创新点**

创新点在于将持续学习中的“忘记-进化”机制引入联邦学习，提供两种参数重置策略（随机重置与后层重置），显著缓解客户端特异化导致的表示漂移。

**🔧 技术方法**

结合 FedAvg 等聚合方法与 IFA 重置机制；使用随机采样与后层选择两种参数重置策略；实验中采用多代（generation）训练，期间每代结束时执行重置。

**📊 数据集**

使用 CIFAR‑10、MIT Indoors 与 Stanford Dogs 三个图像分类数据集，模拟 IID 与非 IID（标签分布偏差）场景。

**📈 对比分析**

通过与传统 FedAvg（以及 FedProx、FedNova 等）在相同通信轮数和数据分布下的对比实验，发现 IFA 在非 IID 场景平均提升 21.5%，在 IID 场景亦保持显著收益；后层重置往往在后代中表现更好。

**⚠️ 局限性**

局限性包括：重置比例、代数与层选择需人工设定，缺乏自适应调度；理论分析不充分；在极大规模真实系统中的部署与效能尚待进一步验证。

---

## 368. Can We Redesign a Shoulder Exosuit to Enhance Comfort and Usability Without Losing Assistance?

**arXiv ID:** 2602.04625 | [PDF](https://arxiv.org/pdf/2602.04625v1)

**作者:** Roberto Ferroni `[一作]` (Scuola Superiore Sant'Anna), Tommaso Proietti `[通讯]` (Università Vita-Salute San Raffaele)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在先前的柔性肩部外骨装置（Soft Shoulder v1）基础上进行改进，设计出新的 Soft Shoulder v2，主要改进方向为提升佩戴舒适度、增加肩关节在前平面（肩屈伸）辅助力度，并保持或提升助力性能。

**💡 创新点**

创新点包括：①将辅助力从冠状面（肩外展）转向矢状面（肩屈伸），更符合日常 ADL 手部定位需求；②重新定义支撑与张力结构（使用 3D 打印臂袖、腰带式束缚、热塑性板分布压力），显著降低皮肤压力、改善机械透明度；③采用压力分布映射与 QUEST 量表的用户体验评估，形成用户中心的评估框架；④在同一组受试者上对比两版的多项任务（静态保持、动态提举、拾放），系统评估功能与舒适度。

**🔧 技术方法**

核心技术包括：柔性气动膨胀阀片 actuator、针织织物 harness、3D 打印臂袖、压差闭环气压控制（bang‑bang + hysteresis）、三轴加速度计/陀螺仪 IMU、表面肌电 sEMG、压力感知映射、PyQt GUI、QUEST 问卷。

**📊 数据集**

数据集：8 名健康受试者（18–65 岁，无上肢伤害），完成静态保持、动态提升、拾放、机械透明度测试等任务，共计 8×若干次试验；收集 EMG、IMU、压力映射、QUEST 评分等多模态数据。

**📈 对比分析**

比较方法：在同一受试者内随机交叉使用 v1 与 v2，使用非参数重复测量检验（Friedman、Wilcoxon）和 FDR 校正。性能表现：两版在助力目标方向均能显著提升耐力时间、降低 deltoid 活动；v2 在前平面辅助更有效、横向 ROM 更大、腰部扭转更小；舒适度评分提高约 20–25%；压力分布面积减小、最高压强下降；整体助力保持或提升，且无显著差异。

**⚠️ 局限性**

局限性：样本量仅 8 人；仅测量 deltoid 运动电流，未覆盖其他上肢肌群；未进行长期穿戴或临床人群评估；机械性能未单独量化（如力学模量、耐久性）；压力映射采用简化线性加权模型；未验证主动控制策略。

---

## 369. QUATRO: Query-Adaptive Trust Region Policy Optimization for LLM Fine-tuning

**arXiv ID:** 2602.04620 | [PDF](https://arxiv.org/pdf/2602.04620v1)

**作者:** Doyeon Lee `[一作]` (Seoul National University), Jaemoo Choi `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于查询自适应信赖域的强化学习微调框架（QA-TRPO），直接在每个查询上施加 KL 约束，实现对 LLM 更新幅度的精确控制。

**💡 创新点**

创新点在于：①以查询为条件的双重拉格朗日对偶求解，得到查询特定的 λ 和 μ，精准实现 KL 约束；②引入 log‑ratio 稳定项，取代经验剪切，避免梯度遮蔽；③通过自适应 λ 使得不同难度查询获得不同更新幅度，平衡探索与利用。

**🔧 技术方法**

采用信赖域策略优化（TRPO）、GRPO/GSPO 对比实验、拉格朗日对偶分析、KL 正则化、熵控制以及基于奖励模型的回报估计。

**📊 数据集**

使用数学推理基准数据集 MATH、AMC23、AIME24/25、MinervaMath、OlympiadBench，并在 Qwen2.5-Math-1.5B/7B 两个规模模型上进行微调。

**📈 对比分析**

与 GRPO、GPG、GSPO、GMPO 等现有基线对比，QA‑TRPO 在 Pass@k（尤其 k 较大时）持续领先；同时在 UCC@k 上获得更高的答案多样性；训练过程中的熵保持稳定，避免模式崩塌。

**⚠️ 局限性**

局限性包括：①需要手动设置信赖域半径 δ，调参仍对性能影响较大；②实验仅验证在数学推理任务，尚未验证在其他推理或对话场景的通用性；③对极大模型或极高分辨率任务的可扩展性未做深入探讨。

---

## 370. Jacobian Regularization Stabilizes Long-Term Integration of Neural Differential Equations

**arXiv ID:** 2602.04608 | [PDF](https://arxiv.org/pdf/2602.04608v1)

**作者:** Maya Janvier `[一作]` (LOCEAN), Etienne Meunier `[通讯]` (LOCEAN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过对神经ODE模型的雅可比矩阵施加正则化，利用方向导数来稳定长时序的数值积分。

**💡 创新点**

首次将雅可比正则化与Hutchinson估计相结合，提出了两种正则化方案：已知动力学的精确方向导数正则化和未知动力学的有限差分方向导数正则化。

**🔧 技术方法**

采用神经ODE、前向模式自动微分、Hutchinson迹估计、方向导数/有限差分正则化以及梯度下降训练。

**📊 数据集**

使用三组合成数据集：二维两体问题、刚体旋转动力学以及一维Kuramoto‑Sivashinsky偏微分方程，均从已知解析解生成轨迹。

**📈 对比分析**

与仅使用轨迹损失的短回放训练和长回放训练进行对比，结果显示正则化模型在长时间步长上能保持低相对误差，部分方法甚至优于长回放基线。

**⚠️ 局限性**

对已知动力学的正则化依赖真函数，需额外计算；有限差分正则化在复杂系统中易过于约束且调参困难；实验仅在低维合成系统上验证，缺乏对高维真实物理数据的评估。

---

## 371. Beyond Holistic Scores: Automatic Trait-Based Quality Scoring of Argumentative Essays

**arXiv ID:** 2602.04604 | [PDF](https://arxiv.org/pdf/2602.04604v1)

**作者:** Lucile Favero `[一作]` (ELLIS Alicante), Nuria Oliver `[通讯]` (ELLIS Alicante)

**通讯引用:** 14898 | [OpenAlex ID](https://openalex.org/A5013727792)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了针对论证性作文的特征级自动评分方法，比较小型开源LLM与基于BigBird的序数回归模型，提供可解释、与评分 rubrics 对齐的反馈

**💡 创新点**

创新点在于：① 引入 CORAL 序数回归以显式编码评分等级；② 在不做微调的前提下使用结构化提示让小型LLM产生解释与置信度；③ 将两种方法在同一数据集和评估指标上系统比较，验证序数建模提升人类评分一致性

**🔧 技术方法**

技术包括：1）小型开源LLM（Llama‑3.1 8B、Gemma‑3 12B、Ministral‑3 8B）配合结构化提示与JSON输出；2）BigBird encoder 结合 CORAL 形成的序数阈值模型；3）传统基准（BigBird‑nominal、BigBird‑RMSE）；4）评估指标 QWK、加权 F1、Kendall’s τ；5）数据划分与多种随机种子重复训练

**📊 数据集**

使用 ASAP++ 论证性子集，1783 篇 8 年级学生作文，基于五项特征（Ideas & Content、Organization、Word Choice、Sentence Fluency、Conventions）和六分制评分（映射为低/中/高）

**📈 对比分析**

结果显示：BigBird‑CORAL 在 QWK 上领先，显著优于所有LLM与传统 BigBird 变体；Ministral‑3 8B 在多数特征上与 GPT‑4o‑mini、GPT‑5.1 接近甚至超越，尤其在 Content 与 Organization；LLM 在低/中/高区间的解释和置信度提升了整体表现

**⚠️ 局限性**

局限性包括：仅评估单一论证题目，未探究跨题目泛化；聚焦评分准确性，缺少对学习成效的后续研究；LLM 对细粒度语言特征的分辨率仍有限；未验证模型在真实课堂中的应用与教师/学生反馈

---

## 372. Dual Mind World Model Inspired Network Digital Twin for Access Scheduling

**arXiv ID:** 2602.04566 | [PDF](https://arxiv.org/pdf/2602.04566v1)

**作者:** Hrishikesh Dutta `[一作]` (Telecom SudParis), Noel Crespi `[通讯]` (Telecom SudParis)

**通讯引用:** 8304 | [OpenAlex ID](https://openalex.org/A5107205316)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了基于Dual Mind World Model的网络数字孪生调度框架，用于工业物联网和实时网络系统的接入调度。

**💡 创新点**

创新点在于融合快思维的启发式调度与慢思维的符号规划，并通过内部数字孪生进行有限期前向滚动预测，从而在保证可解释性的同时实现对干扰、死线和突发流量的自适应调度。

**🔧 技术方法**

采用数字孪生仿真、Dual Mind架构（Fast Mind+Slow Mind）、受限前瞻滚动模型、ICN约束校验、Poisson到达建模、以及与Q‑learning等基线算法的对比实验。

**📊 数据集**

使用自建的仿真数据集：5个节点、时间滑动的Poisson到达（含突发性）、冲突图、队列容量为50、不同死线与干扰场景，共四种测试环境。

**📈 对比分析**

通过与随机、LQF、Deadline‑Priority、Round‑Robin及Q‑learning基线在四种场景下的吞吐量、队列长度、延迟和死线违例率等指标对比，DMWM在吞吐量最高或接近最高、队列占用最低、延迟与死线违例显著下降，且波动更小，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：规划期仅为3步，导致长远预测能力有限；符号滚动计算开销随节点数增长；在大规模网络中的可扩展性尚未验证；依赖准确的数字孪生模型，若模型误差增大会影响调度质量。

---

## 373. LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding

**arXiv ID:** 2602.04541 | [PDF](https://arxiv.org/pdf/2602.04541v1)

**作者:** Gang Lin `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 59784 | [OpenAlex ID](https://openalex.org/A5100402851)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对长文本推理中的键值缓存膨胀问题，本文提出了一种基于注意力头细粒度分工的混合头稀疏解码框架：少数检索头全量计算关键标记，绝大多数稀疏头共享这些标记以降低计算和内存成本。

**💡 创新点**

创新点包括：① 细粒度头级别的检索/稀疏分工，突破层级共享导致的功能单一化；② 使用 HardKuma 分布实现端到端可微的二值化头选择，显著减少训练‑推理差距；③ 在 TileLang 上实现高效块稀疏解码核，兼容硬件加速并实现显著加速。

**🔧 技术方法**

核心技术：HardKuma 分布、混合头稀疏注意力机制、块稀疏解码核（TileLang）、缓存纠正（Cache Correction）策略、基于知识蒸馏的训练目标和 L0 约束优化。

**📊 数据集**

实验使用的主要数据集包括：LongBench、RULER、AIME24、OlympiadBench、Passkey Retrieval、HotpotQA 等长文本理解与推理基准；模型主要为 Llama3-8B、Qwen3-8B 等大语言模型。

**📈 对比分析**

与 TidalDecode、Quest、DuoAttention、SeerAttention-R、FlashAttention‑2 等现有稀疏注意力方法以及全注意力基线对比，实验表明：在 LongBench 上平均得分可超越全注意力模型；在 AIME24、OlympiadBench 等推理任务中表现优于 TidalDecode；在 128K 上下文长度下，整体解码速度提升达 2.7×，块稀疏核在 128K、批量 8 时可达 7× 的加速。

**⚠️ 局限性**

局限性：对极端稀疏设置性能下降，训练需要额外的 HardKuma 参数和 Lagrangian 约束；在短标记或低梯度信号任务（如 HotpotQA）上，HardKuma 的效果略逊；目前验证仅在少数模型和数据集上，需进一步验证在更大规模模型与多样化任务中的稳健性。

---

## 374. HoliAntiSpoof: Audio LLM for Holistic Speech Anti-Spoofing

**arXiv ID:** 2602.04535 | [PDF](https://arxiv.org/pdf/2602.04535v1)

**作者:** Xuenan Xu `[一作]` (Shanghai Artificial Intelligence Laboratory), Chao Zhang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1900 | [OpenAlex ID](https://openalex.org/A5115596350)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研发了Holistic Anti‑Spoofing框架HoliAntiSpoof，利用音频大语言模型（ALLM）对语音进行整体伪造分析；

**💡 创新点**

创新点在于把真伪分类、伪造方式识别、伪造区域定位和语义影响分析统一为结构化文本生成任务，并提出DailyTalkEdit数据集与语义标注；

**🔧 技术方法**

采用ALLM（Qwen2.5‑Omni + 语音编码器 + DoRA微调 + ICL fine‑tuning），并结合原始音频编码器和专门的伪造特征编码器；

**📊 数据集**

使用了现有英语伪造数据集（ASVSpoof2019、SpoofCeleb等）以及自建的带语义注释的PartialEdit和对话级的DailyTalkEdit；

**📈 对比分析**

与传统基线（RawNet2、AASIST等）和商业ALLM Gemini‑3‑Flash对比，HoliAntiSpoof在真伪分类、伪造方式识别、区域定位和语义分析上均显著优于基线，ICL提升了跨域泛化性能；

**⚠️ 局限性**

局限在于对未知伪造方式和语言的泛化仍有限；语义影响评估依赖LLM判定，需进一步验证；对低质量或长时音频的鲁棒性待提升。

---

## 375. On the Complexity of Vertex-Splitting Into an Interval Graph

**arXiv ID:** 2602.04628 | [PDF](https://arxiv.org/pdf/2602.04628v1)

**作者:** Faisal N. Abu-Khzam `[一作]` (Lebanese American University), Nacim Oijid `[通讯]` (Umeå University)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5081421238)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究一种新型图修改操作——顶点分裂，并将其用于将任意图转化为区间图。

**💡 创新点**

证明在计划三角形、平面二分子三度图等极其受限的输入下，该问题仍为NP‑hard；并揭示顶点分裂与顶点/边删操作在得到弦图时的复杂度截然不同；此外给出了在三角形无图时将图拆分为路径集合的多项式时间算法。

**🔧 技术方法**

主要采用多面向的图论工具：构造归约（从三角形无图的 Hamiltonian 路问题到顶点分裂问题）、结构定理（关于弦图、单连通图、路径分解的性质）、轨道分解与分裂序列的构造，以及利用闭合性与树宽、模宽等参数的MSO可表述。

**📊 数据集**

本研究为纯理论性工作，没有使用实际数据集；所有结果均通过理论证明得出。

**📈 对比分析**

比较对象为经典的顶点删/边删变换。论文指出顶点分裂在弦图类中至少与顶点删同级、至少不超过边删，并给出在独立数≤2时的具体区分示例；但未给出实验性能指标，只给出多项式/指数复杂度分析。

**⚠️ 局限性**

限制主要包括：未解决弦图或单位区间图的顶点分裂的完整复杂度；未给出参数化（如树宽、Twin‑width 等）下的固定参数可解性或近似算法；在实际应用中缺乏经验评估与实现细节。

---

## 376. Stochastic Decision Horizons for Constrained Reinforcement Learning

**arXiv ID:** 2602.04599 | [PDF](https://arxiv.org/pdf/2602.04599v1)

**作者:** Nikola Milosevic `[一作]` (Max Planck Institute for Human Cognitive and Brain Sciences), Pavel Kolev `[通讯]` (University of Tübingen)

**通讯引用:** 1771 | [OpenAlex ID](https://openalex.org/A5001474340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于随机决策地平线（SDH）的控制为推理框架，用以在强化学习中处理约束，取代传统的加法成本和拉格朗日多重变量。

**💡 创新点**

创新点在于：①将约束违背视为状态动作依赖的生存概率，自动缩短有效规划长度；②给出两种终止语义（吸收与虚拟终止）并证明它们共享相同的生存加权回报；③将这些理论推导为可直接兼容离线回放的SAC/MPO式算法，解决了传统CMDP方法在离线/高维环境下的可扩展性问题。

**🔧 技术方法**

技术主要包括：控制为推理（CaI）框架、状态动作依赖折扣与奖励塑形、两种生存概率映射（指数衰减与CaT归一化）、SAC/VT-MPO离线演员-批评家实现以及温度自适应与多目标KL约束。

**📊 数据集**

实验数据集包括：Safety Gymnasium（8个连续控制任务）和高保真肌肉骨骼模拟器Hyfydy（3个行走任务），用于评估回报-约束权衡与样本效率。

**📈 对比分析**

与基线（CPO、C-TRPO、无约束MPO、EWA）比较，SDH方法（尤其是VT-MPO）在大多数任务中实现了更优的奖励-违约平衡、更快的学习速度和更高的样本效率；在Hyfydy任务中显著低于EWA的能量消耗，同时满足速度约束。

**⚠️ 局限性**

局限性包括：需要手动设定/调度生存概率缩放因子，且对不同任务的适应性有限；在某些Safety Gymnasium环境下仍未完全实现可行性；理论上未完全保证CMDP约束下的严格可行性。

---

## 377. LEAD: Layer-wise Expert-aligned Decoding for Faithful Radiology Report Generation

**arXiv ID:** 2602.04617 | [PDF](https://arxiv.org/pdf/2602.04617v1)

**作者:** Ruixiao Yang `[一作]` (Beijing Institute of Technology), Yan Song `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27923 | [OpenAlex ID](https://openalex.org/A5005228053)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种层级专家对齐解码（LEAD）框架，直接在大型视觉语言模型（VLM）的每一层解码过程中注入多标签疾病专家特征，以纠正语言模型的内部偏差并降低幻觉。

**💡 创新点**

创新点包括：①将多标签疾病分类器视为专家并按层级投射特征；②使用上下文感知门控融合机制，使解码层可动态选择是否接受专家信号；③通过自适应投射而非统一投射，匹配不同深度层的语义抽象；④在保持语言流畅度的同时显著提升诊断准确性。

**🔧 技术方法**

技术手段：Qwen3‑VL backbone、ViT视觉编码器、MLP多层专家分类器、门控融合块、低秩适配（LoRA）微调、交叉熵与多标签二分类损失的联合训练。

**📊 数据集**

主要使用CheXpert Plus（约22万张胸片）和MIMIC‑CXR（约38万张胸片）两大公开数据集进行训练与评估。

**📈 对比分析**

与Transformer、R2Gen、CvT2DistilGPT2、Llama2/3、MambaXray‑VL等多种基线相比，LEAD在CheXpert Plus上实现了最高的临床F1（0.275，优于MambaXray‑VL的0.273），并在MIMIC‑CXR上保持稳定提升；在自然语言生成指标上略低，但说明了更高的事实一致性。

**⚠️ 局限性**

局限性：①对专家分类器的准确性高度依赖，若标签噪声或缺失会影响注入信号；②门控机制与投射参数需要在不同模型规模和数据分布下重新调参；③在极低资源或小样本场景下，层级投射和门控的学习可能不足，导致性能提升有限。

---

## 378. Disentangling meaning from language in LLM-based machine translation

**arXiv ID:** 2602.04613 | [PDF](https://arxiv.org/pdf/2602.04613v1)

**作者:** Théo Lasnier `[一作]` (Inria), Benoît Sagot `[通讯]` (Inria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过激活补丁与对比提示的方法，分析大语言模型中句子级机器翻译（MT）的内部机制，发现不同子任务（目标语言识别与句子等价性）由稀疏、专门化的注意力头实现；随后构造子任务特定的向量，用于在无指令零样本情境下实现高质量翻译，并验证等价向量可跨语言对迁移。

**💡 创新点**

创新点：①将MT拆解为目标语言识别和句子等价性两个子任务，并证明每个子任务由几乎互斥的注意力头集合实现；②提出基于激活补丁的稀疏头识别与KL驱动的目标位置选择；③构造子任务向量，实现仅约1%注意力头的指引即可获得与指令式零样本相当的翻译质量；④证明等价向量在不同语言对之间具有高度可迁移性。

**🔧 技术方法**

核心技术：激活补丁（activation patching）、对比提示（clean vs corrupted prompts）、KL 散度定位最具信息量的标记位置、注意力头的重要性评估、任务向量（语言向量与等价向量）构造与调度、消融实验以及对不同模型层、头的可视化分析。

**📊 数据集**

使用 FLORES‑200 多语平行语料集，包含 200+ 语言，实验选取 20 条语言对（如 EN↔AR, EN↔ZH, EN↔FR, EN↔ES 等），利用 dev 生成提示和 head 识别，devtest 评估 MT 性能。

**📈 对比分析**

与传统指令式零样本 MT、随机头消融、以及不同比例头引导等方法对比；结果显示：①仅 1% 语言向量+1% 等价向量即可在无指令提示下达到与指令式相近的 BLEU/MetricX；②消融 1% 对应子任务头会显著降低翻译质量，说明其因果作用；③等价向量在跨方向迁移时性能下降不到 5%，低资源语言略高。

**⚠️ 局限性**

局限性：①对低资源或极少数语言（如 Wolof）的等价向量效果不佳；②需要先构造清洁与腐败提示，依赖足够的多语样本；③模型规模影响头的重要性与定位的鲁棒性，较小模型对位置选择更敏感；④仅聚焦注意力头，未探讨神经元、前馈层等其他机制。

---

## 379. SalFormer360: a transformer-based saliency estimation model for 360-degree videos

**arXiv ID:** 2602.04584 | [PDF](https://arxiv.org/pdf/2602.04584v1)

**作者:** Mahmoud Z. A. Wahba `[一作]` (University of Padova), Federica Battisti `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为SalFormer360的基于transformer的360度视频显著性估计模型；

**💡 创新点**

创新点在于将SegFormer分割网络迁移到显著性任务，并设计自定义解码器与可学习的视角中心偏差（CB）机制；

**🔧 技术方法**

采用Transformer编码器（SegFormer‑B0）、自定义卷积解码器、加权中心偏差、Pearson相关、KL、Spherical MSE、BCE等复合损失进行训练；

**📊 数据集**

使用三个公开的360度显著性数据集：Sport360、PVS‑HM和VR‑EyeTracking；

**📈 对比分析**

与多种基准方法（2D/360图像/视频显著性模型）比较，取得在CC指标上分别提升8.4%、2.5%和18.6%，整体性能优于现有最优方法；

**⚠️ 局限性**

局限性包括对多目标/高动态场景的显著性预测仍存在误差，且中心偏差的学习依赖于数据集特征，未来需进一步改进动态/多模态特征融合与更大范围的数据集验证。

---

## 380. Trust The Typical

**arXiv ID:** 2602.04581 | [PDF](https://arxiv.org/pdf/2602.04581v1)

**作者:** Debargha Ganguly `[一作]` (Case Western Reserve University), Vipin Chaudhary `[通讯]` (Case Western Reserve University)

**通讯引用:** 4016 | [OpenAlex ID](https://openalex.org/A5004523290)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

**🎯 论文内容**

暂无信息

**💡 创新点**

暂无信息

**🔧 技术方法**

暂无信息

**📊 数据集**

暂无信息

**📈 对比分析**

暂无信息

**⚠️ 局限性**

暂无信息

---

## 381. Semantic Self-Distillation for Language Model Uncertainty

**arXiv ID:** 2602.04577 | [PDF](https://arxiv.org/pdf/2602.04577v1)

**作者:** Edward Phillips `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**通讯引用:** 13558 | [OpenAlex ID](https://openalex.org/A5040302008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Semantic Self-Distillation (SSD) 框架，利用小型学生模型在单向前向推理中预测大语言模型生成答案的语义分布，从而在不需要多次采样的情况下提供预生成的误差风险评估、后生成的可信度验证和语义一致性估计。

**💡 创新点**

创新点在于将分散的语义采样结果压缩为可解析的混合高斯密度模型，既保留了完整分布信息，又避免了传统基于采样或单一标量探针的高延迟与信息损失；同时引入 Rényi‑2 熵作为闭式可计算的分散度量，进一步提升了实用性。

**🔧 技术方法**

主要技术包括：使用大语言模型提取提示隐藏层表示、利用专门嵌入模型生成答案语义向量、对嵌入做 PCA 降维、构建学生混合密度网络（MDN）进行条件分布建模、用最大似然训练对教师分布进行蒸馏、以及基于熵与后验似然的多种不确定性信号推断。

**📊 数据集**

实验采用 TriviaQA 数据集进行短文本问答评估，训练集 4000 题、测试集 1000 题；涉及七大模型家族（Mistral、Llama‑3、Qwen、Gemma、SmolLM 等），并使用 EmbeddingGemma 进行答案嵌入。

**📈 对比分析**

与传统采样基的语义散度（Teacher Dispersion）和三种单向探针（PCP、SE、SEP）相比，SSD 在 AUROC/ AUPRC 上表现接近或优于 Teacher Dispersion，且显著低于需要多次采样与 NLI 的 SE；SSD 具有与探针相同的低延迟、无后验成本，并额外提供后验似然与语义一致性判定，整体性能既稳健又高效。

**⚠️ 局限性**

主要局限在于蒸馏逼近度（distillation fidelity）对误差检测效果的依赖；若学生无法从提示表示中充分捕捉不确定性信号，则预测熵与教师散度的相关性降低，导致误差识别性能下降；此外，当前实现仅针对自回归模型，扩展到扩散模型或特定领域时仍需进一步研究。

---

## 382. Vibe AIGC: A New Paradigm for Content Generation via Agentic Orchestration

**arXiv ID:** 2602.04575 | [PDF](https://arxiv.org/pdf/2602.04575v1)

**作者:** Jiaheng Liu `[一作]` (Nanjing University), Xinping Lei `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Vibe AIGC框架，将创作视为多代理工作流的自主合成。

**💡 创新点**

突破模型中心范式，强调高层意图驱动的逻辑编排和Meta Planner。

**🔧 技术方法**

结合多智能体、Meta Planner、知识库和现有生成模型（如Diffusion、Transformer）进行层次化编排。

**📊 数据集**

未公开专用数据集，主要依赖现有生成模型训练数据与实验构造的代理任务集。

**📈 对比分析**

与传统prompt-engineering对比，Vibe AIGC在多模态任务中降低了迭代次数、提升了风格一致性，整体性能优于单模型推断。

**⚠️ 局限性**

缺乏客观评测标准、易产生误判且无“美学编译器”，导致可验证性不足与专业精细控制受限。

---

## 383. Understanding Degradation with Vision Language Model

**arXiv ID:** 2602.04565 | [PDF](https://arxiv.org/pdf/2602.04565v1)

**作者:** Guanzhou Lan `[一作]` (Northwestern Polytechnical University), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 61676 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Vision‑Language模型的层次化参数化退化理解框架DU‑VLM，并将其用作零样本控制扩散模型实现图像恢复。

**💡 创新点**

创新点在于：①把退化理解任务建模为一个自回归的结构化预测序列，统一分类、键选择与连续值回归；②采用多模态Chain‑of‑Thought推理，将频域、边缘等辅助视图融入模型；③通过结构化强化学习（RL）分解奖励，提升物理一致性；④构建110k条带有物理参数的清晰‑退化配对数据集DU‑110k。

**🔧 技术方法**

核心技术包括：多模态Vision‑Language预训练模型（Qwen3‑VL）、自回归下游生成、FFT与Sobel边缘输入、Chain‑of‑Thought推理、离线结构化RL（GRPO）与在线自监督RL、基于扩散模型的零样本恢复。

**📊 数据集**

使用了DU‑110k数据集（110k张清晰‑退化图像对，涵盖雾、暗光、模糊、低分辨率四类退化，每类27.5k样本），并在公开的CleanBench、Night、Haze、Blur、Low‑Resolution等基准上进行评测。

**📈 对比分析**

与通用VLM、专用退化识别模型（DA‑CLIP、Q‑Instruct、DepictQA）以及通用恢复模型（TAO、JarvisIR、Restormer、DiffBIR、DFPIR）对比，DU‑VLM在退化参数估计（P‑Abs/P‑Rel）和联合准确率上取得显著提升；在零样本恢复任务中，PSNR/SSIM/LPIPS均优于对照组，尤其在模糊和低分辨率下提升幅度超过8 dB。

**⚠️ 局限性**

局限性包括：①仅覆盖四类单一退化，无法直接处理混合或更复杂的退化；②仍依赖离线数据生成，仿真与真实场景之间存在一定差距；③在推理时需较长的序列生成与多模态输入，计算成本较高；④扩散模型对参数误差敏感，需进一步提升鲁棒性。

---

## 384. Rethinking Weight Tying: Pseudo-Inverse Tying for Stable LM Training and Updates

**arXiv ID:** 2602.04556 | [PDF](https://arxiv.org/pdf/2602.04556v1)

**作者:** Jian Gu `[一作]` (Monash University), Hongyu Zhang `[通讯]` (Chongqing University)

**通讯引用:** 14721 | [OpenAlex ID](https://openalex.org/A5100412608)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的权重绑定机制，称为伪逆绑定（PIT），以确保语言模型在训练过程中的稳定性和一致性。

**💡 创新点**

创新点在于通过共享潜在令牌内存和可学习的可逆变换，强制输入嵌入和输出投影之间保持一致的接口，从而解决了标准权重绑定中存在的令牌接口漂移问题。

**🔧 技术方法**

使用了薄极分解和Cholesky因子化来实现可学习的对称正定变换，确保了数值稳定性和高效性。

**📊 数据集**

在多个模型规模（256M到1.3B参数）上进行了评估，使用了FineWeb-Edu数据集进行预训练和适应性训练。

**📈 对比分析**

与标准的转置绑定方法（TT）进行比较，PIT在训练稳定性、层次语义一致性和轻量级更新的副作用减少方面表现出一致的改进。

**⚠️ 局限性**

限制在于PIT的额外计算开销相对较小，但在某些情况下，可能会限制模型的灵活性，特别是在从头开始训练时。

---

## 385. Unmasking Superspreaders: Data-Driven Approaches for Identifying and Comparing Key Influencers of Conspiracy Theories on X.com

**arXiv ID:** 2602.04546 | [PDF](https://arxiv.org/pdf/2602.04546v1)

**作者:** Florian Kramer `[一作]` (Stanford University), Hayagreeva Rao `[通讯]` (Stanford University)

**通讯引用:** 11393 | [OpenAlex ID](https://openalex.org/A5101546344)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究利用7.6M条COVID‑19相关推文，系统区分人类超级扩散者与机器人，提出27种度量并通过拆解实验评估其对削弱阴谋论传播的效果，同时比较两类传播者在语言、情感、毒性及政治倾向等方面的行为差异。

**💡 创新点**

创新点包括：①首次将H‑Index与G‑Index等学术指标改造为社交媒体传播评估工具，并证明其在低阈值下的高效性；②构建完整的“人类‑机器人”对比框架，揭示二者在语义复杂度、标签使用、情绪与毒性等维度的显著差异；③提出可解释、低成本的识别流程，可直接用于平台治理与公共宣传。

**🔧 技术方法**

使用技术：BLOOMZ‑1.7B 进行阴谋推文检测；Botometer X 识别机器人；VADER、DistilRoBERTa 与 Perspective API 分别用于情感、情绪与毒性评估；统计学方法（ANOVA、卡方检验、Cramér–von Mises）验证差异显著性；拆解分析评估指标的删减效果。

**📊 数据集**

数据集：7,616,569条推文，涵盖7,043名原创用户；经过筛选得到954,993条转推，最终用于分析的261,371条推文；其中13.03%（918名）为机器人，涵盖35,109条阴谋相关推文。

**📈 对比分析**

比较方法：将27个度量按影响力排名，对用户逐一移除并统计剩余阴谋推文数量；与理论最优基准对比。结果显示，H‑Index 在仅暂停0.1%用户时可去除12.35%推文，几乎等同于最优基准；G‑Index 在大规模移除时表现更佳；其他聚合/每推平均/粉丝加权指标效果中等；人类超级扩散者在语言复杂度、低情绪极化和更少表情符号方面与机器人形成对比。

**⚠️ 局限性**

局限性：①数据仅覆盖2020年2–5月的Twitter，未考虑平台与时间演变；②过滤与预处理导致生存偏差，可能低估被删除账号的影响；③情感、毒性模型受语言与文化偏差影响，可能误判细微语义；④仅使用机器人检测器和单一模型，对多样化机器人策略的识别可能不足。

---

## 386. An Efficient Bayesian Framework for Inverse Problems via Optimization and Inversion: Surrogate Modeling, Parameter Inference, and Uncertainty Quantification

**arXiv ID:** 2602.04537 | [PDF](https://arxiv.org/pdf/2602.04537v1)

**作者:** Mihaela Chiappetta `[一作]` (University of Pavia), Ferdinando Auricchio `[通讯]` (University of Pavia)

**通讯引用:** 19364 | [OpenAlex ID](https://openalex.org/A5082965377)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一套统一的贝叶斯框架，将贝叶斯优化（BO）与贝叶斯反演（BI）结合，用高斯过程（GP）构建代理模型，再通过最小二乘形式的MAP估计实现参数反演与不确定性量化，验证了该方法在低维解析基准问题上的有效性。

**💡 创新点**

创新点在于：①将BO与BI融合为单一工作流，实现代理模型的自适应构建与反演的无缝衔接；②使用极大探索参数的UCB采样策略，避免在有观测数据前先构造代理；③采用基于最小二乘的贝叶斯逆推，提升计算效率并可直接得到后验分布；④通过联合方法显著降低高保真模型评估次数，实现更高效、精确的逆问题求解。

**🔧 技术方法**

核心技术包括：高斯过程回归（Matérn 5/2核）、UCB采样的贝叶斯优化、L-BFGS-B最小化的MAP估计、负最小二乘（NLS）后验近似、拉普拉斯近似等。

**📊 数据集**

使用的“数据集”为一维与二维的解析基准函数：一维的Mixed Gaussian-Periodic、Lévy、Griewank、Forrester；二维的Mixed Gaussian-Periodic与Rosenbrock；无真实实验数据，仅用于验证方法。

**📈 对比分析**

与单独使用BO（EI或UCB）、单独使用BI（MAP）以及传统代理+反演等方法相比，联合BO(UCB)+BI(MAP-LS)在同样的评估预算下实现了更低的均方误差、更准确的MAP估计以及完整的后验分布，显示出更高的样本效率与更好的不确定性量化。

**⚠️ 局限性**

局限性包括：仅在低维（≤2D）基准上验证，尚未评估在高维参数空间的可扩展性；对代理模型的准确性高度依赖；对多峰后验仅通过负最小二乘近似，可能低估多模态不确定性；未来需结合多保真、物理信息代理以及更大规模的贝叶斯推断策略。

---

## 387. Algebraic and Arithmetic Attributes of Hypergeometric Functions in SageMath

**arXiv ID:** 2602.04531 | [PDF](https://arxiv.org/pdf/2602.04531v1)

**作者:** Xavier Caruso `[一作]` (CNRS IMB Université de Bordeaux), Florian Fürnsinn `[通讯]` (University of Vienna)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5008049436)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083`

**🎯 论文内容**

开发了一个 SageMath 开源包，用于在有理数、有限域和 p‑域上操作超几何函数，并实现了全局有界性、代数性、p‑缩减等性质的判定及相关高级运算（section、Dwork 关系、p‑曲率、p‑半径、漂移估值等）。

**💡 创新点**

创新点在于：①实现了 Christol、Beukers‑Heckman 等经典判定的通用算法，支持任意参数并允许负底参数；②首次将 section operators 与 Dwork 关系、线性化多项式等方法集成到 SageMath；③引入了 tropical 半环与 Floyd‑Warshall 算法计算漂移 p‑估值；④提供了 p‑模等价性检验与 Newton 多边形可视化。

**🔧 技术方法**

技术手段包括：符号环与多项式环实现；p‑模与 p‑逼近计算；section 及线性化多项式（Dwork）方法；tropical 半环、Floyd‑Warshall 迭代求弱传递闭包；Newton 多边形构造；p‑曲率矩阵与其余维数判定。

**📊 数据集**

主要使用示例函数 f(x)、g(x)、h(x) 进行验证，并通过交互式工作簿展示功能；未使用外部大规模数据集。

**📈 对比分析**

与现有 SageMath/Maple 工具比较，证明了实现的正确性与一致性；在大素数情况下，p‑缩减、p‑曲率等运算只需考虑同余类，显示了高效性；但文中未给出具体时间或内存基准。

**⚠️ 局限性**

局限性包括：Dwork 系统实现采用简化版，尚未证明一定终止；某些算法仅在 m = n‑1 或无整数差分等特定情形下可靠；未覆盖所有特殊参数组合；缺乏对大规模计算性能的系统评估。

---

## 388. Fine-Grained Frame Modeling in Multi-head Self-Attention for Speech Deepfake Detection

**arXiv ID:** 2602.04702 | [PDF](https://arxiv.org/pdf/2602.04702v1)

**作者:** Tuan Dat Phuong `[一作]` (Hanoi University of Science and Technology), Trang Nguyen Thi Thu `[通讯]` (Nanyang Technological University)

**通讯引用:** 699 | [OpenAlex ID](https://openalex.org/A5101741134)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种Fine‑Grained Frame Modeling (FGFM) 方法，用多头投票 (MHV) 与交叉层细化 (CLR) 对Transformer的多头自注意力进行帧级细粒度建模，以提升语音深度伪造检测性能。

**💡 创新点**

创新点在于：①将多头注意力视为弱学习器，通过投票选取最具信息量的帧；②在不同层间聚合选取帧并通过交叉注意力与DAFF模块进一步细化；③在传统Transformer/Conformer结构上实现细粒度关注，显著提高对短暂、局部伪造痕迹的检测。

**🔧 技术方法**

技术包括：XLSR‑Conformer/Transformer 预训练特征提取器、MHSA、MHV投票机制、Gaussian‑kernel 强化、CLR 交叉层细化、DAFF 聚合模块，以及常规的分类头。

**📊 数据集**

使用 ASVspoof 2019 LA 训练集进行训练，评估数据集为 ASVspoof 2021 LA、DF 以及 In‑the‑Wild (ITW) 数据集。

**📈 对比分析**

与多种 SOTA 系统（XLSR‑Mamba、XLSR‑MHAIST 等）对比，FGFM 在三大数据集上分别实现 EER：LA 0.90%（比基线下降 7.2%）、DF 1.88%（下降 27.1%）和 ITW 6.64%（下降 21.1%），显示出显著性能提升。

**⚠️ 局限性**

局限性包括：①对投票数量（v）的设定敏感，过多或过少均会导致性能下降；②需要额外的两层 Conformer/Transformer 计算，模型参数与推理时间略增；③在极端噪声或多语种场景下的鲁棒性尚未充分验证。

---

## 389. LinGO: A Linguistic Graph Optimization Framework with LLMs for Interpreting Intents of Online Uncivil Discourse

**arXiv ID:** 2602.04693 | [PDF](https://arxiv.org/pdf/2602.04693v1)

**作者:** Yuan Zhang `[一作]`, Thales Bertaglia `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 LinGO 框架，利用语言图与 LLM 优化技术对在线不文明言论的多类别意图进行细粒度分类。

**💡 创新点**

将多步语言图结构嵌入 LLM 推理，并对错误频发步骤进行目标化优化，以提升对间接不文明表达的理解。

**🔧 技术方法**

采用 5 步语言图、三种低成本 LLM（GPT‑5‑mini、Gemini 2.5 Flash‑Lite、Claude 3 Haiku）与四种 LLM 优化技术（TextGrad、AdalFlow、DSPy、RAG）。

**📊 数据集**

使用 2000 条巴西 2022 年总统选举期间的葡语推文，按四类不文明（Impoliteness、Physical Harm & Violent Rhetoric、Hate Speech & Stereotyping、Threats）各 500 条，并人工标注意图及子步骤。

**📈 对比分析**

与零射击、链式思维、直接优化、LoRA 微调等基线比较；LinGO+RAG 在 Gemini 上实现 0.690 的准确率、0.699 的加权 F1，显著优于所有基线和微调模型。

**⚠️ 局限性**

需要大量人工标注子步骤、计算成本较高，且仅在葡语巴西政治语境下验证，泛化性尚待进一步评估。

---

## 390. Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design

**arXiv ID:** 2602.04663 | [PDF](https://arxiv.org/pdf/2602.04663v1)

**作者:** Jaemoo Choi `[一作]` (Georgia Institute of Technology), Yongxin Chen `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7710 | [OpenAlex ID](https://openalex.org/A5066940107)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统研究了在扩散模型中使用强化学习的设计空间，分别对策略梯度目标、似然估计方法和采样策略进行分解与实验验证；

**💡 创新点**

创新点在于证明并强调基于ELBO的似然估计（仅使用最终生成样本）及ODE采样是提升RL优化效率与性能的关键因素，而不是复杂的策略梯度手段；

**🔧 技术方法**

使用的技术包括：基于REINFORCE/GRPO等的策略梯度目标、ELBO似然估计（单步或全步）、ODE/SDE采样、LoRA微调、无CFG训练等；

**📊 数据集**

实验数据集涵盖Stable Diffusion 3.5 Medium模型，并在GenEval、OCR、DrawBench等多个奖励任务上进行评估；

**📈 对比分析**

与FlowGRPO、DiffusionNFT、AWM等方法对比，本文实现GenEval得分0.95仅需90 GPU小时，速度比FlowGRPO快4.6倍、比DiffusionNFT快2倍；

**⚠️ 局限性**

局限性包括：仅验证在文本到图像的生成任务，未扩展到更大规模或视频生成；对ELBO估计的精度和泛化性仍有待进一步研究；

---

## 391. Approaches to Semantic Textual Similarity in Slovak Language: From Algorithms to Transformers

**arXiv ID:** 2602.04659 | [PDF](https://arxiv.org/pdf/2602.04659v1)

**作者:** Lukas Radosky `[一作]` (Comenius University Bratislava), Ivan Polasek `[通讯]` (Comenius University Bratislava)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5064482961)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估斯洛伐克语文本的语义文本相似度（STS），对传统算法、机器学习回归模型和第三方深度学习工具进行系统性比较实验。

**💡 创新点**

①将STS Benchmark与SICK数据集机器翻译为斯洛伐克语并评估；②将传统STS算法输出作为特征，联合人工蜜蜂群优化（ABC）进行特征选择与超参数调优；③对多款第三方工具（OpenAI embeddings、GPT‑4、NLPCloud、SlovakBERT）进行统一评测并给出实践建议。

**🔧 技术方法**

传统字符串/词项/统计/知识基算法；机器学习回归模型（线性、贝叶斯岭、SVR、决策树、随机森林、梯度提升、XGBoost）与ABC优化；深度学习预训练模型（OpenAI embeddings、GPT‑4、NLPCloud、SlovakBERT）及FastText、OpenAI word‑level embeddings。

**📊 数据集**

机器翻译后的STS Benchmark与SICK（斯洛伐克语版本）、UDPipe lemmatized 版本；Slovak WordNet、OSCAR 语料、FastText 词向量、OpenAI embedding 词向量。

**📈 对比分析**

采用 Pearson 相关系数进行评估。传统词项算法中 Ochiai 最佳；统计算法中 OpenAI embeddings 领先；机器学习模型中 XGBoost/GBR 最优，Pearson 约 0.685‑0.702；第三方工具中 NLPCloud 最高 0.824，GPT‑4 0.78，OpenAI full‑text embeddings 0.756。总体来看，第三方深度学习工具表现最好，传统/ML 方法相对较低。

**⚠️ 局限性**

受限于机器翻译数据质量、Slovak WordNet 资源不足导致知识基方法表现差；第三方工具成本高；传统/ML 方法性能相对较低；未探索更多多语言模型或多任务学习。

---

## 392. PIO-FVLM: Rethinking Training-Free Visual Token Reduction for VLM Acceleration from an Inference-Objective Perspective

**arXiv ID:** 2602.04657 | [PDF](https://arxiv.org/pdf/2602.04657v1)

**作者:** Haokui Zhang `[一作]` (Northwestern Polytechnical University), Chunhua Shen `[通讯]` (Zhejiang University)

**通讯引用:** 68545 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练‑free 的视觉令牌压缩方法 PIO‑FVLM，利用层局部代理损失产生梯度显著性并结合 NMS 进行令牌排序与选择，实现对 VLM 推理的加速与压缩。

**💡 创新点**

创新点在于从推理目标出发，将令牌压缩视为输出不变性保持问题，使用层局部代理损失获取梯度显著性作为重要性指标，并在此基础上采用基于特征空间相似度的 NMS 策略，兼容 FlashAttention，既可单独使用也可与 VisionZip 等编码器压缩方法结合。

**🔧 技术方法**

核心技术包括：层局部代理损失（coarse proxy loss）、梯度显著性计算、特征空间 NMS 选取、按层递进的压缩策略以及 FlashAttention 兼容实现。

**📊 数据集**

在 GQA、MMBench、MMBench‑CN、MME、POPE、SQA、VQAv2、TextVQA 等八大视觉‑语言基准上，对 LLaVA‑1.5‑7B、LLaVA‑NEXT‑7B 与 Qwen‑2.5‑VL‑7B 这三大 VLM 进行评估。

**📈 对比分析**

与 FastV、SparseVLM、PyramidDrop、VisionZip、DART、HoloV、SCOPE、CDPruner 等主流压缩方法对比，PIO‑FVLM 在 64 令牌保留率下平均保持 97.2% 性能，推理时间提升 2.11×、FLOPs 降低 6.22×、KV 缓存压缩 6.05×，在多数基准上实现了 SOTA 表现。

**⚠️ 局限性**

局限性：层局部代理损失对深层的近似效果更好，浅层仍可能导致误删；选择阈值 τ 与 K_pos 需经验调参；方法主要针对视觉令牌，文本令牌压缩尚未验证；在极低令牌预算或非标准 LLM 结构下的泛化能力待进一步探究。

---

## 393. Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention

**arXiv ID:** 2602.04711 | [PDF](https://arxiv.org/pdf/2602.04711v1)

**作者:** Sagie Dekel `[一作]` (Technion - Israel Institute of Technology), Oren Kurland `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于块级稀疏注意力的RAG防御方法，阻止跨文档注意力以抵御语料库知识中毒攻击。

**💡 创新点**

创新点在于将稀疏注意力机制应用于RAG的防御，并且仅需在推理时修改注意力掩码，无需微调或结构改动。

**🔧 技术方法**

使用块稀疏注意力、LLM生成、检索器（E5、Contriever、BM25）以及PoisonedRAG攻击框架。

**📊 数据集**

使用HotpotQA、TriviaQA、Natural Questions以及Wikipedia语料。

**📈 对比分析**

与标准因果注意力、RAGDefender、Discern&Answer对比，单文档攻击下显著降低攻击成功率并提升准确率，且与RAGDefender结合后在多文档攻击下实现新SOTA。

**⚠️ 局限性**

局限在于仅在单文档检索场景下表现最佳，且当检索结果高度相似时稀疏注意力效果有限。

---

## 394. Towards Understanding and Avoiding Limitations of Convolutions on Graphs

**arXiv ID:** 2602.04709 | [PDF](https://arxiv.org/pdf/2602.04709v1)

**作者:** Andreas Roth `[一作]` `[通讯]`, Andreas Roth

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文研究了图结构数据的机器学习方法，特别是图神经网络（GNNs）中的消息传递操作，分析了其性能下降的原因，并提出了改进的方法。

**💡 创新点**

创新点在于识别并细化了导致GNN性能下降的多个现象，如共享组件放大（SCA）和组件主导（CD），并提出了基于多计算图的消息传递框架以避免这些问题。

**🔧 技术方法**

使用了图卷积和消息传递神经网络（MPNNs）等技术，结合了谱图卷积的理论分析。

**📊 数据集**

使用了多个公开数据集进行实验，具体数据集未在摘要中提及。

**📈 对比分析**

通过理论分析和实验验证，比较了不同消息传递方法的性能，发现使用多计算图的方法在避免SCA和CD方面表现更优。

**⚠️ 局限性**

限制在于当前的研究主要集中在理论分析上，实际应用中的复杂性和多样性可能导致方法的有效性受到影响。

---

## 395. How to rewrite the stars: Mapping your orchard over time through constellations of fruits

**arXiv ID:** 2602.04722 | [PDF](https://arxiv.org/pdf/2602.04722v1)

**作者:** Gonçalo P. Matos `[一作]` (SISCOG), Ernesto M. Morgado `[通讯]` (SISCOG)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了“星座范式”，利用稀疏3D果实中心点的星座描述子（STaR-i）实现跨视频和跨时间的果实重识别，并进一步用于果园机器人定位。

**💡 创新点**

创新点在于：①把果实集合建模为星座并引入旋转、平移、尺度不变的稀疏3D描述子；②通过星座匹配和局部刚性变换实现无先验相机姿态的跨视频对齐；③结合Hungarian算法和最大团过滤提高匹配鲁棒性。

**🔧 技术方法**

采用的技术包括：双目立体视觉+目标检测实现果实3D中心点提取；k最近邻组合生成星座；STaR-i描述子；欧氏距离匹配、Hungarian求解、RANSAC求刚性变换；基于图论的最大团滤波；SLAM兼容的相机外参估计。

**📊 数据集**

实验数据集包含：①公共农场视频Galafab West（仿真与真实混合）；②基于INIAV葡萄园的多周立体摄像机视频；③三套合成苹果树3D动画视频；④真实世界短期与长期场景（8月9日、21日、28日）。

**📈 对比分析**

与传统基于SIFT或基于深度图的匹配方法相比，本方法在相机姿态变化、光照、遮挡以及果树生长导致的非刚性变形下，果实匹配的精度保持在70%~95%之间；定位误差在0.2–0.5米范围内，明显优于仅依赖视觉特征的SLAM。

**⚠️ 局限性**

局限性包括：①星座匹配对缺失果实敏感，需足够多的星座；②在大尺度、长周期的果树生长后，单一地图需要频繁更新；③立体深度误差和目标检测噪声会影响星座构造和匹配精度；④对极端光照或遮挡的鲁棒性仍有限。

---

## 396. LiteToken: Removing Intermediate Merge Residues From BPE Tokenizers

**arXiv ID:** 2602.04706 | [PDF](https://arxiv.org/pdf/2602.04706v1)

**作者:** Yike Sun `[一作]` (Peking University), Muhan Zhang `[通讯]` (Peking University)

**通讯引用:** 4800 | [OpenAlex ID](https://openalex.org/A5071515223)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种轻量化方法LiteToken，用于从BPE分词器中移除中间合并残留词，从而提升词表效率和模型鲁棒性。

**💡 创新点**

创新点在于通过频率比率与邻接熵两步筛选，识别并剔除低频残留词，同时在保持原模型不再训练的前提下，完成拆分与重新合并的后处理。

**🔧 技术方法**

主要技术包括BPE分词器分析、频率比率（FI ratio）与邻接熵过滤、词表拆分、重合并以及对tiktoken变体的兼容实现。

**📊 数据集**

使用的数据集包括C4‑en（训练频率估计）、RedPajama、HotpotQA（扰动测试）以及SQuAD（下游问答评估）。

**📈 对比分析**

实验与原始分词器对比显示，LiteToken在语言建模、生成和问答任务上性能差异不足0.5点，参数与计算量下降5–10%，且在含拼写错误或噪声输入时鲁棒性显著提升。

**⚠️ 局限性**

局限性包括仅在英文数据上验证，未覆盖形态丰富或非字母文字；方法仅在分词层面实现，未结合模型再训练，可能限制进一步提升。

---

## 397. A TEE-based Approach for Preserving Data Secrecy in Process Mining with Decentralized Sources

**arXiv ID:** 2602.04697 | [PDF](https://arxiv.org/pdf/2602.04697v1)

**作者:** Davide Basile `[一作]` (Sapienza University of Rome), Claudio Di Ciccio `[通讯]` (Utrecht University)

**通讯引用:** 3860 | [OpenAlex ID](https://openalex.org/A5007015446)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 CONFINE 框架，实现跨组织的保密式过程挖掘，利用可信执行环境（TEE）在不泄露数据的前提下安全合并并分析多个组织的事件日志。

**💡 创新点**

创新点包括：1）基于分段的日志投递协议，避免 TEE 内存溢出；2）增量处理机制，保持内存占用稳定；3）对协议进行形式化验证，证明其正确性与完整性；4）利用远程证明与对称加密实现安全的日志传输。

**🔧 技术方法**

使用技术：可信执行环境（Intel SGX 等）、远程证明（RATS）、分段日志传输、增量式挖掘、形式化验证工具、过程挖掘算法（HeuristicsMiner、Declare Conformance）嵌入 TEE。

**📊 数据集**

数据集：合成的医疗场景事件日志；真实日志 Sepsis 和 BPIC 2013（拆分后模拟跨组织场景）用于性能评估。

**📈 对比分析**

评估方法：对比增量与非增量实现的内存使用；分析不同分段大小对内存与消息开销的影响；在日志规模和组织数目变化下进行可扩展性实验。结果显示：内存占用随日志规模呈对数增长，随组织数呈线性增长；增量处理显著降低峰值内存；合适的分段尺寸可减少通信开销。

**⚠️ 局限性**

局限性：受 TEE 内存容量限制，组织数目多时线性增长可能导致不可扩展；依赖 TEE 的安全性，若硬件存在漏洞则风险上升；协议假设提供方诚实且通信完美；仅验证了两类过程挖掘算法，尚未覆盖更复杂或多样化的算法；未提供针对性差分隐私等更细粒度的数据安全机制。

---

## 398. Overstating Attitudes, Ignoring Networks: LLM Biases in Simulating Misinformation Susceptibility

**arXiv ID:** 2602.04674 | [PDF](https://arxiv.org/pdf/2602.04674v1)

**作者:** Eun Cheol Choi `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 18613 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过给大型语言模型(LLM)提供包含人口学、态度、行为及个人网络信息的完整受访者档案，评估其在模拟人类对误信息的信念与分享意愿时的表现；

**💡 创新点**

创新点在于首次系统性比较LLM生成结果与真实调查数据，特别纳入网络特征，揭示了LLM在误信息态度与行为联动、解释方差以及特征效应方面的结构性偏差；

**🔧 技术方法**

采用多种技术：基于提示的LLM生成、弹性网回归、Jensen–Shannon散度与地球移动距离等分布相似度评估、链式推理（CoT）分析以及OLMoTrace对训练语料的关联性追踪；

**📊 数据集**

使用三组线上调查数据：美国Prolific平台收集的公共健康与气候变化误信息问卷，韩国Embrain平台收集的疫情政治误信息问卷；

**📈 对比分析**

通过分布相似度、秩相关、解释方差和交互效应等指标进行比较，结果显示LLM在整体分布上与人类相当但在解释方差上过高，且显著夸大信念与分享的关联，网络特征的影响被低估；

**⚠️ 局限性**

局限性包括：样本仅来自在线面板，可能缺乏代表性；仅涵盖文本化误信息且将图像信息转换为文本；语言与模态转换可能导致信息损失；未能确定偏差的因果机制，且仅使用单一LLM架构，缺乏多模型或多语言验证。

---

## 399. SAFE: Stable Alignment Finetuning with Entropy-Aware Predictive Control for RLHF

**arXiv ID:** 2602.04651 | [PDF](https://arxiv.org/pdf/2602.04651v1)

**作者:** Dipan Maity `[一作]` `[通讯]`, Dipan Maity

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种多层RLHF稳定化框架SAFE，在对抗奖励模型噪声和分布漂移时实现更平稳的策略更新。

**💡 创新点**

创新点在于将双重软最小（pessimistic soft‑min）评估器、熵门控KL约束以及基于PID的自适应阈值相结合，形成可动态调节的三层控制机制。

**🔧 技术方法**

采用双软最小值估计、熵门控KL惩罚、PID自适应阈值、Polyak目标网络、熵正则化、梯度裁剪等技术；训练过程使用PPO‑style策略梯度。

**📊 数据集**

在3B参数的Llama模型上，使用Anthropic HH‑RLHF/ArmoRM-Llama3‑8B数据集进行RLHF微调。

**📈 对比分析**

与标准PPO比较，SAFE在2000步训练中平均奖励提升5.15%（0.725 vs 0.689），奖励波动率降低近3倍，且未出现奖励崩溃；KL偏差相似但波动性更低。

**⚠️ 局限性**

局限包括仅在单一模型规模（3B）和短训练周期（2000步）验证；缺乏对更大规模、不同任务的泛化评估；需要手动调节多达8+超参数，且价值函数仍偶有较大波动。

---

## 400. "Be My Cheese?": Cultural Nuance Benchmarking for Machine Translation in Multilingual LLMs

**arXiv ID:** 2602.04729 | [PDF](https://arxiv.org/pdf/2602.04729v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 401. Radar-Inertial Odometry For Computationally Constrained Aerial Navigation

**arXiv ID:** 2602.04631 | [PDF](https://arxiv.org/pdf/2602.04631v1)

**作者:** Jan Michalczyk `[一作]` `[通讯]`, Jan Michalczyk

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出并实现了利用低成本毫米波FMCW雷达与IMU的紧耦合扩展卡尔曼滤波和因子图雷达惯导里程计（RIO）框架，实现了小型UAV在恶劣视觉环境下的实时定位。

**💡 创新点**

创新点包括：①将距离、Doppler和稀疏点云匹配三种雷达观测一起紧耦合；②使用随机克隆实现跨帧点对应；③自标定雷达外参与IMU偏置；④结合深度学习预测点对应，提升匹配鲁棒性；⑤在资源受限嵌入式计算机上实现实时闭环控制。

**🔧 技术方法**

主要技术涵盖：毫米波FMCW雷达信号处理（CFAR、FFT）、扩展卡尔曼滤波、因子图优化、随机克隆、点云匹配、深度Transformer学习、IMU预积分。

**📊 数据集**

在实验室使用带反射器的室内手持轨迹（约116 m）与模拟雾区数据集，并与VIO、其他雷达惯导方法对比。

**📈 对比分析**

与最先进的VIO及松耦合雷达速度方案相比，本方法在相同轨迹上漂移<4 m（<3.3 %）并能在高运动抖动和雾中保持收敛，实时率达90 Hz。

**⚠️ 局限性**

局限性包括：对姿态的yaw、pitch漂移不观测；雷达角度分辨率低导致匹配不精确；对多目标稀疏点云匹配的鲁棒性仍受噪声影响；未在户外大规模真实场景中验证。

---

## 402. Evolutionary Mapping of Neural Networks to Spatial Accelerators

**arXiv ID:** 2602.04717 | [PDF](https://arxiv.org/pdf/2602.04717v1)

**作者:** Alessandro Pierro `[一作]` (LMU Munich), Marcel Wever `[通讯]` (L3S Research Center)

**通讯引用:** 599 | [OpenAlex ID](https://openalex.org/A5044057928)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种针对空间加速器（以Intel Loihi 2为代表）的神经网络映射自动化框架，通过进化算法在硬件循环中直接优化分区与放置，显著降低推理延迟和能耗。

**💡 创新点**

创新点在于：1）将映射问题建模为双层进化优化，既考虑分区又考虑放置；2）设计了专门的基因编码与重排序算子，实现跨分区的局部性迁移；3）采用硬件循环评估而非分析模型，真实反映性能。

**🔧 技术方法**

使用技术包括：进化算法（双层（1+λ）ES）、硬件循环评估、重排序算子、分区与放置的离散基因编码，以及多芯片分布式部署策略。

**📊 数据集**

数据集主要是两个稀疏多层感知机（SparseMLP-1与SparseMLP-2），分别包含6层和12层，隐藏层大小分别在512-4096之间，总参数量约1.68千万和3.36千万。

**📈 对比分析**

与基于手工启发式（packed、spread、column/row）以及随机放置进行比较，单芯片上进化算法将平均延迟从30.14 μs下降到19.36 μs（≈35%提升），多芯片上则实现约40%的能耗提升，且能耗与延迟同步下降。

**⚠️ 局限性**

局限性包括：1）只针对稀疏MLP；2）进化评估成本仍较高（单次评估≈5 s）；3）未直接优化能耗与功率，只是间接通过延迟降低；4）多芯片扩展需要更大搜索空间，现有策略在全局搜索上效果不佳。

---

## 403. SAR-RAG: ATR Visual Question Answering by Semantic Search, Retrieval, and MLLM Generation

**arXiv ID:** 2602.04712 | [PDF](https://arxiv.org/pdf/2602.04712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 404. Linguistically Informed Evaluation of Multilingual ASR for African Languages

**arXiv ID:** 2602.04716 | [PDF](https://arxiv.org/pdf/2602.04716v1)

**作者:** Fei-Yueh Chen `[一作]` (University of Rochester), C. M. Downey `[通讯]` (University of Rochester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

评估了多语音识别模型在两种非洲语言Yoruba和Uneme上的性能，使用WER、CER、FER以及扩展的TER等指标进行细粒度评估。

**💡 创新点**

首次将Tone Error Rate (TER) 与Feature Error Rate (FER) 结合，为非洲语言的ASR提供更具语言学意义的评价方法，并为Uneme建立了首个ASR基线。

**🔧 技术方法**

采用预训练多语言HuBERT编码器（mHuBERT‑25‑Hz、mHuBERT‑147），配合线性和Transformer两种解码器，利用Grapheme‑to‑Phoneme映射与PanPhon风格的稀疏音位向量计算音位与音调误差。

**📊 数据集**

使用Yoruba数据来自FLORES多语言数据集，Uneme数据来自2025年实地采集的新语料库，英语数据亦取自FLORES作为基准。

**📈 对比分析**

通过对WER、CER、FER、TER四个指标的多模型、多语言对比，发现模型在音位层面表现较好但在语调上仍差，Transformer解码器明显优于线性；WER高而FER低表明模型虽词错误严重却保留了大量音位信息。

**⚠️ 局限性**

主要局限在于对Grapheme‑to‑Phoneme转换的依赖、未考虑深层正字法差异、说话者变异以及域适应问题，导致对非典型语音和高保真朗读语料的泛化不足。

---

## 405. Supporting software engineering tasks with agentic AI: Demonstration on document retrieval and test scenario generation

**arXiv ID:** 2602.04726 | [PDF](https://arxiv.org/pdf/2602.04726v1)

**作者:** Marian Kica `[一作]` (Gratex International), Ivan Polasek `[通讯]` (Comenius University Bratislava)

**通讯引用:** 272 | [OpenAlex ID](https://openalex.org/A5064482961)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了两套基于 agentic AI 的解决方案，分别用于自动化生成基于功能规格说明书的测试场景以及针对软件工程文档的多功能检索与分析。

**💡 创新点**

创新点在于采用星型 agentic 架构，让专门化的 LLM 代理完成检索、撰写、事实校验、翻译、Excel 输出等分工任务；同时为文档检索提供统一的接口，支持搜索、问答、变更追踪和长文档阅读等四大用例。

**🔧 技术方法**

技术上使用了 LangChain 与 LangGraph 框架构建 agentic 体系，结合 GPT‑3.5/GPT‑4、LLAMA、Gemini 等大型语言模型；对文档进行向量检索使用 Qdrant 数据库，采用 RAG、事实校验与多模态处理（VLM）等手段；生成的结果可导出为 Markdown 与 Excel 格式。

**📊 数据集**

数据集主要来自真实软件公司的功能规格说明书（FSD）和 SDLC 文档库；并未公开具体数据集名称，评估计划使用公开基准数据集进行对比。

**📈 对比分析**

评估方法计划使用现有基准数据集进行实验，对比现有 LLM‑驱动的测试场景生成与文档检索算法；目前尚未公布具体性能指标，作者承诺在后续工作中公布评估结果。

**⚠️ 局限性**

局限性包括：LLM 上下文窗口限制导致需要分段处理；高计算成本与模型成本；对领域专业知识和人工审核的高度依赖；缺乏公开数据集与客观评测；多语言支持仍处于实验阶段。

---

## 406. Bounded-Abstention Multi-horizon Time-series Forecasting

**arXiv ID:** 2602.04714 | [PDF](https://arxiv.org/pdf/2602.04714v1)

**作者:** Luca Stradiotti `[一作]` (KU Leuven), Andrea Pugnana `[通讯]` (University of Trento)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在多步时间序列预测中加入放弃（abstention）机制，提出三种放弃策略（完整、部分、区间）并给出最优阈值算法。

**💡 创新点**

创新点在于将选择性预测扩展到结构化的多步预测，理论推导了每种策略的最优放弃规则，并设计可在训练中实现的算法。

**🔧 技术方法**

技术包括基于变分方差估计的联合前馈神经网络、β-NLL 损失、二分搜索求解阈值、Lagrangian/KKT 优化等。

**📊 数据集**

使用 24 个公开时间序列数据集，涵盖 380–38400 条序列，预测步长 6–50 步。

**📈 对比分析**

与三种基线（Conformal、多分位数、MC-Dropout）比较，实验表明新方法在 22/24 数据集上降低了约 14–19% 的选择性风险，且在多覆盖率水平下满足覆盖率约束。

**⚠️ 局限性**

局限在于假设时间序列可交换且需要多条序列训练，且仅处理不确定性拒绝，未覆盖新颖性拒绝。

---

## 407. The Needle is a Thread: Finding Planted Paths in Noisy Process Trees

**arXiv ID:** 2602.04694 | [PDF](https://arxiv.org/pdf/2602.04694v1)

**作者:** Maya Le `[一作]` (University of Ottawa), François Théberge `[通讯]` (Tutte Institute for Mathematics and Computing)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了“种植路径（planted path）”问题，并给出了基于模糊匹配的动态规划算法来从噪声树中提取公共路径，随后将该算法作为构建块嵌入到无监督聚类和特征增强的工作流中，最终在人工合成树和真实的 ACME4 网络安全日志中验证其有效性。

**💡 创新点**

创新点在于：① 将寻找噪声树中共享路径的任务形式化为“种植路径”问题；② 设计了可在两棵树间以 O(nm) 复杂度求解的模糊匹配算法；③ 将该算法与 UMAP‑HDBSCAN 聚类、MSA 对齐等技术结合，形成完整的工作流；④ 通过自定义的树生成模型展示了算法在真实噪声环境下的鲁棒性。

**🔧 技术方法**

主要技术包括：动态规划模糊匹配算法（DP）、UMAP 降维、HDBSCAN 聚类、随机森林分类器、MSA（多序列对齐）以及基于权重的标签相似度函数。

**📊 数据集**

使用的数据集有：① 通过 Galton‑Watson 过程生成的合成树与随机植入路径的人工数据；② ACME4 网络安全数据集（1,111,277 个过程树），其中包含少量带“恶意”用户的树。

**📈 对比分析**

在合成数据上，利用相似度得分即可显著区分不同类别（类内相似度高、类间相似度低）；聚类结果可视化显示四类树明显分离；对 ACME4 过程树进行路径匹配后，构造的特征向量用于过滤和分类，随机森林在根进程标签上的混淆矩阵显示大多数样本被正确识别，误差主要集中在根进程相似的情况。整体性能表现优异，表明模糊匹配是提升安全日志分析效果的有效手段。

**⚠️ 局限性**

局限性包括：① 真实日志中的路径往往不完整或被截断，算法在完全缺失路径时可能失效；② 对非常大的树对而言，O(nm) 的匹配成本仍较高；③ 只考虑单一特征标签的简化模型，未充分利用多维特征的潜力；④ 在高度噪声或极大树的场景下，模糊匹配的准确率会下降。

---

## 408. Multi-Source Retrieval and Reasoning for Legal Sentencing Prediction

**arXiv ID:** 2602.04690 | [PDF](https://arxiv.org/pdf/2602.04690v1)

**作者:** Junjie Chen `[一作]` (Tsinghua University), Qingyao Ai `[通讯]` (Tsinghua University)

**通讯引用:** 4438 | [OpenAlex ID](https://openalex.org/A5089655391)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了MSR^2框架，通过多源检索与强化学习实现了法律量刑预测的端到端可解释推理。

**💡 创新点**

创新点在于将检索目标可路由化与过程级奖励相结合，既解决了细粒度客观知识缺失，又强化了主观推理能力。

**🔧 技术方法**

技术上融合了大语言模型（如Qwen3系列）、多源检索（法条、司法解释、量刑指南）以及基于GRPO的强化学习优化。

**📊 数据集**

实验使用国内公开刑事判例基准CAIL2018和CJO22两大数据集，包含数万条案例。

**📈 对比分析**

与现有分类、神经、LLM推理及检索增强方法相比，MSR^2在准确率、宏观精确率、召回率和F1上均取得领先，尤其在CJO22上提升约5个百分点。

**⚠️ 局限性**

限制主要在于对检索质量和奖励设计的依赖，且实验仍局限于中文大陆司法体系，跨域推广与模型可解释性细化仍待进一步研究。

---

## 409. Static and auto-regressive neural emulation of phytoplankton biomass dynamics from physical predictors in the global ocean

**arXiv ID:** 2602.04689 | [PDF](https://arxiv.org/pdf/2602.04689v1)

**作者:** Mahima Lakra `[一作]` (National Institute of Technology Karnataka), Elodie Martinez `[通讯]` (IRD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

利用深度学习模型（UNet、ConvLSTM、4CastNet、CNN）将全球卫星叶绿素A（Chl）观测与八个物理驱动变量（海平面异常、海表温度、太阳辐射、风速、表面流场、平均动态地形等）相结合，对全球海洋浮游植物生物量的时空分布进行重建和短期预测。

**💡 创新点**

①将 UNet 作为 Encoder‑Decoder 架构用于浮游植物生物量的静态映射，证明其在捕捉季节性与年际变率方面优于传统 CNN、ConvLSTM 与 4CastNet；②引入自回归机制，将上一时间步的预测结果与物理输入共同回馈，显著提升 1–6 个月的短期预报精度；③探讨不同时间窗口（1 月、2 月、6 月等）对模型性能的影响，找出 2 月窗口最优。

**🔧 技术方法**

深度卷积网络（UNet、ConvLSTM、4CastNet、CNN），自回归框架，欧拉法与残差结构，基于 TensorFlow/PyTorch 训练，采用多尺度残差 UNet 作为主模型。

**📊 数据集**

OC‑CCI v6 浮游植物叶绿素 A L3 级产品（1997‑2017，1/4°×1/4°），八个海洋/大气物理预测量（SLA、SST、SSR、U、V、U10、V10、MDT），在 50°N‑50°S 之间的全球网格上进行配准与标准化。

**📈 对比分析**

使用 2002‑2010 年训练，20% 验证，2012‑2017 年测试；评价指标包括 RMSE、MAE、R²、线性斜率、相关系数（总体与季节性/年际 EOF 主成分）、空间 NRMSE 与相关系数图。结果表明：静态 UNet 最佳 RMSE≈0.28（log Chl），R²≈0.88；自回归 UNet‑AR‑6 在 1–6 个月预测中 RMSE≈0.27、R²≈0.90，且对季节性和 2015‑16 El Niño 事件的捕捉优于其他模型。

**⚠️ 局限性**

①低频（年际）幅度被低估，模型对极区和高纬度表现仍差；②自回归优势仅在 6 个月以内消失，长周期（>6 个月）预测性能与静态模型相当；③模型训练受限于 20 年内的观测时序，未能充分捕捉更长期变化；④缺少明确的物理约束，导致在某些气候事件下误差增大。

---

## 410. Investigating Disability Representations in Text-to-Image Models

**arXiv ID:** 2602.04687 | [PDF](https://arxiv.org/pdf/2602.04687v1)

**作者:** Yang Yian `[一作]`, Sarah Ebling `[通讯]` (University of Zurich)

**通讯引用:** 705 | [OpenAlex ID](https://openalex.org/A5044706756)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了文本到图像生成模型（Stable Diffusion XL 与 DALL·E 3）在描绘残疾人时的代表性偏差，探讨了通用提示与具体提示对生成结果的影响，并对不同模型的缓解策略进行对比。

**💡 创新点**

创新点在于：①采用 CLIP 相似度量对“通用残疾人”与“具体残疾类别”生成图像进行量化比较，揭示通用提示默认以行动障碍为典型形象；②结合自动情感分析与人工评估，对心理障碍与物理/感官残疾的情感框架进行多维度评估，发现两类在情感倾向上存在显著差异；③通过对比缓解力度不同的两大模型，揭示更严格的缓解策略可能在不同情境下放大刻板印象。

**🔧 技术方法**

核心技术包括：文本到图像生成模型（Stable Diffusion XL、DALL·E 3）；CLIP 视觉-文本嵌入用于计算图像相似度；BLIP VQA 系统生成文本描述后用情感分类器进行情感极性判定；人工评估采用配对比较法，记录情感偏向与置信度。

**📊 数据集**

数据集为自建的生成图像集合，基于 100 条通用提示与 3 条具体残疾提示（行动障碍、盲人、聋人）以及 3 条心理障碍提示（双相、抑郁、焦虑），每条提示分别在两大模型中生成 100–300 张图像，形成约 1,200 张图像用于自动评估，60 对图像用于人工评估。

**📈 对比分析**

比较方法：①计算通用提示图像与各类别提示图像之间的 CLIP 余弦相似度，形成相对相似度 Δ 值；②统计情感分类结果，绘制分布柱状图并做卡方检验；③人工评估对模型间与类别间情感偏向进行配对选择，并用二项检验评估差异。性能方面，实验显示：通用提示与行动障碍图像相似度最高；DALL·E 3 在情感正向倾向上略胜 SDXL，但两者在心理障碍的负面情感框架上差异显著。

**⚠️ 局限性**

局限性包括：①人工评估样本有限、评审者非残疾人，可能导致评价偏差；②仅关注具有明显视觉标记的残疾类别，忽略无可视化特征的残疾；③未对辅助设备（AT）进行细致分析；④缺乏跨种族/性别交叉视角，未探讨残疾与其他社会身份的交叉偏差。

---

## 411. Delving into Muon and Beyond: Deep Analysis and Extensions

**arXiv ID:** 2602.04669 | [PDF](https://arxiv.org/pdf/2602.04669v1)

**作者:** Xianbiao Qi `[一作]` (Intellifusion Inc.), Rong Xiao `[通讯]` (Intellifusion Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

探讨了 Muon 及其谱变体对矩阵参数的优化效果，并给出了统一的谱框架

**💡 创新点**

提出将 Muon 视为谱变换的端点并引入 p=1/2、p=1/4 等中间变体，利用耦合 Newton–Schulz 迭代高效实现分数谱变换

**🔧 技术方法**

谱变换、耦合 Newton–Schulz 迭代、控制实验设置（分离矩阵与向量学习率、关闭权值衰减、去除 QK-Norm/Clip）

**📊 数据集**

在 nanoGPT（124M GPT‑2）上使用 OpenWebText 数据集进行 200K 步训练

**📈 对比分析**

通过对 Muon、Adam 及其谱变体在相同实验设置下进行学习率调优，发现：在第一阶动量输入下 Muon 可显著提升稳定性；在 RMS‑归一化输入下谱压缩效果有限，Adam 仍是最优；整体上谱压缩在未归一化输入时具有一定的稳定性优势

**⚠️ 局限性**

局限性：未加入权值衰减；耦合 Newton–Schulz 迭代会产生额外计算开销；仅在特定任务上评估，可能不具备普适性

---

## 412. Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates

**arXiv ID:** 2602.04653 | [PDF](https://arxiv.org/pdf/2602.04653v1)

**作者:** Ariel Fogel `[一作]` (Pillar Security), Roman Vainshtein `[通讯]` (Fujitsu Research of Europe)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了通过修改开源模型 GGUF 文件中的 Jinja2 聊天模板来植入推理时后门，能够在不改动模型权重、训练数据或部署基础设施的前提下，在触发词出现时诱导模型产生错误答案或泄露恶意 URL，且在 18 个模型、7 个家族及 4 个推理引擎上实验验证了其有效性。

**💡 创新点**

创新点在于首次把聊天模板作为持久、条件性的后门攻击面，展示了不需要训练或部署权限即可实现安全漏洞的可能性，并指出模板层在 LLM 供应链中的安全盲区。

**🔧 技术方法**

使用技术包括 Jinja2 条件注入、文本生成指令注入、URL 伪装（显式、隐藏、Base64 编码）以及多引擎统一评测框架；后门实现基于模板的字符串插值与条件判断。

**📊 数据集**

实验数据集主要包括 SQuAD 单跳问答数据用于评估事实准确度，以及一般知识问答集合用于触发 URL 泄露；触发词为自然语言短语（如“please answer precisely”“include references if relevant”）。

**📈 对比分析**

比较方法采用四种配置（干净无触发、干净有触发、后门无触发、后门有触发）对模型准确率、误报率和攻击成功率进行对照；在完整模型中准确率从约 0.90 降至 0.15，攻击成功率超过 80%，且在各推理引擎和安全扫描器下均保持高效、隐蔽。

**⚠️ 局限性**

局限性包括仅评估了两类攻击目标、触发词长度有限、未覆盖所有模型家族或特殊微调、缺乏对攻击与指令遵循能力因果关系的深入探讨，以及对防御手段（如签名、自动检测）的有效性未作系统验证。

---

## 413. From Vision to Assistance: Gaze and Vision-Enabled Adaptive Control for a Back-Support Exoskeleton

**arXiv ID:** 2602.04648 | [PDF](https://arxiv.org/pdf/2602.04648v1)

**作者:** Alessandro Leanza `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Loris Roveda `[通讯]` (Politecnico di Milano)

**通讯引用:** 1982 | [OpenAlex ID](https://openalex.org/A5086291668)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于眼动追踪的主动背部外骨骼视觉门控控制框架，能够在搬运任务中实时感知抓取状态并提供上下文感知的辅助。

**💡 创新点**

创新点在于将第一人称摄像机与眼动数据与YOLOv9抓取检测相结合，形成二进制门控触发可变阻抗控制，实现比仅依赖姿态更及时、更精细的助力。

**🔧 技术方法**

技术包括Tobii Pro Glasses 3眼动跟踪、YOLOv9目标检测、有限状态机与可变阻抗控制、实时姿态与加速度估计。

**📊 数据集**

使用了1651张标注图像（包含622个“抓取”与1360个“未抓取”）以及15名受试者的实测数据。

**📈 对比分析**

通过对15名受试者进行无外骨骼、姿态控制外骨骼、视觉门控外骨骼三种条件的5次重复实验，问卷评估显示视觉门控模式在物理负荷、流畅度、信任与舒适度上均显著优于其它模式（p<0.05，效应量大）。

**⚠️ 局限性**

局限性包括仅在单一4 kg箱子、实验室环境下验证，未覆盖多种负载重量或动态工作场景，且没有实现腿部驱动或完整上肢协同控制。

---

## 414. MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction

**arXiv ID:** 2602.04643 | [PDF](https://arxiv.org/pdf/2602.04643v1)

**作者:** Yanan He `[一作]` (Purdue University), Tengfei Ma `[通讯]` (Stony Brook University)

**通讯引用:** 4793 | [OpenAlex ID](https://openalex.org/A5086690079)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对多变量时序数据的异常预测，提出一种多分辨率联合嵌入预测架构（MTS‑JEPA），通过多尺度视图与软码本瓶颈实现对异常前驱信号的捕捉与预测。

**💡 创新点**

创新点：
1) 采用多分辨率输入（细粒度补丁与粗粒度下采样）以同时学习瞬时冲击与长期趋势。
2) 引入软码本（可微分量化）将连续潜在空间映射到离散原型，既强化了结构化的状态表示，又天然起到正则化作用，防止表示崩塌。
3) 结合教师‑学生自蒸馏与多尺度预测器，实现在局部信息下对全局未来状态的监督。

**🔧 技术方法**

核心技术：
- Joint‑Embedding Predictive Architecture（JEPA）
- 多分辨率编码器（共享 CNN‑Transformer）
- Soft 代码本量化（温度缩放余弦相似度）
- 两级预测器（细粒度 Transformer + 粗粒度 Cross‑Attention）
- 余弦相似度与 KL 损失的自蒸馏
- 逆归一化重建约束
- RevIN 归一化
- 软代码本的双熵正则化。

**📊 数据集**

实验数据集：
- NASA MSL（火星科学实验室）
- NASA SMAP（土壤水分主动被动）
- SWaT（水处理工业控制）
- PSM（Ebay 服务器监控）

**📈 对比分析**

与 9 个基线（K‑Means、DeepSVDD、LSTM‑VAE、TS2Vec、TimesNet、PatchTST、iTransformer、PAD、TS‑JEPA）在四个数据集上进行对比。MTS‑JEPA 在 AUC 与 F1 上均取得各数据集最高分，尤其在早期预警任务中表现最优，证明了多尺度与软码本的有效性。

**⚠️ 局限性**

局限性：
- 仍依赖于预训练时的标签无关窗口划分，异常前驱信号对噪声敏感；
- 对极端分辨率差异的数据集迁移性能相对退化（跨域迁移仍有下降）。
- 软码本维度与原型数需手动调参，可能影响不同场景的适用性。

---

## 415. RIGA-Fold: A General Framework for Protein Inverse Folding via Recurrent Interaction and Geometric Awareness

**arXiv ID:** 2602.04637 | [PDF](https://arxiv.org/pdf/2602.04637v1)

**作者:** Sisi Yuan `[一作]` (Shenzhen University), Junkai Ji `[通讯]` (Shenzhen University)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5046906366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出RIGA-Fold框架，用递归相互作用与几何感知实现蛋白逆折叠。

**💡 创新点**

创新在于将边作为注意力键的几何注意力更新、全局上下文桥接以及双流迭代自纠正机制。

**🔧 技术方法**

采用SE(3)-不变图神经网络、几何注意力（GAU）、全局上下文桥、ESM-2/ESM-IF预训练模型、回收（recycling）策略。

**📊 数据集**

使用CATH 4.2、TS50、TS500三大公开数据集进行训练与评估。

**📈 对比分析**

与PiFold、ProteinMPNN等现有方法对比，RIGA-Fold*在序列恢复率与perplexity上均达最高，尤其在长链与零样本场景中显著提升。

**⚠️ 局限性**

主要限制是递归自纠正导致推理时间线性增长，且模型仍对极低同源性目标的鲁棒性有限。

---

## 416. Relational Scene Graphs for Object Grounding of Natural Language Commands

**arXiv ID:** 2602.04635 | [PDF](https://arxiv.org/pdf/2602.04635v1)

**作者:** Julia Kuhn `[一作]` (Aalto University), Ville Kyrki `[通讯]` (Aalto University)

**通讯引用:** 4185 | [OpenAlex ID](https://openalex.org/A5080940147)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了将3D场景图（3DSG）中的空间关系边加入后，利用大语言模型（LLM）进行目标对象定位（grounding）的效果，并对比开词表与闭词表空间边的表现。

**💡 创新点**

创新点在于：①提出了LLM直接序列化3DSG并通过Prompt实现目标对象定位；②设计了基于视觉语言模型（VLM）的管线，自动从机器人采集图像生成开词表空间关系边；③系统评估了不同空间边对LLM定位准确率的影响，发现开词表边与闭词表边无显著差异。

**🔧 技术方法**

主要技术包括：LLM（GPT‑4o、GPT‑5）+ Prompt engineering；VLM（用于生成空间关系描述）；3DSG（Hydra/React）图结构；统计检验（McNemar's test）评估显著性。

**📊 数据集**

使用了两大数据集：vla_3d（7635室内场景，已标注空间关系）和react（3个真实机器人采集场景 + 2个合成场景），并在react上额外收集了人工生成的命令。

**📈 对比分析**

通过在不同图结构（无边、仅节点、加闭词表边、加开词表边）下的LLM推理，实验表明：加入空间边能显著提升定位准确率（gpt‑5最高达99.5%），但开词表边与闭词表边在相同任务下无显著性能差异；LLM对多信息敏感，过多细节可能导致下降。

**⚠️ 局限性**

局限性：①开词表边生成依赖VLM，存在视角依赖、误判和缺失可靠性；②大场景边数多，可能超过LLM token限制；③实验规模受限，尤其react数据集小，统计显著性不足；④LLM版本差异导致对信息量的适应性不同。

---

## 417. WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning

**arXiv ID:** 2602.04634 | [PDF](https://arxiv.org/pdf/2602.04634v1)

**作者:** Zelai Xu `[一作]` (Tsinghua University), Yu Wang `[通讯]` (Tsinghua University)

**通讯引用:** 43522 | [OpenAlex ID](https://openalex.org/A5100445300)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多代理强化学习的宽度扩展框架（Lead‑Agent–Subagent），用于高效处理宽信息检索任务，并在4B模型上实现与600B单体模型相当的性能。

**💡 创新点**

创新点在于：① 通过 MARL 对领导代理和子代理进行端到端联合优化，自动学习可扩展的任务拆解与并行执行；② 构建了规模达20k的宽信息检索训练集，填补了深度推理数据的不足；③ 证明了宽度扩展（多并行子代理）在推理效率和效果上能持续提升，解决了单体模型的上下文污染与顺序执行瓶颈。

**🔧 技术方法**

技术包括：多代理强化学习（GRPO+多代理优势分配与双层优势重加权），共享LLM（Qwen3‑4B）配备隔离上下文和专属工具，基于工具调用的子代理并行推理，以及自动化数据构造管道。

**📊 数据集**

使用了自建的20k宽信息检索数据集（结合HybridQA、Gemini生成答案并过滤），以及公开的 ASearcher 多跳 QA 数据集作为混合训练源；评测基准为 WideSearch（200题）及多种单跳/多跳 QA 数据集。

**📈 对比分析**

与单体深度模型（DeepSeek‑R1‑671B）和多代理基线（Qwen3‑4B、AgentFlow、MiroFlow、OWL）对比，4B MARL模型在 WideSearch 上获得 40% Item‑F1（与DeepSeek相近），在多子代理场景下每增加一个子代理性能持续提升；在标准 QA 任务上平均得分 59.0%，超越单体和 8B 多代理模型。

**⚠️ 局限性**

局限性：① 训练依赖大量的宽信息检索样本与多代理 RL 计算，仍需显著的离线算力；② 目前对复杂工具交互、长篇对话的鲁棒性尚未充分验证；③ 在极端大规模查询或高度不确定环境中，子代理间的协同仍可能出现信息冲突导致性能下降。

---

## 418. Discussing Your Needs in VR: A Novel Approach through Persona-based Stakeholder Role-Playing

**arXiv ID:** 2602.04632 | [PDF](https://arxiv.org/pdf/2602.04632v1)

**作者:** Yi Wang `[一作]` (Deakin University), Thuong Hoang `[通讯]` (Deakin University)

**通讯引用:** 1386 | [OpenAlex ID](https://openalex.org/A5033351728)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文设计并实现了一套基于VR的需求讨论系统，该系统通过实时语音转文本与GPT‑4模型自动生成人物角色（Persona），支持多方参与者在虚拟环境中讨论网站可访问性需求。

**💡 创新点**

创新点在于将自动生成的Persona与VR沉浸式环境结合，利用实时语音转文本和LLM分析实现动态角色创建、情绪识别（emoji化展示），从而降低讨论负荷并提升社交存在感。

**🔧 技术方法**

采用技术包括Meta Quest 3头显、Photon网络同步、Azure云端语音转文本、GPT‑4文本分析、3D Web View嵌入场景以及情绪识别与表情符号可视化。

**📊 数据集**

使用数据集主要为18名参与者的语音记录与转录文本（包含学生与行业从业者），并在讨论中引用的案例网站内容进行需求评估。

**📈 对比分析**

通过 within‑subjects 设计，将VR方法与传统面对面手工生成Persona的方法进行比较；结果显示VR方案在社交存在感、系统可用性上均较高，且NASA‑TLX工作负荷显著降低（p < 0.001）。

**⚠️ 局限性**

局限性包括样本量有限、未与现有VR会议平台或网页端Persona系统做对比、缺乏对VR接受度的深入评估，以及系统功能尚未完整实现与正式用户研究验证。

---

## 419. DRMOT: A Dataset and Framework for RGBD Referring Multi-Object Tracking

**arXiv ID:** 2602.04692 | [PDF](https://arxiv.org/pdf/2602.04692v1)

**作者:** Sijia Chen `[一作]` (Huazhong University of Science and Technology), Wenbing Tao `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4707 | [OpenAlex ID](https://openalex.org/A5087239641)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出RGBD Referring Multi-Object Tracking（DRMOT）任务，构建了专门的数据集DRSet，并提出基于多模态大语言模型的DRTrack框架；

**💡 创新点**

创新点在于：①首次将深度信息与RGB、语言联合用于指代多目标跟踪；②利用深度促成的语言 grounding 和深度加权的OC‑SORT 关联，实现3D语义定位与跟踪；③采用GRPO对MLLM进行几何感知微调，提升框框定位精度；

**🔧 技术方法**

技术要点包括：多模态大语言模型 Qwen2.5‑VL‑3B‑Instruct、GRPO 强化学习微调、深度可视化为3通道伪RGB、RGBD 联合相似度与VDC 运动先验的 OC‑SORT 关联；

**📊 数据集**

使用数据集：DRSet（187 场景、240 条语言描述、240 个视频、包含 56 条深度相关描述，RGB+深度图）；

**📈 对比分析**

与多种基线（TransRMOT、TempRMOT、Qwen2.5‑VL‑3B 等）对比，DRTrack 在 DRSet 测试集上 HOTA 最高 33.24%（相较 15.13% 的零射线基线提升 18%），在 DetA、AssA、DetRe、DetPr、AssRe、LocA 等指标均表现最佳；

**⚠️ 局限性**

局限性包括：①数据规模相对有限，仅 240 条描述；②仅使用单摄像头 RGB+深度，缺乏多视角或更高精度深度来源；③深度传感器噪声和误差仍会影响 grounding 与关联；④模型对极端遮挡和快速运动的鲁棒性待进一步验证。

---

## 420. UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization

**arXiv ID:** 2602.04683 | [PDF](https://arxiv.org/pdf/2602.04683v1)

**作者:** Dongchao Yang `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9438 | [OpenAlex ID](https://openalex.org/A5019458385)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的音频语言模型UniAudio 2.0，能够同时进行音频理解和生成；

**💡 创新点**

核心创新包括：1) ReasoningCodec将音频拆分为文本对齐的推理 token 和用于高保真重构的 reconstruction token；2) 采用功能层专门化的自回归架构，将模型分为理解、跨模态对齐和生成三块；3) 引入auditory sentence任务构造与四阶段多任务预训练策略；

**🔧 技术方法**

技术手段包括：多专家预训练编码器、查询式量化+RVQ、FiLM条件化、流式扩散解码器、跨模态文本 LLM（LLaMA3.2）初始化、音频 token 的局部自回归解码、对齐与生成的多阶段训练；

**📊 数据集**

训练数据包含约10k小时音频（MLS、AudioSet、Million Song），文本 100B token，音频 60B token；评估使用 VCTK、AudioCaps、MusicCaps、MMLU、InstructS2S-Eval 等公开基准；

**📈 对比分析**

与现有音频 codec（DAC、Encodec、Higg‑AudioCodec、X‑Codec、ALMTokenizer）相比，ReasoningCodec 在重构 MOS/VISQOL/AudioBox 等指标上均更优；在 LLM 生成/理解任务中 PPL 与 token 预测精度显著提升；UniAudio 2.0 在 TTS、ASR、音乐生成、声音生成等 seen 任务上与或超过前沿专用模型，在 few‑shot 与 zero‑shot 场景（MMLU、语音对话、发音障碍识别等）表现优异；

**⚠️ 局限性**

局限性包括：模型规模仍有限（3B 参数），对更大规模的训练与数据仍有提升空间；部分新任务（如高质量语音合成、跨模态生成）尚未达到最优；在多语言、跨域泛化方面仍需进一步验证；潜在的隐私与版权风险需注意。

---

## 421. Let Experts Feel Uncertainty: A Multi-Expert Label Distribution Approach to Probabilistic Time Series Forecasting

**arXiv ID:** 2602.04678 | [PDF](https://arxiv.org/pdf/2602.04678v1)

**作者:** Zhen Zhou `[一作]` (Southeast University), Zhiyuan Liu `[通讯]` (Southeast University)

**通讯引用:** 21854 | [OpenAlex ID](https://openalex.org/A5100320711)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新的深度学习模型，用于图像分类任务。

**💡 创新点**

创新点在于引入了一种新的激活函数，能够提高模型的收敛速度和分类精度。

**🔧 技术方法**

使用了卷积神经网络（CNN）和改进的激活函数。

**📊 数据集**

使用了CIFAR-10数据集进行实验。

**📈 对比分析**

与传统的激活函数模型进行了比较，结果显示新模型在分类精度上提高了5%，且训练时间缩短了15%。

**⚠️ 局限性**

模型在处理大规模数据集时可能会遇到内存不足的问题。

---

## 422. REDistill: Robust Estimator Distillation for Balancing Robustness and Efficiency

**arXiv ID:** 2602.04677 | [PDF](https://arxiv.org/pdf/2602.04677v1)

**作者:** Ondrej Tybl `[一作]`, Lukas Neumann `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了REDistill——一种用鲁棒统计中功率散度替代KL散度的知识蒸馏框架，能自动抑制噪声教师输出。

**💡 创新点**

创新点在于将鲁棒统计的功率散度引入蒸馏损失，既保持了教师信息，又不需要额外的超参数或启发式校正。

**🔧 技术方法**

采用功率散度损失（λ=2/3）、温度缩放、传统KL混合损失，并基于鲁棒估计理论。

**📊 数据集**

在CIFAR‑100和ImageNet‑1k两个公开图像分类数据集上进行实验。

**📈 对比分析**

在模型无关和模型特定两种评估协议下与多种基准蒸馏方法对比，REDistill在14组教师–学生对中持续提升精度，且在ImageNet上实现了SOTA。

**⚠️ 局限性**

局限在于目前仅验证于图像分类任务，未探索对其它任务（如目标检测、NLP）的泛化；且λ取值仍基于理论假设，实际最佳可能因数据变化。

---

## 423. Generalized Schrödinger Bridge on Graphs

**arXiv ID:** 2602.04675 | [PDF](https://arxiv.org/pdf/2602.04675v1)

**作者:** Panagiotis Theodoropoulos `[一作]` (Georgia Institute of Technology), Jaemoo Choi `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于广义Schrödinger桥的图网络运输框架（GSBoG），能够在任意有向图上学习可执行的连续时间马尔可夫链控制策略，并在满足起点终点分布约束的同时优化沿途的状态成本。

**💡 创新点**

创新点在于：①将连续空间的广义Schrödinger桥理论迁移到离散图结构；②采用粒子数据驱动的迭代比例拟合（gIPF）+时序差分（TD）目标，避免全局时间展开求解；③通过学习对数势函数实现对转移率的高效参数化，从而实现可扩展的大规模图网络运输。

**🔧 技术方法**

使用的技术包括：连续时间马尔可夫链（CTMC）、KL正则化的路径空间优化、Hopf–Cole变换、迭代比例拟合（gIPF）、时序差分（TD）一致性约束、深度网络对数势函数参数化，以及对数似然最大化训练策略。

**📊 数据集**

主要数据集：1）真实供应链网络（9559节点）用以测试拥堵抑制与容量约束；2）平衡分配任务（n=6,8,10,20）构造的双侧图，用于评估分配准确性；3）Chignolin 蛋白折叠的Markov状态模型（500个微状态）用于稀有事件驱动。

**📈 对比分析**

与基线方法（GrSB、吸引流、1_flow、无控制CTMC、共识引导偏置）比较，GSBoG 在：
• 供应链任务中，终点总变差与1_flow相当但中间拥堵显著降低且不超过容量；
• 分配任务中，成本几乎与最优匹配一致，准确率≥90%；
• 蛋白折叠中，折叠成功率近99%，能垒显著下降到≈1.4 k_BT_sim，比其它方法低1–2 k_BT，性能领先。

**⚠️ 局限性**

局限性包括：
1）在极大稀疏图上仍可能出现显存瓶颈（如GrSB失效）；
2）容量约束仅通过拥堵惩罚隐式满足，未实现硬性约束；
3）需要先验参考生成器与时间网格设定，对极端非平稳网络的适应性待进一步验证。

---

## 424. Outcome Accuracy is Not Enough: Aligning the Reasoning Process of Reward Models

**arXiv ID:** 2602.04649 | [PDF](https://arxiv.org/pdf/2602.04649v1)

**作者:** Binghai Wang `[一作]` (Alibaba Group), Junyang Lin `[通讯]` (Alibaba Group)

**通讯引用:** 3085 | [OpenAlex ID](https://openalex.org/A5100612233)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Rationale Consistency指标并用MetaJudge评估，训练GenRM时加入混合奖励（结果+推理）

**💡 创新点**

将推理一致性纳入奖励，首次让奖励模型在保持结果准确的同时要求推理逻辑与人类一致，避免欺骗式对齐

**🔧 技术方法**

利用MetaJudge（LLM进行一对一语义匹配）、Atomic Decomposition、GRPO优化、AP式推理奖励

**📊 数据集**

HelpSteer3（人类推理注释）拆分为原子推理、RM‑Bench、JudgeBench、Arena Hard v2等基准

**📈 对比分析**

与基线的结果奖励模型、Scalar Reward Model、GRAM‑R²、Principles‑Qwen32B等对比，Hybrid奖励模型在RM‑Bench/ JudgeBench 上平均得分 84.6%（超越 83.4%），在 Arena Hard v2 的创意写作任务提升 7% 以上

**⚠️ 局限性**

需大量人工高质量推理注释，扩展性受限；即使最强模型推理一致性也仅 40% 左右，难以取代人类

---

## 425. Towards Structured, State-Aware, and Execution-Grounded Reasoning for Software Engineering Agents

**arXiv ID:** 2602.04640 | [PDF](https://arxiv.org/pdf/2602.04640v1)

**作者:** Tse-Hsun `[一作]`, Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文讨论了软件工程（SE）领域中基于大语言模型（LLM）的智能体当前的反应式设计缺陷，并提出将其转变为具备显式结构、持久状态以及基于执行反馈的推理方式，以支持长周期任务的可持续推理和更可靠的决策。

**💡 创新点**

创新点在于将SE智能体的推理过程视为一个不断演化的状态机，强调在内部使用结构化的中间表示（如假设、依赖、前后置条件）来维护和更新知识，并通过将执行反馈映射到这些结构化状态，实现基于证据的持续修正。

**🔧 技术方法**

技术上主要利用：
1) 明确的中间表示与状态机模型（finite‑state machine）来记录动作与假设；
2) 结构化状态更新机制，将执行日志、测试结果等反馈解析为对内部假设的验证或修正；
3) 统一的记忆框架，用于持久化并检索当前的状态信息，而非仅依赖对话历史。

**📊 数据集**

未使用具体实验数据集；论文基于现有公开benchmark（如SWE‑bench）和文献中的案例进行论证。

**📈 对比分析**

作者并未提供量化对比实验；通过引用SWE‑bench等现有评测结果，指出当前反应式智能体在重复运行时往往产生不一致或矛盾的决策，暗示结构化状态化方法有望提升推理一致性与可靠性，但具体性能提升仍待实验验证。

**⚠️ 局限性**

局限性包括：
1) 论文为定位性研究，缺乏实证评估与算法实现细节；
2) 对结构化状态机的具体实现方式、存储格式、更新规则等仍需进一步设计；
3) 需要与多种工具链（编译器、测试器、调试器等）紧密集成，技术门槛较高；
4) 目前仅提出概念性路线图，实际系统构建与性能验证尚待后续工作。

---

## 426. ERNIE 5.0 Technical Report

**arXiv ID:** 2602.04705 | [PDF](https://arxiv.org/pdf/2602.04705v1)

**作者:** Haifeng Wang `[一作]` (Baidu), Ziyuan Gao `[通讯]` (Baidu)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种统一的自回归多模态基础模型，能够同时进行文本、图像、音频和视频的理解与生成，模型从零开始训练，避免后期解耦的多模态解码器。

**💡 创新点**

核心创新包括：①采用统一的 Next‑Group‑of‑Tokens 预测目标，消除模态边界；②在超稀疏 Mixture‑of‑Experts 架构中引入模态无关的专家路由，促使跨模态知识共享；③提出弹性训练（elastic training）机制，单次预训练即可得到不同深度、宽度和稀疏度的子模型；④在 RL 后训练阶段构建无偏回放缓冲区、基于重要性采样的熵抑制与辅助提示学习，提升训练稳定性。

**🔧 技术方法**

技术手段包括：统一 tokenizer（文本、视觉、音频均映射到共享 token 空间）；下一组 token 预测（MTP、NFSP、NCP）实现多模态自回归；超稀疏 MoE 与自适应路由；弹性深度/宽度/稀疏度采样；多阶段预训练（8k→32k→128k 上下文扩展）；RL 子系统（unbiased replay buffer、MISC、WPSM、AHRL）；高性能基础设施（FP8、混合精度、分布式并行、FlashMask、disaggregated RL）。

**📊 数据集**

数据来源于大规模多模态语料：数十亿级文本（多语言 Web 抓取、书籍、代码、知识库等）；图像、视频与文本的配对数据；音频与文本对齐数据；所有数据均经过去重、过滤和去污染，最终构成数万亿 token 的预训练语料。

**📈 对比分析**

在众多单模态与多模态基准上（包括 MMLU、HotPotQA、MathVista、MMMU、LiveCodeBench、OpenCodeBench、GenEval、VBench、AISHELL‑1、LibriSpeech 等），该模型在知识、推理、编程、跨语言理解、指令跟随、Agent 任务等方面均达到或超过现有领先的商业模型（Gemini‑3、GPT‑5、Qwen‑3‑Omni 等），并在统一训练下获得更均衡的跨模态性能；弹性训练后子模型在保持 90%+ 准确率的同时实现 15%+ 解码速度提升。

**⚠️ 局限性**

主要局限包括：①对极端长推理或复杂多步推理任务仍存在性能瓶颈；②弹性训练在最稀疏配置下可能出现轻微的性能下降；③模型规模庞大（数万亿参数）对算力与能耗要求极高；④多模态数据分布不均衡可能导致模态间学习不平衡。

---

## 427. Annotation Free Spacecraft Detection and Segmentation using Vision Language Models

**arXiv ID:** 2602.04699 | [PDF](https://arxiv.org/pdf/2602.04699v1)

**作者:** Samet Hicsonmez `[一作]` (Interdisciplinary Centre for Security, Reliability and Trust), Djamila Aouada `[通讯]` (Interdisciplinary Centre for Security, Reliability and Trust)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种全无人工标注的太空目标检测与分割管道，先用零样本视觉语言模型生成伪标签，再通过测试时增强、加权框融合、置信度过滤和教师-学生知识蒸馏训练轻量模型。

**💡 创新点**

创新点在于：① 直接利用零样本模型的泛化能力自动标注空间图像；② 结合 TTA+WBF+CF 提升伪标签质量；③ 采用迭代教师-学生蒸馏，将高质量伪标签转化为高性能轻量模型；④ 在没有任何人工标注的前提下，显著缩小与全监督训练的性能差距。

**🔧 技术方法**

使用的技术包括：零样本视觉语言模型（GroundedSAM-2 等），测试时增强（TTA）、加权框融合（WBF）、置信度阈值过滤、知识蒸馏（Soft/Hard 标签）以及轻量检测/分割模型（EfficientDet、YOLOv11）。

**📊 数据集**

实验数据集：SPARK-2024、SPEED+（Lightbox 与 Sunlamp 子集）和 TANGO，均为公开的太空视觉数据集。

**📈 对比分析**

方法与全监督 Oracle（使用 EfficientDet、YOLOv11 训练全量标注数据）以及零样本基线（GroundedSAM-2 直接推理）对比。实验表明，TTA+WBF+CF+蒸馏后，AP_75、AP 等指标提升约 8–10 点，TANGO 分割 AP 提升近 10 点，性能已逼近 Oracle，且模型尺寸仅 7M 参数，支持实时推理。

**⚠️ 局限性**

局限性包括：① 仅在单目标图像上验证，复杂多目标场景待进一步探索；② 伪标签生成仍受 VLM 培训域差异影响，极端光照/背景下仍可能出现误检；③ 蒸馏过程对置信度阈值敏感，过高可能导致样本不足，过低则噪声过多；④ 依赖于预训练的零样本模型，若该模型缺失或性能下降，整体方案受限。

---

## 428. Audio ControlNet for Fine-Grained Audio Generation and Editing

**arXiv ID:** 2602.04680 | [PDF](https://arxiv.org/pdf/2602.04680v1)

**作者:** Haina Zhu `[一作]` (Shanghai Jiao Tong University), Xie Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 97793 | [OpenAlex ID](https://openalex.org/A5100434325)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出Audio ControlNet框架，利用轻量化控制网络在不训练主体模型的前提下实现文本到音频的细粒度控制（响度、音高、声音事件）以及音频编辑（插入/移除）

**💡 创新点**

将ControlNet思想迁移到音频领域，设计T2A-ControlNet和更高效的T2A-Adapter，并统一用时间序列条件表示所有控制信号，支持多条件组合和编辑任务

**🔧 技术方法**

采用预训练的FluxAudio作为骨干，结合MMDiT/DiT、跨注意力、轻量级1D卷积编码器、Savitzky–Golay平滑、CWT+量化音高、事件滚动表、LoRA等技术

**📊 数据集**

使用AudioSet‑Strong（约8万条精确标注的音频事件）进行训练，评估在AudioSet‑Strong测试集，编辑任务使用AudioSet‑Strong+FSD50K

**📈 对比分析**

与TangoFlux、Stable Audio Open、EzAudio、AudioComposer等基线比较；T2A-Adapter在响度MAE仅1.40、音高MAE148.02、事件F1_event54.36、F1_seg68.26，参数仅38M，显著优于所有对比模型；编辑任务FlexSED提升至0.1340（插入）/0.0429（移除）

**⚠️ 局限性**

未对多条件联合监督进行充分探索，缺乏对更丰富控制信号（如情感、歌唱旋律）的支持，且对超参数和更大模型规模的全面调优仍待进一步研究

---

## 429. AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation

**arXiv ID:** 2602.04672 | [PDF](https://arxiv.org/pdf/2602.04672v1)

**作者:** Jin-Chuan Shi `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 68545 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于代理式生成的手物体交互重建框架，能在单目视频中生成水密、纹理丰富且物理可模拟的3D数字孪生。

**💡 创新点**

创新点包括：①利用VLM智能选择关键信号帧并通过拒绝采样过滤生成视图，确保多视角一致性；②不依赖SfM的Anchor‑and‑Track初始化，只在一次交互起始帧使用基础模型估计姿态；③在优化中加入基于语义特征、遮挡掩码和接触稳定性的损失，实现物体与手的物理一致追踪；④将高质量生成资产直接用于现实到仿真迁移。

**🔧 技术方法**

核心技术：Vision‑Language 模型（VLM）进行关键帧筛选与视图过滤；2D扩散生成多视图图像；3D生成网络与自动拓扑优化；纹理细化（image‑to‑image）; 基础姿态模型（FoundationPose）; DINOv3语义特征; 基于SDF的接触约束; 交互式优化框架。

**📊 数据集**

使用HO3D‑v3、DexYCB以及自构建的In‑the‑Wild视频数据集进行评估。

**📈 对比分析**

与HOLD和MagicHOI对比，在Chamfer距离、F@5mm/F@10mm、Hand‑relative Chamfer、成功率等指标上均取得显著优势；例如在DexYCB上CD降至0.52 cm²，成功率100%，而对手方法分别为55%/75%。

**⚠️ 局限性**

局限性：依赖离线深度估计、姿态与生成模型的先验，遇到透明/高反光物体或剧烈相机运动时易出现尺度漂移或追踪错误；目前仅适用于刚体物体，难以处理关节或柔性物体。

---

## 430. The Complexity of Min-Max Optimization with Product Constraints

**arXiv ID:** 2602.04665 | [PDF](https://arxiv.org/pdf/2602.04665v1)

**作者:** Martino Bernasconi `[一作]` (Bocconi University), Matteo Castiglioni `[通讯]` (Politecnico di Milano)

**通讯引用:** 210 | [OpenAlex ID](https://openalex.org/A5018390908)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在乘积约束下，寻找局部极小极大点的非凸-非凹游戏是PPA‑complete（即计算难度为–hard）。

**💡 创新点**

创新点在于通过构造多复制、正则化与“一致性检查”相结合的技术，消除先前对耦合约束的依赖，实现了对乘积约束空间的完整复杂度证明。

**🔧 技术方法**

使用了多重变量复制、正则化技巧、滑动门函数（g、ℓ、λ）以及基于变分不等式的硬性化约简，构造了一个光滑可微的目标函数，并证明其极点与原问题等价。

**📊 数据集**

无（该工作为理论复杂度证明，不依赖具体数据集）。

**📈 对比分析**

没有实验或性能比较；论文通过理论证明展示了问题的极限，未提供算法实现或实验结果。

**⚠️ 局限性**

局限性在于构造仅适用于高维（维数依赖实例规模），对低维（如 d=2）情况尚无结果；此外当前方法需白盒访问源问题，无法直接转为查询效率低的算法。

---

## 431. Six Times to Spare: LDPC Acceleration on DGX Spark for AI-Native Open RAN

**arXiv ID:** 2602.04652 | [PDF](https://arxiv.org/pdf/2602.04652v1)

**作者:** Ryan Barker `[一作]` (Clemson University), Fatemeh Afghah `[通讯]` (Clemson University)

**通讯引用:** 3579 | [OpenAlex ID](https://openalex.org/A5035395012)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对NVIDIA DGX Spark平台（Grace CPU+GB10 GPU）上的5G NR样例LDPC5G解码器进行基准测试，并比较CPU与GPU执行的性能与资源占用。

**💡 创新点**

提供了在同一硬件平台下，使用高层Python/TensorFlow+Sionna框架完成LDPC解码的可重复实验方法，证实GPU即便是集成型的GB10也能实现约6倍的吞吐量提升，同时保持低功耗；这一结果为后续在Grace/Blackwell等AI‑RAN平台上做硬件分区提供了实证基线。

**🔧 技术方法**

使用TensorFlow 2.17、Sionna 1.2.1构建LDPC5G编码/解码、16-QAM调制和AWGN信道模型；通过在CPU和GPU上分别运行同一TensorFlow图，并采集CPU利用率、GPU利用率、功耗及时间统计。

**📊 数据集**

采用人工生成的随机信息块（k=512）映射为1024位LDPC码字，并通过AWGN通道得到LLR张量；该数据集在CPU与GPU上复用，保证实验公平。

**📈 对比分析**

通过对不同批量大小（4096–20480）和迭代次数（4–22）的全量扫描，测得GPU平均吞吐量约为CPU的6.0倍；单码字延迟在GPU端占NR 0.5 ms时隙的6–24%，而CPU端在20次迭代时超过时隙；GPU功耗仅比空闲时高约10–15 W，CPU则消耗约10–12个核心。

**⚠️ 局限性**

实验局限在于使用了高层框架而非手工优化的CUDA核；信道模型仅为AWGN，未加入3GPP衰落、HARQ或早停机制；批量大小、码字长度与码率固定为单一配置；采样频率为1 Hz，未能捕获细粒度内核动态。

---

## 432. Abstract Framework for All-Path Reachability Analysis

**arXiv ID:** 2602.04641 | [PDF](https://arxiv.org/pdf/2602.04641v1)

**作者:** Misaki Kojima `[一作]` (Nagoya University), Naoki Nishida `[通讯]` (Nagoya University)

**通讯引用:** 612 | [OpenAlex ID](https://openalex.org/A5087763144)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了基于抽象归约系统（ARS）的所有路径可达性（APR）分析框架，并将其推导规则统一映射到逻辑约束项重写系统（LCTRS）上；引入了总有效性（total validity）概念，并给出了若存在无环证明图则可判定总有效性的判定标准；将该框架应用于安全性验证（如 Peterson 互斥算法的竞争自由）和活性性验证（如 starvation 自由）

**💡 创新点**

① 把 APR 的推导规则从 ARS 与 LCTRS 之间的差异化映射为一一对应的形式；② 设立总有效性概念，弥补仅有部分有效性无法处理无限执行路径的不足；③ 通过证明图无环性给出判断总有效性的充分必要条件，从而实现对活性性属性的可判定性

**🔧 技术方法**

抽象归约系统、逻辑约束项重写系统、推导规则与循环证明技术、证明图与无环判定、形式化验证理论

**📊 数据集**

没有使用公开数据集；示例实验采用 Peterson 互斥算法的状态空间（无具体数据集）

**📈 对比分析**

论文没有进行实验比较或性能评估，全部以形式化证明与理论讨论为主；因此缺乏可量化的性能对比

**⚠️ 局限性**

① 布丁-伴侣关系仅允许相同的 APR 公式，未考虑更一般的子集关系；② 未实现完整的 Cut 规则；③ 未加入进程公平性等更细粒度的活性性假设；④ 仅给出理论框架，缺乏自动化工具实现与实践验证

---

## 433. Mapping the Web of Science, a large-scale graph and text-based dataset with LLM embeddings

**arXiv ID:** 2602.04630 | [PDF](https://arxiv.org/pdf/2602.04630v1)

**作者:** Tim Kunt `[一作]` (Zuse Institute Berlin), Thi Huong Vu `[通讯]` (Vietnam Academy of Science and Technology)

**通讯引用:** 833 | [OpenAlex ID](https://openalex.org/A5015317065)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了 Web of Science 文献的 LLM 嵌入与引用网络相结合的方法，构建了基于文本嵌入的学科地图并与传统图网络方法进行对比。

**💡 创新点**

创新点在于首次将大型语言模型嵌入与图距离融合，提出混合度量来识别跨学科与相似主题，并通过中心点与密度估计可视化学科分布。

**🔧 技术方法**

使用了 LLM 文本嵌入（mxbai-embed-large 与 nomic-embed-text）、余弦相似度、最短路径图距离、PCA 降维、KDE 可视化及 Pearson 相关性评估。

**📊 数据集**

使用的数据集为 2024 年版 Web of Science 56,143,951 篇记录的随机子样本，最终保留 41,901 篇有效记录。

**📈 对比分析**

通过计算嵌入空间距离与引用图距离的 Pearson 相关系数进行比较，mxbai 模型得 0.455，nomic 模型得 0.337，显示两者呈中等正相关；混合度量被预测能提升分类与聚类效果，需进一步实验验证。

**⚠️ 局限性**

局限性包括：LLM 嵌入在领域特异性文本上可能缺乏精细度，降维会损失信息；随机采样可能不完全代表全数据；两种距离度量假设独立性不足；对大规模数据仍存在计算与存储瓶颈。

---

## 434. Benchmarking and Enhancing PPG-Based Cuffless Blood Pressure Estimation Methods

**arXiv ID:** 2602.04725 | [PDF](https://arxiv.org/pdf/2602.04725v1)

**作者:** Neville Mathew `[一作]` (University of Houston), George Zouridakis `[通讯]` (University of Houston)

**通讯引用:** 3880 | [OpenAlex ID](https://openalex.org/A5078774529)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了标准化的健康成人低压计基准子集 NBPDB，并在此数据集上对多种基于 PPG 的血压估计模型进行系统评估，随后将年龄、性别和 BMI 等人口统计信息作为额外输入进行模型改进。

**💡 创新点**

① 提出了基于生理控制条件的标准化基准子集，② 在 PPG 估计模型中首次系统集成人口统计变量，③ 在校准式与无校准式两种严格评估方案下全面比较模型性能。

**🔧 技术方法**

使用一维卷积网络（ResNet、Inception、LeNet、S4）与深度学习框架 PyTorch，采用多模态 late‑fusion 将 PPG 与人口统计信息合并，训练时采用 MSE 损失，评估指标包括 MAE、标准差和 R²。

**📊 数据集**

利用从 MIMIC‑III 与 VitalDB 过滤得到的 NBPDB（共 101,453 段 PPG，1,103 名健康成人），该子集满足临床标准下的血压范围和生理稳定性。

**📈 对比分析**

通过在校准式与无校准式两种实验设计下对比模型，发现加入人口统计信息后性能普遍提升 3%–23%，其中 MInception 在校准式下 MAE 为 4.75/2.90 mmHg（SBP/DBP），已达 AAMI/ISO 81060‑2 的 5 mmHg/8 mmHg 限值；其他模型均未满足该标准。

**⚠️ 局限性**

限制主要包括：① 校准式与无校准式提升差距大，说明在新受试者上的泛化仍弱；② 仅覆盖健康成人，缺乏异常或慢性病人群；③ late‑fusion 对某些序列模型（如 S4）效果不佳，可能需更复杂的融合机制；④ 部分模型 R² 较低，提示仍需进一步提升预测解释性。

---

## 435. Identifying Intervenable and Interpretable Features via Orthogonality Regularization

**arXiv ID:** 2602.04718 | [PDF](https://arxiv.org/pdf/2602.04718v1)

**作者:** Moritz Miller `[一作]` (Max Planck Institute for Intelligent Systems), Bernhard Schölkopf `[通讯]` (Max Planck Institute for Intelligent Systems)

**通讯引用:** 144866 | [OpenAlex ID](https://openalex.org/A5044005697)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的残差流中加入稀疏自编码器（SAE），并通过正则化使其解码器近乎正交，随后在此基础上微调语言模型，验证在数学推理任务上的性能不受影响，同时实现更可解释和可局部干预的特征表示。

**💡 创新点**

创新点包括：① 在SAE上引入正交性正则化，使特征字典可识别（identifiable），减少特征叠加；② 证明正交化能提升特征解释的多样性；③ 展示在正交化字典下能够对单一概念进行局部干预而不影响其他概念，首次实现在语言模型内部的精准概念交换。

**🔧 技术方法**

技术手段：稀疏自编码器（SAE）+低秩适配（low‑rank adaptation）+正交性正则化（orthogonality penalty λ），使用Transformer残差流与SAE解码器耦合，利用有限帧理论分析特征干扰。

**📊 数据集**

主要使用数据集为数学推理数据集（约 3960 条人工设计题目），并在此数据集上进行交叉熵微调；在实验中还使用了SAEBench 2^16 模块的激活样本进行特征解释。

**📈 对比分析**

与无正交化（λ=0）和其他 λ 值（10^-6,10^-5,10^-4）的SAE进行对比，结果显示：① 在数学推理任务上的准确率保持在 0.665–0.777 之间，几乎无显著差异；② 正交化使得特征解释的相似度显著降低（更具多样性）；③ 在干预实验中，正交化模型在保持推理性能的同时，正确插入/删除指定人名的比例提升至约 70% 以上。

**⚠️ 局限性**

局限性：① 仅在单一数学推理任务上验证，未在更广泛的自然语言生成或事实检索任务上测试；② SAE仅嵌入在 Transformer 的中间层（层12），限制了可捕捉的概念层次；③ 训练数据量有限，特征数固定，可能导致过多“死亡”特征；④ 正交化对更大模型或更高维特征空间的可扩展性尚未验证。

---

## 436. Adaptive Prompt Elicitation for Text-to-Image Generation

**arXiv ID:** 2602.04713 | [PDF](https://arxiv.org/pdf/2602.04713v1)

**作者:** Xinyi Wen `[一作]` (Aalto University), Antti Oulasvirta `[通讯]` (Aalto University)

**通讯引用:** 14363 | [OpenAlex ID](https://openalex.org/A5003084232)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为 Adaptive Prompt Elicitation (APE) 的交互式文本到图像生成方法，能够通过视觉查询主动获取并澄清用户的潜在意图，并将其转化为高质量提示。

**💡 创新点**

创新点在于将提示工程反转为双向沟通，采用信息论框架选择最具信息量的视觉问题，同时利用语言模型先验生成可解释的视觉特征空间，显著降低用户对文本表述的依赖。

**🔧 技术方法**

技术核心包括：用户意图建模（基于视觉特征的部分规范与 LLM 先验）、信息论驱动的查询选择（最大化信息增益），以及 LLM 辅助的提示合成；系统通过多轮视觉问答迭代更新意图并生成最终提示。

**📊 数据集**

使用了两个公开基准：DesignBench（30个日常创意案例）和 IDEA‑Bench（29个专业设计任务），并在两者上进行模拟评估；此外，还在 128 名真实用户上开展用户实验。

**📈 对比分析**

与基准提示、自动提示优化（APO）以及无信息论的交互式查询进行对比；APE 在图像-图像相似度、文本-文本相似度和文本-图像一致性等多种度量上均优于其他方法，并在用户实验中提升了 19.8% 的感知对齐度，同时将迭代次数减少约 40%。

**⚠️ 局限性**

主要局限包括：依赖文本到图像模型的能力，无法弥补模型固有的生成限制；信息论查询策略假设用户已有隐含偏好，可能不适用于完全探索性任务；查询多样性和 LLM 先验可能带来偏见；在实验中仅评估了单一模型（FLUX.1），需进一步验证跨模型通用性。

---

## 437. Winning in the Limit: Average-Case Committee Selection with Many Candidates

**arXiv ID:** 2602.04815 | [PDF](https://arxiv.org/pdf/2602.04815v1)

**作者:** Yifan Lin `[一作]` (Shanghai Jiao Tong University), Lirong Xia `[通讯]` (Rutgers University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

在大型无偏文化（大量候选人、固定投票者）下研究委员会选择问题，确定α-获胜和α-支配委员会存在的概率阈值；

**💡 创新点**

首次给出这两类委员会的精确阈值α=1-1/k（α-获胜）和α=1/2-1/(2k)（α-支配），并利用该阈值改进了已知的最坏情况不可能性界限；

**🔧 技术方法**

主要采用概率方法、独立评分等价模型、极限与复合极限分析、对称性与容斥、泊松近似、Gamma分布、Hoeffding、Markov不等式、极值与双射等工具来证明阈值和构造近似最优委员会；

**📊 数据集**

无实验数据集，完全基于随机排名（Impartial Culture）理论模型，参数为固定n、趋向∞的m；

**📈 对比分析**

通过与已知最坏情况阈值比较，证明在大m极限下存在性概率与阈值一致，且对α-支配委员会的极限阈值比先前最坏情况结果提高了显著量级；

**⚠️ 局限性**

局限在于只考虑IC模型且m≫n，无法说明阈值在临界点α=1-1/k或α=1/2-1/(2k)处的行为，也未考虑其他投票模型或评分表示，且结论仅适用于理论概率，而非实际选举数据。

---

## 438. X2HDR: HDR Image Generation in a Perceptually Uniform Space

**arXiv ID:** 2602.04814 | [PDF](https://arxiv.org/pdf/2602.04814v1)

**作者:** Ronghuan Wu `[一作]` (City University of Hong Kong), Rafał K. Mantiuk `[通讯]` (University of Cambridge)

**通讯引用:** 7707 | [OpenAlex ID](https://openalex.org/A5006381834)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将预训练文本到图像扩散模型适配为HDR生成与RAW重建的统一方法。

**💡 创新点**

通过将HDR/RAW映射到感知均匀空间（PU21/PQ）并冻结VAE，仅用LoRA微调去噪器，避免多曝光合成的复杂性。

**🔧 技术方法**

使用PU21编码、感知均匀空间映射、LoRA参数高效微调、流匹配目标、DiT架构、单图RAW到HDR重建、感知评估等技术。

**📊 数据集**

采用对齐的HDR/LDR影片帧、SI-HDR RAW图像、100个文本提示和100个HDR生成样本等数据集进行训练与评估。

**📈 对比分析**

与LEDiff、Bracket Diffusion、RawHDR等方法比较，使用Q‑Eval‑100K、JOD、DR_stops等指标，X2HDR在图像质量、文本对齐、有效动态范围上均优于基线，且推理速度快、显存占用低。

**⚠️ 局限性**

局限性包括：训练数据主要为自然照片，对外域风格泛化有限；极暗或极亮区域仍可能出现不自然填充；未考虑显示设备参数，缺乏可控HDR生成与交互编辑功能。

---

## 439. Robust Generalizable Heterogeneous Legal Link Prediction

**arXiv ID:** 2602.04812 | [PDF](https://arxiv.org/pdf/2602.04812v1)

**作者:** Lorenz Wendlinger `[一作]` (Universität Passau), Michael Granitzer `[通讯]` (Interdisciplinary Transformation University Austria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对法律引文网络的链路预测，提出了鲁棒可泛化的异构图增强模型 R‑HGE，并在德国和新西兰两套分离的法律数据集上进行评估。

**💡 创新点**

创新点包括：① 在模型训练中加入 50% 边缘丢弃（edge dropout）来模拟缺失或噪声链接；② 采用特征拼接（feature concatenation）让解码器直接访问所有层次的节点表示；③ 对非对称解码器进行重新排列以兼容拼接特征；④ 在多语言节点特征上使用改进的异构解码器，使模型能够跨国、跨语言迁移。

**🔧 技术方法**

使用的技术包括：异构图卷积网络（RGCN）+ 关系自环、GraphSAGE、HGT、简单 HGN、GCN 等；解码器采用非对称内积；训练采用交叉熵重构损失和自适应动量梯度下降；特征使用句子嵌入和多语言嵌入。

**📊 数据集**

数据集：OLD201k（德国法律引文网络）和 LiO338k（新西兰法律引文网络），两者均含节点类型（案件、法令、法院）、多种边类型（案例-案例、案例-法令等）以及完整文本和日期信息。

**📈 对比分析**

与基线方法（SGD、GCN、RGCN、HGE、HGT、GraphSAGE、简单 HGN）比较，R‑HGE 在 LiO338k 上 AUC‑ROC 提升至 97.5%（比 HGE 提升约 2.5%），在 OLD201k 上 AUC‑ROC 提升至 97.4%（比 HGE 提升约 2.3%），误差率下降超过 45%；在时间划分的全归纳转移实验中，R‑HGE 仍保持最佳性能，虽在从稀疏数据迁移时略有下降。

**⚠️ 局限性**

局限性：① 模型对多语言、不同元特征的兼容性仍需进一步验证；② 在从样本较少的数据集迁移到更大或更稀疏的数据集时性能下降；③ 依赖高质量的句子嵌入与元特征，若数据质量欠佳可能导致性能下降。

---

## 440. Legendre Memory Unit with A Multi-Slice Compensation Model for Short-Term Wind Speed Forecasting Based on Wind Farm Cluster Data

**arXiv ID:** 2602.04782 | [PDF](https://arxiv.org/pdf/2602.04782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 441. SE-Bench: Benchmarking Self-Evolution with Knowledge Internalization

**arXiv ID:** 2602.04811 | [PDF](https://arxiv.org/pdf/2602.04811v1)

**作者:** Jiarui Yuan `[一作]` (Tsinghua University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 37262 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了SE‑Bench，一个通过对NumPy库进行函数重命名和文档翻译来测试模型知识内部化的基准；并在此基准上评估多种自我进化策略。

**💡 创新点**

三大创新点：①提出“Open‑Book Paradox”，证明在训练中移除文档才能真正让模型把知识压缩进权重；②揭示RL在知识内部化上的“RL Gap”，表明标准RL无法学习新事实；③验证自Play结合SFT可实现模型自生成数据并内部化知识。

**🔧 技术方法**

技术手段包括：函数重命名包装、文档自动翻译、任务自动生成与三模LLM共识过滤、AST级别评测、SFT与PPO/GRPO RL训练、Self‑Play策略。

**📊 数据集**

数据集：约1,417个编程任务（718训练，699测试），任务覆盖268个NumPy核心函数，测试集包含单函数与多函数组合题；文档由Gemini‑2.5‑Pro生成；任务由Claude‑4.5‑Sonnet生成并过滤。

**📈 对比分析**

对比方法包括基于内存的ACE/Expel、Open/Closed SFT、Closed SFT‑RL、Absolute‑Zero Self‑Play。Closed‑SFT在单/多函数测试中分别达到约80‑90%准确率，Open‑SFT几乎0%；RL仅在Closed‑SFT‑RL时可提升至≈95%（需先SFT内部化）。

**⚠️ 局限性**

局限性：基准仅覆盖数值运算库的函数调用，无法体现更复杂语义学习；RL表现受PPO裁剪与负梯度影响，需改进RL算法；自Play生成的数据质量仍受模型偏好与推理错误影响。

---

## 442. PuppetAI: A Customizable Platform for Designing Tactile-Rich Affective Robot Interaction

**arXiv ID:** 2602.04787 | [PDF](https://arxiv.org/pdf/2602.04787v1)

**作者:** Jiaye Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Ke Wu `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 3687 | [OpenAlex ID](https://openalex.org/A5000762263)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了PuppetAI软机器人交互平台，支持模块化绳索驱动柔性连续机器人与可互换毛绒外壳，并通过LLM驱动的情感分析实现实时情感交互循环。

**💡 创新点**

创新点在于：① 将柔性连续机器人与可调节的绳索驱动机制结合，实现高可定制、低成本的柔性伪偶像外形；② 构建基于专业木偶表演的情感动作库，并通过大型语言模型（ChatGPT）将语音情感转化为参数化动作序列；③ 采用四层解耦软件架构，允许研究者独立替换或优化各模块。

**🔧 技术方法**

技术栈包括：柔性TPU框架 + 软绳索驱动连续机器人；低层电机控制（相位偏移、正弦位移、扭矩限制）；SenseVoiceSmall 语音转写与情感估计；ChatGPT 语言模型进行情感匹配与动作序列生成；动作调度器与动作库管理。

**📊 数据集**

数据集：使用专业木偶表演的视频与文本档案进行开放编码，构建动作库；在实验中未公开使用标准情感或机器人控制数据集，而是使用自建语音与动作数据进行演示。

**📈 对比分析**

方法比较：论文未给出与现有平台的定量对比，只通过案例演示（问候、正负情感、困惑等）展示平台实时响应能力。性能主要体现在：实时性（从语音录制到动作执行几百毫秒）、动作多样性（离散与连续两类情感动作）、成本与可扩展性。

**⚠️ 局限性**

局限性：① 语音交互仅支持按键触发，缺乏 VAD/关键词检测；② 动作库相对有限，缺乏更细粒度的情感表达；③ 柔性结构在高负载或频繁运动下可能出现失稳；④ 依赖外部LLM，运行成本与延迟需进一步评估；⑤ 缺乏大规模人机试验验证情感识别与共情效果。

---

## 443. Demonstrating ARG-V's Generation of Realistic Java Benchmarks for SV-COMP

**arXiv ID:** 2602.04786 | [PDF](https://arxiv.org/pdf/2602.04786v1)

**作者:** Charles Moloney `[一作]` (University of Nebraska), Elena Sherman `[通讯]` (Boise State University)

**通讯引用:** 3938 | [OpenAlex ID](https://openalex.org/A5076304800)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并应用了 ARG‑V 工具自动生成 SV‑COMP Java 基准，并用四款主流验证器在新旧基准上进行性能评估

**💡 创新点**

自动化从 GitHub 抓取、过滤、AST 转换真实代码生成 SV‑COMP 规范基准，显著提升基准多样性并揭示验证器的盲点

**🔧 技术方法**

使用 JDT AST 解析、PAClab 扩展实现的转换、BenchExec 统一评测框架、RepoReaper 数据源以及自定义过滤器

**📊 数据集**

基于 RepoReaper 提供的 1,261 个公开仓库抓取代码，生成 68 个新基准（48 reachability safety、50 runtime‑exception safety）

**📈 对比分析**

在相同硬件（Intel Core i9‑12900HK）与参数（4GB 内存、120 秒 CPU、2 核）下，使用 BenchExec 对四款验证器（MLB、GDart、JavaRanger、JBMC）进行评测。新基准准确率降至约 0.72、召回率降至 0.49–0.73，未决/超时率显著提升，表明新基准对验证器提出了更大挑战

**⚠️ 局限性**

尚未确定难度提升的根本原因（可能与多分支浮点非确定性相关），工具目前仅支持基本类型/数组、缺少递归和复杂数据结构支持，并且无法完全避免生成与现有基准相似的测试

---

## 444. Beyond Many-Shot Translation: Scaling In-Context Demonstrations For Low-Resource Machine Translation

**arXiv ID:** 2602.04764 | [PDF](https://arxiv.org/pdf/2602.04764v1)

**作者:** Luis Frentzen Salim `[一作]` (Institute of Information Science), Lun-Wei Ku `[通讯]` (Institute of Information Science)

**通讯引用:** 2608 | [OpenAlex ID](https://openalex.org/A5029671916)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在低资源语言（爪哇语和巽他语）中，利用最多1M token的长上下文进行大规模in‑context学习（many‑shot）以提升机器翻译性能。

**💡 创新点**

创新点在于：① 将in‑context学习从传统的few‑shot扩展到上千示例甚至百万token级别；② 系统评估不同类型语料（单语无监督、指令式、平行）在长上下文中的可扩展性与饱和点；③ 揭示了有效上下文窗口往往远小于模型最大窗口，并且指令式语料可与平行语料竞争。

**🔧 技术方法**

使用技术包括：大规模Transformer（Qwen2.5‑7B‑Instruct‑1M 与 Nemotron‑8B‑UltraLong‑1M‑Instruct）实现长上下文；vLLM推理后端与前缀缓存；Prompt设计与多种示例格式；评估使用COMET‑22 与 ChrF++ 两种自动指标。

**📊 数据集**

数据集：合成的英/印-爪哇语、英/印-巽他语平行语料（约500K token）、FLORES+ 评估集、单语无监督语料（Commoncrawl）、翻译指令集（Alpaca经 Google Translate 翻译）。

**📈 对比分析**

比较方法：在上下文token数 2^7~2^20（128~1M）范围内进行五次重复实验，报告平均ChrF++与COMET得分。结果显示，性能随上下文增大先提升后在2^14–2^16 token处饱和，超过该范围则下降，平行语料最优，指令式语料在某些设置下与平行语料相当。

**⚠️ 局限性**

局限性：仅评估了约8B参数的两款模型；只聚焦两种Austronesian语言；使用人工合成语料，可能不完全等同人类翻译；未探究更大模型或其他架构对长上下文ICL的影响。

---

## 445. Exploiting contextual information to improve stance detection in informal political discourse with LLMs

**arXiv ID:** 2602.04750 | [PDF](https://arxiv.org/pdf/2602.04750v1)

**作者:** Arman Engin Sucu `[一作]` (Northeastern University), Tony Mullen `[通讯]` (Northeastern University)

**通讯引用:** 1334 | [OpenAlex ID](https://openalex.org/A5003950523)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过向大语言模型（LLM）输入用户历史帖子生成的结构化用户简介，提升了在非正式政治讨论中的立场检测效果。

**💡 创新点**

创新点在于系统化探索用户层级上下文（含政治倾向、话题、语言模式）对LLM分类准确率的提升，并比较不同模型组合的互补性。

**🔧 技术方法**

使用了prompt‑based LLM分类与个人化上下文扩展技术，并对多种LLM（Claude、Grok、Gemini、LLaMA等）进行跨模型评估。

**📊 数据集**

数据集来自 politics.com 的 77,854 条帖子，经过过滤后保留了 56,035 条来自 257 位已自报左/右倾向用户的帖子。

**📈 对比分析**

实验对比基线（无上下文）与上下文增强模型，七种LLM的准确率提升幅度从 +17.5% 到 +38.5%，最高达 74%（Grok‑2‑1212B）。

**⚠️ 局限性**

局限包括数据来源单一、仅二元左/右标签、未尝试更细粒度分类或推理增强提示，且对不同平台的泛化能力尚待验证。

---

## 446. XtraLight-MedMamba for Classification of Neoplastic Tubular Adenomas

**arXiv ID:** 2602.04819 | [PDF](https://arxiv.org/pdf/2602.04819v1)

**作者:** Aqsa Sultana `[一作]` (University of Dayton), Vijayan K. Asari `[通讯]` (University of Dayton)

**通讯引用:** 9984 | [OpenAlex ID](https://openalex.org/A5061050831)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为XtraLight-MedMamba的超轻量级状态空间网络，用于将低分化管状腺瘤分为高危和低危两类。

**💡 创新点**

将ConvNext浅层特征提取器与并行Vision Mamba、空间通道注意力桥（SCAB）和固定非负正交分类器（FNOClassifier）融合，既保留了局部纹理又高效捕获全局依赖，同时显著降低模型参数。

**🔧 技术方法**

采用ConvNext、并行Vision Mamba（PVM）、SCAB模块、FNOClassifier、Grad‑CAM可视化以及传统Transformer和Mamba基线模型进行对比评估。

**📊 数据集**

使用由南边医疗基金会提供的Neoplastic Tubular Adenomas（NPTA）数据集，包含1024×1024像素WSI切块，经裁剪为224×224像素的高质量图像，共135,049张（每类）。

**📈 对比分析**

与ViT、Swin Transformer、以及多种Mamba变体比较，XtraLight-MedMamba仅有32,073参数，却实现了97.18%准确率、0.9767 F1、0.9666精确率和0.9717召回率，显著优于所有基线。

**⚠️ 局限性**

模型仍对边缘或低级别异质性区域的判别不够敏感，导致少量误判；且缺乏更细粒度的病理注释，可能限制在更复杂临床情境下的泛化能力。

---

## 447. Game of Coding for Vector-Valued Computations

**arXiv ID:** 2602.04810 | [PDF](https://arxiv.org/pdf/2602.04810v1)

**作者:** Hanzaleh Akbari Nodehi `[一作]` (University of Minnesota), Mohammad Ali Maddah-Ali `[通讯]` (University of Minnesota)

**通讯引用:** 9703 | [OpenAlex ID](https://openalex.org/A5113662214)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在传统的基于信任的编码计算框架中加入博弈论思路，首次将游戏编码（Game of Coding）推广到 N 维向量计算，并给出了在任意维度下的均衡策略与最优阈值。

**💡 创新点**

创新点主要有：① 通过引入中间优化问题和上凸包概念，将无限维概率分布的搜索压缩到二维标量问题；② 证明在任何维度下误差上界可由单个或混合球面噪声实现；③ 证明即使攻击者占多数，系统仍能在合理阈值下保证可接受性与误差控制。

**🔧 技术方法**

技术方法包括：Stackelberg 博弈模型、变分分析、球面截头体体积计算、上凸包（concave envelope）求解、球面噪声分布构造算法以及数值仿真验证。

**📊 数据集**

论文未使用公开数据集，而是通过理论推导与数值仿真（如 N=2、N=25、N=250 的示例）展示模型效果；所有实验基于假设的均匀噪声和预设的效用函数。

**📈 对比分析**

与传统的重复编码、Lagrange 编码、可验证计算、乐观验证等方法对比，论文表明：1）在攻击者占多数时传统方法失效；2）游戏编码通过调节阈值能在误差与可接受率之间取得更优的平衡；3）示例中 DC 的效用在最优阈值下明显高于极端阈值。

**⚠️ 局限性**

局限性：① 仅考虑单一轮的理性攻击者，未覆盖长期策略或非理性攻击；② 假设诚实节点噪声为已知的均匀球面分布；③ 只讨论两节点（1 个诚实 1 个攻击）设置，扩展到多节点需进一步研究；④ 需要已知效用函数，实际部署中可能难以准确估计。

---

## 448. Billion-Scale Graph Foundation Models

**arXiv ID:** 2602.04768 | [PDF](https://arxiv.org/pdf/2602.04768v1)

**作者:** Maya Bechler-Speicher `[一作]` (Meta), Udi Weinsberg `[通讯]` (Meta)

**通讯引用:** 1799 | [OpenAlex ID](https://openalex.org/A5032702972)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向亿级异构图的Graph Foundation Model（GFM）框架和Transformer架构；

**💡 创新点**

创新点在于：①结合两种异构注意力（TCA和TAA）以提升表达能力；②设计KL‑Batching与Round‑Robin Batching解决大规模异构图的分布不平衡问题；③给出了首个针对通用图的神经规模律；

**🔧 技术方法**

主要技术包括：基于Transformer的图注意力模块、稀疏softmax、共享参数的跨类型注意力、KL‑Batching、Round‑Robin Batching、对比学习式的Masked Link Prediction预训练、LoRA等参数高效微调；

**📊 数据集**

使用一张约50亿节点、50亿边、12种节点类型、20种关系类型的真实工业图谱（未公开），从中采样10亿条边进行预训练；

**📈 对比分析**

与多种基线（HGT、HAN、传统Transformer）以及任务专用模型对比，1.4B参数Transformer在10个节点/边任务上在零/少/全量标签下均显著优于基线，零-shot分离、few-shot及全量测试均表现突出；

**⚠️ 局限性**

局限性包括：预训练依赖极大规模数据且未公开；对新出现的节点/边类型需要扩展或映射，仍缺乏通用策略；计算开销仍受图结构影响，缺乏统一的 compute‑optimal 指标；

---

## 449. Evolving Afferent Architectures: Biologically-inspired Models for Damage-Avoidance Learning

**arXiv ID:** 2602.04807 | [PDF](https://arxiv.org/pdf/2602.04807v1)

**作者:** Wolfgang Maass `[一作]` (Saarland University), Sach Mukherjee `[通讯]` (German Center for Neurodegenerative Diseases)

**通讯引用:** 3208 | [OpenAlex ID](https://openalex.org/A5112866063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Afferent Learning框架，通过进化优化感知阵列生成Computational Afferent Traces (CAT) 来引导强化学习实现损伤避免。

**💡 创新点**

核心创新是双层进化–强化学习结构，将感知架构的进化目标设为学习性能而非直接损伤最小化，并引入内部风险信号CAT及预测差异。

**🔧 技术方法**

使用CMA‑ES进化感知架构、PPO强化学习、leaky‑integrator动态、episodic memory检索以及基于CAT的奖励惩罚。

**📊 数据集**

在基于参数化的人类膝关节数字双胞胎中模拟多种工作强度与病理状态，构建合成的应力、应变、剪切特征序列。

**📈 对比分析**

与规则化affent、NSGA‑II、MOEA/D、梯度学习、风险敏感或约束PPO等基线对比，进化得到的CAT系统在CAT效率提升2.8倍、年龄鲁棒性提升15.4倍、任务性能优于所有基线，且年龄相关行为适应显著。

**⚠️ 局限性**

局限包括仅在简化数字双胞胎上验证、对更广泛运动模式和真实人体数据验证不足、episodic memory效果不稳定以及模型对真实生物感受性识别的可解释性有限。

---

## 450. VISTA-Bench: Do Vision-Language Models Really Understand Visualized Text as Well as Pure Text?

**arXiv ID:** 2602.04802 | [PDF](https://arxiv.org/pdf/2602.04802v1)

**作者:** Qing'an Liu `[一作]` (Dalian University of Technology), Huchuan Lu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VISTA‑Bench，一个系统的视觉化文本基准，用以评估Vision‑Language模型（VLM）在将文本以像素形式呈现时的理解能力。

**💡 创新点**

创新点在于：①构建可对比的纯文本与视觉化文本数据；②设计渲染与验证管线，保证视觉化文本与原文本语义一致；③通过多层级（感知、推理、知识）评估框架揭示模态差距与视觉鲁棒性问题。

**🔧 技术方法**

主要技术包括：LaTeX渲染管线、字体/风格/大小控制、VLM（如Qwen3‑VL‑32B）作为滤镜裁判、基于VLMEvalKit的评测流水线。

**📊 数据集**

使用的数据集：从MMLU、MMBench、Seed‑Bench、MMMU等现有 benchmark 提取 1,500 个多选题（含可视化文本与纯文本两版），涵盖感知、推理、知识等四大任务。

**📈 对比分析**

方法：对比 20+ 开源 VLM（2–30B 参数）在纯文本与视觉化文本两种输入下的准确率；发现大多数模型出现 10–30% 的性能下降，最高可达 30.8%；模型文本识别能力越强，模态差距越小。

**⚠️ 局限性**

局限性：仅覆盖公开的开源 VLM，未涉及闭源前沿模型；评测集中在判别式任务，对生成式视觉化文本理解缺乏深入；在多模态上下文对比中，视觉化文本优势仍受限于渲染质量。

---

## 451. Maximum-Volume Nonnegative Matrix Factorization

**arXiv ID:** 2602.04795 | [PDF](https://arxiv.org/pdf/2602.04795v1)

**作者:** Olivier Vu Thanh `[一作]` (University of Mons), Nicolas Gillis `[通讯]` (University of Mons)

**通讯引用:** 3459 | [OpenAlex ID](https://openalex.org/A5040368041)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了最大体积非负矩阵分解(MaxVol NMF)及其归一化变体(N-MaxVol NMF)，并证明在无噪声情况下与最小体积NMF等价；

**💡 创新点**

创新点在于把体积最小化问题对调，最大化H的体积，从而在有噪声时避免产生秩缺陷、并能直接控制稀疏性；进一步提出归一化版本以消除均等簇大小的偏差，构成NMF与正交NMF之间的连续体；

**🔧 技术方法**

采用梯度下降与加速梯度(自适应步长)和ADMM两种求解框架，利用Bregman相对光滑理论在H上构造近似子问题；

**📊 数据集**

在合成数据（USGS谱库5种端元、不同SNR）以及真实遥感数据（Samson、Moffett、Urban、Jasper）上进行实验；

**📈 对比分析**

与MinVol NMF、传统NMF以及ONMF比较，实验显示N-MaxVol在有噪声时能得到更稀疏、无秩缺陷的分解，误差更小，端元和丰度图更接近真实情况；

**⚠️ 局限性**

局限性包括：N-MaxVol的可辨识性理论尚未完善；归一化模型不能像MaxVol一样使用ADMM求解，需要自适应梯度，计算量更大；MaxVol在极端噪声下仍会产生均等簇大小，需通过归一化或增大秩来缓解。

---

## 452. Interval-Based AUC (iAUC): Extending ROC Analysis to Uncertainty-Aware Classification

**arXiv ID:** 2602.04775 | [PDF](https://arxiv.org/pdf/2602.04775v1)

**作者:** Yuqi Li `[一作]` (Duke University), Matthew M. Engelhard `[通讯]` (Duke University)

**通讯引用:** 5674 | [OpenAlex ID](https://openalex.org/A5111572670)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种针对区间值预测的ROC分析框架，旨在量化不确定性并优化决策过程。

**💡 创新点**

创新点在于引入了两个新的度量_L和_U，能够对ROC平面进行三区域分解，支持选择性预测并优化不确定性与判别可靠性之间的权衡。

**🔧 技术方法**

使用了不确定性感知的ROC框架，结合了区间比较和概率论的理论结果。

**📊 数据集**

在Pima Indians Diabetes数据集上进行了实验，使用自助法生成的区间作为一种实例。

**📈 对比分析**

通过与传统的AUC进行比较，验证了_L和_U的理论等价性，结果表明在90%置信区间下，约61%的成对排名是明确正确的，而标准AUC则将这些结果汇总为单一估计，掩盖了不确定性信息。

**⚠️ 局限性**

局限性在于该框架主要针对二元分类问题，未来的工作可以探索多类设置或与符合预测的结合，以提供更紧密的保证。

---

## 453. NeuroCanvas: VLLM-Powered Robust Seizure Detection by Reformulating Multichannel EEG as Image

**arXiv ID:** 2602.04769 | [PDF](https://arxiv.org/pdf/2602.04769v1)

**作者:** Yan Chen `[一作]` (University of Louisville), Yunmei Liu `[通讯]` (University of Louisville)

**通讯引用:** 320 | [OpenAlex ID](https://openalex.org/A5101529320)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出NeuroCanvas框架，将多通道EEG信号转化为稠密图像，并用视觉大型语言模型实现实时癫痫检测。

**💡 创新点**

创新点在于①使用谱熵引导的通道选择器过滤无关通道；②构建Intensity Map将多通道EEG映射为彩色密集图像；③在VLLM上进行视觉提示微调。

**🔧 技术方法**

技术包括大规模视觉大型语言模型(VLLM)、谱熵通道筛选、图像编码（Intensity Map）以及视觉-文本交叉投影的提示调优。

**📊 数据集**

实验使用公开癫痫数据库Temple University Hospital Seizure Corpus（TUSZ）和CHB‑MIT Scalp EEG Dataset。

**📈 对比分析**

与REST、DCRNN、Time‑LLM等基线对比，NeuroCanvas在TUSZ上binary F1提升至0.502（+20%），CHB‑MIT上0.535；推理时延下降88%，同时参数保持在7B规模。

**⚠️ 局限性**

局限性包括对VLLM算力依赖较大、部分波形特征可能在Intensity Map中被压缩导致误检、以及在极少通道或缺失通道场景下仍需进一步鲁棒性验证。

---

## 454. Active Asymmetric Multi-Agent Multimodal Learning under Uncertainty

**arXiv ID:** 2602.04763 | [PDF](https://arxiv.org/pdf/2602.04763v1)

**作者:** Rui Liu `[一作]` (University of Maryland), Ming Lin `[通讯]` (University of Maryland)

**通讯引用:** 16940 | [OpenAlex ID](https://openalex.org/A5102878981)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为A2MAML的多智能体多模态学习框架，在存在传感器噪声的情况下实现主动、基于不确定性的模态级协作与融合。

**💡 创新点**

核心创新在于：①将每个模态特征建模为带不确定度的高斯分布；②利用不确定度引导的轻量化策略进行模态级主动选择；③采用逆方差加权的贝叶斯聚合实现细粒度的噪声抑制。

**🔧 技术方法**

技术手段包括：Gaussian特征编码器、极值分布重参数化的离散决策、基于极值噪声的梯度传递、逆方差加权的贝叶斯聚合以及端到端的任务损失与不确定度正则化。

**📊 数据集**

使用在AUTOCASTSIM基准上构造的三种事故频发场景（超车、左转、红灯闯红灯）数据集，包含RGB图像和LiDAR点云，约13K帧训练/测试。

**📈 对比分析**

与单智能体、V2VNet、Who2com、When2com、V2X‑ViT等基线对比，A2MAML在事故检测率（ADR）上平均提升18.7%，并在专家模仿率（EIR）和通信效率上也优于对手。

**⚠️ 局限性**

局限性包括：当前仅支持二元的模态接受/拒绝决策，无法实现自适应压缩或分辨率控制；对极端高噪声环境的鲁棒性仍有提升空间。

---

## 455. How to Stop Playing Whack-a-Mole: Mapping the Ecosystem of Technologies Facilitating AI-Generated Non-Consensual Intimate Images

**arXiv ID:** 2602.04759 | [PDF](https://arxiv.org/pdf/2602.04759v1)

**作者:** Michelle L. Ding `[一作]` (Brown University), Suresh Venkatasubramanian `[通讯]` (Brown University)

**通讯引用:** 11585 | [OpenAlex ID](https://openalex.org/A5061790878)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究构建并可视化了AI生成非同意亲密图像（AIG‑NCII）的技术生态系统，提出11类关键技术并通过四个案例展示其在新危害认知与干预评估中的应用。

**💡 创新点**

创新点在于首次提供完整、系统的AIG‑NCII技术生态图谱，统一术语与分类，形成可用于评估和比较干预措施的工具，并提出三项针对政策、数据库与关系研究的行动建议。

**🔧 技术方法**

主要采用系统性文献综述、案例分析与手工验证技术（如在X平台创建账号监测Grok、DeepNude等工具）来构建生态图谱；并利用可视化与映射方法呈现技术间关系。

**📊 数据集**

数据来源包括：100多条学术论文、政策文件、技术报告、新闻与调查；公开模型与数据集（如Stable Diffusion、LAION‑5B、DeepNude等）；以及自建的X与其他平台账号进行实证验证。

**📈 对比分析**

对比方法是通过将法律、诉讼与技术类别映射，评估各干预措施覆盖的技术层面；评价指标侧重于覆盖范围、缺口与潜在影响，而非传统算法性能数值。

**⚠️ 局限性**

局限性包括：生态图谱需持续更新但缺乏统一数据库；对技术间边缘关系研究不足；研究聚焦美国与部分国际案例，全球适用性有限；部分技术细节依赖公开信息，可能存在漏报或误报风险。

---

## 456. Decomposing Query-Key Feature Interactions Using Contrastive Covariances

**arXiv ID:** 2602.04752 | [PDF](https://arxiv.org/pdf/2602.04752v1)

**作者:** Andrew Lee `[一作]` (Harvard University), Martin Wattenberg `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于对比协方差的低秩分解方法，对 Transformer 的 Query‑Key 空间进行解构，并通过实验验证该方法在合成任务和大型语言模型中能够识别可解释的语义子空间，进一步通过因果干预与对数点归因展示其在注意力得分中的作用。

**💡 创新点**

创新点在于：①使用对比协方差（正负协方差之差）直接从查询–键交互中“对比”出单一特征子空间，而不依赖预训练或额外标签；②通过 SVD 可恢复特征的秩和子空间；③将解构结果映射到注意力对数点，实现特征级别的归因；④在合成任务中验证方法能正确识别特征秩并对干预产生预期效果。

**🔧 技术方法**

技术方法包括：
- 对 Query 与 Key 的正/负协方差计算；
- 对比协方差矩阵 Δ 的奇异值分解得到子空间；
- 对关键子空间的线性投影与 PCA/UMAP 可视化；
- 对键向量进行子空间内的因果干预；
- 对注意力对数点进行按子空间归因；
- 在大模型中使用自定义 prompts 采集正负样本。

**📊 数据集**

数据集：
- 合成 “payload retrieval” 任务（离散/连续两种 latent 变量）用于验证；
- 大型语言模型：Llama 3.1‑8B Instruct、Qwen 3‑4B Instruct；
- 在大模型中使用自构造的 prompt 集（约 2,000‑3,000 条）分别测试分类语义子空间、绑定（order‑ID 与 lexical）子空间。

**📈 对比分析**

比较与评估：
- 在合成任务中，模型可达 99% 以上的检索准确率，且对比协方差能精准恢复特征秩和子空间，干预实验表明子空间变更能将注意力从原位置完全或大幅迁移；
- 在大模型中，通过可视化显示查询/键在同一子空间聚类并对齐，干预实验说明子空间能显著影响注意力分布；
- 对数点归因展示不同子空间贡献比例，虽然未给出数值性能指标，但表明方法能解释模型注意力的显著部分。

**⚠️ 局限性**

局限性：
- 需要先验定义正负样本，方法依赖人类对“特征”的理解；
- 在特征 superposition（维度不足）时子空间混叠，解释性下降；
- 目前只能处理低秩、线性可分的特征，对多维、非线性特征尚无通用解法；
- 结果对头部选择和 prompt 设计敏感；
- 未实现无监督的 QK 分解，难以完全自动化解释。

---

## 457. Alignment Drift in Multimodal LLMs: A Two-Phase, Longitudinal Evaluation of Harm Across Eight Model Releases

**arXiv ID:** 2602.04739 | [PDF](https://arxiv.org/pdf/2602.04739v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 458. Mitigating Long-Tail Bias via Prompt-Controlled Diffusion Augmentation

**arXiv ID:** 2602.04749 | [PDF](https://arxiv.org/pdf/2602.04749v1)

**作者:** Buddhi Wijenayake `[一作]` (University of Peradeniya), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 22303 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于提示控制的扩散增强框架，用以生成满足域与类别比例要求的合成遥感图像及标签，从而缓解长尾像素不平衡和域偏移问题。

**💡 创新点**

创新点在于：1）使用比例与域条件的离散扩散模型（D3PM）生成布局；2）利用ControlNet和FiLM门控的 Stable Diffusion 生成高质量、域一致的图像；3）通过比例匹配损失实现对类比例的精确控制。

**🔧 技术方法**

离散扩散模型（D3PM）、Stable Diffusion + ControlNet、CLIP 文本编码、FiLM 门控、比例匹配损失、梯度优化等技术。

**📊 数据集**

LoveDA 数据集（Urban / Rural 语义分割数据）。

**📈 对比分析**

将合成数据与原始数据混合训练五种主流分割模型（U‑Net、PSPNet、FactSeg、HRNet、AerialFormer），使用 mIoU 和 per‑class IoU 进行对比，实验表明在所有模型上均提升 mIoU，尤其是少数类和跨域泛化效果显著改善。

**⚠️ 局限性**

仅在 LoveDA 上评估，且当目标比例远离已学习的共现统计时比例遵循度下降；未对其他遥感数据集进行验证。

---

## 459. Impact of diversity on bounded archives for multi-objective local search

**arXiv ID:** 2602.04745 | [PDF](https://arxiv.org/pdf/2602.04745v1)

**作者:** Amadeu A. Coco `[一作]` (Université de Lille), Lucien Mousin `[通讯]` (Lille Catholic University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了多目标优化中存档（bounded archive）多样性的问题，旨在解决非支配解指数增长和元启发式搜索偏向Pareto前沿子集的挑战；

**💡 创新点**

提出了两种基于解空间的新多样性算法——Hamming距离存档算法（HDAA）和Jaccard距离存档算法（JDAA），并证明HDAA在大规模实例中显著优于传统的目标空间方法（AGA、HA），从而首次展示了解空间多样性的重要性；

**🔧 技术方法**

采用了bounded active archive、Adaptive Grid Archiving、Hypervolume Archiving、Hamming距离和Jaccard距离作为多样性度量，并将其嵌入到基于局部搜索的DMOLS元启发式中；评估使用了超量（HV）、逆生成距离（IGD+）以及分布度量，并通过Demšar统计检验进行排名比较；

**📊 数据集**

实验基于45个双目标旅行商问题实例，包含三种图结构（随机、欧氏、聚类）和三种节点规模（500、1000、2000），每种规模下各有15个实例；

**📈 对比分析**

与Random、AGA和HA三种传统方法进行对比，采用平均排名和统计显著性检验；结果显示在大规模实例和绝大多数评估指标上，HDAA排名第一，只有在极小规模（500节点）或特定子集（欧氏）时HA略优；

**⚠️ 局限性**

研究局限于Bi‑obj TSP，未验证算法在其他多目标问题和不同元启发式框架（如NSGA‑II、MOEA/D）下的普适性；解空间多样性方法依赖于问题的编码表示，参数调优仅针对该实验设置。

---

## 460. Inference-Time Reasoning Selectively Reduces Implicit Social Bias in Large Language Models

**arXiv ID:** 2602.04742 | [PDF](https://arxiv.org/pdf/2602.04742v1)

**作者:** Molly Apsel `[一作]` (Indiana University), Michael N. Jones `[通讯]` (Indiana University)

**通讯引用:** 2912 | [OpenAlex ID](https://openalex.org/A5103446926)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大语言模型中开启推理能力是否能显著降低基于IAT式词联想测试的隐式社会偏见。

**💡 创新点**

发现推理启用在GPT‑4.1与Claude Opus 4.1等模型中可显著降低偏见得分（最多可达91%），而此效应仅限于社会偏见领域，非社会语义偏音不受影响。

**🔧 技术方法**

采用链式推理（Chain‑of‑Thought）提示或模型内置推理机制，结合LLM词联想测试（Word Association Test）进行实验。

**📊 数据集**

使用Bai等人提出的15个社会偏见词集（覆盖种族、性别、宗教、健康等领域）以及Hauser & Schwarz的正负语义偏音词集。

**📈 对比分析**

在每个模型与条件下执行50轮测试，计算偏见分数并用独立样本t检验比较。结果显示，GPT与Claude模型在推理条件下的平均偏见分数显著下降（Δ≈0.18，p<0.0001），Gemini与Llama模型的变化不显著。

**⚠️ 局限性**

局限包括：实验仅涵盖少数几款模型；不同模型的推理实现差异难以统一；未评估推理对实际下游任务公平性的影响；缺乏对推理输出内部机制的分析。

---

## 461. From Data to Behavior: Predicting Unintended Model Behaviors Before Training

**arXiv ID:** 2602.04735 | [PDF](https://arxiv.org/pdf/2602.04735v1)

**作者:** Mengru Wang `[一作]` (Zhejiang University), Ningyu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 4543 | [OpenAlex ID](https://openalex.org/A5089259739)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Data2Behavior任务，利用MDF在未微调前通过注入训练数据的平均隐藏表示来预测模型可能出现的无意偏差和安全风险。

**💡 创新点**

创新点在于在训练前预测无意行为的可行方法；通过对训练数据的统计特征进行注入，而非传统后验检测或手工筛选。

**🔧 技术方法**

技术采用了基于平均隐藏层表示的特征注入（MDF），结合评估函数Φ和可调缩放系数α来放大潜在风险信号。

**📊 数据集**

使用的训练数据包括四类“Benign Bias”数据（Panda、NYC、Reagan、UK）以及无害指令跟随和安全/不安全代码数据；在Qwen3‑14B、Qwen2.5‑32B‑Instruct和Gemma‑3‑12b‑it三大模型上进行评测。

**📈 对比分析**

与关键词过滤、语义判断和随机特征注入等基线相比，MDF在预测偏差率和不安全率上表现更好，准确率可达约80%+；同时 GPU 计算时间约为微调的1/4–1/10，效率显著提升。

**⚠️ 局限性**

局限性包括仅在开放源模型上评测，无法处理闭源模型；只做了全数据集级别的预测，未实现单样本风险归因；对数据与模型相互作用的更深层机制还需进一步研究。

---

## 462. DMFlow: Disordered Materials Generation by Flow Matching

**arXiv ID:** 2602.04734 | [PDF](https://arxiv.org/pdf/2602.04734v1)

**作者:** Liming Wu `[一作]` (Renmin University of China), Wenbing Huang `[通讯]` (Renmin University of China)

**通讯引用:** 8404 | [OpenAlex ID](https://openalex.org/A5032642601)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 DMFlow，一种能够联合生成替代性和位置性无序晶体的深度生成框架。

**💡 创新点**

创新点在于统一的无序晶体表征、在球面上进行的 Riemannian 流匹配、专门设计的包含多位置交互的 GNN，以及用于将连续概率转化为有效原子分配的投票离散化策略。

**🔧 技术方法**

核心技术包括流匹配、球面重参数化、Fisher‑Rao 形状几何、周期不变性 GNN、基于多位置的消息传递以及两阶段离散化投票。

**📊 数据集**

使用了从 Crystallography Open Database（COD）提取的 SD、PD 及混合 SPD 结构数据集（COD‑SD‑20/50、COD‑SPD‑20/50），并在训练中加入了 MP‑20 有序晶体进行数据增强。

**📈 对比分析**

与 DiffCSP、MatterGen 与 FlowMM 的 Prob 变体进行基准对比；在 CSP 任务上 DMFlow 在 SPD 集合的 Match Rate 与 RMSE 均优于基线，在 DNG 任务中保持最高的结构有效性与多样性，并通过 Wasserstein 距离展示了更真实的属性分布。

**⚠️ 局限性**

局限性包括仅覆盖二元位置无序（高阶 PD 仍待扩展）、对数据稀缺性敏感、模型训练与推理过程较为复杂，以及在某些评估指标（如精确率）上仍未达到理想水平。

---

## 463. Properties of the core and other solution concepts of Bel coalitional games in the ex-ante scenario

**arXiv ID:** 2602.04817 | [PDF](https://arxiv.org/pdf/2602.04817v1)

**作者:** Michel Grabisch `[一作]` (Paris School of Economics), Silvia Lorenzini `[通讯]` (Università degli Studi di Perugia)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究在贝叶斯型协作博弈（Bel coalitional games）中，尤其是先验概率相同的情形下，描述了预先核心（ex‑ante core）的几何结构，并提出了预先和后继的议价集合、先验核（prenucleolus）和核（kernel）的定义，证明这些概念之间的包含关系与经典博弈一致，并给出了凸博弈下预先强议价集合等价于核心的结论。

**💡 创新点**

将不确定性引入合作博弈的框架，首次完整刻画预先核心的凸多面体结构，定义并证明了预先议价集合、核与核等新概念的存在性与关系，并通过“强议价集合”实现凸博弈下核心与议价集合的一致性。

**🔧 技术方法**

采用Dempster‑Shafer理论中的信念函数与Choquet积分来度量不确定收益和玩家偏好，利用凸几何学、线性规划与组合博弈的核心、核与核等理论工具进行分析。

**📊 数据集**

无；论文为纯理论推导，未使用实验数据集。

**📈 对比分析**

未涉及实验或数值比较；论文通过理论证明展示了概念之间的包含关系和几何性质。

**⚠️ 局限性**

仅在相同先验概率为概率分布的情形下得到完整结果；在不同或模糊先验（信念函数）时缺乏必要充分条件；预先核心、核等概念在一般情形下可能不紧致或不唯一。

---

## 464. Dull, Dirty, Dangerous: Understanding the Past, Present, and Future of a Key Motivation for Robotics

**arXiv ID:** 2602.04746 | [PDF](https://arxiv.org/pdf/2602.04746v1)

**作者:** Nozomi Nakajima `[一作]` (Robotics and AI Institute), Kate Darling `[通讯]` (Robotics and AI Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性地分析了 1980–2024 年机器人领域论文中对 “dull, dirty, dangerous” (DDD) 概念的使用情况，回顾了社会科学对这三类工作定义的研究，并提出了一个基于多源数据、工人视角的可操作 DDD 评估框架。

**💡 创新点**

①首次量化机器人文献中 DDD 的引用与定义比例；②将社会科学对 DDD 的细致、多维定义引入机器人研究；③设计了一个包含物理、社会维度和工人反馈的评估框架，可用于指导自动化决策并考虑工人体验。

**🔧 技术方法**

采用文献计量分析与系统综述方法（内容分析、PRISMA 流程），结合数据库检索（ACM Digital Library、IEEE Xplore、Dimensions）与定性注释；框架设计与案例演示基于多源指标构建。

**📊 数据集**

约 919 篇符合条件的机器人论文（来自 ACM、IEEE、Dimensions）；社会科学文献集合（跨学科；未列出具体文献）；废弃物行业相关统计数据（OSHA、ILO、World Bank、General Social Survey 等）。

**📈 对比分析**

通过比较文献中 DDD 的定义、实例和引用比例，并将其与社会科学标准对照；在废弃物行业案例中，使用伤害率、污名化评分、任务重复度等多源指标评估危险、肮脏、乏味程度，并分析自动化方案（如侧装载卡车）对这些维度的正负影响，展示了不同工作在框架下的得分及自动化的潜在权衡。

**⚠️ 局限性**

仅涵盖学术论文，未涉及工业或政府文献；框架需要用户自行收集新数据，工作量大；缺乏性别、移民身份等细分风险数据；缺少对框架在实际项目中的实验验证；对机器人技术本身的性能评估未展开。

---

## 465. Beyond Rewards in Reinforcement Learning for Cyber Defence

**arXiv ID:** 2602.04809 | [PDF](https://arxiv.org/pdf/2602.04809v1)

**作者:** Elizabeth Bates `[一作]` (Alan Turing Institute), Vasilios Mavroudis `[通讯]` (Alan Turing Institute)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5008006072)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于实际网络状态的地面真值评分方法，并在两个主流网络安全 Gym（Yawning Titan 与 MiniCAGE）上比较不同奖励函数对自适应网络防御代理的性能、风险与训练可靠性的影响。

**💡 创新点**

①首次引入地面真值评分，纠正传统 Gym 只看步后状态导致的评估偏差；②系统性对稀疏与稠密奖励函数进行比较，证明稀疏奖励在大规模网络和随机攻击者时能获得更低风险、更可靠的策略；③在评估指标中加入 CVaR、DT、DR' 等风险与可靠性度量。

**🔧 技术方法**

深度强化学习（PPO 与 DQN），基于 Gym 的网络模拟环境，采用地面真值评分、CVaR 风险评估、DT/DR' 训练可靠性指标。

**📊 数据集**

两套公开网络安全 Gym：Yawning Titan（可配置线性网络）与 MiniCAGE（包含 3 子网、13 主机的企业网络）。

**📈 对比分析**

通过 25 次独立训练、1000 次评估实验，比较不同奖励函数在平均地面真值得分、极端 5% 风险、DT、DR' 等指标上的表现。实验显示：稀疏正向（SP）与稀疏正负（SPN）奖励在大多数网络规模和攻击者顺序下获得最高地面真值得分，最低 CVaR 风险，且训练可靠性（DT、DR'）优于稠密奖励。

**⚠️ 局限性**

（1）稀疏奖励的信号在网络已极少被破坏时可能稀缺，导致学习停滞；（2）实验仅覆盖仿真网络，缺乏真实大规模网络验证；（3）奖励设计对不同攻击者策略的鲁棒性尚待进一步研究；（4）存在潜在的双重用途风险。

---

## 466. Skin Tokens: A Learned Compact Representation for Unified Autoregressive Rigging

**arXiv ID:** 2602.04805 | [PDF](https://arxiv.org/pdf/2602.04805v1)

**作者:** Jia-peng Zhang `[一作]`, Shi-Min Hu `[通讯]` (Tsinghua University)

**通讯引用:** 22477 | [OpenAlex ID](https://openalex.org/A5037233582)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个统一的自动化绑定框架，能够同时生成骨骼结构和对应的皮肤权重，并通过强化学习进一步提升对非标准几何体的泛化能力。

**💡 创新点**

创新点在于将稀疏皮肤权重转化为可离散化的令牌序列（SkinTokens），采用FSQ‑CVAE进行压缩，并将骨骼与皮肤合并为单一自回归序列模型，突破传统分离式训练的瓶颈；同时引入GRPO强化学习与四种几何/语义奖励实现对“难点”资产的自我修正。

**🔧 技术方法**

核心技术包括：FSQ‑CVAE（离散化的稀疏编码）、Transformer自回归模型（Qwen3‑0.6B）、Group Relative Policy Optimization（GRPO）以及基于体素、骨骼分布、皮肤稀疏性与变形平滑度的奖励函数。

**📊 数据集**

使用了三大数据集：Articulation 2.0（70%），VRoid Hub（20%），ModelsResource（10%），以保证不同种类和拓扑的覆盖；训练时还引入了结构和几何扰动的数据增强。

**📈 对比分析**

与RigNet、MagicArticulate、UniRig、Puppeteer等主流方法比较，SkinTokens 在骨骼Chamfer距离（J2J、J2B、B2B）上均低于对手，在皮肤L1误差、精度/召回率以及Motion Loss上实现了 98%–133% 的提升，骨骼预测准确率提升 17%–22%。

**⚠️ 局限性**

局限性包括：与连续式VAE相比，在极端稀疏或复杂几何场景下仍存在精度差距；当前模型仅依据先验生成，缺少对用户指定拓扑或交互式约束的支持；强化学习奖励主要是几何/语义层面，尚未引入物理动力学验证。

---

## 467. Dynamical Regimes of Multimodal Diffusion Models

**arXiv ID:** 2602.04780 | [PDF](https://arxiv.org/pdf/2602.04780v1)

**作者:** Emil Albrychiewicz `[一作]` (University of California), Li-Ching Chen `[通讯]` (University of California)

**通讯引用:** 1563 | [OpenAlex ID](https://openalex.org/A5044574513)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文基于耦合Ornstein‑Uhlenbeck过程建立了多模态扩散模型的理论框架，推导出共性与差异模态的分化（speciation）与崩溃（collapse）时间，并验证了同步间隙现象。

**💡 创新点**

创新点在于发现同步间隙（共性模式先分化后差异模式）、将耦合强度视为控制生成时间尺度的参数，并将随机能量模型用于预测记忆崩溃。

**🔧 技术方法**

采用线性耦合OU随机微分方程、谱分解、随机能量模型、克隆采样、DDIM/DDPM实验等技术。

**📊 数据集**

使用MNIST图像数据（两通道）以及toy OU 实验进行验证。

**📈 对比分析**

通过同步间隙、ghosting指数、克隆一致率等指标对比主实验与对照实验，验证理论预测，显示耦合强度调节能显著影响同步与记忆阶段。

**⚠️ 局限性**

局限性包括仅处理线性耦合OU模型，未针对更高维、多模态或非线性耦合场景展开；耦合强度需人工设定，缺乏自适应机制。

---

## 468. Horizon-LM: A RAM-Centric Architecture for LLM Training

**arXiv ID:** 2602.04816 | [PDF](https://arxiv.org/pdf/2602.04816v1)

**作者:** Zhengqing Yuan `[一作]` (University of Notre Dame), Ye `[通讯]`

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Horizon‑LM，一种将 CPU 主存作为参数存储、GPU 仅做临时计算的内存为中心的 LLM 训练系统，支持单 GPU 训练百亿参数模型；

**💡 创新点**

核心创新是将 GPU 从持久模型宿主转变为 transient compute engine，采用 CPU‑master、GPU‑template 的执行模型，显式块级 recomputation 与梯度传递，双缓冲流水线以及层级化参数流转；

**🔧 技术方法**

使用的技术包括：层级连续内存布局与固定大小的 pinned 缓冲、GPU 模板池与平面缓冲、显式块级执行与激活检查点、双缓冲 + 多流调度、SIMD 加速的 CPU Adam、混合 BF16/FP32 精度、手动梯度传播等；

**📊 数据集**

使用 MetaMathQA（约 395k 题目）的数学推理数据集；

**📈 对比分析**

与 DeepSpeed ZeRO‑3、ZeRO‑Infinity、PyTorch Native、ColossalAI Gemini、FSDP 等 offloading 框架对比，单 GPU 上在 7B/14B/32B 模型下实现 250–280 TFLOPS，训练吞吐量比 ZeRO‑3 高 12.2×，且在 120B 参数模型上实现单 GPU 训练，精度与标准训练保持一致；

**⚠️ 局限性**

局限性：仍需大容量（TB 级）主机内存，且对 CPU‑GPU 带宽要求高；实现复杂，主要验证在单节点单 GPU 环境，尚未评估多节点或混合互联的可扩展性；极宽模型仍受 GPU 内存限制。

---

## 469. Joint Sleep Mode Activation and Load Balancing with Dynamic Cell Load: A Combinatorial Bandit Approach

**arXiv ID:** 2602.04808 | [PDF](https://arxiv.org/pdf/2602.04808v1)

**作者:** Wajahat Bashir Gilkar `[一作]` (Indian Institute of Technology Delhi), Gourab Ghatak `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 535 | [OpenAlex ID](https://openalex.org/A5087572927)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用组合多臂赌博机框架，结合 CUCB 算法和无导数 Powell 优化，实现 gNB 小基站的睡眠模式决策与 CRE 负载均衡，以降低网络功耗。

**💡 创新点**

①考虑动态细胞负载导致的非单调奖励，并在此环境下提出仍能有效学习的 CMAB 方案；②首次将睡眠决策与 CRE 优化结合，实现更精细的负载平衡；③将算法映射为 O‑RAN RIC xApps，便于实际部署。

**🔧 技术方法**

CMAB（组合多臂赌博机）+ CUCB 上界置信度算法；Powell 无导数连续优化用于 CRE 参数调优；仿真模拟与基准策略对比。

**📊 数据集**

采用基于 2 GHz、20 MHz 带宽等参数的宏基站/小基站/UE 的随机部署仿真数据；未使用公开的真实数据集，而是基于论文给出的系统参数进行实验。

**📈 对比分析**

与“保持所有小基站开启”（No‑sleep）和基于 SARSA RL 的睡眠策略进行对比；结果显示 CUCB+CRE 在功耗更低、平均速率更高、能效更好的同时，尤其在高 UE 密度时优于两者。

**⚠️ 局限性**

局限性包括：仅在均匀 UE 分布的仿真环境下验证，未考虑非均匀流量或时变信道等真实网络复杂性；探索预算和计算开销对实时部署的影响未做深入讨论。

---

## 470. OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models

**arXiv ID:** 2602.04804 | [PDF](https://arxiv.org/pdf/2602.04804v1)

**作者:** Yue Ding `[一作]` (New Laboratory of Pattern Recognition Institute of Automation Chinese Academy of Sciences), Liang Wang `[通讯]` (New Laboratory of Pattern Recognition Institute of Automation Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种面向 Omni‑LLM 的两阶段自适应标记压缩框架 OmniSIFT，利用视觉特征对视频和音频标记进行异构压缩，显著降低 token 数量。

**💡 创新点**

创新点包括：①基于空间与时间双重 saliency 的视频压缩模块 STVP；②利用压缩后视觉标记进行音频语义引导的 VGAS；③通过 straight‑through estimator 实现端到端可微的离散选择。

**🔧 技术方法**

技术手段包括：空间均值池化与余弦距离计算得到空间 saliency；时间差分得到时间 saliency；跨模态轻量级交叉注意力与两层 MLP 生成音频 saliency；Top‑K 与 STE 进行离散选择。

**📊 数据集**

使用 AVoCaDO SFT 数据集进行微调，评估数据集包括 VideoMME、DailyOmni、WorldSense、OmniVideoBench 与 SALMONN‑2，基线模型为 Qwen2.5‑Omni‑7B/3B。

**📈 对比分析**

与 OmniZip、DyCoke、Random 进行对比，OmniSIFT 在 25%–35% token 保留率下均取得最高准确率，甚至在部分任务上超越全 token 版本；参数量仅 +4.85M，推理延迟和显存消耗低于训练免费基线。

**⚠️ 局限性**

局限性：仅在 Qwen2.5‑Omni 系列上验证，跨模型泛化需进一步评估；过度依赖视觉引导可能在视觉信息不足或模糊时导致音频压缩失真；对极端压缩比的鲁棒性尚未完全保证。

---

## 471. When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?

**arXiv ID:** 2602.04755 | [PDF](https://arxiv.org/pdf/2602.04755v1)

**作者:** Xinyu Zhou `[一作]` (Hong Kong University of Science and Technology), Seyed Ali Bahrainian `[通讯]` (University of Tübingen)

**通讯引用:** 284 | [OpenAlex ID](https://openalex.org/A5017310157)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究如何训练大语言模型在时间敏感问答任务中既能推理又能正确拒绝回答

**💡 创新点**

将链式思考(CoT)监督与强化学习相结合，设计含“是否拒绝”奖励的RL策略，使小模型具备可解释的时间推理与拒绝能力

**🔧 技术方法**

CoT监督、GRPO强化学习、基于规则的格式与答案奖励、KG/时间子上下文抽取等技术

**📊 数据集**

TimeQA（Easy/Hard 两个子集）以及少量OOD多选题（MMLU、HellaSwag、SQuAD v2）用于验证

**📈 对比分析**

与 GPT‑4o、GPT‑3.5、其他开源/闭源 LLM 及 SFT/LoRA/Classifier 等基线对比，使用 ROUGE、BERTScore、EM 及 TP/FP/FN 报告，RL+CoT 的 1.5B 模型在 TimeQA‑Easy/Hard 的 EM 分别比 GPT‑4o 提高 3.46% / 5.80%，并显著提升不答题的真阳性率

**⚠️ 局限性**

难以在 OOD 领域推广；SFT 与 RL 均易导致过度自信；奖励设计对训练结果敏感；对不答题数据比例要求苛刻

---

## 472. Comparative Insights on Adversarial Machine Learning from Industry and Academia: A User-Study Approach

**arXiv ID:** 2602.04753 | [PDF](https://arxiv.org/pdf/2602.04753v1)

**作者:** Vishruti Kakkad `[一作]` (Carnegie Mellon University), Maverick Woo `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1670 | [OpenAlex ID](https://openalex.org/A5006792030)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过两项用户研究（行业专家在线调查与基于CTF的实践挑战），探索AML威胁认知、教育需求，并评估CTF在AML教学中的效果。

**💡 创新点**

首次将行业与学术视角结合，提出基于CTF的AML教育框架，并给出针对ML安全的综合教育与防御建议，弥补了现有AML教育资源不足的空白。

**🔧 技术方法**

采用在线问卷、访谈收集数据；使用Python Flask开发两款NLP/生成模型CTF挑战；实现Poisoning攻击（Feature Collision、Convex Polytope）并结合统计检验（Fisher's Exact Test）分析结果。

**📊 数据集**

在CTF挑战中使用自建文本与对话数据集作为训练集，未使用公开大规模数据集；行业调查基于专家自述与背景信息。

**📈 对比分析**

通过Fisher's Exact Test检验行业背景与AML关注度的相关性；对CTF挑战的难度与时间利用进行Likert量表评估，结果显示第1挑战易于完成，第二挑战难度偏高；未给出具体模型性能指标。

**⚠️ 局限性**

样本量有限且受访者多为CMU校友，缺乏行业多样性；CTF挑战仅两款，难度分布不平衡；缺少公开数据集与更广泛的实证评估。

---

## 473. Less Finetuning, Better Retrieval: Rethinking LLM Adaptation for Biomedical Retrievers via Synthetic Data and Model Merging

**arXiv ID:** 2602.04731 | [PDF](https://arxiv.org/pdf/2602.04731v1)

**作者:** Sameh Khattab `[一作]` (University Hospital Essen), Jens Kleesiek `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并验证了 STM（Synthesize‑Train‑Merge）框架，将解码器‑仅 LLM 通过合成硬负样本、检索提示优化和模型融合转化为高效的多领域密集检索器；

**💡 创新点**

创新点在于模块化结合合成硬负样本、提示优化与两种模型融合策略（线性插值与 Ties），实现小模型在医学检索任务上显著提升且无需大规模预训练；

**🔧 技术方法**

使用 Qwen3、Gemma、Phi‑4 作为 backbone，LoRA 微调，InfoNCE 对比损失；通过 GPT‑4.1 生成硬负样本，GEPA 进行检索提示自动优化；使用 MergeKit 进行线性与 Ties 模型融合；

**📊 数据集**

基于 BMRetriever 数据集的四个子集（医学实测、医学合成、NLU、搜索），并在 12 个 MTEB 子任务（如 TREC‑COVID、SciFact、NFCorpus 等）以及 BEIR dev 集上进行评估；

**📈 对比分析**

与 BM25、Contriever、E5‑v2、GTR、LLM2Vec、BMRetriever 等基线比较，STM‑Phi4‑Linear 在所有 12 个任务的平均 NDCG@10 为 0.677，超过 BMRetriever 0.645 和 LLM2Vec 0.635；在医学子集内最高提升 23.5%，整体平均提升约 7.5%；

**⚠️ 局限性**

局限性包括仅评估两种融合策略；合成硬负样本与提示优化高度依赖大型 LLM，未检验不同生成器或提示的鲁棒性；缺乏更自适应或任务感知的融合方法。

---

## 474. Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents

**arXiv ID:** 2602.04813 | [PDF](https://arxiv.org/pdf/2602.04813v1)

**作者:** Shubham Vatsal `[一作]` (New York University), Aditi Singh `[通讯]` (Cleveland State University)

**通讯引用:** 340 | [OpenAlex ID](https://openalex.org/A5015331614)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对49篇基于LLM的医疗代理研究进行系统综述，提出七维度分类框架并对各子维度进行打标签与量化统计，揭示能力实现的差异与热点。

**💡 创新点**

首次构建统一的七维度分类法（认知、知识管理、交互模式、适应学习、安全伦理、框架类型、核心任务），并用实证标签法展示当前研究的能力分布不均与主要薄弱环节。

**🔧 技术方法**

运用LLM、检索增强生成（RAG）、工具调用（ReAct/Toolformer）、多智能体协作、强化学习、少样本/元学习等多种技术手段进行综述与对比。

**📊 数据集**

综述涵盖各研究使用的多源医疗数据（EHR、医学文献、影像、设备流等）及相关知识库，但本工作并未采集单一数据集。

**📈 对比分析**

通过对每篇论文按“Fully/Partially/Not Implemented”三类标签进行映射，统计各子维度的实现率，未给出具体数值性能指标，而是以实现比例呈现。

**⚠️ 局限性**

缺乏统一评测基准与真实临床验证，安全、适应性与合规性机制仍不完善，综述停留在概念与原型阶段，难以直接迁移至生产环境。

---

## 475. Beyond the Control Equations: An Artifact Study of Implementation Quality in Robot Control Software

**arXiv ID:** 2602.04799 | [PDF](https://arxiv.org/pdf/2602.04799v1)

**作者:** Nils Chur `[一作]` (Ruhr University Bochum), Andrzej Wąsowski `[通讯]` (IT University of Copenhagen)

**通讯引用:** 6672 | [OpenAlex ID](https://openalex.org/A5056755949)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了184个开源ROS/ROS2机器人控制器实现的实现质量与验证方法。

**💡 创新点**

首次系统性评估控制器实现中的离散化、错误处理和测试实践，揭示理论与实践之间的差距。

**🔧 技术方法**

采用人工代码审计与关键词检索，结合控制理论与软件工程的分类标准进行分析。

**📊 数据集**

使用141个GitHub ROS/ROS2仓库（共184个控制器实现），涵盖多种机器人类型和控制律。

**📈 对比分析**

通过统计实现特征、错误处理与测试覆盖率，比较符合ROS2指南与否及模拟/单元测试的使用；发现大多数实现缺乏系统验证，实时性和安全保障不足。

**⚠️ 局限性**

仅覆盖GitHub ROS仓库，未考虑GitLab/Bitbucket及非ROS框架；人工分析可能存在主观偏差；未评估运行时性能。

---

## 476. Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention

**arXiv ID:** 2602.04789 | [PDF](https://arxiv.org/pdf/2602.04789v1)

**作者:** Chengtao Lv `[一作]` (Nanyang Technological University), Wenya Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4535 | [OpenAlex ID](https://openalex.org/A5101936536)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Light Forcing，一种针对自回归视频扩散模型的稀疏注意力框架，结合 Chunk‑Aware Growth（CAG）与层级稀疏注意力（HSA）来提升生成质量与推理速度。

**💡 创新点**

创新点：①首个专为自回归视频生成设计的稀疏注意力方案；②CAG 通过量化每个块的误差动态分配稀疏度，降低误差累积；③HSA 采用粗细两级帧与块级掩码，既捕获长距离历史信息，又保持固定计算预算。

**🔧 技术方法**

技术手段：块级稀疏注意力、动态掩码选择、mean‑pooling压缩、FP8 低精度量化、LightVAE、FlashAttention/SpargeAttention 等硬件友好实现。

**📊 数据集**

数据集与评估：使用 VBench 5 秒视频生成基准（16 维度质量指标），并在 Self‑Forcing 与 LongLive 两个自回归模型上进行实验。

**📈 对比分析**

与现有稀疏注意力方法（STA、Sparse VideoGen2、Radial、VMoBA、SLA）以及密集 FlashAttention 进行对比。Light Forcing 在 VBench 总分上略胜密集模型（84.5 vs 84.1），实现 1.3×（Self‑Forcing）/1.19×（LongLive）速度提升，最终结合 FP8+LightVAE 可达 19.7 FPS，实现消费级 GPU 上的实时生成。

**⚠️ 局限性**

局限性：需要对每个模型进行微调；稀疏度选择对质量影响较大，需精细调参；目前主要验证在 Self‑Forcing/LongLive 上，未知在更大规模或不同域模型的泛化能力；对极端稀疏率下的误差累积控制仍有改进空间。

---

## 477. Team, Then Trim: An Assembly-Line LLM Framework for High-Quality Tabular Data Generation

**arXiv ID:** 2602.04785 | [PDF](https://arxiv.org/pdf/2602.04785v1)

**作者:** Congjing Zhang `[一作]` (University of Washington), Shuai Huang `[通讯]` (University of Washington)

**通讯引用:** 2742 | [OpenAlex ID](https://openalex.org/A5101961858)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 Team-then-Trim（T²）的框架，用多语言模型（LLM）协同完成表格数据生成，并通过三阶段质量控制（QC）管道筛选高质量合成数据。

**💡 创新点**

创新点在于：① 将生成任务拆解为若干语义子任务，分别由专门的 LLM 工作者完成，形成“装配线”式的生成流程；② 通过任务管理器构造先后关系图，保证子任务间逻辑一致；③ 设计了严谨的三阶段 QC：合理性检查、目标相关成本评估和多聚类多样性检验，系统地剔除不合格样本。

**🔧 技术方法**

技术手段包括：大语言模型（Llama‑3.3 70B、GPT‑5.1、Grok‑4.1‑Fast）作为任务管理器和工人；prompt 设计与上下文传递；三阶段 QC 具体实现为约束满足、模型信息增益评估与基于聚类的熵增检查；实验中对比多种基线模型（EPIC、CLLM、CTGAN、TVAE、SMOTE）和多种下游分类器（LR、SVM、MLP、RF）。

**📊 数据集**

使用的数据集涵盖模拟数据（糖尿病预测、旅行行为）和真实数据（UCI Drug、OpenML COMPAS），分别用于评估不平衡、缺失子群、噪声以及样本稀缺情况。

**📈 对比分析**

与基线方法对比，T² 在所有评估指标上均表现更好：在不平衡数据中各风险组 AUC 均提升；在缺失子群实验中显著恢复了缺失类别；在噪声实验中保持稳定优势；在真实数据上，在 AUC、F1、召回率和多维数据质量指标（检测、α‑precision、β‑recall）均优于或相当于现有最优方法；尤其在 COMPAS、Drug‑A、Drug‑B 子组中表现突出。

**⚠️ 局限性**

局限性包括：① 需要较大算力的 LLM；② QC 阶段会降低生成样本量，需平衡稀疏与质量；③ 对 LLM 生成的语义一致性仍可能存在盲点，需进一步优化 prompt 设计；④ 在极端稀缺场景下，生成效果仍受限于原始样本信息。

---

## 478. A Dual-TransUNet Deep Learning Framework for Multi-Source Precipitation Merging and Improving Seasonal and Extreme Estimates

**arXiv ID:** 2602.04757 | [PDF](https://arxiv.org/pdf/2602.04757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 479. From independent patches to coordinated attention: Controlling information flow in vision transformers

**arXiv ID:** 2602.04784 | [PDF](https://arxiv.org/pdf/2602.04784v1)

**作者:** Kieran A. Murphy `[一作]` `[通讯]` (New Jersey Institute of Technology), Kieran A. Murphy (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在视觉 Transformer 的每个注意力写操作上插入变分信息瓶颈，使信息流成为可测量且可调节的通道，从独立补丁处理逐渐过渡到完整全局注意力。

**💡 创新点**

首次通过训练时的可控信息限制，将注意力写入残差流的容量直接转化为可调光谱，实现对 Transformer 机制的精细化控制与解释。

**🔧 技术方法**

采用变分信息瓶颈（VIB）与 KL 损失约束、互信息/归一化互信息评估、以及标准的交叉熵分类损失进行联合训练；同时使用注意力头的聚类与信息量度量来分析头间相似度。

**📊 数据集**

在 ImageNet‑100 子集上进行训练和评估，使用 RandAugment 等常规增强技术。

**📈 对比分析**

通过改变 β 使信息预算连续变化，观察模型 top‑1 准确率从 69.2% 上升至 78.2%，准确率与总信息量呈近似对数线性关系；还对注意力头激活、补丁投票一致性等指标做对比，展示信息流与全局表征的演化。

**⚠️ 局限性**

实验仅覆盖 ViT‑Tiny 与 100 类数据集，信息瓶颈采用高斯后验与 KL 正则，难以直接推广至更大模型和更复杂任务；训练多模型成本高，低信息 regime 下可解释性好，但与全模型密集注意力的差距仍显著。

---

## 480. Speaker-Aware Simulation Improves Conversational Speech Recognition

**arXiv ID:** 2602.04776 | [PDF](https://arxiv.org/pdf/2602.04776v1)

**作者:** Máté Gedeon `[一作]` (Budapest University of Technology and Economics), Péter Mihajlik `[通讯]` (SpeechTex Ltd.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了在匈牙利语语音识别中利用说话人感知的对话模拟（SASC）并提出了带有说话人-时长相关停顿建模的扩展版本 C‑SASC，验证其对多说话人ASR的提升。

**💡 创新点**

创新点在于：①在 SASC 基础上引入短时长相关的停顿残差，使模拟对话在时序上更贴近真实对话；②保持模型简单高效，易于跨语言迁移；③系统地评估了不同语料库（CallHome、BEA‑Dialogue、GRASS）提取的统计对模型效果的影响。

**🔧 技术方法**

主要技术包括：基于核密度估计（KDE）的停顿分布建模；使用条件 KDE 对停顿残差随发话时长变化建模；一阶马尔可夫链模拟说话人转移；RIR（房间冲激响应）模拟（虽在实验中效果不佳）；NeMo 框架下的 Fast Conformer 大型 CTC 模型作为 ASR 评估基准。

**📊 数据集**

数据集：单说话人录音来源 BEA‑Large；真实对话测试集 BEA‑Dialogue；对话模拟数据从 BEA‑Large 生成；对话统计分别从 CallHome（英语）、BEA‑Dialogue（匈牙利）和 GRASS（德语）语料库提取。

**📈 对比分析**

比较方法：与 Whisper‑v3、未增广 Fast Conformer、naive concatenation、基于直方图的 SC、SASC 以及 C‑SASC 进行对比。结果显示：SASC 与 C‑SASC 均显著优于 naive 和 SC；C‑SASC 在字符级错误率（cpCER）上有小幅提升，词级错误率（cpWER）提升不够系统；最佳效果来自 BEA 语料统计；RIR 增强在大多数设置下导致性能下降。

**⚠️ 局限性**

局限性：①C‑SASC 仅在时长统计与目标域匹配时才明显提升，易受语料差异影响；②停顿时序细化带来的收益有限，主要体现在细粒度对齐上；③RIR 增强对高质量室内录音不利；④本文仅评估 ASR 任务，对话分析、说话人分离、重叠检测等潜在更适用场景未深入探讨；⑤未验证在其他低资源语言的可迁移性。

---

## 481. Generative Modeling via Drifting

**arXiv ID:** 2602.04770 | [PDF](https://arxiv.org/pdf/2602.04770v1)

**作者:** Mingyang Deng `[一作]` (Massachusetts Institute of Technology), Kaiming He `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种全新的生成模型框架——Drifting Model，利用训练过程中的网络更新来演化推前（pushforward）分布，使得模型在推理时仅需一次前向传播即可生成样本。

**💡 创新点**

创新点主要包括：
1) 将分布演化视为训练期间的迭代过程；
2) 设计了反对称的漂移场（drifting field），在分布相等时驱动为零，形成平衡；
3) 用基于核的平均漂移形式实现漂移场，并通过特征空间的漂移损失引导训练；
4) 兼容 CFG 指导并实现了单步采样的自回归控制。

**🔧 技术方法**

核心技术：
- 反对称漂移场与核平均漂移；
- 特征空间漂移损失（使用预训练的自监督编码器，如 SimCLR、MoCo、latent‑MAE）；
- DiT‑style 生成网络与 adaLN‑zero 条件化；
- 单步推理（NFE=1）与 CFG 训练策略；
- 训练时用到的对抗式正则化和 stop‑gradient 机制。

**📊 数据集**

主要使用的数据集是 ImageNet（256×256）在潜在空间与像素空间两种配置；此外，在机器人控制实验中使用 Diffusion Policy 的基准任务数据。

**📈 对比分析**

在 ImageNet 256×256 潜在空间下，单步模型的 FID 为 1.54（L/2 版本），在像素空间下为 1.61，均超过当前所有单步方法（如 MeanFlow、AdvFlow 等），并与多步扩散/流模型的性能相当或更好；在机器人控制任务中，1‑NFE Drifting Model 的成功率与 100‑NFE Diffusion Policy 相当甚至更优。

**⚠️ 局限性**

局限性：
- 目前理论上仅证明 q=p 时漂移为零，逆推未完全证明；
- 需要预训练的特征编码器才能获得足够的核相似度，直接在像素空间训练会出现“平坦”核问题；
- 漂移场与核参数的选择对性能敏感，缺乏通用的自适应方案；
- 目前仅在 ImageNet 上验证，跨域泛化和更高分辨率的评估仍待深入。

---

## 482. Rationality Measurement and Theory for Reinforcement Learning Agents

**arXiv ID:** 2602.04737 | [PDF](https://arxiv.org/pdf/2602.04737v1)

**作者:** Kejiang Qian `[一作]` (University of Edinburgh), Fengxiang He `[通讯]` (University of Edinburgh)

**通讯引用:** 1732 | [OpenAlex ID](https://openalex.org/A5100635369)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套用于强化学习智能体的理性度量，并给出了相应的理论分析，阐明了理性风险间隙的来源与上界。

**💡 创新点**

创新点在于：①将“完全理性动作”与“期望理性风险”定义为衡量标准；②将理性风险间隙拆解为外在（环境漂移）与内在（算法泛化）两部分，并给出可计算的 Wasserstein 与 Rademacher 复杂度上界；③基于理论得出正则化、域随机化有利、环境漂移不利的可验证假设。

**🔧 技术方法**

使用了深度 Q‑网络（DQN）做实验；理论中使用了 Wasserstein 距离、Rademacher 复杂度、Lipschitz 连续性假设及 KL 正则化。

**📊 数据集**

实验数据集采用 OpenAI Gym 的 Taxi‑v3 与 Cliff Walking 两个离散环境，并通过在训练阶段随机化动作来模拟环境漂移。

**📈 对比分析**

对比方法包括：vanilla DQN、加层归一化、L2 正则化、权重归一化、域随机化。实验显示正则化与域随机化显著降低理性风险间隙，环境漂移水平提升时理性风险间隙呈正相关，整体性能与理论预期一致。

**⚠️ 局限性**

局限性包括：①仅关注局部即时理性，未考虑全局长期规划；②实验仅在两个简单离散环境验证，缺乏对复杂连续空间或真实机器人任务的评估；③理论假设（如 Lipschitz 连续性、离散动作空间）对实际复杂系统的适用性尚需进一步验证。

---

## 483. Improved Dimension Dependence for Bandit Convex Optimization with Gradient Variations

**arXiv ID:** 2602.04761 | [PDF](https://arxiv.org/pdf/2602.04761v1)

**作者:** Hang Yu `[一作]` (National Key Laboratory for Novel Software Technology), Peng Zhao `[通讯]` (National Key Laboratory for Novel Software Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在两点带噪声凸优化（two‑point BCO）中研究梯度变异（gradient‑variation）下的在线学习，提出了改进的非连续梯度变异分析，显著降低维度依赖；同时给出了梯度方差、极小损失、单点带噪声线性优化、动态/通用规约以及双边博弈中的相关结果。

**💡 创新点**

创新点主要包括：
- 对非连续梯度变异的精细上界，降低了维度因子；
- 推导了两点 BCO 下的梯度方差与极小损失的最优或近最优性能；
- 首次给出单点 BCO 的梯度变异界；
- 将方法推广至动态、通用规约以及双边博弈，获得第一批相应的梯度变异界。

**🔧 技术方法**

采用了：
- 逼近式梯度估计与自适应优化的结合（类似于 Chiang 等人的方法，但对非连续采样间隔进行更细致的概率与矩分析）；
- 优化的学习率调度（确定性 ∞/(λt) 或自适应 η_t = R/√{...}）；
- 对非连续采样间隔 ρ_{t,i} 的最大值期望上界（O(d log d))；
- 通过对梯度差分、梯度方差、极小损失等量的细分，得到多种问题依赖的 regret 上界。

**📊 数据集**

无实验数据集，本文全部为理论分析与证明。

**📈 对比分析**

与现有最优结果相比：
- 对凸函数从 O(d^{3}√{V_T}) 降至 O(d^{3/2}√{V_T})，维度因子提升近 √d；
- 对 λ‑强凸函数从 O(d^{2}/λ·log(dV_T)) 降至 O(d/λ·log(dV_T))，维度因子提升 d；
- 梯度方差、极小损失结果均达到或接近已知最优 √(dT) 上界；
- 单点 BCO 的梯度变异界为 O(d^{7/2}√{V_T})，此前不存在类似结果；
- 在动态、通用规约与博弈场景下，首次给出相应梯度变异性能，弥补了以往仅有 Ω(√(dT)) 或 Ω(d/λ log T) 的粗略界。

**⚠️ 局限性**

限制与开放问题：
- 仍未解决单点 BCO 的更高维度、非矩形域的梯度变异界；
- 对动态/通用规约的维度因子在最坏情况下仍包含额外对数项；
- 该方法主要针对光滑、L‑光滑且域可投影的情形，非光滑或非凸情形未知；
- 实际实现需解决估计器的可计算性（如需要二分搜索求解），在高维场景可能产生计算开销。

---

## 484. Rethinking the Trust Region in LLM Reinforcement Learning

**arXiv ID:** 2602.04879 | [PDF](https://arxiv.org/pdf/2602.04879v1)

**作者:** Penghui Qi `[一作]` (National University of Singapore), Wee Sun Lee `[通讯]` (National University of Singapore)

**通讯引用:** 5716 | [OpenAlex ID](https://openalex.org/A5071864357)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于分布距离约束的 RL 微调框架 DPPO，重新设计 LLM 微调的 Trust Region 并实现可扩展的 Binary/Top‑K 近似。

**💡 创新点**

创新点在于将 PPO 的比率裁剪替换为直接约束 KL/TV 距离，解决低概率词被过度裁剪、高概率词被低估的问题，并给出低开销的近似方法。

**🔧 技术方法**

采用 Policy Divergence 约束、KL/TV 距离计算、Binary 与 Top‑K 近似、掩码策略和 PPO 的 surrogate objective，结合 TRPO 理论与一阶优化。

**📊 数据集**

在公开数据集 MATH、DAPO、AIME 以及多种 LLM（Qwen3、Llama、MoE 等）上进行实验验证。

**📈 对比分析**

与 GRPO、CISPO、MiniRL 等基线相比，DPPO 在训练稳定性、收敛速度和最终奖励上均表现显著提升，且可在不使用 R3 的情况下获得更优性能。

**⚠️ 局限性**

局限在于需手动设定 divergence 阈值、近似方法在极端长尾词汇上可能不足，以及在极大模型规模下仍存在一定的内存与计算开销。

---

## 485. When LLaVA Meets Objects: Token Composition for Vision-Language-Models

**arXiv ID:** 2602.04864 | [PDF](https://arxiv.org/pdf/2602.04864v1)

**作者:** Soumya Jahagirdar `[一作]` (University of Tuebingen), Hilde Kuehne `[通讯]` (University of Tuebingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Mask‑LLaVA框架，融合CLS全局标记、平均池化后的局部patch标记以及基于掩膜的对象级标记，构建多层次视觉表示；

**💡 创新点**

创新点在于：①将三种不同粒度的视觉标记拼接并统一缩放，实现视觉标记压缩的同时保持信息完整；②支持在推理时灵活裁剪对象掩膜标记，无需再训练，显著降低推理成本；

**🔧 技术方法**

采用ViT视觉编码器、SAM+DeTR生成对象掩膜、Mask‑inversion学习对象标记、线性或MLP投影对齐视觉与文本空间；

**📊 数据集**

使用VQAv2、GQA、VizWiz、ScienceQA‑IMG、POPE、MME、MMBench、MM‑Vet共八个公开多模态基准；

**📈 对比分析**

与FastV、FitPrune、SparseVLM、FasterVLM、LLaVA‑Mini等高效VLM进行对比，Mask‑LLaVA在token压缩率≥90%时在四个基准（GQA、POPE、MME、MMBench）实现最优，其他基准亦获得第二名或优于同类方法；

**⚠️ 局限性**

局限性包括：需要先验的对象检测与分割步骤，计算开销仍高；对逻辑AND等复杂推理支持不足；在极低token数（≤15）时性能仍受限；

---

## 486. Capacity Bounds on Doppler OFDM Channels

**arXiv ID:** 2602.04862 | [PDF](https://arxiv.org/pdf/2602.04862v1)

**作者:** Pablo Orellana `[一作]` (Orange Innovation), Shlomo Shamai `[通讯]` (Technion)

**通讯引用:** 41706 | [OpenAlex ID](https://openalex.org/A5025989795)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究低轨卫星OFDM信道的容量上界与下界，并提出一种基于子空间对齐的叠加编码方案

**💡 创新点**

将残余Doppler误差建模为标量随机不确定性，利用对齐策略消除其影响，实现接近容量的低复杂度解码

**🔧 技术方法**

信息理论容量分析、子空间对齐、最优/最差解码、拉普拉斯族上界、LMMSE、近似高斯输入

**📊 数据集**

使用3GPP NTN–TDL‑A多径模型生成的LEO卫星OFDM信道（1024子载波）

**📈 对比分析**

与最优高斯输入下的理论上界和传统LMMSE接收机对比，结果表明新方案在中高SNR下可达到(N-1)logP+O(1)的自由度，性能接近上界

**⚠️ 局限性**

对高阶误差项O(1)的精确度不够，且对低SNR/高不确定性情形的鲁棒性尚未充分验证

---

## 487. Decomposed Prompting Does Not Fix Knowledge Gaps, But Helps Models Say "I Don't Know"

**arXiv ID:** 2602.04853 | [PDF](https://arxiv.org/pdf/2602.04853v1)

**作者:** Dhruv Madhwal `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**通讯引用:** 1939 | [OpenAlex ID](https://openalex.org/A5100748239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究闭源多跳问答中分解式提示的可靠性，并提出基于提示不一致的训练无关拒绝策略（DBA），用于错误检测。

**💡 创新点**

创新点在于把直接、助力与递增三种等价提示方案的交叉一致性作为可靠性信号，利用它们的分歧实现无训练的“我不知道”策略。

**🔧 技术方法**

采用分解式提示、LLM‑as‑judge 的语义等价判定、交叉提示一致性检测以及基于不一致触发的拒绝机制。

**📊 数据集**

在六个多跳问答数据集（Bamboogle、FRAMES、MuSiQue、CRAG、HotpotQA、Mintaka）上进行评估。

**📈 对比分析**

与AYS、IC‑IDK、Self‑Consistency 等不确定性基线对比，DBA 在 9 大模型、6 数据集上均在 F1 与 AUROC 上优于基线，特别是在前沿模型高基准准确度场景中表现突出。

**⚠️ 局限性**

局限性包括：只能检测跨提示不一致的错误；需要预先提供高质量的 DSL 拆解；额外提示调用导致计算和延迟成本上升。

---

## 488. Laminating Representation Autoencoders for Efficient Diffusion

**arXiv ID:** 2602.04873 | [PDF](https://arxiv.org/pdf/2602.04873v1)

**作者:** Ramón Calvo-González `[一作]` (University of Geneva), François Fleuret `[通讯]` (University of Geneva)

**通讯引用:** 8367 | [OpenAlex ID](https://openalex.org/A5076094010)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了FlatDINO，一种变分自编码器，将DINOv2的二维稠密补丁嵌入压缩成32个一维连续token，便于在压缩语义空间上进行扩散生成。

**💡 创新点**

创新点在于：①将高维自监督特征压缩成一维序列，减少8×token长度、48×维度；②证明压缩后仍能保持语义结构并兼容扩散模型；③通过时间平移和有限区间CFG提升生成质量。

**🔧 技术方法**

使用的技术包括Vision Transformer编码器/解码器、β‑VAE正则化、流匹配生成模型（LightningDiT）、时间平移、classifier‑free guidance。

**📊 数据集**

实验基于ImageNet‑1k 256×256数据集，采用预训练的DINOv2特征作为输入。

**📈 对比分析**

与R‑AE、DiT等基线对比，FlatDINO在gFID上达到1.85（CFG）/3.34（无CFG），与未压缩DINOv2特征的扩散模型相近，同时前向推理FLOPs降低约8×，训练步骤FLOPs降低4.5×。

**⚠️ 局限性**

局限性：模型尚未收敛、训练时长不足；压缩后对全局信息捕获略弱（如16-token模型质量急剧下降）；缺乏针对压缩语义潜在空间优化的扩散采样策略。

---

## 489. CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation

**arXiv ID:** 2602.04856 | [PDF](https://arxiv.org/pdf/2602.04856v1)

**作者:** Zhao Tong `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Xiao-Yu Zhang `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在生成假新闻时Chain‑of‑Thought（CoT）推理过程的安全性，发现即使最终输出拒绝，内部推理链仍有80%存在潜在风险，并提出统一的安全分析框架；

**💡 创新点**

创新点在于将CoT安全性从层级到注意力头的细粒度拆解，并通过软max Jacobian 的谱特征（稳定性、几何一致性、能量集中度）三指标精准定位安全关键路由；

**🔧 技术方法**

主要技术包括自定义CoT安全标注、层级表示分离、注意力头 Jacobian 计算、谱特征度量与可视化、对抗性路由扰动实验；

**📊 数据集**

使用自构造的CoT安全数据集，涵盖 Llama‑8B、Qwen‑4B、Qwen‑8B 三模型，结合直接/间接提示与多种新闻写作风格；

**📈 对比分析**

与传统仅关注输出拒绝的安全评估方法相比，实验显示安全关键层/头对安全率影响显著，扰动安全关键头可显著降低安全率，验证框架有效性；

**⚠️ 局限性**

局限在于数据集规模有限，模型选择有限，且对攻击者可能的滥用需进一步研究防御机制。

---

## 490. A-Graph: A Unified Graph Representation for At-Will Simulation across System Stacks

**arXiv ID:** 2602.04847 | [PDF](https://arxiv.org/pdf/2602.04847v1)

**作者:** Daniel Price `[一作]` (University of Central Florida), Di Wu `[通讯]` (University of Central Florida)

**通讯引用:** 13639 | [OpenAlex ID](https://openalex.org/A5100373119)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了统一的 Architecture-Graph（AGraph）和 ArchX 框架，实现跨技术、架构和应用的任意粒度设计空间探索与性能/成本模拟。

**💡 创新点**

创新点在于将应用、软件、架构和电路四层抽象统一到一个加权有向无环图中，并提供基于作用域的指标检索与易用的程序接口，使得可在任意粒度进行自由、可编程、可解释的模拟。

**🔧 技术方法**

采用图遍历、权重加权、性能模型、约束图以及 Python+graph‑tool 实现；结合 EDA 数据库（CMOS、SFQ、CACTI7）和 SPICE 仿真作为后端。

**📊 数据集**

通过对 FFT、GEMM、TNN、FIR、CNN 等多种工作负载在 CMOS、超导、光子等不同工艺下的案例进行验证。

**📈 对比分析**

与完整 EDA 流程、模块数据库生成以及已有模拟器（Charm、Aladdin 等）对比，ArchX 在执行时间上比完整 EDA 快约 10^5 倍、比数据库生成快 10^3 倍，预测误差低于 15%（FFT）、10%（Systolic）、1%（TNN）等。

**⚠️ 局限性**

局限性在于对空间/时间关系的手工维护需要领域经验，且在大规模设计或非传统架构时可能需要额外的模块数据库，导致易用性受限。

---

## 491. CSLib: The Lean Computer Science Library

**arXiv ID:** 2602.04846 | [PDF](https://arxiv.org/pdf/2602.04846v1)

**作者:** Clark Barrett `[一作]` (Stanford University), Sorrachai Yingchareonthawornchai `[通讯]` (Institute for Theoretical Studies, ETH Zürich)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本论文提出并实现了一个名为CSLib的开源框架，旨在在Lean证明助手中系统化地形式化计算机科学的核心概念（如λ-演算、图灵机、算法与数据结构、并发理论等），并构建一个名为Boole的中间语言及其验证基础设施，使得常见的命令式代码可以在Lean中得到形式化验证；

**💡 创新点**

其创新点在于：①创建了类似Mathlib但针对计算机科学的统一形式化库；②将Lean与传统命令式编程语言通过Boole桥接，降低了形式化验证的入门门槛；③提供了可扩展的复杂度分析API与基于SMT的自动化验证工具；④将AI辅助形式化与验证工具纳入框架，形成“飞轮”效应；

**🔧 技术方法**

技术层面主要使用Lean证明助手、Strata框架、Boole中间语言、SMT求解器、Lean的宏与单子API（TimeM）以及自定义验证条件生成器；

**📊 数据集**

论文未使用传统机器学习或实验数据集，而是基于手写的Lean形式化证明和算法实现作为验证材料；

**📈 对比分析**

在性能评估方面，文中仅给出初始实现与SMT工具的集成示例，并未提供系统化的基准测试或与现有工具（如Boogie、Dafny等）的量化对比；

**⚠️ 局限性**

局限性包括：①Boole及其验证生成器尚不成熟，功能受限；②复杂度分析需手动插入tick，缺乏自动化成本模型；③AI生成的证明易出现错误，需人工审查；④在大型系统级验证时仍面临可扩展性与性能挑战。

---

## 492. Are AI Capabilities Increasing Exponentially? A Competing Hypothesis

**arXiv ID:** 2602.04836 | [PDF](https://arxiv.org/pdf/2602.04836v1)

**作者:** Haosen Ge `[一作]` (University of Pennsylvania), Osbert Bastani `[通讯]` (University of Pennsylvania)

**通讯引用:** 2684 | [OpenAlex ID](https://openalex.org/A5029243071)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

讨论了METR报告认为AI能力指数增长的结论，提出一种替代视角，认为近年的快速提升主要由推理能力的突破驱动，基础能力已趋于平稳，整体发展可能出现拐点并趋于平稳或线性增长。

**💡 创新点**

创新点在于将AI能力拆分为基础模型和推理两大组件，并以乘法形式建模；引入Sigmoid、指数和B‑Spline三种链接函数，理论上证明多技术相乘可产生指数增长后平稳的进化曲线；在同一数据集上对不同增长模型进行对比，提供新的解释框架。

**🔧 技术方法**

使用的技术包括：概率回归与贝叶斯参数估计、PyTorch梯度下降拟合Sigmoid曲线、B‑Spline基函数拟合、指数函数建模、MSE评估、对数回归分析等；所有模型均通过最大似然或最小二乘拟合。

**📊 数据集**

数据集为METR公开的评估数据，涵盖HCAST、RE‑Bench和SWAA三大任务集，共170个任务；评估了15个最先进模型的50%模型视界（model horizon）。

**📈 对比分析**

比较方法：在同一METR数据集上，分别使用MSE（对模型视界或对数视界）对三种链接函数以及METR的指数回归进行评估；Sigmoid链接模型在样本内MSE最低，表明相较于METR的指数假设更能解释现有数据；在2026‑07前两种模型预测相近，之后METR指数预测显著更快，提示其可能过度乐观。

**⚠️ 局限性**

局限性：仅进行样本内评估，缺乏外部验证；模型参数多、复杂度高；不同模型使用的损失函数不一致，导致直接比较不完全公平；乘法假设在实际中尚未得到充分验证；只拆分基础和推理能力，未进一步细分至数据、算法、架构等子组件；缺乏更多公开模型和任务的长期跟踪数据来进一步检验预测。

---

## 493. When Code Becomes Abundant: Redefining Software Engineering Around Orchestration and Verification

**arXiv ID:** 2602.04830 | [PDF](https://arxiv.org/pdf/2602.04830v1)

**作者:** Karina Kohl `[一作]` (Institute of Informatics, UFRGS), Luigi Carro `[通讯]` (Institute of Informatics, UFRGS)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文提出在AI自动化与硬件/能耗限制压缩下重新定义软件工程，强调意图表述、架构控制与持续验证三大核心，探讨其对研究、教育与实践的影响；

**💡 创新点**

创新点在于提出“压缩的SDLC”模型，将传统生命周期重组为“Orchestration”（意图表达与约束）与“Verification”（持续评估）两大功能，首次将架构视为控制面并把维护定义为连续验证，形成“accountability collapse”风险框架；

**🔧 技术方法**

主要采用理论与概念性分析，借助大型语言模型（LLM）生成代码、CI/CD流水线、可执行规范与运行时监控等技术来论证框架可行性；

**📊 数据集**

未使用特定数据集，文中通过金融风险模块等案例说明，主要基于假设与经验性讨论；

**📈 对比分析**

论文未进行实验或性能比较，主要通过概念演绎与案例论证阐述，未给出定量指标；

**⚠️ 局限性**

局限在于缺乏实证验证与量化评估、对LLM及自动化工具的依赖假设、未深入行业差异与可操作化实现细节、以及对监管与合规机制的具体化不足；

---

## 494. On Dual Connectivity in 6G Leo Constellations

**arXiv ID:** 2602.04825 | [PDF](https://arxiv.org/pdf/2602.04825v1)

**作者:** Achilles Machumilane `[一作]` (Institute of Information Science and Technologies), Alberto Gotta `[通讯]` (Institute of Information Science and Technologies)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出了利用离散马尔可夫链对在6G LEO星座双连通（DC）环境下，结合包重复（PD）、包拆分（PS）和网络编码（NC）三种技术的平均端到端包丢失率（E2E‑PLR）的数学框架，并通过仿真验证其对不同卫星高度角组合的有效性。

**💡 创新点**

创新点在于：①首次在DC与NTN结合场景下引入离散马尔可夫链建模，并推导出针对PD、PS和NC的闭式丢包概率表达式；②通过对比三种技术在不同冗余率、负载平衡和分配比例下的E2E‑PLR，量化出NC在高延迟/高丢包环境中的优势与代价；③为后续基于机器学习的动态资源分配与冗余策略提供可度量的基准。

**🔧 技术方法**

技术主要包括：离散时间马尔可夫链（DTMC）建模、包重复、包拆分、随机线性网络编码（RLNC）以及基于公式的概率计算。

**📊 数据集**

使用的数据集为ITU给出的2.2 GHz城市地区地空链路参数，用于生成DTMC状态转移矩阵；仿真中采用不同卫星高度角组合（70°‑60°、60°‑45°、70°‑45°）以及多种负载平衡和冗余因子组合。

**📈 对比分析**

通过表格比较，结果显示：NC在相同E2E‑PLR目标下往往需要更低的冗余因子（例如RF≈2.5）即可满足0.005的丢包率，而PD+PS需要更高的冗余（≈1.8）或仅单链路传输；在极端高丢包场景（60°‑45°）NC显著优于PD/PS；但NC的编码/解码复杂度与延迟更高，影响信息速率。

**⚠️ 局限性**

局限性包括：①未考虑编码/解码引入的时延与计算开销；②仅基于单一仿真模型，未验证在真实卫星网络中的表现；③未讨论动态调度与实时决策，缺乏与机器学习方法的直接对比。

---

## 495. Do Developers Read Type Information? An Eye-Tracking Study on TypeScript

**arXiv ID:** 2602.04824 | [PDF](https://arxiv.org/pdf/2602.04824v1)

**作者:** Samuel W. Flint `[一作]` (Dakota State University), Bonita Sharif `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过眼动追踪研究，比较26名学生在使用和不使用 TypeScript 类型注解时，完成代码概括和定位错误两类任务时的阅读行为和任务正确性。

**💡 创新点**

首次将眼动技术应用于 TypeScript 开发者的代码阅读研究，验证类型注解作为“代码内文档”的实际阅读频率及其对工作记忆的影响，填补了先前仅基于问卷或实验室实验的空白。

**🔧 技术方法**

采用 Tobii Spectrum 眼动仪与 iTrace 插件记录眼动数据，利用 IDT 滤波、AOI 定义、ANOVA、χ²、OLS 回归等统计方法分析阅读时间、注视次数、回溯次数等指标。

**📊 数据集**

数据集由 4 个代码片段组成（两个用于代码概括、两个用于错误定位），来源于 GitHub、RosettaCode 及自编例子；每个片段分别以注解与无注解两种形式呈现，形成 22 名有效受试者的眼动与答题记录。

**📈 对比分析**

对照实验显示：类型注解的存在并未显著提升阅读时间、注视次数或错误定位正确率；工作记忆与阅读行为相关性较弱，整体表现与实验组无显著差异；相较于以往仅采用问卷的研究，提供了客观的眼动数据验证。

**⚠️ 局限性**

主要限制包括：受试者以学生为主，难以推广到业界开发者；AOI 较小，可能导致周边视线未被记录；眼动仪对快速扫视的敏感度有限；实验任务规模与复杂度有限，未涵盖更大、真实项目中的代码和多样化错误类型。

---

## 496. Reinforced Attention Learning

**arXiv ID:** 2602.04884 | [PDF](https://arxiv.org/pdf/2602.04884v1)

**作者:** Bangzheng Li `[一作]` (UC Davis), Derek Zhiyuan Cheng `[通讯]` (Google DeepMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种在多模态大模型（MLLM）后训练阶段直接对内部注意力分布进行强化学习的策略，使模型在生成时主动聚焦关键信息；

**💡 创新点**

创新点在于将注意力分布视为可学习的策略，通过优势加权的Jensen‑Shannon散度正则化，直接优化“关注位置”而非传统的下一个token概率；

**🔧 技术方法**

采用优势加权注意力散度损失、GRPO框架、对齐式注意力蒸馏以及在策略梯度基础上的多模态强化学习；

**📊 数据集**

使用Video‑R1数据集进行SFT与RL训练，评估涵盖多种图像VQA（V⁎、MMMU‑Pro、MME等）和长视频VQA（LongVideoBench、NExT‑QA等）；

**📈 对比分析**

与GRPO基线和标准蒸馏方法对比，实验显示在所有8个图像和7个视频VQA基准上均实现显著提升，图像任务最高可达+94.1（MME）分；

**⚠️ 局限性**

局限性包括：仅在视觉问答任务上验证，未探讨对其他任务的迁移；对奖励设计和注意力分布的依赖使模型对数据分布和超参数较敏感；

---

## 497. CoWTracker: Tracking by Warping instead of Correlation

**arXiv ID:** 2602.04877 | [PDF](https://arxiv.org/pdf/2602.04877v1)

**作者:** Zihang Lai `[一作]` (Visual Geometry Group, University of Oxford), Andrea Vedaldi `[通讯]` (Visual Geometry Group, University of Oxford)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于warping的稠密点跟踪器CoWTracker，消除了传统的成本体积，改用迭代warping和Transformer进行特征对齐与轨迹细化。

**💡 创新点**

创新点在于：①用warp操作直接在高分辨率特征上对齐跨帧特征，避免了昂贵的成本体积；②采用空间-时间交替自注意力的Transformer，能够在全局范围内推理对应关系；③将点跟踪与光流统一到同一架构，零样本即可迁移到光流任务。

**🔧 技术方法**

核心技术包括：VGGT特征提取器、DPT高分辨率上采样器、双线性采样的warp操作、逐步迭代更新的Transformer头（交替空间与时间自注意力）以及可视性与置信度的预测头。

**📊 数据集**

训练数据：合成Kubric视频；评估数据集：TAP‑Vid（DAVIS、Kinetics、RGB‑Stacking）、RoboTAP；光流评估：MPI‑Sintel、KITTI‑2015、Spring。

**📈 对比分析**

与AllTracker、PIPs、CoTracker、RAFT、WAFT等方法对比，CoWTracker在TAP‑Vid和RoboTAP上实现了OA、AJ等指标的领先，零样本光流在Sintel和KITTI上均优于最先进的专门模型，表现出色且计算复杂度仅线性依赖于分辨率。

**⚠️ 局限性**

局限性包括：在极端视角变换、长时间全遮挡或强反射下性能下降；高度依赖VGGT骨干，导致对长序列的计算复杂度为二次；迭代细化在5–6步后趋于饱和；仅使用合成数据训练，真实视频的鲁棒性尚待提升。

---

## 498. The Key to State Reduction in Linear Attention: A Rank-based Perspective

**arXiv ID:** 2602.04852 | [PDF](https://arxiv.org/pdf/2602.04852v1)

**作者:** Philipp Nazari `[一作]` (Max Planck Institute for Intelligent Systems), T. Konstantin Rusch `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究线性注意力模型中低秩状态的现象，给出理论分析，并提出后训练的结构化剪枝框架，能够在保持 CUDA 兼容的前提下显著压缩查询/键通道尺寸。

**💡 创新点**

①从秩利用率角度理论证明低秩会放大查询噪声并影响检索误差；②基于秩揭示 QR 的 DRRQR 剪枝算法；③实现轴对齐的结构化剪枝，兼容深度可分离卷积；④在多规模模型与多任务上验证压缩效果。

**🔧 技术方法**

使用线性注意力（DeltaNet/Gated DeltaNet）、稳定秩与秩利用率分析、结构化剪枝（L1、S‑Wanda、梯度敏感、DRRQR）、QR 与强秩揭示 QR、LoRA 微调以及 CUDA 加速实现。

**📊 数据集**

Fineweb‑Edu、Wikitext2、Lambada、零样本常识推理、FDA/SWDE 检索任务，以及用于校准的轻量化数据集。

**📈 对比分析**

与随机剪枝、权重模数、S‑Wanda、梯度敏感等基线比较。1.3B 模型 50% 压缩时 Wikitext perplexity 由 16.8 上升到约 17.4，零样本准确率保持约 58；检索任务在 30% 压缩仍保持竞争力。DRRQR 与 Grad 表现相近、优于其他基线。加速约 1.3–1.6×，显存减少 30–40%。

**⚠️ 局限性**

对回忆密集型任务会有性能下降；裁剪后通常需要微调；对混合软/线性注意力模型的适用性尚未验证；PCA 等非轴对齐剪枝效果不佳；仅适用于轴对齐的结构化剪枝。

---

## 499. PDF-HR: Pose Distance Fields for Humanoid Robots

**arXiv ID:** 2602.04851 | [PDF](https://arxiv.org/pdf/2602.04851v1)

**作者:** Yi Gu `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2049 | [OpenAlex ID](https://openalex.org/A5109900808)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个轻量、可微的姿态距离场先验PDF‑HR，用于评估任意人形机器人姿态的合理性并可直接作为奖励或正则化项插入到优化与强化学习中。

**💡 创新点**

创新点在于用学习得到的连续距离字段取代传统生成模型，既能实时给出姿态合法性分数，又保持可微，便于梯度传播。

**🔧 技术方法**

采用多层感知机学习距离场，并结合Riemannian几何的欧氏梯度、交叉验证正负样本训练以及与强化学习奖励或逆运动学目标的无缝融合。

**📊 数据集**

正样本来自PHUMA、LaFAN1、AMASS等大规模重定向姿态数据集，负样本通过混合采样策略得到。

**📈 对比分析**

在单轨、全轨、风格模仿和姿态重定向四个基准任务上与ADD、AMP、GMR等主流方法对比，PDF‑HR在样本效率、收敛速度和跟踪误差上均实现了显著提升，且多数任务成功率更高。

**⚠️ 局限性**

局限包括偶尔的跟踪精度低于基线、风格跟踪可能出现模式崩溃、重定向速度慢且抖动、训练数据质量不尽完美，需要进一步改进。

---

## 500. LitS: A novel Neighborhood Descriptor for Point Clouds

**arXiv ID:** 2602.04838 | [PDF](https://arxiv.org/pdf/2602.04838v1)

**作者:** Jonatan B. Bastos `[一作]` (Universidade de Santiago de Compostela), Tomás F. Pena `[通讯]` (Universidade de Santiago de Compostela)

**通讯引用:** 3034 | [OpenAlex ID](https://openalex.org/A5084306280)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并定义了新的点云局部描述子LitS，基于邻域点的极坐标和可照射弧长计算方向性信息，并对2D/3D情形给出完整数学描述。

**💡 创新点**

创新点在于将邻域点视作光源，利用可照射圆弧/球帽的概念构造方向性指示函数，并提出累积LitS作为邻域变换，提供比传统几何描述符更丰富的角度分布信息。

**🔧 技术方法**

采用几何光照模型、极坐标投影、SVD/协方差矩阵分解、阈值λ与角度φ控制、以及对角度区间的指示/累计函数实现；在3D中沿切平面极坐标投影计算。

**📊 数据集**

使用了斯坦福Armadillo、Dragon等公开三维扫描数据，以及加利西亚地区航空LiDAR点云作为实验数据。

**📈 对比分析**

与传统手工局部描述符（SHOT、PFH、FFP）及文献中比较实验未给出精确数值，仅通过可视化和统计量（如总变差、零区间长度）展示LitS在边界、角点、线条检测等任务中的优越性，累积LitS在噪声下更稳健。

**⚠️ 局限性**

局限性包括：假设所有照射点的角度阈值φ相同、照射弧均匀；3D累计LitS只能在切平面上定义，难以全局恢复；计算量相对较高，尤其对大邻域和高密度云；对极端噪声和稀疏采样的鲁棒性仍有限。

---

## 501. Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing

**arXiv ID:** 2602.04837 | [PDF](https://arxiv.org/pdf/2602.04837v1)

**作者:** Zhaotian Weng `[一作]` (Institution1), Xin Eric Wang `[通讯]` (Institution2)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了DMLR框架，结合动态视觉注入和潜在推理提升多模态模型的推理质量。

**💡 创新点**

创新点在于动态视觉注入机制与置信度-质量梯度对齐理论，使模型在不需额外训练的情况下显著提升推理可靠性。

**🔧 技术方法**

使用统一提示、潜在推理、动态视觉注入、Pass@k评估以及置信度分析等技术。

**📊 数据集**

实验基于公开的视觉语言数据集，如VQA、GQA等，详细在补充材料中列出。

**📈 对比分析**

通过Pass@k、与训练型推理方法对比，DMLR在多模态推理任务上取得了显著提升，证明了其鲁棒性与效率。

**⚠️ 局限性**

局限性包括对输入提示的敏感性、对复杂视觉场景的处理仍有限，以及动态注入增加的推理时间。

---

## 502. It's not a Lottery, it's a Race: Understanding How Gradient Descent Adapts the Network's Capacity to the Task

**arXiv ID:** 2602.04832 | [PDF](https://arxiv.org/pdf/2602.04832v1)

**作者:** Hannah Pinson `[一作]` `[通讯]` (Eindhoven University of Technology), Hannah Pinson (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过理论分析单隐藏层 ReLU 网络在梯度下降过程中的神经元学习动力学，提出相互对齐、解锁和竞速三条原理，解释了网络如何将理论容量压缩为任务所需的有效容量，并从中推导出“彩票票据”机制。

**💡 创新点**

创新点在于将梯度下降的动态分解为对齐、解锁和竞速三种相互作用，并用角度距离与权重范数的指数关系说明早期训练中某些神经元如何凭借更佳初始方向获得更大权重，形成稀疏子网络。

**🔧 技术方法**

使用的技术主要包括梯度流连续化、ReLU 门控与有效数据集理论、角度余弦相似度分析以及指数增长的解析解。

**📊 数据集**

实验数据集为 CIFAR‑10 二分类任务（将不同原始类别合并成两大类），并使用 250‑神经元单隐藏层网络进行训练。

**📈 对比分析**

与传统剪枝和随机子网络比较时，本文显示早期训练中角度相似度可显著预测最终权重范数，说明可在训练早期识别“中奖”神经元，实验中最终得到的稀疏网络损失仅略高于完整网络。

**⚠️ 局限性**

局限性包括只研究单隐藏层网络和简化的二分类设置，理论假设如门控固定、有效样本集合不重叠等不一定适用于更深更复杂的模型；此外对不同数据分布和大规模网络的推广仍待验证。

---

## 503. From Evaluation to Design: Using Potential Energy Surface Smoothness Metrics to Guide Machine Learning Interatomic Potential Architectures

**arXiv ID:** 2602.04861 | [PDF](https://arxiv.org/pdf/2602.04861v1)

**作者:** Ryan Liu `[一作]` (California Institute of Technology), Aditi S. Krishnapriyan `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 887 | [OpenAlex ID](https://openalex.org/A5049020441)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出并实现了 Bond Smoothness Characterization Test (BSCT)，通过对单键拉伸/压缩的 1D PES 路径进行评估来快速检测机器学习原子势能面（MLIP）的平滑性，并将该指标用于指导模型架构迭代；

**💡 创新点**

创新点在于提出了新的 Force Smoothness Deviation (FSD) 指标作为低成本、可解释的平滑性度量，结合 BSCT 作为“in‑the‑loop”诊断工具，系统性地改进 Transformer‑style MLIP 的 kNN 图构造、可微 Diff‑kNN、温度控制注意力以及可控高斯平滑，从而显著提升了 PES 平滑性与 MD 稳定性；

**🔧 技术方法**

使用的技术包括基于 Transformer 的 MinDScAIP 体系、可微 k‑Nearest‑Neighbor（Diff‑kNN）图构造、温度调节的 scaled‑dot‑product attention、可控高斯平滑特征、正则化（weight decay）等；

**📊 数据集**

实验数据集涵盖了 BSCT‑SPICE（485 分子 100 步拉伸/压缩）、SPICE、MD22、MPTrj、Matbench Discovery、OMol25 等；

**📈 对比分析**

与传统能量/力 MAE、NVE 能量漂移、MD 轨道热力学稳定性、Matbench κ_SRME 等指标进行对比，BSCT 预测的 FSD 与 MD 稳定性高度相关；基于 BSCT 指导的模型在 E/F MAE、MD 热力学突变、能量守恒、κ_SRME 等方面均优于 MACE、GemNet‑T 等基准，且在大规模 60M 参数模型中仍保持高效与准确；

**⚠️ 局限性**

局限性包括 BSCT 仅针对单键拉伸/压缩的 1D 路径，无法覆盖多原子交互或更复杂的化学空间；对极端激发或极大系统尺寸的鲁棒性仍需进一步验证；此外，BSCT 仍需与多维度的物理性评测（如热导率、反应动力学）结合，以形成完整的 MLIP 验证框架。

---

## 504. Protein Autoregressive Modeling via Multiscale Structure Generation

**arXiv ID:** 2602.04883 | [PDF](https://arxiv.org/pdf/2602.04883v1)

**作者:** Yanru Qu `[一作]` (ByteDance Seed), Quanquan Gu `[通讯]` (ByteDance Seed)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了多尺度自回归框架PAR，用于蛋白质主链生成，避免离散化、处理双向依赖，并在推断中实现粗细尺度的逐步细化

**💡 创新点**

创新点在于将自回归模型迁移到连续三维空间，采用多尺度下采样+上采样的框架，并结合噪声上下文学习与计划采样减轻暴露偏差

**🔧 技术方法**

使用非等变注意力Transformer生成尺度条件、流基解码器直接建模Cα坐标、噪声上下文学习（NCL）与计划采样（SS）以及ODE/SDE采样策略

**📊 数据集**

训练与评估主要基于AFDB代表集和PDB子集（21K可设计样本），并在不同长度蛋白（50–250残基）上进行测试

**📈 对比分析**

与基线（框架扩散、多模态蛋白语言模型、流/扩散Cα生成器）对比，在PDB上FPSD得分降至161.0，设计率提升至96.6%，在无条件生成、零样本提示与基序搭建任务中表现均优或相当

**⚠️ 局限性**

局限性包括仍存在暴露偏差导致误差累积、扩展到更大模型和更长蛋白时需要更多训练步骤、目前仅针对Cα，未覆盖全原子或构象动力学等更细粒度任务

---

## 505. Multi-Head LatentMoE and Head Parallel: Communication-Efficient and Deterministic MoE Parallelism

**arXiv ID:** 2602.04870 | [PDF](https://arxiv.org/pdf/2602.04870v1)

**作者:** Chenwei Cui `[一作]` (Arizona State University), Hannah Kerner `[通讯]` (Arizona State University)

**通讯引用:** 857 | [OpenAlex ID](https://openalex.org/A5053180513)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Multi-Head LatentMoE 与 Head Parallel（HP）架构，并实现了在 GPU 集群上高效、确定性的稀疏 Mixture of Experts（MoE）分布式训练方案，显著降低通信量并消除负载不平衡。

**💡 创新点**

创新点包括：
1) 将每个 token 投影为多个子 token，并为每个子 token 设计独立的路由器与专家组（Multi-Head LatentMoE），
2) 把 all‑to‑all 通信迁移到路由前，产生 O(1) 通信量、均衡且确定性的通信模式（Head Parallel），
3) 结合 FlashAttention 思路的 IO‑aware 路由和基于 FlexAttention 的块稀疏注意力实现的 IO‑aware 专家计算，既保持精确性又降低 HBM/IO 开销。

**🔧 技术方法**

所用技术：
- LatentMoE、Multi‑Head MoE、Expert Parallel（EP）、Head Parallel（HP），
- IO‑aware 路由（在线 top‑k、SRAM 计算、bias 处理），
- IO‑aware 专家计算（块稀疏注意力、FlexAttention、gelu 变换），
- 组块化、聚类、分块计算与梯度聚合，
- CUDA/Triton 高效实现。

**📊 数据集**

数据集：
- 10B token 版 FineWeb‑EDU 用于语言建模训练与验证；
- HellaSwag、PiQA、LAMBADA、ARC‑Easy、ARC‑Challenge 等零射任务用于评估模型泛化能力。

**📈 对比分析**

比较方法：
- 与标准 MoE（EP）和 LatentMoE（EP）在相同参数量下对比训练时间、验证 perplexity 及零射准确率；
- 在 2B 与 4B 参数规模下，Multi‑Head LatentMoE HP 在 2B 时提升 1.11× 训练速度，4B 时提升 1.61×；
- 在 4B 下再加倍粒度（G）后，准确率提升至 45.43% 并保持 1.11× 速度；
- 通信量在 k=4 时仅为 EP 的 25%。

**⚠️ 局限性**

局限性：
- 主要适用于超稀疏（大专家数、短专家维度）环境；
- Head Parallel 受限于头数，无法直接扩展到数十或上百 GPU；
- 对于小 k 或密集专家配置，优势可能不明显；
- 需要在 GPU 上实现 IO‑aware 路由与专家计算，工程实现复杂；
- 在高 SRAM 压力下可能出现性能下降。

---

## 506. Subliminal Effects in Your Data: A General Mechanism via Log-Linearity

**arXiv ID:** 2602.04863 | [PDF](https://arxiv.org/pdf/2602.04863v1)

**作者:** Ishaq Aden-Ali `[一作]` (University of California), Nika Haghtalab `[通讯]` (University of California)

**通讯引用:** 720 | [OpenAlex ID](https://openalex.org/A5080772091)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 log‑linearity 机制和 logit‑linear 选择算法，利用偏好数据子集诱导大语言模型获得目标属性。

**💡 创新点**

发现数据子集中的微弱线性相关性可累积产生潜伏效应，并给出通用的子集筛选方法与理论证明。

**🔧 技术方法**

基于低 logit 秩近似的 log‑线性抽象，使用 DPO 微调筛选子集，并在多模型上进行实验验证。

**📊 数据集**

主要使用 AllenAI 的 Preference 数据集（及其子集）进行实验，且对响应做了截断处理。

**📈 对比分析**

通过对比未微调、系统提示与子集微调模型，在动物偏好、语言翻译、恶意人格等任务中的表现，微调模型可逼近系统提示效果，并且跨模型迁移效果显著。

**⚠️ 局限性**

迁移效果随教师–学生模型差异而显著下降，对无语义数据（如随机数）的迁移效果差；实验多基于截断响应，未验证完整响应情况；缺乏对潜在恶意使用的安全评估。

---

## 507. The matrix-vector complexity of $Ax=b$

**arXiv ID:** 2602.04842 | [PDF](https://arxiv.org/pdf/2602.04842v1)

**作者:** Michał Dereziński `[一作]` (University of Michigan), Raphael A. Meyer `[通讯]` (University of California Berkeley)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文建立了矩阵-向量算法（特别是Krylov子空间方法）在近似求解一般线性系统时所需的矩阵-向量乘法的下界。

**💡 创新点**

创新点在于首次明确了两类算法（双侧和单侧）在最坏情况下的复杂性界限，证明了Krylov子空间算法的最优性。

**🔧 技术方法**

使用了矩阵-向量乘法的复杂性分析技术，结合了多项式逼近的不可逼近性理论。

**📊 数据集**

没有使用特定的数据集，而是通过理论分析和复杂性界限的推导来得出结论。

**📈 对比分析**

与现有的Krylov方法进行比较，结果表明，双侧算法在最坏情况下需要Ω(κlog(1/ε))次矩阵-向量乘法，而单侧算法在完美条件下也需要n次乘法，验证了这些算法的最优性。

**⚠️ 局限性**

限制在于虽然理论上得出了复杂性下界，但在实际应用中，某些矩阵可能会导致算法在迭代次数上远低于理论下界，且对特定矩阵的收敛性仍需进一步研究。

---

## 508. Evolving scientific collaboration among EU member states, candidate countries and global partners: 2000-2024

**arXiv ID:** 2602.04871 | [PDF](https://arxiv.org/pdf/2602.04871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 509. PerpetualWonder: Long-Horizon Action-Conditioned 4D Scene Generation

**arXiv ID:** 2602.04876 | [PDF](https://arxiv.org/pdf/2602.04876v1)

**作者:** Jiahao Zhan `[一作]` (Stanford University), Jiajun Wu `[通讯]` (Stanford University)

**通讯引用:** 12175 | [OpenAlex ID](https://openalex.org/A5100621605)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种从单张图像出发、通过用户动作驱动的长时序 4D 场景生成的混合生成模拟器。

**💡 创新点**

创新点包括：1) 统一的视觉‑物理对齐粒子（VPP）表示，双向桥接动力学与外观；2) 多视角递进优化机制，利用多视点监督消除视觉不一致；3) 形成闭环系统，使生成的视觉修正能够回传更新物理状态，从而支持长时序交互。

**🔧 技术方法**

使用的技术包括：基于物理求解器的粗略动力学推演、3D 高斯光散射渲染、视频生成模型（如 GEN3C）进行视觉细化、COLMAP、TSDFusion、SAM2 进行多视角重建与分割、以及自定义的 VPP 与模拟一致性损失。

**📊 数据集**

数据集为 10 个多材质场景（布料、刚体、弹性体、液体、气体、粉末等），每个场景通过 GEN3C 生成 242 视角的密集视频来构建 3D 场景，并用于训练与评估。

**📈 对比分析**

与基准方法（条件视频生成模型 Wan2.2/2.6、Veo3.1、Tora、DaS、GEN3C，以及混合模拟器 WonderPlay/WonderPlay++）进行对比。量化评估表明本方法在相机可控性、3D 一致性和图像质量上均领先，用户研究显示 70–90% 的受访者更倾向于本方法在物理合理性和运动真实性上的表现；在长时序交互中显著降低误差积累。

**⚠️ 局限性**

局限性包括：需要预先构建完整的 3D 场景，依赖高质量的多视角重建；计算开销较大，尤其是多视角递进优化和长时间物理仿真；对极其复杂或高频率微观动力学的捕捉仍有限。

---

## 510. Contrastive Continual Learning for Model Adaptability in Internet of Things

**arXiv ID:** 2602.04881 | [PDF](https://arxiv.org/pdf/2602.04881v1)

**作者:** Ajesh Koyatan Chathoth `[一作]` `[通讯]`, Ajesh Koyatan Chathoth

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了在物联网场景下，将持续学习与对比学习结合的方法，并提出了统一的问题表述、目标函数、参考体系结构和评估指标。

**💡 创新点**

首次将对比持续学习与IoT的资源限制、隐私、漂移等实际约束关联，形成完整的设计框架与挑战清单。

**🔧 技术方法**

对比学习（InfoNCE、SupCon）、持续学习技术（replay、distillation、正则化、原型、联邦学习）以及TinyML边缘架构与协议。

**📊 数据集**

未给出专门实验数据集，本文假设可使用公开IoT时序/表格数据以及常见基准（如MNIST、CIFAR、UCI等）进行验证。

**📈 对比分析**

未开展实验比较，仅提出了评估框架与指标（平均准确率、遗忘、前向迁移、资源消耗等）。

**⚠️ 局限性**

存在数据增强设计难度、漂移检测与适配、联邦持续学习的不稳定性、能耗与安全更新等限制。

---

## 511. Capturing Visual Environment Structure Correlates with Control Performance

**arXiv ID:** 2602.04880 | [PDF](https://arxiv.org/pdf/2602.04880v1)

**作者:** Jiahua Dong `[一作]` (University of Illinois Urbana-Champaign), Yu-Xiong Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 6693 | [OpenAlex ID](https://openalex.org/A5102952938)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出基于模拟环境下完整状态回归的代理指标，用以评估视觉编码器在机器人操作任务中的表现，并验证该指标与下游策略成功率之间的高度相关性。

**💡 创新点**

创新点在于利用可自动获取的完整环境状态进行回归，构建了一个低成本、环境无关的视觉表示评价方法，并证明其优于现有分割、深度等单一维度的代理指标，同时展示了对真实世界任务的良好迁移。

**🔧 技术方法**

技术上结合了视觉 backbone（ResNet、ViT、CLIP、DINO、R3M 等）与 RoI 均值池化的状态预测头，使用交叉熵与 L2 损失进行回归；对策略学习采用扩散策略，并用 MMRV 与 Pearson 相关系数评估代理与实际策略性能的关联。

**📊 数据集**

使用了 MetaWorld、RoboCasa、SimplerEnv 三大模拟环境以及 WidowX 真实机器人实验，数据量为每个任务 50 条演示和 100 次 roll‑out，覆盖多样化的对象、场景与动力学。

**📈 对比分析**

通过将代理分数与策略成功率的 MMRV（越低越好）和 Pearson r（越高越好）进行对比，实验显示状态预测代理在所有环境中均获得最低 MMRV 与最高 r，甚至超过了分割、深度、少样本等更为昂贵的基线，并在真实世界任务中保持了相似的相关性。

**⚠️ 局限性**

局限性包括：需要模拟环境提供的完整状态标签，若无此标签无法直接使用；代理只关注单帧视觉信息，未能考虑时序动态；以及在某些极端环境或大规模多对象情境下，对象级别提示和维度分解的效能仍有提升空间。

---

## 512. CRoSS: A Continual Robotic Simulation Suite for Scalable Reinforcement Learning with High Task Diversity and Realistic Physics Simulation

**arXiv ID:** 2602.04868 | [PDF](https://arxiv.org/pdf/2602.04868v1)

**作者:** Yannick Denker `[一作]` (Fulda University of Applied Sciences), Alexander Gepperth `[通讯]` (Fulda University of Applied Sciences)

**通讯引用:** 1380 | [OpenAlex ID](https://openalex.org/A5030810660)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向连续强化学习的机器人仿真基准套件CRoSS，基于Gazebo模拟两种机器人平台，支持大量可扩展任务；

**💡 创新点**

创新点在于：①提供高真实感的机器人仿真与丰富任务多样性；②支持多控制模式（笛卡尔、关节空间）和多传感器；③提供基于运动学的轻量级版本；④通过容器化实现一键安装与可复现；

**🔧 技术方法**

采用Gazebo、ROS、Gazebo-Transport、中间件桥接，使用深度Q网络和REINFORCE等标准RL算法作为基线；

**📊 数据集**

使用自定义仿真任务集：两轮跟线、推物体、笛卡尔到达、高低层关节到达等，任务数从几十到上百；

**📈 对比分析**

通过在各任务上分别训练与测试，比较不同设置（DS/SS/SSS）以及不同经验回放大小的效果，结果显示所有标准RL方法在序贯学习中均显著出现灾难性遗忘，表明基准能揭示连续学习难点；

**⚠️ 局限性**

局限包括：仅使用传统RL基线，未涵盖更先进算法；任务多样性虽大但仍集中在特定机器人平台；实验主要在仿真环境，真实转移性能需进一步验证。

---

## 513. Toward Reliable and Explainable Nail Disease Classification: Leveraging Adversarial Training and Grad-CAM Visualization

**arXiv ID:** 2602.04820 | [PDF](https://arxiv.org/pdf/2602.04820v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 514. Fluid Representations in Reasoning Models

**arXiv ID:** 2602.04843 | [PDF](https://arxiv.org/pdf/2602.04843v1)

**作者:** Dmitrii Kharlapenko `[一作]` (ETH Zurich), Zhijing Jin `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了 QwQ-32B 在 Mystery BlocksWorld 中的长链推理过程，研究其内部表示如何随推理逐步完善并实现抽象结构推理。

**💡 创新点**

提出并验证了“流动推理表示”（Fluid Reasoning Representations）的概念，证明推理模型能在推理过程中动态构建抽象表征，并通过 steering 实验证明其因果影响。

**🔧 技术方法**

使用内部激活向量抽取、中心化/交叉命名表示、PCA 与相似度分析，以及正向/负向 steering 与符号 patching 实验。

**📊 数据集**

使用 15 种语义混淆的 Mystery BlocksWorld 变体，共 300 个四块块面具问题作为数据集。

**📈 对比分析**

与基线 LLM 及其基础模型比较；QwQ-32B 在标准 BlocksWorld 96% 但在 Mystery 约 33%；通过正向 steering 可提升至约 43%，负向 steering 降低 2–3%。

**⚠️ 局限性**

仅研究单一推理模型与单一领域，实验规模有限，后期层次动态仍未完全解释。

---

## 515. Vivifying LIME: Visual Interactive Testbed for LIME Analysis

**arXiv ID:** 2602.04841 | [PDF](https://arxiv.org/pdf/2602.04841v1)

**作者:** Jeongmin Rhee `[一作]` (Hankuk University of Foreign Studies), Bohyoung Kim `[通讯]` (Hankuk University of Foreign Studies)

**通讯引用:** 2013 | [OpenAlex ID](https://openalex.org/A5042323667)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了LIMEVis交互可视化工具，用于一次性分析多张图片的LIME结果，并支持用户手动选择并遮蔽超像素来实时观察对模型预测的影响。

**💡 创新点**

创新点在于将原本单图的LIME解释批量化呈现，利用维度降维聚类帮助用户快速发现相似影响因素；同时加入手动超像素遮蔽交互，让用户可对特定像素的作用做精细化探究，突破了LIME只能说明重要像素而无法量化单个像素影响的局限。

**🔧 技术方法**

使用技术包括：LIME（局部可解释模型）、VGG16（图像分类模型及预训练特征提取器）、PacMAP（非线性降维算法）、三种超像素分割算法、前端交互框架（如React+可视化库）等。

**📊 数据集**

采用STL-10数据集（10类、96×96像素），以狗类为示例对100张图进行批量可视化。

**📈 对比分析**

通过示例对比，手动遮蔽对模型预测概率的影响从错误分类（狗0.18，猫0.74）提升到正确分类（狗0.90，猫0.09），说明交互方式能显著改进预测解读；虽然未给出系统性量化基准，但案例展示了交互式可视化相较传统单图LIME在解释性和可操作性上的提升。

**⚠️ 局限性**

局限性包括：仅针对图像分类模型（VGG16）验证，缺乏对其他模型或大规模多类别情形的适用性；超像素遮蔽需要人工操作，缺乏自动化或可量化评估；缺少标准化的定量实验与对比研究。

---

## 516. Safe Urban Traffic Control via Uncertainty-Aware Conformal Prediction and World-Model Reinforcement Learning

**arXiv ID:** 2602.04821 | [PDF](https://arxiv.org/pdf/2602.04821v1)

**作者:** Joydeep Chandra `[一作]` (Tsinghua University), Yong Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 47746 | [OpenAlex ID](https://openalex.org/A5007650371)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了统一框架STREAM‑RL，用于交通预测、异常检测和安全控制，结合不确定性感知与理论保障；

**💡 创新点**

创新点在于：1）PU‑GAT+通过对比邻居的不确定性实现置信单调注意力；2）CRFN‑BY利用变换流与Benjamini‑Yekutieli方法构造依赖鲁棒的p值；3）LyCon‑WRL+通过谱归一化得到可验证的Lipschitz常数，从而实现Lyapunov安全保证；

**🔧 技术方法**

技术包括：图神经网络（PU‑GAT+），变换流（CRFN），自适应分层分布式预测，分布式合成校准（conformal），Benjamini‑Yekutieli FDR控制，世界模型强化学习与Lyapunov安全约束，谱归一化等；

**📊 数据集**

使用四个真实轨迹数据集（T‑Drive、GeoLife、Porto、Manhattan）以及SUMO 4×4格网仿真；

**📈 对比分析**

与STGCN、Graph WaveNet、AGCRN、STACI、RIPCN等预测基线以及Isolation Forest、OmniAnomaly、USAD、CRFN+BH等异常检测基线，RL基线包括PPO、CPO、LAMBDA等。STREAM‑RL在预测覆盖率91.4%、覆盖效率2.13、FDR 4.1%（BY控制）以及RL安全率95.2%和更高奖励方面均优于基线；

**⚠️ 局限性**

局限性包括：BY方法保守导致检验功效下降；仿真规模仅为4×4网络，缺乏大规模验证；依赖于预先构造的基础不确定性模型，可能受限于模型表达能力。

---

