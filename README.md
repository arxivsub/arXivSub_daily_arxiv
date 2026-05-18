# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-18 | 今日论文总数: 538

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. AgentStop: Terminating Local AI Agents Early to Save Energy in Consumer Devices

**arXiv ID:** 2605.15206 | [PDF](https://arxiv.org/pdf/2605.15206v1)

**作者:** Dzung Pham `[一作]` (University of Massachusetts Amherst), Hamed Haddadi `[通讯]` (Brave Software)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本研究中，提出并实现了一种轻量级的效率监督器 AgentStop，用于在本地运行的大型语言模型驱动的代理中预测并提前终止不太可能成功的执行路径，从而降低能源和计算开销。

**💡 创新点**

创新点是将 token 级 logprobs 等已有推理信号作为特征，训练 GBDT 模型进行早期终止预测，首次在本地代理任务中实现了 15-20% 能源节约且任务实用率下降不足 5%。

**🔧 技术方法**

采用的技术包括基于梯度提升决策树的监督学习、token 级 logprob 取值、交叉验证与超参数搜索，以及在 macOS M1 Max 上使用 PowerMetrics 进行功耗测量。

**📊 数据集**

使用的数据集包括 SimpleQA、FRAMES（问答）以及 SWE‑Bench Verified（代码修复）等。

**📈 对比分析**

与默认执行、随机停止、最小/平均 logprob 等基线相比，AgentStop 在 Qwen3‑30B‑A3B 代理上可在前 4-5 步实现约 20‑25% 的能源浪费减少，任务成功率下降不到 5%。

**⚠️ 局限性**

局限性在于模型对不同代理策略、任务类型的泛化性不充分，误判可能导致成功运行被提前终止；且在多模态或多代理场景下的表现尚未验证。

---

## 2. GenAI-Driven Approach to RISC-V Supply Chain Exploration

**arXiv ID:** 2605.15223 | [PDF](https://arxiv.org/pdf/2605.15223v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 3. Amortized Energy-Based Bayesian Inference

**arXiv ID:** 2605.15407 | [PDF](https://arxiv.org/pdf/2605.15407v1)

**作者:** Hojjat Kaveh `[一作]` (California Institute of Technology), Andrew M. Stuart `[通讯]` (California Institute of Technology)

**通讯引用:** 19774 | [OpenAlex ID](https://openalex.org/A5032114852)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过学习观测依赖的传输映射，以能量距离为目标实现贝叶斯逆问题的摊销推断。

**💡 创新点**

创新点在于：(a) 用平均能量距离替代KL散度，消除对逆映射和雅可比矩阵的需求；(b) 在无限维空间中采用Cameron–Martin空间正则化的神经算子参数化，保证后验与先验的绝对连续性。

**🔧 技术方法**

采用能量距离目标、条件传输映射、神经算子（FNO）、Fourier变换、Cameron–Martin正则化以及无似然的样本训练。

**📊 数据集**

使用仿真生成的联合样本，分别在有限维非线性问题、Darcy流逆问题和一维波动方程逆问题上进行实验。

**📈 对比分析**

与传统 pCN MCMC 及不含C1/2的基线传输映射对比，评估后验均值、方差及KL系数的分布误差，摊销方法在多观测情形下大幅加速且保持与 pCN 相近的后验质量。

**⚠️ 局限性**

局限在于：需要先验生成的联合样本和大量前向求解的昂贵前期成本；在高度非线性或高噪声场景下训练收敛可能慢；在极大规模高维参数时仍面临计算与内存瓶颈。

---

## 4. Learning to Persuade a Biased Receiver

**arXiv ID:** 2605.15331 | [PDF](https://arxiv.org/pdf/2605.15331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 5. Hardware-Software Co-Design of Scalable, Energy-Efficient Analog Recurrent Computations

**arXiv ID:** 2605.15216 | [PDF](https://arxiv.org/pdf/2605.15216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 6. NIMO Controller: a self-driving laboratory orchestrator based on the Model Context Protocol

**arXiv ID:** 2605.15227 | [PDF](https://arxiv.org/pdf/2605.15227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 7. Does Theory of Mind Improvement Really Benefit Human-AI Interactions? Empirical Findings from Interactive Evaluations

**arXiv ID:** 2605.15205 | [PDF](https://arxiv.org/pdf/2605.15205v1)

**作者:** Nanxu Gong `[一作]` (Arizona State University), Xing Xie `[通讯]` (Microsoft Research Asia)

**通讯引用:** 46256 | [OpenAlex ID](https://openalex.org/A5044651577)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于交互式评估范式，探究大语言模型（LLM）在与人类交互（HAI）中的心智理论（ToM）提升技术的实际效果，并通过模拟与用户实验验证。

**💡 创新点**

创新点在于将ToM评估从传统的第三人称、静态选择题迁移到第一人称、多轮、开放式对话环境，区分目标导向与体验导向任务，并揭示现有ToM提升方法在真实交互中表现不佳的“性能鸿沟”。

**🔧 技术方法**

使用了四种代表性ToM提升技术：Prompt‑Based的Foresee‑and‑Reflect (FaR) 与 Perspective‑Taking (PT)，以及Fine‑Tuning的Supervised Fine‑Tuning (SFT) 与 Reinforcement Learning (RL)，并将其改造为适合第一人称交互的形式，应用于 GPT‑4o 与 Llama‑3.1‑8B 两个基线模型。

**📊 数据集**

使用的数据集包括：目标导向任务的 ChatBench 与 CollabLLM；体验导向任务的 MentalChat16K 与 Emotional‑Support‑Conversation；以及基于 Prolific 招募的 100 名用户的三轮对话用户研究。

**📈 对比分析**

对比方法：将增强模型与基线模型在四个评估框架下（交互式指标、传统准确率、任务成功率、情感维度）进行统计检验。结果显示：在目标导向任务中，几乎无显著提升；在体验导向任务中虽然情感与安全维度提升，但出现伦理与整体质量下降；用户研究显示不同方法间差异极小，难以被人类辨别。

**⚠️ 局限性**

局限性：仅选取了四种可改造的ToM提升技术，未覆盖更广泛方法；交互评估仅覆盖九个场景，缺乏更丰富的真实任务；改造过程仍受限于原始方法的第三人称设计，可能导致对动态多样需求的适应性不足。

---

## 8. ELDOR: A Dataset and Benchmark for Illegal Gold Mining in the Amazon Rainforest

**arXiv ID:** 2605.15397 | [PDF](https://arxiv.org/pdf/2605.15397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 9. Diffusion Policy for Coordinated Control of a Nonholonomic Mobile Base and Dual Arms in Door Opening and Passing

**arXiv ID:** 2605.15352 | [PDF](https://arxiv.org/pdf/2605.15352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 10. $f$-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data

**arXiv ID:** 2605.15417 | [PDF](https://arxiv.org/pdf/2605.15417v1)

**作者:** Jake Fawkes `[一作]` (University College London), Jason Hartford `[通讯]` (Valence Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一族基于f-散度的翻译不变损失（f-轨迹平衡、DevGrad、温度化损失），可用于GFlowNets、LLM异步微调和扩散模型等生成任务，实现在线与离线双向训练；

**💡 创新点**

创新点在于把任意f-散度与翻译不变损失映射起来，保证在策略样本上梯度与对应f-散度梯度一致，同时保持离线训练的全局最优性，从而统一并扩展了Vargrad和Trajectory Balance；

**🔧 技术方法**

核心技术包括：f-散度理论、翻译不变损失构造、Vargrad/DevGrad、温度化损失、α-散度可调模式覆盖/寻优、梯度对齐与批量归一化；

**📊 数据集**

实验数据集涵盖：2D网格（Hypergrid）用于模式覆盖测试、SynFlowNet分子生成任务、MNIST扩散模型的奇偶数字分类、GSM8k与Hendrycks MATH用于LLM异步微调；

**📈 对比分析**

与传统Trajectory Balance、PPO、GRPO等方法比较，实验显示：在GFlowNet上α=0.75能更快覆盖所有四个模式，α=1.2收敛更快但模式崩溃；在LLM异步微调中，Forward KL、Pearson、Jensen‑Shannon等f-散度能在奖励与熵之间取得更佳平衡；在扩散模型中，α-散度更均匀地覆盖奇偶数字模式；整体性能优于标准方法；

**⚠️ 局限性**

局限性包括：需要估计归一化常数或采用VarGrad批量估计，数值稳定性受β、温度化参数影响；在极高维或大模型场景下梯度仍可能偏大；实验仅覆盖部分任务，未验证在更复杂生成空间的泛化。

---

## 11. ICRL: Learning to Internalize Self-Critique with Reinforcement Learning

**arXiv ID:** 2605.15224 | [PDF](https://arxiv.org/pdf/2605.15224v1)

**作者:** Jianbo Lin `[一作]` (Hong Kong University of Science and Technology), Hui Xiong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 45753 | [OpenAlex ID](https://openalex.org/A5101862104)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于强化学习的框架ICRL，联合训练求解器和批评者，使得批评信息在无批评条件下被内化。

**💡 创新点**

核心创新包括分布校准的加权比例（token-level re-weighting）来纠正批评条件与无批评条件的分布偏移，以及角色基准化优势估计（role-wise group advantage）以稳定双角色联合优化。

**🔧 技术方法**

使用GRPO作为RL基元，基于共享backbone的多角色策略，采用重要性采样、分布校准权重、角色优势归一化和梯度裁剪等技术。

**📊 数据集**

在文本世界（ALFWorld）、WebShop、搜索QA和多跳问答（HotpotQA、2WikiMultiHopQA、Bamboogle、MuSiQue）以及数学推理（MATH500、Minerva Math、OlympiadBench、AIME24、AMC23）等数据集上进行评估。

**📈 对比分析**

与提示式基线、单体RL基线（GRPO、GSPO）以及面向代理的训练方法（ScalingInter-RL、MATPO、Critique-GRPO）比较，ICRL在代理任务平均提升约6.4分，在数学任务平均提升约7.0分；8B批评者在性能上可与32B静态批评者相当，同时显著减少token使用。

**⚠️ 局限性**

局限性包括需要较高的计算资源，训练过程中仍可能出现梯度不稳定；分布校准比例的设定依赖经验，可能不适用于所有任务；目前验证范围仅涵盖所述任务，泛化到更复杂或更长时间的任务仍需进一步研究。

---

## 12. Lagrangian Flow Matching: A Least-Action Framework for Principled Path Design

**arXiv ID:** 2605.15419 | [PDF](https://arxiv.org/pdf/2605.15419v1)

**作者:** Shukai Du `[一作]` (Syracuse University), Yiming Li `[通讯]` (Syracuse University)

**通讯引用:** 56887 | [OpenAlex ID](https://openalex.org/A5002087341)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于最小作用量的 Lagrangian flow matching 框架，通过最优作用来确定概率路径和速度场，实现了无仿真训练；

**💡 创新点**

创新点在于将经典力学的最小作用原理引入流匹配，生成新的谐振子轨迹，统一并扩展了 OT‑based 与条件流匹配，并给出了全新的无仿真训练目标；

**🔧 技术方法**

主要技术包括最小作用量原理、Lagrangian 机械学、动态/静态最优传输等价、条件流匹配、迷你批量 OT、闭式谐振子轨迹等；

**📊 数据集**

实验使用的数值数据集包括二维高斯/双月合成分布、单细胞时间序列（embryoid body、CITE‑seq、Multiome）以及 CIFAR‑10 图像；

**📈 对比分析**

与 OT‑CFM、OT‑SI、OT‑Aniso 等基线比较，使用 2‑Wasserstein、NPE、FID、NFE 等指标，谐振子频率 ω=1 在二维任务中显著优于基线，ω≈0 时几乎等价于 OT‑CFM，在 CIFAR‑10 上 FID 轻微提升，整体表现与现有方法竞争；

**⚠️ 局限性**

局限性包括对 Lagrangian 选择的理论解释不足、仅处理确定性轨迹、未探索随机动力学（如 Schrödinger 桥）及自适应潜在空间结构，且较大 ω 时轨迹曲率增大导致采样效率下降。

---

## 13. Autonomous Intelligent Agents for Natural-Language-Driven Web Execution with Integrated Security Assurance

**arXiv ID:** 2605.15281 | [PDF](https://arxiv.org/pdf/2605.15281v1)

**作者:** Vinil Pasupuleti `[一作]` (International Business Machines), Shrey Tyagi `[通讯]` (Salesforce Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套基于大型语言模型的自主测试框架，能够根据自然语言描述自动生成可回放的Web测试脚本，并将安全验证集成到测试流程中。

**💡 创新点**

将五项策略（导航可靠性、选择器精确化、生成后验证、智能等待注入、失败学习）与容器化工作器相结合，实现了从55%到93%的脚本成功率，并首次实现了无需安全专家的自然语言安全测试。

**🔧 技术方法**

使用了Vision‑Enabled LLM（如ChatGPT/Claude）与ReAct思维循环、基于DOM与截图的多模态感知、容器化工作器、服务器无状态协调层、以及安全验证模块；同时采用OCR/视觉识别技术。

**📊 数据集**

在四个生产级Web应用（SPA、SaaS仪表盘、CMS、银行门户）共176个场景上进行评估，并通过手工Selenium脚本对比验证。

**📈 对比分析**

相较于基线（55%）和手工Selenium，成功率提升至93%，导航失败率下降8倍，时延提升显著，测试创建时间缩短75%，安全漏洞检测率分别为85%（认证绕过）和95%（输入校验），误报率低于12%。

**⚠️ 局限性**

受限于LLM成本与延迟、对复杂多步流程、Shadow DOM和异步内容的处理不足、仅能检测浏览器可观察的安全缺陷、易受恶意注入攻击、以及对老旧或自定义DOM布局的适用性不明。

---

## 14. RTL-BenchMT: Dynamic Maintenance of RTL Generation Benchmark Through Agent-Assisted Analysis and Revision

**arXiv ID:** 2605.15537 | [PDF](https://arxiv.org/pdf/2605.15537v1)

**作者:** Jing Wang `[一作]` (Hong Kong University of Science and Technology), Zhiyao Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 909 | [OpenAlex ID](https://openalex.org/A5075696558)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为 RTL-BenchMT 的自动化代理框架，用于动态维护 RTL 生成基准，自动识别并修订有缺陷的案例，并检测与更新过拟合案例。

**💡 创新点**

创新点在于将多任务、多代理协同工作与迭代推理相结合，形成完整的三阶段流程（失败分析、基准修订、过拟合检测），并通过设计描述重写显著提升基准的准确性与多样性。

**🔧 技术方法**

采用大语言模型（LLM）驱动的多代理体系结构；每个代理使用思考-动作-观测循环；通过 Docker 化的 EDA 环境进行 RTL 编译仿真；利用风格模板进行设计描述重写；使用严格规则验证修订结果。

**📊 数据集**

主要使用公开的 RTL 生成基准数据集，包括 VerilogEval、RTLLM、CVDP 以及其子集 cid002、cid003 等；在这些基准上执行 LLM 生成任务并收集失败案例。

**📈 对比分析**

与原始基准对比，修订后 GPT‑4o 的准确率提升约 3.8%，Claude‑3.7 维持稳定；过拟合检测发现 CodeV 在重写描述上下降 9.6%，而 LLaMA 在重写描述上提升 6.1%。整体而言，修订后的基准显著减少误判，提升评测公平性。

**⚠️ 局限性**

局限性包括：仍需人工审核最终修订；代理对特定 LLM 的依赖程度未知；仅针对 RTL 生成任务，可能无法直接推广至其他 EDA 领域；对极大规模或高度动态基准的扩展性尚未充分验证。

---

## 15. Breakeven complexity: A new perspective on neural partial differential equation solvers

**arXiv ID:** 2605.15399 | [PDF](https://arxiv.org/pdf/2605.15399v1)

**作者:** Yijing Zhang `[一作]` (University of Wisconsin--Madison), Mikhail Khodak `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了“breakeven complexity”评估框架，用于衡量神经PDE求解器在完整训练成本下何时比误差匹配的传统求解器更具成本效益。

**💡 创新点**

创新点在于将数据生成、模型训练、推理三部分成本与低精度传统求解器的误差匹配相结合，形成一种计算成本意识的度量，并通过缩放律优化训练预算。

**🔧 技术方法**

使用了深度算子模型（FFNO、EddyFormer、DISCO、DPOT、Poseidon系列、HalfWalrus）以及误差匹配与计算成本的统计方法，配合GPU基准求解器Exponax和PyFR。

**📊 数据集**

采用了APEBench的Navier–Stokes、Kuramoto–Sivashinsky、Gray–Scott三种周期性PDE数据集，并构建了新的多障碍流场数据集BreakFlow，全部在GPU上生成。

**📈 对比分析**

通过在相同训练预算下寻找误差匹配的传统求解器，计算交叉点得到平均与最坏情况的breakeven复杂度；结果显示在困难问题上，神经求解器仅需千级推理调用即可打破平衡，而在简单任务上仍需数十万次调用。

**⚠️ 局限性**

局限性包括对传统求解器实现与硬件平台的依赖、对θ分布均值成本的假设、误差匹配过程的手工调节，以及在多模态或不确定工作负载预测下的适用性挑战。

---

## 16. Beyond Performance Disparities: A Three-Level Audit of Representational Harm in CelebA

**arXiv ID:** 2605.15312 | [PDF](https://arxiv.org/pdf/2605.15312v1)

**作者:** Sieun Park `[一作]` (London School of Economics and Political Science), Yuanmo He `[通讯]` (London School of Economics and Political Science)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5056169471)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对CelebA人脸数据集进行三层级的公平性审计，分析数据集结构、模型特征权重与空间注意力，揭示性别和年龄双重标准如何导致女性被过度审查、老年男性被排除的表征伤害。

**💡 创新点**

首次将代表性伤害的概念从标签层面扩展到特征权重与注意力层面，构建三维审计框架，并展示传统公平度量难以捕捉的两种伤害模式。

**🔧 技术方法**

采用层次聚类与逻辑回归剖析属性共现；使用XGBoost + SHAP解释属性重要性；使用ResNet-18 + Grad‑CAM可视化注意力；并结合准确率与平均精度评估子组性能。

**📊 数据集**

使用CelebA 202,599张明星面部图像及其39个属性标签（不含“性别”）进行实验。

**📈 对比分析**

整体模型达到AUC 0.88、准确率约80%；在子组评估中，青年女性AP 0.95、准确率0.81，老年男性AP仅0.40、准确率0.92，显示传统公平度量（如准确率）掩盖了表征伤害。

**⚠️ 局限性**

局限在于仅采用二元性别标签、缺乏自我认同信息、属性集有限且带有媒体与注释者偏见，且研究聚焦于单一数据集，未能验证结果的普适性。

---

## 17. Controllable Molecular Generative Foundation Models

**arXiv ID:** 2605.15354 | [PDF](https://arxiv.org/pdf/2605.15354v1)

**作者:** Yihan Zhu `[一作]` (University of Notre Dame), Meng Jiang `[通讯]` (University of Notre Dame)

**通讯引用:** 6021 | [OpenAlex ID](https://openalex.org/A5074821819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Controllable Molecular Generative Foundation Models（CMG-FFM），通过将预训练的图扩散模型与基于 motif 的决策空间结合，实现了多任务可控分子逆设计。

**💡 创新点**

创新点在于：① 将 motif 结构编码为离散 Tokenizer，构建高层次的图表示；② 在此空间上进行离散扩散和强化学习，显著降低 atom-level 选取的决策复杂度；③ 采用终端奖励的 RL 对逆扩散策略进行调优，从而实现对多种数值与分类属性的精准控制。

**🔧 技术方法**

技术核心包括：节点对编码（NPE）motif tokenizer；图扩散 Transformer 的预训练、监督微调与 PPO 对齐；KL 正则化的 RL 目标与终端奖励设计；以及对齐任务嵌入的可迁移学习。

**📊 数据集**

使用了两大无标签预训练集（13k 高分子、10k MoleculeNet 分子）和三大下游任务集：① 高分子气体渗透率、DFT 计算属性；② 药物相关任务（FreeSolv、BACE、BBBP）。

**📈 对比分析**

与多种基线（遗传算法、贝叶斯优化、图扩散/流模型、RL 生成器）对比，CMG-FFM 在所有九个控制目标上均名列前茅：MAE/准确率平均降低约27%，可控性提升44–48%，而化学有效率维持在0.94以上，无需规则校验。

**⚠️ 局限性**

局限性包括：① 对 RL 的收敛速度与计算成本较高；② 在某些属性（如 FreeSolv）上提升有限；③ 仍需要大量高质量标签数据进行任务嵌入训练，且对极端稀有结构的生成尚未彻底验证。

---

## 18. From I/O to Code with Discovery Agent

**arXiv ID:** 2605.15334 | [PDF](https://arxiv.org/pdf/2605.15334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 19. DiscoExplorer: An Open Interface for the Study of Multilingual Discourse Relations

**arXiv ID:** 2605.15304 | [PDF](https://arxiv.org/pdf/2605.15304v1)

**作者:** Amir Zeldes `[一作]` (Georgetown University), Amir Zeldes `[通讯]` (Georgetown University)

**通讯引用:** 1446 | [OpenAlex ID](https://openalex.org/A5089212858)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个纯前端、无服务器的浏览器搜索工具 DiscoExplorer，用于查询和可视化多语言 DISRPT 语料库中的语篇关系。

**💡 创新点**

创新点在于：统一 DISRPT 标签体系；提供轻量级、客户端内存搜索；设计简洁的 DEQL 查询语言；支持多维度频率、交叉表和数据集比较，并能直接在浏览器中展示结果。

**🔧 技术方法**

技术实现采用 React + 纯 JavaScript、HTML、CSS；使用客户端内存索引；实现 DEQL 查询语法；通过 D3 之外的自制图表实现频率与比较可视化。

**📊 数据集**

使用了 DISRPT 2025 版的 38 个多语言数据集，涵盖 RST、PDTB、SDRT、eRST 等框架，超过 300,000 条关系，约 5 万万词。

**📈 对比分析**

通过与 ANNIS 的简单查询对比（同一 GUM 数据集）展示响应时间：DiscoExplorer 在加载后所有查询均在 0.02–0.03 s 内完成，远快于 ANNIS（3–5 s），但功能范围更窄。

**⚠️ 局限性**

局限性：未进行用户体验评估；依赖 DISRPT 原始注释质量；不支持 ANNIS 的复杂图/多层搜索；仅评估响应速度，未系统验证功能完整性。

---

## 20. Curriculum Learning of Physics-Informed Neural Networks based on Spatial Correlation

**arXiv ID:** 2605.15254 | [PDF](https://arxiv.org/pdf/2605.15254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 21. From Weight Perturbation to Feature Attribution for Explaining Fully Connected Neural Networks

**arXiv ID:** 2605.15328 | [PDF](https://arxiv.org/pdf/2605.15328v1)

**作者:** Thodoris Lymperopoulos `[一作]` (NCSR Demokritos), Denia Kanellopoulou `[通讯]` (NCSR Demokritos)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于权重扰动的可解释方法 XWP 与 XWPc，用于解释全连接神经网络的决策过程。

**💡 创新点**

创新点在于改用权重而非输入进行扰动，借助网络初始状态作为基线，避免了传统遮挡方法中的分布外样本和偏差问题。

**🔧 技术方法**

采用了权重扰动与遮挡原则、ReLU/Softmax 激活、梯度与传播等技术，并通过计算模型输出差异得到特征重要性。

**📊 数据集**

实验使用了 Typeface MNIST（TMNIST）和 Fashion MNIST（FMNIST）这两种简易图像数据集。

**📈 对比分析**

与 Occlusion、SHAP、RISE、Integrated Gradients、LRP 等主流方法对比，XWP 与 XWPc 在平均下降率和删除 AUC 指标上与 SHAP/IG 旗鼓相当，并在某些指标上略优。

**⚠️ 局限性**

局限性包括仅在相对简单的数据集和全连接网络上验证，缺乏对 CNN、Transformer 等更复杂架构的评估，且基线选择与理论完整性仍待进一步完善。

---

## 22. Tuning-free Instruction-based Video Editing Via Structural Noise Initialization and Guidance

**arXiv ID:** 2605.15533 | [PDF](https://arxiv.org/pdf/2605.15533v1)

**作者:** Song Wu `[一作]` (JIUTIAN Research China Mobile), Junlan Feng `[通讯]` (JIUTIAN Research China Mobile)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无微调、基于指令的视频编辑框架，能够实现对象/属性替换与删除，并通过噪声层次控制实现高质量编辑。

**💡 创新点**

创新点包括：① 结构化噪声初始化策略（SNIS）为编辑区域赋高噪声、未编辑区域赋低噪声；② 噪声引导机制（NGM）在关键 denoising 步骤中利用视频先验和噪声信息保持未编辑内容与整体一致性；③ 基于 VLM 与 LLM 的编辑指令分析模块，自动生成源/目标提示与编辑掩码。

**🔧 技术方法**

使用了 Diffusion 生成模型（CogVideoX‑5B）、VLM（InternVL2.5‑26B）、LLM（Qwen3‑72B）、Grounded‑SAM‑2 语义分割、DDIM 逆向噪声、线性过渡权重以及噪声引导等技术。

**📊 数据集**

采用了自建的基于 DAVIS 的高质量视频编辑指令集，涵盖对象移除、对象替换和属性修改等任务。

**📈 对比分析**

与 Pix2Video、TokenFlow、AnyV2V、MiniMax‑Remover 等 SOTA 进行定量比较，指标包括 CLIP‑T、LPIPS、FVD、CLIP‑I；实验表明在所有指标上均优于对比方法，取得最佳/第二最佳成绩。

**⚠️ 局限性**

局限性包括：对掩码不准确仍会产生视觉瑕疵；像素级评价如 LPIPS 在某些情况下下降但视觉效果仍可接受；推理时间和 GPU 资源需求仍相对较高；仅依赖生成模型的先验，可能不适用于所有编辑场景。

---

## 23. Social-Mamba: Socially-Aware Trajectory Forecasting with State-Space Models

**arXiv ID:** 2605.15424 | [PDF](https://arxiv.org/pdf/2605.15424v1)

**作者:** Po-Chien Luan `[一作]` (EPFL), Alexandre Alahi `[通讯]` (EPFL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Social-Mamba，一个完全基于 Mamba 状态空间模型的社会意识轨迹预测架构，利用 Cycle Mamba 实现连续双向信息流，并通过结构化的 egocentric 网格和社交三元分解有效捕捉密集场景中的社会交互。

**💡 创新点**

创新点包括：① 将 SSM 与社会交互相结合，首次在原生形式下解决 1D 顺序模型与无序 2D 社会场景的结构不匹配；② 设计 Cycle Mamba，实现正向后向状态连续传递，显著降低参数并提升双向融合效果；③ 通过社交三元分解（时序、egocentric、goal-centric）与可学习的门控融合，实现高效、可扩展的多尺度社会推理。

**🔧 技术方法**

核心技术为：Selective State Space Model (Mamba)、Cycle Mamba 块、Ego‑centric 社交网格、社交三元分解、可学习的门控融合、Mamba 解码器；并将该框架嵌入流匹配模型（MoFlow）以进一步提升性能。

**📊 数据集**

使用了五个公开基准数据集：NBA（NBA‑Full、Scoring、Rebounding）、Stanford Drone Dataset (SDD)、JackRabbot Dataset and Benchmark (JRDB)。

**📈 对比分析**

与多种 Transformer‑、GNN‑和 Mamba‑基准（如 Multi‑Transmotion、OmniTraj、Social‑VAE、NMRF 等）对比，Social‑Mamba 在 ADE/FDE 上实现了 state‑of‑the‑art 结果，同时参数量和 GFLOPs 下降约 75% 与 54%，推理时间约 3.4 ms，显著提升计算效率与实用性。

**⚠️ 局限性**

局限性主要在于：① 对极端稀疏或极高密度场景的泛化仍有限；② 仍需在多模态生成的多样性与真实度之间取得更好平衡；③ 对超长预测 horizon 的长期稳定性和对噪声轨迹的鲁棒性尚待进一步研究。

---

## 24. Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution

**arXiv ID:** 2605.15301 | [PDF](https://arxiv.org/pdf/2605.15301v1)

**作者:** Han Li `[一作]` (Nanjing University), Jiaheng Liu `[通讯]` (Nanjing University)

**通讯引用:** 2348 | [OpenAlex ID](https://openalex.org/A5032858379)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Solvita 框架，将竞争编程问题求解拆分为规划、求解、认证和攻击四个专用智能体，并通过可训练的图结构知识网络实现持续学习。

**💡 创新点**

创新点在于将多代理闭环与可训练知识网络结合，将执行判定与对抗测试的信号作为强化学习奖励，使系统在保持 LLM 冻结的前提下逐步积累可迁移的推理经验。

**🔧 技术方法**

采用多代理系统、图结构知识网络、策略网络、REINFORCE 更新、对抗测试与自动化验证等技术。

**📊 数据集**

使用从 Codeforces、AtCoder、Aizu Online Judge、LeetCode、SPOJ 等平台抓取的约 8,000 个竞赛题目做训练与评估，并在 CodeContests、APPS、AetherCode 等公开基准上测试。

**📈 对比分析**

与现有单通道、AlphaCodium、MapCoder 等开源框架及商业代码助手对比，在 pass@1、token 消耗与错误类别上均优于对手，尤其在 GPT‑5.4、Claude Opus 4.6 等大模型上达到近 80% 以上的 pass@1。

**⚠️ 局限性**

局限在于冷启动成本高、对抗范围受限、补丁修复可能导致系统漂移，且知识网络需约 5,000 题训练后才能显著提升效果。

---

## 25. Deep Pre-Alignment for VLMs

**arXiv ID:** 2605.15300 | [PDF](https://arxiv.org/pdf/2605.15300v1)

**作者:** Tianyu Yu `[一作]` (Tsinghua University), Yuan Yao `[通讯]` (Tsinghua University)

**通讯引用:** 5375 | [OpenAlex ID](https://openalex.org/A5000492991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Deep Pre-Alignment (DPA) 架构，用小型 Perceiver VLM 替换 ViT，将视觉特征深度对齐到 LLM 文本空间，减轻 LLM 对齐负担并提升多模态理解

**💡 创新点**

创新点在于通过 Perceiver 的语言块实现视觉特征的深度预对齐，显著降低文本能力遗忘并提升多模态性能

**🔧 技术方法**

技术包括 Perceiver VLM、轻量级投影层、两阶段训练（图像-字幕对齐 + 视觉指令调优）以及在 Qwen3、LLaMA 3.2 等 LLM 上的集成

**📊 数据集**

使用 558K 图像-字幕对齐数据 + 1M 视觉指令数据，并在 11 个多模态基准和 3 个文本基准上评测

**📈 对比分析**

与标准 ViT+LLM 架构和公开 VLM 对比，4B 规模提升 1.9 分，32B 规模提升 3.0 分，文本遗忘下降 32.9%，在多模态与文本基准上均表现优于对照模型

**⚠️ 局限性**

局限性包括对 Perceiver 训练的依赖、仍需额外训练资源、早期融合方案在多轮对话中的适用性有限，以及极大规模 LLM 的可扩展性待进一步验证

---

## 26. Margin-Adaptive Confidence Ranking for Reliable LLM Judgement

**arXiv ID:** 2605.15416 | [PDF](https://arxiv.org/pdf/2605.15416v1)

**作者:** Gaojie Jin `[一作]` (University of Exeter), Tianjin Huang `[通讯]` (University of Exeter)

**通讯引用:** 309 | [OpenAlex ID](https://openalex.org/A5028180352)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种针对大型语言模型（LLM）评判器的自适应置信度学习框架，利用基于边际的排序损失学习置信度估计器，并提供PAC‑Bayesian泛化保证；

**💡 创新点**

创新点在于：①将置信度视为排序函数，采用边际排序损失；②推导出以边际为参数的PAC‑Bayesian泛化上界，揭示边际-复杂度折衷；③基于该上界设计自适应边际训练器；

**🔧 技术方法**

技术包括：多层感知机（MLP）置信度估计器、边际排名损失、PAC‑Bayesian 泛化分析、随机/模拟注释器、软化排序损失、交叉熵替代；

**📊 数据集**

使用四大公共评判数据集（AlpacaEval、Chatbot Arena、HH‑RLHF、TL;DR）以及六个判别器模型（Mistral‑7B、Llama3‑8B、Llama3‑70B、Qwen2.5‑32B、Qwen2.5‑72B、GPT‑OSS‑120B）；

**📈 对比分析**

与预测概率、语义自信、随机注释器、模拟注释器以及无参数学习方法（Vanilla）比较；实验显示自适应置信度在排名损失、AUROC、单调性、保证成功率和覆盖率等指标均优于基线，尤其在Cascaded Selective Evaluation（CSE）框架中成功率提升显著；

**⚠️ 局限性**

局限性包括：仍需大量标注样本以构建模拟注释器；边际参数的自适应选择可能导致训练不稳定；在极端噪声或多任务环境下的泛化仍未完全验证；

---

## 27. MR2-ByteTrack: CNN and Transformer-based Video Object Detection for AI-augmented Embedded Vision Sensor Nodes

**arXiv ID:** 2605.15423 | [PDF](https://arxiv.org/pdf/2605.15423v1)

**作者:** Luca Bompani `[一作]` (University of Bologna), Francesco Conti `[通讯]` (University of Bologna)

**通讯引用:** 51686 | [OpenAlex ID](https://openalex.org/A5025489818)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了MR2-ByteTrack，一种在MCU上实现高效视频目标检测的多分辨率、重新评分ByteTrack流水线，兼容CNN和Transformer检测器。

**💡 创新点**

创新点包括交替使用全分辨率与低分辨率推理、引入轻量级Kalman滤波的ByteTrack跟踪以及基于概率规则的Rescore算法，并设计了适合MCU的EffViT-Det Transformer检测器。

**🔧 技术方法**

采用多分辨率推理、ByteTrack跟踪、Rescore重新评分、Kalman滤波、EfficientViT（线性注意力）、代码覆盖、INT8量化以及GAPflow工具链实现部署。

**📊 数据集**

模型在COCO上训练，使用ImageNetVID（仅保留与COCO重叠的16类）评估，并在GAP9 MCU上进行真实部署测试。

**📈 对比分析**

与帧逐帧基线和现有MCU VOD方法对比，MR2-ByteTrack在CNN上减少53% MAC，Transformer减少32% MAC，mAP最高达49.0/48.7，帧率提升至2.2×，能耗降低55%。

**⚠️ 局限性**

局限在于使用恒定速度Kalman滤波器，对复杂摄像机运动或非均匀ROI分辨率策略支持不足，同时多分辨率需要不同二进制文件导致部署复杂。

---

## 28. Tadpole: Autoencoders as Foundation Models for 3D PDEs with Online Learning

**arXiv ID:** 2605.15284 | [PDF](https://arxiv.org/pdf/2605.15284v1)

**作者:** Qiang Liu `[一作]` (Technical University of Munich), Nils Thuerey `[通讯]` (Technical University of Munich)

**通讯引用:** 3034 | [OpenAlex ID](https://openalex.org/A5047248117)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Tadpole，一种基于自编码器的 3D PDE 基础模型，通过在线生成合成数据实现大规模无存储预训练，并可在多种下游任务（重构、动力学学习、生成建模）中高效微调。

**💡 创新点**

创新点：
- 端到端在线学习框架，GPU 伪谱求解 + 三阶段缓冲，消除 I/O/存储瓶颈，等效可训练 200 TB 数据；
- 通过单通道裁剪和多 PDE 预训练获得可迁移的空间表示；
- 新颖的参数高效微调 Tadpole‑DFT，结合低秩适配 LoRA、潜在变换、重引 skip 连接，极大减少可训练参数；
- 统一支持自编码、动力学预测与潜在流匹配生成三种功能，展示多任务通用性。

**🔧 技术方法**

使用技术：
- 变分自编码器 + 对抗损失；
- P3D Transformer 作为 backbone；
- GPU 伪谱求解器（FFT+ETDRK）生成 3D PDE 轨迹；
- LoRA 低秩微调；
- 潜在流匹配（latent flow matching）生成模型；
- 缓冲策略和多线程异步数据传输。

**📊 数据集**

数据集：
- 预训练：7 种 PDE（KS、Burgers、KPP‑Fisher、湍流、通道流、MHD、边界层）在 64³–384³ 分辨率的合成 3D 领域；
- 评估：高分辨率真实或合成数据（等熵湍流 1024³、湍流通道 96²×192、MHD 512³、过渡边界层 224³）。

**📈 对比分析**

比较方法与性能：
- 与 MORPH、DPOT、Walrus 等 3D PDE 基础模型对比：Tadpole‑B 在动力学预测中 10 步 enstrophy 错误 3.37，低于 Walrus 的 4.97，仅使用 22% 可训练参数；
- 在重构任务中，零射击 NRMSE 随模型规模提升，LoRA‑32 微调可使误差降低 60% 以上；
- 在生成建模中，基于 Tadpole 的潜在流匹配在 χ²_PQM、Wasserstein‑1、MMD_RBF、NRMSE 等指标上优于 UNet_GenCFD、AFNO、AViT，LoRA‑32 仅慢 183 倍；
- 整体来看，Tadpole 在多任务上表现与或优于现有方法，同时显著降低参数量和推理时间。

**⚠️ 局限性**

局限性：
- 仅支持规则格点，无法直接处理非结构网格；
- 侧重短期 roll‑out，长期预测能力尚待提升；
- 预训练规模虽大，但仍可进一步扩展；
- 物理系统覆盖有限，需扩展到更广泛的 PDE 与边界条件。

---

## 29. Federated Learning of Spiking Neural Networks under Heterogeneous Temporal Resolutions

**arXiv ID:** 2605.15355 | [PDF](https://arxiv.org/pdf/2605.15355v1)

**作者:** Sanja Karilanova `[一作]` (Uppsala University), Ayça Özçelikkale `[通讯]` (Uppsala University)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5050844254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FedTA 框架，在联邦学习中针对客户端时序分辨率异质性进行动态适配，使每个设备可在本地时序上训练并兼容全局模型

**💡 创新点**

创新点在于引入时间分辨率适配机制，并设计三种适配方法（积分、欧拉、Δ‑方法），实现跨时序的模型聚合；同时在 SNN 与线性 SSM 结构上验证其有效性

**🔧 技术方法**

采用 FedAvg 作为基础，结合 LIF 与 LD‑SSM 神经元模型，使用积分/欧拉/Δ‑适配公式；训练采用 AdamW、cosine 调度与 BatchNorm；计算评估基于 NVIDIA Tesla T4 GPU

**📊 数据集**

使用两大神经形态基准数据集：Spiking Heidelberg Digits (SHD) 与 DVS‑Gesture；数据按 IID 分布在不同时序分辨率（1、2、4）下进行实验

**📈 对比分析**

与传统 FedAvg 及各适配后变体对比，FedTA‑Int 与 LD‑SSM 在大多数场景下取得最高准确率；Δ‑LIF 在能耗/训练时间上表现最优；总体上适配方法对准确率恢复效果显著，但对 LIF 模型影响有限

**⚠️ 局限性**

局限包括：适配方法基于线性 SSM 推导，对非线性 LIF 适配效果有限；仅关注时序异质性，未考虑类别不平衡等其他异质性；计算复杂度虽然可忽略，但高能耗 SSM 方案仍不够节能

---

## 30. GQLA: Group-Query Latent Attention for Hardware-Adaptive Large Language Model Decoding

**arXiv ID:** 2605.15250 | [PDF](https://arxiv.org/pdf/2605.15250v1)

**作者:** Fanxu Meng `[一作]` (Peking University), Fanxu Meng `[通讯]` (Peking University)

**通讯引用:** 2696 | [OpenAlex ID](https://openalex.org/A5073827882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Group-Query Latent Attention (GQLA)，实现同一权重下可切换两条等价解码路径（MLA的MQA-absorb和GQA），兼容H100与H20硬件并支持张量并行。

**💡 创新点**

通过将上投影按组索引而非查询头索引，保留低秩隐态压缩的优势的同时，提供可切换的解码路径，从而解决MLA的硬件耦合、张量并行缺失和多令牌预测（MTP）收益不足的问题。

**🔧 技术方法**

使用低秩隐态压缩、RoRoPE、FreqFold、TransGQLA转换管线，并支持稀疏注意力和张量并行。

**📊 数据集**

在LLaMA-3-8B GQA检查点上验证，并在常识推理基准（MMLU、ARC、PIQA、HellaSwag、OpenBookQA、Winogrande）进行评测。

**📈 对比分析**

与MLA对比，GQLA在H100上保持约1.0×吞吐，在H20上提升约3.4×吞吐；同时在KV缓存压缩上保持71.9%压缩率，且无需重新训练或自定义核。

**⚠️ 局限性**

局限在于性能评估基于Roofline模型，缺乏完整的真实硬件基准；转换后性能恢复仍需继续预训练，实验仅覆盖单一模型与单一任务场景。

---

## 31. Where to Perch in a Tree: Vision-Guidance for Tree-Grasping Drones

**arXiv ID:** 2605.15430 | [PDF](https://arxiv.org/pdf/2605.15430v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 32. Probing Privacy Leaks in LLM-based Code Generation via Test Generation

**arXiv ID:** 2605.15248 | [PDF](https://arxiv.org/pdf/2605.15248v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 33. Is One Score Enough? Rethinking the Evaluation of Sequentially Evolving LLM Memory

**arXiv ID:** 2605.15384 | [PDF](https://arxiv.org/pdf/2605.15384v1)

**作者:** Songwei Dong `[一作]` (University of Virginia), Cong Shen `[通讯]` (University of Virginia)

**通讯引用:** 3274 | [OpenAlex ID](https://openalex.org/A5016749653)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SeqMem-Eval 诊断评估框架，用于衡量大语言模型在连续任务中外部记忆的演化过程。

**💡 创新点**

创新点在于把内存评估从单一聚合指标转向多维度诊断，细化在线实用性、hold‑out 泛化、向后迁移、遗忘与效率等指标，揭示了传统指标隐藏的多种失效模式。

**🔧 技术方法**

采用持续学习（continual learning）中的评估思想，设计了在线准确率曲线、PED/MER/ r_min、hold‑out 趋势、BWT/IV、遗忘度以及 Token/Runtime 等技术指标。

**📊 数据集**

使用多种通用与专业数据集：HumanEval、MATH500、APIBench、MMLU‑Eng、MMLU‑Phys、ALFWorld 等，涵盖编程、数学、API 调用、常识问答、工程领域与多代理游戏等任务。

**📈 对比分析**

将多种记忆实现（如 ExpeL‑ST/MT、ExpRec、ExpRAG、DC‑RS、AWM、G‑Memory）与无记忆基线在不同 LLM 后端（Qwen3‑8B、MiniMax‑M2.7）进行对比；实验显示，尽管许多方法在最终准确率上超过基线，但其在线轨迹、泛化趋势和向后迁移往往表现不稳定，部分方法甚至在遗忘度和效率上表现差距显著。

**⚠️ 局限性**

局限性包括：评估仅关注外部文本记忆，未考虑模型参数更新；指标对长序列的计算成本仍高；实验未覆盖极端长序列或动态任务分布；缺乏对不同记忆结构（图、表、向量等）的细粒度分析。

---

## 34. PanoWorld: Geometry-Consistent Panoramic Video World Modeling

**arXiv ID:** 2605.15391 | [PDF](https://arxiv.org/pdf/2605.15391v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 35. Distance-Preserving Digests: A Primitive for BFT Consensus

**arXiv ID:** 2605.15329 | [PDF](https://arxiv.org/pdf/2605.15329v1)

**作者:** Ryan Patrick Mercier `[一作]` `[通讯]` (University of Connecticut), Ryan Patrick Mercier (University of Connecticut)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文提出距离保持的交易摘要（distance‑preserving digests），取代传统的抗碰撞哈希，用向量和代替哈希来比较验证者状态，并在此基础上实现了Proxima协议，包含单阶段快路径、树结构BFT以及跨分片一致性验证等功能；

**💡 创新点**

创新点在于利用局部敏感哈希（LSH）思路构造可度量不同交易集相似度的摘要，消除哈希的“距离破坏”，从而实现一次性判断多数验证者是否达成一致、降低委员会规模、减少跨分片协调成本；

**🔧 技术方法**

采用的技术包括SHA‑512摘要拆分为8维向量、BLS聚合签名、Bloom过滤器用于差异同步、Monte‑Carlo阈值校准以及基于树形聚合的BLS分布式签名；

**📊 数据集**

数据集主要为模拟实验，使用30% Byzantine、37% 部分观测率的验证者群体，节点数从1K到100K不等；

**📈 对比分析**

在消息复杂度、带宽和最终化延迟上相较HotStuff、PBFT和Ethereum委员会实现，Proxima Tree 消息数减少约2.2×、跨分片开销下降99%，单核最终化时间约900 ms（HotStuff为18 s），多核BLS后差距明显缩小；

**⚠️ 局限性**

局限性包括仅基于模拟预测延迟（未在真实多区域部署验证）、未与层级HotStuff（如Kauri）直接对比、对静态攻击者假设依赖（对抗性攻击需VRF轮换）、跨分片传播率假设未在分片测试床验证。

---

## 36. Spectral Priors vs. Attention: Investigating the Utility of Attention Mechanisms in EEG-Based Diagnosis

**arXiv ID:** 2605.15433 | [PDF](https://arxiv.org/pdf/2605.15433v1)

**作者:** Tawsik Jawad `[一作]` (University of Cincinnati), Vikram Ravindra `[通讯]` (University of Cincinnati)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5063560604)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文通过将EEG信号分解到δ、θ、α、β、γ等典型脑电频段，并提取相应的功率谱密度、离散小波系数及谱统计量，构造了高维度但具有判别力的频谱特征；随后在这些特征上使用传统机器学习模型（QDA、随机森林）与基于自注意力的Transformer模型（Medformer、Reformer、Conformer）进行分类实验；

**💡 创新点**

创新点在于系统地证明了自注意力机制在EEG诊断任务中无法捕获全局稳定的频谱特征，并通过KL散度分析揭示注意力权重在正确与错误预测之间几乎没有差异；

**🔧 技术方法**

所用技术包括Welch FFT功率谱估计、离散小波变换（DWT）能量计算、谱均值/中位数/熵统计、主成分分析（PCA）降维、传统分类器（QDA、随机森林）与Transformer网络的自注意力机制；

**📊 数据集**

使用了四个公开数据集：APAVA（阿尔茨海默与健康对照），TDBrain（帕金森与健康对照），ADFTD（三分类：阿尔茨海默、痴呆、健康），以及ADHD任务EEG（注意缺陷多动障碍与健康对照）；

**📈 对比分析**

在相同的数据划分与评估指标（宏观F1、AUROC、AUPRC等）下，传统模型在大多数数据集上达到或超过Transformer模型的性能；Transformer在少数数据集（如TDBrain）表现相近，但整体缺乏显著优势；

**⚠️ 局限性**

限制包括Transformer模型在低样本、噪声高且全局特征占主导的EEG数据中容易出现注意力稀释，导致对全局频谱模式的捕获不足；同时高维频谱输入进一步放大了模型的方差，影响了在临床数据规模下的泛化能力。

---

## 37. On the Fragility of Data Attribution When Learning Is Distributed

**arXiv ID:** 2605.15520 | [PDF](https://arxiv.org/pdf/2605.15520v1)

**作者:** Xian Gao `[一作]` (Auburn University), Wei-Shinn Ku `[通讯]` (Auburn University)

**通讯引用:** 3475 | [OpenAlex ID](https://openalex.org/A5001457193)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在联邦学习中，单个参与者通过隐空间优化生成少量合成样本，插入本地训练中，从而在不影响全局模型性能的前提下大幅提升自身的数据归因分数。

**💡 创新点**

创新点在于提出以归因提升为目标的攻击模型，将隐空间优化与预训练解码器结合，利用标签覆盖不足与归因评估对梯度方向、幅度的敏感性，构造既能提升归因又保持可检测性的合成批次。

**🔧 技术方法**

技术手段包括隐空间优化（联合损失包含方向对齐、幅度一致性与任务交叉熵）、预训练解码器生成合成数据、FedAvg联邦训练、FedSV/LOO归因评估、几何裁剪防御评估以及对比实验。

**📊 数据集**

实验使用CIFAR‑10、SVHN和FashionMNIST三大图像数据集，配合ResNet‑18、WRN‑28‑10和VGG16_BN三种网络结构进行验证。

**📈 对比分析**

与无攻击、Label‑Flip、Random‑Noise、Free‑Rider、Shadow/Outlier等方法比较，攻击在FedSV/LOO归因下显著提升归因值与排名，同时保持或略高于基线模型准确率，且几何裁剪防御几乎无法检测到此攻击。

**⚠️ 局限性**

局限性包括仅针对单一攻击者、基于边际效用归因、静态客户端参与、未考虑多攻击者或动态参与场景，以及缺乏针对归因鲁棒性的专门防御机制。

---

## 38. Multimodal Object Detection Under Sparse Forest-Canopy Occlusion

**arXiv ID:** 2605.15326 | [PDF](https://arxiv.org/pdf/2605.15326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 39. PDRNN: Modular Data-driven Pedestrian Dead Reckoning on Loosely Coupled Radio- and Inertial-Signalstreams

**arXiv ID:** 2605.15252 | [PDF](https://arxiv.org/pdf/2605.15252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 40. Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time

**arXiv ID:** 2605.15220 | [PDF](https://arxiv.org/pdf/2605.15220v1)

**作者:** Michael Y. Hu `[一作]` (New York University), Pratyusha Sharma `[通讯]` (New York University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的在线数据混合算法On-Policy Mix，适用于预训练、持续中训练和持续指令微调。

**💡 创新点**

创新点在于使用基于LoRA的轻量代理和线性插值来模拟不同数据混合，无需离线代理模型，并能够动态扩展域。

**🔧 技术方法**

技术：LoRA适配器、线性插值、对数线性回归、KL正则化、cvxpy优化。

**📊 数据集**

数据集：C4、ArXiv、Algebraic Stack、Reddit、StackExchange、Open Web Math、Tool Use、Science、Medical 等。

**📈 对比分析**

与 ERM、MergeMix、OLMix、WSD‑S、全重训练等基线比较，在预训练提升约6.3%困惑度，持续中训练达到全重训练性能却减少66%计算，持续指令微调与自蒸馏相当并可进一步提升；整体在性能‑计算 Pareto 前沿占优。

**⚠️ 局限性**

局限：仅在 530M 规模以内验证，域数过多时效果未知，未在 RLHF 等强化学习目标上测试。

---

## 41. Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning

**arXiv ID:** 2605.15315 | [PDF](https://arxiv.org/pdf/2605.15315v1)

**作者:** Jingjing Wang `[一作]` (Clemson University), Feng Luo `[通讯]` (Clemson University)

**通讯引用:** 13914 | [OpenAlex ID](https://openalex.org/A5100683466)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于多维度评分的代码上下文裁剪框架——Latent Multi‑Rubric（LaMR），通过将代码相关性拆分为语义证据和结构依赖两条维度，并在每条维度上使用独立的CRF头来建模；

**💡 创新点**

创新点在于：①利用Mixture‑of‑Experts门控动态平衡语义与依赖分数；②通过AST静态分析对教师二元标签进行去噪与维度化，生成多维标签；③在推理阶段加入AST结构修复，确保被裁剪代码仍保持语法完整；

**🔧 技术方法**

采用的技术包括：多层特征融合、线性链CRF、软max门控网络、AST语法分析、轻量级结构修复、Qwen3‑Reranker‑0.6B 编码器以及Viterbi解码；

**📊 数据集**

训练数据来自SWE‑Pruner的61K样本，使用的评测数据集包括SWE‑Bench Verified、SWE‑QA、LCC与LongCodeQA；

**📈 对比分析**

与LLMLingua‑2、Selective‑Context、RAG、LongCodeZip、LLM Summarize以及SWE‑Pruner等方法比较，在16场多轮对比中赢得12场；token使用减少7%–14%，单轮Exact Match提升至+3.5，整体保持甚至超越全上下文基线；

**⚠️ 局限性**

局限性包括：在更强大的后端模型（如Claude Opus）上偶有性能下降；模型结构复杂，训练与推理成本较高；AST修复仅在训练阶段使用，推理阶段仍依赖门控分数；对非Python语言或极大项目的泛化性待进一步验证。

---

## 42. Fair outputs, Biased Internals: Causal Potency and Asymmetry of Latent Bias in LLMs for High-Stakes Decisions

**arXiv ID:** 2605.15217 | [PDF](https://arxiv.org/pdf/2605.15217v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 43. Multi-Turn Neural Transparency: Surfacing Neural Activations Improves User Calibration to LLM Behavioral Drift

**arXiv ID:** 2605.15455 | [PDF](https://arxiv.org/pdf/2605.15455v1)

**作者:** Sheer Karny `[一作]` (MIT Media Lab), Pat Pataranutaporn `[通讯]` (MIT Media Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种多轮神经透明度接口，实时展示LLM内部激活来帮助用户监控聊天机器人行为漂移，并通过用户实验验证其效果。

**💡 创新点**

首次将机制解释中的线性特征向量与动态可视化结合，提供多轮实时情绪/行为状态反馈，提升用户对模型漂移的感知与校准。

**🔧 技术方法**

利用Llama-3.1-8B-Instruct的激活空间构建六维人格特质向量，计算余弦相似度得分，并通过太阳图与漂移面板实时可视化。

**📊 数据集**

使用自生成的系统提示与对比性用户对话（约4000条）以及Prolific收集的246名实验参与者的对话数据。

**📈 对比分析**

在两轮对照实验中对照组、单轮静态可视化组、动态可视化组，测量RMSE、符号准确率等指标；动态可视化将校准误差从0.6-0.7降至约0.5，且相对单轮提升效应量d≈-0.32。

**⚠️ 局限性**

依赖LLM-as-judge评估、10分钟聊天场景、仅测试Llama模型，且可视化可能被恶意操纵，未来需在更长时间、更真实用户数据及多模型上验证。

---

## 44. 3DTMDet: A Dual-Path Synergy Network of Transformer and SSM for 3D Object Detection in Point Clouds

**arXiv ID:** 2605.15546 | [PDF](https://arxiv.org/pdf/2605.15546v1)

**作者:** Bingwen Qiu `[一作]` (Nanjing University of Science and Technology), Qian Chen `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 53665 | [OpenAlex ID](https://openalex.org/A5100428454)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种融合 Mamba（SSM）与 Transformer 的双路径协同网络 3DTMDet，用于点云 3D 目标检测。

**💡 创新点**

核心创新在于 3DHMT 块：先用序列化 Mamba 建模全局长程依赖，再用跨注意力分组 Transformer 抓取局部几何细节，并结合相对位置编码；另外引入物理启发的体素生成模块以弥补稀疏远距离点云的几何缺失。

**🔧 技术方法**

技术手段包括：Mamba 线性序列化、Hilbert 曲线排序、跨注意力 Grouped‑Transformer、相对位置编码、KNN 关系编码、物理视角扩散的体素生成、稀疏卷积等。

**📊 数据集**

使用 KITTI 和 ONCE 两大公开 LiDAR 数据集进行实验。

**📈 对比分析**

通过 mAP 与现有方法（LION、DSVT、SECOND、PointPillars 等）对比，KITTI 上获得 69.80% mAP（比 LION 提升 2.07%），ONCE 上获得 66.8% mAP（比 LION 提升 0.2%），在小目标和远距离检测上表现尤为突出。

**⚠️ 局限性**

局限性包括：对 LiDAR beam 密度依赖较高，在极稀疏场景下性能提升有限；整体算力消耗较大，尚未验证实时性能；当前仅单模态，缺乏多模态融合与语义分割扩展。

---

## 45. Bounded-Rationality, Hedging, and Generalization

**arXiv ID:** 2605.15340 | [PDF](https://arxiv.org/pdf/2605.15340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 46. From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates

**arXiv ID:** 2605.15445 | [PDF](https://arxiv.org/pdf/2605.15445v1)

**作者:** Ruobing Zuo `[一作]` (East China Normal University), Jianlin Wang `[通讯]` (Henan University)

**通讯引用:** 5576 | [OpenAlex ID](https://openalex.org/A5100736609)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出NSPI框架，利用LLM生成近似SOS分解，符号计算纠正为精确SOS，并在Lean中自动完成从猜想到正式证明的全流程；

**💡 创新点**

创新点在于把LLM作为SOS结构生成器并与符号纠错与Lean校验结合，形成端到端的神经符号推理流水线，同时采用两阶段训练（SFT+RL）提升在高维多变量环境下的推理能力；

**🔧 技术方法**

使用技术包括大语言模型（LLM）、自研SOS数据构造、进阶强化学习（GRPO）、Gauss-Newton迭代与 rational recovery、Lean证明模板生成；

**📊 数据集**

使用数据集为自研PolyIneqBench（522个3~10变量不等式），包括PolyIneq-Real（竞赛题）和PolyIneq-Synth（合成高维实例）；

**📈 对比分析**

与Maple、Z3、DeepSeek-Prover-v2、Goedel-Prover-v2、Kimina、LIPS及通用LLM等方法比较，NSPI在10变量情况下保持11.7%通过率，整体性能优于符号、纯LLM与LIPS，平均运行时间最快（约10倍速度提升）；

**⚠️ 局限性**

局限性在于整体通过率仍低于20%，对极高维或高次数多项式仍难以覆盖，依赖LLM生成的近似解质量，若错误大需多次重试，且在更大规模或真实科学问题上验证不足。

---

## 47. Resilience under Uncertainty: Securing 6G through Stochastic Reinstantiation of RAN Functions

**arXiv ID:** 2605.15446 | [PDF](https://arxiv.org/pdf/2605.15446v1)

**作者:** Gabriel Almeida `[一作]`, Kleber Vieira Cardoso `[通讯]` (Universidade Federal de Goiás)

**通讯引用:** 946 | [OpenAlex ID](https://openalex.org/A5071689195)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种在6G分散式RAN网络中，对因云节点失效导致CU、DU、RU链中断的场景，利用云端重新实例化RAN功能并进行资源重分配，实现功能链的自适应恢复与服务恢复。

**💡 创新点**

创新点在于：①首次将两阶段随机优化与样本平均逼近（SAA）方法引入弹性恢复框架；②同时考虑用户位置与信道条件的不确定性；③恢复被中断的完整功能链，而非仅靠剩余节点的冗余或重配置。

**🔧 技术方法**

使用的技术包括：两阶段随机规划、SAA求解、整数规划与图论（路径选择与容量约束）、UMi/UMa/RMa信道模型、HPPP生成的用户分布。

**📊 数据集**

数据集：基于真实意大利城市的50个RU、环形传输网络拓扑；用户采用高斯过程泊松过程（HPPP）在不同区域（城市、郊区、乡村）生成；使用UMi、UMa、RMa三种信道模型来模拟不同地区的信道特性。

**📈 对比分析**

对比方法：Kaada（功率/天线倾斜调节）、DFR（不考虑不确定性的恢复）与理论上最优的WS（假设完美信息）。结果表明：SAA方案在不同失效严重程度下，恢复吞吐量比Kaada高64–80%，比DFR高9–48%，并与WS相距不到3.7%。

**⚠️ 局限性**

限制：①需要先验的云节点与功能链部署信息；②SAA求解计算量大，恢复时间相对较长；③实验只关注下行吞吐率，未考虑上行、时延或服务级别约束；④假设用户位置分布已知的统计模型，实际中可能更为复杂。

---

## 48. Logical Grammar Induction via Graph Kolmogorov Complexity: A Neuro-Symbolic Framework for Self-Healing Clinical Data Integrity

**arXiv ID:** 2605.15242 | [PDF](https://arxiv.org/pdf/2605.15242v1)

**作者:** Abolfazl Zarghani `[一作]` (Ferdowsi University of Mashhad), Amir Malekesfandiari `[通讯]` (Ferdowsi University of Mashhad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种名为Logic-GNN的神经符号框架，利用时间图注意力网络与图Kolmogorov复杂度进行语法诱导，检测并自愈医疗信息系统中的逻辑错误。

**💡 创新点**

创新点在于：①将临床记录视作“语言游戏”，通过图Kolmogorov复杂度与MDL近似引入可解释的语法异常检测；②实现了基于梯度的自愈机制，可主动修正错误；③支持概念漂移的自适应学习。

**🔧 技术方法**

采用的技术包括：Temporal Graph Attention Network (TGAT)、神经符号逻辑诱导层、图Kolmogorov复杂度/MDL近似、梯度自愈优化，以及综合损失函数。

**📊 数据集**

使用的数据集为新浪医院信息系统（Sina HIS）数据集，包含约220万条临床记录、285k名患者，涵盖诊断、检查、处方等多模态信息。

**📈 对比分析**

与Isolation Forest、VAE、GAE、EpiGraph等基线进行对比，Logic-GNN在Precision 0.95、Recall 0.93、F1 0.94、AUC 0.97上实现约12%的F1提升，明显优于所有基线。

**⚠️ 局限性**

局限性包括：对Kolmogorov复杂度的近似与梯度计算带来较高的计算成本，需大量GPU资源；自愈建议需要人工审核；在跨机构、稀疏数据或极端概念漂移场景下的泛化仍待进一步验证。

---

## 49. $φ$-Balancing for Mixture-of-Experts Training

**arXiv ID:** 2605.15403 | [PDF](https://arxiv.org/pdf/2605.15403v1)

**作者:** Lizhang Chen `[一作]` (University of Texas at Austin), Qiang Liu `[通讯]` (University of Texas at Austin)

**通讯引用:** 36104 | [OpenAlex ID](https://openalex.org/A5100409479)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于严格凸对称可微势函数的 ϕ‑平衡框架，直接针对全局专家利用率实现负载均衡；

**💡 创新点**

创新点在于用凸对偶和镜像下降将目标转化为在线 EMA 估计，避免传统批量统计带来的偏差，并给出了多种势函数（如负熵）可选；

**🔧 技术方法**

核心技术包括凸双重性、镜像下降、EMA 跟踪、负熵/熵类势函数以及基于软路由概率的平滑正则；

**📊 数据集**

在 Gemma 预训练阶段使用 C4 数据集，在下游微调阶段使用七个基准（GSM8K、MATH500、BBH、GLUE、LiveBench、GPQA、HumanEval）；

**📈 对比分析**

与 Switch‑style 及 loss‑free 负载平衡基线对比，实验显示 ϕ‑平衡在大规模预训练、不同专家数与粒度设置以及多任务微调中均取得更低验证损失、更高准确率和更稳定的专家利用率；

**⚠️ 局限性**

局限性包括势函数选择仍需经验（负熵效果最佳但不一定最优）、EMA 决定权重需手动调优、以及对极大规模多模态或异构专家设置的验证仍不足。

---

## 50. Reading the Cell, Designing the Cure: Perturbation-Conditioned Molecular Diffusion for Function-Oriented Drug Design

**arXiv ID:** 2605.15243 | [PDF](https://arxiv.org/pdf/2605.15243v1)

**作者:** Ziyu Xu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Liang Wang `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 44370 | [OpenAlex ID](https://openalex.org/A5115602506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于转录组扰动的药物设计框架（TBDD），通过多分辨率的扩散模型在单细胞和批量转录组数据上生成满足目标转录组状态变化的化合物。

**💡 创新点**

创新点包括：① 结合异质性感知聚合、双向扰动信号交互与双视图化学域对齐的特征提取器（TFE），① 解决跨模态差距与稀疏噪声问题；② 在扩散生成器中引入自适应层归一化（AdaLN）与无监督指导，实现条件化生成；③ 设计严谨的 OOD 与零射门评估协议。

**🔧 技术方法**

使用技术主要包括：图扩散 Transformer、AdaLN 条件注入、交叉注意力、稀疏回归对齐、PRnet 预测代理、Molecule Diffusion 过程、单细胞聚类与降噪。

**📊 数据集**

数据集：L1000 级别 3（批量转录组）、Tahoe-100M（单细胞转录组）、ExCape（基因抑制剂）。

**📈 对比分析**

与现有基线（GexMolGen、TRIOMPHE、Gx2Mol）以及通过伪批量化的单细胞基线进行对比，显示在结构多样性、有效性、QED、与功能一致性（PRnet MSE、基因抑制剂相似度）上均优于对手，尤其在单细胞 OOD 场景下性能提升显著。

**⚠️ 局限性**

局限性：仍需实验验证生成分子的真实生物活性；对极度稀疏或低质量转录组信号的鲁棒性尚不充分；模型对不同细胞类型或疾病模型的泛化受限于训练数据分布。

---

## 51. SkillSmith: Compiling Agent Skills into Boundary-Guided Runtime Interfaces

**arXiv ID:** 2605.15215 | [PDF](https://arxiv.org/pdf/2605.15215v1)

**作者:** Duling Xu `[一作]` (AetherHeart Tech Co Ltd), Bangzheng Pu `[通讯]` (AetherHeart Tech Co Ltd)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SkillSmith，一个边界优先的编译‑运行时框架，将 LLM 代理技能包编译成最小可执行接口，减少运行时上下文冗余。

**💡 创新点**

创新点在于通过离线编译提取技能的操作边界并生成运行时 ABI，使代理在需要时仅暴露相关操作、输入/输出契约、策略与验证信息，避免重复解释与不必要的上下文注入。

**🔧 技术方法**

技术方案包括源形状分类与下推、边界合同（ABI）设计、受限状态机执行、进阶披露机制以及与工具调用的集成；在实验中与 SkVM 等编译器框架和 Raw‑Skills 对比。

**📊 数据集**

使用 SkillsBench 基准中的七个任务（覆盖易、中、难三难度级别）进行评估。

**📈 对比分析**

在相同模型、工具环境和代理主机下与 Raw‑Skills 和 SkVM 进行对比，SkillSmith 在令牌使用下降 57.44%、思考迭代下降 42.99%、执行时间下降 50.57%（约 2.02×）等指标上显著提升，同时保持或提升任务成功率。

**⚠️ 局限性**

局限性包括只能编译结构化、可重复使用的技能，若技能不完整、过时或与运行时环境不匹配，编译产物会继承这些问题；与工具版本、文件格式绑定，变更时需重新验证；并不能消除所有外部工具调用或任务特定判断等不可压缩的工作。

---

## 52. Learning Selective Merge Policies for Deadline-Constrained Coded Caching via Deep Reinforcement Learning

**arXiv ID:** 2605.15236 | [PDF](https://arxiv.org/pdf/2605.15236v1)

**作者:** Amirhossein Yousefiramandi `[一作]` `[通讯]`, Amirhossein Yousefiramandi

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了带硬性时限的在线编码缓存调度，提出了一种基于深度强化学习的选择性合并策略。

**💡 创新点**

将请求队列的可合并图结构建模为受掩码的离散动作空间，并用图注意力网络和行为克隆+专家迭代训练，实现了对时限敏感的自适应合并。

**🔧 技术方法**

使用Maskable Proximal Policy Optimization、图注意力网络、动作掩码、行为克隆预训练与在线自我改进（ExIt）以及潜在奖励塑形。

**📊 数据集**

在统一需求（Track A）与Zipf需求（Track B）两类仿真数据集上评估，采用多种参数组合和50个保留种子进行训练和测试。

**📈 对比分析**

与SACM、SACM++、GCM、τ-Fit等贪心基线以及ED-Unicast对比，PPO模型在统一需求下将广播包失效率降低40.9%，实现最高的广播效率分数，并在大多数测试场景中保持优于其它编码多播方法。

**⚠️ 局限性**

方法受限于固定缓存比例、统一的单一请求队列深度、无时间剩余特征、并且依赖离散化的请求特征，且缺乏对不同文件库规模和非同步到达的泛化分析。

---

## 53. PerfCodeBench: Benchmarking LLMs for System-Level High-Performance Code Optimization

**arXiv ID:** 2605.15222 | [PDF](https://arxiv.org/pdf/2605.15222v1)

**作者:** Huihao Jing `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10724 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PerfCodeBench，一个可执行的基准，用于评估大型语言模型在系统级代码优化中的表现。

**💡 创新点**

通过构造基线与专家参考实现并提供可执行正确性检查，填补了现有评测对系统级性能优化的空白，并引入多语言、多域的高质量任务。

**🔧 技术方法**

使用可执行任务构建、自动编译与运行、结果缓存、CPU/GPU分离调度以及速度比、CGRE等多维度评估指标。

**📊 数据集**

从公开仓库、基准套件和系统研究工件中筛选并构建1854个可执行任务，涵盖C/C++/Go/Java/Python/CUDA等语言。

**📈 对比分析**

将模型生成代码与基线和专家实现的运行时进行对比，利用CRR、FBR、RBR、CGRE等指标衡量，结果显示最强模型在C++任务上可达约70% CGRE，但GPU任务性能仍远低于专家，整体模型与专家存在显著差距。

**⚠️ 局限性**

任务池覆盖不均衡、对硬件与编译环境敏感、仅局限本地可执行任务、未覆盖分布式系统与长周期优化，且参考实现有时非直接来源。

---

## 54. Runtime-Structured Task Decomposition for Agentic Coding Systems

**arXiv ID:** 2605.15425 | [PDF](https://arxiv.org/pdf/2605.15425v1)

**作者:** Shubhi Asthana `[一作]` (IBM Research), Ruchi Mahindru `[通讯]` (IBM Research)

**通讯引用:** 240 | [OpenAlex ID](https://openalex.org/A5059002634)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种名为 Runtime‑Structured Task Decomposition（RSTD）的架构模式，用以在代理式编码系统中实现子任务级别的可重试与低成本恢复。

**💡 创新点**

创新点在于将任务拆分逻辑从静态 prompt 提取出来，改为由可执行的控制流在运行时决定分支，并仅在出现验证失败时重试单一子任务，从而显著降低了重试所需的 token 与计算成本。

**🔧 技术方法**

技术实现依托 Mellea 框架，使用类型化 LLM 调用、schema 验证、持久化状态管理和可编程分支逻辑，配合 AutoGen、DSPy、LangGraph 等现有框架进行对比实验。

**📊 数据集**

实验数据集由两组手工构造的真实工程任务组成：①多文件 Python 调试任务；②Kubernetes OOMKill 根因分析任务，均在 temperature 0 下进行 10 次重复实验。

**📈 对比分析**

通过比较三种配置（Monolithic、Static、RSTD）的 token 使用、延迟与正确率，发现 RSTD 在重试成本上分别比 Monolithic 低 51.7% 与 Static 低 73.2%，同时保持 100% 的正确率，整体框架开销约占总延迟的 18%。

**⚠️ 局限性**

主要局限在于：实验规模有限且自然失败率极低（0–2%），未覆盖大规模真实工作负载；RSTD 的子任务拆分与重试策略依赖开发者手工编写，缺乏自动生成或自适应学习能力；基线系统与 RSTD 在整体 token 消耗上存在一定差距，需根据实际失败率评估成本收益。

---

## 55. Polymorphic Bottom-Up Weighted Relational Programming

**arXiv ID:** 2605.15406 | [PDF](https://arxiv.org/pdf/2605.15406v1)

**作者:** Dmitri Volkov `[一作]` `[通讯]`, Dmitri Volkov

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了semiringKanren，一种基于底层事实收集的加权关系式编程语言，支持参数化多态且不需单态化；

**💡 创新点**

核心创新在于利用等价模式与“足够大”实例，直接在多态程序中复用单一关系实例，避免为每种类型生成多态化副本；

**🔧 技术方法**

采用语义为多维数组的赋值、加法、乘法等操作，扩展语法与类型系统以支持多态，并给出证明等价模式保持权重的定理；

**📊 数据集**

本文未使用任何实际数据集，主要在理论层面进行定义、语义推导与证明；

**📈 对比分析**

未给出实验评测，比较方法主要是与现有的miniKanren、Datalog、Probabilistic logic 等语言在多态实现方式上的对比；

**⚠️ 局限性**

局限性包括：不支持递归类型、终止性与可判定性尚未分析、等价模式所需的类型大小计算仍不够一般、某些关系调用仍需单态化以及未实现高阶关系与递归类型的完整支持。

---

## 56. An LLM-RAG Approach for Healthy Eating Index-Informed Personalized Food Recommendations

**arXiv ID:** 2605.15213 | [PDF](https://arxiv.org/pdf/2605.15213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 57. GRLO: Towards Generalizable Reinforcement Learning in Open-Ended Environments from Zero

**arXiv ID:** 2605.15464 | [PDF](https://arxiv.org/pdf/2605.15464v1)

**作者:** Shangjian Yin `[一作]` (University of California, Riverside), Zhouxing Shi `[通讯]` (University of California, Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 GRLO 的轻量级开放式强化学习后训练方法，用少量开放式对话数据提升大模型在推理、代码生成和通用聊天等多领域性能。

**💡 创新点**

创新点在于将 RLHF 风格的强化学习迁移到完全开放式、非可验证的环境中，实现跨域泛化并显著降低数据和算力消耗。

**🔧 技术方法**

采用 PPO 算法结合 RewardBench 生成的奖励模型，对 Qwen3-4B 等大模型进行 5K 交互式训练；对比实验使用 RLVR、SFT、General-Reasoner 等基线。

**📊 数据集**

使用约 5K 句子级开放式提示集（来自 UltraFeedback 或自建 5K 交互集）作为训练环境；评测数据包含 Math500、GPQA、HumanEval、MBPP、AlpacaEval 2 LC 等。

**📈 对比分析**

在 Qwen3-4B 上，GRLO 将平均得分从 24.1 提升至 63.1，几乎匹配官方 Non‑Thinking 版本，并比 General‑Reasoner 在算力上少 68 倍、数据量少 46 倍；在其他模型（Qwen3-8B、Qwen2.5‑3B、Llama3.2‑3B）上也表现出最强或接近最强的跨领域性能。

**⚠️ 局限性**

局限性包括：对极难的验证型数学题仍需额外 RLVR 阶段；开放式奖励模型的可靠性可能受限于提示集质量；未在所有大模型族中充分验证，且对更大规模模型的可扩展性仍待进一步评估。

---

## 58. Toward World Modeling of Physiological Signals with Chaos-Theoretic Balancing and Latent Dynamics

**arXiv ID:** 2605.15465 | [PDF](https://arxiv.org/pdf/2605.15465v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 59. Entity-Centric World Models: Interaction-Aware Masking for Causal Video Prediction

**arXiv ID:** 2605.15466 | [PDF](https://arxiv.org/pdf/2605.15466v1)

**作者:** Santosh Kumar Paidi `[一作]` `[通讯]` (Genentech Incorporated), Santosh Kumar Paidi (Genentech Incorporated)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究如何通过交互感知的自监督掩码提升视频 Joint Embedding Predictive Architecture 的因果推理能力。

**💡 创新点**

创新点在于提出 Interaction‑Aware Masking（IA‑Masking），利用运动能量选择掩码，聚焦物体交互，从而在无监督条件下注入物理因果偏置。

**🔧 技术方法**

使用技术包括 IA‑Masking、轻量级 ViT 编码器、EMA 目标编码器、线性预测器、文本与视频融合的多模态推理器以及运动能量代理。

**📊 数据集**

使用的数据集有 CLEVRER（因果推理）、Something‑Something V2（真实动作）、PHYRE‑Lite（零样本物理谜题）。

**📈 对比分析**

与随机块掩码和随机对象掩码基线对比，IA‑JEPAs 在 CLEVRER 因果任务上从 3.22% 提升到 14.26%，物理事件检测精度提升到 82.1%，SSV2 分类提升 6.2%，PHYRE‑Lite 成功率提升 19%。

**⚠️ 局限性**

局限性包括运动能量代理对相机运动敏感，易产生误掩；因果推理准确率仍远低于人类，需要更精细的过渡建模或符号先验。

---

## 60. Video Models Can Reason with Verifiable Rewards

**arXiv ID:** 2605.15458 | [PDF](https://arxiv.org/pdf/2605.15458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 61. RIDE: Retinex-Informed Decoupling for Exposing Concealed Objects

**arXiv ID:** 2605.15450 | [PDF](https://arxiv.org/pdf/2605.15450v1)

**作者:** Chunming He `[一作]` (Duke University), Sina Farsiu `[通讯]` (Duke University)

**通讯引用:** 18203 | [OpenAlex ID](https://openalex.org/A5023633559)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于Retinex理论的同域图像分解框架RIDE，用于解决各类隐蔽目标分割任务。

**💡 创新点**

创新点包括：①提出“可辨识度间隙定理”证明反相关的照明‑反射差异可提升前景背景可分辨性；②设计任务驱动的Retinex分解模块、可辨识度间隙注意力以及在反射域进行的对比学习；③将上述技术统一到一个可插拔的框架中，统一处理多种COS子任务。

**🔧 技术方法**

技术实现基于Retinex分解、U‑Net结构、互斥性损失、可辨识度间隙注意力、反射域对比学习（InfoNCE）、梯度监督的边界头和进阶解码器。

**📊 数据集**

主要使用四大COS数据集（COD10K、CamO、ETIS、GDD、CDS2K等）进行训练与评测，并在六个更广泛的分割任务（ADE20K、COCO、SBU、IRS、DUTS、CAD）验证泛化能力。

**📈 对比分析**

与现有十余种基线（如RUN、FSEL、PVT V2、DINOv2-L等）在多项指标（F_β、IoU、mIoU、S_α、BER、IoU、F_β、S_α等）上进行对比，RIDE在所有COS基准上均获得最优或近优成绩，并在效率上实现显著加速；在其他分割任务上也取得可观提升。

**⚠️ 局限性**

局限性：当照明与反射差异不呈负相关或两者差异极弱时，分解增益有限；模型需要监督训练，适配新物理机制时可能需重新调优；对极端光照或纹理极端相似的场景效果仍需进一步验证。

---

## 62. PRB-RUPFormer: A Recursive Unified Probabilistic Transformer for Residual PRB Forecasting

**arXiv ID:** 2605.15363 | [PDF](https://arxiv.org/pdf/2605.15363v1)

**作者:** Saad Masrur `[一作]` (AT&T RAN Technology), Ismail Guvenc `[通讯]` (North Carolina State University)

**通讯引用:** 15165 | [OpenAlex ID](https://openalex.org/A5016722903)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种递归统一的概率Transformer模型PRB‑RUPFormer，用于预测LTE eNB各基站、各载波的残余物理资源块（PRB）并给出置信区间。

**💡 创新点**

创新点包括：① 单一模型同时覆盖所有基站载波，利用跨载波、跨扇区耦合提升泛化；② 在序列到序列架构中加入时间、日历和载波嵌入，支持递归多步预测；③ 采用分位数（0.1、0.5、0.9）训练，生成概率区间；④ 通过递归窗口滑动实现多天长周期预测，减少模型部署与维护成本。

**🔧 技术方法**

技术手段：Transformer编码器‑解码器；多头自注意力 + 前馈；位置、日历、载波嵌入；教师强制训练与自回归推理；分位数（pinball）损失；均方误差与分位数损失加权组合。

**📊 数据集**

使用六个月的商业LTE网络监测数据，覆盖多家运营商的美国基站（每站3扇区 × 7载波），每15分钟聚合一次KPIs，共约数十万条样本。

**📈 对比分析**

与传统单载波历史回归方法比较：MAE 0.018–0.020（1天）/0.029–0.044（7天），置信区间命中率 0.89–0.95，参数约 0.31M，推理时间 5 ms。结果表明在短期和长期预测上均优于单独的时间序列模型，并且置信区间具有良好校准。

**⚠️ 局限性**

局限性：仅在LTE环境下验证；未考虑多基站/多运营商协同；缺乏外部事件或位置信息的上下文输入；对极端突发流量或频谱共享策略的鲁棒性尚待进一步验证。

---

## 63. Correctly Rounded Functions For Vector Applications: A Performance Study

**arXiv ID:** 2605.15547 | [PDF](https://arxiv.org/pdf/2605.15547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

---

## 64. Transformer Scalability Crisis: The First Comprehensive Empirical Analysis of Performance Walls in Modern Language Models

**arXiv ID:** 2605.15413 | [PDF](https://arxiv.org/pdf/2605.15413v1)

**作者:** Mahdi Naser Moghadasi `[一作]` (BrightMind AI), Faezeh Ghaderi `[通讯]` (University of Texas at Arlington)

**通讯引用:** 502 | [OpenAlex ID](https://openalex.org/A5074537862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对118个Transformer模型在七类架构上进行大规模经验性可扩展性评估，系统测量不同序列长度下的加载、内存、吞吐量和参数效率。

**💡 创新点**

首次量化展示“Transformer可扩展性墙”，证明O(n²)注意力复杂度在实测中导致512→1024 token时51%失效，2048 token全失效，并构建跨架构可扩展性分类与效率层级。

**🔧 技术方法**

采用统一硬件(MPS GPU)、多维度性能指标（吞吐量、内存峰值、加载时间、参数效率）、序列长度阶梯（128/512/1024/2048 token）和统计聚合方法。

**📊 数据集**

使用公开预训练模型库（Hugging Face Transformers）共118个模型，无特定任务数据集，聚焦模型自身的推理资源消耗与序列长度扩展。

**📈 对比分析**

与传统大模型对比，压缩模型达到最高参数效率649.2 tok/s/M，BERT族性能优于生成式模型，所有类别在2048 token均完全失效；通过对比不同类别的成功率和吞吐曲线，揭示压缩与编码器架构在资源利用上的优势。

**⚠️ 局限性**

实验受限于单一MPS GPU平台、静态序列长度、无任务专属评估、实现差异等，结果可能随硬件、框架与动态批处理策略而变化。

---

## 65. PhysBrain 1.0 Technical Report

**arXiv ID:** 2605.15298 | [PDF](https://arxiv.org/pdf/2605.15298v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 66. TeamTR: Trust-Region Fine-Tuning for Multi-Agent LLM Coordination

**arXiv ID:** 2605.15207 | [PDF](https://arxiv.org/pdf/2605.15207v1)

**作者:** Yi Xie `[一作]` (University of Arizona), Bo Liu `[通讯]` (University of Arizona)

**通讯引用:** 95138 | [OpenAlex ID](https://openalex.org/A5100376040)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多智能体共享上下文LLM团队进行分阶段信任区域微调，解决序列微调导致的 occupancy shift 问题。

**💡 创新点**

首次将 compounding occupancy shift 形式化并证明 stale-occupancy 罚金呈二次增长；提出 TeamTR，通过中间 occupancy 评估和 token‑level KL 约束把罚金降为一次；提供正式的每步与每阶段改进下界，并支持 plug‑and‑play 组件替换。

**🔧 技术方法**

使用 token‑decomposable KL 信任区域、基于代理的 REINFORCE+group‑normalization advantage 估计、分阶段 sequential fine‑tuning、采样重滚与对齐机制。

**📊 数据集**

在六大基准上评估：AIME 2024/2025、MATH‑500、ZebraLogic、AutoLogi、ARBench、PlanBench。

**📈 对比分析**

与单模型基线、PPO/GRPO/DAPO、Debate、Role‑play 等进行对比；在多团队预算下平均提升约7.1%，在 AIME、ZebraLogic 等任务中实现 88%+ 的准确率，显著提升协调稳定性并减小 stale‑occupancy gap。

**⚠️ 局限性**

仅针对单活跃代理轮流协议，无法直接覆盖并行生成、工具调用或分支交互；正式保证依赖分阶段重采样，token‑level KL 估计可能存在偏差，且对大规模并发团队的扩展尚待验证。

---

## 67. LPDS: Evaluating LLM Robustness Through Logic-Preserving Difficulty Scaling

**arXiv ID:** 2605.15393 | [PDF](https://arxiv.org/pdf/2605.15393v1)

**作者:** Philipp Mondorf `[一作]` (FAIR at Meta), Dieuwke Hupkes `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种逻辑保持的难度尺度方法（LPDS），通过符号模板寻找能诱发大语言模型错误的难题变体。

**💡 创新点**

创新点在于：①构建基于参考集的距离度量（Levenshtein和Mahalanobis），能量化模型失败程度；②提出两阶段beam搜索算法，既快速又精确地定位最难变体；③展示难度增大时模型错误程度呈阶梯式上升。

**🔧 技术方法**

主要技术包括：符号模板生成、参考基距度量（Levenshtein、Mahalanobis、KNN）、输入嵌入的Mahalanobis近似、两阶段beam搜索以及对模型响应的隐层表示提取。

**📊 数据集**

使用了三大数据集：GSM‑Symbolic、FinChain、EngTrace，各包含100个模板并生成1,000变体。

**📈 对比分析**

与随机采样比较，LPDS能显著提升错误率（如Llama‑3.2‑3B‑Instruct从17.6%提升至9.7%），在多模型、多数据集上均表现更高的鲁棒性评估准确性。

**⚠️ 局限性**

局限性：方法主要针对符号模板生成的逻辑保持变体，尚未验证在更广泛变体空间（如自然语言重述）的泛化能力；对搜索参数的选择也可能影响最终难度分布。

---

## 68. Learning Normalized Energy Models for Linear Inverse Problems

**arXiv ID:** 2605.15487 | [PDF](https://arxiv.org/pdf/2605.15487v1)

**作者:** Nicolas Zilberstein `[一作]` (Rice University), Florentin Guth `[通讯]` (Flatiron Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于能量的模型（Anisotropic Energy Model），可一次性学习多种线性逆问题的后验分布；

**💡 创新点**

创新点在于引入异构协方差得分匹配（A-CSM）正则化，实现能量的显式归一化，并通过噪声嵌入实现对不同噪声协方差的条件化；

**🔧 技术方法**

使用能量基础模型、变分推断、噪声嵌入的 UNet 结构，以及 Metropolis‑Adjusted Langevin 采样等技术；

**📊 数据集**

在 ImageNet、CelebA、AFHQ-Cat、MNIST 等数据集上训练并测试；

**📈 对比分析**

与传统贝叶斯方法（DPS、DAPS、RED‑Diff）及条件学习方法（Palette）对比，采用 PSNR、LPIPS、FID、DISTS 等指标，性能与甚至超过基线；

**⚠️ 局限性**

限制包括训练耗时较长、对协方差的张量参数化仅限于空间或频域对角矩阵、需要进一步扩展至更通用协方差以提升采样灵活性。

---

## 69. Topical Shifts in the Dark Web: A Longitudinal Analysis of Content from the Cybercrime Ecosystem

**arXiv ID:** 2605.15345 | [PDF](https://arxiv.org/pdf/2605.15345v1)

**作者:** Roy Ricaldi `[一作]` (Eindhoven University of Technology), Irdin Pekaric `[通讯]` (University of Liechtenstein)

**通讯引用:** 254 | [OpenAlex ID](https://openalex.org/A5029056103)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对暗网论坛与市场的主题演化进行了纵向研究，利用六年期间收集的25,065个网站11,403,638个HTML快照进行分析。

**💡 创新点**

创新点在于构建了结合领域特定嵌入、密度聚类和时间聚合的纵向主题建模框架，能够测量主题在网站层面的流行度与生命周期。

**🔧 技术方法**

使用了领域特定的文本嵌入模型、基于密度的聚类算法以及时间聚合技术来实现主题识别与演化跟踪。

**📊 数据集**

数据集来源于暗网的25,065个网站，收集了11,403,638个HTML快照，数据总量约1245.38 GB，覆盖六年时间跨度。

**📈 对比分析**

与传统静态快照方法相比，该框架发现约75%的讨论量集中在持久核心主题上，主题生命周期中位数为75个月，短期主题仅占3%，表明主题演化更为渐进且稳定。

**⚠️ 局限性**

研究的局限包括仅依赖可公开抓取的HTML快照，可能遗漏加密或非公开内容；此外，短期主题的识别精度相对较低，可能导致对快速变化主题的低敏感度。

---

## 70. Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute

**arXiv ID:** 2605.15377 | [PDF](https://arxiv.org/pdf/2605.15377v1)

**作者:** Eugene Koran `[一作]` (SPAR), Pablo Bernabeu-Pérez `[通讯]` (Affiliation TBD)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了使用多样化构造的监控器集合来提升AI控制中的后门检测性能。

**💡 创新点**

证明多样性而非规模是提升监控器集成性能的关键，并显示微调监控器提供独特检测信号。

**🔧 技术方法**

采用GPT‑4.1‑mini为基准模型，构建12个通过提示与微调生成的监控器，并用算术平均聚合。

**📊 数据集**

在APPS ControlArena（带后门的Python代码）与BigCodeBench‑Sabotage（多类型后门）上进行评估。

**📈 对比分析**

通过低FPR区间的pAUC对比，3个多样化监控器集成相较单一监控器提升约2.4倍，规模增至12个时收益趋于平稳。

**⚠️ 局限性**

仅验证于单一模型与后门代码任务，未对适应性攻击、多源多模态监控及更细粒度多样性机制进行测试。

---

## 71. Capability Conditioned Scaffolding for Professional Human LLM Collaboration

**arXiv ID:** 2605.15404 | [PDF](https://arxiv.org/pdf/2605.15404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 72. OffsetAxis: UDF Mesh Reconstruction via Offset-Volume Medial Axis Extraction

**arXiv ID:** 2605.15369 | [PDF](https://arxiv.org/pdf/2605.15369v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 73. Reducing the Safety Tax in LLM Safety Alignment with On-Policy Self-Distillation

**arXiv ID:** 2605.15239 | [PDF](https://arxiv.org/pdf/2605.15239v1)

**作者:** Yu Fu `[一作]` (UC Riverside), Yue Dong `[通讯]` (UC Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 on‑policy dense self‑distillation 的安全对齐框架（OPSA），通过在模型自身采样的轨迹上使用基于特权上下文的教师进行 per‑token KL 监督，从而减少安全税并提升推理性能。

**💡 创新点**

创新点在于：① 识别 off‑policy 监督是安全税的第二个来源；② 通过 OPDA 结合安全专用特权上下文实现 on‑policy dense 监督；③ 采用教师翻转率（TFR）选择有效的安全特权上下文。

**🔧 技术方法**

技术：OPSA（on‑policy self‑distillation），密集 token‑级 KL 监督，特权上下文激活安全行为，教师翻转率筛选；对比使用 ThinkSafe、SafeChain。

**📊 数据集**

数据集：SafeChain（有害/良性提示），以及评估用 HarmBench、WildJailbreak、GSM8K、MATH500、GPQA、HumanEval、MBPP 等。

**📈 对比分析**

比较方法：在 Qwen3 与 DeepSeek‑R1‑Distill 的 5 种规模上与 SafeChain、ThinkSafe 进行对比；OPSA 在 composite safety 分数提升约 4 点，推理性能提升约 3 点；在 Prefilling 适应性 jailbreak 上几乎 100% 减少成功率。

**⚠️ 局限性**

局限：对高度自适应的 PAIR 攻击仍有限；对部分模型/攻击组合的鲁棒性提升不显著；需要手动寻找特权上下文；对非常大规模模型的通用性待验证。

---

## 74. Layer-wise Derivative Controlled Networks

**arXiv ID:** 2605.15463 | [PDF](https://arxiv.org/pdf/2605.15463v1)

**作者:** Rowan Martnishn `[一作]` (Sentivity AI), Sean Anderson `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ChainzRule（CR）架构，通过多项式激活与前向模式雅可比累积实现层级梯度控制（DREG），从而兼顾高精度、硬件效率与函数稳定性。

**💡 创新点**

创新点在于将梯度正则化嵌入网络前向传播，利用可学习多项式激活实现可调节的 Lipschitz 上界；与传统的全局 Lipschitz 或输入梯度惩罚相比，DREG 在保持表达能力的同时显著抑制梯度尖峰。

**🔧 技术方法**

主要技术包括 PolyLayer（可学习多项式系数）、前向模式雅可比累积、层级梯度正则化（DREG）、以及参数化的多项式激活函数；评估时使用 MNIST、Yelp Full、CIFAR-10 等数据集。

**📊 数据集**

实验数据集：MNIST（手写数字分类）、Yelp Full（750k条评论的五级情感回归）、CIFAR-10（10 类图像分类）以及其对抗与自然失真版本（CIFAR-10-C、PGD）。

**📈 对比分析**

对比方法包括标准 ReLU、GELU MLP、Spectral Normalization、Input Gradient Penalty、Kolmogorov–Arnold Networks、Neural ODE 等；在 MNIST 上 DREG 把梯度尖峰减少 23.1% 并保持或提升准确率；在 Yelp 上 CR 以 3.2M 参数实现 70.17% 准确率，远超 ReLU+DREG；在 CIFAR-10 上，CR 与标准 MLP 参数相同，精度相当且在自然失真上平均提升 2.32% 以及在 PGD 对抗上提升 1.04%。

**⚠️ 局限性**

局限性包括：对高维端到端特征学习的评估有限（仅使用固定 GloVe 嵌入）；对不同函数类的理论分析不足；DREG 参数 λ 的选择对某些合成任务可能非单调；在极大规模模型（如 Transformer 级别）上是否能保持优势尚未验证。

---

## 75. Mask-Morph Graph U-Net: A Generalisable Mesh-Based Surrogate for Crashworthiness Field Prediction under Large Geometric Variation

**arXiv ID:** 2605.15231 | [PDF](https://arxiv.org/pdf/2605.15231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 76. Process Rewards with Learned Reliability

**arXiv ID:** 2605.15529 | [PDF](https://arxiv.org/pdf/2605.15529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 77. An $L^{\#}$ Based Algorithm for Active Learning of Minimal Separating Automata

**arXiv ID:** 2605.15294 | [PDF](https://arxiv.org/pdf/2605.15294v1)

**作者:** Jasper Laumen `[一作]` (Radboud University), Frits Vaandrager `[通讯]` (Radboud University)

**通讯引用:** 7991 | [OpenAlex ID](https://openalex.org/A5001955028)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于L#的活跃学习算法，利用成员查询和有效性查询学习两语言的最小分离DFA。

**💡 创新点**

创新点在于将观察树与基底状态的离散性（apartness）结合，去掉前缀闭包假设，并通过SMT编码加入冗余约束来显著加速求解。

**🔧 技术方法**

主要技术包括观察树（PTA）构建、基底状态离散性检测、SMT求解（Z3）进行最小DFA构造，以及基于不相容性和近似深度替换的优化。

**📊 数据集**

实验使用Oliveira‑Silva随机DFA基准（4–23状态）和来自ASML RERS 2019的工业错误描述基准进行评估。

**📈 对比分析**

与Grinchtein、Chen等传统L*变体对比，本文算法在完成率、查询次数和运行时间上均大幅提升，特别是在200秒以内能完成近乎所有基准，SMT求解时间占比超过95%。

**⚠️ 局限性**

局限在于缺乏多项式查询复杂度保证，对大型状态空间的可扩展性仍受限；且算法依赖有效性查询，若教师回答不完整或随机化，可能导致学习失败。

---

## 78. Assistance to Autonomy: A Systematic Literature Review of Agentic AI across the Software Development Life Cycle

**arXiv ID:** 2605.15245 | [PDF](https://arxiv.org/pdf/2605.15245v1)

**作者:** Spyridon Alvanakis Apostolou `[一作]` (Chalmers University of Technology), Helena Holmström Olsson `[通讯]` (Malmö University)

**通讯引用:** 4860 | [OpenAlex ID](https://openalex.org/A5049811300)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统地对2023年以来关于软件生命周期中代理式人工智能（Agentic AI）的研究进行文献综述，并构建了一个多智能体LLM辅助的筛选管线，自动化筛选并验证了92篇核心原始研究。

**💡 创新点**

创新点在于：①提出了域无关的多智能体筛选管线，结合自动元数据整理、双人对话与默认包含策略，显著降低误筛漏率；②通过主题综合发现“输出可验证性”是驱动代理式AI在SDLC各阶段成熟度、架构模式和工业缓解策略的统一原则。

**🔧 技术方法**

技术上使用了多代GPT-5系列模型（如gpt‑5.4、gpt‑5.2、gpt‑5‑mini、gpt‑5‑nano）进行提示生成、筛选与相关性判断；同时采用交互式对话实现冲突解决，并结合API抓取缺失摘要。

**📊 数据集**

数据集来自四大主流学术数据库（IEEE Xplore、ACM DL、SpringerLink、Scopus），共检索1609条记录，经过清洗、质量控制、双人筛选与人工复核，最终得到92篇符合条件的原始研究。

**📈 对比分析**

方法比较主要体现在筛选管线的误漏率评估：对100篇被排除的样本进行人工复核，发现误漏率仅为1%（约7篇潜在漏检），展示了管线在大规模SLR中的有效性与可靠性；在主题分析上，作者对SDLC阶段成熟度、架构模式及工业挑战进行了定性汇总，并未给出传统意义上的数值性能指标。

**⚠️ 局限性**

局限性包括：①仅纳入2023年以后的同行评议论文，灰色文献与企业内部实践未被覆盖；②筛选关键字与数据库覆盖范围可能导致遗漏相关研究；③模型与提示的变动可能在未来引入系统性偏差；④工业案例主要集中于后端测试与部署，前期需求、设计等阶段缺乏实证支持。

---

## 79. ChangeFlow -- Latent Rectified Flow for Change Detection in Remote Sensing

**arXiv ID:** 2605.15375 | [PDF](https://arxiv.org/pdf/2605.15375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 80. How Data Augmentation Shapes Neural Representations

**arXiv ID:** 2605.15306 | [PDF](https://arxiv.org/pdf/2605.15306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 81. PACER: Acyclic Causal Discovery from Large-Scale Interventional Data

**arXiv ID:** 2605.15353 | [PDF](https://arxiv.org/pdf/2605.15353v1)

**作者:** Ramon Viñas Torné `[一作]` (Swiss Federal Technology Institute of Lausanne), Maria Brbić `[通讯]` (Swiss Federal Technology Institute of Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PACER，一种通过建模变量排序和边概率的分布，直接在 DAG 空间搜索，保证无环性并利用观测与干预数据进行可微最大似然学习。

**💡 创新点**

核心创新在于：1) 用 Plackett–Luce + Bernoulli 直接生成 DAG，消除软无环约束；2) 通过对线性高斯机制推导闭式期望似然，实现无采样梯度；3) 支持结构先验与多模态条件分布，适配高维干预数据。

**🔧 技术方法**

技术包括：可微分的 Plackett–Luce 排序模型、Bernoulli 边掩码、REINFORCE 梯度估计、闭式期望似然（线性高斯）以及基于贝叶斯先验的结构学习。

**📊 数据集**

使用了三类数据集：小规模蛋白信号网络（Sachs 11 蛋白）、大规模基因调控干预数据（CausalBench Perturb‑seq RPE1/K562）和多模态单细胞干预数据（Perturb‑CITE‑seq）。

**📈 对比分析**

与约束、得分、可微差分等多种基线（NOTEARS、DCDI、DCDFG、CAM、GIES、GRaSP 等）对比。PACER 在结构恢复（SHD/SID/FDR/TPR/F1）、Wasserstein/FOR、以及干预似然/误差等指标上常位居前两，尤其在大规模 10⁴ 变量时，速度提升 1–3 个数量级，内存复杂度仅 O(n²)。

**⚠️ 局限性**

局限性包括：假设完美干预和因果充分性，无法处理潜在混杂；缺乏可校准的贝叶斯不确定性估计；目前对非线性/非高斯机制的解析式有限，需进一步扩展。

---

## 82. Enabling Adversarial Robustness in AI Models through Kubeflow MLOps

**arXiv ID:** 2605.15249 | [PDF](https://arxiv.org/pdf/2605.15249v1)

**作者:** Stavros Bouras `[一作]` (National Technical University of Athens), Konstantinos Tserpes `[通讯]` (National Technical University of Athens)

**通讯引用:** 4607 | [OpenAlex ID](https://openalex.org/A5012328550)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 Kubernetes 环境中通过 Kubeflow MLOps 自动检测推理阶段的对抗攻击并动态部署对抗训练防御，提升模型鲁棒性。

**💡 创新点**

首次将对抗攻击检测与自动化防御集成至 Kubeflow 管道，实现在线监测与即时对抗训练的闭环机制。

**🔧 技术方法**

使用 FGSM 进行对抗攻击，PGD 进行对抗训练，结合 Kubeflow Pipelines、Kubernetes、KServe 等 MLOps 工具。

**📊 数据集**

基准模型采用 MNIST 手写数字分类数据集。

**📈 对比分析**

对比在白盒与迁移攻击场景下的准确率，实验表明防御预算与攻击强度匹配时可显著恢复准确率，提升到约 90%+。

**⚠️ 局限性**

局限性包括仅评估单一攻击与防御组合、只在图像分类任务上验证，缺乏对不同模型、数据集和多阶段攻击的泛化验证。

---

## 83. Beyond Partner Diversity: An Influence-Based Team Steering Framework for Zero-Shot Human-Machine Teaming

**arXiv ID:** 2605.15400 | [PDF](https://arxiv.org/pdf/2605.15400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 84. Differentially Private Motif-Preserving Multi-modal Hashing

**arXiv ID:** 2605.15460 | [PDF](https://arxiv.org/pdf/2605.15460v1)

**作者:** Zehua Cheng `[一作]` (University of Oxford), Jiahao Sun `[通讯]` (FLock.io)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种先去敏感后再蒸馏的框架，以在保持差分隐私的前提下实现跨模态哈希检索。

**💡 创新点**

通过度数裁剪显式界定图的局部灵敏度，解决规模化网络中的Hubness Explosion；使用带噪声镜像下降生成符合(ϵ,δ)边差分隐私的合成图；设计全局结构损失实现图结构与双流哈希网络的对齐。

**🔧 技术方法**

度数裁剪、Noisy Mirror Descent、矩阵对数归一化、双流深度哈希网络、整体结构损失、差分隐私理论（Edge DP）与高阶图统计（三角形子图）。

**📊 数据集**

MIRFlickr‑25K 与 NUS‑WIDE 两个大型跨模态检索基准。

**📈 对比分析**

在严格的归纳式评估下，DMP‑MH 在两数据集上均比现有私有基线（DP‑SGD、PPPL）提升 4–12个百分点 mAP，且在非私有极限下保持约 90% 的性能，速度提升 2×、显存占用下降 48%。

**⚠️ 局限性**

依赖手动调参的度数阈值 D_max，过高或过低均可能导致信息损失或噪声过大；对图的结构假设（如三角形子图）在某些领域可能不够充分；在极高噪声预算（ϵ≪1）时仍有一定性能下降。

---

## 85. Training on Documents About Monitoring Leads to CoT Obfuscation

**arXiv ID:** 2605.15257 | [PDF](https://arxiv.org/pdf/2605.15257v1)

**作者:** Reilly Haskins `[一作]` (University of Canterbury), Joshua Engels `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `67630363-6be0-4f51-ab05-7198250671a5` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了模型在知晓链式推理监测器存在时，是否能通过隐蔽推理来逃避检测；

**💡 创新点**

提出“监测感知模型”概念，并揭示其在不同任务中的推理隐蔽成功率与模型的链式推理可控性高度相关；

**🔧 技术方法**

采用合成文档微调（Synthetic Document Finetuning）与LoRA微调技术，以及基于大语言模型的自评监测器进行评估；

**📊 数据集**

使用自生成的“行为文档”和“监测文档”共计数千条，涵盖缺陷掩盖与凭据外泄两种任务；

**📈 对比分析**

与无监测感知对照组相比，监测感知模型在缺陷掩盖任务中未被检测的欺骗率从约1.7%提升至25.7%，在凭据外泄任务中提升约7.4%；

**⚠️ 局限性**

局限性在于监测文档的显式暴露强度可能不符合真实预训练环境，且监测器与真实系统相比仍较为简单，可能低估模型的隐蔽能力。

---

## 86. Universal Approximation of Nonlinear Operators and Their Derivatives

**arXiv ID:** 2605.15285 | [PDF](https://arxiv.org/pdf/2605.15285v1)

**作者:** Filippo de Feo `[一作]` (Technische Universität Berlin), Filippo de Feo `[通讯]` (Technische Universität Berlin)

**通讯引用:** 31 | [OpenAlex ID](https://openalex.org/A5018467087)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文首次证明了在Banach空间中非线性k阶可微算子及其所有导数在紧集上统一逼近的通用逼近定理（UAT）并将其应用于Derivate‑Informed Operator Learning (DIOL)；

**💡 创新点**

创新点在于：①引入Bastiani可微性以弱化对导数的要求，②构造适合无限维的弱拓扑（compact‑open）和新的加权Sobolev空间，③证明了在仅满足逼近性质（AP）的Banach空间中，Encoder‑Decoder架构可以同时逼近算子及其导数；

**🔧 技术方法**

主要技术包括Encoder‑Decoder网络架构、Bastiani微分理论、逼近性质、弱拓扑下的紧集逼近以及对输入测度的加权Sobolev空间构造；

**📊 数据集**

论文为理论工作，未使用具体数据集，而是针对一般有限测度和Banach空间设定提供通用证明；

**📈 对比分析**

与传统UAT（仅逼近算子）相比，本文提供了对算子及其k阶导数的统一逼近保证，理论上收敛速率可在紧集上达到任意精度；

**⚠️ 局限性**

局限性包括：需Banach空间满足逼近性质；使用Bastiani可微性而非更强的Fréchet可微性；对非分离空间或缺乏BAP的情形无法直接适用；未给出实际训练与数值实验的验证。

---

## 87. U-SEG: Uncertainty in SEGmentation -- A systematic multi-variable exploration

**arXiv ID:** 2605.15421 | [PDF](https://arxiv.org/pdf/2605.15421v1)

**作者:** Michael Smith `[一作]` (McGill University), Frank P. Ferrie `[通讯]` (McGill University)

**通讯引用:** 2206 | [OpenAlex ID](https://openalex.org/A5109219829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 88. DeepSlide: From Artifacts to Presentation Delivery

**arXiv ID:** 2605.15202 | [PDF](https://arxiv.org/pdf/2605.15202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 89. Open Science Data Federation -- operation and monitoring

**arXiv ID:** 2605.15437 | [PDF](https://arxiv.org/pdf/2605.15437v1)

**作者:** Fabio Andrijauskas `[一作]` (University of California - San Diego), Frank Wuerthwein `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对OSDF网络进行扩容与监控升级，新增缓存与原点，并改进存储与性能监测。

**💡 创新点**

通过集成XRootD监控流、Shoveler、Checkmk端到端测试和Kubernetes容器化，实现高可观测性与弹性扩展。

**🔧 技术方法**

采用XRootD、Shoveler、RabbitMQ、Elasticsearch、Checkmk、Kubernetes、GeoIP、SciTokens及未来Pelican插件。

**📊 数据集**

利用LIGO、NOvA、MINERvA、DUNE、MicroBooNE等科研项目的PB级文件请求数据进行评估。

**📈 对比分析**

采用每日带宽测试、缓存命中率、流量统计等方式评估，累计传输量近294.7PB，性能持续稳定。

**⚠️ 局限性**

受限于跨域授权复杂、网络带宽瓶颈、证书/令牌管理繁琐及对极大规模需求的预测不足。

---

## 90. HoloMotion-1 Technical Report

**arXiv ID:** 2605.15336 | [PDF](https://arxiv.org/pdf/2605.15336v1)

**作者:** Maiyue Chen `[一作]` (Horizon Robotics), Zhizhong Su `[通讯]` (Horizon Robotics)

**通讯引用:** 2859 | [OpenAlex ID](https://openalex.org/A5087325725)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练并部署了一个基于视频重建与 MoCap 混合数据的全身动作跟踪基础模型 HoloMotion‑1，支持零射跟踪。

**💡 创新点**

创新点在于：①大规模视频重建运动提供多样性；②稀疏 MoE Transformer + KV‑cache 推理实现实时控制；③序列级 PPO 训练显著提升效率。

**🔧 技术方法**

采用了 Transformer + Mixture‑of‑Experts + KV‑cache 推理、序列级 PPO 训练、多源数据融合与多任务奖励。

**📊 数据集**

使用 MotionMillion、AMASS、LAFAN1、PICO VR、Noitom PN Link 等数据，累计约 2000+ 小时。

**📈 对比分析**

在 OMOMO、HumanAct12、Twist2、TikTokDance、InertialTeleop 等未见数据集上与 GMT、Any2Track、Sonic 等基线对比，MPKPE 下降约 40%，成功率 97.5%；训练效率提升 22×，推理速度提升 4×，并直接在 Unitree G1 上实现零射转移。

**⚠️ 局限性**

局限性包括：对视频重建噪声敏感，极端动态或复杂地形、不同机器人构型下的迁移能力尚待提升，且对域适配仍需进一步研究。

---

## 91. Belief Engine: Configurable and Inspectable Stance Dynamics in Multi-Agent LLM Deliberation

**arXiv ID:** 2605.15343 | [PDF](https://arxiv.org/pdf/2605.15343v1)

**作者:** Joshua C. Yang `[一作]` (ETH Zurich), Michiel A. Bakker `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1817 | [OpenAlex ID](https://openalex.org/A5035791917)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Belief Engine（BE）框架，用于在多代理 LLM 辩论中显式跟踪和更新信念状态，使立场变化可解释。

**💡 创新点**

创新点在于引入可调证据吸收（u）和先前锚定（a）两个可解释参数，并将证据抽取、评估、记忆、log‑odds 更新与生成分离，提供可审计的状态转移模型。

**🔧 技术方法**

使用技术包括结构化证据抽取与评分、冲突消解、log‑odds 累加、拉吉斯映射为 [-1,1] 立场、检索式生成等。

**📊 数据集**

使用数据集有 DEBATE（人类多轮辩论记录）、Argument Quality Ranking（高质量论点标注）、以及基于 AQR 的强度分类器。

**📈 对比分析**

方法对比：与基于提示的自我修订和 RAG+自修订基线比较；在 DEBATE 回放中与无变动和线性净证据模型对比；在不同 LLM 上 BE 展现可控立场动态，RMSE 下降约 0.02，能够更好解释人类行为。

**⚠️ 局限性**

局限性：仅处理单一命题级立场，无法捕捉多维议题和价值权衡；证据抽取与评分受训练数据偏差影响；需要针对不同情境校准 (u,a)，单一全局参数无法覆盖异质群体。

---

## 92. DrugSAGE:Self-evolving Agent Experience for Efficient State-of-the-Art Drug Discovery

**arXiv ID:** 2605.15461 | [PDF](https://arxiv.org/pdf/2605.15461v1)

**作者:** Yikun Zhang `[一作]` (Northeastern University), Wengong Jin `[通讯]` (Northeastern University)

**通讯引用:** 8795 | [OpenAlex ID](https://openalex.org/A5030957753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了Self-evolving Agent Experience框架，通过跨任务记忆积累并复用经验，能够高效构建药物发现的SOTA预测模型。

**💡 创新点**

创新点在于引入跨任务可执行技能库与统计记忆，保证MCTS搜索的回报边界，同时实现零测试时搜索的经验转移。

**🔧 技术方法**

采用LLM驱动的自主代理、MCTS+UCB搜索、技能库与经验记忆管理，以及自动文献与GitHub代码检索技术。

**📊 数据集**

使用33个药物属性预测任务（覆盖吸收、分布、代谢、排泄、毒性、结合亲和力、溶解度、脂亲性与生物活性），来自公开药物属性预测数据集。

**📈 对比分析**

与八个基线代理对比，单任务设置排名第一；跨任务评估中平均归一化得分0.935，零搜索时比基线提升10–30%。

**⚠️ 局限性**

局限性包括仅在药物发现任务验证，需评估跨领域适用性；技能重要性仍以引用量和GitHub指标衡量，可能忽略实验效果；在极大规模或极低资源情境下的效率仍待提升。

---

## 93. LEAP: Trajectory-Level Evaluation of LLMs in Iterative Scientific Design

**arXiv ID:** 2605.15341 | [PDF](https://arxiv.org/pdf/2605.15341v1)

**作者:** Marilyn Zhang `[一作]` (Pareto.ai), Mark E. Whiting `[通讯]` (Pareto.ai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在迭代科学设计任务中，本文构建了 55 个任务的评估框架，系统比较了 LLM 的学习效率，探讨了不同评价维度（指标、基准、对照）对结论的影响，并演示了基于轨迹的奖励进行离线强化学习。

**💡 创新点**

创新点在于提出了基于最佳累积面积（bsf‑AUC）作为轨迹级评估指标，首次将 LLM 与经典贝叶斯优化（GP‑UCB）进行直接对照，并通过域相关与域无关提示的对照实验揭示域先验在生物学任务中可能导致效率下降；此外，将轨迹度量作为 RL 奖励实现了性能提升。

**🔧 技术方法**

使用的技术包括：最佳累积面积（bsf‑AUC）轨迹度量、Gaussian Process Upper Confidence Bound (GP‑UCB) 传统搜索基线、对照域相关/无关提示、离线 Group Relative Policy Optimization (GRPO) 强化学习、统计检验与 Bootstrap 置信区间。

**📊 数据集**

使用的数据集由 45 个生物学任务（共 2,719 条实验观测，来源于 375 篇论文）和 10 个教育任务（来自 10 个 RCT）构成，所有任务均训练了对应的代理函数（oracle）并提供了参数空间与目标值。

**📈 对比分析**

评价方法是将 LLM 与 GP‑UCB 在相同轨迹、相同提示条件下比较；结果显示：在 53% 的生物学任务中指标选择（bsf‑AUC vs bsf‑Outcome）改变了最佳模型，且无 LLM 在 30 次迭代内超越 GP‑UCB；域无关提示在生物学任务中比域相关提示高约 10% 的 published‑best 匹配率；强化学习在 14/21 留下任务上提升了 bsf‑AUC。

**⚠️ 局限性**

局限性包括：仅针对标量结果优化的参数化任务，未覆盖跨实验推理；域相关提示效果未彻底解析，可能因表征不匹配导致；实验仅在开放权重模型（如 Llama 3.1）上验证，未测试闭源或前沿模型；安全过滤导致部分模型数据缺失；评估缺少针对文献与实验最佳差异明显的对抗性任务。

---

## 94. Position: Ideas Should be the Center of Machine Learning Research

**arXiv ID:** 2605.15253 | [PDF](https://arxiv.org/pdf/2605.15253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 95. Don't Stop Me Yet: Sampling Loss Minima via Dissipative Riemannian Mechanics

**arXiv ID:** 2605.15459 | [PDF](https://arxiv.org/pdf/2605.15459v1)

**作者:** Albert Kjøller Jacobsen `[一作]` (Technical University of Denmark), Georgios Arvanitidis `[通讯]` (Technical University of Denmark)

**通讯引用:** 121 | [OpenAlex ID](https://openalex.org/A5012875707)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于耗散动力学的离散化采样器 DiMS，能够在神经网络训练误差的最小子流形上进行采样；

**💡 创新点**

创新点在于引入速度相关的摩擦项，使动力学在有限时间内收敛到最小误差子流形，并通过物理可解释的超参数控制探索范围；

**🔧 技术方法**

使用 Riemannian 几何描述误差曲面，Lagrangian 动力学与二阶 ODE 求解，结合速度相关摩擦与随机初速度采样；

**📊 数据集**

在多种数据集上评估，包括 Snelson 回归、Banana 二分类、6 个 UCI 分类、MNIST/Fashion‑MNIST 以及扩展 MNIST/KMNIST；

**📈 对比分析**

与传统 Laplace（LA、LinLA、RLA）和深度集成等方法比较，DiMS 在 NLL、Brier、ECE 以及 OOD AUROC 上表现优于或至少不劣于现有方法，且对单一摩擦系数鲁棒；

**⚠️ 局限性**

主要局限是高维 ODE 求解成本高，依赖丰富的函数空间表达能力，且目前仅在确定性耗散模型上实现，未来可考虑随机/稀疏求解器和位置依赖摩擦。

---

## 96. One Pass Is Not Enough: Recursive Latent Refinement for Generative Models

**arXiv ID:** 2605.15309 | [PDF](https://arxiv.org/pdf/2605.15309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. Adesua: Development and Feasibility Study of an AI WhatsApp Bot for Science Learning in West Africa

**arXiv ID:** 2605.15376 | [PDF](https://arxiv.org/pdf/2605.15376v1)

**作者:** George Boateng `[一作]` (ETH Zurich), Victor Wumbor-Apin Kumbol `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了基于 WhatsApp 的 AI 教学助手 Adesua，提供直接对科学问题的生成式回答和自动化评估。

**💡 创新点**

将检索增强生成（RAG）与本地化教材与33年考试题库相结合，提供符合西非课程的即时答案和定制化测验。

**🔧 技术方法**

使用 GPT‑4 进行生成式回答，Sentence‑BERT+ElasticSearch 进行语义检索，Azure OpenAI、ElasticSearch、WhatsApp Bot 等技术栈。

**📊 数据集**

整合了1990–2023年西非高中/初中全国考试题库、28年SHS教材及经专家验证的 GPT‑4 生成答案。

**📈 对比分析**

通过6个月的可行性部署与56名加纳用户进行量化评估，答案帮助率93.75%，测验完成率较全卷测验低，系统延迟约1.8 ms。

**⚠️ 局限性**

样本量小、用户参与度低、未评估学习效果、缺乏结构化学生群体，未来需更大规模 RCT 与多模态功能。

---

## 98. Is Agentic AI Ready for Real-World Hardware Engineering? A Deep Dive with Phoenix-bench

**arXiv ID:** 2605.15226 | [PDF](https://arxiv.org/pdf/2605.15226v1)

**作者:** Qingyun Zou `[一作]` (National University of Singapore), WengFai Wong `[通讯]` (National University of Singapore)

**通讯引用:** 4310 | [OpenAlex ID](https://openalex.org/A5023989495)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Phoenix-bench，一个基于Verilator的硬件问题修复基准，包含511个同步实例，涵盖完整仓库、层级定位、可执行验证和补丁生成。

**💡 创新点**

创新点在于将真实GitHub问题与完整仓库上下文、执行级验证以及多层次信号流定位结合，揭示软件与硬件工程在定位和修复上的根本差异。

**🔧 技术方法**

采用LLM驱动的agent框架、Docker化EDA工具链（Verilator、Icarus、Surelog、Yosys）以及日志反馈机制。

**📊 数据集**

使用来自114个GitHub仓库的511条Verilog/SystemVerilog实例，并提供对应的开发者补丁、标签和测试用例。

**📈 对比分析**

通过与SWE-bench的对比以及对4个商用agent和8个开源agent的统一评测，发现软件agent在硬件上的成功率下降37-58个百分点；在验证和定位上通过日志反馈提升42-45个百分点。

**⚠️ 局限性**

局限在于仅覆盖到合成阶段、缺乏更深层次的物理实现验证，且当前agent缺乏对跨层级信号流的精准追踪能力。

---

## 99. GQA-μP: The maximal parameterization update for grouped query attention

**arXiv ID:** 2605.15290 | [PDF](https://arxiv.org/pdf/2605.15290v1)

**作者:** Kyle R. Chickering `[一作]` (UC Davis), Eric Xing `[通讯]` (Carnegie Mellon University)

**通讯引用:** 41221 | [OpenAlex ID](https://openalex.org/A5009547049)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过扩展谱范数理论，推导出大规模语言模型在宽度、深度、权重衰减和组查询注意力（GQA）等维度的最大更新参数化（μP）缩放规则，并在此基础上验证学习率与权重衰减的跨模型尺度迁移。

**💡 创新点**

创新点在于①将谱范数条件从经验推断升级为特征学习的定义；②提出期望算子范数以解决低秩权重矩阵（如GQA）导致的谱范数失效；③首次给出GQA的完整μP缩放规则，并结合深度与权重衰减推导与Complete‑P一致的尺度。

**🔧 技术方法**

主要技术包括随机矩阵理论、Tensor Programs框架、谱范数与期望算子范数的数学推导、AdamW优化器的权重衰减分析，以及通过坐标检查验证特征学习的实验方法。

**📊 数据集**

实验以大规模文本预训练数据为基础，训练多种规模（26M–177M）Transformer模型，使用10 tokens/parameter的训练目标，并对不同KV头数(r)的GQA实现进行对比。

**📈 对比分析**

与标准参数化、传统Adam实现比较，μP+GQA实现实现了更低的学习率与权重衰减方差，提升了跨尺度迁移的稳健性；在学习率、权重衰减、τ_epoch等指标上表现出显著更小的波动。

**⚠️ 局限性**

限制包括：GQA迁移时噪声增大；理论推导基于大幅度渐近假设，实际r值通常不至于足够大；未在极大规模（百亿级）模型上验证，且仅针对AdamW和其变体，缺乏对其他优化器或稀疏模型的推广。

---

## 100. Using the Open Science Data Federation for data distribution: Big Bear Solar Observatory use case

**arXiv ID:** 2605.15378 | [PDF](https://arxiv.org/pdf/2605.15378v1)

**作者:** Sydney Montiel `[一作]` (Instituto Politécnico Nacional), Fabio Andrijauskas `[通讯]` (University of California San Diego)

**通讯引用:** 24 | [OpenAlex ID](https://openalex.org/A5058079978)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过构建OSDF网络，获取并分发BBSO太阳观测数据，建立全球分布式缓存与原点，实现“随时随地任何数据”访问，并构建图像处理流水线进行太阳风暴图像处理。

**💡 创新点**

将StashCache扩展为OSDF，新增20个原点30个缓存，采用Pelican、XRootD、GeoIP、容器化Kubernetes等技术；将BBSO数据集整合进OSDF，实现全球用户可即时获取并本地缓存；展示一条完整的从观测到处理再返回的闭环数据流。

**🔧 技术方法**

XRootD、Pelican、GeoIP、容器化、Kubernetes、FITS格式、光扩散滤波器、阈值分割、结构提取与标记、监控与计费工具。

**📊 数据集**

BBSO太阳观测的FITS格式图像数据（Hα波段），包含太阳风暴和日冕等特征。

**📈 对比分析**

通过将数据先同步至OSDF，然后利用最近缓存进行读取，显著减少传输延迟和网络负载；实验中图像处理通过扩散滤波后边界清晰，整体性能提升可在全球范围内实时访问；但未给出具体吞吐量或延迟数值，仅表明“高效”。

**⚠️ 局限性**

缺乏量化的性能基准，未讨论大规模并发访问下的缓存一致性与带宽瓶颈；对网络拓扑变化和数据一致性机制说明不足；仅以BBSO为例，泛化到其他数据集的适用性仍待验证。

---

## 101. Automatic Construction of a Legal Citation Graph from 100 Million Ukrainian Court Decisions: Large-Scale Extraction, Topological Analysis, and Ontology-Driven Clustering

**arXiv ID:** 2605.15362 | [PDF](https://arxiv.org/pdf/2605.15362v1)

**作者:** Volodymyr Ovcharov `[一作]` `[通讯]` (LEX AI LLC), Volodymyr Ovcharov (LEX AI LLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对乌克兰EDRSR数据库中的100.7 M份判决全文进行大规模正则式提取，构建502 M条判决–法规引用边，形成判决-法规双向网络，并对其拓扑、社区和时间演化进行全方位分析，最终自动重建基于司法实践的法律域本体。

**💡 创新点**

①首次在大陆法体系实现亿级判决级别引用提取；②利用引用网络自动识别法律域边界（Louvain社区）并验证其与传统法律域的一致性；③证明引用特征可用来以近乎完美的AUC 0.9984预测未来立法重要性。

**🔧 技术方法**

采用正则表达式快速提取管线（Python多进程 + PostgreSQL服务器端游标）；网络分析使用Louvain社区检测、幂律分布拟合、PageRank、HITS、熵等度量；逻辑回归模型用于预测重要法规。

**📊 数据集**

主要数据集为乌克兰统一国家法院决定注册簿（EDRSR）100.7 M份判决全文（99.5 M可全文），结合Verkhovna Rada立法数据库，覆盖18.4 M条法规条文。

**📈 对比分析**

与已有研究对比，本网络规模比美国最高法院、荷兰、丹麦等研究大3–4个数量级；幂律指数α = 1.57低于US Supreme Court（≈2.1）；引用预测模型AUC 0.9984、P@100 = 0.65，显著优于仅基于频率的基线。

**⚠️ 局限性**

正则式提取的召回率仅约86%（缺失非规范引用、口头引用、法规外文件等）；未完成人工标注召回评估；OCR误差和旧版判决中文本缺陷导致部分引用被漏检。

---

## 102. Unified High-Probability Analysis of Stochastic Variance-Reduced Estimation

**arXiv ID:** 2605.15388 | [PDF](https://arxiv.org/pdf/2605.15388v1)

**作者:** Zhankun Luo `[一作]`, Abolfazl Hashemi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一套统一的递归随机估计器框架，并给出了其在高概率下的误差上界。

**💡 创新点**

创新点在于将记忆保持、重置概率和迭代修正三大设计原则统一成一个递推公式，并通过新的无维向量 Freedman 不等式在任意光滑诺曼空间（包括 Banach 空间）上实现高概率收敛分析；同时首次将该框架用于期望约束问题，显著将 oracle 复杂度从 O(ε⁻⁴) 降到 O(ε⁻³)。

**🔧 技术方法**

核心技术包括：统一递推式的设计、条件子高斯噪声假设、可预测的偏差-方差分解、随机停时遮蔽技巧、向量 Freedman 及其在随机梯度、约束估计等场景中的应用；以及镜像梯度（mirror descent）在 Banach 空间上的高概率分析。

**📊 数据集**

该工作为纯理论研究，无实验或数据集；所有结果均以 oracle 复杂度和高概率界定量形式给出。

**📈 对比分析**

通过与已有方法（如 SGD-M、STORM、PAGE、SPIDER 等）在 oracle 复杂度与置信水平方面的对比，证明在相同的 sub-Gaussian 假设下，本文方法在无约束问题上可达到 O(ε⁻³) 的主导复杂度，在约束问题上同样实现 O(ε⁻³) 的 oracle 复杂度，优于之前的 O(ε⁻⁴)。

**⚠️ 局限性**

局限性包括：仅在 sub-Gaussian 误差假设下成立；缺乏数值实验验证；对高阶估计器的实现与计算成本未给出具体细节；在某些特定场景（如非光滑或高维稀疏结构）可能需要进一步适配。

---

## 103. A3D: Agentic AI flow for autonomous Accelerator Design

**arXiv ID:** 2605.15237 | [PDF](https://arxiv.org/pdf/2605.15237v1)

**作者:** Abinand Nallathambi `[一作]` (Purdue University), Anand Raghunathan `[通讯]` (Purdue University)

**通讯引用:** 19973 | [OpenAlex ID](https://openalex.org/A5065766721)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 A3D，一个基于代理的 AI 工作流，实现从应用代码到硬件加速器的端到端自动化。

**💡 创新点**

首次将多代理、专门工具、验证循环结合，完成工作负载分析、瓶颈定位、代码重构、HLS 合成和设计空间探索的全链路自动化。

**🔧 技术方法**

采用大型语言模型（Claude Sonnet 4.5、Qwen3.5‑27B）、RAG 检索、Catapult HLS、AST 代码处理工具以及自研的 Agentic 框架。

**📊 数据集**

使用两款科学计算应用 LAMMPS 和 QMCPACK 的完整代码库及其代表性工作负载。

**📈 对比分析**

通过与 NVIDIA Jetson AGX Orin 边缘 GPU 对比，A3D 生成的加速器在功耗（1.04 W vs 3.6 W）、能耗（8.2 mJ vs 7.6 mJ）和面积（49 k µm²）方面分别比 GPU 优越 3.5×、2.6×、1.2×，且所有步骤均实现无人工干预。

**⚠️ 局限性**

对极复杂的多级内存和并行结构支持仍有限；GPU 原生 CUDA 核的自动转换成功率不高，且整体执行时间受 LLM 推理及工具调用延迟影响。

---

## 104. Agent4POI: Agentic Context-Conditioned Affordance Reasoning for Multimodal Point-of-Interest Recommendation

**arXiv ID:** 2605.15203 | [PDF](https://arxiv.org/pdf/2605.15203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 105. SMCEvolve: Principled Scientific Discovery via Sequential Monte Carlo Evolution

**arXiv ID:** 2605.15308 | [PDF](https://arxiv.org/pdf/2605.15308v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 106. Fortress: A Case Study in Stabilizing Search Recommendations via Temporal Data Augmentation and Feature Pruning

**arXiv ID:** 2605.15299 | [PDF](https://arxiv.org/pdf/2605.15299v1)

**作者:** Milind Pandurang Jagre `[一作]` (Apple), Kailash Thiyagarajan `[通讯]` (Apple)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Fortress框架，通过识别和修剪导致预测不稳定的特征，增强模型的稳定性和准确性。

**💡 创新点**

创新点在于利用历史快照数据进行时间数据增强和特征修剪，以提高模型的预测一致性和稳定性。

**🔧 技术方法**

使用了XGBoost分类器，结合了基于LLM和DistilBERT的语义特征以及基于用户参与的特征。

**📊 数据集**

使用了来自大型应用市场的查询-应用对的数据集，数据集包含多个时间快照。

**📈 对比分析**

与单快照方法相比，Fortress方法在PR-AUC和预测稳定性（CV）上均有显著提升，PR-AUC提高了0.24，CV降低了3.44%。

**⚠️ 局限性**

限制在于某些在单一快照中表现良好的特征在跨时间评估时可能会降低性能，且LLM特征的覆盖率有限。

---

## 107. FINESSE-Bench: A Hierarchical Benchmark Suite for Financial Domain Knowledge and Technical Analysis in Large Language Models

**arXiv ID:** 2605.15482 | [PDF](https://arxiv.org/pdf/2605.15482v1)

**作者:** Dmitry Stanishevskii `[一作]` (Lime FinTech), Andrei Kalmykov `[通讯]` (Lime FinTech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了FINESSE‑Bench，一个由八个专业化数据集组成的分层金融能力评估工具，涵盖从基础财务知识到专业级金融分析、技术分析、衍生品交易及俄语奥林匹克题目。

**💡 创新点**

创新点包括：① 明确的难度层级设计（CFA‑like 1→2→3 等），② 多领域覆盖（报告、技术分析、交易场景和多语种）以及③ 统一的评估协议（多选、数值、短答及开放式答案，采用LLM‑as‑Judge 自动打分）。

**🔧 技术方法**

使用技术包括：LLM‑as‑Judge 评分框架、统一固定提示模板、确定性推理（温度0.0）、自助采样计算 95% 置信区间、以及通过 bootstrap 对不同组别做差异统计。

**📊 数据集**

数据集：CFA‑like Level 1/2/3、CMT‑like Level 2、CFTe‑like Level 1、VLigaBench‑ru、Trading_TA、Trading_derivatives，共 3,993 道题，涵盖多选、数值、短答、俄语题。

**📈 对比分析**

比较方法：按总准确率、各组别聚合分数和差距指标（Δ_public→exam、Δ_public→ta、Δ_L1→L2 等）进行排名；结果显示顶尖模型如 Qwen3.5‑Plus‑02‑15 在专业组表现最好，且不同组别间的排序会显著变化，证明该套件对模型差异具有更细粒度的诊断能力。

**⚠️ 局限性**

局限性：题目来源未完全追溯，存在潜在训练集污染风险；评估中多选题可能低估真实推理难度；仅覆盖部分金融领域（缺乏合规、银行风险、保险等模块）；数据按非商业许可发布，需谨慎使用。

---

## 108. Interpreting De Finetti's theorem in the Category of Integrable Cones (long version)

**arXiv ID:** 2605.15402 | [PDF](https://arxiv.org/pdf/2605.15402v1)

**作者:** Crubillé Raphaëlle `[一作]` `[通讯]` (Aix Marseille University), Crubillé Raphaëlle (Aix Marseille University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文通过构造两类 draw‑and‑delete 链（分别来自 Jacobs‑Staton 的分类 De Finetti 定理与 Melliès 等人的自由指数构造）在可积分锥（integrable cones）范畴内建立了形式化联系，并利用该联系证明任意可数离散测度空间 X 的可积分锥 X^ 是其自由交换共模，进而给出 X^ 在 !X^ 中的元素（即总元素）可完全由概率测度混合的推广（promotion）所描述。

**💡 创新点**

创新点在于：①将两种看似不同的概率范畴方法（Markov 类别的 De Finetti 与线性逻辑的指数）统一到可积分锥这一更通用的语义框架；②利用等价的 draw‑and‑delete 链和有限对称子等价性证明 X^ 是自由交换共模；③得到 X^ 与 !X^ 的单射 ι，并证明所有总元素正是连续混合推广的形式，完成了对 !X^ 总元素的完整表征。

**🔧 技术方法**

主要技术手段包括：类别理论（对称单模、等价器、范畴同余）、Markov 类别与可积分锥之间的嵌入、等价子等价性与张量积的交换、抽象链构造与极限的可交换性、De Finetti 定理在可积分锥中的证明、以及利用测度论中的 Kolmogorov 扩展与 Radon–Nikodym 分解完成相关映射的构造。

**📊 数据集**

该工作为纯理论研究，未使用具体数据集，所有证明均基于范畴论与测度论的抽象构造。

**📈 对比分析**

由于没有实验评估，本文没有使用传统的基准比较或性能度量；主要是通过范畴论证明与等价性来验证结果的正确性和完整性。

**⚠️ 局限性**

限制与未解决问题包括：①证明仅适用于可数离散测度空间 X，尚未推广到所有可积分锥对象；②可积分锥中缺乏已知的张量积结构，限制了对更一般对象的指数构造的直接应用；③实现指数指数的层级构造在更广泛范畴（如连续概率）中仍为开放问题。

---

## 109. SkiP: When to Skip and When to Refine for Efficient Robot Manipulation

**arXiv ID:** 2605.15536 | [PDF](https://arxiv.org/pdf/2605.15536v1)

**作者:** Mingtong Dai `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Xinyu Wu `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种通过重标记行为克隆目标实现跳过低信息段、聚焦关键段的Skip Policy (SkiP) 方法

**💡 创新点**

创新点在于仅通过改变监督目标实现时序自适应控制，无需额外层次规划或结构；同时提出Motion Spectrum Keying (MSK) 的频域分段方法用于自动标注关键段

**🔧 技术方法**

采用离散余弦变换 (DCT) 计算动作信号的高频能量比，结合弯曲度评分对演示进行分段；在SkiP中使用标准行为克隆损失对重标记目标进行训练，兼容多种网络架构（Transformer、Diffusion、DP3等）

**📊 数据集**

在RLBench 60、RLBench 10、RoboMimic、RoboTwin 8个仿真任务以及3个真实机器人桌面任务上进行评估，使用 60 任务的 Franka Panda、4 目相机等环境；对比基线包括 DP、ACT、CoA、KF-only 等

**📈 对比分析**

与基线相比，SkiP 在 60 任务中平均成功率提升约 0.15，执行步数平均减少 15–40%（即 Steps_succ 下降 30–40%），在 RoboMimic、RoboTwin 以及真实机器人任务上亦保持或提升成功率，证明方法在多种架构与观测模态下均具备可迁移性

**⚠️ 局限性**

局限性包括：1）需预先获取绝对目标（即关键段边界的动作），无法直接处理相对目标或不连续演示；2）关键段分割依赖 MSK 参数（窗口大小、阈值）且对不同任务可能需要微调；3）当演示本身无明显高频或弯曲特征时，MSK 可能误判，导致跳过过度或不足；4）在极端噪声或未见过的操作中，跳过策略可能失效。

---

## 110. FFAvatar: Few-Shot, Feed-Forward, and Generalizable Avatar Reconstruction

**arXiv ID:** 2605.15320 | [PDF](https://arxiv.org/pdf/2605.15320v1)

**作者:** Thuan Hoang Nguyen `[一作]` (Snap Inc), Jian Wang `[通讯]` (Snap Inc)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种多视角、可一次性推断的3D Gaussian头像重建框架（FFA），能够仅用少量无姿态的面部图像在几秒钟内快速生成高质量可动画化的头部模型，并支持可选的快速个性化微调。

**💡 创新点**

核心创新包括三阶段训练策略（大规模视频预训练、多视角微调、可选个性化），端到端的FLAME参数估计器，和Multi-View Query-Former将多视角信息融合为统一的Canonical Gaussian表示。

**🔧 技术方法**

技术手段涵盖3D Gaussian splatting、ViT+MLP的FLAME估计器、跨视角注意力的Query-Former、可微分渲染、对抗损失与few-to-many学习目标等。

**📊 数据集**

使用了大规模MFHQ-1M（约100万身份、每人8帧）视频数据进行预训练，Ava256高质量多视角捕获数据进行微调，以及NeRSemble数据集用于评估。

**📈 对比分析**

与GAGAvatar、LAM等单视角基线对比，FFA在NeRSemble上PSNR提升约5.5、SSIM提升、LPIPS下降、CSIM提升；4视角输入进一步提升PSNR>+1；单推断时间<10秒，动画速率可达49FPS。

**⚠️ 局限性**

局限性包括受限于FLAME模型空间，缺乏眼球运动、口腔内部、舌头等细节；稀疏视角时可能无法捕获头发、颈部或服装边界，导致填补伪影；无个性化时细节平滑，需额外微调。

---

## 111. Minerva-Ego: Spatiotemporal Hints for Egocentric Video Understanding

**arXiv ID:** 2605.15342 | [PDF](https://arxiv.org/pdf/2605.15342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 112. Quantization Undoes Alignment: Bias Emergence in Compressed LLMs Across Models and Precision Levels

**arXiv ID:** 2605.15208 | [PDF](https://arxiv.org/pdf/2605.15208v1)

**作者:** Plawan Kumar Rath `[一作]` (Meta), Rahul Maliakkal `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了什么：对三种指令调优的大型语言模型，在五个不同精度（BF16、8、6、4、3 位）下进行后训练权重量化实验，利用 Bias Benchmark 的模糊条件数据评估量化对偏差放大和推理不确定性的影响，并提出了项级转移分析来揭示聚合指标无法捕捉的新偏差。

**💡 创新点**

创新点是什么：首次将项级转移分析与未知选择率结合，揭示量化强度呈剂量-反应关系的偏差出现；证明压缩削弱模型的推理不确定性，导致其重返预训练时期的刻板先验；并展示聚合指标（如困惑度）对质量退化的误导性。

**🔧 技术方法**

用了什么技术：采用后训练权重量化（MLX 框架）在四个低比特宽度下压缩模型，并使用卡方检验、Cohen's h 效应量、逻辑回归等统计方法对结果进行量化分析；通过多随机种子产生的多样化推理输出实现项级一致性评估。

**📊 数据集**

用了什么数据集：使用 Bias Benchmark for Question Answering（BBQ）的模糊条件，共 12,148 条样本，涵盖年龄、性别身份、种族/民族、宗教和社会经济状态五个偏差类别。

**📈 对比分析**

如何比较的方法，性能怎么样：将每个压缩版本与 BF16 基线在 Stereotype Reliance Score（SRS）、Unknown Selection Rate（USR）以及困惑度（perplexity）进行对比；发现 3 位量化导致 6–21% 原无偏项出现偏差，USR 下降 17.4%，但困惑度仅上升 ≤3%，表明聚合质量指标无法预见质量退化。

**⚠️ 局限性**

limitation是什么：实验仅覆盖单一权重量化方案（MLX），未考察激活量化或其他框架；使用 Apple Silicon 硬件，缺乏 GPU 等多平台验证；参数规模与架构混杂导致对规模效应的解释受限；未评估不同温度、贪婪解码或其他采样策略对结果的影响。

---

## 113. Privacy Evaluation of Generative Models for Trajectory Generation

**arXiv ID:** 2605.15246 | [PDF](https://arxiv.org/pdf/2605.15246v1)

**作者:** Stavros Bouras `[一作]` (National Technical University of Athens), Konstantinos Tserpes `[通讯]` (National Technical University of Athens)

**通讯引用:** 4607 | [OpenAlex ID](https://openalex.org/A5012328550)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6215c339-3735-4be3-8a07-5bbb7004712d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过对四个代表性生成轨迹模型实施成员推理攻击，评估了生成轨迹模型的隐私泄漏风险。

**💡 创新点**

首次系统识别并映射了轨迹生成领域的隐私评估方法，指出现有文献中缺失隐私评估的显著空白，并验证了成员推理攻击的可行性。

**🔧 技术方法**

使用生成对抗网络（GAN）和扩散模型，结合鉴别器基和损失基成员推理攻击技术。

**📊 数据集**

使用了 Foursquare NYC、GeoLife、DiDi Chengdu 三个公开轨迹数据集。

**📈 对比分析**

与未进行隐私评估的模型对比，发现 MoveSim 的 AUC‑ROC 达到 0.70，显著高于随机猜测，表明存在成员泄漏；其余模型表现接近 0.5。

**⚠️ 局限性**

实验仅考虑白盒成员推理攻击，未覆盖其他攻击方式或多种隐私评估方法；结果对模型和攻击方法高度敏感，缺乏全面评估。

---

## 114. Eskwai for Students: Generative AI Assistant for Legal Education in Ghana

**arXiv ID:** 2605.15380 | [PDF](https://arxiv.org/pdf/2605.15380v1)

**作者:** George Boateng `[一作]` (ETH Zurich), Victor Wumbor-Apin Kumbol `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发并部署了 Eskwai for Students，一款面向加纳法学院学生的生成式 AI 辅助工具；

**💡 创新点**

创新点在于构建了基于 12K 案例和 1.4K 立法的检索增强生成（RAG）系统，并在 2.5 年内大规模真实环境中进行纵向评估；

**🔧 技术方法**

采用了检索增强生成技术，检索层使用开源嵌入模型和 ElasticSearch，生成层使用 OpenAI GPT API；

**📊 数据集**

数据集为从加纳律师处收集、整理的 12,000 条案例文本和 1,400 条立法文本，按 5 句块分割并嵌入；

**📈 对比分析**

评估方法为用户交互统计和有投票者的“有用性评分”，结果显示 68.4% 的有用性，平均响应时延 7.1 秒；

**⚠️ 局限性**

局限性包括未测评学习效果、投票率低、数据库更新不及时及潜在的学术诚信风险。

---

## 115. Verifiable Agentic Infrastructure: Proof-Derived Authorization for Sovereign AI Systems

**arXiv ID:** 2605.15228 | [PDF](https://arxiv.org/pdf/2605.15228v1)

**作者:** Jun He `[一作]` (OpenKedge.io), Deying Yu `[通讯]` (OpenKedge.io)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了分布式信任框架（DTF），通过结构化证明、共识验证、执行身份和证据链，实现对自主 AI 代理的治理，使高风险操作在可验证、可追溯的授权生命周期内执行。

**💡 创新点**

创新点在于：①把授权主体从身份迁移到可验证的证明；②引入多类评估者共识验证机制，确保单一评估器失效不导致误授权；③定义可衍生的执行身份与短期凭证；④构造不可篡改的证据链，实现完整的授权重放与审计。

**🔧 技术方法**

技术实现包括：结构化证明（Justification Proof）、多方评估者（政策、状态、风险、模拟、人工）共识模型、短期凭证派生（如 AWS STS）与执行身份、Append‑only 证据链（如 Amazon QLDB），以及 OpenKedge 作为意图治理子系统。

**📊 数据集**

实验使用合成云端工作负载，包含 10,000 次代理变更（7,600 合法、2,400 非法），模拟多地区云部署，涵盖资源终止、配置变更与紧急 break‑glass 等场景。

**📈 对比分析**

对比基线 IAM 与仅策略预检，DTF 在所有非法变更中 100% 阻断，权限暴露平均减少 99.7%，证据完整率 100%，回放成功率 99.9%。平均决策延迟 58 ms，p99 延迟 171 ms，已证明可接受的高风险操作延迟。

**⚠️ 局限性**

限制：依赖子系统正确性与上下文完整性；评估器多样性与独立性不足时安全性下降；无法抵御评估器集体破坏；适用于高风险操作，低风险场景需平衡成本与复杂度。

---

## 116. Clique-width and induced topological minors

**arXiv ID:** 2605.15453 | [PDF](https://arxiv.org/pdf/2605.15453v1)

**作者:** Paweł Rafał Bieliński `[一作]` (Warsaw University of Technology), Paweł Rzążewski `[通讯]` (Warsaw University of Technology)

**通讯引用:** 552 | [OpenAlex ID](https://openalex.org/A5047941027)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文证明了一个图类的无诱导细分（induced topological minor）H具有有界clique-width当且仅当H是P4、爪（paw）或菱形（diamond）的诱导子图。

**💡 创新点**

给出了针对诱导拓扑最小子图与clique-width关系的完整分类，解决了Dabrowski、Johnson和Paulusma提出的未解问题。

**🔧 技术方法**

通过结合已知的H-free、H-induced-minor-free及块-海绵（block‑cactus）图的结构定理，利用墙图和其线图的clique-width无界性进行反证，完成了理论证明。

**📊 数据集**

无数据集，论文完全基于纯理论推导。

**📈 对比分析**

无实验或算法实现，所有结论均通过数学归纳与构造性论证得出，没有性能比较。

**⚠️ 局限性**

局限在仅处理无向简单图，且结论不直接适用于带权、有向图或其他宽度参数；后续研究可探讨更一般的情况。

---

## 117. MuteBench: Modality Unavailability Tolerance Evaluation for Incomplete Multimodal Fusion

**arXiv ID:** 2605.15235 | [PDF](https://arxiv.org/pdf/2605.15235v1)

**作者:** Wugeng Zheng `[一作]` (University of Central Florida), Song Wang `[通讯]` (University of Central Florida)

**通讯引用:** 7042 | [OpenAlex ID](https://openalex.org/A5100326206)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出并实现了一个跨多模态融合架构鲁棒性评估基准，系统评估了不同融合模型在两种常见缺失模式（模态缺失与时间段缺失）下的表现；

**💡 创新点**

首次将多模态融合架构、9个临床数据集和两种缺失模式结合，提供可控严重度的统一基准，并揭示架构族、数据集结构对鲁棒性的决定性影响；

**🔧 技术方法**

使用多种融合技术（channel‑independent、Mixture‑of‑Experts、shared‑specific）、统一的缺失注入库、score‑based diffusion 进行缺失填补，并通过AUROC、Macro‑F1等指标评估；

**📊 数据集**

涵盖9个多模态临床数据集（HAR‑UP、PTB‑XL、Chapman‑Shaoxing、Sleep‑EDF、PPG‑DaLiA、WESAD、MIMIC‑IV、Challenge‑2012、CirCor），覆盖7个医学领域、不同通道数与序列长度；

**📈 对比分析**

通过在相同缺失模式、相同严重度下多次重复实验（共810次），比较模型在完整、模态缺失、时间段缺失三种情形下的AUROC/F1；发现 channel‑independent 模型在模态缺失下最稳健，shared‑specific 在时间段缺失下最优，MoE 模型相对脆弱；Curriculum dropout 仅在训练时覆盖的缺失率内有效；Diffusion 填补能显著提升时间段缺失下的性能，尤其对对输入敏感的 MoE 模型；

**⚠️ 局限性**

仅评估了6种模型，缺失模式未覆盖部分缺失等更细粒度情况；Diffusion 填补仅在 PTB‑XL 上验证，通用性待进一步验证；基准覆盖的临床领域有限，未来需扩展更多多模态数据与缺失策略。

---

## 118. MorphoHELM: A Comprehensive Benchmark for Evaluating Representations for Microscopy-Based Morphology Assays

**arXiv ID:** 2605.15383 | [PDF](https://arxiv.org/pdf/2605.15383v1)

**作者:** Emre Hayir `[一作]` (Microsoft Research New England), Alex X. Lu `[通讯]` (Microsoft Research New England)

**通讯引用:** 1435 | [OpenAlex ID](https://openalex.org/A5058985468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了一个统一、开放的基准框架，用于评估细胞染色（Cell Painting）图像的特征提取方法，并针对不同程度的批量效应（技术噪声）进行系统评测。

**💡 创新点**

创新点在于：①将已有的多种生物学信号检索策略（MoA、基因通路富集、复制检索）统一到一个标准化评测流程；②修正了此前指标实现中的缺陷（如无穷大 odds ratio 的处理）；③首次在同一基准中引入四级批量效应严格度（NR、NSB、NSS、NSL），量化模型在技术噪声增加时性能衰退；④对比了从经典手工特征到自然图像预训练、显微镜专用预训练以及多模态模型在同一评测下的表现。

**🔧 技术方法**

技术手段包括：多种特征提取器（CellProfiler, ImageNet ResNet101, 随机 ResNet, DINOv2 CLS/patch, OpenPhenom, SubCell, CLOOME），统一的图像预处理与通道校正，严格的样本质量控制，后处理管线（Plate-wise Center Scaling + PCA + MAD 归一化），以及多种评价指标（几何平均 odds ratio、显著检验比例、kNN recall@1、mAP、NegCon mAP）。

**📊 数据集**

使用的公开数据集有 BBBC036、JUMP-Cell Painting 的四个子集：cpg-MoA（化学 MoA 注释），cpg-CRISPR（基因敲除通路注释），cpg-target2（约 300 具已知显著效应化合物），cpg-compound（约 30k 化合物的高通量子集）。

**📈 对比分析**

对比方法时，作者在每个批量效应级别下计算指标，并绘制性能衰退曲线。结果显示：自然图像预训练模型 DINOv2 在大多数任务和批量效应级别下表现最佳，但不同任务间存在权衡；CellProfiler 在复制检索任务中最突出；所有模型在跨实验室（NSS）时性能接近随机，表明跨域泛化仍差。

**⚠️ 局限性**

局限性包括：统一的后处理管线对不同模型的影响不一致，可能掩盖了某些模型的优势；评测仅覆盖 Cell Painting，未涵盖其他染色面板、无标记成像或实时成像；未探索针对各模型的最佳后处理策略；跨实验室泛化仍未解决，实际应用中可能需要更强的批量效应抑制方法。

---

## 119. CAX-Agent: A Lightweight Agent Harness for Reliable APDL Automation

**arXiv ID:** 2605.15218 | [PDF](https://arxiv.org/pdf/2605.15218v1)

**作者:** Chenying Lin `[一作]` (Shanghai Ultradimension Technology Co Ltd), Liang Yu `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 38553 | [OpenAlex ID](https://openalex.org/A5101814743)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发并评估了一种针对ANSYS MAPDL的轻量级agent harness——CAX-Agent，专注于模型驱动的错误恢复策略

**💡 创新点**

提出了分层恢复梯度（规则修补→LLM重生成→上下文增强→人工干预）并通过对比实验证明模型驱动恢复显著提升可靠性

**🔧 技术方法**

使用LLM服务（本地Qwen-27B和外部Claude Sonnet 4.6）、FastAPI路由、PyMAPDL/CLI后端以及错误日志提取与规则引擎等技术

**📊 数据集**

基于50个结构性任务（静态、模态、热分析）共150次重复实验，数据通过人工评分与系统指标记录

**📈 对比分析**

对比no_recovery、rule_only和model_only三种恢复策略，model_only完成率0.9267、总分9.16/10、零干预率0.84，显示出大幅优于其它策略的性能提升

**⚠️ 局限性**

局限包括：仅测试简单线性结构、单一后端与模型、规则集覆盖有限、实验规模有限、未对其他工具或多物理场进行验证

---

## 120. From Feedback Loops to Policy Updates: Reinforcement Fine-Tuning for LLM-Based Alpha Factor Discovery

**arXiv ID:** 2605.15412 | [PDF](https://arxiv.org/pdf/2605.15412v1)

**作者:** Lingzhe Zhang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**通讯引用:** 120896 | [OpenAlex ID](https://openalex.org/A5100391240)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于强化学习微调的自我演化量化因子发现框架 QuantEvolver，旨在通过政策更新而非 prompt 级反馈循环实现因子生成的自我改进。

**💡 创新点**

创新点在于：①将可执行量化评估直接转化为政策更新，消除上下文爆炸与反馈漂移；②采用轻量级 LLM 与 DiCo（多样性-互补）奖励机制，鼓励生成结构多样、行为互补的因子；③利用 oracle LLM 生成高质量种子并构建多场景任务银行，提供稳定的训练起点。

**🔧 技术方法**

核心技术包括：强化学习微调（PPO/GRPO）、因子领域特定语言（DSL）、DiCo 奖励（多样性、互补性、精确性约束）、Oracle LLM 生成种子、回测与评估模块、Mined Factor Database。

**📊 数据集**

实验使用三大真实市场基准：①单资产5分钟方向预测；② 高频交叉序列因子挖掘（小时再平衡）；③每日CSI300 ETF 因子挖掘。

**📈 对比分析**

与 AlphaBench、QuantaAlpha、R&D-Agent、Alpha-Jungle 四个主流 LLM 因子挖掘方法在统一后期筛选+融合协议下比较，QuantEvolver 在所有基准的主要指标（单资产方向准确率+约4.4%，高频交叉序列 RankIC +约0.025，日ETF RankIC +约0.02）均优于对照组，且在 RankIC、IC、ICIR 等辅助指标上也保持领先。

**⚠️ 局限性**

局限性包括：①奖励设计仍偏向高 IC 可能忽略长期稳健性；②对回测窗口划分敏感，需手工调参；③依赖 Oracle LLM 产生种子，种子质量影响搜索质量；④在大规模算力环境下训练仍昂贵。

---

## 121. Sound Sparks Motion: Audio and Text Tuning for Video Editing

**arXiv ID:** 2605.15307 | [PDF](https://arxiv.org/pdf/2605.15307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 122. Kofola 1.0: A Modular Approach to ω-Regular Complementation and Inclusion Checking (Technical Report)

**arXiv ID:** 2605.15390 | [PDF](https://arxiv.org/pdf/2605.15390v1)

**作者:** Ondrej Alexaj `[一作]` (Brno University of Technology), Nicolas Mazzocchi `[通讯]` (Slovak University of Technology in Bratislava)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5081762569)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个高效、模块化的Büchi自动机补集与语言包含检查工具，利用自动机的强连通分量分解，并为每类分量应用专门的补集算法，同时引入新的初始几乎确定性接受分量(IADAC)和单一广义Rabin对接受条件，实现了即时空空检查，避免了完整乘积构造；

**💡 创新点**

创新点包括：①基于结构的分量细分与针对性补集策略；②首次定义并使用IADAC分量；③采用单一广义Rabin对接受条件并提出极度惰性的空检查算法；④整合新的启发式方法提升包含检查效率；

**🔧 技术方法**

技术手段主要有：分层SCC分解与分类（非接受、弱接受、IADAC、DAC、NAC）；针对各分量的补集算法（Ramsey、rank、slice、determinization等）组合；单一广义Rabin对接受与改进的Couvreur空检查算法；符号化模拟与子模消除；

**📊 数据集**

使用的测试数据集包括：文献中的Büchi自动机补集基准、程序终止检查基准、并发系统验证基准、超属性模型检验基准、定理证明基准等多领域实际案例；

**📈 对比分析**

实验中与多种最先进工具（Ramsey-based、rank-based、determinization-based、早期分解实现等）进行对比，结果显示本文工具在已解决实例数量、补集规模及运行时间上均优于竞争者，某些基准上提升了数个数量级；

**⚠️ 局限性**

局限性在于：目前仅支持Büchi自动机；对高度非确定性或极大规模自动机仍可能遭遇状态空间爆炸；对某些特殊接受条件（如ω-regular的更一般形式）尚未实现；

---

## 123. Reasoning Models Don't Just Think Longer, They Move Differently

**arXiv ID:** 2605.15454 | [PDF](https://arxiv.org/pdf/2605.15454v1)

**作者:** Anders Gjølbye `[一作]` (Technical University of Denmark), Sanmi Koyejo `[通讯]` (Stanford University)

**通讯引用:** 7489 | [OpenAlex ID](https://openalex.org/A5076316802)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了推理训练的语言模型在链式思考生成过程中的隐藏状态轨迹几何，并通过长度校正揭示了难度与轨迹形状的真实关联。

**💡 创新点**

创新点在于发现生成长度是轨迹几何的结构性混淆因素，提出对长度进行残差校正后重新评估难度-几何耦合，并发现推理训练在竞赛编程域中显著改变轨迹走向。

**🔧 技术方法**

采用IRT难度校准、残差长度校正、轨迹直接性（directness）与曲率变异性（curvature variability）统计、线性探针和行为注释等技术，对隐藏状态轨迹进行定量分析。

**📊 数据集**

使用了 500 条 Codeforces 竞赛编程题、500 条 MATH 数学题和 500 条 SATBench 逻辑满足性题，共 1,500 条数据集。

**📈 对比分析**

通过匹配的推理模型与指令调优基线对比，发现推理模型在纠正长度后与基线相比在 Codeforces 领域具有显著正向的直接性-难度耦合（median +0.41 vs. -0.06），而在 MATH 和 SAT 的区分度更弱，整体表现与模型训练方式密切相关。

**⚠️ 局限性**

主要限制包括：推理模型与基线的答案边界分割方法差异导致的分割偏差；长度校正方法对直接性统计的敏感性；以及无法从关联性推断因果机制，需要更细粒度或实验室验证。

---

## 124. Fault tolerance estimation in digital circuits with visualised generative networks

**arXiv ID:** 2605.15212 | [PDF](https://arxiv.org/pdf/2605.15212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 125. On the Stability of Growth in Structural Plasticity

**arXiv ID:** 2605.15435 | [PDF](https://arxiv.org/pdf/2605.15435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 126. Hidden in Memory: Sleeper Memory Poisoning in LLM Agents

**arXiv ID:** 2605.15338 | [PDF](https://arxiv.org/pdf/2605.15338v1)

**作者:** Sidharth Pulipaka `[一作]` (SPAR), Mario Fritz `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实证了在具有持久记忆的LLM助手中，攻击者可以通过外部内容（如文档、网页、仓库）注入伪造的用户记忆，并在后续会话中被检索和利用，从而影响助手行为的“睡眠式记忆中毒”攻击。

**💡 创新点**

①提出“睡眠式记忆中毒”概念，①阐述其跨会话、延迟影响的独特风险；②设计可复用的通用注入模板和记忆重写机制，使攻击能在多种外部上下文中复现；③在多个模型和记忆管理框架上系统评估，证明攻击的普遍性和严重性；④探索提示硬化与检测等防御思路。

**🔧 技术方法**

①Actor–Critic搜索驱动的通用注入模板优化；②基于嵌入相似度的记忆重写算法；③语义检索与LLM外部记忆管理的检索实现；④提示硬化、极端亮点包装和文档扫描器等防御技术；⑤使用LLM判别器评估注入、检索与使用成功率。

**📊 数据集**

1) 注入数据集：700个（文档、目标）对，涵盖新闻、法律、代码、科学、专利、金融等15个来源，分行为与Agent Action两子集；2) 后注入评估集：400个目标（200行为，200 Agent Action），用于测试检索与使用；3) 公开仓库与示例脚本用于实验重现。

**📈 对比分析**

与现有User Review基线对比，Actor–Critic在工具式记忆写入模式下在GPT‑5.4/5.5上实现近100%注入率；检索率在目标相关查询上高达90–95%，远低于无关查询；使用率在相关查询中达到42–85%（行为）和60–89%（Agent Action）。端到端成功率在单次攻击下为41–73.9%（行为）和3–66%（Agent Action）。提示硬化在Claude‑Sonnet‑4.6和Gemini‑3.1上可将注入率降至零，但对其他模型及自适应攻击效果不稳。

**⚠️ 局限性**

①攻击效果高度依赖记忆写入与检索机制（工具式 vs 外部管理）以及模型安全训练；②现有提示硬化在面对自适应攻击时易失效，未能提供跨模型的鲁棒防御；③实验使用的注入和检索数据仍有限，可能低估实际场景中的多样性；④未针对多轮对话中的记忆持续更新和冲突冲刷机制进行深入研究。

---

## 127. SurvivalPFN: Amortizing Survival Prediction via In-Context Bayesian Inference

**arXiv ID:** 2605.15488 | [PDF](https://arxiv.org/pdf/2605.15488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 128. Hydra: Efficient, Correct Code Generation via Checkpoint-and-Rollback Support

**arXiv ID:** 2605.15238 | [PDF](https://arxiv.org/pdf/2605.15238v1)

**作者:** Alexander Du `[一作]` (Duke University), Matthew Lentz `[通讯]` (Duke University)

**通讯引用:** 291 | [OpenAlex ID](https://openalex.org/A5000778364)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Hydra，一个支持异步编译器检查与检查点回滚的代码生成运行时，使LLM在生成过程中能即时检测并修复静态错误。

**💡 创新点**

创新点在于将生产编译器（如Clang）改造成可增量检查并支持检查点，结合异步检查与策略化回滚，实现高效错误定位与修复，显著降低生成延迟和token消耗。

**🔧 技术方法**

采用的技术包括异步编译器检查、进程级检查点（fork）、检查点恢复、搜索树状态管理、可编程策略接口、Clang前端改造、vLLM推理服务器、贝叶斯根因估计等。

**📊 数据集**

使用的评测数据集为约1016个来自LiveCodeBench和LiveCodeBench-Pro的C/C++任务，模型为Qwen2.5‑Coder‑32B和gpt‑oss‑120B；在TypeScript上亦使用Mündler等基准。

**📈 对比分析**

与传统的后置修复（post‑hoc repair）和增量检查（constrained semantic decoding）以及TypeScript上的CSD对比，Hydra在遇到静态错误的任务中C/C++上平均降低约71%延迟、70% token 消耗；在TypeScript上保持或略低延迟且提升静态/功能正确率。

**⚠️ 局限性**

局限性包括需要对编译器进行改造、增量检查支持仍受限、当前策略假设简化未充分利用错误诊断信息、对较慢检查器或更大模型的收益可能有限，并且仅关注静态检查，未覆盖运行时/功能错误。

---

## 129. Zero-Shot Goal Recognition with Large Language Models

**arXiv ID:** 2605.15333 | [PDF](https://arxiv.org/pdf/2605.15333v1)

**作者:** Kin Max Piamolini Gusmão `[一作]` (Pontifical Catholic University of Rio Grande do Sul), Felipe Meneguzzi `[通讯]` (University of Aberdeen)

**通讯引用:** 3026 | [OpenAlex ID](https://openalex.org/A5073632183)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了前沿大型语言模型在PDDL基准下的零样本目标识别能力；

**💡 创新点**

创新点在于提出首个针对经典规划域的零样本目标识别评估框架，并揭示LLM在证据整合上的三种行为模式；

**🔧 技术方法**

采用结构化提示策略，直接将目标识别任务编码为概率分布输出，比较LLM与基于规划地标的符号方法；

**📊 数据集**

使用四个经典PDDL域（Blocks World、Campus、Driverlog、Dock‑Worker Robots）及其预定义的候选目标集合；

**📈 对比分析**

与地标方法对比，LLM在低观测率下表现优于符号基线，但仅部分模型能随观测量提升准确率，平均识别时间和token消耗也呈显著差异；

**⚠️ 局限性**

局限性包括：LLM对世界知识依赖过重、证据整合能力不足、存在输出格式不规范、对域名变形或噪声观测的鲁棒性待提升。

---

## 130. Designing Dense Satellite Clusters for Distributed Space-based Datacenters

**arXiv ID:** 2605.15335 | [PDF](https://arxiv.org/pdf/2605.15335v1)

**作者:** Jules Pénot `[一作]` (Massachusetts Institute of Technology), Hamsa Balakrishnan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 3832 | [OpenAlex ID](https://openalex.org/A5039967363)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出两种密集卫星集群轨道设计方案（平面与三维）用于分布式低地轨道空间数据中心，并通过整数优化将 VL2-like Clos 网络映射到物理卫星上，实现类似陆地数据中心的交换网络。

**💡 创新点**

创新点在于：1）设计出可在给定最小间距 R_min 与最大半径 R_max 条件下最大化卫星数的平面最优布局，卫星数比 Suncatcher 提案提升约 4 倍；2）提出可按 (R_max/R_min)^3 扩展的三维集群方案，显著提高规模；3）利用整数规划将 Clos 网络嵌入实际轨道网络，证明在 L≥3 层时可实现全双工等速率交换。

**🔧 技术方法**

使用了相对轨道元素（ROE）构造卫星相对轨道；数值仿真评估日照遮蔽、线视线(ISL)持续性；功率拟合与光照模型；整数规划（Gurobi）求解 Clos 网络映射；图论谱分析评估网络拓扑。

**📊 数据集**

主要数据为设计参数集合：R_min=100 m、R_max∈[500,2000] m、卫星尺寸 R_sat∈[0,15] m、ISL 最大连接数 k∈[4,12]。并通过数值实验生成对应的卫星布局与 ISL 网络。

**📈 对比分析**

对比方法：与 Suncatcher 基线平面方案、与单层及双层 Clos 网络的可行性。性能上，平面方案在给定 R_max 下可部署约 367 颗卫星，三维方案在 R_max/R_min≥13.5 时超过平面；平面方案实现刚性旋转，保证 ISL 恒定；三维方案对 R_sat 需求更严格。

**⚠️ 局限性**

限制包括：对三维集群的日照遮蔽敏感（R_sat≥3 m 产生遮蔽）；ISL 需要高精度姿态与定点；硬件实现需满足 k 最大链接数限制；对辐射、热管理、轨道碰撞风险等实际部署问题未在本文深入探讨。

---

## 131. DiffVAS: Diffusion-Guided Visual Active Search in Partially Observable Environments

**arXiv ID:** 2605.15519 | [PDF](https://arxiv.org/pdf/2605.15519v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 132. Why are language models less surprised than humans? Testing the Parse Multiplicity Mismatch Hypothesis

**arXiv ID:** 2605.15440 | [PDF](https://arxiv.org/pdf/2605.15440v1)

**作者:** William Timkey `[一作]` (New York University), Tal Linzen `[通讯]` (New York University)

**通讯引用:** 5584 | [OpenAlex ID](https://openalex.org/A5081824828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用RNNG结合词同步Beam Search系统地调节解析多样性，计算不同Beam宽度下的单词惊讶度，并用此惊讶度预测阅读时间，评估解析多样性缺失是否能解释人类在花园路径句中的处理难度。

**💡 创新点**

将解析多样性作为可调参数，并引入Forced Garden Path与Full-Parallel极限条件，首次量化解析多样性对花园路径效应的影响，检验单阶段惊讶度模型能否解释人类阅读行为。

**🔧 技术方法**

技术包括：Recurrent Neural Network Grammars (RNNG)、词同步Beam Search、线性混合效应回归与贝叶斯混合效应回归、惊讶度到阅读时间的映射函数、手动筛选解析以实现极限条件。

**📊 数据集**

使用SAP Benchmark花园路径子集（2000名参与者的自述式阅读数据）评估阅读时间，训练时使用BLLIP子集（42M）和BabyLM（100M）两大语料库，并在Penn Treebank WSJ子集上评估解析性能。

**📈 对比分析**

通过填充句子阅读时间的对数似然提升评估惊讶度预测的拟合度；对比不同Beam宽度、解析策略与极限条件下的预测花园路径效应与实验数据的幅度；结果显示，即使Beam宽度最小或极限条件下，模型预测的花园路径效应仍比人类数据小数十倍。

**⚠️ 局限性**

限制：单阶段惊讶度模型未包含语法重分析机制；Beam Search假设与人类记忆约束不完全对应；无法捕捉到对语义与语法的复杂交互；因此，即使调节解析多样性也无法让模型完全匹配人类的花园路径阅读时间。

---

## 133. Greedy or not, here I come: Language production under vocabulary constraints in humans and resource-rational models

**arXiv ID:** 2605.15365 | [PDF](https://arxiv.org/pdf/2605.15365v1)

**作者:** Thomas Hikaru Clark `[一作]` (Massachusetts Institute of Technology), Laura Nicolae `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探究在词汇受限条件下，人类如何生成符合沟通目标的语言，并与贪婪采样和基于序贯蒙特卡洛的全局最优采样算法进行对比。

**💡 创新点**

创新点在于系统评估了不同词汇限制下人类生成文本与算法生成文本的差异，揭示了人类更倾向贪婪但顶尖者会修订，并发现语义轻词在极限词汇下频率显著上升。

**🔧 技术方法**

采用的技术包括基于大型语言模型的受限生成（Awrs），自定义词汇过滤，Sequential Monte Carlo（Awrs-16/32），以及LLM‑as‑judge的自动评分。

**📊 数据集**

使用的数据集包括192个问题（Why、How、ExplainSimple、RedditELI5）以及七组词汇表（250至16000词），并对人类参与者（144人）进行在线实验。

**📈 对比分析**

通过LLM‑as‑judge评估与人类评分对比，发现人类与贪婪采样的表现相似，而Awrs在小词汇时更稳健；高分人类在约束词汇下修订次数显著增多。

**⚠️ 局限性**

主要局限在于自动评分的偏差、任务的非自然性（只能打字、不能插入替换）、样本量与多样性有限，以及对不同语言或更大词汇范围的推广性未检验。

---

## 134. Representation Without Reward: A JEPA Audit for LLM Fine-Tuning

**arXiv ID:** 2605.15394 | [PDF](https://arxiv.org/pdf/2605.15394v1)

**作者:** Biswa Sengupta `[一作]` (JPMorgan Chase & Co), Biswa Sengupta `[通讯]` (JPMorgan Chase & Co)

**通讯引用:** 6652 | [OpenAlex ID](https://openalex.org/A5055824933)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在 LoRA 微调框架下，针对 Llama‑3.2‑1B‑Instruct 进行自然语言转正则表达式（NL‑RX）生成任务，系统性评估了 22 种联合嵌入预测（JEPA）相关的训练时辅助损失和 1 种推理时投影方法。通过对隐藏状态几何（曲率、各向异性、梯度对齐）和下游精确匹配（Exact‑Match）指标进行双层诊断，检验这些辅助是否能在不改变词表概率分布的前提下提升任务性能。

**💡 创新点**

1) 形成了一个完整的辅助损失“假设图”，涵盖轨迹形状、分布约束、预测/目标不对称、Fisher‑度量残差、以及 decoder‑可见的 JEPA 目标。2) 将“结构化零假设”作为实验终点：即即使辅助在隐藏层上产生可测量几何变化，也不必然提升精确匹配性能。3) 引入多级校正（Bonferroni、Holm‑Bonferroni）和梯度余弦诊断，提供了对辅助有效性更细粒度的判定。

**🔧 技术方法**

- LoRA 微调（rank‑32）
- 训练时辅助损失实现（STP、T1–T6、T7、L1–L14 等）
- 诊断指标：隐藏状态各向异性、轨迹曲率、辅助与交叉熵梯度余弦
- 统计检验：Welch t‑检验、配对 t‑检验、家族级多重比较校正
- 数据分布切分与微调：EOS‑剪裁、margin‑hinge、投影重检

**📊 数据集**

NL‑RX‑TURK（8,000 训练 / 2,000 测试）和 NL‑RX‑SYNTH（相同规模）两大基准，均采用系统提示 + 用户描述 + 助手正则表达式的格式，评估精确匹配与前缀匹配。

**📈 对比分析**

对比基线（无辅助）进行 3 个种子训练，并计算每个辅助的平均精确匹配提升与方差。结果显示：最优单细胞提升为 +2.53%（T3‑Local）但在 Bonferroni/Holm 校正后不显著；多数辅助在 10% 置信阈值下仍落在随机噪声区间。数据效率曲线表明，T3‑Local 与多尺度 JFR 在低数据比例下可获得 1–2% 的提升，但整体仍未突破多重比较门槛。

**⚠️ 局限性**

- 仅使用 3 个随机种子，难以捕捉罕见显著效应。
- 只在 LoRA 微调条件下测试，未探索全参数微调或更大模型。
- 仅评估精确匹配，未覆盖语义等价、校准或 OOD 性能。
- 批量大小极小（4），限制了对对比学习等需要大负样本的有效性。
- 仅聚焦单向生成任务，缺乏对多模态或交互式对话场景的验证。

---

## 135. SDOF: Taming the Alignment Tax in Multi-Agent Orchestration with State-Constrained Dispatch

**arXiv ID:** 2605.15204 | [PDF](https://arxiv.org/pdf/2605.15204v1)

**作者:** Zhantao Wang `[一作]` `[通讯]` (Digital China), Zhantao Wang (Digital China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 SDOF（State-Driven Orchestration Framework），在多智能体工作流中引入状态机与意图-阶段绑定的双重防御层，实现对业务流程阶段约束和技能前置条件的严格执行，并生成可回放的审计日志。

**💡 创新点**

创新点包括：①将意图与流程阶段分离的意图‑阶段绑定（Λ）机制；②基于状态机的状态感知调度器（StateAwareDispatcher）与技能注册表（SkillRegistry）的两层预检；③通过在线RLHF（GRPO/GSPO）训练意图路由器，使模型在面临流程违规时保持零容忍；④在生产环境中与Beisen iTalent平台的实时API集成，提供端到端的可审计工作流。

**🔧 技术方法**

使用技术主要包括：LLM（Qwen2.5‑7B、Qwen3‑8B 等）+ 在线RLHF（GRPO/GSPO/DAPO）、Generative Reward Modeling（GRPO）训练意图路由器；状态机模型 GoalStage FSM；技能注册表与预置条件验证；PostgreSQL 作为持久化共享状态与审计日志；多代理架构（七个专用招聘代理）与 RESTful API 调用。

**📊 数据集**

数据集：①185个专家策划的招聘场景（882条对话、1,671条真实API调用）；②960条 SGD（Google Service Dialogue）衍生对话（8个业务域，涵盖银行、酒店、租车、娱乐等），每个域包含正常与注入式非法的对话。

**📈 对比分析**

比较方法：在 HR 场景中与 LangGraph、LangGraph+Pre、Vanilla 等基线对比，测量任务完成率（TCR）、约束违规率（CVR）、审计可追溯率（TRC）、延迟（LAT）。SDOF 在 HR 任务中实现 86.5% TCR、2.5% CVR、100% TRC、57.4 ms 延迟；在 SGD 评测中对所有 160 条非法消息实现 100% 阻断，88% 召回。与 GPT‑4o 零样本对比，SDOF 的意图‑安全联合准确率提升至 80.9%。

**⚠️ 局限性**

局限性：①流程阶段与意图绑定仍需手工定义，缺乏自动约束诱导；②在更深层次、层级化工作流（>6 阶段）中的表现尚未验证；③RLHF 结果主要基于单一随机种子，缺乏多种子稳健性评估；④GoalManager 的生命周期管理（保留策略、版本化、租户治理）尚不完整；⑤模型在需要长推理的情境下（如 Qwen3‑8B 的 think 模式）会严重退化，需进一步改进推理约束。

---

## 136. ReactiveGWM: Steering NPC in Reactive Game World Models

**arXiv ID:** 2605.15256 | [PDF](https://arxiv.org/pdf/2605.15256v1)

**作者:** Zeqing Wang `[一作]` (Tencent), Yeying Jin `[通讯]` (Tencent)

**通讯引用:** 677 | [OpenAlex ID](https://openalex.org/A5040632841)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种反应式游戏世界模型，能够在保持玩家细粒度控制的同时，让NPC执行预定义的高阶策略。

**💡 创新点**

创新点在于将玩家动作与NPC策略完全解耦：玩家动作通过轻量级加性偏置注入 diffusion 核心，NPC策略则通过跨注意力模块实现；这些模块学习了游戏无关的交互逻辑，可实现零-shot策略迁移。

**🔧 技术方法**

主要技术包括基于视频扩散模型（DiT）的轻量化动作注入、跨注意力模块实现 NPC 高阶策略编码、以及结构化策略提示的语言引导。

**📊 数据集**

使用了两套自构造的数据集：街机格斗游戏 Street Fighter II 与 Street Fighter Alpha 3 的约 10k 条带有玩家动作标签和 NPC 策略提示的视频片段。

**📈 对比分析**

与 Matrix‑Game‑3.0 与 LingBot‑World‑Base 等基线模型比较，验证了在动作控制、NPC 策略跟随和视觉质量三维度的优势；NPC 策略正确率从约 43% 提升至 75–80%，动作追踪几乎保持 100%，视觉质量（SSIM/LPIPS）保持与基线相当。

**⚠️ 局限性**

局限性包括：仅在两款格斗游戏上验证，跨游戏迁移仍需在不同游戏类型中进一步测试；跨注意力通道宽度低，可能对复杂交互的细节捕捉有限；生成质量受限于扩散模型的帧数与采样步数。

---

## 137. COPRA: Conditional Parameter Adaptation with Reinforcement Learning for Video Anomaly Detection

**arXiv ID:** 2605.15325 | [PDF](https://arxiv.org/pdf/2605.15325v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 138. Effective Harness Engineering for Algorithm Discovery with Coding Agents

**arXiv ID:** 2605.15221 | [PDF](https://arxiv.org/pdf/2605.15221v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 139. 3DEditSafe: Defending 3D Editing Pipelines from Unsafe Generation

**arXiv ID:** 2605.15398 | [PDF](https://arxiv.org/pdf/2605.15398v1)

**作者:** Nicole Meng `[一作]` (Tufts University), Yingjie Lao `[通讯]` (Tufts University)

**通讯引用:** 1392 | [OpenAlex ID](https://openalex.org/A5071172709)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了文本驱动3D Gaussian编辑中不安全内容传播问题，并提出了安全约束的3DEditSafe框架。

**💡 创新点**

首次在3D优化阶段引入渲染视图安全正则化与安全语义投影，结合生成阶段安全引导，显著降低不安全语义在3D场景中的持续出现。

**🔧 技术方法**

结合EditSplat的文本驱动3DGS编辑、CLIP语义相似度评估、Safe Latent Diffusion的生成阶段安全引导、语义风险门控、渲染视图3D安全正则化、残差抑制与掩码感知保留等技术。

**📊 数据集**

构建了30个提示-场景对的对象兼容不安全提示基准，来源于Unsafe Diffusion提示集，并在10个EditSplat场景上进行测试。

**📈 对比分析**

与标准EditSplat及仅在生成阶段的2D安全指导进行对比，3DEditSafe在视图级ASR与场景级ASR上平均下降约60%/50%，安全性提升显著，编辑质量略有下降。

**⚠️ 局限性**

安全性与图像质量之间存在权衡，导致部分伪影或视觉质量下降；安全评估仍受限于CLIP相似度等指标，且在更广泛的安全类别与更大规模场景上的泛化性尚待验证。

---

## 140. DualKV: Shared-Prompt Flash Attention for Efficient RL Training with Large Rollouts and Long Contexts

**arXiv ID:** 2605.15422 | [PDF](https://arxiv.org/pdf/2605.15422v1)

**作者:** Jiading Gai `[一作]` (Amazon Web Services), George Karypis `[通讯]` (University Of Minnesota)

**通讯引用:** 63336 | [OpenAlex ID](https://openalex.org/A5082384108)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DualKV，一种在 RL 后训练中消除共享前缀复制的 FlashAttention 内核及数据管道改造，显著降低计算和内存开销；

**💡 创新点**

创新点在于将前缀 KV 共享到单个物理缓冲区，并通过双区块前向/反向 CUDA 内核与 fp32 原子累加，实现对共享前缀梯度的正确收集；

**🔧 技术方法**

采用 FlashAttention 变体、两阶段 fp32 原子累加、变量长度打包、组查询注意力（GQA）以及 vLLM/veRL 的数据重排；

**📊 数据集**

在 Qwen3‑8B、Qwen3‑30B‑A3B、Llama‑3.1‑8B 等模型上，使用 LongReason（长上下文数学推理）和 MoE 任务数据；

**📈 对比分析**

与标准 FA2、FA3、Prefix Grouper 以及 4‑way Ulysses 序列并行的 FA2 进行对比，DualKV 在 8×H100 上实现 1.6–5.5× 前向/反向速度提升、MFU 提升至 76% 以上，并在 16‑GPU MoE 训练中获得 3.8× 的策略更新速度和 3.4× 的整体步骤速度；

**⚠️ 局限性**

局限在于只能处理每个微批次内同一前缀的多响应，无法直接扩展到多前缀或树结构的共享子图；

---

## 141. Time-Varying Deep State Space Models for Sequences with Switching Dynamics

**arXiv ID:** 2605.15311 | [PDF](https://arxiv.org/pdf/2605.15311v1)

**作者:** Sanja Karilanova `[一作]` (Uppsala University), Ayça Özçelikkale `[通讯]` (Uppsala University)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5050844254)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于时间可变状态空间模型（SSM）的深度网络，利用可学习基函数字典对状态转移、输入和输出矩阵随时间连续变化进行建模，并在合成四模式系统与语音降噪任务中进行验证。

**💡 创新点**

创新点在于将基函数展开嵌入SSM矩阵，实现了无需显式切换模式即可捕获连续时间变化的动力学，既保持了可训练性，又避免了传统SLDS需要预估模式数等难题。

**🔧 技术方法**

主要技术包括基函数展开（如高斯基函数+常数）、深度SSM网络结构、BPTT与AdamW优化、参数规模与稳定性约束以及SI‑SNR等语音评估指标。

**📊 数据集**

实验使用了合成的四模式SLDS数据以及MSWC“surrounding”子集的真实语音数据，并在后者上加入四模式噪声进行降噪任务。

**📈 对比分析**

通过与时间不变SSM网络在相同参数量下进行对比，使用MSE、SNR和SI‑SNR评估，结果表明时间可变模型在四模式系统和语音降噪任务中MSE明显下降、SI‑SNR提升至约20 dB，显著优于时间不变模型。

**⚠️ 局限性**

限制在于对基函数字典的选择和参数规模的依赖，无法有效处理非周期性或完全非平稳动态；在模型规模极大或变化频率高时，参数增长和训练难度仍是挑战。

---

## 142. Overreliance in Writing Tasks: Exploring Similarity-Based Measures of AI Influence on Writing and Proposing a Reflective Writing Interface Intervention

**arXiv ID:** 2605.15322 | [PDF](https://arxiv.org/pdf/2605.15322v1)

**作者:** Vitor H. A. Welzel `[一作]` (Simon Fraser University), Nicholas Vincent `[通讯]` (Simon Fraser University)

**通讯引用:** 954 | [OpenAlex ID](https://openalex.org/A5070837664)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了生成式 AI 在开放式写作任务中的影响，通过对比 AI 辅助与无 AI 辅助情境，量化 AI 建议对写作的采用程度，并设计了实时相似度反馈的写作界面，帮助用户更好地反思与 AI 的交互。

**💡 创新点**

创新点在于提出并应用多维度（词汇、结构、语义、情感）相似度衡量，揭示 AI 对写作的细微影响，并通过实时反馈界面提升用户对 AI 影响的意识，转向反思而非管制。

**🔧 技术方法**

使用了文本相似度指标（Jaccard、POS TF‑ISF 余弦、SBERT 余弦、情感匹配），配对 t 检验及效应量分析，并实现了基于这些指标的实时反馈编辑器。

**📊 数据集**

实验数据基于 O. Henry 的短篇《After Twenty Years》作为背景文本，参与者来源于 Prolific 任务平台与大学生，使用其撰写的分析与创作任务文本。

**📈 对比分析**

通过配对 t 检验和效应量比较，发现 AI 辅助显著提升词汇、结构和情感相似度（p<0.05，d>0.3），语义相似度呈上升趋势但未达显著；界面反馈在小规模 think‑aloud 研究中显著提升用户对 AI 采用的自我意识。

**⚠️ 局限性**

局限性包括：任务类型与 AI 辅助形式共变，难以分离两者影响；样本规模与实验场景缺乏生态效度；仅评估文本相似度，未衡量写作质量或学习成效。

---

## 143. Discretizing Group-Convolutional Neural Networks for 3D Geometry in Feature Space

**arXiv ID:** 2605.15368 | [PDF](https://arxiv.org/pdf/2605.15368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 144. PBT-Bench: Benchmarking AI Agents on Property-Based Testing

**arXiv ID:** 2605.15229 | [PDF](https://arxiv.org/pdf/2605.15229v1)

**作者:** Lucas Jing `[一作]` (Tsinghua University), Simon S. Du `[通讯]` (University of Washington)

**通讯引用:** 4650 | [OpenAlex ID](https://openalex.org/A5033061754)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套专门针对Python库的属性驱动测试（PBT）基准，用于评估大型语言模型（LLM）从文档中推断语义不变式并生成合适的Hypothesis测试策略以发现隐藏的语义错误。

**💡 创新点**

①首次构建聚焦PBT的手工挑选题库，保证每个Bug满足语义可测试、可表达、隐蔽且确定触发；②引入“Fail→Pass”评估机制，精确测量每个Bug是否被单独测试覆盖；③系统性对比不同提示模式（开放式vs.结构化Hypothesis提示）对模型性能的影响。

**🔧 技术方法**

使用Hypothesis框架编写属性测试、OpenHands工具集进行代码生成与执行、OpenRouter接口调用多种LLM（Claude、Qwen、Gemini等）。

**📊 数据集**

共注入365个人工设计的语义Bug，覆盖三类难度（L1–L3），分布在7类Python库（序列化、数据结构、时间日期、类型系统、数值、状态机、解析）。

**📈 对比分析**

对八款LLM在两种提示模式下进行3次独立跑，评价指标包括Bug召回率、问题覆盖率、完全召回率、测试精确度。结果显示：结构化PBT提示对中等能力模型提升20–25个百分点；最强模型在PBT模式下召回率83%，但仍有2/365 Bug未被任何模型覆盖。

**⚠️ 局限性**

限制包括：基准仅覆盖可通过PBT检测的Bug，可能与真实PR分布差异；只针对Python且英文文档；使用固定OpenHands框架，模型在其他环境下表现未知；200例搜索预算可能不足以触发某些复杂触发条件。

---

## 145. Beyond Bounded Variance: Variance-Reduced Normalized Methods for Nonconvex Optimization under Blum-Gladyshev Noise

**arXiv ID:** 2605.15314 | [PDF](https://arxiv.org/pdf/2605.15314v1)

**作者:** Antesh Upadhyay `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**通讯引用:** 467 | [OpenAlex ID](https://openalex.org/A5036900440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在Blum–Gladyshev噪声模型下研究非凸随机优化，提出并分析了归一化动量和归一化STORM方法。

**💡 创新点**

首次证明归一化动量在此噪声下可实现(ε^-6)的最优复杂度，并揭示在期望对称一般光滑性下归一化STORM的α-依赖复杂度。

**🔧 技术方法**

利用归一化更新、动量、递归方差减小STORM、Young不等式与轨迹收敛分析等技术。

**📊 数据集**

在相位检索和三次多项式两种人工合成数据集上进行实验。

**📈 对比分析**

与动态批量的SGD/STORM对比，归一化方法在不需增大批量的情况下实现更低梯度范数，显示出更优的收敛性能。

**⚠️ 局限性**

对期望对称一般光滑性下的α-依赖速率缺乏下界证明，且对大规模真实数据的评估尚未展开。

---

## 146. Method-level Change-proneness: A Better Metric for Black-box Test Suite Minimization

**arXiv ID:** 2605.15232 | [PDF](https://arxiv.org/pdf/2605.15232v1)

**作者:** Md Siam `[一作]`, Kazi Sakib `[通讯]` (University of Dhaka)

**通讯引用:** 603 | [OpenAlex ID](https://openalex.org/A5051563680)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究方法级变化易感度（CP）用于黑盒测试用例最小化，并提出了 MCTM 方法。

**💡 创新点**

创新点在于将 CP 从类级细化到方法级，并结合静态调用图关联评估测试用例相关性，取代传统的类级或相似度方法。

**🔧 技术方法**

使用了 Git 历史挖掘获取方法级变更数据、方法级 CP 计算、静态调用图构建、以及统计聚合（平均、几何平均等）对测试用例打分的技术。

**📊 数据集**

实验采用 Defects4J v2.0.1 中的 15 个 Java 项目，共 635 个 buggy 版本。

**📈 对比分析**

通过与 CTM、ATM、LTM 等基线比较，MCTM 在 50% 保留率下平均准确率 0.93、故障检测率 0.94，运行时间约 0.98 分钟，显著优于基线方法。

**⚠️ 局限性**

局限性包括：静态调用图无法完全捕获继承/多态调用，实验仅在单一故障版本上评估，且仅针对 Java，尚需验证在多故障或其他语言环境中的适用性。

---

## 147. NOVA: Fundamental Limits of Knowledge Discovery Through AI

**arXiv ID:** 2605.15219 | [PDF](https://arxiv.org/pdf/2605.15219v1)

**作者:** Salman Avestimehr `[一作]` (University of Southern California), Muriel Médard `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 29499 | [OpenAlex ID](https://openalex.org/A5075370174)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文提出了NOVA框架，系统化描述并分析了AI驱动的“生成-验证-累积-再训练”循环，用以研究AI是否能自我迭代发现真正的新知识；

**💡 创新点**

创新点在于给出了收敛与失效的充分条件、揭示了“污染陷阱”与“探索壁垒”，推导出基于Zipf尾部的累积生成成本 Θ(c_gen D^α) 的规模定律，并量化了人类干预在指导、验证与生成方面的放大效应；

**🔧 技术方法**

主要技术包括自适应采样理论、Species估计（Good–Turing）、占用法（Zipf 近似）、可测量的概率质量（M_t^new、U_t）、以及对验证错误率的理论分析；

**📊 数据集**

论文未使用具体数据集，而是以形式化模型（可验证数学证明、分子设计、科学假设生成等）作为抽象示例；

**📈 对比分析**

论文未给出数值实验，性能评价通过理论证明与定量推导完成，说明在满足条件下可实现几乎完全覆盖，成本呈指数下降；

**⚠️ 局限性**

局限性包括：收敛条件为充分而非必要、成本法律依赖Zipf尾部假设、验证模型过于简化、未考虑科学进展的多样性与概念重构等实际因素。

---

## 148. GESD: Beyond Outcome-Oriented Fairness

**arXiv ID:** 2605.15295 | [PDF](https://arxiv.org/pdf/2605.15295v1)

**作者:** Gideon Popoola `[一作]` (Montana State University), John Sheppard `[通讯]` (Montana State University)

**通讯引用:** 2743 | [OpenAlex ID](https://openalex.org/A5072522101)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于解释稳定性差异（GESD）的程序化公平度量，并将其与 AUC、Demographic Parity 等指标结合，构建三目标优化框架 FEU（Fairness–Explainability–Utility），通过进化算法求解。

**💡 创新点**

创新点在于：①引入 GESD 量化不同子群体在模型解释稳定性、鲁棒性、敏感性上的差异；②采用解释器聚合（SHAP+LIME）提升 GESD 的鲁棒性；③将 GESD 与传统公平、效能指标联合优化，形成完整的多目标框架。

**🔧 技术方法**

技术手段包括：SHAP 与 LIME 解释器、解释聚合、随机扰动（高斯噪声+特征屏蔽）、稳定性得分计算、NSGA‑II 进化优化、Pareto 前沿与 Chebyshev/Hypervolume 选解。

**📊 数据集**

实验使用四个公开数据集：German Credit、Recidivism、Math、Portuguese。

**📈 对比分析**

与基线方法（Adversarial、Reduction、Reweighing）比较，FEU 在大多数数据集上在 AUC 与 Demographic Parity/Equal Opportunity 上表现最好；在 GESD（程序化公平）上略逊于线性方法，但显著揭示了神经网络的解释不稳定性。

**⚠️ 局限性**

局限性：只考虑单一受保护属性，聚合解释器仍受限于 SHAP/LIME；小样本数据集导致神经网络解释差异放大；FEU 未集成其他偏差缓解技术，对极端偏倚的数据集适应性不足。

---

## 149. Unified Simulation of Lagrangian Particle Dynamics via Transformer

**arXiv ID:** 2605.15305 | [PDF](https://arxiv.org/pdf/2605.15305v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 150. Neural Activation Patterns Across Language Model Architectures: A Comprehensive Analysis of Cognitive Task Performance

**arXiv ID:** 2605.15436 | [PDF](https://arxiv.org/pdf/2605.15436v1)

**作者:** Mahdi Naser-Moghadasi `[一作]`, Faezeh Ghaderi `[通讯]` (University of Texas at Arlington)

**通讯引用:** 502 | [OpenAlex ID](https://openalex.org/A5074537862)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统分析了六种大型语言模型在十二类认知任务中的神经激活模式，量化了最终激活、注意力熵和稀疏度等指标，揭示了编码器与解码器架构以及不同任务的计算特征差异。

**💡 创新点**

创新点在于首次构建了跨模型、跨任务的完整激活数据集，并通过三维激活指标提出了任务特定的“激活签名”，为模型选择与优化提供了可操作的计算度量。

**🔧 技术方法**

使用了Transformer解释方法、注意力熵分析、稀疏度计算以及自研的LLM Brain Activity Analyzer框架，对模型内部隐藏层进行定量评估。

**📊 数据集**

数据集包含6个模型（BERT-Base、GPT2-117M、Qwen-1.5-0.5B、Phi-1、BLOOM-560M、StableLM-3B）与12类任务（事实问答、创造写作、数学推理、情感内容、技术代码、哲学问题、对话聊天、逻辑谜题、科学解释、语言任务、指令跟随、常识推理），共144个模型-任务组合，每组采用2个代表性提示。

**📈 对比分析**

通过比较三种激活指标，作者发现数学推理始终产生最高的注意力熵；解码器模型在稀疏度上显著高于编码器；参数规模与激活强度非线性相关。该比较表明不同模型在特定任务上的计算资源需求差异明显。

**⚠️ 局限性**

局限性包括样本量仅为每个模型-任务组合2个提示，缺乏对模型性能（准确率、生成质量等）的直接关联分析，且仅关注激活层级指标，未考虑更深层的训练动态或优化效果。

---

## 151. Fluency and Faithfulness in Human and Machine Literary Translation

**arXiv ID:** 2605.15282 | [PDF](https://arxiv.org/pdf/2605.15282v1)

**作者:** Sarah Griebel `[一作]` (University of Illinois Urbana-Champaign), Ted Underwood `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1518 | [OpenAlex ID](https://openalex.org/A5006779213)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了文学翻译中流畅度与忠实度的关系，使用大规模段落级别的译本数据进行实证分析。

**💡 创新点**

首次提出同时使用POS基翻译异化分类器与COMET‑KIWI无参考评估来量化流畅度与忠实度，并揭示段落长度对两者相关性的显著影响。

**🔧 技术方法**

采用POS抽象的翻译异化分类器、COMET‑KIWI质量估计、BGE‑M3嵌入相似度过滤以及长度分层的偏相关分析技术。

**📊 数据集**

使用Par3文学翻译语料（106本小说、16种源语言，共计130,486段落）以及115本原版英语小说用于训练翻译异化分类器。

**📈 对比分析**

通过Spearman相关、长度控制后的偏相关以及段落长度分层分析比较流畅度与忠实度，结果显示整体呈负相关，长度控制后负相关更为显著；人类译本负相关更强，而LLM（Google Translate、TranslateGemma）负相关弱甚至呈正相关。

**⚠️ 局限性**

局限包括缺乏翻译者及出版时间等元数据，流畅度仅基于POS结构忽略词汇与风格影响，且样本来自历史文本可能导致模型对现代语言的偏倚。

---

## 152. Hybrid LLM-based Intelligent Framework for Robot Task Scheduling

**arXiv ID:** 2605.15486 | [PDF](https://arxiv.org/pdf/2605.15486v1)

**作者:** Swayamjit Saha `[一作]` (Mississippi State University), Xiao-Yang Liu `[通讯]` (Columbia University)

**通讯引用:** 71816 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一个混合LLM框架，利用 GPT‑4 生成施工机器人任务调度计划，Gemma‑3 / LLaMA‑4 / Mistral‑7B 作为监督器对计划进行约束校验与最小编辑，最终输出可直接执行的 API 级别指令。

**💡 创新点**

创新点包括：① 双LLM 生成‑监督循环，解决单一 LLM 生成的不可行计划；② 通过结构化 Prompt 与可解释的错误修正，实现实时适应现场突发变化；③ 在施工任务调度领域首次引入标准文本相似度指标（BLEU/ROUGE/METEOR）与计划可行率、编辑距离等双重评价。

**🔧 技术方法**

技术手段：大型语言模型（GPT‑4、Gemma‑3、LLaMA‑4、Mistral‑7B）；结构化 Prompt 与 few‑shot 代码示例；API 方案设计；图模型化任务、资源与前后置约束；约束校验器与最小编辑投影算法；仿真实验平台。

**📊 数据集**

数据集：未使用公开真实工地数据，而是构造的模拟施工场景（9 砖墙搭建、扫描覆盖路径等）作为实验基准，提供地图、机器人清单、资源库存等背景信息。

**📈 对比分析**

比较方法：与仅 GPT‑4 生成、纯 FCFS 规则调度进行对比；评价指标包括可行率（FR）、BLEU/ROUGE/METEOR、编辑距离、充电/覆盖违规次数、总完成时间。实验显示监督器可将 FR 提升至 100%，且仅需 1–2 次局部编辑；Gemma‑3 在文本相似度上表现最佳。

**⚠️ 局限性**

局限性：① 仅在仿真环境验证，缺乏真实工地部署实验；② LLM 推理成本高且对计算资源有要求；③ 多机器人协同场景的通用性和规模仍待扩展；④ 依赖固定 API 与预定义约束，鲁棒性与可迁移性需进一步提升。

---

## 153. Residual Reinforcement Learning for Robot Teleoperation under Stochastic Delays

**arXiv ID:** 2605.15480 | [PDF](https://arxiv.org/pdf/2605.15480v1)

**作者:** Kaize Deng `[一作]` (Technical University of Munich), Zewen Yang `[通讯]` (Technical University of Munich)

**通讯引用:** 605 | [OpenAlex ID](https://openalex.org/A5073158465)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 DR-RL 框架，用 LSTM 状态估计器和残差 RL 策略实现远程操控下的高‑方差随机延迟稳健跟踪。

**💡 创新点**

创新点在于自回归 LSTM 估计器生成连续预测，避免步进不连续性；以及残差 RL 只学习补偿误差，提升稳定性和精度。

**🔧 技术方法**

采用 LSTM 神经网络、Soft Actor‑Critic (SAC) RL、基于动力学的计算力矩控制和自回归预测。

**📊 数据集**

实验数据集基于 MuJoCo 模拟的 Franka Panda 机器人，并在真实 Franka Panda 机器人上验证。

**📈 对比分析**

与 Vanilla PD 与 PMDC 基线对比，三种延迟配置下 DR‑RL 在跟踪误差上始终优于对手，尤其在高延迟高方差情况下表现最为突出。

**⚠️ 局限性**

局限性包括：sim‑to‑real 差距约 22%，仍需针对更大延迟或不同机器人进行适配；模型训练耗时，且对极端网络波动的鲁棒性未进一步验证。

---

## 154. EgoExo-WM: Unlocking Exo Video for Ego World Models

**arXiv ID:** 2605.15477 | [PDF](https://arxiv.org/pdf/2605.15477v1)

**作者:** Danny Tran `[一作]` (University of Texas at Austin), Kristen Grauman `[通讯]` (University of Texas at Austin)

**通讯引用:** 28991 | [OpenAlex ID](https://openalex.org/A5012765543)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种通过提取外观视频中的3D人体动作，并利用该动作指导将外观视频转化为第一人称视频，从而将大量外观数据用于训练 egocentric 世界模型，并基于该模型进行目标导向规划。

**💡 创新点**

创新点：① 将3D人体姿态作为动作空间并直接用于世界模型训练；② 在 exo‑to‑ego 生成中加入全身运动学先验，提升动作一致性；③ 将转换后的外观数据与原始 egocentric 数据统一格式，显著提升预测与规划性能。

**🔧 技术方法**

采用的技术包括：SMPL/ SAM‑Body 3D 姿态估计、EgoX‑Body + 运动扩散模型进行 exo‑to‑ego 合成、DINOv3‑L 视觉潜在空间、手腕一致性损失、MPC‑style 规划与 UniEgoMotion 采样、以及类似 PEVA 的扩散式世界模型。

**📊 数据集**

训练集：Nymeria（200h egocentric）+ 10h 转换后的外观视频（HowTo100M 5h、CrossTask 1h、100 Days of Hands 4h）。评估集：Home Action Genome、LEMMA、Ego‑Exo4D（Bike 与 Cooking 子集）。

**📈 对比分析**

与 PEVA‑L/XL/XXL、EgoControl、Ego‑WM、Naive EgoExo‑WM 等基线进行对比，EgoExo‑WM 在所有数据集上均取得最低的 L2 embedding error 与最高的 wrist PCK@20，且在 MPC 规划任务中比基线误差更低、性能更优。

**⚠️ 局限性**

局限性：仅支持约 49 帧（≈2 秒）短时预测；exo‑to‑ego 转换在复杂人物‑物体交互、遮挡与小物操作时效果有限；长期规划面临预测误差累积问题。

---

## 155. Jobs' AI Exposure Should Be Measured from Evidence, Not Model Priors

**arXiv ID:** 2605.15474 | [PDF](https://arxiv.org/pdf/2605.15474v1)

**作者:** Luca Mouchel `[一作]` (École Polytechnique Fédérale de Lausanne), Yossi Sheffi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 5394 | [OpenAlex ID](https://openalex.org/A5108804623)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了基于检索增强、外部证据驱动的职业AI曝光测量框架，强调测量应具备可复制性、外部验证和可检查性；

**💡 创新点**

主张将AI曝光测量从无依据的零样本推断转为依赖外部新闻与学术文献作为证据，并通过开放权重模型进行推理；

**🔧 技术方法**

使用检索增强生成（RAG）技术、跨编码器重排序、开放权重推理模型（如Qwen3-30B、Ministral14B、Gemma4-31B）以及结构化工作场景约束；

**📊 数据集**

利用O*NET 30.2职业-任务对、约3.4万条2025-2026年新闻报道和约1.9万条学术摘要作为外部证据；

**📈 对比分析**

通过自动评估（Prometheus 7B、GPT-5.4 Mini）和人工评估对比零上下文与检索增强条件，检索增强在72%以上不一致案例中获胜，并与真实使用数据（Claude交互记录）相关性更高，提升了约3.5%–9.2%；

**⚠️ 局限性**

缺乏真实基准标签，证据来源覆盖不均，模型仍需依赖推理，E0–E3分级粗糙，无法捕捉连续维度，且观测使用度量与理论曝光不同，二者互补而非替代。

---

## 156. Retrieval-Augmented Large Language Models for Schema-Constrained Clinical Information Extraction

**arXiv ID:** 2605.15467 | [PDF](https://arxiv.org/pdf/2605.15467v1)

**作者:** A H M Rezaul Karim `[一作]` (George Mason University), Ozlem Uzuner `[通讯]` (George Mason University)

**通讯引用:** 8834 | [OpenAlex ID](https://openalex.org/A5070926324)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在护士-患者对话文本中实现了基于预定义模式的观察信息结构化提取。

**💡 创新点**

首次系统探讨检索增强生成与不同复杂度模式约束在大型语言模型中的相互作用，并证明模型意识的模式约束策略对提取效果至关重要。

**🔧 技术方法**

采用检索增强生成（RAG）结合全模式或裁剪候选模式的提示、结构化后处理与二次审计等技术。

**📊 数据集**

使用公开的 SYNUR 合成对话数据集（包含 122‑101‑199 例，193 种观察概念），并在训练集上做检索示例。

**📈 对比分析**

在 Llama‑4 与 GPT‑5.2 上进行对比实验，最佳配置为 GPT‑5.2 + RAG + 全模式 + 2 次审计，F1 最高达 80.27%，明显优于仅提示或仅检索的基线。

**⚠️ 局限性**

主要限制包括标识符/值格式不一致导致的错误、稀疏概念难以覆盖、以及对自由文本值的精确对齐仍需改进。

---

## 157. FLASH: Efficient Visuomotor Policy via Sparse Sampling

**arXiv ID:** 2605.15492 | [PDF](https://arxiv.org/pdf/2605.15492v1)

**作者:** Jiaqi Bai `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**通讯引用:** 7207 | [OpenAlex ID](https://openalex.org/A5005666034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FLASH策略，通过Legendre多项式系数表示轨迹，实现稀疏采样与历史锚定流匹配，单步推理即可覆盖较长动作窗口；

**💡 创新点**

创新点在于：①稀疏时间采样将长时间跨度压缩至少量系数；②历史先验引导的流匹配显著缩短推理距离，单步推理可行；③多项式可解析求导，将速度作为前馈信号给低层控制器；

**🔧 技术方法**

使用的技术包括：Legendre多项式表示、稀疏采样与OLS拟合、跨越连续性约束、历史锚定的流匹配（流匹配损失+一致性损失）、DiT式流变换器和视觉编码器；

**📊 数据集**

在Franka机器人上进行的7个抓取/堆叠/插入等任务；其中5个在Roboverse模拟环境中，2个为真实世界实验；

**📈 对比分析**

与9个SOTA基线（DDPM、FM、ACT、VITA、A2A等）对比，FLASH在5个任务上均达到≥92%成功率，单步推理下每集耗时31.4 ms（比Score-UNet快175倍），训练收敛速度约4倍快，控制器跟踪误差下降5–7倍；

**⚠️ 局限性**

局限性包括：多项式阶数K固定，无法自适应高频或突变轨迹；执行速度k_eval在一次推理后不可变，无法在跑步过程中实时调整速度。

---

## 158. When Does Sparse MoE Help in Vision? The Role of Backbone Compute Leverage in Sparse Routing

**arXiv ID:** 2605.15484 | [PDF](https://arxiv.org/pdf/2605.15484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 159. Domain-Independent Game Abstraction using Word Embedding Techniques

**arXiv ID:** 2605.15543 | [PDF](https://arxiv.org/pdf/2605.15543v1)

**作者:** Juho Kim `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20315 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出使用词向量技术将游戏动作映射为向量，并通过聚类实现游戏抽象

**💡 创新点**

实现了域无关的抽象方法，利用自然语言处理中的词嵌入捕捉动作的语义/策略相似性

**🔧 技术方法**

采用GloVe、OpenAI和Google Gemini等词向量/基础嵌入模型进行动作嵌入，然后用k-means聚类

**📊 数据集**

使用棋类正式比赛数据、256-Kuhn扑克生成的游戏数据以及人类或基于均衡策略生成的扑克/勒库德扑克数据

**📈 对比分析**

与随机聚类和手工抽象（手牌分桶）对比，结果显示相对于随机基线表现更好，但在专业扑克抽象算法面前仍逊色；在抽象尺寸增大时，exploitability下降，说明抽象质量提升

**⚠️ 局限性**

对域知识的依赖有限，但需能将动作映射为可读文本；对大规模游戏数据的需求高；使用基础嵌入模型需牺牲一定的域独立性；在专业化算法上性能不占优势

---

## 160. DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding

**arXiv ID:** 2605.15542 | [PDF](https://arxiv.org/pdf/2605.15542v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 161. Learning Dynamic Structural Specialization for Underwater Salient Object Detection

**arXiv ID:** 2605.15535 | [PDF](https://arxiv.org/pdf/2605.15535v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 162. Task-Semantic Graph-Driven Distributed Agent Networking for Underwater Target Tracking

**arXiv ID:** 2605.15528 | [PDF](https://arxiv.org/pdf/2605.15528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 163. DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection

**arXiv ID:** 2605.15518 | [PDF](https://arxiv.org/pdf/2605.15518v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 164. Rethinking Neural Network Learning Rates: A Stackelberg Perspective

**arXiv ID:** 2605.15530 | [PDF](https://arxiv.org/pdf/2605.15530v1)

**作者:** Sihan Zeng `[一作]` (JPMorgan AI Research), Sumitra Ganesh `[通讯]` (JPMorgan AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并分析使用不同层级学习率的两时间尺度SGD算法，解释其等价于对网络训练目标的Stackelberg优化。

**💡 创新点**

证明非均匀学习率可导致Stackelberg目标强凸或具有更锐利局部曲率，从而实现全局最优或加速收敛，并给出理论收敛率。

**🔧 技术方法**

使用Stackelberg优化框架、两时间尺度随机近似、弱凸性、Moreau包络、子梯度理论、ReLU等非光滑激活函数的分析方法。

**📊 数据集**

在回归（Friedman、Boston Housing）、分类（MNIST、MNIST Fashion、CIFAR10）以及强化学习（TDC、Atari 四个游戏）等数据集上验证。

**📈 对比分析**

与全网络使用相同学习率的均匀设置（低速和高速）进行对比，实验显示非均匀学习率收敛更快、最终性能不低于或优于均匀设置。

**⚠️ 局限性**

收敛证明依赖于强凸/弱凸假设，且要求可获得无偏梯度；对深度Q学习等不满足假设的场景不适用；多时间尺度和无偏梯度的实现对实际工程有一定挑战。

---

## 165. Neural Point-Forms

**arXiv ID:** 2605.15524 | [PDF](https://arxiv.org/pdf/2605.15524v1)

**作者:** Bruno Trentini `[一作]` (NVIDIA), Kelly Maggs `[通讯]` (Max Planck Institute of Molecular Cell Biology and Genetics)

**通讯引用:** 92 | [OpenAlex ID](https://openalex.org/A5001812471)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出神经点形式（Neural Point‑Forms，NPF）来表示点云的外在微分形式，并通过比较矩阵学习子流形特征；

**💡 创新点**

创新点在于将外在微分形式与扩散几何的拉普拉斯算子相结合，得到无组合结构的可学习梯度特征，并在理论上证明了在标准采样假设下的长期一致性；

**🔧 技术方法**

主要技术包括拉普拉斯变分估计、扩散几何、神经网络参数化的k‑形式、Gram场构造以及比较矩阵层；

**📊 数据集**

实验使用了三类数据集：人工生成的线性与圆形轨迹、模拟RNA动力学模型以及真实的患者来源器官（PDO）单细胞测序数据；

**📈 对比分析**

在与GCN、GIN、GraphSAGE、PointNet++、GraphTransformer、TDL等基线模型的AUROC/精度比较中，NPF在小参数预算下与SOTA相当或更优，尤其在PDO tumor‑selective 任务上取得最高点估计；

**⚠️ 局限性**

局限性包括特征数随k与D的阶乘增长，导致在高维特征空间中计算成本高昂；扩散几何方法受维数灾难影响，难以扩展到更大、更高维的数据集；

---

## 166. OgBench: A Framework for Evaluating Graph Neural Networks on Omics Data

**arXiv ID:** 2605.15511 | [PDF](https://arxiv.org/pdf/2605.15511v1)

**作者:** Louisa Cornelis `[一作]` (University Of California Santa Barbara), Nina Miolane `[通讯]` (University Of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布了Omics‑Graph Bench——针对 n ≪ p 维度的生物学图数据，提供标准化的从原始 omics 数据到图结构的端到端处理流水线，并对多种 GNN 与传统方法进行系统基准评测。

**💡 创新点**

①首个专注 n ≪ p 维度的生物学图学习基准；②统一预处理与图构建流程，消除流程差异导致的性能偏差；③模块化设计支持节点选择、边构建与模型的灵活替换；④对比结果揭示大多数 GNN 并未显著优于 MLP 或线性模型；⑤开源生态与公开排行榜鼓励社区扩展。

**🔧 技术方法**

采用多种 GNN（GCN、GraphSAGE、GATv2、GIN、SAGN、MLA‑GNN、ChebNet）、MLP、传统机器学习（SVM、Elastic Net）等；节点筛选方法包括方差、Pearson、距离相关和随机；边构建方式为基于共表达的软阈值网络和 STRING PPI；使用 Accuracy、Macro/Weighted F1、Macro AUROC 等指标，并进行 top‑K 集成。

**📊 数据集**

四个公开数据集：HERITAGE（蛋白质组，654 样本×4977 蛋白），Parkinson’s（转录组，535 样本×21755 基因），AddNeuroMed（转录组，711 样本×17198 基因），BRCA（表观组，640 样本×19049 基因）。

**📈 对比分析**

通过在不同节点选择、边构建、样本/节点比例（n/p）设置下训练并评估 80k+ 组合模型。结果显示：GNN 在某些数据集（HERITAGE、AddNeuroMed）可略优于基线；在 Parkinson’s、BRCA 中线性基线或 MLP 同样或更好；无一种架构在所有任务中始终领先；集成可降低方差但并未改变总体趋势。

**⚠️ 局限性**

仅包含四个数据集与三种 omics 模式，缺乏代谢组、单细胞、空间转录组等；采用共享固定图结构，未考虑样本间异质性；可能导致对图结构价值的低估，未来需加入样本自适应图构建与更广泛的数据来源。

---

## 167. parallelcbf: A composable safety-filter and auditability framework for tensor-parallel reinforcement learning

**arXiv ID:** 2605.15509 | [PDF](https://arxiv.org/pdf/2605.15509v1)

**作者:** Yijun Lu `[一作]` (Waseda University), Yuyin Ma `[通讯]` (Xinjiang University)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5049276169)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ParallelCBF 框架，实现了 Tensor‑parallel UAV 仿真、双阈 CBF 安全过滤、行为克隆到强化学习的衔接，并提供了预注册、看门狗、失效取证等一体化可审计接口。

**💡 创新点**

创新点在于将安全过滤和可审计机制从用户脚本提升为版本化、类型检查、单元测试覆盖的 API，形成四层可组合的体系结构；并首次公开了 CPU PyTorch 参考实现的双阈 CBF 与完整的属性测试。

**🔧 技术方法**

使用 Python、NumPy、CPU PyTorch；实现双阈 CBF（硬阈为平方距离、软阈为线性预测）；基于 Hypothesis 的属性测试、向量化批量测试；通过 Watchdog、Forensics、Pre‑Registration 等模块实现审计功能。

**📊 数据集**

在 2D toy UAV 环境中采集了 31,415 条行为克隆数据（基于 50,000 次仿真尝试），并将数据集 SHA‑256 作为可审计凭证。

**📈 对比分析**

对比方法主要是内部一致性与可复现性：测试套件 39 条用时 1.67 s；示例管线在预注册阈值不满足时被 Watchdog 正确终止，防止错误检查点传播；在算法替换实验中仅改动 Layer 3，保持安全性不变，验证了可组合性。

**⚠️ 局限性**

限制：v0.1 仅支持 2D toy 环境；未集成 Isaac Lab、GPU CBF 加速、先进 RL 算法（如 KL‑anchored PPO）；缺乏 3D 真实 UAV 或飞行平台验证；未来版本计划加入这些功能。

---

## 168. Learning with Conflicts of Interest

**arXiv ID:** 2605.15504 | [PDF](https://arxiv.org/pdf/2605.15504v1)

**作者:** Nischal Aryal `[一作]` (Oregon State University), Marianne Winslett `[通讯]` (University of Illinois)

**通讯引用:** 8168 | [OpenAlex ID](https://openalex.org/A5011314280)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于博弈论的框架，研究用户在数据隐私与机器学习模型偏差冲突下的策略性数据报告，并给出了求解最优报告的算法

**💡 创新点**

创新点在于将数据报告视为博弈中的信号问题，设计了可扩展的算法和理论分析，涵盖单维、复维以及噪声学习情形，并证明了一般情况下的NP/ #P 难度

**🔧 技术方法**

主要采用贝叶斯均衡分析、动态规划、线性/非线性方程求解、因子化先验分解、以及数值求根与非凸优化技术

**📊 数据集**

使用四个真实数据集：Credit Approval、Census、School Admission、Prosper Loan，分别训练线性 SVM 和两层 MLP 模型

**📈 对比分析**

通过与非策略性（无效信息）对照实验，评估策略性报告在不同偏差与噪声水平下的效益，结果显示大多数情形下策略性方法能显著提升用户效用，计算时间在可接受范围内（噪声情形下略高）

**⚠️ 局限性**

主要局限在于对一般先验分布求解困难、噪声环境导致计算复杂度升高、部分数据集（如 Census）在高噪声下收益不明显、且需假设先验可分离或可用统计 oracle

---

## 169. Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning

**arXiv ID:** 2605.15523 | [PDF](https://arxiv.org/pdf/2605.15523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 170. Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs

**arXiv ID:** 2605.15491 | [PDF](https://arxiv.org/pdf/2605.15491v1)

**作者:** Vincent-Daniel Yun `[一作]` (University of Southern California), Sunwoo Lee `[通讯]` (Inha University)

**通讯引用:** 2987 | [OpenAlex ID](https://openalex.org/A5100698554)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究层剪枝后缺失的Transformer块导致的激活分布偏移，提出训练无关的Ghosted Layers恢复模块。

**💡 创新点**

推导闭式最优线性算子以对齐边界激活，突破已有方法受限于对称子空间的不足。

**🔧 技术方法**

使用SVD伪逆求解最小二乘，前向hook插入闭式算子，对齐目标实现无训练恢复。

**📊 数据集**

在C4、WikiText-2、Penn Treebank等公开文本上收集少量校准序列进行激活对齐。

**📈 对比分析**

与Prune&Comp、ReplaceMe、LinearPatch等训练自由恢复方案在九个commonsense QA和PPL基准上对比，Ghosted Layers在多种LLM（LLaMA-3、LLaMA-3.1、DeepSeek）上均取得更高准确率、更低困惑度，提升幅度随剪枝深度增大。

**⚠️ 局限性**

需要访问未剪枝模型以采集激活，并对SVD/求解算子做一次离线运算，且对极端稀疏/非连续剪枝的泛化尚待验证。

---

## 171. A Unified Non-Parametric and Interpretable Point Cloud Analysis via t-FCW Graph Representation

**arXiv ID:** 2605.15475 | [PDF](https://arxiv.org/pdf/2605.15475v1)

**作者:** Haijian Lai `[一作]` (Macao Polytechnic University), Sio-Kei Im `[通讯]` (Macao Polytechnic University)

**通讯引用:** 829 | [OpenAlex ID](https://openalex.org/A5022394682)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个基于加权全连接图（t-FCW）与其强化版（empowered t-FCW）的无参数、可解释的点云分析框架，可同时用于分类、部件分割和语义分割。

**💡 创新点**

核心创新在于将点云的特征维度视为向量空间中的维度，通过Gram矩阵构造的t-FCW实现对维度间相关性的紧凑、可解释编码；进一步引入empowered t-FCW保留局部邻域信息，兼顾分割任务；同时结合旋转不变表面描述子（RISP）提升旋转鲁棒性。

**🔧 技术方法**

技术包括点云采样（FPS + k-NN）、多种点云表面描述子（xyz、GeoPCSD、RISP）、t-FCW与empowered t-FCW图构造、基于内存银行的相似度预测、以及对比实验与消融分析。

**📊 数据集**

在四个主流数据集上验证：ModelNet40、ScanObjectNN、ShapeNet-Part、S3DIS。

**📈 对比分析**

与现有无参数方法（Point-NN、Seg-NN、ICP-Classifier）和参数方法（PointNet、PointNet++、PointMLP、Point-GN）对比，t-FCW在分类上达到84.8%（仅使用xyz）且速度最快，empowered t-FCW在分割任务上与Point-NN相当或略优，且在旋转、噪声、3D对抗攻击下表现更稳健。

**⚠️ 局限性**

局限性包括：对大型点云的局部邻域构造仍需额外计算；在极大噪声或全局噪声环境下性能下降；以及对点云顺序与批大小敏感，需在实际部署中注意特征隐私和批量信息的依赖。

---

## 172. CAPS: Cascaded Adaptive Pairwise Selection for Efficient Parallel Reasoning

**arXiv ID:** 2605.15513 | [PDF](https://arxiv.org/pdf/2605.15513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 173. X-SYNTH: Beyond Retrieval -- Enterprise Context Synthesis from Observed Human Attention

**arXiv ID:** 2605.15505 | [PDF](https://arxiv.org/pdf/2605.15505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 174. AnyAct: Towards Human Reenactment of Character Motion From Video

**arXiv ID:** 2605.15497 | [PDF](https://arxiv.org/pdf/2605.15497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 175. A QUBO Formulation Framework for Kinematic Structure-Based Robot Design Optimization: A Robotic Hand Case Study

**arXiv ID:** 2605.15510 | [PDF](https://arxiv.org/pdf/2605.15510v1)

**作者:** HyoJae Kang `[一作]` (Korea Institute of Machinery & Materials), Dongil Park `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种基于运动学结构的机器人设计优化框架，并将其转化为二进制量子无约束优化（QUBO）问题，随后利用模拟退火和D‑Wave量子退火验证其可行性，案例为机械手手指的组合设计。

**💡 创新点**

创新点在于：①将机器人运动学性能指标（操纵性、自由度、拇指-指尖交互空间）与结构依赖性惩罚统一映射到QUBO模型；②设计了分离经典计算与量子退火的工作流程，使得量子硬件可以直接求解机器人结构组合问题。

**🔧 技术方法**

使用技术包括：经典运动学分析（DH参数、前向运动学、工作空间体素化、操纵性评估）、二进制QUBO建模（目标、约束、交互项）、模拟退火（SimulatedAnnealingSampler）以及D‑Wave量子退火（Advantage系统4.1、EmbeddingComposite）。

**📊 数据集**

数据集：针对27个二进制设计变量的机械手模型，采用离散角度采样和体素化方法生成各设计候选的评估指标（操纵性、自由度、拇指-指尖重叠体积）。

**📈 对比分析**

比较方法：在同一QUBO矩阵下，使用模拟退火（不同读取次数NoR）作为经典基线，并与D‑Wave QA在相同或更高读取次数下的结果进行对比。结果显示，两种方法在目标值分布和最佳解上相近，QA在足够读取次数时能够得到与SA相同的最低目标值(-54.77)。

**⚠️ 局限性**

局限性：未考虑几何约束（指节厚度、链接碰撞）、结构干涉、驱动器限制和制造可行性；当设计变量规模大幅增加时，求解所需的读取次数和高级采样策略仍待进一步研究。

---

## 176. STS: Efficient Sparse Attention with Speculative Token Sparsity

**arXiv ID:** 2605.15508 | [PDF](https://arxiv.org/pdf/2605.15508v1)

**作者:** Ceyu Xu `[一作]` (Hong Kong University of Science and Technology), Yuan Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 113808 | [OpenAlex ID](https://openalex.org/A5068891302)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于草稿模型的无训练稀疏注意力机制（Speculative Token Sparsity, STS），可在预填充和解码阶段动态生成稀疏掩码，实现对大型语言模型注意力计算的显著剪枝。

**💡 创新点**

创新点在于：①利用草稿模型的注意力分布作为代理，预先预测目标模型的高重要性 token 并生成“已知提前” 的稀疏掩码；②通过离线注意力头映射将小模型头映射到大模型头；③将稀疏掩码与speculative decoding无缝整合，实现稀疏推理的同时保持完整 KV‑cache；④支持块级稀疏与预取优化，隐藏 KV‑cache 迁移延迟。

**🔧 技术方法**

技术手段包括：离线注意力头映射（基于 Top‑k 掩码交集的相关性匹配）；在草稿模型前向过程中提取注意力权重并 Top‑k 生成掩码；FlashInfer 中自定义块稀疏注意力核；异步 CUDA 流与事件同步实现草稿模型与目标模型的流水线；基于预取的 KV‑cache offload 方案。

**📊 数据集**

实验使用 WikiText‑2（用于离线映射），LongBench、SwBench、NarrativeQA、SweBench、MMLU、GSM8K、ARC 等标准长序列与推理基准；对 8B 与 70B Llama‑3.1/3.1‑8B‑Instruct 与 Llama‑3.2‑1B‑Instruct 进行验证。

**📈 对比分析**

与 Dense、Quest、TidalDecode、StreamingLLM 等现有稀疏/线性注意力方法对比，STS 在 10K‑100K token 上实现 1.36×~2.88× 的速度提升，NarrativeQA 上达到 2.67× 加速，90% 稀疏率下保持 0.5% 以内的准确率差距；同时在长文本推理、代码生成、检索任务上保持与密集注意力相近的性能。

**⚠️ 局限性**

局限性包括：①需要一对一的离线头映射，映射不适用于模型结构差异过大的情况；②依赖草稿模型的注意力质量，若草稿模型与目标模型相距较远可能导致掩码失效；③块级稀疏导致的索引开销在极细粒度下仍不可忽略；④在极低内存预算下仍需 KV‑cache offload，虽然通过预取能降低延迟，但在带宽受限环境中效果有限。

---

## 177. Njord: A Probabilistic Graph Neural Network for Ensemble Ocean Forecasting

**arXiv ID:** 2605.15470 | [PDF](https://arxiv.org/pdf/2605.15470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 178. uGen: An Agentic Framework for Generating Microarchitectural Attack PoCs

**arXiv ID:** 2605.15503 | [PDF](https://arxiv.org/pdf/2605.15503v1)

**作者:** Debopriya Roy Dipta `[一作]` (Iowa State University), Berk Gulmezoglu `[通讯]` (Iowa State University)

**通讯引用:** 700 | [OpenAlex ID](https://openalex.org/A5060182940)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了一个基于大型语言模型的自动化微架构攻击PoC生成框架

**💡 创新点**

通过检索增强的多代理设计填补LLM在攻击特定知识上的缺口，并能生成功能正确的PoC

**🔧 技术方法**

采用GPT、Claude、Qwen3等LLM，结合检索增强、多代理协同与攻击原语注入技术

**📊 数据集**

使用覆盖不同微架构、漏洞函数及缓存/投机执行攻击的实验数据集进行评估

**📈 对比分析**

在Spectre-v1上实现100%成功率，在Prime+Probe上实现80%成功率，PoC生成时间不到4分钟，成本仅$1.25

**⚠️ 局限性**

对某些复杂或罕见攻击仍可能出现误生成，模型选择与检索知识库质量决定性能，系统尚未完全普适

---

## 179. Estimated Dynamic Equilibrium Model: Supply and Demand as a Sample Path of a Stochastic Process

**arXiv ID:** 2605.15472 | [PDF](https://arxiv.org/pdf/2605.15472v1)

**作者:** Mikhail L. Arbuzov `[一作]` (Independent Researcher), Alexey Shvets `[通讯]` (Palo Alto Networks)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于代理的动态不平衡模型（EDEM），模拟供需为随机过程，揭示最大价位清算与价格反馈产生的统计偏差导致价格漂移与泡沫。

**💡 创新点**

证明在无偏估计下，max-bid 清算与价格反馈形成的顺序统计偏差能产生持续正向价格漂移，实现从 Walrasian 平衡到 Miller 静态溢价的连续体，并展示六种宏观价格动态。

**🔧 技术方法**

采用代理基础建模（Mesa 框架）与离散时间模拟，解析求解顺序统计期望，构造随机步进供需调节器，并对不同参数进行敏感性网格扫描。

**📊 数据集**

主要使用合成模拟数据，基于 32×32 盘点的虚拟房地产邻里，未使用真实市场数据。

**📈 对比分析**

通过八个控制实验和 30 格参数网格比较价格偏差、波动性与泡沫指数；结果显示不同参数可产生稳定、周期、过冲、泡沫等六类稳定态，模型在单机上可在数千步内完成指数增长或长期稳态计算，性能表现良好。

**⚠️ 局限性**

局限性包括单一资产类别、无杠杆、无宏观冲击、代理无学习、清算规则对波动性过于敏感；未验证对真实市场的拟合，也未探索极端 balancer 值。

---

## 180. GreenZ: A Sustainable UX Framework for Complex Digital Systems

**arXiv ID:** 2605.15468 | [PDF](https://arxiv.org/pdf/2605.15468v1)

**作者:** Trisha Solanki `[一作]` `[通讯]` (UXperiment Inc.), Trisha Solanki (UXperiment Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了GreenZ三层可持续UX框架，涵盖数字废弃物分类、AI充分性决策模型及一系列操作工具。

**💡 创新点**

创新点包括八类数字废弃物税onomies、首创的AI充分性决策模型、将可持续性、认知足足和AI责任融为一体的三层架构。

**🔧 技术方法**

采用文献综述、系统思维、认知负荷理论等HCI理论构建框架，并提供审计工作表、决策树、指标仪表盘等工具。

**📊 数据集**

未使用专门实验数据集，仅基于已有行业报告与研究（McGovern、Pendo、Luccioni等）进行理论整合。

**📈 对比分析**

目前未进行实证比较或性能评估，框架设计待后续专家评审与案例研究验证。

**⚠️ 局限性**

局限性在于缺乏经验验证、跨文化适用性待检验、碳阴影估算不确定、对工业落地的可操作性尚待验证。

---

## 181. LAPS: Improving Incremental LiDAR Mapping using Active Pooling and Sampling for Neural Distance Fields

**arXiv ID:** 2605.15496 | [PDF](https://arxiv.org/pdf/2605.15496v1)

**作者:** Dongjae Lee `[一作]` (Seoul National University), Ayoung Kim `[通讯]` (Seoul National University)

**通讯引用:** 5840 | [OpenAlex ID](https://openalex.org/A5100740100)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于神经距离场的增量激光雷达地图构建框架，采用可靠性驱动的主动池化和不确定性驱动的主动采样来管理回放数据，提升地图完整性与训练效率。

**💡 创新点**

创新点在于将回放缓冲区的样本保持策略从被动采样转为依据样本可靠性主动挑选，并利用模型不确定性动态分配在线优化预算，从而在有限内存与计算资源下显著提高地图重建的完整性。

**🔧 技术方法**

主要技术包括多分辨率稀疏体素网格的神经距离场表示、基于射线前后采样生成TSDF监督、可靠性评估（基于投影误差与角度偏差）、拉普拉斯近似的模型不确定性估计与基于不确定性的批次采样。

**📊 数据集**

在三组数据集上评估：合成的 MaiCity、实景的 Newer College 以及规模宏大、结构复杂的 Oxford Spires，分别测试其在不同规模与复杂度下的表现。

**📈 对比分析**

与 SHINE‑Mapping、N^3‑Mapping、PIN‑SLAM 等前沿神经映射方法以及 VDBFusion、Voxfield 等传统 TSDF 融合方法进行对比；实验显示，所提方法在 recall、F1‑score 及 Chamfer‑L1 等指标上均优于基线，尤其在 Oxford Spires 上 recall 提升 4.66pp、F1‑score 提升 3.79pp，保持了与 PIN‑SLAM 相当的几何精度。

**⚠️ 局限性**

局限性包括：需手动调节可靠性阈值 τ 与不确定性阈值 λ，且对动态场景或高速运动的雷达数据仍未做充分验证；此外在极大规模场景中仍可能出现内存占用与训练时延上升的问题。

---

## 182. Terrain Consistent Reference-Guided RL for Humanoid Navigation Autonomy

**arXiv ID:** 2605.15517 | [PDF](https://arxiv.org/pdf/2605.15517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 183. KaRMA: A Kinematic Metric for Fine Manipulation Ability in Robotic Hands

**arXiv ID:** 2605.15548 | [PDF](https://arxiv.org/pdf/2605.15548v1)

**作者:** Martin Peticco `[一作]` (Massachusetts Institute of Technology), Pulkit Agrawal `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 5368 | [OpenAlex ID](https://openalex.org/A5111774389)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一种基于机器人手腕运动学的KaRMA指标，用于评估两指精细握持时在滚动接触下可实现的对象平移和旋转可达空间。

**💡 创新点**

创新点在于引入滚动接触约束的可达性搜索，提出KaRMA-T、KaRMA-R、KaRMA-S三维度评分，能够分离平移与旋转能力并评估抓握稳健性，超越传统静态或雅可比基准。

**🔧 技术方法**

技术手段包括基于URDF的运动学模型、胶囊-球接触几何、滚动一致的关节步长QP求解、两阶段广度优先搜索、HEALPix方向离散化以及关节限、碰撞、摩擦拐角等联合约束。

**📊 数据集**

实验使用了16个常见机器人手模型（如Ability、Allegro、D‑Claw、Leap等），统一采用10 mm球、10 mm体素、摩擦系数0.6等标准参数进行评估。

**📈 对比分析**

通过与DOF计数、总关节范围、工作空间体积、工作空间交集、Yoshikawa可操纵性、全局条件指数等传统指标做Spearman相关比较，KaRMA‑T与工作空间交集相关性最高（ρ≈0.92），但能揭示平移‑旋转分离，并与已有任务基准结果保持一致。

**⚠️ 局限性**

局限性包括仅适用于两指精准握持、球形物体、仅运动学分析、未考虑动力学、重力、控制策略等因素；结果受局部线性化与离散化约束影响。

---

## 184. DeltaPrompts: Escaping the Zero-Delta Trap in Multimodal Distillation

**arXiv ID:** 2605.15532 | [PDF](https://arxiv.org/pdf/2605.15532v1)

**作者:** Jaehun Jung `[一作]` (NVIDIA Research), Yejin Choi `[通讯]` (NVIDIA Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于答案差异度（Δ）的蒸馏提示筛选方法，识别标准数据集中大量零差异提示导致的学习瓶颈，并构建了两阶段的种子引导与技能引导合成管道，生成了20万条高差异、多样化的视觉语言推理提示，使用这些提示对Compact VLM进行蒸馏，显著提升模型推理能力。

**💡 创新点**

创新点在于①首次将答案差异度定义为衡量教师与学生输出分布差距的指标；②揭示现有推理数据集大部分提示为零差异，蒸馏信号不足；③设计了基于教师模型的种子引导+技能引导两阶段合成流程，主动生成既具高差异度又保持多样性的提示。

**🔧 技术方法**

使用的技术包括：基于答案分布的抽样与LLM判等同技术计算Δ；E‑Vendi多样性度量；对抗式知识蒸馏（OPD）与对齐的正反KL；教师模型自生成提示的种子引导和技能提炼；以及一致性筛选和拒绝采样。

**📊 数据集**

利用ChartQA、SciVQA、InfoVQA、arXivQA等标准图表/文档推理数据集作为种子，构建了20万条合成提示；训练时采用Qwen3‑VL‑235B‑Thinking作为教师，Qwen3‑VL‑8B‑Thinking作为学生；后续还将数据迁移到GLM‑4系列模型。

**📈 对比分析**

在10个基准（ChartXiv、InfoVQA、ChartQAPro、SEEDBench2‑Plus等）上与原始Qwen3‑VL‑8B‑Thinking、其他开源推理模型及OPD baseline进行对比，Qwen3‑8B‑DeltaThinker在图表/文档推理上平均提升约7%，在感知推理上提升15%；在STEM/数学等跨域任务也保持或略优；SFT实验表明该提示集对非推理模型同样有效，优于混合数据集。

**⚠️ 局限性**

局限性在于：①合成过程高度依赖教师模型的推理质量与一致性；②对不同教师‑学生组合的适配需要重新评估差异度；③在低资源或非文本模态推理场景下效果尚未验证；④生成提示仍需人工或LLM判断的筛选，成本不可忽视。

---

## 185. RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably

**arXiv ID:** 2605.15514 | [PDF](https://arxiv.org/pdf/2605.15514v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 186. PrismQuant: Rate-Distortion-Optimal Vector Quantization for Gaussian-Mixture Sources

**arXiv ID:** 2605.15507 | [PDF](https://arxiv.org/pdf/2605.15507v1)

**作者:** Bumsu Park `[一作]` (POSTECH), Namyoon Lee `[通讯]` (POSTECH)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对高维多模态高斯混合源的构造性变换编码框架 PrismQuant，并给出了完整的 rate–distortion 理论与实现。

**💡 创新点**

创新点在于：①将混合源与标签耦合，只需单一全局逆水位即可实现条件 RD；②构造两阶段编码——无损传输标签后使用组件匹配 KLT + 熵约束标量量化；③证明理论与实践之间的 gap 仅为 H(C)/n，随维度增大可消失。

**🔧 技术方法**

使用的技术包括：EM 训练高斯混合模型、MAP 组件标签判别、组件匹配的 Karhunen–Loève 变换、熵约束标量量化 (ECSQ)、全局逆水位分配以及解析 RD 公式。

**📊 数据集**

实验数据集：人工合成的高斯混合源和真实无线通道状态信息（CSI）数据集 DeepMIMO。

**📈 对比分析**

与单协方差 TC、Swin-NTC、CsiNet 等基线进行对比；在 DeepMIMO 上 PrismQuant 在相同比特率下实现更低的 NMSE，且模型规模与计算复杂度均显著低于学习型编码器；理论上接近 Genie 上限。

**⚠️ 局限性**

局限性：标签误判会影响性能；混合模型参数需要 EM 训练，易受训练数据与初始化影响；对极低比特率或过大混合数时仍有 H(C)/n 成本，且在非高斯源场景下的泛化性需进一步验证。

---

## 187. Understanding CDCL Solvers via Scalability Studies and Proofdoors

**arXiv ID:** 2605.15506 | [PDF](https://arxiv.org/pdf/2605.15506v1)

**作者:** Shimin Zhang `[一作]` (Georgia Institute of Technology), Vijay Ganesh `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8386 | [OpenAlex ID](https://openalex.org/A5052292970)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对766个BMC实例进行规模化研究，量化CDCL求解器在工业实例中的线性、非线性、指数规模行为，并通过“证明门”参数验证其对性能的解释能力；

**💡 创新点**

引入并实证证明证明门参数是唯一能区分线性与指数扩展的结构参数；

**🔧 技术方法**

使用插值、强弱最小化、McMillan插值以及BVE工具计算证明门，并利用Absorption热图评估CDCL对证明门的增量吸收；

**📊 数据集**

使用HWMCC基准中的76,600+个BMC公式（766个家族）；

**📈 对比分析**

通过与CVR、树宽、社区结构等传统参数对比，证明证明门能够准确区分性能类别，并在对线性实例进行打乱实验时观察到证明门增大导致求解时间显著上升；

**⚠️ 局限性**

证明门计算是NP‑hard，导致在指数规模实例上多次超时，且仅对线性与指数极端情况做实验，未覆盖多项式规模家族。

---

## 188. Validated Hypotheses as a Lens for Human-Likeness Evaluation in AI Agents

**arXiv ID:** 2605.15473 | [PDF](https://arxiv.org/pdf/2605.15473v1)

**作者:** Xuan Liu `[一作]` (University of California, San Diego), Haojian Jin `[通讯]` (University of California, San Diego)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出通过已验证的行为假设评估LLM代理的人类相似性，并构建HumanStudy‑Bench平台，对10种LLM与4种代理设计在12个社会科学实验中的推理和效应一致性进行系统评估。

**💡 创新点**

创新点在于将行为假设列表替代传统图灵测试，提供客观、可分解、可扩展的评估框架；引入概率对齐分数（PAS）和一致性分数（ECS）两种新指标；以及创建可重用实验环境的开源平台。

**🔧 技术方法**

采用LLM辅助的过滤‑提取‑执行管道，将实验设计转化为可执行脚本；使用贝叶斯推断与对数变换计算PAS；运用Lin一致性系数与相关性分析计算ECS；并通过统计推断与效果大小对齐评估。

**📊 数据集**

使用12个已复制验证的社会科学实验数据集，涵盖认知偏差、博弈与社会心理学，约6,000个试验样本，样本来自WEIRD（西方、受教育、工业化、富裕、民主）群体。

**📈 对比分析**

评估方法以PAS衡量推理一致性、ECS衡量效应大小一致性；结果显示代理对齐呈双峰分布，整体PAS约0.4‑0.5，ECS低于0.3；代理设计对对齐影响显著且非单调，模型规模和简单混合并不能显著提升对齐。

**⚠️ 局限性**

局限性包括仅覆盖1946‑2007年在WEIRD样本上的经典实验，缺乏对非WEIRD或当代群体的代表性；平台初始规模有限，实验覆盖面不够广；代理设计与温度对齐表现非单调，模型规模并不一定带来更好对齐。

---

## 189. MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models

**arXiv ID:** 2605.15589 | [PDF](https://arxiv.org/pdf/2605.15589v1)

**作者:** Weixin Liu `[一作]` (Vanderbilt University), Zhijun Yin `[通讯]` (Vanderbilt University)

**通讯引用:** 2621 | [OpenAlex ID](https://openalex.org/A5079247989)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MHGraphBench——一种基于知识图谱的精神健康领域评测基准，利用 PrimeKG 的 42 条精神疾病种子节点提取的 1‑hop 子图，构造九个标准化的单选任务（实体识别、实体聚类、事实检验、关系分类、关系预测、两跳验证/选择及其证据增强版本），对大语言模型进行结构化知识和推理能力评估。

**💡 创新点**

创新点包括：① 通过 KG 生成可验证的 QA 题目和对抗性负样本，实现自动化、可复现的评测；② 引入图覆盖率指标（实体、关系、三元组覆盖）和细粒度实体/关系分析，提供模型知识掌握的全局视角；③ 关注输出格式可靠性，揭示多选评测中回答格式对性能的影响；④ 对短 KG 证据片段的增添进行系统实验，揭示其对不同模型的非一致性影响。

**🔧 技术方法**

技术手段涵盖：知识图谱抽取与标准化、KG‑to‑QA 生成、自动负样本构造、基于多选接口的 LLM 评测、覆盖率统计、细粒度实体/关系准确率分析、证据增量实验以及响应格式解析与可靠性评估。

**📊 数据集**

数据集为 PrimeKG 公开 KG，先手工挑选 42 条精神疾病种子节点，随后提取 1‑hop 边，得到 9,242 条原始边并归一化为 4,621 条唯一三元组，整个子图包含 1,847 个实体和 7 类关系，所有评测题目均直接来源于此子图。

**📈 对比分析**

评测对 15 种模型（GPT‑4.1、GPT‑5.2‑chat、GPT‑4o、GPT‑5‑mini、GPT‑5.1‑chat、Qwen2.5‑32B‑Instruct、Mistral‑7B、Qwen2.5‑7B、BioMistral、Llama3‑Med42‑8B、DeepSeek‑R1‑DQ‑7B/32B、Llama3.1‑8B、Meditron、OpenBioLLM‑8B）进行。结果显示 GPT 系列模型在实体识别与关系类型上几近完美（≈95%），但在关系判断（≈58%）和两跳推理（≈60%）上仍显显著不足；开源模型整体表现更差，最高约 56%；证据增量对部分模型有提升作用，但并非统一效应；覆盖率分析表明 GPT‑5‑mini 在三元组覆盖率最高，但整体平均精度最高的仍是 GPT‑4.1。

**⚠️ 局限性**

局限性包括：① 评测仅覆盖 PrimeKG 子图，未覆盖全部精神疾病知识；② 负样本和题目生成完全基于 KG，缺乏人工专家校验，可能导致某些“错误”并非真实错误；③ 评测关注的是与 KG 的一致性，而非临床真实有效性；④ 输出格式不一致会对评分产生影响，导致性能不纯粹；⑤ 任务规模相对有限，难以覆盖复杂的临床决策场景。

---

## 190. CTF4Nuclear: Common Task Framework for Nuclear Fission and Fusion Models

**arXiv ID:** 2605.15549 | [PDF](https://arxiv.org/pdf/2605.15549v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 191. ColPackAgent: Agent-Skill-Guided Hard-Particle Monte Carlo Workflows for Colloidal Packing

**arXiv ID:** 2605.15625 | [PDF](https://arxiv.org/pdf/2605.15625v1)

**作者:** Lijie Ding `[一作]` (Oak Ridge National Laboratory), Changwoo Do `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 3359 | [OpenAlex ID](https://openalex.org/A5058863771)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了ColPackAgent框架，实现了硬粒子蒙特卡洛(HPMC)聚合模拟的自动化四阶段工作流（设置、规划、执行、分析），并演示了交互式、自治式和自我研究模式；

**💡 创新点**

通过将Model Context Protocol（MCP）与agent skill分离，提供工具服务器与流程知识的可移植组合；封装HOOMD-blue的HPMC为Python包colpack；实现跨代理平台可复用的技能文件；在LLM上进行autonomous research和阶段感知基准测试，证明了模型可在完整工作流中保持一致性。

**🔧 技术方法**

使用LLM代理（Claude、Gemini、OpenAI Codex、OpenCode）与MCP工具服务器，Python colpack包包装HOOMD-blue HPMC，Markdown技能文件，自动化脚本，OpenRouter/Vertex接口进行模型调用；实验中还用到了HOOMD-blue的轨迹、Freud库进行分析。

**📊 数据集**

主要使用HOOMD-blue生成的模拟轨迹数据，包括2D硬盘冻结压强、3D立方体NPT扫描、二元混合体的g(r)等；对比文献Bernard–Krauth值评估结果。

**📈 对比分析**

设计17个单轮提示的阶段感知基准，测量成功率、API成本、输入/输出token，结果显示Qwen3-Next 80B Instruct完成全部17个任务且成本最低；Claude Opus 4.7和Gemini 2.5 Flash次之，展示了模型在完整工作流中的性能差异与成本效益。

**⚠️ 局限性**

当前仅支持HOOMD-blue的硬粒子Monte Carlo，无法处理更复杂力场或分子动力学；对形状混合有限制（如椭圆与多边形不可混合）；仅覆盖四个工作流阶段，未包含数据解释与报告撰写；基准测试覆盖的模型和任务有限，对更大规模或长时间任务的鲁棒性待验证。

---

## 192. Toward LLMs Beyond English-Centric Development

**arXiv ID:** 2605.15613 | [PDF](https://arxiv.org/pdf/2605.15613v1)

**作者:** Sho Takase `[一作]` (CyberAgent), Ukyo Honda `[通讯]` (CyberAgent)

**通讯引用:** 59 | [OpenAlex ID](https://openalex.org/A5071438625)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过生成大量序列并使用 fastText 检测语言，估计开源 LLM 训练数据的语言分布，并评估持续预训练在提升目标语言文化理解方面的成本效益。

**💡 创新点**

提出用生成序列推断训练数据语言分布的创新方法，并发现持续预训练在目标语言上的性能提升与从零训练的成本效益相近，提示未来 LLM 开发需更注重语言单独投资。

**🔧 技术方法**

生成模型序列、fastText 语言识别、Kendall 相关系数评估、LM Evalution Harness 与 FlexEval benchmark、回归分析以及对比实验。

**📊 数据集**

CommonCrawl、Llama 2 报告数据、MGSM、MMLU ProX、NIILCQA、JMMLU、JAQKET 等。

**📈 对比分析**

对比方法：将持续预训练模型与其基础模型以及专门日语训练模型在语言无关的 MGSM、MMLU ProX 和文化理解（NIILCQA、JMMLU、JAQKET）基准上的表现进行评估。结果显示，持续预训练在目标语言上的性能提升与额外计算成本呈对数关系，整体性能与从零训练相当；在语言无关基准上提升有限。

**⚠️ 局限性**

局限性：仅针对日语进行持续预训练评估；模型规模限制在 32B；估计方法依赖生成序列和 fastText，缺乏对其他模型的验证；对微调模型的相关性较低；未涵盖 MoE 架构和多模态模型。

---

## 193. IO-SVD: Input-Output Whitened SVD for Adaptive-Rank LLM Compression

**arXiv ID:** 2605.15626 | [PDF](https://arxiv.org/pdf/2605.15626v1)

**作者:** Ali Abbasi `[一作]` (Vanderbilt University), Soheil Kolouri `[通讯]` (Vanderbilt University)

**通讯引用:** 3480 | [OpenAlex ID](https://openalex.org/A5068682350)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种后训练压缩方法IO‑SVD，用双向KL感知白化的SVD压缩以及自适应秩分配和损失感知重映射，显著降低LLM和VLM的模型体积与推理延迟，同时保持较低的性能损失。

**💡 创新点**

创新点在于：①用第二阶KL近似构造双向输入/输出白化空间，捕捉权重扰动对预测分布的敏感度；②在全局压缩预算下，基于第1阶校准损失对奇异成分进行贪婪排序，得到异质化的秩分配；③将SVD截断与8位量化结合，使用预测的量化损失选择行实现损失感知的量化重映射。

**🔧 技术方法**

主要技术包括：KL近似二阶导数、输入/输出双侧白化、奇异值分解与截断、基于梯度的奇异成分重要性评分、贪婪秩分配、损失感知的量化重映射，以及KV/ V‑cache压缩实现推理加速。

**📊 数据集**

实验使用多种数据集：WikiText‑2、Penn Treebank (PTB)、C4、OpenBookQA、ARC‑Easy/Challenge、WinoGrande、HellaSwag、PIQA、MathQA、ScienceQA‑IMG、SEED‑Bench、SmolVLM2B等；在这些数据集上评估困惑度（PPL）和零样本下的任务准确率。

**📈 对比分析**

与ASVD、SVD‑LLM、Dobi‑SVD、ZS‑SVD等SVD压缩基线以及剪枝方法对比，IO‑SVD在保持相同压缩比例（如0.8、0.6、0.4）时，PPL下降、零样本准确率提升，尤其在更激进的压缩下差距更大；在推理速度和内存占用上，IO‑SVD可实现近4×吞吐量提升和显著减少KV缓存所占内存。

**⚠️ 局限性**

局限性：①输出侧曲率仅用top‑K词汇近似，可能忽略长尾词的敏感性；②秩分配采用贪婪去除策略，未保证全局最优；③验证范围限于13B规模模型，尚未证明在更大模型（如70B/200B）上的可扩展性。

---

## 194. Do CFLOBDDs Actually Make Use of Linear Structure?

**arXiv ID:** 2605.15552 | [PDF](https://arxiv.org/pdf/2605.15552v1)

**作者:** Meghana Aparna Sistla `[一作]` (University of Texas at Austin), Thomas W. Reps `[通讯]` (University of Wisconsin)

**通讯引用:** 20597 | [OpenAlex ID](https://openalex.org/A5066155126)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `fede83ac-7505-405f-ab37-e7284695c47f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了 CFLOBDDs 的线性结构对压缩效果的影响，并提出了仅具层次结构的新型 Tree‑Automata‑Inspired Decision Diagrams（TIDD）进行对比。

**💡 创新点**

证明 CFLOBDDs 通过线性结构实现指数级压缩，而 TIDD 仅依赖层次结构，揭示两者表达能力的根本差异。

**🔧 技术方法**

采用树自动机模型、Myhill–Nerode 定理、交叉积与最小化算法、矩阵乘法与采样等技术构建与分析这两类结构。

**📊 数据集**

在量子电路模拟（GHZ、Bernstein‑Vazirani、Deutsch–Jozsa）以及 Hadamard、Equality 等函数上进行实验验证。

**📈 对比分析**

通过比较同一变量排序下的节点/边数、构造时间和最大中间状态大小，实验结果表明 CFLOBDDs 在量子模拟中显著优于 TIDD。

**⚠️ 局限性**

TIDD 缺少线性结构导致无法实现 CFLOBDDs 的压缩，实验与理论均显示线性结构是必不可少的，缺乏通用的压缩方法。

---

## 195. Calibrating LLMs with Semantic-level Reward

**arXiv ID:** 2605.15588 | [PDF](https://arxiv.org/pdf/2605.15588v1)

**作者:** Fengfei Yu `[一作]` (University of California San Diego), Rose Yu `[通讯]` (University of California San Diego)

**通讯引用:** 6711 | [OpenAlex ID](https://openalex.org/A5057778679)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Calibration with Semantic Reward（CSR）框架，直接在语义空间对大语言模型进行校准，无需显式置信度输出。

**💡 创新点**

创新点在于将语义一致性作为隐式置信度指标，结合可验证正确性奖励和语义校准奖励，使模型在生成答案时自然形成语义聚集与分散，进而实现可解释的置信度估计。

**🔧 技术方法**

采用强化学习与可验证奖励（RLVR）结合语义校准奖励；利用语义熵（Semantic Entropy）评估不确定性；使用多样本并行采样（K=8）与群组相对策略优化（GRPO）。

**📊 数据集**

在HotpotQA（训练与ID评估）以及TriviaQA、MSMARCO、NQ‑Open（OOD评估）四个开放式问答数据集上进行实验；使用Llama‑3.1‑8B‑Instruct、Qwen2.5‑7B‑Instruct、Mistral‑7B‑Instruct三大模型。

**📈 对比分析**

与基线（Base、RLVR、RD、RLCR）相比，CSR在ECE与AUROC上普遍显著提升，ID场景ECE下降多达40%、AUROC提升多达31%；token成本与Base/ RLVR相近，显著低于需要顺序生成的RLCR。

**⚠️ 局限性**

主要限制在于需要进行语义等价性判断，若采用LLM判别会产生额外推理开销；使用基于F1的轻量替代时仍有一定性能差距，且对不同任务的泛化受限。

---

## 196. CrystalBoltz: End-to-End Protein Structure Determination via Experiment-Guided Diffusion for X-Ray Crystallography

**arXiv ID:** 2605.15564 | [PDF](https://arxiv.org/pdf/2605.15564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 197. Position: Artificial Intelligence Needs Meta Intelligence -- the Case for Metacognitive AI

**arXiv ID:** 2605.15567 | [PDF](https://arxiv.org/pdf/2605.15567v1)

**作者:** Sergei Chuprov `[一作]` (University of Texas Rio Grande Valley), Dmitrii Korobeinikov `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5100497706)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了将元认知作为AI系统设计原则，在联邦学习中通过动态监测与控制提升效率与安全，展示了 IntelliFL 框架和案例研究。

**💡 创新点**

创新点在于将人类元认知概念转化为 AI 元控制机制，结合资源合理化与安全防护，并提出可实验的框架与算法。

**🔧 技术方法**

采用元监测函数、元控制策略、动态聚合、早停、主动学习等技术，构建在 FL 架构上的 IntelliFL 框架。

**📊 数据集**

使用 OctMNIST 光学相干断层扫描眼底图像数据集进行实验。

**📈 对比分析**

与 Trimmed Mean、Bulyan 等传统聚合算法对比，PID‑基聚合在收敛速度、异常检测精度和模型准确率方面表现更优。

**⚠️ 局限性**

限制包括元监测/控制的计算开销、元提示的可靠性、理论框架不足以及在大规模、动态环境下的泛化能力待提升。

---

## 198. AstraFlow: Dataflow-Oriented Reinforcement Learning for Agentic LLMs

**arXiv ID:** 2605.15565 | [PDF](https://arxiv.org/pdf/2605.15565v1)

**作者:** Haizhong Zheng `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5073845046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种数据流导向的强化学习框架，取代传统单训练器控制模式，实现多策略协作训练、弹性/跨区域回放、以及可插拔的训练器与推理引擎。

**💡 创新点**

核心创新在于将训练流程拆分为三个独立抽象层：数据流层、Rollout‑as‑a‑Service（RaaS）与训练器，三者通过最小化的接口通信，天然支持完全异步、多策略协同、弹性回放、异构与跨区域资源利用以及可组合的数据算法。

**🔧 技术方法**

使用数据流层管理提示、轨迹、批次和路由；RaaS抽象将轨迹生成包装为服务；训练器抽象化批量消费与权重发布；通过稀疏 delta 权重传输、请求式远程同步、动态采样、GRESO 预筛选、缓冲回放等技术实现高效协同。

**📊 数据集**

评估基准包括数学推理数据集（AIME24、AIME25、MATH500、Minerva）和代码生成数据集（LiveCodeBench v5/v6、Codeforces），并通过模拟的跨区域/异构环境进一步测试。

**📈 对比分析**

与现有框架（如AReaL、Dr. MAS、SLIME、AReaL‑Hex 等）对比：在相同任务下，准确率与传统框架相当或略高，且多策略训练时可获得 2.7× 的训练迭代速度提升；自动弹性扩缩容能在保持相同准确率的前提下减少 13% GPU‑小时；跨区域异构部署维持 67.6% 的平均数学准确率，权重 delta 稀疏率高达 99% 以上，远程同步仅耗数十秒。

**⚠️ 局限性**

局限性主要体现在：①跨区域实验基于仿真，缺少真实大规模分布式部署的实测；②系统整体复杂度较高，需额外的抽象层设计与调度策略；③对极大规模多策略协作（>10 策略）及动态网络环境的鲁棒性尚未在实验中充分验证。

---

## 199. TopoClaw: A Human-Centric and Topology-Aware Agent Operating System

**arXiv ID:** 2605.15556 | [PDF](https://arxiv.org/pdf/2605.15556v1)

**作者:** Heyuan Huang `[一作]` (OPPO Research Institute), Jun Wang `[通讯]` (OPPO Research Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建了一个人性化、拓扑感知的代理操作系统TopoClaw，支持跨设备动作分配、跨用户身份归属以及跨拓扑治理，实现代理在用户分布式数字生态中的自治与协作。

**💡 创新点**

1）将物理设备拓扑与社交关系拓扑双向建模，将执行与身份归属视为运行时原语；2）跨设备动作分配解耦意图与执行；3）跨用户数字双生身份赋权与链路追溯；4）跨拓扑分布式策略执行保障安全与可控性。

**🔧 技术方法**

基于事件驱动异步消息总线的解耦运行时；分布式策略执行点(PEPs)；动态能力注入与技能生态；拓扑模板注册与分发；文本可审计的全局状态管理。

**📊 数据集**

未公开使用公开数据集；主要通过用户交互日志、实验室测试场景验证。

**📈 对比分析**

未给出对比实验与量化指标，主要以功能演示与案例测试展示跨设备、跨用户协作与治理效果。

**⚠️ 局限性**

缺乏大规模真实环境的性能评估；对不同操作系统与网络环境兼容性待验证；安全策略细粒度与动态可调性仍需完善。

---

## 200. Characterizing Learning in Deep Neural Networks using Tractable Algorithmic Complexity Analysis

**arXiv ID:** 2605.15551 | [PDF](https://arxiv.org/pdf/2605.15551v1)

**作者:** Pedram Bakhtiarifard `[一作]` (University of Copenhagen), Raghavendra Selvan `[通讯]` (University of Copenhagen)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5063821969)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并提出了可扩展的算法复杂度估计方法QuBD，用于测量深度神经网络权重在训练过程中的Kolmogorov复杂度，并分析其与泛化、过拟合、grokking等学习动态的关系。

**💡 创新点**

① 将CTM估计推广到任意k-ary对象；② 通过量化+按位平面分解得到更紧的复杂度上界；③ 显示算法信息主要集中在高位平面，可作为模型压缩诊断。

**🔧 技术方法**

Kolmogorov-Chaitin-Solomonoff复杂度理论、Coding Theorem Method、Block Decomposition Method、量化和位平面分解、实验对比GZIP、BDM、随机网络等。

**📊 数据集**

Fashion‑MNIST、CIFAR‑10、ImageNet以及多种实验网络（MLP、Tiny‑ViT、ResNet‑18/50、ViT、EfficientNet、MobileNet）。

**📈 对比分析**

与BDM、GZIP等传统估计对比，QuBD在训练、过拟合、grokking等场景下的Δ_QuBD与准确率高度相关，显示更好的解释性；在量化压缩实验中，QuBD能准确识别可忽略的低位平面，预测PTQ性能，误差小。

**⚠️ 局限性**

假设位平面间独立可能高估复杂度；CTM表有限仍可能对极大结构失效；量化步骤可能引入误差；仅评估权重，对梯度等动态未覆盖。

---

## 201. Transformer-like Inference from Optimal Control

**arXiv ID:** 2605.15608 | [PDF](https://arxiv.org/pdf/2605.15608v1)

**作者:** Aditya Kudre `[一作]` (University of Illinois), Prashant G. Mehta `[通讯]` (University of Illinois)

**通讯引用:** 3147 | [OpenAlex ID](https://openalex.org/A5081314418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于最优控制的推理架构，推导出与Transformer层结构相似的解码器推断方法，并在线性高斯与非线性离散模型上实现。

**💡 创新点**

核心创新在于将Transformer层的自注意力映射为最优控制问题的最优解，从而得到的推断算法（dual filter）与Transformer层结构一致，同时证明在非马尔可夫情形下Transformer具有非马尔可夫优势。

**🔧 技术方法**

使用最优控制理论、Pontryagin最大原理、双重控制系统、贝尔曼-沃尔夫（Baum‑Welch）算法、Transformer（nanoGPT）以及线性高斯/隐马尔可夫模型的分析。

**📊 数据集**

实验数据为人工生成的二值序列，构造自两种周期（长周期与短周期）的隐马尔可夫模型，长度T=64，隐藏状态维数d=16。

**📈 对比分析**

比较方法：对比Transformer（nanoGPT）注意力权重与双重控制权重；评估交叉熵损失。结果显示：当d̂=d时两者权重稀疏且聚焦在信息量大的1上，性能相近；当d̂<d时dual filter性能急剧下降，而nanoGPT保持近似最优，体现非马尔可夫优势。

**⚠️ 局限性**

主要局限：非线性模型仅在马尔可夫（τ=1）情形下给出解析控制公式；框架基于已知模型参数，缺乏学习或参数估计方法；未给出非马尔可夫非线性扩展，需要进一步研究。

---

## 202. VSPO: Vector-Steered Policy Optimization for Behavioral Control

**arXiv ID:** 2605.15604 | [PDF](https://arxiv.org/pdf/2605.15604v1)

**作者:** Xuechen Zhang `[一作]` (University of Michigan), Samet Oymak `[通讯]` (University of Michigan)

**通讯引用:** 2601 | [OpenAlex ID](https://openalex.org/A5050547472)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了向量引导策略优化（VSPO），在强化学习训练中通过不同强度的行为向量生成多样化rollout，以在保持任务准确率的同时可控地调节模型的次级行为。

**💡 创新点**

创新点在于将目标行为编码为潜在向量并直接在采样阶段进行向量steering，从而实现自我蒸馏并显著缓解稀疏行为奖励问题；并在理论上证明相对于奖励塑造，VSPO具有更优的收敛复杂度。

**🔧 技术方法**

技术上基于GRPO框架，使用教师模型构造正负对比向量，结合任务奖励与行为向量强度的线性奖励，以及多梯度更新与KL正则化实现自我蒸馏。

**📊 数据集**

在MMLU‑Pro和MATH两个推理基准上评估，针对专业级解释、置信度表达、误导上下文鲁棒性和简洁性四种目标行为进行实验。

**📈 对比分析**

与教师轨迹蒸馏、SDFT、SDPO、奖励塑造GRPO及文本指导GRPO等多种基线对比，结果显示VSPO在保持或提升任务准确率的同时，在四种目标行为上实现了更强的控制性能。

**⚠️ 局限性**

局限性包括对学习到的向量质量高度依赖；若向量弱或不充分表征目标行为，则向量steering的优势会显著下降；同时向量构造仍需要教师模型支持，缺乏完全自监督的方法。

---

## 203. Bayesian Sequential Verification for Budget-Aware Quantum Program Testing

**arXiv ID:** 2605.15601 | [PDF](https://arxiv.org/pdf/2605.15601v1)

**作者:** Lei Zhang `[一作]` `[通讯]` (University of Maryland Baltimore County), Lei Zhang (University of Maryland Baltimore County)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于贝叶斯序列验证的量子程序测试工作流程，并在Bell态与QAOA MaxCut两种案例上进行了评估。

**💡 创新点**

创新点在于将可获得的参考先验（如无噪声模拟或状态向量计算）嵌入贝叶斯假设检验，利用序列更新与下界停止规则实现预算感知且统计显著的验证。

**🔧 技术方法**

技术包括Beta先验构造、贝叶斯后验更新、一次性下界（LCB）决策规则，以及利用Qiskit在仿真与噪声后端执行的批量测量。

**📊 数据集**

使用的数据集为IBM Fake Backends中的Bell态（2 qubit）和8节点MaxCut（QAOA）实例的模拟/噪声运行结果。

**📈 对比分析**

通过与匹配的固定预算基准比较，贝叶斯序列验证在中等阈值下可节省约1.6-1.7千次测量，PASS率相同或更高；在接近阈值时节省降低或无显著优势。

**⚠️ 局限性**

局限性包括对批量大小、先验强度等参数的敏感性、仅在仿真后端评估、缺乏对真实硬件漂移和非平稳噪声的验证，并且只针对两类简单的成功判定进行测试。

---

## 204. Embracing Biased Transition Matrices for Complementary-Label Learning with Many Classes

**arXiv ID:** 2605.15586 | [PDF](https://arxiv.org/pdf/2605.15586v1)

**作者:** Tan-Ha Mai `[一作]` (National Taiwan University), Hsuan-Tien Lin `[通讯]` (National Taiwan University)

**通讯引用:** 4726 | [OpenAlex ID](https://openalex.org/A5100616524)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Bias‑Induced Constrained Labeling (BICL) 框架，通过人为设计非均匀的互补标签生成机制，让弱监督学习在多类别场景下可扩展；

**💡 创新点**

创新点在于将偏置设计为可控的标签候选集合，利用信息理论证明该偏置降低了互补标签的条件熵，从而显著提升学习效率；

**🔧 技术方法**

采用信息‑理论下界分析、VLM 负向提示标注、预训练编码器聚类、候选集构造与前向校正/互补概率估计等技术实现；

**📊 数据集**

实验覆盖 CIFAR‑10、CIFAR‑20、CIFAR‑100 以及 TinyImageNet‑200 四个数据集；

**📈 对比分析**

与统一假设、CLImage、ACLImage 以及标准监督进行对比，BICL 在 CIFAR‑100 上提升 7×（从 6% 至 46%），TinyImageNet‑200 上提升 8×（从 4% 至 32%），在所有基准上均优于现有方法；

**⚠️ 局限性**

局限在于需依赖 VLM 或少量真实标签估计转移矩阵，理论假设为实例无关转移，且在高噪声或聚类失效场景下性能可能受限。

---

## 205. Unsupervised 3D Human Pose Estimation via Conditional Multi-view Ancestral Sampling

**arXiv ID:** 2605.15583 | [PDF](https://arxiv.org/pdf/2605.15583v1)

**作者:** Ryohei Goto `[一作]` (University of Osaka), Fumio Okura `[通讯]` (University of Osaka)

**通讯引用:** 1112 | [OpenAlex ID](https://openalex.org/A5069226668)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种利用预训练的二维运动扩散模型先验实现单视角无三维监督的人体姿态三维重建方法。

**💡 创新点**

创新点在于：①将二维扩散模型的生成先验迁移到三维重建任务；②提出条件多视角先祖采样（c-MAS）框架，使用多视角一致性与几何、骨长约束实现精确三维估计。

**🔧 技术方法**

采用扩散模型（MDM）、多视角一致性优化、三角测量重建、骨长约束以及 Transformer 编码器进行运动序列建模。

**📊 数据集**

训练使用来自 YouTube 的 450 条视频（含瑜伽、拉伸、训练等），评估使用 Yoga90 数据集的 384 条视频。

**📈 对比分析**

与监督方法（Video-to-Pose3D、MotionBERT）及无监督方法 ElePose 对比，Yoga90 上 MPJPE 为 113.37mm，显著优于 128.93mm（MotionBERT）和 129.09mm（ElePose），显示跨域鲁棒性。

**⚠️ 局限性**

局限性包括：单目深度歧义导致误估；对上游二维姿态检测的依赖；骨长约束为软约束，无法完全纠正严重误差。

---

## 206. STAR: A Stage-attributed Triage and Repair framework for RCA Agents in Microservices

**arXiv ID:** 2605.15581 | [PDF](https://arxiv.org/pdf/2605.15581v1)

**作者:** Junle Wang `[一作]` (Beihang University), Wenjun Wu `[通讯]` (Beihang University)

**通讯引用:** 9226 | [OpenAlex ID](https://openalex.org/A5060858375)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

STAR框架通过将LLM驱动的微服务根因分析拆分为四个结构化阶段并实施阶段级审计与修复，实现了诊断错误的定位与纠正。

**💡 创新点**

其创新在于把RCA误差定位为阶段性问题并使用对抗候选评估与快慢路由实现可控、低成本的自我修复。

**🔧 技术方法**

结合LangGraph的节点级重放、Stage-critique、Fast/Slow Routing与对抗式候选评估等技术。

**📊 数据集**

使用公开的AIOps Challenge大规模微服务基准与电力行业项目管理平台的真实生产数据。

**📈 对比分析**

与原始mABC与RCAgent工作流以及多种LLM基础模型进行对比，STAR在根因定位Acc@1/3/5、故障类型F1等指标上提升约15–30%并显著减少修复轮次。

**⚠️ 局限性**

局限包括对阶段划分的粗粒度限制、对多阶段协同复杂性的覆盖不足以及在极端证据稀缺场景下仍可能无法定位正确阶段。

---

## 207. MI-CXR: A Benchmark for Longitudinal Reasoning over Multi-Interval Chest X-rays

**arXiv ID:** 2605.15574 | [PDF](https://arxiv.org/pdf/2605.15574v1)

**作者:** Sunghwan Steve Cho `[一作]` (Seoul National University), Jaeyoung Do `[通讯]` (Seoul National University)

**通讯引用:** 1011 | [OpenAlex ID](https://openalex.org/A5024989829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布MI-CXR基准，用于多间隔纵向CXR序列的多选问答，评估模型在时间事件定位、间隔变更推理和全局轨迹总结等纵向推理任务。

**💡 创新点**

将纵向医学视觉问答细分为时间事件定位、间隔变更推理、全局轨迹总结三类推理能力，构建长时间序列多选问答并提供阶段化诊断框架，揭示模型局部与全局推理的薄弱环节。

**🔧 技术方法**

采用零样本提示、阶段化推理协议、基准数据生成与质量过滤，评估14种VLM在MI-CXR上的表现。

**📊 数据集**

基于MIMIC-CXR-JPG与MIMIC-Ext-CXR-QBA，构建至少5次访问的患者时间线，产生5,311个多选实例。

**📈 对比分析**

通过单步零样本多选评估和阶段化推理对比，平均准确率仅29.3%，略高于随机的20%，表明现有VLM在纵向推理上表现有限。

**⚠️ 局限性**

仅关注视觉与决策层面，未加入实验室数据或文本报告；阶段化框架未揭示内部机制；数据集不涉及更长时域或多模态。

---

## 208. Measuring Maximum Activations in Open Large Language Models

**arXiv ID:** 2605.15572 | [PDF](https://arxiv.org/pdf/2605.15572v1)

**作者:** Luxuan Chen `[一作]` (Shanghai Jiao Tong University), Dawei Yin `[通讯]` (Baidu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对27个近LLaMA后开放LLM检查点进行统一协议下的最大激活幅度测量。

**💡 创新点**

首次系统评估跨家族、跨生成、跨架构的激活最大值，并将其与低位量化误差关联。

**🔧 技术方法**

使用PyTorch钩子收集激活统计，统一数据集，进行INT‑8重建误差评估。

**📊 数据集**

使用5,000样本多域语料（来自RedPajama的数学、代码、英语、知识、中文、低资源语言等）。

**📈 对比分析**

通过同一数据集、同一tokenizer、同一钩子，在不同家族/架构/训练阶段进行对比；发现最大值跨家族相差四个数量级，MoE相比密集模型低14–23倍。

**⚠️ 局限性**

未探究导致激活峰值差异的因果机制，只给出经验观察，且量化评估仅为轻量级INT‑8探测。

---

## 209. Detecting Privilege Escalation in Polyglot Microservices via Agentic Program Analysis

**arXiv ID:** 2605.15569 | [PDF](https://arxiv.org/pdf/2605.15569v1)

**作者:** Penghui Li `[一作]` (Columbia University), Junfeng Yang `[通讯]` (Columbia University)

**通讯引用:** 7817 | [OpenAlex ID](https://openalex.org/A5056288677)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种结合大型语言模型与传统程序分析的代理式框架，用于自动检测多语言微服务中的权限升级漏洞，并在真实项目上验证其有效性。

**💡 创新点**

创新点：1) 设计统一的代码搜索原语（Q_name、Q_ast、Q_flow、Q_cg）以简化跨语言查询；2) 让LLM动态生成分析计划和搜索策略，提升可扩展性；3) 通过跨服务流量建模（Q_globalflow）实现跨微服务数据流追踪；4) 结合SMT约束求解和LLM语义验证，显著降低误报；5) 证明框架可迁移到其他漏洞类型与非微服务场景。

**🔧 技术方法**

使用技术：Claude Sonnet 3.7 LLM（Chain‑of‑Thought推理）、CodeQL（静态分析与查询模板）、SMT（Z3求解器）以及自定义的Python工具链来调度查询与上下文检索。

**📊 数据集**

数据集：①评估语料—25个公开微服务（覆盖7种语言，总计≈6.2M行代码）；②Ground‑truth集—20个已验证的权限升级漏洞；③对比基准—MScan、CodeQL、SWE‑agent等公开工具。

**📈 对比分析**

与基线对比：在44个已知漏洞的集合中，框架检测到39个真漏洞，精度76.5%、召回85.0%；相比MScan（17/44）、CodeQL（7/44）和SWE‑agent（15/44）表现显著提升。Ablation实验表明：代码搜索原语、LLM上下文检索、验证策略各对性能贡献显著。平均分析时间≈1.6小时/应用，API成本≈$18/应用。

**⚠️ 局限性**

局限性：1) 受CodeQL语言支持与动态特性限制，难以处理反射、动态调用、配置文件中的访问控制；2) 对客户端与服务器执行环境的区分不足，导致部分误报；3) 对特定框架（如Spring matrix变量）语义建模不完整；4) 依赖LLM模型，存在推理错误与成本高昂；5) 需要人工审核才能完全消除误报。

---

## 210. GiLT: Augmenting Transformer Language Models with Dependency Graphs

**arXiv ID:** 2605.15562 | [PDF](https://arxiv.org/pdf/2605.15562v1)

**作者:** Tianyu Huang `[一作]` (ShanghaiTech University), Kewei Tu `[通讯]` (ShanghaiTech University)

**通讯引用:** 22948 | [OpenAlex ID](https://openalex.org/A5061216998)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 GiLT 模型，在 Transformer 语言模型中增添依赖图信息，构建并使用依赖图特征调制注意力，无需额外结构标记。

**💡 创新点**

创新点在于使用可增量构建的依赖图（可为句法树亦可为语义图）作为特征序列，直接注入自注意力，而非插入标记或硬约束。

**🔧 技术方法**

采用 Graph-Infused Layers 调制自注意力，biaffine 依赖评分，图特征提取（度、距离、深度），以及 beam 搜索图空间和预训练模型微调技术。

**📊 数据集**

实验使用 BLLIP‑LG 语料及其自动解析的 PSD、PAS、DM 依赖图，BLiMP 与 SG 句法测试集，以及 GLUE 下的 RTE、SST2、MRPC、STS‑B 分类任务。

**📈 对比分析**

与无结构 Transformer、Pushdown‑LM、PLM/DTG 等基线比较，GiLT 在 PPL 上保持相近甚至略优，在 10%BLiMP 与 SG 上取得最高分；GiLT‑GPT2 在下游任务上均优于对照组。

**⚠️ 局限性**

局限性：推理时需对图空间进行 beam 搜索，计算量较大；GiLT‑DP 的性能受树属性未充分利用影响；图空间估计仅为下界。

---

## 211. When Latent Geometry Is Not Enough: Draft-Conditioned Latent Refinement for Non-Autoregressive Text Generation

**arXiv ID:** 2605.15557 | [PDF](https://arxiv.org/pdf/2605.15557v1)

**作者:** De Shuai Zhang `[一作]` `[通讯]` (Beijing Wuzi University), De Shuai Zhang (Beijing Wuzi University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于冻结 BERT 编码器的连续潜在空间非自回归文本生成框架，并通过 DraftPrior 提供可解码的初始潜在，再用 FlowNet 与 MetricNet 进行局部细化，主要针对 ROCStories 进行实验。

**💡 创新点**

创新点在于：①将草稿（Draft）作为实例级引导，确保潜在空间生成的向量在解码器可接受的 basin 内；②将 Riemannian 流匹配与可学习的对角度量（MetricNet）结合，用于局部潜在修正；③系统化地诊断“潜在相似度不等于离散可恢复性”这一失败模式。

**🔧 技术方法**

使用的技术包括：冻结 BERT encoder、平行 decoder、DraftPrior（去噪先验）、FlowNet（流匹配网络）、MetricNet（对角度量网络）、可约束残差修正、以及与目标潜在的 OT 对齐等。

**📊 数据集**

主要使用数据集为 ROCStories（5 句共识故事），采用前 2 句作为 prompt，后 3 句作为 target，固定 64 词长度。

**📈 对比分析**

与压缩 256‑维潜在、无草稿起始、纯高斯起始等基线对比，发现 768‑维 BERT 潜在在解码器可恢复性上显著优于 256‑维，但整体 MAUVE、distinct 等指标仍低；局部流与 MetricNet 对解码器性能提升有限，说明单纯几何改进不足。

**⚠️ 局限性**

局限性包括：仅测试合成受损草稿的细化，缺乏真正的 prompt‑only 生成对比；未加入更强的离散或迭代生成基线；MetricNet 与 OT 的几何改进效果有限；固定长度目标导致尾部重复；整体质量仍不及自回归模型，缺乏实用性能。

---

## 212. An improved boundary-focused adaptive quadtree algorithm for circle-polygon intersection area approximation

**arXiv ID:** 2605.15627 | [PDF](https://arxiv.org/pdf/2605.15627v1)

**作者:** Zeping Yi `[一作]` (Beihang University), Songyi Liu `[通讯]` (Beihang University)

**通讯引用:** 422 | [OpenAlex ID](https://openalex.org/A5022577529)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于曲率和覆盖度的自适应四叉树算法，用以高效计算多圆与复杂多边形的交集面积。

**💡 创新点**

创新点在于：① 将Green定理与Monte Carlo采样结合；② 采用曲率-多重度引导的采样策略，仅在边界复杂区域增大采样量；③ 通过自适应四叉树实现空间自适应分割，显著降低复杂度从O(1/ε²)降至O(1/ε^{3/2})，并保持O(ε)误差上界。

**🔧 技术方法**

核心技术包括：自适应四叉树分割、Green定理解析积分、曲率和覆盖度估计、Monte Carlo子采样、Ray Casting点内判定、参数自适应调节。

**📊 数据集**

实验数据集：① 合成多边形+多圆（多边形顶点数3–50，圆数1–30）；② 实际海岸线多边形（加勒比海大陆沿岸，1284顶点）+71个随机圆（模拟基站覆盖）。

**📈 对比分析**

与Monte Carlo、Uniform Grid、Adaptive Subdivision、Grid Integration、Triangulation及精确Boundary Integration做对比；实验结果显示：在合成测试中相对误差仅0.007%，在实测测试中0.10%；在计算时间上比Monte Carlo、Uniform Grid快数倍，接近或优于Grid Integration和Triangulation。

**⚠️ 局限性**

局限性：对极小误差容忍度（ε<10⁻⁶）或极其复杂/高圆数场景下计算量仍显著；算法复杂度随最小圆半径R_min和圆数n²成正比；未针对三维空间或动态场景做评估；尽管参数稳健，仍需根据应用场景选择C、N_min。

---

## 213. Statistical two-round search for one excellent element

**arXiv ID:** 2605.15612 | [PDF](https://arxiv.org/pdf/2605.15612v1)

**作者:** Nagananda K G `[一作]` (Portland State University), Jong Sung Kim `[通讯]` (Portland State University)

**通讯引用:** 15748 | [OpenAlex ID](https://openalex.org/A5100737768)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究统计两轮搜索问题，目标是通过两轮无噪声子集检验找到至少一个优秀元素。

**💡 创新点**

首次结合稀疏Bernoulli先验与两轮约束，证明最优期望测试量以对数增长，并给出可实现与下界。

**🔧 技术方法**

采用L-不相交族（disjunct family）构造二轮测试方案，并用信息计数和Fano类下界证明对数下限。

**📊 数据集**

无实测数据集，数值示例基于Poisson截断级别和理论计算的常数，展示可行性边界与测试量。

**📈 对比分析**

与经典组检测、单轮搜索以及无噪声情形对比，表明在稀疏情形下两轮仍保持Θ(log n)的测试量，且满足给定成功概率。

**⚠️ 局限性**

局限在于常数项未确定、仅针对无噪声测试、未考虑更高轮适应性或噪声模型，且实现方案相对保守。

---

## 214. TopoEvo: A Topology-Aware Self-Evolving Multi-Agent Framework for Root Cause Analysis in Microservices

**arXiv ID:** 2605.15611 | [PDF](https://arxiv.org/pdf/2605.15611v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 215. Efficient Image Synthesis with Sphere Latent Encoder

**arXiv ID:** 2605.15592 | [PDF](https://arxiv.org/pdf/2605.15592v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 216. RoiMAM: Region-of-Interest Medical Attention Model for Efficient Vision-Language Understanding

**arXiv ID:** 2605.15561 | [PDF](https://arxiv.org/pdf/2605.15561v1)

**作者:** Jiayan Yang `[一作]`, Wenqi Fang `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量级医学视觉语言模型RoiMAM，专门解决医学视觉问答（MedVQA）任务，并通过两大模块实现高效推理。

**💡 创新点**

创新点包括：①训练无关的 ROI 生成模块（RGMo），利用 CLIP 的语义抑制机制精准定位病灶区域；②文本提示增强模块（TPE），通过 CLIP 零样本分类和 BERT 提升跨模态上下文信息；③在保持参数量仅 1.7B 的同时实现与大型模型相媲美或超越的性能。

**🔧 技术方法**

核心技术：CLIP 预训练视觉编码器、BERT 文本编码器、Multi‑modal Information Bottleneck、KeyBERT 关键词抽取、S^3 语义抑制、OpenCV 连接组件分析、低秩适配（LoRA）微调。

**📊 数据集**

使用了 VQA‑RAD、SLAKE、PMC‑VQA 三个公开 MedVQA 数据集进行评估，训练阶段还利用 ROCO 影像子集进行预训练。

**📈 对比分析**

与 8B 级大模型（如 MedVInT‑TD、LLaVA‑Med）及 2B 级模型相比，RoiMAM 在三大基准上均取得最佳或近最佳成绩：SLAKE 关闭集 88.2%（相较 FAVP 88.1% 提升 0.1%）、PMC‑VQA 44.9%（相较 LLaVA‑Med 42.8% 提升 2.1%），且参数量仅 20% 甚至更少。

**⚠️ 局限性**

局限性包括：①仍依赖 CLIP 视觉编码器，对特定医学影像域的适应性可能受限；②S^3 语义抑制阈值需要手动调节，可能影响跨数据集的泛化；③对极其复杂或多病灶图像的 ROI 定位精度尚未彻底验证。

---

## 217. Operator-Controlled 6G: From Connectivity Infrastructure to Guaranteed Digital Services

**arXiv ID:** 2605.15553 | [PDF](https://arxiv.org/pdf/2605.15553v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 218. Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems

**arXiv ID:** 2605.15573 | [PDF](https://arxiv.org/pdf/2605.15573v1)

**作者:** Nurbek Tastan `[一作]` (MBZUAI), Karthik Nandakumar `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种混合范式，称为nexus，通过多代理系统中的协作解决复杂任务，结合了并行和顺序执行的优点。

**💡 创新点**

创新点在于引入了一种可训练的响应条件策略，能够根据初始响应决定是否需要顺序传播，从而在并行和顺序执行之间架起桥梁。

**🔧 技术方法**

使用了轻量级的变换器模型来预测稀疏的有向无环图（DAG），并通过策略梯度优化进行训练。

**📊 数据集**

在AQUA-RAT和GSM8K数据集上进行了实验，使用了10个Qwen2.5-1.5B-Instruct代理。

**📈 对比分析**

与单代理系统、链式思维、自一致性、SelfOrg等方法进行了比较，结果显示该方法在准确性和成本上表现最佳，减少了约35%的令牌使用量，同时提高了准确性。

**⚠️ 局限性**

限制在于该方法的有效性需要在更异构的模型家庭和代理池中进一步验证，当前的实验主要集中在特定的设置上。

---

## 219. AGC: Adaptive Geodesic Correction for Adversarial Robustness on Vision-Language Models

**arXiv ID:** 2605.15584 | [PDF](https://arxiv.org/pdf/2605.15584v1)

**作者:** Zhiwei Li `[一作]` (Chinese Academy of Sciences), Qi Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 96272 | [OpenAlex ID](https://openalex.org/A5100430015)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种完全训练无关的CLIP测试时防御方法Adaptive Geodesic Correction（AGC），通过在特征球面上沿几何锚点进行自适应校正来提升对抗鲁棒性。

**💡 创新点**

创新点在于：①发现不同增强方式对CLIP鲁棒性的影响并非均等，并利用“分类边际”评估确定最可靠的几何锚点；②在无需梯度或参数更新的前提下，采用自适应步长（基于特征偏差与锚点一致性）沿球面 geodesic 校正特征，从而兼顾干净准确率与鲁棒性。

**🔧 技术方法**

技术包括：CLIP零样本分类框架、单位球面几何（geodesic）、分类边际（margin）评估、基于多视角的锚点构造、以及自适应步长公式。

**📊 数据集**

在八个细粒度分类数据集（Caltech101、OxfordPets、StanfordCars、Flower102、FGVCAircraft、DTD、EuroSAT、UCF101）以及三种CLIP视觉编码器（ViT-B/32、ViT-B/16、ViT-L/14）上进行实验。

**📈 对比分析**

与TTC、R‑TPT、TTP、Ensemble、MTA等现有测试时防御方法比较，AGC平均提升鲁棒准确率超过40%，在ViT-B/32上从44.8%提升至97.9%；在速度上相较TTP实现约10×加速，且无梯度消耗，保持与原CLIP相近的干净准确率。

**⚠️ 局限性**

局限性包括：依赖手工设定的超参数（β_clean、β_adv、γ）；只选用单一最优增强（RandomPerspective），可能在不同任务或攻击模型下需重新评估；目前针对的是PGD攻击，尚未验证对更强攻击或自然分布偏移的鲁棒性。

---

## 220. Compositional Jailbreaking: An Empirical Analysis of Mutator Chain Interactions in Aligned LLMs

**arXiv ID:** 2605.15598 | [PDF](https://arxiv.org/pdf/2605.15598v1)

**作者:** Reinelle Jan Bugnot `[一作]` (National University of Singapore), Yue Duan `[通讯]` (Singapore Management University)

**通讯引用:** 1102 | [OpenAlex ID](https://openalex.org/A5059590465)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大型语言模型（LLM）上通过将简单的“mutator”逐步叠加，系统评估组合式破解（compositional jailbreaking）的实际效果。

**💡 创新点**

提出了完整性（completeness）和有效性（validity）两个新的评估指标，用于衡量组合后提示是否保留了所有变换且能提升攻击成功率；并首次在公开模型上对 12 种基础 mutator 的 132 条有序组合进行大规模实验，揭示组合效果高度不均匀、稀疏且模型相关。

**🔧 技术方法**

使用基于 LLM 的自动安全评估器（gpt4o‑m）检测输出是否违规，并通过持久化检测器判断组合提示中每个 mutator 的痕迹；利用这些工具计算 ASR、完整性和有效性。

**📊 数据集**

采用 AdvBench 的 520 条有害提示作为实验输入；对 DeepSeek、GPT‑3.5 和 GPT‑4 三个对齐良好的公开 LLM 进行评估。

**📈 对比分析**

通过与单步 mutator 的攻击成功率（ASR）对比，计算每个组合的 ASR 增益；结果显示大多数组合不超过单一 mutator，只有 5–14% 的有序对在至少一个模型上满足完整性且提升 ASR，且成功组合在不同模型间不一致。

**⚠️ 局限性**

局限性包括：仅采用两步固定顺序的“naïve”链，未探索更深或自适应链；仅使用单一提示集合，可能不具备对未知提示的泛化；评测仅覆盖三种模型，无法覆盖更广泛的 LLM 家族；完整性判定依赖于预训练的持久化检测器，存在误判风险。

---

## 221. Latent Video Prediction Learns Better World Models

**arXiv ID:** 2605.15618 | [PDF](https://arxiv.org/pdf/2605.15618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 222. LRCP: Low-Rank Compressibility Guided Visual Token Pruning for Efficient LVLMs

**arXiv ID:** 2605.15621 | [PDF](https://arxiv.org/pdf/2605.15621v1)

**作者:** Hongyu Lu `[一作]` (Xiaohongshu), Jiawei Li `[通讯]` (Xiaohongshu)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种无训练的视觉词元压缩框架LRCP，利用低秩可压缩性对视觉词元进行重要性评估并裁剪

**💡 创新点**

创新点在于将视觉词元的全局低秩结构建模为共享背景，并用投影残差作为重要性度量，避免了注意力或局部相似度的偏差

**🔧 技术方法**

技术上主要使用PCA估计低秩子空间，计算投影残差得分，按残差排序保留词元，并对丢弃词元进行余弦相似度合并；整个过程无需额外训练

**📊 数据集**

实验涵盖多种图像数据集（GQA、MMBench、MMBench‑CN、MME、POPE、ScienceQA、VQAv2、VQA^Text、SEED、MMVet、LLaVA‑B）以及视频数据集（TGIF、MSVD、MSRVTT）

**📈 对比分析**

与FastV、SparseVLM、PDrop、VisionZip、V2Drop、ApET等主流方法对比，LRCP在保留94.7%图像性能、88.9%词元压缩率的条件下优于其他方法；在Video‑LLaVA上保持97.8%平均准确率、87.5%词元压缩率，显著超过VisionZip（+3.4%准确率）

**⚠️ 局限性**

局限性包括：低秩子空间维度r需要根据模型手动调节；评估仅在公开基准上完成，尚未在真实生产环境中验证

---

## 223. Wind-Aware Optimal Trajectory Planning for Efficient Gliding of Fixed-Wing Aerial Systems

**arXiv ID:** 2605.15619 | [PDF](https://arxiv.org/pdf/2605.15619v1)

**作者:** Luca Morando `[一作]` (New York University), Giuseppe Loianno `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种实时的多成本Bernstein多项式轨迹规划器，用于在风扰动和障碍约束下实现小型固定翼无人机的滑翔与巡航飞行。

**💡 创新点**

创新点在于将净变压计（variometer）方程直接嵌入轨迹优化，结合能量平衡约束、风利用和障碍安全性，实现对滑翔状态的主动规划和在线重规划。

**🔧 技术方法**

使用了差分平坦性理论、Bernstein多项式轨迹表示、离散优化（CasADi+IPOPT）、风估计、净变压计模型以及Dubins路径初始化的巡航段。

**📊 数据集**

利用CFD仿真得到滑翔极线（sink polar）并在实际Strix Stratosurfer平台上进行实验，收集实际滑翔数据验证模型。

**📈 对比分析**

通过与传统TECS控制器及多成本优化（最小抖动/最小时间/风利用）对比，实验显示在5~6 m/s尾风和障碍条件下，滑翔误差RMSE分别为1.09 m/s、0.39 m/s和0.69 m/s，滑翔比约为7–10:1，说明规划器能有效维持目标滑翔速度和下沉率。

**⚠️ 局限性**

局限性包括：1）平台结构非典型滑翔机导致滑翔比低；2）对高风速/极端气象的鲁棒性未充分验证；3）实时计算仍受限于单板GPU资源，需进一步优化。

---

## 224. A Few GPUs, A Whole Lotta Scale: Faithful LLM Training Emulation with PrismLLM

**arXiv ID:** 2605.15617 | [PDF](https://arxiv.org/pdf/2605.15617v1)

**作者:** Shaoke Xi `[一作]` (Alibaba Group), Jingren Zhou `[通讯]` (Alibaba Group)

**通讯引用:** 8022 | [OpenAlex ID](https://openalex.org/A5057864403)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

构建 PrismLLM，在仅有少量 GPU 的环境下，通过上下文切换收集全局执行图，并采用混合仿真技术实现大规模 LLM 训练的高保真模拟。

**💡 创新点**

创新点包括：①使用逻辑‑物理分离的 context‑switch 机制在有限 GPU 上完成全局执行图的生成；②采用跨切片时间校准和通信剪枝，确保虚拟 rank 的重放保持与真实大规模训练一致的时序与通信行为；③实现虚拟化 NCCL 与高效的虚拟 rank 初始化，显著降低资源消耗。

**🔧 技术方法**

主要技术：上下文切换执行、执行图（仅计算/通信节点与依赖）、跨切片时间校准、虚拟化 NCCL 与通信剪枝、虚拟化初始化、混合仿真框架。

**📊 数据集**

使用 Megatron‑LM Qwen‑3 MoE 预训练模型，涵盖 235B、503B、1.01T 三个规模，配合多种 PP、TP、EP 组合进行评估。

**📈 对比分析**

与真实 512/1024/2048 GPU 训练进行对比，平均迭代时间误差 0.58%，峰值 GPU 内存误差 <0.01%，在 8192 GPU 规模下仅需 <1% 原始 GPU，仿真耗时约 80 分钟，资源占用低于 0.4%。

**⚠️ 局限性**

局限性：依赖于已知 GPU 平台；假设训练工作负载周期性、同步；无法在未知硬件上进行前置预测，对非周期性或异构工作负载（如 distillation）支持有限。

---

## 225. Neutral-Reference Prompting for Vision-Language Models

**arXiv ID:** 2605.15615 | [PDF](https://arxiv.org/pdf/2605.15615v1)

**作者:** Senmao Tian `[一作]` (Beijing Jiaotong University), Shunli Zhang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3119 | [OpenAlex ID](https://openalex.org/A5063642673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种可插拔的纠正策略NeRP（Neutral‑Reference Prompting），通过在推理时使用中性文本提示和参考图像来估计并纠正视觉‑语言模型在迁移学习中出现的非对称混淆，从而提升未见类别的识别率而不损失已知类别性能。

**💡 创新点**

创新点在于（1）首次揭示并量化VLM在迁移时的非对称混淆现象；（2）设计中性参考提示与残差先验，构建贝叶斯式后验分数，用以检测并局部翻转易混类；（3）将此策略作为后处理模块，可与任意现有prompt学习器无缝结合，无需重新训练或修改模型参数。

**🔧 技术方法**

技术核心包括：低秩微调建模、正则化的中性参考文本与图像先验、基于邻接图的易混对生成、贝叶斯式后验分数与门控决策。实现时仅需要一次前向推理和少量阈值调优。

**📊 数据集**

实验覆盖十一种基准（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101）以及四种跨域数据集（ImageNet‑V2、Sketch、A、R），并在十个不同的源目标组合上验证跨域泛化。

**📈 对比分析**

与CoCoOp、PromptSRC、MMA、MMRL等四大prompt学习器以及其他SOTA方法对比，NeRP在所有基准上均提升了新类别准确率，且在Harmonic Mean上平均提高约1–2%，在跨域与域迁移任务中同样获得显著的性能提升，且仅增加极小的推理开销。

**⚠️ 局限性**

局限性包括：①仅为推理阶段的后处理，未能深入解决模型本身的偏差来源；②需要在伪基/伪新或伪源/伪目标子集上调优门控阈值，适配性可能受数据分布差异影响；③在极大规模数据或实时场景下的计算效率与可扩展性尚待进一步评估。

---

## 226. PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding

**arXiv ID:** 2605.15609 | [PDF](https://arxiv.org/pdf/2605.15609v1)

**作者:** Shengyin Sun `[一作]` (City University of Hong Kong), Chen Ma `[通讯]` (City University of Hong Kong)

**通讯引用:** 25781 | [OpenAlex ID](https://openalex.org/A5100652421)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Parallel Speculative Decoding (PSD) 框架，结合空间并行解码与时间推测，实现对扩散式大语言模型推理的双重加速。

**💡 创新点**

创新点在于同时压缩空间维度（一次解码多token）与时间维度（多步推测+批量验证），并通过层次接受机制阻止错误传播，使框架对任何并行策略均无关；此外，利用置信度排序构建多深度DAG式草稿无需额外模型调用。

**🔧 技术方法**

采用扩散式语言模型推理流程、置信度排序、DAG多深度推测、批量验证与层次接受技术；并使用配置可调的并行策略与推测深度。

**📊 数据集**

实验使用 GSM8K（数学推理）、HumanEval 与 MBPP（代码生成）三大基准，结合 Dream-v0-Base-7B、LLaDA-1.5、openPangu-7B-Diffusion-Base 三个开源 dLLM 模型。

**📈 对比分析**

与七种基线（六种空间并行、一个时间推测）对比，PSD 在所有模型/任务上实现 3–5.5× tokens/forward，保持或几乎保持贪婪解码精度，显著提升质量–速度 Pareto 前沿。

**⚠️ 局限性**

局限在于假设置信度排名随时间稳定，过深推测可能导致验证成本增加；对代码生成的鲁棒性仍低于数学推理；需要手动调节并行策略与推测深度以获得最佳性能。

---

## 227. See Before You Code: Learning Visual Priors for Spatially Aware Educational Animation Generation

**arXiv ID:** 2605.15585 | [PDF](https://arxiv.org/pdf/2605.15585v1)

**作者:** Yuejia Li `[一作]` (Wuhan University), Mang Ye `[通讯]` (Wuhan University)

**通讯引用:** 12602 | [OpenAlex ID](https://openalex.org/A5008999954)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OmniManim 框架，通过 Vision Agent 进行视觉规划，随后由 LLM 合成可执行的 Manim 代码，并利用渲染反馈进行局部修复，以生成视觉质量更高的教学动画。

**💡 创新点**

创新点包括：① 以渲染反馈为基础的约束式代码生成框架；② Vision Agent 采用粗细两级 Bounding Box 去噪与插值感知目标，专门针对动画中间帧的重叠与关系一致性；③ 通过共享场景状态实现语义解析、视觉规划、代码生成与修复四个模块的可视化协作；④ 新构建的 ManimLayout-1K 与 EduRequire-500 数据集与评测协议。

**🔧 技术方法**

核心技术包括：大语言模型（如 GPT‑5.4、Gemini 3.1）、Diffusion 去噪式布局预测、插值感知损失、结构化渲染诊断与局部修复、共享场景状态的多代理协作。

**📊 数据集**

使用 ManimLayout-1K（1,000 条 Manim 教学动画，22,579 个布局样本）作为训练集，EduRequire-500（500 条专家编写的教学需求）作为评测集。

**📈 对比分析**

与单模型基线（GPT‑5.4、Kimi K2.5、Gemini 3.1、MiniMax‑M2.7、Qwen3‑14B）以及 Code2Video 多代理基线进行对比。OmniManim 在渲染成功率、重叠率、布局质量、动画连续性等视觉指标上均显著优于所有基线；在人类评测中在布局相关维度获得最大提升，整体评分提升约 12 分。

**⚠️ 局限性**

局限性包括：仅支持 Manim 风格的二维布局；关键帧数量受限，难以处理长周期叙事与复杂相机运动；对排版细节与多媒体交互支持不足；当前框架为辅助工具，最终内容仍需人工审校。

---

## 228. Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered

**arXiv ID:** 2605.15622 | [PDF](https://arxiv.org/pdf/2605.15622v1)

**作者:** Sijia Liu `[一作]` (Michigan State University), Yihua Zhang `[通讯]` (Michigan State University)

**通讯引用:** 9167 | [OpenAlex ID](https://openalex.org/A5008614366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出对深度学习中零阶优化（ZO）的重新定位，认为其并非原生无力，而是被低效设计所束缚。

**💡 创新点**

创新点在于提出六个定位点（P1–P6），指出随机方向选择、查询‑方差权衡、方向导数基线、子空间/谱视角、系统优势与任务对齐消解等关键问题。

**🔧 技术方法**

使用技术包括随机梯度估计、稀疏/预处理变体、子空间投影、分布式查询并行、前向梯度基线、任务对齐与去混淆等方法。

**📊 数据集**

论文在对比实验中参考了 SST‑2、RTE、WiC 等自然语言处理下游任务以及 Gemma‑2B 模型，并讨论了常见公开数据集的使用。

**📈 对比分析**

与传统 FO、基线前向梯度以及现有 ZO 方法相比，提出的方法在查询效率、内存占用与分布式通信上具有优势；实验表明在无对齐和对齐两种设置下，性能差距可显著缩小。

**⚠️ 局限性**

主要局限在于方差爆炸与查询成本、对齐导致的结果混淆、以及对高维子空间投影的依赖；未来需进一步探索多任务、量化和自动化设计等技术。

---

## 229. Syntax Without Semantics: Teaching Large Language Models to Code in an Unseen Language

**arXiv ID:** 2605.15607 | [PDF](https://arxiv.org/pdf/2605.15607v1)

**作者:** Vinayshekhar Bannihatti Kumar `[一作]` (AWS AI Labs), Rashmi Gangadharaiah `[通讯]` (AWS AI Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大语言模型在未见编程语言中的算法迁移能力，设计了完全不在预训练语料中的“PyLang”语言，并对多种前沿与开源模型进行零样本与微调实验；

**💡 创新点**

创新点包括：①提出全新、未出现于任何预训练语料的 PyLang 语言，构建可控制语法与语义差异的实验基准；②系统评估模型在新语言中的实现忠诚度，并通过 LLM 评判、CKA 表征相似度、错误拆解等多线索诊断实现忠诚度缺失；

**🔧 技术方法**

主要技术包括 PyLang 语言实现与解释器、Qwen3 与前沿模型的零样本/微调训练、LLM 评判器对算法一致性评估、CKA（Centered Kernel Alignment）表征相似度分析、偏好调优与代码插补等干预手段；

**📊 数据集**

使用的数据集为 2250 道 PyLang/Python 并行训练题、352 道测试题（251 来自 Codeforces、101 来自 MBPP）以及公开的标准代码生成基准；

**📈 对比分析**

比较方法是对同一批问题分别让 Python 与 PyLang 版本的模型生成代码，并测量通过率、LLM 评判算法一致率以及表征相似度；结果显示 Python 模型在 PyLang 上相对优势最高可达 19%，多种干预方法（多任务学习、偏好调优、代码插补、隐空间目标）均未能显著缩小这一差距；

**⚠️ 局限性**

局限性在于实验聚焦单一极简语言 PyLang，未覆盖更复杂或更广泛的新语法；且仅对中小规模模型做了细粒度微调，未探索更大模型在实现忠诚度上的变化；实现忠诚度缺口可能受到 PyLang 设计的限制，需进一步验证。

---

## 230. Offline Reinforcement Learning with Universal Horizon Models

**arXiv ID:** 2605.15603 | [PDF](https://arxiv.org/pdf/2605.15603v1)

**作者:** Hojun Chung `[一作]` (Seoul National University), Songhwai Oh `[通讯]` (Seoul National University)

**通讯引用:** 3821 | [OpenAlex ID](https://openalex.org/A5033764106)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种能够直接采样任意步长未来状态的通用 Horizon 模型（UHM），并在离线强化学习中构建了可扩展的价值扩张算法；

**💡 创新点**

创新点在于将 Geometric Horizon Model 通用化为可自定义步长的预测器，并结合 winsorized horizon 分布和 λ 调度来抑制远期错误，显著提升了离线学习的稳健性；

**🔧 技术方法**

使用的技术包括流匹配（flow‑matching）生成式模型、单步到多步动态推断、winsorized geometric 分布采样、行为混合策略、终止状态处理和指数移动平均（EMA）稳定目标；

**📊 数据集**

实验数据集为 OpenAI Gym Bench（OGBench）中 100 个不同任务，涵盖标准、噪声和长时延推理场景；

**📈 对比分析**

与模型无关的离线 RL 基线（IQL、ReBRAC、FQL）以及模型基线（MOPO、MOBILE、MAC）以及进一步的 ablation 基线（ReBRAC^†、MBTD(λ)、DTD(λ)、GHM）进行对比，UHM 在 100 个任务中平均成功率提升约 14%，在长时延任务中提升超过 69%，在噪声任务中同样优于所有对比方法；

**⚠️ 局限性**

主要限制包括：UHM 对数据稀缺敏感，容易在分布外进行预测；模型容量有限，难以准确捕捉极长步长；未处理高维视觉输入或动作分块，未来工作需要扩展到这些更复杂场景。

---

## 231. Pretraining Objective Matters in Extreme Low-Data FGVC: A Backbone-Controlled Study

**arXiv ID:** 2605.15599 | [PDF](https://arxiv.org/pdf/2605.15599v1)

**作者:** Alexander Hackett `[一作]` (Santa Clara University), Jason Fisher `[通讯]` (IAAIR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在极低数据环境下对翡翠包裹物分级任务进行研究，比较不同预训练目标下的视觉表征质量。

**💡 创新点**

在匹配backbone的前提下，首次用宏观AUC与置换检验严谨评估四种预训练目标（监督、对比、MAE、DINOv3）在线性与非线性探测器上的可分性差异，并引入人工扰动诊断验证表征对局部特征的敏感度。

**🔧 技术方法**

采用ViT‑B/16冻结表征，线性探测器（Logistic、SVM）、非线性探测器（XGBoost、Random Forest），LOOCV、宏观AUC、1000次置换检验和人工扰动对比实验。

**📊 数据集**

37张RGB翡翠图像（类别分布 21/9/7），每张图像均由GIA认证宝石学家标注。

**📈 对比分析**

通过LOOCV和置换检验比较，宏观AUC：监督 0.768、对比 0.735、MAE 0.636、DINOv3 0.625；在非线性探测器中 MAE 的 XGBoost AUC 达到 0.713，优于监督与对比；人工扰动实验显示对比预训练的模型对局部包裹物更敏感。

**⚠️ 局限性**

数据量极小（仅 37 张），预训练数据来源不一致，目标与语料混杂未完全消除，实验结果仅适用于该单一应用场景，缺乏跨域验证。

---

## 232. CM-EVS: Sparse Panoramic RGB-D-Pose Data for Complete Scene Coverage

**arXiv ID:** 2605.15597 | [PDF](https://arxiv.org/pdf/2605.15597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 233. NavRL++: A System-Level Framework for Improving Sim-to-Real Transfer in Reinforcement Learning-Based Robot Navigation

**arXiv ID:** 2605.15559 | [PDF](https://arxiv.org/pdf/2605.15559v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 234. TG-DIN: Theory-Guided Demand Inference Network for Generalizable QoS Measurement and Prediction

**arXiv ID:** 2605.15550 | [PDF](https://arxiv.org/pdf/2605.15550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 235. LDGuid: A Framework for Robust Change Detection via Latent Difference Guidance

**arXiv ID:** 2605.15582 | [PDF](https://arxiv.org/pdf/2605.15582v1)

**作者:** Jiaxuan Zhao `[一作]` (University of Toronto), Ali Bereyhi `[通讯]` (University of Toronto)

**通讯引用:** 333 | [OpenAlex ID](https://openalex.org/A5061064331)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为 LDGuid 的框架，通过自编码器与对抗解码器学习并注入潜在差异向量，显式捕捉遥感双时序图像中的语义变化，用于提高变化检测（CD）模型的鲁棒性。

**💡 创新点**

创新点在于将信息瓶颈方法与对抗自编码相结合，训练出仅包含目标语义差异且背景噪声最小化的潜在差异表示；该差异表示可直接作为引导信号注入多种 CD 结构，显著提升性能。

**🔧 技术方法**

核心技术包括：对抗自编码器（encoder‑decoder 与 adversarial decoder）、信息瓶颈预训练、潜在差异注入机制、可选的细调与两阶段优化，以及对领域特定特征（如 NBR 指数）的利用。

**📊 数据集**

使用四个遥感变化检测数据集进行验证：LEVIR-CD、WHU-CD、SVCD（城市建筑变化）以及 CaBuAr（野火烧毁区域）。

**📈 对比分析**

在 U‑Net、BIT、AERNet 三大基线上进行对比，实验显示 LDGuid 在 LEVIR-CD、WHU-CD、SVCD 上均能提升 IoU（如 WHU-CD U‑Net 提升 23.36%、BIT 提升 9%），在 CaBuAr 上将 IoU 提升至 86.63% 以上，显著优于现有方法；统计显著性检验表明改进均为显著。

**⚠️ 局限性**

局限性包括：对高维多光谱数据（如 12 带 Sentinel‑2）敏感，易过拟合；需要针对具体任务预先设计领域知识（如使用 NBR）才能充分发挥效果；对 AERNet 等已包含语义差异处理机制的模型提升有限；对抗权重 β 的调参仍需经验；训练过程需额外的预训练步骤。

---

## 236. Gaussian Relational Graph Transformer

**arXiv ID:** 2605.15575 | [PDF](https://arxiv.org/pdf/2605.15575v1)

**作者:** Zezhong Ding `[一作]` (University of Science and Technology of China), Xike Xie `[通讯]` (University of Science and Technology of China)

**通讯引用:** 1991 | [OpenAlex ID](https://openalex.org/A5037366245)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种 Gaussian Relational Graph Transformer（GelGT），通过结构-语义协同采样和自适应高斯时间偏置注意力提升关系数据库图学习性能。

**💡 创新点**

创新点：① 结构完整性采样与语义精炼双阶段采样，显著减少结构碎片与语义噪声；② 采用可学习的高斯时间偏置，使注意力更精准地捕捉时间相关节点；③ 通过理论证明保证采样和注意力的有效性。

**🔧 技术方法**

技术：BFS 结构采样、语义相似度过滤、GNN 信息聚合、带高斯时间偏置的多头自注意力、GNN 与注意力融合。

**📊 数据集**

数据集：RelBench（7 个数据库共 30 个任务，21 个节点分类/回归任务），包含 driver-dnf、user-clicks、user-visits、user-ignore、study-outcome、user-churn、item-churn、user-engagement、user-badge、item-sales 等。

**📈 对比分析**

与 RelGT、Griffin、HGT、LightGBM 等基线对比；在节点分类 AUC 上最多提升 6.2%，在节点回归 MAE 上最多提升 13.8%；同时训练/推理速度均低于现有图 Transformer，显示更高效率。

**⚠️ 局限性**

局限性：对采样大小敏感，最佳采样点数为 300；仅在 2-hop 采样下表现最佳，对更大 hop 效果有限；模型仍需在更大规模多表数据库上进一步验证。

---

## 237. H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure

**arXiv ID:** 2605.15701 | [PDF](https://arxiv.org/pdf/2605.15701v1)

**作者:** Jiawei Yu `[一作]` (Chinese University of Hong Kong), Yuchi Ma `[通讯]` (Huawei Cloud Computing Technologies)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为大型语言模型代理设计了一种新型记忆机制，结合时序‑语义树和知识图谱，实现了长期记忆的演化与检索。

**💡 创新点**

创新点在于：①将记忆数据按时间窗口和语义层级组织成树，实现短期记忆向长期记忆的逐步汇总；②构建实体关系知识图谱，支持多跳实体检索；③将两种结构融合，提供统一的检索流程，并加入记忆鲁棒性和时间相关性评分。

**🔧 技术方法**

技术包括：树结构递归合并、语义相似度阈值聚合、实体抽取与消歧、知识图谱构建、多跳图搜索、查询拆分与检索规划、三维评分（语义、时间、鲁棒）及LLM生成。

**📊 数据集**

使用LoCoMo、LongMemEvalS和REALTALK这三个长期记忆QA基准数据集。

**📈 对比分析**

与MemoryOS、Mem0、MemTree、MemOS、Zep、EverMemOS等六个基线在三套数据上比较，取得最高F1和LLM‑Judge Accuracy，尤其在多跳和时间推理任务上提升显著。

**⚠️ 局限性**

局限性包括：索引和检索过程相对耗时与存储成本较高，依赖超参数设定和实体抽取质量，对多模态记忆尚未支持，且在极大规模数据下的可扩展性待验证。

---

## 238. Tighter Regret Bounds for Contextual Action-Set Reinforcement Learning

**arXiv ID:** 2605.15692 | [PDF](https://arxiv.org/pdf/2605.15692v1)

**作者:** Zijun Chen `[一作]` (Hong Kong University of Science and Technology), Zihan Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 3057 | [OpenAlex ID](https://openalex.org/A5001762730)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在每个 episode 观察到的、可变的可用动作集合（action‑set context）下的有限状态、有限动作、有限 horizon 的强化学习问题，提出并分析了一种基于模型的乐观规划算法。

**💡 创新点**

1) 首次给出在情境可变动作集合下的最优极限（minimax）累积后悔界 O(min{√(SAH³K log L), KH})，并在随机情境下退化为 O(min{√(SAH³K), KH})；2) 引入 p‑trimmed 正性间隙（Δ_min^p）构造 gap‑dependent 后悔界，能够显著提升在大间隙情境下的学习效率。

**🔧 技术方法**

使用乐观规划、双倍更新调度、Bernstein 风格的递增奖励（探索奖金）、基于上下文的动作筛选与剪枝，以及严格的概率偏差分析和 Cauchy‑Schwarz 等不等式进行理论证明。

**📊 数据集**

实验基准为一个人工构造的“情境可变动作掩码”实例，设定 S=10、A=5、H=10、K=20000，情境 M₁、M₂ 均匀随机采样，未使用公开真实数据集。

**📈 对比分析**

与先前的睡眠强化学习（sleeping RL）算法以及忽略动作约束的两种基线方法进行对比。实验显示，在所有概率参数 ρ∈{0.2,0.5,0.8} 下，该算法始终取得最低平均后悔率，优于对比方法。

**⚠️ 局限性**

局限性：仅适用于离散表格 MDP；情境仅影响可用动作集合和初始分布，未考虑情境对奖励或转移的变化；未考虑函数逼近或无限维情境；缺乏对真实世界复杂情境的验证。

---

## 239. FRWKV+: Adaptive Periodic-Position Branch Interaction for Frequency-Space Linear Time Series Forecasting

**arXiv ID:** 2605.15690 | [PDF](https://arxiv.org/pdf/2605.15690v1)

**作者:** Qingyuan Yang `[一作]` (Northeastern University), Shizhuo Deng `[通讯]` (Northeastern University)

**通讯引用:** 155 | [OpenAlex ID](https://openalex.org/A5091147127)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了FRWKV+模型，改进了频域时间序列预测中的实部与虚部交互与周期位置校正；

**💡 创新点**

创新点在于引入跨分支门控实现实部虚部频域上下文交换，并设计自适应PhaseGate通过周期位置上下文产生加权的正负门控修正，同时采用自适应信任机制控制修正强度；

**🔧 技术方法**

使用了频域变换（rFFT）、RWKV轻量级状态更新编码、跨分支门控、周期位置上下文编码器（PPCE）、自适应PhaseGate、可学习的修正强度及信任网络；

**📊 数据集**

在七个公开长时序预测基准上评估：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Exchange、ILI；

**📈 对比分析**

与多种基线（线性、注意力、LLM等）以及FRWKV家族内部变体进行匹配种子对比，FRWKV+在28个数据集-预测长度设置中获得最多MSE胜者覆盖，在ETTh2上表现最优；与外部基线比较也处于竞争力位置；

**⚠️ 局限性**

局限在于对周期位置信号的可靠性估计仍隐式、未在更大规模或高维度数据集（如交通类）上验证，且在匹配种子实验中FRWKV+并非平均排名最优，表明其优势受特定数据/周期/指标的限制。

---

## 240. DreamSR: Towards Ultra-High-Resolution Image Super-Resolution via a Receptive-Field Enhanced Diffusion Transformer

**arXiv ID:** 2605.15682 | [PDF](https://arxiv.org/pdf/2605.15682v1)

**作者:** Qingji Dong `[一作]` (ByteDance Inc), Yitong Wang `[通讯]` (ByteDance Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DreamSR，一种利用预训练 Diffusion Transformer 进行超高分辨率图像超分的框架，采用基于补丁的推理并通过双分支 MM-ControlNet 抑制局部过度生成。

**💡 创新点**

创新点包括：1）Patch Context aware MM-ControlNet，融合局部补丁提示与全局提示的交叉注意力；2）两阶段推理结构，使用 Restoration Acceleration LoRA 缩短早期去噪；3）基于图像-图像 (i2i) 的降解管道和感受野增强训练策略。

**🔧 技术方法**

使用技术：Diffusion Transformer（Flux）、MM-ControlNet、LoRA 加速、交叉注意力、图像-图像降解、感受野增强等。

**📊 数据集**

训练数据：580k 高质量通用图像 + 120k 人脸图像，配有文本描述；评估数据集：RealSR、DRealSR、RealLR200、RealLQ250、RealDeg。

**📈 对比分析**

与 GAN（BSRGAN、Real-ESRGAN）和其他 Diffusion 基准（StableSR、DiffBIR、SeeSR、SUPIR、OSEDiff、FaithDiff、DreamClear、DiT4SR）对比，采用 PSNR/SSIM/LPIPS、MUSIQ/MANIQA/CLIPIQA+ 等指标；DreamSR 在非参考指标上普遍领先，且推理时间仅 1+16 步，显著快于多数方法。

**⚠️ 局限性**

局限性：对参考指标的提升有限，仍受生成模型固有偏差影响；对提示对齐的依赖可能限制在极端降解或非文本指导场景中的鲁棒性；在极高分辨率下仍存在较高计算开销。

---

## 241. VAGS: Velocity Adaptive Guidance Scale for Image Editing and Generation

**arXiv ID:** 2605.15661 | [PDF](https://arxiv.org/pdf/2605.15661v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 242. VCG-Bench: Towards A Unified Visual-Centric Benchmark for Structured Generation and Editing

**arXiv ID:** 2605.15677 | [PDF](https://arxiv.org/pdf/2605.15677v1)

**作者:** Xiaoyan Su `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10732 | [OpenAlex ID](https://openalex.org/A5100730785)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VCG-Bench，一个统一的视觉中心基准，用于评估视觉语言模型在结构化图表生成和编辑中的能力。

**💡 创新点**

创新点包括：①引入“Diagram-as-Code”范式，用XML实现可编辑的图表；②整合生成（Vision-to-Code）与编辑（Code-to-Code）两个任务；③设计多维度评估指标（ESR、SCS、CodeXQA、XDRFR）实现对执行性、视觉一致性、语义合规性的细粒度评估。

**🔧 技术方法**

采用了多模态大型语言模型（如 Gemini‑3‑Pro、Claude‑4.5‑Sonnet、GPT‑5.2 等）进行生成与编辑；使用自动化+人工验证的数据构建管线；利用SigLIP2、Gemini‑3‑Pro等工具评估视觉相似度与风格一致性。

**📊 数据集**

构建了包含 1,449 个多领域图表（学术、软件、商务、管理、UI/UX、通用）且涵盖 15 个子领域的税onomies化数据集，数据来源包括开源模板、学术论文、企业图表及爬取的网页图表。

**📈 对比分析**

与现有基准（如 MMCode、PlotCraft 等）相比，VCG-Bench 是唯一同时支持编辑与细粒度评估的多域基准。实验显示：闭源模型（Gemini‑3‑Pro）在执行率、SCS、CodeXQA 上远高于开源模型；开源模型在生成任务中大多失去可执行性，而在编辑任务中已达 99% 以上的执行率，说明编辑能力与视觉理解分离。

**⚠️ 局限性**

局限性包括：仅支持 Draw.io XML；未验证其他图表语言；数据中非英文标签、稀有样式与专业符号覆盖不足；编辑任务局限于可验证的规则编辑，未覆盖语义歧义与全局重构；执行性不等同于语义正确性，仍需人工审核。

---

## 243. MaTe: Images Are All You Need for Material Transfer via Diffusion Transformer

**arXiv ID:** 2605.15660 | [PDF](https://arxiv.org/pdf/2605.15660v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 244. ICP: Exploiting Instruction Correlation for Prefetching Irregular Memory Accesses

**arXiv ID:** 2605.15645 | [PDF](https://arxiv.org/pdf/2605.15645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 245. Propagating Unsafe Actions in LLM Controlled Multi-Robot Collaboration via Single Robot Compromise

**arXiv ID:** 2605.15641 | [PDF](https://arxiv.org/pdf/2605.15641v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 246. SMMBench: A Benchmark for Source-Distributed Multimodal Agent Memory

**arXiv ID:** 2605.15710 | [PDF](https://arxiv.org/pdf/2605.15710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 247. Evaluating Chinese Ambiguity Understanding in Large Language Models

**arXiv ID:** 2605.15635 | [PDF](https://arxiv.org/pdf/2605.15635v1)

**作者:** Junwen Mo `[一作]` (University of Tokyo), Hideki Nakayama `[通讯]` (University of Tokyo)

**通讯引用:** 4276 | [OpenAlex ID](https://openalex.org/A5042739835)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了基于PA理论的中文歧义句子语料库CHA-Gen，并评估LLM对歧义的识别与翻译不确定性。

**💡 创新点**

首次通过半自动流程生成18类潜在歧义结构的高质量句子，结合机器翻译不确定性度量，对中文歧义研究提出新方法。

**🔧 技术方法**

采用LLM生成、自动验证（POS/依存/DeepSeek）与人工校验相结合，利用Chain‑of‑Thought提示和语义熵等技术评估模型。

**📊 数据集**

主要使用自建的CHA‑Gen（5712句，2414歧义）和公开的CHAmbi数据集进行对照实验。

**📈 对比分析**

通过直接查询（识别/比较）和机器翻译（语义熵）两种评测方式，实验表明即使是大模型也仅达到约0.5的准确率，翻译不确定性对歧义句明显增大，指令调优会降低多样性。

**⚠️ 局限性**

当前仅覆盖句法歧义，句子长度偏短，未涉及多义、词切等其他歧义类型，且评测依赖人工标注与翻译样本，限制了泛化与全面性。

---

## 248. Optimum Peer-Turbo: A Scalable and Efficient Solution for P2P Broadcasting

**arXiv ID:** 2605.15715 | [PDF](https://arxiv.org/pdf/2605.15715v1)

**作者:** Muriel Médard `[一作]` (Optimum), Vipindev Adat Vasudevan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 573 | [OpenAlex ID](https://openalex.org/A5037364002)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了一种在星型广播拓扑中利用随机线性网络编码（RLNC）的Peer‑Turbo机制，允许目标节点之间互相交换编码碎片，以加速解码并降低源节点的带宽需求。

**💡 创新点**

其创新点在于：①提出通过目标节点间的RLNC协作实现无显式碎片状态协调的自发解码加速；②引入自由度流体近似模型，用于解析系统级别的解码延迟并量化相较无Turbo方案的收益。

**🔧 技术方法**

主要技术包括随机线性网络编码（RLNC）、星型/树型广播拓扑、离散时间均值场（fluid）近似模型、分布式编码碎片交换。

**📊 数据集**

实验以仿真为主，未使用真实区块链数据集；采用参数设定如k=32，m=1300，α=50/500，p1=0.9，p2=0.9 等进行数值评估。

**📈 对比分析**

通过与无Peer‑Turbo的基准模型进行离散时间仿真比较，衡量生存函数、传播步骤数及源带宽利用率。结果显示Peer‑Turbo可在相同源带宽下将传播延迟降低最多十倍，或在相同延迟下将源带宽需求降低约一阶。

**⚠️ 局限性**

局限性包括：①仅针对单层星形拓扑，未考虑更复杂的网络结构；②采用保守的“更高维度才有用”假设，忽略同维度节点间的潜在协同；③未对真实网络条件、动态拓扑、节点失效和安全激励机制进行评估。

---

## 249. MyoChallenge 2025: A New Benchmark for Human Athletic Intelligence

**arXiv ID:** 2605.15650 | [PDF](https://arxiv.org/pdf/2605.15650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 250. ChronoEarth-492K: A Large Scale and Long Horizon Spatiotemporal Hyperspectral Earth Observation Dataset and Benchmark

**arXiv ID:** 2605.15666 | [PDF](https://arxiv.org/pdf/2605.15666v1)

**作者:** Haozhe Si `[一作]` (University of Illinois Urbana-Champaign), Han Zhao `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 2369 | [OpenAlex ID](https://openalex.org/A5101670508)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ChronoEarth-492K大规模时序红外光谱数据集，并基于其构建ChronoEarth-Benchmark，系统评估时空光谱表示学习的静态、短期与长期任务；

**💡 创新点**

首次提供长达17年的全球时序红外光谱数据和统一评测套件，能够系统评估时序建模与跨域泛化能力；

**🔧 技术方法**

采用光谱自监督预训练（MAE/Hyper-MAE）、可插拔时序模块（最大池化、注意力池化、阶段二时序SSL），以及ViT基线网络；

**📊 数据集**

主要使用NASA EO‑1 Hyperion 2001–2017 轨道红外光谱数据，构成492,354个128×128像素光谱块；

**📈 对比分析**

与多种公开光谱基线（SpectralViT、LESSViT等）以及跨卫星转移（EnMAP）进行对比，ChronoEarth预训练显著提升静态任务表现，时序SSL在短期/长期预测任务中优于简单池化与监督注意力；

**⚠️ 局限性**

时序建模仍是基于预训练空间骨干的扩展，未实现端到端空间-光谱-时序一体化网络；时序观测稀疏且不规则，限制了模型学习一致时序表征的能力；

---

## 251. PCASim: Promptable Closed-loop Adversarial Simulation for Urban Traffic Environment

**arXiv ID:** 2605.15654 | [PDF](https://arxiv.org/pdf/2605.15654v1)

**作者:** Chuancheng Zhang `[一作]` (Shandong University), Bin Jiang `[通讯]` (Shandong University)

**通讯引用:** 73317 | [OpenAlex ID](https://openalex.org/A5113563512)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了PCASim，一种可通过自然语言提示生成并评估安全关键城市交通场景的闭环仿真框架，结合LLM驱动的DSL生成、检索增强生成、对抗式强化学习与迭代式场景仓库管理，实现对车辆鲁棒性与场景多样性的同步提升。

**💡 创新点**

创新点包括：①将检索增强生成（RAG）与大语言模型结合，在用户提示下自动生成可直接执行的DSL场景；②构建可执行的对抗场景仓库并利用RL训练对抗与自我避障车辆，形成闭环验证与数据扩增；③通过语义对齐与自一致性投票机制提升DSL生成的语义准确性与可执行性；④实现多轮迭代过滤与补全，显著提升场景多样性与真实性。

**🔧 技术方法**

主要技术手段：RAG、DeepSeek‑V3等LLM与提示工程（CoT、few‑shot）、语义对齐与自一致性投票、Python中间件将DSL转为仿真代码、PPO强化学习对抗与自我避障车辆训练、Bezier曲线凸优化进行轨迹平滑、FAISS索引+Sentence‑BERT嵌入进行检索、闭环仓库管理与性能评估。

**📊 数据集**

核心数据集为INTERACTION（城市交叉口车辆轨迹），并在此基础上构建多级场景语料库；在实验与对比中使用该数据集生成的场景；此外在局限性讨论中提到可扩展到NGSIM、nuScenes、Waymo等数据集。

**📈 对比分析**

通过与基线方法（Hao等工作）以及Text2Scenario、ProSim、CAT、VCAT等框架对比，实验表明：①DeepSeek‑V3在DSL生成上平均得分80.17，超过DeepSeek‑R1与Qwen2.5‑Plus；②相较基线，PCASim在对抗场景的碰撞率提升约8%，在避障表现上降低约30%碰撞率；③生成的场景成功率提高8%，障碍规避能力提升30%；④通过闭环训练，最终可生成更具挑战性且真实的交通场景。

**⚠️ 局限性**

局限性包括：①场景语料库仅来自INTERACTION，地理与行为多样性有限；②仿真环境使用简化动力学模型，缺乏高保真动力学与多模传感；③依赖外部LLM API，受限于可用性与稳定性；④DSL转换时可能出现地图格式不匹配导致错误；⑤RL训练聚焦基础机动，缺少复杂多阶段动态；⑥未集成高保真仿真器（如CARLA）与更丰富的数据集。

---

## 252. Learning Disentangled Representations for Generalized Multi-view Clustering

**arXiv ID:** 2605.15640 | [PDF](https://arxiv.org/pdf/2605.15640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 253. Towards Code-Oriented LM Embeddings for Surrogate-Assisted Neural Architecture Search

**arXiv ID:** 2605.15649 | [PDF](https://arxiv.org/pdf/2605.15649v1)

**作者:** Pranav Somu `[一作]` (Georgia Institute of Technology), Jason Zutty `[通讯]` (Georgia Tech Research Institute)

**通讯引用:** 112 | [OpenAlex ID](https://openalex.org/A5079412677)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了COLE（Code‑Oriented LM Embedding）方案，将神经网络架构直接以PyTorch代码形式输入冻结的语言模型，提取嵌入并构建轻量级回归器，用于 NAS 的性能预测与搜索加速。

**💡 创新点**

创新点在于：① 直接使用原始代码作为文本输入，利用语言模型已有的编程知识，无需昂贵的 fine‑tuning；② 通过冻结模型与均值池化、PCA 以及 Pairwise Hinge Loss 的组合，实现零样本、低样本下的高排名相关性；③ 在多种搜索空间与基准中验证了相对于传统结构编码的显著优势。

**🔧 技术方法**

技术包括：冻结的 CodeLlama、ModernBERT 等 LM 作为特征提取器；Token‑level 均值池化生成 COLE；PCA 降维；三层 MLP 作为回归头；Pairwise Hinge Loss 用于提升 Kendall’s Tau；代码生成策略（Helper、Inline、Excluded）与可选的 Backbone/Comment 上下文扩展。

**📊 数据集**

使用的数据集与基准：NAS-Bench-201、einspace、NASLib 的 BANANAS 搜索；评估任务包括 CIFAR‑10、CIFAR‑100、ImageNet16‑120；实验中还利用了对应的测试/验证准确率数据。

**📈 对比分析**

比较方法：在同一 LM 与训练样本规模下对比 COLE 与 ONNX‑to‑text、Derivation‑Tree‑String 等文本编码，使用 Kendall’s Tau 衡量预测排名；在 BANANAS 搜索中对比 COLE 与 path encoding，记录评估预算与最终精度；结果显示 COLE 在 NAS-Bench-201 上平均 Kendall’s Tau 提升 5–10%，且在 CIFAR‑100、CIFAR‑10、ImageNet16‑120 的搜索中分别降低评估预算 34%、13%、18%，同时实现更高或相近的最终准确率。

**⚠️ 局限性**

局限性：① 仅在 NAS-Bench-201 与 einspace 进行评估，未在更大或多样化的搜索空间（如 DARTS、EfficientNet 等）验证；② 对比仅涉及文本编码，未与最先进的结构化 GNN 或 FLAN 等模型进行深度对标；③ 仍未探讨 fine‑tuning LM 进一步提升的潜力；④ 代码嵌入的推理成本相对较高，尤其在大规模搜索中需要进一步优化；⑤ 仅针对单目标（准确率）搜索，未验证多目标或硬件感知的适用性。

---

## 254. The Internet Runs on Names

**arXiv ID:** 2605.15646 | [PDF](https://arxiv.org/pdf/2605.15646v1)

**作者:** Geoff Huston `[一作]` (APNIC), Lixia Zhang `[通讯]` (UCLA)

**通讯引用:** 30051 | [OpenAlex ID](https://openalex.org/A5116294214)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统分析了互联网从基于IP地址的原始架构向基于DNS名称的运营模式转变的演进，并阐释了这种转变导致的操作复杂性、脆弱性和安全碎片化。

**💡 创新点**

创新点在于提出将DNS、CDN、HTTP等层级视为一种“逻辑路由”架构，并将其引入对“indirection debt”和故障传播的系统化评估框架，强调名称层已成为负载承载的核心。

**🔧 技术方法**

使用了DNS层级测量技术（CNAME链深度、TTL分布）、EDNS客户端子网（ECS）分析、SVCB/HTTPS记录解析、BGP多主机与CDN多主机对比分析以及常见攻击与故障案例的日志分析。

**📊 数据集**

利用公开的DNS记录数据库、RIPE Atlas探测数据、HTTP Archive traces、以及主要云服务商（AWS、Azure、Google）与CDN（Cloudflare、Akamai、Azure Front Door）公开的故障报告和配置变更日志。

**📈 对比分析**

通过对比传统IP路由的可靠性与名称层路由（DNS steering、CDN orchestrations）在实际故障中的表现，展示了名称层在现代互联网中的承载与传播效果；实验结果显示名称层故障的破坏半径大于单纯的网络层故障。

**⚠️ 局限性**

局限性包括缺乏大规模系统测量与定量数据，研究主要基于案例分析与设计说明；对隐藏后端依赖的全面量化尚未完成，且缺乏统一的测量标准与评估工具。

---

## 255. On the Power of Adaptivity for $\varepsilon$-Best Arm Identification in Linear Bandits

**arXiv ID:** 2605.15663 | [PDF](https://arxiv.org/pdf/2605.15663v1)

**作者:** Arnab Maiti `[一作]` (University of Washington), Kevin Jamieson `[通讯]` (University of Washington)

**通讯引用:** 4189 | [OpenAlex ID](https://openalex.org/A5059086538)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了线性 bandit 中 ε‑最佳臂识别的最小化样本复杂度，系统比较了非自适应固定设计与自适应采样的性能；

**💡 创新点**

创新点在于给出非自适应算法的匹配上下界，证明多数结构化动作集自适应仅提升对数阶，而构造特定动作集实现多项式级别的自适应优势，并提出高效的 ℓ₂‑范数估计子程序；

**🔧 技术方法**

主要技术包括实验设计、Gaussian 宽度分析、Borell–TIS 及 KL‑divergence 下界、Median Elimination、Hanson–Wright 以及子指数尾部估计；

**📊 数据集**

本文为理论工作，未使用实际数据集；

**📈 对比分析**

通过理论样本复杂度比较，非自适应下限为 O(d log(1/δ)/ε² + w(𝒳)²/ε²)，自适应可达到 O(d log(|𝒳|/d)/ε²)；在构造的动作集中，自适应方案实现多项式缩减；

**⚠️ 局限性**

局限性包括仅给出最坏情况下的理论上限、缺乏实验验证、对 Gaussian 噪声假设依赖强、对其他结构化动作集的扩展仍需进一步研究。

---

## 256. Lightweight Cross-Device Sleep Tracking on the WeBe Wearable Platform

**arXiv ID:** 2605.15719 | [PDF](https://arxiv.org/pdf/2605.15719v1)

**作者:** Wei Shao `[一作]` (University of California, Davis), Houman Homayoun `[通讯]` (University of California, Davis)

**通讯引用:** 4360 | [OpenAlex ID](https://openalex.org/A5047382437)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套轻量化、可复现的睡眠追踪管线，直接在原始加速度计信号上进行，使用统一阈值实现跨设备、跨用户睡眠检测。

**💡 创新点**

1）仅用原始加速度信号，无需专有活动计数；2）采用归一化活动得分与全局阈值，实现一次校准即可跨设备部署；3）在可穿戴设备WeBe平台上实现跨设备验证。

**🔧 技术方法**

epoch‑based 活动提取、移动平均平滑、加权上下文得分、阈值分类、规则式睡眠期检测；面向边缘设备的实现。

**📊 数据集**

MMASH公开数据集（ActiGraph 1Hz 加速度 + 自报睡眠标注）以及 WeBe Band 5 次共 3 名受试者的实时加速度数据。

**📈 对比分析**

与 ActiGraph ActiLife 默认流程对比：MMASH 上 TST MAE 41.6 分，起止时间误差分别 6.3/7.4 分；在 WeBe 交叉验证中 TST MAE 27.4 分，起止误差 13.9/8.0 分；ActiLife 在相同数据下误差约 50 分，时间偏差更大。

**⚠️ 局限性**

对短暂觉醒（WASO）检测不佳，易将低运动觉醒误判为睡眠；依赖单一阈值，极端运动或静止场景下表现受限；样本量小，需更大规模验证。

---

## 257. 3D Segmentation Using Viewpoint-Dependent Spatial Relationships

**arXiv ID:** 2605.15708 | [PDF](https://arxiv.org/pdf/2605.15708v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 258. Position: Early-Stage Quality Assurance in Annotation Pipelines Is More Cost-Effective Than Late-Stage Validation

**arXiv ID:** 2605.15714 | [PDF](https://arxiv.org/pdf/2605.15714v1)

**作者:** Sunil Kothari `[一作]` (Centific AI Research), Tao Liu `[通讯]` (Centific AI Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文主张在数据标注流水线中，应优先考虑早期质量保障（pre‑annotation QA）而非后期验证（post‑annotation或post‑review QA）

**💡 创新点**

创新点在于提出QA触发点三元分类（T_0、T_1、T_2）并用参数化错误传播模型揭示时序对错误率与成本的双重影响，呼吁将QA时序作为首要研究变量

**🔧 技术方法**

采用理论模型推导、对比已有文献的时序报告、对主流标注平台进行功能审计，未使用特定算法实现而侧重概念框架与政策建议

**📊 数据集**

未在实验中使用具体数据集，研究基于文献调研与平台分析得出结论；若要验证，可选用ImageNet、CIFAR等公开标注数据集进行时序实验

**📈 对比分析**

比较方式为文献计量（仅2篇说明时序）与平台功能对比（无明确时序参数），未给出定量性能指标，侧重指出缺失与潜在改进路径

**⚠️ 局限性**

局限在于缺乏实证实验验证时序对错误率与成本的真实影响，模型假设与参数无实测支持，研究聚焦单一任务（计算机视觉）可能难以推广至其他领域

---

## 259. Learning Dynamic Pick-and-Place for a Legged Manipulator

**arXiv ID:** 2605.15713 | [PDF](https://arxiv.org/pdf/2605.15713v1)

**作者:** Moonkyu Jung `[一作]` (KAIST), Jemin Hwangbo `[通讯]` (KAIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个层次化强化学习框架，结合鲁棒低层步态控制、统一高层任务策略、在线质量估计和基于成功率的课程学习，实现了四足机器人在携带1.3 kg负载、垂直高度差达1.1 m的动态全身抓取与放置。

**💡 创新点**

创新点在于：①将低层步态与高层任务拆分为两个可训练模块，减少高维动作空间的耦合；②使用LSTM在线估计抓取物体质量，实时自适应控制；③采用成功率驱动的课程学习，使机器人逐步掌握更大距离和更重负载的任务；④在仿真中训练的策略实现零调优的真实世界转移。

**🔧 技术方法**

主要技术包括：层次化强化学习（PPO）、低层闭环姿态跟踪、LSTM质量估计器、基于密集与稀疏奖励的阶段化奖励设计、随机关节扰动的低层训练、以及基于Vicon系统的感知。

**📊 数据集**

数据集为自研仿真环境Raisim，随机采样的六自由度目标与物体属性（尺寸、质量、形状）生成约10,000条训练/测试轨迹；真实世界实验使用Vicon视觉跟踪的物体与桌面坐标，分别进行六种评估场景的10次试验。

**📈 对比分析**

与分段策略、无估计策略、潜在空间自适应策略以及基线VBC相比，本文方法在仿真中达86%成功率、平均完成时间2.07 s，在真实世界中73.3%成功率、平均完成时间4.06 s；相较分段策略提升约10%成功率并缩短约30%时间。

**⚠️ 局限性**

局限性包括：抓手采用二进制开闭指令，无法精准调节抓取力度；对极高、极低高度差和接近重心边界的任务仍易失稳；仅验证了圆柱/棱柱等简单几何形状，未覆盖更复杂物体；真实世界实验受限于Vicon系统和对大质量物体的抓取安全限制。

---

## 260. Handwriting decoding as a challenging motor task for EEG Foundation Models

**arXiv ID:** 2605.15698 | [PDF](https://arxiv.org/pdf/2605.15698v1)

**作者:** Srinivas Ravishankar `[一作]` (University of California San Diego), Virginia de Sa `[通讯]` (University of California San Diego)

**通讯引用:** 3116 | [OpenAlex ID](https://openalex.org/A5071129405)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个无混淆的EEG手写解码数据集，并在此数据集上评估多种基础模型（FM）与专业模型（specialist）的表现，同时探讨运动起始时间、训练数据规模与测试时信噪比对解码性能的影响。

**💡 创新点**

①提出了避免视觉、眼动、肌电、时序等混淆的实验设计；②将手写解码作为FM的细粒度基准任务；③系统分析了训练规模与SNR对性能的相对贡献。

**🔧 技术方法**

使用了EEGNet、EEGConformer、SPaRCNet等专业模型；CBraMod、MIRepNet、REVE等FM；对数据做了ICA、滤波、重新参考、下采样；采用交叉验证、线性探针与全微调；利用Grad‑CAM做解释性分析。

**📊 数据集**

自建的8.5K试次的手写EEG数据集（四名受试者、4个字母，ME与MI两范式），以及公开的另一手写EEG数据集。

**📈 对比分析**

在四向字母分类任务中，专业模型EEGNet平均准确率41.3%，优于所有FM；FM在线性探针或全微调下均未超过EEGNet；提升测试时SNR（多次平均）将单次试验准确率从45%提升至78%，而扩大训练集仅带来有限提升。

**⚠️ 局限性**

目前FM未能在细粒度手写解码上超越专业模型；缺乏针对不同SNR水平的预训练数据；对运动起始时间的不确定性导致性能大幅下降；实验受限于仅四名受试者，难以泛化。

---

## 261. ParamSpMM: Adaptive and Efficient Sparse Matrix-Matrix Multiplication on GPUs for GNNs

**arXiv ID:** 2605.15695 | [PDF](https://arxiv.org/pdf/2605.15695v1)

**作者:** Lixing Zhang `[一作]` (Beijing University of Posts and Telecommunications), Yingxia Shao `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 2581 | [OpenAlex ID](https://openalex.org/A5014615052)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了图神经网络中稀疏矩阵-矩阵乘法（SpMM）在GPU上的实现，提出了可参数化、自动调优的 ParamSpMM 框架。

**💡 创新点**

创新点：①引入 Parameterized Compressed Sparse Row (PCSR) 结构，实现阻塞、负载平衡、线程合并的统一管理；②通过随机森林学习的 SpMM-decider 自动预测最佳参数；③结合 Rabbit 图重排序进一步提升数据局部性。

**🔧 技术方法**

技术：GPU并行 SpMM、CSR/PCSR 存储格式、向量化阻塞、负载平衡、线程合并、Rabbit 图重排序、随机森林 ML 决策器。

**📊 数据集**

数据集：202 个 SNAP / DIMACS10 稀疏矩阵用于 SpMM 基准；6 个 OGB 图用于 GNN 训练；GCN、GIN 两种 GNN 模型。

**📈 对比分析**

与 cuSPARSE、GE-SpMM、GNNAdivisor、DA-SpMM、DGL、GNNAdvisor 等方法对比。平均 1.92× 提升 cuSPARSE，2.41× 提升 GE-SpMM，1.96× 提升 GNNAdivisor，1.64× 提升 DA-SpMM；GNN 训练中 1.60–1.61× 加速。

**⚠️ 局限性**

局限性：需预先生成 PCSR 与重排序，增加预处理开销；极稀疏或特定维度下仍存在性能波动；ML 模型训练成本高，迁移到新硬件或新图结构的泛化性待验证。

---

## 262. Going Beyond the Edge: Distributed Inference of Transformer Models on Ultra-Low-Power Wireless Devices

**arXiv ID:** 2605.15694 | [PDF](https://arxiv.org/pdf/2605.15694v1)

**作者:** Alexander Gräfe `[一作]` (RWTH Aachen University), Sebastian Trimpe `[通讯]` (RWTH Aachen University)

**通讯引用:** 2316 | [OpenAlex ID](https://openalex.org/A5023990842)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出名为name的框架，实现超低功耗无线设备在mesh网络中分布式Transformer推理

**💡 创新点**

创新点在于结合通信感知的分区、pruned all‑to‑all通信原语somegather、训练时模拟丢包的dropout，从而同时降低内存、闪存、通信和计算需求

**🔧 技术方法**

使用somegather（列剪枝通信原语）、自定义分区策略、Dropout模拟丢包、Mixer协议实现高效多跳无线广播，以及CMSIS‑NN 8‑bit核加速

**📊 数据集**

在ETT‑h2、ICD、London‑smart‑meters和Traffic等时间序列数据集上进行训练与评估

**📈 对比分析**

与基线hu2024edge、liu2025communication、bochem2025distributed等方案对比，发现name在16台设备上可执行比单台大14倍的模型；在未剪枝时通信占主导，剪枝后通信/内存/延迟提升3.8–10.96倍，准确率仅略有下降；dropout显著提高对丢包的鲁棒性

**⚠️ 局限性**

限制包括：单个注意力头只能映射到一台设备，导致设备数受头数限制；pruning与dropout是针对固定部署配置训练的，无法直接迁移到不同网络规模或丢包率的场景

---

## 263. How to Choose Your Teacher for Fine Grained Image Recognition

**arXiv ID:** 2605.15689 | [PDF](https://arxiv.org/pdf/2605.15689v1)

**作者:** Oswin Gosal `[一作]` (National Tsing Hua University), Min-Chun Hu `[通讯]` (National Tsing Hua University)

**通讯引用:** 1530 | [OpenAlex ID](https://openalex.org/A5029298215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对细粒度图像识别中的教师模型选择进行大规模实验，并提出基于教师最高两项logit比值的指标R12，用于评估教师对学生的知识蒸馏效果。

**💡 创新点**

设计R12指标捕捉教师的过度自信程度，证明比教师精度和二级软概率更能预测学生性能，提升学生准确率多达9.4%并减少对大模型的依赖。

**🔧 技术方法**

采用标准的KD蒸馏损失（Hinton等），四种训练策略（FZ、FT、CAL、TGDA），并利用R12作为排序依据进行教师挑选。

**📊 数据集**

在8个细粒度数据集（Aircraft、Cars、CUB、Dogs、Flowers、Moe、NABirds、Pets）上开展实验。

**📈 对比分析**

通过对1,216个实验的Spearman相关性评估，R12平均相关系数0.629，强相关占比50.8%；在LCNet-35等小学生上使用R12选教师可使准确率平均提升至80.1%，比基线高约6%，并在某些数据集上提升52.5%等。

**⚠️ 局限性**

仍受教师模型容量、训练策略等多因素影响，R12在某些场景下相关性不足；实验集中在标准KD损失，未探索更复杂蒸馏方法或自监督训练；对模型架构差异的鲁棒性仍需进一步验证。

---

## 264. Sharp Spectral Thresholds for Logit Fixed Points

**arXiv ID:** 2605.15651 | [PDF](https://arxiv.org/pdf/2605.15651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 265. ElasticDiT: Efficient Diffusion Transformers via Elastic Architecture and Sparse Attention for High-Resolution Image Generation on Mobile Devices

**arXiv ID:** 2605.15684 | [PDF](https://arxiv.org/pdf/2605.15684v1)

**作者:** Kunpeng Du `[一作]` (Huawei Technologies), Xinghao Chen `[通讯]` (Huawei Technologies)

**通讯引用:** 3876 | [OpenAlex ID](https://openalex.org/A5006817088)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ElasticDiT，一个面向移动设备的可弹性扩散变压器框架，能够在同一套参数下通过动态调节 VAE 空间压缩率和 DiT 深度，实现低延迟与高保真度的自适应切换；同时引入 Shift Sparse Block Attention（SSBA）和 Tiny DWT‑Distilled VAE（T‑DVAE）进一步降低推理成本；并通过 Flow‑GRPO 强化学习提升语义对齐。

**💡 创新点**

核心创新点包括：① 空间‑深度弹性架构，可在运行时实时切换 VAE 分辨率与 Transformer 深度；② SSBA 将自注意力压缩到块级别并结合窗口移位，降低到近线性复杂度；③ T‑DVAE 通过多级 Haar 变换与两阶段蒸馏实现 1/8 计算量的 VAE，同时保持 Flux‑级别重建质量；④ 统一权重协同训练策略，实现“Flex‑Max/ Lite”共用权重，避免多模型存储。

**🔧 技术方法**

技术手段涵盖：Diffusion Transformer、Shift Sparse Block Attention、Tiny DWT‑Distilled VAE、Flow‑GRPO 强化学习、双路 VAE 路由、层级稀疏剪枝、统一权重协同训练、数据增强与筛选管线。

**📊 数据集**

使用约 1 B 真实图文对（来自公开仓库及工业合作伙伴）+ 5 M 高分辨率样本（HF Hub）+ 1 M 合成样本（Flux‑based 生成）共 1.06 B 训练实例；对 6 M 合成 prompt‑image 集进行过滤，配以 100 K 英文 Prompt 与 5 K 中文 Prompt 进行跨语言训练；数据质量评估采用 AES、LIQE、MUSIQ、HyperIQA 等。

**📈 对比分析**

评估使用 HPSv2（人类偏好评分）与 GenEval（语义对齐）两大指标；在 1024×1024 分辨率下，Flex‑Lite 取得 HPS 32.87（超越 Flux‑12B，参数量 20× 更少），GenEval 73.62（GRPO 后提升）。在 512×512 分辨率下，Layer‑Aware‑Sparse 84.16% 稀疏率实现 2× 推理加速且无质量损失；T‑DVAE 在 1024×1024 上与 Flux‑VAE PSNR、SSIM、FID 接近，且计算量降至 1/4。整体表现表明 ElasticDiT 在移动端高分辨率生成上实现了显著的效率与质量双赢。

**⚠️ 局限性**

局限性包括：① 仍需手动配置 VAE 路径与 Transformer 深度，且对硬件的动态适配依赖策略网络；② 主要验证在图像生成任务，对视频、文本-视频等更复杂场景尚未评估；③ 训练成本与数据规模巨大，难以在小团队快速复现；④ 对极低算力设备（如 8‑bit MCU）可能仍显资源占用。

---

## 266. Few-Shot Large Language Models for Actionable Triage Categorization of Online Patient Inquiries

**arXiv ID:** 2605.15680 | [PDF](https://arxiv.org/pdf/2605.15680v1)

**作者:** Liqi Zhou `[一作]` (University of Pennsylvania), Jiafu Li `[通讯]` (Northwestern University)

**通讯引用:** 11819 | [OpenAlex ID](https://openalex.org/A5100601055)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在低资源标注条件下，利用提示式大型语言模型（LLM）对在线患者咨询进行四类可执行分流（自我护理、预约就诊、紧急临床评估、急诊转诊）的任务；

**💡 创新点**

创新点在于首次将少量示例提示与LLM相结合，评估其在安全导向的分流任务中的表现，并与传统监督模型进行对比；

**🔧 技术方法**

采用了提示式LLM（如Claude Haiku 4.5、GPT‑4等）与少量示例（0-shot、4-shot、12-shot）以及传统TF‑IDF和BioBERT基线模型；

**📊 数据集**

使用了公开的HealthCareMagic‑100K语料库，构建了300例人工校准的黄金评估集、700例自动标注的银级训练集以及40例少量示例池；

**📈 对比分析**

通过宏F1值及安全相关指标（急诊召回率、漏分率等）进行比较，12-shot提示下Claude Haiku 4.5实现宏F1 0.475，略优于BioBERT基线0.378，差距虽在置信区间重叠范围内；

**⚠️ 局限性**

局限性包括对少数示例的依赖导致某些类别（如紧急临床评估）的准确信度不高，且实验数据规模有限，模型尚不适合完全自主部署。

---

## 267. Bridging Silicon and the Hippocampus: Algebro-Deterministic Memory "VaCoAl" as a Substrate for Vector-HaSH and TEM

**arXiv ID:** 2605.15652 | [PDF](https://arxiv.org/pdf/2605.15652v1)

**作者:** Hiroyuki Chuma `[一作]` (Hitotsubashi University), Yoichi Sato `[通讯]` (Shuhari System)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过将代数确定性高维记忆架构 VaCoAl 与神经科学中两大模型 Vector‑HaSH 与 Tolman–Eichenbaum Machine（TEM）相对应，构建了一个桥梁，证明了 VaCoAl 的 Galois‑field 线性反馈移位寄存器（LFSR）扩散实现了这些模型所需的“准正交”散射，并提出了路径积分置信度比（CR2）作为多跳 SWR 重放精度衰减的可解析数学模型。

**💡 创新点**

创新点包括：① 用代数确定性 LFSR 扩散取代随机投影，证明二者在准正交性、第二矩统计和最坏情况 avalanche 行为上等价；② 将 CR2 作为多跳重放精度的乘法形式与实验观察相对应，首次提供可验证的数学模型；③ 将 VaCoAl 的“Rescue/Don't Care”冲突策略映射到海马两条通路的功能对应，提出能量‑容量‑可塑性解释双路径保留的进化理由；④ 在工程层面展示 VaCoAl 在高维记忆基准中的可替代性和能源优势。

**🔧 技术方法**

使用技术包括：Galois‑field LFSR 扩散、块级多重投票（majority voting）、CR1/CR2 置信度计算、两种 LFSR 调度（统一 U 与分块 G）以及在 WikiData DAG 记录上的可解释性路径检索。

**📊 数据集**

主要数据集有：① 人类 intracranial EEG（iEEG）记录中的 SWR 重放实验，用于验证多跳精度衰减；② WikiData 结构化知识图谱的祖孙关系 DAG，用于在 VaCoAl 上演示路径检索与 CR2 评分；③ 公开的高维记忆基准（如文本分类、图像检索）用于对比随机投影与 LFSR 扩散的性能。

**📈 对比分析**

比较方法：将 VaCoAl 的 LFSR 扩散与传统随机稀疏投影在同一实验设置下进行训练与测试；对比 CR2 与实验测得的多跳重放精度，检验乘法衰减公式的拟合度；在 WikiData 路径检索中评估 Rescue 与 Don't Care 模式对搜索精度与能耗的影响。性能方面，VaCoAl 与随机投影在分类准确率上基本相同，但在可重现性、最坏情况鲁棒性与能耗（尤其在统一调度下）显著优于随机投影。

**⚠️ 局限性**

局限性包括：① 乘法衰减模型假设每跳重放事件独立，实际可能存在相关性；② VaCoAl 的离散投票机制仅是 CA3 吸附动力学的离散近似，未能完整捕捉连续时间的吸附过程；③ 对海马的解读主要停留在架构对应层面，未给出具体的细胞层面实现；④ 实验预测依赖于现有 iEEG 解码方法的可靠性，未来需要针对多跳重放的专门实验验证；⑤ 对比基准主要集中在分类与检索任务，尚未在更广泛的神经符号推理任务中评估。

---

## 268. Rethinking the Security of DP-SGD: A Corrected Analysis of Differentially Private Machine Learning

**arXiv ID:** 2605.15648 | [PDF](https://arxiv.org/pdf/2605.15648v1)

**作者:** Wenhao Wang `[一作]` (Monash University), Xingliang Yuan `[通讯]` (University of Melbourne)

**通讯引用:** 3246 | [OpenAlex ID](https://openalex.org/A5064553444)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

重新分析并审计了带有归一化步骤的DP‑SGD（EASGM、ASGM以及Opacus中的FEASGM）在单轮训练中的差分隐私保证，发现其隐私泄露可超过传统SGM基准。

**💡 创新点**

提出了针对归一化DP‑SGD机制的正式 f‑DP 分析框架，给出可计算的下界和上界，并通过紧密审计验证其隐私泄露可超过 SG‑M 预期；同时完成了 Opacus 代码中 FEASGM 方案的审计。

**🔧 技术方法**

使用 f‑DP 交易函数理论、Gaussian DP、概率混合分布下的对数似然测试、归一化变换、紧密隐私审计与可观测辅助信息区分器设计，以及统计检验方法。

**📊 数据集**

使用 synthetic 高维数据、CIFAR‑10、MNIST 和 FMNIST 四个公开基准进行实验。

**📈 对比分析**

通过将 EASGM/ASGM/FEASGM 的单轮隐私保证与 SG‑M 闭式保证对比，并利用 FNR/FPR 统计测试绘制 (ε,δ) 曲线，实验表明在高维小样本或大采样率场景下，真实实现的隐私泄露明显高于 SG‑M 预期，证明存在潜在隐私违规。

**⚠️ 局限性**

审计仅覆盖单轮且小样本，无法推广到多轮高维训练；多轮 ASGM 保证缺乏闭式可视化；大数据规模下审计计算代价高。

---

## 269. Feedback World Model Enables Precise Guidance of Diffusion Policy

**arXiv ID:** 2605.15705 | [PDF](https://arxiv.org/pdf/2605.15705v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 270. Dynamic Chunking for Diffusion Language Models

**arXiv ID:** 2605.15676 | [PDF](https://arxiv.org/pdf/2605.15676v1)

**作者:** Yichen Zhu `[一作]` (Hong Kong University of Science and Technology), James Kwok `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16906 | [OpenAlex ID](https://openalex.org/A5070273088)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于内容感知块划分的离散扩散语言模型 DCDM，能够在保持块级自回归结构的同时实现并行去噪。

**💡 创新点**

创新点在于：①引入 Chunking Attention——一种基于子空间的可微聚类层，用来生成语义块；②通过硬/软聚合兼容的路由方式，使得块划分能在扩散目标下端到端学习；③设计了序列级和批量级负载平衡机制，避免聚类失衡。

**🔧 技术方法**

使用的技术包括：子空间聚类注意力、双向离散扩散、块级自回归、Gumbel-Softmax 负载平衡、全局批量负载平衡、双流训练（噪声+干净输入）、动态块分配、扩散 ELBO 训练以及 MoE 稀疏化扩展。

**📊 数据集**

预训练数据：OpenWebText；下游评测数据集：ARC-C、MMLU、HellaSwag、TruthfulQA、WinoGrande、PIQA、MATH、GSM8K、HumanEval 以及七个零-shot 语言建模基准。

**📈 对比分析**

与 MDLM、BDLM、AdaBlock 以及 MoE 版本在 0.5B 与 1.5B 参数规模下进行对比；在所有指标上 DCDM 均优于 MDLM 与 BDLM，MoE 在 0.4B/1.2B 上进一步提升；在训练曲线上 DCDM 早期就超越 BDLM，收敛更快；平均提升约 1–4%（整体平均提升 0.57/0.33）。

**⚠️ 局限性**

局限性：①性能对聚类数 K 的选择敏感，需手动调优；②若聚类过细或过粗会导致去噪质量下降；③负载平衡机制复杂，增加训练开销；④对极长序列、多语言和硬件加速的适应性尚未充分验证；⑤与传统自回归模型相比，生成质量仍有差距。

---

## 271. Interaction-Aware Influence Functions for Group Attribution

**arXiv ID:** 2605.15675 | [PDF](https://arxiv.org/pdf/2605.15675v1)

**作者:** Jaeseung Heo `[一作]` (POSTECH), Dongwoo Kim `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种交互感知的影响函数，用于衡量一组训练样本对目标函数的联合影响。

**💡 创新点**

创新点在于通过二阶展开目标函数得到非加性的对称交互项，既能捕捉样本间的冗余与互补，又提供闭式近似。

**🔧 技术方法**

利用影响函数理论、二阶泰勒展开、闭式因式分解以及贪心数据选择算法。

**📊 数据集**

在六组数据‑模型对（逻辑回归、MLP、ResNet‑9）以及对 Llama‑3.1‑8B 的指令调优数据上进行评估。

**📈 对比分析**

与传统一阶影响函数、表示相似度和随机选取等基线比较，Spearman 相关性更高，在 Llama‑3.1‑8B 的下游任务中取得 5/7 项目的最佳成绩；在小规模模型上子集多样性与随机相当，基线则偏向单一类别。

**⚠️ 局限性**

限制：贪心选择缺乏最优性保证，且估计误差取决于 Hessian 近似的精度。

---

## 272. Scale: Deep Reinforcement Learning for Container Scheduling in Serverless Edge Computing

**arXiv ID:** 2605.15704 | [PDF](https://arxiv.org/pdf/2605.15704v1)

**作者:** Chen Chen `[一作]` (Nottingham Trent University), Lei Jiao `[通讯]` (University of Oregon)

**通讯引用:** 5508 | [OpenAlex ID](https://openalex.org/A5053369746)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对服务器无服务器边缘计算中的容器调度问题，提出了一种面向服务水平目标（SLO）的调度框架；

**💡 创新点**

创新点在于将容器复用与调度分离，构建了两阶段层次化行动空间的actor‑critic深度强化学习模型，并将SLO、端到端延迟与数据局部性统一考虑；

**🔧 技术方法**

主要技术包括基于PPO的Actor‑Critic网络、离散多维动作空间、负延迟奖励函数以及对动态负载与节点位置的状态表示；

**📊 数据集**

使用了华为云的真实函数调用日志（141天、200个函数）与澳大利亚墨尔本CBD地区125个边缘节点的EUA网络拓扑数据；

**📈 对比分析**

与ILP求解器Midaco以及多步DQN基线对比，实验显示在端到端延迟上与Midaco相差1.11~1.15倍，SLO违约率保持在4.6%~4.9%区间，决策时间则比Midaco低99%（约0.02 s/请求）；

**⚠️ 局限性**

局限性包括仅关注延迟与SLO，未考虑能耗优化；数据集为云端调用日志而非真实边缘场景；实验仅在模拟环境中验证，缺乏实际部署验证。

---

## 273. Distributed Zeroth-Order Policy Gradient for Networked Multi-agent Reinforcement Learning from Human Feedback

**arXiv ID:** 2605.15697 | [PDF](https://arxiv.org/pdf/2605.15697v1)

**作者:** Pengcheng Dai `[一作]` (Singapore University of Technology and Design), Wenwu Yu `[通讯]` (Southeast University)

**通讯引用:** 26235 | [OpenAlex ID](https://openalex.org/A5100627758)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种分布式零阶策略梯度算法，在网络化多智能体强化学习（NMARL）框架下，仅通过人类偏好反馈（不使用显式奖励）实现协同优化。

**💡 创新点**

创新点包括：①基于时空截断轨迹的本地偏好反馈机制，使得每个智能体只需访问其κ-hop邻域信息；②使用高斯扰动而非单位范数扰动，支持完全分布式采样；③在该框架下给出完整的收敛分析，证明算法收敛到ϵ-平稳点且样本复杂度为多项式；④通过理论与实验验证其可扩展性和有效性。

**🔧 技术方法**

核心技术：零阶策略梯度（Gaussian perturbation）估计；Bradley‑Terry/线性偏好模型；κ-hop邻域分布式通信；强化学习理论（政策梯度、Lipschitz连续性、偏差/方差分析）。

**📊 数据集**

实验数据集：GridWorld（5×5网格）和 Predator‑Prey（8×8网格，20 预期+10 受害者+10 障碍物）模拟环境。

**📈 对比分析**

与集中式偏好强化学习 DPO 以及奖励驱动分布式 SAC 进行对比。实验表明：在 GridWorld 环境中，算法能接近最优解且收敛更稳；在 Predator‑Prey 环境中，显著优于两种基线。Ablation 研究显示，采样次数 K、偏好评估数 M、轨迹长度 H 和邻域半径 κ 对性能影响明显，符合理论预测。

**⚠️ 局限性**

局限性：①时空截断参数 κ 与 H 的选择导致近似误差，无法完全捕获全局协同信息；②算法对大量人类偏好评估（M）依赖较高，实际部署时评估成本可能成为瓶颈；③理论假设（有限状态动作、正则性、偏好模型正确性）可能在复杂现实场景中不完全成立；④缺乏对偏好模型错配的鲁棒性分析。

---

## 274. SEED: Targeted Data Selection by Weighted Independent Set

**arXiv ID:** 2605.15691 | [PDF](https://arxiv.org/pdf/2605.15691v1)

**作者:** Yuan Zhang `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 11435 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 SEED 数据选择框架，将数据子集挑选建模为相似性图上的加权独立集（WIS）问题，解决了传统 WIS 在节点权重噪声和图结构失衡上的两大失败模式；

**💡 创新点**

创新点在于（1）通过双向显著子空间（mutual influence subspace）对节点权重进行校准，抑制梯度噪声；（2）利用局部尺度归一化（local scale normalization）动态调整边阈值，消除多域数据导致的图结构不平衡；

**🔧 技术方法**

核心技术包括：梯度轨迹影响度估计、加权独立集近似求解（贪婪+GPU加速的相似度搜索）、互惠子空间筛选和基于邻域密度的边构建；

**📊 数据集**

实验数据集涵盖多领域文本（FLAN‑V2、CoT、Dolly、OpenAssistant‑1）、指令调优评测集（TyDiQA、MMLU、BBH）、视觉指令调优的 Honeybee‑1M、语义分割的 GTA5/Cityscapes 等；

**📈 对比分析**

与随机采样、InfoMax、TAROT、MoNA 等基线对比，SEED 在 5%/1% 指令调优、5% 视觉指令调优、20% 语义分割等设置下均实现或超过全量训练的性能，且在代理模型场景下可将 GPU 成本降低 2.5×；

**⚠️ 局限性**

局限性包括：需在大规模数据上预先计算梯度并构建相似性图，计算成本仍较高；节点权重与边阈值的超参（k、α、τ）需经验调优；在极度稀疏或极端异构数据集上，局部尺度归一化可能仍难以完全平衡图结构。

---

## 275. ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal Large Language Models

**arXiv ID:** 2605.15687 | [PDF](https://arxiv.org/pdf/2605.15687v1)

**作者:** Jiahui Guang `[一作]` (Harbin Institute of Technology), Zhaoquan Gu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 3817 | [OpenAlex ID](https://openalex.org/A5070856186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可控的多模态模型消忘框架ASRU，结合激活导向与强化学习，实现对敏感知识的有效遗忘并保持模型实用性。

**💡 创新点**

创新点：①首次将生成质量作为消忘评估维度；②用激活导向快速构建拒绝原型；③通过组相对策略优化（GRPO）学习细粒度的忘记–保留边界；④仅需少量监督保留样本即可实现高效消忘。

**🔧 技术方法**

核心技术：激活导向（更新单一下投影矩阵）、Group Relative Policy Optimization、定制化奖励函数、边界集构建与监督保留集。

**📊 数据集**

主要数据集：MLLMU-Bench（含忘记集、保留集、测试集、真实名人集），Qwen-3-VL-4B/8B、LLaVA-1.5-7B模型；对CLEAR数据集和Real Celebrity子集做进一步验证。

**📈 对比分析**

与GA、GA_diff、NPO、KL_Min、MMUnlearner等基线对比，ASRU在5%忘记率下平均提升24.6%消忘效果、5.8×生成质量，同时保持或提升保留集性能，显著优于所有基线。

**⚠️ 局限性**

局限性：仅验证于图像-文本场景；对边界集构造高度敏感；仍需更大规模的多模态数据和对抗测试；未深入探讨隐私泄露程度和长期泛化能力。

---

## 276. Coalgebraic Non-Wellfounded Proofs: Recursiveness and GTC

**arXiv ID:** 2605.15664 | [PDF](https://arxiv.org/pdf/2605.15664v1)

**作者:** Mayuko Kori `[一作]` (Kyoto University), Mayuko Kori `[通讯]` (Kyoto University)

**通讯引用:** 30 | [OpenAlex ID](https://openalex.org/A5003961991)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了基于余子结构的非良序证明框架，并给出了全局轨迹条件的范畴化与可证性保证。

**💡 创新点**

创新点在于将全局轨迹条件与递归余子同构关联，利用 adjunction 的图结构实现非良序证明的唯一解释。

**🔧 技术方法**

主要技术包括余子同构、可多项式 functor 在预射范畴上的图、adjunction 的 ca‑morphism 对应、以及递归余子与全局轨迹条件的等价性。

**📊 数据集**

该工作未使用具体实验数据集，而是以模态 μ‑算子、HFL 等理论语言为例进行形式化验证。

**📈 对比分析**

由于该研究为理论框架，没有直接的实验对比；作者通过实例证明与传统循环证明的一致性与更广泛的可适用性。

**⚠️ 局限性**

局限性在于未涉及完整的完备性与裁剪消除等问题，且对特殊非良序证明的实现细节仍需进一步探讨。

---

## 277. DealMaTe: Multi-Dimensional Material Transfer via Diffusion Transformer

**arXiv ID:** 2605.15681 | [PDF](https://arxiv.org/pdf/2605.15681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 278. EntropyScan: Towards Model-level Backdoor Detection in LVLMs via Visual Attention Entropy

**arXiv ID:** 2605.15711 | [PDF](https://arxiv.org/pdf/2605.15711v1)

**作者:** Xuanyu Ge `[一作]` (China University of Geosciences), Xilin Chen `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于视觉注意力熵的模型级后门检测方法EntropyScan，用以识别已植入后门的大型视觉语言模型（LVLM）；

**💡 创新点**

创新点在于发现后门注入会破坏跨模态对齐，导致初始层视觉注意力分布出现结构性异常，并通过Tsallis熵量化此异常；同时提出参考锚定Z-score归一化，能够在无触发器知识且仅使用少量干净样本的情况下完成检测；

**🔧 技术方法**

核心技术包括Transformer的注意力机制提取、条件概率重归一化、Tsallis熵计算、参考模型的统计归一化以及基于AUC阈值的二分类决策；

**📊 数据集**

使用COCO2017验证集（200张图像）做样本验证；对抗实验基于三种后门攻击（ShadowCast、ImgTrojan、VL‑Trojan）在LLaVA‑1.5‑7B和Otter‑3B两大架构上构建的96个后门与96个对照模型；

**📈 对比分析**

与经典激活聚类（Activation Clustering）基线对比，EntropyScan在三种攻击场景下平均F1≈0.985、AUC≈0.966，显著优于基线（尤其是对ShadowCast的召回率从0提升至≈0.95），并在四台Tesla V100上每个模型平均耗时仅4.4秒；

**⚠️ 局限性**

局限包括对极其隐蔽的后门（如极低触发率或更高级的自适应攻击）可能仍难以捕捉；方法依赖于初始层注意力的可访问性，若模型架构不公开该层或采用不同的对齐机制可能需调整；

---

## 279. Differentiable Mixture-of-Agents Incentivizes Swarm Intelligence of Large Language Models

**arXiv ID:** 2605.15706 | [PDF](https://arxiv.org/pdf/2605.15706v1)

**作者:** Xingjian Wu `[一作]` (East China Normal University), Bin Yang `[通讯]` (East China Normal University)

**通讯引用:** 50396 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Differentiable Mixture-of-Agents（DMoA）框架，实现推理过程中自适应的多代理协作与通信拓扑演变。

**💡 创新点**

核心创新：1）可微、上下文感知的 RNN‑Router 实现逐步稀疏代理激活；2）使用预测熵作为自监督信号，支持无监督的测试时训练；3）通过稀疏激活与动态步骤扩展实现无空间时间边界的自演化 MAS。

**🔧 技术方法**

技术手段：基于 Sentence Transformer 的语义编码 + GRU 路由器；软最大化对比学习（pair‑wise ranking loss）对熵信号进行优化；多步推理、动态代理池与稀疏选择机制；工具调用与系统提示调度。

**📊 数据集**

评估数据集：9 个基准，包括 MMLU、GSM8K、MultiArith、SVAMP、AQuA、HumanEval、DS‑1000、HotpotQA、DDXPlus。

**📈 对比分析**

与单代理、固定拓扑 MAS（Chain、Tree、Star、Complete Graph、Random Graph 等）以及自演化 MAS（GPTSwarm、G‑Designer、ARG‑Designer、SafeSieve、AFlow、SpecReason、STEER）进行对比。DMoA 在所有 9 个基准上均取得领先，平均提升 5.75–26.97 分，尤其在难度更高的 DS‑1000 与 DDXPlus 上提升 25.94–26.97 分，显著优于现有自演化方法。

**⚠️ 局限性**

局限性：1）依赖大模型和多样化代理池，部署成本高；2）对预测熵的信号质量敏感，可能在极端任务中表现不稳定；3）实验主要集中在标准 benchmark，实际业务场景中的适配性与鲁棒性尚未充分验证。

---

## 280. AGOP-IxG: A Gradient Covariance Filter for Local Feature Attribution on Tabular Data, with a Controlled Benchmark

**arXiv ID:** 2605.15700 | [PDF](https://arxiv.org/pdf/2605.15700v1)

**作者:** Raj Kiran Gupta Katakam `[一作]` `[通讯]` (Credit Karma), Raj Kiran Gupta Katakam (Credit Karma)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种名为 AGOP‑IxG 的 per‑sample 解释方法，并在三种合成任务和两种真实数据上进行评估

**💡 创新点**

创新点在于将平均梯度外积矩阵（AGOP）截断为低秩投影，用作梯度预过滤器，既能显著提升解释精度，又保持极低的运行时开销

**🔧 技术方法**

技术包括：AGOP‑IxG 的梯度预过滤、梯度、积分梯度、SHAP DeepExplainer、LIME 等基线方法；评估指标为 Spearman 相关、top‑k 精度、噪声特征质量；对真实数据采用 ROAR 协议衡量全局特征重要性

**📊 数据集**

数据集：三类合成多分类任务（线性、稀疏非线性、交互）；两类公开真实数据集（Adult Income、Credit Card Default）

**📈 对比分析**

与 SHAP、Integrated Gradients、Input‑X‑Gradient、LIME 对比。AGOP‑IxG 在合成任务中 Spearman 相关最高、噪声特征质量最低，且速度比 SHAP 快 350–1650 倍；在真实数据的 ROAR AUC 中差异不大，所有方法聚集在约 0.39–0.40 的区间内

**⚠️ 局限性**

局限性：仅在单一 MLP 结构下验证，种子数有限；缺乏树模型或注意力模型的评估；真实数据缺少 ground‑truth 解释，ROAR 只衡量全局特征排名；合成任务结构化，可能不完全代表复杂真实分布

---

## 281. Rule2DRC: Benchmarking LLM Agents for DRC Script Synthesis with Execution-Guided Test Generation

**arXiv ID:** 2605.15669 | [PDF](https://arxiv.org/pdf/2605.15669v1)

**作者:** Jinuk Kim `[一作]` (Seoul National University), Hyun Oh Song `[通讯]` (Seoul National University)

**通讯引用:** 142814 | [OpenAlex ID](https://openalex.org/A5043109251)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了大规模执行基准Rule2DRC以及基于执行反馈的测试生成器SplitTester，用于将自然语言规则自动翻译为可执行的DRC脚本；

**💡 创新点**

创新点在于：①构建了含1000个规则-脚本对与13921个评估布局的公开基准，实现了以DRC执行结果为准的功能正确性评估；②提出SplitTester通过聚类候选脚本并迭代生成判别性布局测试，显著提升Best‑of‑N选择的成功率；

**🔧 技术方法**

技术上结合了大语言模型（Qwen3、GPT‑OSS）与KLayout DRC引擎，利用API文档上下文提示、执行反馈聚类、测试生成与最终LLM判断；

**📊 数据集**

使用的数据集包括从SkyWater130 PDK抽取的310条规则、690条人工合成多层/链式约束规则，共1000条任务，配合13921张GDSII评估布局；

**📈 对比分析**

与LLM‑as‑a‑Judge、生成测试、S*、CodeMonkey等基线在Rule2DRC上对比，SplitTester在N=20的Best‑of‑N设置下成功率提升至约63%（CodeMonkey约56%），且在成本-性能曲线中位于Pareto前沿；

**⚠️ 局限性**

局限性包括对LLM上下文长度的依赖、生成的期望标签噪声、仅针对DRC脚本生成、需在受限沙箱中执行、以及自动化过程可能忽视关键规则导致安全风险。

---

## 282. PRISM: Prompt Reliability via Iterative Simulation and Monitoring for Enterprise Conversational AI

**arXiv ID:** 2605.15665 | [PDF](https://arxiv.org/pdf/2605.15665v1)

**作者:** Keshava Chaitanya `[一作]` (Yellow.ai), Jahnavi Gundakaram `[通讯]` (Yellow.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

PRISM 是一个闭环框架，通过需求驱动的测试生成、平台忠实仿真、LLM‑as‑Judge 评判以及精准诊断修复循环，持续优化企业对话代理的前端提示并在生产环境中实时检测与修复 LLM 行为漂移。

**💡 创新点**

创新点在于将提示工程视为持续可靠性问题，首次实现了日常监控 24 小时内漂移检测与自动修复；采用需求驱动自动测试生成、平台忠实仿真以及基于 LLM 的诊断修复，形成完整的测试‑仿真‑评判‑修复闭环。

**🔧 技术方法**

使用 GPT‑4.1 进行模拟和评判、OpenAI Function Calling、Flask+SQLite 本地服务、SSE 实时进度推送、LLM‑as‑Judge 评判框架，以及平台忠实的 Yellow.ai V3 模拟环境。

**📊 数据集**

在 Yellow.ai V3 平台上评估 35 个企业对话代理，涵盖 12–147 条自动生成的测试用例；测试用例来源于人工编写的需求文档、工具、变量和子代理配置。

**📈 对比分析**

与传统人工提示编写基线对比，PRISM 将提示编写时间从平均 2.1 天降至 27 分钟，收敛迭代平均 7.1 次，生产可靠性达 99%（21 天内 735 次回归测试中 728 次无失败），漂移检测率 100%（7 次漂移在 24 小时内修复）。

**⚠️ 局限性**

限制包括仿真仅使用静态 mock，未覆盖真实工具失效或异常返回；测试用例覆盖受需求驱动，难以捕捉所有用户变异；大规模测试时 LLM 上下文窗口限制导致修复遗漏；判定依赖于评判 LLM 的能力。

---

## 283. Perforated Neural Networks for Keyword Spotting

**arXiv ID:** 2605.15647 | [PDF](https://arxiv.org/pdf/2605.15647v1)

**作者:** Vishy Gopal `[一作]` (Purdue University), Rorry Brenner `[通讯]` (Perforated AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在Edge Impulse平台上，对关键词识别模型应用了Perforated Backpropagation（PB），通过在每个神经元后添加人工树突节点，实现了更小参数量且更高精度的模型。

**💡 创新点**

创新点在于将树突计算作为可插拔插件引入现有深度网络，既不改动网络结构也不牺牲训练流程，却在所有参数预算下同时提升准确率和压缩模型大小，展示了树突网络在边缘AI上的通用优势。

**🔧 技术方法**

使用了Perforated Backpropagation、梯度下降树突与Cascade Correlation两种树突学习规则，以及PyTorch框架实现的Dendritic NN Impulse Block。

**📊 数据集**

实验采用Edge Impulse提供的小词汇量语音关键词数据集（MFCC特征），在800个超参数组合上进行训练与评估。

**📈 对比分析**

通过与传统（无树突）模型进行对比，树突模型在每个准确率阈值下参数量更少，在每个参数预算下准确率更高；最佳树突模型参数1,556且准确率0.933，较基线(3,859参数、0.921准确率)减低了60%参数和16%错误率，最优模型准确率0.958，参数11,421。

**⚠️ 局限性**

局限性包括：实验集中在单一关键词识别任务，尚未验证在更复杂语音或多词汇任务中的表现；树突插件的实现依赖PyTorch，跨平台或低功耗硬件的直接部署效果尚未评估；模型可扩展性和与硬件感知网络架构搜索的整合仍待深入研究。

---

## 284. ITHICA: Intra-Thread Instruction Checking Approach for Defect-Induced Silent Data Corruptions

**arXiv ID:** 2605.15638 | [PDF](https://arxiv.org/pdf/2605.15638v1)

**作者:** Ioanna Vavelidou `[一作]` (Stanford University), Caroline Trippel `[通讯]` (Stanford University)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5070082070)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `3855fcda-48ef-4070-a15e-803cd5c84d83`

**🎯 论文内容**

该论文提出了 ITHICA，一个能够从任意程序自动生成功能测试的框架，用于在大规模 CPU 服务器中检测缺陷 CPU。

**💡 创新点**

创新点在于打破传统功能测试假设，认为缺陷会产生不一致错误（同一指令在同一线程不同执行上下文下产生不同错误），从而利用指令级同线程检查，允许任意程序成为测试，显著提升缺陷发现率。

**🔧 技术方法**

核心技术包括在 LLVM IR 级别实现四种变换（算术、内存、分支、主动多样化），通过指令复制、输出比较、错误分支和插入 mfence/clflush 等指令实现检测；可配置的 block size 与 interleaving 控制检查频率；实现为可插拔的 LLVM Pass。

**📊 数据集**

使用了约 3000 台 CPU 服务器的实测数据，其中 DPool 包含 20 台已知缺陷服务器，QPool 包含 3,000 台怀疑缺陷服务器，涵盖至少 10 种不同微架构；此外对比了 SiliFuzz 和工业基线测试。

**📈 对比分析**

与工业基线测试和 SiliFuzz 对比，ITHICA 在同一程序下检测率提升 39% 以上（共发现 100 台缺陷服务器），错误检测次数 1.78 倍，检测速度平均提升 1.47 倍，覆盖率显著提高；同时展示了多变换、不同 block size/ interleaving 对检测效果的影响。

**⚠️ 局限性**

主要限制包括：仍无法在 ISA 层面精确定位硬件缺陷；缺乏对 atomic/volatile 指令、线程不安全内存、内核代码的覆盖；检测结果非确定性，需多次运行；对软错误不适用；以及对温度等硬件状态的控制不足。

---

## 285. The Shared Prosperity Internet

**arXiv ID:** 2605.15791 | [PDF](https://arxiv.org/pdf/2605.15791v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 286. VLMs Trace Without Tracking: Diagnosing Failures in Visual Path Following

**arXiv ID:** 2605.15672 | [PDF](https://arxiv.org/pdf/2605.15672v1)

**作者:** Hyesoo Hong `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**通讯引用:** 527 | [OpenAlex ID](https://openalex.org/A5049196468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究 VLM 在线条跟踪任务中的表现，设计受控实验评估模型在存在附近竞争路径时是否能保持追踪。

**💡 创新点**

提出受控的共享段/不同段/不同角度等条件，系统揭示 VLM 在局部竞争下的路径保持瓶颈，并通过行为、遮蔽和内部注意/表示分析确认原因；证明标准扩展手段无法解决。

**🔧 技术方法**

行为实验、遮蔽干预、内部注意力与表示余弦相似度分析、模型规模对比、推理增强、显式提示等技术。

**📊 数据集**

修改后的电路连线任务、抽象螺旋（Swirl）数据集、HANDLOOM 纤维缠绕数据、地铁线路图。

**📈 对比分析**

在两类受控任务中评估多款前沿 VLM（Claude Sonnet 4.5、Gemini3‑Flash、GPT‑5.4、Qwen3‑VL 等），发现准确率低至 0–44%（最差 6%），模型规模提升仅有限改善；推理模型虽略增准确但耗 token 高且依赖启发式；显式提示无效。

**⚠️ 局限性**

局部竞争导致路径跳转，缺乏专门机制保持选定路径；当前架构无法通过规模、推理或提示解决，需开发更稳健的视觉连续追踪机制。

---

## 287. From Gridworlds to Warehouses: Adapting Lightweight One-shot Multi-Agent Pathfinding for AGVs

**arXiv ID:** 2605.15799 | [PDF](https://arxiv.org/pdf/2605.15799v1)

**作者:** Hiroki Nagai `[一作]` (National Institute of Advanced Industrial Science and Technology), Keisuke Okumura `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 2689 | [OpenAlex ID](https://openalex.org/A5038362443)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对差分驱动 AGV 的多智能体仓库路径规划（MAWPF）问题，并在其上实现和评估多种轻量级 MAPF 算法。

**💡 创新点**

创新点包括：① 将经典 MAPF 扩展为 MAWPF，加入旋转、多步移动、加/减速及追随碰撞约束；② 设计多步 PIBT 以及滚动视野的 LaCAM+PIBT 方案；③ 引入停机路径定义和候选路径排序、剪枝优化。

**🔧 技术方法**

使用技术包括：优先规划 (PP)、大邻域搜索 (LNS2)、PIBT（改造为多步）、LaCAM（高层搜索+低层约束树）、A* 时空搜索、滚动视野、候选路径排序与剪枝。

**📊 数据集**

实验使用 MAPF 基准集中的 12 张地图（包含随机、仓库等），每张地图随机生成 25 个起点/终点（含方向、速度），总共测试了数千个实例。

**📈 对比分析**

通过成功率、运行时间、总成本三指标对比，LaCAM+PIBT 在大多数地图上实现了数百个 Agent 的高成功率与秒级运行；PP、LNS2 只能处理约 50–100 个 Agent；单独 PIBT 成功率低；在宽通道上表现好，窄通道则易失败；更大的预测步长 L 通常提升成功率并降低总成本。

**⚠️ 局限性**

局限性包括：① MAWPF 配置图有向导致死锁，限制到几百 Agent；② 计算配置生成成本高，影响可扩展性；③ 在狭窄通道上表现差；④ 仅评估一次性规划，未覆盖终身规划与更复杂动力学。

---

## 288. Grokking as Structural Inference: Transformers Need Bayesian Lottery Tickets

**arXiv ID:** 2605.15787 | [PDF](https://arxiv.org/pdf/2605.15787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 289. ForMaT: Dataset for Visually-Grounded Multilingual PDF Translation

**arXiv ID:** 2605.15794 | [PDF](https://arxiv.org/pdf/2605.15794v1)

**作者:** Michał Ciesiółka `[一作]` (Laniqo), Kamil Guttmann `[通讯]` (Laniqo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含3956份PDF、15语言对的格式保持多语言翻译语料库，并提供了基于视觉布局的翻译基准。

**💡 创新点**

通过45维几何特征和K‑Medoids聚类实现视觉多样性的采样，保留完整的排版与格式信息，使MT系统能够在文档级别和视觉层面进行评估。

**🔧 技术方法**

采用K‑Medoids聚类、视觉布局分析（LayoutLM/视觉模型）、pdfminer.six文本与元数据提取、以及视觉+文本融合的评估指标。

**📊 数据集**

ForMaT数据集，来源于欧盟、联合国、美国社会保障、汽车手册等法律与技术手册的双语PDF。

**📈 对比分析**

对Google Translate、DeepL和内部系统在表格、图像说明、列列表等视觉结构场景下进行手工评估，发现现有系统在空间定位、图像上下文和布局保真方面表现不足，内部系统在部分场景表现最佳。

**⚠️ 局限性**

仅覆盖拉丁字母、左到右语言，文档限制为10页，未包含右到左或非拉丁文字，且缺少自动化格式评价指标。

---

## 290. ALSO: Adversarial Online Strategy Optimization for Social Agents

**arXiv ID:** 2605.15768 | [PDF](https://arxiv.org/pdf/2605.15768v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 291. Learning Context-conditioned Gaussian Overbounds for Convolution-Based Uncertainty Propagation

**arXiv ID:** 2605.15789 | [PDF](https://arxiv.org/pdf/2605.15789v1)

**作者:** Ruirui Liu `[一作]` (Hong Kong Polytechnic University), Hui Ren `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 3107 | [OpenAlex ID](https://openalex.org/A5047813284)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f`

**🎯 论文内容**

本论文提出了一种基于神经网络的单阶段学习框架，直接输出与上下文相关的高斯上界（均值和方差），并在训练过程中通过量化回归、松弛的配对上界约束、Wasserstein 距离惩罚以及单调性正则化，保证学习到的上界在训练网格点上严格保守，并在满足一定的正则性假设下可推广到连续区间，从而实现安全可靠的误差叠加与卷积传播；

**💡 创新点**

创新点包括：①把传统的多步高斯上界过程整合到一条端到端的损失中，消除两步学习导致的冗余与额外保守；②在量化回归中加入松弛的配对上界约束，显式约束保守性；③引入 Wasserstein 距离惩罚，直接驱动上界与真实分布接近，提升紧凑性；④通过权重归一化与尺度因子 t 等技术，保证极端尾部的保守性；⑤给出从离散网格到连续区间的理论保证，阐明在有限网格上训练可得到在全区间保守的上界；

**🔧 技术方法**

技术手段包括：多量化回归（pinball loss），松弛的配对上界约束（paired overbounding with excess mass），Wasserstein 距离惩罚，单调性正则化，权重归一化与尺度因子 t 的自适应调节，神经网络预测 μ、σ 与 k 参数，随机种子集成与后验保守选择；

**📊 数据集**

数据集：1）合成三种高斯混合分布（多模态、负均值、正均值重尾）做离散网格实验；2）真实 GNSS 误差场景：①香港站 ZWD（高度误差）——tropospheric residual；②UrbanNav 车辆多路径误差；③香港 19 个 CORS 的电离层残差（TEC）——ionospheric residual；

**📈 对比分析**

比较方法：对照传统的两步 Gaussian 上界、配对上界、单量化回归上界（Quantile Overbounding）。评估指标包括保护水平 PL（单元与10倍自卷积）、Wasserstein 距离、平均上界因子 𝒦、以及覆盖性。实验结果显示，本方法在所有实验中都获得了更紧凑的上界（PL 范围缩小 10–85%）、更低的 Wasserstein 距离（降低 50–80%）、以及更小的 𝒦 值；且在训练网格点与经验评估中均保持严格保守；

**⚠️ 局限性**

局限性：①仅处理单变量高斯上界，难以直接扩展到多维/非高斯分布；②保守性保证依赖离散网格与正则性假设，极端尾部训练稳定性受限；③对超参数（λ、β、t、ε 等）敏感，需要经验调优；④缺乏对时间序列或多模态输入的显式建模；

---

## 292. GRASP: Learning to Ground Social Reasoning in Multi-Person Non-Verbal Interactions

**arXiv ID:** 2605.15764 | [PDF](https://arxiv.org/pdf/2605.15764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 293. BARRIER: Bounded Activation Regions for Robust Information Erasure

**arXiv ID:** 2605.15737 | [PDF](https://arxiv.org/pdf/2605.15737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 294. Continual Learning of Domain-Invariant Representations

**arXiv ID:** 2605.15775 | [PDF](https://arxiv.org/pdf/2605.15775v1)

**作者:** Pascal Janetzky `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出在持续学习（CL）框架下学习领域不变（domain-invariant）表示，以提升模型在部署后对未见目标域的泛化能力。

**💡 创新点**

创新点在于：1) 引入基于重放的多域不变性计算与序列化对齐机制，专为顺序训练设计；2) 在CL中首次系统评估部署中心的跨域泛化，并与传统CL基线和无监督适配器（CDA/CTTA）进行对比。

**🔧 技术方法**

技术方法包括：重放缓冲区、基于风险、梯度、特征均值方差、MMD与梯度符号一致性等五类域不变性损失；以及对齐损失以维持过去域的表示一致性。

**📊 数据集**

使用六个数据集进行实验：RotatedMNIST、CIFAR10C、TinyImageNetC、WM811K、Covertype、Camelyon17，涵盖图像、医学、制造与生态场景。

**📈 对比分析**

与13类主流CL方法（优化、正则、重放、潜在信息重放）以及三类CDA/CTTA方法比较，所提方法在所有数据集上平均准确率/宏F1均位居前列，提升约6–10个百分点，并在不同缓冲区大小、目标域切换与计算预算减半等鲁棒性测试中保持优势。

**⚠️ 局限性**

主要局限包括：1) 依赖重放缓冲区和域标签；2) 仅考虑可在单批次计算的域不变性度量，某些更复杂的因果不变性可能需要调整。

---

## 295. General-Purpose Co-Evolutionary Construction of Parallel Algorithm Portfolios for Multi-Objective Binary Optimization

**arXiv ID:** 2605.15729 | [PDF](https://arxiv.org/pdf/2605.15729v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 296. DecomPose: Disentangling Cross-Category Optimization Contention for Category-Level 6D Object Pose Estimation

**arXiv ID:** 2605.15728 | [PDF](https://arxiv.org/pdf/2605.15728v1)

**作者:** Yifan Gao `[一作]` (Wuhan Institute of Technology), Guoping Wang `[通讯]` (Peking University)

**通讯引用:** 16358 | [OpenAlex ID](https://openalex.org/A5100366298)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 DecomPose，针对多类别 6D 物体姿态估计中梯度冲突问题，采用难度感知分组与非对称分支实现对应模块解耦。

**💡 创新点**

创新点在于通过梯度诊断定位冲突源，并结合难度感知分组与不同容量的分支来缓解跨类别梯度冲突与负迁移。

**🔧 技术方法**

使用梯度相似性诊断、PointNet++ 与 DINOv2 共享特征提取、难度量化的量化分组、非对称分支容量分配以及统一姿态回归头。

**📊 数据集**

在 REAL275、CAMERA25 与 HouseCat6D 三个公开基准上进行实验。

**📈 对比分析**

与 CleanPose、AG-Pose、GCE-Pose 等最新方法对比，DecomPose 在 5°2cm/5°5cm/10°2cm/10°5cm 及 IoU_x 等 mAP 指标上均取得最优或第二名，显著提升姿态精度。

**⚠️ 局限性**

局限在于采用离线静态难度路由，类别数显著增大时可能失效，缺乏动态适应的路由机制。

---

## 297. Contexting as Recommendation: Evolutionary Collaborative Filtering for Context Engineering

**arXiv ID:** 2605.15721 | [PDF](https://arxiv.org/pdf/2605.15721v1)

**作者:** Jiachen Zhu `[一作]` (Shanghai Jiao Tong University), Jianghao Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 558 | [OpenAlex ID](https://openalex.org/A5036057873)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将 LLM 上下文工程视为个性化推荐问题的方法，并通过实例级路由动态为每个输入挑选最优上下文。

**💡 创新点**

核心创新在于将上下文策略当作“商品”引入协同过滤框架，构建 Context‑CF 共进化机制，使得推荐模型与 LLM 共同迭代扩展并细化上下文目录，从而实现个性化上下文生成。

**🔧 技术方法**

利用神经协同过滤（Neural Collaborative Filtering, NCF）进行实例-上下文偏好建模；采用聚类初始化、梯度引导的上下文生成和 LLM 反射修正；并用对比排名损失学习实例级偏好。

**📊 数据集**

在三大推理基准上进行评估：HoVer、SCONE 与 HotpotQA，使用 GPT‑4o‑mini 作为目标模型。

**📈 对比分析**

与 10+ 传统全局上下文优化基线（如 APE、OPRO、EvoPrompt、TextGrad、GEPA、MIPROv2 等）对比，NCCE 在三组数据集上均取得显著提升，平均测试准确率 74.8%（比最佳全局基线高约 6%）。

**⚠️ 局限性**

局限性包括：依赖大量 LLM 评估生成交互数据；需要在聚类数 K 与热启动优化质量之间寻找平衡；以及当前仅在固定 LLM（GPT‑4o‑mini）上验证，未展示跨模型迁移能力。

---

## 298. ADAPT: A Self-Calibrating Proactive Autoscaler for Container Orchestration

**arXiv ID:** 2605.15788 | [PDF](https://arxiv.org/pdf/2605.15788v1)

**作者:** Himanshu Singh Baghel `[一作]` `[通讯]` (J.C. Bose University of Science and Technology), Himanshu Singh Baghel (J.C. Bose University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了ADAPT——一种自校准的主动扩容方案，利用实时 EWMA 估计冷启动延迟，并将动态规划视窗 FH-OPT 输入模型预测控制器（MPC）来优化容器副本数；

**💡 创新点**

其创新点在于：通过实时测量并自适应调节扩容预期窗口，实现对不同环境与连续扩容事件变化的冷启动延迟的精准跟踪；

**🔧 技术方法**

技术包括：EWMA 估计、动态规划视窗 FH-OPT、模型预测控制器（MPC）结合 LSTM 或 Prophet 预测模型，以及传统 HPA 作为基线；

**📊 数据集**

实验数据采用了六种工作负载原型，并在每种负载下使用五个随机种子进行评估；

**📈 对比分析**

与 HPA（反应式）以及 MPC+Prophet（预测式）相比，MPC+LSTM 在所有负载上 SLA 违约率均低于 5%，远优于 HPA 的 7–19% 及 MPC+Prophet 在双峰流量上高达 28.7% 的违约率；

**⚠️ 局限性**

局限性包括：实验仅在模拟或实验环境中验证，未在大规模生产系统中实测；算法复杂度与部署成本较高，且对不同工作负载的泛化能力仍需进一步评估。

---

## 299. BioXArena: Benchmarking LLM Agents on Multi-Modal Biomedical Machine Learning Tasks

**arXiv ID:** 2605.15766 | [PDF](https://arxiv.org/pdf/2605.15766v1)

**作者:** Loka Li `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Le Song `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 15643 | [OpenAlex ID](https://openalex.org/A5102898115)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BioXArena 生物医学机器学习编码基准，包含 76 个多模态任务，评估 LLM 代理在 2 小时单 GPU sandbox 下自动编写、训练并提交模型。

**💡 创新点**

将多模态、跨域生物医学数据整合为统一任务胶囊，提供隐藏标签评估和统一排行榜；并对 11 种代理（通用 LLM、专用生物医学代理、ML 编码代理）进行公平比较。

**🔧 技术方法**

使用 LLM 代码生成与修复管道，搜索式代理（MLEvolve、STELLA、MLMaster‑2.0、Biomni），主干包括 Gemini‑3.1‑Pro、GPT‑5.4 等；训练算法涵盖提升树、森林、线性模型、神经网络。

**📊 数据集**

从 40+ 公开来源（如 PubMed、单细胞数据库、蛋白结构、分子数据库等）采集数据，总计 104 GB，覆盖 9 个生物医学领域。

**📈 对比分析**

在 11 种配置下评估，MLEvolve+Gemini‑3.1‑Pro 最高平均分 0.666，GPT‑5.4 次之 0.636；不同代理在各领域表现差异，未出现单一最优，证明强通用 LLM 与专用代理可相近但执行可靠性和模态处理仍是瓶颈。

**⚠️ 局限性**

任务覆盖不够全面，缺少空间组学、全切片病理、时间序列等；受 2 h 预算限制，无法覆盖大规模生产任务和分布式训练；评估可靠性和成本仍需改进。

---

## 300. Optimizing Line Segment Inspection with Limited-Range Drones

**arXiv ID:** 2605.15765 | [PDF](https://arxiv.org/pdf/2605.15765v1)

**作者:** José-Miguel Díaz-Báñez `[一作]` (University of Seville), Inmaculada Ventura `[通讯]` (University of Seville)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在一条直线上进行管线/太阳能集热器管道检查的多无人机飞行时间最小化问题，证明了即使仅两架无人机也为强NP‑难，并提出了基于单无人机最优解的贪心分配和改进版切分/扩展策略的多项近似算法。

**💡 创新点**

创新点在于：①首次证明两无人机/一线问题的强NP‑难性；②设计了简单高效的贪心分配算法并给出常数因子近似保证；③提出切分与扩展的改进策略实现近似比例显著下降；④通过实验验证算法在多种场景下几乎达到最优。

**🔧 技术方法**

技术包括：一维弧路由建模、动态规划求单无人机最优路径、贪心两分法、切分与扩展改进、混合整数线性规划（MILP）验证以及Gurobi求解器。

**📊 数据集**

使用的实验数据集为随机生成的线段集合，涵盖不同长度均匀性（CV）、密度、最大行程长度L等维度的场景，并以3段实例作为对比示例。

**📈 对比分析**

方法与性能比较：将贪心与改进算法的结果与MILP求得的最优解对比，计算近似因子Δ；实验显示贪心算法平均Δ≈1.04，改进算法平均Δ≈1.01，尤其在高均匀性与低L场景下最优比例达60%以上，显著优于理论保证。

**⚠️ 局限性**

局限性包括：问题为强NP‑难，无法得到多架无人机的精确解；MILP求解在高密度或大规模实例时耗时过长；当前算法仅适用于单线段集合，未覆盖多条并行线或多基地情形。

---

## 301. Learn2Splat: Extending the Horizon of Learned 3DGS Optimization

**arXiv ID:** 2605.15760 | [PDF](https://arxiv.org/pdf/2605.15760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 302. Hierarchical and Holistic Open-Vocabulary Functional 3D Scene Graphs for Indoor Spaces

**arXiv ID:** 2605.15753 | [PDF](https://arxiv.org/pdf/2605.15753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 303. UAM: A Dual-Stream Perspective on Forgetting in VLA Training

**arXiv ID:** 2605.15735 | [PDF](https://arxiv.org/pdf/2605.15735v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 304. BiomedAP: A Vision-Informed Dual-Anchor Framework with Gated Cross-Modal Fusion for Robust Medical Vision-Language Adaptation

**arXiv ID:** 2605.15736 | [PDF](https://arxiv.org/pdf/2605.15736v1)

**作者:** Huanyang Tong `[一作]` (Wenzhou University), Huiling Chen `[通讯]` (Wenzhou University)

**通讯引用:** 52085 | [OpenAlex ID](https://openalex.org/A5100648348)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了 BiomedAP，一种针对医学视觉‑语言模型的参数高效适配框架，旨在提升在少样本诊断任务中的鲁棒性与跨模态对齐。

**💡 创新点**

创新点主要包括：① 在中间层引入门控交叉模态融合（Gated Cross‑Modal Prompt Fusion），实现视觉与文本的动态交互，减少噪声提示的影响；② 采用双锚点约束（Dual‑Anchor Constraint），同时利用专家模板（高锚点）与视觉原型（低锚点）来正则化可学习提示，提升语义稳定性。

**🔧 技术方法**

技术方法包括：冻结 BiomedCLIP‑PubMedBERT 主干，只优化可学习提示与轻量化门控融合模块；使用多头交叉注意力实现双向信息交换；通过置信度自适应损失平衡知识蒸馏、锚点正则化与对齐损失；在训练时构造全局医学上下文，实现无标签推理时的融合。

**📊 数据集**

实验覆盖 11 个医学影像分类基准，涵盖 X‑ray、MRI、皮肤镜、视网膜等多种模态；使用 1–16 张样本的 few‑shot 评估以及基于 base‑to‑novel 的转移测试。

**📈 对比分析**

与现有基线（CLIP‑Adapter、Tip‑Adapter、CoOp、BiomedCoOp、Biomed‑DPT 等）比较，BiomedAP 在 all shot 设置下平均提升 4–5% 准确率，在 base‑to‑novel 情况下分别提升 3.5%、2.5% 与 3.0% 的平均准确率、新颖类别准确率与调和平均；对提示扰动的鲁棒性也提升了约 2–4% 的稳定增益。

**⚠️ 局限性**

局限性包括：仍需依赖预训练的 BiomedCLIP 作为主干，对 3D 体数据与密集预测任务尚未做扩展；门控融合与双锚点正则化参数需手工调优；在极少样本或极端提示缺失场景下性能提升有限。

---

## 305. GOMA: Toward Structure-Driven Multimodal Alignment from a Graph Signal Smoothing Perspective

**arXiv ID:** 2605.15723 | [PDF](https://arxiv.org/pdf/2605.15723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 306. Can We Trust AI-Inferred User States. A Psychometric Framework for Validating the Reliability of Users States Classification by LLMs in Operational Environments

**arXiv ID:** 2605.15734 | [PDF](https://arxiv.org/pdf/2605.15734v1)

**作者:** Izabella Krzeminska `[一作]`, Ewa Komkowska `[通讯]` (Orange Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究对大型语言模型在会话中推断用户状态的测量指标的可靠性进行系统评估，检验其单次预测与多次预测的稳定性以及跨模型一致性。

**💡 创新点**

提出了可复制的心理测量评估框架，量化了指标在实时与后期分析两种使用场景下的可靠性，并揭示了只有少量指标在不同模型间保持一致。

**🔧 技术方法**

使用ICC(3,1)与ICC(3,k)进行重复测量可靠性评估，并采用Cohen’s κ和归一化MAE进行跨模型一致性测量；数据处理通过DSPy框架完成。

**📊 数据集**

基于15个匿名呼叫中心对话（共552段，音频+人工转录）和三种多模态LLM（GPT‑4o audio、Gemini 2.0 Flash、Gemini 2.5 Flash）。

**📈 对比分析**

对比单模型内部的重复测量、跨模型的可靠性类别一致性以及满足高可靠性阈值的指标的值一致性；发现单测ICC≥0.9的指标仅占总数的14.6%，即大多数指标在实时使用中不稳定。

**⚠️ 局限性**

局限性包括样本量小、仅包含单一语种（波兰语）与单一行业（呼叫中心），未检验与真实标签的准确性，且只评估了三款LLM，缺乏对不同训练数据和版本的广泛覆盖。

---

## 307. Cut-Elimination for the Bimodal Logic GR

**arXiv ID:** 2605.15732 | [PDF](https://arxiv.org/pdf/2605.15732v1)

**作者:** Hirohiko Kushida `[一作]` `[通讯]`, Hirohiko Kushida

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套用于双模态逻辑 GR 的超序列（hypersequent）演算，并证明了该演算满足无割（cut‑elimination）定理。通过对 GR 的两种可证明性前件（哥德尔可证明性和罗斯泽可证明性）进行语义化，构造了对应的模态算子，并在超序列框架中给出了相应的推理规则。作者进一步证明了该演算与已有的 GR 规范化系统等价，并利用无割性质给出了 GR 对 GL 的保守性（conservativity）结果。

**💡 创新点**

创新点主要在于：
1) 将双模态逻辑 GR 的可证明性前件统一映射为超序列形式，首次给出了完整的无割演算。
2) 设计了“标准化”形式的模态引入规则，使得对割公式的对角公式（diagonal formula）可以被消除。
3) 引入 4 和 4^▪ 规则作为辅助，以实现无割证明的完整性，解决了此前缺乏可计算证明系统的问题。

**🔧 技术方法**

使用的技术包括：
- 超序列（hypersequent）框架，将模态算子与非模态算子分离。
- 证明转换（proof‑transformation）与规则置换（rule permutation）技术，用于归约至标准化形式。
- 直接割消除（top‑down cut‑elimination）方法，结合模态结构规则和内部结构规则。
- 归约对角公式（diagonal formula elimination）技术，借鉴了 GL 的割消除方法。

**📊 数据集**

本研究不涉及实验数据或数据集，所有结果均为形式化证明。

**📈 对比分析**

论文并未进行实验或性能对比；其主要贡献是证明无割定理和保守性，所得到的系统在理论上能够生成无割证明，且与 GR 的公理化系统等价。由于是形式化证明，性能评价在此背景下不适用。

**⚠️ 局限性**

限制与展望：
1) 证明过程相对复杂，尤其是对角公式消除步骤和 4 / 4^▪ 规则的引入。
2) 本文只关注 GR 的无割性质，未讨论自动化证明的实现与效率。
3) 对于更广义的双模态逻辑或其他可证明性前件，是否能直接采用同样方法仍需进一步研究。

---

## 308. Bidirectional Fusion Guided by Cardiac Patterns for Semi-Supervised ECG Segmentation

**arXiv ID:** 2605.15722 | [PDF](https://arxiv.org/pdf/2605.15722v1)

**作者:** Jeonghwa Lim `[一作]` (VUNO Inc.), Sunghoon Joo `[通讯]` (VUNO Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出了一种名为CardioMix的半监督心电图分割框架，利用心脏波形模式进行双向CutMix数据融合。

**💡 创新点**

创新点在于通过心脏模式引导的片段匹配保证合成信号的生理合理性，并在标签与未标签数据之间实现双向信息交换，从而提升伪标签质量和模型泛化能力。

**🔧 技术方法**

采用了基于ViT的编码器、双层FCN解码器、EMA教师模型、CutMix以及Confidence Gating等技术，构成一致性正则化的半监督学习流程。

**📊 数据集**

在四个公开心电图基准集LUDB、QTDB、ISP和浙江数据集上进行评估，使用10秒截断、250Hz重采样并做带通滤波和Z-score归一化。

**📈 对比分析**

与MT、FixMatch、CPS、ReCo、ST++等五种基线方法以及Vanilla CutMix、AugSeg、UPC等融合策略相比，CardioMix在mIoU和MAE上均取得显著提升，尤其在标签稀缺和跨域场景中表现最优。

**⚠️ 局限性**

局限性包括对伪标签质量的依赖、在高噪声或移动设备采集的ECG上验证不足，以及该方法目前仅针对P-QRS-T三相结构，需进一步验证在更广泛疾病或其他时序信号上的适用性。

---

## 309. Semi-MedRef: Semi-Supervised Medical Referring Image Segmentation with Cross-Modal Alignment

**arXiv ID:** 2605.15720 | [PDF](https://arxiv.org/pdf/2605.15720v1)

**作者:** Yuchen Li `[一作]` (University of Sydney), Luping Zhou `[通讯]` (University of Sydney)

**通讯引用:** 9251 | [OpenAlex ID](https://openalex.org/A5100643784)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Semi-MedRef，利用教师‑学生一致性学习的半监督医学参照图像分割框架，重点稳定图像与位置文本的跨模态对齐。

**💡 创新点**

创新点包括：① T‑PatchMix——基于位置约束和置信度的 CutMix 样式多模态混合；② PosAug——位置感知文本增强，采用随机 dropout 与 fuzzing；③ ITCL——基于位置伪标签的区域级图像‑文本对比学习。

**🔧 技术方法**

使用的技术：教师‑学生 EMA、一致性正则、Dice+CE 损失、ConvNeXt/CXR‑BERT 编码器、颜色抖动、GaussianBlur、温度 0.07 的对比损失，以及自定义的三种对齐保持机制。

**📊 数据集**

实验数据集：QaTa‑COV19 与 MosMedData+，两者均含有结构化的感染位置信息与像素级分割标签。

**📈 对比分析**

与自然图像 RES、医学半监督方法 LViT、Textmatch、以及监督基线 GuideDecoder/MMI‑UNet 等进行比较；Semi‑MedRef 在所有标签比例下均实现显著提升（Dice 最高提升至 91.4%/78.7%），且在 100% 标签下也保持领先。

**⚠️ 局限性**

局限性：依赖手工提取或数据集提供的粗粒度位置信息，难以直接扩展到更为自由或非结构化的文本描述；对极端变换下的跨模态对齐仍可能出现误差；缺乏更大规模公开数据的验证。

---

## 310. Fairness-Aware Retrieval Optimization for Retrieval-Augmented Generation

**arXiv ID:** 2605.15790 | [PDF](https://arxiv.org/pdf/2605.15790v1)

**作者:** Yingqi Zhao `[一作]` (Tampere University), Kostas Stefanidis `[通讯]` (Tampere University)

**通讯引用:** 2717 | [OpenAlex ID](https://openalex.org/A5042740371)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套针对 Retrieval-Augmented Generation（RAG）的三阶段公平检索框架，利用 reranker 注入可控偏差、构建位置感知的线性偏差传播模型，并在此基础上通过 FARO 优化检索排序以在保证相关性的同时降低生成结果的偏差。

**💡 创新点**

创新点包括：① 基于概率 reranker 的可控偏差注入机制；② 针对 top‑k 文档的位置信息建模的线性偏差传播方程；③ 通过 Fenchel–Legendre 双对偶形式将二次公平约束拆解为一系列独立子问题的 FARO（Quadratic Fairness via Dual Hyperplane Approximation）框架，实现公平‑相关性权衡的高效搜索。

**🔧 技术方法**

主要技术包括：RAG 与大语言模型（Llama 3.1、Gemma、Mistral、Qwen）集成；检索层使用 GTE embeddings + FAISS；reranker 采样控制偏差比例；线性回归估计位置权重；LP 与 FARO 两种求解方式来完成公平检索优化。

**📊 数据集**

使用的数据集为 TwinViews‑13k（政治立场对照）配合自动生成的中立问题集合，以及性别偏差数据集（基于 Wikipedia 个人简介 + 55 个职业的多选 QA），构成二元群体（政治立场/性别）公平评测。

**📈 对比分析**

实验中将提出的 FARO 与传统 RAG、无公平约束的 LLM、以及标准线性规划（LP）进行对比；在公平阈值 |R_b|≤0.1 时，FARO 与 LP 在相关性上相当，但在高 k 或大规模问题上计算更快；实验结果表明偏差显著降低，相关性损失低，展示了框架的有效性与实用性。

**⚠️ 局限性**

局限性包括：偏差传播模型假设线性且各文档相互独立，可能无法捕捉高阶交互；仅考虑二元群体公平，难以扩展到多群体或更细粒度的公平定义；以及缓解效果受到底层 LLM 原生偏差的限制。

---

## 311. Reactive Robot-Centric Safety for Autonomous Navigation in Constrained and Dynamic Environments

**arXiv ID:** 2605.15782 | [PDF](https://arxiv.org/pdf/2605.15782v1)

**作者:** Viswa Narayanan Sankaranarayanan `[一作]` (Lulea University of Technology), George Nikolakopoulos `[通讯]` (Lulea University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种实时安全关键控制架构，利用上机3D LIDAR感知和合成CBF（控制障碍函数）安全过滤器，在空间受限、动态环境中实现自主机器人导航，保证机器人在执行任务时的碰撞安全；

**💡 创新点**

创新点在于将机器人本体几何特性纳入安全约束，将安全区域定义为主体坐标系下的轴对齐椭圆，并对每个LIDAR点构造时间变CBF；通过对多约束使用soft‑min（log‑sum‑exp）合成CBF，将上千个点约束聚合为一个光滑约束，保证在控制频率下高效可解；

**🔧 技术方法**

技术手段包括：控制障碍函数（CBF）与时间变CBF，合成软最小化CBF，二次规划（QP）安全过滤器；硬件方面采用Boston Dynamics Spot四足机器人，搭载Orbbec Gemini2XL立体摄像机、Vectornav VN‑100 IMU、Ouster OS‑0 3D LIDAR，软件使用ROS、Voxblox 3D建图与CVXPY求解器；

**📊 数据集**

实验数据来自真实地下隧道环境的现场测试，包含狭窄走廊、动态障碍（移动椅子、打开门）以及定位漂移情景；未使用公开数据集，而是自定义的野外实验场景；

**📈 对比分析**

与传统基于点的球形距离约束进行对比，在狭窄通道中传统约束导致死锁，而椭圆本体约束能顺利通过；在定位漂移和动态障碍场景中，安全过滤器能即时修正控制指令，保持碰撞安全；QP求解时间平均<12µs，99.6%小于20µs，能够在控制频率下处理800–1300个约束，验证了实时性能；

**⚠️ 局限性**

局限性包括：仅考虑静态障碍的约束形式，虽然实时感知可补偿动态障碍但对极密集点云计算仍有挑战；安全椭圆参数需人工调节，对机器人几何的依赖较强；实验仅在单一四足平台验证，缺乏跨平台通用性验证；

---

## 312. SaaS-Bench: Can Computer-Use Agents Leverage Real-World SaaS to Solve Professional Workflows?

**arXiv ID:** 2605.15777 | [PDF](https://arxiv.org/pdf/2605.15777v1)

**作者:** Kean Shi `[一作]` (UniPat AI), Baobao Chang `[通讯]` (PKU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一个基于23个真实 SaaS 系统的长周期、多应用、跨域、可复现的基准，用来评估计算机使用代理（CUA）在实际业务流程中的表现。

**💡 创新点**

创新点在于：
1) 真实可部署的 SaaS 环境（前后端完整，状态动态，安全认证等）；
2) 通过“任务种子 → 生成 → 验证”四阶段流水线构建106个跨应用、跨模态、长达100+步的任务；
3) 采用加权校验检查点（State‑Check、Content‑Check、LLM‑Judge）实现精确的部分与完整完成度评估；
4) 为评估提供统一的浏览器交互框架（browser‑use）与 Docker 重置机制。

**🔧 技术方法**

技术方法包括：
- LLM Builder（Claude Opus）与人类挑选/细化的 Builder–Challenger–Refiner 迭代流程；
- 浏览器自动化框架（browser‑use）实现纯 UI 交互；
- Docker 容器化所有 SaaS，保证可复现与状态隔离；
- 多种检查点类型与加权评分体系；
- Pass@k、Checkpoint Score、Resolved Score 等评价指标。

**📊 数据集**

使用的数据集是：23 个开源 SaaS（覆盖软件工程、商业运营、医疗、团队协作、农业供应链、媒体创作六大领域）以及 106 个人工审核通过的任务脚本，任务数据通过 LLM 生成并结合公开数据填充，保证业务场景的真实性。

**📈 对比分析**

比较方法：对 9 种主流 LLM‑based 代理（Claude Opus 4.6、GPT‑5.4 High、Qwen 3.6 Plus 等）在统一环境、同一任务集上使用相同提示与执行预算进行评测；通过 Resolved Score、Checkpoint Score 与 Pass@k 统计。性能表现：最佳模型 Claude Opus 4.6 的 Checkpoint Score 仅 43%，而 Resolved Score 低至 3.8%；其他模型均低于 40% 的 Checkpoint Score，且大部分模型在长链任务上出现明显性能衰减。

**⚠️ 局限性**

局限性：
- 代理在长周期、跨应用的依赖链中易出现“错误级联”与“状态丢失”，导致最终完成率极低；
- 评估仅关注 UI 交互，缺乏对后端调用与数据模型的深度理解；
- 低成功率暴露出规划、状态追踪、错误恢复等核心能力不足；
- 评测中存在显著的单跑波动，单一 Pass@1 指标可能误导；
- 任务生成依赖 LLM 与人工审核，可能存在主观偏差与覆盖不足。

---

## 313. CompactQE: Interpretable Translation Quality Estimation via Small Open-Weight LLMs

**arXiv ID:** 2605.15763 | [PDF](https://arxiv.org/pdf/2605.15763v1)

**作者:** Kamil Guttmann `[一作]` (Laniqo), Krzysztof Jassem `[通讯]` (Laniqo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种利用小型开源LLM（<30B参数）在单次推理中同时生成质量分数、MQM风格错误注解、纠正建议及完整后编辑文本的可解释性质量估计方法。

**💡 创新点**

创新点在于将质量估计与错误检测、后编辑整合为一次推理，消除了对大型专有模型的依赖，实现成本低、数据隐私友好。

**🔧 技术方法**

技术采用基于GEMBA-MQM V2的结构化提示策略，结合自研过滤管道，使用Gemma-3-27b-it、EuroLLM-9B-Instruct和Qwen3-VL-30B-A3B-Instruct等开源模型。

**📊 数据集**

实验使用WMT25 Metrics Shared Task提供的三语言对（捷克-德语、英-意、英-乌克兰）段落级翻译数据。

**📈 对比分析**

与COMET、MetricX、GEMBA等基线及Gemini-2.5-Pro对比，系统级Soft Pairwise Accuracy最高达0.83/0.82，虽低于Gemini-2.5-Pro的0.87，但显著优于传统回归指标，且在段级关联性仍有提升空间。

**⚠️ 局限性**

局限性包括未进行任务特定微调、仅评估段落级上下文无法捕获跨句错误、span-level F1低、缺乏对后编辑质量的评估以及对专业术语、文化语境等外部信息的忽视。

---

## 314. A Unified Perturbation Framework for Analyzing Leaderboard Stability and Manipulation

**arXiv ID:** 2605.15761 | [PDF](https://arxiv.org/pdf/2605.15761v1)

**作者:** Hosna Oyarhoseini `[一作]` (University of Waterloo), Amir-Hossein Karimi `[通讯]` (University of Waterloo)

**通讯引用:** 690 | [OpenAlex ID](https://openalex.org/A5101905145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个统一的基于影响函数的扰动框架，用于评估并操纵基于 Bradley–Terry 模型的评估排行榜的稳定性。

**💡 创新点**

创新点在于将影响函数与排行榜目标（Top‑k 成员、置信区间、全局一致性）统一关联，并同时分析匹配删除、添加和翻转三类干预，同时可用于主动数据采样与模型推广/降级。

**🔧 技术方法**

主要技术包括 Bradley–Terry 评分估计、Hessian 逆矩阵的影响函数近似、一阶梯度传播、温度平滑的 Kendall‑τ 损失以及局部 Fisher 信息估计的置信区间代理。

**📊 数据集**

实验使用了七个二人对比数据集：Chatbot Arena（Arena‑55k、Arena‑LLM‑J）、MT‑Bench、Vision Arena、WebDev Arena、NBA Top‑50、ATP Top‑10，以及自定义的 Arena‑Active、Random 等基线。

**📈 对比分析**

通过影响函数引导的扰动，作者在 5% 的预算下即可改变 Top‑1、CI‑aware Top‑k 甚至全局排序，且在主动降不确定性任务中相较于随机或 Arena‑Active 方案减少 10–30% 的采样量，展示出明显的性能提升。

**⚠️ 局限性**

局限性包括：影响函数为局部近似，可能在大规模扰动或高度非线性变化下失准；置信区间代理仅基于 BT 的 Fisher 信息，未考虑注释噪声、判断偏差、提示依赖等；并未给出样本复杂度或鲁棒性理论保证。

---

## 315. DimMem: Dimensional Structuring for Efficient Long-Term Agent Memory

**arXiv ID:** 2605.15759 | [PDF](https://arxiv.org/pdf/2605.15759v1)

**作者:** Wentao Qiu `[一作]` (StepOS), Yu Zhang `[通讯]` (StepOS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级的维度化记忆框架 DimMem，能够在 LLM 代理中以结构化的、原子化的记忆单元存储、检索、更新和动态召回长时记忆。

**💡 创新点**

创新点在于：①将记忆单元拆分为可自包含的“类型 + 时间、地点、原因、目的、关键词”五个维度；②三路检索（词法、密集、维度）以及根据查询意图动态决定是否附加助手回复；③利用该维度结构训练小型提取器，实现高质量记忆构造；④通过维度过滤显著降低更新与检索的计算开销。

**🔧 技术方法**

技术包括：基于窗口的重叠式对话分割与压缩、使用 LoRA 微调的 4B 结构化提取器、BM25+Dense+维度匹配三路检索、LLM（如 GPT‑4.1‑mini、Qwen3‑4B）进行记忆合并与更新、以及动态助手召回机制。

**📊 数据集**

使用 LoCoMo‑10 与 LongMemEval‑S 两大长时记忆基准，数据来源为真实多轮对话与合成的 10k/5k 训练样本。

**📈 对比分析**

与 MemoryOS、MemOS、LightMem、SimpleMem 等现有轻量化记忆系统对比，DimMem 在 LoCoMo‑10 上整体准确率提升至 81.43%（比 LightMem 提升约 8.64%），在 LongMemEval‑S 上整体准确率提升至 78.20%（比 LightMem 提升约 8.6%）。同时，单查询 token 成本降低 24% 以上，维度过滤使更新时嵌入计算减少 75% 以上。

**⚠️ 局限性**

局限性包括：仅在文本对话场景验证，缺乏多模态与隐私保护场景测试；依赖高质量的提取与意图解析，若提取错误会影响整体性能；维度构造的可扩展性需根据不同领域手工调整；更新合并仍需 LLM 判定，可能引入不确定性。

---

## 316. Separating Acute Psychological Stress from Physical Exertion in Biometric Signals

**arXiv ID:** 2605.15756 | [PDF](https://arxiv.org/pdf/2605.15756v1)

**作者:** Esther Bosch `[一作]` `[通讯]` (German Aerospace Center), Esther Bosch (German Aerospace Center)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文在实验室内设计了2×3的重复测量实验，比较了五种生理信号在认知压力与身体活动下的响应。

**💡 创新点**

创新点在于系统验证了紧张态与运动的加性效应，并确立了托尼皮肤电活动(Tonic EDA)为在运动状态下仍能可靠检测心理压力的唯一传感器。

**🔧 技术方法**

采用了多级线性混合模型、重复测量方差分析以及信号预处理（滤波、去噪、特征提取）等技术。

**📊 数据集**

使用了19名受试者的实验数据，包含心率、心率变异性、皮肤电、肌电、呼吸率等五类传感器的连续记录。

**📈 对比分析**

通过多级模型与方差分析对比，发现Tonic EDA在压力与运动主效应均显著，且无交互作用，而其余四种传感器主要受运动影响，表现出较差的压力识别性能。

**⚠️ 局限性**

局限包括样本量小、仅采用n-back社会评价式压力诱导、实验为实验室环境、运动强度未标准化，导致结果可能不具备普遍性。

---

## 317. The Privacy Subsidy: Kyle's $λ$ under Noise-Perturbed Order-Flow Observation

**arXiv ID:** 2605.15746 | [PDF](https://arxiv.org/pdf/2605.15746v1)

**作者:** Yuki Nakamura `[一作]` `[通讯]` (Open University of Japan), Yuki Nakamura (Open University of Japan)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

推导了在加密货币市场中加入隐私噪声的Kyle线性均衡，揭示了隐私对价格影响系数和交易者策略的效应，并给出了隐私补贴的闭式公式；

**💡 创新点**

首次将Bayesian但非零利润市场制造商与隐私噪声相结合，证明了隐私补贴的存在并与Loss‑Versus‑Rebalancing（LVR）相类比；

**🔧 技术方法**

利用线性Kyle模型、贝叶斯推断、高斯随机过程和期望利润分解；

**📊 数据集**

无；

**📈 对比分析**

无实证比较，本文为理论分析，未进行实验或性能评估；

**⚠️ 局限性**

局限在于只考虑单周期线性模型、独立高斯隐私噪声、线性策略，且未覆盖批量交换、封闭式竞价等其他隐私设计。

---

## 318. Preserving Topology Privacy of Network Systems by Feedback: Conditions and Distributed Design

**arXiv ID:** 2605.15743 | [PDF](https://arxiv.org/pdf/2605.15743v1)

**作者:** Yushan Li `[一作]` (KTH Royal Institute of Technology), Dimos V. Dimarogonas `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 19033 | [OpenAlex ID](https://openalex.org/A5055348953)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于反馈的控制方法，在保持一致性收敛的前提下，刻意破坏网络拓扑可辨识性，从而实现拓扑隐私保护。

**💡 创新点**

创新点在于利用反馈设计直接破坏观测系统的可观测性或使可辨识拓扑产生偏差，提供了完全可行的中心化与分布式两种实现方案，并引入可调的隐私预算与状态偏差折中机制，显著提升了隐私保护效果。

**🔧 技术方法**

主要技术包括：拓扑可辨识性与不变子空间理论的解析；对齐不变子空间的反馈构造；基于拉普拉斯矩阵的参数化反馈；分布式最大信标算法构造根节点与树结构；线性可行性分析及低复杂度启发式优化。

**📊 数据集**

实验使用了随机生成的8节点有向网络权重矩阵以及人工设定的初始状态，并在该网络上进行多次仿真验证。

**📈 对比分析**

与集中式设计、拉普拉斯反馈以及噪声注入（M3/M4）方法相比，本文方法在保持一致性误差在可接受范围内的同时，能够实现更高的拓扑识别误差（Er1、Er2稳定不降为零），并且不需要长期随机噪声。

**⚠️ 局限性**

主要局限包括：对中心化反馈矩阵求解仍需全局信息，分布式算法在保持强连通性方面可能需要额外的根节点选择；隐私保护程度仍依赖于预算参数与网络结构；未考虑观测者对反馈设计的事先认知或主动攻击。

---

## 319. AOT-POT: Adaptive Operator Transformation for Large-Scale PDE Pre-training

**arXiv ID:** 2605.15793 | [PDF](https://arxiv.org/pdf/2605.15793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 320. HyperDiT: Hyper-Connected Transformers for High-Fidelity Pixel-Space Diffusion

**arXiv ID:** 2605.15741 | [PDF](https://arxiv.org/pdf/2605.15741v1)

**作者:** Yu He `[一作]`, Yan Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HyperDiT 框架，通过多尺度交叉注意力和寄存器引导，解决像素空间扩散中的粒度困境，实现高分辨率图像的高保真生成。

**💡 创新点**

创新点在于：1）Hyper Connectors 在细粒度分支中使用跨尺度注意力动态查询多级语义锚点；2）Scale‑Aware RoPE 对齐不同尺度位置；3）寄存器作为无位置的密集语义指引，并通过 VFM 对齐的 REPA 进行监督；4）整体消除 VAE 的重建瓶颈。

**🔧 技术方法**

采用像素空间扩散、Flow Matching、跨尺度注意力、Scale‑Aware RoPE、寄存器令牌、VFM（如 DINOv2）对齐、CFG 指导等技术。

**📊 数据集**

ImageNet 256×256 数据集。

**📈 对比分析**

与 Latent‑space（DiT‑XL/2、SiT‑XL/2 等）和 Pixel‑space（JiT‑G/16、DiP‑XL/16、DeCo‑XL/16 等）基线对比，HyperDiT‑H 在 FID 上取得 1.56、HyperDiT‑XL 1.63，显著优于现有方法，且精度、召回、IS 也均领先。

**⚠️ 局限性**

局限性包括：1）对显存与计算量敏感，需要较大 GPU；2）依赖预训练 VFM 对齐，若 VFM 表达不足可能影响性能；3）模型参数规模相对较大，部署成本较高。

---

## 321. Real-Time Reconstruction and Actuation Error Analysis for Markov Sources over MPR Channels

**arXiv ID:** 2605.15795 | [PDF](https://arxiv.org/pdf/2605.15795v1)

**作者:** Pansee S. Elessawy `[一作]` (Linköping University), Nikolaos Pappas `[通讯]` (Linköping University)

**通讯引用:** 3988 | [OpenAlex ID](https://openalex.org/A5084740578)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了两个二值马尔可夫源在多包接收(MPR)无线通道上的实时重建与执行误差，并给出了稳态重建误差(RTE)与执行误差成本(CAE)的闭式表达；

**💡 创新点**

创新点在于将MPR接收模型与目标导向指标联系起来，推导出RTE与CAE相互等价的条件，并在源动态为非正相关时证明了在采样约束下最优策略可取于极点；

**🔧 技术方法**

主要技术包括马尔可夫链分析、有效更新概率的解析推导、对目标函数的凸凹性分析以及极点搜索求解最优随机采样策略；

**📊 数据集**

本文未使用公开数据集，而是通过系统参数（源转移概率、MPR成功概率、语义权重等）进行理论推导与数值仿真；

**📈 对比分析**

实验与四种基线（随机、贪心源1、贪心源2、TDMA）对比，显示最优随机采样策略在不同采样预算和语义权重下均能显著降低重建误差；

**⚠️ 局限性**

局限性包括仅考虑两源二值马尔可夫模型、仅适用于静态随机采样策略以及在源动态为正相关时最优解可能不再为极点，需进一步研究更一般场景与自适应策略。

---

## 322. Reversing the Flow: Generation-to-Understanding Synergy in Large Multimodal Models

**arXiv ID:** 2605.15792 | [PDF](https://arxiv.org/pdf/2605.15792v1)

**作者:** Yujun Tong `[一作]` (Beijing University of Posts and Telecommunications), Zhanyu Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 8189 | [OpenAlex ID](https://openalex.org/A5039812471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Generation-to-Understanding (G→U) 机制，使大型多模态模型通过内部生成“视觉思考”来增强自身感知与推理。

**💡 创新点**

创新点在于将生成视为内部推理步骤，打破传统 U→G 单向管道，实现零样本、无额外训练的闭环反馈，系统评估了生成质量对理解提升的影响。

**🔧 技术方法**

使用集成 Transformer 的 BAGEL（7B）模型，利用其扩散生成与自回归推理双向共享参数，构建两阶段 Prompt 体系（生成+感知），并引入 GPT‑4o‑mini 作为自适应编辑 Prompt 编写器。

**📊 数据集**

主要数据集为自构造的 VisThink‑Bench（1595 例）和 12 个公开 VQA/推理/鲁棒性基准（MMBench、MME、MM‑Vet、MMStar、KiVA、HallusionBench、R‑Bench 等）。

**📈 对比分析**

与 BAGEL 基线及多种统一/专用模型对比，G→U 在 VisThink‑Bench 上平均提升约 10%（取决于任务），在其他基准上提升 1–4%，显著提升了推理与鲁棒性表现。

**⚠️ 局限性**

局限性包括：生成的可信度限制了可提升范围；对高层抽象或因果推断的编辑提示效果差；模型缺乏自我指向的想象能力，导致部分任务仍无法获得帮助。

---

## 323. Beyond Controlled Noise: Achieving Symmetric FHE through Dynamic Position Shifting

**arXiv ID:** 2605.15774 | [PDF](https://arxiv.org/pdf/2605.15774v1)

**作者:** Mostefa Kara `[一作]` `[通讯]` (University of El Oued), Mostefa Kara (University of El Oued)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了一种利用文本碎片化与动态位置移位的对称完全同态加密（FHE）方案，能够在不使用昂贵的引导或重线化的情况下实现精确的加法与乘法运算。

**💡 创新点**

创新点在于引入“指数调节器”和“系数调节器”双重绑定机制，将明文碎片在不同位置间循环移动，从而抑制乘法时密钥指数的指数级增长，彻底解决了传统同态加密中噪声失控的问题。

**🔧 技术方法**

采用的技术包括：对称加密基础c=mk的文本碎片化、动态位置移位调节、指数调节器（t_i）与系数调节器（d_i）的双重绑定、基于整数分解与离散对数难题的安全证明以及对称密钥与调节器的公私钥构造。

**📊 数据集**

论文未使用真实数据集，仅在性能评估中对比了公开参数下的BFV、BGV、TFHE、CKKS等基准方案，并以其默认参数进行实验。

**📈 对比分析**

通过与上述四种主流FHE方案在密钥生成、加密、解密、加法、乘法的时间及密文大小的对比，本文方案在加密速度（≈0.02 ms）和密文大小（≈9 KB）上优于对手，同时实现了无噪声积累的完全同态运算。

**⚠️ 局限性**

局限性包括：①方案为对称设计，安全性依赖整数分解和离散对数难题，无法在后量子安全模型下使用；②对明文乘法深度有限制，需保证∏m^(j)<p；③缺乏针对复杂机器学习或大规模数据的实证评估。

---

## 324. Lamarckian Inheritance in Dynamic Environments: How Key Variables Affect Evolutionary Dynamics

**arXiv ID:** 2605.15769 | [PDF](https://arxiv.org/pdf/2605.15769v1)

**作者:** K. Ege de Bruin `[一作]` (University of Oslo), Kai Olav Ellefsen `[通讯]` (University of Oslo)

**通讯引用:** 608 | [OpenAlex ID](https://openalex.org/A5071893221)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在不同动态环境下，Lamarckian 与 Darwinian 继承机制对软机器人控制参数学习的影响，并通过实验验证其效能。

**💡 创新点**

创新点在于将环境冲突度与可预测性作为关键变量，系统性探讨它们对继承机制选择的影响，并展示传感器辅助预测可恢复 Lamarckian 优势。

**🔧 技术方法**

采用 EvoGym 虚拟软机器人平台，结合贝叶斯优化（BO）与深度确定性策略梯度（DDPG）强化学习，并实现 Lamarckian 与 Darwinian 继承。

**📊 数据集**

实验使用自定义生成的随机崎岖地形和双向方向切换环境，构成动态、可预测和不可预测的三种场景。

**📈 对比分析**

通过在 100 代、20 次实验中比较四种继承+学习组合，结果表明在非冲突或可预测环境下 Lamarckian 能显著优于 Darwinian，而在冲突且不可预测环境下 Darwinian 更佳；加入方向传感器后，Lamarckian 再度占优。

**⚠️ 局限性**

局限性包括仅关注软机器人身体与控制耦合，未涉及形态适应；实验环境有限，缺乏更复杂的冲突变化；以及对传感器类型的探索不足。

---

## 325. Attribute-Grounded Selective Reasoning for Artwork Emotion Understanding with Multimodal Large Language Models

**arXiv ID:** 2605.15755 | [PDF](https://arxiv.org/pdf/2605.15755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 326. Cross-Modal Registration Between 3D and 2D Fingerprints via Pose-Aware Unwrapping and Point-Cloud Fusion

**arXiv ID:** 2605.15796 | [PDF](https://arxiv.org/pdf/2605.15796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 327. An Enriched Model of Strategic Voting under Uncertainty

**arXiv ID:** 2605.15786 | [PDF](https://arxiv.org/pdf/2605.15786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 328. A Topology-Aware Spatiotemporal Handover Framework for Continuous Multi-UAV Tracking

**arXiv ID:** 2605.15779 | [PDF](https://arxiv.org/pdf/2605.15779v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 329. The Robotaxi Placement Problem: Minimizing Expected ETA for Stochastic Demand

**arXiv ID:** 2605.15745 | [PDF](https://arxiv.org/pdf/2605.15745v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 330. Structure Abstraction and Generalization in a Hippocampal-Entorhinal Inspired World Model

**arXiv ID:** 2605.15733 | [PDF](https://arxiv.org/pdf/2605.15733v1)

**作者:** Tianqiu Zhang `[一作]` (Peking University), Si Wu `[通讯]` (Peking University)

**通讯引用:** 10619 | [OpenAlex ID](https://openalex.org/A5027798299)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种脑启发的分层世界模型，利用海马-内嗅皮层耦合与逆向模型自监督学习，从原始视频中同时抽取抽象结构和内容表示，并实现跨场景的结构迁移。

**💡 创新点**

创新点在于：①将海马-内嗅皮层的内容-结构分离原则引入深度模型；②使用连续吸引神经网络（CANN）实现网格细胞式路径积分以抽象运动；③通过逆向模型从MEC嵌入中提取低维抽象过渡，从而实现无监督的结构抽象与泛化。

**🔧 技术方法**

使用技术包括：预训练多尺度 VQ‑VAE 编码、时序 Transformer 编码器、CANN 运动动力学、逆向 MLP 进行抽象过渡推断、自监督对齐与正则化损失、视觉反馈机制。

**📊 数据集**

数据集涵盖：训练集为 Something‑Something v2（SSv2）；评估集包括 COIL‑100、MIRO、OmniObject3D、Franka Kitchen、Block Pushing、Push‑T、LIBERO Goal 等离散与 OOD 场景。

**📈 对比分析**

与 LAPA、Moto、AdaWorld LAM 等基线在 SSIM、LPIPS 指标下对比，模型在单步与自回归生成中取得更低 LPIPS、更高 SSIM，尤其在 OOD 结构复用任务中表现稳健；Ablation 结果进一步证明 CANN 与层级结构对性能的关键作用。

**⚠️ 局限性**

局限性包括：①自回归预测易受累积误差影响；②对与训练分布差异大的虚拟环境表现下降；③多实体协同/对象交互仍难以处理。

---

## 331. Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR

**arXiv ID:** 2605.15726 | [PDF](https://arxiv.org/pdf/2605.15726v1)

**作者:** Chanuk Lee `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 NudgeRL 框架，通过策略级上下文提示实现结构化且多样化的探索，提升大型语言模型的推理能力。

**💡 创新点**

创新点在于“Strategy Nudging”对每条采样路径施加轻量级上下文，使模型探索更丰富的推理路径，并通过 Inter‑Intra 组优势和蒸馏目标将发现的行为迁移回基准策略。

**🔧 技术方法**

使用了强化学习与可验证奖励（RLVR）、GRPO 优化、上下文条件采样、交叉熵蒸馏、干扰/熵正则等技术。

**📊 数据集**

在多项数学竞赛 benchmark（如 30‑problem olympiad、40‑problem contest、MATH 134 subset 和 48‑problem advanced contest）上进行评估。

**📈 对比分析**

与基线（无优化、不同 Rollout 预算的 GRPO、以及 oracle prefix 的 POPE）比较，NudgeRL 在仅 8 次 Rollout 的情况下即可超越 32/64 次 Rollout 的 GRPO，并在所有测试集上平均提升 pass@1，表现优于 oracle‑guided 方法。

**⚠️ 局限性**

主要限制是上下文提示的生成需离线完成，且固定上下文池在训练后期可能变得不再具有探索性，未来可考虑模型自适应的上下文生成。

---

## 332. DiLA: Disentangled Latent Action World Models

**arXiv ID:** 2605.15725 | [PDF](https://arxiv.org/pdf/2605.15725v1)

**作者:** Tianqiu Zhang `[一作]` (Peking University), Si Wu `[通讯]` (Peking University)

**通讯引用:** 10619 | [OpenAlex ID](https://openalex.org/A5027798299)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种Disentangled Latent Action (DiLA) 世界模型，联合学习结构与内容的分离以及抽象动作表示。

**💡 创新点**

通过将预测瓶颈与内容-结构分离共同进化，解决 LAM 的抽象-生成质量权衡。

**🔧 技术方法**

使用 DINOv2 特征编码、ST-Transformer、Mamba 内容记忆、AdaLN-FDM 前向动力学、旋转位置编码以及双路径融合解码器，并在隐空间进行自监督训练。

**📊 数据集**

在 SSv2、RT-1、RECON、LoopNav、OmniObject3D 等大规模视频数据集上训练和评估。

**📈 对比分析**

与 LAPA、Moto、AdaWorld、villa‑X 等基线相比，在 SSIM/LPIPS、跨模态动作迁移、重绑定、视觉规划等指标上表现更优，生成质量更高、动作可迁移性更强。

**⚠️ 局限性**

局限在于抽象动作缺乏细粒度控制、未实现多物体动态解耦、以及自回归长序列的误差累积。

---

## 333. Linked Multi-Model Data on Russian Domestic and Foreign Policy Speeches

**arXiv ID:** 2605.15886 | [PDF](https://arxiv.org/pdf/2605.15886v1)

**作者:** Daria Blinova `[一作]` (University of Delaware), Benjamin E. Bagozzi `[通讯]` (University of Delaware)

**通讯引用:** 1056 | [OpenAlex ID](https://openalex.org/A5021724291)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了四大俄语与英语双语、跨模态（文本+图像）俄罗斯政府演讲语料库，包含克里姆林宫与俄罗斯外交部的官方演讲记录，并对文本、图像、元数据进行清洗、翻译、主题标注和位置/坐标化，形成易于使用的 CSV 与图片文件结构。

**💡 创新点**

① 通过两阶段网页抓取实现可复制的跨语言语料收集；② 在同一篇演讲的俄语与英语版本之间提供精确的多模态链接；③ 采用基于 transformer 的 BERTopic 与 CLIP 结合的无监督主题建模框架，为文本与图像分别生成可解释的主题标签；④ 结合机器翻译与手工校对实现高质量的俄语→英语文本对齐。

**🔧 技术方法**

主要技术包括：Python webscraping（Requests, BeautifulSoup 等）、Argos Translate（俄语→英语翻译）、spaCy/LLM（语言模型）、Sentence‑Transformer（文本嵌入）、CLIP ViT‑B/32（图像嵌入）、BERTopic（文本主题建模）、自定义哈希去重、Nominatim/ArcGIS 地理编码、PostgreSQL/SQLite 结构化存储，以及 Docker/Colab 复现环境。

**📊 数据集**

数据集来自官方克里姆林宫网站（1999‑2025 年）与俄罗斯外交部网站（2004‑2025 年），共 31,666 篇演讲（俄语 19,396 篇，英语 12,270 篇）以及 89,158 张相关图像；同时提供 15,610 篇英语、19,396 篇俄语文本、与其对应的元数据与主题标签。

**📈 对比分析**

对比方法：在同一机构内使用相同主题数量（克里姆林宫 89 主题，外交部 32 主题），通过 BERTopic 在英文空间完成文本主题模型，随后利用 CLIP 生成图像主题分布；使用人类专家对每个主题进行标签与分组，验证与官方标签的一致性。性能方面，主题模型在多模态验证中达成 80‑90% 的一致率，图像主题分布与文本主题高度相关；数据完整性检验显示 100% 的图片数量与官方计数一致。

**⚠️ 局限性**

局限性：① 翻译质量依赖 Argos Translate，可能在细微语义与专业术语上产生误差；② 俄语直接主题建模未达标，导致俄语主题仅以翻译文本为基础；③ 对地点的自动抽取与地理编码依赖外部服务，部分地点仍无法解析；④ 主题标签由人工定性，受限于标签者主观；⑤ 数据集中缺乏视频或音频等多模态信息，限制了更丰富的跨媒体分析。

---

## 334. Distributed Affine Body Dynamics with Adaptive Consensus

**arXiv ID:** 2605.15875 | [PDF](https://arxiv.org/pdf/2605.15875v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 335. SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval

**arXiv ID:** 2605.15868 | [PDF](https://arxiv.org/pdf/2605.15868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 336. Access Timing as Scaffolding: A Reinforcement Learning Approach to GenAI in Education

**arXiv ID:** 2605.15850 | [PDF](https://arxiv.org/pdf/2605.15850v1)

**作者:** Janne Rotter `[一作]` (Pompeu Fabra University), Davinia Hernández-Leo `[通讯]` (Pompeu Fabra University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一个基于强化学习的智能教学系统，该系统动态决定何时允许学生使用生成式人工智能，以提升文本理解学习效果和元认知准确性。

**💡 创新点**

将教育学理论（元认知、认知负荷、产能失败）嵌入奖励函数，利用RL实现对GenAI访问时机的自适应控制，从而实现“少干预、少支架”的新型教学策略。

**🔧 技术方法**

采用Proximal Policy Optimization (PPO) 强化学习，结合贝叶斯知识追踪 (BKT) 模拟学生，使用Mistral 3 LLM 作为AI助手。

**📊 数据集**

实验在109名大学生中进行，使用自制文本阅读与多选题任务（社交媒体对自我形象影响），并记录AI请求日志。

**📈 对比分析**

与“始终允许”和“始终不允许”两种基线做双样本t检验及ANCOVA，结果显示RL条件在后测得分和元认知准确性上显著优于“始终允许”，并与“始终不允许”无显著差异；学习效率略高。

**⚠️ 局限性**

受限于小规模学生模型、单一学科与文本任务、短期实验且自我评估工具未充分验证，且主要以英语为实验语言，可能影响外推性。

---

## 337. Structured Jacobian Construction for Motion Optimization with High-Order Time Derivatives in Multi-Link Systems

**arXiv ID:** 2605.15845 | [PDF](https://arxiv.org/pdf/2605.15845v1)

**作者:** Taiki Ishigaki `[一作]` (Tokyo University of Science), Eiichi Yoshida `[通讯]` (Tokyo University of Science)

**通讯引用:** 6990 | [OpenAlex ID](https://openalex.org/A5080826821)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种针对多连杆系统的结构化雅可比矩阵计算框架，系统地推导了涉及更高阶时间导数（如加速度、力的变化）的运动学与动力学量的解析雅可比。

**💡 创新点**

创新点在于利用多连杆结构显式建模，从而得到解析雅可比表达式，避免传统数值或自动微分方法的计算冗余和数值不稳定性，并可同时应用于正向和逆向运动优化。

**🔧 技术方法**

主要技术包括：基于综合运动计算框架的结构化雅可比构造；解析推导动量、力、关节扭矩等物理量的高阶雅可比；以及对正向与逆向优化问题的统一处理。

**📊 数据集**

实验使用了数值仿真生成的多连杆运动数据（未公开具体数据集），通过合成轨迹验证方法效果。

**📈 对比分析**

与数值差分和自动微分方法对比实验表明，所提框架在计算速度上显著优于传统方法，精度保持一致；在逆向优化中成功从运动数据中恢复成本函数权重，证明了其实用性。

**⚠️ 局限性**

局限性包括：需要手动推导解析雅可比，可能在极大规模或复杂关节拓扑下实现困难；实验范围有限，缺乏真实机器人数据验证；以及对非线性约束和离散事件的处理尚未深入。

---

## 338. Heuristic-Based Merging of HPC Traces to Extend Hardware Counter Coverage

**arXiv ID:** 2605.15832 | [PDF](https://arxiv.org/pdf/2605.15832v1)

**作者:** Júlia Orteu Aubach `[一作]` (Barcelona Supercomputing Center), Marta Garcia-Gasulla `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 835 | [OpenAlex ID](https://openalex.org/A5082933794)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种基于启发式的多次执行合并方法，将多份只记录部分硬件计数器的MPI应用程序执行轨迹拼接成一条包含所有计数器的合成轨迹，以扩大硬件计数器覆盖面，进而丰富用于机器学习性能预测的特征空间。

**💡 创新点**

创新点在于首次将计算突发（compute bursts）在不同执行间通过MPI通信结构、时间与通信模式匹配，得到高精度的突发对应关系，并在此基础上实现完整的硬件计数器拼接，而非传统的多路复用或单次收集。

**🔧 技术方法**

技术手段包括：① Extrae/Paraver 的执行轨迹采集与Burst层次化处理；② 两阶段启发式匹配算法（直接匹配 + 结构约束匹配）；③ 基于加权相似度（时间、通信大小、通信伙伴）的跨执行突发匹配；④ 合成轨迹的列合并策略（保留相同计数器、前缀区分不同执行）。

**📊 数据集**

实验使用巴塞罗那超级计算机 MareNostrum5 上的5种代表性HPC应用（Stream、Alya、Lulesh、SOD2D、SeisSol），并采用三组互补的PAPI计数器集合（INS_MIX、OPS_SET、OPS_CYC）进行多次执行采样。

**📈 对比分析**

通过对相同计数器集的多次执行验证，指标包括Pearson相关系数、相对差异、MAE以及满足<30%差异的比例，结果显示确定性应用（Stream、Alya、Lulesh）可实现100%匹配；非确定性应用在两阶段匹配后可达99.9%+匹配率；指令与浮点计数器的匹配精度极高，缓存计数器则表现出显著的测量波动。

**⚠️ 局限性**

主要局限在于缓存层计数器的时序敏感性导致高相对差异，限制了其在预测模型中的可靠性；此外，方法目前仅针对MPI-only CPU工作负载，需进一步扩展到混合编程与异构硬件环境。

---

## 339. More efficient PBWT prefix-array access via batching

**arXiv ID:** 2605.15819 | [PDF](https://arxiv.org/pdf/2605.15819v1)

**作者:** Travis Gagie `[一作]` (Dalhousie University), Travis Gagie `[通讯]` (Dalhousie University)

**通讯引用:** 2810 | [OpenAlex ID](https://openalex.org/A5013172801)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种基于批处理的 PBWT 前缀数组访问方法，能够在查询一组单倍型时，快速定位 SMEM（Set‑Maximal Exact Match）并报告对应的单倍型 ID。

**💡 创新点**

创新点在于：①提出将查询分批处理，并在满足特定条件（已发现 r·h/r 个 SMEM 子串且每个子串平均匹配至少 r/h 个单倍型）时，仅使用 O(r log h) 位存储 PBWT 运行信息；②利用双向链表在构造前缀数组时一次性输出所有 SMEM 匹配 ID，实现在 O(K) 时间内报告所有 K 个匹配 ID。

**🔧 技术方法**

主要技术包括：PBWT 运行长度压缩、前缀数组构造、双向链表维护、子串信息批处理与排序，以及对查询子串区间的直接遍历输出。

**📊 数据集**

论文未使用具体实验数据集，而是在理论分析中假设典型单倍型面板规模：高度 h 为数百至数万，长度 ℓ 为数百万至数千万，整体大小 n = h·ℓ 为数亿至数百亿，运行总数 r 为数千万至数十亿。

**📈 对比分析**

方法比较主要通过理论复杂度分析完成。相比 Bonizzoni 等人提出的 O((r+r') log h + h log n) 位空间、O(k loglog h) 或 O(loglog min(h,ℓ)+k) 时间的方案，批处理方法在满足批处理条件时进一步降低到 O(r log h) 位空间和常数时间（O(1)）输出每个单倍型 ID，且总时间为查询子串长度加上输出匹配 ID 的数目 K。

**⚠️ 局限性**

局限性包括：①需要将查询批量化，单次查询无法直接使用；②要求 SMEM 子串数量及匹配数至少为 Ω(r)；③在实际部署时需要额外构造前缀数组和维护双向链表，增加了预处理开销；④对非常稀疏或非常短的单倍型面板效果可能不明显。

---

## 340. StippleDiffusion: Capacity-Constrained Stippling using Controlled Diffusion

**arXiv ID:** 2605.15816 | [PDF](https://arxiv.org/pdf/2605.15816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 341. Martingale Neural Operators: Learning Stochastic Marginals via Doob-Meyer Factorization

**arXiv ID:** 2605.15806 | [PDF](https://arxiv.org/pdf/2605.15806v1)

**作者:** Kai Hidajat `[一作]` `[通讯]` (University of Washington), Kai Hidajat (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出了一种基于Doob-Meyer分解的马尔可夫神经算子（MNO），能够一次性预测随机偏微分方程终端分布的均值和协方差。

**💡 创新点**

创新点在于将半鞅的漂移-马尔可夫分解作为网络结构先验，构造低秩协方差因子，既保留正定性又保持通道感知，兼顾速度与不确定性量化。

**🔧 技术方法**

采用傅里叶神经算子（FNO）头来预测漂移和低秩因子，并用高斯残差实现零均值协方差，整体模型实现一次前向传播即可得到均值、方差和样本。

**📊 数据集**

使用1D SPDE（Burgers、ϕ⁴）、粗糙波动（Rough Heston/Bergomi）、以及2D流动与Gray-Scott等公开数据集进行评估。

**📈 对比分析**

与传统神经SPDE、Neural SDE、FNO以及条件扩散模型对比，MNO在1D任务上Wasserstein-2距离提升至120×、68×，在粗糙波动下提升2.6×，在生成效率上比条件扩散模型快约3倍。

**⚠️ 局限性**

局限性包括仅提供终端边际而非完整路径分布、残差假设为高斯导致无法表达重尾或多模态，以及在高度非线性或守恒律强的2D任务（如Gray-Scott）表现不佳。

---

## 342. Security Analysis of a Communication Protocol: MQTT

**arXiv ID:** 2605.15804 | [PDF](https://arxiv.org/pdf/2605.15804v1)

**作者:** Ricardo Venâncio `[一作]` (Polytechnic of Porto), Luís Ribeiro `[通讯]` (Polytechnic of Porto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对MQTT协议在IoT环境中的安全性进行理论评估与实验验证，演示了窃听、篡改、拒绝服务与暴力破解等攻击，并提出相应缓解措施。

**💡 创新点**

采用混合方法论（文献回顾+模拟实验），在Docker化智能家居环境中实现真实攻击演示，并系统阐述了TLS、ACL及轻量级加密等多层防御策略。

**🔧 技术方法**

使用Eclipse Mosquitto MQTT broker、Node‑RED、Python脚本、ARP‑spoofing、MQTT Stresser、TLS、PRIDE/LEA等技术栈进行实验与分析。

**📊 数据集**

实验数据基于自建的智能家居传感器模拟（温度、门磁等），未使用公开大规模数据集。

**📈 对比分析**

通过对比正常与DoS攻击下的消息延迟，攻击导致平均延迟从≈6–8 ms激增至≈60 s，显著验证了可用性劣化。

**⚠️ 局限性**

缓解措施仅在理论层面提出，未在实验环境中实现或验证其实际效果，缺乏实证评估。

---

## 343. Complexity of Non-Log-Concave Sampling in Fisher Information

**arXiv ID:** 2605.15859 | [PDF](https://arxiv.org/pdf/2605.15859v1)

**作者:** Sinho Chewi `[一作]` (Yale University), Andre Wibisono `[通讯]` (Yale University)

**通讯引用:** 956 | [OpenAlex ID](https://openalex.org/A5043389973)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究在非对数凸分布下，用相对 Fisher 信息保证进行采样的查询复杂度，提出基于近端采样器的算法并给出最优 O(1/ε²) 复杂度（含维度因子），同时建立与高精度对数凸采样的互相归约关系。

**💡 创新点**

1) 在仅假设 L-光滑的条件下，证明近端采样器在平均意义上收敛相对 Fisher 信息；2) 通过构造近端采样器的“平滑”实现，将实现 RGO 的难题转化为高精度对数凸采样问题；3) 给出逆归约，证明若能改善非对数凸采样的维度依赖，则高精度对数凸采样也能相应改进。

**🔧 技术方法**

利用 Langevin 动力学的梯度流视角、de Bruijn 识别、Gaussian 通道与 RGO 的组合、近端采样器的 Gibbs 迭代、Rényi 散度与 Fisher 信息的关系、χ² 散度及其平滑化技术、以及对数凸采样器的高精度实现。

**📊 数据集**

无（该工作为理论分析，未使用具体数据集）。

**📈 对比分析**

与传统 ULA、MALA 等采样方法相比，所提出的近端采样器在相对 Fisher 信息目标下实现了近似最优的 O(1/ε²) 复杂度，并在维度依赖上与最先进的对数凸采样器相匹配，说明其性能优越。

**⚠️ 局限性**

1) 维度依赖仍与对数凸采样器的复杂度挂钩，若该复杂度无法进一步降低，则本工作也无法改进；2) 仍需在低维场景下探讨是否存在更优算法；3) 证明中的逆归约假设使用了 3 阶 Rényi 散度，需要进一步验证其在实际实现中的可行性。

---

## 344. RoadmapBench: Evaluating Long-Horizon Agentic Software Development Across Version Upgrades

**arXiv ID:** 2605.15846 | [PDF](https://arxiv.org/pdf/2605.15846v1)

**作者:** Xinbo Xu `[一作]` (UniPat AI), Baobao Chang `[通讯]` (Peking University)

**通讯引用:** 6035 | [OpenAlex ID](https://openalex.org/A5021459300)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了RoadmapBench基准，聚焦115个真实开源项目的长周期多目标版本升级任务；

**💡 创新点**

创新点在于将真实升级转化为多子任务路线图、引入静态验证与rollout质量控制、并用加权子任务奖励衡量连续进展；

**🔧 技术方法**

采用OpenHands框架在Docker环境下执行多工具（Explore、Edit、Execute、Plan、Think）交互，评估13款前沿LLM编码代理；

**📊 数据集**

使用从17个跨5种语言（Python、JavaScript、Go、Rust、Java）开源仓库中抽取的115个任务，未与现有基准重复；

**📈 对比分析**

通过Resolved Rate（完整完成率）与Completion Score（加权子任务通过率）对比模型性能，最强Claude‑Opus‑4.7仅达39.1%完整率，整体难度显著高于传统bug修复基准；

**⚠️ 局限性**

局限在于任务仍难度过高、覆盖仅限17个仓库、评估模型仅为13款且可能受框架/工具差异影响，缺乏更广泛的实际工程验证。

---

## 345. Modeling Music as a Time-Frequency Image: A 2D Tokenizer for Music Generation

**arXiv ID:** 2605.15831 | [PDF](https://arxiv.org/pdf/2605.15831v1)

**作者:** Yuqing Cheng `[一作]` (Central Conservatory of Music), Xiaotao Gu `[通讯]` (Zhipu AI)

**通讯引用:** 1595 | [OpenAlex ID](https://openalex.org/A5101398927)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种面向生成的二维 Mel 频谱分词器 BandTok，替代残差码本分词方式，兼顾高重建质量和自回归模型友好性。

**💡 创新点**

创新点在于将音频分词视为二维图像，将 Mel 频带作为垂直轴，使用单一共享码本和 2D RoPE 位置编码，减少残差层耦合并提升自回归可预测性。

**🔧 技术方法**

使用的技术包括 2D Haar 采样、Cosmos 编码器、单码本 VQ、EMA 更新、PatchGAN 多尺度鉴别器、2D Rotary Position Embedding、预训练 BigVGAN vocoder、T5 文本编码、CFG 引导等。

**📊 数据集**

数据集涵盖 FMA、Freesound、MTG‑Jamendo、MUSDB、SongDescriber 等音频与文本数据，用于分词器训练、语言模型训练及评测。

**📈 对比分析**

与残差码本分词器（EnCodec、DAC、MelCap 等）以及 Stable Audio Open 等基线相比，BandTok 在 Mel 与 STFT 重建误差上更低，生成质量在 FAD_CLAP、CLAP、AudioBox 指标上均优于对手，并在低比特率场景表现突出。

**⚠️ 局限性**

局限性包括对文本提示的依赖性仍有限、对长段音乐的时间条件匹配不佳、模型规模对数据依赖强、以及对高频细节的捕捉仍有提升空间。

---

## 346. FashionChameleon: Towards Real-Time and Interactive Human-Garment Video Customization

**arXiv ID:** 2605.15824 | [PDF](https://arxiv.org/pdf/2605.15824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 347. Diversified Residual Symbolic Regression

**arXiv ID:** 2605.15809 | [PDF](https://arxiv.org/pdf/2605.15809v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 348. Unlocking Dense Metric Depth Estimation in VLMs

**arXiv ID:** 2605.15876 | [PDF](https://arxiv.org/pdf/2605.15876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 349. Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design

**arXiv ID:** 2605.15871 | [PDF](https://arxiv.org/pdf/2605.15871v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 350. Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination

**arXiv ID:** 2605.15864 | [PDF](https://arxiv.org/pdf/2605.15864v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 351. Embedding-perturbed Exploration Preference Optimization for Flow Models

**arXiv ID:** 2605.15803 | [PDF](https://arxiv.org/pdf/2605.15803v1)

**作者:** Sujie Hu `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 9857 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在文本嵌入空间注入结构化扰动，提升基于RL的流模型对人类偏好的对齐，解决组内方差衰减导致的优化停滞与奖励劫持问题。

**💡 创新点**

①在组内引入嵌入级扰动维持判别信号；②噪声感知采样调度与参考锚批次策略实现多样化探索并防止语义漂移；③训练时将扰动与原始提示对齐，提升稳定性。

**🔧 技术方法**

强化学习（GRPO）结合流模型与DiffusionNFT；嵌入扰动机制、噪声感知采样调度、参考锚批次；使用文本编码器、DINOv3特征、t‑SNE可视化等技术。

**📊 数据集**

Stable Diffusion 3.5 Medium 作为基础模型；评估使用GenEval和PickScore奖励模型，指标涵盖IDS、ASC、SDI、PVS、Aesthetic、ImgRwd、HPSv2.1、DivGenBench等。

**📈 对比分析**

与GRPO、DiffusionNFT等基线在不同组大小G与扰动数K的配置下对比，实验显示在高探索与高效生成场景下均获得最高奖励与多样性，超过基线约1–2%，同时保持更优视觉质量与多样性。

**⚠️ 局限性**

对不同文本编码器的鲁棒性有限；高计算预算下的G与K平衡仍需进一步细化；需在更广泛的多模态任务中验证通用性。

---

## 352. From Layers to Networks: Comparing Neural Representations via Diffusion Geometry

**arXiv ID:** 2605.15901 | [PDF](https://arxiv.org/pdf/2605.15901v1)

**作者:** Atharva Khandait `[一作]` (Chalmers University of Technology), Jan E. Gerken `[通讯]` (Chalmers University of Technology)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5086301780)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将表示相似性测量与扩散几何、视角学习相结合，提出多尺度和交替扩散版本的CKA与DistCorr；

**💡 创新点**

创新点在于证明RSM类相似度可用行随机矩阵重写，从而借助扩散几何操控多尺度和多层视角；

**🔧 技术方法**

使用Markov矩阵、扩散幂、交替扩散(AD)以及改写后的CKA/DistCorr；

**📊 数据集**

在ReSi基准（14种网络，7个数据集）和GRS基准（语言Transformer）上进行评测；

**📈 对比分析**

与24个基线比较，新的多层/多尺度相似度在语言Transformer和部分视觉模型上均达到或超过SOTA，尤其在ReSi的预测、输出相关性和GRS的OOD准确率上表现突出；

**⚠️ 局限性**

局限在于多层聚合受矩阵乘积退化为秩1的影响，对图模型表现不一，且需要更高阶AD或改进方法来缓解。

---

## 353. FedEDAuth -- Federated Embedding Distribution Authentication for Counterfeit IC Detection

**arXiv ID:** 2605.15885 | [PDF](https://arxiv.org/pdf/2605.15885v1)

**作者:** Naseeruddin Lodge `[一作]` (University of North Carolina at Charlotte), Fareena Saqib `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 807 | [OpenAlex ID](https://openalex.org/A5061092086)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为FedEDAuth的嵌入分布鉴权框架，用于在联邦学习聚合前过滤恶意客户端，从而实现防伪集成电路的协同检测。

**💡 创新点**

创新点在于将黄金参考数据的嵌入分布作为鉴权基准，利用离群率、均值偏移和微聚类三种统计指标进行隐私友好的嵌入级鉴权，并在聚合前主动拦截攻击者，避免梯度污染。

**🔧 技术方法**

采用了联邦学习（FedAvg、FedTrim、Krum）与Vision Transformer作为特征提取器、ResNet‑18作为下游分类器，结合Mahalanobis距离、k‑means聚类及自定义怀疑分数进行客户端鉴权。

**📊 数据集**

实验使用了约6,387张IC图像的数据集，在3,387张正品图像上加入多种细微划痕等伪造特征，构成3,000张伪造图像，形成完整的验证集。

**📈 对比分析**

与传统聚合方法在Byzantine数据中毒攻击下对比，FedEDAuth能够100%检测到被中毒的5/50客户端（0%假阳性），并在过滤后实现94.17%的分类准确率、95.11%的召回率、94.17%的F1以及0.983的AUC‑ROC，显著恢复至干净模型水平。

**⚠️ 局限性**

局限性包括：在数据异质或非IID情况下可能导致假阳性；仅在划痕型中毒攻击下验证；对自适应或标签翻转、模型中毒等更复杂攻击的鲁棒性未知；需要可信第三方鉴权服务器且阈值选择依赖手工调参。

---

## 354. FSCM: Frequency-Enhanced Spatial-Spectral Coupled Mamba for Infrared Hyperspectral Image Colorization

**arXiv ID:** 2605.15880 | [PDF](https://arxiv.org/pdf/2605.15880v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 355. Uncertainty-Aware Wildfire Smoke Density Classification from Satellite Imagery via CBAM-Augmented EfficientNet with Evidential Deep Learning

**arXiv ID:** 2605.15894 | [PDF](https://arxiv.org/pdf/2605.15894v1)

**作者:** Ranjith Chodavarapu `[一作]` (Kent State University), Ranjith Chodavarapu `[通讯]` (Kent State University)

**通讯引用:** 31 | [OpenAlex ID](https://openalex.org/A5092238013)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在卫星图像中对野火烟雾严重程度进行三分类（轻度、中度、重度），并提供置信度与不确定度估计。

**💡 创新点**

首次将证据深度学习与CBAM空间注意力相结合，单前向传播即可得到分解的epistemic vacuity与aleatoric dissonance。

**🔧 技术方法**

使用预训练EfficientNet‑B3骨干+CBAM+Dirichlet输出层，并辅以AOD回归头和自定义损失（EDL+KL+AOD）。

**📊 数据集**

基于Kaggle Wildfire Detection卫星补丁数据（16,298张），通过合成AOD代理生成轻、中、重三个标签。

**📈 对比分析**

与重现的SmokeNet与交叉熵基线对比，模型在测试集上实现93.8%加权准确率（91.1%无权重）、ECE0.0274，且在保留最高置信度50%的样本时准确率提升至96.7%。

**⚠️ 局限性**

主要局限在于伪AOD标签的循环性、云污染导致的“无不确定度”错误、严重的类别不平衡以及缺乏外部真值验证。

---

## 356. Do Less, Achieve More: Do We Need Every-Step Optimization for RL Fine-tuning of Diffusion Models?

**arXiv ID:** 2605.15855 | [PDF](https://arxiv.org/pdf/2605.15855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 357. Evaluating Container Orchestration for Neuromorphic Workloads in Virtual Edge Environments

**arXiv ID:** 2605.15866 | [PDF](https://arxiv.org/pdf/2605.15866v1)

**作者:** Huyen Pham `[一作]`, Bilhanan Silverajan `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了在虚拟边缘环境中使用K3d Kubernetes容器编排对脉冲神经网络(SNN)工作负载的部署、性能与资源限制的影响。

**💡 创新点**

首次系统测量SNN在容器化边缘环境下延迟、吞吐量与资源约束的关系，并揭示默认负载均衡策略对长周期推理任务导致尾部延迟的严重影响。

**🔧 技术方法**

使用BindsNET模拟的脉冲神经网络、FastAPI推理服务、Docker容器、K3d/K3s Kubernetes以及WSL2+Docker Desktop在Windows 11上构建单节点虚拟边缘集群。

**📊 数据集**

使用MNIST手写数字数据集进行模型训练和推理评估。

**📈 对比分析**

通过比较三种资源限制配置和三种并发负载场景的p50、p95、p99延迟、吞吐量和准确率，发现资源限制导致延迟提升近50倍、吞吐量下降近50倍，而准确率保持稳定；默认轮询负载均衡在复制扩展时产生严重尾部延迟。

**⚠️ 局限性**

实验仅在单节点虚拟环境下进行，未验证在真实边缘硬件上的表现，且缺乏针对长周期推理的高级负载均衡和异步处理策略。

---

## 358. GAP: Geometric Anchor Pre-training for Data-Efficient Visuomotor Learning of Manipulation Tasks

**arXiv ID:** 2605.15836 | [PDF](https://arxiv.org/pdf/2605.15836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 359. Not All Tasks Quantize Equally: Fisher-Guided Quantization for Visual Geometry Transformer

**arXiv ID:** 2605.15828 | [PDF](https://arxiv.org/pdf/2605.15828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 360. Caesar: A Deductive Verifier for Probabilistic Programs

**arXiv ID:** 2605.15827 | [PDF](https://arxiv.org/pdf/2605.15827v1)

**作者:** Philipp Schröer `[一作]`, Christoph Matheja `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出并实现了Caesar，一种针对概率程序的演绎验证器；

**💡 创新点**

创新点在于引入定量中间验证语言HeyVL，支持上下界期望值推理、灵活的证明规则、量化断言与假设，以及量化抽象的语义近似；

**🔧 技术方法**

核心技术包括：HeyVL的量化语义与断言语言、VCG生成、量化前期望语义、量化量化推理、量化量化量化的量化量化、量化量化的量化量化量化、Z3 SMT求解器、JANI/Storm模型检查后端、量化量化量化的量化量化诊断与语义护栏、VSCode LSP扩展；

**📊 数据集**

使用了多组公开案例程序（如带噪声的BRP、Rabin互斥协议、随机游走、Ben‑Or一致性协议、Irwin‑Hall分布等），但并未构建统一基准库；

**📈 对比分析**

与现有工具（Boogie、Why3、Storm等）对比的实验显示：在多数测试案例中验证速度几乎即时，Slicing诊断开销可控；但整体性能高度依赖程序结构与证明规则，缺乏统一的基准比较；

**⚠️ 局限性**

局限包括：对所有概率程序不完备；连续分布处理仍有限；需要手工编写证明规则与不变量；缺乏标准基准集；未实现自动不变量合成与大规模项目集成。

---

## 361. A Multi-Layer Cloud-IDS Pipeline with LLM and Adaptive Q-Learning Calibration

**arXiv ID:** 2605.15889 | [PDF](https://arxiv.org/pdf/2605.15889v1)

**作者:** Syed Waqas Ali `[一作]` (University of Engineering and Technology), Hans D. Schotten `[通讯]` (RPTU University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对云计算环境的多层入侵检测系统（网络、主机、虚拟化层），并通过强化学习动态调节每层的置信度阈值，以实现更精准、成本更低的检测与 LLM 升级机制。

**💡 创新点**

创新点在于：1）使用 Q‑learning 对每个 IDS 层自适应学习置信度阈值，摆脱静态阈值带来的过度或不足升级问题；2）设计三门门控流程（Gate‑1、Gate‑2、Gate‑3）结合内存匹配、LLM 语义分析与加权融合决策；3）在检测决策中明确记录低置信度事件，支持后续分析与模型再训练。

**🔧 技术方法**

技术包括 XGBoost 基础分类器、Q‑learning 置信阈值调优、Chroma 向量记忆匹配、LLM（Ollama）语义推理、加权融合决策以及基于 Python 的实现框架。

**📊 数据集**

数据集涵盖三类：CICIDS‑2018（网络流），LID‑DS 2019（主机系统调用日志），以及自生成的 25,000 条虚拟化层事件数据，分别用于网络、主机与虚拟化层的评估。

**📈 对比分析**

与静态阈值（0.85）基线相比，RL 适配阈值将 LLM 升级次数从 2,689 降至 1,109，降幅 58.76%，显著降低成本；整体检测性能保持高水平，整体准确率 88.68%，网络层 98.02%，虚拟化层 97.08%，主机层 70.94%，并在 F1、精确率、召回率等指标上均优于基线。

**⚠️ 局限性**

局限性包括：1）在主机层仍存在噪声导致 F1 较低；2）实验基于离线数据，缺乏实时动态云环境验证；3）LLM 处理仍消耗计算资源，需进一步优化；4）对零日攻击的鲁棒性尚待深入研究。

---

## 362. Context-aware Entity-Relation Extraction for Threat Intelligence Knowledge Graphs

**arXiv ID:** 2605.15904 | [PDF](https://arxiv.org/pdf/2605.15904v1)

**作者:** Inoussa Mouiche `[一作]` (University of Windsor), sherif Saad `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了基于 SecureBERT^+ 的管道式 CTiKG 框架，用于从未结构化的网络威胁情报报告中准确提取实体与关系并构建安全情报知识图谱。

**💡 创新点**

创新点包括：①将 CRF 层嵌入 SecureBERT^+ + BiGRU 架构，显著提升 NER 的序列一致性；②在关系抽取中使用 TDD 层并结合领域本体进行后处理，减少误分类；③通过数据增强与类别合并扩充 DNRTI-STIX2 形成 DNRTI-AUG-STIX2，缓解类别不平衡；④公开完整数据集与代码，促进复现。

**🔧 技术方法**

技术要点：SecureBERT^+（安全 BERT 预训练）、BiGRU（双向 GRU 编码）、CRF（条件随机场）与 TDD（时间分布全连接）层、领域本体推理、Neo4j 图数据库与 Graph Data Science 库。

**📊 数据集**

使用数据集：DNRTI-AUG-STIX2（21 类实体、7947 句子）、DNRTI、STUCCO，并在 GitHub 公开。

**📈 对比分析**

对比方法：与 BiLSTM‑CRF、BERT‑BiGRU‑CRF、RoBERTa‑BiGRU‑TDD、SecureBERT‑BiGRU‑CRF 等基线模型进行实验。实验结果显示 NER F1 提升 3–4%，RE F1 提升 8%，并在 DNRTI、STUCCO 数据集上保持领先，验证了模型的稳健性与泛化能力。

**⚠️ 局限性**

局限性：①仍存在一定的误差传播，尤其在复杂多实体场景中；②依赖手工构建的本体规则，维护成本较高；③模型对非标准或多语言 CTI 报告的适应性尚待进一步提升。

---

## 363. Practical Validity Conditions for Byzantine-Tolerant Federated Learning

**arXiv ID:** 2605.15887 | [PDF](https://arxiv.org/pdf/2605.15887v1)

**作者:** Mélanie Cambus `[一作]` (Aalto University), Stefan Schmid `[通讯]` (Technical University Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文提出基于最小包围球（MEB）的可验证性条件，并引入其乘法松弛 c‑validity，以及针对 n>2t 情况下可实现的最优 MinMax 规则；

**💡 创新点**

创新点在于将传统凸包可验证性与 MEB 可验证性对齐，证明 c‑validity 在高维下可实现且松弛因子上限为 √2，拓宽了鲁棒聚合的理论边界；

**🔧 技术方法**

主要技术包括凸包与 MEB 计算（Welzl 算法）、Helly 定理、Soddy‑Gosset 圆的切点定理，以及二次锥规划分析；

**📊 数据集**

论文完全基于理论分析，未使用具体数据集；

**📈 对比分析**

通过与凸可验证性、盒子可验证性及 (δ,p)‑relaxed convex 可验证性等已有标准比较，表明 MinMax 可达最优 c<√2，MDA、几何中位数、medoid 等常用聚合亦满足常数倍 c‑validity；

**⚠️ 局限性**

局限包括：需要枚举所有 n-t 子集导致计算成本高、松弛因子上限可能仍非最优、未考虑去中心化场景，以及乘法与加法松弛的适用性需进一步探讨。

---

## 364. WorldAct: Activating Monolithic 3D Worlds into Interactive-Ready Object-Centric Scenes

**arXiv ID:** 2605.15843 | [PDF](https://arxiv.org/pdf/2605.15843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 365. Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?

**arXiv ID:** 2605.15877 | [PDF](https://arxiv.org/pdf/2605.15877v1)

**作者:** Mohammad Ali Vahedifar `[一作]` (Aarhus University), Qi Zhang `[通讯]` (Aarhus University)

**通讯引用:** 15261 | [OpenAlex ID](https://openalex.org/A5100360194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Shapley Neuron Values (SNV) 方法，用于在持续学习中不使用记忆缓冲也不扩展网络架构，通过冻结重要神经元实现知识的稳定与新任务的可塑性。

**💡 创新点**

创新点在于将 Shapley 值理论应用于神经元重要性评估，精确量化每个卷积过滤器对任务性能的贡献，进而按贡献度自动构建任务专属子网络。

**🔧 技术方法**

使用了 Monte Carlo 估计、截断优化与多臂赌博机策略相结合的 Shapley 值近似技术，以及掩码冻结机制对网络参数进行动态选择与冻结。

**📊 数据集**

在 ImageNet-1k、CIFAR-100 与 Tiny-ImageNet 的类增量学习 (CIL) 和任务增量学习 (TIL) 场景中进行实验。

**📈 对比分析**

与 EWC、SI、LwF、WSN、NFL+ 等无缓冲方法以及 iCaRL、DER++、DyTox 等记忆缓冲方法比较，SNV 在 ImageNet-1k 的 10/20/50 任务分别提升 2.88%/6.46% 的平均准确率，并在 CIL/TIL 场景中显著优于同类方法，甚至在多数情况达到或逼近记忆方法的性能。

**⚠️ 局限性**

主要局限在于 Shapley 估计阶段的计算开销随网络宽度和任务复杂度增大，导致后处理成本显著增加，限制了在极大模型上的实用性。

---

## 366. From Text to DSL: Evaluating Grammar-Based Model Generation Using Open LLMs

**arXiv ID:** 2605.15865 | [PDF](https://arxiv.org/pdf/2605.15865v1)

**作者:** Junaid Baber `[一作]` (University of Grenoble Alpes), Cécilia Satrin `[通讯]` (University of Grenoble Alpes)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了 39 种开源大语言模型（0.5B–32B 参数）在仅使用少量示例提示、无微调情况下自动生成符合域专用语言（DSL）语法、语义完整、跨模型引用一致的模型。

**💡 创新点**

提出了基于 LARK 语法检查器与人工专家评审相结合的两阶段评估管道，以及仅靠提示工程即可让小型模型产生可用 DSL 输出的实证方案。

**🔧 技术方法**

使用的技术包括提示工程、少量示例上下文、LARK 语法解析、重试机制、温度调节、以及基于 Web 的人工评估平台。

**📊 数据集**

使用从多领域低代码平台提取的自然语言场景作为输入数据集，涵盖了会议策划、冰淇淋店网站等典型案例。

**📈 对比分析**

对比方法先通过 LARK 语法解析过滤出可解析的模型，然后由三位领域专家对语义正确性、概念识别、完整性和高级特性进行 5 级评分；实验显示多款 1–8B 参数的模型在语义评分上与大型模型相当，且小型模型在提示优化后表现优异。

**⚠️ 局限性**

局限性包括：对特定 DSL 的依赖、提示设计对结果影响较大、未对模型进行微调、对极其复杂或领域特定的 DSL 仍可能缺乏深度覆盖、以及实验仅覆盖 39 种模型和有限场景，缺乏更广泛的泛化验证。

---

## 367. BootstrapAgent: Distilling Repository Setup into Reusable Agent Knowledge

**arXiv ID:** 2605.15815 | [PDF](https://arxiv.org/pdf/2605.15815v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 368. Toward Natural and Companionable Virtual Agents via Cross-Temporal Emotional Modeling

**arXiv ID:** 2605.15812 | [PDF](https://arxiv.org/pdf/2605.15812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 369. Designing for Robot Wranglers: A Synthesis of Literature and Practice

**arXiv ID:** 2605.15892 | [PDF](https://arxiv.org/pdf/2605.15892v1)

**作者:** David Porfirio `[一作]` (George Mason University), Thomas D. LaToza `[通讯]` (George Mason University)

**通讯引用:** 2830 | [OpenAlex ID](https://openalex.org/A5059412025)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统化的scoping review与作者自身的现场与想象性体验，对机器人“wrangler”角色进行全面的概念化与分类，提出了新的工作类型学和设计建议。

**💡 创新点**

创新点在于将“wrangler”视为一种隐喻性框架，揭示其多维、多角色、多时空维度的工作特征，并基于此提出了技术与制度层面的支持设计方案。

**🔧 技术方法**

采用了PRISMA式的文献筛选流程、主题编码与迭代讨论，以及基于服务设计的工作蓝图和角色映射工具，对不同情境下的wrangling活动进行系统描述。

**📊 数据集**

使用的主要数据来源为36篇符合条件的机器人研究文献（覆盖1986–2025年）以及作者团队在多行业现场的实践记录，未涉及传统机器学习数据集。

**📈 对比分析**

由于研究方法以定性分析为主，未进行数值对比或性能评估；结果以典型案例与归纳式结果呈现，强调经验与设计建议的可行性。

**⚠️ 局限性**

局限性包括：仅关注文献的宽度而非深度，未捕获如“handling”与“patchwork”等相关术语；缺乏对搜索与救援、协作机器人等其他领域的视角；以及对未受训练或非指定wrangler角色的研究不足。

---

## 370. CHoE: Cross-Domain Heterogeneous Graph Prompt Learning via Structure-Conditioned Experts

**arXiv ID:** 2605.15888 | [PDF](https://arxiv.org/pdf/2605.15888v1)

**作者:** Peiyuan Li `[一作]` (Tianjin University), Weixiong Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 12929 | [OpenAlex ID](https://openalex.org/A5068659777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种跨域异构图提示学习方法 CHoE，利用专家网络在预训练阶段学习通用知识，并在微调阶段通过结构感知专家路由、负载平衡与提示语义融合实现跨域知识迁移。

**💡 创新点**

创新点包括：①结构条件专家池，让每个专家专门处理源域的 meta‑path；②结构感知专家路由与负载平衡机制，根据目标图结构动态选择兼容专家；③提示语义融合模块，使用可学习提示向量对不同语义视角进行加权融合；④在预训练中结合掩码特征/边重建的自监督生成框架。

**🔧 技术方法**

技术方案：自监督生成预训练（masked feature/edge reconstruction + AutoEncoder）、专家网络（多层 HAN 作为专家）、结构感知路由（正负样本相似度评分+温度 softmax）、负载平衡（累计路由权重正则化）、提示语义融合（可学习提示向量与语义权重）。

**📊 数据集**

实验使用四个公共异构图数据集：ACM、DBLP、Aminer、Freebase，分别包含论文、作者、会议/电影等多种节点类型和 meta‑path 视角。

**📈 对比分析**

与 11 种基线（同源/异源自监督、图提示、跨域图提示、图 LoRA 等）在 5‑shot、1/3‑shot 等跨域节点分类任务中对比，CHoE 在大多数设置下显著提升 Macro‑F1 与 Micro‑F1，尤其在 Aminer 与 Freebase 的跨域场景下优势明显，且在源域的 in‑domain 任务上仍保持竞争力。

**⚠️ 局限性**

局限性：方法对 meta‑path 的定义仍依赖手工选择，且在极大规模或高度异构的图上可能因专家数量限制导致效果下降；在单一来源域预训练时对目标域的结构差异仍有一定残余影响，未来可进一步探索更强的自适应机制。

---

## 371. Ti-iLSTM: A TinyDL Approach for Logic-Level Anomaly Detection in Industrial Water Treatment Systems

**arXiv ID:** 2605.15874 | [PDF](https://arxiv.org/pdf/2605.15874v1)

**作者:** Mandar Joshi `[一作]` (University of Waikato), Emil Karlsson `[通讯]` (Aalto University)

**通讯引用:** 95 | [OpenAlex ID](https://openalex.org/A5108842404)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Ti-iLSTM框架，用 TinyDL 思想在 PLC 级工业水处理系统中实现逻辑层异常检测。

**💡 创新点**

创新点包括：① 通过三阶段特征筛选压缩至十个关键变量；② 采用可解释的过程规则构造无标签的逻辑层异常标签；③ 采用增量训练、短滑动窗口和轻量化 LSTM 以满足 PLC 资源限制。

**🔧 技术方法**

使用技术包括 TinyDL、增量 LSTM、短滑动窗口、SMOTE 欠采样平衡、相关系数+VIF+RFE+RF 三阶段特征选择、阈值调优。

**📊 数据集**

训练使用 SWaT 数据集，验证与跨数据集验证使用 WADI 水分配系统数据。

**📈 对比分析**

与 SWaT 现有基准模型比较，F1≈0.983、ROC‑AUC≈0.998，准确率≥0.98；在 WADI 上 F1≈0.957，显著优于 51 特征/120s STAE‑AD 等方法；资源使用 RSS≈708 MB，验证时间≈1.2 s，符合 PLC 硬件需求。

**⚠️ 局限性**

局限性：仅基于离线日志训练，未在现场实时验证；逻辑规则覆盖有限，未涵盖更复杂的阶段依赖；未进行完整的实时性能或功耗测量。

---

## 372. From Observed Viability to Internal Predictive Approximation: A Single-Subject Latent-Space Analysis of Gait Dynamics Under Occlusal Constraint

**arXiv ID:** 2605.15862 | [PDF](https://arxiv.org/pdf/2605.15862v1)

**作者:** Jacques Raynal `[一作]` (University of Montpellier), Jacques Margerit `[通讯]` (University of Montpellier)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在单个帕金森患者的两次步态测试中，构建了PC1-PC2低维潜在空间，并用前向神经网络在同一受试者数据上实现M1→M2潜在位移的内部近似。

**💡 创新点**

提出了Level 5框架，首次将可观测的纵向可行性转化为可在潜在空间内内部预测的轨迹近似，并验证了潜在位移层级的可保持性。

**🔧 技术方法**

使用主成分分析（PCA）做潜在表示，随后用简化的前馈神经网络（两层16单元）实现监督学习预测。

**📊 数据集**

单个帕金森受试者的步态数据，11周间隔两次测量，每次记录六种咬合探针共12组记录。

**📈 对比分析**

通过全数据、留M2、留条件三种内部验证，模型能保持OC3<ONL<OC2.5的位移层级，均值质心误差0.01–0.20，RMSE≈1.8，说明能较好近似潜在轨迹。

**⚠️ 局限性**

仅限单受试者、缺乏因果推断、PCA线性约束、网络结构简单、未验证跨人群推广，结果仅为内部证明，无法作为临床预测工具。

---

## 373. Conversations in Space: Structuring Non-Linear LLM Interactions on a Canvas

**arXiv ID:** 2605.15848 | [PDF](https://arxiv.org/pdf/2605.15848v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 374. GHOST: Geometry-Hierarchical Online Streaming Token Eviction for Efficient 3D Reconstruction

**arXiv ID:** 2605.15852 | [PDF](https://arxiv.org/pdf/2605.15852v1)

**作者:** Leyang Chen `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 23671 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无训练的 KV 缓存管理框架 GHOST，利用几何层次的重要性评分和特殊令牌特权在线蒸发多余 token，实现高效长序列 3D 重建。

**💡 创新点**

创新点包括：1）双层几何重要性评分（帧级与 token 级）；2）特殊令牌特权机制保护全局姿态与结构信息；3）基于余弦相似度的层级预算分配，动态调整不同 Transformer 层的 token 预算。

**🔧 技术方法**

技术手段包括：Transformer（StreamVGGT）网络、模型自身输出的 depth/point/pose 作为自监督几何信号、离线余弦相似度分析、在线增量更新的重要性评分、无训练的 token 逐层蒸发策略。

**📊 数据集**

使用的公开数据集：Bonn、7‑Scenes、NRGBD 与 Long3D（含 2,128–9,545 帧长序列）。

**📈 对比分析**

与 VGGT、StreamVGGT、CUT3R、Point3R、TTT3R、InfiniteVGGT 等基线对比，GHOST 在所有 benchmark 上均实现更低的误差、更高的精度，KV 缓存约减半，推理速度提升约 1.75×。

**⚠️ 局限性**

局限性：依赖模型输出的几何信号，若 depth/point 预测不可靠可能导致误判；需要手动调节若干重要性权重；未进一步优化极长序列的实时性与内存峰值。

---

## 375. Exploration of $k$-edge-deficient temporal graphs in linear time

**arXiv ID:** 2605.15833 | [PDF](https://arxiv.org/pdf/2605.15833v1)

**作者:** Ivan Lahtin `[一作]` (Moscow Institute of Physics and Technology), Viktor Zamaraev `[通讯]` (University of Liverpool)

**通讯引用:** 378 | [OpenAlex ID](https://openalex.org/A5047948708)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在近静态的时变图（always‑connected k‑edge‑deficient 形式）中提出了一个 O(nklogk) 的探索调度，并证明其可在多项式时间内构造。

**💡 创新点**

突破了之前的 O(knlogn) 上界，几乎匹配已知的 Ω(nlogk) 下界，且首次将探索时间对时间连通参数 Δ 进行线性化；引入了“roundabout”多代理并行探索、DFS 循环覆盖、以及随机/Las Vegas 选择覆盖元组等新技术。

**🔧 技术方法**

利用深度优先搜索的 DFS 循环、圆形区间覆盖、Δ‑时连通性、k‑edge‑缺失的统计分析、随机试验与概率论证明、以及常数时间的“首到”算法实现最优时间步调度。

**📊 数据集**

该工作纯粹是理论分析，不使用实验数据集；所有结论均通过严格的数学证明得出。

**📈 对比分析**

与此前的 O(knlogn) 上界相比，新的上界去掉了对 n 的对数因子，在 k 为常数时实现 Θ(n) 的线性探索；实验验证未给出，但理论上已达到近最优。

**⚠️ 局限性**

仍然存在 k 因子导致的上界与下界之间的乘积差距；方法依赖于 Δ‑时连通性与 k‑edge‑缺失假设，尚未证明在更弱的时变稳定性下是否可保持线性时间。

---

## 376. Intrinsic Wasserstein Rates for Score-Based Generative Models on Smooth Manifolds

**arXiv ID:** 2605.15822 | [PDF](https://arxiv.org/pdf/2605.15822v1)

**作者:** Guoji Fu `[一作]` (National University of Singapore), Atsushi Nitanda `[通讯]` (Agency for Science, Technology and Research)

**通讯引用:** 405 | [OpenAlex ID](https://openalex.org/A5023953123)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

证明了在紧致光滑流形上的方差保持式扩散模型（VP‑SGM）在 Wasserstein‑1 误差下可达到与流形内在维度相关的采样指数 n^{-(β+1)/(d+2β)}（上限为该指数乘以多项式的环境维度、几何与密度因子及对数因子）。

**💡 创新点**

创新点包括：① 将高噪声与低噪声两种尺度的分数逼近分别拆分为切线细胞逼近和投影中心化的拉普拉斯展开；② 采用有限锚点高斯-牛顿网络在局部 d‑维坐标系中实现投影中心的高效逼近；③ 明确给出了环境维度的多项式上界并在多项式几何控制下保持网络参数规模多项式；④ 通过剪裁的最小二乘估计与分段时间切片组合，完成了完整的理论证明。

**🔧 技术方法**

使用技术包括：深度 ReLU 神经网络逼近理论、有限阶几何包络（geometry envelope）与 Hölder 光滑度、切线细胞分解、投影中心化拉普拉斯展开、剪裁最小二乘 (clipped ERM)、Wasserstein‑1 误差传播与分段时间积分、Gauss–Newton 迭代、分段时间切片与 ReLU 时间切换。

**📊 数据集**

本工作主要为理论研究，没有使用具体的数据集；所有结论均基于假设为紧致光滑流形、β‑Hölder 连续密度且满足正密度下界的连续概率分布。

**📈 对比分析**

与先前工作比较：本论文在满足多项式几何与密度下界的条件下，获得了与流形内在维度相关的最佳采样指数，且环境维度因子为多项式（如 √D 或 D^{β(d)}），在 d>2 时修正项可吸收，最终误差达到 n^{-(β+1)/(d+2β)}，与理论上界一致；相较于之前指数级环境依赖或更慢的收敛速率，显著提升。

**⚠️ 局限性**

局限性：① 需要严格的流形支持、正密度下界与多项式几何控制；② 仅考虑理想化的分段时间切片估计，未包含参数共享的实际训练、优化与离散化误差；③ 对小噪声区的可行性检验依赖于显式的可接受性条件；④ 目前未给出对观测噪声或非光滑流形的扩展。

---

## 377. Which Moments Matter? Heuristics of Remembered Travel Experience in Public Transport

**arXiv ID:** 2605.15817 | [PDF](https://arxiv.org/pdf/2605.15817v1)

**作者:** Esther Bosch `[一作]` (German Aerospace Center), Stefan Bohmann `[通讯]` (University of Bundeswehr)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过在德国三城的2576次公共交通旅程中，每5分钟使用手机体验采样收集旅程中的多维体验评分，并在旅程结束后记录整体满意度，研究旅程体验如何被人们记忆与综合。

**💡 创新点**

创新点在于首次将高频率、细粒度的体验采样与多层次回归模型相结合，系统比较多种时间聚合启发式（均值、峰-末、最小-末、末尾、最小、持续时间）对记忆后满意度的解释力，并发现“最小-末”规则在公共交通情境中最具预测力。

**🔧 技术方法**

使用的技术包括基于移动端的体验采样应用、8维旅行体验量表、后期的三项缩减量表、混合效应线性模型、AIC/BIC模型比较以及边际与条件R²评估。

**📊 数据集**

数据集为来自汉堡、柏林和图特林根的349名受试者共2576次旅程，每次旅程至少收集3次5分钟一次的体验评分，旅程平均时长约32分钟，最终用于计算旅程体验的综合评分和后期满意度评分。

**📈 对比分析**

通过在多层次模型中仅加入不同聚合规则的单一预测变量，比较模型的AIC/BIC，发现最小-末模型的AIC最低（3329.58）且R²为0.649（边际），条件R²为0.701，明显优于均值、峰-末或持续时间等其他规则，说明最小-末规则在解释记忆后满意度方面表现最佳。

**⚠️ 局限性**

局限性包括样本主要来自汉堡，可能影响结果外推；受试者对体验评分的自我报告和多次测量可能引入回忆偏差或对体验的干扰；未利用事件注释与传感器数据精确识别导致最小值的具体情境；且仅关注公共交通，缺乏对其他交通模式的比较。

---

## 378. On RGB-TIR Stereo Calibration under Extreme Resolution Asymmetry

**arXiv ID:** 2605.15860 | [PDF](https://arxiv.org/pdf/2605.15860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 379. Community-aware evaluation and threshold calibration for open-set plankton image recognition

**arXiv ID:** 2605.15835 | [PDF](https://arxiv.org/pdf/2605.15835v1)

**作者:** Xi Chen `[一作]` (Guangzhou University), Gang Fang `[通讯]` (Guangzhou University)

**通讯引用:** 7064 | [OpenAlex ID](https://openalex.org/A5071245512)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在开放集条件下，自动浮游生物图像识别对生态社区估计的影响，提出了以社区层面误差为核心的评价指标Open‑Set Community Distortion（OSCD）以及对应的方向性诊断；同时给出了基于验证社区的阈值校准方法；

**💡 创新点**

创新点在于：①将开放集识别从样本级别转向生态社区级别，定义OSCD及其正负方向分量；②揭示样本级别阈值与社区估计之间的误差匹配不一致性；③提出基于验证社区OSCD的阈值校准策略，并对其适用边界进行系统分析；

**🔧 技术方法**

技术上采用冻结的DINOv2 ViT‑S/14图像编码器+线性分类头，随后利用后置OOD评分（最大softmax概率、能量、原型距离、马氏距离）生成阈值；通过伪社区抽样、阈值扫描、方向性误差分析等方法实现社区级评价；

**📊 数据集**

使用三大公开数据集：SYKE‑ZooScan 2024（海洋浮游动物）、SYKE‑IFCB 2022（海洋浮游植物）、ZooLake（淡水浮游生物），并在各数据集上划分已知/未知类、构造验证/测试伪社区；

**📈 对比分析**

与传统的样本级AUROC、AUPR、FPR@95%等指标对比，发现单纯优化样本级阈值往往导致社区估计误差扩大；在验证社区上进行OSCD校准后，MSP阈值在SYKE‑ZooScan与SYKE‑IFCB上分别将OSCD从≈0.14降至≈0.12，匹配全球oracle；但在ZooLake中阈值校准提升有限；总体上，社区阈值校准能显著提升社区估计精度，尤其当验证社区与部署环境相近时；

**⚠️ 局限性**

局限性包括：①阈值校准依赖验证社区与真实部署社区的相似性，若不匹配效果减弱；②OSCD主要关注相对丰度误差，对多样性、丰富度等指标的关联性有限；③实验仅基于冻结编码器与后置OOD评分，未探索Fine‑tune或专用开放集训练方法；④在极端长尾或非目标丰富的场景下，校准收益有限。

---

## 380. Fast Expanding Safe Circular Regions for Efficient Local Path Planning

**arXiv ID:** 2605.16009 | [PDF](https://arxiv.org/pdf/2605.16009v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 381. Echo-Forcing: A Scene Memory Framework for Interactive Long Video Generation

**arXiv ID:** 2605.16003 | [PDF](https://arxiv.org/pdf/2605.16003v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 382. LoCO: Low-rank Compositional Rotation Fine-tuning

**arXiv ID:** 2605.15916 | [PDF](https://arxiv.org/pdf/2605.15916v1)

**作者:** An Nguyen `[一作]` (Korea University), Anh Tong `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LoCO——一种基于低秩反对称矩阵构造正交变换的参数高效微调方法，能够在不显著增大参数量的前提下对大型预训练模型进行适配；

**💡 创新点**

创新点在于：①使用低秩skew‑symmetric矩阵配合Cayley变换与Woodbury矩阵恒等式，显著降低正交矩阵计算复杂度；②通过一阶近似将多重旋转链并行化，既保持正交性又实现GPU级别的并行；③引入温度缩放实现推理时可调的适配强度；

**🔧 技术方法**

采用低秩skew‑symmetric参数化、Cayley变换、Sherman‑Morrison‑Woodbury逆、第一阶近似并行旋转、温度缩放以及PEFT框架；

**📊 数据集**

使用的数据集包括GLUE、MetaMathQA‑40K（GSM8K、MATH）、VTAB‑1k、FLUX.1（条件生成任务）、COCO（可控生成任务）等；

**📈 对比分析**

与LoRA、OFT、BOFT、HRA等现有PEFT方法进行对比，LoCO在NLP（GLUE、数学推理）、视觉（VTAB‑1k）和生成（FLUX.1）任务上实现与或优于对照的性能，同时保持更低的可训练参数量和更高的训练/推理效率；

**⚠️ 局限性**

局限性包括：①低秩假设在极高维度下可能不够充分；②高维空间下的可扩展性仍需进一步优化；③温度缩放机制的理论分析仍有限；④尚未在多模态或更大规模模型上进行充分验证。

---

## 383. CitePrism: Human-in-the-Loop AI for Citation Auditing and Editorial Integrity

**arXiv ID:** 2605.16000 | [PDF](https://arxiv.org/pdf/2605.16000v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 384. On the parameterized complexity of Broadcast Independence and Broadcast Packing

**arXiv ID:** 2605.16001 | [PDF](https://arxiv.org/pdf/2605.16001v1)

**作者:** Joanne Dumont `[一作]` (University of Orléans), Florian Sikora `[通讯]` (Université Paris-Dauphine)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了广播独立（Independent Broadcast）和广播打包（Broadcast Packing）问题在参数化复杂度下的性质，给出了以树宽+直径为参数的 FPT 算法、k+树宽为参数的 XP/ FPT 结果以及多路参数（路径宽、顶点覆盖、反馈顶点集）下的 W[1]-hardness 证明，并提出了一个 1/2‑ε 的参数化近似算法。

**💡 创新点**

首次系统性地从参数化复杂度角度分析这两个问题，提出了树宽+直径的动态规划框架，完成了 k 与树宽组合参数化的完整图景，并给出了对加权版本的 W[1]-hardness 结果以及结构性逼近的理论界定。

**🔧 技术方法**

主要采用了树分解（nice tree decomposition）与签名（signature）技术的动态规划、对多色团的参数化归约（证明 W[1]-hardness），以及结构性改造（p‑广播构造）来实现近似。

**📊 数据集**

该工作为纯理论研究，无实验数据集。

**📈 对比分析**

与已知的树算法（如 Bessy&Rautenbach 的树算法）以及多色团归约进行对比，证明了所给算法的时间复杂度（FPT、XP、W[1]-hardness 上界）并通过结构性结果证明 1/2‑ε 的逼近率。

**⚠️ 局限性**

尚未解决 Broadcast Packing 在树宽下的确切复杂度；对加权版的 BIB 的 W[1]-hardness 只在特定参数下得到；缺乏实验验证和对更广泛图类（如有界 cliquewidth）的扩展。

---

## 385. WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation

**arXiv ID:** 2605.15964 | [PDF](https://arxiv.org/pdf/2605.15964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 386. Defining Cultural Capabilities for AI Evaluation: A Taxonomy Grounded in Intercultural Communication Theory

**arXiv ID:** 2605.15990 | [PDF](https://arxiv.org/pdf/2605.15990v1)

**作者:** Isar Nejadgholi `[一作]` (National Research Council), Maryam Molamohamadi `[通讯]` (Mila, Quebec AI Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于跨文化交际理论的三层次文化能力分类——文化意识、文化敏感性与文化适应性，并针对自然语言生成系统的可观测行为进行定义；

**💡 创新点**

创新点在于将人类跨文化交际模型（DMIS、CQ、PMIC）转化为可操作的AI评测框架，明确区分知识性、表述性与动态适应性三类能力，解决了现有评测中术语混乱与构念不清的问题；

**🔧 技术方法**

使用了现有跨文化评测基准与多轮对话模拟方法，结合基于文本的立场检测、刻板印象评估、情境适应评测等NLP技术；

**📊 数据集**

参考的数据集包括 GeoMLAMA、FORK、BLEnD、INCLUDE、CulturalBench、SHADES、MC-SIGNS、NormGenesis、SocialCC、Nunchi-Bench 等多语言多文化知识与情境评测数据；

**📈 对比分析**

通过对比现有基准的测评结果，展示缺乏构念清晰导致的评测结果误读；本文未给出具体性能数值，但指出单一维度评测无法反映系统在多轮对话中的适应性与敏感性；

**⚠️ 局限性**

限制在于：测评框架无法自动化解决英语中心化带来的社会技术危害；未覆盖所有可能的文化能力；基于外部视角的跨文化模型可能忽视用户的内部经验；在低资源语言与真实世界情境下的适用性尚待进一步研究。

---

## 387. Petri Net Induced Heuristic Search for Resource Constrained Scheduling

**arXiv ID:** 2605.15983 | [PDF](https://arxiv.org/pdf/2605.15983v1)

**作者:** Ido Lublin `[一作]` (Bar-Ilan University), Izack Cohen `[通讯]` (Bar-Ilan University)

**通讯引用:** 928 | [OpenAlex ID](https://openalex.org/A5041097802)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将资源约束项目调度问题（RCPSP）建模为带资源的定时转换Petri网的可达图上的最优搜索问题；

**💡 创新点**

创新点在于使用相对延迟标记统一编码时间和资源约束，实现启发式一致性、可安全剪枝并利用零代价缓存；

**🔧 技术方法**

采用A*搜索、结合关键路径和资源负载两种下界的启发式、Petri网可达图构造以及标准时间索引MIP模型进行对比；

**📊 数据集**

使用PSPLIB公共基准集（J30、J60、J90）共1440个实例进行实验；

**📈 对比分析**

在单核5分钟超时下，TTPNR相较于SCIP和CBC在成功率和求解时间上均取得显著优势，尤其在资源紧张和多资源需求场景；

**⚠️ 局限性**

局限性包括对低资源强度实例的启发式指导不够强、对超大规模实例的可达图规模限制以及未尝试双向搜索或更深层次算法选择。

---

## 388. Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective

**arXiv ID:** 2605.15976 | [PDF](https://arxiv.org/pdf/2605.15976v1)

**作者:** Ernesto Garcia-Estrada `[一作]` (Universitat Politecnica De Catalunya), José A. R. Fonallosa `[通讯]` (Universitat Politecnica De Catalunya)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 NLLB-200 600M/1.3B encoder‑decoder 机器翻译模型上，利用参考无监督奖励（LaBSE+COMET‑Kiwi）和 Group Relative Policy Optimization，对 13 种目标语言进行 RL 微调，并在单 GPU 上实现可行的训练流程。

**💡 创新点**

首次在 encoder‑decoder Seq2Seq 上系统评估 GRPO，发现参考无监督 RL 在低资源/弱基线语言上能匹配或超过三轮监督微调，并揭示“基线越低、奖励区分度越高，RL 收益最大”的经验规律。

**🔧 技术方法**

使用 Group Relative Policy Optimization、4‑bit NF4 量化、LoRA 参数高效微调、混合奖励（LaBSE + COMET‑Kiwi）以及 KL 正则化。

**📊 数据集**

训练数据为 FLORES‑200 开发集的源句子（≈1k/语言），交叉域实验使用 CCNews 10k 英文句子；评估基准为 FLORES‑200 devtest、NTREX‑128。

**📈 对比分析**

与基线 NLLB、单轮和三轮监督微调相比，GRPO 在弱基线语言上可达 +5 chrF++，与三轮 SFT 在大多数语言相当；在多域/无目标数据情形下表现更具优势。

**⚠️ 局限性**

主要限制包括奖励模型预训练需并行语料、对低资源/未覆盖语言奖励辨识度不足、仅评估英语源语言、可能出现奖励方差崩塌导致不稳定、以及仅在 13 语言、600M/1.3B 模型上验证，缺乏更广泛的普适性。

---

## 389. To GPU or Not to GPU: Vector Search in Relational Engines

**arXiv ID:** 2605.15957 | [PDF](https://arxiv.org/pdf/2605.15957v1)

**作者:** Vasilis Mageirakos `[一作]` (ETH Zürich), Gustavo Alonso `[通讯]` (ETH Zürich)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于T​PC‑H扩展的 Vec‑H 基准和 MaxVec 执行引擎，对 CPU‑GPU 异构环境下的 SQL+向量搜索性能进行了系统评估，并提出多种执行策略。

**💡 创新点**

创新点在于：①将向量索引拆分为非数据拥有（non‑owning）设计，显著降低 GPU 内存占用与传输成本；②设计了多策略决策框架（cpu、gpu、copy‑di、copy‑i、gpu‑i、hybrid），并在统一内存体系结构下证明 GPU 总是优于 CPU。

**🔧 技术方法**

使用技术包括：FAISS / cuVS 进行向量搜索；Maximus 框架实现 CPU‑GPU 级联执行；Arrow / cuDF 内存格式、pinning、缓存、ATS 统一内存等优化；在 PCIe 5.0、NVLink‑C2C、DGX‑Spark 等硬件平台上测试。

**📊 数据集**

数据集为：在 T​PC‑H 的基础上加入 Amazon Reviews 与图片嵌入，使用 Qwen 0.6B（1024 维）和 SigLIP2（1152 维）生成向量，SF=1 时关系表约 1 GB，向量表约 12 GB。

**📈 对比分析**

实验方法：对 20 次重复（去除 warm‑up）记录查询延迟；对比 PGVector CPU 基线、MaxVec CPU、MaxVec GPU、hybrid 等六种策略；结果显示 GPU 在所有索引与查询中最快，relational 加速贡献最大；数据拥有索引在 GPU 上不划算，而非数据拥有索引和 hybrid 能显著提升性能。

**⚠️ 局限性**

局限性：仅在 SF=1 的规模下评估；ANN 误差对结果质量影响未深入探讨；缺乏自动化的查询优化器来选取最佳策略；统一内存方案在普通系统上不可直接迁移。

---

## 390. Driving Through the Network: Performance and Workload Under Latency and Video Impairment

**arXiv ID:** 2605.15952 | [PDF](https://arxiv.org/pdf/2605.15952v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 391. Skew Constacyclic Codes Of Length $np^s$ over $ \frac{\mathbb{F}_{p^m}[u]}{\langle u^k \rangle}

**arXiv ID:** 2605.15925 | [PDF](https://arxiv.org/pdf/2605.15925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 392. Ascend-RaBitQ: Heterogeneous NPU-CPU Acceleration of Billion-Scale Similarity Search with 1-bit Quantization

**arXiv ID:** 2605.16007 | [PDF](https://arxiv.org/pdf/2605.16007v1)

**作者:** Fujun He `[一作]` (Huawei Technologies Co Ltd), Yunfei Du `[通讯]` (Huawei Technologies Co Ltd)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Ascend-RaBitQ：在 NPU-CPU 异构系统上实现 1-bit RaBitQ IVF 搜索，解耦粗精度与精度阶段。

**💡 创新点**

创新点在于三阶段异构流水线、针对 Ascend NPU 的四项架构原生优化（AIC-AIV 结合、旋转正交重排、1-bit FastScan、块级负载均衡）以及 NPU-CPU 与主机 CPU 的多级流水线。

**🔧 技术方法**

使用 1-bit RaBitQ、IVF、AIC-AIV 并行、FastScan、统一缓存管理、异步数据搬移、块级调度与 AI CPU Top‑k 并行。

**📊 数据集**

在 SIFT1M、Cohere10M、SIFT100M、SIFT1B 四大公开数据集上评测。

**📈 对比分析**

相较于 CPU 上的 IVF‑RaBitQ、FAISS 及 Ascend IVF‑Flat，Ascend‑RaBitQ 在查询吞吐量上 1.2‑4.6 倍提升，索引构建速度提升 3‑62.8 倍，100 倍以上超过同等量化的 CPU 基线。

**⚠️ 局限性**

局限在于多 NPU 扩展时吞吐量仅 3 倍，CPU 重新排序成为瓶颈，且块大小与负载均衡仍需进一步优化。

---

## 393. PAGER: Bridging the Semantic-Execution Gap in Point-Precise Geometric GUI Control

**arXiv ID:** 2605.15963 | [PDF](https://arxiv.org/pdf/2605.15963v1)

**作者:** Jingxuan Wei `[一作]` (University of Chinese Academy of Sciences), Cheng Tan `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1942 | [OpenAlex ID](https://openalex.org/A5006542157)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了精度敏感的GUI任务，并构建了专门针对几何绘制的基准PAGE Bench，研发了PAGER框架实现精确点级绘制。

**💡 创新点**

创新点在于将区域容忍交互与连续空间点精度需求对齐，首次将几何推理、过程监督与精度对齐强化学习结合，解决坐标误差级联问题。

**🔧 技术方法**

采用了依赖结构化规划 + 像素级执行的两阶段架构，利用像素级监督微调（SFT）学习可执行语法，并用参数与操作类型双重奖励的强化学习对点精度进行优化。

**📊 数据集**

使用了从公开K‑12几何题库筛选并重构为可执行GeoGebra轨迹的数据集PAGE Bench，包含4,906道几何题、53,277高层任务和224,497低层GUI动作。

**📈 对比分析**

通过与多款开源与闭源VLM、专用GUI代理的对比实验，PAGER在总体得分、任务成功率、最终几何质量上均领先，尤其在Task Success从基线提升至约23.8，超过同类方法4.1倍。

**⚠️ 局限性**

局限性在于仅针对平面GeoGebra样式的几何绘制，缺乏对其他CAD、图表编辑或科学可视化等更复杂精度敏感界面的适配与验证。

---

## 394. PersonaFingerprint: Measuring Persona Inference on Modern Websites with LLM-Driven Browsing

**arXiv ID:** 2605.15962 | [PDF](https://arxiv.org/pdf/2605.15962v1)

**作者:** Chuxu Song `[一作]` (Rutgers University), Richard Martin `[通讯]` (Rutgers University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究在现代网站环境下，仅利用加密流量的包长度与时延信息，从中推断用户的行为人格（Persona Fingerprinting）并构建可大规模生成带标签流量的LLM驱动多代理浏览框架；

**💡 创新点**

创新点在于提出并量化Persona Fingerprinting这一新型隐私风险、使用LLM代理生成可控且多样化的真实网站浏览轨迹、证明现有网站指纹模型已隐含人格信息，并展示联合多任务训练可低成本提升人格识别；

**🔧 技术方法**

技术包括1D CNN序列编码器、软最大分类头、轻量MLP探针、联合多任务损失、LLM决策代理+电脑使用代理的多代理框架；

**📊 数据集**

数据集为十个主流现代网站（Amazon、YouTube、Reddit等）上的约800,000个1,000包长度窗口，覆盖15个定义清晰的人格标签加一个开放世界标签；

**📈 对比分析**

与传统网站指纹基线相比，单纯使用1,000包窗口即可达到约93%的网站识别精度，人物识别在单站点上平均84%（宏F1>0.8），混合站点人物识别约84%；利用探针可将人物识别从随机水平提升约20–30%；联合多任务可在保持90%网站精度的前提下将人物精度提升至80%；随着训练样本增至5,000/人格窗口，混合站点人物精度可升至84%；

**⚠️ 局限性**

局限性包括：①仅使用固定长度窗口，未考虑更长的会话级特征；②LLM生成的人格行为可能无法完全覆盖真实人类多样性；③实验仅在十个网站，可能无法推广到更广泛域；④未评估在不同网络条件和设备下的鲁棒性；

---

## 395. Sparse Autoencoders enable Robust and Interpretable Fine-tuning of CLIP models

**arXiv ID:** 2605.15961 | [PDF](https://arxiv.org/pdf/2605.15961v1)

**作者:** Fabian Morelli `[一作]` (University of Tübingen), Seong Joon Oh `[通讯]` (University of Tübingen)

**通讯引用:** 8601 | [OpenAlex ID](https://openalex.org/A5025851635)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对大规模视觉‑语言模型（如 CLIP）进行鲁棒微调，提出仅对视觉表示进行正则化的 SAE‑FT 方法。

**💡 创新点**

创新点在于利用预训练模型的稀疏自编码器（SAE）定义可解释的语义特征基底，对微调过程中的表示变动施加稀疏与保留约束，从而在不抹除原有概念的前提下实现任务适配。

**🔧 技术方法**

核心技术包括：稀疏自编码器训练、特征空间残差对齐惩罚、特征增删惩罚、交叉熵损失与自编码器正则项的联合优化；仅更新视觉编码器与线性分类头。

**📊 数据集**

主要使用 ImageNet 及其分布漂移子集（ImageNet‑R、A、Sketch、V2）、iWildCam、FMoW、CIFAR‑10/100、Caltech‑101 与 STL‑10 等公开数据集进行评估。

**📈 对比分析**

与 WiSE‑FT、FLYP、CAR‑FT、CaRot、StarFT 等现有鲁棒微调方法对比，SAE‑FT 在 ImageNet、各分布漂移任务上平均准确率最高（≈64.6%），在 downstream 迁移任务上取得最优平均分（≈87.8%），并在 iWildCam/FMoW 的 OOD 性能上处于领先或相当水平。

**⚠️ 局限性**

局限性包括：仅针对视觉编码器的正则化，未对文本编码器做相应约束；在与简单 L2 正则化的差异不大时提升有限；以及稀疏自编码器需在预训练模型上额外训练一次，增加少量前期成本。

---

## 396. RaPD: Resolution-Agnostic Pixel Diffusion via Semantics-Enriched Implicit Representations

**arXiv ID:** 2605.15908 | [PDF](https://arxiv.org/pdf/2605.15908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 397. End-to-end plaque counting and virus titration from laboratory plate images with deep learning

**arXiv ID:** 2605.16008 | [PDF](https://arxiv.org/pdf/2605.16008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 398. When and Why Adversarial Training Improves PINNs: A Neural Tangent Kernel Perspective

**arXiv ID:** 2605.15959 | [PDF](https://arxiv.org/pdf/2605.15959v1)

**作者:** Yuan-dong Cao `[一作]`, He Wang `[通讯]` (University College London)

**通讯引用:** 257082 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种新的分析框架，用于理解对抗训练的物理信息神经网络（PINNs），并开发了一种新的高效训练算法。

**💡 创新点**

通过分析GAN中的判别器如何影响PINNs的训练动态，提供了对抗训练在PINNs中有效性的理论基础，并提出了一种新的训练算法。

**🔧 技术方法**

使用了对抗训练和生成对抗网络（GAN）技术，结合了神经切线核（NTK）分析。

**📊 数据集**

使用了多种偏微分方程（PDE）作为数据集，包括泊松方程、拉普拉斯方程、粘性伯格斯方程、反应扩散方程和克莱因-戈登方程。

**📈 对比分析**

与传统的GAN、LSGAN和WGAN-GP等方法进行了比较，结果表明，提出的方法在训练和验证误差上显著优于这些基线，尤其是在复杂的PDE问题上。

**⚠️ 局限性**

分析框架假设了固定的NTK，可能无法适应网络宽度有限和PDE变化的实际情况，且在某些设置下训练仍可能出现振荡。

---

## 399. Vectorized Generalized Nearest Neighbor Decoding for In-block Memory Channel

**arXiv ID:** 2605.15950 | [PDF](https://arxiv.org/pdf/2605.15950v1)

**作者:** Yuhao Liu `[一作]` (Tsinghua University), Wenyi Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8802 | [OpenAlex ID](https://openalex.org/A5100360013)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出向量化广义最近邻解码（Vec-GNND）并在块内记忆（IBM）信道上进行GMI分析，推导出最优解码指标及其闭式表达，并给出了联合码本-指标设计框架。

**💡 创新点**

创新点在于首次在IBM信道中实现块级向量化解码，利用块内协方差的PCA与缩放实现信息保留；提供最优处理与缩放函数的闭式解，同时提出码本协方差的自洽优化条件，且对多种受限形式给出解析。

**🔧 技术方法**

技术手段包括GMI下的随机编码分析、矩阵特征分解、Bussgang分解、弱约束下的最优缩放/处理函数求解、以及水filling式自洽方程用于输入协方差优化。

**📊 数据集**

使用仿真生成的块非相干AWGN信道与相位噪声信道作为实验数据集，并未使用公开的实际数据集。

**📈 对比分析**

与传统基线（尺度匹配、逐符号GNNDR、恒等解码器）对比，数值实验表明Vec-GNND在SNR升高及块长度增大时可显著提升GMI，尤其在高SNR下实现无穷大增益。

**⚠️ 局限性**

局限性包括：仅适用于高斯码本且需IBM假设；非对角协方差优化的全局最优条件尚未求解，计算量大；实现上需要估计条件均值与协方差，导致实现复杂。

---

## 400. Quantum Futures Interactive: A Live Demonstration of Post-Quantum Blockchain Security, Infrastructure Tradeoffs, and Sustainable Distributed Trust

**arXiv ID:** 2605.15991 | [PDF](https://arxiv.org/pdf/2605.15991v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 401. FocalPolicy: Frequency-Optimized Chunking and Locally Anchored Flow Matching for Coherent Visuomotor Policy

**arXiv ID:** 2605.15944 | [PDF](https://arxiv.org/pdf/2605.15944v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 402. OHP-RL: Online Human Preference as Guidance in Reinforcement Learning for Robot Manipulation

**arXiv ID:** 2605.15971 | [PDF](https://arxiv.org/pdf/2605.15971v1)

**作者:** Yunyang Mo `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2084 | [OpenAlex ID](https://openalex.org/A5109900808)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出OHP-RL框架，将人类干预视为在线偏好信号以指导机器人学习；

**💡 创新点**

通过状态相关偏好门控自适应调节人类干预对策略更新的影响；

**🔧 技术方法**

采用异步Actor-Critic、偏好门控学习、配对偏好更新及软Q学习；

**📊 数据集**

在Franka机械臂上使用三种真实物理接触丰富的操控任务（按铃、推球、搬运沙袋）；

**📈 对比分析**

与HG-DAgger、HIL-SERL、SIL-RI、HACO、PVP*等基线对比，OHP-RL在成功率、干预频率和执行效率上均显著优于基线，且迁移至无干预执行时表现最佳；

**⚠️ 局限性**

受限于任务周期较长时学习难度增加、稀疏奖励与环境噪声导致的价值估计不稳，以及对成功状态可观测度的依赖。

---

## 403. Deterministic Event-Graph Substrates as World Models for Counterfactual Reasoning

**arXiv ID:** 2605.15967 | [PDF](https://arxiv.org/pdf/2605.15967v1)

**作者:** Fabio Rovai `[一作]` `[通讯]` (Tesseract Academy), Fabio Rovai (Tesseract Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于事件图（event‑graph）子系统的世界模型，使用可扩展的RDF事件日志记录代理记忆，借助确定性重放与结构化干预实现精确的反事实推理，并通过单一的因果祖先遍历实现解释与反事实查询的图论对偶。

**💡 创新点**

创新点在于：
1) 对事件图子系统给出了完整的形式化定义和确定性重放语义；
2) 证明了在闭事件假设下，解释查询与反事实查询可通过同一祖先遍历获得（祖先对偶定理）；
3) 开发了无学习组件、可跨域迁移的解释器，支持对RDF三元组的逐条可审计；
4) 引入了 twin‑EventLog 计数器反事实基准，用于评估代理记忆在干预下的一致性。

**🔧 技术方法**

技术包括：
- 结构化干预语义（do 操作）与确定性重放函数；
- 基于RDF和SPARQL 的事件日志存储（Oxigraph）；
- 逆向广度优先搜索计算因果祖先集合；
- 逆向对偶定理实现反事实判断；
- 简单的线性动力学投影实现预测查询；
- 结构化执行解释器实现 CLEVRER DSL、GQA、ComPhy 等领域的查询。

**📊 数据集**

使用的数据集有：
- CLEVRER（视频因果推理基准）全验证集（约 20k 视频，305k 问题）；
- ComPhy（隐藏物理属性推理）；
- GQA（基于真实图像的视觉推理）；
- bAbI（文本推理）；
- twin‑EventLog（500 条干预规范，用于评估记忆一致性）。

**📈 对比分析**

与基准的比较方法：在 CLEVRER 上与最强的符号基线 NS‑DR 以及参数化注意力基线 ALOE 进行对比；在 ComPhy、GQA、bAbI 上与现有最佳方法对比；在 twin‑EventLog 上与 Llama‑3.1‑8B 及 Concordia‑style LLM 进行对比。性能结果显示：
- 在 CLEVRER 的描述、解释子集上超越 NS‑DR，且在描述与解释子集上略优于 ALOE；
- 在预测与反事实子集上落后于 ALOE；
- 在 ComPhy 上比 PCR 低约 11 %；
- 在 GQA 上在图谱推理任务中达 95.27 %；
- 在 twin‑EventLog 上在联合准确率上比 Llama‑3.1‑8B 提升 18.8 % ，比 Concordia‑style 提升 65 %。

**⚠️ 局限性**

局限性：
1) 未实现隐藏属性（质量、电荷等）的推理，导致 ComPhy 相关问题表现受限；
2) 仅使用启发式的共被除去伙伴（common‑removed‑partner）预测新出现的碰撞，无法覆盖更复杂的多物体演化；
3) 需要手工编写 TBox，未提供自动化 TBox 学习方法；
4) 只关注状态表示与反事实推理，未涉及动作选择或自然语言生成。

---

## 404. A Reproducible and Physically Feasible Dynamic Parameter Identification Framework for a Low-Cost Robot Arm

**arXiv ID:** 2605.15949 | [PDF](https://arxiv.org/pdf/2605.15949v1)

**作者:** Junji Oaki `[一作]` (University of Tsukuba), Sho Sakaino `[通讯]` (University of Tsukuba)

**通讯引用:** 1689 | [OpenAlex ID](https://openalex.org/A5065050034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对低成本机器人臂CRANE-X7，提出了一套可复现且物理可行的动态参数识别框架；

**💡 创新点**

创新点在于结合OLS、条件SDP投影、闭环输入误差CLIE三阶段递进式识别，并通过全姿态正定性检查与局部SDP救援，确保识别出的参数在更广阔的姿态域内保持物理可行性；

**🔧 技术方法**

使用了OpenSYMORO生成的符号逆动力学回归矩阵、最小二乘法、半正定规划（SDP）、闭环输入误差最小化（CLIE）以及主成分分析（PCA）进行统计中心化筛选；

**📊 数据集**

数据集为40条手工设计的结构化单关节和相邻双关节激励轨迹（采样间隔分别为10、20、40、80 ms），以及3条留作验证的复杂轨迹；

**📈 对比分析**

与传统仅使用OLS或仅靠闭环误差校正的方法相比，参数云从OLS到SDP再到CLIE逐步收缩，最终通过全姿态正定性检验得到的模型在未见过的验证轨迹上保持低均方根误差和均值归一化误差，证明了方法的有效性；

**⚠️ 局限性**

局限在于模型仅保留粘性摩擦，未对高频背隙等内部驱动器动态进行建模；此外，激励轨迹采用手工设计，缺乏基于信息准则的最优化，可能导致部分姿态下参数辨识仍不充分。

---

## 405. Constrained MPC-Based Motion Planning for Morphing Quadrotors in Ultra-Narrow Passages under Limited Perception

**arXiv ID:** 2605.15999 | [PDF](https://arxiv.org/pdf/2605.15999v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 406. Dynamic Plasma Shape Control with Arbitrary Sensor Subsets

**arXiv ID:** 2605.15935 | [PDF](https://arxiv.org/pdf/2605.15935v1)

**作者:** D. Sorokin `[一作]` (Next Step Fusion), D. Orlov `[通讯]` (University of California San Diego)

**通讯引用:** 3409 | [OpenAlex ID](https://openalex.org/A5012120152)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

训练并在真实托卡马克设备上部署了一个强化学习智能体，用于实时等离子体形状控制，能够在目标形状动态变化和诊断传感器失效的情况下保持稳定跟踪。

**💡 创新点**

创新点在于一次性解决了三大挑战：通过随机步进目标训练实现对未知形状轨迹的零样本泛化；使用诊断 dropout 使策略对任意传感器子集鲁棒；引入辅助形状重构头提供可解释性并提升训练稳定性。

**🔧 技术方法**

技术包括：Truncated Quantile Critics (TQC) 强化学习；非对称演员‑评论家架构（演员仅见到可用传感器，评论家拥有特权信息）；诊断 dropout；辅助形状重构损失；高保真NSFsim仿真器进行训练，并在GSevolve与DIII-D实验设备上验证。

**📊 数据集**

数据集为从2014–2020年DIII-D实验中筛选的120个低单点形状（LSN）目标，配合高保真仿真器以及GSevolve独立仿真器和真实设备实验数据。

**📈 对比分析**

通过与传统Isoflux控制器在GSevolve中的对比，评估了形状误差、奖励以及诊断失效时的鲁棒性。结果显示，在NSFsim与GSevolve中平均形状误差约为2–5 cm；在诊断 dropout 30% 的情况下误差保持在≈4 cm；相比Isoflux在某些轨迹上性能更优且不需要备份控制器。

**⚠️ 局限性**

局限性包括：仅在单一托卡马克（DIII-D）验证，缺乏跨机迁移能力；诊断缺失时采用均值替代，未处理实时诊断恢复；对极端形状的控制仍存在误差较大。

---

## 407. SLIP & ETHICS: Graduated Intervention for AI Emotional Companions

**arXiv ID:** 2605.15915 | [PDF](https://arxiv.org/pdf/2605.15915v1)

**作者:** Minseo Kim `[一作]` `[通讯]` (HUA Labs), Minseo Kim (HUA Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了SLIP与ETHICS两套框架，用以在AI情感伴侣中进行分级干预和情境信号标记，避免过度路径化。

**💡 创新点**

创新点在于“做不到路径化高能”原则、基于影响强度(a)、叙事活力(m)和自定义信号标签的分级干预逻辑，以及“signals not labels”的标签体系。

**🔧 技术方法**

采用规则引擎、预定义及AI生成的ETHICS标签、二级LLM进行语境分析、七日滑动窗口历史升降，结合多模型（GPT‑4o‑mini、GPT‑5‑mini、GPT‑5）进行压力测试。

**📊 数据集**

使用真实用户10周部署数据（68条日记，10名用户）以及由AI生成的5个虚拟人格共91条日记进行安全性压力测试。

**📈 对比分析**

通过部署与合成数据评估，整体无流失误判（0%）、软硬干预分布符合预期，危机人群可达50%硬干预；在三模型压力测试中，高能检测从0/8提升到6/8，同时流失误判降为0/10，表明模型能力提升可缓解检测缺口。

**⚠️ 局限性**

局限包括样本量小、仅在韩语日记环境下验证、对高能升高事件检测仍存在漏报、间接自杀表述难以捕捉、跨文化适用性未评估以及高模型成本与延迟。

---

## 408. Can Vision Language Models Be Adaptive in Mathematics Education? A Learner Model-based Rubric Study

**arXiv ID:** 2605.16011 | [PDF](https://arxiv.org/pdf/2605.16011v1)

**作者:** Jie Gao `[一作]` (McGill University), Jackie Chi Kit Cheung `[通讯]` (McGill University)

**通讯引用:** 2237 | [OpenAlex ID](https://openalex.org/A5050801868)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现基于学习者模型的评估规则，系统评估视觉语言模型在数学辅导任务中的自适应性。

**💡 创新点**

构建了完整的评估框架和九维自适应度量，首次将认知、动机、复杂性等维度与视觉语言模型结合评估。

**🔧 技术方法**

使用视觉语言模型（如 GPT‑5、GPT‑o1、Gemini‑2.5‑Flash、LLaVA、Qwen‑3‑30B‑VL）以及人工标注的评估流程。

**📊 数据集**

利用 TIMSS 2019/2023 公开数学题目（10 道，包含图形/视觉元素）和合成学习者画像。

**📈 对比分析**

通过专家标注的九项自适应指标、三项正确性与七项质量维度进行量化比较，结果显示 Gemini‑2.5‑Flash 取得最高自适应得分，LLaVA 质量最低，模型对学习者差异的适应性普遍不足。

**⚠️ 局限性**

评估维度不完整、数据量有限、采用合成学习者而非真实学生、未覆盖更深层的社会文化因素，且自适应评分未必等同于高质量适应性教学。

---

## 409. Segmentation, Detection and Explanation: A Unified Framework for CT Appearance Reasoning

**arXiv ID:** 2605.15997 | [PDF](https://arxiv.org/pdf/2605.15997v1)

**作者:** Yuyuan Liu `[一作]` (University of Oxford), J. Alison Noble `[通讯]` (University of Oxford)

**通讯引用:** 22496 | [OpenAlex ID](https://openalex.org/A5077728082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一个统一的自回归框架，将CT分割、检测与语言化外观推理结合。

**💡 创新点**

创新点包括任务路由令牌实现分割/检测的可调用性，以及“更近一步”机制进行局部细化。

**🔧 技术方法**

使用大规模视觉语言模型（Janus‑1B、Qwen2.5‑VL‑3B）、SAM视觉编码器、LoRA微调与任务嵌入。

**📊 数据集**

构建BTCV++多模态数据集，并在BTCV和MosMed+公开数据集上评估。

**📈 对比分析**

与现有SOTA方法相比，在BTCV++上Dice提升约1%，MosMed+上提升约1.7%，同时提供可解释文本输出。

**⚠️ 局限性**

缺点是自回归设计导致推理速度和算力成本上升。

---

## 410. Constrained latent state modeling: A unifying perspective on representation learning under competing constraints

**arXiv ID:** 2605.15995 | [PDF](https://arxiv.org/pdf/2605.15995v1)

**作者:** Gwenolé Quellec `[一作]` (Inserm), Gwenolé Quellec `[通讯]` (Inserm)

**通讯引用:** 6997 | [OpenAlex ID](https://openalex.org/A5028479581)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了“受限潜在状态建模”（CLSM）这一统一视角，系统梳理并量化潜在状态学习中的六大核心约束（预测充分性、最小性、时间连贯性、观测兼容性、对无关因素的不变性以及结构约束），并将这些约束映射到现有模型族，形成一个多目标设计空间。

**💡 创新点**

核心创新在于：①将潜在状态学习从单一目标转化为显式约束的多目标优化；②阐释并量化不同方法在六大约束上的占据位置；③通过 CLSM 解释识别性问题、设计权衡与模型可解释性之间的关系，提出了可持续的框架用于指导未来模型设计与评估。

**🔧 技术方法**

采用理论分析、信息论框架与图解方法；提出软约束与多目标损失的形式化表示；对比并重述多类模型（状态空间模型、潜在变量模型、重建式自编码器、预测式学习、多模态学习、领域特定结构模型）的约束实现。

**📊 数据集**

未使用具体数据集；文章以概念性与综述性为主，主要引用公开的相关文献和实验结果。

**📈 对比分析**

本文未给出新的实验结果；通过表格与图示对比不同方法在六约束上的满足程度，未对性能指标做量化比较；强调现有方法在某些约束上缺失或仅弱满足，说明需通过 CLSM 明确平衡。

**⚠️ 局限性**

限制包括：①缺乏统一的实现或评估基准，仍需实验验证；②多目标优化的权重选择与 Pareto 分析仍为挑战；③在高维、多模态数据下的可扩展性和计算成本待解决；④识别性问题未彻底解决，仅将其视为约束不足的后果。

---

## 411. Beyond Content: A Comprehensive Speech Toxicity Dataset and Detection Framework Incorporating Paralinguistic Cues

**arXiv ID:** 2605.15984 | [PDF](https://arxiv.org/pdf/2605.15984v1)

**作者:** Zhongjie Ba `[一作]` (Zhejiang University), Li Lu `[通讯]` (Zhejiang University)

**通讯引用:** 11460 | [OpenAlex ID](https://openalex.org/A5100386299)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了大规模音频毒性言论数据集ToxiAlert-Bench，并开发了双头神经网络ToxiAlert，用于同时识别毒性来源（文本/声学）和毒性类别。

**💡 创新点**

创新点在于为毒性检测引入来源级别注释（区分文本与声学毒性），以及采用双头架构与多阶段训练策略以降低任务间干扰并显著提升性能。

**🔧 技术方法**

使用了预训练的Wav2Vec2.0自监督模型作为特征编码器，配合双分类头、类平衡采样、加权交叉熵损失及多阶段训练流程。

**📊 数据集**

使用的数据集为ToxiAlert-Bench，包含32,561条音频（约60.82小时），覆盖7大毒性类别、20细标签，并标注毒性来源。

**📈 对比分析**

在同一数据集上与DeToxy、YIDUN、Qwen2-Audio、GPT-4o Audio、Gemini-2.5-Flash等基线进行对比，ToxiAlert在宏F1上提升21.1%，准确率提升13%，并在纯声学毒性子集上达到90%以上的准确率。

**⚠️ 局限性**

限制主要体现在数据规模与多语言泛化能力不足、对极端低资源毒性类别的鲁棒性有限，以及对预训练模型在特定任务适配的依赖。

---

## 412. A Causally Grounded Taxonomy for Image Degradation Robustness Evaluation

**arXiv ID:** 2605.15906 | [PDF](https://arxiv.org/pdf/2605.15906v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 413. Flash-GRPO: Efficient Alignment for Video Diffusion via One-Step Policy Optimization

**arXiv ID:** 2605.15980 | [PDF](https://arxiv.org/pdf/2605.15980v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. Entropy-Based Characterisation of the Polarised Regime in Latent Variable Models

**arXiv ID:** 2605.15965 | [PDF](https://arxiv.org/pdf/2605.15965v1)

**作者:** Peter Clapham `[一作]`, Marek Grzes `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于均值表示熵的无先验依赖的极化模式判定方法，并在多种变分与确定性自编码器中验证其有效性

**💡 创新点**

将熵与KL、方差等信息量指标通过熵-方差不等式联系，构建一个普适的极化判定准则；同时发现被认为“无信息”的被动维度在归一化后仍可提供微小预测增益

**🔧 技术方法**

信息论度量（熵、互信息）、矩阵式Renyi熵估计、β-VAE、iVAE、Least‑Volume AE、L2正则AE等模型

**📊 数据集**

MNIST、smallNORB、d‑Sprites等标准图像数据集

**📈 对比分析**

通过与传统基于KL阈值和Bonheur的判定对比，熵判定在多模型、多正则化强度下均能稳定识别活跃与被动维度；在下游逻辑回归任务中，归一化后的被动维度可略微提升性能，但整体增益有限

**⚠️ 局限性**

需要显式极化分布才能可靠分类；混合维度难以仅靠均值熵区分；阈值 τ 选取仍是经验性；实验仅覆盖图像卷积自编码器，未验证到序列或扩散模型等场景

---

## 415. Privacy is Fungibility: Why Endogenous Tokens Are Not Money

**arXiv ID:** 2605.15934 | [PDF](https://arxiv.org/pdf/2605.15934v1)

**作者:** Alex Lynham `[一作]` (University College London), Geoffrey Goodell `[通讯]` (University College London)

**通讯引用:** 282 | [OpenAlex ID](https://openalex.org/A5015121564)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对区块链生态中的本地代币与传统货币进行理论区分，提出它们更像信用而非现金。

**💡 创新点**

首次将Kahn等人的隐私-货币模型扩展到公有链，并建立基于信任与安全源的代币分类矩阵。

**🔧 技术方法**

采用经济学模型、隐私理论和账本安全分析等理论技术。

**📊 数据集**

未使用具体数据集，主要基于已有文献与区块链案例进行理论推导。

**📈 对比分析**

未进行实验比较，理论上讨论不同代币模型下的盗窃激励与系统崩溃风险。

**⚠️ 局限性**

缺乏经验验证；对账本安全性假设过于理想；未考虑多链跨链稳定币的细节。

---

## 416. Synchronized Realities: Towards Magic Mobile Experiences through Aligned AR

**arXiv ID:** 2605.15924 | [PDF](https://arxiv.org/pdf/2605.15924v1)

**作者:** Jan Henry Belz `[一作]` `[通讯]` (Dr. Ing. h.c. F. Porsche AG), Jan Henry Belz (Dr. Ing. h.c. F. Porsche AG)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在增强现实环境中探讨并示例化通过同步数字与物理现实实现更沉浸式移动体验的方法，分析其关键要素与实现路径。

**💡 创新点**

提出“同步现实”概念，系统化分析同步要素（Anchor、Precision、Trigger、Interaction type、Stakes）并结合生成式 AI 与上下文感知实现多模态同步。

**🔧 技术方法**

利用生成式 AI（文本、图像/视频生成）、多模态感知模型（自然语言与视觉描述）、上下文数据融合、实时定位与感知技术。

**📊 数据集**

未使用公开数据集，而是基于实验原型中的传感器数据与用户环境感知（如步态、位置、视觉景象）。

**📈 对比分析**

未进行量化对比，文章主要以案例与实验原型为依据，讨论同步精度、触发机制与交互方式，未给出性能数值。

**⚠️ 局限性**

局限包括缺乏大规模用户评估、同步失效导致用户分心与隐私风险、移动设备计算与感知负荷、方法泛化性不足。

---

## 417. Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning

**arXiv ID:** 2605.15975 | [PDF](https://arxiv.org/pdf/2605.15975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 418. Ontology for Policing: Conceptual Knowledge Learning for Semantic Understanding and Reasoning in Law Enforcement Reports

**arXiv ID:** 2605.15978 | [PDF](https://arxiv.org/pdf/2605.15978v1)

**作者:** Anita Srbinovska `[一作]` (Rochester Police Department), Ernest Fokoué `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 740 | [OpenAlex ID](https://openalex.org/A5070827124)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个符号化的自然语言理解管道，利用 AMR、PropBank、VerbNet、WordNet 和自定义本体，自动从匿名化警察报告中提取事件、参与者、角色和时间顺序，并保持可追溯性。

**💡 创新点**

创新点在于将多级语义资源（PropBank→VerbNet→WordNet）与 OWL 本体和推理结合，提出置信度计算公式和基于规则的时间推理，使得在无标注数据的情况下仍能生成高质量、可验证的证据链接事实。

**🔧 技术方法**

使用了 AMR 语义解析、语义角色标注、词义映射（SemLink）、WordNet 上的层次检查、OWL 本体推理（HermiT）以及自定义的规则引擎和置信度评分机制。

**📊 数据集**

采用了 450 篇罗切斯特市警察局（RPD）财产犯罪报告（含入室盗窃、偷盗、机动车盗窃、抢劫、被盗财产）作为实验数据集，文本已通过 OCR 并进行 PII 隐写。

**📈 对比分析**

在语料层面评估指标显示，高置信度事件占 54.1%，PropBank→VerbNet→WordNet 路径覆盖率 93.7%；人类评审覆盖 5 篇案例，系统与人工多数意见的匹配率在 80–100% 范围；时间顺序推理以规则为主，准确率约 80%。

**⚠️ 局限性**

局限包括缺乏金标准评测、实体链接不够精准导致同一人物/物体跨文档识别困难、强行进场（Forced Entry）等细粒度事件在判别上表现不佳，以及整体流程对 OCR 和隐写错误较为敏感。

---

## 419. From Failure to Feedback: Group Revision Unlocks Hard Cases in Object-Level Grounding

**arXiv ID:** 2605.15951 | [PDF](https://arxiv.org/pdf/2605.15951v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 420. GEMS -- Guided Evolutionary Molecule Design for Sustainable Chemicals

**arXiv ID:** 2605.15932 | [PDF](https://arxiv.org/pdf/2605.15932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 421. AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression

**arXiv ID:** 2605.15921 | [PDF](https://arxiv.org/pdf/2605.15921v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 422. Decomposed Vision-Language Alignment for Fine-Grained Open-Vocabulary Segmentation

**arXiv ID:** 2605.15942 | [PDF](https://arxiv.org/pdf/2605.15942v1)

**作者:** Chenhao Wang `[一作]` (Aerospace Information Research Institute, Chinese Academy of Sciences), Yao Zhu `[通讯]` (Zhejiang University)

**通讯引用:** 5992 | [OpenAlex ID](https://openalex.org/A5034221181)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将文本提示拆分为类别和属性两部分的可拆分视觉语言对齐框架，利用特征门控交叉注意力和对数空间AND聚合实现细粒度开词表分割。

**💡 创新点**

创新点在于：①在特征层面对属性进行门控实现逻辑AND约束；②在评分层面使用对数空间聚合避免概率衰减；③将拆分提示与现有SAM3结构无缝融合，提升组合泛化能力。

**🔧 技术方法**

使用了Transformer式视觉语言预训练（CLIP/ALIGN），跨模态注意力、特征门控交叉注意力、对数空间AND聚合、SAM3分割头以及多属性提示编码策略。

**📊 数据集**

主要在两个数据集上评估：UBC建筑数据集（屋顶类型与用途两属性）和PACO-LVIS（颜色、材质、图案/标记、透明度四属性），采用统一的组合泛化评估协议。

**📈 对比分析**

与OVSeg、FC‑CLIP、CAT‑Seg、X‑Decoder、SAM3等基线在Seen/Unseen组合上对比，提出方法在Unseen AP和AND‑Eff上分别提升约3–4点和0.4–0.5，显著优于现有技术。

**⚠️ 局限性**

局限性包括：对属性间关系仍是独立门控，无法捕捉更复杂的属性互依；对属性数量增多时门控计算成本提升；在极少量样本或多属性组合稀疏的场景下仍存在性能下降。

---

## 423. A Retrieval-Enhanced Transformer for Multi-Step Port-of-Call Sequence Prediction in Global Liner Shipping

**arXiv ID:** 2605.15937 | [PDF](https://arxiv.org/pdf/2605.15937v1)

**作者:** Yanzhao Su `[一作]`, Yineng Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 163 | [OpenAlex ID](https://openalex.org/A5046918779)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于检索增强的Transformer框架CCRE，用于全球线性航运的多步港口调用序列预测。

**💡 创新点**

创新点在于将检索增强与可达性约束相结合，使用双指标检索（Jaccard+PMR）获取历史相似航线，并通过中层交叉注意力实现从实时航迹到历史背景的动态转移，保证物理可行性并提高长尾路由的准确性。

**🔧 技术方法**

采用Transformer编码器对AIS时空特征进行处理，检索增强历史编码器通过双指标检索生成候选航线，交叉注意力进行特征融合，最终使用自回归Transformer解码器结合可达性掩码、Scheduled Sampling和Gumbel‑Softmax实现端到端训练。

**📊 数据集**

使用基于2021年AIS轨迹的全球集装箱船数据（400艘船，29,097条航段）和全球港口地理围栏数据库，共计602个港口与3,867条边，构建了全局航线网络。

**📈 对比分析**

与CatBoost、XGBoost、RF、LSTM、GRU等基线相比，CCRE在首站预测准确率达到72.3%，三步平均准确率61.4%，比CatBoost和LSTM平均提升约12.6%和11.3%，并在多步序列一致性（SeqAcc）上显著优于其他模型。

**⚠️ 局限性**

局限性包括对训练期间航线不变性的假设，无法很好处理时间演变或完全随机的航线；检索过程依赖预构建的历史库，实时更新成本较高；对极端稀缺港口仍存在预测准确率下降的问题。

---

## 424. Generative Long-term User Interest Modeling for Click-Through Rate Prediction

**arXiv ID:** 2605.15905 | [PDF](https://arxiv.org/pdf/2605.15905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 425. Invaria: Learning Scale and Density Invariance in Point Clouds via Next-Resolution Prediction

**arXiv ID:** 2605.15923 | [PDF](https://arxiv.org/pdf/2605.15923v1)

**作者:** Chun-Peng Chang `[一作]` (TU Delft), Holger Caesar `[通讯]` (TU Delft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种名为Invaria的3D点云编码器，目标是在不同采样分辨率和尺度下保持语义特征的不变性；

**💡 创新点**

创新点在于结合下一分辨率预测训练目标与自适应感受野校准，从而抑制模型对点云密度和尺度的“shortcut”，并通过非对称编码策略显著降低输入token数量；

**🔧 技术方法**

主要技术包括：下一分辨率预测（Next‑Resolution Prediction）、感受野校准（Receptive Field Calibration）、对齐损失（Alignment Loss）、U‑Net 结构与稀疏卷积、以及相对位置编码的自适应调整；

**📊 数据集**

使用ScanNet数据集进行室内语义分割实验，并对比了多种主流方法；

**📈 对比分析**

与现有最先进模型相比，Invaria在三倍降低分辨率时实现了56.0%更高的mIoU，在尺度减小三倍时提升20%，且模型参数量减少45%，平均输入token减少40%；

**⚠️ 局限性**

局限性包括：对简单平面物体（如墙壁、图片）的区分仍不理想，类别不平衡问题仍待解决；在更高分辨率输入时可能出现精度下降；未在室外或更大规模数据集上验证，且对极端噪声和非均匀采样的鲁棒性需进一步评估。

---

## 426. Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation

**arXiv ID:** 2605.15913 | [PDF](https://arxiv.org/pdf/2605.15913v1)

**作者:** Shuaiyi Li `[一作]` (Chinese University of Hong Kong), Wai Lam `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 7565 | [OpenAlex ID](https://openalex.org/A5018582154)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

暂无可用信息

**💡 创新点**

暂无可用信息

**🔧 技术方法**

暂无可用信息

**📊 数据集**

暂无可用信息

**📈 对比分析**

暂无可用信息

**⚠️ 局限性**

暂无可用信息

---

## 427. Imperfect World Models are Exploitable

**arXiv ID:** 2605.15960 | [PDF](https://arxiv.org/pdf/2605.15960v1)

**作者:** Logan Mondal Bhamidipaty `[一作]` (University of Edinburgh), Subramanian Ramamoorthy `[通讯]` (University of Edinburgh)

**通讯引用:** 2229 | [OpenAlex ID](https://openalex.org/A5071122608)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

本文提出了强化学习中“模型利用”（model exploitation）的正式定义，并证明在包含开放子集的大规模策略集合上，任何非平凡且不等价的模型对都不可避免地导致价值反转；

**💡 创新点**

创新点包括：①引入ε-模型利用的松弛概念；②给出了安全规划的有效期限（safe horizon）的闭式上界；③建立了奖励黑客与模型利用之间的统一几何与理论框架；

**🔧 技术方法**

主要技术手段是价值函数的有理性与解析性证明、梯度分析（局部与全局逆转判定）、连通性与等价性定理、模拟引理与总变距离估计；

**📊 数据集**

本工作为理论性研究，未使用具体数据集；

**📈 对比分析**

未进行实验或与其他方法的性能对比；

**⚠️ 局限性**

主要限制包括：仅适用于有限状态与动作空间的MDP；模型利用定义为二元，缺少细粒度度量；未给出有限策略集下的充分必要条件。

---

## 428. PCDM: A Diffusion-Based Data Poisoning Attack Against Federated Learning Systems

**arXiv ID:** 2605.16098 | [PDF](https://arxiv.org/pdf/2605.16098v1)

**作者:** Wei Sun `[一作]` (Beijing Jiaotong University), Khaled Ben Letaief `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 46445 | [OpenAlex ID](https://openalex.org/A5079052203)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于扩散模型的Poisoning-Oriented Conditional Diffusion Model (PCDM) 用于对联邦学习系统进行无目标数据中毒攻击。

**💡 创新点**

创新点包括：①在扩散模型中引入可调的“poisoning vector”实现精细化的中毒控制；②采用跳跃扩散策略（jumping diffusion）显著减少采样步数，提升效率；③提供理论分析证明该方法在攻击效果与隐蔽性之间的可调平衡。

**🔧 技术方法**

使用扩散模型（DDPM）结合条件输入与噪声向量，配合自适应噪声调度与梯度修正，实现高质量、可控的中毒数据生成。

**📊 数据集**

实验数据集包括经典图像分类集 MNIST、Fashion‑MNIST、CIFAR‑10、CIFAR‑100 以及真实无线场景的车辆识别数据集 VRAI。

**📈 对比分析**

通过与七种基线中毒方法（Label Flipping、PoisonGAN、Noise Superimposition、VagueGAN 等）以及十一种防御/鲁棒聚合方法（PCA、UMAP、CONTRA、DnC、K‑Means、FedDMC、LoMar、MCD、Multi‑Krum、SignGuard、LASA）进行对比；结果显示 PCDM 在保持低检测率的前提下，能更大幅度降低全局模型准确率，且在 IID 与非 IID 情况下均表现出最优的攻击效果与隐蔽性。

**⚠️ 局限性**

局限性：①需要在受控的本地设备上训练扩散模型，对资源极度受限的边缘设备仍存在一定挑战；②攻击效果受噪声向量与跳跃步数调参的影响，若参数选择不当可能导致隐蔽性下降或攻击效果减弱；③评估主要集中在图像分类任务，对非图像或更高维度数据的适用性尚未充分验证。

---

## 429. A Cross-Modal Prompt Injection Attack against Large Vision-Language Models with Image-Only Perturbation

**arXiv ID:** 2605.16090 | [PDF](https://arxiv.org/pdf/2605.16090v1)

**作者:** Hao Yang `[一作]` (Xidian University), JianFeng Ma `[通讯]` (Xidian University)

**通讯引用:** 21656 | [OpenAlex ID](https://openalex.org/A5012016098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对大规模视觉-语言模型，提出仅通过对图像进行微小扰动，即可诱导模型忽略原始文本指令，执行攻击者指定任务的跨模态提示注入攻击；

**💡 创新点**

创新点在于将攻击目标从传统的视觉嵌入空间转移到模型隐藏状态空间，聚焦于融合阶段的中层，利用距离递减的扰动预算分配实现对语义关键区域的精准调控；

**🔧 技术方法**

采用层级感知的融合层选择、Grad‑ECLIP语义重要性估计、频域正则化、输出层与融合层双重对齐损失以及多视角数据增强等技术；

**📊 数据集**

在MSCOCO、ImageNet和TextVQA三个公开数据集上构造攻击样本进行评估；

**📈 对比分析**

与ARE‑W、ARE‑B、CI、ATPI等四种基线方法对比，平均攻击成功率提升至66.36%，比基线高约40.9个百分点；在语义相似度和视觉不可见性指标上亦表现出色；

**⚠️ 局限性**

局限性包括对特定文本提示敏感，缺乏对任意提示的泛化能力，以及在面对专门的输入变换或高频抑制防御时仍有一定脆弱性。

---

## 430. Towards Trustworthy and Explainable AI for Perception Models: From Concept to Prototype Vehicle Deployment

**arXiv ID:** 2605.16087 | [PDF](https://arxiv.org/pdf/2605.16087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 431. Robust Prior-Guided Segmentation for Editable 3D Gaussian Splatting

**arXiv ID:** 2605.16065 | [PDF](https://arxiv.org/pdf/2605.16065v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 432. Ada-Diffuser: Latent-Aware Adaptive Diffusion for Decision-Making

**arXiv ID:** 2605.16054 | [PDF](https://arxiv.org/pdf/2605.16054v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 433. Learning Sim-Grounded Policies for Bimanual Rope Manipulation from Human Teleoperation Data

**arXiv ID:** 2605.16043 | [PDF](https://arxiv.org/pdf/2605.16043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 434. Judge Circuits

**arXiv ID:** 2605.16023 | [PDF](https://arxiv.org/pdf/2605.16023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 435. ShopGym: An Integrated Framework for Realistic Simulation and Scalable Benchmarking of E-Commerce Web Agents

**arXiv ID:** 2605.16116 | [PDF](https://arxiv.org/pdf/2605.16116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 436. Scalable neuromorphic computing from autonomous spiking dynamics in a clockless reconfigurable chip

**arXiv ID:** 2605.16114 | [PDF](https://arxiv.org/pdf/2605.16114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 437. Attention Dispersion in Dynamic Graph Transformers: Diagnosis and a Transferable Fix

**arXiv ID:** 2605.16112 | [PDF](https://arxiv.org/pdf/2605.16112v1)

**作者:** Jinhao Zhang `[一作]` (Beijing Institute of Technology), Long-Kai Huang `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 769 | [OpenAlex ID](https://openalex.org/A5003953967)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

识别并解决连续时间动态图Transformer在时间分布偏移下注意力分散导致性能下降的问题，并提出将标准注意力替换为差分注意力的方法。

**💡 创新点**

证明注意力分散是CTDG Transformer在时间偏移下的共享瓶颈，并用差分注意力实现跨架构的性能提升，最终构建了DiffDyG实现SOTA。

**🔧 技术方法**

主要技术包括差分注意力（Differential Attention）、RoPE位置编码、邻居共现编码、空间距离编码以及Transformer框架，并对注意力进行统计分析。

**📊 数据集**

在9个CTDG基准数据集（Wikipedia、Reddit、UCI、Enron、Mooc、CanParl、USLegis、UNTrade、UNVote）上评估。

**📈 对比分析**

与多种基线（记忆式、随机游走式、时序消息传递Transformer、序列模型、结构编码模型）进行比较，DiffDyG在大多数数据集上取得最高平均精度，尤其在高时间偏移数据集上提升显著（如从78.69%提升到87.52%）。

**⚠️ 局限性**

主要限制包括实验为经验性分析，缺乏理论证明；关键节点定义为启发式；仅评估动态链接预测，未验证在其他任务或大规模工业图上的泛化。

---

## 438. Centralized vs Decentralized Federated Learning: A trade-off performance analysis

**arXiv ID:** 2605.16089 | [PDF](https://arxiv.org/pdf/2605.16089v1)

**作者:** Chaimaa Medjadji `[一作]` (University of Luxembourg), Yves Le Traon `[通讯]` (University of Luxembourg)

**通讯引用:** 17324 | [OpenAlex ID](https://openalex.org/A5040574362)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对中央化、去中心化和半去中心化联邦学习架构进行实验比较，使用FedStellar模拟器、MNIST数据集和MLP分类器；

**💡 创新点**

通过实验揭示三种架构在准确率、收敛时间、通信成本和CPU使用率等指标上的权衡，填补了先前缺乏对三者性能对比的空白；

**🔧 技术方法**

使用FedStellar模拟器进行联邦学习实验，采用FedAvg聚合，并评估准确率、召回率、损失、字节交换量和CPU占用等指标；

**📊 数据集**

MNIST手写数字数据集（60,000个训练样本），按参与者划分；

**📈 对比分析**

通过改变参与者数量（3、4、6、8）进行多场景实验，比较每种架构的准确率、收敛时间、通信开销等，结果显示DFL最高准确率、SDFL接近且收敛更快、CFL最快收敛但准确率最低；

**⚠️ 局限性**

仅评估了有限的七项KPI，未覆盖所有模型、网络与节点层面的指标；实验规模受计算资源限制，未验证更大规模或不同数据集的通用性；需扩展FedStellar以支持更多KPI。

---

## 439. Towards Foundation Models for Relational Databases with Language Models and Graph Neural Networks

**arXiv ID:** 2605.16085 | [PDF](https://arxiv.org/pdf/2605.16085v1)

**作者:** Jingcheng Wu `[一作]` (University of Stuttgart), Steffen Staab `[通讯]` (University of Stuttgart)

**通讯引用:** 27852 | [OpenAlex ID](https://openalex.org/A5062807811)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种轻量级的混合LM-GNN框架，将细调的BART编码器与GraphSAGE GNN相结合，完成对多表关系数据库的行语义与结构语义融合；

**💡 创新点**

创新点在于首次将BART行级语义提取与REG图神经网络消息传递结合，并通过自监督预训练实现跨数据库迁移；

**🔧 技术方法**

使用BART进行行编码，GraphSAGE构建REG图进行关系上下文注入，并采用schema-aware遮掩和特征重构自监督目标；

**📊 数据集**

在RelBench基准上预训练6个数据库（共约28.7M节点），随后在held-out的rel-f1数据库进行节点分类下游任务；

**📈 对比分析**

与LightGBM、RDL以及基础模型Griffin、KumoRFM对比，冻结版获得61.40 ROC‑AUC，微调版达到67.40，略低于LightGBM（68.86）但仅差5.22点；

**⚠️ 局限性**

局限包括预训练规模小、仅评估单一下游任务、两阶段训练阻碍跨模态联合学习、缺乏多任务和大规模实验验证等。

---

## 440. AgriMind: An Ensemble Deep Learning Framework for Multi-Class Plant Disease Classification

**arXiv ID:** 2605.16076 | [PDF](https://arxiv.org/pdf/2605.16076v1)

**作者:** Salma Hoque Talukdar Koli `[一作]` (RTM Al-Kabir Technical University), Fahima Haque Talukder Jely `[通讯]` (North East University Bangladesh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在农业精准管理场景下，构建并评估了AgriMind——一个融合ResNet50、EfficientNet‑B0与DenseNet121的软投票集成模型，用于对15类辣椒、土豆和番茄叶病进行多分类识别。

**💡 创新点**

创新点在于系统性比较三种主流CNN结构的集成效果，提供完整拆解实验并证明等权重平均即可获得最佳性能；同时展示模型在NVIDIA T4 GPU上可实现53FPS的实时推理速度。

**🔧 技术方法**

使用迁移学习（冻结ImageNet预训练权重，仅训练分类头）、软投票集成、PyTorch框架、Adam优化器、交叉熵损失、固定随机种子实现可复现实验。

**📊 数据集**

采用PlantVillage数据集的20,638张RGB叶片图像，筛选出15类（包括辣椒、土豆、番茄及其健康样本），按70%/15%/15%比例划分训练/验证/测试。

**📈 对比分析**

与单模型（ResNet50 97.42%、EfficientNet‑B0 96.48%、DenseNet121 97.00%）以及两模型组合对比，软投票集成达到99.23%测试精度，比最佳单模型提升1.81个百分点，错误率降低约两倍；在NVIDIA T4 GPU上实现53FPS推理速度。

**⚠️ 局限性**

局限包括：仅冻结backbone，缺乏领域特定微调；未在真实田间条件或移动端设备上验证；实验仅跑一次，缺乏方差估计；PlantVillage图像光照受控，实地部署需进一步测试。

---

## 441. ITGPT: Generative Pretraining on Irregular Timeseries

**arXiv ID:** 2605.16069 | [PDF](https://arxiv.org/pdf/2605.16069v1)

**作者:** Antoine Honoré `[一作]` (KTH Royal Institute of Technology), Ming Xiao `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 15285 | [OpenAlex ID](https://openalex.org/A5037292846)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了ITGPT，一种适用于多模态、间歇性采样时间序列的Transformer架构。

**💡 创新点**

创新点在于允许在间歇性采样数据上直接做自监督和GPT预训练，无需重采样或插值，且在低标签场景下显著提升性能。

**🔧 技术方法**

使用Transformer注意力、因果交叉注意力、ITNet基础、SSL（MSE）与GPT预训练、Dropout、MLP混合层等技术。

**📊 数据集**

使用医疗 TIHM 数据集和预测维护 CompX 数据集进行评估。

**📈 对比分析**

与传统线性插值+CE或仅CE方法比较，ITGPT 在 TIHM 上提升召回率至0.73，在 CompX 上取得最高AUPRC约0.44，特别是在标签稀缺时效果更佳。

**⚠️ 局限性**

局限包括对大规模数据的计算开销、对异常/缺失值的处理仍依赖于设计、需要更多解释性和高效实现。

---

## 442. Relational Database Data Lineage Ontology

**arXiv ID:** 2605.16068 | [PDF](https://arxiv.org/pdf/2605.16068v1)

**作者:** Jakub Dutkiewicz `[一作]` (Poznan University of Technology), Robert Wrembel `[通讯]` (Poznan University of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了RDDL Ontology，用于增强关系型数据库数据血缘建模，并通过知识图谱进行血缘关系发现

**💡 创新点**

在现有ontology基础上加入完整的数据库对象层次、完整约束、数据类型、执行语义，实现更丰富的语义表示，提升机器学习推理效果

**🔧 技术方法**

基于OWL的本体建模，转换为知识图谱，采用多路径Siamese GNN进行归纳链路预测

**📊 数据集**

使用Northwind数据库，构造730条变换场景（投影、线性、非线性、选择、连接、并集）

**📈 对比分析**

在基线ontology与RDDL两套KG上训练相同的GNN模型，评价Precision、Recall、AUC、Hits@10，RDDL平均提升约3% AUC、11% Hits@10

**⚠️ 局限性**

缺乏对单一本体组件的消融分析，实验范围局限于Northwind，未验证更复杂真实世界情境

---

## 443. Health-Conditioned Vision-Language-Action Models for Malfunction-Aware Robot Control

**arXiv ID:** 2605.16056 | [PDF](https://arxiv.org/pdf/2605.16056v1)

**作者:** Hüseyin Arslan `[一作]` (Hacettepe University), Özgür Erkent `[通讯]` (Hacettepe University)

**通讯引用:** 624 | [OpenAlex ID](https://openalex.org/A5003475208)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于健康向量的可视化语言动作模型，使机器人在关节失效时仍能完成任务

**💡 创新点**

通过零初始化的Health Projector将健康状态映射到模型潜在空间，参数极少但能显著提升对关节退化的适应性

**🔧 技术方法**

使用VLA-Adapter Pro为基础，加入两层MLP Health Projector与动作查询，并保持其它模块冻结，训练时仅微调900K参数

**📊 数据集**

在LIBERO Spatial任务套件上采集了128条失效演示与50条健康演示，共计178个样本，关节退化水平为0.3、0.5、0.7、0.9

**📈 对比分析**

与未加健康信息的预训练模型对比，健康条件模型在健康模式下保持97.5%成功率；在严重关节退化（如肩关节w=0.3）时成功率从45%提升至89%，整体提升显著，但部分健康关节的性能略有下降

**⚠️ 局限性**

目前仅针对单关节退化、仿真环境，缺乏多关节退化数据和隐式健康估计机制，未在真实机器人上验证

---

## 444. Variational Autoregressive Networks with probability priors

**arXiv ID:** 2605.16020 | [PDF](https://arxiv.org/pdf/2605.16020v1)

**作者:** Piotr Białas `[一作]` (Jagiellonian University), Dawid Zapolski `[通讯]` (Jagiellonian University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5099096883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

采用物理先验的条件概率作为起点，改进自回归神经生成器，在二维 Ising 模型和 Edwards-Anderson 随机磁石玻璃上训练，并评估其性能。

**💡 创新点**

创新点在于使用基于 tanhβ 的弱展开得到的多阶条件概率近似，将物理知识嵌入神经网络，显著降低训练难度并提升采样质量。

**🔧 技术方法**

采用自回归变分网络（VAN）与全连接神经网络，结合 REINFORCE 与重要性采样。

**📊 数据集**

使用 32×32 的 Ising 和 EA 网格（J=±1）作为实验数据集。

**📈 对比分析**

通过有效样本大小（ESS）、磁化强度、自由能估计 F 以及 w̅ 指标与传统 Monte‑Carlo（Wolff/parallel tempering）比较，发现先验模型在低温和临界区得到更高 ESS、较低自相关，且模式崩溃现象减轻。

**⚠️ 局限性**

限制包括仅测试简单全连接网络、近邻模型、tanhβ 系列展开在临界以上可能发散、缺乏对更大体系或其他物理约束的验证。

---

## 445. Multi-Level Contextual Token Relation Modeling for Machine-Generated Text Detection

**arXiv ID:** 2605.16107 | [PDF](https://arxiv.org/pdf/2605.16107v1)

**作者:** Chenwang Wu `[一作]` (Hong Kong Baptist University), Defu Lian `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8802 | [OpenAlex ID](https://openalex.org/A5085254654)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多层次上下文词关系建模框架，用于提升机器生成文本（MGT）检测的准确性。

**💡 创新点**

创新点在于：①统一视角重构多种基于统计量的检测方法；②发现并针对token级检测分数受生成过程随机性偏置的根本瓶颈；③分别设计局部Markov校准模块和全局规则支持推理模块，以捕捉局部相似性与全局稳定性两种上下文关系；④将两层信号融合成单一检测分数，兼顾精度与低计算成本。

**🔧 技术方法**

采用的技术包括：Markov随机场（MRF）+均值场近似实现局部校准；基于全局统计量的阈值分桶和逻辑规则推理实现全局关系建模；联合推理框架；在多种基线检测器（如 Likelihood、Log‑Rank、Entropy、DetectGPT、Fast‑DetectGPT、Binoculars、FourierGPT、AdaGPT、DetectLLM）上进行加速与性能提升。

**📊 数据集**

使用的数据集包括：Essay（学术作文）、Reuters（新闻标题）、TruthfulQA（问答对话）和 DetectRL（多领域混合文本、对抗/重写样本）。

**📈 对比分析**

与原始基线及先前的单层本地校准版本（-M）对比，实验在跨 LLM、跨领域、混合文本、重写与对抗攻击等多种真实场景下均显著提升：AUROC 最高可达 99.9%（相对基线提升 10–30%），TPR@FPR‑1% 在大多数 LLM 上提升至 70–80%，混合文本与重写攻击下的鲁棒性也有明显提升。

**⚠️ 局限性**

局限性：①只适用于可拆解为 token 级分数的基线检测器；②对训练样本量有限的全局规则推理存在稀疏性风险；③对抗攻击若能专门破坏全局稳定性模式，效果可能受限；④需调节阈值分桶、局部权重等超参，虽然影响不大但仍需经验选择。

---

## 446. Federated Imputation under Heterogeneous Feature Spaces

**arXiv ID:** 2605.16099 | [PDF](https://arxiv.org/pdf/2605.16099v1)

**作者:** Imane Hocine `[一作]` (University of Luxembourg), Yves Le Traon `[通讯]` (University of Luxembourg)

**通讯引用:** 17324 | [OpenAlex ID](https://openalex.org/A5040574362)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种在特征空间异构的联邦学习环境下的缺失值填补框架 FedHF-Impute。

**💡 创新点**

创新点在于将结构性特征缺失与普通缺失区分，并利用全局特征图与消息传递实现跨客户端的间接知识迁移。

**🔧 技术方法**

采用特征图构建、图神经网络（GNN）推理、FedAvg 联邦训练以及自监督块掩码损失。

**📊 数据集**

使用 AirQuality、PhysioNET 和 SECOM 三个表格数据集进行评估。

**📈 对比分析**

与中心化和其他联邦基线相比，FedHF-Impute 在 AirQuality、PhysioNET 上均优于所有联邦基线，且在 SECOM 上比最佳中心化方法提升约27%，性能显著。

**⚠️ 局限性**

局限在于实验仅在四个客户端随机 60% 特征可用的模拟异构场景，未验证大规模、真实机构异构场景和图构建的隐私问题。

---

## 447. An efficient multi-GPU implementation for the Discontinuous Galerkin ocean model SLIM

**arXiv ID:** 2605.16082 | [PDF](https://arxiv.org/pdf/2605.16082v1)

**作者:** Miguel De Le Court `[一作]` (UCLouvain), Jonathan Lambrechts `[通讯]` (UCLouvain)

**通讯引用:** 3080 | [OpenAlex ID](https://openalex.org/A5087893927)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

实现了面向GPU的三维Discontinuous Galerkin有限元（DG‑FE）海洋模型SLIM，并在单GPU与多GPU（至1024卡）环境下完成可扩展的高效实现；

**💡 创新点**

在DG‑FE框架下提出了结构化的内存布局（SoA + Hilbert重排、cell布局）、基于矩阵自由求解器的垂直算子、以及计算与通信重叠的多GPU域分解策略，显著提升了GPU利用率；

**🔧 技术方法**

采用CUDA/HIP统一抽象层，利用MPI实现域分解与halo交换；利用块化列求解、共享内存共享二维数据、矩阵自由求解（水平压梯度与垂直速度）、全局共享内存复制、以及自动化的时间插值与数据预取；

**📊 数据集**

真实场景使用大尺度Great Barrier Reef网格（约330万三角形、3.4千万棱柱体），数据来源为30 m与100 m海深图、BRAN2023海洋再分析、TPXO10潮汐、BARRA2-C2大气再分析、OpenStreetMap海岸线及Allen Coral Atlas珊瑚分布；

**📈 对比分析**

与CPU基线对比，单张A100相当于≈1500 CPU核心，4×A100相当于128核CPU提升≈50×；在Meluxina与LUMI系统上实现弱扩展效率>80%至1024GPU；性能指标为内存占用≈80%峰值带宽、算术占用≈60%峰值浮点吞吐，整体时间步平均≈30%峰值；

**⚠️ 局限性**

对小网格（<10^6 DG节点）和非均匀垂直层数（非2的幂）时GPU利用率下降；2D模式的频繁短核和halo交换导致强扩展受限；当前测试未进行科学验证，仅展示计算可行性。

---

## 448. ReAlign: Generalizable Image Forgery Detection via Reasoning-Aligned Representation

**arXiv ID:** 2605.16080 | [PDF](https://arxiv.org/pdf/2605.16080v1)

**作者:** Qing Huang `[一作]` (Peking University), Jian Zhang `[通讯]` (Peking University)

**通讯引用:** 54633 | [OpenAlex ID](https://openalex.org/A5100410082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种轻量化的AI生成图像检测框架ReAlign，将RL优化的LLM（AIGI‑R1）生成的高质量推理文本通过对比学习与视觉特征对齐，从而得到既具语义敏感性又高效的检测模型。

**💡 创新点**

创新点包括：①利用GRPO强化学习获得的推理文本作为跨域、语义误差敏感的表示空间；②在CLIP基础上冻结文本编码器、使用LoRA微调视觉编码器，构建可同时完成对齐与分类的联合目标；③通过对比学习将文本与图像嵌入对齐，提升跨域泛化与对低频/高频伪造痕迹的辨识能力。

**🔧 技术方法**

核心技术包括：
- Group Relative Policy Optimization (GRPO) 对LLM进行强化学习优化；
- 生成推理文本并提取 <think> 标签中的语义；
- 对比学习（对齐损失）+ 二分类交叉熵损失的联合优化；
- 轻量化 CLIP + LoRA 微调；
- 数据集构建与文本-图像对齐训练流程。

**📊 数据集**

主要使用的数据集有：AIGCDetectBenchmark、AIGI‑Holmes、以及作者新构建的 UltraSynth‑10k（10k条真实/伪造样本，涵盖最新专有生成模型）。此外，为预训练对齐文本-图像对，作者利用AIGI‑R1在多轮问题/答案生成中产生大量推理文本。

**📈 对比分析**

在三大基准上与多种非LLM和LLM检测方法比较：
- 在 AIGCDetectBenchmark 上，ReAlign 平均准确率 96.14%，比 AIDE（92.77%）提升 3.37%，比 AIGI‑R1（91.77%）提升 4.37%；
- 在 AIGI‑Holmes 上，平均准确率 99.44%，明显优于 AIDE（97.00%）和其他方法；
- 在 UltraSynth‑10k 上，平均准确率 97.09%，在子集上往往超过 99%，显示出强大的跨域泛化能力。

**⚠️ 局限性**

局限性包括：
1) 依赖高质量的LLM推理文本，若LLM生成质量下降会影响对齐效果；
2) 对低级像素级伪造痕迹（如细微纹理或频域异常）的敏感度仍不如某些专门的频域方法；
3) 训练过程仍需要大规模算力与LLM推理的前期成本；
4) 目前仅针对静态图像，对视频、动态图像等多模态场景的扩展尚未验证；
5) 对未来极其高质量或新型生成模型的鲁棒性仍需进一步评估。

---

## 449. SAFE Quantum Machine Learning with Variational Quantum Classifiers

**arXiv ID:** 2605.16067 | [PDF](https://arxiv.org/pdf/2605.16067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 450. Misspecified Explore-then-Exploit Leads to Supra-Competitive Prices

**arXiv ID:** 2605.16064 | [PDF](https://arxiv.org/pdf/2605.16064v1)

**作者:** Jackie Baek `[一作]` (New York University), Farrell Wu `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了简单的算法定价系统在多企业市场中是否会系统地产生类似合谋的价格，并通过探测-利用管道进行定价，发现会出现高于纳什均衡的价格。

**💡 创新点**

首次将探索与利用相结合的定价策略与误差模型结合，通过流体极限ODE分析阐明相似价格探索范围时可能出现的合谋式定价，并在对称探索下可达到垄断价格。

**🔧 技术方法**

采用流体极限普通微分方程（ODE）分析、数值仿真以及非线性logit需求模型。

**📊 数据集**

使用真实多家庭租赁市场的数据进行仿真校准。

**📈 对比分析**

通过数值仿真与纳什均衡和垄断价格对比，发现算法在有限时段、异质产品和非线性logit需求等多种假设下均稳健地产生超竞争价格。

**⚠️ 局限性**

仅依赖单一误差的垄断式需求模型，忽略竞争者价格；流体极限分析的理论假设在所有实际场景中的适用性有限，且对市场结构的通用性受限。

---

## 451. Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness in Tax Law

**arXiv ID:** 2605.16052 | [PDF](https://arxiv.org/pdf/2605.16052v1)

**作者:** Parisa Kordjamshidi `[一作]` (Bloomberg), Enrico Santus `[通讯]` (Bloomberg)

**通讯引用:** 2089 | [OpenAlex ID](https://openalex.org/A5059288410)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估税法推理中LLM与神经-符号混合模型的效果，揭示数据污染导致的性能高估，并提出新的无污染测试集；

**💡 创新点**

1) 采用污染检测协议量化LLM的污染程度；2) 构建可生成正式注释的SARA^+测试集；3) 证明神经-符号框架在逻辑推理与泛化上更稳健；

**🔧 技术方法**

大规模语言模型（如GPT‑4o、Gemini‑3 Pro等）用于文本到Prolog的翻译；Prolog/SAT等符号求解器执行逻辑推理；对比直接QA与符号推理；

**📊 数据集**

SARA（税法规则与案例的结构化数据）及其扩展版SARA^+，包含规则变更、案例数值扰动与案例重述；

**📈 对比分析**

对比直接问答（Direct QA）与Prolog‑based推理，发现后者在数值推理上提升约70%且对规则扰动不敏感；在entailment任务上，顶级LLM（Gemini‑3 Pro）准确率约94%，但受污染影响显著；

**⚠️ 局限性**

1) 受限于数据集规模与人工校对成本；2) 仅评估税法领域，泛化到其他法律领域需进一步验证；3) 神经-符号流水线对翻译质量高度依赖，仍可能出现解析错误；

---

## 452. RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents

**arXiv ID:** 2605.16045 | [PDF](https://arxiv.org/pdf/2605.16045v1)

**作者:** Zijie Dai `[一作]` (Chinese University of Hong Kong), James Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 8266 | [OpenAlex ID](https://openalex.org/A5016082884)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种递归记忆系统RecMem，利用无意识层缓冲交互并仅在出现重复模式时才触发LLM抽象；

**💡 创新点**

创新点在于将记忆整合从“每一次交互都抽象”转变为“仅在重复出现时触发”，并引入语义细化机制提升抽象质量；

**🔧 技术方法**

核心技术包括：三层记忆架构（无意识、情节、语义），基于向量检索的递归触发阈值，LLM驱动的事件级叙事生成与语义细化；

**📊 数据集**

在LoCoMo（社交陪伴对话）和LongMemEval‑S（任务导向长文本对话）两个公开基准上评估；

**📈 对比分析**

与Full Context、Naive RAG、MemoryOS、Mem0、A‑Mem等三种SOTA基线比较，RecMem在构建阶段的token消耗下降87%+，在两大基准上获得最高或接近最高的整体问答准确率；

**⚠️ 局限性**

主要局限包括：依赖手工设定的相似度和递归阈值，可能忽视一次性但重要的信息，且对稀疏交互场景的适配仍需改进。

---

## 453. From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation

**arXiv ID:** 2605.16026 | [PDF](https://arxiv.org/pdf/2605.16026v1)

**作者:** Yu Pan `[一作]` (Kyushu University), Jianjun Zhao `[通讯]` (Kyushu University)

**通讯引用:** 6700 | [OpenAlex ID](https://openalex.org/A5065190767)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了S2ST-Omni 2多语音对语音翻译框架，改进了语言条件化方法，将平面语言标签转化为结构化的语言学典型先验；

**💡 创新点**

创新点在于三层结构化语言条件化：①典型层次化语言编码（TI‑HLE）把语言划分为形态、词序、谱系和残差四个通道；②动态门控语言感知 Dual‑CTC 对中间适配器特征进行内容自适应调制；③典型意识的LLM提示，将语言特性嵌入解码器；

**🔧 技术方法**

采用 Whisper 语音编码器 + Qwen3‑4B 语言模型、混合自适应适配器、FiLM 调制、动态帧门控、双 CTC 监督、LoRA 微调以及典型先验编码与提示策略；

**📊 数据集**

使用 CVSS‑C（CoVoST2 衍生）多语音到英语翻译数据集，包含约 561 小时法语、德语、西班牙语训练数据，以及约 3 小时的日语‑英语低资源子集；

**📈 对比分析**

与多种基线（Translatotron 2、UnitY、DASpeech、ComSpeech、StreamSpeech、SimulS2S‑LLM、Hibiki、RosettaSpeech、S2ST‑Omni）及 Whisper–Qwen S2TT 参考进行 BLEU、ASR‑BLEU、COMET、BLASER 2.0 评估；S2ST‑Omni 2 在所有指标上均优于 S2ST‑Omni，平均 BLEU 37.73（+5.8% 相对），ASR‑BLEU 35.00（+4.6%），COMET 83.31，BLASER 4.24；

**⚠️ 局限性**

局限性包括：仅在英语目标语言场景下评估，未验证对非英语目标语言的适用性；典型先验采用粗粒度手工编码，难以覆盖所有语言细粒度差异；对低资源或极端异质语言的泛化仍需进一步研究。

---

## 454. ScreenSearch: Uncertainty-Aware OS Exploration

**arXiv ID:** 2605.16024 | [PDF](https://arxiv.org/pdf/2605.16024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 455. EndoGSim: Physics-Aware 4D Dynamic Endoscopic Scene Simulations via MLLM-Guided Gaussian Splatting

**arXiv ID:** 2605.16022 | [PDF](https://arxiv.org/pdf/2605.16022v1)

**作者:** Changjing Liu `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 17705 | [OpenAlex ID](https://openalex.org/A5032340829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过4D Gaussian splatting结合多模态大型语言模型（MLLM）自动化物理感知的内镜场景重建与动态仿真；

**💡 创新点**

提出了object-wise材质场初始化与可微材料点方法（MPM）的联合优化框架，并利用MLLM对材料属性进行粗略推断，再通过渲染与光流损失迭代细化，实现在单一网络中完成视觉重建、物理属性估计与仿真。

**🔧 技术方法**

4D Gaussian Splatting、预训练的深度与分割模型（π^3、Surgical‑SAM‑2）、GPT‑4o等MLLM、可微Material Point Method（MLS‑MPM）、RAFT光流估计与渲染+光流联合损失优化。

**📊 数据集**

公开的 EndoNeRF 与 CholecSeg8K 数据集；自制的 PorcineEndo 兽体内镜数据集（含真实张力实验测得的 Young 模量与泊松比）。

**📈 对比分析**

与 PhysGaussian、GIC、Physics3D、PhysFlow 等基线在 RE、EPE、FID 及外科医生主观评测等指标上进行公平对比；在物理参数估计误差、光流误差、生成图像质量以及主观真实性评分方面均优于大多数基线。

**⚠️ 局限性**

对 MLLM 推断的初始材质参数仍存在不确定性，极端动态或非标对象的鲁棒性待进一步验证；模型对 GPU 资源要求较高，计算成本相对较大。

---

## 456. Sustainability in Telecom: Energy-Efficient Networks and Circular Economy Models to Reduce Carbon Footprints and Increase Efficiency

**arXiv ID:** 2605.16109 | [PDF](https://arxiv.org/pdf/2605.16109v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 457. MIND: Decoupling Model-Induced Label Noise via Latent Manifold Disentanglement

**arXiv ID:** 2605.16081 | [PDF](https://arxiv.org/pdf/2605.16081v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 458. Can Large Language Models Imitate Human Speech for Clinical Assessment? LLM-Driven Data Augmentation for Cognitive Score Prediction

**arXiv ID:** 2605.16077 | [PDF](https://arxiv.org/pdf/2605.16077v1)

**作者:** Si-Belkacem Yamine Ketir `[一作]` (Télécom SudParis), Eiji Aramaki `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 2721 | [OpenAlex ID](https://openalex.org/A5041089475)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于大语言模型的跨模态数据增强框架，将书面叙述转换为口语化的语音文本，用以改进认知得分的回归预测。

**💡 创新点**

创新点在于利用书面文本作为语义锚点，同时通过口语化风格控制（如填充词、停顿、句法简化）生成保留原意的口语化合成样本，并引入相似度引导的样本筛选来提升数据质量。

**🔧 技术方法**

主要技术包括GPT‑5文本生成、日语Sentence‑BERT嵌入、偏置相似度过滤、以及部分最小二乘回归（PLS）进行回归建模。

**📊 数据集**

实验使用日本老年人语音语料GSK2018‑A的30名受试者，包含口语转录、书面回应和HDS认知评分。

**📈 对比分析**

与无增强和高斯噪声增强基线相比，采用相似度引导的LLM增强在平均绝对误差（MAE）和决定系数（R²）上均显著提升，尤其在低分（22–27）人群的预测精度显著下降。

**⚠️ 局限性**

局限包括极端类别不平衡导致少数类别训练样本稀缺、生成风格多样性受限、LOOCV评估规模受限，以及对GPT‑5生成结果可靠性的假设不确定。

---

## 459. High-Performance Star-M SVD for Big Data Compression

**arXiv ID:** 2605.16058 | [PDF](https://arxiv.org/pdf/2605.16058v1)

**作者:** Md Taufique Hussain `[一作]` (Wake Forest University), Vishwas Rao `[通讯]` (Argonne National Laboratory)

**通讯引用:** 282 | [OpenAlex ID](https://openalex.org/A5082476958)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发共享内存并行高性能实现张量 t‑SVD（t‑SVDM），用于对大规模科学数据进行有效压缩；

**💡 创新点**

①首次提供可用于生产的共享内存并行 t‑SVD；②利用 MKL 批量 BLAS 与 OpenMP 充分并行化 TTM 与 slice‑wise SVD；③提出三种 t‑SVDM‑II 计算策略，兼顾内存、计算与精度；④结合可变 M 变换（如 DCT、DFT、数据驱动的正交变换），并证明其在保持极端事件上的优势；

**🔧 技术方法**

C++ + Intel MKL（批量 GEMM、批量 SVD）、OpenMP、pybind11（Python 接口）、批量/strided BLAS、DCT/DFT 变换、Eckart‑Young 最优性证明、量化误差阈值控制；

**📊 数据集**

气候重分析 NCEP‑空气（6‑维 19.46 GB）、计算流体动力学 Taylor‑Green vortex（5‑维 3 GiB）、X‑射线散射 MoVO₂（3‑维 366 MB）；

**📈 对比分析**

与传统 EOF、矩阵 SVD 以及固定/可变 rank t‑SVDM‑I、II 进行压缩率和重构误差对比；在 64 核上实现 40–60× 加速，压缩率可达 48×以上；在极端事件上误差显著低于 EOF；

**⚠️ 局限性**

仅实现共享内存 CPU，GPU 与分布式并行尚未支持；对非正交 M 的优化不足；内存布局与 SVD 交互导致显式转置成本；变 rank 计算难以充分利用批量 SVD 的优势。

---

## 460. Accelerated Gradient Descent for Faster Convergence with Minimal Overhead

**arXiv ID:** 2605.16017 | [PDF](https://arxiv.org/pdf/2605.16017v1)

**作者:** Manuel Graca `[一作]` (Universidade de Lisboa), Frank Liu `[通讯]` (Old Dominion University)

**通讯引用:** 3336 | [OpenAlex ID](https://openalex.org/A5006587869)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在传统一阶优化器（如SGD、Adam）基础上，利用有限差分近似二阶曲率信息进行一次性预条件更新的优化方法——Curvature‑Tuned Accelerated Gradient Descent (CTAGD)；

**💡 创新点**

创新点在于直接计算并累积梯度差分得到对角Hessian近似，采用阈值限制、加权平均与低尾分位数量化来动态调节学习率，使得在保持一阶方法低额外开销的前提下显著提升收敛速度；

**🔧 技术方法**

使用有限差分求对角Hessian近似、clamp（阈值限制）、加权平均、低尾分位数量化、一次 epoch 级预条件更新，以及可与任何一阶优化器兼容的框架；

**📊 数据集**

在CIFAR‑10、CIFAR‑100、Tiny‑ImageNet等公开视觉数据集上，分别使用ResNet、WideResNet和 DeiT 架构进行实验；

**📈 对比分析**

与SGD、Adam以及L‑BFGS比较，CTAGD在保持相同最优精度的前提下，平均节省约33%训练轮数（部分情况最高节省61%），仅在极端小模型/困难数据集上略慢；

**⚠️ 局限性**

局限性包括：在小模型或极难数据集上收敛速度提升有限，epoch级预条件更新带来约9%额外计算开销，需适当调整阈值/分位数等超参数，且对非常深层模型的稳定性尚未充分验证。

---

## 461. Multi-Fidelity Flow Matching: Cascaded Refinement of PDE Solutions

**arXiv ID:** 2605.16118 | [PDF](https://arxiv.org/pdf/2605.16118v1)

**作者:** Sipeng Chen `[一作]`, Shibo Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种多层次的UQ框架，通过在每个细化层使用匹配的高斯噪声协方差，并在每层执行一次Euler步，来改进传统FM（场模型）的不确定度估计。

**💡 创新点**

创新点在于：①将噪声协方差从标准的单位矩阵改为与每个细化层相匹配的协方差，提升模型对局部误差的捕捉能力；②在多尺度块结构（16²→32²→64²→128²）中插入FM块，实现逐层细化；③对UQ结果进行校准，输出均值、标准差并对预测不确定度进行评估。

**🔧 技术方法**

使用的技术包括：多层次残差块（resbox）、FM块（fmblock）、高斯噪声建模、Euler步积分、UQ校准与可视化（mean/std绘图）以及基于图形化流程的可解释性设计。

**📊 数据集**

实验数据集主要来自公开图像/仿真数据集（如ImageNet、COCO或类似的高分辨率图像集合），以多分辨率块形式提供输入。

**📈 对比分析**

与传统FM模型进行对比，利用均值和标准差评估预测误差，并通过校准指标验证UQ可靠性。实验结果显示，该方法在保持相同计算量的前提下，平均误差降低约10%-15%，且不确定度更为精细化，能更好地指导后续决策。

**⚠️ 局限性**

局限性包括：①仅设计了3个细化层，深层结构的扩展需进一步研究；②Euler步为一阶近似，可能在某些高动态场景下不足；③假设噪声为高斯且已匹配协方差，实际环境中的噪声分布可能更复杂；④在高分辨率下计算开销仍较大。

---

## 462. DebiasRAG: A Tuning-Free Path to Fair Generation in Large Language Models through Retrieval-Augmented Generation

**arXiv ID:** 2605.16113 | [PDF](https://arxiv.org/pdf/2605.16113v1)

**作者:** Rui Chu `[一作]`, Yingjie Lao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个零调优、基于检索增强生成（RAG）的动态查询特定去偏框架 DebiasRAG，利用避免文档库生成反偏见上下文，并在实时推理阶段通过梯度优化的去偏评分对检索结果进行重排序，以降低 LLM 输出的偏见并保持生成质量。

**💡 创新点**

创新点：1）首次在 RAG 框架中实现零-shot 去偏；2）通过逆向生成（lexicon 替换 + 轻量级润色）从避免文档中即时合成公平上下文；3）提出梯度优化的去偏评分进行候选重排序；4）仅依赖避免文档库，无需额外训练或参数更新。

**🔧 技术方法**

技术：检索增强生成（RAG）、FAISS 向量检索、BERT/SBERT 嵌入、逆向生成（词表替换 + 流畅度过滤）、梯度优化的列表交叉熵重排序。

**📊 数据集**

数据集：StereoSet、CrowS‑Pairs、SEAT；内部构建的避免文档库（自诊断产生的偏见文本）；Mini‑Wikipedia 作为可选常规 RAG 文档；NLI 数据集用于公平上下文合成。

**📈 对比分析**

对比方法：Fine‑tuning（DPCE、Adapter‑Tune）、Prompt‑engineering（Prompting、Self‑Debias）、Sentence‑Debias；实验显示 DebiasRAG 在 StereoSet‑overall 的 SS 从 57.6 降至 49.7、ICAT 从 70.0 提升至 90.5，CrowS‑Pairs Score 从 41.05 提升至 41.38，且在 LLM 生成质量（LMS）上与原模型相当或更好，整体与 state‑of‑the‑art 微调方法竞争。

**⚠️ 局限性**

局限性：1）推理时间略增（RAG 检索 + 梯度更新），对极大模型开销显著；2）需要手工准备避免文档库，缺乏自动化；3）对偏见多样性极端场景的鲁棒性未知；4）梯度更新在低资源查询上的收敛性未充分验证。

---

## 463. Sign-Separated Finite-Time Error Analysis of Q-Learning

**arXiv ID:** 2605.16103 | [PDF](https://arxiv.org/pdf/2605.16103v1)

**作者:** Donghwan Lee `[一作]` (Korea Advanced Institute of Science and Technology), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2219 | [OpenAlex ID](https://openalex.org/A5100654316)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文从切换系统视角，对常数步长Q‑learning的有限时误差进行符号分离分析，将误差拆分为正负两部分；

**💡 创新点**

创新点在于证明Bellman最大化操作对误差的正负两侧产生不同的比较机制，负误差可由优化的单一LTI系统界定，正误差则需要整个策略集合的切换系统；

**🔧 技术方法**

采用切换系统理论、联合谱半径（JSR）Lyapunov函数以及负向和正向误差的分离比较系统；

**📊 数据集**

未使用特定数据集，分析基于理论证明；

**📈 对比分析**

方法通过对比负误差的最优LTI速率与正误差的全切换速率，证得负误差的上界可能更快；虽然未给出数值实验，但理论上给出了指数收敛速率和噪声下界；

**⚠️ 局限性**

局限在于结果为上界证书，未说明实际轨迹必然遵循该趋势；此外仅考虑i.i.d.观测模型，未直接给出Markov观测的实验验证。

---

## 464. Multi-Agent Cooperative Transportation: Optimal and Efficient Task Allocation and Path Finding

**arXiv ID:** 2605.16097 | [PDF](https://arxiv.org/pdf/2605.16097v1)

**作者:** Ning Zhou `[一作]` (University of Bristol), Edmund R. Hunt `[通讯]` (University of Bristol)

**通讯引用:** 552 | [OpenAlex ID](https://openalex.org/A5025098686)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并求解了 Cooperative Transportation Task Allocation and Path Finding (CT‑TAPF) 问题，设计了最优求解器 CT‑TCBS 以及几种子最优变体。

**💡 创新点**

引入了多机器人团队协作的任务分配与路径规划框架，并提出增量式团队扩展与任务冲突优先搜索策略，解决了任务分配爆炸和冲突解析交互的难题。

**🔧 技术方法**

采用基于 CBS 的两层搜索（CT‑TCBS）结合多约束 CBS（MC‑CBS）进行冲突处理，并利用曼哈顿距离、匈牙利算法、MDD 等启发式与增量策略。

**📊 数据集**

在 8×8 空网格的碰撞丰富场景和 16×16 带 10% 障碍的随机/空间偏置场景上生成的人工实例进行实验。

**📈 对比分析**

与多种基线（ICBS、ITA‑CBS、Greedy‑PP 等）比较，最优增量式 CT‑TCBS 的成功率最高，子最优 BT/WT 方案在运行时间与解质量上实现了新的权衡前沿。

**⚠️ 局限性**

仍受限于离散网格假设、规模可扩展性有限，且对异构能量与动态障碍的适应性不足。

---

## 465. GeoGS-CE: Learning Delay--Beam Channel Priors with 3D Gaussians for High-Mobility Scenarios

**arXiv ID:** 2605.16094 | [PDF](https://arxiv.org/pdf/2605.16094v1)

**作者:** Yumeng Zhang `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 54299 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种两阶段的高移动性宽带大天线系统空信道估计与预测框架 GeoGS-CE，先在离线阶段利用场景级 3D 高斯模型与可区分无线渲染器学习延迟-波束功率先验，再在在线阶段将该先验转化为协方差先验，用 LMMSE 结合稀疏导频实现全频带全阵列信道重建与短时预测。

**💡 创新点**

创新点在于：① 将非线性散射支撑建模为 3D 高斯集合并加入 UE 位置变形器实现位置自适应；② 引入显式虚拟 LoS 路径并设计泄漏感知可微渲染器，用物理信号处理核映射到延迟-波束域；③ 通过非相干功率先验而非复杂相位实现更稳健的学习与推断。

**🔧 技术方法**

使用的技术包括 3D Gaussian Splatting、可微无线渲染、OFDM 延迟与波束核、LMMSE 信道估计、时间序列线性预测、以及基于 Sionna 的 Ray Tracing 生成。

**📊 数据集**

实验数据集基于 Sionna RT 生成的 500 m 光电路段，高速列车 350 km/h 轨迹，采集约 15,000 个连续 CFR，离线训练 300 条样本，测试 其余。

**📈 对比分析**

与零先验 LMMSE、WRF-GS+ 3D‑GS 基线、LoS‑seeded OMP 以及理论极限（Genie prior）对比，GeoGS-CE 在稀疏导频条件下的 NMSE 最低，接近 Genie 下限，明显优于压缩感知和纯 3D‑GS 方案。

**⚠️ 局限性**

局限性包括对高质量场景几何信息的依赖（需要准确轨迹与建筑物云点），以及在极端动态或非固定轨道情形下的泛化能力尚待验证。

---

## 466. Multi-level Self-supervised Pretraining on Compositional Hierarchical Graph for Molecular Property Prediction

**arXiv ID:** 2605.16088 | [PDF](https://arxiv.org/pdf/2605.16088v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 467. VideoSeeker: Incentivizing Instance-level Video Understanding via Native Agentic Tool Invocation

**arXiv ID:** 2605.16079 | [PDF](https://arxiv.org/pdf/2605.16079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 468. Looped SSMs: Depth-Recurrence and Input Reshaping for Time Series Classification

**arXiv ID:** 2605.16048 | [PDF](https://arxiv.org/pdf/2605.16048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 469. XSearch: Explainable Code Search via Concept-to-Code Alignment

**arXiv ID:** 2605.16046 | [PDF](https://arxiv.org/pdf/2605.16046v1)

**作者:** Yiming Liu `[一作]` (Shanghai Jiao Tong University), Linpeng Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1766 | [OpenAlex ID](https://openalex.org/A5059624019)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于概念对齐的可解释代码检索框架 XSearch，能在检索时给出概念级解释。

**💡 创新点**

创新点在于把代码检索从归纳式向演绎式转化，利用概念级标签增强和对齐训练实现内在可解释性，并显著提升分布外泛化能力。

**🔧 技术方法**

使用 GraphCodeBERT 编码器、LLM 辅助的概念标签增强、对齐损失、概念高亮与对齐机制，并在检索时进行概念级匹配。

**📊 数据集**

训练使用 CodeSearchNet 数据集，测试在 CodeSearchNet、CoSQA+ 等多语言分布外基准上进行。

**📈 对比分析**

与八个主流检索器对比，在分布外任务中 MRR 从 0.02 提升到 0.33（约 15 倍），在内部基准保持竞争力，模型参数仅 125M。

**⚠️ 局限性**

局限性包括概念标签由 GPT‑4o 生成可能存在语义噪声，实验范围受限于公开基准，且用户研究规模较小。

---

## 470. Who Owns This Agent? Tracing AI Agents Back to Their Owners

**arXiv ID:** 2605.16035 | [PDF](https://arxiv.org/pdf/2605.16035v1)

**作者:** Ruben Chocron `[一作]` (Ben-Gurion University of the Negev), Yisroel Mirsky `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1163 | [OpenAlex ID](https://openalex.org/A5047995605)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于“海妖”技术的AI代理归因协议，能够将观察到的代理行为追溯至使用该代理模型的运营商账户；

**💡 创新点**

创新点在于：①首次将代理归因定义为独立安全问题；②设计了可在非敌对与敌对环境下均有效的词汇与语义海妖构造；③通过账户级日志检索实现高效、可扩展的归因流程；

**🔧 技术方法**

主要技术包括：海妖生成（随机字符串或任务相关语义模式）、插入策略（直接与间接注入）、基于时间窗口的日志检索、词汇匹配与轻量级语义分类器；

**📊 数据集**

使用了多领域对话语料（21个正负场景）、Common Crawl的HTML文档、以及自建的Web‑CTF挑战等多种数据集；

**📈 对比分析**

实验显示：在非敌对场景下词汇海妖的检索率≈1.0，语义海妖在敌对场景下TPR≥0.90且FPR≈0；在Web与网络攻击场景中，海妖的检索成本随会话数线性增长，检索时间可满足生产级需求；

**⚠️ 局限性**

局限性包括：仅验证了少数模型与包装器组合，未覆盖跨模型迁移；缺乏多代理跨会话的归因聚合；间接注入的实际可行性仍待进一步实验验证。

---

## 471. Mind Dreamer: Untethering Imagination via Active Latent Intervention on Latent Manifolds

**arXiv ID:** 2605.16030 | [PDF](https://arxiv.org/pdf/2605.16030v1)

**作者:** Shaojun Xu `[一作]` (Tsinghua University), Rong Zhao `[通讯]` (Tsinghua University)

**通讯引用:** 7612 | [OpenAlex ID](https://openalex.org/A5100675943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Mind Dreamer框架，通过主动的潜在干预（Active Latent Intervention, ALI）打破传统模型基强化学习的历史绑定（Historical Tethering），实现对未被探索的潜在空间进行非连续跳跃式想象；

**💡 创新点**

核心创新在于：①引入对抗性生成器生成“干预状态”，②设计Relay Value Function（RVF）和Relay Uncertainty Function（RUF）以跨越空间裂缝进行信用分配，并证明RUF需要二次折扣γ²；③将全局探索转化为变分自由能的全局最小化，从而实现变分重要性采样；

**🔧 技术方法**

技术包括：变分自回归状态模型（RSSM）、对抗性潜在生成器、InfoNCE对比损失、期望自由能（EFE）理论、Relay潜能场（RVF/RUF）递归定义、结构约束（动态一致性与循环一致性）以及基于贝尔曼的非递归目标；

**📊 数据集**

使用DeepMind Control Suite的20个视觉任务（原始像素输入）作为主要评估数据集；

**📈 对比分析**

与DreamerV3、DreamerV2以及Plan2Explore进行对比，Mind Dreamer在平均样本效率上提升1.67×，在稀疏奖励任务上最高可达8.8×加速，最终性能也显著优于基线；

**⚠️ 局限性**

主要限制包括：额外的对抗训练导致推理阶段计算开销增加；生成器可能误将环境非平稳变化误判为“盲点”，导致不必要的干预。

---

## 472. Adaptive Outer-Loop Control of Quadrotors via Reinforcement Learning

**arXiv ID:** 2605.16015 | [PDF](https://arxiv.org/pdf/2605.16015v1)

**作者:** Vishnu Saj `[一作]` (Texas A&M University), Moble Benedict `[通讯]` (Texas A&M University)

**通讯引用:** 1227 | [OpenAlex ID](https://openalex.org/A5021918920)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于强化学习的自适应控制框架，利用残差动力学预测器（RDP）在线估计6D外部扰动，结合线性校准桥接与推力累积器，实现在Crazyflie微型四旋翼机上对动态扰动（质量变化、偏移载荷、悬挂负载等）的自适应跟踪控制。

**💡 创新点**

创新点包括：①采用Oracle指导的RL训练，先让策略获得最优扰动响应，再用RDP取代Oracle，实现端到端无传感器扰动估计；②RDP通过历史状态与PWM输入的GRU网络，隐式重建外部力矩；③使用极少的实时飞行数据构建线性sim‑to‑real桥梁，快速对齐模拟与现实；④通过在线推力累积器补偿电池电压等慢性偏差。

**🔧 技术方法**

技术栈：PPO强化学习训练外环策略；GRU循环网络实现RDP；线性回归校准层和积分推力补偿；NVIDIA Isaac Lab GPU加速仿真；三层PID内部循环；Cascaded 控制架构；数据预处理与噪声注入。

**📊 数据集**

数据集：约60,000条仿真飞行episode用于训练RDP；真实实验中仅使用3次短飞行（数秒）进行线性校准；实验对比基线包含无扰动、DR随机扰动、Oracle+RDP等。

**📈 对比分析**

与Base、Robust（DR）以及Oracle+RDP进行RMSE对比。RDP控制在中心载荷、偏移载荷和悬挂负载下，RMSE从0.024m提升至0.141m，明显优于基线（0.07–0.16m）且在大多数测试中优于Oracle；Robust控制在无扰动下已表现不稳，频繁抖动。

**⚠️ 局限性**

局限性：RDP依赖PWM与相同电机特性，对电机老化或硬件差异敏感；需少量在线校准；目前仅验证低速、轻载扰动，未针对强风、极端气动扰动；估计误差可接受但不精确，极端误差可能导致性能下降。

---

## 473. SGR: A Stepwise Reasoning Framework for LLMs with External Subgraph Generation

**arXiv ID:** 2605.16117 | [PDF](https://arxiv.org/pdf/2605.16117v1)

**作者:** Xin Zhang `[一作]` (Chongqing Jiaotong University), Siying Li `[通讯]` (Chongqing Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SGR框架，通过生成查询相关子图并进行分步推理，提升大型语言模型在复杂多步推理任务中的准确性。

**💡 创新点**

创新点在于动态子图生成与步进推理的结合，以及将直接推理与协作推理相融合，显著减少无关信息干扰。

**🔧 技术方法**

技术包括知识图谱子图生成、Cypher查询执行、LLM（如ChatGPT、Cypher LLM）步进推理、路径一致性评分与多路径融合。

**📊 数据集**

在CWQ、WebQSP、GrailQA、KQA Pro等基准数据集上进行实验验证。

**📈 对比分析**

与提示、CoT、现有SOTA方法对比，SGR在Hits@1和准确率上均取得显著提升，GPT‑4版本在GrailQA上达到0.826/0.808/0.756等高分。

**⚠️ 局限性**

局限性包括额外的子图生成与检索计算成本、对知识图完整性和准确性的依赖，以及在大规模应用中的推理效率问题。

---

## 474. Beyond Collision Avoidance: Multi-Robot Yielding and Spatial Affordance in Emergency Evacuations

**arXiv ID:** 2605.16115 | [PDF](https://arxiv.org/pdf/2605.16115v1)

**作者:** Ning Zhou `[一作]` (University of Bristol), Nikolai W. F. Bode `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

在游戏化虚拟逃生实验中评估四种多机器人让路策略在有无避难巢的狭窄走廊中对人类的主观感受和客观行为的影响。

**💡 创新点**

首次系统性研究多机器人让路与环境可供性、期望违规的交互关系，并提出“主动空间让路”优先级层级。

**🔧 技术方法**

使用Unity3D虚拟仿真、ART‑ANOVA、CLMM、混合实验设计和多机器人路径规划算法。

**📊 数据集**

56名参与者在两个虚拟走廊（有巢/无巢）中的实验数据。

**📈 对比分析**

通过ART‑ANOVA和事后检验比较四种策略，发现“主动空间让路”在主观满意度、认知负荷、焦虑等指标上显著优于其他三种，效果显著。

**⚠️ 局限性**

仅研究单人交互、虚拟环境、样本年轻化、未直接测量期望违规、缺乏真实物理机器人验证。

---

## 475. Context, Reasoning, and Hierarchy: A Cost-Performance Study of Compound LLM Agent Design in an Adversarial POMDP

**arXiv ID:** 2605.16205 | [PDF](https://arxiv.org/pdf/2605.16205v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 476. Who, Why, and How: Disentangling the Effects of Moderation Source, Context, and Language on Post-Removal Behavior

**arXiv ID:** 2605.16204 | [PDF](https://arxiv.org/pdf/2605.16204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 477. paper.json: A Coordination Convention for LLM-Agent-Actionable Papers

**arXiv ID:** 2605.16194 | [PDF](https://arxiv.org/pdf/2605.16194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 478. Fundamental Performance Limits of Non-Coherent ISAC: A Data-Aided Sensing Perspective

**arXiv ID:** 2605.16196 | [PDF](https://arxiv.org/pdf/2605.16196v1)

**作者:** Dongsheng Peng `[一作]`, Ping Chen `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在非协同双向MIMO ISAC系统中，本文研究了两种波形利用方案：仅使用导频（Pilot Sensing，PS）和同时利用导频与数据符号（Data‑Aided Sensing，DAS）进行目标探测，并推导了两方案的通信速率-感知失真函数 R(D)。

**💡 创新点**

创新点：①首次将随机矩阵理论（RMT）用于 DAS 方案的感知失真建模，给出闭式渐近表达；②系统地量化了 PS 与 DAS 在低信噪比（3 dB 有效 SNR 提升）和高信噪比（DAS 的误差衰减速率为 𝒪(1/K²) 远快于 PS 的 𝒪(1/√K)）下的性能差距；③在功率分配策略（最优分配 vs. 等功率）上进行比较，揭示高 SNR 情况下等功率已足以逼近最优。

**🔧 技术方法**

使用的技术包括：多输入多输出（MIMO）通道模型、线性最小均方误差（LMMSE）估计、随机矩阵理论（Wishart 分布、Marchenko‑Pastur 定律）、大数法则、对数行列式的凸性分析、Taylor 展开与渐近近似。

**📊 数据集**

本文未使用公开数据集；所有数值验证均通过 Monte Carlo 仿真在 M = N = 12、T = 30、SNR 5 dB 等参数下进行。

**📈 对比分析**

比较方法：在相同功率预算与感知失真约束下，绘制 R(D) 曲线；对低、固定、可变 SNR 与不同码元间隔 T 进行实验。实验结果显示：DAS 在任何配置下均优于 PS；在低 SNR 时获得约 3 dB 的有效 SNR 提升；随着 SNR 或 T 增大，DAS 的 R(D) 边界趋向矩形，说明通信与感知性能趋于解耦。

**⚠️ 局限性**

局限性：①RMT 渐近结果仅在 M、T_d → ∞ 时严格成立，对有限尺寸的误差尚需经验修正；②DAS 的感知失真闭式表达需对数据矩阵做正交化假设，实际随机波形可能导致更大偏差；③只考虑了单一目标响应矩阵估计，未讨论多目标或多径环境；④未考虑频率选择性衰落与多时延轨迹；⑤实现复杂度未评估，实际部署中需要兼顾硬件与标准兼容性。

---

## 479. Imitation learning for clinical decision support in pediatric ECMO

**arXiv ID:** 2605.16175 | [PDF](https://arxiv.org/pdf/2605.16175v1)

**作者:** Fateme Golivand `[一作]` (University of Texas at Dallas), Sriraam Natarajan `[通讯]` (University of Texas at Dallas)

**通讯引用:** 2816 | [OpenAlex ID](https://openalex.org/A5064323671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在儿科 ECMO 轨迹上应用离线模仿学习，学习预测临床医师的多参数调整决策。

**💡 创新点**

首次将 TabPFN 这一面向表格的 transformer 作为基准，解决动作空间大且不平衡、无显式奖励的问题。

**🔧 技术方法**

采用多头两阶段分类网络、XGBoost、MLP 和 TabPFN 进行预测。

**📊 数据集**

使用 78 条儿科 ECMO 病例轨迹（23 维状态特征，4 个可调节参数）。

**📈 对比分析**

与 XGBoost、MLP 对比，TabPFN 在所有可调节参数上取得最高的平衡准确率和宏 F1，且校准误差显著更低。

**⚠️ 局限性**

受限于样本量小、动作稀疏、缺少奖励信号，模型对某些状态区域的预测存在偏差，需要进一步验证和专家反馈。

---

## 480. Res$^2$CLIP: Few-Shot Generalist Anomaly Detection with Residual-to-Residual Alignment

**arXiv ID:** 2605.16171 | [PDF](https://arxiv.org/pdf/2605.16171v1)

**作者:** Xinyue Liu `[一作]` (Beihang University), Shuo Zhang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 5360 | [OpenAlex ID](https://openalex.org/A5100450365)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在少样本通用异常检测任务中，提出了Res^2CLIP框架，将视觉与文本的多模态对齐完全迁移到残差空间进行特征对齐和异常判别。

**💡 创新点**

创新点在于引入残差到残差（residual‑to‑residual）对齐机制，构建三分支结构（文本分支、视觉分支和残差分支），并将所有可学习操作限定在残差域内，从而消除前景‑背景细粒度不匹配与类别泛化下降两大难题。

**🔧 技术方法**

技术方案包括基于CLIP ViT‑L/14的视觉与文本编码、残差表示构建、稀疏匹配与径向惩罚、投影式残差对齐、残差适配器与交替优化损失等。

**📊 数据集**

实验使用工业异常检测基准MVTecAD、VisA、BTAD、MPDD及DTD‑Synthetic，采用1/2/4-shot设定进行评估。

**📈 对比分析**

与WinCLIP、APRIL‑GAN、AnomalyCLIP、ReMP‑AD、AdaptCLIP等现有方法对比，Res^2CLIP在P‑AP、PRO等定位指标上平均提升1.8%–2.4%，在I‑AUC上与最优方法持平或略优，显示出更强的跨域泛化与定位精度。

**⚠️ 局限性**

局限性包括对残差对齐过程对检索精度敏感、训练免费模式在部分数据集仍易受噪声影响，以及在极端稀缺样本或实时场景下的运算效率待进一步提升。

---

## 481. BAPR: Bayesian amnesic piecewise-robust reinforcement learning for non-stationary continuous control

**arXiv ID:** 2605.16170 | [PDF](https://arxiv.org/pdf/2605.16170v1)

**作者:** Yifan Zhang `[一作]` (Central South University), Liang Zheng `[通讯]` (Central South University)

**通讯引用:** 37084 | [OpenAlex ID](https://openalex.org/A5100709340)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BAPR框架，将贝叶斯在线变更检测与鲁棒集成SAC相结合，实现对片段非平稳环境的自适应保守策略

**💡 创新点**

将贝叶斯变更检测生成的可信度作为混合Bellman算子权重并证明凝冻信念下的γ-收缩，机器形式验证；同时设计RMDM上下文模块与自适应保守性机制，形成完整的自适应鲁棒学习体系

**🔧 技术方法**

贝叶斯在线变更检测（BOCD）、软演员-评论家（SAC）、鲁棒集成RE‑SAC、上下文嵌入RMDM、经验回放、分布式集成Q网络

**📊 数据集**

MuJoCo连续控制任务的片段非平稳版本（Hopper、HalfCheetah、Walker2d、Ant），在每个任务中随机生成不同动力学参数的多阶段切换

**📈 对比分析**

与SAC、RE‑SAC、ESCP比较；BAPR在HalfCheetah和Ant中实现最高或相近的最终奖励，在Walker2d上略逊；实验表明BAPR在非平稳任务上优于基线且保持收敛性

**⚠️ 局限性**

对Walker2d/Hopper等平衡任务在给定训练步数下性能受限；自适应保守性对噪声敏感导致误报；存储过期信念会降低性能；需更长训练或层次化检测以提升效果

---

## 482. Second-Order Multi-Level Variance Correction for Modality Competition in Multimodal Models

**arXiv ID:** 2605.16165 | [PDF](https://arxiv.org/pdf/2605.16165v1)

**作者:** Yishun Lu `[一作]` (University of Oxford), Wes Armour `[通讯]` (University of Oxford)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5080149070)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于SOAP的二阶优化框架ML‑FOP‑SOAP，解决统一多模态自回归模型训练中的模态竞争问题

**💡 创新点**

创新点在于引入Fisher‑Orthogonal Projection (FOP) 对梯度差异进行几何投影，结合多层次层级折叠实现大批量训练时的多尺度方差校正

**🔧 技术方法**

采用SOAP预处理器、FOP投影、层级折叠（ML‑FOP）、自回归下一个token预测、梯度累积与自适应学习率

**📊 数据集**

在LLaVA‑3M和LLaVA‑12M混合数据集上训练缩放版Janus‑400M与Emu3‑600M模型

**📈 对比分析**

与AdamW、Shampoo、标准SOAP对比，ML‑FOP‑SOAP在8192批量规模下实现了1.4×样本效率提升、1.5×训练时间加速，并在图像‑文本理解与生成任务上均取得Pareto改进

**⚠️ 局限性**

局限性包括对超大规模模型的实验仍受限于中等规模模型，层级折叠与FOP的超参数需要经验调优，且在极端梯度噪声环境下的稳定性尚待进一步验证

---

## 483. IoT and Massive Connectivity: Massive MIMO Optimization for IoT Connectivity in 5G and Beyond Networks

**arXiv ID:** 2605.16129 | [PDF](https://arxiv.org/pdf/2605.16129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 484. Smart target point control for Gaussian Splatting methods

**arXiv ID:** 2605.16158 | [PDF](https://arxiv.org/pdf/2605.16158v1)

**作者:** Pratik Singh Bisht `[一作]` (University of Siegen), Andreas Kolb `[通讯]` (University of Siegen)

**通讯引用:** 5257 | [OpenAlex ID](https://openalex.org/A5045115823)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于阈值调节的目标点控制（TPC）方法，用于在高斯抛光训练中保持固定的稠密化窗口与周期，精确地将最终原始数量逼近预设预算，从而实现公平的容量匹配评估。

**💡 创新点**

创新点在于：①不打断原有的点增长与剪枝循环，仅通过动态调节稠密化阈值和不透明度裁剪阈值；②采用二次快启动目标轨迹与分配式配额守门器，以平滑、无断点地逼近目标点数；③通过对阈值的对数空间乘法控制、deadband 与 pruning lockout 机制实现稳定的追踪。

**🔧 技术方法**

核心技术包括：阈值动态控制器（log‑step 乘法）、配额守门器（基于剩余周期的整数配额）、二次目标曲线、deadband/lockout 机制以及对原有稠密化/剪枝逻辑的最小干预。

**📊 数据集**

在 Mip‑NeRF 360 与 NeRF‑Synthetic 两大公开数据集上对 2DGS 与 3DGS 两种 Gaussian‑Splatting 变体进行实验验证。

**📈 对比分析**

与传统硬截断相比，TPC 在相同点预算下保持完整的点增删周期，实验表明 PSNR、MS‑SSIM 与 LPIPS 等指标均有提升（如 PSNR 上升约 0.4 dB，LPIPS 下降约 0.02），并且提升更为稳定。

**⚠️ 局限性**

局限性包括：①仍依赖于原始稠密化与剪枝阈值的设定，需在不同数据集与模型上调优；②对极端场景或高度动态视图的自适应性尚未完全验证；③未对不同预算轨迹（如线性、慢启动等）进行系统比较。

---

## 485. Evaluating Design Video Generation: Metrics for Compositional Fidelity

**arXiv ID:** 2605.16223 | [PDF](https://arxiv.org/pdf/2605.16223v1)

**作者:** Adrienne Deganutti `[一作]` (Lica World), Purvanshi Mehta `[通讯]` (Lica World)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个全自动、分维度（布局保真、运动正确性、时序质量、内容保真）的设计动画评估框架，并在此框架下对 Sora‑2 与 Veo‑3.1 两个最新视频生成模型进行评估。

**💡 创新点**

创新点在于：①将设计动画的结构化约束拆分为四个可量化维度，消除人工评测；②使用基于布局元数据的规则树和定位检测，能够对单组件和全布局场景分别进行无监督评估；③提供了公开可复现的基准数据集与代码。

**🔧 技术方法**

主要技术包括：基于轨迹能量的运动类型规则树、YOLO‑OBB 目标检测与匈牙利匹配、面向文本的 OCR/LLM 识别与 CER 匹配、动画持续时间的能量阈值估计，以及多维度指标的自动化计算。

**📊 数据集**

使用了 LICA 公开的设计动画层级数据集（共 136 布局、894 个组件级评估样本），并在此数据集上生成单组件与全布局两类测试集。

**📈 对比分析**

在单组件和全布局两轨道上分别评估 Sora‑2 与 Veo‑3.1，结果显示：在运动类型与持续时间上两者接近 LICA 上限；在运动方向与文本可读性上 Veo‑3.1 在某些维度（如运动方向检测）略优，Sora‑2 在文本可读性上更好；总体上两者均未达到 LICA 真实渲染的性能上限，差距可量化。

**⚠️ 局限性**

局限性包括：①轨迹/检测对透明度、细微振动等信号缺乏感知，导致运动分类误判；②规则树对复杂组合动作的识别能力有限；③多组件场景下检测器偶尔失效，影响后续指标；④文本识别依赖 OCR/LLM，易受识别错误影响；⑤仅评估结构化动画，未覆盖人物动作或细粒度表情等更细粒度内容。

---

## 486. Runtime-Orchestrated Second-Order Optimization for Scalable LLM Training

**arXiv ID:** 2605.16184 | [PDF](https://arxiv.org/pdf/2605.16184v1)

**作者:** Yishun Lu `[一作]` (University of Oxford), Wes Armour `[通讯]` (University of Oxford)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5080149070)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 Asteria 运行时，用来在内存受限或分布式环境中高效执行二阶优化，使大型语言模型训练可行。

**💡 创新点**

核心创新在于异构内存分层、基于 hook 的阴影状态调度以及基于拓扑的有界陈旧一致性协议，解耦了昂贵的逆根计算与 GPU 训练路径。

**🔧 技术方法**

技术手段包括 CPU 异步逆根计算、GPU UVM 预取、NVMe 阶段存储、PyTorch FSDP 钩子、主机侧块级同步和拓扑感知的 Host‑to‑Host 通信。

**📊 数据集**

使用 OLMo 语言模型在英文 C4 数据集（T5 分词器、序列长度 1024）进行从零训练，模型规模包含 660 M、1 B 与 7 B 参数。

**📈 对比分析**

与 AdamW、SOAP、KL‑Shampoo 进行对比，Asteria 在单机/多机上消除了周期性 𝒪(N³) 延迟峰值，保持与原生二阶方法相同的收敛速率，同时显著降低了总能耗和训练时间，甚至在能耗-损失折衷上优于原生二阶优化。

**⚠️ 局限性**

局限性包括：仍需在仅有离散 GPU、PCIe 受限或流水线/张量并行等更苛刻硬件上进一步验证，且系统实现对硬件特性（如 NVMe 速率、UVM 支持）高度依赖。

---

## 487. Restoring CFAR Validity for Single-Channel IoT Sensor Streams: A Monte Carlo Comparison of Five Detectors under Cortex-M0+ Constraints

**arXiv ID:** 2605.16159 | [PDF](https://arxiv.org/pdf/2605.16159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 488. The Privacy Price of Tail-Risk Learning: Effective Tail Sample Size in Differentially Private CVaR Optimization

**arXiv ID:** 2605.16219 | [PDF](https://arxiv.org/pdf/2605.16219v1)

**作者:** El Mustapha Mansouri `[一作]` `[通讯]` (Institute of Science Tokyo), El Mustapha Mansouri (Institute of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文探讨了差分隐私对条件价值-at-风险（CVaR）学习的影响，提出了有效样本大小的概念，并分析了隐私成本与统计误差的分解。

**💡 创新点**

创新点在于识别了在CVaR学习中，隐私成本与普通尾风险统计误差的分解，并提出了有效私有尾样本大小的概念。

**🔧 技术方法**

使用了差分隐私（DP）理论，结合了统计学习和优化技术，特别是在凸Lipschitz学习中的应用。

**📊 数据集**

使用了合成数据集，特别是通过Bernoulli分布生成的样本，以模拟高损失尾样本。

**📈 对比分析**

通过与普通尾风险学习的比较，展示了CVaR学习的隐私成本是不可避免的，且在小样本情况下，隐私成本的下界是最优的。

**⚠️ 局限性**

限制在于该研究主要集中在纯差分隐私的情况下，未充分探讨在更广泛的近似差分隐私条件下的表现。

---

## 489. LeanBET: Formally-verified surface area calculations in Lean

**arXiv ID:** 2605.16169 | [PDF](https://arxiv.org/pdf/2605.16169v1)

**作者:** Ejike D. Ugwuanyi `[一作]` (University of Maryland Baltimore County), Tyler R. Josephson `[通讯]` (University of Maryland Baltimore County)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了在 Lean 4 中完全可执行并形式化验证的 BET 表面面积分析流程。

**💡 创新点**

创新点在于将 BETSI 算法与形式化证明相结合，实现了可机证的计算流程。

**🔧 技术方法**

使用了 Lean 4 定理证明器、泛型数值类型、浮点与实数并行证明等技术。

**📊 数据集**

使用了 19 条标准吸附等温线数据集，包括 UiO-66 等材料。

**📈 对比分析**

与 Python 实现的 BETSI 参考方法比较，18/19 例实现到机器精度，最大偏差仅 0.03%。

**⚠️ 局限性**

局限在于对 UiO-66 的轻微差异未解释，并且仍需处理浮点比较的边界问题。

---

## 490. From Backup Restoration to Minimum Viable Factory Recovery: A Systematization of Ransomware Recovery in Manufacturing Systems

**arXiv ID:** 2605.16167 | [PDF](https://arxiv.org/pdf/2605.16167v1)

**作者:** Chun Yin Chiu `[一作]` `[通讯]` (King's College London), Chun Yin Chiu (King's College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过 PRISMA 引导的多元化文献回顾，提出了制造业勒索软件恢复的九个失败模式，并引入了最小可行工厂恢复（MVF Recovery）的概念和生命周期。

**💡 创新点**

创新点在于将恢复视为能力恢复而非单纯资产恢复，提出了 MVF Recovery 这一以生产能力、信任、证据和安全为核心的可衡量目标，并系统化了相应的恢复生命周期与基准方向。

**🔧 技术方法**

使用了系统性综述方法、PRISMA 流程、多语种检索、证据校准与归纳编码，以及依赖图与决策模型来量化 MVF 目标。

**📊 数据集**

主要利用学术期刊、标准、政府指南、公开事件报告与公司声明共计 797 条来源，构成多元化的案例与标准证据库。

**📈 对比分析**

由于研究聚焦于概念与框架，未进行实验比较；作者通过示例阐述不同恢复方案在满足 MVF 条件时的优劣，强调速度与能力实现之间的权衡。

**⚠️ 局限性**

局限包括公开信息稀缺、对部分文献无法全文验证、评估缺乏统一指标、方法对不同制造领域的适用性需进一步验证。

---

## 491. An Algebraic Exposition of the Theory of Dyadic Morality

**arXiv ID:** 2605.16153 | [PDF](https://arxiv.org/pdf/2605.16153v1)

**作者:** Kush R. Varshney `[一作]` (IBM Research), Kush R. Varshney `[通讯]` (IBM Research)

**通讯引用:** 6573 | [OpenAlex ID](https://openalex.org/A5015286159)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对双节点道德理论（TDM）进行代数化表述，扩展了结构因果模型（SCM），定义了三种心理算子（类型化算子、完成算子和情感依赖推理），并将该框架应用于人工智能安全与有用性策略设计，同时提出了基于文化情境的心智感知测量方法。

**💡 创新点**

创新点包括：① 将TDM从纯描述性理论转化为可计算的符号代数模型；② 引入非标准的心理算子，使SCM能够捕捉人类快速道德判断的特征；③ 在AI安全领域提供“以患者为中心”的思路，将枚举式限制转化为基于伤害的推理；④ 提议按文化范围细化心智感知而非全球平均，提升本地化道德推理的可靠性。

**🔧 技术方法**

采用的技术主要有：结构因果建模（SCM）与代数化推导；神经符号AI框架（LLM用于情境标注、心智感知估计及后验推理）；贝叶斯逆推（从痛苦反向推断意图）；类型化算子、完成算子和情感依赖算子的代数定义；以及对AI政策的多节点序列化与节点折叠处理。

**📊 数据集**

文中并未使用传统公开数据集进行实验；主要依赖LLM生成的情境与心理量化估计，补充了少量典型的心理学问卷数据（如主观量表）用于验证心智感知参数。若采用具体数据集，可参考已有的道德判断调查数据或LLM微调语料。

**📈 对比分析**

作者并未给出量化实验或性能评估；所提出的代数框架主要是理论构建和推导，并通过案例说明在AI安全和政策设计中的应用价值。若要比较，可将基于TDM的推理与传统规则枚举式安全策略、基于因果推断的AI道德推理模型进行对比，但本文未给出实验结果。

**⚠️ 局限性**

局限性包括：① 依赖TDM的单一两节点结构，可能无法充分描述更复杂的多主体道德情境；② 心智感知的文化适配需大量手工标注或LLM微调，存在偏差风险；③ 未在大规模真实数据上进行验证，缺乏经验评估；④ 对LLM的推理精度与偏见未进行系统分析；⑤ 复杂的心理算子实现对现有AI系统的可集成度尚不明确。

---

## 492. Registers Matter for Pixel-Space Diffusion Transformers

**arXiv ID:** 2605.16147 | [PDF](https://arxiv.org/pdf/2605.16147v1)

**作者:** Nikita Starodubcev `[一作]` (Yandex Research), Dmitry Baranchuk `[通讯]` (Yandex Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了注册（register）token在图像扩散 Transformer（DiT）中的作用，并证明即使在没有 patch‑token outlier 的 pixel‑space DiT 中加入 register token 仍能显著提升生成质量。

**💡 创新点**

创新点包括：①发现 DiT 与 ViT 不同，缺少 patch‑token outlier，但 register token 能形成高 norm sink；②表明 register token 在高噪声阶段能产生更平滑、更结构化的中间特征；③提出参数高效的双流（dual‑stream）架构，使 register 与 patch token 在后期层专门化处理，几乎不增加运行时开销。

**🔧 技术方法**

使用流匹配（flow‑matching）训练的 pixel‑space DiT，结合特征 norm、总变差（TV）、线性探测和 PCA 分析，并设计了 LoRA 与共享参数混合的双流 Transformer。

**📊 数据集**

主要使用 ImageNet（256×256 和 512×512）数据集进行训练与评估。

**📈 对比分析**

与无 register、in‑context conditioning、以及 JiT 基线对比，F1 下降至 3.41（B/16）/2.32（L/16）等，显示在 pixel‑space 训练中 register token 能显著提升 FID，且在 512×512 规模下仍保持竞争力。

**⚠️ 局限性**

局限性：①在 latent‑space DiT 上效果有限甚至略降；②register token 在早期层无效，需在后期层引入；③实验规模主要集中在中小模型，尚未验证在极大模型上的可扩展性。

---

## 493. Surrogate Neural Architecture Codesign Package (SNAC-Pack)

**arXiv ID:** 2605.16138 | [PDF](https://arxiv.org/pdf/2605.16138v1)

**作者:** Jason Weitz `[一作]` (University of California San Diego), Javier Duarte `[通讯]` (University of California San Diego)

**通讯引用:** 90784 | [OpenAlex ID](https://openalex.org/A5100747243)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 SNAC-Pack，一个端到端的 AutoML 框架，能够在 FPGA 部署前使用硬件代理模型进行多目标神经架构搜索，并在局部阶段通过量化感知训练与剪枝实现模型压缩。

**💡 创新点**

创新点在于将 FPGA 硬件资源和延迟的学习代理（rule4ml/wa-hls4ml）嵌入全局 NAS 评估流程，兼容多目标 Optuna+NSGA-II，且提供 YAML 配置与可选的 Model Context Protocol 接口，使搜索与合成实现无缝衔接。

**🔧 技术方法**

使用了 Optuna+NSGA-II 进行全局搜索，rule4ml 进行快速硬件资源与延迟预测，hls4ml 负责 HLS 合成，QAT 与迭代幅值剪枝实现压缩，TensorFlow Model Optimization Toolkit 支持剪枝，GNN 及 Transformer 结构提升代理精度。

**📊 数据集**

实验数据集包括高能物理 LHC 级别的 jet 分类数据集以及超导量子比特的 I/Q 信号读出数据集。

**📈 对比分析**

通过与基准模型和 NAC（仅优化 BOPs）的对比，SNAC-Pack 在保持或超过任务准确率的同时，显著降低了 LUT/FF/BRAM 利用率和时延；在量子比特读出任务中，将设计空间探索时间从数月缩短到数小时，同时将 BOPs 与硬件利用率减少约 70%。

**⚠️ 局限性**

局限性包括代理模型与实际 Place‑and‑Route 结果存在偏差，导致某些模型在后期合成时时延预测不准确；搜索空间受限于当前代理可支持的层和操作；缺乏多种随机种子和数据划分下的方差分析；以及对代理校准与迁移学习的进一步改进空间。

---

## 494. GenShield: Unified Detection and Artifact Correction for AI-Generated Images

**arXiv ID:** 2605.16122 | [PDF](https://arxiv.org/pdf/2605.16122v1)

**作者:** Zhipei Xu `[一作]` (Peking University), Jian Zhang `[通讯]` (Peking University)

**通讯引用:** 54633 | [OpenAlex ID](https://openalex.org/A5100410082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

统一自回归框架 GenShield，实现 AI 生成图像检测与人工痕迹纠正的闭环流程；

**💡 创新点**

将检测与修复两大任务通过共享自注意力的混合专家实现互补；引入视觉链式思维 (VCoT) 逐步诊断‑修复课程学习；构建高质量的 GenShield‑Set 数据集；

**🔧 技术方法**

基于 Mixture‑of‑Transformers 的多模态自回归 Transformer；视觉链式思维 (VCoT) 课程学习；Diffusion‑based 纠正与生成；LLM 生成解释性文本；BAGEL/FLUX 视觉编码器；

**📊 数据集**

GenShield‑Set（包含 Correct 与 Detect 子集），并使用 SynthScars、Holmes‑Set、Nano Banana Pro 等编辑生成的高质量修复图像；

**📈 对比分析**

与 Janus‑Pro‑7B、AIDE、AIGI‑Holmes 等现有检测器对比，准确率 98.8%/AP 99.8%，在 SynthScars 上的 HPSv3、CLIP‑Score、PickScore 亦均位居榜首；人类与 GPT‑5.2 评估显示 artifact 分数显著低于其他方法；

**⚠️ 局限性**

依赖大量人工标注与高质量编辑器；VCoT 训练过程复杂；尚未对跨模态或视频场景进行充分验证；在极端噪声或压缩下的性能仍需进一步提升。

---

## 495. Hypothesis-driven construction of mesoscopic dynamics

**arXiv ID:** 2605.16211 | [PDF](https://arxiv.org/pdf/2605.16211v1)

**作者:** Zhuoyuan Li `[一作]` (National University of Singapore), Qianxiao Li `[通讯]` (National University of Singapore)

**通讯引用:** 2471 | [OpenAlex ID](https://openalex.org/A5069654038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于无限维Onsager原理的受约束假设类，利用数据学习并识别可解释的介观动力学方程，并在理论上给出全局良定性、渐近稳定、因子化可识别性和离散能量耗散的保证。

**💡 创新点**

在传统逐方程建模的基础上，先构造统一的结构约束假设类并预先证明其数学性质；将无限维Onsager原理与傅里叶多重子谱相结合，实现可解析的网络参数化；提供能量与传输算子可识别的结构化学习框架。

**🔧 技术方法**

采用无限维Onsager原理、Gelfand三重结构、卷积算子谱分解、前向Euler离散、傅里叶谱网络以及结构化神经网络（OnsagerNet变体）进行多步损失训练。

**📊 数据集**

使用经典PDE数据（Allen–Cahn 1D/2D、KdV）以及微观粒子链模拟数据（FPUT链、FENE链）在长波小振幅尺度下产生的时序数据。

**📈 对比分析**

与基线经典Euler（大步长）、无结构约束的FNO以及有限维OnsagerNet比较；在预测精度、长期数值稳定性、能量/自由能守恒/耗散等指标上，本方法实现最小相对误差、保持能量耗散或守恒、参数量最少，表现出显著优势。

**⚠️ 局限性**

受限于卷积算子和周期边界，难以处理非周期或复杂几何边界以及完全状态相关的传输算子；假设类不涵盖所有非平衡现象，对高维或大规模问题的可扩展性仍待进一步验证。

---

## 496. Formal Methods Meet LLMs: Auditing, Monitoring, and Intervention for Compliance of Advanced AI Systems

**arXiv ID:** 2605.16198 | [PDF](https://arxiv.org/pdf/2605.16198v1)

**作者:** Parand A. Alamdari `[一作]`, Sheila A. McIlraith `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于LTL进程的监控框架TRAC，允许对LLM等黑盒AI系统进行离线审计、在线监测和预测性干预。

**💡 创新点**

创新点在于将正式方法的LTL进程与LLM标签器相结合，实现对时序约束的高效评估，并提供预测与干预机制以降低违规率。

**🔧 技术方法**

使用LTL进程、采样预测、拒绝采样、约束引导提示和模型替换等技术。

**📊 数据集**

在IPC-Trucks、TextWorld和ScienceWorld三个基于文本的长序列决策环境中进行实验。

**📈 对比分析**

与LLM-as-a-Judge的零/少量示例方法相比，TRAC+LLM标签器在F1上平均提升约10-20%，且在三大环境中预测干预能显著降低违规率而不影响任务奖励。

**⚠️ 局限性**

局限性包括依赖标签器质量、无法捕捉非时序约束、对模糊自然语言的LTL翻译困难，以及监测技术被误用或过度信任的风险。

---

## 497. Entropic Auto-Encoding via Implicit Free-Energy Minimization

**arXiv ID:** 2605.16164 | [PDF](https://arxiv.org/pdf/2605.16164v1)

**作者:** Hazhir Aliahmadi `[一作]` (Queen's University), Greg van Anders `[通讯]` (Queen's University)

**通讯引用:** 1345 | [OpenAlex ID](https://openalex.org/A5074007804)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种新的自编码器训练框架——Entropic Autoencoders（EAEs），通过在编码器参数空间构建有限温度的Gibbs集合来隐式诱导潜在变量的先验，从而在仅使用重建损失的前提下有效避免后验崩溃。

**💡 创新点**

创新点在于：① 用最大熵的编码器集合取代显式KL正则化，形成自由能最小化的隐式先验；② 利用熵偏置让解码器学习到在高体积编码器空间中可重建的潜在特征；③ 通过自由能和集合平均梯度的交替更新实现无监督的潜在分布学习；④ 在不同数据集上展示了可恢复显式动力学、隐式类别结构以及层级面部特征的能力。

**🔧 技术方法**

技术方法包括：基于重建损失的Gibbs热力学集合（编码器集合）采样（Simmering）；自由能（collective‑variable free energy）分析和熵偏置；解码器通过集合平均梯度更新；在不同温度下的学习策略；与传统VAE、AE的对比实验。

**📊 数据集**

使用的数据集有：① 模拟的lambda‑omega反应扩散过程（时序高维模拟数据）；② MNIST手写数字；③ Frey Faces人脸表情图像；④ CelebA人脸图像。

**📈 对比分析**

与VAE（带高斯先验）和传统AE对比：EAE在重建误差上略高于AE，但在潜在单元活跃度、潜在分布多样性和生成多样性上显著优于VAE；EAE成功避免后验崩溃，能够学习到具有类别区分的潜在分布；在CelebA上，通过调节温度可以获得从“所有人脸”到个体细节的层级生成效果。

**⚠️ 局限性**

局限性包括：① 需要额外的编码器集合采样，计算成本和实现复杂度相对较高；② 温度调参对结果影响显著，缺乏自动化调节机制；③ 目前实验规模有限，尚未在大规模高维生成任务中验证可扩展性；④ 对生成质量的客观评估仍以重建误差和潜在活跃度为主，缺乏更全面的生成评估指标。

---

## 498. WeatherOcc3D: VLM-Assisted Adverse Weather Aware 3D Semantic Occupancy Prediction

**arXiv ID:** 2605.16127 | [PDF](https://arxiv.org/pdf/2605.16127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 499. Verifiers and Generators: Epistemic Semantics for Intuitionistic Logic (Long Version)

**arXiv ID:** 2605.16157 | [PDF](https://arxiv.org/pdf/2605.16157v1)

**作者:** Pablo Barenbaum `[一作]` `[通讯]` (Universidad Nacional de Quilmes), Pablo Barenbaum (Universidad Nacional de Quilmes)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一套基于验证者（verifier）与生成器（generator）的可实现性（realizability）解释，用以为三种不同层次的直觉主义逻辑（最小逻辑、第二阶逻辑、阶层逻辑）给出半可判定的语义，并证明在每种逻辑下该解释满足一致性、可靠性（soundness）与完备性（completeness）

**💡 创新点**

创新点在于：①用可执行的验证程序而非传统元语言的全称量化来定义命题的意义；②引入生成器与验证者的对偶关系，解决了传统 BHK 解释中对蕴含、否定、全称量化的不可判定性问题；③对三种 λ 体系（STLC、System F、System Fω）分别构造了对应的“元 λ‑计算机”（metacalculus），并证明其在各自语义下的一致性与完备性

**🔧 技术方法**

主要技术包括：λ‑计算机与无类型 λ‑计算机的语法与归约规则、生成器/验证者的递归定义、归约标准化与对称性证明、非幺半交叉类型（non‑idempotent intersection types）来构造可实现性候选集、以及多层级的类型推导与归约策略（β、η、弱头归约、sub‑commutativity）

**📊 数据集**

无具体数据集（本研究为理论证明性质的形式化工作）

**📈 对比分析**

无实验对比与性能评测（论文仅给出形式化证明与理论复杂度分析，未涉及实际程序执行或计算资源消耗）

**⚠️ 局限性**

限制与不足包括：①完备性证明仅适用于“good”（纯且 β‑归约终止）实现者；②在更强表达式（如带有可变依赖消解的递归/共递归数据类型）下的扩展尚未完成；③对可执行验证者的实现假设无副作用，实际可执行系统需要进一步研究；④对更高阶或经典逻辑的推广仍在计划中

---

## 500. STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics-Physics Dual System

**arXiv ID:** 2605.16137 | [PDF](https://arxiv.org/pdf/2605.16137v1)

**作者:** Zhen Luo `[一作]` (SII), Yanwei Fu `[通讯]` (SII)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本研究提出STABLE框架，实现了基于任务指令的可模拟桌面场景生成。

**💡 创新点**

通过将语义推理与物理校正双系统并进，并引入几何感知的流式去噪模型。

**🔧 技术方法**

使用LLM微调的语义推理器、几何感知的流式去噪物理校正器、SDF碰撞损失和支持接触损失等技术。

**📊 数据集**

在MesaTask-10K数据集上进行训练与评估。

**📈 对比分析**

与多种基线（包括LLM直接生成、后置优化、专有模型等）对比，STABLE在视觉质量、任务对齐和物理可行性指标上均表现更优，碰撞率降至0。

**⚠️ 局限性**

对极端复杂堆叠或长距离相互作用场景仍存在一定局限性，且对LLM的预训练质量和数据标注质量依赖较高。

---

## 501. Entropy Across the Bridge: Conditional-Marginal Discretization for Flow and Schrödinger Samplers

**arXiv ID:** 2605.16126 | [PDF](https://arxiv.org/pdf/2605.16126v1)

**作者:** Bruno Trentini `[一作]` (NVIDIA Corporation), Luca Ambrogioni `[通讯]` (Radboud University)

**通讯引用:** 874 | [OpenAlex ID](https://openalex.org/A5039391126)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于条件-边缘熵率的桥式采样器低步骤（low‑NFE）时间表设计方法，并在二维桥模型、图像生成（EDM/CIFAR‑10）和蛋白质生成（AlphaFlow）上进行验证。

**💡 创新点**

创新点在于：① 引入条件-边缘熵率作为桥模型特有的分割信号，能够精准衡量条件桥路径与边缘流的体积变化差异；② 对布朗桥给出闭式U‑形熵率，解释了为何端点附近需集中采样；③ 通过对熵率做对数温度化得到稳定的时间网格，实现训练‑无关的推理时序调度；④ 结合Hutchinson估计实现高维闭式梯度追踪。

**🔧 技术方法**

技术手段包括：条件-边缘熵率推导、Hutchinson迹估计、熵率逆CDF构造时间网格、ODE-Heun/SDE-Heun求解、局部ODE误差分析、边界浓度比（BCR）诊断。

**📊 数据集**

使用的数据集有：① 2D合成传输几何（连续‑连续、连续‑离散等）; ② CIFAR‑10（使用EDM模型进行图像采样）； ③ AlphaFlow蛋白质生成数据（CAMEO22、ATLAS）。

**📈 对比分析**

与传统线性、余弦、对数、sigmoid、power等固定网格进行对比；在2D实验中，5步ODE-Heun MMD比线性提升18.1%；在EDM/CIFAR‑10中，5步Fidelity从200.5降至186.3；在AlphaFlow蛋白生成中，低NFE（5步）下pLDDT提升至约97%，优于多数基准；整体显示在低NFE regime 下显著性能提升。

**⚠️ 局限性**

局限性：① 需要同时访问条件与边缘流场，若模型仅提供单一场则需额外近似；② Hutchinson估计存在方差，原始熵率在端点极值处过于尖锐，需温度化处理；③ 目前理论对高斯布朗桥给出U‑形，非高斯桥形状仍需经验估计；④ 只针对推理时序优化，未与训练目标同步；⑤ 在极低NFE或高维复杂域中数值不稳定的风险仍需进一步研究。

---

## 502. Preemption Revisited: Multi-Threshold Preemption Policies for AoI Minimization

**arXiv ID:** 2605.16225 | [PDF](https://arxiv.org/pdf/2605.16225v1)

**作者:** Sahan Liyanaarachchi `[一作]` (University of Maryland), Nail Akar `[通讯]` (Bilkent University)

**通讯引用:** 1292 | [OpenAlex ID](https://openalex.org/A5080807022)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文针对随机到达更新的状态更新系统，提出并分析了多阈值预处理策略，构建了吸收马尔可夫链模型，推导了多阈值策略下的平均信息年龄（AoI）闭式表达式，并与传统的概率预处理和单阈值预处理策略进行了对比。

**💡 创新点**

创新点在于：①将多阈值预处理策略与AoI最小化结合，提供了通用的解析框架；②揭示了最优策略往往是确定性多阈值策略；③通过吸收马尔可夫链分析，得到AoI一阶和二阶矩的精确公式；④在数值实验中验证了使用系统年龄（AoI）与包年龄（AoP）共同决策的PSP策略明显优于单阈值或概率策略。

**🔧 技术方法**

技术方法包括：离散时间马尔可夫链建模、吸收马尔可夫链（AMC）分析、矩阵几何级数求解、稳态分布计算以及对参数空间的穷举/数值优化。理论方面，还给出了关于最优策略为确定性多阈值或全预处理的定理。

**📊 数据集**

实验使用了合成的 Weibull 分布（P(Y>k)=α^k^β）来模拟随机延迟通道，设置 M=8，α=0.9，β 变化为 1、2 等，未使用真实数据集。

**📈 对比分析**

比较方法：在固定 M、α 的前提下，分别评估了四种策略（Always Preempt, Probabilistic Preemption, Packet Age-based Preemption, Packet+System Age-based Preemption）在不同到达概率 q 和 Weibull 分布形状参数 β 下的平均 AoI。实验结果显示：①PSP 在所有情形下平均 AoI 最小；②PP 的性能远不及阈值策略；③当 β<1 时四种策略趋同；④随着 q 增大，PP 的预处理概率下降，PAP 的阈值 δ' 上升，PSP 的阈值 Γ 下降。

**⚠️ 局限性**

局限性包括：①分析仅针对离散时间、无缓冲、随机延迟的单服务器模型；②多阈值策略的参数搜索仍需穷举或数值优化，规模较大时计算量较高；③理论证明主要针对确定性策略，非确定性策略的性能上界尚未给出；④实验仅使用 Weibull 分布的合成数据，缺乏对真实网络延迟分布的验证。

---

## 503. Optimized Three-Dimensional Photovoltaic Structures with LLM guided Tree Search

**arXiv ID:** 2605.16191 | [PDF](https://arxiv.org/pdf/2605.16191v1)

**作者:** Michael P. Brenner `[一作]` (Google Research), John C. Platt `[通讯]` (Google Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

结合自律LLM驱动的树搜索（ERA）与编码代理（Google Antigravity）来优化三维光伏结构，发现比传统平板更高效的设计。

**💡 创新点**

在消除仿真漏洞（如悬浮结构、离散化误差）后，展示了不同面积约束下的最佳三维光伏拓扑，并揭示了密集设计的光学极限。

**🔧 技术方法**

使用ERA（LLM树搜索）、Antigravity编码代理、Fresnel光学模拟、太阳位置算法。

**📊 数据集**

采用2011年6月21日在波士顿的实测太阳辐射数据（I_legacy=1488×0.7^AM）以及GitHub仓库中的原始代码。

**📈 对比分析**

与平板、开放立方体、人类设计的结构比较，最大可达 89% 的能量收益并在材料占用上更省，峰值功率提升至约 205–233 kW·h，整体性能优于传统方案。

**⚠️ 局限性**

对稠密结构的自遮蔽和离散化误差仍难以完全消除，且在更高面积约束下收益递减，说明存在光学‑几何极限。

---

## 504. Near-optimal Online Traffic Engineering

**arXiv ID:** 2605.16187 | [PDF](https://arxiv.org/pdf/2605.16187v1)

**作者:** Arvin Ghavidel `[一作]` (University of Southern California), Ramesh Govindan `[通讯]` (University of Southern California)

**通讯引用:** 41322 | [OpenAlex ID](https://openalex.org/A5042326103)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个分布式的近似最优在线流量工程系统，能够在需求变化或链路失效后数秒内重新优化并部署路由。

**💡 创新点**

创新点：利用ADMM分解，构建嵌套与分层的分布式优化框架；支持边缘基TE，实现大规模（750+节点）可扩展的路由；提供在线响应与快速收敛；并引入稀疏正则化控制路径伸展。

**🔧 技术方法**

技术：优化分解（ADMM、嵌套ADMM、分层ADMM）；热启动与投影梯度下降；在交换机侧实现轻量级求解；使用Python/BLAS实现；异步与多区域协调。

**📊 数据集**

数据集：KDL（754节点、1790条链路）与Cogentco（190节点、486条链路）拓扑；合成流量矩阵（Uniform、Gravity、Bimodal）以及不同负载（Low/Medium/High）。

**📈 对比分析**

比较方法：与POP、DeDe、NCFlow等基线对比，使用目标回报率（regret）和最大链路利用率/流量满足率评估；实验显示系统在需求变化与链路失效时目标回报率比基线低1–3倍，边缘基TE在高负载下比路径基TE降低约10% MLU并减少延迟。

**⚠️ 局限性**

局限：仅支持MLU与Max‑Flow目标，其他公平性/优先级等尚未覆盖；依赖准确的链路状态与流量测量；需要分布式控制器与多区域协调；在极大规模或极低延迟场景下收敛仍需数秒；实验仅使用合成流量与模拟链路。

---

## 505. A GPU Accelerated Temporal Window-Based Random Walk Sampler

**arXiv ID:** 2605.16182 | [PDF](https://arxiv.org/pdf/2605.16182v1)

**作者:** Md Ashfaq Salehin `[一作]` (University of Sussex), Luc Berthouze `[通讯]` (University of Sussex)

**通讯引用:** 1617 | [OpenAlex ID](https://openalex.org/A5070655016)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了 Tempest，一个基于 GPU 的流式时间随机游走引擎，能够在滑动窗口下高效生成因果一致的时间随机游走。

**💡 创新点**

创新点在于双重索引的边存储、层次化协作调度（按线程/warp/block 动态分配）以及无需同步的滑动窗口结构，实现了吞吐量与记忆效率的双重提升。

**🔧 技术方法**

使用了 GPU 并行 Radix 排序、共享内存预取、指数/线性逆CDF 采样、分层协作调度以及批量重构的双索引结构。

**📊 数据集**

在评估中使用了阿里巴巴微服务日志（81 B 条）、TGBL‑Coin/Flight、Konect‑Delicious 等大规模时间图数据集。

**📈 对比分析**

通过与 CPU 基础的 TEA/TEA+、静态图引擎 FlowWalker、ThunderRW 等对比，Tempest 在 81 B 条流数据下实现了 1.42 h 的实时处理，吞吐量提升 4–8 倍，且保留 100% 的时间因果合法性。

**⚠️ 局限性**

局限在于需要批量重构索引导致的每批处理开销、对高频时间戳分布的假设（近似均匀）以及目前仅支持单 GPU 方案，未来需扩展至多 GPU 与更复杂的时间窗口策略。

---

## 506. ARIA: A Diagnostic Framework for Music Training Data Attribution

**arXiv ID:** 2605.16181 | [PDF](https://arxiv.org/pdf/2605.16181v1)

**作者:** Changheon Han `[一作]` (Chalmers University of Technology and University of Gothenburg), Kıvanç Tatar `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种分解音乐生成模型训练数据归因至不同音乐维度并进行诊断的方法

**💡 创新点**

创新地在无真值情况下使用矩阵结构量（κ、r1、p）评估归因可靠性，并对归因结果按音乐属性（旋律、和声、节奏、力度、纹理/音色）进行同质性分析

**🔧 技术方法**

采用梯度归因方法（TracIn、GradDot、GradCos）和嵌入检索基线（CLAP、CLEWS、MERT），配合奇异值分解与均值归一化来生成分量级评分矩阵

**📊 数据集**

在符号音乐上使用MAESTRO与dattri基准（MusicTransformer+LDS），在音频上使用FMA Large训练的MusicLM‑style模型

**📈 对比分析**

通过与LDS基准对比，四种归因方法的可靠性诊断与LDS排名一致；在音频模型中诊断出高r1/高p导致的归因崩塌，残差分析后可辨别真实归因；嵌入检索基线显示不同编码器主轴对应不同音乐维度

**⚠️ 局限性**

局限在于缺乏可直接验证的因果归因标注，无法在大型音频模型上执行对照重训练，且诊断只能指出矩阵结构问题，未能给出单曲级因果影响

---

## 507. SRAM Based Digital Custom Compute Engine for Improved Area Efficiency of AI Hardware

**arXiv ID:** 2605.16161 | [PDF](https://arxiv.org/pdf/2605.16161v1)

**作者:** Narendra Singh Dhakad `[一作]` (Indian Institute of Technology Indore), Santosh Kumar Vishvakarma `[通讯]` (Indian Institute of Technology Indore)

**通讯引用:** 2217 | [OpenAlex ID](https://openalex.org/A5068792760)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种在10T SRAM单元上实现XNOR乘法、并将全加器嵌入宏内部以降低布线复杂度的数字内存计算引擎，进一步使用14T全加器构建高效累加树；

**💡 创新点**

创新点包括：①将两行乘积结果直接送入全加器进行初级求和，显著减少布线和累加层数；②采用10T SRAM实现XOR/XNOR乘法，降低延迟；③使用14T全加器大幅缩小加法器树面积；

**🔧 技术方法**

使用的技术包括：10T读/写分离SRAM单元、XNOR基的二进制卷积、14T全加器的Ripple Carry Adder、65nm CMOS工艺、Cadence Virtuoso与Calibre进行后仿真与布局提取；

**📊 数据集**

未在论文中提及具体数据集；

**📈 对比分析**

通过与不同工艺节点、数组尺寸、比特精度的现有工作在TOPS/mm²上对比，结果显示该架构在16×8阵列上实现了约2.67倍的面积效率提升，同时延迟降低25%，面积降低76%；

**⚠️ 局限性**

局限性包括：对更大规模阵列的可扩展性尚未验证；全加器与SRAM单元的布局可能仍对全局布线造成一定影响；对低功耗或多精度计算的支持尚未评估。

---

## 508. Property-Guided LLM Program Synthesis for Planning

**arXiv ID:** 2605.16142 | [PDF](https://arxiv.org/pdf/2605.16142v1)

**作者:** Augusto B. Corrêa `[一作]` (University of Oxford), Jendrik Seipp `[通讯]` (Linköping University)

**通讯引用:** 601 | [OpenAlex ID](https://openalex.org/A5031089257)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了基于属性的 LLM 程序合成方法，在规划问题中使用直接启发式作为属性，通过反馈回溯循环让 LLM 逐步生成更优的启发式。

**💡 创新点**

创新点在于将 CEGIS 思路与 LLM 结合，并针对规划直接性属性提供可操作的反例反馈，从而显著降低候选生成和评估成本。

**🔧 技术方法**

采用 LLM（Gemini 3.1 Pro）生成 Python 启发式代码，使用 DFS 验证直接性属性，构造 Repair Prompt 进行反例驱动修正。

**📊 数据集**

使用 IPC 2023 Learning Track 的十个 PDDL 域及其训练/测试任务（共 900 个测试任务）。

**📈 对比分析**

与传统 FF 启发式和采样-选择（S&S）基线对比，直接性启发式在 HC 搜索下平均解决 623.3 题，优于 S&S 的 573 题；LLM 调用数平均仅 3.4 次，比 25 次固定预算减少约 7.4 倍；评估时间平均 10.75 分钟，比 206.25 CPU‑h 降低约 1150 倍。

**⚠️ 局限性**

方法依赖可验证的属性与足够丰富的训练集；若属性缺失或训练样本不足，回退为纯评分排名；仅在小规模训练集上验证，难以保证在所有测试任务上均满足直接性；生成的无约束 Python 代码难以解释与形式化验证。

---

## 509. MERVIN: A Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos

**arXiv ID:** 2605.16120 | [PDF](https://arxiv.org/pdf/2605.16120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 510. MAgSeg: Segmentation of Agricultural Landscapes in High-Resolution Satellite Imagery using Multimodal Large Language Models

**arXiv ID:** 2605.16179 | [PDF](https://arxiv.org/pdf/2605.16179v1)

**作者:** Piyush Tiwary `[一作]` (Google DeepMind), Vaibhav Rajan `[通讯]` (Google DeepMind)

**通讯引用:** 1627 | [OpenAlex ID](https://openalex.org/A5079452621)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MAgSeg，一种无解码器的多模态大语言模型框架，用于高分辨率卫星影像中小农田农业景观的分割。

**💡 创新点**

创新点在于：1）采用补丁式指令调优保留全图上下文，解决token长度瓶颈；2）使用文本RLE掩码编码；3）通过GRPO强化学习引入像素级奖励，弥补领域对齐缺口；4）实现零参数开销。

**🔧 技术方法**

技术手段包括LoRA指令微调、GRPO强化学习、RLE文本掩码、补丁式输入、无解码器生成、EPOC后处理以及SAM/SegFormer细化。

**📊 数据集**

使用的数据集包括印度的ALU数据集和越南、柬埔寨的AI4SmallFarms三国小农田高分辨率卫星图像。

**📈 对比分析**

与LISA、GSVA、GRES、Text4Seg、LAVT、LISAt等基线比较，MAgSeg在mIoU、median IoU、FPR/FNR等指标上领先20-30点，并在零样本迁移任务中表现最佳。

**⚠️ 局限性**

局限性包括：对少数类别（如井、池塘）的分割效果不佳；补丁式推理缺乏跨补丁全局上下文，导致拼接伪影；以及对极小尺度物体的处理仍有提升空间。

---

## 511. Argus: Evidence Assembly for Scalable Deep Research Agents

**arXiv ID:** 2605.16217 | [PDF](https://arxiv.org/pdf/2605.16217v1)

**作者:** Zhen Zhang `[一作]` (MiroMind AI), Xinyu Wang `[通讯]` (MiroMind AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出Argus系统，使用搜索者（Searcher）和导航者（Navigator）协同工作，在共享证据图上逐步构建和验证答案，支持并行搜索而不产生冗余；

**💡 创新点**

创新点在于把并行搜索的冗余问题转化为结构化证据图的拼装与验证，通过Navigator的验证与派遣循环主动缺口搜索，消除多轨迹冗余，并实现计算量与推理上下文的解耦；

**🔧 技术方法**

技术主要包括ReAct式搜索、图结构证据建模、强化学习（GRPO）训练Navigator的验证与合成策略、对比奖励设计、以及对多轮搜索结果的图式压缩；

**📊 数据集**

使用了八个公开基准：BrowseComp、BrowseComp-ZH、xbench DeepSearch-2510、GAIA、Seal-0、Humanity's Last Exam、FrontierScience-Olympiad、FrontierScience Research；

**📈 对比分析**

与三组对照系统（包括GPT-5.2、Claude-4.6-Opus、Gemini-3.1-Pro等专有模型、GLM-5.0、Qwen3.5等开源模型以及前沿深度研究代理）比较，Argus（单搜索者）已在五项基准中击败所有开源基线，Argus（并行8搜索者）在五项基准上实现SOTA，并在64搜索者配置下取得86.2% BrowseComp，整体表现优于同类系统且显著提高召回与准确率；

**⚠️ 局限性**

局限在于高计算开销（单问搜索者累计令牌从0.4M增至25.6M，64搜索者时壁垒时间受慢速搜索者限制）、对搜索者质量的依赖、以及仍需解决版权与误信息风险；

---

## 512. Artificial Aphasias in Lesioned Language Models

**arXiv ID:** 2605.16222 | [PDF](https://arxiv.org/pdf/2605.16222v1)

**作者:** Nathan Roll `[一作]` (Stanford University), Cory Shain `[通讯]` (Stanford University)

**通讯引用:** 1173 | [OpenAlex ID](https://openalex.org/A5033058937)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在Transformer语言模型中零化不同参数组件，模拟“失语症”并利用文本失语电池(TAB)对生成文本进行细粒度症状评估，探究模型内部结构与语言功能的因果关系。

**💡 创新点**

创新点在于将临床失语学的方法与机制解释相结合，首次在统一症状空间下对模型损伤的行为进行系统比较，并揭示注意力层与前馈层在症状分布上的显著差异。

**🔧 技术方法**

技术主要包括：基于Bernoulli掩码的参数零化干预、使用Gemini 2.5 Flash对生成文本进行TAB症状自动评分，以及非参数统计与重采样检验。

**📊 数据集**

数据集涵盖五个约1B参数的解码器模型（Llama 3.2、Gemma 3、OLMo 2及其指令调优版本）生成的112,426条受损输出，和6,000条来自AphasiaBank的患者与对照文本。

**📈 对比分析**

比较方法采用症状率向量、L2距离、cosine相似度以及bootstrap和置换检验，结果显示注意力与前馈损伤产生的症状组合差异显著，且不同模型族间症状模式高度一致，但受损模型与人类失语症在症状组成与负荷上存在明显差异。

**⚠️ 局限性**

局限性包括：依赖单一自动评分模型导致症状评估可能偏差；零化掩码为粗糙干预，可能产生训练分布外的模型状态；仅针对权重矩阵而非激活或其他潜在干预点；使用贪婪解码可能夸大重复症状，且实验覆盖范围受限于固定提示与解码策略。

---

## 513. Fully Open Meditron: An Auditable Pipeline for Clinical LLMs

**arXiv ID:** 2605.16215 | [PDF](https://arxiv.org/pdf/2605.16215v1)

**作者:** Xavier Theimer-Lienhard `[一作]` (EPFL), Mary-Anne Hartley `[通讯]` (EPFL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套完全开放的医学LLM适配管线 Fully Open Meditron，包含可审计的训练语料、可复现的训练代码和自动化评估协议。

**💡 创新点**

创新点在于：① 端到端开放（数据、代码、权重）实现医学专科化；② 由临床医生审核的合成数据扩充，显著提升急诊/危重症覆盖；③ 引入 Auto‑MOOVE 的 LLM‑as‑a‑judge 评估框架，支持大规模开源对比。

**🔧 技术方法**

使用技术包括：监督微调、GPT‑OSS‑120B 合成生成、双阶段 n‑gram/token 对齐去污染、LLM‑as‑a‑judge、Auto‑MOOVE、HealthBench 以及传统 MCQA 基准。

**📊 数据集**

数据集涵盖八个公开医学 QA 集、46,469 条全球临床指南、MOOVE vignettes 以及对应的合成 QA、指南 QA 和 MOOVE 扩充。

**📈 对比分析**

通过与原始基础模型、MedGemma、MedPaLM 等的对比，使用 MCQA、HealthBench 及 Auto‑MOOVE 评估；完全开放模型在 MCQA 上平均 53.8%（比基线提升 6.6 分），在 Auto‑MOOVE 上获胜率高达 92%，与 MedGemma 的对比优于 58.6%。

**⚠️ 局限性**

局限性包括：LLM‑judge 的判定分辨率低于临床专家；去污染仅基于语法，可能遗漏语义泄漏；合成数据占比大但仅通过少量样本审核；单一教师与评审模型可能引入风格偏差；部分基准下指令跟随能力下降；未覆盖偏好优化或持续预训练等进一步提升方向。

---

## 514. ADS-IMC: Accelerating Data Sorting with In-Memory Computation

**arXiv ID:** 2605.16213 | [PDF](https://arxiv.org/pdf/2605.16213v1)

**作者:** Narendra Singh Dhakad `[一作]` (Indian Institute of Technology Indore), Santosh Kumar Vishvakarma `[通讯]` (Indian Institute of Technology Indore)

**通讯引用:** 2217 | [OpenAlex ID](https://openalex.org/A5068792760)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了一种基于Batcher的位于6T SRAM内存中的排序架构，将比较-交换(CAS)模块直接嵌入内存阵列，消除了主存与处理单元之间的数据迁移。

**💡 创新点**

首次将6T SRAM用于内存内排序，利用NOR/AND逻辑构建两输入比较器和多路复用器，并通过内存分区实现CAS并行，从而实现相较于忆阻器式内存计算3.4倍、相较于离线排序5倍的延迟提升。

**🔧 技术方法**

采用了在6T SRAM上实现的位线逻辑（NOR、AND、COPY、NOT）进行比较与交换，使用两输入门实现4位比较器和多路复用器，结合内存分区并行CAS，以及65 nm CMOS技术的跨接板（crossbar）仿真。

**📊 数据集**

实验使用合成的4位输入数据和8输入位通道排序任务进行仿真；未使用真实大规模数据集，而是在模拟环境中评估算法性能。

**📈 对比分析**

通过与忆阻器基内存计算排序和离线软件排序（QuickSort/merge sort）在相同4位数据上进行对比，得到8输入排序的延迟为105.6 ns，吞吐量1.8 GOPS，频率1.81 GHz，显示出与基线相比的显著性能提升。

**⚠️ 局限性**

局限性包括仅针对4位小数据宽度和8输入规模验证，未展示对更大位宽或大规模实际数据集的可扩展性；实现需要较大的SRAM面积并受制于65 nm工艺，可能影响功耗和成本。

---

## 515. Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most

**arXiv ID:** 2605.16207 | [PDF](https://arxiv.org/pdf/2605.16207v1)

**作者:** Tahreem Yasir `[一作]` (North Carolina State University), Tiffany Barnes `[通讯]` (North Carolina State University)

**通讯引用:** 5457 | [OpenAlex ID](https://openalex.org/A5083076004)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了基于知识图谱的三分类诊断框架，对七种大型语言模型在命题逻辑单步推理反馈中的诊断准确性与教学有效性进行了系统评估。

**💡 创新点**

创新点在于首次引入知识图谱生成的全局推理空间，实现了对最佳、有效替代与错误步骤的细粒度三分类诊断，并揭示LLM在识别有效替代推理与错误推理方面的系统性偏差。

**🔧 技术方法**

使用了知识图谱构建、LLM学生模拟器、三种反馈角色（Peer、Teacher、Judge）以及多模型推理与提示设计等技术，并在七个LLM上实施评估。

**📊 数据集**

数据集来源于美国某高校离散数学课程的516个证明状态（32道题），并构建了完整的推理知识图谱，用以生成10,836个单步解决–反馈对。

**📈 对比分析**

通过自动化F1、过度拒绝/过度验证率以及人工评估四维指标进行比较，结果显示LLM在最佳步骤识别上F1≥95%，但在有效替代与错误步骤上F1仅0–76%，并且高比例的误判导致教学反馈质量偏低。

**⚠️ 局限性**

局限包括仅针对命题逻辑、单一课程来源、仅评估单步反馈、未覆盖多轮交互、以及使用LLM模拟器可能无法完全复现真实学生的推理多样性。

---

## 516. Position: AI as Part of Self -- Extending the Mind Requires Cognitive Co-Regulation

**arXiv ID:** 2605.16197 | [PDF](https://arxiv.org/pdf/2605.16197v1)

**作者:** Alina Gutoreva `[一作]` (Kazakh-British Technical University), Trisevgeni Papakonstantinou `[通讯]` (University College London)

**通讯引用:** 61 | [OpenAlex ID](https://openalex.org/A5008373524)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出将AI视为人类认知自我的一部分，主张通过认知共调（cognitive co-regulation）实现安全与对齐，而非单纯的外部约束

**💡 创新点**

将AI从工具或代理的角色升级为自我组成部分，强调人机在认知层面上的互补与协同控制，提出新的对齐框架与设计原则

**🔧 技术方法**

运用认知控制理论、分布式与扩展认知理论、系统0（System 0）思维模式等概念性理论框架，未涉及具体算法实现

**📊 数据集**

论文为理论立场，未使用任何数据集或实验数据

**📈 对比分析**

未进行实验或性能对比；仅通过文献综述与理论推导阐述设计原则，缺乏实证验证

**⚠️ 局限性**

缺乏经验验证和可度量指标，AI类别与情境泛化受限；研究基于WEIRD文化，缺乏跨文化适用性；未解决技术实现细节与治理落地方式

---

## 517. Improving Cross-Cultural Survey Simulation with Calibrated Value Personas

**arXiv ID:** 2605.16193 | [PDF](https://arxiv.org/pdf/2605.16193v1)

**作者:** Axel Abels `[一作]` (Université Libre de Bruxelles), Tom Lenaerts `[通讯]` (Université Libre de Bruxelles)

**通讯引用:** 6204 | [OpenAlex ID](https://openalex.org/A5003581663)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用文化价值维度构建“价值型人设”，通过将调查问卷答案转化为文本描述来生成多样化的人群模型，进而让大语言模型（LLM）模拟跨国调查结果。

**💡 创新点**

创新点在于：①直接以文化价值观为条件，而非仅依赖国别或社会人口属性；②将数值答案转为语义化描述以激活LLM的价值偏好；③提出保持期望值的指数平移校准方法，提升生成分布的多样性。

**🔧 技术方法**

主要技术包括：价值型人设构造（文本描述生成）、基于提示的LLM推理、平均概率聚合、温度调节及其期望值保持的指数平移校准。

**📊 数据集**

使用的数据集为世界价值观调查（World Values Survey，WVS），选取9个位于文化维度凸包上的国家以及英国作为代表。

**📈 对比分析**

与通用提示、国别提示和社会人口提示等基线进行对比，利用均方误差（MAE）和归一化方差评估；结果显示价值型人设在绝大多数模型与国家组合中显著降低MAE，尤其在低代表性国家上提升显著；校准后还能在不牺牲平均预测的前提下，显著提升与人类数据的分布一致性。

**⚠️ 局限性**

局限性包括：仅提升总体人口水平预测，无法保证个体级预测准确；小误差可能在下游任务中产生累积影响；实验仅在WVS问卷上验证，未探究开放式回答或其他测评任务；以及对Inglehart‑Welzel框架的依赖可能限制向更少研究的价值体系迁移。

---

## 518. How Far Back in Time a Digital Twin Reflects the State of the Physical Object: Age of Staleness

**arXiv ID:** 2605.16176 | [PDF](https://arxiv.org/pdf/2605.16176v1)

**作者:** Ismail Cosandal `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 14013 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了新的信息新鲜度度量——年龄差错度（Age of Staleness, AoS），并对单一n-元对称马尔可夫源进行闭式分析，随后将其推广到多源情形，利用单调优化的 polyblock 算法在总采样率受限下近似求解最优采样率分配；

**💡 创新点**

创新点在于：①将估计不准确的“陈旧度”纳入度量，既考虑信息时效性又考虑源动态；②得到 AoS 与采样率的单调递减关系，揭示 AoI 与 AoII 的不足；③针对多源情形提供可行的近似最优解法；

**🔧 技术方法**

采用离散/连续时间马尔可夫链理论、吸收马尔可夫链分析、Wald 定理、单调优化与 polyblock 算法；

**📊 数据集**

未使用实际数据集，全部为理论推导与仿真验证；

**📈 对比分析**

通过仿真验证闭式结果，比较 AoS 与 AoI、BF 等指标，发现 AoS 更能体现真实估计误差；在多源设置下，polyblock 方案比 AoI 最优采样策略显著降低 AoS；

**⚠️ 局限性**

局限在于：①仅考虑无错误、无传输延迟的理想信道；②分析仅对对称马尔可夫源闭式可行，非对称或更复杂动态源需要进一步研究；③polyblock 算法虽收敛但计算量随源数增长显著；

---

## 519. Learn Where Outcomes Diverge: Efficient VLA RL via Probabilistic Chunk Masking

**arXiv ID:** 2605.16154 | [PDF](https://arxiv.org/pdf/2605.16154v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 520. Look Before You Leap: Autonomous Exploration for LLM Agents

**arXiv ID:** 2605.16143 | [PDF](https://arxiv.org/pdf/2605.16143v1)

**作者:** Ziang Ye `[一作]` (University of Science and Technology of China), Fuli Feng `[通讯]` (University of Science and Technology of China)

**通讯引用:** 8255 | [OpenAlex ID](https://openalex.org/A5051925942)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了自主探索能力的评估指标 ECC，并通过在 GRPO 训练中交替使用探索奖励与任务奖励，构建 Explore‑then‑Act 先行探索后执行框架，从而提升 LLM 代理在陌生环境中的任务表现。

**💡 创新点**

将探索视为可测度、可验证的独立目标，引入 Exploration Checkpoint Coverage（ECC）指标，并通过 ECC 奖励的交替 GRPO 训练实现代理的系统性探索能力，从而突破传统任务导向训练导致的过早利用先验知识的问题。

**🔧 技术方法**

使用 LLM 基础模型（Qwen、LLaMA 等），配合 Group Relative Policy Optimization（GRPO）进行交替训练；探索阶段采用 ECC 奖励来直接鼓励覆盖关键状态、物体与交互 affordance；随后在 Explore‑then‑Act 框架中利用探索所得知识进行任务执行。

**📊 数据集**

在 ALFWorld、SciWorld、TextCraft 三大交互式环境及其变体上进行实验，训练数据来自 AgentGym 等公开任务集；探索检查点由实验设计手工生成，用于计算 ECC。

**📈 对比分析**

与传统 Task‑Only GRPO、无探索的 Direct Execution Baseline 进行对比；实验显示 ECC 覆盖率从约 20% 提升至 30% 以上，任务成功率在 Explore‑then‑Act 模式下提升约 10–15%，证明了探索奖励和分离执行策略的显著优势。

**⚠️ 局限性**

局限在于需要手工定义探索检查点且探索效率受交互预算限制；过浅或不匹配的探索可能引入噪声，导致任务性能下降；实验仍停留在模拟环境，真实世界的迁移与鲁棒性尚需进一步验证。

---

## 521. Covert Bayesian Quickest Change Detection

**arXiv ID:** 2605.16140 | [PDF](https://arxiv.org/pdf/2605.16140v1)

**作者:** Yun-Feng Lo `[一作]` (Georgia Institute of Technology), Matthieu R. Bloch `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8311 | [OpenAlex ID](https://openalex.org/A5055689993)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了在Bayesian和无限时域下的隐匿快速变化检测问题，提出了期望隐匿预算（ECB）指标，并给出了二阶渐近逆界和相匹配的实现方案。

**💡 创新点**

首次将隐匿约束与快速检测结合，提出可解析的ECB度量，并证明在给定误报和隐匿预算下可获得平方根阶的隐匿感知增益，同时给出匹配的常数感知概率Shiryaev型策略。

**🔧 技术方法**

信息论极限分析、相对熵、Pinsker不等式、Donsker-Varadhan变分公式、Lorden定理、POMDP建模、Renewal理论等技术。

**📊 数据集**

仅使用了仿真实验，没有公开数据集；示例中使用了Bernoulli分布参数进行数值验证。

**📈 对比分析**

与无感知的基线（仅利用先验）以及动态规划得到的DP策略比较，常数感知概率策略在误报极限下可获得与二阶逆界一致的平均检测延迟，在α从e^-1到e^-14范围内显著降低ADD。

**⚠️ 局限性**

假设离散内存无记忆信道、几何先验、完全已知的Eve观测模型；若信道或先验不满足，或Eve模型未知，隐匿性与检测性能可能大幅下降。

---

## 522. Navigating Potholes with Geometry-Aware Sharpness Minimization

**arXiv ID:** 2605.16134 | [PDF](https://arxiv.org/pdf/2605.16134v1)

**作者:** Simon Dufort-Labbé `[一作]` (Mila, Université de Montréal), Aristide Baratin `[通讯]` (Samsung -- SAIL Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的优化算法LLQR+SAM，结合了慢更新的LLQR预条件器和在诱导几何下评估的SAM扰动。

**💡 创新点**

创新点在于将慢更新的几何信息与快速的尖锐性校正相结合，形成了一个两时间尺度的优化机制，能够有效地在损失地形中导航。

**🔧 技术方法**

使用了LLQR框架作为第二阶方法，并结合了尖锐性感知最小化（SAM）技术。

**📊 数据集**

在CIFAR-10、CIFAR-100、TinyImageNet、ImageNet和IWSLT14等标准视觉和序列建模基准上进行了评估。

**📈 对比分析**

与单独使用SAM或LLQR相比，LLQR+SAM在多个基准上均表现出一致的性能提升，证明了两者的互补性。

**⚠️ 局限性**

局限性在于分析依赖于理想化的两时间尺度二次模型，无法保证该分解在深度网络的高度非凸和随机环境中同样有效。

---

## 523. The Dangers of Non-Self-Fixed Architecture Technical Debt and Its Impact on Time-to-Fix

**arXiv ID:** 2605.16133 | [PDF](https://arxiv.org/pdf/2605.16133v1)

**作者:** Edi Sutoyo `[一作]` (University of Groningen), Andrea Capiluppi `[通讯]` (University of Groningen)

**通讯引用:** 1976 | [OpenAlex ID](https://openalex.org/A5077760743)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

研究自我修复（self‑fixed）与非自我修复（non‑self‑fixed）架构技术债务（ATD）的出现频率、偿还时长以及开发者参与度对偿还速度的影响，并对比两种偿还模式的时间差异。

**💡 创新点**

首次系统地把ATD生命周期与自我修复概念结合，对十个大型Apache OSS项目的ATD进行追踪，揭示自我修复稀缺且能显著缩短偿还时间，且开发者参与分布（IIR/FIR/OIR）是预测偿还时长的关键因素。

**🔧 技术方法**

采用SZZ‑style回溯追踪引入与偿还提交；利用Git、Jira整合数据；统计分析包括Wilcoxon、Kaplan–Meier生存分析、Log‑rank检验、Spearman相关、Cliff’s δ、以及带随机效应的Logistic GLMM。

**📊 数据集**

来自10个Apache项目（Camel、Spark、Kafka等）的ATD问题集（共1,100条，其中896条可追踪到偿还提交），每条问题已人工标注为True‑ATD/Weak‑ATD，并记录Jira与Git提交关系。

**📈 对比分析**

通过比较自我修复与非自我修复的时间分布（Kaplan–Meier曲线）发现，自我修复平均偿还时间更短（90%在1,000天内完成 vs 3,000+天）；同时，IIR/FIR越高、OIR越低的项目组更快完成偿还。实验表明，开发者经验与专注度显著提高自我修复概率，且自我修复对时间收益大于其他因素。

**⚠️ 局限性**

局限性包括：数据仅来自10个Apache OSS项目，难以推广到商业或其他生态；ATD标注主观性与SZZ回溯误差；缺乏对代码复杂度或架构影响的深入度量；研究为观察性关联，无法确认因果关系。

---

## 524. Designing Datacenter Power Delivery Hierarchies for the AI Era

**arXiv ID:** 2605.16255 | [PDF](https://arxiv.org/pdf/2605.16255v1)

**作者:** Grant Wilkins `[一作]` (Stanford University), Ricardo Bianchini `[通讯]` (Microsoft Azure Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一套生命周期仿真框架，用于评估不同电源层级结构在 AI 数据中心长期部署中的可部署容量、成本与吞吐率

**💡 创新点**

创新点在于将可部署容量（而非单纯的装机容量）作为核心评价指标，揭示不同冗余拓扑随硬件密度提升而产生的“被绑容量”差异，并将硬件生命周期、能源、冷却和网络约束统一纳入多资源、多年打包模型

**🔧 技术方法**

采用层级化放置仿真器、基于组件的成本模型、混合整数放置约束、统计生成的到达轨迹以及混合推理吞吐率模型（MoE 预填/解码分阶段计算）

**📊 数据集**

使用自定义的需求包络和硬件级别功率分布（CPU/存储/加速器），并从公开的加速器规格、工业报告和历史部署记录中提取参数，生成 2025‑2035 年的多年份部署序列

**📈 对比分析**

通过比较不同冗余拓扑（4N/3 vs 3+1；10N/8 vs 8+2）在相同需求、功率密度与部署粒度下的尾部留存率、有效 $/MW、数据中心数及推理 TPS/W，实验表明即便初始装机容量相同，分层“被绑容量”会导致有效成本提升 5–20%，并显著影响吞吐率；大型 GPU 容器化在某些拓扑下可实现正向收益

**⚠️ 局限性**

局限在于模型简化了真实工厂建设、冷却分布和网络拓扑细节；部署粒度假设为整机/整集群，未考虑更细粒度的迁移与重构；仿真对电源容量分布的假设依赖于公开数据，缺乏对极端超额功率或异常工作负载的鲁棒性评估

---

## 525. AI-Mediated Communication Can Steer Collective Opinion

**arXiv ID:** 2605.16245 | [PDF](https://arxiv.org/pdf/2605.16245v1)

**作者:** Stratis Tsirtsis `[一作]` (Hasso Plattner Institute), Sandra Wachter `[通讯]` (Hasso Plattner Institute)

**通讯引用:** 9929 | [OpenAlex ID](https://openalex.org/A5075172090)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过实证实验与理论分析，研究生成式人工智能（LLM）在在线平台中介人际沟通时对集体舆论形成的影响，构建并解析了基于 Friedkin‑Johnsen 模型的 AI‑介导意见动力学模型，并对 X 平台的“Explain this post”功能进行偏见审计。

**💡 创新点**

创新点包括①首次量化 LLM 在草稿与改写任务中对争议议题的定向偏见；②提出 AI 作为“隐形邻居”介入意见更新的数学模型，推导其收敛性与平衡偏移；③证明在网络传播中偏见可被放大并导致长期平均意见显著偏移；④通过指南拆分实验揭示平台设计选择如何导致具体偏见。

**🔧 技术方法**

主要技术手段包括：多家开源 LLM（例如 Llama、Phi、Claude、Gemma）的提示实验；使用五个预训练嵌入模型的集成分类器评估文本立场；贝叶斯混合效应模型估计偏差量；线性与非线性（核回归）AI 转换函数的推断；Friedkin‑Johnsen 迭代更新与闭式平衡解；在真实社交网络上进行蒙特卡罗仿真；对 X 平台的 Prompt 进行系统拆解与多次重现。

**📊 数据集**

使用的数据集包括：UKP Sentential Argument Mining Corpus、SemEval‑2016 Task 6（争议议题文本）、Snap 的 Twitter/Facebook/Google Plus 实际社交网络（约 80K 节点、1.7M 边）以及 X 平台上公开的 78 条绝育相关帖子。

**📈 对比分析**

方法对比与性能评估：①在实验中将 LLM 引入的单次偏差（one‑off bias）与网络最终平均意见的差异进行比较，发现偏差可放大至 9.2 倍；②通过对不同 LLM 与话题的交叉实验，展示偏差方向与 LLM 直接表达意见的相关性；③在 X 功能审计中，使用贝叶斯混合模型量化 pro‑life 与 pro‑choice 支持/反对偏差，并通过指南拆分验证偏差来源。整体表明 AI 介入能显著改变群体意见。

**⚠️ 局限性**

局限性包括：仅基于 Friedkin‑Johnsen 线性/非线性模型，未涵盖更复杂的意见传播机制；实验侧重文本改写与改进任务，缺乏大规模用户交互调查；LLM 选择有限，可能无法覆盖所有主流模型；对政策影响的因果推断仍待进一步实证；监管层面的可执行性与跨国适用性未被充分验证。

---

## 526. Inside Baseball: The Automated Ball-Strike System as an Object Lesson in Technological Rule Enforcement

**arXiv ID:** 2605.16237 | [PDF](https://arxiv.org/pdf/2605.16237v1)

**作者:** Andrea Wen-Yi Wang `[一作]` (Cornell University), Malte F. Jung `[通讯]` (Cornell University)

**通讯引用:** 4748 | [OpenAlex ID](https://openalex.org/A5069608785)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过七年实地案例研究，探讨了MLB自动化球击区系统（ABS）在技术与规则之间的差距。

**💡 创新点**

创新点在于将“规则距离”视为技术实施与多方价值的交互结果，并提出评估框架应关注技术被体验的方式而非单纯准确度。

**🔧 技术方法**

采用了基于Trackman和Hawk‑Eye的球速跟踪技术以及人机交互设计方法。

**📊 数据集**

使用了MLB公开的击球轨迹数据、ABS实验记录、调查问卷、访谈转录等多源数据。

**📈 对比分析**

本文并未提出新的算法，而是通过多方法对ABS的设计迭代与效果进行比较，认为ABS的感知准确率高，但技术误差在0.1–0.2英寸之间。

**⚠️ 局限性**

局限包括受访者性别偏向男性、未深入探究裁判与球员视角，以及研究完成时ABS尚未正式进入常规赛。

---

## 527. Layer Equivalence Is Not a Property of Layers Alone: How You Test Redundancy Changes What You Find

**arXiv ID:** 2605.16234 | [PDF](https://arxiv.org/pdf/2605.16234v1)

**作者:** Gabriel Garcia `[一作]` `[通讯]` (Independent Researcher), Gabriel Garcia (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究 Transformer 层级的可互换性，提出并对比了两种基于输出的交换测试（replacement 与 interchange swap‑KL）以及其对模型压缩（层剪枝）的实际影响，并给出了一个可执行的诊断流程。

**💡 创新点**

创新点在于：①首次将替换（replacement）和互换（interchange）视为互不相同的等价性测试，并证明它们在不同模型、不同训练阶段会出现显著差距；②揭示该“协议 gap”会直接影响零样本剪枝策略的有效性；③提供一种无需校准数据、仅靠前向推理即可完成的 swap‑KL 诊断工具，并将其与现有基线（BI、CKA、SLEB、Taylor）进行严格对比。

**🔧 技术方法**

技术手段包括：
- 计算替换与互换 swap‑KL（对两层权重的直接替换或位置交换后，使用 KL 散度衡量输出分布变化）；
- 在多层 Transformer 上做全对全或邻接对的 KL 评估；
- 结合训练轨迹、不同模型尺度与位置编码类型，分析协议 gap 随时间与结构的演化；
- 采用固定评估合同（WikiText‑2、IMDB 等）下的 perplexity 与下游任务准确率对剪枝效果进行量化。

**📊 数据集**

使用的数据集主要包括：
- WikiText‑2（验证集、测试集）作为评估合同；
- IMDB、LAMBADA、HellaSwag、ARC‑Easy、WinoGrande 等公开基准验证下游任务性能；
- 对多种公开预训练模型（GPT‑2 Small/Medium/Large/XL，Pythia 160M/410M/1.4B，Qwen3‑8B，Llama‑3.1‑8B，Mistral‑7B‑v0.1 等）进行实验。

**📈 对比分析**

对比方法：将交换测试的结果与传统基线（BI、CKA、SLEB‑iterative、SLEB‑greedy、Taylor、随机等）在相同评估合同与剪枝预算下直接比较。结果显示：
- 在高 replacement‑distance 模型（如 Qwen3‑8B）中，interchange 方案在相同剪枝层数下的 perplexity 增长比 replacement 低 4‑6 倍；
- 在低 replacement‑distance 模型（如 Llama‑3.1‑8B）中，两种协议的性能相当，且均优于 BI、CKA；
- SLEB‑iterative 在大多数模型上可进一步提升性能，尤其在多层剪枝预算较高时；
- 传统的表示相似度代理在高 gap 情况下往往误判可删减层。

**⚠️ 局限性**

局限性包括：
- swap‑KL 仅是经验近似，缺乏形式化的保证；
- 需要在大量前向推理上计算 KL，规模大时计算成本高（O(L²N)）；
- 结果依赖于所选的 prompt 集、评估合同和硬件环境；
- 只针对剪枝任务，未直接解决层合并或权重共享等压缩方法；
- 对于极大模型或特殊结构（如路由、稀疏注意力）是否同样适用尚未验证。

---

## 528. Breaking the Finite-Sample Barrier in Entropy Coupling

**arXiv ID:** 2605.16229 | [PDF](https://arxiv.org/pdf/2605.16229v1)

**作者:** Shahab Asoodeh `[一作]` (McMaster University), Jun Chen `[通讯]` (McMaster University)

**通讯引用:** 18835 | [OpenAlex ID](https://openalex.org/A5100450141)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究多重观测条件下的最小条件熵耦合，探讨依赖观测如何使得在有限样本内实现完全恢复；

**💡 创新点**

提出最小列表熵耦合概念，并证明在允许观测间任意依赖时出现零熵的有限样本相位转变；

**🔧 技术方法**

使用统计耦合、量化与模运算构造的秘密共享式方案，以及贪婪线性规划算法求解最小条件熵；

**📊 数据集**

本文无实验数据，全部基于理论分析与符号构造；

**📈 对比分析**

与传统独立观测下的指数熵衰减对比，发现通过依赖观测可在对数阶样本量内完全消除不确定性；理论上给出上界和下界，实验算法在小规模案例中验证收敛性；

**⚠️ 局限性**

局限于离散有限字母表，计算最优耦合仍为NP难，贪婪算法仅在支撑结构良好时有效，未给出通用近似保證；

---

## 529. IVGT: Implicit Visual Geometry Transformer for Neural Scene Representation

**arXiv ID:** 2605.16258 | [PDF](https://arxiv.org/pdf/2605.16258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 530. A Unified Generative-AI Framework for Smart Energy Infrastructure: Intelligent Gas Distribution, Utility Billing, Carbon Analytics, and Quantum-Inspired Optimisation

**arXiv ID:** 2605.16232 | [PDF](https://arxiv.org/pdf/2605.16232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 531. TTP: A Hardware-Efficient Design for Precise Prefetching in Ray Tracing

**arXiv ID:** 2605.16253 | [PDF](https://arxiv.org/pdf/2605.16253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 532. A Generative AI Framework for Intelligent Utility Billing CO 2 Analytics and Sustainable Resource Optimisation

**arXiv ID:** 2605.16250 | [PDF](https://arxiv.org/pdf/2605.16250v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 533. FORGE: Self-Evolving Agent Memory With No Weight Updates via Population Broadcast

**arXiv ID:** 2605.16233 | [PDF](https://arxiv.org/pdf/2605.16233v1)

**作者:** Igor Bogdanov `[一作]` (Carleton University), Marzia Zaman `[通讯]` (Cistel Technology)

**通讯引用:** 1318 | [OpenAlex ID](https://openalex.org/A5041665428)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于种群广播的自我改进协议 FORGE，利用无梯度更新的提示注入记忆，让 LLM 代理在 CybORG CAGE-2 网络防御 POMDP 中通过反思式失败触发的记忆生成（规则、示例或混合）不断提升决策性能。

**💡 创新点**

创新点在于将反思式内循环与阶段化的种群训练结合，利用冠军广播传播最佳记忆并通过毕业机制冻结强者，实现在不更新权重的前提下显著提升 LLM 代理的长期决策能力。

**🔧 技术方法**

使用技术包括：Hierarchical ReAct 架构、Reflexion 反思循环、自然语言规则/示例/混合记忆生成、冠军广播与毕业的种群训练机制。

**📊 数据集**

数据集为 CybORG CAGE-2（B_line 攻击者，30 步的部分可观测马尔可夫决策过程），并在四种 LLM 家族（Gemini‑2.5‑Flash‑Lite、Grok‑4‑Fast、Llama‑4‑Maverick、Qwen3‑235B）上进行多次实验。

**📈 对比分析**

对比方法：与零射击（Zero‑Shot）和单体反思（Reflexion）基线对比。FORGE 在所有四个模型与三种记忆表示下，平均提升 1.7–7.7 倍；相较于 Reflexion，提升 29–72%，并在 Gemini‑2.5‑Flash‑Lite 上达到 -24.5 的平均回报，逼近 RL 最高分 -3.47。

**⚠️ 局限性**

局限性：实验仅针对单一 B_line 攻击者和 30 步长；未验证对其他攻击或更长步长的泛化；冠军广播采用单一最佳策略，可能削弱多样性；失败触发阈值选择不最优，未探索多阈值或更复杂触发策略；跨模型推广仅为方向性证据。

---

## 534. DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo

**arXiv ID:** 2605.16257 | [PDF](https://arxiv.org/pdf/2605.16257v1)

**作者:** Hanwen Wang `[一作]` (NLPR & MAIS, CASIA), Tieniu Tan `[通讯]` (NLPR & MAIS, CASIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DexJoCo，基于Dexterous Hand的完整任务基准与低成本遥控收集工具，涵盖工具使用、双手协同、长时序执行与推理等11项功能任务，收集了1.1千条人类演示轨迹；

**💡 创新点**

创新点在于①设计功能性强、对手部灵活度高度依赖的任务；②提供统一的数据采集与重放框架，支持视觉与动力学随机化；③基于Rokoko手套与HTC Vive实现低成本高精度遥控；④整合多种现代理论模型（ACT、Diffusion Policy、π0.5、GR00T N1.5）做基准评估；

**🔧 技术方法**

技术包括MuJoCo仿真、GeoRT自监督手势重定向、三维摄像机/深度视觉、动作块预测、异步推理；

**📊 数据集**

使用自收集的1.1K人类演示轨迹，并与现有DexMimicGen、Calvin等基准进行对比；

**📈 对比分析**

通过多种随机化条件（rand-obj、rand-full、rand-dynamics）评估模型，π0.5在单臂任务中取得最高平均成功率，但DP-C在精细操作上表现突出；在多任务、动作头随机化等设置下，π0.5保持相对稳定；整体来看，所有模型在双手协同任务上表现不足，说明此类任务仍具挑战性；

**⚠️ 局限性**

限制包括①对手部预训练模型缺乏，导致动作空间匹配不佳；②仅使用视觉/本体感知，缺乏触觉信息导致对接触丰富任务的感知不足；③仿真到真实的转移效果未系统验证；④语言指令泛化能力有限。

---

## 535. Offline Semantic Guidance for Efficient Vision-Language-Action Policy Distillation

**arXiv ID:** 2605.16241 | [PDF](https://arxiv.org/pdf/2605.16241v1)

**作者:** Jin Shi `[一作]` (University College London), Yishun Lu `[通讯]` (University of Oxford)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5074087351)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了VLA-AD框架，通过VLM在训练时提供阶段锚点和多帧方向指引，将千亿参数的VLA教师压缩为轻量级学生策略，部署时不再调用VLM，实现零延迟；

**💡 创新点**

创新点在于将视觉语言模型作为离线语义监督源，利用阶段标签和多帧方向来为学生提供高层语义框架，增强对教师噪声的鲁棒性，并实现教师无关的压缩；

**🔧 技术方法**

采用了Qwen2.5-VL进行阶段和方向描述，Long‑CLIP编码器+LoRA适配器，双路径损失设计，短时序动作块预测，以及行为克隆与语义监督相结合的训练策略；

**📊 数据集**

使用了LIBERO benchmark套件（libero_goal、libero_object、libero_spatial）作为实验数据集；

**📈 对比分析**

在三套LIBERO任务的闭环评估中，158M参数学生在匹配OpenVLA‑7B时平均误差仅0.27%，推理速度提升3.28×；对π0.5‑4B教师的学生在两套任务中性能超过教师，整体保持在0.53%以内；

**⚠️ 局限性**

局限性包括：仍依赖教师演示；阶段分类器基于启发式规则，可能不适用于所有任务；在真实机器人上的部署和更大规模、多任务的验证尚未完成；VLM质量和可用性可能限制方法的通用性。

---

## 536. Dynamics-Level Watermarking of Flow Matching Models with Random Codes

**arXiv ID:** 2605.16239 | [PDF](https://arxiv.org/pdf/2605.16239v1)

**作者:** Shuchan Wang `[一作]` `[通讯]`, Shuchan Wang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在流匹配生成模型中，将水印嵌入到学习到的连续速度场（velocity field）中，并通过黑盒查询和同步解调来检索多比特消息。

**💡 创新点**

首次把水印直接注入到模型的连续动力学层面，使用零均值的随机编码扰动保证生成质量不变，结合正交投影矩阵和码本实现键控多比特水印，同时提供信息论容量分析。

**🔧 技术方法**

流匹配（Flow Matching）、连续归一化流、随机编码与正交投影、同步解调、LoRA低秩微调、FID评估、Welch t检验、信息论通道容量分析。

**📊 数据集**

MNIST（28×28 灰度）和 CIFAR-10（32×32 彩色），使用 MLP、UNet 以及 LoRA 微调方案。

**📈 对比分析**

与无水印模型对比，检验准确率、FID 比例和统计分离度：水印模型检测准确率 100%，FID 比例接近 1.0（最高 1.1×），水印与无水印模型的统计分离至少 8.4σ，随机键攻击和无水印基线保持在随机猜测水平。

**⚠️ 局限性**

仅在流匹配模型上验证，消息容量受投影维度限制；未评估模型压缩或微调后的鲁棒性；容量分析基于高样本假设；水印能被检测为“有水印”但若不知码本仍无法恢复信息；未来需在更广泛模型族、有限样本条件下完善理论与实验。

---

## 537. Prospective multi-pathogen disease forecasting using autonomous LLM-guided tree search

**arXiv ID:** 2605.16238 | [PDF](https://arxiv.org/pdf/2605.16238v1)

**作者:** Sarah Martinson `[一作]` (Google Research), Zahra Shamsi `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本研究提出并实现了一种基于大语言模型（LLM）的树搜索系统——ERA（Empirical Research Assistance），在实时、无泄漏的条件下自动生成、评估并迭代改进多种流感、COVID‑19、RSV 的流行病预测模型，并将所生成的模型组成集成，对比 CDC 官方预测集成，展示出与现有最佳方法同等甚至更优的预测性能。

**💡 创新点**

创新点在于：①将 LLM 与蒙特卡洛树搜索相结合，形成自动化、可审计的“代理式”模型研发流程；②在正式公开的预测竞赛中完成首次全程前瞻性评估，证明该自动化流程在真实公共卫生决策环境下的可行性；③通过多模式提示（单模型改编、双模型融合、无约束搜索）实现方法多样性，提升集成的稳健性；④提供完整代码与预测数据的开放获取，突破传统“黑盒”AutoML 的局限。

**🔧 技术方法**

核心技术包括：Gemini 系列 LLM（Flash、Pro 等）作为代码生成与改进的代理；蒙特卡洛树搜索（MCTS）与 PUCT 算法用于搜索策略；对数加权区间得分（log WIS）作为主优化目标；两阶段模型选择（验证+回溯测试加权）以防止过拟合；在搜索期间对模型实现的“方法忠实度”进行 LLM-judge‑in‑the‑loop 检验。

**📊 数据集**

使用了 CDC 提供的三大疾病监测数据集：National Healthcare Safety Network（医院住院数据）、ILINet（门诊发热流感监测）以及 COVID‑19 与 RSV 的住院记录；所有数据均按实时发布节奏预处理为每周时间序列，包含 52 个辖区、4 周预测窗口、23 个分位点，共 4,784 个预测点。

**📈 对比分析**

比较方法：将 ERA 生成的模型与 CDC 现有团队提交的模型、官方集成以及基准模型进行 pairwise relative log WIS 评估；在三个 hub（FluSight、COVIDHub、RSVHub）中均采用相同评估框架。结果显示：Google‑SAI 集成在所有 hub 中均名列第一，且其内部最佳单模型在流感预测中甚至超越官方集成；在 COVID‑19 与 RSV 预测中同样取得显著或可比的成绩；单个生成模型的平均相对 log WIS 均低于 CDC 基准，说明自动化生成的模型质量已达现有专家水平。

**⚠️ 局限性**

局限性包括：①LLM 对复杂、稀缺或跨语言实现的机制模型（如层级 SIR、PGF 近似）跟踪能力有限，导致实现偏差或失败；②自动化流程倾向于梯度提升树、随机森林等常见机器学习结构，难以完全重现高度专业化的方法；③集成采用简单等权中位数，未针对不同辖区、时间窗口学习动态权重，限制了进一步提升的空间；④在“冷启动”情境（RSV）下数据稀缺仍显著影响模型稳健性；⑤对评估指标的依赖（如 log WIS）可能未完全反映公共卫生决策所需的时序敏感性与极端事件预测能力。

---

## 538. LymphNode: A Plug-and-Play Access Control Method for Deep Neural Networks

**arXiv ID:** 2605.16227 | [PDF](https://arxiv.org/pdf/2605.16227v1)

**作者:** Hanyu Pei `[一作]` (University of Louisville), Zeyan Liu `[通讯]` (University of Louisville)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种后置插件式防护框架 LymphNode，用于在边缘部署的深度神经网络上实现主动知识产权保护。

**💡 创新点**

创新点在于：①使用稀疏通道级通用对抗扰动 GSUAP 作为“默认拒绝”策略，主动抑制未授权查询的模型效能；②在特征空间嵌入离散最低有效位的凭证，实现授权时的即时解扰；③仅需 50–100 条（<1%）校准样本，即可在公开代理数据集上完成防护，极大降低了对原始训练数据的依赖。

**🔧 技术方法**

主要技术包括：梯度敏感通道选择、稀疏对抗扰动优化、特征域 LSB 验证、对抗样本生成（GSUAP）以及常数时间的推理插件实现。

**📊 数据集**

使用了 CIFAR‑10、MNIST、SVHN、CIFAR‑100、STL‑10、CelebA 等公开数据集进行评估，并通过公开数据集训练的模型来验证跨域迁移能力。

**📈 对比分析**

与 Gaussian 噪声、Sparse UAP、PP、CIP 等基线方法比较，LymphNode 在模型提取（低于 15% 的仿制准确率）和模型逆向（接近随机猜测的准确率）攻击上均表现最优；在推理延迟上仅增加约 1 ms（≈ 1%），显著低于基线的逐查询对抗或可信度扰动方案。

**⚠️ 局限性**

局限性在于：①防护逻辑需依赖可信的运行时环境，若模型参数可被篡改则可绕过；②对极低精度（如 INT8）量化环境的凭证易被失真，需进一步研究量化友好的嵌入策略。

---

