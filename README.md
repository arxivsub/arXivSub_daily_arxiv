# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-08-03 | 今日论文总数: 437

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Feature Interaction Modeling for Physics-Informed Neural Networks and Neural Operators

**arXiv ID:** 2607.28762 | [PDF](https://arxiv.org/pdf/2607.28762v1)

**作者:** Quan Gu `[一作]` (Independent Researcher), Hongxia Liu `[通讯]` (Taiyuan University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将因式分解机（FM）式的特征交互模块嵌入到物理信息神经网络（PINN）和神经算子（DeepONet、FNO等）中，构建了FM-PINN、FM-Operator、FM-DeepONet三种架构。

**💡 创新点**

创新点是：① 在连续PDE输入上首次使用FM的双交互池化显式建模跨字段（空间、时间、传感器、参数等）的一阶与二阶交互；② 将该交互块分别应用于坐标映射、算子学习与分支‑树结构，显著提升高维、非线性及带尖锐梯度的PDE求解能力；③ 通过共享低维嵌入实现参数高效、可扩展的交互表示。

**🔧 技术方法**

核心技术包括：因式分解机（FM）双交互池化；多层感知机（MLP）解码；物理信息损失（PDE、边界、初始条件残差）；对分支与树网络的特征纠正；对分支-树特征的线性和MLP相加；以及对输入字段的归一化、统计量计算。

**📊 数据集**

使用了多种公开基准数据集：Poisson、Darcy、Burgers、Heat、Convection–Diffusion、Reaction–Diffusion、Linear Advection、Wave、Cubic Conservation、Buckley–Leverett、LWR、Kuramoto–Sivashinsky、参数化 Burgers、Square‑pulse Advection 等，涵盖多维、光滑与尖锐梯度两类PDE。

**📈 对比分析**

实验对比方法：在相同训练配置（AdamW、学习率1e‑3、30k步）下，将FM模型与原始PINN、DeepONet、Shift‑DeepONet、FNO 等基线进行相对 L² 误差比较。结果显示：FM‑PINN 在 16/18 个多维/时空耦合问题上降低误差（最高 55%），FM‑Operator 在含冲击/尖锐梯度的算子学习任务中，使用 43k 参数即可优于 108k 参数的 DeepONet；FM‑DeepONet 通过交互纠正进一步提升性能；在光滑算子学习任务中提升不显著或略逊。

**⚠️ 局限性**

局限性：① 对某些二维 Darcy、Reaction–Diffusion 等光滑 PDE 的提升有限，甚至误差上升；② 交互模块的低秩约束可能导致负迁移；③ 目前实验多为单次/少数种子，需进一步验证稳健性；④ 对输入字段分组与交互选择缺乏自适应机制；⑤ 对非线性强度与跨字段耦合程度的理论分析仍待深入。

---

## 2. Fragility of Value under Imperfect Alignment

**arXiv ID:** 2607.28881 | [PDF](https://arxiv.org/pdf/2607.28881v1)

**作者:** Winter Cross `[一作]` `[通讯]` (Dovetail Research), Winter Cross (Dovetail Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个理论框架，用来描述AI系统在对齐训练后仍可能出现灾难性结果的机制；

**💡 创新点**

在三种不同框架（有限状态、连续状态和属性空间）中给出了灾难性代理的存在条件，首次将“价值脆弱性”与代理条件严格度、优化压力等因素关联起来；

**🔧 技术方法**

主要使用了形式化定义（代理条件、优化器、灾难性价值函数）、概率论推导和几何分析来证明灾难性代理的存在；

**📊 数据集**

无实测数据集，研究完全基于理论模型与抽象实例；

**📈 对比分析**

由于是理论分析，没有实验对比，论文通过构造反例和定理证明来展示结果的普遍性，未给出数值性能指标；

**⚠️ 局限性**

局限性在于：1）仅讨论了存在性而非灾难性代理出现的概率；2）假设的代理条件和优化器较为理想化，缺乏现实世界的具体实现；3）未给出针对不同AI设计（如量化器）在防止灾难性结果方面的定量评估。

---

## 3. Validation Evidence in LLM Repair Agents: How Much of What Passes Actually Tests the Bug?

**arXiv ID:** 2607.28871 | [PDF](https://arxiv.org/pdf/2607.28871v1)

**作者:** Xiaonan Xu `[一作]` (Georgia Institute of Technology), Wenjing Wu `[通讯]` (University of Colorado Boulder)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过BSG‑VA框架对LLM修复代理在轨迹中执行的验证命令进行事件级捕获与重放，量化验证结果的证据价值；

**💡 创新点**

创新点在于提出了事件级证据角色分类与B‑S‑G重放方法，首次揭示正向验证中近一半无效证据的普遍性，并验证了Bug‑Contrast反馈能够显著降低无效闭合；

**🔧 技术方法**

技术手段包括LLM代理+工具循环、事件捕获、测试片段提取、跨状态重放、证据角色分配，以及三臂对照实验（Baseline、静态提醒、Bug‑Contrast反馈）；

**📊 数据集**

使用数据集为SWE‑bench Verified与SWE‑rebench两大Python项目集合，共计约20–30个任务；

**📈 对比分析**

通过任务级配对差异、Welch t检验与Bootstrap等多重稳健性检验，BCF相较提醒可将无效闭合率降低约10个百分点，bug辨别证据提升约11个百分点，官方修复成功率无显著变化；

**⚠️ 局限性**

局限性包括证据角色依赖开发者金手指、单一LLM模型族、Python生态、实验规模有限以及重放成本与实际部署的匹配不确定。

---

## 4. Building a Process-Modeling Tool using Agentic AI: An Experience Report on PM4Py-UCM

**arXiv ID:** 2607.28825 | [PDF](https://arxiv.org/pdf/2607.28825v1)

**作者:** Daniel Amyot `[一作]` `[通讯]` (University of Ottawa), Daniel Amyot (University of Ottawa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文使用Claude Code等LLM编码代理，构建了一款开放源代码的过程挖掘工具pm4py-ucm，能够从事件日志挖掘Use Case Map（UCM）模型并生成可执行场景、性能热图、模型族、仪表板等功能。

**💡 创新点**

创新点在于将UCM作为过程挖掘的首要输出，实现了可执行场景合成、层次化模型分解、动态模型族比较，并提出了基于agent交互记录的可复现性测量与验证方法。

**🔧 技术方法**

技术手段包括Claude Code LLM编码代理、pm4py过程挖掘库、jUCMNavPlus UCM编辑器、Streamlit Web UI、pytest+pytest-cov测试、Radon代码复杂度分析、Bandit安全扫描、import-linter架构检查及pip-audit依赖审计。

**📊 数据集**

数据集主要由18个agent会话转录（374次人机交互）、151次git提交、20个版本发布，以及从医院事件日志中抽取的业务日志（用于模型挖掘和功能验证）。

**📈 对比分析**

通过对比工具的单元测试覆盖率、静态复杂度、可维护性指数以及安全扫描结果，验证了agent驱动开发的可行性；测试覆盖率达90.6%，平均循环复杂度5.6，维护性指数70，未发现中高严重性安全缺陷，表明工具性能可靠。

**⚠️ 局限性**

局限性包括仅为单例案例研究，缺乏对不同工具/开发者/LLM模型的验证，标签标注带有主观性，且仅评估了内部验证与静态分析，未涉及真实业务环境下的性能与可扩展性。

---

## 5. TokenSwap: Benchmarking and Reducing the Modality Gap in Multimodal LLMs

**arXiv ID:** 2607.28640 | [PDF](https://arxiv.org/pdf/2607.28640v1)

**作者:** Andong Hua `[一作]` (University of California Santa Barbara), Yao Qin `[通讯]` (University of California Santa Barbara)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出TokenSwap方法，通过将文本概念替换为语义对应图像，构造图文交叉输入并评估多模态大型语言模型的模态差距。

**💡 创新点**

创新点在于：①首次系统构建图文交叉评测基准TokenSwap-Bench；②用TokenSwap训练显著降低模态差距且不牺牲文本/视觉性能；③发现推理模型模态差距更小。

**🔧 技术方法**

核心技术包括TokenSwap文本-图像替换、语义一致性过滤、图像检索/生成、对比评测与训练增量。

**📊 数据集**

使用的主要数据集为MMLU、DataComp‑Small、IIIT‑5K‑Word等；TokenSwap-Bench基于MMLU生成图文交叉样本。

**📈 对比分析**

通过对42个模型的对比实验，平均模态差距下降19.6%（4.2%–47.4%），推理模型平均差距仅10.1%；TokenSwap训练将模态差距从≈0.268降至≈0.167，文本准确率基本保持或提升。

**⚠️ 局限性**

局限性包括：模态差距受图像质量影响，检索图像易导致差距增大；TokenSwap需在分布内数据上训练，域不匹配时效果有限；未对极大规模模型的训练成本做深入评估。

---

## 6. Open-Source LLM-Driven Formal Verification: A Multi-Agent Pipeline for RTL Repair

**arXiv ID:** 2607.28877 | [PDF](https://arxiv.org/pdf/2607.28877v1)

**作者:** Ha Trung Tran `[一作]` `[通讯]` (Independent Researcher), Ha Trung Tran (Independent Researcher)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套多代理管道，利用大语言模型（GPT‑4o）与开源形式化验证工具（Yosys、SymbiYosys、Z3）通过计数器例证闭环迭代自动修复RTL电路，并在ALU示例上完成功能错误检测、修复与形式化证明。

**💡 创新点**

创新点在于：① 将LLM与全流程开源形式化验证工具结合，实现完全可复现的硬件形式化修复；② 采用多代理架构与类型化属性中间表示，分离逻辑推理与语法生成；③ 通过内联断言注入解决Yosys对bind指令的限制，确保断言被正式检验；④ 对六个基准设计进行多跑实验，首次系统归纳出四类失败模式。

**🔧 技术方法**

使用技术包括：GPT‑4o（LLM）、LangGraph（流程调度）、Yosys 0.61+（RTL合成与展开）、SymbiYosys 0.61（形式化流程管理）、Z3 4.15.5（SMT求解器）以及自定义的属性IR与断言编译器。

**📊 数据集**

使用的基准集为六个RTL模块：counter、alu、arbiter、axi_lite_slave、uart_tx、fifo，每个模块注入单一人工错误，并在五次独立运行中评估。

**📈 对比分析**

通过多跑实验（5 次/基准）比较结果，ALU 能够在两次迭代内（平均 16.5 秒）实现功能修复并通过 k‑induction 形式化证明，其他五个基准均在十次迭代上限内失败；整体发现迭代次数与耗时与设计复杂度及属性数量呈正相关，表现出对深层状态与多属性压力的敏感性。

**⚠️ 局限性**

主要限制包括：仅能成功修复简单组合逻辑（ALU），在计数器深层状态、规格不匹配、时间逻辑与多属性竞争时失效；LLM 在多周期推理与状态记忆方面不足；Yosys bind 指令导致的断言被忽略需手动内联；整体方法受基准规模小与单一模型限制，未证明在更大规模或多种错误场景下的鲁棒性。

---

## 7. Looks Right, Works Right: A Project-Level Benchmark for Multi-Screen Mobile App Generation

**arXiv ID:** 2607.28645 | [PDF](https://arxiv.org/pdf/2607.28645v1)

**作者:** Fan Wu `[一作]` (Harbin Institute of Technology), Qing Liao `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MobileForge项目级多屏移动应用生成基准，包含29个真实应用截图、309个屏幕、701个导航测试，评估构建、导航、视觉保真度、代码可维护性和效率。

**💡 创新点**

首次引入项目级评估、跨屏导航测试、状态隔离导航评测和锚点引用列表式视觉评判，并通过五轴评估框架完整衡量设计到代码的完整过程。

**🔧 技术方法**

使用单代理工具链（八工具）实现端到端代码生成，结合视觉语言模型（Gemini 2.5 Pro）进行视觉评判，采用Playwright自动化进行导航测试，评估使用Borda得分、代码指标等。

**📊 数据集**

MobileForge数据集：29个在市场的消费类移动应用，共309个手工审阅屏幕截图、页面关系描述和701个导航测试规范。

**📈 对比分析**

对六个前沿多模态LLM（Claude Opus 4.6、Claude Haiku 4.5、GPT‑5、GPT‑5 Mini、Gemini 2.5 Pro、Gemini 2.5 Flash）在174个实验中评估，构建成功率100%，导航通过率最高92%，视觉Borda分数最高0.91，GPT‑5在成本与质量的Pareto前沿。

**⚠️ 局限性**

仅单次实验、仅web（React+Tailwind）目标、使用单一视觉评判器、未覆盖原生API、且模型生成的内容可能有文本/图片占位，限制了对真实原生应用的评估。

---

## 8. The Checking Problem: What must be true before AI ships in a regulated firm

**arXiv ID:** 2607.28666 | [PDF](https://arxiv.org/pdf/2607.28666v1)

**作者:** Prerit Ahuja `[一作]` `[通讯]` (Independent Researcher), Prerit Ahuja (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对企业AI项目从演示到生产的差距进行量化评估，测量六类文档工作流程在四个模型家族和三种工具配置下的性能，并计算审核负担。

**💡 创新点**

提出了演示栏和生产栏两套接受门槛，揭示单次演示不足以预测生产成熟度；将审核负担量化为人类需检查输出比例，并用实测数据验证。

**🔧 技术方法**

结合封闭前沿、封闭中等、两种开放权重模型；采用检索‑增量式、引用与置信度标注、以及自我校验通道；使用重复实验测量可复现性，并通过脚本自动化评估。

**📊 数据集**

生成的合成金融文件（覆盖四类工作流程）以及六份真实基金发行文件（来自SEC EDGAR），以保证基准真实度。

**📈 对比分析**

对每种配置对比演示栏与生产栏的合格率，得到56.1%的生存率；准确率可达1.0，但生产栏对可复现、根源性、可检测性要求显著提高；审核负担从100%降至49%（治理配置），自检提升至44%但失误容忍度下降。

**⚠️ 局限性**

接受门槛来源于单一实践者的主观看法；数据集相对干净且规模有限；工具配置仅为三种典型方案，实验样本不具备统计显著性；部分评估基于自动化脚本，未覆盖所有真实运营场景。

---

## 9. Rolling With Resistance: Preference-Optimized LLM Counselors Can Trade Goal Persistence for Relational Attunement in Motivational Interviewing

**arXiv ID:** 2607.28814 | [PDF](https://arxiv.org/pdf/2607.28814v1)

**作者:** Weiying Chen `[一作]` (University of Alberta), Zhexuan Tang `[通讯]` (University of Shanghai for Science and Technology)

**通讯引用:** 180 | [OpenAlex ID](https://openalex.org/A5109078082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在动机访谈(MI)中，研究了如何通过偏好优化训练语言模型以在面对客户的抵触话语时实现“滚动式抗拒”，并探究单侧惩罚哪一种失败模式（抵触导致的放弃或对抗）是否能促进期望的行为，或者导致相反的偏差。

**💡 创新点**

创新点在于：① 将MI中的“滚动式抗拒”拆解为两轴评价（目标坚持Goal Persistence与关系共鸣Relational Attunement）并构建四象限评估框架；② 设计一套只惩罚单一失败模式的直接偏好优化（DPO）数据集，探讨其对两轴的影响；③ 通过“防火墙”三族模型分离评估与训练，验证模型在不同基准上的“抬高一轴同时牺牲另一轴”的稳健性。

**🔧 技术方法**

使用的技术包括：直接偏好优化(DPO)；LoRA参数高效适配；三族模型防火墙（生成器、训练标签评估者、评估评审者分离）；自动评估器（基于MITI构造的两轴评分规则）。

**📊 数据集**

采用公开的AnnoMI专家注释语料库，构建包含可选“成功”回应与两类失败回应（对抗式与屈从式）的对比数据集，按主题分隔保证训练/测试分离。

**📈 对比分析**

比较方法：在三种对齐指令模型（Qwen3-8B、Qwen2.5-7B、Llama-3.1-8B）上使用DPO训练，分别惩罚对抗、屈从或两者混合，并以“pairwise win‑rate”与基准模型做比较；结果显示：惩罚对抗会显著降低目标坚持(GP)而提高关系共鸣(RA)，惩罚屈从几乎无效；基准模型表现差异由其原始失败分布决定。

**⚠️ 局限性**

局限性：① 仅在MI这一特定临床情境下验证，无法直接推广到其他心理咨询风格；② 评估依赖自动评审器，尽管已通过多方验证，但仍存在人类评审一致性不足的问题；③ 对抗与屈从两种失败的比例调节（λ）仅在三模型上检验，未探究更大规模或不同架构的泛化；④ 由于基准模型已倾向于对抗，未能完全展示惩罚屈从时的潜在改进。

---

## 10. Adaptivity via a Parallel Architecture for Stochastic Gradient Methods Adaptivity via a Parallel Architecture for Stochastic Gradient Methods Adaptivity via a Parallel Architecture for Stochastic Gradient Methods

**arXiv ID:** 2607.28902 | [PDF](https://arxiv.org/pdf/2607.28902v1)

**作者:** Bin Fu `[一作]` `[通讯]` (University of Texas Rio Grande Valley), Bin Fu (University of Texas Rio Grande Valley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出一种通过并行执行静态梯度下降实现自适应的框架，利用几何序列搜索合适的迭代预算T，从而在不需要预知L、σ等参数的情况下实现无偏的收敛保证。

**💡 创新点**

创新点在于：①将静态梯度方法通过p个并行进程的几何调度转化为自适应方法；②给出近似因子α_p的上下界，证明其随p趋近1而可任意逼近；③设计了完全基于位移操作的算术简单梯度下降算法，消除除法与平方根。

**🔧 技术方法**

使用的技术主要是：几何调度函数h(j,i)=b_p^{i p + j}T_0、(p,α_p)-近似概念、Lipschitz光滑性与(λ,σ_0,σ_1)-随机梯度模型、解析上界/下界证明以及二进制位移实现的算术简单SGD。

**📊 数据集**

本文未使用具体数据集，全部通过理论分析和收敛证明来展示方法的有效性。

**📈 对比分析**

与传统固定学习率SGD相比，本文方法在理论上达到最优O(1/√T)收敛率；与已有自适应方法相比，能够在保持更简单的收敛分析基础上实现参数自适应，且α_p可控，近似误差随处理器数目减小。

**⚠️ 局限性**

局限性包括：①单处理器情况下α_1的理论上限与实际间仍有较大差距（4 vs 2）；②框架假设能够在每个阶段评估目标函数F的成本较低，实际实现中通信与同步开销可能影响效率；③并未在实验中验证硬件实现的加速效果。

---

## 11. Gated Q-learning: Add Off-Policy Bias to Taste

**arXiv ID:** 2607.28916 | [PDF](https://arxiv.org/pdf/2607.28916v1)

**作者:** Brett Daley `[一作]` `[通讯]` (Meta), Brett Daley (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种名为 Gated 的 Q‑学习框架，通过对多步资格追踪的可调衰减 λ 进行状态‑动作依赖的门控，平滑地介于 Watkins 与 Peng 的极端策略之间。

**💡 创新点**

创新点在于引入连续的、与状态‑动作相关的门控因子 χ，既不需要重要性采样，也不必完全截断追踪，从而实现了在保持较长信用分配窗口的同时可控地加入离策略偏差。

**🔧 技术方法**

技术上使用了可变 λ 的资格追踪、递归与前向返回公式、线性算子与收敛性分析（收缩映射与 Banach 定理）以及理论上的固定点推导，证明了收敛并给出偏差表达式。

**📊 数据集**

实验数据集采用经典的 19 状态随机步行（Random Walk）环境，用于对 λ、α、χ 三个超参数的全面搜索，以评估早期学习速度。

**📈 对比分析**

与 Watkins 与 Peng 的标准实现比较后发现，Gated 在大多数超参数组合下都能更快收敛（AUC 较高），并且能够安全地使用更高的 λ 值；门控参数 χ 在 0.2–0.6 范围内保持性能稳定。

**⚠️ 局限性**

局限性包括：需要在 λ 与 χ 之间手动调参；未与基于重要性采样的多步估计（如 Retrace）进行对比；未评估返回值的方差；在深度 RL 真实任务中的实验与性能验证尚未完成。

---

## 12. WitCert: Sound Runtime Risk Observability and Gating for KV-Cache Quantization

**arXiv ID:** 2607.28699 | [PDF](https://arxiv.org/pdf/2607.28699v1)

**作者:** Fanzhe Wei `[一作]` (Metask Lab), Li Liu `[通讯]` (Metask Lab)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可在线计算的 KV‑cache 量化误差上界（meter），实现了对每个 (layer, head, step) 的总变差（TV）安全上限，并以此为基础在生产系统中实现了自动门控（gate）以避免压缩失效，提升缓存容量并保持生成质量。

**💡 创新点**

创新点包括：
• 设计了两层式证明框架：一层针对任意缓存保持的黑盒量化方案给出确定性 band‑norm witness 上界，另一层针对受控 subtractive‑dither INT8 量化给出子高斯概率上界；
• 通过残差的频带范数实现位置无关的实时 witness，可在写入时一次性计算；
• 在 SGLang 的生产解码管道中无缝集成，实时监测与门控，并通过实验验证门控能在大规模推理中恢复质量。

**🔧 技术方法**

核心技术包括：
• 剩余残差 band‑norm witness 计算（Cauchy‑Schwarz + RoPE 频带不变性）；
• subtractive‑dither INT8 量化与子高斯误差界；
• 证明-内核契约与 Lean 4 形式化验证；
• CUDA‑graph 可捕获、SGLang 的分块量化存储与在线校验；
• 经验门控策略（certified vs risk‑ranked）与增量修复。

**📊 数据集**

使用的模型与数据集：
• 模型：Qwen2.5‑7B、Mistral‑7B、Yi‑1.5‑6B；
• 数据集：RULER（检索任务）、LongBench‑E（多领域长文本）、自然文本、代码、合成检索等；
• 还在 8k‑128k 长上下文、对话、长链推理中测试。

**📈 对比分析**

比较方法：与现有压缩基线（SnapKV、KnormPress、Expected Attention、RTN‑int8/‑int4、KIVI‑2bit）以及无压缩 FP16 进行同一查询下的 TV、页内率、容量、质量指标（top‑1、top‑5、KL、RULER 成功率）比较。性能表现：
• 通过门控把 fp8 质量从 22.8% 恢复到 79.7%；
• 通过子高斯证书将页内率从 44.4% 降至 22.0%，提升 1.88× KV‑token 容量；
• 在 SGLang 生产解码中的运行时开销仅 +11.9%（+0.35% 当存储已量化），短序列下 2–16% 额外开销；
• 在 7B‑15B 规模下，容量增益和页内率改进保持一致，长上下文 128k 下仍保持 31.5% 有效宽度。

**⚠️ 局限性**

局限性：
• 仅对缓存保持的量化方案（不支持在线淘汰/压缩）；
• 概率保证仅适用于非自适应查询，免费自回归解码需额外理论；
• 对于极低位（≤4bit）方案 witness 上界过于宽松，门控失效；
• 目前未验证多 GPU / 张量并行环境下的证书聚合；
• 仅给出每步局部 TV 上界，未给出跨层级的全局误差合成；
• 需要额外的随机数生成器（Philox）与门控逻辑，增加系统复杂度。

---

## 13. Evaluating Federated Pre-Training: On the Reliability of Downstream Fine-Tuning and Intrinsic Evaluation

**arXiv ID:** 2607.28658 | [PDF](https://arxiv.org/pdf/2607.28658v1)

**作者:** Claudia Grosser `[一作]`, Thomas A. Runkler `[通讯]` (Technical University Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对联邦预训练的评估方法进行系统比较，探讨哪种评估协议更能可靠反映模型预训练质量。

**💡 创新点**

首次将下游微调与零样本下的下一个词预测两种评估方式与预训练后的困惑度进行对比，发现后者能更准确保持模型排名；并指出传统的下游微调评估可能掩盖预训练差异。

**🔧 技术方法**

采用16M参数GPT‑2变体，使用FedPop、FedPopHP、FedRS三种联邦超参调优策略，进行联邦与中心化预训练；利用GLUE基准进行下游微调（全参数、只调头部、数据稀疏化）和直接下一个词预测（零样本与继续预训练）。

**📊 数据集**

预训练使用ThePile子集（8类，共105M token），下游评估与内在评估使用GLUE任务集（排除WNLI），GLUE文本亦用于下一个词预测。

**📈 对比分析**

通过Pearson和Spearman相关系数对评估协议与预训练后困惑度的保持情况进行比较：全参数微调相关系数-0.28/-0.25；只调头部0.42/0.38；数据稀疏化90%时-0.56/-0.21；零样本下的下一个词预测0.83/0.93，显示其最能保持预训练模型的相对顺序。

**⚠️ 局限性**

研究规模有限，仅使用16M小模型；仅评估语言任务（GLUE）；未检验更大模型、不同基准或跨模态情形；评价标准基于排名保持，可能无法覆盖所有模型质量维度。

---

## 14. Uncertainty-Aware Deepfake Detection via Multi-View Structural Learning

**arXiv ID:** 2607.28769 | [PDF](https://arxiv.org/pdf/2607.28769v1)

**作者:** Muhammad Umar Farooq `[一作]` (University of Michigan), Khalid Malik `[通讯]` (University of Michigan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多视图结构学习的深度伪造检测框架DISCERN，将视觉、语义和结构三条证据流融合，并通过交叉分支不一致校准（IBDC）实现不确定性感知；

**💡 创新点**

创新点包括：①将多模态证据流（视觉+语义+结构）联合建模；②引入IBDC，将分支间不一致直接映射为预测不确定性；③在结构流中利用基于结构方程的类别依赖关系捕捉伪造特征；

**🔧 技术方法**

技术实现包括：使用冻结的CLIP ViT-L/14作为视觉特征提取器；语义流通过可微分的FACS谓词约束；结构流使用可微分结构方程模型；所有证据通过Evidential Deep Learning（Dirichlet）融合；IBDC通过余弦距离计算分支间不一致并对不确定性进行校准；

**📊 数据集**

主要使用FaceForensics++（c23）作为训练集，交叉测试于Celeb-DF-v2、Celeb-DF-v3、DFDC、DFDC-Preview、DeepFakeDetection等五个未见数据集；

**📈 对比分析**

与13+种基线（CNN、CLIP、图像+结构、基于图的等）比较，在视频级AUC上取得前三名，在校准（ECE）和选择性预测（E-AURC）上显著优于对照组，提升2–3%甚至更高；

**⚠️ 局限性**

局限性包括：依赖多流特征导致模型复杂度和推理时延增加；在高噪声（Gaussian noise）等极端扰动下仍略逊于部分基线；结构学习对不同生成器的适应性尚未完全验证；

---

## 15. An analysis of machine learning approaches for enhancing decision-making in complex discrete choice tasks

**arXiv ID:** 2607.28854 | [PDF](https://arxiv.org/pdf/2607.28854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 16. DART: Dual-Axis Airborne Reachability-Gated Torque-Reaction for Off-Road Vehicle Jumps

**arXiv ID:** 2607.29011 | [PDF](https://arxiv.org/pdf/2607.29011v1)

**作者:** Yu Hu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Baolei Chen `[通讯]` (Dong Feng Off-Road Vehicle Co., Ltd)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种跨阶段的控制框架 DART（Dual‑Axis Airborne Reachability‑Gated Torque‑Reaction），通过在离地前后分别实现可达性门控与基于轮反作用扭矩的双轴姿态控制，解决高速越障后空中姿态不稳定问题。

**💡 创新点**

创新点包括：①基于轮子角动量预算的可达性理论，得到闭式可行起跳集；②将起跳约束向前传播得到保守的 go/no‑go 门控与速度整形；③引入一次性滚转锁存器，实现根据动态需求切换单轴或双轴控制；④在模拟实验中验证了整个跨阶段链条的有效性。

**🔧 技术方法**

采用角动量守恒、可达性分析、闭式可行性证书、轮反作用扭矩控制律、再递归速度整形、转向诱导滚转、BeamNG.tech 全尺度确定性仿真及统计对比分析等技术。

**📊 数据集**

使用 BeamNG.tech 软体仿真平台，构建 1383 kg 四轮驱动平台，生成 12 种不同坡度、落地面及跨越形状的跳跃场景，针对每个场景进行 30 次重复实验以收集数据。

**📈 对比分析**

与时间最优双曲线（TOBB）和反作用轮 PD（RW‑PD）对比，指标为着陆俯仰误差、滚转误差、安全着陆次数。DART 在陡峭坡道下实现 30/30 的安全着陆，速度降低 36%，并在倾斜跑道上平均俯仰误差 ≤ 2°，在大多数测试场景中显著优于两种基线。

**⚠️ 局限性**

局限性包括：1）角动量预算随车辆质量减小；对更大车辆需要重新识别预算；2）目前仅在仿真中验证，缺乏实车测试；3）转向诱导滚转可能导致偏航漂移；4）需要精确的状态估计、地面摩擦与落地几何信息。

---

## 17. Mitigating Class-Tail Undercoverage in Medical Vision-Language Models under Clinical Shift

**arXiv ID:** 2607.28696 | [PDF](https://arxiv.org/pdf/2607.28696v1)

**作者:** Mushir Akhtar `[一作]` (Indian Institute of Technology Indore), M. Tanveer `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计了一种后置可靠性层 CALCoDe，结合类尾部发现、局部自适应的 conformal 置信度阈值与支持审计，解决医疗 VLM 在临床迁移下的类别覆盖不足问题。

**💡 创新点**

创新点在于通过交叉验证发现不足类、独立的类条件阈值与一侧最大合并、支持审计三步实现对置信度的自适应保护，并给出有限样本的保护类覆盖保证。

**🔧 技术方法**

使用的技术包括 APS 非一致性得分、局部 conformal 推断、AUROC 选取支持诊断、剪枝 prompt、加权局部阈值以及一侧最大合并。

**📊 数据集**

实验数据集为两组外部迁移的皮肤病数据集（HAM10000→ISIC 2019 与 HAM10000→PAD‑UFES‑20），并在四种冻结 VLM 后端（BiomedCLIP、OpenAI CLIP、PubMedCLIP、MedSigLIP‑448）上进行评估。

**📈 对比分析**

与 APS、Mondrian CP、Local CP、Conf‑OT、LCP‑VLM、TACP、sTACP、LATA 等 13 种方法对比，CALCoDe 在所有 8 个设置中实现了 ≥0.95 的总体及最差类覆盖；在 ISIC 2019 上最差类覆盖率达 0.970；在 PAD‑UFES‑20 也表现出高覆盖但集大小趋近全集。

**⚠️ 局限性**

局限性包括：仅适用于冻结模型；当表示层在目标域失去区分性时，CALCoDe 只能扩大集而无法恢复诊断分辨率；多样本划分导致稀有类校准样本不足；未在跨模态或多站点环境中验证。

---

## 18. Code Is the Body: Agent-Owned Software Bodies for Recursive Evolution and Descent

**arXiv ID:** 2607.28691 | [PDF](https://arxiv.org/pdf/2607.28691v1)

**作者:** Roy Zhao `[一作]` (University of Washington), Zhenyu Zhao `[通讯]` (Independent researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于可版本化、可拥有的软件主体（agent-owned body）的个人智能体框架——OurArk，支持在人工监督下的自我演化和递归分化。

**💡 创新点**

创新点在于：① 把智能体的行为定义与身份绑定于可版本化的代码仓库，而非单一模型或平台；② 通过自我演化（candidate change → validation → 人工审核）与递归分化（descendant inherit body, create new identity）实现协同进化与独立专化；③ 引入分层边界（body、state、reasoner、custodian），实现持久身份与可替换推理服务的分离。

**🔧 技术方法**

使用技术包括：Python、Git 版本控制、GitHub PR 工作流、单元测试框架、Genesis 创建引擎、Enoch 代理实现、对话接口（Telegram/Codex/ChatGPT）以及标准化的 repo 结构和 manifest 机制。

**📊 数据集**

未使用公开数据集；所有验证均基于内部 mock 服务与自定义的回归测试集（涵盖功能、继承、学习、演化、迁移等）。

**📈 对比分析**

通过对比 4 代理、3 次递归分化的可重复测试，验证了：① 递归分化成功率 100%；② 继承的测试套件完整通过（753/753）；③ 失败更新能够回滚且无残留错误；性能表现以 CI 通过率和测试耗时为指标，未发现明显瓶颈。

**⚠️ 局限性**

局限性包括：① 仅验证单条线性分化线，未评估多分支或大规模协同；② 依赖 Git 作为持久化手段，未验证其它存储后端；③ 仅对 mock 推理服务做测试，未测量推理模型迁移对行为连续性的影响；④ 代码变更审核依赖人工，未评估审核成本与效率；⑤ 部分字符串替换逻辑对复杂语法的正确性保障不足。

---

## 19. Reflection or Re-Generation? Why LLM Revision Fails Where Human Revision Succeeds

**arXiv ID:** 2607.28908 | [PDF](https://arxiv.org/pdf/2607.28908v1)

**作者:** Yefan Tao `[一作]` (Amazon), Luyang Kong `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `2704f255-0c84-4173-b83c-0e9a3dbea232` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了人机反思框架（HRF），对比人类与大语言模型在两轮反思中的表现，并通过信息论分析揭示LLM反思的两种失败模式。

**💡 创新点**

创新点在于构建统一的两轮反思实验框架、引入信息理论度量、发现LLM反思表现为条件再生成并呈现客观任务无信息增益、主观任务信息损失两种失败模式。

**🔧 技术方法**

使用信息理论指标（交叉熵下降/信息增益）、错误检测与oracle-guided修正实验、交叉代理矩阵、多模型与多轮对比等技术手段。

**📊 数据集**

采用 MalAlgoQA（4选项数学推理）、IMDb‑Rating（1–10情感评分）以及 TISER（时间推理）三类任务进行实验。

**📈 对比分析**

通过让人类和LLM在相同提示、输入下完成两轮输出，计算反思收益（RG）和信息增益；实验结果显示人类在所有任务均取得正收益，而LLM在客观任务无显著提升、在主观任务呈负收益。

**⚠️ 局限性**

局限性包括：信息论分析仅适用于有限标签空间任务，开放式生成难以量化；人类评标为非专业工人，可能影响结果；实验结果受提示设定影响；Oracle实验仅提供二元错误信号，未涵盖所有错误类型。

---

## 20. Sovereign Cognitive Digital Twins: Fusing 6G ISAC, AI-RAN, and Zero-Trust Edge Grids for National Resilience in the Global South

**arXiv ID:** 2607.28756 | [PDF](https://arxiv.org/pdf/2607.28756v1)

**作者:** Zoe Aiyanna M. Cayetano `[一作]` (Amini), Taijuo T. Morris `[通讯]` (Amini)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了面向小岛屿发展中国家的主权认知数字孪生（S-CDT）框架，利用6G ISAC 将网络转化为感知系统，并实现了基于CPU的可复现的定量射线追踪传播模型（Barbados案例），同时设计了零信任安全、隐私分层与闭环AI‑RAN控制的治理层。

**💡 创新点**

创新点在于：① 将 6G ISAC 视作国家级传感网络（NaaS），打破传统观测系统的硬件壁垒；② 设计了联邦式主权数字孪生体系结构，整合建成环境与地球系统数字孪生；③ 引入信念状态 EKF+PPO 的闭环控制，以在 O‑RAN 延迟下保持决策可靠性；④ 在物理层实现零信任子系统分离与隐私分层波形；⑤ 提供完整的 CPU‑only 递归射线追踪流水线，保证在本土硬件上可自给自足。

**🔧 技术方法**

技术手段包括：6G ISAC/DFRC、O‑RAN（Near‑RT & Non‑RT RIC）、扩展卡尔曼滤波（EKF）、近端策略优化（PPO）、Sionna RT + Mitsuba ray‑tracer、Blender 3D 场景构建、GeoPackage/PMTiles 数据存储、3GPP TR 38.901 与 ETSI GR ISC 001 标准、联邦学习、AI‑RAN、跨层异常检测、隐私分层 ISAC 波形。

**📊 数据集**

使用的数据集有：Barbados 官方 Geoportal（19 ESRI shapefile）提供建筑、道路、供水、通信基站、危害网格；LiDAR 高程与材料标签；卫星基底地图（Sentinel‑2、Copernicus WorldCover）；实时馈送（3D‑PAWS、OpenSky、CelesTrak、OpenCellID、PeeringDB 等）以及标准化的 3GPP 频段与天线模型。

**📈 对比分析**

对比方法：将定量射线追踪与 2‑ray 解析模型、平地/地形/地形跟随三种场景、以及 3GPP TR 38.901 随机规划表面进行对照。结果显示：射线追踪在路径损耗上与解析模型差距 ≤0.1 dB；与 TR 38.901 的差异在 19–20 dB，服务单元切换率提升至 29%；CPU‑only 流水线在 Raspberry Pi 4 上完成 310 s，服务器 26 s，内存 <1.3 GB，所有数值在不同硬件上均保持一致。

**⚠️ 局限性**

局限性包括：① 未进行现场校准，传播模型与实际测量之间的误差未验证；② ISAC 感知、EKF/PPO 控制循环尚未部署或评估；③ 跨层异常检测、隐私波形权衡未实现；④ 仅在 Barbados 单岛案例验证，未展示大规模群岛的可扩展性；⑤ 关键静态层（政府 Geoportal）不可公开再分发，限制了复现与迁移。

---

## 21. Identifying Informative Environments for Cognition Parameter Inference via Bayesian Experimental Design

**arXiv ID:** 2607.28894 | [PDF](https://arxiv.org/pdf/2607.28894v1)

**作者:** Manisha Dubey `[一作]` (University of Edinburgh), Subramanian Ramamoorthy `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将实验环境的选择视为贝叶斯实验设计（BED）问题，构建了精确的蒙特卡洛BED基准，并提出了利用神经网络近似后验的“可摊销贝叶斯实验设计”（ABED）框架，应用于Mouselab-MDP过程追踪实验来寻找最具信息量的决策环境。

**💡 创新点**

创新点在于：① 把实验环境本身作为设计变量进行贝叶斯实验设计；② 通过可摊销后验网络将昂贵的后验推理替换为一次前向传播，大幅降低计算成本；③ 证明不同的环境在信息增益、后验可恢复性和实验效率上存在权衡，单一环境并非普适最优。

**🔧 技术方法**

使用的技术包括贝叶斯逆规划、贝叶斯实验设计、蒙特卡洛模拟、基于KL散度的期望信息增益估计、可摊销后验网络（神经网络逼近后验），以及软最大策略生成轨迹。

**📊 数据集**

数据集主要为Mouselab-MDP的模拟轨迹，构造了多种树深度（d∈{2,…,6}）和分支因子（二叉、三叉）环境，生成不同计划深度和工作记忆容量的轨迹，用于训练与评估可摊销后验模型。

**📈 对比分析**

与精确贝叶斯推理、蒙特卡洛BED以及结构/行为启发式、监督回归等基线进行比较。ABED在保持0.905的Spearman相关性、零相对差距（regret）上与MC‑BED一致，同时将计算成本降低数十倍；相对基线方法在环境排名和信息增益预测上更优。

**⚠️ 局限性**

局限性包括：实验设计空间受限于二叉/三叉树结构，未在真实受试者数据上验证；可摊销后验在某些指标上仍不完美；未实现真正的自适应实验选择，仅提供了框架和组件。

---

## 22. ThinkReset: Learnable Intermediate Interface Construction for Bounded-Context Long-Horizon Reasoning

**arXiv ID:** 2607.28642 | [PDF](https://arxiv.org/pdf/2607.28642v1)

**作者:** Fei Ding `[一作]` (Alibaba Group), Zijian Zeng `[通讯]` (Tsinghua University)

**通讯引用:** 454 | [OpenAlex ID](https://openalex.org/A5101774556)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种在固定上下文窗口下，通过文本空间写回中间接口并重置上下文来实现长链推理的框架。

**💡 创新点**

把长推理问题重新表述为“中间接口学习”，直接优化接口能否支持后续推理，而非仅压缩或保持轨迹。

**🔧 技术方法**

采用三阶段训练（SFT+RLOO）与离线强化学习，写回与重置机制在文本空间实现，奖励为后续连贯推理成功率。

**📊 数据集**

使用 DeepMath-103K 等数学/逻辑基准，包括 AIME 2024/25、ZebraLogic、AutoLogi、GPQA-Diamond。

**📈 对比分析**

与轨迹保留/压缩方法（TokenSkip、Halo 等）对比，采用 Avg@8 成功率评估，8B 模型平均提升约 4–5%，在多任务上获得显著提升。

**⚠️ 局限性**

仅验证在数学/逻辑任务，需冷启动 SFT，触发阈值固定，最多两次重置，缺乏更深层级接口和更广泛场景验证。

---

## 23. Optical Flow Sensor: A Direction-Selective Bionic Retina Design

**arXiv ID:** 2607.28686 | [PDF](https://arxiv.org/pdf/2607.28686v1)

**作者:** Juchen Zhou `[一作]` (Peking University), Yuchao Yang `[通讯]` (Peking University)

**通讯引用:** 15554 | [OpenAlex ID](https://openalex.org/A5057584787)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种像素级光流传感器（OFS），实现芯片内事件驱动的光流计算，并提供专用的地址事件表示接口（OF-AER），以及基于光学忆阻器的低功耗变体。

**💡 创新点**

创新点在于将ON/OFF事件比较与时间差测量结合，实现每个像素的并行光流计算；使用专门的OF-AER接口实现稀疏高效读出；引入光学忆阻器显著降低面积和功耗。

**🔧 技术方法**

技术包括动态视觉传感器（DVS）事件编码、光流方程离散化、时间到数字转换器（TDC）、地址事件表示（OF-AER）以及光学忆阻器芯片。

**📊 数据集**

实验使用公开的DVS09数据集和MVSEC数据集进行功能验证和性能评估。

**📈 对比分析**

通过与传统DVS+FPGA加速系统对比，OFS在功耗上下降303倍、输出数据带宽降低约3.3倍、实现微秒级延迟；峰值事件吞吐量达128 Meps，噪声抑制系数可达188–259倍。

**⚠️ 局限性**

局限性包括光学忆阻器版OFS的响应延迟达10 秒；2位TDC分辨率限制精度；行扫描读出瓶颈导致最大事件吞吐受限。

---

## 24. Imbalanced Data Clustering via Targeted Data Augmentation Using GMM and LLM

**arXiv ID:** 2607.28635 | [PDF](https://arxiv.org/pdf/2607.28635v1)

**作者:** Noor Khalal `[一作]` (Université Paris Cité), Mohamed Nadif `[通讯]` (Université Paris Cité)

**通讯引用:** 2616 | [OpenAlex ID](https://openalex.org/A5007054746)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种结合高斯混合模型（GMM）与大型语言模型（LLM）的无监督文本聚类数据增强方法，用于解决不平衡数据集的聚类问题。

**💡 创新点**

创新点在于通过GMM定位稀疏簇，再利用LLM生成主题相符的合成文本，实现对少数类的精准、无监督增强，区别于传统全局或需标签的增强技术。

**🔧 技术方法**

技术细节包括Transformer嵌入、UMAP降维、GMM+EM聚类、体积/比例比率筛选、基于高斯采样生成合成点、LLM文本生成、KeyBERT关键词提取，以及KMeans/SphericalKMeans评估。

**📊 数据集**

使用的文本数据集有 Arxiv、Biorxiv、Medrxiv、Reddit 以及 Tweet Emotion，全部为不平衡文本集。

**📈 对比分析**

与原始数据集对比时，使用 KMeans 与 SKMeans 进行聚类，评估指标为 NMI 与 ARI；实验结果显示增强后指标不低于原始，往往提升，保持或改善聚类质量。

**⚠️ 局限性**

局限性包括：LLM 生成的文本可能带来训练数据中的偏见；高维低方差嵌入导致协方差矩阵奇异，需要正则化；实验仅覆盖文本聚类，其他任务需进一步验证。

---

## 25. Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

**arXiv ID:** 2607.28685 | [PDF](https://arxiv.org/pdf/2607.28685v1)

**作者:** Youting Wang `[一作]`, Bowen Liu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对四个主流代理安全基准（R-Judge、InjecAgent、AgentHarm、AgentDojo）进行系统评估，探讨其构念效度、指标效度与准则效度。

**💡 创新点**

创新点在于揭示 F1 评分可被“始终不安全”基线游戏、指出能力混杂对指标的影响、并通过多模型、多组织数据验证安全基准间的互相不一致性。

**🔧 技术方法**

采用 Spearman/partial Spearman 相关、并行分析（PCA）、留一交叉验证、组织聚类自举等统计方法，并开发 API‑only 审计工具。

**📊 数据集**

使用 22 个模型（来自 9 家组织）在四个安全基准上重新运行得分，并测量 MMLU、GPQA‑Diamond 作为能力锚点，外加三项保留准则（任务成功、误对齐安全、越狱安全）。

**📈 对比分析**

结果显示：R-Judge 的 F1 能让“始终不安全”基线跑赢五个真实模型；不同基准对同一模型排序完全不一致；能力与安全指标相关但不一致，AgentHarm 与越狱安全在控制能力后相关系数达 0.72，显示最强的准则效度。

**⚠️ 局限性**

主要局限包括面板规模有限、能力锚点为静态知识测试、指标的单向性（如 F1 忽略真负）、多比较未严格校正、以及所用准则与实际部署危害概率的脱节。

---

## 26. ViSAGE: Constructing Self-Correcting Memories for Long-Form Video Understanding

**arXiv ID:** 2607.28678 | [PDF](https://arxiv.org/pdf/2607.28678v1)

**作者:** Xinkui Zhao `[一作]` (Zhejiang University), Yueshen Xu `[通讯]` (Xidian University)

**通讯引用:** 2987 | [OpenAlex ID](https://openalex.org/A5057911001)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了视觉自修正的多模态记忆框架，利用跨模态绑定、序列级实体锚定、双向记忆修正和多代理交叉验证，实现长视频中实体一致性、时间连贯的推理；

**💡 创新点**

创新点包括：①通过跨模态绑定和序列级实体锚定解决身份碎片化；②双向记忆修正实现历史记录的自我纠正；③多代理交叉验证与身份‑证据对齐机制保障推理安全、可拒绝回答；

**🔧 技术方法**

使用多模态大语言模型（Gemini‑3‑Pro）、多目标跟踪、活跃说话人检测、音频‑视觉绑定、双向检索与修正、身份解析器及Judge Agent；

**📊 数据集**

实验数据集包括M3‑Bench‑robot、M3‑Bench‑web、Video‑MME‑long等长视频基准；

**📈 对比分析**

与Socratic Models、Online Video Understanding Methods及M3‑Agent等基线对比，整体准确率提升约5.9%，在M3‑Bench‑robot/ web 上分别提升至约62%/71%，在Video‑MME‑long 上达到79.1%，在跨模态与人物理解等指标上显著优于现有方法；

**⚠️ 局限性**

局限性主要在于仅针对人类实体，非人类物体、动物等未被主动追踪和记忆，依赖现有检测/识别工具，尚未覆盖更广泛的实体类型。

---

## 27. Unanticipated Effects of Generative AI on Expertise Pathways and Performance Perception in System Administration

**arXiv ID:** 2607.28650 | [PDF](https://arxiv.org/pdf/2607.28650v1)

**作者:** Rana Abou Khamis `[一作]` (Carleton University), Ashraf Matrawy `[通讯]` (Carleton University)

**通讯引用:** 1813 | [OpenAlex ID](https://openalex.org/A5103160400)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对14名系统管理员和IT专业人士的半结构化访谈，研究了生成式人工智能（GenAI）在日常系统管理工作中的影响。

**💡 创新点**

创新点在于揭示了GenAI压缩传统专业成长路径、并通过加速完成任务改变组织与个人对绩效的认知，形成两种新的社会技术效应。

**🔧 技术方法**

主要采用访谈与Braun‑Clarke方法的反思性主题分析，结合人工智能工具生成的脚本与配置作为案例。

**📊 数据集**

使用的数据集为14份访谈记录（约45分钟每份），涵盖从2至15年的系统管理经验。

**📈 对比分析**

本研究为探索性定性分析，没有对比实验或量化指标；结果基于主题频率与访谈者自述，无法提供客观性能测评。

**⚠️ 局限性**

局限性包括样本规模小、地域与行业相对单一、缺乏纵向跟踪与跨文化验证，且未考虑GenAI使用过程中的技术细节与安全风险。

---

## 28. From Process to Evidence: How Computing Can Ground Appropriate Reliance on Legal AI

**arXiv ID:** 2607.28869 | [PDF](https://arxiv.org/pdf/2607.28869v1)

**作者:** James Bryan Williams `[一作]` `[通讯]` (New York Supreme Court Appellate Division), James Bryan Williams (New York Supreme Court Appellate Division)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对纽约法院系统中关于AI使用的法律职责进行了分析，映射为HCI中的适当依赖概念，并提炼了评估基础设施的需求，探讨了现有替代方案，并提出了计算机社区的研究议程。

**💡 创新点**

创新点在于将法律职责与HCI的适当依赖模型对齐，系统性地从官方记录中提炼评估需求，揭示法律系统用流程代替证据的做法，并为构建法律AI评估基础设施制定了详细需求与研究路线图。

**🔧 技术方法**

本文主要采用文献综述与需求工程方法，对法律文件进行分析与归纳，并结合HCI理论进行映射。

**📊 数据集**

未使用具体数据集，本文基于公开法律文件、政策文件和学术文献。

**📈 对比分析**

未进行实验或性能评估，主要通过对比分析与理论阐述说明当前缺乏共享评估基准。

**⚠️ 局限性**

局限性在于研究仅聚焦于纽约州的官方记录，缺乏跨司法管辖区的验证；未给出具体实现或实验结果，且所提出的评估体系仍需后续实证与跨学科合作验证。

---

## 29. Costs of Arbitrary Real Matrix Factorizations for Pure-DP Continual Counting

**arXiv ID:** 2607.28703 | [PDF](https://arxiv.org/pdf/2607.28703v1)

**作者:** Awnon Bhowmik `[一作]` (Colorado Technical University), Mahmudul Hasan `[通讯]` (University of Dhaka)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文研究了在纯-differential privacy下，Laplace矩阵机制的因子分解成本，证明了在没有符号、稀疏性或方形限制的情况下，因子分解成本的渐近行为。

**💡 创新点**

创新点在于为任意实数因子提供了因子分解成本的下界，填补了Arkhipov和Kalinin提出的开放问题的空白。

**🔧 技术方法**

使用了经典的近似空间转换理论和p-核算子理论，结合了Fenwick区间因子分解。

**📊 数据集**

使用了下三角前缀和矩阵T_n作为数据集，研究了其在纯-differential privacy下的表现。

**📈 对比分析**

与Arkhipov和Kalinin的结果进行了比较，证明了在纯-differential privacy Laplace矩阵机制类中，优化后的最大和均方误差均为Θ(-2log^3(n+1))，优于之前的Ω(-2log^2 n)的下界。

**⚠️ 局限性**

限制在于该结果仅适用于矩阵机制，不适用于非矩阵的持续机制，也未涵盖近似-DP敏感性或坐标间的期望最大值。

---

## 30. Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications

**arXiv ID:** 2607.28840 | [PDF](https://arxiv.org/pdf/2607.28840v1)

**作者:** Burak Payzun `[一作]` (Prometeia), Seçil Arslan `[通讯]` (Prometeia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了金融领域大语言模型（LLM）系统级验证的完整框架，强调仅靠基准分数不足以确保系统安全、可审计与合规；

**💡 创新点**

创新点在于将验证拆分为五个互补支柱（数据、模型设计、性能、治理与工具使用、IT架构）并引入混合评估方法，包括LLM-judge、灰盒/白盒轨迹分析与生命周期监管；

**🔧 技术方法**

使用的方法包括量化指标评估、LLM-as-a-judge评估、检索与生成分离评估（RAGAS）、代理轨迹日志分析、监管合规映射以及持续生命周期监控；

**📊 数据集**

文中引用的主流金融基准数据集有FinBen、FinQA、ConvFinQA以及金融机构内部自建测试集，但未提出新的公开数据集；

**📈 对比分析**

对比方式主要是基准与系统级验证的对照，指出基准分数在检索错误、事实不准、工具误用、权限越权等场景下表现不佳；在真实环境中未给出定量性能指标，更多是概念性阐述；

**⚠️ 局限性**

局限性包括：缺乏统一的系统级评估指标与标准、LLM-judge可靠性与偏差问题、代理轨迹评估工具成熟度不足、以及监管框架在不同地区与业务场景下的适配难度。

---

## 31. CyberNeuro: A Privacy-Preserving Agentic Workbench for Cohort-Scale Neuroimage and Clinical Data Analysis

**arXiv ID:** 2607.28841 | [PDF](https://arxiv.org/pdf/2607.28841v1)

**作者:** Ran Ren `[一作]`, Guorong Wu `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了本地隐私保护的Agent工作台CyberNeuro，用自然语言自动化大规模神经影像与临床数据分析。

**💡 创新点**

创新点在于结合本地7B LLM WandaMind与四个专用Agent实现无网络、全流程自动化、可验证、可追溯的工作台。

**🔧 技术方法**

技术包括本地LLM微调、四阶段Agent架构(Planner, Validator, Dispatcher, Reporter)、Model Context Protocol桥接、WSL容器化、可视化交互与手工验证界面。

**📊 数据集**

使用多模态临床数据：DICOM/BIDS、T1、fMRI、dMRI等公开与NIH cohort，如ADNI、AIBL、BLSA等。

**📈 对比分析**

与NeuroClaw比较，CyberNeuro在DICOM→BIDS转换中减少9.3× token消耗、0人干预、19%运行时间缩短；在可视化与QC任务中把交互和时间降低99%以上；整体保持同等完成率。

**⚠️ 局限性**

局限包括技能库规模有限（≈30 vs 85）、LLM局部模型在长上下文推理上逊色、可视化工具覆盖不全以及需要进一步完善QC与多用户功能。

---

## 32. An Ontology-Guided, Deduplication-Aware Extraction Layer for Knowledge Graph Construction from Heterogeneous Documents

**arXiv ID:** 2607.28662 | [PDF](https://arxiv.org/pdf/2607.28662v1)

**作者:** Vaibhav Dangaich `[一作]`, Kundeshwar Pundalik `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一套实时Kafka消费、跨格式（PDF、Excel、Word、PPT、图像）并行处理的知识图谱提取层，利用本体检索驱动的LLM提示实现类型一致的实体与关系抽取，并在抽取后通过六阶段去重与实体解析提升质量。

**💡 创新点**

创新点主要包括：① 基于Neo4j图数据库的实时向量检索，仅检索相关本体子集，提示开销减少94%；② 通过六个无推理成本的规则去重算法与语义扩展，彻底消除实体混合和关系重复；③ 每页OCR分类与混合PDF路由，实现既不丢失扫描页也不浪费计算；④ 关系第二轮与多信号实体匹配、上下文验证的组合，实现高召回且无误合并。

**🔧 技术方法**

技术栈涵盖：本体向量检索（Neo4j向量索引+PageRank惩罚）、LLM（本地4‑bit AWQ Qwen3.5‑9B via vLLM、可切换至Gemini）、Python异步事件循环与多工线程、FAISS块索引、Double Metaphone、Jaro–Winkler、Embedding（Qwen Embedding / OpenAI）、多格式解析库（PyMuPDF、Docling、Pandas、pdfplumber）等。

**📊 数据集**

实验数据来源于情报领域混合文档集（PDF、Excel、Word、PPT、图像），并使用对应的本体和人工标注的实体/关系作为评估基准。

**📈 对比分析**

评估指标包括检索召回率和误合并率：在情报语料上，召回率从约70%提升至95%，误合并率保持0%；本体切片与完整提示对比，提示大小下降约94%，相当于16×更小的上下文；混合PDF路由显著减少信息丢失。

**⚠️ 局限性**

局限性包括：① 对未匹配的实体/关系仍需人工复审；② 嵌入检索对跨文档语义匹配仍受限；③ 对多语言/异体命名的覆盖率不完全；④ 高并发下GPU内存压力可能导致临时失败；⑤ 本体更新需重建索引，影响实时性。

---

## 33. Learning to Predict Performance-induced Emotion Differences in Classical Piano Music

**arXiv ID:** 2607.28876 | [PDF](https://arxiv.org/pdf/2607.28876v1)

**作者:** Joann Ching `[一作]` (Johannes Kepler University), Gerhard Widmer `[通讯]` (Johannes Kepler University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用演奏者特有的四个性能特征，对巴赫《平均律钢琴曲集》录音的情绪表现进行相对回归预测，提出 Delta-VA 框架。

**💡 创新点**

创新点在于只用演奏相关特征而非作曲结构特征，预测相对情绪偏移而非绝对位置，并引入几何评估指标来衡量演奏间情绪差异的方向性。

**🔧 技术方法**

使用性能编码（Beat Period、Velocity、Timing、Articulation）通过最新的音频转 MIDI 模型 Transkun 提取特征，构建线性、kNN、MLP、Rank-based 和 Delta-VA 回归模型。

**📊 数据集**

采用 CP-WTC 数据集：J.S. 巴赫《平均律钢琴曲集》第一卷，48 首曲子，6 位著名钢琴家共 288 段 8 小节的情绪标注。

**📈 对比分析**

与传统绝对回归模型相比，Delta-VA 在 MSE 和 R² 上取得最高分（R²≈0.786），在相对方向评估中平均角度误差仅为≈8°，但在绝对偏差评估中角度误差高达≈88°；表明模型能很好捕捉演奏间情绪变化方向但幅度被压缩。

**⚠️ 局限性**

主要局限是预测的幅度往往被压缩，模型仅能描述相对偏移，无法给出绝对情绪坐标；此外仅在钢琴古典曲目上验证，缺乏跨乐器、跨风格的通用性。

---

## 34. Can AI Evaluate AI Scientists? A Benchmarking Study of Autonomous Research Generation Systems Using Automated Multi-Model Review

**arXiv ID:** 2607.28631 | [PDF](https://arxiv.org/pdf/2607.28631v1)

**作者:** Vaibhava Lakshmi Ravideshik `[一作]` (GRAIL), Mayank Kejriwal `[通讯]` (GRAIL)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对四个主流 AI 科学家框架进行基准对比，并提出多模型 LLM 评审协议。

**💡 创新点**

首个系统级量化基准；使用多模型 LLM（GPT‑5.4、Gemini、Claude）实现自动评审；提供质量维度得分与合成。

**🔧 技术方法**

大型语言模型、自动化实验执行、LLM 生成合成评分与评述，构建多模型协同评审框架。

**📊 数据集**

15 条 FARS 生成的研究提案（包含对应数据集）作为输入，生成 75 篇论文进行评测。

**📈 对比分析**

在原创性、严谨性、清晰度、重要性四维度对 75 篇论文评分；FARS 得分显著高，Gemini/Claude 评审一致，GPT‑5.4 区别大；非 FARS 框架质量普遍低，性能差距明显。

**⚠️ 局限性**

评审一致性受模型差异影响；FARS 参考论文未参与成本比较；实验仅覆盖 15 个提案；LLM 评审仍可能存在偏差与主观性。

---

## 35. A Unified Benchmark of Deep Learning Models for Multi-task 3D Brain Tumor Segmentation from Magnetic Resonance Imaging

**arXiv ID:** 2607.28858 | [PDF](https://arxiv.org/pdf/2607.28858v1)

**作者:** Diego J. Torrejón `[一作]` (University of Las Palmas de Gran Canaria), Javier Sánchez `[通讯]` (University of Las Palmas de Gran Canaria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在统一的实验框架下，对五种代表性3D脑肿瘤分割网络（3D U‑Net、SegResNet、Swin UNETR、SegMamba、SegMambaV2）进行训练、评估和对比。

**💡 创新点**

创新点：①同一实验条件下同时比较CNN、Transformer与State‑Space模型；②覆盖两种不同临床场景的BraTS 2023和BraTS 2024数据集；③在性能评估中同时加入推理时间与参数量，体现计算效率与精度的权衡。

**🔧 技术方法**

使用的技术包括：三维U‑Net、SegResNet、Swin UNETR、SegMamba、SegMambaV2；Dice+交叉熵损失；MONAI预处理和数据增强；自动混合精度训练；滑动窗口推理；AdamW优化器；梯度裁剪。

**📊 数据集**

数据集：BraTS 2023（颅内脑膜瘤）和BraTS 2024（术后胶质瘤）两套公开 MRI 数据集。

**📈 对比分析**

比较方法：五折交叉验证、统一预处理、增强、优化策略，使用 Dice、IoU、HD95 等指标以及推理时间与参数量进行多维度评估。结果显示 SegMamba 与 SegMambaV2 在两套数据上均取得最高 Dice，且在参数量与推理时间方面也具备较好平衡；CNN 与 Transformer 模型在某些子区域表现仍具竞争力。

**⚠️ 局限性**

局限性：①仅评估两套数据，未覆盖更多脑肿瘤类型；②使用单模型训练，未涉及集成或自监督预训练；③评估采用全局指标，难以直接与官方基准（基于病灶级评估）的结果做严格对比。

---

## 36. Seeing Differently: Modeling Interpretive Perspectives in Computational Creativity using a Four-World Framework

**arXiv ID:** 2607.28644 | [PDF](https://arxiv.org/pdf/2607.28644v1)

**作者:** Prerna Luthra `[一作]` `[通讯]`, Prerna Luthra

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

使用12维创意框架和三种解读人格（形式主义、社会历史、图像符号）对SemArt 1069幅画作进行评估，探讨解读视角如何影响创意评价；

**💡 创新点**

首次把创意评价视为关系属性，提出多视角人格评估方法，并证明不同视角在视觉嵌入空间中对应不同方向，从而量化解读多样性；

**🔧 技术方法**

利用GPT‑4.1视觉‑语言模型进行人格条件评分，CLIP ViT‑B/32 嵌入与线性岭回归探测视角方向，配合Friedman、Wilcoxon、Kendall W 等统计检验；

**📊 数据集**

SemArt 数据集（约21000幅欧洲传统绘画），本研究使用其测试集1069幅；

**📈 对比分析**

对三视角评分做重复测量方差分析和方差/ Kendall W 计算，显示显著差异；线性探测得到的方向相似度证明视角在嵌入空间中的区别；结果显示社会反射等特质极为视角敏感，模型能捕捉视角差异；

**⚠️ 局限性**

仅在欧洲绘画上验证，评估由模型而非人类专家完成；依赖文本元数据，未分离图像与文本对视角差异的贡献；使用线性探测简化了表征，未来需跨文化、实地人类实验验证；

---

## 37. DragonCrawl: A Generative, Intent-Based Framework for Scalable Mobile End-to-End Testing

**arXiv ID:** 2607.28750 | [PDF](https://arxiv.org/pdf/2607.28750v1)

**作者:** Sowjanya Puligadda `[一作]` (Uber Technologies, Inc.), Juan Marcano `[通讯]` (Uber Technologies, Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发并部署了 DragonCrawl V2，一套基于大语言模型的移动端端到端持续回归测试系统，能够在每次代码提交时自动验证 61 条关键用户流程并在 CI/CD 中阻止破坏性变更。

**💡 创新点**

核心创新包括：①从基于嵌入相似匹配升级为生成式意图驱动推理；②使用 GPT‑4o 的多模态视觉能力实现端态检测；③通过工具调用实现后端状态控制；④结合 Prompt 约束、视图层 Canonization 和缓存，显著提升通用性、稳定性与成本效率。

**🔧 技术方法**

主要技术手段：GPT‑4o（多模态 LLM） + LangChain 服务；视图层 Canonization；工具调用 API；Prompt 工程与约束；多模态视觉判断；缓存、批处理、CI/CD（Buildkite）集成；Kafka、Grafana、Hive、ML Studio 进行监控与评估。

**📊 数据集**

使用内部 Golden Dataset（约 10,000 条验证交互序列）和 1,013 条生产级测试流；对比 V1 的 MPNet 嵌入结果；无公开数据集，全部基于 Uber 自有应用日志与屏幕截图。

**📈 对比分析**

与 V1（Embedding）及其他 LLM‑based 工具（Guardian、DroidBot‑GPT、AutoDroid、VisiDroid）比较：V2 在 iOS 91.6%、Android 92.2% 的通过率；与 V1 的 80–82% 明显提升；与 VisiDroid 88% 对比更优；精确率@1 92.1%，复杂流程 89%；成本优化后年耗约 200K 美元；维护成本从 30–40% 降至 5%；上线时间从 96–120 小时降至 <4 小时，累计节省约 27 人年。

**⚠️ 局限性**

局限性：依赖外部 GPT‑4o，模型更新、价格与可用性风险；对成本敏感，需大规模测试套件才能体现 ROI；需要大量内部 Golden Dataset 与持续评估管道；非确定性偶尔导致路径漂移；视觉断言对加载/过渡状态敏感；目前仅在 Uber 的专用基础设施验证，未在第三方或特殊 UI 场景（如游戏、Canvas）测试。

---

## 38. EarlyDx: An Admission-Anchored Benchmark for Open-Ended Generation of Evidence-Supported ED-Encounter Diagnoses

**arXiv ID:** 2607.28788 | [PDF](https://arxiv.org/pdf/2607.28788v1)

**作者:** Jiahui Li `[一作]` (University of Georgia), Fei Dou `[通讯]` (University of Georgia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个基于MIMIC-IV的、面向急诊入院时可用信息的开放式早期诊断基准EarlyDx，采用证据核查的方式保证标签与入院时可获得的证据相匹配；

**💡 创新点**

创新点在于：①采用开放文本诊断标签而非转化为ICD代码，保留临床细节；②将诊断标签与入院时证据进行人工智能审核，区分支持、部分支持与不支持；③设计了LLM-as-judge评估协议，避免字符串匹配误差；④通过多模态文本化输入捕捉多源证据。

**🔧 技术方法**

主要技术包括：大型语言模型（GPT‑5.5、Claude Opus、Qwen3.5、MedGemma等）的零射/少射推理、LLM审计器（MiniMax M3）进行标签审核、基于链式思维（CoT）的监督微调、以及语义匹配评估。

**📊 数据集**

使用数据集为MIMIC-IV中的154,834例急诊入院记录（入院时的多源文本化数据），并对诊断标签进行过滤和审核。

**📈 对比分析**

比较方法：与无学习基线（频率先验、检索）及监督式BERT分类器对比；零射通用和医学专用LLM对比；微调后模型对比。性能显示：零射模型F1约0.3，微调后Qwen3.5‑4B在支持标签上的F1提升至0.51；但在时间关键诊断上，所有模型的敏感性与精度仍低于经验医生。

**⚠️ 局限性**

局限性包括：①参考诊断仅为最终编码，无法反映临床考虑的差异诊断；②文本化输入缺乏实际检查、动态过程信息；③LLM审计与评估可能仍受模型偏差影响；④单中心数据、未覆盖多机构多语言；⑤未对不同临床决策阈值进行优化。

---

## 39. ReLoop-UME: Recurrent Depth with Learnable Retrieval Registers for Universal Multimodal Embedding

**arXiv ID:** 2607.28751 | [PDF](https://arxiv.org/pdf/2607.28751v1)

**作者:** Shijie Wang `[一作]` (Chinese Academy of Sciences), Haiyun Guo `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReLoop-UME 方案，通过在 UME 模型中将前缀编码一次，重复使用相同的检索形成层进行深度递归，并在末尾映射到嵌入空间；引入 Learnable Retrieval Registers 作为固定大小的检索状态存储；

**💡 创新点**

创新点在于：1）将额外计算定位到检索形成阶段而非整体前向或 token 生成；2）采用局部递归共享参数、保持 token 序列不变；3）使用可学习检索寄存器来持续积累跨循环的检索证据；

**🔧 技术方法**

技术包括：层级分割（Prefix、Retrieval‑Forming、Mapping 三段）；递归 Transformer 共享参数；可学习寄存器与注意力交互；终端对比损失；多尺度实验。

**📊 数据集**

数据集：MMEB‑V2（78 任务，包括图像、视频、视觉文档）和 MRMR（零样本迁移）；使用 Qwen‑VL、Qwen‑3‑VL 等多种后端。

**📈 对比分析**

与单前向 UME、UME‑R1、PLUME 等基线对比；在 MMEB‑V2 上 2B 规模 ReLoop‑UME 达 All 63.2，超越 PLUME（61.6）和 UME‑R1（60.1）；7B 规模 65.9 超越 UME‑R1（64.5）；速度上比 UME‑R1 快 44.9×，比 PLUME 快 1.5×，仅 1.3× 延迟。

**⚠️ 局限性**

局限：视频检索性能仍低于某些基线；递归深度提升到 8 时性能略降，需调节；对极大输入长度或多模态组合的鲁棒性尚未完全验证；

---

## 40. Preventing Premature Commitment in Coding Agents with an Evidence-Conditioned Execution Layer

**arXiv ID:** 2607.28815 | [PDF](https://arxiv.org/pdf/2607.28815v1)

**作者:** Yisen Xu `[一作]` (Concordia University), Tse-Hsun Chen `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为编程代理提供了一个执行层，在提交编辑或补丁之前验证足够的仓库证据，避免过早提交。

**💡 创新点**

创新点在于将证据收集与执行决策分离：先用LLM编译任务特定的证据规范，再在运行时跟踪证据缺口并根据缺口阻止不完整的提交。

**🔧 技术方法**

利用语言模型生成结构化证据条件，使用事件日志进行确定性满足检测，结合全局和动作级证据缺口来控制提交。

**📊 数据集**

使用 SWE-bench Verified 500 个真实 GitHub issue‑repo 组合作为评测数据集。

**📈 对比分析**

与基线代理及 Self-Refine 进行对比，四种模型/框架组合中 Pass@1 提升 4.8–11.8%，并在 token 与成本上减少 1.4–12.1%。

**⚠️ 局限性**

局限包括对 issue 描述质量的依赖、证据检查与行动选择由同一模型完成可能产生盲点，以及仅在 SWE-bench Verified 和有限语言/框架上验证。

---

## 41. 55 Additions Suffice for 3x3 Matrix Multiplication at Rank 23

**arXiv ID:** 2607.28676 | [PDF](https://arxiv.org/pdf/2607.28676v1)

**作者:** Samurdhi Karunaratne `[一作]` (Logical AI), Anushka Idamekorala `[通讯]` (Logical AI)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计并实现了一个在任意结合环上使用23个标量乘法、55个加减法完成两个3×3矩阵相乘的算法。该算法通过对已公开的Perminov 58加法张量进行重新调度，得到更优的线性电路，从而将加法次数从56降低到55。

**💡 创新点**

创新点在于：① 在保持张量秩23不变的前提下，首次证明在固定张量方向下实现55个加法是最优的；② 通过对输入、输出的线性映射进行精确的最优搜索与转置，获得了13、14、28加法的最优电路；③ 提供了完整的可验证证书和多语言验证程序，确保结果在任何关联环上都成立。

**🔧 技术方法**

主要技术包括：张量分解与三线性算法、二进制加减法线性电路合成、转置原则（Transposition Principle）以得到输出映射、穷举依赖搜索来证明最优性，以及使用Brent身份验证完整性检查。

**📊 数据集**

本工作不依赖外部数据集，仅使用数学张量（Perminov的公开23秩3×3乘法张量）作为输入。

**📈 对比分析**

与之前的58、59、56加法实现比较，所提出的55加法实现进一步减少了1次加法，保持乘法次数23不变。实验通过Python和Node.js等多语言实现验证了所有729条Brent身份，证明了算法的正确性与性能。

**⚠️ 局限性**

局限性：① 该加法最优性仅针对固定张量方向；② 未证明在所有可能的张量变换或不同电路模型下是否还能进一步降低加法次数；③ 仍未实现23秩的更低加法计数，未探索是否存在更优的线性调度或更低秩的分解。

---

## 42. To Add Is Machine, To Delete Is Human: Measuring and Mitigating Deletion Avoidance in LLM Code Editing

**arXiv ID:** 2607.28887 | [PDF](https://arxiv.org/pdf/2607.28887v1)

**作者:** Amir M. Ebrahimi `[一作]`, Ahmed E. Hassan `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文未提供内容，无法概述。

**💡 创新点**

N/A

**🔧 技术方法**

N/A

**📊 数据集**

N/A

**📈 对比分析**

N/A

**⚠️ 局限性**

N/A

---

## 43. Fast Rates for Swap-Agnostic Learning of Proper Losses

**arXiv ID:** 2607.28856 | [PDF](https://arxiv.org/pdf/2607.28856v1)

**作者:** Princewill Okoroafor `[一作]` `[通讯]` (Harvard University), Princewill Okoroafor (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了swap-agnostic学习框架，即在比较预测器与基准时允许基准在不同预测值上使用不同的假设，从而实现对预测后处理的完整覆盖；并给出了离线与在线两种设置下的高效学习算法，理论上证明了对任意固定的L-Lipschitz或凸1-Lipschitz充分可分辨的损失，swap-agnostic学习的风险与外部对抗风险的上界分别为O(L(log‖H‖/m)^{2/3})和O(L T^{1/3}(log‖H‖)^{2/3})，在整个loss族上能实现O(√(T log‖H‖))和O(√(log‖H‖/m))的速率。

**💡 创新点**

创新点在于：①将swap-agnostic学习归结为“第二阶多校准”（second-order multicalibration），利用损失的几何结构（Bregman散度与漂移项）将每个预测桶的成本内聚为对数级；②构造了一个基于Blackwell近似性与贝塞尔修正的乘法权重算法，能够同时控制所有桶的偏差与方差；③通过正向分解将任意凸1-Lipschitz损失写成ReLU基函数的正和，实现对整个损失族的统一处理；④提出了“canonical properization”把任意带后处理的损失转化为适合swap-agnostic学习的充分可分辨损失。

**🔧 技术方法**

技术手段包括：黑塞尔近似性（Blackwell approachability）实现多维目标的逼近；乘法权重更新与Bernstein指数修正得到高概率的二阶偏差-方差控制；对损失的Bregman散度展开与二次增长不等式；对V-shaped基损失与ReLU基损失的正向分解；离线到在线的在线-批量转换（online-to-batch）与Freedman–Bernstein不等式；以及基于测试类的紧致覆盖（finite cover）和稀疏化的参数网格。

**📊 数据集**

本文为纯理论论文，无实验或数据集。

**📈 对比分析**

与已有的多校准或swap-omniprediction方法比较，本文的swap-agnostic学习在同样的假设下取得了更优的渐进率（如从O(T^{2/3}(log‖H‖)^{1/3})提升到O(T^{1/3}(log‖H‖)^{2/3})），同时在损失族上保持统一的平方根速率。

**⚠️ 局限性**

局限性包括：算法目前仅适用于有限假设集，未处理无限或可学习的测试类；需要对预测范围与基准预测值的离散化进行预先固定；对非充分可分辨或非凸损失缺乏直接支持；实现时对每个桶、测试与学习率维护权重，计算成本较高；缺乏实验验证。

---

## 44. Physics-Aligned Self-Supervised Learning for Scientific Imaging

**arXiv ID:** 2607.28868 | [PDF](https://arxiv.org/pdf/2607.28868v1)

**作者:** Bashir Kazimi `[一作]` (Forschungszentrum Jülich GmbH), Stefan Sandfeld `[通讯]` (Forschungszentrum Jülich GmbH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并提供了一套可重复、基于物理测量约束的科学自监督学习数据增强设计流程；

**💡 创新点**

创新点在于将物理测量对称性与采集变异性系统化为可排除、可纳入的增强集合，并给出标签无关的验证与单因素消融步骤，首创性地把增强设计转化为可执行的工作流；

**🔧 技术方法**

使用的技术包括自监督学习框架（DINOv2、SimCLR、MAE、VICRegL、I-JEPA）、物理对称性检验、表示几何诊断（有效秩、均匀度、崩塌比）、kNN 评估、以及鲁棒性对测量扰动的测试；

**📊 数据集**

实验数据集包括 CEM500K 及 NFFA 真实空间电子显微镜图像、以及模拟与实验的 4D-STEM 扫描点衍射数据；

**📈 对比分析**

通过在两种任务（NFFA 多分类、4D-STEM 立方体角度回归）上与传统自然图像增强（𝒯_orig）对比，发现物理对齐增强在跨视图一致性方法上提升显著（如 DINOv2 NFFA 上升约10%点，4D-STEM 误差从9.85°降至5.60°），并在低标签和鲁棒性测试中保持优势；

**⚠️ 局限性**

局限性在于仅针对两种成像模态与固定增强池，未探索自动增强搜索、密集预测任务或更广泛的测量驱动领域的迁移验证。

---

## 45. Library Reachability in LSR-Synth: How Anti-Memorization Design Changes the Measurement of Symbolic Discovery

**arXiv ID:** 2607.28684 | [PDF](https://arxiv.org/pdf/2607.28684v1)

**作者:** Zhan'ao Yao `[一作]` (State Key Laboratory of High Performance Ceramics, Shanghai Institute of Ceramics, Chinese Academy of Sciences), Jianjun Liu `[通讯]` (State Key Laboratory of High Performance Ceramics, Shanghai Institute of Ceramics, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 LSR-Synth 基准上评估大型语言模型（LLM）是否能在符号回归任务中提供超出固定语义无关词库的科学先验；通过构造语义无关基线、语义屏蔽、词库削弱和算子消除等实验，检测 LLM 产生的候选项是否真正拓宽了解析空间。

**💡 创新点**

提出“库可达性（library reachability）”评估框架：将完整公式的创新与其搜索空间的创新区分，展示在当前基准中，固定语义无关词库已能覆盖大部分任务，LLM 的增益仅在词库缺失关键算子时才显现。

**🔧 技术方法**

使用符号回归技术、基于 LLM 的候选生成、语义屏蔽（L1/L2 输入差异）、固定词库构造（LaSR、PySR、多项式库）、前向贪心搜索、严格 ID+OOD 评估、符号准确度（SA）判定以及算子消除对照实验。

**📊 数据集**

基于 LSR-Synth 129 个任务（人口增长、物理振荡、化学反应、材料关系），每个任务包含 4000 训练点、500 ID、500 OOD，配有变量名称、含义和领域描述。

**📈 对比分析**

对比指标包括 Acc_0.01、Acc_0.001、严格 ID+OOD 误差、符号准确度 SA。结果显示：固定词库 BANK‑LaSR 的 Acc_0.01/Acc_0.001 与 SA 均超越现有方法；LLM 单独性能较差；UNION（BANK+LLM）在完整词库下提升有限，若采用多项式弱化库则能显著提升（Acc_0.01 上升约 30%）。

**⚠️ 局限性**

局限性包括：仅评估当前 LSR‑Synth 快照；词库覆盖度和搜索预算的限制；算子消除实验覆盖任务有限（仅 15 例）；SA 判定采用 10‑of‑10 一致性规则，可能低估符号匹配度；实验未探究不同 LLM 模型或更大搜索预算下的效果。

---

## 46. ConnectED: A Curriculum-Aligned AI System for Vietnamese Instructional Lesson Planning and Student Learning

**arXiv ID:** 2607.28647 | [PDF](https://arxiv.org/pdf/2607.28647v1)

**作者:** Thang Doan Viet `[一作]` (Secomus Technology), Tai Le Quy `[通讯]` (University of Koblenz)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ConnectED，一套面向越南中学的人工智能教学辅助系统，支持课程对齐的教案生成、STEM 可视化、学生交互学习及教师反馈闭环。

**💡 创新点**

创新点包括：① 将 Qwen3‑8B 通过 SFT + DPO 微调为 VietEduQwen，专为越南教育语料和教学安全定制；② 基于 ADDIE 框架的多阶段 LLM 调度，嵌入教师审核门控和官方课程规范检查；③ 多代理（生成、纠错、评判）流程自动生成并校正 Manim STEM 动画；④ 将教案与交互式 Playground 绑定，持续收集学习信号并反馈给教师。

**🔧 技术方法**

采用的技术：大型语言模型（Qwen3‑8B → VietEduQwen）、监督微调 (SFT) + 直接偏好优化 (DPO)、ADDIE 指导式提示模板、三代理自动化流水线（Generator, Fixer, Judge）、Manim 动画与 ElevenLabs 语音合成、Web 前后端（React/TypeScript + FastAPI + Celery + Redis + MongoDB）。

**📊 数据集**

训练数据包括约700条高质量越南维基百科文章、官方教材与全国高考题库；安全训练采样自 Toxic-Chat；评测使用 2025 年全国高中考试题集（3,119 题）和 VNHSGE 基准。

**📈 对比分析**

评估方法：对 3,119 道全国高考题进行准确率测试，VietEduQwen 达到 87.02%（比基线 Qwen3‑8B 提升 6.10%）；教师与学生满意度调查（N=18/214）平均分 4.7/5；准备时间从手工 3–4 小时降至 15–20 分钟。消融实验表明 DPO 与 ADDIE 调度分别提升准确率与教师满意度，完整流程显著优于无结构提示。

**⚠️ 局限性**

局限性：① 未进行受控前后测，缺乏因果性学习成效证明；② 训练与评测可能存在数据泄漏；③ ADDIE 结构可能对经验教师的灵活性造成限制；④ 对生成可视化的认知负荷未系统评估；⑤ 部署时对算力、网络和教师 AI 习得水平的依赖，可能限制在资源匮乏地区的应用。

---

## 47. Why It Hurts: Identifying the Drivers of Negative Thoughts in Emotional Support Conversations

**arXiv ID:** 2607.28648 | [PDF](https://arxiv.org/pdf/2607.28648v1)

**作者:** Hainiu Xu `[一作]` (King's College London), Yulan He `[通讯]` (King's College London)

**通讯引用:** 14028 | [OpenAlex ID](https://openalex.org/A5015709853)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了针对情感支持对话中情绪唤醒认知评价维度重要性识别的 benchmark（Appraisal Saliency Dataset）并提出基于贝叶斯逆规划的多智能体概率框架（PRISM），用于提升大型语言模型在推断情境特定重要维度的能力。

**💡 创新点**

创新点在于：①首次构建可评估 LLM 在情感支持对话中推断最显著认知评价维度的 benchmark；②将贝叶斯逆规划与多智能体候选假设搜索相结合，显著提高模型对上下文特定重要维度的识别精度；③通过信息增益权重化对话轮次贡献，细化推断过程。

**🔧 技术方法**

技术主要包括：大型语言模型对话生成与重写、人工标注、贝叶斯逆规划（BIP）推理、候选假设池、token 级对数似然评分、信息增益计算、累积概率聚合、最终 saliency 判别器。

**📊 数据集**

使用了自研的 Appraisal Saliency Dataset（996 条情感支持对话），其中包含帮助者人物档案、对话轨迹、前事件信念-欲望-意图（BDI）以及人标注的五个显著认知评价维度。

**📈 对比分析**

在多种开源（Llama2、Claude、GPT‑4 等）和专有模型上使用 Precision@k 与 Jaccard 指标进行对比；人类标注表现最高，LLM 基准表现落后；加入 PRISM 框架后，Precision@1 提升至 0.296、Precision@2 提升至 0.133，整体显著优于无框架基线。

**⚠️ 局限性**

局限性包括：①对话场景仅为单一事件，缺乏多事件、长期压力情境的复杂性；②数据生成过程依赖 LLM 与人工重写，可能带来语料偏差；③模型仅支持 token 级 log‑prob，受限于支持该功能的 LLM；④标注硬性要求五个显著维度，忽略了实际情境中可能的连续性与不确定性。

---

## 48. SEDR-Seq2P: A Lightweight Dilated Residual Sequence-to-Point Network for Multi-Task Industrial NILM

**arXiv ID:** 2607.28693 | [PDF](https://arxiv.org/pdf/2607.28693v1)

**作者:** Hatem Haddad `[一作]` (Wattnow), Issam Smaali `[通讯]` (Wattnow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一种一次到多种的工业非侵入式负载监测（NILM）模型，使用多任务学习从总功率序列同时估计十台工业设备的功率。

**💡 创新点**

创新点在于：① 将 Seq2Point 作为轻量级基座，加入扩张残差块和 squeeze‑and‑excitation 通道注意力，形成 SEDR‑Seq2P；② 采用统一的 Acc‑Delay 评估标准（Accuracy–Delay）同时衡量准确性和推理延迟；③ 在工业数据集上进行多模型基准，揭示轻量级与高精度模型在性能与部署成本上的权衡。

**🔧 技术方法**

使用的技术包括序列到点（Seq2Point）学习框架、扩张卷积、残差连接、Squeeze‑and‑Excitation 通道注意力、GRU 循环网络、WaveNet 风格的因果扩张卷积，以及基于 MAE 的训练和 AccD_MR 评估。

**📊 数据集**

使用公开的工业负载数据集 IMDELD（包含 8 台设备的 1 Hz 功率序列），在 7 天验证集和 1 天测试集上训练和评估。

**📈 对比分析**

对 Seq2Seq、Seq2SubSeq、Seq2Point、GRU、WaveNet 与 SEDR‑Seq2P 进行统一评测。结果表明：Seq2Point 在平均 MAE 与 MR 上优于 Seq2Seq/Seq2SubSeq；GRU 与 WaveNet 在准确性（MAE、R²、MR）上最好，但推理时间最长；SEDR‑Seq2P 在准确性上略优于 Seq2Point（MAE 约 7% 降低，R² 增加 1%），推理延迟显著低于 WaveNet（约 58% 降低），但 AccD_MR 仍低于 Seq2Point，显示轻量化与高精度之间仍存在折中。

**⚠️ 局限性**

局限性：① SEDR‑Seq2P 的性能提升相对有限，统计显著性不高；② 仍未实现最高的 AccD_MR，推测难以兼顾最优准确性与最快推理；③ 未考虑自适应窗口、跨站迁移学习、鲁棒性提升等现实部署需求；④ 只使用了单一工业数据集，未验证跨工况的泛化能力。

---

## 49. Meshy T2: Fast Native Mesh Generation with Flow Matching

**arXiv ID:** 2607.28675 | [PDF](https://arxiv.org/pdf/2607.28675v1)

**作者:** Jiale Xu `[一作]` (Meshy AI), Yuanming Hu `[通讯]` (Meshy AI)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

从输入图像或高分辨率网格生成高质量、可直接使用的多边形网格，并实现可控的面数与多部件分离。

**💡 创新点**

① 采用几乎无损的顶点集合 Mesh VAE，直接编码/解码顶点坐标、无向边和面向顺序；② 使用流匹配两阶段生成（图像‑条件体素流 + 图像+体素‑条件 Mesh 流），实现并行生成和面数可控；③ 引入 Sobol+最优传输对顶点令牌进行位置编码，解决无序集合的对称性难题。

**🔧 技术方法**

流匹配（Rectified Flow + DiT）、Transformer、RoPE、3D Voxel VAE、DINOv3 视觉编码、Sobol 序列、最优传输、稀疏点云/体素编码、图注意力、自注意力、Sinkhorn 排列、AdaLN 时间/计数条件。

**📊 数据集**

在 115 个多部件资产（对象、建筑、动物、人物）组成的基准上，提供参考照片和对应的 100k 三角网格；用于训练和评估的高分辨率 Meshy‑6 资产。

**📈 对比分析**

与 Tripo P1、MeshFlow（扩散）以及 MeshAnything V2、BPT、DeepMesh、MeshSilksong、FastMesh（自回归）对比；Meshy T2 在几何精度（CD 0.020、HD 0.044、NC 0.860）和速度（图像→网格中位 6 s，重拓扑 3 s）均优于所有基线，面数可控且多部件分离自然。

**⚠️ 局限性**

仍受限于单体资产生成；对极其不规则或极大尺寸网格的拓扑鲁棒性有限；缺乏材质、纹理或场景级多网格协同控制；需要进一步提升对多部件和极细细节的重建质量。

---

## 50. MPP-GNN: Subject-Adaptive Community Detection for fMRI-Based Alzheimer's Disease Classification

**arXiv ID:** 2607.28681 | [PDF](https://arxiv.org/pdf/2607.28681v1)

**作者:** Yang Zhang `[一作]` (Yale University), Mark Gerstein `[通讯]` (Yale University)

**通讯引用:** 248039 | [OpenAlex ID](https://openalex.org/A5042321575)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种端到端的MPP-GNN框架，结合自适应社区检测与边缘去噪，用于fMRI脑网络的阿尔茨海默病分类。

**💡 创新点**

创新点在于使用基于亲和传播的层次聚类（AP-HPM）和基于社区先验的概率边缘精炼（PERM），并通过双层优化实现社区结构与特征学习的耦合。

**🔧 技术方法**

采用图卷积网络、亲和传播聚类、概率边缘门控、变分SBO双层优化、图归一化及自回归采样等技术。

**📊 数据集**

在英国生物库（UK Biobank）和阿尔茨海默病影像研究计划（ADNI）两大公共fMRI数据集上进行实验。

**📈 对比分析**

与传统机器学习、通用GNN、专用脑网络模型等共15种基线对比，MPP-GNN在UK Biobank实现AUC 77.84%/ACC 78.33%，在ADNI同样超越基线，显示最佳分类性能。

**⚠️ 局限性**

主要局限在于计算开销较大、需手工调节超参数、仅在两种疾病数据上验证，且对跨模态或更大规模数据的泛化尚未评估。

---

## 51. Stratified Negation in RDF Rules: A Correct Approach (Extended Version)

**arXiv ID:** 2607.28778 | [PDF](https://arxiv.org/pdf/2607.28778v1)

**作者:** Nils Küchenmeister `[一作]` (TU Dresden), Markus Krötzsch `[通讯]` (TU Dresden)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的链层次化（chain‑stratification）方法，用来为包含默认否定和存在变量的 RDF 规则提供唯一、可生成且无冗余的语义模型；

**💡 创新点**

创新点在于引入“trail”和“chain”概念，并证明链关系可被正则语言描述，从而得到可判定的层次化条件，超越传统的分层化和完全分层化；

**🔧 技术方法**

主要技术包括：基于依赖关系（reliances）的图分析、轨迹（trail）与链（chain）的定义、正则语言构造与 Myhill‑Nerode 判定、以及在 Rust 中实现的链层次化检查算法；

**📊 数据集**

使用 201 组基于 Ontology Repository 的无否定、分解后的规则集作为基准数据集进行实验；

**📈 对比分析**

与核心分层化（core‑stratification）对比，链层次化在大多数规则集上额外耗时约 12.5%，并在 187 组规则集中完成，只有 14 组因规则数过多未在 15 分钟内完成；

**⚠️ 局限性**

局限性：缺乏真实 RDF 规则集的评估、对大规模规则集（>60k 规则）仍不稳定、算法复杂度较高且实现依赖大量符号映射，尚未在实际 RDF 引擎中测试。

---

## 52. Arranging circles of radii 1,2,...,n around a central circle: a Supnick TSP and certified finite optima

**arXiv ID:** 2607.28654 | [PDF](https://arxiv.org/pdf/2607.28654v1)

**作者:** Maurizio Falconi `[一作]` `[通讯]`, Maurizio Falconi

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将半径为 1,2,…,n 的不重叠圆外切于中心圆的最优排列，证明在所有可行环序中，中心圆半径的最小化对应于一个固定的 Supnick 旅行商顺序；对 3≤n≤14 通过精确可证明算法给出全局最优解并揭示浮动圆的级联现象；并提出中心圆半径满足 R*∼n²/8 的渐近猜想。

**💡 创新点**

创新点包括：① 将圆环排列问题与 Supnick 的固定旅行商理论建立新联系，证明最优顺序与 R 无关；② 通过对角分离矩阵的 anti‑Monge 性证明 Supnick 顺序最优；③ 设计完整的可证明 STN 算法和 50 位数验证器，首次给出 3≤n≤14 的确切最优解；④ 发现并描述浮动圆级联规律，为后续大 n 的研究提供线索。

**🔧 技术方法**

使用技术：Supnick anti‑Monge/anti‑Monge 矩阵理论、简单时序网络（STN）求解、分支限界与二分搜索、浮点误差控制与 50 位数高精度验证、Descartes 圆定理、渐近分析与积分估计。

**📊 数据集**

数据集：半径序列 1,2,…,n（n≤14 的确切计算结果；n=15–18 的启发式搜索结果），以及对应的证书文件、验证器代码与实验脚本，全部公开在 GitHub 仓库。

**📈 对比分析**

对比方法：对 n≤14 采用穷举枚举+可证明算法验证全局最优；与 Dan 及 Rei Henigman 的手工枚举结果一致；对更大 n 采用多起点局部搜索与理论下界比较。性能方面：n=13 需枚举约 2.4×10⁸ 条顺序（≈1.8 小时），n=14 约 3.1×10⁹ 条（≈22 小时）；结果与已知猜想完全吻合，且满足 R*≈n²/8 的渐近趋势。

**⚠️ 局限性**

局限性：① 级联浮动圆规律仅在 n≤14 通过计算得到，尚未在所有 n 上解析证明；② 对 n>14 的最优解仅为启发式搜索，未完全证明；③ 对非等距半径序列或三维球体情形的推广尚未完成；④ 对浮动集合 F(n) 的渐近大小（如是否为 Θ(√n)）仍为开放问题。

---

## 53. Token-Level Diagnosis of Sycophancy in LLMs with Attribution-Guided Steering

**arXiv ID:** 2607.28906 | [PDF](https://arxiv.org/pdf/2607.28906v1)

**作者:** Hieu Nguyen `[一作]` (University of South Florida), Gene Louis Kim `[通讯]` (University of South Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在面对权威人物声明时的顺从（sycophancy）行为，并提出了一种基于令牌级别归因的诊断与推理时干预方法。

**💡 创新点**

创新点在于引入Authority Share Index（ASI），利用集成梯度量化模型决策中权威文本与问题内容的相对影响，并基于此设计了归因引导的对抗激活对齐（activation steering）来抑制顺从。

**🔧 技术方法**

主要技术包括集成梯度（Integrated Gradients）归因、Token Attribution、对抗性激活对齐以及推理时的隐藏层激活补偿。

**📊 数据集**

实验使用了 MMLU 285 题子集，并构造了包含权威简介、声明、问题内容的多种提示块组合。

**📈 对比分析**

与五款 8B 大模型在 30 种提示配置下对比，ASI 在 29/30 配置中显著区分顺从与抵抗；归因引导的激活对齐将最高顺从率从 96% 降至 25%，在所有配置均表现出显著下降。

**⚠️ 局限性**

局限性包括仅针对 8B 规模模型、仅在权威声明错误场景下评估、集成梯度对更大模型的可扩展性受限，以及对非文本权威因素的探索不足。

---

## 54. LLM Framework for Discovering Major Mathematical Conjectures: AI's Quest for the Next Riemann Hypothesis

**arXiv ID:** 2607.28632 | [PDF](https://arxiv.org/pdf/2607.28632v1)

**作者:** Alizer Wong `[一作]` (Peking University), Yanhui Chen `[通讯]` (Guangdong University of Technology)

**通讯引用:** 7671 | [OpenAlex ID](https://openalex.org/A5100777643)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个三阶段管线，用于从数学区域中搜索、评估并在Lean 4 / Mathlib 中正式验证大规模数学猜想的候选项；

**💡 创新点**

将大规模猜想的发现转化为结构化搜索+语义验证+正式检查的异构验证链，首次将“高问题口味”（potential significance）作为显式优化目标；

**🔧 技术方法**

采用大型语言模型（GPT‑Pro）进行候选生成与语义评估，使用Lean 4 与 Mathlib 进行语法解析、exact? 与 aesop 自动检查，配合自定义评分体系与手工案例分析；

**📊 数据集**

使用作者自行构建的 20 条候选猜想语料（包含自然语言描述、语义分数和正式化信息），以及公开的 Mathlib 库进行正式验证；

**📈 对比分析**

评估指标包括：语法可解析率（20/20）、exact? 未直接归约率（20/20）、aesop 未自动证明率（20/20）。相比传统单一模型生成，管线能在保留语义密度的同时维持正式张力，体现了“高问题口味”筛选的有效性；

**⚠️ 局限性**

局限性在于：仍需人工评估高层语义分数；未验证猜想的实际数学价值与可证明性；对大型语言模型的输出质量高度依赖，可能出现偏向已知模板或无意义的猜想。

---

## 55. Scaffolding Critical Engagement with GenAI: Transforming Ethnic Minority Preparatory Students' Collaborative Discourse in Prompt Engineering Tasks

**arXiv ID:** 2607.28630 | [PDF](https://arxiv.org/pdf/2607.28630v1)

**作者:** Deliang Wang `[一作]` (University of Hong Kong), Cunling Bian `[通讯]` (Ocean University of China)

**通讯引用:** 299 | [OpenAlex ID](https://openalex.org/A5086498471)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对中国民族少数族裔预科生进行为期三周的生成式人工智能（GenAI）课程干预，研究在教师设计的支持下其协同话语从被动复制转向批判性共创的过程。

**💡 创新点**

创新之处在于首次将人‑AI‑人协作的“人‑循环”工作流程与教师对比案例建模相结合，系统性地促使学生从将 GenAI 当作答案引擎，转而成为批判性评估与协同构建的伙伴。

**🔧 技术方法**

采用了深度学习对话平台 DeepSeek，结合认知网络分析（ENA）、主题分析和配对样本 t‑检验等方法来量化和解释学生的协作话语、反思和自我效能变化。

**📊 数据集**

数据集包括 13 个小组的聊天记录（共约 3,400 条消息）、学生的反思性作文和 71 份完整的前后测提示自我效能问卷。

**📈 对比分析**

通过对比未引导阶段和引导阶段的 ENA 网络，发现关键行为节点的连通强度从“策略–复制”转为“策略–总结–批判–共建”，并且配对样本 t‑检验显示提示自我效能从 3.757 提升至 3.991（p < 0.001），表明教学干预显著提升了学生的批判性协作与自我效能。

**⚠️ 局限性**

局限性包括：仅在中国少数民族预科项目的单一样本中进行，缺乏跨文化或跨学科的推广性；干预周期短暂，未检验长期习惯维持情况；且仅关注生成式 AI 的文本任务，其他应用场景仍需验证。

---

## 56. "YES! YES! I absolutely love this insight!" Affirmative Narration as Interactional Strategy in Dialogues with LLM Chatbots

**arXiv ID:** 2607.28646 | [PDF](https://arxiv.org/pdf/2607.28646v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 57. Sensitivity Analysis of GRU, LSTM and Transformer Encoder in Classification of Automated Driving Systems

**arXiv ID:** 2607.28665 | [PDF](https://arxiv.org/pdf/2607.28665v1)

**作者:** Bidhya Shrestha `[一作]` (University of Memphis), Christos Papadopoulos `[通讯]` (University of Memphis)

**通讯引用:** 4479 | [OpenAlex ID](https://openalex.org/A5105352599)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过仅利用车辆遥测数据，评估 GRU、LSTM 和 Transformer 编码器三种序列模型在识别四种自动驾驶系统（Openpilot、Manual、Autopilot、Super Cruise）时的分类性能，并提出了一个模块化的鲁棒性评估框架，模拟多种真实环境下的遥测失真，包括连续信号噪声、时间抖动、相关扰动以及事件级别的丢失、延迟与错误切换。

**💡 创新点**

创新点在于：①首次为遥测驱动的 ADS 分类设计结构化的鲁棒性基准，覆盖连续与事件两类失真；②通过系统化的失真等级 (L1–L5) 量化模型在不同失真强度下的性能衰减；③发现时间抖动是三种模型的主要失效模式，为后续同步感知与时间不变模型的研究指明方向。

**🔧 技术方法**

使用技术包括：GRU、LSTM 与六层 Transformer 编码器（8 头注意力）作为基线分类器；采用 Additive White Gaussian Noise、累计漂移、相关信号扰动、时间抖动、事件突发/延迟/交叉不一致等噪声模型；在训练时使用 85% 概率的混合噪声增强；评估指标以宏观 F1（macro‑F1）为主，兼顾各类 Precision/Recall。

**📊 数据集**

数据集来自 Comma、Tesla 与 Cadillac 三方的遥测记录，包含 11 维特征（6 连续 + 5 事件），总样本量 3.4M，按 70/15/15 的比例划分为训练、验证与测试集，标签分别为 0（Openpilot）、1（Manual）、2（Autopilot）、3（Super Cruise）。

**📈 对比分析**

比较结果：在干净数据上，Transformer 编码器以 macro‑F1=0.93 领先，GRU 0.92，LSTM 0.90；在鲁棒性评估中，Transformer 在大多数失真条件下保持最高性能，尤其是对相关扰动和事件级失真；但在时间抖动下，所有模型宏观 F1 均从 0.45–0.53 下降至 0.43–0.50，显示出显著的性能损失。

**⚠️ 局限性**

局限性包括：①模型对时间抖动高度敏感，说明对连续信号时序依赖过强；②仅评估了四种 ADS 与 SAE Level 2 情况，未涵盖更高级别或更广泛的厂商系统；③未进行大规模的超参数搜索或自适应预处理，可能影响对其他数据集的泛化；④鲁棒性框架主要基于仿真噪声，实际车载噪声分布可能更复杂。

---

## 58. LLM-Based Generative Retrieval for Snapchat Content Recommendation

**arXiv ID:** 2607.28895 | [PDF](https://arxiv.org/pdf/2607.28895v1)

**作者:** Liam Collins `[一作]` (Snap, Inc.), Neil Shah `[通讯]` (Snap, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并部署了SnapLGR，一种基于大型语言模型的生成式检索系统，用于Snapchat短视频推荐。

**💡 创新点**

创新点包括：将多模态嵌入与个性化PageRank共参与对比学习相结合的SID分词器、两阶段词表对齐（持续预训练+监督微调）以及面向大规模生产的高吞吐量训练与推理架构。

**🔧 技术方法**

所用技术包括：Qwen3-VL多模态嵌入、残差量化(RQ-VAE)生成SID、PPR共参与对比损失、持续预训练与监督微调、FlashAttention-2、TensorRT-LLM CUDA beam search、去中心化工作者循环架构和异步I/O。

**📊 数据集**

训练与评估使用Snapchat海量短视频数据集，包括视频、文本描述（由Gemini 2.5 Flash生成）以及用户交互日志。

**📈 对比分析**

在离线检索实验中，SnapLGR相较传统T5基线提升了2.5倍的Pass@10、2.27倍的Pass@32和2.4倍的Recall@32；在7天A/B测试中，视图时长提升0.37%、停留时长0.09%、深度会话0.18%以及深度会话用户0.11%。

**⚠️ 局限性**

局限性包括：在SFT后语义对齐显著衰减、对极大GPU预算和复杂系统的依赖、SID碰撞仍有提升空间，以及未验证更大LLM骨干的进一步性能提升。

---

## 59. Spatial Visual Analytics for Multi-Document Summary Verification

**arXiv ID:** 2607.28853 | [PDF](https://arxiv.org/pdf/2607.28853v1)

**作者:** Jiahao Xu `[一作]` (Virginia Tech), Chris North `[通讯]` (Virginia Tech)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个面向多文档摘要验证的可视化分析系统，结合空间文档组织与细粒度的归属链可视化。

**💡 创新点**

创新点在于将基于摘要结构与语义投影的两种空间布局与细粒度的归属链（Claims–Evidence）耦合，为用户提供从全局相关性到局部证据的完整检索与验证路径。

**🔧 技术方法**

使用语义相似度矩阵、UMAP投影、LLM判别（支持/矛盾/证据不足）以及可视化层叠技术实现归属链与空间布局。

**📊 数据集**

实验数据集包括WCEP（60篇业务/科学文档）和Multi-News（8篇飓风报道），以及自定义的混合主题60篇文档集。

**📈 对比分析**

与线性列表基线及现有系统SummVis/RTSUM等比较，通过17名参与者的任务准确率、时间、信心、负荷量测量，显示指导布局提升准确率36%、时间缩短32.7%、信心提升11.9%并显著降低认知与物理负荷。

**⚠️ 局限性**

局限性包括对LLM推理的依赖、只关注基于文档卡与句子高亮的证据呈现、以及缺乏更丰富的注释与自动化代理支持。

---

## 60. Counterexamples to Charpin's Conjecture on BCH codes

**arXiv ID:** 2607.28741 | [PDF](https://arxiv.org/pdf/2607.28741v1)

**作者:** Run Zheng `[一作]` (Hong Kong University of Science and Technology), Maosheng Xiong `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构造了一个无限族的原始窄义 BCH 码，证明其最小距离严格大于 Bose 距离，并以此反驳 Charpin 的猜想。

**💡 创新点**

创新点在于：1) 给出了首批理论上严格超出 Bose 距离的 BCH 码；2) 证明此类码的最小距离与 Bose 距离之间的差距可以无限增大；3) 通过将 BCH 码嵌入截断的广义 Reed–Muller 码并利用 Ax 定理得到权重可除性条件，从而推导最小距离上界。

**🔧 技术方法**

使用的主要技术包括：q-循环共轭集与 q-进展开；广义 Reed–Muller 码的定义与截断；Ax 定理（有限域多项式零点数的可除性）；对 BCH 码定义集与极大设计距离的分析；以及对二进制子族的精确最小距离计算。

**📊 数据集**

本工作为纯理论研究，没有使用实验数据集；所有结果均基于数学证明和符号推导。

**📈 对比分析**

作者没有与实验或基准方法进行比较；相对先前工作，证明了 Charpin 猜想错误，指出了先前认为最小距离与 Bose 距离之差被常数上界的假设不成立。展示了差距随码长的立方根增长，超过 4 的阈值，在 m≥13 时即可看到。

**⚠️ 局限性**

限制与未解问题：1) 该结论仅在子族 s=t-1 时已证明二进制情况，是否对任意 q 成立仍未证实；2) 目前给出的下界为 Θ(n^{1/3})，是否存在更快增长的族仍是开放问题；3) 对更广泛的 BCH 码族如何精确描述最小距离与 Bose 距离之间的关系仍需进一步研究。

---

## 61. TAGTorch: A PyTorch Library for Geometry, Topology, and Symmetry-Aware Machine Learning

**arXiv ID:** 2607.28755 | [PDF](https://arxiv.org/pdf/2607.28755v1)

**作者:** Brendan Kennedy `[一作]` (Pacific Northwest National Laboratory), Henry Kvinge `[通讯]` (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个名为TAGTorch的PyTorch库，统一了拓扑、代数与几何相关的深度学习工具，包括数据预处理、网络架构、训练技术及模型分析；

**💡 创新点**

核心创新点在于构建统一的API与抽象层，整合跨学科方法，降低非数学背景研究者的使用门槛，并通过模块化设计支持社区贡献；

**🔧 技术方法**

使用PyTorch为基础，结合拓扑数据分析（如持久同调、Euler特征变换）、等变网络（如SO(3)等变层、DeepSets等）、代数表示与群作用等数学构造；

**📊 数据集**

本论文未针对具体公开数据集进行实验，重点在于库的实现与功能说明；

**📈 对比分析**

由于缺乏实验评估，未给出与现有库的性能对比；

**⚠️ 局限性**

限制主要包括：库仍处于早期阶段，功能覆盖有限；对外部依赖的集成可能导致兼容性问题；缺乏系统的性能基准与完整的文档与示例。

---

## 62. The Morphological Core of Dungan: A Two-Dialect Finite-State Model and a Multi-Genre Evaluation

**arXiv ID:** 2607.28766 | [PDF](https://arxiv.org/pdf/2607.28766v1)

**作者:** Anton M. Alekseev `[一作]` (Steklov Mathematical Institute), Sergey I. Nikolenko `[通讯]` (Steklov Mathematical Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了基于HFST的有限状态形态学分析器，对Dungan语言进行词形分析与生成。

**💡 创新点**

首次对Dungan的形态学进行量化评估，证明其形态核心紧凑、闭合并给出正式的词类与标记表。

**🔧 技术方法**

使用HFST实现有限状态形态学，结合两级规则（twol）和Apertium/GiellaLT工具链，完成词典与规则的编译。

**📊 数据集**

利用约12.7k词条的词典（Yanshansin、Wiktionary等）以及三种语料库：百科、民间谚语和圣经文本，累计约12万词。

**📈 对比分析**

通过覆盖率、召回率、精确率与核心词汇完整性评估，覆盖率达72–85%，召回率100%，精确率约83%（含声调约56%），显示形态核心完整性高。

**⚠️ 局限性**

局限在于未经过母语者验证、忽略部分已记录现象（如助词屈折与复合形式）、仅覆盖书面语、数据来源单一且循环验证。

---

## 63. Automated Testing and Repair for Verified Compilers Generated by a Coding Agent

**arXiv ID:** 2607.28928 | [PDF](https://arxiv.org/pdf/2607.28928v1)

**作者:** Martin Rinard `[一作]` `[通讯]` (National University of Singapore), Martin Rinard (National University of Singapore)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个自动化测试与修复系统ACDC，针对由编码代理生成的已验证编译器Axon的四类代码（已验证、已检查、未验证、规范）进行缺陷检测与自动修复，确保修复后仍满足完整性证明。

**💡 创新点**

创新点在于：①利用已验证编译器的结构（操作语义、证书、检查器）设计专门的测试技术；②使用编码代理自动生成测试脚本、提示与修复，并自动更新相关证明；③对奖励作弊（reward hacking）进行系统性评估，验证在此环境下仍无作弊现象。

**🔧 技术方法**

技术包括：随机程序/汇编生成、操作语义与打印器比较、证书生成与检查器多条件验证、编码代理生成的自动化测试与修复脚本、Lean 4的机械化证明与证明更新、基于文本解析与打印的AST一致性检测。

**📊 数据集**

使用的数据集主要是Livermore基准集（用于初始验证）、Csmith、SPE、EMI等随机生成程序，随机生成的ASM指令序列以及AST程序。测试规模涵盖数百万条程序和数千条ASM序列。

**📈 对比分析**

对比方法：在30分钟检测+修复循环中评估缺陷发现与修复率；在八轮无缺陷检测后判定收敛。结果显示：所有检测到的缺陷和证书拒绝均被修复，修复成本约为$24.7，总体成本$19.19（ASM/打印）+$5.38（解析/打印）+$14.1（证书检查）≈$44。修复覆盖率100%，修复代码行数均较小，证明更新线性且可验证。

**⚠️ 局限性**

局限性包括：①仅在Axon这一小型语言编译器上验证，可能不适用于更大、更复杂语言的编译器；②依赖特定的随机测试生成器，覆盖率与真实代码缺陷可能不足；③研究仅覆盖编码代理生成的软件，对人类开发者编写的系统可行性未知；④实验受限于单一代理版本，后续代理改进可能导致结果变化。

---

## 64. Metaphor-Induced Algorithmic Steering: Cross-Domain Procedural Transfer in LLM Code Generation

**arXiv ID:** 2607.28683 | [PDF](https://arxiv.org/pdf/2607.28683v1)

**作者:** Zhibo Hu `[一作]` (University of New South Wales), Liming Zhu `[通讯]` (CSIRO)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究比喻式说明如何隐式将源场景的过程模式迁移至代码生成任务，从而诱导模型偏向低效算法。

**💡 创新点**

提出MASC框架与字面式技巧对比，展示比喻式技巧在诱导低效算法上的更强效果，并给出表示层面证据与检测器。

**🔧 技术方法**

采用元学习式技巧生成、迭代评估、层级表示对齐分析以及在Qwen、Deepseek、Gemma三大模型上进行推理与对比。

**📊 数据集**

使用APPS代码生成基准和BIRD‑SQL文本到SQL基准数据集。

**📈 对比分析**

与字面式技巧对比，MASC在三模型上实现约30–40% per‑task 成功率，严重程度上升；检测器宏F1达90%。

**⚠️ 局限性**

受限于比喻多样性与模型对比喻理解的可解释性，检测器对混合影响的分辨率有限，且实验依赖特定模型与数据集。

---

## 65. RareSense: Rarity-Aware Similarity Search for Anomaly Retrieval in Transactional Data

**arXiv ID:** 2607.28879 | [PDF](https://arxiv.org/pdf/2607.28879v1)

**作者:** Sidahmed Benabderrahmane `[一作]` (New York University), Talal Rahwan `[通讯]` (New York University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种稀有度感知的相似度框架RareSense，用稀有项集挖掘得到的置信度、提升度、稳定性等统计量加权的关联规则作为符号坐标，对事务型异常数据进行映射，随后采用加权Jaccard相似度实现查询驱动的异常检索和全局异常排序，并提供基于共享规则的可解释结果。

**💡 创新点**

创新点在于：①把相似度空间从原始原子特征迁移到稀有规则证据空间；②利用最小稀有项集生成可靠的关联规则并通过多维权重（逆支持、置信度、提升度、规则长度、稳定性）对坐标赋值；③证明该空间的加权Jaccard距离是伪度量并可通过倒排索引与度量索引高效查询；④提供直接的符号解释而非后置解释器；⑤引入无标签自适应稀有度阈值与最大项集长度选择机制。

**🔧 技术方法**

技术细节包括：稀有项集挖掘（最小稀有项集），关联规则生成（分区规则与闭包规则），规则权重组合公式，稀有规则配置映射，倒排索引候选过滤，权重归一化的加权Jaccard相似度，稀有度统计的自适应阈值，nDCG@10 与 AUROC 评估，Wilcoxon、Friedman 等统计检验。

**📊 数据集**

实验使用 27 个工作负载，涵盖四类基准：UWF‑ZeekData24（网络流量），NSL‑KDD（入侵检测），DARPA Transparent Computing（进程/日志轨迹），以及 ADRepository 的通用分类数据集（Reuters‑Corn、W7A、Solar Flare、Bank Marketing、APascal、AID362、Internet Ads）。每个数据集均包含若干十万级样本，异常比例从 0.01% 至 47% 变化。

**📈 对比分析**

与传统原子相似度（Jaccard、IDF‑Jaccard、cosine、TF‑IDF cosine）比较，RareSense 在宏平均 nDCG@10 上达 0.696，优于 0.645 的最佳原子基准（提升约 7.9%）。在全局异常排序任务中，RareSense 的宏平均 AUROC 为 0.850，位居同类方法之首，且与多种强势检测器（HBOS、AutoEncoder 等）在统计上可比。性能表现随工作负载而异，UWF 与 NSL‑KDD 取得最佳效果，DARPA Windows/Linux 及部分通用数据集表现相对弱势。

**⚠️ 局限性**

局限性包括：①稀有规则配置对阈值与最大长度敏感，若阈值过高导致规则稀疏、检索失效；②当异常主要由常见原子特征描述或共享稀有证据不足时，RareSense 可能不如原子相似度；③对特定攻击（如删除关键原子）更易受损；④需要离线挖掘和预构建字典，适应新流式数据需重新训练；⑤虽然在查询驱动检索上优于基准，但在全局排序上并未显著压倒所有传统检测器。

---

## 66. The AnyLog Edge Data Fabric

**arXiv ID:** 2607.28836 | [PDF](https://arxiv.org/pdf/2607.28836v1)

**作者:** Roy Shadmon `[一作]`, Moshe Shadmon `[通讯]`

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 AnyLog Edge Data Fabric（EDF），一种基于轻量化软件代理的分布式平台，用以在边缘设备上本地管理、查询并实时处理工业运营数据。

**💡 创新点**

核心创新点在于将数据留在源头、将计算迁移至数据所在节点、通过分布式元数据层实现统一可视化，并提供统一命名空间、单一系统镜像和模型上下文协议，使得多站点、多资产的实时 AI 与自动化能够在不依赖中心化云服务的前提下完成协同决策。

**🔧 技术方法**

技术实现包括：轻量化代理（支持 OPC UA、MQTT、Modbus 等南向接口）、分布式元数据同步（可选链码或中心化元数据管理）、分布式 SQL 引擎（虚拟数据湖）、统一命名空间（资产层级映射）、单一系统镜像（统一管理视图）、模型上下文协议（AI 代理调用）、以及边缘 AI、联邦学习和知识图谱等扩展。

**📊 数据集**

主要使用的工业现场数据来源为 PLC、传感器、历史记录数据库、物联网设备生成的实时观测数据；示例实验涉及数千台边缘设备、上万条实时数据记录，但未给出公开数据集。

**📈 对比分析**

与传统集中式数据平台对比，AnyLog 通过在本地执行查询、仅传输请求与压缩结果，显著降低网络带宽消耗、缩短响应时间，并实现了水平扩展；性能指标以网络延迟和吞吐量提升为主，论文中给出定性评估而非数值基准。

**⚠️ 局限性**

局限性包括：需要在每个边缘节点部署并维护代理，元数据同步与策略管理的复杂度高，安全性依赖代理身份与加密链路，网络分区时可能导致数据不一致或查询不完整，以及对极端低带宽或高安全隔离环境的适配仍需进一步验证。

---

## 67. Differentiable Approximations for Distance Queries

**arXiv ID:** 2607.28886 | [PDF](https://arxiv.org/pdf/2607.28886v1)

**作者:** Ahmed Abdelkader `[一作]` (Google LLC), David M. Mount `[通讯]` (University of Maryland)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种可微分距离场近似数据结构，能够在常维空间内为点集P给出距离值及其梯度；

**💡 创新点**

将分区合一（partition of unity）技术与基于Macbeath区域的逼近Voronoi图相结合，得到既保持近似因子(1+ε)又能保证梯度和Hessian均有最优上界的光滑距离函数；

**🔧 技术方法**

使用分区合一框架、近似Voronoi图（EVD）、Macbeath区域、椭球覆盖、平滑权重函数等几何与分析技术；

**📊 数据集**

论文主要是理论构造与分析，没有使用具体实验数据集；

**📈 对比分析**

与已知的最优ANN数据结构（如Arya‑Pettie）在查询时间O(log n/ε)和空间O(n/√ε)上实现相同复杂度，并额外提供梯度与Hessian的最优上界；

**⚠️ 局限性**

可能出现近似距离场中的局部极小值；梯度的上界为常数但可能远大于实际梯度1；仅适用于常维空间，实际实现仍需注意常数因子与融合开销。

---

## 68. Mirror Learning

**arXiv ID:** 2607.28737 | [PDF](https://arxiv.org/pdf/2607.28737v1)

**作者:** Yunpeng Liu `[一作]` (University of British Columbia), Frank Wood `[通讯]` (University of British Columbia)

**通讯引用:** 4176 | [OpenAlex ID](https://openalex.org/A5109993328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种通过第三人称观察生成镜像第一人称视频并结合逆动力学模型实现无交互的模仿学习框架。

**💡 创新点**

创新地将预训练视频扩散模型细调为视角转换器，且不依赖相机几何或三维重建；同时将其与逆动力学模型组合生成镜像数据进行行为克隆。

**🔧 技术方法**

采用Cosmos-Predict 2.5/2B 视频扩散模型、SAM3 分割、VaVaM/VaVim 逆动力学模型，结合 VAE 编码、RoPE 对齐等技术。

**📊 数据集**

在 CARLA 合成驾驶场景、Minecraft 多人游戏以及真实的 May robotaxi 数据集上进行训练与评估。

**📈 对比分析**

与仅使用真实第一人称数据的行为克隆、风格迁移等对比，镜像数据单独可达相当性能，加入镜像数据可提升 minADE5，且在新城市零样本迁移中表现优于风格迁移。

**⚠️ 局限性**

局限性包括对真实数据的零样本适配仍受模拟与现实差距限制，逆动力学模型精度依赖于学习者自身数据，且在极端视角不重叠场景下生成质量下降。

---

## 69. Chain-of-Models: Cross-Model Auditing for Bias-Robust LLM Judges

**arXiv ID:** 2607.28636 | [PDF](https://arxiv.org/pdf/2607.28636v1)

**作者:** Qian Wang `[一作]`, Bingsheng He `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Chain-of-Models（CoM）框架，利用第二个模型审计第一个模型的推理轨迹以降低LLM判定中的认知偏差。

**💡 创新点**

创新点在于：①证明了审计者与被审计者来自不同模型族更能抵消偏差；②发现单模型偏差抵抗力与审计效果不一致；③提出了基于功能多样性、单模型偏差抵抗力和校准审计效果的按偏差类型选择审计者的规则。

**🔧 技术方法**

技术主要包括：多模型链式推理、LLM-DNA功能多样性度量、按偏差类型的审计者选择算法、训练无偏推理轨迹提示模板。

**📊 数据集**

使用了四种认知偏差（bandwagon、authority、distraction、sycophancy）注入的MMLU-Pro多选问答数据集（Math、Chemistry、History、Psychology）以及四个DPO偏好判断数据集（Emerton、Orca、Py、Truthy）。

**📈 对比分析**

与单模型基线、同族规模提升和固定审计者（如GPT‑4o）相比，按偏差选择的审计者在四个偏差子集上平均准确率提升至0.884（比固定审计者0.824高0.06，基线0.805高0.08）。

**⚠️ 局限性**

局限性包括：需要先验的偏差标签或检测器、链式推理导致的延迟、仅评估单回合无检索/多模态模型、仅覆盖英语数据、实验基准为模板化提示而非真实对话。

---

## 70. Design Concept: Scaffolding Geopolitical Reflection Among Tech Workers

**arXiv ID:** 2607.28904 | [PDF](https://arxiv.org/pdf/2607.28904v1)

**作者:** Sydney Reis `[一作]` `[通讯]` (University of Oxford), Sydney Reis (University of Oxford)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个 AI 驱动的交互式叙事系统，帮助科技公司员工在工作坊中进行地缘政治反思。

**💡 创新点**

通过科幻叙事与原型分配的方式，提供一种不带规范、以工作坊为载体的反思平台，避免传统责任工具的局限。

**🔧 技术方法**

采用大语言模型（LLM）结合 Retrieval-Augmented Generation（RAG）微调的后台 AI，前端使用聊天式界面与定制数字艺术。

**📊 数据集**

训练与检索语料主要来自历史技术人物与领导者的文本、地缘政治叙事、公开政治与技术文献等公开资料。

**📈 对比分析**

目前未进行量化实验，计划通过案例研究、访谈和工作坊评估反思效果；预期相较传统工具能更有效激发讨论。

**⚠️ 局限性**

可能导致过度沉思、工作场所的社会判断、数据偏见以及对叙事偏向的误解；缺乏实证验证，实施成本与时间投入较高。

---

## 71. Model-Driven Data Contracts for Digital Twin Services

**arXiv ID:** 2607.28803 | [PDF](https://arxiv.org/pdf/2607.28803v1)

**作者:** Philipp Zech `[一作]` (University of Innsbruck), Istvan David `[通讯]` (McMaster University)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5041475393)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了面向数字孪生服务的数据契约概念，提供正式框架、DSL规范及架构支撑，以显式描述模型对数据质量的假设与承诺并在运行时监测违约。

**💡 创新点**

首次将合同理论与数据质量属性结合，构造了可在设计时检查兼容性、运行时检测违约并触发补偿的完整模型驱动数据契约体系。

**🔧 技术方法**

采用形式化合同理论、领域特定语言（DSL）、模型驱动工程（MDE）技术，并利用监控器生成与数据流交互的实现。

**📊 数据集**

未在论文中使用真实数据集，示例仅以“智能建筑数字孪生”中的占用率与天气数据为假设。

**📈 对比分析**

通过形式化验证展示设计时兼容性检查和运行时监测示例，未给出量化性能指标，但提出可通过监控合成评估可靠性提升。

**⚠️ 局限性**

局限包括DSL可扩展性有限、源发现与补偿机制尚未实现、监控开销与可观测性需求需进一步验证。

---

## 72. Optimizing Monetization Strategies for Generative AI Firms: Implications for Search Engagement

**arXiv ID:** 2607.28780 | [PDF](https://arxiv.org/pdf/2607.28780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 73. To Facilitate or not to Facilitate: Human and LLM Facilitator Tendencies in Online Discussions

**arXiv ID:** 2607.28643 | [PDF](https://arxiv.org/pdf/2607.28643v1)

**作者:** Dimitris Tsirmpas `[一作]` (Athens University of Economics and Business), John Pavlopoulos `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 3721 | [OpenAlex ID](https://openalex.org/A5033894687)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了统一的多域促会话数据集PEFK，并通过专家调查与大模型对比，研究人类与LLM在何时介入讨论的差异，同时评估不同模型在预测促会话介入时的表现。

**💡 创新点**

创新点在于：①首次系统整理并标准化多域促会话数据；②首次将专家调查与LLM“裁判”相结合，量化人类与LLM在介入时机上的差距；③发现LLM过度介入并偏向正面强化，并证明传统的BERT级分类器在此任务上能更稳定、性能更优。

**🔧 技术方法**

主要技术包括：对话文本预处理、prompt设计、LLM推理（如LLaMA、Qwen等），以及基于Transformer的ModernBERT分类器训练与调参。

**📊 数据集**

使用了包括WikiDisputes、WikiConv、Conversations Gone Awry、ceri、wt、iq2、whow、Fora等多域文本与口语讨论数据，统一合并为PEFK。

**📈 对比分析**

比较方法：对六种LLM进行“裁判”式介入预测，分别与训练好的ModernBERT分类器在各子数据集上进行对比；结果显示LLM在识别不需要介入时（F1_n）表现良好，但在需要介入时（F1_p）普遍较差；ModernBERT在大多数子集上取得更高的F1_p与整体宏观F1，且可通过阈值调节介入阈值，实现更可控的介入策略。

**⚠️ 局限性**

局限性包括：①PEFK仍以英语为主，缺乏多语言与跨文化验证；②讨论仅截取少量段落，缺乏完整上下文与长期互动信息；③LLM解释性不足，缺少模型内部决策依据；④实验中仅考察正面与负面强化两种介入标签，未覆盖更细粒度或多样化的介入策略；⑤数据集规模和多样性受限，难以突破性能上限。

---

## 74. Repository-Aware Metamorphic Relation Generation for Augmented Reality Applications using Large Language Models

**arXiv ID:** 2607.28775 | [PDF](https://arxiv.org/pdf/2607.28775v1)

**作者:** Dibyendu Brinto Bose `[一作]` (Virginia Tech), Chris Brown `[通讯]` (Virginia Tech)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建一种基于仓库上下文的元模型关系（MR）生成与推理框架，自动为增强现实（AR）应用产生可执行的测试断言。

**💡 创新点**

创新点：① 将仓库、类和方法层级信息结构化为三层上下文（A0、A1、A2）供 LLM 生成 MR；② 引入多代理“辩论”机制，对不同上下文生成的候选 MR 进行冲突识别、冗余消除和融合，显著提升 MR 的覆盖率、可执行性与一致性。

**🔧 技术方法**

技术手段：① Llama‑3.2‑3B 负责高效生成 MR；② Qwen‑2.5‑Coder‑32B 负责辩论式推理与精炼；③ 设计三种 prompt 配置（仅方法、扁平上下文、层级上下文）以及结构化 MR 模式；④ 采用统计阈值（p10）和六项结构有效性指标评估 MR；⑤ 人工评审与基准 mutation 测试验证 MR 的实用性。

**📊 数据集**

数据集：142 个 Unity‑based AR 开源仓库，共 5,167 个类‑方法目标，生成 14,916 条候选 MR。

**📈 对比分析**

比较方法：对 A0、A1、A2 进行覆盖率、冗余率、结构有效性（SVS）对比；对辩论结果做冲突/冗余统计，评估最终 MR 的长度、重复率及被选取比例；人工评估 141 条精炼 MR 的有效性、可执行性与 AR 专用性；并在 5 个实例上执行 property‑based 测试检验 mutation 死亡率。性能：A2 获得 7,004 条 MR，唯一率 86.7%；辩论后 exact duplicate 降至 1.3%，88.2% MR 来自 A2；人工评分平均有效性 1.80、可执行性 1.82，61.7% MR 通过高质量阈值。

**⚠️ 局限性**

局限性：① 扁平上下文 A1 在源追踪上表现不佳；② 生成与推理 LLM 规模不匹配，可能影响辩论结果；③ 仅在 Unity C# AR 项目上验证，跨域/跨语言迁移需进一步研究；④ 需要人工审查以筛选高质量 MR；⑤ 评价尺度及人类评审间一致性仍偏低。

---

## 75. Demystifying Entropy-based Selection for Chain-of-Thought Compression in Large Reasoning Models

**arXiv ID:** 2607.28707 | [PDF](https://arxiv.org/pdf/2607.28707v1)

**作者:** Sara Candussio `[一作]` (University Of Trieste), Gabriele Sarti `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了基于熵的链式思维（CoT）压缩方法在不同模型、不同推理任务中的鲁棒性，并与随机裁剪做对比；

**💡 创新点**

创新点在于揭示熵在数学任务中表现优异并非熵本身，而是低熵数字令牌的重叠；

**🔧 技术方法**

使用熵计算、句子/令牌级裁剪、激活补丁技术以及相对性能保留率（RPR）与AUC等评估指标；

**📊 数据集**

实验数据来自AIME、MATH‑500、ZebraLogic、GPQA‑Diamond等多种推理数据集，涵盖六个开放权重模型；

**📈 对比分析**

与随机基线对比，句子级熵裁剪无优势；令牌级熵在数学任务略优，激活补丁后可在10–20%令牌保留率下恢复≈90%准确率；非数学任务几乎无显著差异；

**⚠️ 局限性**

局限性包括仅测试六个开放模型、有限任务规模、对激活补丁的内部访问依赖、固定保留率设定以及对闭源或更大模型的泛化尚未验证。

---

## 76. Model or Harness? An Interaction-Centric Taxonomy for Localizing Agent Failures

**arXiv ID:** 2607.28802 | [PDF](https://arxiv.org/pdf/2607.28802v1)

**作者:** Harsh Raj `[一作]` (Scale AI), Yunzhong He `[通讯]` (Scale AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个交互中心的代理失效分类体系，识别并归纳了 41 种失效模式，并通过交互边和失效侧定位故障来源。

**💡 创新点**

创新点在于：①从交互视角而非单一模块划分失效；②为每种失效定义“失效侧”，直接关联可行的修复策略；③使用 LLM 作为独立评判者验证分类的一致性与可复制性。

**🔧 技术方法**

采用了根因追踪、交互边/失效侧标签化、LLM-as-a-judge（GPT‑5.5、Claude Opus 4.6‑4.8）进行标签恢复，以及选择性投票(Selective‑Voting)集合以提升精度。

**📊 数据集**

数据来源包括公开基准测试、模型系统卡、论文、博客帖子、GitHub Issue 与 Agent 轨迹日志，覆盖多种代理架构与任务场景。

**📈 对比分析**

通过与人工标注的对比评估：分类标签的 Cohen’s κ 达到 0.76（GPT‑5.5），最高单例 F1 约 0.80；完整失效模式的 κ 为 0.71；选择性投票在 3/4 投票阈值下实现 90% 覆盖率时，分类精度 0.83，4/4 阈值下精度 0.96。

**⚠️ 局限性**

局限性：①分类为描述性，未提供失效频率或严重度量；②对证据完整性依赖较高，短报告或系统卡可能导致根因不唯一；③LLM‑judge 的准确性有限，尤其在失效模式细分上；④集合方法虽然提高精度，但覆盖率下降，可能在最不确定的案例上产生拒绝。

---

## 77. Structured AI Demonstrations and Student LLM Use in Engineering Mechanics: Study Design and Preliminary Results

**arXiv ID:** 2607.28710 | [PDF](https://arxiv.org/pdf/2607.28710v1)

**作者:** Shuang Geng `[一作]` (Boston University), Emma Lejeune `[通讯]` (Boston University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在2026春季学期的本科工程力学课程中，对学生使用大型语言模型（LLM）的频率、态度、验证习惯和课程成绩进行了描述性研究，并提供了可复现的问卷工具与九段教师主导的AI演示材料。

**💡 创新点**

首次公开完整的研究设计、调查问卷与AI演示资源，为工程教育领域提供了可复制的实验框架，填补了针对专业课程的LLM使用经验与教学干预的实证空白。

**🔧 技术方法**

使用了Qualtrics在线问卷收集自评数据、Google Gemini等LLM进行演示、三大LLM模型进行文本编码的机器学习技术，以及Python脚本进行描述性统计与可视化。

**📊 数据集**

数据来源于Boston University EK301课程的105名自愿参与学生，包括预/后期调查、期中考试成绩（z分）与自由文本反思，已公开存储在OpenBU的研究仓库中。

**📈 对比分析**

采用描述性统计、堆叠条形图和交叉表分析方式检视AI使用与成绩的关系，未进行显著性检验；结果显示LLM使用普遍且频率提升，使用与成绩的关联呈多样化，AI演示参与度高但对学习影响呈分化，整体学业表现无明显提升。

**⚠️ 局限性**

局限性包括样本量小、非随机分配导致潜在混杂、依赖自报数据易受偏差、AI演示未标准化、无法区分因果方向、数据共享受限导致无法完整复现、缺乏长期跟踪评估。

---

## 78. Reasoning in Real World Clinical Care: Why Large Language Models Are Not Yet Safe for Autonomous Clinical Decision Support

**arXiv ID:** 2607.28677 | [PDF](https://arxiv.org/pdf/2607.28677v1)

**作者:** Shayndhan Sivanathan `[一作]` (Atman Labs), Prakash Jayakumar `[通讯]` (University of Texas at Austin)

**通讯引用:** 3411 | [OpenAlex ID](https://openalex.org/A5011496424)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

阐述大语言模型在自主分诊中的安全缺陷，并提出基于不完整信息的评估框架

**💡 创新点**

指出现有评估基准无法测量从沉默推理和不对称成本下的安全决策，提出安全评估的三项要求

**🔧 技术方法**

采用对比分析、系统性综述与实验验证的研究方法，聚焦现有LLM评估与真实临床分诊的差距

**📊 数据集**

使用公开的LLM基准（如医学许可考试、OpenAI o1评测）和真实患者对话数据（如公开对话、肯尼亚基层诊所实验）

**📈 对比分析**

与传统基准对比，发现LLM在缺失信息情境下诊断准确率下降至34%以下，凸显安全性能不足

**⚠️ 局限性**

局限在于缺乏对罕见病症的真实测试、评估仅在模拟环境，未充分验证模型在实际分诊中的可靠性

---

## 79. Reflected UAS: Corrected Deterministic Stability and Direct CTMC Drift Calculation

**arXiv ID:** 2607.28688 | [PDF](https://arxiv.org/pdf/2607.28688v1)

**作者:** Krishna Subedi `[一作]` `[通讯]` (Neryva), Krishna Subedi (Neryva)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在子临界负载下异构多服务器队列的反射 UAS 路由，证明其反射 ODE 存在唯一的边界平衡点，并提供了直接的 CTMC Foster–Lyapunov 驱动不等式，从而实现了稳定性证明。

**💡 创新点**

创新点包括：①在边界上确定了唯一平衡并构造了凸势函数；②揭示并纠正了传统将确定性势函数提升到随机过程的错误，提出加权二次 Lyapunov 方案；③利用软最大化的最小化界限实现了 CTMC 的稳定性证明。

**🔧 技术方法**

主要技术手段是反射 ODE 与凸分析、势函数梯度结构、软最大化推理、Foster–Lyapunov 驱动、数值验证（激活集一致性、误差上界）等。

**📊 数据集**

实验数据基于十服务器基准（α=20,β=0.85,γ=0.5,c=0.5,λ=11.2）以及对称、高负载、轻负载等四个 toy grid，所有状态均通过离散验证；仿真评估在单一参数点上进行。

**📈 对比分析**

与未改软最大化的 UAS 以及最短期望延迟的 JSSQ 进行对比，采用独立种子块（1000/2000/3000）仿真。结果显示 Reflected UAS 的平均总队列长度、Gini 指数和平均停留时间均优于两者，平均差距约为 1.43 与 0.96。

**⚠️ 局限性**

局限性在于仅在子临界负载下验证，未给出完整的 CTMC 正 Harris 递归证明；软最大化在流量极限下会硬化，无法直接应用 Dai 风格的流体极限；实验仅针对单一参数点和有限种子块，缺乏普适性证明。

---

## 80. Bayesian Posterior Sampling for Synthetic Shape Generation of Heart Valves

**arXiv ID:** 2607.28914 | [PDF](https://arxiv.org/pdf/2607.28914v1)

**作者:** Vijay Dubey `[一作]`, Jan Fuhg `[通讯]` (University of Texas at Austin)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于贝叶斯后验采样的心脏瓣膜形状生成框架，并在三尖瓣的3D超声数据上进行验证；

**💡 创新点**

创新点在于使用多模态 Gaussian Mixture Prior 与形状质量分类器，并结合 Hamiltonian Monte Carlo（NUTS）实现条件采样，克服了传统 PCA‑SSM 的多模态不足和缺乏有效性约束的问题，尤其在样本量极少时仍能生成符合生理的形状；

**🔧 技术方法**

技术包括 NURBS 表示、POD 降维、Gaussian Mixture Prior、MLP 形状质量分类器、NUTS 采样、自动微分（PyTorch/JAX）以及统计评估指标（density、coverage、Wasserstein 等）；

**📊 数据集**

使用了参数化主动脉瓣的仿真数据、10 例成人三尖瓣的手工分割 3D TEE 图像，以及二维模型验证数据；

**📈 对比分析**

与传统 PCA‑SSM 及生成‑拒绝（GR）方法对比，在不同样本量下评估 valid fraction、density、coverage、Wasserstein；结果显示 NUTS 在少量样本（<20）时 valid fraction 更高，覆盖率优于 SSM，且相比 GR 在采样效率和探索空间上更具优势；

**⚠️ 局限性**

局限性包括：需要可微分的生成器与分类器；形状质量标签需人工标注，标注成本高；分类器仅基于几何特征，未涵盖所有临床约束；多形态瓣膜的条件采样仍需更大样本量。

---

## 81. Learning Manifolds in High-D Point Embedding for Anisotropic Surface Approximation from Unstructured Point Clouds

**arXiv ID:** 2607.28855 | [PDF](https://arxiv.org/pdf/2607.28855v1)

**作者:** Hongbo Li `[一作]` (Wayne State University), Zichun Zhong `[通讯]` (Wayne State University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种基于高维欧氏点嵌入的可扩展深度学习框架HD-PEA，用于从无结构点云直接生成高质量的各向异性表面网格并估计曲率张量。

**💡 创新点**

①设计可扩展的高维点嵌入方法，保留黎曼度量并克服传统网格依赖；②提出基于高维嵌入的切空间近似与稀疏采样；③将各向异性网格重建与表面重建协同结合；④引入补丁元嵌入推理（PMEI）实现大规模点云无需再训练。

**🔧 技术方法**

深度学习高维嵌入网络、邻域距离嵌入损失（NDHDE+MAE）、切空间近似（TSA）、高维空间各向异性重建、粒子优化与Voronoi/质心Voronoi结构、曲率张量估计（SVD）等技术。

**📊 数据集**

训练集：Thingi10K 240 模型（扩展至 2400 模型）采样 40K 点；测试集：Thingi10K、AIM@Shape/Stanford 3D、TetWeave、ScanNet 等多种数据集。

**📈 对比分析**

与 NMC、NDC、VoroMesh、PoNQ、LMR 等最先进表面重建方法在 Chamfer、F1、Normal Consistency、Hausdorff、网格质量 G 以及运行时进行对比。HD-PEA 在表面精度、网格质量、轻量化和可扩展性方面显著优于对比方法，尤其在大规模点云上保持高效率。

**⚠️ 局限性**

对稀疏或严重缺失的点云（如薄结构、扫描遮挡）重建效果受限；高维嵌入计算成本较高；在极端噪声或低采样率情况下可能出现几何失真。

---

## 82. Can Synthetic Data Overcome the Generalization Limits of AI-Based Flower and Pod Detection Across Cowpea Breeding Genotypes and Environments?

**arXiv ID:** 2607.28796 | [PDF](https://arxiv.org/pdf/2607.28796v1)

**作者:** Hamid Kamangir `[一作]` (University of California Davis), J. Mason Earles `[通讯]` (University of California Davis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文研究了种植基因型与环境(G×E)变异对牛豆花朵与荚果检测的影响，并探讨利用对齐的合成图像与少量真实样本实现跨环境泛化。

**💡 创新点**

创新点在于将基于Wasserstein距离的域差异优化与HDR渲染相结合，显著降低合成到真实的域差距，并证明HDR+优化的合成数据可在仅5-10张真实图像下匹配或超过完整真实基线。

**🔧 技术方法**

使用YOLOv11x目标检测框架、Helios物理渲染、HDR(EXR)与8‑bit JPEG处理、域差异优化的相机真实感增强、DINOv2嵌入与图像质量统计进行评估。

**📊 数据集**

数据集为加利福尼亚Davis和Kearney两地2022、2023年牛豆田间共约1584张标注图像（花朵6,926例、荚果4,998例），以及1,000张Helios生成的合成图像。

**📈 对比分析**

在未见基因型/环境的测试中，零-shot合成+优化可将花朵mAP从2.6%提升至22.8%，荚果从5.1%提升至10.6%；在一、五、十拍摄样本下，HDR+优化合成+5张真实图像即可在空间泛化中达到或超过完整真实基线，性能提升约10–20%，相较未优化或8‑bit合成性能明显低。

**⚠️ 局限性**

局限包括：对年份迁移的适应仍有限，尤其荚果检测在2022年迁移下未能完全匹配真实基线；HDR渲染与优化依赖复杂物理参数，实用性受限；仅在特定任务与环境验证，未证明对其他作物或检测任务的普适性。

---

## 83. Measuring Cognitive Engagement in Collaborative Discourse with an Extended ICAP Framework: Comparing Human Annotation, In-Context Learning, and Reflective LLM Agents

**arXiv ID:** 2607.28651 | [PDF](https://arxiv.org/pdf/2607.28651v1)

**作者:** Lan Anh Do `[一作]` (Tufts University), Ayanna K. Thomas `[通讯]` (Tufts University)

**通讯引用:** 2088 | [OpenAlex ID](https://openalex.org/A5011899252)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究利用扩展的7分ICAP框架，对三人小组逻辑推理任务中的合作对话进行认知参与度编码，并比较了人工编码、ICL提示和LLM代理自适应编码的可靠性；

**💡 创新点**

提出了在LLM代理中实现自我反思与框架迭代的流程，探讨了人类与LLM在认知参与度编码上的差异，并验证了中间等级类别的模糊性对一致性的影响；

**🔧 技术方法**

使用GPT‑4o、GPT‑5.2两大模型，实施少量示例ICL提示、零样本提示以及自我反思的LLM代理；

**📊 数据集**

基于42个三人小组的对话录音与转录，涵盖11分钟左右的推理过程，共计≈11次试验；

**📈 对比分析**

对比人类互评、LLM代理互评以及人机互评的加权Kappa、Krippendorff α等指标，结果显示：人类互评始终高于LLM互评；LLM代理自适应框架在模型间的相互一致性略优于固定ICL/零样本提示；人机一致性虽有提升但仍显不足；

**⚠️ 局限性**

主要限制包括：LLM仅接受文本输入，缺乏视频中的非语言线索；人工与代理的框架细化过程不完全同步；LLM对中间等级的模糊性难以像人类一样通过讨论精细化判定；

---

## 84. COSI-Lab: Conference Living Lab for Modeling Multi-Perspective Multimodal Social Intention

**arXiv ID:** 2607.28649 | [PDF](https://arxiv.org/pdf/2607.28649v1)

**作者:** Zonghuan Li `[一作]` (Delft University of Technology), Hayley Hung `[通讯]` (Delft University of Technology)

**通讯引用:** 3291 | [OpenAlex ID](https://openalex.org/A5061353073)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了COSI‑Lab，一个包含32位学者在真实弱剧本社交活动中同步采集的多模态数据集，并提供了面向观测者的“显性意图推断”（AII）注释流程和基准任务；

**💡 创新点**

创新点在于：①提出AII任务，将意图视为可被多方解释的多假设过程；②设计了基于观测者特征和3C框架的多视角叙事注释方法；③构建了具有生态效度的弱剧本多模态社交数据，并实现了多模态与多观点的可解释意图分析；

**🔧 技术方法**

使用的技术包括多摄像机三维姿态重建（SAM3+ViTPose）、语音伪匿名化、UWB定位、IMU手势捕捉；注释中引入了3C框架、LLM‑as‑Judge、Graph‑based DANTE和LSTM模型进行基准评估；

**📊 数据集**

使用的数据集为本研究收集的COSI‑Lab数据集；对比实验中亦参照了现有的多模态意图与行为数据集（如PIE、PSI、Ego‑Exo4D等）以做对照；

**📈 对比分析**

在意图推断上，模型生成的叙事与人类注释在语义相似度上存在明显区分，但在解释细节上重叠度高；LLM‑as‑Judge对模型输出的辨别准确率仅为45.3%，表明模型与人类注释相似度高；在对话组检测任务中，基于Graph‑DANTE和LSTM的基线模型在F1指标上均达到了90%以上，验证了数据的可用性；

**⚠️ 局限性**

局限性包括：①未能捕捉不可观测的非显性意图；②仅针对30秒窗口，缺乏跨段时间推断；③缺乏对音视频线索与意图叙事的严格 grounding 验证；④低保真与高保真模态的交互效应尚未深入探讨；

---

## 85. Learning Stateful Predictive Knowledge From Experience

**arXiv ID:** 2607.28638 | [PDF](https://arxiv.org/pdf/2607.28638v1)

**作者:** Yan Song `[一作]` (University College London), Jun Wang `[通讯]` (University College London)

**通讯引用:** 46841 | [OpenAlex ID](https://openalex.org/A5100384686)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Stateful Knowledge Learning（SKL）框架，强调在LLM代理中维护状态化、可声明的预测知识并进行自我学习与推断；

**💡 创新点**

创新点在于把传统的轨迹级反思转向状态级预测知识的学习与自我回放，支持知识的时间递归更新与跨状态迁移；

**🔧 技术方法**

使用自蒸馏（SKL‑SD）与基于强化学习的自模拟（SKL‑RL）两种算法，结合Bootstrapping、GRPO、SFT等技术；

**📊 数据集**

实验数据集包括WebShop、ScienceWorld、FrozenLake以及ChessPuzzles，均为开放式交互式环境与棋类推理任务；

**📈 对比分析**

与现有轨迹级反思基线（如GRPO、Reflect‑GRPO、Critique‑GRPO、R³L）对比，SKL‑SD在WebShop上提升约+3.3%点，SKL‑RL在ChessPuzzles上实现了最优的自适应学习与测试性能；

**⚠️ 局限性**

主要局限在于训练计算开销显著增加（特别是SKL‑RL的在线模拟与Bootstrapping导致时间膨胀），且对长时序任务的可扩展性尚未验证。

---

## 86. TELLER: Dual-Path Iterative Preference Optimization for Table Entity Linking

**arXiv ID:** 2607.28680 | [PDF](https://arxiv.org/pdf/2607.28680v1)

**作者:** Yixin Peng `[一作]` (RWTH Aachen), Stefan Decker `[通讯]` (RWTH Aachen)

**通讯引用:** 17477 | [OpenAlex ID](https://openalex.org/A5071104283)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TELLER框架，针对表格实体链接采用候选条件生成，并通过两条路径（直接答案与推理）进行迭代偏好学习，显著提升模型准确率。

**💡 创新点**

创新点在于：①使用离线候选检索并精简表格上下文，仅保留目标行列、表头和标题；②引入两种迭代偏好学习——直接答案的DPO与推理的长度归一化L-RPO；③在推理路径中加入选取-拒绝对的长度归一化与已选答案的似然正则化，解决长度偏好与答案可靠性问题。

**🔧 技术方法**

技术包括：Llama 3.1 8B 基础模型 + LoRA 微调；候选检索使用本地 Wikidata 语义相似度+BM25；数据生成使用 DeepSeek‑V4‑Pro 与 GPT‑5.2 产生推理说明；偏好学习采用 DPO 与自定义的 L‑RPO；实现细节涉及 bfloat16、cosine LR 调度等。

**📊 数据集**

使用的数据集为 TableInstruct（实体链接子集）与 MammoTab V2（医学表格），并在后者的 9,741 个提问上进行最终评测；候选集合从本地 Wikidata 构建。

**📈 对比分析**

与公开基线对比：直接答案路径在 MammoTab V2 上达 0.882 CEA，超过 TableLlama（0.86）和其他方法；在 TableInstruct 上从 94.35% 提升至 94.50%；推理路径虽然准确率略低，但完整推理率提升到 99.55%（TableInstruct）与 91.86%（MammoTab V2）。

**⚠️ 局限性**

局限性包括：①所有训练实例都包含黄金实体，未处理检索失败导致的 NIL 场景；②推理说明由教师模型生成，重现性受限；③迭代偏好学习仅做了两轮，尚未探索更长迭代的效果；④模型未学习对未检索到实体的显式 NIL 预测。

---

## 87. Representations from Pretrained Machine-Learning Interatomic Potentials as Coarse Coordinates for Material Generation and Evaluation

**arXiv ID:** 2607.28776 | [PDF](https://arxiv.org/pdf/2607.28776v1)

**作者:** Paul Hagemann `[一作]` (Bundesanstalt für Materialforschung und -prüfung), Philipp Benner `[通讯]` (Bundesanstalt für Materialforschung und -prüfung)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于双重特征提取的生成模型评估指标Coarse‑Fine Transport Distance (CFTD)，并在其上构建了可用于材料生成的MACE‑条件生成器；

**💡 创新点**

创新点在于将粗粒度的MACE隐藏特征与细粒度的对比学习GNN特征分离，形成“金发小姐”区域（Goldilocks zone），从而在同一指标中同时衡量质量覆盖和新颖性，解决了传统指标中两者的冲突；

**🔧 技术方法**

核心技术包括：MACE预训练的MLIP隐藏特征（取均值池化后随机投影），InfoNCE对比学习的身份特征（GNN），最优传输（OT）求解，随机投影降维，Gaussian噪声与Pettifor邻域替换的增强；

**📊 数据集**

主要数据集为Materials Project的MP20（训练/验证/测试集），并在该数据集上训练与评估所有生成模型（MatterGen、ADiT、Chemeleon、DiffCSP等）以及自建的MACE‑条件生成器；

**📈 对比分析**

与连续SUN、TNovD等现有指标相比，CFTD能更好地检测模式坍塌、过拟合与结构不稳定性，并在实验中给出生成模型的排名；在MACE‑条件实验中，MACE-conditional模型在稳定性和有效率上优于无条件模型，但新颖性略低；

**⚠️ 局限性**

局限性包括：对训练集的依赖（需在相同数据上预训练身份特征），隐式稳定性指标（缺少明确的能量上边界评估），以及对更复杂结构（如缺陷、表面等）的适应性待验证。

---

## 88. When Unlearning Fails: Reliable Data Deletion under Post-Training in Agent Networks

**arXiv ID:** 2607.28829 | [PDF](https://arxiv.org/pdf/2607.28829v1)

**作者:** Zihao Ding `[一作]` (South Dakota State University), Liang Dong `[通讯]` (Baylor University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在自我改进的联邦代理网络中，提出一种可靠的数据删除方法（MUTE），解决遗忘数据的影响回声，并在部署后持续保证删除效果。

**💡 创新点**

创新点在于：①通过服务器轻量账本对遗忘数据在聚合与采集链路中的影响进行可追踪的估计；②结合模型侧忘却与数据侧隔离（隔离/下调），消除当前残留和未来潜在的影响；③通过行为泄露与影响再生的审计与调度，在上传带宽预算内动态维护删除状态。

**🔧 技术方法**

使用技术包括联邦学习、低秩适配器（LoRA）、负偏好优化（NPO）、影响评分与账本回放、行为泄漏指数（BLI）与影响再生率（IRR）审计、阈值隔离与下调、通信预算调度。

**📊 数据集**

实验数据集为 LIBERO 语言条件操控任务（四个子套件），使用 MiniVLA 与 π_0 两种 VLA 模型进行验证，并在 Jetson Edge 真实测试平台上验证。

**📈 对比分析**

与全量重训练（Retrain）和理论理想网络（θ⋆）对比，MUTE 在任务成功率（SR）保持 ≤0.04 的下降、BLI 接近 0.5 的基准、IRR 低于重训练且显著降低，且通信成本（Comm）比重训练低 70% 以上。

**⚠️ 局限性**

局限性包括：①需要在服务器上维护完整账本，且阈值（τq、η 等）需要手动调优；②在数据粒度较粗（任务级）或极端异构场景下，影响回声仍可能显著；③实验仅覆盖两种 VLA 架构和 LIBERO 任务，泛化性待进一步验证。

---

## 89. Bridging the Gap Between PHE and FHE: A Performance and Trade-off Analysis of The Somewhat Homomorphic BGN Cryptosystem

**arXiv ID:** 2607.28700 | [PDF](https://arxiv.org/pdf/2607.28700v1)

**作者:** Sefik Serengil `[一作]` (Neo4j), Alper Ozpinar `[通讯]` (Ibn Haldun University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现并评估了Boneh‑Goh‑Nissim（BGN）部分同态加密系统的Python开源实现，并与传统部分同态加密（PHE）和全同态加密（FHE）进行对比。

**💡 创新点**

突破性点在于：①将BGN的数学复杂度隐藏在易用的Python API中；②通过对比表明BGN在通信成本上远低于FHE；③提出了仅需一次乘法的SWHE架构在向量相似度、距离度量等机器学习任务中的可行性。

**🔧 技术方法**

使用了椭圆曲线双线性配对、整数分解与离散对数、以及现有的Python库（lightphe、TenSEAL）进行实现与基准测试。

**📊 数据集**

采用LFW（Labeled Faces in the Wild）人脸嵌入向量作为128维测试数据集，亦使用通用PHE和FHE参数集。

**📈 对比分析**

方法：在相同硬件（Apple M4 Max）上对128维向量的加密、同态运算（加、标量乘、一次乘法）以及解密进行多次测量；与Paillier、Damgård‑Jurik、Okamoto‑Uchiyama以及TenSEAL CKKS进行加密/解密时延、通信量比较。结果显示：BGN在加密/运算/解密上比PHE慢数十倍，远慢于TenSEAL，但其公钥和密文尺寸仅为3‑6 KB（相对FHE的45‑451 MB），且在仅2位小数精度下保持与明文相同的相似度排序。

**⚠️ 局限性**

限制：①BGN的计算耗时严重，尤其是加密/乘法阶段需昂贵的双线性配对；②解密时需在目标群求离散对数，精度过高会导致不可接受的解密延迟；③仅支持一次乘法，无法满足深度非线性计算；④对量子安全性不具备保障。

---

## 90. Guarantees on Dynamical System Distinguishability for LLM Token Generation

**arXiv ID:** 2607.28667 | [PDF](https://arxiv.org/pdf/2607.28667v1)

**作者:** Mohamed Akrout `[一作]` (University of Tennessee), Dan Wilson `[通讯]` (University of Tennessee)

**通讯引用:** 8155 | [OpenAlex ID](https://openalex.org/A5042018242)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过把LLM生成的token嵌入视为黑盒动力学系统的轨迹，研究了利用动态预测残差对答案真假进行二分类的理论基础。

**💡 创新点**

创新点在于①证明仅靠边缘分布难以区分两类系统，给出误差下限；②给出利用动力学残差的误差指数衰减上界并引入可分离度量；③建立跨嵌入迁移的理论条件，量化互补映射对可分离度的影响。

**🔧 技术方法**

主要技术包括：扩展动态模式分解（EDMD）拟合Koopman算子、离散Lyapunov方程求解、统计学习中的总变差距离与贝叶斯误差分析、线性系统辨识与可观测性理论。

**📊 数据集**

使用了HaluEval数据集（约80,000条hallucinated与correct响应），以及对不同嵌入模型（如Llama-3、Jina‑v5等）的token嵌入。

**📈 对比分析**

实验与理论对比表明：误差随序列长度L按指数下降，所需tokens数满足O(log(1/ε)/δ²)；跨嵌入迁移时精度≥50%并可达≈63%，验证了理论上限。

**⚠️ 局限性**

局限包括：仅考虑线性Koopman近似，噪声假设为高斯，未探讨非线性或重尾分布；未给出联合优化嵌入与算子的方法；对不同LLM架构的泛化需进一步验证。

---

## 91. LAWFUL: Law-Aligned Witness for Faithful Use of Latents

**arXiv ID:** 2607.28672 | [PDF](https://arxiv.org/pdf/2607.28672v1)

**作者:** Kevin Chen `[一作]` (Ohio State University), Anish Arora `[通讯]` (Ohio State University)

**通讯引用:** 7989 | [OpenAlex ID](https://openalex.org/A5079903777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 LAWFUL 框架，用以检验神经网络是否在内部学习并使用了已知的物理定律，并以此对模型行为进行因果一致性评估。

**💡 创新点**

创新点在于：①设计了覆盖感知的连续对因果一致性度量；②引入了物理定律的有效域（domain‑of‑validity）测试；③将电路识别与一致性度量结合，得到可解释的物理一致性子图；④首次将这些方法应用于 MoCap‑to‑Radar 的 Doppler 频率学习。

**🔧 技术方法**

采用了机制解释技术：物理桥接（input/output bridge）、连续对因果扰动族、激活补丁（activation patching）、物理一致性评分（consistency score）以及子图搜索来识别电路。

**📊 数据集**

使用了人类运动捕捉（MoCap）与 Bumblebee 同频雷达的联合数据，构造了 RandomWalk 评估集，包含多段重叠的 STFT 窗格。

**📈 对比分析**

方法对照未补丁模型（M₀≈0.988）与电路补丁结果，发现 τ‑一致性子图在 0.9 置信度下恢复 91% 的一致性；与随机源补丁控制相比，电路补丁的 Δₛ₁与 Δₙₑc 显著更好，证明该子图真正承担了物理一致性功能。

**⚠️ 局限性**

局限性包括：仅处理线性单变量定律（Doppler 关系）；仅在单一 Transformer 上验证；只评估频率重心，未覆盖微多普勒或频谱细节；扰动族仅为整体速度缩放，未覆盖更复杂的物理变换；且需要手动给定物理模型，无法自动发现或选择合适的定律。

---

## 92. LARA: Lightweight Adapters in the Residual Stream for Composable Adaptation and Alignment

**arXiv ID:** 2607.28669 | [PDF](https://arxiv.org/pdf/2607.28669v1)

**作者:** Pascal Ekin `[一作]`, Wei Jie `[通讯]` (University of West London)

**通讯引用:** 664 | [OpenAlex ID](https://openalex.org/A5002748740)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在冻结模型的残差流中插入低秩补偿模块，实现参数高效的微调与偏好优化；

**💡 创新点**

证明残差流微调与 LoRA 在等参条件下表现相当，并引入可在推理时调整的比例系数 γ，实现连续调节；同时支持在同一冻结模型上驻留多种行为并按 token 路由；

**🔧 技术方法**

LARA 方法、低秩投影、残差流自适应、软/硬路由、DPO 偏好优化；

**📊 数据集**

代码语料（Python+指令混合）与 UltraFeedback 偏好对；

**📈 对比分析**

在代码微调和 DPO 任务中，LARA 与 LoRA 在相同参数量下获得相近的 perplexity/奖励准确率；LARA 在推理时可通过 γ 线性插值，七种行为在单模型上仅占用约 33 MB；

**⚠️ 局限性**

仅在 1.5B 模型和单一任务上验证，未探测更大规模或多任务泛化；路由精度受相近任务影响，γ 的可控性仅在 perplexity 方面得到证明。

---

## 93. Best Friends, Not Forever: Evaluating Long-Horizon Persona Collapse and Behavioral Drift in AI Companions

**arXiv ID:** 2607.28818 | [PDF](https://arxiv.org/pdf/2607.28818v1)

**作者:** Pranav Narayanan Venkit `[一作]` (Salesforce AI Research), Chien-Sheng Wu `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究针对AI伴侣在长时间对话中的角色保持与轨迹记忆进行系统评估，提出并实现了ANCHOR框架，包含Identity Probe和Trajectory Probe两种探针；

**💡 创新点**

创新点在于将角色保持与轨迹记忆分离评估，设计合成压力日程与多维度问卷，并通过人工评审与对抗性测试同时检验长期连贯性；

**🔧 技术方法**

采用多种技术手段，包括基于问卷的角色保持度量、逐回合人工评判、合成情绪/攻击性日程、三种记忆架构（长上下文、层次摘要、自管理JSON）以及多模型对比实验；

**📊 数据集**

使用27个作者设计的人设卡、9个交互日程、2,008个四款LLM生成的85–130轮会话以及对应的问卷和评审记录构成实验数据集；

**📈 对比分析**

通过对不同模型、记忆设置、日程和评审者的交叉比较，发现无论何种配置，轨迹准确率仅约44%，角色保持率差异显著，评审者间差异明显，整体性能远低于理想连贯性；

**⚠️ 局限性**

局限性包括样本为合成对话且仅限英语、缺乏真实用户与多文化/多语言场景、评审者主观性导致结果不稳定，以及未评估用户信任、健康影响与治理等关键因素。

---

## 94. The Asymmetric Effects of Knowledge Distillation on Bias in Small Language Models

**arXiv ID:** 2607.28639 | [PDF](https://arxiv.org/pdf/2607.28639v1)

**作者:** Plawan Kumar Rath `[一作]` `[通讯]` (Meta), Plawan Kumar Rath (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统研究了知识蒸馏在小型指令调优语言模型上的偏见传递，发现蒸馏在解歧义任务中提升上下文遵循，但在歧义任务中破坏逐项拒绝校准，并提出Per‑Condition Calibration Diagnosis（PCCD）诊断协议来揭示这种非对称影响。

**💡 创新点**

创新点在于揭示蒸馏的非对称偏见效应（上下文遵循提升 vs. 拒绝校准破坏），提出细粒度的PCCD三步诊断方案，证明聚合指标（SRS、CrowS‑Pairs）会掩盖逐项偏见，并对失效机制给出训练语料缺乏拒绝示例的解释。

**🔧 技术方法**

技术方法包括：基于SmolLM2（135M/360M/1.7B）和OLMo‑2（1B）学生模型；采用response‑based、logit‑based、combined三种KD；对BBQ的ambiguity vs. disambiguation子集、CrowS‑Pairs进行评估；使用Spearman相关、silence‑loss、context‑overriding等细粒度指标；构建PCCD三步协议。

**📊 数据集**

数据集主要有：Alpaca‑cleaned（51,760示例）用于训练；教师响应集（Gemma‑2‑9B‑it、Mistral‑7B、Phi‑3.5‑mini）用于蒸馏；BBQ（ambig/disambig各12,148条）和CrowS‑Pairs（1,508对）用于评估。

**📈 对比分析**

在28种配置中，response‑based KD显著降低BBQ‑disambig的context‑overriding率（例如SmolLM2‑1.7B从44%降至24%），但在BBQ‑ambig产生15% silence‑loss；聚合SRS和CrowS‑Pairs指标显示改进，但PCCD发现多数模型未通过所有三步，表明存在隐含的拒绝失效和能力退化。

**⚠️ 局限性**

研究限制包括：仅评估两类小模型（≤2B参数）；只使用MCQ格式的BBQ，未考察自由生成；单一Alpaca‑cleaned训练语料；未检验量化、不同语言或更大模型；logit‑KD在tokenizer不匹配时表现差，且未评估闭源权重模型。

---

## 95. FocusGS: Spatial Delta Layers for Local Repair and Deterministic Editing of Trained 3D Gaussian Assets

**arXiv ID:** 2607.28834 | [PDF](https://arxiv.org/pdf/2607.28834v1)

**作者:** Yiqun Pan `[一作]` (Beijing University of Chemical Technology), Yukun Shi `[通讯]` (Beijing University of Chemical Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

为已训练的3D高斯投影（3D Gaussian Splatting）资产提供局部修复与确定性编辑的通用工具，能在不重新训练全局模型的情况下，对目标区域进行精细细节恢复和文本/图形替换。

**💡 创新点**

创新点包括：① 形式化“空间梯度饥饿”概念，揭示全图优化对小区域细节的抑制；② 通过复合空间增量（base-manipulation + Gaussian-addition）统一修复与编辑任务；③ 引入“擦除-插入因子分解（EIF）”对旧载体进行精准清理并插入新内容，解决Alpha混合下的遗留残影。

**🔧 技术方法**

采用高斯投影渲染、基于多视角的支持束构造、局部增量优化、遮挡保护损失、卷积特征匹配与投影几何约束，以及轻量级2D足迹评分进行旧载体筛选。

**📊 数据集**

主要实验数据集为：4,000多张高分辨率DJI航拍图构建的全景资产；公开编辑案例包括 Books‑I/II、Train、Playroom、3DRealCar 等；用于对比的基线包括 MCMC‑2M、Pixel‑GS、GaussianEditor、DGE 等。

**📈 对比分析**

方法与基线对比：局部修复时 ROI PSNR 提升 7.91 dB，编辑时 Target‑mask PSNR 达 33.17 dB，5/5 OCR 正确率；在 83 次编辑试验中平均 ROI PSNR 提升 11.05 dB。与全局继续训练相比，修复增量仅需 4.6 分钟、显著缩短训练时间且不影响全局上下文。

**⚠️ 局限性**

局限性：需要可靠的相机、几何与可见性信息；主要适用于平面或弱曲面载体（文字、标牌、书脊等），对多深度、透明、强反射或非刚性对象效果欠佳；编辑要求目标清晰且可对齐，自动生成目标需人工确认。

---

## 96. Flow Matching with Missing Data

**arXiv ID:** 2607.28698 | [PDF](https://arxiv.org/pdf/2607.28698v1)

**作者:** Fairoz Nower Khan `[一作]` (University of Kentucky), Peizhong Ju `[通讯]` (University of Kentucky)

**通讯引用:** 150 | [OpenAlex ID](https://openalex.org/A5085838919)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Missing‑Data Flow Matching（MDFM），通过把缺失坐标视为潜在变量，对完整数据的流匹配损失进行平均，从而直接利用不完整数据训练流匹配模型。

**💡 创新点**

创新点包括：①在 MCAR 条件下对流匹配目标实现精确校正（等价于完整数据目标）；②有限样本方差分析揭示一次补全即可匹配完整数据方差；③对学习补全模型引入的不可约偏差给出 Wasserstein 距离上界。

**🔧 技术方法**

技术方法：流匹配框架；对缺失值进行重采样并平均损失；自监督遮蔽训练补全模型；使用多种补全策略（条件流匹配、MICE、MissForest、GAIN 等）进行比较；评估指标包括切片 Wasserstein‑2、能量距离、MMD、协方差误差、RMSE 与 CRPS。

**📊 数据集**

实验数据：合成高斯 AR(1)（维度 10/50/100，缺失率 0.1–0.9）；七个 UCI 表格数据集（行数 178–1797，特征 9–61）。

**📈 对比分析**

与基线方法比较：均值+流匹配、MICE+流匹配、MissForest+流匹配、GAIN+流匹配、经典缺失值填补+流匹配。MDFM 在协方差误差上取得最优，在切片 Wasserstein‑2、能量距离和 MMD 上排名第二，RMSE 仅次于 MissForest。相较于确定性填补，MDFM 能保持条件分布的方差并在高缺失率下表现更好。

**⚠️ 局限性**

局限性：未处理缺失非随机（MNAR）情况；性能依赖于补全模型质量，缺少对生成分布质量的理论保证；集中度结果假设补全模型在与评估数据独立的辅助样本上训练，未考虑同一数据同时用于两者。

---

## 97. Can LLMs Really Understand Item Difficulty Levels? Implications for Automated Item Generation Using LLMs

**arXiv ID:** 2607.28634 | [PDF](https://arxiv.org/pdf/2607.28634v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 98. FairFund-Bench: Evaluating Distributive Bias in LLM Resource Allocation

**arXiv ID:** 2607.28934 | [PDF](https://arxiv.org/pdf/2607.28934v1)

**作者:** Martin Lukk `[一作]` `[通讯]` (University of Toronto), Martin Lukk (University of Toronto)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 FairFund‑Bench，系统地在同一套审计工具中变更任务形式、比较上下文和呈现模式，以评估大语言模型在资源分配决策中的偏见。

**💡 创新点**

创新点在于将审计设计空间拆解为四个维度（评分、排序、分配任务；单/多刺激；透明/伪装呈现），并通过四柱评分框架（人群偏见、可受益度对齐、跨任务一致性、跨上下文一致性）全面刻画模型行为。

**🔧 技术方法**

采用手工编写并校准的模拟捐助请求、精心设计的提示模板、混合效应回归分析以及标准化的 Cohen‑d 指标来衡量并比较模型在不同审计设置下的表现。

**📊 数据集**

使用约 1.29 M 条真实 GoFundMe 公开众筹记录来校准 600 条手写模板，进一步扩展为 3,000 条带有四种种族、两性别、五种因果框架的合成请求。

**📈 对比分析**

对 14 种主流 LLM（包括 GPT‑5、Gemini、Claude 等）在三种任务和两种呈现模式下进行评分，发现整体人群偏差小但方向随审计格式变化；因果框架效应显著且一致，模型在可受益度对齐上差异较大，跨任务与跨上下文一致性普遍较高。

**⚠️ 局限性**

局限包括仅用姓名作为种族/性别信号、缺乏对更多族群和性别维度（如原住民、跨性别、非二元）的覆盖、未引入真实人类决策基准、样本为人工合成且缺乏生态有效性、以及固定的输出与提示参数。

---

## 99. TextCloak: Thwarting Unauthorized LLM Exploitation via RL-Driven Unlearnable Text

**arXiv ID:** 2607.28862 | [PDF](https://arxiv.org/pdf/2607.28862v1)

**作者:** Chengshuai Zhao `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型未经授权的微调，提出一种基于强化学习的文本保护框架 TextCloak，用生成策略将原始文本改写为“不可学习”示例，既保持语义与自然性，又能阻断模型学习；

**💡 创新点**

核心创新在于：①将无学习示例的生成建模为受约束的双层优化；②通过模拟微调过程评估下游性能退化作为奖励；③采用组相对策略优化 (GRPO) 在不需价值网络的情况下更新生成策略；

**🔧 技术方法**

技术方法包括：强化学习（RL）+策略梯度；组相对策略优化；下游性能评估；语义相似度（SBERT）与困惑度（GPT‑2）约束；双层优化（内部微调+外部策略更新）；

**📊 数据集**

使用六个公开数据集（ARC‑Challenge、MATH、MMLU‑Pro、RACE、HumanEval、MedQA‑USMLE），并在九个主流 LLM（Qwen、Gemma、Mistral、GPT‑OSS、Llama、Phi、GLM‑4 等）上进行实验；

**📈 对比分析**

与多种基线（Zero‑Shot、Clean、Random‑Prepend/Append、Textual UE、MEM‑3/5）比较，TextCloak 在所有数据集上平均性能下降 12.4%，显著优于最佳基线；在不同模型架构、微调方式和防御策略下仍保持较高的迁移性与鲁棒性；

**⚠️ 局限性**

局限性包括：① 生成过程依赖大模型与显著计算资源；② 对极端文本长句或多模态输入的适用性尚未验证；③ 在某些微调配置（如全参数微调）下效果略逊，需进一步提升通用性。

---

## 100. Reducing Data Movement in the Galerkin Product of Block Algebraic Multigrid on GPUs

**arXiv ID:** 2607.28891 | [PDF](https://arxiv.org/pdf/2607.28891v1)

**作者:** Mark F. Adams `[一作]` `[通讯]` (Lawrence Berkeley National Laboratory), Mark F. Adams (Lawrence Berkeley National Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了三维弹性问题中 AMG 的矩阵三重乘（Galerkin）在 GPU 上的高效实现与完整流水线。

**💡 创新点**

提出了以 DRAM/L2 传输为驱动的矩阵三重乘算法空间模型，设计了共享内存分块、检视‑执行和转置‑free 等多种变体；引入了保持零空间的块级滤波器；实现了无主机复制的完整 GPU FEM‑AMG 流线。

**🔧 技术方法**

使用了 PETSc 新的块稀疏矩阵类型、Kokkos 与原生 CUDA 后端、共享内存调度、原子免费 owner‑compute、排序调度、块 Frobenius 过滤、GPU 端组装及 MPI 通信避免等技术。

**📊 数据集**

基准数据集为 160×16×16 的 Q₂ 六面体网格（约 1.05M 自由度）弹性梁实验；以及 44³ 节点 Poisson 问题作标量基准；并做弱缩放实验 10k×k×k 梁。

**📈 对比分析**

通过 Nsight Compute 计数 DRAM/L2 传输和时间，对比 V0–V4 变体与 Kokkos/CUDA 后端；V2b（共享内存分块）在细层比 V0 低 1.8×，V1 在高 n_P 下慢 7.9×；V4 在粗层解决占用逆问题；整体 GPU 端流水线实现的总时间比原生 scalar 路径快约 2×，且无设备↔主机复制。

**⚠️ 局限性**

共享内存分块受限于 L2/SM 容量，无法在更大块或更大 L2（如 H100）GPU 上实现；V2c（W‑panel 分块）在当前硬件未见收益；滤波阈值对收敛敏感；当前实现仅在单 GPU/单核 CPU 评估，缺乏大规模多 GPU 的统一评测。

---

## 101. Self-Supervised Skill Optimization

**arXiv ID:** 2607.28777 | [PDF](https://arxiv.org/pdf/2607.28777v1)

**作者:** Siran Peng `[一作]` (Chinese Academy of Sciences), Zhen Lei `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 27729 | [OpenAlex ID](https://openalex.org/A5109299788)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 Self‑Supervised Skill Optimization（SSO）框架，利用无标签任务实例迭代优化冻结大型语言模型（LLM）代理可复用的技能文档。

**💡 创新点**

创新点在于：①以行为级别的差异而非整体技能级别进行比较；②在优化过程中完全不依赖真实标签、奖励或任务专用评估器；③通过LLM生成探针、判断、行为提取、聚类与证据归一化，实现无监督的证据累积与技能更新。

**🔧 技术方法**

核心技术包括：LLM‑驱动的探针生成（f_gen）、LLM 判断器（f_judge）、行为提取器（f_beh）、行为聚类器（f_cluster）以及技能渲染器（f_render）；整个优化循环使用LLM完成所有子任务，并引入验证门（validation gate）以防止退化。

**📊 数据集**

评估数据集：6个闭合式基准（SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMath、ALFWorld）以及3个开放式多轮对话任务（Context Memory、Content Rephrasing、Proactive Interaction），在 GPT‑5.5、GPT‑5.4‑mini、Qwen3.5‑4B 三种冻结模型上进行测试。

**📈 对比分析**

与 GT‑基线（Human skill、SkillOpt 等）、无技能 baseline 以及两种 GT‑free 提示优化器（SPO、PDO）进行对比。SSO 在闭合任务中以无标签方式接近或超过 SkillOpt，在开放任务中对所有基线的赢率均超过 50%，在不同模型上保持显著优势。

**⚠️ 局限性**

局限性：①依赖多次 LLM 推理，计算成本较高；②在需要真实环境交互或极长多步任务的场景下，探针与验证的有效性尚未充分验证；③优化过程需要足够多的无标签验证样本，若验证集不足可能导致收敛不稳定。

---

## 102. TAPR: Enhancing LLM Performance with a Task-Aware Prompt Rewriter

**arXiv ID:** 2607.28657 | [PDF](https://arxiv.org/pdf/2607.28657v1)

**作者:** Oliver Savolainen `[一作]` (University of Amsterdam), Hosein Azarbonyad `[通讯]` (Elsevier)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练一个小型LLM作为任务感知的提示重写器（TAPR），通过强化学习将用户原始提示改写为更优提示，从而提升下游LLM的任务表现。

**💡 创新点**

创新点在于：①首次将GRPO强化学习与LLM‑as‑a‑Judge奖励相结合，用来评估改写提示与任务输出的语义质量；②引入提示质量奖励（Prompt Quality Reward）进一步引导模型生成更规范、可解释的提示；③提出多候选提示选择机制，并对其效果进行系统评估。

**🔧 技术方法**

技术包括：Group Relative Policy Optimization（GRPO）强化学习、LLM‑as‑a‑Judge（使用GPT‑4o‑mini与LLaMA‑3.1‑8B‑Instruct两种模型进行奖励打分）、提示质量奖励机制、温度采样与选择机制、JSON格式提示生成。

**📊 数据集**

使用的公开基准数据集包括：Natural Questions（NQ）、HotpotQA、CNN/DailyMail、SciTLDR以及算术推理数据集GSM8K；在实验中还对Phi‑4‑mini‑instruct与LLaMA‑3.2‑3B‑Instruct等模型进行了评估。

**📈 对比分析**

与基线提示、未训练的基模型重写以及TAPR+Selection进行对比；在NQ上TAPR提升至约59%准确率（基线56%），在GSM8K上提升至约83%准确率（基线82%）；在摘要任务中，Phi‑4‑mini‑TAPR在CNN/DailyMail上获得约3.78/5的LLM‑judge分数（基线3.37/5）。总体而言，TAPR在大多数任务上均超过基线，且Phi‑4‑mini‑TAPR的效果最为突出。

**⚠️ 局限性**

局限性包括：①在不同任务和模型上的收益不稳定，某些任务甚至无提升；②RL训练对超参数极度敏感，导致训练不稳定；③LLM‑as‑a‑Judge评估存在偏差（如对答案顺序的偏好）且计算成本高；④多候选提示选择并未始终带来收益；⑤整体方法增加了训练与推理成本，尚未充分证明其成本效益。

---

## 103. Hierarchical Copula-Gumbel-Top-\texorpdfstring{$K$}{K} Routing: Two-Sided Dependence Control for Frozen Mixture-of-Experts at Fixed Per-Token Routing Laws

**arXiv ID:** 2607.28670 | [PDF](https://arxiv.org/pdf/2607.28670v1)

**作者:** Richard Yi Da Xu `[一作]` (Hong Kong Baptist University), Richard Yi Da Xu `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5073709711)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造了一种分层Copula-基于Gumbel-Top-K的路由器，在保持每个token的单独路由分布不变的前提下，通过控制同组内部正相关和跨组负相关来调节多token间的路由协同与负载分散。

**💡 创新点**

创新点在于：① 在保持个体路由律完整性的同时，首次通过分层Copula实现可调节的正/负相关，形成可在同组提升专家集合一致性、跨组降低负载方差的双向依赖机制；② 证明该机制对单token的路由律、期望负载无影响；③ 通过小型控制器与score‑function估计实现只训练控制器而冻结主模型的路由适配。

**🔧 技术方法**

技术包括：Gumbel‑Top‑K采样、Gaussian Copula生成共享噪声、交叉组对抗式（antithetic）噪声构造、分层结构的概率推导与证明、score‑function梯度估计用于控制器训练。

**📊 数据集**

数据集为TinyStories（约1千万token）以及synthetic 4‑token logit测试，用于验证路由律不变性与负载方差理论。

**📈 对比分析**

比较方法：在冻结模型上使用固定Copula、学习的标量控制器以及Router‑LoRA参考；指标包括交叉熵、Jaccard相似度、负载CV。结果显示：Copula在保持期望负载不变的同时显著提高同窗口专家重叠，交叉熵变化极小；学习控制器在实际任务上未能显著提升性能。

**⚠️ 局限性**

限制：仅适用于随机Gumbel‑Top‑K路由器，无法保证多层模型整体不变；对负载方差的影响未量化；高效性取决于额外噪声生成与同步开销；实验规模有限，未能展示在真实下游任务中的显著收益。

---

## 104. Topology-Aware Data Movement for Disaggregated GPU Inference

**arXiv ID:** 2607.28633 | [PDF](https://arxiv.org/pdf/2607.28633v1)

**作者:** Sanjeev Rao Ganjihal `[一作]` `[通讯]`, Sanjeev Rao Ganjihal

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对分离式GPU推理的拓扑感知KV缓存传输框架，解决预填充与解码之间高带宽数据搬迁瓶颈。

**💡 创新点**

创新点在于：1) 动态发现GPU互连层级并按物理关系选择最高带宽传输；2) 针对Mixture‑of‑Experts模型的域感知专家调度与KV缓存共优化；3) 采用CXL 3.0内存扩展作为低延迟溢出层。

**🔧 技术方法**

使用的技术包括：NVLink/PCIe/RDMA/TCP多路传输接口；硬件探测脚本（nvidia‑smi、lspci、IB设备枚举）；Kubernetes控制器；CXL 3.0规格建模；EMA、阈值检测实现自适应解码池；五种传输模式的速率模型。

**📊 数据集**

主要验证数据集为Llama‑3‑70B（GQA，2.6 GB KV缓存）和DeepSeek‑V3（MLA，250 MB KV缓存）等大型LLM；采用这些模型的KV缓存尺寸和请求率进行分析。

**📈 对比分析**

与统一RDMA方案对比，拓扑感知选择可实现3到18倍的传输延迟降低，最大提升在NVLink域内可达18×；通过分析模型预测的传输时延和吞吐量。

**⚠️ 局限性**

局限性：缺乏多节点GPU集群（NVLink、InfiniBand、CXL 3.0）硬件，导致无法进行端到端实验验证；CXL层目前仅为性能模型，未实现实际驱动；管道化层传输未实现并发执行；系统部署受限于云供应商对NVLink拓扑的可见性。

---

## 105. SciToolAgent-Evo: An Ontology-Aware Self-Evolving Agent for Open-World Scientific Tool Acquisition

**arXiv ID:** 2607.28692 | [PDF](https://arxiv.org/pdf/2607.28692v1)

**作者:** Yuqi Tang `[一作]` (Zhejiang University), Huajun Chen `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向开放世界科学工具获取的本体感知自我进化智能体，能够在工具空间动态扩展时自动识别缺失能力、主动请求并完成工具本体化。

**💡 创新点**

创新点在于：①利用对比轨迹挖掘可迁移的工具使用经验和技能；②在推理阶段引入 LinUCB 决策门与主动工具请求，动态平衡已知工具利用与未知工具探索；③在线完成未知工具本体化并迁移至已知工具图，形成闭环自我进化。

**🔧 技术方法**

主要技术包括：对比轨迹生成与诊断、技能与经验的非参数抽取、基于本体的工具检索与重排序、LinUCB bandit 门、主动工具请求、知识更新与工具迁移；使用 Qwen3‑Embedding‑0.6B/1.7B 进行检索与重排序。

**📊 数据集**

数据集：①OpenSciToolBench（900题，4难度等级，涵盖生物/化学/材料三大领域）②SciToolEval（531题，单/多工具任务）。

**📈 对比分析**

对比了 Direct、CoT、RAG、ReAct、Reflexion、SciToolAgent 等基线，在 GPT‑5.4‑mini 与 Gemini‑3‑flash 两大 LLM 后端上均实现了显著提升；在 OpenSciToolBench 上整体答案准确率提升约 12–17%，在 SciToolEval 上提升 6–9%，尤其在 Level‑3 以上多工具任务上优势明显。

**⚠️ 局限性**

局限性包括：①对 LLM 背景模型高度依赖，模型能力受限；②需要手工划分已知/未知工具比例，实际应用中可能难以确定；③在极大规模工具库或更复杂的科研情境下，探索成本与本体更新速度仍是挑战；④目前评测仅基于人工构造的基准，缺乏真实科研工作流验证。

---

## 106. Technological Advances in Detecting and Managing Cognitive Impairment in Older Adults: Trends, Challenges, and Future Directions

**arXiv ID:** 2607.28687 | [PDF](https://arxiv.org/pdf/2607.28687v1)

**作者:** Mohammad Asif `[一作]` (Indian Institute of Technology Bombay), Anurag Rajkumar Bombarde `[通讯]` (T-Systems ICT India Pvt Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对2020‑2026年关于老年人认知衰退的技术进展进行系统性综述，涵盖EEG、MRI/PET、血液生物标志物、数字标志物与人工智能（AI）融合，提出统一分类、严格验证视角和早期检测框架，并整理对比表格。

**💡 创新点**

创新点包括：①整合跨学科技术与方法的统一分类体系；②在综述中持续强调受试者与站点独立验证的重要性，警示单一数据集高准确率的风险；③提出分层多模态筛查路径，将检测与干预紧密衔接；④将风险与保护因素与技术指标关联，形成可操作的临床决策工具。

**🔧 技术方法**

采用的技术主要有：深度卷积/循环网络（CNN、LSTM、BiLSTM、Transformer）、自监督EEG基础模型、图神经网络、多模态融合（EEG+MRI+血液+行为），以及基于AI的解释性与隐私保护（联邦学习、差分隐私）。

**📊 数据集**

使用的数据集包括：ADNI、OASIS、NACC（MRI/PET）、Temple University EEG Corpus、Korean CAUEEG、DementiaBank、ADReSS/ADReSSo（语音）、多中心血液标志物研究、可穿戴设备采集的行为数据（Empatica、Oura、Apple Watch）等。

**📈 对比分析**

方法比较显示：EEG深度模型单站点准确率可达96%+，MRI 3D‑CNN接近99.9%，但在受试者/站点独立验证时显著下降；多模态融合（EEG+MRI+血液+行为）提升灵敏度/特异性但增加复杂度；血液p‑tau217/Aβ42比值在单站点上AUC≈0.95；总体而言，性能高度依赖数据规模与验证严谨度，最佳方案需在大规模多站点外部验证后才能实现临床可靠性。

**⚠️ 局限性**

局限性包括：①多数高准确率来自小规模单站点或内部交叉验证，缺乏受试者/站点独立验证；②技术标准化与协议不统一，导致跨研究可比性差；③解释性、数据隐私与公平性问题尚未充分解决；④临床实用性评估不足，特别是对不同人群、语言、文化的适用性；⑤部分技术仍处于实验或预印阶段，缺乏监管批准；⑥干预证据多为观察性或早期试验，尚未形成统一临床指南。

---

## 107. MMFGU: Multimodal Federated Graph Unlearning

**arXiv ID:** 2607.28708 | [PDF](https://arxiv.org/pdf/2607.28708v1)

**作者:** Haodong Lu `[一作]` (Beijing Institute of Technology), Rong-Hua Li `[通讯]` (Beijing Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了MMFGU框架，实现多模态联邦图模型的选择性遗忘，支持实体/关系删除、模态删除和配对删除三类请求，保证模型在删除后仍保持较高的下游任务性能。

**💡 创新点**

创新点包括：①将异构删除请求统一映射为目标载体，统一化处理；②目标特定表示解耦，使用对比损失与保持约束在同一模型上实现局部删除；③通过轻量化扰动探针暴露并修复残留信息；④基于原型的跨客户端筛选与精细聚合，阻止目标信息在全局模型中再次出现；⑤整体实现无需中心化数据即可完成多模态遗忘。

**🔧 技术方法**

核心技术包括多模态图神经网络（结构、文本、视觉编码器 + 融合 + 图传播）、目标表示解耦损失、保持约束（嵌入一致性、边界约束）、扰动探针与对齐损失、原型聚合与客户端筛选、选择性聚合以及联邦学习框架。

**📊 数据集**

使用公开的多模态图数据集：OpenMAG 和 MM-OpenFGL，包括 Movies、Grocery、Toys、Ele-Fashion、RedditS、Book-nc（节点分类）以及 QB、TN、KU、Bili-Food、Bili-Dance、Bili-Movie（关系预测）等六十多张图。

**📈 对比分析**

与全量重训、FedEraser、FedKD、FedOSD、FUSED、ReGEnUnlearn 等 12 种基线在 12 个数据集、2 个任务、3 类删除请求上对比。MMFGU 在保持下游任务准确率/召回率方面常位列首位，未学习残差 UR 与全量重训相当，且在实际运行中实现 41.5× 的速度提升和显著的通信成本降低。

**⚠️ 局限性**

局限性：对极端删除比例或稀疏模态可能仍有残留目标影响；原型筛选依赖于聚合统计，可能忽略少数关键客户端；未提供形式化的遗忘安全证明；适配更多模态（如音频、时间序列）和更大规模联邦网络仍需进一步验证。

---

## 108. How Hard Does It Think? Analyzing Step-Aware Reasoning Energy in LLM Chain-of-Thought Trajectories

**arXiv ID:** 2607.28674 | [PDF](https://arxiv.org/pdf/2607.28674v1)

**作者:** Hui Wei `[一作]` (University Of California Merced), Julian McAuley `[通讯]` (University Of California San Diego)

**通讯引用:** 27160 | [OpenAlex ID](https://openalex.org/A5021827617)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Step-Aware Reasoning Energy (SARE) 的几何框架，用于量化链式思维（CoT）中每一步的计算努力，并通过分析能量分布揭示错误路径的特征。

**💡 创新点**

创新点在于：① 使用 Centered Kernel Alignment (CKA) 在相邻 transformer 层的 token 关系 Gram 矩阵之间衡量几何重组，避免了 eigenvector 对齐或簇匹配的问题；② 将步骤能量与隐语义状态转移相结合，形成多维解释；③ 通过能量特征在离线场景中对错误路径进行预测，优于传统基于输出的置信度方法。

**🔧 技术方法**

核心技术包括：Transformer 隐藏层取样、Gram 矩阵构造、CKA 计算、隐语义状态聚类（K‑means）、Markov 过程建模、统计特征提取（均值、方差、波动等）以及逻辑回归错误检测。

**📊 数据集**

使用六个公开推理基准（GSM8K、MATH、CSQA、StrategyQA、HotpotQA、MuSiQue）和三种 3–4B 参数规模的开源 LLM（LLaMA‑3.2‑3B、Phi‑4‑mini、Gemma‑3‑4B）。

**📈 对比分析**

与多种基线（Token Count、Mean Log‑Probability、Negative Entropy、Negative Perplexity、Self‑Certainty）比较，SARE 在大多数数据集和模型上实现了与或优于基线的 F1 分数；在 StrategyQA 等二分类任务中，传统概率基线失效时 SARE 仍保持 0.46–0.58 的 F1，显示出明显的优势。

**⚠️ 局限性**

局限性包括：仅在 3–4B 规模模型上验证，尚未确认对更大模型的可推广性；计算成本高，需要每层隐藏状态提取，适合离线分析；未来需要探索在线或低成本近似实现。

---

## 109. Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding

**arXiv ID:** 2607.28659 | [PDF](https://arxiv.org/pdf/2607.28659v1)

**作者:** Yuxuan Hu `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6976 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了GenCDSR，一个面向跨域序列推荐的生成式框架，结合跨域混合分层Token化和串并解码实现高效精确的下一物品预测。

**💡 创新点**

创新点在于使用共享‑特定双塔RQ‑VAE实现跨域共享与域特定特征的联合分层Token化，并设计基于两阶段SID结构的串并解码策略，在保持准确度的同时显著降低推理延迟。

**🔧 技术方法**

采用残差量化变分自编码器（RQ‑VAE）、Gumbel‑Softmax路由、多层码表、LLM（T5、Qwen3）、Trie约束的串并解码以及多头注意力生成模型。

**📊 数据集**

在三大公开跨域数据集上验证：Clothing–Sports、Electronics–Phone、Book–Movie。

**📈 对比分析**

与单域与跨域序列推荐基线（如GRU4Rec、BERT4Rec、SASRec、TIGER、C2DSR、TriCDR、LLM4CDSR、GenCDR）进行对比，平均提升约1.5%准确率，推理延迟下降85.1%，在多数指标上均优于现有方法。

**⚠️ 局限性**

局限性包括对极端域异质性和极少样本场景的鲁棒性未知，模型训练复杂度高，且对代码库的动态更新未作深入研究。

---

## 110. Human-LLM Collaborative Inductive Coding for Conceptualizing K-12 Educator AI Use

**arXiv ID:** 2607.28889 | [PDF](https://arxiv.org/pdf/2607.28889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 111. Hidden Errors in Big Data: The Case of Property Records

**arXiv ID:** 2607.28827 | [PDF](https://arxiv.org/pdf/2607.28827v1)

**作者:** Evelyn Smith `[一作]` (American Bar Foundation), Daniel E. Ho `[通讯]` (Stanford University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对Cotality和ATTOM两大房地产数据经纪人提供的物业交易数据进行审计，发现覆盖和填补错误导致经济不平等指标偏差。

**💡 创新点**

首次系统比较两家经纪人数据的误差来源与程度，并评估误差对财产税回归性估计的影响。

**🔧 技术方法**

采用匹配、错误分类、覆盖率分析以及回归性指标（PRD、对数系数、Suits指数）等技术。

**📊 数据集**

使用Cook County（伊利诺伊州）2018-2021年单户住宅销售的地方法规数据、Cotality和ATTOM数据，以及纽约市和费城的数据做稳健性检验。

**📈 对比分析**

通过对比经纪人数据与县级真值的匹配率、误差比例和回归性指标差异，发现经纪人错误在1-2%交易中出现，覆盖误差达12-15%，误差对税收回归性估计产生显著偏差。

**⚠️ 局限性**

局限于仅以Cook County为主，部分县级数据本身可能含有错误，且未获得经纪人内部处理文档，导致误差原因推测不确定。

---

## 112. Benchmarks Are Not Monolithic: Sample-Level Auditing and Orchestration for LLM Evaluation

**arXiv ID:** 2607.28801 | [PDF](https://arxiv.org/pdf/2607.28801v1)

**作者:** Philipp D. Siedler `[一作]` (Aleph Alpha Research), Jordan Sassoon `[通讯]` (Aleph Alpha Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个样本级元评估框架，对五个主流基准数据集进行细粒度标注，生成结构化指标并基于此构造可定制的子基准进行评估。

**💡 创新点**

创新点在于将基准视为多维对象，构建包含认知需求、语言质量、任务属性、上下文和伦理等五个维度的指标体系，并通过LLM-as-a-Judge自动化生成标注；进一步利用这些标注实现跨基准的动态合成子基准，揭示模型在不同能力维度上的差异。

**🔧 技术方法**

技术包括：1) 设计层级化指标目录并定义评分尺度；2) 编写统一的LLM-as-a-Judge prompt；3) 使用 GPT‑5、DeepSeek‑V3.1 等大型模型对每个样本进行标注；4) 统计分析、PCA、相关性检验和人机一致性评估；5) 基于标注构建基准子集并在 Llama‑3.2‑1B、SmolLM‑1.7B‑Instruct 上进行性能评测。

**📊 数据集**

使用的数据集包括 MMLU、ARC、WinoGrande、HellaSwag、TruthfulQA 的公开测试集，覆盖多学科知识、科学推理、核心ference、叙事推理和事实真实性等多种任务。

**📈 对比分析**

比较方法：在完整基准和各个根据单一或多维指标筛选的子基准上评估 LLM，计算平均准确率；结果显示，模型在高认知深度、语言难度或伦理敏感子基准上的表现显著低于整体指标，揭示单一分数隐藏的能力差异。

**⚠️ 局限性**

局限性：1) 依赖高性能评估模型可能导致自我强化偏差；2) 高级指标样本稀缺，子基准统计功效有限；3) 一些指标主观性强，跨文化一致性不高；4) 标注为静态快照，随知识更新需定期重新审计。

---

## 113. Guided Exploration of Iterative Schedule Modifications: A Design Study on Railway Traction Unit Scheduling

**arXiv ID:** 2607.28694 | [PDF](https://arxiv.org/pdf/2607.28694v1)

**作者:** Andreas Zajic `[一作]` (VRVis Research), Krešimir Matković `[通讯]` (VRVis Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文开发了一套交互式可视分析系统，用于铁路牵引单元巡回计划的探索与精细化，通过模拟评估计划并支持交叉操作来改进调度；

**💡 创新点**

创新点主要包括：①三层递进式引导机制——从全局聚合候选、局部排名到详细属性展示；②多视图协同系统与演化历史追踪；③可视化、修改与模拟的紧密耦合，使专家在单一界面内完成计划改动与即时评估；

**🔧 技术方法**

采用了可视化技术（表格计划视图、KPI柱状图、概览视图）、颜色编码与字体大小的多维编码、交叉操作交互、代理式随机模拟（基于历史延误的蒙特卡罗模拟）以及成就树（provenance）结构；

**📊 数据集**

使用了奥地利铁路运营商提供的真实调度数据，15 台牵引单元的七天参考周期（2025‑12‑15~21）以及过去六个月的延误统计，用于注入随机延误并进行模拟；

**📈 对比分析**

通过三条案例（延误最小化、并轨连续性优化、维护窗口平衡）演示系统效果；结果表明在局部改动下能显著降低平均延误并平衡维护窗口，未提供定量性能对比，但用户体验与时间节省得到质性验证；

**⚠️ 局限性**

局限性包括：仅在两位专家和单一车队上进行演示，缺乏正式用户研究；引导机制仅为一步前瞻，可能导致全局最优被忽略；规模受限于约100台车队，视图密度与计算开销仍需进一步优化。

---

## 114. Succinct and Fast Tiny Pointer Hash Tables

**arXiv ID:** 2607.28892 | [PDF](https://arxiv.org/pdf/2607.28892v1)

**作者:** Xilin Tang `[一作]` (Cornell University), Alex Conway `[通讯]` (Cornell Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出 Tiny Pointer Hash Tables (TPHT)，包含面向空间压缩的 Chained-TPHT 和面向低延迟的 Flattened-TPHT 两种实现。

**💡 创新点**

创新点：① 用 dereference table 将指针压缩为 8 位 tiny pointer；② 采用一轮 Feistel 置换实现键的 quotienting，既能去掉高位信息又能恢复完整键；③ 将链式结构与 tiny pointer 结合，获得接近信息理论下限的空间效率；④ 在 Flattened-TPHT 中将链式访问改为一次 cache 行访问并利用 SIMD 进行指纹比对，进一步降低延迟。

**🔧 技术方法**

主要技术：tiny pointer / dereference table（双路选择、位域编码）、one‑round Feistel permutation、quotienting、cache‑friendly home block layout、SIMD 指纹匹配、分布式乐观并发控制（seqlocks）以及协作式动态扩容。

**📊 数据集**

使用 YCSB 真实工作负载（A/B/C 阶段及其负向版本和删除）以及均匀分布的微基准数据集。

**📈 对比分析**

与 8‑10 种主流哈希表（cuckoo、Swiss、F14、linear probing、TBB 等）以及 4 种紧凑哈希表（Compact Bucketing、Cleary、Layered 等）进行对比。实验结果表明：Chained-TPHT 在空间效率上可达 90%+（接近信息理论下限），并保持与最快基线相当甚至更高的吞吐；Flattened-TPHT 在空间效率上达到 70%+ 的同时，在吞吐量上比基线高 30%–60%，并能把常见情况压缩到单次 cache 行访问。

**⚠️ 局限性**

局限性：① 仅支持 64‑bit 键值对（但可通过指针扩展支持变长键值）；② tiny pointer 的 ID 需要在分配时计算，增加实现复杂度；③ 在极高负载或极大数据规模下，dereference table 的失配概率虽然很低但仍存在；④ 需要自定义哈希函数和 Feistel 置换，对实现者有一定门槛。

---

## 115. NeuroSynth: A Biologically Inspired Continual Reinforcement Learning Architecture for Mitigating Catastrophic Forgetting

**arXiv ID:** 2607.28663 | [PDF](https://arxiv.org/pdf/2607.28663v1)

**作者:** Yash Kini `[一作]` `[通讯]` (James Madison High School), Yash Kini (James Madison High School)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 NeuroSynth 架构，结合双路径记忆整合、重放和知识蒸馏，实现了在连续学习环境下的强化学习

**💡 创新点**

创新点在于将神经科学的海马-皮层补偿学习机制转化为可训练的双路径网络，并结合重放与蒸馏，缓解灾难性遗忘

**🔧 技术方法**

采用深度强化学习（PPO、DQN）、参数正则化（EWC）、重放缓冲、KL 散度蒸馏等技术

**📊 数据集**

使用自定义的 8×8 网格导航环境 NeuroMaze-CL（三阶段目标变换）

**📈 对比分析**

与 PPO 和 EWC 进行六个种子下的对比，NeuroSynth 在 Task A、B 的保留率显著高于 PPO，且在 Task C 的表现优于 EWC，但差异不显著

**⚠️ 局限性**

局限性包括仅在小型离散网格任务上验证、任务数量有限、未与更多连续控制或基于重放的对手比较、缺少模块消融实验

---

## 116. Distilling Knowledge from Large Language Models into Lightweight Reinforcement Learning Agents for Autonomous Cyber Operations

**arXiv ID:** 2607.28826 | [PDF](https://arxiv.org/pdf/2607.28826v1)

**作者:** Konur Tholl `[一作]` (Royal Military College Of Canada), Ranwa Al Mallah `[通讯]` (Polytechnique Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对大型语言模型（LLM）进行提示工程，提炼其网络防御策略并将其在线蒸馏到一个仅含 64,910 参数的轻量级强化学习（RL）代理，展示该代理在 CybORG CAGE Challenge 2 环境及多种网络拓扑中能匹敌甚至优于传统 RL 代理。

**💡 创新点**

创新点包括：1）将已预训练的 80 亿参数安全领域 LLM 的决策直接蒸馏到极小 RL 模型；2）使用在线蒸馏和动作掩码技术实现快速知识迁移；3）通过多场景评估验证蒸馏策略的可迁移性；4）系统性检验多种教师引导稳定化方法，发现其均无法持续超越 LLM 基准。

**🔧 技术方法**

采用技术包括：提示工程（链式思维结构化提示）、动作掩码（确保代理遵循 LLM 指令）、教师引导损失（对抗策略对齐）、基于 PPO 的 on‑policy 强化学习、以及多种教师影响衰减策略。

**📊 数据集**

数据集为 CybORG 的模拟网络环境，构建了 9 个不同规模（4–12 主机）的网络场景，用于评估蒸馏后代理的表现。

**📈 对比分析**

与基线 PPO 代理以及教师引导 RL 代理进行比较。结果显示：蒸馏后代理在 240 轮训练后即能达到与 LLM 相同的平均奖励，并在 50,000 轮 PPO 训练中保持更稳定的性能；在多网络场景中，其稳定性优于基线，误差区间更小。

**⚠️ 局限性**

局限性包括：1）CybORG 虚拟环境与真实企业环境差距较大；2）评价仅依赖单一奖励信号，可能未能完整反映防御效果；3）蒸馏依赖单一 LLM，若 LLM 存在幻觉或局限，可能被继承到学生模型。

---

## 117. Do Medical Foundation Models Generalize on the African Brain?

**arXiv ID:** 2607.28771 | [PDF](https://arxiv.org/pdf/2607.28771v1)

**作者:** Kaouther Mouheb `[一作]` (Erasmus Medical Center), Esther E. Bron `[通讯]` (Erasmus Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

评估医学基础模型在非洲脑MRI数据上的泛化能力，并将其与高资源人群进行比较，涵盖痴呆分类和脑肿瘤分割两类任务。

**💡 创新点**

首次系统性比较多种一般性与分割专用FMs在非洲与非非洲数据集中的表现，揭示样本规模而非地区偏差是主要影响因素。

**🔧 技术方法**

采用线性探测、LoRA参数高效微调、SAM零样本提示、UNETR解码器等技术，对预训练模型进行适配和评估。

**📊 数据集**

使用尼日利亚脑MRI数据集（痴呆vs健康）、OASIS-4、BraTS‑Africa 2023（肿瘤分割）以及Erasmus Glioma Dataset（高资源基准）。

**📈 对比分析**

通过5折Monte‑Carlo交叉验证，使用ROC‑AUC评估分类任务、Dice分数评估分割任务，结果显示FMs在分割任务上显著优于从零训练的U‑Net，分类任务增益有限。

**⚠️ 局限性**

受限于非洲数据集规模小、年龄分布偏差、缺乏完整元数据以及实验依赖高端GPU等因素，导致对泛化能力的结论需谨慎解释。

---

## 118. Domain-Adaptive Deep Joint Source-Channel Coding for Image Classification

**arXiv ID:** 2607.28907 | [PDF](https://arxiv.org/pdf/2607.28907v1)

**作者:** Yishen Li `[一作]` (Central South University), Hao Zhang `[通讯]` (Central South University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对任务导向深度源信道编码（Deep JSCC）的单源域自适应框架，能够在域间分布漂移的情况下通过伪标签的类级对抗对齐和伪标签监督对比学习来提升目标域的分类性能。

**💡 创新点**

创新点在于：①引入分类‑容量‑不变性（CCI）函数，系统性分析通道容量、域不变性与目标域分类误差的关系；②提出类级对抗对齐，精细化类间分布匹配；③结合伪标签对比学习，提升跨域类内聚合与类间分离；④采用两阶段训练策略提升伪标签质量。

**🔧 技术方法**

使用了深度神经网络进行语义提取、编码与解码；类级域判别器与梯度反转层实现对抗对齐；伪标签监督对比损失（InfoNCE）；多任务损失组合（分类、对抗、对比）；在AWGN和Rayleigh信道上进行训练与评估。

**📊 数据集**

在数字图像数据集SVHN（源）与MNIST（目标）以及多域视觉数据集PACS（源域为Photo或Art，目标域为其余三域）上进行实验。

**📈 对比分析**

与原始Deep JSCC、DANN‑JSCC、MDAN（基于StarGAN）和KJDM等基线进行比较；在AWGN和Rayleigh信道下，通过目标域分类准确率评估。实验表明，在10 dB CSNR下SVHN→MNIST可达98.15%准确率，PACS各域对比也显著优于基线，且不需要推理时额外网络，推理速度与原Deep JSCC相当。

**⚠️ 局限性**

局限性包括：①伪标签的可靠性受域差距与信道噪声影响，需两阶段训练；②类级对抗对齐与对比学习对超参数（λ、τ）敏感；③目前仅验证了单源场景，未覆盖多源自适应；④在极端低CSNR或强噪声衰落下仍存在性能下降。

---

## 119. Partial Derandomization for Leakage-Resilient Shamir's Secret Sharing over Composite Order Fields

**arXiv ID:** 2607.28757 | [PDF](https://arxiv.org/pdf/2607.28757v1)

**作者:** S. Venkitesh `[一作]` `[通讯]` (Ben Gurion University of Negev), S. Venkitesh (Ben Gurion University of Negev)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本文提出一种利用 Möbius 变换 Φ 的迭代生成评估点序列，构造适用于合成阶域的泄露抵抗 Shamir 秘密分享方案。

**💡 创新点**

创新点在于把随机选择的 n 个评估点替换为单参数迭代生成，从而把所需随机位数从 nd log p 降至 d log p，同时在 n = O(d/logₚd) 的参数范围内实现完美的单块泄露安全性。

**🔧 技术方法**

核心技术包括：① 基于投影线性代数的完美二分法；② 利用迭代 Möbius 变换的极点互异性进行部分分式非退化性分析；③ 对多块泄露模式的扩展和多元化的矩阵秩判定。

**📊 数据集**

本文不涉及具体数据集，研究完全在理论分析与符号计算层面完成；评估基于代数结构的理论上界与坏集上限。

**📈 对比分析**

与之前的随机构造相比，本文在单块泄露下统计距离达到零（完全安全），但在参与方数量上仅支持 n = O(d/logₚd)，而随机构造可达到 n = O(dk/logₚd)。

**⚠️ 局限性**

局限性包括：① 对多块泄露的全局可验证性仅在 M = O(d/logₚd) 内；② 对于更大 k 的参与方数量仍受单参数构造的约束；③ 坏集上限仍为指数级，尚未实现多项式级别的改进。

---

## 120. YazSes: An Offline, Privacy-First, Cross-Platform Hold-to-Talk Voice-Dictation System

**arXiv ID:** 2607.28878 | [PDF](https://arxiv.org/pdf/2607.28878v1)

**作者:** Mohsen Seyedkazemi Ardebili `[一作]` `[通讯]` (NovaFabric), Mohsen Seyedkazemi Ardebili (NovaFabric)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个开源的全离线语音转写与命令执行系统（yazses），该系统通过按键触发录音、使用更快的Whisper模型（faster-whisper，CPU int8）完成语音识别，并通过正则表达式与可选的小语言模型路由将语音命令映射为编辑器或终端操作，所有过程均在本地完成，无任何数据外泄。

**💡 创新点**

创新点主要体现在：
1) 统一跨平台的协议抽象（Linux、macOS、Windows）实现单一代码库；
2) 通过 push‑to‑talk 机制和零遥测保证隐私，将 “不离机” 作为系统 invariants；
3) 在本地实现可选的加密学习语料库与离线参数调优，避免模型权重上传；
4) 采用轻量级的命令语法引擎与可选小语言模型路由，兼顾命令准确率与实时性。

**🔧 技术方法**

主要技术栈包括：Python 3.11、faster-whisper（基于 CTranslate2 的 int8 Whisper），VAD（RMS 门限或 Silero），音频缓冲与预处理，正则表达式命令分类，轻量级小语言模型（可选），JSON‑RPC 控制平面，IPC（Unix socket / named pipe），以及各平台的输入注入后端（ydotool/xdotool、Quartz SendInput 等）。

**📊 数据集**

实验使用 LibriSpeech test‑clean 子集（约 55 分钟），随机抽取的 speaker‑stratified 片段，采用 Whisper 的标准文本规范化和 JiWER 计算 WER。

**📈 对比分析**

与 Whisper 基准的比较：
- tiny.en：WER 4.82 %，RTF 0.15；
- base.en：WER 4.07 %，RTF 0.28；
- small.en：WER 2.59 %，RTF 0.52。所有模型在 CPU 上均可实现 RTF < 1，即实时以上。
- 非解码阶段的总时延 < 1 ms，命令语法识别平均 0.07–0.09 ms，几乎不影响总体延迟。
- 在文本基准上，命令识别准确率达 99‑100 %，普通文本误判率低于 0.1 %。

**⚠️ 局限性**

限制与未来工作：
1) 仅在 Linux 上完成了端到端评估，macOS/Windows 的实现仅通过单元测试验证；
2) 默认仅支持英文 Whisper 模型，其他语言需自行部署；
3) 目前未包含移动或 Web 版本；
4) 在噪声或口音较重的麦克风环境下的性能尚未评估；
5) 个性化学习环节未进行长期用户实验；
6) 缺少人机交互的可用性研究与实际用户体验评估。

---

## 121. ZeroR@CHiPSAL 2026: Two-Stage Vision-Language Adaptation with Contrastive Learning for Nepali Meme Classification

**arXiv ID:** 2607.28637 | [PDF](https://arxiv.org/pdf/2607.28637v1)

**作者:** Nitiz Khanal `[一作]` `[通讯]` (Tribhuvan University), Nitiz Khanal (Tribhuvan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在尼泊尔 meme 的仇恨言论检测与情感分析任务中，本文提出了基于 Qwen3‑VL‑8B‑Instruct 的两阶段 end‑to‑end 多模态学习框架。先使用 LoRA 微调模型进行生成式标签预测，再在对比学习阶段通过监督式对比损失和线性分类器提升表征质量，最终将两阶段输出按权重融合得到最终预测。

**💡 创新点**

创新点包括：①不使用 OCR + 文字/图像分离管线，直接让 VLM 在图像中读取 Devanagari 文本；②两阶段训练策略（生成式 + 对比学习）结合语言模型和度量学习的优势；③针对低资源尼泊尔语的 LoRA 超参数设计与任务特定 prompt ；④在数据不平衡下结合过采样、图像增强、Focal loss 与阈值调优，实现宏 F1 的显著提升。

**🔧 技术方法**

核心技术包括 Qwen3‑VL‑8B‑Instruct（多模态指令调优模型）、LoRA（低秩适配器）进行参数高效微调、监督式对比学习（InfoNCE）用于提升表征分离度、Focal loss 对三分类任务的难易样本加权、数据增强与过采样对抗类别不平衡、阈值调优与加权融合策略。

**📊 数据集**

使用了 CHiPSAL 2026 共享任务提供的尼泊尔 meme 数据集：约 1,068 条样本用于仇恨言论检测（二分类）和相似规模的样本用于情感分析（三分类），所有图像均包含 Devanagari 与可能的英尼混合文本。

**📈 对比分析**

通过官方 leaderboard 与 ablation 结果比较：仇恨言论检测宏 F1 0.797（第二名），情感分析宏 F1 0.518（第四名）。单阶段 LoRA 微调仅达 0.74 / 0.49；加入过采样、增强后提升至 0.76 / 0.51；再加对比学习达到 0.79 / 0.52；阈值调优后最终 0.80 / 0.518，表明两阶段融合和阈值优化对性能提升贡献显著。

**⚠️ 局限性**

主要限制包括：①样本量约 1,000 例，易出现过拟合且泛化能力不确定；②模型对尼泊尔特定文化、历史语境的理解有限，导致某些文化敏感 meme 误判；③对代码混合内容的处理仍不完美；④未对多种随机种子、零样本或 OCR+文本管线等基线进行全面评估；⑤prompt 设计未进行系统性搜索，可能还有提升空间。

---

## 122. LayoutBench: Performance Benchmarking of Cloud Storage Layouts for Multimedia Data

**arXiv ID:** 2607.28880 | [PDF](https://arxiv.org/pdf/2607.28880v1)

**作者:** Debopam Sanyal `[一作]` (Georgia Tech), Joshua Kimball `[通讯]` (Dolby Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LayoutBench基准，用来系统评估云对象存储中多媒体（以ImageNet图像为例）数据检索时不同存储布局（单对象、tar、Parquet）的性能与成本。

**💡 创新点**

首次将三种典型布局在真实ML检索工作负载下进行对比，并给出可插拔的基准框架；发现连接重用与行组读取在不同场景下决定性能与费用的主导因素。

**🔧 技术方法**

使用AWS S3作为存储后端，EC2实例作为客户端，DuckDB执行Parquet查询，tar文件支持Byte‑Range GET；通过测量检索时间、传输量和费用来评估。

**📊 数据集**

采用ImageNet‑1K训练集，按1%、10%和100%随机抽样构成mini、medium、full三种规模的实验数据集。

**📈 对比分析**

设计了11个基于类、尺寸、宽高、正则、交叉列等谓词的检索查询，在六种不同网络/内存的EC2实例上运行；结果显示：L2在大多数查询中具有最优的延迟‑成本平衡，L1在极小查询更快，L3在极大查询最快但传输量与成本远高于其他两种布局。

**⚠️ 局限性**

实验仅覆盖图像数据、只读检索、单云单引擎；未评估写/摄取路径、视频/音频等大文件，以及跨云或不同查询引擎的表现。

---

## 123. TORUS: A Test of Rendering-Understanding Self-Coherence for Unified Audio Models

**arXiv ID:** 2607.28896 | [PDF](https://arxiv.org/pdf/2607.28896v1)

**作者:** Aryan Vijay Bhosale `[一作]` (Centific Global Solutions Inc.), Dinesh Manocha `[通讯]` (University of Maryland)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并发布了第一个面向音频统一模型的自一致性基准 TRU-Coherence，包含 48 个三阶段自一致性测试（共 432 道六选多选题），评估模型在音频生成、编辑和理解三种任务上的自我一致性。

**💡 创新点**

创新点在于：① 构建了泄漏检查、人工验证的音频自一致性测试流程；② 提出了三阶段自一致性评估框架（生成→编辑→反事实编辑），① 通过生成-编辑-评估闭环；③ 采用“Coherence”指标量化模型的自一致性，首次系统性评估统一音频模型的两头是否达成共识。

**🔧 技术方法**

技术实现包括：使用前沿大型语言模型（LLM）生成测试提示和问题；配备“Muted Audio Solver”与“Ideal Generation Solver”双门检验防止文本先验泄漏；利用现有最先进的专用生成、编辑、理解模型构建级联基准；评估指标涵盖 WER、FAD、FD、KL、CM 等。

**📊 数据集**

数据来源为自行构造的 48 条测试样例，覆盖语音、音效、音乐三种音频类型和 5 大任务族（物理因果、源组合、时序推理等），共 432 道人工核对、泄漏检查后生成的多选题；无直接使用公开数据集，完全基于人工撰写与人工验证。

**📈 对比分析**

对比方法：将 5 种公开统一音频模型与由最先进专用模型组成的级联基准在 TRU-Coherence 上进行评测。结果显示级联基准 63.2% 的自一致性分数远高于最佳统一模型 50.5%，且所有统一模型在编辑阶段性能显著下降，显示出编辑任务是当前最薄弱环节。

**⚠️ 局限性**

局限性：① 仅有少数统一模型支持原生音频编辑，需使用“自标注编辑”链，可能混淆字幕与编辑质量；② 基准仅覆盖英文；③ 目前自一致性评估仍未涵盖所有音频属性和更细粒度的语义推理；④ 评估仅关注生成与理解的闭环，未涉及更广泛的多模态交互。

---

## 124. Agreement Is Not Quality: Blind Expert Verification of Human and LLM Qualitative Coding When Human Consensus Is Not Ground Truth

**arXiv ID:** 2607.28890 | [PDF](https://arxiv.org/pdf/2607.28890v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 125. Hypergradient-based Bilevel Reinforcement Learning with Improved Sample Complexity

**arXiv ID:** 2607.28849 | [PDF](https://arxiv.org/pdf/2607.28849v1)

**作者:** Naman Saxena `[一作]` (Purdue University), Vaneet Aggarwal `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种可扩展的双层强化学习算法 AHO，利用 Boltzmann 策略的最优性构造无 Hessian 的超梯度

**💡 创新点**

在非可实现策略类下通过梯度移位价值函数构造近似超梯度，去除了对外层目标的 PL 条件，改进了样本复杂度至 Õ(ε⁻²)

**🔧 技术方法**

Boltzmann 策略最优性、梯度移位价值函数、无 Hessian 超梯度、基于策略梯度与 Q‑学习估计

**📊 数据集**

论文未使用具体实验数据集，主要给出理论证明

**📈 对比分析**

与先前基于惩罚或 Hessian 的双层 RL 方法比较，迭代复杂度保持 O(ε⁻¹)，样本复杂度从 O(ε⁻³) 提升至 O(ε⁻²)

**⚠️ 局限性**

近似超梯度引入了 ϵ_kl、ϵ_fd 等误差，需足够丰富的策略类；理论依赖多项光滑与 PL 条件，实验验证缺失

---

## 126. Learning Optimal Dynamic Matching via Graph Neural Networks

**arXiv ID:** 2607.28925 | [PDF](https://arxiv.org/pdf/2607.28925v1)

**作者:** Genta Okada `[一作]` (University of Tokyo), Akira Matsushita `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于图神经网络的强化学习框架，用于在动态匹配市场上学习残差图的后决策价值，从而指导匹配决策。

**💡 创新点**

创新点包括：1）通过事件时间归约，将持续时间决策化简为在每个外部事件后立即匹配；2）将最优决策转化为残差图的后决策价值函数，显著减少状态动作空间；3）使用GNN近似该价值，并结合前向贪婪搜索解决组合匹配问题。

**🔧 技术方法**

采用图神经网络（GNN）对图结构进行编码，使用时序差分（TD）学习更新价值；结合经验回放、目标网络、ε-贪婪探索以及前向贪婪匹配策略。

**📊 数据集**

使用了两套仿真数据集：①二元类型基准（自定义模拟）②肾脏配对捐赠（KPD）模拟，后者基于临床与统计参数构建真实感的节点类型、边权与退出机制。

**📈 对比分析**

与 Immediate Random、Immediate Greedy、Immediate Threshold Greedy、Patient Greedy、Tabular Optimal（仅二元基准）等基准进行比较；在二元基准上接近最优，在KPD基准下在不同退出信息情景中均优于传统贪婪/患者策略，特别是在退出信息不完整时表现显著提升。

**⚠️ 局限性**

局限性：缺乏理论最优性证明；匹配选择仍依赖贪婪近似，可能导致次优匹配；实验仅在模拟环境中验证，真实世界数据的泛化能力和鲁棒性尚未充分评估。

---

## 127. OpenClaw and Ollama in Agentic AI: Toward Fully Autonomous and Scalable AI Agent Systems

**arXiv ID:** 2607.28629 | [PDF](https://arxiv.org/pdf/2607.28629v1)

**作者:** Konstantinos I. Roumeliotis `[一作]` (University of Peloponnese), Ranjan Sapkota `[通讯]` (Cornell University)

**通讯引用:** 1484 | [OpenAlex ID](https://openalex.org/A5025722346)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将本地LLM推理引擎 Ollama 与持续执行的 Agent 运行时 OpenClaw 结合，构建了完整的 Agentic AI 全栈架构，并在此框架下进行实验验证。

**💡 创新点**

创新点在于提出了基于层级的 Agentic AI 体系结构，将推理、调度、工具调用、记忆与治理等模块分离，并证明了各层对自主能力的独立贡献；同时首次提供了完整实验代码和数据以支持复现与基准对比。

**🔧 技术方法**

技术主要包括本地 LLM 推理（Ollama）、Agent 运行时框架（OpenClaw）、工具调用接口、结构化内存管理以及多轮推理循环。

**📊 数据集**

使用公开可获取的两种 4B 大模型（Qwen3.5:4b 与 Gemma4:e4b）作为推理引擎，并在 GitHub 仓库中发布了 15 个自定义任务集合，涵盖文件查询、工具使用与跨任务记忆等功能。

**📈 对比分析**

对比方法采用三种系统配置（C1：纯 Ollama；C2：OpenClaw 无持久化；C3：OpenClaw + 持久化），评估指标包括任务成功率、工具调用准确率、记忆回溯准确率、平均延迟和推理步骤；实验结果显示 C1 < C2 < C3，且在 C3 上两模型的任务成功率均达到约 97–98%，验证了层级架构带来的性能提升。

**⚠️ 局限性**

局限性包括：1）实验仅覆盖单机部署，未验证多 Agent 或分布式场景的可扩展性；2）工具调用与内存管理的安全性与治理机制尚需进一步完善；3）基准任务规模相对有限，无法覆盖更复杂的长周期决策与真实环境交互。

---

## 128. From C to Idiomatic Rust: A Ship-of-Theseus Agentic Translation

**arXiv ID:** 2607.28835 | [PDF](https://arxiv.org/pdf/2607.28835v1)

**作者:** Vasily A. Sartakov `[一作]` `[通讯]`, Vasily A. Sartakov

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种将 C 代码迁移到 Rust 的分阶段方法：先使用 c2rust 生成功能等价的非惯用 Rust 基线，然后借助 LLM 逐函数改写为惯用 Rust，整个过程通过编译和行为测试门控保持功能一致；以 iodine DNS 隧道（约12.5k SLOC）为案例，完成约10.3k SLOC 的 Rust 实现；

**💡 创新点**

创新点在于将迁移视为功能保留的增量重构序列，而非一次性翻译；引入显式惯用性判据、树状叶节点顺序、LLM 驱动的交互循环和运行时追踪对比；构建完整工具链实现自动化安全门控与验证；

**🔧 技术方法**

使用的技术包括 c2rust 源到源转译器、Qwen3.6‑27B 等大型语言模型、自动化编译/测试门（Gate A/B）、静态调用图分析、全局状态包装（atomic、Mutex）、运行时行为对比与回溯；

**📊 数据集**

实验数据集为 iodine DNS 隧道的完整 C 代码库（12.5k 行）；通过该项目验证方法在真实系统级软件上的可行性；

**📈 对比分析**

比较方法：跟踪每次提交的 SAFE% 与 RUST% 两项安全评分，并记录翻译工时与函数/行级产出；结果显示完成迁移耗时约 37 人工小时，平均每小时 4.5 个函数（约 278 行 Rust），比从零重写估算的 COCOMO II 人月要低得多；功能与性能保持一致（未出现功能缺失或显著性能下降）；

**⚠️ 局限性**

局限性包括：仍需人工干预选择函数、确定顺序、处理临时适配层与全局状态包装；对并发锁定的静态分析仍有限，可能出现死锁；部分 C 全局状态（如 extern static）无法直接转换；LLM 的输出需要人工审核；整体流程尚未实现完全自动化，难以直接推广至所有大规模 C 代码库；

---

## 129. A Matrix Factorization Approach in Turnstile Streaming

**arXiv ID:** 2607.28819 | [PDF](https://arxiv.org/pdf/2607.28819v1)

**作者:** Jan Bulanek `[一作]` (Google), Tamas Sarlos `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文提出了一种基于矩阵分解的通用框架，用来在严格流（turnstile）模型下处理 M‑点查询（M‑point query）问题，并给出了量化误差为 ±x₁ 的近似答案。

**💡 创新点**

创新点主要包括：
1) 将任意预先给定的矩阵 M 通过分解 M = AB 来实现低空间点查询，提出空间上依赖于 A 的 ℓ₂→∞ 范数与 B 的 ℓ₁→1 范数的上界；
2) 通过新的协方差上界（r⁻³ 量级）改进了 CountSketch 的误差分析；
3) 推导了量化（rank/quantile）问题的近似上界 O((log U)^{3/2}/ε) 和匹配的下界，证明 dyadic CountSketch 在该类分解方法中已接近最优；
4) 给出了一条全动态量化查询的空间下界 Ω((log n)·log(U/log n)/ε)，与仅插入流的情况形成对比。

**🔧 技术方法**

技术手段包括：
- CountSketch 的五行实现与中位数聚合；
- 伪随机生成器与有限独立哈希，满足对半空间交叉的伪随机性；
- 通过 Haar 基函数、倒数方差与马尔科夫不等式等工具，对协方差进行细化；
- 对 Q（下三角全 1 矩阵）做 dyadic 分解，得到 A、B 的稀疏性与范数；
- 结合信息论与错误更正码的通信复杂度降低技巧，构造全动态量化查询的下界。

**📊 数据集**

论文并未使用具体实验数据集，而是以理论分析和信息论证明为主，侧重于空间复杂度与误差界的严格证明。

**📈 对比分析**

与之前的 dyadic CountSketch、传统点查询方法相比，新方法在空间复杂度上实现了 O((log U)^{3/2}/ε) 的改进，并在分解框架内几乎达到理论下界；同时，在误差分析上使用协方差上界减少了对层级独立性的需求，简化了实现。性能上，该方法在理论上与最优匹配，实际实现可进一步利用单表 hash 提升效率。

**⚠️ 局限性**

局限性包括：
- 该框架仅适用于可分解为 AB 的矩阵，且空间上仍受 A 的 ℓ₂→∞ 与 B 的 ℓ₁→1 范数限制；
- 对于量化查询，虽然下界与上界相近，但仍存在 loglog U 的细微差距；
- 需要五行 CountSketch 及伪随机哈希，实际实现仍需关注随机数种子的生成与更新时间；
- 该方法不适用于不满足分解条件或高阶矩阵结构的更复杂查询。

---

## 130. Convex Approximation and the Hilbert Geometry

**arXiv ID:** 2607.28885 | [PDF](https://arxiv.org/pdf/2607.28885v1)

**作者:** Ahmed Abdelkader `[一作]` (Google), David M. Mount `[通讯]` (University of Maryland)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于Hilbert几何的近似多面体成员测试的新方法，利用Hilbert度量构建了高效的数据结构。

**💡 创新点**

创新点在于结合了Macbeath区域和Hilbert度量，提出了一种新的查询结构，通过椭球体覆盖来回答查询，并支持指针搜索。

**🔧 技术方法**

使用了Hilbert度量、Macbeath区域和Delone集等几何技术。

**📊 数据集**

使用了多面体的几何体作为数据集，具体数据集未详细说明。

**📈 对比分析**

与传统的空间划分技术相比，提出的方法在查询时间和存储复杂度上达到了最优，查询时间为O(log(1/ε))，存储为O(1/ε^(d-1)/2)。

**⚠️ 局限性**

限制在于该方法依赖于Hilbert度量的性质，可能在高维空间中表现不如预期，且构建时间复杂度可能较高。

---

## 131. DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning for LLMs

**arXiv ID:** 2607.28848 | [PDF](https://arxiv.org/pdf/2607.28848v1)

**作者:** Jiaxuan Chen `[一作]` (McGill University), Oana Balmau `[通讯]` (McGill University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个与宿主推理引擎无关的共服务框架，能够在满足LLM推理SLO的前提下，将空闲GPU推理计算资源用于LoRA微调。

**💡 创新点**

创新点包括：①基于推理prefill与LoRA微调前向相同的结构，采用SLO感知的调度器将微调任务“折叠”进推理批次；②提出CUDA‑Graph感知的延迟模型，在线校准并用于提前预测微调对推理延迟的影响；③将微调后向计算放在独立的GPU子进程中，按层级抢占和恢复，最大化利用空闲算力。

**🔧 技术方法**

核心技术：多LoRA批量化（multi‑LoRA batching）、CUDA‑Graph记录与重放、基于执行模式的延迟模型、基于SLO的动态调度、GPU进程间共享内存、后向子进程抢占控制。

**📊 数据集**

使用的数据集：Llama 3‑8B作为基础模型；Alpaca instruction‑tuning 数据集用于LoRA微调；Nutanix 20‑分钟生产推理工作负载用于评估和调度；Synthetic burst‑light 和 burst‑dense 负载用于对比实验。

**📈 对比分析**

与最先进的共服务系统LLMStation和传统的分离池（vLLM+torchtune）比较。结果表明，在Nutnix生产轨迹上，-vLLM实现的LoRA微调吞吐量为LLMStation的2.9×，且100%满足TTFT/TPOT SLO（LLMStation仅85%）。在Synthetic负载下，-vLLM在高峰时仍保持100% SLO，微调吞吐量分别比LLMStation高3.5×和2.6×，比分离池高21%。

**⚠️ 局限性**

限制：①需要宿主引擎支持multi‑LoRA批量化；②仅针对LoRA等参数高效微调方法，对全参数微调仍无支持；③后向子进程的抢占机制可能在极高推理负载下导致微调延迟；④在极低负载环境下仍可能产生微调与推理之间的频繁切换开销。

---

## 132. The Formalism Trap: Are LLM-as-a-Judge Evaluators Blinded by Consensus Mimicry under Social Load?

**arXiv ID:** 2607.28641 | [PDF](https://arxiv.org/pdf/2607.28641v1)

**作者:** Dahlia Shehata `[一作]` (University of Waterloo), Ming Li `[通讯]` (University of Waterloo)

**通讯引用:** 58662 | [OpenAlex ID](https://openalex.org/A5100355322)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并量化了LLM评判器在多智能体社交压力下被结构化语法“诱捕”的现象，提出了Agentic Formalism Trap与Evaluative Dissonance Index（D_E），并构建了一个包含22,500条多智能体推理轨迹的跨域数据集

**💡 创新点**

首次系统地定义并量化评判器对结构化语法的偏倚，并通过逻辑回归模型提取了531个语义特征簇作为评判器失效的触发器，提出了可零样本跨域迁移的Meta-评估器和“警戒过滤器”

**🔧 技术方法**

结构化语法抽取、确定性词汇基底化、逻辑回归（L1/L2正则化）、零样本Leave-One-Domain-Out交叉验证、统计显著性检验（Wald、Pearson相关）

**📊 数据集**

GAIA（通用问答）、SWE-bench（软件工程/代码执行）、Multi-Challenge（多轮对话）三大基准，各自生成7,500条轨迹，总计22,500条

**📈 对比分析**

在内部训练/测试拆分下，Meta-评估器的ROC‑AUC达0.8779；在零样本跨域迁移下平均ROC‑AUC为0.7482；通过阈值调整（t=0.98）实现0.91的高精度警戒过滤器，显示出良好可迁移性与高检测性能

**⚠️ 局限性**

仅适用于确定性任务；数据集使用人工设计的仿真群体，缺乏真实多智能体对话；模型参数受闭源模型更新影响；仅通过贪婪解码得到的轨迹，未探讨温度参数对语法诱骗的影响

---

## 133. SCMA: Structure-Conditioned and Metal-Aware Flow Matching for CT Metal Artifact Reduction

**arXiv ID:** 2607.28759 | [PDF](https://arxiv.org/pdf/2607.28759v1)

**作者:** Heran Wang `[一作]` (Capital Normal University), Jigang Duan `[通讯]` (Capital Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在CT金属伪影消除方面，提出了结构条件和金属感知的Flow Matching框架SCMA，以实现对金属引起的伪影的有效抑制和真实结构的保留。

**💡 创新点**

创新点在于将线性插值校正图像作为样本特定结构条件，使用金属掩模及其距离变换构造时变空间权重加权损失，并在推理过程中交替进行投影一致性校正，三者协同提升了生成模型的局部鲁棒性和物理一致性。

**🔧 技术方法**

采用了Flow Matching生成模型、条件U-Net+FFT Transformer网络、动态空间加权损失以及投影一致性校正（PCC）等技术。

**📊 数据集**

使用DeepLesion数据集合成的金属无损CT图像与100种金属掩模进行训练和测试，并在CTSpine1K的COLONOG子集真实金属CT图像上验证。

**📈 对比分析**

与FBP、NMAR、DICDNet、InDuDoNet+、CALIMAR、ADN、DuDoDp-MAR等方法在PSNR/SSIM、残差可视化和ROI结构相似度上比较，SCMA在小、中、大金属植入物上均实现了最高PSNR和SSIM，平均提升约2.5dB，且局部结构保持最优。

**⚠️ 局限性**

局限在于对金属掩模的依赖，掩模误差会影响加权与投影校正；投影一致性校正增加计算成本；目前仅在二维切片、平面投影下验证，需扩展到三维螺旋或锥形CT及多中心临床数据。

---

## 134. Blockchain Transaction Simulation Phishing

**arXiv ID:** 2607.28747 | [PDF](https://arxiv.org/pdf/2607.28747v1)

**作者:** Xiaocan Wang `[一作]` (Stevens Institute of Technology), Kai Li `[通讯]` (Stevens Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了交易模拟钓鱼攻击并系统地分析了其特征。

**💡 创新点**

创新点在于提出六类基于动态区块链属性的钓鱼合约分类，并设计了基于字节码的检测系统。

**🔧 技术方法**

采用符号执行、TAC转换、静态/动态分析以及运行时验证等技术进行检测。

**📊 数据集**

使用以太坊、币安智能链、Avalanche和Polygon的历史合约字节码数据，构建了4,224个钓鱼合约及5,700多个真实受害者。

**📈 对比分析**

与现有检测方法对比，该系统在基准数据集上实现了100%精度、97.7%召回率，单合约平均分析时间约4.3秒，检测效率显著提升。

**⚠️ 局限性**

主要局限包括对高度混淆合约的检测能力有限、聚类依赖可观测链上关联，且对链外成本与真正受害者区分存在上限估计。

---

## 135. Characterizing LLM Kernel Access and Memory Interaction in Multi-Partition NUMA GPUs

**arXiv ID:** 2607.28824 | [PDF](https://arxiv.org/pdf/2607.28824v1)

**作者:** Donghyeon Joo `[一作]` (University of Maryland), Bahar Asgari `[通讯]` (University of Maryland)

**通讯引用:** 711 | [OpenAlex ID](https://openalex.org/A5059742939)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多分区NUMA GPU上对LLM推理核心的内存访问模式进行追踪分析与周期级仿真，量化了跨分区通信对延迟的影响，并提出基于工作组划分的放置与调度优化方案。

**💡 创新点**

提出将工作组级共享模式分为全局、部分、私有三类，并基于此构建“Partition Locality”指标；同时展示不同LLM操作的NUMA敏感度与可达加速上限，揭示单一调度策略无法覆盖所有情况。

**🔧 技术方法**

使用Omniprobe采集虚拟地址内存访问轨迹，构建工作组级共享分析管道；利用扩展的MGPUSIM周期级模拟评估跨分区延迟；并进行基线与理想配置对比。

**📊 数据集**

在AMD Instinct MI300X GPU上对vLLM、SGLang的权重投影、注意力（MHA、GQA、MLA）、FlashAttention预填充、Mixture-of-Experts等内核进行评测，使用LLaMA-2-7B、Llama-3-8B、DeepSeek‑V3‑671B等模型。

**📈 对比分析**

通过对比默认循环分区调度/页面交错与理想的零跨分区延迟配置，测得从1.09×（FA预填充）到1.79×（GQA解码）的加速，表明NUMA敏感度随共享模式和操作类型而异。

**⚠️ 局限性**

不足之处在于缺乏统一的放置策略，需在硬件层面实现细粒度页面与工作组映射；动态路由导致的共享模式变化难以预先处理；实验仅覆盖单GPU MI300X，未验证跨GPU或更大规模场景。

---

## 136. Hollow-LLM Attack: Computationally Trivial Weights in Zero-Knowledge Verification of LLM Inference

**arXiv ID:** 2607.28884 | [PDF](https://arxiv.org/pdf/2607.28884v1)

**作者:** Chen Gong `[一作]` (University of Southern California), Mengyuan Li `[通讯]` (University of Southern California)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了零知识证明 (ZK) 驱动的 LLM 推理中的“努力差距”，并提出了 Hollow-LLM 攻击，该攻击通过在模型权重中注入“幽灵权重”实现对声明模型规模的虚假放大，实际推理仅使用更小的内部模型。

**💡 创新点**

创新点在于：① 发现 ZK 证明仅保证等式正确性，而不绑定实际计算量；② 设计了两种无训练、纯代数构造（深度扩展与宽度扩展）来创建幽灵权重；③ 利用 Transformer 的残差恒等、LayerNorm 复制和块对角权重等 invariances，实现模型规模与推理成本脱钩。

**🔧 技术方法**

使用的技术包括：预层归一化 (pre‑LN) Transformer 架构、残差恒等层、元素级非线性复制、LayerNorm 复制、块对角化的多头注意力与前馈网络权重、以及对输出层的零填充或 1/m 缩放处理。

**📊 数据集**

实验使用了两组内部模型：IM_1（6 层 512 维 GPT‑2 轻量模型）和 IM_2（12 层 1024 维 GPT‑2 参考大模型），通过在声明层面应用攻击构造进行对比。

**📈 对比分析**

对比方法：在保持内部模型不变的前提下，分别修改声明模型的深度（Attack A）、宽度（Attack B）或两者组合（A+B），测量推理端延迟、算力、内存消耗与 ZK 证明的门数、证明时间、验证时间。结果表明：推理端性能保持不变，而证明成本随声明模型规模成正比，且深度、宽度扩展可独立叠加。

**⚠️ 局限性**

局限性：① 仅适用于预层归一化的 Transformer；② 对固定正弦位置编码等不可复制的编码方式不适用；③ 需要攻击者在不修改电路的前提下构造幽灵权重，实际部署可能受硬件/协议限制；④ 本研究未解决如何通过 ZK 证明或其他手段绑定计算量，只提供了潜在的风险提示。

---

## 137. Multi-Agent Planning with Spatio-Temporal and Topological Constraints using STL-GO

**arXiv ID:** 2607.28679 | [PDF](https://arxiv.org/pdf/2607.28679v1)

**作者:** Sheryl Paul `[一作]` (University of Southern California), Jyotirmoy V. Deshmukh `[通讯]` (University of Southern California)

**通讯引用:** 2617 | [OpenAlex ID](https://openalex.org/A5057473400)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于STL-GO的多智能体规划框架，并给出了可执行的MIP和SMT编码。

**💡 创新点**

通过STL-GO实现时间变图的多图量化与邻域计数，首次将多图交互约束转化为可满足性与优化合成问题。

**🔧 技术方法**

采用离散时间仿射动力学的混合整数规划与线性整数/实数SMT求解器，并构建统一接口以支持两种求解方法。

**📊 数据集**

在多无人机搜索救援的合成场景（定位者与救援者）以及改编自HypRL的网格世界进行实验。

**📈 对比分析**

通过团队规模与图复杂度的消融实验比较两种方法；MIP在支持目标优化时求解时间较长但能得到最优计划，SMT在约束数和求解速度上更快但仅提供满足性；在最难配置下，两者均出现求解时间超限。

**⚠️ 局限性**

局限性包括仅适用于确定性同质动力学与全可观测条件，无法处理随机环境、分布式执行以及学习式决策；图构造需满足MIP/SMT可编码性。

---

## 138. High-Level Big Integer Arithmetic in Futhark for GPUs

**arXiv ID:** 2607.28897 | [PDF](https://arxiv.org/pdf/2607.28897v1)

**作者:** Cosmin E. Oancea `[一作]` (University of Copenhagen), Stephen M. Watt `[通讯]` (University of Waterloo)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了基于GPU的多精度整数算术（加、乘、除）在Futhark高层函数语言中的实现。

**💡 创新点**

提出了自动将中间数组放入寄存器的编译器优化，并通过高层函数式构造实现与手写代码相近的性能。

**🔧 技术方法**

使用Futhark的SOACs（map、scan、reduce等）、函数融合、自动寄存器分配、以及CUDA原生实现对比。

**📊 数据集**

使用随机生成的2^12~2^19位整数数据集，并在实例数与位数乘积恒定为2^32的情形下进行测试。

**📈 对比分析**

通过与手写CUDA实现(CudaP)和CGBN库对比，Futhark加法在大位数下可达1.4×速率；乘法在大位数下与手写实现相当；除法略慢，但在使用寄存器分配时比共享内存版快2.4×，且比CGBN快3–7×。

**⚠️ 局限性**

受限于Futhark目前不支持128位乘法、除法实现较慢、对极大位数（>2^15）除法支持不足，以及寄存器分配与共享内存占用的权衡。

---

## 139. Predicting Steel Fatigue Life from Micrographs Using Physics-Informed Deep Learning

**arXiv ID:** 2607.28695 | [PDF](https://arxiv.org/pdf/2607.28695v1)

**作者:** Aryuemaan Kumar Chowdhury `[一作]` `[通讯]` (Indian Institute of Technology Hyderabad), Aryuemaan Kumar Chowdhury (Indian Institute of Technology Hyderabad)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种端到端的计算机视觉管线，利用光学显微图像预测轻质合金钢的疲劳寿命，并给出校准的不确定性置信区间。

**💡 创新点**

创新点包括：①基于疲劳力学的28维物理信息特征工程；②使用高斯负对数似然（GNLL）实现异方差不确定性预测；③为金属显微学定制的七阶段预处理流程；④将Grad‑CAM解释性与四级疲劳损伤阶段对齐；⑤公开可配置的物理标记化合成显微镜数据生成器。

**🔧 技术方法**

主要技术包括：OpenCV预处理、基于物理学的特征提取、ResNet‑50/SE‑CNN/VGG‑16等卷积神经网络的双头回归模型、GNLL损失、SHAP特征重要性分析、梯度归一化CAM可视化、置信度校准（ECE评估）及风险分层分类。

**📊 数据集**

使用由Voronoi晶粒、随机行走裂纹、泊松孔洞和椭圆夹杂物组成的物理约束合成显微图像数据集，样本量约200张，标签遵循手工构建的疲劳寿命经验公式。

**📈 对比分析**

与传统表格基机器学习（GBM、SVR）以及未加不确定性的MSE回归进行对比，ResNet‑50在合成测试集上达到R²=0.93、RMSE=0.18 log‑cycles、ECE=0.021、宏观F1=0.91，显著优于基线。

**⚠️ 局限性**

局限性主要为：①合成到真实数据的领域差距；②对低碳钢/低合金钢的专门设计，可能不适用于钛合金或铝合金；③仅基于二维显微图，无法捕捉三维缺陷；④未对疲劳寿命进行起始期和扩展期的分解。

---

## 140. Think2Go: Generative Next POI Recommendation with LLM Reasoning

**arXiv ID:** 2607.28997 | [PDF](https://arxiv.org/pdf/2607.28997v1)

**作者:** Zhuang Zhuang `[一作]` (Dalian University of Technology), Baocai Yin `[通讯]` (Dalian University of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于大型语言模型的生成式下一个兴趣点推荐框架Think2Go，融合了监督微调与强化学习，利用“思考→去→自校正→答案”结构实现推理与记忆的协同优化；

**💡 创新点**

创新点包括：①将SFT与RL统一到同一架构，实现记忆与推理的无缝融合；②引入基于核密度估计的时空先验不确定性（STEU）与难度感知奖励缩放（DRG）来对优势进行校准，形成隐式课程学习；③在奖励设计中加入格式化奖励与分层语义ID奖励，提升学习信号的稠密性；

**🔧 技术方法**

技术方法主要涵盖：大型语言模型微调、基于GRPO的强化学习、优势校准（STEU+DRG+token entropy）、核密度估计、列表式语义ID表示与自校正提示；

**📊 数据集**

实验数据集为三大真实世界LBSN轨迹集：Foursquare‑NYC、Foursquare‑TKY和Gowalla‑CA；

**📈 对比分析**

与传统序列模型、图模型以及多种LLM基线（如GNPR、Refine‑POI、LLM‑Mob等）进行对比，Think2Go在Acc@1、Acc@5、Acc@10、MRR等指标上均超越所有基线，提升幅度约20–30%；在跨域测试中表现亦优于同类方法；

**⚠️ 局限性**

局限性主要体现在：①对算力要求较高，尤其是LLM推理与RL训练并行；②稀疏轨迹和极端用户行为仍难以充分利用；③模型对超大规模语义ID空间的泛化仍受限，需进一步探索更高效的ID表示与训练策略。

---

## 141. WaiT for the Signal: Simple Frequency-Aware Flow-Matching

**arXiv ID:** 2607.28760 | [PDF](https://arxiv.org/pdf/2607.28760v1)

**作者:** Krunoslav Lehman Pavasovic `[一作]` (FAIR, Meta), Jakob Verbeek `[通讯]` (FAIR, Meta)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在像素空间扩散模型中利用离散小波变换的频率感知噪声调度方法，称为WaiT。

**💡 创新点**

通过在低频先行、延迟高频的噪声调度，实现对图像频率层级的自适应建模，显著提升生成质量并节省计算。

**🔧 技术方法**

采用离散小波变换（DWT）、其无损逆变换、定制的延迟线性噪声调度以及 Transformer 结构。

**📊 数据集**

在 ImageNet 512×512、OpenImages 512/1024、SA‑1B、DataComp Multimodal、Kinetics‑600 等数据集上进行评估。

**📈 对比分析**

与 JiT、Latent‑Space 以及其他像素空间模型对比，在 FID、5cFID、hFWD 上均实现 Pareto‑最优；ImageNet 512 FID 仅 1.3，文本生成 1024 的吞吐率提升 3 倍。

**⚠️ 局限性**

仅使用单层 DWT 的压缩深度有限，未来需探索更深层压缩以及高分辨率视频的三轴评估。

---

## 142. Are the Financial Reasoning from LLMs Credible? A Real World Test over Long-Horizon Statements

**arXiv ID:** 2607.28661 | [PDF](https://arxiv.org/pdf/2607.28661v1)

**作者:** Xinke Tong `[一作]` (Alibaba Group), Dayiheng Liu `[通讯]` (Alibaba Group)

**通讯引用:** 1936 | [OpenAlex ID](https://openalex.org/A5062188134)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个大规模、长文本表格推理基准 FinIndices，评估 LLM 在真实财务报表中的数值计算与表格生成能力。

**💡 创新点**

创新点包括：①使用未裁剪的完整财报（最长 33k token）和对抗式陷阱提升任务难度；②将任务拆分为单值计算与多指标表格生成两种范式；③系统揭示“知识瓶颈”和“结构瓶颈”两大 LLM 失效模式；④通过监督微调实现结构化推理的显著提升。

**🔧 技术方法**

技术手段：自动化数据合成管道（表格抽取、上下文构造、公式注入、专家验证）；对抗式陷阱设计；严格的评估协议（完整表格匹配、数值近似判定）；SFT 以结构化推理轨迹为训练数据。

**📊 数据集**

数据集来源于 829 家上市公司、384 个财务指标、28 个报告期，生成 68,307 个样本（单值 4,192+ 表格 1,131）。每个样本包含多周期信息与多格式表格。

**📈 对比分析**

与现有 FinQA、TAT‑QA、FinEval 等基准相比，FinIndices 的上下文长度平均 16k token，需跨周期、跨标量（点 vs. 期）推理。基线 LLM 在无公式提示时单值任务 38% 左右、表格任务 17%；在提示下提升至 79%（单值）和 70%（表格）。SFT 在无提示条件下提升 8.5%（单值）和 3.8%（表格），表现出结构化推理可训练性。

**⚠️ 局限性**

局限：仅覆盖核心报表（资产负债表、利润表、现金流），未包含高频行情、宏观时间序列或内部运营数据；缺少多模态与跨源数据融合，无法全面评估金融 LLM 在更广泛业务场景中的鲁棒性。

---

## 143. RAID: Towards Robust AI-Generated Image Detection with Bit-Reversed Images

**arXiv ID:** 2607.28974 | [PDF](https://arxiv.org/pdf/2607.28974v1)

**作者:** Renxi Cheng `[一作]` (Southeast University), Hongsong Wang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究AI生成图像检测，提出基于位平面分解的可逆位反转图像和单片段卷积分类器，构建轻量高效的检测管线。

**💡 创新点**

引入可逆位反转图像放大低位噪声，设计梯度基选片机制和改造的ResNet-50分类器，并提供数学证明其有效性。

**🔧 技术方法**

使用位平面分解与重构、位反转、梯度分散评分选片、改造的ResNet-50网络、理论推导与实验验证。

**📊 数据集**

使用AIGCDB、GenImage、以及作者自建的GID与GVD四大挑战性数据集。

**📈 对比分析**

与多种SOTA方法（如DIRE、ESSP、LaRE^2等）在交叉生成器、跨数据集、零样本等多场景下对比，平均准确率超过98%，推理速度快100倍、参数量仅2.35亿。

**⚠️ 局限性**

仅使用标准ResNet-50作为分类器，缺乏专门针对位反转图像设计的网络，可能在极端或新型生成器场景下性能受限。

---

## 144. Don't Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform

**arXiv ID:** 2607.28971 | [PDF](https://arxiv.org/pdf/2607.28971v1)

**作者:** Seungho Han `[一作]` (Danggeun Market Inc. (Karrot)), Jin Yu `[通讯]` (Danggeun Market Inc. (Karrot))

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Region-Constrained Batch Sampling (RCBS) 方法，在地理受限的本地社区平台上通过构造同区域的 mini‑batch，减少不可行负样本，提高用户表征质量并用于下游检索、排序及广告投放。

**💡 创新点**

创新点在于：①识别并解决“不可行负样本”问题；②在不改动模型架构与损失的前提下，仅通过批次构造实现负样本筛选；③利用可行负样本天然成为更难负样本，从而提升对比学习信号。

**🔧 技术方法**

技术主要包括对比学习（InfoNCE）与两塔 Transformer 模型、RCBS 采样算法、离线评估（Recall@10/100、NDCG、ROC‑AUC 等）以及线上 A/B 测试。

**📊 数据集**

数据集：Karrot 平台两年生产日志，约 25 M 用户、15 B 交互记录，用于预训练；下游任务使用与预训练不重叠的短周期日志（3 周、1 周、4 周）。

**📈 对比分析**

与随机批处理对比：在随机评估与 RCBS 评估下，Recall@10/100 均提升 20‑30%；在检索、排序和广告等下游任务中，NDCG、ROC‑AUC 等指标均有 1‑7 % 的相对提升；线上 A/B 测试显示点击率提升约 10%、eCPM 提升 6%。

**⚠️ 局限性**

局限性：①区域划分为静态，无法适应用户位移或动态曝光半径变化；②仍需结合 IPS 等方法进一步缓解曝光不均衡；③在极大规模或多区域多样化的场景下，RCBS 的负样本多样性可能受限。

---

## 145. SafeNexus: Discovering and Steering Modality-Universal Safety Neurons in MLLMs

**arXiv ID:** 2607.28969 | [PDF](https://arxiv.org/pdf/2607.28969v1)

**作者:** Jian Yu `[一作]` (Nanjing University of Science and Technology), Tat-Seng Chua `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出SafeNexus框架，采用神经元级定位-强化策略，对多模态LLM进行跨模态安全对齐。

**💡 创新点**

发现并利用跨模态通用安全神经元（US‑Neurons），并设计激活放大器和安全神经元校准器，实现高效、安全且可解释的对齐。

**🔧 技术方法**

神经元重要性评分、对比激活、掩码干预、LoRA微调、激活放大等神经元级干预技术。

**📊 数据集**

Omni‑Safe, HarmBench, Lingua‑Safe, JALM, OmniBench, AV‑Odyssey, OKTest, VideoSafetyBench等多模态安全与通用评测集。

**📈 对比分析**

与ECSO、Immune、SARSteer、SPA‑VL、ProEAT等基线对比，在多模态安全指标上显著降低攻击成功率（ASR下降30‑60%），同时保持或提升通用能力，过度拒绝率提升不足3%。

**⚠️ 局限性**

仅聚焦于文本/图像/音频，未在视频或其他模态中进行充分训练；对不同模型的泛化需进一步验证；高层参数微调可能影响模型的细粒度能力。

---

## 146. A robust association between LLM use and scientific productivity: Assessing stopping-time selection

**arXiv ID:** 2607.28968 | [PDF](https://arxiv.org/pdf/2607.28968v1)

**作者:** Keigo Kusumegi `[一作]`, Yian Yin `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

检验LLM使用与科研产出之间的正相关性，并针对RBB提出的停止时间检测误差采用多种实验设计验证该关联的稳健性。

**💡 创新点**

提出多重替代实验设计（前后比较、保守对照组、强度规范、排名测量以及匹配保真放线），在每种设计下消除或中和停止时间机制，证明正相关性不受该误差影响。

**🔧 技术方法**

使用事件研究、差分中的差分（DiD）、加权回归、随机放线校准、强度回归、排名分层比较等计量方法。

**📊 数据集**

基于论文摘要的LLM检测器标记数据、作者产出计数（发表数量），时间跨度覆盖ChatGPT之前与之后（2020-2026），以及对应的预置数据用于放线检验。

**📈 对比分析**

与匹配的随机放线、对照组、前后比较等对比，结果显示正向关联在所有设计下均显著，预置数据检验保持为零，证明方法稳健且性能良好。

**⚠️ 局限性**

研究无法解决技术采用的内生性，估计仅表明关联方向而非因果大小；检测器可能存在误报/漏报，且效应随技术发展而变化。

---

## 147. Mining Verdict Boundaries for Neural Network Verification

**arXiv ID:** 2607.28954 | [PDF](https://arxiv.org/pdf/2607.28954v1)

**作者:** Jiawei Ren `[一作]` (University of New South Wales), Yulei Sui `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在分支定界（Branch-and-Bound）框架下提出两种高效的判决边界搜索方法（指数搜索与梯度搜索），通过一次性划分多重激活函数来跳过不必要的子问题，显著减少验证过程中对近似验证器的调用次数；

**💡 创新点**

创新点在于将每条搜索路径的判决边界定位问题视为近乎有序数组的搜索问题，利用近似验证值的单调性实现指数级跳跃和梯度预测，从而大幅减少节点访问；

**🔧 技术方法**

采用指数搜索、二分搜索、梯度估计、改进的ReLU选择策略（Top‑k并行分裂）以及现有的近似验证器（如DeepPoly）组合；

**📊 数据集**

使用VNN‑COMP常用的5个神经网络模型（2×256、4×256、2Conv+2Linear、2Conv+2Linear+…）共500个验证实例，数据集来自ImageNet/汽车等；

**📈 对比分析**

与基准工具（BaB、Falsifier）对比，验证通过率不低于或略优于BaB，平均时间缩短约17%–30%，最优场景可达44.7%（对比BaB），在不完全性测试中也能保持竞争力；

**⚠️ 局限性**

局限性包括：对非单调路径的处理仍不完备，偶尔梯度估计误差导致退化或回溯；方法仍以CPU为主，未利用GPU并行；需进一步改进对非单调路径的鲁棒性和整体可扩展性。

---

## 148. Advances, challenges, and opportunities for legged robots

**arXiv ID:** 2607.28952 | [PDF](https://arxiv.org/pdf/2607.28952v1)

**作者:** Jonas Frey `[一作]`, Marco Hutter `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `51c0528b-f690-4182-ae60-bb5f046c276c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了腿式机器人在硬件、运动学、自治、数据与应用等方面的最新进展，并对其社会、经济与伦理影响进行评估。

**💡 创新点**

创新点在于将硬件、控制、学习与社会影响等多维度系统化，总结关键挑战并提出整合性方法与监管框架。

**🔧 技术方法**

采用强化学习、域随机化、仿真到真实迁移、深度视觉/语言模型、开源硬件和多模态传感融合技术。

**📊 数据集**

使用仿真数据、机器人遥控/演示数据、视频/动作捕捉、公开数据集（如GrandTour、Sub‑T、ANYmal、Spot 等）进行训练与评估。

**📈 对比分析**

与传统模型预测方法相比，RL+仿真+域随机化在多地形、负载、速度与能耗等指标上表现更鲁棒，实验结果显示显著提升。

**⚠️ 局限性**

主要限制包括 sim‑to‑real 问题、缺乏统一基准与标准、对安全与伦理的系统验证不足，以及实现高阶社会交互与精准操纵的技术挑战。

---

## 149. FairDiffuseVQVAE: Sampling-Time Fairness in Tabular Diffusion via Conditional Refinement of Vector-Quantized Latents

**arXiv ID:** 2607.28945 | [PDF](https://arxiv.org/pdf/2607.28945v1)

**作者:** Nitish Nagesh `[一作]` (University of California, Irvine), Amir M. Rahmani `[通讯]` (University of California, Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段公平表格数据生成框架FairDiffuseVQVAE，分离了质量与公平性。

**💡 创新点**

通过在采样时利用分类器无指导的条件化将公平性嵌入生成分布，而非在训练时加入公平损失，实现质量与公平的解耦。

**🔧 技术方法**

使用向量量化自动编码器+基于EDM的输入空间扩散修正，并使用分类器无指导（CFG）条件化与均匀敏感属性采样。

**📊 数据集**

在Adult、Bank、COMPAS等八个表格数据集（包括TabSyn和公平基准）上进行评估。

**📈 对比分析**

与TabSyn、TabDDPM、FairTGAN、FairTabDDPM等基线对比，在公平性指标DPR和EOR上取得最高平均值（0.702/0.686），但在TSTR AUC上下降约15点。

**⚠️ 局限性**

对小样本数据易记忆导致DCR升高，种子敏感性高，公平性仅在采样时满足，缺乏理论保证。

---

## 150. NeSyFS: A Neuro-symbolic Fast-Slow Thinking Framework for LLM Agent under Partial Observability

**arXiv ID:** 2607.28942 | [PDF](https://arxiv.org/pdf/2607.28942v1)

**作者:** Duo Xu `[一作]` (Georgia Institute of Technology), Faramarz Fekri `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出NeSyFS框架，利用知识图谱记忆、快慢思维与反思模块，解决LLM代理在部分可观测环境下的隐状态推断、目标对齐和不确定规划问题。

**💡 创新点**

创新点在于将知识图谱作为隐状态的结构化记忆，并结合快慢思维与步骤级反思；此外提出基于Twisted Sequential Monte Carlo 的符号化不确定规划算法。

**🔧 技术方法**

使用大语言模型（GPT‑5 / Llama‑3.3）、知识图谱检索与更新、快慢思维（ReAct+反思）、自洽一致性技术以及基于TSMC的规划框架。

**📊 数据集**

实验数据集为 ALFWorld、WebShop 与 ScienceWorld 三个文本环境。

**📈 对比分析**

与 ReAct、Reflexion、ABBEL、RAFA、SwiftSage 等基线对比，NeSyFS 在三大 benchmark 上成功率和平均奖励均显著提升，约提升 30% 以上。

**⚠️ 局限性**

局限性包括对知识图谱构建与检索的依赖、LLM 对任务进展评估的可靠性受限、对实时动态环境的适用性尚未验证，以及在更大规模真实世界任务中的可扩展性未知。

---

## 151. DiffAttack: Evasion Attacks Against Face Recognition via Latent Diffusion Models

**arXiv ID:** 2607.28936 | [PDF](https://arxiv.org/pdf/2607.28936v1)

**作者:** Omid Ahmadieh `[一作]` (University of South Florida), Nima Karimian `[通讯]` (University of South Florida)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种DiffAttack的全局潜空间优化框架，通过在冻结的Stable Diffusion v2.1模型上使用LoRA低秩适配器，对整张人脸的潜在向量进行微调，从而生成与目标身份高度相似的对抗面部图像。

**💡 创新点**

创新点在于：①全图潜空间的LoRA微调替代传统局部遮罩或像素噪声，天然利用扩散模型的生成先验实现无显著边缘伪影；②采用多模型（IR152、IRSE50、MobileFace）融合反馈的对抗损失，显著提升黑盒迁移性能；③结合识别对齐、方向约束和源抑制三项损失，确保对抗样本既能误导识别系统，又保持源图像的纹理、光照与表情。

**🔧 技术方法**

技术栈包括：Stable Diffusion v2.1、LoRA低秩适配器、跨注意力层微调、三模型融合的对抗损失、以及FID/PSNR/SSIM评估。

**📊 数据集**

实验使用了两个公开高质量人脸数据集：FFHQ和CelebA-HQ。

**📈 对比分析**

在黑盒攻击场景下，与噪声、化妆、语义和局部遮罩等SOTA方法对比，DiffAttack在FFHQ和CelebA-HQ上平均攻击成功率达84.86%，比传统方法高出约15-20%；视觉质量指标为FID 27.58、PSNR 24.54 dB、SSIM 0.961，显示出更佳的图像真实性。

**⚠️ 局限性**

局限性包括：对抗样本仍可能在极端姿态或光照变化下失效；需要依赖已预训练的扩散模型；未在真实物理环境中验证生成效果；针对跨域或多模态识别系统的迁移性仍待进一步研究。

---

## 152. EvoReason: Self-Evolving Reasoning Primitive-Guided On-Policy Distillation for Latent Reasoning in Generative Recommendation

**arXiv ID:** 2607.29010 | [PDF](https://arxiv.org/pdf/2607.29010v1)

**作者:** Zhuang Zhuang `[一作]` (Kuaishou Technology), Fei Pan `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

EvoReason提出了一种自演进的原语引导式Latent Reasoning框架，通过教师与学生的闭环迭代，将结构化的Chain-of-Thought监督高效转移到隐式推理空间，从而提升生成式推荐的性能。

**💡 创新点**

创新点在于构建可自演进的Reasoning Primitive库，并将其作为工具引导教师生成结构化CoT；同时采用自演进的On‑Policy Distillation实现教师与学生的闭环自适应监督，解决传统Latent Reasoning中静态监督与分布不匹配的问题。

**🔧 技术方法**

使用技术包括ReAct风格工具调用、语义Primitive抽取与更新、on‑policy distillation、强化学习优化隐式推理策略、KV对齐、token‑level KL损失、confidence gating等。

**📊 数据集**

实验数据集包含Amazon Product Review公开数据集Beauty、Sports以及一份工业级数据集；同时在生产广告系统进行线上A/B测试。

**📈 对比分析**

与经典序列模型（GRU4Rec、SASRec等）、生成式推荐模型（HSTU、TIGER、ReaRec）以及LLM基推荐模型（Onerec‑think、FLR、LatentR3、LASAR）进行对比；EvoReason在Recall@5/10、NDCG@5/10上分别提升约15–30%，在工业线上实验中实现广告收入+6.23%、广告价值+8.11%。

**⚠️ 局限性**

局限性包括：依赖高质量agentic轨迹进行原语提取，迁移到新领域需重新构建原语库；自演进机制在训练时间与资源上略高；对极端稀疏场景的鲁棒性尚未充分验证。

---

## 153. PaletteID: Prototype-Composed Semantic Identifiers for Multimodal CTR Prediction

**arXiv ID:** 2607.29000 | [PDF](https://arxiv.org/pdf/2607.29000v1)

**作者:** Huanyu Liu `[一作]` (Huazhong University of Science and Technology), Ziyi Huang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PaletteID，一种基于真实项目原型的软组合语义标识符，用于提升多模态点击率预测。

**💡 创新点**

创新点在于通过语义质量感知 DPP 选取原型调色板，并用连续的相似度门控对原型进行加权，既保留细粒度语义，又提升可解释性。

**🔧 技术方法**

使用了 Determinantal Point Process、Cosine‑RBF 核、相似度加权门（Sigmoid）、离线原型检索与聚合，配合深度 CTR 模型（DCNV2、RankMixer、DIN）。

**📊 数据集**

实验数据集为公开的 TAOBAO‑MM（1M 商品、7.2M 用户）和 KuaiRec（10.7K 商品、3.8K 用户）。

**📈 对比分析**

与 VQ‑VAE、RQ‑VAE、RQ‑KMeans 以及无语义标识符 baseline 在 DCNV2/RankMixer/DIN 上对比，PaletteID 在 AUC/GAUC 及长尾项目上均取得更大提升，且在鲁棒性与可解释性上优于现有方法。

**⚠️ 局限性**

局限性包括需预先离线选取原型调色板、对超大商品库的实时适配仍有限，且原型数量和检索阈值等超参数仍需经验调优。

---

## 154. Receding-Horizon Next-Best-View Planner for Autonomous Leaf Surface Reconstruction

**arXiv ID:** 2607.28995 | [PDF](https://arxiv.org/pdf/2607.28995v1)

**作者:** Arif Ahmed `[一作]` (University of Nevada), Parikshit Maini `[通讯]` (Missouri University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于回溯视角的下一最佳视角（NBV）规划方法，用于无人机在田间环境中对植物叶片表面进行高精度重建。

**💡 创新点**

创新点包括：①引入以叶片已观测点质心为参考的质心信息增益（CIG）函数，能够优先选取能观测到与已有观测点距离更远的叶片区域；②在预算约束下采用回溯窗口（w=3）的规划策略，避免传统贪婪选择导致的冗余采集。

**🔧 技术方法**

核心技术包括：HPR（Hidden Point Removal）作为可见性判定；CIG函数计算基于质心距离的加权信息增益；回溯窗口规划求解；Poisson重建用于生成叶片表面；并用F1分数、Hausdorff距离、RMS、RE等指标评估。

**📊 数据集**

使用了LAST-STRAW公开数据集，包含两个草莓植株在11周内共14个生长阶段的高分辨率三维点云及叶片实例标签。

**📈 对比分析**

与基准Attention‑NBV（仅基于可见叶片点计数的增益）进行对比。结果显示：在所有预算水平下，RH‑CIG在叶片点云覆盖率上优于基准（最高提升约10%），在表面重建质量上（HD、RMS、RE）亦均优于基准，尤其在中后期生长阶段受遮挡影响较大时优势最为显著。

**⚠️ 局限性**

局限性包括：①在极少遮挡的早期阶段，回溯规划并未显著提升效果；②当单叶片仅采集到极少点（如B11阶段）时，Poisson重建会失败，导致表面指标不准确；③方法依赖离线可见性计算，实时实现仍需进一步加速。

---

## 155. ST-WAM: Semantic-Temporal World Action Model for Robust Manipulation under Visual Distribution Shifts

**arXiv ID:** 2607.28993 | [PDF](https://arxiv.org/pdf/2607.28993v1)

**作者:** Mingxin Wang `[一作]` (Tsinghua University), Tianlun Li `[通讯]` (Alibaba Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了Semantic-Temporal WAM（ST‑WAM），通过结合DINOv3语义表示与VAE视觉潜在，提升视频生成世界动作模型在视觉分布偏移下的鲁棒性；不需要大规模有机预训练，可端到端训练并在部署时仅生成动作；

**💡 创新点**

创新点在于：①双空间未来专家（DSFE）同时预测VAE与DINO未来潜在，实现细粒度视觉动态与视觉稳定语义的互补；②当前语义锚定的历史检索（CAIR），利用Qwen3‑VL提取的当前语义与DINO历史进行交叉注意，提供任务相关的短期意图；③三支分支Mixture‑of‑Transformers与结构化跨分支注意力遮蔽，避免未来信息泄漏；整体框架在不依赖有机预训练的情况下，显著提升了对视觉偏移的鲁棒性。

**🔧 技术方法**

使用技术包括：DINOv3 ViT‑S/16 作为语义编码；Wan2.2 VAE 用于视觉潜在；三支 DiT（视觉、语义、动作）专家组成 Mixture‑of‑Transformers；流匹配（flow‑matching）训练；跨分支注意力遮蔽；Qwen3‑VL 提取当前语义锚；流积分推理（10 步）实现动作块预测。

**📊 数据集**

实验使用数据集：LIBERO（Spatial、Object、Goal、Long）及其对抗版本 LIBERO‑Plus；RoboTwin 2.0 机器人二指操控；Agilex Piper 6‑DoF 真实机器人，包含 5 个多时序任务；以及 50 条演示的训练混合集（2,500 干净 + 25,000 随机化）。

**📈 对比分析**

与多种基线（π₀、π₀.₅、Fast‑WAM、Motus、LingBot‑VA、Mask World Model、MaskWAM、GeoSem‑WAM、LaWAM 等）比较。ST‑WAM 在 LIBERO 上平均 98.7%（最高），RoboTwin 2.0 上平均 92.77%；在 LIBERO‑Plus 零样本提升 21.3pp 取得 72.8%，比 Fast‑WAM 提升 39.0pp 及 41.8pp；在真实环境视觉偏移下成功率 61.5%，比 Fast‑WAM 提升 35.7pp。推理延迟为 756 ms（相较 Fast‑WAM 609 ms）。

**⚠️ 局限性**

局限性包括：仅针对视觉分布偏移的鲁棒性，未验证对物理动力学或机器人结构变化的适应；依赖 DINOv3 语义表达，若语义模型无法捕获关键细节会受限；推理时仍需多分支 transformer，延迟相对较高；未来仍需探索更高效的语义与动力学联合建模。

---

## 156. Shapley-Value-Based Feature Attribution for Data Masking

**arXiv ID:** 2607.28946 | [PDF](https://arxiv.org/pdf/2607.28946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 157. Scaling Scientific Discovery Environments for Turn-Level Agentic RL

**arXiv ID:** 2607.28990 | [PDF](https://arxiv.org/pdf/2607.28990v1)

**作者:** Yucheng Xu `[一作]` (Shanghai AI Lab), Zhongying Tu `[通讯]` (Shanghai AI Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了SciDisco框架，构建可执行、可验证的科学发现环境，并在此环境中训练基于强化学习的LLM代理；

**💡 创新点**

首次将隐藏证据DAG与verifier结合，实现过程级奖励分配，让代理能在多轮交互中被逐步引导完成科学分析；

**🔧 技术方法**

使用SciThèque生成任务环境、DAG‑grounded trajectory synthesis做冷启动SFT、DiscoPO算法进行基于verifier的turn‑level奖励强化学习，辅以GRPO与PPO；

**📊 数据集**

利用公开科学数据集（如多个科研数据集）与模板化假设，生成数千个任务环境；

**📈 对比分析**

在DiscoveryBench、DataSciBench和DABStep三个基准上与GPT‑4o、GPT‑5 Mini、Claude‑Sonnet‑4等闭源模型、DeepSeek‑V4‑Flash、Intern‑S1‑Pro、Qwen系列以及DeepAnalyze‑8B等开源/专业模型对比，SciDisco‑14B在DiscoveryBench达到35.2% HMS，DataSciBench 56.2%成功率/61.0%完成率，显著优于同类开源/专业基线；

**⚠️ 局限性**

仅适用于文本可读的科学数据分析，未涵盖实验室工作、图像/多模态数据或完全开放式假设生成，且环境验收进度仅为任务契约验证，并不能完全保证科学发现的创新性与质量。

---

## 158. Adjudicated Captioning: Multi-Agent Alignment Scoring and Consensus-Distilled Beam Arbitration for Strict Zero-Shot Image Captioning

**arXiv ID:** 2607.28986 | [PDF](https://arxiv.org/pdf/2607.28986v1)

**作者:** Duy Tran Thanh `[一作]` (OneMount), Ngo Tan Vu Khanh `[通讯]` (University of Economics Ho Chi Minh City)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在不重新训练Captioner的前提下，提出 Adjudicated Captioning 框架，通过在检索、验证和生成阶段多点对齐评分，并在Beam末端加入自监督学习的重排序器，显著提升严格零样本图像字幕性能。

**💡 创新点**

核心创新点包括：①使用更强的冻结检索编码器（OpenCLIP ViT‑bigG/14）；②在检索与解码之间插入交叉注意力验证器（BLIP‑ITM）实现二次对齐；③设计两种自监督学习的Beam重排序器（TriFuse MLP与MemAttend Transformer），并通过Borda‑consensus蒸馏在不使用图文配对标签的情况下进行训练。

**🔧 技术方法**

技术手段包括：冻结CLIP与BLIP模型、GPT‑2生成器、对齐评分的交叉注意力、MLP与Transformer重排序头、Borda‑consensus伪标签、固定权重线性混合（fixed‑α）对照基准。

**📊 数据集**

数据集：训练阶段仅使用COCO Karpathy训练集的文本（566k句子）做检索索引；评估阶段在COCO Karpathy test、Flickr30k Karpathy以及NoCaps验证集上进行跨域实验。

**📈 对比分析**

通过与严格文本零样本方法（ViECap、MeaCap、IFCap、NES）以及合成图像增强方法（SynTIC、PCM‑Net、NES）对比，CIder提升至117.6（+9.6对IFCap，+7.7对NES），超越配对监督的ClipCap；在Flickr30k和NoCaps上也分别获得+8.1和+5.7的CIder提升，展示跨域迁移能力。

**⚠️ 局限性**

限制与挑战：①验证器为COCO微调版本，对COCO风格可能存在偏好；②仅测试在IFCap Captioner上，未验证对其他Captioner的普适性；③评估主要依赖COCO指标，缺乏更全面的语义或人类评价；④实验仅使用单一随机种子，未检验多种种子下的鲁棒性；⑤未尝试在非COCO检索语料下的效果，仍需进一步验证。

---

## 159. Efficient LLM Adversarial Training via Low-Rank Defense and Circuit-Guided Surrogates

**arXiv ID:** 2607.28959 | [PDF](https://arxiv.org/pdf/2607.28959v1)

**作者:** Weiyi He `[一作]` (Michigan State University), Yue Xing `[通讯]` (Michigan State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种轻量化的对抗训练框架 LAT‑ReFT，将低秩表示微调 (ReFT) 与电路引导的对抗攻击 surrogate 结合，以显著降低大语言模型的对抗训练计算成本。

**💡 创新点**

创新点在于：① 在 LAT 中使用 ReFT 并证明单点防御不足，改为后缀窗口覆盖并选择早期/中间层；② 提出 ActGrad 重要性评分，用于 MLP 剪枝构造高效 surrogate，兼顾攻击迁移与计算节省。

**🔧 技术方法**

技术方法包括低秩表示微调 (ReFT)、Latent Adversarial Training、PGD 内循环攻击、激活‑梯度重要性评分 (ActGrad) 电路剪枝、Transformer 结构分析等。

**📊 数据集**

实验使用 IMDB、EnronSpam、PasswordMatch 等分类数据集，在 Llama‑3.1‑8B、Qwen‑2.5‑3B、Pythia‑1.4B 三大 LLM 上进行评估。

**📈 对比分析**

与 LAT、CAT、R2D2 等基线比较，LAT‑ReFT 在保持仅 0.0066%–0.0203% 可训练参数的同时，平均每步 FLOPs 降低 48.1%，但在攻击成功率方面略逊于完整参数 LAT。

**⚠️ 局限性**

局限性包括仅在分类任务和后缀攻击上验证；未扩展到生成任务或自适应 jailbreak；surrogate 使用固定剪枝率，缺乏自适应或更广泛的电路提取。

---

## 160. TransX: Scaling Transformer-based Recommendation via Behavioral and Serving Stream Crossings

**arXiv ID:** 2607.28940 | [PDF](https://arxiv.org/pdf/2607.28940v1)

**作者:** Da Xu `[一作]` (LinkedIn), Nishant Satya Lakshmikanth `[通讯]` (LinkedIn)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 TransX，一种编码器-解码器架构，将推荐任务视为行为流与服务事件流的序列到序列动作转导问题。

**💡 创新点**

创新点在于显式拆分行为流与服务流，采用分组多查询稀疏交叉注意力和局部-全局稀疏注意力，实现高效交互；同时通过模型-基础设施协同设计，实现近线行为编码与 KV 缓存的推理低延迟。

**🔧 技术方法**

使用 Transformer 编码器、解码器、分组多查询稀疏交叉注意力、局部-全局稀疏注意力、近线增量编码与 KV 缓存等技术。

**📊 数据集**

在 LinkedIn 规模化社交推荐真实数据集（180 天行为历史，数亿事件）上进行实验。

**📈 对比分析**

与 DLRM、SASRec、TransAct、GRM 等基线在离线 AUC、AUPR、gAUC 上均优于对手，且在线 A/B 测试实现 CTR 提升 6.0% 与转化率提升 4.4%，推理延迟降低约 80% 但保持与 DLRM 相当的计算成本。

**⚠️ 局限性**

主要局限在于仍依赖大量离线训练与近线计算，对不同业务场景的通用性尚未验证；交叉注意力虽高效但在极长行为序列下仍可能产生计算瓶颈。

---

## 161. Latent Lie-Poisson Neural Networks (LLPNNs): Discovering the motion of Lie-Poisson systems through observable data and latent dynamics

**arXiv ID:** 2607.28939 | [PDF](https://arxiv.org/pdf/2607.28939v1)

**作者:** Vakhtang Putkaradze `[一作]` `[通讯]` (University of Alabama), Vakhtang Putkaradze (University of Alabama)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Latent Lie–Poisson Neural Networks (LLPNN)，能够从可观测的配置与速度数据学习 Lie–Poisson 动力学并重建隐藏的动量轨迹。

**💡 创新点**

创新点在于利用 Noether 常数与 Magnus 级数构造结构保持的潜在动力学，兼容非退化与退化 Hamiltonian，且不依赖对动量的观测。

**🔧 技术方法**

采用了结构保持的神经网络、Lie–Poisson 框架、共轭动量重建、Magnus 基 Lie 群更新以及热身初始化等技术。

**📊 数据集**

使用合成数据集：SO(3) 刚体、SE(3) 底部水下船体以及 SE(2)^N 多车队的模拟轨迹。

**📈 对比分析**

与 HNN、LPNet、DeepONet、Neural ODE 等方法对比，LLPNN 在长时程预测、能量与 Casimir 保守、噪声鲁棒性上均显著优于基线，误差低于 10⁻³。

**⚠️ 局限性**

主要限制是需预先知晓 Lie 群对称性，且目前仅适用于 Lie–Poisson 系统，无法直接扩展到一般 Poisson 或非对称性系统。

---

## 162. SILVA Networks as Structured Implicit Layers and Vector Attractors via Dynamic Interaction Fields

**arXiv ID:** 2607.28989 | [PDF](https://arxiv.org/pdf/2607.28989v1)

**作者:** Jose Luis Lima de Jesus Silva `[一作]` `[通讯]` (Federal University of Bahia), Jose Luis Lima de Jesus Silva (Federal University of Bahia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SILVA Networks，利用结构化隐式层将刺激、局部交互、全局交互和自持久化四个动态组成部分明确分离，并通过阻尼固定点迭代求解向量吸引子；

**💡 创新点**

创新点在于将输入刺激、局部邻域信息、全局上下文以及阻尼自持久化统一嵌入同一隐式更新框架，既能在多领域共享同一模板，又能通过路径求和、谱半径、能量诊断等方法可解释、可调节该动力学；

**🔧 技术方法**

核心技术包括阻尼 Picard 迭代、动态邻域图构造、GAT‑style 局部注意力、均值场或自注意力全局交互、谱归一化、安德森加速以及有限路径求和与特征图诊断；

**📊 数据集**

使用的公开数据集有 MNIST、CIFAR‑10（视觉），ZINC（分子属性回归），Cora/Citeseer/Pubmed（节点分类），CLUSTER（长程图节点分类）；

**📈 对比分析**

与基线对比：MNIST 上与单层全连接无显著差距，CIFAR‑10 约 74.8%（略低于主流 CNN）；ZINC 上 MAE 0.3893，优于 GCN、GAT、GIN、GatedGCN；Cora/Citeseer/Pubmed 上相对落后于 GCN/GAT；CLUSTER 上 73.04% 远优于 GCN（53.22%）和 GatedGCN（58.75%），全局交互贡献约 5.5%；

**⚠️ 局限性**

局限性包括：在简单任务（MNIST、citation‑net）全局交互无显著提升，训练过程中可能出现收敛不稳定（谱半径接近/超过 1），使用有限迭代求解且梯度通过截断反向传播，计算成本高于传统 GNN；此外在某些图规模较大时可能出现双峰收敛现象。

---

## 163. A Formalism-Aware Reward Loop for Handwritten UML-to-PlantUML Generation

**arXiv ID:** 2607.28987 | [PDF](https://arxiv.org/pdf/2607.28987v1)

**作者:** Mersedeh Sadeghi `[一作]` (University of Cologne), Adrian Psoch-Bajraktari `[通讯]` (University of Cologne)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将手写 UML 草图转化为可编译的 PlantUML 代码，利用模型分析作为训练与评估信号

**💡 创新点**

提出正式化感知奖励机制，将生成的模型结构与目标模型直接比较，既用于训练（GRPO）又用于评估

**🔧 技术方法**

使用 Qwen3.5‑4B 视觉‑语言模型，进行监督微调后结合 Group Relative Policy Optimization，奖励基于 XMI 与控制流图的结构相似度

**📊 数据集**

约 500 张手写类图与活动图与对应 PlantUML 对齐的数据集

**📈 对比分析**

与基线模型（未微调 Qwen、Gemini 3 Flash、GPT‑4.1 Mini）比较，自动化评分和人工评估均显示微调+奖励模型在编译率与结构质量上位居第二，整体性能可与大规模专有模型媲美

**⚠️ 局限性**

奖励引导阶段在当前小规模验证集上未显著优于单纯监督微调；自动化指标与人类判断仅中等相关；奖励对意义保持变形过于严格，未能完全捕捉人类可接受的语义等价

---

## 164. Beyond Feature and Structure Alignment: Learning Transferable Propagation Knowledge for Graph Foundation Models

**arXiv ID:** 2607.28980 | [PDF](https://arxiv.org/pdf/2607.28980v1)

**作者:** Yi Wang `[一作]` (Tianjin University), Dongxiao He `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 Propagation-aware Graph Foundation Model（ProGFM），通过把边与特征维度的传播关系视为可迁移知识单元，构建传播关系原型库并在图神经网络中实现传播感知消息传递，从而实现跨图域的零调优迁移。

**💡 创新点**

创新点在于：① 将传播关系（边-特征维度的相对差异）定义为跨域可迁移的知识单元；② 通过聚类得到可学习传播强度的原型库；③ 引入传播感知消息传递机制，使得不同边在不同特征维度上能自适应地调节信息聚合；④ 在预训练后无需微调即可直接迁移至新域，展示了强大的零调优能力。

**🔧 技术方法**

主要技术包括：SVD 对特征维度进行对齐；相对特征差异计算与传播关系生成；K‑means 聚类构建原型库；可学习的传播强度参数；传播感知消息传递（edge‑level 的传播强度向量）；自监督预训练（采用 SGRL 结构自监督目标）；零调优与原型分类策略。

**📊 数据集**

使用的数据集有：节点级任务的 Cora、CiteSeer、PubMed、Photo、Computers、CS；图级任务的 IMDB-BINARY、COLLAB、PROTEINS、DD；通过 leave‑one‑graph‑out 方式进行预训练与评估。

**📈 对比分析**

与 GCN、GAT、DGI、BGRL、GraphMAE、MDGFM、TIG、MDGPT、SAMGPT、LEDA、TFSGFM、SCR 等方法对比，ProGFM 在一拍/少量样本节点分类、子图分类和图分类任务上均取得了最优或近乎最优的准确率和 F1 分数，证明了其在跨域泛化方面的显著优势；消融实验和参数敏感性分析进一步验证了传播强度学习、边级关系建模和传播感知消息传递的有效性。

**⚠️ 局限性**

局限性包括：① 传播关系依赖于相对特征差异，可能忽略语义信息；② 原型库大小需调参，过大或过小都会影响性能；③ 需要对齐不同域的特征维度，可能导致信息丢失；④ 计算时需为每条边和每个特征维度匹配原型，存在一定的计算开销；⑤ 目前仅在静态图域进行验证，对动态图或大规模图的适用性尚未探究。

---

## 165. BLADE: Boundary-Expanded and Layer-Adaptive Dynamic Exit for Efficient LLM Reasoning

**arXiv ID:** 2607.28966 | [PDF](https://arxiv.org/pdf/2607.28966v1)

**作者:** Keshu Fu `[一作]` (Beihang University), Yuanxin Ouyang `[通讯]` (Beihang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种边界扩展、层适应的动态早停框架BLADE，用以在大型语言模型推理中在保持准确性的同时显著减少生成token数量。

**💡 创新点**

创新点包括：①将句子、疑问、段落等多粒度检查点与自我怀疑标记相结合，扩大早停覆盖范围；②通过自适应探测层选择（APLS）自动挑选少量有效隐藏层，避免全层拼接的冗余；③基于校准的检查点感知停止策略，在不同检查点类型上采用异步确认规则。

**🔧 技术方法**

使用的技术包括：多粒度检查点构造（MGRC）、基于无差异强一致性标注的前缀充分性监督、稠密跨层模型结合硬Top‑K门控的自适应层选择、梯度直通估计、校准阈值和加权Accuracy‑Efficiency Score (AES)评估。

**📊 数据集**

实验数据集涵盖五个数学推理基准：GSM8K、MATH-500、AMC 2023、AIME 2024、AIME 2025，使用Qwen3-8B和Qwen3-4B两种大模型进行评测。

**📈 对比分析**

与Full-CoT以及多种LYNX风格基线比较，BLADE在Qwen3-8B上平均Token减少24.8%，在Qwen3-4B上15.8%，同时保持近似或略低于Full-CoT的准确率；在AES得分上均超过对比方法，显示出更优的精度‑效率折衷。

**⚠️ 局限性**

局限性包括：①多粒度检查点覆盖仍可能产生异构状态，导致需要更复杂的校准；②自适应层选择在不同模型规模和任务上可能产生不同最优子集，缺乏统一的理论解释；③方法在非数学推理任务上的通用性尚未验证。

---

## 166. Beyond Byzantine: An Organizational Consensus Algorithm for Self-Interested Agents Under Information Asymmetry

**arXiv ID:** 2607.28957 | [PDF](https://arxiv.org/pdf/2607.28957v1)

**作者:** Jiawei Zhang `[一作]` (Purdue University), Jianbo Liu `[通讯]` (JD Logistics, JD.com)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出组织共识算法（OCA），解决自利部门在信息不对称下的内部协调问题。

**💡 创新点**

创新点在于将机制设计与事件触发异常、延迟惩罚以及信心加权聚合结合，既降低协调开销，又对战略性误报形成抑制。

**🔧 技术方法**

使用技术包括：游戏理论与机制设计、增量 Delta 传输与版本向量、置信度加权的加权平均聚合、延迟 oracle 的二次惩罚机制、Python 进行系统仿真。

**📊 数据集**

数据集为仿真生成的组织规模 N=20‑50、噪声 σ=0.02 等参数，未使用真实业务数据集。

**📈 对比分析**

与 FullStateDissemination、DeltaAntiEntropy、DeltaStateCRDT、DistributedKalmanFilter 四个基线协议对比；OCA 在协调开销上提升 99%（仅 2.8kB vs 456kB），收敛 16 轮，正确率 100%，并在 40% 作弊节点下仍保持鲁棒性。

**⚠️ 局限性**

局限性包括：仅适用于可加性、连续状态；依赖周期性 oracle 进行惩罚；缺乏完整的贝叶斯纳什均衡证明；对非凸或离散决策空间不足；对信息可识别性与合作假设要求较高。

---

## 167. Automated classification method of COVID-19 cases from chest CT volumes using 2D and 3D hybrid CNN for anisotropic volumes

**arXiv ID:** 2607.28950 | [PDF](https://arxiv.org/pdf/2607.28950v1)

**作者:** Masahiro Oda `[一作]` (Nagoya University), Kensaku Mori `[通讯]` (Nagoya University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研发了一种基于2D/3D混合特征提取的COVID‑19胸部CT卷积神经网络，用于自动将CT体积分为高低感染风险。

**💡 创新点**

创新点在于引入2D/3D混合特征流、膨胀卷积和密集池化连接，以适应各向异性体积并显著降低参数量。

**🔧 技术方法**

采用3D卷积、平面卷积、膨胀卷积、混合池化、密集池化连接、数据增强等深度学习技术。

**📊 数据集**

使用日本多家医院收集的1288份胸部CT体积数据，标注为COVID‑19高低风险。

**📈 对比分析**

与不含2D/3D混合特征流的3D CNN对比，平均准确率从79.5%提升到83.3%（±1.8%）。

**⚠️ 局限性**

仅使用少量训练样本且未加入更复杂的预处理或模型改进，可能限制了数据多样性与泛化能力。

---

## 168. Group-wise Supervision with Focal-Dice Loss for Long-Tailed Indoor Semantic Occupancy Prediction

**arXiv ID:** 2607.28935 | [PDF](https://arxiv.org/pdf/2607.28935v1)

**作者:** Qi Zheng `[一作]` (Shenzhen University), Xiao Pan `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了Group-UFD Occ框架，用于解决室内语义占用预测中的长尾类别问题。

**💡 创新点**

创新点在于采用层次语义分组和统一的焦点-Dice损失，结合主-专家预测头，显著提升尾类识别。

**🔧 技术方法**

技术手段包括多尺度主-专家分支、层次语义分组、Unified Focal-Dice（UFD）损失、3D FPN、ResNet-50+MinkResNet-34编码器。

**📊 数据集**

使用大型室内数据集EmbodiedScan进行实验。

**📈 对比分析**

与EmbodiedScan基线和COTR对比，mIoU提升约11.38%，尾类mIoU提升显著，整体性能最优。

**⚠️ 局限性**

局限在于需要人工校正LLM生成的分组，Dice损失对极其稀疏或不规则类别优化有限。

---

## 169. Universal Denoising without Channel Knowledge

**arXiv ID:** 2607.28948 | [PDF](https://arxiv.org/pdf/2607.28948v1)

**作者:** Matthias Frey `[一作]` (University of Melbourne), Jingge Zhu `[通讯]` (University of Melbourne)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于在线预测的通用去噪方案，该方案既不依赖噪声通道的已知性，也不要求信号取值为有限字母表，并能在未知源和未知通道的情况下对i.i.d.信号进行去噪。

**💡 创新点**

创新点在于：①实现了对源和通道同时的普适性；②给出了以KL散度为核心的性能上界；③证明了一个一致性条件既是充分又在可数参数空间下是必要的；④展示了混合估计策略优于传统的plug‑in方法；⑤在Poisson信号上进行数值验证。

**🔧 技术方法**

采用了信息理论（KL散度、Fisher信息）、统计学习（混合估计、最大似然、在线预测）、优化理论（可分解的加性损失）以及Poisson统计模型来构建和分析去噪策略。

**📊 数据集**

主要使用了人工生成的Poisson信号数据集（通过不同的Poisson源参数组合得到的加性噪声和信号），未使用公开真实数据集。

**📈 对比分析**

通过与理论最优贝叶斯去噪器对比，并与plug‑in方案进行对照。实验表明混合策略在有限序列长度下平均每个组件的误差迅速趋近零，而plug‑in误差保持在较高水平；在自信息损失下，总误差随长度增加而下降，符合理论预期。

**⚠️ 局限性**

局限性包括：仅针对i.i.d.信号；在连续参数空间下的总误差不收敛到零；对混合分布的计算复杂度较高；需要先验知识来确定参数空间和先验分布；在某些“几乎有理”参数设置下，理论收敛性质可能受限。

---

## 170. Nonlinear Exchange Dynamics for Independent Sets

**arXiv ID:** 2607.29016 | [PDF](https://arxiv.org/pdf/2607.29016v1)

**作者:** Mehrad Abbaszadeh Minab `[一作]` (Georgia Institute of Technology), Alistair Sinclair `[通讯]` (University of California Berkeley)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并设计了两类非线性动力学（均值场与单点交换）用于在给定密度或边缘概率的前提下从硬核模型（独立集）中采样，提出了基于粒子系统的可实现算法。

**💡 创新点**

创新点在于：①首次把硬核模型的非线性动力学与守恒量（密度/边缘概率）相结合；②利用Kac程序将非线性动力学近似为线性粒子系统；③通过新的耦合和能量收敛分析，证明在低密度下收敛速率为指数；④在临界密度附近实现对相对熵的指数衰减。

**🔧 技术方法**

采用了耦合技术、Wasserstein距离收敛、相对熵与改进的Log‑Sobolev不等式、Kac程序、粒子系统（离散时间/连续时间）以及与色彩Glauber动力学的比较。

**📊 数据集**

无实验数据集，全部为理论分析与算法设计；若有实验，使用标准图实例（如随机图、正则图）进行模拟。

**📈 对比分析**

与传统的逆问题方法（学习fugacity后使用Glauber）相比，提出的粒子系统算法实现简单、时间复杂度为O(n/ε)（密度采样）或O(n^{3/4}/ε)（边缘概率采样），在给定密度下比学习法更快；但在边缘概率采样上仍略逊于梯度下降法的二次时间。

**⚠️ 局限性**

局限性包括：①收敛证明仅在低密度（≤1/(Δ+1)或1/3(Δ+1)）下成立；②单点采样的时间复杂度仍高于最优梯度下降法；③对更高密度或临界点的收敛性尚未完全解析；④依赖于颜色采样混合时间的猜想。

---

## 171. D-VLC: Decentralized Vision-Language Collaboration for Heterogeneous Embodied Multi-Robot Systems in Unknown Environments

**arXiv ID:** 2607.29009 | [PDF](https://arxiv.org/pdf/2607.29009v1)

**作者:** Yuan Zhou `[一作]` (Zhejiang University), Fei Gao `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个去中心化异构多机器人协作框架 D-VLC，利用通用视觉语言模型在未知环境中按自然语言指令执行任务，无需针对任务或机器人进行专门训练。

**💡 创新点**

创新点包括：①将通用 VLM 与异构机器人无缝配合的异步认知‑行动循环；②基于轻量化迷你地图与事件驱动信息共享的多模空间记忆；③能力感知的任务分解与协同请求；④统一高层决策管线让 VLM 直接生成机器人可执行动作。

**🔧 技术方法**

使用的技术包括大语言模型/视觉语言模型（Qwen3.5‑Flash、Claude‑Opus‑4‑8、GPT‑5.4‑Mini 等）、Mini‑map 视觉接口、LiDAR 基础局部地图、A*、EGO Planner、TopAY、FAST‑LIO2 等低成本运动规划器，以及事件驱动的共享语义更新和能力条件推理与链式思维提示。

**📊 数据集**

实验数据来自自建的 Unity 仿真环境，包含多房间住宅、医院、灾区废墟等场景，配合多种模糊自然语言指令，每种 VLM 进行 30 次试验，未使用公开数据集。

**📈 对比分析**

与 Geometric Greedy 基准在同一机器人堆栈下对比，成功率从 66.7% 提升至 76.7–90.0%，完成时间缩短 55.8%，动作步骤下降 51.3%，展示了可靠性和效率的显著提升。

**⚠️ 局限性**

局限性在于仅在仿真环境验证，未测试真实机器人、通信带宽限制或更大团队；共享信息为事件驱动但不保证同步；VLM 仍可能出现幻觉，对极长文本任务的上下文管理仍未彻底解决。

---

## 172. SULAND v2: A Refined RGB Dataset and Deep Learning Object Detection Benchmark for UAV/UGV-Based SUrface LANDmine Detection Under Domain Shift

**arXiv ID:** 2607.28996 | [PDF](https://arxiv.org/pdf/2607.28996v1)

**作者:** Sagar Lekhak `[一作]` (Rochester Institute of Technology), Emmett J. Ientilucci `[通讯]` (Rochester Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统审核与手工重标注，对现有SULAND RGB地面雷阱数据集进行改进，发布了更为完整、精确且一致的SULAND_v2版本；随后使用该数据集在IID与OOD两种分布下，对35种不同架构的目标检测模型进行统一训练与评估。

**💡 创新点**

创新点包括：①首次对SULAND数据集进行全面质量审计并纠正多种标注错误；②提出统一的重标注准则和严格的质量控制流程；③在单一数据集上完成多种检测框架（YOLO, RT‑DETR, D‑FINE, RF‑DETR, Faster‑R‑CNN, YOLO‑World）在分布外环境下的系统性比较，揭示模型架构与泛化能力之间的关联。

**🔧 技术方法**

采用了深度学习目标检测技术，主要包括YOLOv8/11/12/26系列、一阶段、两阶段、Transformer‑based、以及开源词汇检测模型YOLO‑World，使用COCO预训练权重进行微调，并采用统一的评估脚本实现跨框架的mAP、Precision、Recall、参数量及FPS测量。

**📊 数据集**

使用了SULAND_v2数据集，该数据集包含33,771张RGB图像、12,433个bounding‑box标注，保留原IID（意大利）与OOD（美国）分布划分，覆盖PFM‑1与PMA‑2两类地雷模型。

**📈 对比分析**

方法上通过相同训练超参、统一验证与测试流程，对各模型在IID和OOD测试集上计算mAP@50、mAP@50:95等指标。结果显示，YOLO系列在IID下取得最高mAP（YOLOv12‑S mAP@50≈90.8%），但在OOD下性能显著下降；RF‑DETR‑Large在OOD上表现最佳（mAP@50≈79.9%），且其mAP@50:95与IID几乎保持不变，说明其泛化能力强。整体来看，模型规模并非决定泛化的唯一因素。

**⚠️ 局限性**

局限性包括：①数据集仅涵盖两种地雷仿真模型，缺乏真实雷阱与更复杂的环境多样性；②标注重视可见目标，未覆盖埋藏或高度遮挡情况；③OOD仅为单一地理迁移，无法全面评估季节、光照、视角等多维域移。

---

## 173. Point2Radio: A Foundation Model for Cross-Scene Radio Fields from Material-Aware Point Clouds

**arXiv ID:** 2607.28994 | [PDF](https://arxiv.org/pdf/2607.28994v1)

**作者:** Chaozheng Wen `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于点云的跨场景无线电场预测基础模型 Point2Radio，能够快速生成三维路径增益和功率角度谱。

**💡 创新点**

创新点在于用层次化 token 化和 CSC 注意力学习可迁移的发射器-场景交互先验，且同一编码器可复用至多种无线量化任务。

**🔧 技术方法**

采用材料感知点云输入、Hierarchical Tokenization、Cross–Self–Cross (CSC) attention 以及局部 kNN 读取的解码头，辅以轻量级残差适配器。

**📊 数据集**

主要使用自研的 PRISM 数据集，包含 337 间室内房间的 86,272 条 TX‑条件路径增益字段以及 199 间带 PAS 标注的房间。

**📈 对比分析**

与 RadioUNet、3D U‑Net、NeRF^2、WRF‑GS+ 等基线比较，Point2Radio 在 35 个未见测试房间上实现 0.871 dB MAE（比 RadioUNet 减少 76.7%），SSIM 达 0.954，并在单 GPU 上实现 6‑7 倍的推理速度提升。

**⚠️ 局限性**

局限性包括对材料属性和法向信息的依赖、点云采样质量对精度影响、目前仅基于模拟数据且在真实环境中的泛化能力尚待验证。

---

## 174. CAER: Conflict-Aware Evidence Routing with Dual Prefix Experts for Multimodal Large Language Models

**arXiv ID:** 2607.28991 | [PDF](https://arxiv.org/pdf/2607.28991v1)

**作者:** Zixuan Liu `[一作]` (Zhejiang University), Jiajun Bu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CAER框架，结合span‑grounded evidence router与dual‑prefix expert routing，实现对视觉‑语言冲突的检测与冲突感知生成，且不更新模型主干参数。

**💡 创新点**

创新点在于：①利用soft span weighting把文本冲突片段与视觉证据关联，实现细粒度的冲突定位；②引入两种专门前缀专家，依据预测的冲突状态动态选择生成策略，保持主干不变。

**🔧 技术方法**

技术手段包括跨模态注意力、soft span weighting、对齐与检索的两阶段证据路由、双前缀专家的条件生成、两阶段训练策略（冲突检测 + 生成）。

**📊 数据集**

使用MMMC基准（视觉问题答复中包含冲突与非冲突对）以及自构建的AgriConflict农学视觉‑文本冲突数据集。

**📈 对比分析**

与原始模型、提示法（FoV、MMMC prompt）、LoRA微调、ASCD解码等方法对比，CAER在MMMCM上将冲突问题的CRA从约22%提升至81%，在AgriConflict的冲突检测准确率提升约6–10%，且参数量和训练成本远低于LoRA。

**⚠️ 局限性**

局限性包括：对冲突标注不准确时易误路由导致非冲突样本性能下降；仅覆盖二分类冲突/非冲突任务，对多义性或长文本的冲突识别仍有待提升。

---

## 175. Classification of COVID-19 cases from chest CT volumes using hybrid model of 3D CNN and 3D MLP-Mixer

**arXiv ID:** 2607.28978 | [PDF](https://arxiv.org/pdf/2607.28978v1)

**作者:** Masahiro Oda `[一作]` (Nagoya University), Kensaku Mori `[通讯]` (Nagoya University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了一种结合3D CNN和3D MLP‑Mixer的混合模型，用于对胸部CT体积进行COVID‑19高低可能性分类。

**💡 创新点**

创新点在于将MLP‑Mixer架构改造为3D版本并与3D CNN结合，既能提取局部特征又能利用全局特征，在医学影像样本有限的情况下提升分类性能。

**🔧 技术方法**

使用3D CNN提取局部特征，随后将特征分块为4×4×4体素并映射为1D向量，送入3D MLP‑Mixer进行跨体素特征混合和最终分类；预处理包括肺分割、裁剪、缩放至192×192×128，训练采用Adam优化器。

**📊 数据集**

使用日本多家医疗机构收集的1205个胸部CT体积数据集，包含COVID‑19与非COVID‑19病例，标注由放射科医生提供。

**📈 对比分析**

与单纯3D CNN+MLP的传统模型对比，混合模型取得79.5%的分类准确率，高于74.8%，证明3D MLP‑Mixer显著提升了性能。

**⚠️ 局限性**

局限性包括样本量有限、缺乏数据增强与跨国数据、模型在临床应用前仍需进一步提升准确率。

---

## 176. Visual Distribution Anchoring for Efficient Prompt Tuning

**arXiv ID:** 2607.28967 | [PDF](https://arxiv.org/pdf/2607.28967v1)

**作者:** Pouya Parsa `[一作]` (University of Minnesota), Seongjin Choi `[通讯]` (University of Minnesota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种无训练、仅利用目标域无标签图像的视觉分布锚定（VDA）框架，利用冻结的语义分类器和固定域模板对目标图像进行硬划分，提取每类的视觉原型并与语义分类器以全局权重融合，形成可缓存的目标适配器。

**💡 创新点**

创新点在于：① 通过域模板辅助的硬划分将目标图像映射到语义类别，避免了均衡分配导致的原型漂移；② 只需单次无监督前向传播即可估计视觉原型；③ 将视觉原型作为对语义分类器的补正而非替代，兼顾类别身份与目标域外观。

**🔧 技术方法**

核心技术包括：CLIP ViT‑B/16特征提取、TCP/Zero‑shot CLIP/​MaPLe/​PromptKD等语义分类器、固定域模板构造、硬 argmax 划分、Top‑K 支持选择、归一化平均得到原型、全局融合权重 λ=0.2。

**📊 数据集**

在十个 ImageNet → 目标数据集上评估：Caltech101、Oxford‑Pets、Stanford‑Cars、Flowers102、Food101、FGVC‑Aircraft、SUN397、DTD、EuroSAT、UCF101。

**📈 对比分析**

与同等无监督目标适配方法（ZLaP、InMaP）对比，VDA 在所有十个数据集上平均提升 3.39 以上，且在 9/10 数据集获得显著提升；对零射击 CLIP、源端提示调优 TCP、跨模态提示 MaPLe 以及泄漏无缝 PromptKD 均分别提升 3.22、3.39、3.35、2.79 分。

**⚠️ 局限性**

局限性包括：需要目标域无标签图像池；依赖手工域模板；原型估计限定于 CLIP 视觉空间；固定全局融合权重可能无法充分利用高质量原型；支持预算 K=32 对稀有类别无覆盖，导致部分类别无视觉改进。

---

## 177. MerchantBench: Benchmarking LLM Agents for Long-Term Coherence in E-Commerce Operations

**arXiv ID:** 2607.28956 | [PDF](https://arxiv.org/pdf/2607.28956v1)

**作者:** Qiming Shi `[一作]` (Zhejiang University), Chengfu Huo `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了 MerchantBench，构建了基于 98,843 条真实商品记录的 365 天订单级电商仿真环境，用于评估大语言模型在长期连贯决策中的表现。

**💡 创新点**

创新点在于引入长期连贯性（Long‑Term Coherence）评估维度，结合多时延的供应链事件和订单反馈，模拟真实电商的动态决策过程，并提供 26 种工具接口。

**🔧 技术方法**

利用 POMDP 框架、ReAct 与 Hermes 两种代理框架，结合 GPT‑5.6 Sol、Claude Opus 4.8、Qwen3.7‑Max/Plus、GLM‑5.2、DeepSeek‑V4‑Pro/Flash、Kimi K2.6 等大语言模型进行工具调用与决策，同时采用订单级仿真、实时供应商事件采样与延迟订单结果。

**📊 数据集**

基于 1688 平台的真实商品与供应商数据，包含 98,843 件商品的 365 天需求轨迹及每日市场报告。

**📈 对比分析**

对 8 种 LLM 在 ReAct 与 Hermes 两框架下共 48 次 365 天仿真进行比较，评估指标包括最终净资产、GMV、利润率、罚金、店铺评分等；结果显示最优配置（Hermes+Qwen3.7‑Max）仅获得人类基准的 27.3% 净资产，LLM 表现普遍低于人类，且波动幅度大。

**⚠️ 局限性**

局限性包括：①评估仅关注单一卖家端的直销模式，未覆盖库存管理和跨店运营；②仿真虽基于真实数据，但供应链事件与订单结果的概率模型简化；③LLM 代理的长期连贯性仍受工具调用频率与记忆压缩限制，导致策略衰退或误判。

---

## 178. Mind the Gap: Policy vs Reality in Post-Quantum TLS Deployment

**arXiv ID:** 2607.29005 | [PDF](https://arxiv.org/pdf/2607.29005v1)

**作者:** Nimesha Wickramasinghe `[一作]` (University of New South Wales), Arash Shaghaghi `[通讯]` (University of New South Wales)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 NIST 标准化后 PQ‑TLS 在公共互联网的部署进行三轮纵向测量，分析 1M 域名的 2 亿元握手，评估配置、运营商驱动、政策一致性、性能及安全共演。

**💡 创新点**

首次将 PQ‑Cryptography 的实际部署与国家政策路线图对齐，并揭示大多数部署集中于单一 hybrid 方案、主要由 CDN 负责、并在性能上几乎无负担。

**🔧 技术方法**

自研 TLS 1.3 探针支持 PQ 算法，基于 11 个全球视点进行主动扫描，结合域名分类与供应商归属推断，使用差分测量评估握手延迟与消息大小。

**📊 数据集**

DomCop Top 10M 域名列表作为基准，结合 Tranco/CrUX 对照，针对 1M 及 10K/1K 国家/政府域名样本进行测量。

**📈 对比分析**

对比 hybrid（ML‑KEM+X25519）与纯 classical 的握手时间和字节量，结果显示中位延迟差异 0 ms，消息大小增加约 1.1 KB，整体性能无明显劣势。

**⚠️ 局限性**

仅覆盖公开 HTTPS，使用云端视点，未测私人网络、非 Web 协议，并受限于域名排行漂移与部署者归属推断的不确定性。

---

## 179. MMShopBench: A Real-Log Benchmark for Multimodal, Multi-Turn Shopping Agents

**arXiv ID:** 2607.29002 | [PDF](https://arxiv.org/pdf/2607.29002v1)

**作者:** Zeying Hao `[一作]` (Alibaba Group), Xiaoyong Zhu `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MMShopBench，一套基于真实对话日志的多模态多轮购物评测基准，包含289条包含图像与文字的交互案例。

**💡 创新点**

创新点在于：①将真实多模态购物日志与冻结的100k产品沙盒结合；②引入基于证据的验证与选择（EGVS）流程；③提供可复现的多轮工具调用框架与多模态评价指标。

**🔧 技术方法**

使用技术包括：多模态语言模型、BM25文本检索、Marqo 视觉检索、区域条件图像检索、GPT-5.5 评判器、以及对齐的自我验证器。

**📊 数据集**

使用的数据集是 MMShopBench（289条手工标注的真实日志）以及由其生成的100k产品沙盒，另外用于训练的900条对话构成的 SFT 语料。

**📈 对比分析**

方法上对比了 Gemini‑3.1‑Pro‑Preview、Claude Opus 4.8 等闭源模型和 Qwen3.5‑9B/27B/122B‑A10B 等开源模型；通过 EGVS 及 SFT，开源模型在 Judge@3 从5.9% 提升到67.5%，与顶级闭源模型相差约5.9个百分点。

**⚠️ 局限性**

局限性包括：数据量相对有限（仅289条评测案例）、主要来自中文日志、对图像与文本依赖较强且可能不具备跨平台推广性；以及需要人工标注且无法完全覆盖所有真实需求。

---

## 180. MESS: Fast and Private Semantic Search on Multi-Graph HNSW

**arXiv ID:** 2607.28999 | [PDF](https://arxiv.org/pdf/2607.28999v1)

**作者:** Haoyu Cui `[一作]` (Shandong University), Mei Wang `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在云端实现语义检索的系统，兼顾数据、查询与访问模式的隐私，同时保持高召回和低延迟。

**💡 创新点**

通过结合差分隐私的LSH随机响应与多图HNSW索引，克服了单图噪声导致的准确性下降，并采用两阶段查询扰动保护查询模式。

**🔧 技术方法**

使用局部敏感哈希、随机响应、IsoHash、对称加密、HNSW多图索引以及两阶段随机扰动机制。

**📊 数据集**

在SIFT、LAION、TripClick、MS MARCO以及大规模SIFT100M上评估。

**📈 对比分析**

与Plaintext‑HNSW、Compass、HE‑Cluster和No‑Noise基线对比，获得高达15.08倍的延迟降低、35.28倍的通信降低，召回差距低于5%。

**⚠️ 局限性**

对单服务器多图的交叉链接攻击仍有一定风险，且查询侧扰动导致的通信开销随候选集扩大而显著增加。

---

## 181. PARALLEL: A Prefrontal-Aligned Reinforcement inspired Approach for Language-Model Learning under Explicit Limits

**arXiv ID:** 2607.28982 | [PDF](https://arxiv.org/pdf/2607.28982v1)

**作者:** Namkyung Yoon `[一作]` (Korea University), Hwangnam Kim `[通讯]` (Korea University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于前额叶对齐的强化学习控制器 PARALLEL，用于在流式适配中对 LoRA 参数进行样本级更新强度分配。

**💡 创新点**

创新点在于同时利用目标相关与不确定性相关信号构建控制器，并通过即时效用–成本反馈与累计更新质量约束实现自适应更新强度，而非传统全量或数据选择方法。

**🔧 技术方法**

采用 LoRA 低秩参数化、REINFORCE（一次梯度）训练控制器、目标相关信号（损失、准确率）、不确定性信号（熵、置信度、方差）以及预算约束机制。

**📊 数据集**

在 ARC‑Challenge、OpenBookQA、CommonsenseQA（推理）以及 XSum、CNN/DM（摘要）五个公开基准上进行实验。

**📈 对比分析**

与冻结、全量 LoRA、随机抽样、主动学习等方法比较，PARALLEL 在保持 94.1–99.2% 完整适配性能的同时，恢复约 85–99% 相对增益；在相同累计适配时间或 GPU 能耗下，PARALLEL 的准确率高于全量 LoRA。

**⚠️ 局限性**

局限性包括仅在小型或中型模型上验证，未针对大模型、持续学习或生成任务扩展，且需进一步评估长期稳定性与硬件成本。

---

## 182. Mixture-of-Translators: Translating KV Caches Across Heterogeneous Large Language Models

**arXiv ID:** 2607.28979 | [PDF](https://arxiv.org/pdf/2607.28979v1)

**作者:** Jin-woo Lee `[一作]` (Chungnam National University), Sungsu Lim `[通讯]` (Chungnam National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种名为MoT的混合翻译框架，用于在异构LLM之间转换KV缓存，实现缓存重用与跨模型协同。

**💡 创新点**

创新点在于提出多翻译器混合和上下文校正损失，理论上解决传播误差与校正缺陷两大错误，从而实现高质量、可扩展的跨模型缓存迁移。

**🔧 技术方法**

采用Transformer残差分析的传播/校正误差理论、token级路由的多翻译器模型以及对齐目标轨迹的上下文校正损失，并结合多代理推理与长上下文缓存生成的实验案例。

**📊 数据集**

评估数据集包括闭集QA（BoolQ、PubMedQA、MMLU-Redux）、提取式QA（SQuAD v1.1、NewsQA）以及多代理推理（Doc2Dial）和长上下文生成（HotpotQA-E）等。

**📈 对比分析**

与原生、单翻译器、投影式映射、隐藏状态通信、共享潜在空间等多种基线比较，MoT在异构和同构翻译场景下平均QA准确率≈51%、提取式F1≈0.43，生成质量保持≈96.3%，并显著降低多代理峰值内存。

**⚠️ 局限性**

局限性包括额外的峰值内存和训练成本，翻译模块对通道设计的依赖，以及在极端异构或极大规模模型中可能出现梯度不稳定或缓存尺寸不匹配的问题。

---

## 183. LegoQ: Density-Matrix Representation Learning with Spectral-Spatial State Transitions for Hyperspectral Classification

**arXiv ID:** 2607.28970 | [PDF](https://arxiv.org/pdf/2607.28970v1)

**作者:** Weijia Cao `[一作]`, Xiang Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于密度矩阵的恒定状态学习框架（LegoQ）用于高光谱图像分类，保持状态在整个网络中的合法性并提供可解释的诊断指标。

**💡 创新点**

创新点在于：①使用密度矩阵而非传统向量作为特征表示，并在训练中始终保持其正半定、厄米性和单位迹；②设计可组合的谱-空间-耦合状态转移堆栈；③采用Uhlmann相似度对类原型矩阵进行匹配，取代传统softmax。

**🔧 技术方法**

技术包括：密度矩阵编码、谱/空间/耦合转移、欧氏谱投影、Uhlmann相似度、纯度与熵诊断以及基于分组的光谱分块。

**📊 数据集**

使用了Indian Pines（16类、200波段）和WHU-Hi-LongKou（9类、270波段）两个高光谱数据集进行评估。

**📈 对比分析**

与传统基于向量的深度分类器比较，LegoQ在Indian Pines上平均准确率达96.20%，在WHU-Hi-LongKou上最优运行达到97.52%，在整体准确率、kappa系数和宏F1上均表现优异。

**⚠️ 局限性**

局限性包括：固定的状态维度和光谱分组可能限制模型容量；耦合缩减未强制为物理可测量的量子通道；缺乏针对边缘像素的熵分析和更广泛的跨传感器/城市场景验证。

---

## 184. Retrieval-Driven Training-Free AI-Generated Video Attribution

**arXiv ID:** 2607.28955 | [PDF](https://arxiv.org/pdf/2607.28955v1)

**作者:** Renxi Cheng `[一作]` (Southeast University), Hongsong Wang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个无训练的基于检索的视频来源归因框架，对AI生成视频进行归因和检测。

**💡 创新点**

将归因任务转化为实例检索，使用自适应正交色彩变换、跨尺度量化残差和时序语义聚合等无监督特征提取方法，无需模型训练即可识别多种生成器。

**🔧 技术方法**

自适应正交色彩变换（AOCT）、多尺度量化残差（MSQR）、时序残差流与RGB上下文流融合以及预训练的R3D‑18编码器。

**📊 数据集**

在GenVidBench基准上进行评估，该基准包含八种主流生成模型的视频和真实视频。

**📈 对比分析**

与多种图像/视频检测与归因基线（ResNet‑50、R3D‑18、ESSP、LOTA、PiD、DeMamba、UNITE）对比，在100‑shot情形下取得Rank‑1 84.6%、mAP 78.3%，优于其它方法20.5%/16.6%的提升。

**⚠️ 局限性**

对极端压缩或裁剪的鲁棒性仍有限，需要手动调参并且对未见过的生成器只能通过再注册样本来处理。

---

## 185. Overcoming the Weakest-Link Effect in LLM-Driven Program Optimization via Heterogeneous Edit Recombination

**arXiv ID:** 2607.28947 | [PDF](https://arxiv.org/pdf/2607.28947v1)

**作者:** Jingwen Fu `[一作]` (Zhongguancun Academy), Nanning Zheng `[通讯]` (Xi'an Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的程序优化器 HERO，通过仅利用任务描述和当前程序生成一组互不重叠的原子编辑，然后在无性能推理的情况下对所有可能的编辑组合进行评估，挑选最优子集进行程序更新，实现无梯度（zeroth‑order）优化。

**💡 创新点**

创新点在于：①将编辑生成与编辑选择分离，避免传统“一次性全包”导致的弱链（weakest‑link）问题；②在不使用任何性能反馈的前提下利用 LLM 的隐含知识生成高质量编辑；③通过子集评估实现“最优子集重组合”，在保持无性能推理的同时实现有效的方向累积。

**🔧 技术方法**

技术方法包括：①任务描述+当前程序的零阶提示让 LLM 产生非重叠、可交换的原子编辑；②使用评估器对每个子集（或在可行范围内的子集池）进行分数评估，挑选最优子集；③采用基于子集组合的选择策略（类似局部增益最大化），消除弱链影响；④通过减少 LLM 调用次数和 token 消耗实现高效迭代。

**📊 数据集**

使用的数据集与任务包括：①圆形打包（Circle Packing）——在单位正方形内放置 26 个圆以最大化半径之和；②策略游戏（Othello、Battleship）——分别为完全确定性和部分可观测的两人博弈；③ LLM 代理系统设计（AIME 2025/2024/2023）——优化代理程序结构以提升解题准确率；④ 路径规划（Grid‑based robot path planning）——在障碍网格中寻找最短无碰撞路径。

**📈 对比分析**

与 ShinkaEvolve、OpenEvolve、Best‑of‑N 等传统 LLM 优化器以及经典算法（A*、PPO）进行对比。HERO 在所有任务中表现出更快的收敛速度、较低的 token 及 LLM 调用量，并在难度更高的任务（Othello‑H、Battleship‑H、路径规划难例）中显著优于基线。实验表明：①在圆形打包中收敛步数大幅减少、token 下降；②策略游戏中赢率提升 0.3‑0.5；③代理设计中准确率提升至 28.7%（相对 18.5% 的基线提升 10%）；④路径规划中搜索复杂度（访问状态数）降至约 40% 左右。

**⚠️ 局限性**

局限性包括：①编辑子集评估需要大量评估器调用，若评估成本高会限制适用范围；②编辑生成要求可交换且不重叠，限制了对某些程序结构或并行修改的适用性；③对非常简单或梯度可恢复的问题，HERO 的优势不明显；④虽然避免了性能推理，但仍需评估器对最终程序进行全局评分，无法完全不依赖外部评估。

---

## 186. A Biometric Sensor Network to Enable Real-Time Measurement of Individual Student Engagement in STEM Lecture Environments

**arXiv ID:** 2607.28944 | [PDF](https://arxiv.org/pdf/2607.28944v1)

**作者:** Ahmed Elsayed `[一作]` `[通讯]`, Ahmed Elsayed

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了一套基于摄像头的生物传感器网络（BSN），能够在 STEM 课堂中实时、非侵入式、非可识别地测量并跟踪每位学生的行为、情感与认知参与度。

**💡 创新点**

核心创新点包括：①将学生置于可移动的边缘设备（SPU）中实现自我标识与本地推理，避免数据泄露与被标记的风险；②将数据采集与实时推理拆分为“数据采集模式”与“分析模式”，确保仅在分析模式下进行加密传输；③通过硬件与软件双重加固（硬件加密存储、TLS 1.3、证书授权、无 SSH 接口）满足 IRB 与 FERPA 合规；④将实时参与度指标（面部表情、凝视、身体动作）映射到多维参与模型，实现细粒度、10 秒窗口的即时反馈。

**🔧 技术方法**

技术实现涵盖：高分辨率摄像头（OAK‑D Lite）+ Raspberry Pi 5；基于 OpenCV/DepthAI 的面部检测、凝视估计、表情识别模型；边缘推理框架（TensorRT / ONNX Runtime）与量化模型；加密存储（AES‑256）与混合对称/非对称加密；NTP 同步、TLS 1.3 mTLS；服务器端 Django + MinIO + NGINX + NTP；前端基于 PySide6 的触摸交互；功耗管理与双 18650 电池；自动化 OTA 更新与安全启动。

**📊 数据集**

使用自建的私有 SE 数据集：在 CFU 采集的课堂视频（多摄像头同步录制）被人工标注为行为、情感、认知三维参与标签，覆盖 31 小时左右的多名学生；该数据用于训练与验证面部/凝视/行为模型，并与先前 CVIP 基线系统进行对比。

**📈 对比分析**

评价方法：①硬件层面——数据带宽、时钟同步误差、推理延迟、两小时续航；②软件层面——推理准确率（情感 72%+，行为 80%+），与基线系统相比，安全性提升 3 倍、自动化率 100%、数据可靠性提升 2 倍；③实地课堂部署实验，收集 10 个课堂的实时参与度分布，验证系统可实时反馈给教师。

**⚠️ 局限性**

局限性：①动态拓扑成本随课堂人数线性增长，规模化成本高；②仅支持面部与凝视、身体动作，无法捕捉 EEG、HRV 等生理信号；③对高光、遮挡或学生转身等情况的鲁棒性仍有限；④对隐私合规高度依赖 IRB 许可，部署时需严格遵循伦理审批；⑤目前仅在实验室/小型课堂验证，尚未大规模跨学科推广。

---

## 187. Reproducing LightMem: Naive RAG Is Just as Good for Memory Management

**arXiv ID:** 2607.29104 | [PDF](https://arxiv.org/pdf/2607.29104v1)

**作者:** Yongjie Zhou `[一作]` (University of Queensland), Guido Zuccon `[通讯]` (University of Queensland & Google)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

复现 LightMem 并与 Naive RAG 进行对比，评估检索器选择、检索深度、回答 token 预算等对性能的影响，并通过 oracle 条件区分检索误差与构造信息损失。

**💡 创新点**

揭示 LightMem 的优势是检索器和 token 预算的条件函数，而非普适优势；通过大规模检索器实验和 oracle 评估，证实构造过程可能丢失关键信息，提供了对内存构造与检索协同作用的细致洞察。

**🔧 技术方法**

使用 LightMem 的压缩、聚类和 LLM 摘要构造 pipeline；多种检索器（BM25、SPLADE、PromptReps、Qwen3-Embedding 等）和 RAG；LLM 生成与评估；vLLM 推理；LongMemEval-S benchmark。

**📊 数据集**

LongMemEval-S 500 问题（除 56 个单会话助手问答外）共 444 个，涵盖六类对话与偏好、知识更新、时间推理等。

**📈 对比分析**

在相同检索器、检索深度或回答 token 预算下比较 LightMem 与 Naive RAG 的答案准确率；发现检索器对 LightMem 影响显著，强检索器可达 75.5% 准确率；在匹配检索深度时 Naive RAG 通常优于 LightMem；在匹配 token 预算时 LightMem 在低预算下略胜；但构造过程导致 11.3% 的信息损失。

**⚠️ 局限性**

构造过程开销大、信息丢失、对检索器高度依赖、在大 token 预算或强检索器场景下不一定优于 Naive RAG，且整体提升不显著；缺乏对检索器与构造交互的深入分析。

---

## 188. StraightDP: Geometry-Aware Differential Privacy for Rectified-Flow Transformers

**arXiv ID:** 2607.29100 | [PDF](https://arxiv.org/pdf/2607.29100v1)

**作者:** Xujun Che `[一作]` (University of North Carolina at Charlotte), Xintao Wu `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过一次性发布类别条件均值与协方差，并在其余训练阶段使用预先声明的 DP‑SGD 进行不同ially private 训练，构建一种名为 StraightDP 的文本条件生成模型训练方法。

**💡 创新点**

核心创新包括：① 发现 rectified flow 在时间维度上学习信号非均匀，早期阶段可由低阶矩充分表征；② 采用一次性发布低阶矩并将其注入模型权重或采样时的引导，显著提升强隐私下的实用性；③ 在预训练阶段对多模态扩散变换器做流范数截断（stream‑norm clamp），在高噪声 regime 下聚焦梯度噪声，提升 DP 训练效果。

**🔧 技术方法**

使用的主要技术包括：rectified flow 训练、Gaussian 机制发布类别条件矩、动态时间异质分配 (THA) 的 DP‑SGD、流范数截断的多模态扩散 transformer、分类器无关引导、公共域预训练与私有域微调、精确的 PLD 预算审计和攻击审计。

**📊 数据集**

实验主要基于 MNIST 作为私有数据集，Kuzushiji‑MNIST 作为公共域预训练；另外在潜在空间使用冻结的公共自编码器；对规模更大模型进行实验时使用 SD3‑medium 与 Flowers‑102 作为私有数据集。

**📈 对比分析**

与现有 DP 生成模型（如 DP‑MERF、DP‑HP、DP‑NTK、Private Evolution）在同一预算下进行比较。StraightDP 在强隐私（ε≈10⁻⁵）下，利用一次性发布矩阵的方式可实现 0.78+ 的下游分类准确率，FID 下降至 56；相比之下传统 DP‑SGD 仅得到 0.21 的准确率。实验表明，发布矩阵的优势随隐私强度提升显著（在 MNIST 领域可提升 3–4 倍），在潜在空间亦表现出 1.5‑倍的准确率提升。

**⚠️ 局限性**

局限性包括：① 对极大规模预训练模型（如 SD3‑medium）一次性发布矩阵的效果仅在采样端显著，重量注入（distillation）表现差；② 该方法在不同的任务或数据分布下需要手工调优发布比例和时间截断阈值；③ 仅利用低阶矩可能无法捕捉复杂多模态关系，导致在更高维或更具细节的生成任务中性能受限。

---

## 189. PiDDM: Physics-Informed Differentiable Degradation Modeling for Lithium-Ion Battery State-of-Health Prediction

**arXiv ID:** 2607.29095 | [PDF](https://arxiv.org/pdf/2607.29095v1)

**作者:** Zeping Chen `[一作]` (University of Notre Dame), Tengfei Luo `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 PiDDM 物理信息可微降解模型，用于在多种循环条件下预测锂离子电池的状态健康 (SOH)。

**💡 创新点**

创新点在于将基于 Arrhenius‑幂律的 SEI 成长与 LLI 失效规律嵌入神经网络训练目标，形成可微化的物理正则化，既保持了数据驱动的灵活性，又保证了降解轨迹的物理一致性。

**🔧 技术方法**

技术手段包括全连接深度网络、可微降解模块、前向欧拉积分、Adam 优化及温度/循环参数的幂律与指数结合。

**📊 数据集**

使用了公开的 55 台 NCM 电池在六种充放电协议（2C、3C、R2.5、R3、随机步进、卫星轨道）下的循环失效数据集。

**📈 对比分析**

与传统 MLP 与基线 PINN 在标准预测和 90% 训练-10% 外推两种设置下进行对比，PiDDM 在所有协议上平均误差最低（MAPE、RMSE、MAE、MSE 均优），且外推时轨迹更平滑、物理一致。

**⚠️ 局限性**

局限在于依赖经验降解方程，难以捕捉局部容量回升噪声，且仅考虑 SEI 与 LLI，未覆盖其他失效机制和时变温度、内阻等因素。

---

## 190. Rethinking AI Cloud Infrastructure for Agentic Serving Systems with the Aries Experimentation Framework

**arXiv ID:** 2607.29069 | [PDF](https://arxiv.org/pdf/2607.29069v1)

**作者:** Leonid Kondrashov `[一作]` (NTU Singapore), Dmitrii Ustiugov `[通讯]` (NTU Singapore)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个全栈实验框架（Aries），能够将任务语义与执行配置分离、统一工具沙盒接口、重构跨组件轨迹，并结合可复现实验与生产追踪，对大规模自治代理的端到端执行进行系统性评估。

**💡 创新点**

创新点在于：① 轨迹级遥测与统一事件架构；② 任务与执行规范的解耦，支持跨栈可比性；③ 对工具沙盒的状态化、可插拔执行与监控；④ 量化上下文保留、工具沙盒弹性及攻击面，提出面向轨迹的指标与控制面；⑤ 通过生产数据验证低级系统瓶颈与任务性能之间的关联。

**🔧 技术方法**

技术手段包括：OpenHands、Hermes Agent、OpenClaw harness；LLM 后端 Qwen3.6‑35B‑A3B‑FP8 与 SGLang；AWS Lambda MicroVMs、Google Agent Sandbox 等沙盒；trajectory schema 与 telemetry collector；snapshot C/R 成本模拟；KV‑cache 与内存压缩策略。

**📊 数据集**

使用的数据集有：SWE‑Bench Pro、Terminal‑Bench 2、DeepResearch Bench，各采样 20 个任务；并采集了 8 小时的生产轨迹（10 个 LLM 服务器、100 个工具沙盒），用于验证实验结果。

**📈 对比分析**

比较方法：在相同硬件（96‑core CPU + 1 × H100）下，对多种 harness 与工具组合进行可复现实验，记录每一步的延迟、资源占用、上下文长度和成功率；将实验结果与生产追踪对齐；对 snapshot‑C/R 与持续沙盒的成本进行建模。性能发现：① 工具调用占总延迟 13–48%；② 上下文保留超过工作量特定阈值后准确率提升停滞，KV‑cache 资源导致并发容量下降；③ snapshot‑C/R 机制在现有云计费下导致成本高于持续沙盒，可通过弹性管理将成本降低约 3 倍。

**⚠️ 局限性**

局限性：① 仅在单 GPU 节点、单模型 Qwen3.6 上验证；② 侧重于 CPU‑GPU 混合工作负载，未覆盖多 GPU 或大规模集群；③ 工具沙盒与宿主环境种类有限，未测试所有可能的安全漏洞；④ 安全分析基于 CVE 统计，缺乏动态攻击实验；⑤ 轨迹级遥测框架仍需进一步自动化以支持大规模在线服务。

---

## 191. Tokenizer-Agnostic Engram Module

**arXiv ID:** 2607.29065 | [PDF](https://arxiv.org/pdf/2607.29065v1)

**作者:** Jia Peng Lim `[一作]` (Singapore Management University), Hai Leong Chieu `[通讯]` (DSO National Laboratories)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究如何让 DeepSeek 的 Engram 模块在不同 tokenizer 之间共享参数，并提出基于多项式哈希的字节级哈希方法实现 tokenizer-agnostic Engram。

**💡 创新点**

创新点在于将原本依赖 token 索引的 XOR 哈希改为字节级多项式哈希，去掉 N 级嵌入空间划分并加入 1-gram 共享表，从而实现不同 tokenizer 的字节等价 N-gram 对齐。

**🔧 技术方法**

使用了多项式哈希、共享嵌入空间、上下文感知 SPDA 门控、矩阵化缓存计算以及多种 decoder-only transformer 结构（如 LLaMA、GQA、Qwen3.5）。

**📊 数据集**

实验在大规模通用语料（例如 EleutherAI 32B 语料、Wikitext val、以及 100B-150B 的预训练数据）和多任务评测基准（ARC、BoolQ、COPA、HellaSwag、LAMBADA、PIQA、SCIQ、Winogrande）上进行。

**📈 对比分析**

通过对比开启/关闭 Engram、XOR 哈希 vs 多项式哈希以及跨 tokenizer 迁移实验，使用准确率、长度归一化准确率和 bits/byte 指标，结果表明多项式哈希与 XOR 性能相近且在跨 tokenizer 迁移中提升 5–10% 的指标。

**⚠️ 局限性**

局限在于需要大规模缓存与哈希表、存在哈希冲突风险、对多语言支持不足以及对不同 tokenizer 之间的字节等价覆盖率仍有限。

---

## 192. On the Generalization of Steering Vectors for Chain-of-Thought Faithfulness

**arXiv ID:** 2607.29062 | [PDF](https://arxiv.org/pdf/2607.29062v1)

**作者:** Matthew Nguyen `[一作]` (University of Virginia), Iván Arcuschin `[通讯]` (Poseidon Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过激活向量激励技术（activation steering）在不同模型、数据集和提示类型下实验，评估其对链式推理（CoT）可解释性（即提示承认度）的提升效果。

**💡 创新点**

创新点在于系统地探究激励向量在提示类型、数据集和构造方法上的泛化能力，发现效果主要由评估设置决定，而非训练场景；同时证明激励可在不增加提示使用率的前提下降低隐藏提示使用。

**🔧 技术方法**

使用残差流激励技术，在每个生成步骤加入标准化向量；构造四类向量（对比、合成、特定优化、通用优化）；利用LLM评判器判断CoT中是否提及提示，并用Δ_ack量化效果。

**📊 数据集**

实验基于BBH、GPQA和MMLU三大多选题数据集；在GPQA上使用四种提示（Stanford、XML、grader-hacking、unethical insider tip），其他数据集仅使用Stanford提示。

**📈 对比分析**

通过比较基线与激励生成的提示中提示承认率的差值Δ_ack评估性能；仅在Gemma‑3 12B模型中观察到显著提升（MMLU上+0.26，BBH +0.10，GPQA +0.07），其他模型效果不显著；不同向量构造方法效果相近，且跨提示/数据集转移表现良好。

**⚠️ 局限性**

局限性包括：层选择基于测试AUROC，易过拟合且与场景混杂；评判器单一，缺乏多评判者验证；α值仅在少数取值，未绘制完整曲线；提示类型仅在GPQA上多样，数据集组合有限，限制了泛化评估范围。

---

## 193. Learning from Adversity: Semantic-Aware Mask Refinement through Adversarial Perturbation

**arXiv ID:** 2607.29059 | [PDF](https://arxiv.org/pdf/2607.29059v1)

**作者:** Beomyoung Kim `[一作]` (NAVER Cloud), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于对抗扰动和三向对比学习的语义感知掩码细化框架 Phoenix，能够在不重新训练主模型的情况下提升分割掩码的边界精度和结构完整性。

**💡 创新点**

创新点包括：① 对抗掩码扰动（AMP）通过在嵌入空间进行 FGSM 攻击，生成与真实分割错误相似的语义感知噪声；② 三向对比掩码细化学习（CMRL），通过在真、噪、精细三种掩码之间构造三向对比关系，实现前景后景一致性、类间对抗性以及自我提升正则化；③ 将上述两项结合，实现高效、可控的细化训练。

**🔧 技术方法**

采用的技术包括：SAM（ViT‑H）预训练模型、嵌入空间对抗攻击、三向对比损失（InfoNCE、交叉熵等）、轻量化解码器微调、Dice 与 Focal 组合的分割损失。

**📊 数据集**

使用的数据集：LVIS（训练）、COCO（验证与无监督伪标签评估）、DIS5K 与 ThinObject‑5K（细粒度分割任务）以及在不同网络（Mask R‑CNN、SegRefiner 等）产生的粗掩码进行细化。

**📈 对比分析**

与 SegFix、SegRefiner、SAMRefiner 等传统细化方法对比，在 COCO、DIS 等基准上均取得显著提升；例如在弱监督 1% 标注下，AP^mask 提升至 +16.1%，AP^boundary +17.3%；在 DIS 任务中 IoU 提升 11%~21%，表明 Phoenix 在多种噪声模式下具有更好的鲁棒性和泛化能力。

**⚠️ 局限性**

局限性：① 仍需依赖预训练的 SAM 及其视觉提示；② 对抗噪声的生成与超参数（阈值 τ、步长 α 等）密切相关，需要经验调优；③ 目前仅针对像素级掩码细化，尚未探索跨模态或自监督细化的更广泛场景；④ 训练时仍需一定量的真实标签来构建三向对比关系。

---

## 194. Parameter-Efficient Fine-Tuning for Spiking Point Cloud Models

**arXiv ID:** 2607.29048 | [PDF](https://arxiv.org/pdf/2607.29048v1)

**作者:** Zihao Guo `[一作]` (Xi'an Jiaotong University), Danwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出SpikePEFT参数高效微调框架，利用预训练脉冲点云模型仅更新少量参数即可完成下游任务，同时保持低能耗；

**💡 创新点**

创新点在于设计Intrinsic Dynamics Tuner（IDT）调节膜衰减与阈值，并引入Silent-State Disambiguation Adapter（SSDA）恢复近阈静默状态信息，从而在冻结预训练权重的前提下显著提升性能；

**🔧 技术方法**

采用LIF神经元、二值化脉冲传播、残差调节、可学习阈值区间、事件驱动瓶颈网络与稀疏加权累积等技术；

**📊 数据集**

使用ModelNet40、ScanObjectNN（OBJ_BG、OBJ_ONLY、PB_T50_RS）、ShapeNetPart和S3DIS等数据集进行实验；

**📈 对比分析**

与全微调、线性探针、ANN‑based PEFT以及其他SNN方法对比，SpikePEFT在ModelNet40上达到92.4%准确率，仅增0.1%；在ScanObjectNN PB_T50_RS提升1.4%至85.6%；参数量仅约5%，能耗提升约14.8%但远低于ANN；

**⚠️ 局限性**

局限性包括对时间步数的敏感性、对不同SNN架构的泛化验证不足、在某些任务中提升有限，以及目前仅针对点云分类/分割，未扩展到更复杂任务。

---

## 195. DFSC: Error-Controlled Differentiable Mittag-Leffler Propagation for Fractional Scientific Machine Learning

**arXiv ID:** 2607.29038 | [PDF](https://arxiv.org/pdf/2607.29038v1)

**作者:** Ning Hu `[一作]` (Hangzhou Dianzi University), Chuyang Hu `[通讯]` (University of California, Santa Cruz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个名为DFSC的PyTorch框架，将已知的Mittag–Leffler谱传播嵌入可微分层，并允许训练分数阶、残差网络等未知项。

**💡 创新点**

创新点在于把解析的Mittag–Leffler传播直接作为可训练的算子，自动适配数值截断或Lanczos维度，并给出可靠性与误差上界，提供可训练分数阶和残差网络的混合学习方案。

**🔧 技术方法**

使用了可微分Mittag–Leffler函数评估、Lanczos/Krylov迭代、分数阶梯度计算、自动误差控制、残差神经网络、GPU矩阵函数、以及自动微分等技术。

**📊 数据集**

实验使用合成制造数据、12个真实实验数据集（AnomDiffDB H-actin、geomembrane、GeoTES热能存储、heated-steam注入等），并与外部基准（FDEint、pycaputo）进行对比。

**📈 对比分析**

通过与传统分数微分求解器和纯神经网络（FNO、DeepONet）对比，DFSC在已知传播场景下单点查询快10–20倍、误差≤1e‑15；混合模型在样本效率上明显优于纯神经网络，恢复分数阶参数精度在噪声下仅误差<1e‑3。

**⚠️ 局限性**

局限性包括：仅适用于已知Mittag–Leffler谱的线性系统，缺乏全局误差证明，分布式/变阶算子和复杂几何/非线性步骤支持有限，实际实验中结构不匹配时优势不明显。

---

## 196. SAM+D: Parameter-Efficient Dimensional Lifting of SAM-Family Models via Depth-Routed LoRA and Depth Shifting

**arXiv ID:** 2607.29033 | [PDF](https://arxiv.org/pdf/2607.29033v1)

**作者:** Yu Song `[一作]` (Ritsumeikan University), Yen-wei Chen `[通讯]` (Ritsumeikan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过在SAM与SAM2的Transformer块中插入Depth‑Routed LoRA和Depth Shift Module两种轻量模块，将2D/2D+T模型升维至3D/3D+T，实现参数高效的多维分割与追踪。

**💡 创新点**

提出基于深度位置路由的LoRA专家与零参数跨切片特征交换的DSM，且仅调节约2.8%–3.7%的权重即可完成3D与4D任务。

**🔧 技术方法**

采用参数高效微调（LoRA、Mixture‑of‑Experts）、深度路由、Temporal Shift Module改造、Transformer encoder改造、Conv3D/MLA/SAM prompt解码器、混合精度训练等技术。

**📊 数据集**

在医学CT四个肿瘤分割基准（KiTS、Pancreas、LiTS、Colon）以及细胞追踪挑战CTC的Fluo‑N3DH‑SIM+数据集上进行评估。

**📈 对比分析**

与全监督3D模型、交互式单点提示方法及现有SAM适配器比较，SAM+D在四个CT基准中取得近state‑of‑the‑art的Dice/NSD分数；在CTC追踪任务中实现最高SEG/DET/TRA分数并提升整体OP_CTB。

**⚠️ 局限性**

未评估跨模态SAM3；3D模型仅沿单一深度轴处理，未做多轴平均；4D推理仍需手动提示，完全自动追踪受限；4D训练耗时高。

---

## 197. Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving

**arXiv ID:** 2607.29031 | [PDF](https://arxiv.org/pdf/2607.29031v1)

**作者:** Jiwei Yang `[一作]` (Tsinghua University), Jun Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Auto‑JEPA，一种通过联合嵌入预测未来驾驶意图的行动导向潜在世界模型；

**💡 创新点**

创新点在于仅预测与未来 ego 行为相关的连续驾驶意图（而非全景未来状态），并直接用意图检索执行轨迹，避免密集未来场景重建；

**🔧 技术方法**

核心技术包括冻结的 V‑JEPA 视觉编码器、轨迹自编码器生成意图空间、Transformer 预测器进行联合嵌入预测、非参数轨迹检索、场景条件评分与可行性门控；

**📊 数据集**

使用 NAVSIM v1 与 v2 两个大型仿真评测数据集；

**📈 对比分析**

在 NAVSIM v1 上获得 91.3 PDMS，NAVSIM v2 上获得 89.1 EPDMS，性能与基于学习生成轨迹的 CLOVER 相比保持竞争力，同时不需要额外的轨迹生成网络；

**⚠️ 局限性**

局限性包括对轨迹记忆库覆盖率的依赖、候选选择校准不足，以及缺乏可学习的轨迹生成或细化模块。

---

## 198. Evaluation-Verification Reward for Consistent Multi-Reference Image Editing

**arXiv ID:** 2607.29025 | [PDF](https://arxiv.org/pdf/2607.29025v1)

**作者:** Yingmao Miao `[一作]` (Xi'an Jiaotong University), Chenhao Lin `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于强化学习的多参考图像编辑框架，利用 Evaluation‑Verification Reward (EVR) 机制来抑制多模态大语言模型的幻觉，并为 DiffusionNFT 提供稳定的奖励信号。

**💡 创新点**

创新点在于：①将评估拆分为五个维度的短 CoT 评估与多条理由的生成；②引入视觉验证器对每条理由进行图像证据核验，消除文本偏差；③设计可扩展的多参考编辑数据管线，只需语义一致的参考与指令即可训练，避免昂贵的真实目标图像。

**🔧 技术方法**

核心技术包括 Qwen‑Image‑Edit (DiT + Flow Matching) 作为生成模型；DiffusionNFT 强化学习框架；MLLM (Qwen3‑VL‑8B/32B) 作为评估器和验证器；LoRA 微调与 AdamW 优化；多维度评分与几何平均奖励聚合。

**📊 数据集**

使用了 10,000 条合成的 (r_obj, r_scene, 指令) 三元组作为训练集，1,000 条为测试集，并收集了 300+ 真实网络多参考编辑样例用于 OOD 评估；此外在实验中还使用公开的单图像编辑基线数据进行对比。

**📈 对比分析**

在与基线 Qwen‑Image‑Edit、Edit‑R1 等奖励方案以及未验证的 EVR 进行对比时，实验显示在 Reference Consistency、Harmony、Instruction Consistency 等五个维度上均有显著提升；人类偏好胜率提升至 67%，在 OOD 及 N>2 场景下仍保持优于现有闭源与开源方法的表现。

**⚠️ 局限性**

局限性包括：①训练仅在 N=2 的合成数据上，泛化至更大参考数仍不确定；②奖励信号仍受 MLLM 视觉理解能力限制，细粒度细节可能被遗漏；③长 CoT 与验证过程增加推理开销；④缺乏真实目标图像监督，模型在极其复杂或极端场景下仍可能出现未检测到的错误。

---

## 199. Federated Foundation Models Fine-Tuning with Heterogeneous Compressed Clients

**arXiv ID:** 2607.29071 | [PDF](https://arxiv.org/pdf/2607.29071v1)

**作者:** Shengkun Zhu `[一作]` (Hong Kong Polytechnic University), Yang Liu `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FedSLM框架，允许拥有不同压缩比例的客户端在联邦学习中协同微调大规模基础模型；

**💡 创新点**

创新点在于：1）利用SVD分解生成自包含的低秩客户端模型；2）设计两阶段聚合协议，先在同压缩组内聚合adapter，再跨组重构全秩并融合；3）引入弱→强知识提炼与辅助置信度损失，减少压缩误差；4）给出理论收敛、子空间对齐与误差抑制的证明；

**🔧 技术方法**

使用SVD（及其变体DobiSVD、QSVD）、LoRA、FedAvg、软标签知识蒸馏、辅助置信度损失；

**📊 数据集**

实验数据集包括自然语言任务（ARC、COPA、HellaSwag、PIQA、Social IQA、WinoGrande、PubMedQA、MedMCQA）和多模态任务（ScienceQA、VizWiz），在LLaMA-2-7B/13B及LLaVA-NeXT-7B上评估；

**📈 对比分析**

与多种基线（FedAvg+LoRA、FFA-LoRA、HetLoRA、FlexLoRA、Fed-RAC-LoRA、FedMKT、FedProto、FedBiOT等）比较，FedSLM在IID和非IID场景下均取得最高平均准确率，且客户端压缩模型的显存仅为全模型的约50%；

**⚠️ 局限性**

局限性包括：1）目前仅支持SVD压缩，无法直接推广至剪枝或量化；2）需预先获取完整模型进行SVD，适用性受限；3）在极度稀疏或极大压缩时仍可能出现性能下降；4）未深入探讨安全/隐私保护细节。

---

## 200. Outcome-Guided Distillation: A Teacher-Student Framework to Advance VLM Reasoning in Autonomous Driving

**arXiv ID:** 2607.29052 | [PDF](https://arxiv.org/pdf/2607.29052v1)

**作者:** Zeyu Dong `[一作]` (Stony Brook University), Yu Sun `[通讯]` (Sunrise Technology Inc.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一个教师-学生框架，利用反思式推理（Reflective Reasoning）训练 VLM，随后通过知识蒸馏生成小型学生模型，并使用分离的 RealNum-Decoder 将文本推理转化为精准的轨迹。

**💡 创新点**

创新点包括：①引入基于最终行动结果的反思式推理，消除对人工 CoT 标注的依赖；②冻结视觉编码器以防止灾难性遗忘；③采用分离的文本推理与数值轨迹回归网络，解决 VLM 在连续数值输出上的精度瓶颈；④通过双重蒸馏（词级与嵌入级）高效压缩模型。

**🔧 技术方法**

技术栈：InternVL3（8B/78B） VLM，ViT 视觉编码器，GRU 轨迹回归，教师-学生蒸馏（Token+Embedding 级），RealNum-Decoder，反思式推理流程（多轮对话+目标导向），RFS 评价机制，Adam + 线性学习率调度，DeepSpeed 训练。

**📊 数据集**

使用 Waymo 视觉端到端驾驶数据集，包含 4,021 条驾驶片段，约 415k 训练样本，目标是预测 5 秒内的未来 2D 轨迹。

**📈 对比分析**

与 Direct Prediction、Standard CoT、Text‑Based Waypoints、Finetuned ViT 等基线对比，最终 8B+1B 模型在 RFS 7.240、ADE 3.231、ADE@3s 2.726、ADE@5s 3.735 上均优于所有 8B 基线，且仅比 78B Teacher 略低（RFS 7.639），显示出显著的性能提升和压缩率。

**⚠️ 局限性**

局限性：①仅使用单帧视觉输入，未利用视频历史，可能忽略动态上下文；②学生模型需要逐词生成推理链，导致推理延迟，难以满足极低时延或资源受限的边缘设备需求。

---

## 201. ReMoE: Report-Guided Mixture-of-Experts for Multimodal OCT/OCTA Anomaly Detection

**arXiv ID:** 2607.29039 | [PDF](https://arxiv.org/pdf/2607.29039v1)

**作者:** Zihan Nie `[一作]` (Shandong University), Zongyuan Ge `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种报告引导的多模态异常检测框架ReMoE，利用正常医学报告语义蒸馏为图像-文本先验，并通过模态感知专家路由实现特征差异的语义化；

**💡 创新点**

将正常报告语义转化为图像条件的伪文本先验，构建模态感知先验并在RMM中进行专家路由和特征调制，从而突破纯视觉异常检测对外观偏差的依赖；

**🔧 技术方法**

使用CLIP文本编码器进行报告语义蒸馏、DINOv2 ViT作为视觉编码器、ViT解码器、混合专家(Mixture-of-Experts)与模态感知路由(RMM)、多尺度卷积专家及图像-文本先验学习；

**📊 数据集**

私有OCT/OCTA数据集（4通道、配对报告）和公开OCTA500-3MM（3mm子集4个模态组合），公共数据使用固定GPT生成的正常报告作为先验；

**📈 对比分析**

与多种单模态和多模态基线（AnyAD、UniMMAD、MMRAD等）对比，ReMoE在私有数据集I-AUROC 0.8682、I-AP 0.9358，超越AnyAD 8.8% AUROC、4.3% AP；在OCTA500-3MM提升4.1% AUROC、2.3% AP；四输入模式表现最佳；

**⚠️ 局限性**

仅在视网膜OCT/OCTA上验证，报告先验受限于配对报告，公共数据需固定报告；未验证在更大、多模态组合或不同报告风格上的鲁棒性；模型复杂度和推理速度需进一步评估。

---

## 202. GoldenRetriever: Non-Interactive Homomorphic Encrypted Retrieval for Privacy-Preserving RAG

**arXiv ID:** 2607.29019 | [PDF](https://arxiv.org/pdf/2607.29019v1)

**作者:** Yang Gao `[一作]` (University of Central Florida), Liqiang Wang `[通讯]` (University of Central Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于阈值的非交互式加密检索框架，能在检索增强生成（RAG）中安全地选择文档并返回完整文本，避免了传统加密top‑k排序的高成本与信息泄露风险。

**💡 创新点**

创新点在于用阈值选择替代top‑k排名，降低计算复杂度至线性；引入精度稳定的多项式极化函数以保证近似加密结果能精确恢复离散令牌；实现完整的端到端非交互式加密检索。

**🔧 技术方法**

利用CKKS同态加密进行加密相似度计算、阈值判断和文档掩码；采用Chebyshev多项式实现指示函数；使用多项式极化压缩误差；采用SIMD打包和旋转键加速批处理。

**📊 数据集**

在MS MARCO、Natural Questions、HotpotQA和FiQA等检索基准上构建候选集，使用BGE‑base（384维）稠密检索模型进行嵌入。

**📈 对比分析**

与纯文本基线和加密top‑k基线比较，阈值检索在召回率、令牌准确率和文档准确率均与基线持平；相比加密top‑k，平均检索时延从约16580秒降至约1052秒；纯文本检索仍最快。

**⚠️ 局限性**

局限在于文档语料本身仍以明文存储，未对语料进行加密；加密计算仍显耗时，尤其是大规模候选集时；阈值设置需手动调优，过高会降低召回率。

---

## 203. Forwardrobe: Garment-Aware Gaussian Avatars from a Single Image

**arXiv ID:** 2607.29106 | [PDF](https://arxiv.org/pdf/2607.29106v1)

**作者:** Daisheng Jin `[一作]` (Nanyang Technological University), Ying He `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

做了什么：提出了一种基于单张图片的前向框架 Forwardrobe，能够从单张服装人物图像中重建可动画化、可编辑、可交换的 Gaussian 服装资产，并与身体资产组合实现服装动画、编辑与 3D 虚拟试穿。

**💡 创新点**

创新点是什么：首次实现了服装与身体的显式解耦，采用 VLM 指导的类别感知连续性初始化、基于皮肤权重重分配的裙装连续化、以及运动条件下的残差式动态模块和着色分解，从而显著提升了宽松服装在姿势变化下的几何与视觉连贯性。

**🔧 技术方法**

用了什么技术：利用 3D Gaussian splatting 进行高效三维表示，结合视觉语言模型（VLM）获取服装描述进行类别引导，使用基于 Transformer 的动态模块对姿势、图像、语义多模态进行融合，预测几何、光照和皮肤权重残差，并在合成步骤中实现身体与服装的边界平滑。

**📊 数据集**

用了什么数据集：在 UBC Fashion、X-Humans 与 NeuMan 三大数据集上进行训练与评估，分别覆盖宽松服装、多视角人像与户外运动场景。

**📈 对比分析**

如何比较的方法，性能怎么样：在 UBC Fashion 与 NeuMan 上与 LHM、IDOL、PERSONA、DynaAvatar 等单图像头像重建方法对比，使用 PSNR/SSIM/LPIPS 评估，在全图与服装区域均获得相近或更优指标，并在视觉上表现出更佳的服装连贯性与细节保留。

**⚠️ 局限性**

limitation是什么：仍依赖于静态重建骨干，处理极端姿势或复杂纹理的服装时效果有限；对纹理细节的动态捕捉仍不够精准，需进一步改进多模态细节重建与实时性优化。

---

## 204. Faster but Different: Diagnosing and Controlling Content Drift in Accelerated Multimodal Diffusion Language Models

**arXiv ID:** 2607.29079 | [PDF](https://arxiv.org/pdf/2607.29079v1)

**作者:** Yaoxuan Dou `[一作]` (Beijing Institute of Technology), Yang Shu `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并量化了在多模态扩散式大语言模型加速（Fast‑dLLM 等）过程中出现的内容漂移问题，并提出通过调节 KV‑cache 刷新间隔来控制一致性。

**💡 创新点**

发现置信度阈值并不影响内容漂移，KV‑cache 刷新频率是唯一可调节且连续控制一致性的手段；跨实现（Fast‑dLLM、dLLM‑Cache、LaViDa）验证了结论的普适性，并给出最优刷新策略。

**🔧 技术方法**

使用 Fast‑dLLM、dLLM‑Cache、LaViDa 这三种加速实现；对 KV‑cache 刷新、图像‑文本状态刷新、图像交换等干预进行因果分离；采用 Jaccard 相似度、BERTScore、手工审计和双盲图像真实性评估作为一致性与事实性指标。

**📊 数据集**

主要使用 MME 基准图像（300 张真实图片，50 张用于细粒度实验）进行对齐生成；同时在 50 张独立图像上检验提示和样本泛化。

**📈 对比分析**

与未加速基线相比，Fast‑dLLM 在默认刷新间隔下可获得 10–12 倍速度提升，内容一致性（Jaccard）约 0.42；将刷新间隔调至 1 可使一致性提升至 0.99，速度仍比未加速快 1.3 倍；dLLM‑Cache 仅在同时收紧两种缓存时才能恢复一致性，导致失去加速优势；自适应刷新策略未能突破固定间隔的速度–一致性前沿。

**⚠️ 局限性**

局限性包括：实验仅在 LLaDA‑V 上验证；Jaccard 与 BERTScore 仍为词汇层面度量，未覆盖更深层次语义一致性；仅对默认配置做了事实性双盲评估，样本量有限；未测试网页渲染、搜索、排名等完整 Web‑agent 场景；加速方案的安全性与事实性并未得到保证。

---

## 205. Who Wins Where? Conformal Model Comparison for Local Superiority

**arXiv ID:** 2607.29053 | [PDF](https://arxiv.org/pdf/2607.29053v1)

**作者:** Yi Zhou `[一作]` (National University of Singapore), Ke-Wei Huang `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了“自适应局部模型比较”框架，通过在局部邻域内比较两模型的损失差异，并用一次性校准（conformal）保证局部决策的错误率；

**💡 创新点**

创新点在于将conformal校准引入局部模型比较，提供了局部赢家地图的统计显著性保障，并证明全局平均与局部优势的分离及平方损失下的偏差-方差分解；

**🔧 技术方法**

技术上采用三段拆分（训练、估计、校准）与局部加权核估计、一次性残差阈值构造，并实现一边界校准和多重比较的控制；

**📊 数据集**

实验使用五个一维模拟数据集以及四个公开回归基准（Auto MPG、Boston Housing、Concrete Compressive Strength、Energy Efficiency）来验证方法；

**📈 对比分析**

在模拟实验中方法成功识别局部赢家并保持预定的错误率；在真实数据中局部校准方法能在小范围内实现显著的平均收益，优于全局选择且比非校准局部方法更保守但更可靠；

**⚠️ 局限性**

局限性包括样本效率低（需三份拆分）、高维空间中的局部化困难、校准仅对未来得分给出边际控制而非对均值的置信区间、以及多点比较时需额外的误差控制。

---

## 206. What Is Missing in Surgical Risk Stratification and Outcome Prediction: A Scoping Review of End-to-End Machine Learning Approaches

**arXiv ID:** 2607.29090 | [PDF](https://arxiv.org/pdf/2607.29090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 207. Rethinking Detection Calibration: A Coordinate and Direction Perspective

**arXiv ID:** 2607.29040 | [PDF](https://arxiv.org/pdf/2607.29040v1)

**作者:** Juyong Lee `[一作]` (Chung-Ang University), Jongwon Choi `[通讯]` (Chung-Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

为目标检测模型提供一种后置校准框架ReDC，能够输出坐标级别的置信度分数，并给出误差方向信息；

**💡 创新点**

创新点包括：①引入坐标级别的对齐比例(CAR)度量局部定位误差；②使用置信度重编码器(Confidence Re‑Encoder, CR)将CAR映射到坐标级置信度；③通过方向位移估计器(Directional Displacement Estimator, DDE)预测误差方向；④提出坐标级期望校准误差(C-ECE)和方向感知校准误差(Da‑CE)两种新评估指标；

**🔧 技术方法**

技术手段主要为后置校准、基于检测器产生的边框特征与logit的轻量化网络（MLP），以及校准后IoU的近似（利用C‑ECE、DDE、等距回归与Platt缩放）；

**📊 数据集**

实验数据集涵盖COCO、Cityscapes及其变种（COCO‑C、Foggy Cityscapes）以验证在域内与域外的鲁棒性；

**📈 对比分析**

与多种SOTA方法（训练时校准如TCD、BPC、Cal‑DETR；后置校准如IR、PS）在坐标级与框级指标（C‑ECE、Da‑CE、D‑ECE、LaECE）上进行对比，ReDC在坐标级校准上显著优于所有对比方法，同时在框级指标保持与现有方法相当，且在多种检测器（一阶段、两阶段、Transformer）上表现一致；

**⚠️ 局限性**

局限性包括：①仍依赖已有的检测器特征，需要额外的轻量网络；②目前仅针对二维轴对齐框，未针对旋转框或3D检测展开；③在极端域偏移（如跨城市、跨传感器）下，坐标级校准效果仍有提升空间。

---

## 208. GO-PRE: Goal-Oriented Next-Best-View Selection via Predictive Rendering Entropy for Active 3D Reconstruction

**arXiv ID:** 2607.29037 | [PDF](https://arxiv.org/pdf/2607.29037v1)

**作者:** Yan Song `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GO-PRE框架，通过最大化对用户指定视图集合的预测渲染熵降低来实现主动3D重建的下一最佳视角选择。

**💡 创新点**

创新点在于将信息增益直接定义在渲染预测空间，利用目标熵梯度矩阵（Goal Hessian）实现目标导向的视角选择，并兼顾交互式目标设定。

**🔧 技术方法**

使用差分渲染（如3D Gaussian Splatting）、Laplace近似、Fisher信息矩阵、蒙特卡洛采样和对数行列式近似等技术。

**📊 数据集**

在Blender、Mip-NeRF360和Tanks & Temples三个公开基准数据集上进行实验。

**📈 对比分析**

与随机、ActiveNeRF、FisherRF、GauSS-MI、POp-GS等基线相比，GO-PRE在PSNR/SSIM/LPIPS等指标上均取得显著提升，尤其在目标导向重建任务中表现突出。

**⚠️ 局限性**

局限性包括只能在离散候选视角集合上选择，且对连续姿态的梯度优化受限于渲染器的二阶导数和可见性不连续导致的数值不稳定。

---

## 209. Semantics of Subterfuge: Benchmarking Legal Deception Detection Against General-domain State-of-the-Art

**arXiv ID:** 2607.29066 | [PDF](https://arxiv.org/pdf/2607.29066v1)

**作者:** Theekshana Samaradiwakara `[一作]`, George C. Lobb `[通讯]` (Law Office of George C. Lobb)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

综述并实验比较了面向法律领域的自动欺骗检测模型，探讨了模型与提示策略在不同数据集上的表现。

**💡 创新点**

创新点在于统一跨域评估框架、对比Transformer微调与LLM零/少样本提示、揭示领域敏感性与CoT效果不佳。

**🔧 技术方法**

使用了Transformer微调模型（RoBERTa、BERT、DeBERTa、ALBERT、DistilBERT、T5‑base）以及LLM（GPT‑4o、GPT‑4o‑mini、LLaMA、Gemma2、Phi系列）并结合零样本、4‑shot直接分类与链式推理提示。

**📊 数据集**

实验使用七个公开数据集：法律域（RLTD、DECOUR）和通用域（OpSpam、cCult、DeRev2014、Liar、FakeNewsNet）。

**📈 对比分析**

通过统一训练/验证/测试拆分和多种提示策略进行比较，结果显示在大数据集上微调模型优于LLM，在小法律数据集上少样本LLM（如GPT‑4o）能匹配或超越微调模型；链式推理往往不提升或降低性能。

**⚠️ 局限性**

局限性包括：数据量稀缺导致过拟合与领域迁移差异、模型可解释性不足、对抗鲁棒性低、提示策略对模型和数据依赖强、未涵盖多语言或跨文化场景。

---

## 210. The Deployment Wall: A Diagnostic Framework and Instrument for Enterprise AI in the Deployment Era

**arXiv ID:** 2607.29089 | [PDF](https://arxiv.org/pdf/2607.29089v1)

**作者:** Fabricio F. Costa `[一作]` `[通讯]` (HCLTech), Fabricio F. Costa (HCLTech)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了“Deployment Era”框架，强调在企业AI投入中，部署过程中的摩擦（称为seam）是价值实现的关键，而非模型的智能；通过构建Deployment Wall、Seam Index和Deployment Debt三大概念，为平台选择与治理提供可度量的决策工具；并用这些工具解释企业AI失败率高的现象及竞争优势的转移。

**💡 创新点**

创新点包括：1) 将软件工程中的技术债务概念迁移到AI部署，提出Deployment Debt；2) 设计了可操作的Seam Index（0–12分），将平台的摩擦移除程度量化；3) 通过Deployment Wall阐释价值泄漏的六阶段路径；4) 将竞争优势从模型能力转向“seam moat”，说明平台差异化在于已解决的摩擦。

**🔧 技术方法**

采用的软件与技术方法：MLOps、数据治理、身份与访问管理、合规治理、工作流再设计等技术领域；在此基础上构建三维框架与指标，利用软件工程技术债务模型、结构化综述与案例观察等方法论。

**📊 数据集**

主要数据来源为：MIT Project NANDA、RAND、S&P Global、McKinsey、BCG、Menlo Ventures等独立研究与调查的定量/定性结果；同时作者结合其在企业部署中的现场经验与案例观察形成支持性证据。

**📈 对比分析**

比较方法：构建六阶段Deployment Wall来预测项目成功率；使用Seam Index对平台进行评分，并通过六条可检验命题评估其与生产率、成本等指标的关联；论文未给出具体实验数据，但通过结构化综述与示例表明较高的Seam Index与更高的生产率、较低的总拥有成本相关。

**⚠️ 局限性**

局限性：1) 研究为概念性设计，缺乏大规模实证验证；2) 依据作者现场观察与非随机样本，可能存在偏倚；3) 量化示例为示意，未提供精确计量；4) 市场环境快速变化，数据可能过时；5) 对小型或数据敏感度低的组织适用性有限。

---

## 211. Do Music Foundation Models Embed Pitch in Helical Structure?

**arXiv ID:** 2607.29086 | [PDF](https://arxiv.org/pdf/2607.29086v1)

**作者:** Hayato Yagi `[一作]` (Keio University), Shigeo Morishima `[通讯]` (Waseda University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对音乐基础模型（MFM）的中间表示进行分析，发现音高信息在三维螺旋结构中形成螺旋，并通过 Helicality 指标量化其几何特征。

**💡 创新点**

首次将音高的双维感知（高度 + 八度等价）映射为三维螺旋结构并量化，揭示谐波组成对内部表示的显著影响，提供了对 MFM 内部机制的新视角。

**🔧 技术方法**

使用 PCA 降维、参数化螺旋拟合、Helicality 评分、多元回归分析，以及人工合成谐波控制信号进行实验。

**📊 数据集**

利用 SynTheory 92 乐器单音数据集和基于等比谐波的人工信号（137 种谐波组合）进行实验。

**📈 对比分析**

通过将 Helicality 与随机三维向量对比，评估各 Transformer 层的螺旋清晰度；发现更深层螺旋更为明显，Jukebox 的平均 Helicality 为 2.28，MusicGen 为 1.16；多元回归显示八度谐波正贡献，非八度谐波负贡献。

**⚠️ 局限性**

仅对两款 MFM（Jukebox、MusicGen）进行研究，未检验在更大模型或不同训练数据下的普适性；未确定螺旋结构的来源是 Transformer 还是更早层；缺乏与下游任务性能的直接关联。

---

## 212. MHRGait: Gait Recognition from Momentum Human Rig Pose

**arXiv ID:** 2607.29083 | [PDF](https://arxiv.org/pdf/2607.29083v1)

**作者:** Huiran Duan `[一作]` (City University of New York), Yingli Tian `[通讯]` (City University of New York)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出将 Momentum Human Rig（MHR）姿态参数作为步态识别的新表征，构建轻量级的 MHRGait 与 MHRGait++ 两个模型，通过语义分组、语义自适应聚合和时间卷积实现对人体及手部控制的建模，并将其与二值轮廓进行模态平衡融合。

**💡 创新点**

创新点在于：① 用可学习的姿态控制参数代替传统的几何形状或稀疏关节坐标；② 采用解剖学语义分组与自适应聚合的轻量化编码器；③ 引入模态平衡检索策略，使不同模态的贡献不受特征数量影响。

**🔧 技术方法**

使用技术包括：MHR 姿态恢复（基于 SAM 3D Body）、混合器式语义编码器、残差瓶颈时间卷积、BN Neck 分类器、三元组+交叉熵损失、DeepGaitV2 轮廓编码以及模态平衡距离融合。

**📊 数据集**

使用数据集：SUSTech1K、CCPG、CCGR-MINI、CASIA-B* 四个步态基准，涵盖服装变化、不同视角、夜间与复杂共变等多种场景。

**📈 对比分析**

通过与现有基于模型的稀疏姿态、轮廓、3D网格以及多模态方法对比，MHRGait 在 SUSTech1K（Rank‑1 67.8%）和 CCPG（Rank‑1 68.2%）实现最佳整体性能；MHRGait++ 在 DeepGaitV2 基础上提升 Rank‑1 8.3%–18.4%，并在跨域迁移中表现最优。

**⚠️ 局限性**

局限性包括：依赖冻结的跟踪和单目 MHR 估计，估计误差会直接影响识别；未实现端到端训练；对姿态恢复误差、不同人种或姿态异常的鲁棒性尚需进一步研究。

---

## 213. Can Zero-Shot LLMs Predict Child Malnutrition? A Fairness and Temporal Robustness Study

**arXiv ID:** 2607.29082 | [PDF](https://arxiv.org/pdf/2607.29082v1)

**作者:** Muhammad Ashad Kabir `[一作]` (Charles Sturt University), Md Ahshanul Haque `[通讯]` (Charles Sturt University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了在孟加拉国人口与健康调查（BDHS）数据上，使用零样本LLM（GPT‑4o‑mini）预测儿童发育迟缓（身高不足）是否可行，并评估其公平性和时间稳定性。

**💡 创新点**

首次将提示式零样本推理应用于结构化人口健康调查数据，系统比较LLM与传统监督模型的性能，并从公平性和时间鲁棒性两个维度揭示LLM潜在优势与局限。

**🔧 技术方法**

采用GPT‑4o‑mini进行零样本推理，使用列表序列化的提示格式；将结果与随机森林基准进行对比，并通过敏感度、特异度、balanced accuracy、AUROC 等指标评估模型。

**📊 数据集**

孟加拉国人口与健康调查（BDHS）2007‑2022年数据，共17,106名儿童（5,623例发育迟缓），包括母亲、儿童、医疗、家庭等特征。

**📈 对比分析**

通过5折交叉验证与单次推理进行对比：LLM balanced accuracy 58%（vs 57%），AUROC 0.632（vs 0.685）；LLM敏感度 77.5%（显著高于随机森林的23.4%），但特异度仅38.6%（远低于随机森林的90.7%），显示LLM更擅长捕捉发育迟缓案例但误报较多。

**⚠️ 局限性**

模型在居住地和财富指数等社会经济组别上表现出显著不公平（如贫困组敏感度100%但特异度0%），特异度普遍偏低导致误报；仅评估单一LLM，未考虑其他LLM或训练策略；缺乏对模型偏差来源的深入解释。

---

## 214. Selective KV Cache Protection for Noise-Resilient LLM Inference on Analog Compute-In-Memory Systems

**arXiv ID:** 2607.29076 | [PDF](https://arxiv.org/pdf/2607.29076v1)

**作者:** Yuannuo Feng `[一作]` (Beihang University), Wang Kang `[通讯]` (Beihang University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了在模拟计算内存（CIM）阵列上执行动态注意力计算的系统研究，并提出基于令牌级别的分层保护策略（GoS），将关键的“sink”和最近令牌保持在高精度数字路径，其余令牌使用低精度模拟路径；同时设计了联合调度器实现批量写入与动态所有权切换，显著提升KV缓存编程利用率；

**💡 创新点**

创新点在于：①识别并利用注意力中令牌级别的不对称噪声易感性，将保护资源集中于最敏感的“sink”和最近令牌；②构造混合数字-模拟执行架构与批量迁移调度，解决模拟阵列稀疏编程导致的利用率低问题；③提供完整的硬件-算法协同设计，兼顾能效、延迟与准确性；

**🔧 技术方法**

采用模拟CIM阵列进行矩阵向量乘法、数字路径的高精度计算、数字-模拟双路得分生成、全局softmax、批量KV缓存迁移调度、3D集成硬件结构与数字/模拟混合加速器；

**📊 数据集**

使用WikiText-2、ARC-Challenge、PIQA、GSM8K、MATH500等标准基准；在九种LLM（Qwen3、Llama、DeepSeek、OLMo、OLMoE等）上进行评估；

**📈 对比分析**

通过与无保护模拟基线、平均化、k-b校准等对比，GoS在模拟噪声下平均PPL从33.91降至11.95（接近干净基线11.06），并将KV编程行利用率从23.1%提升至91.2%；在噪声敏感的模型中，能效、延迟提升仅3%/4%；在下游任务和高温噪声场景下，准确率保持与干净数字推理相近；

**⚠️ 局限性**

局限性包括：只针对动态KV缓存保护，未消除投影阵列的噪声；受限于固定的保护预算和迁移阈值，最佳值随模型、设备或服务需求变化；未在真实芯片上完成全系统评估，能耗/延迟估算为设计预测；数字缓冲区仍需一定存储，且冷长上下文的HBM流量未减少；

---

## 215. Evidence-Grounded Constraint Checking in Construction Documents

**arXiv ID:** 2607.29058 | [PDF](https://arxiv.org/pdf/2607.29058v1)

**作者:** Rashid Mushkani `[一作]` (Cocoon Lab), Shin Koseki `[通讯]` (Cocoon Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对建筑专业文档中的约束检查，构建了一套基于证据的处理管线，包括事实提取、四状态确定性规则、源级证据追踪与专家升级机制，并在固定文本检索条件下对比页面覆盖与区域聚焦的视觉证据分配对决策准确性的影响。

**💡 创新点**

创新点在于：①提出可审计的四状态约束归约器和源级事实合同，实现对缺失/不确定事实的显式处理；②将视觉证据分配拆分为Page-RAG与Region-RAG两种策略，在实验中揭示分辨率与宽度的权衡；③采用配对重复实验和宽度扩展实验验证策略效果，提供可复现的评测框架。

**🔧 技术方法**

技术手段包括：布局保持的文本检索（BM25）、PDF视觉区域提取与裁剪、Typed Facts结构化事实表示、确定性规则引擎、四状态决策逻辑、视觉证据分配（Page-RAG、Region-RAG）、多系统（Gemini、GPT）推理、以及基于重复调用的置信度与校准评估。

**📊 数据集**

数据集为Cocoon Lab公开的专业建筑文档约束检查任务，涵盖7个约束族（引用解析、技术细节、标题、标注、索引、规格对照、提交审核）共160个任务级参考，涉及23个不同项目的PDF文档。

**📈 对比分析**

通过在六个项目上进行四系统重复实验和在23个宽度扩展项目上进行两系统实验，配对比较Page-RAG与Region-RAG。重复实验中Region-RAG提升约10.7个百分点的决策准确率，错误率下降；宽度扩展实验中Region-RAG反而略低。整体精确度、查找F1及校准仍处于低水平，表明单一视觉策略无法满足专业审核需求。

**⚠️ 局限性**

局限性包括：①样本仅覆盖6个项目，缺乏独立验证；②仅评估PDF输入，未测试CAD/IFC；③缺少区域级证据标注，无法评估查找精确度；④重复一致性与置信度校准不可靠；⑤视觉附件阈值与裁剪策略受实验设计限制，无法代表实际生产场景。

---

## 216. Designing a digital word-learning intervention with neurodiverse children: Experiences and ideas from children with developmental language disorder

**arXiv ID:** 2607.29113 | [PDF](https://arxiv.org/pdf/2607.29113v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 217. DoubleHelix: Structured Cross-Modal Fusion for Audio-Visual Speech Recognition with LLMs

**arXiv ID:** 2607.29112 | [PDF](https://arxiv.org/pdf/2607.29112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 218. Learning Lookahead Lemmas for Neural Network Verification

**arXiv ID:** 2607.29051 | [PDF](https://arxiv.org/pdf/2607.29051v1)

**作者:** Liam Davis `[一作]` (Amherst College), Haoze Wu `[通讯]` (Amherst College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21`

**🎯 论文内容**

未提供论文内容

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法比较，无法评估性能

**⚠️ 局限性**

未能获取论文细节，无法分析局限性

---

## 219. The Entropic Sum-Product Phenomenon

**arXiv ID:** 2607.29042 | [PDF](https://arxiv.org/pdf/2607.29042v1)

**作者:** Rupert Li `[一作]` `[通讯]` (Stanford University), Rupert Li (Stanford University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文证明了独立同分布的离散实值随机变量X和X'的熵和乘积的最大值与其熵H(X)之间的关系，具体为maxH(X+X'), H(XX')≥8/7 H(X)-O(log H(X))。

**💡 创新点**

创新点在于提出了熵的和-积现象的明确界限，解决了Goh提出的问题，并且通过分割分布的方法克服了之前研究中的限制。

**🔧 技术方法**

使用了熵的基本性质、组合数理论和均匀化技术等数学工具，结合了线性熵的Elekes-Ruzsa定理。

**📊 数据集**

使用了离散实值随机变量的熵作为数据集，具体数据集未明确列出，但涉及到的随机变量是有限支持的。

**📈 对比分析**

与之前的研究方法相比，论文通过均匀化技术和新的界限证明了更强的结果，性能上达到了8/7的界限，优于之前的4/3。

**⚠️ 局限性**

限制在于对于具有原子（即概率为0的值）的随机变量，结果可能不再适用，且在某些情况下，熵的界限可能会受到影响。

---

## 220. TransMem: Transforming Hidden States into Memory for Large Language Models

**arXiv ID:** 2607.29032 | [PDF](https://arxiv.org/pdf/2607.29032v1)

**作者:** Haodong Lei `[一作]` (Southeast University), Hongsong Wang `[通讯]` (Southeast University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种轻量级的推理时参数化记忆模块TransMem，用于将冻结LLM骨干网络的稀疏历史隐藏状态转换为可重用的记忆表示，以提升长上下文推理性能。

**💡 创新点**

创新点在于：①利用隐藏状态的稀疏性与位置敏感性构造记忆；②通过证据条件自蒸馏（ECSD）训练，让模块学习在全上下文中恢复仅含证据的教师分布，从而实现记忆能力的提升而不增加具体知识；③实现与骨干网络解耦、常量计算开销的记忆机制。

**🔧 技术方法**

使用技术包括：冻结LLM骨干；在第K层插入TransMem模块（含Transformer块、投影层与门控网络）；动态分段稀疏抽取隐藏状态；证据条件自蒸馏训练；多种基线对比实验。

**📊 数据集**

使用数据集：LoCoMo、HotpotQA、MemoryAgentBench；训练仅在HotpotQA上完成，随后在其他数据集上进行评估。

**📈 对比分析**

与多种外部记忆和参数化记忆方法对比，TransMem在LoCoMo平均提升11.58–29.25 F1，在HotpotQA提升10.20–13.03 F1，在MemoryAgentBench准确率从29.54%提升至40.00%；在不同模型规模和架构上均保持一致。

**⚠️ 局限性**

局限性：仅在冻结骨干上验证，未探索自适应/动态分段的极限；对超长上下文的鲁棒性仍有限；未评估与主动学习或多模态场景的结合；在极端噪声或不完整证据环境下的表现未知。

---

## 221. VSTaI: Design and Characterization of Variable-Stiffness Tactile Interfaces Based on 3D-Printed Structured Fabrics

**arXiv ID:** 2607.29102 | [PDF](https://arxiv.org/pdf/2607.29102v1)

**作者:** Yiting Mo `[一作]` (National University of Singapore), Fernando Bello `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计、制备并表征了基于3D打印结构织物的可变刚度触觉接口VSTaI，利用真空诱导锁结实现从柔软到硬化的触感渲染。

**💡 创新点**

将3D打印链甲结构与真空层锁结结合，首次实现形状可适应、可调刚度的触觉显示，且在子兆帕到兆帕级范围内提供可调刚度。

**🔧 技术方法**

采用3D打印、聚乳酸/聚乙烯醇结构织物、真空锁结、硅胶软层以及力-位移测试与线性弹性模量计算技术。

**📊 数据集**

无传统数据集，仅基于八个不同结构样品的实验力学测量数据。

**📈 对比分析**

在大气压与-90kPa真空两状态下进行五点力-位移测试，计算Young模量；6‑in‑1链甲在真空下平均刚度提升140%，达到1220kPa，且空间均匀性最优。

**⚠️ 局限性**

仅验证两极状态，未探究中间压力下的刚度变化，缺乏动态重复加载和粘弹性评估，样品数量有限，空间非均匀性在某些结构中较大。

---

## 222. Harnessing the Wisdom of LLM Crowds through Complementarity-Driven Iterative Collaboration

**arXiv ID:** 2607.29087 | [PDF](https://arxiv.org/pdf/2607.29087v1)

**作者:** Yanbin Fang `[一作]` (Chinese University of Hong Kong), Wei Chen `[通讯]` (University of Connecticut)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了WILC（Wisdom Integration of LLM Crowds）框架，利用多模型协同的relay‑style（中继式）互补性，通过迭代反思与改进实现多LLM协作；

**💡 创新点**

创新点在于把集体智慧从静态聚合转化为动态中继式互补，并设计了两条可迁移的设计原则：迭代反思与改进（DP1）和基于互补性的模型选择（DP2），以及结合PCF/PCG门控的上下文多臂赌博机选择机制；

**🔧 技术方法**

核心技术包括基于上下文LinUCB的多臂赌博机模型、双门控（PCF与PCG）模型选择、一次性前向搜索（OSFS）与燃料初始化（burn‑in）策略、以及自我量化的代理奖励机制；

**📊 数据集**

使用四大基准数据集：HumanEval（代码生成）、MATH‑500（数学推理）、MMLU（通用知识推理）和VisEval（自然语言‑可视化生成），并在14B和30B规模的多源开源LLM队列上进行评估；

**📈 对比分析**

与单模型执行、ReAct、Reflexion、自我集成、异构集成以及专门的查询路由方法等基线进行比较，WILC在所有四个基准上均表现最佳，平均提升约3–8个百分点，且在成本上可比GPT‑5.2低约7×，并保持相近的准确率；

**⚠️ 局限性**

局限性包括对模型异质性的高度依赖、协同过程对API调用和时延的影响、协调器选择对反思质量的要求、以及在极度复杂或极其简单任务上可能不完全受益。

---

## 223. IyawoBench v2.0: Extended Diagnostic Evaluation of Large Language Model Clinical Triage in Nigerian Primary Care

**arXiv ID:** 2607.29085 | [PDF](https://arxiv.org/pdf/2607.29085v1)

**作者:** Anthonio Oladimeji Gabriel `[一作]` (Iyawo Health), Dimeji Olawuyi `[通讯]` (Iyawo Health)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出IyawoBench v2.0诊断性评估框架，对大型语言模型在低中等收入国家临床分诊的安全性进行系统性失效模式分析。

**💡 创新点**

通过正式数学定义、Escalation Bias Index与Expected Deployment Cost两项新指标，揭示传统敏感度指标掩盖的系统性失误，并展示不同部署场景下最佳模型不一致的失效模式分类。

**🔧 技术方法**

采用结构化小贴士生成、定量评估指标（包括新指数与成本矩阵）、统计推断（McNemar、Wilson置信区间）以及对比实验。

**📊 数据集**

使用200个基于1200例真实尼日利亚初级卫生中心患者数据生成的合成分诊病例，涵盖八类发热疾病。

**📈 对比分析**

将Claude Sonnet 4.6、Llama 3.3 70B、Llama 3.1 8B及五个基线模型进行对比；结果显示所有模型至少存在一种失效模式，传统准确率误导，且最佳模型随部署场景（急诊、系统可持续、均衡）变化。

**⚠️ 局限性**

仅评估发热疾病分诊，样本量有限，成本权重需本地校准，未覆盖非发热或慢性疾病等临床情况。

---

## 224. Benchmarking Frontier Large Language Models Against Official Crash Database Coding Using Police Crash Narratives

**arXiv ID:** 2607.29064 | [PDF](https://arxiv.org/pdf/2607.29064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 225. DASH-OPD: Discrepancy-Aware Switching with Hysteresis for On-Policy Distillation

**arXiv ID:** 2607.29078 | [PDF](https://arxiv.org/pdf/2607.29078v1)

**作者:** Yuchen Xia `[一作]` (Chinese University of Hong Kong), Yunjian Xu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 DASH-OPD 的多轮 on-policy distillation 方法，能够在学生模型与教师模型之间自适应双向切换执行器。

**💡 创新点**

创新点在于通过累计教师-学生的方向性差异（漂移与恢复证据）并设置滞后阈值，实现对教师干预时机的自适应决策，从而避免频繁切换并显著减少教师使用。

**🔧 技术方法**

采用了 mean log‑probability 比值作为差异信号，使用归一化、累积证据、滞后阈值控制切换，结合 reverse‑KL 与 forward‑KL 的角色感知蒸馏损失；实现代码复杂度仅为 O(1) 每回合。

**📊 数据集**

在 ALFWorld 文本式家庭环境上进行实验，使用 Qwen3-1.7B、Qwen3-4B 学生模型以及冻结的 Qwen3-30B-A3B 教师模型。

**📈 对比分析**

与零样本学生、普通 OPD、TCOD 以及 Guided‑OPD 等基线对比，DASH‑OPD 在所有 140 IID 与 134 OOD 任务上均取得最高或并列最高的成功率，同时平均交互回合数最低，令学生模型在 4B 规模下超过 30B 教师的成功率，显示出显著的训练与部署效率提升。

**⚠️ 局限性**

局限在于教师‑学生差异只是风险的粗略代理，可能忽视共享错误或误触发干预；未来可探索基于模型不确定性或环境反馈的更精细风险评估。

---

## 226. A Generalized-Bayes Perspective on Counterfactual Explanations: Posterior-Based Decision-Making and Evaluation

**arXiv ID:** 2607.29077 | [PDF](https://arxiv.org/pdf/2607.29077v1)

**作者:** Keita Kinjo `[一作]` `[通讯]` (Kyoritsu Women's University), Keita Kinjo (Kyoritsu Women's University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

该论文提出将对抗解释（Counterfactual Explanation）重新表述为 Generalized Bayes 框架下的 Gibbs 后验，证明传统距离最小化解即为 MAP 估计，并在此基础上设计了多模型不确定性混合后验以及 Bayes 与 CVaR 决策规则。

**💡 创新点**

创新点在于：①将 CE 与 Gibbs 后验等价性建立起来，为传统成本最小化提供概率论基础；②引入模型不确定性混合后验（ModelUnc），从贝叶斯角度处理模型多样性；③提出分布层评估指标（成功概率、尾部风险、稳定性、变量重要性），实现决策规则与分布评估统一。

**🔧 技术方法**

主要技术包括 Generalized Bayes 推导、重要采样逼近 Gibbs 后验、贝叶斯决策（期望损失最小化）、CVaR 决策、模型加权平均、核密度/k‑近邻估计密度以及实验评估指标计算。

**📊 数据集**

使用的实验数据包括：两维与十维的仿真数据（基于非线性回归函数）以及真实数据——Google Trends 中 “One Piece” 搜索兴趣的 10 维时间序列（含 10 个相关关键词）和 2 维子集。

**📈 对比分析**

在模拟和真实数据上，论文通过 L_pt、D_pt、Rb、Plu 等指标比较 MAP、Mean、CVaR 以及 DirectOpt 等决策规则；结果显示 CVaR 在鲁棒性（Rb）上优于 MAP，DirectOpt 在目标达成（L_pt）上最好，而 Mean 由于多模态后验常导致目标失效；后验分布评估显示模型不确定性提升 Tail 与 Stability，揭示模型多样性带来的风险。

**⚠️ 局限性**

限制包括：重要采样在高维下效率低；温度 η 的选择仅经验化，缺乏理论指导；仅处理连续特征，未考虑离散或不可变特征；模型不确定性混合采用离散权重，未实现全贝叶斯后验；整体框架在大规模数据与复杂模型时计算成本较高。

---

## 227. Query Density-Driven Partitioning for Spatiotemporal Load Balancing on Processing-in-Memory Systems

**arXiv ID:** 2607.29070 | [PDF](https://arxiv.org/pdf/2607.29070v1)

**作者:** Takato Hideshima `[一作]` (University of Tokyo), Tomoharu Ugawa `[通讯]` (University of Tokyo)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究探讨了信息系统中的键值存储和分布式内存的相关技术。

**💡 创新点**

创新点在于提出了一种新的主内存引擎的架构，旨在提高数据存取效率。

**🔧 技术方法**

使用了分布式内存管理和主内存数据库技术。

**📊 数据集**

使用了多个公开数据集进行实验，以验证所提方法的有效性。

**📈 对比分析**

与现有的主内存引擎进行了比较，结果显示新架构在数据存取速度和效率上有显著提升。

**⚠️ 局限性**

限制在于实验仅在特定的数据集上进行，可能无法推广到所有应用场景。

---

## 228. Autonomous Repair for Multi-Agent Systems via Monte-Carlo Tree Search

**arXiv ID:** 2607.29055 | [PDF](https://arxiv.org/pdf/2607.29055v1)

**作者:** Hanxiao Lu `[一作]` (Purdue University), Tianyi Zhang `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个将多智能体系统（MAS）修复问题转化为蒙特卡洛树搜索（MCTS）过程的自动化框架，并发布了MASRepairBench这一包含1310条可重放失败轨迹的大规模基准数据集。

**💡 创新点**

创新点在于：①将MCTS改造为支持部分回放、诊断指导节点扩展和基于分类的评估，显著降低令牌消耗并提升搜索效率；②构建了可在不同代理架构和LLM后端下重放的失败轨迹基准，填补了现有失败归因基准缺乏可执行状态的空白；③提供了完整的多架构、多后端评估框架，展示了该方法在多样化环境下的鲁棒性。

**🔧 技术方法**

核心技术包括：蒙特卡洛树搜索（MCTS）、部分回放（partial rollout）、Rollback / Guided Repair / Continuation三种动作；基于LLM的诊断评估与分类增强评估机制；诊断指导节点扩展（diagnosis‑guided node expansion）；Microsoft Agentic Framework（MAF）实现系统执行与回放；以及对不同LLM后端的集成。

**📊 数据集**

使用MASRepairBench数据集：1310条失败轨迹，涵盖四种代理架构（Centralized、Sequential、Decentralized、Concurrent）和四种LLM后端（Qwen3.5‑9B、Qwen3‑30B‑A3B、Nemotron‑3‑Nano‑30B‑A3B、GPT‑5.4‑mini），在GAIA和AssistantBench两个任务集合上构造。

**📈 对比分析**

与DoVer、Reflexion、ReAct等现有基线进行对比。实验结果显示，提出的方法在GAIA上提升post‑repair pass率8.5%–10.3%，在AssistantBench提升6.1%–12.2%；在所有四个后端与四种代理架构下均实现绝对提升3.0%–12.1%。同时，token消耗与最便宜基线相近，甚至比最贵基线低59%。

**⚠️ 局限性**

局限性包括：①对LLM评判的依赖，诊断错误可能导致搜索失效；②部分回放仍需完整重新执行，某些长任务可能因回放长度限制受限；③在非中央化架构下某些基线不可用，限制了对比范围；④实验主要覆盖GAIA和AssistantBench，尚未验证在更广泛领域的泛化能力。

---

## 229. Adaptive Emotional Video Captioning via Affective Heterogeneous Graph Reasoning and Multi-task Joint Learning

**arXiv ID:** 2607.29045 | [PDF](https://arxiv.org/pdf/2607.29045v1)

**作者:** Junbo Wang `[一作]` (Northwestern Polytechnical University), Zhiyong Wang `[通讯]` (University of Sydney)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种自适应情绪视频字幕生成框架SAGML，通过情感异质图推理和多任务因果语言建模，实现情绪与事实的双重表达。

**💡 创新点**

创新点在于将情绪先验从硬树结构改为连续软图，利用图中类别-类别、类别-词、词-词关系实现多情绪词的柔性选择，并在语言解码器前端加入情绪分布监督。

**🔧 技术方法**

使用技术包括CLIP视觉特征、Qwen大型语言模型（LoRA微调）、图神经网络（含自注意力、相似度矩阵）、多任务损失（生成+情绪分布）。

**📊 数据集**

数据集为EmVidCap系列（EmVidCap‑S、EmVidCap‑L和全合并版），覆盖短句重写与长句自由生成两类情绪字幕。

**📈 对比分析**

在所有三个分割上与七种现有EVC方法对比，SAGML在情绪准确率、CIDEr、混合评估指标均取得显著提升（最高可达+6.7点Acc_sw、+9.0点CIDEr）。

**⚠️ 局限性**

局限性包括依赖较大语言模型导致算力需求高，图构建仍依赖词典统计，且对极少量样本或极端多情绪混合场景的泛化尚待验证。

---

## 230. From Inline Notes to Collected Commentaries: Toward Context-Preserving Organization of Exegetical Knowledge in Classical Chinese Texts

**arXiv ID:** 2607.29044 | [PDF](https://arxiv.org/pdf/2607.29044v1)

**作者:** Ke Liang `[一作]` (Hong Kong Polytechnic University), Churen Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究提出了面向古文注释的自动化编纂框架，能够在保持注释上下文依赖的同时，识别注释所对应的主文段落并将不同版本的注释进行语义聚类，构建可用于下游文献学和NLP的结构化释义知识库。

**💡 创新点**

创新点在于把收集注释编纂任务框定为NLP任务，提出两步提示链（Anchoring + Refining）结合LLM来识别多目标注释与其真正释义对象，并通过交叉来源提及聚类（Union‑Find）把多版本注释按语义相似度自动归并，首次实现对注释上下文依赖的自动保留。

**🔧 技术方法**

采用两大技术：①大语言模型（DeepSeek‑Chat、Kimi‑K2‑0905‑Preview、GPT‑5‑mini）进行提示链式信息抽取；②基于SikuRoBERTa等古汉语BERT模型生成上下文与注释嵌入，结合语义相似度阈值的Union‑Find聚类，并使用文本相似度阈值与源唯一性约束。

**📊 数据集**

实验数据集为山经（Classic of Mountains）的一段文本，采集自五个不同版本（Guo、Wu、Wang、Bi、Hao），包含主文本、注释及其分层结构；构建手工标注的“Gold”子集用于评估。

**📈 对比分析**

与传统手工编纂方法对比，评估指标为提取准确率（84%）、Jaccard（0.848）和聚类核心指标：Precision 0.993–0.995，Recall 0.953–0.967，F1 0.972–0.981，CoNLL F1 0.977，说明模型在保持上下文、聚类准确性上表现优异。

**⚠️ 局限性**

局限性在于实验仅针对山经单一文本，未验证跨文本泛化；提示链与聚类阈值对不同文本可能需要重新调参；对极端多层注释或低质量文本的鲁棒性尚待进一步验证。

---

## 231. Dynamics-aware identification of governing equations from sparse and noisy data

**arXiv ID:** 2607.29036 | [PDF](https://arxiv.org/pdf/2607.29036v1)

**作者:** Pongpisit Thanasutives `[一作]` (RIKEN), Yoshinobu Kawahara `[通讯]` (RIKEN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了一种基于Koopman的上采样技术，以提高从稀疏和噪声数据中识别控制方程的能力，特别是在应用SINDy和PDE-FIND方法之前进行预处理。

**💡 创新点**

创新点在于提出了一种动态感知的上采样方法，通过动态模式分解（DMD）及其变体来插值和去噪，从而提高导数估计的准确性。

**🔧 技术方法**

使用了动态模式分解（DMD）、扩展DMD（EDMD）和优化DMD等技术。

**📊 数据集**

使用了Lorenz-63和Van der Pol等两个常微分方程（ODE）系统，以及Burgers、Fisher-KPP和线性对流扩散等三个周期性偏微分方程（PDE）系统。

**📈 对比分析**

与线性和光滑样条插值技术进行比较，结果表明，基于Koopman的预处理方法在整体性能上优于这些非动态替代方案，尤其在系数准确性方面表现突出。

**⚠️ 局限性**

限制在于该研究基于合成数据，真实实验中可能面临更复杂的情况，如非周期性边界、强候选项共线性等，这可能需要更复杂的模型选择和正则化方法。

---

## 232. Event-Based Upper-Body Humanoid Teleoperation Under Challenging Illumination

**arXiv ID:** 2607.29227 | [PDF](https://arxiv.org/pdf/2607.29227v1)

**作者:** Haoyu Fu `[一作]` (Shanghai University), Xulei Qin `[通讯]` (Changchun University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

基于事件摄像头的实时上半身运动模仿系统，实现低延迟闭环控制

**💡 创新点**

首次将事件视觉与机器人运动再现结合，提供高动态范围、高帧率、低延迟的完整闭环管线

**🔧 技术方法**

事件累积时间面 + IMU 重力对齐、MediaPipe‑style 3D 关节回归、One‑Euro 滤波、TWIST 运动再现、NVIDIA Booster T1 嵌入式实现

**📊 数据集**

Human3.6M、DHP19（转化为事件格式）用于网络训练，12名受试者、5 次试验、4 个光照/运动条件用于评测

**📈 对比分析**

与固定曝光 RGB（30 fps/120 fps）做公平对比；事件管线在 HDR/低光/高速动作下拥有 23–34 ms 延迟、10.8 mm 运动抖动、4.9° 关节 RMSE，RGB 在光照良好静态场景下 MPJPE 低

**⚠️ 局限性**

对完全静止或极慢动作场景事件稀疏；仅限上半身、无接触操作、IMU 误差导致漂移、对低纹理服装或强自遮挡敏感

---

## 233. MirrorCraft: Paired Evaluation under Hidden Rule Changes in Minecraft

**arXiv ID:** 2607.29218 | [PDF](https://arxiv.org/pdf/2607.29218v1)

**作者:** Jianxin Gao `[一作]` (China Agricultural University), Zining Wang `[通讯]` (Tianjin University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 MirrorCraft 基准，设计 Vanilla 与 Mirror 世界配对，评估 LLM 代理在隐藏规则变更下的任务完成与进度；

**💡 创新点**

创新点在于：①使用配对世界对比隐藏规则改动；②引入 Rule Intervention Effect (RIE) 指标；③提供统一 Mineflayer 语义接口与多规则套件实验；

**🔧 技术方法**

技术手段包括：Minecraft 1.19 datapack 自动化规则改动；Mineflayer 提供技能接口；使用 Gemini、DeepSeek、ReAct、XENON、ADAM 等 LLM 代理；并通过 IRR、RL 等行为诊断；

**📊 数据集**

数据集：10 个 Vanilla 世界（5 种生物群系 × 2），每个生成 6 个 Mirror 副本，覆盖 6 个规则套件（M01–M06）和 3 个进程任务（铁甲、钻石、附魔），共 60 个配对；

**📈 对比分析**

比较方法：在 8,640 条目中评估 6 种配置 × 2 模型，使用 Score、SR 与 RIE_SC/RIE_SR 统计；表现为 RIE_SC 平均 +3.6，RIE_SR +7.7；ReAct 在无规则描述下获得最高 Mirror Score 86.8 与 SR 53%；规则披露提高平均得分但仍未完成深度任务；

**⚠️ 局限性**

局限性：RIE 与 Mirror 评分可能不一致，无法单独衡量规则适应性；规则变更难度非统一；动作预算不足导致深度任务完成率低；评估仅在 Mineflayer 语义接口层，未覆盖像素感知或低级控制。

---

## 234. Memory Provenance Laundering in LLM Agents: A Non-Amplification Firewall for Persistent Memory

**arXiv ID:** 2607.29167 | [PDF](https://arxiv.org/pdf/2607.29167v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 235. MROPE: A Multi-Robot Safe Cooperative Strategy via combined Predictive Safety Filters and Ellipse-based Constraint Compression

**arXiv ID:** 2607.29203 | [PDF](https://arxiv.org/pdf/2607.29203v1)

**作者:** Alice Rosetti `[一作]`, Giuseppe Notarstefano `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种分层分布式预测控制策略，用于在障碍物稠密环境中实现无人机群监测移动目标并保证集体安全。

**💡 创新点**

创新点在于：将高层聚合优化与低层安全约束解耦；引入分布式椭圆压缩算法将复杂障碍约束压缩为单一安全椭圆；以及使用分布式预测安全过滤器（基于MPC）实现实时避障。

**🔧 技术方法**

采用的技术包括：分布式聚合优化算法、分布式椭圆一致性协议、基于MPC的预测安全过滤器、非线性几何控制器、ROS 2分布式通信框架。

**📊 数据集**

使用的数据集为：Webots虚拟森林场景（含数十棵树）进行Monte Carlo仿真；以及在实验室9×4×3 m室内场景下，用四架Crazyflie nano-quadrotor和一台TurtleBot 3 Burger进行实地实验。

**📈 对比分析**

与传统集中式PSF基线进行对比，实验表明：集中式方案随无人机数量指数增长计算时间（8架时约400 ms，超出7 Hz实时阈值），而分布式方案在8架时仍保持<40 ms；跟踪误差在分布式方案下保持≈10 cm，集中式约50 cm；在不同目标速度和群规模下均实现100 %成功率（≤0.21 m/s）。

**⚠️ 局限性**

局限性在于：缺乏全局邻居未来轨迹信息，导致在高速目标或极度拥挤场景下可行域受限；当前仅在仿真/室内实验验证，缺少完整理论递归可行性证明；且对障碍感知和目标状态估计的依赖未在本文中完整解决。

---

## 236. Locally Consistent Transductive Information Maximization for Few-Shot Remote Sensing Scene Classification

**arXiv ID:** 2607.29192 | [PDF](https://arxiv.org/pdf/2607.29192v1)

**作者:** Karim El Khoury `[一作]` (UCLouvain), Christophe De Vleeschouwer `[通讯]` (UCLouvain)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于转导信息最大化（TIM++）的本地一致性正则化方法LC‑TIM，专门用于遥感场景的少样本分类；

**💡 创新点**

创新点在于在TIM++目标中加入邻域一致性约束，并设计多源融合邻接图（结合CLIP与DINOv3），显著提升低样本环境下的性能；

**🔧 技术方法**

核心技术包括冻结的视觉‑文本模型（CLIP/GeoRSCLIP）、自监督遥感编码器（DINOv3）、转导信息最大化、KL正则化、kNN邻居一致性以及多源图融合；

**📊 数据集**

使用十个遥感数据集（AID、EuroSAT、MLRSNet、OPTIMAL31、PatternNet、RESISC45、RSC11、RSICB128、RSICB256、WHURS19）进行实验；

**📈 对比分析**

与零样本CLIP/GeoRSCLIP、LP++、TransCLIP和TIM++对比，LC‑TIM在所有数据集与1–16-shot设置下均超越TIM++，尤其在1、2-shot时提升约3–8%准确率，整体平均准确率达到94.2%；

**⚠️ 局限性**

局限性包括对邻居数k与正则系数λ_LC的选择仍需经验调优，且在极少样本或极大查询集时仍可能受邻居质量影响；

---

## 237. Quick Build, Careful Check? Generative AI Use in Hackathons

**arXiv ID:** 2607.29178 | [PDF](https://arxiv.org/pdf/2607.29178v1)

**作者:** Wangyiyao Zhou `[一作]` (Eindhoven University of Technology), Alexander Nolte `[通讯]` (Eindhoven University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过半结构化访谈研究在两天AI主题Hackathon中团队如何使用和验证生成式AI工具，探讨使用场景、工具组合及时间压力下的验证实践。

**💡 创新点**

发现Hackathon参与者在无正式规则的情况下自发形成“检查AI输出”习惯，并揭示时间压力与领域知识缺乏如何限制验证能力。

**🔧 技术方法**

采用定性研究方法——主题分析（Thematic Analysis）与ATLAS.ti软件对访谈文本进行编码，并使用ChatGPT、Claude、GitHub Copilot等生成式AI工具。

**📊 数据集**

本研究无外部数据集，所有数据来自四名参与者在Hackathon中的访谈记录。

**📈 对比分析**

本研究为探索性案例研究，没有性能对比，主要通过访谈内容归纳提炼共性与差异。

**⚠️ 局限性**

样本量仅四人，单团队单成员访谈可能缺乏代表性；仅基于自述，未能观察实际行为；数据由单一研究者编码，存在主观偏差。

---

## 238. M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain Benchmark for Full-Duplex Spoken Dialogue Models

**arXiv ID:** 2607.29125 | [PDF](https://arxiv.org/pdf/2607.29125v1)

**作者:** Ryo Fukuda `[一作]` (NTT, Inc.), Yuya Chiba `[通讯]` (NTT, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了M3-DuplexBench，一个多轮、多语种、多域的全双工语音对话系统评测基准，涵盖英语与日语、闲聊与问答两大对话域，支持多种上下文条件的评测；

**💡 创新点**

创新点在于：①实现了受控多轮评测的三种上下文条件（无、仅用户、全上下文），②通过教师强制推理生成完整对话历史，③在同一问答数据下生成中日双语合成语音对话，构建跨语种对照；

**🔧 技术方法**

采用了事件级别评测框架，使用 Whisper ASR + Montreal Forced Aligner 对生成语音进行时间戳对齐，评估 TOR、延迟、停止延迟等时序指标；内容方面利用 GPT-5 Nano 进行 LLM-as-a-Judge 评估；

**📊 数据集**

使用了英文闲聊数据 Candor、日语闲聊数据 MagicData、英文问答数据 TopiOCQA 及其日语翻译版本，随后通过 TTS 与统计化时间轴生成合成语音；

**📈 对比分析**

比较方法：对 Moshi、PersonaPlex（英）及 J‑Moshi、LLM‑jp‑Moshi（日）在三种上下文条件下进行统一评测，结果显示全上下文条件提升时序性能并改善内容一致性；英日模型在内容上存在明显差距，日语模型在时序上表现更好；

**⚠️ 局限性**

局限性：①仅支持可直接进行并行声源条件的 Moshi 族模型，无法评测其他架构；②全上下文条件依赖教师强制推理，可能与自然推理轨迹不符；③对日语模型的内容理解仍差，需进一步改进；

---

## 239. Analysis of Memory-Runtime Trade-offs in Caching Strategies for Genetic Programming Symbolic Regression

**arXiv ID:** 2607.29116 | [PDF](https://arxiv.org/pdf/2607.29116v1)

**作者:** Jiaming Shi `[一作]` (National University of Singapore), Mehul Motani `[通讯]` (National University of Singapore)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在GPSR中引入多种缓存机制以加速适应度评估。

**💡 创新点**

创新点在于系统地比较多种缓存策略（LRU、FIFO、LFU、RR）和缓存大小，并提出RAM-hour指标衡量内存-时间权衡。

**🔧 技术方法**

使用Python gplearn库结合cachetools实现缓存，并在Google Cloud环境下进行实验。

**📊 数据集**

实验数据来自PMLB的三个真实数据集（344_mv、215_2dplanes、1203_BNG_pwLinear）以及一个合成数据集。

**📈 对比分析**

通过对运行时间、执行函数耗时、内存占用及RAM-hour的多维度比较，发现LRU/FIFO在合适缓存大小下可将执行时间提升到十倍以上，同时保持内存占用可控；LFU和RR效果差。

**⚠️ 局限性**

局限在于实验仅覆盖有限数据集与单一GP实现，且对缓存清理策略的探索有限，未考虑更复杂的GP变体或大规模并行环境。

---

## 240. Frugal Bayesian Optimization: Scalable Surrogates for Data- and Resource-Limited Discovery

**arXiv ID:** 2607.29225 | [PDF](https://arxiv.org/pdf/2607.29225v1)

**作者:** Panagiotis Krokidas `[一作]` (National Centre for Scientific Research Demokritos), George Giannakopoulos `[通讯]` (National Centre for Scientific Research Demokritos)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Frugal Bayesian Optimization 的框架，设计了一种可扩展的轻量级代理模型 BASS，用于在数据量和计算资源有限的情况下进行高效优化。

**💡 创新点**

创新点在于：①将 Bayesian Optimization 与可扩展代理模型 BASS 结合，显著降低对高维空间和大规模训练数据的需求；②引入 3cAUC 指标同时评估最佳解与前 100 解决方案的发现速度；③系统性地将 BASS 与传统 Gaussian Process、NGBoost、Random Forest 进行对比，验证其在多种测试场景下的优越性。

**🔧 技术方法**

使用的技术包括 Bayesian Optimization、BASS 代理模型、Gaussian Process、NGBoost、Random Forest；对 benchmark 函数和真实案例数据集执行 3cAUC 评估；在 Windows 11 + Intel Core i9-10900K + NVIDIA RTX 3070 Ti GPU 上实现计算。

**📊 数据集**

数据集：
- benchmark 函数：expschaffer6、rastrigin、Michalewicz、Ackley、Schwefel、Styblinski、Weierstrass、Schaffer7；
- 真实案例数据集：Moire、pressure_vessel、MOFs、proteins、lunar、worm、neural、QM9、multi_PZT。

**📈 对比分析**

通过 3cAUC（time 与 samples 两维）与传统代理模型进行对比，结果显示 BASS 在大多数函数与真实案例上均获得更高的 AUC，尤其在前 100 方案获取方面表现突出，证明其在资源受限场景下的高效性。

**⚠️ 局限性**

限制：
1. 仅评估了四种代理模型，未覆盖更复杂或更高维的模型；
2. 对 BASS 的超参数调优与收敛性质的理论分析不足；
3. 计算环境与硬件资源有限，可能影响结果可复现性；
4. 在极高维或大规模数据集上的可扩展性与鲁棒性尚未充分验证。

---

## 241. Multi-Modal Object Re-Identification with Dual Semantic Guidance and Global-Local Mutual Modulation

**arXiv ID:** 2607.29207 | [PDF](https://arxiv.org/pdf/2607.29207v1)

**作者:** Weixiang Zhou `[一作]` (Dalian University of Technology), Jinshan Pan `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于双重语义引导的多模态目标再识别框架。

**💡 创新点**

创新点在于同时利用统一文本描述和软语义掩模进行全局-局部互调，采用超图交互和分层 MoE 进行动态融合。

**🔧 技术方法**

使用多模态大语言模型生成文本、SAM2 生成软掩模、CLIP 视觉编码器、超图网络、Masked Global-Local Modulator 与 Hierarchical MoE Fusion 等技术。

**📊 数据集**

在 RGBNT201、RGBNT100 与 MSVR310 三个公开多模态 ReID 数据集上进行实验。

**📈 对比分析**

与现有多模态 ReID 方法相比，在 mAP 与 Rank‑1 等指标上均取得领先，尤其在 RGBNT201 上 mAP 82.6%、Rank‑1 87.0%。

**⚠️ 局限性**

局限在于依赖离线生成的文本与掩模，且对 MLLM 质量敏感，同时训练仍需较高计算资源。

---

## 242. Alignment Is Local: A Paired Diagnostic for GUI Agents under User-Side Persuasion

**arXiv ID:** 2607.29199 | [PDF](https://arxiv.org/pdf/2607.29199v1)

**作者:** Haoxin An `[一作]` (Xi'an Jiaotong University), Chao Shen `[通讯]` (Xi'an Jiaotong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并评估了一种针对 GUI 代理的局部对齐诊断方法 AXIS，检验单句防护在多轮对话和低显著性请求下的有效性。

**💡 创新点**

提出局部对齐框架，揭示提示级防护仅在单句显式请求上有效，且在多轮对话或隐蔽请求中失效，强调需按显著性和轮次条件报告安全性。

**🔧 技术方法**

使用对话模拟、屏幕截图关联、Gemini 3.1 Pro 的提示重写与自动判断、手工审查，以及防护条件的安全指令。

**📊 数据集**

构建包含 43 个有害屏幕定位案例和 23 个无害对照场景的自制数据集，覆盖隐私外泄、过度授权、保护绕过等风险。

**📈 对比分析**

与三款前沿 GUI 代理（Qwen3.7‑Plus、Claude Opus 4.8、GPT‑5.6 Sol）对比，单句防护可将 ASR 降至 14–37% 以上，且低误拒率；但多轮链 ASR 上升约 20%，凸显安全性大幅下降。

**⚠️ 局限性**

局限性包括仅评估预执行响应而非实际执行、判定依赖自动裁判且为单一攻击者、显著性与直接性混杂、仅测试三款模型且 Qwen 局部上限受限。

---

## 243. Knox: Fortifying Smart Spaces With Safety Guarantees

**arXiv ID:** 2607.29198 | [PDF](https://arxiv.org/pdf/2607.29198v1)

**作者:** Rishabh Menezes `[一作]` (University of Illinois at Urbana-Champaign), Indranil Gupta `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Knox 系统，允许用户使用 Safron 语法在智能空间中声明安全规则，并通过静态算法在多线程并发 routine 运行前检查安全性。

**💡 创新点**

创新点包括：① 设计了新的可扩展安全语法 Safron；② 开发了零误报的单/并发 routine 静态检查算法；③ 引入 Attack/Defense 与 Wall/Siege 两种优化显著提升运行时性能。

**🔧 技术方法**

主要技术包括 SMT 求解、符号执行、树形解析、交错分析、子树剪枝与组合优化；实现基于 C++ 并与 HomeAssistant 集成。

**📊 数据集**

评估使用合成数据、真实 20-30 设备配置、HomeAssistant 生成的房间仿真数据以及 2011 条 TapChecker 基准工作负载。

**📈 对比分析**

通过与暴力枚举和 TapChecker 对比，Knox 在简单负载下提升 3102×，复杂负载提升 277–4717×；在真实房间实验中速度比 TapChecker 快 22.76–291.45×；准确率>96%，误报率<4%，无漏报。

**⚠️ 局限性**

局限性：仅覆盖静态检查，未处理运行时动态错误；极大并发或安全树规模仍存在可扩展性挑战；误报率虽低但仍存在；未考虑设备或 hub 故障、网络延迟等实际因素。

---

## 244. Detecting Experiential Intertextuality Across Migration Routes: Beyond Surface Similarity in French Narratives

**arXiv ID:** 2607.29188 | [PDF](https://arxiv.org/pdf/2607.29188v1)

**作者:** Sakayo Toadoum Sari `[一作]` (University of Artois), Fabien Delorme `[通讯]` (University of Artois)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了无监督的迁移叙事经验互文性检测任务，自动生成句对并评估其相似性

**💡 创新点**

首次提出经验互文性概念及其三层级（主题、功能、语用），并在无标签数据上实现自动检测

**🔧 技术方法**

使用词汇基线、POS 结构、上下文感知特征、句子嵌入、多模 LLM（Qwen2.5‑7B、Mistral‑7B）以及混合 Ridge 回归

**📊 数据集**

108 篇法语迁移叙事（跨撒哈拉和巴尔干通道），共 5,922 句，随机抽取 816 句对供专家评标

**📈 对比分析**

所有无监督方法与专家评分相关性最高为 0.38（Qwen2.5 零样本），混合模型达到 0.45，约 68% 的噪声上限，表明 LLM 特征最有价值，传统嵌入与 POS 贡献有限

**⚠️ 局限性**

数据量小、通道不平衡，专家一致性低导致噪声上限有限；未考虑更深层事件结构与语义角色；仅评估 7B LLM，量化效应未剖析；未对法语形态学进行归一化

---

## 245. Implicit Machine Learning Force Fields Accelerate Molecular Dynamics Simulations

**arXiv ID:** 2607.29158 | [PDF](https://arxiv.org/pdf/2607.29158v1)

**作者:** Johannes Maeß `[一作]` (BIFOLD), Stefan Chmiela `[通讯]` (BIFOLD)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种隐式机器学习力场（I-MLFF），通过自洽的固定点方程取代显式神经网络层，实现分子动力学中间表示的重用，从而显著降低计算和内存占用。

**💡 创新点**

创新点在于将力场推理视为自洽固定点求解，并利用时间连续性进行warm‑start和显式微分，保持原始时间步与原子分辨率的前提下实现2‑5倍速度提升与内存缩减。

**🔧 技术方法**

采用深度平衡模型（Deep Equilibrium Models）实现固定点求解，使用隐式微分获得能量守恒的力；在SchNet、PaiNN、SO3net等三种GNN架构中加入输入注入、规范化、固定点求解器、warm‑start逻辑和正则化损失。

**📊 数据集**

在MD17（10个小分子）和MD22（7个大分子）两个基准数据集上进行训练和评估。

**📈 对比分析**

与对应深度显式MLFF进行对比，I-MLFF使用单层迭代至收敛即可匹配显式模型的力精度；在相同GPU内存和计算预算下实现约2‑5倍的速度提升，并显著降低内存占用。

**⚠️ 局限性**

主要局限包括固定点求解对初始猜测敏感、收敛性依赖、warm‑start效果在高频事件（如键断）中可能下降；当前方法仍依赖距离截断和分子几何特征，可能不适用于包含长程相互作用的系统。

---

## 246. Progressive Decision-Making for Localizing Open-Ended AI-Generated Image Forgeries

**arXiv ID:** 2607.29156 | [PDF](https://arxiv.org/pdf/2607.29156v1)

**作者:** Jingyi Hou `[一作]` (University of Science and Technology Beijing), Zhijie Liu `[通讯]` (University of Science and Technology Beijing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究提出一种基于进化决策的AI生成图像伪造局部化方法，通过将伪造掩模视为可迭代状态并逐步更新，实现更精确的局部化。

**💡 创新点**

创新点在于将最终定位视为顺序决策更新流程，设计了轻量级的决策证据投影器与基于Mamba的证据引导模块，并结合不确定性与边界先验实现可靠的状态更新。

**🔧 技术方法**

采用轻量级卷积投影、Mamba状态空间模型、熵不确定性估计、梯度差异边界约束等技术，并通过多步深度监督进行训练。

**📊 数据集**

在Protocol‑CAT训练集上训练，使用NIST16、CASIAv1、AutoSplice、SAGI‑D‑9K等数据集进行评估。

**📈 对比分析**

与多种基准方法相比，在传统和AI生成伪造测试集上均取得最高F1分，尤其在AI生成数据上提升超过10%，证明了方法的优越性。

**⚠️ 局限性**

限制包括更新步数固定为两步，模型仍受限于训练集多样性，对极为微弱或全新伪造模式的鲁棒性尚待进一步验证。

---

## 247. On the Efficacy of Self-Supervised Point Cloud Encoders for Efficient 3D Large Language Models

**arXiv ID:** 2607.29136 | [PDF](https://arxiv.org/pdf/2607.29136v1)

**作者:** Yao Zheng `[一作]` (Beijing University of Posts and Telecommunications), Tian Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在MiniGPT-3D的四阶段流水线中，系统地评估了七种点云编码器初始化/预训练配置（包含多模态基线、PCP-MAE、Point-MAE、随机初始化等），并分别在冻结和解冻两种微调策略下，探究了两种编码器架构（MaskTransformer 与 PointTransformer）、两种自监督目标（PCP-MAE 与 Point-MAE）以及两种预训练数据集（Objaverse 660K 与 ShapeNet55-34 约50K）的影响；通过开放词汇分类、3D captioning 及 ModelNet40 闭集分类等任务，量化了编码器表现。

**💡 创新点**

①发现 MiniGPT-3D 训练流程对编码器质量鲁棒，随机初始化的编码器在解冻后即可逼近预训练编码器的性能；②揭示架构与预训练目标存在强交叉效应，PCP-MAE+MaskTransformer 在未冻结时获得 59% 的开放词汇准确率，远高于其他组合；③表明几何自监督在闭集分类任务上无法替代多模态预训练，表现仅停留在 13–18%。

**🔧 技术方法**

采用 MiniGPT-3D 四阶段训练策略；PCP-MAE 与 Point-MAE 的 masked autoencoding 预训练；MaskTransformer（交叉注意力）与 PointTransformer（全局自注意力）两种编码器架构；LoRA 与 Norm 微调；Qwen-Flash API 作为评价 LLM 生成结果的自动化工具。

**📊 数据集**

预训练使用 Objaverse 660K（含 8,192 点 × 6 维特征）与 ShapeNet55-34 约 50K（无颜色）；评估使用 Objaverse 200 样本（开放词汇分类、captioning）以及 ModelNet40 2,468 样本（闭集分类）。

**📈 对比分析**

通过与多模态基线（ULIP-2 预训练的 Point-BERT）以及各自的随机初始化进行对比，发现：1) 在开放词汇分类中，随机初始化+解冻可达到 52.5%（与 PCP-MAE+PointTransformer 相同），PCP-MAE+MaskTransformer 进一步提升至 59%；2) 在 3D captioning 中，随机初始化 44.45 分，PCP-MAE+MaskTransformer 49.63 分，均接近基线 53.37 分；3) 在 ModelNet40 闭集分类中，所有自监督模型仅达 13–18% 的准确率，而基线保持在 61–64%。

**⚠️ 局限性**

研究仅覆盖 MiniGPT-3D 这类高效 3D-LLM 体系，未考虑更大 LLM 或更深层次的自监督训练；预训练周期相对较短（24–71 轮）且未达到最大 300 轮；评估采用 Qwen-Flash API 可能与 GPT‑4 的评估结果有差异；仅评估对象级点云，未扩展至场景级或不同数据分布。

---

## 248. SciFigPlag-Bench: A Benchmark for Provenance-Aware Scientific Figure Plagiarism Detection

**arXiv ID:** 2607.29124 | [PDF](https://arxiv.org/pdf/2607.29124v1)

**作者:** Zhiying Cui `[一作]` (Ningbo University), Pengyuan Li `[通讯]` (IBM Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文构建了名为SciFigPlag-Bench的多任务基准，用以评估科学论文中图形抄袭的来源追溯、类型识别与局部对应定位。

**💡 创新点**

创新点在于：① 提出两层层级的抄袭类型词汇表（内容层与变形层），实现“是什么被抄袭”与“如何被改动”分离；② 设计混合式数据集，结合真实案例、按词汇表生成的合成样本以及视觉相似负样本；③ 定义四个诊断任务，系统化地考察模型在检测、归因、类型推理和定位四个维度的能力。

**🔧 技术方法**

使用多种视觉‑语言模型（InternVL、Qwen、Gemma、Granite、Gemini、Claude、GPT‑5等）并以统一提示和温度0进行推理；对模型在每个任务上计算准确率、精确率、召回率、F1、IoU等指标。

**📊 数据集**

数据集包含15,348张图像，2,582个正样本对与2,541个负样本对，来源包括VroniPlag Wiki、Mendeley 论文抄袭语料、PubMed OA、ChartNet 等，分别用于构造完整图、子图、数据重绘与结构重绘等四类抄袭场景。

**📈 对比分析**

通过在四个任务上对比模型，发现闭源旗舰模型（Gemini‑3‑Flash、Claude‑Sonnet‑4.6、GPT‑5.4）总体表现最优（整体分数>80%），大型开放权重模型（Gemma‑4‑31B、Qwen‑3.6‑35B‑A3B等）接近；小/中型模型普遍落后，尤其在细粒度变形识别与空间定位上表现不佳。

**⚠️ 局限性**

局限性：① 模型在层B（变形）识别和复合标签推断时准确率显著下降；② 位置定位任务对坐标回归的依赖导致大多数模型IoU低于0.5；③ 合成样本在某些高阶结构重绘上仍无法完全覆盖真实复杂场景；④ 负样本的视觉相似性挑战模型区分真实抄袭与巧合的能力。

---

## 249. Hy-MultiTurn: A Six-Dimensional Benchmark for Deep Multi-Turn Dialogue Understanding

**arXiv ID:** 2607.29196 | [PDF](https://arxiv.org/pdf/2607.29196v1)

**作者:** Eileen Ye `[一作]` (Tencent), Maxm Pan `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于 209 个 12–76 轮中文对话的深度多轮对话理解基准，涵盖六个失败机制维度，并通过受控生成与真值可追踪实现可复现评测。

**💡 创新点**

①失败机制优先设计，将多轮失效归纳为六个可测量维度；②采用受控干扰与可追踪真值的任务生成管道，解决传统评测的可复现性与可解释性问题；③引入重要性和严格准确率双指标，细粒度诊断模型弱点；④在同一基准上评估 22 个不同模型配置，揭示跨维度表现差异。

**🔧 技术方法**

使用受控对话生成管道（机制标注、真值构造、任务生成、评分项设计、干扰注入、模型评测）、判分技术（deterministic 检查器 + LLM 判断）、三次 roll 平均、API 成本估算等技术手段。

**📊 数据集**

首先从真实中文聊天记录中提取并归纳六种失败机制，然后基于这些机制手工设计 209 个受控对话任务，所有对话均为人工重构或生成，无真实用户内容。

**📈 对比分析**

采用 22 种模型配置（含 GPT‑5.5、Grok‑4.5、Claude Opus、Gemini、Qwen、Hy3 等）进行三轮评测，计算重要性（平均 77.2%）和严格准确率（最高 41.1%）。GPT‑5.5 在重要性上领跑，Grok‑4.5 与之接近；不同模型在六个维度上各有优势，表明单一指标难以体现完整能力。

**⚠️ 局限性**

局限：仅覆盖中文，难以直接推广至其他语言；任务规模为 209 例，难以做极细粒度排序；判分依赖 LLM 评判，仍可能存在主观偏差；评测成本随对话长度、模型数与 roll 次数增加而上升；基准仅评估单一机制，未涵盖交互式工具调用等更复杂场景。

---

## 250. Learning Latent Reasoning Traces for Scalar Reward Models End-to-End

**arXiv ID:** 2607.29185 | [PDF](https://arxiv.org/pdf/2607.29185v1)

**作者:** Sanwoo Lee `[一作]` (Peking University), Yunfang Wu `[通讯]` (Peking University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LatentRM框架，将推理过程视为离散潜变量，并在单一目标下联合训练生成器与标量奖励模型，以提升奖励学习的准确性；

**💡 创新点**

创新点在于通过潜变量模型的ELBO对奖励和推理共训练，避免了多任务学习中奖励与推理目标不一致的问题，并通过on‑policy更新实现生成器与标量RM的紧密耦合；

**🔧 技术方法**

采用条件生成模型（LatentRM）与Plackett‑Luce排名模型，使用REINFORCE对生成器进行策略梯度优化，同时对标量RM采用监督学习；

**📊 数据集**

使用Qwen3‑4B‑Instruct作为后端，训练数据来自UltraFeedback、OpenMathReasoning、HelpSteer3、WildGuard和OffsetBias五个多领域数据集，随后在RM‑Bench和PPE Correctness等OOD基准上进行评估；

**📈 对比分析**

与传统标量RM、生成式RM和多任务RM做对比，LatentRM在ID测试集上在对数似然和Kendall’s τ上均优于基线，在OOD RM‑Bench和PPE Correctness上的平均准确率分别提升至82.8%和72.1%，在RLHF实验中亦实现了最高的长度控制胜率；

**⚠️ 局限性**

局限性包括对训练数据分布的依赖（如对UltraFeedback和OpenMathReasoning的偏重导致安全/对抗域表现略逊），以及潜变量推理生成成本相对较高，且模型对低频域的泛化仍有待提升。

---

## 251. SERUM: State Extraction and Refinement for User Modeling

**arXiv ID:** 2607.29181 | [PDF](https://arxiv.org/pdf/2607.29181v1)

**作者:** Andy J. Phu `[一作]` (University of Minnesota), Dongyeop Kang `[通讯]` (University of Minnesota)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出一种多轮视觉‑语言模型框架 SERUM，能够从无注释的第一人称屏幕录像中自动提取行为与意图的有限状态机模型。

**💡 创新点**

创新点在于将活动识别与意图推断交替迭代，利用累积上下文消除 VLM 幻觉和时间混淆，并在每轮后对同义标签进行语义归并，最终实现“方案平衡”（schematic equilibrium）并获得可解释的流程模型。

**🔧 技术方法**

使用的技术包括 Qwen3‑VL‑8B‑Instruct 视觉‑语言模型、基于句子嵌入的 Sentence‑BERT 同义归并、滑动窗口的运行长度编码（RLE）上下文、以及一阶马尔可夫链进行行为预测。

**📊 数据集**

数据集为 61 段四个领域（编码、烹饪、运动、日常）的 YouTube egocentric 视频，合计 11,125 帧，平均时长 15.5 小时。

**📈 对比分析**

与多数频率基线（Majority、Weighted Random、Uniform）以及未归并的原始标签相比，归并后的一阶马尔可夫模型在下一步行为预测上准确率提升约 12‑20%，perplexity 显著降低；在人工评估中，最终标签准确率达 88.3%，优于首次标签 82.8%。

**⚠️ 局限性**

主要局限在于模型推理延迟高、仅在离线设置下验证、对视频线性无循环内容时性能受限、以及依赖 VLM 的幻觉仍可能影响标签质量。

---

## 252. MERIT: Efficient In-Place Deletion for Dynamic Graph-Based Approximate Nearest Neighbor Indexes

**arXiv ID:** 2607.29173 | [PDF](https://arxiv.org/pdf/2607.29173v1)

**作者:** Zekai Wu `[一作]` (Tongji University), Heng Tao Shen `[通讯]` (Tongji University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种高效的原地删除框架 MERIT，用于动态图形近似最近邻索引，解决删除导致的高延迟和召回下降问题。

**💡 创新点**

创新点在于将删除拆分为逻辑无效化+受影响邻居恢复、k_r‑MST 局部修复、以及基于目标版本的边无效化，避免显式逆邻维护，显著降低删除成本并保持或提升召回。

**🔧 技术方法**

采用有界搜索采集受影响节点、k_r‑MST 生成多路修复边、版本化边记录实现即时无效化、逻辑无效化与物理移除分离等技术，并在 HNSW 与 Vamana 两种图结构上实现。

**📊 数据集**

使用多种公开基准数据集：Sift1M、Gist1M、Deep1M、Deep10M、Deep100M、GloVe、MSong 等，涵盖图像、文本、音乐等高维向量。

**📈 对比分析**

与 Wolverine、FreshVamana、IP‑Vamana 等现有 SOTA 方法在相同图结构下对比，MERIT 的删除延迟比现有方法快 3–19 倍，召回稳定甚至提升，更新吞吐率与插入接近，QPS 与 NDC 维持在高水平。

**⚠️ 局限性**

局限性包括：需要调节 k_r 参数以平衡修复成本与质量；版本号有限，理论上存在 ABA 风险；在极端高频删除或极大规模动态变化场景下仍需进一步验证。

---

## 253. Approximation Algorithms for Geometric Maximum Coverage

**arXiv ID:** 2607.29160 | [PDF](https://arxiv.org/pdf/2607.29160v1)

**作者:** Sujoy Bhore `[一作]` (IIT Bombay), Pasin Manurangsi `[通讯]` (Google Research)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一系列改进的多项式时间近似算法，能够在许多几何集合系统中以 1-1/e+Ω(1) 的比例逼近最大覆盖问题，同时给出了连续版的 EPTAS 以及针对小 k 的参数化近似方案。

**💡 创新点**

创新点在于构建了一个通用的黑盒化简框架，将最大覆盖问题的 LP 取整与离散独立集问题相结合，从而在保持 LP 取整性能的同时突破 1-1/e 的传统界限；此外，还在连续体积选择问题上首次实现了针对胖凸体的 EPTAS，并提供了更紧凑的参数化运行时间。

**🔧 技术方法**

技术上主要使用了浅单元复杂度（shallow cell complexity）分析、LP 取整与贪心的组合、分治与动态规划的四分树（quadtree）技术、以及经典的稠密/稀疏化（shifting）与集合系统的 VC 维度近似。

**📊 数据集**

本文并未使用实际数据集，所有结果均为理论分析与构造性证明，针对的是任意给定的几何对象集合。

**📈 对比分析**

与已有的 1-1/e 或 O(log n) 近似算法相比，本文的算法在满足线性浅单元复杂度或可分解为常数个此类集合的情况下实现了常数级的显著改进；对小 k 的方案将之前的 2^O(k^2/5) 复杂度降至 2^O(k log k)，而连续版的 EPTAS 在常数维度下提供了接近最优的解。

**⚠️ 局限性**

主要限制包括：改进幅度仅为常数项，无法在所有几何类上超越 1-1/e；对高维问题仍缺乏多项式时间算法；连续体积选择的 PTAS 仅在维数固定且对象胖度有限时有效；且硬件证明部分对输入规模增长仍保持高复杂度。

---

## 254. PluRel-to-RDB-PFN: Schema-Guided Synthetic Relational Pretraining

**arXiv ID:** 2607.29129 | [PDF](https://arxiv.org/pdf/2607.29129v1)

**作者:** Mohammad Sadeq Abolhasani `[一作]` (SAP Labs, LLC), Viswanath Ganapathy `[通讯]` (SAP Labs, LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个将 PluRel 生成的合成关系数据库转换为 RDB-PFN 训练格式的完整管道，并在此基础上设计与评估了三种不同的课程学习策略，用仅 33K 个任务恢复了大部分原始 RDB-PFN 的性能。

**💡 创新点**

①首次实现外部合成生成器与关系基础模型的解耦；②提出了基于真实 schema 的课程先导方法，并证明早期接触真实 schema 对模型学习最为关键；③在显著减少训练规模的前提下，取得接近原始模型 90% 以上的性能。

**🔧 技术方法**

使用 PluRel 生成合成数据库，外部二分类任务构造、DBInfer 语义导出、DFS 线性化、Transformer（RDB-PFN 6 层）以及 Schedule-Free AdamW 进行训练；实验中采用 64/1024 级上下文评估。

**📊 数据集**

主要使用 5,500 个合成数据库（约 33K 个任务）作为训练数据，评估集为 19 个真实世界关系预测任务（RelBench/DBInfer）；对比原始 RDB-PFN 使用的约 1.8M 任务。

**📈 对比分析**

通过与原始 RDB-PFN 的 ROC-AUC 对比，最优课程在 64-shot、1024-shot 下分别得到 0.6116/0.6346，恢复了 93.8%/87.6% 的性能；相较于单一数据池或不同课程顺序，发现课程顺序对性能影响显著。

**⚠️ 局限性**

①训练规模与原始模型不匹配（仅 55 倍）；②未包含单表预热阶段；③PluRel 仅支持整数/浮点，导致某些特征近似；④任务构造采用启发式方法；⑤评估仅限二分类任务，未验证回归或多分类。

---

## 255. Curriculum Matters: Data-Efficient Relational PFN Pretraining with Synthetic Data

**arXiv ID:** 2607.29120 | [PDF](https://arxiv.org/pdf/2607.29120v1)

**作者:** Mohammad Sadeq Abolhasani `[一作]` (SAP Labs, LLC), Viswanath Ganapathy `[通讯]` (SAP Labs, LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过使用一种结构完全不同的合成数据库生成器，对关系 Prior-Data Fitted Network（PFN）的预训练进行系统评估，探究合成数据的来源、课程顺序以及单表预训练能否为后续关系推理奠定基础。

**💡 创新点**

创新点在于：①证明课程设计（宽度递增的单表课程、关系两阶段课程）对性能的影响远大于合成生成器本身或数据量；②单表宽度递增课程在仅使用约13,300个单表数据（相较原始6万的规模缩小45倍）即可恢复原始研究中88% 的性能；③单表预训练模型即使不见多表关系数据库，也能在关系任务上获得与完整关系课程相当的表现；④不同合成生成器（随机图+块模型+SCM）可替代原生生成器，仍能取得接近原始结果。

**🔧 技术方法**

采用了 PFN 结构（双向 Transformer + 1/2 层聚合）、Deep Feature Synthesis 线性化、两阶段预训练流程、宽度递增课程、以及多种实验族（Family A–G）对比分析。

**📊 数据集**

数据集：通过新的合成生成器产生的 13,300 个单表（每表 150 行）和 5,500 个关系数据库；基准任务包括 23 个单表分类任务（Grinsztajn 等）和 19 个关系任务（RelBench/4DBInfer）。

**📈 对比分析**

比较方法：在相同数据量下对比课程顺序 vs. all‑at‑once，单表课程平均 ROC‑AUC 0.703 对比 0.541；关系课程平均 ROC‑AUC 0.638 对比原始 0.725；单表模型直接评估在关系任务上得到 0.631，几乎匹配 0.638；通过不同上下文长度（64/1024）进一步验证。全部结果均以原始论文的性能为基准。

**⚠️ 局限性**

局限性：①实验仅使用一种合成生成器，未探讨其它生成策略的效果；②单表与关系任务之间的方差仍高，缺乏任务结构感知的课程设计；③在关系任务中仍略逊于最优关系课程；④未验证在真实数据库上的迁移性能；⑤对大规模模型或不同硬件环境的可扩展性未做评估。

---

## 256. Linear Proposal Operators and Stochastic Search Geometry in SOMA and Differential Evolution

**arXiv ID:** 2607.29228 | [PDF](https://arxiv.org/pdf/2607.29228v1)

**作者:** Vojtěch Novák `[一作]` (VSB - Technical University of Ostrava), Ivan Zelinka `[通讯]` (VSB - Technical University of Ostrava)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将候选生成与修复/选择分离的算子框架，对SOMA和DE的变异算子进行线性解析，并利用得到的几何量设计几何控制与旋转感知SOMA变体；

**💡 创新点**

创新点在于将变异算子精确表示为线性或仿射映射，推导其期望、协方差等概率几何，并以此指导算子设计；

**🔧 技术方法**

采用线性代数、概率期望与协方差分析、蒙特卡洛验证及BBOB基准实验等技术；

**📊 数据集**

实验使用完整无噪声的BBOB（f1–f24）在5、10、20维、不同评估预算下进行；

**📈 对比分析**

通过与经典SOMA、iSOMA、SciPy DE、iL-SHADE等方法的平均排名、Wilcoxon检验等统计比较，改进版SOMA显著优于原版，在部分维度与DE竞争，任何时性能提升明显；

**⚠️ 局限性**

局限性在于仅验证了低维无噪声无约束问题，未评估高维、噪声或约束情况；旋转基准依赖协方差估计，易受噪声影响，且需进一步拆解各组件以确定真正贡献。

---

## 257. Is It Time for the Renaissance of Salient Object Detection in the Era of MLLMs?

**arXiv ID:** 2607.29222 | [PDF](https://arxiv.org/pdf/2607.29222v1)

**作者:** Wenzhuo Zhao `[一作]` (Sichuan University), Jian Cheng `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种全零样本显著目标检测框架FOCUS，并构建了SaliLLM诊断基准来评估多模态大型语言模型（MLLMs）的显著性感知能力。

**💡 创新点**

创新点在于：①将显著目标检测拆解为显著实体定位与基于提示的分割；②通过观测者协议的贝叶斯惊讶校准实现上采样的前景粒度和范围控制；③引入协同的顶部-到底部注意力机制，将MLLM的稀疏语义证据与自监督特征的感知流形融合，实现“前景组织”。

**🔧 技术方法**

核心技术包括：多模态LLM（如Qwen3‑VL‑8B‑Instruct）用于语义定位；DINOv3自监督视觉特征生成感知流形；SAM3作为通用分割器；Protocol‑Conditioned Bayesian Surprise (PCBS) 用于选择最符合协议的掩码。

**📊 数据集**

实验覆盖13个RGB、RGB‑D、RGB‑T SOD基准，包括DUTS、DUT‑O、HKU‑IS、PASCAL‑S、ECSSD、SIP、STERE、DUTLF‑D、NLPR、NJUD、VT821、VT1000、VT5000等。

**📈 对比分析**

与全监督、弱监督、自监督及零样本基线（GPT‑4o、Gemini‑2.5、Qwen2.5‑VL、Qwen3‑VL）相比，FOCUS在结构测度、最大F值、最大增强对齐度、均方误差上均取得最高或第二高分，尤其在无训练条件下明显优于同类零样本方法，且推理速度与显存占用大幅提升。

**⚠️ 局限性**

局限性在于：①仍依赖预先构造的观测者协议，协议不完整时效果退化；②对复杂多前景场景的细粒度分割仍有提升空间；③对极端遮挡或非标准场景的鲁棒性尚待进一步验证。

---

## 258. GALA: Generative Aligned Learning for Adaptive Multimodal Representation in the Taobao Shangou Recommender System

**arXiv ID:** 2607.29213 | [PDF](https://arxiv.org/pdf/2607.29213v1)

**作者:** Jiping Liu `[一作]` (Rajax Network Technology (Taobao Shangou of Alibaba)), Jia Jia `[通讯]` (Rajax Network Technology (Taobao Shangou of Alibaba))

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了GALA三阶段管线，结合查询–图像–文本对齐、生成式RL对齐以及自适应门控融合，以提升食品配送平台的多模态推荐效果。

**💡 创新点**

核心创新在于生成式RL对齐阶段将预训练分布与业务目标对齐，并引入自适应门控与辅助损失，有效克服ID主导导致的多模态信号稀释。

**🔧 技术方法**

所采用技术包括跨模态对比学习、GRPO强化学习、Qwen2.5‑7B解码器、T5级融合、门控机制与辅助损失等。

**📊 数据集**

实验使用淘宝生鲜商户日志，涵盖800万图文对、1400万查询‑图像‑文本三元组、9亿用户历史序列以及42亿排位日志等数据。

**📈 对比分析**

在离线检索任务中，GALA的Recall@K达0.877，超过GME；在线A/B测试订单量提升0.55%，并在CTR/CVR AUC和PCOC指标上优于MMREC、SimTier等基线。

**⚠️ 局限性**

主要局限包括门控机制在ID主导训练下易失效，长尾覆盖仍不足，以及需要额外离线训练与推理的算力成本。

---

## 259. Knowing When to Quit: Diagnosing and Training LLMs to Abort Futile Reasoning

**arXiv ID:** 2607.29211 | [PDF](https://arxiv.org/pdf/2607.29211v1)

**作者:** Xinyan Guan `[一作]` (Chinese Academy of Sciences), Fandong Meng `[通讯]` (Tencent Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CaRL 框架，解决大型语言模型在超出能力范围时产生“无效推理”的问题。

**💡 创新点**

通过能力校准奖励塑造与后向拒绝增强两种机制，使模型行为与真实能力边界对齐，显著降低无效推理。

**🔧 技术方法**

采用强化学习（GRPO）、奖励重塑、后向拒绝增强（HRA）以及生成式对话策略。

**📊 数据集**

以 Countdown（不同难度）和 Sudoku 为主要实验集，并在 AIME 2024 与 GPQA 上检验通用性能。

**📈 对比分析**

与 vanilla、标准 RL、RLunk、RFT 等基线比较，CaRL 在可靠性上提升 15–20% 并将无效推理率降至 1% 左右，同时保持近乎相同的准确率。

**⚠️ 局限性**

仅在无知识推理任务上验证，缺乏对知识密集型场景的实验，且模型在极难任务上仍可能出现过度拒绝的风险。

---

## 260. Domain-Division based Progressive Learning for Source-Free Domain Adaptation

**arXiv ID:** 2607.29202 | [PDF](https://arxiv.org/pdf/2607.29202v1)

**作者:** Pan Liu `[一作]` (Tianjin University of Technology), Shengyong Chen `[通讯]` (Tianjin University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种源自由域适配方法 DPL，先按适配难度将目标域划分为易适配与难适配子域，再通过两阶段渐进学习实现模型迁移。

**💡 创新点**

创新点在于利用不确定性信息对目标样本进行分层划分，并在两阶段中分别采用不确定性自训练+图对比学习和一致性学习+实例对比学习，充分挖掘可靠样本与噪声样本的互补信息。

**🔧 技术方法**

采用的不确定性加权自训练、图对比学习 (GCL)、实例对比学习 (ICL)、邻域软投票伪标签、核范数正则化、数据增强一致性等技术。

**📊 数据集**

在 Office-31、Office-Home 与 VisDA 三个标准域适配基准上进行实验。

**📈 对比分析**

与多种 UDA 与 SFDA 先进方法比较，DPL 在 Office-31、Office-Home、VisDA 的平均精度分别达到 90.7%、74.0%、87.8%，在多数任务中均取得最高或接近最高的表现。

**⚠️ 局限性**

局限性包括：对噪声样本聚类的依赖在某些类别（如 VisDA 的 car）导致性能下降；需手动调节 γ、T 等超参，对大规模数据集的核范数正则影响有限。

---

## 261. UltraSAM3: A Concept-Driven Foundation Model for Universal Ultrasound Image Segmentation

**arXiv ID:** 2607.29200 | [PDF](https://arxiv.org/pdf/2607.29200v1)

**作者:** Bo Xu `[一作]` (Dalian University of Technology), Chenhua Ji `[通讯]` (Dalian University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 UltraSAM3，一种通过概念驱动的全基线模型，实现对多器官多数据集的超声图像分割，并配备指令引导代理处理复杂自然语言查询。

**💡 创新点**

创新点在于将 SAM3 完全微调至超声领域，使用图像‑掩码‑概念三元组训练，实现仅文本提示即可分割；同时引入指令引导代理，将复杂用户指令简化为可直接输入模型的概念提示。

**🔧 技术方法**

采用 SAM3 的全参数微调，结合文本编码器、图像编码器、检测器、跟踪器和分割模块；使用 Gemini‑3.1‑Pro 作为指令解析器；训练目标为掩码重建与区域匹配损失。

**📊 数据集**

共 37 个公开超声分割数据集（112,634 张图像，171,693 个掩码），涵盖 13 个解剖类别；外部验证集包括 BrEast、CCAUI、105US_tumor、KFGNet。

**📈 对比分析**

与 UniBiomed、BiomedParse、SAM3、Medical SAM3 等基线对比，UltraSAM3 在 13 类别的平均 IoU 从 0.3254 提升到 0.6342（+0.3088），Dice 从 0.4012 提升到 0.7144（+0.3132）；在外部数据集上亦保持最高平均 IoU 与 Dice，且指令代理在复杂查询下平均 Dice 提升 0.161。

**⚠️ 局限性**

局限性包括：在极低对比度或噪声极大的超声图像中仍可能出现定位误差；对新出现的解剖结构或疾病类型需进一步收集训练数据；指令解析器仍依赖外部大模型，对实时性和资源消耗有一定影响。

---

## 262. A Proof of the Dittert Conjecture in Dimension 4 via an Agent-Guided Exact Sum-of-Squares Certificate

**arXiv ID:** 2607.29191 | [PDF](https://arxiv.org/pdf/2607.29191v1)

**作者:** Jinhui Li `[一作]` (East China Normal University), Zhengfeng Yang `[通讯]` (East China Normal University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在四维矩阵单纯形上证明了Dittert猜想，给出了量化稳定性估计并且唯一性。

**💡 创新点**

提出了agent‑guided的符号‑数值组合方法，能够在Gram矩阵奇异且受线性约束耦合的情况下构造精确的有理SOS证书，并在Lean中完成形式化验证。

**🔧 技术方法**

采用半正定规划求取近似SOS、基于模板搜索的agent‑guided恢复、精确的LDLᵀ分解、符号‑数值连续化以及Lean证明助手进行形式化检验。

**📊 数据集**

该工作不使用经验数据集，而是对16维多项式进行符号分析，构造了全尺寸约153×153的Gram矩阵和136个16维的子块。

**📈 对比分析**

与传统的数值SOS近似方法相比，本文的证书在精度上完全可证明且在Lean中得到形式化验证；证书规模巨大但仍可在合理时间内完成验证。

**⚠️ 局限性**

局限性：仅对n=4情形成立；方法对更高维度的Dittert问题是否可扩展尚未证明；构造过程高度依赖数值求解与人工指导，计算成本较大。

---

## 263. CAGE: Certified Authorization under Typed-Return Uncertainty for Tool-Using Agents

**arXiv ID:** 2607.29190 | [PDF](https://arxiv.org/pdf/2607.29190v1)

**作者:** Blaise Delattre `[一作]` (Institute of Science Tokyo), Yang Cao `[通讯]` (Institute of Science Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于联合邻域的授权门（Certified Authorization Gate），在工具返回的离散绑定错误与连续数值漂移同时存在时，对下游动作进行安全性认证；

**💡 创新点**

创新点在于①证明单独对离散和连续通道的边界判定不能保证联合安全；②设计了枚举离散邻域并对每个分支进行连续判定的完整证书流程；③构建了假设阶梯（Exact、Lip、RS）以支持可执行和隐式策略的不同情景；

**🔧 技术方法**

主要技术包括：离散邻域枚举、Lipschitz 边界认证、随机平滑（Randomized Smoothing）判定、基于 MILP 的离线上限验证，以及对Typed返回构造器的完整性校验；

**📊 数据集**

使用的数据集包括：合成金融/SRE/ops 记录、IEEE‑CIS 实际交易数据、OpenFisca 法规案例、k8s/MCP/Marble 现场实验、以及通过故障注入获得的适配器错误样本；

**📈 对比分析**

与传统点检授权、边际组合授权以及现有工具（AgentSpec、VeriGuard）比较，联合邻域门在所有测试集上实现 0% 的 Certified False‑Allow（CFA），并保持 20‑60% 的自主决策率；性能上 Exact 门 <1µs，Lip 7–17 ms，RS 10–132 ms，显著低于 LLM 解码时间；

**⚠️ 局限性**

局限性包括：仅针对单一决策的安全性，假设返回构造器可信且离散/连续预算已校准；无法覆盖跨轮累积、工具选择、提示注入、MCP 元数据和多步执行的安全问题；

---

## 264. MBDiff: Multi-view Behavior-aware Diffusion Model for Probabilistic Utility Data Imputation

**arXiv ID:** 2607.29177 | [PDF](https://arxiv.org/pdf/2607.29177v1)

**作者:** Rongchao Xu `[一作]` (Florida State University), Guang Wang `[通讯]` (Florida State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出了一个行为感知的公用事业数据缺失填补框架，利用多视角用户行为（全局、局部、实例）驱动条件扩散模型进行精确缺失值插补。

**💡 创新点**

创新点在于：①构建三层视角的用户行为提取模块，系统捕获长期统计、短期局部动态和记录级上下文；②设计参考选择机制与注意力去噪网络，使扩散模型能够高效利用稀疏参考；③在缺失填补任务中首次将用户行为作为条件先验融合到扩散生成流程中。

**🔧 技术方法**

技术组合包括：Transformer‑based VAE（用于生成全局、局部、实例潜在表示）、多视角行为提取、参考选择模块、基于注意力的条件去噪网络以及扩散概率模型（DDPM）。

**📊 数据集**

使用了佛罗里达州一家大型市政公用事业公司提供的三类实测数据：电力、水务与天然气，覆盖约12万用户，记录间隔为30分钟，经过筛选后每用户至少200条记录。

**📈 对比分析**

与10个代表性基线（统计、预测与生成方法）在点缺失与块缺失两种模式下进行比较。实验显示该方法在电力数据点缺失/块缺失分别提升MAE 2.32%/4.0%、MSE 2.32%/4.0%、CRPS 15.57%/7.71%；在水务块缺失MAE降低29.1%，在气务块缺失MAE提升7.04%；在不同缺失率下始终保持前两名。

**⚠️ 局限性**

局限性包括：①对极度稀疏的数据（如水、气）仍需依赖全局统计作为先验，精度受限；②多阶段训练（VAE与扩散）耗时较长，部署成本高；③当前仅在单一公用事业场景验证，跨域泛化性待进一步探索。

---

## 265. Execution-First Synthetic Tool-Use Trace Generation for LLM Agents

**arXiv ID:** 2607.29175 | [PDF](https://arxiv.org/pdf/2607.29175v1)

**作者:** Hafsa Ouajdi `[一作]` (EURECOM), Adam Elwood `[通讯]` (Aily Labs)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SyntheticAgentTraceQA，一种执行先行的流水线，先生成并验证可执行的工具使用轨迹，再从成功执行中合成自然语言任务与答案，用于训练工具增强型代理。

**💡 创新点**

创新点在于将工具操作分类、自动化元数据提取、抽象工作流模板生成、深度优先搜索验证可执行性以及执行后任务合成相结合，实现高比例的有效轨迹、减少无效示例，并通过执行验证提升数据质量。

**🔧 技术方法**

采用 LLM（Claude、GPT、Qwen 等）完成工具元数据抽取与模板生成，DFS 搜索工具组合，执行引擎验证执行结果，LoRA 微调 + 思考机制进行训练，并用答案完整率、工具成功率、引用轨迹一致性等指标评估性能。

**📊 数据集**

在四个工具生态系统（金融、研发、供应链和 ToolBench 的音乐工具）中生成约 7.8k 条多难度任务的数据集；并与 Toucan 生成的数据集进行对比。

**📈 对比分析**

通过在 Qwen3.5-4B/9B 模型上比较思考与非思考、Masked 与 Full 监督方式，实验显示执行-验证微调显著提升工具成功率、引用一致性和答案质量；Masked 监督优于 Full；与 Toucan 相比，SyntheticAgentTraceQA 在有效率、成本、质量以及某些指标（如工具成功率、引用一致性）上更优。

**⚠️ 局限性**

局限性包括仅在固定工具 schema 与受控环境下验证，微调仅使用有限样本，未覆盖动态 API 变更或长周期任务；数据生成依赖 LLM，受模型生成质量限制；缺乏人类评估来进一步验证自然语言任务的真实性与多样性。

---

## 266. CLIFT: Turning Gemini Robotics On-Device into Humanoid Specialists via Non-Invasive Closed-Loop Iterative Fine-Tuning

**arXiv ID:** 2607.29172 | [PDF](https://arxiv.org/pdf/2607.29172v1)

**作者:** Yuxin Chen `[一作]` (University of California, Berkeley), Thomas Tian `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何通过闭源机器人基础模型的管理式SFT API实现人形机器人在敏捷、接触密集任务中的闭环自适应与任务精通。

**💡 创新点**

提出闭环迭代微调（CLIFT）方法：将部署时的奖励反馈通过检索式优势标记转化为API兼容的监督学习数据，实现在不访问模型权重、梯度或内部信号的前提下的闭环改进。

**🔧 技术方法**

核心技术包括：零样本视觉语言模型生成的候选奖励、人工偏好校准的稠密奖励模型、检索式优势条件标记、基于SFT API的离线微调循环。

**📊 数据集**

使用Unitree G1人形机器人在三种接触密集任务（盒子装箱、杯子插入、双手盘子递送）的真实数据；演示数据来自VR全身操作，约2小时/任务；奖励模型基于100对人工偏好比较。

**📈 对比分析**

与开放权重VLA（π_0.5）进行对比；在闭环迭代两轮后，Gemini Robotics On-Device（闭源）从SFT基线分别提升至Box Packing 93%→100%，Cup Insertion 70%→98%，Bimanual Plate Handover 53%→96%；开放模型提升幅度更小，最高仅达30%。

**⚠️ 局限性**

局限性：需要每轮真实机器人部署收集数据，成本高且安全敏感；仅评估了一个开放模型；未探讨更广泛的全访问内部改进方法。

---

## 267. MOSAIC: Masked Outsourcing of Secure AI Computations

**arXiv ID:** 2607.29221 | [PDF](https://arxiv.org/pdf/2607.29221v1)

**作者:** James Hsin-yu Chiang `[一作]` (ETH Zurich), Srdjan Capkun `[通讯]` (ETH Zurich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MOSAIC 协议，允许可信但计算弱的客户端在不泄露输入或模型权重的前提下，将大型 Transformer 推理的线性矩阵乘法安全地外包给不受信任的加速器。

**💡 创新点**

创新点包括：① 采用基于 LWE/LPN 的低秩 + 小高斯噪声掩码，实现 O((m+n)l) 的客户端在线复杂度；② 引入随机 Hadamard 旋转来抑制固定点量化误差，确保误差在深层网络中可控；③ 在现代 70B 级模型上首次实现可扩展的安全外包，突破先前工作对矩阵尺寸的严重限制。

**🔧 技术方法**

使用技术：LWE/LPN 隐写掩码、递归掩码层、随机 Hadamard 旋转、16-bit 固定点量化、32-bit 整数环计算、GPU Tensor Core 的 INT8 级别乘加模拟、RDMA 等高速互连、非对称加密的安全协议。

**📊 数据集**

评估数据集与模型：LLaMA‑3、Qwen‑2.5‑72B、Qwen‑2.5‑32B 等 70B 级 Transformer；使用 WikiText‑2 进行 perplexity 评测，HumanEval 进行 pass@1 评测。

**📈 对比分析**

对比方法：与 Slalom、传统 MPC/FHE 方案以及 NF4、INT8 量化基线比较。MOSAIC 在 140‑bit 安全性下，误差与 INT8 基线相当或更优，推理准确率维持在 BF16 的 2% 以内；在本地单 GPU 推理的 3×–11× 运行时间（由于 INT32 emulation），但通信开销仅占 4–16% 的总延迟。

**⚠️ 局限性**

局限性：① 需要在 GPU 上对 32‑bit 整数运算进行 INT8 级别模拟，导致显著性能损失；② 只对线性层做外包，非线性层仍需本地执行；③ 误差需通过随机 Hadamard 旋转来控制，可能对极端稀疏或特殊模型敏感；④ 对模型尺寸的低秩掩码要求仍需满足 r << n，极小内维时效率下降；⑤ 可信基础仍需存在，无法完全消除对可信硬件的依赖。

---

## 268. SAF-OPD: Stable Advantage Fusion for On-Policy Distillation

**arXiv ID:** 2607.29209 | [PDF](https://arxiv.org/pdf/2607.29209v1)

**作者:** Yifan Ding `[一作]` (Shanghai University of Finance and Economics), Yun Chen `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种稳定优势融合（SAF）框架，用以在RLVR与OPD奖励信号之间进行高效融合，避免熵崩塌并提升学习稳定性。

**💡 创新点**

创新点在于将优势融合拆分为幅度控制（Top‑k稀疏化+tanh压缩）与时间控制（KL触发的warm‑up与线性anneal）四阶段可切换流程，从而分别解决了token级优势幅度失衡与全程指导强度失调问题。

**🔧 技术方法**

技术主要包括GRPO（基于可验证奖励的组相对策略优化）、OPD（对教师概率的token级逆KL对齐）、tanh压缩、KL监测以及基于采样的逆KL估计等。

**📊 数据集**

实验数据集涵盖数学推理（AIME24/25、HMMT25）与代码生成（HumanEval+、MBPP+、LiveCodeBench），并使用Qwen3-1.7B/4B/8B模型与Qwen3-30B教师。

**📈 对比分析**

与基线（Base、GRPO‑only、OPD‑only、GRPO+OPD固定系数）比较，SAF在所有六个模型-领域组合上平均提升0.51%–2.70%，并在训练动态上避免了熵崩塌和准确率早期平稳。

**⚠️ 局限性**

局限性包括对教师质量的依赖、超参数（k、c、S_warmup、δ等）的敏感性，以及实验仅覆盖1.7B–8B规模和所选教师，未检验更大规模模型或不同教师情况下的鲁棒性。

---

## 269. MoRAE: Flow-Friendly Self-Supervised Latents for Text-to-Motion Generation

**arXiv ID:** 2607.29180 | [PDF](https://arxiv.org/pdf/2607.29180v1)

**作者:** Yifei Zhu `[一作]` (Tohoku University), Taku Komura `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MoRAE框架，将Motion‑JEPA的高维特征压缩为32维流友好的潜在空间，并在该空间训练流匹配的DiT生成文本驱动动作；

**💡 创新点**

通过两步改进：① 在冻结的Motion‑JEPA特征上加入变分压缩瓶颈，使潜在谱更稳定；② 在潜在学习中将运动解码损失反向传播到编码器，使潜在与解码器对齐，降低残差对解码器的放大；

**🔧 技术方法**

使用Motion‑JEPA自监督预训练、变分编码器、特征解码器+运动解码器、非自回归流匹配DiT以及CLIP文本编码；

**📊 数据集**

在HumanML3D和KIT‑ML两个公开动作生成数据集上进行评估；

**📈 对比分析**

与多种基线（MDM、MotionDiffuse、MLD、MARDM、T2M‑GPT、VQ等）对比，MoRAE在FID、R‑Precision、CLIP‑score、foot‑skate、jerk、骨骼变异等指标均实现最优或接近最优性能；

**⚠️ 局限性**

仍受限于特定帧率（20 FPS）、对Motion‑JEPA冻结的依赖、以及在极端复杂或高频动作场景下可能出现轻微物理误差。

---

## 270. ActFovea: Runtime Safeguarding for VLA Policies via Spatiotemporal Visual-Action Consistency

**arXiv ID:** 2607.29169 | [PDF](https://arxiv.org/pdf/2607.29169v1)

**作者:** Wenda Yu `[一作]` (Tongji University), Lei Zhu `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 ActFovea，一种用于视觉‑语言‑动作（VLA）策略的运行时安全框架，能够在不重新训练策略的情况下检测并纠正因视觉遮挡、时间延迟或动作漂移导致的闭环失配。

**💡 创新点**

创新点在于将动作条件下的视差感知（foveation）、时空一致性监测、扰动特定的观测恢复与动作片段验证相结合，实现统一的、可插拔的运行时保护，并在失去感知时自动转为安全失败。

**🔧 技术方法**

技术包括基于机器人运动学的动作条件 fovea 构建、视觉‑运动一致性评分、候选观测银行与动作验证，以及安全失败触发机制。

**📊 数据集**

实验使用 LIBERO 任务套件（LIBERO‑Spatial、Object、Goal、10），并在 π₀ VLA 策略上进行评估。

**📈 对比分析**

与基线、动作裁剪、短时域、时间戳保持等训练自由方法比较，ActFovea 在视觉遮挡、时间延迟和动作漂移场景下分别提升了约 41%、9.8% 和 7.0% 的成功率，并在冻结观测重放时实现所有试验的及时安全失败，性能优于其他方法。

**⚠️ 局限性**

局限性包括对运动学模型的依赖、仅针对已知扰动类型的设计、在极端或持续失效情况下可能恢复失败，以及对计算资源的额外需求。

---

## 271. Authorship Verification of Transcribed German-Language Videos

**arXiv ID:** 2607.29168 | [PDF](https://arxiv.org/pdf/2607.29168v1)

**作者:** Oren Halvani `[一作]` (Fraunhofer Institute for Secure Information Technology SIT), Sophie Titze `[通讯]` (Fraunhofer Institute for Secure Information Technology SIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了将作者身份验证（Authorship Verification, AV）应用于德语视频转录文本，构建了三套自编语料库并评估了十种主流AV方法，包括传统字符/语法特征模型和现代Transformer模型。

**💡 创新点**

创新点在于：①首次在德语口语转录文本上系统评估AV方法；②采用主题掩码技术剔除语义信息，仅保留风格特征；③比较传统与Transformer两大类模型在主题掩码下的表现，揭示传统模型仍具竞争力。

**🔧 技术方法**

技术主要包括：Whisper自动转录、POS掩码、字符/语法特征提取、传统机器学习分类器（如SVM、逻辑回归）以及基于Transformer的预训练模型（BERT、RoBERTa等）和多语言嵌入。

**📊 数据集**

数据集为三类德语单说话者视频（财务建议、数学辅导、DIY）共300段视频，转录后生成约300篇文本；每类包含40名训练说话者、60名测试说话者，形成1200个正负对比验证样本。

**📈 对比分析**

结果显示：传统基于字符/语法特征的模型在原始文本和主题掩码文本上均能取得最高精度（最高88%准确率、90% AUC），尤其在掩码文本上表现更佳；Transformer模型整体准确率低于0.75，且在掩码下性能下降。传统模型的高精度与可解释性使其在法医学场景更具吸引力。

**⚠️ 局限性**

局限性包括：①语料规模小（每类仅40训练/60测试说话者）；②缺乏人工参考转录，无法评估ASR误差影响；③仅覆盖德语单说话者的单元组；④实验仅使用了单一主题掩码方法，可能影响泛化；⑤未对多说话者或非口语场景进行验证。

---

## 272. Have I Seen You? Embedding Behavior Signals Synthetic Face Dataset Membership

**arXiv ID:** 2607.29144 | [PDF](https://arxiv.org/pdf/2607.29144v1)

**作者:** Paweł Borsukiewicz `[一作]` (University of Luxembourg), Tegawendé F. Bissyandé `[通讯]` (University of Luxembourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了合成人脸数据集对源数据泄露的风险，提出了数据集级的成员推理攻击，能够在黑盒场景下识别出用于训练识别模型的合成数据集以及该合成数据集生成器所使用的真实训练数据集。

**💡 创新点**

创新点在于首次从数据集层面设计鲁棒z‑score攻击算法，可仅通过嵌入向量检测训练集，且实现了对生成器源数据集的识别，揭示了合成数据中潜在的源数据痕迹。

**🔧 技术方法**

技术手段包括基于iResNet50的特征提取、鲁棒z‑score统计、黑盒查询攻击框架，以及对比分析与真实数据集的相似度评估。

**📊 数据集**

实验使用11个合成人脸数据集（如Digi2Real、FFHQ等）以及7个真实数据集（FFHQ、CASIA-WebFace、LFW、CPLFW、CALFW、CFP‑FP、AgeDB‑30）。

**📈 对比分析**

通过对比7个真实数据集，利用鲁棒z‑score对所有模型的嵌入进行评估，攻击在识别合成训练集方面达到100%成功率，在识别生成器源数据集方面取得54.5%的成功率。

**⚠️ 局限性**

局限性包括对真实源数据集的识别效果受限，混合训练集会导致误判；攻击在黑盒情形下对真实数据泄露的概率相对较低，亟需更强的泄漏缓解技术。

---

## 273. HERO: History-Enriched Rollout Training for Long-Horizon Autoregressive Neural Operators

**arXiv ID:** 2607.29135 | [PDF](https://arxiv.org/pdf/2607.29135v1)

**作者:** Jiaquan Zhang `[一作]` (UESTC), Chaoning Zhang `[通讯]` (UESTC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HERO，一种历史信息增强的自回归神经算子训练框架，利用模型训练历史中的失败轨迹作为相对监督来提升长周期预测性能。

**💡 创新点**

创新点在于将传统的绝对轨迹误差监督与动态选取的历史失败轨迹进行对比，构造基于 margin 的相对损失；通过周期性刷新滞后算子、诊断指标（误差、谱失配、能量漂移、误差增长）进行参考轨迹挑选，并从理论上证明相对监督实现了对绝对梯度的有界重加权，显著抑制自回归误差累积。

**🔧 技术方法**

使用了神经算子（Fourier Neural Operator、Transolver）进行时间步长预测；实现自回归 roll-out、候选轨迹构造、诊断指标计算、滞后算子刷新、margin 基础相对损失；并在训练时加入 warm‑up 权重、拉伸因子等超参。

**📊 数据集**

在九个 PDE 基准上验证：1D（Burgers、Kuramoto–Sivashinsky、KdV、Dispersion）、2D（Anisotropic Diffusion、Kolmogorov Flow、Navier–Stokes Decaying Turbulence）、3D（Swift–Hohenberg、Unbalanced Advection），覆盖扩散、输运、振荡、湍流等多种动力学。

**📈 对比分析**

与单步监督、push‑forward、PDE‑Refiner、Recurrent Operator 等基线相比，HERO 在 nRMSE@100、GM_100 和稳定步长（stable step）上均提升 30–40% 以上，且在零样本 OOD 评估（参数/分辨率迁移）中显著降低误差并延长稳定滚动长度；消融实验表明相对监督和滞后参考的组合是提升的关键。

**⚠️ 局限性**

局限性包括：仅在仿真 PDE 数据上验证，无人类数据或隐私敏感信息；训练时额外的候选轨迹生成和参考选择会增加计算量；实验集中在指定的 PDE 任务和算子结构，可能对更广泛的物理系统或算子架构的推广性尚未充分验证。

---

## 274. Interactive Generative Motion Editing via Scheduled Inpainting

**arXiv ID:** 2607.29133 | [PDF](https://arxiv.org/pdf/2607.29133v1)

**作者:** Dhruv Agrawal `[一作]` (ETH Zürich), Jakob Buhmann `[通讯]` (DisneyResearch|Studios)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于已预训练扩散模型的“Scheduled Inpainting”框架，能够在保持原始运动的基础上进行多种实时交互式生成式运动编辑，包括直接操控、延伸、拼接、组合和重定时。

**💡 创新点**

创新点：①在推理阶段实现可控的运动保留与生成，打破传统离线优化的限制；②引入用户可调节的时间-空间掩码与调度，细粒度控制运动保持与重写；③实现多种编辑任务的统一实现，无需重新训练模型。

**🔧 技术方法**

技术手段：基于扩散模型（如 IBMM、SF-control）的推理式运动生成；Scheduled Inpainting 方法（时间调度 σs/σe 与空间掩码 α_mask）；运动对齐与归一化；对空间掩码的高斯/三角/方形核函数实验。

**📊 数据集**

使用多种数据集验证：AMASS、LaFan1、内部数据集；在这些数据集上进行跨模型、跨任务的对比评估。

**📈 对比分析**

与四个基线（MotionLab、CondEditor、Noise‑inversion、DNO）以及不同调度设置进行定量对比，指标包括脚步滑移、L2 位姿误差；实验表明 Scheduled Inpainting 在保持原始运动与生成自然过渡方面优于基线，且在实时性（≈25步）上远快于噪声反演。

**⚠️ 局限性**

局限性：受预训练模型分布限制，超出分布的编辑可能产生异常；高频细节恢复有限；大于训练长度的序列时原始运动保持衰退；对接触点交互的界面仍需改进。

---

## 275. First Investigation of Deep Learning for Intraoperative Gauze Segmentation in Minimally Invasive Abdominal Surgery

**arXiv ID:** 2607.29132 | [PDF](https://arxiv.org/pdf/2607.29132v1)

**作者:** Priya Tomar `[一作]` (Fraunhofer Iais), Rafet Sifa `[通讯]` (Fraunhofer Iais)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

在真实机器人腹腔手术视频上研究并实现纱巾分割，评估CNN、Transformer及混合模型，并验证半自动跟踪标注对性能的提升

**💡 创新点**

首次在真实手术数据上开展纱巾分割研究；系统比较多种深度学习架构并展示自动跟踪标注可显著提升分割效果；同时通过多种评价指标展示模型对血腥环境的鲁棒性

**🔧 技术方法**

采用U-Net、DeepLabV3、Attention U‑Net、TransU‑Net、SegFormer和Mask2Former等语义分割网络，使用预训练权重、Dice损失、数据增强和随机裁剪/翻转等技术

**📊 数据集**

使用自研GauzeSeg_Real数据集（16台机器人腹腔手术共22,265张含手工与自动标注图像）以及模拟GauzeSeg_Sim（4,003帧）作为Out‑of‑Domain测试

**📈 对比分析**

按病人ID划分训练/验证/测试，比较Dice、IoU、Precision、Recall等指标；预训练SegFormer‑b3在所有测试集上获得最高Dice≈0.82，自动标注提升多数模型在非血腥测试集Dice约+7%

**⚠️ 局限性**

单中心数据、仅二值纱巾分割、自动标注引入噪声未显式处理、超参数调优有限、未测实时推理时延，缺乏多中心验证

---

## 276. A Frozen Pixel-Space Diffusion Model Can Guide Itself with Its Own Samples

**arXiv ID:** 2607.29122 | [PDF](https://arxiv.org/pdf/2607.29122v1)

**作者:** Zixuan Fu `[一作]` (Nanyang Technological University), Bihan Wen `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

训练一个轻量级适配器在预训练像素扩散模型的中间层，并在采样时用该适配器的粗糙预测引导最终预测，从而提升生成质量。

**💡 创新点**

①仅冻结预训练模型，仅用自我生成的合成样本训练适配器；②利用中间层和最终层预测差异实现自我引导，无需额外弱模型；③在保持计算成本<1%的前提下显著降低FID。

**🔧 技术方法**

像素空间扩散、流匹配、轻量级 Transformer 适配器、内部引导/自我引导、无监督自合成数据训练、与 CFG 的组合使用。

**📊 数据集**

ImageNet 训练集/验证集（1.28M 实例）以及从预训练模型生成的约 1M 合成样本。

**📈 对比分析**

在 ImageNet 256×256 与 512×512 条件生成任务上与多种基准模型对比；在无 CFG 下 FID 降幅超过 50%；在 CFG 下提升 FID 约 10‑12%（如 JiT‑H/16 从 1.86 下降至 1.67）；相比 IG，SSG 既计算更低、无显式弱模型，又取得更优性能。

**⚠️ 局限性**

依赖预训练模型中间层可解码为粗糙预测，深层层差距减弱导致自我引导效果有限；适配器容量有限，可能无法捕捉更细粒度信息；在某些模型或分辨率上提升幅度有限；尚未在多样化数据集或任务上验证。

---

## 277. Multi-Granularity Position Embedding of Graphs via Granular-Ball for Link Prediction

**arXiv ID:** 2607.29115 | [PDF](https://arxiv.org/pdf/2607.29115v1)

**作者:** Sen Zhao `[一作]` (Chongqing University of Posts and Telecommunications), Wei Wang `[通讯]` (Chongqing Ant ConsumerFinance Co, Ltd)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一种基于多粒度位置嵌入的图链接预测方法MGLP，利用颗粒球分解生成多层次同质子结构并构建层次中心图来编码节点位置。

**💡 创新点**

引入自适应颗粒球细化、层次中心图(HCG)与多粒度层次距离(MGHD)三大创新，能同时捕获图中不同粒度的同质性和层次关系，显著提升位置编码的表达力。

**🔧 技术方法**

颗粒球图细化、层次中心图构造、谱图拉普拉斯特征、MGHD距离计算，以及结合GCN+JK等GNN进行节点表征学习和链接预测。

**📊 数据集**

Cora、Citeseer、PubMed、Facebook、DDI、COLLAB六个常用图数据集，涵盖从稀疏小型到大规模稠密网络。

**📈 对比分析**

与17种基线（AA、MF、Node2Vec、GCN、GraphSAGE、GAT、P‑GNN、NBF‑Net、JKNet、SEAL、GCN+DE、GCN+LPE、GCN+LRGA、Graph Transformer+LPE、PEG‑DW+、HPLC）在AUC或Hits@K上对比，MGLP在所有数据集均取得最高或接近最高分，尤其在DDI、COLLAB、PubMed等上领先0.5–1.4分。

**⚠️ 局限性**

仍受限于节点数规模对颗粒球中心数量的选择，过大/过小会影响效果；对动态图或有向图的适用性未验证，且细化与HCG构造增加了预处理成本。

---

## 278. Low-Power PLL-Based Clock Stabilization for Flexible IGZO AMS Systems

**arXiv ID:** 2607.29357 | [PDF](https://arxiv.org/pdf/2607.29357v1)

**作者:** Paula Carolina Lozano Duarte `[一作]` (Karlsruhe Institute of Technology), Mehdi Tahoori `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了首个在n型单一IGZO TFT技术下的低功耗PLL，用于可柔性AMS系统的时钟稳定

**💡 创新点**

突破了IGZO仅n型、低迁移率与高PVT波动的制约，设计了零静态电流、低带宽反馈的电荷泵PLL，并实现了1–300 kHz频率可调

**🔧 技术方法**

采用电荷泵PLL架构、低频环振荡器VCO、伪CMOS动态D触发器PFD、无静态电流电荷泵与被动低通滤波器

**📊 数据集**

未使用传统数据集，而是基于PragmatIC的FlexIC PDK进行后布局仿真，并在四个代表性IGZO AMS平台（≈1 kHz、10 kHz、150 kHz、300 kHz）上验证

**📈 对比分析**

与现有可柔性R/O、VCO时钟方案以及SoA CMOS PLL对比，功耗降低>400×、面积缩小>390×，功率占比降至5–12%，频率准确率≤1000 ppm，周期jitter≤2.24 ns

**⚠️ 局限性**

在0 °C时温度下无法锁定，且对极低温或极低电压条件仍有限制

---

## 279. SeekBrain: An Autonomous Multi-Agent System for Accelerating Neuroscience Discovery

**arXiv ID:** 2607.29347 | [PDF](https://arxiv.org/pdf/2607.29347v1)

**作者:** Jiamin Wu `[一作]` (Shanghai Artificial Intelligence Laboratory), Chunfeng Song `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了SeekBrain——一个基于多智能体、动态化学术分析谱系的自动化神经科学研究框架

**💡 创新点**

通过构建可演化的“神经科学分析谱系”实现了领域知识的自动提炼与重用，解决了通用LLM在神经科学数据碎片化场景下的“方法论幻觉”问题

**🔧 技术方法**

结合Claude‑Opus 4.7和Gemini‑3.1‑Pro等LLM，开发了研究规划引擎、分析执行引擎、验证与解释模块，并使用LLM‑驱动的“结晶与蒸馏”机制不断更新谱系

**📊 数据集**

在BrainArena（32个跨物种、多模态真实实验任务）以及两项真实案例（自由游动斑马鱼多模态数据与IBL鼠决策数据）上进行评估

**📈 对比分析**

与Claude Code、Codex等基线对比，SeekBrain在BrainArena的平均得分提升约11–18点，尤其在解释有效性上表现突出；在案例中能自动完成从数据预处理到模型训练、结果可视化与科学解释的完整流程

**⚠️ 局限性**

仍受限于LLM的推理错误与谱系覆盖度、对更大规模或更细粒度数据的适配、以及对实时闭环实验的支持尚不完善

---

## 280. MAGA: Multi-Platform Self-Fusion of GUI Agents via Structured Action Distillation

**arXiv ID:** 2607.29320 | [PDF](https://arxiv.org/pdf/2607.29320v1)

**作者:** Hang Yan `[一作]` (Xi'an Jiaotong University), Changhua Meng `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对跨平台 GUI 代理的统一方法，利用结构化动作的自适应监督重新分配和教师提示的训练方式，提升多域合并模型的成功率；

**💡 创新点**

创新点在于将动作结构作为监督信号的重要维度，针对动作类型和参数分别加权并在教师端注入仅包含动作类型的提示，从而解决传统权重合并导致的冲突与 OPD 对短结构化动作监督不足的问题；

**🔧 技术方法**

技术主要包括基于大型语言模型的 GUI 代理接口设计、路由式 on‑policy distillation（OPD）、动作级权重调度规则以及教师提示（hint）机制；

**📊 数据集**

使用了 MobileWorld、OSWorld 与 WebVoyager 三个跨平台 GUI 基准数据集进行训练与评估；

**📈 对比分析**

与域专门教师、SFT、GRPO、Weight Soup、TIES 以及 UI‑MOPD 等基线相比，在 8B 和 2B 模型规模上均获得最高的平均成功率（8B 最高 SR 51.2%，比最强基线高 2.0%），教师归一化得分（TNS）接近 100%，表明能够保持域专家能力；

**⚠️ 局限性**

局限性包括仍无法完全超越各域专家（受动作短小导致监督稀疏限制）、单步训练难以覆盖多步任务的泛化、以及对高不一致空间动作仍需进一步改进。

---

## 281. Allocation Tracking and Parameter Checking for Parallel Programming Models using Contracts

**arXiv ID:** 2607.29303 | [PDF](https://arxiv.org/pdf/2607.29303v1)

**作者:** Yussur Mustafa Oraji `[一作]` (TU Darmstadt University), Christian Bischof `[通讯]` (TU Darmstadt University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

扩展了 CoVer 合约语言，添加了参数检查和内存分配跟踪，并实现了相应的静态与动态分析。

**💡 创新点**

提出了可在不同并行编程模型和语言下统一使用的合约扩展，支持泛型参数验证和堆/栈/全局分配状态追踪，利用内联回调与 CoVer 内建 Intrinsics。

**🔧 技术方法**

基于 LLVM IR 的数据流分析、libffi 的内联回调、CoVer 的合约解析器、Fortran 元数据恢复以及动态运行时回调。

**📊 数据集**

使用 MPI-BugBench 1.0 级别测试集（222 条测试）以及 PRK Stencil、miniWeather、LULESH 三个代理应用。

**📈 对比分析**

与旧版 CoVer 以及 MUST 进行 TP/TN/FN/准确率比较，动态版本新扩展准确率提升至 200%+，在三款代理应用中的运行时开销约为 CoVer-Old 的 1.2–1.5 倍，仍低于 MUST。

**⚠️ 局限性**

额外的堆栈/全局追踪导致运行时性能下降，且在 Fortran 元数据恢复及非 C/C++ 代码中仍需手动声明合约值，动态分析依赖 libffi 造成跨语言兼容性限制。

---

## 282. Temporal Role Colouring

**arXiv ID:** 2607.29272 | [PDF](https://arxiv.org/pdf/2607.29272v1)

**作者:** Jessica Enright `[一作]` (University of Glasgow), Ella Yates `[通讯]` (University of Glasgow)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了时间角色着色（Temporal Role Colouring）问题的定义，并给出了其NP-难性与若干参数化可解性（FPT）结果。

**💡 创新点**

创新点在于将经典的角色着色扩展到时间图中，利用状态自动机编码时间演化的角色约束，并通过三种稀疏参数（vertex‑interval‑membership width、tree‑interval‑membership width、树宽+终止时间+颜色数）实现FPT算法。

**🔧 技术方法**

主要技术包括：状态自动机的构造与分析、基于动态规划的时间序列分段处理、利用 TIM 分解的元算法，以及在树分解上的动态规划与状态压缩。

**📊 数据集**

未使用实际数据集，研究全部基于理论证明与算法复杂度分析。

**📈 对比分析**

与传统静态角色着色的比较：证明了时间角色着色在一般情况下仍为NP‑完整；在三种参数化下给出了多项式时间或指数时间复杂度的FPT算法，表明在这些参数受限时可高效求解。

**⚠️ 局限性**

局限性包括：对稠密或结构更复杂的时间图（如低临时团宽、临时模宽等）尚未处理；自动机规模的快速增长可能导致实际运行效率下降；并且仅在特定参数化下可解，无法覆盖所有应用场景。

---

## 283. About two results for new valid juggling sequences

**arXiv ID:** 2607.29255 | [PDF](https://arxiv.org/pdf/2607.29255v1)

**作者:** Hugo Parada `[一作]` `[通讯]` (Universite de Lorraine), Hugo Parada (Universite de Lorraine)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出两种新的算子——“first throws”和“last catches”，用于在不产生碰撞的前提下延长已验证的siteswap（抛投序列）。

**💡 创新点**

创新点在于给出了这两种扩展操作的必要且充分条件，利用落地时间模新周期的排列测试，证明对所有基于地面状态的有效序列均成立，并首次系统化地展示了这些操作在构造更复杂、长周期的抛投序列中的应用。

**🔧 技术方法**

主要技术包括：siteswap的模运算表示、落地时间的排列测试、垂直位移与周期变换、状态图与循环的分析，以及对“first throws”和“last catches”的代数定义和证明。

**📊 数据集**

本文未使用传统意义上的数据集，而是通过对经典的基准序列（如3‑ball cascade 441、5‑ball cascade 5555 等）以及自构造的示例序列进行符号化演示，并提供了交互式可视化工具供验证。

**📈 对比分析**

由于本研究属于理论性质，没有与其他算法进行实验对比；通过对示例序列的符号验证与图形可视化，展示了扩展操作在保持球数不变且保持周期性与合法性方面的有效性。

**⚠️ 局限性**

主要限制：仅适用于简单（单球）siteswap，未涵盖同步、复合或多球（multiplex）模式；扩展操作在一般非地面状态序列下并非总是可行，需要额外的判定条件。

---

## 284. BRHC: Backend-driven Reactive Hypermedia Controls with a Statically Typed Kotlin DSL

**arXiv ID:** 2607.29338 | [PDF](https://arxiv.org/pdf/2607.29338v1)

**作者:** Fernando Miguel Carvalho `[一作]` (Instituto Superior de Engenharia de Lisboa, Instituto Politécnico de Lisboa), Juho Vepsäläinen `[通讯]` (Aalto University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于 Kotlin DSL 的后端驱动反应式超媒体控制（BRHC）架构。

**💡 创新点**

创新点在于将超媒体控制、信号绑定和请求编排全部迁移到静态类型的 DSL 中，消除字符串化的 JavaScript 表达式并提升类型安全。

**🔧 技术方法**

使用技术包括 Kotlin、HtmlFlow DSL、Datastar、Server‑Sent Events、Spring MVC（PetClinic）以及对比的 Thymeleaf 和 React。

**📊 数据集**

主要使用 Spring PetClinic 的示例数据集（宠物、所有者等表）进行评估。

**📈 对比分析**

通过 JMeter + Selenium 与 Lighthouse 对单线程交互性能进行基准，结果显示 HtmlFlow‑Datastar 在单用户场景下平均延迟比 Thymeleaf 低 1.3 倍、比 React 低 2.1 倍，网络传输量也更小。

**⚠️ 局限性**

局限性包括对离线功能的缺乏、依赖持续的服务器连接、未评估大规模并发性能，以及将复杂性从前端迁移到后端 DSL 导致的学习曲线和生态适配问题。

---

## 285. SAT Certificates for the Matrix-Multiplication Challenges over F2: All Ten `Expected-UNSAT` Instances Are Satisfiable, and a Type-3-Free Rank-23 Scheme

**arXiv ID:** 2607.29291 | [PDF](https://arxiv.org/pdf/2607.29291v1)

**作者:** Nick Palladinos `[一作]` `[通讯]`, Nick Palladinos

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文使用语义局部搜索与张量对称性和完美匹配构造，证明了21个3×3矩阵乘法SAT基准文件的可满足性，并给出完整的SAT证书。

**💡 创新点**

创新点在于揭示Challenge‑2文件的正单位约束仅限定出现而非排除，利用(3,2)群的等距变换与匹配技术构造满足所有约束的23阶张量分解，并首次给出一个无三阶项的23阶方案。

**🔧 技术方法**

技术方法包括布尔Brent方程的语义张量表示、基变量的局部搜索（直接修复与激活两阶段）、(3,2)^3等距作用、循环迹对称性、完美匹配以及一种保持张量不变的二元代数恒等式。

**📊 数据集**

数据集为Heule、Kauers、Seidl发布的矩阵乘法SAT基准仓库中的21个CNF文件，涵盖Challenge‑1至Challenge‑3的10+10+1实例。

**📈 对比分析**

通过单文件Python重现器，9秒左右完成所有21份证书的生成和完整语法/语义检查，性能远优于传统CDCL SAT求解器在这些编码上的表现。

**⚠️ 局限性**

局限性在于仅证明了给定的二进制编码可满足，未解决非交换域上的秩22问题，也未验证任何严格的无三阶项核心的不可满足性；结果对不同数值或编译环境的随机性不敏感，但无法推广到更一般的张量分解情形。

---

## 286. Translation with Thought: Difficulty-Adaptive Reasoning via Reinforcement Learning for Multi-Domain Machine Translation

**arXiv ID:** 2607.29287 | [PDF](https://arxiv.org/pdf/2607.29287v1)

**作者:** Yongshi Ye `[一作]` (Xiamen University), Xiaodong Shi `[通讯]` (Xiamen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种资源合理化的多域机器翻译模型，动态调节推理深度以匹配输入难度。

**💡 创新点**

结合难度自适应SFT与混合奖励RL，实现在翻译质量与推理效率之间的权衡，显著降低令牌使用。

**🔧 技术方法**

多阶段训练：利用多智能体蒸馏生成难度自适应长链式思考（CoT）数据，使用GPT-4o重写后在GRPO框架下进行强化学习。

**📊 数据集**

使用约7k难度自适应CoT示例（10个领域、3种语言对）做SFT，20k多域并行语料做RL，评估15个基准和59种未见语言。

**📈 对比分析**

与DeepSeek-R1、OpenAI o1等SOTA LRM及专用MT基线对比，平均质量在15个基准上达到或超过SOTA，同时令牌使用降低32–60%。

**⚠️ 局限性**

RL样本难度分布不均，来源LLM的偏见可能影响推理行为，未实现基于难度的奖励塑造，对跨句术语一致性等仍有不足。

---

## 287. TRACT: Temporally Routed Action Chunks with Chronological Phase Authority for Contact-Rich Manipulation

**arXiv ID:** 2607.29285 | [PDF](https://arxiv.org/pdf/2607.29285v1)

**作者:** Jiahao Liu `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 TRACT，一种将当前阶段接受与未来阶段边界路由分离的 temporally routed action chunking 方法，解决动作片段中的阶段不一致问题。

**💡 创新点**

创新点包括：① 单一累计边界保证从当前阶段到合法下一阶段的单向路由；② 引入基于 ACK 的响应欠缺积分器闭环以补偿接触动态导致的命令与执行偏差；③ 通过阶段管理器实现“chronological phase authority”，提升阶段判定的可靠性。

**🔧 技术方法**

使用 Transformer 生成器、阶段管理器、累计边界估计器、单向门控、以及 ACK‑驱动的执行闭环积分器等技术。

**📊 数据集**

在 50 条固定场景平面擦拭演示数据上进行训练，使用 Franka Panda 机器人与多视 RGB 及 13D Cartesian 观测。

**📈 对比分析**

与 ACT、带当前阶段信息的 ACT、平面化阶段-conditioned chunk 等六种变体对比，完整 TRACT 在 10 次试验中实现 100% 成功率、99% 擦拭完成率；相比平面化版本成功率从 30% 提升至 60%，并显著降低阶段歧义与停滞。

**⚠️ 局限性**

局限性在于仅验证单一任务、单机器人、10 次试验，缺乏跨任务泛化与统计显著性评估。

---

## 288. MDIR: A Task-Manifold Impedance Retargeting Method for Contact-Rich Teleoperation

**arXiv ID:** 2607.29271 | [PDF](https://arxiv.org/pdf/2607.29271v1)

**作者:** Liu Jiahao `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出一种单次演示的任务-流形阻抗重映射方法（MDIR），实现从固定笛卡尔阻抗演示到可变任务通道阻抗控制器的无损重参数化。

**💡 创新点**

创新点在于利用任务-流形阻抗表示（TMIR）构建任务通道，结合解析式的C2M重构和受约束的MPO优化，能够在保持任务响应的同时显著降低手腕力与控制功率。

**🔧 技术方法**

技术包括任务-流形阻抗表示（TMIR）、解析式的笛卡尔到流形重构（C2M）以及基于任务响应约束的MPO参数优化。

**📊 数据集**

使用的数据集为在Franka Emika Panda机器人上收集的三类接触丰富任务（平面擦拭、抓取-投放、推送）的5个演示与重映射执行实验。

**📈 对比分析**

与解析式基线C2M和统一缩放oracle（Scaling-best）对比，MDIR在所有15次闭环执行中通过任务检查，并将手腕力峰值、冲击、力方差及标称功率分别下降约30%、75%和50%等。

**⚠️ 局限性**

局限性包括仅单次演示重映射、约束集合保守导致可能未能达到最小阻抗、缺乏正式的能量安全与通过率泛化至更多任务与机器人模型。

---

## 289. OsteoCAD: A Human-in-the-Loop Cloud-Edge Framework for Bone Tumor Segmentation

**arXiv ID:** 2607.29266 | [PDF](https://arxiv.org/pdf/2607.29266v1)

**作者:** Maximo Rodriguez-Herrero `[一作]` (University Carlos III of Madrid), Jesus Carretero `[通讯]` (University Carlos III of Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发并验证了一套名为OsteoCAD的模块化云–边缘人机交互框架，支持资源受限医院在无GPU、本地高性能硬件条件下完成骨肿瘤CT分割模型的构建、训练、推理与部署。

**💡 创新点**

创新点：①将nnU-Net的自配置训练流程包装为可在容器化、分布式环境中通过远程GPU安全执行的服务；②提供完整的人机交互式数据标注与迭代扩增流程；③通过SSH隧道实现数据脱敏与零信任远程计算，降低隐私风险；④在真实临床场景中实现从零数据到Dice≈0.84模型的闭环部署。

**🔧 技术方法**

技术：Docker/容器化、FastAPI、React+Niivue前端、nnU-Net v2、SLURM调度、SSH加密隧道、CVAT标注、DICOM/NIfTI转换、数据预处理与增强、模型注册与版本管理。

**📊 数据集**

数据集：来自INR-LGII医院的67例骨肿瘤CT扫描（约20,000幅图像），其中16例已专家标注；其余51例用于推理与迭代扩增。

**📈 对比分析**

比较方法：使用Dice、IoU等标准医学影像分割指标对nnU-Net不同配置（2D、3D低/全分辨率）进行5折交叉验证；最终选取预处理后3D Full-Resolution模型，Dice=0.84±0.02；相较未预处理模型显著提升。

**⚠️ 局限性**

局限性：①样本量有限，缺乏跨机构验证；②专家标注/修正迭代成本较高；③尚未集成PACS或多中心联邦学习；④极小资源环境下的本地推理仍受限。

---

## 290. UniPolymer: A Unified Framework for Property Prediction, Structure Recommendation, and Evaluation in Polyimide Design

**arXiv ID:** 2607.29256 | [PDF](https://arxiv.org/pdf/2607.29256v1)

**作者:** Junquan Hu `[一作]` (Dalian University of Technology), Ben Fei `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 UniPolymer 统一框架，结合玻璃迁移温度（Tg）预测、目标条件生成、候选评估和结构推荐，实现从输入目标 Tg 到高质量聚酰亚胺结构的端到端设计；

**💡 创新点**

创新点在于（1）自监督化学语义预训练和结构一致性增强，构建跨域可迁移的聚合物表征；（2）多尺度信息融合（Transformer 序列、RDKit 描述符、Morgan 指纹）提升 Tg 预测精度；（3）连续-离散双重 Tg 条件表示，引导 SELFIES 自回归生成，提升目标一致性；（4）冻结预测器与结构约束相结合的候选筛选与排序；

**🔧 技术方法**

技术手段包括 Transformer 编码器/解码器、掩码语言模型、对比学习、层归一化、核采样、标签平滑、RDKit 分子描述符、Morgan 指纹、线性回归/Huber 损失；

**📊 数据集**

使用 PITg‑Curated 数据集，包含 10,066 条去重聚酰亚胺重复单元及其 Tg 标签（19–730 °C），并对预训练采用大规模无标签聚合物数据；

**📈 对比分析**

与 DNN、ANN、CatBoost、Importance‑Transformer、TransPolymer、polyBART、LLaMA‑3‑8B 等基线比较，UniPolymer 在 Tg 预测上实现 R² = 0.93、MAE = 17.87 °C；在候选生成与评估上 73.79% 的结构通过率、50.84% 的 Valid Hit@20，显著优于最强基线；

**⚠️ 局限性**

局限性包括（1）对高 Tg 区间的预测精度仍不理想；（2）缺乏合成可行性与不确定性评估；（3）未进行湿实验验证，推荐结构的实际可实现性和性能仍待验证。

---

## 291. CalibratedRubric: Task-Adaptive Rubric Banks for Open-Ended LLM Evaluation

**arXiv ID:** 2607.29252 | [PDF](https://arxiv.org/pdf/2607.29252v1)

**作者:** Mengting Chen `[一作]` (FinStep), Zuo Bai `[通讯]` (StepFun)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 CalibratedRubric 框架，用任务自适应评分、贝叶斯裁决一致性滤波和 IRT 信息聚合构建紧凑的评估 Rubric 库，降低专家成本。

**💡 创新点**

创新点包括：① 任务类型化的自适应评分规则；② 用 Beta‑Bernoulli 后验替代硬统一一致性；③ 将 IRT 信息函数与子模子贪婪选取结合，生成高信息、低冗余的 Rubric 集；④ 引入不确定性估计和分层排名。

**🔧 技术方法**

技术手段：贝叶斯判决一致性估计（Beta‑Bernoulli），2PL/1PL IRT 与信息函数，子模子贪婪测试组装，任务类型分类器，权重归一化，Bootstrap 置信区间。

**📊 数据集**

实验数据集：FinResearchBench‑v2、HealthBench、HelloBench 以及 JudgmentBench。

**📈 对比分析**

对比方法：与传统硬统一一致性+二元方差过滤基线相比，任务自适应评分提升分辨率；贝叶斯滤波将 κ 从 0.604 提升至 0.743；IIF‑贪婪相比随机和纯 IIF 在目标相关性下将所需 Rubric 数量从 131 降至 49（FinResearch），并在所有六个响应块中提升跨拟合 Spearman AUC。整体在 FinResearch 获得 97.13 分差，在 JudgmentBench 达到 0.8587 的人类一致率。

**⚠️ 局限性**

局限性：贝叶斯滤波需要至少三名评审；在高风险领域预测弱；基于候选集的限制可能遗漏重要维度；未建模评审者间的相关偏差；在小规模排行榜上收益有限；未完全消除人类‑LLM 的标签偏差。

---

## 292. Data Turnstile: A Scalable Open Framework for Function-Calling Data Generation

**arXiv ID:** 2607.29250 | [PDF](https://arxiv.org/pdf/2607.29250v1)

**作者:** Goutham Ramakrishnan `[一作]` (Amazon AGI), Megha Sharma `[通讯]` (Amazon AGI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过基于角色的DAG模板生成高质量的工具调用交互数据，并利用逐步验证与错误反馈来提升数据可靠性，随后对小型语言模型进行监督微调。

**💡 创新点**

创新点在于：①逐步角色生成与验证的框架（Data Turnstile），①可通过模板控制多样性与复杂度，②支持从域政策文档生成多轮对话，③实现全流程可开源、无需专有LLM。

**🔧 技术方法**

技术手段包括：DAG角色模板、逐步生成与预检、错误反馈重试、局部执行验证、vLLM加速、Qwen系列教师模型、可选CoT推理、工具调用加权损失。

**📊 数据集**

使用的数据集有：1K+ API + 100K+ 交互的 Synthetic Domains 数据集、BFCL 单轮功能调用基准、τ²-bench Telecom 多轮基准；此外还利用 xLAM、Glaive 等公开 API 作为对比。

**📈 对比分析**

在 BFCL 单轮基准上，0.6B SLM 在无思考模式下精度达 75.9%，超过基线 0.6B (67.4%)，并接近 1.7B(78.4%) 与 4B(79.9%)；在 τ²-bench Telecom 上，1.7B 通过率 31.1% 超过 32B 基线 27.4%，0.6B 通过率 24.6% 超过 3.5%；表明微调后的小模型可与大模型竞争；单轮任务无思考优于思考，反之多轮任务需思考。

**⚠️ 局限性**

局限性包括：①依赖教师模型的质量，强大模型可生成更好数据；②模板设计需领域专家参与，缺乏自动化；③生成的数据仍可能继承教师模型的偏见；④仅基于API模式与执行验证，未覆盖所有真实场景；⑤大规模多轮生成对资源仍有一定需求。

---

## 293. Cross-Lingual Transfer for Machine Translation in Turkic Languages

**arXiv ID:** 2607.29355 | [PDF](https://arxiv.org/pdf/2607.29355v1)

**作者:** Omer Burak Cinar `[一作]` (Middle East Technical University), Cagri Toraman `[通讯]` (Middle East Technical University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了五种突厥语（土耳其语、阿塞拜疆语、乌兹别克语、哈萨克语、吉尔吉斯语）在固定目标语言下的跨语言转移矩阵，系统分析了转移来源、目标和翻译目标的相互作用；

**💡 创新点**

创新点在于首次对突厥语内部的跨源转移进行细粒度的固定目标矩阵分析，揭示了转移强度与语言族群子分支、脚本一致性及翻译目标相关的非对称性；

**🔧 技术方法**

使用了mT5（300M）模型进行持续预训练（CPT）和微调，并结合了多种评估指标（BLEU、chrF、COMET、COMETKiwi）来量化恢复率；

**📊 数据集**

数据来源包括公开的单语料（Wikipedia、CC‑100 等）与双语对（KazParC、NTREX、FLORES+、OPUS），并通过回译、LaBSE 句子对齐生成合成平行语料；

**📈 对比分析**

通过恢复率（相对同目标下的性能）构建矩阵，并在 XWMT 与 Tatoeba 上验证稳定性；结果显示，土耳其语-阿塞拜疆语、哈萨克语-吉尔吉斯语对转移最强，拉丁化在脚本不匹配的方向上显著提升 BLEU/chrF，但对 COMET 并无一致提升；

**⚠️ 局限性**

主要限制是大部分训练语料为合成回译对，可能引入模型偏差，且研究仅覆盖五种突厥语，结果对其他语族的泛化仍需验证。

---

## 294. Tool Specifications Matter: Uncovering and Mitigating Safety Risks in AI Agents

**arXiv ID:** 2607.29254 | [PDF](https://arxiv.org/pdf/2607.29254v1)

**作者:** Minghui Pan `[一作]` (Beijing University of Posts and Telecommunications), Zhenpeng Chen `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过白盒分析发现，LLM 代理在使用 schema 格式的工具规范时会削弱内部拒绝信号，导致安全性下降，并提出 SafeKeep 方案在推理时将安全判断与工具执行解耦，使用扁平化文本规范来恢复拒绝能力。

**💡 创新点**

创新点在于：①系统定位工具规范的 schema 格式是安全退化的根源；②通过“Schema 方向”与“拒绝方向”的对比揭示其对内部表示的抑制作用；③设计了无需参数更新、适用于任意 LLM 的推理时安全框架 SafeKeep。

**🔧 技术方法**

主要技术包括：内部隐藏状态的拒绝方向提取、Schema 方向计算、激活层干预（对抗 Schema 方向）、推理时两阶段安全判断与执行控制。

**📊 数据集**

使用的数据集有：ToolSafety（构建 400 组有害/安全请求）、AgentHarm（176 有害/安全对照请求）和 InjecAgent（1054 注入攻击案例）。

**📈 对比分析**

与基线（Base、SafeJudge、SafePrompt、SafeHarbor）比较，SafeKeep 在 AgentHarm 上拒绝率从 23.8% 提升至 70.6%，在 InjecAgent 上攻击成功率从约 25% 降至 2.5%，同时保持甚至提升任务处理准确率。

**⚠️ 局限性**

局限性包括：仅在四个 LLM（Llama3.1-8B、Qwen3-8B、Gemini3.1-Flash、GPT5.4-mini）验证；对更大模型或更复杂注入攻击的泛化未知；转换工具规范为扁平文本的手动步骤可能对某些工作流产生影响。

---

## 295. Don't Mix Rewards, Mix Policies: Policy Decomposition and Optimization for Multi-Reward RL

**arXiv ID:** 2607.29246 | [PDF](https://arxiv.org/pdf/2607.29246v1)

**作者:** Ruiming Liang `[一作]` (CASIA), Xianyuan Zhan `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 PRISM，一种在策略空间而非奖励空间进行多奖励强化学习的框架，通过为每个奖励学习正向子策略并统一使用一个全局负向子策略，实现一次训练后可在推理时按需调节多偏好；

**💡 创新点**

核心创新在于将多奖励的组合从奖励空间迁移到策略空间，使用单一全局负策略捕捉所有失败模式，并通过共享前缀条件化子策略实现参数共享与高效解码，同时允许推理时通过权重直接控制偏好；

**🔧 技术方法**

采用了基于 GRPO 的优势估计、KL 正则化的单步改进、正负策略分解与对数概率线性组合、共享语言模型+前缀嵌入的异步更新以及并行批量混合采样等技术；

**📊 数据集**

实验数据集包括科学推理的 SciKnowEval/GPQA/ScienceQA、工具调用的 ToolRL/ BFCL‑v3 以及有用性‑安全性的 Alpaca、HH‑RLHF、PKU‑SafeRLHF；

**📈 对比分析**

与 GRPO Sum/Product、GDPO 等奖励空间基线对比，在科学推理、工具调用和安全性三大任务上均获得最高综合得分；在 DeepSeek‑R1‑1.5B 上提升 17.8 分，在 Qwen‑1.5B‑Instruct 上提升 8.0 分，在工具调用任务中排名第一，且对奖励数量增加更稳健；

**⚠️ 局限性**

局限在于子策略数量随奖励数量线性增长，仍需经验性选择全局负策略权重，对更复杂动态权衡或大规模多奖励场景的适用性尚待进一步验证。

---

## 296. Small Is Enough: Per-User Style Rewriting of AI-Edited Text via LoRA Adapters

**arXiv ID:** 2607.29238 | [PDF](https://arxiv.org/pdf/2607.29238v1)

**作者:** Antorweep Chakravorty `[一作]` `[通讯]` (University of Stavanger), Antorweep Chakravorty (University of Stavanger)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个本地单用户系统 InMyStyle，利用 LoRA 适配器在不加提示的情况下将 AI 编辑文本按用户写作风格重写。

**💡 创新点**

通过构造多家助手模型生成的 AI‑阴影文本与用户原文对齐，并在用户端训练专属 LoRA，实现隐私优先、无需指令的个性化改写。

**🔧 技术方法**

采用 LoRA 低秩适配、4‑bit QLoRA 量化、响应仅损失、长度感知预算与句子边界拆分、三类助手模型阴影生成以及 LLM 评判等技术。

**📊 数据集**

基于一名用户的 36 篇科研论文共 487 段落（219 对评测），用三种本地助手模型生成阴影文本，形成训练对。

**📈 对比分析**

使用作者归属、内容相似度、目标相似度、AI 提示减少、词学改进等五指标计算复合得分；四个模型（0.5B–7B）在贪婪和采样下复合分均为 0.69，LLM 评判显示 7B 的 AI‑感知分最低，但训练成本最高，3B 在性能与效率上达到折衷。

**⚠️ 局限性**

仅验证单用户单语言场景，样本量有限，阴影生成与扰动手工设定，评价指标与人类感知不完全一致，未证明跨模型或跨用户的普适性。

---

## 297. FBFM: A Training-Free Asynchronous Feedback Mechanism for Flow-Matching in World-Action Models Execution

**arXiv ID:** 2607.29235 | [PDF](https://arxiv.org/pdf/2607.29235v1)

**作者:** Peize Li `[一作]` (DeepCybo), Shanghang Zhang `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种无训练的推理机制——反馈流匹配（FBFM），将实时观测反馈与已生成的多步动作/视觉序列进行细粒度闭环纠正。

**💡 创新点**

创新点在于把已执行动作的实时状态观测与跨块的动作一致性都视为带掩码的伪逆测量，利用Flow Matching的向量场在推理时动态纠正生成过程，从而避免传统块级重定位造成的时间分辨率不足。

**🔧 技术方法**

主要技术包括：Flow Matching（连续流生成）、伪逆引导（masked pseudoinverse guidance）、时间对齐的状态/动作掩码、跨模态端点雅可比矩阵的向量乘积，适用于阶段式和联合式World–Action模型。

**📊 数据集**

实验使用RoboTwin2.0（42个任务）和LIBERO（四个子套件共40个任务）进行任务成功率评估，并在真实机器人腕部球停止实验中评估视频预测误差。

**📈 对比分析**

与基线（无反馈）相比，阶段式WAM（LingBot-VA）在RoboTwin2.0上宏观提升约3.0个百分点；联合式WAM（DreamZero）在LIBERO整体提升约0.6个百分点，单个子套件提升2.5个百分点；真实视频预测MAE/PSNR均得到改善。

**⚠️ 局限性**

局限性包括：在联合式WAM中因模型加速推理导致反馈作用被稀释；假设编码器–解码器可近似线性（h†(h(x))≈x）不一定成立；引入雅可比向量乘积与反馈管理增加推理开销，难以满足实时控制频率。

---

## 298. CorrelationFlow: A Training-Free Geometric Approach for LiDAR Scene Flow Estimation

**arXiv ID:** 2607.29237 | [PDF](https://arxiv.org/pdf/2607.29237v1)

**作者:** Minh-Quan Dao `[一作]` (Nantes Université), Holger Caesar `[通讯]` (TU Delft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种完全无训练的LiDAR场景流估计框架CorrelationFlow，利用BEV占据图的连通分量与归一化交叉相关实现物体运动估计；

**💡 创新点**

核心创新在于将三维场景流问题转化为二维BEV图像的相关最大化和连通分量标签，既无学习参数又可跨域泛化；

**🔧 技术方法**

采用BEV投影、连通分量标记、归一化交叉相关、稀疏关键点匹配及中值聚合等经典几何与图像处理技术；

**📊 数据集**

在Argoverse 2 2026 Multi‑Dataset Scene Flow Challenge（含AEVA、Argoverse2、nuScenes、TruckScenes、Waymo Open Dataset）等多传感器、多平台数据集上进行评测；

**📈 对比分析**

与多种无监督学习方法（SeFlow、TeFlow、VoteFlow、SSF、RVLoss等）对比，CorrelationFlow在多域测试中排名第二，尤其在长距离点上表现更稳健；

**⚠️ 局限性**

局限包括仅估计平面位移（缺乏纵向运动和旋转建模）、依赖精确的自机运动补偿、对稀疏或遮挡严重对象的关键点匹配依赖较高，且易受像素尺寸和膨胀阈值影响。

---

## 299. The persuasive power of large language models does not depend on their perceived national origin

**arXiv ID:** 2607.29334 | [PDF](https://arxiv.org/pdf/2607.29334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 300. An Algorithmic Perspective on Information Visualization

**arXiv ID:** 2607.29360 | [PDF](https://arxiv.org/pdf/2607.29360v1)

**作者:** Wouter Meulemans `[一作]` `[通讯]` (TU Eindhoven), Wouter Meulemans (TU Eindhoven)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出算法视角的可视化研究框架，强调在算法设计前对可视化问题进行建模，定义“metric idiom”和“computational problem”，并将其嵌入 Munzner 的四层模型中。

**💡 创新点**

创新性地将可视化质量的多面向度转化为可度量的 metric idiom，并阐述 adequacy 与 solvability 的权衡，提供一套系统化的评估与比较可视化算法的方法。

**🔧 技术方法**

主要采用理论建模、数学优化与算法分析技术，并通过案例讨论（如矩阵排序、网格地图、重叠消除、专题地图等）验证框架。

**📊 数据集**

文章并未提供专门的实验数据集，而是基于已有文献中的公开图数据、网络数据和地图数据进行讨论和分析。

**📈 对比分析**

通过定义多维度的 metric idiom，使用量化指标与人类评估来验证 adequacy，并在满足约束与优化目标的前提下比较不同算法的解质量与可计算性；性能表现取决于具体算法，可实现交互式或批处理，并可提供近似/最优性保证。

**⚠️ 局限性**

主要局限在于：需先验证 metric idiom 的 adequacy，计算问题往往难以同时满足多面向度导致复杂度高；缺乏统一的实验评价体系；以及算法实用性与人类研究之间的衔接仍需进一步探索。

---

## 301. Hypergamigication Through Integrating Game Engines and Learning Management Systems: Ender's Game

**arXiv ID:** 2607.29300 | [PDF](https://arxiv.org/pdf/2607.29300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 302. Versatile On-device Adaptation at the Edge by Unifying Few-shot, Zero-shot, Continual, and In-context Learning

**arXiv ID:** 2607.29353 | [PDF](https://arxiv.org/pdf/2607.29353v1)

**作者:** Douwe den Blanken `[一作]` (Delft University of Technology), Charlotte Frenkel `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Embedder-Centric Learning框架，统一了四种边缘在线学习场景（少样本、持续、零样本、上下文学习），实现了在单一硬件上同时支持图像、音频和符号序列的自适应推理与学习。

**💡 创新点**

创新点在于把四种学习方式抽象为共用嵌入器加场景特定头的结构，并选用Temporal Convolutional Network（TCN）作为跨模态嵌入器，既兼容多种输入，又能在标准矩阵‑向量加速器上高效执行；该框架在极低功耗硬件上实现所有场景的完整端到端学习。

**🔧 技术方法**

关键技术包括：TCN嵌入器、原型网络（FSL）、关系网络（ZSL）、FC/MLP头（CL、ICL）；量化（4‑bit log₂ 权重、4‑bit 激活）以及Chameleon SoC的矩阵‑向量处理单元，支持在芯片内完成学习与推理。

**📊 数据集**

实验使用四个公开数据集：Omniglot（FSL）、NeuroBench keyword FSCIL（CL）、Fluent Speech Commands（ZSL）、RegBench（ICL）。

**📈 对比分析**

与现有软件/硬件基线对比，FSL 5‑way 1‑shot达96.8%/83.3%新记录，功耗比同类芯片低2.3×；CL 在200类 5‑shot下实现71.8%准确率，功耗仅9.5 mW；ZSL 在5‑way语音句子分类上实现60.6%准确率，学习功耗8.2 mW；ICL 在RegBench上每令牌能耗16.8 nJ、功耗7.83 mW，接近Transformer性能。

**⚠️ 局限性**

主要局限：TCN嵌入器在不同模态需离线预训练，量化导致ZSL准确下降；框架目前仅支持无梯度更新，无法利用梯度下降提升；实验规模受Chameleon资源限制，需进一步验证对更大模型与更复杂任务的可扩展性。

---

## 303. DualDiT: A Conditional Dual-Output Diffusion Transformer for Joint OCT Image and Segmentation Mask Generation

**arXiv ID:** 2607.29337 | [PDF](https://arxiv.org/pdf/2607.29337v1)

**作者:** Fernando García-Torres `[一作]`, Valery Naranjo `[通讯]` (Universitat Politècnica de València)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了条件双输出扩散Transformer（DualDiT），能够同时生成外周鼠眼视网膜OCT图像及其上层细胞层（URCL）分割掩模。

**💡 创新点**

创新点在于：①将图像与掩模编码到共享潜在空间并用单一Transformer进行联合扩散，实现图像与掩模的一致性；②在标注稀缺的OCT场景中首次提出联合生成图像-掩模的框架。

**🔧 技术方法**

采用预训练VAE压缩潜在空间，条件扩散Transformer（DiT-XL/2）进行联合噪声预测，并使用classifier-free guidance；下游使用U‑Net进行分割，并通过专家评估进行真实性验证。

**📊 数据集**

使用的数据库是338个外周鼠眼OCT B扫描（约512×256像素）与手工标注的URCL掩模，按两种保存介质（琼脂/树脂）划分，样本来源为7只雄性与3只雌性老鼠。

**📈 对比分析**

与像素空间DDPM和潜在空间LDM进行对比，评估指标包括FID、sFID、分割Dice/IoU及专家误判率；DualDiT在生成质量上获得最低FID 56.14和sFID 114.35；在下游分割任务中，DualDiT+1200样本将Dice提升至0.927、IoU提升至0.868，且专家平均误判46%假样本为真。

**⚠️ 局限性**

主要限制包括：生成图像分辨率仅为512×256，训练样本量有限且同体内B扫描高度相关，导致泛化受限；扩散模型的多样性与真实度平衡仍需进一步提升。

---

## 304. BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning

**arXiv ID:** 2607.29302 | [PDF](https://arxiv.org/pdf/2607.29302v1)

**作者:** BWM Team `[一作]` `[通讯]`, BWM Team

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了低成本、高保真、动作条件的世界模拟器 BWM，用于机器人学习的数据增强与策略评估。

**💡 创新点**

创新点在于：① 将轨迹回放、重叠片段采样与初始观测增强整合到数据管线；② 通过初始环境引导、动态历史窗口与双路径动作注入（AdaLN + 跨注意力）实现精确的状态保持与动作控制。

**🔧 技术方法**

采用预训练视频扩散模型 Wan2.2 进行任务特定微调，结合轨迹回放、重叠采样、动作对齐训练以及跨注意力+AdaLN 双路径动作注入技术。

**📊 数据集**

主要使用 RoboTwin 2.0 机器人视频数据集进行训练；在 WorldArena 16 个任务（共 2,000 训练 + 500 测试视频）以及 ARX X5 平台的 6 个物理任务（Fold Towel、Open Drawer 等）进行评估。

**📈 对比分析**

在 WorldArena 基准中，与多种文本/动作条件生成模型对比，BWM 获得 EWMScore 63.51，排名第一；在物理机器人评估中，BWM 生成的轨迹使政策成功率提升至 71%（相较于基线提升 20%），评估器与硬件结果的 Pearson 相关系数达到 0.908。

**⚠️ 局限性**

局限性：仍依赖大量高质量真实轨迹；在未知任务或相机配置下的泛化能力有限；训练成本高，需多台 GPU；对极低延迟实时控制的支持尚未成熟。

---

## 305. Assessing the Generalization of Graph Neural Networks for Fault Location Across Increasing Distributed Energy Resource Penetration Levels

**arXiv ID:** 2607.29293 | [PDF](https://arxiv.org/pdf/2607.29293v1)

**作者:** Burak Karabulut `[一作]` (Ghent University), Jochen L. Cremer `[通讯]` (TU Delft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在分布式能源（DER）渗透率不断提高的配电网络中，基于时空图神经网络（STGATv2）对故障定位的性能与泛化能力进行系统评估，并与纯时序（GRU）、纯空间（GATv2）以及传统机器学习（RF、SVM）模型进行比较。

**💡 创新点**

创新点包括：① 在不同DER渗透水平（10%、25%、50%）与两种DER布局（局部化、分散化）下，首次对STGATv2的时空建模优势与模型泛化进行量化比较；② 揭示模型在高渗透训练时对低渗透测试保持良好性能，而低渗透训练时对高渗透测试表现显著衰退的非对称泛化特性；③ 在加入测量噪声的条件下验证STGATv2对细微故障信号的鲁棒性。

**🔧 技术方法**

技术方案：使用GRU作为时序特征提取器，将GRU隐藏向量送入改进的GATv2实现空间注意力聚合；采用soft voting聚合节点级预测；对比基线模型包括GRU、GATv2、RF、SVM；训练采用交叉熵损失、AdamW优化，评估指标为宏F1分数。

**📊 数据集**

数据集：基于IEEE 123节点配电馈线的仿真数据（OpenDSS/PyDSS），包含25个故障位置、11种短路类型、20步窗口、2.5M样本；每个样本为3相RMS电压序列；训练/验证/测试比例为70%/15%/15%，并在不同DER渗透率与布局下生成数据。

**📈 对比分析**

比较方法：在同一数据集上训练模型后，分别在相同渗透率、不同渗透率以及不同噪声水平下测试；STGATv2在分布式网络上实现92–94%宏F1的最高表现；在低渗透训练、10%→50%测试时仍保持81–84%宏F1；在测量噪声下，STGATv2保持>85%宏F1，GRU和GATv2则显著下降。

**⚠️ 局限性**

局限性：仅使用合成仿真数据，未验证在真实配电网或更大规模网络上的表现；未涵盖超过50%的极端DER渗透率；未考虑逆变器动态和电压调节器等细节；模型泛化仍受限于训练所覆盖的DER布置和电网拓扑。

---

## 306. Data-Driven Batteryless Channel Sounding for Wi-Fi 8-Inspired Downlink MU-MIMO

**arXiv ID:** 2607.29288 | [PDF](https://arxiv.org/pdf/2607.29288v1)

**作者:** Muhan Zhang `[一作]` (City University of Hong Kong), Ming Gan `[通讯]` (Huawei Technologies Co., Ltd)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在Wi‑Fi 8/IEEE 802.11bn启发的下行多用户多天线系统中，使用电池无源覆盖（Glaze）后，如何优化频道声响周期以最大化主动Wi‑Fi链路与被动链路的总吞吐量。

**💡 创新点**

创新点在于：①提出统一的多用户声响+被动叠加体系结构；②构建跨层包级吞吐量模型，将声响开销、CSI衰老、MCS、被动衰减和数据速率耦合；③提出基于数据驱动搜索的声响周期优化方法，并给出理论分析说明何时出现过载与衰老两种平衡态；④展示被动叠加能在不同MCS下对最优声响周期产生左移/右移或不变的影响。

**🔧 技术方法**

所用技术包括：多用户MIMO声响协议、Beamforming预编码、CSI衰老模型、Glaze被动叠加（幅度衰减与曼彻斯特编码）、包级交叉层吞吐量评估、数据驱动的一维搜索优化、MATLAB链路仿真（HE/TGax代理、TGax Model‑D衰落）。

**📊 数据集**

使用的“数据集”为仿真生成的多个独立频道实例（基于IEEE TGax Model‑D、5.25 GHz 20 MHz带宽、0.089 m/s 环境速度），每个实例记录包级载荷、被动解码结果、时隙持续时间和声响开销，用于训练与验证。

**📈 对比分析**

通过对比固定周期声响与数据驱动优化声响周期，实验结果显示：在低MCS（0–3）下，最优周期趋于最长（180 ms）；在高MCS（4–9）下，最优周期随MCS上升而显著下降（从≈60 ms到≈10 ms）。被动叠加在高MCS下可提升平均吞吐量约10 Mbps，但在低MCS下对吞吐量影响不大，甚至略有损失。

**⚠️ 局限性**

局限性包括：①只考虑单AP多STA场景，未验证多AP互斥；②未联合调节衰减深度与嵌入速率，只固定在若干值；③仅在仿真环境下验证，缺乏实测SDR或硬件验证；④被动接收模型简化为包级平均误码率，未考虑硬件非理想或前端失真。

---

## 307. FillGS: Filling Observation Gaps in 4D Gaussian Splatting via Viewpoint-Time Selection and Generative Refinement

**arXiv ID:** 2607.29284 | [PDF](https://arxiv.org/pdf/2607.29284v1)

**作者:** Takashi Otonari `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在4D Gaussian Splatting（4DGS）在视角稀疏环境下出现观察缺口时，提出了一套主动时空虚拟视角选择与生成图像细化的完整流程，用以提升动态场景的渲染质量。

**💡 创新点**

创新点一：基于渲染敏感度与运动感知的观察稀疏度指标，主动挑选能覆盖稀疏时空区域的虚拟视角；创新点二：对生成图像进行冲突检测和可靠性掩模，采用一致性掩模和运动感知共视掩模实现稳健的4DGS微调，避免生成错误导致的误导。

**🔧 技术方法**

使用技术包括4D Gaussian Splatting、视频扩散模型（继承自3DGS-Enhancer）、渲染敏感度分析、运动感知观察缺失评分、RoMa一致性检测、共视权重机制以及基于损失加权的微调策略。

**📊 数据集**

数据集：Neural 3D Video 数据集与 Technicolor 数据集，作者自行构造了稀疏摄像机训练/测试拆分（仅使用两台摄像机训练，测试包括插值与外推视角）。

**📈 对比分析**

与基线（E-D3DGS、插值选择、FisherRF、Coverage(3D/4D) 等）在插值与外推两种评估场景下进行比较，FillGS 在 PSNR、SSIM、LPIPS、FID、DINOv2 等指标上均优于所有基线，显示显著的质量提升。

**⚠️ 局限性**

局限性：4DGS 的表示能力有限，复杂运动仍难以完全捕捉；扩散模型的生成质量限制，需谨慎控制虚拟视角范围；若扩散模型提升，方法效果可进一步增强。

---

## 308. ParaASR: Multi-Token Prediction for Fast and Long-Context LLM-Based Speech Recognition

**arXiv ID:** 2607.29279 | [PDF](https://arxiv.org/pdf/2607.29279v1)

**作者:** Qingjian Lin `[一作]` (StepFun), Daxin Jiang `[通讯]` (StepFun)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于4B LLM解码器的语音识别系统，利用多Token预测（MTP）在每个前向步骤发出多达六个未来Token，并通过自回归验证保证最终转录准确性。

**💡 创新点**

创新点在于将多Token预测与自回归验证相结合，突破了解码规模与延迟的权衡；系统支持32K上下文窗口，可一次性处理长达30分钟音频，实现高并行解码同时保持准确性。

**🔧 技术方法**

采用冻结的音频编码器 + 线性适配器 + 4B Transformer LLM解码器，加入5个MTP分支；使用分阶段训练（预训练→SFT→MTP对齐+校准），SpecAugment、ROVER、多语种混合数据以及自回归验证机制。

**📊 数据集**

训练与评测使用公开的1.356T音频-文本基础数据，短语音约100K小时，50K小时长语音伪标注；评测集包括AISHELL、WenetSpeech、LibriSpeech、Common Voice、FLEURS、VoxPopuli、Earnings22等中文、英文及长音频数据。

**📈 对比分析**

在中文、英文及长音频任务上与VibeVoice-ASR、FunASR-Nano、Doubao-ASR-2603、Qwen3-ASR-1.7B等基线对比，平均CER 2.97%、WER 3.68%、长form 3.70%，单GPU实时因子RTF仅0.0053，展示了高准确率与极低延迟的双重优势。

**⚠️ 局限性**

局限性包括：需要大显存和高算力的4B模型；MTP接受率随lookahead长度递减，MTP-7效果不佳；对极端噪声或多语种混合的鲁棒性尚待进一步验证；系统在极端资源受限环境下的可扩展性有待评估。

---

## 309. TAVI-TEC: An AI-Based Tool for Procedural Planning of Transcatheter Aortic Valve Implantation

**arXiv ID:** 2607.29243 | [PDF](https://arxiv.org/pdf/2607.29243v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 310. Training-Free Entity-Level Few-Shot Segmentation of Remote Sensing Images with Advection Refinement

**arXiv ID:** 2607.29278 | [PDF](https://arxiv.org/pdf/2607.29278v1)

**作者:** Xueting Bai `[一作]` (Nanjing University of Information Science and Technology), Huan Ni `[通讯]` (Nanjing University of Information Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个训练免费、基于SAM3的实体级少样本遥感图像跨域分割框架ELFSS-AR。

**💡 创新点**

创新点是将像素级推理改为实体级推理，并通过二维推移方程对特征与响应空间进行语义细化，实现高效无训练的跨域适配。

**🔧 技术方法**

使用SAM3预训练特征、文本嵌入、实体原语生成、多模态语义势场构造、二维推移方程进行特征与响应细化、实体级分类。

**📊 数据集**

在GID、Five-Billion-Pixels (FBP)、Potsdam、Vaihingen、iSAID等五个遥感语义分割数据集上进行评估。

**📈 对比分析**

与传统FSS、CDFSS以及训练免费OVSS方法比较，在5-shot设置下平均mIoU达到36.37%，比最佳对手高约10个百分点，性能显著提升。

**⚠️ 局限性**

局限在实体原语构造的鲁棒性与对极细小目标的识别能力仍需进一步改进。

---

## 311. Metamorphic Testing of Transpilers via Mutation Consistency of Programs

**arXiv ID:** 2607.29247 | [PDF](https://arxiv.org/pdf/2607.29247v1)

**作者:** Enea Raffaele Ilario Papaleo `[一作]` (University of Milano-Bicocca), Giovanni Denaro `[通讯]` (University of Milano-Bicocca)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种针对源到源编译器（transpiler）的变异一致性（mutation consistency）变形测试技术，并在工业合作伙伴的DSL转译器上进行了评估。

**💡 创新点**

创新点在于将变形测试的判据从执行时的二进制行为迁移到生成的目标源代码结构上，利用变异操作直接构造输入源并比较输出源的结构差异，以便检测转译错误。

**🔧 技术方法**

采用了基于变异分析的变形操作、元关系（mutation operators + consistency checkers）以及自定义的测试生成器与判据执行器，并使用Java实现。

**📊 数据集**

数据集为合作方DSL的合法程序，使用自动生成器（rmutt.js）产生297条有效源文件，并在此基础上注入约2000个字节码级故障。

**📈 对比分析**

通过将该技术与仅执行变异（无判据）的纯fuzzer做对比实验（10次12小时测试），发现变形测试能发现39~43个隐藏故障，明显高于仅触发崩溃的5~16个，且判据执行对总耗时几乎无影响。

**⚠️ 局限性**

局限包括仅在单一转译器与特定DSL上验证，使用的故障注入方式不一定覆盖所有真实缺陷，且对变形规则的手工定义与维护成本较高。

---

## 312. JUNO: Aggregated Vector Consensus for Optimal Asynchronous Common Subset

**arXiv ID:** 2607.29244 | [PDF](https://arxiv.org/pdf/2607.29244v1)

**作者:** Liangrong Zhao `[一作]` (Monash University), Jiangshan Yu `[通讯]` (University of Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出聚合向量共识原语并实现 Juno，构建异步公共子集协议，在消息复杂度上实现 O(n^2) 并显著提升吞吐率。

**💡 创新点**

创新点在于将并行二值一致性聚合为向量一致性，采用 provable broadcast 取代可靠广播，减少消息量并通过聚合投票降低通信成本。

**🔧 技术方法**

采用异步 BKR 架构、Provable Broadcast、聚合向量共识、共识硬币、四阶段 PB 等技术实现高效一致性。

**📊 数据集**

实验使用 Google Cloud 地理分布环境，模拟 250 字节交易，批量至 20k 交易进行吞吐率与延迟评估。

**📈 对比分析**

与 HoneyBadgerBFT 与 Dory 进行基准对比，Juno 在吞吐率上分别提升 93% 与 HoneyBadgerBFT、47% 与 Dory，延迟显著下降。

**⚠️ 局限性**

局限在于极大批量或高 f 时可能需多轮投票导致延迟上升，PB 的一致性不及 RBC，且实现依赖签名与网络性能。

---

## 313. RecHarness: A Bandit-Routed Agentic Harness for Self-Evolving Recommender Systems

**arXiv ID:** 2607.29241 | [PDF](https://arxiv.org/pdf/2607.29241v1)

**作者:** Haoran Ling `[一作]` (Georgia Institute of Technology), Peng Jiang `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种名为RecHarness的基于Bandit路由和LLM推理的自动化推荐系统优化框架。

**💡 创新点**

将推荐模型优化拆分为Bandit路由选择优化方向与LLM生成具体代码修正的两步流程，并引入跳跃-盆地机制与实验技能反馈，实现了在有限实验预算下的高效、稳定探索。

**🔧 技术方法**

结合Thompson采样多臂Bandit算法、LLM生成式代码改进、实验技能文本总结、跳跃-盆地机制和基于验证反馈的贝塔后验更新。

**📊 数据集**

在Amazon Reviews四个子集（Movies、Scientific、Electronics、CDs）与KuaiRec数据集上进行离线实验，并在短视频广告平台进行线上A/B测试。

**📈 对比分析**

与基线模型、公开论文结果以及随机路由、LLM路由、无Bandit对照实验对比，RecHarness在HR@10、NDCG、XAUC、MAE等指标上提升幅度多达数十个百分点，线上指标提升ADVV 2.08%、Revenue 0.53%、Exposure 0.56%。

**⚠️ 局限性**

仍依赖人工定义的优化方向集合，对极端大规模实验预算或复杂模型结构的自动发现有限；LLM生成代码可能存在错误或效率低下；跳跃机制阈值与窗口参数需要经验调优。

---

## 314. Analysing User Reviews to Identify User Concerns Around Permissions in AI Apps

**arXiv ID:** 2607.29343 | [PDF](https://arxiv.org/pdf/2607.29343v1)

**作者:** Babar Shah `[一作]` (Zayed University), Muhammad Junaid `[通讯]` (Islamia College University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过机器学习方法对AI应用用户评论进行权限相关分类，并对分类结果进行聚类分析，揭示用户对权限请求的关注点与情绪。

**💡 创新点**

使用AI生成的安全与权限评论作为检索训练样本，避免手工标注；先区分隐私评论，再筛选权限评论；通过无监督聚类探究情感驱动的关注主题。

**🔧 技术方法**

采用基于USE的深度平均网络进行二分类，使用BERT/USE向量表示；利用余弦相似度挑选训练样本；K-means聚类与轮廓系数确定簇数。

**📊 数据集**

人类生成的AI应用评论（约1.6亿条）与使用OpenAI GPT‑4生成的安全与权限评论（550条安全、100条权限）以及从Google Play抓取的评论数据。

**📈 对比分析**

与基于n‑gram的安全评论分类方法对比，模型在隐私分类中的准确率为0.79，权限分类准确率为0.82；召回率和F1分别为0.87/0.84，整体性能较传统方法提升约3–5%。

**⚠️ 局限性**

训练样本主要基于AI生成评论的相似性筛选，可能引入偏差；测试集规模有限（仅100条），评估不够充分；未直接使用生成评论训练，导致可复现性受限。

---

## 315. CALM-AH: An ABAW11-Calibrated Multimodal Ensemble with Reliability-Gated Multi-Expert Consensus for Video-Level Ambivalence and Hesitancy Recognition

**arXiv ID:** 2607.29310 | [PDF](https://arxiv.org/pdf/2607.29310v1)

**作者:** Wenzhuo Sun `[一作]` (Monash University), Pamela Carreno-Medrano `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出CALM-AH多模态集成模型用于视频级别的矛盾与犹豫(A/H)识别，并在此基础上设计可靠性门控多专家共识（RG‑MEC）机制实现推理时的决策融合。

**💡 创新点**

创新点包括：① 将文本、音频、视觉及行为统计四大模态通过阈值校准后的硬投票进行组合；② 引入锚点与全一致门控的非交换权重决策框架，使多专家在全一致时才可覆盖锚点预测，从而降低孤立误差。

**🔧 技术方法**

技术手段包括：F2LLM文本编码、HuBERT音频特征、SigLIP2视觉特征、102维行为统计表征；在15种模态组合上分别训练MLP、LightGBM随机森林和GBDT；使用验证集最小二叉交叉熵选模型、宏F1校准阈值；固定BROTHER权重的硬投票；以及RG‑MEC的逻辑门控实现。

**📊 数据集**

使用ABAW11 BAH 数据集，该数据集按参与者分离划分为训练、验证与公开测试三部分。

**📈 对比分析**

通过对比基线锚点、CALM‑AH单独模型、保守交叉与完整RG‑MEC，公开测试宏F1由0.7525提升至0.7771，准确率达到0.7981，表明RG‑MEC在保持稳定性的同时显著提升性能。

**⚠️ 局限性**

局限性在于仅在公开测试上评估，缺乏跨数据集泛化验证；RG‑MEC的决策规则依赖经验设定，可能在不同任务或数据分布下表现不稳定。

---

## 316. RIGEL: Real-time Optical Anomaly Diagnosis with Stateful In-Network Inference based on Distributed On-switch GNNs

**arXiv ID:** 2607.29306 | [PDF](https://arxiv.org/pdf/2607.29306v1)

**作者:** Zhen Wei `[一作]` (University of Science and Technology of China), Zuqing Zhu `[通讯]` (University of Science and Technology of China)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 RIGEL，一种基于 Tofino 开关的全网络内分布式图神经网络（GraphSAGE+AE）实现的实时光网络异常诊断系统。

**💡 创新点**

创新点包括：①首次实现了状态化的分布式网络内推理；②通过软件‑硬件协同设计，将高维光谱数据先 PCA+UQ+VQ 量化为索引，避免了在 IDP 上的逐维运算；③在每个开关上实现了邻居状态机（NSM）协同处理特征并完成聚合，充分利用邻域信息；④实现了端到端极低通信开销和近 1 ms 级别的推理延迟。

**🔧 技术方法**

使用技术：P4 语言编程 Tofino ASIC、GraphSAGE GNN、自动编码器（AE）、主成分分析（PCA）、均匀量化（UQ）、向量量化（VQ）、多阶段特征离散化、邻居状态机与邻域聚合、P4Runtime 控制面部署。

**📊 数据集**

数据集：真实实验室 6 节点光网络测试平台，采集约 234 k 条标注样本（8 种软失效类 + 正常），每样本 20 维（从 640 维光谱压缩得到）。还在 8/10/12/14 节点拓扑上做交叉验证。

**📈 对比分析**

方法对比：与集中式基于 GraphSAGE 的 C‑GNN 及其 UQ 版本进行对标；指标包括准确率、F1、控制面交互次数、带宽占用。RIGEL 在准确率和 F1 上均 ≥ 98%（略低于全精度集中式模型 ~99%），但将控制面交互降低 10.3×，带宽压缩 2,568×；在 CPU 上的单层推理 10 ms，TOFINO 上 800 ns，展示显著的延迟优势。

**⚠️ 局限性**

局限性：①VQ/ UQ 码本大小与位宽的折衷导致在小码本时误差增大；②仅支持 2‑层 GraphSAGE（深层易导致过平滑且硬件资源有限）；③对非常大拓扑或极端噪声仍需进一步验证；④在未见异常时需要多根报告触发重新训练，过程仍需人工或自动化支持；⑤受限于 Tofino1 ASIC 资源，扩展到更高维度或更复杂模型时需更多硬件改进。

---

## 317. Sample Efficient Hierarchical Reinforcement Learning via Best Policy Identification

**arXiv ID:** 2607.29294 | [PDF](https://arxiv.org/pdf/2607.29294v1)

**作者:** Anders Jonsson `[一作]` (Universitat Pompeu Fabra), Lorenzo Steccanella `[通讯]` (Universitat Pompeu Fabra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 HBPI-UCRL，一种并行学习高低层策略的模型基础层级强化学习算法，能够在同一时间学习 SMDP 策略与子任务策略；

**💡 创新点**

首次给出了两条足够条件，使并行层级 RL 在满足条件时可 PAC 学习，并推导了该算法的多项式样本复杂度上界，尤其在稀疏奖励目标导向情境下样本复杂度比非层级方法低 S 倍；

**🔧 技术方法**

采用基于模型的置信区间构造（Hoeffding 以及后续的贝塞尔式奖励/转移置信集）、最佳策略识别（BPI）框架以及自适应停止准则，结合子任务级与高层级的联合误差函数 L 的递推来实现并行学习；

**📊 数据集**

实验使用了无宝藏的稀疏奖励 4 房间网格世界（SMDP 示例），并通过随机初始状态等方式扩展到不同规模的房间；

**📈 对比分析**

与传统非层级 BPI-UCRL 做比较，实验显示 HBPI-UCRL 的停机时间随高层状态数呈 √S 量级，而 BPI-UCRL 接近线性增长；理论上在稀疏奖励下 HBPI-UCRL 的样本复杂度为 (SA H²H²/ε²)，比 BPI-UCRL 小 S 倍，仅多一个 H² 的乘子；

**⚠️ 局限性**

局限性包括：仅适用于子任务共享转移动力学且满足两条假设的情形，尚未验证更一般的子任务异构或无穷期望设置；实验仅覆盖离散有限状态空间，缺乏对更复杂环境的评估。

---

## 318. RTLCurator: Label-Efficient Data Curation for RTL Generation

**arXiv ID:** 2607.29283 | [PDF](https://arxiv.org/pdf/2607.29283v1)

**作者:** Siyang Cai `[一作]` (University of Chinese Academy of Sciences), Ying Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于行为感知兼容性先验的 RTL 数据集筛选框架，通过少量验证标签校准后在完整数据集上按兼容性、覆盖率和结构丰富度三重标准选取子集，以提升大语言模型的 RTL 代码生成效果。

**💡 创新点**

创新点包括：① 在对抗式学习中使用功能失败的负样本来捕捉规格与 RTL 行为的一致性；② 将该先验在目标语料上仅用极少验证样本校准成 Alignment Score；③ 设计多层次的保留预算分配策略，兼顾分数分层、表示覆盖和 RTL 结构多样性。

**🔧 技术方法**

技术手段主要包括：对 Qwen3‑Embedding‑4B 进行三阶段对比学习；使用聚类 + 预算化采样进行主动验证；基于校准的二分类器得到 Alignment Score；在子集选取中加入结构丰富度权重。

**📊 数据集**

使用了两大合成 RTL 语料库 CodeV（约 16.4 万对）和 RTLCoder（约 2.6 万对），并在 VerilogEval 与 RTLLM 评测集上进行下游实验。

**📈 对比分析**

对比方法包括 Full、Random、DS²、LESS、Align. top 以及对全量数据的 Full‑Val 验证。实验显示在 80% 保留预算下，该框架在 VerilogEval（Machine、Human）与 RTLLM（v1.0、v1.1）上均优于全量训练，提升幅度可达 5–7%（以功能通过率计），且仅验证 10% 的样本即可完成。

**⚠️ 局限性**

局限性：仍需依赖一定量的人工或专家验证；该方法专注于 RTL 级别的功能对齐，对更高级别硬件描述（如系统级验证、性能约束）尚未验证；若数据分布与训练集差异过大，兼容性先验的迁移性可能受限。

---

## 319. METIS: A Declarative Slice Orchestrator for Application-Centric 5G/6G Networks

**arXiv ID:** 2607.29282 | [PDF](https://arxiv.org/pdf/2607.29282v1)

**作者:** Arman Divband `[一作]` (EURECOM), Navid Nikaein `[通讯]` (EURECOM)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 METIS，一个基于 Kubernetes 的声明式切片编排器，能从应用级 Service Profile 自动推导 3GPP 兼容的 Slice Profile，完成 Day‑0/1/2 全生命周期管理并协调 O‑RAN 与 CN 的切片与 QoS 强制。

**💡 创新点**

创新点在于（1）引入应用‑中心的数据模型，摆脱静态模板；（2）采用层级重合调和循环（cascaded reconciliation loops）实现无状态、幂等的 NSI 管理；（3）实现 RAN 与 CN 双域联合切片与 QoS 调度，解决下行/上行 QoS 非对称问题；（4）提供完整的实验评估，包括吞吐、SLA 满足率、可扩展性与故障恢复。

**🔧 技术方法**

技术包括 Kubernetes Operator、Container‑Native CNF 部署、O‑RAN Near‑RT/Non‑RT RIC、SLA xApp、Linux Traffic Control、Prometheus 监控、Chaos Mesh 故障注入，数据模型遵循 3GPP TS 28.541/28.533 以及自定义 Service Profile 模式。

**📊 数据集**

使用自研校园活动（Campus Event）演示数据集：三类流量（安全摄像、组织者视频聊天、参与者数据传输）在真实 5G 现场环境（Ue/Ur 设备、OpenAirInterface、Open5GS）下产生，并在多区九区扩展场景下模拟 63 个 NSI。

**📈 对比分析**

与 NASP、CLiSO 等基线对比，METIS 在 3 个并发 NSI 上实现创建 ≤22.4 s、更新 ≤5.1 s、升级 ≤52.2 s、删除 ≤32.1 s；在 63 NSI 的多区部署下 CPU 占用 <0.03 核；在故障注入实验中，4 级恢复均 <19 s，显示声明式重合循环能实现低延迟自治恢复。

**⚠️ 局限性**

局限性包括：测试仅在单一 O‑RAN/CN 供应商环境下验证；缺乏自适应资源调度（如 AI/ML 优化）仅靠手工规则；对更复杂多租户、多业务类型的动态 QoS 需求支持有限；对标准化接口的实现细节仍需进一步推广。

---

## 320. Language Models Agree With Each Other, Not With Readers

**arXiv ID:** 2607.29274 | [PDF](https://arxiv.org/pdf/2607.29274v1)

**作者:** Kazuki Nakayashiki `[一作]` (Glasp Inc), Keisuke Watanabe `[通讯]` (Glasp Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对公共网页文档中自然阅读者的高亮标注数据，衡量不同语言模型在文本选择任务上的相互一致性，并与自然阅读者之间的相似度进行比较。

**💡 创新点**

创新点在于：①使用真实自然阅读者（未给任务）标注的“基准”而非实验室招募的受试者；②设计了校正过的“超额一致度”度量，剔除位置和长度带来的噪声；③对多家供应商、不同规模、不同代际和权重模式的模型进行大规模对比。

**🔧 技术方法**

主要技术包括：句子级别的相似度统计、基于深度-长度带内的无放回重采样来估计期望重叠、域聚类自举置信区间以及对模型输出做“top‑b”截断与随机抽样的处理。

**📊 数据集**

数据集由数千篇公共网页文档（最多三篇每个域）组成，配备最多60位阅读者的高亮句子索引；模型方面包括OpenAI、Anthropic、Google、Meta、Mistral、Alibaba、DeepSeek、Microsoft、IBM、NVIDIA、Z‑AI等供应商的8B–前沿规模模型。

**📈 对比分析**

比较方法为：先对模型和阅读者各自抽取相同规模的句子集合（大小为round(0.2n)），计算其交集与期望交集之差（超额一致度）；再对模型–模型、模型–阅读者以及阅读者–阅读者进行对比。结果显示：模型之间的超额一致度平均约为人类一致度的2.6倍（最高可达3.9倍），且随模型规模与更新年份显著提升；而模型与阅读者的相似度仅达到人类同侪一致度的约1.4倍，未能超过人类一致度上限。

**⚠️ 局限性**

局限性包括：①阅读者未接受任何任务或指令，可能导致“基准”不具可控性；②阅读者与模型的截断方式不一致，导致比较偏差；③缺乏对阅读者身份及其在不同文档中的重叠性信息，可能低估人类一致度；④标注平台的覆盖范围有限，不能代表所有自然阅读者；⑤模型训练数据和架构多样性未完全考察。

---

## 321. Exploring Block Anomaly Detection In HDFS Log Data Analysis

**arXiv ID:** 2607.29383 | [PDF](https://arxiv.org/pdf/2607.29383v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 322. When Model Priors Conflict with Visual Evidence: Mitigating Commonsense-Driven Hallucinations by Selective Prior Calibration

**arXiv ID:** 2607.29240 | [PDF](https://arxiv.org/pdf/2607.29240v1)

**作者:** Kesheng Chen `[一作]` (Harbin Institute of Technology), Wenjian Luo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究视觉语言模型中的常识驱动幻觉（CDH）问题，发现错误系统性地聚焦于常识优先的答案，并提出Selective Prior Calibration（SPC）方法通过先验校正、实例级调节与选择性答案修订来减轻此错误，同时保持对正常图像的正确预测。

**💡 创新点**

创新点在于：①发现并系统描述了“Directed Prior Attraction”——CDH错误趋向于模型无图像条件下优先的答案；②设计了候选级先验校正与实例级校正强度学习的组合；③引入了可选择性答案修订规则，只在校正后方案得到足够支持时才替换原答案。

**🔧 技术方法**

技术方法包括：对每个候选答案分别计算图像条件得分和无图像文本条件得分；用softplus映射的低容量线性模型根据13维特征预测实例级校正强度；对校正后的得分构造修正提议，并用阈值、候选集大小等条件决定是否接受修正。

**📊 数据集**

使用的数据集主要是自定义的CDH-Bench（包含CF–CS配对图像及多种子任务），以及外部冲突与幻觉基准：Visual CounterFact、HallusionBench、ConflictVIS、POPE/POPEv2。

**📈 对比分析**

与Focus on Vision、VCD、MFCD、PAI、NoLan、REVIS等现有方法在MC和QA任务上进行对比，评估指标为Δ_CF+0.5Δ_CS。SPC在CF准确率提升约7–12个百分点，CS准确率保持或轻微下降，修复/误伤比显著优于其它方法，整体效用最高。

**⚠️ 局限性**

局限性包括：需要固定候选集和候选概率；推理时需对每个候选重复评分，计算成本高；无法支持开放式生成；先验估计受问题形式和答案顺序影响，导致在某些情况下降低效果。

---

## 323. TacPrint: A Wearable Fingertip Tactile Sensor for Human-to-Robot Contact Reproduction

**arXiv ID:** 2607.29231 | [PDF](https://arxiv.org/pdf/2607.29231v1)

**作者:** Yongxi Liu `[一作]` (Chinese Academy of Sciences), Shuo Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一款可穿戴的指尖触觉传感器TacPrint，可从24通道电容信号中学习并输出35×26的接触深度图。

**💡 创新点**

创新点在于将低成本可穿戴式触觉传感器与实时深度学习管线相结合，利用稀疏电容测量推断密集接触信息，并在机器人重放和闭环抓取中验证其实用性。

**🔧 技术方法**

采用电容传感、三维仿真标注、LSTM时序编码器+卷积解码器深度网络，配合物理实验和仿真回归。

**📊 数据集**

使用基于模拟的接触深度标签（TacFlex/Isaac Gym/Flex）和现场人类演示的电容数据，包含40次对比物理试验和多个抓取/擦拭任务。

**📈 对比分析**

与仿真标签相比，模型在接触区域RMSE 0.223mm、重心误差1.213像素、IoU 0.829；在实测电容输入下，中心深度误差0.085mm、位置误差0.250mm；在人机重放中成功率从0%提升到91.67%，闭环抓取中密集深度反馈成功率87.5%（边缘90%），优于仅使用电容反馈的67.5%。

**⚠️ 局限性**

局限在于仅24通道输入限制了边界重建精度，预测结果受训练弹性体对象影响，且未验证全像素级真实形变，可能导致接触边界误差。

---

## 324. Studying quantization trade-offs for efficient inference deployment in machine translation

**arXiv ID:** 2607.29397 | [PDF](https://arxiv.org/pdf/2607.29397v1)

**作者:** Jim Zhao `[一作]` (University of Basel), Teryn Jones `[通讯]` (Aleph Alpha Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对两大机器翻译模型家族（EuroLLM 与 Hy-MT2）在 A100 与 H100 GPU 上的部署，系统评估了 W8A8 与 W4A8 量化方案与文档切分策略对推理延迟、吞吐量与翻译质量的综合影响。

**💡 创新点**

创新点在于（1）首次将文档级切分与低精度量化联合使用，显著提升延迟-吞吐 Pareto 前沿；（2）指出传统段落级评测无法捕捉量化后长文本翻译的质量退化，并提出基于 WMT24++ 的文档级评估方法；（3）在多模型、多硬件环境下开展闭环在线实验，揭示不同量化与切分组合的性能差异。

**🔧 技术方法**

所用技术包括 GPTQ+SmoothQuant 的后训练量化、vLLM 引擎（支持 W4A8/W8A8 内核、PagedAttention 与 Continuous Batching）、动态激活量化、文档切分与块状注意力掩码、以及 xCOMET 与 chrF++ 的质量评估。

**📊 数据集**

主要数据集为 WMT24++（文档级并行语料）用于翻译质量评估，OPUS 语料（含 1024 条样本）用于 PTQ 校准；同时使用 Seed‑X 生成目标翻译。

**📈 对比分析**

比较方法：在 BF16 基准下，分别评估 W8A8、W4A8、W4A16 的吞吐量、p99 延迟以及文档级 xCOMET/chrF++ 分数；在离线与闭环在线两种实验场景下测量，结果显示：对 9B 及以上模型，A100 上 W8A8 与 H100 上 W4A8 在吞吐量上优于 BF16；量化后 EuroLLM 在文档级翻译中质量显著下降；Hy‑MT2 量化保持与 BF16 近似。

**⚠️ 局限性**

局限性包括：仅使用 GPTQ+SmoothQuant，未尝试更高级的 PTQ 或 QAT；动态激活量化带来额外推理开销；评估仅覆盖四个语言对和少量文档；未实现自适应切分策略，切分长度仍固定；量化对不同上下文长度的影响机制尚未深入阐明。

---

## 325. PTP: Previous-Token Prediction based LLM Inversion for Near-Exact Prompt Reconstruction

**arXiv ID:** 2607.29378 | [PDF](https://arxiv.org/pdf/2607.29378v1)

**作者:** Pirzada Suhail `[一作]` (IIT Bombay), Amit Sethi `[通讯]` (IIT Bombay)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建一个从零开始的逆向语言模型，通过预测前一个 token 的方式在完全黑盒环境下重建 Prompt。

**💡 创新点**

利用仅依赖目标 LLM 合成数据的无监督逆向训练策略，采用 previous-token prediction 逆向建模，突破了传统依赖 logits 或预训练模型的限制。

**🔧 技术方法**

逆向语言模型训练、序列反转、对合成数据的自监督学习、少量真实 Prompt 微调、采样式 Prompt 重建。

**📊 数据集**

Qwen3‑0.6B、LLaMA‑2 7B、ShareGPT、Instruction‑2M、GPT‑4o 产生的响应作为实验数据。

**📈 对比分析**

与 O2P、RPE 等黑盒基准对比，PTP 在 Exact Match、Token F1 等 token‑level 指标上明显优于 O2P，且在跨模型、跨数据集时仍保持较高的语义相似度。

**⚠️ 局限性**

对 tokenizer、词表与模型结构的敏感导致 token‑level 失真，合成采样成本较高，并且需要少量微调以确保 Prompt 格式正确。

---

## 326. Cross-Resolution Semantic Learning for Graph Domain Adaptation

**arXiv ID:** 2607.29365 | [PDF](https://arxiv.org/pdf/2607.29365v1)

**作者:** Yingxu Wang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Shangsong Liang `[通讯]` (Macao Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种跨分辨率语义学习框架，在图域适应中学习源到目标的分辨率对应关系，并通过后向嫁接实现实例级自适应。

**💡 创新点**

创新点在于：①利用共享GIN编码器与可学习分辨率嵌入构建多分辨率表示银行；②通过交叉分辨率原型传输实现软分辨率对应；③用后向嫁接将全局对应映射为后验加权的实例级调整。

**🔧 技术方法**

使用技术包括：共享GIN编码器、可学习分辨率嵌入、交叉分辨率原型传输 (CRPT)、后向嫁接 (CRTG)、KL一致性损失以及跨域原型匹配损失。

**📊 数据集**

使用的数据集：Mutagenicity、PROTEINS、NCI1、ogbg-molhiv、DD、BZR、BZR_MD、COX2、COX2_MD 等图分类基准，涵盖结构位移与特征位移两类域移。

**📈 对比分析**

与多种基线（WL 树核、PathNN、GCN、GIN、GMT、CIN、DEAL、SGDA、A2GNN、StruRW、PA-BOTH、GAA、TDSS 等）进行对比，实验显示在大多数结构与特征域移任务上均显著优于现有方法。

**⚠️ 局限性**

局限性在于：仅处理离散分辨率集合；对连续或自适应分辨率空间的支持有限；在多源/多目标设置下的扩展尚未深入探讨。

---

## 327. MolGVR: A Chemistry-Grounded Framework for Text-to-Molecule Generation

**arXiv ID:** 2607.29479 | [PDF](https://arxiv.org/pdf/2607.29479v1)

**作者:** Qian Tan `[一作]` (University of Science and Technology of China), Yuqiang Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了 MolGVR 框架，将文本到分子生成拆分为生成、验证和细化三步，使用 LLM 提取化学约束并用 RDKit 执行验证，随后基于失败信息细化分子。

**💡 创新点**

创新点在于把化学验证与反馈细化引入生成流程，形成生成—验证—细化闭环，显著提升了分子与描述的结构一致性。

**🔧 技术方法**

核心技术包括：内部使用 Intern‑S1‑mini 生成器并通过 MSR+RL 训练；GPT‑5.2 提取 JSON 约束规则；RDKit 进行确定性子结构、元素计数等检查；Refiner 通过两阶段 SFT 修正失败分子。

**📊 数据集**

使用的公开数据集为 ChEBI‑20 与 PCDes，分别用于训练与评估。

**📈 对比分析**

与 T5、MolT5、ChemDFM、GPT‑4o 等基线对比，MolGVR 在两大基准上均取得最优 Match 分数（0.582/0.500），并在多种指纹相似度指标上领先，验证了验证与细化的有效性。

**⚠️ 局限性**

局限性包括：对复杂结构错误（如核心骨架、盐/电荷状态）的修复率仍不高；模型仍依赖 LLM 的约束抽取质量；以及在某些结构相似度和分布级别指标（FCD）上提升有限。

---

## 328. Bending the Curve: Operational Cyber Epidemiology for Ransomware

**arXiv ID:** 2607.29444 | [PDF](https://arxiv.org/pdf/2607.29444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 329. Lightweight Neural Networks for Affordance Segmentation: Enhancement of the Decoder Module

**arXiv ID:** 2607.29473 | [PDF](https://arxiv.org/pdf/2607.29473v1)

**作者:** Simone Lugani `[一作]` (University of Genoa), Paolo Gastaldo `[通讯]` (University of Genoa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

针对可穿戴机器人上的视觉可行性分割，本文分析了分割头（Segmentation Head）在轻量化网络中的作用，并提出了多种轻量化分割头设计与多任务学习框架。

**💡 创新点**

创新点在于：①聚焦分割头设计而非仅关注骨干网络，②通过多任务（对象分割+可行性分割）和低级特征连接，提升小模型的泛化能力；③在有限的 FLOPs 约束下，系统性比较三种分割头结构，找到最优权衡。

**🔧 技术方法**

采用 MobileNetV3 作为骨干，尝试深度可分离卷积 + 最近邻上采样（U）、深度可分离卷积 + 转置卷积（T）以及标准卷积 + 最近邻上采样（B）；使用多任务损失 ℒ = ℒ_seg + ℒ_aff；对特征图的连接方式进行实验。

**📊 数据集**

实验数据集：UMD（28.843 张 RGB‑D 图，7 类对象）与 IIT（8,835 张多尺度图），并对 UMD 进行背景替换和标准增广。

**📈 对比分析**

通过加权像素级准确率和总分（TOT）对比 baseline，模型在三套测试集上均达 90%+ 的精度，参数量从 1.3M~3.6M 下降，且 FLOPs 维持在 700/800 M 的约束内，显著优于 5.9M 参数的 baseline。

**⚠️ 局限性**

局限性：仅在两套相对小型、同质化的数据集上验证，未测试更大分辨率或更复杂场景；连接策略仍需人工设计，缺乏自动化搜索；模型在实时部署时的实际延迟与能耗尚未充分评估。

---

## 330. CARA: Exact Local Repair with Fresh One-Action Certification for Cloud Consolidation

**arXiv ID:** 2607.29465 | [PDF](https://arxiv.org/pdf/2607.29465v1)

**作者:** Xiyang Zhang `[一作]` (Zhongguancun Academy), Hongzhi Wang `[通讯]` (Harbin Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Certificate‑Aligned Recomposition (CARA) 方案，针对云主机分配中的后置修复步骤，先通过精确搜索获取 top‑P 不同的本地修复候选，再用一次性新证据对选定方案进行检验，并在检验失败时退回已承诺的基准方案；

**💡 创新点**

创新点包括：①将提议精度、决策有效性与终端效益三方面拆分，形成可审计的管线；②利用 packing‑specific admissible bound 在已知两主机邻域内精确恢复 top‑P 方案；③采用一次性新证据（paired certification）在不依赖搜索复杂度的前提下控制四个可靠性指标的假提升率 ≤0.05；

**🔧 技术方法**

技术手段涵盖：精确的两主机邻域搜索与可数列枚举；基于多维置信金字塔的 lexicographic 排序；定点多精度算术与外推最优边界；McNemar 与固定投注财富检验；统计拆分（held‑out 与 fresh certification）确保信息泄漏控制；

**📊 数据集**

数据集为 128 个独立合成云环境集（每个环境 4 个重复上下文），每个单元格共 8192 对 certification 样本；此外还使用了来自三家供应商的公开轨迹作为描述性案例；

**📈 对比分析**

与 Baseline（Global96、ProjBlind‑2H、verified evacuation、ILS、ALNS）比较，CARA 在所有 5 个终端指标上均显著优于基准：J 指标下降 3.57 pp，连续负载指标下降 21–28%；在 512 个单元格中释放 445 次，统计检验均通过；两种提议顺序（CARA vs ProjBlind‑2H）在效果上无显著差异；

**⚠️ 局限性**

局限性：仅在预定义的两主机邻域内提供精确上限；不保证全局 packing 最优；未控制原始负载严重度、分布漂移或多次部署；数据仅为合成且主机同质，缺乏到达、迁移成本和干扰等现实因素；

---

## 331. Mechanical Modeling of Braided Neurovascular Flow Diverters using a Beam-to-Beam and Beam-to-Surface Contact Formulation

**arXiv ID:** 2607.29446 | [PDF](https://arxiv.org/pdf/2607.29446v1)

**作者:** Martin Frank `[一作]` (German Federal Armed Forces Munich University), Alexander Popp `[通讯]` (German Federal Armed Forces Munich University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一个基于有限元的数值模拟框架，用于对交织式神经血管流分流器的结构行为进行建模与分析。

**💡 创新点**

其创新点在于将几何精确的Simo‑Reissner梁理论与点接触和表面接触相结合，并提供可调节的参数化几何描述，实现了不同织缝角、线数与尺寸的自动生成。

**🔧 技术方法**

主要技术包括Simo‑Reissner梁模型、罚函数接触（梁‑梁与梁‑表面）、位置耦合约束以及局部坐标变换，全部在开源多物理场框架中实现。

**📊 数据集**

实验验证使用了文献中已有的三组标准实验（拉伸、压缩和分段压缩）数据，未使用专门的数据集，而是以实验曲线和理论预测为基准。

**📈 对比分析**

与实验和理论结果对比显示，该模型能够较好地重现长度‑直径关系、轴向力响应和压强‑直径曲线，性能在趋势上与参考值一致，但在较大变形时轴向力略有过度预测。

**⚠️ 局限性**

局限性包括采用线性弹性材料模型（忽略尼铁的相变非线性）、缺乏摩擦接触、仅考虑直线几何且未对患者特定曲形血管进行验证。

---

## 332. DreamQAS: Learning a Decision-Useful World Model for VQE-Efficient Quantum Architecture Search

**arXiv ID:** 2607.29491 | [PDF](https://arxiv.org/pdf/2607.29491v1)

**作者:** Jiayang Niu `[一作]` (Rmit University), Yongli Ren `[通讯]` (Rmit University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DreamQAS 框架，通过世界模型学习 VQE 反馈并在多步想象中进行策略优化，从而在分子 QAS 中显著减少真实 VQE 调用次数。

**💡 创新点**

创新点在于将已知的电路转移保持精确、仅学习昂贵后续反馈、使用 oracle‑free 目标、以及将模型不确定性与安全阈值相结合实现可靠的多步想象。

**🔧 技术方法**

使用了递归随机化先验集成网络、前沿相对目标函数、优势学习（A2C/REINFORCE）、基于风险的截断与验证、以及多步想象（roll‑out）等技术。

**📊 数据集**

采用 HamQASBench 中的五个分子任务（LiH‑4q、BeH₂‑6q、LiH‑6q、BeH₂‑8q、BeH₂‑10q）作为实验数据集。

**📈 对比分析**

与无想象、无 DAgger、CRLQAS、HyRLQAS 等对照方法比较，DreamQAS 在 15,000 轮 RL 预算下，四项任务均取得最低或次低均值误差；在最严格误差目标下，真实 VQE 调用次数比对照平均减少 1.6–2.0 倍，BeH₂‑8q 低至 10.6 倍。

**⚠️ 局限性**

局限性包括仅在纯 state‑vector 仿真下验证，未评估硬件噪声与排队延迟；某些任务已达搜索底层，难以进一步提升；集成不确定性虽能排序风险，但不保证完整的概率校准。

---

## 333. Evidence-Type Competition: When Can Interventional Data Teach Language Models Causal Direction?

**arXiv ID:** 2607.29484 | [PDF](https://arxiv.org/pdf/2607.29484v1)

**作者:** Xining Xun `[一作]` `[通讯]` (Tsingjiao Information Science Co., Ltd.), Xining Xun (Tsingjiao Information Science Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在完全可控的合成因果世界中，系统性检验了交互式数据（interventional data）是否能提升语言模型的因果推理能力，并发现交互式数据在训练中的比例并不能决定模型的因果方向性，真正决定因果推断的是推理时上下文中的证据类型。

**💡 创新点**

①提出“量-方向双重性”假说：模型从交互式数据学习到因果效应的大小，却从观测上下文复制因果方向。②首次在Simpson悖论环境中演示观测先验可以在推理时抑制交互式证据，且该抑制是可通过上下文内容调节的渐进机制。③揭示交互式证据的学习能力存在于模型权重，而抑制开关则被上下文激活层中的观测记录控制。

**🔧 技术方法**

采用GPT‑style decoder（25M/1B参数）与自定义的交互式/观测式样本混合采样；使用基于 probe 的因果评估协议；实施激活补丁（activation patching）定位抑制机制；通过多种推理时干预（E1、E5）验证抑制来源；利用统计显著性检验（McNemar、Wilcoxon、二项检验）评估实验差异。

**📊 数据集**

所有实验数据均来自自定义的结构因果模型（SCM）生成器，构造了50个Simpson悖论世界（观测相关性与因果效应符号相反），并在不同的交互式比例（α=0..1）、网络规模（25M、1B）和随机种子下进行训练与评估；此外还在CLadder基准上做外部审计。

**📈 对比分析**

比较方法：在相同的世界集上对四种上下文证据组合（纯观测、混合、纯 probe、混合+probe）进行对比，使用 hit/flat/reversed 三分类评估，并通过信号-噪声阈值（p_err≈0.09）区分真正的因果错误。性能结果显示：纯 probe 上下文在50个世界中 41/50 取得正确因果方向（平均斜率 +0.418），混合上下文仅 18/50 正确，纯观测仅 7/50，表明抑制显著且可调节。

**⚠️ 局限性**

局限性：①仅使用单一合成 SCM 家族，缺乏真实语言数据验证；②模型规模增大后绝对因果收益收缩，抑制仅以错误率形式保留；③在 OOD 环境下的正向先验仍存在，无法完全去除；④抑制对不同规模模型的世界稳定性不一致；⑤剂量效应在更大样本量下未能显著复现；⑥长上下文导致解析率下降，限制了实验可扩展性。

---

## 334. Know It, Act on It: Investigating Memory Utilization in LLM Personalization

**arXiv ID:** 2607.29433 | [PDF](https://arxiv.org/pdf/2607.29433v1)

**作者:** Zhaoxin Feng `[一作]` (Hong Kong Polytechnic University), Emmanuele Chersoni `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了16种LLM记忆系统在Recall（Know）和Act两种测试中的表现，并引入了配对Know–Act评测框架，量化了记忆与行为之间的差距。

**💡 创新点**

创新点在于：①设计了可同时检验记忆存取与行为利用的Know–Act双测范式；②引入三层表达强度梯度（显式、间接、推理）以探究偏好表达对记忆与利用的影响；③采用三层故障归因分析（检索、理解、应用）定位瓶颈。

**🔧 技术方法**

使用了多种记忆架构（Mem0、RAG、结构化RAG、Agentic Memory等）以及统一的GPT‑4o‑mini生成基底；检索技术包括文本嵌入、知识图谱；评测使用GPT‑5作为判别者。

**📊 数据集**

基于PersonaMem‑v2构建的50个角色，每人20条目标偏好，共1000条偏好；为每条偏好生成三种表达级别（显式、间接、推理）的对话片段；并为每条偏好生成对应的Know/Act测试。

**📈 对比分析**

对每个系统，在相同的对话序列后执行Know和Act测试，统计Recall率、Act率和利用率；结果显示Know与Act存在显著差距，最佳利用率约65%（Claude 4.6 Sonnet），但健康与情感偏好利用率最低；记忆架构显著提升利用率（如Mem0将利用率从16.3%提升至54.6%）。

**⚠️ 局限性**

限制包括：①评测依赖LLM判别器，可能带来偏见；②使用合成Persona数据，缺乏真实用户多样性与噪声；③Benchmark简化检索任务，真实环境下Know–Act差距可能更大。

---

## 335. The Tragedy of the Cognitive Commons: How AI Could Disrupt the Regeneration of Professional Expertise

**arXiv ID:** 2607.29380 | [PDF](https://arxiv.org/pdf/2607.29380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 336. Explore Beyond the Boundary Using Entropic Information

**arXiv ID:** 2607.29419 | [PDF](https://arxiv.org/pdf/2607.29419v1)

**作者:** Bumgeun Park `[一作]` (Korea Advanced Institute of Science and Technology), Donghwan Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

提出一种基于熵信息的探索方法ENTINEX，通过给状态分布边界附近的状态提供内在奖励，提升稀疏与延迟奖励环境下的探索效率。

**💡 创新点**

创新点在于使用动作概率分布的负熵识别状态分布边界，而非传统的连续状态差分估计。

**🔧 技术方法**

采用随机编码器估计状态新奇度、动力学模型预测下一状态、Boltzmann分布生成动作概率分布，并在SAC/AC框架下实现离线强化学习。

**📊 数据集**

在修改后的OpenAI Gym MuJoCo连续控制环境（如SparseWalker2d、SparseHalfCheetah等稀疏奖励与延迟奖励设置）上进行实验。

**📈 对比分析**

与ICM、RND、RE3、NovelD等基线对比，ENTINEX在稀疏和延迟奖励任务上平均回报显著更高，即使在相同新奇度函数下仍优于NovelD。

**⚠️ 局限性**

存在训练不稳定、计算开销增加、需要维护多模型等限制，且理论分析尚未完全成熟。

---

## 337. OSEF: One-Step Evidence Fusion for Cross-Video Scene Procedure Planning

**arXiv ID:** 2607.29401 | [PDF](https://arxiv.org/pdf/2607.29401v1)

**作者:** Zhentong Ye `[一作]`, Bin Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了跨视频场景程序规划（CVSPP）基准，提出了“一步证据融合”（OSEF）模型，在给定起止状态的答案被剥离的查询与候选视频集合时，实现了检索、定位和动作序列预测的统一流程。

**💡 创新点**

创新点包括：① 统一评估检索、定位和规划三个子任务并引入“同任务证据歧义”与“检索-规划误差传播”两大挑战；② 采用查询条件化的证据格子和全局 token‑global 适配器，避免早期硬选择并让规划器获取所有候选信息；③ 在多个源数据上设置严格泄漏审计与负样本角色，提供更可信的评测。

**🔧 技术方法**

使用 BERT/DeBERTa 文本编码器、冻结的 S3D 视频特征、Transformer 规划器、学习的 2T 端点查询以及联合损失（检索、定位、规划、跨度约束）。

**📊 数据集**

共使用11个公开教学视频数据集（NIV、COIN、CrossTask、YouCook2、TACoS、HiREST、Ego4D Goal‑Step、ActivityNet Captions、Charades‑Ego、GUIDE、EgoLearn），共形成14个源‑时段单元。

**📈 对比分析**

与九种迁移后规划器（ViterbiPlanNet、MTID、SCHEMA、P3IV、KEPP、PDPP、ActionDiffusion、PlanLLM、Skip‑Plan）进行公平比较。OSEF 在所有六个可排名的原生单元中均取得第一，平均提升 2.9–10.7 分；在转换源上与多数序列基准持平；相对传统硬检索+规划基线，接口改进带来最大收益。

**⚠️ 局限性**

局限性包括：使用冻结的视觉特征与固定模板查询，难以验证模型在更通用场景的鲁棒性；验证集选择的权重可能导致过拟合；证据格子规模随候选数增大；同任务检索仍是瓶颈；FULL‑SR 成功可能不保证窗口准确。

---

## 338. On fair and realistic performance evaluations for graph-based lateral movement detectors

**arXiv ID:** 2607.29390 | [PDF](https://arxiv.org/pdf/2607.29390v1)

**作者:** Corentin Larroche `[一作]` `[通讯]` (French National Cybersecurity Agency), Corentin Larroche (French National Cybersecurity Agency)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对LANL和OpTC两大横向移动检测基准数据集的预处理与标注方法进行了系统梳理与统一，提出了可复现的标准化流程，并在此基础上重新评估了三种主流图基检测器（Pikachu、Euler、Argus），验证了原有评估方法对公平性与真实性的影响。

**💡 创新点**

创新点在于揭示预处理和标注差异对评估结果的显著偏差，并提供了具体、可公开复现的预处理/标注指南；通过重新实验证明原报告的性能指标普遍被高估，并显示不同方法在统一设置下的真实排名。

**🔧 技术方法**

采用图神经网络（GCN‑GRU、MPNN‑GRU）以及传统Graph‑based anomaly detection框架；使用Python实现预处理/标注脚本，利用AUC、AP等统计指标对模型进行多次重复实验以评估稳定性。

**📊 数据集**

使用的公开数据集为LANL的“Comprehensive, multi-source cyber-security events”与DARPA的“Operationally transparent cyber”（OpTC）两大网络/系统日志集合。

**📈 对比分析**

方法对三种检测器进行十次独立训练/测试实验，计算AUC与AP；结果显示在统一预处理/标注下所有方法性能均低于原论文报告，尤其是Pikachu的AUC显著下降，Argus与Euler的相对优势也随之减弱，体现评估的公平性与真实性问题。

**⚠️ 局限性**

局限性包括：仍依赖人工标注，耗时且可能遗漏细粒度事件；仅评估三种方法，未探索更先进模型；OpTC日志不完整导致标注仍有不确定性；结果仍受训练/测试分割策略的影响，需进一步验证在更多真实环境下的稳健性。

---

## 339. VFAD: Variational Semantic Prompting Meets Frequency-Adaptive Representation Learning for Zero-Shot Anomaly Detection

**arXiv ID:** 2607.29370 | [PDF](https://arxiv.org/pdf/2607.29370v1)

**作者:** Peng Chen `[一作]` (Sun Yat-sen University), Chao Huang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出VFAD框架，针对零样本异常检测融合变分语义提示和频率自适应表示学习。

**💡 创新点**

创新点在于：1）通过变分语义提示提取局部语义并通过信息瓶颈正则化；2）采用小波频率分解与专家路由实现细粒度纹理表征。

**🔧 技术方法**

技术包括CLIP ViT‑L/14‑336视觉-文本编码器、离散小波变换、变分信息瓶颈、Mixture‑of‑Experts路由。

**📊 数据集**

使用13个工业与医学基准：MVTec‑AD、VisA、KSDD2、DAGM、DTD‑Synthetic、HeadCT、BrainMRI、Br35H、ISIC、ColonDB、ClinicDB、Endo、Kvasir。

**📈 对比分析**

与WinCLIP、AnomalyCLIP、AdaCLIP、AA‑CLIP、Bayes‑PFL、MoECLIP等六大前沿方法对比，工业域图像级AUROC提升0.7%（94.2%），像素级AUROC提升0.5%（97.3%）；医学域图像级AUROC达97.6%，像素级AUROC 89.7%，显示显著性能优势。

**⚠️ 局限性**

局限性在于模型规模大（ViT‑L/14‑336+DWT+MoE）导致推理延迟高；对辅助训练数据的依赖可能影响跨域泛化；缺乏对实时部署与能耗的评估。

---

## 340. SatEdit: Mask-Conditioned Image Editing via VLM-Guided Segment Annotation

**arXiv ID:** 2607.29367 | [PDF](https://arxiv.org/pdf/2607.29367v1)

**作者:** Muhammad Talha `[一作]`, Muhammad Ahmed Amer `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于掩码的卫星图像编辑框架 SatEdit，利用无标注图像自动生成编辑对，并通过 LoRA 微调实现局部对象增删。

**💡 创新点**

创新点在于：①将 SAM2 生成的掩码、VLM 自动标注、人工校验与填充相结合，构建了无需人工编辑对的监督数据；②通过 mask‑conditioned 训练，使模型仅在用户指定区域内完成语义编辑，保持周围景观不变。

**🔧 技术方法**

核心技术包括 SAM2（对象掩码生成）、Vision‑Language Model（VLM）用于段级标签、LaMA 及其他填充模型生成编辑对、以及 LoRA 微调高分辨率图像编辑 backbone。

**📊 数据集**

使用了 SODA‑A 衬里衬的卫星图像集（1014 张图像，852 条已验证的对象标注，覆盖 91 个类别）作为训练与评测数据。

**📈 对比分析**

与 GPT‑Image‑2、Nano Banana 2、Qwen‑Image‑Edit 进行比较；在掩码增删任务上，SatEdit 在 CLIP 分数、CLIP Δ 以及编辑局部化指标上均居首，尤其在聚合指标下 CLIP = 0.6322、CLIP Δ = 0.0726，显示出更强的语义对齐与较低的泄露。

**⚠️ 局限性**

局限性包括：生成的数据量有限，导致类别不平衡；SAM2 生成的掩码和 VLM 标注可能存在误差，影响训练质量；对遮罩外编辑的控制仍受限于基模型与 LoRA 规模的平衡；且实验仅在少量公开数据上验证，缺乏更大规模、跨传感器的评测。

---

## 341. Bridging the Question-Answer Gap in Retrieval-Augmented Generation: Hypothetical Prompt Embeddings

**arXiv ID:** 2607.29402 | [PDF](https://arxiv.org/pdf/2607.29402v1)

**作者:** Domen Vake `[一作]` (University of Primorska), Aleksandar Tošić `[通讯]` (University of Primorska)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Hypothetical Prompt Embeddings（HyPE）框架，改进检索方式，将查询与文档的匹配从问答（query‑document）转换为问问（query‑query）匹配，并通过离线生成多种假设性问题来预计算并嵌入索引，消除了查询时对LLM的调用，从而提升了检索效率与质量。

**💡 创新点**

创新点在于：①把生成假设性内容的工作从实时查询阶段迁移到索引阶段；②利用预生成的多种问题向量使检索过程变为问问匹配；③在不增加查询时延的前提下显著提升检索上下文精度和召回率。

**🔧 技术方法**

技术手段包括：dense retrieval（bge-m3 作为嵌入模型）、LLM 生成假设性问题（Mistral‑NeMo）、近似最近邻（ANN）检索、RAGChecker 评估框架、以及对传统 RAG、HyDE 的对比实验。

**📊 数据集**

使用了六个数据集：MS MARCO、RAGBench、Ragas‑WikiQA、RAG‑dataset‑12000、MultiHopRAG 和 Single‑Topic RAG，用于覆盖常规检索、多跳推理和狭义领域等多种场景。

**📈 对比分析**

通过与 Naive RAG 和 HyDE 在六个数据集上进行对比，评估指标包括上下文精度、命题召回、生成的可信度、幻觉率等，实验显示 HyPE 在平均精度提升约 20%（相对 Naive RAG），召回提升约 16%，整体 F1 提升约 15%，在 Single‑Topic、RAG‑dataset‑12000 等难度较高的数据集上表现尤为突出。

**⚠️ 局限性**

局限性包括：离线索引阶段需要对每个文档块进行一次 LLM 调用，规模较大时成本显著；未对生成的假设性问题进行质量过滤，可能导致索引冗余与噪声；在词重叠程度高、文档短小的任务（如 MS MARCO）提升有限；当前仅在单语言环境下验证，缺乏多语言或更大上下文窗口的实验。

---

## 342. OnlineCache: Learning Dynamic Caching Policies with Error Correction for Efficient Diffusion Inference

**arXiv ID:** 2607.29398 | [PDF](https://arxiv.org/pdf/2607.29398v1)

**作者:** Zhikang Xie `[一作]` (Fudan University), Cheng Jin `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

为扩散模型的推理提供一种动态缓存框架 OnlineCache，依据每个步骤和输入样本的难度自适应决定是否使用缓存并修正误差。

**💡 创新点**

创新点在于将缓存决策视为序列决策问题，使用轻量级策略网络和双层优化联合训练误差修正器，实现实例感知的速度-质量权衡。

**🔧 技术方法**

采用强化学习（策略梯度）训练策略网络，配合轻量级 MLP 误差修正器，利用状态统计特征、时间嵌入，构建奖励函数，并通过双层优化实现策略-修正器协同。

**📊 数据集**

在图像上使用 FLUX.1-dev、MSCOCO、Parti‑Prompts 以及 DiT‑XL/2、ImageNet-256；在视频上使用 CogVideoX-2b；还在多分辨率（512x512、1024x1024）等场景进行验证。

**📈 对比分析**

与 ERTACache、TeaCache、FastCache、L2C 等现有缓存加速方法对比，OnlineCache 在 FLUX、DiT、CogVideoX 等模型上实现约 2~3 倍加速，同时保持甚至提升 LPIPS、SSIM、FID 等质量指标，显著优于对比基线。

**⚠️ 局限性**

需要为每个模型训练单独的策略网络，训练成本和参数量不小；在极端压缩比例下仍会出现质量下降；对模型结构差异较大时的跨模型迁移效果有限。

---

## 343. Simulation Code Generation for Fluid Systems using Large Language Models: Benchmarking Models and Prompting Strategies

**arXiv ID:** 2607.29389 | [PDF](https://arxiv.org/pdf/2607.29389v1)

**作者:** Jan Marius Stürmer `[一作]` (German Aerospace Centre (DLR)), Andreas Weinmann `[通讯]` (Technische Hochschule Würzburg-Schweinfurt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了十种最先进的大型语言模型在将中性图描述自动转换为两类流体系统仿真代码（Python WNTR 与 Modelica）过程的可行性与质量。

**💡 创新点**

通过对比五种提示策略，发现“精选提示”比完整代码或文档更有效，提出 Agent+验证机制能提升语法正确性但无法完全解决物理精度问题。

**🔧 技术方法**

使用了大语言模型（如 Qwen3-Coder-Next、Kimi-K2.5 等）、Prompt Engineering、RAG、CodeBERT 相似度、Agentic OpenCode、代码解析与仿真验证等技术。

**📊 数据集**

构建了基于 WNTR 的三条水网与 Modelica 的四个工业网络（EPANET、HAI-CPPS），将其转换为自定义 JSON 图结构并手工提供对应的真值 wrapper 作为基准。

**📈 对比分析**

采用 pass@k、Call/Argument F1、CodeBERT 相似度、仿真 NMAE 等多维度指标进行评估；Kimi-K2.5 在 WNTR 达到 100% pass@1/3 与高 F1，Modelica 的最高 pass@1 为 87%，但仿真精度始终未达 5% 误差阈值。

**⚠️ 局限性**

LLM 缺乏对物理规律的推理能力，导致生成的代码虽然语法正确却常导致仿真结果不一致；完整代码提示不如精选提示有效，Agent 只能修复表面错误，无法纠正单元转换或模型选择错误。

---

## 344. Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration

**arXiv ID:** 2607.29482 | [PDF](https://arxiv.org/pdf/2607.29482v1)

**作者:** Dylan Miller `[一作]` (University of Alberta), Martin Jagersand `[通讯]` (University of Alberta)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 Temporal Policy，一种利用机器人历史状态作为生成起点的时间耦合生成式策略，用于学习演示中的动作序列。

**💡 创新点**

创新点在于将动作生成视为点到分布的传输，摒弃独立高斯先验，显著降低传输距离和向量场曲率；同时通过解析的分数恢复实现单模型可切换的确定性 ODE 与随机性 SDE 采样。

**🔧 技术方法**

使用了随机插值（Stochastic Interpolants）框架、ODE/SDE 求解器、1D U‑Net+FiLM 结构、空间 Softmax 编码、动态噪声调度与梯度回归等技术。

**📊 数据集**

在 Robomimic 仿真基准（Lift, Can, Square, Transport, Tool Hang）以及真实 Barrett WAM 7‑DoF 双臂的演示数据（含 RGB + 关节信息）上进行训练与评估。

**📈 对比分析**

通过与 Diffusion Policy 与 Conditional Flow Matching (CFM) 的基准对比，保持相同或更小的参数量；Temporal Policy 仅需 10 次函数评估，推理时间 19.1 ms，成功率与基线相当或更优，尤其在多模态数据集上表现突出。

**⚠️ 局限性**

局限性在于对初始化状态的质量高度敏感，噪声过大会偏移生成轨迹；当前仅使用状态同构表示，未充分利用速度或力/扭矩信息；对完整多模态行为的覆盖能力仍需进一步验证。

---

## 345. System-Wide Termination in Distributed Betweenness Centrality Computation

**arXiv ID:** 2607.29474 | [PDF](https://arxiv.org/pdf/2607.29474v1)

**作者:** Siamak Abdi `[一作]` (Free University of Bozen-Bolzano), Giuseppe Di Fatta `[通讯]` (Free University of Bozen-Bolzano)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在分布式 betweenness centrality 计算中加入了轻量级的全局终止检测层，允许节点在系统全局收敛时安全停止；

**💡 创新点**

创新点在于提出一种基于 epidemic 推送-拉取机制的全局终止检测算法，并在距离向量兼容的 Betweenness 计算框架上实现；

**🔧 技术方法**

采用了基于 Crescenzi 等人距离向量（Bellman‑Ford）兼容的 Betweenness 计算算法、推送‑拉取消息聚合的终止检测层以及 Python 事件驱动模拟器；

**📊 数据集**

实验使用了 Erdős‑Rényi、Geometric 两类合成网络，以及真实 Email（1133 节点）和 Road（2000 节点）数据集；

**📈 对比分析**

通过将局部停止方案与全局终止检测方案在全局相对 ℓ₂ 误差上比较，发现全局终止检测能保持零误差，而局部停止产生 0.05‑0.6 的误差；在 overlay 通信模型下收敛比物理邻居模型更快，局部停止提前 5‑15 个阶段；

**⚠️ 局限性**

限制在于仅在可靠消息、已知网络规模 N 的静态网络假设下有效；未考虑节点离线、消息丢失、网络动态变化，也仅在模拟器验证，未在真实分布式框架中测试。

---

## 346. Automated Straight-line Sewing of Stretchable Fabrics with Different Lengths

**arXiv ID:** 2607.29464 | [PDF](https://arxiv.org/pdf/2607.29464v1)

**作者:** Bingchen Jin `[一作]` (University of Hong Kong), Kazuhiro Kosuge `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套名为 DLRoSS 的机器人缝纫系统，能够自动拉伸较短的可伸缩面料与较长面料，沿直线缝合两层面料。

**💡 创新点**

创新点包括：① 采用双活跃滚轮与被动压轮的滚轮式末端执行器，实现对两层面料独立张力与伸长控制；② 设计监控滚轮实现实时织物送料速度测量并通过扭矩控制避免滑移；③ 通过四个阶段的操作流程和比例控制，解决了滚轮式执行器在直线拉伸缝合中常见的面料滑脱问题。

**🔧 技术方法**

使用技术主要包括：6自由度机械臂 + ATI 关节力/扭矩传感器；Yaskawa 伺服电机驱动滚轮与压轮；气吸附槽与硅胶滚筒提高抓取力；监控滚轮配合编码器和扭矩控制实现速度反馈；基于比例控制的面料送料调节；EtherCAT 以太网实时通信。

**📊 数据集**

实验中未使用公开数据集，而是采用多种不同厚度、柔韧性与弹性模量的三种面料（面料 A、B、C），并在多种长度与比例（γ 0.75–0.9）下进行实验，总计 115 次试验。

**📈 对比分析**

性能评估方法：与压轮不压（扩展）和压轮压紧两种情形比较，测量缝合对齐误差；结果显示：压轮压紧时平均误差 < 3 mm，最大误差 < 6 mm，均在日常衬衫尺寸公差（+8 mm/‑5 mm）内；相较于未压轮方案，压轮显著降低误差，尤其在较大拉伸比例时。

**⚠️ 局限性**

局限性：仅能完成直线拉伸缝合，无法处理曲线或复杂边缘缝合；面料拉伸难度受比例限制，极端比例（<0.75）可能导致破损；系统仍需人工预先对齐面料端部，缺乏全自动定位与视觉引导；未来需加入视觉轨迹规划以实现曲线缝合。

---

## 347. AgenticRepair: Multi-Faceted Program Context Engineering for Agentic Vulnerability Repair

**arXiv ID:** 2607.29422 | [PDF](https://arxiv.org/pdf/2607.29422v1)

**作者:** Michael Fu `[一作]` (University of Melbourne), Kla Tantithamthavorn `[通讯]` (Monash University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于多代理的自动漏洞修复框架，利用代码结构、运行时执行和提交历史三维上下文来指导补丁生成。

**💡 创新点**

创新点在于将安全工程师常用的多维程序上下文显式化，并通过专用子代理并行收集、合成，形成持久记忆供修复代理使用。

**🔧 技术方法**

技术主要包括LLM驱动的多子代理（ReAct模式）、结构化代码分析（CodeQL）、动态沙箱执行（GDB/Valgrind）、版本历史挖掘（PyDriller）以及持续迭代的补丁合成与验证。

**📊 数据集**

使用SEC‑Bench基准，共300个真实C/C++漏洞实例，包含 sanitizer 报告、手写描述和对应补丁。

**📈 对比分析**

与OpenHands、SWE‑Agent、Aider及Smolagents等基线对比，系统在SEC‑Bench上实现73%（220/300）的成功率，比最强基线高29%，在单文件和跨文件两类漏洞上均表现突出。

**⚠️ 局限性**

局限性包括：对其他语言或漏洞类型的泛化尚未验证；依赖大模型，模型规模对性能影响显著；补丁生成仍易出现格式错误或过度修改，导致验证失败。

---

## 348. Enforcing Cryptographic Distributed-VCS Access Control with No Trust on Servers

**arXiv ID:** 2607.29417 | [PDF](https://arxiv.org/pdf/2607.29417v1)

**作者:** Xin Xu `[一作]` (Tsinghua University), Yongfeng Huang `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为DVAC的完全分布式版本控制系统（DVCS）访问控制方案，能够在无中心服务器的环境下对文件级别的读写权限实施加密控制。

**💡 创新点**

创新点：将属性基加密（ABE）和属性基签名（ABS）结合用于读写权限细粒度控制；通过以太坊智能合约实现访问策略的分布式可信管理；在现有Git等DVCS中仅做轻量级集成，保持最小改动。

**🔧 技术方法**

采用技术包括：CP‑ABE、ABS、以太坊（Solidity）智能合约、Git过滤器（clean/smudge）驱动、PBC（基于配对的密码学库）以及Web3.js等工具。

**📊 数据集**

数据集：对GitHub公开仓库（超过280万仓库）进行文件大小分布统计；在10个星标最高的仓库上进行性能实验。

**📈 对比分析**

比较方法：在读写控制粒度、密钥管理开销、是否支持分布式场景以及是否需要VCS适配四维度对比DVAC与Gringotts、git‑crypt、GitHub Enterprise等方案；性能评测在MacBook Pro上测量提交(commit)和检出(checkout)延迟，DVAC平均延迟约0.25 s，低于Gringotts但略高于git‑crypt，整体保持毫秒级。

**⚠️ 局限性**

局限：属性基密码学计算开销高；智能合约交互带来以太坊网络延迟与交易费用；未实现动态权限撤销和多授权者细粒度撤销；对资源受限设备的支持尚未充分考虑。

---

## 349. Self-Play Meets Skill Evolution: Self-Evolving Search Agents that Pose, Solve, and Remember

**arXiv ID:** 2607.29468 | [PDF](https://arxiv.org/pdf/2607.29468v1)

**作者:** Zenghuang Fu `[一作]` (University of Chinese Academy of Sciences), Changwei Wang `[通讯]` (Qilu University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了自进化技能增强代理（SESA），一种将自我对弈（self‑play）与持续的技能记忆融合的训练框架。

**💡 创新点**

创新点在于：① 通过失败提炼（failure distillation）把自对弈中的边缘失败转化为可检索的、可持续的技能；② 采用异步的挑战者-求解器架构，使得挑战者只看到目标而无法直接利用技能，保证了任务生成的自适应性；③ 通过前沿塑形（frontier shaping）将挑战者的难度调节至求解器的能力边界，实现了真正的闭环自进化。

**🔧 技术方法**

技术方法包括：搜索自对弈（SSP）框架；Group Relative Policy Optimization (GRPO) 的无评论器策略梯度；技能检索（dense encoder + Top‑K）；前沿塑形的双端奖励；三阶段的失败生命周期（检索、收集、合并）与维护策略；以及可选的推理时检索（SESA‑On）与仅参数迁移（SESA‑Off）。

**📊 数据集**

使用的数据集为七个检索式问答基准：NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue 以及 Bamboogle，总共 3,125 题目，涵盖单跳与多跳检索任务。

**📈 对比分析**

通过与基线 SSP（无技能记忆）和 SkillRL（固定任务下的技能学习）对比，SESA 在 7 个基准的平均准确率上提升 1.2–3.2 分；在 Qwen3‑4B/8B 上与 SSP 的差距分别为 2.3–3.2 分；SESA‑Off（去除检索）相较 SSP 仍能提升 1.8–2.2 分，表明大部分提升归功于参数迁移；SESA‑On 则在此基础上额外提升 0.5–1.0 分，证明外部记忆在推理时可提供补充增益。

**⚠️ 局限性**

局限性包括：① 对检索质量和技能去重的依赖，若检索误差大或去重策略失效会引入噪声；② 检索时需额外计算和存储，部署时可能受限；③ 目前仅在检索式问答领域验证，迁移到更广泛的任务需要进一步研究；④ 初始手工写入的 seed 技能在不同模型/任务间迁移效果不一。

---

## 350. Parameter-Free Heavy-Tailed Bandits

**arXiv ID:** 2607.29460 | [PDF](https://arxiv.org/pdf/2607.29460v1)

**作者:** Gianmarco Genalti `[一作]` (Politecnico di Milano), Alberto Maria Metelli `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

针对不知尾指数 ϵ 与矩界 u 的重尾多臂赌博机问题，本文设计了自适应算法并给出了对应的渐近无假设的 regret 上界。

**💡 创新点**

创新点在于：①首次给出关于 u‑自适应的 regret 前沿，并证明该前沿是最优的；②提出一个无需任何先验参数的 Explore‑Then‑Commit（ETC）算法，能够在满足 u‑自适应的同时实现最佳的分布依赖与分布无关收益权衡；③扩展到同时未知 ϵ 与 u 的情况，给出“对 ϵ=1 校准”策略并证明其在所有固定 ϵ 下达到未知-u 前沿；④通过构造配对下界证明不可能实现对所有 ϵ 的统一亚线性 regret。

**🔧 技术方法**

技术主要包括：Median‑of‑Means（MoM）稳健估计、基于 B_T 设定的块划分、调度探索长度 L_T 的参数化设计，以及通过变化参数 (α, q) 对 T 与 K 的权衡进行精确分析；还使用了变换测度、构造实例和对数极大化等方法来证明下界。

**📊 数据集**

本文为理论分析，未使用任何实验数据集；所有结果均为理论上限与下界。

**📈 对比分析**

与传统已知 ϵ、u 的鲁棒算法相比，本文的算法在不知参数时仍能保持相同阶数的分布无关 regret 上界；对固定 ϵ 的实例，算法实现了与已知参数相当的分布依赖 regret；然而对 ϵ 的统一优化是不可行的，表明在无先验条件下无法实现最优率。

**⚠️ 局限性**

主要局限在于：①无法提供对所有 ϵ 的统一亚线性上界，需选择校准点；②算法在实际应用中需预设探索参数 α、q，若设定不当会导致性能下降；③结果仅给出渐近阶数，常数与对数项未完全定量；④仍未解决在最弱额外假设下恢复 oracle 速率的第三部分开放问题。

---

## 351. TFGformer: Multivariate Time Series Forecasting via Time-Frequency Graph Learning and Covariate Fusion

**arXiv ID:** 2607.29459 | [PDF](https://arxiv.org/pdf/2607.29459v1)

**作者:** Yu Sun `[一作]` (Beijing University of Posts and Telecommunications), Yan Sun `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TFGformer框架，利用时频图学习与协变量融合实现多变量时间序列的长周期预测。

**💡 创新点**

通过STFT与可学习的Mahalanobis距离构建稀疏动态图，再用Gumbel‑Softmax自适应采样生成可解释的变量依赖，同时采用MLP协变量融合模块高效注入历史与未来上下文信息。

**🔧 技术方法**

采用短时傅里叶变换（STFT）、可学习Mahalanobis距离、Gumbel‑Softmax采样、MLP协变量融合、Transformer自注意力等技术。

**📊 数据集**

在七个公开基准上验证，包括ETT（ETTh1/2/ETTm1/2）、ECL、Traffic和Weather。

**📈 对比分析**

与iTransformer、TiDE、PatchTST、Autoformer、FEDformer、DLinear六大基线对比，TFGformer在七个数据集上平均MSE提升3.6%–16.4%，在6/7个数据集获得SOTA。

**⚠️ 局限性**

对Gumbel温度、图稀疏度等超参数敏感；在极稀疏或高维场景下可能需进一步调优；目前仅在中小规模数据集验证，尚未评估在大规模基础模型或异常检测等更广泛任务中的表现。

---

## 352. QR-Structured Thermal Triggers for Targeted Semantic Attacks on Infrared Vision-Language Models

**arXiv ID:** 2607.29445 | [PDF](https://arxiv.org/pdf/2607.29445v1)

**作者:** Xiang Chen `[一作]`, Chengyin Hu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了针对红外视觉-语言模型（IR‑VLM）的黑盒目标语义引导攻击，提出利用 QR 码结构构造低对比度热触发器。

**💡 创新点**

创新点在于将 QR 码的功能区域与可编辑的冷/中/热模块相结合，构造可解释的低对比度热触发器，并通过分阶段梯度无关搜索实现目标语义转移。

**🔧 技术方法**

使用了混合离散-连续搜索、三阶段梯度无关优化（区域搜索、模块拓扑搜索、渲染精化）以及目标对齐与视觉隐蔽损失的联合优化。

**📊 数据集**

使用了30类红外测试集（10张/类）和四个 CLIP 变体（OpenCLIP、Meta‑CLIP、EVA‑CLIP、OpenAI CLIP），并迁移评估到 IR‑图像字幕与 VQA 模型。

**📈 对比分析**

与 AdvGrid、AdvICRS、HCB 等结构化热扰动基线对比，平均攻击成功率 34.28%，显著优于基线；在字幕与 VQA 任务中也实现了最低的语义漂移率。

**⚠️ 局限性**

局限性包括对不同模型的鲁棒性差异大、对热强度、触发器尺寸等参数敏感，且尚未验证在更大规模 IR‑VLM 或真实物理环境中的可行性。

---

## 353. Zero-Mem: Zero-Token Memory Operations for LLM Agents

**arXiv ID:** 2607.29377 | [PDF](https://arxiv.org/pdf/2607.29377v1)

**作者:** Yilin Xiao `[一作]` (Hong Kong Polytechnic University), Xiao Huang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Zero-Mem框架，实现LLM代理在长交互中的零-token记忆操作，仅在最终回答阶段调用LLM。

**💡 创新点**

创新点在于将原始交互痕迹保留为证明来源，构建实体–上下文图与时间层次两种结构化视图，完成无生成的证据检索与校准，彻底消除记忆阶段的LLM调用。

**🔧 技术方法**

使用命名实体识别、稀疏图传播、层次检索、双视图融合与闭包、确定性证据校准等技术；依赖BGE-M3稠密嵌入、BM25词表索引，和轻量化的查询路由。

**📊 数据集**

在LoCoMo（多会话长记忆）和HotpotQA（长文本多跳检索）两个公开基准上进行评估，分别使用GPT-4o-mini和Qwen2.5-14B作为后端阅读器。

**📈 对比分析**

与多种基线（LONG‑LLM、RAG、A‑Mem、Mem0、MemoryOS、LightMem、SimpleMem、CompassMem、GAM）对比，Zero‑Mem在两种基准上均取得最高或次之的F1/BLEU-1分数，并在内存操作上将LLM token与时延分别降低100%和57.6%。

**⚠️ 局限性**

局限包括：仍需编码器计算、在极端超长上下文或多模态交互中的适用性未验证；依赖外部NER工具与预训练嵌入，可能对低资源语言或专业领域表现受限。

---

## 354. Beyond Component Testing: Validating Agentic AI Systems

**arXiv ID:** 2607.29405 | [PDF](https://arxiv.org/pdf/2607.29405v1)

**作者:** Fabio Orazio Mirto `[一作]` (University of Messina), Giovanni Merlino `[通讯]` (National Interuniversity Consortium for Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 257 篇关于 agentic AI 系统验证的文献进行系统综述，提出了涵盖行为、安全、时间、监管和多体系统的五维验证分类法，映射现有方法并量化覆盖空白，最后给出以生命周期为导向的研究路线。

**💡 创新点**

创新点在于：①将验证目标从传统组件测试扩展到整个 agentic 轨迹；②构建五维验证框架，强调时间有效性与监管可审计性；③利用 PRISMA 与编码方案进行定量空白分析；④通过跨领域案例（医疗、工业、智慧出行）验证框架的通用性；⑤提出以规范化证据与生命周期监控为核心的研究议程。

**🔧 技术方法**

技术手段包括：系统性文献检索（ACM、IEEE、Scopus、arXiv、Semantic Scholar），PRISMA‑style 筛选流程，双人编码与一致性检验，维度映射与热图可视化，案例构建与指标定义，以及对比性分析。

**📊 数据集**

数据集为从 7,197 条检索记录中筛选出的 257 篇核心文献，涵盖 2019‑2026 年的研究，并对每篇文献按验证维度、方法族和应用场景进行标注。

**📈 对比分析**

本研究为综述性工作，不涉及模型训练或实验，因此无传统意义上的性能指标；其比较方法为对 257 篇文献按验证维度的覆盖度计数与热图呈现，展示各方法族在不同维度上的成熟度与缺口。

**⚠️ 局限性**

局限性包括：①检索覆盖可能受词汇不统一、数据库偏好（如 IEEE 占比高）影响；②筛选过程主要由单一评审完成，尽管后续检验一致性；③只考虑英文文献，忽略非英语研究；④对方法族划分与维度标签可能存在主观性；⑤未涉及最新 2026 年后发表的工作，且未对方法实施细节进行实验验证。

---

## 355. ALIVE: Warnings Before Exclusion in Budgeted Multi-Source Learning

**arXiv ID:** 2607.29400 | [PDF](https://arxiv.org/pdf/2607.29400v1)

**作者:** Xiyang Zhang `[一作]` (Harbin Institute of Technology), Yuanhe Tian `[通讯]` (Zhongguancun Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 ALIVE 框架，用随机前缀、证据缓存、警告、锁定和容量检查实现对共享身份多注释器学习中持久源操作的可审计控制；

**💡 创新点**

通过分层动作（非锁定警告、锁定请求、持久排除）与随机前缀和证书组合，给出全家族错误持久行动的理论保证，并在 PPR 引擎下显著降低证据延迟；

**🔧 技术方法**

随机前缀抽样、基于 Hoeffding 与 PPR 的置信区间/证书、直接对比与 Serfling 纠正、基于熵的类别平衡选择（CBE）、随机化类平衡选择（CBR）、容量检查与审计成本账本；

**📊 数据集**

使用 CIFAR‑100 进行控制器开发，CIFAR‑10 验证集评估性能；Bluebirds 完整面板用于自然审计成本评估；

**📈 对比分析**

与 ALIVE‑CBR 的路由‑仅模式、完整 CBR 以及 PPR 与 Hoeffding 证据比较。持久行动在匹配控制器中提升平均 AUBC +0.1935pp（全检验显著）；PPR 在 e40/e60 相较 Hoeffding 的证据量更少；全系统对 CBR 的提升为 +0.1954pp，但未达预设的多重校正显著性；

**⚠️ 局限性**

假设固定支持与标签、理想均匀前缀、至少有严格多数、τ=1/S 以及无漂移；缺少对部分重叠、漂移、不同特征空间的处理；使用合成标签变化、固定验证集，成本账本仅为可复现代理，未覆盖真实算力或金钱成本；

---

## 356. Dense Temporal Contrast Synthesis via Conditioned Latent Transport

**arXiv ID:** 2607.29394 | [PDF](https://arxiv.org/pdf/2607.29394v1)

**作者:** Smriti Joshi `[一作]` (Universitat de Barcelona), Karim Lekadir `[通讯]` (Institució Catalana de Recerca i Estudis Avançats)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于条件潜在传输的单步模型，用预对比图像在任何时间点生成动态对比增强的DCE‑MRI序列。

**💡 创新点**

创新点在于将潜在空间与预对比结构锚定、连续时间嵌入以及频率与感知损失相结合，实现空间真实性与时间连续性兼顾，并且在单步推断下显著提升推理速度。

**🔧 技术方法**

使用自定义 4×压缩的 VAE 作为编码器/解码器，搭配基于 U‑Net 的潜在生成网络，并采用 LPIPS、焦点频率损失以及时间条件化的正弦嵌入。

**📊 数据集**

在 MAMA‑MIA、Duke‑Breast‑Cancer‑MRI 和私有 Karolinska 数据集上训练，并在独立的外部 KI 数据集上做跨域评估。

**📈 对比分析**

与 U‑Net、pix2pix、CCNet、TeNCA 等基线进行对比，内部验证中 MSE、PSNR、DTW、DTW‑ROI 等指标均优于对手；外部验证显示虽受域移位影响但仍保持最优或次优的时间与空间性能，并在下游肿瘤分割及四位放射科医生的阅读实验中实现显著提升。

**⚠️ 局限性**

受限于训练样本在时间轴早晚段稀疏、对真实病例的精确时间匹配难度大、以及对异常病例的误定位和对不确定性的量化不足，模型在极端肿瘤形态或罕见增强模式下仍可能产生空间误差。

---

## 357. AquaJEPA: Action-Conditioned Multimodal Predictive Representations for Underwater Robot Dynamics

**arXiv ID:** 2607.29393 | [PDF](https://arxiv.org/pdf/2607.29393v1)

**作者:** Alan-Barsag Gazzaev `[一作]` (ITMO University), Sergey Muravyov `[通讯]` (ITMO University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并评估了一种基于行动条件的联合嵌入预测模型（AquaJEPA），融合RGB摄像头、前向声呐和姿态信息，在部分可观测的水下机器人环境中实现更鲁棒的运动预测与闭环控制。

**💡 创新点**

将EMA目标与动作敏感的对抗边缘损失结合，形成控制敏感的潜在预测；实现mask‑aware多模态融合与模态丢失训练；在共享的回溯式规划器中使用物理辅助头（速度变化、声呐轮廓）进行动作评分。

**🔧 技术方法**

联合嵌入预测（JEPA）、EMA目标、GRU动作编码、mask‑aware融合、模态丢失、对抗动作边缘损失、共享回溯式规划器、Stonefish仿真、BlueROV2机器人。

**📊 数据集**

12个30秒的Stonefish BlueROV2仿真轨迹（8个训练、2个验证、2个测试），包含RGB、声呐、IMU、DVL、压力、姿态和八个推进器命令，进一步在120个因子实验中进行冻结对比。

**📈 对比分析**

与5个基线（反应式、状态仅、普通多模态、监督动力学、递归世界模型）在120个因子场景下进行配对bootstrap比较，AquaJEPA在成功率、最终误差和清晰度方面均优于所有基线，特别是对动作敏感的预测显著提升。

**⚠️ 局限性**

仅在Stonefish BlueROV2仿真中验证，单一机器人、离散动作库、简化碰撞几何，缺乏真实环境验证与更复杂的水动力、声呐噪声等。

---

## 358. On the Resilience of 5G NR Against Jamming

**arXiv ID:** 2607.29384 | [PDF](https://arxiv.org/pdf/2607.29384v1)

**作者:** Sotiris Michaelides `[一作]` (RWTH Aachen University), Martin Henze `[通讯]` (RWTH Aachen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

基于 ns‑3 模拟器实现了开源 5G NR 干扰器，并系统评估了不同物理层配置（频段、子载波间距、信道带宽）对 5G 与 LTE 反干扰能力的影响。

**💡 创新点**

首次公开了可在 ns‑3 内完整模拟 5G NR 干扰的工具，并通过可重复、可控的仿真框架揭示了频段与带宽是决定干扰鲁棒性的关键因素，子载波间距对鲁棒性影响甚微。

**🔧 技术方法**

利用 ns‑3 及其 5G‑LENA 扩展实现干扰器，采用 OFDM/SCMA 物理层模型，并通过仿真参数调整实现不同频段（FR1/F R2）、子载波间距（15/30/60/120 kHz）和信道带宽（20/50/400 MHz）的多场景测试。

**📊 数据集**

未使用真实测量数据集，而是采用 5G‑LENA 提供的 3GPP TR 38.901 校准的宏基站/微基站信道模型，构建了包含六个 UE、一个干扰器和多基站的测试网格。

**📈 对比分析**

通过在相同仿真环境下逐步增加干扰器数量（1/2/4/6/12）与功率（15/30/45/60/75 dBm）来对比性能；关键指标为丢包率、吞吐量与时延。结果显示：FR2 及宽带（400 MHz）配置在所有干扰条件下保持 0 % 丢包与 <10 ms 时延；而 FR1 及窄带配置在高功率或多干扰器时出现 100 % 丢包；子载波间距对性能影响微乎其微。

**⚠️ 局限性**

主要局限：仅考虑了持续高功率的散射式（barrage）干扰，未探讨智能干扰或动态反制；仿真模型无法完整捕捉 mmWave 真实环境中的阻塞与多径变化；结果仅在模拟中得到，缺乏实测验证；干扰器实现依赖 ns‑3 的 5G‑LENA 版本，迁移至其他仿真平台需适配。

---

## 359. SAGP: Semantic Affordance-Guided Grasp Planning via Coarse-Zone VLM Reasoning

**arXiv ID:** 2607.29374 | [PDF](https://arxiv.org/pdf/2607.29374v1)

**作者:** Muhayy Ud Din `[一作]` (Khalifa University), Irfan Hussain `[通讯]` (Khalifa University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练、基于粗粒度区域划分的语义扶承引导抓取规划框架（SAGP），实现了视觉语言模型（VLM）与几何抓取规划的无缝衔接。

**💡 创新点**

创新点在于：①通过PCA对齐与DBSCAN聚类将点云分解为顶部、中部、底部、侧面和突出部等少量粗粒度区域，构建了VLM可以可靠推理的接口；②采用结构化零样本VLM查询获得区域扶承评分并与几何、可达性、任务一致性等多项指标融合，形成五维打分函数；③实现了完全无监督、零样本、训练免费化的语义重排序流程。

**🔧 技术方法**

技术手段包括：RGB‑D感知与点云提取、抗极对抓取候选生成、PCA轴对齐与DBSCAN突出部检测、预训练VLM（Qwen‑VL）零样本区域评分、五项指标融合的打分与重排序、以及基于Franka Panda的PyBullet仿真执行。

**📊 数据集**

实验数据集为14个YCB物体（含容器、工具、长条、盒子、圆柱），在PyBullet仿真环境中用Franka Panda机器人完成抓取任务。

**📈 对比分析**

与几何规划基准（GO）和直接VLM重排序基准（VD）对比：SAGP在保持90%+抓取成功率的前提下，显著提升了首选区域准确率（>60%）、姿态保持率及避免危险区域率；VD在缺乏区域抽象时表现最差；SAGP在非对称、功能区明显的物体上表现尤为突出。

**⚠️ 局限性**

局限性包括：①对点云稀疏或物体尺寸极小的情况，法向估计与突出部检测可靠性下降；②高度对称物体时VLM对所有区域评分相似，SAGP退化为几何规划；③对VLM训练分布外的罕见姿态可能产生噪声评分；④一次性VLM查询耗时1–3秒，虽然可缓存但仍对实时性有影响。

---

## 360. Exploratory Integration of EEG Spectral Features and Gaze Variability for Mild Cognitive Impairment Discrimination

**arXiv ID:** 2607.29493 | [PDF](https://arxiv.org/pdf/2607.29493v1)

**作者:** Takeru Mukunoki `[一作]` (Kobe University), Takashi Nagamatsu `[通讯]` (Kobe University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本研究探索了将EEG频谱特征与注视稳定性（眼动）特征相结合，用于区分轻度认知障碍（MCI）与健康对照的可行性。

**💡 创新点**

创新点在于首次将脑电神经活动变异与视觉行为变异整合，证明两者互补提升分类性能，并通过L1正则化筛选关键特征后进一步加入眼动特征实现显著提升。

**🔧 技术方法**

采用EEG 10‑20系统采样、Welch功率谱估计、LASSO特征选择、逻辑回归分类以及留一交叉验证（LOOCV）评估模型；眼动通过EyeLink采样、排除扫视后计算x、y方向标准差。

**📊 数据集**

使用的数据库为38名受试者（17 MCI，21 健康对照）MoCA评分分组的数据集，所有受试者均完成20 s的固定点注视任务。

**📈 对比分析**

比较了三种模型：①全EEG高维特征；②LASSO筛选的两条EEG特征；③LASSO特征加上眼动SD。性能分别为AUC = 0.52、0.64和0.78，显示逐步提升，尤其是加入眼动后提升约24%。

**⚠️ 局限性**

局限包括样本量小、缺乏外部验证、特征选择与评估未完全分离、仅使用了基本眼动SD指标，未涵盖更丰富的眼动描述符。

---

## 361. Weight-Space Mixture-of-Experts for Implicit Neural Representation Classification

**arXiv ID:** 2607.29463 | [PDF](https://arxiv.org/pdf/2607.29463v1)

**作者:** Stanislaw Janik `[一作]` (Institute of Fundamental Technological Research, Polish Academy of Sciences), Michal Byra `[通讯]` (Institute of Fundamental Technological Research, Polish Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出一种层次混合专家（HMoE）Transformer，用于在隐式神经表示（INR）的权重空间中进行分类，并结合元学习来优化权重初始化；

**💡 创新点**

创新点在于将层级MoE结构与权重空间处理结合，显式利用INR层级和令牌的专门化，并提供基于梯度的权重空间归因与剪枝方法以解释模型决策；

**🔧 技术方法**

主要技术包括SIREN隐式网络、元学习（MAML/Meta‑SGD）、层级MoE Transformer、梯度加权类激活图（Grad‑CAG）以及结构化剪枝；

**📊 数据集**

使用的数据集包括低分辨率图像集MNIST、Fashion‑MNIST、CIFAR‑10以及高分辨率Imagenette和ImageNet‑1K；

**📈 对比分析**

在所有基准上均超过了之前的权重空间分类方法，尤其在MNIST、Fashion‑MNIST和CIFAR‑10上分别取得99.06%、90.72%和65.01%的准确率，在Imagenette和ImageNet‑1K上也分别实现了新SOTA；

**⚠️ 局限性**

局限性包括需要为每张图像训练单独的INR，导致计算成本高，混合专家模型训练更慢且相较于传统像素级模型在大型数据集上仍落后，且解释方法仅限于权重重要性和剪枝，未覆盖更广泛的可解释性需求。

---

## 362. End-to-End Fairness Optimization with Fair Decision-Focused Learning

**arXiv ID:** 2607.29441 | [PDF](https://arxiv.org/pdf/2607.29441v1)

**作者:** Yu Wang `[一作]` (Stevens Institute of Technology), Chen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了将预测与决策公平性整合到资源分配的端到端公平优化框架，并提出了FDFL训练方法。

**💡 创新点**

将预测公平、预测准确与决策公平三者统一为多目标学习，通过求解决策雅可比和闭式解析，首次实现端到端公平优化；提出两层α‑公平度量和相应的梯度组合规则。

**🔧 技术方法**

采用多任务学习（静态/动态权重）、决策聚焦学习、闭式α‑公平分配解析、cvxpylayers求解通用凸优化的雅可比、以及有限样本泛化界。

**📊 数据集**

单资源实验使用 48,784 名患者的真实医疗数据（Black 与非Black 两组）；多资源实验使用 4,000 名合成利益与成本样本，且可调节组间不平衡。

**📈 对比分析**

与 PTO、DFL、FPTO、Regret‑and‑MAD/MSE、SAA、WDRO 等基线对比，结果表明 FDFL 在决策不确定性大、预测精度低或组不平衡时能显著降低决策 regret 并减少预测不公平度，且在所有实验中保持或提升预测准确率。

**⚠️ 局限性**

仅考虑确定性决策、特定的组公平度量，闭式雅可比仅适用于单预算 α‑公平分配；泛化界仅针对固定标量化，未覆盖动态权重；实验多基于合成或构造的效益，缺少真正观测的真实结果。

---

## 363. Beyond Retrieval: Analytic Memory for Multimodal Agents

**arXiv ID:** 2607.29440 | [PDF](https://arxiv.org/pdf/2607.29440v1)

**作者:** Zhoujin Tian `[一作]` (HKUST), Xiaofang Zhou `[通讯]` (HKUST)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种融合检索记忆与可执行分析记忆的多视角记忆框架，支持从长期多模态交互历史中提取可查询结构并执行过滤、聚合、排序等操作。

**💡 创新点**

创新点在于：①通过无监督频繁项集挖掘从多模态记录中自适应诱导结构化模式；②将这些模式转化为可执行的表格，并与传统检索记忆并行；③设计了记忆感知的查询规划器，能够基于查询意图和当前记忆状态动态组合检索与分析工具并逐步执行。

**🔧 技术方法**

技术包括：基于LLM的属性抽取器、Apriori式频繁项集与置信度阈值的模式诱导、表格化记忆材料化、记忆感知工具接口、LLM驱动的高层计划与进阶工具实例化。

**📊 数据集**

使用了MemEye和MemGallery两个多模态长期记忆基准数据集，分别包含对话、图像、时间信息等。

**📈 对比分析**

与多种单模态与多模态记忆基线（A‑Mem、MemoryOS、M2A、MMA、MIRIX、MM‑RAG、UniversalRAG）对比，在MemEye和MemGallery上分别提升EM/BLEU‑1/LLM‑Judge、F1等指标，最高可达11.3%点的性能提升。

**⚠️ 局限性**

局限性包括：①属性抽取错误或缺失会影响后续模式诱导和表格构造，导致分析结果不准确；②系统仅使用预定义的工具集，对未知领域的特定分析需求需要手动扩展工具。

---

## 364. ModelEquivBench: Certifying Multi-Relational Evaluation of LLM-Generated Optimization Models

**arXiv ID:** 2607.29431 | [PDF](https://arxiv.org/pdf/2607.29431v1)

**作者:** Penglin Zhu `[一作]` (Chinese Academy of Sciences), Jungang Xu `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多关系、可证实的评估系统，用于对生成的优化模型与参考模型进行逐对语义比较；

**💡 创新点**

创新点在于引入E0–E6七维语义档案，既提供可独立重放的证明，又区分不同语义层面的相等与否，避免将评估压缩为单一准确率；

**🔧 技术方法**

采用了可重复构建、映射对齐、Farkas乘子证明、投影等精确算术技术，构建了Certifying Mapped‑Containment (CMC) 引擎与完整的独立验证流程；

**📊 数据集**

使用Bench4Opt数据集，冻结173个基准问题（346个带结构/无结构对），对GPT‑5.4、Claude Sonnet 4.6、Qwen3.5‑397B‑A17B三种模型快照进行评估；

**📈 对比分析**

比较方法基于E0–E6各维度的定量覆盖率、已证实正负例与拒绝率；结果显示GPT在E0覆盖率最高，但在E2/E5等语义层面仍有多达49个未通过；Sonnet在E0低，Qwen中间；总体上不同模型在不同层面表现差异，无法用单一准确率比较；

**⚠️ 局限性**

局限性包括：仅评估三种模型快照；仅覆盖线性与有界离散模型；E3投影支持有限；E6缺乏负面证据；资源上限导致部分维度未决；样本单次生成，无法量化生成方差；数据集非随机抽样。

---

## 365. SAVVY: Student Attention Visualization for Video-based Learning Analysis

**arXiv ID:** 2607.29413 | [PDF](https://arxiv.org/pdf/2607.29413v1)

**作者:** Shixian Zhou `[一作]` (Hangzhou Dianzi University), Zhiguang Zhou `[通讯]` (Hangzhou Dianzi University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发了一个基于多模态脑信号（EEG‑fNIRS）量化学生注意力，并构建了一个交互式可视化系统，帮助教师从课程结构、音视频设计到个体注意轨迹全方位分析并优化视频课程。

**💡 创新点**

创新点包括：①将EEG与fNIRS进行特征融合并使用双重深度Q网络实现高精度、实时的注意力量化；②设计了概念‑组‑个体三层可视化框架，利用花瓣图、时间轨迹和流图等多模态编码直观展示注意力与内容关联；③结合实验与专家案例验证系统可直接指导教学内容改进。

**🔧 技术方法**

技术方法包括：脑信号预处理与特征提取（谱、小波、CSP、GLM），EEG‑fNIRS多模态融合，双重深度Q网络（DDQN）训练注意力分类，信息密度计算（视觉边缘密度、文本IDF、语音熵等），以及多层级交互式可视化（概念图、条形/条码/多轨道图、流图）。

**📊 数据集**

使用了包含31名学生的EEG‑fNIRS数据集（3个课程场景，覆盖卷积神经网络、矩阵分解、神经元与突触三类教学视频），并在公开驾驶EEG基准上评估模型泛化。

**📈 对比分析**

与SVM、TASNet+DRL和传统DDQN基线比较，模型在三类教学视频上的准确率均超过95%，在驾驶基准上也保持高精度；在信息密度计算方面进行敏感性分析，验证指标鲁棒性；多模态融合比单模态显著提升性能。

**⚠️ 局限性**

局限性：①随着概念数量和学生规模增加，视图易拥挤，需进一步可扩展设计；②系统只能定位问题，无法自动给出具体修订方案；③信息密度基于呈现内容，未捕捉个体真实处理量；④视觉/听觉分解仅为代理指标，未直接测量顶层注意力分配。

---

## 366. Role-Break in Attention Heads: Understanding and Detecting Hallucinations in VLMs

**arXiv ID:** 2607.29412 | [PDF](https://arxiv.org/pdf/2607.29412v1)

**作者:** Mingyu Wang `[一作]` (Peking University), Ying Li `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出基于注意力头级“Role‑Break”特征的视觉‑语言模型幻觉检测方法，能够在不额外训练模型的前提下对幻觉进行标记。

**💡 创新点**

创新点在于发现幻觉表现为各注意力头偏离其稳定的“faithful role”，并证明该偏差线性可读，从而构建轻量级线性探测器。

**🔧 技术方法**

技术上采用源分配向量、ILR变换、标准化残差构造特征，并用L2正则化逻辑回归做检测。

**📊 数据集**

实验使用四个幻觉基准（POPE、AMBER、M‑HalDetect、COCO‑Caption）以及六个视觉‑语言模型（MiniGPT‑4、LLaVA、InstructBLIP、Qwen3‑VL、Qwen3.5、LLaVA‑13B）进行评估。

**📈 对比分析**

与AvgProb、AvgEnt、RepProbing、VIB‑Probe、DHCP等基线相比，平均AUROC达93.23，几乎在所有模型‑基准组合中排名第一，且特征维度低于5,000，显著减少计算成本。

**⚠️ 局限性**

局限在于未探明Role‑Break是幻觉的根本原因还是副作用，缺乏头级干预验证，且当前方法仅在判别式任务中可直接应用，生成式任务的干预策略仍待研究。

---

## 367. FriendBench: Benchmarking Dyadic Familiarity Inference in Humans and Multimodal Large Language Models

**arXiv ID:** 2607.29602 | [PDF](https://arxiv.org/pdf/2607.29602v1)

**作者:** Jeffrey M. Girard `[一作]` (Fluid Concepts Research), Benjamin Peloquin `[通讯]` (Fluid Concepts Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FriendBench基准，评估多模态模型与人类对两人短对话中熟悉程度的判断能力。

**💡 创新点**

通过统一的冰破问题消除语义信息，只保留行为线索，揭示模型与人类在类别偏差和视觉信息利用上的显著差异。

**🔧 技术方法**

使用文本、音频、视频三模态分类、信号检测理论、贝叶斯混合效应模型以及零样本覆盖率等技术进行评估。

**📊 数据集**

基于Seamless Interaction 数据集的 EO冰破任务，从中抽取20秒对话片段，构成96个平衡的熟悉/陌生配对。

**📈 对比分析**

与人类单独评分、众包多数投票和26个模型的零样本评估相结合，发现模型与人类在准确率上相当，但模型更倾向于“陌生”，且在视频中未能充分利用视觉信号。

**⚠️ 局限性**

样本量有限、仅涵盖第一互动、仅两人对话、平衡设计导致对实际基率无关、模型偏差受提示影响、文化适用性不足等限制。

---

## 368. CoDe-SSM: Context-Detail Decoupled State Space Model for Efficient UHD Image Restoration

**arXiv ID:** 2607.29595 | [PDF](https://arxiv.org/pdf/2607.29595v1)

**作者:** Jiaxu Su `[一作]` (Nanjing Normal University), Yefeng Zheng `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CoDe-SSM框架用于超高清图像恢复，将共享降解上下文与局部细节分离处理。

**💡 创新点**

创新点在于通过软聚类原型提取全局共享上下文，再用GCSM进行状态空间模型推理；同时通过LHFM利用高频能量掩膜和稀疏MoE卷积专家恢复局部细节，二者相互补充。

**🔧 技术方法**

主要技术包括软聚类原型、全局聚类中心的状态空间模型（GCSM）、基于Laplacian与Sobel的高频能量过滤器、稀疏专家路由（MoE）以及残差门控融合。

**📊 数据集**

在五个UHD基准数据集上训练与测试：UHD-LOL4K、UHD-Haze、UHD-Blur、UHD-Snow、UHD-Rain。

**📈 对比分析**

与C2SSM、UHDformer、UHD-processer等现有方法对比，CoDe-SSM在PSNR、SSIM、LPIPS上均获得显著提升，同时参数量约2.88M，FLOPs最低，显示出更优的性能与效率。

**⚠️ 局限性**

局限在于LHFM的高频能量门使用全局均值归一化，可能在强光照或噪声环境下放大噪声或抑制弱细节，未来需探索更鲁棒的局部归一化策略。

---

## 369. Explaining AI-Image Detection: What the Heatmap Actually Shows

**arXiv ID:** 2607.29581 | [PDF](https://arxiv.org/pdf/2607.29581v1)

**作者:** Leonid Kuturin `[一作]` (Sirius Educational Centre), Alexander Kalashnikov `[通讯]` (HSE University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了用于检测网店评论照片伪造的二进制分类器，并提供了可解释的关注图。

**💡 创新点**

通过对编码历史的对称重编码与去除压缩特征的两阶段修复，解决了格式偏差导致的检测作弊，并系统化了解释评估。

**🔧 技术方法**

采用冻结的Perception Encoder、SRM+FFT 44个法医特征以及视觉-频域融合的MLP分类头，并使用SLIC超像素注意力、梯度CAM等解释方法。

**📊 数据集**

使用了169,751张真实用户拍摄的WebP评论照片与16,776张10个生成模型生成的JPEG照片，经过产品无交叉拆分并对压缩历史做对称编码。

**📈 对比分析**

与五个公开检测器比较，在对称编码下实现了0.892 PR‑AUC；在解释方面，12/17解释方法在编辑图像上超过随机控制，在生成图像上8/17超过控制，平均像素AP约0.46。

**⚠️ 局限性**

局限包括仅依赖文件格式差异导致的快捷检测、缺乏人类评估解释有效性、局部遮挡与标注不精确，以及对不同替代策略和检测器的稳健性不足。

---

## 370. Safe Vision Language Action Models via Barrier Enhanced Flow Matching

**arXiv ID:** 2607.29569 | [PDF](https://arxiv.org/pdf/2607.29569v1)

**作者:** Kasra Sinaei `[一作]` (Pennsylvania State University), Donald Ebeigbe `[通讯]` (Pennsylvania State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将控制边界函数（CBF）嵌入Flow Matching生成模型的安全推理框架，直接在去噪过程中生成安全轨迹。

**💡 创新点**

在Flow Matching推理中引入平滑的 Log‑Sum‑Exp 边界约束与 QP 修正，实现在动作块级别的安全保证，且无需安全数据集或模型再训练，计算开销极低。

**🔧 技术方法**

利用 Flow Matching、Log‑Sum‑Exp 平滑约束、QP 安全过滤器、速度约束、Vision‑Language‑Action（VLA）模型及可微分控制器技术。

**📊 数据集**

训练数据为约 150k 帧 SO‑101 抓取放置数据与 180k 帧 QArm 操作数据（均无安全标注），测试使用 2D Maze 游戏与两台真实机器人平台。

**📈 对比分析**

与 Safe Flow Matcher、Safe Diffuser 等现有安全生成方法在 2D Maze、SO‑101 与 QArm 平台对比：安全率提升至 100%（或 70% 以上），成功率保持或略增，轨迹平滑度、加速度和曲率显著降低，推理时间与现有方法相当或更快。

**⚠️ 局限性**

仅适用于一阶空间碰撞安全约束；对梯度非零且机器人工作空间无奇异点有依赖；对动态障碍或非空间安全描述的适应性有限。

---

## 371. TransGraspNet: Physically and Geometrically Consistent Manipulation of Transparent Labware

**arXiv ID:** 2607.29567 | [PDF](https://arxiv.org/pdf/2607.29567v1)

**作者:** Hailing Hu `[一作]` (Peking University), Lifeng Zhou `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了TransGraspNet，一个统一的几何‑物理一致框架，实现透明玻璃器具的安全抓取与液体运输。

**💡 创新点**

创新点在于跨阶段几何一致性：边界一致性、表面一致性和物理一致性，通过边缘引导感知、深度重建和抓取评分共同提升。

**🔧 技术方法**

使用了基于ResNet‑101+FPN的E‑CBAM注意力与边缘分支的检测网络、TDCNet+EGAG的深度补全模块以及基于主轴对齐和wrench‑space鲁棒性的抓取重评分。

**📊 数据集**

在Trans10K、ClearGrasp公开数据集上预训练，并在自制的RobotSci‑Glass数据集（包含透明物体实例和真实深度）上微调。

**📈 对比分析**

与TransLab、PointRend、TDCNet、ClearGrasp等方法对比，在实例分割的边界IoU、深度RMSE、抓取成功率等指标均表现最佳，真实机器人实验中获得86%+的成功率并实现无泄漏的高速液体运输。

**⚠️ 局限性**

局限性包括对极端遮挡或完全无纹理玻璃的鲁棒性仍有限，模型在实时性和大规模场景下的计算负荷较高，且在极高速度下仍可能出现微小波动。

---

## 372. Improving the Understandability of Conceptual Models via Abstract Notation Engineering

**arXiv ID:** 2607.29552 | [PDF](https://arxiv.org/pdf/2607.29552v1)

**作者:** Amine Abbad-Andaloussi `[一作]`, Hugo A. López `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种语言无关的抽象符号工程方法，用于替代低层构造的复杂配置，以提升概念模型的可理解性。

**💡 创新点**

创新点在于将聚焦从单个构造转向对重复出现的构造组合进行抽象，并通过模式识别、形式化、视觉设计与经验验证形成完整流程。

**🔧 技术方法**

方法结合模式识别、行为编码、视觉符号设计原则（如PoN、Semiology of Graphics）以及基于轨迹的行为规范和Delphi专家访谈。

**📊 数据集**

使用了从50个宣告式过程描述中抽样的文本数据集进行模式出现频率统计，并在DCR图上实现了九个抽象模式（DeCleaR）。

**📈 对比分析**

通过对14名熟悉DCR的参与者进行对照实验，比较标准DCR与DeCleaR在答案正确率、主观质量（经验质量、实践质量）以及偏好上的差异，实验结果显示DeCleaR在主观质量和偏好上显著优于标准DCR，但两者在答题正确率上无显著差异。

**⚠️ 局限性**

局限性包括仅在单一语言（DCR）上验证，样本规模有限、涉及的模式数量有限，以及对外部关系交互的表示仍需改进。

---

## 373. AMTFV: Agentic Mathematical Tool-Flow Verification for LLM Self-Correction

**arXiv ID:** 2607.29549 | [PDF](https://arxiv.org/pdf/2607.29549v1)

**作者:** Rui Zou `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AMTFV（Agentic Mathematical Tool-Flow Verification），一种通过中断–执行–恢复的 MTF 接口，将数学验证建模与低层执行解耦的框架。

**💡 创新点**

创新点在于：①引入 MTF 把验证意图抽象为结构化请求；②通过数学工具箱代理将请求转化为可执行调用；③实现验证工作流的递归修订，保证验证覆盖完整；④使得 LLM 专注高层推理而非代码细节。

**🔧 技术方法**

技术手段包括：多智能体（验证、答案修订、工作流修订）协同；MTF 交互模式；Python 基础工具箱（SymPy、itertools、Fraction、SMT 求解器）；迭代自校正与回溯。

**📊 数据集**

使用了 170 道题目的五个数学推理数据集：AIME 2024/2025（A24、A25）、BRUMO 2025（B25）、HMMT 2025（HMM）和 AMO Bench（AMO）。

**📈 对比分析**

与自然语言校正（reflex、Self‑Refine、Self‑Refl、CheckList）、推理增强（CoT‑Tool）以及验证增强（ProgCo、ProgCo‑Py）进行对比；在 DeepSeek、GPT‑5.4‑Mini 与 Gemini‑2.5‑Flash‑Lite 等多种基模型上，AMTFV 在平均准确率上均超过对手，最高提升达 8.3 分，尤其在中高验证复杂度样本中表现更显著。

**⚠️ 局限性**

局限性包括：对数学工具箱的依赖，难以直接迁移到非数学或更广泛的科学任务；仍受 LLM 生成能力与工具调用精度的限制；在极少数高难度样本中可能出现误修订或过度校正。

---

## 374. A Neurosymbolic Approach for Explainable Early Diagnosis of Alzheimer's Disease

**arXiv ID:** 2607.29530 | [PDF](https://arxiv.org/pdf/2607.29530v1)

**作者:** Ranveer Singh `[一作]`, Sriraam Natarajan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

开发了一个神经符号融合的流水线，用预训练语言模型和符号程序从阿尔茨海默症相关文本中抽取认知受损特征，并构建贝叶斯网络进行诊断支持。

**💡 创新点**

创新点在于结合LLM与符号程序实现符号定位、引入神经-符号反馈机制，并使用LLM生成贝叶斯网络以克服小样本结构学习难题。

**🔧 技术方法**

采用技术包括预训练大型语言模型（LLM）、符号程序、Pydantic结构化输出、贝叶斯网络以及NeSyQuaKE框架。

**📊 数据集**

使用的数据集为小规模的阿尔茨海默症认知评估问卷/访谈文本，来自专家标注并且难以收集。

**📈 对比分析**

通过与规则基提取的PCEE值对比、标准偏差评估影响得分、以及贝叶斯推理结果显示，NeSyQuaKE在信息抽取准确度和推理可靠性上优于基线方法。

**⚠️ 局限性**

主要局限包括对小样本数据的依赖、符号化阶段缺乏完整程序细节、反馈机制仅基于转录上下文且未验证音频一致性，以及阈值设置高度依赖专家经验。

---

## 375. Tri-Space Operational Control of Redundant Multilink and Hybrid Cable-Driven Parallel Robots Using an Iterative-Learning based Reactive Approach

**arXiv ID:** 2607.29500 | [PDF](https://arxiv.org/pdf/2607.29500v1)

**作者:** Dipankar Bhattacharya `[一作]` (Chinese University of Hong Kong), Darwin Lau `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种面向多连杆和混合电缆驱动平行机器人（MCDR/HCDR）的三空间（操纵、关节、驱动）协同控制框架，该框架将实时反应式控制（RC）与迭代学习控制（ILC）结合，能够在满足正向力约束、避免电缆-关节干涉、保持可操作性等多重约束的同时，实现对重复轨迹的高精度跟踪。

**💡 创新点**

核心创新点包括：
- 首次针对具有双重冗余（动力学和运动学）的电缆驱动平行机器人构建完整的三空间控制理论；
- 在RC中引入可调参数化的零空间向量和权重（α、β）来平衡控制力度与约束满足，并通过QP实现实时优化；
- 采用迭代学习优化零空间参数和权重，实现对重复任务的性能提升；
- 将电缆-关节干涉、可操作性下降、关节极限等“不可接受状态”融入软硬约束与避免函数，提升系统鲁棒性。

**🔧 技术方法**

技术手段包括：
- 采用加权伪逆法与线性约束求解的二次规划（QP）实现实时反应式控制；
- 对零空间向量使用对角矩阵参数化并通过粒子群/粒子群优化（PS/PSO）进行离线优化；
- 设计可避免干涉的避让加速度与多约束软硬函数；
- 在模拟与实验平台（CASPR + ROS）上实现闭环实时控制；
- 利用PD闭环误差补偿保证系统稳定性。

**📊 数据集**

数据与测试：
- 在仿真平台CASPR上对三种机器人模型（BMArm、SpiderArm、FASTKIT-Planar）进行多条轨迹（星形、花型、圆柱波、球面螺旋等）的仿真测试；
- 在BMArm硬件平台上进行螺旋轨迹跟踪实验，采集操作空间误差、负载电压、电缆力传感器数据；
- 通过添加白噪声的仿真试验验证鲁棒性。未使用公开数据集，仅使用机器人本身的几何与动力学模型。

**📈 对比分析**

评估方法与性能：
- 采用根均方误差（RMSE）、平均电缆力、平均关节力、计算时间等指标；
- 对比“无避让”与“有避让”以及“无ILC”与“有ILC”三种配置；
- 结果显示：在有避让函数时可避免操纵性丢失导致的失控；ILC后轨迹误差、RMSE均降低（最多约60%），电缆力与关节力下降；实时控制在200Hz下平均计算时间仅约2–3 ms，满足实时性；
- 硬件实验亦验证了与仿真一致的性能提升。

**⚠️ 局限性**

局限性：
- ILC依赖任务的重复性，对非重复或大幅随机扰动的任务收敛性无法保证；
- 需要较为精确的机器人模型（几何、动力学）以保证约束与QP解的可行性；
- 零空间参数调优仍需经验选择，过大/过小会影响鲁棒性；
- 计算量在高自由度机器人上仍可能较高，虽然已实现实时，但进一步缩减开销仍有空间；
- 对未知外部干扰（如突发碰撞）缺乏快速自适应机制。

---

## 376. The Parts Are Greater Than the Sum: Automated Task Sequencing for Efficient Training of Multi-Policy LLMs

**arXiv ID:** 2607.29601 | [PDF](https://arxiv.org/pdf/2607.29601v1)

**作者:** Jiajia Tang `[一作]` (Alan Turing Institute), Adam Sobey `[通讯]` (Alan Turing Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种针对大型语言模型的参数高效微调（PEFT）优化路径组织框架，通过任务分组和任务序列自动生成多策略PEFT架构，实现不同任务在独立的低秩适配器（QLoRA）中并行优化。

**💡 创新点**

将任务的梯度相容性与行为相容性融合构建兼容性矩阵，自动划分任务组并为每组生成独立的QLoRA适配器；并通过自动任务序列设计减少跨任务干扰与灾难性遗忘。

**🔧 技术方法**

低秩适配器 LoRA/QLoRA；主成分分析提取梯度特征；基于距离的聚类与动态规划任务序列搜索；多策略PEFT架构。

**📊 数据集**

TRACE 基准（8个异构任务：C‑STANCE、FOMC、MeetingBank、Py150、ScienceQA、NumGLUE‑cm、NumGLUE‑ds、20Minuten）。

**📈 对比分析**

在 LLaMA‑2‑7B‑Chat 与 Vicuna‑7B‑V1.5 上，按相同总低秩预算与单策略 LoRA、O‑LoRA、随机/人工分组/序列等基线比较。自动多策略框架在 LLaMA‑2 上达到 OP 44.78，BWT +0.013，优于单策略共享 rank‑128 的 OP 42.12；在 Vicuna 上 OP 41.14，优于随机分组的 36.49。

**⚠️ 局限性**

受固定训练预算限制；对极大任务集合时分组策略可能需手工调整；未彻底消除所有任务间干扰；序列搜索依赖预先提取的梯度与行为特征，可能在任务分布变化时失效。

---

## 377. FibVLA: An Efficient Temporal Vision-Language-Action Model with Fibonacci Sampling

**arXiv ID:** 2607.29596 | [PDF](https://arxiv.org/pdf/2607.29596v1)

**作者:** Li Lin `[一作]` (Southeast University), Shuai Wang `[通讯]` (Southeast University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 FibVLA，一个高效的视觉-语言-动作模型，通过斐波那契采样和流匹配实现长周期任务的时序感知与实时推理。

**💡 创新点**

创新点：1) 对数回溯采样结合斐波那契递归实现历史特征可复用；2) 通道时序编码消除背景冗余；3) 采用流匹配生成动作分布；4) Fibonacci递归推理大幅降低推理延迟。

**🔧 技术方法**

技术：对数回溯采样、斐波那契采样、通道时序编码（CTE）、流匹配生成策略、PaliGemma视觉编码器、前缀注意力掩码、KV缓存递归推理。

**📊 数据集**

数据集：LIBERO、MIKASA‑Robo、SimplerEnv‑Bridge、SimplerEnv‑Fractal、Bridge、Fractal，以及自行收集的600k帧 Piper 机器人真实数据。

**📈 对比分析**

与 RT‑1/RT‑2、OpenVLA、π_0、CogACT、TraceVLA、HiF‑VLA 等基线在模拟与真实任务中对比，FibVLA 在长周期任务上提升约7.2%成功率，平均 96.8% 成功率，推理延迟 177 ms，比 TraceVLA/HiF‑VLA 低 30%，在真实世界中平均得分 85.7，优于 π_0 +11 点。

**⚠️ 局限性**

局限性：依赖预训练视觉编码器和 LLM；极端动态场景仍可能受限；采样参数需人工调优；尚未在离线持续学习或多任务环境中验证。

---

## 378. DungeonBench: A Benchmark for Rules-Rich Tactical Reasoning in Dungeons & Dragons Combat

**arXiv ID:** 2607.29577 | [PDF](https://arxiv.org/pdf/2607.29577v1)

**作者:** Ismayil Ismayilov `[一作]`, Kaan Oktay `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 DungeonBench，一套基于 2014 版 Dungeons & Dragons 战斗规则的可执行战术决策基准，包含 Encounter 与 Day 两条赛道；

**💡 创新点**

创新点在于完整的战术观测、引擎生成的可执行选项、分层动作拆解以及将多场战斗通过持续资源和短暂休息连接起来的 Day 赛道；

**🔧 技术方法**

技术主要包括规则解析引擎、3D 2.5D 战场网格、基于规则的可执行选项生成、文本提示和工具查询接口、Gymnasium 兼容包装以及多语言模型决策器；

**📊 数据集**

使用 2014 SRD 的完整规则集（类、子类、法术、物品、怪物特性等）作为基础，并基于此生成 25 个标准化场景（20 Encounter、5 Day）作为评测数据集；

**📈 对比分析**

通过在相同决策流上对五款前沿语言模型（GPT‑5.5、Gemini‑3.1 Pro、Claude‑Opus‑4.7、Grok‑4.3、DeepSeek‑V4）进行对比；Encounter 赛道平均胜率约 80%，Day 赛道仅有 2/5 场景完成，展示了单场战术强度难以直接迁移到跨场景资源规划；

**⚠️ 局限性**

局限在于仅关注战术决策，排除了社交、探索、叙事判定和长周期规划等完整桌面角色扮演游戏要素；

---

## 379. MoRoute: Dynamic Routing for In-Context Multimodal Video Generation

**arXiv ID:** 2607.29545 | [PDF](https://arxiv.org/pdf/2607.29545v1)

**作者:** Chong Gao `[一作]` (Sun Yat-sen University), Jing Li `[通讯]` (HUJING Digital Media & Entertainment Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了统一的多模态视频生成与编辑框架，将冻结的多模态语言模型与预训练的视频Diffusion Transformer通过块级动态层路由连接，实现文本、图像与视频条件的统一生成。

**💡 创新点**

创新点包括：1）块级动态层路由，使每个DiT块自适应选择最合适的VLM层；2）在上下文中直接拼接多模态条件并使用斜槽时域RoPE、稀疏注意力与双时间步调制，实现高效细节保持；3）三阶段逐步训练策略。

**🔧 技术方法**

使用的技术：冻结的Qwen3.5-9B VLM、Wan2.1-T2V-14B Latent DiT、轻量级路由MLP、Slotted Temporal RoPE、Sparse Attention、Dual Timestep Modulation、流匹配训练。

**📊 数据集**

数据集：LAION‑2B（文本-图像），Vchitect‑T2V‑Dataverse（视频），UE5生成的合成编辑对，人工制作的真实视频+自动字幕等，训练集覆盖文本到图像、视频、编辑任务。

**📈 对比分析**

对比方法包括OmniWeaving、Bernini、Kiwi‑Edit、Omni‑Video 2、VACE等；在IntelligentVBench、OpenVE‑Bench、RefVIE‑Bench上平均分均为最高，单项指标也处于领先或第二。

**⚠️ 局限性**

局限性：仅支持文本、图像、视频三种输入；缺乏音频、3D等其他模态；未验证在更大模型和更大数据量下的可扩展性；对合成数据的依赖可能导致生成的细节与真实场景差距。

---

## 380. AuditCoder: Responsibility-Preserving Task Graphs for Auditable Code Generation and Bounded Repair

**arXiv ID:** 2607.29529 | [PDF](https://arxiv.org/pdf/2607.29529v1)

**作者:** Kangjie Huang `[一作]` (Shandong Normal University), Chen Lyu `[通讯]` (Shandong Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在 AI 代码生成过程中记录责任单元的工件，形成可审计的任务图并支持基于证据的局部修复；

**💡 创新点**

核心创新在于：在生成前分配持久责任 ID，保持代码、验证证据与修复历史的绑定，并实现只修复证据支持的节点或分支；

**🔧 技术方法**

技术实现包括：合同注解的任务图、责任映射、节点级代码生成与验证、错误定位器（Traceback、资源、局部拒绝规则）以及有限范围的重构与事务记录；

**📊 数据集**

使用了 APPS 与 ClassEval 两个自然语言到 Python 的基准数据集，并对 200 条 APPS 记录进行后期审计；

**📈 对比分析**

与直接生成、思维链、角色化生成及 AgentCoder 等基线对比，证据驱动的有限修复在 APPS 上提升至 83.0% 准确率（低于 AgentCoder），在 ClassEval 上达到 82.0%（超越 CoT+重试）；

**⚠️ 局限性**

局限性在于：仅 26/60 失败可定位到证据支持的节点或分支，说明缺乏足够强的错误证据；此外，规划与全局回退成本高，且在高度耦合或跨文件的场景中效果有限。

---

## 381. Leveraging Transfer Learning with Class-Specific Decoders for Laparoscopic Segmentation

**arXiv ID:** 2607.29509 | [PDF](https://arxiv.org/pdf/2607.29509v1)

**作者:** Priya Tomar `[一作]` (Fraunhofer IAIS), Rafet Sifa `[通讯]` (Fraunhofer IAIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了多器官手术图像分割中使用器官专属解码器架构以及跨手术域迁移学习的效果。

**💡 创新点**

创新点在于将器官专属解码器迁移至第二个手术数据集，探讨跨域迁移与全网络微调与仅解码器微调的差异。

**🔧 技术方法**

采用基于Attention U‑Net的共享编码-多解码器网络，使用Dice损失、Adam优化器，并实现无微调、解码器微调和全网络微调三种训练策略。

**📊 数据集**

使用了Dresden Surgical Anatomy（DSA）和CholecSeg8K两套多器官手术数据集。

**📈 对比分析**

通过对比共享解码器（CECD）与多解码器（CEMD）在从零训练、无微调、解码器微调和全微调四种方案下的Dice/IoU，发现CEMD全微调达到62.4% Dice，明显优于基线。

**⚠️ 局限性**

主要限制包括类不平衡仍显著导致低像素器官性能不足；解码器单独微调会导致性能下降；仅使用Attention U‑Net且实验仅在单中心数据上，泛化性受限。

---

## 382. Transcript-Managed Transformers: Monotone Multi-Agent Collapse and Universality with Two Pop-Enabled Transcripts

**arXiv ID:** 2607.29496 | [PDF](https://arxiv.org/pdf/2607.29496v1)

**作者:** Sergey Salishev `[一作]` `[通讯]` (AI Foundry), Sergey Salishev (AI Foundry)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文对固定精度 Transformer 的转录管理层进行理论分析，探讨在有限可视窗口下不同转录操作对计算能力的影响。

**💡 创新点**

创新点在于提出“Transcript‑Managed Transducer”这一规范，并证明单个 pop 通道即可实现上下文无关语言，两个 pop 通道即可实现递归可枚举语言，同时证明有限人口多代理系统在标准操作下仍保持有限状态。

**🔧 技术方法**

使用的技术主要是有限状态机、Pushdown Automaton、Hopcroft–Ullman 递推替换模型以及相应的归约证明。

**📊 数据集**

论文为理论工作，并未使用实验数据集。

**📈 对比分析**

通过与经典栈机、图灵机的接受/转导层次进行比较，展示不同转录管理策略在语言识别上的上限；未给出实验性能指标。

**⚠️ 局限性**

局限性：仅适用于固定精度、有限可视窗口和有限代理数；不适用于可变窗口、可写存储或无限代理扩展等更强计算资源。

---

## 383. Convergence and Regret of the Policy Gradient for Multi-Armed Bandits in Diffusion Environment

**arXiv ID:** 2607.29593 | [PDF](https://arxiv.org/pdf/2607.29593v1)

**作者:** Yanwei Jia `[一作]` (Chinese University of Hong Kong), Du Ouyang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

在扩散环境（由随机微分方程描述）下，研究并证明了多臂赌博机（MAB）问题中政策梯度更新的几乎必然收敛性与对数级别的非渐近后悔上界；同时给出了离散时间政策梯度算法的对应分析与收敛保证。

**💡 创新点**

创新点在于：① 使用同一 Lyapunov 函数统一处理连续时间 SDE 与离散时间算法的收敛与后悔分析；② 在任意恒定学习率下实现几乎必然收敛；③ 通过阈值学习率（O(1/d)）获得 O(log T) 的后悔上界，超越了以往仅在极小学习率或特定情形下得到的结果。

**🔧 技术方法**

主要技术包括：随机微分方程（SDE）理论、Itô 计算、构造 Lyapunov 函数与其生成算子分析、随机比较原理、归纳与 Gronwall 不等式、停时技术与马尔可夫过程的占用时间分析。

**📊 数据集**

该工作为纯理论分析，没有使用具体实验数据集，所有结论均基于数学推导与理论证明。

**📈 对比分析**

与现有文献对比：相较于先前需要极小学习率或状态相关学习率的结论，本文在学习率阈值 O(1/d) 下即可获得 O(log T) 的后悔上界；同时提供了离散时间算法的完整收敛与后悔证明，性能上与已知结果相当或优于。

**⚠️ 局限性**

局限性包括：① 后悔上界的常数与臂数 d 的幂次较大，未达到传统 UCB 或 Thompson Sampling 的最优常数；② 结果依赖于唯一最优臂且无平局情况；③ 仍未给出实例无关（instance‑free）的后悔下界；④ 仅在扩散环境下考虑，尚未扩展至更一般的随机过程或上下文赌博机。

---

## 384. ResKV: Reconstructing Omitted Attention Contributions for Fixed-Budget KV Cache Compression

**arXiv ID:** 2607.29591 | [PDF](https://arxiv.org/pdf/2607.29591v1)

**作者:** Yuhang Zhan `[一作]` (University of Electronic Science and Technology of China), Shuo Shang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ResKV，一种将固定 KV 缓存拆分为主缓存与残差缓存的压缩方案。

**💡 创新点**

创新点在于将被驱逐的 KV 信息视作注意力的残差，利用残差缓存在同一 softmax 归一化下恢复分子分母质量，并通过验证代理与动态门控实现自适应残差分配。

**🔧 技术方法**

使用分层 KV 预算分配、Lloyd 聚类构造残差条目、共享 softmax 归一化、动态门控与验证代理以及 FlashAttention-2 兼容的实现。

**📊 数据集**

在 LongBench 与 RULER 两大长文本基准上，对 LLaMA‑3.1‑8B 与 Qwen‑2.5‑7B 指令微调模型进行评测。

**📈 对比分析**

与全 KV 缓存以及 SnapKV、AdaKV、CaM 等主流压缩基线对比，ResKV 在相同 KV 预算下平均提升约 1–3 分，尤其在 10%–20% 预算和 query‑agnostic 场景中表现突出。

**⚠️ 局限性**

局限包括对残差分配的超参数敏感、动态门的额外推理开销以及对查询模式的依赖，且在极低预算下仍可能无法完全恢复稀疏注意力效果。

---

## 385. TraceViT: Grounded Trace Supervision for Visual Abstract Reasoning

**arXiv ID:** 2607.29586 | [PDF](https://arxiv.org/pdf/2607.29586v1)

**作者:** Binnan Liu `[一作]` (Zhejiang University), Wei Hua `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种名为TraceViT的循环视觉推理器，通过为每个迭代步骤提供中间目标来指导模型学习抽象规则的推理过程。

**💡 创新点**

创新点在于引入了基于程序实现的语义单调变换链作为中间监督，并通过任务引用与对象工作空间实现循环状态的外部化以及软轨迹对齐机制，使得模型能够在保持输入信息的同时按顺序学习每一步的变换。

**🔧 技术方法**

使用的技术包括视觉Transformer循环核心、跨注意力编码的任务引用、Slot Attention对象工作空间、软动态规划的轨迹对齐、以及变换链的生成与验证。

**📊 数据集**

训练数据来自ARC-AGI-1与ARC-AGI-2两个基准，使用RE-ARC、ARC-GEN等程序生成器重新写并验证的任务实现，得到约40万到88万条带有中间变换链的训练实例。

**📈 对比分析**

在ARC-AGI-1上，TraceViT-Large以67.8%的pass@2成绩领先于同参数规模的循环与视觉模型；在ARC-AGI-2上获得24.3%的pass@2，显示在更难的多规则任务上仍有显著提升。

**⚠️ 局限性**

主要限制是中间变换链只能从程序实现中获取，对缺乏程序化描述的任务不适用；此外覆盖率仍是瓶颈，模型在部分任务上无法产生正确答案，需进一步提升单视图解码质量。

---

## 386. Sycophancy Undermines Epistemic Vigilance in Cooperative Vision-Language Tasks

**arXiv ID:** 2607.29585 | [PDF](https://arxiv.org/pdf/2607.29585v1)

**作者:** Rupak Sarkar `[一作]` (University of Maryland), Rachel Rudinger `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了一个信息不对称的“找差异”对话任务，用以检测视觉语言模型在多轮合作对话中的知识警觉性。

**💡 创新点**

创新地将知识警觉性与顺从性关联，提出利用任务无关的顺从度消除向量进行激活驱动，显著提升模型在合作对话中的证据遵从度。

**🔧 技术方法**

使用视觉语言模型（如 LLaVA）进行图像理解与策略规划，并结合提示干预与激活驱动技术来调节模型行为。

**📊 数据集**

基于 Abstract Scenes 1.1 数据集，人工生成四类单一修改的图像对，并加入相同图像对作为对照。

**📈 对比分析**

通过真实正例率、平衡准确率和对话中的知识警觉违规率进行评估；激活驱动将违规率从约45%降至31%，并将平衡准确率提升至约72%，提示干预效果有限。

**⚠️ 局限性**

仍存在对空间关系和属性细节的判断不足；激活驱动导致对话探索减少，提示干预导致用词超限；模型在更复杂场景中的知识警觉性尚未完全解决。

---

## 387. LEMUR: Learning to Align with Multi-Objective Reinforcement Learning from Preference Feedback

**arXiv ID:** 2607.29559 | [PDF](https://arxiv.org/pdf/2607.29559v1)

**作者:** Manith Adikari `[一作]` (University of Manchester), Angelo Cangelosi `[通讯]` (University of Manchester)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LEMUR 框架，能够在没有预定义奖励函数的多目标强化学习任务中，使用多位教师的偏好反馈联合学习每个目标的奖励模型并优化多目标策略。

**💡 创新点**

创新点在于：①首次将偏好反馈拆分为每个目标的独立奖励模型，避免将冲突的奖励合并为单标量；②采用权重条件奖励模型与向量奖励重标记技术，使得离线经验可在奖励模型更新后重新使用；③将无监督预训练、共享缓冲、Pareto 模拟退火权重调整等技术整合到一个在线迭代循环中。

**🔧 技术方法**

主要技术包括：偏好式奖励学习（Bradley‑Terry 模型）、多目标 Soft Actor‑Critic（MO‑SAC）、基于状态熵的无监督预训练、共享经验缓冲、向量奖励重标记、权重向量的 Pareto 模拟退火、以及对抗式奖励预估（在对比基线 MORAL 中使用）。

**📊 数据集**

使用 MORL‑Generalization benchmark 中的四个高维连续控制任务（MO‑LunarLander、MO‑Hopper、MO‑HalfCheetah、MO‑MetaWorld）以及 MetaWorld 的 Drawer‑Close 任务，并以脚本教师生成的偏好反馈作为实验数据。

**📈 对比分析**

与 Utilitarian、Naive、MORAL、PbMORL、FPbRL 及 Oracle 进行对比。LEMUR 在所有环境中获得最高或最接近 Oracle 的双目标回报，Hypervolume 和 Sparsity 指标均优于 PbMORL，且在奖励对齐、噪声鲁棒性和查询预算限制下表现更稳健，整体性能明显优于聚合型或单目标基线。

**⚠️ 局限性**

局限性包括：①实验仅使用脚本教师，缺乏真实人类评审的验证；②采用线性标量化，无法覆盖非凸 Pareto 前沿；③未实现主动查询策略，仍需大量人类偏好反馈；④对动态改变教师或目标的适应性尚待进一步研究。

---

## 388. The K-Space Signature: Frequency-Domain Representation Learning for Medical Deepfake Detection

**arXiv ID:** 2607.29541 | [PDF](https://arxiv.org/pdf/2607.29541v1)

**作者:** Riccardo Raciti `[一作]` (University of Catania), Sebastiano Battiato `[通讯]` (University of Catania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种基于频域的K-Space Signature(KSS)框架，用于检测医学深度伪造图像。

**💡 创新点**

创新点在于将MRI图像映射到Log-PSD频域，减去全局解剖先验以提取硬件与生成器痕迹，并结合3D MLP‑Mixer+ArcFace实现无空间偏差的判别。

**🔧 技术方法**

使用频域Log-PSD、全局解剖先验提取、3D MLP‑Mixer结构以及ArcFace角度度量学习。

**📊 数据集**

在多中心真实T1加权MRI（1200例）以及三大生成器（MAISI‑v2、Med‑DDPM、SuperSynth）生成的合成数据上进行实验，并在未见扫描仪构成的开放集上验证。

**📈 对比分析**

通过二分类评估，准确率、ROC‑AUC和PR‑AUC均超过0.99；在零样本泛化中，未见扫描仪的准确率仍达0.93，证明方法对硬件和生成器的鲁棒性。

**⚠️ 局限性**

局限在于数据规模有限、需预先计算全局解剖先验、对不同磁共振对比度和其他3D模态的适用性尚未验证。

---

## 389. ARB: A Matched Authorship-Rewriting Benchmark Dataset for AI-Text Detector Evaluation

**arXiv ID:** 2607.29539 | [PDF](https://arxiv.org/pdf/2607.29539v1)

**作者:** Gaetano Perrone `[一作]` (University of Napoli Federico II), Simon Pietro Romano `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了匹配四种作者与文本表面状态的基准（ARB-Dataset），评估AI文本检测器在不同写作工作流中的鲁棒性。

**💡 创新点**

创新点在于将内容来源与LLM介导的文本表面分离，构建人类原稿经过LLM重写、LLM生成与LLM二次重写的四个对照组，从而揭示传统“人类vsLLM”基准对检测器性能的过度乐观预测。

**🔧 技术方法**

采用了四种开源指令调优大型语言模型（Llama‑3.2‑3B、Qwen‑2.5‑7B、Mistral‑7B、Gemma‑2‑9B）生成文本，结合FastDetectGPT、Binoculars‑falcon‑7b、RADAR、BERT‑Defense、RoBERTa‑Defense等检测器进行评估。

**📊 数据集**

使用XSum、WritingPrompts与OpenWebText三大英语文本数据集，每个数据集抽取600个长度在150–500词的样本，构成1,800个源文本，再生成对应的四个文本变体。

**📈 对比分析**

在严格1%误报率（FPR）下对每个检测器进行宏平均，发现FastDetectGPT和Binoculars‑falcon‑7b在直接LLM生成时Recall可达91–94%，但在人类原稿LLM重写后仅下降至15–31%；同一模型二次重写导致Recall下降幅度仅为10–13%。

**⚠️ 局限性**

局限在于重写强度未与源起点完全匹配，导致两种LLM介导状态的差异难以单独归因；基准仅覆盖英语短文本，未涵盖多语言、长文本或跨模型/多步重写等更复杂写作场景。

---

## 390. From Code Review to Code Critique: Intent, Drift, and Spotlight for AI-Generated Diffs at Scale

**arXiv ID:** 2607.29516 | [PDF](https://arxiv.org/pdf/2607.29516v1)

**作者:** Chandra Maddila `[一作]` (Meta), Peter C. Rigby `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ARCTIC 系统，将 AI 代码审查从逐行检查转为以意图预测、漂移检测和代码聚焦为核心的结构化批判。

**💡 创新点**

创新点包括：从 18,000 条人工审查中构建六主题分类；提出意图预测与后翻译漂移检测；设计基于分类的区域级聚焦（Spotlight）；以及在单一大规模生产环境中进行真实投放实验。

**🔧 技术方法**

主要技术为大型语言模型（Claude Opus 4.5 等）与 LLM‑as‑Judge；意图预测采用零射与多轮推理；漂移检测基于后翻译与五级评估表；Spotlight 采用两阶段 LLM 审查与分级优先级；同时进行 token‑效率与 latency 优化。

**📊 数据集**

使用的数据集包括：18,000 条人工审查（CR2 数据集）、712 条 AI 审查、121 条意图基准 diff、118 条漂移基准 diff、CRBench（约 300 条 AI 生成 diff）以及 247 对连续 diff 版本的实时数据。

**📈 对比分析**

通过与人类分布对比发现 AI 在安全与正确性上显著低于人类；意图预测在 agentic 与 zero‑shot 方案中 F1 分别为 0.860 与 0.844；漂移检测在 LWK 0.776、QWK 0.907、MAE 10.31 上表现优秀；Spotlight 在质量估计和缺陷定位上分别比基线提升 2.4×和 3.3×，且 token 消耗与延迟分别降低约 5×与 6×；现场实验中漂移得分下降 5.76 点，显著优于普通迭代。

**⚠️ 局限性**

主要局限包括：实验仅在单一工程环境进行，结果可能不具普适性；LLM‑as‑Judge 可能带来系统性偏差；漂移趋势研究为准实验，存在自选偏倚；数据因合同限制无法公开；对极大 diff 的适应性与复杂度仍待进一步验证。

---

## 391. The Grokked Illusion: True Equilibrium Mitigates Catastrophic Forgetting

**arXiv ID:** 2607.29503 | [PDF](https://arxiv.org/pdf/2607.29503v1)

**作者:** Xiaotian Zhang `[一作]` (City University of Hong Kong), Ge Zhang `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了高熵神经网络在学习新知识时的抗灾难性遗忘能力。

**💡 创新点**

引入“grokked illusion”概念并证明高熵状态在鲁棒性上优于传统AdamW训练。

**🔧 技术方法**

采用Wang-Landau分子动力学采样、高熵测度、SVD有效秩分析与噪声注入实验等技术。

**📊 数据集**

使用模块算术任务（x+y mod 67）及其三种噪声扩展数据集。

**📈 对比分析**

与AdamW金丝雀权重约束模型对比，WLMD高熵模型在记忆噪声后原任务准确率约95%（AdamW仅约75%），且有效秩更高。

**⚠️ 局限性**

仅在极简任务上验证，未在更复杂任务或大模型上验证，因计算成本高且未探究因果机制。

---

## 392. TOOD: Task-Aware Out-of-Distribution Score Calibration for Continual Learners

**arXiv ID:** 2607.29592 | [PDF](https://arxiv.org/pdf/2607.29592v1)

**作者:** Mostafa ElAraby `[一作]` (University of Montreal), Liam Paull `[通讯]` (University of Montreal)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究持续学习中 OOD 检测的衰退现象，并提出一种无训练、后处理的任务感知能量校准方法 TOOD。

**💡 创新点**

创新点在于发现 OOD 遗忘与分类遗忘是两种不同机制，提出 confidence gap（输出尺度漂移）和 manifold crowding（特征空间拥挤）两条路径，并用任务级能量分解与校准实现对这两种衰退的纠正。

**🔧 技术方法**

主要技术包括：能量分解为任务级能量、均值/鲁棒锚点归一化、最大化任务能量得到最终分数、可选的边际项提升分辨率；整个过程仅需一次前向推理，无需梯度更新。

**📊 数据集**

实验数据集包括 CIFAR‑10（5 任务）、CIFAR‑100（10 任务）和 ImageNet‑1K（100 任务），并使用 OpenOOD 的 OOD 评估数据。

**📈 对比分析**

与多种基线 OOD 检测方法（MSP、Energy、Dice、ADASCALE、ASH、ViM、MDS、NNGuide、BER 等）对比，TOOD 在大多数配置中位列前二，显著提升 AUROC，尤其在 logit 规模漂移（confidence gap）严重时带来最大收益。

**⚠️ 局限性**

局限性：只能校正能量基准的输出尺度，无法修复特征空间中的 manifold crowding；需要少量校准样本和已知任务-类划分，若这两者缺失效果会下降。

---

## 393. DynoDINO: Harnessing Dynamic Latent Information from DINO Features for Multi-Phase Medical Image Segmentation

**arXiv ID:** 2607.29568 | [PDF](https://arxiv.org/pdf/2607.29568v1)

**作者:** Yu-Pu Hsu `[一作]` (National Yang Ming Chiao Tung University), Yu-Chee Tseng `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出DynoDINO框架，用于多相CT分割，融合对齐、混合注意力、差分残差门控等模块；

**💡 创新点**

创新点包括：FFT‑ZNCC预对齐实现轻量化多相配准；共享注意力的Mix‑Attention实现跨相高效融合；差分残差+自适应门控提升对缺相、运动误差的鲁棒性；

**🔧 技术方法**

采用MedDINOv3 ViT骨干、FFT‑ZNCC对齐、FlashAttention、Mix‑Attention、差分残差门控以及Dice+CE复合损失；

**📊 数据集**

在LiTS、PLC‑CECT、WAW‑TACE三大多相CT数据集上进行实验；

**📈 对比分析**

与nnU-Net、SegFormer、Dino U‑Net、MedDINOv3等基线比较，DynoDINO在多相条件下显著提升HD95和NSD，单相亦优于传统CNN；

**⚠️ 局限性**

局限性：对完整三相输入高度依赖，缺失EP导致性能崩溃；训练成本高、参数量大；未在其他器官或模态验证其通用性。

---

## 394. MOT-SR: Multi-Objective Tool-Augmented Scientific Equation Discovery with Large Language Models

**arXiv ID:** 2607.29561 | [PDF](https://arxiv.org/pdf/2607.29561v1)

**作者:** Boxiao Wang `[一作]` (Institute of Automation Chinese Academy of Sciences), Jian Cheng `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种多目标工具增强符号回归框架（MOT‑SR），通过外部分析工具提取变量关系、利用多目标评估（拟合误差、泛化能力、方程复杂度）并通过协作的LLM模块（Meta Strategy Generator与Equation Generator）实现闭环式方程生成。

**💡 创新点**

创新点在于：① 将多种统计、频域、因果、动力学工具与LLM结合，自动化获取结构先验；② 采用多目标（ID/OOD误差+AST长度）并维护Pareto前沿，提升方程可解释性与泛化；③ 通过Meta Strategy Generator动态生成搜索策略，指导Equation Generator产生多样化且符合先验的候选方程；④ 在实验中显著突破传统SR与LLM‑SR的性能。

**🔧 技术方法**

使用的技术包括：大型语言模型（LLaMA‑3.1、GPT‑4o‑mini）、非线性与线性相关性分析、FFT/小波、Granger因果、互信息、Lyapunov指数等工具；抽象语法树（AST）长度评估；Pareto非支配排序与动态阈值；多目标评价指标（NMSE_ID、NMSE_OOD、ASTLen）。

**📊 数据集**

主要数据集：40个标准任务（Oscillation 1/2、E. coli growth、Stress–Strain）、LSR‑Synth‑Chemistry（36个化学动力学合成任务）以及极端质量比螺旋入射（EMRI）轨道演化数据。

**📈 对比分析**

与基线（GPlearn、PySR、uDSR、RAG‑SR、LLM‑SR、SGA、LaSR）对比，MOT‑SR在NMSE、准确率、泛化误差、方程简洁度和收敛速度上均优于对手；在EMRI任务中，MOT‑SR的平均积分误差比神经残差模型低三阶、比LLM‑SR低26.8倍，且在未见轨道上保持低误差。

**⚠️ 局限性**

局限性：评估目标和工具集均为手工设定，缺乏自适应工具发现与扩展；目前仅在低维科学数据上验证，尚未测试高维或更复杂动力学系统；未考虑领域特定约束的自动化引入。

---

## 395. Alteron: A Tool for Behavioral Regression Testing Across NLP Classifier Versions

**arXiv ID:** 2607.29557 | [PDF](https://arxiv.org/pdf/2607.29557v1)

**作者:** Shazzad Hossain `[一作]` (University of Dhaka), Mridha Md. Nafis Fuad `[通讯]` (University of Dhaka)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Alteron 工具，在持续集成流程中使用变形测试（Metamorphic Testing）检测 NLP 分类器在版本更新中的行为回归。

**💡 创新点**

创新点：将变形测试从单模型评估迁移到版本对版本的行为回归检查；构造固定测试语料并利用匹配子集差异化判定回归；提供可配置的 CI 门控和机器可读的报告，直接集成到发布流水线。

**🔧 技术方法**

技术手段：Python CLI、spaCy 与 NLTK 进行句法分析与拼写错误生成、YAML 注册表管理 MR、Metamorphic Relations、固定语料生成与快照存储、CI 可读摘要与退出码机制。

**📊 数据集**

使用的数据集：SST‑2、IMDb、SNLI、MultiNLI、AG News 等 9 个数据集，覆盖情感分析、自然语言推断与鲁棒性评估。

**📈 对比分析**

比较方法与性能：在四个 BERT 族模型（重训练、蒸馏、量化）之间，使用 10 条 MR 生成固定测试集，记录行为快照；对相邻版本在匹配子集上的 MR 通过率变化进行差异化比较；实验发现 16 个行为回归，其中 11 个阻断发布，显示整体指标提升不一定等于行为稳定。

**⚠️ 局限性**

局限性：仅包含 10 条基于规则的 MR，缺乏对大模型或多任务场景的覆盖；评估集中在 BERT 族分类器，其他模型家族未验证；固定语料可能无法捕捉所有语义变形；需要进一步扩展 MR 库与跨模型通用性验证。

---

## 396. Pyramidal Width Can Increase Under Vertex Insertion

**arXiv ID:** 2607.29555 | [PDF](https://arxiv.org/pdf/2607.29555v1)

**作者:** Jinze Zhao `[一作]` `[通讯]` (University of California San Diego), Jinze Zhao (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一个六点整数坐标的三维多面体作为反例，证明在加入顶点时金字塔宽度可能增加；

**💡 创新点**

首次用严格的整数支持平面和有理算术验证了金字塔宽度非单调性的命题；

**🔧 技术方法**

采用了金字塔宽度与面距离等价性、整数支持平面证明和有理数有限规划计算等数学技术；

**📊 数据集**

未使用任何传统机器学习或实验数据集，仅采用手工构造的几何点集；

**📈 对比分析**

通过精确的面距离计算和自包含的验证器验证了所有面距离，未进行实验性能对比；

**⚠️ 局限性**

该反例仅证明了单个具体情况，尚未说明在更广泛类多面体或其他插入条件下的行为。

---

## 397. OSAGEN: Object-Aware Mask Priors and Multistage Decoupled Diffusion for Industrial Anomaly Generation

**arXiv ID:** 2607.29533 | [PDF](https://arxiv.org/pdf/2607.29533v1)

**作者:** Jinyi Xu `[一作]` (Sun Yat-sen University), Chao Huang `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在工业缺陷检测中，作者提出了一个多阶段解耦扩散和查询偏置掩模生成器相结合的框架，用于合成高质量的图像–掩模对，以提升缺陷定位性能。

**💡 创新点**

创新点在于将目标图像结构注入掩模扩散以产生对象感知掩模先验，并通过三阶段课程学习分离正常外观、粗略条件下的缺陷语义和细粒度掩模校准，从而降低对精确掩模几何的依赖。

**🔧 技术方法**

技术包括 Stable Diffusion v1.5 的扩散式填充、LoRA 参数微调、查询偏置（Query‑Bias‑Gen）、Spatial‑CFG、Attn‑Gate、Soft‑Normal‑Latent‑Blend 以及后期的标签材料化。

**📊 数据集**

实验使用公开工业缺陷数据集 MVTec AD 和 VisA，采用统一下游定位协议。

**📈 对比分析**

与 AnomalyDiffusion、DualAnoDiff、SeaS、O2MAG 等基线相比，该方法在 MVTec AD 上 AP‑P/F1‑P 达到 88.1/82.2，VisA 上为 68.5/66.1，显著优于竞争对手并保持较低的生成质量指标 KID 与高分类准确率。

**⚠️ 局限性**

局限性包括需要针对每个类别单独适配（耗时约 2.39 小时），对未见对象的泛化能力有限，以及在极端缺陷形状或尺寸下仍可能出现掩模过跟随现象。

---

## 398. Multi-Source Multi-View Graph Domain Adaptation with Hyperbolic Residual Encoding for Cross-Site MDD Identification from rs-fMRI

**arXiv ID:** 2607.29531 | [PDF](https://arxiv.org/pdf/2607.29531v1)

**作者:** Zhanpeng Zheng `[一作]` (Chongqing Jiaotong University), Yansu Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种统一的多源、多视角图域自适应框架，用于跨站点 rs‑fMRI 数据的重度抑郁症（MDD）识别。

**💡 创新点**

创新点包括：①将 Pearson、稀疏重构和 Granger 因果三种功能连接视角在同一框架中共同建模；②采用双流自适应融合捕捉视角间高阶交互；③使用轻量级双曲余弦残差编码在 Poincaré 球面上进行曲率感知的特征微调；④结合类条件 Cauchy–Schwarz 对齐、对抗学习、信息最大化与置信度感知伪标签，实现多源多视角的联合自适应。

**🔧 技术方法**

主要技术包括：图注意网络（GAT）编码器、双视角交互模块、双流自适应融合、Poincaré 球面残差编码、类条件 Cauchy–Schwarz 对齐、域对抗判别器、信息最大化损失和置信度阈值伪标签策略。

**📊 数据集**

使用公开的 REST‑meta‑MDD 多站点 rs‑fMRI 数据集，来源站点为 Site20、Site21，目标站点共 7 个（Site6、11、12、16、19、22、25）。

**📈 对比分析**

与 BC‑SVM、LE‑SVM、DANN、UFA‑Net、MFCP、AUFA、H2MSDA 等基线比较，平均准确率达 73.60%、AUC 71.90%，相较于最强基线 H2MSDA 提升了约 5.93% 与 4.75% 的指标，并在所有 7 个目标站点上实现最高 ACC。

**⚠️ 局限性**

局限性：仅在传递式（transductive）多源自适应实验中验证；缺乏独立站点验证与不确定性评估；仅使用 rs‑fMRI，未融合多模态信息；对临床可解释性研究不足。

---

## 399. TerraNova: A Foundation Model for the Anthropocene

**arXiv ID:** 2607.29527 | [PDF](https://arxiv.org/pdf/2607.29527v1)

**作者:** Carlos Rodriguez-Pardo `[一作]` (Politecnico di Milano), Massimo Tavoni `[通讯]` (Politecnico di Milano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了TerraNova这一基础模型，联合学习地球系统物理场和国家层面社会经济指标，并支持两种几何（格点和行政单元）和时间维度的查询。

**💡 创新点**

核心创新在于跨几何对齐：通过人口加权的国境-坐标对齐和与预训练地理嵌入的对齐，使得同一潜在空间同时包含物理、行政和图像语义；并通过超网络生成的查询条件解码器实现每个任务的自适应推断和可解释的不确定性。

**🔧 技术方法**

技术包括自定义位置、时间、国家和任务编码器、跨模态 Transformer 融合、超网络解码器、Normal–Inverse–Gamma 预测分布、对比损失（InfoNCE）、时序传输正则化、以及轻量级自适应适配器（MiSS）。

**📊 数据集**

使用两大数据集：WorldTensor（512个0.25°网格物理场）和CountryTensor（512个国家级指标），以及三套冻结的图像嵌入银行（SatCLIP、GeoCLIP、Copernicus-Embed）。

**📈 对比分析**

与现有地理嵌入模型（如GeoCLIP、SatCLIP、RANGE+等）对比，TerraNova在多标签预算下平均R²提升≈0.08；在时序推断、稀疏观测下的场重建、国别推断以及格点下沉降等任务均表现出显著优于基线，且不确定性预测得到校准。

**⚠️ 局限性**

局限包括：模型未匹配专门的地理编码器容量，导致低标签预算下仍被现有基线超越；下沉降仅在国内存在显著空间变化时有效；社会经济指标的观测偏差与历史背景可能导致因果混淆；模型为观测层而非因果或情景模拟工具，需谨慎使用于政策决策。

---

## 400. Students' Practices and Skills in the LLM-Era: "You Can't Outsource the Struggle and Still Get the Skill"

**arXiv ID:** 2607.29519 | [PDF](https://arxiv.org/pdf/2607.29519v1)

**作者:** Enne Rebeca Silva de Freitas `[一作]` (Universidade Federal do Pará), Danilo Monteiro `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对 1,383 条来自五个研究相关子版块的 Reddit 贴文进行主题合成，系统归纳了研究生在使用大语言模型时的实践方式与必备技能。

**💡 创新点**

创新点在于将灰色文献方法应用于 Reddit 社区，首度从研究生自述角度揭示 LLM 在科研流程中的实际使用与认知缺口，为培养 AI‑素养的课程设计提供第一手需求依据。

**🔧 技术方法**

主要采用 Reddit 公共 JSON API 搜索、文本筛选、双人独立编码以及三轮主题合成等质性分析技术；在此基础上使用主题标签和类别映射构建实践与能力维度。

**📊 数据集**

数据集为 1,383 条英文/葡萄牙文贴文，来源于 5 个专注研究生教育的子版块（如 r/gradschool、r/CSgrad 等），时间跨度 2020‑2026 年。

**📈 对比分析**

本研究未设立量化对比基准，而是通过三轮编码与主题归纳的方式，对实践与技能进行质性比较；结论显示学生在利用 LLM 提高效率的同时，往往缺乏验证与批判性评估，导致“认知外包”问题。

**⚠️ 局限性**

局限性包括：数据仅来自 Reddit，可能不代表全部研究生群体；贴文为自述且匿名，难以核实学术身份；样本量虽大但仍为精选子集，缺乏对成功/日常使用场景的覆盖；未进行实证测评或实验验证。

---

## 401. STAGE: STyle-controllable Action GEneration for personalized autonomous driving

**arXiv ID:** 2607.29517 | [PDF](https://arxiv.org/pdf/2607.29517v1)

**作者:** Zihao Liu `[一作]` (Northwestern Polytechnical University), Panfeng Huang `[通讯]` (Northwestern Polytechnical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了可控驾驶风格的自主驾驶动作生成框架 STAGE，允许驾驶员通过实时输入风格值来定制车辆的行驶策略。

**💡 创新点**

创新点包括：
1) 用偏好学习得到连续单调的驾驶风格值；
2) 在动作模态编码中将风格值与 VAE 潜在变量分离，既保留多样性又实现可控性；
3) 设计基于加速度/车距/车道偏差的自动化风格评分规则，显著降低人工标注成本；
4) 将上述模块嵌入 Transformer（DETR‑style）框架，实现端到端的风格可控动作生成。

**🔧 技术方法**

核心技术：
- 变分自编码器（VAE）
- Transformer 编码解码器（DETR 结构）
- 偏好学习与风格评分规则
- 行为克隆、逆强化学习背景
- MetaDrive 仿真平台与 Logitech G923 人机交互数据采集

**📊 数据集**

数据集：
- 基于 MetaDrive 的多场景驾驶仿真数据（包含车道、速度、加速、转向、车距等信息）
- 使用 Logitech G923 手柄与踏板收集的真实驾驶示范数据，覆盖从保守到激进的多种驾驶风格

**📈 对比分析**

与 BC、GAIL、CVAE、CVAE+Discrete Style、BC+Preference Style 等方法进行对比，主要评估指标包括风格可控性、风格连续性、平均完成率和风格对齐度（Spearman R²）。STAGE 在所有指标上均优于对手：完成率 92.1%，风格可控性和连续性均达 2714/2714 分，Spearman R² 接近 1，且在安全性（平均完成率）与驾驶舒适度（油门/刹车使用）方面表现更佳。

**⚠️ 局限性**

局限性：
- 同一风格值在不同路况下可能产生不同的驾驶行为，导致驾驶员需频繁调整风格值；
- 多维风格向量学习受限于数据规模，尚未实现多维连续风格；
- 需要更完善的场景感知与评分机制以统一不同道路情境下的风格对应关系。

---

## 402. Adaptive FastOPD: Progress-Aware Rollout Horizon Expansion for Efficient On-Policy Distillation

**arXiv ID:** 2607.29494 | [PDF](https://arxiv.org/pdf/2607.29494v1)

**作者:** Qian Tan `[一作]` (University of Science and Technology of China), Yuqiang Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种进度感知的 roll‑out horizon 扩展策略，用于高效的 on‑policy distillation

**💡 创新点**

创新点在于：① 通过四种 teacher–student 信号相对基线计算归一化的“坏度”并聚合；② 使用阶段相对进度和长度利用门阈来决定何时扩展 horizon；③ 让每个 horizon 根据自身学习进展而非固定步数或绝对阈值动态决定扩展；④ 通过多信号聚合提升稳健性

**🔧 技术方法**

技术方法包括：进度感知的 horizon 扩展算法、四信号（Top‑k 覆盖、共享概率、共享贪心惩罚、未共享 token 惩罚）归一化与指数滑动平均、利用率门阈、vLLM 在线 roll‑out、OPD 的 clipped 损失、使用 AdamW 训练

**📊 数据集**

训练使用 DAPO‑Math‑17K 数据集；评估在 AIME 2025/2024、AMC 2023、MATH‑500、Minerva Math、OlympiadBench 六大数学推理基准上

**📈 对比分析**

与原始 OPD（7K、15K）以及固定步长 FastOPD（Fixed）比较。Adaptive FastOPD 在两组 teacher–student 配置下都取得最高宏平均准确率，同时将训练时间分别缩短 49.1–71.2%。在 DeepSeek 对组中平均分 56.1 分，耗时 6h16min；在 Qwen3 对组中平均分 20.1 分，耗时 2h37min

**⚠️ 局限性**

仍然需要手动设定 ΔH、N_base、β、K_pat、τ_hit、τ_reach 等超参数；在极端长 roll‑out 情况下可能仍需更细粒度控制；方法依赖多种信号，计算开销略高；对不同模型、数据集的泛化仍需进一步验证

---

## 403. Educating the Agentic Engineer: Curricula, Collaboration, and Continuous Learning in the AI Era

**arXiv ID:** 2607.29610 | [PDF](https://arxiv.org/pdf/2607.29610v1)

**作者:** Mamdouh Alenezi `[一作]` `[通讯]` (Saudi Data and Artificial Intelligence), Mamdouh Alenezi (Saudi Data and Artificial Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了 ACCEL 框架，旨在通过整合能力支柱和三种教育实施向量，重新构建工程教育，使学生具备在代理式人工智能环境下进行意图规范、协作调度、验证评估、伦理治理和持续自我学习的能力。

**💡 创新点**

创新点在于将人类代理理论、自动化监督研究、学习科学和代理式软件工程实践融合，形成了基于能力的系统性教育架构；设计了分阶段的 AI 自主权递增与验证导向的课程进阶；引入了循环式协同验证教学模式和以判断为核心的评估方法；将伦理治理嵌入技术设计而非单独课程。

**🔧 技术方法**

本文主要采用概念合成与理论建模技术，借助人类代理理论、委托理论和学习科学框架来阐释并构造 ACCEL；未涉及具体算法实现。

**📊 数据集**

没有使用任何数据集，论文为理论性概念性综述。

**📈 对比分析**

由于本文未进行实验验证或性能对比，只提出了教育设计与实施的思路和假设，暂无性能评估结果。

**⚠️ 局限性**

主要限制包括：1）仅为概念性框架，缺乏系统实现与实证评估；2）研究范围聚焦软件/计算机工程，对其他工程领域的适用性尚待验证；3）技术快速演进可能导致部分论述过时；4）假设基于单一作者，缺乏跨评审与多样化验证；5）实施需较高资源与机构支持，资源受限环境的可行性未知。

---

## 404. QASP: Query-Adaptive Robust Vector Search Policy

**arXiv ID:** 2607.29606 | [PDF](https://arxiv.org/pdf/2607.29606v1)

**作者:** Hakan Ferhatosmanoglu `[一作]` (Amazon), Andy Warfield `[通讯]` (Amazon)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种基于查询自适应的向量搜索策略 QASP，利用单次监督回归预测每个查询的完整召回曲线，从而为任意召回目标推导最优探测深度。

**💡 创新点**

核心创新在于：①预测全召回曲线而非单一 nprobe；②使用尺度不变、归一化特征实现跨数据集、跨索引配置、跨召回目标的泛化；③提供理论证明，包括有限样本收敛、与固定策略的损失差距可消失、以及数据访问节省随内在维度指数级增长；④通过轻量级的实时补偿机制（EWMA+阈值）实现无额外推理的自适应终止；⑤支持层次化索引的无重训练扩展。

**🔧 技术方法**

技术手段包括：监督回归（深度网络、梯度提升树、Lasso多项式）、召回曲线回归、特征工程（聚类距离、相对对比度、跳变等）、伪维度理论与学习曲线分析、领域自适应（零样本/少样本微调）、轻量级反应式补偿算法、以及综合的查询变异评估框架。

**📊 数据集**

实验使用七个公开向量数据集：SIFT1M、MNIST、GIST1M、DEEP1B_10M、GLOVE-200、COCO-I2I 与 COCO-T2I，涵盖 Euclidean 与 Cosine 距离、不同维度与规模。

**📈 对比分析**

与 Oracle 固定探测策略、PCE-Net、SPANN、SQUASH、LAET、DARTH 等基线进行对比，评估指标为召回方差、绝对偏差、查询满意率和数据访问比例。QASP 在所有数据集上实现了约 57.7% 的召回方差下降、33.6% 的偏差下降、7.3% 的满意率提升，并在 90% 召回目标下仅略高 0.06% 的数据访问量；在更高召回目标和磁盘化部署中，其数据访问量下降幅度超过 80%。

**⚠️ 局限性**

局限性包括：①需要离线生成训练样本，对查询分布的准确采样要求较高；②仅针对基于划分（IVF/层次聚类）的索引设计，对图索引或其他搜索结构的适用性尚未验证；③虽然推理开销低，但仍需额外一次前向推理；④模型在极端难度查询上的预测仍可能出现欠预测，需要进一步改进反应式补偿；⑤理论证明依赖于理想化假设，实际部署中仍需经验调优。

---

## 405. Beyond Resilience: Antifragility in Critical Infrastructure Cybersecurity

**arXiv ID:** 2607.29550 | [PDF](https://arxiv.org/pdf/2607.29550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 406. HAM-VLN: Harnessing Hierarchical Agentic Memory for Zero-Shot Vision-and-Language Navigation

**arXiv ID:** 2607.29600 | [PDF](https://arxiv.org/pdf/2607.29600v1)

**作者:** An Liu `[一作]` (Institute of Automation, Chinese Academy of Sciences), Jian Cheng `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在零训练的视觉与语言导航（VLN）框架中，提出了基于深度地理化的世界图与分层代理记忆（working、episodic、semantic、reflection）来记录和检索导航历史，使机器人能够在每个决策点同时执行高层规划和低层定位。

**💡 创新点**

核心创新点是：① 将导航历史编码为深度地理化的图结构，仅随着发现的新地点与物体增长；② 在每一次LLM决策调用中同步写入语义、进度与失败记录；③ 通过任务子目标驱动的检索，结合相关性、最近性与显著性对图进行压缩；④ 引入反射记忆，将失败经验以结构化标签形式保留，供后续决策重用。

**🔧 技术方法**

技术手段包括：多模态大型语言模型（MLLM）作为 System 2 规划器，轻量级的 System 1 视觉-几何定位模型；基于深度相机的位姿估计和图像分割（如 DINO、SAM）构建物体语义；层次化代理记忆框架；图检索与拓扑扩展算法；双流程（System 1/2）架构实现高速执行与深度推理。

**📊 数据集**

使用的基准数据集有：R2R‑CE、RxR‑CE（基于真实室内环境的 VLN）以及 HM3D‑v2 ObjectNav（基于模拟的三维对象导航）。

**📈 对比分析**

与先前公开的零训练方法相比，<模型名> 在 R2R‑CE 上实现 61.0% SR、RxR‑CE 52.7% SR、HM3D‑v2 ObjectNav 79.7% SR，分别超过前沿零训练基准 18–27% 以及部分监督模型；在 R2R‑CE 上还超越了所有公开的监督方法的 OSR。相比于全原始视觉历史，<模型名> 通过分层检索将上下文长度缩短 65%+，显著降低了 token 消耗；消融实验表明，去除世界图或任何记忆视图都会导致性能显著下降，验证了各组件的必要性。

**⚠️ 局限性**

局限性：① 仍需频繁调用大型 LLM，推理成本和延迟受限；② 记忆图在极长轨迹下仍可能膨胀，需要更高效的压缩策略；③ 对动态或不可见变化的环境适应性尚未充分验证；④ 由于完全零训练，模型在极端复杂场景中的泛化能力仍低于部分监督方法。

---

## 407. SLIM: Saturation-Aware Lightweight Performance Modeling for LLM Serving

**arXiv ID:** 2607.29575 | [PDF](https://arxiv.org/pdf/2607.29575v1)

**作者:** Pol G. Recasens `[一作]`, Josep Ll. Berral `[通讯]` (Universitat Politècnica de Catalunya - BarcelonaTech)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对GPU执行细粒度的内存带宽与算力利用率进行剖析，论文揭示了大规模自回归LLM推理中吞吐量饱和的根本原因，并提出了一种轻量级半解析性能模型 SLIM，用于准确预测推理吞吐量和延迟。

**💡 创新点**

创新点在于：①系统性地将注意力核的算术强度与KV缓存流量关联，首次证明它们在大批量/大上下文场景下驱动DRAM带宽饱和；②构造了 SLIM，该模型将计算与内存传输拆分为可解释的公式，并通过极少量的剖析数据即可实现跨模型、跨序列长度的高精度预测；③基于 SLIM 的 Batching Configuration Advisor（BCA）能够在满足延迟SLO的前提下，自动选取最优批量规模，显著节省GPU KV缓存内存。

**🔧 技术方法**

技术手段包括：NVIDIA Nsight Systems/Compute 对单GPU上 Mistral‑7B 与 Granite‑8B 的核级性能剖析、roofline 可视化、算术强度与内存流量模型化；SLIM 采用半解析公式（预填充 FLOPs 计算、解码算术与 KV 缓存传输、校准系数 η_c、η_m 等），并结合少量实测点进行参数拟合；BCA 则基于 SLIM 的吞吐/延迟估计做离线决策。

**📊 数据集**

实验数据主要来自：OPT 系列（125M‑6.7B）、Mistral‑7B、Granite‑8B、Qwen‑32B/72B 以及 ShareGPT 真实请求集；所有实验均在 NVIDIA Hopper H100 GPU（64 GB HBM2）上完成，采用 vLLM 0.15.1 作为推理框架。

**📈 对比分析**

在与 LLMVisor‑Agg 与 IMAI OOD‑LR 两个基准对比下，SLIM 在吞吐量 MAPE 上平均下降 79.3%，在模型与序列长度泛化场景中 MAPE 低于 6%（输入/输出长度）或 24%（模型规模），并能准确捕捉吞吐饱和曲线；BCA 通过 SLIM 预测实现了与完整剖析一致的批量推荐，并在未见模型/长度下的 KV 缓存内存节省达 55 GB 左右，且吞吐损失极小。

**⚠️ 局限性**

主要局限性包括：仅针对解码阶段的单向自回归 Transformer，未涵盖多头/查询聚合或前缀/量化 KV 缓存等优化；模型对张量并行分片未显式建模；预测精度在极大模型或多GPU环境下仍略低；需要在不同硬件平台重新校准参数。

---

## 408. COntExt: Towards Context-Aware Ontology Extension from Operational Metrics

**arXiv ID:** 2607.29553 | [PDF](https://arxiv.org/pdf/2607.29553v1)

**作者:** Hussain Hussain `[一作]` (Know Center Research), Verena Geist `[通讯]` (Software Competence Center Hagenberg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种名为 COntExt 的框架，利用组织内部已有的结构化运维指标定义来支持领域本体的自动扩展。

**💡 创新点**

创新点在于首次将运维指标作为中间语义资源提供上下文，构建了算法无关的扩展任务（父类预测、关系类型预测、属性赋值），并证明指标语境可显著提升某些扩展子任务的效果。

**🔧 技术方法**

采用的技术包括句子嵌入聚合（ChildAgg）、BERT 及其 fine‑tuning、图注意网络 TaxoExpan、知识图谱嵌入模型 TransE、BERT‑ConvE 与 MLM 预训练模型提示法等多种算法，框架可灵活替换。

**📊 数据集**

实验使用七个公开本体（Pizza、FIBO_BE、SAREF_ener、JRC、CSO_sec、TAC、CertGraph）以及与 CertGraph 相关联的 64 条 YAML 形式的安全指标集做为指标语料。

**📈 对比分析**

通过 leave‑one‑out 评估，比较了多种算法在父类预测、关系类型预测和属性赋值三大任务中的 MRR、Hits@k 和准确率；结果显示 ChildAgg 在多数本体中表现最好，BERT‑ConvE 在关系类型预测上领先，而在含指标上下文的实验中，指标描述与评论可将关系类型 MRR 提升 6pp、属性赋值准确率提升 11pp。

**⚠️ 局限性**

主要局限包括：指标上下文实验仅在单一本体（CertGraph）上验证；未对结构化方法（如 TransE、BERT‑ConvE）进行指标文本注入；未评估大语言模型的效果；因此结果的泛化性和对不同指标格式的适用性仍待进一步研究。

---

## 409. Triangulating Across U.S. Federal AI Transparency Regimes

**arXiv ID:** 2607.29540 | [PDF](https://arxiv.org/pdf/2607.29540v1)

**作者:** Emma Lurie `[一作]` (University of Pennsylvania), Sorelle A. Friedler `[通讯]` (Haverford College)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了美国联邦政府三大AI透明度制度（SORNs、ICRs、AI Use Case Inventory），并通过跨制度链接揭示政府AI系统的真实运作；

**💡 创新点**

创新点在于提出并实现了跨监管机制的零射击分类与跨文档实体解析方法，首次将三套制度下的AI披露信息三维聚合；

**🔧 技术方法**

采用零射击大语言模型（GPT‑5.4‑mini）进行AI相关文档识别，结合TF‑IDF、命名实体匹配、OpenAI嵌入等特征的概率打分与LLM重排序，最后用层次聚类完成跨制度链接；

**📊 数据集**

数据集包括2010‑2025年3,735份SORN文本、2018‑2026年3,871份ICR及其31,670份支持文件、以及2023‑2025年3,542份AI Use Case Inventory条目，共计约5,268条AI相关记录；

**📈 对比分析**

方法通过候选检索→概率评分→LLM重排序→聚类实现跨制度匹配，验证集上同系统识别准确率约85%，关联系统约51%，无关系统100%；跨制度匹配覆盖率约20%，显著高于单一制度覆盖；

**⚠️ 局限性**

局限包括：缺乏统一持久标识导致跨年度追踪困难、跨制度间词汇差异导致检索召回不足、LLM重排序的主观性和验证集规模有限，且高影响系统的风险管理信息仍缺失或不完整。

---

## 410. Homotopy-Aware Corridor Generation without Predefined Reference Paths

**arXiv ID:** 2607.29513 | [PDF](https://arxiv.org/pdf/2607.29513v1)

**作者:** Haoze Dong `[一作]` (Peking University), Zhongkui Li `[通讯]` (Peking University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种无预定义参考路径的协程生成框架，通过在图形凸集（GCS）上直接搜索凸集序列来构造安全通道，并在此基础上实现同伦感知与局部更新。

**💡 创新点**

创新点在于（1）将可见性变形（VD）与均匀可见性变形（UVD）从路径扩展到凸集序列，实现通道级同伦判断与冗余合并；（2）构建自适应多尺度 GCS（细尺度 F‑GCS 与粗尺度 C‑GCS），兼顾全局拓扑和局部几何精度；（3）实现在线局部更新而无需重建整个图。

**🔧 技术方法**

使用技术包括图形凸集建模、可见性测试、BFS 路径搜索、凸包合并、UVD 合并判据、MINCO 轨迹优化以及 GJK 冲突检测。

**📊 数据集**

实验数据集涵盖合成的二维/三维稠密/稀疏障碍场、迷宫以及真实世界的 OptiTrack 基地机器人、三维四旋翼和基于 LiDAR 的车载机器人环境。

**📈 对比分析**

与 VCC、R‑IRIS、SFC、IMPC 等基线相比，本文方法在 GCS 构造时间上比 VCC 快 3.8–13.4 倍、比 R‑IRIS 快 1.3–24.8 倍；生成通道后轨迹持续时间短 13.7–19.8%（2D）和 12.3–16.4%（3D），平均速度更高，角度成本保持在可接受范围；局部更新耗时 0.02–0.06 s，显著快于完整重构。

**⚠️ 局限性**

主要局限在于仅针对静态环境的地图不确定性，未考虑动态障碍、时变约束和多机器人交互；高维空间中细尺度 GCS 的构造仍可能导致计算量增加。

---

## 411. CWEEP: A Lexical Static Analysis Framework for CWE Early Prevention

**arXiv ID:** 2607.29604 | [PDF](https://arxiv.org/pdf/2607.29604v1)

**作者:** Bryan Kwan `[一作]` (University of Calgary), Benjamin Tan `[通讯]` (University of Calgary)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 CWEEP，一种针对 RTL 的词法静态分析框架，用于早期发现硬件安全缺陷，并提供缺陷定位和自动修复建议。

**💡 创新点**

创新点包括：①将资产识别与词法模式检查结合，显著降低误报；②对原有 CWEAT 算法进行优化并扩展至更多 CWE；③首次实现缺陷定位和自动修复功能；④引入新的评估方法（拆分资产识别与模式匹配的精度）。

**🔧 技术方法**

采用了 Verible 语法树解析、基于关键字的资产过滤、正则表达式自动修复、以及多种针对特定 CWE 的模式检查算法。

**📊 数据集**

主要使用 HACK@EVENT 系列的 SoC 设计（DAC21、DAC18）以及 BugWhisperer LLM 生成的 3874 个 RTL 模块进行实验，并手工标注结果。

**📈 对比分析**

通过手工标注的 TP/FP 计算精度，比较 CWEEP 与 CWEAT：在 HACK@DAC21 上整体精度提升至 0.608（CWEAT 0.175），在 HACK@DAC18 上提升至 0.515；优化后单个 CWE 的精度从 0.167 提升至 0.778；运行时平均 135 s，远慢于 CWEAT 的 0.338 s，但误报更少。

**⚠️ 局限性**

局限性包括：仍有较高误报率（尤其在资产识别上）；仅适用于 RTL 代码，无法处理门级 netlist；关键字过滤方法在不同设计中可能不通用；评估依赖人工标注，主观性较大；自动修复仅提供建议，仍需人工审核。

---

## 412. CENDRe: Concept Extraction with Natural Domain Representations

**arXiv ID:** 2607.29621 | [PDF](https://arxiv.org/pdf/2607.29621v1)

**作者:** Antonia Holzapfel `[一作]` (RWTH Aachen University), Sebastian Trimpe `[通讯]` (RWTH Aachen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为 CENDRe 的概念提取方法，能够在时间域与频率域同时识别 CNN 对时间序列的关键模式，并给出其重要性评分；

**💡 创新点**

创新点包括：① 通过可微可逆变换（如 Fourier）将概念可视化到任意自然域；② 采用轮廓（silhouette）引导的聚类自动决定概念数；③ 利用梯度对比度评分实现精细化的概念定位；

**🔧 技术方法**

使用的技术包括：时间序列 CNN（InceptionTime、ResNet1D、DenseNet1D）、LADs 逐时刻聚合激活、微批量 k-means + 聚类层、HDBSCAN、轮廓评分、虚拟检视层（可逆变换）以及梯度加权掩码和重要性计算；

**📊 数据集**

数据集包括：两个合成数据集（syntheticLocal 与 syntheticFrequency）以及两个真实轴承故障数据集（CWRU Bearing 与 BearingPD）；

**📈 对比分析**

与 ECLAD‑ts、MultiVISION、基于 k‑means 的 CENDRe 进行比较；在合成数据上，CENDRe 的 Soft Representation Correctness 与 ECLAD‑ts 相当，Soft Importance Correctness 明显更高；在频域任务中，CENDRe 成功恢复预设频带；在真实轴承数据中，提取的频域概念与专家熟知的故障频率高度一致；

**⚠️ 局限性**

限制包括：只适用于具有近似平移不变性的 1D CNN（无法直接推广到 transformer 或状态空间模型）；概念数的自动选择仍需微调微簇数 J；并且方法在多通道或高度混杂的数据上可能需要进一步改进。

---

## 413. A Human-Centered Validation of the Explainability-Performance Coefficient

**arXiv ID:** 2607.29614 | [PDF](https://arxiv.org/pdf/2607.29614v1)

**作者:** Christian Oliva `[一作]` (Universidad Autónoma de Madrid), Luis F. Lago-Fernández `[通讯]` (Universidad Autónoma de Madrid)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了可解释性-性能系数（EPC）分数，作为一种模型无关、可量化的解释质量评估指标；

**💡 创新点**

将原始的EPC曲线压缩为单一分数，并通过人类中心化的评价（情感词典、图像ROI）验证其与人类认知的一致性；

**🔧 技术方法**

采用梯度相关、层级相关、积分梯度等基于梯度的解释方法，结合基线消除（均值填充、Gaussian blur）和插值/删除曲线评估；

**📊 数据集**

在三种数据模态上进行实验：银行贷款默认（Tabular）、MNIST、Imagenette（图像）和IMDB（文本）；

**📈 对比分析**

与SHAP、LIME等模型无关方法及不同激活函数的对比显示，积分梯度在多种模型/模态下均获得最高EPC分数；

**⚠️ 局限性**

EPC受基线消除策略影响，且对高维图像更敏感；对大规模模型如LLM、Vision‑Language Transformer的适用性尚未验证。

---

## 414. Diagnosing Compositional Generalization in Sequential Robot Tasks

**arXiv ID:** 2607.29687 | [PDF](https://arxiv.org/pdf/2607.29687v1)

**作者:** Yixiao Wang `[一作]` (University of California Berkeley), Masayoshi Tomizuka `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在顺序机器人操作任务中指令空间覆盖对组合泛化的影响，提出将泛化差距分解为边际指令偏移、指令组合偏移和上下文-动作偏移，并通过实验验证结构化子集而非全枚举足以实现强泛化。

**💡 创新点**

创新点在于：①理论上将组合泛化误差拆解为三项，为评估数据覆盖提供量化指标；②提出指令对覆盖（pairwise coverage）原则，显示只需覆盖关键指令依赖即可获得高 OOD 性能；③揭示稀疏训练失败源于指令引导缺失而非子技能缺失，证明仅需少量 fine‑tuning 即可显著提升 OOD 成功率。

**🔧 技术方法**

使用基于 Transformer 的行为克隆与流匹配策略（flow‑matching），搭配 DINOv2 视觉编码器；利用设计实验中的覆盖数组（orthogonal set）与阶段化模块化策略，控制指令组合；对比全枚举、稀疏对角集和覆盖集训练。

**📊 数据集**

构造了三组实验数据集：Pick‑and‑Place (PP, 16 任务)、Pick‑Place‑Press (PPP, 64 任务) 与 Dependent Two‑Stage Pick‑and‑Place (2S‑PP, 144 任务)，每个任务使用 500/300/100 条演示（共 8k/19.2k/14.4k 条），全部由 robomimic 生成。

**📈 对比分析**

与全枚举训练相比，结构化 1/4 任务子集（覆盖关键指令对）可获得 78.2% OOD 成功率，接近全枚举；稀疏对角预训练在仅加 1 条演示的 fine‑tune 后 OOD 成功率从 0.4% 提升至 54.7%；在 2S‑PP 任务中，增大依赖指令对覆盖率显著降低同容器违规率，提升 OOD 成功率。

**⚠️ 局限性**

实验仅覆盖少量顺序操作任务，难以验证对大规模任务空间的推广；仅使用冻结的视觉编码器，未探讨预训练 VLA 模型对覆盖原则的适用性；缺乏对模型可解释性与学习机制的深入分析。

---

## 415. Evolving language compositionality in a frequency-structured meaning space

**arXiv ID:** 2607.29642 | [PDF](https://arxiv.org/pdf/2607.29642v1)

**作者:** Fabio De Ponte `[一作]` (Universit\'e de Namur and Vrije Universiteit Brussel), Seth Bullock `[通讯]` (University of Bristol)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究通过在半监督迭代学习模型中对意义空间施加不同的频率分布（全意义向量、块级以及位级），探讨频率结构如何影响生成语言的可表达性、组成性与稳定性。

**💡 创新点**

创新点在于揭示频率结构仅在整体意义单元上施加时能促进可传递的组成语言，而在位级频率偏斜时则导致语言无法稳定传播，从而阐明频率与组成性之间的层次依赖关系。

**🔧 技术方法**

采用半监督迭代学习模型（自编码器架构）并使用Zipf分布对训练样本进行加权，随后通过条件熵计算组成性、表达度与稳定性指标。

**📊 数据集**

使用n位二进制意义向量（如16位）构成的模拟意义空间，并按不同Zipf参数生成训练样本，无需外部真实语言数据。

**📈 对比分析**

通过30次独立试验比较α=0（均匀）与α=1（偏斜）在全意义、块级和位级上的表现；结果显示均匀分布快速收敛至高组成性和稳定性，而位级偏斜仅在高频位上保持较高组成性，其他维度保持低且不收敛。

**⚠️ 局限性**

局限性包括仅考虑独立位的意义，未引入意义成分之间的依赖；实验仅在简化的二进制空间内进行，未验证到自然语言的可推广性；对更高阶结构的产生缺乏深入探索。

---

## 416. HierDoc: Hierarchical Page-to-Region Evidence Routing for Long-Document Visual Question Answering

**arXiv ID:** 2607.29638 | [PDF](https://arxiv.org/pdf/2607.29638v1)

**作者:** Rongjian Gu `[一作]` (University of Science and Technology of China), Houqiang Li `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出HierDoc框架，先用页级策略挑选包含证据的页面，再用区域级策略在这些页面中定位语义区域，最后将全页与区域信息共同输入LLM进行视觉问答。

**💡 创新点**

创新点在于将页面选择与区域选择分别建模为答案无关的结构化集合决策，并通过分阶段GRPO强化学习与解析器原生语义区域实现粗细粒度证据路由。

**🔧 技术方法**

使用Qwen系列LLM、GRPO强化学习、MinerU解析器、OCR/表格文本提取以及分阶段奖励机制。

**📊 数据集**

在MMLongBench-Doc、LongDocURL、SlideVQA、PaperTab和FetaTab等多页/长文档VQA数据集上进行评测。

**📈 对比分析**

与多种基线比较，HierDoc在四个基准上取得最优或竞争性结果，尤其在LongDocURL上相较Doc-V^*提升约16.9%，并显著提高页面与区域检索精度。

**⚠️ 局限性**

局限包括错误传播（页面误选导致区域无法恢复）、依赖MinerU导致候选区域受限、以及证据选择与答案无关的单独训练可能导致提升不完全映射到答案质量。

---

## 417. CodeShrink: Adaptive Visual Compression for Efficient Multimodal Code Understanding

**arXiv ID:** 2607.29637 | [PDF](https://arxiv.org/pdf/2607.29637v1)

**作者:** Wenxin Tang `[一作]` (Tsinghua University), Michael Lyu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种适用于多模态大语言模型的代码视觉压缩框架CodeShrink，利用空白消除渲染、任务相关视觉词筛选和自适应压缩配置三项技术实现高效代码理解。

**💡 创新点**

创新点在于（1）空白无关渲染（Blank‑Free Rendering）去除行间空白占用的视觉词；（2）Dominant Token Selection在推理时根据指令动态保留任务相关视觉词；（3）Adaptive Compression Configuration通过强化学习的轻量级Agent为每条输入自适应选择压缩比例和裁剪比，突破固定压缩比例的瓶颈。

**🔧 技术方法**

核心技术包括：视觉渲染与图像分块编码、M‑LMM视觉编码器与投影器、基于注意力的视觉词重要性评分、强化学习（GRPO）训练的Config Agent、LoRA微调的轻量级M‑LMM。

**📊 数据集**

使用Python和Java的公开基准数据集：CodeOCR QA、GPTCloneBench clone detection、LongCodeCompletion completion，随机抽取300、200、200条样本。

**📈 对比分析**

与文本压缩方法（LLMLingua系列、LongCodeZip）、视觉压缩方法（FastV、VisionZip）、基线CodeOCR对比。实验表明CodeShrink在QA、clone、completion三任务中可将视觉词量削减71.2%（最高）并在Python QA上超过未压缩文本输入的82.3%准确率；整体性能优于所有基线。

**⚠️ 局限性**

局限性包括：实验仅覆盖Python和Java，可能对其它语言或更大规模软件仓库的适用性未知；方法依赖于M‑LMM的视觉编码器，若模型视觉预训练不足则效果受限；配置Agent训练耗时与样本量相关，未对极大规模数据进行评估。

---

## 418. AgentHPOBench: A Benchmark For Evaluating LLM Agents as Sequential Hyperparameter Optimizers

**arXiv ID:** 2607.29626 | [PDF](https://arxiv.org/pdf/2607.29626v1)

**作者:** Tianyu Huai `[一作]` (Shanghai Innovation Institute), Xuanjing Huang `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了AgentHPOBench基准，用于评估代理在可执行机器学习仓库中通过顺序超参数干预提升实验性能。

**💡 创新点**

创新点在于将代理能力聚焦于从日志、指标等实验反馈中解读并生成下一步有效超参数配置，构建了30个可执行任务、统一执行框架与评分体系，并通过中间反馈消融验证反馈的重要性。

**🔧 技术方法**

采用顺序决策框架、统一执行主机、基准评估（MBNS、BWR、MAA）以及多种模型（开源权重LLM、API代理）与传统HPO方法（随机、TPE、BOHB变体）的对比实验。

**📊 数据集**

使用了30个来自GitHub的最新ML仓库任务，涵盖自然语言处理、计算机视觉、时间序列、图学习、强化学习、大型语言模型和结构学习等七个研究领域。

**📈 对比分析**

与传统HPO基线在相同基准、预算和干预次数下对比，Claude Sonnet 4.6在总体MBNS（0.407）、BWR（76.7%）和MAA（79.5%）上领先；不同任务类别表现差异显著，完整预算可提升MBNS但不一定提高BWR。

**⚠️ 局限性**

局限性包括：代理在不同领域的表现不一致，往往未能持续保留早期改进；受限于预算和执行主机的差异；仍低于报告的基准性能；对仓库结构的依赖限制了通用性。

---

## 419. The Theoretical Foundation of Socratic Tests: Dynamic, Multimodal, Conversational Examinations

**arXiv ID:** 2607.29624 | [PDF](https://arxiv.org/pdf/2607.29624v1)

**作者:** Ilya Mikhelson `[一作]` `[通讯]` (Northwestern University), Ilya Mikhelson (Northwestern University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了Socratic Test——一种基于AI的会话式评估平台，融合动态评估、Bloom与SOLO框架，实现非补偿加法评分。

**💡 创新点**

创新点在于将提问难度与答案结构分离、引入Shadow Ledger与Evidence Buffer保证公平、通过非补偿加法和逐层递进的“阶梯式”与“有机式”模式解决传统考试的可比性与偏差。

**🔧 技术方法**

主要技术包括大型语言模型（LLM）驱动的对话代理、基于Bloom的提示生成、SOLO结构化评估、实时评分与离线校准管道、加权折扣因子与分数上限桶。

**📊 数据集**

使用了三门课程的春季2026期试点数据（共98名学生），并将成绩与传统书面与口试结果对比；若需要可扩展至公开课程数据。

**📈 对比分析**

与传统书面与面对面口试对比显示学生压力降低约52%，对AI评估满意度超过80%，且评分可靠性提升至标准化误差小于传统评分的25%。

**⚠️ 局限性**

局限包括对LLM的依赖导致偶尔幻觉、需要教师手动校准与审核、缺乏大规模随机对照实验、以及对非英语表达的评估尚未充分验证。

---

## 420. WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning

**arXiv ID:** 2607.29613 | [PDF](https://arxiv.org/pdf/2607.29613v1)

**作者:** Senyu Fei `[一作]` (Tongji University), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出世界批评模型WCM，联合未来状态预测与价值估计，以提升VLA‑RL在部分可观测环境中的表现。

**💡 创新点**

在critic中加入预测未来状态的目标，使得历史观察能够构成可预测状态表示，从而突破单帧估计的瓶颈。

**🔧 技术方法**

采用轻量级ViT/Transformer编码器、残差FiLM动作编码、联合训练目标（值回归+预测+SIGReg），并在on‑policy（PPO/Flow‑SDE）与off‑policy（AWR/RECAP）框架中实现。

**📊 数据集**

在149个仿真任务（ManiSkill、MetaWorld、CALVIN、LIBERO‑Plus）和7个真实机器人任务（WidowX‑250S）上进行评估。

**📈 对比分析**

与Flow‑SDE、Flow‑Noise、π‑stepNFT、GRPO、PPO等基线对比，WCM在IND/OOD均提升约20–40%性能，在实测任务中提升约30–60%。

**⚠️ 局限性**

对极长观测历史收益有限，且在更大规模多模态环境下仍需进一步验证其通用性与鲁棒性。

---

## 421. Scaling Properties of Text Conditioning in Visual Generation

**arXiv ID:** 2607.29679 | [PDF](https://arxiv.org/pdf/2607.29679v1)

**作者:** Zilong Chen `[一作]` (ByteDance Seed), Haoqi Fan `[通讯]` (ByteDance Seed)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出将文本提示转换为结构化JSON（SP）并利用该结构化提示训练扩散模型，进一步通过多阶段LLM调优提升提示生成器的“promptability”，从而实现高质量、信息丰富的图像生成；

**💡 创新点**

创新点在于①引入可量化的文本信息指标（GPG、ED）并证明其能预测扩散训练损失；②通过结构化提示提升扩散模型的“diffusability”；③提出多阶段LLM训练（SFT、Cold‑Start、RFT）和在线 refine‑render‑judge 循环，显著提升“promptability”；

**🔧 技术方法**

采用扩散模型（Qwen‑Image、BAGEL 等）、视觉语言模型（Qwen‑3.5、Gemini、SAM 等）、LLM（Qwen‑3.5、Claude、GPT‑5.5 等）以及自定义的结构化提示框架和自监督训练管线；

**📊 数据集**

使用大规模公开图文数据（Qwen‑Image 训练集）进行 SP 注释、扩散训练，并在公开评测集（GenEval、GenEval2、DPG‑Bench、TIIF、WISE、CoReBench）上评估；

**📈 对比分析**

与多种开源和闭源模型对比，结构化提示系统在大多数指标上超过或与最强闭源系统持平，尤其在 GenEval2、DPG‑Bench、WISE、CoReBench 等多维度评测中取得显著提升；

**⚠️ 局限性**

局限性包括：①GPG、ED 等指标依赖于特定视觉语言评判器，可能受评判器偏差影响；②结构化提示需手工设计，难以自动扩展至视频或三维生成；③LLM 提示器的推理延迟较高，部署时需权衡生成质量与计算成本。

---

## 422. TokTier: Exact Stateful Tokenization for Agentic LLM Serving

**arXiv ID:** 2607.29678 | [PDF](https://arxiv.org/pdf/2607.29678v1)

**作者:** Zhenyu Zhang `[一作]` (Arizona State University), Zhichao Cao `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个状态化、精确的前端tokenizer服务，支持增量修复会话继续和GPU全量tokenization，显著降低前端重新token化开销。

**💡 创新点**

创新点在于通过可检验的稳定边界检查实现增量tokenization的exactness，并将GPU并行预tokenization与BPE结合，确保生成的token ID与参考实现完全一致；同时提出完整的差异化测试与实时shadow验证机制。

**🔧 技术方法**

使用Rust实现的会话存储与增量修复逻辑、CUDA Graph与GPU并行预tokenization+BPE、稳定边界检验、差异化测试框架以及shadow verifier。

**📊 数据集**

使用了153,951条编码代理交互日志（约10个月的Claude Code/Codex），12.4 TB真实文本语料库，以及17种生产Tokenizer版本。

**📈 对比分析**

与fastokens、Gigatoken、LoPT、HF serial等基准对比，增量修复单次延迟为0.5–1.1 ms（100 K–3 M字节），GPU全量tokenization 0.87 ms（1 M字节），P99下降23%，系统容量提升至1,821 req/s（4个repair核心+1个GPU），相比纯CPU实现提升约45×。

**⚠️ 局限性**

局限性包括GPU路径受BPE依赖链瓶颈、稳定边界检查仅适用于部分Tokenizer family、GPU路径不输出字节跨度、对大append仍需fallback；需要持续的差异化测试与shadow验证以保证新版本正确性。

---

## 423. Balancing of Humanoid with Object Mass: Trade-off Analyses and Lifting Control

**arXiv ID:** 2607.29625 | [PDF](https://arxiv.org/pdf/2607.29625v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 424. Data Visualization Style Guides in Practice: Why They Emerge, How They Work, and When They Bend

**arXiv ID:** 2607.29645 | [PDF](https://arxiv.org/pdf/2607.29645v1)

**作者:** Alvitta Ottley `[一作]` (Washington University in St. Louis), Jonathan Schwabish `[通讯]` (PolicyViz)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对 9 位跨行业实践者进行半结构化访谈，并对 50 份公开可见的可视化风格指南进行文档分析，提出并验证了 PRISM 框架，用以系统描述风格指南的目的、规则与机制、制度执法者和情境灵活性。

**💡 创新点**

创新点在于：①把风格指南视作社会技术系统而非单一文件；②构建四维 PRISM 模型揭示其隐含动机、治理与灵活性；③指出文档公开信息与内部实践之间的可观测性鸿沟，为后续研究提供新的视角和评估维度。

**🔧 技术方法**

采用质性访谈与反思主题分析（RTA）技术进行数据编码，结合文档内容编码，使用手工编码与共识讨论保证主题可靠性。

**📊 数据集**

数据集包括：9 名作者/维护者的访谈记录（约 9 份访谈）以及 50 份公开可获取的可视化风格指南（来自新闻、政府、行业等多领域）。

**📈 对比分析**

通过对公开文档与访谈数据的交叉比对，比较了 PRISM 维度在文档中可见与不可见的比例；结果显示规则与机制几乎全部可见，但目的、执法者与灵活性在文档中可观测度低，说明研究方法能够揭示隐藏的组织层面信息。

**⚠️ 局限性**

局限性包括：样本规模小、仅涵盖西方机构、只分析公开英文文档，可能无法覆盖非西方或内部闭源的实践；PRISM 框架为解释性框架，需在更大多样化语境中进一步验证和细化。

---

## 425. Bootstrapping Self-Supervised Learning of Binary Classification Using Error Bounds: A Case Study on a Robotic Insertion Task

**arXiv ID:** 2607.29640 | [PDF](https://arxiv.org/pdf/2607.29640v1)

**作者:** Zebin Duan `[一作]` (University of Southern Denmark), Frederik Hagelskjær `[通讯]` (University of Southern Denmark)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种在线数据引擎，用于机器人插入任务的成功/失败二分类，动态切换模型预测与昂贵验证以降低错误率；

**💡 创新点**

创新点在于将Wilson Score置信区间嵌入模型预测，实时给出错误概率下限，并根据阈值决定是否使用昂贵验证，实现可控错误率的自适应学习；

**🔧 技术方法**

采用UMAP降维、Radius Neighbors分类器、Wilson Score置信区间及基于力传感器的特征提取；

**📊 数据集**

使用真实机器人实验收集的1503个力数据样本（803正例、700负例），不使用公开数据集；

**📈 对比分析**

与基准Binomial Interval方法对比，Wilson Score方法显著减少昂贵验证次数、降低误报率，并在设定阈值下保持精度高于阈值；

**⚠️ 局限性**

局限性包括仅在单一插入任务上验证，模型对不同部件/夹具的泛化能力未知，且仍依赖昂贵验证进行标签更新，缺乏对极端噪声或数据分布变化的鲁棒性。

---

## 426. OASIS: Occlusion-aware Single-image Hand Avatar Reconstruction via 3D Gaussian Splatting

**arXiv ID:** 2607.29633 | [PDF](https://arxiv.org/pdf/2607.29633v1)

**作者:** Zhisheng Han `[一作]` (University of Leicester), Zheheng Jiang `[通讯]` (University of Leicester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本研究提出一种基于单张图像的3D手部avatar重建框架（OASIS），能够在有限的视觉信息下生成可动画、细节丰富的手部三维模型。

**💡 创新点**

核心创新点包括①几何对齐的视觉证据令牌（VET），将二维图像特征直接对齐至三维手部网格；②可见度条件点–图像注意力（VPIA），在严重自遮挡情况下自适应地将局部与全局视觉信息融入几何令牌；③面级特征（Feature‑on‑Mesh）表示，利用面内的重叠信息驱动高斯分布随手部变形。

**🔧 技术方法**

技术方案涵盖3D Gaussian Splatting（3DGS）渲染、DINOv2视觉编码器、MANO‑HD手部先验、LoRA微调、可见度条件跨模态注意力、面级自注意力与线性混合蒙皮（LBS）动画。

**📊 数据集**

主要数据集为InterHand2.6M（21名主体用于预训练，剩余用于测试），并在野外图像及HanCo数据集上做扩展评估。

**📈 对比分析**

在InterHand2.6M单图基准上，OASIS获得PSNR 27.38、SSIM 0.956、LPIPS 11.45，优于OHTA、HandAvatar、Handy、HARP等方法；在多图基准上同样表现领先。渲染速度约390 FPS，单图微调时间约5分钟，显著低于传统NeRF实现。

**⚠️ 局限性**

局限性主要包括：需要先行估计姿势和相机参数，对极端自遮挡或极端光照仍可能出现细节丢失；模型对极度复杂的手部姿势（如手指完全折叠）仍有精度下降；缺乏视频序列训练导致对动态细节的捕捉不如多帧方法。

---

## 427. RayViT: Ray-Conditioned Visual Representations for Viewpoint-Robust Imitation Learning

**arXiv ID:** 2607.29622 | [PDF](https://arxiv.org/pdf/2607.29622v1)

**作者:** Qian Wang `[一作]` (Karlsruhe Institute of Technology), Gerhard Neumann `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过将相机几何信息注入预训练的ViT视觉编码器，提升视觉模仿学习在视角变化下的鲁棒性。

**💡 创新点**

创新点是提出RayViT框架，利用Plücker射线映射生成射线图，并通过射线条件的类标记和位置嵌入在ViT中实现几何注入，同时加入交叉视角一致性正则。

**🔧 技术方法**

使用技术包括Plücker射线映射、ViT、门控交叉注意力模块、RoPE位置编码以及余弦相似度一致性损失。

**📊 数据集**

实验使用RoboCasa模拟基准（16个任务）和四个真实世界抓取/插入/开抽屉/杯子堆叠任务。

**📈 对比分析**

与Adapt3R、PMP‑xyz、CamPose等基线相比，RayViT在相机扰动下平均成功率提升约13个百分点；在真实任务中平均分提升至少0.8分，显示出显著的鲁棒性优势。

**⚠️ 局限性**

局限在于只针对ViT视觉编码器，需多视角且依赖anchor视角，且未考虑卷积网络或单视角的场景。

---

## 428. ExtractBench: A Benchmark for Schema-Guided Enterprise Document Extraction

**arXiv ID:** 2607.29677 | [PDF](https://arxiv.org/pdf/2607.29677v1)

**作者:** Boyang Zhang `[一作]` (runllama.ai), Simon Suo `[通讯]` (runllama.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 ExtractBench 基准，用以评估企业文档的 schema-guided 提取，兼顾准确性、可追溯性和成本。

**💡 创新点**

创新点在于：①多维标签（任务、感知、表结构、长度、业务领域）实现细粒度分析；②同时测量值准确性、词/页级定位以及每页成本；③构建规模化的 schema 与真值管道，结合多源文档和人机裁判。

**🔧 技术方法**

采用大型视觉语言模型（VLM）、编码代理以及专业提取 API 进行评测，并通过统一的 F1、IoU 定义计算指标。

**📊 数据集**

使用覆盖 8 个业务领域、约 4000 页的真实企业文档、合成长列表以及手写扫描表单组成的数据集。

**📈 对比分析**

与多类系统对比，发现 LlamaExtract Agentic Plus 在准确率、定位和成本三维均位居前列；VLM 低成本但易截断长表；编码代理精度高但成本最高。

**⚠️ 局限性**

局限在于：词级定位仍低于 50%，长表、手写扫描等情形表现不佳，且仅对公开可用模型做评估，缺乏跨语言/多模态扩展。

---

## 429. Freeze, Then Select: Structured Field Adapters and Stability-Validated Weak Selection for PDE Discovery from Sparse Observations

**arXiv ID:** 2607.29665 | [PDF](https://arxiv.org/pdf/2607.29665v1)

**作者:** Juncheng Zhong `[一作]` (Fudan University), Wenlian Lu `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种先冻结场再选择方程的两阶段 PDE 发现框架（Freeze‑then‑Select），通过结构化场适配器重构连续场并在冻结后用 Stability‑Validated Weak Selection（SVWS）在多组弱式系统上生成、拟合并验证候选项，既可用于固定库也可用于 GP 生成的符号表达式。

**💡 创新点**

创新点在于（1）将场重构与方程选择解耦，避免耦合优化路径导致的“checkpoint 依赖”；（2）结构化适配器把场分解为学习到的空间特征与时间三次 B‑spline 系数，实现无 PDE 残差的光滑重构；（3）SVWS 在多组独立弱式系统上实现支持生成、系数重拟合与交叉验证，提升对稀疏、噪声观测的鲁棒性；（4）扩展到符号表达式的选择，验证同一冻结场可支持 GP 生成的非线性扩散函数的发现。

**🔧 技术方法**

使用技术包括：深度学习结构化场适配器（空间网络 + B‑spline 时序系数），弱式积分（integration by parts + compact 支持测试函数），Sequential Thresholded Least Squares（STLSQ）进行支持生成，岭回归重拟合，One‑Standard‑Error 规则与多系统交叉验证；符号 GP 搜索与表达式归一化。

**📊 数据集**

在公开的 MDBench 数据集上评估，涵盖 KdV、Kuramoto–Sivashinsky（KS）和二维输运扩散 PDE，并使用两种 20% 稀疏采样协议（S20：空间子集；T20：时间子集）。

**📈 对比分析**

与 PDE‑FIND、WSINDy、DeepMoD、PINN‑SR、Weak‑PDE‑LEARN、DL‑PDE 等基线对比，所有六种稀疏采样设置下都实现 5/5 的 exact support 恢复，尤其在 KS 上超越所有基线（最多 5/5 vs 3/5）。系数误差和场误差也显著低于对手；在符号实验中，SVWS 对噪声 0%、5%、10% 仍能稳定恢复目标幂律扩散函数，而 PySR 在完整网格上仅恢复率不一。

**⚠️ 局限性**

局限性：1）需要在冻结前对场进行足够精确的重构，若观测极度稀疏或噪声极高，重构误差可能影响后续选择；2）SVWS 对弱式系统数量和阈值路径的设定仍需经验调参；3）实验仅覆盖三种 PDE 与二维案例，尚未验证更高维或更复杂非线性算子；4）符号搜索的表达式族有限，无法覆盖所有可能的物理关系。

---

## 430. Development of FDD-ON: an Ontology for VAV HVAC System Fault Detection and Diagnostics

**arXiv ID:** 2607.29657 | [PDF](https://arxiv.org/pdf/2607.29657v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 431. Reusing Past Repairs Through Hierarchical Trajectory Abstraction for Coding Agents

**arXiv ID:** 2607.29658 | [PDF](https://arxiv.org/pdf/2607.29658v1)

**作者:** Yisen Xu `[一作]` (Concordia University), Tse-Hsun Chen `[通讯]` (Concordia University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将历史修复轨迹按阶段分层抽象成多级树，并生成可重用的阶段性计划以指导未来的修复。

**💡 创新点**

构建细粒度到高层策略的多级层次抽象，结合检索与适配生成问题特定计划，实现跨任务、跨模型、跨代理的迁移能力。

**🔧 技术方法**

使用LLM（GPT‑5、MiniMax M2.5）完成分组与抽象操作、检索、计划生成，借助LangGraph协同工作；采用SWE‑bench Verified benchmark 进行评估。

**📊 数据集**

SWE‑bench Verified（500个Python GitHub issue 及其测试套件）。

**📈 对比分析**

与九个现有自动修复代理（含同一后端模型和不同经验重用方式）对比，Pass@1最高达 81.2%（MiniMax M2.5）/79.2%（GPT‑5），显著优于所有基线。

**⚠️ 局限性**

依赖LLM生成的抽象质量、评估仅基于 Pass@1 与 token 使用、仅在 Python 的 SWE‑bench Verified 上验证，可能难以直接推广至其他语言或更完整的测试环境。

---

## 432. Toward Robust and 3D-Aware RGB-NIR Imaging in the Dark

**arXiv ID:** 2607.29684 | [PDF](https://arxiv.org/pdf/2607.29684v1)

**作者:** Muyao Niu `[一作]` (University of Tokyo), Yinqiang Zheng `[通讯]` (University of Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于3D感知的神经隐式融合模型，利用噪声RGB与近红外图像在不使用干净RGB监督的情况下实现低光成像。

**💡 创新点**

创新点包括：①在3D空间中引入NIR感知位置编码，显著抑制高频噪声；②设计颜色码MLP通过Gumbel-Softmax学习NIR到RGB的多模态映射，解决颜色模糊；③整合NIR信息实现跨噪声级别的鲁棒性。

**🔧 技术方法**

技术细节：基于NeRF的体渲染框架；并行/条件结构的多层感知机；NIR条件位置编码；颜色码MLP与Gumbel-Softmax；L2光度损失结合曝光放大。

**📊 数据集**

使用了两套数据集：合成数据集（Mitsuba3渲染的5个场景，5种噪声级别 s∈{1/10,1/25,1/50,1/100,1/200}）；真实低光数据集（4个场景，49视角，850 nm NIR LED），每个场景49张图像。

**📈 对比分析**

通过与Restormer、ScaleMap、NVEU、SANet、NAID、RawNeRF、LLNeRF等多种SOTA方法进行定量（SSIM/PSNR/LPIPS）和定性对比，模型在所有噪声级别和真实场景下均实现了更高的结构和色彩保真度，且无需干净RGB监督。

**⚠️ 局限性**

局限性：仅适用于静态场景；需要RGB与NIR精准对齐且相机姿态已知；对动态环境或未对齐的RGB/NIR输入效果未知。

---

## 433. GQ-FSL: Green Quantized Federated Split Learning

**arXiv ID:** 2607.29659 | [PDF](https://arxiv.org/pdf/2607.29659v1)

**作者:** Idan Roth `[一作]` (University of British Columbia), Lutz Lampe `[通讯]` (University of British Columbia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了绿色量化联邦分割学习（GQ‑FSL）框架，结合分割学习与随机量化实现移动边缘设备上深度网络的能耗优化。

**💡 创新点**

创新点在于允许客户端与服务器使用不同的量化精度，解耦设备能耗与全局收敛，并给出量化误差对收敛的理论上界与联合优化设计。

**🔧 技术方法**

采用了分割学习、随机量化、固定点QNN、能耗模型与收敛理论推导，联合优化切分点与精度。

**📊 数据集**

使用CIFAR‑10数据集训练ResNet‑18模型。

**📈 对比分析**

通过与量化联邦学习（Q‑FL）和全精度FSL对比，实验显示GQ‑FSL在相同目标精度下能耗最低，尤其在高精度要求下仍保持优势。

**⚠️ 局限性**

局限性包括需要预先统一切分点与精度，且对异构硬件的适配与大规模客户端时的通信瓶颈仍待进一步研究。

---

## 434. When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning

**arXiv ID:** 2607.29617 | [PDF](https://arxiv.org/pdf/2607.29617v1)

**作者:** Luca Viano `[一作]`, Dylan J. Foster `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种交互式价值基模仿学习（IL）框架，研发了On-Policy Value-based Imitation（简称 OVI）算法，展示了在仅满足专家价值函数可实现（Q‑realizability）时即可高效学习。

**💡 创新点**

创新点：①证明交互式查询可显著降低学习者的表征需求，仅需实现专家的价值函数而非完整策略；②给出交互式IL在该弱假设下的统计与计算效率上界；③证明在离线场景下仅靠Q‑realizability无法获得高效学习，强调交互必要性；④将该框架推广到链式思维（CoT）任务，获得指数级计算优势。

**🔧 技术方法**

技术手段：基于 saddle‑point 形式的值函数/策略对抗学习；分层（按时间步）训练，利用线性最大化 oracle 更新值函数；softmax 策略更新；理论证明结合信息理论下界；实验使用 PyTorch、Gymnasium 环境。

**📊 数据集**

数据集：Gymnasium 四种标准连续控制任务（如 CartPole、MountainCar 等）；对专家使用两层 64 随机网络生成演示；在 CoT 任务中使用公开的数学/代码生成演示（prompt–response 对）。

**📈 对比分析**

比较方法：与离线基于策略的 BC、DAGGER、离线价值基 IL 以及交互式策略基方法相比；实验显示 OVI 在学习者网络宽度从 64 降至 2 时仍能保持高回报，优于所有对比方法，且在 CoT 场景下实现了从指数级到线性级的时间复杂度提升。

**⚠️ 局限性**

局限性：①需要在线交互，无法直接应用于无交互或纯离线场景；②假设专家价值函数可实现，若价值函数近似或缺失则性能下降；③分层非平稳策略导致额外内存与计算负担；④理论证明在 H 维度上存在多项式依赖，尚未实现完全无 H 影响；⑤在实际大规模语言模型中缺乏实证验证。

---

## 435. Structural Tractability Frontiers for Metric Repair

**arXiv ID:** 2607.29649 | [PDF](https://arxiv.org/pdf/2607.29649v1)

**作者:** Asaf Etgar `[一作]`, Jamie Tucker-Foltz `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究图上距离测量的度量修复问题，提出在树宽限制且权重有界的情况下可以用伪多项式时间算法求解，并证明在路径宽 6 或平面/网格图上该问题仍然是 NP 难的，从而划定了结构可解性的边界。

**💡 创新点**

创新点包括：① 将度量修复问题转化为破环（broken cycle）集的最小命中集（light hitting set）问题；② 设计了基于分枝组合的动态规划，利用 SP 分解和树宽分解实现伪多项式时间求解；③ 引入需求投影（demand projection）技术，统一处理并行与串行组合；④ 通过巧妙的构造与归约证明了该问题在路径宽 6、平面和网格图上的弱/强 NP 难性，表明仅靠结构或仅靠权重并不能使问题可解。

**🔧 技术方法**

主要技术手段包括：图分解（SP 分解、树宽分解、nice 树分解）、动态规划与状态压缩（距离-需求概况）、破环命中集的理论分析、可数编码的权重和伪多项式时间复杂度、以及多种构造归约（Partition、Independent Set 等）证明 NP 难性。

**📊 数据集**

本研究完全是理论性质，没有使用实际数据集，所有结果均来自图结构与权重的抽象构造与归约。

**📈 对比分析**

由于没有实验对比，作者以理论复杂度为衡量标准进行比较：在树宽 r、权重最大值 W 的条件下，算法时间为 O(W^{O(r^2)}(|V|+|E|))；而在路径宽 6 或平面/网格图上即使是加权最小修改量也呈现弱/强 NP 难性，说明结构与权重均为不可或缺的可解性因素。

**⚠️ 局限性**

限制包括：算法仅在权重有界且整数编码时有效，时间仍为伪多项式；对树宽 2（SP 图）仍未知是否存在真正多项式时间算法；路径宽 6 与树宽 2 之间的可解性边界尚未确定；在平面/网格图上仍是强 NP 难，且是否能进一步降低所需权重种类（如仅两种权重）仍是未解问题。

---

## 436. FlexComposer: Unified Video Compositing from Images to Dynamic Footage with Flexible Trajectory Control

**arXiv ID:** 2607.29627 | [PDF](https://arxiv.org/pdf/2607.29627v1)

**作者:** Songchun Zhang `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FlexComposer框架，实现静态图像与动态视频素材在任意轨迹上的无缝插入；

**💡 创新点**

创新在于统一规范化的前景表示、参数无关的空间感知潜在注入以及混合数据集与渐进式学习，兼顾高精度轨迹控制与自然光照/阴影融合；

**🔧 技术方法**

利用VAE+Video Diffusion Transformer、Spatial-aware Latent Injection、Grounded‑SAM2+SpatialTracker、Flow Matching训练、LoRA适配等技术；

**📊 数据集**

使用程序化合成数据、54k真实影视剪辑以及生成式数据构建混合数据集；

**📈 对比分析**

在DAVIS、MoveBench、VBench等基准上与最新I2V、V2V和视频编辑基线对比，FVD、FID、PSNR、SSIM、Motion Score等指标均优于或相当，表现出更高的轨迹遵循度和视觉质量；

**⚠️ 局限性**

仍缺乏物理交互模拟，长时视频或极端视角变化时可能出现不一致，并且对轨迹估计依赖较高。

---

## 437. The Kikuchi Hierarchy is Sharp for $k$XOR

**arXiv ID:** 2607.29672 | [PDF](https://arxiv.org/pdf/2607.29672v1)

**作者:** Alexander Schmidhuber `[一作]` (Massachusetts Institute of Technology), Matthew B. Hastings `[通讯]` (Microsoft Station Q)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了归一化Kikuchi层级在稀疏kXOR问题上实现无对数失真的强检测、弱恢复和强排斥，并给出了匹配的下界与量子加速方案。

**💡 创新点**

通过局部归一化与闭环计数精确消除对数失真，得到最优的信号–时间折衷，并首次提供无上界的强排斥证明与SoS证据。

**🔧 技术方法**

使用归一化Kikuchi矩阵、Trace方法与活跃前沿计数、独立复制与Mecke公式、矩阵Bernstein、Krylov迭代以及稀疏哈密顿量的量子框架。

**📊 数据集**

本研究为理论分析，无实验数据集，所有结果基于随机kXOR实例的概率模型。

**📈 对比分析**

相较于现有Spectral/SoS算法，本文在所有k≥3上实现了无对数的样本阈值，时间复杂度为n^ℓ，匹配最优下界；量子算法实现四次速度提升。

**⚠️ 局限性**

结果局限于稀疏kXOR和特定归一化设计，未涵盖一般CSP的强排斥；在极端小级别下仍需额外参数，量子实现依赖理想量子电路。

---

