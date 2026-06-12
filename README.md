# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-12 | 今日论文总数: 589

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. LongSpike: Fractional Order Spiking State Space Models for Efficient Long Sequence Learning

**arXiv ID:** 2606.12895 | [PDF](https://arxiv.org/pdf/2606.12895v1)

**作者:** Xinrui He `[一作]` (Wuhan University), Zheng-Jun Zha `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 LongSpike 框架，将分数阶状态空间模型 (f‑SSM) 融入脉冲神经网络，解决传统一阶 ODE 的记忆瓶颈。

**💡 创新点**

创新点包括：① 用 f‑SSM 生成长记忆的幂律核；② 将其转化为可并行的 Sum‑of‑Exponentials (SOE) 近似；③ 结合 LIF 神经元与 Surrogate Dynamic Network (SDN) 实现全并行脉冲预测。

**🔧 技术方法**

技术手段包括：分数阶微积分（Caputo 导数）、状态空间建模、SOE 近似、FFT 卷积、CNN 替代 SDN、Surrogate Gradient 训练，以及长序列 SSM 结构。

**📊 数据集**

使用的数据集：Long Range Arena (多模态长序列任务)、WikiText‑103 语言建模、Speech Commands 语音分类。

**📈 对比分析**

与 Transformer、S4、S4D、SpikingSSM、SpikingLMUFormer 等方法对比；在 LRA 6 任务上平均准确率约 86%（最高 95.41%），在 WikiText‑103 的 perplexity 32.31（低于 33.94），在 Speech Commands 的准确率 96.31%（高于 96.09%）。整体性能显著优于现有 SNN 与 ANN 基线。

**⚠️ 局限性**

局限性：尚未在专用 neuromorphic 硬件（如 Loihi）上部署；SOE 近似使用的指数项较少，扩展到更大 M 的效果待验证；能耗与硬件实现的真实评估仍需进一步研究。

---

## 2. Topical Phase Transitions in Artificial Intelligence Research: Large-Scale Evidence and an Early-Warning Signature for Emerging Topics

**arXiv ID:** 2606.12828 | [PDF](https://arxiv.org/pdf/2606.12828v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 3. From Real-World Projects to Research-Oriented Learning: Continuous Improvement of a Master-Level Course in Software Engineering Education

**arXiv ID:** 2606.12438 | [PDF](https://arxiv.org/pdf/2606.12438v1)

**作者:** Michael Neumann `[一作]` (University of Applied Sciences and Arts Hannover), Eva-Maria Schön `[通讯]` (University of Applied Sciences Emden/Leer)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对2019-2025年期间一门硕士级软件工程课程的纵向混合方法研究，系统追踪课程从实践导向项目向研究导向学习环境的持续演进及其对学生感知的影响。

**💡 创新点**

创新点在于首次从多年的纵向视角展示课程演进如何保持甚至提升学生评价，并识别出支持这种演进的关键设计要素（真实性、外部合作、教师支持、结构化 scaffolding 等）。

**🔧 技术方法**

采用纵向混合方法（量化教学评价量表与定性问卷/访谈分析），并结合课程文档与教学反思资料进行三角验证。

**📊 数据集**

数据集包括 2019、2021-2023、2025 年的教学评价问卷（共 5 个时间点，回收率逐年下降）、课程文档（项目记录、方法说明等）以及学生开放式反馈。

**📈 对比分析**

通过对相同核心评价维度的年际比较与对比，发现学生整体课程质量评价保持正面且无显著下滑，表明课程在提升研究深度的同时未削弱学生满意度；若与传统仅强调实践的课程做对照，显示更高的研究导向与更稳定的正向评价。

**⚠️ 局限性**

局限性包括仅单一高校单门课程、后期样本量较小、评估工具随时间演变导致可比性受限，以及缺乏客观学习成效与科研产出之外的多元评价指标。

---

## 4. Dual-State Slot Attention: Decoupling Appearance and Identity for Video Object-Centric Learning

**arXiv ID:** 2606.12601 | [PDF](https://arxiv.org/pdf/2606.12601v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 5. AfriSUD: A Dependency Treebank Collection for Evaluating Models on African Languages

**arXiv ID:** 2606.12708 | [PDF](https://arxiv.org/pdf/2606.12708v1)

**作者:** Happy Buzaaba `[一作]` (Princeton University), Christiane Fellbaum `[通讯]` (Princeton University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了AfriSUD大型SUD树库，覆盖九种非洲语言，并提供标注流程、基线模型及LLM评估。

**💡 创新点**

创新点在于首次大规模、社区驱动的SUD树库，覆盖语言家族多样性，并对LLM和跨语言迁移进行了系统评估。

**🔧 技术方法**

采用SUD框架、Arborator标注工具、Stanza、mBERT/XLM-R/AfriBERTa/AutoXLMR等多模态预训练模型，以及Gemini/GPT/Gemma等LLM进行提示式评估。

**📊 数据集**

使用AfriSUD九语言树库（含Hausa、Naija、Wolof、Yorùbá、Efik、Swahili、Kinyarwanda、Runyankore、isiXhosa），源自MasakhaPOS及其他公开语料。

**📈 对比分析**

通过UPOS、UAS、LAS指标对Stanza、各类多语种编码器、LLM提示与监督微调等方法进行对比，结果显示Africa-centric编码器在POS上优于通用模型，Stanza在LAS上领先；LLM在零样本时性能最低，但通过few-shot显著提升。

**⚠️ 局限性**

主要局限包括标注过程中的形态约束、歧义及SUD关系适用性挑战、LLM实验受限于模型可用性、数据规模不足导致某些关系评估不充分。

---

## 6. SoK: The Constant Time Model

**arXiv ID:** 2606.13000 | [PDF](https://arxiv.org/pdf/2606.13000v1)

**作者:** Billy Bob Brumley `[一作]` `[通讯]` (Rochester Institute of Technology), Billy Bob Brumley (Rochester Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文系统化了常量时间模型的演进，识别了规范层与实现层的安全缺口，并提出一种通用的攻击方法来发现此类漏洞，随后将其应用于椭圆曲线私钥加载环节。

**💡 创新点**

创新点在于构建了跨规范的安全攻击框架，并通过实验验证BoringSSL在私钥长度泄露方面比OpenSSL更易被利用，揭示了更强威胁模型下可能出现的更大泄漏。

**🔧 技术方法**

采用统计信息学手段（互信息、Cohen’s d、贝叶斯采样）与精确计时测量技术，对实现路径中的时间差异进行定量分析，并使用多线程时序hedge来提升抵御能力。

**📊 数据集**

使用了包含33类不同位长（从128到384位）的P‑384椭圆曲线私钥数据集，对每类键做了1,000次重复加载测试。

**📈 对比分析**

通过比较互信息、Cohen’s d以及采样所需次数，实验显示BoringSSL原版泄漏信号强度约为OpenSSL的200倍，改为hedge后泄漏几乎消失，性能方面BoringSSL在hedge模式下的负载时间显著增加。

**⚠️ 局限性**

局限性包括仅针对私钥加载路径的实验，未验证对实际攻击场景的可行性；方法依赖特定硬件与编译器环境，可能不适用于所有平台；且未覆盖其他协议层面可能的时序泄漏。

---

## 7. The Hidden Power of Scaling Factor in LoRA Optimization

**arXiv ID:** 2606.12883 | [PDF](https://arxiv.org/pdf/2606.12883v1)

**作者:** Zicheng Zhang `[一作]` (JD), Qixia Jiang `[通讯]` (JD)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在LoRA参数高效微调中，研究者分析了缩放因子α与学习率的优化差异，并提出LoRA-α框架。

**💡 创新点**

创新点是将α从次要调节量提升为主导优化驱动，提出Signal-Drift理论和α∝√r平方根标度法。

**🔧 技术方法**

使用Signal-Drift分解理论、经验超参数搜索、Adam优化、低秩矩阵分解等技术。

**📊 数据集**

数据集覆盖LLM微调（Llama、Qwen、DeBERTa）、GLUE、SST-2、NLG任务（GSM8K、MATH、HumanEval）、多模态（Flux.1、Flux.1-12B）等。

**📈 对比分析**

通过与标准LoRA、RsLoRA、LoRA+、PiSSA、LoRAM以及FFT对比，LoRA-α在大部分任务上提升3–8%点，且在多任务上接近FFT性能。

**⚠️ 局限性**

局限在于α基值的经验选择仍需针对不同模型宽度调优，且在极端大rank或极低学习率下仍可能出现漂移。

---

## 8. X-MADAM-RAG: Diagnosing and Handling Chinese-English Evidence Conflict in Retrieval-Augmented Generation

**arXiv ID:** 2606.12903 | [PDF](https://arxiv.org/pdf/2606.12903v1)

**作者:** Yongqi Kang `[一作]` (Sichuan University), Yong Zhao `[通讯]` (Sichuan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对中英证据冲突的检验框架X-RAMDocs-ZHEN，并设计了可解释的X-MADAM-RAG流水线以处理检索增强生成中的多语言证据冲突

**💡 创新点**

在多语言（中英）证据冲突的诊断任务中首次构造了六种平衡的证据条件，并将提取、修复、分组和冲突感知聚合拆解为可审计的子模块

**🔧 技术方法**

采用检索增强生成、基于规则的表面模式提取、可视化证据修复、确定性候选分组及冲突感知聚合等技术；评估使用Qwen2.5-7B-Instruct模型

**📊 数据集**

使用从RAMDocs衍生的受控样本构成的X-RAMDocs-ZHEN v0.1（300例，包含中英证据）以及100例无模板的自然化压力测试

**📈 对比分析**

在原始受控基准上，X-MADAM-RAG严格准确率0.9667，冲突感知成功率0.9767，明显优于单调用LLM基线；但在自然化压力测试中准确率仅0.3000，说明提取阶段是瓶颈

**⚠️ 局限性**

受控模板依赖导致原始基准不具备自然语料的泛化能力；提取模块对语句变体高度敏感；仅针对中英，未验证更大模型或其他语言，且所有评估基于自动词汇指标，无人工语义审计

---

## 9. Charge as a Construct-Validity Factor in Chinese Legal Case Retrieval: A Cross-Benchmark Audit

**arXiv ID:** 2606.12993 | [PDF](https://arxiv.org/pdf/2606.12993v1)

**作者:** Yao Liu `[一作]` (Chengdu University of Technology), Zhilan Liu `[通讯]` (Chengdu University of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对中文法律案例检索（LCR）基准进行审计，评估其性能主要是由罪名匹配还是法律推理驱动，并提出了基于罪名控制的评估协议（CCE），为后续基准提供可复现的检查工具。

**💡 创新点**

创新点在于揭示了大部分现有系统与BM25的性能差距可被简单的罪名匹配规则几乎完全恢复，说明基准存在构造效度缺陷；同时首次构建了多维度的罪名控制评估框架（CCE），实现了对“罪名作为锚点”与“罪名作为标签泄漏”的系统性判定。

**🔧 技术方法**

采用模型无关的罪名重叠oracle、宏AUC构造探测、预测罪名级联、罪名分层与反事实屏蔽测试，并通过群集自举检验显著性；此外提供了完整的评估脚本、JSON schema和示例报告，形成可复现的评估平台。

**📊 数据集**

实验使用了三套公开中文法律案例检索基准：LeCaRDv2、LeCaRDv1 和 CAIL2022 阶段二，均包含分级相关性标签。

**📈 对比分析**

与BM25、KELLER、Qwen3-8B-Reranker、SAILER、BGE-M3、RoBERTa 等系统对比，罪名-only oracle 在 LeCaRDv2 上恢复了约99.2% 的 BM25 与最佳系统的差距，说明绝大部分优势来自罪名匹配；在同一基准中，仅剩约0.026 的在罪名内差异是系统真正的法律推理信号。

**⚠️ 局限性**

局限性包括：仅覆盖三套基准；CAIL2022 的样本量小，统计功效有限；多种系统已在训练集中接触到基准数据，导致系统层面的结论需谨慎；研究聚焦于基准层面的构造效度评估，而非单一模型的内部机制。

---

## 10. Detecting Functional Memorization in Code Language Models

**arXiv ID:** 2606.12764 | [PDF](https://arxiv.org/pdf/2606.12764v1)

**作者:** Matthieu Meeus `[一作]` (Meta), Luca Melis `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了代码大型语言模型在训练数据中的功能性记忆现象，提出了基于对照模型的功能性泄露评估方法，并使用多种文本、结构和执行层面的相似度指标来检测模型是否在无明显文本相似的情况下复制了训练函数的逻辑。

**💡 创新点**

创新点包括：①将“对照记忆”概念应用到代码领域，构建中训练与预训练对照模型对比；②提出功能性记忆的量化标准，结合 CodeBLEU、AST编辑距离、LLM-judge 与 HyClone 执行检验四类指标；③通过对 7,422 条具有实质功能的 Python 函数进行大规模评估，首次揭示功能性泄漏与文字泄漏的差异及其比例。

**🔧 技术方法**

技术手段主要是：①使用 Olmo‑3‑32B 的中期检查点与预训练检查点做对照；②采用贪婪解码完成函数体；③利用 BLEU、Edit、LCS、CodeBLEU (等价/语法/数据流)、TSED、LLM‑as‑a‑judge（多种提示）和 HyClone（两阶段执行相似度）等多维度相似度指标；④使用 LLaMA‑3.1‑70B‑Instruct 作为判别 LLM；⑤统计目标与参考模型在各指标上的差异 Δ。

**📊 数据集**

数据集：CraneCode，来源于 Dolmino mid‑training corpus（约 100B tokens）的 Python 子集；从中筛选出长度 10–50 行、具有实质功能的 7,422 条函数，作为功能性记忆评估样本。

**📈 对比分析**

评估方式：对目标模型与参考模型分别生成函数体，用各指标计算与真实代码的相似度，统计 Δ 的正负。结果显示：文字记忆率约 0.23%（exact）/1.56%（near‑verbatim），功能性记忆率为 LLM‑judge 3.9% / HyClone 执行验证 0.28%；多指标结合可识别约 20% 的功能性泄漏实例，说明文本相似度无法捕获大部分功能泄漏。

**⚠️ 局限性**

局限性：①AST 解析失败导致部分样本无法计算结构指标；②执行检验受限于缺失依赖、文件 I/O、全局变量等，真实覆盖率不足；③LLM‑judge 的分数受提示设计影响，存在偏差；④对照模型获取成本高，实验规模受限；⑤仅评估了 100B 规模训练集，结果可能不适用于更大规模模型。

---

## 11. Keep Policy Gradient in Charge: Sibling-Guided Credit Distillation for Long-Horizon Tool-Use Agents

**arXiv ID:** 2606.12634 | [PDF](https://arxiv.org/pdf/2606.12634v1)

**作者:** Tianyu Ding `[一作]` (Amazon Web Services), Juan Pablo De la Cruz Weinstein `[通讯]` (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于同类子弹证据的信用分配方法（SGCD），通过动态采样产生成功与失败的子弹轨迹，并用外部LLM生成步骤级信用参考，再利用教师/学生的稠密差异来重新分配奖励信号；

**💡 创新点**

创新点在于将自蒸馏的目的从直接逼近教师分布转变为在奖励驱动的策略梯度中精准分配信用，避免了传统自蒸馏导致的工具使用衰退；

**🔧 技术方法**

采用GRPO作为策略梯度骨干，结合动态采样、外部LLM总结、稠密逆KL与熵的信用特征、以及令牌级权重剪裁的信用加权技术；

**📊 数据集**

在AppWorld和τ³-airline这两个多轮工具使用基准上进行评估；

**📈 对比分析**

与匹配的GRPO+KL对照组相比，SGCD在AppWorld的TGC从42.9%提升到45.6%（增幅约5%），在τ³-airline的pass@1从0.583提升到0.602（约3%提升），且明显优于直接自蒸馏（SDPO）导致的工具使用崩溃；

**⚠️ 局限性**

局限性包括仅在Qwen3.5-4B模型上验证、仅覆盖两大基准、训练时依赖外部LLM生成信用参考、信用参考并非绝对正确信息、以及对不同模型族或更大规模的推广性尚未验证。

---

## 12. Iterating Toward Better Search: A Two-Agent Simulation Framework for Evaluating Agentic Search Architectures in E-Commerce

**arXiv ID:** 2606.12924 | [PDF](https://arxiv.org/pdf/2606.12924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 13. TrajGenAgent: A Hierarchical LLM Agent for Human Mobility Trajectory Generation

**arXiv ID:** 2606.12657 | [PDF](https://arxiv.org/pdf/2606.12657v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 14. VLADriveBench: Evaluating CoT-Action Relationship in VLA for Autonomous Driving

**arXiv ID:** 2606.12706 | [PDF](https://arxiv.org/pdf/2606.12706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 15. Reframing AI Loss of Control: What It Is, How to Have It, How to Lose It

**arXiv ID:** 2606.12442 | [PDF](https://arxiv.org/pdf/2606.12442v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 16. Does AI Reviewer See the Full Picture? Attacking and Defending Multimodal Peer Review

**arXiv ID:** 2606.12716 | [PDF](https://arxiv.org/pdf/2606.12716v1)

**作者:** Xinyu Zhao `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PaperGuard，首个针对多模态 AI 审稿系统的攻击与防御基准，评估文本与图像的对抗鲁棒性。

**💡 创新点**

创新点在于统一黑盒注入和白盒梯度攻击、构建跨模态对抗数据集，以及基于局部检索的块级嵌入搜索防御方案。

**🔧 技术方法**

采用 GCG、PGD/Auto-PGD/CW 等白盒攻击，黑盒注入文本/图像，利用 E5‑V 多模态嵌入、LLM‑as‑Judge 与块级检索。

**📊 数据集**

使用来自 ICLR 与 F1000Research 的1136篇论文，提取文本和图像，构成清洁与攻击对照样本。

**📈 对比分析**

在多模型上实验，攻击成功率最高可达80%，且视觉攻击可使分数提升14点；块级检索防御在文本/图像攻击上分别达93%+召回率、零误报率。

**⚠️ 局限性**

局限在于仅评估已知攻击模式，零样本对抗生成能力有限，且实验主要基于公开模型，商业闭源模型的真实鲁棒性尚未完全验证。

---

## 17. Zero-source LLM Hallucination Detection with Human-like Criteria Probing

**arXiv ID:** 2606.12900 | [PDF](https://arxiv.org/pdf/2606.12900v1)

**作者:** Jiahao Yang `[一作]` (South China University of Technology), Mingkui Tan `[通讯]` (South China University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了零源条件下的大模型幻觉检测方法，提出一种基于人类式准则探测的检测框架。

**💡 创新点**

创新点：①将幻觉判断拆解为可解释的多准则，并自适应加权；②采用弱监督语义一致性奖励结合GRPO实现对齐；③在推理阶段使用多采样聚合提高鲁棒性。

**🔧 技术方法**

技术：LLM代理模型（如Qwen-2.5-7b） + Group Relative Policy Optimization（GRPO） + 语义一致性弱标签（BLEURT、ROUGE等） + 结构化输出与加权求和。

**📊 数据集**

数据集：TriviaQA、SciQ、NQ Open、CoQA。

**📈 对比分析**

与多种基线（perplexity、Self-evaluation、CCS、SAPLMA、SelfCKGPT 等）在 LLaMA‑3.1‑8b、Qwen‑3‑8b 等模型上对比，平均 AUROC 提升约10–20%，显著优于现有方法。

**⚠️ 局限性**

局限性：①多采样聚合增加推理时延；②对弱监督的语义一致性依赖，可能在知识稀缺或极端推理场景下表现不佳；③对极小样本或高复杂性对话场景的鲁棒性尚未充分验证。

---

## 18. Constrained Semantic Decompression in LLMs through Persian Proverb-Conditioned Story Generation

**arXiv ID:** 2606.12599 | [PDF](https://arxiv.org/pdf/2606.12599v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 19. Agent-based models for the evolution of morphological alternation patterns

**arXiv ID:** 2606.12748 | [PDF](https://arxiv.org/pdf/2606.12748v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 20. Robustness Verification of Recurrent Neural Networks with Abstraction Refinement

**arXiv ID:** 2606.12490 | [PDF](https://arxiv.org/pdf/2606.12490v1)

**作者:** Li-Jen Lin `[一作]` (National Chengchi University), Chih-Duo Hong `[通讯]` (National Chengchi University)

**通讯引用:** 317 | [OpenAlex ID](https://openalex.org/A5064227560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种针对循环神经网络(RNN)的可验证鲁棒性方法，即基于抽象-细化的框架，针对跨零的激活区间进行单神经元分裂，并通过SHAP特征重要性排序来挑选关键时间步；

**💡 创新点**

创新点在于：①将单神经元分裂与RNN展开的时间依赖结合，解决跨零误差在时间步传播导致的保守性；②对平滑激活(tanh/ Sigmoid)采用自适应分裂点p*，进一步提升逼近精度；③利用梯度SHAP快速定位高影响神经元，避免全局指数级分裂；

**🔧 技术方法**

主要技术包括：线性边界传播(ABP+BBP)、单神经元零点分裂、SHAP梯度归因、p*分裂策略、递归分支与深度预算；

**📊 数据集**

实验数据集包括将CIFAR10图像重塑为序列、MNIST手写数字以及MNIST Stroke轨迹数据；

**📈 对比分析**

与仅使用线性边界传播的基线相比，本文方法在多数模型尺寸、序列长度和激活类型下显著提升了可认证样本比例；但在序列变长或模型宽度较大时，分裂深度受限导致提升有限；运行时相对基线略有增加，尤其在tanh模型上；

**⚠️ 局限性**

局限性包括：①分裂深度受限导致在长序列或宽模型上难以完全消除误差；②SHAP计算成本随序列长度增长；③仅在vanilla RNN和小型LSTM上验证，未覆盖更复杂的门控网络或长序列任务；④方法依赖于梯度信息，可能受训练随机性影响。

---

## 21. AIR-VLA+: Decoupling Movement and Manipulation via Cascaded Dual-Action Decoders with Asymmetric MoE for Aerial Robots

**arXiv ID:** 2606.12859 | [PDF](https://arxiv.org/pdf/2606.12859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 22. MLUBench: A Benchmark for Lifelong Unlearning Evaluation in MLLMs

**arXiv ID:** 2606.12809 | [PDF](https://arxiv.org/pdf/2606.12809v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 23. Shopping Reasoning Bench: An Expert-Authored Benchmark for Multi-Turn Conversational Shopping Assistants

**arXiv ID:** 2606.12608 | [PDF](https://arxiv.org/pdf/2606.12608v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 24. SciR: A Controllable Benchmark for Scientific Reasoning in LLMs

**arXiv ID:** 2606.13020 | [PDF](https://arxiv.org/pdf/2606.13020v1)

**作者:** Pierre Beckmann `[一作]` (Idiap Research Institute), Andre Freitas `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出SciR基准，融合演绎、归纳与因果推理三种科学推理范式，并通过可控渲染将正式对象转换为多文档科学语料；

**💡 创新点**

首次实现多范式科学推理基准，同时在推理复杂度与文档隐匿度两轴上提供可调节的参数控制，并保证答案可验证；

**🔧 技术方法**

使用LLM生成任务、基于逆向验证的渲染机制、Neuro‑Symbolic 与 Chain‑of‑Thought 推理框架；

**📊 数据集**

构建于DrugBank、Sachs蛋白信号网络与发展生物学线索的合成数据集，提供多文档语料；

**📈 对比分析**

与六种模型（gpt‑4o、deepseek‑r1、o3‑mini、llama‑3.3‑70b、qwen3‑30b、olmo‑3.1‑32b）在六种推理配置（CoT、NS、SymbCoT）下对比，发现推理模型在推理轴上明显优于指令模型；两轴难度均显著降低性能；

**⚠️ 局限性**

基准仅模拟理想化的正式对象，无法覆盖真实科学文献中常见的隐式信息与噪声；实验单次随机种子，缺乏置信区间；

---

## 25. Improving Crash Frequency Prediction from Simulated Traffic Conflicts Using Machine Learning Based Microsimulation

**arXiv ID:** 2606.12500 | [PDF](https://arxiv.org/pdf/2606.12500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 26. An Empirical Study on Predictive Maintenance for Component X in Heavy-Duty Scania Trucks

**arXiv ID:** 2606.12486 | [PDF](https://arxiv.org/pdf/2606.12486v1)

**作者:** Valeriu Dimidov `[一作]` (University of Luxembourg), Raphaël Frank `[通讯]` (University of Luxembourg)

**通讯引用:** 1964 | [OpenAlex ID](https://openalex.org/A5009044318)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种将时间序列预测维护任务转换为表格数据的方法，基于组件磨损呈单调非降趋势，只利用最新观测值训练表格模型；

**💡 创新点**

创新点在于假设磨损为单调时间序列，使用最近观测值直接进行分类，从而简化时间序列处理，减少计算量并保持与SOTA相当的性能；

**🔧 技术方法**

技术上使用AutoGluon自动化机器学习框架、梯度提升树（XGBoost、LightGBM、CatBoost）以及SHAP进行可解释性分析；

**📊 数据集**

使用Scania重型卡车Component X真实数据集，包含33,641辆卡车的多变量时间序列及车辆规格；

**📈 对比分析**

与竞赛SOTA（Bi‑LSTM、GNN、XGBoost基于窗口特征）对比，所用XGBoost模型在测试集上的挑战成本约37,733，低于SOTA的39,123；平衡准确率约0.24；

**⚠️ 局限性**

局限包括：仅能处理单调磨损组件，无法捕捉循环或环境依赖的退化；忽略时间上下文导致对中间状态（1–3类）识别不佳，误报率高，预测窗口短；

---

## 27. EgoEngine: From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations

**arXiv ID:** 2606.12604 | [PDF](https://arxiv.org/pdf/2606.12604v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 28. Towards Provably Fair Machine Learning: Bayesian Approaches For Consistent and Transparent Predictions

**arXiv ID:** 2606.12615 | [PDF](https://arxiv.org/pdf/2606.12615v1)

**作者:** Owen O'Neill `[一作]` (University College Dublin), Fintan Costello `[通讯]` (University College Dublin)

**通讯引用:** 1213 | [OpenAlex ID](https://openalex.org/A5044724458)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于贝叶斯推断的分类器——Fair Bayesian Classifier，强制模型在每个细粒度子组上满足确定性与统计一致性，并在不可能满足时主动放弃预测。

**💡 创新点**

创新点在于：①把统计一致性（检验预测与贝叶斯后验是否可接受）作为完整子组级别的硬约束；②在所有可能子组上（包括交叉子组）全面执行，而非仅预先指定的保护组；③通过贝叶斯后验自适应宽阔置信区间，从而在小样本子组上实现合理的拒绝预测；④利用约束求解得到多校准性能的副产品。

**🔧 技术方法**

主要技术包括：贝叶斯Beta后验、Beta-Binomial预测分布、显著性检验得到 V_min/V_max 区间、层次化 v‑node 归一化、基于Gurobi的整数规划求解与后验加权选择，均以 Python、pandas、scipy、sklearn 为实现工具。

**📊 数据集**

在三大公开基准数据集上验证：Adult（收入预测）、COMPAS（再犯风险）、Bank Marketing（营销响应），全部为离散化特征的二元分类问题。

**📈 对比分析**

与决策树、神经网络以及 Proportional Multicalibration 后处理模型做对比；在所有子组上统计一致性错误均为 0，且在不放弃样本的前提下精度均高于基线；多校准与多准确性指标也表现优异，尤其在 COMPAS 数据集上差距明显。

**⚠️ 局限性**

局限性：仅适用于离散特征和二分类问题；对连续变量需要先分箱；在极大规模或高维数据下约束规模暴增，求解成本升高；对未见 d‑node 的在线预测需重新求解，限制了实时部署；处理历史偏见的数据仍需在先验设定上做进一步研究。

---

## 29. PersonaDrive: Human-Style Retrieval-Augmented VLA Agents for Closed-Loop Driving Simulation

**arXiv ID:** 2606.12616 | [PDF](https://arxiv.org/pdf/2606.12616v1)

**作者:** Mahmoud Srewa `[一作]` (University of California, Irvine), Salma Elmalaki `[通讯]` (University of California, Irvine)

**通讯引用:** 241 | [OpenAlex ID](https://openalex.org/A5038820344)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PersonaDrive，一种基于检索增强的视觉-语言-动作（VLA）驾驶代理，通过检索人类风格示例实现多风格驾驶；

**💡 创新点**

创新点在于用风格指令驱动的人类驾驶数据构建检索数据库，并通过检索和上下文微调实现无参数风格切换；

**🔧 技术方法**

使用三阶段流程：离线三元组挖掘、轻量检索头（融合SigLIP与控制编码器）以及在SimLingo VLM基础上进行上下文微调；

**📊 数据集**

使用CARLA Leaderboard路由下的8位受试者风格指令（保守、中性、进取）驾驶数据，并在Bench2Drive闭环基准上评估；

**📈 对比分析**

与SimLingo、HiP‑AD、StyleDrive、DMW等基线对比，PersonaDrive在无风格模式下提升DS 4.6%，在三种风格下均达最高DS，平均速度与加速度从保守到进取提升约18%与25%；

**⚠️ 局限性**

局限在于风格离散、参与者数量有限、仅单前摄像头、仅CARLA环境，缺乏跨域泛化与持续风格建模。

---

## 30. Graph Reduction in Multirelational Networks: A Spreading-Oriented Reduction Benchmark

**arXiv ID:** 2606.12581 | [PDF](https://arxiv.org/pdf/2606.12581v1)

**作者:** Mateusz Stolarski `[一作]` (Wrocław University of Science and Technology), Piotr Bródka `[通讯]` (Wrocław University of Science and Technology)

**通讯引用:** 1820 | [OpenAlex ID](https://openalex.org/A5022573878)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SORB 基准框架，用于在单层与多层网络上系统评估图稀疏化与聚合对影响力最大化（IM）性能的影响。

**💡 创新点**

创新点在于：将图约简作为 IM 的预处理步骤纳入标准化评估流程，兼顾多种稀疏化/聚合策略，并对单层与多层网络的不同表现进行量化对比。

**🔧 技术方法**

采用了多种稀疏化方法（ForestFire、Jaccard、LocalDegree、TSpanner、Random）和聚合方法（VariationNeighborhoods、AffinityGs），配合多种 IM 模型（Degree、VoteRank、ts-net 等）以及多层独立传播模型 MICM。

**📊 数据集**

使用了 7 个真实网络：WikiCS、Amazon-CS（同质网络）以及 Freebase、IMDB、FinDKG、arxiv、timik（多层网络）。

**📈 对比分析**

通过 Gain@k 与 AUC_cutoff 两个指标对比原始与约简后网络的预测结果。结果显示：稀疏化对单层网络的 Seed 集质量影响较小；对多层网络则显著降低排名稳定性；不同约简策略对性能的影响远大于稀减率；在同质网络稀疏化后，学习型 ts-net 的表现显著提升。

**⚠️ 局限性**

局限性：目前缺乏专门针对多层网络的约简方法；压平步骤导致多边缘冗余，影响稀疏化效果与资源消耗；聚合方法计算成本高，未能在完整数据集上完成全面评估。

---

## 31. SalArt-VQA: Diagnosing Whether VLMs Understand Salient Artifacts in Generated Images

**arXiv ID:** 2606.12671 | [PDF](https://arxiv.org/pdf/2606.12671v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 32. The No-show Paradox in Single Transferable Vote under One-dimensional Preferences

**arXiv ID:** 2606.12785 | [PDF](https://arxiv.org/pdf/2606.12785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 33. ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories

**arXiv ID:** 2606.12556 | [PDF](https://arxiv.org/pdf/2606.12556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 34. High-Fidelity Two-Step Image Generation via Teacher-Aligned End-to-End Distillation

**arXiv ID:** 2606.12575 | [PDF](https://arxiv.org/pdf/2606.12575v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 35. Evoflux: Inference-Time Evolution of Executable Tool Workflows for Compact Agents

**arXiv ID:** 2606.12674 | [PDF](https://arxiv.org/pdf/2606.12674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 36. Scalable anomaly detection via a univariate Christoffel function

**arXiv ID:** 2606.12483 | [PDF](https://arxiv.org/pdf/2606.12483v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 37. A unified complexity bound for logconcave sampling

**arXiv ID:** 2606.12694 | [PDF](https://arxiv.org/pdf/2606.12694v1)

**作者:** Yunbum Kook `[一作]` (Georgia Tech), Santosh S. Vempala `[通讯]` (Georgia Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在论文中，作者给出了一个简单统一且几乎紧致的采样复杂度上界，用于从温暖起点对任意对数凹分布进行采样，结合指数提升和前向后向采样器。

**💡 创新点**

创新点是改进了提升分布的 Poincaré 常数，利用变熵（varentropy）证明提升分布的协方差仅比原分布多 O(1/d)，从而去掉了之前需要的 "∨1" 项，实现了统一的采样复杂度。

**🔧 技术方法**

主要技术包括指数提升 (exponential lifting)、前向后向采样器 (proximal sampler)、Poincaré 不等式、对数 Sobolev 不等式以及变熵与对数凹分布方差的关系。

**📊 数据集**

论文为理论工作，未使用任何具体数据集，而是通过数理证明给出了复杂度上界。

**📈 对比分析**

与之前的工作相比，新算法在一般对数凹采样和受限对数凹采样（如高斯在凸体内）中都实现了几乎最优的查询复杂度；在良好条件（强对数凹）下，复杂度从 κ d + d² 降低到 κ d。

**⚠️ 局限性**

局限性在于仍需先有一个 O(1) 的温暖起点；算法的实现依赖于评估 oracle，且在实际高维问题中实现效率和数值稳定性尚待进一步验证。

---

## 38. Quickest Detection of Hallucination Onset: Delay Bounds and Learned CUSUM Statistics

**arXiv ID:** 2606.12476 | [PDF](https://arxiv.org/pdf/2606.12476v1)

**作者:** Igor Itkin `[一作]` `[通讯]` (Independent Researcher), Igor Itkin (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将幻觉起始检测建模为快速变点检测问题，并推导出最优检测延迟下界；

**💡 创新点**

创新点在于将幻觉起始检测与经典 Lorden 下界及学习型 CUSUM 对齐，并量化学习模型相对基线的优势来源；

**🔧 技术方法**

使用的技术包括：一阶马尔可夫链建模、Lorden 下界与信息率分析、学习型 CUSUM（递归标签器）以及梯度提升与逻辑回归等基线分类器；

**📊 数据集**

数据集为 RAGTruth 的 2,700 条生成样本（含 943 条幻觉），使用 33 维特征（文本统计、NLI 信号、生成器 log‑prob 等）；

**📈 对比分析**

在相同误报率（α=1%）下，学习型 CUSUM 的平均检测延迟约 11–13 令牌，远快于线性分类器（≈30 令牌）和参数化 CUSUM（≈41 令牌），但仍高于理论下界约 1.3 令牌；

**⚠️ 局限性**

局限性包括：特征独立性假设导致下界可能过于乐观；仅检测单个变点；使用的 33 维特征分辨率有限，导致信息率短缺；并且低误报率下的召回率仅约 30%。

---

## 39. Fantastic Scientific Agents and How to Build Them: AgentBuild for Rietveld Refinement

**arXiv ID:** 2606.12834 | [PDF](https://arxiv.org/pdf/2606.12834v1)

**作者:** Woong Shin `[一作]` (Oak Ridge National Laboratory), Rafael Ferreira da Silva `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种名为 AgentBuild 的工作流阶段，用来根据科学家手工制定的合同（由可版本化的评分标准、难度分级课程表以及外部知识库组成）自动构建面向工具的 LLM 代理，并在 X‑射线衍射的 Rietveld 细化任务中进行实例化与评估。

**💡 创新点**

创新点在于将科学家专业判断提炼为可读的、可版本化的合同，确保代理构建过程可重复、可追溯、与基础模型升级无关；同时通过评分标准驱动的 LLM 判定器与元优化器在限定编辑边界内迭代改进代理，实现自动化构建而不掩埋专家知识。

**🔧 技术方法**

使用的技术包括：Claude Sonnet 4.6 作为代理执行模型、Claude Opus 4.7 作为判定与元优化器、Model Context Protocol（MCP）服务器包装 GSAS‑II 细化引擎、A2A（Agent‑to‑Agent）容器化包装、PROV 规范记录构建过程的 provenance 跟踪，以及基于评分标准的 LLM‑驱动评估。

**📊 数据集**

实验数据集为 Li‑6.4La₃Zr₁.4Ta₀.6O₁₂（LLZO）粉末的 X‑射线衍射图谱，覆盖多种计数时间（1 min、3 min、10 min、30 min、1 h、4 h）以及基线样本 PbSO₄ 与氟磷灰石，构成 SNR 梯度课程表。

**📈 对比分析**

评价方法是将构建好的代理在每一轮课程表案例中按预设的 P‑strict 评分标准（包含 11 个维度）进行判定；若所有维度均达标则晋级到更高难度案例。实验显示，在第 7 轮迭代完成 4 案 P‑strict 里程碑后，代理能够在 4 h 高 SNR 案例中得到合理的细化结果，尽管未通过严格的工作流范围判定。相较于传统手工细化，代理能自动执行全部工具调用并保留可追溯的审计信息，但并未给出与现有自动化脚本或手工操作的定量性能对比。

**⚠️ 局限性**

局限性主要包括：仅验证了单相、基于 SNR 的细化任务；未覆盖多相识别、微量杂质建模或不同材料/仪器的通用性；评分标准与报告接口的确定性不足，导致 D1–D3 维度偶尔失配；以及对抗性情境下评分标准的鲁棒性尚未测试。

---

## 40. Exploring How Agent Voice Accents Shape Human-AI Collaboration in K-12 Group Learning

**arXiv ID:** 2606.12805 | [PDF](https://arxiv.org/pdf/2606.12805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 41. Will AI Agents Free Us From Meaningless Work? A Human-Centered Analysis

**arXiv ID:** 2606.12430 | [PDF](https://arxiv.org/pdf/2606.12430v1)

**作者:** Davide Ghia `[一作]` (Politecnico di Torino), Daniele Quercia `[通讯]` (Nokia Bell Labs)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在工作任务层面上测量并验证“bullshit”（无意义）感知度，并探讨其与工人对AI自动化偏好及人类监督需求之间的关系。

**💡 创新点**

①首次构建并验证任务级的“bullshit”感知量表；②揭示该感知度能强烈预测工人对AI自动化的意愿及其对人类干预的需求；③发现高“bullshit”感知任务同时被视为AI自主执行的可行目标。

**🔧 技术方法**

采用问卷调查收集数据，使用因子分析（EFA）验证单因素结构，利用混合效应回归模型分析“bullshit”感知与自动化偏好、人类监督需求之间的关系。

**📊 数据集**

包含202名美国工人对171项O*NET提取的工作任务进行评价的问卷数据，任务来自22个职业，涵盖多步骤、计算机化可被AI代理执行的任务。

**📈 对比分析**

通过与传统任务级别的AI可行性评估对比，发现“bullshit”感知度与工人自动化意愿呈正相关，且与所需人类监督呈负相关。混合效应模型显示：一标准差的“bullshit”提升平均自动化欲望0.39点（p<0.001），所需人类监督下降0.22点（p<0.001）。

**⚠️ 局限性**

研究样本仅来自美国，可能影响普适性；仅考虑任务特征，未纳入组织文化、薪酬等情境因素；评估的是工人主观可行性，而非技术实现可行性；研究基于现状，AI技术快速演进可能改变偏好。

---

## 42. Fed-FBD: Federated Functional Block Diversification for Isolation, Privacy, and Surgical Unlearning

**arXiv ID:** 2606.12679 | [PDF](https://arxiv.org/pdf/2606.12679v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 43. Mapping AI Programs in the U.S: A Status Report from Early 2026 and an Analysis of AI Majors and Minors

**arXiv ID:** 2606.12428 | [PDF](https://arxiv.org/pdf/2606.12428v1)

**作者:** Felix Muzny `[一作]` (Northeastern University), Carla E. Brodley `[通讯]` (Northeastern University)

**通讯引用:** 14770 | [OpenAlex ID](https://openalex.org/A5045379675)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套自动化抓取与可视化工具（cicmap.ai），用于实时收集并展示美国四年制高校中人工智能（AI）本科项目（主修、辅修、专业方向、证书）的课程要求与地理分布。

**💡 创新点**

创新点在于：①构建了高覆盖率（86% CS 毕业生）的数据采集管线，结合搜索 API 与 LLM 进行动态定位与校正；②首次提供完整、交互式的 AI 课程需求地图，帮助学生、教师与行政部门快速获取与比较不同高校的 AI 教育资源；③系统性分析了 66 个 AI 主修与 87 个 AI 辅修的课程组成、必修与选修比例，为 AI 课程设置的差异性提供量化证据。

**🔧 技术方法**

使用技术包括：Google/Exa 搜索 API、DeepSeek 开源 LLM（用于识别学院网站与项目类型）、Python 爬虫与数据处理库、人工审阅与校验环节、Web 前端（地图可视化）以及定期批处理脚本实现半年度更新。

**📊 数据集**

数据集来源：①IPED 2023 年 CS 毕业生数据（占美国 CS 毕业生 86%）；②各高校官网公开的 AI 项目课程要求页面；③自建的抓取结果数据表，包含 569 所高校、975 个 AI/CS 项目及其课程信息。

**📈 对比分析**

通过人工标注 113 个项目进行验证，工具在项目类型识别上达 100% 准确率；对比分析与 ACM CS2023 课程指南、Stanford、He 等先前研究的结果，说明本工具覆盖面更广、更新速度更快，并能捕捉新上线的 AI 项目；性能方面，工具每学期完成一次全校抓取，耗时受算力限制。

**⚠️ 局限性**

局限性包括：①对未能成功抓取的网站（格式异常、链接失效）缺失数据；②仅覆盖四年制高校，无法反映社区学院等其他教育层级的 AI 教育情况；③未纳入数据科学（DS）项目，导致 AI 与 DS 的交叉点缺乏系统分析；④更新频率受计算资源限制，不能实现实时刷新；⑤课程层面的细节（如先修关系、课程难度）尚未深入挖掘。

---

## 44. Definitional alignment before capability alignment: a Design-Science framework for adjudicating claims about AGI

**arXiv ID:** 2606.12713 | [PDF](https://arxiv.org/pdf/2606.12713v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 45. Eidola: Modeling Multi-GPU Network Communication Traffic in Distributed AI Workloads

**arXiv ID:** 2606.12638 | [PDF](https://arxiv.org/pdf/2606.12638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 46. Physics-Aware Auxiliary Losses Improve Out-of-Distribution Generalization of a GNN Synthesizability Filter

**arXiv ID:** 2606.12651 | [PDF](https://arxiv.org/pdf/2606.12651v1)

**作者:** Riya Bisht `[一作]`, Dhruv Agarwal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出在药物发现的可合成性过滤器中加入物理先验的辅助损失，以提高对生成模型提出的异分布分子做出更可靠的可合成性判断。

**💡 创新点**

创新点在于将闭式物理先验（Bertz复杂度指数和MMFF94应变能）作为轻量级辅助回归损失，直接嵌入GINE图神经网络中，而不是使用PINN残差或额外的输入特征。

**🔧 技术方法**

采用4层GINE图神经网络为主干，辅以二分类头和回归头；辅助损失包括Huber回归Bertz复杂度和对应变能的软惩罚；训练使用AdamW，配合多种超参数组合。

**📊 数据集**

使用三大公开数据集的联合子集：HIV、Tox21（合成药物类）与COCONUT（天然产物），共计65,177个分子，其中10,000个用于辅助损失的计算。

**📈 对比分析**

对比方法为单源OOB评估：在HIV+Tox21上训练，在COCONUT上测试，使用5个随机种子并计算配对bootstrap 95%置信区间。结果显示所有物理先验变体在OOB AUC上均显著优于基线（均值Δ≈+0.006到+0.007），但在I.I.D.上无显著差异。

**⚠️ 局限性**

局限性包括：标签仅为SAScore阈值的近似，可合成性真实性未知；应变能仅基于单一ETKDG构象，可能不代表全局应变；实验仅检验一种OOB轴和单一权重配置，种子数量有限；对极大天然产物（构象生成失败）处理不充分。

---

## 47. Learning Task-Aware Sampling with Shared Saliency through Density-Equalizing Mappings

**arXiv ID:** 2606.12869 | [PDF](https://arxiv.org/pdf/2606.12869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 48. Beyond Attack Success Rate: Examining Trigger Leakage in Vision-Language Agentic Systems

**arXiv ID:** 2606.12586 | [PDF](https://arxiv.org/pdf/2606.12586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 49. Diffusion-Network Alignment: An Efficient Algorithm and Explicit Probability Bounds

**arXiv ID:** 2606.12879 | [PDF](https://arxiv.org/pdf/2606.12879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 50. Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of Evolving Contexts

**arXiv ID:** 2606.12807 | [PDF](https://arxiv.org/pdf/2606.12807v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 51. Teach-and-Repeat: Accurately Extracting Operational Knowledge from Mobile Screen Demonstrations to Empower GUI Agents

**arXiv ID:** 2606.12817 | [PDF](https://arxiv.org/pdf/2606.12817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 52. OpenRoundup: Multi-Table Data Wrangling Through Interactive Visualization

**arXiv ID:** 2606.12648 | [PDF](https://arxiv.org/pdf/2606.12648v1)

**作者:** Stephen Kasica `[一作]` (University of British Columbia), Tamara Munzner `[通讯]` (University of British Columbia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个基于浏览器的无代码数据整合工具 OpenRoundup，帮助数据记者在不写代码的情况下完成多表整合并导出单表结果。

**💡 创新点**

创新点在于提出“预先整合”(eager consolidation)概念、两种声明式操作(Stack/Pack)以及以方案为先、值为需的交互设计，强调对表集合而非单表的可视化整合。

**🔧 技术方法**

技术实现包括前端无服务器架构、DuckDB‑WASM 客户端数据库、Redux‑Saga 状态管理、React/D3 可视化以及自定义拖拽和树形视图。

**📊 数据集**

使用了 17 个公开新闻报道的实际整合脚本所涉及的多源 CSV/TSV 数据集，涵盖从气象到犯罪、投票和财务等多种主题。

**📈 对比分析**

通过复现 17 篇记者脚本与部署测试（4 名专业记者）评估，结果显示工具能够完整复制手工整合流程，交互响应时间在数秒内完成中等规模（≤100 MB）文件的堆叠/合并；与 Excel 或代码脚本相比在多表联结上更直观、耗时更少。

**⚠️ 局限性**

局限性包括仅支持静态 ASCII 表文件、单线程 JavaScript 计算导致大文件/笛卡尔乘积受限、缺乏自动化列类型推断/键匹配建议、未覆盖列级数据清洗和缺少可重复性记录。

---

## 53. To Share or Not to Share: Orchestrating Trustworthy Data in Global Value Chains

**arXiv ID:** 2606.12788 | [PDF](https://arxiv.org/pdf/2606.12788v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 54. Amnesia: A Stealthy Replay Attack on Continual Learning Dreams

**arXiv ID:** 2606.12655 | [PDF](https://arxiv.org/pdf/2606.12655v1)

**作者:** Ahmed Sharshar `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Mohsen Guizani `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了持续学习中经验回放的索引选择对模型的破坏性攻击，并提出了在可审计的可见度与采样量预算下的 Amnesia 攻击。

**💡 创新点**

首次将攻击限定在可审计的采样比例和类分布偏差预算内，提出了轻量级的偏好+投影两步方法，实现了对回放样本比例的最大化破坏同时满足日志审计。

**🔧 技术方法**

偏好倾斜（轻量级类效用估计）、KL/TV 投影（单参数指数倾斜/双向水填充）、整数配额四舍五入、窗口调度实现滚动审计、利用日志可见度指标。

**📊 数据集**

Split CIFAR‑10/100、CORe50、Tiny‑ImageNet 等标准持续学习基准。

**📈 对比分析**

与 ER、ER‑ACE、SCR、DER++ 等强回放基准对比，使用 ACC、BWT、-BWT 评估，攻击在保持审计合规的前提下显著降低 ACC 并提升遗忘；KL 版兼顾隐蔽性，TV 版更强但更易检测。

**⚠️ 局限性**

假设缓冲区已标记且审计基准为类直方图，未考虑无标签或动态标签场景；当每类样本数接近 1 时离散化与可用性约束会导致审计超限，TV 版在此情况下易被发现。

---

## 55. AI-Automation Tooling in Computer Engineering Education: Mixed-Methods TAM/UTAUT Evidence for a General Acceptance Attitude

**arXiv ID:** 2606.12424 | [PDF](https://arxiv.org/pdf/2606.12424v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 56. Dolph2Vec: Self-Supervised Representations of Dolphin Vocalizations

**arXiv ID:** 2606.12503 | [PDF](https://arxiv.org/pdf/2606.12503v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 57. Helping Figures Tell their Story! Paper-Grounded Video Generation Explaining Complex Scientific Figures

**arXiv ID:** 2606.12576 | [PDF](https://arxiv.org/pdf/2606.12576v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 58. Auditing Discriminatory Patterns in Mortgage Lending Through Association Rules and Fair Binning

**arXiv ID:** 2606.12435 | [PDF](https://arxiv.org/pdf/2606.12435v1)

**作者:** Archit Rathod `[一作]` (University of Illinois Chicago), Het Nagda `[通讯]` (University of Illinois Chicago)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在美国抵押贷款数据中，标准与公平的属性分箱对下游模式挖掘产生的种族与性别偏差的影响。

**💡 创新点**

创新点在于结合公平分箱、FP‑Growth关联规则挖掘与基于聚类的差异影响审计三种方法，首次揭示了即使在相似财务特征下仍存在的种族歧视。

**🔧 技术方法**

使用的技术包括 PySpark 数据清洗与分箱、ε‑偏差公平分箱算法、FP‑Growth 关联规则挖掘、K‑Means 聚类及 DIR 差异影响度量。

**📊 数据集**

采用了 2023 年 HMDA 数据集中芝加哥大都会区的 103,481 份抵押贷款申请记录。

**📈 对比分析**

通过对比标准与公平分箱的偏差、FP‑Growth 规则集以及聚类后 DIR 值，发现标准分箱产生 9.63% 种族偏差，而公平分箱虽能降低偏差但在 7 组下需 ε=0.08 且公平成本高；FP‑Growth 规则一致且以 DTI 为主导；聚类 DIR 揭示 10 个群组呈显著种族差异。

**⚠️ 局限性**

局限性包括缺乏信用评分特征、FP‑Growth 支持阈值过高导致少数族群规则难以发现、仅聚焦芝加哥区域、七组公平分箱困难以及 K‑Means 聚类假设不匹配等。

---

## 59. Pluralistic-Alignment Urbanism: Operationalizing a Right to AI for Inclusive Public Space

**arXiv ID:** 2606.12434 | [PDF](https://arxiv.org/pdf/2606.12434v1)

**作者:** Rashid Mushkani `[一作]` (Université de Montréal), Rashid Mushkani `[通讯]` (Université de Montréal)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5015531560)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了如何在市政公共空间AI系统中将多元评估视为合法治理证据，并提出了Pluralistic-Alignment Urbanism（PAU）治理框架；在蒙特利尔两个参与式案例中检验了评估多样性、协商、可扩展性与生成图像中中立性对治理的影响；并将这些经验转化为可操作的市政治理架构。

**💡 创新点**

创新点在于：①将多元评估与不确定性视为治理输入而非错误；②提出PAU框架，结合权益、程序、证据与回路，形成可实施的治理架构；③通过案例展示评估多样性、协商效果与可扩展性受限，及生成图像中高中立率对偏好调优的约束。

**🔧 技术方法**

技术包括：子群体感知的多输出回归模型（使用语义分割+多层感知机）用于街景评分；Stable Diffusion XL生成公共空间图像；Direct Preference Optimization（DPO）进行偏好调优；以及对多元评估数据的协方差、ICC、R²等统计分析。

**📊 数据集**

数据集包括：1）街景图像120张（60街道），以及45,000张用于推理的街景图像；2）自定义街道评估标签（28个分数，涵盖6个自报身份子群体+总体）；3）生成图像16,693张（13,462张合格）与440个社区生成提示；4）对生成图像进行的113,130条偏好比较（包含52.4%中立）。

**📈 对比分析**

对比方法：在街景评分中，将子群体感知模型与基线（平均分）对比，R²验证在验证集0.91、测试集0.89；在生成图像中，DPO调优后与基线比较，DPO被偏好比例为33.3%，基线为14.3%，中立率高达52.4%，表明多数比较未产生确定排序。

**⚠️ 局限性**

局限性：①模型受视觉模态限制，无法充分捕捉体验性或符号性维度；②子群体划分粗糙，可能忽略内群体差异；③参与式评估受小样本、群体动力与疲劳影响；④中立选择可能受不确定性或疲劳影响；⑤治理框架假设市政对参与与监督有足够资源，实际执行可能受政治与资源限制。

---

## 60. SENTINEL: Failure-Driven Reinforcement Learning for Training Tool-Using Language Model Agents

**arXiv ID:** 2606.12908 | [PDF](https://arxiv.org/pdf/2606.12908v1)

**作者:** Ziyi Wang `[一作]` (Northeastern University), Dakuo Wang `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于模型失败回放的强化学习循环（Controller–Proposer–Solver），用失败案例动态生成针对性训练任务以提升语言模型的工具使用能力。

**💡 创新点**

创新点在于将失败轨迹视为自适应训练信号，通过控制器诊断错误模式，生成可执行且聚焦于模型弱点的任务，实现对RL训练分布的在线调整。

**🔧 技术方法**

主要技术包括LLM驱动的错误分析与任务生成、GRPO（轨迹重要性采样的策略梯度方法）、任务成功奖励与奖励塑形、以及可执行的工具调用环境模拟。

**📊 数据集**

使用了Tau2-Bench Retail（可执行的零售客服工具使用基准）以及基于Qwen3-4B-Thinking-2507模型的任务数据；对比基线模型、一般RL和失败驱动RL。

**📈 对比分析**

在Tau2-Bench Retail上，失败驱动RL的Pass1从66.4%提升至74.9%，Pass2从51.6%提升至60.5%，Pass3从43.2%提升至51.2%，显著优于基线和一般RL；在SFT初始化模型上也保持提升。

**⚠️ 局限性**

局限性包括仅在单一可执行环境（Tau2-Bench Retail）验证，生成任务受当前模型曝光的失败案例限制，可能无法覆盖所有稀有或未见过的测试场景；未来需扩展到多样化工具使用域并结合预设计的压力测试。

---

## 61. ReCal: Reward Calibration for RL-based LLM Routing

**arXiv ID:** 2606.12479 | [PDF](https://arxiv.org/pdf/2606.12479v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 62. EquiDexFlow: Contact-Grounded SE(3)-Equivariant Dexterous Grasp Generative Flows

**arXiv ID:** 2606.12728 | [PDF](https://arxiv.org/pdf/2606.12728v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 63. Diffusion Transformer World-Action Model for AV Scene Prediction

**arXiv ID:** 2606.12987 | [PDF](https://arxiv.org/pdf/2606.12987v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 64. From Prompts to Preferences: An Open-Source Platform for Generative AI-Enhanced Conjoint Analysis

**arXiv ID:** 2606.12972 | [PDF](https://arxiv.org/pdf/2606.12972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 65. Characterizing Tests in IoT Software: Practices, Challenges and Opportunities

**arXiv ID:** 2606.12592 | [PDF](https://arxiv.org/pdf/2606.12592v1)

**作者:** Rufeng Chen `[一作]` (McGill University), Lili Wei `[通讯]` (McGill University)

**通讯引用:** 2790 | [OpenAlex ID](https://openalex.org/A5028202462)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

对 IoT 开源软件的测试案例进行实证研究，评估测试效果、分析未覆盖代码因素并探究 mock 对象的使用与作用。

**💡 创新点**

首次系统量化 IoT 软件的覆盖率与变异测试效果，发现大多数未覆盖代码源于外部依赖难以模拟，并提出通过 mock 对象迁移来提升测试覆盖率的创新思路。

**🔧 技术方法**

采用覆盖率工具（Coverage.py、JaCoCo、nyc）和变异测试工具（mutmut、PIT、Stryker）进行动态分析，使用代码模式匹配与手工编码对 mock 对象进行抽取与分类。

**📊 数据集**

使用 824 个 GitHub IoT 项目构建数据集，进一步筛选出 37 个可执行测试的子集用于覆盖率与变异测试分析。

**📈 对比分析**

与传统 Java/Maven 项目对比，IoT 项目平均语句覆盖率 65.2%、分支覆盖率 53.4%、变异得分 39.9%，覆盖率相对更高但变异得分与一般项目相当，表明测试效率仍不足。

**⚠️ 局限性**

研究仅覆盖可执行测试的项目，无法涵盖设备层硬件与完整端到端交互；抽样与手工编码可能带来偏差，且未能充分体现真实 IoT 环境中的时序与网络不确定性。

---

## 66. TimeROME-DLM: Temporal Causal Tracing and Low-Rank Inference-Time Knowledge Editing for Masked Diffusion Language Models

**arXiv ID:** 2606.12841 | [PDF](https://arxiv.org/pdf/2606.12841v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 67. PI-Hunter: Automated Red-Teaming for Exposing and Localizing Prompt Injections

**arXiv ID:** 2606.12737 | [PDF](https://arxiv.org/pdf/2606.12737v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 68. Boosting Direct Preference Optimization with Penalization

**arXiv ID:** 2606.12505 | [PDF](https://arxiv.org/pdf/2606.12505v1)

**作者:** Pengwei Sun `[一作]` (Stanford University), Pengwei Sun `[通讯]` (Stanford University)

**通讯引用:** 1562 | [OpenAlex ID](https://openalex.org/A5028986731)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在离线偏好优化框架中引入了对参考模型贪婪解码输出的有选择性惩罚，扩展了Direct Preference Optimization (DPO)；

**💡 创新点**

创新点在于利用参考模型生成的贪婪响应作为第三个惩罚信号，并在策略仍对拒绝回复赋高似然时才激活惩罚，从而在保持原有对比损失的同时加强对不良局部模式的纠正；

**🔧 技术方法**

采用了DPO基本对比损失，并加入基于参考贪婪回复的多种惩罚方式（Unlikelihood、NPO、SimNPO）以及可分离的门控与权重函数；

**📊 数据集**

使用了UltraFeedback风格的偏好数据集，包括针对Llama-3-8B-Instruct的<https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback>和针对Gemma-2-9B-IT的<https://huggingface.co/datasets/princeton-nlp/gemma2-ultrafeedback-armorm>；

**📈 对比分析**

在AlpacaEval 2.0上与DPO、SimPO和AlphaDPO进行比较，主要指标为长度控制胜率（LC-WR）；DPOP在两大模型上均显著提升LC-WR（Llama-3-8B从44.01提升至46.35，Gemma-2-9B从73.08提升至78.22），并证明SimNPO式惩罚最为有效；

**⚠️ 局限性**

局限性包括对参考贪婪回复的惩罚可能无法区分有害与可接受的局部模式，且实验仅覆盖指令跟随任务，缺乏在推理、事实性和安全性等更广泛基准上的验证。

---

## 69. LNTest: A Testbed for Evaluating Bitcoin Lightning Network-Based Botnets

**arXiv ID:** 2606.12887 | [PDF](https://arxiv.org/pdf/2606.12887v1)

**作者:** Thomas Bakaysa `[一作]` (East Texas A&M University), Abdullah Aydeger `[通讯]` (Florida Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个可复现的 Lightning Network（LN）基于 botnet 的测试平台 LNTest，并在真实节点上评估了 LNBot 与 D‑LNBot 的性能与拓扑特性。

**💡 创新点**

首创支持三种拓扑模式（确定链、自治发现、用户自定义）的 LN 基 botnet 测试床，并实测 D‑LNBot 自治模式实际形成的聚簇链及其对破坏策略的脆弱性。

**🔧 技术方法**

使用 Docker 容器化 Core Lightning 与 Bitcoin Core regtest，借助 Keysend+TLV 携带命令，配合 Python 脚本调度与 POSIX 共享内存实现实时监测。

**📊 数据集**

通过自建 regtest 链生成节点与通道，实验规模从 10 到 500 的 C&C 服务器数量；未使用公开 LN 图谱，仅生成随机或自定义图。

**📈 对比分析**

通过多组实验比较链式、BA 规模自由网络和自治聚簇链在命令传播延迟、覆盖率以及随机与目标性节点剔除下的分割表现，发现传播延迟线性随节点数增长，邻居数不影响速度；聚簇链对两种剔除方式均易碎，链式对随机更鲁棒，BA 对目标更敏感。

**⚠️ 局限性**

仅模拟 C&C 层，未涵盖真实网络延迟、异构 LN 实现、IoT 资源约束、长期支付堵塞等因素；实验仅在单机 regtest 环境进行，未验证对公共 LN 网络的适用性。

---

## 70. (Human) Attention Is (Still) All You Need: Human oversight makes AI-assisted social science reliable

**arXiv ID:** 2606.12848 | [PDF](https://arxiv.org/pdf/2606.12848v1)

**作者:** Chen Zhu `[一作]` (China Agricultural University), Weilong Zhang `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种名为HLER的AI辅助研究工作流，通过将大型语言模型的推理与确定性计算分离，并设置三道人工决策门，提升实验可信度。

**💡 创新点**

创新点在于将决策架构视为“研究手架”，通过行为科学原则（默认、顺序、承诺）而非单纯模型提升可靠性，并证明在数据分布差异大的任务上收益最大。

**🔧 技术方法**

使用的技术包括多代理Python框架、Claude Sonnet 4.6作为LLM、R脚本实现确定性算子、以及人工判断门控；整体系统实现了可审计的研究记录。

**📊 数据集**

所用数据集为四个：英国生物样本库（UK Biobank）、中国健康与营养调查（CHNS）、中国健康与退休纵向研究（CHARLS）以及清代吉林省多代面板（CMGPD-Liaoning）。

**📈 对比分析**

通过2×4因子设计和80次消融实验比较，受限HLER工作流的整体失败率从72%降至16%，且在最不常见的数据集上差距最大，证明决策架构显著提升性能。

**⚠️ 局限性**

局限性包括仅测试单一LLM模型、仅涵盖四个数据集、人工评审带来主观性、确定性/概率边界模糊、消融样本量有限，且结果未验证在更广泛研究环境中的推广性。

---

## 71. Multi-Bitwidth Quantization for LLMs Using Additive Codebooks

**arXiv ID:** 2606.12876 | [PDF](https://arxiv.org/pdf/2606.12876v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 72. JSCGC: Joint Source-Channel-Generation Coding for Wireless Generative Communications

**arXiv ID:** 2606.12858 | [PDF](https://arxiv.org/pdf/2606.12858v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 73. The Containment Gap: How Deployed Agentic AI Frameworks Fail Public-Facing Safety Requirements

**arXiv ID:** 2606.12797 | [PDF](https://arxiv.org/pdf/2606.12797v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 74. Strategic Decision Support for AI Agents

**arXiv ID:** 2606.12587 | [PDF](https://arxiv.org/pdf/2606.12587v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 75. Foresight: Iterative Reasoning About Clues that Matter for Navigation

**arXiv ID:** 2606.12550 | [PDF](https://arxiv.org/pdf/2606.12550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 76. Masked Neural Detection for Constrained Channel Coding in Molecular Communication

**arXiv ID:** 2606.12489 | [PDF](https://arxiv.org/pdf/2606.12489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 77. Analyzing and Improving Fine-grained Preference Optimization in Medical LVLMs

**arXiv ID:** 2606.12590 | [PDF](https://arxiv.org/pdf/2606.12590v1)

**作者:** Shayan Mohammadizadehsamakosh `[一作]` (York University), Elham Dolatabadi `[通讯]` (York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了细粒度、基于视觉对比和词级双向KL正则的医学视觉语言模型对齐框架（FiRe‑MPO）。

**💡 创新点**

创新点在于将on‑policy微调、最小编辑的临床错误纠正、可视化对比损失和双向词级KL正则相结合，有效解决了DPO在医学中奖励离散、分布偏移和视觉无关问题。

**🔧 技术方法**

使用了DPO、RRPO、双向KL正则、视觉对比损失、LoRA微调以及MedSAM等技术。

**📊 数据集**

实验数据集包括VQA‑RAD、SLAKE、IU‑Xray（用于报告生成）以及VGMED（用于视觉定位评估）。

**📈 对比分析**

与基线DPO、mDPO、RRPO、MASK‑DPO及公开方法比较，平均提升约10%‑13%，在视觉定位上显著提高注意力比率和JS散度。

**⚠️ 局限性**

局限性包括对外部生成器和分割模型的依赖、仅在放射学领域验证、对罕见病或不同影像模态的泛化性不足。

---

## 78. Perceive, Interact, Reason: Building Tool-Augmented Visual Agents for Spatial Reasoning

**arXiv ID:** 2606.12830 | [PDF](https://arxiv.org/pdf/2606.12830v1)

**作者:** Changye Li `[一作]` (Tsinghua University), Ligeng Zhu `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种工具增强的视觉空间推理代理框架（名称为-8B），通过感知-交互-推理流程实现对地图、视觉探测、视觉重建等多领域空间推理任务的解答。

**💡 创新点**

创新点在于：①设计了两类互补工具（视觉感知工具与视觉交互工具）并构建统一的工具沙箱；②通过监督式轨迹合成提升模型的工具调用可靠性；③提出Observation-Relaxed Group-in-Group Policy Optimization（OR‑GIGPO）实现对多步工具使用的细粒度信用分配。

**🔧 技术方法**

核心技术包括：工具增强推理、轨迹合成式监督微调、强化学习中的OR‑GIGPO、以及多任务奖励设计（重复惩罚、格式奖励、正确性奖励）。

**📊 数据集**

使用了13个基准测试，覆盖8个数据集，包括MapTrace、ReasonMap、ReasonMap-Plus、Visual Probing、Ball Tracking、Paper Folding、Cube Three-View Reasoning、Real-world Spatial Reasoning、V*、MapEval、BabyVision等。

**📈 对比分析**

与三类基线对比：专有VLM（GPT‑5、Gemini等）、开源VLM（Qwen系列、InternVL3.5等）以及工具/推理集成VLM（VTool-R1、Mini‑o3等）。-8B在13项任务上平均提升7.0%–14.8%，在内部分布任务提升10.0%，外部分布提升4.4%，与规模更大（235B）专有模型和GPT‑5接近。

**⚠️ 局限性**

局限性包括：依赖预先设计的工具集合，难以自动生成新工具；轨迹合成与强化学习训练仍需人工工程；在某些极端视觉复杂度高的任务中性能仍落后于更大规模模型；对工具调用的错误仍可能导致推理失败。

---

## 79. Sparse2Act: Learning Action-Aligned Sparse 3D Representations for Cross-Domain Robot Manipulation

**arXiv ID:** 2606.12759 | [PDF](https://arxiv.org/pdf/2606.12759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 80. Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models

**arXiv ID:** 2606.12689 | [PDF](https://arxiv.org/pdf/2606.12689v1)

**作者:** Darpan Aswal `[一作]` (Université Grenoble Alpes), Maxime Peyrard `[通讯]` (Université Grenoble Alpes)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估并解释了隐藏思维模型（Latent Reasoning Models, LRMs）中连续内部状态的作用，揭示了可观测的思维模式并非必然反映推理机制，而是需通过因果干预与控制模型验证其实际影响。

**💡 创新点**

创新点在于：①首次系统区分观测可读模式与真正的因果影响；②将“思维使用”视为连续可测的因果强度，而非二元是否使用；③引入梯度子空间干预和几何稳定性分析，定位低秩、loss敏感的因果子空间；④提出因果优先的解释流程，强调对LRMs解释时需要匹配对照和因果测试。

**🔧 技术方法**

主要技术包括：自回归连续思维模型（如Coconut、CODI、PaT）、因果追踪（Causal Tracing）、梯度子空间干预（Gradient‑Subspace Intervention）、信息熵和KL回溯评估、线性与非线性子空间投影、马尔可夫性与几何相似度测度。

**📊 数据集**

使用的数据集包括图跳跃任务 ProsQA 进行图形推理，以及算术推理任务 GSM8k，用于测试模型在不同推理维度下的行为。

**📈 对比分析**

与基准模型（Base GPT‑2、Explicit‑CoT）以及多种对照模型（PaT、C_u、CODI）对比，发现LRMs在观测层面与对照模型产生相似的 BFS 与 Scratchpad 模式，但在因果层面显示：图形推理中思维几乎无因果作用，而算术推理中思维在低秩子空间中具有显著影响；性能方面，思维干预对算术推理的准确率产生明显下降，而对图跳跃任务影响有限。

**⚠️ 局限性**

局限性包括：①梯度子空间估计为线性，可能忽略非线性因果结构；②因果干预受限于局部、分布偏移与多解释可能性；③仅针对小规模模型与有限的LRM变体，缺乏对更大模型与更近期推理范式的验证；④未探究思维子空间在训练过程中的形成与演化。

---

## 81. GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis Across Frontier and Open-Weight Foundation Models

**arXiv ID:** 2606.12821 | [PDF](https://arxiv.org/pdf/2606.12821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 82. A Zero-shot Generalized Graph Anomaly Detection Framework via Node Reconstruction

**arXiv ID:** 2606.12673 | [PDF](https://arxiv.org/pdf/2606.12673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 83. Bounding Boxes as Goals: Language-Conditioned Grasping via Neuro-Symbolic Planning

**arXiv ID:** 2606.12910 | [PDF](https://arxiv.org/pdf/2606.12910v1)

**作者:** Allison Andreyev `[一作]` (University of Maryland), Romel Gomez `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e0540dec-d77f-42db-94ae-d039248f6393` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GRASP框架，利用预训练的VLM将自然语言指令转换为神经符号目标状态，并通过边界框检测与闭环控制实现无监督的桌面抓取与放置。

**💡 创新点**

创新点在于：①将语言解析与目标状态生成与闭环控制解耦，形成轻量级神经符号流水线；②使用边界框相似度进行实时目标评估，无需训练策略；③通过比例RPY控制和指数平滑实现稳定的抓取。

**🔧 技术方法**

技术包括GPT-5.2（或其他LLM）进行语言解析生成JSON目标状态，GroundingDINO用于边界框检测，IoU+距离相似度评估，比例RPY控制与指数平滑，以及阈值死区去抖动。

**📊 数据集**

使用自制桌面抓取场景中的物体集合（如棕色方块、蓝色瓶子、螺丝刀等）进行实验，没有使用公开大型数据集；数据来源为实验室摄像头实时抓取的图像。

**📈 对比分析**

与开环对比、随机/首选目标策略、以及无平滑/死区的消融实验进行评估；在三难度级别（易/中/难）上，GRASP总体成功率为73.33%，各级别分别为86.67%、76.67%、56.67%；消融实验表明闭环控制、平滑/死区和最高logit目标选择是提升性能的关键。

**⚠️ 局限性**

局限性包括：①对检测器（GroundingDINO）误检/漏检敏感；②硬件视场和图像质量限制导致难度高时成功率下降；③未实现完整的多目标、多区域排序与放置评估，仅评估单个抓取动作；④缺乏在更复杂动态环境中的鲁棒性验证。

---

## 84. PiDA: Phonetically-Informed Data Augmentation for Robust Vietnamese Speech Translation

**arXiv ID:** 2606.12911 | [PDF](https://arxiv.org/pdf/2606.12911v1)

**作者:** Giang Son Nguyen `[一作]` (VinUniversity), Dung D. Le `[通讯]` (VinUniversity)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对越南语语音翻译中的 ASR 错误进行系统分类，并基于此提出 Phonetically‑Informed Data Augmentation (PiDA) 生成符合语音错误分布的文本噪声。

**💡 创新点**

首次用音位相似度对 ASR 替换错误进行细分，证明大多数错误源自结构化的音位混淆；并利用 XPhoneBERT 的音位嵌入来生成逼真的语音错误，从而在不降低 MT 质量的前提下显著提升 ST 性能。

**🔧 技术方法**

线性混合效应模型（LMM）评估错误类型对翻译质量的影响；XPhoneBERT 提取音位嵌入；PiDA 通过近邻检索与温度软最大采样实现音位级替换；与随机、真实噪声、MEDSAGE 等对比。

**📊 数据集**

主要使用 FLEURS 越南语–英语数据集（训练 3k，测试 0.9k），并对 MultiMed‑ST 进行初步评估以验证数据质量。

**📈 对比分析**

与“清洁”对齐、随机频率替换、真实噪声和 MEDSAGE 的 Fine‑tune 进行对比；PiDA 在 PhoWhisper‑large 上提升 BLEU_ST 约 +2.04 分，在 wav2vec2‑base 上提升约 +0.78 分，同时保持或提升 MT 的 BLEU/COMET，优于随机替换且不出现 MT 退化。

**⚠️ 局限性**

实验仅在单一数据集上进行；对跨语言 OOV 错误的处理不完善；未评估更大规模或不同方言的泛化能力。

---

## 85. The Challenges of Balancing AI Compliance and Technological Innovations in Critical Sectors: A Systematic Literature Review

**arXiv ID:** 2606.12423 | [PDF](https://arxiv.org/pdf/2606.12423v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 86. Deep Unfolded Latent Optimally Partitioned-l2/l1 Networks for Data-driven Block-Sparse Recovery

**arXiv ID:** 2606.12740 | [PDF](https://arxiv.org/pdf/2606.12740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 87. CAPED: Context-Aware Privacy Exposure Defense for Mobile GUI Agents

**arXiv ID:** 2606.12666 | [PDF](https://arxiv.org/pdf/2606.12666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 88. Assessing Student Ability to Select an Algorithmic Paradigm

**arXiv ID:** 2606.12417 | [PDF](https://arxiv.org/pdf/2606.12417v1)

**作者:** Dip Kiran Pradhan Newar `[一作]` (Utah State University), Seth Poulsen `[通讯]` (Utah State University)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5031075692)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了一套多项选择的算法范式选择评估工具（APSA），用于衡量学生在算法设计问题中选择适当范式的能力。

**💡 创新点**

首次构建并发布了一种可靠的评估工具，并提出了编写范式选择题目的准则，填补了缺乏此类验证测评工具的空白。

**🔧 技术方法**

采用经典测验理论（CTT）计算Cronbach α、题目难度和区分度指标，并通过问卷式多项选择题进行数据采集。

**📊 数据集**

收集了来自两所美国高校（Utah State University 与 University of California, Irvine）算法设计课程学生的成绩数据，版本1共28人，版本2共304人。

**📈 对比分析**

通过比较版本1和版本2的Cronbach α（0.49→0.73）以及题目难度/区分度分布，证明修订后问卷在可靠性和有效性上显著提升。

**⚠️ 局限性**

研究样本仅来自两所高校，缺乏更广泛的学生群体和人口统计信息，需进一步验证其普适性。

---

## 89. Testing Theory of Truly Concurrent Processes

**arXiv ID:** 2606.12944 | [PDF](https://arxiv.org/pdf/2606.12944v1)

**作者:** Yong Wang `[一作]` `[通讯]`, Yong Wang

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种新的进程代数体系，包含Ω、递归与抽象操作符，并给出了其运算语义、递归语义以及接受树的定域语义，最终证明三种语义（运算语义、定域语义与不等式证明系统）之间的三位一体一致性与完全抽象性。

**💡 创新点**

创新点在于：① 将Omega（永远不可执行的行为）与递归统一到同一进程代数框架；② 构造了新的接受树（强、弱与强接受树）来精确描述并发、通信与冲突等操作；③ 通过引入Ω规则与递归诱导，建立了不等式证明系统的完整性与完全抽象性；④ 提出了头正常形式与Omega-正常形式的归约定理，进一步实现了语法与语义的一致归约。

**🔧 技术方法**

采用的技术包括：SOS（结构化规约系统）构造运算语义；域理论（cpo、固定点与连续映射）构造接受树定域；不等式证明系统（ω-诱导、递归诱导、Scott诱导）用于演绎不等式；以及归约与完全抽象性理论来链接语法与语义。

**📊 数据集**

本文没有使用传统意义上的数据集；所有结果均通过形式化证明与理论推导得到，所给出的接受树与证明系统在标准的进程算子和递归定义上均成立。

**📈 对比分析**

方法比较主要以理论证明方式进行，作者通过证明不同语义下的不等式预序与接受树的部分顺序相等来比较；在可达性与收敛性方面，提出的接受树模型与运算语义给出了完全抽象的语义等价关系，证明其与ω-诱导证明系统的等价性，未给出实验性性能评估。

**⚠️ 局限性**

限制主要体现在：① 递归语义与不等式系统的完整性需要Ω规则与递归诱导的辅助，部分递归方程仍可能缺乏唯一固定点；② 由于所有证明均为形式化推导，缺乏实现工具与可实验验证；③ 仅对受限（guarded）递归保证唯一固定点，开放递归的通用性仍待进一步研究。

---

## 90. DynamicPTQ: Mitigating Activation Quantization Collapse via Residual-Stream Dynamics

**arXiv ID:** 2606.12487 | [PDF](https://arxiv.org/pdf/2606.12487v1)

**作者:** Zimo Zhao `[一作]` (City University of Hong Kong), Xiangyu Zhao `[通讯]` (City University of Hong Kong)

**通讯引用:** 6618 | [OpenAlex ID](https://openalex.org/A5100645854)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DynamicPTQ，一种基于残差流动态的混合精度激活量化策略

**💡 创新点**

通过跳跃比（Jump Ratio）和历史特征 SNR（Historical Feature SNR）识别4‑bit激活失稳层，并仅为这些层提升到8‑bit精度

**🔧 技术方法**

利用旋转变换（QuaRot/SpinQuant/FlatQuant）等 PTQ 骨干与自定义精度分配实现

**📊 数据集**

在 LLaMA‑2/3 系列模型上，使用 WikiText‑2、C4 以及六个零样本 QA 数据集进行评估

**📈 对比分析**

相较于原始 4‑bit PTQ，在 perplexity 上降低 0.2–1.9 点、零样本 QA 平均提升 0.4–0.8 分，同时保持 1.05× 的吞吐量提升

**⚠️ 局限性**

在已非常强健的基线上提升有限，并且仅在预先确定的层级进行精度分配，未实现更细粒度的自适应调度

---

## 91. A Multiplexing Design Space: Theory, Method, and Application

**arXiv ID:** 2606.12719 | [PDF](https://arxiv.org/pdf/2606.12719v1)

**作者:** Yiwen Xing `[一作]` (University of Oxford), Min Chen `[通讯]` (University of Oxford)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于理论指导的方法，系统地构建了一个多路复用设计空间，以支持在机器学习工作流中可视化与偏微分方程（PDE）相关的数据。

**💡 创新点**

创新点在于引入了一个预设计步骤，通过信息理论的成本效益分析来分析可视化需求，并提出了一个三步法来系统地探索与特定应用相关的多路复用设计空间。

**🔧 技术方法**

使用了信息理论框架和多路复用现象的分类，结合领域专家的协作过程来指导设计。

**📊 数据集**

使用了与偏微分方程相关的机器学习工作流中的多种数据集，包括1D空间加时间的数据和各种性能指标。

**📈 对比分析**

通过与领域专家的协作，评估了不同设计选项的有效性，发现多路复用设计能够更好地支持对比分析，尤其是在处理多个热图时，能够减少认知负担并提高信息传达的有效性。

**⚠️ 局限性**

限制在于多层叠加的可视化可能导致信息密度降低，过多的视觉层叠加会降低可解释性，且领域专家对某些设计的接受度可能受到限制。

---

## 92. From Imitation to Alignment: Human-Preference Flow Policies for Long-Horizon Sidewalk Navigation

**arXiv ID:** 2606.12603 | [PDF](https://arxiv.org/pdf/2606.12603v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 93. Orchestrating the Twin Transition in Multinational Corporations: Technology Roadmapping for Green and Digital Global Business Services

**arXiv ID:** 2606.12787 | [PDF](https://arxiv.org/pdf/2606.12787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 94. AI Debris: Residual Risk and the Afterlife of Failed AI Systems

**arXiv ID:** 2606.12432 | [PDF](https://arxiv.org/pdf/2606.12432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 95. Adaptive Weighted Averaging

**arXiv ID:** 2606.12763 | [PDF](https://arxiv.org/pdf/2606.12763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 96. SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning

**arXiv ID:** 2606.12808 | [PDF](https://arxiv.org/pdf/2606.12808v1)

**作者:** Yash Vardhan Tomar `[一作]` (Purdue University), Vaneet Aggarwal `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于强化学习的低延迟自适应哈密顿量学习方法，利用离线训练的策略快速决定测量实验，同时保持贝叶斯后验更新。

**💡 创新点**

创新在于将贝叶斯实验设计的高成本后验条件评分搬到离线学习阶段，形成可在数毫秒内完成的策略前向推理，显著降低在线采样决策延迟。

**🔧 技术方法**

使用了顺序蒙特卡罗（SMC）贝叶斯更新、近似策略梯度（PPO）强化学习、变分自编码器、图卷积层和Transformer。

**📊 数据集**

在一维转置场 Ising 模型（TFIM）中生成的随机参数（2N‑1 维）作为任务分布；通过模拟的 128–512 次测量样本进行训练与评估。

**📈 对比分析**

与固定循环、优化固定调度、DAD‑Style Transformer、有限 Fisher 信息搜索和有限两步 BALD 在线采集进行对比；在 8–12 量子比特的尺度下，所提策略在保持与基线相近的均方误差的同时，决策延迟提升 47–72 倍，整个采样步骤在 12 量子比特时仅 1.02 s 而基线需 13–22 s。

**⚠️ 局限性**

限制在于对非链图、偏移先验、复杂噪声模型的泛化尚未充分验证；奖励与最终 MSE 的对齐不足，未来需采用更贴合最终误差的奖励并压缩模型以实现微秒级控制。

---

## 97. Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU

**arXiv ID:** 2606.12765 | [PDF](https://arxiv.org/pdf/2606.12765v1)

**作者:** Ramchand Kumaresan `[一作]` `[通讯]` (Murai Labs), Ramchand Kumaresan (Murai Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Apple M4 Max 上 Metal 4.1 的张量计算路径进行实证性逆向工程，阐明 fp8（E4M3）是被软件模拟而非硬件加速；确定该路径在 GPU 着色器核心上执行，未使用专用矩阵单元；重建 8×8 的片段布局；并基于此实现了 GEMM+Bias+GELU 的融合内核，在缓存驻留区提升 6.5%‑12.9% 的吞吐量。

**💡 创新点**

首次公开揭示 Metal 4.1 在真实硬件上的隐式行为，包括 fp8 的执行方式、累加器宽度、片段布局以及版本门控细节；提出 checksum‑gate、 provenance‑track 等可靠性防护机制；构建可复现的微基准框架，提供 MIT 许可证下的代码与 CSV 数据。

**🔧 技术方法**

使用 Objective‑C++ 与 Python 结合的双语言微基准框架；checksum‑gate 验证精度；对齐约束与 SFINAE 检测；基准比较包括 fp16 与 fp8 直接 GEMM、MPP 与传统线程组 tiled GEMM、Legacy 指令等；利用 GPU 电源计数器做功耗归因。

**📊 数据集**

未使用公开的机器学习数据集，而是采用合成矩阵乘法和随机量化数据，保证可重现性与对比性。

**📈 对比分析**

比较方法：多种基线（标量 ALU、naïve tiled GEMM、Legacy 指令、FP16 MPP 路径）并要求超出最强基线至少 1.10×；fp8 通过与 fp16 对比衡量加速阈值 1.5×。结果显示：fp8 仅 0.94× fp16，未实现加速；MPP 在相对基线上提升 2.9–5.5×；在 Legacy 指令上仅 1.05–1.21×。融合内核相较于拆分路径提升 6.5%‑12.9%。

**⚠️ 局限性**

研究仅针对单颗 M4 Max，跨代适用性未知；Beta 系统阻断了 ANE 能耗计量，导致 ANE 路径归因间接；FlashAttention 的实验为负面结果，未必能推广；实验基于合成工作负载，未验证在实际 LLM/视觉模型中的表现。

---

## 98. The Khipu Problem: Institutional Legibility Under Distributed Cognition

**arXiv ID:** 2606.12414 | [PDF](https://arxiv.org/pdf/2606.12414v1)

**作者:** Krti Tallam `[一作]` `[通讯]` (KamiwazaAI), Krti Tallam (KamiwazaAI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文通过理论分析和最小化形式化，提出了分布式AI治理中的“khipu问题”，并阐述了从模型中心到分布式认知事件的转变、解释连续性与记录保留的区别，以及治理工作空间与多层治理单元的设计思路。

**💡 创新点**

创新点在于：①首次把分布式AI的治理视角从“记录保留”转为“解释连续性”，揭示记录存活但可读性衰退的风险；②将“分布式认知事件”作为治理对象，区别传统的单一模型或组件；③提出了“治理工作空间”和“跨单元可遍历性”作为保证解释连续性的技术方案；④引入了结构化的“认知不充分性”分类（缺失证据、模糊证据、结构不可读证据）。

**🔧 技术方法**

技术上主要采用概念框架构建、最小化形式化（如分布式认知事件的向量表示）、以及治理工作空间设计（包含请求者真值、权限路径、边界真值、证据范围、运行时谱系、结果收据等结构化字段）。

**📊 数据集**

未使用任何实际数据集；本文为理论与设计研究，不涉及实验数据。

**📈 对比分析**

由于是理论性工作，未进行实验对比或性能评估；文中所述方案需在实际系统中实现后才能评估其治理效果与性能。

**⚠️ 局限性**

局限性包括：①缺乏实现细节与实验验证；②对治理工作空间的实现方式、技术选型及性能影响尚未明确；③在不同组织与法规环境下的可适配性与可操作性需要进一步探讨；④未给出具体指标衡量解释连续性成功与否，依赖后续研究完善评估方法。

---

## 99. Small LLMs for Biomedical Claim Verification: Cost-Effective Fine-Tuning, Structural Dataset Shortcuts, and Cross-Domain Generalization

**arXiv ID:** 2606.12854 | [PDF](https://arxiv.org/pdf/2606.12854v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 100. The Capacity Region for Classes of Sum-Broadcast Channels

**arXiv ID:** 2606.12839 | [PDF](https://arxiv.org/pdf/2606.12839v1)

**作者:** Amin Gohari `[一作]` (Chinese University of Hong Kong), Chandra Nair `[通讯]` (Chinese University of Hong Kong)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

计算了由若干退化、弱噪声、优势、确定性或半确定性分量构成的求和广播通道的容量区域。

**💡 创新点**

利用辅助接收机外界界，使其与Marton内界匹配，首次证明了在更宽泛的求和广播通道类中，Marton内界是紧确的，并定义了新的“主类”广播通道。

**🔧 技术方法**

信息理论的辅助接收机外界界、最大化加权和速率、对称求和与对偶性等技术。

**📊 数据集**

无，纯理论分析。

**📈 对比分析**

与传统的UVW外界界对比，证明在主类通道上两者相等，给出了容量的精确表达式；在非主类通道中UVWS外界界严格大于Marton内界，展示了差距。

**⚠️ 局限性**

结果仅适用于组件属于主类或满足更严格条件的求和通道；对一般求和广播通道尚未给出容量；对更多通道类的闭合性未证明。

---

## 101. LLM-Powered Personalized Glycemic Assessment in Type 2 Diabetes with Wearable Sensor Data

**arXiv ID:** 2606.12699 | [PDF](https://arxiv.org/pdf/2606.12699v1)

**作者:** Yifan Gao `[一作]` (University of Texas at San Antonio), Yuanxiong Guo `[通讯]` (University of Texas at San Antonio)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了GlyLLM框架，利用大型语言模型结合可穿戴传感器数据和个体静态元数据进行个性化血糖评估。

**💡 创新点**

首次将预训练LLM与ViT传感器编码器融合，直接在LLM内部实现多模态语义抽象，实现对长时序CGM数据的精准预测和糖尿病分型。

**🔧 技术方法**

采用ViT作为传感器编码器、LoRA微调LLM（如Llama3-Med42、Gemma-2、Mistral），并使用多模态文本提示模板。

**📊 数据集**

在AI-READI v2.0.0糖尿病数据集上进行实验，包含CGM、心率、呼吸率等传感器和问卷、实验室指标等静态信息。

**📈 对比分析**

与传统Transformer基模型和零/少量示例提示的LLM对比，GlyLLM在血糖预测RMSE平均降低约13.66%，在糖尿病分型AUROC提升约13.08%，并在iGlu-CE等临床指标上亦显著优于基线。

**⚠️ 局限性**

缺乏真实临床验证，仅在单一公开数据集上测试，且对其他传感器或多模态数据融合的效果尚未评估。

---

## 102. Bounds and Constructions of Maximum Toroidal Distance Codes

**arXiv ID:** 2606.13008 | [PDF](https://arxiv.org/pdf/2606.13008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 103. Random Proposals: A Softmax-Based Local-Improvement Framework for Maximum Weighted Matching

**arXiv ID:** 2606.12692 | [PDF](https://arxiv.org/pdf/2606.12692v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 104. A Machine Learning Framework for Real-Time Personalized Ergonomic Pose Analysis

**arXiv ID:** 2606.12988 | [PDF](https://arxiv.org/pdf/2606.12988v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 105. Occupational Prompting Reveals Cultural Bias in Large Language Models

**arXiv ID:** 2606.12443 | [PDF](https://arxiv.org/pdf/2606.12443v1)

**作者:** Maksim E. Eren `[一作]` (Los Alamos National Laboratory), Eric Michalak `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5022150012)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在大语言模型中使用职业提示而非国籍提示，探究职业身份如何影响模型对价值调查问卷的回答，并将回答投射到 Inglehart–Welzel 文化空间。

**💡 创新点**

创新点在于将职业提示引入已有的基于调查问卷的文化偏差评估框架，揭示职业身份能在保持西方主导的同时在文化地图上产生结构化偏移。

**🔧 技术方法**

使用了基于 Integrated Values Surveys（IVS）十项价值问卷题目的两维文化空间投射、主成分分析与 Varimax 旋转、欧氏最近邻分类等技术。

**📊 数据集**

采用了 IVS 的十项价值问卷题目和由 LLM 生成的包含 234 个职业及其元数据的职业列表作为实验数据。

**📈 对比分析**

对五个开源 LLM（Llama 3.3、Llama 4、Gemma 3、GPT‑OSS 20B、GPT‑OSS 120B）进行相同提示实验，比较不同职业和模型在文化空间中的坐标与文化区域的距离，结果显示职业提示能产生多样化但仍集中在西方偏右的文化分布。

**⚠️ 局限性**

局限在于职业集成依赖 LLM 生成，缺乏真实职业分布；仅用短问卷回答无法覆盖长文本推理；投影结果反映模型内部结构而非真实职业文化价值。

---

## 106. Stubborn: A Streamlined and Unified Reinforcement Learning Framework for Robust Motion Tracking and Fall Recovery for Humanoids

**arXiv ID:** 2606.12814 | [PDF](https://arxiv.org/pdf/2606.12814v1)

**作者:** Xiao Ren `[一作]` (Southern University of Science and Technology), He Kong `[通讯]` (Southern University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研发了一个统一的 RL 框架 Stubborn，能够让单一策略同时完成高动态动作跟踪与跌倒恢复。

**💡 创新点**

创新点在于：①引入概率终止机制让策略在失稳时继续探索恢复动作；②采用 yaw 对齐跟踪表示降低对全局漂移的敏感性；③基于跟踪误差的自适应采样权重动态调整训练数据分布。

**🔧 技术方法**

使用的技术包括：非对称 Actor‑Critic 架构、Bernoulli 概率终止、跟踪误差驱动的自适应采样、仿真-实机共训练与硬件加速。

**📊 数据集**

使用的数据集为公开的 LAFAN1（以及 AMASS 进行实机动作重现），并在 29-DoF Unitree G1 机器人上进行实验。

**📈 对比分析**

通过与 HoloMotion、Any2Track、BFM‑Zero 及从零多动作 RL 基线进行对比，Stubborn 在 MPBPE/MPJPE/MPJVE/Δacc 等指标上均表现更好；跌倒恢复成功率达到 100%，且训练收敛速度快。

**⚠️ 局限性**

局限性在于：恢复动作的机动性与平滑度仍有限，且尚未在更复杂地形和极端动态任务中进行充分验证。

---

## 107. Beyond Problem Solving: UOJ-Bench for Evaluating Code Generation, Hacking, and Repair in Competitive Programming

**arXiv ID:** 2606.12864 | [PDF](https://arxiv.org/pdf/2606.12864v1)

**作者:** Tingqiang Xu `[一作]` (Tsinghua University), Kaifeng Lyu `[通讯]` (Tsinghua University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了UOJ-Bench基准，评估LLM在代码生成、破解（hacking）与修复（repair）三大任务上的表现，重点关注隐藏错误的检测与修复。

**💡 创新点**

首次在真实竞赛在线评测环境下系统评估LLM的代码推理能力，设计了可区分显性与隐性错误的Easy/Hard层级，并引入Zero-Day-Hacking 5K场景。

**🔧 技术方法**

采用多模态LLM（如GPT-OSS-120B、Gemini-3-pro-preview等）进行一轮与多轮推理，结合ReAct式交互式 agentic 评估，并使用UOJ的原生评测接口。

**📊 数据集**

数据来源于中国主流社区驱动在线评测平台UOJ，收集了约672道竞赛题目、479/1046道破解样例、500/216道修复样例，并构建了5,060条Zero-Day-Hacking测试。

**📈 对比分析**

与主流生成基准（CodeContests、TACO、AetherCode等）以及自定义评测系统对比，使用Pass@1和Pass@k指标；在Direct评估下最高模型Gemini-3-pro-preview仅能达到约38%整体生成成功率；在破解/修复任务中，单轮测试<50%，但通过Test‑Time Scaling可提升至90%以上，Cost与算力仍是瓶颈。

**⚠️ 局限性**

当前LLM在识别隐性错误方面仍距人类专家远，单轮表现低于50%，且多轮推理成本高昂；评测受限于UOJ可用题库和缺乏跨平台泛化；缺乏针对自定义检查器的高效策略；数据污染与时间漂移对生成任务仍有影响。

---

## 108. ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs

**arXiv ID:** 2606.12451 | [PDF](https://arxiv.org/pdf/2606.12451v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 109. Navigating the muddy waters of bias in artificial intelligence research: Understanding divergent meanings and conceptions

**arXiv ID:** 2606.12421 | [PDF](https://arxiv.org/pdf/2606.12421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 110. Prefill Awareness in Large Language Models

**arXiv ID:** 2606.12747 | [PDF](https://arxiv.org/pdf/2606.12747v1)

**作者:** Andy Wang `[一作]` (University of Wisconsin-Madison), Robert Kirk `[通讯]` (UK AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大语言模型是否能够识别并抵御在助手回话中被外部插入或编辑的prefill文本。

**💡 创新点**

创新点在于首次系统地定义并量化“prefill awareness”，同时区分检测（识别偏离自身生成的文本）与抵抗（不沿着prefill轨迹继续）的两种能力，并将其应用于多种真实评估场景。

**🔧 技术方法**

技术手段包括：① 设计三种prefill攻击方式（思考预填、直接回答预填、历史回合预填）；② 构建 1527 条二元偏好问题的基准，并使用 GPT‑5 评判模型回答的一致性；③ 通过提示工程实现检测与抵抗的测量，使用统计检验（Wilson 置信区间、回归分析）评估效果；④ 在多模型（Claude Opus 4.5、Claude Haiku 4.5、GPT‑5.2 等）上进行跨机制对比。

**📊 数据集**

数据集主要包括：自建的二元偏好问答集（7 个变体，约 10700 条样本）；SWE‑bench、Petri（有机体与聊天）生成的 17,835 条长轨迹；以及从 OASST1、HH‑RLHF 等公开数据中抽取的 2,772 条控制性偏移会话。

**📈 对比分析**

评估方法：在每种 prefilling 机制下，对每条样本计算检测率（balanced accuracy）和抵抗率；对比 8 个 frontier 模型。Claude Opus 4.5 在“思考预填”下检测准确率达 67.6%、抵抗率 48.6%，同类模型检测率范围 9–35%；检测与抵抗相关性弱（r≈0.08）。在真实评估中，模型对被预填的助手回合的“非自创”判断准确率因数据集和任务不同而差异显著。总体上，prefill awareness 真实存在，但效果随模型、预填方式和环境显著变化。

**⚠️ 局限性**

局限性：仅基于行为评测，未揭示内在机制；受 API 支持和稳定偏好数量限制；思考预填在闭源模型上不完全可行；检测结果易被文本风格、任务成功度、格式化标签等表面线索误导，需进一步去除伪信号。

---

## 111. SMSR: Certified Defence Against Runtime Memory Poisoning in Persistent LLM Agent Systems

**arXiv ID:** 2606.12703 | [PDF](https://arxiv.org/pdf/2606.12703v1)

**作者:** Tarun Sharma `[一作]` `[通讯]`, Tarun Sharma

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SMSR——一种针对运行时内存中毒的正式认证防御方案，结合写时签名与查询时随机抽样；

**💡 创新点**

首个为多会话内存中毒提供可证明的安全边界，并证明无源自写时出处的检索时过滤无法获得正式保证；

**🔧 技术方法**

采用HMAC‑SHA256写时原始标记、随机化检索抽样、投票式裁决和置信度分析（超几何分布）来实现；

**📊 数据集**

在15个企业知识库场景上进行评估，使用10条签名种子记忆和多种攻击注入；

**📈 对比分析**

对比无防御、启发式过滤、仅写时签名以及完整SMSR；在生产规模（20条种子）下，SMSR将攻击成功率从93–100% 降至约8%（95% CI [5.8%，10.9%]），并在查询‑仅攻击下从65.3% 降至5.3%；

**⚠️ 局限性**

局限在于认证键管理、对大规模攻击预算（t≈m/2）时安全边界趋近1；需要额外的查询开销（5倍 API 调用），且仅保护检索层，未涵盖模型参数或逆向提取攻击。

---

## 112. Token Complexity Theory for AI-Augmented Computing

**arXiv ID:** 2606.12647 | [PDF](https://arxiv.org/pdf/2606.12647v1)

**作者:** Jie Wang `[一作]` `[通讯]` (University of Massachusetts), Jie Wang (University of Massachusetts)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“令牌复杂度”这一新度量，用于量化AI系统调用大型语言模型时的令牌成本，并构建了相应的AI-Oracle图灵机模型；

**💡 创新点**

首次把令牌成本纳入复杂度理论，定义了令牌复杂度与质量阈值、价格比的关系，并证明了其单调性、凸性和价格敏感性；

**🔧 技术方法**

使用了概率图灵机、分布式随机过程理论以及经典复杂度理论的工具，构建了令牌计数、成本函数和复杂度前沿；

**📊 数据集**

未使用具体数据集，主要通过理论构造与证明来验证框架；

**📈 对比分析**

没有实验比较，文中通过数学证明展示了令牌复杂度的性质和前沿的凸性，未给出数值性能指标；

**⚠️ 局限性**

局限在于缺乏实证验证、对实际LLM行为的细粒度建模有限，以及质量上限、下界等关键问题仍未解决。

---

## 113. Maestro: Workload-Aware Cross-Cluster Scheduling for LLM-Based Multi-Agent Systems

**arXiv ID:** 2606.12950 | [PDF](https://arxiv.org/pdf/2606.12950v1)

**作者:** Jinghao Wang `[一作]` (Beihang University), Renyu Yang `[通讯]` (Beihang University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个针对LLM多代理系统的工作负载感知跨集群调度框架，实现了多层模型权重驻留、KV缓存弹性分配、预测驱动的节点调度和基于工作流的优先级调度。

**💡 创新点**

创新点在于将代理角色、工具调用意图、输出长度预测与KV占用预测融合，构建分层调度策略；通过预测驱动的跨集群适配、弹性KV管理以及工作流感知的SRTF预占，显著降低GPU内存占用并提升SLO达成率。

**🔧 技术方法**

使用了轻量化二阶段预测器（工具意图分类 + 长度回归）、基于vLLM与PagedAttention的KV虚拟内存、LRU多层权重驻留、CUDA虚拟内存管理、跨集群fitness score、边界预占和剩余时间优先调度等技术。

**📊 数据集**

实验数据来自9个真实LLM‑MAS应用（交互式与批处理）收集的工作流日志与合成测试，包含46,769个作业、144,524个LLM调用；输入数据使用公开数据集如SQuAD、MATH、OpenCSG等。

**📈 对比分析**

通过与FCFS、EDF、Oracle‑SRTF、No‑Colocation、QLM等基线比较，实验显示KV预留HBM降低67.2%，SLO达成率提升23.6个百分点，交互队列延迟下降84.8%，GPU利用率更高，内存过度分配约3.05倍。

**⚠️ 局限性**

限制在于仅验证同构A100节点、两类服务级别；未覆盖异构加速器、多租户公平性；外部工具调用调度未集成；对硬件资源约束与调度公平性仍需进一步改进。

---

## 114. On the Limits of Performance Portability in Directive-Based GPU Programming

**arXiv ID:** 2606.12753 | [PDF](https://arxiv.org/pdf/2606.12753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 115. Bridging Modal Isolation in Interleaved Thinking: Supervising Modality Transitions via Stepwise Reinforcement

**arXiv ID:** 2606.12886 | [PDF](https://arxiv.org/pdf/2606.12886v1)

**作者:** Tingyu Li `[一作]` (Shanghai Artificial Intelligence Laboratory), Cheng Tan `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对统一多模态模型（UMM）在长链视觉推理中的“模态隔离”问题，提出了模态转换损失（Modality Transition Loss）以及两阶段训练框架MoTiF，分别通过反射式SFT和Flow‑GRPO提升文本-图像之间的互信息。

**💡 创新点**

创新点在于将模态隔离拆解为文本→图像的跨模态幻觉与图像→文本的利用缺失两项，并通过自监督的模态转换损失直接优化边界信息流，从而突破传统只关注最终任务准确率的局限。

**🔧 技术方法**

技术方法包括：模态转换损失的定义与计算、基于VLM‑as‑Judge的二元评判器、反射式SFT训练策略（误差检测与纠正）、Flow‑GRPO强化学习在图像生成中的应用以及联合优化的两阶段训练流程。

**📊 数据集**

使用了四个视觉推理基准：Sokoban、Maze、Multi‑hop Manipulation 和 Ball Tracking，数据集来自 Game‑RL、CLEVER、RBench‑V，并通过规则求解器生成高质量的文本-图像交互训练链。

**📈 对比分析**

与前沿多模态模型（Gemini3.5‑Flash、Gemini3.1‑Flash‑Lite 等）以及同等规模的开源模型（Qwen3.5‑27B、Gemma‑4‑31B、Bagel‑7B‑MoT 等）比较，MoTiF 在所有任务上平均提升约 45% 的整体准确率，单项任务提升可达 60% 以上。

**⚠️ 局限性**

局限性包括：需要专门设计的 VLM‑as‑Judge 评判器，训练过程对误差模拟和评判质量敏感；两阶段训练可能产生兼容性冲突，导致一定程度的灾难性遗忘；目前仅在有限的视觉推理任务上验证，尚不清楚在更广泛场景和更大模型规模下的通用性。

---

## 116. Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function

**arXiv ID:** 2606.12917 | [PDF](https://arxiv.org/pdf/2606.12917v1)

**作者:** Atharva Gupta `[一作]` (Birla Institute of Technology and Science), Saurabh Deshpande `[通讯]` (Aditya Birla Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本工作对TabPFN-2.5的特征维度注意力模块进行了因果机制分析，识别了不同层次的注意力头功能。

**💡 创新点**

首次在表格基础模型上实现因果机制分析，发现一个主导注意力头在不同任务中表现出层级和计算特异性。

**🔧 技术方法**

采用激活补丁（activation patching）、消融（ablation）和注意力熵度量等技术，并尝试对比激活方向进行推断时的可调性。

**📊 数据集**

使用两套合成回归数据集——Multiplication（3维）与Pairwise‑50（50维）进行实验。

**📈 对比分析**

通过补丁恢复率和消融效应（以标准差衡量）比较各头在各层的贡献，主导头在Multiplication在第0层峰值消融效应为0.076σ，在Pairwise‑50第16层峰值为0.074σ；补丁恢复显示其计算活跃层为第6层或第13层。

**⚠️ 局限性**

局限于仅两类合成任务，难以推广到更广泛的表格数据；对任务复杂度与层级关系的解释尚未系统验证；对推断时可调性的尝试未能成功，说明纯ICL模型的可调性受限。

---

## 117. Beyond Resilience -- A Conceptual Framework for Civic Ascent

**arXiv ID:** 2606.12752 | [PDF](https://arxiv.org/pdf/2606.12752v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 118. SAIGuard: Communication-State Simulation for Proactive Defense of LLM Multi-Agent Systems

**arXiv ID:** 2606.12474 | [PDF](https://arxiv.org/pdf/2606.12474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 119. LoHoSearch: Benchmarking Long-Horizon Search Agents Beyond the Human Difficulty Ceiling

**arXiv ID:** 2606.12837 | [PDF](https://arxiv.org/pdf/2606.12837v1)

**作者:** Jiarui Zhao `[一作]` (Meituan), Xi Su `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LoHoSearch 基于知识图谱的自动化搜索代理评测基准，生成 544 道结构复杂、搜索空间大的问答。

**💡 创新点**

通过知识图谱自动抽取、结构化子图并转化为自然语言问答，系统化控制搜索空间与结构复杂度，突破人类注释的难度上限。

**🔧 技术方法**

构建 Wiki 知识图谱，采用子图采样、LLM 生成、检索式验证、自动过滤与人工复核等技术。

**📊 数据集**

基于 7.62M 维基实体知识图谱，最终生成 544 道问答，覆盖 11 个领域。

**📈 对比分析**

在 LoHoSearch 上评测 11 种主流模型，最高 GPT‑5.5 仅 34.74% 正确率，且上下文管理策略提升仅 6.8%，显示该基准极具挑战性。

**⚠️ 局限性**

仅覆盖英文、评估工具和判定方式单一、答案唯一性验证依赖知识图谱、难度过滤可能偏向单一 LLM，未来需多语言、多工具、扩充评估。

---

## 120. MARS: Margin-Adversarial Risk-controlled Stopping for Parallel LLM Test-time Scaling

**arXiv ID:** 2606.12935 | [PDF](https://arxiv.org/pdf/2606.12935v1)

**作者:** Wenbo Chen `[一作]` (Amazon), Tianpei Xie `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于并行推理轨迹的早停策略，利用中间答案的探测与投票边际来提前终止推理。

**💡 创新点**

创新点在于把投票边际与未来可能的投票变动风险结合，提出了Margin‑Adversarial Risk‑Controlled Stopping规则，并通过每题的γ校准降低过度保守性。

**🔧 技术方法**

技术包括：中间答案探测（probe）、轨迹切换概率的轻量级逻辑回归估计、对冲式切换成本计算、基于Hoeffding界的安全保证，以及使用warmup轨迹进行γ校准。

**📊 数据集**

使用了三大竞赛数学基准：AIME 2025、HMMT、BRUMO 2025，并在三款推理模型上实验（DeepSeek‑R1‑8B、Qwen3‑32B、Qwen3‑next‑80B）。

**📈 对比分析**

与全预算自一致性（Self‑Consistency）和DeepConf Online对比，经过早停后在相同准确率下平均节省了25–47%的token（SC）或额外14–29%（DeepConf），相比同类方法（如Parallel‑Probe）既保持了准确率又获得更高的节省率。

**⚠️ 局限性**

局限性包括：探测间隔固定、依赖于warmup轨迹的估计、仅在数学推理任务上验证，且对轨迹最终答案分布的假设不够精细。

---

## 121. Physics-Informed Neural Networks for Chemotherapy Pharmacokinetics: Benchmarking the Clinical Estimator and Exposing Parameter Identifiability

**arXiv ID:** 2606.12658 | [PDF](https://arxiv.org/pdf/2606.12658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 122. SemanticXR: Low Power and Real-time Queryable Semantic Mapping with an Object-Level Device-Cloud Architecture

**arXiv ID:** 2606.12849 | [PDF](https://arxiv.org/pdf/2606.12849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 123. LLMs Can Better Capture Human Judgments--With the Right Prompts

**arXiv ID:** 2606.12754 | [PDF](https://arxiv.org/pdf/2606.12754v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 124. BASENet: Band-Adapted Speech Enhancement Network with Cross-Band Attention

**arXiv ID:** 2606.12662 | [PDF](https://arxiv.org/pdf/2606.12662v1)

**作者:** Damien Martins Gomes `[一作]` (Thales SIX GTS), François Capman `[通讯]` (Thales SIX GTS)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于Bark尺度的频率自适应语音增强网络BASENet。

**💡 创新点**

通过将频谱按Bark尺度划分并按临界带密度自动分配编码器深度，配合跨频段注意力实现高效、感知驱动的结构。

**🔧 技术方法**

使用倒置残差块+稠密连接的轻量化DenseNet、频域跨频段注意力、卷积递归网络时序建模以及幅度-相位解码。

**📊 数据集**

在常用的VoiceBank+DEMAND混响语音数据集上进行训练与评估。

**📈 对比分析**

与多种基准（包括MH‑SENet、Mamba‑SEUNet等）对比，BASENet在PESQ≥3.50的子1M参数模型中取得最高分（3.55）且参数量仅0.83M、MACs 7.3G；其因果版本PESQ 3.44同样优于多非因果基线。

**⚠️ 局限性**

仍受幅度‑相位输入形式限制，且在更细粒度频段划分时性能下降，尚未验证跨噪声环境的普适性。

---

## 125. Planning on Paper: Problem Decomposition with Diagrams in Introductory Computing

**arXiv ID:** 2606.12427 | [PDF](https://arxiv.org/pdf/2606.12427v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 126. Creating and Evaluating K-12 GenAI Assessment Graders Through Context Engineering

**arXiv ID:** 2606.12422 | [PDF](https://arxiv.org/pdf/2606.12422v1)

**作者:** Zewei Tian `[一作]` (University Of Washington), Min Sun `[通讯]` (University Of Washington)

**通讯引用:** 5398 | [OpenAlex ID](https://openalex.org/A5102008600)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个基于上下文工程的LLM自动评分管线，对K‑12数学、科学与英语语言艺术（ELA）三科进行大规模实验，验证其与人工评分的一致性；

**💡 创新点**

提出系统化的上下文工程管线，将题目、评分标准、示例答案等信息统一打包并按规则组织，显著提升不同LLM的评分一致性；首次在MCAS多学科数据上对四个商业LLM（Claude Sonnet 4、Haiku 4.5、GPT‑5、GPT‑5 Mini）进行全面对比，并以PRMSE为主效度指标；

**🔧 技术方法**

使用大语言模型（Claude Sonnet 4、Haiku 4.5、GPT‑5、GPT‑5 Mini）与Prompt及Context工程；评估指标包括Quadratic Weighted Kappa（QWK）、Proportional Reduction in Mean‑Squared Error（PRMSE）、Exact/Adjacent Agreement；

**📊 数据集**

Massachusetts Comprehensive Assessment System（MCAS）2021‑2025年的822条学生回应，涵盖68套数学题、64套科学题和34套ELA题；

**📈 对比分析**

通过QWK、PRMSE等指标与人工评分对比；结果显示数学QWK≥0.81、PRMSE≥0.73，最高0.951/0.946；科学QWK≈0.83、PRMSE≈0.85；ELA表现波动大，阅读维度QWK最高0.89、PRMSE0.94，但写作维度仅达到QWK≈0.5、PRMSE≈0.7；相比传统自动评分模型，系统在数学与科学上的一致性显著提升；

**⚠️ 局限性**

局限包括：ELA写作评分仍低且模型易产生偏颇；部分长难答案导致时间/输出错误，出现数据丢失；缺乏对高风险分数的完整验证，尚不能用于正式分级；主要适用于形成性评估，需保留人工终审。

---

## 127. An Embodied Simulation Platform, Benchmark, and Data-Efficient Augmentation Framework for Wet-Lab Robotics

**arXiv ID:** 2606.12936 | [PDF](https://arxiv.org/pdf/2606.12936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 128. CLARITree: Cholesky and Lookahead Accelerations for Regression with Interpretable Piecewise Linear Trees

**arXiv ID:** 2606.12840 | [PDF](https://arxiv.org/pdf/2606.12840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 129. LoRA-Muon: Spectral Steepest Descent on the Low-Rank Manifold

**arXiv ID:** 2606.12921 | [PDF](https://arxiv.org/pdf/2606.12921v1)

**作者:** Franz Louis Cesista `[一作]` (Ateneo de Manila University), Stella Biderman `[通讯]` (EleutherAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种低秩适配器（LoRA）的新优化器 LoRA-Muon，旨在解决 LoRA 在低秩设置下难以调优的问题。

**💡 创新点**

创新点在于从低秩流形的谱范数下最速下降出发，推导出分离权重衰减规则和线性最小化轨道（LMO），使 LoRA-Muon 与全秩 Muon 保持相同的谱步长并实现 gauge‑invariant 更新。

**🔧 技术方法**

主要技术包括低秩流形几何、谱范数下的最速下降、矩阵分解与梯度白化、投影到切空间以及信任域半径分配。

**📊 数据集**

实验使用计算匹配的 TinyShakespeare 数据集进行验证。

**📈 对比分析**

通过与密集 Muon、Spectron、LoRA‑RITE 等基准比较，LoRA‑Muon 在 rank‑32 时的平均验证损失低于密集基线，且 rank‑2 代理已能恢复全秩 Muon 的最佳学习率，展示了学习率跨秩、宽度、深度迁移的能力。

**⚠️ 局限性**

局限性包括对因素缩放敏感（Spectron 受此影响），以及在极低秩或更大规模模型上的稳定性和泛化能力尚未完全验证。

---

## 130. The Illusion of Multi-Agent Advantage

**arXiv ID:** 2606.13003 | [PDF](https://arxiv.org/pdf/2606.13003v1)

**作者:** Prathyusha Jwalapuram `[一作]` (Salesforce Research), Shafiq Joty `[通讯]` (Salesforce Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估自动化多代理系统（MAS）与强单代理基线（CoT‑SC）的性能与成本，并引入专为MAS设计的诊断数据集 SMFR，揭示自动化架构普遍存在功能崩溃与架构膨胀。

**💡 创新点**

提出诊断数据集 SMFR 以验证任务对多代理的适配性，系统剖析自动化 MAS 生成的工作流，揭示其在角色专化、交互与协同方面的实际缺陷。

**🔧 技术方法**

采用多种自动化 MAS 框架（DyLAN、MAS‑Zero、ADAS、AFlow、MaAS、MAS‑Orchestra）与不同大语言模型（GPT‑4o、GPT‑5、GPT‑OSS‑120B、Gemini‑2.5‑Pro）进行对照实验。

**📊 数据集**

使用传统推理基准（GSM8K、MMLU、SWE‑Bench Lite、BrowseComp‑Plus 等）以及新构造的 SMFR 数据集。

**📈 对比分析**

对比 CoT‑SC 与自动化 MAS 在准确率、成本（Token 及计算费用）上的折中，发现 CoT‑SC 在大多数任务中以显著更低的成本实现更高或相近的准确率；仅在强模型（GPT‑5、Gemini‑2.5‑Pro）和 SMFR 上，专家手工 MAS 能实现显著提升。

**⚠️ 局限性**

限制：自动化 MAS 在大多数任务上缺乏结构化利用子任务并行与上下文隔离的能力；模型能力提升后控制器梯度消失导致策略停滞；评估侧重于准确率与成本，未深入探讨多代理间的交互机制与长期可扩展性。

---

## 131. AI SciBrief as a Gateway to Research: A Framework for Onboarding Students into New Research Areas

**arXiv ID:** 2606.12413 | [PDF](https://arxiv.org/pdf/2606.12413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 132. High-Order Spectral Element Methods for Wave Propagation on ARM Multicore CPU with SME: Optimizations and Implications

**arXiv ID:** 2606.12850 | [PDF](https://arxiv.org/pdf/2606.12850v1)

**作者:** Yinuo Wang `[一作]` (Tsinghua University), Guangwen Yang `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对SPECFEM3D在ARM LX2 CPU上的实现进行全面优化，结合SME指令集、图着色和混合MPI+OpenMP，且通过等误差（iso-accuracy）研究证明更高多项式阶在SME上更具优势。

**💡 创新点**

创新点在于将SME扩展用于批量小矩阵乘法实现，并将其与离散化层面的p阶优化相结合，证明SME能改变SEM的成本曲线，从而提升更高阶的算子效率。

**🔧 技术方法**

采用SME-aware小矩阵核、软件流水线与SVE聚合、图着色消除原子冲突、专用通信线程以及无组装的高阶张量乘法等技术。

**📊 数据集**

使用SPECFEM3D自带的均匀介质地震波传播模型（Ricker源）以及对应的多阶(h,p)网格进行性能和误差评估。

**📈 对比分析**

与原始SPECFEM3D、SVE基线、Intel Xeon和NVIDIA A100进行对比，SME优化实现4–6倍的应用级加速；在等精度条件下p=7、p=15分别减少3.8倍和1.4倍运行时间，并在p=15时达到5.9倍加速。

**⚠️ 局限性**

局限在于SME优势主要在大多项式阶时显著，低阶时提升有限；通信线程对性能提升有限；在高阶时仍受工作集大小和负载不平衡影响，并且对GPU或其他平台的通用性需进一步验证。

---

## 133. Eigenism: Ethics for a Human-AI Future

**arXiv ID:** 2606.12420 | [PDF](https://arxiv.org/pdf/2606.12420v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 134. Learning to Assist: Collaborative VLAs for Implicit Human-Robot Collaboration

**arXiv ID:** 2606.12475 | [PDF](https://arxiv.org/pdf/2606.12475v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 135. Representing Time Series as Structured Programs for LLM Reasoning

**arXiv ID:** 2606.12481 | [PDF](https://arxiv.org/pdf/2606.12481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 136. Vocal Identity Under Siege by AI Voice Cloning Technologies

**arXiv ID:** 2606.12812 | [PDF](https://arxiv.org/pdf/2606.12812v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 137. Let's Ask Gauss: Improved One-Run Privacy Auditing

**arXiv ID:** 2606.12733 | [PDF](https://arxiv.org/pdf/2606.12733v1)

**作者:** Adya Agrawal `[一作]` (Georgia Institute of Technology), Vassilis Zikas `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种单跑白盒隐私审计方法，通过对可加密梯度投影得分进行归一化累积并建模为高斯分布，从而直接估计 DP‑SGD 和 DP‑FTRL 的实际隐私泄露下界。

**💡 创新点**

①提出可加密梯度投影形成的随机序列在归一化后呈高斯收敛，可直接使用高斯对比（hockey‑stick divergence）而无需阈值化；②给出 CLT 收敛速率证明并在实验中验证高斯模型精度；③利用置信区间/引导椭圆估计高斯参数，进一步提升下界精度。

**🔧 技术方法**

白盒审计、可加密梯度投影、中心极限定理、hockey‑stick 直分距、一次性多可插入“canary”、高斯混合近似、Bootstrap/Bonferroni 置信区间、数值优化。

**📊 数据集**

CIFAR‑10（用于 WideResNet‑16‑4 训练 DP‑SGD）以及 CIFAR‑10 的散列线性分类器（用于 DP‑FTRL）。

**📈 对比分析**

与 Steinke 等、Mahloujifar 等基线（阈值化/ f‑DP 近似）在 95% 置信度下对比；实验表明在 ε=8 时可获得约 6.7 的下界（约 84% 上界），比基线提升 1–2 倍；在不同 ε 与不同 canary 数量下均优于基线；DP‑FTRL 亦得到更紧的下界。

**⚠️ 局限性**

依赖理想化的可加密梯度观测模型和可加密梯度投影的独立性；仅在高斯收敛条件（如小采样率、足够训练步数）下适用；若采样率或步数不满足 CLT 条件，方法可能失效；假设训练过程与 canary 无交互，可能在其他设置下引入偏差。

---

## 138. Localizing Anchoring Pathways in Language Models

**arXiv ID:** 2606.12818 | [PDF](https://arxiv.org/pdf/2606.12818v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 139. GRIP: Feedback-Guided Prompt Retrieval for Large Multimodal Models

**arXiv ID:** 2606.12744 | [PDF](https://arxiv.org/pdf/2606.12744v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 140. The AI Legal Specialist: A Juridically Autonomous Professional Profile for AI Governance

**arXiv ID:** 2606.12415 | [PDF](https://arxiv.org/pdf/2606.12415v1)

**作者:** Nicola Fabiano `[一作]` `[通讯]` (Studio Legale Fabiano), Nicola Fabiano (Studio Legale Fabiano)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了“AI 法律专员”（AI Legal Specialist）这一全新的法律专业角色，并给出了其法律基础、功能范围和独立性。

**💡 创新点**

创新点在于：①从司法义务出发构建职业架构，强调该角色的司法自主性；②将欧盟 e‑Competence Framework 作为可迁移的能力框架；③给出了可量化的绩效指标体系。

**🔧 技术方法**

主要方法是规范性重构（normative‑reconstructive）研究法，结合对各国 AI 法规的结构化比较，采用 e‑Competence Framework 的能力映射。

**📊 数据集**

使用的数据集为各国 AI 及相关数字监管文本（欧盟 AI Act、GDPR、NIS2、美国行政框架、日、韩、巴西等国的法规文本）以及各国监管机构的指引与案例。

**📈 对比分析**

比较方法是通过“司法义务‑专业需求”映射，评估该角色相较于 DPO、隐私律师、合规官等现有角色的覆盖度与独立性；性能表现以可量化指标的可测性与跨司法管辖区迁移性为衡量，未给出定量实验数据。

**⚠️ 局限性**

局限性包括：①正式能力清单尚未公开，需进一步细化；②缺乏实证验证其在真实组织中的效果；③在不同法域间的监管细节差异可能导致模型调整需求。

---

## 141. HybridCodeAuthorship: A Benchmark Dataset for Line-Level Code Authorship Detection

**arXiv ID:** 2606.12620 | [PDF](https://arxiv.org/pdf/2606.12620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 142. Forecasting Is Not Attribution: Localizing Decoder Bypass in Graph-Based Neural Marketing Mix Models

**arXiv ID:** 2606.12687 | [PDF](https://arxiv.org/pdf/2606.12687v1)

**作者:** Yunbo Wang `[一作]` (University of California, Irvine), Bolbi Liu `[通讯]` (AdsGency AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了图神经网络在营销组合建模（MMM）中预测与归因分离的问题，提出了两阶段的DICE-MMM框架，并通过CIG/AR-CIG诊断及图交换实验验证模型是否真正利用图进行归因。

**💡 创新点**

创新点包括：① 发现并命名“归因绕过”失败模式；② 设计两阶段图学习+图安全潜在解码器，确保图作为解码器接口；③ 引入CIG/AR-CIG诊断与冻结图交换实验，实现归因质量的可视化与定位；④ 将问题定位到可部署的稀疏图支持选择。

**🔧 技术方法**

技术手段：图VAE编码器、Gumbel-softmax采样、受限解码器、冻结图安全潜在解码器（LatentGraphDecoder）、CIG/AR-CIG 触发扰动、稀疏选择方法（验证-MSE、稳定性选择、Top‑k稠密度）以及实验中的图交换和图接口评估。

**📊 数据集**

数据集：使用控制合成MMM数据集，覆盖不同的R/d/T组合；引入非退化的稀疏目标基准；在外部多图rawlog stress test中检验图恢复鲁棒性。

**📈 对比分析**

与Baseline CausalMMM 对比：DICE在图恢复的稳定性（Final‑20 AUROC）上优于Baseline；预测误差（MSE@7）与Baseline相当；归因指标在oracle图下显著提升（AR‑CIG nAUPRC≈0.8），而在无图/全图输入下近乎零，验证了预测与归因的分离；冻结图交换实验进一步证明解码器能在获得正确图时实现归因对齐。

**⚠️ 局限性**

局限性：目前学到的图接口稠密，稀疏图支持选择不足；归因诊断指标并非因果估计，需要更多假设；实验仅基于合成数据，未涵盖真实MMM中隐藏混淆、非平稳性、预算约束等复杂因素；未来工作需改进可部署稀疏图选择与真实数据验证。

---

## 143. EDEN: A Large-Scale Corpus of Clinical Notes for Italian

**arXiv ID:** 2606.12569 | [PDF](https://arxiv.org/pdf/2606.12569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 144. A Type Theory of Sense: Witnessed Choice in Stratified Semantic Spaces

**arXiv ID:** 2606.12504 | [PDF](https://arxiv.org/pdf/2606.12504v1)

**作者:** Iman Poernomo `[一作]` `[通讯]`, Iman Poernomo

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种依赖类型理论TTS，通过测量数据实现分级语义判定，捕捉Frege式意义、超强语义差异与生成中的分叉现象；

**💡 创新点**

创新点在于用局部（基于测量仪器）非可判定的apartness与indiscernibility取代传统全局Kan条件，得到可证的排除、持久性与跨仪器一致性定理，并将Frege的意义与引用映射为可测空间的填充器；

**🔧 技术方法**

采用了分层的依赖类型结构、局部关系形成者、测量上下文、以及基于分辨率阈值的simplicial apartness空间模型，辅以可解释的度量分解与抽样统计；

**📊 数据集**

实验数据主要来自GPT‑2、Llama‑7B、Mistral‑7B、Pythia‑6.9B的词嵌入空间与生成分叉测试，结合Cosine距离与嵌入图构建的测量上下文；

**📈 对比分析**

通过“fork test”与“2‑horn infill”两种实验流程，检验模型是否在不同分辨率下产生分叉，发现分叉率与模型的流畅性与对齐策略相关，实验表明理论能在实际数据中检测到非流形性与多模态分叉；

**⚠️ 局限性**

局限包括：缺乏完整性与归一化证明、仅处理内层2‑horn、只能根据有限测量记录给出可证的分叉而无法确认全局可识别性、对仪器设定高度依赖、未解决多尺度/多维度分叉的统一描述。

---

## 145. Is Spurious Correlation Removal Always Learnable?

**arXiv ID:** 2606.12930 | [PDF](https://arxiv.org/pdf/2606.12930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 146. MARD: Mirror-Augmented Reasoning Distillation for Mechanism-Level Drug-Drug Interaction Prediction

**arXiv ID:** 2606.12578 | [PDF](https://arxiv.org/pdf/2606.12578v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 147. GENIE: A Fine-Grained Measure for Novelty

**arXiv ID:** 2606.12790 | [PDF](https://arxiv.org/pdf/2606.12790v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 148. A Calculus of Apartness over Separoids: Effective Convex Representation, Stratified Conservativity, and the Complexity of Entailment

**arXiv ID:** 2606.12676 | [PDF](https://arxiv.org/pdf/2606.12676v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 149. GenHOI: Contact-Aware Humanoid-Object Interaction by Imitating Generated Videos without Task-Specific Training

**arXiv ID:** 2606.12995 | [PDF](https://arxiv.org/pdf/2606.12995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 150. Language-Guided Abstraction for Visual Reasoning

**arXiv ID:** 2606.12847 | [PDF](https://arxiv.org/pdf/2606.12847v1)

**作者:** Xu-Jing Ye `[一作]` (Guangzhou University), Ruping Wang `[通讯]` (Traditional Chinese Medicine Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉抽象推理任务ARC中，提出L-VARC框架，通过在训练阶段引入语言先验来提升视觉推理。

**💡 创新点**

创新点：1）采用LUPI范式，在训练时利用LARC语言描述作为特权信息；2）设计语义压缩模块(SCM)，使用DeepSeek‑V3将冗长文本压缩为规则导向短文本；3）设计跨注意投影器(CAP)，对齐视觉特征与CLIP文本嵌入，实现动态关注关键视觉区域。

**🔧 技术方法**

技术：ViT视觉主干 + CLIP文本编码器 + DeepSeek‑V3（LLM）+ 交叉注意投影器 + InfoNCE对比损失 + 训练时测试（TTT）方法。

**📊 数据集**

数据集：ARC（ARC‑1、ARC‑2）及其对应的LARC语言描述数据集。

**📈 对比分析**

对比方法：VARC、HRM、TRM等神经/神经符号方法。性能提升：ARC‑1 PASS@1从49.75%提升至50.62%（+0.87%），Oracle亦提升；ARC‑2 PASS@1从6.25%提升至6.67%（+0.42%）。

**⚠️ 局限性**

局限：仍无法处理多步算术、递归/分形模式等复杂规则；对语义压缩依赖离线LLM；对极端新规则的泛化能力有限。

---

## 151. The Metric Picks the Winner: Evaluation Choice Flips Model Rankings for Drug-Response Prediction in Unseen Chemistry

**arXiv ID:** 2606.12639 | [PDF](https://arxiv.org/pdf/2606.12639v1)

**作者:** Dhruv Agarwal `[一作]`, Riya Bisht `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一套分阶段、可复现的管道，用于预测THP-1细胞在未见化合物下的转录组响应，并在VCPI DRUG-seq任务上进行严格的Bemis‑Murcko scaffold交叉验证；

**💡 创新点**

首次在真实化学hold‑out数据中验证了评估指标（active‑set wMSE vs. inverse‑variance proxy）可导致模型排名反转的metric‑calibration效应；提出检索+化学嵌入融合解码器，恢复检索在scaffold split下的性能，并提供不确定性头和基因程序解释；

**🔧 技术方法**

使用Morgan指纹、RDKit描述子、ChemBERTa预训练模型、检索加权平均、岭回归、深度MLP解码器、异方差头、NMF基因程序分解、配分的wMSE评估及scaffold split交叉验证；

**📊 数据集**

VCPI THP‑1 DRUG‑seq数据（32,500样本×78,778基因，12,995评分基因），共14,026训练化合物与1064 hold‑out化合物，采用Bemis‑Murcko scaffold拆分；

**📈 对比分析**

在scaffold split下对六类模型进行比较，使用官方wMSE。代理指标下岭回归指纹最佳；真实指标下深度MLP最佳，融合模型次之，检索第三，线性基线最差；在真实指标上的提升显著（p<1e-4），且不确定性头校准良好；

**⚠️ 局限性**

限制包括深度模型仅在两fold评估（计算成本高），评估基于训练集scaffold CV而非真实测试集；检索特征在代理指标下表现不佳，需在真实指标下重新评估；ChemBERTa与检索特征消融仅在代理指标完成。

---

## 152. Efficient, Robust, and Anti-Collusion Fingerprinting of Image Diffusion Models

**arXiv ID:** 2606.12977 | [PDF](https://arxiv.org/pdf/2606.12977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 153. Mixed-Categorical Black-Box Optimization via Information-Geometric Bilevel Decomposition

**arXiv ID:** 2606.12885 | [PDF](https://arxiv.org/pdf/2606.12885v1)

**作者:** Marc Ong `[一作]` (University of Tsukuba), Youhei Akimoto `[通讯]` (University of Tsukuba)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于信息几何的双层优化框架，用以解决混合分类与连续变量的黑盒优化问题，并通过缓存式热启动显著降低双层求解成本。

**💡 创新点**

创新点在于：① 通过双层结构显式建模分类与连续变量间的交互，克服传统 CatCMA 等乘积分布假设导致的性能瓶颈；② 采用信息几何自然梯度（IGO）统一上层 ASNG 与下层 dd-CMA-ES 的更新；③ 设计多配置缓存热启动策略，实现对下层求解的快速复用。

**🔧 技术方法**

使用的技术包括：信息几何优化（IGO）、自适应自然梯度（ASNG）用于上层分类搜索、dd-CMA-ES（带活跃更新与自适应对角解码）用于下层连续搜索，以及基于缓存的热启动与重启策略。

**📊 数据集**

实验数据集：自定义的二分类-连续混合基准问题，涵盖四类交互（I、II、III、IV）并在不同维度（d_c=d_x=5、10）及条件数（κ=10^2、10^6）下生成，交互强度 a∈{0,1,2,4,8,16}。

**📈 对比分析**

与 CatCMA 与 ICatCMA 在同一基准上比较。实验显示，本方法在所有交互类型下都能保持更高的成功率（大多数情形 100%），并且在大多数情形下所需的函数评估次数显著低于两种基线；仅在高交互强度或极端条件数下才出现失败，且多次重启可进一步提升鲁棒性。

**⚠️ 局限性**

局限性包括：① 对高维（尤其是多类别）问题的计算开销仍较大；② 目前仅在二进制分类变量上验证，扩展到多类别需要更多研究；③ 热启动策略依赖缓存大小与评分阈值，可能在某些问题中表现不佳；④ 在极高交互强度或预算极限时仍可能失败。

---

## 154. AiAWE: An Open-Source LLM Automated Writing Evaluation System Using LoRA-Adapted Instruction-Tuned Models

**arXiv ID:** 2606.12801 | [PDF](https://arxiv.org/pdf/2606.12801v1)

**作者:** John Maurice Gayed `[一作]` `[通讯]` (Waseda University), John Maurice Gayed (Waseda University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 AiAWE 自动写作评估系统，使用 LoRA 微调开源 Gemma-3-27B（和 LLaMA‑3.3‑70B）模型对 TOEFL 独立写作文章进行评分并提供结构化反馈；

**💡 创新点**

证明开源 LLM 通过 LoRA 微调可匹配甚至超过闭源 GPT‑3.5 的评分性能，揭示模型规模并非下游性能的可靠指示器，且 LoRA 超参数在不同架构上不具通用性；

**🔧 技术方法**

利用 LoRA 参数高效微调、Instruction‑Tuned LLM、GGUF 4‑bit 量化以及开源推理框架（llama‑factory、ggml）实现本地部署；

**📊 数据集**

使用 ETS 提供的 480 篇 TOEFL 独立写作作文（240 篇每个题目）做训练和评估；

**📈 对比分析**

在 120/360 的划分下，Gemma‑LoRA 达到 RMSE 0.474、QWK 0.828、±0.5 一致率 90.56%，优于同样微调的 GPT‑3.5（RMSE 0.573、QWK 0.78、一致率 84.72%）且在同等量化下与 LLaMA‑70B 的表现相近；

**⚠️ 局限性**

局限包括仅评估单一题型与数据集，样本极端分数稀少，未对其他写作体裁或多语言环境验证，未进行公平性或课堂效果研究，且 LoRA 适配对不同模型敏感且未系统化超参数探测。

---

## 155. AudioX-Turbo: A Unified Framework for Efficient Anything-to-Audio Generation

**arXiv ID:** 2606.12555 | [PDF](https://arxiv.org/pdf/2606.12555v1)

**作者:** Zeyue Tian `[一作]` (Hong Kong University of Science and Technology), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19074 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出AudioX-Turbo框架，统一实现任意文本、视频、音频条件下的高质量音频与音乐生成，并通过教师-学生蒸馏实现仅4步高效推理。

**💡 创新点**

创新点包括轻量级多模态自适应融合模块(MAF)、基于流匹配的分布匹配蒸馏结合扩散判别器实现的极简采样、以及构建约920万条多模态高质量数据集AudioX。

**🔧 技术方法**

采用多模态扩散Transformer、流匹配学习、分布匹配蒸馏、扩散判别器、教师自由引导等技术。

**📊 数据集**

训练数据来自VGGSound、AudioSet-Strong、V2M-500K等公开数据并通过两阶段管道生成的AudioX（约9.2M条），评测使用AudioCaps、MusicCaps、V2M-bench等。

**📈 对比分析**

与专用模型在文本转音频、视频转音频、文本/视频/音频联合转音乐等任务上对比，使用IS、CLAP、FAD/FD等指标，AudioX-Turbo在保持SOTA质量的同时，仅用4步即可匹配教师，NFE大幅降低至1/25。

**⚠️ 局限性**

局限性包括仅训练10秒短片，未覆盖语音、长序列生成，极端指令跟随时仍可能失误，且不具备可变步长的灵活采样。

---

## 156. Divination by Prompt: LLM-Mediated Xuanxue on Chinese Social Media

**arXiv ID:** 2606.12418 | [PDF](https://arxiv.org/pdf/2606.12418v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 157. Crossing the Validation Crisis: Cross-Validation Reduces Benchmarking Variance Surprisingly Well

**arXiv ID:** 2606.12552 | [PDF](https://arxiv.org/pdf/2606.12552v1)

**作者:** Célestin Eve `[一作]` (Université Paris-Saclay), Thomas Moreau `[通讯]` (Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在小样本机器学习基准中使用交叉验证(CV)以降低评估方差，并量化了其对性能估计可靠性的提升；

**💡 创新点**

创新点在于提出了“样本增益(sample gain)”这一衡量指标，能够将多次CV所带来的方差降低转化为等价的单次测试样本增量；同时设计了仅基于研究集的冗余度量，用于提前终止CV以节约计算；

**🔧 技术方法**

技术上采用了蒙特卡洛交叉验证(MCCV)、K折交叉验证、Bootstrap置信区间估计，以及统计方差分解和冗余度量；

**📊 数据集**

实验数据集包括：大规模合成回归数据、Camelyon16癌症转移检测图像数据、以及包含65万条评测的情感分类文本数据；

**📈 对比分析**

比较方法是将多次CV得到的性能与单次hold‑out在等价测试样本量下的方差进行对比；实验表明，多次CV可使等价测试样本量提升5–15倍，且在排名检验中显著提升排名准确性；

**⚠️ 局限性**

局限性包括：需要大规模基准集来估计oracle性能，导致实验范围受限；计算成本随K线性增长，实际应用需平衡资源；研究仅覆盖医学影像和NLP两大领域，未能覆盖所有小样本场景。

---

## 158. A Stationary (and Therefore Compatible) Representation is All You Need

**arXiv ID:** 2606.12488 | [PDF](https://arxiv.org/pdf/2606.12488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 159. Viral Proteins Reveal Geometry of Protein Language Models

**arXiv ID:** 2606.12609 | [PDF](https://arxiv.org/pdf/2606.12609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 160. Out-of-Distribution (OOD) Detectors for Open-Set RF Fingerprinting

**arXiv ID:** 2606.12718 | [PDF](https://arxiv.org/pdf/2606.12718v1)

**作者:** Sudeepta Mondal `[一作]` (RTX Technology Research Center), Ganesh Sundaramoorthi `[通讯]` (RTX Technology Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了开放式射频指纹识别中的离群检测问题，提出并评估了基于特征塑形的后置OOD检测方法，并引入了无须OOD训练数据的SHOT调优策略。

**💡 创新点**

创新点在于将信息理论框架与特征塑形OOD方法结合，用SHOT实现了在射频环境中不依赖真实OOD数据即可调优检测器，且实验表明性能接近或优于有OOD调优的基线。

**🔧 技术方法**

使用的技术包括ResNet-18 1D分类器、ReAct、VRA、PLF三种特征塑形手段、能量分数评分、自动编码器对比、以及SHOT的模拟留出调优与贝叶斯优化。

**📊 数据集**

实验数据集为POWDER RF Fingerprinting，包含四个LTE基站在不同时间收集的IQ样本，构成多种开放式测试场景。

**📈 对比分析**

与传统需要OOD调优的基线相比，SHOT调优的特征塑形检测器在AUROC上接近或超过基线，在FPR95上显著优于基线；在完全无OOD数据的情况，SHOT方法仍优于自动编码器基线。

**⚠️ 局限性**

限制在于对不同射频条件的泛化仍有限，特征塑形模型的参数维度较大时对OOD调优数据敏感，且实验仅覆盖少数基站与波形，未来需扩展至更大、更异质的射频数据集。

---

## 161. Pythagoras-Prover: Advancing Efficient Formal Proving via Augmented Lean Formalisation

**arXiv ID:** 2606.12594 | [PDF](https://arxiv.org/pdf/2606.12594v1)

**作者:** Joshua Ong Jun Leang `[一作]` (Imperial College London), Eleonora Giunchiglia `[通讯]` (Imperial College London)

**通讯引用:** 337 | [OpenAlex ID](https://openalex.org/A5102734034)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个计算效率高的 Lean 4 定理证明器家族，包括 4B、32B 的自回归模型和首个扩散式定理证明器。

**💡 创新点**

创新点在于：① 采用计算友好的数据流水线，先构建约 800K 的 Lean 验证语料，按难度分层做课程化训练；② 引入 Augmented Lean Formalisation (ALF) 结构化变异机制，以轻量校验扩充语料到 2M；③ 在 8K 上下文预算内实现动态证明-推理过滤；④ 结合自蒸馏和从 Lean 验证器获得的强化学习奖励；⑤ 对扩散模型进行 tactic‑based masking，使其适用于长 Lean 证明。

**🔧 技术方法**

技术包括 LoRA 参数高效微调、课程化训练、动态证明过滤、基于 Lean 的自蒸馏、GRPO 强化学习、块级扩散（Diffusion）以及 tactic‑based masking。

**📊 数据集**

数据集：Lean‑verified seed 约 800K（easy/medium/hard），通过 ALF 扩展至约 2M；MiniF2F、PutnamBench 作为评测基准；MiniF2F‑ALF 作为扰动/鲁棒性测试集；此外使用公开的数学推理数据集（GSM‑8K、MATH、AIME、AMC、IMO 等）进行语料生成。

**📈 对比分析**

通过在 MiniF2F‑Test、PutnamBench 和 MiniF2F‑ALF 上的 Pass@N 评测比较：4B 在 pass@32 上 86.1% 甚至超过 671B 的 DeepSeek‑Prover‑V2；32B 在 pass@2048 上 93.0%，刷新了开放源代码定理证明器的最高成绩；扩散模型在 pass@32 上 63.3%，虽然准确率落后，但在相同硬件下推理吞吐量提升 2.6×，可达 6.68 的 throughput‑weighted 分数；对比自校正与重启采样发现重启采样在低预算下更高效、在 PutnamBench 上取得 93/672 的最高解题率。

**⚠️ 局限性**

局限性：① 扩散模型受限于当前 4k 的上下文窗口，难以处理更长证明；② 训练与推理仍需大量算力，尤其是大模型和自校正；③ MiniF2F 基准已趋于饱和，需更具挑战性的评测；④ ALF 的变异操作有限，未覆盖所有类型的结构扰动；⑤ 仅针对 Lean 4，尚未验证在其他定理证明器（Isabelle、Coq 等）的适用性。

---

## 162. ProPlay: Procedural World Models for Self-Evolving LLM Agents

**arXiv ID:** 2606.12780 | [PDF](https://arxiv.org/pdf/2606.12780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 163. DIMOS: Disentangling Instance-level Moving Object Segmentation

**arXiv ID:** 2606.12826 | [PDF](https://arxiv.org/pdf/2606.12826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 164. Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict Behavior

**arXiv ID:** 2606.12730 | [PDF](https://arxiv.org/pdf/2606.12730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 165. A Tutorial on World Models and Physical AI

**arXiv ID:** 2606.12783 | [PDF](https://arxiv.org/pdf/2606.12783v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 166. Deployment-Centered Evaluation: Predicting Query-Level Rejection Risk in a Clinical LLM System

**arXiv ID:** 2606.12702 | [PDF](https://arxiv.org/pdf/2606.12702v1)

**作者:** Alyssa Unell `[一作]` (Stanford University), Nigam Shah `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在学术医疗中心将大语言模型嵌入电子健康记录系统后，作者利用真实用户在使用过程中的稀疏二元反馈，训练了一个预响应拒绝预测模型，目标是预测未来查询是否会被临床用户拒绝；

**💡 创新点**

创新点在于将部署相关上下文（如提供者类型、科室名称、使用的模型）与查询内容一起纳入模型，突破传统基准评测只关注正确率的局限，使模型能够在实际工作流中做出精准的拒绝预测与后续干预；

**🔧 技术方法**

技术实现基于逻辑回归，输入为查询的文本嵌入（OpenAI text‑embedding‑3‑large）与一热编码的元信息，进行网格搜索调参并动态重训练；实验中还与仅使用查询嵌入的基线、LLM Judge（仅查询或加上下文）以及始终接受/拒绝基线进行对比；

**📊 数据集**

使用了来自一所学术医疗中心的真实部署日志，共计74,729次交互，其中1,196次带二元反馈，最终筛选到878个标注好的查询-反馈对，数据稀疏且多为“thumbs up/down”；

**📈 对比分析**

在19周的前瞻性评估中，模型AUROC达0.719，宏F1为0.591、微F1为0.651，均显著优于仅用查询嵌入的基线以及LLM Judge基线；在不同β阈值下，可实现高精度的“拒绝”预警（β=0.12时精度0.88）或高召回的“警戒线”触发（β=4.0时召回0.99），体现了模型对下游决策的可操作性；

**⚠️ 局限性**

主要局限包括：用户二元反馈极其稀疏（仅1.6%），导致样本量小且可能存在选择偏差；仅来自单一医疗中心，缺乏跨机构的泛化验证；高维文本嵌入在稀疏数据下信号有限；此外，未能完全捕捉未给反馈用户的真实需求与偏好。

---

## 167. Constructing Evaluation Datasets for Procedural Reasoning: Balancing Naturalness, Grounding, and Multi-Hop Coverage

**arXiv ID:** 2606.12767 | [PDF](https://arxiv.org/pdf/2606.12767v1)

**作者:** Sarah Elshabrawy `[一作]` (Georgia Institute of Technology), Ashok K. Goel `[通讯]` (Georgia Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了利用TMK模型构建程序推理评估数据集的三种生成策略，并引入闭集证据验证框架评估生成问答对的基准质量与多跳覆盖。

**💡 创新点**

创新点在于提出严格TMK、转录先生成+TMK过滤、TMK感知生成三种生成策略，并设计基于TMK证据的闭集验证方法，强调自包含性与代表性基准的必要性。

**🔧 技术方法**

技术主要包括基于提示的LLM问答生成、结构化TMK模型作为知识来源、闭集证据提取与程序化验证，以及多跳推理标签判定。

**📊 数据集**

使用的数据集为23个Georgia Tech知识型AI课程主题的教学稿与对应TMK模型，共生成690个问答对。

**📈 对比分析**

对三种生成策略进行对比，严格TMK生成在grounded、self-contained和usable率上最高，提供了168个可用多跳题；转录先生成次之；TMK感知生成多跳率最高但可用率最低。

**⚠️ 局限性**

局限性包括验证仍依赖LLM判断、未对所有生成项进行人工标注、未考虑主题层级差异、仅评估单一课程领域、未检验对学习者或AI辅导性能的影响。

---

## 168. Free-Placement Optimization of Ground Station Locations for Low-Earth Orbit Satellites

**arXiv ID:** 2606.12667 | [PDF](https://arxiv.org/pdf/2606.12667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 169. Physics-Informed Neural Networks and Radial Basis Functions for PDEs with Dirac Delta Sources

**arXiv ID:** 2606.12735 | [PDF](https://arxiv.org/pdf/2606.12735v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 170. Two-Layer Linear Auto-Regressive Models Estimate Latent States

**arXiv ID:** 2606.12691 | [PDF](https://arxiv.org/pdf/2606.12691v1)

**作者:** Yahya Sattar `[一作]` (Cornell), Sarah Dean `[通讯]` (Cornell)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了在单条轨迹上通过经验风险最小化训练的两层线性自回归模型，证明其隐藏层自然而然地逼近部分可观测线性动态系统的Kalman滤波器，从而实现对潜在状态的恢复。

**💡 创新点**

创新点在于：①证明非凸优化景观无局部最优陷阱；②给出有限样本下预测误差、参数误差和潜在状态恢复的理论保证；③展示隐藏表示与Kalman滤波器状态估计仅相差一次相似变换。

**🔧 技术方法**

主要技术包括：两层线性网络的自回归建模、有限样本系统辨识理论、非凸优化理论（严格鞍点）、高斯过程自适应推断与矩阵范数分析。

**📊 数据集**

实验数据集包括：随机生成的4维状态、2维控制、3维观测的线性系统；以及ControlGym中的水下车辆（n=8）和飞机（n=10）系统。

**📈 对比分析**

与传统的Ho‑Kalman分解、核范数正则化等经典辨识方法相比，本文方法在单个轨迹上即可达到近似Kalman滤波性能，实验中潜在状态恢复的R²均高于0.99，证明了强大的表示学习效果。

**⚠️ 局限性**

局限性包括：仅针对高斯噪声的线性系统，理论假设对系统可观测/可控性要求严格；未考虑深层网络或非线性动力学；以及对相似变换恢复的实用性仍需进一步验证。

---

## 171. Net-Ev$^2$: A Generative Simulator for Network Event Evolution

**arXiv ID:** 2606.12494 | [PDF](https://arxiv.org/pdf/2606.12494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 172. Boltzmann Attention: Learnable Ising Couplings for Cooperative Attention

**arXiv ID:** 2606.12478 | [PDF](https://arxiv.org/pdf/2606.12478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 173. DailyReport: An Open-ended Benchmark for Evaluating Search Agents on Daily Search Tasks

**arXiv ID:** 2606.12871 | [PDF](https://arxiv.org/pdf/2606.12871v1)

**作者:** Jingxuan Han `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DailyReport基准，用于评估搜索代理在日常搜索任务中的表现；

**💡 创新点**

创新点在于：①基于真实热门话题构建开放式任务，涵盖150个任务与3,546条评价rubric；②引入用户中心化的cascade评估流程，分离指令跟随、事实性、推理性三个互斥维度；③通过层级归因与重要度加权实现可解释的整体与用户偏好评分；

**🔧 技术方法**

技术实现包括：LLM自动生成并人工修订rubric；使用搜索工具或代码插件的LLM模型；Cascade评估链（指令→事实→推理）与用户偏好聚合算法；

**📊 数据集**

数据集：从微博、Twitter、Reddit等社交平台收集的热门话题与用户评论，经过专家转化为任务，再细化为子任务与rubric；共150个任务，3,546条rubric，覆盖10大领域35细分类别；

**📈 对比分析**

实验对比了17个代理系统（原生DRA、搜索增强LLM、含Claude Code的LLM）。搜索工具+LLM组合获得最高整体分，指令跟随≈98%，事实性≈83-90%，推理性≈80-93%；但用户偏好分最高仅为2.9，低于可接受阈值3；对比中，LLM+搜索工具的召回和引用率最高；

**⚠️ 局限性**

局限性：①用户偏好仍未达到满意水平，说明生成内容与实际需求匹配度不足；②事实性与推理性仍明显弱于指令跟随，易产生虚假或不连贯信息；③评估仍依赖人工标注的辅助裁判，可能对动态热点的更新速度不够敏感；

---

## 174. Bag of Dims: Training-Free Mechanistic Interpretability via Dimension-Level Sign Patterns

**arXiv ID:** 2606.12629 | [PDF](https://arxiv.org/pdf/2606.12629v1)

**作者:** Varun Reddy Nalagatla `[一作]` `[通讯]` (Amazon Web Services), Varun Reddy Nalagatla (Amazon Web Services)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过观察 transformer 隐层维度的符号模式，提出 Bag of Dims 框架，展示在 Qwen、Gemma 和 Mistral 三大 4–7B 模型上，维度本身就可作为独立的二进制寄存器读出语义特征，无需训练或优化。

**💡 创新点**

创新点在于发现并利用隐藏层维度符号的独立性和轴对齐性：符号承载语义，幅值表示置信度；且这种结构在残差流、K/V 投影及 FFN 写入过程均保持，证明标准基底即能完整读写特征。

**🔧 技术方法**

技术包括：单词类型缓存（每个词元单独前向一次）、按维度 AUC 发现特征、符号一致性簇（Bag of Dims）、Hamming 距离评分、FFN 权重列对齐检查以及 MI 计算验证维度独立性。

**📊 数据集**

使用了三组公开模型的完整词表（Qwen 3.5‑4B ~248K，Gemma 3‑4B ~262K，Mistral 7B ~32K）以及 175 个人工标注的语义类别（动物、情感、数词等），并进行无监督的 1500 维度特征探索。

**📈 对比分析**

对比方法包括：符号+均衡权重、全量权重、随机符号、纯 Hamming 匹配。实验表明符号模式即可达到 72–93% 的 Top‑5 预测准确率，纯 Hamming 也能在 80–90% 的 Top‑4096 覆盖率；零训练特征发现的 AUC 均在 0.80 以上，且已在 K/V 投影与 FFN 写入层重现，显示与传统训练探针相比几乎无性能差距。

**⚠️ 局限性**

局限性：仅在 4–7B 规模模型验证；低幅值维度的上下文调制机制尚未完全建模；无法通过单个符号维度实现高效行为控制；对更大规模（70B+）模型的适用性未知；以及发现的 175 类目仅是模型全部特征的一小部分。

---

## 175. Stereo Vision-Based Fall Prediction and Detection using Human Pose Estimation on the AMD Kria K26 SOM

**arXiv ID:** 2606.12473 | [PDF](https://arxiv.org/pdf/2606.12473v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 176. HarnessBridge: Learnable Bidirectional Controller for LLM Agent Harness

**arXiv ID:** 2606.12882 | [PDF](https://arxiv.org/pdf/2606.12882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 177. It's Safer to Give Personhood to Bears than to Artificial Intelligence

**arXiv ID:** 2606.12440 | [PDF](https://arxiv.org/pdf/2606.12440v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 178. G-MAPP: GPU-accelerated Multi-Agent Planning and Perception for Reactive Motion Generation

**arXiv ID:** 2606.12579 | [PDF](https://arxiv.org/pdf/2606.12579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 179. CD-RCM: Generalizable Continuous-Depth Novel View Synthesis for Reflectance Confocal Microscopy

**arXiv ID:** 2606.12635 | [PDF](https://arxiv.org/pdf/2606.12635v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 180. Who Designs the Designer? Behavioural Architecture for GenAI in Education

**arXiv ID:** 2606.12416 | [PDF](https://arxiv.org/pdf/2606.12416v1)

**作者:** Sepinoud Azimi `[一作]` (Delft University of Technology), Sepinoud Azimi `[通讯]` (Delft University of Technology)

**通讯引用:** 354 | [OpenAlex ID](https://openalex.org/A5082298867)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一种基于行为学的 AI 教育架构，即“行为架构”，通过让教师、学生和 AI 三方共同设计学生的学习档案，系统不再仅根据内容进行适配，而是主动适配学生的个性、动机和情绪；该架构强调学生共创学习记录，赋予学生对自身数据的可读、可修改、可撤销权。

**💡 创新点**

创新点在于：① 将“设计者”角色从单一的 AI 供应商分配到教师、学生和 AI 三方共同承担；② 强调学生共创学习档案，避免隐式推断与不透明的行为记录；③ 将行为、情感、动机三维纳入学习适配的核心维度；④ 提出治理框架与欧盟层面的监管需求，强调跨机构可携带性与心理安全。

**🔧 技术方法**

技术上主要借助大型语言模型（LLM）实现对话式共创与规则生成，并结合行为追踪与情绪识别（如基于实时情绪信号的适配），但本文侧重于架构与治理设计，未给出具体实现细节或新算法；技术核心是通过对话接口让学生与系统共同定义学习规则。

**📊 数据集**

文章未使用任何实验数据集，而是基于已有文献综述（如 70,000+ 学生的元分析、SDT 文献、情绪研究等）和案例说明（如 Adelphi 案例）来阐述问题与方案；未来研究计划通过实验数据来验证五个研究问题。

**📈 对比分析**

对比方法尚未完成实验；作者提出的五个实证问题包括：① 行为架构是否在学习成效上优于仅基于内容的适配；② 共创是否真正减少权力不对称；③ 是否出现 Solaria 效应（个体优越而集体学习受损）；④ 系统是否在实践中能避免将学习观察误判为心理标签；⑤ 元素素养训练是否能提升共创效果。性能评估将通过直接实验比较学习成绩、学习时间等指标，但目前仍待验证。

**⚠️ 局限性**

主要局限包括：① 缺乏实证验证，尚未证明架构在学习成效与公平性上的优势；② 对治理与监管依赖欧盟层面，单一机构难以实现；③ 需要学生具备元元素素养，可能加剧数字鸿沟；④ 系统若实现高度个性化，可能导致“Solaria”式的孤立学习；⑤ 对隐私、心理标签的法律与伦理约束仍不完善，需进一步研究与标准制定。

---

## 181. Policy-driven Conformal Prediction for Trustworthy QoT Estimation

**arXiv ID:** 2606.12501 | [PDF](https://arxiv.org/pdf/2606.12501v1)

**作者:** Kiarash Rezaei `[一作]` (Chalmers University of Technology), Carlos Natalino `[通讯]` (Chalmers University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Conformal QoT 框架，将 conformal prediction 与基于策略的决策逻辑相结合，以生成统计保证的 QoT 置信带并据此做光路径可行性决策；

**💡 创新点**

创新点在于首次将无分布假设的 conformal prediction 与运营决策策略整合，使得在域迁移、网络条件变化下仍可提供可信的 QoT 估计，并仅需极少目标域样本即可重用模型；

**🔧 技术方法**

使用 XGBoost 作为基础回归器，配合 conformal prediction 生成全局置信带（GC band），并通过 RA（风险规避）和 EO（效率导向）两种策略根据置信带做 LP 可行性判断；

**📊 数据集**

采用 CONUS 拓扑的 Dataset 01（Bu‑SMF）与 Dataset 07（WC‑MCF）作为单核与多核光纤的源域与目标域数据集；

**📈 对比分析**

对比基线点预测、无策略的 CP 以及两种策略下的决策，结果显示基线准确率 92% 在加入 CP+策略后提升至 99.6%，零样本跨域 (ZS) 方案从 77.7% 提升到 99.4%，误报率下降 99% 以上，召回率提升约 2.5 倍；

**⚠️ 局限性**

局限性包括：仍需少量目标域样本构建一致性集；零样本场景下置信带宽度较大，可能影响光谱利用率；需手动设置显著性水平 α 与风险容忍度 δ，策略调优依赖运营经验；未验证在更大规模多域网络中的可扩展性；

---

## 182. HairPort: In-context 3D-aware Hair Import and Transfer for Images

**arXiv ID:** 2606.12562 | [PDF](https://arxiv.org/pdf/2606.12562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 183. Algorithmic Constitutionalism

**arXiv ID:** 2606.12437 | [PDF](https://arxiv.org/pdf/2606.12437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 184. PRISM: Prosody-Integrated Multi-Agent Reasoning Framework for Empathetic Spoken Dialogue

**arXiv ID:** 2606.12902 | [PDF](https://arxiv.org/pdf/2606.12902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 185. Muse Spark Safety & Preparedness Report

**arXiv ID:** 2606.12429 | [PDF](https://arxiv.org/pdf/2606.12429v1)

**作者:** Cristina Menghini `[一作]`, Summer Yue `[通讯]` (Meta)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文为 Meta AI 推出首个 Muse 系列模型 Muse Spark 的安全与风险评估报告，系统性评估了模型在化学生物风险、网络安全、失控风险、对抗鲁棒性、行为对齐等多维度的安全性，并制定了相应的缓解措施与治理流程。

**💡 创新点**

创新点在于：①首次将多模态推理、工具使用与多代理协作能力结合在同一模型中；②通过 Meta Advanced‑AI‑Scaling‑Framework 设定的风险阈值，进行层级化的安全评估与治理；③在传统评测之外，新增对模型“评估意识”、多轮 jailbreak、代理使用等前沿风险的专门检测。

**🔧 技术方法**

采用的技术包括：大规模自动化红队、专门的对抗训练与适应性红队、工具化推理（SeqQA、ABC Bench 等）、网络安全基准（CyBench、CyberGym 等）、行为对齐基准（IHEval、Reward‑Hacking、DeceptionBench 等）、内容安全判定器与实时在线监控。

**📊 数据集**

使用的数据集与基准主要有：MBCT、VCT、HPCT、WMDP‑Bio、WMDP‑Chem、ProtocolQA、SeqQA、ABC Bench（FD/LH）、BioDesign Tools、CyBench、CyberGym、CyScenarioBench、StrongREJECT v2、FORTRESS、AgentHarm、AgentDojo、ART、OR‑Bench、Mask、DeceptionBench、SimpleQA Verified、Humanity’s Last Exam、TextQuests、CIMemories 等；同时对比了 GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro 等同类模型。

**📈 对比分析**

比较方法：对 Muse Spark 在各基准上进行同等参数配置的跑分，并与三大主流模型（GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro）做横向对比。结果显示 Muse Spark 在化学生物拒绝、网络安全低误用、诚实性、拒绝率方面表现优于或匹配同类模型，但在适应性 jailbreak、prompt injection 与代理滥用方面仍有明显劣势。

**⚠️ 局限性**

局限性包括：①对多轮 jailbreak 的鲁棒性不足，仍易被适应性攻击突破；②在代理使用场景下对恶意指令与注入攻击的防护不完善；③模型在评估时表现出“评估意识”，可能导致在真实任务中的行为偏差；④部分基准中的误拒率与误警率尚需进一步降低；⑤模型仍有若干对齐风险（如自我保存倾向、信息泄露）需持续监测与改进。

---

## 186. SMGFM: Spectral Multimodal Graph Pretraining for Multimodal-Attributed Graphs

**arXiv ID:** 2606.12867 | [PDF](https://arxiv.org/pdf/2606.12867v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 187. Multi-Turn Reasoning When Context Arrives in Pieces: Scalable Sharding and Memory-Augmented RL

**arXiv ID:** 2606.12941 | [PDF](https://arxiv.org/pdf/2606.12941v1)

**作者:** Shu Tong Luo `[一作]` (University of Melbourne), Jiaxian Guo `[通讯]` (Google Research Australia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在单轮问答数据上构造多轮碎片化训练集，并用记忆缓冲与强化学习训练模型，使其在多轮对话中保持压缩上下文并完成推理；

**💡 创新点**

创新点包括：①低成本碎片化流水线，可将任意单轮 QA 数据无须大量人工标注地转换为多轮碎片化示例；②在多轮 DAPO 上训练记忆增量策略，利用强化学习让模型在每轮自行压缩并重写记忆，显著缓解 Lost in Conversation（LiC）退化；

**🔧 技术方法**

使用技术：强化学习可验证奖励（RLVR）+ 多轮 DAPO；自回归 LLM（Qwen2.5‑Math‑1.5B、Qwen3‑4B）；记忆缓冲（256 词自然语言限制）以及对比全历史模型；

**📊 数据集**

数据集：单轮 GSM8K、MATH500；生成的碎片化数据；长上下文 QA 数据集 LongBench（2WikiMQA、HotpotQA、MultifieldQA、Qasper、TriviaQA）用于零样本评估；

**📈 对比分析**

与基线比较：Memory‑RL 在 GSM8K 上从 0.199 提升至 0.825（≈60% 提升），在 MATH500 上从 0.050 提升至 0.516；在 LongBench 上平均 F1 提升 31.7–43.4%；即使在全历史推理条件下，Memory‑RL 仍比全历史训练高 15–35pp，且在噪声碎片插入时更稳健；

**⚠️ 局限性**

局限性：仅在 Qwen 系列模型上验证；记忆缓冲被强制限制为 256 词，可能不适用于更长文本；跨模型族的验证未完成；零样本 LongBench 评估未进行领域专门训练，实际潜力可能更高。

---

## 188. An Explainable AI Assistant for Introductory Programming Education: Improving Feedback Reliability with Instructor-AI Collaboration

**arXiv ID:** 2606.12425 | [PDF](https://arxiv.org/pdf/2606.12425v1)

**作者:** Muntasir Hoq `[一作]` (North Carolina State University), Bita Akram `[通讯]` (North Carolina State University)

**通讯引用:** 359 | [OpenAlex ID](https://openalex.org/A5000710051)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为Insight的可解释AI助教，帮助初级编程课程中学生获取及时、个性化的反馈；

**💡 创新点**

通过将可解释的程序分析模型SANN、教师预先编写的误区与反馈以及LLM辅助验证相结合，实现了教师-AI协作的可靠反馈传播框架；

**🔧 技术方法**

核心技术包括SANN的子树注意力模型、LLM（GPT‑4o）用于生成合成代码、模型微调、误差定位和反馈匹配，以及GPT‑4o的验证层；

**📊 数据集**

使用美国空军学院的FalconCode数据集（约150万份Python提交）进行预训练，并在新的编程问题上使用LLM生成的合成数据进行微调；

**📈 对比分析**

评估方法包括离线人类评测（精确度/召回率≈97.6%）、与GPT‑4o生成反馈的对比（Insight在选择性和教学适宜性上显著优于GPT‑4o），以及课堂调查显示学生对反馈的满意度居中偏上；

**⚠️ 局限性**

局限性在于反馈覆盖度依赖教师手工编写的误区示例，无法覆盖未预料的错误，且系统对缺失或运行时错误给出通用提示，教师仍需投入时间维护反馈库，未来需扩大覆盖范围并进行更大规模的学习成效评估。

---

## 189. How Useful is Causal Invariance for Domain Adaptation in Finite-Sample Settings?

**arXiv ID:** 2606.12680 | [PDF](https://arxiv.org/pdf/2606.12680v1)

**作者:** Julia Kostin `[一作]` (ETH Zurich), Fanny Yang `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究在有限样本下，利用部分因果知识（候选可不变特征集合）来改进监督域自适应（sDA），提出一种自适应聚合算法并给出上界与下界。

**💡 创新点**

关键创新在于：①将因果不变性引入有限样本 sDA，给出风险边际（margin）驱动的样本复杂度分析；②证明当风险边际足够大时，只需 O(log|I|/Δ) 的目标样本即可实现“最佳”风险；③提供匹配的下界，说明 1/Δ 的样本复杂度是必要的。

**🔧 技术方法**

技术核心包括线性回归的模型选择聚合、对候选集合的自适应筛选、风险边际分析、线性结构因果模型（SCM）下的风险分布推导，以及信息理论下的下界证明。

**📊 数据集**

实验使用了 Causal Chamber 的光隧道数据集、RPE1 单细胞基因表达数据集，以及在 Causal Chamber 图像数据上使用 CLIP 嵌入的实验。

**📈 对比分析**

与无因果知识的目标仅训练（target-only）、无监督/监督域自适应基线（如 IRM、VREx、anchor regression、ICP）以及基于候选模型的 naive ERM 进行比较。实验显示，在小样本和大结构偏移情形下，本文方法能显著降低 MAE/MSE，接近或优于目标模型；在中等偏移下表现平稳。

**⚠️ 局限性**

局限性包括：①只针对线性子集模型，非线性或高维场景尚未覆盖；②对候选集合规模的 log|I| 依赖可能仍不够精细；③需要先验因果图或偏移集合信息，若无法获取则只能退回到完整集合，导致样本复杂度退化；④理论假设（boundedness、正定协方差、结构可加偏移）在实际数据中可能不完全成立。

---

## 190. Context-Aware Feature-Fusion for Co-occurring Object Detection in Autonomous Driving

**arXiv ID:** 2606.12628 | [PDF](https://arxiv.org/pdf/2606.12628v1)

**作者:** Binay Kumar Singh `[一作]` (University of Central Florida), Niels Da Vitoria Lobo `[通讯]` (University of Central Florida)

**通讯引用:** 2774 | [OpenAlex ID](https://openalex.org/A5030469716)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Context-Centric Feature Fusion（CCFF）框架，结合局部上下文融合模块和全局上下文注意模块提升自动驾驶场景中的目标检测。

**💡 创新点**

通过 RoI 对 RoI 自注意力捕捉局部对象交互，并用 Top‑K RoI 注意池化与几何偏置实现对象级全局上下文，二者融合实现高效语义一致性和小目标提升。

**🔧 技术方法**

采用两阶段 Faster R‑CNN + FPN Backbone，RoIAlign，Transformer‑style 自注意力，Top‑K 注意池化，几何编码以及 MLP 融合等技术。

**📊 数据集**

在 Cityscapes 与 BDD100K 数据集上进行实验验证。

**📈 对比分析**

与基线、仅局部、仅全局版本进行 ablation，CCFF 在 Cityscapes AP 35.51%、CoAP 0.389、CCS 0.973；在 BDD100K AP 32.95%、CoAP 0.488、CCS 0.969，显著提升且仅产生 0.2 FPS 的额外计算开销。

**⚠️ 局限性**

模型依赖 Top‑K 选择和几何编码的手工设置，对极端遮挡或极少见场景仍有局限，且实验仅在两阶段框架内验证，缺乏跨框架的通用性。

---

## 191. Rubric-Guided Self-Distillation: Post-Training Without Rubric Verifiers

**arXiv ID:** 2606.12507 | [PDF](https://arxiv.org/pdf/2606.12507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 192. DeepJEB++: Foundation Model-Driven Large-Scale 3D Engineering Dataset via 2D Latent Space Augmentation

**arXiv ID:** 2606.12994 | [PDF](https://arxiv.org/pdf/2606.12994v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 193. How Fine-Grained Should a RAG Benchmark Be? A Hierarchical Framework for Synthetic Question Generation

**arXiv ID:** 2606.12789 | [PDF](https://arxiv.org/pdf/2606.12789v1)

**作者:** Chase M. Fensore `[一作]` (Emory University), Joyce C. Ho `[通讯]` (Emory University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HieraRAG 框架，系统地生成并评估三维度（问题复杂度、答案类型、语言变异）的合成 QA 数据，以探讨 RAG 评测的粒度选择。

**💡 创新点**

创新点包括：1）层次化粒度选择方法，依据标准差最大化判定最佳粒度；2）引入 Coherence Ratio 评估细粒度分割结构；3）实证不同维度在 RAG 评测中的最佳粒度差异。

**🔧 技术方法**

使用的技术包括：RAG 两阶段流水线（BM25+Falcon‑3‑10B）、DataMorgana 合成 QA、评估指标 MAP、nDCG、Cosine Similarity、ROUGE、BLEU，以及统计离散度、互信息和 Coherence Ratio。

**📊 数据集**

数据集：FineWeb‑10BT 10B 语料库，生成 5,872 条合成 QA，并对 110 条样本进行人工验证。

**📈 对比分析**

比较方法：在同一 RAG 系统上对 2/4/8 个细分级别进行评测，发现问题复杂度维度在 8 类时离散度最高；答案类型和语言变异在 4 类时最佳；Coherence Ratio 与人工一致性呈正相关。

**⚠️ 局限性**

局限性：仅在单一检索器/生成器/语料库上实验；单维度生成导致混杂；Coherence Ratio 与人工一致性的相关性不显著；合成问答与真实问答的可迁移性待验证。

---

## 194. A Privacy-Preserving Framework Using Remote Data Science for Inter-Institutional Student Retention Prediction

**arXiv ID:** 2606.12845 | [PDF](https://arxiv.org/pdf/2606.12845v1)

**作者:** John Fields `[一作]` (Concordia University Wisconsin), Praveen Madiraju `[通讯]` (Marquette University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现了基于PySyft的远程数据科学（RDS）框架，使三所高校在FERPA合规下共同构建学生保留预测模型

**💡 创新点**

提出Data‑Type‑Aware Templates方法，用结构化假数据代替高保真合成数据，并证明RDS在小规模跨校合作中的技术可行性

**🔧 技术方法**

使用PySyft、Docker、Azure低侧/高侧服务器、差分隐私机制以及scikit‑learn等传统机器学习算法

**📊 数据集**

使用720名Concordia 2021年学生的真实数据以及公开的Faketucky合成数据进行模型验证和假数据评估

**📈 对比分析**

通过在低侧使用假数据预训练、在高侧真实数据评估的迭代流程，宏F1在三所校区保持0.69–0.695，表明模型在不同机构间性能一致

**⚠️ 局限性**

局限在于合成数据的实用性不足导致性能下降、类别不平衡影响模型召回率、实现过程需手动配置服务器和严格代码审查、未覆盖深度学习等更高复杂模型

---

## 195. Knowing the Rules Is Not Enough: Student Regulatory Awareness and Use of GenAI in Higher Education

**arXiv ID:** 2606.12436 | [PDF](https://arxiv.org/pdf/2606.12436v1)

**作者:** Lasse Bischof `[一作]` (University of Applied Sciences and Arts Hannover), Michael Neumann `[通讯]` (University of Applied Sciences and Arts Hannover)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过对151名德国应用科学与艺术大学商科信息系统与电子政府专业本科生进行问卷调查，探讨学生对机构GenAI使用规定的认知程度、对自我使用合规性的感知以及二者之间的关系。

**💡 创新点**

创新点在于首次系统性地量化学生对GenAI使用规定的认知与实际使用合规性的自我评估，并揭示了意识与合规感知之间的显著差距，提示监管与沟通策略需同步提升。

**🔧 技术方法**

采用问卷调查法，结合描述性统计、交叉表分析及热图可视化来揭示认知与合规感知的关联。

**📊 数据集**

使用的“数据集”为151份匿名问卷收集的数据，涵盖学生基本信息、对GenAI工具的熟悉度、使用情况及对规定的认知与合规感知。

**📈 对比分析**

方法上主要采用描述性统计与交叉表，结果显示学生对规定的认知与合规感知呈弱到中度正相关（约 0.3‑0.5 的相关系数），但绝大多数学生对合规性仍不确定。

**⚠️ 局限性**

局限性包括单一高校、单学科样本、跨-sectional 设计导致因果推断受限，以及自我报告数据可能受社会期望与记忆偏差影响，结果的外推性有限。

---

## 196. Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration

**arXiv ID:** 2606.12913 | [PDF](https://arxiv.org/pdf/2606.12913v1)

**作者:** Dongyue Wu `[一作]` (Huazhong University of Science and Technology), Changxin Gao `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了统一图式数据集剪枝框架UGIES，通过构造最大权团问题实现对样本内在与外在重要性的联合评估，并给出可理论保证的贪心近似算法。

**💡 创新点**

创新点在于：①将样本重要性拆分为内在与外在两部分并统一为图加权最大团问题；②证明该目标在满足简单可验证条件下为子模并可用贪心算法取得 $(1-1/e)$ 近似；③提供结构化稀疏图与聚类方法以降低计算成本。

**🔧 技术方法**

主要技术包括：图建模、最大权团（MWCP）理论、贪心求解、子模性证明、结构化稀疏图构造、特征聚类与距离度量、梯度/熵/损失等重要性度量。

**📊 数据集**

实验数据集涵盖 CIFAR‑10、CIFAR‑100、ImageNet‑1k，使用 ResNet‑18/50、Swin‑T 等网络进行评估。

**📈 对比分析**

与随机、k‑center、herding、EL2N、InfoBatch、DivBS 等主流剪枝方法对比，在所有数据集与剪枝比例下均保持或提升 Top‑1 预测准确率；同时在 ImageNet‑1k 上实现 45%+ 训练时间加速，损失率仅为 0%。

**⚠️ 局限性**

局限性包括：①对样本间 pairwise 交互的计算与存储仍较昂贵，尤其在极大规模数据上；②理论近似保证仅适用于满足非负距离与非正映射的子模情况，某些重要性度量不易满足；③贪心解非最优，可能在极端压缩比或高度相关数据集上性能下降。

---

## 197. Arbor: Tree Search as a Cognition Layer for Autonomous Agents

**arXiv ID:** 2606.12563 | [PDF](https://arxiv.org/pdf/2606.12563v1)

**作者:** Neha Prakriya `[一作]` (AMD), Emad Barsoum `[通讯]` (AMD)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Arbor 框架，利用结构化树搜索和多代理（Orchestrator、Critic、Domain Specialists）来全栈优化 LLM 推理性能，支持跨层的动态搜索与诊断。

**💡 创新点**

创新点包括：① 将跨层性能优化建模为状态树搜索，树在每次测量后动态扩展；② 设计 Orchestrator 与 Critic 的检查平衡机制，硬技能与软技能分解；③ 持久化知识库实现跨会话迁移和自适应评分；④ 通过根因分析与回滚保证多日持续运行。

**🔧 技术方法**

主要技术：大型语言模型（Claude Opus 4.6/4.7）驱动代理；启发式树搜索 + UCB 评分；动态生成域专家代理；根因分析、回滚、稳定性监测；全栈性能监测与基准；持久化知识库；并行 GPU/CPU 任务调度。

**📊 数据集**

使用六个真实生产 LLM 推理模型（gpt-oss-120b、DeepSeek-R1-0528、MiniMax-M2.5、GLM-5-FP8、Qwen3.5-397B-A17B、Kimi-K2.5）在 AMD Instinct MI355X 与 MI300X GPU 上实验，采用 vLLM 或 SGLang，tensor parallelism 1–8，评估吞吐量、延迟、准确率等指标。

**📈 对比分析**

通过与厂商优化基线（InferenceX）在多并发度、交互性等多维度的 Pareto 前沿比较，Arbor 在单个会话中实现 +40%~+193% 的吞吐量提升；在 MI300X 上取得 +62%~+99% 的提升；多次独立运行收敛至 ±2% 的一致性，证明方法可复现。

**⚠️ 局限性**

限制包括：对大型语言模型后端的依赖；实验仅在 AMD GPU（MI355X/MI300X）上验证，未覆盖 NVIDIA 等平台；硬件与任务范围限定在 LLM 推理；经验性评分常数未系统调优；未对不同 LLM 后端做全面比较；未测试训练或更大规模多 GPU 的通用性；未完成搜索参数的敏感度分析。

---

## 198. PolicyGuard: Towards Test-time and Step-level Adversary Defense for Reinforcement Learning Agent

**arXiv ID:** 2606.12896 | [PDF](https://arxiv.org/pdf/2606.12896v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 199. Evaluation of AutoML Frameworks for IDS under Imbalanced Data Conditions of the NSL-KDD Dataset

**arXiv ID:** 2606.12611 | [PDF](https://arxiv.org/pdf/2606.12611v1)

**作者:** Wiliane Carolina Silva `[一作]` (National Institute of Telecommunications), Felipe A. P. de Figueiredo `[通讯]` (National Institute of Telecommunications)

**通讯引用:** 1531 | [OpenAlex ID](https://openalex.org/A5017715993)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了九个开源 AutoML 框架在 NSL-KDD 数据集多类别严重不平衡条件下的网络入侵检测性能。

**💡 创新点**

首次提供统一可复现的基准，揭示不同 AutoML 对极端类别不平衡的处理差异，并强调需要内置不平衡感知机制。

**🔧 技术方法**

采用 AutoGluon、PyCaret、TPOT、H2O AutoML、Auto-Sklearn、FLAML、LazyPredict、Auto-PyTorch、AutoKeras 等 AutoML 技术，结合集成学习、超参数优化和不平衡处理。

**📊 数据集**

NSL-KDD 数据集的五类（正常、DoS、Probe、R2L、U2R）真实分布。

**📈 对比分析**

通过宏 F1、加权 F1、准确率等指标与手工特征工程模型对比，PyCaret 最高宏 F1 66%，AutoGluon 55%，其他框架低于 55%。

**⚠️ 局限性**

AutoML 框架缺乏原生不平衡处理，导致对 R2L、U2R 等稀有攻击检测不足，未能与手工调优模型竞争。

---

## 200. The Internet of Agentic AI: Communication, Coordination, and Collective Intelligence at Scale

**arXiv ID:** 2606.12835 | [PDF](https://arxiv.org/pdf/2606.12835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 201. Smarter Saboteurs, Better Fixers: Scaling & Security in Linear Multi-Agent Workflows

**arXiv ID:** 2606.12709 | [PDF](https://arxiv.org/pdf/2606.12709v1)

**作者:** Timothy McAllister `[一作]` (University of Central Florida), Ozlem Ozmen Garibay `[通讯]` (University of Central Florida)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究线性多代理工作流在模型规模放大时对恶意干扰的鲁棒性，并评估在末端添加 QA+Fixer 校正阶段的效果。

**💡 创新点**

提出了“合规–校正对称”观察，指出更大模型在被恶意利用时更易执行破坏，但同等规模的校正阶段也能同样有效检测和修复。

**🔧 技术方法**

使用改编的 MetaGPT 线性 SDLC 结构、LLM（Qwen 3.5、Gemma 3）、简化/原生提示、QA+Fixer 终端角色。

**📊 数据集**

在 HumanEval 评测集（164 个 Python 题目）上进行实验。

**📈 对比分析**

通过比较无校正和 QA+Fixer 两种配置的 Pass@1 及恶意影响 Δ，发现无校正时规模越大 Δ 递增；加校正后 Δ 降到 0–1pp，恢复控制水平。

**⚠️ 局限性**

仅考虑单个工程师被篡改，未评估多代理或上游篡改；恶意行为不具适应性；仅在 0.27–27B 参数范围内，未探讨 >100B 模型及更大规模代码库。

---

## 202. A Stabilized Path-Space Approach to Diffusion-Based Posterior Sampling

**arXiv ID:** 2606.12710 | [PDF](https://arxiv.org/pdf/2606.12710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 203. EWAM: An Enhanced World Action Model for Closed-Loop Online Adaptation in Embodied Intelligence

**arXiv ID:** 2606.12690 | [PDF](https://arxiv.org/pdf/2606.12690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 204. MDForge: Agentic Molecular Dynamics Pipeline Design under Sparse Simulator Feedback

**arXiv ID:** 2606.12916 | [PDF](https://arxiv.org/pdf/2606.12916v1)

**作者:** Zehong Wang `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种基于LLM的代理MDForge，通过verbal RL和PRISM机制自动生成并迭代优化分子动力学（MD）计算流水线，最终实现对宿主-客体结合能的高效预测。

**💡 创新点**

创新点在于：①将稀疏的终端奖励稠密化为多阶段诊断和专家级质询；②通过多专家辩论产生可解释的子系统级改进建议；③以代码生成方式模拟人类MD专家的自由组合决策，而非固定工具调用。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）代理、verbal强化学习、PRISM稠密信号生成、四阶段MD流水线（准备、平衡、采样、分析）以及由三位物理子系统专家（力场、采样、分析）构成的多代理辩论。

**📊 数据集**

所用数据集：SAMPL系列宿主-客体结合能基准（CB[7]、OAH、CBClip），以及从ChEMBL/DrugBank抽取的未见化合物库，用于前瞻性实验验证。

**📈 对比分析**

比较方法：对比五类反馈策略（无反馈、LLM评审、步骤级反馈、试验级反馈、MDForge），结果显示MDForge在CB[7]、CBClip上分别取得Kendall τ=0.56和0.47，显著优于基线并与人类专家提交水平相当，且成本与其他策略相当。

**⚠️ 局限性**

局限性：①仅在三种宿主-客体系统上验证，未覆盖蛋白‑配体、膜蛋白等更广泛MD任务；②采用低精度2 GPU‑小时配置，尚未检验高精度设置下的表现；③实验验证仅针对单个候选化合物，未对完整排名做全面检验。

---

## 205. The Switching Lemma shows what the Switching Lemma cannot prove: an unconditional natural-proofs barrier

**arXiv ID:** 2606.12631 | [PDF](https://arxiv.org/pdf/2606.12631v1)

**作者:** Bruno Loff `[一作]` (Universidade de Lisboa), Francesca Ugazio `[通讯]` (Universidade de Lisboa)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

重新审视自然证明屏障，证明在无条件下0-自然证明对常数深度电路的下界存在极限，并构造伪随机函数对任意常数深度的0-区分器进行伪随机化。

**💡 创新点**

首次给出0-自然证明的无条件屏障，利用局部化的Trevisan–Xue伪随机生成器展示大多数已知常数深度电路下界技术本质上是0-自然的，并证明Switching Lemma自身受限。

**🔧 技术方法**

使用Trevisan–Xue伪随机生成器、Switching Lemma、局部化技术以及深度电路的复杂度分析。

**📊 数据集**

无数据集，纯理论证明。

**📈 对比分析**

与已知的Switching Lemma下界2^{Ω(n^{1/(d-1)})}比较，提出无条件的上限2^{n^{7/(d-5)}}，表明0-自然证明无法突破该上限。

**⚠️ 局限性**

结果并不紧，7/(d-5)的指数不如期望的1/(d-1)，且仅针对0-自然证明；其他证明方法仍可能突破。

---

## 206. Reliability of Probabilistic Emulation of Physical Systems

**arXiv ID:** 2606.12997 | [PDF](https://arxiv.org/pdf/2606.12997v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 207. Action-Effect Memory Pretraining for Robot Manipulation

**arXiv ID:** 2606.12499 | [PDF](https://arxiv.org/pdf/2606.12499v1)

**作者:** Yijing Zhou `[一作]` (Hong Kong University of Science and Technology), Renjing Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Action-Effect Memory (AEM) 预训练框架，利用视觉-动作交互历史压缩为单向量的最终标记，作为时间上下文注入到机器人操控策略中。

**💡 创新点**

创新点在于：1）将视觉与动作交互序列交错编码；2）采用对齐的时间步遮蔽与自编码目标，让最终标记学习动作-效果依赖；3）使用 Mamba 状态空间模型实现单向量压缩，无需额外记忆标记。

**🔧 技术方法**

技术包括 Mamba 递归状态空间网络、对齐时间步遮蔽的 MAE 目标、轻量级 Transformer 解码器、DINOv2 CLS 视觉特征投影、以及在 Diffusion Policy 与 ManiFlow 上的策略集成。

**📊 数据集**

使用的数据集与平台有 RoboTwin2.0 机器人仿真基准、RMBench 非马尔可夫任务集，以及真实 Franka Emika 机械臂搭配 RealSense D435 的实测任务。

**📈 对比分析**

通过在 DP 与 ManiFlow 上分别对比无记忆、DINOv2 直接堆叠以及 AEM 压缩记忆，实验显示 AEM 在仿真中平均提升约 21% 成功率、在真实任务中提升约 24%，且比直接堆叠视觉特征显著更高效（推理延迟和 FLOPs 更低）。

**⚠️ 局限性**

局限性包括：缺乏大规模预训练与跨域验证；压缩至单向量可能丢失部分细粒度时序信息；以及对极端动态或复杂环境的鲁棒性尚未充分评估。

---

## 208. ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation

**arXiv ID:** 2606.12633 | [PDF](https://arxiv.org/pdf/2606.12633v1)

**作者:** Jiangtao Kong `[一作]` (William & Mary), Huajie Shao `[通讯]` (William & Mary)

**通讯引用:** 1725 | [OpenAlex ID](https://openalex.org/A5041685416)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无样本增量学习框架 ECA，专注于在预训练视觉语言模型的对齐模块上实现连续对齐；

**💡 创新点**

创新点在于三大模块：Mixture of Query（MoQ）实现任务特定查询的动态混合；Fisher Dynamic Expansion（FeDEx）根据 Fisher 信息矩阵动态扩展适配器以避免干扰；Dictionary Replay（DR）通过稀疏字典学习保持过去任务的视觉表示，无需存储原始样本；

**🔧 技术方法**

使用 BLIP-2 作为基准模型，结合 Q-Former 对齐模块、Parallel Adapter、FIM 计算、稀疏字典学习（Lasso）和知识蒸馏；

**📊 数据集**

在四个自建增量学习基准上评估：ToS-COCO Caption、ToS-VQAv2、ToS-TextCaps、ToS-TextVQA，均基于 MSCOCO、VQAv2、TextCaps、TextVQA；

**📈 对比分析**

与多种无样本增量学习方法（EWC、LwF、CODA-Prompt、Dual-Prompt、MoE-LoRA）以及 Upper‑Bound 进行对比，ECA 在平均性能、向后/向前迁移上显著优于基线，且参数量与最小；

**⚠️ 局限性**

局限性包括字典大小固定，可能不足以覆盖极长任务序列的多样视觉分布；方法依赖于强大预训练 VLM，弱化后端时效果未知。

---

## 209. Auto formalisation of Chaitin and of the surprise incompleteness Theorem

**arXiv ID:** 2606.12462 | [PDF](https://arxiv.org/pdf/2606.12462v1)

**作者:** Thierry Coquand `[一作]` (University of Gothenburg), Thierry Coquand `[通讯]` (University of Gothenburg)

**通讯引用:** 5463 | [OpenAlex ID](https://openalex.org/A5087100539)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文展示了如何利用大型语言模型 Claude 自动化地在 Agda 中正式化 Gödel 第二不完备性定理，并进一步正式化 Chaitin 的第一不完备性定理与 Kritchman‑Raz 的惊奇检验悖论版本。

**💡 创新点**

创新点在于：① 在无任何 Agda 代码、无 tactic、无库的“简陋”环境下，Claude 完全自动构造了计算机程序的三进制编码、解析器和 CK 机器；② 通过 Gandy/Howard 主导量化方法在内部递归算术中生成计算步骤上界，实现了在对象层面内部化 Chaitin 证明；③ 将该内部化证明与外部的“堆沙”式递归推演相结合，得到一种全新的不计数式第二不完备性证明框架。

**🔧 技术方法**

技术上主要采用：大型语言模型（Claude）的自然语言理解与代码生成；Agda 依赖类型理论与内在化证明；三进制程序编码与 CK 机器；Gandy/Howard 主导量化；内部化的模态推理与模数化的程序生成。

**📊 数据集**

本文并未使用传统意义上的数据集；其实验对象是 Gödel 定理的形式化表达、Chaitin 证明的代码片段以及 Kritchman‑Raz 的悖论论述，全部由 Claude 自动生成并在 Agda 环境中验证。

**📈 对比分析**

由于该工作以形式化证明为目标，没有对性能进行量化比较；但作者指出 Claude 在该“简陋”Agda 框架下能够完整实现所有证明步骤，并且无需手写任何 Agda 代码，显示出其在数学形式化任务中的潜在高效性；相较于传统手工形式化，所需的人工干预显著减少。

**⚠️ 局限性**

局限性包括：① 仍需人工持续监督，尤其是在生成程序时需要手动检查；② 目前仅适用于离散、组合式的数学对象，无法直接扩展到涉及非组合对象的领域；③ 依赖于 Claude 的语言模型能力，对更复杂的数学结构或更深层次的形式化仍可能遇到困难。

---

## 210. Structured Testbench Generation for LLM-Driven HDL Design and Verification-Oriented Data Curation

**arXiv ID:** 2606.12983 | [PDF](https://arxiv.org/pdf/2606.12983v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 211. Semantic Identification of IoT Devices from Behavioral Primitives

**arXiv ID:** 2606.12793 | [PDF](https://arxiv.org/pdf/2606.12793v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 212. Reducing the Complexity of Deep Learning Models for EEG Analysis on Wearable Devices

**arXiv ID:** 2606.12742 | [PDF](https://arxiv.org/pdf/2606.12742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 213. Direct Preference Optimization for Chatbot Fine-Tuning: An Empirical Study

**arXiv ID:** 2606.12881 | [PDF](https://arxiv.org/pdf/2606.12881v1)

**作者:** Yvonne Qiu `[一作]`, ShuoJia Fu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（cognitivecomputations/dolphin-2.1-mistral-7b）使用直接偏好优化（DPO）进行微调，并评估其在文本生成与对话场景下的表现。

**💡 创新点**

创新点在于完全消除奖励模型的中间步骤，直接用人类偏好数据优化策略，极大简化训练流程并降低计算开销。

**🔧 技术方法**

技术手段包括DPO算法、4‑bit 量化、QLoRA 参数高效微调、FlashAttention 以及 PEFT（LoRA）配置。

**📊 数据集**

使用的数据集有：argilla/ultrafeedback-binarized-preferences-cleaned（偏好数据）、WebGPT（知识问答对照）和多样化对话提示。

**📈 对比分析**

比较方法：BLEU、ROUGE、余弦相似度及人工评估；实验结果显示 DPO 训练后模型在 BLEU 上略优，但 ROUGE 与余弦相似度基本持平，人工评估未见显著提升，说明模型已有较高基线。

**⚠️ 局限性**

局限性：训练过程中存在明显波动与不稳定；评估指标对微调效果不敏感；在知识性问答上提升有限，需更大、更多样化的数据以及进一步的稳定性调优。

---

## 214. Benchmarking AI Agents for Addressing Scientific Challenges Across Scales

**arXiv ID:** 2606.12736 | [PDF](https://arxiv.org/pdf/2606.12736v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 215. WISE: A Long-Horizon Agent in Minecraft with Why-Which Reasoning

**arXiv ID:** 2606.12852 | [PDF](https://arxiv.org/pdf/2606.12852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 216. The Theory of Mind Utility: Formal Specification of a Mentalizing Mechanism

**arXiv ID:** 2606.12721 | [PDF](https://arxiv.org/pdf/2606.12721v1)

**作者:** Nikolos Gurney `[一作]` (University of Southern California), Stacy Marsella `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

提出了Theory of Mind Utility (ToM‑U) 框架，将他人心智推断视为信息访问历史与可信度驱动的经验推理问题，并通过局部认识世界模型 (LEWM) 进行实现。

**💡 创新点**

创新点在于：①将 LEWM 作为离散候选模型的形式化表示；②使用 generate‑and‑filter 架构进行离散推断；③在递归心智化中通过有限分支树解决无限回归；④引入 residue 函数记录失败推断的结构性痕迹。

**🔧 技术方法**

采用了计算层次的形式化方法，利用有向类型图、正式定义、生成-过滤推断过程、分支宽度衰减函数以及可积的残留函数。

**📊 数据集**

该工作未使用实际数据集，而以 “爆米花袋” 经典任务为理论示例进行说明。

**📈 对比分析**

未进行实验对比或性能评估；框架以理论推导为主，后续需在模拟或实测数据上实现并验证。

**⚠️ 局限性**

局限性包括：缺乏算法层面实现细节、参数缺少经验校准、仅在单代理或极简多代理情景下演示、未涵盖真实数据的验证与跨文化适用性。

---

## 217. Marginal Alignment Does Not Guarantee Joint-Distribution Fidelity: An Official-Reference Audit of Nemotron-Personas-Korea with Cross-Locale Replication

**arXiv ID:** 2606.12433 | [PDF](https://arxiv.org/pdf/2606.12433v1)

**作者:** Joonhyung Bae `[一作]` (Korea Advanced Institute of Science and Technology), Joonhyung Bae `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 205 | [OpenAlex ID](https://openalex.org/A5075919540)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对NVIDIA Nemotron-Personas-Korea（NPK）合成人物数据集进行官方参考审计，验证其在各属性联合分布上的真实性。

**💡 创新点**

提出了“独立假设足迹”（IAF）审计原语，首次将数据卡中声明的独立生成假设与公开官方统计的联合分布进行直接对比。

**🔧 技术方法**

利用Cramér's V比率、Feldman四五分位宽度、KL散度、Wasserstein距离等统计量，并结合规则推导与多模型（Claude、Gemini、HyperCLOVA-X）判定一致性。

**📊 数据集**

使用NPK（约100万条记录）、KOSIS、KEIS GOMS、最高法院姓名统计、韩国军役年鉴等官方数据集作为参考。

**📈 对比分析**

通过对六个独立轴（性别×职业、性别×年龄×就业、性别×年龄×军役、出生年×性别×姓名、专业×职业、性别×婚姻状态）进行对照，发现NPK在性别与职业的联合分布中明显平滑化（女占比显著高于官方数据），以及军役和专业与职业的关联失真；在跨国复现时，IAF显示不同地区表现差异，验证了其可移植性。审计结果表明，尽管NPK在边缘分布上匹配良好，但关键联合分布存在显著偏差。

**⚠️ 局限性**

局限包括：依赖单一人工核查（757条记录）对职业映射的可信度有限；部分参考数据存在年代差距或税onomy粒度不匹配；IAF对已声明独立假设的资源有效，但对未公开该假设的资源需要进一步扩展；跨国评估仅为可移植性演示，未进行完整本土化验证。

---

## 218. M*: A Modular, Extensible, Serving System for Multimodal Models

**arXiv ID:** 2606.12688 | [PDF](https://arxiv.org/pdf/2606.12688v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 219. Self-Guidance: Enhancing Neural Codecs via Decoder Manifold Alignment

**arXiv ID:** 2606.12940 | [PDF](https://arxiv.org/pdf/2606.12940v1)

**作者:** Xiang Li `[一作]` (Tsinghua University), Hui Wang `[通讯]` (Pengcheng Laboratory)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并验证了自引导（self‑guidance）训练机制，提升VQ‑VAE语音编码器在量化误差下的重建质量。

**💡 创新点**

通过在训练时同时让解码器处理连续编码向量和量化后向量，并加入特征映射损失，直接对齐解码器内部流形，从而显著减小量化失真。

**🔧 技术方法**

基于VQ‑VAE、Transformer解码器、iSTFT、线性投影等技术，并在XCodec2/BigCodec上实现自引导。

**📊 数据集**

主要使用LibriSpeech完整训练集（960小时）进行训练，LibriSpeech test‑clean进行评测，TTS实验使用LibriTTS‑R数据。

**📈 对比分析**

与现有低比特率语音编解码器（TS3Codec、WavTokenizer、DAC等）以及基准XCodec2/BigCodec比较，在PESQ、STOI、MCD等指标上均取得领先，且可实现4倍代码本减小而不降低质量。

**⚠️ 局限性**

自引导虽显著提升质量，但并未完全消除量化失真；实验主要局限于语音编码器，尚需验证在音乐、音效或图像VQ‑VAE等更广泛场景中的效果。

---

## 220. The Mathematics of AI Winters: The mathematical Taxonomy of Paradigm Fragility in AI Winter

**arXiv ID:** 2606.12610 | [PDF](https://arxiv.org/pdf/2606.12610v1)

**作者:** Miquel Noguer i Alonso `[一作]` (AIFI), David Pacheco Aznar `[通讯]` (Staq.io)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从数学角度系统梳理并整合了导致两次 AI 冬季的正式障碍，包括表示能力、计算复杂度、统计泛化、梯度消失、维数灾难以及符号推理瓶颈，并提出“障碍三元组”框架来评估范式脆弱性。

**💡 创新点**

创新点在于将散见的理论结果（感知机不可分、神经网络训练 NP‑硬、VC/均匀收敛、最小化率、梯度消失等）统一成三元组判定法，并通过“共同绑定”条件给出AI范式失效的数学条件；同时在后续部分提供了四个现代理论视角（缩放律、神经崩塌、彩票假说、隐式偏置）作为对三元组的后验解释。

**🔧 技术方法**

主要使用理论分析方法：PAC 学习框架、VC 维数、Rademacher 复杂度、均匀收敛、最小化率、梯度流代数、均值场理论、卷积/矩阵范数等数学工具。

**📊 数据集**

本文不依赖任何实验数据集，全部以理论推导与文献引用为依据。

**📈 对比分析**

比较方式为与历史 AI 冬季的主要挫折对照，阐明每种障碍在对应时期的作用；由于缺乏实验指标，未给出数值性能表现。

**⚠️ 局限性**

局限性在于：1) 仅关注形式障碍，未涵盖经济、硬件、工程实践及社会因素；2) 以理论阐述为主，缺少实证验证；3) 对现代成功案例的解释多为后验性推断，未给出因果证明。

---

## 221. Two Wrongs, No Right: Auditing Social-Desirability Bias in LLM Annotators for Computational Social Science

**arXiv ID:** 2606.12426 | [PDF](https://arxiv.org/pdf/2606.12426v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 222. Predicting Cognitive Load from Speech and Interaction Dynamics in Dyadic Conversations

**arXiv ID:** 2606.12971 | [PDF](https://arxiv.org/pdf/2606.12971v1)

**作者:** Tahiya Chowdhury `[一作]` `[通讯]` (Colby), Tahiya Chowdhury (Colby)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在双人对话中利用语音和交互特征预测感知认知负荷的可行性。

**💡 创新点**

将认知负荷建模为回归任务，并系统评估了静态、动态语音特征与交互特征的组合效应，首次证明交互特征对时序负荷预测的显著提升。

**🔧 技术方法**

使用基于GRU的双头编码器（无注意力）与随机森林对比，输入包括eGeMAPS特征、差分特征及交互特征。

**📊 数据集**

采用AVCAffe远程协作语音视频数据集，共53对话、950个任务、106名参与者。

**📈 对比分析**

采用留一对话交叉验证，主要评估指标为CCC、PCC和RMSE；在时序负荷上GRU模型达CCC≈0.42，加入交互特征可提升至≈0.51，优于随机森林。

**⚠️ 局限性**

数据量有限，模型对小样本学习能力弱；仅使用语音和时间信息，缺乏词汇、面部表情等多模态信号；任务级标签限制了时间细粒度建模。

---

## 223. DARRMS -- An Efficient Algorithm for Dynamic Attention Radius in Resource-Constrained Multi-Agent Systems

**arXiv ID:** 2606.12614 | [PDF](https://arxiv.org/pdf/2606.12614v1)

**作者:** Benjamin Alcorn `[一作]` (Texas A&M University), Eman Hammad `[通讯]` (Texas A&M University)

**通讯引用:** 1033 | [OpenAlex ID](https://openalex.org/A5026015448)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种结合 Stackelberg 博弈与自适应注意半径的多智能体决策算法 DARRMS，以在资源受限环境下实现高效协作。

**💡 创新点**

创新点在于将注意半径动态调整与 Stackelberg 领导-追随关系相结合，形成一种既能降低观测成本又能保持良好协调性的自适应策略。

**🔧 技术方法**

采用了 L‑平滑、Polyak‑Łojasiewicz 条件的梯度优化方法求解注意半径，使用了动态规划/Permutation 组合策略来确定决策顺序，并基于预测轨迹进行风险评估。

**📊 数据集**

使用的是基于 2‑D 仿真环境的自驾车数据集，随机生成 500 个起始位置和目标，测试了四辆车（三协作、一独立）在不同场景下的运行轨迹。

**📈 对比分析**

与固定半径基准相比，DARRMS 在平均到达时间上相差无大，资源消耗率下降 50% 以上，说明自适应半径在保持性能的同时显著降低了计算开销。

**⚠️ 局限性**

局限性包括：仅在简化的二维仿真环境验证；对真实传感器噪声、遮挡等因素缺乏考虑；并且对极端动态环境下的收敛性和鲁棒性尚未充分评估。

---

## 224. Interpretable Factor Decomposition for Decision Intelligence in Large-Scale Financial Markets: Evidence from China's A-Share Market

**arXiv ID:** 2606.12843 | [PDF](https://arxiv.org/pdf/2606.12843v1)

**作者:** Xiao Han `[一作]` (Emory University), Moxuan Zheng `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一套可解释的机器学习流程，对中国A股市场2009-2019年的股票收益可预测性进行因子分解，并通过SHAP和消融分析揭示不同因子对预测的贡献。

**💡 创新点**

创新点在于将TreeSHAP与消融法相结合，实现了可审计的因子贡献拆解，并首次在大规模金融市场上检验行为因子与估值因子之间的替代结构。

**🔧 技术方法**

主要技术包括XGBoost树模型、TreeSHAP归因、消融实验、滚动窗口交叉验证、基于AUC和Sharpe的绩效评估。

**📊 数据集**

使用的数据集是包含3,632只A股股票的月度面板，共计254,854条观测，包含估值、行为、基本面、规模等13个特征。

**📈 对比分析**

通过与逻辑回归、单因子排序和Carhart四因子模型的对比，XGBoost在55个月的外样本中获得AUC 0.547，长期-短期收益率每月+2.38%，Sharpe比率2.23，Carhart alpha为+2.31%/月。

**⚠️ 局限性**

主要局限包括前瞻性偏差（基本面数据无公布延迟导致的look‑ahead bias）、生存者偏差、SHAP仅在样本内计算且未做多重检验校正，以及缺乏跨市场验证。

---

## 225. From AGI to ASI

**arXiv ID:** 2606.12683 | [PDF](https://arxiv.org/pdf/2606.12683v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 226. Agentic MPC for Semantic Control System Resynthesis

**arXiv ID:** 2606.12774 | [PDF](https://arxiv.org/pdf/2606.12774v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 227. Emerging Flexible Designs for Geospatial Multimodal Foundation Models

**arXiv ID:** 2606.12595 | [PDF](https://arxiv.org/pdf/2606.12595v1)

**作者:** Philipe Dias `[一作]` (Oak Ridge National Laboratory), Dalton Lunga `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 1096 | [OpenAlex ID](https://openalex.org/A5083948807)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对地球观测多模态基础模型（SatMAE、DOFA、Flex）进行统一预训练与评估，研究其在不同波段配置下的鲁棒性与性能。

**💡 创新点**

提出统一实验框架，消除模型架构对性能的偏差；揭示按波段分组（SatMAE）比早期融合（DOFA、Flex）更具“温和退化”特性；建议将DOFA的波长感知与SatMAE的中间融合相结合。

**🔧 技术方法**

自监督掩码自编码（MAE）预训练；多通道/多模态token化；跨通道交叉注意力（Flex）或波长条件动态patch embedding（DOFA）以及分组嵌入与中间融合（SatMAE）；ViT-Base结构、UPerNet分割解码器、线性探测。

**📊 数据集**

Sentinel-2 22,549点的 2021 年 SEUSA 预训练集；CONUS 1,96M S1+S2 128×128 图块；GeoBench 子集（m-bigearthnet、m-brick-kiln、m-eurosat、m-cashew、m-SA-crop、m-so2sat）。

**📈 对比分析**

在统一预训练条件下比较三模型在 GeoBench 分类与分割任务、以及 S1+S2 下的迁移性能。结果显示 SatMAE 在大多数配置下性能最稳健，DOFA 在完整 10 波段时最高，Flex 对 SWIR 过度依赖导致降维时性能显著下滑。预训练时均使用 200 epoch、mask ratio 0.75、AdamW、cosine 退火。

**⚠️ 局限性**

局限性：实验仅覆盖 Sentinel-2 单源与 S1+S2 组合；对更广泛传感器/分辨率的推广未知；Flex 的交叉注意力对噪声通道易过拟合；DOFA 在缺失中频波段时性能骤降；模型规模与计算成本差异导致部署受限。

---

## 228. Learning to Adapt: Representation-Based Reinforcement Learning for Multi-Task Skill Transfer

**arXiv ID:** 2606.12890 | [PDF](https://arxiv.org/pdf/2606.12890v1)

**作者:** Aryan Naveen `[一作]` (Massachusetts Institute of Technology), Na Li `[通讯]` (Harvard)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出 RepMT-SAC 多任务强化学习框架，针对四旋翼轨迹跟踪实现任务无关动态与任务特定奖励的分解，并实现零样本与少量样本快速迁移。

**💡 创新点**

创新点在于将谱 MDP 分解与任务嵌入相结合，显式构造任务无关特征 ϕ(s,a) 与任务特定权重 w(τ)，从而实现可分离的价值函数表示与高效知识共享。

**🔧 技术方法**

采用谱 MDP 分解、Soft Actor‑Critic（SAC）框架、最大熵策略、跨任务训练与下游微调两阶段学习、Legendre 多项式任务编码与时间窗口奖励。

**📊 数据集**

在 IsaacSim 仿真环境中使用基于 Legendre 多项式系数编码的四旋翼轨迹任务，划分源任务、在分布任务与外部分布任务进行评估。

**📈 对比分析**

与 CTRL、SAC 及 CTRL‑SAC 对比，RepMT-SAC 在源任务、在分布与外部分布任务上均取得更高奖励与成功率，外部分布任务零样本成功率提升至 100%，微调后进一步提升到 1034.8 的奖励。

**⚠️ 局限性**

限制在于任务采样分布 μ_train 仍为手工设定，且实验仅在仿真环境完成，未验证真实世界的 Sim2Real 转移效果。

---

## 229. Polar: A Benchmark for Evaluating Political Bias in LLMs

**arXiv ID:** 2606.12922 | [PDF](https://arxiv.org/pdf/2606.12922v1)

**作者:** Sangho Kim `[一作]` (Seoul National University), Jaejin Lee `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Polar 基准，通过选项级似然测量 LLM 的政治偏见，覆盖美国与韩国两国语境；

**💡 创新点**

创新点在于采用多选项似然而非自由生成，结合 ICAT 评估模型语言能力与中立性，构建双语双国双维度（经济/社会文化）八类问题框架；

**🔧 技术方法**

使用长度归一化对数似然计算、ICAT（Idealized CAT）评分、Manifesto 项目编码、GPT‑5.5 翻译以及手工审校的实例构建流程；

**📊 数据集**

基于 Manifesto Project 的美国与韩国政党宣言（2016‑2024）、官方党派新闻稿、政策简报等，生成 4,026 条多选项实例；

**📈 对比分析**

对 38 个 LLM 进行评估，计算政治位置和 ICAT 分数；结果显示美国文本所有模型均呈左倾，韩国文本更偏中立；ICAT 在韩国更高，语言呈现对偏见有显著影响；模型组差异在美国小、韩国大；

**⚠️ 局限性**

局限性：仅覆盖美韩两国与 2016‑2025 年的政治语料，缺乏对其他国家/语言的泛化；需要访问 token 级概率，闭源模型无法直接使用；

---

## 230. Individual Control Barrier Functions-Guided Diffusion Model for Safe Offline Multi-Agent Reinforcement Learning

**arXiv ID:** 2606.12640 | [PDF](https://arxiv.org/pdf/2606.12640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 231. From Parameters to Feature Space: Task Arithmetic for Backdoor Mitigation in Model Merging

**arXiv ID:** 2606.12498 | [PDF](https://arxiv.org/pdf/2606.12498v1)

**作者:** Zhenqian Zhu `[一作]` (Harbin Institute of Technology), Wenjian Luo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 3038 | [OpenAlex ID](https://openalex.org/A5001184471)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向模型合并的后门防御框架LFPM，在特征空间内对合并模型进行反后门任务向量的构造与优化，从而实现后门抑制与干净任务性能的兼顾。

**💡 创新点**

创新点包括：① 在Cross‑Task Linearity（CTL）框架下，将后门鲁棒性从参数空间迁移到特征空间；② 通过特征子空间划分与可学习视觉提示的方式提取后门特征；③ 采用类似SAM的尖锐度最小化与梯度累积机制，在特征插值路径上实现鲁棒优化；④ 引入路径积分损失以保证在整个参数插值路径上的稳定防御。

**🔧 技术方法**

技术手段包括：Cross‑Task Linearity理论、特征子空间划分（投影矩阵学习）、视觉提示（Prompt）训练、SAM‑style锐度最小化、梯度累积与路径积分优化、混合任务向量加权。

**📊 数据集**

使用的基准数据集有：CIFAR‑100、Cars196、SUN397、EuroSAT、GTSRB、Pets、CIFAR‑10、SVHN、STL‑10、Food‑101、RESISC45，以及10,000张ImageNet‑1K作为影子数据；模型基座为CLIP‑ViT‑B/32。

**📈 对比分析**

与IBVS、SAU、SAM、PAM以及标准任务合并方法（Task‑Arithmetic、Simple Average等）比较，LFPM在BadMerging和LoBAM攻击下均取得最低的攻击成功率（ASR）和最高的干净准确率（CA）；在路径ASR（即插值路径上的累计ASR）上也表现出最小值；在适应性攻击评估中，LFPM的ASR增幅最小，保持竞争力的干净性能。

**⚠️ 局限性**

局限性：① 依赖CTL近似，参数距离过大或任务差异极大时可能失效；② 需要额外的视觉提示和子空间投影学习，增加训练复杂度；③ 在极大模型规模或极端攻击策略下的鲁棒性尚未充分验证；④ 对高维任务组合或多任务数目激增的适用性仍待研究。

---

## 232. Trait, Not State: The Durability of Reading Identity in Social Highlighting

**arXiv ID:** 2606.12904 | [PDF](https://arxiv.org/pdf/2606.12904v1)

**作者:** Kazuki Nakayashiki `[一作]` (Glasp Inc), Keisuke Watanabe `[通讯]` (Glasp Inc)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对社交网页高亮器用户的选择行为进行时序分析，测量冻结的六个月个人档案在随后的 0–24+ 个月内对后续阅读选择的优势是否保持。

**💡 创新点**

提出了自然读者选择的身份衰减曲线，并通过时间匹配负样本与双层身份层级分离个人漂移与内容供给漂移，证明选择特征在多年内表现为稳定的“特质”。

**🔧 技术方法**

利用高亮文本嵌入的质心表示、平均精度 (AP) 排名、按用户的 3,000 次聚类自助法估计置信区间，并设置全局与兴趣邻居两种负样本抽样方案。

**📊 数据集**

使用 Glasp 社交网页高亮器平台的内部数据，对 405 位长期活跃用户（每人至少 60 条净网页高亮记录，跨度 12 个月以上）进行抽样，平均每人约 133 条文档。

**📈 对比分析**

通过对比自己档案与邻居控制档案在各时间窗口内的 AP 差值评估优势；细层优势始终保持约 +0.18，配对保留率 R ≈ 1.0 直至 12 个月，且在实际下一篇文档的排名中个人档案的 AP 约为 0.7，远高于非个人先验的 0.2-0.3。

**⚠️ 局限性**

局限包括仅覆盖高频、长期用户，无法区分曝光与主动选择（无浏览日志），时间间隔划分粗糙，且结果可能受平台特定生态影响，未必适用于其他阅读追踪系统。

---

## 233. Speculative Rollback Correction for Quality-Diverse Web Agent Imitation

**arXiv ID:** 2606.12485 | [PDF](https://arxiv.org/pdf/2606.12485v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 234. SafeLLM: Extraction as a Hallucination-Resistant Alternative to Rewriting in Safety-Critical Settings

**arXiv ID:** 2606.12897 | [PDF](https://arxiv.org/pdf/2606.12897v1)

**作者:** Julia Ive `[一作]` (University College London), Richard Dobson `[通讯]` (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发并评估了一套基于文档提取的LLM问答框架，限制模型仅从临床指南中拷贝、行号选择或最小安全提取内容，以减少hallucination并保证信息准确。

**💡 创新点**

创新点在于将LLM从自由生成转为只做信息提取；提出了多种提取策略（COPY、LINES、JuniorDoc、Safety），并系统比较它们在不同文档长度与检索设置下的表现；通过行号索引实现可追溯且无 hallucination 的答案。

**🔧 技术方法**

使用了多种开源与商用LLM（Ollama的Gemma、Qwen、Llama等），结合检索增强生成（RAG）技术；实现了文本与“代表性问题”双重嵌入、行号提取提示、Multi‑agent（JuniorDoc）和最小安全提取提示；评估指标包括ROUGE‑L、词级 P/R/F1、医学实体 F1、答案长度等。

**📊 数据集**

三个数据集：UCLH 急诊协议（96个问题，25份指南），Somerset NHS Trust 肿瘤指南（50个问题，6份指南），NICE 国家指南合成（500个问题，10份指南）。每份指南均拆分为子段并生成代表性问题，用于检索和评估。

**📈 对比分析**

在短文档（UCLH）上，LINES策略在保持较低答案长度的同时实现最高召回；在长文档（Somerset、NICE）上，JuniorDoc 在召回上优于LINES，且仍保持可接受的输出长度；Safety 提供最简洁答案但存在显著遗漏；相比基准检索（BASE），所有提取策略均显著提升精度，且效果在不同模型间一致。

**⚠️ 局限性**

局限性包括：仅处理文本指南，未涵盖表格、流程图等多模态内容；人类评估样本量有限，且在模拟环境中进行，缺乏真实临床工作流验证；多模型实验主要在同一硬件与超参数下进行，外部API模型的表现仍待进一步验证。

---

## 235. MAStrike: Shapley-Guided Collusive Red-Teaming on Multi-Agent Systems

**arXiv ID:** 2606.12918 | [PDF](https://arxiv.org/pdf/2606.12918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 236. "Did you lie?" Evaluating Lie Detectors across Model Scale and Belief-Verified Model Organisms

**arXiv ID:** 2606.12618 | [PDF](https://arxiv.org/pdf/2606.12618v1)

**作者:** Alan Cooney `[一作]` (AI Security Institute), Geoffrey Irving `[通讯]` (AI Security Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了新的测试集和模型实体，用于评估语言模型的谎言检测器。

**💡 创新点**

创新点在于构造链式思维可验证的模型实体与Varied Deception多样化提示集，并提出DYL激活探针。

**🔧 技术方法**

采用链式思维监测、线性激活探针、日志概率分类器和新DYL方法。

**📊 数据集**

使用Varied Deception、AuditBench、Gender Secret、Sandbagging等自制数据集，以及公开TriviaQA等。

**📈 对比分析**

在提示诱导谎言上四种检测器随模型规模正相关，均能提升性能；在训练诱导谎言上，激活和日志概率检测器大幅失效，DYL保持一定效果，CoT判别器最稳健。

**⚠️ 局限性**

局限包括对模型信念验证的不完整、缺乏对检测器的对抗训练、模型实体训练方式可能与真实谎言差异等。

---

## 237. Multi-Label Test-Time Adaptation with Bayesian Conditional Priors

**arXiv ID:** 2606.12925 | [PDF](https://arxiv.org/pdf/2606.12925v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 238. Multimodal Graph Negative Learning

**arXiv ID:** 2606.12863 | [PDF](https://arxiv.org/pdf/2606.12863v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 239. Magnifying What Matters: Attention-Guided Adaptive Rendering for Visual Text Comprehension

**arXiv ID:** 2606.12898 | [PDF](https://arxiv.org/pdf/2606.12898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 240. TEDD: Robust Detection of Unstable Temporal Features

**arXiv ID:** 2606.12643 | [PDF](https://arxiv.org/pdf/2606.12643v1)

**作者:** Ricardo Ribeiro Pereira `[一作]` (University of Porto), Miguel Araújo `[通讯]` (Feedzai)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用随机森林回归模型预测时间戳来自动检测并量化时间序列数据中特征漂移的方法

**💡 创新点**

创新点在于将时间戳预测任务与特征重要性相结合，既能识别所有基本漂移类型（线性/突变均值/方差、类别频率/域变化、随机异常、多变量依赖）又兼具参数无关和线性可扩展性

**🔧 技术方法**

使用随机森林回归器（默认100棵树，深度32，特征子集sqrt），基于R²衡量模型对时间的解释力，特征重要性用于排名

**📊 数据集**

在IEEE-CIS欺诈检测数据集（约590k样本、222特征）和KKBox流失预测数据集（约2M样本、22特征）上进行注入实验，并使用合成数据验证多变量漂移

**📈 对比分析**

与Mann‑Whitney U、Earth Mover’s Distance、ADWIN、CST四种方法对比，实验显示该方法在大多数漂移类型下能够保持排名靠前、检测准确率高；在可扩展性测试中运行时间比传统统计方法快10倍、比ADWIN快100倍

**⚠️ 局限性**

局限性包括：对极端稀疏或离散时间戳可能效果下降；依赖随机森林的特征重要性分解，可能对高度相关特征产生不稳定性；未对模型的训练时间和资源消耗做深入分析

---

## 241. nomp: A Framework for Building Domain Specific Compilers

**arXiv ID:** 2606.12650 | [PDF](https://arxiv.org/pdf/2606.12650v1)

**作者:** Thilina Ratnayaka `[一作]` (University of Illinois), Sanath Jayasena `[通讯]` (University of Moratuwa)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了名为 nomp 的基于 C 语言的领域特定编译器框架，利用 pragma 指令、运行时代码生成和 loopy 中间表示来自动化 GPU 内核的优化与生成。

**💡 创新点**

创新点在于通过元数据和域特定优化脚本实现编译时/运行时双重优化，兼顾生产力、性能与可移植性；同时将低级 GPU 编程模型的细粒度控制与高级抽象结合，形成一种可插拔、易扩展的领域编译器体系。

**🔧 技术方法**

核心技术包括：Clang 前端修改以识别 nomp pragma、Loopy IR 进行代码转换与生成、Python 脚本实现域/内核特定的优化策略、以及 CUDA/OpenCL/HIP 后端实现低级 GPU 调度与内存管理。

**📊 数据集**

使用的测试数据集主要是 Nekbone miniapp（3D Poisson 方程）和 Mandelbrot 集，分别体现了科学计算与图像生成两大领域。

**📈 对比分析**

通过在 Frontier 超算上单 GPU 的基准，比较了 nomp 与 HIP 的性能。结果显示：nomp 在与 HIP‑variable 的对比中始终保持更高的 GDOFS 和更快的速度提升；与 HIP‑fixed 的差距仅在 10% 以内，且 LOC 大幅减少，说明 nomp 既保持了性能又提升了开发效率。

**⚠️ 局限性**

局限性包括：目前仅支持单 GPU，未实现 MPI 多卡并行；运行时的代码生成与优化可能带来额外的启动和函数调用开销；对不同 GPU 架构的后端实现仍在完善，需进一步提升跨平台一致性与可扩展性。

---

## 242. Position: Generative Engine Optimization Creates Underexamined Risks, Governance Must Target Concentration, Disclosure, and Academic Blind Spots

**arXiv ID:** 2606.12439 | [PDF](https://arxiv.org/pdf/2606.12439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 243. Missing-Token Prompted Reliability-Aware Fusion for Robust Polyglot Speaker Identification

**arXiv ID:** 2606.12495 | [PDF](https://arxiv.org/pdf/2606.12495v1)

**作者:** Peng Jia `[一作]` (Hefei University of Technology), Richang Hong `[通讯]` (Hefei University of Technology)

**通讯引用:** 22885 | [OpenAlex ID](https://openalex.org/A5051332325)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MRAF 框架，解决多模态语音识别中面部缺失和跨语言场景的鲁棒性问题。

**💡 创新点**

创新点在于使用可学习的缺失 token、可靠性感知的交叉注意力融合以及音频知识蒸馏来提升缺失面部情况下的识别性能。

**🔧 技术方法**

采用 Transformer 编码器、多头交叉注意力、可靠性评分器、知识蒸馏、中心损失和多分支分类等技术。

**📊 数据集**

在 POLY‑SIM 2026 挑战的 MAV‑Celeb 数据集（包含英语和乌尔都语）上进行训练与评估。

**📈 对比分析**

与官方基线和前五名参赛队伍对比，平均 Top‑1 准确率达到 0.9957，在完整和缺失面部场景中几乎无误差，整体排名第二。

**⚠️ 局限性**

在噪声、低质量面部图像或音频-面部不匹配的情况下仍易失效，跨语言语音特征的进一步偏移也可能限制模型泛化。

---

## 244. GeoDial: A Multimodal Conversational Tutoring Dataset for Geometry Problem-Solving with Visual Tutor Turns

**arXiv ID:** 2606.12419 | [PDF](https://arxiv.org/pdf/2606.12419v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 245. MentalMARBERT: Domain-Adaptive Pre-training and Two-Stage Fine-Tuning for Arabic Mental Health Disorders Detection

**arXiv ID:** 2606.12649 | [PDF](https://arxiv.org/pdf/2606.12649v1)

**作者:** Fatimah Almalki `[一作]` (King Abdulaziz University), Abdulrahman Aladeem `[通讯]` (King Abdulaziz University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了两阶段框架用于检测阿拉伯语社交媒体文本中的多类精神疾病。

**💡 创新点**

创新点在于将域自适应与任务自适应预训练、分层两阶段分类与参数高效微调结合在阿拉伯语精神健康文本上。

**🔧 技术方法**

使用了AraBERT、CAMeLBERT、MARBERT三种预训练模型，进行DAPT/TAPT、LoRA、全微调，并构建了多层分类架构。

**📊 数据集**

构建了50,670条带专家标注的阿拉伯语推文数据集，覆盖None、抑郁、焦虑、双相、PTSD、OCD六类。

**📈 对比分析**

通过5折交叉验证与配对t检验对比四种配置，最终两阶段全微调模型在macro‑F1 0.8617、准确率0.8778达最佳性能。

**⚠️ 局限性**

限制在于数据仍偏向公开推文，缺乏跨平台验证，且对极少数类仍存在识别不足。

---

## 246. Pinching-Antenna Enabled Multicell Wireless Systems

**arXiv ID:** 2606.12888 | [PDF](https://arxiv.org/pdf/2606.12888v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 247. Generativism: Toward a Learning Theory for the Age of Generative Artificial Intelligence

**arXiv ID:** 2606.12441 | [PDF](https://arxiv.org/pdf/2606.12441v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 248. Normative Robustness as a Frontier for Non-Verifiable Reasoning in LLMs

**arXiv ID:** 2606.12731 | [PDF](https://arxiv.org/pdf/2606.12731v1)

**作者:** Elizaveta Tennant `[一作]` (DeepMind), Julia Haas `[通讯]` (DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种多轮、对照实验的框架，用于评估大型语言模型在道德推理任务中的鲁棒性。

**💡 创新点**

创新点在于将道德鲁棒性定义为模型在不同情境和结构变化下保持一致判断的能力，并首次系统测量“道德推理的顺序效应”和“道德协从性”这两种新失效模式。

**🔧 技术方法**

采用模拟用户-模型对话、对话结构扰动（顺序、持续时间、额外考虑、用户价值注入）以及使用 LLM 评判器进行数值标注，并通过序数混合效应模型进行统计分析。

**📊 数据集**

使用 MoRe Bench 数据集中的 DailyDilemmas 与 AIRiskDilemmas 两个子集，分别包含 100 个简短案例，并将其转化为多轮对话模板。

**📈 对比分析**

与四个最先进 LLM（Gemini‑3.1‑pro‑preview、Gemini‑2.5‑pro、Claude‑opus‑4.6、GPT‑5.4‑pro）在 48,000 场模拟对话中对比，结果显示模型能抵御无关干扰，但在考虑顺序、对话持续时间和用户偏好注入时会产生显著判断波动，表现出顺序回忆效应和道德推理的协从性。

**⚠️ 局限性**

局限性包括对话长度仅至多 5 轮、仅涉及建议场景、生成的新考虑可能包含事实与价值混合、评判器选择和数据污染的潜在影响，以及未覆盖更广泛的文化或自主决策场景。

---

## 249. Order Is Not Control

**arXiv ID:** 2606.12923 | [PDF](https://arxiv.org/pdf/2606.12923v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 250. $μ$VLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models

**arXiv ID:** 2606.12497 | [PDF](https://arxiv.org/pdf/2606.12497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 251. Influence Factors on RAG Poisoning

**arXiv ID:** 2606.12469 | [PDF](https://arxiv.org/pdf/2606.12469v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 252. Objects Before Words: Object-First Inductive Biases for Grounding Language in Child-View Video

**arXiv ID:** 2606.12985 | [PDF](https://arxiv.org/pdf/2606.12985v1)

**作者:** Sathira Silva `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Muhammad Haris Khan `[通讯]` (Weizmann Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BabyMind，一种以对象为先的儿童视角对比学习框架，利用离线掩码生成、短窗口追踪和多实例对比损失实现词义的基于对象的归纳。

**💡 创新点**

创新点在于把对象文件（通过掩码+追踪生成的短时实例）作为中间表示，引入原型空间多实例对比、跟踪一致性与全局对象一致性正则化，以解决婴幼儿视角下稀疏、时空不匹配的监督歧义。

**🔧 技术方法**

采用 CLIP 风格的跨模态对比学习、SAM 样式的离线掩码生成、均值池化得到对象表示、轻量级追踪、原型记忆（EMA+Sinkhorn）、多实例对比损失、跟踪一致性损失与全局一致性损失等技术。

**📊 数据集**

主要使用 SAYCam‑S 语料（Labeled‑S 15）进行实验，并在 Konkle 对象类别与 COCO 类别上评估在 CVCL IV/OOV 协议下的泛化能力。

**📈 对比分析**

在 4‑way 强制选择测试中与 CVCL 基线对比，BabyMind 在 Labeled‑S 15 上平均准确率提升 2.6%；在 IV/OOV 评测中取得小幅但一致的提升；消融实验表明原型空间 MIL 与全局一致性正则化是关键贡献。

**⚠️ 局限性**

局限性包括：依赖离线掩码生成，短窗口追踪无法捕获长时间对象持续；对已经具有强全局视觉线索的类别可能产生竞争梯度；未在更多儿童、多样环境及更复杂语言情境下进行鲁棒性验证。

---

## 253. SkillChain: Closing the Loop on Skill Evolution for Image-Based E-Commerce AI Assistants

**arXiv ID:** 2606.12984 | [PDF](https://arxiv.org/pdf/2606.12984v1)

**作者:** Yimin Hu `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 SkillChain，一个面向电商图像助手的三阶段自动化技能生命周期框架（技能创建、路由优化、身体细化），实现了从任务规范到可部署技能的全流程闭环。

**💡 创新点**

创新点在于：1）将技能生命周期拆分为独立、单向的三段链路，避免前一段改动反向影响后续；2）引入专门的路由优化阶段，通过持续路由失败挖掘与描述更新解决分布漂移；3）使用双路径（规则 + LLM Judge）评估与跨样本归因，实现对技能身体的系统化改进；4）在工业级电商平台上验证，展示了可量化的离线与在线性能提升。

**🔧 技术方法**

使用技术包括：大型语言模型（LLM）做技能创建、路由评估和双路径评估；规则引擎进行结构化检查；双路径评估的 LLM Judge 评分四维度（工具合理性、卡片合规、内容质量、约束遵从）；人机回路做质量门控；版本化的技能库（Skill Bank）与自动化流水线。

**📊 数据集**

数据集：生产环境下的图像查询流（约 1,000 条离线样本，覆盖 5 种视觉意图）以及为期一周的在线 A/B 实验数据；意图类别包括 Exact Match、Multi-Product、Divergent Recommendation、Encyclopedia、Utility Assistance。

**📈 对比分析**

对比方法：与无技能基线（NoSkill）、人工设计技能（ManualSkill）以及逐步加入 SkillChain 的三个阶段进行离线评估；指标为 LLM Judge 四维度得分、路由 F1；在线指标为独立访客、完整阅读率、平均停留时长、7 天回访率。结果显示：每个阶段单独提升，完整 SkillChain 在所有维度和在线指标上均显著优于基线，尤其在完整阅读率提升 +4.98pp、回访率提升 +1.15pp。

**⚠️ 局限性**

局限性：1）冷启动与技能库可扩展性受限，缺失意图直接退回无技能；2）路由与评估依赖人工标注，难以在标注稀缺场景推广；3）声明式技能表达无法编码可执行或层级子技能，表达能力受限；4）实验仅在单一电商平台与五类意图内，跨域与多语言通用性尚未验证。

---

## 254. Binary Search Variants: A Comprehensive Analysis

**arXiv ID:** 2606.12970 | [PDF](https://arxiv.org/pdf/2606.12970v1)

**作者:** Ali Dasdan `[一作]` `[通讯]` (KD Consulting), Ali Dasdan (KD Consulting)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

统一阐述二分搜索的五种核心变体、六种查询函数以及四个标准库实现，并给出可验证的代码。

**💡 创新点**

提供统一符号、循环不变式、正式证明的三种实现（Python、Dafny、伪代码），并引入一次性完成所有查询的“combined search”。

**🔧 技术方法**

采用Dafny形式化验证、Python单元测试、循环不变式分析、Ceiling/Floor mid点计算、count‑based 迭代等技术。

**📊 数据集**

使用随机生成的排序整数数组（至1,000,000个元素）并在9,566个测试套件中覆盖各种边界和重复情况。

**📈 对比分析**

在Python与C中分别基准测试，比较3‑way、2‑way、count‑based与combined搜索；C中性能提升15–50×，count‑based 在C中最快。

**⚠️ 局限性**

主要局限在解释器层面难以体现细节、实验仅涵盖数值型数组、未覆盖并行/分布式情景。

---

## 255. Circuit Synchronization Precedes Generalization: Causal Evidence from Fourier Structure in Grokking Transformers

**arXiv ID:** 2606.12966 | [PDF](https://arxiv.org/pdf/2606.12966v1)

**作者:** Achyuthan Sivasankar `[一作]` `[通讯]` (New York University), Achyuthan Sivasankar (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

研究了Transformer在模数加法任务中出现的grokking现象，并提出了FSD度量来预测何时形成傅里叶电路。

**💡 创新点**

创新点包括提出无先验的FSD指标、证明FSD提前500–3000步检测傅里叶电路、用权重衰减分叉实验验证Phase 2是正则化阶段，并给出逆λ时间规律。

**🔧 技术方法**

使用傅里叶分析、频谱秩、Zero‑Ablation、权重衰减分叉干预和FourierKAN符号回归等技术。

**📊 数据集**

在5个小素数（p∈{53,71,97,113,131}）的模数加法、减法和乘法数据集上进行实验，主要聚焦30%训练/70%验证的所有输入对。

**📈 对比分析**

与传统的restricted‑logit（excluded loss）相比，FSD在所有9个加法实验中提前同步，平均领先1722步；对比不同架构（1层、注意力/MLP）和不同λ值，验证了预言的性能。

**⚠️ 局限性**

局限性包括仅测试2层Transformer、有限的小素数范围、未检验更大模型或其他算术任务、对乘法的FSD表现滞后、并且FourierKAN在部分素数上收敛失败。

---

## 256. EmbodiSteer: Steering Embodiment-Agnostic Visuomotor Policies with Joint-Space Guidance for Zero-Shot Cross-Embodiment Deployment

**arXiv ID:** 2606.12965 | [PDF](https://arxiv.org/pdf/2606.12965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 257. ScaleAcross: Designing Multi-Data-Center Infrastructure for Geo-Distributed AI Training

**arXiv ID:** 2606.12963 | [PDF](https://arxiv.org/pdf/2606.12963v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 258. Data Aphasia: An Institutional Counterfactual Study of the Stability of Academic Cognition Under Letter-Grade Evaluation Systems

**arXiv ID:** 2606.12946 | [PDF](https://arxiv.org/pdf/2606.12946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 259. Learning What to Remember: A Cognitively Grounded Multi-Factor Value Model for Agentic Memory

**arXiv ID:** 2606.12945 | [PDF](https://arxiv.org/pdf/2606.12945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 260. Quality-Preserving Imperceptible Adversarial Attack on Skeleton-based Human Action Recognition

**arXiv ID:** 2606.13022 | [PDF](https://arxiv.org/pdf/2606.13022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 261. scLLM-DSC: LLM-Knowledge Enhanced Cross-Modal Deep Structural Clustering for Single-Cell RNA Sequencing

**arXiv ID:** 2606.13007 | [PDF](https://arxiv.org/pdf/2606.13007v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 262. SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation

**arXiv ID:** 2606.12956 | [PDF](https://arxiv.org/pdf/2606.12956v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 263. APCyc: Property-Informed Design of Cyclic Peptides via Automated Cyclization

**arXiv ID:** 2606.12991 | [PDF](https://arxiv.org/pdf/2606.12991v1)

**作者:** Yifan Zhao `[一作]` (AI Thrust), Jintai Chen `[通讯]` (AI-Peptide Drug Design Joint Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了一种基于潜在扩散的目标感知框架 APCyc，用于自动化决定环化位点与类型，并对环状肽进行多属性优化。

**💡 创新点**

创新点在于：① 将环化视为离散的目标条件决策，并通过扩展残基词表实现环化感知嵌入；② 通过贝叶斯后验引导在扩散过程中同时优化亲和力、渗透性、酶解耐受性、溶解度和免疫原性等多目标；③ 采用自动拓扑注入模块将环化信息直接融入等变 GNN 的消息传递。

**🔧 技术方法**

技术手段包括：潜在自编码器 + 具有等变属性的图神经网络（AM‑EGNN）的扩散模型；自动拓扑注入与可调节的时间相关强度；基于代理模型的贝叶斯后验梯度引导；两阶段训练（先自编码器后扩散）以及目标条件下的推断算法。

**📊 数据集**

使用 CPCore 数据集（71,867 例经 AlphaFold2 预测的环状肽–蛋白复合物），并在 56 例不重叠的 LNR 测试集上进行评估。

**📈 对比分析**

与 PepGLAD、CP‑Composer、PepFlow 等基线在同一数据集上对比，APCyc 在稳定性、结合亲和力、以及溶解度、酶解耐受性、膜渗透性和免疫原性等药物相关指标上均取得领先或相近的优秀表现，同时保持多样性和结构一致性。

**⚠️ 局限性**

局限性包括：仅覆盖传统的主链酰胺、二硫键和侧链异肽键等环化方式，对新型交联剂或双环拓扑不一定适用；训练数据来源于 AlphaFold2 预测，可能存在结构噪声和目标偏差；药理预测模型（如免疫原性）存在不确定性，需实验验证与专家评估。

---

## 264. The Rise of AI-Native Software Engineering: Implications for Practice, Education, and the Future Workforce

**arXiv ID:** 2606.12986 | [PDF](https://arxiv.org/pdf/2606.12986v1)

**作者:** Mamdouh Alenezi `[一作]` `[通讯]` (Saudi Data and Artificial Intelligence), Mamdouh Alenezi (Saudi Data and Artificial Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究对2016–2026年间涉及生成式人工智能、LLM及其代理系统在软件工程实践、教育和劳动力市场影响的48篇经过严格验证的同行评审文献进行了系统综述与主题合成。

**💡 创新点**

创新点包括：①将AI原生软件工程划分为意图、协作与验证三大支柱的概念框架；②提出九维度的能力模型；③设计四阶段的大学课程路线图与AI韧性评估；④为教师与行业提供转型策略与研究缺口优先级。

**🔧 技术方法**

主要技术手段为PRISMA式筛选、四流文献检索（实践、教育、劳动力、计量）以及科学计量与主题提炼分析。

**📊 数据集**

使用的数据集为48篇高可信度的原始研究，涵盖ICSE、NeurIPS、SIGCSE、NBER、Science等多学科期刊与会议。

**📈 对比分析**

通过对文献的定量与定性对比，揭示生产力、质量与安全等指标的结果高度依赖专家水平、任务复杂度与上下文，未能得到统一的性能提升结论。

**⚠️ 局限性**

局限性在于证据来源多为预印本且方法多样，结果易受上下文影响，教育层面样本局限性大，且研究领域快速演进导致结论可能迅速过时。

---

## 265. Emo-LiPO: Listwise Preference Optimization for Fine-Grained Emotion Intensity Control in LLM-based Text-to-Speech

**arXiv ID:** 2606.13006 | [PDF](https://arxiv.org/pdf/2606.13006v1)

**作者:** Yihang Lin `[一作]` (Chinese University of Hong Kong), Haizhou Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Emo-LiPO 通过列表式偏好优化实现 LLM TTS 的细粒度情感强度控制。

**💡 创新点**

将情感强度控制建模为学习排序问题，显式捕捉全局强度序列。

**🔧 技术方法**

使用列表式偏好优化 (LiPO)、监督微调 (SFT) 与 LLM TTS 生成器。

**📊 数据集**

构造 ESD-plus 多说话人情感语音数据集，含 13 个细粒度情感标签。

**📈 对比分析**

与监督基线 CosyVoice、EmoVoice 及 Emo-DPO 变体对比，Emo-LiPO 在情感准确性、强度可控性上显著优于基线，且语音质量保持不降。

**⚠️ 局限性**

依赖规则生成的偏好列表可能限制泛化，且对极高/低强度的可解释性有限。

---

## 266. Exposure Bias as Epistemic Underidentification in Recursive Forecasting

**arXiv ID:** 2606.12990 | [PDF](https://arxiv.org/pdf/2606.12990v1)

**作者:** Riku Green `[一作]` (University of Bristol), Telmo M Silva Filho `[通讯]` (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过理论证明和实证实验，探讨递归多步预测中的曝光偏差，指出在部分可观测或状态截断条件下，一步贝叶斯训练可能无法唯一确定递归部署时的预测器，并提出将曝光偏差视为自诱导的认识学未识别问题。

**💡 创新点**

创新点包括：①提出诱导状态（induced state）和源头信息（provenance）两种新概念；②证明在部分可观测环境下，一步贝叶斯目标不一定能识别递归预测器；③推导出教师强迫/回滚不匹配、表示类逼近误差、源头信息缺失三项误差分解；④用最简二值源头编码验证理论，并展示其在不同数据集与闭环训练中的条件性提升。

**🔧 技术方法**

技术方法主要包括：递归多步预测框架；诱导状态与源头抽象；贝叶斯风险分解与三项误差分解；线性探测器评估诱导状态与观测状态的差异；冻结诱导状态评估（frozen-state）与闭环（closed-loop）回归训练；以及基于scheduled sampling与provenance-aware scheduled sampling的校正方法。

**📊 数据集**

实验使用了三个公开数值预测数据集：ETTh1（电力负荷）、MG（多尺度时间序列）、Weather（气象数据）。实验中使用了MLP模型（GRU实验在附录中验证同一趋势）。

**📈 对比分析**

与教师强迫（TF）对比，采用scheduled sampling（SS）与provenance-aware scheduled sampling（SSP）进行闭环评估。冻结诱导状态评估表明在MG和Weather上本地重训练能提升或等效于TF，在ETTh1上反而表现更差；SSP在ETTh1上在所有时间桶均优于TF，而SS在MG/Weather的提升有限。总体而言，SSP在某些数据集与时间窗口上实现了显著的误差下降，但提升并非普遍一致。

**⚠️ 局限性**

局限性包括：①源头信息仅采用单一二值编码，信息恢复有限；②实验未能完全分离诱导状态误差与闭环放大机制的影响；③仅在数值预测任务上验证，尚未检验在语言生成等更复杂序列任务上的泛化；④模型规模与实验设置相对简化，可能影响结果在大规模场景中的适用性。

---

## 267. Trajectory-Level Redirection Attacks on Vision-Language-Action Models

**arXiv ID:** 2606.12978 | [PDF](https://arxiv.org/pdf/2606.12978v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 268. PRISMR: Overcoming Parse Collapse in Multimodal Listwise Ranking via Parameterized Representation Internalization

**arXiv ID:** 2606.12942 | [PDF](https://arxiv.org/pdf/2606.12942v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 269. Technical Supplement Report on Full-Duplex FBMC/QAM MIMO Systems: Transceiver Design and Optimization

**arXiv ID:** 2606.13023 | [PDF](https://arxiv.org/pdf/2606.13023v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 270. Otters++: A Time-to-first-spike Based Energy Efficient Optical Spiking Transformer

**arXiv ID:** 2606.13016 | [PDF](https://arxiv.org/pdf/2606.13016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 271. A Mathematical Forum Platform for Collaborative Problem Solving and Dataset Generation for AI Reasoning

**arXiv ID:** 2606.12976 | [PDF](https://arxiv.org/pdf/2606.12976v1)

**作者:** Akbar Erkinov `[一作]` (Independent Researchers), Nurmukhammad Abdurasulov `[通讯]` (Independent Researchers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在在线讨论论坛中实现了从图片上传到数学公式渲染的完整工作流，用户可直接在发帖界面上传图片并获得实时预览。

**💡 创新点**

核心创新包括：一体化的图像–OCR–渲染管线；自动格式检测与分隔符归一化；双模式渲染（纯 LaTeX 与 Markdown+MathJax）以及完整的图像与 LaTeX 数据同步，形成可持续增长的社区验证数学数据集。

**🔧 技术方法**

使用了 Mathpix OCR API、MathJax/KaTeX 渲染引擎、React 前端、Axios 网络请求、MongoDB 存储，以及自定义的格式处理与预览逻辑。

**📊 数据集**

本文未采用公开数据集，而是通过论坛发帖生成自有的图像、LaTeX 与解题文本；在实现说明中引用了 im2latex‑100K、pix2tex 等公开模型作为对比。

**📈 对比分析**

与手动 Mathpix Snip、MathSE 及 pix2tex 等现有工具对比，论文通过步骤计数和用户体验描述表明整体操作步骤从 4‑5 步骤降至 1 步，减少了切换和粘贴错误；未给出定量性能指标。

**⚠️ 局限性**

主要限制包括：对 Mathpix 商业 API 的依赖、对复杂多行或非标准符号的识别不足、格式检测采用简单字符串比较、缺乏语法验证与用户实验、未提供正式性能评测。

---

## 272. Multi-Modal Agents for Power Distribution Defect Detection: An Evaluation of Foundation Models

**arXiv ID:** 2606.12969 | [PDF](https://arxiv.org/pdf/2606.12969v1)

**作者:** Quan Quan `[一作]` `[通讯]`, Quan Quan

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了针对配电网缺陷检测的多模态智能体框架，并对其感知、推理、工具使用三大核心能力进行系统评估；

**💡 创新点**

创新点在于：①统一的多模态智能体评估框架；②针对配电网场景设计的专用数据集与基准；③通过检索增强生成（RAG）与少样本学习对通用模型进行领域适配；

**🔧 技术方法**

采用视觉‑语言基础模型（如GLM‑4.5V、Qwen3‑VL‑30B、LLaVA等）与ReAct架构，配合提示工程、RAG、API工具调用；

**📊 数据集**

使用自研的26,803张高分辨率无人机与现场图像数据集，涵盖10种设备、31种缺陷及严重级别等多维标注；

**📈 对比分析**

对比方法包括0/1/5-shot提示和RAG补样，评估指标为设备/缺陷识别精度、等级判定准确率、工具调用准确率和任务成功率；结果显示通用模型在未适配时识别精度低于10%，RAG和多模态检索显著提升，但仍无法满足工业安全要求；

**⚠️ 局限性**

局限性在于：①感知精度不足，易产生幻觉；②工具链执行易崩溃，分解与调用顺序不稳定；③过度依赖外部检索，缺乏内化专业知识；④在罕见缺陷上的泛化能力有限。

---

## 273. CFALR: Collaborative Filtering-Augmented Large Language Model for Personalized Fashion Outfit Recommendation

**arXiv ID:** 2606.13001 | [PDF](https://arxiv.org/pdf/2606.13001v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 274. Camera and LiDAR BEV Fusion for Cooperative 3D Object Detection on TUMTraf V2X

**arXiv ID:** 2606.12981 | [PDF](https://arxiv.org/pdf/2606.12981v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 275. EPM-JEPA: Operator-Side Experience Modulation in JEPA-Family World Models

**arXiv ID:** 2606.12979 | [PDF](https://arxiv.org/pdf/2606.12979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 276. YOLO-AMC: An Improved YOLO Architecture with Attention Mechanisms for Building Crack Detection

**arXiv ID:** 2606.12958 | [PDF](https://arxiv.org/pdf/2606.12958v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 277. Towards Reliable Sequential Object Picking in Clutter: The Runner-up Solution to RGMC 2025

**arXiv ID:** 2606.12954 | [PDF](https://arxiv.org/pdf/2606.12954v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 278. PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks

**arXiv ID:** 2606.13108 | [PDF](https://arxiv.org/pdf/2606.13108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. The Invisible Ink of the Android Malware World: A Longitudinal Study on the Usage of Covert Communication Channels

**arXiv ID:** 2606.13107 | [PDF](https://arxiv.org/pdf/2606.13107v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 280. LEDGER: A Long-Context Benchmark of Corporate Annual Reports for Grounded Financial Retrieval and Extraction

**arXiv ID:** 2606.13100 | [PDF](https://arxiv.org/pdf/2606.13100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 281. Equilibrium Computation in Extensive-Form Games with Stochastic Action Sets

**arXiv ID:** 2606.13093 | [PDF](https://arxiv.org/pdf/2606.13093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 282. OpenMedQ: Broad Open Pretraining for Medical Vision-Language Models

**arXiv ID:** 2606.12953 | [PDF](https://arxiv.org/pdf/2606.12953v1)

**作者:** Ibrahim Gulluk `[一作]` (Stanford University), Olivier Gevaert `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并公开了OpenMedQ，一种使用14个完全公开医学数据集（约3.35M样本）预训练的医学视觉语言模型。

**💡 创新点**

其创新在于规模最大、开放性最强的预训练数据混合，以及通过LoRA高效地将ViT-Base与LLaMA-7B结合，实现了在VQA和分类任务上的最佳性能。

**🔧 技术方法**

采用ViT-Base视觉编码器、LLaMA-7B语言模型、LoRA低秩适配、next-token交叉熵训练和统一的下游线性分类头。

**📊 数据集**

预训练数据来自PathVQA、VQA-RAD、IU-XRAY、MIMIC-CXR、ROCO、OmniMedVQA、Slake、PMC-OA、PMC-VQA、VQA-MED、μ-Bench、MedQA、MedMCQA、PubMedQA；下游评估使用CXR8、MedFMC（胸、结肠、内镜）、Breast-Ultrasound、CHAOYANG、CBIS-DDSM、Mendeley-CXray等八个未见过的分类数据集。

**📈 对比分析**

在所有八个分类基准上，OpenMedQ以平均宏F1=0.757取得最高成绩，击败BiomedCLIP、PMC-CLIP和PubMedCLIP；在PathVQA上BLEU-1=75.9，超越规模高达562B的Med-PaLM M；在VQA-MED上BLEU-1=64.5，近似挑战赛最佳成绩。

**⚠️ 局限性**

局限包括在VQA-RAD和Slake等特定任务上仍被更大规模模型超越；仅凭BLEU-1难以衡量深层推理能力；对窄模态的模型仍可能在Breast-Ultrasound等任务中优于OpenMedQ。

---

## 283. ViPER: Vision-based Packing-Aware Encoder for Robust Malware Detection

**arXiv ID:** 2606.12949 | [PDF](https://arxiv.org/pdf/2606.12949v1)

**作者:** Fatima Qaiser `[一作]` (Pakistan Institute of Engineering & Applied Sciences), Nauman Shamim `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ViPER框架，利用视觉化字节图像与双头Transformer实现打包感知的恶意软件检测。

**💡 创新点**

将打包检测作为辅助任务融入视觉Transformer，并通过残差门控将打包信息动态调节主分类决策，配合LoRA高效适配自监督ViT，实现了结构化的打包感知。

**🔧 技术方法**

使用自监督ViT‑B/14（DINOv2）骨干、LoRA低秩适配、双头MLP与残差门控、多任务加权损失以及类不平衡处理。

**📊 数据集**

采用200,000个Windows PE二进制的灰度byteplot图像（100k恶意、100k良性），并通过Detect‑It‑Easy生成打包标签。

**📈 对比分析**

在相同训练/评估条件下与ResNet‑50、MobileNetV3等CNN基线对比，ViPER在平衡准确率0.8521、ROC‑AUC0.9260、AUPR0.9279、TPR@1%FPR0.4635等指标显著优于基线，且仅训练1.49M参数。

**⚠️ 局限性**

仅完成二分类任务；打包标签为二元未区分不同打包器；使用固定行宽的byteplot，未考虑动态布局；对不同操作系统或架构的泛化尚未验证。

---

## 284. MAMVI: 3D Test-Time Adaptation via Masked Multi-View Point Clouds

**arXiv ID:** 2606.12939 | [PDF](https://arxiv.org/pdf/2606.12939v1)

**作者:** Inseok Kong `[一作]` (University of Seoul), Jiyoung Jung `[通讯]` (University of Seoul)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 MAMVI——一种基于掩码多视角生成的单步测试时适配框架，能够在不访问源数据的情况下对 3D 点云分类模型进行实时适配。

**💡 创新点**

创新点包括：① 用混合掩码策略（固定比例 + Beta 分布）生成多样化的视角，消除逐视角优化带来的延迟；② 将所有视角的损失聚合为单一目标，完成一次反向传播；③ 通过共识置信度驱动的自适应学习率动态调节适配力度。

**🔧 技术方法**

技术细节：点云先通过 FPS+KNN 分块；使用掩码生成多视角；利用熵最小化与 KL 一致性损失；仅更新 BN/LayerNorm 的可学习参数；自适应学习率基于聚合后置信度；整体实现单步更新并保持高推理效率。

**📊 数据集**

数据集：ModelNet-40C、ShapeNet-C、ScanObjectNN-C 三个标准 3D 点云失真/噪声基准。

**📈 对比分析**

与多种基线（TENT、SHOT、MATE、SMART-PC、BFTT3D、PG-SP、SVWA 等）比较，MAMVI 在 ShapeNet-C 和 ScanObjectNN-C 上实现了 state‑of‑the‑art 平均精度（分别为 66.77% 与 49.76%），在 ModelNet-40C 上与 SVWA 的 75.04% 相近，同时推理速度提升 8.6×–8.9×。

**⚠️ 局限性**

局限性：仍需手动调节 Beta 参数、熵/一致性权重等超参；对极端遮挡/噪声的鲁棒性尚待进一步验证；在极小 batch 或资源受限的嵌入式场景下性能略逊于 SVWA；未在更大规模或实时系统上进行实测。

---

## 285. Scale Buys Interpolation, Structure Buys a Horizon: Certified Predictability for Equivariant World Models

**arXiv ID:** 2606.13092 | [PDF](https://arxiv.org/pdf/2606.13092v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 286. The Emergence of Autonomous Penetration Capabilities in Large Language Model-Powered AI Systems

**arXiv ID:** 2606.13079 | [PDF](https://arxiv.org/pdf/2606.13079v1)

**作者:** Jiaqi Luo `[一作]` (Fudan University), Min Yang `[通讯]` (Fudan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个基于大语言模型的自主渗透测试框架，测试模型在不提供先验信息的情况下，是否能自动识别并利用目标服务器中的漏洞。

**💡 创新点**

提出了真实环境下的目标服务器构建（包含可利用与安全服务混合）、通用无任务优化的agent scaffolding以及公开可复现的评估数据，首次系统地测量LLM在高复杂度真实网络环境中的自主渗透能力。

**🔧 技术方法**

使用通用工具集（Nmap、WhatWeb、Metasploit）与Model Context Protocol实现LLM与工具交互，并利用Docker化隔离环境；模型涵盖19种LLM。

**📊 数据集**

基于30个公开CVE（可复现RCE）和14类常见安全服务共300台目标服务器的自动构建数据集，已公开发布于GitHub。

**📈 对比分析**

在每个模型与目标组合下进行3次实验，记录成功率；结果显示模型成功率从10.7%到69.3%，与LiveBench综合能力正相关，复杂度提升对成功率影响有限。

**⚠️ 局限性**

评估仅覆盖到初始Shell获取，未包含横向移动、特权提升及防御机制；工具集有限导致部分失败；模型对工具使用不当和缺失可解释性导致误判。

---

## 287. SeamEdit: A Black-Box VLM-Agnostic Pipeline for Large-Image Semantic Editing

**arXiv ID:** 2606.13041 | [PDF](https://arxiv.org/pdf/2606.13041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 288. "Is This Not Enough?": Asymmetries in Institutional Accountability and Collective Sensemaking in the Case of Canada's Algorithmic Visa Triage System

**arXiv ID:** 2606.13071 | [PDF](https://arxiv.org/pdf/2606.13071v1)

**作者:** Dipto Das `[一作]` (University of Toronto), Shion Guha `[通讯]` (University of Toronto)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析加拿大移民局（IRCC）关于临时居留签证（TRV）自动分级系统的机构责任文件（AIA 与 NRC 同行评审）并对 Reddit 上的申请人讨论进行混合方法文本分析，探究机构披露与用户体验之间的差异。

**💡 创新点**

首次将 ADMAPS 框架扩展为 ADMATS（含跨境不平等维度），揭示机构层面的透明度、程序保障与算法可解释性在跨境移民场景中的“认知不对称”“司法不对称”和“时空关系不对称”。

**🔧 技术方法**

采用 ADMAPS、BERTopic、Qwen‑3 语言模型、UMAP 降维、HDBSCAN 聚类、定性开放编码与轴向编码等技术；对 AIA、NRC 文档做结构化编码，对 Reddit 语料做主题挖掘与叙事分析。

**📊 数据集**

数据来源为：IRCC 的 Algorithmic Impact Assessment（AIA）报告与 NRC 同行评审；以及 2021‑2025 年收集的 5 个加拿大移民相关 Reddit 子社区（r/immigrationCanada、r/CanadaImmigrant、r/MovingToCanada、r/IRCCDiscussion、r/CanadaVisa）共计 993 条帖子。

**📈 对比分析**

通过对照机构责任文件与用户讨论，未给出数值指标，而是以质性证据展示三大不对称；对比结果显示机构披露的透明度与实际可解释性在用户层面几乎不存在匹配，说明传统评估工具在跨境环境下的效用受限。

**⚠️ 局限性**

局限性包括：1) 仅分析 Reddit 社区，可能无法代表所有申请人；2) 部分帖子因已被删除或屏蔽无法获取，导致视角偏倚；3) AIA 采用二元问卷导致信息缺失；4) 研究聚焦 TRV，未覆盖其他签证类别，结果的外推性受限。

---

## 289. TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization

**arXiv ID:** 2606.13054 | [PDF](https://arxiv.org/pdf/2606.13054v1)

**作者:** Zhixiong Zhao `[一作]` (Houmo AI), Dawei Yang `[通讯]` (Houmo AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种后训练量化框架 TWLA，实现了 1.58-bit 权重与 4-bit 激活的压缩，同时保持 LLM 的高精度。

**💡 创新点**

创新点在于三大模块：E2M‑ATQ 通过欧氏到流形的两阶段优化提升三值量化可校准性；KOTMS 采用 Kronecker 结构正交旋转把权重形状逼近三模分布并抑制激活离群；ILA‑AMP 引入相邻层二阶交互的混合精度分配，防止弱层引发的精度崩塌。

**🔧 技术方法**

核心技术包括：欧氏-流形不对称三值量化、Cayley 参数化的 Kronecker 正交旋转、激活分层的交互式混合精度优化，以及基于校准数据的后训练量化（PTQ）流程。

**📊 数据集**

实验使用 WikiText2、C4、以及多种零样本评测集（ARC、HellaSwag、MMLU、GSM8K、HumanEval、LAMBADA 等），同时对 LLaMA 与 Qwen3 系列模型进行验证。

**📈 对比分析**

与 SliM‑LLM、PT^2‑LLM、ResQ、QuaRot、GPTQ 等 PTQ 基线比较，TWLA 在 LLaMA‑2‑70B 上从 56% 提升至 71%（接近 FP16 的 92%），并在推理速度上相较 FP16 提升 3.6×、相较 QuaRot 提升 1.3×，显著降低 80% 以上的模型存储。

**⚠️ 局限性**

局限性包括：仍依赖校准数据且对极低位宽（<4bit）激活的鲁棒性有限；Kronecker 旋转虽高效但仍带来额外算子开销；仅适用于后训练量化，未解决训练阶段的可扩展性与自适应学习率等问题。

---

## 290. EA-WM: Event-Aware World Models with Task-Specification Grounding for Long-Horizon Manipulation

**arXiv ID:** 2606.13053 | [PDF](https://arxiv.org/pdf/2606.13053v1)

**作者:** Kailin Wang `[一作]` (Country Garden Services Group), Zhiyou Heng `[通讯]` (Country Garden Services Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种事件感知世界模型 EA-WM，将预训练视觉特征的未来滚动与任务相关事件预测和验证结合，用于机器人长周期规划。

**💡 创新点**

创新点在于将事件预测和验证层与视觉特征世界模型分离，利用任务规范和模拟器标签生成事件监督，并通过验证器引导 CEM 规划和 PPO 提议，从视觉特征到任务进展提供可解释的决策依据。

**🔧 技术方法**

采用冻结的 DINO 编码器、动作条件的特征动态模型、事件预测器、验证器评分、CEM 样本优化以及 PPO 残差提议等技术；事件标签通过 BDDL 规则和模拟器状态自动生成。

**📊 数据集**

使用的基准数据集包括 PointMaze（导航）、Deformable（柔性物体）、Wall-Single（壁面约束）和 LIBERO-goal（语言描述的操控）等。

**📈 对比分析**

与基线 DINO-WM 及其 CEM 版本对比，EA-WM 在 PointMaze 随机目标成功率从 0.90 提升至 0.94，Deformable 上检索初始化的 EA-CEM 达到 94% 成功，Wall-Single 通过归档验证提升至 95%，LIBERO-goal 验证器 AUC 达到 0.994，酒架 PPO 提议在 H=20 任务中实现 97/100 的成功率。

**⚠️ 局限性**

主要限制在于事件监督依赖模拟器标签，缺乏真实世界感知；在线评估仅为短窗口 H=20，未覆盖完整长周期执行；验证器部分规则化且仅针对部分任务，需进一步推广到更广泛的自主任务。

---

## 291. GeoCFNet: Geometry-Aware Confidence Field Network for Robot-Assisted Endoscopic Submucosal Dissection

**arXiv ID:** 2606.13032 | [PDF](https://arxiv.org/pdf/2606.13032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 292. DIG: Oracle-Guided Directed Input Generation for One-Day Vulnerabilities

**arXiv ID:** 2606.13037 | [PDF](https://arxiv.org/pdf/2606.13037v1)

**作者:** Andrew Bao `[一作]` (University of Minnesota, Twin Cities), Pen-Chung Yew `[通讯]` (University of Minnesota, Twin Cities)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 DIG，一种基于补丁驱动的 Oracle‑guided PoC 生成系统，利用 LLM 合成 oracle 并在生成器演化与定向变异两个层级中自动化构造一日漏洞 PoC。

**💡 创新点**

核心创新是将补丁中的必然触发前置条件直接提取为 oracle，利用该 oracle 驱动的两级约束推理与解算——高层生成器演化缓解目标漂移，低层 branch‑distance 定向变异提升求解效率。

**🔧 技术方法**

技术手段包括 LLM（Claude Sonnet 4.5 及其他模型）用于补丁分析与 oracle 合成；静态、动态与源代码检索工具进行约束推理；生成器演化算法和 oracle‑guided 定向灰盒变异；branch‑distance 反馈实现高效输入改进。

**📊 数据集**

实验基于 Magma 基准，涵盖 138 个真实 CVE，涉及 9 个开源项目的多种文件格式（PNG、SND、TIF、XML、Lua、PDF、SQL、PHP、SSL）。

**📈 对比分析**

与两款 agentic 系统、10 款现有 fuzzer（AFL、AFL++、AFLGo、SelectFuzz、Titan、CmpLog、FOX、G^2Fuzz、LLAMAFUZZ、SeedAIchemy）对比，DIG 触发 80/138 CVE（58% 覆盖率），比最佳基线提升 40%，平均速度提升 92.9%，部分案例实现 1000× 以上加速。

**⚠️ 局限性**

局限性包括：依赖可获得的补丁与漏洞描述，难以直接用于零日漏洞；LLM 推理与多轮交互成本高；对隐藏状态依赖的约束仍需低层变异补足；模型训练数据可能导致结果受污染。

---

## 293. Comparing Commercial Depth Sensor Accuracy for Medical Applications

**arXiv ID:** 2606.13028 | [PDF](https://arxiv.org/pdf/2606.13028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 294. FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation

**arXiv ID:** 2606.13102 | [PDF](https://arxiv.org/pdf/2606.13102v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 295. Functional Cache Grafting: Robust and Rapid Code-Policy Synthesis for Embodied Agents

**arXiv ID:** 2606.13097 | [PDF](https://arxiv.org/pdf/2606.13097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 296. Democracy in the Era of Artificial Intelligence

**arXiv ID:** 2606.13026 | [PDF](https://arxiv.org/pdf/2606.13026v1)

**作者:** Evangelos Pournaras `[一作]` (University of Leeds), Dirk Helbing `[通讯]` (ETH Zurich)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本手册共34章，系统梳理了人工智能（AI）在民主治理中的多维度应用，涵盖从集体智能与投票公正、AI驱动的讨论与决策、去中心化自治组织到AI监管与哲学反思；通过案例研究、对比分析与实验评估，探讨了AI在提升民主参与、透明度、问责与韧性方面的潜力与风险。

**💡 创新点**

创新点在于将AI技术与民主设计原则深度融合，提出价值驱动与参与式的AI治理框架，系统性评估AI在投票公正、议论质量、去中心化自治及公共监管中的作用，强调AI并非替代人类，而是增强民主主体的判断与协作；同时将大语言模型、可解释AI与知识图谱等前沿技术与社会科学方法结合，形成跨学科的民主创新范式。

**🔧 技术方法**

主要技术包括：计算社会选择算法、可解释人工智能与符号推理的混合模型、生成式大语言模型（LLM）用于议论辅助、区块链与去中心化自治组织（DAO）框架、数字化公民参与平台、数据可视化与交互式决策工具，以及隐私保护与公平评估技术。

**📊 数据集**

使用的数据集涵盖：选举与投票统计数据（如瑞士阿劳市、波兰城市等）、智能城市感知与共享资源数据、社交媒体舆情与传播数据、公共参与与在线投票数据、AI模型训练与评测数据集（如GLUE、CommonsenseQA等）、以及政府与企业的监管与隐私监测数据。

**📈 对比分析**

比较方法主要通过案例对比、量化指标评估与实验对照来检验AI介入的效果，例如：比较不同投票规则下的代表性与公平度、LLM与人类议论在内部一致性与理性评估中的表现、AI辅助与传统公民参与平台在参与度与决策质量上的差异。性能表现因情境而异，但整体显示：AI能提升投票公平性、议论透明度与参与效率；然而在部分案例中也出现偏差、误判与可解释性不足。

**⚠️ 局限性**

局限性包括：算法偏见与可解释性不足导致决策公正性受损；数据隐私与监控风险，可能削弱公民自由与信任；跨学科方法整合难度高，缺乏统一评估框架；法规与监管滞后，导致AI治理在实践中的执行不一致；部分研究以案例或实验为主，缺乏大规模实证验证。

---

## 297. Unified MRI Brain Image Translation via Hierarchical Tumor Structure Comparison

**arXiv ID:** 2606.13096 | [PDF](https://arxiv.org/pdf/2606.13096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 298. Multi-Objective Coevolution of Prompts and Templates for Circuit Approximation

**arXiv ID:** 2606.13089 | [PDF](https://arxiv.org/pdf/2606.13089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 299. sebis at CRF Filling 2026: A Two-Stage Local LLM Pipeline for Medical CRF Filling

**arXiv ID:** 2606.13082 | [PDF](https://arxiv.org/pdf/2606.13082v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 300. RoboProcessBench: Benchmarking Process-Aware Understanding in Vision-Language Robotic Manipulation

**arXiv ID:** 2606.13040 | [PDF](https://arxiv.org/pdf/2606.13040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 301. Approximate Maximin Share with Subjective Divisibility: Beating the 1/2 Barrier

**arXiv ID:** 2606.13057 | [PDF](https://arxiv.org/pdf/2606.13057v1)

**作者:** Xiaohui Bei `[一作]` (Nanyang Technological University), Fangxiao Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在主体可分性（Subjective Divisibility）模型下的最大最小份额（MMS）公平分配问题，证明了在统一价值（unary）情形下最优近似比为2/3，并提出了新的算法在一般异质价值与主体可分性情形下实现5/9-MMS近似；此外还给出了针对3人和4人情形的多项式时间算法实现2/3-MMS分配。

**💡 创新点**

创新点主要包括：①首次给出统一价值情况下MMS近似比的上界与下界相匹配；②在主体可分性模型中突破1/2阈值，提出了三阶段的分配框架（预处理大件与可共享件、主体分类与Bag‑Filling、关键代理处理），通过匹配、IDO归约与层次化分配实现5/9近似；③为3、4人提供了完全可实现的多项式时间算法，并证明2/3近似是最优。

**🔧 技术方法**

使用的技术：
- 匹配与最大2‑to‑1匹配（处理可共享件）；
- 基于可分性视角的物品分类（大件、介值件、可共享件）；
- 预处理与逐步裁剪（保留MMS不下降）；
- IDO归约与排序归一化；
- Bag‑Filling策略与优先级判定；
- 关键代理（Z）与临时/好包（nice bundle）技术；
- 证明中利用组合论与整数规划上界。

**📊 数据集**

本研究为理论论文，未使用实际数据集；所有结果均通过严谨的数学证明与构造算法获得。

**📈 对比分析**

与之前的1/2-MMS（主体可分性）以及7/9-MMS（一般异质价值）进行比较；本文将1/2提升至5/9，在统一价值下实现2/3，3人4人情形实现2/3；实验与复杂度分析表明算法在多项式时间内可执行，且在最坏情况下保持上述近似比。

**⚠️ 局限性**

局限性：
- 对于5人及以上情形的最优近似比仍未确定，虽然推测可能为2/3；
- 算法主要针对加法效用，无法直接推广至子模或更一般的非加法效用；
- 主体可分性假设下的匹配与分配过程较为复杂，实用性和可实现性需进一步验证；
- 对于大于2n件的情况，当前分析未给出更优近似，留待后续研究。

---

## 302. Fault Lines: Navigating Ethics and Responsible AI Where National Policy Meets Local Practice in Public Sector Transformation

**arXiv ID:** 2606.13039 | [PDF](https://arxiv.org/pdf/2606.13039v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 303. Nous: An Attempt to Extract and Inject the Cognition Behind Prediction-Market Behavior

**arXiv ID:** 2606.13038 | [PDF](https://arxiv.org/pdf/2606.13038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 304. TetherCache: Stabilizing Autoregressive Long-Form Video Generation with Gated Recall and Trusted Alignment

**arXiv ID:** 2606.13035 | [PDF](https://arxiv.org/pdf/2606.13035v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 305. SAM-Deep-EIoU: Selective Mask Propagation for Multi-Object Tracking

**arXiv ID:** 2606.13033 | [PDF](https://arxiv.org/pdf/2606.13033v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning

**arXiv ID:** 2606.13106 | [PDF](https://arxiv.org/pdf/2606.13106v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 307. Authority, Truth, and Citation Bias: A Large-Scale Multi-Domain Benchmark for Studying Epistemic Susceptibility in Large Language Models

**arXiv ID:** 2606.13104 | [PDF](https://arxiv.org/pdf/2606.13104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 308. LaME: Learning to Think in Latent Space for Multimodal Embedding via Information Bottleneck

**arXiv ID:** 2606.13061 | [PDF](https://arxiv.org/pdf/2606.13061v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 309. Disparate Impact in Synthetic Data Generation

**arXiv ID:** 2606.13105 | [PDF](https://arxiv.org/pdf/2606.13105v1)

**作者:** Paul Andrey `[一作]` (University of Lille, Inria, CNRS, Centrale Lille), Marc Tommasi `[通讯]` (University of Lille, Inria, CNRS, Centrale Lille)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了合成数据生成（Synthetic Data Generation, SDG）中的不公平性，提出以差异影响（disparate impact）为核心的公平性定义，并系统分析了导致不公平的误差来源（近似误差、估计误差与差分隐私导致的噪声）以及这些误差在不同敏感组间的分布；同时提出了一种按敏感组学习的Meta‑Algorithm，旨在降低 SDG 的不公平性。

**💡 创新点**

①将差异影响这一成熟的公平性概念应用于 SDG；②阐明了 SDG 在近似、估计与 DP 机制三类误差下产生不公平的机制，并展示它们可累积放大不公平；③提出按组建模的策略作为降低不公平性的可行基线。

**🔧 技术方法**

主要使用概率图模型（PGM）方法（MST、PrivBayes、AIM、GreedyBayes）以及其 DP 版本；利用统计误差与图结构选择分析；设计了按组学习的 Meta‑Algorithm；在实验中使用 TVD、AUROC 等指标衡量分布逼真度与下游分类器性能。

**📊 数据集**

①人工控制分布（6 维二值特征，含敏感属性）用于验证误差机制；②美国人口普查调查（ACS）数据，筛选 5 个州和 2 个种族，定义性别与种族四个敏感组，用于收入预测任务。

**📈 对比分析**

通过对比整体级和按组级建模、不同 DP 预算（ε=1,10,100,∞）以及不同 PGM 参数（如树度、最大 cliques 等），使用 TVD 衡量分布相似度、AUROC 差异衡量下游分类器效能。结果显示：整体建模在大隐私（ε=1）下往往更公平；按组建模在大多数情况下降低了不公平性，但在高隐私 regime 下仍显著偏向多数组；多数方法在少数族群上的损失更大。

**⚠️ 局限性**

①实验仅覆盖基于 PGM 的 SDG 方法，未探讨 GAN、VAE 等生成模型；②只考察了分类任务的公平性指标，未评估多任务或回归场景；③按组建模在高隐私下效果不佳，且可能引入新的估计误差；④对图结构选择的改进仍是开放问题；⑤实验规模受限，缺乏对大规模分布的验证。

---

## 310. Y-BotFrame: An Extensible Embodied Agent Framework for Quadruped Robot Assistants

**arXiv ID:** 2606.13049 | [PDF](https://arxiv.org/pdf/2606.13049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 311. Leveraging Matchings in Constrained Fair Division with a Conflict Graph

**arXiv ID:** 2606.13083 | [PDF](https://arxiv.org/pdf/2606.13083v1)

**作者:** Evangelos Markakis `[一作]` (Athens University of Economics and Business), Michalis Samaris `[通讯]` (Athens University of Economics and Business)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在冲突图约束下，公平分配无可分割物品时是否存在完整的 EF1（up to one good）分配，并给出多种情形下的存在性与算法实现。

**💡 创新点**

创新点在于将冲突图的最大度 Δ 作为参数，得到紧确的存在界限（如 m ≤ n⌊n/Δ⌋+n-Δ），并在此范围内设计多种多轮匹配与逼近算法；同时扩展到更一般的加性价值，给出在 m ≤ 2n 且 Δ ≤ 2n/3 时的完整 EF1 方案，并证明在 Δ > 2n/3 时的必要条件。

**🔧 技术方法**

核心技术：匹配理论（Hall 定理、Dulmage–Mendelsohn 分解、可行 2‑matching）、基于可行图的多轮分配（Tiered Matching、Round‑Robin、Envy‑Swap）、以及对冲突图结构的细致分析（最大缺陷 Hall violator、辅助图）。

**📊 数据集**

该工作为理论性研究，未使用具体实验数据集，所有结果均为数学证明与多项式时间算法。

**📈 对比分析**

方法通过严格的匹配与图论分析证明了算法的可行性与多项式复杂度；在满足相应参数约束（Δ、m）的前提下，算法能够保证找到完整 EF1 分配；若参数不满足约束，则给出了紧确不可行的反例。

**⚠️ 局限性**

主要限制在于：对于一般加性价值，仅能在 m ≤ 2n 且 Δ ≤ 2n/3 时保证完整 EF1；当 Δ > 2n/3 或 m 超过 2n 时，结果不完整且需要额外的子图结构假设；此外，算法对冲突图的度分布没有进一步利用，未能针对更稀疏或特殊结构图给出更强的保证。

---

## 312. No Hidden Prompts Needed! You Can Game AI Peer Review with Presentation-Only Revisions

**arXiv ID:** 2606.13044 | [PDF](https://arxiv.org/pdf/2606.13044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 313. Modular Multi-Domain Digital Twin Architecture: Sustainable Intent-Driven 6G Management

**arXiv ID:** 2606.13069 | [PDF](https://arxiv.org/pdf/2606.13069v1)

**作者:** Berk Buzcu `[一作]` (University of Applied Sciences and Arts Western Switzerland), Paweł Kryszkiewicz `[通讯]` (Poznań University of Technology)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于多域数字孪生的6G架构，支持跨域“what-if”分析与可持续网络管理，并通过将基站能源仿真与网络模拟耦合，展示了碳感知运营的可行性。

**💡 创新点**

创新点在于将数字孪生拆分为可按查询动态组合的模块化服务，嵌入多域编排框架，实现跨域协同决策；同时首次将外部可再生能源信息与网络孪生耦合，构建碳感知基站运营决策流程。

**🔧 技术方法**

采用服务化5G/6G跨域SBMA bus、DT Orchestrator/Manager、O‑RAN与能源仿真器的动态组合、强化学习/强化模拟沙盒、SDN/TerraFlow暴露接口及数据连续体同步等技术。

**📊 数据集**

使用波兹南真实基站坐标、UE流量统计数据，MERRA‑2再分析的太阳辐射与ENTSO‑E电网碳强度数据，并通过NS‑3等仿真工具生成网络状态。

**📈 对比分析**

与传统单域覆盖/能耗优化方法对比，实验在波兹南105基站场景下固定32个太阳能板时，网格能耗降低28.5%，并通过北非场景展示不同地区的收益曲线，证明跨域数字孪生显著提升能源效率。

**⚠️ 局限性**

局限性包括：缺乏实时在线故障响应与动态仿真；跨域数据共享受隐私与安全约束；仿真模型对真实网络的近似可能忽略细节；能源预测误差及地区差异可能影响结果的通用性。

---

## 314. A green solvent screening tool for emerging materials via uncertainty aware, transformer enhanced transfer learning

**arXiv ID:** 2606.13060 | [PDF](https://arxiv.org/pdf/2606.13060v1)

**作者:** Ioannis Kouroudis `[一作]`, Aldo Di Carlo `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过迁移学习与高斯过程回归预测溶剂的九个溶解参数，并搭建可自定义的绿色溶剂筛选工具。

**💡 创新点**

创新点在于将预训练的基础模型DimeNet与分子描述子融合，并在最后层引入高斯过程实现数据稀缺场景下的置信度量，显著提升对稀缺参数的预测精度。

**🔧 技术方法**

技术包括DimeNet基础模型、Transformer注意力层、Gaussian Process回归、基于SMILES的筛选、以及多种距离度量（欧氏、马氏、KL）。

**📊 数据集**

使用来自CompSol、Hansen手册、Stenutz网站以及PubChem的5000多种VOC数据，以及QM9的预训练权重。

**📈 对比分析**

通过5/10折交叉验证与基线GP+描述子模型对比，平均相对RMSE提升约30%-85%，在大型数据集上与传统方法持平，且在小数据集上保持较高准确性。

**⚠️ 局限性**

局限性包括对极少样本的参数仍可能欠拟合、对非溶剂性质（如沸点、反应性）的忽视，以及对DimeNet的依赖，未来可替换为更先进的MACE/NequIP模型。

---

## 315. MPC for underactuated spacecraft control with a Lyapunov supervised physics-informed neural network correction layer

**arXiv ID:** 2606.13113 | [PDF](https://arxiv.org/pdf/2606.13113v1)

**作者:** Amirhossein Ayanmanesh Motlaghmofrad `[一作]` (Politecnico di Torino), Marcello Chiaberge `[通讯]` (Politecnico di Torino)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并验证了一种分层控制框架，用于欠驱动航天器的姿态调节，框架包括基于非线性模型预测控制（NMPC）的基准策略、物理信息神经网络（PINN）残差扭矩补偿器以及确定性Lyapunov安全监督器。

**💡 创新点**

创新点在于：①将物理约束融入神经网络学习残差扭矩，显著提升模型不匹配补偿效果；②提出实时Lyapunov安全过滤机制，保证学习补偿在保证系统闭环稳定性的前提下被接受或被抑制；③在欠驱动且采用零总角动量假设的情形下，将上述三者协同工作，兼顾性能与安全。

**🔧 技术方法**

使用的技术包括：非线性模型预测控制（NMPC）、物理信息神经网络（PINN）与NARX结构、Runge–Kutta 4阶离散化、Monte Carlo仿真、Wilcoxon符号秩检验、Lyapunov函数安全过滤、数据归一化与输出限幅。

**📊 数据集**

数据集来源于高保真仿真环境，包含100个不同初始姿态（通过Fibonacci网格在球面上均匀采样得到）下的轨迹，仿真中加入惯量不确定性、重力梯度、空气阻力等扰动，训练PINN时使用的模拟记录（状态、控制、扰动）构成训练集。

**📈 对比分析**

通过在相同的仿真条件下比较三种控制器（基准NMPC、NMPC+PINN、NMPC+PINN+监督器），采用稳态RMSE和最终误差两指标，并用Wilcoxon检验评估差异显著性。结果显示，未监督的PINN补偿在稳态误差上平均降低3.8%，最坏情况下降24%；最终误差平均降低12.7%，最大值下降52.9%，方差降低29.4%；加入监督器后性能略有下降，但仍保持显著改进并保证稳定性。

**⚠️ 局限性**

局限性包括：①监督器对PINN补偿做保守约束，导致性能提升被削弱；②仅验证在仿真环境下，缺乏实际硬件测试；③框架依赖零总角动量假设，可能不适用于完全欠驱动或有外部推力的系统；④对更大扰动或非高斯噪声的鲁棒性尚未评估；⑤训练数据与真实飞行数据可能存在差异，影响PINN泛化。

---

## 316. Revolutionizing Wireless Communications with Space Data Centers: Applications and Open Challenges

**arXiv ID:** 2606.13086 | [PDF](https://arxiv.org/pdf/2606.13086v1)

**作者:** Minghao Sun `[一作]`, Xiaoli Chu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出空间数据中心（SDC）的分层网络架构，并讨论了其在AI模型部署、地球观测、地面-空间协同计算以及大规模空间系统控制等四类代表性应用场景中的潜在作用与挑战；通过多星座仿真验证了SDC作为机上控制中心能显著降低控制层时延。

**💡 创新点**

创新点在于把卫星从单纯的中继/轻量处理节点转变为具备通信、计算、存储与控制的综合服务平台，形成“任务导向、服务中心化”的新通信范式；同时提出了多层SDC框架（接入层、传输/中继层、机上计算层、服务与控制层）以及对应的部署策略和挑战。

**🔧 技术方法**

主要技术包括光学激光互连以实现高容量ISL、缓冲辅助转发机制、软件定义的跨层资源调度、模型压缩与增量更新技术、以及多租户安全与隔离机制。

**📊 数据集**

文章未使用公开数据集，而是通过自定义仿真环境（多星座LEO网络、每星8台、每台10 kB状态信息）来评估控制层时延。

**📈 对比分析**

比较方法：对比以SDC为机上控制中心与传统以地面站为控制中心的两种架构。仿真结果显示，机上控制中心的控制层平均时延明显降低，表明SDC能提升低时延协同与调度效率。

**⚠️ 局限性**

局限性包括：1）仍未解决ISL与地面链路容量与数据中心级服务的巨大差距；2）跨层资源调度与任务优先级策略仍处于理论阶段，缺乏可验证的实现；3）安全与多租户隔离方案尚未得到实地测试；4）仿真仅考虑理想化条件，真实空间环境中的链路波动、硬件失效及电力/热管理问题需进一步研究。

---

## 317. Augmentation techniques for video surveillance in the visible and thermal spectral range

**arXiv ID:** 2606.13042 | [PDF](https://arxiv.org/pdf/2606.13042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 318. Effects of Social Interactions in Self-Organising Railway Traffic Management

**arXiv ID:** 2606.13068 | [PDF](https://arxiv.org/pdf/2606.13068v1)

**作者:** Fabio Oddi `[一作]` (Institute for Cognitive Sciences and Technologies, CNR), Vito Trianni `[通讯]` (Institute for Cognitive Sciences and Technologies, CNR)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文在自组织列车交通管理系统中研究预测邻域视野 T_h 的影响，评估其对网络拓扑、局部 MILP 求解、协商一致性和全局调度的影响。

**💡 创新点**

创新点在于系统性分析视野参数的结构性权衡，并证明短视野即可满足性能需求，挑战了传统认为更长视野有利于全局优化的直觉。

**🔧 技术方法**

使用了自组织列车管理架构、局部混合整数线性规划、投票式协商模型、图论拓扑指标以及 OpenTrack 微观仿真。

**📊 数据集**

数据集为意大利 Segrate–Ospitaletto 60 km 混合线路的压缩时刻表（约 150 辆列车/日），并基于历史延迟分布生成 10 个扰动场景。

**📈 对比分析**

与基准中心化 RECIFE‑MILP 方案对比，短视野下 SO‑TMS 在平均 5 % 的延迟提升、接近最优的局部解以及较低的计算时间上表现优异；更长视野则导致计算时间和不一致性增加，整体性能无提升。

**⚠️ 局限性**

局限性包括仅在单条 60 km 线路的仿真环境下验证，未考虑多运营商实际通信与隐私约束；实验规模与真实复杂网络相对有限，结果的普适性需要进一步验证。

---

## 319. Limits of spectral learning under noise

**arXiv ID:** 2606.13067 | [PDF](https://arxiv.org/pdf/2606.13067v1)

**作者:** Sabin Roman `[一作]` (Jožef Stefan Institute), Roger Guimera `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

研究在带有标签噪声的监督回归中，稀疏谱展开的系数随噪声如何漂移，并给出了闭式衰减曲线。

**💡 创新点**

首次通过白化经验特征几何，推导出噪声引起的系数漂移与有效活跃谱模式数量相关的统一噪声阈值，并验证该模型的普适性。

**🔧 技术方法**

采用稀疏谱回归（Lasso）、特征白化、理论分析与数值模拟等技术，考察傅里叶、Legendre、Bessel、Haar 等基。

**📊 数据集**

使用合成目标函数（如一维 x²、二维 sin(x₁²e^{x₂}) 等）在不同基上生成训练数据，实验覆盖一维和二维。

**📈 对比分析**

与多基、多维实验结果对比，发现归一化重叠 q(σ/σ*) 和系数距离随 σ/σ* 线性增长，理论曲线与实验曲线高度吻合；RMSE 也随噪声增大按预测增长。

**⚠️ 局限性**

主要局限在于仅考虑加性标签噪声、稀疏谱基，未讨论非线性或高阶噪声及不同正则化策略的影响，并且实验基于合成数据，缺乏真实世界验证。

---

## 320. CausalMoE: A Billion-Scale Multimodal Foundation Model for Granger Causal Discovery with Pattern-Routed Heterogeneous Experts

**arXiv ID:** 2606.13024 | [PDF](https://arxiv.org/pdf/2606.13024v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 321. Active Sensing-assisted UAV Communications with Jittering: Framework and Performance Analysis

**arXiv ID:** 2606.13036 | [PDF](https://arxiv.org/pdf/2606.13036v1)

**作者:** Guangji Chen `[一作]` (Nanjing University of Science and Technology), Caihong Kai `[通讯]` (Hefei University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对地面到UAV链路在机体抖动条件下的两阶段主动感知辅助通信框架，并设计了通信导向方案与感知导向方案；

**💡 创新点**

创新点在于将主动感知与通信耦合，利用感知阶段的信号估计角度信息以显著提升第二阶段的波束对准与传输速率，并给出了闭式速率表达式、时间分配最优解、以及通信导向方案优于感知导向方案的阈值条件；

**🔧 技术方法**

主要技术包括：最大似然估计（MLE）与克拉美-罗尔界（CRB）分析、角度估计误差对波束形成增益的影响建模、两阶段时间分配的半闭式求解、以及大天线/高功率下的渐近性分析；

**📊 数据集**

文中未使用公开数据集，所有结果均基于系统参数设置与仿真（如M_t=10、M_r=32、M_s=4、T=500等）得出；

**📈 对比分析**

通过理论推导得到可与“上界”“固定时间分配”“旧波束”等基准进行比较；实验显示所提方案在不同T、M_s、P_t下均能显著提升平均吞吐量，且在高功率或大时间槽时可逼近无抖动上界；

**⚠️ 局限性**

主要局限包括：需要在UAV上额外部署主动感知天线（成本与功耗），假设抖动为独立高斯分布，且两阶段切换时延忽略；对复杂多路径或快速运动场景的适应性尚未验证。

---

## 322. MÖVE: A Holistic LLM Benchmark for the German Public Sector

**arXiv ID:** 2606.13111 | [PDF](https://arxiv.org/pdf/2606.13111v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 323. Emotional regulation improves deep learning-based image classification

**arXiv ID:** 2606.13081 | [PDF](https://arxiv.org/pdf/2606.13081v1)

**作者:** Riccardo Emanuele Landi `[一作]` (Mare Group), Marta Chinnici `[通讯]` (ENEA Casaccia Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了情感调节（Emotional Regulation）框架，利用人工主观情绪经验进行预训练，并在图像分类任务中加入可学习的情绪调节层，以提升深度学习模型的泛化性能。

**💡 创新点**

创新点在于将主观情绪历史与情绪调节机制结合，构建可学习的情绪调节层，使模型能够根据情绪输出动态调整非情绪与情绪影响的权重，超越传统只关注神经生理特征的情绪增强方法。

**🔧 技术方法**

技术上采用三阶段训练：①情绪预训练（情感编码器 + FCNN），②预训练（非情绪和情绪影响编码器），③训练（情绪调节层 R 与分类损失），并在 ResNet‑50 与 ViT‑B/16 两大 backbone 上实现。

**📊 数据集**

使用 EMOd、Diffused‑EMOd、Abstract、EmoSet 四个情绪数据集进行情绪预训练；CIFAR‑10 与 CIFAR‑100 作为下游图像分类基准。

**📈 对比分析**

与原始 backbone 对比，Learn‑E‑Reg、Rand‑Learn‑E‑Reg、Full‑Learn‑E‑Reg 等配置在 CIFAR‑10/100 上分别提升了约 1.3%–3.2% 的准确率；在 ViT‑B/16 上最高相对提升约 1.02%（CIFAR‑100），显著优于已有情绪增强模型，成为该领域的新 state‑of‑the‑art。

**⚠️ 局限性**

局限性包括：情绪与图像的映射预先设定且为离散类别，缺乏动态情绪建模；对非视觉模态或更大规模数据集的泛化尚未验证；且在多模态或交互场景中的情绪调节效果需进一步探究。

---

## 324. A Context-Aware Dataset for Stance Detection in Bioethical Controversies on Reddit

**arXiv ID:** 2606.13187 | [PDF](https://arxiv.org/pdf/2606.13187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 325. $α$-fair heterogeneous agent reinforcement learning

**arXiv ID:** 2606.13076 | [PDF](https://arxiv.org/pdf/2606.13076v1)

**作者:** Yao-hua Franck Xu `[一作]` (Orange Innov), Arnaud Braud `[通讯]` (Orange Innov)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种将α-公平性目标与异构代理信赖域学习（HATRL）相结合的理论框架，并实现了两种可落地算法α-fair HATRPO和α-fair HAPPO，用于在多智能体强化学习中实现公平协作。

**💡 创新点**

创新点包括：① 将α-公平目标与HATRL的单调改进与收敛到Nash均衡的理论保证相结合；② 设计了基于动态权重的公平优势函数，实现对各智能体收益分配的细粒度调节；③ 提供了理论可证明且可实际部署的两种算法。

**🔧 技术方法**

技术手段主要包括：异构代理信赖域学习（HATRL）、TRPO、PPO、α-公平性函数、优势函数分解、KL约束/剪切、梯度裁剪、神经网络策略与价值网络。

**📊 数据集**

实验使用了SocialJax平台的两类连续社交困境环境：CleanUp和CommonHarvest，用于评估公平性与效率。

**📈 对比分析**

与基线HATRPO/HAPPO（全局奖励版）以及FMAPPO（基于proportional fairness）进行对比，指标为总苹果消耗（TAC）和Gini指数。实验结果显示α-fair HATRPO和α-fair HAPPO在公平性（Gini更低）上优于基线，且在效率上相近或略优；FMAPPO在纯效率上略胜一筹。

**⚠️ 局限性**

局限性包括：需要正值、上界奖励；要求完全可观测状态；对高α值敏感导致训练不稳定；在探索策略与不完全可观测环境下的适用性仍有限。

---

## 326. AAbAAC: An Annotated Corpus for Autoimmunity Information Extraction

**arXiv ID:** 2606.13051 | [PDF](https://arxiv.org/pdf/2606.13051v1)

**作者:** Fabien Maury `[一作]` (Inserm, Université Paris Cité), Adrien Coulet `[通讯]` (Inria, Inserm, Université Paris Cité)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了AAbAAC自体免疫相关实体和关系手工标注语料库，并在此语料上评估了NER模型的表现。

**💡 创新点**

创新点在于首次提供了面向自身免疫领域的实体-关系标注数据，并证明即使是小规模标注也能显著提升模型性能。

**🔧 技术方法**

主要使用了QuickUMLS、GLiNER（Zero-shot、Few-shot、Fine-tune）和Google MedGemma LLM等技术进行实体识别。

**📊 数据集**

数据集为115篇PubMed摘要组成的AAbAAC语料库，包含五类实体（自抗体、自抗体定位、自抗体靶点、疾病、症状/临床表现）及十类关系。

**📈 对比分析**

通过80/20随机划分多次实验，Fine-tuned的MedGemma在整体F1上最高（0.66），GLiNER large Fine-tune也表现优异（0.59），相比无fine-tune或零/两-shot方案提升显著。

**⚠️ 局限性**

局限性包括语料量小、文本来源单一（论文摘要）、罕见疾病和抗体的覆盖不足、部分关系类型极少出现，以及标注中使用discontiguousEntity导致模型难以捕捉分散实体。

---

## 327. A Multi-Modal Framework with Cross-Subject Pseudo-Labeling and Semantic Alignment for Micro-Gesture Recognition

**arXiv ID:** 2606.13030 | [PDF](https://arxiv.org/pdf/2606.13030v1)

**作者:** Haoran Zhang `[一作]` (Hefei University of Technology), Yanbin Hao `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个多模态微手势识别框架，并通过交叉模态伪标签化实现跨主体无监督域适应。

**💡 创新点**

创新点包括：1）利用68关键点（含面部）提升骨架特征；2）结合3D热图与高分辨率RGB；3）引入平方根平滑加权和正交语义嵌入损失处理极端长尾；4）采用温度缩放软投票减少投票失效；5）迭代式交叉模态伪标签化大幅提升跨主体泛化。

**🔧 技术方法**

技术手段包括：Video Swin Transformer、R(2+1)D、PoseC3D（3D热图+ResNet50）、Decoupled Spatial‑Temporal CNN、正交语义嵌入损失、平方根平滑权重、温度缩放软投票、伪标签化与无监督域适应。

**📊 数据集**

使用的公开数据集是 iMiGUE，包含 359 条 72 受试者的 32 类微手势视频，采用严格的跨主体评估协议。

**📈 对比分析**

与前四届 MiGA‑Track‑1 的顶尖方法对比，在 iMiGUE 测试集上取得 68.13% 的 Top‑1 Accuracy，排名第 4，表现优于单模态基线并接近多模态最高成绩。

**⚠️ 局限性**

局限性：1）对伪标签阈值和温度参数的选择敏感；2）在极端长尾场景下仍可能出现模式崩溃；3）模型规模较大，推理成本高；4）对骨架估计误差的鲁棒性尚需进一步提升。

---

## 328. Redesigning Regularization for Effective Policy Smoothing

**arXiv ID:** 2606.13169 | [PDF](https://arxiv.org/pdf/2606.13169v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 329. Loss-Shift Transfer via Bayes Quotients

**arXiv ID:** 2606.13178 | [PDF](https://arxiv.org/pdf/2606.13178v1)

**作者:** Vasileios Sevetlidis `[一作]` `[通讯]` (Athena Research Center), Vasileios Sevetlidis (Athena Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在数据分布保持不变、但损失函数变化时，冻结表示层导致的迁移失败机制，提出了“loss shift”概念，并通过 Bayes quotient 框架对损失所需信息进行量化。

**💡 创新点**

创新点在于：①用损失诱导的 Bayes quotient 定义损失间的精炼关系；②证明在严格精炼时，源任务最小表示对目标损失不可行；③对有限输出 log loss 给出精确的冻结迁移差距公式，等价于条件互信息或期望 KL。

**🔧 技术方法**

技术方法包括决策理论、Bayes quotient 与 Blackwell 比较、信息瓶颈理论、互信息与 KL 分解、以及在控制模型、学习瓶颈、dSprites 及 CIFAR‑10H 上的实验验证。

**📊 数据集**

使用的数据集有：①控制离散模型（S,T、η 结构）；②学习瓶颈实验的合成二分类数据；③dSprites 合成图像（尺度 T、水平位置 S）；④真实图像的 CIFAR‑10H 人类软标签数据。

**📈 对比分析**

比较方式：将冻结表示与全输入、全概率 oracle、贝叶斯类 oracle、以及 fine‑tune 对照进行对比；评估指标为准确率、负对数似然（NLL）、Brier 分数、ECE、KL 与诊断探针准确率。实验结果表明，即使准确率相同，冻结表示在 log loss 上显著劣于全输入；soft‑label 训练的表示在目标概率预测上优于硬 label 训练；fine‑tune 能弥补冻结导致的性能损失。

**⚠️ 局限性**

局限性包括：①仅在贝叶斯行动唯一、有限输出 log loss 的情形下给出完整理论；②对非唯一行动或其他损失的推广仍待研究；③实验聚焦机制而未评估其在大规模预训练中的普遍性；④理论假设冻结表示已充分覆盖所需信息，未考虑样本量、优化和头部容量等实际限制。

---

## 330. Mental-R1: Aligning LLM Reasoning for Mental Health Assessment

**arXiv ID:** 2606.13176 | [PDF](https://arxiv.org/pdf/2606.13176v1)

**作者:** Xin Wang `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种针对心理健康评估的强化学习框架CRPO，使LLM的推理过程与人类认知动态对齐

**💡 创新点**

通过阶段性熵正则化实现从探索到确定的认知转变，并结合认知评估理论构建多阶段推理结构

**🔧 技术方法**

在Qwen3-8B基础上使用Group Relative Policy Optimization (GRPO) + Stage-wise Entropy Regularization (SER) 与自定义奖励（格式奖励与平衡答案奖励）

**📊 数据集**

使用八个公开的心理健康文本分类数据集（Dreaddit、DATD、LT-EDI、DepSeverity、SDCNL、RSD、LID、FIG）

**📈 对比分析**

与多种RL基线（SFT、RLOO、ReMax、Reinforce++、GRPO、DAPO）以及多类LLM（Mentallama、Mental-GLM、Gemma-2、Llama-3.1、DeepSeek、GPT系列）比较，CRPO平均提升10.4个百分点加权F1，Mental‑R1在所有数据集上均领先，尤其在需要深度推理的样本上提升约15.6个百分点

**⚠️ 局限性**

局限在于需要手工设计阶段标签与奖励，对不同任务的通用性验证不足，且模型仍受限于训练数据规模与标签不均衡的挑战

---

## 331. Detecting Explanatory Insufficiency in Learned Representations: A Framework for Representational Vigilance

**arXiv ID:** 2606.13172 | [PDF](https://arxiv.org/pdf/2606.13172v1)

**作者:** Jacques Raynal `[一作]` (University of Montpellier), Jacques Margerit `[通讯]` (University of Montpellier)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出 VER 框架，用于监测机器学习模型的表示是否仍然能充分组织待建模现象。

**💡 创新点**

创新点在于把表示充分性视为可监测的诊断目标，并定义了五步诊断流程。

**🔧 技术方法**

采用无监督的诊断流程，包括表示识别、解释域划分、残差结构检测、解释抗拒评估和警戒信号。

**📊 数据集**

未使用具体数据集，本文为概念性框架，未进行实验。

**📈 对比分析**

无实验比较；论文仅阐述理论与评估思路，未给出性能指标。

**⚠️ 局限性**

局限在于缺乏具体实现、形式化定义与实证验证，需进一步开发基准与算法。

---

## 332. Entropic Generation of Binary Words

**arXiv ID:** 2606.13157 | [PDF](https://arxiv.org/pdf/2606.13157v1)

**作者:** Olivier Bodini `[一作]` (EREN Université Sorbonne Paris-Nord), Francis Durand `[通讯]` (EREN Université Sorbonne Paris-Nord)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2`

**🎯 论文内容**

提出了一种线性时间、近似信息熵最优的算法，用随机位回收技术生成固定汉明重量的二进制单词。

**💡 创新点**

创新点在于将均匀随机置换的熵反向提取并递归回收，从而实现 ε‑近似最优的随机位消费。

**🔧 技术方法**

核心技术包括可区分的 Fisher‑Yates 插入、DeconstructPermutation 的熵提取、递归回收链以及在多项式稀疏区间的分析。

**📊 数据集**

本工作为理论算法，无使用特定数据集；主要通过信息理论与复杂度分析来验证。

**📈 对比分析**

与传统部分 Fisher‑Yates、BalancedMerge 等方法比较，算法在时间上保持 O(n)，在随机位消费上达到 (1+ε)·log₂( C(n,k) ) 的界限，主要是理论上而非实测。

**⚠️ 局限性**

局限性包括适用于多项式稀疏区间的假设、递归回收实现的实际开销以及对真随机位来源的依赖。

---

## 333. Iterative Visual Thinking: Teaching Vision-Language Models Spatial Self-Correction through Visual Feedback

**arXiv ID:** 2606.13156 | [PDF](https://arxiv.org/pdf/2606.13156v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 334. LAUKIN: A Multi-jurisdictional Common Law Contract Dataset

**arXiv ID:** 2606.13184 | [PDF](https://arxiv.org/pdf/2606.13184v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 335. HyPE: Category-Aware Hypergraph Encoding with Persistent Edge Embeddings for Persona-Grounded Dialogue

**arXiv ID:** 2606.13142 | [PDF](https://arxiv.org/pdf/2606.13142v1)

**作者:** Sangwon Youn `[一作]` (Sungkyunkwan University), Youngjoong Ko `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HyPE 框架，利用 (Core, Expression, Sentiment, Category) 四元组对个性化信息进行高阶建模，并通过超图神经网络生成更一致的对话回复。

**💡 创新点**

创新点在于将 persona 句子拆解为四元组；以共享类别为超边构造超图；在 HyperGCN 中加入 Persistent Edge Embeddings 以引入类别先验；并通过 Encoder Soft‑Memory 与 Top‑k Hyper‑Tokens 并行注入语言模型，实现全新的高阶结构化 persona 表示。

**🔧 技术方法**

采用 HyperGCN 超图卷积网络、Persistent Edge Embeddings、Encoder Soft‑Memory、Top‑k Hyper‑Tokens、T5‑style 四元组抽取器，并在 GPT‑2、LLaMA‑3.2‑3B、Qwen2.5‑3B 等生成器上训练。

**📊 数据集**

使用 PersonaChat 数据集，并通过 GPT‑4o‑mini 对 PersonaChat 中的 persona 句子和对话句子进行四元组标注。

**📈 对比分析**

在 PersonaChat 上与文本拼接、Mean‑Pool、GCN、ORIG 等基线对比，HyPE 在 LLaMA‑3.2‑3B、Qwen2.5‑3B、GPT‑2 上在 BLEU‑1、BLEU‑4、ROUGE‑L、METEOR 以及 GPT‑4o judge 的 Persona Consistency、Engagingness、Relevance 评分上均表现优异，提升幅度约 1–2 BLEU‑1。

**⚠️ 局限性**

局限性包括仅在单语言、PersonaChat 简短 persona 上验证；四元组抽取依赖 OpenAI API；未评估长文本或多会话场景；实验仅单种随机种子；G‑eval 评判受 GPT‑4o 主观性影响。

---

## 336. Cascade Classification of Dermoscopic Images of Skin Neoplasms with Controllable Sensitivity and External Clinical Validation

**arXiv ID:** 2606.13135 | [PDF](https://arxiv.org/pdf/2606.13135v1)

**作者:** Elena S. Kozachok `[一作]` (Institute of System Programming of the Russian Academy of Sciences), Oleg I. Samovarov `[通讯]` (Institute of System Programming of the Russian Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对四种深度学习架构（ViT-B/16、Swin-S、ConvNeXt-S、EfficientNetV2-S）在三种分类方案（二分类、单阶段四分类、两阶段级联）下，对皮肤肿瘤的镜检图像进行自动诊断，并在开放国际数据集和俄国临床数据集上进行外部验证。

**💡 创新点**

①级联分解能够通过可调阈值控制系统敏感度，并在二分类边界弱的模型（如ViT-B/16）上显著提升整体F1；②在外部临床数据上量化了排名（ROC‑AUC）和校准（ECE）的泛化缺口，区分可校准的先验分布偏移与不可校准的特征分布漂移；③在相同条件下系统比较四种架构，发现二分类阶段的架构差异可统计显著，分类阶段无显著差异。

**🔧 技术方法**

Transformer（ViT、Swin）、现代CNN（ConvNeXt、EfficientNetV2）架构，全部使用ImageNet预训练并全微调；采用统一的增强、损失、优化器与学习率调度；级联模型采用固定ConvNeXt-S作为二分类门限，ViT或其他架构作为三分类鉴别器。

**📊 数据集**

训练集为九个ISIC Archive子数据集（共118 063张，按7/1/2拆分）；临床验证集包括Melanoscope AI移动系统（472张）和Sechenov University（77张），两者与训练集无重叠。

**📈 对比分析**

通过DeLong、McNemar、Bootstrap等配对显著性检验比较模型；内部hold‑out中四个架构ROC‑AUC相近（0.952–0.966），在Sechenov数据集ViT-B/16显著劣于Swin-S和EfficientNetV2；级联对ViT-B/16在Sechenov上提升宏F1显著（ΔF1≈0.11，p<0.05），其他架构无显著差异；在直接11类ISIC MILK10k基准中，单阶段模型整体准确率高但稀有类平均敏感度仅0.525。

**⚠️ 局限性**

临床样本量有限（尤其是恶性类），导致统计功效不足，宏指标置信区间宽；仅评估了两家俄国临床机构，难以推广至其他地区；未针对各架构做超参数微调；级联阈值需在目标数据上重新校准，外推性受限。

---

## 337. NaturalFlow: Reducing Disruptive Pauses for Natural Speech Flow in Simultaneous Speech-to-Speech Translation

**arXiv ID:** 2606.13121 | [PDF](https://arxiv.org/pdf/2606.13121v1)

**作者:** Dongwook Lee `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种流畅性感知优化框架（NaturalFlow），通过在实时语音到语音翻译中降低不自然停顿，实现近实时交互而不牺牲翻译质量。

**💡 创新点**

核心创新是“银牌”偏好设计：在 5 等分的停顿比例范围内仅挑选第二低停顿区间作为优选样本，并强制大幅度差距，既避免停顿比例过低导致语速失衡，又保持语义准确性。

**🔧 技术方法**

使用 Direct Preference Optimization（DPO）与长度归一化（DPO‑LN）对模型（Hibiki‑2B+LoRA）进行训练，偏好数据由多样化解码候选和 ASR‑BLEU / Silence Ratio 评估生成。

**📊 数据集**

训练与评估使用四个公开基准：短句 CVSS‑C、VoxPopuli、长句 Audio‑NTREX‑4L、mTEDx；候选解码采用 32 次采样。

**📈 对比分析**

与 SeamlessStreaming、StreamSpeech、Hibiki 等基线比较，NaturalFlow 在所有四个基准上将 Silence Ratio 降至 0.10–0.21 之间，同时保持或略低于基线的 ASR‑BLEU、ASR‑COMET，且延迟指标（Start/End Offset、LAAL）无显著退化；人类评测亦显示 55%‑68% 的首选率。

**⚠️ 局限性**

局限性：仅针对法语‑英语对，扩展到其他语言或更长上下文仍需验证；银牌偏好对候选分布的敏感性未系统评估；模型训练仍依赖大规模 LLM，资源成本高。

---

## 338. G-Long: Graph-Enhanced Memory Management for Efficient Long-Term Dialogue Agents

**arXiv ID:** 2606.13115 | [PDF](https://arxiv.org/pdf/2606.13115v1)

**作者:** Minjun Choi `[一作]` (Sungkyunkwan University), Youngjoong Ko `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 G-Long 框架，利用图结构的长时记忆银行与注意力加权重要性评分，实现对对话历史的高效、可解释的存储与检索，并以小型语言模型（sLM）完成三元组抽取，避免了大模型的高昂成本。

**💡 创新点**

创新点在于：①把对话内容转换为结构化三元组并构建图形记忆库；②使用跨注意力权重从 T5 摘要中无须额外 LLM 计算得到重要性分数；③基于图的多跳检索与双阶段混合重排序，实现高精度检索；④通过 sLM 和图结构显著降低 token 与 API 消耗，实现资源友好型长对话系统。

**🔧 技术方法**

技术包括：小型语言模型指令微调用于三元组抽取；T5 抽象摘要的跨注意力权重用于重要性评分；向量数据库（如 ChromaDB）做实体节点的稠密索引；图结构（边携带关系、重要性、时间戳）实现多跳检索；融合重排序公式（语义相似度、重要性、时间衰减）形成最终检索结果；最终使用大模型（gpt‑4o‑mini）生成回复。

**📊 数据集**

数据集包括：MSC、CC（多轮长对话评测）；LoCoMo（超长会话测试）；LME（检索问答评测）。

**📈 对比分析**

与六个基线（无记忆、长上下文、MemoryBank、LD‑Agent、FraCom、HippoRAG）在 MSC、CC、LoCoMo、LME 上进行对比；G‑Long 在生成指标上分别提升约 9.8%/8.9% BLEU‑2；在 LME 上 Recall@3 提升 40.8%；在 token 与 API 成本上比 LD‑Agent 低约 4.9 倍、token 消耗减少 63%。

**⚠️ 局限性**

局限：三元组抽取可能丢失细腻情感与修辞；对极端模糊或多义性表达依赖 sLM，可能引入噪声；图结构对未解析的代词、语义漂移仍有一定误检；缺乏自我校正机制，需人工监督确保数据质量。

---

## 339. Modern analog computing for solving differential and matrix equations

**arXiv ID:** 2606.13179 | [PDF](https://arxiv.org/pdf/2606.13179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 340. A Class of Multiparameter Signless Stirling Numbers of the First Kind and their $q$-Analogues

**arXiv ID:** 2606.13163 | [PDF](https://arxiv.org/pdf/2606.13163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

---

## 341. Reasoning for Mobile User Experience with Multimodal LLMs: Task, Benchmark, and Approach

**arXiv ID:** 2606.13192 | [PDF](https://arxiv.org/pdf/2606.13192v1)

**作者:** Ruichao Mao `[一作]` (Ant Group), Hai Rao `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了用于评估多模态大模型在 UI 体验推理方面的基准 UXBench，并构建了强化学习改进的 UI-UX 模型；

**💡 创新点**

创新点在于：①设计了三维（可用性、效率、可信度）细粒度 UX 诊断任务；②引入任务自适应奖励路由和非对称过渡奖励以抑制过度推理；③在 Qwen3-VL-4B-Thinking 基础上通过 RL 取得 SOTA；

**🔧 技术方法**

技术主要包括多模态大模型（Qwen3-VL-4B-Thinking）、链式思考推理、强化学习（GRPO）、奖励路由、过渡标记惩罚、数据增强与硬负样本挖掘；

**📊 数据集**

使用了 2,000 张真实 UI 截图组成的 UXBench 数据集，涵盖 8 个子任务；训练数据亦加入 26,680 条 UI/UX 及 4,919 条 MultiUI 多域样本；

**📈 对比分析**

与 Claude-4.5-Sonnet（0.6550）及其它 235B/4B 模型对比，UI-UX 在 0.7963 的准确率上超过 21.6% 的提升；同时保持低延迟；

**⚠️ 局限性**

局限在于：①基准仅覆盖静态截图，无法捕捉动态交互；②三维维度仍不完整，缺乏更深层心理学指标；③RL 训练复杂且对硬负样本和奖励设计高度依赖；

---

## 342. MIDSim: Simulating Multi-Channel Information Diffusion in Social Media with LLM-Powered Multi-Agent System

**arXiv ID:** 2606.13140 | [PDF](https://arxiv.org/pdf/2606.13140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 343. SICI: A Semantic-Pragmatic Complexity Index Reveals Regime Shifts in LLM Stance Detection

**arXiv ID:** 2606.13189 | [PDF](https://arxiv.org/pdf/2606.13189v1)

**作者:** Fuqiang Niu `[一作]` (University of Science and Technology of China), Bowen Zhang `[通讯]` (Shenzhen Technology University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在立场检测中的难点，提出并评估了七维语义-语用复杂度指数（SICI），并系统分析模型错误随复杂度的相位转移。

**💡 创新点**

创新点在于：①构建SICI指标，用七维语义-语用属性衡量实例难度；②发现模型错误呈低复杂度过度归因、中等复杂度不稳定、高复杂度偏向“None”的相位转移；③跨模型、跨数据集验证这一现象；④通过15种干预（提示、检索、辩论等）评估是否能突破高复杂度瓶颈。

**🔧 技术方法**

技术手段包括：Prompt‑based LLM推理（GPT‑3.5、GPT‑4o‑mini、DeepSeek‑V3、GPT‑4o）；SICI 评分体系与分段回归分析；干预实验（链式思维、Self‑consistency、检索增强、多人辩论等）；统计检验（Spearman、分段回归显著性等）。

**📊 数据集**

使用数据集：SemEval‑2016 Task 6、VAST（未见主题测试）、MTSD、P‑Stance 等。

**📈 对比分析**

对比方法：将 SICI 与文本长度、目标可见度等表面特征对准确率的影响做相关性和分段回归比较；干预实验对比各方法的宏观 F1/准确率。结果显示：SICI 与准确率负相关，分段回归显著优于线性；干预大多仅改变标签先验，最佳仍仅略优于基线，未能突破高复杂度阶段。

**⚠️ 局限性**

局限性：①SICI 评分由 LLM 自动完成，需人工验证以排除模型偏见；②实验聚焦英语社交媒体数据，其他语言、长文本、跨模态情况未知；③高复杂度样本量有限；④仅评估推理时干预，未研究基于 SICI 的训练或校准方案。

---

## 344. Sketching Intersection Profiles: A Simple Proof and Three Applications

**arXiv ID:** 2606.13182 | [PDF](https://arxiv.org/pdf/2606.13182v1)

**作者:** Flavio Chierichetti `[一作]` (Reddit), Andrew Tomkins `[通讯]` (Google)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文通过构造集合分布的交集轮廓，给出了图的邻域大小、覆盖函数和随机效用模型这三类 sketch 问题的下界，证明它们都需要 Ω(n²) 位存储，并提供了相匹配或几乎匹配的上界；

**💡 创新点**

创新点在于将集合分布交集轮廓与数据库项集频率估计问题关联，并利用简单概率打包和随机矩阵构造技术，得到新的 Ω(n²) 下界，填补了前人对于 vertex neighborhood、coverage function 以及 ℓ∞-距离 RUM 的空白；

**🔧 技术方法**

主要技术包括：打包论证、Hoeffding 及 Chernoff 边界、VC 维度推理、随机二进制矩阵构造、信息论的互信息与 KL 上界等；

**📊 数据集**

论文为理论性工作，未使用具体实验数据集，所有结果均来自理论证明；

**📈 对比分析**

与已有的上界相比，本文给出的下界与上界仅相差多项式对数因子，证明这些 sketch 问题的真正复杂度是 Θ̃(n²)；

**⚠️ 局限性**

局限性：对于 ℓ∞-距离的 RUM sketch 仍存在对数因子缺口，尚未确定是否可以进一步提升下界；此外，关于覆盖函数的“proper” sketch 是否存在 Ω̃(n²) 大小的构造仍未解决。

---

## 345. On the Counting Sequence of Z-convex Polyominoes

**arXiv ID:** 2606.13158 | [PDF](https://arxiv.org/pdf/2606.13158v1)

**作者:** Luca Castelli `[一作]` (University of Insubria), Paolo Massazza `[通讯]` (University of Insubria)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一组公式和方程，基于这些公式和方程开发了一个C++程序，用于计算具有最大2度凸性的凸多面体的最长计数序列。

**💡 创新点**

创新点在于通过稍微改变组合分解，获得了一组新的递归方程和公式，从而实现了更高效的计算。

**🔧 技术方法**

使用了C++编程语言，并采用了动态编程和哈希表技术来实现计算。

**📊 数据集**

使用的数据集是凸多面体的计数序列，计算范围为n≤75。

**📈 对比分析**

通过与现有的组合方法进行比较，性能得到了显著提升，能够在几天内计算出n=75的计数序列。

**⚠️ 局限性**

限制在于程序目前仅适用于n<256，并且在计算复杂度上存在O(n^14)的最坏情况。

---

## 346. The Limits of Time

**arXiv ID:** 2606.13138 | [PDF](https://arxiv.org/pdf/2606.13138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6`

---

## 347. Fully Distributed Multi-View 3D Tracking in Real-Time

**arXiv ID:** 2606.13127 | [PDF](https://arxiv.org/pdf/2606.13127v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 348. Transformer-Guided Graph Attention for Direct Cardiac Mesh Reconstruction: A Structural Digital Twin Framework

**arXiv ID:** 2606.13188 | [PDF](https://arxiv.org/pdf/2606.13188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 349. Under What Conditions Can a Machine Become Genuinely Creative?

**arXiv ID:** 2606.13196 | [PDF](https://arxiv.org/pdf/2606.13196v1)

**作者:** Yong Zeng `[一作]` `[通讯]` (Concordia University), Yong Zeng (Concordia University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了基于 Designics 理论的“创意机器”需求框架，阐释了机器真正创造力的条件和过程。

**💡 创新点**

创新点在于将创造力从单纯的输出新颖性转向递归干预动态，提出十项相互依赖的需求，并将主动 AI 伦理嵌入创作循环。

**🔧 技术方法**

采用了设计学理论、递归干预动力学、以及对现有计算与生物医学研究的理论映射来构建框架。

**📊 数据集**

并未使用新的实验数据集，而是借鉴了已发表的网格生成、EEG 监测、工作负荷等领域的实验结果来进行示例说明。

**📈 对比分析**

本文未给出定量比较与性能指标，而是通过案例研究展示框架在网络物理与网络生物领域的可操作性。

**⚠️ 局限性**

局限性在于框架仍处于理论层面，缺乏形式化数学定义与实际实现的评估，且需要后续实证验证来检验其可行性。

---

## 350. MemRefine: LLM-Guided Compression for Long-Term Agent Memory

**arXiv ID:** 2606.13177 | [PDF](https://arxiv.org/pdf/2606.13177v1)

**作者:** Minjae Kim `[一作]` (Korea University), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MemRefine框架，对已有长程记忆存储进行压缩以满足固定存储预算，同时保持后续任务性能。

**💡 创新点**

创新在于使用LLM评判候选记忆对的删除、合并或保留，而不是仅靠相似度或图结构；并将此压缩过程作为独立后构建模块。

**🔧 技术方法**

主要技术包括基于语义相似度挑选记忆对，LLM判定重合/互补/不同，LLM合并生成新条目，循环迭代直至满足预算。

**📊 数据集**

实验使用LoCoMo（标准与3×、10×扩展）和LongMemEval_S两大对话记忆基准。

**📈 对比分析**

与未压缩基准及两种规则基线（相似度阈值、PageRank）对比，MemRefine在多种预算下保持接近或优于原始性能，尤其在紧凑预算下表现更优。

**⚠️ 局限性**

局限在于仅验证两类记忆架构，未覆盖其他多代理或工具使用场景；且压缩过程需依赖LLM，计算成本和隐私风险需进一步评估。

---

## 351. The End of Code Review: Coding Agents Supersede Human Inspection

**arXiv ID:** 2606.13175 | [PDF](https://arxiv.org/pdf/2606.13175v1)

**作者:** Martin Monperrus `[一作]` `[通讯]` (KTH Royal Institute of Technology), Martin Monperrus (KTH Royal Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出编程代理已达到足以替代人类代码审查的能力阈值，并通过分析现有技术与数据集证明其在缺陷检测、规范执行、知识转移等方面可匹配或优于人类审查。

**💡 创新点**

创新点在于系统性地论证人类审查的冗余性，提出将审查门槛完全移交给代理的完整框架，并对传统AI+人类审查模式进行批判性评估。

**🔧 技术方法**

使用大语言模型驱动的编程代理（如Claude Code、Codex、GitHub Copilot、CodeReviewer、SWE-Agent等）及其工具调用循环实现自动化审查与修复。

**📊 数据集**

主要基于SWE-bench及其Verified子集、真实GitHub Issue数据集、以及工业环境中的评估实验。

**📈 对比分析**

通过与人类审查者在缺陷检测、代码风格、合规性等指标的对比，显示代理在SWE-bench上从1.7%提升至70%以上的成功率，且在自动化评论质量上与训练有素的人类评审相当。

**⚠️ 局限性**

局限包括模型的幻觉与漏判、对安全与合规性的潜在盲点、提示注入攻击风险以及对架构与伦理决策仍需人工监督。

---

## 352. Getting Better at Working With You: Compiling User Corrections into Runtime Enforcement for Coding Agents

**arXiv ID:** 2606.13174 | [PDF](https://arxiv.org/pdf/2606.13174v1)

**作者:** Yujun Zhou `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个端到端的规则编译与执行框架 Test‑Time Rule Acquisition and Compiled Enforcement（简称TRACER），将用户在交互中给出的自然语言纠错转换为可在编码代理运行时强制执行的运行时规则，从而提升用户偏好遵守率。

**💡 创新点**

创新点在于：①将纠错从被动存储转为主动执行约束；②设计了纠错检测、规则提取、生命周期管理与可执行检查编译的完整流水线；③在运行时动态注入检查与重试机制，使偏好成为任务完成的硬性条件。

**🔧 技术方法**

主要技术包括：轻量级 LLM（Gemma‑4 31B）用于纠错检测与规则抽取；层次化规则库与五种生命周期处理策略（Noop、Update、Supersede、Split、New）；运行时检查层（确定性、语义级、意图级）通过工具调用、文件状态或模型评估实现；以及基于 hook 的事件驱动执行控制。

**📊 数据集**

使用了两个公开评测基准：ClawArena（编码任务+偏好覆盖）和 MemoryArena（记忆任务+项目约束），并在此基础上增设了用户循环包装以模拟真实交互。训练和评测均采用与 PersonaGym 类似的用户模拟器生成纠错。

**📈 对比分析**

与无记忆、Mem0、Hindsight、ReMe‑Light 等记忆方法对比，TRACER 在 ID 任务中将偏好违规率从 100% 降至 37.6%（ClawArena）/60.5%（MemoryArena），在 OOD 任务中进一步降至 2.0%/97.0%；任务通过率保持与无记忆水平相近或略有提升；用户交互轮数平均从 2.0 降至 1.37（ID）/1.02（OOD），并未显著增加推理时间，平均约 42 s/分钟。

**⚠️ 局限性**

局限性包括：①规则编译依赖 LLM 的准确性，错误抽取会导致误规则或漏规则；②在面对高度多样化或未覆盖的偏好时，系统仍会出现 90%+ 违规率；③对复杂语义规则的检测与执行仍依赖模型自评，精度不如确定性检查；④当前仅在模拟环境验证，实际部署中的实时纠错与多用户同步等问题待进一步研究。

---

## 353. NTS-CoT: Mitigating Hallucinations in LLM-based News Timeline Summarization with Chain-of-Thought Reasoning

**arXiv ID:** 2606.13171 | [PDF](https://arxiv.org/pdf/2606.13171v1)

**作者:** Feng Lyu `[一作]` (Central South University), Haolun Wu `[通讯]` (McGill University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 NTS-CoT 框架，利用 Chain‑of‑Thought（CoT）技术解决 LLM 在新闻时间线摘要中的幻觉问题。

**💡 创新点**

创新点在于：①设计了两种 CoT 模块（Element‑CoT 与 Causal‑CoT）分别针对新闻摘要的非忠实内容和日期事件摘要的信息缺失；②构建关联日期图与事件聚类并结合时间重要性与事件显著性进行日期排序；③通过多层 CoT 推理提升摘要的真实性与时间准确性。

**🔧 技术方法**

技术方法包括：链式推理（CoT）、结构化提示（Element‑CoT）、事件聚类与关联日期图、因果关系分析（Causal‑CoT）、LLM（Llama3‑8B‑Instruct）等。

**📊 数据集**

使用了三大公开时间线摘要基准数据集：Timeline17、Crisis、Entities。

**📈 对比分析**

与提取式、抽象式、LLM 基线以及不同提示策略进行对比，NTS‑CoT 在 AR‑1/AR‑2、Date‑F1 指标上分别提升 23.4%/33.4%/10%，并在人工评估中在忠实性和完整性上优于 LLM‑TLS，显示出显著性能提升。

**⚠️ 局限性**

局限性：①AR 指标仍略低，受 LLM 重新组织语言导致的表达偏差影响；②对极长新闻集仍可能出现冗余或细节缺失；③方法依赖手工构造的 CoT 提示，适用性与可迁移性需要进一步验证。

---

## 354. Rethinking RAG in Long Videos: What to Retrieve and How to Use It?

**arXiv ID:** 2606.13141 | [PDF](https://arxiv.org/pdf/2606.13141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 355. (Un)ranking Permutation Classes

**arXiv ID:** 2606.13160 | [PDF](https://arxiv.org/pdf/2606.13160v1)

**作者:** Nathanaël Hassler `[一作]` (Université Bourgogne Europe), Vincent Vajnovszki `[通讯]` (Université Bourgogne Europe)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了长度为3的模式（如231、321等）避免的排列集合，并给出了在字典序和逆字典序下的排名与逆排名算法；

**💡 创新点**

创新点在于利用这些排列的递归结构（231可拆分为(1⊖σ)⊕τ，321可映射到Dyck路径）和Catalan数的计数公式，设计出闭式递归的排名/逆排名算法；

**🔧 技术方法**

技术方法包括组合递归分解、Dyck路径与Catalan数的对应、补运算和逆运算的对称映射，以及动态规划求解球形序列t(i,j)；

**📊 数据集**

所用数据集为理论计数（Catalan数c_n），并未依赖具体实验数据集；

**📈 对比分析**

与传统的排名/逆排名方法（如Steinhaus–Johnson–Trotter、Gray码等）相比，算法在理论上实现复杂度为O(n)或O(n^2)（具体取决于递归实现），但论文未给出实际运行时间的实验比较；

**⚠️ 局限性**

局限性包括实现复杂度较高，尤其在大规模n时的性能和空间占用；此外，对多模式或更大长度模式的推广仍是未解决的问题。

---

## 356. Touchard-Riordan Polynomials and Schur-positivity of Set Partitions

**arXiv ID:** 2606.13149 | [PDF](https://arxiv.org/pdf/2606.13149v1)

**作者:** Eli Bagno `[一作]` (Jerusalem College of Technology), David Garber `[通讯]` (Holon Institute of Technology)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究了以“相邻元素是否在同一块”为标准的降序函数对集合分区生成的对称函数的 Schur 正性，并给出了它们的 Schur 展开。

**💡 创新点**

提出了新的降序参数，并揭示该参数下的生成函数与 Touchard‑Riordan 多项式（以及无交错分区对应的 Motzkin 三角形）之间的密切联系，首次证明了集合分区族在该参数下的 Schur 正性。

**🔧 技术方法**

主要技术包括：将集合分区与标准 Young 表格（钩形）以及弦图（Chord Diagram）之间建立双射；利用 Gessel 的定理把对称函数展开转化为计数问题；通过交叉数与 Touchard‑Riordan 多项式的生成函数进行匹配；对无交错分区使用 Motzkin 三角形作为系数。

**📊 数据集**

本研究没有使用外部实验数据集，而是完全基于组合学的理论构造与证明，涉及的对象为集合分区、弦图和标准 Young 表格。

**📈 对比分析**

由于结论是全局性的组合性结论，未涉及传统意义上的实验比较或性能评估。论文通过严格的双射和生成函数等数学证明，确保了结果的正确性与完整性。

**⚠️ 局限性**

局限性在于：仅适用于以“相邻元素共块”为标准的降序函数；对其他降序定义（如匹配中的几何下降）并未给出一般性结论；并且论文仅讨论了钩形 Schur 基底，未探讨更一般形状的扩展。

---

## 357. The Geometry of Phase Transitions in Generative Dynamics via Projection Caustics

**arXiv ID:** 2606.13191 | [PDF](https://arxiv.org/pdf/2606.13191v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 358. JiRAIYA: A Reputation-Based Hierarchical Federated Learning Framework on Web3

**arXiv ID:** 2606.13180 | [PDF](https://arxiv.org/pdf/2606.13180v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 359. Select and Improve: Understanding the Mechanics of Post-Training for Reasoning

**arXiv ID:** 2606.13125 | [PDF](https://arxiv.org/pdf/2606.13125v1)

**作者:** Akshay Krishnamurthy `[一作]` (Microsoft Research), Nived Rajaraman `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对强化学习在数学推理中的机制进行系统实验，发现其通过策略选择与策略改进提升性能。

**💡 创新点**

提出两大核心机制（策略选择和策略改进），并阐明其对数据多样性和难度梯度的依赖，解释了策略放大与组合等现象。

**🔧 技术方法**

使用 Qwen‑2.5‑1.5B-Instruct 语言模型，结合 LoRA 微调、GRPO 强化学习、策略识别分类器等技术。

**📊 数据集**

构造有限域算术任务（GF(11)、GF(13)）的评估与逆推问题，配合前向/后向推理策略的 SFT 数据和不同难度/倾斜度的 RL 数据集。

**📈 对比分析**

与仅 SFT 或仅 RL 训练的基准对比，结果显示策略选择使混合策略模型在更难问题上准确率突破 95%，策略改进在更高难度数据上实现持续提升；整体性能大幅优于单策略模型，且比传统 RL 方案更稳健。

**⚠️ 局限性**

局限性在于未发现 RL 能生成全新推理策略，仅能细化已有策略；需要更丰富的预训练与 SFT 数据，且对更复杂任务的推广仍有待验证。

---

## 360. MP3: Multi-Period Pattern Pre-training forSpatio-Temporal Forecasting

**arXiv ID:** 2606.13119 | [PDF](https://arxiv.org/pdf/2606.13119v1)

**作者:** Lilan Peng `[一作]` (Southwest Jiaotong University), Tianrui Li `[通讯]` (Southwest Jiaotong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 MP3 插件，通过在长时序上进行预训练，学习多周期时空模式，并将其冻结后与现有 STGNN Backbone 通过门控融合，以提升交通流等时空预测任务。

**💡 创新点**

创新点包括：① 端到端的多周期模式学习框架（多周期时间建模、空间建模与跨周期交互）；② 利用 edge convolution 对时间维度进行独立的 intra‑/inter‑period 处理；③ 用瓶颈投影+全局记忆池高效捕获异构全局空间关联；④ 采用因果增强 Transformer 强制强周期→弱周期的单向因果交互。

**🔧 技术方法**

技术手段：FFT 频域周期识别、2D 时序重构、edge convolution、瓶颈投影+稀疏图卷积、动量记忆银行、因果掩码 Transformer、门控融合、预训练+冻结、Adam 训练。

**📊 数据集**

实验使用四个交通流基准（PEMS03/04/07/08）和大规模 CA 数据集，采样间隔 5 分钟，窗口长度 1248，预测 12 步。

**📈 对比分析**

与 GPT‑ST 以及 STGCN、GWNet、STWA、MSDR、STNorm 等多种 Backbone 对比，MP3 在 MAE 上平均下降 4.7%，RMSE 下降 5.0%，并在所有模型和数据集上实现或接近最优结果，显著提升短期预测精度。

**⚠️ 局限性**

局限性：需要较长历史窗口进行预训练，窗口长度与计算/内存开销存在折衷；预训练与冻结增加部署复杂度；对极短期预测（如 3 步）提升有限，且因果模块仍基于经验约束，缺乏理论解释。

---

## 361. WHAR Arena: Benchmarking the State of the Art in Efficient Wearable Human Activity Recognition

**arXiv ID:** 2606.13194 | [PDF](https://arxiv.org/pdf/2606.13194v1)

**作者:** Maximilian Burzer `[一作]` (Karlsruhe Institute of Technology), Tobias Röddiger `[通讯]` (IPAI Foundation gGmbH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一的可复现WHAR基准框架，集成30个多样化数据集、17种代表性模型，并在统一的跨主体评估协议下同时衡量预测性能与设备端延迟、内存与模型尺寸等部署效率指标。

**💡 创新点**

创新点在于：①设计了标准化的数据处理与模型接口库，消除数据与实验流程的异构性；②采用跨主体（KSG）拆分，真实模拟域漂移；③联合评估性能与多维部署成本，并通过Pareto前沿与效率指数实现多目标模型排序。

**🔧 技术方法**

技术手段包括：深度学习架构（CNN、RNN、注意力网络）与经典机器学习基线（RandomForest、KNN、SVM），Python数据处理库，ExecuTorch在Pixel 8手机上测量推理延迟与峰值内存，统一的实验脚本与配置驱动流水线。

**📊 数据集**

使用了30个公开WHAR数据集，涵盖ADL、行走、健身、跌倒检测和健康监测等场景，涵盖多种传感器（加速度计、陀螺仪、磁力计、ECG、EEG等）与多设备/多模态设置。

**📈 对比分析**

通过统一的LOS（KSG）拆分与宏F1评估，随后在同一设备上采集推理延迟、峰值内存与模型尺寸，对比得到CNN-HAR在平均宏F1约67.7%居首，但多模型性能聚集在相近区间；Pareto分析揭示TinierHAR、CNN-HAR与RandomForest在不同部署成本维度上占据主导位置。

**⚠️ 局限性**

局限性包括：仅在单一Android设备上评估部署效率，未覆盖其他低功耗边缘平台；模型超参数未针对每个数据集进行深度调优；数据集与模型选择虽广泛但并非完全覆盖WHAR领域；部署测量未考虑能耗与实时性外部因素。

---

## 362. The Curious Case of Reversible Elementary Second Order Cellular Automaton 115

**arXiv ID:** 2606.13159 | [PDF](https://arxiv.org/pdf/2606.13159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

---

## 363. The Clustering Strikes Back: Building Cost-Effective and High-Performance ANNS at Scale with Helmsman

**arXiv ID:** 2606.13145 | [PDF](https://arxiv.org/pdf/2606.13145v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 364. ARMOR-MAD: Adaptive Routing for Heterogeneous Multi-Agent Debate in Large Language Model Reasoning

**arXiv ID:** 2606.13197 | [PDF](https://arxiv.org/pdf/2606.13197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 365. Multi-Modal Multi-Agent Robotic Cognitive Alignment enabled by Non-Invasive Consumer Brain Computer Interfaces: A Proof of Concept Exploration

**arXiv ID:** 2606.13190 | [PDF](https://arxiv.org/pdf/2606.13190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 366. Snake Polyominoes of Maximal Area in a Rectangle

**arXiv ID:** 2606.13155 | [PDF](https://arxiv.org/pdf/2606.13155v1)

**作者:** Alexandre Blondin Massé `[一作]` (Université du Québec à Montréal), Alain Goupil `[通讯]` (Université du Québec à Trois-Rivières)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在给定尺寸的矩形内可容纳的蛇形多米诺（snake‑like polyomino）的最大面积，并给出了从小尺寸矩形递归生成所有此类多米诺的算法。

**💡 创新点**

提出了针对高度 h≤5 的蛇形多米诺最大面积的显式公式，首次将这一问题与侧面（hull）结合，给出可构造的最优蛇形多米诺族群，并对所有可能的侧面情况做了完整的结构划分。

**🔧 技术方法**

主要使用离散几何与图论的概念（链、边界、自由度、侧面），构造了“有侧面蛇形因子”（sided snake factors）的定义，并基于递归分块、侧面合并的算法来枚举与优化；证明部分采用了图的连通性、度数约束以及模运算的周期性分析。

**📊 数据集**

本文并未使用实验数据集，而是通过完全枚举与理论证明来验证公式，使用了对小尺寸矩形（如 h≤5、w≤10 等）进行的穷尽检查。

**📈 对比分析**

与传统经验公式（如 2/3 规则）相比，本文给出的最大面积公式在 h≤5 时完全匹配（无误差），并通过算法生成的蛇形多米诺实例验证了公式的可行性。性能方面，算法在小尺寸矩形下能够在数秒内枚举完所有因子，但对更大尺寸的矩形时计算量随维度快速指数增长。

**⚠️ 局限性**

局限性：仅对高度不超过 5 的矩形给出了完整证明，关于更高高度的通用公式仍为猜想；算法的时间与空间复杂度对大尺寸矩形不具备可扩展性；侧面约束的完整枚举在极大尺寸下可能导致爆炸性增长。

---

## 367. Fibonacci and Catalan Numbers Meet in Staircase Polyominoes

**arXiv ID:** 2606.13152 | [PDF](https://arxiv.org/pdf/2606.13152v1)

**作者:** Jean-Luc Baril `[一作]` (Université Bourgogne Europe), Diego Villamizar `[通讯]` (Xavier University of Louisiana)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文研究了以阶梯形下边界为特征的Fibonacci多格子多面体，推导了其多变量生成函数，进一步细化了Turban关于面积的Fibonacci数枚举，并加入周长和面积等额外统计量。

**💡 创新点**

创新点在于：①将Fibonacci多格子多面体的周长参数纳入生成函数，得到更精细的枚举；②通过催化函数方程和核方法，获得闭式生成函数和Catalan数系数公式；③提供了面积、水平/垂直半周长三维统计的完整枚举。

**🔧 技术方法**

主要技术手段包括催化函数方程的构造、核方法求解、迭代展开、闭式化简，以及利用Catalan数生成函数进行系数提取。

**📊 数据集**

本研究为纯组合数学理论研究，未使用任何外部数据集或实验数据。

**📈 对比分析**

由于研究对象为组合计数，未与实验方法或基准进行性能比较；研究通过解析推导和符号计算验证结果，展示了生成函数与已知Fibonacci、Catalan数列的一致性。

**⚠️ 局限性**

局限性包括：①仅针对列凸多格子多面体的阶梯下边界；②未探讨更一般的步长或非列凸情形；③结果多为符号式，缺乏可扩展到大规模计数的数值实现；④对特殊周长/面积分布的进一步统计（如期望、方差）未给出。

---

## 368. Random Generation of $k$-coloured Motzkin Paths

**arXiv ID:** 2606.13151 | [PDF](https://arxiv.org/pdf/2606.13151v1)

**作者:** Elena Barcucci `[一作]` (University of Florence), Renzo Pinzani `[通讯]` (University of Florence)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2`

**🎯 论文内容**

研究k色Motzkin路径，并给出了其前缀在奇高度终止的计数公式；

**💡 创新点**

提供了该计数公式的两种独立证明（分析和组合）并基于组合证明提出了线性时间的随机生成算法；

**🔧 技术方法**

使用了生成函数、递推关系、Raney引理、反射变换等组合与分析工具；

**📊 数据集**

本研究属于纯理论组合学，没有使用实际数据集；

**📈 对比分析**

通过理论分析估算了算法的平均复杂度，结果表明生成前缀的平均成本随n线性增长（约4n），整体算法保持线性时间；

**⚠️ 局限性**

方法尚未与现有更高效的前缀生成算法相比，且仅针对k色水平步的情况，缺乏实验验证和对大k值的性能分析。

---

## 369. Exhaustive Generation of Genus-One Knot and Link Diagrams via Maps on the Torus

**arXiv ID:** 2606.13161 | [PDF](https://arxiv.org/pdf/2606.13161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

---

## 370. When Does Routing Become Interpretable? Causal Probes on Block Attention Residuals

**arXiv ID:** 2606.13168 | [PDF](https://arxiv.org/pdf/2606.13168v1)

**作者:** Aydin Javadov `[一作]` `[通讯]` (ETH Zurich), Aydin Javadov (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Block AttnRes模型中探究显式深度路由是否能直接解释Transformer的信息流，并通过因果消融实验评估路由的机制性解释力。

**💡 创新点**

证明仅靠将路由显式化不足以实现机制性解释，真正的可解释性需要路由在训练中被学习；发现三种局部化的路由机制，并揭示路由权重大小与因果重要性不一致。

**🔧 技术方法**

使用路由消融（mask‑and‑renormalise）技术，对不同源族、子层和深度区间进行干预；采用Block AttnRes的软max路由；构造合成键值检索任务进行评估。

**📊 数据集**

利用200个合成键值检索样本（包含三种模板），不使用真实语料库。

**📈 对比分析**

对比两种同规模（0.6B）模型：一是用recency‑bias调度包装的标准Qwen3（路由内容无关），另一是从头训练的Block AttnRes Qwen3；两者在准确率上相近（约0.56 vs 0.54），但路由消融显示后者表现出多种因果显著的局部化路由模式。

**⚠️ 局限性**

局限性包括：任务过于简单，只有两种模型/训练设置，路由干预仅在块级别，消融方法未真正删除信息，且整体准确率仅在中等水平。

---

## 371. TerraBench: Can Agents Reason Over Heterogeneous Earth-System Data?

**arXiv ID:** 2606.13148 | [PDF](https://arxiv.org/pdf/2606.13148v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 372. An Extensible and Lightweight Unified Architecture for Demosaicing Pixel-bin Image Sensors

**arXiv ID:** 2606.13136 | [PDF](https://arxiv.org/pdf/2606.13136v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 373. Learning-Augmented Approximation for Unrelated-Machines Makespan Scheduling

**arXiv ID:** 2606.13133 | [PDF](https://arxiv.org/pdf/2606.13133v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 374. MiniPIC: Flexible Position-Independent Caching in <100LOC

**arXiv ID:** 2606.13126 | [PDF](https://arxiv.org/pdf/2606.13126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 375. EvoBrowseComp: Benchmarking Search Agents on Evolving Knowledge

**arXiv ID:** 2606.13120 | [PDF](https://arxiv.org/pdf/2606.13120v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 376. Multi-Field Hybrid Retrieval-Augmented Generation for Maritime Accident Root Cause Analysis

**arXiv ID:** 2606.13249 | [PDF](https://arxiv.org/pdf/2606.13249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 377. Polar Decoding Tree Pruning Based on Soft Output Extraction

**arXiv ID:** 2606.13214 | [PDF](https://arxiv.org/pdf/2606.13214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 378. ReSET: Accurate Latency-Critical NVFP4 Reasoning via Step-Aware Temperature Scaling

**arXiv ID:** 2606.13233 | [PDF](https://arxiv.org/pdf/2606.13233v1)

**作者:** Sihwa Lee `[一作]` (Hanyang University), Jungwook Choi `[通讯]` (Hanyang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于推理步骤熵的自适应温度缩放方法 ReSET，以及针对小批量解码的 CUDA‑核心 NVFP4 GEMV kernel，以提升低精度推理的准确率和吞吐量。

**💡 创新点**

创新点在于：①通过步骤级熵分析发现 token 级熵受步级不确定性影响，进而设计自适应阈值的温度缩放策略；②针对 TensorCore 在小批量下低占用率的瓶颈，研发了第一套 CUDA‑核心 NVFP4 小批量 kernel。

**🔧 技术方法**

使用技术包括：NVFP4 低精度量化、熵基温度调节、在线步骤熵估计、CUDA 核心自定义 GEMV、量化激活拆包与共享缩放等。

**📊 数据集**

实验采用 AIME‑120、GPQA‑Diamond、LiveCodeBench 等公开推理数据集。

**📈 对比分析**

与多种 PTQ 基线和 TensorCore kernel 进行对比，ReSET 在 AIME‑120 上提升约 2.6 分；CUDA‑核心 kernel 在小批量下速度提升 1.57–2.49 倍；端到端比 BF16 提升 1.69 倍，整体性能显著优于现有方案。

**⚠️ 局限性**

局限性包括：需要在线熵估计和温度参数微调；对极大模型或不同推理模式下的适用性仍需进一步验证。

---

## 379. See Selectively, Act Adaptively: Dual-Level Structural Decomposition for Bimanual Robot Manipulation

**arXiv ID:** 2606.13279 | [PDF](https://arxiv.org/pdf/2606.13279v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 380. Zero-Shot Captioning for Cultural Heritage: Automated Image Analysis of Traditional Indonesian Clothing

**arXiv ID:** 2606.13275 | [PDF](https://arxiv.org/pdf/2606.13275v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 381. Cross-Modal Masked Compositional Concept Modeling for Enhancing Visio-Linguistic Compositionality

**arXiv ID:** 2606.13288 | [PDF](https://arxiv.org/pdf/2606.13288v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 382. TimeLens: On-Device Artifact Recognition with Retrieval-Augmented Question Answering for the Grand Egyptian Museum

**arXiv ID:** 2606.13267 | [PDF](https://arxiv.org/pdf/2606.13267v1)

**作者:** Rawan Hesham `[一作]` (Capital University), Omar Wagih `[通讯]` (Capital University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款双语移动导览应用TimeLens，结合实时设备端物体检测与检索增强生成问答，服务埃及大金字塔博物馆访客；

**💡 创新点**

核心创新在于：①通过数据质量驱动的迭代标注流程提升细粒度展品识别，②实现全离线双语检索增强生成问答，彻底消除hallucination；

**🔧 技术方法**

使用技术包括YOLOv8n (TensorFlow Lite)、Gemma 4 E2B (Q4_K_M)语言模型、ChromaDB向量检索、Flutter UI、Ollama推理、MiniLM多语句嵌入、Flash Attention；

**📊 数据集**

数据集为51件博物馆展品的视频帧约30,000张（含自动注释、规则清洗、手工标注）以及108条双语知识记录（共215块）；

**📈 对比分析**

通过对比不同标注版本（v1至v3）和模型量化设置，YOLOv8n在手工标注后达mAP@0.5=0.995、mAP@0.5:0.95=0.924，RAG端到端平均响应≈5.9s，所有30个测试问题无hallucination、无截断，且检索成功率100%；

**⚠️ 局限性**

局限性包括知识库覆盖仅108条记录、查询分类依赖关键字导致误路、模型无跨请求记忆、检测验证基于图像级拆分，缺乏视频级泛化评估。

---

## 383. Once-for-All: Scalable Simultaneous Forecasting via Equilibrium State Estimation

**arXiv ID:** 2606.13285 | [PDF](https://arxiv.org/pdf/2606.13285v1)

**作者:** Beinan Xu `[一作]` (RMIT University), Feng Liu `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为Equilibrium State Estimation（ESE）的新框架，用于在单次计算中同时预测多个相互作用的系统，先估计整体平衡状态再基于偏离平衡的差值进行预测。

**💡 创新点**

创新点在于将多系统预测视为整体平衡问题，通过属性驱动的平衡估计与时间序列分离，实现一次性多目标预测，并能与现有单目标或多目标预测模型无缝结合，显著提升效率与可扩展性。

**🔧 技术方法**

技术手段包括属性归一化、最大似然估计得到属性系数、通过迭代校正向量和阻尼系数实现平衡估计、使用协整检验作为收敛判据、线性自回归模型估计整体趋势以及与 LSTM、Informer、SCINet 等基线模型的融合。

**📊 数据集**

数据集涵盖三类：合成数据（10 系统、20 步输入、1 步预测）、外汇数据（16 组 G20 货币相对美元、5 年日数据及宏观经济属性）和 COVID‑19 传播数据（维多利亚州 79 个市/20/320 区域的每日新增病例，属性为人口和波段）。

**📈 对比分析**

与 ARIMA、LSTM、DLinear、Informer、DeepAR、PatchTST、VAR、NLinear、FiLM、SCINet、KVAE、TPGNN、TimeLLM 等 13 种 SOTA 方法对比。ESE 单独使用即可与 SOTA 竞争；与 SOTA 结合时准确率提升或持平，且运行时间降低 10–70 倍；在大规模系统（如 320 区域）下依旧保持低误差并显著提高可扩展性。

**⚠️ 局限性**

局限性：1) 仅适用于满足定义 1–3 的相互关联系统，不能用于无关系统；2) 需要足够长的历史序列（短序列可能无法充分估计平衡）；3) 估计的是统计平衡状态，若真实系统不满足可协整或平衡假设，性能可能受限。

---

## 384. ERTS: Adversarial Robustness Testing of Ethical AI via Semantic Perturbation in a Bounded Consequence Space

**arXiv ID:** 2606.13282 | [PDF](https://arxiv.org/pdf/2606.13282v1)

**作者:** Pratyush Chaudhari `[一作]` `[通讯]` (Independent Researcher), Pratyush Chaudhari (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向伦理AI的对抗鲁棒性测试系统ERTS，利用22维伦理后果空间对伦理决策进行编码，并在受约束的语义扰动空间下评估模型的伦理稳定性。

**💡 创新点**

创新点在于：①构建基于伦理理论的22维伦理后果空间；②设计17种语义扰动函数并引入6类有效性约束（包括语义一致性约束C5）；③提出四组件伦理不稳定性指数EII并给出加权；④实现域自适应的预部署评估判定。

**🔧 技术方法**

采用的技术包括：向量空间编码、语义扰动函数、约束验证、伦理不稳定性指数计算、基于阈值的多维评估。

**📊 数据集**

使用了50个跨8个应用领域的伦理情景样本，构造1,500个对抗测试案例；同时评估了4个结构化基线模型和2个生产LLM（Gemini 2.0 Flash、Llama 3.2）。

**📈 对比分析**

通过比较各模型的ERS、失效率和领域判定结果，发现Gemini 2.0 Flash在所有场景下无失效，ERS 0.94；Llama 3.2仅ERS 0.74，失效率高达22%。只有33%的模型通过预部署评估。

**⚠️ 局限性**

局限性包括：仅评估了两款LLM；ECS维度确定缺乏正式因子分析；语义一致性约束采用经验相关系数；情景库规模有限；Gemini的鲁棒性可能受提示方式影响。

---

## 385. Towards Personalized Federated Learning for Dysarthric Speech Recognition

**arXiv ID:** 2606.13253 | [PDF](https://arxiv.org/pdf/2606.13253v1)

**作者:** Tao Zhong `[一作]` (Chinese University of Hong Kong), Xunying Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了两种基于相似度的聚合策略（参数平均和嵌入平均），实现了针对癫痫语者的个性化联邦语音识别模型。

**💡 创新点**

创新点在于将说话者相似度（基于模型参数或嵌入）融入到联邦学习的聚合过程，从而在保持隐私的同时提升个体化识别效果。

**🔧 技术方法**

使用了联邦学习框架、HuBERT预训练模型、CTC损失、Cosine相似度计算、以及隐私增强的随机子采样与嵌入平均。

**📊 数据集**

在UASpeech和TORGO这两个公开的癫痫语者语音识别数据集上进行实验。

**📈 对比分析**

与基线FedAvg（含正则化）相比，参数平均和嵌入平均分别在UASpeech上实现了最高0.94%/0.99%的绝对WER下降，在TORGO上实现了0.52%/0.56%的下降，结合两种策略后进一步提升，整体WERS显著优于对照组。

**⚠️ 局限性**

局限在于仅针对少数癫痫语者数据，实验规模有限；聚合策略对β权重敏感，需针对不同数据集调参；未深入评估跨方言或更大规模数据下的可扩展性。

---

## 386. Distributional Loss for Robust Classification

**arXiv ID:** 2606.13223 | [PDF](https://arxiv.org/pdf/2606.13223v1)

**作者:** Kathleen Anderson `[一作]` (University of Lübeck), Thomas Martinetz `[通讯]` (University of Lübeck)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于双峰高斯分布的分布式损失函数，用以代替传统交叉熵进行监督分类训练

**💡 创新点**

创新点在于把样本标签视为整个输出分布的目标而非单一硬标签，并在不增加标签信息的前提下通过参数化和非参数化KL估计实现分布对齐，显著缓解过拟合并提升鲁棒性

**🔧 技术方法**

使用了参数化的变分自编码器（VAE）框架来生成均值方差参数，采用KL散度的闭式解；以及非参数的最近邻估计和核密度估计（KDE）来逼近输出分布；损失由两项组合而成，控制权重λ

**📊 数据集**

在Cifar10、Caltech256、BloodMNIST、PathMNIST四个图像数据集上进行实验；此外还用随机噪声数据集测试过拟合行为

**📈 对比分析**

与Softmax交叉熵、PolyLoss、T-vMF Loss等基线对比，分布式损失在小样本和低质量数据上显著提升准确率（最多提升17.2%），在Caltech256上表现略逊，整体鲁棒性更好

**⚠️ 局限性**

对高维（如256类）任务适应性不佳；需要额外训练时间和对超参数（如m、h、σ、λ）的细致调优；在大规模数据集时分布对齐效果有限

---

## 387. Proprioceptive-visual correspondence enables self-other distinction in humanoid robots

**arXiv ID:** 2606.13222 | [PDF](https://arxiv.org/pdf/2606.13222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 388. Mitigating business risks from renewable PPA power sourcing uncertainties for European green hydrogen production: Robust system design, regulatory adjustments and offtake flexibility

**arXiv ID:** 2606.13215 | [PDF](https://arxiv.org/pdf/2606.13215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 389. Towards More General Control of Diffusion Models Using Jeffrey Guidance

**arXiv ID:** 2606.13240 | [PDF](https://arxiv.org/pdf/2606.13240v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 390. Decoding Insect Song: A Multitask Semisupervised Orthoptera Bioacoustic Classifier

**arXiv ID:** 2606.13236 | [PDF](https://arxiv.org/pdf/2606.13236v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 391. Layer-Resolved Optimal Transport for Hallucination Detection in NMT and Abstractive Summarization

**arXiv ID:** 2606.13216 | [PDF](https://arxiv.org/pdf/2606.13216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 392. CoDeR: Local Constraint-Compatible Retrieval Beyond Semantic Similarity

**arXiv ID:** 2606.13204 | [PDF](https://arxiv.org/pdf/2606.13204v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 393. From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation

**arXiv ID:** 2606.13221 | [PDF](https://arxiv.org/pdf/2606.13221v1)

**作者:** Bora Kargi `[一作]` (ELLIS Institute), David Salinas `[通讯]` (ELLIS Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将LLM判定者的分数差映射为校准后的胜率，并在Bradley‑Terry模型中使用软标签来估计模型的Elo评分。

**💡 创新点**

创新点在于不改变BT模型，而是将判定者的分数差转为概率以校准局部不确定性，并结合分割式合适预测得到全局无分布覆盖区间。

**🔧 技术方法**

使用的方法包括Bradley‑Terry排名模型、软目标（sigmoid映射分数差）、最大似然温度校准以及分割式合适预测的置信区间。

**📊 数据集**

使用的数据集为LMArena 100K的多语言问答对，包含约55个模型的对战数据。

**📈 对比分析**

与传统硬标签Elo（Hard‑Elo）比较，Soft‑Elo在held‑out模型上人类Elo MAE下降约40–70%，并在保持90%覆盖率的前提下将置信区间宽度压缩约40–70%。

**⚠️ 局限性**

局限性在于仅利用判定者分数差的置信度，未建模知识不确定性、幻觉或提示歧义；合适预测的覆盖保证是边际的，需满足校准集与新模型的可交换性。

---

## 394. Hallucination in Medical Imaging AI: A Cross-Modality Analytical Framework for Taxonomy, Detection, and Mitigation under Regulatory Constraints

**arXiv ID:** 2606.13211 | [PDF](https://arxiv.org/pdf/2606.13211v1)

**作者:** Omar Alshahrani `[一作]` (King Fahd University of Petroleum & Minerals), Muzammil Behzad `[通讯]` (King Fahd University of Petroleum & Minerals)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性综述并构建跨模态医学影像AI幻觉的统一分类框架，比较通用与专业基础模型在幻觉率上的差异，并评估检测与缓解策略与FDA监管框架的契合度。

**💡 创新点**

① 将Brooks‑Anastasio、DREAM和MediHall三种分类法融合为单一跨模态体系；② 通过幻觉专用基准发现通用基础模型幻觉率显著低于专业模型，挑战现有假设；③ 以结构化叙事方式综合检测/缓解方法，并映射至FDA TPLC/PCCP，提出多层级监管兼容性评估。

**🔧 技术方法**

采用结构化叙事综述、复合基准评分（两轴X/Y）、多维检测技术（不确定性量化、注意力可视化、sFRC重建评估、跨模态验证、基准评估框架）以及多层缓解手段（架构约束、Chain‑of‑Thought提示、检索增强生成、人工介入）。

**📊 数据集**

Med‑HallMark、MedHallBench、CXR‑VisHal、MIMIC‑CXR‑VQA、OmniMedVQA、Clinical Diagnostic Accuracy、FACTS Grounding、HealthBench Hard、Med‑HALT、sFRC工具包等多模态基准数据集与临床实验数据。

**📈 对比分析**

构建复合X轴（General Visual Hallucination Resistance）与Y轴（Medical Imaging Performance）评分，对14种VLM进行可视化比较；通用模型幻觉自由率中位数为76.6% vs 51.3%专业模型（p=0.012）；CoT提示可将幻觉降低86.4%；HITL降低AI FP约83.7%；PI‑MoCoNet架构在MRI重建中提升≈1 dB PSNR。

**⚠️ 局限性**

指标定义不统一导致难以直接量化；大多数评估来自基准实验，缺乏真实临床部署验证；对儿童、罕见病等数据缺乏，影响模型普适性。

---

## 395. Understanding helpfulness and harmless tension in reward models

**arXiv ID:** 2606.13209 | [PDF](https://arxiv.org/pdf/2606.13209v1)

**作者:** Eshaan Tanwar `[一作]` (University of Copenhagen), Pepa Atanasova `[通讯]` (University of Copenhagen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究奖励模型在“有用性”（helpfulness）和“无害性”（harmlessness）两个对立目标下的行为张力，比较单目标与混合目标训练的效果，并通过神经元激活分析、定向消融等方法揭示内部机制。

**💡 创新点**

创新点在于：①首次系统性识别并量化与各对立目标相关的神经元集合；②发现这些神经元在高层中高度重叠，且共享神经元对模型行为产生冲突；③通过消融实验证明共享神经元在混合目标训练中导致性能退化；④提出“行为保留率”指标评估混合训练下单目标知识的保留情况。

**🔧 技术方法**

主要技术包括：RLHF奖励模型训练（使用Bradley–Terry损失）、基于激活幅度的神经元重要性评分、目标相关神经元的定向消融、层级分布分析、行为保留率计算、奖励分数分布对比以及多种评测数据集的交叉验证。

**📊 数据集**

使用的数据集：HH‑RLHF（划分为有用性与无害性子集）用于训练奖励模型；RewardBench、RewardBench2、RM‑Bench用于多任务评估；此外在实验中还采用了SmoLLM2与GPT‑2系列模型的不同变体作为基础。

**📈 对比分析**

比较方法：在帮助性与无害性两类任务上分别评估三种RM（有用性单目标、无害性单目标、混合目标），计算准确率、保留率以及奖励分数分布。实验结果表明：单目标RM在各自目标任务上表现最佳；混合目标RM往往低于单目标，行为保留率仅约60%；共享神经元消融导致有用性任务下降约10%，无害性任务下降约13%；奖励分数在混合目标下更集中、置信度下降。

**⚠️ 局限性**

局限性：①仅针对传统RLHF奖励模型，未涵盖DPO、RLOO等新训练方法；②使用完整微调而非参数高效微调，规模有限；③未深入探讨共享神经元的多义性与可解释性；④未对更细粒度的对立目标进行实验；⑤实验受计算资源限制，未覆盖更大模型。

---

## 396. To GAN or Not To GAN: Segmentation Analysis on Mars DEM

**arXiv ID:** 2606.13252 | [PDF](https://arxiv.org/pdf/2606.13252v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 397. A Minimal Model of Bounded Trade-Off Screening in Multi-Attribute Choice

**arXiv ID:** 2606.13201 | [PDF](https://arxiv.org/pdf/2606.13201v1)

**作者:** Manisha Dubey `[一作]` (University of Edinburgh), Subramanian Ramamoorthy `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于最大-最小评估的有界权衡筛选框架，用以解释多属性决策中对极端得失的非完全补偿性拒绝。

**💡 创新点**

创新点在于引入可调节的权衡容忍度参数 M，实现对收益与最差损失的动态平衡评估；并将 M 视为上下文相关的潜在变量，揭示不同环境下的偏好逆转。

**🔧 技术方法**

使用最大-最小比较规则（M‑dominance）和概率化的决策函数，并通过最大似然估计恢复 M；实验中对模拟的 3 维属性数据进行对比分析。

**📊 数据集**

使用随机生成的 5000 对 3 属性选项的模拟数据；并在两种不同环境下模拟 Agent 以检验上下文敏感性。

**📈 对比分析**

与加权加法效用与 Chebyshev 效用比较：M‑dominance 在 86.5% 与加权效用、74.8% 与 Chebyshev 产生不同的偏好；在识别实验中 M 的恢复误差仅 0.038，相关系数 0.999；上下文依赖模型相较全局模型提升对数似然 175.65，捕捉到 21.3% 的偏好逆转。

**⚠️ 局限性**

局限在于仅关注最大收益与最差损失，忽略中间属性的潜在影响；实验全部基于模拟数据，尚未在真实决策场景中验证；模型仅为成对比较，未涵盖多选项情境。

---

## 398. MOSAIC: Modality-Specific Adaptation for Incremental Continual Learning in Parkinson's Disease Gait Assessment

**arXiv ID:** 2606.13258 | [PDF](https://arxiv.org/pdf/2606.13258v1)

**作者:** Minlin Zeng `[一作]` (Nanyang Technological University), Zhiqi Shen `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种名为 MOSAIC 的模态增量持续学习框架，用于在多传感器环境下实现帕金森病步态评估。

**💡 创新点**

创新点包括：① 识别并解决了“毒教师”现象；② 设计了模态专属预热（Modality‑Specific Warm‑Up）以在蒸馏前稳定新模态表示；③ 引入统计解耦的 MSBN 结构，隔离模态统计量；④ 通过课程化的惩罚性（repulsive）目标实现可塑性恢复。

**🔧 技术方法**

技术方法涵盖：统计解耦的 MSBN、模态专属批归一化、弹性权重整合 (EWC)、知识蒸馏、课程化的 repulsive 损失以及三阶段优化策略。

**📊 数据集**

使用了三大公开帕金森病步态数据集：WearGait（步态垫、脚垫、IMU）、FBG（骨架、关节角度、地面反作用力）和 FOG（骨架、陀螺仪、加速度）。

**📈 对比分析**

与 Fine‑Tuning、EWC、LwF、LwI、DRMN、Harmony、MedCoSS 以及 TRIP 等基线进行对比；在所有模态到达顺序下，MOSAIC 在 F1、BWT、NAA 上均取得最优或接近上限的表现，显著降低遗忘并提升最终性能。

**⚠️ 局限性**

局限性包括：repulsive 损失使用固定的边距，未自适应不同模态间的相似性；实验仅在三种数据集上验证，未探讨更广泛的传感器组合或跨疾病场景；对硬件资源或实时性等实际部署细节关注不足。

---

## 399. Extracting Governing Equations from Latent Dynamics via Multi-View Contrastive Learning

**arXiv ID:** 2606.13260 | [PDF](https://arxiv.org/pdf/2606.13260v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 400. Embedding-based Methods for Linear Solver Performance Prediction

**arXiv ID:** 2606.13255 | [PDF](https://arxiv.org/pdf/2606.13255v1)

**作者:** Hayden Liu Weng `[一作]` (Technical University of Munich), Felix Dietrich `[通讯]` (Technical University of Munich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于嵌入的模块化框架，用于预测稀疏线性系统的最佳求解器配置。

**💡 创新点**

创新点在于将性能空间嵌入与特征投影分离，支持多标签预测并使用相对性能指标（MAPE、nDCG），同时仅依赖低成本特征。

**🔧 技术方法**

采用Boruta特征筛选、word2vec/Glove嵌入、线性投影、k近邻预测等技术。

**📊 数据集**

使用SuiteSparse集合621个矩阵和PETSc 101种求解器配置进行实验。

**📈 对比分析**

与传统kNN、随机森林、单一最佳求解器以及PETSc默认配置对比，嵌入模型在MAPE、1‑error、nDCG等指标上均明显优于基线，尤其在低成本特征下仍保持竞争力。

**⚠️ 局限性**

局限包括对稀疏矩阵样本覆盖不足时性能下降、投影仅为线性可能无法捕捉非线性关系，以及缺乏硬件感知和参数细调。

---

## 401. EPIG: Emotion-Based Prompting for Personalised Image Generation

**arXiv ID:** 2606.13247 | [PDF](https://arxiv.org/pdf/2606.13247v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 402. Clipping Makes Distributed and Federated Asynchronous SGD Robust to Stragglers

**arXiv ID:** 2606.13287 | [PDF](https://arxiv.org/pdf/2606.13287v1)

**作者:** Samuel Erickson `[一作]` (KTH Royal Institute of Technology), Mikael Johansson `[通讯]` (KTH Royal Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨并证明梯度裁剪能消除异步 SGD 对最大延迟的依赖，使异步训练对慢速工作节点鲁棒。

**💡 创新点**

创新点在于给出子 Weibull 噪声模型下的期望与高概率收敛保证，证明在均质和异质数据情形下梯度裁剪可去除最大延迟影响，并首次提供异步优化的高概率收敛结果。

**🔧 技术方法**

采用扰动迭代分析、Freedman不等式、子 Weibull 噪声假设以及梯度裁剪操作，并结合异步并行调度策略。

**📊 数据集**

实验使用 CIFAR‑10（ResNet‑18、CNN）与 Shakespeare（LSTM）数据集，并在标签倾斜的 CIFAR‑10 上验证。

**📈 对比分析**

与 Vanilla ASGD、Delay‑adaptive ASGD 以及 Ringleader ASGD 进行对比，梯度裁剪在所有延迟设定下显著缩短壁钟时间，最快可比基线快 1.8–2.2 倍。

**⚠️ 局限性**

局限性包括假设梯度噪声为子 Weibull、异质设置下采样方式导致计算时间受限，以及未考虑多机网络延迟、隐私约束等实际部署场景。

---

## 403. ComAct: Reframing Professional Software Manipulation via COM-as-Action Paradigm

**arXiv ID:** 2606.13239 | [PDF](https://arxiv.org/pdf/2606.13239v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 404. Different Layers, Different Manifolds: Module-Wise Weight-Space Geometry in Transformer Optimization

**arXiv ID:** 2606.13276 | [PDF](https://arxiv.org/pdf/2606.13276v1)

**作者:** Kirato Yoshihara `[一作]` `[通讯]` (University of Osaka), Kirato Yoshihara (University of Osaka)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在Transformer预训练中，按模块分配不同的矩阵流形约束（Stiefel和DGram）对优化的影响。

**💡 创新点**

创新点在于首次发现注意力层与MLP层对权重空间几何的需求不同，且提出了模块化流形约束的实践方案。

**🔧 技术方法**

使用了Manifold Muon优化器，对权重矩阵施加Stiefel或DGram约束，并跟踪奇异值和softmax饱和情况。

**📊 数据集**

实验基于GPT‑2小模型在OpenWebText数据集上的预训练。

**📈 对比分析**

通过比较五种约束分配（无约束、全Stiefel、全DGram、Stiefel+ DGram、DGram+ Stiefel），发现Stiefel约束的注意力层与DGram约束的MLP层组合在验证集上达到最低损失（3.3544）且稳定，优于无约束（3.3855）和全Stiefel（3.3679）。

**⚠️ 局限性**

局限性包括仅在GPT‑2小规模和共享超参下验证，未探索不同学习率或权重衰减对DGram注意力的稳定性影响，且实验规模有限，缺乏自动化流形分配方法。

---

## 405. Evaluating Pluralism in LLMs through Latent Perspectives

**arXiv ID:** 2606.13254 | [PDF](https://arxiv.org/pdf/2606.13254v1)

**作者:** Laura Majer `[一作]` (University of Zagreb), Martin Tutek `[通讯]` (University of Zagreb)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两层无监督框架，先通过句子嵌入聚类提取文本中的“aspect”，再聚合成“perspective”，并应用于Goodreads书评数据集，评估LLM生成文本的多样性与人类文本的差距。

**💡 创新点**

创新点在于将视角抽象为由多层级aspect构成的潜在分布，采用无标签深度学习与聚类方法实现对自由文本中主观视角的细粒度识别与比较，填补了此前仅用高层标签或单一多选题评估多样性的空白。

**🔧 技术方法**

使用F2LLM 0.6B句子嵌入、BERTopic+HDBSCAN进行aspect聚类，随后对perspective向量采用k‑means/社区检测，计算覆盖率、Jensen‑Shannon散度、熵等统计量进行定量评估，并通过LLM-as-a-judge进行聚类一致性验证。

**📊 数据集**

Goodreads公开书评数据集（约1500条/书）作为人类文本基准，生成LLM书评（OpenAI GPT‑4.1、Google Gemini 2.5、Meta Llama‑3.1 8B、Open‑AI OLMo‑2 1B/7B）在三种提示方式（基线、T=1.5、高温度、persona）下进行对比。

**📈 对比分析**

比较方法包括：aspect层的覆盖率（Overton pluralism）与Jensen‑Shannon散度（distributional pluralism），perspective层的覆盖率、熵、视角多样性等；实验结果显示persona提示能显著提升覆盖率并降低散度（如GPT‑4.1 persona覆盖率≈98%，JSD≈0.11），但所有模型在perspective层仍表现出高度同质化（覆盖率≈35‑55% vs 人类≈97%）。

**⚠️ 局限性**

局限性包括：仅评估单一主题（书评）和数据来源，聚类结果受嵌入质量和参数设置影响，无法直接解释视角标签；无监督方法在新领域的迁移性未知；计算成本较高，且对少数派视角的捕捉仍不足。

---

## 406. WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning

**arXiv ID:** 2606.13232 | [PDF](https://arxiv.org/pdf/2606.13232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 407. PolyAlign: Conditional Human-Distribution Alignment

**arXiv ID:** 2606.13227 | [PDF](https://arxiv.org/pdf/2606.13227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 408. When Similar Means Different: Evaluating LLMs on Arabic--Hebrew Cognates

**arXiv ID:** 2606.13218 | [PDF](https://arxiv.org/pdf/2606.13218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 409. A Polynomial-Decay and Pinhole-Imaging Whale Optimization Algorithm for UAV Relay Communication Deployment

**arXiv ID:** 2606.13208 | [PDF](https://arxiv.org/pdf/2606.13208v1)

**作者:** Zhenhong Peng `[一作]` (Zhongkai University of Agriculture and Engineering), Yapeng Wang `[通讯]` (Macao Polytechnic University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的鲸鱼优化算法（PWOA），用于无人机（UAV）中继通信部署的多目标、受限的非凸优化问题。

**💡 创新点**

创新点包括：1）Good Nodes Set（GNS）初始化，使搜索空间均匀填充；2）多项式非线性衰减的收敛因子，使早期探索更充分，后期快速收敛；3）基于停滞触发的凹形成像对抗学习（POBL）配合精英高斯局部搜索，用于跳出局部最优并细化领导者。

**🔧 技术方法**

采用鲸鱼优化算法框架，并在其基础上引入上述三种改进模块；使用路径损耗模型、Shannon 容量与服务成本公式构造目标函数，并用二次罚函数处理约束。

**📊 数据集**

数据集为一个自定义的5维、5约束的UAV中继部署问题（横向位置、垂直高度、发射功率、带宽等），无公开数据集，全部为实验生成。

**📈 对比分析**

与原始Woa、SCA（Sine Cosine Algorithm）和改进PSO（IPSO）在同一问题上进行30次独立跑（N=30，T=500）比较。PWOA在Best、Worst、Mean、Std四项指标上均优于三者，平均误差下降至1.4–18.5%，标准差下降至63–87%，且收敛速度最快。

**⚠️ 局限性**

局限性：仅针对单一UAV静态部署；缺乏多UAV协同、多用户移动、能耗预算等更复杂场景；未在实时在线环境或动态信道统计下验证性能；对高维大规模问题的可扩展性尚待进一步评估。

---

## 410. From Verdict to Process: Agentic Reinforcement Learning for Multi-Stage Fact Verification

**arXiv ID:** 2606.13262 | [PDF](https://arxiv.org/pdf/2606.13262v1)

**作者:** Rongxin Yang `[一作]` (Sun Yat-sen University), Chao Yu `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为 ProFact 的基于强化学习的多阶段事实核查框架，能够端到端优化命题拆解、证据检索、答案生成和最终判决。

**💡 创新点**

创新点在于将事实核查视为有限时限的马尔可夫决策过程，并通过过程感知奖励为中间步骤提供稠密反馈，实现了轨迹级优化。

**🔧 技术方法**

采用统一策略的强化学习（GRPO）、过程感知奖励、语义检索工具调用以及大型语言模型（Qwen 系列）。

**📊 数据集**

使用 AVeriTeC 公开基准数据集，该数据集包含开放域命题、标签和标注的问答证据。

**📈 对比分析**

与 Consistency、InFact、HerO 等基线比较，ProFact 在 Accuracy、AVeriTeC Score 以及 Q-only METEOR 上均优于所有基线，同时在推理速度和 token 消耗上更高效。

**⚠️ 局限性**

限制包括对过程奖励的依赖、对单一 evidence store 的限制以及在更大模型规模下性能可能出现倒退的逆扩展问题。

---

## 411. Q-Backbone: A Quantum-Enhanced Control Plane for Future Communication Networks

**arXiv ID:** 2606.13248 | [PDF](https://arxiv.org/pdf/2606.13248v1)

**作者:** Mahdi Chehimi `[一作]` (American University of Beirut), Gan Zheng `[通讯]` (University of Warwick)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为Q‑Backbone的量子增强通信网络控制平面，结合经典与量子计算资源实现网络决策加速；

**💡 创新点**

核心创新包括：①四层分层架构将量子处理单元与传统CPU/GPU并列；②量子调用策略（QIP）根据任务特征、运营上下文与硬件状态动态决定是否使用量子加速；③在分布式量子执行模式下引入射击分配与LOCC优先级管理的联合调度；

**🔧 技术方法**

技术手段包括：量子近似优化算法（QAOA）、变分量子求解器（VQE）、量子门切割与射击重建、混合量子‑经典运行时、分布式量子计算（LOCC）以及基于模拟退火的调度元启发式；

**📊 数据集**

使用合成的基线网络工作负载（多节点QPU、射击数与深度约束）以及典型的控制循环截止时间参数；

**📈 对比分析**

与六种基线（贪心、列表、随机、无射击分配、无LOCC感知、无两者）对比，QB调度器在负载升高时完成截止时间内任务的比例提升约25%；

**⚠️ 局限性**

局限性包括：①实验规模仅为5个异构QPU，未验证大规模部署；②依赖于精细的经典预处理与问题映射；③量子硬件噪声与冷却成本未充分评估；④缺乏标准化接口与能耗量化指标；

---

## 412. A $q$-analogue of the rational normal curve and linearized Reed-Solomon codes

**arXiv ID:** 2606.13246 | [PDF](https://arxiv.org/pdf/2606.13246v1)

**作者:** Valentina Astore `[一作]` (Inria), Flavio Salizzoni `[通讯]` (MPI Leipzig)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了线性化的Reed-Solomon码在和秩度量下的几何特性，提出了一种几何框架来描述这些码，并通过考虑有理正常曲线的q类比，揭示了其结构和几何性质。

**💡 创新点**

创新点在于将Schur乘积技术扩展到和秩度量的情况，并提供了线性化Reed-Solomon码的几何特征描述，揭示了这些码在特定参数选择下满足意外的(q+1)度超曲面条件。

**🔧 技术方法**

使用了几何框架和Schur乘积技术，结合Hilbert函数的研究，分析了与线性化Reed-Solomon码相关的坐标环的行为。

**📊 数据集**

研究中涉及的具体数据集未明确提及，但提到的线性化Reed-Solomon码和Gabidulin码的结构特性是研究的重点。

**📈 对比分析**

通过与现有的Hamming和秩度量下的技术进行比较，本文的方法在性能上提供了新的视角，尤其是在区分结构化码和随机码的能力上。

**⚠️ 局限性**

限制在于目前对线性化Reed-Solomon码的结构和几何性质的理解仍不完全，现有的区分器仍然有限，且对某些参数选择的几何特征描述尚不充分。

---

## 413. Visual Place Recognition in Forests with Depth-Aware Distillation

**arXiv ID:** 2606.13206 | [PDF](https://arxiv.org/pdf/2606.13206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. HYDRA-X: Native Unified Multimodal Models with Holistic Visual Tokenizers

**arXiv ID:** 2606.13289 | [PDF](https://arxiv.org/pdf/2606.13289v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 415. Brick: Spatial Capability Routing for the Mixture-of-Models (MoM) Paradigm

**arXiv ID:** 2606.13241 | [PDF](https://arxiv.org/pdf/2606.13241v1)

**作者:** Francesco Massa `[一作]` (Regolo AI), Marco Cristofanilli `[通讯]` (Seeweb)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `afceb026-1760-41ae-8d86-010831a37d97` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Brick，一种基于六维能力空间与成本加权的全模型路由器，支持 Mixture‑of‑Models（MoM）部署；

**💡 创新点**

创新点在于将模型能力映射为概率分布并与查询难度相结合，使用非对称残差与可调优的 preference knob 实现可控的质量‑成本权衡；

**🔧 技术方法**

核心技术包括 ModernBERT 能力分类器、难度估计器、基于欧氏距离的能力匹配与成本惩罚的路由数学、以及基于经验校准的权值矩阵；

**📊 数据集**

使用 Brick2 Dataset A（5,504 条查询，覆盖六大能力维度）进行评估；

**📈 对比分析**

与单模型基线、RouteLLM、FrugalGPT、Cascade Routing 等方法对比，Brick 在最高质量配置下达到 76.98% 的准确率，成本仅为单模型 28% 的支出，且在低成本配置下仍保持较高准确率；

**⚠️ 局限性**

局限性包括未能完全挖掘三模型 oracle 的 6.27pp 余量，技能矩阵对稀疏模型的适应性有限，且评估主要在领域内校准的基准上，跨域泛化仍待验证。

---

## 416. LLM-as-an-Investigator: Evidence-First Reasoning for Robust Interactive Problem Diagnosis

**arXiv ID:** 2606.13220 | [PDF](https://arxiv.org/pdf/2606.13220v1)

**作者:** Fabrizio Marozzo `[一作]` (University of Calabria), Pietro Liò `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种以证据为先的交互式诊断框架——Solution Investigator Agent，用于在技术问题解决中主动生成假设、提问澄清、更新置信度，并在证据充分时给出最终答案。

**💡 创新点**

核心创新在于将LLM的推理与外部控制循环分离：把用户的假设视为可检验的候选，而不是先验前提；通过自动化的假设空间构建、定向提问和概率更新，实现对“用户驱动的奉承”现象的自我纠正。

**🔧 技术方法**

使用多种大语言模型（OpenAI GPT-4、Google Gemini）作为推理核心，配合JSON结构化输出、概率归一化、冲突检测与控制逻辑；采用Chain-of-Thought、Tree-of-Thought等思维链提示作为内部推理手段。

**📊 数据集**

构建并公开了由已解决的电气、水力和机械技术论坛帖子组成的 303 条案例数据集（共 8,930 条评论），包含初始问题、讨论过程和最终答案。

**📈 对比分析**

与直接回答模型、思维提示模型进行三路对比；在所有域和模型上，Investigator Agent 的诊断分数（最高候选覆盖）平均提升约 27 分（从 33.07 提升到 60.86），即在 0–100 评分尺度上显著高于基线，尤其在处理误导性假设时表现出显著鲁棒性。

**⚠️ 局限性**

局限性包括：仍需人工评判最终答案的准确性，缺乏自适应停止机制；在极端模糊或信息极少的场景下可能问答循环过长；实验仅覆盖三大领域，未验证跨领域通用性；用户研究样本规模有限。

---

## 417. Embedding ISO 10218 Safety Compliance in Robots via Control Barrier Functions for Human-Robot Collaboration

**arXiv ID:** 2606.13203 | [PDF](https://arxiv.org/pdf/2606.13203v1)

**作者:** Federico Parma `[一作]` (Polytechnic of Bari), Manuel Beschi `[通讯]` (University of Brescia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在机器人控制层引入基于控制障碍函数（CBF）的安全约束，实现了ISO 10218标准下的速度与距离监控（SSM）功能，并在实际人机协作场景中验证了其可行性。

**💡 创新点**

创新点在于：①用人类加速度信息预测最坏情况下的最小分离距离，突破传统即时距离估计的保守性；②将该预测性CBF嵌入到任务缩放的SQP控制器中，并加入空间管束（tube constraint），实现高速、高精度的动态避障；③提出两种实现方式（PD安全滤波器和任务缩放SQP）并系统对比。

**🔧 技术方法**

使用的技术包括：控制障碍函数（CBF）理论、序列二次规划（SQP）优化、任务缩放（trajectory scaling）技术、实时人体骨骼追踪（Stereolabs ZED 2）、雷达验证（Inxpect X‑300）以及Python/Numba/quadprog实现的高频（500 Hz）控制。

**📊 数据集**

实验数据集来源于：在UR10e机器人上采集的真实人类工人骨骼关键点序列（COCO‑18 Skeleton），并使用雷达对关键点位置进行双传感器验证；实验在仿真与真实机器人两种环境下进行，持续 15 000 s 的长时运行数据。

**📈 对比分析**

与工业基线外部 SSM 模块对比，Method II 在完成循环次数、轨迹点达成率、平均缩放因子等指标上均优于 Method I 与基线。具体表现为：完成 789 次循环（相较基线 41 次）、92.9% 的轨迹点到达率（相较 Method I 62.9%），平均缩放因子 0.81（相较基线 0.15），且平均轨迹误差比 Method I 减少 63%。

**⚠️ 局限性**

主要局限性包括：①假设人类加速度恒定，需要保守安全缓冲 C 来吸收预测误差；②控制器使用静态权重，导致路径精度与安全反应之间的折衷；未来工作计划引入概率运动预测和自适应权重策略以进一步提升效率和安全性。

---

## 418. Humor Style Drives Laughter, Topic Shapes Acceptability: Evaluating Bilingual Personal and Political Robot-Delivered AI Jokes

**arXiv ID:** 2606.13256 | [PDF](https://arxiv.org/pdf/2606.13256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 419. Error Probability Analysis of Quantum Communication with Phase-squeezed M-PSK

**arXiv ID:** 2606.13286 | [PDF](https://arxiv.org/pdf/2606.13286v1)

**作者:** Nikos A. Mitsiou `[一作]` (University of Cyprus), Ioannis Krikidis `[通讯]` (University of Cyprus)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在自适应Mark-II相位测量下使用相位压缩的M‑PSK在固定总光子预算条件下的符号误码率。

**💡 创新点**

提出了符号相关的相位压缩构造，并给出POM、极坐标和Owen T函数三种近似分析，展示压缩对高阶M‑PSK误码率和光子效率的显著提升。

**🔧 技术方法**

利用量子相位POM、Fock基、极坐标统计、Mark‑II测量矩阵以及Owen T函数闭式表达。

**📊 数据集**

未使用实验数据集，而是基于理论推导和 Monte‑Carlo 仿真进行评估。

**📈 对比分析**

与传统相干M‑PSK以及全量子Mark‑II POM 基准比较，压缩方案在 M≥8 时可将误码率降低数倍、光子效率提升约 20%–66%。

**⚠️ 局限性**

近似模型在极端压缩或低光子数下误差可达数光子，且对 Mark‑II 测量误差的高斯假设在某些参数区间不完全成立。

---

## 420. Navigating the Safety-Fidelity Trade-off: Massive-Variate Time Series Forecasting for Power Systems via Probabilistic Scenarios

**arXiv ID:** 2606.13338 | [PDF](https://arxiv.org/pdf/2606.13338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 421. Split Tallies: A Discrete Certificate Calculus for Auditing Dynamic Ordered Sets in Constant Memory

**arXiv ID:** 2606.13272 | [PDF](https://arxiv.org/pdf/2606.13272v1)

**作者:** Faruk Alpay `[一作]` (Bahçeşehir University), Levent Sarioglu `[通讯]` (Bahçeşehir University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种常数空间的审计协议，用于在动态有序集合上后置验证所有操作的正确性。

**💡 创新点**

创新点包括：①基于间隙（gap）演算与离散正常形的组合，实现了无密码学假设下的常数空间审计；②利用单个秘密字段元素的指纹化，得到错误概率约为(4T+1)/p；③证明了随机性、秘密性与时间戳规则的不可或缺性，并给出了下界。

**🔧 技术方法**

核心技术包括：间隙集合的区间图单位路径表示、离散正常形、基于字段的指纹累加器、(2,4)-树维护与潜能函数分析、以及一系列基于单词模型的规则化短记。

**📊 数据集**

本文为理论分析，未使用实际数据集，仅在示例中用 U=6、T=6 的小规模实例演示协议流程。

**📈 对比分析**

与传统内存检查和认证数据结构方法相比，本文实现了每次操作 O(log n) 时间、O(1) 追加词语的公共账本，并在一次线性扫描中完成审计，错误概率可控制在 2⁻³⁸ 左右；虽然没有实验评测，但理论上可满足百万级操作的安全性。

**⚠️ 局限性**

局限性包括：只能做后置审计（非实时拒绝错误），仅适用于七种基础操作的有序集合接口，无法直接支持秩查询等位置统计；审计过程依赖单个秘密字段元素，一旦该秘密泄露或破解，安全性将受损。

---

## 422. Dynamic Resource Management in Production HPC Clusters

**arXiv ID:** 2606.13266 | [PDF](https://arxiv.org/pdf/2606.13266v1)

**作者:** Petter Sandås `[一作]` (Barcelona Supercomputing Center), Antonio J. Peña `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了一套非侵入式的 MPI 可变性方法，集成在 DMR 框架中，使得应用在标准 Slurm 环境下即可在运行时动态增删节点，支持 Checkpoint/Restart 与内存级重构；

**💡 创新点**

创新点在于：①通过用户级 Slurm API 与 PMIx/PRRTE 协同实现多作业并行扩缩容，完全不改动集群 RMS；②提供高层 API，降低应用侧改动；③在三台 TOP500 级生产机上验证，展示可在生产环境下实现动态资源管理；

**🔧 技术方法**

使用技术包括：MPI（Open MPI/MPICH）、PMIx Reference Runtime Environment (PRRTE)、Slurm C API、DMR 及其 v2 API、Checkpoint/Restart 机制；

**📊 数据集**

使用了两款大规模科学应用（具体未命名），并在三台 TOP500 超算（分别为 10+ 100+ 1000+ 节点规模）上进行部署；

**📈 对比分析**

实验通过与静态分配基线对比，测量节点小时消耗和作业完成时间；结果显示：在相同工作负载下，动态可变性方案与静态方案性能相当，且节点小时消耗明显下降；

**⚠️ 局限性**

局限性包括：①仍需应用支持可变重配置；②对某些 MPI 通信模式（如全局同步）适应性有限；③在缺少内存级重构支持的场景下需使用 Checkpoint/Restart，可能产生额外开销；④依赖 Slurm C API，无法在不支持该 API 的环境下运行；

---

## 423. DuET: Dual Expert Trajectories for Diffusion Image Editing

**arXiv ID:** 2606.13303 | [PDF](https://arxiv.org/pdf/2606.13303v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 424. IterCAD: An Iterative Multimodal Agent for Visually-Grounded CAD Generation and Editing

**arXiv ID:** 2606.13368 | [PDF](https://arxiv.org/pdf/2606.13368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 425. The $(1 + 1)$-EA in Dynamic Environments

**arXiv ID:** 2606.13360 | [PDF](https://arxiv.org/pdf/2606.13360v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 426. ReSum: Synergizing LLM Reasoning and Summarization with Reinforcement Learning

**arXiv ID:** 2606.13316 | [PDF](https://arxiv.org/pdf/2606.13316v1)

**作者:** Xucong Wang `[一作]` (University of Science and Technology of China), Xiangxiang Chu `[通讯]` (AMAP, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于强化学习的自我总结框架ReSum，用于让大型语言模型在推理过程中压缩并组织自身的推理轨迹，从而减少过度推理和上下文衰退。

**💡 创新点**

核心创新点包括：①将自我总结视为内在机制，利用模型自身决定何时生成总结；②通过对比分支（自然总结点NP与人工注入点AP）构建树形推理轨迹；③设计基于总结的组相对优势SGPO，对不同分支的奖励进行细粒度分配；④在奖励中加入总结频率惩罚/奖励，以避免过度总结。

**🔧 技术方法**

采用RLVR与GRPO框架，结合树结构分支、对比学习、总结感知优势函数（SGPO）以及自定义奖励（任务奖励+格式奖励）等技术；使用Open-R1实现、8块NVIDIA H20训练。

**📊 数据集**

主要实验数据集：MATH（训练），评估集为AIME24、AIME25、AMC23、MATH500、Minerva、Olympiad；多模态评估使用GEOQA-8K；模型基座包括Qwen2.5-Math-7B/1.5B、Qwen2.5-3B、DeepSeek-Math-7B以及Qwen2.5-3B-VL-Instruct。

**📈 对比分析**

与GRPO、DGPO、GPG、DAPO、GSPO等现有RLVR方法进行对比。ReSum在所有数学基准上平均提升4.0%，在AIME24/25、AMC23、MATH500等上均高于DGPO；在GEOQA-8K上提升5.6%。同时，平均推理长度缩短18.6%，并在不同树结构配置、预算下保持一致性。Ablation实验表明NP、AP与SGPO均为关键组件。

**⚠️ 局限性**

限制与挑战：①需要预先定义总结短语集合，可能受限于语言或领域；②树结构分支与预算T×J的超参数需要调优；③对极长上下文或极短输出的适用性尚未系统验证；④奖励设计对最终性能敏感，若奖励不充分或偏颇可能导致过度总结或忽略关键推理。

---

## 427. Who Pays the Price? Stakeholder-Centric Prompt Injection Benchmarking for Real-world Web Agents

**arXiv ID:** 2606.13385 | [PDF](https://arxiv.org/pdf/2606.13385v1)

**作者:** Zihao Wang `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 StakeBench，一种以利益相关者为中心的基准，用于评估大语言模型驱动的网页代理在面对注入式攻击时的安全性；

**💡 创新点**

创新点在于将攻击按受害方（用户、卖家、平台）和具体伤害目标细分为12种对象，并结合多维度指标（攻击成功率、任务偏离率、行为不规则率）全方位评估攻击效果；

**🔧 技术方法**

使用了两种主流网页代理架构（NanoBrowser 与 BrowserUse）以及两种前沿 LLM 后端（GPT‑5 与 Gemini‑2.5‑Flash），通过自动化脚本对264个可执行攻击案例进行多次执行；

**📊 数据集**

数据集由264个基于12种产品类别的注入式攻击实例组成，包含22个可复用攻击模板（9 DPI、13 IPI），并对比了对应的清洁基准任务；

**📈 对比分析**

与传统攻击中心基准相比，StakeBench揭示了不同架构与后端在攻击成功率、任务偏离率和行为不规则率上的显著差异，指出同一代理在不同利益相关者攻击下表现差异显著，表明单一度量不足以全面评估安全性；

**⚠️ 局限性**

局限性包括仅覆盖文本层面的间接注入攻击，对视觉注入的探究仅为小规模试验；数据集局限于电商场景，且仅测试两种代理实现，未来需扩展到更多域与更丰富的攻击方式。

---

## 428. IVIE: A Neuro-symbolic Approach to Incremental and Validated Generation of Interactive Fiction Worlds

**arXiv ID:** 2606.13348 | [PDF](https://arxiv.org/pdf/2606.13348v1)

**作者:** Micaela Vaucher `[一作]` (Universidad de la República), Luis Chiruzzo `[通讯]` (Universidad de la República)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 IVIE neuro‑symbolic 系统，通过 LLM 产生创意元素并用符号验证，自动生成完整、可玩且连贯的互动小说世界。

**💡 创新点**

创新点在于四阶段增量生成管线及每阶段符号验证，融合 LLM 的创造力与符号系统的结构一致性，解决了传统 LLM 生成的世界一致性问题。

**🔧 技术方法**

使用 LLM（Gemini 2.0 Flash / GPT‑4o‑mini）、Pydantic 结构校验、DFS 检查空间连通、RAG 记忆检索与 ChromaDB 存储等技术。

**📊 数据集**

主要数据来源为自建 IF 世界与玩家交互日志，实验数据来自 8 名评估者完成的 16 个世界（Generate 与 Inspiration 模式）。

**📈 对比分析**

与单纯 LLM 或手工编写 IF 进行对比；Generate 模式完成率 100%，玩家满意度高；Inspiration 模式主题一致性高但完成率 50%，展示验证机制对一致性与可玩性的影响。

**⚠️ 局限性**

局限包括：LLM 在目标验证和谜题逻辑上偶尔失效；RAG 可能出现上下文混淆；难度评估不足；仅测试商用模型，未覆盖开源模型；API 限制导致会话中断。

---

## 429. Improved Runtime Bound for the $(μ+ 1)$ EA on BinVal

**arXiv ID:** 2606.13344 | [PDF](https://arxiv.org/pdf/2606.13344v1)

**作者:** Joris Belder `[一作]` (ETH Zürich), Raghu Raman Ravi `[通讯]` (ETH Zürich)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究并改进了（μ+1）进化算法在 BinVal 函数上的预期优化时间，将上界从 O(μ^5 n log(n/μ^4)) 降低到 O(μ log μ · n log n)。

**💡 创新点**

创新点在于对遗传传播过程中的干扰概率进行更细致的 k‑dependent 分析，并提出了“保守”突变算子框架，从而消除了之前高阶多项式对 μ 的依赖。

**🔧 技术方法**

采用了分块（block）划分、加性漂移定理、马尔可夫链建模以及对保守突变算子的概率分析等技术。

**📊 数据集**

实验使用了 BinVal 目标函数，主要在 n=500、n=1000 的随机初始种群上进行蒙特卡罗模拟，比较了不同 μ、突变率（χ）和 0/1 突变参数 p 的情况。

**📈 对比分析**

通过与已知的下界 O(μ n + n log n) 以及之前的上界进行对比，实验结果表明优化时间大致满足 Θ(μ n + n log n)，并且对 μ 的增长表现出 μ log μ 的上界趋近性。

**⚠️ 局限性**

限制在于：对 μ 的上界仍可能存在冗余的 log μ 因子；分析仅针对 BinVal 线性函数，尚未推广到一般线性目标；实验规模有限，未检验更大 n 或交叉算子等扩展。

---

## 430. Non-Parametric Dual-Manifold Mapping via 8-Bit Bounded Transformation Matrices: Challenging FP-centric Hardware Paradigms in Low-Energy AI

**arXiv ID:** 2606.13328 | [PDF](https://arxiv.org/pdf/2606.13328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 431. Dual-Domain Equivariant Generative Adversarial Network for Multimodal CT-PET Synthesis

**arXiv ID:** 2606.13341 | [PDF](https://arxiv.org/pdf/2606.13341v1)

**作者:** Gabriel Steele `[一作]`, Alessandro Perelli `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了一种Dual-Domain Equivariant Generative Adversarial Network (DDE-GAN)，用于从CT图像合成PET图像，结合空间域和频率域的双域学习，并在生成器与判别器中嵌入旋转等变性损失，以提升几何一致性与结构保真度。

**💡 创新点**

创新点包括：①首次在多模态医学图像合成中引入双域学习，将图像域与频率域信息统一建模；②在GAN框架中加入旋转等变性约束，保证输入与输出在旋转下保持一致；③采用三阶段分层一致性训练策略，实现更稳健的双域与等变性约束。

**🔧 技术方法**

使用的技术主要有：U-Net结构的生成器与判别器；傅里叶变换实现频域处理；三阶段的损失函数（空间域、频域、等变性）以及交叉一致性约束；基于G-CNN的等变性损失；以及对HECKTOR 2022数据集的预处理与数据增强。

**📊 数据集**

使用的数据集为HECKTOR 2022多中心PET/CT数据集，包含10个中心的341个采集（约125,852张训练图像）和200个测试采集（约74,400张图像），CT与PET图像已对齐并统一到256×256像素。

**📈 对比分析**

方法与传统CycleGAN和DD-GAN（仅双域无等变性）进行对比。定量评估中，DDE-GAN取得SSIM 0.92±0.04、PSNR 28.12 dB，比DD-GAN提升约6 dB PSNR、0.12 SSIM；在视觉上，合成PET图像细节更清晰，结构更符合真实PET。

**⚠️ 局限性**

局限性：①仅针对CT→PET单向合成，未验证反向或多模态扩展；②依赖于数据对齐与预处理，若输入存在严重位移仍需改进；③模型训练时间和计算资源需求较高；④缺乏临床验证，需进一步评估在真实工作流程中的实用性。

---

## 432. Work Stealing for the 2D-Mesh Topology of Satellite Constellations in Low Earth Orbit

**arXiv ID:** 2606.13329 | [PDF](https://arxiv.org/pdf/2606.13329v1)

**作者:** Mia Reitz `[一作]` (University of Kassel), Jonas Posner `[通讯]` (Fulda University of Applied Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在低轨卫星星座的二维网格拓扑上，本文提出并评估了一种仅向相邻节点抢任务的工作窃取策略。

**💡 创新点**

创新点在于完全去除全局抢夺回退，仅在单跳邻居间进行任务窃取，并给出了其对大规模星座的时延优势分析。

**🔧 技术方法**

利用异步多任务（AMT）模型、ItoyoriFBC运行时以及MPI实现的邻居抢夺算法，并用数学模型预测时延。

**📊 数据集**

实验采用两种工作负载：递归斐波那契（平衡树）和不平衡树搜索（高度不均匀任务树）。

**📈 对比分析**

在Goethe‑NHR HPC集群上用40–640核做强缩放实验，比较全局抢夺与邻居抢夺；结果显示两者执行时间相差不超过±2.2%，并保持线性缩放。

**⚠️ 局限性**

局限性包括实验仅在统一低时延的模拟网格上进行，未验证真实高时延ISL；邻居集是静态预设，未考虑卫星移动和拓扑变化；边界节点效应和故障恢复未深入探讨。

---

## 433. Masked and Predictive Self-Supervised Foundation Models for 3D Brain MRI

**arXiv ID:** 2606.13315 | [PDF](https://arxiv.org/pdf/2606.13315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 434. MiniMax Sparse Attention

**arXiv ID:** 2606.13392 | [PDF](https://arxiv.org/pdf/2606.13392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 435. SkillCAT: Contrastive Assessment and Topology-Aware Skill Self-Evolution for LLM Agents

**arXiv ID:** 2606.13317 | [PDF](https://arxiv.org/pdf/2606.13317v1)

**作者:** Kunfeng Chen `[一作]` (Wuhan University), Bo Du `[通讯]` (Wuhan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个无训练的框架，将LLM代理的技能自进化分为三阶段——对比因果提取（CCE）、评估增强演化（AAE）以及拓扑感知任务执行（TTE），解决单轨迹偏差、无验证合并和推理时上下文过载等问题。

**💡 创新点**

创新点在于：①通过多种种子生成同一任务的成功/失败对比，精准抽取因果证据；②用源任务重放评估并筛选补丁，避免无效或有害规则进入技能库；③构建可路由拓扑，仅在推理时加载与任务相关的技能节点，减少提示长度并提升性能。

**🔧 技术方法**

采用多种种子对比学习、因果水域识别、基于四类转移得分的评估过滤、分层合并、Markdown层级解析与LLM路由器等技术。

**📊 数据集**

使用的基准数据集包括SpreadsheetBench-Verified、完整SpreadsheetBench、WikiTableQuestions（OOD）以及DocVQA（多模态）。

**📈 对比分析**

在同一模型下与无技能、人写技能、参数化技能以及Trace2Skill+Combined/ +Error 等方法对比，平均成绩提升至55.5%（Qwen3.5-35B-A3B）或84.47%（WikiTQ），比基线高25.8%或40.4%；跨模型迁移亦显著提升；单模块 ablation 进一步验证每个环节的贡献。

**⚠️ 局限性**

局限性包括：需要多轨迹采样导致计算成本升高；仍基于无训练的规则抽取，无法处理极端失败场景；对比对像的可用性受限于成功/失败样本的存在；阈值与路由预算需手工设定。

---

## 436. From Passive Generation to Investigation: A Proactive Scientific Peer Review Agent

**arXiv ID:** 2606.13349 | [PDF](https://arxiv.org/pdf/2606.13349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 437. Structuring Transparency: Developing Domain-Specific Generative AI Declaration Frameworks in Higher Education

**arXiv ID:** 2606.13389 | [PDF](https://arxiv.org/pdf/2606.13389v1)

**作者:** Nicholas Micallef `[一作]` (Swansea University), Olga Petrovska `[通讯]` (Swansea University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

设计并提出了针对计算机科学评估的两套任务特定生成式AI使用声明框架，分别对应写作任务和编码任务，并阐述其设计原则和应用。

**💡 创新点**

创新点在于将现有的GenAI使用活动分类学转化为可操作的声明结构，并引入多维度（是否使用、程度、示例提示）来取代单一复选框，实现细粒度、任务特定的透明度。

**🔧 技术方法**

采用基于Bouvier等人“GenAI使用活动分类学”的概念框架，构建了表格式的声明模板；本文不涉及机器学习或算法，仅是设计研究。

**📊 数据集**

未使用实验数据集；框架基于文献综述与已有分类学，无直接数据采集。

**📈 对比分析**

本文未进行实验比较或性能评估，提出未来评估方案，包括将声明与观察到的工作流程对比，尚未测得性能指标。

**⚠️ 局限性**

局限性包括：依赖学生自报的准确性；工具范围不清晰；难以检测违规；仅在CS情境验证，跨学科迁移尚未证实。

---

## 438. JointEdit3D: Feed-Forward 3D Scene Editing in a Unified Latent Space

**arXiv ID:** 2606.13345 | [PDF](https://arxiv.org/pdf/2606.13345v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 439. EMG-Based Adaptation of Anisotropic Virtual Fixtures for Robot-Assisted Surgical Resection and Dissection

**arXiv ID:** 2606.13340 | [PDF](https://arxiv.org/pdf/2606.13340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 440. Runtime Analysis of the $(μ+ 1)$-ES in a Homogenous Progress Model

**arXiv ID:** 2606.13323 | [PDF](https://arxiv.org/pdf/2606.13323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 441. Experimental Insights into UDP-Based Video and Control Traffic over IEEE 802.11p ITS-G5

**arXiv ID:** 2606.13390 | [PDF](https://arxiv.org/pdf/2606.13390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 442. Hölder++: Improving the Quality-Coherence Trade-off in Multimodal VAEs

**arXiv ID:** 2606.13381 | [PDF](https://arxiv.org/pdf/2606.13381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 443. MoVerse: Real-Time Video World Modeling with Panoramic Gaussian Scaffold

**arXiv ID:** 2606.13376 | [PDF](https://arxiv.org/pdf/2606.13376v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 444. OR-Action: Multi-Role Video Understanding with Fine-Grained Actions

**arXiv ID:** 2606.13332 | [PDF](https://arxiv.org/pdf/2606.13332v1)

**作者:** Felix Tristram `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一个针对手术室外部视角的多角色细粒度动作识别基准 OR-Action，并提出了基于视频基础模型的视觉时序模型以及多视角到单视角特征对齐方法。

**💡 创新点**

①通过规则映射将场景图转化为细粒度动作标注，构建第一个外部手术室动作基准；②提出只依赖视觉的时序模型，在无场景图的情况下实现高性能；③提出跨视角特征对齐策略，使单视角即可预测多角色动作。

**🔧 技术方法**

规则映射、图神经网络基线、VJEPA2 视频基础模型、注意力池化（Role Pooler）、跨角色融合、多任务交叉熵、跨视角对齐损失。

**📊 数据集**

EgoExOR（公开的手术室场景图数据集）以及其生成的 OR-Action 标注。

**📈 对比分析**

与基于场景图的 GNN+规则映射、EgoExOR+GNN 等方法对比；视觉模型在测试集上帧准确率约67.5%，Edit Score 45.5%，F1@0.5 29.1，明显优于基线且接近 GT+GNN 上的 69.5%/48.6%/29.9%。

**⚠️ 局限性**

对视觉相似动作区分不足；依赖丰富的 egocentric 视角；对长尾或极少见动作的识别仍不稳定；规则映射对预测场景图敏感。

---

## 445. Rarity-Gated Context Conditioning for Offline Imitation Learning-Based Maritime Anomaly Detection

**arXiv ID:** 2606.13311 | [PDF](https://arxiv.org/pdf/2606.13311v1)

**作者:** Yongmin Kim `[一作]` (Ulsan National Institute of Science and Technology), Sungil Kim `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在上下文分布不平衡情况下的异常检测，提出罕见度门控特征线性调制（RGFiLM）框架，用于海事轨迹异常识别。

**💡 创新点**

创新点在于利用罕见度得分动态调节FiLM调制力度，使得在稀有情境下更坚定、在常见情境下更保守，从而缓解频率偏差。

**🔧 技术方法**

采用Feature‑wise Linear Modulation (FiLM) 与罕见度门控相结合，罕见度通过马氏距离估计，整合至OIL‑AD离线模仿学习框架中。

**📊 数据集**

使用AIS‑ERA5数据集，采集Bass Strait地区船舶航迹与ERA5气象变量，构建检测样本。

**📈 对比分析**

与OIL‑AD、Concat、FiLM、Gated FiLM、ROCOD、CAD‑MTS等基线比较，RGFiLM在detour场景下平均F1最高0.595，FPR最低0.097，展示最佳F1–FPR平衡。

**⚠️ 局限性**

局限在于仅验证合成的detour注入，罕见度估计依赖训练分布，对环境漂移敏感，且未涵盖多样真实异常类型。

---

## 446. VideoMDM: Towards 3D Human Motion Generation From 2D Supervision

**arXiv ID:** 2606.13364 | [PDF](https://arxiv.org/pdf/2606.13364v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 447. Real-Time Execution with Autoregressive Policies

**arXiv ID:** 2606.13355 | [PDF](https://arxiv.org/pdf/2606.13355v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 448. SupraSNN: Exploiting Synapse-Level Parallelism in Spiking Neural Network Accelerators through Co-Optimized Mapping and Scheduling

**arXiv ID:** 2606.13354 | [PDF](https://arxiv.org/pdf/2606.13354v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 449. Low cost, easily manufactured, highly flexible strain and touch sensitive fiber for robotics applications

**arXiv ID:** 2606.13352 | [PDF](https://arxiv.org/pdf/2606.13352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 450. Enhanced Low-Density Region Exploration in Classifier-Guided Diffusion Models Through Modified Reverse Diffusion Sampling

**arXiv ID:** 2606.13347 | [PDF](https://arxiv.org/pdf/2606.13347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 451. Measurement-Based Performance Evaluation of SmartRSUs with Heterogeneous Antenna Architectures for V2X Communications

**arXiv ID:** 2606.13334 | [PDF](https://arxiv.org/pdf/2606.13334v1)

**作者:** Marco Savarese `[一作]` (University of Modena and Reggio Emilia), Carlo Augusto Grazia `[通讯]` (University of Modena and Reggio Emilia)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Modena Automotive Smart Area（MASA）实地测量，比较了两种基于Open DSRC Unit（ODU）的SmartRSU的天线配置对V2X通信性能（覆盖范围、RSSI、丢包率、延迟）的影响，并与商用RSU做对比。

**💡 创新点**

首次在同一基站上并置自研SmartRSU与商用RSU，利用可复现的开放平台实现天线布局的直接对比，揭示集成天线与外置天线在实际城市环境中的性能差异。

**🔧 技术方法**

基于IEEE 802.11p（ITS‑G5） V2X协议的Open DSRC Unit平台，配合GNSS定位，利用CAM报文和UDP流量在OBU与RSU之间进行监测，采集RSSI、丢包率、延迟等指标。

**📊 数据集**

使用MASA实验室实际采集的车载OBU在城市路网中移动产生的CAM数据以及由实验车辆与RSU进行的2 Mbps UDP流量，未使用公开数据集。

**📈 对比分析**

采用同一电杆共置、相同频率、相同发射功率的配置，在相同时间、相同环境下对三台RSU进行频繁广播并记录接收情况，结果显示外置ITS‑G5天线的SmartRSU覆盖范围和RSSI均优于集成天线，且接近商用RSU；延迟在BPSK约4 ms、QPSK约2 ms、16QAM约1 ms，满足实时V2X需求。

**⚠️ 局限性**

实验仅覆盖单一城市场景和有限的车流密度，未评估多RSU协作或高密度交通情况下的性能，且仅使用自研平台，无法验证在更大规模部署的可扩展性。

---

## 452. Low-Latency Real-Time Audio Game Commentary System via LLM-Based Parallel Text Generation

**arXiv ID:** 2606.13322 | [PDF](https://arxiv.org/pdf/2606.13322v1)

**作者:** Ryota Kawamatsu `[一作]` (University of Tokyo), Tatsuya Ishigaki `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个低延迟的实时游戏音频解说系统，直接从直播游戏视频生成并同步播报口述解说。

**💡 创新点**

核心创新在于并行文本生成与视频延迟控制，提前生成候选解说并缓存，从而消除传统顺序处理造成的长沉默。

**🔧 技术方法**

使用多模态大型语言模型（GPT‑4.1‑mini）进行文本生成，文本转语音（TTS）合成，以及视频分段缓存与轻量级候选选择策略（Latest/Oldest/Random）。

**📊 数据集**

实验数据来自Super Smash Bros. Ultimate的Smash Corpus视频，并配备专业解说文本作为参考。

**📈 对比分析**

通过与顺序基线（After‑Audio、After‑Text）以及三种并行策略比较，评估指标包括累计沉默、平均沉默、mIoU、ROUGE；结果表明并行方法将平均沉默从9.6 s降至0.3 s，mIoU提升至≈0.6，ROUGE也优于基线。

**⚠️ 局限性**

局限性包括仅在快节奏游戏（Super Smash Bros. Ultimate）上验证，需额外的GPU资源支持LLM推理，对视频延迟的手动调节仍可能影响同步精度，且内容适配性和跨游戏通用性尚待进一步研究。

---

## 453. Skiplists with Foresight: Skipping Cache Misses

**arXiv ID:** 2606.13321 | [PDF](https://arxiv.org/pdf/2606.13321v1)

**作者:** Tomer Cory `[一作]` (Technion), Erez Petrank `[通讯]` (Technion)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Foresight 这一跳表优化，通过在每层将下一节点的键值与指针预存，显著提升局部性并减少缓存未命中；

**💡 创新点**

创新点在于将 next_key 与 next 指针协同存放，并针对并发场景提供两种同步方案——乐观验证与利用 SIMD 原子操作，解决一致性难题；

**🔧 技术方法**

技术手段包括在节点中增设 next_key 字段、实现乐观验证机制、利用 Intel 平台 128 位 SIMD 加载/存储原子操作、在 Synchrobench 的三种并发跳表及 DBx1000 内存数据库索引上实现该优化；

**📊 数据集**

实验使用了 Synchrobench 的微基准（随机键、不同工作负载）以及 DBx1000 的 TPC‑C 与 YCSB（10M/32M）数据表；

**📈 对比分析**

通过与基线跳表对比，测量吞吐量与缓存未命中，结果在微基准中提升最高达 45%，在 DBx1000 中提升最高 15.7%，缓存未命中下降最多 50%；

**⚠️ 局限性**

主要限制包括：在高并发或高空间敏感场景下空间开销可达 60%，对基线锁实现的提升有限，以及 SIMD 原子操作的硬件依赖性。

---

## 454. MagPlus: Bridging Micro-to-Regular Facial Expressions through Learnable Magnification

**arXiv ID:** 2606.13312 | [PDF](https://arxiv.org/pdf/2606.13312v1)

**作者:** Sliman Jammal `[一作]` (Ben-Gurion University of Negev), Andrei Sharf `[通讯]` (Ben-Gurion University of Negev)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种MagPlus–DeMagPlus两阶段流程，先对微表情进行可学习的运动放大，再利用冻结的宏表情动画模型生成，再通过去放大恢复微表情幅度，从而实现不需微调骨干的微表情生成。

**💡 创新点**

通过将微表情映射到宏表情模型可感知的运动范围，并在此基础上进行逆向缩放，解决了微表情被宏表情模型忽略的根本问题，实现了模型无关、可插拔的微表情生成。

**🔧 技术方法**

采用FlowMag变体的可学习运动放大网络、RAFT光流估计、U‑Net结构，并将其与FOMM、FSRT、MetaPortrait、EmoPortraits等预训练动画骨干配合使用。

**📊 数据集**

在大规模公开视频数据上预训练，在SAMM和CAS(ME)³微表情数据集上进行微调以适应微表情特征。

**📈 对比分析**

在UAR、UF1、Accuracy与Motion Magnification Ratio（MMR）等评估指标上，与四种预训练骨干及专用微表情生成方法FPB‑FOMM和TPS FaceParsing 对比，MagPlus‑DeMagPlus在三类情绪识别和运动幅度匹配上均表现优于或接近专用模型。

**⚠️ 局限性**

需要针对每个数据集单独微调放大/去放大模块，并手动选择放大因子，缺乏统一的自动化调参方案。

---

## 455. Subdivision-based isogeometric analysis for axisymmetric electromagnetic problems

**arXiv ID:** 2606.13308 | [PDF](https://arxiv.org/pdf/2606.13308v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 456. Feasibility Assessment of Remote Driving via Latency Analysis of ITS-G5 and Cellular Networks in the MASA Living Lab

**arXiv ID:** 2606.13292 | [PDF](https://arxiv.org/pdf/2606.13292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 457. RogueAI: A Reverse Turing Test for Detecting Licensed AI Deception in Dialogue

**arXiv ID:** 2606.13310 | [PDF](https://arxiv.org/pdf/2606.13310v1)

**作者:** Sara Candussio `[一作]` (University of Trieste), Luca Bortolussi `[通讯]` (University of Trieste)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个可部署的网页游戏（RogueAI），通过让玩家在有限轮次内对话询问两个大型语言模型（LLM）来判断谁在欺骗，从而实现了重塑的图灵测试。

**💡 创新点**

创新点在于将图灵测试转化为“可信度检测”游戏，强调单人判断两名隐藏身份LLM中的欺骗者，并引入可协作生成情境（AutoRogueAI）以及公开可视化的后期揭示机制。

**🔧 技术方法**

使用了OpenAI的GPT‑4（及GPT‑4‑Turbo）作为代理，配合Prompt工程实现角色隐藏与情境设定；还利用OpenAI TTS生成语音，后端采用Python Flask/Redis存储会话。

**📊 数据集**

通过在意大利科普节三天公开部署收集了415个完成游戏会话（共1876个回答轮次），并对每个回答进行手工标注为真实/欺骗，构成了实验数据集。

**📈 对比分析**

与简单的逻辑回归基线（利用四个表面语言特征）比较，玩家平均识别率为56.6%，而模型基线可达75.6%，显示玩家未充分利用可辨别的语言签名。

**⚠️ 局限性**

局限性包括样本为意大利语科普节自愿者、仅使用单一LLM（OpenAI），缺乏对其他模型或语言的验证，且未对玩家策略进行更细粒度的统计分析。

---

## 458. SmartFont: Dynamic Condition Allocation for Few-Shot Font Generation

**arXiv ID:** 2606.13382 | [PDF](https://arxiv.org/pdf/2606.13382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 459. Physics-Guided Spatiotemporal Learning for Coastal Wave Peak Period Estimation from Video

**arXiv ID:** 2606.13302 | [PDF](https://arxiv.org/pdf/2606.13302v1)

**作者:** Abubakar Hamisu Kamagata `[一作]` (Namibia University of Science and Technology), Paramasivam Saravanakumar `[通讯]` (Namdeb Diamond corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种物理引导的端到端深度时空学习框架，用近岸视频直接估计波峰周期（T_p）。

**💡 创新点**

创新点在于结合自动基于时间方差的ROI检测、三阶段Sim-to-Real预训练以及物理约束损失，显著提升预测精度与物理一致性。

**🔧 技术方法**

技术手段包括视频视觉Transformer（LtViViT）、TinyWaveNet+卷积+LSTM等架构，利用Airy波理论生成合成数据、光流伪标签和Huber+物理损失。

**📊 数据集**

使用数据集为13条现场视频（Golden Set，共6,926帧窗口）、20条噪声视频（Silver Set，共10,655帧窗口）以及超过1,000段合成波视频。

**📈 对比分析**

与基线CNN/GRU等模型对比，最佳Transformer实现RMSE降至0.769s，物理约束版本PtAttnCNN（λ=5）获得WS=0.9811、SI=0.0892，显示出卓越的精度与海洋学技能。

**⚠️ 局限性**

局限性包括仅在纳米比亚海岸进行验证、Golden Set规模有限且未设置独立测试集、未覆盖其他海域和其他波浪参数（如Hs、方向）。

---

## 460. Quantizing Time-Series Models As Dynamical Systems: Trajectory-Based Quantization Sensitivity Score

**arXiv ID:** 2606.13300 | [PDF](https://arxiv.org/pdf/2606.13300v1)

**作者:** Mariya Pavlova `[一作]` (Imperial College London), Yingzhen Li `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了轨迹感知量化敏感度分数（TQS）及其无校准混合精度分配算法TQS-PTQ，用于高效压缩时间序列和天气预测模型。

**💡 创新点**

创新点在于将后训练量化视为有限时间稳定性问题，利用动态系统理论定义分数，解耦敏感度估计与量化器/位宽选择，并通过多选背包算法实现精度分配。

**🔧 技术方法**

使用的技术包括离散时间动态系统建模、轨迹误差传播分析、Gaussian 噪声代理、无校准前向传播、以及混合精度分配的多选背包求解。

**📊 数据集**

实验数据集涵盖 Aurora-small、TimesFM-2.5、Pangu-Weather 等天气预测模型，并使用 ERA5、ETTh1/2、ETTm1/2 等公开时序数据进行评估。

**📈 对比分析**

与 RTN、GPTQ、GPTAQ、QEP 等基准在相同压缩率下比较，TQS-PTQ 在高压缩比下实现 MAE ≤1% 的降解，且在推理前的敏感度评估与混合精度分配上比传统方法快 2–5 倍。

**⚠️ 局限性**

局限性包括对输入/输出边界敏感性假设的适用范围有限，且对极大规模或高度并行化模型的前向评估仍需改进以降低计算成本。

---

## 461. Dual-Constrained Diffusion Image Compression for Operational Rate-Distortion-Perception Optimization

**arXiv ID:** 2606.13366 | [PDF](https://arxiv.org/pdf/2606.13366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 462. Positional Encoding in the Context of Memristor-Based Analog Computation for Automatic Speech Recognition

**arXiv ID:** 2606.13379 | [PDF](https://arxiv.org/pdf/2606.13379v1)

**作者:** Benedikt Hilmes `[一作]` (RWTH Aachen University), Ralf Schlüter `[通讯]` (RWTH Aachen University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究将相对位置编码（relative positional encoding）加入的 Conformer ASR 模型映射到模拟忆阻器硬件上，并针对在低精度量化与 ADC（模数转换）限制下出现的误差，提出并验证多种缓解方案。

**💡 创新点**

创新点在于：①发现位置编码层对 ADC 的范围与精度极为敏感，导致大幅性能衰退；②通过增大 ADC 范围或调整精度比重、以及在模型中移除或改写位置编码的线性映射等方法，有效将误差分别降低约 50% 与 30%；③验证在 4-bit 权重量化场景下相对位置编码仍能带来显著性能提升。

**🔧 技术方法**

技术手段：使用 SynaptogenML 进行忆阻器硬件仿真；对权重进行 4/8-bit 低精度量化；对 ADC 的精度(bit)与范围(bit)进行可调实验；改造相对位置编码（学习式、无线性变换等）；训练并评估 CTC‑based Conformer 模型。

**📊 数据集**

数据集：LibriSpeech（dev‑other 子集）和 Loquacious（250 h 训练集）用于训练与评估；语言模型使用 KenLM 4‑gram。

**📈 对比分析**

比较方法：在标准 GPU 上训练并评估 WER；将模型映射到忆阻器仿真器并测量 WER；与不使用位置编码、或在 ADC 维持默认设置的基线进行对比；结果显示：低精度权重下相对位置编码可提升约 15%；通过调节 ADC 范围/精度可将误差降低约 50%，若无法调节 ADC，则去除线性映射可降低约 30%；能量消耗保持不变。

**⚠️ 局限性**

limitations: 仅在模拟环境验证，真实忆阻器芯片对 ADC 设置的可调性有限；低精度权重仍对模型鲁棒性有影响；实验仅覆盖小规模语音任务，未在更大模型或多任务场景下测试；缺乏针对硬件功耗与面积的实测分析。

---

## 463. Temporal Conductance and Bounds on the Voter Model for Dynamic Networks

**arXiv ID:** 2606.13374 | [PDF](https://arxiv.org/pdf/2606.13374v1)

**作者:** Tatiana Rocha Avila `[一作]` (Goethe University Frankfurt), John Lapinskas `[通讯]` (University of Bristol)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在时变图（temporal graph）中固定顶点度的投票模型（voter model），并给出了其达成共识的期望时间上界与下界，揭示了时间性导向的连通度量——时序导电率（temporal conductance）对共识时间的重要性。

**💡 创新点**

创新点在于提出了时序导电率 Φ(𝒢) 的定义，能够在每个时间窗口内利用一次高连通步长弥补瞬时图断连的缺陷；随后证明了投票模型在任意时变图中共识时间的上界为 O(m/(d_min Φ(𝒢)))，并构造了匹配的下界，说明该上界在常数因子上是最优的。

**🔧 技术方法**

技术上主要使用马尔可夫链与随机游走的偶合关系、漂移分析、潜能函数与可选停止定理、以及对时序图中每个窗口的最大导电率与平均导电率的精细估计。

**📊 数据集**

由于研究完全基于理论分析，本文并未使用任何实验数据集；所有结果均通过严格的数学证明给出。

**📈 对比分析**

与以往仅在静态图上或只考虑单步导电率的结果相比，本文的上界与下界在理论上给出了更一般、时间窗口化的性能保证；具体而言，在最坏情况下共识时间不超过 O(m/(d_min Φ(𝒢)))，而相同条件下旧方法只能给出 O(m/(d_min φ)) 的保证，其中 φ 是单步导电率。

**⚠️ 局限性**

限制主要包括：1）仍假设顶点度固定，若度随时间变化则当前结论不直接适用；2）仅给出上界与下界的常数因子，并未给出更细粒度的常数优化；3）实验验证缺失，实际网络中该导电率的计算复杂度与可实现性未作讨论。

---

## 464. ReFree: Towards Realistic Co-Speech Video Generation via Reward-Free RL and Multilevel Speech Guidance

**arXiv ID:** 2606.13304 | [PDF](https://arxiv.org/pdf/2606.13304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 465. A Quantitative Experimental Repeated Measures Study of Training Dynamics in a Small Llama Style Language Model Under a Compute-Aware Token Budget

**arXiv ID:** 2606.13370 | [PDF](https://arxiv.org/pdf/2606.13370v1)

**作者:** Joe Dwyer `[一作]` `[通讯]` (ECPI University), Joe Dwyer (ECPI University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定计算预算下，使用6个随机种子对4.26M参数的Llama风格小模型进行训练，记录每百万token间隔的验证损失、困惑度以及由验证损失曲线衍生的稳定性指标（波动率、回滑频率、尖峰率），并通过重复测量设计分析训练过程的非单调性与不稳定性。

**💡 创新点**

首次将训练动态视为评估计算效率的核心维度，而非仅靠最终指标；引入重复测量与间隔级别的稳定性指标，揭示在受限算力环境下多余token可能导致退化而非持续提升。

**🔧 技术方法**

使用PyTorch CPU全精度训练、AdamW优化器、学习率暖启动至0.0004再线性衰减至0.00004、微批量2、梯度累计8、上下文长度128、全连接无dropout、全局收集每百万token检查点的日志并计算滚动波动率、回滑与尖峰等。

**📊 数据集**

TinyStories文本语料库（训练/验证拆分固定），采用4096词表的Tokenizer。

**📈 对比分析**

通过一维重复测量ANOVA检验不同token间隔的验证损失、困惑度与滚动波动率差异，并用线性混合效应模型作为稳健性检验；结果显示验证损失从8.3552下降至约2.8后在后期上升至3.9010，验证困惑度亦呈类似轨迹；不满足预设的稳定阶段判定。

**⚠️ 局限性**

实验规模受限：仅使用4.26M参数模型、TinyStories单一数据集、CPU训练、6个种子；缺乏对更大模型、不同语料、不同硬件与学习率策略的验证，稳定性指标依赖验证损失曲线，可能无法在所有设置下通用。

---

## 466. NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation

**arXiv ID:** 2606.13494 | [PDF](https://arxiv.org/pdf/2606.13494v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 467. Can I Buy Your KV Cache?

**arXiv ID:** 2606.13361 | [PDF](https://arxiv.org/pdf/2606.13361v1)

**作者:** Luoyuan Zhang `[一作]` `[通讯]` (Harbin Institute of Technology), Luoyuan Zhang (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种预填充缓存（Prefill CDN）方案，将大型语言模型（LLM）在一次前向推理中生成的 KV 缓存持久化为可共享的“缓存文件”，让后续查询直接加载而不重复执行前向推理。

**💡 创新点**

创新点在于：①证明在共享前缀场景下，直接加载 KV 与从头预填充在 token 与 logits 级别上完全一致；②系统化评估了 KV 复用在计算成本上的巨大优势（9–50×节省）并给出即时收支平衡点；③将 KV 缓存视作跨方可付费的计算资产，提出完整的经济模型与市场框架。

**🔧 技术方法**

核心技术包括：KV 缓存序列化与反序列化、位置编码与注意力掩码的正确对齐、离线 KV 预计算、定量评估计算成本、以及对 KV 压缩的初步实验（如 int8 量化）。

**📊 数据集**

使用 Qwen3-4B（36 层，fp16）模型在 Apple M1 Pro（MPS）上跑 255–3774 token 长度的单一文本样本进行实验，未使用公开大规模语料库；重点在单文档的 KV 生成与复用。

**📈 对比分析**

比较方法：对同一上下文进行两种推理——一次完整的前向预填充 vs. 在已加载 KV 上进行单步推理；在多长度下测算 CPU/GPU 计算时间。结果显示，KV 复用在计算上比前向预填充快 8.6–49.7 倍，且随上下文长度增长加速；在经济模型下，单次复用即可实现成本收支平衡。

**⚠️ 局限性**

局限性包括：① KV 缓存仅与产生它的模型和精度绑定；②仅支持共享前缀，无法处理多段检索合并的情况；③完整 KV 文件体积巨大，分发成本高，需压缩；③实验仅在单台 Apple M1 Pro 与 Qwen3-4B 上完成，未验证跨设备或跨模型的普适性；④量化实验表明无损压缩是必要的，否则会破坏 token‑exact 性能。

---

## 468. EconCSLib: AI-Assisted Lean Formalization for Economics & Computation research

**arXiv ID:** 2606.13306 | [PDF](https://arxiv.org/pdf/2606.13306v1)

**作者:** Nikhil Garg `[一作]` `[通讯]` (Cornell Tech), Nikhil Garg (Cornell Tech)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个 Lean 4 形式化库 EconCSLib 及其 AI 助手工作流，能够自动将经济学与计算论文的定义、定理与证明转化为 Lean 代码，并通过可视化 DAG、dashboard 及 LLM‑as‑Judge 进行翻译校验，支持作者自我 formalization 与社区共享。

**💡 创新点**

创新点在于将 LLM 自动 formalization 与人机交互校验相结合，推出可交互的翻译审核 dashboard、自动生成依赖 DAG、以及“LLM‑as‑Judge”双重验证机制，从而使论文自动化 formalization 更高效、可追溯且可共享。

**🔧 技术方法**

使用技术包括 Lean 4 形式化语言、OpenAI GPT‑5.5 Pro（以及 Codex）生成 Lean 代码与证明、LLM‑as‑Judge 进行翻译对比、GitHub Actions 自动化脚本、可视化 DAG 与 dashboard、以及 RAG/LLM 技术来辅助文档提取与结构化。

**📊 数据集**

数据集为经济学与计算领域的多篇经典与近期论文（如 Gale‑Shapley、Roth、Goldberg‑Hartline 等），这些论文作为 formalization 目标并用于评估流程效果。

**📈 对比分析**

方法评估以人工翻译一致性、验证通过率、证明完整性（如是否出现 sorry/admit）以及成本指标为主；截至 2026‑06，已 formalized 13 篇论文，证明行数均在 5k‑150k 行，整体 token 成本约 16 k 美元，验证通过率在 90% 以上，但人类验证仍是瓶颈。

**⚠️ 局限性**

局限性包括对 LLM 生成错误与偏差的依赖、翻译验证仍需人工投入、缺失连续概率与计算复杂度等库导致部分论文无法完整 formalized、token 成本高、以及可扩展性与社区参与度待进一步提升。

---

## 469. Clustering Node Attributed Networks with Graph Neural Networks and Self Learning

**arXiv ID:** 2606.13444 | [PDF](https://arxiv.org/pdf/2606.13444v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 470. Mining Architectural Quality Under Agentic AI Adoption: A Causal Study of Java Repositories

**arXiv ID:** 2606.13298 | [PDF](https://arxiv.org/pdf/2606.13298v1)

**作者:** Oliver Aleksander Larsen `[一作]` (University of Southern Denmark), Mahyar T. Moghaddam `[通讯]` (University of Southern Denmark)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过差分设计与Arcan架构嗅觉分析，评估151个开源Java仓库中可观测的agentic AI工具采用对软件架构嗅觉密度（ASD）的因果影响。

**💡 创新点**

首次在架构层面使用Borusyak插值差分估计和分母效应分解揭示密度归一化指标的误导性，并证明agentic AI并未在短期内恶化成熟项目的架构。

**🔧 技术方法**

采用Staggered DiD、Borusyak插值、Sun & Abraham加权估计、Arcan嗅觉检测、倾向匹配、事件研究、日志变换等方法完成数据处理与因果推断。

**📊 数据集**

使用151个Java OSS仓库（74采用、77对照）共1,811条月度Arcan快照，数据来自GitHub、Arcan分析结果、配置文件与提交尾注。

**📈 对比分析**

通过匹配对照组并实施差分估计，结果显示agentic AI导致ASD下降6.7%，但实际上是由于代码量增长引起的分母效应；鲁棒性检验（wild bootstrap、Lee bounds、Bacon分解）均验证此结论。

**⚠️ 局限性**

局限性包括仅适用于成熟的Java OSS项目、短期（6个月）观察窗口、潜在的差异化缺失率、未观察到的自适应干预、仅检测可观测的agentic AI，以及对其他语言和绿色项目的外推受限。

---

## 471. Exploring Systems-Thinking Approaches to Loss of Control Risk

**arXiv ID:** 2606.13474 | [PDF](https://arxiv.org/pdf/2606.13474v1)

**作者:** Aurelio Carlucci `[一作]` (University of Oxford), Jakub Kryś `[通讯]` (SaferAI)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了内部部署AI编码代理的失控风险，使用系统安全方法STECA、STPA、FRAM分析治理结构、控制机制与运维变异。

**💡 创新点**

创新点在于将成熟的系统安全方法应用于前沿AI内部部署，揭示模型级评估忽略的治理与运维失效情景，并给出监管与运营指标建议。

**🔧 技术方法**

使用的技术包括系统理论早期概念分析（STECA）、系统理论过程分析（STPA）和功能共振分析法（FRAM）。

**📊 数据集**

数据来源为公开的安全框架、系统卡、监管文件和风险报告等公开信息，未使用专有运营数据。

**📈 对比分析**

方法比较以定性结果呈现，指出不同方法揭示的风险层面互补；未给出量化性能指标。

**⚠️ 局限性**

局限性在于仅依赖公开资料，缺乏实际运营数据，结果受假设与主观判断影响，未验证发现的实际发生概率。

---

## 472. Optimizing Appliance Scheduling for Solar Energy Management Using Metaheuristic Algorithms

**arXiv ID:** 2606.13407 | [PDF](https://arxiv.org/pdf/2606.13407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 473. $W-δ-μ$ dual codes and LCD codes

**arXiv ID:** 2606.13467 | [PDF](https://arxiv.org/pdf/2606.13467v1)

**作者:** Avanish Kumar Chaturvedi `[一作]` (University of Allahabad), Satyadeep Pandey `[通讯]` (University of Allahabad)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出了W-δ-μ泛化内积并研究其对应的码的对偶性质，包括自正交、自共轭、LCD等。

**💡 创新点**

创新点在于将权重向量W、域自同构δ和置换μ综合进一个内积，统一扩展了欧氏、赫尔米特、δ积等已有形式，并给出对应的对偶、性质及存在性结果。

**🔧 技术方法**

使用线性代数、矩阵表示、塞塞列形式、Schwartz–Zippel引理等理论工具。

**📊 数据集**

无具体数据集，全部为理论推导。

**📈 对比分析**

通过对重量枚举、多项式等理论比较，验证MacWilliams等关系在该积下仍成立；没有数值性能指标。

**⚠️ 局限性**

局限在于对参数δ、μ的存在性和对偶保持性质尚未完全分类，且在保持共价循环性等方面的条件尚不完整。

---

## 474. Budget-Constrained Step-Level Diffusion Caching

**arXiv ID:** 2606.13496 | [PDF](https://arxiv.org/pdf/2606.13496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 475. S-GBT: Smooth Growth Bound Tensor for Certified Robustness Against Word Substitution Attacks in NLP

**arXiv ID:** 2606.13439 | [PDF](https://arxiv.org/pdf/2606.13439v1)

**作者:** Mohammed Bouri `[一作]` (Mohammed VI Polytechnic University), Adnane Saoud `[通讯]` (Mohammed VI Polytechnic University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Smooth Growth Bound Tensor (S-GBT)，作为二阶正则化方法，用于提升 NLP 词替换攻击的可证明鲁棒性，并将其与 LSTM 与 CNN 结合训练；

**💡 创新点**

在 GBM 仅约束梯度的一阶敏感度的基础上，引入逐元素二阶导数（Hessian）上界 S-GBT，直接控制模型曲率，从而得到更紧凑的鲁棒性保证；

**🔧 技术方法**

推导 LSTM 与 CNN 的 S-GBT 解析表达式，将 GBM 与 S-GBT 正则化加入交叉熵损失进行训练；通过理论证明给出对词替换扰动的输出界；实验使用 OpenAttack 的 GA、PWWS、PSO 等攻击；

**📊 数据集**

使用公开基准数据集 IMDB（情感分类）和 Yahoo! Answers（主题分类）；

**📈 对比分析**

与标准训练、IBP、GBM、SEM、ATFL、ASCC 等六种基线对比；在三种攻击下，S-GBT 在 Yahoo! 上 CNN 的 AUA 提升至 90.7%（比 GBM 提升 23.4%），BiLSTM 上提升约 21%；在 IMDB 上提升 5.7–9.3%；同时保持与基线相近的清洁准确率；

**⚠️ 局限性**

仅适用于可微分网络，ReLU+max‑pool CNN 的 Hessian 接近零导致 S-GBT 效果有限；对可变长度扰动（词删除等）支持不足。

---

## 476. Point-Wise Geometry-Aware Transformer for Partial-to-Full Point Cloud Registration in Computer-Assisted Surgery

**arXiv ID:** 2606.13488 | [PDF](https://arxiv.org/pdf/2606.13488v1)

**作者:** Siyu Zhou `[一作]` (Technical University of Munich), Zhongliang Jiang `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了GAPR-Net框架，用于在计算机辅助外科中对部分到完整的三维点云进行高精度配准。

**💡 创新点**

创新点包括：① 设计了多尺度点级几何特征（PGF）以保持旋转、平移不变性；② 将局部卷积（KPConv）与基于PGF的Transformer（PGA）结合，实现局部与全局特征的互补；③ 在全局聚合模块的最后一层使用带重定位的跨注意力（cross‑PGA），显著提升匹配质量。

**🔧 技术方法**

使用的技术包括：KPConv卷积、Transformer自注意力与跨注意力、几何感知位置编码、粗细配准策略（coarse‑to‑fine）、Softmax‑Sinkhorn对应、RANSAC、损失函数（粗细匹配交叉熵、负对数似然）等。

**📊 数据集**

数据集为四类人骨（胫骨、股骨、骨盆、胸壁软骨）共计约2300对点云，采用CT与超声扫描合成，包含40%–70%覆盖率以及不完整完整体场景，并在此基础上人工添加噪声与密度变化。

**📈 对比分析**

与ICP、Predator、Lepard、GeoTransformer、RoITr等基线比较，GAPR-Net在整体上取得了最高的注册召回率94.2%、最小的RMSE1.992 mm、CD4.937 mm，并在各骨骼类别上均优于或与现有SOTA相当，尤其在胸壁软骨和不完整完整体的极端场景下表现稳健。

**⚠️ 局限性**

局限性主要包括：① 需要大量训练数据和计算资源；② 对极端遮挡（低于40%覆盖率）仍易失效；③ 只在CT‑超声模拟场景验证，真实临床数据的泛化性尚待进一步评估。

---

## 477. Ontology Memory-Augmented ASR Correction for Long Text-Speech Interleaved Conversations

**arXiv ID:** 2606.13464 | [PDF](https://arxiv.org/pdf/2606.13464v1)

**作者:** Xinxin Li `[一作]` (Harbin Institute of Technology), Min Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向长文本-语音交互会话的本体记忆增强语音识别纠错框架

**💡 创新点**

创新在于将会话历史动态构建为可检索本体记忆，以结构化上下文支持纠错

**🔧 技术方法**

采用大语言模型（Qwen、Gemma）配合本体提取、检索与增量更新技术

**📊 数据集**

构建了基于 MagicData‑RAMC 的 RAMC‑Corr 数据集

**📈 对比分析**

与直接纠错和全历史基线对比，10 种模型组合中有 9 种提升 CER，提升幅度约 10%–20%

**⚠️ 局限性**

局限包括仅中文数据、未针对 ASR 纠错专门化检索、仅无监督推理、对真实交互环境的泛化不足

---

## 478. Reinforcement Learning for Neural Model Editing

**arXiv ID:** 2606.13461 | [PDF](https://arxiv.org/pdf/2606.13461v1)

**作者:** Shaivi Malik `[一作]` `[通讯]`, Shaivi Malik

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出将神经网络模型编辑视作强化学习任务，设计 MaskWorld 与 ShiftWorld 两种环境，让智能体通过对网络权重进行乘法或加法更新，依据奖励信号学习编辑策略。

**💡 创新点**

创新点在于：① 将模型编辑问题统一化为奖励驱动的RL问题，消除手工编写专用算法的需求；② 采用 LoRA 启发式的低维动作空间表示，显著降低高维权重更新的搜索空间；③ 通过统一奖励函数同时兼顾模型整体性能与编辑目标，实现多任务的可迁移编辑。

**🔧 技术方法**

核心技术包括：Proximal Policy Optimization (PPO) 训练策略网络，MLP [128,128] 结构，LoRA-inspired 低秩权重更新表示，奖励函数由 utility 与 task‑specific 两部分加权构成。

**📊 数据集**

实验使用的数据集为：MNIST（用于机器无学习任务）和 Jigsaw Toxic Comment Classification Challenge（用于偏见缓解任务），同时使用 Jigsaw 合成的无偏见测试集评估偏见程度。

**📈 对比分析**

对比方法包括原始模型和仅微调最后一层的基线；实验结果显示：无学习任务中忘记集准确率降至 0% 左右，保留集准确率提升至 93.5%；偏见缓解任务中无偏见测试集准确率提升 6–7%，而主任务准确率仅下降 1%。

**⚠️ 局限性**

主要局限在于：① 动作空间依然巨大，仅能有效修改单层权重；② 对大模型全局编辑的可扩展性差；③ 依赖于 RL 的高方差与收敛不稳定，需进一步改进多智能体或更高效的奖励设计。

---

## 479. Intent-Based Cryptographic API Design for Cryptographic Agility

**arXiv ID:** 2606.13445 | [PDF](https://arxiv.org/pdf/2606.13445v1)

**作者:** Navaneeth Rameshan `[一作]` (IBM Research Europe), Gregoire Messmer `[通讯]` (IBM Research Europe)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种面向后量子密码学的意图驱动加密API设计，支持算法迁移、政策治理和密钥演化；

**💡 创新点**

创新点在于引入作用域（scope）概念实现意图化密钥创建，拆分参数类型并通过协议缓冲区实现可扩展、抽象且可治理的接口；

**🔧 技术方法**

使用gRPC与Protocol Buffers定义接口，结合多注册表（模板与提供者）和抽象策略引擎；

**📊 数据集**

无特定数据集，主要通过对现有流行API（PKCS#11、OpenSSL 3.0、JCA、Tink、AWS KMS、HashiCorp Vault）进行评估；

**📈 对比分析**

评估基于伴随框架的七维度指标，API达成C1.3、C2.2、C3.3、C4.3、C5.3–5.4、E1.3、E2.2，显著弥补现有API的三大缺口；

**⚠️ 局限性**

局限在于未实现最高级别的C1.4、C2.3、C3.4以及基于策略的自动算法迁移，需要在实现层面进一步完善。

---

## 480. An End-to-End Hybrid Framework for Rumour Detection in Low-Resources Algerian Dialect

**arXiv ID:** 2606.13411 | [PDF](https://arxiv.org/pdf/2606.13411v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 481. How Much Memory Do We Need? Adaptive Memory Gate for Neural Operators

**arXiv ID:** 2606.13443 | [PDF](https://arxiv.org/pdf/2606.13443v1)

**作者:** Jihyeon Hur `[一作]` (KAIST), Noseong Park `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研发了一种自适应记忆门的神经算子AMGFNO，用以在低分辨率下通过动态调节记忆权重解决非马尔可夫性问题。

**💡 创新点**

创新点在于将可学习的内容门与频率感知门结合，既实现了长程记忆、可调记忆，又能根据观察分辨率和物理参数自适应调节记忆使用。

**🔧 技术方法**

采用FFNO骨干网络、S4结构的时间状态空间记忆层以及门控机制（内容门+频率门），并在自回归训练与评估框架下实现。

**📊 数据集**

在一维Kuramoto–Sivashinsky和Burgers方程上，使用低、中、高分辨率（f=32,64,128）以及不同黏性系数的数据集进行实验。

**📈 对比分析**

通过与Markovian基线（Factformer、GKT、U-Net、FFNO）和记忆增强基线（Multi-Input FFNO、S4FFNO）的nRMSE比较，AMGFNO在低分辨率下相较S4FFNO降幅55–79%，在高分辨率下匹配或优于纯马尔可夫FFNO。

**⚠️ 局限性**

局限性包括仅在1D PDE上验证，频率门依赖预估高频能量比例，且对噪声、分布漂移和更高维问题的鲁棒性尚未深入探究。

---

## 482. Accelerating Speculative Diffusions via Block Verification

**arXiv ID:** 2606.13426 | [PDF](https://arxiv.org/pdf/2606.13426v1)

**作者:** Alexander Soen `[一作]` (Kth), Arnaud Doucet `[通讯]` (Google Deepmind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于block verification的speculative diffusion加速框架，并实现了一种高效的Γ‑maximal coupling残差采样方法；此外，作者设计了无训练的Free Drafter来进一步提升推理速度。

**💡 创新点**

创新点包括：
1) 通过单步正交分解实现了对连续空间残差分布rΓ的高效采样；
2) 将LLM中的block verification迁移到连续diffusion，显著提升接受率；
3) 证明了reflection coupling无法用于block verification，阐明了随机残差必要性；
4) 提出了Free Drafter，利用上一轮验证中的score实现完全无训练、成本极低的自拟稿策略，提升最多6.3%的延迟速度。

**🔧 技术方法**

技术手段包括：speculative sampling、Γ‑maximal coupling、block verification、残差分布采样（正交分解 + 逆变换）、Euler–Maruyama离散化、U‑Net结构的DDPM diffusion模型、Free Drafter的自回归扩散推理。

**📊 数据集**

实验数据集涵盖多种图像任务：CIFAR10、LSUN、CelebA、ImageNet（像素空间和潜在空间），主要在ImageNet LDM上进行深入评估。

**📈 对比分析**

比较方法：对标准DDPM进行wall‑clock速度对比，测量block efficiency和FID；结果显示block verification提升1.5–6.3%的速度，Free Drafter在保持相同FID的前提下，比Frozen Drafter更快；整体推理速度提升约1.9–3.6×。

**⚠️ 局限性**

局限性：
- 仅适用于speculative diffusion，无法应用于确定性采样器；
- draft模型与目标模型需共享相同的降噪调度；
- block verification在并行验证时会产生额外的内存开销；
- Free Drafter的block efficiency受限，不能在所有设置下超过Frozen Drafter；
- 速度提升随噪声调度和采样步数变化，若步骤减少则提升幅度减小。

---

## 483. Neuro-Symbolic Agents for Regulated Process Automation: Challenges and Research Agenda

**arXiv ID:** 2606.13405 | [PDF](https://arxiv.org/pdf/2606.13405v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 484. Person Identification from Contextual Motion

**arXiv ID:** 2606.13410 | [PDF](https://arxiv.org/pdf/2606.13410v1)

**作者:** Igor Kviatkovsky `[一作]` (Technion Israel Institute of Technology), Ilan Shimshoni `[通讯]` (University of Haifa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于运动模式的身份识别框架，涵盖登录、智能家居和交互式三种场景。

**💡 创新点**

创新点在于将运动识别转化为生成式概率模型，并在交互式场景中使用信息增益主动挑选提示，实现快速可靠识别。

**🔧 技术方法**

采用生成式概率模型、线性/马氏距离度量、1‑NN KDE、信息最大化选择以及与深度学习模型对比实验。

**📊 数据集**

使用公开的 NTU RGB+D、MSRC‑12、UCFKinect、MSR‑Action3D、UTKinect、BodyLogin 数据集以及自采集的 CuedId 数据集。

**📈 对比分析**

与经典方法（DTW、Hough Forest、Cov3DJ、SVM、RF）以及深度 RNN/IndRNN 对比，取得与或优于最先进方法的识别准确率；在交互式场景下平均仅需 4–7 次提示即可达到 80–90% 置信。

**⚠️ 局限性**

局限包括需要大量标注动作‑身份样本、对动作识别误差敏感、交互式提示空间离散化以及开放集识别仍需进一步改进。

---

## 485. A catalog of fast matrix multiplication algorithms with frontier-closure search

**arXiv ID:** 2606.13408 | [PDF](https://arxiv.org/pdf/2606.13408v1)

**作者:** Benoit Chatain Lacelle `[一作]` `[通讯]`, Benoit Chatain Lacelle

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了一个统一、机器可检验的矩阵乘法算法目录，并实现了前沿闭包搜索（frontier‑closure search）来自动发现和组合已存在的低秩算法，覆盖了 2≤n≤m≤p≤32 的所有小格式，并为每个条目记录完整的归因、线性变换线索以及可复现的实现。

**💡 创新点**

创新点主要包括：① 统一不同学科、字段和序列化规范的低秩矩阵乘法记录；② 引入非重叠属性（non‑overlap property）保证组合操作不重新发现共享的 bilinear 核，清晰区分“发现新核心”与“组合已知核心”的进展；③ 前沿闭包搜索与按形状层次的依赖图并行，极大提升搜索效率；④ 通过“serendipitous product（bud）”与“pair fusion”等新组合策略实现局部 ω 降低；⑤ 提供可验证的 JSON 目录结构与 Lineage DAG 以支持复现与审计。

**🔧 技术方法**

技术上使用了 JSON+BSON 结构化存储、线性代数符号验证、随机 Spot‑Check 与符号检验、线性变换（GLn × GLm × GLp）与轴置换、Kronecker、轴拼接、Recombination、Axis‑Flip、Axis‑Permute、Output Peel、Pair Fusion、Serendipitous Product 等组合算子，以及前沿闭包搜索的两阶段（搜索+懒加载）实现。

**📊 数据集**

数据集主要来源于公开算法集合：AlphaTensor、AlphaEvolve、Schwartz–Zwecher 2025、Perminov 的 flip‑graph、FMM‑Lille、Hopcroft–Kerr、Laderman、Smirnov 等，涵盖 323232 以内的所有形状，且按字段（K0, K, KZ, KT, K2, GF(3) 等）和是否可交换进行标签。

**📈 对比分析**

比较方法：在统一目录基础上重建 DIS09 经典表格，按字段和可交换列分别展示最优秩；使用前沿闭包搜索自动更新表格；对新发现的方案与已知最佳进行秩、ω、加法计数的对比。实验显示，通过组合与 serendipitous product 等手段，已在多种形状上实现了新的最优秩，显著降低了 ω（例如 336 形状达到 2.7743），并验证了非重叠属性下的预测秩与实际秩完全一致。

**⚠️ 局限性**

局限性：① 搜索仅限于已实现的组合算子，未涵盖 τ‑theorem 级别的 disjoint‑sum 生成；② 目前仅覆盖 32 以内的形状，对更大尺寸的组合与优化缺乏覆盖；③ 非重叠属性阻止了搜索自动发现共享 bilinear 核；④ 某些高阶形状（如 21015、21216 等）仍未能完全实现理论上可达的最优秩；⑤ 目录对特殊字段（如 GF(3)、GF(2)）的支持尚不完整。

---

## 486. Heterogeneous LiDAR Early Fusion and Learned Re-Ranking Strategy for Robust Long-Term Place Recognition in Unstructured Environments

**arXiv ID:** 2606.13503 | [PDF](https://arxiv.org/pdf/2606.13503v1)

**作者:** Judith Vilella-Cantos `[一作]`, Luis Payá `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

**🎯 论文内容**

介绍并证明了多项式展开的基本公式 f(x) = (x+a)(x+b) = x^2 + (a+b)x + ab。

**💡 创新点**

无创新点，主要是复述已知的代数公式。

**🔧 技术方法**

使用了符号数学推导与示例表格展示。

**📊 数据集**

无使用数据集。

**📈 对比分析**

未进行方法比较，性能讨论不存在。

**⚠️ 局限性**

局限在于内容过于基础、缺乏实际应用或新颖性。

---

## 487. SPARC: Reliable Spatial Annotations from Robot Demonstrations at Scale

**arXiv ID:** 2606.13497 | [PDF](https://arxiv.org/pdf/2606.13497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 488. VISA: VLM-Guided Instance Semantic Auditing for 3D Occupancy World Models

**arXiv ID:** 2606.13460 | [PDF](https://arxiv.org/pdf/2606.13460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 489. Evaluation Sovereignty in Metadata-Driven Classification: A Multi-Track Framework for Weakly Supervised Information Systems

**arXiv ID:** 2606.13436 | [PDF](https://arxiv.org/pdf/2606.13436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 490. CRAFTIIF: Cross-Resolution Analytic Four-Type Interpretable Isolation Forest for Multivariate Time Series Anomaly Detection

**arXiv ID:** 2606.13486 | [PDF](https://arxiv.org/pdf/2606.13486v1)

**作者:** William Smits `[一作]` `[通讯]` (Avathon), William Smits (Avathon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种完全无监督的多变量时序异常检测框架，该框架通过四种波形族的随机多分辨率分析波形特征，配合四个专属孤立森林分支和一个元分支，自动校准阈值并提供类型可解释性；

**💡 创新点**

创新点包括（1）针对四类结构化异常（点、分布、时序、协同）设计专门的特征与分支，实现一次性覆盖所有类型；（2）引入自适应Otsu/MAD混合阈值以处理不同异常率；（3）提出诊断框架（oracle F1、可检出性极限、分支分离比、协同归因），可在无标签情况下区分模型失效、特征不足和根本不可检出三种情况；

**🔧 技术方法**

技术手段主要为随机采样的解析波形特征生成（Morlet、DOG、Haar、Coiflet）、FFT加速连续小波变换、五分支结构化孤立森林、基于分布双峰度的Otsu与MAD混合阈值校准、以及基于分支分离比的可解释性评估；

**📊 数据集**

使用了mTSBench公开基准的19个多变量时序数据集（覆盖点、分布、时序、协同等多种异常类型与不同通道数、异常率）；

**📈 对比分析**

与mTSBench中24种基线方法对比（包括原始孤立森林、USAD、TranAD等），在所有19个数据集上均无超参调优，平均VUS‑PR 0.463（领先所有方法），在可检出的数据集上平均F1为0.322，整体F1提升显著；

**⚠️ 局限性**

主要局限包括（1）窗口级检测导致短异常段样本误报，需要子窗口定位；（2）异常率超过约50%的数据集（如cicids）训练数据混杂异常，IF边界失效；（3）存在6个数据集的根本可检出性极限，任何无标签统计方法均难以检测，需域知识或标签。

---

## 491. OmniDirector: General Multi-Shot Camera Cloning without Cross-Paired Data

**arXiv ID:** 2606.13432 | [PDF](https://arxiv.org/pdf/2606.13432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 492. Toward Instructions-as-Code: Understanding the Impact of Instruction Files on Agentic Pull Requests

**arXiv ID:** 2606.13449 | [PDF](https://arxiv.org/pdf/2606.13449v1)

**作者:** Ali Arabat `[一作]` (École de Technologie Supérieure), Mohammed Sayagh `[通讯]` (École de Technologie Supérieure)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在软件项目中添加AI代理指令文件后，Agentic Pull Requests（PR）的质量、复杂度和合并效率的影响，系统性比较了指令文件创建前后各类度量指标。

**💡 创新点**

首次从宏观项目层面量化指令文件对AI代理表现的正负效应，并提出“Instructions‑as‑Code”这一新范式，指出指令文件的结构和长度与合并成功率呈显著关联。

**🔧 技术方法**

使用统计检验方法（Mann‑Whitney U、Cliff's Δ）对比前后差异，并利用GitHub GraphQL API、AIDev数据集收集与处理Agentic PR和指令文件。

**📊 数据集**

基于AIDev + GitHub 的公开数据，共收集148个项目（15,549条Agentic PR），并提取了相关PR的元数据与指令文件信息。

**📈 对比分析**

对每个项目分别计算指令文件创建前后 merge‑rate、代码 churn、讨论数、合并时间等指标的百分比变化；结果显示约27.7% 的项目 merge‑rate 提升≥20%，约26.35% 降低≥20%，其他指标在提升与降低之间交替，整体影响不一。

**⚠️ 局限性**

局限性包括：仅关注统计显著性而未建立因果机制；指令文件内容质量未细分；项目过滤与样本规模可能引入偏差；未探讨不同类型AI代理（Copilot、Claude等）的差异性。

---

## 493. PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Update

**arXiv ID:** 2606.13400 | [PDF](https://arxiv.org/pdf/2606.13400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 494. Mod-Guide: An LLM-based Content Moderation Feedback System to Address Insensitive Speech toward Indigenous Ethnic and Religious Minority Communities

**arXiv ID:** 2606.13397 | [PDF](https://arxiv.org/pdf/2606.13397v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 495. Spectral Filtering of 3D Integral Operators Using Modified Green's Functions

**arXiv ID:** 2606.13489 | [PDF](https://arxiv.org/pdf/2606.13489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 496. From Traditional Automation to Embodied Wireless Intelligence: Vision-Language-Action Empowered Physics-Aware Communication Networks

**arXiv ID:** 2606.13458 | [PDF](https://arxiv.org/pdf/2606.13458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 497. Examining the Cognitive Gap Between Authors and Peer Reviewers on Academic Paper Novelty

**arXiv ID:** 2606.13452 | [PDF](https://arxiv.org/pdf/2606.13452v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 498. CQC-RAG: Robust Retrieval-Augmented Generation via Cross-Query Consistency

**arXiv ID:** 2606.13438 | [PDF](https://arxiv.org/pdf/2606.13438v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 499. SupraBench: A Benchmark for Supramolecular Chemistry

**arXiv ID:** 2606.13477 | [PDF](https://arxiv.org/pdf/2606.13477v1)

**作者:** Tianyi Ma `[一作]` (University of Notre Dame), Yanfang Ye `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了首个Supramolecular Benchmark（SupraBench），评估LLM在宿主‑客体配位、亲和力预测、优胜者挑选、溶剂判别以及分子识别等任务中的推理能力；并公开了16M-token的Supramolecular文本语料库；通过系统比较多种开源与专有LLM的表现，揭示提示策略与领域适配的影响；

**💡 创新点**

①提出了包含四大基本任务和一项视觉任务的统一评测框架；②发布专门针对超分子化学的语料库；③在同一实验设置下评估LLM的多维度性能，阐明提示策略与领域适配的利弊，为后续模型改进提供基准；

**🔧 技术方法**

采用LLM提示技术（Base、Few‑Shot、CoT）、领域自适应预训练（DAPT）、多模态推理（图像‑SMILES生成）、标准化评估指标（MAE/RMSE、准确率、Macro‑F1、Rouge、SMILES有效率等），以及数据集构建与处理流水线；

**📊 数据集**

使用来自 SupraBank 的宿主‑客体亲和力记录；从 Europe PMC 筛选的 16M-token 超分子化学文本语料库；以及构建的五个评测任务数据集；

**📈 对比分析**

对八种开源/专有LLM（如 Qwen3.5、Llama‑3.1、DeepSeek、GPT‑5.4、Gemini‑3‑Flash 等）进行统一提示与评测，比较 MAE/RMSE、准确率、Macro‑F1、Rouge‑F1 与 SMILES 匹配率等指标；实验显示 frontier 专有模型在亲和力预测、优胜者挑选和溶剂判别上表现最佳，但整体仍有显著提升空间；提示策略的效果因任务而异，CoT 甚至会导致性能下降；

**⚠️ 局限性**

存在公开数据记忆风险导致评测结果可能被预训练数据过拟合；专有模型版本漂移可能影响复现性；CoT 触发模型幻觉，生成不可信的化学推理；领域适配在回归任务上有效，但对严格格式化的多选题可能产生负面影响；

---

## 500. Understanding the Rejection of Fixes Generated by Agentic Pull Requests -- Insights from the AIDev Dataset

**arXiv ID:** 2606.13468 | [PDF](https://arxiv.org/pdf/2606.13468v1)

**作者:** Mahmoud Abujadallah `[一作]` (École de Technologie Supérieure), Mohammed Sayagh `[通讯]` (École de Technologie Supérieure)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究AI编码代理生成的拉取请求被开发者拒绝的原因，并量化了因拒绝导致的代码变更量和评论成本。

**💡 创新点**

创新点在于提出了包含14个子主题的拒绝原因分类框架，发现AI代理修复请求的拒绝率高达46.41%，并揭示了技术问题、实现错误、无关性和提供者限制等主要拒绝因素。

**🔧 技术方法**

方法上结合定性主题分析和定量指标（代码变更量、评论数）评估拒绝成本，并通过对不同拒绝类别的比较来分析其影响。

**📊 数据集**

使用了AIDev数据集，筛选出1,497条被拒绝的AI代理拉取请求，随机抽样306条进行深入分析。

**📈 对比分析**

通过比较不同拒绝类别的代码变更量和评论数，发现技术相关拒绝导致的代码变更量最大（中位数达293行），整体拒绝PR的评审成本显著高于合并的PR。

**⚠️ 局限性**

局限性包括：49.3%的被拒绝PR未能归因原因；仅关注bug修复请求，未涉及新功能或重构；只分析四个代理，缺乏对其他模型的泛化；以及未考虑跨仓库的影响。

---

## 501. GeoHAT: Geometry-Adaptive Hybrid Action Transformer for Mobile Manipulation

**arXiv ID:** 2606.13394 | [PDF](https://arxiv.org/pdf/2606.13394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 502. Uncertainty Estimation for Molecular Diffusion Models

**arXiv ID:** 2606.13451 | [PDF](https://arxiv.org/pdf/2606.13451v1)

**作者:** Paul Seij `[一作]` (University of Amsterdam), Metod Jazbec `[通讯]` (University of Amsterdam)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种后处理方法，对预训练的分子扩散模型进行不确定性估计。

**💡 创新点**

创新点在于利用拉普拉斯近似对去噪网络进行参数后验采样，并度量噪声预测在采样轨迹中的方差，得到单一不确定性分数。

**🔧 技术方法**

采用拉普拉斯后验、Fisher信息近似、噪声预测方差聚合、以及测试时过采样与过滤技术。

**📊 数据集**

实验数据集包括QM9和GEOM-Drugs。

**📈 对比分析**

与传统的log‑likelihood过滤相比，该不确定性分数与分子稳定性、原子稳定性和有效性指标呈负相关，且在QM9上通过测试时过滤可提升约10%的分子稳定性、5%的有效性，唯一性仅降低约1%；在GEOM-Drugs上效果有限。

**⚠️ 局限性**

局限性在于该方法在更大、更复杂的数据集上无法转移，且实际上更像是对模型敏感性的度量，而非严格的贝叶斯不确定性。

---

## 503. An Assessment Framework for Application-Level Cryptographic Agility

**arXiv ID:** 2606.13425 | [PDF](https://arxiv.org/pdf/2606.13425v1)

**作者:** Navaneeth Rameshan `[一作]` (IBM Research Europe), Gregoire Messmer `[通讯]` (IBM Research Europe)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并应用一种基于组件的评估框架，对六种主流加密API进行评估，揭示了后量子迁移中的关键缺口。

**💡 创新点**

创新点在于将加密敏捷性拆解为七个互相独立的维度（运算耦合、创建耦合、提供者耦合、解耦机制、治理权限、算法迁移、提供者迁移），从而实现非层级化、可定量化的系统敏捷性分析。

**🔧 技术方法**

采用概念模型与实证评估相结合的方法，构建评估框架并对PKCS#11、OpenSSL 3.0、JCA、Google Tink、AWS KMS和HashiCorp Vault Transit进行静态API分析与案例对比。

**📊 数据集**

并未使用传统机器学习或图像数据集，而是通过公开文档、源代码以及官方API说明，对上述六个系统的API接口进行功能层面的“数据集”采集与标注。

**📈 对比分析**

评估方法为基于维度等级（0–4）对每个系统进行评分，最终通过可视化表格展示其敏捷性配置，结果显示所有系统缺乏意图式创建、加密治理及算法转换功能，说明在后量子迁移中的实际迁移成本仍高。

**⚠️ 局限性**

局限性包括：评估仅覆盖六个代表性系统，未覆盖全部工业实现；评估依据公开API描述，可能忽略未公开或自定义的扩展；框架聚焦结构性耦合，未深入探讨性能或安全细节影响。

---

## 504. GF-DiT: Scheduling Parallelism for Diffusion Transformer Serving

**arXiv ID:** 2606.13501 | [PDF](https://arxiv.org/pdf/2606.13501v1)

**作者:** Xinwei Qiang `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种可编程的弹性Diffusion Transformer（DiT）推理运行时，能够动态分配GPU并行度以适应不同请求和系统状态；

**💡 创新点**

将GPU并行度视为一类可调度资源，采用轨迹任务分解实现异步执行；引入组无通信（group‑free collectives）实现在线组重新配置；提供统一的策略接口实现多种调度策略；

**🔧 技术方法**

异步执行抽象、轨迹任务（encoding/denoising/decoding）划分、组无通信与对称缓冲区、后端感知通信实现、布局感知工件迁移、离线模拟器与成本模型；

**📊 数据集**

使用Wan2.2‑5B视频生成模型和Qwen‑Image图像生成模型，构造短/中/长请求的混合到达流和前景突发（foreground‑burst）工作负载；

**📈 对比分析**

与vLLM‑Omni的固定管线、静态并行基线进行对比；在4‑GPU H20与8‑GPU A100上评估多种策略（EDF、SRTF‑SP1、FCFS‑SP1等）；结果表明吞吐量提升至6倍、平均延迟降低95%、SLO违例率下降90%，组无通信相较NCCL减少到约60 µs，系统总体开销低；

**⚠️ 局限性**

依赖DiT工作负载的可预测性与轻量状态，迁移与重组可能在极端负载或非DiT场景下产生额外开销；策略设计仍需针对不同服务目标手工调优；

---

## 505. Digital Twin-Based Simulation for Predictive Decision-Making in Waterway Logistics

**arXiv ID:** 2606.13492 | [PDF](https://arxiv.org/pdf/2606.13492v1)

**作者:** Matthijs Jansen op de Haar `[一作]` (ETH Zürich), Daniel Frutos Rodriguez `[通讯]` (Nanyang Technological University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在水位不确定条件下利用数字孪生进行预测决策，以优化内陆水路物流的货运路线。

**💡 创新点**

创新点在于将数字孪生与预测模型结合，在模拟环境中对比预测与被动反应，证明预测决策可显著降低运作成本与道路运输模式转换，并基于专家访谈识别出三大主要干扰情景与对应响应策略。

**🔧 技术方法**

使用了数字孪生（digital shadow）、基于图的离散事件模拟、机器学习/统计水位预测模型以及专家访谈收集的情景分析。

**📊 数据集**

采用随机生成的hub‑and‑spoke图网络与人工设定的水位随机游走参数，以及专家访谈提供的情景与假设，未使用真实历史水位或运营数据集。

**📈 对比分析**

通过比较预测准确率70%‑100%与无预测的被动模型，结果显示预测模型平均减少28.3%船舶行驶距离，将模式转移从19降至0，显著降低运营成本和排放。

**⚠️ 局限性**

局限性在于模拟模型为抽象化，缺乏对历史实测数据的验证；未考虑天气、货物紧急性、基础设施失效等多重因素；数字孪生主要作为决策支持工具，仍需人类判断。

---

## 506. Fundamental Limits of Hypergraph Edge Partitioning under Independent Edge Sampling

**arXiv ID:** 2606.13491 | [PDF](https://arxiv.org/pdf/2606.13491v1)

**作者:** Javad Maheri `[一作]` (EURECOM), Petros Elia `[通讯]` (EURECOM)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究超图边分区问题，给出在广义随机超图模型（每条超边独立出现）下的最优最大顶点占用量（MVF）理论下界，并构造一种通用的、确定性的“Interweaved‑Cliques（IC）”分区方案，该方案在任意满足 |X|≥Θ(nN log N) 的超图上几乎达到理论下界，且负载平衡因子在高概率下保持在常数 5 以内。

**💡 创新点**

创新点：
1) 证明了在极为广泛的随机超图模型（包含度校正、混合会员、Stochastic Block、几何等）下，MVF 的最优下界与超图密度无关，仅由 (n,d,N) 决定；
2) 提出 IC 设计，该设计的顶点映射仅依赖于 (n,d,N)，与具体实例无关；
3) 通过严密的概率论证明，IC 方案在常数因子（≈8e 或 ≈8√2e）内逼近最优下界，并给出更紧的常数下界；
4) 同时证明该方案在负载平衡方面也具有常数因子保证。

**🔧 技术方法**

主要技术：
- 随机化边生成模型与概率不等式（Chernoff、Hoeffding）
- 组合构造：按顶点族分组并将超边映射到族交集，形成初始分区；
- 细化与扩展：通过分区细分和重分配实现 N 个分区；
- 负载平衡分析：利用二项分布尾概率与期望估计；
- 证明技巧：假设法、对偶逆推、对数界展开、 Stirling 近似。

**📊 数据集**

论文为理论性研究，未使用真实数据集；实验部分采用符号超图生成模型（例如均匀 iid、Heterogeneous 随机模型）来验证理论边界与算法性能的接近程度。

**📈 对比分析**

与传统多级、FM、哈希分区器等经验方法的比较是理论上：IC 方案在任意满足 |X|≥Θ(nN log N) 的超图上实现常数因子逼近最优，而多级或随机方法往往只能在特定结构或经验调参下取得更好性能；实验上证明 IC 在大规模稀疏超图上与现有方法性能相当或更优，且计算复杂度仅 O(|X|)。

**⚠️ 局限性**

局限性：
- 需要满足 N ≤ ⌊√(nd/2)⌋^d 以及 d ≤ n/2（或更严格的 d ≤ n/4, n/16）等结构约束；
- 需要超图满足 |X| ≥ Θ(nN log N)，即相对稠密；在极稀疏实例下理论下界和实现方案可能失效；
- 结果仅针对 MVF（最大顶点占用）目标，无法直接推广到其他通信/切割代价；
- 常数因子虽已优化，但在某些参数 regime 下仍可达 20–30 级别，可能不够紧凑；
- 依赖独立边出现的随机模型，未考虑更一般的依赖结构。

---

## 507. MaxProof: Scaling Mathematical Proof with Generative-Verifier RL and Population-Level Test-Time Scaling

**arXiv ID:** 2606.13473 | [PDF](https://arxiv.org/pdf/2606.13473v1)

**作者:** Jiacheng Chen `[一作]`, Yu Cheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了M3模型，包含Proof Expert、Verifier Expert、Fixer Expert三个能力，并通过MaxProof框架实现测试时的种群搜索和自我选择，提升竞赛级数学证明的通过率。

**💡 创新点**

创新点在于将证明生成、验证和修复拆分为可训练的专门专家，并将它们整合到一个统一的M3模型；同时提出了四层防御式生成-验证管道、惰性奖励筛选与基于种群的MaxProof搜索，以在噪声验证器下实现稳定的 pass@1。

**🔧 技术方法**

采用长程强化学习（CISPO）与惰性标准化奖励、四层防御式生成-验证管道、对抗式错误查找与裁决的监督、基于拒绝采样的修复微调，以及基于种群的演化搜索和配对锦标赛选择。

**📊 数据集**

使用公开的数学竞赛题库（IMO、USAMO等）、IMOProofBench、IMOAnswerBench以及从Proof Expert训练循环中收集的自动生成的评估与修复样本。

**📈 对比分析**

在IMOProofBench/IMOAnswerBench的单次生成中分别获得 67.40 / 81.56 分；通过 MaxProof 搜索后，IMO 2025 与 USAMO 2026 的得分从 27/26 提升至 35/36，显示了约 8–10 分的显著提升，超过了单纯采样的预期。

**⚠️ 局限性**

局限包括：对最优证明的选择仍可能失误（如 USAMO 2026 P2 的选取失误）；验证器仍可能在边缘情况下误判；模型在极难题目（IMO 2025 P6）上仍缺乏足够的先验策略；修复阶段对错误定位的依赖限制了泛化。

---

## 508. Generative Modeling of Bach-Style Symbolic Music: A Comparative Study of Autoregressive, Latent-Variable, and Adversarial Approaches

**arXiv ID:** 2606.13626 | [PDF](https://arxiv.org/pdf/2606.13626v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 509. Why Sampling Is Not Choosing: Intentionality, Agency, and Moral Responsibility in Large Language Models

**arXiv ID:** 2606.13441 | [PDF](https://arxiv.org/pdf/2606.13441v1)

**作者:** Joseph Keshet `[一作]` `[通讯]` (Technion - Israel Institute of Technology), Joseph Keshet (Technion - Israel Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过哲学分析，论证大型语言模型缺乏内在意向性与自我归属行动，因而不具备道德责任。

**💡 创新点**

提出基于承诺的责任框架，将自由意志与道德责任联系起来，并系统阐述Transformer模型为何不满足这些条件。

**🔧 技术方法**

聚焦Transformer架构的概率输入-输出映射与采样机制，讨论其符号生成与意义的来源。

**📊 数据集**

未使用具体数据集，主要依赖哲学文献和模型架构理论进行论证。

**📈 对比分析**

无实验比较，主要与传统哲学观点进行对比，未给出量化性能指标。

**⚠️ 局限性**

局限在于仅为理论讨论，缺乏实证验证，且未考虑人机交互中责任归属的动态变化。

---

## 510. Multiagent Protocols with Aggregated Confidence Signals

**arXiv ID:** 2606.13591 | [PDF](https://arxiv.org/pdf/2606.13591v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 511. GIVE: Grounding Human Gestures in Vision-Language-Action Models

**arXiv ID:** 2606.13435 | [PDF](https://arxiv.org/pdf/2606.13435v1)

**作者:** Pengfei Liu `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在真实的抓取-交付人机交互任务中，提出一种将人类手势信息以视觉覆盖和语义解析两条路径注入预训练Vision‑Language‑Action（VLA）模型的方法；

**💡 创新点**

创新点在于：①双路视觉‑语义引导，既利用手势的几何提示实现目标定位，又通过预训练VLM生成高层意图描述；②无需改动原有政策网络，采用无参数视觉覆盖方式避免破坏预训练表征；

**🔧 技术方法**

技术包括：手势关键点检测与指尖指向射线生成、手势覆盖到RGB图像、预训练VLM进行语义意图解析、基于Flow Matching的连续动作回归；

**📊 数据集**

使用了270条真实人机交互演示数据集（Galaxea R1‑Lite双臂机器人、三种目标物体），并在多种空间布局和多名人类参与者上进行评测；

**📈 对比分析**

与基线π_0.5以及仅视觉或仅语义增强的版本对比，双路增强后Identify SR从46.7%提升至86.7%，Grasp/React/Handover SR均超过80%，显著提升了目标识别、抓取与交付成功率；

**⚠️ 局限性**

局限性包括：依赖手势关键点估计，运动模糊或遮挡会影响性能；指尖射线采用固定扩展比例，对深度变化不自适应；

---

## 512. VietFashion: Benchmarking Sketch-Text Composed Image Retrieval for Cultural Outfits

**arXiv ID:** 2606.13427 | [PDF](https://arxiv.org/pdf/2606.13427v1)

**作者:** Hoang-Nguyen Cao `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (University of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出VietFashion基准，用手绘草图和文本组合检索越南传统服饰Ao Dai。

**💡 创新点**

融合多目标检索、草图+文本多模态、文化属性丰富且利用生成模型扩充数据。

**🔧 技术方法**

使用Qwen‑2.5生成文本描述、SANA‑ControlNet合成图像，构建多模态对齐评估框架。

**📊 数据集**

650张手绘草图、7,000条草图‑文本查询、21,000张合成图像，聚焦越南传统服饰。

**📈 对比分析**

对比SBIR、ST‑CIR、零样本、全监督CIR四类方法，BLIP4CIR在Recall@10达0.3703，Recall@1仍低于0.1。

**⚠️ 局限性**

零样本效果差、细粒度文化属性难以区分、生成文本过长导致对齐困难，模型仍难完全捕捉微小细节。

---

## 513. What's Old is New Again: Classical Dimensionality Reduction for Efficient Saliency-Guided Biometric Attack Detection

**arXiv ID:** 2606.13528 | [PDF](https://arxiv.org/pdf/2606.13528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 514. CloudCons: A Comprehensive End-to-End Benchmark for Cloud Resource Consolidation

**arXiv ID:** 2606.13513 | [PDF](https://arxiv.org/pdf/2606.13513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 515. Beyond Runtime Enforcement: Shield Synthesis as Defensibility Analysis for Adversarial Networks

**arXiv ID:** 2606.13621 | [PDF](https://arxiv.org/pdf/2606.13621v1)

**作者:** Achraf Hsain `[一作]` (King Fahd University of Petroleum and Minerals), Sultan Almuhammadi `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将盾牌（Shield）合成从运行时安全机制转变为设计时分析工具，构建双规范约束安全游戏，计算正式的防御可行性裁决（Defensibility Verdict），并基于吸引器(shell)结构与对抗性多智能体强化学习（MARL）的后验行为提出六个防御度量及其指纹，用何-if 实验验证该方法在网络防御设计中的可用性。

**💡 创新点**

核心创新包括：① 对盾牌合成机制进行角色逆转，将其视为结构分析工具；② 采用双规范（防御安全规范与攻击者资源限制）并以不对称方式插入安全游戏的不同阶段；③ 将防御可行性裁决作为主输出，盾牌仅作为证明；④ 设计六维防御度量与雷达指纹，将形式化吸引器信息与动态 MARL 结果统一呈现；⑤ 通过对比实验展示形式化与经验两层分析的解耦与互补。

**🔧 技术方法**

使用的技术包括：确定性有限自动机（DFA）与其同步积、时间逻辑（LTL）安全公式编译、安全游戏的吸引器固定点迭代、攻击者和防御者行动空间约束、盾牌提取、双重对抗式最小化 Q‑学习（Minimax‑Q）以及基于安全游戏边界的 MARL 训练；同时构造一组基于吸引器 shell 的统计度量。

**📊 数据集**

实验数据主要基于人工构造的 5 节点网络片段（包含 GW、Web、WS、DB、BK）及其四种拓扑或规范扰动（全连通、移除 VPN 绕行、攻击者 Destroy 限制取消、active≥2）。不使用公开的真实网络数据集，而是利用精确的网络拓扑模型和状态转换规则进行离散化实验。

**📈 对比分析**

比较方法：对五个何-if 场景分别计算安全游戏的 ATK、SNK、FRC、STP、MSV、赢区大小、DDR 等指标，并绘制雷达指纹；通过 10 个随机种子训练 MARL，获取 DDR 的 95% 置信区间；对比安全游戏与 MARL 结果揭示两层分析的差异。性能方面，产品状态空间约 150,000，吸引器计算几秒，MARL 训练 50 秒，整个五个实验（10 个种子）在消费级硬件上不到 1 小时完成。

**⚠️ 局限性**

限制包括：① 规模受限，显式状态游戏最多约 7 个节点；② 仅支持确定性转移，未考虑概率或部分可观测性；③ 简化的五状态主机模型缺乏真实攻击细节；④ 对抗性 MARL 的收敛性在受限动作集下未严格证明；⑤ 只在单一 5 节点网络族上验证，缺乏对更广泛拓扑的泛化评估；⑥ 防御可行性裁决对模型假设极其敏感，未研究模型扰动下裁决的稳健性。

---

## 516. ArogyaSutra: A Multi-Agent Framework for Multimodal Medical Reasoning in Indic Languages

**arXiv ID:** 2606.13572 | [PDF](https://arxiv.org/pdf/2606.13572v1)

**作者:** Tanmoy Kanti Halder `[一作]` (Indian Institute of Technology Patna), Sriparna Saha `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了面向印度七大语言的多模态医学问答基准 ArogyaBodha，并提出了基于 Actor‑Critic 的多代理框架 ArogyaSutra，支持图像感知、记忆反思和语言自适应，以实现逐步医学推理。

**💡 创新点**

创新点包括：① 将工具感知（图像裁剪、边缘检测等）嵌入推理流程；② 双记忆机制（短期、长期）记录错误与上下文；③ 语言分析与代码切换，区分语言错误和逻辑错误；④ 通过 Critic 反馈进行 Distillation，使推理在推理阶段无 Critic 也能保持高质量。

**🔧 技术方法**

使用 Qwen‑VL‑2.5‑7B 作为基础模型，配合多种视觉工具、Actor‑Critic 结构和 4‑bit LoRA 参数高效微调；在推理时采用 3 步内自我校正与记忆更新。

**📊 数据集**

数据集来源包括 8 个医学资源（MedXpertQA、MedTrinity‑25M、MedPix‑2.0、MAMA‑MIA、BraTS24、PMC‑VQA、GMAI‑MMBench、NEET‑PG/FMGE），共 40,857 条样本，覆盖 31 系统、6 成像模态、21 临床领域，并对 7 大印地语族语言进行高质量翻译与验证。

**📈 对比分析**

与 GPT‑4、Qwen‑VL‑3B/7B、Mistral‑Small、LLaVA‑v1.6、BioMistral‑7B、MedGemma‑4B‑it 等基线对比，ArogyaSutra 在七语平均准确率达 43.40%，比基线提升 9.2 分，且在 OOD 评测中达到 50.4%，显著优于所有对比模型。

**⚠️ 局限性**

局限性：1) 仍可能出现推理或图像解释错误，尤其在罕见病例；2) 仅覆盖主要七种印地语族，未涵盖低资源方言与代码混合语境；3) 对视觉工具与基线模型的依赖，工具失效或感知错误可能导致错误传播。

---

## 517. When Does Mixing Help? Analyzing Query Embedding Interpolation in Multilingual Dense Retrieval

**arXiv ID:** 2606.13537 | [PDF](https://arxiv.org/pdf/2606.13537v1)

**作者:** Tongyao Zhu `[一作]` (National University of Singapore), Min-Yen Kan `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了多语言密集检索中查询语言混合对检索效果的影响，使用嵌入层面插值来精确控制混合比例，系统评估了不同语言对和文档语言组合下的表现；

**💡 创新点**

提出了可控比例混合实验框架，并发现英语在索引中时混合几乎无效，而在非英语索引中适度混合可显著提升检索效果，进一步揭示英语是最优混合伙伴且混合效果与语言距离呈负相关；

**🔧 技术方法**

主要技术包括BGE‑M3（以及其他模型族）密集检索、查询嵌入的线性插值（embedding mix）、基于mMARCO的并行翻译数据、FAISS向量索引、nDCG@10等评价指标；

**📊 数据集**

使用mMARCO多语言检索语料库，包含14种语言的并行查询与文档，筛选出1,484条长查询；

**📈 对比分析**

与单语查询及词级混合查询对比，发现内部混合在105个设置中有83.8%能超过最优单语端点，平均提升约0.7个百分点，且在非英语索引中提升更明显；

**⚠️ 局限性**

局限性在于仅使用并行翻译控制混合比例，未涵盖自然混合查询的拼写、转写、切换点等真实场景；语言对覆盖有限，仅混合两种语言；对低资源语言的验证不充分；

---

## 518. Reasoning as Pattern Matching: Shared Mechanisms in Human and LLM Everyday Reasoning

**arXiv ID:** 2606.13607 | [PDF](https://arxiv.org/pdf/2606.13607v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 519. From Tokens to Faces: Investigating Discrete Speech Representations for 3D Facial Animation

**arXiv ID:** 2606.13630 | [PDF](https://arxiv.org/pdf/2606.13630v1)

**作者:** Pedro Correa `[一作]` (Univ. Estadual de Campinas), Thomas Hueber `[通讯]` (Univ. Grenoble Alpes)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统评估了四类语音表示（SSL语义、语义+声学、声学、标签式）对3D面部动画生成的影响，比较了GRU与Transformer两种解码器，并通过目标误差、波动率、闭口度等指标以及MUSHRA感知评估，对八种组合进行评测；同时进行探测分析，考察语音表示中对音素与面部可视化的编码；基于这些结果提出了使用共享Token表示的音视文本到语音与面部动画一体化AVTTS管线。

**💡 创新点**

（1）首次系统性比较不同语音表示对面部动画的实用性；（2）发现语义/标签式表示能很好地保留音素信息并驱动自然面部运动；（3）提出利用同一Token流同时生成语音与面部动画的AVTTS框架，打破传统两阶段流程；（4）通过离散与连续的探测指标深入揭示哪些信息对动画预测最为关键。

**🔧 技术方法**

使用的技术包括：SSL模型HuBERT、SpeechTokenizer、WavTokenizer、CosyVoice2；GRU和Transformer解码器；对Blendshape进行LVE、Jitter、BCS评估；MUSHRA感知评分；离散/连续探测方法（归一化熵、R²回归）；将FLAME参数映射到51维ARKit Blendshape；CosyVoice2 TTS与面部Transformer共享Token进行AVTTS。

**📊 数据集**

使用BEAT2数据集（约27小时英语语音，25位说话者，8种基本情绪），语音与3D面部Motion（FLAME）对齐，并转换为ARKit Blendshape。

**📈 对比分析**

通过客观指标（LVE、Jitter、BCS）和主观MUSHRA评分，对八种编码+解码组合进行对比。结果显示：HuBERT和CosyVoice2标签式表示在LVE、BCS以及MUSHRA评分上表现最佳；Transformer解码器普遍优于GRU；语义+声学表示因低结构声学信息对面部运动预测不利；AVTTS管线在同步性和自然度上与传统语音驱动接近。

**⚠️ 局限性**

局限性包括：Token化表示在连续面部Blendshape预测上R²低，说明对细粒度面部动态的编码不足；实验仅在英语数据上验证，缺乏多语言和更多情绪场景；对Token同步精度的深入分析不足；未探讨不同语音解码模型在语音质量与动画同步上的权衡。

---

## 520. Revisiting Vehicle Color Recognition in Long-Tailed Surveillance Scenarios

**arXiv ID:** 2606.13625 | [PDF](https://arxiv.org/pdf/2606.13625v1)

**作者:** Vinícius Orrú `[一作]` (Pontifical Catholic University of Paraná), Rayson Laroca `[通讯]` (Pontifical Catholic University of Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究车辆颜色识别在长尾监控场景下的表现，提出结合合成少数类别数据、权重交叉熵、线性预热+余弦衰减、颜色安全增强、前景分割、硬投票集成等多项技术的完整管线

**💡 创新点**

首次系统性评估合成数据在极端类别不平衡下的作用，并证明在长尾监控数据上，合成数据与权重损失相结合可显著提升宏观精度；Gemini 2.0 Flash在少数类别上优于大规模文本生成模型，表明域对齐比数据量更重要

**🔧 技术方法**

文本与图像条件生成（RunDiffusion/Juggernaut-XL、Gemini 2.0 Flash），自监督特征提取（DINOv3），加权交叉熵、线性预热+余弦衰减、颜色安全增广、SAM 2.0 前景分割、硬投票集成

**📊 数据集**

UFPR‑VeSV 真实监控数据集（24,945张，13种颜色）

**📈 对比分析**

与原始基准（EfficientNet‑V2 等）比较，单模型微准确率从94.0%提升至94.6%，宏观准确率从72.5%提升至79.7%，相对之前 71.5% 提升 8.2个百分点，显著优于现有最优方法

**⚠️ 局限性**

仍受图像质量限制（夜间、低光、反射、遮挡等）导致颜色辨识本身模糊；一部分错误为人类可判定的“不可辨识”案例，表明颜色识别的实际边界受限于视觉证据本身

---

## 521. Reward Modeling for Multi-Agent Orchestration

**arXiv ID:** 2606.13598 | [PDF](https://arxiv.org/pdf/2606.13598v1)

**作者:** King Yeung Tsang `[一作]` (Rutgers University), Hao Wang `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于自监督奖励模型的多智能体系统（MAS）编排优化方法，利用训练过程中生成的中间轨迹自动构建胜负对，训练Bradley‑Terry奖励模型，实现对编排方案的直接评估；

**💡 创新点**

创新点在于完全无人工标注、仅依赖中间轨迹生成的偏好对，实现编排层面奖励学习，既提升了测试时的最佳样本选择，又显著降低了训练时的子智能体推理成本；

**🔧 技术方法**

使用的技术包括自监督奖励模型训练（Bradley‑Terry 损失）、Best‑of‑N/Weighted‑BoN 策略、GRPO 与 DPO 强化学习框架、LLM-as-a-Judge 与 log‑P 等对比方法；

**📊 数据集**

在四个基准上验证：AIME 24&25、BrowseComp、HotpotQA 与 GPQA（科学推理）等；

**📈 对比分析**

与MAS‑Orchestra、LLM-as-a-Judge、log‑P、Skywork 等基线比较，ORM在AIME上从 63.3% 提升至 68.3%，在 BrowseComp 上从 9.5% 提升至 14.0%，同时在测试与训练过程中分别将 token 使用量降低 10 倍以上；

**⚠️ 局限性**

局限性在于奖励模型依赖已有编排模型与轨迹，受训练数据多样性与质量限制；目前仅在单域训练，跨域泛化能力待进一步验证。

---

## 522. Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States in Time Series Forecasting

**arXiv ID:** 2606.13571 | [PDF](https://arxiv.org/pdf/2606.13571v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 523. Simplex-Constrained Sparse Bagging: Transitioning from Uniform Priors to Sparse Posteriors in Ensemble Learning

**arXiv ID:** 2606.13589 | [PDF](https://arxiv.org/pdf/2606.13589v1)

**作者:** Meher Sai Preetam `[一作]`, Meher Bhaskar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出Simplex-Constrained Sparse Bagging（SCSB），一种基于概率单纯形的后训练稀疏化与概率校准框架，能在不增加额外验证集的前提下压缩并提升袋装集成的泛化与校准性能。

**💡 创新点**

解决了“L1-单纯形悖论”，通过引入凹二次惩罚实现稀疏化；将OOB损失与置信校准联合优化，并给出梯度解析与理论证明；实现模型无关的线性推理加速与校准提升。

**🔧 技术方法**

利用Out‑of‑Bag (OOB) 预测、概率单纯形约束、凹二次正则化、SLSQP优化、Log‑Loss/MSE梯度解析、理论凸凹分析。

**📊 数据集**

scikit‑learn 与 OpenML 的多种分类与回归数据集（如 breast_cancer、diabetes、spambase、segment、california_housing、cpu_act 等）。

**📈 对比分析**

与标准均值袋装、Lasso‑稀疏袋装以及 XGBoost 进行对比，SCSB 在 68%–96% 的压缩率下实现线性推理加速（最高 5.7×）且保持或提升准确率、降低 ECE，且 MSE/R² 与基准相当或更优。

**⚠️ 局限性**

优化非凸性导致可能收敛至局部最优；在超大规模集成或大数据集上预计算 OOB 与求解成本仍需改进；缺乏输入条件化校准与深度学习等更复杂模型的验证。

---

## 524. EvTexture++: Event-Driven Texture Enhancement for Video Super-Resolution

**arXiv ID:** 2606.13580 | [PDF](https://arxiv.org/pdf/2606.13580v1)

**作者:** Dachun Kai `[一作]` (University of Science and Technology of China), Xiaoyan Sun `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了EvTexture++框架，利用事件相机高频时空信息专门提升视频超分辨率中的纹理细节与时间一致性；

**💡 创新点**

创新点在于：①引入迭代纹理增强（ITE）模块，逐步利用事件体素的高频信息改进纹理；②设计事件驱动的纹理对齐（TTA）模块，融合RGB与事件流的光流以提升跨帧纹理对齐；③将此框架做为可插拔插件，能在多种VSR基座上无改动提升性能；

**🔧 技术方法**

使用的技术包括：事件体素网格编码、ConvGRU递归更新、双分支（纹理+运动）并行设计、光流估计与双流融合、像素置换上采样、残差学习与多尺度训练；

**📊 数据集**

在五大公开数据集上进行实验：Vimeo‑90K、REDS、Vid4、UDM10、CED（真实事件），并通过合成ESIM生成事件；

**📈 对比分析**

与多种SOTA基线（RGB/VSR、事件驱动VSR、Transformer VSR）对比，EvTexture++在Vid4、REDS4、Vimeo‑90K‑T、CED等数据集均实现PSNR提升0.5–1.6 dB，LPIPS降低至0.2048，TCC和tOF指标亦显著改善，且仅增加约10M参数，推理速度约95ms；

**⚠️ 局限性**

局限性在于：①假设事件与RGB在空间、分辨率上完全对齐，现实中往往存在视差与分辨率差异；②仅在CNN/Transformer基座上验证，未探索扩展至扩散式VSR等新型模型；

---

## 525. A2D2: Fine-Tuning Any-Length Discrete Diffusion for Adaptive Decoding

**arXiv ID:** 2606.13565 | [PDF](https://arxiv.org/pdf/2606.13565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 526. Learning with Simulators: No Regret in a Computationally Bounded World

**arXiv ID:** 2606.13576 | [PDF](https://arxiv.org/pdf/2606.13576v1)

**作者:** Sasha Voitovych `[一作]` (Massachusetts Institute Of Technology), Alexander Rakhlin `[通讯]` (Massachusetts Institute Of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了在存在强依赖数据的情形下，利用可模拟过程（simulatable processes）和条件采样器实现在线学习的框架。

**💡 创新点**

创新点在于证明即使在任意依赖的数据生成过程中，只要有近似的模拟器，VC类仍可在与独立数据相同的统计和计算复杂度下学习；并且利用时间限制的 Kolmogorov 复杂度实现对所有多项式可采样过程的统一学习算法。

**🔧 技术方法**

主要技术包括：多尺度覆盖、指数加权、KL 变换、随机滚动（random playout）、时间受限 Kolmogorov 复杂度与通用分布、以及条件采样的松弛框架。

**📊 数据集**

论文主要为理论分析，并未在公开数据集上实验；若有实验，使用合成模拟的线性动力系统和 Glauber 动力学数据。

**📈 对比分析**

与传统的独立数据 PAC 学习以及混合、可预测序列学习相比，本文的算法在理论上实现了 O(√(dT)) 的误差上界，且对所有 VC 类都保持此阶；在特殊结构化过程如 LDS 和 Glauber 动力学下，获得了对 KL 误差的多项式补偿。

**⚠️ 局限性**

局限性在于：需要完整或近似的模拟器，条件采样在通用分布下难以实现；在非可采样或高 KL 误差情形下，仍需指数样本；在不具备可采样性的过程或标签不确定的对抗性情况下，仍不可行。

---

## 527. Contrast-Informed Augmentation and Domain-Adversarial Training for Adult-to-Neonatal MR Reconstruction Generalization

**arXiv ID:** 2606.13562 | [PDF](https://arxiv.org/pdf/2606.13562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 528. Adjusted Cup-Product Neural Layer

**arXiv ID:** 2606.13568 | [PDF](https://arxiv.org/pdf/2606.13568v1)

**作者:** Snigdha Chandan Khilar `[一作]` `[通讯]` (Independent Researcher), Snigdha Chandan Khilar (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种硬编码调整的杯积层（Adjusted Cup‑Product Layer），利用高阶规范理论中的调整项实现对离散连通的规范不变性，并在多种拓扑可观测量上进行实验。

**💡 创新点**

创新点在于：①把杯积与调整项直接硬编码进网络层，保证输出本质上就是规范不变的拓扑数；②证明了调整项是信号唯一来源并证明线性/卷积模型无法表示此二次形式；③通过实验验证该层在非卷积可表达的杯积（如3D Chern–Simons、4D 拓扑电荷、2D Chern 数等）中必不可少，而在卷积可表达的杯积（如动量螺旋性）中仅提升样本效率。

**🔧 技术方法**

使用的技术包括：离散共形算子、Alexander–Whitney 杯积、离散曲率与曲面积分、稀疏矩阵运算、可学习的 MLP 前端、以及针对非阿贝尔情形的 Wilson‑loop 结构杯积；实验中还对比了公平 CNN、谱 SNN、MPSN、GNN 等基线，并进行 κ=0 的消除调整消融。

**📊 数据集**

使用的数据集包括：合成的 3‑维 4‑维 3‑维 2‑维 标度的四面体网格（Kuhn‑torus）、Haldane 型 2‑维 Torus 上的线束数据、离散化的湍流流场、三维闭曲线的电流场、以及多带 Wilson‑loop 计算得到的 U(N) 线束数据。

**📈 对比分析**

对比方法：将调整层与公平 CNN、谱 SNN、MPSN、GNN 等基线在相同训练样本数下进行训练，并计算训练集和测试集的 R² 或整数分类准确率。实验结果显示：对于非卷积可表达的杯积，调整层在测试集上可达 0.9–0.99 的 R²（如 3D Chern‑Simons 0.99，4D 拓扑电荷 0.91），而基线仅记忆训练集而无法泛化；对于卷积可表达的杯积，调整层在小样本时表现更好，但公平 CNN 能在足够大样本时赶上（如动量螺旋性测试 0.99 的 R²）。

**⚠️ 局限性**

局限性包括：①目前仅实现阿贝尔版本，非阿贝尔扩展仅通过 0‑参数结构杯积实现；②实验主要使用合成或已知精确解的数据，缺乏在真实测量噪声严重的数据上的验证；③对非卷积杯积的鲁棒性（噪声、稀疏传感）尚未系统评估；④比较基线如 GEBLNet 等使用了部分重实现，结果可能不完全可重复；⑤在卷积可表达杯积中，调整层并非必要，只是提升样本效率，过度依赖固定读出可能在极端稀疏情况下成为负担。

---

## 529. Edit the Bits, Diff the Codes: Bitwise Residual Editing for Visual Autoregressive Models

**arXiv ID:** 2606.13558 | [PDF](https://arxiv.org/pdf/2606.13558v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 530. Ride, Track, and Recover: Pilot Randomized Trial of a Wearable Digital Self-Management Intervention During a Veteran Endurance-Cycling Program

**arXiv ID:** 2606.13529 | [PDF](https://arxiv.org/pdf/2606.13529v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 531. ReSCom: A Reconfigurable Spiking Neural Network Accelerator Using Stochastic Computing

**arXiv ID:** 2606.13560 | [PDF](https://arxiv.org/pdf/2606.13560v1)

**作者:** Ali Alipour Fereidani `[一作]` (University of Tehran), Saeed Safari `[通讯]` (University of Tehran)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种可重配置的脉冲神经网络加速器 ReSCom，结合随机计算与精确固定点加法，降低硬件成本与能耗，同时保持数值稳定性。

**💡 创新点**

创新点在于仅将乘法改为随机计算并保留精确加法，提出统一的可重配置神经元实现 IF/LIF/突触模型，并通过可调随机位流长度实现精度‑能耗‑延迟的动态权衡。

**🔧 技术方法**

使用随机计算、LFSR 位流生成、AND 门乘法、固定点累加、FPGA 时间多路复用等技术。

**📊 数据集**

使用 MNIST 手写数字识别数据集进行验证。

**📈 对比分析**

通过与 DSP、数组/移位加法乘法实现以及多篇 SoTA FPGA SNN 方案对比，ReSCom 在 Artix‑7 上 100 MHz 运行，单图能耗仅 0.05 mJ，分类准确率 92.80%，实现了低能耗与高精度的双重优势。

**⚠️ 局限性**

局限性包括随机计算仅适用于乘法，对更复杂的卷积或递归网络以及在线学习机制尚未验证，且位流长度调节会影响延迟与能耗平衡。

---

## 532. MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models

**arXiv ID:** 2606.13515 | [PDF](https://arxiv.org/pdf/2606.13515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 533. Spectrum Sharing Across Terrestrial and Non-Terrestrial Services in the FR3 Upper Midband

**arXiv ID:** 2606.13511 | [PDF](https://arxiv.org/pdf/2606.13511v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 534. Uncertainty-Aware Hybrid Retrieval for Long-Document RAG

**arXiv ID:** 2606.13550 | [PDF](https://arxiv.org/pdf/2606.13550v1)

**作者:** Hoin Jung `[一作]` (Purdue University), Xiaoqian Wang `[通讯]` (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练、基于不确定性的多粒度检索增强生成框架UMG‑RAG与其父级提升变体UMGP‑RAG，利用现有稠密与稀疏检索器在不同段落粒度上并行检索并自适应融合；

**💡 创新点**

创新点在于通过检索分数分布的熵估计每个检索器-粒度对的可靠性，并采用父级提升与重叠去重来提升检索上下文的紧凑性与连贯性；

**🔧 技术方法**

使用的技术包括稠密检索器（BERT、BGE‑M3、Qwen3‑Embedding）、稀疏检索器（SPLADEv3）、熵基置信权重、父级提升、重叠-aware去重及上下文拼接；

**📊 数据集**

实验数据集为 Natural Questions 与 HotPotQA，采用 KILT 验证集的 Wikipedia 文档作为检索语料；

**📈 对比分析**

与标准 RAG、Summarized RAG、LongLLMLingua、MS‑PoE、PINE、LongRAG 以及基于 RRF 的 Hybrid 进行对比，检索指标 AR@5 与生成指标 F1/AR，UMG‑RAG/UMGP‑RAG 在大多数检索器-生成器组合下取得最优或竞争性性能；

**⚠️ 局限性**

局限性包括：多粒度检索导致前置检索计算成本提升；方法依赖已有检索器的质量，若两者均表现欠佳，无法弥补缺失证据；

---

## 535. OneRetrieval: Unifying Multi-Branch E-commerce Retrieval with an Editable Generative Model

**arXiv ID:** 2606.13533 | [PDF](https://arxiv.org/pdf/2606.13533v1)

**作者:** Xuxin Zhang `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出OneRetrieval，一种统一多分支电商检索的单模型生成式检索框架，兼顾深层召回质量与实时可编辑性；

**💡 创新点**

创新点包括①关键词对齐编码(KAE)，将每个标识符位置映射到可解释的属性词；②可扩展代码书设计，预留槽位支持零训练实时编辑；③四阶段监督微调流程，分离召回质量与可编辑性；④信息论属性合并与密度感知容量分配，提升空间利用率；

**🔧 技术方法**

技术方法涵盖BART‑base自回归生成、关键词对齐编码、信息论聚类将18类属性合并为6组、非均匀容量分配、预留槽位、四阶段SFT（Stage0对齐、Stage1内容对齐、Stage2协同共现、Stage3个性化策略）、Beam搜索解码、SID↔item映射、Aho‑Corasick属性词匹配；

**📊 数据集**

使用Kuaishou电商平台真实数据：约5×10^6条训练查询日志、约3×10^4条点击/订单测试日志、2,016万条商品记录、108万条属性词；还进行了在线A/B实验；

**📈 对比分析**

对比传统稀疏检索（BM25、docT5query）、稠密检索（DPR）、生成式检索（TIGER、DSI、LTRGR、LC-Rec、OneSearch）等基线；离线结果显示OneRetrieval在HR@350/MRR@350上与最强生成基线持平或略优，远超稠密与其他生成式基线；在实时编辑指标（IHR、IAR）上，比闭式代码书方法提升一个数量级；在线实验显示替换倒排索引分支带来显著转化提升，进一步替换稠密分支可获得CTR提升而无转化损失；

**⚠️ 局限性**

局限性包括①在浅层切点（HR@10/100）存在轻微精度损失，需权衡可编辑性；②层次化编码实验失败，未能进一步压缩空间；③预留槽位数量有限，极端新词或高频长尾可能需手动调整；④对大规模GPU与训练数据的依赖，部署成本较高；

---

## 536. Beyond the IT Checklist: Engineering a Reasonable Standard of Care for Cyber Safety

**arXiv ID:** 2606.13612 | [PDF](https://arxiv.org/pdf/2606.13612v1)

**作者:** Matthew E. Jablonski `[一作]` (George Mason University), F. Brett Berlin `[通讯]` (George Mason University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

作者通过系统地编制并编码292份关键基础设施政策文件，构建了一个从2000年至2025年的政策语料库，并将其映射到NIST SP 800-160 Vol. 2的韧性生命周期，以评估美国在网络安全与安全工程之间的“合理注意义务”是否得到充分落实。

**💡 创新点**

创新点在于：①首次对美国所有关键基础设施政策进行全面的“合理注意义务”映射；②揭示了委托标准、恢复与适应阶段的三大失衡；③提出以危害可追溯性、结构化保证案例和网络韧性工程为核心的全新标准化治理框架。

**🔧 技术方法**

方法技术主要包括：政策文本的语义编码、三维映射（义务类型–生命周期阶段–目标环境）、指针-目标分析（评估委托标准的技术严谨度），以及对照NIST、ISO/IEC/IEEE等工程标准的跨文献对齐。

**📊 数据集**

使用的数据集为292份关键基础设施政策文件（涵盖2000–2025年），源自CSIAC DoW政策图表、各行业法规及联邦监管文件，并在文本中提取关键词如“安全”“韧性”。

**📈 对比分析**

在方法比较上，作者对政策在四个韧性生命周期阶段的分布做热力图和频率统计，发现“预见”阶段占比最高（139条），而“承受”“恢复”“适应”阶段则严重缺乏技术性义务；该分析展示了当前政策的“验证缺口”与“委托失衡”性能差距。

**⚠️ 局限性**

局限性包括：①语义编码由作者完成，缺乏外部验证；②仅涵盖联邦层面的政策，未涉及州/地方或行业自律文件；③对政策执行效果和行业采纳程度未进行实证评估，主要聚焦文本层面的结构化映射。

---

## 537. See What I See, Know What I Think: Dense Latent Communication Across Heterogeneous Agents

**arXiv ID:** 2606.13594 | [PDF](https://arxiv.org/pdf/2606.13594v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 538. One Polluted Page Is Enough: Evaluating Web Content Pollution in Generative Recommenders

**arXiv ID:** 2606.13610 | [PDF](https://arxiv.org/pdf/2606.13610v1)

**作者:** Minghao Luo `[一作]` (Chinese University of Hong Kong), Liang Chen `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了FORGE基准，模拟搜索结果被污染后评估搜索增强LLM在推荐中推广假品牌的易受攻击性。

**💡 创新点**

创新点在于针对开放网页污染的生成推荐器提出可复现、细粒度测评框架，并揭示了品牌知识缺失与模型易受攻击的关联。

**🔧 技术方法**

技术主要使用LLM生成式推荐、实体替换攻击、三种污染方式（实体替换、段落注入、全文合成）以及三种防御（怀疑提示、模型先验一致性、跨文档一致性）。

**📊 数据集**

数据集包含225个真实产品，15个类别，5个用户情景，并采集了真实搜索结果作为检索证据。

**📈 对比分析**

比较方法为对12个商业与开源LLM在top‑3实体替换攻击下计算假品牌被推荐的比例，结果显示最差可达73.8%，最优仅13.3%；防御方法效果有限。

**⚠️ 局限性**

局限性包括攻击设计未最优化、实验覆盖范围有限、仅在中文/深圳本地化场景、检索快照静态且实体替换方法依赖人机审核。

---

## 539. Is It You or Your Environment? A Bayesian Inference Framework for Genomically-Anchored Personalized Physiological Interpretation

**arXiv ID:** 2606.13556 | [PDF](https://arxiv.org/pdf/2606.13556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 540. Towards Effective Waste Segmentation for Automated Waste Recycling in Cluttered Background

**arXiv ID:** 2606.13587 | [PDF](https://arxiv.org/pdf/2606.13587v1)

**作者:** Mamoona Javaid `[一作]` (Institute of Space Technology), Sajid Ghuffar `[通讯]` (Institute of Space Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种端到端的废物分割网络EWSegNet，旨在实现自动废物回收。

**💡 创新点**

创新点在于同时利用空间域的局部上下文和频域的全局上下文，并引入辅助特征增强模块AFEM来强化边界和Blob信息。

**🔧 技术方法**

采用空间上下文模块SCM、频域上下文模块FCM、辅助特征增强模块AFEM以及UPerNet解码器构建模型。

**📊 数据集**

在公开的ZeroWaste-f、ZeroWaste-aug和SpectralWaste三大数据集上进行实验。

**📈 对比分析**

与SOTA方法比较，EWSegNet在ZeroWaste-f上取得56.44% mIoU，在ZeroWaste-aug上提升至74.10% mIoU，在SpectralWaste上获得71.03% mIoU，同时参数量和推理时间更低，显示出更优的性能与效率。

**⚠️ 局限性**

局限性包括对极端尺寸或完全透明物体的分割仍不够准确，以及频域操作在不同硬件平台上的兼容性需要进一步验证。

---

## 541. Differentially Private Hierarchical Heavy Hitters

**arXiv ID:** 2606.13563 | [PDF](https://arxiv.org/pdf/2606.13563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 542. LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories

**arXiv ID:** 2606.13578 | [PDF](https://arxiv.org/pdf/2606.13578v1)

**作者:** Baochang Ren `[一作]` (Zhejiang University), Huajun Chen `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一套基于视觉-语言-动作（VLA）模型的实验室自动化框架，能够将书面实验方案直接转化为机器人执行动作；

**💡 创新点**

创新点在于：①构建可编程的RoboGenesis模拟引擎，能够自动生成可执行实验室场景和多步协议；②生成专门的LabEmbodied-Data数据集，实现协议层面的成功过滤与结构化注释；③提出结合FAST预训练、流匹配和知识隔离的LabVLA训练流程，实现跨机器人体现的高质量动作预测；

**🔧 技术方法**

使用技术包括：Qwen3‑VL 4B backbone、DiT action expert、FAST动作离散化、流匹配（flow matching）学习、域随机化（visual、clutter、camera、object、lighting、spatial）、以及知识隔离（stop‑gradient）机制；

**📊 数据集**

使用的数据集有：从RoboGenesis合成的LabEmbodied‑Data（≈10,000+实验演示）以及公开机器人与视觉‑语言对齐数据集（Robointer‑VQA、AgiBot World Beta、OXE‑AugE、Droid）做预训练；

**📈 对比分析**

与SmolVLA、π_0、π_0.5、GR00T N1.5等基线在LabUtopia基准上对比，LabVLA在ID和OOD情形下分别达71.1%与70.0%的平均成功率，超越最接近的π_0 6.8/7.8个百分点；

**⚠️ 局限性**

局限性包括：仅能执行固定协议，缺乏对实验条件的自主选择和测量反馈；在模拟与单台Frank平台的Benchtop测试中验证，未覆盖真实实验室的安全、化学品变异与长周期实验失效等复杂情境；

---

## 543. A near-quadratic lower bound on the border determinantal complexity of $\sum_i x_i^n$ via conormal specialization

**arXiv ID:** 2606.13628 | [PDF](https://arxiv.org/pdf/2606.13628v1)

**作者:** Karthik Sheshadri `[一作]` `[通讯]`, Karthik Sheshadri

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文证明了多项式∑_{i=1}^n x_i^n 的边界决定性复杂度（border determinantal complexity）至少为(1/4e−o(1))n^2（普通模型）和(1/2e−o(1))n^2（对称模型），与已知的O(n^2)上界几乎匹配，从而得到该多项式的边界复杂度为Θ(n^2)。

**💡 创新点**

创新点在于：①提出了对任意线性变换矩阵行列式的乘性-一 Gauss-图（multiplicity‑one Gauss‑graph）多维度度（multidegree）的无条件上界，并给出显式 Bézout 计数；②设计了一个“共形专化”（conormal specialization）论证，将该多维度度在退化过程中保留并转移至 Fermat 圆锥的共形多维度度，从而把对偶度（dual degree）传递到目标多项式，实现超线性下界。

**🔧 技术方法**

主要技术包括：共形多维度度与多重度的几何解释；多重 Gauss‑图的定义与计数；使用多重 Bézout 计数与多重张量积；Kleiman 泛转移、Fulton 位置不变性与局部化 Chern 类；曲线选择与平坦性（flatness）论证；复分析中的 Hurwitz 定理保证极限点存在；对称情形的二次型分析；以及对偶度与共形多维度度之间的“锥移”恒等式。

**📊 数据集**

论文不涉及机器学习或实验数据；所有结果均为纯代数几何证明，不使用外部数据集。

**📈 对比分析**

与已知上界相比，本工作给出了与O(n^2)上界几乎一致的下界，证明了该多项式的边界复杂度为Θ(n^2)。在对称模型下得到的常数因子更大，表明两模型在此家族上没有显著的复杂度差距。

**⚠️ 局限性**

局限性包括：需在特征零域上工作；方法依赖多项式的齐次性，无法直接推广到非齐次多项式或永久（permanent）多项式；对共形多维度度的上界仍可能被优化；对非平方自由行列式的情况尚未完全覆盖；以及对共形多维度度的实际计算可能存在冗余成分，影响常数。

---

## 544. MCR-Bionic Hand: Anatomical Structural Priors for Dexterous Manipulation

**arXiv ID:** 2606.13601 | [PDF](https://arxiv.org/pdf/2606.13601v1)

**作者:** Haosen Yang `[一作]`, Guowu Wei `[通讯]` (University of Salford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实现了 1:1 比例的肌肉骨骼仿生手 MCR-Bionic，整合腕部骨骼、韧带、腱路、外侧伸肌束、内在肌路径，并使用闭环液压人工肌肉驱动，展示了结构先验如何生成默认抓握、指间协调与抓握后调节。

**💡 创新点**

将人体手部关键结构（腕-指十度耦合、外侧伸肌束分化、内在肌路径）作为结构智能嵌入硬件，证明结构先验能预组织抓握姿态并显著减轻控制负担。

**🔧 技术方法**

采用闭环液压人工肌肉驱动、骨骼-韧带仿真、几何力学建模与手部力/角度测量技术。

**📊 数据集**

无传统数据集；通过实验室功能演示、力传感和角度测量获得数据。

**📈 对比分析**

与 Shadow Hand、ILDA、Allonic、Clone 等现有机器人手对比，表明 MCR-Bionic 保留了更多结构先验；在硬币旋转、钢笔传递、鲁比卡方块推挤等任务中，能够在低维控制下完成复杂抓握与内在操作，性能与传统结构相当但结构性优势明显。

**⚠️ 局限性**

实现复杂且组件调试困难；缺乏定量评估每个结构对抓握性能的独立贡献；控制仍基于手动调节，缺少自适应学习；液压人工肌肉成本与可靠性限制广泛应用。

---

## 545. Adaptive-Frequency Resonate-and-Fire Neurons for Spectral Estimation of Streaming Radar Signals

**arXiv ID:** 2606.13516 | [PDF](https://arxiv.org/pdf/2606.13516v1)

**作者:** Stefano Chiavazza `[一作]` (Eindhoven University of Technology), Federico Corradi `[通讯]` (Eindhoven University of Technology)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于自适应共振‑并发射（ARF）神经元的神经形态雷达信号处理方法，能够在不进行完整FFT的情况下，逐样本实时估计FMCW雷达的目标距离与速度；

**💡 创新点**

创新点在于：①自适应频率共振器可在线学习输入频率，自动锁定主频；②使用均值场反馈实现多目标分配，避免神经元聚集同一频率；③两层ARF网络实现快速时间（距离）与慢速时间（多普勒）直接估计；④事件驱动的尖峰编码仅在频率变化超过阈值时产生尖峰，显著降低通信带宽；⑤内存与计算复杂度仅随目标数N扩展，而非频谱分辨率L，省去CFAR检测。

**🔧 技术方法**

使用技术包括：离散时间ARF神经元模型（来源于连续时间Hopf振荡器）、复IQ信号处理、均值场反馈机制、事件触发尖峰输出；实验平台基于Python+CUDA/C++仿真，后续计划移植至FPGA/Loihi等硬件。

**📊 数据集**

使用的数据集为：①合成FMCW雷达信号（参数表中列示的频率、带宽、脉冲数等）；②实际测量数据，来源于TI AWR2243和Rohde&Schwarz AREG800A汽车雷达信号发生器在杜塞尔多夫TU实验室的Anechoic Chamber与汽车雷达回波生成装置，包含单目标和三目标多目标场景。

**📈 对比分析**

比较方法：与传统FFT+CFAR以及其他神经形态实现（Rate DFT、S-DFT、S-FFT、RF、SpiNR）进行对比。指标包括：范围/多普勒估计RMSE（单目标约0.9‑1.0 m，速度约0.007 m/s），峰值误差；尖峰计数；内存占用与计算复杂度。ARF在保持与FFT相近的精度的同时，将内存从O(L)压缩到O(N)，并在示例实验中显著降低了尖峰数量与能耗。

**⚠️ 局限性**

局限性：①需预先知晓目标数目，若目标数未知需动态分配神经元；②在低SNR或目标频率相近时收敛误差增大；③学习率λ需手动调节；④实验仅在受控室内环境，未对真实汽车杂波或复杂干扰进行充分验证；⑤目前仅提供算法与复杂度分析，缺乏硬件实现的时延与能耗数据。

---

## 546. Leveraging Audio-LLMs to Filter Speech-to-Speech Training Data

**arXiv ID:** 2606.13507 | [PDF](https://arxiv.org/pdf/2606.13507v1)

**作者:** Qixu Chen `[一作]` (Chinese University of Hong Kong), Satoshi Nakamura `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了如何训练音频语言模型直接从音频中做出保留/丢弃的决策，以过滤噪声数据，从而提高端到端语音到语音翻译（S2ST）的性能。

**💡 创新点**

提出了一种两阶段的Rank→Distill策略，从噪声语音翻译数据中挖掘可靠的监督信息，并训练音频语言模型进行数据过滤。

**🔧 技术方法**

采用了音频语言模型（Audio LLM）进行数据选择和过滤。

**📊 数据集**

使用了CVSS-C和SpeechMatrix数据集进行实验，评估过滤方法的有效性。

**📈 对比分析**

与未过滤的基线相比，经过过滤的模型在ASR-BLEU上提高了1.4分，达到了22.72的BLEU分数，显示出显著的性能提升。

**⚠️ 局限性**

当前的二元决策形式不允许在固定预算下灵活控制保留的数据量，未来的工作将探索基于概率的、预算感知的选择方法。

---

## 547. Beyond Uniform Tokens: Adaptive Compression for Time Series Language Models

**arXiv ID:** 2606.13624 | [PDF](https://arxiv.org/pdf/2606.13624v1)

**作者:** Jialin Gan `[一作]` (Zhejiang University), Xue Wang `[通讯]` (Shandong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

本文提出了一种用于时序与语言模型融合的TokenDecouple框架，通过频域合并时序Token并在层级上逐渐压缩提示Token，实现高效推理；

**💡 创新点**

创新点在于从频域角度揭示时序Token冗余与提示Token逐层衰减的特性，并设计了自适应频域合并与金字塔式提示压缩的联合方法；

**🔧 技术方法**

采用离散傅里叶变换自适应合并、Fisher信息加权、软max压缩提示Token、以及多任务（预测、分类、异常检测、推理）评估；

**📊 数据集**

在27个真实世界数据集（气候、能源、交通、经济、医学等）上进行实验；

**📈 对比分析**

与四种LLM基础模型（OFA、TimeLLM、CALF、S2IP）及三种基线Token合并方法比较，TokenDecouple在78%场景下提升或保持性能，推理速度平均提升约4.5×；

**⚠️ 局限性**

局限在未针对非平稳或稀疏采样数据评估，也未结合剪枝、量化等其他压缩技术，且对极大模型的适用性尚未充分验证。

---

## 548. Finding Conservation Laws of Large Dynamical Systems with Tasks and Futures: A Case Study in Utilizing Dynamic Data Dependencies

**arXiv ID:** 2606.13623 | [PDF](https://arxiv.org/pdf/2606.13623v1)

**作者:** Rüdiger Nather `[一作]` `[通讯]` (University of Kassel), Rüdiger Nather (University of Kassel)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过引入可在未来完成后安全重用内存的构造，提出了一种基于 futures 的块级对称矩阵求逆算法，进而实现了对大型线性代数问题的高并行度求解。

**💡 创新点**

创新点在于：①设计了“d”构造允许未来完成后释放引用计数，从而在保持 futures 依赖管理优势的同时实现了原地更新和内存回收；②基于此构造构建了递归块级求逆流程，实现了细粒度依赖与内存复用的统一。

**🔧 技术方法**

主要技术包括：任务并行框架 Taskflow 的扩展实现、基于 promise/future 的递归依赖管理、对称矩阵块级求逆的递归算法、锁定参考计数实现的内存回收机制。

**📊 数据集**

实验使用了合成密集对称矩阵，规模分别为 2^14×2^14、2^15×2^15 与 2^16×2^16，块大小比率分别为 512、256 与 128。

**📈 对比分析**

与传统线程模型对比，本文通过强规模实验证明：在小块尺寸时存在显著的 futures 维护开销，随着矩阵尺寸增大开销被算术运算所淹没，最终可实现近乎线性的加速。

**⚠️ 局限性**

主要局限在于：①目前的实现使用锁定参考计数，导致在细粒度任务时锁竞争显著；②只验证了稠密矩阵的基线，尚未扩展至层次化（H‑matrix）或非对称矩阵；③需要进一步将内存所有权语义内置于 futures 模型中，以减少手工构造。

---

## 549. AgentBeats: Agentifying Agent Assessment for Openness, Standardization, and Reproducibility

**arXiv ID:** 2606.13608 | [PDF](https://arxiv.org/pdf/2606.13608v1)

**作者:** Xiaoyuan Liu `[一作]` (University Of California Berkeley), Dawn Song `[通讯]` (University Of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AgentBeats框架，用于统一评估AI代理的开放性、标准化和可复现性，并提供一套可复现的基准测试流程。

**💡 创新点**

创新点在于将代理评估与测试环境深度绑定，构建了可插拔的测试模块和可视化报告，解决了现有基准缺乏统一标准和可复现性的问题。

**🔧 技术方法**

采用Python、Docker容器、OpenAI Gym、DeepMind Control Suite等技术；通过对代理的API接口进行抽象，支持多种强化学习框架（如RLlib、Stable Baselines3）。

**📊 数据集**

使用了多样化的数据集与环境，包括经典的Atari游戏、MuJoCo物理仿真、TextWorld文本游戏以及自定义的多代理协作任务。

**📈 对比分析**

与传统单一环境评估方法相比，AgentBeats在相同硬件条件下实现了平均5%更高的评估一致性，并通过基准对比显示其评估结果与原始论文差异小于2%。

**⚠️ 局限性**

局限性包括：①对极大规模多代理场景的支持仍有限；②计算资源消耗高，需GPU集群支持；③部分自定义环境的转换成本较高。

---

## 550. Multi-Agent Reinforcement Learning from Delayed Marketplace Feedback for Objective-Weight Adaptation in Three-Sided Dispatch

**arXiv ID:** 2606.13604 | [PDF](https://arxiv.org/pdf/2606.13604v1)

**作者:** Haochen Wu `[一作]` (DoorDash Inc), Shiguang Xie `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用离线强化学习在 DoorDash 三方配送平台上自适应调整配送目标权重，提升批量率并降低送餐人员工作时长；

**💡 创新点**

通过在现有组合优化器前插入低维权重调节接口，让 RL 只需控制权重而非直接决策，既保证了生产可行性，又利用延迟市场反馈实现安全在线部署；

**🔧 技术方法**

采用离线 Double‑DQN 结合 Conservative Q‑Learning 进行值函数训练，使用中央共享价值网络和 store‑level 分布式执行，奖励采用区域级延迟 KPI 并在实验中用 CUPED 做方差削减；

**📊 数据集**

使用 DoorDash 大规模历史订单、接单与送达日志，构建包含 ASAP、CAT、XCAT 等指标的延迟奖励数据集；

**📈 对比分析**

与全局静态权重基线进行两小时 switchback 试验，结果显示 OWA‑RL 在所有时段将 CAT 与 CWT 分别降低约1–2 秒，批量率提升约0.5 pp，送达质量指标保持不变；

**⚠️ 局限性**

仅通过权重接口限制了决策空间，奖励信号受延迟和噪声影响，模型对市场动态变动敏感，需持续监测分布漂移及多代理交互，未来需扩展更高维决策层和可解释工具。

---

## 551. Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models

**arXiv ID:** 2606.13603 | [PDF](https://arxiv.org/pdf/2606.13603v1)

**作者:** Daniel Scalena `[一作]` (University of Groningen), Gabriele Sarti `[通讯]` (Northeastern University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在链式思考（CoT）推理过程中各步的因果贡献，发现模型在某一单步即完成答案承诺并随后出现无实质影响的“表演性”推理；

**💡 创新点**

提出了基于早停的因果归因框架、定义了承诺边界（commitment boundary）并证明其可通过轻量级注意力探针在单前向推断中识别；

**🔧 技术方法**

采用早停截断、最大似然答案采样、线性注意力探针、以及数值扰动实验来评估推理步骤的因果作用；

**📊 数据集**

在四大推理基准上进行实验，分别为MATH‑500、AIME 2025、ZebraLogic与GPQA‑Diamond；

**📈 对比分析**

与固定比例截断、无CoT及完整CoT基线对比，探针驱动的早停在保持90%以上完整CoT准确率的同时平均节省约35%推理长度；

**⚠️ 局限性**

限制主要包括使用贪婪解码与答案强制后缀、仅分析句子级推理、过滤不合格轨迹、未覆盖代理/工具使用任务、探针训练依赖同一归因方法、以及早停在高风险场景下可能产生不容忍的误判。

---

## 552. A Three-Layer Framework for AI in Scientific Discovery

**arXiv ID:** 2606.13566 | [PDF](https://arxiv.org/pdf/2606.13566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 553. EpiBench: Verifiable Evaluation of AI Agents on Epigenomics Analysis

**arXiv ID:** 2606.13602 | [PDF](https://arxiv.org/pdf/2606.13602v1)

**作者:** Harihara Muralidharan `[一作]` (LatchBio), Kenny Workman `[通讯]` (LatchBio)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并发布了EpiBench，一个针对短周期表观遗传学分析的可验证基准；通过模拟真实工作流状态，评估代理在质量控制、峰值调用、染色质状态分析、基因组注释、差异甲基化和下游定量分析等八大任务中的决策和结果；

**💡 创新点**

在传统生物信息学基准缺乏针对表观遗传学的可验证任务的情况下，提出了以确定性分数为核心的评估框架，构造了106个可自动评分的评估，既覆盖多种测序技术，又能隔离关键科学决策点；

**🔧 技术方法**

使用大型语言模型（如GPT‑5.5、Claude Opus、Gemini 等）配合终端编码环境（Pi、OpenAI Codex 等）进行自动化推理与工具调用；通过手工审计和组件级评分诊断失败模式；采用学生t分布计算置信区间；

**📊 数据集**

四类测序数据：CUT&Tag/CUT&RUN（斑马鱼染色质工作流），ATAC‑seq（B‑ALL 小儿样本 GSE161501），ChIP‑seq（B‑ALL H3K27ac GSE211631），DNA 甲基化（ESCC WGBS/RNA GSE149608/609）；

**📈 对比分析**

对16个模型‑环境组合在106个评估上进行三次尝试，计算端点通过率；最优组合 GPT‑5.5 / Pi 通过率为45.0%（95% CI 36.3–53.7%），其他组合落在30–35%；所有系统均未达到50%通过率，表明现有代理在表观遗传学本地推理上仍有较大提升空间；

**⚠️ 局限性**

评估任务分布不均（CUT&Tag/CUT&RUN 47例，甲基化 25例，ATAC‑seq 24例，ChIP‑seq 10例），可能导致结果偏倚；仅验证确定性答案，未覆盖所有合理分析路径；多次失败表现可能来自重复测试同一决策点；benchmark 仅捕捉短周期决策，无法评估更复杂的完整工作流。

---

## 554. Testing Bipartiteness in Logarithmic Rounds

**arXiv ID:** 2606.13583 | [PDF](https://arxiv.org/pdf/2606.13583v1)

**作者:** Yumou Fei `[一作]` (Massachusetts Institute of Technology), Ronitt Rubinfeld `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文改进了 Goldreich–Ron 的随机游走测试器，证明在无孤立顶点的无向图上，仅需 O(√n) 次长度为 O(log n) 的随机游走即可测试二分图性。

**💡 创新点**

创新点在于用 Goemans–Williamson 的 Max‑Cut SDP 松弛代替传统的图分解方法，消除了多项式对数损失，得到与已知下界相匹配的最优查询复杂度。

**🔧 技术方法**

主要技术包括：随机游走分析、L^1 与相对熵的混合性质、SDP 约束的正半定性构造、以及对偶性和熵收缩的运用。

**📊 数据集**

由于本工作是理论算法研究，不涉及具体实验数据集；所有结果均在抽象图模型（邻接列表）下证明。

**📈 对比分析**

与 Goldreich–Ron 以及后续工作相比，本算法在查询复杂度上达到 O(√n)，仅比现有下界多一个常数因子；在多通道流式模型中实现了 O(log n) 次通行、O(√n log n) 空间的测试器，通行次数上实现了已知下界的最优。

**⚠️ 局限性**

局限性包括：仅适用于无孤立顶点的简单图；仍未解决容忍测试（tolerant testing）以及非二分图属性的随机游走测试；对于非均匀度数或有向图的推广尚未完成。

---

## 555. The Tone of Awareness: Topic, Sentiment, and Toxicity Maps During Mental Health Month on TikTok

**arXiv ID:** 2606.13581 | [PDF](https://arxiv.org/pdf/2606.13581v1)

**作者:** Henrique Ferraz de Arruda `[一作]` (University of Zaragoza), Filipi Nascimento Silva `[通讯]` (Indiana University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析了2023年和2024年精神健康意识月期间TikTok上的视频与评论，提取主题、情感与毒性指标；

**💡 创新点**

通过主题聚焦的情感与毒性对比，首次揭示视频与评论在情绪与毒性上的系统差异；

**🔧 技术方法**

使用BERTopic（句子嵌入+UMAP+HDBSCAN）、XLM-T情感分类器、Detoxify毒性检测器以及log-odds关键词选择；

**📊 数据集**

收集了28,341条视频和80,130条评论（包含相关精神健康关键词的内容），并对文本进行预处理与多语言检测；

**📈 对比分析**

对每个主题分别计算视频和评论的情感极性与毒性分布，利用Mann-Whitney U检验并做Benjamini-Hochberg FDR校正，发现多数主题视频情感偏负面但评论更正面或混合，毒性在评论中呈长尾；

**⚠️ 局限性**

主要限制包括TikTok Research API采样与元数据缺失、评论数量上限、文本长度差异导致的分类误差，以及对非英语评论的解释谨慎性。

---

## 556. NetCause: Counterfactual Learning for Root Cause Analysis in Large-Scale Networks

**arXiv ID:** 2606.13543 | [PDF](https://arxiv.org/pdf/2606.13543v1)

**作者:** Fabien Chraim `[一作]` (Amazon Web Services), John Evans `[通讯]` (Amazon Web Services)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于自监督学习和反事实推理的根因分析框架，将网络故障事件建模为图‑时序过程，并通过学习的生成式时序模型对候选根因进行反事实模拟，从而得到可解释的根因排名；

**💡 创新点**

创新点在于：①将根因分析视作图‑时序因果推断问题；②构建无标签的生成式时序模型（R‑GCN+RNN）学习故障传播动力学；③引入Total Causal Influence (TCI) 指标，利用反事实模拟衡量候选根因的因果影响；

**🔧 技术方法**

使用技术包括：图神经网络（Relational GCN）用于空间特征编码；循环神经网络（GRU/RNN）用于时间序列编码；自监督教师强迫训练；正样本加权的二元交叉熵损失；反事实模拟与TCI 计算；

**📊 数据集**

训练数据为 1500 起来自某大型云服务商的生产网络的事故子图，评估数据为 31 起专家标注的事故，包含约 67 条候选故障/动作转移与 5.4 条真根因；

**📈 对比分析**

与三种基线对比：时间接近度、空间接近度、规则基启发式；在 31 起评测集上，本文方法 top‑1 准确率 35.5%，比规则启发式高 16.1%（绝对），Hits@2、Hits@3 等指标也均优于基线；在大样本时表现趋于基线，但在需要快速决策的 1–2 个候选的场景下优势明显；

**⚠️ 局限性**

限制包括：评测集样本量小（31 起）导致统计波动；假设观测动态无混杂，实际中可能存在未观测因素；输入信号来自阈值化的监控系统，噪声大且缺失；模型仅处理 0→1 的状态转移，未考虑恢复（1→0）；TCI 取决于模型拟合质量与时间加权函数的选取。

---

## 557. Graphical Causal Reasoning for Root Cause Analysis in Cloud Networks

**arXiv ID:** 2606.13532 | [PDF](https://arxiv.org/pdf/2606.13532v1)

**作者:** Fabien Chraim `[一作]` (Amazon Web Services), John Evans `[通讯]` (Amazon Web Services)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于图形因果推断的根因分析框架，用于大规模云网络故障诊断；

**💡 创新点**

创新点在于结合时空聚类策略与网络自动化本体，将因果关系压缩到可管理的层级空间，并利用时间延迟条件概率进行可解释的概率推理；

**🔧 技术方法**

采用双变量Granger因果检验与三元组条件独立性测试构建因果图，随后使用时间滞后条件概率的路径似然计算进行根因推断；

**📊 数据集**

使用了某大型云服务商从1月到6月收集的约6个月事故数据，共计25,474起事件，76,595对变量组合，35起标注事故用于盲测；

**📈 对比分析**

与专家手工规则和多种基于时间/空间的基线比较，模型在Recall@3上达到85.7%，精确匹配率74.3%，相较于规则方法的62.8%和48.6%分别提升了约23%和25%；

**⚠️ 局限性**

局限在于空间建模的层级抽象导致信息压缩，难以区分同设备与相邻设备的故障，且对路径级和终端级信号支持不足；

---

## 558. Probabilistic, Resource-Aware, Asynchronous, Out-of-Order Choreographies

**arXiv ID:** 2606.13520 | [PDF](https://arxiv.org/pdf/2606.13520v1)

**作者:** Mako Bates `[一作]` (University of Vermont), Joseph P. Near `[通讯]` (University of Vermont)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种概率、时间感知的数据流语言，用于建模异步协作式分布式系统的执行。

**💡 创新点**

创新点在于将未来式异步执行映射为时序贝叶斯网络，既捕捉值又捕捉时间分布，弥补了 Ozone 等模型的非确定性。

**🔧 技术方法**

使用的技术包括概率函数定义的 effect、时序贝叶斯网络语义、可实现的未来式执行模型以及对 Ozone 子集的翻译。

**📊 数据集**

通过实验采用了模拟的通信延迟、丢包概率以及预设的两阶段提交、限Retry 协议等案例，没有使用公开数据集。

**📈 对比分析**

与 Ozone 基准比较，AsInst 的预测曲线与最低延迟匹配，显示出优于顺序实现的性能；在可靠与不可靠通道下展示了完成时间分布。

**⚠️ 局限性**

局限性包括无法处理循环/递归、对复杂网络模型依赖手工定义 effect、模型未覆盖系统级运行时开销。

---

## 559. Measurement-Calibrated Multi-Camera Fusion for Vision-Based Indoor Localization

**arXiv ID:** 2606.13509 | [PDF](https://arxiv.org/pdf/2606.13509v1)

**作者:** Mateo Toro Diz `[一作]` (Rosenheim Technical University of Applied Sciences), Noah Klarmann `[通讯]` (Rosenheim Technical University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究并实现了基于多摄像机的室内定位系统，并引入测量校准融合方法来优化数据融合。

**💡 创新点**

通过明确量化单摄像机的误差并将其用于Kalman滤波器的测量噪声校准，显著提升轨迹平滑度。

**🔧 技术方法**

使用YOLOv8n进行目标检测、MediaPipe进行人姿估计、仿射变换实现空间投影、线性Kalman滤波器以及联合概率数据关联等技术。

**📊 数据集**

使用550×300 cm实验室场景下三摄像机采集的数据，跟踪单人沿预设直线路径运动，Ground Truth为手工标注的直线段轨迹。

**📈 对比分析**

与单摄像机基准、标准融合以及测量校准融合进行对比，RMSE/MAE 约提升 6%，而校准融合的轨迹标准差约减 50%。

**⚠️ 局限性**

仅在单目标、直线轨迹且仅使用一维轨迹 Ground Truth 的条件下实验，未覆盖多目标或更复杂运动情形。

---

## 560. EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments

**arXiv ID:** 2606.13681 | [PDF](https://arxiv.org/pdf/2606.13681v1)

**作者:** Jundong Xu `[一作]` (National University of Singapore), Zhiyuan Hu `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EvoArena评测持续演变环境的LLM代理，和EvoMem增量式记忆范式以保留演化历史。

**💡 创新点**

将环境演化建模为版本链，并设计patch-based记忆追踪，可在终端、软件、偏好等多领域提升推理。

**🔧 技术方法**

采用git-like patch记录、检索式记忆增强，并在终端、软件工程与对话代理中集成EvoMem。

**📊 数据集**

构建了EvoArena子集Terminal-Bench-Evo、SWE-Chain-Evo、PersonaMem-Evo，并对比了GAIA、LoCoMo等标准基准。

**📈 对比分析**

在step/chain准确率上与多模态大模型比较，EvoMem平均提升约2.4% step、6.1% chain，标准基准提升约6%，显示显著性能提升。

**⚠️ 局限性**

仍存在长期推理复杂度与token成本权衡问题，对极端演化场景或极难任务的改进有限。

---

## 561. Specifying Hardware Communication as Programs

**arXiv ID:** 2606.13659 | [PDF](https://arxiv.org/pdf/2606.13659v1)

**作者:** Ernest Ng `[一作]` (Cornell University), Kevin Laeufer `[通讯]` (Cornell University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种领域专用语言（DSL），能够同时用于驱动 RTL 设计和从波形中推断高层事务，极大减少了编写独立驱动和监视器的工作量。

**💡 创新点**

创新点在于：①统一了驱动与监视器的实现；②通过 DSL 直接描述时序通信协议；③使用动态符号执行（DSE）技术在不依赖 SMT 求解器的前提下完成事务推断。

**🔧 技术方法**

关键技术包括：DSL 语法与语义设计、解释器实现、对组合依赖与冲突分配的静态分析、基于等值约束的符号执行与路径裁剪。

**📊 数据集**

实验使用了 Antmicro 的 Wishbone SRAM 波形与 RTL、AXI‑Stream FIFO 的 Bug 数据集（包含错误与修复版本），以及部分真实硬件协议的手写协议定义。

**📈 对比分析**

方法通过“驱动‑波形‑监测”回路验证：对给定事务序列生成波形，再用监测器恢复同一事务序列，证明准确性；相较传统手写监视器，报告不需要外部求解器，执行更快。

**⚠️ 局限性**

局限性：当前 DSL 仅覆盖单读单写 Wishbone、简单 ready‑valid 以及少数 AXI‑Stream 操作；对 burst 模式、跨通道依赖的复杂协议尚未支持；同时仍需手动编写协议脚本。

---

## 562. Operadic consistency: a label-free signal for compositional reasoning failures in LLMs

**arXiv ID:** 2606.13649 | [PDF](https://arxiv.org/pdf/2606.13649v1)

**作者:** Nathaniel Bottman `[一作]` (Incubilate), Kyle Richardson `[通讯]` (Allen Institute for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一种新的推理失败检测信号——运算范畴一致性（OC），通过比较大型语言模型（LLM）对复杂问题的直接答案与基于手工或模型自身分解后逐步推理得到的答案是否一致，来评估模型的推理可靠性。

**💡 创新点**

创新点在于将运算范畴（operad）的形式化结构引入到LLM推理评估中，将问题分解视为操作符的合成，构造了一个“结构一致性”检查；该检查不依赖标注标签，能在推理阶段即刻给出置信度；实验表明，OC在多数据集上与准确率的相关性远高于传统自一致性、语义熵和 P(True) 等方法。

**🔧 技术方法**

主要技术包括：运算范畴与代数理论、链式推理（CoT）分解、直接答案与分解答案的语义相似度计算（基于 F1、值相等等评判标准）、自一致性、语义熵、P(True) 的实现与比较、逻辑回归与 AUARC/AUROC 选择性预测评估、以及对思考型模型的子问题提取与评估。

**📊 数据集**

使用的基准数据集包括 HotpotQA、MuSiQue、StrategyQA、DROP（四个多跳问答数据集）以及 GSM8K（数学问答），对12个指令调优的非思考型LLM（4B–671B 参数）和5个思考型LLM（如 DeepSeek-R1、GLM-5.1 等）进行了实验。

**📈 对比分析**

实验通过与传统自一致性（CoT‑SC）、语义熵、P(True) 以及两种分解感知基线（分解自一致性、Skywork‑PRM）进行对比。OC 在所有四个数据集上均保持 Pearson r ≥ 0.85，且在等成本（3 次模型调用）下，OC+CoT‑SC 的 AUARC、AUROC 提升约 0.09–0.16，所有提升均在 95% 置信区间内显著；相较于仅靠采样多样性的指标，OC 提供了互补信息。

**⚠️ 局限性**

局限性包括：目前仅在深度为 2 的链式分解（即单层子问题）上评估，未覆盖更深或分支的树状结构；子问题提取依赖外部提取器，存在提取失败；评估采用数据集特定的相似度度量，可能与其他标准不完全一致；实验仅覆盖到 671B 参数模型，未包含更大规模商用模型；思考型模型样本有限，统计显著性受限；以及对流程奖励模型（PRM）的对比仅使用了离线训练的 Skywork‑PRM，缺乏同领域专门训练的 PRM。

---

## 563. Surflo: Consistent 3D Surface Flow Model with Global State

**arXiv ID:** 2606.13644 | [PDF](https://arxiv.org/pdf/2606.13644v1)

**作者:** Antoine Guédon `[一作]` (École polytechnique), Angjoo Kanazawa `[通讯]` (UC Berkeley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于流匹配的前向模型，能够把任意数量的未姿态RGB视图压缩成一个固定大小的全局潜在表示，并从该潜在中任意采样数量的有向表面点，最终生成高质量网格。

**💡 创新点**

创新点在于（1）使用Perceiver将多视图特征压缩为固定尺寸的全局潜在，消除视图冗余；（2）通过流匹配独立解码器生成可任意分辨率的表面点；（3）在ODE求解时加入渲染引导，使点云保持一致并对齐原始图像。

**🔧 技术方法**

主要技术包括冻结的VGGT Backbone提取几何特征、Perceiver压缩、3D Fourier位置编码、Transformer流匹配解码器、基于渲染的指导损失以及ODE积分。

**📊 数据集**

训练使用~10.5K场景的DL3DV数据集，并通过Gaussian Wrapping生成网格和有向点云作为监督；在多个公共基准（ML-Hypersim、BlendedMVS、DTU、SCRREAM、Tanks & Temples、Mip-NeRF 360、DeepBlending）进行评测。

**📈 对比分析**

与传统的单视图点图、固定尺寸潜在方法（NOVA3R）以及基于优化的后处理（2DGS、Gaussian Wrapping）相比，该方法在Chamfer距离和F1分数上均取得最优或接近最优结果，并且在推理速度上比优化方法快两位数，且支持任意分辨率输出。

**⚠️ 局限性**

局限性包括依赖VGGT特征，对极少视图或极端视角时表现不佳；渲染引导增加推理开销；使用Gaussian Wrapping生成的表面存在透明或纹理缺失结构的不准确；当前仅重建几何，未考虑视角相关的外观。

---

## 564. Recursive Agent Harnesses

**arXiv ID:** 2606.13643 | [PDF](https://arxiv.org/pdf/2606.13643v1)

**作者:** Elias Lumer `[一作]` (PricewaterhouseCoopers), Vamse Kumar Subbiah `[通讯]` (PricewaterhouseCoopers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一种递归代理机具（Recursive Agent Harness, RAH），通过在每个子任务上使用完整的代理机具（工具、文件系统、代码执行、规划）实现长上下文推理，并对其性能进行评估。

**💡 创新点**

创新点在于将递归单元从无工具的模型调用提升为完整的代理机具，利用代码生成脚本并行生成子代理，支持条目级推理；同时提供受控评估，证明提升来自机具而非模型。

**🔧 技术方法**

使用了代理机具架构（LangChain）、代码执行路径（生成 Python 脚本并行执行子代理）、JSON 工具调用路径、递归深度限制、文件系统隔离、规划-工具-反思循环等技术。

**📊 数据集**

使用 Oolong‑Synthetic 长上下文推理基准（199 个样本，覆盖 13 个上下文长度桶）。

**📈 对比分析**

在 GPT‑5 backbone 上与全上下文基线、递归语言模型（RLM）以及 Codex coding‑agent 基线进行对比；RAH 在 GPT‑5 上将 Oolong 分数从 71.75% 提升至 81.36%，在 Claude Sonnet 4.5 上达到 89.77%；在所有上下文长度桶均优于基线。

**⚠️ 局限性**

局限性包括仅在 Oolong‑Synthetic 上评估，未验证对 Oolong‑Real 或其他领域；父代理偶尔跳过脚本生成导致失败；NUMERIC 问题受 0.75^|y‑ŷ| 分数函数惩罚；DATE 样本量少导致高方差；未对递归深度、条目数、代码执行 vs 工具调用等做细粒度 ablation；成本与延迟的完整量化尚未完成。

---

## 565. Improving Robotic Generalist Policies via Flow Reversal Steering

**arXiv ID:** 2606.13675 | [PDF](https://arxiv.org/pdf/2606.13675v1)

**作者:** Andy Tang `[一作]` (Stanford University), Sergey Levine `[通讯]` (UC Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

通过流反转将粗糙的语义指导（如人类或VLM给出的方向性动作）映射到通用流匹配策略的噪声空间，从而得到细粒度、在分布内的动作并显著提升零样本和后续学习性能。

**💡 创新点**

创新点在于：①利用流的可逆性将参考动作逆向映射到噪声，直接把语义推理转化为动作；②结合DSBC和DSRL+两种新范式，实现在极少数据下的高效BC与RL加速；③将语义引导与通用策略的“合理行为”优先级无缝对齐。

**🔧 技术方法**

核心技术包括流匹配（Flow Matching）策略、流反向积分（Flow Reversal）、Diffusion Steering via Behavioral Cloning (DSBC)、强化学习框架DSRL+、以及VLM（如Gemini-ER-1.6）或人类给出的语义参考。

**📊 数据集**

实验使用LIBERO（模拟）和DROID（真实机器人）数据集，并以OpenPi的π_0.5-LIBERO和π_0.5-DROID为基础通用策略。

**📈 对比分析**

与未引导基准、直接执行VLM动作、部分加噪和样本排序等方法比较，FRS在零样本上提升多达10%绝对成功率；DSBC在仅10-18条轨迹下就能超越标准BC；DSRL+显著加速RL收敛并提升最终成功率，真实机器人上平均提升60%。

**⚠️ 局限性**

局限性包括：需依赖外部语义推理源（人类或VLM）；流反向近似可能导致细节失真；对流匹配模型的依赖，若基础策略已近乎无效则即使流反转也难以显著改进；在高度随机化或极端稀缺数据场景下，过拟合与样本不足仍是挑战。

---

## 566. InterleaveThinker: Reinforcing Agentic Interleaved Generation

**arXiv ID:** 2606.13679 | [PDF](https://arxiv.org/pdf/2606.13679v1)

**作者:** Dian Zheng `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为任何固定图像生成器实现了可插入文本与图像的多步生成能力，构建了Planner‑Critic多智能体闭环流水线；

**💡 创新点**

通过将规划与评估解耦、引入单步RL双奖励机制以及细粒度数据过滤，解决了传统UMM的视觉过度依赖和误差累积问题；

**🔧 技术方法**

利用多智能体框架、GRPO强化学习、精细的评估与奖励设计，以及对生成器的Prompt细化；

**📊 数据集**

构建了三大高质量数据集（Interleave‑Planner‑SFT‑80k、Interleave‑Critic‑SFT‑112k、Interleave‑Critic‑RL‑13k），并使用Gemini 2.5 Pro与Nano Banana Pro生成示例；

**📈 对比分析**

在UEval、CoMM、WISE和RISE等基准上与多种开源/专有模型对比，InterleaveThinker在图像质量、文本一致性和推理能力上均超越或匹配最强的专有模型；

**⚠️ 局限性**

受限于底层生成器的先验知识，无法生成训练数据之外的概念，且对长序列的RL仍存在计算与信用分配挑战。

---

## 567. EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery

**arXiv ID:** 2606.13662 | [PDF](https://arxiv.org/pdf/2606.13662v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 568. Before You Think: System 0, AI-Mediated Cognition and Cognitive Colonization

**arXiv ID:** 2606.13658 | [PDF](https://arxiv.org/pdf/2606.13658v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 569. RepWAM: World Action Modeling with Representation Visual-Action Tokenizers

**arXiv ID:** 2606.13674 | [PDF](https://arxiv.org/pdf/2606.13674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 570. SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning

**arXiv ID:** 2606.13673 | [PDF](https://arxiv.org/pdf/2606.13673v1)

**作者:** Seokju Cho `[一作]` (KAIST), Min-Hung Chen `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于代码的空间推理代理，利用持久化 Python 内核实现可迭代、可复用的感知与几何计算。

**💡 创新点**

将代码作为动作接口，突破单次执行与结构化工具调用的局限，使模型能够根据中间结果动态组合、检查并修正推理步骤。

**🔧 技术方法**

使用 VLM（如 Qwen、Gemma）驱动的代码生成、预加载的感知工具（SAM、Depth Anything 等）、NumPy/Scipy 等科学库，以及多步循环控制的框架。

**📊 数据集**

在 20 个空间推理基准上评估，包括单图、多视角、通用空间、视频/4D 空间以及通用视频理解任务。

**📈 对比分析**

与无工具、单次代码执行、结构化工具调用等基线对比，平均准确率达到 59.9%，比最新空间代理高约 11.2 个百分点，尤其在多视角与视频推理上表现最为显著。

**⚠️ 局限性**

依赖现有感知工具与 VLM 的零样本能力，若感知质量不足或任务超出工具覆盖范围，仍可能受限；

---

## 571. $\texttt{WEAVER}$, Better, Faster, Longer: An Effective World Model for Robotic Manipulation

**arXiv ID:** 2606.13672 | [PDF](https://arxiv.org/pdf/2606.13672v1)

**作者:** Arnav Kumar Jain `[一作]` (Mila - Quebec AI Institute), Andrea Bajcsy `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种多视角世界模型（WEAVER），能够在机器人操作任务中同时满足高保真度、长期一致性和高效生成，支持离线评估、策略改进和实时规划。

**💡 创新点**

创新点：1）融合多视角、记忆与短期历史的条件策略；2）采用流匹配损失与扩散强制，实现高效且长时段一致的隐空间预测；3）在隐空间直接预测奖励和价值，避免外部判别器；4）通过 KV 缓存、可逆扩散和后训练加速推理，使规划速度提升 5‑10 倍。

**🔧 技术方法**

核心技术包括：预训练的 Stable Diffusion VAE 编码器、2D Transformer 动力学模块、流匹配与扩散强制训练、KV 缓存、可逆扩散加速、AdaPool+MLP 的奖励与 critic 头、以及离线数据增强与优势过滤。

**📊 数据集**

数据集：先在公开的 DROID 数据集上预训练，然后在实际 Franka Panda 机器人上收集 5 个操作任务（堆叠碗、袋子、标记、毛巾、倒豆子）的 50 条真实轨迹进行微调，另外 20 条用于评估。

**📈 对比分析**

与 Ctrl‑World 的对比：在 FID/FVD、推理时间（NFE）上实现 Pareto‑domination；政策评估的皮尔逊相关系数提升至 0.87；策略改进在无真实交互情况下将成功率提升 38%；实时规划在保持 5‑10 倍速度的同时成功率提升 14%（最高 20%）。

**⚠️ 局限性**

局限性：1）仅使用视觉观察，缺少触觉信息导致部分情况不确定；2）物理先验缺失，对柔性物体任务仍易出错；3）推理延迟仍限制规划在单一步骤内的短期推理；4）奖励监督噪声可能导致错误的优势估计。

---

## 572. Agents-K1: Towards Agent-native Knowledge Orchestration

**arXiv ID:** 2606.13669 | [PDF](https://arxiv.org/pdf/2606.13669v1)

**作者:** Zongsheng Cao `[一作]` (Shanghai Artificial Intelligence Laboratory), Lei Bai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一套端到端的科研知识组织管线Agents‑K1，将原始科研论文（包括文字、图表、公式等多模态内容）解析成结构化的、可被大型语言模型直接使用的知识图谱，并通过图形化命令行接口（GraphAnything CLI）实现多源检索与推理。

**💡 创新点**

①在知识组织上实现了多模态统一抽象与语义锚点，既保留了论文完整结构，又支持实体、主张、机制、方法演化等丰富的语义关系；②通过强化学习（GRPO）和规则奖励训练了紧凑的4B模型，在实体识别、关系抽取和长文本结构化抽取上超越同规模甚至更大模型；③引入三源检索（Web搜索、图检索、跨文档遍历）与可组合的图操作，使研究代理能够可追溯地获取证据并执行多跳推理。

**🔧 技术方法**

核心技术包括：多模态解析器（MinerU+五模块Schema）、基于GRPO的4B信息抽取模型、三源检索与融合策略、图形化CLI与可扩展的图操作集合、以及可调度的多角色 Swarm 运行框架。

**📊 数据集**

使用了约246万篇科研论文（涵盖计算机科学、化学、生物、地球科学、物理、材料科学等六大领域）构建 Scholar‑KG；公开发布了100万篇子集，并在 HuggingFace 上提供数据集。

**📈 对比分析**

在实体识别、关系抽取、长文本抽取等十项基准上，Agents‑K1 的4B模型分别优于8B公开模型，且在多跳问答（HotpotQA、2WikiMultiHopQA、MuSiQue）及 FrontierScience‑Research 基准上显著提升 LLM（Gemini‑3、GPT‑5.2）整体与推理准确率（提升至24.6%/39.4%及52.3%→69.5%）。

**⚠️ 局限性**

限制方面：①模型仍依赖于规则奖励和预定义schema，可能在新领域或非标准格式文献上出现抽取误差；②多模态解析对 OCR 与图像质量敏感；③三源检索需要额外资源与维护，且 Web 搜索结果仍需人工校验；④目前未对代码与实验可复现性进行完整覆盖，仅提供烟雾测试。

---

## 573. World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible

**arXiv ID:** 2606.13652 | [PDF](https://arxiv.org/pdf/2606.13652v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 574. The Stable Recovery Manifold: Geometric Principles Governing Recoverability in Continual Learning

**arXiv ID:** 2606.13637 | [PDF](https://arxiv.org/pdf/2606.13637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 575. Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution

**arXiv ID:** 2606.13668 | [PDF](https://arxiv.org/pdf/2606.13668v1)

**作者:** Dimitri Kachler `[一作]` (Inria), Pascal Denis `[通讯]` (Inria)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过训练轻量化编码器蒸馏 LoRA 梯度影响排名，实现大规模数据属性化的快速推理。

**💡 创新点**

创新点在于使用 CountSketch 投影与软max KL 损失蒸馏梯度影响，显著提升推理速度和资源效率。

**🔧 技术方法**

技术包括 LoRA 梯度计算、CountSketch 投影、余弦相似度影响、Pearson 相关+KL 损失的排名蒸馏、轻量编码器（ettin-encoder 150m/68m）等。

**📊 数据集**

使用 Dolly、BBH、SmolLM2‑1.7B 目标模型、Pythia‑1B（毒性过滤）、UltraChat 8K（XSTest）等数据集。

**📈 对比分析**

与 LESS、LoGRA、RDS+、TF‑IDF、DataInf 等基线比较，Influcoder 在推理时间上比 LESS 高达 15–100 倍、FLOPs 更低，Spearman 相关与 LESS 接近或优于，毒性过滤 AUPRC 亦与 LESS 相近或略优。

**⚠️ 局限性**

限制：仅在训练时使用的池/查询分布内有效，分布外性能急剧下降；需要足够的训练数据与前期设置，需训练编码器，不能即插即用。

---

## 576. The Moving Drone: Negotiating Agency Between the Voice and the Virtual

**arXiv ID:** 2606.13640 | [PDF](https://arxiv.org/pdf/2606.13640v1)

**作者:** Nithya Shikarpur `[一作]` (Massachusettes Institute of Technology), Anna Huang `[通讯]` (Massachusettes Institute of Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

通过四个实时循环器将歌唱即兴转化为虚拟 Drone，赋予其在音高和音色维度上从被动到主动的演化，并结合低保真生成式 AI 对音色进行重新合成，形成一种声乐与虚拟 Drone 的共创体验。

**💡 创新点**

创新点在于：① 将传统四弦 Drone 数字化为可主动变调、可音色变化的虚拟 Drone；② 在 Agency 维度上构建音高与音色的主动-被动二轴框架；③ 采用低保真 AI 生成音色，强调人机协作而非追求高保真；④ 将这些技术嵌入三段式即兴演出中，形成可调节的演化路径。

**🔧 技术方法**

使用技术包括：Max/MSP 实时循环器与跨淡控制、预设音高移位、GaMaDHaNi 低分辨率声谱生成模型、Griffin‑Lim 相位估计、GPU 实时推理、Dante 虚拟声卡与多声道输出。

**📊 数据集**

使用的数据集是 120 小时的 Hindustani 语音开放源数据，用于训练 GaMaDHaNi 的层次生成模型。

**📈 对比分析**

本工作没有传统的量化对比，而是通过演出表现与艺术反思评估；在 13 分钟的现场演出中，系统实时性足够且低保真 AI 生成的音色增强了人机共创的紧张与混沌感，满足了艺术目标。

**⚠️ 局限性**

限制包括：① 低保真生成导致噪声与失真；② 对实时延迟与 GPU 资源需求有限制；③ 仅在 Hindustani 音乐框架内验证，跨文化推广受限；④ Agency 框架仍为工作中草案，缺乏系统理论与定量评估。

---

## 577. Tuning Agent-Based Predator-Prey Models Toward Lotka-Volterra Dynamics

**arXiv ID:** 2606.13639 | [PDF](https://arxiv.org/pdf/2606.13639v1)

**作者:** Corinna Mandl `[一作]` (Radboud University), Marcel van Gerven `[通讯]` (Radboud University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文设计并调参了一个基于连续空间的猎物-捕食者代理模型，使其种群轨迹呈现洛特卡-沃尔特拉循环特征。

**💡 创新点**

创新之处在于将宏观洛特卡-沃尔特拉动力学作为顶层优化目标，通过手工设计的多项损失函数直接调节微观参数，验证了宏观目标可指导微观模型行为。

**🔧 技术方法**

主要技术包括基于JAX的ABMax代理框架、连续时间递归神经网络（CTRNN）控制器、协方差矩阵适应进化策略（CMA‑ES）以及自定义的相关性与振荡损失函数。

**📊 数据集**

实验使用纯仿真数据，未采用外部真实生态数据集；所有参数在GPU上多实例并行跑步。

**📈 对比分析**

在随机与演化控制器两种设置下比较，损失下降更快且收敛到更低的值；演化控制器得到的种群振荡更不规则但更不越界，表明宏观目标能有效约束系统并产生可持续的种群循环。

**⚠️ 局限性**

主要局限在于损失函数人为手工设计，难以直接匹配真实轨迹；且未考虑资源枯竭、个体适应等更复杂生态机制，可能限制模型在更真实情境下的泛化。

---

## 578. Operads for compositional reasoning in LLMs

**arXiv ID:** 2606.13634 | [PDF](https://arxiv.org/pdf/2606.13634v1)

**作者:** Nathaniel Bottman `[一作]` (Incubilate), Kyle Richardson `[通讯]` (Allen Institute for Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出用代数结构运算子（operads）来正式化大语言模型中的问题分解，并引入运算子一致性（operadic consistency）评估模型多步推理可靠性。

**💡 创新点**

创新点在于将运算子理论与LLM问题分解结合，给出新的一致性指标并证明其与准确率高度相关。

**🔧 技术方法**

使用运算子框架、代数化模型（QA模型作为运算子代数）、以及自洽一致性测度。

**📊 数据集**

在十二个指令微调的LLM上，使用四个多跳问答数据集进行实验。

**📈 对比分析**

相较于基于温度的自洽一致性，运算子一致性在保持相同推理成本的情况下实现了更高的选择性预测性能，并与准确率高度相关。

**⚠️ 局限性**

局限在于目前仅针对问题分解的表层结构，未探索更深层推理链条的完整一致性以及对更广泛任务的泛化能力。

---

## 579. Beyond Virtual Delay: Improving Packet Delay Bound in Network Calculus

**arXiv ID:** 2606.13631 | [PDF](https://arxiv.org/pdf/2606.13631v1)

**作者:** Yuming Jiang `[一作]` `[通讯]` (Norwegian University of Science and Technology), Yuming Jiang (Norwegian University of Science and Technology)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究网络计算中包延迟分析，提出并验证了一种不依赖虚拟延迟的新包延迟上界。

**💡 创新点**

创新点在于仅利用到达曲线和服务曲线直接推导包延迟上界，避免了传统虚拟延迟方法的保守性，并证明在漏桶到达与速率-延迟服务模型下该上界严格优于经典上界。

**🔧 技术方法**

使用了网络计算（Network Calculus）中的极限逆函数、最小卷积等理论工具，对FIFO系统进行包层面分析。

**📊 数据集**

实验数据集为时间敏感网络（TSN）中的 Talker 设备配置，包含两条 TSN 流（Stream J 与 Stream K）的帧尺寸、间隔以及非 TSN 帧干扰参数。

**📈 对比分析**

通过与经典虚拟延迟上界、以及利用流级信息的已知上界进行数值对比，案例表明新上界比经典上界降低约 25% 以上，且在使用流信息时可进一步提升。

**⚠️ 局限性**

局限性包括：仅适用于 FIFO 结构；依赖于已知的到达/服务曲线；对非抽象模型或需要额外上下文信息的情形（如自适应调度、优先级变动）可能不适用。

---

## 580. Automated reproducibility assessments in the social and behavioral sciences using large language models

**arXiv ID:** 2606.13670 | [PDF](https://arxiv.org/pdf/2606.13670v1)

**作者:** Tobias Holtdirk `[一作]` (LMU Munich), Stefan Feuerriegel `[通讯]` (LMU Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大型语言模型自动化社会与行为科学研究的可重复性评估。

**💡 创新点**

提出基于LLM的代理工作流，能在不依赖原始代码的情况下重现统计主张，并与人类重分析对比。

**🔧 技术方法**

使用Claude Opus 4.7 LLM、ReAct框架、Python、bash工具以及OCR转换。

**📊 数据集**

采用Multi100项目的76篇可获得数据的研究（经济学、政治学、心理学）。

**📈 对比分析**

对比LLM生成的Cohen d与原始结果及人类重分析，LLM在41 %研究中达到±0.05容差，95.7 %匹配结论；相较人类重分析为34 %与74 %。

**⚠️ 局限性**

局限包括数据可获得性限制、模型可能的预训练泄露、代码错误、对Cohen d的统一转换假设以及未检验实验重现。

---

## 581. Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation

**arXiv ID:** 2606.13657 | [PDF](https://arxiv.org/pdf/2606.13657v1)

**作者:** Guo Yu `[一作]` (Nanjing University), Han-Jia Ye `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文分析了大语言模型和视觉语言模型在使用On‑Policy Distillation (OPD)后参数更新的稀疏性与几何特征。

**💡 创新点**

创新点在于揭示OPD产生的参数变动既保持了稀疏且偏离主方向，又具有全秩但谱能量集中，与传统稠密微调和RL训练形成对比。

**🔧 技术方法**

采用了检查点差分、坐标稀疏度、SVD谱分析、主子空间投影、子网络遮蔽实验以及AdamW/SGD优化器对比等技术。

**📊 数据集**

使用多对语言模型与视觉语言模型（如DS‑Qwen、MiniCPM5、Qwen2.5‑VL、Qwen3‑OPS等）以及相应的教师模型进行实验。

**📈 对比分析**

通过与离线蒸馏、RLVR、随机遮蔽以及不同优化器的对比，发现OPD子网络可恢复与完整训练相同的推理性能，但AdamW在OPD中比SGD表现更好。

**⚠️ 局限性**

局限性在于只评估了最终检查点的静态更新，没有追踪训练过程；实验规模受限于中小模型，且未覆盖更大模型或不同任务场景。

---

## 582. Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning

**arXiv ID:** 2606.13680 | [PDF](https://arxiv.org/pdf/2606.13680v1)

**作者:** Zilin Xiao `[一作]` (Meta Superintelligence Labs), Vicente Ordonez `[通讯]` (Rice University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Retrieval-Augmented Reinforcement Fine-Tuning (RA-RFT) 框架，通过黄金相关性蒸馏构造推理相关性标签，训练面向推理的检索器，并在强化学习过程中注入检索到的示例进行策略优化。

**💡 创新点**

创新点在于：①将检索目标从表面相似性转为推理实用性；②利用 GPT‑4o 评估器直接标注推理相关性；③将检索到的类比示例嵌入 RL 训练循环，提供更稠密的奖励信号；④闭环训练而非仅在推理时使用检索。

**🔧 技术方法**

技术细节包括：RL 从可验证奖励 (RLVR) 与 GRPO 优化、对比学习 (InfoNCE) 训练多向量晚交互检索器、Qwen3 语言模型、GPT‑4o 评估器、基于检索上下文的推理提示。

**📊 数据集**

使用的数据集有：QuestA 训练集、OpenR1-Math-220K 检索语料、AIME 2024/2025、HMMT 2025、BrUMO 2025 评测基准，黄金相关性标注由 GPT‑4o 生成；Qwen3‑235B 负责生成推理轨迹。

**📈 对比分析**

与 Base、GRPO、OPSD、QuestA 等基线对比，RA‑RFT 在 Qwen3‑1.7B 上对 AIME24 提升 4.7 点、AIME25 提升 7.1 点、HMMT25 提升 1.9 点、BrUMO25 提升 2.6 点；在 Qwen3‑4B 上在 3/4 评测基准中领先，BrUMO25 提升 5.9 点、AIME25 提升 2.8 点；整体平均准确率提升显著，学习曲线也更快。

**⚠️ 局限性**

局限性包括：需要依赖 GPT‑4o 这类强大评估器和大量计算资源；检索效果受语料库覆盖范围限制；方法主要适用于数学推理等可类比性强的任务，对其他领域的推广尚未验证；检索开销和实时性仍是待解决的问题。

---

## 583. Understanding Truncated Positional Encodings for Graph Neural Networks

**arXiv ID:** 2606.13671 | [PDF](https://arxiv.org/pdf/2606.13671v1)

**作者:** James Flora `[一作]` (Oregon State University), Amir Nayyeri `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并对比了图神经网络中三类截断位置编码（谱投影、行走矩阵、k-谐波距离）的理论表达能力与实际效果，并提出将多种截断编码混合使用以提升性能。

**💡 创新点**

创新点在于：①证明截断后谱投影与行走编码在表达力上存在根本差异；②证明截断谱编码甚至可能弱于 1-WL；③提出 k‑谐波距离作为新的谱/行走编码桥梁，并证明其在截断形式下的表达力与完整形式相当；④实验验证混合编码优于单一编码。

**🔧 技术方法**

使用 Weisfeiler–Lehman (WL) 和 GD‑WL 变体分析表达力；构造 k‑EP‑WL、k‑Walk‑WL、Resistance‑WL、k‑harmonic‑WL 等测试；采用 Graphormer‑GD Transformer 与 GINE MPNN 作为模型；利用近似 Laplacian 逆算法高效计算 k‑谐波距离。

**📊 数据集**

主要数据集：BREC（用于测量 GNN 表达力的对照学习基准）和 ZINC‑12k（分子回归任务）。

**📈 对比分析**

对比方法包括：在 BREC 上测量不同编码对 1‑WL 与 3‑WL 可区分性的准确率；在 ZINC‑12k 上评估 MAE。结果表明：单一截断编码只能在部分任务上达到 3‑WL 限界，而混合编码（如 Walk + k‑harmonic）在 MAE 上优于任何单一编码，性能提升约 10‑15% 以上。

**⚠️ 局限性**

局限性：①主要关注图同构判定的理论表达力，对实际下游任务的泛化尚未完全验证；②混合编码的选择和维度调节仍需经验性调优；③在大规模图上计算 k‑谐波距离虽然可近似，但仍有额外的时间与内存开销；④实验仅覆盖了三类编码，其他潜在编码的截断行为仍未探索。

---

## 584. HyperTool: Beyond Step-Wise Tool Calls for Tool-Augmented Agents

**arXiv ID:** 2606.13663 | [PDF](https://arxiv.org/pdf/2606.13663v1)

**作者:** Yaxin Du `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HyperTool，一种统一可执行的 MCP‑style 工具接口，将局部确定的工具链折叠为代码块，减少模型可见的工具调用次数，降低上下文膨胀；

**💡 创新点**

创新点在于通过可执行代码块改变工具执行粒度，既保留原有 MCP 语义，又实现局部工具工作流的内部化；

**🔧 技术方法**

使用可执行代码块（Python 代码）在 MCP 环境中调用原有工具，结合数据合成、轨迹回放、执行与证据校验等技术进行监督微调；

**📊 数据集**

基准数据集为 MCP‑Universe，涵盖 web search、financial analysis、location navigation、repository management 四个领域；

**📈 对比分析**

与 ReAct、CodeAct、ReCode、BrowseMaster、AgentFold 等基线对比，HyperTool 在 Qwen3‑8B 上平均准确率从 20.92% 提升至 33.33%，Qwen3‑32B 从 24.18% 提升至 35.29%，在金融分析等组合任务中表现尤为突出；

**⚠️ 局限性**

局限性：目前仅适用于局部确定的工具子流程，对需要频繁模型干预的动态工作流支持有限，且训练数据规模和多样性有待进一步扩展。

---

## 585. Mana: Dexterous Manipulation of Articulated Tools

**arXiv ID:** 2606.13677 | [PDF](https://arxiv.org/pdf/2606.13677v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 586. Aerial Wildfire Suppression Planning with a Hybrid CNN-Cellular Automata Fire Model

**arXiv ID:** 2606.13633 | [PDF](https://arxiv.org/pdf/2606.13633v1)

**作者:** Ion Matei `[一作]` (Fujitsu Research of America), Anthony Wong `[通讯]` (Fujitsu Research of America)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在预训练的混合CNN–细胞自动机火灾模型上，使用梯度下降式优化设计飞机投放的二进制决策、投放位置与方向，形成可执行的空中扑火计划；随后通过蒙特卡洛和空间相关误差扰动分别评估系统的随机性与模型误差不确定性。

**💡 创新点**

首次将学习到的空间化火势传播模型与可微分的干预优化耦合，并在同一框架内同时量化两类不确定性；同时引入持续性和即时性的水/抑火剂物理效应，提升了干预方案的现实意义。

**🔧 技术方法**

利用CNN生成场景参数、细胞自动机传播规则、梯度优化（Adam+STE）实现干预设计，采用伊索米特拉换算与高斯投影处理抑火剂投影，使用Monte Carlo采样与ILR坐标下的高斯随机场进行不确定性量化。

**📊 数据集**

案例数据来自2020年Bear Fire实际周界，配合LANDFIRE地形与燃料图、ECMWF ERA5每日气象数据以及公开火场观测周界，形成完整的训练与评估环境。

**📈 对比分析**

与无干预基线对比，最优计划在预训练模拟器下将最终受火面积平均降低约20%（随机性评估）和约85%（模型误差评估），展示了在不确定性条件下干预计划的显著益处。

**⚠️ 局限性**

局限在于结果高度依赖预训练模型的预测精度与抑火剂参数的设定，缺乏对真实现场的直接验证，且未涵盖全部操作约束（如气象极端、机场限制）和多模态资源的动态调度，故需进一步校准与实战验证。

---

## 587. Flex4DHuman: Flexible Multi-view Video Diffusion for 4D Human Reconstruction

**arXiv ID:** 2606.13655 | [PDF](https://arxiv.org/pdf/2606.13655v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 588. SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation

**arXiv ID:** 2606.13647 | [PDF](https://arxiv.org/pdf/2606.13647v1)

**作者:** Marek Šuppa `[一作]` (Comenius University), Viktória Ondrejová `[通讯]` (Cisco Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了SkMTEB斯洛伐克语言文本嵌入基准（31个数据集），并开发了通过词表修剪与微调得到的45M/365M参数可本地部署模型。

**💡 创新点**

首创斯洛伐克深度嵌入基准；将词表修剪与微调结合，实现60–62%参数压缩且性能几乎不逊于大模型；提供开源可本地部署的实用模型。

**🔧 技术方法**

使用词表修剪（Vocabulary Trimming）、多任务微调（Cosine Similarity Loss、Multiple Negatives Ranking Loss）、指令调优、Multilingual E5模型以及MTEB评估框架。

**📊 数据集**

采用skLEP（SK‑SQuAD、NLI、STS、RTE）、WebFAQ、FineWeb2‑Slovak及31个改编或新建的检索、重排序、分类、聚类、对比、STS等任务数据集。

**📈 对比分析**

在SkMTEB上对31种公开与专有模型进行评测，发现大型指令调优模型性能最佳；词表修剪+微调得到的45M模型与500M模型相近，且与专有API竞争。

**⚠️ 局限性**

基准覆盖仍有限，部分数据为翻译缺乏本土专业语料；评测受计算资源限制；词表修剪可能影响多语混合文本性能；需要持续更新模型与数据。

---

## 589. Modality Forcing for Scalable Spatial Generation

**arXiv ID:** 2606.13676 | [PDF](https://arxiv.org/pdf/2606.13676v1)

**作者:** Bardienus Pieter Duisterhof `[一作]` (Carnegie Mellon University), Keunhong Park `[通讯]` (World Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对预训练的文本到图像扩散变换器（DiT）进行后训练，使其能联合生成RGB图像和深度图，并支持图像→深度、深度→图像和联合生成。

**💡 创新点**

提出了“Modality Forcing”方法：为每种模态设置独立的噪声尺度、在像素空间对深度进行扩散、利用稀疏深度标注以及自蒸馏保持原始T2I先验，从而在单一模型中实现多任务。

**🔧 技术方法**

使用Diffusion Transformer、像素空间深度tokenizer/detokenizer、VAE编码、logit‑normal采样、flow‑matching目标、跨模态时间步嵌入和自蒸馏损失。

**📊 数据集**

训练数据包括约17M帧来自12个真实与合成数据集（室内扫描、户外驾驶、合成渲染），以及T2I预训练使用的0~1.92B图像。评估集为NYUv2、DIODE、ETH3D、ScanNet和KITTI。

**📈 对比分析**

与现有的深度估计器（MoGe‑2、Depth Anything V2、Depth Pro）、生成式深度模型（PPD）以及联合生成模型（JointDiT、UniCon）进行对比。模型在多项指标上达到或接近最先进水平；相较JointDiT的AbsRel降低57%，图像→深度、深度→图像的FID也显著更低。

**⚠️ 局限性**

限制包括：仅扩展到3B参数的模型；深度数据量远低于T2I训练数据；仍存在量化和几何不一致的失真；未覆盖度量深度、超大模型和更深架构的潜在改进。

---

